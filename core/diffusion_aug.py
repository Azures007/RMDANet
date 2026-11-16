"""Diffusion-based generative augmentation for cross-scene HSI (data-level only).

Pipeline (no change to the model or the loss):

1. Train a small class-conditional DDPM on *source centre spectra*.
2. Sample spectra per class from the diffusion model.
3. Match the generated source spectra to *target* per-band mean/std
   (target-style moment matching, uses only unlabeled target pixels).
4. Build synthetic source patches: copy a real source patch of the same class
   and replace its centre-pixel spectrum with the generated target-style one.
5. Append the synthetic patches to the source training set.

Because the synthetic samples carry their source label, the whole procedure is
target-label-free; it only changes the training data distribution.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _sinusoidal(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1))
    angles = t.float().view(-1, 1) * freq.view(1, -1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class _Denoiser(nn.Module):
    def __init__(self, bands: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(bands + 32 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, bands),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, _sinusoidal(t), self.class_embed(labels)], dim=1))


class SpectrumDiffusion(nn.Module):
    def __init__(self, bands: int, num_classes: int, steps: int = 100, hidden: int = 256) -> None:
        super().__init__()
        self.bands = bands
        self.steps = steps
        self.net = _Denoiser(bands, num_classes, hidden)
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def loss(self, spectra: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(spectra)
        t = torch.randint(0, self.steps, (spectra.size(0),), device=spectra.device)
        ab = self.alpha_bar[t].view(-1, 1)
        xt = ab.sqrt() * spectra + (1.0 - ab).sqrt() * noise
        return F.mse_loss(self.net(xt, t, labels), noise)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        x = torch.randn(labels.size(0), self.bands, device=labels.device)
        for step in reversed(range(self.steps)):
            t = torch.full((labels.size(0),), step, device=labels.device, dtype=torch.long)
            eps = self.net(x, t, labels)
            alpha = self.alpha[step]
            alpha_bar = self.alpha_bar[step]
            mean = (x - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
            if step > 0:
                x = mean + self.beta[step].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


class SyntheticPatchDataset(Dataset):
    def __init__(self, cubes: np.ndarray, labels: np.ndarray) -> None:
        self.cubes = np.asarray(cubes, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        patch = np.ascontiguousarray(self.cubes[index].transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(patch), int(self.labels[index])


def build_synthetic_dataset(
    source_image: np.ndarray,
    source_coordinates: np.ndarray,
    source_labels: np.ndarray,
    target_image: np.ndarray,
    patch_size: int,
    num_classes: int,
    device: torch.device,
    seed: int,
    per_class: int = 40,
    train_steps: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> SyntheticPatchDataset:
    rng = np.random.default_rng(seed + 777)
    torch.manual_seed(seed + 778)
    radius = patch_size // 2

    source_spectra = np.asarray(
        source_image[source_coordinates[:, 0], source_coordinates[:, 1]], dtype=np.float32)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    target_spectra = np.asarray(target_image.reshape(-1, target_image.shape[-1]), dtype=np.float32)
    if target_spectra.shape[0] > 4000:
        index = rng.choice(target_spectra.shape[0], 4000, replace=False)
        target_spectra = target_spectra[index]

    src_mean = source_spectra.mean(axis=0, keepdims=True)
    src_std = source_spectra.std(axis=0, keepdims=True).clip(min=1e-5)
    tgt_mean = target_spectra.mean(axis=0, keepdims=True)
    tgt_std = target_spectra.std(axis=0, keepdims=True).clip(min=1e-5)

    normalized = (source_spectra - src_mean) / src_std
    spectra_t = torch.from_numpy(normalized).to(device)
    labels_t = torch.from_numpy(source_labels).to(device)

    diffusion = SpectrumDiffusion(source_spectra.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    n = spectra_t.size(0)
    for _ in range(train_steps):
        index = torch.randint(0, n, (min(batch_size, n),), device=device)
        loss = diffusion.loss(spectra_t[index], labels_t[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    cubes = []
    out_labels = []
    padded = np.pad(np.asarray(source_image, dtype=np.float32),
                    ((radius, radius), (radius, radius), (0, 0)), mode="symmetric")
    for class_id in range(num_classes):
        class_index = np.where(source_labels == class_id)[0]
        if class_index.size == 0:
            continue
        labels_gen = torch.full((per_class,), class_id, device=device, dtype=torch.long)
        generated = diffusion.sample(labels_gen).cpu().numpy()
        generated = generated * src_std + src_mean
        generated = (generated - src_mean) / src_std * tgt_std + tgt_mean
        for row in generated:
            base = class_index[rng.integers(0, class_index.size)]
            r, c = source_coordinates[base]
            patch = padded[r:r + patch_size, c:c + patch_size].copy()
            patch[radius, radius] = row.astype(np.float32)
            cubes.append(patch)
            out_labels.append(class_id)

    if not cubes:
        raise RuntimeError("diffusion augmentation produced no synthetic patches")
    print(f"[diffusion-aug] generated {len(cubes)} target-style source patches "
          f"for {len(set(out_labels))} classes")
    return SyntheticPatchDataset(np.stack(cubes), np.asarray(out_labels))
