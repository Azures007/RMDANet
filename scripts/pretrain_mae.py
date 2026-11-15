"""S3: MAE self-supervised pretraining for the 7x7 HSI patch encoder.

Pretrains the Transformer token encoder (S2 backbone) on *unlabeled* source
and target patches (random spatial-token masking + spectrum reconstruction).
The saved encoder state dict can be loaded by trainer via
``--pretrained_backbone``; keys match ``TransformerTokenEncoder`` so the
UDA model is initialised from the pretrained encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from data import align_target_bands, load_source_scene, load_target_image_only  # noqa: E402
from model import TransformerTokenEncoder  # noqa: E402

PAVIA_DATA = "/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets"
DATASET_DIR = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}


class MAEEncoder(nn.Module):
    """Same sub-module names as TransformerTokenEncoder (encoder part only)."""

    def __init__(self, bands: int, dim: int = 64, depth: int = 4, heads: int = 4):
        super().__init__()
        self.bands = int(bands)
        self.dim = int(dim)
        self.local_context = nn.Conv3d(
            self.bands, self.bands, kernel_size=(3, 3, 3), padding=(1, 1, 1),
            groups=self.bands, bias=False,
        )
        self.token_proj = nn.Sequential(nn.LayerNorm(self.bands), nn.Linear(self.bands, self.dim))
        self.pos = nn.Parameter(torch.zeros(1, 49, self.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=heads, dim_feedforward=self.dim * 2,
            dropout=0.10, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, 96), nn.GELU(), nn.Linear(96, self.bands)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        batch, bands, height, width = x.shape
        context = F.silu(self.local_context(x.unsqueeze(2)), inplace=True).squeeze(2)
        x = x + context
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, bands)
        emb = self.token_proj(tokens) + self.pos[:, : height * width]
        mask_tokens = mask.unsqueeze(-1)
        emb = torch.where(mask_tokens, self.mask_token.expand_as(emb), emb)
        emb = self.blocks(emb)
        emb = self.norm(emb)
        recon = self.decoder(emb)
        return recon, tokens


class PatchSampler:
    def __init__(self, images: list[np.ndarray], patch: int = 7):
        self.images = images
        self.patch = patch
        self.coords = []
        half = patch // 2
        for idx, image in enumerate(images):
            norm = np.linalg.norm(image, axis=-1)
            valid = norm > 0
            rows, cols = np.nonzero(valid)
            keep = (
                (rows >= half) & (rows < image.shape[0] - half)
                & (cols >= half) & (cols < image.shape[1] - half)
            )
            rows, cols = rows[keep], cols[keep]
            if rows.size > 120000:
                take = np.random.default_rng(1237).choice(rows.size, 120000, replace=False)
                rows, cols = rows[take], cols[take]
            self.coords.append((idx, rows.astype(np.int64), cols.astype(np.int64)))
        self.total = sum(rows.size for _, rows, _ in self.coords)

    def sample(self, batch: int) -> torch.Tensor:
        patches = []
        half = self.patch // 2
        for _ in range(batch):
            which = np.random.randint(len(self.coords))
            idx, rows, cols = self.coords[which]
            k = np.random.randint(rows.size)
            r, c = int(rows[k]), int(cols[k])
            patch = self.images[idx][r - half : r + half + 1, c - half : c + half + 1, :]
            patches.append(np.transpose(patch, (2, 0, 1)))
        return torch.from_numpy(np.stack(patches)).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_DIR), required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/Datasets")
    parser.add_argument("--config", default="")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1237)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = args.config or str(ROOT / "datasets" / args.dataset / "config.yaml")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    source_name = cfg["dataset"]["source_name"]
    target_name = cfg["dataset"]["target_name"]
    num_classes = int(cfg["dataset"]["num_classes"])
    strength = float(cfg.get("training", {}).get("band_alignment_strength", 0.0))

    if args.dataset == "pavia":
        data_path = str(Path(PAVIA_DATA) / "Pavia")
    else:
        data_path = str(Path(args.data_root) / DATASET_DIR[args.dataset])

    source = load_source_scene(data_path, source_name, num_classes, normalize_mode="l2")
    target = load_target_image_only(data_path, target_name, normalize_mode="l2")
    aligned, _ = align_target_bands(source.image, target.image, strength=strength, renormalize=True)

    sampler = PatchSampler([source.image.astype(np.float32), aligned.astype(np.float32)])
    bands = source.image.shape[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MAEEncoder(bands).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = max(1, args.epochs * args.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(args.steps):
            patch = sampler.sample(args.batch_size).to(device)
            mask = (torch.rand(patch.size(0), 49, device=device) < args.mask_ratio)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                recon, tokens = model(patch, mask)
                diff = (recon - tokens).pow(2)
                loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum().clamp_min(1) * bands)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.item())
            step += 1
        print(f"[mae] epoch {epoch}/{args.epochs} loss={running / args.steps:.5f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    output = args.output or str(ROOT / "mae_init" / f"{args.dataset}_mae.pt")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.state_dict(), "dataset": args.dataset, "bands": bands}, output)
    print(f"[mae] saved {output}")


if __name__ == "__main__":
    main()
