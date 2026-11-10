"""Shared protocol for UDA baselines on our HSI data.

All baselines use the SAME data pipeline as RMDANet:
  * PaviaU -> PaviaC (or Houston/HyRANK) via core/data.py
  * per-pixel L2 normalization, band alignment, patch 7x7
  * exactly `source-per-class` labelled source samples per class
  * 50 epochs x 40 steps (2000 updates), batch 128
  * evaluation with core/trainer.evaluate_target (same target GT, one shot)
Only the model / adaptation loss differs between methods.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import data as hsidata  # noqa: E402
import trainer as base_trainer  # noqa: E402

DATASET_DIR = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}
PAVIA_DATA = "/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets"


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
class HSICNN(nn.Module):
    """Small spectral-spatial CNN used by DAN / DANN / JAN baselines."""

    def __init__(self, bands: int, num_classes: int, feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(bands, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(nn.Linear(128, feature_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        conv = self.features(x).flatten(1)
        feat = self.projection(conv)
        logits = self.classifier(self.dropout(feat))
        # Single-branch baseline: all three heads return the same logits and the
        # (unused) spectral branch reports full reliability, so the shared
        # evaluator's conservative posterior fusion reduces to argmax(logits).
        return SimpleNamespace(
            logits=logits,
            features=feat,
            context_logits=logits,
            spectral_logits=logits,
            spectral_reliability=torch.ones(x.size(0), device=x.device),
        )


# --------------------------------------------------------------------------- #
# protocol helpers
# --------------------------------------------------------------------------- #
def protocol_args(dataset: str, data_path: str | None = None) -> argparse.Namespace:
    ap = base_trainer.build_parser()
    a = ap.parse_args([])
    a = base_trainer.load_config_into_args(a, str(ROOT / "datasets" / dataset / "config.yaml"))
    a.num_workers = 0
    a.skip_final_evaluation = True
    if data_path is None:
        if dataset == "pavia":
            a.data_path = str(Path(PAVIA_DATA) / "Pavia")
        else:
            a.data_path = str(Path("/root/autodl-tmp/Datasets") / DATASET_DIR[dataset])
    else:
        a.data_path = data_path
    return a


def _extract_patch(image: np.ndarray, r: int, c: int, size: int) -> np.ndarray:
    half = size // 2
    h, w, bands = image.shape
    r0, r1 = max(0, r - half), min(h, r + half + 1)
    c0, c1 = max(0, c - half), min(w, c + half + 1)
    patch = np.zeros((size, size, bands), dtype=np.float32)
    patch[r0 - (r - half): r1 - (r - half), c0 - (c - half): c1 - (c - half)] = image[r0:r1, c0:c1]
    return patch.transpose(2, 0, 1)


def build_arrays(args: argparse.Namespace, seed: int, source_per_class: int, target_pool: int):
    norm_mode = "median" if getattr(args, "amplitude_channel", False) else "l2"
    src = hsidata.load_source_scene(args.data_path, args.source_name, args.num_classes,
                                   normalize_mode=norm_mode)
    tgt = hsidata.load_target_image_only(args.data_path, args.target_name, normalize_mode=norm_mode)
    aligned, _ = hsidata.align_target_bands(
        src.image, tgt.image,
        strength=args.band_alignment_strength,
        renormalize=norm_mode != "median",
    )
    rng = np.random.default_rng(seed)
    gt = src.ground_truth
    xs, ys = [], []
    for cls in [int(v) for v in np.unique(gt) if int(v) > 0]:
        coords = np.argwhere(gt == cls)
        rng.shuffle(coords)
        for r, c in coords[:source_per_class]:
            xs.append(_extract_patch(src.image, int(r), int(c), args.patch_size))
            ys.append(cls - 1)
    source_x = torch.from_numpy(np.stack(xs))
    source_y = torch.tensor(ys, dtype=torch.long)

    valid = (np.linalg.norm(aligned, axis=-1) > 0)
    tcoords = np.argwhere(valid)
    rng.shuffle(tcoords)
    tcoords = tcoords[:min(target_pool, len(tcoords))]
    target_x = torch.from_numpy(
        np.stack([_extract_patch(aligned, int(r), int(c), args.patch_size) for r, c in tcoords])
    )
    return src, aligned, source_x, source_y, target_x


@torch.no_grad()
def evaluate(model: nn.Module, args: argparse.Namespace, target_image: np.ndarray) -> dict:
    loader = base_trainer.build_target_evaluation_loader(args, target_image)
    device = next(model.parameters()).device
    return base_trainer.evaluate_target(model, loader, device,
                                        num_classes=args.num_classes,
                                        use_tta=bool(getattr(args, "final_tta", False)))


# --------------------------------------------------------------------------- #
# training loop (shared): source CE + method-specific adaptation loss
# --------------------------------------------------------------------------- #
def train_baseline(
    method: str,
    dataset: str,
    seed: int = 1237,
    epochs: int = 50,
    steps_per_epoch: int = 40,
    batch_size: int = 128,
    lr: float | None = None,
    source_per_class: int = 180,
    out_root: str = "results_benchmark",
    extra: dict | None = None,
) -> dict:
    args = protocol_args(dataset)
    if lr is None:
        lr = float(args.lr)
    extra = extra or {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_trainer.set_seed(seed, True)

    src, aligned, sx, sy, tx = build_arrays(args, seed, source_per_class,
                                            int(args.target_train_max_samples))
    model = HSICNN(src.image.shape[-1], args.num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    total_steps = int(epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    adapt = _make_adapter(method, model, device, extra)
    sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
    n_s, n_t = sx.size(0), tx.size(0)

    for step in range(total_steps):
        model.train()
        optim.zero_grad(set_to_none=True)
        si = torch.randint(0, n_s, (batch_size,), device=device)
        ti = torch.randint(0, n_t, (batch_size,), device=device)
        out_s = model(sx[si])
        out_t = model(tx[ti])
        loss = F.cross_entropy(out_s.logits, sy[si])
        loss = loss + adapt(out_s, out_t, step / max(1, total_steps - 1), ys=sy[si])
        loss.backward()
        if hasattr(adapt, "post_step"):
            adapt.post_step()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()
        if (step + 1) % 400 == 0:
            print(f"[{method}] step {step + 1}/{total_steps} loss={loss.item():.4f}", flush=True)

    metrics = evaluate(model, args, aligned)
    run_dir = Path(out_root) / dataset / f"{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "selection": {"method": method, "seed": seed}}, run_dir / "selected_model.pt")
    (run_dir / "target_final_once.json").write_text(
        json.dumps({"metrics": metrics, "method": method, "seed": seed}, indent=2),
        encoding="utf8",
    )
    (run_dir / "config.json").write_text(
        json.dumps({"method": method, "seed": seed, "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch, "batch_size": batch_size,
                    "lr": lr, "source_per_class": source_per_class, **extra}, indent=2),
        encoding="utf8",
    )
    print(f"[{method}] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} "
          f"Kappa={metrics['Kappa']:.4f} -> {run_dir}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# method adapters
# --------------------------------------------------------------------------- #
class _Zero(nn.Module):
    def forward(self, out_s, out_t, p, ys=None):
        return out_s.logits.new_zeros(())


def _make_adapter(method, model, device, extra):
    if method == "source":
        return _Zero()
    if method == "dan":
        from losses import DANLoss
        return DANLoss(weight=float(extra.get("mmd_weight", 1.0)))
    if method == "dann":
        from losses import DANNLoss
        return DANNLoss(model.feature_dim, device,
                        domain_weight=float(extra.get("domain_weight", 1.0)))
    if method == "jan":
        from losses import JANLoss
        return JANLoss(weight=float(extra.get("jmmd_weight", 1.0)))
    if method == "mtlda":
        from losses import MTLDALoss
        return MTLDALoss(model.feature_dim, device,
                         domain_weight=float(extra.get("domain_weight", 0.5)),
                         entropy_weight=float(extra.get("entropy_weight", 0.05)))
    if method == "cacl":
        from losses import CACLLoss
        return CACLLoss(contrast_weight=float(extra.get("contrast_weight", 0.5)),
                        pseudo_weight=float(extra.get("pseudo_weight", 0.3)),
                        confidence=float(extra.get("confidence", 0.9)))
    if method == "uipcl":
        from losses import UIPCLLoss
        return UIPCLLoss(model.num_classes,
                         prototype_weight=float(extra.get("prototype_weight", 0.5)),
                         pseudo_weight=float(extra.get("pseudo_weight", 0.5)))
    if method == "pcdm":
        from losses import PCDMLoss
        return PCDMLoss(model.num_classes,
                        prior_weight=float(extra.get("prior_weight", 0.5)),
                        contrast_weight=float(extra.get("contrast_weight", 0.3)),
                        confidence=float(extra.get("confidence", 0.9)))
    if method == "vcmoe":
        from losses import VCMoELoss
        return VCMoELoss(model.feature_dim, device,
                         adv_weight=float(extra.get("adv_weight", 0.3)),
                         contrast_weight=float(extra.get("contrast_weight", 0.3)),
                         diversity_weight=float(extra.get("diversity_weight", 0.1)))
    raise ValueError(f"unknown baseline method: {method}")
