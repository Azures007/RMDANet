"""P8: Platt-style class-wise temperature/bias calibration fitted on SOURCE data.

Uses only source labels (allowed) to fit C temperatures + C biases by minimizing
source cross-entropy; the calibrated logits are then applied to the unlabeled
target at evaluation time.  No target labels are used for fitting.

Usage:
  python -u scripts/eval_platt.py --checkpoint <pt> --data-path <dir> \
      --output <json> --confirm-final-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import (  # noqa: E402
    PatchDataset, align_target_bands, load_source_scene, load_target_image_only,
)
from trainer import (  # noqa: E402
    DualReliabilityEncoder, build_parser, build_target_evaluation_loader,
)
from eval_prior_calib import metrics_from_confusion  # noqa: E402


def tta_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x).logits.float()
    logits = logits + model(torch.flip(x, dims=(-1,))).logits.float()
    logits = logits + model(torch.flip(x, dims=(-2,))).logits.float()
    logits = logits + model(torch.flip(x, dims=(-2, -1))).logits.float()
    return logits / 4.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--confirm-final-eval", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_eval:
        raise PermissionError("Explicit --confirm-final-eval required")
    path = Path(args.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_config = checkpoint["args"]
    ap = build_parser()
    ap.set_defaults(**train_config)
    t = ap.parse_args([])
    t.data_path = args.data_path
    t.num_workers = 0
    try:
        t.bands = int(checkpoint["model"]["spectral_encoder.raw_projection.1.weight"].shape[1])
    except Exception:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mode = "median" if t.amplitude_channel else "l2"
    source = load_source_scene(t.data_path, t.source_name, t.num_classes, normalize_mode=mode)
    target = load_target_image_only(t.data_path, t.target_name, normalize_mode=mode)
    aligned, _ = align_target_bands(source.image, target.image,
                                    strength=t.band_alignment_strength,
                                    renormalize=mode != "median")
    has_residual_alpha = any(str(k).endswith("_alpha_logit") for k in checkpoint["model"].keys())
    has_spectral_branch = any(str(k).startswith(("spectral_encoder.", "spectral_fusion."))
                              for k in checkpoint["model"].keys())
    model = DualReliabilityEncoder(
        bands=int(getattr(t, "bands", source.image.shape[-1])),
        num_classes=t.num_classes, projection_dim=t.projection_dim,
        projection_norm=t.projection_norm, aspp_mode=t.aspp_mode,
        pooling_mode=t.pooling_mode, spectral_fusion_mode=t.spectral_fusion_mode,
        amplitude_channel=t.amplitude_channel, use_spectral_branch=has_spectral_branch,
        pool_center_size=getattr(t, "pool_center_size", 5),
        pool_temperature=getattr(t, "pool_temperature", 0.30),
        pool_gate_init=tuple(float(v) for v in str(
            getattr(t, "pool_gate_init", "0.6,0.3,0.1")).split(",")),
        pool_residual=has_residual_alpha,
        pool_alpha_logit=getattr(t, "pool_alpha_logit", -4.0),
        fusion_alpha=getattr(t, "fusion_alpha", 1.0),
        fusion_delta=getattr(t, "fusion_delta", -0.2),
        fusion_tau_a=getattr(t, "fusion_tau_a", 0.20),
        fusion_t_a=getattr(t, "fusion_t_a", 0.10),
        fusion_t_w=getattr(t, "fusion_t_w", 0.15),
        fusion_w_min=getattr(t, "fusion_w_min", 0.10),
        fusion_w_max=getattr(t, "fusion_w_max", 0.90),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # ---- source calibration data (source labels only) ----
    gt = source.ground_truth
    coordinates = np.argwhere(gt > 0)
    labels = gt[gt > 0].astype(np.int64) - 1
    source_dataset = PatchDataset(source.image, coordinates, patch_size=t.patch_size,
                                  labels=labels, augment=False)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128,
                                                shuffle=False, num_workers=0)
    source_logits, source_labels = [], []
    with torch.no_grad():
        for batch in source_loader:
            source_logits.append(tta_logits(model, batch[0].to(device)).cpu())
            source_labels.append(batch[1].cpu())
    source_logits = torch.cat(source_logits)
    source_labels = torch.cat(source_labels)

    log_temperature = torch.zeros(t.num_classes, requires_grad=True)
    bias = torch.zeros(t.num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature, bias], lr=0.05)
    for _ in range(int(args.steps)):
        calibrated = source_logits / log_temperature.exp() + bias
        loss = torch.nn.functional.cross_entropy(calibrated, source_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        source_loss = float(torch.nn.functional.cross_entropy(
            source_logits / log_temperature.exp() + bias, source_labels))
    print(f"[platt] source CE after calibration: {source_loss:.4f}")

    # ---- target evaluation ----
    print("[platt] Loading TARGET GT NOW for reporting ONLY")
    loader = build_target_evaluation_loader(t, aligned)
    logits_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            logits_list.append(tta_logits(model, batch[0].to(device)).cpu())
            label_list.append(batch[1].cpu())
    target_logits = torch.cat(logits_list)
    target_labels = torch.cat(label_list).numpy().astype(np.int64)
    with torch.no_grad():
        calibrated = (target_logits / log_temperature.exp() + bias).numpy()
    prediction = calibrated.argmax(axis=1)
    c = t.num_classes
    confusion = np.zeros((c, c), dtype=np.int64)
    for truth, pred in zip(target_labels, prediction):
        confusion[truth, pred] += 1
    metrics = metrics_from_confusion(confusion)
    metrics["calibration"] = {
        "temperature": log_temperature.exp().detach().cpu().numpy().tolist(),
        "bias": bias.detach().cpu().numpy().tolist(),
        "source_ce": source_loss,
    }
    payload = {"checkpoint": str(path), "selection": checkpoint.get("selection", {}),
               "metrics": metrics}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[platt] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} Kappa={metrics['Kappa']:.4f}")
    print(f"[platt] saved: {out}")


if __name__ == "__main__":
    main()
