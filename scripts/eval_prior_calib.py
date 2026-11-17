"""P2: inference-time class-prior calibration (no retraining, no target labels).

1. Run the checkpoint over the unlabeled target patches and collect probabilities.
2. Estimate the target class prior with the Saerens-Latinne-Decaestecker EM
   algorithm (only model outputs are used; target GT is never touched).
3. Adjust logits:  z' = z + alpha * log(pi_target / pi_source),  pi_source uniform.
4. Predict and report OA / AA / Kappa / per-class accuracy.

Usage:
  python -u scripts/eval_prior_calib.py \
      --checkpoint <selected_model.pt> --data-path <dataset dir> \
      --output <json> --alpha 1.0 --confirm-final-eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "core"))

from data import align_target_bands, load_source_scene, load_target_image_only  # noqa: E402
from trainer import (  # noqa: E402
    DualReliabilityEncoder,
    build_parser,
    build_target_evaluation_loader,
    predict_probabilities,
)


def em_target_prior(probabilities: np.ndarray, source_prior: np.ndarray,
                    iterations: int = 30) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM estimate of the target prior."""
    prior = np.full_like(source_prior, 1.0 / len(source_prior), dtype=np.float64)
    ratio = np.ones_like(source_prior)
    for _ in range(iterations):
        ratio = prior / np.clip(source_prior, 1e-8, None)
        weighted = probabilities * ratio[None, :]
        normalizer = weighted.sum(axis=1, keepdims=True).clip(min=1e-12)
        posterior = weighted / normalizer
        prior = posterior.mean(axis=0)
        prior = prior / prior.sum()
    return prior


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    total = matrix.sum()
    oa = float(np.trace(matrix) / max(total, 1) * 100.0)
    recalls = np.diag(matrix) / np.clip(matrix.sum(axis=1), 1, None)
    aa = float(recalls.mean() * 100.0)
    row = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    col = matrix.sum(axis=0, keepdims=True) / max(total, 1)
    pe = float((row.sum(axis=0) * col).sum()) if False else float(
        (matrix.sum(axis=0) / max(total, 1) * matrix.sum(axis=1) / max(total, 1)).sum())
    kappa = float((oa / 100.0 - pe) / max(1e-12, 1.0 - pe))
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "class_accuracy": (recalls * 100.0).tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior_smooth", type=float, default=0.5)
    parser.add_argument("--prior_floor", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-final-eval", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_eval:
        raise PermissionError("Explicit --confirm-final-eval required")
    path = Path(args.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Cannot overwrite final evaluation: {out}")

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
    aligned, _ = align_target_bands(
        source.image, target.image, strength=t.band_alignment_strength,
        renormalize=mode != "median")
    has_residual_alpha = any(str(k).endswith("_alpha_logit") for k in checkpoint["model"].keys())
    has_spectral_branch = any(
        str(k).startswith(("spectral_encoder.", "spectral_fusion.")) for k in checkpoint["model"].keys())
    model = DualReliabilityEncoder(
        bands=int(getattr(t, "bands", source.image.shape[-1])),
        num_classes=t.num_classes,
        projection_dim=t.projection_dim,
        projection_norm=t.projection_norm,
        aspp_mode=t.aspp_mode,
        pooling_mode=t.pooling_mode,
        spectral_fusion_mode=t.spectral_fusion_mode,
        amplitude_channel=t.amplitude_channel,
        use_spectral_branch=has_spectral_branch,
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
    print("[prior-calib] Loading TARGET GT NOW for reporting ONLY")
    loader = build_target_evaluation_loader(t, aligned)

    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            labels = batch[1]
            probabilities = predict_probabilities(model, x, use_tta=t.final_tta)
            all_probabilities.append(probabilities.float().cpu())
            all_labels.append(labels.cpu())
    probabilities = torch.cat(all_probabilities).numpy().astype(np.float64)
    labels = torch.cat(all_labels).numpy().astype(np.int64)
    num_classes = t.num_classes
    source_prior = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)

    target_prior = em_target_prior(probabilities, source_prior)
    target_prior = ((1.0 - args.prior_smooth) * target_prior
                    + args.prior_smooth * source_prior)
    target_prior = np.clip(target_prior, args.prior_floor, None)
    target_prior = target_prior / target_prior.sum()
    log_ratio = np.log(np.clip(target_prior, 1e-8, None) /
                       np.clip(source_prior, 1e-8, None))
    adjusted = probabilities * np.exp(args.alpha * log_ratio)[None, :]
    prediction = adjusted.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(labels, prediction):
        confusion[truth, pred] += 1
    metrics = metrics_from_confusion(confusion)
    metrics["alpha"] = float(args.alpha)
    metrics["target_prior"] = target_prior.tolist()
    payload = {
        "checkpoint": str(path),
        "selection": checkpoint.get("selection", {}),
        "metrics": metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[prior-calib] target_prior={np.round(target_prior, 4).tolist()}")
    print(f"[prior-calib] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} "
          f"Kappa={metrics['Kappa']:.4f}")
    print(f"[prior-calib] saved: {out}")


if __name__ == "__main__":
    main()
