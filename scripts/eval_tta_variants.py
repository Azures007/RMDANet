"""P9 (MC-Dropout averaging) and P10 (sliding-window overlap fusion) evaluation.

Both are inference-only variants applied on top of the v7.4 checkpoint; the
target-prior calibration (P2) is applied afterwards so results are comparable
with the v7.4 reported numbers.

Usage:
  python -u scripts/eval_tta_variants.py --checkpoint <pt> --data-path <dir> \
      --mode mcdropout|sliding --output <json> --confirm-final-eval
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
    PatchDataset, align_target_bands, coordinates_from_mask, labels_at_coordinates,
    load_source_scene, load_target_ground_truth_only, load_target_image_only,
)
from trainer import DualReliabilityEncoder, build_parser  # noqa: E402
from eval_prior_calib import em_target_prior, metrics_from_confusion  # noqa: E402


def build_model(checkpoint, t, device):
    has_residual_alpha = any(str(k).endswith("_alpha_logit") for k in checkpoint["model"].keys())
    has_spectral_branch = any(str(k).startswith(("spectral_encoder.", "spectral_fusion."))
                              for k in checkpoint["model"].keys())
    model = DualReliabilityEncoder(
        bands=int(getattr(t, "bands", 0)), num_classes=t.num_classes,
        projection_dim=t.projection_dim, projection_norm=t.projection_norm,
        aspp_mode=t.aspp_mode, pooling_mode=t.pooling_mode,
        spectral_fusion_mode=t.spectral_fusion_mode, amplitude_channel=t.amplitude_channel,
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mode", choices=["mcdropout", "sliding"], required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", required=True)
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
        raise FileExistsError(out)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ap = build_parser()
    ap.set_defaults(**checkpoint["args"])
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
    model = build_model(checkpoint, t, device)
    model.eval()

    print("[tta-variant] Loading TARGET GT NOW for reporting ONLY")
    gt = load_target_ground_truth_only(t.data_path, t.target_name, t.num_classes)
    valid = (gt > 0) & (np.linalg.norm(aligned, axis=-1) > 0)
    coordinates = coordinates_from_mask(valid)
    labels = labels_at_coordinates(gt, coordinates) - 1
    labels_t = torch.from_numpy(labels.astype(np.int64))

    def run_loader(loader):
        outputs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                outputs.append(torch.softmax(model(x).logits.float(), dim=1).cpu())
        return torch.cat(outputs)

    if args.mode == "mcdropout":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.train()
        dataset = PatchDataset(aligned, coordinates, patch_size=t.patch_size,
                               labels=labels, augment=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        probabilities = torch.zeros(len(dataset), t.num_classes)
        for _ in range(int(args.repeats)):
            probabilities += run_loader(loader)
        probabilities /= float(args.repeats)
    else:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        height, width = aligned.shape[:2]
        probabilities = torch.zeros(len(coordinates), t.num_classes)
        for dy, dx in offsets:
            shifted = coordinates + np.array([dy, dx])
            shifted[:, 0] = np.clip(shifted[:, 0], 0, height - 1)
            shifted[:, 1] = np.clip(shifted[:, 1], 0, width - 1)
            dataset = PatchDataset(aligned, shifted, patch_size=t.patch_size,
                                   labels=labels, augment=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            probabilities += run_loader(loader)
        probabilities /= float(len(offsets))

    probs = probabilities.numpy().astype(np.float64)
    num_classes = t.num_classes
    source_prior = np.full(num_classes, 1.0 / num_classes)
    target_prior = em_target_prior(probs, source_prior)
    target_prior = 0.5 * target_prior + 0.5 * source_prior
    target_prior = np.clip(target_prior, 0.02, None)
    target_prior = target_prior / target_prior.sum()
    adjusted = probs * np.exp(0.5 * np.log(target_prior / source_prior))[None, :]
    prediction = adjusted.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(labels_t.numpy(), prediction):
        confusion[truth, pred] += 1
    metrics = metrics_from_confusion(confusion)
    metrics["variant"] = args.mode
    metrics["repeats"] = int(args.repeats)
    payload = {"checkpoint": str(path), "selection": checkpoint.get("selection", {}),
               "metrics": metrics}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[tta-variant] {args.mode} OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} "
          f"Kappa={metrics['Kappa']:.4f}")
    print(f"[tta-variant] saved: {out}")


if __name__ == "__main__":
    main()
