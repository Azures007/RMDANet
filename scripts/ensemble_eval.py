#!/usr/bin/env python3
"""Locked ensemble evaluation of two independently trained checkpoints.

Model A: main-branch reliability model (DualReliabilityEncoder, L2 input).
Model B: lit-recipe model (SSSE + two classifiers, per-band z-score input).

Both checkpoints were trained without target labels; the target GT is opened
only here, once, for reporting.  Softmax probabilities are averaged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from data import align_target_bands, load_source_scene, load_target_image_only
from model_lit import ResClassifier, SSSE
from trainer import (
    DualReliabilityEncoder,
    build_parser,
    build_target_evaluation_loader,
    classification_metrics,
    confusion_matrix_numpy,
)


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _args_from_checkpoint(checkpoint: dict):
    parser = build_parser()
    parser.set_defaults(**checkpoint["args"])
    return parser.parse_args([])


def _reliability_model(checkpoint: dict, data_path: str, device: torch.device):
    args = _args_from_checkpoint(checkpoint)
    args.data_path = data_path
    args.num_workers = 0
    mode = str(
        getattr(args, "input_normalization", "")
        or ("median" if getattr(args, "amplitude_channel", False) else "l2")
    )
    amplitude_input = bool(getattr(args, "amplitude_input", False))
    source = load_source_scene(args.data_path, args.source_name, args.num_classes,
                               normalize_mode=mode)
    target = load_target_image_only(args.data_path, args.target_name, normalize_mode=mode)
    aligned, _ = align_target_bands(
        source.image, target.image, strength=args.band_alignment_strength,
        renormalize=mode == "l2",
    )
    if amplitude_input:
        aligned = np.concatenate([aligned, target.amplitude[..., None]], axis=-1)
    model = DualReliabilityEncoder(
        bands=aligned.shape[-1],
        num_classes=args.num_classes,
        projection_dim=args.projection_dim,
        projection_norm=args.projection_norm,
        aspp_mode=args.aspp_mode,
        pooling_mode=args.pooling_mode,
        spectral_fusion_mode=args.spectral_fusion_mode,
        amplitude_channel=args.amplitude_channel,
        amplitude_input=amplitude_input,
        backbone=str(getattr(args, "backbone", "ours")),
        enable_dual_head=bool(
            getattr(args, "dual_head_filter", False)
            or getattr(args, "lambda_dual_source", 0.0) > 0
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return args, model, aligned


def _lit_model(checkpoint: dict, data_path: str, device: torch.device):
    args = _args_from_checkpoint(checkpoint)
    args.data_path = data_path
    args.num_workers = 0
    mode = str(getattr(args, "input_normalization", "") or "band")
    source = load_source_scene(args.data_path, args.source_name, args.num_classes,
                               normalize_mode=mode)
    target = load_target_image_only(args.data_path, args.target_name, normalize_mode=mode)
    aligned, _ = align_target_bands(
        source.image, target.image, strength=args.band_alignment_strength,
        renormalize=mode == "l2",
    )
    feature_dim = int(getattr(args, "lit_feature_dim", 128))
    backbone = SSSE(aligned.shape[-1], args.patch_size, feature_dim=feature_dim).to(device)
    f1 = ResClassifier(feature_dim, args.num_classes).to(device)
    f2 = ResClassifier(feature_dim, args.num_classes).to(device)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    f1.load_state_dict(checkpoint["f1"], strict=True)
    f2.load_state_dict(checkpoint["f2"], strict=True)
    backbone.eval()
    f1.eval()
    f2.eval()
    return args, backbone, f1, f2, aligned


@torch.no_grad()
def _reliability_probabilities(model, loader, device, tta: bool = False):
    probabilities, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if tta:
            views = (x, torch.flip(x, (-1,)), torch.flip(x, (-2,)), torch.flip(x, (-2, -1)))
            probs = torch.stack(
                [F.softmax(model(view).logits.float(), dim=1) for view in views], dim=0
            ).mean(dim=0)
        else:
            probs = F.softmax(model(x).logits.float(), dim=1)
        probabilities.append(probs.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


@torch.no_grad()
def _lit_probabilities(backbone, f1, f2, loader, device, tta: bool = False):
    probabilities, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = (x,) if not tta else (
            x, torch.flip(x, (-1,)), torch.flip(x, (-2,)), torch.flip(x, (-2, -1))
        )
        probs = []
        for view in views:
            features = backbone(view)
            logits = 0.5 * (f1(features) + f2(features))
            probs.append(F.softmax(logits.float(), dim=1))
        probabilities.append(torch.stack(probs, dim=0).mean(dim=0).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, num_classes: int) -> dict:
    matrix = confusion_matrix_numpy(labels, probabilities.argmax(axis=1), num_classes)
    return classification_metrics(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked ensemble of two checkpoints")
    parser.add_argument("--checkpoint-a", required=True, help="main-branch reliability model")
    parser.add_argument("--checkpoint-b", required=True, help="lit-recipe model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true", help="report a weight sensitivity sweep")
    parser.add_argument("--tta", action="store_true", help="flip TTA on both models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-final-eval", action="store_true")
    options = parser.parse_args()
    if not options.confirm_final_eval:
        raise PermissionError("Explicit --confirm-final-eval required")
    output = Path(options.output)
    if output.exists():
        raise FileExistsError(f"Cannot overwrite final evaluation: {output}")
    device = torch.device(
        options.device if torch.cuda.is_available() and options.device.startswith("cuda") else "cpu"
    )

    checkpoint_a = _load(Path(options.checkpoint_a))
    checkpoint_b = _load(Path(options.checkpoint_b))
    print("[ENSEMBLE] A:", options.checkpoint_a)
    print("[ENSEMBLE] B:", options.checkpoint_b)

    args_a, model_a, aligned_a = _reliability_model(checkpoint_a, options.data_path, device)
    args_b, backbone_b, f1_b, f2_b, aligned_b = _lit_model(checkpoint_b, options.data_path, device)

    print("[ENSEMBLE] Loading TARGET GT NOW for reporting ONLY")
    loader_a = build_target_evaluation_loader(args_a, aligned_a)
    loader_b = build_target_evaluation_loader(args_b, aligned_b)
    probs_a, labels_a = _reliability_probabilities(model_a, loader_a, device, tta=options.tta)
    probs_b, labels_b = _lit_probabilities(backbone_b, f1_b, f2_b, loader_b, device, tta=options.tta)
    if probs_a.shape[0] != probs_b.shape[0] or not np.array_equal(labels_a, labels_b):
        raise RuntimeError(
            "The two preprocessing pipelines produced different target sample sets."
        )

    weight_a = float(np.clip(options.weight_a, 0.0, 1.0))
    probs_ensemble = weight_a * probs_a + (1.0 - weight_a) * probs_b
    num_classes = int(args_a.num_classes)
    metrics_a = _metrics(labels_a, probs_a, num_classes)
    metrics_b = _metrics(labels_a, probs_b, num_classes)
    metrics_ensemble = _metrics(labels_a, probs_ensemble, num_classes)

    print(f"[ENSEMBLE] A  OA={metrics_a['OA']:.2f} AA={metrics_a['AA']:.2f}")
    print(f"[ENSEMBLE] B  OA={metrics_b['OA']:.2f} AA={metrics_b['AA']:.2f}")
    print(
        f"[ENSEMBLE] A/{weight_a:.2f}+B/{1.0-weight_a:.2f}  "
        f"OA={metrics_ensemble['OA']:.2f} AA={metrics_ensemble['AA']:.2f} "
        f"Kappa={metrics_ensemble['Kappa']:.4f}"
    )
    print("[ENSEMBLE] per-class:", metrics_ensemble["class_accuracy"])

    sweep_rows = []
    if options.sweep:
        print("[ENSEMBLE] weight sensitivity (A weight -> OA / AA), for analysis only:")
        for weight in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            probs = weight * probs_a + (1.0 - weight) * probs_b
            candidate = _metrics(labels_a, probs, num_classes)
            sweep_rows.append({"weight_a": weight, "OA": candidate["OA"], "AA": candidate["AA"]})
            print(f"  {weight:.2f} -> OA={candidate['OA']:.2f} AA={candidate['AA']:.2f}")

    payload = {
        "checkpoint_a": options.checkpoint_a,
        "checkpoint_b": options.checkpoint_b,
        "weight_a": weight_a,
        "individual": {"a": metrics_a, "b": metrics_b},
        "metrics": metrics_ensemble,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    print("[ENSEMBLE] saved:", output)


if __name__ == "__main__":
    main()
