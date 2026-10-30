#!/usr/bin/env python3
"""Locked target-GT evaluation for MSDA-recipe checkpoints (official components)."""
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
from trainer import (
    build_parser,
    build_target_evaluation_loader,
    classification_metrics,
    confusion_matrix_numpy,
)
from trainer_msda import load_official


def _unpack(output):
    """Official models return either tensors or (logits, features) tuples."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--msda-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-final-eval", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_eval:
        raise PermissionError("Explicit --confirm-final-eval required")
    path = Path(args.checkpoint)
    output = Path(args.output)
    if not path.is_file():
        raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Cannot overwrite final evaluation: {output}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "f1", "f2", "args", "selection"):
        if key not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint: missing {key!r}")
    train_config = checkpoint["args"]
    print("[FINAL EVAL] Checkpoint locked:", path, "selection:", checkpoint["selection"])

    official = load_official(args.msda_dir or train_config.get("msda_dir") or None)
    ap = build_parser()
    ap.set_defaults(**train_config)
    t = ap.parse_args([])
    t.data_path = args.data_path
    t.num_workers = 0
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    mode = str(getattr(t, "input_normalization", "") or "band")
    src = load_source_scene(t.data_path, t.source_name, t.num_classes, normalize_mode=mode)
    target = load_target_image_only(t.data_path, t.target_name, normalize_mode=mode)
    source_bands = src.image
    target_bands = target.image
    band_subset = str(getattr(t, "band_subset", "") or "")
    if band_subset:
        from data import select_band_subset

        source_bands = select_band_subset(source_bands, band_subset)
        target_bands = select_band_subset(target_bands, band_subset)
        print(f"[FINAL EVAL] band subset {band_subset!r} -> {source_bands.shape[-1]} bands")
    aligned, _ = align_target_bands(
        source_bands, target_bands, strength=t.band_alignment_strength,
        renormalize=mode == "l2",
    )
    backbone_name = str(getattr(t, "backbone_class", "SSSE_test") or "SSSE_test")
    backbone_factory = official["backbones"].get(backbone_name)
    if backbone_factory is None:
        raise ValueError(
            f"Unknown backbone {backbone_name!r}; available: {sorted(official['backbones'])}"
        )
    backbone = backbone_factory(
        input_channels=aligned.shape[-1], patch_size=t.patch_size, n_classes=t.num_classes
    ).to(device)
    f1 = official["ResClassifier"](t.num_classes, training=True).to(device)
    f2 = official["ResClassifier"](t.num_classes, training=True).to(device)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    f1.load_state_dict(checkpoint["f1"], strict=True)
    f2.load_state_dict(checkpoint["f2"], strict=True)
    backbone.eval()
    f1.eval()
    f2.eval()

    print("[FINAL EVAL] Loading TARGET GT NOW for reporting ONLY")
    loader = build_target_evaluation_loader(t, aligned)
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            features = _unpack(backbone(x))
            logits1 = _unpack(f1(features))
            logits2 = _unpack(f2(features))
            if logits1.dim() == 1:
                logits1 = logits1.unsqueeze(0)
            if logits2.dim() == 1:
                logits2 = logits2.unsqueeze(0)
            logits = 0.5 * (logits1.float() + logits2.float())
            if logits.dim() > 2:
                logits = logits.reshape(logits.size(0), -1)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(labels.numpy())
    target_labels = np.concatenate(labels_all)
    predictions = np.concatenate(preds_all)
    matrix = confusion_matrix_numpy(target_labels, predictions, t.num_classes)
    metrics = classification_metrics(matrix)
    payload = {
        "checkpoint": str(path),
        "selection": checkpoint["selection"],
        "recipe": checkpoint.get("recipe", "msda"),
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"[FINAL EVAL] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} Kappa={metrics['Kappa']:.4f}")
    print("[FINAL EVAL] per-class:", metrics["class_accuracy"])
    print("[FINAL EVAL] saved:", output)


if __name__ == "__main__":
    main()
