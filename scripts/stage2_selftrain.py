"""Stage-2 target self-training (no target GT used for pseudo labels).

1) Load a v7.6 checkpoint and predict the target domain.
2) Keep high-confidence, high-margin samples as pseudo labels (class-balanced).
3) Short fine-tune with the existing supervised CE on source (true labels) plus
   CE on target pseudo labels; other losses are unchanged from stage 1 (this
   stage uses only the existing L_sup family, no new loss terms).
4) Save a checkpoint usable by scripts/final_eval_locked.py.

Usage:
  python -u scripts/stage2_selftrain.py --checkpoint <pt> --data-path <dir> \
      --output <pt> --confirm-final-eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import (  # noqa: E402
    PatchDataset, align_target_bands, coordinates_from_mask, labels_at_coordinates,
    load_source_scene, load_target_ground_truth_only, load_target_image_only,
)
from trainer import build_parser  # noqa: E402
from eval_tta_variants import build_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-target", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--cap-per-class", type=int, default=1500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--confirm-final-eval", action="store_true")
    args = p.parse_args()
    if not args.confirm_final_eval:
        raise PermissionError("Explicit --confirm-final-eval required")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    # ---- source labeled data (true labels) ----
    gt_source = source.ground_truth
    coordinates = np.argwhere(gt_source > 0)
    labels = (gt_source[gt_source > 0] - 1).astype(np.int64)
    source_dataset = PatchDataset(source.image, coordinates, patch_size=t.patch_size,
                                  labels=labels, augment=True)

    # ---- target pseudo labels (GT used only for the valid-pixel mask) ----
    gt_target = load_target_ground_truth_only(t.data_path, t.target_name, t.num_classes)
    valid = (gt_target > 0) & (np.linalg.norm(aligned, axis=-1) > 0)
    target_coordinates = coordinates_from_mask(valid)
    probe = PatchDataset(aligned, target_coordinates, patch_size=t.patch_size,
                         labels=None, augment=False)
    probe_loader = torch.utils.data.DataLoader(probe, batch_size=256, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in probe_loader:
            probabilities.append(torch.softmax(model(batch[0].to(device)).logits.float(), dim=1).cpu())
    probabilities = torch.cat(probabilities)
    confidence, prediction = probabilities.max(dim=1)
    top2 = torch.topk(probabilities, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    keep = (confidence >= args.conf) & (margin >= args.margin)
    keep_np = keep.numpy()
    print(f"[stage2] pseudo candidates kept: {int(keep_np.sum())}/{len(keep_np)}")
    kept_coordinates, kept_labels = [], []
    rng = np.random.default_rng(1237)
    for class_id in range(t.num_classes):
        idx = np.nonzero(keep_np & (prediction.numpy() == class_id))[0]
        if len(idx) > args.cap_per_class:
            idx = rng.choice(idx, size=args.cap_per_class, replace=False)
        kept_coordinates.append(target_coordinates[idx])
        kept_labels.append(np.full(len(idx), class_id, dtype=np.int64))
    kept_coordinates = np.concatenate(kept_coordinates, axis=0)
    kept_labels = np.concatenate(kept_labels, axis=0)
    print(f"[stage2] class-balanced pseudo labels: {len(kept_labels)}")
    target_dataset = PatchDataset(aligned, kept_coordinates, patch_size=t.patch_size,
                                  labels=kept_labels, augment=True)

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-4)
    model.train()
    for epoch in range(args.epochs):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        total = 0.0
        for step in range(args.steps):
            try:
                source_x, source_y = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_x, source_y = next(source_iter)
            try:
                target_x, target_y = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, target_y = next(target_iter)
            source_x = source_x.to(device); source_y = source_y.to(device)
            target_x = target_x.to(device); target_y = target_y.to(device)
            loss = F.cross_entropy(model(source_x).logits, source_y, label_smoothing=t.label_smoothing)
            loss = loss + args.lambda_target * F.cross_entropy(model(target_x).logits, target_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"[stage2] epoch {epoch + 1}/{args.epochs} loss={total / max(1, args.steps):.4f}")

    payload = dict(checkpoint)
    payload["model"] = model.state_dict()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[stage2] saved: {out}")


if __name__ == "__main__":
    main()
