#!/usr/bin/env python3
"""Diagnostic: evaluate our msda checkpoint with the OFFICIAL data pipeline.

If this script reports ~93 on Pavia while our own eval reports ~62, the bug is
in our evaluation data preparation (patch extraction / normalization), not in
the trained model.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _unpack(output):
    """Official ResClassifier returns logits directly in eval mode."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--msda-dir", default=os.environ.get("MSDA_DIR", "/root/autodl-tmp/project"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    msda = args.msda_dir
    sys.path.insert(0, msda)
    from HybridSN import SSSE_test  # type: ignore
    from basenet import ResClassifier  # type: ignore
    from utils import get_all_data, load_data_pavia  # type: ignore

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    patch = int(train_args["patch_size"])
    half_width = patch // 2
    num_classes = int(train_args["num_classes"])
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    data_t, label_t = load_data_pavia(f"{msda}/data/Pavia/pavia.mat", f"{msda}/data/Pavia/pavia_gt_7.mat")
    test_id, test_x, test_y, grid, rand_perm, row, column = get_all_data(data_t, label_t, half_width)
    print("[diag] official target set:", test_x.shape, "classes", np.unique(test_y))

    backbone = SSSE_test(
        input_channels=test_x.shape[1], patch_size=patch, n_classes=num_classes
    ).to(device)
    f1 = ResClassifier(num_classes, training=True).to(device)
    f2 = ResClassifier(num_classes, training=True).to(device)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    f1.load_state_dict(checkpoint["f1"], strict=True)
    f2.load_state_dict(checkpoint["f2"], strict=True)
    backbone.eval()
    f1.eval()
    f2.eval()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_x), torch.tensor(test_y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    preds1, preds2, preds_ens, labels = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = _unpack(backbone(x))
            logits1 = _unpack(f1(features))
            logits2 = _unpack(f2(features))
            if logits1.dim() == 1:
                logits1 = logits1.unsqueeze(0)
            if logits2.dim() == 1:
                logits2 = logits2.unsqueeze(0)
            preds1.append(logits1.argmax(dim=1).cpu().numpy())
            preds2.append(logits2.argmax(dim=1).cpu().numpy())
            preds_ens.append(((logits1 + logits2) / 2).argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    labels = np.concatenate(labels)
    for name, preds in (("f1", preds1), ("f2", preds2), ("ens", preds_ens)):
        predictions = np.concatenate(preds)
        oa = 100.0 * (predictions == labels).mean()
        per_class = []
        for class_id in range(num_classes):
            mask = labels == class_id
            per_class.append(100.0 * (predictions[mask] == class_id).mean() if mask.any() else 0.0)
        print(f"[diag] official-pipeline {name}: OA={oa:.2f} per-class="
              + ", ".join(f"{v:.2f}" for v in per_class))


if __name__ == "__main__":
    main()
