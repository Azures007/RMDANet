#!/usr/bin/env python3
"""Efficiency profile for the TAPDA backbone (params / FLOPs / forward time).

Run on the server (needs torch):
    cd ~/autodl-tmp/RMDANet/third_party/tapda
    python ../../scripts/profile_efficiency.py --bands 102 --classes 7 --patch 9
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=int, default=102)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--patch", type=int, default=9)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "third_party" / "tapda"))
    from modules.HybridSN import SSSE_Contrastive  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSSE_Contrastive(args.bands, args.patch, args.classes).to(device).eval()
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"backbone=SSSE params={total} trainable={trainable}")

    dummy = torch.randn(args.batch, args.bands, args.patch, args.patch, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / args.iters
    print(f"forward batch={args.batch} patch={args.patch}: {elapsed * 1000:.1f} ms/iter")

    try:
        from thop import profile  # type: ignore
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs(2*MACs)={2 * macs / 1e9:.2f} G")
    except Exception:
        print("[warn] thop not installed: pip install thop  (FLOPs skipped)")


if __name__ == "__main__":
    main()
