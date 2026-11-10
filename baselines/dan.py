#!/usr/bin/env python3
"""DAN baseline (Long et al., 2015): source CE + multi-kernel MMD."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pavia", choices=["pavia", "houston", "hyrank"])
    p.add_argument("--seed", type=int, default=1237)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--source-per-class", type=int, default=180)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mmd-weight", type=float, default=1.0)
    p.add_argument("--out-root", default="results_benchmark")
    a = p.parse_args()
    common.train_baseline(
        "dan", a.dataset, seed=a.seed, epochs=a.epochs,
        steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size,
        lr=a.lr, source_per_class=a.source_per_class, out_root=a.out_root,
        extra={"mmd_weight": a.mmd_weight},
    )


if __name__ == "__main__":
    main()
