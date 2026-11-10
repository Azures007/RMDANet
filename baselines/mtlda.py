#!/usr/bin/env python3
"""MTLDA baseline: classification + domain adversarial + entropy minimisation."""
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
    p.add_argument("--normalize-mode", default="l2", choices=["l2", "median"])
    p.add_argument("--out-root", default="results_benchmark")
    p.add_argument("--domain-weight", type=float, default=0.5)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    a = p.parse_args()
    common.train_baseline(
        "mtlda", a.dataset, seed=a.seed, epochs=a.epochs,
        steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size,
        lr=a.lr, source_per_class=a.source_per_class,
        out_root=a.out_root, normalize_mode=a.normalize_mode,
        extra=dict(
            domain_weight=a.domain_weight,
            entropy_weight=a.entropy_weight,
        ),
    )


if __name__ == "__main__":
    main()
