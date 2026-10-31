#!/usr/bin/env python3
"""Paired significance tests for two per-seed OA lists.

Usage:
    python scripts/significance_test.py --a 92.3,91.9,92.4,92.0,92.2 --b 88.2,88.5,88.1,88.4,88.3
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="comma separated per-seed OA of method A")
    parser.add_argument("--b", required=True, help="comma separated per-seed OA of method B")
    parser.add_argument("--name-a", default="ours")
    parser.add_argument("--name-b", default="baseline")
    args = parser.parse_args()

    a = np.array([float(item) for item in args.a.split(",")], dtype=np.float64)
    b = np.array([float(item) for item in args.b.split(",")], dtype=np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"seed counts differ: {a.shape} vs {b.shape}")
    if a.size < 2:
        raise SystemExit("need at least two seeds")

    diff = a - b
    print(f"A = {args.name_a}: mean={a.mean():.2f} std={a.std(ddof=1):.2f} n={a.size}")
    print(f"B = {args.name_b}: mean={b.mean():.2f} std={b.std(ddof=1):.2f} n={b.size}")
    print(f"delta = {diff.mean():+.2f} ± {diff.std(ddof=1):.2f}")

    try:
        from scipy import stats
    except ImportError:
        print("[warn] scipy missing, only descriptive stats reported")
        return

    t_stat, t_p = stats.ttest_rel(a, b)
    print(f"paired t-test: t={t_stat:.3f} p={t_p:.4f}")
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
        print(f"wilcoxon:      W={w_stat:.3f} p={w_p:.4f}")
    except ValueError as exc:
        print(f"[warn] wilcoxon skipped: {exc}")
    alpha = 0.05
    print("significant at 0.05:", bool(t_p < alpha))


if __name__ == "__main__":
    main()
