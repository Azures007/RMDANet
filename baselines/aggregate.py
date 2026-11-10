#!/usr/bin/env python3
"""Aggregate Pavia (or any dataset) baseline results into one table."""
import argparse
import json
from pathlib import Path


def load_one(p: Path):
    try:
        m = json.loads(p.read_text(encoding="utf8"))["metrics"]
        return m
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pavia")
    p.add_argument("--seed", type=int, default=1237)
    p.add_argument("--out-root", default="results_benchmark")
    p.add_argument("--rmdanet-json", default=None,
                   help="optional RMDANet target_final_once.json for the same dataset")
    a = p.parse_args()

    rows = []
    base = Path(a.out_root) / a.dataset
    for d in sorted(base.glob(f"*_seed{a.seed}")):
        m = load_one(d / "target_final_once.json")
        if m:
            rows.append((d.name.replace(f"_seed{a.seed}", "").upper(), m))
    if a.rmdanet_json:
        m = load_one(Path(a.rmdanet_json))
        if m:
            rows.append(("RMDANET", m))

    print(f"\n{'Model':<10s}{'OA':>9s}{'AA':>9s}{'Kappa':>9s}   (# per-class acc)")
    print("-" * 96)
    for name, m in rows:
        print(f"{name:<10s}{m['OA']:>9.2f}{m['AA']:>9.2f}{m['Kappa']:>9.4f}")
    print()


if __name__ == "__main__":
    main()
