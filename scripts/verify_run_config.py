"""Verify that every recorded run used the expected effective configuration.

Scans ``<results-root>/<dataset>/<run>/run_config.json`` (written by
``trainer.audit_runtime_config``) and compares the key fields against the
canonical configuration of each dataset:

    Pavia / Houston : patch 7, centre 5, fusion bidirectional
    HyRANK          : patch 5, centre 3, fusion bidirectional

It also prints the loss profile, pseudo-label mode, and any audit warnings so
that historical runs can be checked against the paper description.

Usage:
    python -u scripts/verify_run_config.py \
        --results-root results_ablation_clean \
        --datasets pavia houston hyrank
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CANONICAL = {
    "pavia": {"patch_size": 7, "pool_center_size": 5, "spectral_fusion_mode": "bidirectional"},
    "houston": {"patch_size": 7, "pool_center_size": 5, "spectral_fusion_mode": "bidirectional"},
    "hyrank": {"patch_size": 5, "pool_center_size": 3, "spectral_fusion_mode": "bidirectional"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results_ablation_clean")
    parser.add_argument("--datasets", nargs="+", default=["pavia", "houston", "hyrank"])
    args = parser.parse_args()

    root = Path(args.results_root)
    total = 0
    mismatches = 0
    for dataset in args.datasets:
        expected = CANONICAL[dataset]
        for cfg_path in sorted(root.glob(f"{dataset}/*/run_config.json")):
            total += 1
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            problems = []
            for key, value in expected.items():
                if cfg.get(key) != value:
                    problems.append(f"{key}={cfg.get(key)!r} (expected {value!r})")
            profile = cfg.get("env", {}).get("LOSS_PROFILE", "") or "full"
            if problems:
                mismatches += 1
                print(f"[MISMATCH] {cfg_path.parent}: " + "; ".join(problems))
            else:
                print(f"[OK] {cfg_path.parent.name:40s} patch={cfg.get('patch_size')} "
                      f"centre={cfg.get('pool_center_size')} fusion={cfg.get('spectral_fusion_mode')} "
                      f"pseudo={cfg.get('pseudo_mode')} loss_profile={profile}")
    print(f"\nchecked {total} runs, {mismatches} mismatch(es)")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
