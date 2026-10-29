#!/usr/bin/env python3
"""Launcher for the literature training recipe (lit-recipe branch).

Usage:
    python scripts/run_lit.py <dataset> <seed> [--data-root ...] [--out-root ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from trainer import load_config_into_args
from trainer_lit import build_lit_parser, train

DATASETS = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}


def get_args(options):
    parser = build_lit_parser()
    args = parser.parse_args([])
    args = load_config_into_args(
        args, str(ROOT / "datasets" / options.dataset / "config_lit.yaml")
    )
    if options.dataset not in DATASETS:
        raise ValueError(options.dataset)
    directory = DATASETS[options.dataset]
    root = Path(options.data_root).expanduser().resolve()
    args.data_path = str(root if root.name.casefold() == directory.casefold() else root / directory)
    args.save_path = str(Path(options.out_root).expanduser().resolve() / options.dataset)
    args.run_name = f"lit_epochs{options.epochs or args.epochs}_seed{options.seed}"
    args.seed = options.seed
    args.gpu = options.gpu
    args.num_workers = options.workers
    if options.epochs is not None:
        args.epochs = options.epochs
    if options.batch_size is not None:
        args.batch_size = options.batch_size
    if options.patch_size is not None:
        args.patch_size = options.patch_size
    # Strict UDA protocol: never touch target GT during training.
    args.use_gt_target_mask = False
    args.target_gt_mix_ratio = 0.0
    args.diagnostic_target_oa = False
    args.skip_final_evaluation = True
    return args


def main():
    parser = argparse.ArgumentParser(description="lit-recipe trainer (TAPDA/MSDA-style)")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets",
    )
    parser.add_argument("--out-root", default=str(ROOT / "results_lit"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    args = get_args(options)
    results = Path(args.save_path) / args.run_name
    summary = {
        "dataset": options.dataset,
        "recipe": "lit",
        "seed": args.seed,
        "data_path": args.data_path,
        "save_path": str(results),
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "source_samples_per_class": args.source_samples_per_class,
        "input_normalization": getattr(args, "input_normalization", "band"),
        "optimizer_type": getattr(args, "optimizer_type", "sgd"),
        "lr": args.lr,
        "target_GT_during_train": False,
        "final_evaluation_in_training": False,
    }
    print("[lit config]", json.dumps(summary, ensure_ascii=False, indent=2))
    if options.dry_run:
        return
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {results}")
    args.save_path = str(results)
    train(args)


if __name__ == "__main__":
    main()
