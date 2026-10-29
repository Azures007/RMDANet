#!/usr/bin/env python3
"""Launcher for the faithful MSDA recipe (imports official components)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from trainer import load_config_into_args
from trainer_msda import build_msda_parser, train

DATASETS = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}


def get_args(options):
    parser = build_msda_parser()
    args = parser.parse_args([])
    args = load_config_into_args(
        args, str(ROOT / "datasets" / options.dataset / "config_msda.yaml")
    )
    directory = DATASETS[options.dataset]
    root = Path(options.data_root).expanduser().resolve()
    args.data_path = str(root if root.name.casefold() == directory.casefold() else root / directory)
    args.save_path = str(Path(options.out_root).expanduser().resolve() / options.dataset)
    args.run_name = f"msda_epochs{options.epochs or args.epochs}_seed{options.seed}"
    args.seed = options.seed
    args.gpu = options.gpu
    args.num_workers = options.workers
    if options.msda_dir:
        args.msda_dir = options.msda_dir
    if options.epochs is not None:
        args.epochs = options.epochs
    if options.batch_size is not None:
        args.batch_size = options.batch_size
    if options.use_official_data:
        args.use_official_data = True
    args.use_gt_target_mask = False
    args.target_gt_mix_ratio = 0.0
    args.diagnostic_target_oa = False
    args.skip_final_evaluation = True
    return args


def main():
    parser = argparse.ArgumentParser(description="MSDA-recipe trainer (official components)")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets",
    )
    parser.add_argument("--out-root", default=str(ROOT / "results_msda"))
    parser.add_argument("--msda-dir", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--use-official-data",
        action="store_true",
        help="Use the official MSDA Pavia data pipeline (utils.get_sample_data/get_all_data).",
    )
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    args = get_args(options)
    results = Path(args.save_path) / args.run_name
    summary = {
        "dataset": options.dataset,
        "recipe": "msda",
        "seed": args.seed,
        "data_path": args.data_path,
        "save_path": str(results),
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "source_samples_per_class": args.source_samples_per_class,
        "input_normalization": getattr(args, "input_normalization", "band"),
        "mcc_temperature": args.mcc_temperature,
        "teacher_alpha": args.teacher_alpha,
        "msda_dir": args.msda_dir or "(env MSDA_DIR)",
        "target_GT_during_train": False,
    }
    print("[msda config]", json.dumps(summary, ensure_ascii=False, indent=2))
    if options.dry_run:
        return
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {results}")
    args.save_path = str(results)
    train(args)


if __name__ == "__main__":
    main()
