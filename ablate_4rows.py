#!/usr/bin/env python3
"""Combined 4-row ablation for the clean v7.0 build.

Rows: base / drse / rftm / drse_rftm / full
Baseline = complete framework minus {RFTM, DRSE, RTML}.
DRSE  = center-guided pooling + reliability-gated spectral fusion (+aux)
RTML  = cross-epoch reliability memory (history + dependent gates)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
import trainer

DATASET_DIR = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}
PAVIA_DATA = "/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets"

ROWS = {
    "base":      dict(prat=0, drse=0, rtml=0),
    "drse":      dict(prat=0, drse=1, rtml=0),
    "rftm":      dict(prat=1, drse=0, rtml=0),
    "drse_rftm": dict(prat=1, drse=1, rtml=0),
    "full":      dict(prat=1, drse=1, rtml=1),
}

_orig_apply = trainer.apply_ablation_profile


def apply_ablate(args):
    args.ablation_id = "full"
    args = _orig_apply(args)
    row = ROWS[args.row]

    args.use_prat = bool(row["prat"])

    if row["drse"]:
        args.pooling_mode = "guided"
        args.spectral_fusion_mode = "reliable"
        args.use_spectral_aux = True
    else:
        args.pooling_mode = "gap"
        args.spectral_fusion_mode = "off"
        args.use_spectral_aux = False

    if row["rtml"]:
        args.pseudo_mode = "history"
        args.use_probability_history = True
        args.use_time_score = True
        args.use_feature_score = True
        args.use_view_score = True
        args.use_guard = True
        args.use_prototype_veto = True
        args.use_spatial_reliability = True
        args.use_target_prototypes = True
        args.reliability_weighted_prototype_update = True
        args.adaptive_anchor = True
        args.use_confusion_loss = True
    else:
        args.pseudo_mode = "current"
        args.use_probability_history = False
        args.use_time_score = False
        args.use_feature_score = False
        args.use_view_score = False
        args.use_guard = False
        args.use_prototype_veto = False
        args.use_spatial_reliability = False
        args.use_target_prototypes = False
        args.reliability_weighted_prototype_update = False
        args.adaptive_anchor = False
        args.use_confusion_loss = False
    return args


trainer.apply_ablation_profile = apply_ablate


def main():
    p = trainer.build_parser()
    p.add_argument("--row", choices=sorted(ROWS), required=True)
    p.add_argument("--dataset", choices=sorted(DATASET_DIR), required=True)
    p.add_argument("--data-root", default="/root/autodl-tmp/Datasets")
    p.add_argument("--out-root", default=str(ROOT / "results_ablation_clean"))
    a = p.parse_args()

    a = trainer.load_config_into_args(a, str(ROOT / "datasets" / a.dataset / "config.yaml"))
    if a.dataset == "pavia":
        a.data_path = str(Path(PAVIA_DATA) / "Pavia")
    else:
        a.data_path = str(Path(a.data_root) / DATASET_DIR[a.dataset])
    a.save_path = str(Path(a.out_root) / a.dataset)
    a.run_name = f"{a.row}_seed{a.seed}"
    a.skip_final_evaluation = True
    print(f"[ablate] {a.dataset} row={a.row} data={a.data_path} run={a.run_name}")
    trainer.train(a)


if __name__ == "__main__":
    main()
