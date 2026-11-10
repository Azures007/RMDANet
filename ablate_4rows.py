#!/usr/bin/env python3
"""Combined ablation for the clean v7.0 build.

Presets (--row):
  base      prat0 pool=gap    fusion=off      rtml0
  drse      prat0 pool=guided fusion=reliable rtml0
  rftm      prat1 pool=gap    fusion=off      rtml0
  drse_rftm prat1 pool=guided fusion=reliable rtml0
  full      prat1 pool=guided fusion=reliable rtml1

Arbitrary combination (--tag NAME --prat 0/1 --pool guided/gap
                        --fusion reliable/off --rtml 0/1)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
import model
import drse_patch  # optional runtime tuning of CenterGuidedPooling (env-gated)

drse_patch.patch(model)
import trainer

DATASET_DIR = {"pavia": "Pavia", "houston": "Houston", "hyrank": "HyRANK"}
PAVIA_DATA = "/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets"

ROWS = {
    "base":      dict(prat=0, pool="gap",    fusion="off",      rtml=0),
    "drse":      dict(prat=0, pool="guided", fusion="reliable", rtml=0),
    "rftm":      dict(prat=1, pool="gap",    fusion="off",      rtml=0),
    "drse_rftm": dict(prat=1, pool="guided", fusion="reliable", rtml=0),
    "full":      dict(prat=1, pool="guided", fusion="reliable", rtml=1),
}

_orig_apply = trainer.apply_ablation_profile


def apply_ablate(args):
    args.ablation_id = "full"
    args = _orig_apply(args)
    row = args._ablate_cfg

    args.use_prat = bool(row["prat"])

    if row["pool"] == "guided":
        args.pooling_mode = "guided"
    else:
        args.pooling_mode = "gap"

    if row["fusion"] == "reliable":
        args.spectral_fusion_mode = "reliable"
        args.use_spectral_aux = True
    else:
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
    p.add_argument("--row", choices=sorted(ROWS))
    p.add_argument("--tag", default=None, help="custom run name; requires explicit switches")
    p.add_argument("--abl-prat", type=int, choices=[0, 1])
    p.add_argument("--abl-pool", choices=["guided", "gap"])
    p.add_argument("--abl-fusion", choices=["reliable", "off"])
    p.add_argument("--abl-rtml", type=int, choices=[0, 1])
    p.add_argument("--dataset", choices=sorted(DATASET_DIR), required=True)
    p.add_argument("--data-root", default="/root/autodl-tmp/Datasets")
    p.add_argument("--out-root", default=str(ROOT / "results_ablation_clean"))
    a = p.parse_args()

    if a.tag:
        missing = [n for n, v in (("prat", a.abl_prat), ("pool", a.abl_pool),
                                  ("fusion", a.abl_fusion), ("rtml", a.abl_rtml)) if v is None]
        if missing:
            raise SystemExit(f"--tag requires explicit: {', '.join('--' + m for m in missing)}")
        cfg = dict(prat=a.abl_prat, pool=a.abl_pool, fusion=a.abl_fusion, rtml=a.abl_rtml)
        run_name = a.tag
    else:
        if not a.row:
            raise SystemExit("provide --row preset or --tag with explicit switches")
        cfg = ROWS[a.row]
        run_name = a.row

    a = trainer.load_config_into_args(a, str(ROOT / "datasets" / a.dataset / "config.yaml"))
    if a.dataset == "pavia":
        a.data_path = str(Path(PAVIA_DATA) / "Pavia")
    else:
        a.data_path = str(Path(a.data_root) / DATASET_DIR[a.dataset])
    a.save_path = str(Path(a.out_root) / a.dataset)
    a.run_name = f"{run_name}_seed{a.seed}"
    a.skip_final_evaluation = True
    a._ablate_cfg = cfg
    print(f"[ablate] {a.dataset} cfg={cfg} data={a.data_path} run={a.run_name}")
    trainer.train(a)


if __name__ == "__main__":
    main()

