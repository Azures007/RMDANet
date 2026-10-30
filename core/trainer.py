from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils import data

from data import (
    PatchDataset,
    align_target_bands,
    build_neighbor_ids,
    coordinates_from_mask,
    labels_at_coordinates,
    load_source_scene,
    load_target_ground_truth_only,
    load_target_image_only,
    select_band_subset,
    spatially_stratified_subset,
    stratified_source_split,
)
from memory import ReliabilityPrototypeMemory
from model import DualReliabilityEncoder, ModelOutput
from prat import PhaseReliabilityResidualAmplitudeTransfer
from semantic_safety import (SafetyFeedback, class_basis_from_centers, transport,
                             judge_augmentation, weighted_ce, true_margin)
from v71_controls import (reliability_lmmd, maturity_snapshot, independent_spectral_view,
                          js_agreement, classwise_cnt_shift, source_teacher_backtrack)



def lmmd_loss(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
    temperature: float = 0.10,
    min_samples: int = 4,
) -> torch.Tensor:
    """Local MMD: align class-conditional distributions."""
    sf = source_features.float()
    tf = target_features.float()
    total = sf.new_tensor(0.0)
    used = 0
    for c in range(num_classes):
        sm = source_labels.eq(c)
        tm = target_labels.eq(c)
        if int(sm.sum().item()) < min_samples or int(tm.sum().item()) < min_samples:
            continue
        s = sf[sm]
        t = tf[tm]
        s_mean = s.mean(dim=0, keepdim=True)
        t_mean = t.mean(dim=0, keepdim=True)
        total = total + (s_mean - t_mean).pow(2).sum()
        used += 1
    if used == 0:
        return sf.new_tensor(0.0)
    return total / used


def coral_loss(source_features, target_features):
    """Deep CORAL: align second-order statistics."""
    if source_features.size(0) < 2 or target_features.size(0) < 2:
        return source_features.new_tensor(0.0)
    d = source_features.size(1)
    ns = source_features.size(0)
    nt = target_features.size(0)
    src = source_features.float()
    tgt = target_features.float()
    src_c = src - src.mean(dim=0, keepdim=True)
    tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
    src_cov = (src_c.t() @ src_c) / (ns - 1)
    tgt_cov = (tgt_c.t() @ tgt_c) / (nt - 1)
    return ((src_cov - tgt_cov) ** 2).sum() / (4.0 * d * d)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone strict unsupervised dual-reliability hyperspectral domain adaptation"
    )
    parser.add_argument("--data_path", type=str, default="./Datasets/Houston/")
    parser.add_argument("--save_path", type=str, default="./results_standalone_dual_reliability/")
    parser.add_argument("--source_name", type=str, default="Houston13")
    parser.add_argument("--target_name", type=str, default="Houston18")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes (excluding background)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1233)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--patch_size", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--min_steps_per_epoch", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help=">0 forces exactly this many source update batches per epoch")
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_norm", choices=["layer", "batch"], default="batch")
    parser.add_argument("--use_gt_target_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_gt_mix_ratio", type=float, default=0.0)
    parser.add_argument("--lambda_cca", type=float, default=0.0)
    parser.add_argument("--cca_start_epoch", type=int, default=40)
    parser.add_argument("--cca_warm_epochs", type=int, default=0)
    parser.add_argument("--cca_temperature", type=float, default=0.10)
    # In strict mode the ratio is the TOTAL source-label budget, not only train.
    parser.add_argument("--strict_source_budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_budget_train_fraction", type=float, default=0.80)
    parser.add_argument("--alignment_maturity_min_samples", type=int, default=12)
    parser.add_argument("--alignment_maturity_min_classes", type=int, default=2)
    parser.add_argument("--alignment_maturity_min_value", type=float, default=0.15)
    parser.add_argument("--align_maturity_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmmd_min_samples", type=int, default=2)
    parser.add_argument("--cnt_classwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cnt_target_min_support", type=float, default=3.0)
    parser.add_argument("--cnt_target_min_confidence", type=float, default=0.65)
    parser.add_argument("--cnt_teacher_backtrack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ssf_independent_target_probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--independent_target_reweight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ssf_probe_strength", type=float, default=0.025)
    parser.add_argument("--ssf_probe_min_agreement", type=float, default=0.90)
    parser.add_argument("--source_train_ratio", type=float, default=0.95)
    parser.add_argument("--source_samples_per_class", type=int, default=0)
    parser.add_argument("--target_train_max_samples", type=int, default=100000)
    parser.add_argument(
        "--target_pool_from_gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Literature protocol: the unlabeled target training pool is the set of "
             "labeled target pixels.  The GT is used ONLY to select which pixels form "
             "the pool; labels never enter any loss.",
    )
    parser.add_argument(
        "--source_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random source-patch augmentation (the official MSDA recipe does not use it).",
    )
    parser.add_argument("--target_sampling_grid", type=int, default=30)
    parser.add_argument("--target_sampling_stratified_fraction", type=float, default=0.82)
    parser.add_argument("--neighbor_radius", type=int, default=3)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--band_alignment_strength", type=float, default=0.45)

    # v7.0 opt-in only: C0 preserves the original v6.0 training path.
    parser.add_argument("--ssf_mode", choices=["C0", "C1", "C2", "C3", "C4"], default="C0")
    parser.add_argument("--ssf_start_epoch", type=int, default=15)
    parser.add_argument("--ssf_margin_tolerance", type=float, default=0.30)
    parser.add_argument("--ssf_min_source", type=int, default=8)
    parser.add_argument("--ssf_min_target", type=int, default=20)
    parser.add_argument("--lambda_cnt", type=float, default=0.12)
    parser.add_argument("--cnt_warm_epochs", type=int, default=20)
    parser.add_argument("--cnt_max_shift", type=float, default=0.12)

    parser.add_argument("--epochs", type=int, default=280)
    parser.add_argument("--refine_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--refine_lr", type=float, default=7.5e-5)
    parser.add_argument("--weight_decay", type=float, default=4.0e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--include_target_scale_mi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_ema_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor_freeze_epoch", type=int, default=40)
    parser.add_argument("--label_smoothing", type=float, default=0.035)
    parser.add_argument("--source_class_weight_power", type=float, default=0.32)
    parser.add_argument("--source_sampler_power", type=float, default=0.30)
    parser.add_argument(
        "--balance_weight_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pavia-only: inverse-frequency source sampling",
    )
    parser.add_argument(
        "--pseudo_balance_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pavia-only: strict pseudo-label class balancing (target side)",
    )
    parser.add_argument(
        "--disable_prat_flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable PRAT only for PaviaU->PaviaC. "
            "Ignored for Houston and all other dataset pairs."
        ),
    )

    parser.add_argument(
        "--collapse_guard_flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable a PaviaU->PaviaC-only running pseudo-class collapse guard. "
            "Ignored for Houston and all other dataset pairs."
        ),
    )
    parser.add_argument(
        "--prototype_align_flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable PaviaU->PaviaC-only source-class prototype alignment using "
            "high-confidence target pseudo labels. No effect on other dataset pairs."
        ),
    )
    parser.add_argument("--prototype_weight", type=float, default=0.05)
    parser.add_argument("--prototype_confidence", type=float, default=0.95)
    parser.add_argument("--prototype_temperature", type=float, default=0.08)

    parser.add_argument("--prat_style_max", type=float, default=0.64)
    parser.add_argument("--prat_warm_epochs", type=int, default=55)
    parser.add_argument("--prat_ema_momentum", type=float, default=0.96)
    parser.add_argument("--prat_low_frequency_ratio", type=float, default=0.34)
    parser.add_argument("--prat_amplitude_temperature", type=float, default=0.80)
    parser.add_argument("--prat_energy_temperature", type=float, default=0.65)
    parser.add_argument("--prat_delta_clip", type=float, default=1.25)
    parser.add_argument("--prat_reference_mad_clip", type=float, default=3.0)
    parser.add_argument("--prat_reference_update_interval", type=int, default=4)

    parser.add_argument("--pseudo_start_epoch", type=int, default=80)
    parser.add_argument("--pseudo_warm_epochs", type=int, default=80)

    parser.add_argument("--pseudo_confidence", type=float, default=0.92)
    parser.add_argument("--pseudo_margin", type=float, default=0.10)
    parser.add_argument("--reliability_threshold", type=float, default=0.55)
    parser.add_argument("--cempa_class_cap", type=int, default=12)
    parser.add_argument("--cempa_num_subprototypes", type=int, default=4)
    parser.add_argument("--cempa_min_age", type=int, default=2)
    parser.add_argument("--cempa_min_streak", type=int, default=2)
    parser.add_argument(
        "--pseudo_min_per_class",
        type=int,
        default=0,
        help="per-class minimum accepted pseudo labels per step (0 = disable)",
    )

    parser.add_argument("--lambda_style_supervision", type=float, default=1.0)
    parser.add_argument("--lambda_spectral_aux", type=float, default=0.060)
    parser.add_argument("--lambda_source_metric", type=float, default=0.055)
    parser.add_argument("--lambda_style_pair", type=float, default=0.080)
    parser.add_argument("--lambda_scale_mi", type=float, default=0.010)
    parser.add_argument("--lambda_scale_pair", type=float, default=0.020)
    parser.add_argument("--lambda_center_pair", type=float, default=0.012)
    parser.add_argument("--lambda_pseudo", type=float, default=0.38)
    parser.add_argument("--lambda_soft_consistency", type=float, default=0.12)
    parser.add_argument("--lambda_feature_consistency", type=float, default=0.04)
    parser.add_argument("--lambda_scale_teacher", type=float, default=0.008)
    parser.add_argument("--lambda_coral", type=float, default=0.0)
    parser.add_argument("--lambda_lmmd", type=float, default=0.0)
    parser.add_argument("--dual_head_filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_dual_source", type=float, default=0.0)
    parser.add_argument("--lmmd_start_epoch", type=int, default=80)
    parser.add_argument("--lmmd_warm_epochs", type=int, default=30)
    parser.add_argument("--lambda_mpa", type=float, default=0.12)
    parser.add_argument("--lambda_confusion", type=float, default=0.040)

    parser.add_argument("--target_noise_std", type=float, default=0.004)
    parser.add_argument("--target_scale_std", type=float, default=0.015)
    parser.add_argument("--target_band_drop", type=float, default=0.006)
    parser.add_argument("--target_cutout_probability", type=float, default=0.10)

    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_topk", type=int, default=5)
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Student optimizer. 'sgd' (momentum 0.9) matches the HSI-UDA literature.",
    )
    parser.add_argument(
        "--constant_lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the base learning rate constant instead of using cosine decay "
             "(literature-aligned SGD runs use a constant lr).",
    )
    parser.add_argument(
        "--select_final_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Literature-aligned protocol: report the final epoch instead of "
             "label-free checkpoint selection (pair with an empty source validation split).",
    )
    parser.add_argument(
        "--select_mode",
        type=str,
        default="",
        choices=["", "final", "label_free"],
        help="Checkpoint selection: 'final' = last epoch, 'label_free' = pick the "
             "epoch with the best label-free reliability score (no source "
             "validation, no target labels). Empty = legacy behaviour.",
    )
    parser.add_argument("--checkpoint_average", type=int, default=1)
    parser.add_argument("--final_tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_final_evaluation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablation_id", type=str, default="full")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--diagnostic_target_oa", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--backbone",
        type=str,
        default="ours",
        choices=["ours", "dcrn"],
        help="Local feature extractor: 'ours' (v7.1 encoder) or 'dcrn' "
             "(SCLUDA/DCRN stem; only the stem is replaced).",
    )
    parser.add_argument("--amplitude_channel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--input_normalization",
        type=str,
        default="l2",
        choices=["l2", "median", "band", "scale", "raw"],
        help="Input normalization: per-pixel L2 (default), scene-median scaling, "
             "per-band z-score over valid pixels (band), or per-band z-score over "
             "all pixels (scale, literature-style).",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="symmetric",
        choices=["symmetric", "constant"],
        help="Patch padding: symmetric (ours) or constant zeros (official MSDA).",
    )
    parser.add_argument(
        "--band_subset",
        type=str,
        default="",
        help='Band subset for the literature tasks: "first:N" or "0,1,2,...".',
    )
    parser.add_argument(
        "--amplitude_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a per-pixel log-magnitude map as an extra input channel "
             "(spectral shape is kept, unlike --amplitude_channel).",
    )
    parser.add_argument("--config", type=str, default="", help="YAML config file path")
    parser.add_argument("--use_early_target_adaptation", action=argparse.BooleanOptionalAction, default=False, help="Memory update from epoch 1 (was hardcoded for Pavia)")
    return parser



ABLATION_DESCRIPTIONS: dict[str, str] = {
    "main_base": "Source-only stable spectral-spatial encoder",
    "main_prat": "Source supervision plus PRAT, without target self-training",
    "main_ema_st": "PRAT plus current-epoch EMA pseudo-label self-training",
    "main_reliability": "PRAT plus cross-epoch unified reliability, without prototypes/guard/spatial",
    "main_cempa": "PRAT plus unified reliability and multi-prototype CEMPA, without guard/spatial",
    "full": "Complete dual-reliability model",
    "prat_off_fulladapt": "Full high-level adaptation without PRAT",
    "prat_fixed": "Fixed low-frequency residual amplitude transfer",
    "prat_phase": "Phase-coherence gated residual transfer",
    "prat_phase_energy": "Phase and energy gated residual transfer",
    "prat_replace": "Hard low-frequency amplitude replacement",
    "rel_no_prob_history": "Full model without probability-history labels",
    "rel_no_time": "Full model without age/streak reliability",
    "rel_no_feature": "Full model without feature-drift reliability",
    "rel_no_view": "Full model without teacher/student/strong-view agreement",
    "rel_no_guard": "Full model without frozen semantic guard",
    "rel_no_proto_veto": "Full model without source-prototype reliability/veto",
    "rel_no_spatial": "Full model without spatial reliability/veto/loss",
    "proto_source_only": "CEMPA with source anchors only",
    "proto_single_fixed": "One target prototype per class, fixed update and fixed anchor weight",
    "proto_multi_unweighted": "Three target subprototypes, unweighted update and fixed anchor weight",
    "proto_multi_weighted": "Three target subprototypes, reliability-weighted update and fixed anchor weight",
    "proto_adaptive": "Three target subprototypes with reliability update and adaptive anchors",
    "proto_full3": "Three target subprototypes with adaptive anchors and hard-negative confusion loss",
    "aspp_single": "Single standard 3x3 context branch",
    "aspp_equal": "Four MicroASPP branches with equal weights",
    "pool_gap": "Global average pooling",
    "pool_center": "Center 3x3 pooling",
    "spectral_off": "Context branch only, without spectral auxiliary fusion",
    "spectral_average": "Fixed average of context and spectral representations",
}


def load_config_into_args(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    """Load YAML config and override args attributes."""
    import yaml
    from pathlib import Path as _Path
    cfg_path = _Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got {type(cfg)}")
    for section_name, section in cfg.items():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            setattr(args, key, value)
    print(f"[config] loaded {config_path}")
    return args


def apply_ablation_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.ablation_id not in ABLATION_DESCRIPTIONS:
        available = ", ".join(sorted(ABLATION_DESCRIPTIONS))
        raise ValueError(f"Unknown ablation_id={args.ablation_id!r}. Available: {available}")

    # Complete-model defaults. Profiles below only change the studied mechanism.
    args.use_prat = True
    args.prat_mode = "full"
    args.pseudo_mode = "history"  # off | current | history
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
    args.use_spectral_aux = True
    args.aspp_mode = "adaptive"
    args.pooling_mode = "guided"
    args.spectral_fusion_mode = "reliable"
    args.fixed_target_anchor_weight = 0.50

    profile = args.ablation_id
    if profile == "main_base":
        args.use_prat = False
        args.pseudo_mode = "off"
        args.use_guard = False
        args.use_prototype_veto = False
        args.use_spatial_reliability = False
        args.use_target_prototypes = False
        args.use_confusion_loss = False
    elif profile == "main_prat":
        args.pseudo_mode = "off"
        args.use_guard = False
        args.use_prototype_veto = False
        args.use_spatial_reliability = False
        args.use_target_prototypes = False
        args.use_confusion_loss = False
    elif profile == "main_ema_st":
        args.pseudo_mode = "current"
        args.use_guard = False
        args.use_prototype_veto = False
        args.use_spatial_reliability = False
        args.use_target_prototypes = False
        args.use_confusion_loss = False
    elif profile == "main_reliability":
        args.use_guard = False
        args.use_prototype_veto = False
        args.use_spatial_reliability = False
        args.use_target_prototypes = False
        args.use_confusion_loss = False
    elif profile == "main_cempa":
        args.use_guard = False
        args.use_spatial_reliability = False
    elif profile == "prat_off_fulladapt":
        args.use_prat = False
    elif profile in {"prat_fixed", "prat_phase", "prat_phase_energy", "prat_replace"}:
        args.prat_mode = profile.removeprefix("prat_")
    elif profile == "rel_no_prob_history":
        args.use_probability_history = False
    elif profile == "rel_no_time":
        args.use_time_score = False
    elif profile == "rel_no_feature":
        args.use_feature_score = False
    elif profile == "rel_no_view":
        args.use_view_score = False
    elif profile == "rel_no_guard":
        args.use_guard = False
    elif profile == "rel_no_proto_veto":
        args.use_prototype_veto = False
    elif profile == "rel_no_spatial":
        args.use_spatial_reliability = False
    elif profile == "proto_source_only":
        args.use_target_prototypes = False
        args.use_confusion_loss = False
    elif profile == "proto_single_fixed":
        args.cempa_num_subprototypes = 1
        args.reliability_weighted_prototype_update = False
        args.adaptive_anchor = False
        args.use_confusion_loss = False
    elif profile == "proto_multi_unweighted":
        args.cempa_num_subprototypes = 3
        args.reliability_weighted_prototype_update = False
        args.adaptive_anchor = False
        args.use_confusion_loss = False
    elif profile == "proto_multi_weighted":
        args.cempa_num_subprototypes = 3
        args.reliability_weighted_prototype_update = True
        args.adaptive_anchor = False
        args.use_confusion_loss = False
    elif profile == "proto_adaptive":
        args.cempa_num_subprototypes = 3
        args.reliability_weighted_prototype_update = True
        args.adaptive_anchor = True
        args.use_confusion_loss = False
    elif profile == "proto_full3":
        args.cempa_num_subprototypes = 3
    elif profile == "aspp_single":
        args.aspp_mode = "single"
    elif profile == "aspp_equal":
        args.aspp_mode = "equal"
    elif profile == "pool_gap":
        args.pooling_mode = "gap"
    elif profile == "pool_center":
        args.pooling_mode = "center"
    elif profile == "spectral_off":
        args.spectral_fusion_mode = "off"
        args.use_spectral_aux = False
    elif profile == "spectral_average":
        args.spectral_fusion_mode = "average"

    if bool(getattr(args, "disable_prat_flag", False)):
        args.use_prat = False

    return args


def zero_prat_metadata(source: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = source.size(0)
    zeros = source.new_zeros(batch)
    return {
        "frequency_reliability": source.new_zeros(source.shape),
        "sample_reliability": zeros,
        "sample_transfer": zeros,
        "mean_transfer": source.new_tensor(0.0),
    }

def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cosine_ramp(epoch: int, start: int, warm_epochs: int) -> float:
    if epoch < start:
        return 0.0
    if warm_epochs <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (epoch - start + 1) / warm_epochs))
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def cosine_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    total_epochs: int,
    base_lr: float,
    minimum_lr: float = 1.0e-6,
    warmup_epochs: int = 5,
) -> float:
    if epoch_index < warmup_epochs:
        lr = base_lr * float(epoch_index + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch_index - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        lr = minimum_lr + 0.5 * (base_lr - minimum_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    with torch.no_grad():
        student_state = dict(student.named_parameters())
        for name, teacher_parameter in teacher.named_parameters():
            teacher_parameter.data.mul_(decay).add_(student_state[name].data, alpha=1.0 - decay)
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher.named_buffers():
            source = student_buffers[name]
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(decay).add_(source.data, alpha=1.0 - decay)
            else:
                teacher_buffer.data.copy_(source.data)


def strong_target_augmentation(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    augmented = x.clone()
    batch, bands, height, width = augmented.shape
    horizontal = torch.rand(batch, device=x.device) < 0.5
    vertical = torch.rand(batch, device=x.device) < 0.5
    if horizontal.any():
        augmented[horizontal] = torch.flip(augmented[horizontal], dims=(-1,))
    if vertical.any():
        augmented[vertical] = torch.flip(augmented[vertical], dims=(-2,))

    scale = 1.0 + float(args.target_scale_std) * torch.randn(
        batch, bands, 1, 1, device=x.device
    )
    scale = scale.view(batch, 1, bands)
    scale = F.avg_pool1d(scale, kernel_size=5, stride=1, padding=2)
    scale = scale.view(batch, bands, 1, 1)
    augmented = augmented * scale
    augmented = augmented + float(args.target_noise_std) * torch.randn_like(augmented)

    drop_probability = float(args.target_band_drop)
    if drop_probability > 0:
        drop_mask = torch.rand(batch, bands, 1, 1, device=x.device) < drop_probability
        # Preserve at least one band for every sample.
        all_dropped = drop_mask.flatten(1).all(dim=1)
        if all_dropped.any():
            drop_mask[all_dropped, bands // 2] = False
        augmented = augmented.masked_fill(drop_mask, 0.0)

    if float(args.target_cutout_probability) > 0 and height >= 9 and width >= 9:
        size = max(3, min(height, width) // 5)
        flags = np.random.random(batch) < float(args.target_cutout_probability)
        rows = np.random.randint(0, max(1, height - size + 1), size=batch)
        cols = np.random.randint(0, max(1, width - size + 1), size=batch)
        center = height // 2
        for index in np.nonzero(flags)[0].tolist():
            row = int(rows[index])
            col = int(cols[index])
            # Do not erase the center label location.
            if row <= center < row + size and col <= center < col + size:
                row = 0
                col = 0
            augmented[index, :, row : row + size, col : col + size] = 0.0
    return augmented


def scale_mutual_information_loss(scale_weights: torch.Tensor) -> torch.Tensor:
    weights = scale_weights.float().clamp_min(1e-8)
    sample_entropy = -(weights * weights.log()).sum(dim=1).mean()
    mean_weights = weights.mean(dim=0)
    batch_entropy = -(mean_weights * mean_weights.log()).sum()
    return sample_entropy - batch_entropy


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.detach().float().clamp_min(1e-6)
    return torch.sum(values.float() * weights) / weights.sum().clamp_min(1e-6)


def style_pair_loss(
    source: ModelOutput,
    styled: ModelOutput,
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    source_probability = F.softmax(source.logits.float(), dim=1).clamp_min(1e-8)
    styled_probability = F.softmax(styled.logits.float(), dim=1).clamp_min(1e-8)
    symmetric_kl = 0.5 * (
        torch.sum(source_probability * (source_probability.log() - styled_probability.log()), dim=1)
        + torch.sum(styled_probability * (styled_probability.log() - source_probability.log()), dim=1)
    )
    feature_distance = 1.0 - torch.sum(source.features.float() * styled.features.float(), dim=1)
    scale_distance = torch.mean(
        (source.scale_weights.float() - styled.scale_weights.float()).pow(2), dim=1
    )
    center_distance = torch.mean(
        (source.center_weights.float() - styled.center_weights.float()).pow(2), dim=1
    )
    total = weighted_mean(
        symmetric_kl + 0.60 * feature_distance + 0.25 * scale_distance + 0.15 * center_distance,
        sample_weights,
    )
    return total, {
        "style_kl": weighted_mean(symmetric_kl, sample_weights),
        "style_feature": weighted_mean(feature_distance, sample_weights),
        "style_scale": weighted_mean(scale_distance, sample_weights),
        "style_center": weighted_mean(center_distance, sample_weights),
    }


def teacher_consistency_losses(
    strong_output: ModelOutput,
    teacher_output: ModelOutput,
    reliability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    teacher_probability = F.softmax(teacher_output.logits.detach().float(), dim=1).clamp_min(1e-8)
    log_student = F.log_softmax(strong_output.logits.float(), dim=1)
    soft_ce = -(teacher_probability * log_student).sum(dim=1)
    feature_distance = 1.0 - torch.sum(
        strong_output.features.float() * teacher_output.features.detach().float(), dim=1
    )
    scale_distance = torch.mean(
        (
            strong_output.scale_weights.float()
            - teacher_output.scale_weights.detach().float()
        ).pow(2),
        dim=1,
    )
    weights = reliability.detach().pow(2).clamp_min(1e-5)
    return {
        "soft": weighted_mean(soft_ce, weights),
        "feature": weighted_mean(feature_distance, weights),
        "scale": weighted_mean(scale_distance, weights),
    }


def source_metric_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    valid_samples = valid[labels]
    if not valid_samples.any():
        return features.new_tensor(0.0)
    features = F.normalize(features[valid_samples].float(), dim=1)
    labels = labels[valid_samples]
    normalized_prototypes = F.normalize(prototypes.detach().float(), dim=1)
    similarities = features @ normalized_prototypes.t()
    positive = similarities.gather(1, labels[:, None]).squeeze(1)
    negatives = similarities.clone()
    negatives[:, ~valid] = -1e4
    negatives.scatter_(1, labels[:, None], -1e4)
    hard_negative = negatives.max(dim=1).values
    return F.softplus(hard_negative - positive + margin).mean()


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def weighted_average_state_dicts(
    records: list[dict[str, Any]],
    maximum: int,
) -> dict[str, torch.Tensor]:
    selected = sorted(records, key=lambda item: item["score"], reverse=True)[: max(1, maximum)]
    scores = torch.tensor([record["score"] for record in selected], dtype=torch.float64)
    weights = torch.softmax((scores - scores.max()) / 0.25, dim=0)
    averaged: dict[str, torch.Tensor] = {}
    reference = selected[0]["state"]
    for name, tensor in reference.items():
        if tensor.dtype.is_floating_point:
            accumulator = torch.zeros_like(tensor, dtype=torch.float64)
            for weight, record in zip(weights, selected):
                accumulator.add_(record["state"][name].to(torch.float64), alpha=float(weight.item()))
            averaged[name] = accumulator.to(tensor.dtype)
        else:
            averaged[name] = tensor.clone()
    return averaged


def confusion_matrix_numpy(target: np.ndarray, prediction: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(target.tolist(), prediction.tolist()):
        if 0 <= true_label < num_classes and 0 <= predicted_label < num_classes:
            matrix[true_label, predicted_label] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    oa = 100.0 * correct / max(1, total)
    row_sum = matrix.sum(axis=1)
    class_accuracy = np.divide(
        np.diag(matrix), row_sum, out=np.zeros_like(row_sum, dtype=np.float64), where=row_sum > 0
    )
    aa = 100.0 * float(class_accuracy.mean())
    col_sum = matrix.sum(axis=0)
    expected = float(np.dot(row_sum, col_sum)) / max(1.0, float(total * total))
    observed = float(correct) / max(1.0, float(total))
    kappa = (observed - expected) / max(1e-12, 1.0 - expected)
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "class_accuracy": (100.0 * class_accuracy).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


@torch.no_grad()
def target_consistency_score(
    model: nn.Module,
    target_fit_loader: data.DataLoader,
    device: torch.device,
) -> float:
    """无标签目标域多视图预测一致性。

    对每个目标样本，取原始、水平翻转、垂直翻转三种视图，
    统计三个视图 argmax 一致的比例。不读 GT，不用作训练信号。
    """
    model.eval()
    total_consistent = 0
    total_samples = 0
    for x, _ in target_fit_loader:
        x = x.to(device, non_blocking=True)
        p1 = F.softmax(model(x).logits.float(), dim=1).argmax(dim=1)
        p2 = F.softmax(model(torch.flip(x, dims=(-1,))).logits.float(), dim=1).argmax(dim=1)
        p3 = F.softmax(model(torch.flip(x, dims=(-2,))).logits.float(), dim=1).argmax(dim=1)
        consistent = p1.eq(p2) & p1.eq(p3)
        total_consistent += int(consistent.sum().item())
        total_samples += int(x.size(0))
    return total_consistent / max(1, total_samples)


def label_free_selection_score(
    source_accuracy: float,
    diagnostics: dict[str, float],
    selected_fraction: float,
    target_consistency: float = 0.0,
) -> float:
    """Non-monotonic checkpoint score that does not reward memory age itself."""
    reliability = min(float(diagnostics["reliability"]), 0.62)
    entropy = float(diagnostics["entropy"])
    entropy_quality = math.exp(-((entropy - 0.30) / 0.18) ** 2)
    selected_quality = math.exp(-((float(selected_fraction) - 0.34) / 0.24) ** 2)
    over_selection = max(0.0, float(selected_fraction) - 0.62)
    source_penalty = 1.0 * max(0.0, 98.0 - float(source_accuracy))
    return (
        float(source_accuracy)
        + 7.0 * reliability
        + 2.5 * float(diagnostics["coverage"])
        + 2.5 * float(diagnostics["prototype_coverage"])
        + 2.0 * float(diagnostics["prior_stability"])
        + 2.0 * entropy_quality
        + 3.0 * selected_quality
        + 6.0 * float(target_consistency)
        - 12.0 * over_selection
        - source_penalty
    )


def reliable_output_probability(output: ModelOutput) -> torch.Tensor:
    """Conservative posterior fusion for training and inference.

    The main classifier remains dominant.  Auxiliary context and center-spectrum
    heads can only contribute when their top class agrees with the main head;
    disagreement therefore cannot sharpen an incorrect pseudo label.
    """
    main = F.softmax(output.logits.float(), dim=1).clamp_min(1e-8)
    context = F.softmax(output.context_logits.float(), dim=1).clamp_min(1e-8)
    spectral = F.softmax(output.spectral_logits.float(), dim=1).clamp_min(1e-8)
    main_label = main.argmax(dim=1)
    context_agreement = main_label.eq(context.argmax(dim=1)).float()
    spectral_agreement = main_label.eq(spectral.argmax(dim=1)).float()
    context_weight = 0.12 * context_agreement
    spectral_weight = (
        0.035
        * context_agreement
        * spectral_agreement
        * output.spectral_reliability.detach().float().clamp(0.0, 1.0)
    )
    probability = (
        (1.0 - context_weight[:, None] - spectral_weight[:, None]) * main
        + context_weight[:, None] * context
        + spectral_weight[:, None] * spectral
    )
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)



@torch.no_grad()
def predict_probabilities(model: nn.Module, x: torch.Tensor, use_tta: bool) -> torch.Tensor:
    if not use_tta:
        return reliable_output_probability(model(x))
    views = (x, torch.flip(x, (-1,)), torch.flip(x, (-2,)), torch.flip(x, (-2, -1)))
    probabilities = [reliable_output_probability(model(view)) for view in views]
    probability = torch.stack(probabilities, dim=0).mean(dim=0)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def evaluate_source(model: nn.Module, loader: data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        prediction = predict_probabilities(model, x, use_tta=False).argmax(dim=1)
        correct += int(prediction.eq(labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_target(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    num_classes: int,
    use_tta: bool,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        probabilities = predict_probabilities(model, x, use_tta=use_tta)
        predictions.append(probabilities.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    prediction = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
    matrix = confusion_matrix_numpy(target, prediction, num_classes)
    return classification_metrics(matrix)


def build_loaders(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    normalize_mode = str(
        getattr(args, "input_normalization", "")
        or ("median" if getattr(args, "amplitude_channel", False) else "l2")
    )
    source_scene = load_source_scene(
        args.data_path, args.source_name, args.num_classes, normalize_mode=normalize_mode
    )
    target_scene = load_target_image_only(
        args.data_path, args.target_name, normalize_mode=normalize_mode
    )
    if source_scene.image.shape[-1] != target_scene.image.shape[-1]:
        raise ValueError("Source and target scenes have different band counts.")
    band_subset = str(getattr(args, "band_subset", "") or "")
    if band_subset:
        source_bands = select_band_subset(source_scene.image, band_subset)
        target_bands = select_band_subset(target_scene.image, band_subset)
        print(f"[bands] subset {band_subset!r} -> {source_bands.shape[-1]} bands")
        if getattr(args, "amplitude_input", False):
            raise ValueError("band_subset is incompatible with amplitude_input")
    else:
        source_bands = source_scene.image
        target_bands = target_scene.image
    target_image, alignment = align_target_bands(
        source_bands,
        target_bands,
        strength=args.band_alignment_strength,
        # Only the per-pixel L2 pipeline re-normalizes after alignment.
        renormalize=normalize_mode == "l2",
    )
    if getattr(args, "amplitude_input", False):
        if source_scene.amplitude is None or target_scene.amplitude is None:
            raise ValueError("amplitude_input requires per-scene amplitude maps")
        # Trailing channel: standardized log spectral magnitude (label-free).
        source_image = np.concatenate(
            [source_scene.image, source_scene.amplitude[..., None]], axis=-1
        )
        target_image = np.concatenate(
            [target_image, target_scene.amplitude[..., None]], axis=-1
        )
    else:
        source_image = source_bands

    if getattr(args, "strict_source_budget", True):
        from data import budgeted_source_split
        source_train_gt, source_validation_gt = budgeted_source_split(
            source_scene.ground_truth,
            total_budget_ratio=float(args.source_train_ratio),
            seed=args.seed + 11,
            samples_per_class=int(getattr(args, "source_samples_per_class", 0)),
            train_fraction=float(args.source_budget_train_fraction),
        )
        eligible = int((source_train_gt > 0).sum() + (source_validation_gt > 0).sum())
        print(f"[v7.1 source-budget] accessible GT={eligible} "
              f"train={(source_train_gt > 0).sum()} val={(source_validation_gt > 0).sum()} "
              "(all other source labels excluded from training and selection)")
    else:
        source_train_gt, source_validation_gt = stratified_source_split(
            source_scene.ground_truth,
            train_ratio=args.source_train_ratio,
            seed=args.seed + 11,
            samples_per_class=int(getattr(args, "source_samples_per_class", 0)),
        )
    source_train_coordinates = coordinates_from_mask(source_train_gt > 0)
    source_validation_coordinates = coordinates_from_mask(source_validation_gt > 0)
    source_train_labels = labels_at_coordinates(source_train_gt, source_train_coordinates) - 1
    source_validation_labels = labels_at_coordinates(
        source_validation_gt, source_validation_coordinates
    ) - 1
    # num_classes is the number of valid classes after subtracting background (0)
    # Labels are 1-N in ground truth, 0-(N-1) after -1, so max of ground truth gives N
    num_classes = int(source_scene.ground_truth.max())
    bands = int(source_image.shape[-1])

    target_mask_full = np.linalg.norm(target_image, axis=-1) > 0
    if getattr(args, "use_gt_target_mask", False) or float(getattr(args, "target_gt_mix_ratio", 0.0)) != 0.0:
        raise RuntimeError("v7.1 strict UDA: target-GT training masks are forbidden")
    target_coordinates_all = coordinates_from_mask(target_mask_full)
    if bool(getattr(args, "target_pool_from_gt", False)):
        # Literature protocol: the unlabeled pool is the set of labeled target
        # pixels.  The GT is read ONLY to select which pixels belong to the pool;
        # no label is ever used in a loss (verified by the locked final eval).
        pool_gt = load_target_ground_truth_only(
            args.data_path, args.target_name, args.num_classes
        )
        pool_mask = (pool_gt > 0) & target_mask_full
        target_coordinates = coordinates_from_mask(pool_mask)
        print(
            f"[TARGET-MASK] literature protocol: labeled-pixel pool "
            f"{len(target_coordinates)} (GT used for pool selection only)"
        )
    else:
        subset = spatially_stratified_subset(
            target_coordinates_all,
            maximum=args.target_train_max_samples,
            seed=args.seed + 17,
            grid_size=args.target_sampling_grid,
            stratified_fraction=args.target_sampling_stratified_fraction,
        )
        target_coordinates = target_coordinates_all[subset]
        print(f"[TARGET-MASK] image-derived unlabeled scene: {len(target_coordinates)}")
    neighbor_ids = build_neighbor_ids(
        target_coordinates,
        radius=args.neighbor_radius,
        max_neighbors=args.max_neighbors,
    )

    patch_padding = str(getattr(args, "patch_padding", "symmetric") or "symmetric")
    source_train_dataset = PatchDataset(
        source_image,
        source_train_coordinates,
        patch_size=args.patch_size,
        labels=source_train_labels,
        augment=bool(getattr(args, "source_augment", True)),
        padding=patch_padding,
    )
    source_validation_dataset = (
        PatchDataset(
            source_image,
            source_validation_coordinates,
            patch_size=args.patch_size,
            labels=source_validation_labels,
            augment=False,
            padding=patch_padding,
        )
        if len(source_validation_coordinates) > 0
        else None
    )
    target_dataset = PatchDataset(
        target_image,
        target_coordinates,
        patch_size=args.patch_size,
        labels=None,
        augment=False,
        padding=patch_padding,
    )

    counts = np.bincount(source_train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = (counts.mean() / counts) ** float(args.source_class_weight_power)
    class_weights = class_weights / class_weights.mean()
    source_sampler_power = float(args.source_sampler_power)
    sample_weights = (counts.mean() / counts[source_train_labels]) ** source_sampler_power

    source_generator = torch.Generator().manual_seed(args.seed + 101)
    target_generator = torch.Generator().manual_seed(args.seed + 202)
    sampler_generator = torch.Generator().manual_seed(args.seed + 303)
    source_steps = (int(args.steps_per_epoch) if int(args.steps_per_epoch) > 0
                    else max(int(args.min_steps_per_epoch),
                             len(source_train_dataset) // args.batch_size))
    if source_steps < 1:
        raise ValueError("source_steps must be >=1")
    # Replacement does not create new labels; it prevents zero-step epochs.
    source_sampler = data.WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=source_steps * args.batch_size,
        replacement=True,
        generator=sampler_generator,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    source_loader = data.DataLoader(
        source_train_dataset,
        batch_size=args.batch_size,
        sampler=source_sampler,
        drop_last=True,
        generator=source_generator,
        **loader_kwargs,
    )
    target_loader = data.DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(target_dataset) >= args.batch_size,
        generator=target_generator,
        **loader_kwargs,
    )
    eval_kwargs = {
        "num_workers": min(args.num_workers, 4),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    source_validation_loader = (
        data.DataLoader(
            source_validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            **eval_kwargs,
        )
        if source_validation_dataset is not None
        else None
    )
    target_fit_loader = data.DataLoader(
        target_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **eval_kwargs,
    )
    return {
        "source_loader": source_loader,
        "target_loader": target_loader,
        "target_fit_loader": target_fit_loader,
        "source_validation_loader": source_validation_loader,
        "target_image": target_image,
        "target_coordinates": target_coordinates,
        "neighbor_ids": torch.as_tensor(neighbor_ids, dtype=torch.long, device=device),
        "num_classes": num_classes,
        "bands": bands,
        "class_weights": torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        "alignment": alignment,
        "source_samples": len(source_train_dataset),
        "target_samples": len(target_dataset),
    }


def build_target_evaluation_loader(
    args: argparse.Namespace,
    target_image: np.ndarray,
) -> data.DataLoader:
    ground_truth = load_target_ground_truth_only(args.data_path, args.target_name, args.num_classes)
    if ground_truth.shape != target_image.shape[:2]:
        raise ValueError(
            f"Target image and ground-truth shapes differ: {target_image.shape[:2]} vs {ground_truth.shape}."
        )
    valid = (ground_truth > 0) & (np.linalg.norm(target_image, axis=-1) > 0)
    coordinates = coordinates_from_mask(valid)
    labels = labels_at_coordinates(ground_truth, coordinates) - 1
    patch_padding = str(getattr(args, "patch_padding", "symmetric") or "symmetric")
    dataset = PatchDataset(
        target_image,
        coordinates,
        patch_size=args.patch_size,
        labels=labels,
        augment=False,
        padding=patch_padding,
    )
    return data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 4),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )




def prototype_align_flagment_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "prototype_align_flag", False))


def run_epoch(
    epoch: int,
    total_epoch_index: int,
    total_schedule_epochs: int,
    args: argparse.Namespace,
    student: DualReliabilityEncoder,
    teacher: DualReliabilityEncoder,
    guard_model: DualReliabilityEncoder | None,
    prat: PhaseReliabilityResidualAmplitudeTransfer,
    memory: ReliabilityPrototypeMemory,
    source_loader: data.DataLoader,
    target_loader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    class_weights: torch.Tensor,
    device: torch.device,
    refinement: bool,
    feedback: SafetyFeedback | None = None,
    class_basis: torch.Tensor | None = None,
    source_spectral_means: torch.Tensor | None = None,
    alignment_state: dict | None = None,
) -> dict[str, float]:
    if not hasattr(args, "pseudo_mode"):
        if not hasattr(args, "ablation_id"):
            args.ablation_id = "full"
        args = apply_ablation_profile(args)
    student.train()
    teacher.eval()
    ssf_mode = str(getattr(args, "ssf_mode", "C0"))
    cnt_enabled = ssf_mode in ("C1", "C2", "C3")
    safety_enabled = ssf_mode in ("C2", "C3", "C4") and epoch >= int(args.ssf_start_epoch)
    feedback_enabled = ssf_mode in ("C3", "C4")
    lr = cosine_learning_rate(
        optimizer,
        total_epoch_index,
        total_schedule_epochs,
        args.refine_lr if refinement else args.lr,
        minimum_lr=1.0e-6,
        warmup_epochs=2 if refinement else 5,
    )
    if bool(getattr(args, "constant_lr", False)):
        lr = float(args.refine_lr if refinement else args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
    target_iterator = iter(target_loader)
    accumulator: dict[str, float] = {}
    batches = 0

    if bool(getattr(args, "use_early_target_adaptation", False)):
        memory_update_start = 1
    else:
        memory_update_start = int(getattr(args, "memory_update_start_epoch", args.pseudo_start_epoch))
    target_adaptation = (
        args.pseudo_mode != "off"
        and (refinement or epoch >= memory_update_start)
    )
    pseudo_ramp = (
        0.0 if not target_adaptation else
        (1.0 if refinement else cosine_ramp(epoch, args.pseudo_start_epoch, args.pseudo_warm_epochs))
    )
    snapshot = maturity_snapshot(memory,
        min_samples=int(args.alignment_maturity_min_samples),
        min_classes=int(args.alignment_maturity_min_classes))
    # Do not align until unlabeled predictions show nontrivial maturity.
    if alignment_state is not None and snapshot['maturity'] >= float(args.alignment_maturity_min_value):
        alignment_state['active_epochs'] = int(alignment_state.get('active_epochs', 0)) + 1
        if alignment_state.get('first_epoch') is None:
            alignment_state['first_epoch'] = epoch
    alignment_active_epochs = int(alignment_state.get('active_epochs', 0)) if alignment_state else 0
    maturity_factor = (min(1.0, snapshot['maturity'] / max(1e-6, args.alignment_maturity_min_value))
        if args.align_maturity_gate else 1.0)
    style_strength = (
        args.prat_style_max * cosine_ramp(epoch, 1, args.prat_warm_epochs)
        if args.use_prat else 0.0
    )

    for source_x, source_y in source_loader:
        try:
            target_x, target_ids = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_x, target_ids = next(target_iterator)

        source_x = source_x.to(device, non_blocking=True)
        source_y = source_y.to(device, non_blocking=True)
        target_x = target_x.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_prat:
                if bool(getattr(args, "amplitude_input", False)):
                    # PRAT transfers spectral style; the amplitude channel is
                    # not part of the spectrum and is re-attached afterwards.
                    styled_spectral, prat_metadata = prat(
                        source_x[:, :-1], target_x[:, :-1], style_strength
                    )
                    styled_source = torch.cat(
                        [styled_spectral, source_x[:, -1:]], dim=1
                    )
                else:
                    styled_source, prat_metadata = prat(source_x, target_x, style_strength)
                if feedback_enabled and feedback is not None:
                    class_scale = feedback.scale_prat[source_y].view(-1, 1, 1, 1)
                    styled_source = source_x + (styled_source - source_x) * class_scale
            else:
                styled_source = source_x
                prat_metadata = zero_prat_metadata(source_x)

            teacher_target = None
            teacher_probability = None
            guard_probability = None
            strong_target = None
            if target_adaptation:
                strong_target = strong_target_augmentation(target_x, args)
                with autocast_context(device, args.amp):
                    teacher_target = teacher(target_x)
                teacher_probability = reliable_output_probability(teacher_target)
                if args.use_prototype_veto or args.use_target_prototypes:
                    teacher_probability = memory.refine_probabilities(
                        teacher_probability, teacher_target.features
                    )
                if args.use_guard and guard_model is not None:
                    with autocast_context(device, args.amp):
                        guard_target = guard_model(target_x)
                    guard_probability = reliable_output_probability(guard_target)

        # v2.4 CNT and semantic safety: source-GT only; target has no labels here.
        # All teacher safety decisions are detached. C0 leaves v6.0 unchanged.
        with torch.no_grad():
            prat_safe = torch.ones_like(source_y, dtype=torch.bool)
            cnt_safe = torch.ones_like(source_y, dtype=torch.bool)
            cnt_source = None
            cnt_shift = 0.0
            class_scales = (feedback.scale_cnt if feedback is not None
                            else torch.ones(args.num_classes, device=device))
            if cnt_enabled:
                if class_basis is None:
                    raise RuntimeError("CNT requires a source-only class basis")
                if args.cnt_classwise:
                    if source_spectral_means is None:
                        raise RuntimeError("classwise CNT needs budgeted source class means")
                    if teacher_target is None:
                        with autocast_context(device, args.amp):
                            cnt_target_output = teacher(target_x)
                        target_p = F.softmax(cnt_target_output.logits.float(), dim=1)
                    else:
                        target_p = F.softmax(teacher_target.logits.float(), dim=1)
                    cnt_source, cnt_shift, _ = classwise_cnt_shift(
                        source_x, target_x, source_y, target_p,
                        source_spectral_means, class_basis, class_scales,
                        max_ratio=float(args.cnt_max_shift),
                        min_support=float(args.cnt_target_min_support),
                        target_confidence=float(args.cnt_target_min_confidence))
                else:
                    cnt_source, cnt_shift = transport(
                        source_x, target_x, class_basis, class_scales, source_y,
                        max_ratio=float(args.cnt_max_shift))
                if args.cnt_teacher_backtrack and cnt_shift > 1e-8 and epoch >= int(args.ssf_start_epoch):
                    # A teacher-dependent practical guard, not a correctness proof.
                    with torch.enable_grad():
                        cnt_source, cnt_backtrack_safe = source_teacher_backtrack(
                            teacher, source_x, cnt_source, source_y,
                            tolerance=float(args.ssf_margin_tolerance))
                    cnt_safe &= cnt_backtrack_safe
            if safety_enabled:
                with autocast_context(device, args.amp):
                    teacher_clean_logits = teacher(source_x).logits.detach()
                    if args.use_prat:
                        teacher_prat_logits = teacher(styled_source).logits.detach()
                        prat_safe, _ = judge_augmentation(
                            teacher_clean_logits, teacher_prat_logits, source_y,
                            tolerance=float(args.ssf_margin_tolerance),
                        )
                    if cnt_enabled:
                        teacher_cnt_logits = teacher(cnt_source).logits.detach()
                        cnt_judged, _ = judge_augmentation(
                            teacher_clean_logits, teacher_cnt_logits, source_y,
                            tolerance=float(args.ssf_margin_tolerance),
                        )
                        cnt_safe &= cnt_judged
                if feedback_enabled and feedback is not None:
                    if args.use_prat:
                        feedback.observe_source(source_y, prat_safe, branch="prat")
                    if cnt_enabled and cnt_shift > 1e-8:
                        feedback.observe_source(source_y, cnt_safe, branch="cnt")

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            source_output = student(source_x)
            styled_output = student(styled_source) if args.use_prat else source_output
            cnt_output = student(cnt_source) if cnt_enabled and cnt_shift > 1e-8 else None
            target_output = student(target_x) if target_adaptation else None
            strong_target_output = student(strong_target) if target_adaptation else None

            source_loss = F.cross_entropy(
                source_output.logits,
                source_y,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            if getattr(source_output, "logits2", None) is not None:
                source_loss2 = F.cross_entropy(
                    source_output.logits2, source_y,
                    weight=class_weights,
                    label_smoothing=args.label_smoothing,
                )
            else:
                source_loss2 = source_loss.new_tensor(0.0)
            styled_supervision = (
                weighted_ce(styled_output.logits, source_y, prat_safe.float(),
                            label_smoothing=args.label_smoothing, class_weights=class_weights)
                if args.use_prat and safety_enabled else
                F.cross_entropy(
                    styled_output.logits,
                    source_y,
                    weight=class_weights,
                    label_smoothing=args.label_smoothing,
                ) if args.use_prat else source_loss.new_tensor(0.0)
            )
            if args.use_spectral_aux:
                spectral_terms = [
                    F.cross_entropy(source_output.context_logits, source_y, weight=class_weights),
                    F.cross_entropy(source_output.spectral_logits, source_y, weight=class_weights),
                ]
                if args.use_prat:
                    if safety_enabled and bool(prat_safe.any().item()):
                        spectral_terms.extend([
                            weighted_ce(styled_output.context_logits, source_y, prat_safe.float(),
                                        class_weights=class_weights),
                            weighted_ce(styled_output.spectral_logits, source_y, prat_safe.float(),
                                        class_weights=class_weights),
                        ])
                    elif not safety_enabled:
                        spectral_terms.extend([
                            F.cross_entropy(styled_output.context_logits, source_y, weight=class_weights),
                            F.cross_entropy(styled_output.spectral_logits, source_y, weight=class_weights),
                        ])
                spectral_aux = torch.stack(spectral_terms).mean()
            else:
                spectral_aux = source_loss.new_tensor(0.0)

            if args.use_prat:
                transfer_weight = (
                    prat_metadata["sample_reliability"].to(device)
                    * (prat_metadata["sample_transfer"].to(device) + 0.01).sqrt()
                ).clamp_min(1e-4)
                if safety_enabled:
                    transfer_weight = transfer_weight * prat_safe.float()
                if safety_enabled and not bool(prat_safe.any().item()):
                    pair_loss = source_loss.new_tensor(0.0)
                else:
                    pair_loss, _ = style_pair_loss(source_output, styled_output, transfer_weight)
                if safety_enabled:
                    scale_distance = (source_output.scale_weights.float() - styled_output.scale_weights.float()).pow(2).mean(1)
                    center_distance = (source_output.center_weights.float() - styled_output.center_weights.float()).pow(2).mean(1)
                    scale_pair = (scale_distance * prat_safe.float()).sum() / prat_safe.float().sum().clamp_min(1.)
                    center_pair = (center_distance * prat_safe.float()).sum() / prat_safe.float().sum().clamp_min(1.)
                else:
                    scale_pair = F.mse_loss(
                        source_output.scale_weights.float(), styled_output.scale_weights.float()
                    )
                    center_pair = F.mse_loss(
                        source_output.center_weights.float(), styled_output.center_weights.float()
                    )
            else:
                pair_loss = source_loss.new_tensor(0.0)
                scale_pair = source_loss.new_tensor(0.0)
                center_pair = source_loss.new_tensor(0.0)

            scale_inputs = [source_output.scale_weights]
            if args.use_prat:
                if not safety_enabled:
                    scale_inputs.append(styled_output.scale_weights)
                elif bool(prat_safe.any().item()):
                    scale_inputs.append(styled_output.scale_weights[prat_safe])
            if target_adaptation and target_output is not None:
                if args.include_target_scale_mi:
                    scale_inputs.append(target_output.scale_weights)
            scale_mi = (
                scale_mutual_information_loss(torch.cat(scale_inputs, dim=0))
                if len(scale_inputs) > 1 else source_loss.new_tensor(0.0)
            )

        with torch.no_grad():
            prototype_features = [source_output.features.detach()]
            prototype_labels = [source_y]
            if args.use_prat:
                if feedback_enabled and safety_enabled:
                    if bool(prat_safe.any().item()):
                        prototype_features.append(styled_output.features[prat_safe].detach())
                        prototype_labels.append(source_y[prat_safe])
                else:
                    prototype_features.append(styled_output.features.detach())
                    prototype_labels.append(source_y)
            if feedback_enabled and safety_enabled and cnt_output is not None and bool(cnt_safe.any().item()):
                prototype_features.append(cnt_output.features[cnt_safe].detach())
                prototype_labels.append(source_y[cnt_safe])
            memory.update_source_prototypes(
                torch.cat(prototype_features, dim=0),
                torch.cat(prototype_labels, dim=0),
            )

            if target_adaptation:
                assert target_output is not None and strong_target_output is not None
                assert teacher_target is not None and teacher_probability is not None
                target_probability = F.softmax(target_output.logits.detach().float(), dim=1)
                strong_probability = F.softmax(strong_target_output.logits.detach().float(), dim=1)
                if feedback_enabled and feedback is not None:
                    teacher_prediction = teacher_probability.argmax(1)
                    student_agrees = (teacher_prediction.eq(target_probability.argmax(1)) &
                                      teacher_prediction.eq(strong_probability.argmax(1)))
                if args.pseudo_mode == "current":
                    reliability_batch = memory.current_pseudo_batch(
                        target_ids,
                        teacher_probability,
                        target_probability,
                        teacher_target.features,
                        strong_probability,
                    )
                else:
                    reliability_batch = memory.update(
                        target_ids,
                        teacher_probability,
                        target_probability,
                        teacher_target.features,
                        strong_probability,
                        guard_probability,
                    )
                # A perturbation test that does not reuse the stable_mask as evidence.
                # This attenuates (does not hard-reject) pseudo-supervision when
                # the prediction is brittle under a smooth spectral response change.
                if args.independent_target_reweight or (feedback_enabled and feedback is not None
                                                         and args.ssf_independent_target_probe):
                    with torch.no_grad():
                        probe_x = independent_spectral_view(target_x, args.ssf_probe_strength)
                        with autocast_context(device, args.amp):
                            probe_output = teacher(probe_x)
                        probe_prob = F.softmax(probe_output.logits.float(), dim=1)
                        base_prob = F.softmax(teacher_target.logits.float(), dim=1)
                        agreement_score = js_agreement(base_prob, probe_prob)
                        same = base_prob.argmax(1).eq(probe_prob.argmax(1))
                        agreement_score = agreement_score * same.float()
                        if args.independent_target_reweight:
                            reliability_batch.reliability = (
                                reliability_batch.reliability *
                                (0.4 + 0.6 * agreement_score).detach())
                        if feedback_enabled and feedback is not None and args.ssf_independent_target_probe:
                            feedback.observe_target_independent(
                                base_prob.argmax(1), agreement_score)
                elif feedback_enabled and feedback is not None:
                    feedback.observe_target(
                        reliability_batch.labels.detach(),
                        reliability_batch.stable_mask.detach(),
                        student_agrees.detach())
            else:
                fallback_probability = F.softmax(source_output.logits.detach().float(), dim=1)
                reliability_batch = memory.empty_batch(
                    target_ids[: fallback_probability.size(0)],
                    fallback_probability,
                    source_output.features.detach(),
                )

        with autocast_context(device, args.amp):
            metric_loss = source_metric_loss(
                source_output.features,
                source_y,
                memory.source_prototypes,
                memory.source_valid,
            )
            if target_adaptation:
                assert target_output is not None and strong_target_output is not None and teacher_target is not None
                if (
                    args.dual_head_filter
                    and getattr(strong_target_output, "logits2", None) is not None
                    and reliability_batch is not None
                ):
                    p1 = F.softmax(strong_target_output.logits.float(), dim=1)
                    p2 = F.softmax(strong_target_output.logits2.float(), dim=1)
                    consistent = p1.argmax(dim=1).eq(p2.argmax(dim=1))
                    reliability_batch.stable_mask = reliability_batch.stable_mask & consistent
                pseudo_loss = memory.weighted_pseudo_loss(
                    strong_target_output.logits, reliability_batch
                )
                teacher_losses = teacher_consistency_losses(
                    strong_target_output, teacher_target, reliability_batch.reliability
                )
                proto_loss = (
                    memory.prototype_align_flagment_loss(
                        target_output.features,
                        reliability_batch,
                        min_confidence=float(getattr(args, "prototype_confidence", 0.95)),
                        temperature=float(getattr(args, "prototype_temperature", 0.08)),
                    )
                    if prototype_align_flagment_enabled(args)
                    else source_loss.new_tensor(0.0)
                )
                if prototype_align_flagment_enabled(args):
                    proto_mask = (
                        reliability_batch.stable_mask
                        & reliability_batch.confidence.ge(
                            float(getattr(args, "prototype_confidence", 0.95))
                        )
                    )
                    if bool(memory.source_valid.any().item()):
                        proto_mask = (
                            proto_mask
                            & memory.source_valid[reliability_batch.labels]
                        )
                    proto_used = proto_mask.float().sum()
                else:
                    proto_used = source_loss.new_tensor(0.0)
            else:
                pseudo_loss = source_loss.new_tensor(0.0)
                teacher_losses = {
                    "soft": source_loss.new_tensor(0.0),
                    "feature": source_loss.new_tensor(0.0),
                    "scale": source_loss.new_tensor(0.0),
                }
                proto_loss = source_loss.new_tensor(0.0)
                proto_used = source_loss.new_tensor(0.0)

            lmmd_term = source_loss.new_tensor(0.0)
            if (
                args.lambda_lmmd > 0.0
                and (args.align_maturity_gate or epoch >= args.lmmd_start_epoch)
                and target_adaptation
                and target_output is not None
                and reliability_batch is not None
                and reliability_batch.labels.numel() > 0
                and maturity_factor > 0.0
            ):
                lmmd_term = reliability_lmmd(
                    source_output.features, source_y, target_output.features,
                    reliability_batch.labels, reliability_batch.stable_mask,
                    reliability_batch.reliability, memory.num_classes,
                    min_samples=int(args.lmmd_min_samples))
            coral_term = source_loss.new_tensor(0.0)
            if args.lambda_coral > 0.0 and maturity_factor > 0.0 and target_adaptation and target_output is not None:
                coral_term = coral_loss(source_output.features, target_output.features)
            # Label-free maturity and warmup gates are applied to all aligners.
            lmmd_time = (alignment_active_epochs if args.align_maturity_gate
                         else epoch - args.lmmd_start_epoch + 1)
            lmmd_warm = (min(1.0, max(0.0, lmmd_time /
                                     max(1, int(args.lmmd_warm_epochs))))
                         if int(args.lmmd_warm_epochs) > 0 else 1.0)
            effective_lmmd_lambda = args.lambda_lmmd * lmmd_warm * maturity_factor
            effective_coral_lambda = args.lambda_coral * maturity_factor
            confusion_weight = args.lambda_confusion if args.use_confusion_loss else 0.0
            cca_loss = source_loss.new_tensor(0.0)
            cca_lambda = float(args.lambda_cca)
            if (
                args.lambda_cca > 0.0
                and (args.align_maturity_gate or epoch >= args.cca_start_epoch)
                and target_adaptation
                and strong_target_output is not None
                and bool(memory.source_valid.all().item())
                and maturity_factor > 0.0
            ):
                cca_loss = memory.class_conditional_alignment(
                    strong_target_output.features,
                    temperature=float(args.cca_temperature),
                )
                # CCA warm-up: 从 0 线性升到 lambda_cca
                warm = int(getattr(args, "cca_warm_epochs", 0))
                if warm > 0:
                    used = (alignment_active_epochs if args.align_maturity_gate else epoch - args.cca_start_epoch)
                    progress = min(1.0, max(0.0, used / max(1, warm)))
                    cca_lambda = args.lambda_cca * progress

            cca_lambda *= maturity_factor

            # CNT supervises an unlabeled-target-derived, bounded SOURCE augmentation.
            if cnt_output is not None:
                cnt_weights = (cnt_safe.float() if (safety_enabled or args.cnt_teacher_backtrack)
                               else torch.ones_like(source_y, dtype=torch.float))
                cnt_ce = weighted_ce(cnt_output.logits, source_y, cnt_weights,
                                     label_smoothing=args.label_smoothing, class_weights=class_weights)
                reference_margin = true_margin(source_output.logits, source_y).detach()
                cnt_margin = true_margin(cnt_output.logits, source_y)
                violation = F.relu(reference_margin - cnt_margin - 0.1).square()
                cnt_margin_loss = (violation * cnt_weights).sum() / cnt_weights.sum().clamp_min(1.)
                cnt_term = cnt_ce + 0.25 * cnt_margin_loss
                cnt_lambda = float(args.lambda_cnt) * min(1., epoch / max(1, int(args.cnt_warm_epochs)))
            else:
                cnt_term = source_loss.new_tensor(0.0)
                cnt_lambda = 0.0

            total_loss = (
                source_loss
                + (args.lambda_style_supervision if args.use_prat else 0.0) * styled_supervision
                + args.lambda_spectral_aux * spectral_aux
                + args.lambda_source_metric * metric_loss
                + (args.lambda_style_pair if args.use_prat else 0.0) * pair_loss
                + args.lambda_scale_mi * scale_mi
                + (args.lambda_scale_pair if args.use_prat else 0.0) * scale_pair
                + (args.lambda_center_pair if args.use_prat else 0.0) * center_pair
                + effective_coral_lambda * coral_term
                + args.lambda_dual_source * source_loss2
                + effective_lmmd_lambda * lmmd_term
                + cca_lambda * cca_loss
                + pseudo_ramp * args.lambda_pseudo * pseudo_loss
                + pseudo_ramp * args.lambda_soft_consistency * teacher_losses["soft"]
                + pseudo_ramp * args.lambda_feature_consistency * teacher_losses["feature"]
                + pseudo_ramp * args.lambda_scale_teacher * teacher_losses["scale"]
                + pseudo_ramp * float(getattr(args, "prototype_weight", 0.05)) * proto_loss
                + cnt_lambda * cnt_term
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if getattr(args, "use_ema_teacher", True):
            update_ema(student, teacher, args.ema_decay)
        else:
            teacher.load_state_dict(student.state_dict())

        batch_values = {
            "loss": total_loss.detach(),
            "sup": source_loss.detach(),
            "style_sup": styled_supervision.detach(),
            "pseudo": pseudo_loss.detach(),
            "pseudo_effective": pseudo_loss.detach() * float(pseudo_ramp),
            "soft": teacher_losses["soft"].detach(),
            "feature": teacher_losses["feature"].detach(),
            "metric": metric_loss.detach(),
            "style_pair": pair_loss.detach(),
            "scale_mi": scale_mi.detach(),
            "selected": reliability_batch.stable_mask.float().mean().detach(),
            "reliability": reliability_batch.reliability.mean().detach(),
            "transfer": prat_metadata["mean_transfer"].detach(),
            "proto": proto_loss.detach(),
            "proto_used": proto_used.detach(),
            "cnt": cnt_term.detach(),
            "lmmd": lmmd_term.detach(),
            "cca": cca_loss.detach(),
            "maturity": source_loss.new_tensor(snapshot['maturity']),
            "alignment_scale": source_loss.new_tensor(maturity_factor),
            "alignment_active_epochs": source_loss.new_tensor(alignment_active_epochs),
            "lmmd_scale": source_loss.new_tensor(effective_lmmd_lambda),
            "cnt_shift": source_loss.new_tensor(cnt_shift),
            "prat_safe": prat_safe.float().mean().detach(),
            "cnt_safe": cnt_safe.float().mean().detach(),
            "ssf_scale_cnt": class_scales.mean().detach(),
            "ssf_scale_prat": (feedback.scale_prat.mean().detach()
                               if feedback is not None else source_loss.new_tensor(1.)),
        }
        for key, value in batch_values.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value.float().item())
        batches += 1

    metrics = {key: value / max(1, batches) for key, value in accumulator.items()}
    if feedback_enabled and feedback is not None:
        # Feedback is committed once per epoch and used only in subsequent epochs.
        metrics.update(feedback.end_epoch(use_target=True))
    metrics.update({
        "lr": float(lr),
        "pseudo_ramp": float(pseudo_ramp),
        "style_strength": float(style_strength),
    })
    return metrics

def train(args: argparse.Namespace) -> dict[str, Any]:
    args = apply_ablation_profile(args)
    if getattr(args, "use_gt_target_mask", False) or float(getattr(args, "target_gt_mix_ratio", 0)) != 0 or args.diagnostic_target_oa:
        raise RuntimeError("v7.1 training forbids all target GT access and in-training OA diagnostics")
    if not getattr(args, "skip_final_evaluation", True):
        raise RuntimeError("v7.1 requires locked independent final evaluation")
    if args.dual_head_filter and args.lambda_dual_source <= 0:
        raise ValueError("dual_head_filter requires supervised classifier2: --lambda-dual-source > 0")
    if not 0.0 < float(args.source_budget_train_fraction) <= 1.0:
        raise ValueError("source_budget_train_fraction must be in (0, 1]")
    set_seed(args.seed, args.deterministic)
    if not torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_dir = Path(args.save_path) / args.run_name
    else:
        run_dir = (
            Path(args.save_path)
            / f"{args.source_name}_to_{args.target_name}"
            / f"{args.ablation_id}_seed{args.seed}_{timestamp}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    loaders = build_loaders(args, device)
    ssf_mode = str(getattr(args, "ssf_mode", "C0"))
    class_basis = None
    if ssf_mode in ("C1", "C2", "C3"):
        source_dataset = loaders["source_loader"].dataset
        coordinates = source_dataset.coordinates
        radius = source_dataset.radius
        centers = source_dataset.image[
            coordinates[:, 0] + radius, coordinates[:, 1] + radius, :
        ]
        class_basis = class_basis_from_centers(
            torch.as_tensor(centers, dtype=torch.float32),
            torch.as_tensor(source_dataset.labels, dtype=torch.long),
            loaders["num_classes"],
        ).to(device)
    source_spectral_means = None
    if class_basis is not None:
        source_dataset = loaders["source_loader"].dataset
        c = torch.as_tensor(centers, dtype=torch.float32, device=device)
        labels_c = torch.as_tensor(source_dataset.labels, dtype=torch.long, device=device)
        source_spectral_means = torch.zeros(loaders["num_classes"], c.size(1), device=device)
        for class_id in range(loaders["num_classes"]):
            sample_mask = labels_c.eq(class_id)
            if sample_mask.any():
                source_spectral_means[class_id] = c[sample_mask].mean(0)
            else:
                source_spectral_means[class_id] = torch.nan
    feedback = (SafetyFeedback(loaders["num_classes"], device,
                               min_source=args.ssf_min_source,
                               target_support=args.ssf_min_target)
                if ssf_mode in ("C3", "C4") else None)
    diagnostic_target_loader = None  # fail-closed; target GT only in locked evaluation

    if bool(getattr(args, "pseudo_balance_flag", False)):
        print("[config] pseudo balance ON")
    if bool(getattr(args, "disable_prat_flag", False)):
        print("[config] PRAT OFF")
    if bool(getattr(args, "collapse_guard_flag", False)):
        print("[config] collapse guard ON")

    student = DualReliabilityEncoder(
        bands=loaders["bands"],
        num_classes=loaders["num_classes"],
        projection_dim=args.projection_dim,
        projection_norm=args.projection_norm,
        aspp_mode=args.aspp_mode,
        pooling_mode=args.pooling_mode,
        spectral_fusion_mode=args.spectral_fusion_mode,
        amplitude_channel=args.amplitude_channel,
        amplitude_input=bool(getattr(args, "amplitude_input", False)),
        backbone=str(getattr(args, "backbone", "ours")),
        enable_dual_head=(args.dual_head_filter or args.lambda_dual_source > 0),
    ).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    guard_model: DualReliabilityEncoder | None = None
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    spectral_bands = int(loaders["bands"]) - (
        1 if bool(getattr(args, "amplitude_input", False)) else 0
    )
    prat = PhaseReliabilityResidualAmplitudeTransfer(
        bands=spectral_bands,
        patch_size=args.patch_size,
        ema_momentum=args.prat_ema_momentum,
        low_frequency_ratio=args.prat_low_frequency_ratio,
        amplitude_temperature=args.prat_amplitude_temperature,
        energy_temperature=args.prat_energy_temperature,
        delta_clip=args.prat_delta_clip,
        mad_clip=args.prat_reference_mad_clip,
        reference_update_interval=args.prat_reference_update_interval,
        mode=args.prat_mode,
    ).to(device)
    memory = ReliabilityPrototypeMemory(
        num_samples=loaders["target_samples"],
        num_classes=loaders["num_classes"],
        feature_dim=args.projection_dim,
        neighbor_ids=loaders["neighbor_ids"],
        device=device,
        probability_momentum=0.92,
        feature_momentum=0.94,
        target_prior_momentum=0.985,
        num_subprototypes=args.cempa_num_subprototypes,
        source_prototype_momentum=0.97,
        target_prototype_update_rate=0.16,
        target_prototype_create_similarity=0.90,
        reliability_threshold=args.reliability_threshold,
        pseudo_confidence=args.pseudo_confidence,
        pseudo_margin=args.pseudo_margin,
        class_cap=args.cempa_class_cap,
        min_age=args.cempa_min_age,
        min_streak=args.cempa_min_streak,
        pseudo_min_per_class=int(getattr(args, "pseudo_min_per_class", 0)),
        use_probability_history=args.use_probability_history,
        use_time_score=args.use_time_score,
        use_feature_score=args.use_feature_score,
        use_view_score=args.use_view_score,
        use_guard=args.use_guard,
        use_prototype_veto=args.use_prototype_veto,
        use_spatial=args.use_spatial_reliability,
        use_target_prototypes=args.use_target_prototypes,
        reliability_weighted_prototype_update=args.reliability_weighted_prototype_update,
        adaptive_anchor=args.adaptive_anchor,
        fixed_target_anchor_weight=args.fixed_target_anchor_weight,
        strict_class_balance_strength=float(
            getattr(
                args,
                "strict_class_balance_strength",
                1.0 if bool(getattr(args, "pseudo_balance_flag", False)) else 0.0,
            )
        ),
        collapse_guard=(
            bool(getattr(args, "collapse_guard_flag", False))
        ),
    )

    if str(getattr(args, "optimizer_type", "adamw")).lower() == "sgd":
        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    candidates: list[dict[str, Any]] = []
    log_path = run_dir / "training_log.jsonl"

    print(
        "Standalone dual-reliability UDA ablation | "
        f"id={args.ablation_id} device={device} source={args.source_name} "
        f"target={args.target_name} seed={args.seed}"
    )
    print(f"Ablation definition: {ABLATION_DESCRIPTIONS[args.ablation_id]}")
    if prototype_align_flagment_enabled(args):
        print(
            "Pavia prototype alignment ON: high-confidence target features are anchored "
            f"to source class prototypes (weight={args.prototype_weight:.3f})"
        )
    print(
        f"bands={loaders['bands']} classes={loaders['num_classes']} "
        f"source_samples={loaders['source_samples']} target_samples={loaders['target_samples']}"
    )
    print(
        "Strict unsupervised target training is active: target labels are inaccessible "
        "until the final benchmark step."
    )
    print(f"[v7.1] SSF={ssf_mode}; CNT class-basis rank="
          f"{class_basis.shape[1] if class_basis is not None else 0}; "
          f"feedback={'on' if feedback is not None else 'off'}")
    print(
        f"patch={args.patch_size} projection_norm={args.projection_norm} "
        f"band_alignment_scale={loaders['alignment']['mean_abs_scale_delta']:.5f} "
        f"band_alignment_shift={loaders['alignment']['mean_abs_shift_delta']:.5f}"
    )

    alignment_state = {'active_epochs': 0, 'first_epoch': None}
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        if (
            args.pseudo_mode != "off"
            and epoch >= args.pseudo_start_epoch
            and epoch % 20 == 0
            and hasattr(memory, "refresh_clean_mask")
        ):
            memory.refresh_clean_mask()
        if args.use_guard and epoch == args.anchor_freeze_epoch:
            guard_model = copy.deepcopy(teacher).to(device)
            guard_model.eval()
            for parameter in guard_model.parameters():
                parameter.requires_grad_(False)
            print(f"Frozen the independent semantic guard at epoch {epoch}.")
        metrics = run_epoch(
            epoch=epoch,
            total_epoch_index=epoch - 1,
            total_schedule_epochs=args.epochs,
            args=args,
            student=student,
            teacher=teacher,
            guard_model=guard_model,
            prat=prat,
            memory=memory,
            source_loader=loaders["source_loader"],
            target_loader=loaders["target_loader"],
            optimizer=optimizer,
            scaler=scaler,
            class_weights=loaders["class_weights"],
            device=device,
            refinement=False,
            feedback=feedback,
            class_basis=class_basis,
            source_spectral_means=source_spectral_means,
            alignment_state=alignment_state,
        )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"loss={metrics['loss']:.4f} sup={metrics['sup']:.4f} "
                f"pseudo_eff={metrics['pseudo_effective']:.4f} pRamp={metrics['pseudo_ramp']:.2f} "
                f"pProto={metrics.get('proto', 0.0):.4f} "
                f"pN={metrics.get('proto_used', 0.0):.1f} "
                f"cnt={metrics['cnt']:.4f} cnt_shift={metrics['cnt_shift']:.4f} "
                f"LMMD={metrics['lmmd']:.4f} CCA={metrics['cca']:.4f} "
                f"maturity={metrics['maturity']:.3f} align_gate={metrics['alignment_scale']:.2f} "
                f"pSafe={metrics['prat_safe']:.2f} cSafe={metrics['cnt_safe']:.2f} "
                f"rel={metrics['reliability']:.3f} "
                f"selected={100.0 * metrics['selected']:.1f}% transfer={metrics['transfer']:.4f} "
                f"lr={metrics['lr']:.2e} time={elapsed:.1f}m"
            )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            select_mode = str(getattr(args, "select_mode", "") or "")
            if select_mode == "final":
                final_epoch_protocol, label_free_only = True, False
            elif select_mode == "label_free":
                final_epoch_protocol, label_free_only = False, True
            else:
                final_epoch_protocol = bool(
                    getattr(args, "select_final_epoch", False)
                ) or loaders["source_validation_loader"] is None
                label_free_only = False
            if final_epoch_protocol:
                # Literature-aligned protocol: no source validation split, the
                # final epoch is reported. Diagnostics are kept for the log only.
                source_student = 0.0
                source_teacher = 0.0
                score = 0.0
                diagnostics = memory.unsupervised_diagnostics()
            elif label_free_only:
                # Hide no labels at all: the checkpoint is chosen by the label-free
                # reliability score only (memory diagnostics + target consistency).
                source_student = 0.0
                source_teacher = 0.0
                diagnostics = memory.unsupervised_diagnostics()
                target_consistency = target_consistency_score(
                    teacher, loaders["target_fit_loader"], device
                )
                score = label_free_selection_score(
                    0.0, diagnostics, metrics["selected"], target_consistency
                )
            else:
                source_student = evaluate_source(
                    student, loaders["source_validation_loader"], device
                )
                source_teacher = evaluate_source(
                    teacher, loaders["source_validation_loader"], device
                )
                diagnostics = memory.unsupervised_diagnostics()
                target_consistency = target_consistency_score(
                    teacher, loaders["target_fit_loader"], device
                )
                score = label_free_selection_score(
                    source_teacher, diagnostics, metrics["selected"], target_consistency
                )
            record = {
                "epoch": epoch,
                "score": float(score),
                "source_student": float(source_student),
                "source_teacher": float(source_teacher),
                "diagnostics": diagnostics,
                "state": state_dict_to_cpu(teacher),
            }
            candidates.append(record)
            if final_epoch_protocol:
                candidates = [record]
                print(
                    f"  final-epoch protocol: epoch={epoch} "
                    f"reliability={diagnostics['reliability']:.3f} "
                    f"coverage={diagnostics['coverage']:.3f} "
                    f"entropy={diagnostics['entropy']:.3f} "
                    "(no source validation; final epoch reported)"
                )
            else:
                candidates = sorted(
                    candidates, key=lambda item: item["score"], reverse=True
                )[: args.checkpoint_topk]
                print(
                    "  label-free selection: "
                    f"source(student/teacher)={source_student:.2f}/{source_teacher:.2f} "
                    f"score={score:.3f} reliability={diagnostics['reliability']:.3f} "
                    f"coverage={diagnostics['coverage']:.3f} "
                    f"prototype={diagnostics['prototype_coverage']:.3f} "
                    f"entropy={diagnostics['entropy']:.3f} "
                    f"pseudo_peak={diagnostics.get('pseudo_peak', 0.0):.3f} "
                    f"best_epoch={candidates[0]['epoch']}"
                )
            if hasattr(memory, "flip_diagnostics"):
                flip_stats = memory.flip_diagnostics()
                if epoch % 10 == 0 or epoch == args.epochs:
                    print(
                        f"  [flip-diag] valid={flip_stats['n_valid']} "
                        f"mean_flips={flip_stats['mean_flips']:.2f} "
                        f"zero={flip_stats['pct_zero']*100:.1f}% "
                        f"ge3={flip_stats['pct_ge3']*100:.1f}% "
                        f"ge5={flip_stats['pct_ge5']*100:.1f}%"
                    )

            if diagnostic_target_loader is not None:
                diag = evaluate_target(
                    teacher, diagnostic_target_loader, device,
                    num_classes=loaders["num_classes"], use_tta=False,
                )
                print(
                    f"  [diagnostic] REAL target OA={diag['OA']:.2f} "
                    f"AA={diag['AA']:.2f} Kappa={diag['Kappa']:.4f} "
                    f"(epoch {epoch}, NOT used for selection)"
                )
                with log_path.open("a", encoding="utf-8") as dhandle:
                    dhandle.write(json.dumps({
                        "stage": "diagnostic", "epoch": epoch,
                        "diagnostic_target_OA": diag["OA"],
                        "diagnostic_target_AA": diag["AA"],
                        "diagnostic_target_Kappa": diag["Kappa"],
                    }, ensure_ascii=False) + "\n")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "stage": "main",
                            "epoch": epoch,
                            "metrics": metrics,
                            "source_student": source_student,
                            "source_teacher": source_teacher,
                            "selection_score": score,
                            "diagnostics": diagnostics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if not candidates:
        raise RuntimeError("No label-free checkpoint candidate was collected.")
    if args.checkpoint_average > 1:
        selected_state = weighted_average_state_dicts(candidates, args.checkpoint_average)
    else:
        selected_state = copy.deepcopy(candidates[0]["state"])
    student.load_state_dict(selected_state, strict=True)
    teacher.load_state_dict(selected_state, strict=True)
    base_record = {
        "epoch": 0,
        "score": float(candidates[0]["score"]),
        "source_student": float(candidates[0]["source_teacher"]),
        "source_teacher": float(candidates[0]["source_teacher"]),
        "diagnostics": candidates[0]["diagnostics"],
        "state": copy.deepcopy(selected_state),
    }
    refinement_candidates: list[dict[str, Any]] = [base_record]

    if args.refine_epochs > 0:
        optimizer = AdamW(student.parameters(), lr=args.refine_lr, weight_decay=args.weight_decay)
        scaler = make_grad_scaler(args.amp and device.type == "cuda")
        print("Starting conservative single-model reliability refinement.")
        refinement_start = time.time()
        for refine_epoch in range(1, args.refine_epochs + 1):
            metrics = run_epoch(
                epoch=args.epochs + refine_epoch,
                total_epoch_index=refine_epoch - 1,
                total_schedule_epochs=args.refine_epochs,
                args=args,
                student=student,
                teacher=teacher,
                guard_model=guard_model,
                prat=prat,
                memory=memory,
                source_loader=loaders["source_loader"],
                target_loader=loaders["target_loader"],
                optimizer=optimizer,
                scaler=scaler,
                class_weights=loaders["class_weights"],
                device=device,
                refinement=True,
                feedback=feedback,
                class_basis=class_basis,
                source_spectral_means=source_spectral_means,
                alignment_state=alignment_state,
            )
            if refine_epoch == 1 or refine_epoch % args.log_interval == 0 or refine_epoch == args.refine_epochs:
                elapsed = (time.time() - refinement_start) / 60.0
                print(
                    f"refine {refine_epoch:03d}/{args.refine_epochs} "
                    f"loss={metrics['loss']:.4f} pseudo_eff={metrics['pseudo_effective']:.4f} "
                    f"rel={metrics['reliability']:.3f} "
                    f"selected={100.0 * metrics['selected']:.1f}% lr={metrics['lr']:.2e} "
                    f"time={elapsed:.1f}m"
                )
            if refine_epoch % args.eval_interval == 0 or refine_epoch == args.refine_epochs:
                source_teacher = evaluate_source(teacher, loaders["source_validation_loader"], device)
                source_student = evaluate_source(student, loaders["source_validation_loader"], device)
                diagnostics = memory.unsupervised_diagnostics()
                target_consistency = target_consistency_score(
                    teacher, loaders["target_fit_loader"], device
                )
                score = label_free_selection_score(
                    source_teacher, diagnostics, metrics["selected"], target_consistency
                )
                refinement_candidates.append(
                    {
                        "epoch": refine_epoch,
                        "score": float(score),
                        "source_student": float(source_student),
                        "source_teacher": float(source_teacher),
                        "diagnostics": diagnostics,
                        "state": state_dict_to_cpu(teacher),
                    }
                )
                refinement_candidates = sorted(
                    refinement_candidates, key=lambda item: item["score"], reverse=True
                )[: args.checkpoint_topk]
                print(
                    "  refinement selection: "
                    f"source={source_teacher:.2f} score={score:.3f} "
                    f"reliability={diagnostics['reliability']:.3f} "
                    f"coverage={diagnostics['coverage']:.3f} "
                    f"entropy={diagnostics['entropy']:.3f} "
                    f"best_refine_epoch={refinement_candidates[0]['epoch']}"
                )
    else:
        print("Reliability refinement is disabled to avoid late pseudo-label drift.")

    final_record = refinement_candidates[0]
    teacher.load_state_dict(final_record["state"], strict=True)
    teacher.eval()
    checkpoint_path = run_dir / "selected_model.pt"
    torch.save(
        {
            "model": state_dict_to_cpu(teacher),
            "prat": {name: tensor.detach().cpu() for name, tensor in prat.state_dict().items()},
            "prototype_summary": {
                "target_prior": memory.target_prior.detach().cpu(),
                "source_prototypes": memory.source_prototypes.detach().cpu(),
                "source_valid": memory.source_valid.detach().cpu(),
                "target_prototypes": memory.target_prototypes.detach().cpu(),
                "target_valid": memory.target_valid.detach().cpu(),
            },
            "args": vars(args),
            "selection": {
                "main_best_epoch": int(candidates[0]["epoch"]),
                "refinement_best_epoch": int(final_record["epoch"]),
                "score": float(final_record["score"]),
                "source_validation_accuracy": float(final_record["source_teacher"]),
                "unsupervised_diagnostics": final_record["diagnostics"],
            },
        },
        checkpoint_path,
    )

    result: dict[str, Any] = {
        "source": args.source_name,
        "target": args.target_name,
        "seed": args.seed,
        "ablation_id": args.ablation_id,
        "ablation_description": ABLATION_DESCRIPTIONS[args.ablation_id],
        "ssf_mode": ssf_mode,
        "strict_unsupervised": True,
        "selected_checkpoint": str(checkpoint_path),
        "selection": {
            "main_best_epoch": int(candidates[0]["epoch"]),
            "refinement_best_epoch": int(final_record["epoch"]),
            "score": float(final_record["score"]),
            "source_validation_accuracy": float(final_record["source_teacher"]),
            "unsupervised_diagnostics": final_record["diagnostics"],
        },
        "band_alignment": loaders["alignment"],
    }

    if not args.skip_final_evaluation:
        print("Target ground truth is now loaded for the final benchmark only.")
        target_evaluation_loader = build_target_evaluation_loader(args, loaders["target_image"])
        target_metrics = evaluate_target(
            teacher,
            target_evaluation_loader,
            device,
            num_classes=loaders["num_classes"],
            use_tta=args.final_tta,
        )
        result["target_metrics"] = target_metrics
        print("Final target benchmark")
        print(np.asarray(target_metrics["confusion_matrix"], dtype=np.int64))
        print(
            f"OA={target_metrics['OA']:.2f} AA={target_metrics['AA']:.2f} "
            f"Kappa={target_metrics['Kappa']:.4f}"
        )
    else:
        print("Final target benchmark was skipped; target ground truth was never opened.")

    (run_dir / "final_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "config", ""):
        args = load_config_into_args(args, args.config)
    train(args)


if __name__ == "__main__":
    main()
