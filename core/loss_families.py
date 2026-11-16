"""Four loss families (paper-level formulation).

The training objective of RMDANet is organised into four named loss functions::

    L = L_sup + lambda_1 * L_align + lambda_2 * L_cons + lambda_3 * L_self

Each family is a weighted sum of sub-terms.  The numerical objective is the
same as the 14 individual terms used by the trainer; this module only provides
the family-level view used for reporting/ablation, so it does not change the
optimisation order (and therefore not the accuracy).

    L_sup   (multi-view supervision) : source CE + PRAT styled CE + spectral aux CE
    L_align (class-conditional align) : CCA + source metric + CORAL + LMMD + prototype
    L_cons  (consistency regulariser) : style pair + scale pair + center pair + scale MI + teacher soft
    L_self  (self-training)           : weighted pseudo-label CE
"""
from __future__ import annotations

from typing import Mapping

import torch

FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "L_sup": ("source_ce", "prat_styled_ce", "spectral_aux_ce"),
    "L_align": ("cca", "source_metric", "coral", "lmmd", "prototype", "hard_contrastive"),
    "L_cons": ("style_pair", "scale_pair", "center_pair", "scale_mi", "teacher_soft"),
    "L_self": ("pseudo_ce",),
}

SUBTERM_DESCRIPTIONS: dict[str, str] = {
    "source_ce": "source-domain cross entropy",
    "prat_styled_ce": "CE on PRAT style-transferred source view",
    "spectral_aux_ce": "auxiliary CE of the centre-spectrum branch",
    "cca": "class-conditional prototype alignment (hard-negative softplus)",
    "source_metric": "source prototype metric learning (hard-negative hinge)",
    "coral": "second-order (covariance) feature alignment",
    "lmmd": "reliability-weighted class-conditional mean alignment",
    "prototype": "target features -> source class prototypes",
    "hard_contrastive": "hard-class cross-domain contrastive (source prototypes)",
    "style_pair": "source vs styled-source composite consistency",
    "scale_pair": "source vs styled-source scale-weight consistency",
    "center_pair": "source vs styled-source centre-weight consistency",
    "scale_mi": "scale-weight mutual-information regulariser",
    "teacher_soft": "EMA-teacher soft consistency",
    "pseudo_ce": "reliability-weighted target pseudo-label CE",
}


def family_totals(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Aggregate weighted sub-terms into the four loss families."""
    totals: dict[str, torch.Tensor] = {}
    for family, members in FAMILY_MEMBERS.items():
        available = [values[name] for name in members if name in values]
        if not available:
            continue
        total = available[0]
        for item in available[1:]:
            total = total + item
        totals[family] = total
    return totals


def family_report(values: Mapping[str, torch.Tensor]) -> str:
    """One-line human-readable family summary (for training logs)."""
    totals = family_totals(values)
    parts = [f"{name}={float(value.detach()):.4f}" for name, value in totals.items()]
    return "[loss-families] " + " ".join(parts)
