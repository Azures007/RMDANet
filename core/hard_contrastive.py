"""Hard-class cross-domain contrastive loss (Stage 3 / docx 7.2).

For high-confidence target samples, the loss pulls the sample towards its
pseudo-class source prototype and pushes it away from the *hardest* wrong
prototype (the most similar wrong class).  This targets the specific confusions
found in the error analysis (Pavia: Meadow/Trees, BareSoil/Bricks,
Bitumen/Asphalt; HyRANK: C7/C8, C8/C9, C5/C4) without hard-coding class pairs.

All inputs are detached except the target features; no target GT is used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_pair_contrastive(
    target_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_valid: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    margin: float = 0.3,
) -> torch.Tensor:
    """Sample-level contrastive with hardest-negative mining over prototypes."""
    if target_features.numel() == 0:
        return target_features.new_tensor(0.0)
    f = F.normalize(target_features.float(), dim=1)
    protos = F.normalize(source_prototypes.float(), dim=1)
    sim = f @ protos.t()
    valid = source_valid.bool().view(1, -1)
    sim = sim.masked_fill(~valid, -2.0)
    labels = pseudo_labels.long().view(-1, 1)
    positive = sim.gather(1, labels).squeeze(1)
    negative = sim.clone()
    negative.scatter_(1, labels, -2.0)
    hardest_negative = negative.max(dim=1).values
    loss = F.softplus(hardest_negative - positive + float(margin))
    if sample_weights is not None:
        weight = sample_weights.float().clamp_min(0.0)
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss.mean()
