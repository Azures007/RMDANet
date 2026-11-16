"""Stage 2: target-domain credibility for class-conditional alignment.

Builds a target-label-free soft weight per target sample from four signals:

1. source-prototype margin  (cosine similarity to the best source prototype
   minus the second best; both computed on teacher features),
2. teacher confidence,
3. unified reliability (confidence x multi-view x spatial, from memory),
4. a warm-up schedule (weights are exactly 1 during the first epochs).

The weight is intentionally SOFT: cross-domain shift can push a genuinely
shared class far from the source prototypes early in training, so hard
filtering is avoided.  ``w`` is detached (stop-gradient) and multiplies the
target samples of CORAL / CCA / LMMD.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def target_credibility(
    target_features: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_valid: torch.Tensor,
    confidence: torch.Tensor,
    reliability: torch.Tensor,
    epoch: int,
    warmup_epochs: int = 10,
    margin_center: float = 0.05,
    margin_sharpness: float = 6.0,
    mode: str = "full",
    beta: float = 1.0,
) -> torch.Tensor:
    """Return per-target-sample credibility in [0, 1] (detached).

    mode="full"  : geometric mean of prototype margin, confidence, reliability.
    mode="proto" : prototype margin only (Stage-2b soft variant).
    beta         : maximum down-weighting, w' = 1 - beta * (1 - w).
    """
    if target_features.numel() == 0:
        return target_features.new_zeros(target_features.size(0))
    f = F.normalize(target_features.float(), dim=1)
    protos = F.normalize(source_prototypes.float(), dim=1)
    sim = f @ protos.t()
    valid = source_valid.bool().view(1, -1)
    sim = sim.masked_fill(~valid, -1.0)
    k = min(2, sim.size(1))
    top = sim.topk(k, dim=1).values
    margin = top[:, 0] - top[:, 1] if k > 1 else top[:, 0]
    s_proto = torch.sigmoid(margin_sharpness * (margin - margin_center))
    if mode == "proto":
        weight = s_proto
    else:
        s_conf = confidence.float().clamp(0.0, 1.0)
        s_rel = reliability.float().clamp(0.0, 1.0)
        weight = (s_proto.clamp_min(1e-6) * s_conf.clamp_min(1e-6) * s_rel.clamp_min(1e-6)).pow(1.0 / 3.0)
    if warmup_epochs > 0:
        progress = min(1.0, max(0.0, float(epoch) / float(max(1, warmup_epochs))))
    else:
        progress = 1.0
    weight = (1.0 - progress) + progress * weight
    weight = 1.0 - float(beta) * (1.0 - weight)
    return weight.detach()


def coral_loss_weighted(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deep CORAL with credibility-weighted target covariance."""
    if source_features.size(0) < 2 or target_features.size(0) < 2:
        return source_features.new_tensor(0.0)
    src = source_features.float()
    tgt = target_features.float()
    d = src.size(1)
    ns = src.size(0)
    src_c = src - src.mean(dim=0, keepdim=True)
    src_cov = (src_c.t() @ src_c) / (ns - 1)
    if weights is None:
        tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
        tgt_cov = (tgt_c.t() @ tgt_c) / (tgt.size(0) - 1)
    else:
        w = weights.float().clamp_min(0.0)
        if float(w.sum()) < 1e-6:
            return source_features.new_tensor(0.0)
        w = w / w.sum()
        mean = (tgt * w[:, None]).sum(dim=0)
        tgt_c = tgt - mean
        denominator = (1.0 - (w * w).sum()).clamp_min(1e-6)
        tgt_cov = (tgt_c.t() * w) @ tgt_c / denominator
    return ((src_cov - tgt_cov) ** 2).sum() / (4.0 * d * d)
