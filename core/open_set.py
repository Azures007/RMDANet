"""P0: open-set virtual-class assignment for target-domain pseudo labels.

Target scenes may contain classes that do not exist in the source domain
(e.g. Pavia Water / Tiles).  Instead of silently forcing every target sample
into a shared class (or soft-down-weighting it), we model an explicit
"unknown/virtual" class and solve a balanced assignment with Sinkhorn
(entropy-regularised optimal transport) over C shared classes + 1 dustbin.

The known-class column masses follow the (uniform) source class prior and the
dustbin gets ``unknown_frac`` of the total mass.  Samples assigned to the
dustbin are excluded from pseudo-label supervision (stable_mask is cleared),
so they stop polluting the shared-class decision boundary.

No target labels are used; this is a transductive assignment on teacher
statistics only.  No new loss term is added.
"""
from __future__ import annotations

import torch


def sinkhorn_assignment(
    logits: torch.Tensor,
    unknown_frac: float = 0.1,
    temperature: float = 0.05,
    iterations: int = 20,
    class_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (assignment, unknown_mask).

    assignment: int tensor [N] with values in [0, C]; C means unknown.
    unknown_mask: bool tensor [N].
    """
    if logits.numel() == 0:
        return logits.new_zeros(0, dtype=torch.long), logits.new_zeros(0, dtype=torch.bool)
    n, c = logits.shape
    device = logits.device
    scores = torch.cat([logits.float(), torch.zeros(n, 1, device=device)], dim=1)

    if class_prior is None:
        known_prior = torch.full((c,), 1.0 / c, device=device)
    else:
        known_prior = class_prior.float().to(device).clamp_min(0.0)
        known_prior = known_prior / known_prior.sum().clamp_min(1e-8)

    frac = float(min(max(unknown_frac, 0.0), 1.0))
    col_mass = torch.cat([
        known_prior * (1.0 - frac),
        torch.tensor([frac], device=device),
    ])
    row_mass = torch.full((n,), 1.0 / n, device=device)

    f = scores / max(float(temperature), 1e-4)
    log_u = torch.zeros(n, device=device)
    log_v = torch.zeros(c + 1, device=device)
    log_row = torch.log(row_mass.clamp_min(1e-12))
    log_col = torch.log(col_mass.clamp_min(1e-12))
    for _ in range(int(iterations)):
        log_u = log_row - torch.logsumexp(f + log_v.view(1, -1), dim=1)
        log_v = log_col - torch.logsumexp(f + log_u.view(-1, 1), dim=0)
    assignment = torch.argmax(f + log_u.view(-1, 1) + log_v.view(1, -1), dim=1)
    unknown_mask = assignment.eq(c)
    return assignment, unknown_mask
