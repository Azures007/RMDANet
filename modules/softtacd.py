"""Soft-relation TACD: replace hard pseudo-label positive pairs with soft relation weights.

For target samples i, j the teacher class distributions give a soft same-class
probability

    q_ij = sum_c p_i(c) * p_j(c)

The contrastive objective for anchor i is the cross-entropy between the student
similarity distribution and the soft relation distribution q_i (excluding self),
i.e. uncertain pairs contribute proportionally less attraction.  This weakens
the amplification of wrong positive pairs without changing the sample set or the
teacher.
"""

from __future__ import annotations


def soft_relation_contrastive(student_proj, teacher_prob, temperature=0.1, eps=1e-8,
                              max_pos_mass=1.0):
    """Soft-label SupCon style loss.

    student_proj: (N, D) student projections of the selected target samples
    teacher_prob: (N, C) teacher class distributions of the same samples
    """
    import torch
    import torch.nn.functional as F

    z = F.normalize(student_proj, dim=1)
    sim = torch.matmul(z, z.t()) / float(temperature)
    n = sim.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)

    q = torch.matmul(teacher_prob, teacher_prob.t()).clamp_min(0.0)
    q = q * mask
    row_sum = q.sum(dim=1, keepdim=True).clamp_min(eps)
    if max_pos_mass is not None and float(max_pos_mass) > 0:
        q = q / row_sum * float(max_pos_mass)
    else:
        q = q / row_sum

    log_prob = sim - torch.logsumexp(sim.masked_fill(~mask, -1e9), dim=1, keepdim=True)
    loss = -(q * log_prob).sum(dim=1).mean()
    return loss
