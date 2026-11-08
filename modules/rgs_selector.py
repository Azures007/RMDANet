"""RGS independent selection branch.

The original RGS inherits the PATS target-contrastive selection set, so the
target-domain classification CE is constrained by contrastive-learning pairing
rules (per-class top-k, prototype ranking, pair filtering) that a plain CE does
not need.  This module implements an *independent* selector that builds
``S_cls`` from the whole target batch:

    legacy            : no independent selection (caller keeps the old path)
    independent_conf  : teacher confidence >= threshold
    independent_agree : confidence + teacher/F2 class agreement
    independent_proto : confidence + source-prototype margin support
    independent_joint : confidence + agreement + prototype support

All tensors are detached; the selector never builds a graph and never touches
target ground truth.
"""

from __future__ import annotations

import torch


MODES = ("legacy", "independent_conf", "independent_agree",
         "independent_proto", "independent_joint")


def _prototype_support(student_proj, labels, proto_state, eps=1e-8):
    """Cosine margin of the pseudo class prototype vs. the best other class."""
    if proto_state is None:
        return None
    bank = proto_state.get("prototypes")
    seen = proto_state.get("seen")
    if bank is None or seen is None or bank.numel() == 0:
        return None
    if bank.size(1) != student_proj.size(1):
        return None
    proj = torch.nn.functional.normalize(student_proj.float(), dim=1)
    bank_n = torch.nn.functional.normalize(bank.float(), dim=1)
    sim = torch.matmul(proj, bank_n.t())                 # (N, C)
    valid = seen[:bank.size(0)] > 0
    if not valid.any():
        return None
    # mask invalid prototypes
    sim = sim.masked_fill(~valid.unsqueeze(0), -1e4)
    own = sim.gather(1, labels.view(-1, 1)).squeeze(1)
    other = sim.clone()
    other.scatter_(1, labels.view(-1, 1), -1e4)
    best_other = other.max(dim=1).values
    margin = own - best_other
    own_valid = valid[labels]
    return margin, own_valid


def select_rgs_samples(teacher_pred=None, teacher_conf=None,
                       f1_logits=None, f2_logits=None,
                       student_proj=None, source_prototype_state=None,
                       num_classes=7, ep=1, args=None):
    """Return (indices, labels, weights, stats).

    ``indices`` are local to the current target batch; an empty tensor means
    the batch must be skipped (no relaxation is applied).
    """
    if args is None:
        raise ValueError("args is required")
    mode = str(getattr(args, "rgs_selector", "legacy"))
    stats = {
        "mode": mode,
        "total": 0, "selected": 0, "conf_mean": 0.0, "agree_rate": 0.0,
        "proto_rate": 0.0, "per_class": [], "empty": False,
    }
    if mode == "legacy":
        return None, None, None, stats

    device = teacher_conf.device
    n = int(teacher_conf.numel())
    stats["total"] = n
    if n == 0:
        stats["empty"] = True
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, device=device), stats)

    with torch.no_grad():
        conf = teacher_conf.detach().float().view(-1)
        pred = (teacher_pred.detach().long().view(-1)
                if teacher_pred is not None else conf.new_zeros(n, dtype=torch.long))

        mask = conf >= float(getattr(args, "rgs_conf_threshold", 0.95))

        if mode in ("independent_agree", "independent_joint") and f2_logits is not None:
            f2_pred = f2_logits.detach().argmax(dim=1)
            agree = f2_pred.eq(pred)
            stats["agree_rate"] = float(agree.float().mean().item())
            mask = mask & agree

        proto_mask = None
        if mode in ("independent_proto", "independent_joint") and student_proj is not None:
            support = _prototype_support(student_proj.detach(), pred,
                                         source_prototype_state)
            if support is not None:
                margin, own_valid = support
                proto_mask = (margin > 0) & own_valid
                stats["proto_rate"] = float(proto_mask.float().mean().item())
                mask = mask & proto_mask

        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            stats["empty"] = True
            stats["selected"] = 0
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device), stats)

        labels = pred[idx]
        conf_sel = conf[idx]

        # optional per-class cap (ablation only; disabled by default)
        cap = int(getattr(args, "rgs_class_cap", 0))
        if cap > 0:
            keep = torch.zeros_like(conf_sel, dtype=torch.bool)
            for c in labels.unique().tolist():
                pos = (labels == int(c)).nonzero(as_tuple=True)[0]
                order = conf_sel[pos].argsort(descending=True)
                keep[pos[order[:cap]]] = True
            idx, labels, conf_sel = idx[keep], labels[keep], conf_sel[keep]

        mode_w = str(getattr(args, "rgs_weight_mode", "teacher_conf"))
        if mode_w == "uniform":
            weights = torch.ones_like(conf_sel)
        else:
            weights = conf_sel.clamp_min(1e-6)

        per_class = torch.bincount(labels, minlength=int(num_classes)).tolist()
        stats.update({
            "selected": int(idx.numel()),
            "conf_mean": float(conf_sel.mean().item()) if idx.numel() else 0.0,
            "per_class": per_class,
            "empty": False,
        })
        return idx.long().to(device), labels.long().to(device), weights.to(device), stats
