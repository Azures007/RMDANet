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
    independent_union : confidence + (agreement OR prototype support)
    independent_balanced : agreement backbone + prototype top-up for classes
                           with fewer than ``rgs_min_per_class`` samples
    independent_auto  : label-free diagnostic chooses between
                        {agree, balanced(fill 2), balanced(fill 2)+class cap 4}

All tensors are detached; the selector never builds a graph and never touches
target ground truth.
"""

from __future__ import annotations

import torch


MODES = ("legacy", "independent_conf", "independent_agree",
         "independent_proto", "independent_joint", "independent_union",
         "independent_balanced", "independent_auto")


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


def _auto_tag(args, pred, num_classes, stats, ep=1):
    """Label-free diagnostic: class-distribution Gini + missing-class fraction.

    The histogram of the teacher pseudo-label distribution is tracked with an
    EMA over batches.  A concentrated distribution (high Gini-Simpson, e.g.
    Pavia) means prototype top-up hurts, so plain agreement is used; a flat
    distribution with missing classes (e.g. Houston rare classes / HyRANK 12
    classes) means the selection needs balancing and a per-class cap.

    The tag is decided once, a few epochs after pseudo-labels start, and then
    frozen for the rest of the run (early pseudo-label histograms are noisy).
    """
    num_classes = int(num_classes)
    state = getattr(args, "_rgs_auto_state", None)
    if state is None:
        state = {"counts": [0.0] * num_classes, "calls": 0, "tag": "fill2",
                 "decided": False, "late_calls": 0}
        try:
            setattr(args, "_rgs_auto_state", state)
        except Exception:
            pass
    hist = torch.bincount(pred, minlength=num_classes).float().tolist()
    hist = (hist + [0.0] * num_classes)[:num_classes]
    if state["calls"] == 0:
        state["counts"] = list(hist)
    else:
        for i in range(num_classes):
            state["counts"][i] = 0.9 * state["counts"][i] + 0.1 * hist[i]
    state["calls"] += 1

    total = float(sum(state["counts"])) or 1.0
    probs = [c / total for c in state["counts"]]
    gini = 1.0 - sum(p * p for p in probs)
    frac_zero = sum(1 for c in state["counts"] if c / total < 0.01) / max(1, num_classes)
    # long-tail fraction: classes holding < 2% of the pseudo-label mass
    frac_tiny = sum(1 for c in state["counts"] if c / total < 0.02) / max(1, num_classes)

    pseudo_start = int(getattr(args, "pseudo_start_epoch", 0) or 0)
    late = ep >= pseudo_start + 3
    if late:
        state["late_calls"] += 1

    if (not state["decided"]) and late and state["late_calls"] >= 25:
        if num_classes >= 10 and (gini >= 0.75 or frac_zero >= 0.15 or frac_tiny >= 0.2):
            # many-class long-tailed scene (e.g. HyRANK 12 classes): cap the
            # dominant classes but keep the agreement gate.
            state["tag"] = "cap4"
        elif frac_zero >= 0.35:
            state["tag"] = "cap4"
        elif gini >= 0.72:
            state["tag"] = "agree"
        else:
            state["tag"] = "fill2"
        state["decided"] = True

    stats["auto_tag"] = state["tag"]
    stats["auto_gini"] = round(gini, 4)
    stats["auto_frac_zero"] = round(frac_zero, 4)
    stats["auto_frac_tiny"] = round(frac_tiny, 4)
    return state["tag"]


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

        conf_thr = float(getattr(args, "rgs_conf_threshold", 0.95))
        mask = conf >= conf_thr

        agree_mask = None
        if mode in ("independent_agree", "independent_joint", "independent_union",
                    "independent_balanced", "independent_auto") and f2_logits is not None:
            f2_pred = f2_logits.detach().argmax(dim=1)
            agree_mask = f2_pred.eq(pred)
            stats["agree_rate"] = float(agree_mask.float().mean().item())
            if mode != "independent_union":
                mask = mask & agree_mask

        proto_mask = None
        if mode in ("independent_proto", "independent_joint", "independent_union",
                    "independent_balanced", "independent_auto") and student_proj is not None:
            support = _prototype_support(student_proj.detach(), pred,
                                         source_prototype_state)
            if support is not None:
                margin, own_valid = support
                proto_mask = (margin > 0) & own_valid
                stats["proto_rate"] = float(proto_mask.float().mean().item())
                if mode not in ("independent_union", "independent_balanced", "independent_auto"):
                    mask = mask & proto_mask

        if mode == "independent_union":
            # union: agreement OR prototype support (whichever is available)
            support = None
            for part in (agree_mask, proto_mask):
                if part is None:
                    continue
                support = part.clone() if support is None else (support | part)
            if support is not None:
                mask = mask & support

        # ---- balanced / auto fill -----------------------------------------
        do_fill = mode == "independent_balanced"
        cap_override = int(getattr(args, "rgs_class_cap", 0))
        if mode == "independent_auto":
            tag = _auto_tag(args, pred, num_classes, stats, ep=ep)
            if tag == "fill2" and proto_mask is not None:
                # balanced semantics: the fill backbone requires prototype
                # support (this is what keeps Houston stable).
                mask = mask & proto_mask
            # fill2 = balanced min-2; cap4 = plain agreement + per-class cap 4
            # (the configuration validated on HyRANK); agree = agreement only.
            do_fill = tag == "fill2"
            if tag == "cap4" and cap_override <= 0:
                cap_override = 4

        if do_fill:
            base = mask.clone()
            min_per_class = int(getattr(args, "rgs_min_per_class", 2))
            if min_per_class > 0:
                for c in range(int(num_classes)):
                    own = base & (pred == c)
                    have = int(own.sum().item())
                    if have >= min_per_class:
                        continue
                    need = min_per_class - have
                    candidates = ((pred == c) & (conf >= conf_thr) & (~base))
                    if proto_mask is not None:
                        candidates = candidates & proto_mask
                    if int(candidates.sum().item()) == 0 and proto_mask is not None:
                        candidates = ((pred == c) & (conf >= conf_thr) & (~base))
                    if int(candidates.sum().item()) == 0:
                        continue
                    idx_c = candidates.nonzero(as_tuple=True)[0]
                    order = conf[idx_c].argsort(descending=True)
                    chosen = idx_c[order[:need]]
                    base[chosen] = True
                mask = base

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
        cap = int(cap_override)
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
