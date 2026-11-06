"""Class-Conditioned Cross-domain Feature Mixing (CFM).

Motivation (see docs/evidence/TAPDA_PROBLEM_EVIDENCE.md):
  * supervising target pseudo-labels is hopeless here: teacher ~79% accuracy with
    0.945 confidence, source->target calibration does not transfer, and every
    pseudo-label CE variant is neutral or harmful;
  * the training pipeline is already at a precise local optimum, so changing the
    input (normalization / FDA) or the number of steps makes it worse;
  * however, the target *feature distribution* is information that TAPDA never
    injects into the supervised loss.

CFM therefore keeps the label source untouched (labels always come from the
source domain) and only mixes target features into source-labelled features:

    f_mix = lam * f_s + (1 - lam) * f_t,   label = y_s

where f_t is selected by a high-precision consensus (teacher prediction ==
student prediction and confidence >= threshold).  The pseudo-label value itself
is never used as a training target, so pseudo-label errors cannot inject wrong
labels; they can only make the mixed feature slightly off-class.
"""

from __future__ import annotations

import torch


def build_mixed_features(feat, src_size, source_labels, teacher_pred, teacher_conf,
                         student_pred, num_classes, lam=0.5, conf_thr=0.95,
                         max_per_class=16):
    """Return (mixed_features, source_labels) or None when no class is usable."""
    with torch.no_grad():
        feat_s = feat[:src_size]
        feat_t = feat[src_size:]
        ys = source_labels.detach()
        pred = teacher_pred.detach()
        conf = teacher_conf.detach()
        spred = student_pred.detach()
        high = (conf >= float(conf_thr)) & (spred == pred)

        mixed, labels = [], []
        for c in range(int(num_classes)):
            sel_t = ((pred == c) & high).nonzero(as_tuple=True)[0]
            if sel_t.numel() == 0:
                continue
            idx_s = (ys == c).nonzero(as_tuple=True)[0]
            if idx_s.numel() == 0:
                continue
            k = min(int(idx_s.numel()), int(max_per_class))
            ps = idx_s[torch.randperm(idx_s.numel(), device=idx_s.device)[:k]]
            pt = sel_t[torch.randint(0, sel_t.numel(), (k,), device=sel_t.device)]
            mixed.append(float(lam) * feat_s[ps] + (1.0 - float(lam)) * feat_t[pt])
            labels.append(torch.full((k,), c, device=feat.device, dtype=torch.long))
    if not mixed:
        return None
    return torch.cat(mixed, 0), torch.cat(labels, 0)
