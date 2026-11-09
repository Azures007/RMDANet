"""v7.1 reliability-aware alignment and label-free source/target safeguards.

No target ground truth is accepted by this module. All target evidence is
predictive/perturbational, not proof of correctness.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def reliability_lmmd(source_features, source_labels, target_features,
                     target_labels, stable_mask, reliability, num_classes,
                     min_samples=2, min_effective=1.5):
    """Source-vs-reliable-target class-conditional mean alignment.

    Skips a class with inadequate reliable support; NEVER falls back to all
    predicted labels. The pseudo labels are still noisy estimates.
    """
    sf, tf = source_features.float(), target_features.float()
    weights = reliability.detach().float().clamp(0, 1) * stable_mask.detach().float()
    total = sf.sum() * 0.0
    used = 0
    for c in range(num_classes):
        sm = source_labels == c
        tm = target_labels == c
        if int(sm.sum()) < min_samples or int((tm & stable_mask).sum()) < min_samples:
            continue
        w = weights[tm]
        if float(w.sum()) < min_effective:
            continue
        tmean = (tf[tm] * w[:, None]).sum(0) / w.sum().clamp_min(1e-8)
        smean = sf[sm].mean(0)
        total = total + (smean - tmean).square().sum()
        used += 1
    return total / used if used else total


@torch.no_grad()
def maturity_snapshot(memory, min_samples=12, min_classes=2):
    """Only unlabeled memory statistics; bounded, may be high when predictions wrong."""
    seen = memory.valid
    if not bool(seen.any()):
        return {'maturity': 0.0, 'eligible': 0, 'classes': 0,
                'stable_fraction': 0.0, 'flip_fraction': 1.0}
    stable = (seen & (memory.age >= max(2, memory.min_age)) &
              (memory.streak >= max(2, memory.min_streak)) &
              (memory.reliability_memory >= memory.reliability_threshold))
    eligible = int(stable.sum().item())
    labels = memory.prob_memory[stable].argmax(1) if eligible else torch.empty(0,device=seen.device,dtype=torch.long)
    classes = int(torch.unique(labels).numel())
    flip = float((memory.label_flip_count[seen] > 0).float().mean().item())
    stable_fraction = float(stable.sum().item() / max(1, seen.sum().item()))
    confidence = float(memory.reliability_memory[stable].mean().item()) if eligible else 0.0
    # A minimum of 2 classes and a nonzero label-free support is mandatory.
    if eligible < min_samples or classes < min(min_classes, memory.num_classes):
        maturity = 0.0
    else:
        support_term = min(1.0, eligible / max(min_samples * 3, 1))
        class_term = min(1.0, classes / max(2, memory.num_classes // 2))
        maturity = max(0.0, min(1.0, support_term * class_term *
                                (1.0 - 0.5 * flip) * confidence))
    return {'maturity': maturity, 'eligible': eligible, 'classes': classes,
            'stable_fraction': stable_fraction, 'flip_fraction': flip}


@torch.no_grad()
def independent_spectral_view(x, strength=0.025):
    """Random smooth wavelength response; independent of pseudo-label mask.

    This is a robustness probe, not an independent labeling oracle.
    """
    b, channels, _, _ = x.shape
    knots = torch.randn((b, 1, min(8, channels)), device=x.device, dtype=torch.float32)
    gains = F.interpolate(knots, size=channels, mode='linear', align_corners=False)
    gains = torch.tanh(gains).view(b, channels, 1, 1)
    return (x.float() * (1.0 + strength * gains)).to(x.dtype)


def js_agreement(p, q):
    p, q = p.detach().float().clamp_min(1e-7), q.detach().float().clamp_min(1e-7)
    middle = 0.5 * (p + q)
    js = 0.5 * ((p * (p.log() - middle.log())).sum(1) +
                (q * (q.log() - middle.log())).sum(1))
    return (1 - js / math.log(2)).clamp(0, 1)


@torch.no_grad()
def classwise_cnt_shift(source, target, source_labels, target_probabilities,
                        source_class_means, basis, scale, max_ratio=0.12,
                        min_support=3.0, target_confidence=0.65):
    """Prior-debiased, class-conditional transport without target labels.

    When a target class lacks confident soft support, falls back to a
    prior-debiased shared shift; when all support is absent, shifts are zero.
    Source means must be estimated ONLY from budgeted source TRAIN labels.
    """
    source_f = source.float(); target_f = target.float()
    sc = source_f[:, :, source.shape[-2]//2, source.shape[-1]//2]
    tc = target_f[:, :, target.shape[-2]//2, target.shape[-1]//2]
    k, channels = source_class_means.shape
    valid = torch.isfinite(source_class_means).all(1)
    if not valid.any():
        return source, 0.0, torch.zeros_like(source_class_means)
    source_hist = torch.bincount(source_labels, minlength=k).float()
    source_prior = source_hist / source_hist.sum().clamp_min(1)
    source_prior_mean = (source_class_means * source_prior[:,None]).sum(0)
    p = target_probabilities.detach().float().softmax(1) if target_probabilities is not None and (
        (target_probabilities.min() < 0) or (target_probabilities.max() > 1.0)) else target_probabilities
    if p is None:
        return source, 0.0, torch.zeros_like(source_class_means)
    p = p.clamp_min(0); p = p / p.sum(1, keepdim=True).clamp_min(1e-8)
    # No hidden target GT: pseudo soft membership only if teacher is confident.
    accept = p.max(1).values >= target_confidence
    p = p * accept[:,None]
    support = p.sum(0)
    sufficient = (support >= min_support) & valid
    if not sufficient.any():
        return source, 0.0, torch.zeros_like(source_class_means)
    tprior = p.sum(0) / p.sum().clamp_min(1e-8)
    # Remove class-prior-driven displacement before estimating global style.
    prior_displacement = ((tprior - source_prior)[:,None] * source_class_means).sum(0)
    global_shift = tc[accept].mean(0) - sc.mean(0) - prior_displacement
    directions = torch.zeros_like(source_class_means)
    for c in range(k):
        if sufficient[c]:
            tmean = (tc * p[:, c:c+1]).sum(0) / support[c].clamp_min(1e-8)
            directions[c] = 0.85 * (tmean - source_class_means[c]) + 0.15 * global_shift
    if basis.numel():
        q=basis.to(device=source.device, dtype=torch.float32)
        directions = directions - (directions @ q) @ q.T
    directions[~valid] = 0
    # Relative trust bound: measured on budgeted source spectra.
    median_norm = sc.norm(dim=1).median().clamp_min(1e-6)
    allowed = max_ratio * median_norm
    norms = directions.norm(dim=1).clamp_min(1e-8)
    directions = directions * (allowed/norms).clamp(max=1)[:,None]
    delta = directions[source_labels].clone()
    # Local direction correction, shrunk toward the class displacement so that
    # close source/target residual pairs can have different spectral shifts.
    for class_id in range(k):
        sm = source_labels.eq(class_id)
        if not sm.any() or not sufficient[class_id]:
            continue
        tm = p[:, class_id] > 0.65
        if not tm.any():
            continue
        src_residual = sc[sm] - source_class_means[class_id]
        tgt_residual = tc[tm] - ((tc * p[:, class_id:class_id+1]).sum(0)
                                         / support[class_id].clamp_min(1e-8))
        similarities = F.normalize(src_residual, dim=1) @ F.normalize(tgt_residual, dim=1).T
        nearest = tgt_residual[similarities.argmax(1)]
        correction = 0.15 * (nearest - src_residual)
        if basis.numel():
            correction = correction - (correction @ q) @ q.T
        correction_norm = correction.norm(dim=1).clamp_min(1e-8)
        correction = correction * (0.25 * allowed / correction_norm).clamp(max=1)[:,None]
        delta[sm] += correction
    norm = delta.norm(dim=1).clamp_min(1e-8)
    delta = delta * (allowed / norm).clamp(max=1)[:,None]
    delta = delta * scale[source_labels,None]
    pixel_valid = source_f.abs().sum(1, keepdim=True) > 1e-10
    augmented = source_f + delta[:,:,None,None] * pixel_valid
    relative = float((delta.norm(dim=1).mean()/median_norm).item())
    return augmented.to(source.dtype), relative, directions


@torch.no_grad()
def source_teacher_backtrack(teacher, source, augmented, labels,
                             tolerance=0.30, rounds=3):
    """Reduce untrusted source perturbations using labeled source teacher margins.

    A class label-preserving **test** for the current teacher, NOT a proof of
    semantic invariance. Returns tested augmented inputs and a safety mask.
    """
    from semantic_safety import judge_augmentation
    clean_logits = teacher(source).logits.detach()
    candidate = augmented
    good = torch.zeros_like(labels, dtype=torch.bool)
    for step in range(rounds + 1):
        logits = teacher(candidate).logits.detach()
        good, _ = judge_augmentation(clean_logits, logits, labels, tolerance)
        if good.all():
            break
        # Fallback toward the untouched source sample for failed candidates.
        if step < rounds:
            candidate = torch.where(good[:,None,None,None], candidate,
                                    source + 0.5 * (candidate - source))
    return candidate, good
