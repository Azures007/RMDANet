"""v3.0 component ports for the TAPDA screening (all additive, default off).

Ported from v3.0 ``core/trainer.py`` and ``core/v71_controls.py``:
teacher consistency, style-pair loss, source metric loss, class-wise CNT
shift, independent spectral view JS consistency, and a reliability-weighted
LMMD.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def confidence_consistency_loss(teacher_prob, student_logits, teacher_proj, student_proj):
    """Weighted weak-strong consistency: KL(teacher || student) + projection cosine."""
    teacher_prob = teacher_prob.detach().float().clamp_min(1e-8)
    log_student = F.log_softmax(student_logits.float(), dim=1)
    consistency = -(teacher_prob * log_student).sum(dim=1).mean()
    projection = 1.0 - F.cosine_similarity(
        student_proj.float(), teacher_proj.detach().float(), dim=1
    ).mean()
    return consistency + 0.1 * projection


def style_pair_loss(source_features, target_features, eps=1e-6):
    """v3.0 style-pair loss: match per-channel mean/std of source and target features."""
    source = source_features.float()
    target = target_features.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_std = source.std(dim=0, unbiased=False).clamp_min(eps)
    target_std = target.std(dim=0, unbiased=False).clamp_min(eps)
    dim = source_mean.numel()
    return ((source_mean - target_mean) ** 2).sum() / dim + (
        (source_std - target_std) ** 2
    ).sum() / dim


def source_metric_loss(features, labels, num_classes, margin=0.30, eps=1e-6):
    """v3.0 source metric loss: intra-class compactness + inter-class separation."""
    features = F.normalize(features.float(), dim=1)
    labels = labels.long()
    device = features.device
    class_means = []
    compactness = features.new_tensor(0.0)
    valid = 0
    for class_id in range(num_classes):
        mask = labels.eq(class_id)
        if int(mask.sum().item()) < 2:
            continue
        class_feature = features[mask]
        mean = F.normalize(class_feature.mean(dim=0), dim=0)
        class_means.append(mean)
        compactness = compactness + (1.0 - (class_feature * mean).sum(dim=1)).mean()
        valid += 1
    if valid == 0 or len(class_means) < 2:
        return features.new_tensor(0.0)
    compactness = compactness / valid
    means = torch.stack(class_means)
    similarity = means @ means.t()
    off_diagonal = similarity - torch.eye(means.size(0), device=device) * 2.0
    separation = F.relu(off_diagonal - (1.0 - margin)).mean()
    return compactness + separation


def classwise_cnt_shift_loss(source_prototypes, source_seen, target_features,
                             pseudo_labels, pseudo_prob, num_classes):
    """v3.0 class-wise CNT: per-class mean shift between source prototypes and target."""
    valid = source_seen > 0
    if int(valid.sum().item()) < 2:
        return target_features.new_tensor(0.0)
    features = F.normalize(target_features.float(), dim=1)
    proto_dim = int(source_prototypes.size(1))
    if features.size(1) != proto_dim:
        # Keep the comparison well-defined across projection/feature spaces.
        features = F.normalize(features[:, :proto_dim], dim=1)
    total = features.new_tensor(0.0)
    count = 0
    for class_id in range(num_classes):
        mask = pseudo_labels.eq(class_id) & (pseudo_prob >= 0.5)
        if int(mask.sum().item()) < 2 or not bool(valid[class_id].item()):
            continue
        target_mean = F.normalize(features[mask].mean(dim=0), dim=0)
        source_mean = F.normalize(source_prototypes[class_id].float(), dim=0)
        total = total + (1.0 - (target_mean * source_mean).sum()).clamp_min(0.0)
        count += 1
    if count == 0:
        return features.new_tensor(0.0)
    return total / count


def spectral_view_js_loss(base_prob, view_prob, eps=1e-8):
    """v3.0 independent spectral view: symmetric JS divergence between predictions."""
    base = base_prob.float().clamp_min(eps)
    view = view_prob.float().clamp_min(eps)
    mean = 0.5 * (base + view)
    return 0.5 * (
        (base * (base / mean).log()).sum(dim=1).mean()
        + (view * (view / mean).log()).sum(dim=1).mean()
    )


def weighted_lmmd_loss(source_features, source_labels, target_features,
                       target_labels, class_weights, num_classes, min_samples=4):
    """v3.0 reliability-weighted LMMD: class-wise MMD scaled by reliability weights."""
    source_features = source_features.float()
    target_features = target_features.float()
    source_labels = source_labels.long()
    target_labels = target_labels.long()
    total = source_features.new_tensor(0.0)
    weight_total = source_features.new_tensor(0.0)
    for class_id in range(num_classes):
        source_mask = source_labels.eq(class_id)
        target_mask = target_labels.eq(class_id)
        if int(source_mask.sum().item()) < min_samples or int(target_mask.sum().item()) < min_samples:
            continue
        source_class = source_features[source_mask]
        target_class = target_features[target_mask]
        source_kernel = source_class @ source_class.t() / source_features.size(1)
        target_kernel = target_class @ target_class.t() / source_features.size(1)
        cross_kernel = source_class @ target_class.t() / source_features.size(1)
        mmd = source_kernel.exp().mean() + target_kernel.exp().mean() - 2.0 * cross_kernel.exp().mean()
        weight = class_weights[class_id].detach() if class_weights is not None else source_features.new_tensor(1.0)
        total = total + weight * mmd
        weight_total = weight_total + weight
    if float(weight_total.item()) <= 0:
        return source_features.new_tensor(0.0)
    return total / weight_total
