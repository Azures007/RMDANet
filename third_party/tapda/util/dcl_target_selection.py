"""Target pseudo-label selection, quota adaptation, and prototype scoring."""

import torch
import torch.nn.functional as F

from util.dcl_loss_utils import (
    compute_projection_contrastive_loss,
    get_target_contrastive_threshold,
    get_target_contrastive_weight,
)

def get_target_contrastive_selection_mode(args):
    return getattr(args, 'target_contrast_selection', 'threshold')

def get_target_contrastive_topk_per_class(args):
    return max(0, int(getattr(args, 'target_contrast_topk_per_class', 0)))

def get_target_class_quota_mode(args):
    return getattr(args, 'target_class_quota_mode', 'none')

def use_target_class_quota(args):
    return get_target_class_quota_mode(args) != 'none'

def get_target_class_quota_start_epoch(args):
    return int(getattr(args, 'target_class_quota_start_epoch', getattr(args, 'target_contrast_start_epoch', 1)))

def get_target_class_quota_momentum(args):
    return min(0.9999, max(0.0, float(getattr(args, 'target_class_quota_momentum', 0.9))))

def get_target_class_quota_max_extra(args):
    return max(0, int(getattr(args, 'target_class_quota_max_extra', 2)))

def get_target_class_quota_difficulty_weight(args):
    return min(1.0, max(0.0, float(getattr(args, 'target_class_quota_difficulty_weight', 0.7))))

def use_target_prototype_filter(args):
    return bool(getattr(args, 'target_prototype_filter', False))

def use_target_prototype_scoring(args):
    return bool(getattr(args, 'target_prototype_scoring', False))

def get_target_prototype_momentum(args):
    return min(0.9999, max(0.0, float(getattr(args, 'target_prototype_momentum', 0.9))))

def get_target_prototype_margin(args):
    return max(0.0, float(getattr(args, 'target_prototype_margin', 0.0)))

def get_target_prototype_score_weight(args):
    return max(0.0, float(getattr(args, 'target_prototype_score_weight', 0.25)))

def create_target_class_quota_state(num_classes, device, args):
    return {
        'enabled': use_target_class_quota(args),
        'difficulty_ema': torch.zeros(num_classes, device=device),
        'freq_ema': torch.zeros(num_classes, device=device),
        'seen': torch.zeros(num_classes, device=device),
    }

def create_source_prototype_state(num_classes, device, args):
    return {
        'enabled': use_target_prototype_filter(args) or use_target_prototype_scoring(args),
        'prototypes': None,
        'seen': torch.zeros(num_classes, device=device),
    }

@torch.no_grad()
def update_source_prototype_state(prototype_state, source_projections, source_labels, num_classes, args):
    if prototype_state is None or not prototype_state.get('enabled', False):
        return
    if source_projections is None or source_labels is None:
        return

    source_projections = F.normalize(source_projections.detach().float(), dim=1)
    source_labels = source_labels.detach().long()
    valid_mask = (source_labels >= 0) & (source_labels < num_classes)
    if valid_mask.sum().item() == 0:
        return

    source_projections = source_projections[valid_mask]
    source_labels = source_labels[valid_mask]
    if prototype_state['prototypes'] is None:
        prototype_state['prototypes'] = torch.zeros(
            num_classes,
            source_projections.size(1),
            device=source_projections.device,
            dtype=source_projections.dtype
        )

    momentum = get_target_prototype_momentum(args)
    valid_classes = torch.unique(source_labels)
    for class_idx in valid_classes.tolist():
        class_mask = source_labels == class_idx
        batch_proto = F.normalize(source_projections[class_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        if prototype_state['seen'][class_idx] <= 0:
            prototype_state['prototypes'][class_idx] = batch_proto
        else:
            prototype_state['prototypes'][class_idx] = F.normalize(
                momentum * prototype_state['prototypes'][class_idx] + (1.0 - momentum) * batch_proto,
                dim=0
            )
        prototype_state['seen'][class_idx] += class_mask.sum()

def compute_confusion_aware_class_quotas(pseudo_labels, pseudo_probs, logits_t1, logits_t2, quota_state, ep, args):
    base_topk = get_target_contrastive_topk_per_class(args)
    stats = {
        'quota_mode': get_target_class_quota_mode(args),
        'quota_active_classes': 0,
        'quota_avg': float(base_topk),
        'quota_max': int(base_topk),
        'quota_difficulty_mean': 0.0,
        'quota_scarcity_mean': 0.0,
    }
    if quota_state is None or not quota_state.get('enabled', False):
        return None, stats
    if stats['quota_mode'] == 'none' or ep < get_target_class_quota_start_epoch(args):
        return None, stats
    if stats['quota_mode'] != 'confusion_ema':
        raise ValueError(f'Unsupported target class quota mode: {stats["quota_mode"]}')
    if base_topk <= 0:
        return None, stats

    prob_t1 = F.softmax(logits_t1.detach(), dim=1)
    prob_t2 = F.softmax(logits_t2.detach(), dim=1)
    discrepancy = torch.mean(torch.abs(prob_t1 - prob_t2), dim=1)
    uncertainty = 1.0 - pseudo_probs.detach()
    sample_difficulty = 0.5 * (discrepancy + uncertainty)

    pseudo_labels = pseudo_labels.detach().long()
    class_counts = torch.bincount(pseudo_labels, minlength=quota_state['seen'].numel()).float()
    total_count = class_counts.sum().clamp(min=1.0)
    batch_freq = class_counts / total_count
    momentum = get_target_class_quota_momentum(args)

    valid_batch_classes = torch.nonzero(class_counts > 0, as_tuple=False).squeeze(1)
    for class_idx in valid_batch_classes.tolist():
        class_mask = pseudo_labels == class_idx
        batch_difficulty = sample_difficulty[class_mask].mean()
        if quota_state['seen'][class_idx] <= 0:
            quota_state['difficulty_ema'][class_idx] = batch_difficulty
            quota_state['freq_ema'][class_idx] = batch_freq[class_idx]
        else:
            quota_state['difficulty_ema'][class_idx] = (
                momentum * quota_state['difficulty_ema'][class_idx] + (1.0 - momentum) * batch_difficulty
            )
            quota_state['freq_ema'][class_idx] = (
                momentum * quota_state['freq_ema'][class_idx] + (1.0 - momentum) * batch_freq[class_idx]
            )
        quota_state['seen'][class_idx] += class_counts[class_idx]

    seen_mask = quota_state['seen'] > 0
    active_classes = torch.nonzero(seen_mask, as_tuple=False).squeeze(1)
    stats['quota_active_classes'] = int(active_classes.numel())
    if active_classes.numel() < 2:
        return None, stats

    difficulty = quota_state['difficulty_ema'][active_classes]
    freq = quota_state['freq_ema'][active_classes]

    if difficulty.numel() == 0:
        return None, stats

    diff_range = difficulty.max() - difficulty.min()
    if diff_range <= 1e-6:
        difficulty_norm = torch.zeros_like(difficulty)
    else:
        difficulty_norm = (difficulty - difficulty.min()) / diff_range

    freq_range = freq.max() - freq.min()
    if freq_range <= 1e-6:
        scarcity_norm = torch.zeros_like(freq)
    else:
        scarcity_norm = 1.0 - (freq - freq.min()) / freq_range

    difficulty_weight = get_target_class_quota_difficulty_weight(args)
    quota_score = difficulty_weight * difficulty_norm + (1.0 - difficulty_weight) * scarcity_norm
    quota_extra = torch.round(quota_score * get_target_class_quota_max_extra(args)).long()

    per_class_topk = torch.full_like(quota_state['seen'], fill_value=base_topk, dtype=torch.long)
    per_class_topk[active_classes] = base_topk + quota_extra

    stats['quota_avg'] = float(per_class_topk[active_classes].float().mean().item())
    stats['quota_max'] = int(per_class_topk[active_classes].max().item())
    stats['quota_difficulty_mean'] = float(difficulty.mean().item())
    stats['quota_scarcity_mean'] = float(scarcity_norm.mean().item())
    return per_class_topk, stats

def get_target_contrastive_candidate_indices(selection_scores, ep, args):
    threshold = get_target_contrastive_threshold(ep, args)
    selected_mask = selection_scores >= threshold
    selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(1)
    return selected_indices, threshold

def select_topk_per_class(indices, pseudo_labels, selection_scores, num_classes, topk_per_class, per_class_topk=None):
    if indices.numel() == 0 or topk_per_class <= 0:
        return indices

    selected_chunks = []
    for class_idx in range(num_classes):
        class_mask = pseudo_labels[indices] == class_idx
        class_indices = indices[class_mask]
        if class_indices.numel() == 0:
            continue
        class_topk = topk_per_class
        if per_class_topk is not None:
            class_topk = max(0, int(per_class_topk[class_idx].item()))
        if class_topk <= 0:
            continue
        if class_indices.numel() <= class_topk:
            selected_chunks.append(class_indices)
            continue

        class_scores = selection_scores[class_indices]
        _, topk_pos = torch.topk(class_scores, k=class_topk, largest=True, sorted=False)
        selected_chunks.append(class_indices[topk_pos])

    if not selected_chunks:
        return indices.new_empty((0,), dtype=torch.long)

    selected_indices = torch.cat(selected_chunks, dim=0)
    return torch.sort(selected_indices).values

def filter_target_contrastive_pairs(selected_indices, pseudo_labels, num_classes):
    if selected_indices.numel() == 0:
        return selected_indices

    selected_labels = pseudo_labels[selected_indices]
    class_counts = torch.bincount(selected_labels, minlength=num_classes)
    pair_mask = class_counts[selected_labels] >= 2
    return selected_indices[pair_mask]

def filter_target_by_source_prototypes(
    selected_indices,
    pseudo_labels,
    target_projections,
    prototype_state,
    args,
    num_classes
):
    stats = {
        'prototype_filter': 'off',
        'prototype_before': int(selected_indices.numel()),
        'prototype_after': int(selected_indices.numel()),
        'prototype_keep_rate': 1.0,
        'prototype_valid_classes': 0,
    }
    if not use_target_prototype_filter(args):
        return selected_indices, stats

    stats['prototype_filter'] = 'source_ema'
    if selected_indices.numel() == 0:
        stats['prototype_keep_rate'] = 0.0
        return selected_indices, stats
    if (
        prototype_state is None
        or prototype_state.get('prototypes') is None
        or target_projections is None
    ):
        empty = selected_indices.new_empty((0,), dtype=torch.long)
        stats['prototype_after'] = 0
        stats['prototype_keep_rate'] = 0.0
        return empty, stats

    prototypes = prototype_state['prototypes'].detach()
    seen = prototype_state.get('seen')
    if seen is None:
        valid_prototypes = torch.ones(num_classes, device=prototypes.device, dtype=torch.bool)
    else:
        valid_prototypes = seen.detach() > 0
    stats['prototype_valid_classes'] = int(valid_prototypes.sum().item())
    if valid_prototypes.sum().item() == 0:
        empty = selected_indices.new_empty((0,), dtype=torch.long)
        stats['prototype_after'] = 0
        stats['prototype_keep_rate'] = 0.0
        return empty, stats

    selected_labels = pseudo_labels[selected_indices].detach().long()
    valid_label_mask = (selected_labels >= 0) & (selected_labels < num_classes)
    safe_labels = selected_labels.clamp(0, num_classes - 1)

    target_projections = F.normalize(target_projections.detach().float(), dim=1)
    prototypes = F.normalize(prototypes.float(), dim=1)
    similarity = torch.matmul(target_projections[selected_indices], prototypes.t())
    similarity[:, ~valid_prototypes] = -float('inf')
    nearest_proto_labels = torch.argmax(similarity, dim=1)

    keep_mask = valid_label_mask & valid_prototypes[safe_labels] & (nearest_proto_labels == selected_labels)

    margin = get_target_prototype_margin(args)
    if margin > 0.0 and valid_prototypes.sum().item() > 1:
        pseudo_similarity = similarity.gather(1, safe_labels.view(-1, 1)).squeeze(1)
        alt_similarity = similarity.clone()
        alt_similarity[torch.arange(alt_similarity.size(0), device=alt_similarity.device), safe_labels] = -float('inf')
        best_alt_similarity = alt_similarity.max(dim=1).values
        keep_mask = keep_mask & ((pseudo_similarity - best_alt_similarity) >= margin)

    filtered_indices = selected_indices[keep_mask]
    stats['prototype_after'] = int(filtered_indices.numel())
    stats['prototype_keep_rate'] = (
        float(filtered_indices.numel()) / float(max(1, selected_indices.numel()))
    )
    return filtered_indices, stats

def compute_target_prototype_ranking_scores(
    pseudo_probs,
    pseudo_labels,
    target_projections,
    prototype_state,
    args,
    num_classes
):
    stats = {
        'prototype_scoring': 'off',
        'prototype_score_weight': 0.0,
        'prototype_score_valid': 0,
        'prototype_score_valid_classes': 0,
        'prototype_margin_mean': 0.0,
        'prototype_score_mean': float(pseudo_probs.detach().float().mean().item()) if pseudo_probs.numel() > 0 else 0.0,
    }
    if not use_target_prototype_scoring(args):
        return None, stats

    stats['prototype_scoring'] = 'source_margin'
    score_weight = get_target_prototype_score_weight(args)
    stats['prototype_score_weight'] = score_weight
    if (
        score_weight <= 0.0
        or prototype_state is None
        or prototype_state.get('prototypes') is None
        or target_projections is None
        or pseudo_probs.numel() == 0
    ):
        return pseudo_probs.detach().float(), stats

    prototypes = prototype_state['prototypes'].detach()
    seen = prototype_state.get('seen')
    if seen is None:
        valid_prototypes = torch.ones(num_classes, device=prototypes.device, dtype=torch.bool)
    else:
        valid_prototypes = seen.detach() > 0
    stats['prototype_score_valid_classes'] = int(valid_prototypes.sum().item())
    if valid_prototypes.sum().item() < 2:
        return pseudo_probs.detach().float(), stats

    pseudo_labels = pseudo_labels.detach().long()
    valid_label_mask = (pseudo_labels >= 0) & (pseudo_labels < num_classes)
    safe_labels = pseudo_labels.clamp(0, num_classes - 1)
    valid_score_mask = valid_label_mask & valid_prototypes[safe_labels]
    if valid_score_mask.sum().item() == 0:
        return pseudo_probs.detach().float(), stats

    target_projections = F.normalize(target_projections.detach().float(), dim=1)
    prototypes = F.normalize(prototypes.float(), dim=1)
    similarity = torch.matmul(target_projections, prototypes.t())
    similarity[:, ~valid_prototypes] = -float('inf')

    pseudo_similarity = similarity.gather(1, safe_labels.view(-1, 1)).squeeze(1)
    alt_similarity = similarity.clone()
    alt_similarity[torch.arange(alt_similarity.size(0), device=alt_similarity.device), safe_labels] = -float('inf')
    best_alt_similarity = alt_similarity.max(dim=1).values
    proto_margin = pseudo_similarity - best_alt_similarity
    valid_score_mask = valid_score_mask & torch.isfinite(proto_margin)

    ranking_scores = pseudo_probs.detach().float().clone()
    if valid_score_mask.sum().item() > 0:
        score_multiplier = (1.0 + score_weight * proto_margin[valid_score_mask]).clamp(min=0.0, max=2.0)
        ranking_scores[valid_score_mask] = ranking_scores[valid_score_mask] * score_multiplier
        stats['prototype_score_valid'] = int(valid_score_mask.sum().item())
        stats['prototype_margin_mean'] = float(proto_margin[valid_score_mask].mean().item())
        stats['prototype_score_mean'] = float(ranking_scores[valid_score_mask].mean().item())
    return ranking_scores, stats

def summarize_selected_target_classes(selected_indices, pseudo_labels, num_classes):
    if selected_indices.numel() == 0:
        return [0] * num_classes

    selected_labels = pseudo_labels[selected_indices]
    class_counts = torch.bincount(selected_labels, minlength=num_classes)
    return [int(x) for x in class_counts.tolist()]

def summarize_selected_target_confidence(selected_indices, pseudo_labels, selection_scores, num_classes):
    if selected_indices.numel() == 0:
        return [0.0] * num_classes

    selected_labels = pseudo_labels[selected_indices].detach().long()
    selected_scores = selection_scores[selected_indices].detach().float()
    score_sums = torch.zeros(num_classes, device=selected_scores.device)
    score_sums.scatter_add_(0, selected_labels, selected_scores)
    class_counts = torch.bincount(selected_labels, minlength=num_classes).float()
    class_means = score_sums / class_counts.clamp(min=1.0)
    class_means = torch.where(class_counts > 0, class_means, torch.zeros_like(class_means))
    return [float(x) for x in class_means.tolist()]

def select_target_contrastive_samples(
    pseudo_labels,
    pseudo_probs,
    ep,
    args,
    num_classes,
    selection_scores=None,
    ranking_scores=None,
    per_class_topk=None,
    prototype_filter_projections=None,
    source_prototype_state=None
):
    selection_mode = get_target_contrastive_selection_mode(args)
    if selection_scores is None:
        selection_scores = pseudo_probs
    if ranking_scores is None:
        ranking_scores = selection_scores
    selected_indices, threshold = get_target_contrastive_candidate_indices(selection_scores, ep, args)
    selected_indices, prototype_stats = filter_target_by_source_prototypes(
        selected_indices,
        pseudo_labels,
        prototype_filter_projections,
        source_prototype_state,
        args,
        num_classes
    )

    if selection_mode == 'class_topk':
        selected_indices = select_topk_per_class(
            selected_indices,
            pseudo_labels,
            ranking_scores,
            num_classes,
            get_target_contrastive_topk_per_class(args),
            per_class_topk=per_class_topk
        )
    elif selection_mode != 'threshold':
        raise ValueError(f'Unsupported target contrastive selection mode: {selection_mode}')

    selected_indices = filter_target_contrastive_pairs(selected_indices, pseudo_labels, num_classes)
    selected_per_class = summarize_selected_target_classes(selected_indices, pseudo_labels, num_classes)
    metadata = {
        'threshold': threshold,
        'selection_mode': selection_mode,
        'selected_per_class': selected_per_class,
        'selected_classes': int(sum(count > 0 for count in selected_per_class)),
    }
    metadata.update(prototype_stats)
    return selected_indices, metadata

def compute_target_contrastive_loss(
    teacher_proj,
    student_proj,
    pseudo_labels,
    pseudo_probs,
    ep,
    num_classes,
    args,
    contrastive_loss_func,
    selection_scores=None,
    per_class_topk=None,
    prototype_filter_projections=None,
    source_prototype_state=None
):
    weight = get_target_contrastive_weight(ep, args)
    zero = student_proj.new_tensor(0.0)
    threshold = get_target_contrastive_threshold(ep, args)
    stats = {
        'weight': weight,
        'threshold': threshold,
        'selection_mode': get_target_contrastive_selection_mode(args),
        'selected': 0,
        'selected_classes': 0,
        'selected_per_class': [0] * num_classes,
        'selected_conf_per_class': [0.0] * num_classes,
        'avg_conf': 0.0,
        'prototype_filter': 'off',
        'prototype_before': 0,
        'prototype_after': 0,
        'prototype_keep_rate': 1.0,
        'prototype_valid_classes': 0,
        'prototype_scoring': 'off',
        'prototype_score_weight': 0.0,
        'prototype_score_valid': 0,
        'prototype_score_valid_classes': 0,
        'prototype_margin_mean': 0.0,
        'prototype_score_mean': 0.0,
    }
    if weight <= 0:
        return zero, stats

    ranking_scores, prototype_scoring_stats = compute_target_prototype_ranking_scores(
        pseudo_probs,
        pseudo_labels,
        prototype_filter_projections,
        source_prototype_state,
        args,
        num_classes
    )
    selected_indices, selection_metadata = select_target_contrastive_samples(
        pseudo_labels,
        pseudo_probs,
        ep,
        args,
        num_classes,
        selection_scores=selection_scores,
        ranking_scores=ranking_scores,
        per_class_topk=per_class_topk,
        prototype_filter_projections=prototype_filter_projections,
        source_prototype_state=source_prototype_state
    )
    stats['threshold'] = selection_metadata['threshold']
    stats['selection_mode'] = selection_metadata['selection_mode']
    stats['selected'] = int(selected_indices.numel())
    stats['selected_classes'] = selection_metadata['selected_classes']
    stats['selected_per_class'] = selection_metadata['selected_per_class']
    stats['prototype_filter'] = selection_metadata['prototype_filter']
    stats['prototype_before'] = selection_metadata['prototype_before']
    stats['prototype_after'] = selection_metadata['prototype_after']
    stats['prototype_keep_rate'] = selection_metadata['prototype_keep_rate']
    stats['prototype_valid_classes'] = selection_metadata['prototype_valid_classes']
    stats.update(prototype_scoring_stats)
    if stats['prototype_valid_classes'] == 0:
        stats['prototype_valid_classes'] = stats.get('prototype_score_valid_classes', 0)
    selected_probs = None
    if selected_indices.numel() > 0:
        if selection_scores is None:
            selected_probs = pseudo_probs[selected_indices]
            selected_score_source = pseudo_probs
        else:
            selected_probs = selection_scores[selected_indices]
            selected_score_source = selection_scores
        stats['avg_conf'] = float(selected_probs.mean().item())
        stats['selected_conf_per_class'] = summarize_selected_target_confidence(
            selected_indices,
            pseudo_labels,
            selected_score_source,
            num_classes
        )
    if selected_indices.numel() < args.target_contrast_min_samples:
        return zero, stats

    selected_labels = pseudo_labels[selected_indices]
    target_contrastive = compute_projection_contrastive_loss(
        teacher_proj[selected_indices],
        student_proj[selected_indices],
        selected_labels,
        contrastive_loss_func
    )
    return weight * selected_probs.mean() * target_contrastive, stats
