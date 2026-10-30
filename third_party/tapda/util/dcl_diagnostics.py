"""Batch and epoch diagnostic summaries for target pseudo labels."""

import torch
import torch.nn.functional as F

def create_target_diagnostic_state(num_classes):
    return {
        'num_classes': int(num_classes),
        'pseudo_count': torch.zeros(num_classes, dtype=torch.float64),
        'pseudo_conf_sum': torch.zeros(num_classes, dtype=torch.float64),
        'selected_count': torch.zeros(num_classes, dtype=torch.float64),
        'selected_conf_sum': torch.zeros(num_classes, dtype=torch.float64),
        'disagreement_sum': torch.zeros(num_classes, dtype=torch.float64),
        'disagreement_count': torch.zeros(num_classes, dtype=torch.float64),
        'teacher_ensemble_agree_sum': torch.zeros(num_classes, dtype=torch.float64),
        'teacher_ensemble_agree_count': torch.zeros(num_classes, dtype=torch.float64),
        'pseudo_acc_true_sum': torch.zeros(num_classes, dtype=torch.float64),
        'pseudo_acc_true_count': torch.zeros(num_classes, dtype=torch.float64),
        'pseudo_precision_pred_sum': torch.zeros(num_classes, dtype=torch.float64),
        'pseudo_precision_pred_count': torch.zeros(num_classes, dtype=torch.float64),
    }

def _scatter_add_cpu(target, labels, values):
    if labels.numel() == 0:
        return
    target.scatter_add_(0, labels.detach().cpu().long(), values.detach().cpu().double())

def update_target_diagnostic_state(
    state,
    pseudo_labels,
    pseudo_probs,
    target_contrastive_stats,
    logits_t1=None,
    logits_t2=None,
    target_labels=None
):
    if state is None:
        return

    num_classes = state['num_classes']
    pseudo_labels_cpu = pseudo_labels.detach().cpu().long()
    pseudo_probs_cpu = pseudo_probs.detach().cpu().double()
    valid_pseudo_mask = (pseudo_labels_cpu >= 0) & (pseudo_labels_cpu < num_classes)
    pseudo_labels_cpu = pseudo_labels_cpu[valid_pseudo_mask]
    pseudo_probs_cpu = pseudo_probs_cpu[valid_pseudo_mask]
    if pseudo_labels_cpu.numel() == 0:
        return

    pseudo_counts = torch.bincount(pseudo_labels_cpu, minlength=num_classes).double()
    state['pseudo_count'] += pseudo_counts
    _scatter_add_cpu(state['pseudo_conf_sum'], pseudo_labels_cpu, pseudo_probs_cpu)

    selected_per_class = torch.tensor(
        target_contrastive_stats.get('selected_per_class', [0] * num_classes),
        dtype=torch.float64
    )
    selected_conf_per_class = torch.tensor(
        target_contrastive_stats.get('selected_conf_per_class', [0.0] * num_classes),
        dtype=torch.float64
    )
    if selected_per_class.numel() == num_classes:
        state['selected_count'] += selected_per_class
        state['selected_conf_sum'] += selected_per_class * selected_conf_per_class

    if logits_t1 is not None and logits_t2 is not None:
        prob_t1 = F.softmax(logits_t1.detach(), dim=1)
        prob_t2 = F.softmax(logits_t2.detach(), dim=1)
        disagreement = torch.mean(torch.abs(prob_t1 - prob_t2), dim=1).detach().cpu().double()
        disagreement = disagreement[valid_pseudo_mask]
        _scatter_add_cpu(state['disagreement_sum'], pseudo_labels_cpu, disagreement)
        state['disagreement_count'] += pseudo_counts

        ensemble_pred = (0.5 * (prob_t1 + prob_t2)).argmax(dim=1).detach().cpu().long()
        ensemble_pred = ensemble_pred[valid_pseudo_mask]
        teacher_ensemble_agree = (ensemble_pred == pseudo_labels_cpu).double()
        _scatter_add_cpu(state['teacher_ensemble_agree_sum'], pseudo_labels_cpu, teacher_ensemble_agree)
        state['teacher_ensemble_agree_count'] += pseudo_counts

    if target_labels is not None:
        target_labels_cpu = target_labels.detach().cpu().long()
        target_labels_cpu = target_labels_cpu[valid_pseudo_mask]
        valid_target_mask = (target_labels_cpu >= 0) & (target_labels_cpu < num_classes)
        if valid_target_mask.any():
            true_labels = target_labels_cpu[valid_target_mask]
            pred_labels = pseudo_labels_cpu[valid_target_mask]
            correct = (pred_labels == true_labels).double()
            true_counts = torch.bincount(true_labels, minlength=num_classes).double()
            pred_counts = torch.bincount(pred_labels, minlength=num_classes).double()
            state['pseudo_acc_true_count'] += true_counts
            state['pseudo_precision_pred_count'] += pred_counts
            _scatter_add_cpu(state['pseudo_acc_true_sum'], true_labels, correct)
            _scatter_add_cpu(state['pseudo_precision_pred_sum'], pred_labels, correct)

def _safe_divide(numerator, denominator):
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=1.0),
        torch.zeros_like(numerator)
    )

def _format_int_list(values):
    return '[' + ','.join(str(int(round(float(x)))) for x in values) + ']'

def _format_float_list(values, precision=2):
    return '[' + ','.join(f'{float(x):.{precision}f}' for x in values) + ']'

def format_target_diagnostic_summary(ep, state):
    pseudo_conf = _safe_divide(state['pseudo_conf_sum'], state['pseudo_count'])
    selected_conf = _safe_divide(state['selected_conf_sum'], state['selected_count'])
    disagreement = _safe_divide(state['disagreement_sum'], state['disagreement_count'])
    teacher_ensemble_agree = _safe_divide(
        state['teacher_ensemble_agree_sum'],
        state['teacher_ensemble_agree_count']
    )
    pseudo_acc_true = _safe_divide(state['pseudo_acc_true_sum'], state['pseudo_acc_true_count'])
    pseudo_precision_pred = _safe_divide(
        state['pseudo_precision_pred_sum'],
        state['pseudo_precision_pred_count']
    )

    return (
        'Diag Ep: {} PseudoCnt: {} PseudoConf: {} SelCnt: {} SelConf: {} '
        'DisByPseudo: {} TeacherEnsAgree: {} PseudoAccByTrue: {} PseudoPrecByPred: {}'
    ).format(
        ep,
        _format_int_list(state['pseudo_count'].tolist()),
        _format_float_list(pseudo_conf.tolist(), precision=2),
        _format_int_list(state['selected_count'].tolist()),
        _format_float_list(selected_conf.tolist(), precision=2),
        _format_float_list(disagreement.tolist(), precision=3),
        _format_float_list(teacher_ensemble_agree.tolist(), precision=2),
        _format_float_list(pseudo_acc_true.tolist(), precision=2),
        _format_float_list(pseudo_precision_pred.tolist(), precision=2),
    )
