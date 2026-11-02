"""Loss helpers and epoch ramp-up schedules used by DCL experiments."""

import torch
import torch.nn.functional as F

from loss.bnm import BNM

def compute_bnm_loss(logits_t1, logits_t2, weight=1.0):
    prob_t1 = F.softmax(logits_t1, dim=1)
    prob_t2 = F.softmax(logits_t2, dim=1)
    if weight <= 0:
        return logits_t1.new_tensor(0.0), prob_t1, prob_t2
    return weight * (BNM(None, prob_t1) + BNM(None, prob_t2)), prob_t1, prob_t2

def get_rampup_weight(ep, start_epoch, rampup_epochs, max_weight):
    if max_weight <= 0 or ep < start_epoch:
        return 0.0
    if rampup_epochs <= 0:
        return max_weight
    progress = (ep - start_epoch + 1) / float(rampup_epochs)
    return max_weight * min(1.0, max(0.0, progress))

def get_bnm_weight(ep, step, args):
    if args.bnm_scope == 'none' or step not in args.bnm_scope or ep < args.bnm_start_epoch:
        return 0.0
    return get_rampup_weight(ep, args.bnm_start_epoch, args.bnm_rampup_epochs, args.bnm_weight)

def get_target_contrastive_weight(ep, args):
    return get_rampup_weight(
        ep,
        args.target_contrast_start_epoch,
        args.target_contrast_rampup_epochs,
        args.target_contrastive_weight
    )

def get_target_contrastive_threshold(ep, args):
    if ep <= args.target_contrast_start_epoch:
        return args.target_contrast_threshold_start
    if args.target_contrast_rampup_epochs <= 0:
        return args.target_contrast_threshold_end
    progress = (ep - args.target_contrast_start_epoch + 1) / float(args.target_contrast_rampup_epochs)
    progress = min(1.0, max(0.0, progress))
    return args.target_contrast_threshold_start + progress * (
        args.target_contrast_threshold_end - args.target_contrast_threshold_start
    )

def compute_projection_contrastive_loss(teacher_proj, student_proj, labels, contrastive_loss_func):
    all_features = torch.cat([teacher_proj.unsqueeze(1), student_proj.unsqueeze(1)], dim=1)
    return contrastive_loss_func(all_features, labels)
