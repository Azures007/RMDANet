from __future__ import print_function

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import util.utils as utils
from loss.contrastive_loss import SupConLoss
from modules.HybridSN import SSSE_Contrastive
from modules.reliability_encoder import ReliabilityEncoderContrastive
from modules.cosine_head import CosineClassifierHead
from modules.prat_full import PhaseReliabilityResidualAmplitudeTransfer
from modules.memory_full import ReliabilityPrototypeMemory
from modules.v30_losses import (
    classwise_cnt_shift_loss,
    confidence_consistency_loss,
    source_metric_loss,
    spectral_view_js_loss,
    style_pair_loss,
    weighted_lmmd_loss,
)
from modules.basenet import *
from modules.teacher_cl import EMATeacher
from util.dcl_experiment_utils import (
    TeeLogger,
    compute_bnm_loss,
    compute_confusion_aware_class_quotas,
    compute_projection_contrastive_loss,
    compute_target_contrastive_loss,
    create_source_prototype_state,
    create_target_class_quota_state,
    create_target_diagnostic_state,
    create_experiment_dir,
    evaluate_results,
    format_target_diagnostic_summary,
    format_train_status,
    get_bnm_weight,
    save_dataset_result,
    save_run_metadata,
    save_summary,
    select_eval_logits,
    update_source_prototype_state,
    update_target_diagnostic_state,
    use_target_class_quota,
)
from util.dcl_loss_utils import get_rampup_weight
from util.utils import *

NUM_CLASSES = 7
N_BANDS = 102
PAVIA_DATA_PATHS = {
    'source_data': "datasets/Pavia/paviaU.mat",
    'source_label': "datasets/Pavia/paviaU_gt_7.mat",
    'target_data': "datasets/Pavia/pavia.mat",
    'target_label': "datasets/Pavia/pavia_gt_7.mat",
}
SEEDS = [1726, 1040, 1885, 1083, 1937, 1921, 1270, 1343, 1914, 1077]


_INSTABILITY_EMA = {'value': None}

# ---------------------------------------------------------------------------
# RGS diagnosis buffer (analysis only: stores ids, predictions, confidences).
# ---------------------------------------------------------------------------
_DIAG_BUFFER = []

_RCS_READY = {'model': None}


def _diag_reset():
    _DIAG_BUFFER.clear()


def _diag_add(ids, teacher, conf, student, pats_mask, rgs_idx):
    import numpy as _np
    with torch.no_grad():
        total = int(teacher.numel())
        rgs_mask = torch.zeros(total, dtype=torch.bool, device=teacher.device)
        if rgs_idx is not None and rgs_idx.numel() > 0:
            rgs_mask[rgs_idx] = True
        _DIAG_BUFFER.append({
            'id': ids.detach().cpu().numpy().astype('int32'),
            'teacher': teacher.detach().cpu().numpy().astype('int16'),
            'conf': conf.detach().cpu().numpy().astype('float32'),
            'student': student.detach().cpu().numpy().astype('int16'),
            'in_pats': pats_mask.detach().cpu().numpy().astype('bool'),
            'in_rgs': rgs_mask.detach().cpu().numpy().astype('bool'),
        })


def _diag_flush(path):
    import os as _os
    import numpy as _np
    if not _DIAG_BUFFER:
        return
    keys = list(_DIAG_BUFFER[0].keys())
    merged = {k: _np.concatenate([row[k] for row in _DIAG_BUFFER]) for k in keys}
    _os.makedirs(_os.path.dirname(path) or '.', exist_ok=True)
    _np.savez_compressed(path, **merged)
    _DIAG_BUFFER.clear()
    print(f"[rgs-diag] dumped {path} ({merged['id'].size} samples)")


def build_parser():
    parser = argparse.ArgumentParser(description='UP -> PC PDCL training')

    runtime_group = parser.add_argument_group('Training runtime')
    runtime_group.add_argument('--batch-size', type=int, default=32, metavar='N',
                               help='training mini-batch size')
    runtime_group.add_argument('--num_epoch', type=int, default=50, metavar='N',
                               help='number of training epochs for each seed')
    runtime_group.add_argument('--lr', type=float, default=0.1, metavar='LR',
                               help='learning rate used by the selected optimizer')
    runtime_group.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                               help='optimizer name: momentum, adam, adamw, or any other value for Adadelta')
    runtime_group.add_argument('--weight_decay', type=float, default=5e-4,
                               help='weight decay for all optimizers')
    runtime_group.add_argument('--label_smoothing', type=float, default=0.0,
                               help='label smoothing for the classification loss')
    runtime_group.add_argument('--lr_schedule', default='none', choices=['none', 'cosine'],
                               help='learning-rate schedule over epochs')
    runtime_group.add_argument('--num_k', type=int, default=3, metavar='K',
                               help='number of Step C generator updates per batch')
    runtime_group.add_argument('--gpu', default=0, type=int,
                               help='CUDA device id when CUDA is available')
    runtime_group.add_argument('--no-cuda', action='store_true', default=False,
                               help='disable CUDA-specific tensor moves')
    runtime_group.add_argument('--log-interval', type=int, default=5, metavar='N',
                               help='batches between training status prints')
    runtime_group.add_argument('--num_seeds', type=int, default=10,
                               help='number of seeds evaluated from SEEDS (default: all 10)')
    runtime_group.add_argument('--seed_offset', type=int, default=0,
                               help='index of the first seed taken from SEEDS')

    data_group = parser.add_argument_group('Data and model shape')
    data_group.add_argument('--input_norm', default='none', choices=['none', 'band'],
                            help='per-band z-score of both scenes (label-free recalibration)')
    runtime_group.add_argument('--target_repeats', default=1, type=int,
                               help='multiply the target batches per epoch (coverage scheduling)')
    data_group.add_argument('--fda', action='store_true', default=False,
                            help='Fourier/statistics target re-parameterization of source patches')
    data_group.add_argument('--fda_start_epoch', default=0, type=int,
                            help='epoch from which the source patches are re-parameterized')
    data_group.add_argument('--halfwidth', default=4, type=int,
                            help='half width of the spatial patch; patch_size = 2 * halfwidth + 1')
    data_group.add_argument('--backbone_impl', default='ssse', choices=['ssse', 'reliability'],
                            help='feature extractor: TAPDA SSSE (ssse) or v3.0 DualReliabilityEncoder (reliability)')
    data_group.add_argument('--classifier_impl', default='res', choices=['res', 'cosine'],
                            help='dual-classifier head: TAPDA ResClassifier (res) or v3.0 cosine head (cosine)')

    logging_group = parser.add_argument_group('Experiment logging')
    logging_group.add_argument('--log_results', action='store_true',
                               help='enable wandb logging in addition to local logs')
    logging_group.add_argument('--log_name', type=str, default='log',
                               help='wandb run name and default local run prefix')
    logging_group.add_argument('--output_root', type=str, default='runs',
                               help='root directory for local training logs')
    logging_group.add_argument('--dataset_name', type=str, default='UP_to_PC',
                               help='dataset-specific output directory name')
    logging_group.add_argument('--run_name', type=str, default='',
                               help='local run directory name; default: log_name + timestamp')
    logging_group.add_argument('--log_target_diagnostic', action='store_true', default=False,
                               help='print epoch-level per-class target pseudo-label diagnostics')

    teacher_group = parser.add_argument_group('EMA teacher and source contrast')
    teacher_group.add_argument('--alpha', default=0.9, type=float,
                               help='EMA momentum for the teacher model')
    teacher_group.add_argument('--pseudo_label_weight', default='prob',
                               help='target pseudo-label weighting mode used by EMATeacher')
    teacher_group.add_argument('--source_contrastive_weight', default=1.0, type=float,
                               help='weight for source supervised contrastive distillation')

    supcon_group = parser.add_argument_group('Supervised contrastive loss')
    supcon_group.add_argument('--supcon_temperature', default=0.1, type=float,
                              help='temperature passed to SupConLoss')
    supcon_group.add_argument('--supcon_contrast_mode', default='one', choices=['one', 'all'],
                              help='anchor mode passed to SupConLoss')

    bnm_group = parser.add_argument_group('Batch nuclear-norm maximization')
    bnm_group.add_argument('--bnm_weight', default=0.15, type=float,
                           help='maximum weight for batch nuclear-norm maximization loss')
    bnm_group.add_argument('--bnm_scope', default='A', choices=['none', 'A', 'AC', 'ABC'],
                           help='training steps that receive BNM loss')
    bnm_group.add_argument('--bnm_start_epoch', default=5, type=int,
                           help='epoch to start applying BNM loss')
    bnm_group.add_argument('--bnm_rampup_epochs', default=10, type=int,
                           help='epochs used to linearly ramp up BNM weight')

    target_contrast_group = parser.add_argument_group('Target contrastive distillation')
    target_contrast_group.add_argument('--target_contrastive_weight', default=0.2, type=float,
                                       help='maximum weight for target pseudo-label contrastive distillation')
    target_contrast_group.add_argument('--target_contrast_start_epoch', default=10, type=int,
                                       help='epoch to start applying target contrastive distillation')
    target_contrast_group.add_argument('--target_contrast_rampup_epochs', default=15, type=int,
                                       help='epochs used to linearly ramp up target contrastive weight')
    target_contrast_group.add_argument('--target_contrast_threshold_start', default=0.75, type=float,
                                       help='initial confidence threshold for target pseudo labels')
    target_contrast_group.add_argument('--target_contrast_threshold_end', default=0.98, type=float,
                                       help='final confidence threshold for target pseudo labels')
    target_contrast_group.add_argument('--target_contrast_selection', default='class_topk',
                                       choices=['threshold', 'class_topk'],
                                       help='target sample selection strategy for contrastive loss')
    target_contrast_group.add_argument('--target_contrast_topk_per_class', default=4, type=int,
                                       help='base number of high-confidence target samples kept per pseudo class')
    target_contrast_group.add_argument('--target_contrast_min_samples', default=8, type=int,
                                       help='minimum selected target samples required to compute target contrastive loss')
    target_contrast_group.add_argument('--log_target_contrast_detail', action='store_true', default=False,
                                       help='print target contrastive sample-selection details in training logs')

    quota_group = parser.add_argument_group('Adaptive target class quota')
    quota_group.add_argument('--target_class_quota_mode', default='none', choices=['none', 'confusion_ema'],
                             help='adaptive per-class quota mode for target class_topk selection')
    quota_group.add_argument('--target_class_quota_start_epoch', default=10, type=int,
                             help='epoch to start applying adaptive target class quotas')
    quota_group.add_argument('--target_class_quota_momentum', default=0.9, type=float,
                             help='EMA momentum for target class difficulty and frequency statistics')
    quota_group.add_argument('--target_class_quota_max_extra', default=1, type=int,
                             help='maximum extra target samples allocated beyond base top-k')
    quota_group.add_argument('--target_class_quota_difficulty_weight', default=0.7, type=float,
                             help='weight of class difficulty versus scarcity in adaptive quotas')
    quota_group.add_argument('--log_target_quota_detail', action='store_true', default=False,
                             help='print adaptive target class quota details in training logs')

    prototype_group = parser.add_argument_group('Prototype-aware target selection')
    prototype_group.add_argument('--target_prototype_filter', action='store_true', default=False,
                                 help='keep only candidates whose pseudo label matches the nearest source prototype')
    prototype_group.add_argument('--target_prototype_scoring', action='store_true', default=True,
                                 help='rerank target candidates with prototype-aware scores')
    prototype_group.add_argument('--target_prototype_score_weight', default=0.05, type=float,
                                 help='weight of source prototype margin in target candidate ranking scores')
    prototype_group.add_argument('--target_prototype_momentum', default=0.9, type=float,
                                 help='EMA momentum for source class prototypes')
    prototype_group.add_argument('--target_prototype_margin', default=0.0, type=float,
                                 help='optional cosine margin over the nearest non-pseudo source prototype')
    prototype_group.add_argument('--log_target_prototype_detail', action='store_true', default=False,
                                 help='print prototype-aware target selection details in training logs')

    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--eval_mode', default='ensemble', choices=['f1', 'f2', 'ensemble'],
                            help='prediction head used for reported OA/AA/Kappa')
    eval_group.add_argument('--eval_every_epoch', action='store_true', default=False,
                            help='evaluate after every epoch and keep the best target-test OA '
                                 '(this reproduces the protocol implicitly used by TAPDA paper numbers)')

    ours_group = parser.add_argument_group('Ours: reliability-gated pseudo-label supervision')
    ours_group.add_argument('--rcs', action='store_true', default=False,
                            help='risk-certified supplemental supervision (RCS)')
    ours_group.add_argument('--rcs_delta', default=0.10, type=float,
                            help='risk budget delta for the certified supervision set')
    ours_group.add_argument('--rcs_min_frac', default=0.05, type=float,
                            help='minimum fraction of the target batch to supervise')
    ours_group.add_argument('--rcs_max_frac', default=0.60, type=float,
                            help='maximum fraction of the target batch to supervise')
    ours_group.add_argument('--rcs_start_epoch', default=15, type=int,
                            help='epoch where the source calibration starts')
    ours_group.add_argument('--rcs_fit_epochs', default=3, type=int,
                            help='number of epochs used to collect source calibration samples')
    ours_group.add_argument('--lscs', action='store_true', default=False,
                            help='label-shift-aware corrective supervision (LSCS)')
    ours_group.add_argument('--lscs_delta', default=0.10, type=float,
                            help='risk budget for the certified corrective set')
    ours_group.add_argument('--lscs_class_cap', default=8, type=int,
                            help='maximum corrected samples per class per batch')
    ours_group.add_argument('--lscs_min_frac', default=0.02, type=float)
    ours_group.add_argument('--lscs_max_frac', default=0.60, type=float)
    ours_group.add_argument('--lscs_start_epoch', default=15, type=int)
    ours_group.add_argument('--lscs_fit_epochs', default=3, type=int)
    ours_group.add_argument('--m1_select', default='pats',
                            choices=['pats', 'all', 'random', 'excluded', 'rcs'],
                            help='RGS sample set for diagnosis/intervention: '
                                 'pats (default), all, random, excluded')
    ours_group.add_argument('--m1_dump', default='', type=str,
                            help='directory for per-sample RGS diagnostics (analysis only)')
    ours_group.add_argument('--m1_dump_epochs', default='', type=str,
                            help='comma separated epochs to dump; empty = final epoch')
    ours_group.add_argument('--reliab_ce_weight', default=0.0, type=float,
                            help='maximum weight of the PATS-gated pseudo-label CE (0 disables)')
    ours_group.add_argument('--reliab_ce_min_prob', default=0.95, type=float,
                            help='teacher confidence threshold applied on top of the PATS selection')
    ours_group.add_argument('--reliab_ce_agreement', action='store_true', default=False,
                            help='keep only samples where the two classifier heads agree')
    ours_group.add_argument('--reliab_ce_balance', default='none',
                            choices=['none', 'inverse', 'sqrt'],
                            help='class-frequency reweighting for the pseudo-label CE')
    ours_group.add_argument('--reliab_ce_start_epoch', default=10, type=int,
                            help='epoch to start the pseudo-label CE')
    ours_group.add_argument('--reliab_ce_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the pseudo-label CE weight')
    ours_group.add_argument('--proto_anchor_weight', default=0.0, type=float,
                            help='maximum weight of the source-prototype anchor loss (0 disables)')
    ours_group.add_argument('--proto_anchor_start_epoch', default=10, type=int,
                            help='epoch to start the prototype anchor loss')
    ours_group.add_argument('--proto_anchor_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the prototype anchor weight')
    ours_group.add_argument('--coral_weight', default=0.0, type=float,
                            help='maximum weight of the CORAL second-order alignment (0 disables)')
    ours_group.add_argument('--coral_start_epoch', default=10, type=int,
                            help='epoch to start the CORAL alignment')
    ours_group.add_argument('--coral_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the CORAL weight')
    ours_group.add_argument('--prat_prob', default=0.0, type=float,
                            help='probability of applying PRAT amplitude transfer to target batches')
    ours_group.add_argument('--prat_strength', default=0.5, type=float,
                            help='blend strength of source amplitude in PRAT augmentation')
    ours_group.add_argument('--prat_start_epoch', default=0, type=int,
                            help='epoch to start PRAT augmentation')
    ours_group.add_argument('--lmmd_weight', default=0.0, type=float,
                            help='maximum weight of the v3.0 LMMD class-conditional alignment (0 disables)')
    ours_group.add_argument('--lmmd_start_epoch', default=10, type=int,
                            help='epoch to start the LMMD alignment')
    ours_group.add_argument('--lmmd_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the LMMD weight')
    ours_group.add_argument('--lmmd_min_samples', default=4, type=int,
                            help='minimum samples per class required to compute LMMD')
    ours_group.add_argument('--cca_weight', default=0.0, type=float,
                            help='maximum weight of the v3.0 CCA class-conditional alignment (0 disables)')
    ours_group.add_argument('--cca_start_epoch', default=10, type=int,
                            help='epoch to start the CCA alignment')
    ours_group.add_argument('--cca_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the CCA weight')
    ours_group.add_argument('--cca_temperature', default=0.1, type=float,
                            help='soft-assignment temperature of the CCA alignment')
    ours_group.add_argument('--cca_margin', default=0.3, type=float,
                            help='margin of the CCA softplus objective')
    ours_group.add_argument('--prat_full_mode', default='off',
                            choices=['off', 'fixed', 'phase', 'phase_energy', 'full', 'replace'],
                            help='v3.0 full PRAT mode applied to source samples (off disables)')
    ours_group.add_argument('--prat_full_strength', default=0.3, type=float,
                            help='transfer strength of the full PRAT augmentation')
    ours_group.add_argument('--prat_full_start_epoch', default=1, type=int,
                            help='epoch to start the full PRAT augmentation')
    ours_group.add_argument('--prat_full_ema', default=0.96, type=float,
                            help='EMA momentum of the target amplitude reference in full PRAT')
    ours_group.add_argument('--memory_reliab', action='store_true', default=False,
                            help='replace the M1 sample gate with the v3.0 reliability memory')
    ours_group.add_argument('--memory_min_conf', default=0.95, type=float,
                            help='confidence threshold applied on top of the memory selection')
    ours_group.add_argument('--memory_reliab_threshold', default=0.22, type=float,
                            help='reliability threshold of the v3.0 memory')
    ours_group.add_argument('--memory_class_cap', default=18, type=int,
                            help='per-class cap of the v3.0 memory selection')
    ours_group.add_argument('--memory_margin', default=0.055, type=float,
                            help='margin gate of the v3.0 memory')
    ours_group.add_argument('--memory_prob_momentum', default=0.92, type=float,
                            help='probability memory momentum')
    ours_group.add_argument('--memory_feature_momentum', default=0.94, type=float,
                            help='feature memory momentum')
    ours_group.add_argument('--teacher_consistency_weight', default=0.0, type=float,
                            help='v3.0 weak-strong teacher consistency weight (0 disables)')
    ours_group.add_argument('--style_pair_weight', default=0.0, type=float,
                            help='v3.0 source-target style-pair loss weight (0 disables)')
    ours_group.add_argument('--source_metric_weight', default=0.0, type=float,
                            help='v3.0 source metric loss weight (0 disables)')
    ours_group.add_argument('--cnt_shift_weight', default=0.0, type=float,
                            help='v3.0 class-wise CNT mean-shift weight (0 disables)')
    ours_group.add_argument('--spec_view_js_weight', default=0.0, type=float,
                            help='v3.0 independent spectral view JS weight (0 disables)')
    ours_group.add_argument('--spec_view_strength', default=0.05, type=float,
                            help='noise strength of the independent spectral view')
    ours_group.add_argument('--reliab_lmmd_weight', default=0.0, type=float,
                            help='v3.0 reliability-weighted LMMD weight (0 disables)')
    ours_group.add_argument('--m1_maturity_gate', action='store_true', default=False,
                            help='scale the M1 CE weight by pseudo-label maturity (v3.0 maturity gate)')
    ours_group.add_argument('--m1_maturity_low', default=0.85, type=float,
                            help='mean confidence below which the maturity gate is 0')
    ours_group.add_argument('--m1_maturity_high', default=0.999, type=float,
                            help='mean confidence above which the maturity gate is 1')
    ours_group.add_argument('--m1_class_cap', default=0, type=int,
                            help='per-class cap of the M1 selected samples (0 disables, v3.0 SafetyFeedback)')
    ours_group.add_argument('--m1_soft_label', action='store_true', default=False,
                            help='use the full teacher distribution as a soft target (M2-S distillation)')
    ours_group.add_argument('--m1_trim_ratio', default=0.0, type=float,
                            help='drop the top fraction of the M1 per-sample losses (M2-T trimmed robust loss)')
    ours_group.add_argument('--m1_class_adaptive', default=0.0, type=float,
                            help='lower the M1 threshold for rare pseudo-label classes (M2-C class-adaptive gate)')
    ours_group.add_argument('--m1_class_quota', default=0, type=int,
                            help='per-class selection quota scaled by inverse target prior (M2-B, 0 disables)')
    ours_group.add_argument('--m1_teacher_temperature', default=0.0, type=float,
                            help='temperature used to recalibrate teacher confidence before gating (M3-T, 0 disables)')
    ours_group.add_argument('--m1_margin_gate', default=0.0, type=float,
                            help='minimum teacher top1-top2 margin for a sample to enter M1 (M3-M, 0 disables)')
    ours_group.add_argument('--m1_gate_mode', default='fixed', choices=['fixed', 'quantile'],
                            help='maturity gate: v1.0 fixed thresholds or data-adaptive quantile')
    ours_group.add_argument('--m1_gate_k', default=2.0, type=float,
                            help='standard-deviation multiplier of the adaptive maturity gate')
    ours_group.add_argument('--m1_topk_ratio', default=0.0, type=float,
                            help='coverage-driven selection: keep the top ratio of PATS samples (0 disables)')
    ours_group.add_argument('--m1_quota_mode', default='none', choices=['none', 'inverse_prior'],
                            help='class quota mode: fixed cap (legacy) or inverse-prior quota')
    ours_group.add_argument('--m1_quota_base', default=4, type=int,
                            help='base quota of the inverse-prior class quota')
    ours_group.add_argument('--m1_gate_epoch_start', default=0, type=int,
                            help='first epoch in which the maturity gate is active (0 = from the beginning)')
    ours_group.add_argument('--m1_gate_epoch_end', default=0, type=int,
                            help='last epoch in which the maturity gate is active (0 = until the end)')
    ours_group.add_argument('--m1_consensus', action='store_true', default=False,
                            help='use the F1/F2/teacher three-head consensus for M1 labels and confidence')
    ours_group.add_argument('--m1_prior_correct', action='store_true', default=False,
                            help='reweight teacher probabilities by the estimated target/source prior ratio (D2)')
    ours_group.add_argument('--m1_prior_power', default=0.5, type=float,
                            help='exponent applied to the prior ratio in D2')
    ours_group.add_argument('--m1_diag', action='store_true', default=False,
                            help='print unlabeled diagnostics used by the adaptive operator selection')
    ours_group.add_argument('--instability_adaptive', action='store_true', default=False,
                            help='adapt the RGS weight from an early training-instability score (innovation 4)')
    ours_group.add_argument('--instability_threshold', default=0.10, type=float,
                            help='instability score at which the adaptive weight reaches its maximum')
    ours_group.add_argument('--instability_base_weight', default=0.02, type=float,
                            help='RGS weight used for a stable run')
    ours_group.add_argument('--instability_max_weight', default=0.10, type=float,
                            help='RGS weight used for an unstable run')
    ours_group.add_argument('--ssti', action='store_true', default=False,
                            help='source-structured transductive inference at evaluation time')
    ours_group.add_argument('--ssti_alpha', default=0.3, type=float,
                            help='weight of the label-shift prior term in the ICM score')
    ours_group.add_argument('--ssti_beta', default=0.5, type=float,
                            help='weight of the spatial-transition term in the ICM score')
    ours_group.add_argument('--ssti_iters', default=3, type=int,
                            help='number of ICM iterations')
    ours_group.add_argument('--ssti_temperature', default=1.0, type=float,
                            help='temperature applied to the logits before the ICM')
    ours_group.add_argument('--ssti_conf_threshold', default=0.95, type=float,
                            help='only pixels below this confidence may be relabelled by the ICM')
    eval_group.add_argument('--eval_teacher', action='store_true', default=False,
                            help='include the EMA teacher head in the evaluation ensemble')
    eval_group.add_argument('--eval_teacher_weight', default=0.34, type=float,
                            help='weight of the EMA teacher logits in the evaluation ensemble')
    eval_group.add_argument('--eval_bn_adapt', action='store_true', default=False,
                            help='re-estimate batch-norm statistics on the target test data before evaluation')
    ours_group.add_argument('--ssti_src_epochs', default=20, type=int,
                            help='epochs of the source-only retraining used for the strict structure map')
    ours_group.add_argument('--ssti_src_lr', default=0.01, type=float,
                            help='learning rate of the source-only retraining')
    ours_group.add_argument('--ssti_pred_conf', default=0.9, type=float,
                            help='confidence threshold of the predicted source map used for the prior')
    ours_group.add_argument('--ssti_structure', default='potts', choices=['potts', 'gt', 'predicted'],
                            help="transition prior source: 'predicted' uses the source model (strict label budget), "
                                 "'gt' uses the full source ground truth (extra supervision)")
    return parser


def lmmd_loss(source_features, source_labels, target_features, target_labels,
              num_classes, min_samples=4):
    """v3.0 Local MMD: align class-conditional feature distributions."""
    source = source_features.float()
    target = target_features.float()
    total = source.new_tensor(0.0)
    used = 0
    for class_idx in range(num_classes):
        source_mask = source_labels.eq(class_idx)
        target_mask = target_labels.eq(class_idx)
        if int(source_mask.sum().item()) < min_samples or int(target_mask.sum().item()) < min_samples:
            continue
        source_mean = source[source_mask].mean(dim=0, keepdim=True)
        target_mean = target[target_mask].mean(dim=0, keepdim=True)
        total = total + (source_mean - target_mean).pow(2).sum()
        used += 1
    if used == 0:
        return source.new_tensor(0.0)
    return total / used


def cca_alignment_loss(target_features, source_prototypes, source_seen,
                        num_classes, temperature=0.1, margin=0.3):
    """v3.0 soft-assignment class-conditional alignment (CCA)."""
    valid = source_seen > 0
    if int(valid.sum().item()) < 2:
        return target_features.new_tensor(0.0)
    features = F.normalize(target_features.float(), dim=1)
    protos = F.normalize(source_prototypes.float(), dim=1)[valid]
    sim = features @ protos.t() / max(float(temperature), 1e-4)
    soft = F.softmax(sim, dim=1)
    target_protos = F.normalize(soft.t() @ features, dim=1)
    positive = (protos * target_protos).sum(dim=1)
    sim_mat = protos @ target_protos.t()
    neg_mask = torch.eye(protos.size(0), device=sim_mat.device, dtype=torch.bool)
    negative = sim_mat.masked_fill(neg_mask, -10.0).max(dim=1).values
    return F.softplus(negative - positive + float(margin)).mean()


def coral_loss(source_features, target_features):
    """Second-order (CORAL) alignment between source and target projections."""
    source = source_features.float()
    target = target_features.float()
    dim = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().matmul(source_centered) / max(1, source.size(0) - 1)
    target_cov = target_centered.t().matmul(target_centered) / max(1, target.size(0) - 1)
    return ((source_cov - target_cov) ** 2).sum() / (4.0 * dim * dim)


def prat_augment(source_images, target_images, strength=0.5, probability=0.5):
    """PRAT-style amplitude transfer (Fourier domain, spatial dims).

    Keeps the target phase and blends the target amplitude with the batch-mean
    source amplitude, transferring the source-domain spectral statistics.
    """
    if probability <= 0 or strength <= 0:
        return target_images
    if torch.rand(1).item() > probability:
        return target_images
    with torch.no_grad():
        source = source_images.float()
        target = target_images.float()
        source_fft = torch.fft.fftn(source, dim=(-2, -1))
        target_fft = torch.fft.fftn(target, dim=(-2, -1))
        source_amp = source_fft.abs().mean(dim=0, keepdim=True)
        target_amp = target_fft.abs()
        target_phase = target_fft / (target_amp + 1e-8)
        blended_amp = (1.0 - strength) * target_amp + strength * source_amp
        restored = torch.fft.ifftn(blended_amp * target_phase, dim=(-2, -1)).real
    return restored.type_as(target_images)


def parse_runtime_args():
    parsed_args = build_parser().parse_args()
    parsed_args.cuda = not parsed_args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(parsed_args.gpu) if torch.cuda.is_available() else 'cpu')
    return parsed_args, device


def setup_local_logging(args):
    output_dir = create_experiment_dir(args)
    log_file = open(os.path.join(output_dir, 'train.log'), 'w', encoding='utf-8')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_file)
    sys.stderr = TeeLogger(original_stderr, log_file)
    save_run_metadata(args, output_dir)
    print('Local output directory:', output_dir)
    return output_dir, log_file, original_stdout, original_stderr


def restore_logging(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def setup_wandb_logging(args):
    if not args.log_results:
        return None
    os.environ['WANDB-API_KEY'] = ''
    os.environ['WANDB_MODE'] = ''
    run = wandb.init(project='TAPDA', name=args.log_name)
    wandb.config.update(args)
    return run


def create_metric_buffers(num_datasets, num_classes):
    return {
        'acc': np.zeros([num_datasets, 1]),
        'acc_f1': np.zeros([num_datasets, 1]),
        'acc_f2': np.zeros([num_datasets, 1]),
        'acc_ensemble': np.zeros([num_datasets, 1]),
        'A': np.zeros([num_datasets, num_classes]),
        'k': np.zeros([num_datasets, 1]),
    }


def get_up_pc_paths():
    return (
        PAVIA_DATA_PATHS['source_data'],
        PAVIA_DATA_PATHS['source_label'],
        PAVIA_DATA_PATHS['target_data'],
        PAVIA_DATA_PATHS['target_label'],
    )


args, DEV = parse_runtime_args()
num_epoch = args.num_epoch
batch_size = args.batch_size
num_k = args.num_k


def load_data_and_preprocess(data_path_s, label_path_s, data_path_t, label_path_t, halfwidth, seed, fda=False, input_norm='none'):
    data_s, label_s = utils.load_data_pavia(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_pavia(data_path_t, label_path_t)
    if fda:
        data_s = fda_lowfreq_match(data_s, data_t)
        print('[fda] source cube low-frequency amplitude matched to target (phase preserved)')
    if input_norm == 'band':
        data_s = band_standardize(data_s)
        data_t = band_standardize(data_t)
        print('[norm] source/target cubes per-band standardized (label-free)')
    utils.seed_everything(seed, use_deterministic=True)

    train_x, train_y = utils.get_sample_data(data_s, label_s, halfwidth, 180)
    test_id, test_x, test_y, gr, rand_perm, row, column = utils.get_all_data(data_t, label_t, halfwidth)
    test_id_s, test_x_s, test_y_s, gr_s, rand_perm_s, row_s, column_s = utils.get_all_data(
        data_s, label_s, halfwidth
    )

    return {
        'trainX': train_x,
        'trainY': train_y,
        'testX': test_x,
        'testY': test_y,
        'testX_s': test_x_s,
        'testY_s': test_y_s,
        'Gr': gr,
        'RandPerm': rand_perm,
        'Row': row,
        'Column': column,
    }


class IndexedTensorDataset(torch.utils.data.Dataset):
    """Tensor dataset that also returns the sample index (needed by the memory)."""

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index], index


def create_data_loaders(train_x, train_y, test_x, test_y, test_x_s, test_y_s, batch_size):
    test_source_dataset = TensorDataset(torch.tensor(test_x_s), torch.tensor(test_y_s))
    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    indexed_test_dataset = IndexedTensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_tar_loader = DataLoader(indexed_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    source_test_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return {
        'train_loader': train_loader,
        'train_tar_loader': train_tar_loader,
        'test_loader': test_loader,
        'source_test_loader': source_test_loader,
        'len_src_loader': len(train_loader),
        'len_tar_train_loader': len(train_tar_loader),
        'len_src_dataset': len(train_loader.dataset),
        'len_tar_train_dataset': len(train_tar_loader.dataset),
        'len_tar_dataset': len(test_loader.dataset),
        'len_tar_loader': len(test_loader),
    }


def initialize_models_and_optimizers(n_bands, patch_size, num_classes, device, args):
    if getattr(args, 'backbone_impl', 'ssse') == 'reliability':
        G = ReliabilityEncoderContrastive(
            input_channels=n_bands,
            patch_size=patch_size,
            n_classes=num_classes,
        ).to(device)
    else:
        G = SSSE_Contrastive(
            input_channels=n_bands,
            patch_size=patch_size,
            n_classes=num_classes,
        ).to(device)
    if getattr(args, 'classifier_impl', 'res') == 'cosine':
        F1 = CosineClassifierHead(num_classes, feature_dim=152, training=True)
        F2 = CosineClassifierHead(num_classes, feature_dim=152, training=True)
    else:
        F1 = ResClassifier(num_classes, training=True)
        F2 = ResClassifier(num_classes, training=True)
        F1.apply(weights_init)
        F2.apply(weights_init)

    F1_teacher = EMATeacher(G, F1, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)
    contrastive_loss_func = SupConLoss(
        temperature=args.supcon_temperature,
        contrast_mode=args.supcon_contrast_mode,
    ).to(device)

    if args.cuda:
        G.cuda()
        F1.cuda()
        F2.cuda()

    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_f = optim.SGD(
            list(F1.parameters()) + list(F2.parameters()),
            momentum=0.9,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer_g = optim.AdamW(list(G.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_f = optim.AdamW(
            list(F1.parameters()) + list(F2.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_g = optim.Adadelta(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_f = optim.Adadelta(
            list(F1.parameters()) + list(F2.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    return {
        'G': G,
        'F1': F1,
        'F2': F2,
        'F1_teacher': F1_teacher,
        'contrastive_loss_func': contrastive_loss_func,
        'optimizer_g': optimizer_g,
        'optimizer_f': optimizer_f,
    }


def train_epoch(ep, train_loader, train_tar_loader, G, F1, F2, F1_teacher,
                contrastive_loss_func, optimizer_g, optimizer_f,
                batch_size, len_src_loader, num_k, args, seed, num_classes, target_quota_state,
                source_prototype_state, prat_full=None, memory=None):
    iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(DEV)
    G.train()
    F1.train()
    F2.train()
    num_iter = len_src_loader * max(1, int(getattr(args, 'target_repeats', 1)))
    _diag_reset()
    target_diagnostic_state = (
        create_target_diagnostic_state(num_classes)
        if args.log_target_diagnostic else None
    )

    for batch_idx in range(1, num_iter):
        if batch_idx % len(train_tar_loader) == 0:
            iter_target = iter(train_tar_loader)
        if batch_idx % len_src_loader == 0:
            iter_source = iter(train_loader)
        data_source, label_source = next(iter_source)
        data_target, label_target, target_ids = next(iter_target)
        if args.cuda:
            data1, target1 = data_source.cuda(), label_source.cuda()
            data2 = data_target.cuda()
            label_target_diag = label_target.cuda()
        else:
            data1, target1 = data_source, label_source
            data2 = data_target
            label_target_diag = label_target

        if args.prat_prob > 0 and ep >= args.prat_start_epoch:
            data2 = prat_augment(data1, data2, args.prat_strength, args.prat_prob)

        if prat_full is not None and ep >= args.prat_full_start_epoch:
            data1, _ = prat_full(data1, data2, args.prat_full_strength)

        eta = 1.0
        src_batch_size = data1.size(0)
        data = Variable(torch.cat((data1, data2), 0))
        target1 = Variable(target1)
        F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
        F1_pseudo_label_t, F1_pseudo_prob_t, teacher_feat_t, teacher_proj_t = F1_teacher(data2)

        _, _, teacher_feat_s, teacher_proj_s = F1_teacher(data1)
        _, student_proj_s = G(data1)
        contrastive_loss_s = args.source_contrastive_weight * compute_projection_contrastive_loss(
            teacher_proj_s,
            student_proj_s,
            target1,
            contrastive_loss_func,
        )
        update_source_prototype_state(source_prototype_state, teacher_proj_s, target1, num_classes, args)
        student_feat_t, student_proj_t = G(data2)
        memory_batch = None
        if memory is not None:
            with torch.no_grad():
                teacher_logits_mem, _ = F1_teacher.f(teacher_feat_t)
                teacher_probs_mem = torch.softmax(teacher_logits_mem, dim=1)
                student_logits_mem, _ = F1(student_feat_t.detach())
                student_probs_mem = torch.softmax(student_logits_mem, dim=1)
                memory.update_source_prototypes(teacher_feat_s, target1)
            memory_batch = memory.update(
                target_ids.to(DEV), teacher_probs_mem, student_probs_mem,
                teacher_feat_t.detach(),
            )
        dynamic_class_topk = None
        quota_stats = {
            'quota_mode': 'none',
            'quota_active_classes': 0,
            'quota_avg': float(args.target_contrast_topk_per_class),
            'quota_max': int(args.target_contrast_topk_per_class),
            'quota_difficulty_mean': 0.0,
            'quota_scarcity_mean': 0.0,
        }
        if use_target_class_quota(args):
            with torch.no_grad():
                target_logits1, _ = F1(student_feat_t.detach())
                target_logits2, _ = F2(student_feat_t.detach())
                dynamic_class_topk, quota_stats = compute_confusion_aware_class_quotas(
                    F1_pseudo_label_t,
                    F1_pseudo_prob_t,
                    target_logits1,
                    target_logits2,
                    target_quota_state,
                    ep,
                    args,
                )
        contrastive_loss_t, target_contrastive_stats = compute_target_contrastive_loss(
            teacher_proj_t,
            student_proj_t,
            F1_pseudo_label_t,
            F1_pseudo_prob_t,
            ep,
            num_classes,
            args,
            contrastive_loss_func,
            per_class_topk=dynamic_class_topk,
            prototype_filter_projections=teacher_proj_t,
            source_prototype_state=source_prototype_state,
            return_indices=True,
        )
        target_contrastive_stats.update(quota_stats)
        contrastive_loss = contrastive_loss_s + contrastive_loss_t

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output, _ = G(data)
        output1 = F1(output)
        output2 = F2(output)

        output_s1 = output1[0][:src_batch_size, :]
        output_s2 = output2[0][:src_batch_size, :]
        output_t1 = output1[0][src_batch_size:, :]
        output_t2 = output2[0][src_batch_size:, :]

        loss1 = criterion(output_s1, target1.long())
        loss2 = criterion(output_s2, target1.long())

        bnm_loss_a, output_t1_prob, output_t2_prob = compute_bnm_loss(
            output_t1, output_t2, get_bnm_weight(ep, 'A', args)
        )
        update_target_diagnostic_state(
            target_diagnostic_state,
            F1_pseudo_label_t,
            F1_pseudo_prob_t,
            target_contrastive_stats,
            logits_t1=output_t1,
            logits_t2=output_t2,
            target_labels=label_target_diag,
        )
        all_loss = loss1 + loss2 + bnm_loss_a + contrastive_loss

        # ---- Ours: PATS-gated pseudo-label CE (M1) + source-prototype anchor (M3) ----
        ramp_fraction = get_rampup_weight(
            ep, args.reliab_ce_start_epoch, args.reliab_ce_ramp_epochs, 1.0
        )
        reliab_ce_weight = ramp_fraction * args.reliab_ce_weight
        if args.instability_adaptive:
            with torch.no_grad():
                probability_1 = torch.softmax(output_t1.detach().float(), dim=1)
                probability_2 = torch.softmax(output_t2.detach().float(), dim=1)
                agreement = float(
                    probability_1.argmax(dim=1).eq(probability_2.argmax(dim=1)).float().mean()
                )
                confidence_mean = float(F1_pseudo_prob_t.detach().float().mean())
                instability = (1.0 - agreement) + max(0.0, 0.995 - confidence_mean) * 2.0
            previous = _INSTABILITY_EMA['value']
            smoothed = instability if previous is None else 0.9 * previous + 0.1 * instability
            _INSTABILITY_EMA['value'] = smoothed
            level = min(max(smoothed / max(args.instability_threshold, 1e-6), 0.0), 1.0)
            adaptive_weight = args.instability_base_weight + (
                args.instability_max_weight - args.instability_base_weight
            ) * level
            reliab_ce_weight = ramp_fraction * adaptive_weight
            if batch_idx % 50 == 0:
                print(
                    f"[instability] ep={ep} agree={agreement:.3f} conf={confidence_mean:.4f} "
                    f"score={smoothed:.4f} weight={reliab_ce_weight:.4f}"
                )
        proto_anchor_weight = get_rampup_weight(
            ep, args.proto_anchor_start_epoch, args.proto_anchor_ramp_epochs,
            args.proto_anchor_weight
        )
        pats_indices = target_contrastive_stats.get('selected_indices')
        if memory_batch is not None:
            pats_indices = memory_batch.stable_mask.nonzero(as_tuple=True)[0]
        # ---- RGS diagnosis / intervention: override the supervised set ----
        _m1_sel = ('lscs' if getattr(args, 'lscs', False)
                   else 'rcs' if getattr(args, 'rcs', False)
                   else getattr(args, 'm1_select', 'pats'))
        if _m1_sel != 'pats':
            _total = F1_pseudo_label_t.numel()
            _all_idx = torch.arange(_total, device=F1_pseudo_label_t.device)
            _cur = pats_indices if pats_indices is not None else _all_idx[:0]
            if _m1_sel == 'all':
                pats_indices = _all_idx
            elif _m1_sel == 'excluded':
                _keep = torch.ones(_total, dtype=torch.bool, device=_all_idx.device)
                if _cur.numel() > 0:
                    _keep[_cur] = False
                pats_indices = _all_idx[_keep]
            elif _m1_sel == 'random':
                _k = int(_cur.numel()) if _cur.numel() > 0 else 32
                pats_indices = torch.randperm(_total, device=_all_idx.device)[:_k]
            elif _m1_sel == 'rcs':
                with torch.no_grad():
                    teacher_logits_s, _ = F1_teacher.f(teacher_feat_s)
                    teacher_prob_s = torch.softmax(teacher_logits_s.float(), dim=1)
                    src_feat = rcs_signals(teacher_prob_s, output_s1, output_s2)
                    src_pred = teacher_prob_s.argmax(dim=1).cpu().numpy()
                    src_lab = target1.detach().cpu().numpy()
                    rcs_observe(src_feat, (src_pred == src_lab).astype('int64'))
                    if ep >= int(args.rcs_start_epoch) + int(args.rcs_fit_epochs):
                        rcs_fit()
                    teacher_logits_t, _ = F1_teacher.f(teacher_feat_t)
                    teacher_prob_t = torch.softmax(teacher_logits_t.detach().float(), dim=1)
                    tgt_feat = rcs_signals(teacher_prob_t, output_t1, output_t2)
                    q = rcs_predict(tgt_feat)
                    if q is not None:
                        keep_idx = rcs_risk_select(
                            q, args.rcs_delta, args.rcs_min_frac, args.rcs_max_frac)
                        pats_indices = torch.from_numpy(keep_idx).long().to(
                            F1_pseudo_label_t.device)
            elif _m1_sel == 'lscs':
                with torch.no_grad():
                    teacher_logits_s, _ = F1_teacher.f(teacher_feat_s)
                    teacher_prob_s = torch.softmax(teacher_logits_s.float(), dim=1)
                    lscs_observe_source(teacher_prob_s, output_s1, output_s2, target1)
                    teacher_logits_t, _ = F1_teacher.f(teacher_feat_t)
                    teacher_prob_t = torch.softmax(teacher_logits_t.float(), dim=1)
                    lscs_update_target_hist(teacher_prob_t.argmax(dim=1))
                    if ep >= int(args.lscs_start_epoch) + int(args.lscs_fit_epochs):
                        lscs_fit()
                    keep_idx = lscs_select(
                        teacher_prob_t, output_t1, output_t2,
                        delta=args.lscs_delta,
                        class_cap=args.lscs_class_cap,
                        min_frac=args.lscs_min_frac,
                        max_frac=args.lscs_max_frac)
                    pats_indices = torch.from_numpy(
                        np.asarray(keep_idx, dtype=np.int64)).long().to(
                        F1_pseudo_label_t.device)

        _pats_mask = None
        if getattr(args, 'm1_dump', ''):
            _pats_mask = torch.zeros(F1_pseudo_label_t.numel(), dtype=torch.bool,
                                     device=F1_pseudo_label_t.device)
            if pats_indices is not None and pats_indices.numel() > 0:
                _pats_mask[pats_indices] = True

        if ((reliab_ce_weight > 0 or proto_anchor_weight > 0)
                and pats_indices is not None and pats_indices.numel() > 0):
            with torch.no_grad():
                if args.m1_teacher_temperature > 0:
                    teacher_logits_m3, _ = F1_teacher.f(teacher_feat_t)
                    teacher_prob_m3 = torch.softmax(
                        teacher_logits_m3.detach().float() / args.m1_teacher_temperature, dim=1
                    )[pats_indices]
                    pats_probs, pats_labels = teacher_prob_m3.max(dim=1)
                else:
                    pats_labels = F1_pseudo_label_t.detach()[pats_indices].long()
                    pats_probs = F1_pseudo_prob_t.detach()[pats_indices].float()
                if args.m1_prior_correct:
                    with torch.no_grad():
                        source_counts = torch.bincount(
                            target1.detach().long(), minlength=num_classes
                        ).float()
                        source_prior = (source_counts / source_counts.sum().clamp_min(1.0)).clamp_min(1e-3)
                        target_counts = torch.bincount(
                            F1_pseudo_label_t.detach().long(), minlength=num_classes
                        ).float()
                        target_prior = (target_counts / target_counts.sum().clamp_min(1.0)).clamp_min(1e-3)
                        prior_ratio = (target_prior / source_prior).pow(args.m1_prior_power)
                        teacher_logits_pc, _ = F1_teacher.f(teacher_feat_t)
                        prob_pc = torch.softmax(teacher_logits_pc.detach().float(), dim=1)
                        prob_pc = prob_pc * prior_ratio.unsqueeze(0)
                        prob_pc = prob_pc / prob_pc.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        prob_pc = prob_pc[pats_indices]
                        pats_probs, pats_labels = prob_pc.max(dim=1)
                if args.m1_consensus:
                    with torch.no_grad():
                        teacher_logits_cs, _ = F1_teacher.f(teacher_feat_t)
                        prob_cons = (
                            torch.softmax(teacher_logits_cs.detach().float(), dim=1)
                            + torch.softmax(output_t1.detach().float(), dim=1)
                            + torch.softmax(output_t2.detach().float(), dim=1)
                        ) / 3.0
                        prob_cons = prob_cons[pats_indices]
                        pats_probs, pats_labels = prob_cons.max(dim=1)
                if args.m1_class_adaptive > 0 and pats_labels.numel() > 0:
                    prior_counts = torch.bincount(
                        F1_pseudo_label_t.detach().long(), minlength=num_classes
                    ).float()
                    priors = prior_counts / prior_counts.sum().clamp_min(1.0)
                    max_prior = priors.max().clamp_min(1e-6)
                    adaptive_thr = args.reliab_ce_min_prob - args.m1_class_adaptive * (
                        1.0 - priors[pats_labels] / max_prior
                    )
                    confident = pats_probs >= adaptive_thr
                else:
                    confident = pats_probs >= args.reliab_ce_min_prob
                if args.m1_margin_gate > 0:
                    with torch.no_grad():
                        teacher_logits_m3m, _ = F1_teacher.f(teacher_feat_t)
                        teacher_prob_m3m = torch.softmax(
                            teacher_logits_m3m.detach().float(), dim=1
                        )[pats_indices]
                        top2 = teacher_prob_m3m.topk(2, dim=1).values
                        confident = confident & ((top2[:, 0] - top2[:, 1]) >= args.m1_margin_gate)
                if args.m1_topk_ratio > 0 and pats_probs.numel() > 0:
                    topk = max(1, int(round(args.m1_topk_ratio * pats_probs.numel())))
                    confident = pats_probs >= pats_probs.topk(topk).values.min()
                if args.reliab_ce_agreement:
                    agreement = (output_t1[pats_indices].argmax(dim=1)
                                 == output_t2[pats_indices].argmax(dim=1))
                    confident = confident & agreement
            if reliab_ce_weight > 0 and ep % 10 == 0:
                print(f"[ours] ep={ep} pats_selected={int(pats_indices.numel())} "
                      f"confident={int(confident.sum())} mean_prob={float(pats_probs.mean()):.3f}")
                if args.m1_diag and ep % 5 == 0:
                    with torch.no_grad():
                        sel_n = int(sel_idx.numel()) if 'sel_idx' in dir() else int(confident.sum())
                        sel_mean = float(sel_probs.mean()) if 'sel_probs' in dir() else float(pats_probs.mean())
                        sel_std = float(sel_probs.std(unbiased=False)) if 'sel_probs' in dir() else float(pats_probs.std(unbiased=False))
                        batch_n = max(1, int(pats_indices.numel()))
                        all_labels = F1_pseudo_label_t.detach().long()
                        counts = torch.bincount(all_labels, minlength=num_classes).float()
                        probs = counts / counts.sum().clamp_min(1.0)
                        gini_all = float(1.0 - (probs ** 2).sum())
                        sel_counts = torch.bincount(pats_labels.long(), minlength=num_classes).float()
                        sel_probs_dist = sel_counts / sel_counts.sum().clamp_min(1.0)
                        gini_sel = float(1.0 - (sel_probs_dist ** 2).sum())
                        print(f"[diag] ep={ep} sel_all={batch_n} confident={sel_n} "
                              f"conf={sel_mean:.4f}+-{sel_std:.4f} "
                              f"gini_all={gini_all:.4f} gini_sel={gini_sel:.4f}")
            if confident.any():
                sel_idx = pats_indices[confident]
                if getattr(args, 'm1_dump', '') and _pats_mask is not None:
                    _diag_add(target_ids, F1_pseudo_label_t, F1_pseudo_prob_t,
                              output_t1.detach().argmax(dim=1), _pats_mask, sel_idx)

                sel_labels = pats_labels[confident]
                sel_probs = pats_probs[confident]
                sel_weight = sel_probs
                if args.m1_class_cap > 0 and sel_idx.numel() > 0:
                    keep = torch.zeros_like(sel_labels, dtype=torch.bool)
                    for class_id in sel_labels.unique().tolist():
                        class_mask = sel_labels.eq(int(class_id)).nonzero(as_tuple=True)[0]
                        if class_mask.numel() > args.m1_class_cap:
                            order = sel_probs[class_mask].argsort(descending=True)
                            class_mask = class_mask[order[:args.m1_class_cap]]
                        keep[class_mask] = True
                    sel_idx, sel_labels, sel_weight = sel_idx[keep], sel_labels[keep], sel_weight[keep]
                if (args.m1_quota_mode == 'inverse_prior' or args.m1_class_quota > 0) and sel_labels.numel() > 0:
                    quota_base = args.m1_quota_base if args.m1_quota_mode == 'inverse_prior' else args.m1_class_quota
                    target_counts = torch.bincount(
                        F1_pseudo_label_t.detach().long(), minlength=num_classes
                    ).float().clamp_min(1.0)
                    inverse_prior = target_counts.mean() / target_counts
                    keep = torch.zeros_like(sel_labels, dtype=torch.bool)
                    for class_id in sel_labels.unique().tolist():
                        class_mask = sel_labels.eq(int(class_id)).nonzero(as_tuple=True)[0]
                        quota = int(max(1, round(quota_base * float(inverse_prior[int(class_id)]))))
                        order = sel_probs[class_mask].argsort(descending=True)
                        keep[class_mask[order[:quota]]] = True
                    sel_idx, sel_labels, sel_weight = sel_idx[keep], sel_labels[keep], sel_weight[keep]
                gate_active = True
                if args.m1_gate_epoch_start > 0 or args.m1_gate_epoch_end > 0:
                    gate_end = args.m1_gate_epoch_end if args.m1_gate_epoch_end > 0 else 10 ** 9
                    gate_start = args.m1_gate_epoch_start if args.m1_gate_epoch_start > 0 else 1
                    gate_active = gate_start <= ep <= gate_end
                if args.m1_maturity_gate and gate_active and sel_weight.numel() > 0:
                    maturity = float(sel_probs.mean().item())
                    if args.m1_gate_mode == 'fixed':
                        span = max(args.m1_maturity_high - args.m1_maturity_low, 1e-6)
                        gate = min(max((maturity - args.m1_maturity_low) / span, 0.0), 1.0)
                    else:
                        ref_mean = float(sel_probs.mean().item())
                        ref_std = float(sel_probs.std(unbiased=False).clamp_min(1e-6).item())
                        width = args.m1_gate_k * ref_std
                        gate = min(max((maturity - (ref_mean - width)) / max(width, 1e-6), 0.0), 1.0)
                    reliab_ce_weight = reliab_ce_weight * gate
                if args.reliab_ce_balance != 'none':
                    class_counts = torch.bincount(sel_labels, minlength=num_classes).float()
                    class_freq = class_counts[sel_labels].clamp_min(1.0)
                    if args.reliab_ce_balance == 'inverse':
                        class_weight = 1.0 / class_freq
                    else:
                        class_weight = 1.0 / class_freq.sqrt()
                    class_weight = class_weight / class_weight.mean().clamp_min(1e-6)
                    sel_weight = sel_probs * class_weight
                if reliab_ce_weight > 0:
                    if args.m1_soft_label:
                        with torch.no_grad():
                            teacher_logits_m1, _ = F1_teacher.f(teacher_feat_t)
                            teacher_prob_m1 = torch.softmax(
                                teacher_logits_m1.detach().float(), dim=1
                            )
                        log_student = F.log_softmax(output_t1[sel_idx].float(), dim=1)
                        ce_all = -(teacher_prob_m1[sel_idx] * log_student).sum(dim=1)
                    else:
                        ce_all = F.cross_entropy(output_t1[sel_idx], sel_labels, reduction='none')
                    if args.m1_trim_ratio > 0 and ce_all.numel() > 2:
                        drop = max(1, int(ce_all.numel() * args.m1_trim_ratio))
                        if drop < ce_all.numel():
                            _, drop_idx = torch.topk(ce_all.detach(), drop)
                            keep = torch.ones_like(ce_all, dtype=torch.bool)
                            keep[drop_idx] = False
                            ce_all, sel_weight = ce_all[keep], sel_weight[keep]
                    rel_ce = (ce_all * sel_weight).sum() / sel_weight.sum().clamp_min(1e-6)
                    all_loss = all_loss + reliab_ce_weight * rel_ce
                if proto_anchor_weight > 0:
                    proto_bank = source_prototype_state.get('prototypes')
                    proto_seen = source_prototype_state.get('seen')
                    if proto_bank is not None and proto_seen is not None:
                        valid_class = proto_seen[sel_labels] > 0
                        if valid_class.any():
                            proj = F.normalize(
                                student_proj_t[sel_idx][valid_class].float(), dim=1
                            )
                            anchor = 1.0 - F.cosine_similarity(
                                proj, proto_bank[sel_labels[valid_class]], dim=1
                            )
                            all_loss = all_loss + proto_anchor_weight * anchor.mean()

        coral_weight = get_rampup_weight(
            ep, args.coral_start_epoch, args.coral_ramp_epochs, args.coral_weight
        )
        if coral_weight > 0:
            all_loss = all_loss + coral_weight * coral_loss(student_proj_s, student_proj_t)

        lmmd_weight = get_rampup_weight(
            ep, args.lmmd_start_epoch, args.lmmd_ramp_epochs, args.lmmd_weight
        )
        if lmmd_weight > 0:
            lmmd_value = lmmd_loss(
                output[:src_batch_size], target1.long(),
                output[src_batch_size:], F1_pseudo_label_t.long(),
                num_classes, min_samples=args.lmmd_min_samples,
            )
            all_loss = all_loss + lmmd_weight * lmmd_value

        cca_weight = get_rampup_weight(
            ep, args.cca_start_epoch, args.cca_ramp_epochs, args.cca_weight
        )
        if cca_weight > 0:
            proto_bank = source_prototype_state.get('prototypes')
            proto_seen = source_prototype_state.get('seen')
            if proto_bank is not None and proto_seen is not None:
                cca_value = cca_alignment_loss(
                    student_proj_t, proto_bank, proto_seen, num_classes,
                    temperature=args.cca_temperature, margin=args.cca_margin,
                )
                all_loss = all_loss + cca_weight * cca_value

        # ---- v3.0 component screening (all additive) ----
        screen_values = {}
        if (args.teacher_consistency_weight > 0 or args.style_pair_weight > 0
                or args.source_metric_weight > 0 or args.cnt_shift_weight > 0
                or args.spec_view_js_weight > 0 or args.reliab_lmmd_weight > 0):
            teacher_logits_aux, _ = F1_teacher.f(teacher_feat_t)
            teacher_prob_aux = torch.softmax(teacher_logits_aux.detach().float(), dim=1)
            if args.teacher_consistency_weight > 0:
                value = confidence_consistency_loss(
                    teacher_prob_aux, output_t1, teacher_proj_t, student_proj_t)
                screen_values['S1_teacher_consistency'] = value
                all_loss = all_loss + args.teacher_consistency_weight * value
            if args.style_pair_weight > 0:
                value = style_pair_loss(student_proj_s, student_proj_t)
                screen_values['S2_style_pair'] = value
                all_loss = all_loss + args.style_pair_weight * value
            if args.source_metric_weight > 0:
                value = source_metric_loss(student_proj_s, target1, num_classes)
                screen_values['S3_source_metric'] = value
                all_loss = all_loss + args.source_metric_weight * value
            if args.cnt_shift_weight > 0:
                value = classwise_cnt_shift_loss(
                    source_prototype_state['prototypes'], source_prototype_state['seen'],
                    student_feat_t, F1_pseudo_label_t.detach(), F1_pseudo_prob_t.detach(),
                    num_classes)
                screen_values['S4_cnt_shift'] = value
                all_loss = all_loss + args.cnt_shift_weight * value
            if args.reliab_lmmd_weight > 0:
                value = weighted_lmmd_loss(
                    student_proj_s, target1, student_proj_t,
                    F1_pseudo_label_t.detach(), None, num_classes)
                screen_values['S6_reliab_lmmd'] = value
                all_loss = all_loss + args.reliab_lmmd_weight * value
            if args.spec_view_js_weight > 0:
                view = data2 + args.spec_view_strength * torch.randn_like(data2) * data2.std()
                view_feat, _ = G(view)
                view_logits = F1(view_feat)[0]
                base_prob = torch.softmax(output_t1.detach().float(), dim=1)
                view_prob = torch.softmax(view_logits.float(), dim=1)
                value = spectral_view_js_loss(base_prob, view_prob)
                screen_values['S5_spec_view_js'] = value
                all_loss = all_loss + args.spec_view_js_weight * value
            if screen_values and (ep <= 1 and batch_idx <= 3):
                report = ' '.join(
                    f'{key}={float(val.detach()):.4f}(grad={int(val.requires_grad)})'
                    for key, val in screen_values.items()
                )
                print(f'[screen] ep={ep} iter={batch_idx} {report}')

        if args.log_results:
            wandb.log({f'{seed}_A_loss': all_loss})
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output, _ = G(data)
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[0][:src_batch_size, :]
        output_s2 = output2[0][:src_batch_size, :]
        output_t1 = output1[0][src_batch_size:, :]
        output_t2 = output2[0][src_batch_size:, :]
        bnm_loss_b, output_t1_prob, output_t2_prob = compute_bnm_loss(
            output_t1, output_t2, get_bnm_weight(ep, 'B', args)
        )
        loss1 = criterion(output_s1, target1.long())
        loss2 = criterion(output_s2, target1.long())

        loss_dis = torch.mean(torch.abs(output_t1_prob - output_t2_prob))
        F_loss = loss1 + loss2 - eta * loss_dis + bnm_loss_b
        if args.log_results:
            wandb.log({f'{seed}_B_loss': F_loss})
        F_loss.backward()
        optimizer_f.step()

        for i in range(num_k):
            optimizer_g.zero_grad()
            output, _ = G(data)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[0][:src_batch_size, :]
            output_s2 = output2[0][:src_batch_size, :]
            output_t1 = output1[0][src_batch_size:, :]
            output_t2 = output2[0][src_batch_size:, :]

            bnm_loss_c, output_t1_prob, output_t2_prob = compute_bnm_loss(
                output_t1, output_t2, get_bnm_weight(ep, 'C', args)
            )
            loss_dis = torch.mean(torch.abs(output_t1_prob - output_t2_prob))
            loss_g = loss_dis + bnm_loss_c

            if args.log_results:
                wandb.log({f'{seed}_C_loss': loss_g})
            loss_g.backward()
            optimizer_g.step()

        if batch_idx % args.log_interval == 0:
            print(
                format_train_status(
                    ep, batch_idx, data1, train_loader, loss1, loss2, loss_dis,
                    bnm_loss_a, contrastive_loss_s, contrastive_loss_t,
                    target_contrastive_stats, args,
                )
            )

        if batch_idx == 1 and ep > 1:
            G.train()
            F1.train()
            F2.train()

    if args.log_target_diagnostic:
        print(format_target_diagnostic_summary(ep, target_diagnostic_state))


from modules.ssti import refine_labels
_FDA_STATS = {'mu_s': None, 'sd_s': None, 'mu_t': None, 'sd_t': None}
def fda_lowfreq_match(src, tar, radius=0.1):
    """Source phase + target low-frequency amplitude (per band), cube level.

    Keeps the source spatial/spectral structure (phase) and replaces only the
    low-frequency magnitude statistics with the target ones.  This is the
    standard Fourier domain adaptation (FDA) operation adapted to HSI cubes.
    """
    import numpy as _np
    src = _np.asarray(src, dtype=_np.float64)
    tar = _np.asarray(tar, dtype=_np.float64)
    h, w, b = src.shape
    fy = _np.fft.fftfreq(h)[:, None]
    fx = _np.fft.fftfreq(w)[None, :]
    r = _np.sqrt(fy ** 2 + fx ** 2)
    low = r <= float(radius)
    out = _np.empty_like(src)
    for i in range(b):
        fs = _np.fft.fft2(src[..., i])
        ft = _np.fft.fft2(tar[..., i])
        scale = _np.abs(ft)[low].mean() / max(1e-8, _np.abs(fs)[low].mean())
        fs = _np.where(low, fs * scale, fs)
        out[..., i] = _np.real(_np.fft.ifft2(fs))
    return out.astype(_np.float32)

def band_standardize(cube):
    """Per-band z-score over the valid (non-zero) pixels of one scene.

    Label-free input recalibration: removes the scene-level brightness/scale
    mismatch (source vs target mean spectra differ by ~92% in Houston).
    """
    import numpy as _np
    cube = _np.asarray(cube, dtype=_np.float32)
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    valid = _np.abs(flat).sum(axis=1) > 0
    if valid.sum() == 0:
        return cube
    mu = flat[valid].mean(axis=0, keepdims=True)
    sd = flat[valid].std(axis=0, keepdims=True) + 1e-6
    flat = flat.copy()
    flat[valid] = (flat[valid] - mu) / sd
    return flat.reshape(h, w, b).astype(_np.float32)

from modules.lscs import (lscs_reset, lscs_observe_source, lscs_update_target_hist,
                         lscs_fit, lscs_select)
from modules.rcs import (rcs_reset, rcs_signals, rcs_observe,
                         rcs_fit, rcs_predict, rcs_risk_select)

def test_model(test_loader, G, F1, F2, device, len_tar_dataset, args, seed, ssti_context=None, teacher=None):
    if getattr(args, 'eval_bn_adapt', False) and teacher is None:
        G.train()
        F1.train()
        with torch.no_grad():
            for data, _ in test_loader:
                G(data.to(device))
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct_ensemble = 0
    labels = np.array([], dtype=np.int64)
    predict_f1 = np.array([], dtype=np.int64)
    predict_f2 = np.array([], dtype=np.int64)
    predict_ensemble = np.array([], dtype=np.int64)
    outputdata = []
    eval_logits_batches = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data1, target1 = Variable(data), Variable(target)
            output, _ = G(data1)
            outputdata.append(output.cpu().numpy())
            logits1 = F1(output)
            logits2 = F2(output)
            ensemble_logits = 0.5 * (logits1 + logits2)
            if teacher is not None and args.eval_teacher:
                teacher_output = teacher(data1)
                teacher_features = teacher_output[2] if isinstance(teacher_output, (tuple, list)) else teacher_output
                teacher_logits = teacher.f(teacher_features)
                if isinstance(teacher_logits, (tuple, list)):
                    teacher_logits = teacher_logits[0]
                weight = float(args.eval_teacher_weight)
                ensemble_logits = (1.0 - weight) * ensemble_logits + weight * teacher_logits
            eval_logits = select_eval_logits(logits1, logits2, args.eval_mode)
            if teacher is not None and args.eval_teacher and args.eval_mode == 'ensemble':
                eval_logits = ensemble_logits
            eval_logits_batches.append(eval_logits.detach().cpu())
            test_loss += F.cross_entropy(eval_logits, target1.long(), reduction='sum').item()

            pred1 = logits1.data.max(1)[1]
            pred2 = logits2.data.max(1)[1]
            pred_ensemble = ensemble_logits.data.max(1)[1]
            correct1 += pred1.eq(target1.data).cpu().sum()
            correct2 += pred2.eq(target1.data).cpu().sum()
            correct_ensemble += pred_ensemble.eq(target1.data).cpu().sum()
            predict_f1 = np.append(predict_f1, pred1.cpu().numpy())
            predict_f2 = np.append(predict_f2, pred2.cpu().numpy())
            predict_ensemble = np.append(predict_ensemble, pred_ensemble.cpu().numpy())
            labels = np.append(labels, target.cpu().numpy())

        test_loss /= len_tar_dataset
        acc1 = 100. * float(correct1) / float(len_tar_dataset)
        acc2 = 100. * float(correct2) / float(len_tar_dataset)
        acc_ensemble = 100. * float(correct_ensemble) / float(len_tar_dataset)
        if ssti_context is not None and getattr(args, 'ssti', False):
            logits_all = torch.cat(eval_logits_batches, 0)
            refined = refine_labels(
                logits_all, ssti_context, labels,
                alpha=args.ssti_alpha, beta=args.ssti_beta,
                iterations=args.ssti_iters, temperature=args.ssti_temperature,
                confidence_threshold=args.ssti_conf_threshold,
            )
            predict_ensemble = refined
            acc_ensemble = 100. * float((refined == labels).sum()) / float(len_tar_dataset)
        if args.eval_mode == 'f1':
            predict_eval = predict_f1
            acc_eval = acc1
        elif args.eval_mode == 'f2':
            predict_eval = predict_f2
            acc_eval = acc2
        else:
            predict_eval = predict_ensemble
            acc_eval = acc_ensemble

        print(
            '\nTest set: Average loss: {:.4f}, F1: {:.2f}% F2: {:.2f}% Ens: {:.2f}% Eval({}): {:.2f}%\n'.format(
                test_loss, acc1, acc2, acc_ensemble, args.eval_mode, acc_eval
            )
        )
        if args.log_results:
            wandb.log({
                f'{seed}_result_f1': acc1,
                f'{seed}_result_f2': acc2,
                f'{seed}_result_ensemble': acc_ensemble,
                f'{seed}_result_eval': acc_eval,
            })

    return {
        'acc1': acc1,
        'acc2': acc2,
        'acc_ensemble': acc_ensemble,
        'acc_eval': acc_eval,
        'predict_f1': predict_f1,
        'predict_f2': predict_f2,
        'predict_ensemble': predict_ensemble,
        'predict_eval': predict_eval,
        'labels': labels,
        'outputdata': outputdata,
    }


def update_dataset_metrics(metrics_state, dataset_index, test_result):
    metrics_state['acc_f1'][dataset_index] = test_result['acc1']
    metrics_state['acc_f2'][dataset_index] = test_result['acc2']
    metrics_state['acc_ensemble'][dataset_index] = test_result['acc_ensemble']
    metrics_state['acc'][dataset_index] = test_result['acc_eval']

    confusion = metrics.confusion_matrix(test_result['labels'], test_result['predict_eval'])
    class_acc = np.diag(confusion) / np.sum(confusion, 1, dtype=np.float64)
    kappa = metrics.cohen_kappa_score(test_result['labels'], test_result['predict_eval'])
    metrics_state['A'][dataset_index, :] = class_acc
    metrics_state['k'][dataset_index] = kappa
    return class_acc, kappa


def run_one_dataset(dataset_index, seed, experiment_config, metrics_state, best_test_acc, output_dir, args):
    num_classes = experiment_config['num_classes']
    patch_size = experiment_config['patch_size']
    halfwidth = experiment_config['halfwidth']

    print('#######################idataset######################## ', dataset_index)

    data_path_s, label_path_s, data_path_t, label_path_t = get_up_pc_paths()
    data = load_data_and_preprocess(data_path_s, label_path_s, data_path_t, label_path_t, halfwidth, seed, getattr(args, 'fda', False), getattr(args, 'input_norm', 'none'))

    if getattr(args, 'm1_dump', ''):
        os.makedirs(args.m1_dump, exist_ok=True)
        np.save(os.path.join(args.m1_dump, 'target_labels.npy'),
                np.asarray(data['testY']))


    loaders = create_data_loaders(
        data['trainX'], data['trainY'], data['testX'], data['testY'],
        data['testX_s'], data['testY_s'], batch_size,
    )
    train_loader = loaders['train_loader']
    train_tar_loader = loaders['train_tar_loader']
    test_loader = loaders['test_loader']
    len_src_loader = loaders['len_src_loader']
    len_tar_dataset = loaders['len_tar_dataset']

    models = initialize_models_and_optimizers(experiment_config['n_bands'], patch_size, num_classes, DEV, args)
    G = models['G']
    F1 = models['F1']
    F2 = models['F2']
    F1_teacher = models['F1_teacher']
    contrastive_loss_func = models['contrastive_loss_func']
    optimizer_g = models['optimizer_g']
    optimizer_f = models['optimizer_f']
    target_quota_state = create_target_class_quota_state(num_classes, DEV, args)
    source_prototype_state = create_source_prototype_state(num_classes, DEV, args)
    ssti_context = None
    if getattr(args, 'ssti', False):
        from modules.ssti import build_context
        source_image = utils.load_data_pavia(data_path_s, label_path_s)[0]
        source_label_map = utils.load_data_pavia(data_path_s, label_path_s)[1]
        target_label_map = utils.load_data_pavia(data_path_t, label_path_t)[1]
        permutation = np.asarray(data['RandPerm'], dtype=np.int64)
        rows = np.asarray(data['Row'], dtype=np.int64)[permutation] - halfwidth
        cols = np.asarray(data['Column'], dtype=np.int64)[permutation] - halfwidth
        ssti_context = build_context(
            source_label_map, target_label_map, num_classes, DEV,
            source_loader=train_loader, G=G, F1=F1,
            coords=(rows, cols),
            structure=args.ssti_structure,
            source_image=source_image,
            halfwidth=halfwidth,
            pred_confidence=args.ssti_pred_conf,
            source_only_epochs=args.ssti_src_epochs,
            source_only_lr=args.ssti_src_lr,
        )
    prat_full = None
    if getattr(args, 'prat_full_mode', 'off') != 'off':
        prat_full = PhaseReliabilityResidualAmplitudeTransfer(
            bands=experiment_config['n_bands'],
            patch_size=patch_size,
            ema_momentum=args.prat_full_ema,
            mode=args.prat_full_mode,
        ).to(DEV)
    memory = None
    if getattr(args, 'memory_reliab', False):
        memory = ReliabilityPrototypeMemory(
            num_samples=len_tar_dataset,
            num_classes=num_classes,
            feature_dim=152,
            neighbor_ids=torch.zeros(len_tar_dataset, 0, dtype=torch.long),
            device=DEV,
            probability_momentum=args.memory_prob_momentum,
            feature_momentum=args.memory_feature_momentum,
            reliability_threshold=args.memory_reliab_threshold,
            pseudo_confidence=args.memory_min_conf,
            pseudo_margin=args.memory_margin,
            class_cap=args.memory_class_cap,
            use_spatial=False,
            use_guard=False,
        )

    epoch_train_start = time.time()
    epoch_test_start = epoch_train_start
    epoch_test_end = epoch_train_start
    best_seed_acc = -1.0
    rcs_reset()
    lscs_reset(num_classes)
    for ep in range(1, num_epoch + 1):
        if args.lr_schedule == 'cosine':
            progress = (ep - 1) / max(1, num_epoch - 1)
            lr_now = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for optimizer in (optimizer_g, optimizer_f):
                for group in optimizer.param_groups:
                    group['lr'] = lr_now
        train_epoch(
            ep, train_loader, train_tar_loader, G, F1, F2, F1_teacher,
            contrastive_loss_func, optimizer_g, optimizer_f,
            batch_size, len_src_loader, num_k, args, seed, num_classes, target_quota_state,
            source_prototype_state, prat_full, memory,
        )
        if getattr(args, 'm1_dump', ''):
            _wanted = [int(x) for x in str(getattr(args, 'm1_dump_epochs', '') or '').split(',') if x.strip()]
            if (ep in _wanted) or (not _wanted and ep == num_epoch):
                _diag_flush(os.path.join(args.m1_dump, f"dump_seed{seed}_ep{ep}.npz"))
        if (ep % num_epoch == 0) or args.eval_every_epoch:
            epoch_test_start = time.time()
            test_result = test_model(test_loader, G, F1, F2, DEV, len_tar_dataset, args, seed, ssti_context=ssti_context, teacher=F1_teacher)
            epoch_test_end = time.time()

            if (not args.eval_every_epoch) or test_result['acc_eval'] > best_seed_acc:
                best_seed_acc = test_result['acc_eval']
                class_acc, kappa = update_dataset_metrics(metrics_state, dataset_index, test_result)
                if test_result['acc_eval'] >= best_test_acc:
                    best_test_acc = test_result['acc_eval']
                save_dataset_result(
                    output_dir, dataset_index, seed, test_result,
                    class_acc, kappa, best_test_acc, args.eval_mode,
                )
            if args.eval_every_epoch:
                print(f"[eval] ep={ep} oa={test_result['acc_eval']:.2f} best={best_seed_acc:.2f}")

    timing = {
        'train_start': epoch_train_start,
        'train_end': time.time(),
        'test_start': epoch_test_start,
        'test_end': epoch_test_end,
    }
    return best_test_acc, timing


def main():
    output_dir, log_file, original_stdout, original_stderr = setup_local_logging(args)
    try:
        setup_wandb_logging(args)

        experiment_config = {
            'num_classes': NUM_CLASSES,
            'n_bands': N_BANDS,
            'halfwidth': args.halfwidth,
            'patch_size': 2 * args.halfwidth + 1,
        }
        run_seeds = SEEDS[args.seed_offset:args.seed_offset + args.num_seeds]
        metrics_state = create_metric_buffers(len(run_seeds), NUM_CLASSES)
        best_test_acc = 0
        last_timing = {'train_start': 0, 'train_end': 0, 'test_start': 0, 'test_end': 0}

        for dataset_index, seed in enumerate(run_seeds):
            best_test_acc, last_timing = run_one_dataset(
                dataset_index,
                seed,
                experiment_config,
                metrics_state,
                best_test_acc,
                output_dir,
                args,
            )

        evaluate_results(
            metrics_state['acc'],
            metrics_state['A'],
            metrics_state['k'],
            last_timing['train_end'],
            last_timing['train_start'],
            last_timing['test_end'],
            last_timing['test_start'],
            best_test_acc,
            NUM_CLASSES,
            args,
            metrics_state['acc_f1'],
            metrics_state['acc_f2'],
            metrics_state['acc_ensemble'],
        )
        save_summary(
            output_dir,
            metrics_state['acc'],
            metrics_state['A'],
            metrics_state['k'],
            best_test_acc,
            NUM_CLASSES,
            args,
            metrics_state['acc_f1'],
            metrics_state['acc_f2'],
            metrics_state['acc_ensemble'],
        )
        print('Saved logs and results to:', output_dir)
    finally:
        restore_logging(log_file, original_stdout, original_stderr)


if __name__ == '__main__':
    main()
