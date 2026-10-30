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
    target_contrast_group.add_argument('--target_contrast_threshold_end', default=0.85, type=float,
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

    ours_group = parser.add_argument_group('Ours: reliability-gated pseudo-label supervision')
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
    return parser


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


def load_data_and_preprocess(data_path_s, label_path_s, data_path_t, label_path_t, halfwidth, seed):
    data_s, label_s = utils.load_data_pavia(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_pavia(data_path_t, label_path_t)
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


def create_data_loaders(train_x, train_y, test_x, test_y, test_x_s, test_y_s, batch_size):
    test_source_dataset = TensorDataset(torch.tensor(test_x_s), torch.tensor(test_y_s))
    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_tar_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
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
                source_prototype_state):
    iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(DEV)
    G.train()
    F1.train()
    F2.train()
    num_iter = len_src_loader
    target_diagnostic_state = (
        create_target_diagnostic_state(num_classes)
        if args.log_target_diagnostic else None
    )

    for batch_idx in range(1, num_iter):
        if batch_idx % len(train_tar_loader) == 0:
            iter_target = iter(train_tar_loader)
        data_source, label_source = next(iter_source)
        data_target, label_target = next(iter_target)
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

        eta = 1.0
        src_batch_size = data1.size(0)
        data = Variable(torch.cat((data1, data2), 0))
        target1 = Variable(target1)
        F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
        F1_pseudo_label_t, F1_pseudo_prob_t, _, teacher_proj_t = F1_teacher(data2)

        _, _, _, teacher_proj_s = F1_teacher(data1)
        _, student_proj_s = G(data1)
        contrastive_loss_s = args.source_contrastive_weight * compute_projection_contrastive_loss(
            teacher_proj_s,
            student_proj_s,
            target1,
            contrastive_loss_func,
        )
        update_source_prototype_state(source_prototype_state, teacher_proj_s, target1, num_classes, args)
        student_feat_t, student_proj_t = G(data2)
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
        reliab_ce_weight = get_rampup_weight(
            ep, args.reliab_ce_start_epoch, args.reliab_ce_ramp_epochs, args.reliab_ce_weight
        )
        proto_anchor_weight = get_rampup_weight(
            ep, args.proto_anchor_start_epoch, args.proto_anchor_ramp_epochs,
            args.proto_anchor_weight
        )
        pats_indices = target_contrastive_stats.get('selected_indices')
        if ((reliab_ce_weight > 0 or proto_anchor_weight > 0)
                and pats_indices is not None and pats_indices.numel() > 0):
            with torch.no_grad():
                pats_labels = F1_pseudo_label_t.detach()[pats_indices].long()
                pats_probs = F1_pseudo_prob_t.detach()[pats_indices].float()
                confident = pats_probs >= args.reliab_ce_min_prob
                if args.reliab_ce_agreement:
                    agreement = (output_t1[pats_indices].argmax(dim=1)
                                 == output_t2[pats_indices].argmax(dim=1))
                    confident = confident & agreement
            if confident.any():
                sel_idx = pats_indices[confident]
                sel_labels = pats_labels[confident]
                sel_probs = pats_probs[confident]
                sel_weight = sel_probs
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
                    ce_all = F.cross_entropy(output_t1[sel_idx], sel_labels, reduction='none')
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


def test_model(test_loader, G, F1, F2, device, len_tar_dataset, args, seed):
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

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data1, target1 = Variable(data), Variable(target)
            output, _ = G(data1)
            outputdata.append(output.cpu().numpy())
            logits1 = F1(output)
            logits2 = F2(output)
            ensemble_logits = 0.5 * (logits1 + logits2)
            eval_logits = select_eval_logits(logits1, logits2, args.eval_mode)
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
    data = load_data_and_preprocess(data_path_s, label_path_s, data_path_t, label_path_t, halfwidth, seed)

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

    epoch_train_start = time.time()
    epoch_test_start = epoch_train_start
    epoch_test_end = epoch_train_start
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
            source_prototype_state,
        )
        if ep % num_epoch == 0:
            epoch_test_start = time.time()
            test_result = test_model(test_loader, G, F1, F2, DEV, len_tar_dataset, args, seed)
            epoch_test_end = time.time()

            class_acc, kappa = update_dataset_metrics(metrics_state, dataset_index, test_result)
            if test_result['acc_eval'] >= best_test_acc:
                best_test_acc = test_result['acc_eval']
            save_dataset_result(
                output_dir, dataset_index, seed, test_result,
                class_acc, kappa, best_test_acc, args.eval_mode,
            )

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
