from __future__ import print_function

import argparse
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
from util.utils import *
from util.dcl_loss_utils import get_rampup_weight

NUM_CLASSES = 7
N_BANDS = 270
YRD_DATA_PATHS = {
    'source_data': "datasets/YRD/yrd_nc12.mat",
    'source_label': "datasets/YRD/yrd_nc12_7gt.mat",
    'target_data': "datasets/YRD/yrd_nc13.mat",
    'target_label': "datasets/YRD/yrd_nc13_7gt.mat",
}
SEEDS = [1838, 1785, 1433, 1621, 1353, 1896, 1598, 1869, 1700, 1814]


_INSTABILITY_EMA = {'value': None}


_CALIBRATION = {'temperature': 1.0}


_CALIBRATION = {'temperature': 1.0}
_REFERENCE_TEACHER = {'ref': None}


def build_parser():
    parser = argparse.ArgumentParser(description='NC12 -> NC13 PDCL training')

    runtime_group = parser.add_argument_group('Training runtime')
    runtime_group.add_argument('--batch-size', type=int, default=32, metavar='N',
                               help='training mini-batch size')
    runtime_group.add_argument('--num_epoch', type=int, default=50, metavar='N',
                               help='number of training epochs for each seed')
    runtime_group.add_argument('--lr', type=float, default=0.1, metavar='LR',
                               help='learning rate used by the selected optimizer')
    runtime_group.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                               help='optimizer name: momentum, adam, or any other value for Adadelta')
    runtime_group.add_argument('--num_k', type=int, default=3, metavar='K',
                               help='number of Step C generator updates per batch')
    runtime_group.add_argument('--gpu', default=0, type=int,
                               help='CUDA device id when CUDA is available')
    runtime_group.add_argument('--no-cuda', action='store_true', default=False,
                               help='disable CUDA-specific tensor moves')
    runtime_group.add_argument('--log-interval', type=int, default=5, metavar='N',
                               help='batches between training status prints')

    data_group = parser.add_argument_group('Data and model shape')
    data_group.add_argument('--halfwidth', default=1, type=int,
                            help='half width of the spatial patch; patch_size = 2 * halfwidth + 1')

    logging_group = parser.add_argument_group('Experiment logging')
    logging_group.add_argument('--log_results', action='store_true',
                               help='enable wandb logging in addition to local logs')
    logging_group.add_argument('--log_name', type=str, default='log',
                               help='wandb run name and default local run prefix')
    logging_group.add_argument('--output_root', type=str, default='runs',
                               help='root directory for local training logs')
    logging_group.add_argument('--dataset_name', type=str, default='NC12_to_NC13',
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
    bnm_group.add_argument('--bnm_weight', default=0.0, type=float,
                           help='maximum weight for batch nuclear-norm maximization loss')
    bnm_group.add_argument('--bnm_scope', default='A', choices=['none', 'A', 'AC', 'ABC'],
                           help='training steps that receive BNM loss')
    bnm_group.add_argument('--bnm_start_epoch', default=5, type=int,
                           help='epoch to start applying BNM loss')
    bnm_group.add_argument('--bnm_rampup_epochs', default=10, type=int,
                           help='epochs used to linearly ramp up BNM weight')

    target_contrast_group = parser.add_argument_group('Target contrastive distillation')
    target_contrast_group.add_argument('--target_contrastive_weight', default=0.25, type=float,
                                       help='maximum weight for target pseudo-label contrastive distillation')
    target_contrast_group.add_argument('--target_contrast_start_epoch', default=10, type=int,
                                       help='epoch to start applying target contrastive distillation')
    target_contrast_group.add_argument('--target_contrast_rampup_epochs', default=15, type=int,
                                       help='epochs used to linearly ramp up target contrastive weight')
    target_contrast_group.add_argument('--target_contrast_threshold_start', default=0.75, type=float,
                                       help='initial confidence threshold for target pseudo labels')
    target_contrast_group.add_argument('--target_contrast_threshold_end', default=0.88, type=float,
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
    quota_group.add_argument('--target_class_quota_max_extra', default=2, type=int,
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
    prototype_group.add_argument('--target_prototype_score_weight', default=0.15, type=float,
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
    runtime_group.add_argument('--num_seeds', type=int, default=10,
                               help='number of seeds evaluated from SEEDS (default: all 10)')
    runtime_group.add_argument('--seed_offset', type=int, default=0,
                               help='index of the first seed taken from SEEDS')
    ours_group = parser.add_argument_group('Ours: reliability-gated pseudo-label supervision')
    ours_group.add_argument('--reliab_ce_weight', default=0.0, type=float,
                            help='maximum weight of the PATS-gated pseudo-label CE (0 disables)')
    ours_group.add_argument('--reliab_ce_min_prob', default=0.95, type=float,
                            help='teacher confidence threshold applied on top of the PATS selection')
    ours_group.add_argument('--reliab_ce_start_epoch', default=10, type=int,
                            help='epoch to start the pseudo-label CE')
    ours_group.add_argument('--reliab_ce_ramp_epochs', default=10, type=int,
                            help='epochs used to linearly ramp up the pseudo-label CE weight')
    ours_group.add_argument('--m1_maturity_gate', action='store_true', default=False,
                            help='scale the M1 CE weight by pseudo-label maturity')
    ours_group.add_argument('--m1_maturity_low', default=0.98, type=float,
                            help='mean confidence below which the maturity gate is 0')
    ours_group.add_argument('--m1_maturity_high', default=0.999, type=float,
                            help='mean confidence above which the maturity gate is 1')
    ours_group.add_argument('--m1_class_cap', default=0, type=int,
                            help='per-class cap of the M1 selected samples (0 disables)')
    ours_group.add_argument('--m1_class_quota', default=0, type=int,
                            help='per-class selection quota scaled by inverse target prior (0 disables)')
    ours_group.add_argument('--m1_scope', default='pats', choices=['pats', 'all'],
                            help='supervision scope: only PATS-selected samples, or all confident targets')
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
    ours_group.add_argument('--src_calibration', action='store_true', default=False,
                            help='calibrate the teacher temperature on the labelled source domain before gating')
    ours_group.add_argument('--m1_agreement_strength', default=0.0, type=float,
                            help='weight each M1 sample by the dual-classifier agreement (0 disables)')
    ours_group.add_argument('--crt_enable', action='store_true', default=False,
                            help='counterfactual reference teacher: gate supervision by drift from a lagged teacher')
    ours_group.add_argument('--crt_ref_alpha', default=0.995, type=float,
                            help='EMA momentum of the lagged reference teacher')
    ours_group.add_argument('--crt_keep_ratio', default=1.0, type=float,
                            help='legacy quantile gate fraction (1.0 = disabled, robust gate is used)')
    ours_group.add_argument('--crt_beta', default=1.0, type=float,
                            help='down-weighting strength on the robust bias z-score')
    ours_group.add_argument('--crt_outlier_k', default=2.5, type=float,
                            help='robust z threshold above which a sample is treated as drifted')
    ours_group.add_argument('--instability_adaptive', action='store_true', default=False,
                            help='adapt the RGS weight from an early training-instability score (innovation 4)')
    ours_group.add_argument('--instability_threshold', default=0.10, type=float,
                            help='instability score at which the adaptive weight reaches its maximum')
    ours_group.add_argument('--instability_base_weight', default=0.02, type=float,
                            help='RGS weight used for a stable run')
    ours_group.add_argument('--instability_max_weight', default=0.10, type=float,
                            help='RGS weight used for an unstable run')
    ours_group.add_argument('--m1_teacher_temperature', default=0.0, type=float,
                            help='teacher temperature used before gating (0 disables)')
    ours_group.add_argument('--m1_margin_gate', default=0.0, type=float,
                            help='minimum teacher top1-top2 margin (0 disables)')
    ours_group.add_argument('--m1_class_adaptive', default=0.0, type=float,
                            help='rare-class threshold relaxation (0 disables)')
    ours_group.add_argument('--m1_soft_label', action='store_true', default=False,
                            help='use the teacher distribution as a soft target')
    ours_group.add_argument('--reliab_ce_balance', default='none', choices=['none', 'inverse', 'sqrt'],
                            help='class-balanced weighting of the M1 loss')
    ours_group.add_argument('--m1_source_threshold', default=0.0, type=float,
                            help='per-class threshold adjustment from source per-class accuracy')
    ours_group.add_argument('--m1_view_consistency', default='none', choices=['none', 'noise', 'flip'],
                            help='input-level view consistency operator (noise perturbation or flip)')
    ours_group.add_argument('--m1_view_weight', default=0.1, type=float,
                            help='weight of the view-consistency JS loss')
    ours_group.add_argument('--m1_view_strength', default=0.05, type=float,
                            help='noise strength of the noise view')
    ours_group.add_argument('--m1_spatial_weight', default=0.0, type=float,
                            help='weight of the spatial-neighbourhood consistency supervision (0 disables)')
    ours_group.add_argument('--m1_spatial_alpha', default=0.5, type=float,
                            help='blend ratio between the teacher probability and its neighbourhood mean')
    ours_group.add_argument('--m1_spatial_map_decay', default=0.9, type=float,
                            help='EMA decay of the running target probability map')
    ours_group.add_argument('--m1_spatial_start_epoch', default=0, type=int,
                            help='epoch from which the spatial consistency loss is applied')
    return parser


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


def get_nc122nc13_paths():
    return (
        YRD_DATA_PATHS['source_data'],
        YRD_DATA_PATHS['source_label'],
        YRD_DATA_PATHS['target_data'],
        YRD_DATA_PATHS['target_label'],
    )


args, DEV = parse_runtime_args()
num_epoch = args.num_epoch
batch_size = args.batch_size
num_k = args.num_k


def get_sample_data(sample_data, sample_label, halfwidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:', sample_data.shape)
    n_band = sample_data.shape[2]

    data = np.pad(sample_data, ((halfwidth, halfwidth), (halfwidth, halfwidth), (0, 0)), mode='constant')
    label = np.pad(sample_label, halfwidth, mode='constant')

    train = {}
    train_indices = []
    row, column = np.nonzero(label)
    num_classes = int(np.max(label))
    print(f'num_class : {num_classes}')

    val = {}
    val_indices = []

    for i in range(num_classes):
        indices = [j for j, x in enumerate(row.ravel().tolist()) if label[row[j], column[j]] == i + 1]
        np.random.shuffle(indices)
        num = len(indices)
        if num < num_per_class:
            train[i] = indices[:num_per_class]
            train[i] += indices[:num_per_class - num]
        else:
            train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(num_classes):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    print('the number of val data:', len(val_indices))
    n_val = len(val_indices)
    val_data = np.zeros([n_val, n_band, 2 * halfwidth + 1, 2 * halfwidth + 1], dtype=np.float32)
    val_label = np.zeros([n_val], dtype=np.int64)
    rand_perm = np.array(val_indices)

    for i in range(n_val):
        val_data[i, :, :, :] = np.transpose(
            data[
                row[rand_perm[i]] - halfwidth: row[rand_perm[i]] + halfwidth + 1,
                column[rand_perm[i]] - halfwidth: column[rand_perm[i]] + halfwidth + 1,
                :
            ],
            (2, 0, 1)
        )
        val_label[i] = label[row[rand_perm[i]], column[rand_perm[i]]].astype(np.int64)
    val_label = val_label - 1

    print('the number of processed data:', len(train_indices))
    n_train = len(train_indices)
    processed_data = np.zeros([n_train, n_band, 2 * halfwidth + 1, 2 * halfwidth + 1], dtype=np.float32)
    processed_label = np.zeros([n_train], dtype=np.int64)
    rand_perm = np.array(train_indices)

    for i in range(n_train):
        processed_data[i, :, :, :] = np.transpose(
            data[
                row[rand_perm[i]] - halfwidth: row[rand_perm[i]] + halfwidth + 1,
                column[rand_perm[i]] - halfwidth: column[rand_perm[i]] + halfwidth + 1,
                :
            ],
            (2, 0, 1)
        )
        processed_label[i] = label[row[rand_perm[i]], column[rand_perm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label


def load_data_and_preprocess(data_path_s, label_path_s, data_path_t, label_path_t, halfwidth, seed):
    data_s, label_s = utils.load_data_YRD(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_YRD(data_path_t, label_path_t)
    utils.seed_everything(seed, use_deterministic=True)

    train_x, train_y = get_sample_data(data_s, label_s, halfwidth, 180)
    test_id, test_x, test_y, gr, rand_perm, row, column = utils.get_all_data(data_t, label_t, halfwidth)
    test_id_s, test_x_s, test_y_s, gr_s, rand_perm_s, row_s, column_s = utils.get_all_data(
        data_s, label_s, halfwidth
    )

    target_coords = np.stack(
        [row[rand_perm] - halfwidth, column[rand_perm] - halfwidth], axis=1
    ).astype(np.int64)
    return {
        'targetCoords': target_coords,
        'targetHW': (int(data_t.shape[0]), int(data_t.shape[1])),
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    indexed_test_dataset = IndexedTensorDataset(torch.tensor(test_x), torch.tensor(test_y))
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
    G = SSSE_Contrastive(
        input_channels=n_bands,
        patch_size=patch_size,
        n_classes=num_classes,
    ).to(device)
    F1 = ResClassifier(num_classes, training=True)
    F2 = ResClassifier(num_classes, training=True)
    F1.apply(weights_init)
    F2.apply(weights_init)

    F1_teacher = EMATeacher(G, F1, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)
    F1_reference = EMATeacher(G, F1, alpha=args.crt_ref_alpha, pseudo_label_weight='prob').to(device)
    _REFERENCE_TEACHER['ref'] = F1_reference
    contrastive_loss_func = SupConLoss(
        temperature=args.supcon_temperature,
        contrast_mode=args.supcon_contrast_mode,
    ).to(device)

    if args.cuda:
        G.cuda()
        F1.cuda()
        F2.cuda()

    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(
            list(F1.parameters()) + list(F2.parameters()),
            momentum=0.9,
            lr=args.lr,
            weight_decay=0.0005,
        )
    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    else:
        optimizer_g = optim.Adadelta(G.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(
            list(F1.parameters()) + list(F2.parameters()),
            lr=args.lr,
            weight_decay=0.0005,
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
                source_prototype_state, target_coords=None, spatial_map=None, spatial_count=None):
    iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
    criterion = nn.CrossEntropyLoss().to(DEV)
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
        data_target, label_target, target_ids = next(iter_target)
        if args.cuda:
            data1, target1 = data_source.cuda(), label_source.cuda()
            data2 = data_target.cuda()
            label_target_diag = label_target.cuda()
        else:
            data1, target1 = data_source, label_source
            data2 = data_target
            label_target_diag = label_target

        eta = 1.0
        src_batch_size = data1.size(0)
        data = Variable(torch.cat((data1, data2), 0))
        target1 = Variable(target1)
        F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
        if _REFERENCE_TEACHER['ref'] is not None:
            _REFERENCE_TEACHER['ref'].update_weights(G, F1, ep * num_iter + batch_idx)
        F1_pseudo_label_t, F1_pseudo_prob_t, teacher_feat_t, teacher_proj_t = F1_teacher(data2)

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
                quota_logits1, _ = F1(student_feat_t.detach())
                quota_logits2, _ = F2(student_feat_t.detach())
                dynamic_class_topk, quota_stats = compute_confusion_aware_class_quotas(
                    F1_pseudo_label_t,
                    F1_pseudo_prob_t,
                    quota_logits1,
                    quota_logits2,
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

        # ---- Ours: PATS-gated pseudo-label CE (M1) ----
        sel_idx = None
        sel_weight = None
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
            level = min(max(smoothed / max(args.instability_threshold, 1e-6), 0.0), 1.0) ** 2
            adaptive_weight = args.instability_base_weight + (
                args.instability_max_weight - args.instability_base_weight
            ) * level
            reliab_ce_weight = ramp_fraction * adaptive_weight
            if batch_idx % 10 == 0:
                print(
                    f"[instability] ep={ep} agree={agreement:.3f} conf={confidence_mean:.4f} "
                    f"score={smoothed:.4f} weight={reliab_ce_weight:.4f}"
                )
        pats_indices = target_contrastive_stats.get('selected_indices')
        if args.m1_scope == 'all':
            pats_indices = torch.arange(F1_pseudo_label_t.numel(), device=F1_pseudo_label_t.device)
        if reliab_ce_weight > 0 and pats_indices is not None and pats_indices.numel() > 0:
            with torch.no_grad():
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
                if args.m1_teacher_temperature > 0:
                    teacher_logits_tp, _ = F1_teacher.f(teacher_feat_t)
                    teacher_prob_tp = torch.softmax(
                        teacher_logits_tp.detach().float() / args.m1_teacher_temperature, dim=1
                    )[pats_indices]
                    pats_probs, pats_labels = teacher_prob_tp.max(dim=1)
                if args.src_calibration:
                    with torch.no_grad():
                        source_logits_cal = output_s1.detach().float()
                        source_labels_cal = target1.long()
                        best_temperature, best_nll = 1.0, None
                        for candidate_temperature in (0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0):
                            candidate_nll = F.cross_entropy(
                                source_logits_cal / candidate_temperature, source_labels_cal
                            ).item()
                            if best_nll is None or candidate_nll < best_nll:
                                best_temperature, best_nll = candidate_temperature, candidate_nll
                        _CALIBRATION['temperature'] = (
                            0.9 * _CALIBRATION['temperature'] + 0.1 * best_temperature
                        )
                        teacher_logits_cal, _ = F1_teacher.f(teacher_feat_t)
                        teacher_prob_cal = torch.softmax(
                            teacher_logits_cal.detach().float() / _CALIBRATION['temperature'], dim=1
                        )[pats_indices]
                        pats_probs, pats_labels = teacher_prob_cal.max(dim=1)
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
                    teacher_logits_mg, _ = F1_teacher.f(teacher_feat_t)
                    teacher_prob_mg = torch.softmax(
                        teacher_logits_mg.detach().float(), dim=1
                    )[pats_indices]
                    top2 = teacher_prob_mg.topk(2, dim=1).values
                    confident = confident & ((top2[:, 0] - top2[:, 1]) >= args.m1_margin_gate)
                if args.m1_source_threshold > 0:
                    with torch.no_grad():
                        source_pred = output_s1.detach().argmax(dim=1)
                        source_correct = source_pred.eq(target1.long())
                        class_acc = torch.zeros(num_classes, device=source_correct.device)
                        for class_id in range(num_classes):
                            mask = target1.long().eq(class_id)
                            if int(mask.sum().item()) > 0:
                                class_acc[class_id] = source_correct[mask].float().mean()
                        adjust = 1.0 - class_acc.clamp(0.0, 1.0)
                        class_thr = args.reliab_ce_min_prob - args.m1_source_threshold * adjust
                        confident = pats_probs >= class_thr[pats_labels]
                if args.m1_topk_ratio > 0 and pats_probs.numel() > 0:
                    topk = max(1, int(round(args.m1_topk_ratio * pats_probs.numel())))
                    confident = pats_probs >= pats_probs.topk(topk).values.min()
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
            reference_teacher = _REFERENCE_TEACHER['ref']
            if args.crt_enable and reference_teacher is not None:
                with torch.no_grad():
                    _, _, reference_features, _ = reference_teacher(data2)
                    reference_logits, _ = reference_teacher.f(reference_features)
                    reference_prob = torch.softmax(reference_logits.detach().float(), dim=1)
                    main_logits_crt, _ = F1_teacher.f(teacher_feat_t)
                    main_prob_crt = torch.softmax(main_logits_crt.detach().float(), dim=1)
                    main_selected = main_prob_crt[pats_indices].clamp_min(1e-8)
                    reference_selected = reference_prob[pats_indices].clamp_min(1e-8)
                    kl_bias = (main_selected * (main_selected / reference_selected).log()).sum(dim=1)
                    label_bias = main_selected.argmax(dim=1).ne(
                        reference_selected.argmax(dim=1)
                    ).float()
                    bias = kl_bias + label_bias
                    if bias.numel() > 1:
                        bias_median = bias.median()
                        bias_mad = (bias - bias_median).abs().median().clamp_min(1e-6)
                        bias_scale = 1.4826 * bias_mad
                        excess = ((bias - bias_median) / bias_scale).clamp_min(0.0)
                    else:
                        excess = torch.zeros_like(bias)
                    confident = confident & (excess <= args.crt_outlier_k)
                crt_factor = torch.exp(-args.crt_beta * excess)
            else:
                crt_factor = None
            if confident.any():
                sel_idx = pats_indices[confident]
                sel_labels = pats_labels[confident]
                sel_probs = pats_probs[confident]
                sel_weight = sel_probs
                if crt_factor is not None:
                    sel_weight = sel_weight * crt_factor[confident]
                if args.m1_agreement_strength > 0:
                    with torch.no_grad():
                        dual_agreement = (
                            output_t1[sel_idx].detach().argmax(dim=1)
                            == output_t2[sel_idx].detach().argmax(dim=1)
                        ).float()
                    sel_weight = sel_weight * (
                        (1.0 - args.m1_agreement_strength)
                        + args.m1_agreement_strength * dual_agreement
                    )
                if args.reliab_ce_balance != 'none' and sel_labels.numel() > 0:
                    class_counts = torch.bincount(sel_labels.long(), minlength=num_classes).float()
                    class_freq = class_counts[sel_labels.long()].clamp_min(1.0)
                    if args.reliab_ce_balance == 'inverse':
                        class_weight = 1.0 / class_freq
                    else:
                        class_weight = 1.0 / class_freq.sqrt()
                    class_weight = class_weight / class_weight.mean().clamp_min(1e-6)
                    sel_weight = sel_probs * class_weight
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
                if args.m1_soft_label:
                    with torch.no_grad():
                        teacher_logits_soft, _ = F1_teacher.f(teacher_feat_t)
                        teacher_prob_soft = torch.softmax(
                            teacher_logits_soft.detach().float(), dim=1
                        )[sel_idx]
                    log_student = F.log_softmax(output_t1[sel_idx].float(), dim=1)
                    ce_all = -(teacher_prob_soft * log_student).sum(dim=1)
                else:
                    ce_all = F.cross_entropy(output_t1[sel_idx], sel_labels, reduction='none')
                rel_ce = (ce_all * sel_weight).sum() / sel_weight.sum().clamp_min(1e-6)
                all_loss = all_loss + reliab_ce_weight * rel_ce

        if (args.m1_spatial_weight > 0 and spatial_map is not None
                and target_coords is not None and sel_idx is not None and sel_idx.numel() > 0
                and ep >= args.m1_spatial_start_epoch):
            height, width = int(spatial_map.shape[1]), int(spatial_map.shape[2])
            with torch.no_grad():
                teacher_logits_sp, _ = F1_teacher.f(teacher_feat_t)
                teacher_prob_sp = torch.softmax(teacher_logits_sp.detach().float(), dim=1)
                sample_rows = target_coords[target_ids.long(), 0].clamp(0, height - 1)
                sample_cols = target_coords[target_ids.long(), 1].clamp(0, width - 1)
                map_flat = spatial_map.view(num_classes, height * width)
                flat_index = sample_rows * width + sample_cols
                previous = map_flat[:, flat_index].clone()
                map_flat[:, flat_index] = (
                    args.m1_spatial_map_decay * previous
                    + (1.0 - args.m1_spatial_map_decay) * teacher_prob_sp.t()
                )
                count_flat = spatial_count.view(1, height * width)
                flat_count = count_flat[0, flat_index]
                count_flat[0, flat_index] = flat_count + 1.0
                neighbour_sum = torch.zeros_like(teacher_prob_sp.t())
                neighbour_cnt = torch.zeros_like(flat_count)
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1),
                                     (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    neighbour_rows = (sample_rows + d_row).clamp(0, height - 1)
                    neighbour_cols = (sample_cols + d_col).clamp(0, width - 1)
                    neighbour_flat = neighbour_rows * width + neighbour_cols
                    neighbour_sum = neighbour_sum + map_flat[:, neighbour_flat]
                    neighbour_cnt = neighbour_cnt + count_flat[0, neighbour_flat]
                safe_cnt = neighbour_cnt.clamp_min(1e-6).unsqueeze(0)
                neighbour_mean = (neighbour_sum / safe_cnt).t()
                empty_mask = (neighbour_cnt <= 0).unsqueeze(1)
                neighbour_mean = torch.where(empty_mask, teacher_prob_sp, neighbour_mean)
            soft_target = (
                (1.0 - args.m1_spatial_alpha) * teacher_prob_sp
                + args.m1_spatial_alpha * neighbour_mean
            )
            soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            log_student_sp = F.log_softmax(output_t1[sel_idx].float(), dim=1)
            spatial_ce = -(soft_target[sel_idx] * log_student_sp).sum(dim=1)
            all_loss = all_loss + args.m1_spatial_weight * (
                (spatial_ce * sel_weight).sum() / sel_weight.sum().clamp_min(1e-6)
            )

        # ---- view-consistency operator (input level) ----
        if args.m1_view_consistency != 'none' and args.m1_view_weight > 0:
            if args.m1_view_consistency == 'noise':
                view = data2 + args.m1_view_strength * torch.randn_like(data2) * data2.std()
            else:
                view = torch.flip(data2, dims=[3])
            view_feat, _ = G(view)
            view_logits = F1(view_feat)[0]
            base_prob = torch.softmax(output_t1.detach().float(), dim=1).clamp_min(1e-8)
            view_prob = torch.softmax(view_logits.float(), dim=1).clamp_min(1e-8)
            mean_prob = 0.5 * (base_prob + view_prob)
            js_loss = 0.5 * (
                (base_prob * (base_prob / mean_prob).log()).sum(dim=1).mean()
                + (view_prob * (view_prob / mean_prob).log()).sum(dim=1).mean()
            )
            all_loss = all_loss + args.m1_view_weight * js_loss

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

    data_path_s, label_path_s, data_path_t, label_path_t = get_nc122nc13_paths()
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
    spatial_map = None
    spatial_count = None
    target_coords = None
    if getattr(args, 'm1_spatial_weight', 0.0) > 0:
        height, width = data['targetHW']
        spatial_map = torch.zeros(num_classes, height, width, device=DEV)
        spatial_count = torch.zeros(1, height, width, device=DEV)
        target_coords = torch.tensor(data['targetCoords'], device=DEV)

    epoch_train_start = time.time()
    epoch_test_start = epoch_train_start
    epoch_test_end = epoch_train_start
    for ep in range(1, num_epoch + 1):
        train_epoch(
            ep, train_loader, train_tar_loader, G, F1, F2, F1_teacher,
            contrastive_loss_func, optimizer_g, optimizer_f,
            batch_size, len_src_loader, num_k, args, seed, num_classes, target_quota_state,
            source_prototype_state, target_coords, spatial_map, spatial_count,
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
