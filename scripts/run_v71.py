#!/usr/bin/env python3
"""Strict UDA launcher for original-v6 configs + opt-in CNT/SSF.

The original dataset YAML is loaded first; explicit v7 CLI options are then
applied so YAML cannot silently override user inputs. No target GT in training.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from trainer import build_parser, load_config_into_args, train

DATASETS = {'pavia': 'Pavia', 'houston': 'Houston', 'hyrank': 'HyRANK'}


def get_args(options):
    parser = build_parser()
    args = parser.parse_args([])
    args = load_config_into_args(args, str(ROOT / 'datasets' / options.dataset / 'config.yaml'))
    if options.dataset not in DATASETS:
        raise ValueError(options.dataset)
    directory = DATASETS[options.dataset]
    root = Path(options.data_root).expanduser().resolve()
    args.data_path = str(root if root.name.casefold() == directory.casefold() else root / directory)
    args.save_path = str(Path(options.out_root).expanduser().resolve() / options.dataset)
    args.run_name = f'{options.mode}_epochs{options.epochs or args.epochs}_seed{options.seed}'
    args.seed = options.seed
    args.gpu = options.gpu
    args.ssf_mode = options.mode
    args.num_workers = options.workers
    if options.epochs is not None:args.epochs = options.epochs
    if options.steps_per_epoch is not None:args.steps_per_epoch = options.steps_per_epoch
    if options.batch_size is not None:args.batch_size = options.batch_size
    if options.patch_size is not None:args.patch_size = options.patch_size
    if options.source_train_ratio is not None:
        args.source_train_ratio = options.source_train_ratio
        # A ratio is not meaningful with a fixed per-class sample budget.
        args.source_samples_per_class = 0
    if options.source_samples_per_class is not None:
        args.source_samples_per_class = options.source_samples_per_class
    if options.target_train_max_samples is not None:
        args.target_train_max_samples = options.target_train_max_samples
    if options.ssf_start_epoch is not None:args.ssf_start_epoch = options.ssf_start_epoch
    if options.lambda_cnt is not None:args.lambda_cnt = options.lambda_cnt
    if options.source_budget_train_fraction is not None:
        args.source_budget_train_fraction = options.source_budget_train_fraction
    if options.no_alignment_gate:
        args.align_maturity_gate = False
    if options.no_classwise_cnt:
        args.cnt_classwise = False
    if options.no_independent_target_probe:
        args.ssf_independent_target_probe = False
    if options.no_independent_target_reweight:
        # v6-compatible behavior: do not downweight brittle target pseudo labels.
        args.independent_target_reweight = False
        args.ssf_independent_target_probe = False
    if options.dual_head_filter:
        args.dual_head_filter = True
        args.lambda_dual_source = options.dual_source_weight
    # Enforce only the training protocol; original v6 algorithms/configs stay unchanged.
    args.use_gt_target_mask = False
    args.target_gt_mix_ratio = 0.0
    args.diagnostic_target_oa = False
    args.skip_final_evaluation = True
    return args


def main():
    p = argparse.ArgumentParser(description='RMDANet v7.1 original v6 + CNT/SSF')
    p.add_argument('dataset',choices=DATASETS)
    p.add_argument('mode',choices=['C0','C1','C2','C3','C4'])
    p.add_argument('seed',type=int)
    p.add_argument('--data-root',default='/root/autodl-tmp/Datasets')
    p.add_argument('--out-root',default=str(ROOT/'results_v71'))
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int)
    p.add_argument('--steps-per-epoch',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--patch-size',type=int)
    p.add_argument('--source-train-ratio',type=float)
    p.add_argument('--source-samples-per-class',type=int)
    p.add_argument('--target-train-max-samples',type=int)
    p.add_argument('--ssf-start-epoch',type=int)
    p.add_argument('--lambda-cnt',type=float)
    p.add_argument('--source-budget-train-fraction',type=float)
    p.add_argument('--no-alignment-gate',action='store_true')
    p.add_argument('--no-classwise-cnt',action='store_true')
    p.add_argument('--no-independent-target-probe',action='store_true')
    p.add_argument('--no-independent-target-reweight',action='store_true')
    p.add_argument('--dual-head-filter',action='store_true')
    p.add_argument('--dual-source-weight',type=float,default=0.10)
    p.add_argument('--dry-run',action='store_true')
    opt=p.parse_args()
    a=get_args(opt)
    if a.batch_size < 1:raise ValueError('batch_size must be positive')
    if a.ssf_mode in ('C1','C2','C3') and a.lambda_cnt < 0:
        raise ValueError('lambda_cnt must be nonnegative')
    results=Path(a.save_path)/a.run_name
    summary={'dataset':opt.dataset,'mode':a.ssf_mode,'seed':a.seed,
             'data_path':a.data_path,'save_path':str(results),
             'epochs':a.epochs,'patch_size':a.patch_size,'batch_size':a.batch_size,
             'steps_per_epoch':a.steps_per_epoch,
             'min_steps_per_epoch':a.min_steps_per_epoch,
             'source_train_ratio':a.source_train_ratio,
             'source_samples_per_class':a.source_samples_per_class,
             'strict_source_budget':a.strict_source_budget,
             'source_budget_train_fraction':a.source_budget_train_fraction,
             'alignment_maturity_gate':a.align_maturity_gate,
             'cnt_classwise':a.cnt_classwise,
             'ssf_start_epoch':a.ssf_start_epoch,
             'target_GT_during_train':False,'final_evaluation_in_training':False}
    print('[v7.1 config]',json.dumps(summary,ensure_ascii=False,indent=2))
    if opt.dry_run:return
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {results}')
    train(a)

if __name__=='__main__':main()
