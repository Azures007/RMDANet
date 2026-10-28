# PMDANet 261002

## 最优结果（唯一打包结果）
- PaviaU -> PaviaC: OA=75.86%, AA=71.17%, Kappa=0.7093
- 源域标签：6.5%（2556 samples）
- 目标域：30000 samples

## 关键数据文件
- paviaU_7gt.mat counts = [3064, 6631, 3682, 1330, 947, 18649, 5029]
- paviaC_7gt.mat counts = [7598, 9248, 2685, 7287, 2863, 3090, 6584]
  （含 9248=Tiles，来自 Pavia_standard 版本，非 Meadows 版本）
- 数据路径：Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia/

## 有效改进（相对基线 68.01%）
| 改进 | OA |
|---|---|
| 基线 main_reliability | 68.01% |
| + PRAT 0.4 | 74.89% |
| + 不确定性加权（温度 0.3） | 75.16% |
| + ema_decay 0.994 | 75.86% |

## 复现命令
cd src
python -u train_standalone_uda.py --gpu 0 --data_path <数据路径>/Datasets/Pavia --save_path ./results_pavia --source_name paviaU --target_name paviaC --num_classes 7 --ablation_id main_reliability --pavia_class_balance --band_alignment_strength 0.0 --source_train_ratio 0.065 --target_train_max_samples 30000 --pseudo_start_epoch 80 --pseudo_warm_epochs 80 --epochs 110 --eval_interval 110 --patch_size 17 --batch_size 128 --eval_batch_size 256 --lr 0.001 --num_workers 8 --seed 1236 --prat_style_max 0.4 --ema_decay 0.994

## 核心机制
- DRSE: CenterGuidedPooling (guided pooling)
- RFTM: PRAT (prat_style_max=0.4)
- RTML: Unified cross-epoch reliability
- 附加改进：不确定性加权（memory.py 温度 0.3）+ ema_decay=0.994

## 消融实验（PaviaU -> PaviaC, 6.5% 源域标签）
| 模块 | OA | Delta |
|---|---|---|
| Baseline main_reliability | 68.01% | -- |
| + PRAT 0.4 | 74.89% | +6.88 |
| + 不确定性加权 | 75.16% | +0.27 |
| + ema 0.994 | 75.86% | +0.70 |
| + CORAL | 67.86% | -0.15 |
| + LMMD | 66.66% | -1.35 |
| + 双头一致性 | 59.84% | -8.17 |
| + RBPCL 0.05/0.15/0.30 | 75.07/75.06/75.03% | ~0 |
| + cleanlab | 75.86% | 0 |
| label_smoothing 0.06 | 60.64% | -14.5 |
| PRAT ema 0.94 | 63.11% | -12.05 |

## 数据版本关键说明
- paviaC_7gt.mat 必须是 Pavia_standard 版本（含 Tiles=9248）
- 如果误用含 Meadows=42826 的版本，OA 会掉到 10-13%

## 目录结构
PMDANet_261002/
  README.md
  src/
    train_standalone_uda.py
    model.py
    memory.py
    data.py
    prat.py
    contrastive_loss.py
    mmd.py
    Weight.py
  best_result_6.5pct/
    selected_model.pt
    final_metrics.json
    training_config.json
    run.log

## 打包日期
2026-10-02
