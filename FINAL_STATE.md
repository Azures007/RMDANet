# RMDANet v6.0 — 代码清理版

## 版本主题
三数据集统一 + 死代码删除 + 未用 loss 层删除

## 三数据集最优结果

| 数据集 | OA | AA | Kappa |
|---|---|---|---|
| Pavia | 86.56 | 84.81 | 0.8384 |
| HyRANK | 60.92 | 67.15 | 0.5384 |
| Houston | 76.64 | 82.07 | 0.6480 |

## 相对 v5.1 的改动

### 删除的死代码（约 47 行）
- `GradientReversalFunction` / `GradientReversal` / `DomainDiscriminator`
- `import math`（prat.py）/ `from dataclasses import asdict`（trainer.py）
- `self.num_samples`（memory.py）
- `class_histogram`（data.py）

### 删除的未用 loss 层（约 150 行）
- CEMPA loss：`cempa_start_epoch` / `cempa_warm_epochs` / `prototype_losses` / `cempa_ramp`
- spatial loss：`spatial_start_epoch` / `spatial_warm_epochs` / `spatial_consistency_loss` / `spatial_ramp` / `lambda_spatial`
- entropy loss：`entropy_start_epoch` / `entropy_warm_epochs` / `entropy_ramp` / `entropy_loss` / `lambda_entropy`
- 辅助开关：`use_cempa_losses` / `use_spatial_loss`

### 保留的功能（真在用）
- `use_spatial` / `use_spatial_reliability`（memory.update 里的 spatial veto）
- `use_target_prototypes`（refine_probabilities 用）
- `cempa_num_subprototypes` / `cempa_class_cap` / `cempa_min_age` / `cempa_min_streak`（target prototype 更新参数）

## 复现
```bash
cd ~/autodl-tmp/pavia-test/RMDANet_v6.0
bash scripts/run_pavia.sh 1236
