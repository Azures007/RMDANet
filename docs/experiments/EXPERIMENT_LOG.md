# 实验记录（每次修改后必须追加）

> 规则：每次修改代码后，记录「修改内容 → 运行配置 → 三数据集 3-seed 结果（OA mean±std，逐 seed）→ 结论」。
> 基线（纯 TAPDA，3 seed）：Houston 78.93、HyRANK 48.80、Pavia 91.14、YRD 94.56。
> 当前方法（SSTI+RATC，3 seed）：Houston 80.23、HyRANK 49.23、Pavia 92.15、YRD 92.44。

## 汇总表

| # | 日期 | 修改/配置 | 运行配置 | Houston | HyRANK | Pavia | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | RGS v2 `independent_joint` | `--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint` | **81.31 ± 1.39** | 48.71 ± 1.39 | 91.63 ± 0.36 | vs SSTI+RATC：+1.08 / −0.52 / −0.52 |
| 2 | 2026-11-09 | RGS v2 `independent_agree` | `--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_agree` | 79.94 ± 0.62 | 48.93 ± 0.59 | **93.13 ± 1.30** | vs SSTI+RATC：−0.29 / −0.30 / **+0.98（3/3 正）** |
| 3 | - | - | - | - | - | - | - |

## 详细记录

### 2026-11-09 修改 1：RGS 独立选样分支（independent_joint）
- 改动文件：`modules/rgs_selector.py`（新增）、三个入口（参数+接入）、`tests/test_rgs_selector.py`、`tools/rgs_v2_run.sh`
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint`
- 结果（3 seed，实验目录 `/root/autodl-tmp/experiments/20261108-234348_rgs_independent_joint`）：
  - Houston：82.59 / 79.83 / 81.51 → 81.31 ± 1.39（AA 77.87，Kappa 70.08），配对差 +2.38，3/3 正
  - HyRANK：47.27 / 48.81 / 50.05 → 48.71 ± 1.39（AA 58.61，Kappa 41.53），配对差 −0.09
  - Pavia：91.34 / 91.52 / 92.04 → 91.63 ± 0.36（AA 92.82，Kappa 89.99），配对差 +0.49
- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：Houston **+1.08**，HyRANK **−0.52**，Pavia **−0.52**
- 结论：新 RGS 对 Houston 有显著收益；HyRANK/Pavia 相对 SSTI+RATC 掉点。

### 2026-11-09 修改 2：RGS `independent_agree`（去掉原型条件）
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_agree`
- 结果（3 seed，实验目录 `/root/autodl-tmp/experiments/20261109-004517_rgs_independent_agree`）：
  - Houston：80.52 / 79.29 / 80.02 → 79.94 ± 0.62（AA 72.46，Kappa 67.99），配对差 +1.01
  - HyRANK：48.54 / 49.61 / 48.65 → 48.93 ± 0.59（AA 60.27，Kappa 41.87），配对差 +0.13
  - Pavia：91.66 / 93.59 / 94.14 → 93.13 ± 1.30（AA 93.40，Kappa 91.76），配对差 +1.99，3/3 正
- 对比 SSTI+RATC：Houston −0.29，HyRANK −0.30，Pavia **+0.98**
- 结论：不带原型的独立选样在 Pavia 上最好（+0.98）；Houston 的 class0 在无原型约束时崩到 35.99%。

### 选样统计（来自 joint 日志）
- HyRANK：每 batch 32 样本，选 8–21 个（约 25%–65%），proto 通过率 0.50–0.88；
- Pavia：每 batch 32 样本，选 19–30 个（约 60%–94%），proto 通过率 0.75–0.97。

## 待测配置（下一步）

| 配置 | 目的 |
|---|---|
| `independent_agree --svtta` | 用 SV-TTA 修复 HyRANK 的 TENT 误选，看能否补齐 HyRANK |
| `independent_agree --rgs_class_cap 4` | 类上限，改善 HyRANK 少数类覆盖 |
| `independent_joint --svtta` | 保留 Houston 收益 + 修 HyRANK |

## 记录模板

```
### YYYY-MM-DD 修改 N：<简述>
- 改动文件：
- 运行配置：
- 结果：
  - Houston：xx.xx / xx.xx / xx.xx → mean ± std
  - HyRANK：...
  - Pavia：...
- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：
- 结论：
```
