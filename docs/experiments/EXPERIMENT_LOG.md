# 实验记录（每次修改后必须追加）

> 规则：每次修改代码后，记录「修改内容 → 运行配置 → 三数据集 3-seed 结果（OA mean±std，逐 seed）→ 结论」。
> 基线（纯 TAPDA，3 seed）：Houston 78.93、HyRANK 48.80、Pavia 91.14、YRD 94.56。
> 当前方法（SSTI+RATC，3 seed）：Houston 80.23、HyRANK 49.23、Pavia 92.15、YRD 92.44。

## 汇总表

| # | 日期 | 修改内容 | 运行配置 | Houston | HyRANK | Pavia | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | RGS v2 独立选样 `independent_joint`（置信度+Teacher/F2一致+源原型） | `--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint` | **81.31 ± 1.39**（82.59/79.83/81.51） | 48.71 ± 1.39（47.27/48.81/50.05） | 91.63 ± 0.36（91.34/91.52/92.04） | vs SSTI+RATC：Houston **+1.08**、HyRANK −0.52、Pavia −0.52；对 Houston 有效，HyRANK/Pavia 掉点，待测其他模式 |

## 详细记录

### 2026-11-09 修改 1：RGS 独立选样分支（independent_joint）
- 改动文件：`modules/rgs_selector.py`（新增）、`houston13218.py`/`up2pc.py`/`nc122nc13.py`（参数+接入）、`tests/test_rgs_selector.py`、`tools/rgs_v2_run.sh`
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint`
- 结果（3 seed，实验目录 `/root/autodl-tmp/experiments/20261108-234348_rgs_independent_joint`）：
  - Houston：82.59 / 79.83 / 81.51 → 81.31 ± 1.39（AA 77.87，Kappa 70.08），配对差 +2.38，3/3 正
  - HyRANK：47.27 / 48.81 / 50.05 → 48.71 ± 1.39（AA 58.61，Kappa 41.53），配对差 −0.09
  - Pavia：91.34 / 91.52 / 92.04 → 91.63 ± 0.36（AA 92.82，Kappa 89.99），配对差 +0.49
- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：Houston **+1.08**，HyRANK **−0.52**，Pavia **−0.52**
- 结论：新 RGS 对 Houston 有显著收益（且 seed 稳定性好），但相对 SSTI+RATC 在 HyRANK/Pavia 掉点；
  推测原因：独立选样缺乏类平衡（旧 PATS 有每类 top-k），HyRANK 的 class9（0.82%）等少数类被冷落。
- 下一步：测试 `independent_conf`、`independent_agree`（去掉原型条件），并考虑加 `--rgs_class_cap` 做类上限消融。

## 待测配置清单（RGS v2）

| 模式 | 命令参数 | 说明 |
|---|---|---|
| legacy | `--rgs_selector legacy`（默认） | 旧 RGS（PATS 选样），对照 |
| independent_conf | `--rgs_selector independent_conf` | 全 batch 按教师置信度 |
| independent_agree | `--rgs_selector independent_agree` | 置信度 + Teacher/F2 一致 |
| independent_proto | `--rgs_selector independent_proto` | 置信度 + 源原型支持 |
| independent_joint | `--rgs_selector independent_joint` | 置信度 + 一致 + 原型（已测，见上） |

统一公共开关（全部创新点打开）：`--ssti --ratc --reliab_ce_weight 0.05`

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
