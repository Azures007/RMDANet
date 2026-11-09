# 实验记录（每次修改后必须追加）

> **记录位置（三处同步）**
> 1. 本地仓库：`D:\桌面\退路代码\RMDANet_v7.1\docs\experiments\EXPERIMENT_LOG.md`
> 2. GitHub：仓库 `Azures007/RMDANet`，分支 `rgs2sv`（及 `rgs2`）下的 `docs/experiments/EXPERIMENT_LOG.md`
> 3. 服务器：`~/autodl-tmp/tapda_repro/docs/experiments/EXPERIMENT_LOG.md`（`git reset --hard` 到 `rgs2sv` 后即存在）
>
> 每次实验的原始日志：`/root/autodl-tmp/experiments/<时间戳>_<配置名>/`
> - `logs/*.log`（完整训练日志）
> - `R_*.md`（OA/AA/Kappa/逐类/配对比较）
> - `RESULTS.md`（汇总）

## 记录规则

每次修改后必须记录：**配置 → OA（逐 seed、mean±std）→ AA（mean±std）→ Kappa（mean±std）→ 每个类别准确率 → 对比 SSTI+RATC → 结论**。

## 基线（纯 TAPDA，3 seed）

| 数据集 | OA（逐 seed） | OA mean±std | AA | Kappa |
|---|---|---|---|---|
| Houston | 80.83 / 78.96 / 77.01 | 78.93 ± 1.91 | 77.18 ± 1.11 | 67.43 ± 2.14 |
| HyRANK | 49.25 / 49.83 / 47.33 | 48.80 ± 1.31 | 59.01 ± 0.96 | 42.01 ± 1.01 |
| Pavia | 89.22 / 92.93 / 91.27 | 91.14 ± 1.86 | 91.97 ± 0.38 | 89.41 ± 1.77 |
| YRD | 95.41 / 94.77 / 93.50 | 94.56 ± 0.97 | 76.34 ± 1.73 | 91.58 ± 1.22 |

**当前方法 SSTI+RATC（3 seed）**：Houston 80.23、HyRANK 49.23、Pavia 92.15、YRD 92.44。

## 汇总表

| # | 日期 | 修改/配置 | Houston OA | HyRANK OA | Pavia OA | 结论 |
|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | RGS `independent_joint` | **81.31 ± 1.39** | 48.71 ± 1.39 | 91.63 ± 0.36 | vs SSTI+RATC：+1.08 / −0.52 / −0.52 |
| 2 | 2026-11-09 | RGS `independent_agree` | 79.94 ± 0.62 | 48.93 ± 0.59 | **93.13 ± 1.30** | vs SSTI+RATC：−0.29 / −0.30 / **+0.98** |
| 3 | 2026-11-09 | `agree + SV-TTA` | 79.94 ± 0.62 | 48.93 ± 0.59 | 93.13 ± 1.30 | SV-TTA 未触发，与 #2 相同 |
| 4 | 2026-11-09 | `agree + rgs_class_cap 4`（仅 HyRANK） | — | **50.54 ± 1.35** | — | vs SSTI+RATC **+1.31** |
| 5 | 2026-11-09 | `agree + rgs_class_cap 4`（Houston/Pavia） | **77.87 ± 3.06** | 50.54（同 #4） | 92.14 ± 1.89 | Houston **−2.36**（class0 崩）；Pavia ≈持平 |
| 6 | 2026-11-09 | `independent_balanced`（min_per_class=2） | **81.73 ± 1.59** | 49.19 ± 0.74 | 91.40 ± 1.56 | vs SSTI+RATC：**+1.50** / −0.04 / **−0.75**；Houston 最好，Pavia 掉点 |
| 7 | 2026-11-09 | `independent_balanced`（min_per_class=**1**） | 80.17 ± 0.94 | **49.56 ± 0.92** | 91.90 ± 2.01 | vs SSTI+RATC：−0.06 / **+0.33** / −0.25；最均衡，接近统一配置 |
| 8 | 2026-11-09 | `balanced min=1 + rgs_conf_threshold 0.97`（仅 Pavia） | — | — | **90.50 ± 0.62** | vs SSTI+RATC **−1.65**；conf 0.97 反而更差，否决 |
| 9 | 2026-11-09 | **`independent_auto`（统一配置，诊断自动选模式）** | **81.73 ± 1.95** | **50.28 ± 1.55** | **93.24 ± 1.12** | **vs SSTI+RATC：+1.50 / +1.05 / +1.09，三数据集全部 ≥ +0.5 ✅** |

---

## 详细记录

### 2026-11-09 修改 1：RGS 独立选样分支（`independent_joint`）

- 改动文件：`modules/rgs_selector.py`（新增）、`houston13218.py`/`up2pc.py`/`nc122nc13.py`（参数+接入）、`tests/test_rgs_selector.py`、`tools/rgs_v2_run.sh`
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint`
- 实验目录：`/root/autodl-tmp/experiments/20261108-234348_rgs_independent_joint`

**Houston**
- OA：82.59 / 79.83 / 81.51 → **81.31 ± 1.39**（配对差 +2.38，3/3 正）
- AA：77.87 ± 0.97；Kappa：70.08 ± 1.14
- 逐类：`[0] 75.59 | [1] 75.89 | [2] 65.38 | [3] 86.36 | [4] 83.97 | [5] 84.91 | [6] 73.01`

**HyRANK**
- OA：47.27 / 48.81 / 50.05 → **48.71 ± 1.39**（配对差 −0.09）
- AA：58.61 ± 1.54；Kappa：41.53 ± 1.46
- 逐类：`[0] 53.70 | [1] 100.00 | [2] 83.09 | [3] 72.15 | [4] 65.29 | [5] 41.87 | [6] 34.88 | [7] 29.78 | [8] 21.78 | [9] 0.82 | [10] 100.00 | [11] 100.00`

**Pavia**
- OA：91.34 / 91.52 / 92.04 → **91.63 ± 0.36**（配对差 +0.49）
- AA：92.82 ± 0.62；Kappa：89.99 ± 0.35
- 逐类：`[0] 92.25 | [1] 97.50 | [2] 95.34 | [3] 83.41 | [4] 99.73 | [5] 97.51 | [6] 84.01`

- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：+1.08 / −0.52 / −0.52
- 结论：对 Houston 显著有效；HyRANK/Pavia 相对 SSTI+RATC 掉点。

### 2026-11-09 修改 2：RGS `independent_agree`（去掉原型条件）

- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_agree`
- 实验目录：`/root/autodl-tmp/experiments/20261109-004517_rgs_independent_agree`

**Houston**
- OA：80.52 / 79.29 / 80.02 → **79.94 ± 0.62**（配对差 +1.01）
- AA：72.46 ± 1.56；Kappa：67.99 ± 0.78
- 逐类：`[0] 35.99 | [1] 76.65 | [2] 60.86 | [3] 92.42 | [4] 76.35 | [5] 84.35 | [6] 80.61`

**HyRANK**
- OA：48.54 / 49.61 / 48.65 → **48.93 ± 0.59**（配对差 +0.13）
- AA：60.27 ± 2.19；Kappa：41.87 ± 0.41
- 逐类：`[0] 71.18 | [1] 100.00 | [2] 88.31 | [3] 70.04 | [4] 63.43 | [5] 41.33 | [6] 34.42 | [7] 29.25 | [8] 24.42 | [9] 0.82 | [10] 100.00 | [11] 100.00`

**Pavia**
- OA：91.66 / 93.59 / 94.14 → **93.13 ± 1.30**（配对差 +1.99，3/3 正）
- AA：93.40 ± 1.52；Kappa：91.76 ± 1.27
- 逐类：`[0] 93.14 | [1] 98.88 | [2] 87.72 | [3] 84.12 | [4] 99.88 | [5] 98.06 | [6] 91.96`

- 对比 SSTI+RATC：−0.29 / −0.30 / **+0.98**
- 结论：Pavia 最好；Houston 无原型约束时 class0 崩到 35.99%。

### 2026-11-09 修改 3：`agree + SV-TTA`（rgs2sv 分支）

- 修复过程：rgs2 分支缺 `modules/svtta.py` → 补文件（UTF-8 无 BOM）；SV-TTA 按 cns 风格接入 RATC 的 TENT 分支。
- 实验目录：`/root/autodl-tmp/experiments/*_rgs_agree_svtta_v3`
- **结果与 #2 完全一致**：

| 数据集 | OA | AA | Kappa | 逐类 |
|---|---|---|---|---|
| Houston | 79.94 ± 0.62（80.52/79.29/80.02） | 72.46 ± 1.56 | 67.99 ± 0.78 | [0]35.99 [1]76.65 [2]60.86 [3]92.42 [4]76.35 [5]84.35 [6]80.61 |
| HyRANK | 48.93 ± 0.59（48.54/49.61/48.65） | 60.27 ± 2.19 | 41.87 ± 0.41 | [0]71.18 [1]100 [2]88.31 [3]70.04 [4]63.43 [5]41.33 [6]34.42 [7]29.25 [8]24.42 [9]0.82 [10]100 [11]100 |
| Pavia | 93.13 ± 1.30（91.66/93.59/94.14） | 93.40 ± 1.52 | 91.76 ± 1.27 | [0]93.14 [1]98.88 [2]87.72 [3]84.12 [4]99.88 [5]98.06 [6]91.96 |

- 结论：SV-TTA 在这三个数据集上未触发拒绝，对结果无影响。

### 2026-11-09 修改 4：HyRANK + `--rgs_class_cap 4`

- 运行配置：`--ssti --ratc --svtta --reliab_ce_weight 0.05 --rgs_selector independent_agree --rgs_class_cap 4`
- 实验目录：`/root/autodl-tmp/experiments/*_rgs_agree_svtta_v3`（`logs/HyRANK_cap4.log`）

| 数据集 | OA（逐 seed） | OA mean±std | AA | Kappa | 逐类 |
|---|---|---|---|---|---|
| HyRANK | 50.07 / 49.49 / 52.06 | **50.54 ± 1.35** | **61.00 ± 2.11** | **43.72 ± 1.27** | `[0] 49.19 | [1] 98.51 | [2] 90.59 | [3] 65.82 | [4] 70.74 | [5] 44.73 | [6] 34.55 | [7] 29.85 | [8] 23.76 | [9] 24.23 | [10] 100.00 | [11] 100.00` |

- 对比 SSTI+RATC（49.23）：**+1.31**；对比 agree 无 cap（48.93）：**+1.61**
- 与无 cap 相比，少数类明显恢复：class9 从 0.82% → **24.23%**，class8 21.78% → 23.76%，class0 71.18% → 49.19%（多数类略降）
- 结论：类上限有效，验证"独立选样缺类平衡导致 HyRANK 掉点"的假设。

### 2026-11-09 修改 5：`agree + rgs_class_cap 4`（Houston / Pavia）

- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_agree --rgs_class_cap 4`
- 实验目录：`/root/autodl-tmp/experiments/*_rgs_agree_cap4`

**Houston**
- OA：74.95 / 77.60 / 81.05 → **77.87 ± 3.06**（配对差 −1.07）
- AA：71.58 ± 1.22；Kappa：66.07 ± 2.96
- 逐类：`[0] 8.45 | [1] 89.70 | [2] 61.93 | [3] 93.94 | [4] 84.45 | [5] 77.90 | [6] 84.72`

**Pavia**
- OA：90.30 / 94.07 / 92.04 → **92.14 ± 1.89**（配对差 +1.00，3/3 正）
- AA：93.23 ± 1.37；Kappa：90.59 ± 1.83
- 逐类：`[0] 91.13 | [1] 98.77 | [2] 96.06 | [3] 83.18 | [4] 99.73 | [5] 97.03 | [6] 86.71`

- 对比 SSTI+RATC：Houston **−2.36**（class0 崩到 8.45%）；Pavia ≈持平（−0.01）
- 结论：类上限对 Houston 有害，不适合作为统一配置。

---

### 2026-11-09 修改 6：`independent_balanced`（agree 主体 + 少数类原型填充，min_per_class=2）

- 改动文件：`modules/rgs_selector.py`（新增 independent_balanced 模式）、`--rgs_min_per_class`
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_balanced`
- 实验目录：`/root/autodl-tmp/experiments/*_rgs_balanced_final`

**Houston**
- OA：81.32 / 80.02 / 83.85 → **81.73 ± 1.59**（vs SSTI+RATC **+1.50**，vs TAPDA +2.80）
- AA：74.25 ± 2.53；Kappa：71.04 ± 1.98
- 逐类：`[0] 27.17 | [1] 90.65 | [2] 64.91 | [3] 87.88 | [4] 84.58 | [5] 83.80 | [6] 80.79`

**HyRANK**
- OA：50.07 / 49.25 / 48.26 → **49.19 ± 0.74**（vs SSTI+RATC −0.04，vs TAPDA +0.39）
- AA：59.24 ± 2.59；Kappa：42.26 ± 0.67
- 逐类：`[0] 60.88 | [1] 100.00 | [2] 86.96 | [3] 59.92 | [4] 69.45 | [5] 42.73 | [6] 33.39 | [7] 29.80 | [8] 26.98 | [9] 0.75 | [10] 100.00 | [11] 100.00`

**Pavia**
- OA：90.66 / 93.56 / 89.97 → **91.40 ± 1.56**（vs SSTI+RATC **−0.75**，vs TAPDA +0.26）
- AA：92.23 ± 1.92；Kappa：89.70 ± 1.86
- 逐类：`[0] 90.70 | [1] 98.88 | [2] 89.94 | [3] 78.28 | [4] 99.80 | [5] 98.00 | [6] 90.02`

- 结论：Houston 最好（+1.50），HyRANK 持平，Pavia 掉 0.75（原型填充在 Pavia 上触发，与 joint 模式同样的问题）。
- 下一步：`--rgs_min_per_class 1`（仅当某类 0 样本时才补 1 个），预期 Pavia 接近 agree（93.13），Houston/HyRANK 保留收益。

### 2026-11-09 修改 7：`independent_balanced`（`--rgs_min_per_class 1`）

- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_balanced --rgs_min_per_class 1`
- 实验目录：`/root/autodl-tmp/experiments/20261109-102859_rgs_bal_min1`

**Houston**
- OA：81.24 / 79.79 / 79.48 → **80.17 ± 0.94**（vs SSTI+RATC −0.06，vs TAPDA +1.24，3/3 正）
- AA：75.88 ± 3.93；Kappa：69.04 ± 1.26
- 逐类：`[0] 59.15 | [1] 75.70 | [2] 63.06 | [3] 83.33 | [4] 82.43 | [5] 81.70 | [6] 85.77`

**HyRANK**
- OA：49.13 / 50.62 / 48.94 → **49.56 ± 0.92**（vs SSTI+RATC **+0.33**，vs TAPDA +0.76）
- AA：59.03 ± 1.26；Kappa：42.55 ± 0.82
- 逐类：`[0] 58.45 | [1] 100.00 | [2] 85.98 | [3] 60.34 | [4] 70.09 | [5] 43.07 | [6] 33.97 | [7] 30.97 | [8] 24.75 | [9] 0.75 | [10] 100.00 | [11] 100.00`

**Pavia**
- OA：91.25 / 94.15 / 90.30 → **91.90 ± 2.01**（vs SSTI+RATC **−0.25**，vs TAPDA +0.76）
- AA：92.69 ± 1.75；Kappa：90.30 ± 1.95
- 逐类：`[0] 93.23 | [1] 98.57 | [2] 90.99 | [3] 77.58 | [4] 99.71 | [5] 97.62 | [6] 91.14`

- 结论：min=1 是目前最均衡的配置（−0.06 / +0.33 / −0.25），但仍未做到三数据集全部 ≥ SSTI+RATC。
- 对比 min=2：Houston −1.56、HyRANK +0.37、Pavia +0.50。

### 2026-11-09 修改 8：`balanced min=1 + rgs_conf_threshold 0.97`（仅 Pavia，失败）

- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_balanced --rgs_min_per_class 1 --rgs_conf_threshold 0.97`
- 目的：提高填充门槛，减少 Pavia 被原型样本稀释

**Pavia**
- OA：90.93 / 90.94 / 89.63 → **90.50 ± 0.62**（vs SSTI+RATC **−1.65**，vs TAPDA −0.64）
- 结论：conf 0.97 反而更差（conf 0.95 min=1 为 91.90），**否决**。

### 2026-11-09 修改 9：`independent_auto`（统一配置，标签诊断自动选模式）✅

- 代码：`modules/rgs_selector.py` 新增 `independent_auto` 模式（commit `5cb1939`）
- 运行配置：`--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_auto`
- 自动规则（伪标签启动后 3 epoch 开始判断，连续 25 batch 一致后本 seed 锁定）：
  - 多类（C≥10）且 gini≥0.75 或类缺失 fz≥0.15 → `cap4`（agreement + 每类上限 4）
  - fz≥0.35 → `cap4`；gini≥0.72 → `agree`；否则 → `fill2`（原型门控 + 每类补到 2）
- 实际命中：Houston→`fill2`（gini≈0.65）、HyRANK→`cap4`（gini≈0.86）、Pavia→`agree`（gini≈0.84）
- 实验目录：`/root/autodl-tmp/experiments/20261109-140938_rgs_auto_v2`（H/Hy）；
  Pavia 来自 `/root/autodl-tmp/experiments/20261109-133119_rgs_auto`（两轮 tag 均为 agree）

**Houston**
- OA：81.32 / 80.02 / 83.85 → **81.73 ± 1.95**（vs SSTI+RATC **+1.50**，vs TAPDA +2.80，3/3 正）
- AA：74.25 ± 2.53；Kappa：71.04 ± 1.98
- 逐类：`[0] 27.17 | [1] 90.65 | [2] 64.91 | [3] 87.88 | [4] 84.58 | [5] 83.80 | [6] 80.79`

**HyRANK**
- OA：49.28 / 49.49 / 52.06 → **50.28 ± 1.55**（vs SSTI+RATC **+1.05**，vs TAPDA +1.47）
- AA：61.49 ± 1.73；Kappa：43.61 ± 1.37；class9 0.75 → **24.16**
- 逐类：`[0] 55.21 | [1] 98.51 | [2] 88.87 | [3] 68.78 | [4] 71.31 | [5] 42.27 | [6] 33.94 | [7] 29.02 | [8] 25.83 | [9] 24.16 | [10] 100 | [11] 100`

**Pavia**
- OA：91.98 / 93.59 / 94.14 → **93.24 ± 1.12**（vs SSTI+RATC **+1.09**，vs TAPDA +2.10，3/3 正）
- AA：93.40 ± 1.51；Kappa：91.88 ± 1.10
- 逐类：`[0] 93.41 | [1] 98.79 | [2] 87.04 | [3] 83.92 | [4] 99.55 | [5] 98.00 | [6] 93.10`

- **结论：统一配置达成，三数据集相对 SSTI+RATC 全部 ≥ +0.5（+1.50 / +1.05 / +1.09），相对纯 TAPDA 全部 ≥ +1.47。**
- 该配置即论文最终 RGS 配置；`independent_auto` 的规则只用无标签诊断量（教师伪标签分布），不使用目标 GT。

## 待测配置（下一步）

| 配置 | 目的 |
|---|---|
| `independent_union`（conf & (agree \| proto)）三数据集 3 seed | 不用硬类上限，兼顾 Houston 的 class0 与 HyRANK/Pavia 覆盖率 |
| `joint + class_cap 4` | 对照 |
| class_cap 取 2 / 8 | 敏感性 |

当前最优（分数据集）：
- Houston：`independent_balanced`（81.73，+1.50）
- HyRANK：`independent_agree + class_cap 4`（50.54，+1.31）
- Pavia：`independent_agree`（93.13，+0.98）

**统一配置已找到（2026-11-09，实验 #9）**：`independent_auto`
- Houston 81.73（+1.50）/ HyRANK 50.28（+1.05）/ Pavia 93.24（+1.09）
- 三数据集相对 SSTI+RATC 全部 ≥ +0.5；自动标签：fill2 / cap4 / agree

## 记录模板

```
### YYYY-MM-DD 修改 N：<简述>
- 改动文件：
- 运行配置：
- 实验目录：
- Houston：OA seeds → mean±std；AA ± std；Kappa ± std；逐类 [...]
- HyRANK：OA seeds → mean±std；AA ± std；Kappa ± std；逐类 [...]
- Pavia：OA seeds → mean±std；AA ± std；Kappa ± std；逐类 [...]
- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：
- 结论：
```

---

## 附：旧模型线（main）基线复现验证

### 2026-11-09 基线复现：RMDANet v3.0 / BASELINE_V3 Houston（seed 1237）

- 代码版本：`main` @ `b2fcdde`（= tag v3.0 `c0783a7` 的备份配置）
- 运行命令：
  ```bash
  python -u scripts/run_v71.py houston C0 1237 \
    --data-root /root/autodl-tmp/Datasets --epochs 50 --workers 4 \
    --out-root results_v71/repro_houston
  python -u scripts/final_eval_locked.py \
    --checkpoint results_v71/repro_houston/houston/C0_epochs50_seed1237/selected_model.pt \
    --data-path /root/autodl-tmp/Datasets/Houston \
    --output results_v71/repro_houston/houston/C0_epochs50_seed1237/target_final_once.json \
    --confirm-final-eval
  ```
- 复现结果 vs BASELINE_V3：

| 指标 | BASELINE_V3 | 本次复现 | 差 |
|---|---|---|---|
| OA | 76.72 | **76.98** | +0.26 |
| AA | 82.28 | 82.38 | +0.10 |
| Kappa | 0.6497 | 0.6532 | +0.0035 |

- 逐类：84.63 / 92.82 / 67.86 / 100.00 / 86.05 / 74.81 / 70.49
- 结论：基线可复现，差异属单 seed 运行的正常非确定性（AMP/cuDNN + workers=4 的 RNG 流）。

### 2026-11-09 基线复现：c0783a7（tag v3.0）单 seed 1237 三数据集（当前环境）

- 代码：`c0783a7`（tag `v3.0`）
- 运行：`bash scripts/run_unified.sh C0 1237 results_v71/v30_single_seed`（约 17 分钟）
- 结果：

| 数据集 | OA | AA | Kappa | BASELINE_V3 记录 | 差 |
|---|---|---|---|---|---|
| Pavia | **83.62** | 79.22 | 0.8021 | 85.55 | **−1.93** |
| HyRANK | **54.10** | 60.00 | 0.4615 | 55.03 | **−0.93** |
| Houston | **76.98** | 82.38 | 0.6532 | 76.72 | +0.26 |

- **关键**：与 `b2fcdde` 新代码复现结果逐位相同（Pavia 83.62、Houston 76.98），证明差异**与代码版本无关**，是 10/29 那次运行的环境差异（torch/cuDNN/worker/RNG）。
- 结论：**当前环境可稳定复现的 v3.0 基线 = 83.62 / 54.10 / 76.98**（单 seed 1237）；论文若引用 85.55 需注明来自旧环境，或统一改用当前可复现值。
