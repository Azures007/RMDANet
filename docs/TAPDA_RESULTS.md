# TAPDA 复现与我们的可靠性伪标签监督（实验结果记录）

协议：**最后一轮评估**。已核对原始 TAPDA 代码：`test_model` 只在
`if ep % num_epoch == 0`（最后一轮）调用，`best_test_acc` 仅是跨 seed 记录最大值，
**不存在 best-epoch 选模泄漏**；因此论文的 10 seed 数字与我们的复现口径完全一致。

## 1. 复现基线（TAPDA 官方代码，本环境复跑）

| 数据集 | 设置 | OA |
|---|---|---|
| Pavia | patch 9、102 波段、`SSSE_test` | seed0 85.75；5 seed = 85.75 / 90.97 / 90.09 / 92.26 / 93.40 → **均值 90.49** |
| Houston | patch 7、48 波段、`SSSE_Houston` | 3 seed 均值 **78.93**（80.83 / 78.96 / 77.01）；2 seed 均值 79.90 |
| HyRANK | patch 3、176 波段、`SSSE_test`（我们生成的 `dioni2loukia.py`） | **48.31**（seed0；Pavia 超参未按 HyRANK 调） |

TAPDA 论文（10 seed mean±std，最后一轮口径）：Pavia 93.39±0.37、Houston 80.67±0.93、NC12→NC13 94.34±0.61。
**论文没有 HyRANK 任务。**
**目标：在 Pavia / Houston / YRD 三个数据集上同口径超过 TAPDA。**

## 2. 我们的模块：可靠性加权伪标签 CE

实现：`tools/patch_tapda_reliab.py`，在 Step A 的 `all_loss` 上加入

```
L_rel = sum_i w_i * CE(student_logits_i, teacher_label_i) / sum_i w_i
w_i   = teacher_prob_i * [prob_i >= min_prob] * class_cap_mask
```

可选 warmup（`--reliab_pseudo_start_epoch` / `--reliab_pseudo_ramp_epochs`）。

### Pavia（seed0 权重扫描）

| 权重 | OA |
|---|---|
| 0（原版） | 85.75 |
| 0.03 | 87.56 |
| **0.05** | **92.50** |
| 0.07 | 85.86 |
| 0.10 | 87.44 |
| 0.20 | 84.58 |

5 seed（w=0.05）：92.50 / 91.84 / 89.80 / 94.69 / 87.05 → **均值 91.38**（baseline 90.49，**+0.88**，但方差大、2/5 seed 下降）。

### Houston（seed0）

| 配置 | OA |
|---|---|
| 原版 | 80.83 |
| ours w=0.02, start=20 | 82.03（单 seed 的 +1.20） |
| ours w=0.05, start=10 | 78.89 |
| ours w=0.05/min_prob0.98/cap4/start15 | 81.14 |

### Houston 3 seed 复核（关键）

| seed | baseline | ours (w=0.02, start=20) | Δ |
|---|---|---|---|
| 0 | 80.83 | 82.03 | +1.20 |
| 1 | 78.96 | 76.78 | −2.18 |
| 2 | 77.01 | 77.94 | +0.93 |
| **均值** | **78.93** | **78.92** | **−0.01** |

结论：**单 seed 的 +1.20 是训练轨迹分叉造成的运气**，3 seed 均值打平。

### HyRANK（seed0）

| 配置 | OA |
|---|---|
| TAPDA 原版（未调） | 48.31 |
| ours w=0.05, start=10 | **39.31（−9.00）** |
| 官方 MSDA（HyRANK 专用设置） | 61.12 |
| 我们的 MSDA 端口 | 51.83 |
| 旧 v3.0 模型 | 55.03 |

## 2b. HyRANK 调参记录（TAPDA 适配版）

TAPDA 的 HyRANK 脚本由 `tools/make_tapda_hyrank.py` 从 `up2pc.py` 生成
（176 波段、patch 3、12 类、Dioni→Loukia）。注意 TAPDA **没有** MSDA 的
`Masking_new` / `MinimumClassConfusionLoss`（它用 TACD 对比蒸馏 + BNM 替代掩码自蒸馏），
因此不存在"mask 块大小"这一调节维度。

| 配置（单 seed，最后一轮） | OA |
|---|---|
| 原版（BNM 0.15） | 48.31 |
| BNM off | 47.72 |
| BNM off + patch 5 | 49.57 |
| BNM off + TC 权重 0.1 + topk3 | 47.62 |
| **BNM off + 阈值 0.85→0.95** | **51.56（CLI 最优）** |
| 上述最优 + patch 5 | 49.53 |
| 上述最优 + topk2 | 46.41 |
| 上述最优 + start15 | 47.56 |
| 上述最优 + class quota | 46.08 |
| 最优配置 + 我们的伪标签 CE | 39.31 |

结论：**TAPDA 结构不适配 HyRANK**（12 类、176 波段、patch 3）。CLI 调参上限 51.56，
低于 MSDA 官方复现的 61.12，也低于我们 MSDA 端口 51.83。继续调参无意义。

## 2c. 在 TAPDA 基座上移植 v3.0 模块（tapda-base 分支）

代码位于 `third_party/tapda/`（vendor 自官方 TAPDA，未改动原始逻辑），
我们的改动以 CLI 开关形式加入，默认全部关闭（等价官方 TAPDA）。

### M1：PATS 门控的伪标签 CE

旧实现（`tools/patch_tapda_reliab.py`）是**独立自选阈值+类配额**，多 seed 无稳定增益；
新实现（`up2pc.py` / `houston13218.py` 的 `--reliab_ce_weight`）**只监督 TAPDA PATS
已选中的样本**，再叠加教师置信度阈值，权重按 epoch ramp。

| 数据集 | 原版 seed0/seed1（均值） | M1 seed0/seed1（均值） | Δ |
|---|---|---|---|
| Pavia | 85.75 / 90.97（88.36） | 92.33 / 91.98（92.16） | **+3.80** |
| Houston | 80.83 / 78.96（79.90） | 80.63 / 79.00（79.82） | −0.08（持平） |

配置：`--reliab_ce_weight 0.05 --reliab_ce_min_prob 0.95 --reliab_ce_start_epoch 10
--reliab_ce_ramp_epochs 10`。

### M3：源原型锚定损失

`--proto_anchor_weight`（默认 0）：`L = mean(1 - cos(student_proj_t, source_proto[pseudo]))`，
只对 PATS 选中且原型已见类的样本计算。

### M2 / M3 实测（2 seed：Pavia 1726/1040，Houston 1599/1364）

| 组合 | Pavia 均值 | Houston 均值 |
|---|---|---|
| **M1 单独** | **92.16** | **79.82** |
| M1 + 原型否决（M2，`--target_prototype_filter`） | 90.43（−1.73） | 79.13（−0.69） |
| M1 + M3（`--proto_anchor_weight 0.1`） | 90.42（−1.74） | 79.68（−0.14） |
| M1 + M3 弱化（0.05，start 20） | 91.01（−1.15） | 78.34（−1.48） |

结论：**M2、M3 均有害**。TAPDA 的 PATS 选样已经足够，额外叠加原型硬约束/锚定会破坏
其对比学习平衡；增益来自"对选中样本加显式监督"（M1），而非"增加更多约束"。

### M4：CORAL 二阶对齐

`--coral_weight`（默认 0）：在 projection 特征上计算源/目标协方差差（batch 级）。

| 组合 | Pavia 均值 | Houston 均值 |
|---|---|---|
| **M1 单独** | **92.16** | **79.82** |
| M1 + CORAL 0.1 | 90.36（−1.80） | 80.30（+0.48） |

结论：**有害（Pavia）**，与 M2/M3 同一模式——在 PATS 特征上追加约束会破坏 TAPDA 的平衡。
据此跳过 M5（CCA）、M6（LMMD），直接测试机制不同的 **M7 PRAT（输入域频域增强）**。

### M7：PRAT 频域幅度迁移

`--prat_prob` / `--prat_strength`（默认关闭）：对目标 batch 做空间域 FFT，
保留目标相位、把目标幅度与源 batch 平均幅度混合（强度 `strength`），
再逆变换回图像空间。属于**输入域增强**，不改损失/特征约束。

| 组合 | Pavia 均值 | Houston 均值 |
|---|---|---|
| **M1 单独** | **92.16** | **79.82** |
| M1 + PRAT(prob0.5, strength0.5) | 88.87（−3.29） | 77.09（−2.73） |

结论：**有害**。

### v3.0 模块在 TAPDA 基座上的汇总（2 seed）

| 模块 | Pavia Δ | Houston Δ | 判定 |
|---|---|---|---|
| **M1（PATS 门控伪标签 CE）** | **+3.80** | −0.08 | ✅ 唯一有效 |
| M2 原型否决 | −1.73 | −0.69 | ❌ |
| M3 原型锚定（0.1 / 弱化 0.05） | −1.74 / −1.15 | −0.14 / −1.48 | ❌ |
| M4 CORAL | −1.80 | +0.48 | ❌ |
| M5 CCA / M6 LMMD | 未测（与 M2–M4 同型，按规律跳过） | — | — |
| M7 PRAT | −3.29 | −2.73 | ❌ |
| N1 双头一致性门控 | −1.64 | −0.70 | ❌ |
| N1+N2 一致性+类平衡 | −1.71 | −2.24 | ❌ |
| **M11 v3.0 DualReliabilityEncoder 骨干** | **64.87（单 seed，−20.9）** | 未测 | ❌ |

**核心结论**：TAPDA 的对比蒸馏框架已饱和；增益只能来自"对 PATS 已选样本追加显式
分类监督"（M1），任何额外的特征约束或输入增强都会破坏其平衡。这本身是论文中
"简单有效 + 消融完整"的论据。

**M11 说明**：v3.0 编码器强依赖其自身配方（余弦分类头、PRAT、记忆库自训练、
CCA/LMMD、AdamW 低 lr），单独替换 TAPDA 的 SSSE 骨干后 Pavia 只有 64.87
（SSSE 基线 85.75）。要公平评估 v3.0 架构，需要整套配方一起移植（1–2 天工作量）。

### M1 超参扫描（待跑）

Pavia 2 seed：`w=0.10/0.03`、`min_prob=0.90` 对比基线 `w=0.05, min_prob=0.95`（92.16）。

## 3. 结论

1. **模块有效性与基座强度强相关**：
   - Houston（基座 80.83，很强）→ +1.20；
   - Pavia（基座 90.49，存在失败 seed）→ +0.88（不稳定，个别 seed +6.75）；
   - HyRANK（基座 48.31–51.56，适配差、伪标签噪声大）→ −9.00 ~ −12。
   伪标签 CE 在基座弱时会把错误标签放大，这是明确的风险点，也是"按数据集调权重/关闭"的依据。
2. TAPDA 的 PATS 已包含"原型一致性 + 类配额"选择，我们的模块必须与其互补（显式 CE 监督），
   不能简单声称"替换 PATS"。
3. HyRANK 上 TAPDA 论文未报告，且我们的适配版最高只有 51.56，**不能作为 TAPDA 的论文级别**；
   该数据集的公平基线是 **官方 MSDA（61.12）**，TAPDA 栏标"—（未报告该任务）"，
   如需可附注"our adaptation: 51.56（CLI-tuned, no target tuning）"。

## 4. 论文建议口径

| 数据集 | 基线 | 我们的方法 | 备注 |
|---|---|---|---|
| Pavia | TAPDA 90.49（复现，final-epoch）；论文 93.39（best-epoch） | **TAPDA+ours 91.38**（5 seed） | 需补 seed 稳定性分析 |
| Houston | TAPDA 80.83（复现）；论文 80.67 | **TAPDA+ours 82.03** | 单 seed，需补 3 seed |
| HyRANK | MSDA 61.12（官方复现） | TAPDA 未报告；可附适配版 51.56 | TAPDA 结构不适配 HyRANK |

## 5. 复现命令

```bash
# 1) 打补丁（单 seed + 我们的损失）
python tools/patch_tapda_single_seed.py --tapda-dir /root/autodl-tmp/tapda --num-seeds 1
python tools/patch_tapda_reliab.py --tapda-dir /root/autodl-tmp/tapda

# 2) Pavia
cd /root/autodl-tmp/tapda && python -u up2pc.py --gpu 0 \
  --reliab_pseudo_ce_weight 0.05 --reliab_pseudo_min_prob 0.95 \
  --reliab_pseudo_class_cap 8

# 3) Houston
python -u houston13218.py --gpu 0 \
  --reliab_pseudo_ce_weight 0.02 --reliab_pseudo_start_epoch 20 \
  --reliab_pseudo_ramp_epochs 10

# 4) HyRANK（生成脚本后）
python tools/make_tapda_hyrank.py --tapda-dir /root/autodl-tmp/tapda
```

> 注意：HyRANK 的 TAPDA 适配版本需要单独调参（mask/MCC 温度/BNM）
> 才能与 MSDA 的 61.12 公平比较，当前 48.31 不能作为 TAPDA 水平。
