# TAPDA 问题证据包（Houston13 → Houston18）

> 本文件记录"TAPDA 在目标域缺少可靠判别监督"的全部实验证据，供论文写作与后续复现使用。
> 数据产生于 `TAPDA+SSTI` 代码线（分支 `rcs`，含诊断开关），GPU：RTX 4090。

## 0. 实验设置

| 项 | 值 |
|---|---|
| 数据集 | Houston13 → Houston18（7 类 / 48 波段 / patch 7×7） |
| 源域标签 | 每类 180，共 1260（无源域验证集） |
| 训练 | 50 epoch；每 epoch 39 个源域 batch；目标域每轮采样 1248 个样本 |
| 评估 | 最后 epoch；目标测试集 53,200 像素 |
| 随机性 | CUDA 种子 + cudnn deterministic（可复现） |
| 诊断种子 | 1599（10-seed 结果见 §6） |

## 1. 结论

TAPDA 完全依赖源域标签 + 特征对齐获得目标域判别能力；其目标域教师严重过置信
（置信度 0.945 vs 准确率 79.25%）；PATS 每轮丢弃 57.3% 目标样本，且被丢弃样本的
伪标签可靠性并不低于保留样本（79.72% vs 78.61%），过滤还存在类别偏置。
因此任何基于置信度阈值或源域校准的目标域伪标签监督，都会以约 20% 的噪声率供样，
无法带来稳健增益。

## 2. 证据 1：TAPDA 没有目标域判别监督

TAPDA 的损失构成（`houston13218.py`）：

| 损失项 | 代码位置 | 是否直接监督目标域分类 |
|---|---|---|
| 源域 CE | `loss1/loss2 = criterion(output_s1/s2, target1)` | 否 |
| 源域对比 | `compute_projection_contrastive_loss(teacher_proj_s, student_proj_s)` | 否 |
| 目标对比蒸馏 TACD | `compute_target_contrastive_loss(teacher_proj_t, student_proj_t, ...)` | 否（仅特征） |
| 双分类器对抗 / BNM | `bnm_loss_a`, discrepancy | 否 |
| 目标域 CE | **不存在** | — |

## 3. 证据 2：目标域教师严重过置信（seed 1599）

| epoch | 教师平均置信度 | 教师伪标签准确率 | 差距 |
|---|---|---|---|
| 20 | 0.9396 | 80.29% | 13.7 pt |
| 35 | 0.9340 | 78.69% | 14.7 pt |
| 50 | 0.9450 | 79.25% | **15.3 pt** |

## 4. 证据 3：PATS 过滤丢弃 57% 目标样本，且被丢弃样本同样可靠

| epoch | N_pre | N_post（保留） | N_excluded（排除） | 排除比例 | 保留集教师 acc | 排除集教师 acc |
|---|---|---|---|---|---|---|
| 20 | 1248 | 522 | 726 | 58.17% | 80.08% | **80.44%** |
| 35 | 1248 | 529 | 719 | 57.61% | 79.96% | 77.75% |
| 50 | 1248 | 533 | 715 | 57.29% | 78.61% | **79.72%** |

置信度：保留集 0.9876 / 0.9884 / 0.9927；排除集 0.9052 / 0.8939 / 0.9095。

### 类别偏置（ep50）

| 类别 | 总数 | PATS 保留 | 排除 | 排除样本教师准确率 |
|---|---|---|---|---|
| 0 | 37 | 19 | 18 | 22.22% |
| 1 | 113 | 96 | 17 | 88.24% |
| 2 | 71 | 32 | 39 | 53.85% |
| 3 | 2 | 0 | 2 | 100% |
| 4 | 111 | 69 | 42 | 54.76% |
| 5 | 759 | 233 | **526** | **88.02%** |
| 6 | 155 | 84 | 71 | 59.15% |

→ 被排除的 715 个样本中有 526 个来自 class 5，其教师准确率高达 88%。

## 5. 证据 4：RGS 与"被排除集"的交集为 0

| epoch | RGS 监督样本数 | RGS ∩ 被排除集 |
|---|---|---|
| 20 | 481 | **0** |
| 35 | 492 | **0** |
| 50 | 505 | **0** |

原因（代码）：`sel_idx = pats_indices[confident]`，RGS 只从 PATS 保留集中取样本。

## 6. 证据 5：源域可靠性标准无法迁移到目标域

| 指标 | 源域（RCS 校准器） | 目标域（实测） |
|---|---|---|
| 教师正确率 | 93.2%（positive rate 0.932） | 79.25% |
| 估计伪标签错误率 | 0.68%（certified risk ≈ 0.0068） | 真实错误率 ≈ 20.75% |
| 低估倍数 | — | **约 30 倍** |

## 7. 证据 6：所有"目标域补充监督"变体均无稳健增益

| 变体 | 协议 | OA | 相对 TAPDA |
|---|---|---|---|
| TAPDA（基线） | 10 seed | **80.18 ± 1.52** | — |
| 全部样本 CE | 1 seed (1599) | 80.84 | +0.01 |
| 随机等量 CE | 1 seed (1599) | 77.67 | −3.16 |
| RGS（PATS∩高置信） | 3 seed | 77.41 | −1.52 |
| RGS-v2（被排除∩高置信） | 10 seed | 80.44 | +0.25（p=0.61） |
| RCS（风险认证） | 3 seed | 79.93 | +1.00（−1.12/+2.42/+1.68） |
| RGS 去掉置信门控 | 1 seed (1599) | 80.90 | +0.07 |

### TAPDA 10-seed 逐类精度（AA 76.42）

| 类别 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Acc | 56.92 | 81.29 | 59.41 | 92.73 | 83.51 | 82.51 | 78.58 |

## 8. 复现命令

```bash
# 1) Houston 诊断 dump（RGS 开启、不加 SSTI）
python -u houston13218.py --gpu 0 --num_seeds 1 \
  --reliab_ce_weight 0.05 --m1_maturity_gate \
  --m1_dump /root/autodl-tmp/diag/houston --m1_dump_epochs 20,35,50

# 2) 离线分析（目标 GT 仅用于分析，不参与训练）
python tools/analyze_rgs_dump.py --dataset houston \
  --dump-dir /root/autodl-tmp/diag/houston \
  --out /root/autodl-tmp/diag/report_houston.md

# 3) 干预实验（全部样本 / 随机 / PATS / 被排除 / 去门控）
python -u houston13218.py --gpu 0 --num_seeds 1 --reliab_ce_weight 0.05 --m1_maturity_gate --m1_select all
python -u houston13218.py --gpu 0 --num_seeds 1 --reliab_ce_weight 0.05 --m1_maturity_gate --m1_select random
python -u houston13218.py --gpu 0 --num_seeds 1 --reliab_ce_weight 0.05 --m1_maturity_gate
python -u houston13218.py --gpu 0 --num_seeds 1 --reliab_ce_weight 0.05 --m1_maturity_gate --m1_select excluded
python -u houston13218.py --gpu 0 --num_seeds 1 --reliab_ce_weight 0.05 --m1_maturity_gate --m1_select excluded --reliab_ce_min_prob 0.0

# 4) RCS 风险认证
python -u houston13218.py --gpu 0 --num_seeds 1 \
  --reliab_ce_weight 0.05 --reliab_ce_min_prob 0.0 --rcs --rcs_delta 0.05 --rcs_max_frac 1.0

# 5) 10-seed 基线
python -u houston13218.py --gpu 0 --num_seeds 10
```

## 9. 相关代码

| 文件 | 作用 |
|---|---|
| `modules/rcs.py` | RCS：源域校准 + 风险认证选择 |
| `tools/analyze_rgs_dump.py` | 诊断 dump 的离线分析（N_pre/N_post/准确率/交集） |
| `--m1_dump / --m1_dump_epochs` | 训练期导出逐样本诊断数据（不参与训练） |
| `--m1_select {pats,all,random,excluded,rcs}` | 样本选择模式（干预实验用） |
| `--rcs*` | 风险认证补充监督开关 |
