# 四损失族表述（论文可直接使用）

> 目标：论文正文只出现 **4 个损失函数**；v7.3 起代码为 **11 个加权子项**（按审阅意见删除 Center Pair、Scale Pair、Prototype Alignment、Hard Contrastive）。

## 1. 总目标

```
L = L_sup + λ1·L_align + λ2·L_cons + λ3·L_self
```

| 损失族 | 含义 | 子项（代码） | 权重 |
|---|---|---|---|
| **L_sup** | 多视图监督 | 源域 CE、PRAT 风格化 CE、光谱辅助 CE | 1 / λ_style / λ_aux |
| **L_align** | 类条件域对齐 | CCC（Class-conditional Contrastive Alignment，原 CCA）、Source Metric、CORAL、reliable class-mean alignment（linear-kernel MMD 形式，原 LMMD） | λ_ccc / λ_metric / λ_coral / λ_lmmd |
| **L_cons** | 一致性正则 | Style Pair（内部含 SKL+特征+尺度+中心）、Scale MI、教师 soft | λ_style_pair / λ_scale_mi / λ_soft |
| **L_self** | 自训练 | 加权伪标签 CE | λ_pseudo |

## 2. 子项 → 损失族映射（11 项）

| # | 子项 | 损失族 | 作用 |
|---|---|---|---|
| 1 | 源域 CE | L_sup | 源域分类主监督 |
| 2 | PRAT 风格化 CE | L_sup | 风格迁移视图监督 |
| 3 | 光谱辅助 CE | L_sup | 中心光谱分支辅助监督 |
| 4 | CCC（原 CCA） | L_align | 类条件原型对比对齐（hard-negative；论文不写作 Canonical Correlation Analysis） |
| 5 | Source Metric | L_align | 源域原型度量学习 |
| 6 | CORAL | L_align | 二阶统计对齐 |
| 7 | reliable class-mean alignment（原 LMMD） | L_align | 可靠性加权类条件均值对齐（linear-kernel MMD 形式；与 DSAN 核 MMD 不同） |
| 8 | Style Pair | L_cons | 源 vs 风格化源复合一致性（含尺度/中心距离，作为复合损失内部项） |
| 9 | Scale MI | L_cons | 尺度权重互信息正则 |
| 10 | 教师 soft 一致性 | L_cons | EMA 教师软蒸馏 |
| 11 | 加权伪标签 CE | L_self | 目标域自训练 |

## 3. 训练阶段表（课程式训练）

| 阶段 | epoch | 生效损失族 | 说明 |
|---|---|---|---|
| Warm-up | 1–16 | L_sup + L_cons + L_align(CCC) | 源域监督 + 一致性 + CCC 预热 |
| Adaptation | 17–24 | + L_self + 教师 soft | 开始伪标签自训练 |
| Full | 25–50 | + LMMD（warm 8 epoch） | 完整对齐 |

## 4. 组级消融（用已有单子项消融聚合）

| 损失族 | 删除后主要影响（seed1237） | 结论 |
|---|---|---|
| L_align | Pavia：CCC −10.96、Source Metric −7.97、CORAL −6.52、LMMD −2.17 | 必需 |
| L_cons | Pavia：Style Pair（含尺度/中心）删除 −1.88、教师 soft −1.02、Scale MI −2.35 | 必需 |
| L_sup / L_self | 训练核心，删除即无法学习 | 必需 |

## 5. v7.3 精简说明（2026-11-16）

按审阅意见删除以下 4 项，代码从 14 个加权子项变为 **11 个**：

| 删除项 | 原权重 | 删除原因（审阅意见） | 备注 |
|---|---|---|---|
| Center Pair（外部项） | λ≈0.012 | 与 Style Pair 内部中心项重复 | 中心一致性仍由 Style Pair 内部项保留 |
| Scale Pair（外部项） | λ≈0.020 | 与 Style Pair 内部尺度项重复 | 尺度一致性仍由 Style Pair 内部项保留 |
| Prototype Alignment | λ_proto≈0.03 | 与 CCC / Pseudo CE / Source Metric 功能重合 | `memory.prototype_align_flagment_loss` 保留为独立消融模块，不进入主训练 |
| Hard Contrastive | 默认 0 | 2 seed 无稳定收益 | `core/hard_contrastive.py` 保留为实验模块，不进入主训练 |

> 注：前期实验显示删除这些项会在 Pavia/HyRANK 上掉点（见 `RELIABILITY_ABLATION_RESULTS.md` 10.16/10.17），本版按审阅意见"先不考虑准确率"执行精简；准确率对比见 v7.3 结果（10.19）。

## 6. 2026 年先例（审稿人熟悉）

| 论文 | 论文口径损失数 | 内部子项打包方式 |
|---|---|---|
| D²AFC（TIP 2026） | 4 个 | L_fsl = 源 CE + 目标 CE（2 项打包） |
| SoDa2（TGRS 2026） | 3 个 | L_mmd = 光谱 MMD + 空间 MMD |
| DPSKDnet（TGRS） | 4–5 个 | cls_loss = 6 个 CE；pair_kd_loss = 3 个 KD |
| PCDM-UDA（JAG 2025） | 3 + 2 | 主阶段 3 项、微调 2 项 |

> 代码实现：`core/loss_families.py` 提供四族聚合与日志（`[loss-families]`）；v7.3 起训练总损失为 11 项加和。
