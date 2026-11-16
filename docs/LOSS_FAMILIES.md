# 四损失族表述（论文可直接使用）

> 目标：论文正文只出现 **4 个损失函数**，附录列出 14 个子项；**代码数值完全不变**（不改变优化顺序，不掉精度）。

## 1. 总目标

```
L = L_sup + λ₁ · L_align + λ₂ · L_cons + λ₃ · L_self
```

| 损失族 | 含义 | 子项（代码不变） | 权重 |
|---|---|---|---|
| **L_sup** | 多视图监督 | 源域 CE、PRAT 风格化 CE、光谱辅助 CE | 1 / λ_style / λ_aux |
| **L_align** | 类条件域对齐 | CCA、Source Metric、CORAL、LMMD、Prototype | λ_cca / λ_metric / λ_coral / λ_lmmd / λ_proto |
| **L_cons** | 一致性正则 | Style Pair、Scale Pair、Center Pair、Scale MI、教师 soft | λ_style_pair / λ_scale_pair / λ_center_pair / λ_scale_mi / λ_soft |
| **L_self** | 自训练 | 加权伪标签 CE | λ_pseudo |

## 2. 子项 → 损失族 映射（14 项）

| # | 子项 | 损失族 | 作用 |
|---|---|---|---|
| 1 | 源域 CE | L_sup | 源域分类主监督 |
| 2 | PRAT 风格化 CE | L_sup | 风格迁移视图监督 |
| 3 | 光谱辅助 CE | L_sup | 中心光谱分支辅助监督 |
| 4 | CCA | L_align | 类条件原型对齐（hard-negative） |
| 5 | Source Metric | L_align | 源域原型度量学习 |
| 6 | CORAL | L_align | 二阶统计对齐 |
| 7 | LMMD | L_align | 可靠性加权类条件均值对齐 |
| 8 | Prototype | L_align | 可靠目标特征 → 源原型 |
| 9 | Style Pair | L_cons | 源 vs 风格化源复合一致性 |
| 10 | Scale Pair | L_cons | 尺度权重一致性 |
| 11 | Center Pair | L_cons | 中心权重一致性 |
| 12 | Scale MI | L_cons | 尺度权重互信息正则 |
| 13 | 教师 soft 一致性 | L_cons | EMA 教师软蒸馏 |
| 14 | 加权伪标签 CE | L_self | 目标域自训练 |

## 3. 训练阶段表（课程式训练）

| 阶段 | epoch | 生效损失族 | 说明 |
|---|---|---|---|
| Warm-up | 1–16 | L_sup + L_cons + L_align(CCA) | 源域监督 + 一致性 + CCA 预热 |
| Adaptation | 17–24 | + L_self + Prototype + 教师 soft | 开始伪标签自训练 |
| Full | 25–50 | + LMMD（warm 8 epoch） | 完整对齐 |

## 4. 组级消融（用已有单子项消融聚合）

| 损失族 | 删除后主要影响（seed1237） | 结论 |
|---|---|---|
| L_align | Pavia：CCA −10.96、Source Metric −7.97、CORAL −6.52、LMMD −1.42 | 必需 |
| L_cons | Pavia：外部 pair −1.88、教师 soft −1.02、Scale MI −2.35 | 必需 |
| L_sup / L_self | 训练核心，删除即无法学习 | 必需 |

## 5. 2026 年先例（审稿人熟悉）

| 论文 | 论文口径损失数 | 内部子项打包方式 |
|---|---|---|
| D²AFC（TIP 2026） | 4 个 | L_fsl = 源 CE + 目标 CE（2 项打包） |
| SoDa2（TGRS 2026） | 3 个 | L_mmd = 光谱 MMD + 空间 MMD |
| DPSKDnet（TGRS） | 4–5 个 | cls_loss = 6 个 CE；pair_kd_loss = 3 个 KD |
| PCDM-UDA（JAG 2025） | 3 + 2 | 主阶段 3 项、微调 2 项 |

> 代码实现：`core/loss_families.py` 提供四族聚合与日志（`[loss-families]`），训练总损失表达式与数值保持不变。
