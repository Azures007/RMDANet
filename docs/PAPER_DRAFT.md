# 顶刊论文初稿骨架（TGRS/JSTARS 正刊定位）

> 状态：框架稿，待实验数字回填。所有占位符用 `[[ ]]` 标记。

## Title（候选）
Reliability-Gated Supervision for Contrastive Domain Adaptation in Hyperspectral Imagery

## Abstract（骨架）
跨场景高光谱无监督域适应（UDA）在少标签源域条件下仍受伪标签选择偏差困扰。我们以对比蒸馏式 UDA 框架为基座，[[贡献1]]，[[贡献2]]，[[贡献3]]。在 Pavia / Houston / YRD 三个跨场景任务上，方法在同协议同 seed 设置下较复现基线分别提升 [[Pavia +x.xx]]、[[Houston +x.xx]]、[[YRD +x.xx]]。

## Contributions（三条）

1. **机制一：可靠性门控监督（M1）**。现有对比蒸馏 UDA 只把选中的目标样本用于对比学习，从未对它们施加显式分类监督；我们证明在教师高置信子集上追加概率加权 CE 可稳定提升 Pavia (+3.80) 与 Houston（配合类配额 +1.23），并给出"选样—监督"耦合的误差分析。
2. **机制二：[[待定：M2-S 软标签蒸馏 / M2-C 类自适应门控 / C 组门控]]**。针对[[长尾类 / 硬标签确认偏置]]，设计[[机制]]，在 [[Pavia/YRD]] 上带来 [[+x.xx]]，与机制一互补（消融见 Table 4）。
3. **可迁移性分析**。在 13 组同协议同 seed 对照下系统报告通用 DA 组件（余弦头、AdamW 配方、LMMD、CCA、CORAL、PRAT、记忆库、骨干替换）在对比蒸馏框架上的负迁移现象，给出适用条件。

## Method

### 3.1 问题定义
源域 $\mathcal{D}_s=\{(x_i^s,y_i^s)\}$ 每类 180 标签；目标域仅 $x^t$；目标 GT 训练期不可见，仅最终评估使用。

### 3.2 基座：对比蒸馏 UDA
EMA 教师 $f_\theta$、双分类器对抗、BNM、PATS 选样 + TACD 目标对比、源域监督对比。

### 3.3 机制一：PATS 门控的可靠性监督（M1）
选样集合 $S_b$ 由 PATS 产生；门控 $\mathcal{G}=\{i\in S_b: p_i\ge\tau\}$；损失
$$\mathcal{L}_{M1}=\frac{\sum_{i\in\mathcal{G}} w_i\,\mathrm{CE}(f(x_i^t),\hat y_i)}{\sum_{i\in\mathcal{G}} w_i},\quad w_i=p_i .$$
权重由 ramp 调度 $\lambda(e)$ 控制；超参 $\tau=0.95$，$\lambda_{\max}=0.05$。

### 3.4 机制二：[[待定]]
[[根据 G1/G2 结果填写：软标签蒸馏 / 类自适应阈值 / 成熟度门控]]

### 3.5 误差分析（动机）
[[推导：top-k 选样的类条件选择偏差 → 门控监督降低伪标签噪声期望；给出 Proposition 与直觉证明]]

## Experiments

### Table 1 主表（同协议、同 seed、5 seed 均值±std）
行：Source-only / Deep CORAL / DANN / SCLUDA / MSDA / MLUDA / S⁴DL / MTLDA（引用）/ SGLPCL（引用）/ TAPDA（复现）/ Ours。
列：Pavia OA/AA/Kappa、Houston OA/AA/Kappa、YRD OA/AA/Kappa。

### Table 2 逐类精度（三数据集）

### Table 3 与 2026 SOTA 的对照说明（协议差异注脚）
TAPDA 93.39 / 94.34 / 80.67（10 seed，原文）；CAPLG 92.72 / 76.73（原文）等。

### Table 4 消融
- M1 on/off、M1 权重、M1 门控方式（硬阈值/类自适应/软标签）
- 机制二 on/off、组合
- 类配额 on/off（Houston）
- 成熟度门控 / 类上限

### Table 5 超参敏感性
$\tau\in\{0.90,0.93,0.95,0.97\}$、$\lambda_{\max}\in\{0.03,0.05,0.10\}$、patch∈{7,9}。

### Table 6 效率
参数量、FLOPs、每 epoch 时间、显存；与 MSDA/TAPDA 对比。

### 统计检验
每数据集 5 seed，配对 t 检验 + Wilcoxon，报告 p 值。

## Limitations
- 机制一依赖教师可靠性，教师早期不可靠时收益有限；
- 目标域类别极不均衡时（YRD 小类）需专用门控；
- 未覆盖 open-set / source-free 设定。

## Reproducibility
`scripts/run_5seeds.sh`、`scripts/run_baselines_msda.sh`、`scripts/collect_results.py`、`scripts/significance_test.py`、`scripts/profile_efficiency.py`；协议与 seed 清单见 `docs/PAPER_RESULTS.md`。

## Draft: Introduction (English)

Unsupervised domain adaptation (UDA) for hyperspectral image (HSI) classification aims to transfer a model trained on a labelled source scene to an unlabelled target scene acquired by a different sensor, season, or region. Recent contrastive-distillation frameworks have pushed this setting close to supervised accuracy by combining an EMA teacher, dual-classifier adversarial alignment, batch nuclear-norm maximization, and target contrastive distillation over a selected subset of pseudo-labelled target samples. In these frameworks, however, the selected subset is used *only* as positive/negative pairs for contrastive objectives. The classification loss itself never sees the selected target samples, so the very signal that the selection procedure deemed reliable is not directly optimised.

We start from this observation. Our analysis in Section 3.5 shows that top-k target selection induces a class-conditional selection bias: classes with larger target priors contribute more selected samples, and the selected samples are systematically easier than the class-conditional pseudo-label distribution. Purely contrastive use of these samples therefore reinforces the teacher's existing decision boundary instead of correcting it. We propose a lightweight reliability-gated supervision (RGS) that adds a probability-weighted cross-entropy term on the intersection of the selected set and a high-confidence gate, linearly ramped over training. RGS introduces no new parameters, adds one forward pass of the existing classifiers, and is orthogonal to the contrastive objective.

A second issue appears when the target prior is heavily long-tailed (e.g., the YRD NC12->NC13 task). A global confidence gate then suppresses rare classes, whose pseudo-labels are exactly the ones that need supervision. We therefore introduce [[mechanism-2 name]] that [[adapts the gate per class / distils the full teacher distribution / trims noisy losses]], which improves [[dataset]] by [[x.xx]] on top of RGS.

Beyond the two mechanisms, we report a systematic same-protocol, same-seed study of eleven generic DA components (cosine classifiers, AdamW recipes, LMMD, CCA, CORAL, full PRAT, reliability memory, backbone substitution, teacher consistency, spectral-view consistency, class-wise CNT). Ten of them degrade accuracy on the contrastive-distillation base, which we summarise as a transferability analysis. This negative result is practically useful: it indicates that on strong contrastive UDA bases, generic distribution-alignment modules should not be added without per-task validation.

Experiments on three cross-scene tasks (Pavia, Houston, YRD) under a fixed 180-labels-per-class budget, five seeds, and identical evaluation protocols show that RGS improves Pavia by [[+3.80]] and Houston by [[+1.23]] over the reproduced baseline, while the reproduced baseline itself already exceeds the published TAPDA numbers on YRD (95.09 vs 94.34).

## Draft: Method sections (English)

### 3.2 Base: contrastive-distillation UDA
Let $f_\theta$ be an EMA teacher of the student $(g, h)$ (encoder plus dual classifiers), and let $\mathcal{B}_t$ be a target minibatch. The base framework (i) aligns source and target features adversarially through the two classifiers, (ii) maximises the batch nuclear norm of target predictions, and (iii) performs target contrastive distillation over a selected subset $S_b\subseteq\mathcal{B}_t$ obtained by PATS (per-class top-k with source-prototype agreement). The selected subset is used only in the contrastive term.

### 3.3 Mechanism 1: reliability-gated supervision (RGS)
We reuse the same selected subset but treat it as a supervision signal:
$$\mathcal{L}_{\mathrm{RGS}} = \lambda(e)\,\frac{\sum_{i\in\mathcal{G}} p_i\,\mathrm{CE}\big(h(g(x_i^t)),\,\hat y_i\big)}{\sum_{i\in\mathcal{G}} p_i},\qquad
\mathcal{G}=\{i\in S_b:\ p_i\ge\tau\},$$
with teacher confidence $p_i$, threshold $\tau$, and a linear ramp $\lambda(e)$ over the second half of training. RGS adds no parameters and reuses the classifiers already computed in the same iteration.

### 3.4 Mechanism 2: class-aware gating for long-tailed targets
Three variants share the same interface:
- **M2-C (class-adaptive threshold)**: $\tau_c=\tau-\alpha\,(1-\pi_c/\max_c\pi_c)$, where $\pi_c$ is the running target pseudo-label prior; rare classes are gated more permissively.
- **M2-B (inverse-prior quota)**: at most $q_c=\lceil q\cdot\bar\pi/\pi_c\rceil$ samples per class enter $\mathcal{G}$, which prevents frequent classes from dominating the supervised subset.
- **M2-S (soft-label distillation)**: replace the one-hot CE by $-\sum_c \hat p_c\log p_\theta(c|x)$ using the full teacher distribution, reducing hard-label confirmation bias.
- **M2-T (trimmed robust loss)**: drop the top-$\rho$ fraction of per-sample losses in $\mathcal{G}$, suppressing noisy pseudo-labels.

### 3.5 Selection-bias analysis (motivation)
Let $\pi_c$ be the target prior and $q_c^{(k)}$ the probability that a class-$c$ sample enters the top-$k$ selection. PATS thresholds are global, so $q_c^{(k)}$ is increasing in $\pi_c$ and decreasing in the class-conditional loss. Consequently the selected subset over-represents large classes and easy samples, and a purely contrastive objective on $S_b$ cannot correct this bias. RGS adds a de-biased supervised term whose expectation is $\sum_c \pi_c\,\mathbb{E}[\ell_c]$ restricted to $\mathcal{G}$; the class-aware mechanisms in Section 3.4 further rebalance the per-class mass of $\mathcal{G}$. [[Formal proposition and proof to be completed once M2 is fixed.]]

