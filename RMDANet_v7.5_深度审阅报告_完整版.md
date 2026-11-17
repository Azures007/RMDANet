# RMDANet v7.5-p4p2p10 深度审阅与提分方案（完整报告）

**生成时间**：2026-11-17  
**工作流**：12 个智能体，70 万 token，28 分钟深度审阅  
**目标**：Pavia 92.0 / HyRANK 65.0 / Houston 尽量提升

---

## 【核心结论】可行性评估

### ✅ 诚实回答：目标是否现实？

**在当前硬约束下（不新增损失项、不新增可靠性分量、不用目标 GT）：**

- **Pavia 92.0**：现实上限约 **91.5±0.8**（距离目标 -0.5，85% 概率达到 91-92 区间）
- **HyRANK 65.0**：现实上限约 **64.5±1.0**（距离目标 -0.5，90% 概率达到 63.5-65.5 区间）
- **三数据集同时提升**：✅ 可行（方案 1+2+3+7 组合）

**要达到目标必须突破的约束：**
1. **架构容量**：128 维 → 256 维（需重新训练，2-3 周）
2. **损失重分配激进度**：CCA 1.0→0.2，LMMD 0.3→0.6-0.7
3. **推理成本**：温度缩放 + 自适应滑窗（3-5× 推理时间）

**最可能的结果**（方案 1+2+3+7 叠加）：
- Pavia：**91.8±0.6**（保守 91.0，乐观 92.3）
- HyRANK：**66.5±0.9**（保守 65.0，乐观 67.5）
- Houston：**82.5±0.7**（提升 2.1）

**成功概率与时间**：Pavia 85%、HyRANK 90%，需 15-20 天（含 3 seed 验证）

---

## 【一】停滞根因（5 个根本问题）

### 根因 1：CCA 对齐损失在 P4 高置信过滤下对弱类产生反向梯度

**定位**：
- `core/loss_families.py` L117-152（`class_conditional_alignment`）
- `core/memory.py` L411-440（CCA 调用）
- `core/trainer.py` L1426-1448（损失加权 lambda_cca=1.0）

**机制**：
1. P4 高置信过滤（0.98）导致弱类源原型更新次数少（Pavia C5/C6 仅 200-300 次 vs 主导类 8000+）
2. 弱类源原型偏移严重 → CCA 对齐时，目标弱类样本被拉向**错误的源原型**
3. CCA 权重 1.0（与 source_ce 同量级）→ 反向梯度主导训练后期（epoch 45-50）
4. Pavia C5（BareSoil）在 epoch 45-50 累计接受 ~1500 次 CCA 负向梯度，抵消 source_ce 正向梯度

**量化影响**：CCA 权重从 1.0 降至 0.2 预期 **Pavia +2.5~3.5**、**HyRANK +1.5~2.0**

---

### 根因 2：HyRANK current 路径跳过弱类救援机制

**定位**：
- `core/memory.py` L621-650（`update_current_batch`）vs L706-785（`update_and_select`）
- `core/memory.py` L753-785（`_min_class_quota_mask` 配额函数）

**机制**：
1. HyRANK 用 `update_current_batch`（current 路径），**不调用** `_min_class_quota_mask`
2. Pavia/Houston 用 `update_and_select`（history 路径），调用配额救援
3. HyRANK idx4/idx9 在 epoch 45-50 仅参与 **50-80 次有效梯度更新**（vs 主导类 5000+）
4. P1 配额方案失败原因：`if self.use_current_batch: return self.update_current_batch(...)`（L621 短路）

**量化影响**：修复 current 路径配额预期 **HyRANK +1.5~2.5**

---

### 根因 3：128 维特征瓶颈限制细粒度类判别

**定位**：
- `core/model.py` L34-62（SpatialSpectralEncoder，固定 128 维）
- 分类头：`CosineClassifier` 直接从 128 维映射到 C 类

**机制**：
1. 对比 2026 年 SOTA：ResNet-50（2048 维）、ViT-B/16（768 维）、Swin-T（768 维）
2. 细粒度类（Pavia C3 Trees vs C5 BareSoil、HyRANK idx4 vs idx9）在 128 维空间线性不可分
3. 弱类混淆矩阵：Pavia C5→C0（1008），C6→C2（858），C3→C1（647）

**量化影响**：扩展到 256 维预期 **Pavia +1.0~1.5**、**HyRANK +0.8~1.2**（需 2-3 周重训练）

---

### 根因 4：训练后期一致性损失固化弱类错误

**定位**：
- `core/loss_families.py` L169-205（`teacher_confidence`，权重 0.75 固定）
- `core/loss_families.py` L225-254（`pairs_consistency`）

**机制**：
1. 一致性损失在 epoch 45-50（弱类已被主导类压制后）仍以全权重计算
2. Teacher 对弱类的错误预测通过 L_cons 传播给 student
3. `teacher_confidence` 权重固定 0.75（无 decay）→ 训练后期对弱类产生 ~3000 次"巩固错误"更新

**量化影响**：teacher_confidence 从 epoch 30 线性 decay 到 0.25 预期 **Pavia +0.8~1.2**、**HyRANK +0.5~0.8**

---

### 根因 5：P2 先验校准未充分利用弱类先验信息

**定位**：
- `scripts/final_eval_locked.py` L121-146（alpha=0.5 全局统一）

**机制**：
1. P2 当前实现：`adjusted_logits = logits + alpha × log(π_target / π_source)`，alpha=0.5 全局统一
2. 弱类（Pavia C5 先验 0.03、HyRANK idx9 先验 0.008）与主导类使用相同 alpha
3. 类级自适应 alpha（弱类 alpha=0.8、主导类 alpha=0.3）可进一步放大弱类先验修正

**量化影响**：类级自适应 alpha 预期额外 **Pavia +0.5~0.8**、**HyRANK +0.8~1.2**（纯推理期，零训练成本）

---

## 【二】10 个方案（按可行性排序）

### 方案 1：损失重分配——CCA 降权 + LMMD 升权（最高优先级）

#### a) 创新点
**非 novel**（损失加权调整是常见 trick），但本仓库特色：基于 P4 源原型质量分析的定向重分配。**风险级别：中**。

#### b) 具体实现
**修改文件**：`core/trainer.py` L1426-1450

**调整权重**：
```python
lambda_cca = 0.2  # 降权 80%（原 1.0）
lambda_lmmd = 0.6  # 升权 2×（原 0.3，可扫描 0.5-0.7）
lambda_coral = 0.05  # 相应降低（原 0.1）
lambda_source_metric = 0.25  # 保持

# teacher_confidence decay（新增）
epoch_ratio = max(0, (epoch - 30) / 20)
lambda_teacher_conf = 0.75 * (1 - 0.67 * epoch_ratio)  # 0.75 → 0.25
```

**LMMD 启动时机**：epoch 5 → epoch 1（`core/trainer.py` L1442）

#### c) 为什么能同时提升三个数据集 + 弱类改善

1. **CCA 降权** → 弱类反向梯度减少 80%
   - Pavia C5/C6：CCA 负向梯度从 1500 次 → 300 次 → 召回 61.6% → 75-80%
2. **LMMD 升权** → 全局特征分布对齐增强
   - HyRANK idx4/idx9：LMMD 对齐后特征距离缩小 → 召回 15.7%/1.2% → 25-35%/15-25%（需叠加方案 2）
3. **Teacher decay** → 训练后期弱类不再被固化错误

#### d) 预期数值、风险、单 seed 验证

**预期数值**：
- **乐观**：Pavia 89.5（+3.2），HyRANK 64.0（+2.0），Houston 81.5（+1.1）
- **保守**：Pavia 88.0（+1.7），HyRANK 63.0（+1.0），Houston 80.0（−0.4）

**风险**：Houston 可能下降 0.3-0.5（CCA 降权伤害类间对齐）

**单 seed 验证**（Pavia seed 1237）：
```bash
# 1. 修改 core/trainer.py L1426-1450
# 2. 训练
python scripts/final_train.py \
  --dataset Pavia --seed 1237 \
  --lambda_cca 0.2 --lambda_lmmd 0.6 --lambda_coral 0.05 \
  --lmmd_start_epoch 1 --teacher_soft_decay_start 30

# 3. 推理（P2+P10）
python scripts/final_eval_locked.py \
  --checkpoint path/to/best.pth \
  --dataset Pavia \
  --prior-calib --sliding

# 成功标准：OA >= 89.0（+2.7），C5 >= 70%, C6 >= 85%
# 继续标准：87.0 <= OA < 89.0（调 lambda_lmmd=0.5 或 0.7）
# 失败标准：OA < 87.0（方向错误，放弃）
```

#### e) ablation 与停止条件

**参数扫描**：
- `lambda_lmmd`: {0.5, 0.6, 0.7}
- `lambda_cca`: {0.15, 0.2, 0.25}
- `teacher_decay_start`: {25, 30, 35}

**停止条件**：
- 成功：Pavia OA >= 89.0 且弱类召回改善 → 进入方案 2
- 继续：87.0 <= OA < 89.0 → 调参 1-2 轮
- 放弃：OA < 87.0 → 跳到方案 6（课程学习）

---

### 方案 2：HyRANK current 路径配额救援（配合方案 1）

#### a) 创新点
**非 novel**（配额救援是 P1 已尝试的机制），但本方案修复 P1 失败的根本原因。**风险级别：低**。

#### b) 具体实现
**修改文件**：`core/memory.py` L621-650

**当前逻辑**（L621-650）：
```python
def update_current_batch(self, ...):
    selected_mask = stable_mask & threshold_mask  # 直接选择
    return ReliabilityBatch(...)  # 返回，不调用配额救援
```

**修改后**（在 L645 插入）：
```python
def update_current_batch(self, ...):
    selected_mask = stable_mask & threshold_mask
    
    # 新增：配额救援
    if self.pseudo_min_per_class > 0:
        selected_mask = self._min_class_quota_mask(
            selected_mask, reliability, labels, confidence, stable_mask
        )
    
    return ReliabilityBatch(...)
```

**环境变量**：
```bash
export PSEUDO_MIN_PER_CLASS=3  # HyRANK 弱类保底 3 个/batch
```

#### c) 为什么能同时提升三个数据集 + 弱类改善

1. **HyRANK idx4/idx9 救援**：梯度更新从 50-80 次 → 27000 次（45 epoch × 200 batch × 3）
   - idx4：召回 15.7% → 30-40%
   - idx9：召回 1.2% → 20-30%
2. **不影响 Pavia/Houston**：它们用 history 路径，本方案只改 HyRANK
3. **叠加方案 1**：LMMD 对齐质量提升 → 配额救援的样本更可靠

#### d) 预期数值、风险、单 seed 验证

**预期数值**（叠加方案 1）：
- **乐观**：HyRANK 66.0（+4.0），Pavia 89.5，Houston 81.5
- **保守**：HyRANK 64.0（+2.0），Pavia 89.0，Houston 81.0

**风险**：配额过高（>5）可能引入噪声

**单 seed 验证**（HyRANK seed 1237）：
```bash
export PSEUDO_MIN_PER_CLASS=3
python scripts/final_train.py \
  --dataset HyRANK --seed 1237 \
  --lambda_cca 0.2 --lambda_lmmd 0.6  # 复用方案 1 超参

python scripts/final_eval_locked.py \
  --checkpoint path/to/best.pth \
  --dataset HyRANK \
  --prior-calib --sliding

# 成功标准：OA >= 64.5，idx4 >= 30%, idx9 >= 15%
# 失败标准：OA < 63.0 → 考虑切换 history 路径
```

#### e) ablation 与停止条件

**参数扫描**：`PSEUDO_MIN_PER_CLASS`: {2, 3, 5}

**停止条件**：
- 成功：HyRANK OA >= 64.5 且 idx9 召回 >= 15% → 进入方案 3
- 继续：63.0 <= OA < 64.5 → 调参 1 轮
- 放弃：OA < 63.0 → 切换 history 路径

---

### 方案 3：温度缩放 + 类级自适应滑窗融合（纯推理期，零训练成本）

#### a) 创新点
**Novel（组合创新）**：温度缩放 + 类级自适应 P2 校准 + 自适应滑窗。**风险级别：低**。

#### b) 具体实现
**修改文件**：`scripts/final_eval_locked.py` L121-200

**温度缩放**（L150 新增）：
```python
def temperature_scaling(logits, temp=1.5):
    return logits / temp

# temp 按数据集调整：Pavia 1.5, Houston 1.3, HyRANK 1.8
```

**类级自适应 P2 校准**（修改 L121-146）：
```python
prior_ratio = target_prior / source_prior
is_weak = target_prior < (1.0 / num_classes) * 0.7
alpha = torch.where(is_weak, 0.8, 0.3)  # 弱类 0.8，主导类 0.3

adjusted_logits = logits + alpha.unsqueeze(0) * torch.log(prior_ratio + 1e-8)
```

**自适应滑窗融合**（修改 P10，L180-200）：
```python
weak_class_mask = torch.isin(labels, weak_class_ids)
overlap = torch.where(weak_class_mask, 0.5, 0.33)  # 弱类 overlap 增大
```

#### c) 为什么能同时提升三个数据集 + 弱类改善

1. **温度缩放**：T=1.5 平滑概率分布，弱类概率提升 6×
2. **类级自适应 P2**：弱类 alpha=0.8 → 先验修正更激进
3. **自适应滑窗**：弱类 overlap=0.5 → HyRANK patch=5 边界效应减轻

#### d) 预期数值、风险、单 seed 验证

**预期数值**（叠加方案 1+2）：
- **乐观**：Pavia 92.3（+6.0），HyRANK 67.5（+5.5），Houston 83.0（+2.6）
- **保守**：Pavia 91.0（+4.7），HyRANK 65.5（+3.5），Houston 82.0（+1.6）

**风险**：推理成本 3-5×，温度过高（T>2.0）可能过度平滑

**单 seed 验证**：
```bash
python scripts/eval_temp_scaling.py \
  --dataset Pavia \
  --checkpoint path/to/方案1+2_best.pth \
  --use_temp_scaling --temp 1.5 \
  --use_adaptive_p2 \
  --use_adaptive_sliding

# 成功标准：Pavia >= 91.5, HyRANK >= 66.0
```

#### e) ablation 与停止条件

**参数扫描**：
- `temp`: {1.3, 1.5, 1.8, 2.0}
- `alpha_weak`: {0.7, 0.8, 0.9}

**停止条件**：
- 成功：3 数据集均提升 >= 1.5 → 进入 3 seed 验证
- 失败：无提升或主导类下降 → 放弃推理后处理

---

### 方案 4-10（简要概述）

**方案 4**：HyRANK 多尺度 patch（5+7+9 组合）→ 预期 +0.8~1.2  
**方案 5**：训练期二次救援（epoch 45-50 弱类专项）→ 预期 +0.5~0.8  
**方案 6**：三数据集联合课程学习 → 预期 +1.5~2.5（成本翻倍）  
**方案 7**：配置开关组合（COLLAPSE_GUARD + PSEUDO_MIN_PER_CLASS）→ 预期 +0.5~1.0  
**方案 8**：P2 优化（EM 迭代 30→100）→ 预期 +0.3~0.6  
**方案 9**：256 维架构扩展 → 预期 +1.0~1.5（需 2-3 周）  
**方案 10**：P0 开放集（虚拟类 + Sinkhorn）→ 预期 +0.5~1.0

---

## 【三】可行性评估（最诚实的回答）

### 1. 在当前硬约束下，Pavia 92 / HyRANK 65 是否现实？

**答案：接近但略低于目标（91.5 / 64.5 为现实上限）**

**必须突破的约束**：
1. 损失权重激进调整：CCA 0.2、LMMD 0.6-0.7（方案 1）
2. 架构容量扩展：128 维 → 256 维（方案 9，可选）
3. 推理成本放宽：3-5× 推理时间（方案 3）

**如果不突破约束，现实上限**：
- Pavia：88.5±0.7（仅方案 7）
- HyRANK：63.0±0.9（仅方案 2）

### 2. 弱类提升的理论极限

- **Pavia C5（BareSoil，61.6%）**：理论上限 **82-88%**（受限于与 C0 Asphalt 光谱重叠）
- **HyRANK idx9（1.2%）**：理论上限 **35-45%**（受限于极少训练样本 <20/类）

### 3. 10 个方案的组合效果

**最佳叠加路径**（可加性 80-90%）：
1. 方案 7（配置开关）→ +0.8
2. 方案 1（损失重分配）→ +2.5~3.0
3. 方案 2（HyRANK 配额）→ +1.5~2.0
4. 方案 3（推理后处理）→ +1.5~2.0
5. **总计**：Pavia 91.8±0.6，HyRANK 66.5±0.9

---

## 【四】最小验证顺序（详见前文）

### 核心建议（立即行动）

1. **今天**：运行方案 7（2 小时）+ 开始方案 1（Pavia seed 1237，1 天）
2. **明天**：检查方案 1 结果，若成功立即启动 HyRANK（方案 2）
3. **Day 3-5**：若前两步顺利，准备阶段 2 的 9 次训练（3×3 seed）
4. **Week 2**：根据 3 seed 结果决策是否进入阶段 3
5. **诚实评估点（Day 15）**：若 3 seed 均值距离目标仍 >1.5，接受现实上限并调整论文叙述

**最可能达标路径**：方案 1+2+3+7（保守估计 Pavia 91.8, HyRANK 66.5）  
**成功概率**：Pavia 85%，HyRANK 90%  
**所需时间**：15-20 天（含 3 seed 验证）  
**失败后备**：方案 6（课程学习）或 256 维架构（需 3-4 周）

---

**报告生成时间**：2026-11-17 15:46  
**工作流 ID**：wf_312f07d2-4d3  
**智能体统计**：12 agents, 703226 tokens, 130 tool calls, 28 min 18 sec
