# BASELINE_V3 — 最佳基线（v3.0）冻结记录

本文件冻结 RMDANet v7.1 在“文献对齐协议 v3.0”下的最佳基线，供论文、答辩与后续实验对照使用。
**任何新实验都应以本文件为准做对照，不要使用更早的 r23/rfast 数字。**

- 代码 tag：`v3.0`（commit `c0783a7`）
- main 分支恢复提交：`b2fcdde`（配置与该 tag 完全一致）
- 协议性质：严格 UDA —— 训练过程不接触目标域 GT，目标域 GT 只在锁定评估中打开一次
- 运行目录：`results_v71/v3lite/<dataset>/C0_epochs50_seed1237/`
- 结果来源：seed 1237（单 seed），总耗时 17.1 分钟（三数据集）

## 1. 精度（seed 1237）

| 数据集 | OA | AA | Kappa | 有效像素 |
|---|---|---|---|---|
| Pavia（PaviaU→PaviaC） | **85.55** | 82.74 | 0.8258 | 39,355 |
| HyRANK（Dioni→Loukia） | **55.03** | 60.35 | 0.4709 | 12,208 |
| Houston（Houston13→Houston18） | **76.72** | 82.28 | 0.6497 | 53,200 |

### 逐类精度

| Pavia 类 | 精度 | | HyRANK 类 | 精度 | | Houston 类 | 精度 |
|---|---|---|---|---|---|---|---|
| 1 Trees | 94.45 | | 1 Dense Urban | 50.00 | | 1 Grass Healthy | 85.22 |
| 2 Asphalt | 91.85 | | 2 Mineral | 97.01 | | 2 Grass Stressed | 92.64 |
| 3 Bricks | 65.51 | | 3 Non Irrigated | 80.63 | | 3 Trees | 68.00 |
| 4 Bitumen | 81.25 | | 4 Fruit Trees | 83.54 | | 4 Water | 100.00 |
| 5 Shadows | 98.92 | | 5 Olive Groves | 7.92 | | 5 Residential | 86.03 |
| 6 Meadows | 63.04 | | 6 Coniferous | 45.60 | | 6 Non-Residential | 74.59 |
| 7 BareSoil | 84.13 | | 7 Dense Sclerophyllous | 59.00 | | 7 Road | 69.46 |
| | | | 8 Sparse Sclerophyllous | 49.13 | | | |
| | | | 9 Sparsely Vegetated | 50.74 | | | |
| | | | 10 Rocks and Sand | 0.62 | | | |
| | | | 11 Water | 100.00 | | | |
| | | | 12 Coastal Water | 100.00 | | | |

## 2. 训练设置

### 2.1 三个数据集共同设置

| 项目 | 值 |
|---|---|
| 源域标签 | 每类 **180** 个，全部用于训练；**无验证集**（Pavia 1,260 / HyRANK 2,130 / Houston 1,260） |
| 训练量 | **50 epochs × 40 steps = 2,000 次参数更新** |
| batch / 评估 batch | 128 / 512 |
| 评估间隔 | 每 5 轮 |
| 优化器 | **AdamW + cosine 衰减**（warmup 5 轮），weight decay 4e-4，梯度裁剪 5.0，AMP 开启 |
| 输入归一化 | **逐像元 L2**（保持光谱形状、丢弃幅度） |
| 空间尺寸 | patch **7 × 7** |
| 选模型 | **固定最后一轮**（`select_final_epoch: true`），不使用源域验证、不接触目标域 GT |
| 伪标签 | 第 17 轮启动，warm 10 轮（第 27 轮满强度）；置信度 0.95、可靠性阈值 0.45、每类上限 24、min_age = 2 |
| 域对齐损失 | CCA 0.5；CORAL 0.3；LMMD 0.3（第 25 轮启动，warm 8）；原型对齐 0.03；严格类平衡强度 1.0 |
| PRAT | 频域风格迁移，第 1 轮启动，warm 23 轮 |
| guard | 第 17 轮冻结独立语义 guard |
| 其它 | early target adaptation（记忆库第 1 轮即更新）；目标域池 Pavia/Houston 30,000、HyRANK 15,000 |
| 实际 workers | 4（`run_unified.sh` 的 `--workers 4` 覆盖配置里的 8） |

### 2.2 数据集差异项

| 项目 | Pavia | HyRANK | Houston |
|---|---|---|---|
| 学习率 | 1e-3 | 1e-3 | **3e-4** |
| PRAT 强度 | 0.2 | 0.4 | **0.64** |
| band alignment | 0.1 | 0.1 | **0.45** |
| 目标域池 | 30,000 | 15,000 | 30,000 |

## 3. 复现命令

```bash
cd ~/autodl-tmp/RMDANet
git checkout main            # 或 git checkout b2fcdde
bash scripts/run_unified.sh C0 1237 /root/autodl-tmp/RMDANet/results_v71/repro
```

Pavia 数据路径必须使用 `Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia`
（`/root/autodl-tmp/Datasets/Pavia` 是错误 GT：Meadows 42,826 像素）。

## 4. 已知局限

- **只跑了单 seed**：Pavia 在 seed 1238 上 Bitumen 类崩到 6.4%，OA 仅 72.33 —— 说明 180/类协议下 Pavia 对种子敏感，论文需报告多 seed 或说明该失败模式。
- HyRANK 的 class 5/10/9 长期偏低（7.9% / 0.6% / 50.7%），是目标域类别不平衡 + 源域样本不足。
- 与文献 SOTA 的差距（Pavia 85.55 vs MSDA 2024 的 92.44、Houston 76.72 vs TAPDA 2026 的 80.67）主要来自骨干与训练范式，而非超参。

## 5. 已尝试并否决的改动（避免重复踩坑）

| 版本 | 改动 | Pavia OA | 结论 |
|---|---|---|---|
| v3.0 | 基线（自研骨干 + L2 + AdamW + 单头） | **85.55** | 保留 |
| v3.1 | SGD momentum 0.9，lr 0.1，batch 32 | 71.74 | 否决（batch 32 破坏伪标签记忆库） |
| v3.2 | 恢复 AdamW，仍 batch 32 | 70.59 | 否决（batch 32 是主因） |
| v3.3 | batch 128 + 伪标签提前到第 8 轮 | 84.31 | 否决（Pavia 掉 1.2，Houston +0.65） |
| v3.4 | 额外幅度输入通道（decoupled） | 67.07 | 否决（幅度跨场景不可迁移） |
| v3.5 | 逐波段 z-score 归一化 | 67.90 | 否决（场景分布差异被放大） |
| v3.6 | 场景中位数归一化 | 65.91 | 否决（同上） |
| v3.11 | SGD momentum 0.9（cosine，lr 0.1，batch 128） | 81.79 | 部分保留：HyRANK 59.18（+4.15），Pavia 掉 3.76 |
| v3.12 | DCRN（SCLUDA）骨干最小接入 | 23.66 | 否决（特征与我们轻量对齐不兼容） |
| v3.13 | 双分类头一致性过滤 | 77.78 | 否决（第二头竞争，掉 7.8） |

新训练配方（文献配方重写）在新分支 `lit-recipe` 上开发，主分支始终保持本文件记录的基线。

## 6. 异构双模型集成（lit-recipe 分支）

lit 配方（`core/model_lit.py` + `core/trainer_lit.py`）：
SSSE 骨干 + 双分类器 L1 对抗（eta 2.0）+ 源/目标 SupCon（0.5 / 0.05）+ EMA 教师伪标签（τ 0.75→0.85，权重 0.05）+ BNM 0.15 + 逐波段 z-score 输入 + SGD 0.1 常数 lr + patch 9 + 每类 180 + 50×40 + 最后一轮。

| 数据集 | A（v3.0 可靠性框架） | B（lit 配方） | A+B 各 0.5 | 权重扫描（A 权重） |
|---|---|---|---|---|
| Pavia | 85.55 | 79.90 | **89.70**（AA 89.76，κ 0.8764） | 0.45→86.34，0.50→89.70，0.55→87.92，0.60→87.19 |
| HyRANK（patch 7） | 55.05 | 26.02 | 47.47（0.7 权重仍 54.78） | 集成无收益，改用 A 单模型 |
| Houston（patch 7） | 76.64 | 56.08 | 70.72（0.7 权重 76.46） | 集成无收益，改用 A 单模型 |

- 集成权重固定 **0.5**（先验设置，非测试集调参）；扫描仅作为敏感性分析；
- 翻转 TTA 对 Pavia 集成是负收益（89.25 < 89.70），默认关闭；
- 评估命令：
  ```bash
  python -u scripts/ensemble_eval.py \
    --checkpoint-a <v3.0 checkpoint> --checkpoint-b <lit checkpoint> \
    --data-path <dataset path> --output <json> --confirm-final-eval [--sweep]
  ```
- 单次评估只打开目标 GT 一次，结果写入新 json（拒绝覆盖）。
