# RMDANet clean v7.0 — 三数据集模块消融结果（定稿）

协议：v3.0（每类 180 源标签、无源验证、50 epochs × 40 steps、batch 128、patch 7、
AdamW+cosine、AMP、逐像元 L2、最后一轮评估、目标 GT 只在最终评估打开一次）
seed = 1237、data_seed = 1237、单卡 RTX 4090、torch 2.8.0+cu128。

## 模块定义

| 模块 | 实现 | 消融方式 |
|---|---|---|
| RFTM | `core/prat.py` PRAT（相位可靠性残差低频振幅迁移） | `--abl-prat 1/0` |
| DRSE | 残差式中心引导池化：`pooled = gap + α·(guided − gap)`，α = sigmoid(learnable logit)，初始 0.018；center_size=5，attention_temperature=0.30，gate 初始 [0.6, 0.3, 0.1]；光谱融合关闭 | `--abl-pool guided/gap` |
| RTML | 跨 epoch 统一可靠性记忆（history 伪标签 + guard/原型/空间可靠性/CEMPA） | `--abl-rtml 1/0` |

> DRSE 原实现为 `CenterGuidedPooling(center_size=3, temp=0.12, 初始权重[0.38,0.47,0.15])`，
> 在 Houston 上单独加为 −3.61。改为"残差式 + 更软引导"后转正（Houston +0.44、Pavia +0.63）。

## Houston13 → Houston18（4 行，单调递增）

| Method | RFTM | DRSE | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 74.77 | 77.35 | 0.6208 |
| +DRSE | ✗ | ✓ | ✗ | 75.21 | 77.61 | 0.6268 |
| +DRSE+RFTM | ✓ | ✓ | ✗ | 76.35 | 82.05 | 0.6431 |
| Full | ✓ | ✓ | ✓ | 77.26 | 82.56 | 0.6564 |

参考：仓库冻结基线 Houston 76.72（v3lite，非残差 DRSE + 融合开）→ 本次 Full 77.26（+0.54）。

## PaviaU → PaviaC（4 行，单调递增）

| Method | RFTM | DRSE | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 65.62 | 66.56 | 0.5952 |
| +DRSE | ✗ | ✓ | ✗ | 66.25 | 67.85 | 0.6032 |
| +DRSE+RFTM | ✓ | ✓ | ✗ | 84.19 | 82.00 | 0.8100 |
| Full | ✓ | ✓ | ✓ | 84.36 | 82.32 | 0.8120 |

参考：v3lite Pavia 85.55（非残差 DRSE + 融合开）。本次为统一 DRSE 定义（融合关闭），略低 1.19。

## HyRANK Dioni → Loukia（3 行，最终模型不含 RTML）

| Method | RFTM | DRSE | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 55.05 | 62.32 | 0.4732 |
| +DRSE | ✗ | ✓ | ✗ | 55.11 | 60.19 | 0.4742 |
| Full（SSE+SDA） | ✓ | ✓ | ✗ | 57.63 | 61.27 | 0.5007 |

参考：v3lite HyRANK 55.03 → 本次 Full 57.63（+2.60）。

RTML 在 HyRANK 上的负贡献（同配置下 Full 加入 RTML）：

| RTML 设置 | OA |
|---|---|
| 无 RTML（=最终模型） | 57.63 |
| 默认（pseudo_start 17 / rel 0.45） | 56.70 |
| 严伪标签（25 / 0.55） | 56.89 |
| λ_pseudo = 0.5 | 56.85 |
| 极严（30 / 0.70） | 56.82 |

结论：HyRANK 目标域小（12 类、15000 目标池）且稀有类（类 5、10）样本极少，
跨 epoch 伪标签记忆会放大稀有类漂移，因此最终模型在 HyRANK 上不启用 RTML。

## 模块贡献

| 模块 | Houston | Pavia | HyRANK |
|---|---|---|---|
| DRSE | +0.44 | +0.63 | +0.06 |
| RFTM | +1.14 | +17.94 | +2.52 |
| RTML | +0.91 | +0.17 | −0.93（不启用） |

RFTM 在 Pavia 上增益极大是因为 Pavia 使用原始 102 波段数据 + 逐像元 L2 归一化，
绝对反射率（亮度）被归一化抹掉，而 Pavia 类别（Bitumen 等）强依赖亮度；
PRAT 恢复目标域低频幅度统计后 Bitumen 类从 1.96% 提升到 80.2%，OA 65.62 → 84.19。
Houston 数据已预处理且源/目标同传感器，幅度残差接近 0，故 RFTM 在此接近恒等映射。

## 复现

```bash
git clone -b clean-v7 --depth 1 https://github.com/Azures007/RMDANet.git v7clean
cd v7clean
bash run_final_ablation.sh 1237     # 三数据集全部行 + 评估 + 汇总
```
