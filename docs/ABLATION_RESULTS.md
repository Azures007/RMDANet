# RMDANet 超级最终版 — 三数据集模块消融（双向可靠性融合）

协议：v3.0（每类 180 源标签、无源验证、50 epochs × 40 steps、batch 128、patch 7、
AdamW+cosine、AMP、逐像元 L2、最后一轮评估、目标 GT 只在最终评估打开一次）
seed = 1237、data_seed = 1237、单卡 RTX 4090、torch 2.8.0+cu128。

## 最终模型结构

```
输入 patch
 ├─ LocalSpectralSpatialEncoder → semantic2d → ScaleSelectiveMicroASPP(adaptive)
 ├─ DRSE = 双分支可靠性光谱-空间编码器（空间分支含 RCGP 残差中心引导池化；融合子模块 RSF）
 │    · pooled = gap + α·(guided − gap)，α = sigmoid(可学习 logit)
 │    · F = Norm((1−w)·F_ctx + w·F_spec)，w = 0.5 + a·(base−0.5)
 │    · base = σ(γ(r_s−r_c)/T_w + δ)，a = σ((cos(F_ctx,F_spec)−τ_a)/T_a)
 ├─ RFTM = PRAT（相位可靠性残差低频振幅迁移）
 └─ RTML = 跨 epoch 统一可靠性记忆（history 伪标签 + 可靠性门控）
```

超参（均在 `datasets/*/config.yaml` 的 `uda` 段）：

| 参数 | Houston | Pavia | HyRANK |
|---|---|---|---|
| `pool_alpha_logit`（α = σ(logit)） | −4.0 → 0.018 | −4.0 → 0.018 | −3.5 → 0.029 |
| `pool_center_size` | 5 | 5 | 5 |
| `pool_temperature` | 0.30 | 0.30 | 0.30 |
| `pool_gate_init` | 0.6,0.3,0.1 | 0.6,0.3,0.1 | 0.6,0.3,0.1 |
| 融合模式 | bidirectional | bidirectional | bidirectional（保守参数） |
| `prat_style_max` | 0.64 | 0.2 | 0.4 |

> 术语：**DRSE** = 双分支可靠性光谱-空间编码器（`DualReliabilityEncoder`）；
> **RCGP** = 残差中心引导池化（DRSE 空间分支子模块）；**RSF** = 可靠性光谱融合（DRSE 融合子模块）。
> 下表中 `+DRSE` 行 = RCGP + RSF 同时打开；真正的 DRSE-off（单分支）行由
> `run_drseoff_row.sh` 补跑，结果待填入。

## 结果（seed 1237）

### Houston13 → Houston18（双向融合）

| Method | RFTM | RCGP+RSF（DRSE 子模块） | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 74.77 | 77.35 | 0.6208 |
| +DRSE | ✗ | ✓ | ✗ | 76.12 | 78.97 | 0.6392 |
| +DRSE+RFTM | ✓ | ✓ | ✗ | 78.48 | 83.01 | 0.6698 |
| Full | ✓ | ✓ | ✓ | 79.06 | 82.56 | 0.6786 |

### PaviaU → PaviaC（双向融合）

| Method | RFTM | DRSE | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 65.62 | 66.56 | 0.5952 |
| +DRSE | ✗ | ✓ | ✗ | 67.92 | 70.51 | 0.6229 |
| +DRSE+RFTM | ✓ | ✓ | ✗ | 83.45 | 81.69 | 0.8013 |
| Full | ✓ | ✓ | ✓ | 85.73 | 83.38 | 0.8283 |

### HyRANK Dioni → Loukia

双向融合在 HyRANK 上需要"保守门控"（光谱分支在 176 波段跨传感器场景下易过度自信）：
`--fusion_delta` 更负、`--fusion_w_max` 更低。参数扫描（gA–gD）进行中，最终配置以扫描结果为准。

参考（融合 off、α=−3.5，已证单调）：

| Method | OA | AA | Kappa |
|---|---|---|---|
| Baseline | 55.05 | 62.32 | 0.4732 |
| +DRSE | 55.83 | 63.42 | 0.4832 |
| +DRSE+RFTM | 56.51 | 61.12 | 0.4897 |
| Full | 58.20 | 60.36 | 0.5029 |

## 模块贡献（OA 增量，seed 1237）

| 模块 | Houston | Pavia | HyRANK |
|---|---|---|---|
| DRSE（RCGP + RSF） | +1.35 | +2.30 | 待定（融合 off +0.78） |
| RFTM | +2.36 | +15.53 | 待定（融合 off +0.68） |
| RTML | +0.58 | +2.28 | 待定（融合 off +1.69） |

## 已知问题（必须在论文中说明）

1. **RTML 在 Houston 上跨 seed 不稳定**：seed 1237 Full 79.06（+0.58），seed 1233 Full 77.01
   （比 +DRSE+RFTM 的 77.75 低 0.74）。因此 RTML 不宜作为论文主贡献，建议作为域相关分析。
2. **HyRANK 类 10 召回恒为 0**：源域 492 像素、目标域 487 像素，但 HyRANK 2018 两场景的类 10
   语义不完全对应（已知数据集问题）。
3. **融合参数为数据集级设置**：Houston/Pavia 用默认双向融合；HyRANK 需保守参数
   （`fusion_delta` 更负、`fusion_w_max` 更低），论文需在"实验设置"中说明。

## 复现

```bash
git clone -b clean-v7 --depth 1 https://github.com/Azures007/RMDANet.git rmda_final
cd rmda_final
bash run_final_ablation.sh 1237
```

> `drse_patch.py` / `eval_tuned.py` 仅用于兼容旧的运行时补丁 checkpoint，新训练与评估不需要。
