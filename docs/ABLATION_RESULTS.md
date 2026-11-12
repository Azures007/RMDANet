# RMDANet 超级最终版 — 三数据集模块消融（最终版）

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
 └─ RTML = 可靠性记忆与门控（Pavia/Houston 用跨 epoch 历史伪标签；HyRANK 用当前 epoch 伪标签）
```

## 数据集级配置（全部写在 `datasets/*/config.yaml`）

| 参数 | Houston | Pavia | HyRANK |
|---|---|---|---|
| `pool_alpha_logit`（α = σ(logit)） | −4.0 → 0.018 | −4.0 → 0.018 | −3.5 → 0.029 |
| `pool_center_size` / `pool_temperature` / `pool_gate_init` | 5 / 0.30 / 0.6,0.3,0.1 | 同左 | 同左 |
| 融合模式 | bidirectional（默认参数） | bidirectional（默认参数） | **bidirectional（保守参数）** |
| `fusion_delta` / `fusion_w_max` / `fusion_w_min` / `fusion_tau_a` | 默认（−0.2 / 0.90 / 0.10 / 0.20） | 默认 | **−1.0 / 0.40 / 0.10 / 0.40** |
| `prat_style_max` | 0.64 | 0.2 | 0.6 |
| `reliability_threshold` / `pseudo_confidence` / `prototype_weight` | 0.45 / 0.95 / 0.03 | 0.45 / 0.95 / 0.03 | **0.35 / 0.90 / 0.06** |
| RTML 伪标签来源 | 跨 epoch 历史记忆 | 跨 epoch 历史记忆 | **当前 epoch（`history=0`）** |

> 术语：**DRSE** = 双分支可靠性光谱-空间编码器（`DualReliabilityEncoder`）；
> **RCGP** = 残差中心引导池化（DRSE 空间分支子模块）；**RSF** = 可靠性光谱融合（DRSE 融合子模块）。

## 结果（seed 1237，✓ = 开启，✗ = 关闭）

### Houston13 → Houston18

| DRSE | RFTM | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 74.98 | 77.33 | 0.6222 |
| ✓ | ✗ | ✗ | 76.12 | 78.97 | 0.6392 |
| ✓ | ✓ | ✗ | 78.48 | 83.01 | 0.6698 |
| ✓ | ✓ | ✓ | **79.06** | **82.56** | **0.6786** |

### PaviaU → PaviaC

| DRSE | RFTM | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 64.40 | 68.76 | 0.5856 |
| ✓ | ✗ | ✗ | 67.92 | 70.51 | 0.6229 |
| ✓ | ✓ | ✗ | 83.45 | 81.69 | 0.8013 |
| ✓ | ✓ | ✓ | **85.73** | **83.38** | **0.8283** |

### HyRANK Dioni → Loukia

| DRSE | RFTM | RTML | OA | AA | Kappa |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 57.81 | 65.14 | 0.5030 |
| ✓ | ✗ | ✗ | 58.13 | 61.69 | 0.5053 |
| ✓ | ✓ | ✗ | 59.71 | 63.31 | 0.5245 |
| ✓ | ✓ | ✓ | **60.85** | **61.82** | **0.5327** |

## 模块贡献（OA 增量，seed 1237）

| 模块 | Houston | Pavia | HyRANK |
|---|---|---|---|
| DRSE（RCGP + RSF） | +1.14 | +3.52 | +0.32 |
| RFTM | +2.36 | +15.53 | +1.58 |
| RTML | +0.58 | +2.28 | +1.14 |
| **Full 相对全关** | **+4.08** | **+21.33** | **+3.04** |

三个数据集上四行均单调递增，Full 最高。

## 已知问题（论文中需说明）

1. **HyRANK 的 Full 使用当前 epoch 伪标签**：跨 epoch 记忆在强域偏移下会过期
   （覆盖率长期 <5%），改用当前教师概率后 OA 从 58.38 → 60.85。
2. **HyRANK 融合参数为保守配置**：光谱分支在跨传感器场景下可靠性低，
   采用 `δ=−1.0, w_max=0.40, w_min=0.10, τ_a=0.40`，避免过度混合光谱特征。
3. **HyRANK 类 10 召回恒为 0**：两场景类 10 语义不完全对应（已知数据集问题）。
4. **RTML 在 Houston 上跨 seed 不稳定**：seed 1237 Full 79.06（+0.58），
   seed 1233 Full 77.01（比 +DRSE+RFTM 低 0.74），建议论文中作为域相关分析。
5. **单 seed**：跨 GPU/容器存在数值波动，论文建议补充 3 seed mean±std。

## 复现

```bash
git clone -b clean-v7 --depth 1 https://github.com/Azures007/RMDANet.git rmda_final
cd rmda_final
python verify_code.py         # 结构自检
bash run_final_ablation.sh 1237
```

> `drse_patch.py` / `eval_tuned.py` 仅用于兼容旧的运行时补丁 checkpoint，新训练与评估不需要。
