# PMDAnet 消融包（clean v7.0 + DRSE 双分支编码器 + 三数据集消融脚本）

本包 = `RMDANet_v7.0+干净版`（c0783a7，已移除 CNT/SSF）+ 本次新增的：
**DRSE 双分支可靠性光谱-空间编码器（含 RCGP 残差中心引导池化）**、**组合消融运行器**、**调参脚本**、**锁定评估脚本**、**结果文档**。

## 0. 术语（重要）

| 名称 | 含义 | 代码 |
|---|---|---|
| **DRSE** | 双分支可靠性光谱-空间编码器（整个模块：空间分支 + 光谱分支 + 可靠性融合） | `core/model.py::DualReliabilityEncoder` |
| **RCGP** | 残差中心引导池化（DRSE 空间分支的子模块） | `core/model.py::CenterGuidedPooling` |
| **RSF** | 可靠性光谱融合（DRSE 的融合子模块） | `core/model.py::SpectralReliabilityFusion` |

> DRSE ≠ 残差中心引导池化：RCGP 只是 DRSE 内部的一个池化层。
> `--abl-drse 0` 关闭 **整个 DRSE**；`--abl-pool gap` 只关闭 **RCGP**。

## 1. 模块与消融开关

| 模块 | 实现 | 消融方式 |
|---|---|---|
| **RFTM**（PRAT） | `core/prat.py` | `--abl-prat 1/0` |
| **DRSE**（双分支编码器，整体） | `core/model.py::DualReliabilityEncoder` | `--abl-drse 1/0` |
| **RCGP**（DRSE 子模块） | `core/model.py::CenterGuidedPooling` | `--abl-pool guided/gap` |
| **RSF**（DRSE 子模块） | `core/model.py::SpectralReliabilityFusion` | `--abl-fusion reliable/off/bidirectional` |
| **RTML**（跨 epoch 统一可靠性记忆） | `core/memory.py` + trainer | `--abl-rtml 1/0` |

### RCGP（残差中心引导池化，DRSE 空间分支子模块，已合入 `core/model.py`）

```python
# core/model.py::CenterGuidedPooling
self._alpha_logit = nn.Parameter(torch.tensor(alpha_logit))   # residual=True 时创建
...
if self.residual and self.mode == "guided":
    alpha = torch.sigmoid(self._alpha_logit)
    pooled = global_descriptor + alpha * (pooled - global_descriptor)
```

```
pooled = gap + α · (guided − gap)
α = sigmoid(可学习 logit)，初始值由配置 pool_alpha_logit 给定
Houston/Pavia: pool_alpha_logit=-4.0  → α_init = 0.018
HyRANK:        pool_alpha_logit=-3.5  → α_init = 0.029
center_size=5, attention_temperature=0.30, gate 初始权重=[0.6,0.3,0.1]（global/center/attended）
```

超参在 `datasets/*/config.yaml` 的 `uda` 段配置（`pool_center_size / pool_temperature /
pool_gate_init / pool_residual / pool_alpha_logit`），也可用 CLI 覆盖：

```bash
--pool_center_size 5 --pool_temperature 0.30 --pool_gate_init 0.6,0.3,0.1 \
--pool-residual --pool_alpha_logit -4.0
```

> `drse_patch.py` 保留仅用于兼容旧 checkpoint（旧的运行时补丁版）；新训练/评估都不需要它，
> checkpoint 用 `scripts/final_eval_locked.py` 直接加载即可。

## 2. 快速开始

```bash
# 数据路径（AutoDL 服务器）
PAVIA_DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia
HOUSTON_DATA=/root/autodl-tmp/Datasets/Houston
HYRANK_DATA=/root/autodl-tmp/Datasets/HyRANK

# 一键跑三数据集全部消融行 + 评估 + 汇总
bash run_final_ablation.sh 1237
```

单行运行示例（Houston 的 +DRSE+RFTM）：

```bash
cd /root/autodl-tmp/v7clean
python -u ablate_4rows.py --dataset houston --seed 1237 --gpu 0 --num_workers 4 \
  --abl-prat 1 --abl-pool guided --abl-fusion off --abl-rtml 0 --tag h_drse_rftm

RUN=results_ablation_clean/houston/h_drse_rftm_seed1237
python -u scripts/final_eval_locked.py --checkpoint "$RUN/selected_model.pt" \
  --data-path /root/autodl-tmp/Datasets/Houston \
  --output "$RUN/target_final_once.json" --confirm-final-eval
```

> 评估不需要任何环境变量或 `eval_tuned.py`——残差 α 已经在模型定义里。

### 消融运行器参数

| 参数 | 含义 |
|---|---|
| `--dataset {houston,pavia,hyrank}` | 数据集（自动加载对应 config.yaml） |
| `--row {base,drseoff,drse,rftm,drse_rftm,full}` | 预设行（`drseoff` = DRSE 整体关闭；`base` 保留 DRSE 但 RCGP/RSF 关闭） |
| `--abl-drse 1/0` | **DRSE 整体开关**（0 = 去掉光谱分支 + 融合，退化为单分支） |
| `--abl-prat 0/1` | RFTM 开关 |
| `--abl-pool guided/gap` | **RCGP 开关**（DRSE 空间分支子模块） |
| `--abl-fusion reliable/off/bidirectional` | **RSF 开关**（DRSE 融合子模块；默认 off；bidirectional = 双向可靠性融合） |
| `--abl-rtml 0/1` | RTML 开关 |
| `--abl-guard / --abl-spatial / --abl-proto / --abl-confusion / --abl-history 0/1` | RTML 子组件的细粒度开关（仅 rtml=1 时生效） |
| `--tag NAME` | 自定义运行名（与上面 `--abl-*` 配合使用） |

### 补跑"真正关闭 DRSE"的一行

```bash
# 三数据集各跑一行 DRSE-off（已有 DRSE 开的结果保持不变）
bash run_drseoff_row.sh 1237
```

### 验证残差公式确实在源码里（而不是补丁）

```bash
# 1) 源码中应能看到 residual 与 _alpha_logit
grep -n "_alpha_logit\|residual and self.mode" core/model.py

# 2) 数值验证：pooled == gap + alpha*(guided-gap)
python tests/test_drse_residual.py
```

## 3. 结果（seed 1237，定稿）

详见 `docs/ABLATION_RESULTS.md`。摘要（seed 1237，双向可靠性融合）：

| Dataset | DRSE-off | +DRSE（RCGP+RSF） | +DRSE+RFTM | Full |
|---|---|---|---|---|
| Houston（α=−4） | 待补跑 | 76.12 | 78.48 | 79.06 |
| Pavia（α=−4） | 待补跑 | 67.92 | 83.45 | 85.73 |
| HyRANK（α=−3.5） | 待补跑 | 55.83 | 56.51 | 58.20 |

> 注意：过去的 `Baseline` 行（74.77/65.62/55.05）是"保留双分支但关 RCGP+RSF"，
> 并非真正的 DRSE-off；真正的 DRSE-off 由 `run_drseoff_row.sh` 补跑。
> `+DRSE` 行 = RCGP + RSF 同时打开。

模块贡献（OA 增量，seed 1237）：

| 模块 | Houston | Pavia | HyRANK |
|---|---|---|---|
| DRSE（RCGP + RSF） | +1.35（相对旧 base） | +2.30（相对旧 base） | 待定 |
| RFTM | +2.36 | +15.53 | 待定 |
| RTML | +0.58（seed 1233 为 −0.74，不稳定） | +2.28 | 待定 |

RCGP 残差强度 α 为数据集级超参，写在各自 `datasets/*/config.yaml` 的 `uda.pool_alpha_logit`：
Houston/Pavia 用 `-4.0`（α_init=0.018），HyRANK 用 `-3.5`（α_init=0.029）。
HyRANK 的双向融合使用保守参数（`--fusion_delta` 更负、`--fusion_w_max` 更低），
可通过环境变量 `HYRANK_FUSION_CFG` 覆盖。

## 4. 文件说明

```
core/model.py               DualReliabilityEncoder = DRSE（双分支）
                            CenterGuidedPooling = RCGP（DRSE 空间分支子模块，含残差 α）
                            SpectralReliabilityFusion = RSF（DRSE 融合子模块）
                            use_spectral_branch=False 时整体关闭 DRSE（单分支）
core/trainer.py             pool_*/fusion_*/use_spectral_branch 参数解析与模型构建
datasets/*/config.yaml      uda 段配置 pool_center_size/pool_temperature/.../pool_alpha_logit
ablate_4rows.py             组合消融运行器（预设行 + 任意 --abl-* 组合）
tests/test_drse_residual.py RCGP 残差公式数值验证脚本
drse_patch.py               旧运行时补丁（仅用于兼容旧 checkpoint，新训练不需要）
eval_tuned.py               旧 checkpoint 评估入口（DRSE_LEGACY=1 时启用补丁）
run_final_ablation.sh       三数据集一键复现脚本（旧 base 行 + +DRSE 行）
run_drseoff_row.sh          补跑真正的 DRSE-off 行（三数据集）
run_ablation_clean.sh       单数据集预设 4 行脚本
docs/ABLATION_RESULTS.md    结果与模块机制说明
core/, datasets/, scripts/  其余为 clean v7.0 原始代码
```

α 初值在 config 中配置：Houston/Pavia `pool_alpha_logit: -4.0`（0.018），
HyRANK `pool_alpha_logit: -3.5`（0.029）。

## 5. 注意

- Pavia 必须使用 `Test_1/.../Datasets/Pavia`（Tiles=9248 的标准版），否则 OA 会掉到 10–13%；
- 训练命令不要接 `head`/`grep` 管道（会 SIGPIPE 提前终止训练）；
- 目标 GT 只在 `final_eval_locked.py` 中打开一次，训练过程严格无标签；
- 单 seed 波动 ±1~2 分，论文建议 3 seed 取 mean±std。
