# PMDAnet 消融包（clean v7.0 + 残差 DRSE + 三数据集消融脚本）

本包 = `RMDANet_v7.0+干净版`（c0783a7，已移除 CNT/SSF）+ 本次新增的：
**残差式 DRSE**、**组合消融运行器**、**调参脚本**、**锁定评估脚本**、**结果文档**。

## 1. 三个创新模块与消融开关

| 模块 | 实现 | 消融方式 |
|---|---|---|
| **RFTM**（PRAT） | `core/prat.py` | `--abl-prat 1/0` |
| **DRSE**（残差式中心引导池化） | `core/model.py::CenterGuidedPooling`（已合入源码） | `--abl-pool guided/gap` |
| **RTML**（跨 epoch 统一可靠性记忆） | `core/memory.py` + trainer | `--abl-rtml 1/0` |

### 残差式 DRSE（已合入 `core/model.py`，不再是运行时补丁）

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
| `--row {base,drse,rftm,drse_rftm,full}` | 预设行（DRSE 行固定 fusion=off） |
| `--abl-prat 0/1` | RFTM 开关 |
| `--abl-pool guided/gap` | DRSE 开关 |
| `--abl-fusion reliable/off/bidirectional` | 光谱可靠性融合开关（默认 off；bidirectional = 双向可靠性融合） |
| `--abl-rtml 0/1` | RTML 开关 |
| `--abl-guard / --abl-spatial / --abl-proto / --abl-confusion / --abl-history 0/1` | RTML 子组件的细粒度开关（仅 rtml=1 时生效） |
| `--tag NAME` | 自定义运行名（与上面 `--abl-*` 配合使用） |

### 验证残差公式确实在源码里（而不是补丁）

```bash
# 1) 源码中应能看到 residual 与 _alpha_logit
grep -n "_alpha_logit\|residual and self.mode" core/model.py

# 2) 数值验证：pooled == gap + alpha*(guided-gap)
python tests/test_drse_residual.py
```

## 3. 结果（seed 1237，定稿）

详见 `docs/ABLATION_RESULTS.md`。摘要：

| Dataset | Baseline | +DRSE | +DRSE+RFTM | Full |
|---|---|---|---|---|
| Houston（α=−4） | 74.77 | 75.21 | 76.35 | 77.26 |
| Pavia（α=−4） | 65.62 | 66.25 | 84.19 | 84.36 |
| HyRANK（α=−3.5） | 55.05 | 55.83 | 56.51 | 58.20 |

模块贡献（OA 增量）：

| 模块 | Houston | Pavia | HyRANK |
|---|---|---|---|
| DRSE | +0.44 | +0.63 | +0.78 |
| RFTM | +1.14 | +17.94 | +0.68 |
| RTML | +0.91 | +0.17 | +1.69 |

DRSE 残差强度 α 为数据集级超参，写在各自 `datasets/*/config.yaml` 的 `uda.pool_alpha_logit`：
Houston/Pavia 用 `-4.0`（α_init=0.018），HyRANK 用 `-3.5`（α_init=0.029）。

## 4. 文件说明

```
core/model.py               CenterGuidedPooling 已含残差 α 公式（_alpha_logit）
                            SpectralReliabilityFusion 已含 bidirectional 模式
core/trainer.py             pool_*/fusion_* 参数解析与模型构建
datasets/*/config.yaml      uda 段配置 pool_center_size/pool_temperature/.../pool_alpha_logit
ablate_4rows.py             组合消融运行器（预设行 + 任意 --abl-* 组合）
tests/test_drse_residual.py 残差公式数值验证脚本
drse_patch.py               旧运行时补丁（仅用于兼容旧 checkpoint，新训练不需要）
eval_tuned.py               旧 checkpoint 评估入口（DRSE_LEGACY=1 时启用补丁）
run_final_ablation.sh       三数据集一键复现脚本
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
