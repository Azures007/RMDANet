# PMDAnet 消融包（clean v7.0 + 残差 DRSE + 三数据集消融脚本）

本包 = `RMDANet_v7.0+干净版`（c0783a7，已移除 CNT/SSF）+ 本次新增的：
**残差式 DRSE**、**组合消融运行器**、**调参脚本**、**锁定评估脚本**、**结果文档**。

## 1. 三个创新模块与消融开关

| 模块 | 实现 | 消融方式 |
|---|---|---|
| **RFTM**（PRAT） | `core/prat.py` | `--abl-prat 1/0` |
| **DRSE**（残差式中心引导池化） | `core/model.py` + `drse_patch.py` | `--abl-pool guided/gap` |
| **RTML**（跨 epoch 统一可靠性记忆） | `core/memory.py` + trainer | `--abl-rtml 1/0` |

### 残差式 DRSE（本次改进，解决原 guided 池化在 Houston 上 −3.61 的问题）

```
pooled = gap + α · (guided − gap)
α = sigmoid(learnable logit)，初始 0.018
center_size = 5, attention_temperature = 0.30
gate 初始权重 = [0.6, 0.3, 0.1]（global / center / attended）
spectral fusion = off
```

由 `drse_patch.py` 在运行时注入（不改原 `core/model.py`），用环境变量控制：

```bash
export DRSE_TUNE=1 DRSE_RESIDUAL=1 DRSE_CENTER_SIZE=5 DRSE_TEMP=0.30 DRSE_INIT=0.6,0.3,0.1
```

评估带补丁的 checkpoint 时必须使用 `eval_tuned.py`（否则严格的 state_dict 加载会因 `pool._alpha_logit` 失败）。

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
export DRSE_TUNE=1 DRSE_RESIDUAL=1 DRSE_CENTER_SIZE=5 DRSE_TEMP=0.30 DRSE_INIT=0.6,0.3,0.1

python -u ablate_4rows.py --dataset houston --seed 1237 --gpu 0 --num_workers 4 \
  --abl-prat 1 --abl-pool guided --abl-fusion off --abl-rtml 0 --tag h_drse_rftm

RUN=results_ablation_clean/houston/h_drse_rftm_seed1237
python -u eval_tuned.py --checkpoint "$RUN/selected_model.pt" \
  --data-path "$HOUSTON_DATA" --output "$RUN/target_final_once.json" --confirm-final-eval
```

### 消融运行器参数

| 参数 | 含义 |
|---|---|
| `--dataset {houston,pavia,hyrank}` | 数据集（自动加载对应 config.yaml） |
| `--row {base,drse,rftm,drse_rftm,full}` | 预设行（DRSE 行固定 fusion=off） |
| `--abl-prat 0/1` | RFTM 开关 |
| `--abl-pool guided/gap` | DRSE 开关 |
| `--abl-fusion reliable/off` | 光谱可靠性融合开关（默认 off） |
| `--abl-rtml 0/1` | RTML 开关 |
| `--abl-guard / --abl-spatial / --abl-proto / --abl-confusion / --abl-history 0/1` | RTML 子组件的细粒度开关（仅 rtml=1 时生效） |
| `--tag NAME` | 自定义运行名（与上面 `--abl-*` 配合使用） |

## 3. 结果（seed 1237，定稿）

详见 `docs/ABLATION_RESULTS.md`。摘要：

| Dataset | Baseline | +DRSE | +DRSE+RFTM | Full |
|---|---|---|---|---|
| Houston | 74.77 | 75.21 | 76.35 | 77.26 |
| Pavia | 65.62 | 66.25 | 84.19 | 84.36 |
| HyRANK | 55.05 | 55.11 | 57.63 | 57.63（RTML 不启用，待定） |

## 4. 文件说明

```
ablate_4rows.py            组合消融运行器（预设行 + 任意 --abl-* 组合）
drse_patch.py             残差 DRSE 运行时补丁（环境变量控制）
eval_tuned.py             带 DRSE 补丁的锁定评估入口
run_final_ablation.sh     三数据集一键复现脚本
run_ablation_clean.sh     单数据集预设 4 行脚本
docs/ABLATION_RESULTS.md  结果与模块机制说明
core/, datasets/, scripts/ 原始 clean v7.0 代码（未改动）
```

## 5. 注意

- Pavia 必须使用 `Test_1/.../Datasets/Pavia`（Tiles=9248 的标准版），否则 OA 会掉到 10–13%；
- 训练命令不要接 `head`/`grep` 管道（会 SIGPIPE 提前终止训练）；
- 目标 GT 只在 `final_eval_locked.py` 中打开一次，训练过程严格无标签；
- 单 seed 波动 ±1~2 分，论文建议 3 seed 取 mean±std。
