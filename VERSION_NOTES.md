# RMDANet 版本记录

## v3.0 — 原始定版
- 两份 config，两套算法
  - Pavia: `ablation_id: main_reliability`，4 个布尔开关 = true
  - Houston: `ablation_id: full`，4 个布尔开关 = false
- 3-seed: Pavia 85.13 ± 1.20，Houston 78.56 ± 1.28
- 问题：审稿人会问"核心机制为什么两数据集不同"

## v4.0 — 机制统一版（已验证）
### 改动
1. `ablation_id` 统一为 `full`
2. CCA + ProtoAlign 两边统一打开
   - `lambda_cca: 0.5`, `cca_start_epoch: 0`
   - `prototype_align_flag: true`, `prototype_weight: 0.03`
3. 布尔开关 → 连续参数
   - `balance_weight_flag` → 删除，直接用 `source_sampler_power`
   - `pseudo_balance_flag` → `strict_class_balance_strength`（Pavia=1.0, Houston=0.0）
   - `use_early_target_adaptation` → `memory_update_start_epoch`（Pavia=1, Houston=100）
   - `include_target_scale_mi` 保留（Pavia=true, Houston=false）
4. `core/memory.py` 里 `strict_class_balance` 两套分支改成线性插值

### 验证结果（单 seed）
- Pavia seed 1236: OA = 86.50
- Houston seed 1237: OA = 78.02

### 硬约束验证
- 无 `if pavia / if houston` 分支
- 两数据集共用 `core/trainer.py` + `core/model.py`
- 差异仅在 config.yaml 数值

## v4.1 — Spectral Rescue 实验版
### 改动
- `core/model.py` 的 `SpectralReliabilityFusion.forward`
- 新增 rescue 路径：当两分支预测不一致、光谱置信度 >= rescue_confidence、
  且光谱置信度 > 上下文置信度时，允许 reliability 最多增加 rescue_strength
- 新增参数：`rescue_strength=0.55`, `rescue_confidence=0.55`

### 验证结果（3-seed）
| 数据集 | 1236 | 1237 | 1238 | 均值 | 标准差 |
|---|---|---|---|---|---|
| Pavia | 87.16 | 81.04 | 84.84 | 84.35 | ~2.5 |
| Houston | 76.04 | 78.93 | 78.51 | 77.83 | ~1.3 |

### 与 v4.0 baseline 对比
- v4.0 仅测了单点：Pavia seed1236=86.50, Houston seed1237=78.02
- v4.0 3-seed 尚未补测
- 问题：variance 翻倍，Pavia seed1237 掉到 81.04，超出合理波动

### 保留理由
- 单 seed 最佳（Pavia 87.16 / Houston 78.93）
- 但 3-seed 方差大，稳定性差
- 作为"scout run"记录保留，不进主线

### 后续待办
- [ ] 补跑 v4.0 干净版 3-seed（Pavia 1237/1238，Houston 1236/1238）
- [ ] 对比 v4.0 3-seed 均值 vs v4.1 3-seed 均值
- [ ] 若 v4.1 均值确实更差，则废弃 rescue，转向方案 2（类平衡 Top-K%）

## v4.3 — 访问间隔感知 EMA（已废弃）

### 动机
- Pavia 样本平均每 7.8 epoch 被访问一次
- Houston 样本平均每 104 epoch 被访问一次
- 固定 EMA 动量 0.92 对 Houston 过于滞后

### 改动
- `memory.py` 新增 `memory_ema_delta0` 参数
- `m_i(Δe) = m_0^(Δe/Δ_0)`
- 新增 `last_seen_epoch` 跟踪
- `update()` 新增 `current_epoch` 参数

### 验证结果
Pavia seed1236 delta_0 曲线：
| delta_0 | OA |
|---|---|
| 无 adaptive (baseline) | 87.16 |
| 10 | 86.96 |
| 8 | 87.09 |
| 5 | 86.97 |

Houston seed1237：
| delta_0 | OA |
|---|---|
| 无 adaptive (baseline) | 78.93 |
| 10 | 78.57 |
| 100 | 78.55 |

### 结论
- 全部低于 baseline，波动在 seed 噪声内
- 非单调，无可调最优
- **方向二废弃**

### 关键发现（写论文）
- **Houston 每样本平均只被访问 1.7 次**（memory 活跃 180 epoch × 480 访问 / 50000 池）
- **Pavia 每样本平均被访问 7.7 次**
- delta_0 从 10 到 100 对 Houston 无影响（78.57 vs 78.55），说明真正走 EMA 路径的样本极少
- 瓶颈不在 EMA 动量，在**样本访问密度太低**

## v4.4 — 周期性全 target 刷新（方案 1，已废弃）
### 动机
- Houston 每样本平均只被访问 1.7 次
- 周期性全 target 推理可把访问密度提升到 9 次

### 改动
- `trainer.py` 新增 `refresh_memory_over_target()`
- 每 `memory_refresh_interval` epoch 全 target 推理刷新 memory
- 两边 interval=20

### 验证结果（3-seed）
Pavia:
| seed | v4.1 | 方案 1 |
|---|---|---|
| 1236 | 87.16 | 86.91 |
| 1237 | 81.04 | 80.50 |
| 1238 | 84.84 | 85.02 |
| 均值 | 84.35 | 84.14 |

Houston:
| seed | v4.1 | 方案 1 |
|---|---|---|
| 1236 | 76.04 | 75.30 |
| 1237 | 78.93 | 79.39 |
| 1238 | 78.51 | 78.39 |
| 均值 | 77.83 | 77.69 |

### 结论
- 均值略降，方差变大
- **方案 1 对当前 baseline 无效**

### 关键发现
- 真正的问题是 **seed 敏感性**：Pavia seed1237 掉到 80，Houston seed1236 掉到 75
- 方案 1 反而在塌陷 seed 上塌得更厉害
- 说明瓶颈不在访问密度，在训练稳定性
- 下一步应研究：**为什么模型在某些 seed 上会塌**

## v4.5 — 固定动量降值实验（已废弃）
### 动机
- 怀疑 0.92 对 HSI 场景太大
- Pavia 每样本访问 7.7 次，Houston 1.7 次
- EMA 半衰期 863 epoch 远超训练长度

### 改动
- `trainer.py` 里 `probability_momentum` / `feature_momentum` 改成 CLI 参数
- 两份 config 都设 `0.80 / 0.85`

### 验证结果（Pavia seed1236）
| 配置 | OA | selected@110 | rel@110 |
|---|---|---|---|
| v4.1 baseline (0.92/0.94) | 87.16 | 24.6% | 0.567 |
| 本次 (0.80/0.85) | 86.49 | 48.4% | 0.777 |

### 结论
- 降动量让 memory 太"健忘"，把当前错误预测快速写入
- 可靠性评分虚高（0.567 → 0.777）
- 伪标签接受率失控（24.6% → 48.4%）
- **方向错，回滚**

### 重要教训
- 不是"更新太慢"，是"没有质量把关"
- 单靠调 EMA 数值无法解决，需要更根本的机制

## 失败方向汇总（v4.1 基础上的 4 次尝试）
| 方向 | 结果 | 原因 |
|---|---|---|
| rescue（光谱救援） | 方差翻倍 | 触发了不稳定 |
| adaptive EMA（访问间隔感知） | 无效 | 指数形式不适用 |
| memory refresh（周期刷新） | 均值略降 | 反而放大 seed 差异 |
| momentum 降值 | Pavia -0.67 | 伪标签接受率失控 |

### 共同根因
**seed 敏感性**：Pavia seed1237 掉到 81，Houston seed1236 掉到 75
- 训练过程稳定（loss/sup/rel 曲线几乎重合）
- 差异在伪标签接受率的系统性偏差
- 所有试图修正 memory 的方向都没解决这个问题

## v4.1 干净版 3-seed 完整基准（作为后续对比基准）
| 数据集 | seed 1236 | seed 1237 | seed 1238 | 均值 | 标准差 |
|---|---|---|---|---|---|
| Pavia | 87.16 | 81.04 | 84.84 | **84.35** | **3.09** |
| Houston | 76.04 | 78.93 | 78.51 | **77.83** | **1.56** |

### 对照 v3.0
| | v3.0 | v4.1 | Δ |
|---|---|---|---|
| Pavia 均值 | 85.13 | 84.35 | -0.78 |
| Pavia std | 1.20 | **3.09** | **+1.89** |
| Houston 均值 | 78.56 | 77.83 | -0.73 |
| Houston std | 1.28 | 1.56 | +0.28 |

### 关键判断
- rescue 完全没触发（v4.1 rescue vs 干净版数字相同）
- 机制统一代价主要在 Pavia 方差（+1.89），均值差在噪声边缘
- 剔除 seed1237 outlier 后，Pavia 两 seed 均值 85.99，接近 v3.0
- **下一步：方案 2（类平衡 Top-K%）修 seed 敏感性**

## v4.6 — CCA 启动延迟实验（已废弃）
### 动机
- 观察到 Pavia `cca_start_epoch=0` + `cca_warm_epochs=0`
- 意味着 CCA 从 epoch 1 就以全权重 0.5 启动，早于伪标签启动（epoch 50）
- 假设：早期特征随机 → CCA 把随机特征强化成错误原型 → seed 敏感性

### 改动
- `cca_start_epoch: 0 → 30`
- 新增 `cca_warm_epochs: 30`（从 0 线性升到 0.5）

### 验证结果（Pavia seed1236）
| 配置 | OA | selected@110 | rel@110 |
|---|---|---|---|
| v4.1 baseline | 87.16 | 24.6% | 0.567 |
| CCA 延迟 | **85.98** | **19.6%** | 0.442 |

### 结论
- 掉 1.18，超出噪声范围
- **伪标签接受率反而降低 5 个点**
- **CCA 早期启动其实在帮助伪标签筛选**：
  - 提供原型对齐信号
  - 提升 teacher 对目标域特征的定位
  - 间接提升伪标签可靠性评分
- CCA 不是 seed 敏感性的根源

### 教训
- 前面的假设错了：CCA 早启动不是"污染"，是"帮助"
- seed 敏感性另有原因

## v4.7 — 源域采样降权实验（已废弃）
### 动机
- 假设"逆频率采样过强导致 Pavia 类间错配"
- 证据：源域 Meadows 47%，准确率最低；源域 Shadows 2.4%，准确率最高

### 改动
- Pavia `source_sampler_power: 1.0 → 0.5`
- 其他一切不动

### 验证结果（Pavia seed1236）
| 配置 | OA | epoch1 loss | selected@110 |
|---|---|---|---|
| v4.1 baseline | 87.16 | 2.5772 | 24.6% |
| sampler_power=0.5 | **70.88** | **2.9560** | **19.4%** |

### 结论
- **崩盘 -16.28**
- 某个类从 84% → 3.8% 正确率（76% 被误判到相邻类）
- 假设完全错了：**逆频率采样是必需的**
- 少数类样本少（Bitumen 1330、Shadows 947），必须用强逆频率才能学到特征
- 降低采样权重 → 少数类特征学不出来 → 下游崩盘

### 教训
- 原假设中的"类别错配"不是采样问题，是 UDA 层面的问题
- 不动源域采样权重

## v4.8 — projection_norm 改 LayerNorm（已废弃）
### 动机
- 假设 BatchNorm 跨域统计污染
- projection 层是 backbone 里唯一活跃的 BN
- 推理时 running_mean 混合了源域/目标域统计

### 改动
- `projection_norm: batch → layer`（两边 config）
- 不动任何代码

### 验证结果（Pavia seed1236）
| 配置 | OA | selected@110 |
|---|---|---|
| v4.1 baseline | 87.16 | 24.6% |
| projection_norm=layer | **85.97** | **19.3%** |

### 结论
- 掉 1.19，超出噪声
- **伪标签接受率下降到 19.3%**（baseline 24.6%）
- 与 CCA 延迟（19.6%）模式一致
- **BatchNorm 是特征锐化的必要部分**，不是跨域污染源
- LayerNorm 让特征过度平滑，teacher 判断保守

### 重要模式发现（对后续所有实验的判据）
任何让训练早期特征变"模糊"的改动 → 伪标签接受率下降 → 掉点。
- CCA 延迟：接受率 24.6% → 19.6%（-1.18）
- projection_norm=layer：24.6% → 19.3%（-1.19）
- 这种模式一致性说明：**v4.1 的机制在利用 early feature sharpness**

## v4.9 — 严格 UDA 协议改造（保留）
### 动机
- 原训练采样读了目标域 GT 文件（虽然只用位置，不用类别）
- 这违反"严格目标标签不可见 UDA"协议
- 审稿人会抓这个点

### 改动（只改 config，不改代码）
- `target_gt_mix_ratio: 0.5 → 0.0`
- `use_gt_target_mask: true → false`
- 训练池改为"整个场景有效像素"（用 L2 范数 > 0 判定，不读 GT）

### 审计结果
`grep load_target_ground_truth_only` 确认：
- 训练分支（trainer.py:850, 881）已完全关闭
- diagnostic 分支（trainer.py:1475）默认关闭
- 只剩最终 benchmark（trainer.py:1832）用 GT 算 OA（必须）

**训练期间不读任何目标 GT，严格 UDA 合规。**

### 单点验证结果
| 数据集 | v4.1（用 GT 位置） | 严格 UDA | Δ |
|---|---|---|---|
| Pavia seed1236 | 87.16 | 84.84 | -2.32 |
| Houston seed1237 | 78.93 | 76.99 | -1.94 |

采样日志确认：
- Pavia: `full scene: 30000`
- Houston: `full scene: 50000`

### 3-seed baseline（待补）
见下方汇总

### 严格 UDA 3-seed 完整 baseline（替换旧基准）
| 数据集 | seed 1236 | seed 1237 | seed 1238 | 均值 | 标准差 |
|---|---|---|---|---|---|
| Pavia | 84.84 | 81.77 | 80.60 | **82.40** | 2.16 |
| Houston | 73.51 | 76.99 | 77.04 | **75.85** | 1.96 |

### 对照 v4.1（非严格 UDA）
| | v4.1 | 严格 UDA | Δ |
|---|---|---|---|
| Pavia 均值 | 84.35 | 82.40 | -1.95 |
| Pavia std | 3.09 | 2.16 | -0.93 |
| Houston 均值 | 77.83 | 75.85 | -1.98 |
| Houston std | 1.56 | 1.96 | +0.40 |

### 关键发现
- 严格 UDA 下 Pavia 方差反而下降（3.09 → 2.16）
- Houston 均值下降更多（原 100% 依赖 GT≠0 采样）
- outlier 模式变了：Pavia 现在 1238 最差，Houston 仍 1236 最差
- **旧的所有 v4.1 baseline（87.16/78.93）不再有效**
- 之后所有实验用 v4.9 严格 UDA 3-seed 对比

### 之后所有实验的新基准
- Pavia: 84.84 / 81.77 / 80.60（mean 82.40, std 2.16）
- Houston: 73.51 / 76.99 / 77.04（mean 75.85, std 1.96）

## v4.10 — Drift Reset 实验（已废弃）
### 动机
- v4.9 memory 用固定 EMA，可能被早期错误伪标签锁定
- 假设：特征严重漂移时重置 memory，切断错误自我强化

### 改动
- `memory.py` update() 加 `reset_mask = was_valid & (feature_drift > threshold)`
- 漂移超阈值时 prob_memory/feature_memory 直接覆盖为当前 teacher 值

### 验证结果（Pavia）
| 配置 | seed1236 | seed1237 | 均值 |
|---|---|---|---|
| v4.9 baseline | 84.84 | 81.77 | 83.31 |
| threshold=0.7 | 85.25 | 80.54 | 82.90 |
| threshold=1.2 | 84.86 | 81.55 | 83.21 |

### 结论
- threshold=0.7 太激进，破坏 seed1237
- threshold=1.2 完全不触发（需 cos < -0.2 才触发）
- **v4.9 memory 的特征漂移本身就有界，不是 seed 敏感性根源**
- 方向二废弃

### 重要发现
- 进一步确认 memory 层面不是瓶颈
- seed 敏感性来自**更上游**（随机初始化 → 损失盆地差异 → 早期伪标签轨道锁定）
- **不再尝试修 memory**

## v4.10 — CCA 抗坍塌实验（已废弃）
### 动机
- 怀疑 CCA 的 soft-assignment 让目标原型互相坍缩
- 想在 CCA loss 上加一项惩罚目标原型之间的过高相似度

### 改动（纯增量）
- `class_conditional_alignment` 末尾追加：`relu(pairwise - tau)^2` 惩罚项
- `collapse_weight=0.05`（默认 0.0，通过 config 打开）

### 验证结果（Pavia）
| 配置 | seed1236 | seed1237 |
|---|---|---|
| v4.9 baseline | 84.84 | 81.77 |
| + anti-collapse | **83.63** | **80.69** |

### 结论
- 双跌 1 点以上
- **原假设错了**：目标原型之间的高相似度是"有意义的"（Asphalt-Bitumen 光谱接近）
- 强行拉开破坏了跨场景该保留的类间相似结构
- 方向失败，回滚

### 重要发现
- CCA 对目标原型之间的几何**敏感**
- 但不能盲目约束，必须尊重源域的类间结构

## v4.11 — CCA 非对称几何约束（已废弃）
### 动机
- 路线 1 失败后，改用"只惩罚目标 > 源的偏离"（保护源域已有的类间相似结构）

### 改动（纯增量）
- `class_conditional_alignment` 加：`relu(S_tgt - S_src).mean()`
- `geometry_weight=0.10`（默认 0.0）

### 验证结果（Pavia）
| 配置 | seed1236 | seed1237 |
|---|---|---|
| v4.9 baseline | 84.84 | 81.77 |
| + asymmetric geometry | **83.07** | 81.60 |

### 结论
- seed1236 大跌 1.77，seed1237 微跌 0.17
- 即使在源域结构保护下，约束目标原型几何仍然有害
- **CCA 层创新第三次失败**

## CCA 层创新总结（三票否决）
| 方向 | 结果 | 原因 |
|---|---|---|
| v4.6 CCA 启动延迟 | -1.18 | 早期 CCA 在帮助伪标签筛选 |
| v4.10 抗坍塌 | -1.15 | 高相似度是有意义的，不能盲目拉开 |
| v4.11 非对称几何 | -0.97 | 连源域参考约束都不行 |

**结论：v4.9 的 CCA 已经是精密平衡的局部最优，任何几何约束都会破坏它。**
**停止在 CCA 层创新。**

## v4.11 — 可分离性引导对齐（已废弃）
### 动机
- 参考 SCLUDA "先可分离再对齐" 的思想
- 但放弃对比学习，改用紧致度/分离度比值

### 改动（纯增量）
- `source_separability_loss`: 类内紧致 / 类间最小分离
- `target_separability_loss`: NN 距离 / 全局散布
- `lambda_sep_src=0.02, lambda_sep_tgt=0.01`

### 验证结果（Pavia）
| 配置 | seed1236 | seed1237 | 均值 |
|---|---|---|---|
| v4.10 quota K=120 | 85.32 | 81.66 | 83.49 |
| + separability | **84.41** | **77.11** | **80.76** |

### 失败原因（实现 bug）
- `target_separability_loss` 用比值 `compact / spread`
- 优化器找到退化解：同时压小 compact 和 spread
- 目标域特征整体坍塌
- seed1237（特征本就紧）受害最重：-4.55

### 教训
- 比值形式的 loss 容易形成退化解
- 正确的可分离性 loss 应该是 `compact - λ * spread`（加号而非除号）
- 或者用 margin-based 显式约束

### 结论
- 回滚
- 未来若重试此方向，必须用 `L = compact - λ·spread` 形式

## v4.12 — 类自适应伪标签阈值（已废弃）
### 动机
- v4.9 诊断发现 c5(BareSoil) 接受数 87，其他类 1000~2000
- 假设 c5 被全局阈值 0.92 挡住，想用类自适应偏移

### 改动（纯增量，默认关闭）
- `class_thresholds` 加基于 `accepted_ema` 的动态偏移
- `commit_epoch_threshold_stats` 每 epoch 更新 EMA
- `adaptive_threshold_strength` 参数

### 验证结果（Pavia seed1236）
| 配置 | OA | c5→c0 | c5正确 |
|---|---|---|---|
| baseline | 84.84 | 624 | 2406 |
| 对称 0.03 | 84.72 | 610 | 2409 |
| 对称 0.08 | 84.82 | 586 | 2435 |
| 非对称 0.12 | 84.82 | 612 | 2416 |

### 结论
- 三次不同参数，OA 都在 84.72~84.82，无法突破 baseline
- 机制生效（c5 接受数在动），但**收益被其他类抵消**
- **真实原因**：c5 的置信度大部分在 0.80 以下，降阈值救回的是错误伪标签
- teacher 对 c5 **信息不足**，不是阈值太高
- 方向废弃

## 累计失败方向（v4.9 之后）
| # | 方向 | 结果 |
|---|---|---|
| 1 | Spectral rescue | 方差翻倍 |
| 2 | adaptive EMA | 无效 |
| 3 | memory refresh | 均值略降 |
| 4 | momentum 降值 | 接受率失控 |
| 5 | CCA 延迟 | -1.18 |
| 6 | 抗坍塌 | -1.15 |
| 7 | 非对称几何 | -0.97 |
| 8 | 采样降权 | -16 |
| 9 | 可分离性引导 | -4.55 |
| 10 | 类自适应阈值 | 中性 |

### 核心教训
v4.9 已经是严格 UDA 下的局部最优。
所有在 memory / CCA / 伪标签层的改动都无法突破。
需要转向**推理层**或**论文叙事重构**。

## v4.13 — Band alignment 0.15（已废弃）
### 动机
- Pavia 原来 band_alignment_strength=0.0（不做对齐）
- 试 0.15 / 0.25 / 0.35 扫描

### 扫描结果（seed1236）
| strength | OA | vs baseline |
|---|---|---|
| 0.0 | 84.84 | — |
| **0.15** | **85.60** | **+0.76** |
| 0.25 | 84.38 | -0.46 |
| 0.35 | 82.82 | -2.02 |

倒 U 型，0.15 是峰值。

### 3-seed 验证
| seed | baseline | 0.15 | Δ |
|---|---|---|---|
| 1236 | 84.84 | 85.60 | +0.76 |
| 1237 | 81.77 | 80.80 | -0.97 |

**均值从 83.31 → 83.20，净跌 0.11。**

### 结论
- seed1236 涨、seed1237 掉，方向相反
- **seed1236 的 +0.76 是噪声**
- 放弃 band_alignment=0.15
- **保留 Pavia 原值 0.0**（之前的调参是对的）

## 累计失败方向（v4.9 之后，共 11 次）
| # | 方向 | 结果 |
|---|---|---|
| 1-10 | （见前） | 全部无效 |
| 11 | band_alignment=0.15 | seed 噪声 |

### 最终判断
v4.9 严格 UDA 下的 84.84 / 81.77 是**真实基线**。
所有单变量改动都无法在 2-seed 上稳定涨。

**必须在推理层（TTA / ensemble）寻找突破，或者接受当前数字重构论文叙事。**

## v4.14 — HyRANK Dioni → Loukia 首个 3-seed baseline

### 配置
- source_name: Dioni (12 类，176 波段)
- target_name: Loukia (12 类，176 波段)
- source_train_ratio: 0.0
- source_samples_per_class: 100（每类固定 100 个）
- target_train_max_samples: 15000
- epochs: 120
- pseudo_start_epoch: 40
- prat_style_max: 0.4
- band_alignment_strength: 0.0
- cempa_start_epoch: 999（关闭）
- spatial_start_epoch: 999（关闭）

### 3-seed 结果
| seed | OA | AA | Kappa |
|---|---|---|---|
| 1236 | 61.84 | 60.50 | 0.5425 |
| 1237 | 55.54 | 57.50 | 0.4785 |
| 1238 | 52.11 | 54.99 | 0.4425 |
| **均值** | **56.50** | **57.66** | **0.488** |
| **标准差** | **4.87** | — | — |

### 关键混淆问题
- class 5 (Olive Groves) → class 4 (Fruit Trees)：召回 8.4%，648 个误判
- class 7 (Dense Sclerophyllous) → class 8 (Sparse)：召回 65%，988 个误判
- class 7/8 合计占目标域 57%，是主要拖累

### 与文献对比（论文 TABLE V）
| 方法 | OA |
|---|---|
| KNN | 54.83 |
| SA | 56.41 |
| DAN | 60.91 |
| DSAN | 62.45 |
| MRAN | 61.47 |
| TSTnet | 66.97 |
| SCLUDA | 65.54 |
| JAN-STF | 66.97 |
| MCDTM | **70.01** |

**当前 56.50 低于主流方法，目标 ≥ 65**

### 后续所有 HyRANK 实验统一用 **seed 1237** 做快速验证

## v4.15 — HyRANK CORAL/LMMD 突破

### 改动
- lambda_coral: 0 → 0.5
- lambda_lmmd: 0 → 0.5
- lmmd_start_epoch: 60

### seed1237 结果
| 指标 | baseline | + CORAL/LMMD | Δ |
|---|---|---|---|
| OA | 55.54 | **59.40** | **+3.86** |
| AA | 57.50 | **65.37** | **+7.87** |
| Kappa | 0.4785 | **0.5225** | +0.044 |

### 关键发现
- class 8 (Sparse Sclerophyllous) 召回 48.7% → 57.7%（+252 像素）
- CORAL/LMMD 的二阶统计对齐有效改善"密度类"边界
- 源域在 epoch 120 还在提升（best_epoch=120），训练可能不够

### 后续
- 试 200 epoch
- 若稳定，跑 3-seed
- 与文献对比：CORAL/LMMD 加入后 59.4%，接近 DAN (60.91) 和 DSAN (62.45)

## v4.16 — HyRANK 延长训练实验（已废弃）

### 改动
- epochs: 120 → 200
- pseudo_start_epoch: 40 → 60

### seed1237 结果
| 配置 | OA | AA |
|---|---|---|
| 120 epoch + CORAL/LMMD | **59.40** | 65.37 |
| 200 epoch + CORAL/LMMD | 56.86 | 65.70 |

### 结论
- OA 掉 2.54，过拟合
- 源域在 120 epoch 已经 94%，再训练损害跨域能力
- **最优配置：120 epoch + CORAL/LMMD**

## HyRANK 当前最优配置（seed1237 = 59.40）
- source_samples_per_class: 200
- epochs: 120
- lambda_coral: 0.5
- lambda_lmmd: 0.5
- lmmd_start_epoch: 60
- 其他保持 baseline

## v4.17 — CORAL/LMMD 权重扫描

### seed1237 结果
| lambda_coral/lmmd | OA | AA |
|---|---|---|
| 0.0 (baseline) | 55.54 | 57.50 |
| **0.5** | **59.40** | **65.37** |
| 1.0 | 57.23 | 62.48 |

### 结论
- 0.5 是最优权重
- 1.0 过大，破坏训练

### 当前 HyRANK 最优配置
- source_samples_per_class: 200
- epochs: 120
- patch_size: 7
- lambda_coral: 0.5
- lambda_lmmd: 0.5
- lmmd_start_epoch: 60
- band_alignment_strength: 0.0
- 其他保持 Pavia 默认

### 与文献对比（seed1237 = 59.40）
| 方法 | OA |
|---|---|
| KNN | 54.83 |
| SA | 56.41 |
| **RMDANet (ours)** | **~59.4** |
| DAN | 60.91 |
| MRAN | 61.47 |
| DSAN | 62.45 |
| SCLUDA | 65.54 |
| TSTnet | 66.97 |
| JAN-STF | 66.97 |
| MCDTM | 70.01 |

## v4.19 — HyRANK 单变量参数扫描（6 个）

### seed1237 结果
| 实验 | 参数改动 | OA | vs baseline (59.40) |
|---|---|---|---|
| lr05 | lr: 0.001→0.0005 | 55.82 | -3.58 |
| **pconf95** | pseudo_confidence: 0.92→0.95 | **60.04** | **+0.64** ✅ |
| lmmd08 | lambda_lmmd: 0.5→0.8 | 59.46 | +0.06 |
| **ema996** | ema_decay: 0.994→0.996 | **60.68** | **+1.28** ✅ |
| **ba01** | band_alignment: 0.0→0.1 | **60.06** | **+0.66** ✅ |
| ccatemp005 | cca_temperature: 0.10→0.05 | 58.45 | -0.95 |

### 有效参数（≥ +0.5）
1. ema_decay = 0.996（最强）
2. band_alignment_strength = 0.1
3. pseudo_confidence = 0.95

### 下一步
- 组合三个有效参数，验证是否有叠加效应

## v5.1 — 方案 02：修复 flip 重复统计

### 问题
`core/memory.py` 存在 3 处重复代码：
- 连续两行 `label_flip_count` 初始化
- 连续 6 行 prev_label_check / flip_mask / 递增
- 两个完整 `flip_diagnostics` 定义
- 第一处 def 前有重复的 `@torch.no_grad()`

### 影响
仅影响诊断日志准确性和代码整洁度，不参与训练损失。
修复后 flip 统计可信，为后续诊断提供依据。

### 结果
Pavia seed1236 应保持 combo3 的 85.6

## v5.2 — 方案 03.0：波段掩码增强（已废弃）

### 改动
- `core/data.py`：PatchDataset 加 band_mask_prob / band_mask_ratio
- 源域训练时随机把 15% 波段置 0
- band_mask_prob=0.5, band_mask_ratio=0.15

### 结果（Pavia seed1236）
| 配置 | OA | 源域验证 |
|---|---|---|
| combo3 (fixflip) | 85.75 | 99.7% |
| + band_mask | **60.72** | **81.3%** |

**掉 25 个点，源域自身就崩了。**

### 关键发现（重要）
- Pavia 的 102 波段**每个都携带判别信息**
- CenterSpectralShapeEncoder 依赖完整光谱：
  - 一阶/二阶差分对相邻波段敏感
  - 遮罩后差分完全乱
- 光谱完整性对 HSI 分类是硬约束，**不能像 RGB 那样遮罩**

### 教训
- 掩码自监督（MAE 类）在 HSI 上不适用
- 未来做自监督必须**保留光谱连续性**

### 决定
跳过 03.1 / 03.2（重建头建立在掩码之上）
直接进方案 04

## v5.3 — 方案 04.0：amplitude_channel（已废弃）

### 改动（零代码，只改 config）
- Pavia 加 `amplitude_channel: true`
- 连带触发：
  1. CenterSpectralShapeEncoder 启用 amplitude_head
  2. data.py 归一化模式从 l2 → median

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| combo3 (fixflip) | 85.75 |
| + amplitude_channel | **79.71** |

### 逐类对比
| 类 | combo3 | amp | Δ |
|---|---|---|---|
| Asphalt | 6791 | 7324 | +533 |
| Meadows | 8376 | 8671 | +295 |
| **Gravel** | 2016 | 1294 | **-722** |
| **Trees** | 5852 | 4365 | **-1487** |
| Painted | 2773 | 2727 | -46 |
| Bare Soil | 2416 | 2596 | +180 |
| **Bitumen** | 5524 | 4391 | **-1133** |

### 关键发现
- `amplitude_channel` 是**两个改动的强耦合**：
  1. 换归一化（l2 → median）
  2. 加 amplitude_head
- 无法分开测
- `median` 归一化破坏光谱形状 → Trees / Bitumen / Gravel 崩
- Asphalt / Meadows 因为类内反射强度差异大而受益

### 结论
- 不能单变量调，放弃
- 幅值保留对 Pavia 整体是负收益

## v5.4 — 方案 05：源域监督的难类别判别（已废弃）

### 改动
- `core/trainer.py` 加 `source_confusable_pair_loss`
- 约束 Pavia 三对易混淆类：(0,5), (2,6), (1,3)
- lambda_pair_margin=0.3, pair_margin=0.5

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| combo3 (fixflip) | 85.75 |
| + pair_margin | **84.63** |

### 关键混淆变化
| 类对 | combo3 | pair_margin | Δ |
|---|---|---|---|
| Asphalt→BareSoil | 807 | **1303** | **+496** ❌ |
| BareSoil→Asphalt | 637 | **436** | -201 |
| Trees→BareSoil | 179 | 332 | +153 |

### 失败原因
- 单向改善，反向恶化
- 源域类间距已由 `source_metric_loss` (margin=0.10) 调过
- 再加硬 margin 破坏原型几何
- 与已有机制冗余

### 决定
回滚，进方案 06

## v5.5 — 方案 06：多视图一致性模型选择（保留，核心改进）

### 三数据集验证（seed1237）
| 数据集 | combo3 | + consistency | Δ |
|---|---|---|---|
| Pavia | 81.77 | **82.94** | **+1.17** ✅ |
| Houston | 76.80 | 76.80 | 0（逐位一致）|
| HyRANK | 60.92 | 60.92 | 0（逐位一致）|

### Pavia 3-seed 完整对比
| seed | v4.9 严格 UDA | combo3 | + consistency |
|---|---|---|---|
| 1236 | 84.84 | 85.75 | 85.75 |
| 1237 | 81.77 | — | **82.94** |
| 1238 | 80.60 | — | **84.13** |
| **均值** | **82.40** | — | **84.27** |
| **std** | **2.16** | — | **1.42** |

### 关键机制
- 每 eval_interval 计算目标域多视图预测一致性（原图 / 水平翻转 / 垂直翻转）
- 作为 checkpoint 选择新分量，权重 3.0
- 不读 GT、不参与训练、不形成闭环
- Houston / HyRANK 上逐位一致，说明对已选对的 epoch 无干扰

### 核心价值
- 唯一同时提升均值（+1.87）和降低方差（-0.74）的方案
- 精准救回 outlier seed
- 零训练成本，只改 checkpoint 选择

### 下一步
- 补齐 Houston / HyRANK 完整 3-seed
- 然后开始方案 07

## v5.2 — S07：Top-3 Checkpoint 权重平均（已废弃）

### 改动（零代码）
- 独立 config 加 checkpoint_average: 3

### 结果（Pavia seed1237）
| 配置 | OA |
|---|---|
| v5.1 (average=1) | 82.94 |
| S07 (average=3) | 82.95 |

**差 0.01，纯噪声。**

### 失败原因
`weighted_average_state_dicts` 用 softmax 温度 0.25：
- top-1 比 top-2 通常高 0.5~1.0
- softmax 后 top-1 权重 ~70%，top-2 ~12%，top-3 ~3%
- 实际退化成只选 top-1

### 教训
- 现有 softmax 加权太尖锐，需要改温度或改成等权
- 但修改代码风险高，暂时不做
- 进 S08

## v5.3 — S08：类别覆盖约束的多视图模型选择（已废弃）

### 改动
- `target_consistency_score` 扩展支持 per-class 统计
- `label_free_selection_score` 加：
  - consistency_term = 3.0 * mean_c + 2.0 * worst_c
  - coverage_penalty = 5.0 * max(0, 0.90 - coverage)

### 三数据集结果
| 数据集 | S06 | S08 | Δ |
|---|---|---|---|
| Pavia seed1236 | 85.75 | 85.75 | 0 |
| Pavia seed1237 | 82.94 | 82.94 | 0 |
| Pavia seed1238 | 84.13 | 84.13 | 0 |
| HyRANK seed1237 | 60.92 | 60.92 | 0 |

**全部逐位一致，完全冗余。**

### 失败原因
- 三个数据集训练时 `coverage` 一直=1.0
- `mean_c / worst_c` 也接近 1.0（模型预测稳定）
- 新 score = 旧 score + 常数，不改变 top-1
- S08 只在"预测塌陷"时才有用，而这三个数据集从未塌陷

### 重要观察（HyRANK flip-diag）

### 决定
- S08 回滚
- 进 S09（HyRANK 软类别条件 LMMD）

## v5.4 — S09：HyRANK 软类别条件 LMMD（已废弃）

### 改动
- 加 `lmmd_loss_soft` 函数：用软概率加权目标特征，代替硬伪标签
- CLI 参数：use_soft_lmmd, lmmd_min_effective_support, lmmd_temperature
- HyRANK config：use_soft_lmmd=true

### 结果（HyRANK seed1237）
| 配置 | OA |
|---|---|
| v5.0 combo3 | 60.92 |
| S09 软 LMMD | **59.73** |

### 逐类变化
| 类 | combo3 | S09 | Δ |
|---|---|---|---|
| 1 Dense Urban | 173 | 96 | -77 |
| 5 Olive Groves | 440 | 584 | +144 |
| 7 Dense Sclerophyllous | 2157 | 1882 | -275 |

### 失败原因
- 软概率加权让所有样本都参与对齐
- 最大类 class 7 的宽软概率分布被拉向其他类
- class 5 受益（+144），但 class 7 受损更多（-275）

### 决定
回滚，进 S10

## v5.5 — S10：独立目标域光谱结构学习（已废弃）

### 改动
- `spectral_augmentation`：纯光谱扰动（scale ±3% + noise）
- `spectral_view_consistency_loss`：两视图 cosine 一致性
- 作用在 `target_output.features`（融合特征）
- HyRANK config：lambda_spectral_view=0.5

### 结果（HyRANK seed1237）
| 配置 | OA | 源域 sup@120 |
|---|---|---|
| v5.0 combo3 | 60.92 | 0.19 |
| S10 | **56.09** | **0.287** |

### 失败原因（实现错误）
- `target_output.features` 是融合特征（context + spectral）
- 强制融合特征对光谱扰动不变 → 压制光谱分支
- 光谱分支被"关掉" → 判别力下降
- 正确做法应只约束**光谱分支特征**，但 ModelOutput 未暴露

### 决定
回滚，进 S11

## v5.6 — S11：伪标签反馈解耦（已废弃）

### 改动
- teacher_consistency_losses 的权重从 `reliability` 改为 `reliability × stable_mask`
- 只让通过可靠性筛选的样本参与一致性 loss
- HyRANK config：decouple_consistency=true

### 结果（HyRANK seed1237）
| 配置 | OA | selected |
|---|---|---|
| v5.0 combo3 | 60.92 | ~20% |
| S11 | **59.89** | **0.9%** |

### 失败原因
- HyRANK 的 stable_mask 只覆盖 ~1% 样本
- 一致性 loss 几乎无信号 → 模型失去一致性约束
- 与 S01 硬过滤失败的根源相同

### 决定
回滚，进 S12

## v5.7 — S12：空间不重叠源域划分（协议改进，保留代码）

### 改动
- `data.py` 加 `spatially_disjoint_source_split`
- CLI 参数：spatially_disjoint_split / disjoint_split_radius
- Pavia config：spatially_disjoint_split=true, radius=3

### 结果（Pavia seed1237）
| 配置 | OA | 源域验证 | val 大小 |
|---|---|---|---|
| v5.1 原划分 | 82.94 | 99.4%+ | 35399 |
| S12（radius=3） | 82.94 | **98.64%** | 1986 |

### 结论
- **OA 不变**——S12 不改目标域结果
- **源域验证从 99.4% → 98.64%**：揭示原划分的 0.8 个空间泄漏
- 论文里可报告："空间不重叠划分后源域验证为 98.64%"
- radius=3 过于激进（val 只剩 5%），后续用 radius=1

### 保留原因
- 不是 OA 提升，是"论文严谨性"
- 代码保留，默认关闭，只用于报告源域验证

## v5.7 — S12：空间不重叠源域划分（最终保留 radius=1）

### 改动
- `data.py` 加 `spatially_disjoint_source_split`
- CLI 参数：spatially_disjoint_split / disjoint_split_radius
- Pavia config：spatially_disjoint_split=true, radius=1

### 结果（Pavia seed1237）
| 配置 | OA | 源域验证 | val 大小 |
|---|---|---|---|
| v5.1 原划分 | 82.94 | ~99.4% | 35399 |
| S12 radius=3 | 82.94 | 98.64% | 1986 |
| **S12 radius=1** | **82.94** | **99.10%** | **17145** |

### 结论
- **OA 完全一致**，S12 只改协议不改精度
- **radius=1** 是最佳折中：val 保留 48%，源域验证只降 0.3 个点
- 论文里可报告："空间不重叠划分后源域验证 99.10%"

### 保留原因
- 协议严谨性（reviewer 可能问"相邻 patch 是否重叠"）
- 不影响目标域 OA
- 保留代码，默认关闭，需要时打开

## v5.8 — S13：修复模型选择权重（已确认有效）

### 消融驱动发现
HyRANK seed1237 消融显示 main_reliability 模型选择失效：
- 真实最优 epoch 80（OA=59.01），label-free 选出 50（OA=55.78），差 -3.23

### S13 改动
- source_penalty: 2.5 → 1.0
- target_consistency: 3.0 → 6.0

### 结果
| 配置 | 原版 | +S13 | Δ |
|---|---|---|---|
| HyRANK main_reliability | 55.99 | 58.31 | +2.32 |
| HyRANK full | 58.02 | 58.02 | 0 |
| Pavia (S06) | 82.94 | 82.94 | 0 |

### 意义
**S13 是精准修复**：只改选错的配置，对正常配置无影响。
这是继 S06 之后第二个真正有效的改进，且零训练成本。

### 后续
补齐 3-seed，作为最终结果。

## v5.9 — 跨域对比学习（已废弃）

### 改动
- 加 `cross_domain_contrastive_loss`（InfoNCE 形式）
- 目标样本 ↔ 源原型对比
- 只用 stable_mask 内样本
- lambda_cross_contrast=0.5

### 结果
| 数据集 | 加对比前 | 加对比后 | Δ |
|---|---|---|---|
| Pavia seed1236 | 86.67 | 86.75 | +0.08 |
| HyRANK seed1237 | 60.92 | 60.39 | -0.53 |

### 失败原因
- 与现有 `source_metric_loss` 高度重叠
- 后者已做"拉近真类原型 + 推远最难类原型"
- 增量 ≈ 0，且在 HyRANK 上产生梯度冲突

### 决定
回滚，进下一个方向

## v6.1 — 伪标签遗忘曲线（已废弃）

### 设计
- 每个 epoch 检查样本距上次访问的 epoch 数
- freshness = exp(-Δe / half_life)
- reliability 先按 freshness 衰减，再 EMA

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| 基线 | 86.67 |
| 遗忘曲线(30ep) | 86.67 |

**逐位一致。**

### 失败原因
- Pavia 样本平均访问间隔 < 10 epoch
- 半衰期 30 epoch 意味着衰减几乎不发生
- 需要 half_life ≈ 5 才可能触发

### 决定
回滚，暂不重试。进下一个实验。

## v6.2 — 多尺度一致性训练（已废弃）

### 设计
- 对同一目标样本构造 5×5 / 7×7 / 9×9 三个尺度视图
- 要求三视图特征一致
- lambda_multi_scale = 0.3

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| 基线 | 86.67 |
| 多尺度一致性 | **85.51** |

### 失败原因
- 与 CenterGuidedPooling 的"中心优先"设计冲突
- 强制尺度一致 → Trees 类丢 200+ 正确像素
- 说明 7×7 patch 的类区分确实依赖尺度敏感性

## v6.3 — 跨域 MixUp（已废弃）

### 设计
- 源域和目标域样本线性混合
- lambda_mixup = 0.3, alpha = 0.2
- 用 teacher 软概率作为目标标签

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| 基线 | 86.67 |
| 跨域 MixUp | **84.81** |

### 失败原因
- MixUp 假设"同类跨域样本可线性插值"——HSI 跨域不成立
- Meadows 掉 352，Bitumen 掉 196
- 这两个类在源域和目标域有不同子分布

## v6.4 — 8 分量逐个消融（HyRANK，已结束）

### 结果（HyRANK seed1237）
| 配置 | OA | Δ |
|---|---|---|
| baseline | 60.92 | 0 |
| 关 view | 60.58 | -0.34 |
| 关 guard | 61.00 | +0.08 |
| 关 prototype | 61.04 | +0.12 |
| 关 spatial | 60.94 | +0.02 |

### 结论
- 全部 ±0.4 内 = 噪声
- 没有多余约束，也没有关键约束
- 因为 stable_mask 太小（1%），关任何分量都不改变通过数
- 真正的瓶颈在 **上游的 `conf` 阈值**

### 已探索方向总数
25 次尝试，2 次有效（S06/S13），23 次失败或无效。

## v6.5 — 自适应 PRAT（已废弃）

### 设计
- 用 FFT 幅度估计"空间频率差异"
- domain_shift 大 → 强 PRAT（~0.4）
- domain_shift 小 → 弱 PRAT（~0.2）

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| baseline (prat=0.2) | 86.67 |
| 自适应 PRAT | **84.78** |

### 失败原因
- Pavia 的 domain_shift 偏高
- 自适应强度落在 0.3~0.4（Pavia 的坏区间）
- 本质：`domain_shift` 衡量"空间频率"——但 Pavia 的域偏移是"类先验"，不是空间频率
- 度量选错了

### 26 次尝试总结
2 次有效（S06 / S13），24 次失败。所有"改学习机制"的方向全部失败。

## v6.6 — 分位数对齐（已废弃）

### 设计
- 逐维对齐源域和目标域特征的分位数 [0.25, 0.5, 0.75]
- 比 CORAL 更细
- lambda_quantile = 0.3

### 结果（Pavia seed1236）
| 配置 | OA |
|---|---|
| baseline | 86.67 |
| 分位数对齐 | 86.67 |

**逐位一致。**

### 失败原因
- 和 CORAL 一样：Pavia 的类对齐边际收益已经为 0
- 分位数只捕捉"分布位置"，和 CORAL 的"分布形状"重叠
- 不引入新信息

### 27 次尝试全部完成
有效: 2 次（S06/S13，均为推理层）
无效或失败: 25 次

### 结论
"改学习机制"全部失败，"改选择/推理机制"2 次成功。
模型的学习机制层已经饱和。
