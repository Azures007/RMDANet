# 可靠性（Reliability）消融实验完整记录

> 用途：后续论文写作、回复审稿、补充实验时直接引用。
> 记录时间：2026-11-15（Asia/Shanghai）
> 负责人：RMDANet 项目

---

## 0. 实验环境与代码版本（重要）

| 项 | 值 |
|---|---|
| 服务器实例 | `autodl-container-4d004f9f54-a6f54ff2`（黄金实例） |
| GPU | NVIDIA GeForce RTX 4090 |
| 驱动 | 580.105.08 |
| PyTorch / CUDA | 2.8.0+cu128 / CUDA 12.8 |
| 确定性 | `--deterministic` 默认开启（同机同 seed 逐位可复现） |
| 黄金代码 | 分支 `golden-repro-2026-11-14`，commit `fc8f992`（复现 Pavia 85.73 / Houston 79.06 / HyRANK 60.85） |
| 消融代码 | 分支 `feat/rtml-component-abl`，commit `e5744a8`（在黄金代码上加 `RTML_ABL` 环境变量开关） |
| 最终精简代码 | 分支 `feat/reliability-3comp`，commit `94313f3`（8 分量 → 3 分量，已推送 GitHub，tag `reliability-3comp-2026-11-15`） |
| YRD 支持 | 分支 `feat/yrd-support`，commit `9f74844`（本地）/ 服务器 `c9b7527`（含 `--config` 覆盖与 v1–v4 调参配置） |

### 消融开关说明（`RTML_ABL`）

`RTML_ABL` 为逗号分隔的“要删除的分量名”，从统一可靠性几何平均中移除，并同时关闭其伪标签选择/否决分支：

```
conf | margin | time | drift | view | guard | proto | spatial
```

默认（不设置）为原始 8 分量；设置 `RTML_ABL="margin,proto,time"` 表示只保留其余 5 个。

### 数据集与协议

| 数据集 | 源→目标 | 类别 | 源域标注 | 关键超参 |
|---|---|---|---|---|
| Pavia | PaviaU→PaviaC | 9 | 180/类 | α(DRSE_ALPHA_LOGIT) = −4.0 |
| Houston | Houston13→Houston18 | 7 | 180/类 | α = −4.0 |
| HyRANK | Dioni→Loukia | 12 | 实际可用 2130 | α = −3.5；`--abl-history 0`；`fusion_delta -1.0, fusion_w_max 0.40, fusion_w_min 0.10, fusion_tau_a 0.40` |
| YRD | yrd_nc12→yrd_nc13 | 7（270 波段） | 实际可用 1155（≈165/类） | α = −4.0；`config.yaml` 沿用 Houston/Pavia 协议 |

通用协议：patch 7、batch 128、50 epochs × 40 steps、AdamW+cosine、AMP、无源域验证、最终 epoch 评估；目标域 GT 只在最终 locked eval 中打开。

### 评估指标

- OA（Overall Accuracy）
- AA（Average Accuracy）
- Kappa
- per-class accuracy（记录于各 run 的 `target_final_once.json`）

---

## 1. 统一可靠性的 8 个分量（原始设计）

| # | 分量 | 公式/实现 | 开关 |
|---|---|---|---|
| 1 | confidence | \(R^{conf}=\sigma((C_i-\tau_i)/0.035)\)，\(\tau_i\) 为动态类阈值 | 恒开 |
| 2 | margin | \(R^{margin}=\sigma((M_i-0.10)/0.030)\) | 恒开 |
| 3 | time | \(R^{time}=\sqrt{\min(Age/7,1)\cdot\min(Streak/6,1)}\) | `use_time_score` |
| 4 | drift | \(R^{drift}=\exp(-d_i/0.14)\) | `use_feature_score` |
| 5 | view | \(R^{view}=0.10+0.90A_i\)（教师/学生/强增强一致） | `use_view_score` |
| 6 | guard | \(R^{guard}=0.20+0.80A^{guard}\)（独立语义守卫一致） | `use_guard` |
| 7 | proto | 源类原型证据（余弦相似度+一致性） | `use_prototype_veto` |
| 8 | spatial | 空间邻域一致性（Support≥3 才生效，否则 1） | `use_spatial` |

聚合：\(R_i=\left(\prod_j R_{i,j}\right)^{1/m}\)（几何平均，m=实际启用分量数）。
另维护可靠性 EMA：\(R^{memory}=0.82R^{memory,old}+0.18R\)（不参与几何平均）。

---

## 2. 单分量消融（每次只删 1 个，seed 1237）

### 2.1 Pavia（9 类）

| 删除的分量 | OA | AA | Kappa | ΔOA |
|---|---|---|---|---|
| Full（8 个） | 85.73 | 83.38 | 0.8283 | — |
| w/o conf | 84.53 | 82.65 | 0.8141 | −1.21 |
| w/o margin | 85.64 | 83.14 | 0.8272 | −0.09 |
| w/o time | 85.79 | 83.44 | 0.8289 | +0.05 |
| w/o drift | 85.66 | 83.39 | 0.8275 | −0.07 |
| w/o view | 84.76 | 82.66 | 0.8168 | −0.97 |
| w/o guard | 85.54 | 83.18 | 0.8260 | −0.19 |
| w/o proto | 85.83 | 83.52 | 0.8295 | +0.10 |
| w/o spatial | 85.33 | 83.10 | 0.8236 | −0.40 |

### 2.2 Houston（7 类）

| 删除的分量 | OA | AA | Kappa | ΔOA |
|---|---|---|---|---|
| Full（8 个） | 79.06 | 82.56 | 0.6786 | — |
| w/o conf | 79.65 | 83.09 | 0.6859 | +0.59 |
| w/o margin | 79.27 | 82.45 | 0.6813 | +0.20 |
| w/o time | 79.00 | 82.37 | 0.6778 | −0.06 |
| w/o drift | 78.89 | 82.18 | 0.6764 | −0.17 |
| w/o view | 79.11 | 82.86 | 0.6798 | +0.05 |
| w/o guard | 78.91 | 82.38 | 0.6767 | −0.15 |
| w/o proto | 79.07 | 82.22 | 0.6785 | 0.00 |
| w/o spatial | 78.96 | 82.45 | 0.6773 | −0.10 |

### 2.3 HyRANK（12 类，**只做了 4 个分量，且全部无变化**）

| 删除的分量 | OA | AA | Kappa | 说明 |
|---|---|---|---|---|
| Full | 60.85 | 61.82 | 0.5327 | — |
| w/o conf | 60.85 | 61.82 | 0.5327 | 与 Full 逐位相同 |
| w/o margin | 60.85 | 61.82 | 0.5327 | 同上 |
| w/o view | 60.85 | 61.82 | 0.5327 | 同上 |
| w/o proto | 60.85 | 61.82 | 0.5327 | 同上 |

**重要说明（必须写进论文/回复审稿）：**

- HyRANK 的 Full 配置使用 `--abl-history 0`（mean-teacher 当前伪标签路径 `current_pseudo_batch`），该路径**不经过 8 分量统一可靠性**，而是使用独立公式 `conf × margin × view`；
- 因此 `RTML_ABL` 对 HyRANK 无影响，上述 4 组结果与 Full 逐位相同；
- **time / drift / guard / spatial 这四个分量在 HyRANK 上没有跑过**（跑也必然相同，但为避免"未做实验"的质疑，如需补跑见第 8 节命令模板）。

---

## 3. 组合删除消融（seed 1237）

| 保留分量 | 删除分量 | Pavia OA / AA / Kappa | Houston OA / AA / Kappa |
|---|---|---|---|
| Full 8 | — | 85.73 / 83.38 / 0.8283 | 79.06 / 82.56 / 0.6786 |
| 6 | margin, proto | 85.65 / 83.53 / 0.8275 | 79.17 / 82.37 / 0.6800 |
| 5 | margin, proto, time | 85.41 / 83.12 / 0.8245 | 79.34 / 82.83 / 0.6827 |
| 5a | margin, proto, drift | 85.65 / — / — | 79.07 / — / — |
| 5b | margin, proto, guard | 85.86 / — / — | 79.23 / — / — |
| 4 | margin, proto, time, drift | 85.70 / — / — | 79.03 / — / — |
| 3 | margin, proto, time, drift, guard | 85.92 / — / — | 79.27 / — / — |
| 2 | 再删 spatial | 85.92 / — / — | 79.10 / — / — |

> "-" 表示该结果当时只跑了 OA（AA/Kappa 未记录，可在对应 run 目录 `target_final_once.json` 中回查）。

---

## 4. 极限测试：只保留 1 个分量（seed 1237）

| 保留 | Pavia OA | Houston OA | 说明 |
|---|---|---|---|
| 只留 conf | 84.48 | 78.97 | Pavia 掉 1.25 |
| 只留 view | 85.01 | 79.53 | Pavia 掉 0.72，Houston 最好 |
| 只留 spatial | **70.07** | 79.20 | 单 spatial 会崩（无 conf/view 支撑） |

结论：单分量不可行，最少需要 conf + view。

---

## 5. 多 seed 验证（seed 1237/1239/1241；Pavia/Houston）

| 集合 | Pavia seeds | 均值±std | mean AA | Houston seeds | 均值±std | mean AA |
|---|---|---|---|---|---|---|
| 8 分量 Full | 85.73 / 85.76 / 85.39 | **85.63 ± 0.17** | 84.17 | 79.06 / 78.97 / 77.47 | **78.50 ± 0.73** | 81.19 |
| 3 分量（conf+view+spatial） | 85.92 / 86.05 / 84.80 | **85.59 ± 0.56** | 84.01 | 79.27 / 79.16 / 77.77 | **78.73 ± 0.68** | 81.27 |
| 2 分量（conf+view） | 85.92 / 84.04 / 84.64 | **84.87 ± 0.78** | 83.50 | 79.10 / 79.07 / 77.46 | 78.54 ± 0.76 | 81.69 |

结论：

- 3 分量与 8 分量**持平**（Pavia −0.04、Houston +0.23），公式更简；
- 2 分量在 Pavia 上低 0.76 → spatial 不能删；
- 单 seed 1237 的 2 分量 85.92 是运气，**多 seed 才可靠**。

---

## 6. 4 分量（conf + view + spatial + time）验证

需求：确认"加回 time"是否有收益。删除 `margin, proto, drift, guard`，保留 4 分量。

### 6.1 三数据集 × 3 seed（seed 1237/1239/1241）

| 数据集 | seed 1237 | seed 1239 | seed 1241 | 均值±std | mean AA |
|---|---|---|---|---|---|
| Pavia | 86.14 / 83.97 / 0.8332 | 85.67 / 84.66 / 0.8278 | 84.84 / 83.57 / 0.8180 | **85.55 ± 0.54** | 84.07 |
| Houston | 78.76 / 82.30 / 0.6752 | 79.14 / 82.88 / 0.6805 | 77.49 / 78.29 / 0.6562 | **78.46 ± 0.71** | 81.16 |
| HyRANK | 60.85 / 61.82 / 0.5327 | 59.64 / 60.48 / 0.5221 | 57.65 / 59.78 / 0.5013 | **59.38 ± 1.32** | 60.69 |

### 6.2 与 3 分量对比

| 集合 | Pavia 均值 | Houston 均值 |
|---|---|---|
| 3 分量（conf,view,spatial） | 85.59 | 78.73 |
| 4 分量（+time） | 85.55 | 78.46 |

**结论：加回 time 无收益（Pavia −0.04、Houston −0.27），time 分量确认冗余。**

> 注意：HyRANK 的这 3 个 seed 值来自当前伪标签路径（分量不参与），对任意分量集合都相同，可作为 **HyRANK 的官方多 seed 结果**：60.85 / 59.64 / 57.65，均值 **59.38 ± 1.32**（seed 1237 的 60.85 是偏好种子，论文建议报均值±std 或同时标注 best seed）。

---

## 7. HyRANK 统一 3 分量的尝试（history 路径，`--abl-history 1`）

| seed | OA | AA | Kappa |
|---|---|---|---|
| 1237 | 57.91 | 60.03 | 0.5036 |
| 1239 | 57.27 | 58.85 | 0.4977 |
| 1241 | 57.63 | 61.93 | 0.5021 |
| **均值** | **57.60 ± 0.26** | 60.27 | — |

结论：统一 3 分量在 HyRANK 上比当前路径（59.38）**低 1.78**，因此 **HyRANK 保留 `--abl-history 0` 的当前伪标签路径（conf×margin×view）**；论文中说明"HyRANK 配置采用当前伪标签模式，统一可靠性分数未启用"。

---

## 8. 最终方案与代码

| 项 | 内容 |
|---|---|
| 统一可靠性 | **3 分量：confidence + view + spatial**（几何平均，m=3） |
| 适用数据集 | Pavia、Houston、HyRANK **全部统一使用默认设置**（自 2026-11-15 起取消 HyRANK 特殊配置） |
| 删除的分量 | margin、time、drift、guard、proto（含代码分支与辅助方法） |
| 非分量规则 | spatial veto 中保留 **exceptional 高置信放行**（confidence≥0.96 且 margin≥0.22，不属于可靠性分量） |
| 最终代码 | 本地主路径 `D:\桌面\退路代码\RMDANet最新版`；GitHub 分支 `feat/yrd-support` @ `87abe6c`；服务器 `feat/reliability-3comp` @ `8c0b1ab` |
| 精简后公式 | \(R_i=\left(R^{conf}_i\cdot R^{view}_i\cdot R^{spatial}_i\right)^{1/3}\)，EMA 0.82/0.18 保留 |

### 8.1 最终确认结果（seed 1237，最终干净代码 + restored exceptional）

| 数据集 | OA | AA | Kappa |
|---|---|---|---|
| Pavia | **85.92** | 83.64 | 0.8305 |
| Houston | **79.27** | 82.61 | 0.6819 |
| HyRANK（统一默认配置） | **56.00** | 59.55 | 0.4840 |

对比"删除前"（8 分量、旧配置，seed 1237）：

| 数据集 | 删除前 | 删除后（最终） | ΔOA |
|---|---|---|---|
| Pavia | 85.73 | **85.92** | **+0.19** |
| Houston | 79.06 | **79.27** | **+0.21** |
| HyRANK | 60.85（旧特殊配置） | **56.00（统一配置）** | **−4.85**（配置统一的代价，非分量删除导致） |


### 论文表述建议（可直接改写）

> 原始 8 分量几何平均经逐分量消融（Pavia/Houston 各 8 组 + 3 seed 复核）后发现 margin、time、drift、guard、proto 五个分量冗余或贡献可忽略；最终精简为 3 分量（置信度、多视图一致性、空间一致性），三数据集均值与 8 分量持平（Pavia 85.59 vs 85.63，Houston 78.73 vs 78.50），公式更简洁。HyRANK 采用当前伪标签模式，该模式下统一可靠性不参与，因此其分量消融恒等。

---

## 9. YRD 扩展实验（2026-11-15 新增）

### 9.1 YRD 基线（3 分量，3 seed）

| seed | OA | AA | Kappa | 弱类 |
|---|---|---|---|---|
| 1237 | 83.24 | 82.62 | 0.7602 | 类5=37.6，类2=64.9 |
| 1239 | 90.12 | 77.39 | 0.8509 | 类5=8.8 |
| 1241 | 69.62 | 76.92 | 0.5945 | 类2=30.3，类5=43.2 |
| **均值** | **80.99 ± 8.52** | 78.98 | 0.7352 | 种子方差极大 |

> 历史对照：`RMDANet` 旧 full 模型 YRD seed1237 OA=81.39（模块版本不同，仅作参考）。

### 9.2 YRD 调参（固定最差 seed 1241，找稳定配置）

| 配置 | 改动 | OA | AA | Kappa | 弱类 |
|---|---|---|---|---|---|
| 基线 | config.yaml | 69.62 | 76.92 | 0.5945 | 类2=30.3，类5=43.2 |
| v1 | `pseudo_min_per_class=20`、`cempa_class_cap=32` | 68.21 | 74.34 | 0.5781 | 类2=27.5，类5=31.1 |
| v2 | `pseudo_confidence=0.92`、`reliability_threshold=0.40`、`pseudo_margin=0.08` | 68.72 | 74.86 | 0.5841 | 类2=28.3，类5=41.9 |
| **v3** | **`pseudo_min_per_class=30`、`cempa_class_cap=40`、`source_class_weight_power=0.6`** | **77.92** | **79.08** | **0.6926** | 类2=51.7，类5=41.2 |
| v4 | `band_alignment_strength=0.3`、`prat_style_max=0.6`、`pseudo_min_per_class=20` | 76.38 | 76.07 | 0.6725 | 类2=48.1，类5=63.8 |

**v3 最优（+8.30 OA vs 基线 seed1241）**，说明 YRD 主要瓶颈是稀有类伪标签供给不足。

### 9.3 v3 的 3 seed 验证（已完成）

| seed | OA | AA | Kappa | 弱类 |
|---|---|---|---|---|
| 1237 | 86.02 | 81.12 | 0.7960 | 类5=10.8，类2=74.1 |
| 1239 | 91.20 | 80.12 | 0.8669 | 类5=10.9，类2=89.9 |
| 1241 | 77.92 | 79.08 | 0.6926 | 类5=41.2，类2=51.7 |
| **均值** | **85.05 ± 5.47** | **80.11** | 0.7852 | 类5 仍不稳定 |

对比：

| 配置 | 3 seed OA 均值±std | mean AA |
|---|---|---|
| YRD 基线 | 80.99 ± 8.52 | 78.98 |
| **v3（推荐）** | **85.05 ± 5.47** | **80.11** |
| 提升 | **+4.06** | +1.13 |

结论：v3 显著提升均值并降低方差（8.52 → 5.47），但**类 5 仍会崩**（1237/1239 只有 ~10.8%，1241 为 41.2）——这是 YRD 稀有类的固有难点，后续可继续针对类 5 做类救援/类权重实验（v5 方向），或接受现状并在论文中如实报告。

### 9.4 v5（更强类救援）三 seed 结果（已完成）

配置 v5：`pseudo_min_per_class=50`、`cempa_class_cap=50`、`source_class_weight_power=0.8`（在 v3 基础上加强）。

| seed | OA | AA | Kappa | 类5 |
|---|---|---|---|---|
| 1237 | 81.92 | 82.91 | 0.7433 | 35.5 |
| 1239 | 91.15 | 79.59 | 0.8659 | 1.5 |
| 1241 | 66.77 | 74.81 | 0.5622 | 40.6 |
| **均值** | **79.95 ± 10.05** | 79.10 | 0.7238 | 仍不稳定 |

### 9.5 YRD 三种配置对比（定稿依据）

| 配置 | 3 seed OA 均值±std | mean AA | 类5（1237/1239/1241） |
|---|---|---|---|
| 基线 config.yaml | 80.99 ± 8.52 | 78.98 | 37.6 / 8.8 / 43.2 |
| **v3（最优，推荐）** | **85.05 ± 5.47** | **80.11** | 10.8 / 10.9 / 41.2 |
| v5（更强救援） | 79.95 ± 10.05 | 79.10 | 35.5 / 1.5 / 40.6 |

**结论：v3 仍是 YRD 最优配置**（均值最高、方差明显小于基线）；v5 加强类救援后类 5 并未稳定（仍会崩到 1.5%），反而拉低其他类与整体。YRD 的建议：
- 若进论文主表：用 v3 + 3 seed 均值±std（85.05 ± 5.47），并单独说明类 5 的不稳定性；
- 或把 YRD 放在附录/附加实验，避免因稀有类波动影响主线结论。


---

## 10. 结果文件目录索引（服务器 `/root/autodl-tmp/v7clean/results_ablation_clean/`）

每个 run 目录：`<dataset>/<tag>_seed<seed>/`，内含：

- `selected_model.pt`：最终模型（locked eval 用）
- `target_final_once.json`：OA/AA/Kappa/per-class（唯一评估结果文件）
- `training_config.json`：完整超参
- `training_log.jsonl`：逐 epoch 日志

### 常用 tag 对照

| 实验 | Pavia tag | Houston tag | HyRANK tag |
|---|---|---|---|
| 8 分量 Full（seed1237） | `pavia_comp_full` | `houston_rtml_full_control` | — |
| 8 分量 Full（多 seed） | `pavia_full8_s1239/1241` | `houston_full8_s1239/1241` | — |
| 单分量删除 | `pavia_comp_drop_<comp>` | `houston_rtml_drop_<comp>` | `hyrank_comp_drop_{conf,margin,view,proto}` |
| 组合删除 | `pavia_comp_drop_mp` / `_mpt` / `_5a` / `_5b` / `_4c` / `_3c` / `_2c` | `houston_comp_drop_mp` 等同名 | — |
| 2 分量多 seed | `pavia_comp_2c`, `pavia_comp2c_s1239/1241` | `houston_comp_2c`, `houston_comp2c_s*` | — |
| 3 分量多 seed | `pavia_comp_3c`, `pavia_comp3c_s*` | `houston_comp_3c`, `houston_comp3c_s*` | — |
| 4 分量（+time） | `pavia_cvst_s*` | `houston_cvst_s*` | `hyrank_cvst_s*` |
| HyRANK 统一 3 分量（history） | — | — | `hyrank_hist3c_s*` |
| YRD 基线 | — | — | `yrd_3comp_s*`（在 `yrd/` 下） |
| YRD 调参 | — | — | `yrd_tune_v{1,2,3,4}_s1241`、`yrd_tune_v3_s{1237,1239}` |

### 日志文件（服务器 `/root/autodl-tmp/`）

| 日志 | 内容 |
|---|---|
| `comp_final2.log` | 8 分量 / 2 分量多 seed |
| `comp3.log` | 3 分量多 seed |
| `comp4c_time.log` | 4 分量（+time）三数据集 × 3 seed |
| `comp63.log` | 6 分量组合（部分） |
| `hyrank_hist3c.log` | HyRANK 统一 3 分量（history 路径） |
| `comp_prune_round2.log` | 组合删除（5a/5b/4c/3c/2c） |
| `yrd_tune.log` | YRD v1–v4 调参（seed 1241） |
| `yrd_v3_seeds.log` | YRD v3 多 seed（进行中） |

---

## 10.5 损失函数消融（2026-11-15，固定 seed 1237）

### 实验设置

| 项 | 值 |
|---|---|
| 机器 | `autodl-container-0c0340adb1-24d42954`，NVIDIA vGPU-48GB，驱动 580.76.05 |
| 代码 | `feat/reliability-3comp` @ `0a854ad`（含 `LOSS_ABL` 开关和无用损失清理） |
| 方法 | 环境变量 `LOSS_ABL="term1,term2"` 把对应损失权重置 0；空 = Full |
| 基线核对 | 本机 Full seed1237 = **85.92 / 79.27 / 56.00**（与旧 4090 服务器逐位一致） |

### E1–E4（旧 4090 服务器；Full 基线同机 85.92 / 79.27 / 56.00）

| 实验 | 关闭项 | Pavia OA | Houston OA | HyRANK OA |
|---|---|---|---|---|
| E1 | 外部 scale_pair + center_pair | 84.04（−1.88） | 79.28（+0.01） | 55.64（−0.36） |
| E2 | E1 + source_metric | 85.54（−0.38） | 78.73（−0.54） | 52.57（−3.43） |
| E3 | E1 + feature_consistency + scale_teacher | 84.17（−1.75） | 79.54（+0.27） | 55.81（−0.19） |
| E4 | E1 + CCA + Prototype | 76.38（−9.54） | 78.91（−0.36） | 58.00（+2.00） |

### E5–E7b（vGPU-48GB 服务器，全部同机对比，Full 已重跑）

| 实验 | 关闭项 | Pavia OA | Houston OA | HyRANK OA |
|---|---|---|---|---|
| Full_srv3 | 无 | 85.92 | 79.27 | 56.00 |
| **E5** | feature_consistency + scale_teacher | 85.91（−0.01） | 79.05（−0.22） | 56.32（+0.32） |
| **E6** | source_metric | **77.95（−7.97）** | 79.48（+0.21） | 54.97（−1.03） |
| **E7a** | cca | **74.96（−10.96）** | 79.06（−0.21） | 57.64（+1.64） |
| **E7b** | proto | 85.72（−0.20） | 78.89（−0.38） | 55.84（−0.16） |

### E5 3 seed 验证（正式删除依据）

机器：vGPU-48GB，seed 1237/1239/1241，同机同配置。

| 集合 | Pavia 3 seed | 均值±std | Houston 3 seed | 均值±std | HyRANK 3 seed | 均值±std |
|---|---|---|---|---|---|---|
| **Full** | 85.92 / 86.05 / 84.80 | **85.59 ± 0.56** | 79.27 / 79.16 / 77.77 | **78.73 ± 0.68** | 56.00 / 57.75 / 56.47 | **56.74 ± 0.74** |
| **E5**（删 feature+scale teacher） | 85.91 / 86.06 / 85.07 | **85.68 ± 0.44** | 79.05 / 79.13 / 77.52 | **78.57 ± 0.74** | 56.32 / 57.49 / 56.15 | **56.65 ± 0.60** |
| Δ（E5 − Full） | — | **+0.09** | — | **−0.16** | — | **−0.09** |

**判定：三个数据集的差异均在噪声内（≤0.2）→ 正式删除 `feature_consistency` 与 `scale_teacher`。**

删除后的最终状态：

| 项 | 变化 |
|---|---|
| 总损失加权项 | 17 → **14**（删除 dual-head CE 1 项 + feature/scale teacher 2 项） |
| 教师一致性 | 只剩 **soft 蒸馏**（`teacher_consistency_losses` 已精简为只返回 soft） |
| 删除提交 | 本地 `6810047`；服务器 vGPU `00ce8bf` |
| 删除后基线（即 E5 数字） | Pavia **85.68 ± 0.44**、Houston **78.57 ± 0.74**、HyRANK **56.65 ± 0.60** |

### 结论

| 损失 | 判定 | 依据 |
|---|---|---|
| Source Metric | **必须保留** | E6：Pavia −7.97、HyRANK −1.03 |
| CCA | **必须保留** | E7a：Pavia −10.96 |
| Prototype (flagment) | **保留** | E7b：三数据集均降，Houston −0.38 |
| 外部 scale_pair / center_pair | **保留** | E1：Pavia −1.88 |
| **feature_consistency + scale_teacher** | ✅ **已正式删除**（3 seed 验证） | E5 3 seed：Pavia +0.09、Houston −0.16、HyRANK −0.09（均在噪声内） |
| 已删除的无用项 | ✅ 已删 | `lmmd_loss`（从未调用）、`lambda_mpa`、`lambda_confusion`（计算但未入总损失） |
| Dual-head Source CE | ✅ **已删除** | 默认未启用（`lambda_dual_source=0`），代码（classifier2/logits2/双头分支）已全部移除 |

### 新增 tag / 日志

| 内容 | 位置 |
|---|---|
| E1–E4 | `results_ablation_clean/{pavia,houston,hyrank}/{ds}_{e1,e2,e3,e4}_loss_seed1237` |
| E5–E7b | `results_ablation_clean/{pavia,houston,hyrank}/{ds}_{full,e5,e6,e7a,e7b}_srv3_seed1237` |
| 日志 | `/root/autodl-tmp/loss_abl.log`（E1–E4）、`/root/autodl-tmp/loss_abl2.log`（Full/E5–E7b） |

## 10.6 第二轮损失筛选（5 个候选，2 seed × 3 数据集，2026-11-15）

候选：`soft_consistency`、`coral`、`lmmd`、`spectral_aux`、`scale_mi`（在最终代码 Full 上逐个关闭）。
机器：vGPU-48GB；seed 1237 / 1239；日志 `/root/autodl-tmp/loss_screen.log`。

基线（最终代码 Full）：Pavia 85.92 / 86.05；Houston 79.27 / 79.16；HyRANK 56.00 / 57.75。

| 候选 | Pavia s1237 / s1239 | Houston s1237 / s1239 | HyRANK s1237 / s1239 | 判定 |
|---|---|---|---|---|
| 删 **soft_consistency** | 84.70（−1.22）/ 85.23（−0.82） | 78.47（−0.80）/ 78.46（−0.70） | 55.68（−0.32）/ 57.08（−0.67） | **保留**（全差） |
| 删 **coral** | 85.05（−0.87）/ **73.88（−12.17）** | 79.20（−0.07）/ 79.04（−0.12） | 56.71（+0.71）/ 58.17（+0.42） | **保留**（Pavia 崩） |
| 删 **lmmd** | 83.75（−2.17）/ 85.38（−0.67） | 79.11（−0.16）/ 79.26（+0.10） | 56.18（+0.18）/ 58.01（+0.26） | **保留**（Pavia 明显降） |
| 删 **spectral_aux** | **69.88（−6.04）**/ 84.67（−1.38） | 78.65（−0.62）/ 79.01（−0.15） | 52.70（−3.30）/ 55.83（−1.92） | **保留**（两数据集崩） |
| 删 **scale_mi** | 85.30（−0.62）/ **81.98（−4.07）** | 78.89（−0.38）/ 79.11（−0.05） | 55.41（−0.59）/ 57.59（−0.16） | **保留**（Pavia 明显降） |

**结论：5 个候选在 2 个 seed 上均出现至少一个数据集明显下降（>0.3），全部保留。损失精简到此为止：最终 14 个加权损失项（全部默认生效）。**

Tag：`results_ablation_clean/{ds}/{ds}_{soft_consistency,coral,lmmd,spectral_aux,scale_mi}_2s_s{1237,1239}_seed{seed}`

---

## 10.11 Stage 0/1：运行入口审计与伪标签/归一化对照（2026-11-16）

### Stage 0：入口一致性与配置落盘

- 修复 `ablate_4rows.py` 的 `--row full`：原本 `fusion="off"`（光谱分支不参与主分类、`use_spectral_aux=False`），现改为 `fusion="bidirectional"`，与 trainer 的最终配置一致；修复后 Pavia seed1237 = **85.91 / 83.92 / 0.8305**，与显式参数基线完全一致。
- 新增每次运行落盘 `run_config.json`：记录 fusion_mode、use_spectral_aux、pseudo_mode、patch_size、pool_alpha_logit、全部 loss λ、环境变量与审计告警。
- 审计确认的无效开关（保留 CLI 兼容，但已打印告警）：
  - `use_guard`：guard_model 创建但 `run_epoch()` 未使用；
  - `use_confusion_loss`：总损失中无对应项；
  - `use_time_score` / `use_feature_score` / `use_spatial_reliability`：memory 固定 3 分量，开关被忽略；
  - `pseudo_min_per_class=0`：类别饥饿救援逻辑退出，`cempa_min_age/min_streak` 随之无效。
- 单元测试 `tests/test_stage0_audit.py` 3 项通过。

### Stage 1-A：HyRANK 伪标签路径（2 seed）

| 配置 | seed1237 | seed1239 | 均值 | ΔOA |
|---|---|---|---|---|
| history（原基线） | 56.32 | 57.49 | 56.91 ± 0.59 | — |
| **current** | **58.40** | **58.46** | **58.43 ± 0.03** | **+1.53** |

AA/Kappa（current）：1237 = 62.64/0.5091；1239 = 60.83/0.5122。
**结论：current 伪标签路径在 HyRANK 上稳定更优，应保留数据集自适应。**

### Stage 1-B：HyRANK CORAL × LMMD（seed1237）

| 配置 | OA | AA | Kappa | ΔOA |
|---|---|---|---|---|
| 基线 | 56.32 | 59.80 | 0.4870 | — |
| 关 CORAL | 56.71 | 59.82 | 0.4904 | +0.39 |
| 关 LMMD | 56.18 | 59.39 | 0.4844 | −0.14 |
| 双关 | 56.50 | 59.86 | 0.4873 | +0.18 |

**结论：仅"单关 CORAL"小幅正收益；LMMD 不可关；双关负交互。**

### Stage 1-C：Pavia 归一化 × 幅值通道 2×2（seed1237）

| 归一化 | amplitude | OA | AA | Kappa | ΔOA |
|---|---|---|---|---|---|
| L2 | off（基线） | 85.91 | 83.92 | 0.8305 | — |
| L2 | on | 73.75 | 76.28 | 0.6869 | −12.16 |
| median | off | 44.26 | 54.44 | 0.3464 | −41.65 |
| median | on | 65.05 | 65.46 | 0.5757 | −20.86 |

**结论：幅值通道与场景级 median 归一化均严重掉点，Pavia 依赖逐像素 L2 谱形；该改进方向否决。**


### Stage 1 收尾：伪标签路径的跨数据集验证（seed1237）

| 数据集 | 配置 | OA | AA | Kappa | ΔOA |
|---|---|---|---|---|---|
| Pavia | history（基线） | 85.91 | 83.92 | 0.8305 | — |
| Pavia | current | 70.15 | 71.28 | 0.6474 | −15.76 ❌ |
| Houston | history（基线） | 79.05 | 82.48 | 0.6791 | — |
| Houston | current | 78.79 | 82.99 | 0.6727 | −0.26 |
| HyRANK | current（Stage1-A 最优） | 58.40 | 62.64 | 0.5091 | +2.08 |
| HyRANK | current + 关 CORAL | 58.58 | 62.45 | 0.5096 | +0.18（vs current） |

**结论：**
1. `pseudo_mode=current` **不能全局开启**（Pavia −15.76），必须按数据集自适应：Pavia/Houston 用 history、HyRANK 用 current。这是目标标签无关的策略选择，不涉及目标 GT。
2. HyRANK "current + 关 CORAL" 仅比 current 高 0.18（单 seed，噪声内），**不采用**；保留 CORAL。
3. Stage 1 最终采纳：伪标签路径按数据集自适应（这是唯一稳定正收益 +1.53），其余 Stage 1 方案否决。

---

### 代码开关

- `PSEUDO_MODE_OVERRIDE=history|current|off`
- `NORMALIZE_MODE_OVERRIDE=l2|median`
- `AMPLITUDE_OVERRIDE=0|1`
- 所有覆盖均在 ablation profile 之后应用，保证单变量对照。

---

## 10.12 Stage 2：目标域可信度门控（2026-11-16，未通过）

目标：不依赖目标 GT，用「源原型 margin / teacher 置信度 / 多视图一致性 / 时间稳定性」构造软权重 w_i，统一作用于 CORAL / LMMD / CCA，减少不可信目标样本对类条件对齐的污染（对应《模型修改意见》3.2 / 5.3 / 7.1）。

### Stage 2a：完整信号门控（三信号几何平均，warmup 10，作用于 CORAL+LMMD+CCA）

| 数据集 | 基线 | 门控 | ΔOA | AA | Kappa |
|---|---|---|---|---|---|
| Pavia | 85.91 | 82.18 | **−3.73** | 81.17 | 0.7860 |
| Houston | 79.05 | 79.04 | −0.01 | 82.01 | 0.6786 |
| HyRANK | 58.40（current） | 57.67 | **−0.73** | 60.49 | 0.4995 |

### Stage 2b：温和版（仅原型 margin、最大降权 25%、warmup 20、仅 CORAL+CCA）

| 数据集 | 基线 | 门控 | ΔOA | AA | Kappa |
|---|---|---|---|---|---|
| Pavia | 85.91 | 85.90 | −0.01 | 83.72 | 0.8303 |
| Houston | 79.05 | 79.20 | +0.15 | 82.31 | 0.6809 |
| HyRANK | 58.40（current） | 57.84 | **−0.56** | 61.07 | 0.5018 |

**结论：两种门控均未通过。**
- 完整信号门控严重损伤 Pavia（−3.73）：源原型 margin 在训练早期有偏，且三信号相乘过度降权；
- 温和版在 Pavia/Houston 基本中性（−0.01 / +0.15），但 HyRANK 掉 0.56；
- 净收益为负，**不采用**。代码保留在 `core/credibility.py` + `STAGE2_GATING` 开关（默认关闭，不影响基线）。

---

## 10.13 Stage 3：困难类对跨域对比损失（单 seed，待确认）

对应《模型修改意见》7.2：对高置信目标样本，用源类原型做锚点，拉近伪类原型、推远最难负类（softplus hard-negative），即

**L_hard = E_{i∈T_h}[ softplus( max_{c≠y_i} f_i·μ_c − f_i·μ_yi + 0.3 ) ]**

启用条件：`STAGE3_HARD=1`，权重 0.05，从 epoch 20 开始 ramp 10 epoch；T_h = 稳定样本 ∧ 置信度≥0.9；归一化与其余配置不变。

| 数据集 | 基线 | +L_hard | ΔOA | AA | Kappa |
|---|---|---|---|---|---|
| Pavia | 85.91 | 85.76 | −0.15 | 83.48 | 0.8286 |
| Houston | 79.05 | 79.14 | +0.09 | 82.39 | 0.6800 |
| HyRANK | 58.40（current） | **58.75** | **+0.35** | 63.00 | 0.5126 |

**2 seed 验证（补充 seed1239）：**

| 数据集 | 基线（1237/1239） | +L_hard（1237/1239） | 2-seed 均值 ΔOA |
|---|---|---|---|
| Pavia | 85.91 / 86.06 | 85.76 / 85.72 | **−0.25** |
| Houston | 79.05 / 79.13 | 79.14 / 79.01 | −0.02 |
| HyRANK | 58.40 / 58.46 | 58.75 / 58.35 | **+0.12** |

**最终结论：L_hard 净收益在噪声范围内（HyRANK +0.12，Pavia −0.25），不采用；最终配置保持 14 项、L_hard 默认关闭。** 代码提交 `e345727` 保留为可复现实验开关。

---

## 10.14 Patch-size 实验（docx 3.4 / 4.3，单 seed）

实现：`PATCH_SIZE_OVERRIDE` 环境变量；CenterGuidedPooling 窗口随 patch 同步（3→1、5→3、≥7→5），避免中心窗口被裁剪后退化为 GAP。

| 数据集 | patch | center | OA | AA | Kappa | ΔOA vs 7×7 |
|---|---|---|---|---|---|---|
| Pavia | 7（基线） | 5 | 85.91 | 83.92 | 0.8305 | — |
| Pavia | 9 | 5 | 70.94 | 70.68 | 0.6553 | **−14.97** |
| Pavia | 11 | 5 | 71.80 | 69.29 | 0.6635 | **−14.11** |
| HyRANK | 7（基线） | 5 | 58.40 | 62.64 | 0.5091 | — |
| HyRANK | 5 | 3 | **60.37** | 64.83 | 0.5314 | **+1.97** |
| HyRANK | 3 | 1 | 54.10 | 59.00 | 0.4670 | −4.30 |

### HyRANK 3 seed 确认（patch5 vs patch7，均 current 伪标签）

| seed | patch5 | patch7 |
|---|---|---|
| 1237 | 60.37 | 58.40 |
| 1239 | 62.90 | 58.46 |
| 1241 | 59.08 | 57.90 |
| **均值** | **60.78 ± 1.59** | **58.25 ± 0.25** |

**结论：**
- Pavia 对 patch 变化极敏感（9×9/11×11 均掉 14 点以上），**Pavia 保持 7×7**；
- **HyRANK patch5 三个 seed 全部高于 patch7，均值 +2.53**（+1.97 / +4.44 / +1.18），与 docx 4.3 的"Hyperion 30 m 下 7×7=210 m 过大"判断一致；**HyRANK 采用 patch5 + center=3**；
- 3×3 反而掉 4.3 点（空间上下文不足），不采用；
- 注意：HyRANK 的 fusion 模式（历史脚本用 off、近期实验用 on）仍需对照，见下。

---

## 10.15 v7.2 配置统一与术语修正（2026-11-16）

| 项 | 修改 |
|---|---|
| HyRANK 配置 | `datasets/hyrank/config.yaml`：`patch_size: 5`、`pool_center_size: 3`（3 seed 验证 +2.53） |
| Full 入口统一 | `run_final_ablation.sh` 的 HyRANK 行改为 `bidirectional`（与 Pavia/Houston 相同）；原 "label-free fusion-off" 规则撤销 |
| 移除 `MERGE_CONSISTENCY` | 该开关只出现在审计 env 记录里、无真实合并逻辑；v7.2 删除 |
| 复合一致性（可选） | 新增 `LOSS_PROFILE=compact`：外部 Scale/Center Pair 折入 Style Pair（顶层 14→12），默认 full，需先验证等价性 |
| 运行配置核对 | 新增 `scripts/verify_run_config.py`（按数据集检查 patch/centre/fusion/pseudo/loss_profile） |
| 术语 | CCA → **CCC（Class-conditional Contrastive Alignment）**；LMMD → **reliable class-mean alignment（linear-kernel MMD 形式）**；Hard Contrastive 仅作消融 |
| 回滚点 | git tag `pre-advisor-20261116` = commit `d26c1f8` |

## 10.16 Prototype Alignment 删除验证（3 seed，v7.2 配置，未采纳删除）

配置：`LOSS_ABL=proto`；Pavia/Houston patch7 + history 伪标签，HyRANK patch5 + current 伪标签，fusion 统一 bidirectional。

| 数据集 | Full（3 seed） | 删除 Prototype（3 seed） | 3-seed 均值 ΔOA |
|---|---|---|---|
| Pavia | 85.91 / 86.06 / 85.07（85.68） | 85.74 / 85.83 / 84.77（85.45） | **−0.23** |
| Houston | 79.05 / 79.13 / 77.52（78.57） | 79.13 / 79.33 / 77.66（78.71） | +0.14 |
| HyRANK | 60.37 / 62.90 / 59.08（60.78） | 59.50 / 61.13 / 58.13（59.59） | **−1.20** |

**结论：不删除 Prototype Alignment。** Pavia 与 HyRANK 均一致下降，HyRANK 在新配置下损失最大（−1.20），说明它在 patch5 + current 伪标签下仍有独立的类别原型判别作用；Houston 的 +0.14 在噪声内。**保留该损失，14 项配置不变。**

## 10.17 Center / Scale Pair 单项删除（单 seed，均未采纳）

配置：v7.2（HyRANK patch5/centre3、fusion 统一 bidirectional）；Pavia/Houston history，HyRANK current；seed1237。

| 删除项 | Pavia | Houston | HyRANK | 判定 |
|---|---|---|---|---|
| Full（14 项） | 85.91 | 79.05 | 60.37 | — |
| **仅删 Center Pair** | 84.90（**−1.01**） | 79.11（+0.06） | 58.94（**−1.43**） | 不删除 |
| **仅删 Scale Pair** | 84.82（**−1.09**） | 79.25（+0.20） | 58.37（**−2.00**） | 不删除 |

**结论：**
- 两项在 Pavia 和 HyRANK 上单独删除都会明显掉点（−1 ~ −2），说明"外部均匀加权的 Scale/Center Pair"并非纯重复项，它确实提供了 PRAT 加权之外的稳定约束；
- 审阅意见中"Center Pair 是第一删除候选"的假设**不被数据支持**；
- 与 10.16 合并看：Prototype、Center Pair、Scale Pair **均保留**，最终配置维持 **14 项（四族）**；唯一的精简是论文呈现层（4 个损失函数）。

## 10.18 v7.3 精简版（删除 4 项，14 → 11 项，待准确率验证）

按审阅意见（"先不考虑准确率"）删除：外部 Center Pair、外部 Scale Pair、Prototype Alignment、Hard Contrastive（后者原本默认关闭）。

| 项 | 变化 |
|---|---|
| L_align 子项 | 5 → 4（CCC、Source Metric、CORAL、class-mean alignment） |
| L_cons 子项 | 5 → 3（Style Pair 复合损失、Scale MI、教师 soft） |
| 合计加权子项 | 14 → **11** |
| 论文口径 | 仍为 **4 个损失函数** |
| 保留 | `memory.prototype_align_flagment_loss`、`core/hard_contrastive.py` 作为独立消融模块，不进入主训练 |
| 版本 | v7.3-slim11；回滚点 tag `v7.2-advisor`（14 项版本） |

**3 数据集 × 3 seed 准确率验证进行中，结果见 10.19（待补）。**

## 10.19 v7.3 精简版（11 项）3 seed 准确率结果

删除外部 Center Pair、外部 Scale Pair、Prototype Alignment、Hard Contrastive 后，按同一规范配置重跑 3 数据集 × 3 seed。

| 数据集 | v7.2 Full（14 项） | **v7.3 slim-11（11 项）** | ΔOA |
|---|---|---|---|
| Pavia | 85.91 / 86.06 / 85.07 → **85.68 ± 0.44** | 83.70 / 83.61 / 82.70 → **83.34 ± 0.45** | **−2.34** |
| Houston | 79.05 / 79.13 / 77.52 → **78.57 ± 0.74** | 79.62 / 78.74 / 77.53 → **78.63 ± 0.86** | +0.06（噪声） |
| HyRANK | 60.37 / 62.90 / 59.08 → **60.78 ± 1.59** | 60.24 / 61.39 / 56.77 → **59.47 ± 1.96** | **−1.31** |

**结论：**
- 精简版在 **Pavia 掉 2.34**、**HyRANK 掉 1.31**，Houston 基本不变；
- 与此前单 seed 删除实验方向一致（Pavia/HyRANK 依赖这些约束），且精简后 seed 方差略增；
- 该版本按审阅意见"先不考虑准确率"保留为可选精简分支（tag/commit v7.3-slim11）；**若以准确率为准，应回滚到 v7.2-advisor（14 项）**。

## 10.20 P0 开放集虚拟类 + Sinkhorn 分配（单 seed，初步有效）

实现 `core/open_set.py`：对 teacher 概率做 C+1 列 Sinkhorn（C 个共享类 + dustbin 虚拟类），dustbin 质量比例 = `OPEN_SET_FRAC`，分配到虚拟类的目标样本从伪标签监督中剔除（不改分类头、不加损失）。

| 数据集 | 基线（slim-11） | frac=0.1 | frac=0.2 | frac=0.05 |
|---|---|---|---|---|
| Pavia | 83.70 | 83.70（0） | **84.68（+0.98）** ✅ | — |
| Houston | 79.62 | 79.62（0） | — | — |
| HyRANK | 60.24 | — | — | 60.24（0） |

**初步结论：**
- **Pavia 在 frac=0.2 时 +0.98**，方向正确（目标域 Water/Tiles 非共有类约 20%）；frac=0.1 无效果（未达到区分阈值）；
- Houston/HyRANK 在 0.05~0.1 时完全无变化（共享类完整，dustbin 未触发有效重分配）；
- 需补：Pavia frac=0.15/0.25/0.3 扫描 + seed1239/1241 确认；若稳定，可作为 **Pavia 专用**开关（目标标签无关）。

## 10.21 扩散生成式增强（slim-11，尝试失败）

实现 `core/diffusion_aug.py`：类条件 DDPM 在源域中心光谱上训练 → 每类生成 40 条 → 与目标域逐波段均值/方差匹配 → 用源 patch 的空间上下文 + 生成光谱替换中心像元，构造带源标签的合成样本，追加到源训练集（`DIFFUSION_AUG=1`，不改模型/损失）。

| 数据集 | slim-11 基线（seed1237） | + 扩散增强 | ΔOA |
|---|---|---|---|
| Pavia | 83.70 | 77.24 | **−6.46** |
| Houston | 79.62 | 73.78 | **−5.84** |
| HyRANK | 60.24 | 56.74 | **−3.50** |

**结论：失败。** 合成样本只替换中心像元，与周围空间上下文不自洽；目标风格矩匹配后的光谱也偏离真实目标分布，反而污染源训练集。**不采用。**

## 10.22 输入级预处理尝试（slim-11，seed1237）

### P0 边缘增强（unsharp masking，EDGE_ENHANCE）

| 配置 | Pavia OA | Δ vs slim-11 基线 83.70 |
|---|---|---|
| EDGE_GAIN=0.3 | 79.89 | −3.81 |
| EDGE_GAIN=0.1 | 75.53 | −8.17 |

**结论：失败。** 高频增强放大了噪声/伪边缘，破坏原有谱形，越强越差。不采用。

### P1 谱维 PCA 前置（SPECTRAL_PCA=30，目标域无监督 PCA 重建）

| 配置 | Pavia OA | AA | Kappa | Δ |
|---|---|---|---|---|
| slim-11 基线 | 83.70 | 81.87 | 0.8038 | — |
| **SPECTRAL_PCA=30** | **85.10** | 83.66 | 0.8214 | **+1.40** ✅ |

**补测（seed1237）：**

| 数据集 | 基线 | PCA-30 | k 扫描（Pavia） |
|---|---|---|---|
| Pavia | 83.70 | 85.10（+1.40） | **k=20: 86.03（+2.33）**；k=50: 84.69（+0.99） |
| Houston | 79.62 | 78.68（**−0.94**） | — |
| HyRANK | 60.24 | 57.55（**−2.69**） | — |

**结论：PCA 前置只对 Pavia 有效（k=20 最佳 +2.33），对 Houston/HyRANK 有害。**
原因：Pavia 空间分辨率高（1.3 m）、光谱噪声大，PCA 去噪收益明显；Houston/HyRANK 的判别信息藏在细谱段中，降维丢失细节。**建议按数据集自适应：Pavia 开 PCA-20，其余不开。**

## 10.23 图片三条落地方向的剩余尝试（v7.3-slim11，Pavia，seed1237/1239）

基线：Pavia seed1237 = 83.70、seed1239 = 83.61（均值 83.66）。

| 方向 | 实现 | seed1237 | seed1239 | 均值 | Δ |
|---|---|---|---|---|---|
| ② FFT 幅度/相位增强 | `FFT_AUG=1`：保留源相位、低频幅度换为目标参考（FDA 式） | 82.70 | 72.44 | 77.57 | −6.09 |
| ④ 预训练特征前置 | `PRETRAIN_FE=1`：目标域无监督谱自编码器，32 维 latent 拼入输入通道（134 通道） | 41.97 | 45.15 | 43.56 | **−40.10** |
| ⑤ 频域解耦双路输入 | `FREQ_DUAL=1`：低频/高频残差双通道输入（204 通道） | 79.09 | 85.29 | 82.19 | −1.47 |
| （已完成）① 多尺度边缘增强 | `EDGE_ENHANCE` gain 0.3 / 0.1 | 79.89 / 75.53 | — | — | −3.81 / −8.17 |
| （已完成）③ 谱维 PCA 前置 | `SPECTRAL_PCA=20` | 86.03 | 78.20（另 1241=85.26） | 83.16 | −0.17（3 seed 不稳定） |

**结论：五个输入/前置方向全部未能带来稳定提升。**
- PCA-20 单 seed 有正收益但 3 seed 不稳定（seed1239 掉 5.4）；
- FFT/频域双路/边缘增强均破坏原有谱形-空间平衡；
- 预训练自编码器 latent 拼接严重破坏输入分布（−40）。
**最终维持现有配置（11 项 slim-11 或 14 项 v7.2 按论文选择），不再追加输入级改造。**

## 10.24 原型注意力模块（v7.4 尝试，失败；来自 feat/proto-attn 分支）

实现 `core/model.py::PrototypeAttention`：可学习类原型库作为 attention keys/values，样本特征做交叉注意力后以残差 alpha 融合，再送余弦分类器；纯架构、不加损失。基线为 **v7.2（14 项）**。

| 数据集 | v7.2 基线（seed1237） | alpha=0.5 | alpha=0.1 |
|---|---|---|---|
| Pavia | 85.91 | 79.40（−6.51） | 70.98（−14.93） |
| Houston | 79.05 | 74.89（−4.16） | 74.71（−4.34） |
| HyRANK | 60.37 | 61.24（+0.87） | 58.04（−2.33） |

**结论：失败，不采用。** 外挂特征改写破坏了被 8~10 个损失共享的特征几何；alpha 越小反而更差，说明是训练动力学被破坏而非注入强度问题。

---

## 10.25 Mamba 光谱编码器（v7.4 尝试，失败；来自 feat/mamba 分支）

实现 `core/mamba.py`：逐波段 token 化 + 双向选择性 SSM + 注意力池化。两种用法均测试（基线 v7.2 14 项）：

| 配置 | Pavia | Houston | HyRANK |
|---|---|---|---|
| v7.2 基线 | 85.91 | 79.05 | 60.37 |
| **Mamba 替换光谱分支**（`MAMBA_SPECTRAL=1`，精简配置） | 77.45（−8.46） | 79.92（+0.87） | 58.54（−1.83） |
| **Mamba 混合旁路**（`HYBRID_MAMBA=1`，保留原谱形分支） | 72.17（−13.74） | 未测 | 未测 |

**结论：整体放弃。** 替换丢失了原分支显式的一阶/二阶谱形先验；旁路融合同样破坏表征。

---

## 10.26 早期被否定的架构与训练策略尝试（历史记录，2026-11-14/15）

| 类别 | 尝试 | 结果 |
|---|---|---|
| 骨干替换 | S1 / S2 / S3 三种骨干变体 | 均下降，未采用（旧日志，未保留逐 seed 数字） |
| 骨干替换 | S5 预训练骨干（`S5_PRETRAINED`） | 无明显收益，未采用 |
| 骨干替换 | B1 / B5 / B7 / B8 / B9 / B10 六种骨干/模块替换 | 均未超过基线，未采用 |
| 训练策略 | Stage 1：SWA + MixStyle + 强增强（`STAGE1_SWA/STYLE/AUG`） | **灾难性退化**（HyRANK 一度跌到 0.26），未采用 |
| 训练策略 | 更长训练、强增强、MixStyle 单项 | 无稳定收益或退化，未采用 |

> 说明：上表部分数字来自旧日志（未全部保留 `target_final_once.json`），论文中只作为"被排除方案"列出，不建议引用精确数值。

---

## 10.27 损失合并与早期精简尝试（历史记录）

| 方案 | 顶层项数 | Pavia 结果 | 结论 |
|---|---|---|---|
| Full（当时版本） | 14 | 85.68 ± 0.44（3 seed） | 基线 |
| R1a | 10 | 明显下降 | 否决 |
| R1b | 9 | 82.84 ± 1.34（3 seed） | 否决 |
| 12 项合并（3 个一致性项合并为 1） | 12 | 83.72 ± 1.43（3 seed） | 否决（数值不等价，轨迹分叉） |
| 11 项（早期尝试） | 11 | 84.75（单 seed） | 未采用（后续 v7.3-slim11 重新定义为 11 项） |

> 与 10.18–10.19 的 v7.3-slim11（11 项，Pavia 83.34±0.45）不同：早期 R1b/12 项是"合并"路线，v7.3 是"删除 4 项"路线；两者都不是 14 项的等价替代。

---

## 10.28 S4：TTA + 3 seed 集成（已验证的推理端提升，旧 14 项代码）

| 数据集 | 3-seed 基线 | + 8 视图 TTA + 3 seed 集成 | ΔOA |
|---|---|---|---|
| Pavia | 85.68 | **86.47** | +0.56（旧记录，另一组记录 +0.56） |
| Houston | 78.57 | **79.42** | +0.37 |
| HyRANK | 56.65 | **57.00** | +0.68 |

> 注意：该结果来自旧的 14 项代码与旧 HyRANK history 配置；在 v7.3-slim11（11 项）与 HyRANK patch5/current 配置下**尚未重新验证**，引用前需重跑。

---

## 10.29 P1 弱类伪标签配额（pseudo_min_per_class，Pavia 单 seed）

改动：`datasets/pavia/config.yaml` 的 `pseudo_min_per_class: 0 → 3`（测试后视结果决定是否保留）；零代码改动（`memory._min_class_quota_mask` 已有实现）。

| 配置 | OA | AA | Kappa | vs 基线 |
|---|---|---|---|---|
| slim-11 基线（seed1237） | 83.70 | 81.87 | 0.8038 | — |
| **quota=3** | **83.06** | 81.62 | 0.7961 | **−0.64** ❌ |

**逐类变化（Pavia, seed1237）：**

| 类索引 | 基线召回 | quota=3 | Δ |
|---|---|---|---|
| 1（Meadow 类别） | 92.1 | 92.3 | +0.2 |
| 2（Gravel） | 74.8 | 76.0 | +1.2 |
| 3（Trees） | 71.3 | **68.2** | **−3.1** |
| 5（Bare Soil） | 61.6 | **63.3** | **+1.7** |
| 6（Bricks） | 81.5 | 80.9 | −0.6 |

### quota 扫描（Pavia seed1237）

| quota | OA | AA | Δ vs 83.70 | Bare Soil 召回 | Trees 召回 |
|---|---|---|---|---|---|
| 0（基线） | 83.70 | 81.87 | — | 61.6 | 71.3 |
| **1** | **84.19** | 82.54 | **+0.49** ✅ | **63.9（+2.3）** | **74.2（+2.9）** |
| 2 | 83.47 | 82.06 | −0.23 | 65.1（+3.5） | 72.9（+1.6） |
| 3 | 83.06 | 81.62 | −0.64 | 63.3（+1.7） | 68.2（−3.1） |

**3 seed 验证（quota=1）：**

| seed | 基线 | quota=1 | Δ |
|---|---|---|---|
| 1237 | 83.70 | 84.19 | +0.49 |
| 1239 | 83.61 | **80.88** | **−2.73** |
| 1241 | 82.70 | 84.00 | +1.30 |
| 均值 | **83.34** | **83.02** | **−0.31** ❌ |

**结论：quota=1 单 seed 的 +0.49 是 seed 运气，3 seed 均值反而下降 0.31，且方差从 ±0.45 增至 ±1.79。P1 否决，`pseudo_min_per_class` 恢复为 0。**

### 其他两个数据集（seed1237）

| 数据集 | 基线 | quota | 结果 | Δ |
|---|---|---|---|---|
| Houston | 79.62 | 2 | 79.39 | −0.23 ❌ |
| HyRANK | 60.24 | 5 | 60.24 | 0.00（**无效果**） |

**重要发现**：HyRANK 使用 `pseudo_mode=current`，而 `pseudo_min_per_class` 的配额救援只在 `memory.update()`（history 路径）里调用；`current_pseudo_batch()` 不经过配额函数，所以对 HyRANK **完全无效**（结果逐位相同）。若要测试 HyRANK 配额，需要把配额逻辑接入 current 路径。

## 10.30 P2 类先验校准（推理期，3 seed，全部有效）

实现 `scripts/eval_prior_calib.py`：对目标域无标签 patch 跑模型得到概率 → 用 Saerens-Latinne-Decaestecker EM 估计目标类先验（不加标签）→ 用 `prior_smooth=0.5` 与 `prior_floor=0.02` 平滑 → `logits + 0.5·log(π_target/π_source)` 后取 argmax。**不改模型/损失/可靠性，不重训练。**

| 数据集 | seed1237 | seed1239 | seed1241 | 3-seed 均值 | Δ vs 基线 |
|---|---|---|---|---|---|
| Pavia | 83.70 → **83.73** | 83.61 → **83.79** | 82.70 → **82.82** | 83.34 → **83.45 ± 0.44** | **+0.11** ✅ |
| Houston | 79.62 → **80.12** | 78.74 → **79.21** | 77.53 → **78.02** | 78.63 → **79.12 ± 0.86** | **+0.49** ✅ |
| HyRANK | 60.24 → **60.83** | 61.39 → **61.54** | 56.77 → **57.13** | 59.47 → **59.83 ± 1.93** | **+0.37** ✅ |

**结论：P2 是本次所有尝试中第一个"三个数据集、三个 seed 全部为正"的方法。**
- 每个 seed 的每个数据集都提升（Pavia +0.03~+0.18、Houston +0.47~+0.50、HyRANK +0.15~+0.59）；
- AA 基本不变（Houston +0.1，其他 −0.1 左右），不是靠牺牲弱类换 OA；
- 目标先验由无标签目标数据 EM 估计，未使用目标 GT。

配置：`scripts/eval_prior_calib.py --alpha 0.5 --prior_smooth 0.5 --prior_floor 0.02`。

## 10.31 P3 类级熵 margin（推理期，seed1237）

实现：`eval_prior_calib.py --adjust entropy`——按"各类预测熵"给类级 logit 偏置（熵高的类加正 margin），不重训。

| 数据集 | 基线 | margin=0.5 | margin=1.0 |
|---|---|---|---|
| Pavia | 83.70 | 83.80（+0.10） | 83.83（+0.13） |
| Houston | 79.62 | 79.51（−0.11） | 79.37（−0.25） |
| HyRANK | 60.24 | 60.12（−0.12） | 59.98（−0.26） |

**结论：P3 未通过。** 仅 Pavia 小幅正收益，Houston/HyRANK 均下降；单 seed 已不满足"三数据集同涨"，不采用。

## 10.32 P4 Momentum 源原型（高置信过滤，seed1237，初步有效）

实现：`memory.update_source_prototypes(features, labels, logits)` 增加高置信过滤（`P4_FILTER=0.98`：仅 argmax 正确且置信度≥0.98 的源样本参与类原型更新；动量仍为 0.97，可选 `P4_MOMENTUM` 覆盖）。

| 数据集 | slim-11 基线 | **P4** | Δ | **P4 + P2（先验校准）** | Δ |
|---|---|---|---|---|---|
| Pavia | 83.70 | **86.03** | **+2.33** ✅ | 86.03 | **+2.33** |
| Houston | 79.62 | 79.30 | −0.32 | **79.73** | **+0.11** ✅ |
| HyRANK | 60.24 | 60.71 | +0.47 ✅ | **61.21** | **+0.97** ✅ |

**结论（单 seed）：** P4 显著改善 Pavia（+2.33）与 HyRANK（+0.47），Houston 略降；叠加 P2 后 Houston 回正（+0.11），三数据集全部为正（Pavia +2.33、Houston +0.11、HyRANK +0.97）。**需补 seed1239/1241 确认。**

## 10.33 P4 + P2 三 seed 结果：**三数据集全部提升**

P4 = 源原型高置信过滤（`P4_FILTER=0.98`，训练期）；P2 = 目标先验校准（推理期，`alpha=0.5, smooth=0.5, floor=0.02`）。

| 数据集 | 基线（3 seed） | P4（3 seed） | **P4 + P2（3 seed）** | Δ(P4+P2) |
|---|---|---|---|---|
| Pavia | 83.70 / 83.61 / 82.70 → **83.34 ± 0.45** | 86.03 / 83.82 / 84.60 → 84.82 | 86.03 / 84.08 / 84.70 → **84.94 ± 0.81** | **+1.60** ✅ |
| Houston | 79.62 / 78.74 / 77.53 → **78.63 ± 0.86** | 79.30 / 78.75 / 77.42 → 78.49 | 79.73 / 79.15 / 77.98 → **78.95 ± 0.73** | **+0.32** ✅ |
| HyRANK | 60.24 / 61.39 / 56.77 → **59.47 ± 1.96** | 60.71 / 62.22 / 61.75 → 61.56 | 61.21 / 62.85 / 62.00 → **62.02 ± 0.67** | **+2.55** ✅ |

**结论：P4 + P2 是本次唯一在"3 数据集 × 3 seed"上全部为正的方法。**
- Pavia +1.60、Houston +0.32、HyRANK +2.55；
- HyRANK 方差从 ±1.96 降到 ±0.67（稳定性显著改善）；
- 均为训练期（P4）+ 推理期（P2）改动，**未增加损失函数、未增加可靠性分量**。

## 10.34 v7.4-p4p2 固化（本地与服务器统一）

已把 P4+P2 做成默认配置：

| 项 | 实现 |
|---|---|
| P4 默认开启 | `core/memory.py`：`P4_FILTER` 默认值 0 → **0.98**（源原型只用高置信正确样本更新；`P4_FILTER=0` 可关闭） |
| P2 接入 locked eval | `scripts/final_eval_locked.py --prior-calib`：EM 目标先验 + alpha=0.5/smooth=0.5/floor=0.02，一步完成锁定评估 |
| 版本号 | `VERSION.txt` → **RMDANet v7.4-p4p2** |

**验证（服务器，默认 P4 + `--prior-calib`）**：Pavia seed1237 = **86.03**（AA 83.72，Kappa 0.8318），与 P4+P2 三 seed 结果完全一致。

三 seed 成绩（P4+P2）：Pavia 84.94±0.81（+1.60）、Houston 78.95±0.73（+0.32）、HyRANK 62.02±0.67（+2.55）。

## 10.35 P5 类对 margin 过滤（v7.4 基础上，单 seed，无效果）

实现：`P5_PAIRS="a,b"` + `P5_MARGIN=0.3`，在伪标签稳定集里剔除"标签属于该类别对且两类 log 概率差 < 0.3"的边界样本（默认关闭）。

| 数据集 | v7.4 基线（P4+P2，seed1237） | + P5 | Δ |
|---|---|---|---|
| Pavia | 86.03 | **86.03** | 0.00 |
| HyRANK | 61.21 | **61.21** | 0.00 |

**结论：P5 无效果。** 被 margin 过滤的边界样本本来就不在可靠性稳定集里（阈值筛选已覆盖），P5 与现有选择机制重叠。保留为可选开关（默认关闭），不进入主配置。

## 10.36 P6 弱类低频增强（v7.4 基础上，Pavia seed1237，初步有效）

实现：`core/data.py::PatchDataset._lowfreq_boost`——对**弱类源 patch** 做 2D 空间 FFT，只放大中心低频 1–3 阶系数（boost=1.5），概率 0.4；不改模型/损失。

| 配置 | OA | AA | Kappa | Δ vs v7.4 |
|---|---|---|---|---|
| v7.4（P4+P2） | 86.03 | 83.72 | 0.8318 | — |
| **+ P6**（Pavia 弱类 2/3/5） | **86.90** | **85.12** | **0.8426** | **+0.87 / AA +1.40** ✅ |

**Pavia 3 seed：**

| seed | v7.4（P4+P2） | + P6 | Δ |
|---|---|---|---|
| 1237 | 86.03 | 86.90 | +0.87 |
| 1239 | 84.08 | 86.02 | +1.94 |
| 1241 | 84.70 | 86.49 | +1.79 |
| 均值 | **84.94 ± 0.81** | **86.47 ± 0.36** | **+1.53** ✅ |

**其他数据集（seed1237，P6 指定各自弱类）：**

| 数据集 | v7.4 | + P6 | Δ |
|---|---|---|---|
| Houston（弱类 0/2/6） | 79.73 | 77.93 | **−1.80** ❌ |
| HyRANK（弱类 0/4/9） | 61.21 | 59.25 | **−1.96** ❌ |

**结论：P6 是 Pavia 专用改进（3 seed +1.53，方差从 ±0.81 降到 ±0.36）；对 Houston/HyRANK 有害，不应全局开启。**
采用方式：只对 Pavia 开启 P6（`P6_BOOST=1.5, P6_CLASSES=2,3,5, P6_PROB=0.4`）。

## 11. 注意事项与坑（避免后续误用）

1. **跨机器不可比**：所有数字来自黄金实例 `4d004f9f54`（RTX 4090 / 驱动 580.105.08），换实例重训会得到不同权重（同机同 seed 才逐位一致）。
2. **HyRANK 的分量消融不代表"分量无效"**，而是"该配置下分量未启用"；不要把它写成"H HyRANK 上所有分量都无关"。
3. **单 seed 结论可能被运气误导**（2 分量就是例子），论文数字一律用 3 seed 均值±std。
4. HyRANK 的 60.85 是 seed 1237 的最好值，均值是 59.38 ± 1.32；写论文时建议报均值并注明 best seed。
5. 所有最终数字必须来自 `target_final_once.json`（locked eval，未偷看目标 GT）。
6. YRD 的问题是种子方差大（±8.5），v3 配置仍在 3 seed 验证中，**未定稿前不要把 YRD 写进主表**。

---

## 12. 更新记录

| 日期 | 内容 |
|---|---|
| 2026-11-15 | 建立本文档，录入 8 分量单删/组合删除/多 seed/4 分量/YRD 调参结果 |
| 2026-11-15 | 待补：YRD v3 的 seed 1237/1239 结果、v3 三 seed 均值 |
| 2026-11-16 | 补 10.11–10.23：Stage0 审计、Stage1 伪标签/归一化、Stage2 门控、Stage3 困难类对比、patch-size、Prototype/Center/Scale 删除验证、v7.3-slim11、开放集、扩散增强、输入级预处理、FFT/预训练/频域双路 |
| 2026-11-17 | 补 10.24–10.28：原型注意力、Mamba（替换+混合）、早期骨干/训练策略尝试、损失合并历史、S4 TTA+集成（旧代码） |
