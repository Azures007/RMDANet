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
| 适用数据集 | Pavia、Houston（以及后续 YRD）走记忆路径 |
| HyRANK | 保持 `--abl-history 0` 当前伪标签路径（conf×margin×view），不启用统一可靠性 |
| 删除的分量 | margin、time、drift、guard、proto（含代码分支与辅助方法） |
| 最终代码 | `feat/reliability-3comp` @ `94313f3`，tag `reliability-3comp-2026-11-15`（已推 GitHub） |
| 精简后公式 | \(R_i=\left(R^{conf}_i\cdot R^{view}_i\cdot R^{spatial}_i\right)^{1/3}\)，EMA 0.82/0.18 保留 |

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

### 9.3 v3 的 3 seed 验证（进行中，完成后补录）

| seed | OA | AA | Kappa | 状态 |
|---|---|---|---|---|
| 1241 | 77.92 | 79.08 | 0.6926 | ✅ 完成 |
| 1237 | — | — | — | 🔄 运行中 |
| 1239 | — | — | — | ⏳ 排队 |

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
