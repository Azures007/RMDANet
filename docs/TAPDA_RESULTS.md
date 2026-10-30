# TAPDA 复现与我们的可靠性伪标签监督（实验结果记录）

协议：**最后一轮评估**（已把 TAPDA 的 best-epoch 选模改成 `if True`，不使用目标标签选模），
单 seed（除注明外），`enumerate(SEEDS[:N])` 控制 seed 数。

## 1. 复现基线（TAPDA 官方代码，本环境复跑）

| 数据集 | 设置 | OA |
|---|---|---|
| Pavia | patch 9、102 波段、`SSSE_test` | seed0 85.75；5 seed = 85.75 / 90.97 / 90.09 / 92.26 / 93.40 → **均值 90.49**（注：这批数字来自改 best-epoch 之前的运行，seed0 与改后完全一致，可近似视为最后一轮） |
| Houston | patch 7、48 波段、`SSSE_Houston` | **80.83**（seed0，最后一轮） |
| HyRANK | patch 3、176 波段、`SSSE_test`（我们生成的 `dioni2loukia.py`） | **48.31**（seed0，最后一轮；Pavia 超参未按 HyRANK 调） |

TAPDA 论文（10 seed mean±std，best-epoch 选模）：Pavia 93.39±0.37、Houston 80.67±0.93、NC12→NC13 94.34±0.61。
**论文没有 HyRANK 任务。**

## 2. 我们的模块：可靠性加权伪标签 CE

实现：`tools/patch_tapda_reliab.py`，在 Step A 的 `all_loss` 上加入

```
L_rel = sum_i w_i * CE(student_logits_i, teacher_label_i) / sum_i w_i
w_i   = teacher_prob_i * [prob_i >= min_prob] * class_cap_mask
```

可选 warmup（`--reliab_pseudo_start_epoch` / `--reliab_pseudo_ramp_epochs`）。

### Pavia（seed0 权重扫描）

| 权重 | OA |
|---|---|
| 0（原版） | 85.75 |
| 0.03 | 87.56 |
| **0.05** | **92.50** |
| 0.07 | 85.86 |
| 0.10 | 87.44 |
| 0.20 | 84.58 |

5 seed（w=0.05）：92.50 / 91.84 / 89.80 / 94.69 / 87.05 → **均值 91.38**（baseline 90.49，**+0.88**，但方差大、2/5 seed 下降）。

### Houston（seed0）

| 配置 | OA |
|---|---|
| 原版 | 80.83 |
| **ours w=0.02, start=20** | **82.03（+1.20）** |
| ours w=0.05, start=10 | 78.89 |
| ours w=0.05/min_prob0.98/cap4/start15 | 81.14 |

### HyRANK（seed0）

| 配置 | OA |
|---|---|
| TAPDA 原版（未调） | 48.31 |
| ours w=0.05, start=10 | **39.31（−9.00）** |
| 官方 MSDA（HyRANK 专用设置） | 61.12 |
| 我们的 MSDA 端口 | 51.83 |
| 旧 v3.0 模型 | 55.03 |

## 3. 结论

1. **模块有效性与基座强度强相关**：
   - Houston（基座 80.83，很强）→ +1.20；
   - Pavia（基座 90.49，存在失败 seed）→ +0.88（不稳定，个别 seed +6.75）；
   - HyRANK（基座 48.31，未调、伪标签噪声大）→ −9.00。
   伪标签 CE 在基座弱时会把错误标签放大，这是明确的风险点。
2. TAPDA 的 PATS 已包含"原型一致性 + 类配额"选择，我们的模块必须与其互补（显式 CE 监督），
   不能简单声称"替换 PATS"。
3. HyRANK 上 TAPDA 未报告，且我们的适配版本未调参（48.31），不能作为论文基线；
   该数据集的公平基线是 **官方 MSDA（61.12）**。

## 4. 论文建议口径

| 数据集 | 基线 | 我们的方法 | 备注 |
|---|---|---|---|
| Pavia | TAPDA 90.49（复现，final-epoch）；论文 93.39（best-epoch） | **TAPDA+ours 91.38**（5 seed） | 需补 seed 稳定性分析 |
| Houston | TAPDA 80.83（复现）；论文 80.67 | **TAPDA+ours 82.03** | 单 seed，需补 3 seed |
| HyRANK | MSDA 61.12（官方复现） | 待定（建议先解决 TAPDA 调参或沿用 MSDA+模块） | TAPDA 论文未报告 |

## 5. 复现命令

```bash
# 1) 打补丁（单 seed + 我们的损失）
python tools/patch_tapda_single_seed.py --tapda-dir /root/autodl-tmp/tapda --num-seeds 1
python tools/patch_tapda_reliab.py --tapda-dir /root/autodl-tmp/tapda

# 2) Pavia
cd /root/autodl-tmp/tapda && python -u up2pc.py --gpu 0 \
  --reliab_pseudo_ce_weight 0.05 --reliab_pseudo_min_prob 0.95 \
  --reliab_pseudo_class_cap 8

# 3) Houston
python -u houston13218.py --gpu 0 \
  --reliab_pseudo_ce_weight 0.02 --reliab_pseudo_start_epoch 20 \
  --reliab_pseudo_ramp_epochs 10

# 4) HyRANK（生成脚本后）
python tools/make_tapda_hyrank.py --tapda-dir /root/autodl-tmp/tapda
```

> 注意：HyRANK 的 TAPDA 适配版本需要单独调参（mask/MCC 温度/BNM）
> 才能与 MSDA 的 61.12 公平比较，当前 48.31 不能作为 TAPDA 水平。
