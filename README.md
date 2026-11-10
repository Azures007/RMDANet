# RMDANet v7.0 干净版（tag v3.0，移除 CNT / SSF）

- 基础版本：`c0783a7`（tag `v3.0`，`protocol v3.0: literature-aligned ...`，2026-10-29）
- 本包在 v3.0 基础上**删除 CNT 与 SSF 全部代码**，只保留可复现基线的核心训练逻辑。

## 一、删除内容

| 删除项 | 说明 |
|---|---|
| `core/semantic_safety.py` | CNT（Class-Null Transport）投影/有界平移/教师 margin 门控、SafetyFeedback 逐类安全上限 |
| `core/class_null_transport.py` | CNT 兼容层 |
| `core/v71_controls.py` 中 `classwise_cnt_shift`、`source_teacher_backtrack` | 类条件 CNT 与教师回退 |
| `core/trainer.py` 中 CNT/SSF 全部代码 | `--ssf_mode C0..C4`、`--lambda_cnt`、`--cnt_*`、`--ssf_*` 参数；CNT 增强分支、SSF 安全门控、逐类反馈、CNT 损失项、相关日志字段 |
| `tests/test_ssf.py`、`tests/test_v71.py` 中 CNT 用例 | —— |
| `scripts/run_v7.py`、`scripts/run_three_v7.sh`、`scripts/eval_three_v7.sh` | v7.0 CNT/SSF 启动器 |
| `README_V7.md`、`README_V71.md`、`CHANGELOG_V7.md`、`CHANGELOG_V71.md`、`AUDIT_25_ITEMS_V71.md`、`VERSION_NOTES.md` 等 | CNT/SSF 相关历史文档 |
| `_reference/`、`core/memory.py.v5.1_fixed` | 未被引用的参考代码与备份文件 |

## 二、保留内容

- `core/`：`model.py`（双可靠性编码器）、`trainer.py`、`memory.py`（跨 epoch 可靠性记忆 RTML）、`prat.py`（RFTM/PRAT）、`data.py`（数据与归一化）
- `core/v71_controls.py`：`reliability_lmmd`、`maturity_snapshot`、`independent_spectral_view`、`js_agreement`（v6.0 的可靠性对齐与目标扰动探针仍需使用）
- `datasets/`：pavia / hyrank / houston / indianpines_salinas / shanghai_hangzhou / whu_hi 配置与入口
- `scripts/`：`run_v71.py`（CLI 仅接受 `C0`）、`run_unified.sh`、`final_eval_locked.py`、`report_metrics.py`、各数据集启动脚本

## 三、冻结基线（该版本产出，seed 1237）

| 数据集 | OA | AA | Kappa |
|---|---|---|---|
| Pavia | **85.55** | 82.74 | 0.8258 |
| HyRANK | **55.03** | 60.35 | 0.4709 |
| Houston | **76.72** | 82.28 | 0.6497 |

原始结果目录（服务器）：`results_v71/v3lite/<dataset>/C0_epochs50_seed1237/target_final_once.json`

## 四、协议（v3.0）

- 源域每类 180 标签，无源验证集；目标 GT 只在最终锁定评估打开一次
- 50 epochs × 40 steps，batch 128，eval batch 512
- AdamW + cosine（warmup 5），weight decay 4e-4，grad clip 5.0，AMP
- 逐像元 L2 归一化，patch 7×7，选最后一轮
- Pavia：lr 1e-3、PRAT 0.2、band align 0.1、目标池 30000
- HyRANK：lr 1e-3、PRAT 0.4、band align 0.1、目标池 15000
- Houston：lr 3e-4、PRAT 0.64、band align 0.45、目标池 30000

## 五、复现命令

```bash
# 数据路径
PAVIA_DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets
OTHER_DATA=/root/autodl-tmp/Datasets

# 三数据集一起（约 17 分钟）
bash scripts/run_unified.sh C0 1237 results_v71/repro

# 单数据集（以 Houston 为例）
python -u scripts/run_v71.py houston C0 1237 \
  --data-root "$OTHER_DATA" --epochs 50 --workers 4 \
  --out-root results_v71/repro_houston
RUN=results_v71/repro_houston/houston/C0_epochs50_seed1237
python -u scripts/final_eval_locked.py --checkpoint "$RUN/selected_model.pt" \
  --data-path "$OTHER_DATA/Houston" --output "$RUN/target_final_once.json" --confirm-final-eval
python -u scripts/report_metrics.py "$RUN/target_final_once.json"
```

Pavia 的 `--data-root/--data-path` 必须指向 `Test_1/.../Datasets`（Pavia_standard 版，Tiles=9248），否则 OA 掉到 10–13%。

## 六、注意

- 2026-11-09 复现核验：Houston 76.98（+0.26）、Pavia 83.62（−1.93，Pavia 对 RNG/worker 极敏感）；HyRANK 未复跑。
- 训练存在随机性，单 seed 允许 ±1~2 波动。
- 本包不含 CNT/SSF，`run_v71.py` 的 `mode` 参数只接受 `C0`（保留参数只为兼容旧命令）。
