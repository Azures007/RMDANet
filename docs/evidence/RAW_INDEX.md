# 证据原始数据索引

本目录汇总论文所需的原始实验数据。服务器上的原始日志路径一并列出，便于复核。

## 1. Houston 训练期诊断（核心证据）

文件：`RAW_HOUSTON_DIAG.md`（由 `report_houston.md` 抄录）

服务器原始文件：
- `/root/autodl-tmp/diag/report_houston.md`
- `/root/autodl-tmp/diag/houston/target_labels.npy`
- `/root/autodl-tmp/diag/houston/dump_seed1599_ep{20,35,50}.npz`

生成方式：
```bash
python -u houston13218.py --gpu 0 --num_seeds 1 \
  --reliab_ce_weight 0.05 --m1_maturity_gate \
  --m1_dump /root/autodl-tmp/diag/houston --m1_dump_epochs 20,35,50
python tools/analyze_rgs_dump.py --dataset houston \
  --dump-dir /root/autodl-tmp/diag/houston --out /root/autodl-tmp/diag/report_houston.md
```

## 2. 基线（TAPDA 10-seed）

服务器：`/root/autodl-tmp/runs_report/logs/TAPDA_10.log`
报告：`/root/autodl-tmp/runs_report/RESULTS_FULL.md`

seeds OA：80.83, 78.96, 77.01, 81.86, 79.68, 79.55, 81.84, 80.30, 80.03, 81.77
均值 ± std：**80.18 ± 1.52**；AA 76.42 ± 2.85；Kappa 68.81 ± 1.97

## 3. RGS / RGS-v2 / RCS 结果

| 报告 | 服务器路径 | 结论 |
|---|---|---|
| RGS 5-seed | `/root/autodl-tmp/rgs_final/RESULTS_FINAL.md` | +1.01，p=0.27 |
| RGS-only 3-seed | `/root/autodl-tmp/rgs_only/RESULTS_RGS_ONLY.md` | +0.07 |
| RGS-v2 3/5/10-seed | `/root/autodl-tmp/rgs_v2/RESULTS_V2.md`、`/root/autodl-tmp/rgs_v2b/RESULTS_V2B.md`、`/root/autodl-tmp/runs_report/RESULTS_FULL.md` | 10-seed +0.25，p=0.61 |
| 变体（被排除/类配额/逆先验） | `/root/autodl-tmp/rgs_variant/RESULTS_VARIANT.md` | 被排除 +1.20（1 seed），10-seed 归零 |
| RCS | `/root/autodl-tmp/runs_rcs/RESULTS_RCS_FAST.md` | 3-seed +1.00，不稳定 |

## 4. 关键数字速查

| 指标 | 值 |
|---|---|
| 每轮目标样本数 | 1248 |
| PATS 保留 / 排除 | 533 / 715（ep50） |
| 排除比例 | 57.29% |
| 保留集教师准确率（ep50） | 78.61% |
| 排除集教师准确率（ep50） | 79.72% |
| 教师置信度（ep50） | 0.9450 |
| 教师整体准确率（ep50） | 79.25% |
| RGS ∩ 排除集 | 0 |
| 源域教师正确率 | 93.2% |
| RCS 估计错误率 vs 真实错误率 | 0.68% vs 20.75% |
