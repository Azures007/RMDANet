# 实验记录（每次修改后必须追加）

> 规则：每次修改代码后，记录「修改内容 → 运行配置 → 三数据集 3-seed 结果（OA mean±std，逐 seed）→ 结论」。
> 基线（纯 TAPDA，3 seed）：Houston 78.93、HyRANK 48.80、Pavia 91.14、YRD 94.56。
> 当前方法（SSTI+RATC，3 seed）：Houston 80.23、HyRANK 49.23、Pavia 92.15、YRD 92.44。

| # | 日期 | 修改内容 | 运行配置 | Houston | HyRANK | Pavia | YRD | 结论 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-08 | RGS 独立选样分支（modules/rgs_selector.py，`--rgs_selector`），与 PATS 解耦 | `--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector independent_joint` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | - | - | - | - | - | - | - | - |

## 待测配置清单（RGS v2）

| 模式 | 命令参数 | 说明 |
|---|---|---|
| legacy | `--rgs_selector legacy`（默认） | 旧 RGS（PATS 选样），对照 |
| independent_conf | `--rgs_selector independent_conf` | 全 batch 按教师置信度 |
| independent_agree | `--rgs_selector independent_agree` | 置信度 + Teacher/F2 一致 |
| independent_proto | `--rgs_selector independent_proto` | 置信度 + 源原型支持 |
| independent_joint | `--rgs_selector independent_joint` | 置信度 + 一致 + 原型 |

统一公共开关（全部创新点打开）：`--ssti --ratc --reliab_ce_weight 0.05`

## 记录模板

```
### YYYY-MM-DD 修改 N：<简述>
- 改动文件：
- 运行配置：
- 结果：
  - Houston：xx.xx / xx.xx / xx.xx → mean ± std
  - HyRANK：...
  - Pavia：...
- 对比 SSTI+RATC（80.23 / 49.23 / 92.15）：
- 结论：
```
