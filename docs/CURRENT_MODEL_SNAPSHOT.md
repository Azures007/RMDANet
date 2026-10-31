# 当前模型版本快照（路线 1 改动的回退点）

## 版本标识

| 项 | 值 |
|---|---|
| 分支 | `tapda-base` |
| commit | `44bfd90` |
| tag | **`v1.0-m1c1quota`**（已推送 GitHub） |
| 内容 | TAPDA 基座 + M1（门控伪标签监督）+ C1（成熟度门控）+ 类配额 |

## 各数据集配置与结果（此版本）

| 数据集 | 命令关键参数 | 结果 |
|---|---|---|
| Pavia | `--reliab_ce_weight 0.05 --m1_maturity_gate` | **92.99 ± 0.84**（5 seed） |
| Houston | `--reliab_ce_weight 0.05 --target_class_quota_mode confusion_ema` | **81.13**（2 seed） |
| YRD | 纯基线（无开关） | **94.56**（3 seed）/ 95.09（2 seed） |
| Pavia 纯 TAPDA | 无参数 | 91.59 ± 2.03（5 seed） |

## 已知问题（路线 1 要解决的）

1. C1 在 Houston（−1.97）与 YRD（+0.09 中性）上不可用；
2. 类配额在 Pavia（−1.78）与 YRD（崩）上不可用；
3. M1 均值增益不显著（+0.09~+1.40），主要贡献是降低方差（2.03→0.52）。

## 回退方法（路线 1 失败时）

```bash
cd ~/autodl-tmp/RMDANet
git fetch https://gh-proxy.com/https://github.com/Azures007/RMDANet.git --tags
git checkout -B tapda-base v1.0-m1c1quota
# 验证
git log --oneline -1
```

或本地直接：

```bash
git checkout tapda-base
git reset --hard v1.0-m1c1quota
```

## 各数据集的运行命令（回退后复现）

```bash
cd ~/autodl-tmp/RMDANet/third_party/tapda
ln -sfn /root/autodl-tmp/Datasets/YRD_extract/YRD datasets/YRD

# Pavia（M1+C1）
python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight 0.05 --m1_maturity_gate

# Houston（M1+类配额）
python -u houston13218.py --gpu 0 --num_seeds 2 \
  --reliab_ce_weight 0.05 --target_class_quota_mode confusion_ema

# YRD（纯基线）
python -u nc122nc13.py --gpu 0 --num_seeds 3
```
