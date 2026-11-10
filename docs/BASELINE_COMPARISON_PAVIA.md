# Pavia 对比实验（UDA 基线 vs RMDANet）

协议：PaviaU → PaviaC，源域 **180 标签/类**，目标域训练零标签，目标 GT 只在最终评估打开一次；
训练 50 epochs（RMDANet 为 50×40 steps）；seed 1237；单卡 RTX 4090。

| Method | 来源 | OA | AA | Kappa |
|---|---|---|---|---|
| Source-only | 复现（同协议） | 63.50 | 69.00 | 0.5739 |
| DAN | 复现（同协议） | 67.61 | 74.10 | 0.6214 |
| DANN | 复现（同协议） | 67.46 | 74.48 | 0.6167 |
| JAN | 复现（同协议） | 70.53 | 76.08 | 0.6545 |
| MTLDA | 复现（官方代码缺失） | 69.21 | 75.81 | 0.6381 |
| **CACL** | **官方代码** | **79.11** | **82.30** | **0.7488** |
| UIPCL | 复现（官方实现为半监督设定） | 60.91 | 62.62 | 0.5329 |
| PCDM-UDA | 复现（官方求解器需 Gurobi license） | 61.75 | 66.89 | 0.5500 |
| VCMoE | 复现（官方代码缺失） | 71.04 | 76.22 | 0.6602 |
| **RMDANet（本文）** | 本文 | **85.73** | **83.38** | **0.8283** |

## 说明（论文脚注用）

1. **CACL** 使用作者官方代码（github.com/Ding-Kexin/IEEE_TCSVT_CACL），保留其原始设置
   （patch 5、min-max 归一化、lr 5e-4、batch 36），仅将源域采样改为 180/class、训练轮数改为 50。
2. **DAN / DANN / JAN / MTLDA / UIPCL / PCDM / VCMoE** 为在**完全相同的 UDA 协议**下的复现实现：
   共享轻量 HSI-CNN backbone（3 conv + GAP + FC），仅适配损失不同；使用保幅度 median 归一化。
   其中 MTLDA / VCMoE 无公开官方代码；UIPCL 官方仓库的训练集使用目标域标签（半监督），
   不适用于 UDA 比较；PCDM-UDA 官方实现依赖 Gurobi 求解器（受限 license 无法处理 Pavia 规模）。
3. **RMDANet** 使用逐像元 L2 归一化 + 残差 DRSE + 双向可靠性融合 + RFTM + RTML。
4. 所有方法共享同一数据切分与同一评估代码（OA/AA/Kappa，逐像素混淆矩阵）。

## 结论

- RMDANet 相对最强官方基线 CACL：**OA +6.62、AA +1.08、Kappa +0.0795**；
- 相对最强复现基线 VCMoE：**OA +14.69**；
- 复现基线的绝对数值低于其原文（backbone 与训练预算不同），论文中已注明，不与原文数字直接比较。
