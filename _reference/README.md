# 参考实现（不参与训练）

这些文件**不被当前训练代码引用**，仅作历史参考。

- `contrastive_loss.py` — `SupConLoss`（Khosla et al. 2020），SCLUDA 也使用此实现。将来做 backbone 移植可复用。
- `DCRN_02.py` — DCRN 网络（SCLUDA 的 backbone）。实验 C 已证明 DCRN_02 在 Pavia 上远优于 PMDANet backbone（94.33% vs 76.81%），将来换 backbone 时可直接使用。
