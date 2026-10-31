# 理论小节草稿（Proposition 与证明骨架）

> 目标：为"选择偏差 → 门控监督"提供可写进论文的形式化动机。当前为草稿，数值常数待实验后确定。

## 设定

目标域样本 $x_i^t$，真实标签 $y_i\in\{1..C\}$，类别先验 $\pi_c=\Pr(y=c)$。
教师给出校准概率 $p_i(c)=P_\theta(y=c\mid x_i^t)$，置信度 $p_i=\max_c p_i(c)$，伪标签 $\hat y_i=\arg\max_c p_i(c)$。
PATS 选样：对每个类取分数最高的 $k$ 个样本，得集合 $S$。

## 引理 1（类条件选择偏差）

设类 $c$ 的第 $j$ 个样本分数为 $s_{c,j}$，选样阈值为 $q_c^{(k)}$（每类第 $k$ 大分数）。
则 $|\{i\in S: \hat y_i=c\}| = k$ 恒成立，但**选样后的类质量比例**为
$$\rho_c=\frac{k}{\sum_{c'}k}=\frac{1}{C},$$
而目标域真实质量比例为 $\pi_c$。因此选样集合的类分布与目标分布之差为
$$\left|\rho_c-\pi_c\right|=\left|\tfrac{1}{C}-\pi_c\right|,$$
当 $\pi_c\ll 1/C$ 时该差接近 $1/C$——**小类在选样集合中被系统性低估或高估**，取决于选样策略按类取 top-k 还是全局取 top-k。

> 注：PATS 为逐类 top-k，选样集合本身类均衡；但**进入监督的样本经全局置信度门控 $\tau$ 后**，小类因伪标签置信度低而被进一步剔除，形成"双重筛选偏差"。

## 引理 2（门控后的类覆盖衰减）

设类 $c$ 的置信度分布 $F_c(p)$，门控 $\mathcal{G}=\{p_i\ge\tau\}$。则类 $c$ 的覆盖率为
$$\kappa_c(\tau)=1-F_c(\tau).$$
若小类伪标签置信度受类先验影响（$F_c$ 随机占优于 $F_{c'}$ 当 $\pi_c>\pi_{c'}$），则 $\kappa_c(\tau)$ 随 $\pi_c$ 单调递增，即
$$\pi_c<\pi_{c'}\ \Rightarrow\ \kappa_c(\tau)\le\kappa_{c'}(\tau).$$
故全局门控使监督信号进一步向大类集中。

## Proposition（RGS 的去偏效果）

记目标域风险 $R(f)=\mathbb{E}_{x^t}\ell(f(x^t),y)$，置信集风险 $R_\tau(f)=\mathbb{E}[\ell(f(x),y)\mid p\ge\tau]$。
在教师校准假设 $p_i=P(y=\hat y_i\mid x_i)$ 下，RGS 的目标
$$\mathcal{L}_{\mathrm{RGS}}=\frac{\sum_{i\in S\cap\{p_i\ge\tau\}}p_i\,\ell(f(x_i),\hat y_i)}{\sum_{i\in S\cap\{p_i\ge\tau\}}p_i}$$
是 $R_\tau(f)$ 的加权经验估计，其偏差满足
$$\left|\mathbb{E}[\mathcal{L}_{\mathrm{RGS}}]-R_\tau(f)\right|\le \underbrace{\epsilon_{\mathrm{cal}}}_{\text{校准误差}}+\underbrace{\sqrt{\tfrac{\log(2/\delta)}{2n_\tau}}}_{\text{有限样本}},\quad n_\tau=|\mathcal{G}|.$$
并且总体风险可分解为
$$R(f)=\kappa(\tau)\,R_\tau(f)+\bigl(1-\kappa(\tau)\bigr)\,R_{\bar\tau}(f),$$
其中 $R_{\bar\tau}$ 为未过门控样本的风险，$1-\kappa(\tau)$ 即覆盖率损失。

因此：(i) 门控监督以覆盖率损失换取噪声更低的训练信号；(ii) **类自适应门控**（M2-C/M2-B）等价于令各类 $\kappa_c(\tau_c)$ 均衡，从而降低最坏类风险
$$\min_{\tau_c}\ \max_c\ \bigl[\kappa_c(\tau_c)R_{\tau_c}(f)+(1-\kappa_c(\tau_c))R_{\bar\tau_c}(f)\bigr].$$

## 证明骨架

1. 由校准假设，$\mathbb{E}[p_i\ell(f(x_i),\hat y_i)]=\Pr(p\ge\tau)\cdot R_\tau(f)$ 的一阶近似成立；
2. 用 Hoeffding 不等式控制加权经验均值与期望的偏差（权重有界于 $[0,1]$）；
3. 风险分解按指示函数 $\mathbb{1}\{p\ge\tau\}$ 展开，得覆盖率项；
4. 类自适应部分：对每类应用同一不等式，取最大值即得最坏类界。

## 待补

- 校准误差 $\epsilon_{\mathrm{cal}}$ 的经验估计（ECE，可用教师输出直接算）；
- 覆盖率 $\kappa(\tau)$ 的实测值（训练日志里已有 `selected` 比例）；
- 把 M2-S（软标签）解释为对 $\mathbb{E}[\ell]$ 的方差缩减版本。
