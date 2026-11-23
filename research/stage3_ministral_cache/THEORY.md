# EM-FloatFormer 第 3 阶段（Section A）理论推导：冻结两窗缓存、第三窗留出预测与迟到导频的精确条件分析

> **基线继承**：本文严格衔接 `research/fourphase_matched-cache@504f3a440e737e290ba7e0fa62fd6667a04f5cde` 的两窗缓存规范，以及 `research/fourphase_memory_pilot` 的离散 $C_4$ 随机游走与连续圆周漂移框架。所有结论均可在 [verify_theory.py](verify_theory.py) 中通过精确有限状态枚举与加密圆周求积独立复算。

---

## 1. 生成模型、时间轴与严格信息可见性

设潜在四相发射符号与初始公共未知信道旋转满足独立同分布均匀先验：
$$X_1, X_2, G \stackrel{\text{i.i.d.}}{\sim} \operatorname{Unif}(\mathbb{Z}_4), \qquad s_u := i^u \in \{1, i, -1, -i\}\quad (u \in \mathbb{Z}_4).$$

我们显式分离四个观测窗口的时间坐标，其中**两个数据窗间隔 $h \ge 0$** 与**第二数据窗至第三留出窗间隔 $\ell \ge 0$** 分别独立控制，迟到已知导频位于可配置时刻 $t_p \ge 0$：
- 第 1 数据窗：$t_1 = 0$，发射 $X_1$，无噪接收相位 $s_{X_1 + G} e^{i\phi_1}$（令 $\phi_1 \equiv 0$ 吸收进未知 $G$ 与后续位移中，或等价地 $\phi(t)$ 表示相对 $t_1=0$ 的信道相位轨迹，$G$ 为 $t_1$ 时刻的四相基线参考；两种写法仅在于 $t_p < h$ 时从 $t_1$ 到 $t_p$ 的位移分布）。
- 第 2 数据窗：$t_2 = h$，发射 $X_2$，相对 $t_1$ 的信道相位增量记为 $\Phi_{12}$（离散模型取 $U \in \mathbb{Z}_4$，连续模型取 $\phi_2 \in \mathbb{R}$）。
- 第 3 留出窗：$t_3 = h + \ell$，在两种对照分支下改变**仅**发射符号关系：
  - **正例分支（`repeat_x1`）**：协议公开声明 $X_3 = X_1$（重发第一窗符号），取全新独立接收噪声 $N_3$；
  - **负对照分支（`independent_x3`）**：协议公开声明 $X_3 \sim \operatorname{Unif}(\mathbb{Z}_4)$，与 $(X_1, X_2, G, \text{漂移过程}, N_1, N_2, N_p)$ 完全独立，取全新独立接收噪声 $N_3$。
- 迟到已知导频窗：$t_p \ge 0$，发射已知单位导频 $X_p = 0$（对应 $s_0 = 1$），相对 $t_1=0$ 的信道相位增量记为 $\Phi_{1p}$（离散模型取 $\Delta_p \in \mathbb{Z}_4$，连续模型取 $\phi_p \in \mathbb{R}$）。

### 1.1 复基带观测与充分统计量

每个窗口 $j \in \{1, 2, 3, p\}$ 包含 $N = 3$ 个复样点（窗口总发射能量归一化为 $E_s = 1$，窗口信噪比 $\rho := E_s / N_0 = 10^{\text{SNR}_{\text{dB}}/10}$，基线 $2\text{ dB}$ 时 $\rho = 10^{0.2}$）：
$$Y_{j,m} = \frac{1}{\sqrt{N}}\, s_{X_j + G} e^{i\phi_j} + N_{j,m}, \qquad N_{j,m} \stackrel{\text{i.i.d.}}{\sim} \mathcal{CN}\!\left(0, \frac{1}{\rho}\right),\quad m \in \{0,1,2\}.$$

正交匹配和充分统计量为
$$T_j := \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} Y_{j,m} = s_{X_j + G} e^{i\phi_j} + \tilde N_j, \qquad \tilde N_j \sim \mathcal{CN}\!\left(0, \frac{1}{\rho}\right),$$
其实部与虚部独立同分布于 $\mathcal{N}(0, \sigma_n^2)$，其中 $\sigma_n = \frac{1}{\sqrt{2\rho}}$。因此对于任意给定无噪接收相量 $\mu = e^{i\theta} \in \mathbb{C}$，观察到 $Y_j$ 的对数似然仅依赖 $T_j$：
$$\log p(Y_j \mid \mu) = \text{const}(Y_j, \rho) + 2\rho\,\Re\!\bigl(T_j \overline{\mu}\bigr).$$

### 1.2 四个扇区边界与三类随机变量的严格区分

在整项研究中严格区分三类量：
1. **发射符号**：$X_3 \in \mathbb{Z}_4$（`repeat_x1` 时 $X_3 = X_1$，`independent_x3` 时 $X_3 \perp \mathcal{F}_{\text{pre}}$）；
2. **无噪接收相量／相位**：离散模型为 $U_3 = (X_3 + G_3) \bmod 4$，连续模型为 $\Theta_3 = \frac{\pi}{2}(X_3 + G) + \phi_3 \pmod{2\pi}$；
3. **含噪评分观测与四个标准扇区**：实际含噪匹配统计量 $T_3 \in \mathbb{C}$ 及其所在半开四分扇区 $K_3 := \operatorname{sector}(T_3) \in \{0,1,2,3\}$。

采用与基线 `research/matched_cache/learned.py` 完全一致的半开扇区划分（以四个星座点 $1, i, -1, -i$ 为中心，沿 $\pm 45^\circ$ 对角线分割复平面）。令 $u = \Re(z) + \Im(z)$、$v = \Im(z) - \Re(z)$：
$$\operatorname{sector}(z) := \begin{cases}
1\ (+90^\circ,\ \arg z \in [\frac{\pi}{4}, \frac{3\pi}{4})), & u > 0 \land v \ge 0,\\
2\ (180^\circ,\ \arg z \in [\frac{3\pi}{4}, \frac{5\pi}{4})), & u \le 0 \land v > 0,\\
3\ (270^\circ,\ \arg z \in [\frac{5\pi}{4}, \frac{7\pi}{4})), & u < 0 \land v \le 0,\\
0\ (0^\circ,\ \arg z \in [-\frac{\pi}{4}, \frac{\pi}{4})), & \text{其余（含 } u > 0 \land v < 0,\ (u=0 \land v=0)\text{）}.
\end{cases}$$
对所有非零点 $z \in \mathbb{C}\setminus\{0\}$（连续噪声下为概率 1 事件），严格满足四分之一圈等变性 $\operatorname{sector}(iz) = (\operatorname{sector}(z) + 1) \bmod 4$。

### 1.3 信息可见时序（Chronological Visibility）

1. **一次编码阶段（Encode-Once）**：仅输入两段数据窗 $(Y_1, Y_2)$，生成整数缓存 $Z = \operatorname{pack}(\operatorname{enc}(Y_1), \operatorname{enc}(Y_2)) \in \{0,\dots,2^B-1\}$（$B \in \{4, 8\}$）；随后对所有有限缓存系统丢弃 $(Y_1, Y_2)$。
2. **Q1（揭示前相对相位）**：仅可见 $(Z, h, \text{漂移元数据})$（完整 I/Q 参照可见 $(Y_1, Y_2, h, \text{漂移元数据})$），输出 $R = (X_2 - X_1)\bmod 4$ 的四分类后验 $q_1$。
3. **Q3（揭示第三窗前预测）**：在揭示 $Y_3$ 与任何导频前，**仅可见** $Z$ 与公开协议标签 $\mathcal{M} \in \{\texttt{repeat\_x1}, \texttt{independent\_x3}\}$、时间间隔 $(h, \ell)$ 与声明漂移参数。正例和负对照在同一题面下获得**同类型可见信息**（仅协议标签不同）。系统必须先将 $q_3(k) = \Pr(K_3 = k \mid Z, \mathcal{M})$ 及条件均值估计 $\hat\mu_3 = \mathbb{E}[T_3 \mid Z, \mathcal{M}]$ **写入逐例记录文件**，之后评分器才揭示实际含噪第三窗样本 $Y_3$，计算：
   - 扇区准确率 $\mathbb{I}[\arg\max_k q_3(k) = \operatorname{sector}(T_3)]$、对数损失 $-\log q_3(K_3)$、ECE；
   - 条件均值归一化均方误差 $\text{NMSE} = \frac{\sum |T_3 - \hat\mu_3|^2}{\sum |T_3|^2}$。
4. **迟到参考揭示与 Q2**：揭示包含 `acquisition_id` 与 `shares_channel` 标志的参考列表，先按唯一有效 `acquisition_id` 去重再更新第一符号 $X_1$ 的四分类预测 $q_2$。（另设未见问题顺序：先问 Q3、再问 Q1、再揭示参考后回答 Q2 与“揭示导频后的 Q3”。）

---

## 2. 离散 $C_4$ 相位转移模型下的精确有限求和

设单步增量服从对称四元信道 $P(\delta) = (1-d)\,\mathbb{I}[\delta=0] + \frac{d}{3}\mathbb{I}[\delta \neq 0]$，$d \in [0, \frac{3}{4}]$，单步可靠度 $k(d) := 1 - \frac{4}{3}d \in [-\frac{1}{3}, 1]$。独立增量在 $\mathbb{Z}_4$ 上的循环卷积满足**可靠度相乘**：
- $t_1 \to t_2$（间隔 $h$ 步）的总增量 $U$ 分布为 $p_U(u) = c_h \delta_0(u) + \frac{1-c_h}{4}$，其中 $c_h = k(d)^h$；
- $t_2 \to t_3$（间隔 $\ell$ 步）的独立增量 $V$ 分布为 $p_V(v) = c_\ell \delta_0(v) + \frac{1-c_\ell}{4}$，其中 $c_\ell = k(d)^\ell$；
- 故 $t_1 \to t_3$ 的合成增量 $W = U + V \pmod 4$ 具有精确两段分离形式
  $$p_W(w) = c_{h+\ell}\delta_0(w) + \frac{1-c_{h+\ell}}{4}, \qquad c_{h+\ell} = c_h c_\ell = k(d)^{h+\ell}.$$

由变量代换 $(X_1, X_2, G) \mapsto (U_1, U_2, G) = (X_1+G, X_2+G, G)$ 在 $\mathbb{Z}_4^3$ 上双射且保持均匀乘积分布，并结合增量与噪声同全部符号及 $G$ 独立，令单窗后验（对完整 I/Q 取精确解析似然 $a_j(u) \propto \exp(2\rho\,\Re(T_j\overline{s_u}))$，对冻结有限缓存 $Z=(z_1, z_2)$ 取静态查找表 $a_1 = \operatorname{table}[z_1],\ a_2 = \operatorname{table}[z_2]$）：

### 2.1 三道题在任意 $(h, \ell, t_p)$ 下的有限求和公式

1. **Q1（揭示前相对相位 $R = X_2 - X_1$）**：由于 $X_2 - X_1 = U_2 - U_1 - U \pmod 4$，精确边缘后验为有限三重求和（化简为二重卷积）：
   $$q_1(r) = \sum_{u_1=0}^3 \sum_{u_2=0}^3 a_1(u_1) a_2(u_2) p_U\bigl((u_2 - u_1 - r)\bmod 4\bigr) = c_h \sum_{u=0}^3 a_1(u) a_2\bigl((u+r)\bmod 4\bigr) + \frac{1-c_h}{4}.$$
2. **Q3（揭示 $Y_3$ 前对第三窗含噪扇区 $K_3$ 与条件均值 $\hat\mu_3$ 的预测）**：
   - 在正交旋转 $45^\circ$ 坐标系下，零噪离散点 $s_w$ 在独立双正交高斯噪声 $\mathcal{N}(0, 1/(2\rho))$ 下的四扇区混淆核为精确闭式
     $$M_\rho = \bigl((1-q)^2,\; q(1-q),\; q^2,\; q(1-q)\bigr), \qquad q(\rho) = \frac{1}{2}\operatorname{erfc}\!\left(\sqrt{\frac{\rho}{2}}\right).$$
   - **正例分支（`repeat_x1`，已知 $X_3 = X_1$）**：由于 $U_3 = X_1 + G + W = U_1 + W \pmod 4$，仅通过第一窗相位后验 $a_1$ 即可严格边缘化所有隐变量 $(X_1, X_2, G, U, V)$：
     $$p_{U_3 \mid Z}(w) = \sum_{u_1=0}^3 a_1(u_1)\, p_W\bigl((w - u_1)\bmod 4\bigr) = c_{h+\ell}\,a_1(w) + \frac{1 - c_{h+\ell}}{4},$$
     $$q_3^{\text{rep}}(k \mid Z) = \sum_{w=0}^3 p_{U_3 \mid Z}(w)\, M_\rho\bigl((k - w)\bmod 4\bigr) = \bigl(p_{U_3 \mid Z} * M_\rho\bigr)(k).$$
     对应的复数条件均值估计为
     $$\hat\mu_3^{\text{rep}}(Z) = \mathbb{E}[T_3 \mid Z, \texttt{repeat\_x1}] = \sum_{w=0}^3 p_{U_3 \mid Z}(w)\, s_w = c_{h+\ell}\sum_{u=0}^3 a_1(u)\, s_u.$$
     （注：在离散对称漂移与 $U_2 \perp U_1$ 的结构下，$a_2$ 不携带关于 $W = U+V$ 模 4 值的额外信息；而在连续漂移与有效时序下，下文第 3 节指出 $(h,\ell)$ 通过完整 $Y_2$ 可估计分数部分漂移！）
   - **负对照分支（`independent_x3`）**：由 $X_3 \sim \operatorname{Unif}(\mathbb{Z}_4)$ 独立于 $(U_1, G_3, N_1, N_2, N_3)$，$U_3 = X_3 + G_3 \sim \operatorname{Unif}(\mathbb{Z}_4)$，故
     $$q_3^{\text{ind}}(k \mid Z) \equiv \frac{1}{4}, \qquad \hat\mu_3^{\text{ind}}(Z) \equiv 0.$$
3. **Q2（揭示去重导频集合 $\mathcal{P}_{\text{dedup}}$ 后的第一符号 $X_1$）**：对所有满足 `shares_channel == True` 且按唯一 `acquisition_id` 去重后的有效导频（若各自位置分别为 $t_{p,m}$），令先导合并相位转移可靠度为 $c_{p} = k(d)^{|t_p - t_1|}$（单导频情景，或等价地在有向马氏链上条件求和 $G_1 = G$ 与 $G_2 = G+U$），当导频位于 $t_p = 0$ 或 $t_p \ge h$ 时（$t_p = h+\text{gap}$ 时 $U + V_p$ 与 $Y_2$ 是否有关？**重要洞察**：当 $0 < t_p < h$ 时，$t_p$ 位于第 1 窗与第 2 窗之间，此时 $U = \Delta_{0, t_p} + \Delta_{t_p, h}$ 被导频所在中间时刻劈裂！完整 I/Q 与完整贝叶斯马尔可夫求和均能精确计算任意 $(t_1=0, t_p, t_2=h)$ 次序下的 $p(G \mid Y_1, Y_2, P)$ —— 即对潜状态 $(u_1, u_2, g, \delta_a, \delta_b)$ 求有限和）。在本框架与 [analytic.py](analytic.py) 中，我们**直接使用按时间排序的精确隐马尔可夫有限求和**覆盖全部 $t_p$ 相对位置（$t_p=0$、$0<t_p<h$、$t_p=h$ 及 $t_p > h$），无任何中间近似！

### 2.2 相干极限、失相干极限与必然优于均匀先验的充分必要条件

**命题 1（有限缓存严格降低 Q3 期望对数损失的充要条件）**  
在随机变量生成模型下，均匀先验在 Q3 上的期望对数损失恒为 $\log 4 \approx 1.386294\text{ nats}$。对于采用精确匹配后验 $q_3^*(\cdot \mid Z, \mathcal{M})$ 的任意有限缓存编码器 $Z = \operatorname{enc}(Y_1, Y_2)$，其相对均匀先验的期望对数损失降低量严格等于条件互信息：
$$\Delta\mathrm{NLL}(Z) := \log 4 - \mathbb{E}\bigl[-\log q_3^*(K_3 \mid Z, \mathcal{M})\bigr] = I(K_3; Z \mid \mathcal{M}) = \mathbb{E}_Z\bigl[D_{\mathrm{KL}}\bigl(q_3^*(\cdot \mid Z, \mathcal{M}) \,\big\|\, \operatorname{Unif}_4\bigr)\bigr].$$
从而在离散 $C_4$ 转移模型下：
$$\Delta\mathrm{NLL}(Z) > 0 \iff (\mathcal{M} = \texttt{repeat\_x1}) \ \land\ (c_{h+\ell} \neq 0)\ \land\ (\rho > 0)\ \land\ \bigl(I(U_1; Z) > 0\bigr).$$
*证明要点*：对任意分布 $p$ 满足 $M_\rho * p$ 非均匀当且仅当 $\rho > 0$（从而 $M_\rho$ 傅里叶系数 $\widehat{M_\rho}(m) \neq 0$）且 $p$ 非均匀。由 $p_{U_3\mid Z} = c_{h+\ell}\, p_{U_1\mid Z} + \frac{1-c_{h+\ell}}{4}$，$\Pr_Z(p_{U_3\mid Z} \neq \operatorname{Unif}_4) > 0 \iff c_{h+\ell} \neq 0 \land \Pr_Z(p_{U_1\mid Z} \neq \operatorname{Unif}_4) > 0 \iff c_{h+\ell} \neq 0 \land I(U_1; Z) > 0$。若采用随机读出而非精确 $q_3^*$，只需使预测分布满足 $\mathbb{E}[D_{\mathrm{KL}}(\operatorname{Unif}_4 \| q_3)] < I(K_3; Z) + \dots$ 即当总散度 $\mathbb{E}_Z[D_{\mathrm{KL}}(q_3^*(\cdot\mid Z)\|\hat q_3(\cdot\mid Z))] < I(K_3; Z \mid \mathcal{M})$ 时严格优于 $\log 4$。$\square$

**两端极限与反例验证**：
- **完全相干极限（$h = \ell = 0$ 或 $d = 0 \implies c_{h+\ell} = 1$）**：即使得知精确无噪标签 $U_1$（Oracle 无噪上界），含噪观测 $K_3$ 仍受 $\rho$ 约束，精确 Oracle 上界为 $\text{Acc}_{\text{oracle}} = (1-q(\rho))^2$（2 dB 下 `80.28%`）、$\text{NLL}_{\text{oracle}} = 2 H_b(q(\rho))$（2 dB 下 `0.6677 nats`）；完整 I/Q 通过 `a_1` 可达 2 dB 下的缓存无关上界（40,000 次蒙特卡洛复算实测 `Acc = 66.19%`、`NLL = 0.8992 nats`，其中 `hand_b4` 达 `66.19% / 0.9627 nats / NMSE=0.6166`，`hand_b8` 达 `66.19% / 0.9190 nats / NMSE=0.5854`）。
- **完全失相干极限（$c_{h+\ell} \to 0$，即 $h+\ell \to \infty$ 且 $d \in (0, 3/4]$，或单步 $d = 3/4$）**：无论 $Z$ 或完整 I/Q 携带多少比特，$p_{U_3\mid Z} \to \operatorname{Unif}_4$，$q_3 \to (1/4, 1/4, 1/4, 1/4)$，$\hat\mu_3 \to 0$，期望 NLL 严格收敛至 $\log 4$，期望准确率收敛至 $1/4$。
- **仅存相对相位的 2-bit 缓存反例**：若缓存只保存 $Z = H(Y_1, Y_2) := \arg\max_r p_R(r)$（`fourphase_memory_pilot` 中的 2 比特相对缓存），由于联合变换 $(Y_1, Y_2) \mapsto (i Y_1, i Y_2)$ 以概率 1 保持 $H$ 不变同时将 $U_1 \mapsto U_1 + 1 \pmod 4$ 均匀循环移位，故 $H \perp U_1$，即 **$I(U_1; H) = 0$**。即使在 $h=\ell=0,\ \rho=100$ 几乎无噪的极端相干极限下，2-bit 相对缓存对 Q3 的期望准确率仍锁死在 **`25.00%`**、期望损失锁死在 **`log 4`**；而保留绝对方向位的 4-bit／8-bit 相位保持表示则严格打破 $\log 4$ 界！

---

## 3. 连续相位漂移模型与圆周数值求积

设连续轨迹相对于 $t_1 = 0$（$\phi_1 := 0$）满足一般高斯随机游走加线性频偏（CFO）与可选突变：
$$\phi_2 = \omega h + D_h, \quad D_h \sim \mathcal{N}(0, h\sigma_\phi^2), \qquad \phi_3 = \phi_2 + \omega \ell + D_\ell, \quad D_\ell \sim \mathcal{N}(0, \ell\sigma_\phi^2),$$
以及在位置 $t_p \ge 0$ 处的导频相位增量 $\phi_p$，其中 $(\phi_2, \phi_3, \phi_p)$ 具有由布朗运动增量协方差 $\operatorname{Cov}(B_s, B_t) = \sigma_\phi^2 \min(s, t)$ 确定的精确联合高斯协方差矩阵（当 $0 < t_p < h$ 时，$\phi_p$ 与 $\phi_2$ 正相关；当 $t_p = h+\text{gap}$ 时，两者共享前 $h$ 步增量）。

### 3.1 给定连续无噪接收相位 $\theta$ 的精确含噪四扇区分布

由于扇区划分对应将复平面按 $+45^\circ$ 对角线正交旋转得到坐标
$$(Z_+, Z_-) := \left(\frac{\Re(T_3) + \Im(T_3)}{\sqrt{2}},\ \frac{\Im(T_3) - \Re(T_3)}{\sqrt{2}}\right) \text{或等效旋转 }\ \tilde T_3 = e^{i\pi/4} T_3,$$
具体地，利用扇区 $k=0$ 对应 $\Re(\tilde T_3) \text{与} \Im(\tilde T_3)$ 同号的正区间（更准确地，以扇区中心 $s_0 = 1$ 为例，旋转 $+45^\circ$ 后 $1 \mapsto \frac{1+i}{\sqrt{2}}$，两条决策边界恰好为 $\Re(z e^{i\pi/4}) = 0$ 与 $\Im(z e^{i\pi/4}) = 0$ 对应直线 $\arg z = -\pi/4$ 与 $\arg z = +\pi/4$：令 $z = r e^{i\theta}$，投影到向量 $e^{-i\pi/4} \leftrightarrow \cos(\theta + \pi/4)$ 与 $e^{+i\pi/4} \leftrightarrow \sin(\theta + \pi/4)$）：
验证：
- 当 $\theta = 0$（正实轴）时，$\cos(\pi/4) = \sin(\pi/4) = +1/\sqrt{2}$，此时点同时处于 $u = \Re z + \Im z = \sqrt{2} \cos(\theta - \pi/4)$? 检验边界：
  - 记 $u(z) = \Re(z)+\Im(z) = \sqrt{2} r \sin(\theta + \pi/4)$（注意 $\cos(\theta - \pi/4) = \frac{\cos\theta+\sin\theta}{\sqrt{2}} = \sin(\theta+\pi/4)$），
  - 记 $v(z) = \Im(z)-\Re(z) = \sqrt{2} r \sin(\theta - \pi/4) = -\sqrt{2} r \cos(\theta + \pi/4)$！
  - 校验 $\theta = 0$：$u = +1 > 0,\ v = -1 < 0 \implies \text{sector}=0$；概率 $p_u = \Phi(\sqrt{2\rho}\sin(\theta+\pi/4))$、$p_v^- = \Pr(V < 0) = \Phi(\sqrt{2\rho}\cos(\theta+\pi/4))$。故 $k=0$ 对应 $u>0, v<0$ 概率为 $p_+ q_+$！
因此定义正交符号正概率（标准正态累积函数 $\Phi$）：
$$p_+(\theta) := \Phi\!\bigl(\sqrt{2\rho}\cos(\theta + \pi/4)\bigr) = \Pr(V < 0 \mid \theta),\qquad q_+(\theta) := \Phi\!\bigl(\sqrt{2\rho}\sin(\theta + \pi/4)\bigr) = \Pr(U > 0 \mid \theta),$$
则对任意连续无噪接收相位 $\theta \in [-\pi, \pi)$，实际含噪统计量 $T_3 = e^{i\theta} + \tilde N_3$ 落入四个扇区的**零积分近似精确闭式概率向量**为：
$$M_{\text{cont}}(\theta; \rho) = \Bigl(p_+(\theta) q_+(\theta),\ \bigl(1-p_+(\theta)\bigr) q_+(\theta),\ \bigl(1-p_+(\theta)\bigr)\bigl(1-q_+(\theta)\bigr),\ p_+(\theta)\bigl(1-q_+(\theta)\bigr)\Bigr).$$

### 3.2 完整 I/Q 与有限冻结缓存的圆周数值求积

- **冻结有限缓存 $Z = (z_1, z_2)$**：若读出仅保留窗口后验查表 $a_1 = \operatorname{table}[z_1]$，由于 $z_1$ 的单元中心不保留 $(\phi_2 \bmod \frac{\pi}{2})$ 的精细信息，直接对 $\phi_3 \sim \mathcal{N}(\omega(h+\ell), (h+\ell)\sigma_\phi^2)$ 在 $[-\pi, \pi)$ 上作 128 节点均匀圆周求积（包裹高斯尾和至 $\pm 10\text{ SD}$）：
  $$q_3^{\text{rep}}(k \mid Z) = \sum_{u=0}^3 a_1(u) \int_{-\pi}^{\pi} p_{\text{wrap}}(\phi_3)\, M_{\text{cont}}\!\left(\frac{\pi}{2}u + \phi_3;\, \rho\right)_k d\phi_3,$$
  $$\hat\mu_3^{\text{rep}}(Z) = \mathbb{E}[e^{i\phi_3}] \sum_{u=0}^3 a_1(u)\, s_u = \exp\!\left(i\omega(h+\ell) - \frac{1}{2}(h+\ell)\sigma_\phi^2\right) \sum_{u=0}^3 a_1(u)\, s_u.$$
- **完整 I/Q 条件解析参照（`full_iq_known_param`）**：与有限缓存不同，完整 I/Q 可以利用**未经量化的第 2 窗观测 $Y_2$** 来联合推断第 2 窗处相对于 $C_4$ 星座的**分数部分连续漂移角 $\phi_2$**，再经马尔可夫核 $p(\phi_3 \mid \phi_2) = \mathcal{N}(\phi_2 + \omega\ell, \ell\sigma_\phi^2)$ 前推至 $\phi_3$！
  - 注意对称性核心要点：$Y_2$ 关于 $X_2 + G$ 的边缘似然为
    $$L_2(\phi_2; T_2) = \frac{1}{4}\sum_{v=0}^3 \exp\!\left(2\rho\,\Re\!\bigl(T_2 \overline{s_v} e^{-i\phi_2}\bigr)\right),$$
    **该似然函数对 $T_2 \mapsto i T_2$ 严格周期不变**！因此完整 I/Q 既能超越离散表提取关于 $\phi_2 \bmod \frac{\pi}{2}$ 的全部连续相位轨迹信息，又严格保持在单转第二窗 $(0, 1)$ 下的 Q3 预测完全不变性。
  - 在 `analytic.py` 与 `verify_theory.py` 中，我们对 $\phi_2$ 和 $\phi_3$ 采用 $M = 128$ 节点正圆周求积，并运行 $M = 256$ 节点加密对照以验证收敛精度（$\max|\Delta| < 10^{-12}$）。

---

## 4. 迟到已知导频在揭示前的四扇区均匀性、连续复平面非均匀性及其揭示后增益机理

这是本阶段理论推导必须说透的核心机制：

1. **揭示前四扇区严格等概率（$1/4$）**：
   设 $G \sim \operatorname{Unif}(\mathbb{Z}_4)$ 与全部发射符号 $(X_1, X_2)$、全部信道相位漂移路径 $(\phi_2, \phi_3, \phi_p)$ 及接收噪声 $(N_1, N_2, N_3, N_p)$ 完全相互独立。考虑 $(X_1, X_2, G) \mapsto (U_1, U_2, G) = (X_1+G, X_2+G, G)$ 在 $\mathbb{Z}_4^3$ 上的模 4 可逆线性双射：对任意固定漂移与噪声实现，
   $$\Pr(U_1=u_1, U_2=u_2, G=g \mid \text{drift}, \text{noise}) = \frac{1}{64} = \Pr(U_1=u_1, U_2=u_2)\Pr(G=g).$$
   由此立即得到 $G$ 与包含 $(Y_1, Y_2)$ 在内的一切前序观测（乃至整个编码缓存 $Z$）严格统计独立：
   $$G \;\perp\!\!\!\perp\; \bigl(Y_1, Y_2, Z, \phi_p, N_p\bigr).$$
   因此对揭示前的任意候选观测，旋转算子 $T_p \mapsto i T_p$（对应 $G \mapsto G + 1 \bmod 4$）保持条件概率密度严格 $C_4$ 循环不变：$p(i t \mid Y_1, Y_2) = p(t \mid Y_1, Y_2)$。由于四个半开扇区在 $\mathbb{C}\setminus\{0\}$ 上互为精确 $+90^\circ$ 旋转像且边界为零测度集，对四扇区积分立刻得到：
   $$\Pr\bigl(\operatorname{sector}(T_p) = k \mid Y_1, Y_2\bigr) = \int_{S_k} p(t \mid Y_1, Y_2)\,dt = \int_{S_0} p(i^k t \mid Y_1, Y_2)\,dt = \int_{S_0} p(t \mid Y_1, Y_2)\,dt = \frac{1}{4},\quad \forall k \in \{0,1,2,3\}.$$
   这甚至对不丢弃任何比特的完整 I/Q 也同样严格成立。

2. **四扇区等概率 $\neq$ 复平面均匀分布（连续密度的四叶玫瑰结构）**：
   虽然四个宏观扇区积分严格等于 $1/4$，但条件密度 $p(t \mid Y_1, Y_2)$ 在整个连续复平面 $\mathbb{C}$ 上**既非均匀分布、亦非各向同性的圆对称分布**！确切地，记已知相干轨迹条件下的导频相位后验密度为 $p(\phi_p \mid Y_1, Y_2)$，则
   $$p(T_p = t \mid Y_1, Y_2) = \frac{1}{4}\sum_{g=0}^3 \int_{-\pi}^{\pi} p(\phi_p \mid Y_1, Y_2)\, \frac{\rho}{\pi}\exp\!\Bigl(-\rho\bigl|t - s_g e^{i\phi_p}\bigr|^2\Bigr)\, d\phi_p.$$
   这是一个**严格集中在单位圆附近、具有四重花瓣（4-lobed）结构与清晰径向聚集性**的非均匀密度函数：
   - 若无连续漂移（$\phi_p = 0$），它在 $\{1, i, -1, -i\}$ 四点处呈现高度局域峰；在任意扇区内部，峰值位置与半径分布高度不均；
   - 若存在有限连续漂移且已观测到完整 $(Y_1, Y_2)$（如 $0 < t_p < h$ 或 $h > 0$），$(Y_1, Y_2)$ 还能推知小数部分漂移角 $\phi_p \bmod \frac{\pi}{2}$，从而使这四片花瓣在各自象限内部同步旋转偏转，此时甚至**互信息 $I(T_p; Y_1, Y_2) > 0$**（因为知道了角度小数偏移，能部分预测导频落在哪个细角度！），而一旦按四个整数 $90^\circ$ 扇区求和积分，对称性立刻将四片花瓣等权重平均为精确的 $(1/4, 1/4, 1/4, 1/4)$！

3. **揭示后的样本为何仍能有效恢复第一符号 $X_1$（三体协同信息机制）**：
   第一符号与两类边际量单独独立：
   $$I(X_1; Y_1, Y_2) = 0, \qquad I(X_1; P) = 0, \qquad \text{但是}\quad I(X_1; Y_1, Y_2, P) = I(X_1; P \mid Y_1, Y_2) > 0.$$
   物理与信息论原因极其清澈：$Y_1$ 把信息锁定在和相 $U_1 = X_1 + G \pmod 4$ 上，而未知公共基准 $G$ 完全掩盖了 $X_1$ 的边际分布；尽管 $P$ 本身的边际分布与 $G$ 关于 $(Y_1, Y_2)$ 都保持 $C_4$ 循环均匀（四个粗扇区等概率 $1/4$），一旦实际揭示了 $T_p$ 的具体实现值（落于某个特定象限及分数偏移内），$P$ 就打破了 $G$ 的条件对称性，直接估计出 $G \approx \text{sector}(T_p e^{-i\hat\phi_p})$；随后只需通过差模关系 $X_1 = U_1 - G \pmod 4$ 与缓存中保存的 $U_1$ 后验 $a_1$ 卷积，$G$ 的公共盲区被抵消，从而将 $X_1$ 的期望准确率从盲猜 $25\%$ 拉升至约 $69\%+$（相干 2 dB 条件）。

---

## 5. 配对旋转变换规律与适用假设声明

保持题面与每点幅度 $|Y_{j,m}|$ 不变，在仅转动第二窗 `(0, 1)` 与两窗同转 `(1, 1)` 以及全局坐标系同转 `(1, 1, 1, 1)` 下，Q1、Q2（给定固定导频 $P$）、Q3（揭示前预测第三窗 $K_3$）的标准概率向量严格服从：

| 干预类型 | Q1 ($X_2-X_1$) | Q2 ($X_1$, 导频固定) | Q3 正例 `repeat_x1` ($K_3, \hat\mu_3$) | Q3 负对照 `independent_x3` |
| :--- | :---: | :---: | :---: | :---: |
| **`(0, 1)` 仅转第二窗 ($Y_2 \mapsto i Y_2$)** | **顺时针/模 4 加一格旋转 (`roll +1`)** | **严格保持 (`hold`)** | **严格保持 (`hold`)**（即便在连续漂移下完整 I/Q 亦保持！） | 恒均匀保持 (`1/4`) |
| **`(1, 1)` 两窗同转 ($Y_1, Y_2 \mapsto i Y_1, i Y_2$)** | **严格保持 (`hold`)** | **按 $+1$ 旋转 (`roll +1`)** | **按 $+1$ 旋转 (`roll +1`)，$\hat\mu_3 \mapsto i\hat\mu_3$** | 恒均匀保持 (`1/4`) |
| **全局坐标系旋转 ($Y_1, Y_2, P, Y_3 \mapsto i\cdot$)** | 保持 (`hold`) | 保持 (`hold`) | 观测标签随接收机坐标系按 $+1$ 旋转 | 恒均匀保持 (`1/4`) |

> **关键物理对比**：`(0, 1)` 与 `(1, 1)` 在 **Q1** 与 **Q3** 上的变换性质**恰好互补反转**！
> - Q1 测两窗**相对相位**，故仅转 $Y_2$ 使其旋转、两窗同转使其保持；
> - Q3 测固定 $t_3$ 接收帧中相对于**$Y_1$（因为协议只重发 $X_1$）**的绝对相位分布，故仅转 $Y_2$ 时（$Y_1$ 未转）**必须保持**，两窗同转（$Y_1$ 转了 $+90^\circ$）时**必须跟随 $+90^\circ$ 旋转**！这一正一反的对称约束是对编码器是否真实区分绝对与相对相位、而非套用固定模板的最尖锐判据。

### 证明与公式适用的协议假设列表
1. 符号先验 $(X_1, X_2, G)$（及负对照下的 $X_3$）相互独立且均匀分布于 $\mathbb{Z}_4$；
2. 已知单位导频固定发射 $s_0 = 1$ 且仅在 `shares_channel == True` 时共享同一随机信道实现；多条具有相同 `acquisition_id` 的导频属同一物理采集副本，只计入一次似然；
3. 各窗口接收噪声 $N_1, N_2, N_3, N_p$ 相互独立且与符号及相位漂移独立，符合已知信号功率与圆对称高斯白噪声模型；
4. 马尔可夫独立增量假设（$t_2 - t_1$ 与 $t_3 - t_2$ 增量在给定公共轨迹模型下条件独立，连续情况下由给定确定性 CFO $\omega$ 与布朗增量方差 $\sigma_\phi^2$ 描述）；当出现模型失配（如编码器停留在 $2\text{ dB}$ 训练支持域外的新 SNR 或强 CFO 失配）时，本推导同时提供与失配边界对应的新假设诊断（`C5`, `C6`）。
