# EM-FloatFormer Stage 5 & 6 数学猜想证明、反例证伪与优化定理汇编（`C1–C7`）

> **基线与零开销约束**：本文档建立在 [`STAGE5_6_COLAB_SCALING_REPORT.md`](./STAGE5_6_COLAB_SCALING_REPORT.md) 归档的 Stage 4/5/6 全量实验事实之上（基线提交 `4692089` / `344eada`），全部理论推演与数值校验在 `$0` 额外 GPU 开销与不修改锁定基线代码的前提下完成。所有定理、反例与常数均经双轮独立数学审校及逐行符号/数值验算确认；下文默认自然对数 $\log$（单位 `nat`）。

---

## 执行总览：`C1–C7` 审定结论与 Stage 6b / 7 直接算法收益

| 编号 | 研究命题 | 审定结论 | 核心闭合公式 / 关键反例 | Stage 6b / 7 直接优化落地 |
|---|---|---|---|---|
| **C1** | 误设后验主动风险、精确二元模量与对跖 EIG 律 | **`[SHARPENED & PROVED]`** | $R_6(q)-R_6(p)=[I_p(a_p)-I_p(a_q)]+B_{a_q}(p,q)$，$B_a=J_a-D(m_p^a\|m_q^a)$（**负号**）；$I_q(a)=\frac{1}{8}\Delta^\top G_a(m)\Delta+\mathcal{R}$；精确最优二元模量 $\Omega(J_a)$ 与均匀 CFO 必选动作 2（`0.002055 nat`） | 放弃非凸 $B_a$，改用**无偏单样本蒙特卡洛平均预测交叉熵** $\widehat{L}_{\rm avg}$ 最小化 $\bar{J}=\frac{1}{4}\sum_a J_a$；选择与揭示风险分开记账 |
| **C2** | 非线性 `GELU` 函数融合秩障、构造压缩与 Cayley 曲率 | **`[SHARPENED & PROVED]`** | 归档实为 **2 个宽度-11（22 单元）$\to\le 3$ 单元**；二阶 Hermite 下界 $\frac{1}{2}\sum_{j>r}\sigma_j(T)^2$；$CA=1$ 符号翻转反例反超零基线 $+\frac{\sqrt{3}}{2\pi}$；Cayley 曲率界 $\frac{1}{2}w(1-w)\|D\|_2\|D\|_F$ | 以**全混沌经验 Gram 投影** $C=HK^\dagger$ 替代线性 `CA` 的 SVD；以**$\binom{22}{3}=1540$ 枚举+一二阶联合谱初始化**消除 $\pm a_k$ 符号歧义 |
| **C3** | 残差深度、逐层 Pre-RMSNorm 与多位置读出权衡 | **`[REFUTED WITH COUNTEREXAMPLE & SHARPENED]`** | 显式反例证明仅凭 $\|\text{resid}\|=\Theta(\sqrt{L})$ 与 3:1 混合率无法推导消失或秩-1坍缩；加入垂直有界性后逐层 Pre-RMSNorm 给出**原始 $\Theta(\sqrt{L})$、归一化 $\Theta(1)$**；多抽头增益精确为 $I(Y;T\mid U)$ | 暂不重训 27B 主干权重：先在同包、同模式下对比 `last-soft` 与 `last-prose` 中/末层读出与方向雅可比 |
| **C4** | `MixKL` 双侧共模/差模运行时扰动与跨任务子空间隔离 | **`[SHARPENED & PROVED]`** | 证伪 $\sqrt{2D}\operatorname{std}_q\delta$（正确因子为 $\sqrt{\chi^2}$）；**纯共模**满足 $e^{-b_0}D\le D(p_{\delta_0}\|q_{\delta_0})\le e^{b_0}D$（全局无 $O(\sqrt{D})$ 项）；**双侧** $(e^{-b_0/2}\sqrt{D}-b_\Delta/\sqrt{8})_+^2 \le D_{\rm new}\le (e^{b_0/2}\sqrt{D}+b_\Delta/\sqrt{8})^2$；自然文本投影允许 $L=Me_2e_2^\top$ 破坏新任务软 token | 采样配对冻结/适配 logits 直接测算 $(b_0,b_\Delta)$ 认证跨平台裕量；Stage 6 评估严格挂载**独立头并彻底旁路 Stage-4 适配器** |
| **C5** | `S17 ⊂ C16` AdamW 条件数、轨迹包含与残差-SVD 解锁 | **`[REFUTED WITH COUNTEREXAMPLE & SHARPENED]`** | 构造 2 点反例推翻无条件一步符号 SNR 退化；揭示**全局梯度裁剪**会打破仅 $\gamma(0)=0$ 的等价性；给出 **4 类门控/约束的闭式最优 $V$**（标量门 $\|V\|_F=1 \Rightarrow V=\pm R/\|R\|_F$） | warmup 期锁定 $\gamma=0$ 并排除在全局裁剪外（严格复刻 S17），解锁瞬间对 $R=\sum_i u_i z_i^\top$ 取**标量门+归一化 $R$（或截断 $R_k$）** |
| **C6** | 极坐标缓存三层误差、Hölder 温度界与 EVM 次-$\log 4$ 定理 | **`[SHARPENED & PROVED]`** | 区分连续密度 KL、量化后验 KL 与 Q2 循环相关 Top-1 翻转；**Hölder 界**证明大有限 $T$ 保序降 C16 NLL 至趋近 $\log 4$；构造 1,280 行反例证明三项汇总指标**无法认证温度低于 $\log 4$**，但同一 EVM 认证**均匀混合 $q=(1-\lambda)U+\lambda p^\circ$ 严格低于 $\log 4$（$\le 1.381942$）** | 先用末位温度/混合缩放验证实网校准（保 Top-1）；部署层采用**含 S17 端点的簇级 Hoeffding 有限样本回退判据** |
| **C7** | 离散粗化与受限读出头 $\mathcal{V}$-信息勾股分解 | **`[PROVED & TWO-REGIME SHARPENED]`** | 离散揭示风险差精确分解 $R_T^*-R_W^*=I(H;W\mid T,Z_{a_T})+\mathbb{E}_W[I_W(H;Z_{a_W})-I_W(H;Z_{a_T})]$；$\mathcal{V}_{\rm lin}$ 满足精确三-KL 勾股恒等式，连续实解析单射宿主的瓶颈不在香农熵而在精度碰撞与 $\mathcal{V}$ 逼近差 | 在 27B 上同步拟合**常数、仿射-softmax、宽-96 GELU**三类读出头，直接定位“表示碰撞” vs “头函数类受限” |

---

## 一、前沿开放问题与文献相关性筛选

在进入具体定理推演前，我们首先结合 `EM-FloatFormer` 的精确几何（`5` 个旧包窗口 $\times$ `4` 个揭示动作 $\times$ `16` 个潜态 $\times$ `2` 值揭示比特）与混合宿主机制，对相关数学与机器学习前沿方向执行严谨筛选：

### 1.1 保留的四条高杠杆数学前线

1. **模型误设下的有限贝叶斯实验设计（Misspecified BOED）与信息单调性模量**
   - **相关文献**：Go & Isaac（UAI 2022, [PMLR v180](https://proceedings.mlr.press/v180/go22a.html)）、Catanach & Das（2023, [`arXiv:2304.07949`](https://arxiv.org/abs/2304.07949)）、Go, Qian & Yoon（2026, [`arXiv:2605.26093`](https://arxiv.org/abs/2605.26093)）、Sakai & Iwata（2016, [`arXiv:1602.04577`](https://arxiv.org/pdf/1602.04577) Lemma 2）。
   - **对接代码与机理**：[`verify_m5.py`](../stage4_math_and_design/M5/verify_m5.py)（第 678–843 行）完整给出了已知转移核 $k_{ahs}(z)$ 与二元下游决策，因此无需照搬通用渐近界，可直接推导出误设代理后验 $q(r,g)$ 对选窗与揭示解码的双重精确遗憾，并导出可直接训练的凸代理目标。
2. **非线性适配器函数融合的逼近与对称对齐理论**
   - **相关文献**：Ainsworth, Hayase & Srinivasa（*Git Re-Basin*, [`arXiv:2209.04836`](https://arxiv.org/abs/2209.04836)）。
   - **审视修正**：通用“线性模式连通性（LMC）”在仓库已有 SVD 合并中点处无法给出精度保证（仅证存在低损路径而非算术中点）。真正的高杠杆问题应重塑为**固定隐藏宽度下的函数逼近问题**：即通过 Wiener-Hermite 混沌展开与全混沌 Gram 投影，将 22 个激活 `GELU` 单元无歧义压缩至固定预算，同时对照 `SwiGLU` 对跖双线性精确秩（[`THEOREM_W1.md`](../stage4_math_and_design/weight_mod/W1/THEOREM_W1.md)）与 Cayley 李代数几何（[`THEOREM_W2.md`](../stage4_math_and_design/weight_mod/W2/THEOREM_W2.md)）。
3. **混合注意力/状态空间前缀可辨识性与受限 $\mathcal{V}$-信息**
   - **相关文献**：Petrov et al.（ICLR 2024, [`arXiv:2310.19698`](https://arxiv.org/abs/2310.19698)）、Siems et al.（*DeltaProduct*, [`arXiv:2502.10297`](https://arxiv.org/abs/2502.10297)）、Xu et al.（ICLR 2020, [`arXiv:2002.10689`](https://arxiv.org/abs/2002.10689)）。
   - **审视修正**：`Petrov et al.` 约束的是固定自注意力对内容矩阵的相对注意比重，*DeltaProduct* 研究的是跨 token Householder 乘积的状态追踪表达力；二者均不直接推出带可学习前缀、逐层 Pre-RMSNorm 及末位文本读出的深度衰减定律。必须将其改写为**可测量的分段雅可比可辨识性与受限读出头 $\mathcal{V}$-信息**（定理 `C3` 与 `C7`）。
4. **极坐标量化传感后验校准与有限样本选择界**
   - **相关文献**：Guo et al.（ICML 2017, [PMLR v70](https://proceedings.mlr.press/v70/guo17a.html)）、Qian, Liang & Adams（JMLR 2025, [JMLR v26](https://www.jmlr.org/papers/v26/24-0700.html)）。
   - **对接代码与机理**：[`sdr_real_pipeline.py`](../stage4_real_and_surgical/sdr_real_pipeline.py)（第 1589–1733 行）将极坐标缓存 `m5-c16-v1-le` 经中点还原、质量相关方差倍增、精度截断 `[0.05, 50]` 及导频循环卷积给出 Q2 概率，导致实网 NLL 与 Top-1 严重分裂；必须从密度似然、量化后验、循环相关与末位概率校准四层解耦分析。

### 1.2 剪枝的三个低收益方向

1. **一般非阿贝尔群 $S_n / A_n$ 字问题（Word Problems）**：本任务的物理旋转对称群严格为阿贝尔群 $C_4 / SO(2)$，与非阿贝尔替换字群同态复杂度无关。
2. **$m, n \to \infty$ 渐近近似消息传递（AMP）与副本（Replica）极限**：M5 问题具有极小但精确可积分的有限几何（5 窗、4 动作、16 态），大系统极限反而会丢失 `CFO` 轨道间距与窗间重叠导致的精确排序信息。
3. **任意无校准跨宿主循环隐状态直接交接**：不同 LLM 架构/尺度间无对齐的隐状态迁移优先级远低于单宿主冻结层可恢复性判定（`C7`）。

---

## 二、七大定理的严格证明、反例与构造性算法（`C1–C7`）

### 定理 C1：Stage 6 误设后验主动风险分解、精确二元模量 $\Omega(j)$、二次对跖律与单样本无偏训练

#### 1a. 精确选择—揭示风险分解与绝对负号法则
在给定旧包 $W$ 及其所属公开物理家族后，记潜态为 $S=(r,g) \in \{0,1,2,3\}^2$，$H \sim \mathrm{Unif}\{0,1\}$ 为独立揭示比特。对候选动作 $a \in \{0,1,2,3\}$ 与复高斯揭示核 $k_{ahs}(z) = \frac{1}{\pi v_a}\exp(-|z-\mu_{ahs}|^2/v_a)$，任意权重分布 $u$ 诱导条件密度与边际密度：
$$
f_{u,h}^a(z) = \sum_s u_s k_{ahs}(z),\qquad m_u^a(z) = \frac{1}{2}\big(f_{u,0}^a(z) + f_{u,1}^a(z)\big),\qquad \eta_{u,h}^a(z) = \frac{f_{u,h}^a(z)}{2 m_u^a(z)},
$$
对应的模型主观预期信息增益（EIG）为 $I_u(a) = \frac{1}{2}\sum_h D_{\mathrm{KL}}\big(f_{u,h}^a \,\|\, m_u^a\big)$。

设 $p$ 为当前条件下的真实状态后验，$q$ 为 Stage 6 读出头输出的代理后验，$a_u \in \arg\max_a I_u(a)$。固定动作 $a$ 时，真实后验 $p$ 的期望对数损失为 $\log 2 - I_p(a)$。由于 [`reveal_and_score`](../stage4_math_and_design/M5/verify_m5.py#L843) 同时用代理 $q$ 执行选窗与推断比特后验 $\eta_{q,h}^a(Z)$，固定动作 $a$ 的期望超额对数损失精确等于：
$$
B_a(p,q) := \frac{1}{2}\sum_{h\in\{0,1\}} \int f_{p,h}^a(z)\log\frac{\eta_{p,h}^a(z)}{\eta_{q,h}^a(z)}\,dz
= \underbrace{\frac{1}{2}\sum_h D_{\mathrm{KL}}\big(f_{p,h}^a \,\|\, f_{q,h}^a\big)}_{J_a(p,q)} - D_{\mathrm{KL}}\big(m_p^a \,\|\, m_q^a\big)
= \mathbb{E}_{m_p^a}\big[D_{\mathrm{KL}}(\eta_p^a(\cdot\mid Z)\,\|\,\eta_q^a(\cdot\mid Z))\big] \ge 0.
$$
> **符号修正（反例证伪 $+D(m_p^a\|m_q^a)$）**：贝叶斯分母中的 $m^a$ 导致边际散度以**负号**进入超额损失。反例：构造对任意状态 $s$ 均满足 $k_{a0s} = k_{a1s}$（即揭示不携带 $H$ 信息），而各态观测分布各异的实验。此时无论 $p,q$ 为何，二者始终输出完全一致的后验 $1/2$，精确超额比特损失 $B_a(p,q) \equiv 0$，但边际散度 $D_{\mathrm{KL}}(m_p^a\|m_q^a) > 0$，若取正号将得出严格大于零的错误结果。

在总超额风险中加减最优动作处 $a_q$ 的物理贝叶斯风险 $\log 2 - I_p(a_q)$，立得精确分解恒等式：
$$
\boxed{R_6(q) - R_6(p) = \big[I_p(a_p) - I_p(a_q)\big] + B_{a_q}(p,q).}
$$
**归档解读注记**：在 [`verify_m5.py`](../stage4_math_and_design/M5/verify_m5.py#L751) 中，`geometry_utilities` 采用间距约 `0.36` 的中点网格估算 EIG（细网格校验在前 64 行执行）；若实现网格相对连续积分的均匀误差不超过 $\eta$，选窗上界需叠加 $+2\eta$。此外，[`score_rows`](../stage4_math_and_design/M5/verify_m5.py#L843) 对概率施加了 $10^{-280}$ 下限截断；若截断下限记作 $e^{-c}$，则截断后损失恒等于无截断损失减去 $\mathbb{E}[(\ell-c)_+]$。

#### 1a 进阶：三级收紧选择遗憾界（含 Sakai–Iwata 精确最优二元模量）
记 $K := D_{\mathrm{KL}}(p\|q) = D_{\mathrm{KL}}(p_R\|q_R) + \sum_r p_R(r) D_{\mathrm{KL}}(p_{G\mid r}\|q_{G\mid r})$。对信道通过数据处理不等式即有 $B_a \le J_a \le K$。针对关键选择项 $|I_p(a) - I_q(a)| = |H_P(H\mid Z_a) - H_Q(H\mid Z_a)|$（记 $P,Q$ 为 $(H,Z_a)$ 在权重 $p,q$ 下的联合分布，故 $J_a = D_{\mathrm{KL}}(P\|Q)$），依次确立以下三个收紧层级：

1. **Hellinger 梯度初界与联合二元最大值收紧**：令 $u_h(z) = \sqrt{P(h,z)}$，$\theta_h(z) = u_h(z)^2 / \|u(z)\|^2$，则条件熵密度 $F(u) = -\sum_h u_h^2 \log(u_h^2/\|u\|^2)$ 的梯度范数平方精确为：
   $$
   \|\nabla F(u)\|^2 = 4\|u\|^2 \big[\theta\log^2\theta + (1-\theta)\log^2(1-\theta)\big] = 4\|u\|^2 c(\theta).
   $$
   分项用 $x\log^2 x \le 4/e^2$ 放缩得 $c(\theta) \le 8/e^2 \approx 1.082682$ 与单动作因子 $\frac{4\sqrt{2}}{e} \approx 2.081040$；而直接求导 $c'(x) = \log\frac{x}{1-x}[\log(x(1-x))+2] = 0$，得唯一对称内部极大点：
   $$
   x_* = \frac{1 - \sqrt{1 - 4e^{-2}}}{2} = 0.1613782099,\qquad c_* = c(x_*) = 0.5628799121.
   $$
   沿 $\sqrt{P}$ 与 $\sqrt{Q}$ 的直线段积分并利用 Jensen 不等式 $\|\sqrt{P}-\sqrt{Q}\|_2 \le \sqrt{J_a}$，将单动作因子全局收紧约 `28%` 至 $2\sqrt{c_*} = 1.500506464$（对应选择项因子从 $4.162081$ 降至 $4\sqrt{c_*} = 3.001012928$）。
2. **一般公平二元实验的不可改进最优精确模量 $\Omega(j)$（基于 Sakai–Iwata Lemma 2）**：设 $h(\cdot)$ 为二元自然熵，$d(t\|u)$ 为伯努利 KL 散度，定义：
   $$
   \boxed{\Omega(j) := \max_{\substack{t,u\in[0,1/2]\\ d(t\|u)\le j}} |h(t) - h(u)|.}
   $$
   **证明**：取 $t,u\in[0,1/2]$ 使得 $h(t)=H_P(H\mid Z)$、$h(u)=H_Q(H\mid Z)$。对任意 $\gamma > -1$ 令 $\alpha = 1/(1+\gamma) > 0$。对任意二元分布 $b,c$，Gibbs 变分不等式给出 $\gamma H(b) - D_{\mathrm{KL}}(b\|c) \le \log\|c\|_\alpha$（等号在 $b_i \propto c_i^\alpha$ 时取得）。将其先应用于条件分布 $P(H\mid Z), Q(H\mid Z)$、结合 KL 链式法则再对 $Q_Z$ 使用 $\log$-MGF 凸对偶及 [Sakai–Iwata (2016) Lemma 2](https://arxiv.org/pdf/1602.04577) 中“二元 $\alpha$-范数 $\|\mathrm{Ber}(p)\|_\alpha$ 是二元熵 $h(p)$ 的全局凹函数”这一核心引理：
   $$
   \gamma h(t) - J_a \le \log \mathbb{E}_{Q_Z}\|Q(H\mid Z)\|_\alpha \le \log \|\mathrm{Ber}(u)\|_\alpha.
   $$
   选取使 $\mathrm{logit}(t) = \alpha\,\mathrm{logit}(u)$ 成立的 $\alpha > 0$，即令外侧单点变分公式取等，直接推得 $J_a \ge d(t\|u)$（端点由连续性及直接代入覆盖），因此 $|I_p(a)-I_q(a)| \le \Omega(J_a)$ 严格成立。另一方面，对 $H\sim\mathrm{Unif}\{0,1\}$ 经过交叉概率分别为 $t,u$ 的两个二元对称信道，其联合 KL 精确为 $d(t\|u)$ 而互信息差精确为 $|h(t)-h(u)|$，从而证明 **$\Omega(J_a)$ 是仅依赖于 $J_a$ 的最紧全局可能界**。

   综合精确分解即得收紧后的全局遗憾不等式：
   $$
   \boxed{R_6(q) - R_6(p) \le B_{a_q} + \min\big\{\log 2,\ \Omega(J_{a_p}) + \Omega(J_{a_q})\big\},\qquad 0 \le B_{a_q} \le J_{a_q}.}
   $$
   （下表展示三组常数在不同 $j$ 下的数值对比，其中 $\Omega(j)$ 由 $1000\times 1000$ 网格搜索给出下界参考值）：

   | 散度预算 $j$ | 最优模量 $\Omega(j)$ | 收紧 Hellinger 界 $2\sqrt{c_* j}$ | 初版界 $\frac{4\sqrt{2}}{e}\sqrt{j}$ |
   |---:|---:|---:|---:|
   | `0.001` | `0.029921` | `0.047450` | `0.065808` |
   | `0.010` | `0.096779` | `0.150051` | `0.208104` |
   | `0.050` | `0.225329` | `0.335523` | `0.465335` |
   | `0.100` | `0.326781` | `0.474502` | `0.658083` |

#### 1b. 二次对跖 EIG 坍缩律与均匀先验 CFO 必选动作 2 证明
记对跖差 $\Delta_j = q(r,g) - q(r,g+2)$（$j=(r,g), g\in\{0,1\}$），对跖核差 $b_j = k_{\mu_{a0rg}} - k_{-\mu_{a0rg}}$。在 `jump_interval`（仅 $r=a$ 激发）与 `selective_fade`（全部四模态均有互异正半径）下，有 $f_0 - f_1 = \sum_j b_j \Delta_j$。令 $m = (f_0+f_1)/2$、$t = (f_0-f_1)/(2m) \in [-1,1]$，则：
$$
I_q(a) = \int_{\mathbb{C}} m(z)\,\psi(t(z))\,dz,\qquad \psi(t) = \frac{(1+t)\log(1+t)+(1-t)\log(1-t)}{2}.
$$
- **零 EIG 充要定理**：非负被积函数恒为零当且仅当 $f_0 \equiv f_1$。由于复高斯卷积的傅里叶变换处处非零，它在有限符号原子测度空间上**严格单射**。两家族中涉及的正负均值中心彼此互异，故对应有符号原子质量必逐一为零，即 **$I_q(a) = 0 \iff$ 全部活跃对跖差 $\Delta_j = 0$**（对未来若含重合中心的目录，充要条件推广为各互异中心处的总符号质量为零）。
- **均匀四阶余项界**：利用 [-1,1] 上的幂级数精确夹逼 $0 \le \psi(t) - t^2/2 \le (\log 2 - 1/2)t^4$，得：
  $$
  \boxed{I_q(a) = \frac{1}{8}\Delta^\top G_a(m)\Delta + \mathcal{R},\qquad [G_a(m)]_{ij} = \int \frac{b_i(z)b_j(z)}{m(z)}\,dz,}
  $$
  $$
  0 \le \mathcal{R} \le \frac{\log 2 - 1/2}{16} \int \frac{(b^\top \Delta)^4}{m^3}\,dz.
  $$
  上述单射性确保 $G_a(m) \succeq \pi v_a \int b b^\top dz \succ 0$；对混合权 $\lambda_i$ 用凸性 $m^{1-k} \le \sum_i \lambda_i k_i^{1-k}$ 结合复高斯积分配方法 $\int k_x^k / k_y^{k-1} = \exp\big(k(k-1)|x-y|^2/v_a\big)$（$k=2,4$），即可证明在固定动作下即使逼近单纯形边界，**$\mathcal{R} = O(\|\Delta\|^4)$ 且 $I_q(a) = \Theta(\|\Delta_{\rm rel}\|^2)$ 一致成立**。
- **均匀先验 CFO 为何稳定选中动作 2**：在 [`verify_m5.py`](../stage4_math_and_design/M5/verify_m5.py#L693) 的 `cfo_alias` 家族中，各动作的额外旋转相位到最近四分之一直角的角距分别为 `0`（动作 0）、`\pi/6`（动作 1）、`\pi/5`（动作 2）、`\pi/14`（动作 3）。在均匀权重 $q=1/16$ 下（`[-4,4]^2` 上 $h=1/160$ 二维积分加尾部、Simpson 与梯度曲率严格区间封闭）：

  | 零基动作 | 归档中点网格积分 (`nat`) | 严格连续积分区间 (`nat`) | 物理诠释 |
  |---:|---:|---:|---|
  | `0` | `0` | 精确 `0` | 恰好命中 $\pi/2$ 倍数，四点轨道全同不变 |
  | `1` | `0.001743506` | `[0.001738, 0.001907]` | 偏离四分之一圈 $\pi/6$，次优 |
  | **`2`** | **`0.002054621`** | **`[0.002049, 0.002219]`** | 偏离 $\pi/5$ 且与其余三模态重叠最弱，**严格全局最大** |
  | `3` | `0.000451522` | `[0.000450, 0.000610]` | 偏离仅 $\pi/14$，EIG 显著较小 |

  该理论值 `0.002055 nat` 与 27B 冻结 C16 在 Stage 6 实测的 CFO `0.002036 nat` 构成近乎一致的行为指纹。

#### 1c. $B_a(p,q)$ 的非凸性反例与无偏单样本蒙特卡洛训练损失
1. **$B_a(p,q)$ 在单纯形上严格非凸**：
   - *闭式解析反例*：设状态 $A$ 精确揭示 $Z=H$，状态 $B$ 恒输出 $Z=0$；取 $p=A$、$q_t = (1-t)A + tB$（$t\in(0,1)$）。$H=1,Z=1$ 时无损失；$H=0,Z=0$ 时预测正确概率为 $1/(1+t)$，故 $B_a(p,q_t) = \frac{1}{2}\log(1+t)$ **严格凹**。
   - *真实 M5 目录验证*：在 `jump_interval` 动作 `0` 上混合活跃态（均值 $(-1,+1)$）与非活跃态（均值 $(-1,-1)$），真值全落于活跃态，$v_a=0.45$。非活跃态权重取 $t=0.25, 0.50, 0.75$ 时一维积分测得 $B_a = 0.10091919,\, 0.18732780,\, 0.26319171$，中点超出两端均值为 **`+0.00527235 nats`**。
2. **仅匹配对跖差亦不充分**：取单一活跃模态 $q=(1/4+u+d/2,\, 1/4-u,\, 1/4+u-d/2,\, 1/4-u)$（$d\neq 0$），改变 $u$ 不改变模态总质量与两项对跖差 $\Delta$，却改写分母 $m(z)$ 进而改变比特后验。
3. **无偏单样本蒙特卡洛凸训练目标**：
   令联合全动作虚拟实验 $P_u^{\rm all}$ 对一次 $(S,H)\sim u\otimes\mathrm{Unif}\{0,1\}$ 独立采样四个潜在观测 $Z_{1:4}$。仅需在每个训练样本上按教师权重抽 $S\sim p$、公平 $H$，并采样一组四个独立高斯观测 $k_{as}:=k_{aHs}(Z_a)$，立即得到关于单纯形 $q$ **严格凸**且关于 logits $l$ 梯度有界（分别就是 $q-\Pr_q(S\mid H,Z_{1:4})$ 与 $\frac{1}{4}\sum_a[q-\Pr_q(S\mid H,Z_a)]$）的无偏估计量：
   $$
   \widehat{L}_{\rm all} = -\log\sum_{s=0}^{15} q_s \prod_{a=0}^3 k_{aHs}(Z_a),\qquad
   \widehat{L}_{\rm avg} = -\frac{1}{4}\sum_{a=0}^3 \log\sum_{s=0}^{15} q_s k_{aHs}(Z_a).
   $$
   二者的梯度与方差满足：
   - **认证范围**：$\bar{J} := \frac{1}{4}\sum_a J_a$ 直接锚定四个真实单揭示实验（$\bar{J}\le\max_a J_a\le 4\bar{J}$），而 $J_{\rm all} \in [\max_a J_a,\, D_{\mathrm{KL}}(p\|q)]$ 可能额外惩罚单次揭示不可观测的跨窗高阶关联；
   - **分数方差序关系**：在最优解 $p=q$ 处，边际梯度 $\widehat{g}_{\rm avg}$ 的每一项恰好是全动作梯度 $\widehat{g}_{\rm all}$ 的条件期望，由条件 Jensen 不等式严格成立矩阵方差偏序 $\mathrm{Cov}(\widehat{g}_{\rm avg}) \preceq \mathrm{Cov}(\widehat{g}_{\rm all})$；但一般 $p\neq q$ 处的标量交叉熵方差并不存在绝对单调序（四独立同分布非均匀传感器时联合方差为 $4\sigma^2$ vs 平均为 $\sigma^2/4$，而相关确定性传感器反例下则相反）。故推荐 Stage 6b 首选以 **$\widehat{L}_{\rm avg}$** 替代无加权 16 维 KL，并同步记录各单动作 $J_a$。

---

### 定理 C2：非线性 `GELU` 函数融合秩障、构造压缩与符号恢复、对跖 `SwiGLU` 秩及 Cayley 曲率界

#### 2a. 归档上下文修正与 Wiener-Hermite 二阶压缩下界
代码审校确认 [`surgical_adapter.py`](../stage4_real_and_surgical/surgical_adapter.py#L346) 中 `r_schur` 默认值为 **`11`**，两套专家各含 11 个激活 `GELU` 单元（[`SchurIrrepReadout`](../stage4_real_and_surgical/surgical_adapter.py#L259) 输出 $C_i\,\mathrm{GELU}(A_i h)$）。[`real_sdr_and_stage6_gpu.py`](../stage4_real_and_surgical/real_sdr_and_stage6_gpu.py#L197-L222) 中的融合流程实际上通过平均线性积 $w_1 C_1 A_1 + w_2 C_2 A_2$ 并取 SVD，**将合计 22 个激活非线性单元强行压缩到至多 3 个单元**（浮点 `4.7095e-19` 仅度量线性矩阵乘积重构误差，且该实验同时伴随 LoRA 与 Cayley 改动）。

令 $X\sim\mathcal{N}(0,I_d)$、$g(z)=z\Phi(z)$，吸收输出矩阵与融合权重后目标为 $F(X)=\sum_j c_j g(a_j^\top X)$。利用高斯分部积分两次与 $g''(z)=(2-z^2)\phi(z)$，定义每个输出坐标 $o$ 对应的精确二阶 Hermite 对称矩阵 $T_o$（并纵向堆叠为矩阵 $T$）：
$$
T_o := \mathbb{E}\big[F_o(X)(XX^\top - I_d)\big] = \sum_j c_{oj}\,\kappa(\|a_j\|)\,a_j a_j^\top,\qquad \kappa(s) = \frac{2+s^2}{\sqrt{2\pi}(1+s^2)^{3/2}}.
$$
任意宽度-$r$ 学生网络对应的二阶混沌矩阵堆叠 $T_{\rm stu}$ 的行空间被其 $r$ 条输入向量张成，故 $\operatorname{rank}(T_{\rm stu}) \le r$。由不同阶 Gaussian-Hermite 混沌的正交性以及最佳秩-$r$ Frobenius 逼近定理，立得严格的二阶压缩下界：
$$
\boxed{\inf_{\text{width } r}\mathbb{E}\|F(X)-\widehat{F}(X)\|^2 \ge \frac{1}{2}\sum_{j>r}\sigma_j(T)^2.}
$$
同时可确立三条关键结构性事实：
1. **原始行空间秩不是障碍判据**：$g(x_1)+g(x_2)$ 与 $-g(x_1)+g(x_3)$ 的输入行空间共 3 维，但等权均值仅需宽度 `2`；无偏置 `GELU` 网络（$g(z)=z/2+e(z)$，含全阶正偶泰勒项）的严格函数相等**当且仅当**一阶线性部分相等且按**无符号向量 $a\sim -a$（保留范数）**分组后的 $e(a^\top x)$ 向量系数逐一相同（沿通用射线 $x=tv$ 由偶数阶幂展开构成可逆 Vandermonde 系统证毕）。
2. **仅匹配线性积 $CA$ 可能严格劣于直接输出零！** 取标量目标 $f(x)=g(x)$ 与学生 $\hat{f}(x)=-g(-x)$。二者线性乘积同为 `1`（重构误差 `0`），但由 $g(x)-g(-x)=x$ 及高斯积分 $\mathbb{E}[X^2\Phi(X)^2] = \frac{1}{3}+\mathbb{E}[\phi(X)^2] = \frac{1}{3}+\frac{1}{2\pi\sqrt{3}}$，可得精确均方误差差：
   $$
   \mathbb{E}(f-\hat{f})^2 - \mathbb{E}f^2 = 3\mathbb{E}g(X)^2 - 1 = \frac{\sqrt{3}}{2\pi} \approx +0.275664 > 0.
   $$
   SVD 的正负号自由度在穿过非线性 `GELU` 后完全可能破坏偶次谐波相干性。
3. **二阶界在更高偶次混沌处仍可能严格松弛（反例）**：取 $d=r=1$ 偶目标 $F(x)=g(x)+g(-x)$，其 $T$ 的被丢弃奇异值为 `0`，但任意单神经元 $cg(ax)$ 的奇部范数 $t=|ca|/2$ 且偶部范数不超过 $t$（因 $|\Phi(ax)-1/2|\le 1/2$），于是其真实全空间均方误差严格大于零，至少为 $t^2+(\|F\|_2-t)_+^2 \ge \|F\|_2^2/2 > 0$。

#### 2a 进阶：构造性一/二阶联合压缩、符号恢复与全混沌 Gram 投影
1. **联合一阶+二阶 Hermite 光谱压缩与 $\pm a_k$ 符号精确可辨识性**：
   构造包含一阶奇部目标 $L=\frac{1}{2}\sum_j c_j a_j^\top$ 与二阶偶部目标 $T_o$ 的矩阵 $M$，其第 $o$ 行为 $\big[L_o,\, \operatorname{vec}(T_o)^\top/\sqrt{2}\big]$。将学生参数写作 $\hat{a}_k = \sigma_k s_k n_k$（$\|n_k\|_2=1, s_k\ge 0, \sigma_k\in\{\pm 1\}$），构造基行 $D_k = \big[\sigma_k s_k n_k^\top/2,\, \kappa(s_k)s_k^2 \operatorname{vec}(n_k n_k^\top)^\top/\sqrt{2}\big]$，则一+二阶合并误差精确为 $\|M-CD\|_F^2$，最优输出权重由闭式解 $C=MD^\dagger$ 给出，从而规约成 $\inf_{\{n_k,s_k,\sigma_k\}} \|M(I-D^\dagger D)\|_F^2$。
   - **双轨初始化（用于 $22\to 3$ 压缩）**：(i) 取 $L^\top L + \frac{1}{2}\sum_o T_o^\top T_o$ 的前 $r$ 个特征向量并枚举 $2^r=8$ 种符号；(ii) 对已知 22 条教师行暴力枚举 $\binom{22}{3}=1540$ 组三元子集与全部 8 组符号，各做一次最小二乘解取更优者再精修。
   - **符号可辨识充要条件**：记 $A_k = c_k s_k n_k^\top$，在偶字典与 $C$ 固定时，一阶条件 $2L=\sum_k \sigma_k A_k$ 允许唯一符号**当且仅当不存在非空子集 $S$ 满足抵消条件 $\sum_{k\in S}\sigma_k A_k = 0$**（因此 $\operatorname{vec}(A_k)$ 线性无关即可充分保障；反之三个标量加权行 $e_1^\top, e_2^\top, -(e_1+e_2)^\top$ 之和为零，全部取反号仍同时匹配奇偶两部分）。
2. **全混沌 Gram 无截断最终投影**：令学生特征 $\varphi_k = g(\hat{a}_k^\top X)$，$K = \mathbb{E}[\varphi\varphi^\top]$，$H = \mathbb{E}[F\varphi^\top]$。全阶 $L^2(\mathcal{N}(0,I))$（或实网经验激活分布）下的闭式最优输出矩阵直接为 **$C_{\rm opt} = H K^\dagger$**（其中每对神经元期望仅为解析/二元高斯积分，实网下直接取共享隐藏激活上的经验 Gram 矩阵）。

#### 2b. 对跖 `SwiGLU` 双线性精确最小秩
由 $\sigma(u)+\sigma(-u)=1 \implies [\mathrm{SiLU}(u)-\mathrm{SiLU}(-u)]v = uv$（将每对对跖门计作 1 个双线性乘积特征 / 2 个物理门控单元）：
- **分离输入标量映射 $x^\top M y$**：精确最小特征秩等于 **$\operatorname{rank}(M)$**（由 SVD 实现），双宽度-$r$ 专家合并至多需 $\min(2r, d_x, d_y)$。
- **同输入对称二次型 $x^\top M x$**：由惯性定律与平方差分解 $u^2-v^2=(u+v)(u-v)$，精确最小特征数等于 **$\max\big(n_+(M), n_-(M)\big)$**。
- **分离输入向量映射**：精确最小秩为三阶张量 $\sum_j c_j \otimes a_j \otimes b_j$ 的**实张量秩**（直接拼接上界 $2r$，不再受输入维度 $d$ 上界限制：如 $d=2$ 时二次映射 $(x_1^2, x_1 x_2, x_2^2)$ 已需 3 个线性无关双线性特征）。共享同一 $B_{\rm align}$ 并共享同一门/值基（各自维度 $r$）时，展开发生交叉项界为 $r^2$（同输入同基时为 $r(r+1)/2$），而精确共享对应 $r$ 条基特征则仍可保持秩 $r$。

#### 2c. Cayley 李代数平均的全局二阶曲率界与酉性亏格
按仓库归档约定 $Q(S)=2(I+S/2)^{-1}-I$（$S^*=-S$），记 $D=S_1-S_2$、$\bar{S}=wS_1+(1-w)S_2$。沿线段 $S_2+tD$ 的预解式满足 $\|R(t)\|_2 \le 1$，二阶导数为 $Q''=RDRDR$，线性插值余项立得**全局二阶曲率界**：
$$
\boxed{\|wQ(S_1)+(1-w)Q(S_2)-Q(\bar{S})\|_F \le \frac{1}{2}w(1-w)\|D\|_2\|D\|_F,}
$$
其由纯虚可交换标量例 $S_1=it, S_2=-it, w=1/2$ 渐近达紧（差值精确为 $\frac{2(t/2)^2}{1+(t/2)^2}$，说明该误差源于 Cayley 映射曲率而非对易子 $[S_1,S_2]$——BCH 对易子控制的是**矩阵乘积**的对数而非算术平均）。同时算术平均 $A=wQ_1+(1-w)Q_2$ 严格满足恒等式：
$$
I - A^*A = w(1-w)(Q_1-Q_2)^*(Q_1-Q_2),
$$
对任意 $0<w<1$，$A$ 可落于酉群（或等于 $Q(\bar{S})$）当且仅当 $Q_1=Q_2$。若代码对 $\bar{S}$ 额外做了投影与截断得 $\widetilde{S}$，由预解恒等式误差还需叠加 $\|\widetilde{S}-\bar{S}\|_F$（而现行 4 个随机向量的范数检验无法探测生成元截断误差）。

---

### 定理 C3：残差深度、逐层 Pre-RMSNorm 与多位置读出权衡（反例证伪与精炼）

#### 3i. 分块因果雅可比递推、深度缩放反例与逐层 Pre-RMSNorm 界
将长度划分为前部 8 个软 token（$S$）与后部固定文本（$P$），第 $\ell$ 层因果块雅可比写为分块下三角矩阵 $\begin{pmatrix}A_\ell & 0 \\ B_\ell & C_\ell\end{pmatrix}$（含残差直连与归一化导数）。从 $S'_0=E'_\theta(x),\, P'_0=0$ 展开精确递推，末层文本总响应矩阵为：
$$
Z_L(x) = \sum_{k=0}^{L-1} C_{L-1:k+1}\,B_k\,A_{k-1:0},
$$
同族两数据包 $x, x'$ 的敏感度差则必包含雅可比跨包变化项 $Z_L(x)(E'_\theta(x)-E'_\theta(x')) + (Z_L(x)-Z_L(x'))E'_\theta(x')$。

- **显式反例证伪“仅靠 3:1 混合+残差范数 $\Theta(\sqrt{L})$ 必导致坍缩”**：令软态固定 $s=e_0$，文本残差初始为 $e_0$，$P_\perp=I-e_0e_0^\top$，在每层注入增量 $(\sqrt{\ell+2}-\sqrt{\ell+1})e_0 + b_\ell P_\perp N(s)$。所有文本残差范数严格满足 $\|h_{\ell,\mathrm{last}}\|=\sqrt{\ell+1}$，若在 3 个 GDN 层令 $b_\ell=0$、每第 4 层（FullAttn）允许微调符号，则切向总雅可比 $\big(\sum_{\ell<L}b_\ell\big)P_\perp DN(e_0)$ 保持秩 $d-1$ 且幅值可自由取 **`0`、常数或 $\Theta(L\epsilon)$**。
- **RMSNorm 精确导数**：对于 $N(h) = \Gamma h / s$（$s^2=\|h\|^2/d+\eta$），导数为：
  $$
  DN(h) = \frac{\Gamma}{s}\left(I - \frac{hh^\top}{ds^2}\right).
  $$
  当 $s=\Theta(\sqrt{L})$ 时，它先以 $O(s^{-3})$ 优先消除**唯一一条径向方向**，并将所有正交切向方向同尺度缩放 $1/s = O(L^{-1/2})$，本身并不压缩切向多维流形至秩 1。
- **逐层 Pre-RMSNorm 对强弱分支的影响**：若垂直传输块一致有界（$\|A\|\le M_S, \|C\|\le M_P$）且各层源软 token 均方根按 $\ge\sqrt{k+1}$ 增长（$B_k$ 在归一化输入上有界常数 $b$），则 $\|B_k\|\le b/\sqrt{k+1}$ 给出：
  $$
  \|Z_L\| \le M_S M_P b \sum_{k=0}^{L-1}(k+1)^{-1/2} \le M_S M_P b(2\sqrt{L}-1),
  $$
  叠加上末层 $O(L^{-1/2})$ 归一化即可得 **原始 $\Theta(\sqrt{L})$、归一化 $\Theta(1)$**（而单次浅层注入后仅靠恒等透传则会得到原始 $\Theta(1)$、归一化 $\Theta(L^{-1/2})$）。反之若缺乏垂直乘积有界性假设，逐层 Pre-RMSNorm 提供的 $I+O(k^{-1/2})$ 因子连乘仍可在特定方向上发散至 $\exp(\Theta(\sqrt{L}))$。此外，[`THEOREM_W1.md`](../stage4_math_and_design/weight_mod/W1/THEOREM_W1.md#L27-L55) 中宽 4 因果 `conv1d` 也允许深层前 3 个文本键在第 1 层起即直接携带软前缀痕迹。

#### 3ii. `last-soft` 与多深度残差抽头精确权衡
因果性保障各层末位软 token 恒满足 $\partial h_{\ell,\mathrm{soft}_K}/\partial X_{\rm prose} \equiv 0$。联合多个抽头 $H_i$ 时，堆叠雅可比严格满足：
$$
J_{\rm stack}^\top J_{\rm stack} = \sum_i J_i^\top J_i,\qquad \ker J_{\rm stack} = \bigcap_i \ker J_i,
$$
对无容量限制的总体头而言，向已有读出 $U$ 中增补新抽头 $T$ 所降低的最优风险严格等于条件互信息 **$I(Y; T\mid U)$**。然而它并不直接替代有限步 240-step 优化表现，真正决定因素仍需在 27B 同包/同模式下实测跨位置对比。

---

### 定理 C4：`MixKL` 双侧共模/差模运行时扰动定理与跨任务子空间失配

#### 4a. 固定参考 `sqrt(2D)*std` 反例、双侧共模乘性界与差模平方根界
记冻结 logits 分布为 $p$、适配后分布为 $q \propto p e^v$（基准 $D := D_{\mathrm{KL}}(p\|q)$），一般运行时漂移后冻结态与适配态分别受到对数偏移 $\delta_0$ 与 $\delta_1$，记共模偏移 $\delta_0$（振幅 $b_0=\operatorname{osc}\delta_0$）与差模偏移 $\Delta\delta = \delta_1 - \delta_0$（振幅 $b_\Delta = \operatorname{osc}\Delta\delta$）。

1. **反例证伪通用一阶因子 $\sqrt{2D}\operatorname{std}_q\delta$**：取 $p(A)=0.1, q(A)=0.001, \delta=t\mathbf{1}_A$，则 $D=0.366593$、$\sqrt{2D}\operatorname{std}_q\delta < 0.028$，而在 $t\downarrow 0$ 时一阶微扰斜率精确为 $|p(A)-q(A)| = 0.099$。正确的不依重尾奇异分布失效的 $L^2$ 系数是 $\sqrt{\chi^2(p\|q)}\operatorname{std}_q\delta$；仅依赖 $D$ 与振幅 $b$ 时（由 $\mathrm{TV}(p,q)\le\sqrt{D/2}$ 与二阶余项积分方差 $\le b^2/4$），紧界为 $|\Delta D| \le b\sqrt{D/2}+b^2/8$。
2. **核心新定理（纯共模平移全球严格消去 $O(\sqrt{D})$ 项，呈纯乘性收缩/扩张）**：
   设 $\Delta\delta = 0$，即冻结参考与适配模型共同承受同一分布偏移 $\delta_0$。由指数族对数配分函数 Taylor 积分余项：
   $$
   D = \int_0^1 (1-t)\operatorname{Var}_{p_t}(v)\,dt,\qquad D(p_{\delta_0}\|q_{\delta_0}) = \int_0^1 (1-t)\operatorname{Var}_{(p_t)_{\delta_0}}(v)\,dt,
   $$
   其中 $p_t \propto p e^{tv}$。因倾斜测度比 $(p_t)_{\delta_0}/p_t$ 全局夹在 $[e^{-b_0}, e^{b_0}]$ 之间，对变分形式 $\operatorname{Var}(v)=\inf_c \mathbb{E}(v-c)^2$ 逐点夹逼并积分，立即对**任意大或小的 $v,\delta_0$** 全局成立：
   $$
   \boxed{e^{-b_0} D \le D(p_{\delta_0}\|q_{\delta_0}) \le e^{b_0} D.}
   $$
   在稀少原子承载适配对比的二元极限下该因子可渐近达紧，绝对漂移不超过 $(e^{b_0}-1)D = O(b_0 D)$，彻底消除了全局 $O(\sqrt{D})$ 敏感度。
3. **一般双侧运行时认证定理**：先经共模边界至 $[e^{-b_0}D,\, e^{b_0}D]$，再对剩余差模 $\Delta\delta$ 双向施加固定参考平方根变分界，得双侧精确包络及多提示词 Minkowski 平均认证：
   $$
   \boxed{\left(e^{-b_0/2}\sqrt{D} - \frac{b_\Delta}{\sqrt{8}}\right)_+^2 \le D(p_{\delta_0}\|q_{\delta_1}) \le \left(e^{b_0/2}\sqrt{D} + \frac{b_\Delta}{\sqrt{8}}\right)^2,\qquad \sqrt{\overline{D_{\rm new}}} \le \sqrt{\overline{e^{b_0}D}} + \sqrt{\frac{\overline{b_\Delta^2}}{8}}.}
   $$
   同时在固定参考下可写无分布假设绝对包络 $U(d,b)=\sup_{\mathrm{kl}(s\|t)\le d,\, 0\le s,t\le 1}\{\log[1+(e^b-1)t]-bs\}$ 及显式训练目标 $\varepsilon_{\rm train} \le (\sqrt{\varepsilon_{\rm cert}} - b/\sqrt{8})^2$。
- **归档数值定量标定**：Modal 均值 `0.00091643`（[`cn_sft_4b_probe_results.json`](./results/cn_sft_4b_probe_results.json#L299)）漂移至 Colab 均值 `0.00106704`（[`stage4_real_sdr_4b_cnsft_results.json`](./results/stage4_real_sdr_4b_cnsft_results.json#L251)）的 `+0.00015061` 增量，在纯差模固定参考模型下仅需均方根振幅 **$b_{\Delta,\rm RMS} \ge 0.006768$**（注意在 logit 幅值 `1.0` 处一个 BF16 ULP 已为 `2^{-7} = 0.0078125`），而在纯共模下则需 $b_0 \ge \log(1.06704/0.91643) = 0.152155$。实际定位必须成对采样双平台冻结与适配 logits。

#### 4b. 自然文本零空间投影无法约束跨任务软前缀干扰
对受保护文本特征子空间 $U$ 构造正交投影矩阵 $P_\perp$ 及线性适配层 $L = W P_\perp$，虽能使 $U$ 内一阶特征漂移精确为零（平均二阶范数 $\operatorname{tr}(W P_\perp C P_\perp W^\top)$），但它对与 $U$ 正交的子空间完全不设防。极简反例取 $U=\operatorname{span}(e_1)$、$L=M e_2 e_2^\top$：文本输入 $e_1$ 零干扰（完美通过 `MixKL`），而正交新任务软 token $e_2$ 立刻承受高达 $M=\|L\|_2$ 的满幅冲击（具体物理影响取决于后继任务头是否刚好湮灭 $e_2$）。在 [`cn_sft_adapter.py`](../stage4_real_and_surgical/cn_sft_adapter.py#L1831) 与 [`real_sdr_and_stage6_gpu.py`](../stage4_real_and_surgical/real_sdr_and_stage6_gpu.py#L906) 中，Stage 6 仅置 `carrier=None` 但保留 `g_em=1` 与软 token 掩码，故 9B `U1` 出现 `MixKL=8.498e-4` 合格的同时在 Stage 6 跌落 `-18.99 pp` 完全自洽。最彻底的数学隔离即是跨任务推理时完整旁路旧适配钩子。

---

### 定理 C5：`S17 ⊂ C16` AdamW 条件数、严格轨迹包含与一步残差-SVD 解锁

#### 5a. 无条件“一步符号 SNR 必退化”的反例证伪与高斯对照
[`equi_connector.py`](../stage4_real_and_surgical/equi_connector.py#L265) 确认 136 维 C16 特征将 30 个连续尾比特列严格并列在 S17 核心特征之外。在预激活 $a=W_c x + W_t z$ 上首步偏差修正 AdamW 近似遵循 $\Delta w \approx -\eta\big(\frac{g}{|g|+\epsilon_A}+\lambda w\big)$，尾特征通过反传向量 $d=\partial\ell/\partial a$ 改变核心梯度 $dx^\top$。
- **反例证伪无条件退化**：取平方损失、全冗特征 $x=z=1$、核心初值 $w=0$、标签 $Y\sim\mathrm{Unif}\{-1,3\}$。S17 总体核心真梯度为 $-1$，引入常数尾偏置 $b$ 后样本梯度变为 $b-Y$：当 $b=0$ 时符号正确率 `0.5`，当 $b=-2$ 时符号正确率升至 **`1.0`**，当 $b=4$ 时降至 **`0.0`**，表明无尾分布与相关性假设时并无单边降阶律；反之对独立零均值高斯尾噪声 $N(0,\sigma^2)$（$x=Y=1, w=0$），符号正确率严格下降为 $\Phi(1/\sigma)$。

#### 5b. 捕捉全局梯度裁剪泄漏的严格轨迹包含定理与一步最优解锁
1. **严格轨迹包含定理**：注意 [`colab_stage6_scaling.py`](../stage4_real_and_surgical/colab_stage6_scaling.py#L738) 同时以 `seed + bits + len(mname)` 设置了初始化、小批次抽样与 `clip_grad_norm_(1.0)` 全局梯度裁剪。即使将门控初始化为 $\gamma(0)=0$，只要 $\gamma$ 处于可微且参与全局梯度范数计算的状态，第 1 步非零 $\nabla_\gamma L$ 就会修改全局裁剪分母，从而打破轨迹一致性！只有当：**在 warmup 期物理冻结 $\gamma \equiv 0$（从全局裁剪集合排除），并锁定核心权重初始值、优化器状态及全部数据随机种子**，前向输出与每步核心迭代才构成逐位全等的精确轨迹包含。
2. **构造性一步残差-SVD 解锁最优解**：在冻结 warmup 结束时刻计算残差-尾互协方差 $R := \sum_i u_i z_i^\top$（$u_i = \partial\ell_i/\partial e_i$），线性尾分支设为 $e_i = c_i + \gamma \odot V z_i$。若 $R \neq 0$，各门控约束下最大化初始解锁梯度范数 $\|\nabla_\gamma L\|$（同时最大化欧氏步 $-\eta g$ 瞬间下降量）的精确全局最优矩阵 $V$ 如下：

| 门控形式与矩阵约束 | 可达最大初始解锁梯度 | 精确最优初始化矩阵 $V$ |
|---|---|---|
| **标量 $\gamma$，$\|V\|_F=1$** | $\lvert\partial_\gamma L\rvert = \|R\|_F$ | **$V = \pm R / \|R\|_F$** |
| **标量 $\gamma$，$\|V\|_F=1$ 且 $\operatorname{rank}V \le k$** | $\|R_k\|_F = \big(\sum_{j\le k}\sigma_j(R)^2\big)^{1/2}$ | **归一化截断 SVD $V = \pm R_k / \|R_k\|_F$** |
| **向量 $\gamma$，逐行 $\|V_j\|_2=1$** | $\|\nabla_\gamma L\|_2 = \|R\|_F$ | 每非零行逐一对齐 $R_j$ 方向（正负号同可） |
| **向量 $\gamma$，仅总范数 $\|V\|_F=1$** | $\|\nabla_\gamma L\|_2 = \max_j \|R_j\|_2$ | 全部 Frobenius 范数集中到范数最大行对齐 $R_j$ |

对带零一阶矩 AdamW 步，其瞬时下降对应于在 Frobenius 预算下使 $\|V_j\|\propto \|R_j\|$ 的 $\|g\|_1$ 最大化，与欧氏集中单行解不同；因此 **Stage 6b 推荐直接采用无歧义的“标量门 $\gamma$ + 归一化 $R$（或截断 $R_k$）”**。

---

### 定理 C6：极坐标缓存 `m5-c16-v1-le` 三层误差、Hölder 校准界与 EVM 次-$\log 4$ 可辨识性

#### 6a. 三层误差分解与 Q2 循环相关 Top-1 翻转反例
针对 [`real_sdr_cpu_physical_report.json`](../stage4_real_and_surgical/results/real_sdr_cpu_physical_report.json#L254-L400) 上 CTTC 域内 `ref_two` 测试（1,280 样本 / 64 簇）：S17 `0.6953125` Acc / `0.7294025` NLL vs C16 `0.6921875` Acc / `1.7322821` NLL（簇配对差 `+1.0028796 nats`, $p=3.6559\times 10^{-7}$），严格拆分三层数学机理：
1. **连续复高斯密度超额**：若残差满足真实分散 $\mathbb{E}|T-\mu_Y|^2 = v_{\rm eff}$ 而评分假设方差 $\hat{v}$（$\beta=v_{\rm eff}/\hat{v}$），即使真实分布非高斯，精确密度交叉熵超额恒为 $\beta - 1 - \log\beta$；当真值仅知落于 $[v_-, v_+]$ 时，极小极大最优方差为**对数均值** $\hat{v}_* = (v_+-v_-)/\log(v_+/v_-)$。
2. **量化后验链式法则与代码机制**：同先验、同量化 $W$ 下的条件后验散度满足 $\mathbb{E}_{P_W} D_{\mathrm{KL}}(P_{Y\mid W}\|Q_{Y\mid W}) = D_{\mathrm{KL}}(P_{Y,W}\|Q_{Y,W}) - D_{\mathrm{KL}}(P_W\|Q_W)$。而在 [`sdr_real_pipeline.py`](../stage4_real_and_surgical/sdr_real_pipeline.py#L1589-L1733) 中，C16 不仅将幅值缩放为 $\propto\sqrt{v_{\rm nom}}$（改变 $v_{\rm nom}$ 使未截断处 logits 按平方根倒数放缩），还将精度截断至 `[0.05, 50]` 且未读取 S17 所查的旧内建索引位。
3. **末位 logits 放缩 vs 窗口 logits 放缩（Q2 Top-1 翻转反例）**：对**末位 Q2 概率向量**加正温度严格保序 Top-1（且对实测错分边距 $M=\max_j s_j - s_Y$ 满足紧夹逼 $\beta M \le -\log p_\beta(Y) \le \beta M + \log 4$）。但在导频循环卷积 $q_c = \sum_k b_k a_{k+c}$ 中缩放窗口 logits 却**会改变 Top-1 类别**：令窗口 logits $s=(1,-0.9,-1,0.9)$ 且导频 $b=(0.51,0.47,0.01,0.01)$，极小温度 $\beta\downarrow 0$ 时一阶系数给出 $c=3$（`0.91 > 0.086`）胜出，而 $\beta\to\infty$ 时窗口坍缩到类 `0` 使 $c=0$ 胜出。

#### 6b. 任意温度 Hölder 校准界与三项汇总指标的次-$\log 4$ 可辨识性分裂定理
对末位截断后分布 $s_k = \max(p_k, c)$（$c=10^{-12}$）施加幂次温度 $\alpha = 1/T \in (0,1]$ 得 $q_k = s_k^\alpha / \sum_j s_j^\alpha$，由 Hölder 不等式 $\sum_j s_j^\alpha \le 4^{1-\alpha}(1+3c)^\alpha$ 立得任意温度校准保证：
$$
\boxed{L_T \le \frac{1}{T} L + \left(1-\frac{1}{T}\right)\log 4 + \frac{1}{T}\log(1+3c).}
$$
代入归档值 $L = 1.7322820933 > \log 4 = 1.3862943611$：$T=2$ 时严格小于 **`1.559289`**（保序降至少 `0.17299 nats`）；当取大有限温度 $T$ 时，在严格保留 **100% 原始 Top-1 决策（69.22%）** 的前提下，可保证 NLL 降至任意逼近 $\log 4$（保证改善高达 **`0.345988 nats`**）。

> **核心可辨识性分裂定理**：同时利用归档的三项汇总指标（[`acc = 0.6921875`, `nll = 1.7322820933`, `posterior_rms_evm = 0.7089112379`](../stage4_real_and_surgical/results/real_sdr_cpu_physical_report.json#L307)）会引发如下精确分岔：
> 1. **温度缩放无法通过三项指标认证严格低于 $\log 4$（1,280 行显式反例）**：取全部真类为 `0`（或任意循环置乱）、后验形态 $(x,a,b,a)$（$a=(1-x-b)/2$，单行平方 EVM 为 $(1-x+b)^2$）的三类样本构成 1,280 行数据：
>    | 行数 $n_i$ | 真类概率 $x_i$ | 相邻误类概率 $a_i$ | 对跖误类概率 $b_i$ | 计入正确率 |
>    |---:|---:|---:|---:|---|
>    | `886` | `0.47` | `0.212582290873` | `0.104835418254` | 正确 (`886/1280 = 0.6921875`) |
>    | `344` | `0.29` | `0.305` | `0.1` | 错 (`a > x`) |
>    | `50` | `1.77713959737e-10` | `0.449999999911` | `0.1` | 错 (`a > x`) |
>
>    所有概率均严格大于 $10^{-12}$ 且精确复现全部三项归档指标，但凸损失函数 $L(\alpha)$ 在 $\alpha=0$ 处的右导数满足 **$L'(0) = \mathbb{E}[\frac{1}{4}\sum_k\log p_k - \log p_Y] = +0.0206776533 > 0$**，故**任意有限正温度在该构造上的 NLL 均严格高于 $\log 4$**！
> 2. **然而同一 `posterior_rms_evm` 值可严格认证“均匀混合校准”突破 $\log 4$！**：令归一化分布 $p^\circ = s/\sum_j s_j$，旋转真类到 `1`，在 [`_score_probs`](../stage4_real_and_surgical/sdr_real_pipeline.py#L1692-L1695) 的实部上用 Cauchy–Schwarz 得 $\mathbb{E}[s_Y - s_{Y+2}] \ge 1-e$，于是真类期望均值下界为 $m := (1-e)/(1+3c) = 0.2910887621 > 1/4$。对严格 $C_4$-等变且 **100% 保 Top-1** 的概率混合 $q = (1-\lambda)U + \lambda p^\circ$ 利用凸函数 $-\log((1-\lambda)/4+\lambda x)$ 在 $[0,1]$ 上的弦线上界：
>    $$
>    \boxed{\lambda_* = \frac{4m-1}{3} = 0.0547850161 \implies L(q) \le \log 4 - d(m\,\|\,1/4) = 1.381942264 < \log 4.}
>    $$
> 此外，因普遍支配 S17 在任意数据下不可能成立（固定输入 S17 与真值为 $(0.7,0.1,0.1,0.1)$ 而 C16 为均匀 logits 时），严谨的工程回退只能基于**含 S17 端点的族** $(1-\lambda)p_{\rm S17}+\lambda p_{{\rm C16},T}$ 并采用独立验证集上的**簇级 Hoeffding 回退判定**：$n$ 个有界于 $[0,B]$ 的独立验证簇、$m$ 个候选下，仅当实测相对 S17 均差加上 $B\sqrt{2\log(m/\delta)/n} \le 0$ 时才切换至 C16 校准版。

---

### 定理 C7：双态表示损失与受限读出头 $\mathcal{V}$-信息勾股定理

记细粒度数据包（含公开上下文）为 $W$，读出或压缩表示为 $T=f(W)$，教师或真后验为 $p(W)$，条件期望教师为 $\bar{p}(T) := \mathbb{E}[p(W)\mid T]$。

1. **态一：离散粗化（如 `S17 ⊂ C16`）下的主动揭示风险差恒等式**：将同物理实验潜在揭示耦合在同一概率空间上，粗表示 $T$ 与细包 $W$ 的最优单次揭示二元对数风险之差精确拆分如下（加减 $H(H\mid W, Z_{a_T})$ 即证）：
   $$
   \boxed{R_T^* - R_W^* = I(H; W \mid T, Z_{a_T}) + \mathbb{E}_W\big[I_W(H; Z_{a_W}) - I_W(H; Z_{a_T})\big] \ge 0,}
   $$
   即严格拆为“(i) 固定在粗选动作上所丢失的精细解码信息”与“(ii) 粗表示丧失的最优行动切换收益”。
2. **态二：受限读出头类 $\mathcal{V}$ 上的总误差分解与指数族勾股定理**：对任意候选读出头族 $\mathcal{V}$ 均精确成立：
   $$
   \boxed{\inf_{q\in\mathcal{V}}\mathbb{E}D_{\mathrm{KL}}\big(p(W)\,\|\,q(T)\big) = \underbrace{\mathbb{E}D_{\mathrm{KL}}\big(p(W)\,\|\,\bar{p}(T)\big)}_{\text{抽头本身的信息损失}} + \underbrace{\inf_{q\in\mathcal{V}}\mathbb{E}D_{\mathrm{KL}}\big(\bar{p}(T)\,\|\,q(T)\big)}_{\text{受限头函数类 }\mathcal{V}\text{ 的逼近误差}}.}
   $$
   - **连续实解析冻结宿主与有限精度碰撞**：对实算术下的冻结连续神经网络 $f$，只要任两不同输入包存在能区分它们的参数设置，碰撞集合位于有限个真解析零点集的并中，从而泛型单射成立，即**香农损失 $\mathbb{E}D_{\mathrm{KL}}(p(W)\|\bar{p}(T)) \equiv 0$**！在实机部署中导致第一项非零的机制是 **BF16 / NF4 舍入导致的输出向量精确碰撞且对应碰撞包具备不同教师权重**。
   - **对数线性头 $\mathcal{V}_{\rm lin}$ 精确三-KL 勾股恒等式**：令 $q_B(j\mid t)=\mathrm{softmax}(B\phi(t))_j$（$\phi$ 含截距项）并记 $B_*$ 为无正则总体极小点，由一阶矩匹配驻点条件 $\mathbb{E}[(q_{B_*}-\bar{p})\phi^\top]=0$ 结合指数族对数比线性结构，严格成立条件 $\mathcal{V}$-信息勾股定理（参见 [`arXiv:2002.10689`](https://arxiv.org/abs/2002.10689)）：
     $$
     \mathbb{E}D_{\mathrm{KL}}(\bar{p}\,\|\,q_B) = \mathbb{E}D_{\mathrm{KL}}(\bar{p}\,\|\,q_{B_*}) + \mathbb{E}D_{\mathrm{KL}}(q_{B_*}\,\|\,q_B).
     $$
   - **香农无损 vs 头类失效三点反例**：令 $T\in\{-1,0,1\}$ 取质 $(1/4, 1/2, 1/4)$，对应二元教师概率 $(0.9, 0.1, 0.9)$，且 $W=T$。此时表示的**香农损失严格为 0**，但**任意仿射-softmax 头因对称性必然全部坍缩到常数 `0.5`**，而一个 **2 单元偶 `GELU` 头**（$g(t)+g(-t)$ 加偏置）即可实现**零误差精确拟合**。

---

## 三、面向 Stage 6b / Stage 7 的六项可落地数学优化优先级

| 优先级 | 数学优化方案 | 实测对照与可证伪判据 |
|---|---|---|
| **Priority 1** | **选择/揭示解耦审计 + 无偏单样本 $\widehat{L}_{\rm avg}$ 损失训练** | 分别记录 `S_oracle(a_q)` 与 `B_{a_q}(p,q)`，以及各动作 $J_{0..3}$；若训练 $\widehat{L}_{\rm avg}$ 在同一 `random/nearest` 动作与 `max_EIG` 上均稳定压缩 $J_a$ 与超额损失，即坐实原 16 维 KL 训练的代理失配。 |
| **Priority 2** | **27B 多深度/多位置抽头与三级 $\mathcal{V}$-信息分级定位** | 同包、同模式（并对齐 `eval` vs `train+grad_ckpt`）下测试 `last-soft` 与 `last-prose` 中层/末层读出，同步统计精确 BF16 碰撞组以及**常数 / 仿射-softmax / 宽-96 GELU**三类头损，并在训练前直接隔离“精度碰撞 vs 头表达受限 vs 检查点后端梯度畸变”。 |
| **Priority 3** | **冻结 warmup 的严格 `S17 ⊂ C16` 对照与标量门+残差-SVD 解锁** | 固定核心初始值、随机种子与数据流，warmup 期冻结 $\gamma=0$ 并移出 `clip_grad_norm_(1.0)` 验证轨迹完全重合后，按 $V=\pm R/\|R\|_F$（或截断 SVD $R_k$）解锁，彻底解决“双种子偏移 vs 尾特征条件数干扰”之争。 |
| **Priority 4** | **Q2 末位保序校准（含均匀混合）+ 簇级 Hoeffding 安全回退** | 在独立切分簇上对比末位温度 $T$、均匀混合 $\lambda$ 与利用 S17 内建索引的增强解码器，仅在实测赢出且满足独立簇 Hoeffding 裕量时超越/替代 S17。 |
| **Priority 5** | **全激活拼接基线与全混沌 Gram `GELU` 压缩 (`C=HK^\dagger`)** | 在同输入上评估最多 22 路并列激活 `GELU` 单元作为零压缩误差上限对照，使用联合一+二阶谱/1540 字典枚举初始化与经验 Gram 投影 $C=HK^\dagger$ 取代 `CA` SVD 合并，同时单独定量核验 Cayley 生成元截断误差。 |
| **Priority 6** | **Stage 6 严格解耦 Stage 4 钩子 + 双侧 `(b_0, b_\Delta)` MixKL 裕量证书** | 同一已训 Stage 6 头下切换 `off / attached-zero / U1` 立刻隔离前向干扰 vs 重训偏移；收集跨平台成对冻结/适配 logits 并按 $\sqrt{\overline{e^{b_0}D}}+\sqrt{\overline{b_\Delta^2}/8}$ 为 `1e-3` 阈值出具可验证安全裕量。 |
