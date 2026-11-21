---
title: "全局内容寻址何时严格优于固定局部窗口"
subtitle: "EM-FloatFormer · 第一条数学研究"
date: "2026 年 11 月 21 日"
---

# 1　结论

在下述等边际复高斯模型中，若可观测内容把足够多的权重分给有效参考，单头注意力能在平方损失上严格优于**所有**固定 $k<M$ 个参考值的可测估计器；已知匹配似然的经典全局估计器能达到同一 Bayes 最优值。

# 2　符号、联合分布与信息集

| 符号 | 分布、含义与可见性 |
|:--|:--|
| $M,k,S$ | $M\ge2$，$0\le k\le M$。候选来自**同一条链路**。$S\subseteq\{1,\ldots,M\}$ 和大小 $k$ 在读取这条链路之前固定。允许用独立于整条链路的随机数预选，证明再对它条件化。 |
| $\theta,J,\varepsilon$ | 独立抽取 $\theta\sim\mathcal{CN}(0,v_0)$、均匀索引 $J$ 和 $\varepsilon\sim\mathcal{CN}(0,\sigma^2)$，$v_0,\sigma^2>0$。$\theta$ 是待估计的复标量。$J$ 只在生成、证明和离线评测可见。 |
| $u,V$ | $u=v_0+\sigma^2$。另抽 $M$ 个独立的 $U_j\sim\mathcal{CN}(0,u)$，与前三者独立；令 $V_J=\theta+\varepsilon$，其余 $V_j=U_j$。复高斯均指 proper 且圆对称，实虚部独立、各有一半方差，$\mathbb E\lvert Z\rvert^2=v$。 |
| $T=(Q,K)$ | 给定 $J=j$，按任意核 $\mathcal L(T\mid J=j)$ 用另一个独立随机源生成。等价地，给定 $J$，$T$ 与 $(\theta,\varepsilon,U_{1:M})$ 条件独立。$T$ 能携带位置匹配信息，单独观察它时 $\mathbb E[\theta\mid T]=0$。 |
| $\lambda,v_1,D$ | $\lambda=v_0/u$，$v_1=v_0\sigma^2/u$，$D=v_0-v_1=\lambda^2 u=v_0^2/u>0$。$D$ 是已知 $J$ 并读到有效值所能消除的方差。 |
| 信息与损失 | 强化的固定局部类使用 $\mathcal F_S=\sigma(T,V_S)$，拥有所有 keys，仅读取 $S$ 中的值。全局类使用 $\mathcal F_G=\sigma(T,V_{1:M})$。每类允许该信息下**任意可测**复估计，风险 $R(g)=\mathbb E\lvert\theta-g\rvert^2$。无穷风险自然满足下界。 |

严格本地的接口只提供 $Q,K_S,V_S$，其函数类包含于强化局部类，因此同一下界也适用。后面另给它自己的最优解。模型权重、训练库或缓存均不得含这次试验隐藏的 $J,\theta$ 或参考值；用已有训练数据定好参数可先条件于这些独立数据。访问多少 keys、多少 values 和计算开销分别计量。若多层局部网络最终能跨窗传递远端状态，应按最终有效感受野计入全局访问。

# 3　主定理与证明

## 3.1　固定局部估计器的下界

**命题 A。** 给定任意上述 $S$ 和任意局部估计 $g$，

$$
R(g)\ \ge\ L_k:=\frac{k}{M}v_1+\left(1-\frac{k}{M}\right)v_0
=v_0-D\,k/M .
\tag{1}
$$

**证明。** 条件于 $J=j$，有效参考的似然给出
$$
\frac{|\theta|^2}{v_0}+\frac{|V_j-\theta|^2}{\sigma^2}
=\frac{|\theta-\lambda V_j|^2}{v_1}+\frac{|V_j|^2}{u}.
$$
故 $\theta\mid(J=j,V_j)\sim\mathcal{CN}(\lambda V_j,v_1)$。给定 $J=j$，其他 $V_i$ 与 $(\theta,\varepsilon)$ 独立，$T$ 又与整个这组随机量独立。向局部估计器额外透露 $J$ 后，若 $j\in S$，条件方差就是 $v_1$；若 $j\notin S$，所有可读值及 $T$ 对 $\theta$ 不提供信息，条件方差为 $v_0$。平方损失下，条件均值 $m=\mathbb E[\theta\mid\mathcal F]$ 最优，因为
$$
\mathbb E|\theta-g|^2=\mathbb E|\theta-m|^2+\mathbb E|m-g|^2
$$
中交叉项的条件期望为零。辅助估计器的最小风险依全概率公式恰为 $L_k$，其中 $\Pr(J\in S)=k/M$。真实局部类的信息包含于辅助类，故其风险至少为 $L_k$。证毕。

## 3.2　可观测打分的精确风险与充分条件

**命题 B。** 取任意有限实分数 $s_j(T)$，$\tau>0$。令
$$
\alpha_j=\frac{\exp(s_j/\tau)}{\sum_{\ell=1}^M\exp(s_\ell/\tau)},\qquad
\widehat\theta_\alpha=\lambda\sum_{j=1}^M\alpha_j V_j .
$$
设 $e_J$ 为第 $J$ 个标准基向量（仅用于证明）。对任何 $T$-可测的概率权重都成立
$$
R_\alpha=v_1+D\,\mathbb E\|e_J-\alpha\|_2^2
=v_1+D\,\mathbb E\left[(1-\alpha_J)^2+\sum_{j\ne J}\alpha_j^2\right].
\tag{2}
$$

**证明。** 写 $\xi=\theta-\lambda V_J$。由配方可知，在给定 $J$ 时，$\xi$ 的条件分布不依赖 $V_J$，其均值为零、方差为 $v_1$；它也独立于干扰值。给定 $(T,J)$ 后，这些结论保持成立，各分量 $V_i$ 仍是独立的零均值 $\mathcal{CN}(0,u)$，且 $\alpha$ 已固定。误差为 $\xi+\lambda(e_J-\alpha)^\top V$。前后两项的交叉矩为零；第二项展开时 $i\ne j$ 的乘积条件期望也为零，每个对角项贡献 $\lambda^2u(e_{J,i}-\alpha_i)^2$。对 $(T,J)$ 取期望即得 (2)。

**间隔上界。** 设 $G=\{s_J-\max_{j\ne J}s_j\ge\Delta\}$，$\Delta>0$，$\Pr(G)\ge1-\eta$，$0\le\eta\le1$。在 $G$ 上，
$$
\sum_{j\ne J}e^{(s_j-s_J)/\tau}\le b=(M-1)e^{-\Delta/\tau},
\qquad 1-\alpha_J\le r=\frac b{1+b}.
$$
因权重非负，$\sum_{j\ne J}\alpha_j^2\le(\sum_{j\ne J}\alpha_j)^2=(1-\alpha_J)^2$。故 (2) 中的括号在 $G$ 上至多为 $2r^2$，在 $G^c$ 上至多为 $2$。按事件分别积分，并用 $r^2\le1$，得
$$
R_\alpha\le U:=v_1+2D[(1-\eta)r^2+\eta].
\tag{3}
$$
若
$$
2[(1-\eta)r^2+\eta]<1-k/M ,
\tag{4}
$$
则 $R_\alpha\le U<L_k\le R(g)$ 对每一个固定局部估计器成立。(4) 是充分条件；界过松或条件不满足时，仍可直接比较精确风险。

## 3.3　所有局部函数的最优解，以及相同访问量的对照

设 $p_j=\Pr(J=j\mid T)$。给定 $(J,T)$，$V$ 的联合边际密度始终是 $M$ 个 $\mathcal{CN}(0,u)$ 的乘积，故 $V\perp(J,T)$。这不意味着 $\theta$ 与 $V$ 独立。给定 $(T,V_S)$ 后，$J$ 的后验仍为 $p$；逐个对 $J$ 的条件均值加权得
$$
m_S=\lambda\sum_{j\in S}p_j V_j,\qquad
R_S^*=v_0-D\,\mathbb E\sum_{j\in S}p_j^2 .
\tag{5}
$$
这里缺失位置对条件均值贡献零，$S$ 内的后验权重**不重新归一化**。第二式由正交恒等式 $R_S^*=\mathbb E|\theta|^2-\mathbb E|m_S|^2$ 及 $V$ 的等方差对角二阶矩得到。取 $S$ 为全部索引即得 $m_G,R_G^*$。特别地，$\sum_{j\in S}\mathbb Ep_j^2\le\sum_{j\in S}\mathbb Ep_j=k/M$，再次得到 (1)。

由相同正交分解，$R_\alpha-R_G^*=D\,\mathbb E\|\alpha-p\|_2^2$。因此对于指定 $S$，**实际局部最优风险**被严格击败的充要条件为
$$
\mathbb E\|\alpha-p\|_2^2 < \mathbb E\sum_{j\notin S}p_j^2 .
\tag{6}
$$
若须同时击败任意事先选择的 $k$ 元集合，取右侧对这些集合的最小值。由于 $\mathbb Ep_j=1/M$，还有
$$
R_S^*-R_G^*
=D\left[\frac{M-k}{M^2}+\sum_{j\notin S}\operatorname{Var}(p_j)\right]>0
\quad(k<M).
\tag{7}
$$
第一项在索引完全不可辨认时就存在，代表多读数值的收益。把全局方法固定为均匀平均，可得 $R_{\rm unif}=v_0-D/M$；与同读 $M$ 个值的全局 Bayes 相比，它额外付出 $D\sum_j\operatorname{Var}(p_j)$。该项衡量在相同访问预算下识别内容的收益。$J$ 可由 $T$ 唯一判定时，$p=e_J$，$R_G^*=v_1$ 且 $R_S^*=L_k$。

## 3.4　将目标解释为公共导频后的创新

设公共信息为 $P$。以 $H\sim\mathcal{CN}(m_0,v_H)$、$P=aH+n$、$n\sim\mathcal{CN}(0,\nu)$ 且独立为例，$\nu,v_H>0$。展开 $|H-m_0|^2/v_H+|P-aH|^2/\nu$ 并配方得
$$
m(P)=m_0+\frac{v_H a^*}{|a|^2v_H+\nu}(P-am_0),\quad
v(P)=\frac{v_H\nu}{|a|^2v_H+\nu}.
$$
定义 $\theta=H-m(P)$。其给定 $P$ 的分布为 $\mathcal{CN}(0,v(P))$，此例中 $v(P)$ 为常数，所以创新独立于 $P$。用 $m(P)+\widehat\theta$ 估计 $H$ 的损失逐样本恰为创新损失。

一般情形需要额外的**条件生成假设**。取 $m(P)=\mathbb E[H\mid P]$，$\theta=H-m(P)$。对几乎处处的 $P=z$，设 $\Pr(J=j\mid P=z)=\pi_j(z)$。给定 $(P=z,J=j)$，要求 $\theta\sim\mathcal{CN}(0,v(z))$、$\varepsilon\sim\mathcal{CN}(0,\sigma^2(z))$ 和各 $V_i\sim\mathcal{CN}(0,u(z))$（$i\ne j$）相互独立，$u(z)=v(z)+\sigma^2(z)$，$V_j=\theta+\varepsilon$；打分观测用独立随机源生成。这也保证 $\theta$ 给定 $P$ 后与 $J$ 独立。估计器知道 $P$ 及相应的模型系数。逐个 $z$ 应用 3.1–3.2 的配方，得 $\lambda(z)=v(z)/u(z)$，$D(z)=v(z)^2/u(z)$，$v_1(z)=v(z)-D(z)$。再用迭代期望，
$$
L=\mathbb E\!\left[v(P)-D(P)\sum_{j\in S}\pi_j(P)\right],\qquad
R_\alpha=\mathbb E\!\left[v_1(P)+D(P)\|e_J-\alpha\|_2^2\right].
$$
在每层代入 (3) 的 $\Delta(P),\eta(P),r(P)$ 后比较两个积分即可。只有相应条件先验均匀或满足相应加权命中恒等式，才能把命中项化为 $k/M$。从每个原始 helper 减去同一个 $m(P)$ 并不会自动形成这种新模型。例如原本独立于 $P$ 的零均值干扰在相减后的条件均值是 $-m(P)$。同一导频引入的相关性也须重新检验。

# 4　从带噪指纹得到可观测分数

## 4.1　随机二元指纹、温度与计算量

每条链路独立抽取 $M$ 个长度 $d$ 的均匀随机符号串 $C_j\in\{-1,+1\}^d$，允许完全重复。取独立的翻转符号 $B_\ell$，$\Pr(B_\ell=-1)=\rho$。令 $K_j=C_j/\sqrt d$，$Q=(C_J\odot B)/\sqrt d$。指纹、翻转和信道数值均由独立随机源产生。推断时只提供带噪 $Q$、各 $K_j$ 和获准读取的值。相同索引在不同链路有不同指纹；接收器只能比较内容。

令 $H_j$ 为 Hamming 距离，$s_j=Q^\top K_j=1-2H_j/d$。给定 $J,Q,H_J$，每个干扰指纹仍独立均匀，故其 $H_j$ 相互独立且为 $\operatorname{Bin}(d,1/2)$，条件分布不随 $Q,H_J$ 改变。有效距离 $H_J\sim\operatorname{Bin}(d,\rho)$。设 $X,Y$ 分别具有这两个分布，$g=\lceil d\Delta/2\rceil$。正间隔事件需要所有干扰的距离至少比有效距离大 $g$，于是
$$
1-\eta=\sum_{h=0}^{d}\Pr(X=h)\Pr(Y\ge h+g)^{M-1}.
\tag{8}
$$
相等距离导致正间隔失败；有限和已包含指纹碰撞。$\rho$ 增大会使 $X$ 随机增大，从而降低 (8)。任一干扰串等于观测 query 即破坏正间隔，所以 $1-\eta\le(1-2^{-d})^{M-1}$，固定 $d$ 时趋于零。即使已保证 $\Delta$，$r$ 本身仍随 $M$ 增大。

也可用显式维度界。记 $\mu=1-2\rho$，$0<\Delta<\mu$，$t=(\mu-\Delta)/2$。若 $s_J\ge\mu-t$ 且各干扰 $s_j\le t$，则间隔至少 $\Delta$。每个分数是 $d$ 个取值于 $[-1,1]$ 的独立变量之平均，有效分数均值 $\mu$，干扰均值为零。对一个下尾、$M-1$ 个上尾分别用 Hoeffding 不等式与并集界，得
$$
\eta\le \min\{1,\, M e^{-d(\mu-\Delta)^2/8}\}.
$$
因此为使该界至多为 $\eta_0\in(0,1)$，$d\ge 8\log(M/\eta_0)/(\mu-\Delta)^2$ 足够；这里也明确了为保持可辨认性所付的宽度代价。

在 $0<\rho<1/2$ 时，均匀先验下 $p_j\propto(1-\rho)^{d-H_j}\rho^{H_j}$。令 $\beta=(d/2)\log((1-\rho)/\rho)$，即有 $p=\operatorname{softmax}(\beta s)$。实现取一个 query、$M$ 个 keys、二维实 value $(\Re V_j,\Im V_j)$、跨 $M$ 个位置的 softmax、温度 $\tau=1/\beta$，最后以 $\lambda$ 乘两路输出并组回复数。若采用标准内积除以 $\sqrt d$ 的接口，可送入 $Q'=\sqrt d\, Q/\tau$ 和原来的 $K$，使最终 logits 恰为 $Q^\top K_j/\tau$。无需位置 one-hot、训练权重或多头。经典枚举二项似然后验平均逐样本给出同一输出。$\rho=1/2$ 时可用零 logits 实现均匀权重。$\rho=0$ 时精确后验在所有零距离上均分，可用匹配掩码，或取零温极限；这里不将无穷 logits 当作有限温度实现。

**一个完全可手算的严格例子。** 另取 $M=2,k=1,d=1$，在每条链路随机打乱一对相反的标记 $+1,-1$ 作两把 key；$Q$ 以 $1/20$ 概率翻转所选标记。给定 $J$ 后这些观测与信道数值独立。$s_j=QK_j$，取 $\tau=2/\log19$。观察者只看两次乘积，后验给匹配者 $19/20$ 的权重。取 $\Delta=2$，得 $\eta=r=1/20$。式 (4) 左侧为 $419/4000<1/2$。实际两种最优风险是 $R_G^*=v_0-(181/200)D$ 和 $R_S^*=v_0-(181/400)D$；这些数来自 $(19/20)^2+(1/20)^2=181/200$ 及两索引对称。这个例子无需随机模拟即可复算严格性。

## 4.2　信息范围、可学习参数与主要运算

以下都是**每个 query**的推断成本。候选池为当前链路固定的 $M$ 条参考，无外部检索库；$v_0,\sigma^2,\rho$ 为已知生成参数，使用到的 $\lambda,\beta$ 或温度是常数。表中构造的可训练参数均为零。手工写出的恒等投影不必存为稠密矩阵；若改用一般可学习的 $d\times d$ 投影，每个矩阵有 $d^2$ 个参数，其运算也需另计。

| 方法 | 可见内容与读取量 | 主要运算 |
|:--|:--|:--|
| 严格固定局部 Bayes | $Q$、$k$ 个 keys、$k$ 个 values | $k$ 次 $d$ 维匹配，共 $O(kd+k)$。缺失索引的似然被积分掉。 |
| 强化固定局部 Bayes，式 (5) | 全部 $M$ 个 keys、固定 $k$ 个 values | $O(Md+k)$。与全局法同付打分成本。 |
| 全局均匀平均 | $M$ 个 values，无需 keys | $O(M)$。用于分离内容识别的贡献。 |
| 全局非学习硬匹配 | 扫 $M$ 个 keys，读获胜的一个 value | Hamming 最近邻，即此信道的最大似然检索，固定索引顺序破平局。$O(Md)$。 |
| 枚举全局 Bayes | $M$ 个 keys 和 $M$ 个 values | 枚举 $M$ 个 $J$ 的似然，归一化后加权，$O(Md+M)$。 |
| 单头 attention | 同枚举 Bayes，可单独设置温度 | 同为 $O(Md+M)$。每条 key 的宽度为 $d$，每个 value 宽度为 2。 |
| 自适应选 $k$ 个值 | 先读 $M$ 个 keys，再读选中的 $k$ 个 values | 后验打分 $O(Md)$；实现用排序 $O(M\log M)$，线性选择也可；聚合 $O(k)$。 |

若保存整个池，$M$ 个指纹占 $Md$ 个比特（解包为浮点则占 $Md$ 个实数），另有 $M$ 个复数值。一条 query 的 attention 无须 $M\times M$ 分数矩阵。允许预建索引、额外历史样本或计算预算变化时，应另报建库与查询代价。

严格本地公式可直接审查。独立指纹模型中，对每个 $j\in S$，已读指纹的似然比为 $L_j=2^d(1-\rho)^{d-H_j}\rho^{H_j}$；对每个未读索引，积分掉其均匀指纹后 query 均匀，似然比为 1。故以 $L_j/(M-k+\sum_{i\in S}L_i)$ 替代 (5) 中的 $p_j$ 就是该接口的精确后验系数。数值实验选用更强的、能看所有 keys 的局部 Bayes。

# 5　反例、失效边界与未解决部分

1. **$k=M$ 或参考极弱。** 全局和固定局部的最优风险相同，式 (4) 不可能成立。固定 $v_0$，若 $\sigma^2\to\infty$ 则 $D\to0$，即使完美识别，最优差距也至多为 $D$。有限噪声可有严格但任意小的收益。若进一步把有效值改为与 $\theta$ 独立，风险完全相等。
2. **无法识别 $J$。** $T\perp J$ 时，$p_j=1/M$。全局软平均风险为 $v_0-D/M$，强化局部最优为 $v_0-kD/M^2$。小幅严格差距全部来自多读 values。随意硬选一个索引的风险为 $v_1+2D(1-1/M)$。这也是式 (4) 未成立却仍有全局最优优势的反例。
3. **高置信错配。** 若概率 $e$ 的事件上某个错误位置获得至少 $1-\delta$ 的权重，(2) 给出 $R_\alpha\ge v_1+2D e(1-\delta)^2$。例如使用两把相反的 key、无翻转、故意把打分符号反转，使错误位置获 $3/4$ 权重（有限温度 $\tau=2/\log3$）。风险为 $v_0+D/8$，比恒映射零还差。
4. **过冷与过热。** 上述相反标记若以 $0.45$ 概率翻转，匹配方后验为 $0.55$。取零温极限硬选匹配者，风险为 $v_0-0.10D$；固定 $k=1$ 的局部最优是 $v_0-0.2525D$；校准全局风险为 $v_0-0.505D$。一般 softmax 零温时在并列最高分上均分。另一端，给定完美辨认的不同标记，$\tau\to\infty$ 使有界打分产生均匀权重。当 $1<k<M$ 时，它比固定局部最优高 $D(k-1)/M$。两端严格差距由连续性在相应有限温度也成立。
5. **候选太多。** (8) 给出固定宽度时的正间隔损失。即便无翻转，重复指纹也会带来不可区分的索引。只声称“有一个有效参考”不足以保证某个给定温度有效。
6. **允许内容自适应选窗。** 若 $A=A(T)$，$|A|=k$，每个给定 $T$ 上重复 (5) 的证明，得 $R_A^*=v_0-D\mathbb E\sum_{j\in A}p_j^2$，真实命中率为 $\mathbb E\sum_{j\in A}p_j$。任意子集最佳选 $p_j$ 最大的 $k$ 项；若要求连续窗，则在合法窗中最大化 $\sum p_j^2$。完美可辨时通常只读一个 value 就达到 $v_1$（连续窗也可覆盖该位置）。若选窗前已扫描所有 values，访问预算还须计入这些读取。
7. **未指定噪声与索引的联合独立性。** 单独给 $\varepsilon\perp\theta$ 和 $J\perp\theta$ 不充分。例取 $M=2$，先独立抽 $\theta,\varepsilon$，再以 $|\varepsilon|^2\le\sigma^2\log2$ 定义 $J=1$，否则 $J=2$，其余值独立按原边际生成。$J$ 均匀且可由一对独立于信道的随机相反标记和相应 query 完美解码。对 $S=\{1\}$，命中时噪声方差为 $\sigma^2(1-\log 2)$（单位指数分布下半段的均值），仍与 $\theta$ 独立。仅按该方差作线性收缩就严格低于 $v_1$，未命中用零。总风险严格低于 (1)。因此第 2 节明确给出了完整抽样顺序。

## 5.1　同一带噪观测既用于打分又用于求值

保持 $(\theta,J,V)$ 的原分布，但令打分信息 $T$ 含有与数值相关的统计量。写 $w=e_J-\alpha$、$\xi=\theta-\lambda V_J$，定义条件二阶矩的偏离
$$
\Gamma=\mathbb E[VV^{\mathrm H}\mid T,J]-uI,\qquad
c=\mathbb E[V\xi^*\mid T,J].
$$
此处 $w$ 为实列向量。在给定 $(T,J)$ 下展开平方，再取期望得到
$$
R_\alpha=v_1+D\mathbb E\|w\|^2
+\lambda^2\mathbb E[w^\top\Gamma w]
+2\lambda\,\Re\,\mathbb E[w^\top c].
\tag{9}
$$
遗漏项绝对值至多为 $\lambda^2\mathbb E[\|\Gamma\|_{\rm op}\|w\|^2]+2\lambda\mathbb E[\|c\|\|w\|]$。两条件矩偏差均为零是 (2) 的充分条件，并非必要。若真实 $\theta,V$ 的边际也改变，应将首项换成实际 $\mathbb E|\xi|^2$，重新估计基准协方差和局部条件风险。在离线独立链路上，可对实际误差减去 (2) 的预测逐链路配对，并分别估计两个修正项的投影。按噪声、幅度或场景分层能揭示抵消；有限样本的相合不能证明条件独立。若已另证真实局部下界为 $L$，只有风险上界连同修正余量严格低于 $L$ 才构成分离证明。

**可算反例。** 用两个不同的一位标记让 query 无误地匹配一行。若该行 $X=|V_J|^2/u>t>0$，改选另一行，否则保留匹配行。算法从标记得到索引，从观测取得幅度，未读取标签。$X\sim\mathrm{Exp}(1)$，另一行的均方为 $u$ 且相位独立。故在换行时，附加损失的条件期望为 $D(X+1)$；对 $\int_t^\infty(x+1)e^{-x}\,dx$ 积分，得到
$$
R_{\rm true}=v_1+D(t+2)e^{-t},\qquad
R_{\rm nominal}=v_1+2D e^{-t}.
$$
只凭错配频率代入 (2) 少算 $Dte^{-t}$。有限 softmax 也有反例：将硬选择改为给所选项 $1-\delta$、另一项 $\delta$，$0<\delta<1/2$。因错项权重从 $\delta$ 切换到 $1-\delta$，真实风险与名义风险之差变为 $D[(1-\delta)^2-\delta^2]te^{-t}>0$。选择独立子观测作指纹仍需检查共享潜在信道造成的相关性。

本研究给出一组可实现的参数与校准条件。训练从何种初始化达到这些参数、需多少带标签或无标签样本、分布改变时是否仍有校准，均未证明。已知二元匹配信道时，经典似然已经达到 Bayes 风险，任何同信息估计器都不能严格超过它。

# 6　EM-FloatFormer 的对应关系与最小验证

## 6.1　指定实现的观测与隔离审查

审查基线为分支 `research/p12-first-principles-diagnostics` 的指定提交 `655de6aa42a08fe73e4f67f7060fb667d95518f1`。以下链接均固定到该 SHA。本轮新增目录为 `research/content_addressing_scalar/`，可独立解压；实验没有调用 P12 生成器或运行正式种子。

| 定理量 | 在实际接收器中可定义或检查的量 |
|:--|:--|
| $\theta$ | 预先固定目标行 $t$ 和矩阵坐标 $(a,b)$，估计 $H_B[t,a,b]$；研究创新时减去仅从公共数据得出的同一基准 $m_C$。预测可取所属折的 `h_mix[t,a,b]`。离线评分才从 `h_b_true_mat` 读目标。$m_C$ 若只是 plug-in，须另证其条件均值性质。 |
| $J$ | 当前接口没有“唯一有效 helper”标签。可在前瞻合成试验中独立生成一个与目标共享物理状态的有效参考，并由评测器保留索引。现实中可能有多个、没有或效用连续；需另建模型。按实现后误差最小倒推 $J$ 会改变联合分布。 |
| $Q$ | 可由公共 Frame-A 导频、B-anchor 和目标的**公开位置**生成对目标轨迹的预测特征。如果五个 AUX 参与公共状态 $C$，两臂必须用同样的五行；这些信息对所有臂的成本都要计入。 |
| $K_j$ | 可从合法 helper 行和公共模型构造匹配信息，例如候选轨迹预测残差或留一行拟合的物理似然；这是今后可比较的分数。现有码字可靠性 margin 比较同一行的两个码字，并不测量真参考相对其他行的间隔。 |
| $V_j$ | 现有实现把 helper 接收行、盲解码符号和有效噪声方差一起送入重拟合，没有独立复标量 $V_j$ 接口。一个可实现的候选是固定 $C$ 后仅加入第 $j$ 行，取其目标坐标预测相对公共预测的变化。其比例、等边际、噪声与交叉矩均待检验。 |
| $S,M$ | 在同一折的七个合法 helper 中，固定 $k$ 个，例如按到 anchor 的距离及行号预选；这给出真正固定的局部访问对照。共享 $C$ 时，$M=7$ 只数额外 helper。全局对照读同一合法池，继续屏蔽目标及 probe。 |

**接收入口。** P11 的 `OffGridObservation` 仅公开 `y_a,y_b,training_rows,probe_row,n0,pilot_tx_b,pilot_tx_a`。模拟器将公开的 B-probe 接收行置为 NaN；真实 $H$、数据比特、码字、probe 接收及干净波形、额外模拟状态和未遮盖全帧只供评测。使用已知设计的 probe 发射序列作预测是合法的。P12 完整入口只接收 observation，Frame-A 原子发现用四行各 16 个已知导频格点，B 支撑重置用 anchor。 [P11 观测与真值](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p11/contract.py#L446-L476) · [模拟器](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_physics.py#L240-L286) · [Frame-A 输入](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p11/offgrid_channel.py#L529-L555) · [接收入口](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L379-L485)

**访问范围。** 每折有六个交错目标、五个 AUX、七个其余合法 helper；三折的 15 个 AUX 互异。几何分配使首波最近两行分居 anchor 两侧，同距时按升序行号。默认波前启用，处理两行后仍尝试余下五行。每折先遮盖六个所属目标及 probe，再进行 AUX 评分、盲解码、同化。目标行的接收能量仅在后续突发噪声估计使用，最终解调也读取目标接收行；对应输出信道矩阵由已拟合射线均值和时延矩阵合成。该隔离是**按目标所属折和处理阶段**定义的。 [三折几何](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_physics.py#L842-L903) · [遮盖、波次、同化](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L822-L930) · [目标输出](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L932-L965) · [信道合成与解调](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p11/offgrid_channel.py#L751-L788)

**现有物理对照与成本。** 默认 P12 是能访问每折全部合法 helper 的非学习物理方法，按波次与可靠性接纳多行。保留 A 支撑时每条射线通常有 $21+20+16=57$ 个模式；B-only reset 使用 20 个。默认估计射线数 5、每行 $N=32$，这三个数字分别不同于 helper 数。其主要开销包括最多三条强射线的候选模式上、每个 AUX 行的稠密协方差 Cholesky（每次 $O(N^3)$），最多七个 helper 行的解码尝试、首波多模式码字并集搜索和每次八轮阻尼 Jacobi 拟合。默认流程无可训练网络权重，物理常数、模式库和运行时估计状态仍有存储和计算成本。现有方法没有枚举物理世界的精确联合后验。 [模式与先验](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L457-L521) · [重拟合与 AUX](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L586-L689) · [码字 margin](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/offgrid_receiver.py#L776-L819) · [P11 盲解码](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p11/wavefront_pipeline.py#L636-L681)

**共享观测与版本边界。** helper 的同一带噪行影响码字选择、残差权重和后续条件拟合；第二波还继承第一波产生的状态。即便拆分独立热噪声，公共潜在信道也可能耦合各量。P12 的修复切断了 B-anchor 后验回灌 family 先验，并让各次再拟合从固定有效先验与去重过的行开始；AUX 使用 anchor 条件化的近似似然。旧的 3,584 链路正式结果属于旧实现。修复说明中的 14 链路配对诊断也不用于本研究的风险、BLER 或优越性结论。 [修复、诊断与历史冻结说明](https://github.com/20171806011/EM-FloatFormer/blob/655de6aa42a08fe73e4f67f7060fb667d95518f1/p12/P12_CONSISTENCY_FIX.md)

若将来比较物理接收器，应在同一目标、同一 $C$、同一七行池上并排给出几何固定局部、全池物理似然硬检索与软聚合、现有 P12 和 attention；可枚举的小型物理生成模型可另给同信息精确 Bayes。各臂记录读取量、解码与重拟合次数、模型参数及推断耗时。同化阈值、估计量和取样单位需预先固定；风险用留出链路上的复信道 MSE，BLER 是另一指标。目标位置公开，所属目标行接收样本不得进入信道拟合；probe 接收值、真实信道、真实参考索引及目标码字由离线评测器保管。

## 6.2　预先固定的小规模核对

参数和策略先写入 `verification/PLAN.json`，随后运行。每个场景 32,768 条独立合成样本，共九个主场景及一个相关权重反例。同一场景下各策略共用样本以配对估计。十个 PCG64 种子从 2026112101 至 2026112110；默认 $v_0=1,\sigma^2=0.25,k=2$，固定窗口取前 $k$ 个，局部与全局共享所有 keys。预定的 14 个场景/策略组合全部保留。统计量的标准误为逐样本配对差的样本标准差除以 $\sqrt n$；六个标准误的核对阈值只用于发现实现错误。

下表为同一组观测的原始平方损失均值。括号是“该策略减固定局部”的**一个配对标准误**。校准 attention 与枚举 Bayes 用两种打分写法得到相同后验。其他较弱或较强的对照均使用第 4.2 节声明的读取预算。

| 场景与策略 | 全局策略 MSE | 固定局部 Bayes MSE | 差值（标准误） |
|:--|--:|--:|--:|
| 强匹配，校准 | 0.201431 | 0.804517 | −0.603086 (0.005084) |
| 同场景，过热 | 0.902945 | 0.804517 | +0.098428 (0.003765) |
| 同场景，反向高置信 | 1.700982 | 0.804517 | +0.896465 (0.008374) |
| 弱匹配，校准 | 0.799430 | 0.952651 | −0.153220 (0.003374) |
| 同场景，过冷 | 1.011787 | 0.952651 | +0.059136 (0.006729) |
| 盲匹配，均匀 | 0.895784 | 0.973497 | −0.077714 (0.002151) |
| 同场景，过冷 | 1.440023 | 0.973497 | +0.466526 (0.007477) |

**可复算的总体上界。** 强匹配取 $M=8,k=2,d=48,\rho=0.08,g=8$，$\Delta=1/3$，校准温度。有限和 (8) 得 $\Pr(G)\simeq0.994810220635$。此时 $b=(M-1)[\rho/(1-\rho)]^g$，所以 (3) 和 (1) 可完全用有理数比较。严格结果为 $U\simeq0.208303646984<L_k=0.8$。浮点数仅用于显示。在弱匹配场景 $d=8,\rho=0.32,g=1$，式 (3) 只给 $1.612381$，未能通过充分条件；由 (7)，校准全局方法仍严格优于固定窗。该组非学习硬检索、仅取后验前两项的自适应估计、全局 Bayes 的实测 MSE 分别为 $1.180697,0.832609,0.799430$。由此可见“检索一个”和“整合不确定性”在同一实验里有不同代价。

**其余边界。** 固定 $d=8,\rho=0.2,g=1$ 时，$M=4,8,32$ 的精确正间隔概率依次为 $0.691130,0.529133,0.261567$。$M=4,d=4,\rho=0.1$ 可枚举全部 $5^4=625$ 个距离元组；全局、固定 $k=2$ 局部的总体风险分别为 $0.4389581,0.7194791$，$k=4$ 时两者同为 $0.4389581$。该方案仅把 $\sigma^2$ 改为 9999，精确风险为 $0.99992987,0.99996493$，小差异应读有限和。盲匹配 $\rho=0.5,M=8,k=2$ 的精确风险为 $0.900000,0.975000$。相关权重反例取 $t=1$，实际风险 $1.082911$，名义值 $0.788607$，差额 $0.294304$；实测配对差为 $0.298695$，标准误 $0.007777$。

**复现。** 在解压目录的上级执行以下命令。第一条校验只依赖 Python 标准库；第二条使用 Python 3.12.14、NumPy 2.3.5，脚本其余依赖均为标准库。输出写到新位置。目录内的 CSV 保留所有策略、全局硬匹配、全局 Bayes、自适应取值、解得的上界、成对差与公式残差。JSON 记录方案指纹、运行版本和核对结果。

```bash
cd research/content_addressing_scalar
python3 exact_certificate.py
python3 verification/verify.py --out ./rerun
diff verification/results/results.csv ./rerun/results.csv
diff verification/results/checks.json ./rerun/checks.json
```

位级复现应使用记录的版本。不同平台可能产生末位浮点差；有限有理数比较不受其影响。

# 7　结论的证据等级

| 等级 | 本研究能够支持的陈述 |
|:--|:--|
| 已证明 | 在第 2 节的模型中，(1)–(7) 给出下界、精确风险、充分条件及同一信息下的最优对照。带噪指纹提供真实的可观测打分构造；经典后验加权与校准单头 attention 相同。对于固定 $S$，(7) 将远端取值与额外内容识别的收益精确分开。 |
| 合成模型验证 | 固定网格覆盖成功、无证书但成功、过冷或过热失败、无识别、小收益、全窗相等及权值相关反例。有限和与有理数比较是数值复核；随机抽样用于核对实现。 |
| 真实接收器待验证 | 需定义单一有效参考，或适配多参考模型；验证导频条件化、相关交叉项、温度校准与相同访问预算的强经典对照。本文没有测量修复版 P12 的信道误差、BLER、运行时，也没有得到 Transformer 的训练、样本效率、硬件或分布外优势。 |

**与已有工作的关系。** 条件期望在平方损失下最优、Gaussian 配方、Bayes 公式、Hoeffding 尾界与点积注意力均为经典工具。本次针对项目提出的等边际参考模型推导了风险分解、充要比较、间隔界及共享观测反例，并对齐到指定实现；独立完成这些推导并不确立它们在数学文献中的首创性。

**核对过的教材与原始论文**

1. Hossein Pishro-Nik, *Introduction to Probability, Statistics, and Random Processes*, Kappa Research, LLC, 2014, §9.1.4 “Conditional Expectation (MMSE),” ISBN 978-0-9906372-0-2。用于条件均值的平方损失最优性；上文给出其复数形式的证明。 [教材原文](https://www.probabilitycourse.com/chapter9/9_1_4_conditional_expectation_MMSE.php)
2. Wassily Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” *Journal of the American Statistical Association*, 1963, 58(301), 13–30, DOI 10.1080/01621459.1963.10500830。用于第 4 节的可选维度界。 [DOI](https://doi.org/10.1080/01621459.1963.10500830) · [原文 PDF](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)
3. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, “Attention Is All You Need,” *Advances in Neural Information Processing Systems 30*, 31st Conference on Neural Information Processing Systems, 2017, 5998–6008, §3.2.1，arXiv:1706.03762。用于单头 scaled dot-product 实现式。 [会议原文](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) · [开放版本](https://arxiv.org/abs/1706.03762)
