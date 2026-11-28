# Phase 2 · 有理线性边信息矩阵的双标度定理

本附录推广同一阶段的坐标图定理，并直接识别严格无噪声端点。它只用一次连续输入样本；用独立重复来证明下界的对象是辅助的有限群索引编码问题。所有信息量和码长采用以 2 为底的对数。

## 1. 模型与结论

固定正整数 $n$。$X\sim\mathrm{Unif}([0,1]^n)$，坐标独立。任务 $i$ 要求重建 $X_i$，观测 $O_i=A_i X+\sigma Z_i$，其中 $A_i\in\mathbb Q^{r_i\times n}$ 固定。各噪声为标准高斯向量，与源独立。可以令不同任务的噪声独立，因为逐任务平方误差只用各自的条件边缘律。一个确定性可测的 $b$ 位消息服务全部任务，只要求 $\max_i E[(\hat X_i-X_i)^2]\le\varepsilon$。写 $b_*(\varepsilon,\sigma)$ 为最少固定长度位数，$L=\frac12\log_2(1/\varepsilon)$。对固定 $\eta\in[0,1]$，取 $0<c\le C<\infty$ 且 $c\,2^{-\eta L}\le \sigma\le C\,2^{-\eta L}$。这允许连续观测的质量一直逼近无噪声。

通过乘一个共同正整数清除各 $A_i$ 的分母，得到整数矩阵 $H_i$。对侧信息作同样的可逆缩放，噪声上下界常数也同乘该数。以下用调整后的 $c,C$。系数可以为负和含零，矩阵可以秩亏。记 $m=\sum_i r_i$、$h=\max_{\text{行 }j}\sum_\ell|H_{j\ell}|$（没有行时取 0）。

定义 $B_H(k)$ 为下述**精确的一次有限问题**的最小固定广播位数。每个坐标是 $k$ 位的任意字 $W_j\in\mathbb Z/2^k\mathbb Z$。译码器 $i$ 得到 $S_i=H_i W\bmod 2^k$，必须从广播和 $S_i$ 对每个输入都恢复 $W_i$。编码器和译码器可非线性。设 $B_H(0)=0$。下文证明极限
$$
\beta_H=\lim_{k\to\infty} \frac{B_H(k)}{k}
$$
存在。结论为
$$
\boxed{\lim_{\varepsilon\downarrow0}\frac{b_*(\varepsilon,\sigma(\varepsilon))}
 {\log_2(1/\varepsilon)}=\frac{n(1-\eta)+\eta\beta_H}{2}.}
$$
严格 $\sigma=0$ 时的极限也等于 $\beta_H/2$。因此这个量确实给出连续模型的无噪端点，并且不依赖清分母时选择的公倍数。坐标选择矩阵恢复普通索引编码的向量广播率；任意整数矩阵的系数与进位均进入定义。有限域标量线性最小秩描述的是另一种限制。

## 2. 进位及抽象反证工具

使用余数 $0,\ldots,2^k-1$。对任意行，整数提升 $\lfloor (H_iW)_j/2^k\rfloor$ 在 $[-h,h]$ 内，负数采用向下取整。一次可传所有行的提升，开销至多 $C_H=m\lceil\log_2(2h+1)\rceil$。把长字拆为低 $k$ 位和高 $\ell$ 位，低块模观测可直接取，长观测商的高块减去低块的提升即为正确的高块模观测。这给出
$$
B_H(k+\ell)\le B_H(k)+B_H(\ell)+C_H.
$$
由次可加性，$\beta_H=\inf_{k\ge1}(B_H(k)+C_H)/k$。特别地 $\beta_H\in[0,n]$。

更一般地，若 $t$ 个独立的 $k$ 位实例可用 $D$ 位对所有输入零差错编码，把 $t$ 块按 $2^k$ 为基数拼为一个 $kt$ 位实例。提供每块每行的提升，共至多 $tC_H$ 位。具体地，若 $H_iW^{(j)}=S_i^{(j)}+2^k c_i^{(j)}$，从长观测减去 $\sum_{j=0}^{t-1}2^{(j+1)k}c_i^{(j)}$ 并模 $2^{kt}$ 后，恰得基数展开 $\sum_j 2^{jk}S_i^{(j)}$。故 $B_H(kt)\le D+tC_H$。

**随机辅助消息引理。** 对均匀独立的 $k$ 位字 $W$ 和任意有限随机辅助消息 $U$（允许 $P_{U|W}$ 任意），
$$
 k\beta_H\le I(W;U)+\sum_{i=1}^n H(W_i\mid U,S_i)+C_H. \tag{1}
$$

设 $\Omega=(\mathbb Z/2^k)^n$。称 $E\subseteq\Omega$ 可同时正确译码，是指不发送消息时，每一位接收者都能在 $E$ 上从 $S_i$ 决定 $W_i$。写 $\alpha$ 为这类集合的最大基数，即混淆图的独立数。固定任意 $u$ 且 $P(U=u)>0$。为每个 $i$ 及每个可能的输入 $s_i$，独立抽一个猜测 $g_i(s_i)$，其分布取 $P(W_i=\cdot\mid U=u,S_i=s_i)$；对条件为零的地方任取概率分布。每次抽样中全体猜对的输入集合都可同时正确译码，故取其期望大小得
$$
\alpha\ \ge\ \sum_{w\in\Omega}\prod_i P(W_i=w_i\mid U=u,S_i=S_i(w)).
$$
将这些乘积归一化，应用 $P_{W|u}$ 相对该分布的 KL 非负性，再对 $u$ 取平均，得到
$$
\log_2\alpha\ge H(W|U)-\sum_i H(W_i|U,S_i).
$$
最后，一个最大可同时正确译码集的 $t$ 次笛卡尔积仍可同时正确译码；其在 $\Omega^t$ 中的比例为 $p=(\alpha/|\Omega|)^t$。取 $J=\lceil(\ln|\Omega^t|+1)/p\rceil$ 个独立均匀群平移。未覆盖输入数的期望至多 $(1-p)^J|\Omega^t|<1$，故存在一组全覆盖平移。发送覆盖该输入的平移编号即为零差错码，因为各模矩阵作为群同态，接收者可据平移调整 $S_i$ 并撤销目标的平移。每次重复的码长上界趋于 $\log_2(|\Omega|/\alpha)$。结合进位打包得 $k\beta_H\le\log_2(|\Omega|/\alpha)+C_H$。利用均匀性 $H(W)=\log_2|\Omega|$ 即得 (1)。这个证明适用于把粗、细位乃至不同坐标任意混合的消息。

## 3. 带噪下界

令 $k=\lfloor\eta L\rfloor$，$W=\lfloor 2^kX\rfloor$，余数 $R=2^kX-W$ 与 $W$ 独立，均匀于单位立方体。边界点按零测集忽略。$M$ 是任意合格的共用消息。写 $\lambda=2^k\sigma$，有 $c/2\le\lambda\le C$。给定 $W$ 后，堆叠后的观测经缩放和平移为 $H R+\lambda Z$。各行的信号取值范围长度至多 $h$，故
$$
I(X;O_{\rm all}\mid W)\le
\Gamma,\qquad \Gamma=\frac m2\log_2(1+h^2/c^2).
$$
这里逐行用输出熵次可加性、区间长 $h$ 的方差至多 $h^2/4$ 和高斯最大熵。类似地，给定 $S_i$，先提供该观测矩阵的有界提升，然后减去已知的 $H_iW$。这给出 $I(X;O_i\mid S_i)\le r_i\lceil\log_2(2h+1)\rceil+\frac{r_i}2\log_2(1+h^2/c^2)$。

给定 $M,O_i$，可用离 $2^k\hat X_i$ 最近的整数估计 $W_i$。整数误差的均方根至多 $1+1/2+2^k\sqrt\varepsilon\le5/2$。任意整数误差 $D$ 的熵至多 $\frac12\log_2[2\pi e(E D^2+1/12)]$，因为加入独立均匀单位宽抖动后微分熵等于原离散熵。因此 $H(W_i\mid M,O_i)\le C_e=\frac12\log_2[2\pi e(25/4+1/12)]$。再由链式法则、$W_i,M$ 是 $X$ 的函数，
$$
H(W_i\mid M,S_i)
\le H(W_i\mid M,O_i)+I(X;O_i\mid S_i).
$$
向 (1) 代入 $U=M$，得 $I(W;M)\ge k\beta_H-nC_e-2C_H-\Gamma$（$k=0$ 直接用非负性即可）。

另一部分满足
$$
\begin{aligned}
H(M\mid W)&\ge I(X;M\mid W,O_{\rm all})\\
&=h(X\mid W)-I(X;O_{\rm all}\mid W)-h(X\mid M,W,O_{\rm all})\\
&\ge n(L-k)-\Gamma-\tfrac n2\log_2(2\pi e).
\end{aligned}
$$
最后一行用 $h(X|W)=-nk$ 和原有每题 MSE 解码器；把所有观测同时交给下界中的解码器以后仍可使用它们。结合 $b\ge H(M)=I(W;M)+H(M|W)$，有
$$
b\ \ge\ n(L-k)+k\beta_H - nC_e-2C_H-2\Gamma-\tfrac n2\log_2(2\pi e). \tag{2}
$$
这个下界对一般可测编码器成立。

## 4. 构造性上界

令 $\ell=\lceil L\rceil+2$。直接传每个坐标最后 $\ell-k$ 个精细位，再用 $B_H(k)$ 位传粗字 $W$ 的一次模整数索引码。接收方的带噪观测必须还原正确的 $S_i$。对于每一行，编码器另发整数 $(H_iW)_j\bmod P$，其中
$$
P=\left\lceil 2(h+C\sqrt{2\ln(4\max(1,m)/\varepsilon)}+1)\right\rceil .
$$
将 $2^kO_{ij}$ 投影到该余数类中最近的整数。由于其等于 $(H_iW)_j+(H_iR)_j+\lambda Z_{ij}$，在 $|Z_{ij}|\le\sqrt{2\ln(4\max(1,m)/\varepsilon)}$ 时误差严格小于 $P/2$，故精确恢复该行整数，再取模即可。所有行同时成功的概率至少为 $1-\varepsilon/2$。成功时用精细格中心重建，每坐标平方误差至多 $2^{-2\ell-2}$。失败时最后结果截断至 $[0,1]$，每坐标平方误差至多 1。总期望平方误差至多 $\varepsilon$，固定长度
$$
b\ \le\ n(\ell-k)+B_H(k)+m\lceil\log_2 P\rceil. \tag{3}
$$
固定矩阵时最后一项是 $O_H(\log(L+2))$，其常数亦可依赖 $C$。由 (2)、(3) 和 $B_H(k)/k\to\beta_H$ 得所述极限。这里没有假设 $B_H(k)-k\beta_H$ 本身为对数阶。

## 5. 严格无噪声端点

$\sigma=0$ 时取 $k=\lfloor L\rfloor$。同一上界只需固定 $P>2(h+1)$。为给出下界，条件化于 $R=r$。平均来看，正规化总误差 $2^{2k}\sum_i E[(\hat X_i-X_i)^2]$ 至多 $n$，所以可选一个所有解码器都有定义的 $r$，使给定 $r$ 的总正规化均方误差至多 $2n$。固定此 $r$，留下均匀的 $W$。用原消息 $M((W+r)/2^k)$ 加上全部有界提升作为辅助消息 $U$。从 $S_i$ 和提升可构造 $2^{-k}H_i(W+r)$，因而可使用原无噪声解码器。把 $2^k\hat X_i-r_i$ 四舍五入后，估计 $W_i$ 的整数误差均方根至多 $\sqrt{2n}+1/2$；抖动熵界给出 $H(W_i|U,S_i)=O_n(1)$。又有 $I(W;U)\le b+C_H$，代入 (1) 得 $b\ge k\beta_H-2C_H-O_n(1)$。所以严格无噪声与 $\eta=1$ 的带噪高精度**斜率**一致。这两个模型在有限精度时允许有不同的最优码长。

## 6. 任意固定实矩阵：全部尺度的双边估计

现在允许 $A_i\in\mathbb R^{r_i\times n}$，令 $h=\max_j\sum_\ell|A_{j\ell}|$，其余符号仍如第 1 节。对 $k$ 位立方格 $\Omega=\{0,\ldots,2^k-1\}^n$ 定义
$$
S_i^A(w)=\lfloor A_iw\rfloor\in\mathbb Z^{r_i}.
$$
这是普通整数，**不取模**。记 $B_A^\Box(k)$ 为有此精确离散边信息时、恢复 $w_i$ 的单次零差错固定广播位数，$B_A^\Box(0)=0$。存在常数 $D_1,D_2$，仅依赖于这些固定矩阵、维数和 $c,C$，使一切充分小的 $\varepsilon$、全部 $\eta\in[0,1]$ 同时满足
$$
\left|b_*(\varepsilon,\sigma)-n(L-k)-B_A^\Box(k)\right|
 \le D_1+D_2\log_2(L+2),\qquad k=\lfloor\eta L\rfloor. \tag{4}
$$
它不要求 $B_A^\Box(k)/k$ 收敛。

**对实系数进行平移而只支付有限进位。** 照第 2 节定义可同时正确译码集的最大大小 $\alpha_k$，这里用 $S_i^A$。写 $V=|\Omega|=2^{nk}$。每个颜色类都可同时正确译码，故 $\log_2(V/\alpha_k)\le B_A^\Box(k)$。对最大集合取
$J=\lceil(V/\alpha_k)(\ln V+1)\rceil$ 个均匀随机的环面平移，按第 2 节的未覆盖点期望估计，存在全覆盖的选择。要从 $w$ 移到覆盖它的译码集，发送平移号 $v$，并在每个坐标计算
$$
w'=w+v-2^k c_w,\qquad c_w\in\{0,1\}^n .
$$
编码器发送 $c_w$ 和每个观测分量的进位 $e_{i,w}$。分量逐一满足
$$
\lfloor A_iw'\rfloor
 =\lfloor A_iw\rfloor+\lfloor A_i(v-2^k c_w)\rfloor+e_{i,w},
\qquad e_{i,w}\in\{0,1\}^{r_i}.
$$
这条公式只使用两个实数相加时的小数进位界，与系数有理性无关。接收者因此重建平移后所需的边信息，译出 $w'_i$ 再反移。写
$\Delta_k=n+m+\lceil\log_2(\ln V+2)\rceil+1$，则
$$
\log_2(V/\alpha_k)\le B_A^\Box(k)
 \le \log_2(V/\alpha_k)+\Delta_k.
$$
第 2 节随机猜测与 KL 的推导在任意确定性边信息上成立，故任意随机消息 $M$ 满足
$$
 I(W;M)\ge B_A^\Box(k)-\Delta_k-\sum_i H(W_i\mid M,S_i^A).
$$
这给出一条单尺度消息下界。这里不使用整数矩阵的跨尺度进位打包。

**连续观测的上下界。** 第 3 节中整数误差熵界 $C_e$ 不变。给定 $S_i^A$ 后，可从 $2^kO_i$ 减去它，所剩信号为 $\{A_iW\}+A_iR$ 加 $\lambda Z_i$，其中 $\{\cdot\}$ 是逐分量小数部分。每一行的无噪信号范围长度至多 $1+h$。因此 $\sum_i I(X;O_i\mid S_i^A)\le\Gamma'=\frac m2\log_2(1+(1+h)^2/c^2)$。给定完整粗字后的所有观测仍满足 $I(X;O_{\rm all}\mid W)\le\Gamma=\frac m2\log_2(1+h^2/c^2)$。原来的熵分解给出
$$
 b\ge n(L-k)+B_A^\Box(k)-\Delta_k-nC_e-\Gamma'-\Gamma-\tfrac n2\log_2(2\pi e).
$$
$k=0$ 时 $B_A^\Box(0)=0$，直接使用细层下界。上界发送 $B_A^\Box(k)$ 位索引码、全部细位和每一行的 $\lfloor (A_iW)_j\rfloor\bmod P$。采用第 4 节的 $P$ 并将其中 $h$ 替为 $h+1$，以覆盖 $\{A_iW\}$ 引起的误差。余数格上的最邻近校正精确恢复 $\lfloor (A_iW)_j\rfloor$；在同一个高概率事件上译码成功。由此
$b\le n(\ell-k)+B_A^\Box(k)+m\lceil\log_2 P\rceil$，得到 (4)。

令 $\rho_-,\rho_+$ 为 $\liminf$、$\limsup$ 的 $B_A^\Box(k)/k$。当 $0<\eta\le1$，以 $\varepsilon\downarrow0$ 取连续极限时，$k$ 经过所有充分大的整数，故带噪斜率的下极限和上极限分别恰为 $[n(1-\eta)+\eta\rho_\pm]/2$。若一个正 $\eta$ 的极限存在，则 $\rho_-=\rho_+$，全部正 $\eta$ 的极限均存在；$\eta=0$ 总是给出 $n/2$。对一般实矩阵，(4) 建立的是相对于**噪声与目标标准误差同阶**的端点的结论。严格无噪声时，精确 $A_i(W+r)$ 可在一个量化格中携带额外信息，第 5 节的有界提升论证在这里没有建立。

**有理行空间推论。** 若每个实矩阵的行空间都由有理向量张成，则严格无噪声端点与带噪端点仍相等并存在。取该空间的一组整数基作为矩阵 $H_i$ 的行，可写 $A_i=P_i H_i$，$P_i$ 满列秩。将观测投影到 $P_i$ 的列空间，正交补为与源及投影噪声独立的纯高斯噪声，无法降低给定消息时的最优平方风险。对投影使用固定左逆，得到 $H_i X$ 加正定协方差的高斯噪声，协方差为 $\sigma^2(P_i^\top P_i)^{-1}$。用所有固定矩阵的极端特征值将这些接收端观测夹在两组各向同性高斯实验之间（额外独立高斯作退化信道；取条件均值去掉解码随机性）。两组有相同的第 1 节斜率。在零噪声下该预处理精确可逆于信号子空间。零秩接收者丢弃纯噪声即可。

## 7. 适用边界与可复审文献

有理系数让清分母后的跨位进位落在有限整数集合，并使模观测在分块投影与平移纠错时保持群同态。一般实矩阵的带噪双尺度比较由 (4) 给出。对于真正的无理行空间，严格无噪声端点尚需桥梁。例如将 $(w_1,w_2)$ 映到 $w_1+\sqrt2w_2$，离散有界整数输入的精确实输出可区分全部输入，有限精度测量的区分能力却受间距限制。用精确实点索引替代有限分辨率的观测，会破坏上述粗尺度下界的意义。这一现象本身没有证伪实系数连续模型。本附录的连续模型固定比较一次查询所获取的给定矩阵观测。可自由改变测量矩阵的动作集合及连续查询扩展见阶段主文所定义的协议。

**形式化边界。** 本附录是书面证明。一般矩阵的随机译码器与 KL 论证、独立集平移覆盖、矩阵进位、互信息链式界和连续高斯量化尚未作为一般矩阵定理整体进入 Lean。阶段主文分别列出经过内核验证的抽象和具体验证项。

- Blasiak, Kleinberg & Lubetzky, *Index Coding via Linear Programming*, https://arxiv.org/abs/1004.1379 .
- Ünal & Wagner, *LP Bounds for Rate-Distortion with Variable Side Information*, §VI-D, Theorem 12, https://arxiv.org/pdf/1612.03459 . 其高斯奇环有独立高斯源及无噪邻居观测，讨论记忆无关源的块率失真函数。
- Byrne & Calderini, *Error Correction for Index Coding With Coded Side Information*, https://doi.org/10.1109/TIT.2017.2687933 . 其有限域线性索引码刻画可用于区分离散有限域模型与本附录的模整数端点。
