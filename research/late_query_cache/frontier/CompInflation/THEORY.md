# Conjecture 2: Computation-Limited Readout & Dimension Inflation (`CompInflation`)

**Mathematical Monograph & Formal Verification Scope Boundary**

---

## 1. Problem Formulation, Notation, and Summary of Regimes

Let $(\mathcal{X}, \mathcal{F}, \mu)$ be a probability space and $X \sim \mu$ be a source random variable with intrinsic information dimension $d_{\mathrm{info}}$ (for canonical continuous benchmarks, $\mathcal{X} = [0,1]^d$ with $\mu = \mathrm{Unif}([0,1]^d)$ and $d_{\mathrm{info}} = d$). A latent memory/cache encoder maps $X$ to a memory object $M = E(X)$ (possibly randomized) and emits a finite-dimensional cache representation
$$z(M) \in \mathbb{R}^r$$
(or a quantized representation $\hat{z}(M)$ using $B$ total bits, or per-coordinate bit budgets $b_1, \dots, b_r$). Given a query family $\{T_q\}_{q \in Q} \subset L^2(\mu)$ of scalar target functionals $T_q : \mathcal{X} \to \mathbb{R}$, a query-dependent readout $g_q \in \mathcal{G}$ from a computationally restricted class $\mathcal{G}$ predicts $T_q(X)$ from $z(M)$ via $g_q(M) = \Phi_q(z(M))$.

We measure simultaneous approximation accuracy across $Q$ by:
- **Worst-case $L^2$ Mean Squared Error (MSE)**: $\sup_{q \in Q} \mathrm{MSE}(T_q, g_q) \le \varepsilon$, where $\mathrm{MSE}(T_q, g_q) := \mathbb{E}\bigl[(T_q(X) - g_q(M))^2\bigr]$;
- **Worst-case $L^\infty$ error**: $\sup_{q \in Q} \|T_q - g_q\|_{L^\infty(\mu)} \le \delta$.

When $E$ is deterministic, writing $\phi_j := z_j \circ E : \mathcal{X} \to \mathbb{R}$ ($j = 1, \dots, r$) places the readout problem directly on $(\mathcal{X}, \mu)$. (If $E$ is stochastic, conditioning on the internal randomness of $M$ and applying Jensen's inequality shows that replacing $g_q(M)$ by $\bar{g}_q(x) := \mathbb{E}[g_q(M) \mid X = x]$ strictly or weakly reduces MSE; hence all $L^2$ lower bounds for deterministic representations extend immediately to randomized caches.)

**Core Thesis (Computation-Limited Dimension & Bit-Rate Inflation).** Although an unconstrained measurable readout allows $r = d_{\mathrm{info}}$ and cache bit-rate $\frac{d_{\mathrm{info}}}{2}\log_2(1/\varepsilon)$, restricting $\mathcal{G}$ forces the cache to pre-compute target features, inflating the required cache dimension $r$, norm radius $R_z$, and/or bit-rate slope $\frac{d_{\mathrm{comp}}}{2}$ far above $d_{\mathrm{info}}$:

| Readout Class $\mathcal{G}$ | Dimension / Width / Radius Lower Bound | Cache Bit-Rate Slope (bits vs. $\log_2(1/\varepsilon)$) | Collapse Mechanism & Boundary |
|---|---|---|---|
| **R2: Linear / Affine** ($w_q^\top z + c_q$) | $r \ge d_{\mathrm{comp}} = \dim \overline{\mathrm{span}}_{L^2}\{T_q\}$ (or $d_{\mathrm{comp}}^{\mathrm{aff}}$) | $\frac{d_{\mathrm{comp}}}{2}$ (full-rank joint VQ or per-coordinate scalar quantization) | Collapses to $\frac{d_{\mathrm{info}}}{2}$ bits *only* under joint VQ on a $d_{\mathrm{info}}$-dim manifold, never in dimension $r$ |
| **R3: $L$-Lipschitz** ($\|\Phi_q\|_{\mathrm{Lip}} \le L$, $\|z\|_2 \le R_z$) | $\left(1 + \frac{8 L R_z}{A}\right)^r \ge \mathcal{N} \ge \frac{3}{4}\lfloor \Omega \rfloor^d$ | $\ge d \log_2 \lfloor \Omega \rfloor - \log_2\frac{4}{3}$ (codebook); $\frac{r_{\mathrm{eff}}}{2}$ slope under scalar quantization | Bypassed if $R_z = \infty$, $Q$ is singleton, or high gradients live on negligible measure |
| **R4: Depth-2 ReLU** ($r = d_{\mathrm{info}} = 1$) | $W \ge \left(\frac{1}{180\,\varepsilon}\right)^{1/4} - 1$ on $U^2$; $W = \Theta(\nu \varepsilon^{-1/4})$ on $\sin(2\pi\nu u)$ | $\ge \frac{d_{\mathrm{info}}}{2}\log_2\frac{1}{2\pi e \varepsilon}$ (unbreakable) | Collapses $d_{\mathrm{comp}} \to d_{\mathrm{info}}$ in dimension, at polynomial width $W = \Theta(\varepsilon^{-1/4})$ |
| **R4: Deep ReLU** (Yarotsky 2017) | Depth $D = O(\log k \log(k/\delta))$, width $O(1)$ on $U^k$ | $\ge \frac{d_{\mathrm{info}}}{2}\log_2\frac{1}{2\pi e \varepsilon}$ (unbreakable) | Collapses width to $O(1)$ via $\mathrm{polylog}(1/\delta)$ depth; norm-bounded ReLU reverts to R3 |

---

## 2. R2: Linear and Affine Readout Dimension and Bit-Rate Inflation

### 2.1 Linear & Affine Readout Subspaces

Define the closed target subspace and centered affine target subspace in $L^2(\mu)$:
$$V_Q := \overline{\mathrm{span}}_{L^2(\mu)}\{T_q : q \in Q\}, \qquad d_{\mathrm{comp}} := \dim V_Q \in \mathbb{N} \cup \{\infty\},$$
$$V_Q^{\mathrm{aff}} := \overline{\mathrm{span}}_{L^2(\mu)}\{T_q - \mathbb{E}_\mu[T_q]\mathbf{1} : q \in Q\}, \qquad d_{\mathrm{comp}}^{\mathrm{aff}} := \dim V_Q^{\mathrm{aff}} \in \mathbb{N} \cup \{\infty\}.$$

**Lemma 2.1 (Linear vs. Affine Dimension Relation).** *Let $\mathbf{1} \in L^2(\mu)$ be the constant function $1$. Then:*
1. *$V_Q^{\mathrm{aff}} = (V_Q + \mathrm{span}\{\mathbf{1}\}) \cap \mathbf{1}^\perp = \Pi_{\mathbf{1}^\perp}(V_Q)$, where $\Pi_{\mathbf{1}^\perp}(f) := f - \mathbb{E}_\mu[f]\mathbf{1}$.*
2. *If $\mathbf{1} \in V_Q$, then $d_{\mathrm{comp}}^{\mathrm{aff}} = d_{\mathrm{comp}} - 1$ (with $\infty - 1 = \infty$). If $\mathbf{1} \notin V_Q$, then $d_{\mathrm{comp}}^{\mathrm{aff}} = d_{\mathrm{comp}}$.*

*Proof.* Since $V_Q$ is closed in $L^2(\mu)$ and $\mathrm{span}\{\mathbf{1}\}$ is finite-dimensional, their algebraic sum $V_Q + \mathrm{span}\{\mathbf{1}\}$ is closed in $L^2(\mu)$. Because $\Pi_{\mathbf{1}^\perp}$ is a bounded linear projection onto $\mathbf{1}^\perp$, $\Pi_{\mathbf{1}^\perp}(V_Q) = (V_Q + \mathrm{span}\{\mathbf{1}\}) \cap \mathbf{1}^\perp$ is closed and contains $\mathrm{span}\{T_q - \mathbb{E}_\mu[T_q]\mathbf{1} : q \in Q\}$ as a dense subspace, proving (1). For (2), $\ker(\Pi_{\mathbf{1}^\perp}|_{V_Q}) = V_Q \cap \mathrm{span}\{\mathbf{1}\}$, which has dimension $1$ if $\mathbf{1} \in V_Q$ and $0$ if $\mathbf{1} \notin V_Q$. Since $\Pi_{\mathbf{1}^\perp}|_{V_Q} : V_Q \to V_Q^{\mathrm{aff}}$ is surjective, rank-nullity (or isomorphism when $\ker = \{0\}$) yields (2). $\square$

For an $r$-dimensional cache $\phi = (\phi_1, \dots, \phi_r) \in L^2(\mu)^r$, let $W_\phi := \mathrm{span}_{L^2(\mu)}\{\phi_1, \dots, \phi_r\}$ ($\dim W_\phi \le r$):
- **Linear readout**: $g_q = \sum_{j=1}^r w_{q,j} \phi_j \in W_\phi$.
- **Affine readout**: $g_q = \sum_{j=1}^r w_{q,j} \phi_j + c_q \in W_\phi + \mathrm{span}\{\mathbf{1}\}$. Setting $\tilde{T}_q := T_q - \mathbb{E}_\mu[T_q]\mathbf{1}$ and $\tilde{g}_q := g_q - \mathbb{E}_\mu[g_q]\mathbf{1} = \sum_{j=1}^r w_{q,j}(\phi_j - \mathbb{E}_\mu[\phi_j]\mathbf{1}) \in \Pi_{\mathbf{1}^\perp}(W_\phi)$, orthogonality $\tilde{T}_q - \tilde{g}_q \perp \mathbf{1}$ gives
  $$\mathrm{MSE}(T_q, g_q) = \|\tilde{T}_q - \tilde{g}_q\|_{L^2(\mu)}^2 + \bigl(\mathbb{E}_\mu[T_q] - \mathbb{E}_\mu[g_q]\bigr)^2 \ge \|\tilde{T}_q - \tilde{g}_q\|_{L^2(\mu)}^2,$$
  reducing affine readouts on $\{T_q\}$ to linear readouts on $\{\tilde{T}_q\}$ with cache $\tilde{\phi} = \Pi_{\mathbf{1}^\perp}(\phi)$ of dimension $\le r$.

---

### 2.2 Subspace-Rank Obstruction and Irreducible Error Bounds

**Theorem R2.1 (Linear Readout Irreducible Error — Spectral & Rank-Nullity Bounds).**
*Let $\{T_{q_1}, \dots, T_{q_m}\} \subset V_Q$ be any linearly independent subset of size $m \le d_{\mathrm{comp}}$ with Gram matrix $G \in \mathbb{R}^{m \times m}$, $G_{ij} := \langle T_{q_i}, T_{q_j} \rangle_{L^2(\mu)}$, and eigenvalues $\lambda_1(G) \ge \lambda_2(G) \ge \dots \ge \lambda_m(G) > 0$. If the cache dimension satisfies $r < m$, then for every $\phi \in L^2(\mu)^r$ and every readout matrix $W \in \mathbb{R}^{m \times r}$ ($g_i := \sum_{j=1}^r W_{ij}\phi_j$):*
1. **(Ky Fan Spectral Tail Bound)**
   $$\sum_{i=1}^m \mathrm{MSE}(T_{q_i}, g_i) \ge \sum_{j=r+1}^m \lambda_j(G), \qquad \max_{1 \le i \le m} \mathrm{MSE}(T_{q_i}, g_i) \ge \frac{1}{m}\sum_{j=r+1}^m \lambda_j(G) > 0.$$
2. **(Elementary Rank-Nullity + Pythagoras Bound)**
   $$\sum_{i=1}^m \mathrm{MSE}(T_{q_i}, g_i) \ge \lambda_m(G), \qquad \max_{1 \le i \le m} \mathrm{MSE}(T_{q_i}, g_i) \ge \frac{\lambda_m(G)}{m} > 0.$$

*Proof.*
First note that $G$ is symmetric positive definite: for any $c \in \mathbb{R}^m \setminus \{0\}$, $c^\top G c = \|\sum_{i=1}^m c_i T_{q_i}\|_{L^2(\mu)}^2 > 0$ by linear independence, so $\lambda_m(G) > 0$.

**(1) Proof of Ky Fan Spectral Bound.** Define the finite-rank linear operator $A : \mathbb{R}^m \to L^2(\mu)$ by $A c := \sum_{i=1}^m c_i T_{q_i}$, with adjoint $A^* f = (\langle T_{q_i}, f \rangle_{L^2(\mu)})_{i=1}^m \in \mathbb{R}^m$. Then $A^* A = G$, so the nonzero eigenvalues of the finite-rank self-adjoint positive operator $A A^* : L^2(\mu) \to L^2(\mu)$ coincide with the eigenvalues $\lambda_1(G) \ge \dots \ge \lambda_m(G) > 0$ of $G$, with corresponding orthonormal eigenfunctions $u_1, \dots, u_m \in \mathrm{span}\{T_{q_1}, \dots, T_{q_m}\}$ satisfying $A A^* = \sum_{j=1}^m \lambda_j(G)\, u_j \langle u_j, \cdot \rangle$.

Let $P := \Pi_{W_\phi}$ be the orthogonal projection in $L^2(\mu)$ onto $W_\phi$, and let $k := \dim W_\phi \le r$ with orthonormal basis $\{e_1, \dots, e_k\}$ of $W_\phi$. Since $g_i \in W_\phi$, Pythagoras gives $\|T_{q_i} - g_i\|_{L^2}^2 \ge \|(I - P) T_{q_i}\|_{L^2}^2 = \|T_{q_i}\|_{L^2}^2 - \|P T_{q_i}\|_{L^2}^2$. Summing over $i = 1, \dots, m$:
$$\sum_{i=1}^m \|T_{q_i} - g_i\|_{L^2}^2 \ge \sum_{i=1}^m \|A e_i^{\mathbb{R}^m}\|_{L^2}^2 - \sum_{i=1}^m \sum_{\ell=1}^k \langle e_\ell, A e_i^{\mathbb{R}^m} \rangle^2 = \mathrm{Tr}(G) - \sum_{\ell=1}^k \|A^* e_\ell\|_2^2 = \sum_{j=1}^m \lambda_j(G) - \sum_{\ell=1}^k \langle e_\ell, A A^* e_\ell \rangle.$$
If $r = 0$, then $k = 0$ ($P = 0$), so $\sum_{\ell=1}^k \langle e_\ell, A A^* e_\ell \rangle = 0 = \sum_{j=1}^0 \lambda_j(G)$ and the display above directly yields $\sum_{i=1}^m \mathrm{MSE}(T_{q_i}, g_i) \ge \mathrm{Tr}(G) = \sum_{j=1}^m \lambda_j(G)$. For $1 \le r < m$, to bound $\sum_{\ell=1}^k \langle e_\ell, A A^* e_\ell \rangle$ (Ky Fan's maximum principle), expand in $\{u_j\}_{j=1}^m$: letting $p_j := \sum_{\ell=1}^k \langle e_\ell, u_j \rangle^2$, Bessel's inequality in $L^2(\mu)$ gives $0 \le p_j \le \|u_j\|_{L^2}^2 = 1$, and $\sum_{j=1}^m p_j \le \sum_{\ell=1}^k \|e_\ell\|_{L^2}^2 = k \le r$. Since $\lambda_1(G) \ge \dots \ge \lambda_m(G) > 0$, we have $(\lambda_j - \lambda_r)(p_j - 1) \le 0$ for $j \le r$ and $(\lambda_j - \lambda_r)p_j \le 0$ for $j > r$, whence
$$\sum_{\ell=1}^k \langle e_\ell, A A^* e_\ell \rangle - \sum_{j=1}^r \lambda_j(G) = \sum_{j=1}^m \lambda_j(G) p_j - \sum_{j=1}^r \lambda_j(G) = \sum_{j=1}^r (\lambda_j - \lambda_r)(p_j - 1) + \sum_{j=r+1}^m (\lambda_j - \lambda_r) p_j - \lambda_r\Bigl(r - \sum_{j=1}^m p_j\Bigr) \le 0.$$
Hence $\sum_{\ell=1}^k \langle e_\ell, A A^* e_\ell \rangle \le \sum_{j=1}^r \lambda_j(G)$ for all $0 \le r < m$, yielding
$$\sum_{i=1}^m \mathrm{MSE}(T_{q_i}, g_i) \ge \sum_{j=1}^m \lambda_j(G) - \sum_{j=1}^r \lambda_j(G) = \sum_{j=r+1}^m \lambda_j(G).$$
Pigeonhole over $i \in \{1, \dots, m\}$ immediately gives $\max_{1 \le i \le m} \mathrm{MSE}(T_{q_i}, g_i) \ge \frac{1}{m}\sum_{j=r+1}^m \lambda_j(G) > 0$.

**(2) Proof of Elementary Rank-Nullity + Pythagoras Bound.** Consider the linear map $L : \mathbb{R}^m \to \mathbb{R}^r$ defined by
$$(L c)_j := \Bigl\langle \sum_{i=1}^m c_i T_{q_i}, \phi_j \Bigr\rangle_{L^2(\mu)} \qquad (1 \le j \le r).$$
Since $\dim \mathbb{R}^m = m > r = \dim \mathbb{R}^r$, rank-nullity forces $\ker(L) \ne \{0\}$. Choose $c \in \ker(L) \setminus \{0\}$, and set $v := \sum_{i=1}^m c_i T_{q_i}$ and $w := \sum_{i=1}^m c_i g_i = \sum_{j=1}^r (\sum_{i=1}^m c_i W_{ij}) \phi_j \in W_\phi$. By construction, $\langle v, \phi_j \rangle_{L^2(\mu)} = 0$ for all $j = 1, \dots, r$, so $\langle v, w \rangle_{L^2(\mu)} = 0$. By the Pythagorean theorem,
$$\lambda_m(G)\|c\|_2^2 \le c^\top G c = \|v\|_{L^2}^2 \le \|v\|_{L^2}^2 + \|w\|_{L^2}^2 = \|v - w\|_{L^2}^2 = \Bigl\|\sum_{i=1}^m c_i (T_{q_i} - g_i)\Bigr\|_{L^2}^2.$$
Applying the triangle inequality and Cauchy–Schwarz,
$$\Bigl\|\sum_{i=1}^m c_i (T_{q_i} - g_i)\Bigr\|_{L^2}^2 \le \Bigl(\sum_{i=1}^m |c_i| \|T_{q_i} - g_i\|_{L^2}\Bigr)^2 \le \|c\|_2^2 \sum_{i=1}^m \|T_{q_i} - g_i\|_{L^2}^2.$$
Dividing by $\|c\|_2^2 > 0$ gives $\sum_{i=1}^m \mathrm{MSE}(T_{q_i}, g_i) \ge \lambda_m(G)$. $\square$

**Corollary 2.2 ($\varepsilon \to 0^+$ Dimension Inflation for Linear & Affine Readouts).**
*For linear readouts, achieving $\sup_{q \in Q} \mathrm{MSE}(T_q, g_q) \le \varepsilon$ as $\varepsilon \to 0^+$ forces $r \ge d_{\mathrm{comp}}$ (if $d_{\mathrm{comp}} = \infty$, no finite $r$ suffices for all $\varepsilon > 0$). For affine readouts $g_q = w_q^\top \phi + c_q$, applying Theorem R2.1 to $\{\tilde{T}_{q_i}\}$ with covariance Gram matrix $\Sigma_{ij} = \mathrm{Cov}_\mu(T_{q_i}, T_{q_j})$ forces $r \ge d_{\mathrm{comp}}^{\mathrm{aff}}$ as $\varepsilon \to 0^+$.*

---

### 2.3 Bit-Rate Slope Inflation $d_{\mathrm{comp}} / 2$: Exact Regime Split

Dimension inflation $r \ge d_{\mathrm{comp}}$ does not automatically inflate the total cache bits $B$ when $d_{\mathrm{info}} < d_{\mathrm{comp}}$ *unless* either (a) the target vector has full $m$-dimensional support or (b) the $r$ coordinates are quantized individually. Below we prove the exact $d_{\mathrm{comp}} / 2$ bit-rate slope in both regimes.

**Theorem R2.2 (Bit-Rate Slope Inflation $\frac{d_{\mathrm{comp}}}{2}$).**
1. **(Joint Vector-Quantized Cache on Full-Rank $m$-Dimensional Targets)** *Let $T(X) = (T_{q_1}(X), \dots, T_{q_m}(X))^\top \in \mathbb{R}^m$ take values with an absolutely continuous distribution on $\mathbb{R}^m$. Suppose a quantized cache $\hat{z}(M)$ takes at most $2^B$ values and produces $\hat{T} = W \hat{z}(M)$ (or $\hat{T} = W \hat{z}(M) + c$) satisfying $\max_{1 \le i \le m} \mathrm{MSE}(T_{q_i}, \hat{T}_i) \le \varepsilon$, so $\mathbb{E}[\|T - \hat{T}\|_2^2] \le m \varepsilon$.*
   - *(Shannon EPI / Gaussian Max-Entropy Bound)*: If $T(X)$ has finite differential entropy $h_2(T) > -\infty$ (in bits, base $2$), then
     $$B \ge \frac{m}{2}\log_2\frac{1}{\varepsilon} + h_2(T) - \frac{m}{2}\log_2(2\pi e).$$
   - *(Deterministic $L^2$ Ball-Packing Bound)*: If $T(X)$ has Lebesgue density bounded by $f_T(y) \le M_T$ on $\mathbb{R}^m$, then
     $$B \ge \frac{m}{2}\log_2\frac{1}{\varepsilon} - \log_2\bigl(2 M_T \omega_m (2m)^{m/2}\bigr), \qquad \omega_m := \frac{\pi^{m/2}}{\Gamma(m/2 + 1)}.$$

2. **(Per-Coordinate Scalar-Quantized Cache at $r = K$, Even When $d_{\mathrm{info}} = 1$)** *Let $X = U \sim \mathrm{Unif}[0,1]$ ($d_{\mathrm{info}} = 1$) and $T_k(U) = U^k$ ($k = 1, \dots, K$), with Gram matrix $G_{ij} = \frac{1}{i+j+1}$ and smallest eigenvalue $\lambda_K(G) > 0$. Suppose a minimal-dimension cache ($r = K$) has scalar-quantized coordinates $\hat{z}_j(U)$ each taking at most $M_j = 2^{b_j}$ distinct values ($B = \sum_{j=1}^K b_j$ total bits) and linear readout $\hat{T} = W\hat{z}(U)$ achieves $\max_{1 \le k \le K} \mathrm{MSE}(U^k, \hat{T}_k) \le \varepsilon$ with $\varepsilon < \frac{\lambda_K(G)}{K}$. Then $W \in \mathbb{R}^{K \times K}$ is invertible, every coordinate requires*
   $$b_j \ge \frac{1}{2}\log_2\frac{1}{\varepsilon} - \frac{1}{2}\log_2\frac{27 K^{5/2}}{\lambda_K(G)^{3/2}} \qquad (1 \le j \le K),$$
   *and hence the total bit budget inflates with slope $K / 2 = d_{\mathrm{comp}} / 2$:*
   $$B = \sum_{j=1}^K b_j \ge \frac{K}{2}\log_2\frac{1}{\varepsilon} - \frac{K}{2}\log_2\frac{27 K^{5/2}}{\lambda_K(G)^{3/2}}.$$

*Proof.*
**Part (1), Shannon Bound:** Since $\hat{T}$ takes at most $2^B$ values, $B \ge H(\hat{T}) \ge I(T; \hat{T}) = h_2(T) - h_2(T \mid \hat{T}) = h_2(T) - h_2(T - \hat{T} \mid \hat{T}) \ge h_2(T) - h_2(Z)$, where $Z := T - \hat{T} \in \mathbb{R}^m$ has covariance $K_Z = \mathbb{E}[(Z - \mathbb{E}Z)(Z - \mathbb{E}Z)^\top]$. By the Gaussian maximum-entropy theorem, Hadamard's inequality, and AM–GM:
$$h_2(Z) \le \frac{1}{2}\log_2\bigl((2\pi e)^m \det K_Z\bigr) \le \frac{1}{2}\log_2\Bigl((2\pi e)^m \prod_{i=1}^m (K_Z)_{ii}\Bigr) \le \frac{m}{2}\log_2\Bigl(\frac{2\pi e}{m}\mathrm{Tr}(K_Z)\Bigr) \le \frac{m}{2}\log_2(2\pi e \varepsilon),$$
since $\mathrm{Tr}(K_Z) \le \mathbb{E}[\|Z\|_2^2] \le m\varepsilon$. Subtracting yields the claim.

**Part (1), Ball-Packing Bound:** By Markov's inequality, $\mathbb{P}(\|T - \hat{T}\|_2^2 > 2m\varepsilon) \le \frac{\mathbb{E}[\|T - \hat{T}\|_2^2]}{2m\varepsilon} \le \frac{1}{2}$, so $\mathbb{P}(\|T - \hat{T}\|_2 \le R) \ge \frac{1}{2}$ with $R := \sqrt{2m\varepsilon}$. Since $\hat{T}$ takes values in a codebook $\{y_1, \dots, y_{N_c}\}$ of size $N_c \le 2^B$, the event $\{\|T - \hat{T}\|_2 \le R\}$ is contained in $\{T \in \bigcup_{c=1}^{N_c} B_2^m(y_c, R)\}$. Integrating $f_T \le M_T$ over this union gives
$$\frac{1}{2} \le \mathbb{P}(\|T - \hat{T}\|_2 \le R) \le 2^B M_T \omega_m R^m = 2^B M_T \omega_m (2m\varepsilon)^{m/2}.$$
Taking $\log_2$ yields $B \ge \frac{m}{2}\log_2\frac{1}{\varepsilon} - \log_2(2 M_T \omega_m (2m)^{m/2})$.

**Part (2), Per-Coordinate Scalar Quantization:** Let $T(u) = (u, u^2, \dots, u^K)^\top$. If $W \in \mathbb{R}^{K \times K}$ were singular, there would exist $v \in \mathbb{R}^K$ with $\|v\|_2 = 1$ and $v^\top W = 0$, implying $v^\top \hat{T}(U) \equiv 0$. Then
$$\lambda_K(G) \le v^\top G v = \mathbb{E}\bigl[(v^\top T(U))^2\bigr] = \mathbb{E}\bigl[(v^\top(T(U) - \hat{T}(U)))^2\bigr] \le \|v\|_2^2 \mathbb{E}\bigl[\|T(U) - \hat{T}(U)\|_2^2\bigr] \le K\varepsilon,$$
contradicting $\varepsilon < \lambda_K(G)/K$. Hence $W$ is invertible.

For each $j \in \{1, \dots, K\}$, let $r_j^\top$ be the $j$-th row of $W^{-1}$ and normalize $u_j := r_j / \|r_j\|_2$ ($\|u_j\|_2 = 1$). Then $u_j^\top W = \|r_j\|_2^{-1} e_j^\top$, so
$$\hat{Y}_j(U) := u_j^\top \hat{T}(U) = \frac{1}{\|r_j\|_2} \hat{z}_j(U)$$
takes at most $M_j = 2^{b_j}$ distinct scalar values $\{y_1, \dots, y_{M_j}\}$. Meanwhile, the polynomial $p_j(u) := u_j^\top T(u) = \sum_{k=1}^K u_{j,k} u^k$ satisfies:
- $\mathrm{MSE}(p_j, \hat{Y}_j) = \mathbb{E}[(u_j^\top(T(U) - \hat{T}(U)))^2] \le \|u_j\|_2^2 \mathbb{E}[\|T(U) - \hat{T}(U)\|_2^2] \le K\varepsilon$;
- $p_j(0) = 0$, so $L_j := \mathrm{osc}_{[0,1]}(p_j) \ge \|p_j\|_{L^\infty([0,1])} \ge \|p_j\|_{L^2([0,1])} = \sqrt{u_j^\top G u_j} \ge \sqrt{\lambda_K(G)}$;
- $|p_j'(u)| = \left|\sum_{k=1}^K k u_{j,k} u^{k-1}\right| \le \|u_j\|_2 \sqrt{\sum_{k=1}^K k^2} \le \sqrt{K \cdot K^2} = K^{3/2}$ for all $u \in [0,1]$.

By continuity of $p_j$, the image $I_j := p_j([0,1])$ is a closed interval of Lebesgue measure $L_j$. Set $\delta := \frac{L_j}{3 M_j}$ and define the bad value set $B_j := I_j \setminus \bigcup_{\ell=1}^{M_j} [y_\ell - \delta, y_\ell + \delta]$. Then $\mathrm{Leb}(B_j) \ge L_j - 2 M_j \delta = \frac{L_j}{3}$. Letting $S_j := p_j^{-1}(B_j) \subset [0,1]$, we have $p_j(S_j) = B_j$ (since $B_j \subset I_j = p_j([0,1])$). Because $p_j$ is $K^{3/2}$-Lipschitz on $[0,1]$, $\mathrm{Leb}(B_j) = \mathrm{Leb}(p_j(S_j)) \le K^{3/2} \mathrm{Leb}(S_j)$, hence $\mathrm{Leb}(S_j) \ge \frac{L_j}{3 K^{3/2}}$. For every $u \in S_j$, $|p_j(u) - \hat{Y}_j(u)| > \delta$, so
$$K\varepsilon \ge \mathrm{MSE}(p_j, \hat{Y}_j) \ge \delta^2 \mathrm{Leb}(S_j) \ge \frac{L_j^2}{9 M_j^2} \cdot \frac{L_j}{3 K^{3/2}} = \frac{L_j^3}{27 K^{3/2} 2^{2 b_j}} \ge \frac{\lambda_K(G)^{3/2}}{27 K^{3/2} 2^{2 b_j}}.$$
Solving for $2^{2 b_j}$ and taking $\frac{1}{2}\log_2$ yields the stated bound on $b_j$, and summing over $j = 1, \dots, K$ yields $B \ge \frac{K}{2}\log_2\frac{1}{\varepsilon} - \frac{K}{2}\log_2\frac{27 K^{5/2}}{\lambda_K(G)^{3/2}}$. $\square$

*(Remark: The same bound holds for affine readouts $\hat{T} = W\hat{z} + c$ with $\varepsilon < \frac{\lambda_K(\Sigma)}{K}$ by replacing $T$ with $T - \mathbb{E}[T]$, $\hat{T}$ with $\hat{T} - \mathbb{E}[T]$, and $\lambda_K(G)$ with $\lambda_K(\Sigma)$. Indeed, shifting $\hat{Y}_j$ by the constant $u_j^\top(c - \mathbb{E}[T])$ preserves the number of distinct values of $\hat{Y}_j$, and while the centered polynomial $\tilde{p}_j(u) := u_j^\top(T(u) - \mathbb{E}[T])$ need not satisfy $\tilde{p}_j(0) = 0$, it has $\mathbb{E}[\tilde{p}_j(U)] = 0$, so continuity on $[0,1]$ forces $\min_{[0,1]} \tilde{p}_j \le 0 \le \max_{[0,1]} \tilde{p}_j$ and hence $\mathrm{osc}_{[0,1]}(\tilde{p}_j) \ge \|\tilde{p}_j\|_{L^\infty([0,1])} \ge \|\tilde{p}_j\|_{L^2([0,1])} \ge \sqrt{\lambda_K(\Sigma)}$, while $\tilde{p}_j'(u) = p_j'(u)$ is unchanged.)*

---

### 2.4 Sharp Benchmark: $U \sim \mathrm{Unif}[0,1]$, $T_k(U) = U^k$ ($k = 1, \dots, K$)

Here $d_{\mathrm{info}} = 1$, yet linear and affine readouts inflate the cache dimension to $d_{\mathrm{comp}} = d_{\mathrm{comp}}^{\mathrm{aff}} = K$.

**Theorem R2.3 (Shifted-Legendre / Hilbert Irreducible Error Benchmark).**
*For $X = U \sim \mathrm{Unif}[0,1]$ and $T_k(U) = U^k$ ($1 \le k \le K$):*
1. **(Linear Independence & Affine Invariance)** *The functions $\{U, U^2, \dots, U^K\}$ are linearly independent in $L^2([0,1])$, and $\mathbf{1} \notin \mathrm{span}\{U, \dots, U^K\}$. Hence $d_{\mathrm{comp}} = d_{\mathrm{comp}}^{\mathrm{aff}} = K$.*
2. **(Gram & Covariance Matrices)** *The $K \times K$ Gram and covariance matrices are the shifted Hilbert matrix and its centered rank-1 update:*
   $$G_{ij} = \int_0^1 u^{i+j}\, du = \frac{1}{i+j+1}, \qquad \Sigma_{ij} = \frac{1}{i+j+1} - \frac{1}{(i+1)(j+1)} \qquad (1 \le i,j \le K).$$
3. **(Exact Shifted-Legendre Expansion & Determinants)** *Let $\tilde{P}_j(u) := \sqrt{2j+1}\, P_j(2u-1)$ ($j \ge 0$) be the orthonormal shifted Legendre basis on $L^2([0,1])$ ($\langle \tilde{P}_i, \tilde{P}_j \rangle = \delta_{ij}$). Then $U^k = \sum_{j=0}^k c_{kj} \tilde{P}_j(U)$ with*
   $$c_{kj} = \sqrt{2j+1}\,\frac{(k!)^2}{(k-j)!(k+j+1)!} \qquad (0 \le j \le k),$$
   *and*
   $$\det(\Sigma) = \prod_{k=1}^K c_{kk}^2 = \prod_{k=1}^K \frac{1}{(2k+1)\binom{2k}{k}^2}, \qquad \det(G) = \frac{\prod_{1 \le i < j \le K}(j-i)^2}{\prod_{i,j=1}^K (i+j+1)}.$$
4. **(Closed-Form Spectral Bounds & Irreducible Error for $r < K$)** *Let $\rho := 3 + 2\sqrt{2} = (1+\sqrt{2})^2$. For every $K \ge 1$, the smallest eigenvalue $\lambda_K(G)$ satisfies*
   $$\frac{12\sqrt{2}-16}{2K+1}\,(1+\sqrt{2})^{-4K} \le \lambda_K(G) \le \min\!\left(\frac{4K}{(2K-1)(2K+1)\binom{2K}{K}^2},\; \frac{64\pi^2 K^3}{4K^2 - 1}\,(1+\sqrt{2})^{-4K}\right),$$
   *matching the exact Szegő-type scaling $\lambda_K(G) = \Theta\bigl(\sqrt{K}(1+\sqrt{2})^{-4K}\bigr)$. Consequently, whenever $r < K$, any $r$-dimensional linear readout incurs strictly positive irreducible error:*
   $$\sum_{k=1}^K \mathrm{MSE}(U^k, g_k) \ge \sum_{j=r+1}^K \lambda_j(G) \ge \lambda_K(G) \ge \frac{12\sqrt{2}-16}{2K+1}\,(1+\sqrt{2})^{-4K} > 0.$$

*Proof.*
**(1)** If $\sum_{k=1}^K \alpha_k u^k = 0$ in $L^2([0,1])$, continuity forces equality for all $u \in [0,1]$. Evaluating at any $K$ distinct nodes $t_1, \dots, t_K \in (0,1]$ yields $\sum_{k=1}^K \alpha_k t_i^k = 0$, or $\mathrm{diag}(t_1,\dots,t_K)\, V(t)^\top \alpha = 0$ where $V(t)_{ik} = t_i^{k-1}$ is the Vandermonde matrix with $\det V(t) = \prod_{1 \le i < j \le K}(t_j - t_i) \ne 0$. Since $t_i > 0$, $\alpha = 0$. Moreover, if $\mathbf{1} = \sum_{k=1}^K \alpha_k u^k$ in $L^2([0,1])$, continuity at $u = 0$ gives $1 = 0$, a contradiction. By Lemma 2.1, $d_{\mathrm{comp}}^{\mathrm{aff}} = d_{\mathrm{comp}} = K$.

**(2)** Direct integration: $\langle U^i, U^j \rangle = \int_0^1 u^{i+j} du = \frac{1}{i+j+1}$, and $\mathbb{E}[U^i] = \frac{1}{i+1}$.

**(3)** By Rodrigues' formula $P_j(x) = \frac{1}{2^j j!}\frac{d^j}{dx^j}[(x^2-1)^j]$, substituting $x = 2u-1$ ($\frac{d}{dx} = \frac{1}{2}\frac{d}{du}$, $x^2 - 1 = 4u(u-1)$) yields
$$P_j(2u-1) = \frac{(-1)^j}{j!}\frac{d^j}{du^j}\bigl[u^j (1-u)^j\bigr] = \sum_{m=0}^j (-1)^{j+m}\binom{j}{m}\binom{j+m}{m} u^m.$$
Since $u^j(1-u)^j$ vanishes to order $j$ at $u = 0$ and $u = 1$, integrating by parts $j$ times against $u^k$ ($k \ge j$) gives
$$c_{kj} = \sqrt{2j+1}\int_0^1 u^k P_j(2u-1)\, du = \sqrt{2j+1}\,\frac{k!}{j!(k-j)!}\int_0^1 u^k (1-u)^j\, du = \sqrt{2j+1}\,\frac{k!}{j!(k-j)!}\cdot\frac{k!\,j!}{(k+j+1)!},$$
using the Beta integral $\int_0^1 u^k(1-u)^j\, du = B(k+1,j+1) = \frac{k!j!}{(k+j+1)!}$, while $c_{kj} = 0$ for $j > k$ by orthogonality.

Because $c_{k0} = \mathbb{E}[U^k]$ and $\tilde{P}_0 \equiv 1$, we have $U^k - \mathbb{E}[U^k] = \sum_{j=1}^k c_{kj} \tilde{P}_j(U)$. Writing $C_0 := (c_{kj})_{1 \le k,j \le K}$ (lower triangular), $\Sigma = C_0 C_0^\top$, so
$$\det(\Sigma) = \prod_{k=1}^K c_{kk}^2 = \prod_{k=1}^K (2k+1)\left(\frac{(k!)^2}{(2k+1)!}\right)^2 = \prod_{k=1}^K \frac{1}{(2k+1)\binom{2k}{k}^2}.$$
Finally, $G_{ij} = \frac{1}{a_i + b_j}$ with $a_i = i + \frac{1}{2}$, $b_j = j + \frac{1}{2}$ is a Cauchy matrix, whose determinant is $\frac{\prod_{i<j}(a_j - a_i)(b_j - b_i)}{\prod_{i,j}(a_i + b_j)} = \frac{\prod_{1\le i < j \le K}(j-i)^2}{\prod_{i,j=1}^K (i+j+1)}$.

**(4) Lower bound on $\lambda_K(G)$:** Let $p(u) = \sum_{k=1}^K \alpha_k u^k = \sum_{j=0}^K \beta_j \tilde{P}_j(u)$ with $\alpha \in \mathbb{R}^K$. Expanding $\tilde{P}_j(u) = \sqrt{2j+1}\sum_{k=0}^j (-1)^{j+k}\binom{j}{k}\binom{j+k}{k}u^k$, the $\ell_1$-norm of the monomial coefficients of $\tilde{P}_j$ is $\sqrt{2j+1}\sum_{k=0}^j \binom{j}{k}\binom{j+k}{k} = \sqrt{2j+1}\, P_j(3)$ (evaluating at $u = -1$ and using $P_j(-3) = (-1)^j P_j(3)$). By the generating function of Legendre polynomials,
$$\sum_{j=0}^\infty P_j(3) t^j = (1 - 6t + t^2)^{-1/2} = (1 - \rho t)^{-1/2}(1 - \rho^{-1} t)^{-1/2}, \qquad \rho = 3 + 2\sqrt{2} = (1+\sqrt{2})^2.$$
Since both factors have nonnegative Taylor coefficients ($a_m = \binom{2m}{m}4^{-m} > 0$) and $\sum_{m=0}^j a_m a_{j-m} \rho^{j-2m} \le \rho^j \sum_{m=0}^j a_m a_{j-m} = \rho^j \cdot 1 = \rho^j$ (because $\sum_{m=0}^\infty a_m t^m = (1-t)^{-1/2}$ implies $\sum_{m=0}^j a_m a_{j-m} = [t^j](1-t)^{-1} = 1$), we obtain $P_j(3) \le \rho^j$ for every $j \ge 0$. Hence by Cauchy–Schwarz and Parseval ($\|\beta\|_2 = \|p\|_{L^2}$):
$$\|\alpha\|_2 \le \|\alpha\|_1 \le \sum_{j=0}^K |\beta_j|\sqrt{2j+1}\, P_j(3) \le \|p\|_{L^2}\sqrt{\sum_{j=0}^K (2j+1)\rho^{2j}}.$$
Summing the geometric series, $\sum_{j=0}^K (2j+1)\rho^{2j} \le (2K+1)\sum_{j=0}^K \rho^{2j} \le (2K+1)\frac{\rho^{2K+2}}{\rho^2 - 1}$. Using $\frac{\rho^2 - 1}{\rho^2} = \frac{16 + 12\sqrt{2}}{17 + 12\sqrt{2}} = 12\sqrt{2} - 16 \approx 0.97056$ and $\rho^{2K} = (1+\sqrt{2})^{4K}$:
$$\lambda_K(G) = \min_{\alpha \ne 0} \frac{\|p\|_{L^2}^2}{\|\alpha\|_2^2} \ge \frac{12\sqrt{2}-16}{2K+1}\,(1+\sqrt{2})^{-4K}.$$

**(4) Upper bound on $\lambda_K(G)$:** Consider the degree-$K$ test polynomial
$$p^*(u) := \tilde{P}_K(u) - (-1)^K \sqrt{2K+1}\,(1-u)^{K-1}.$$
At $u = 0$, $\tilde{P}_K(0) = \sqrt{2K+1}\,P_K(-1) = (-1)^K \sqrt{2K+1}$, so $p^*(0) = 0$ and $p^*(u) = \sum_{k=1}^K \alpha_k^* u^k \in \mathrm{span}\{u, \dots, u^K\}$. Since $\tilde{P}_K$ is $L^2$-orthogonal to every polynomial of degree $\le K-1$, Pythagoras in $L^2([0,1])$ gives
$$\|p^*\|_{L^2}^2 = \|\tilde{P}_K\|_{L^2}^2 + (2K+1)\int_0^1 (1-u)^{2K-2}\,du = 1 + \frac{2K+1}{2K-1} = \frac{4K}{2K-1}.$$
For the coefficient norm $\|\alpha^*\|_2^2$:
- *(Binomial leading-term bound)*: Since $\deg((1-u)^{K-1}) = K-1$, the leading coefficient is $\alpha_K^* = \sqrt{2K+1}\binom{2K}{K}$, giving $\lambda_K(G) \le \frac{\|p^*\|_{L^2}^2}{(\alpha_K^*)^2} = \frac{4K}{(2K-1)(2K+1)\binom{2K}{K}^2}$.
- *(Exponential $(1+\sqrt{2})^{-4K}$ bound)*: For $1 \le k \le K$, $|\alpha_k^*| = \sqrt{2K+1}\bigl(\binom{K}{k}\binom{K+k}{k} - \binom{K-1}{k}\bigr)$, where $\binom{K-1}{K} = 0$. Checking $k=1$ ($K(K+1)-(K-1) = K^2+1 \ge \frac{1}{2}K(K+1)$) and $k \ge 2$ ($\binom{K+k}{k} \ge 3$ and $\binom{K}{k} \ge \binom{K-1}{k}$) shows $|\alpha_k^*| \ge \frac{\sqrt{2K+1}}{2}\binom{K}{k}\binom{K+k}{k}$ for all $1 \le k \le K$. Summing $k=1,\dots,K$ and using $P_K(3) - 1 \ge \frac{2}{3}P_K(3)$ (as $P_K(3) \ge P_1(3) = 3$) gives $\|\alpha^*\|_1 \ge \frac{\sqrt{2K+1}}{3} P_K(3)$. By Laplace's integral representation for Legendre polynomials,
  $$P_K(3) = \frac{1}{\pi}\int_0^\pi (3 + 2\sqrt{2}\cos\theta)^K\, d\theta = \frac{\rho^K}{\pi}\int_0^\pi \Bigl(1 - \frac{4\sqrt{2}}{\rho}\sin^2\frac{\theta}{2}\Bigr)^K d\theta.$$
  Since $\frac{4\sqrt{2}}{\rho} = 12\sqrt{2}-16 < 1$ and $\sin(\theta/2) \le \theta/2$, on $\theta \in [0, 1/\sqrt{K}]$ we have $(1 - \frac{4\sqrt{2}}{\rho}\sin^2(\theta/2))^K \ge (1 - \frac{\theta^2}{4})^K \ge 1 - \frac{K\theta^2}{4} \ge \frac{3}{4}$, whence $P_K(3) \ge \frac{3}{4\pi\sqrt{K}}\rho^K$. By Cauchy–Schwarz, $\|\alpha^*\|_2^2 \ge \frac{1}{K}\|\alpha^*\|_1^2 \ge \frac{2K+1}{16\pi^2 K^2}\rho^{2K}$, and thus
  $$\lambda_K(G) \le \frac{\|p^*\|_{L^2}^2}{\|\alpha^*\|_2^2} \le \frac{64\pi^2 K^3}{4K^2 - 1}\,(1+\sqrt{2})^{-4K}.$$
Combining both upper bounds yields the theorem, and applying Theorem R2.1 finishes the $r < K$ irreducible error bound. $\square$

**Concrete Numerical Verification ($K = 10$):** Direct high-precision eigenvalue evaluation of $G_{10}$ gives:
$$\lambda_1 \approx 1.076,\quad \lambda_2 \approx 9.849\times 10^{-2},\quad \lambda_3 \approx 6.249\times 10^{-3},\quad \lambda_4 \approx 3.120\times 10^{-4},\quad \lambda_5 \approx 1.212\times 10^{-5},$$
$$\lambda_6 \approx 3.571\times 10^{-7},\quad \lambda_7 \approx 7.699\times 10^{-9},\quad \lambda_8 \approx 1.144\times 10^{-10},\quad \lambda_9 \approx 1.046\times 10^{-12},\quad \lambda_{10} \approx 4.435\times 10^{-15}.$$
At $r = 5 < 10$, the Ky Fan irreducible sum is $\sum_{j=6}^{10}\lambda_j(G_{10}) \approx 3.649\times 10^{-7}$. Meanwhile, fitting $U^2$ by an affine linear readout of its 5 binary expansion bits on a uniform 32-point grid incurs $\max\text{err} = 0.1514$ and $\mathrm{RMSE} = 0.0744$.

---

### 2.5 R2 Failure Modes and Stress-Test Counterexamples

1. **Counterexample R2-CE1 (Joint Vector Quantization on a 1D Manifold — Bit-Rate Does NOT Inflate to $K/2$):**
   Let $U \sim \mathrm{Unif}[0,1]$, $T_k(U) = U^k$ ($k = 1, \dots, K$). Partition $[0,1]$ into $2^B$ uniform bins of width $2^{-B}$ with midpoint $\hat{U}$, and set $\hat{z}(U) := (\hat{U}, \hat{U}^2, \dots, \hat{U}^K) \in \mathbb{R}^K$ using a joint $B$-bit codebook ($|\mathcal{C}| = 2^B$). With $w_k = e_k$,
   $$\mathrm{MSE}(U^k, w_k^\top \hat{z}(U)) = \mathbb{E}\bigl[(U^k - \hat{U}^k)^2\bigr] \le k^2 \mathbb{E}\bigl[(U - \hat{U})^2\bigr] = \frac{k^2}{12} 2^{-2B}.$$
   To achieve $\max_{1 \le k \le K}\mathrm{MSE} \le \varepsilon$, $B = \frac{1}{2}\log_2\frac{1}{\varepsilon} + \log_2\frac{K}{\sqrt{12}}$ suffices! Thus **dimension inflates** ($r = K$), while **joint codebook bits stay at slope $d_{\mathrm{info}}/2 = 1/2$**. Slope $K/2$ requires either full $K$-dimensional target entropy (Theorem R2.2(1)) or per-coordinate scalar quantization (Theorem R2.2(2)).

2. **Failure Mode R2-FM2 (Finite-$\varepsilon$ Practical Rank Collapse on Ill-Conditioned Polynomials vs. Orthonormal Targets):**
   Because $\lambda_j(G_K)$ decays as $(1+\sqrt{2})^{-4j} \approx e^{-3.5255 j}$, for $K = 10$ and any moderate target error $\varepsilon \ge 10^{-6}$, $r = 5$ already achieves $\max_k \mathrm{MSE} \le 3.65\times 10^{-7}$ (since $\sum_{j=6}^{10}\lambda_j \approx 3.65\times 10^{-7}$). In stark contrast, for orthonormal targets $T_k(U) = \tilde{P}_k(U)$ or $\sqrt{2}\sin(\pi k U)$ ($k=1,\dots,K$), the Gram matrix is $G = I_K$, so Theorem R2.1 gives $\sum_{k=1}^K \mathrm{MSE}(T_k, g_k) \ge K - r$ with **zero finite-$\varepsilon$ collapse** until $\varepsilon \approx 1$.

3. **Failure Mode R2-FM3 (Affine Constant Shift):**
   Adding $T_0(U) \equiv 1$ to $\{U, \dots, U^K\}$ gives $d_{\mathrm{comp}} = K + 1$ for linear readouts ($w_q^\top z$) yet $d_{\mathrm{comp}}^{\mathrm{aff}} = K$ for affine readouts ($w_q^\top z + c_q$). Excluding $T_0$ leaves $\mathbf{1} \notin V_Q$, making linear and affine dimension bounds identical ($K$).

4. **Counterexample R2-CE4 (Average MSE under a Decaying Query Prior):**
   If $Q = \mathbb{N}$ with $T_k(U) = \tilde{P}_k(U)$ ($d_{\mathrm{comp}} = \infty$) and queries are weighted by a summable prior $\pi_k = c_\alpha k^{-\alpha}$ ($\alpha > 1$), then projecting onto the first $r$ basis functions gives $\sum_{k=1}^\infty \pi_k \mathrm{MSE}(T_k, g_k) = \sum_{k=r+1}^\infty \pi_k = O(r^{-(\alpha-1)}) \le \varepsilon$ with finite $r_\varepsilon = O(\varepsilon^{-1/(\alpha-1)})$. Simultaneous *worst-case* $\sup_{q \in Q} \mathrm{MSE} \le \varepsilon$ is essential for $r \to d_{\mathrm{comp}}$.

---

## 3. R3: Lipschitz / Bounded-Gradient Nonlinear Readout Tax

Now suppose the readout $\Phi_q : \mathbb{R}^r \to \mathbb{R}$ may be arbitrarily nonlinear, subject to a uniform Lipschitz bound
$$|\Phi_q(u) - \Phi_q(v)| \le L\|u - v\|_2 \qquad \forall u, v \in \mathbb{R}^r, \;\forall q \in Q,$$
and the cache representation is norm-bounded: $z : \mathcal{X} \to B_2^r(R_z) := \{u \in \mathbb{R}^r : \|u\|_2 \le R_z\}$.

### 3.1 Metric Distortion & Radius Lower Bound

**Theorem R3.1 (Lipschitz Metric Distortion & Radius Inflation).**
*If $\sup_{x \in \mathcal{X}} |T_q(x) - \Phi_q(z(x))| \le \delta$ and $\Phi_q$ is $L$-Lipschitz ($L \ge 0$), then for all $x, x' \in \mathcal{X}$:*
$$|T_q(x) - T_q(x')| - 2\delta \le L\|z(x) - z(x')\|_2 \le 2 L R_z.$$
*In particular, when $L > 0$, $R_z \ge \frac{\mathrm{osc}_{\mathcal{X}}(T_q) - 2\delta}{2L}$ where $\mathrm{osc}_{\mathcal{X}}(T_q) := \sup_{x,x' \in \mathcal{X}}|T_q(x) - T_q(x')|$.*

*Proof.* By the triangle inequality,
$$|T_q(x) - T_q(x')| \le |T_q(x) - \Phi_q(z(x))| + |\Phi_q(z(x)) - \Phi_q(z(x'))| + |\Phi_q(z(x')) - T_q(x')| \le 2\delta + L\|z(x) - z(x')\|_2,$$
and $\|z(x) - z(x')\|_2 \le \|z(x)\|_2 + \|z(x')\|_2 \le 2 R_z$. Taking $\sup_{x,x' \in \mathcal{X}}$ yields the radius bound. $\square$

### 3.2 Sharp $L^2$ Multi-Query Rademacher-Cell Covering Lower Bound

To rule out single-query memorization inside $z$ (see Counterexample R3-CE2), consider a multi-query family $\{T_\sigma\}_{\sigma \in \{\pm 1\}^N}$ of high-frequency targets on $X \sim \mathrm{Unif}([0,1]^d)$ ($d = d_{\mathrm{info}}$).

**Construction of the Oscillatory Family.** Let $\Omega \ge 2$, $M := \lfloor \Omega \rfloor \ge 2$, and $N := M^d$. Partition $[0,1]^d$ into $N$ congruent cubes $C_{\mathbf{k}} := \frac{1}{M}(\mathbf{k} + [0,1]^d)$ ($\mathbf{k} \in \{0, \dots, M-1\}^d$), each containing a concentric central subcube $\tilde{C}_{\mathbf{k}} := \frac{1}{M}(\mathbf{k} + [\frac{1}{4}, \frac{3}{4}]^d)$ of side $\frac{1}{2M}$ and Lebesgue measure $2^{-d}/N$. Let $\eta \in C_c^\infty((0,1))$ satisfy $0 \le \eta \le 1$, $\eta|_{[1/4, 3/4]} \equiv 1$, and $\|\eta'\|_\infty \le 5$ (or the piecewise-linear tent $\eta_{\mathrm{PL}}(t) = \min(1, 4\min(t, 1-t))$ with $\|\eta_{\mathrm{PL}}'\|_\infty = 4$). Set $\psi(y) := \prod_{j=1}^d \eta(y_j)$, so $0 \le \psi \le 1$, $\psi|_{[1/4, 3/4]^d} \equiv 1$, and $\|\nabla \psi\|_\infty \le 5\sqrt{d}$ (resp. $4\sqrt{d}$). For each $\sigma \in \{\pm 1\}^N$ (indexed by $\mathbf{k}$) and amplitude $A > 0$, define
$$T_\sigma(x) := A \sum_{\mathbf{k} \in \{0,\dots,M-1\}^d} \sigma_{\mathbf{k}}\,\psi(Mx - \mathbf{k}).$$
Because the supports $\frac{1}{M}(\mathbf{k} + (0,1)^d)$ are pairwise disjoint, $\|\nabla T_\sigma\|_\infty \le 5\sqrt{d}\, A M$ (resp. $4\sqrt{d}\, A M$), $\mathrm{osc}(T_\sigma) \le 2A$, and $T_\sigma|_{\tilde{C}_{\mathbf{k}}} \equiv A\sigma_{\mathbf{k}}$.

**Theorem R3.2 ($L^2$ Multi-Query Lipschitz Covering, Dimension, Radius, and Bit-Rate Tax).**
*Suppose $z : [0,1]^d \to B_2^r(R_z) \subset \mathbb{R}^r$ and $L$-Lipschitz readouts $\{\Phi_\sigma\}_{\sigma \in \{\pm 1\}^N}$ ($L > 0$) achieve*
$$\sup_{\sigma \in \{\pm 1\}^N} \mathrm{MSE}(T_\sigma, \Phi_\sigma \circ z) \le \varepsilon \le \frac{A^2}{64 \cdot 2^d}.$$
*Then:*
1. **(Packing & Covering Lower Bound)** *(i) Any cover of $B_2^r(R_z)$ (or of $z([0,1]^d)$) by $K_{\mathrm{cov}}$ Euclidean balls of radius $\rho_0 := \frac{A}{4L}$ satisfies $K_{\mathrm{cov}} \ge \frac{3}{4} N = \frac{3}{4}\lfloor \Omega \rfloor^d \ge \frac{N}{2} = \frac{\lfloor \Omega \rfloor^d}{2}$. (ii) In particular, choosing a maximal $\rho_0$-separated packing cover of $B_2^r(R_z)$—and hence the covering number $\mathcal{N} := \mathcal{N}(B_2^r(R_z), \|\cdot\|_2, \rho_0)$—satisfies*
   $$\Bigl(1 + \frac{8 L R_z}{A}\Bigr)^r \ge \mathcal{N}(B_2^r(R_z), \|\cdot\|_2, \rho_0) \ge \frac{3}{4}\lfloor \Omega \rfloor^d \ge \frac{3}{2} > 1.$$
   *Because $N = \lfloor \Omega \rfloor^d \ge 2$ forces $\frac{3}{4} N \ge \frac{3}{2} > 1$, this guarantees both $r \ge 1$ and $R_z > 0$ strictly.*
2. **(Dimension & Radius Inflation)**
   $$r \ge \frac{d\log_2\lfloor \Omega \rfloor - \log_2(4/3)}{\log_2\!\bigl(1 + \frac{8 L R_z}{A}\bigr)}, \qquad R_z \ge \frac{A}{8L}\Bigl(\Bigl(\frac{3}{4}\Bigr)^{1/r}\lfloor \Omega \rfloor^{d/r} - 1\Bigr) \;\Bigl(\text{so } R_z \ge \frac{A}{8L}\Bigl(\Bigl(\frac{3}{4}\Bigr)^{1/d}\lfloor \Omega \rfloor - 1\Bigr) \text{ at } r = d\Bigr).$$
3. **(Quantization Bit-Overhead)** *Any discrete $B$-bit cache $\hat{z}$ ($|\mathcal{C}| \le 2^B$) achieving the same MSE satisfies $2^B \ge \frac{3}{4}N$, hence $B \ge d\log_2\lfloor \Omega \rfloor - \log_2(4/3)$ (and $B \ge \max(1, d\log_2\lfloor \Omega \rfloor - 1)$ since $2^B$ single-point balls cover $\hat{z}([0,1]^d)$ and $\frac{3}{4}N > 1$). Furthermore, under per-coordinate scalar quantization of $[-R_z, R_z]^r$ with $b$ bits per coordinate ($L\|z - \hat{z}\|_2 \le \sqrt{\varepsilon}$, step $\Delta \le \frac{2\sqrt{\varepsilon}}{L\sqrt{r}}$ so $2^b \ge \frac{2 R_z}{\Delta} \ge \frac{L R_z \sqrt{r}}{\sqrt{\varepsilon}}$), since $r \ge 1$ and $R_z > 0$ from (1), total bits satisfy*
   $$B = r b \ge \frac{r_{\mathrm{eff}}}{2}\log_2\frac{1}{\varepsilon} + r_{\mathrm{eff}}\log_2(L R_z \sqrt{r}), \qquad r_{\mathrm{eff}} := \frac{d\log_2\lfloor \Omega \rfloor - \log_2(4/3)}{\log_2\!\bigl(1 + \frac{8 L R_z}{A}\bigr)} > 0.$$
4. **(Chaotic Map Iteration Scaling)** *If $T_{\sigma, t} = T_\sigma \circ F^{\circ t}$ is generated by $t$ iterations of an expanding map $F$ on $[0,1]^d$ with uniform expansion factor $e^\lambda > 1$ (Lyapunov exponent $\lambda > 0$, e.g. $d = 1$ tent map $F(u) = 1 - |2u - 1|$ with $\Omega = 2^t$, $\lambda = \ln 2$), then $\Omega$ scales as $e^{\lambda t}$, forcing $r \ge \frac{d(\lambda t / \ln 2) - \log_2(4/3)}{\log_2(1 + 8 L R_z / A)} = \Omega(t)$ (for fixed $R_z, L$) or $R_z = \Omega(e^{\lambda t} A / L)$ (for fixed $r = d$).*

*Proof.*
**Step 1 (Fubini / Probabilistic Point Extraction).** Restricting the $L^2([0,1]^d)$ integral to $\bigsqcup_{\mathbf{k}} \tilde{C}_{\mathbf{k}}$ (each of measure $2^{-d}/N$) and averaging over $\sigma \sim \mathrm{Unif}(\{\pm 1\}^N)$:
$$\varepsilon \ge \mathbb{E}_\sigma\bigl[\mathrm{MSE}(T_\sigma, \Phi_\sigma \circ z)\bigr] \ge \mathbb{E}_\sigma\Bigl[\sum_{\mathbf{k}} \int_{\tilde{C}_{\mathbf{k}}} (A\sigma_{\mathbf{k}} - \Phi_\sigma(z(x)))^2\, dx\Bigr] = 2^{-d}\, \mathbb{E}_{x_{\mathbf{k}} \sim \mathrm{Unif}(\tilde{C}_{\mathbf{k}})}\Bigl[\frac{1}{2^N N}\sum_{\sigma \in \{\pm 1\}^N}\sum_{i=1}^N \bigl(A\sigma_i - \Phi_\sigma(z(x_i))\bigr)^2\Bigr],$$
identifying $\{0,\dots,M-1\}^d$ with $\{1,\dots,N\}$. Thus there exists a deterministic tuple $(x_1, \dots, x_N)$ ($x_i \in \tilde{C}_i$) satisfying
$$\frac{1}{2^N}\sum_{\sigma \in \{\pm 1\}^N}\sum_{i=1}^N \bigl(A\sigma_i - \Phi_\sigma(z(x_i))\bigr)^2 \le 2^d N \varepsilon \le \frac{N A^2}{64}.$$

**Step 2 (Volume-Packing Bound for $\mathcal{N}$).** Let $\{c_1, \dots, c_{K_{\mathrm{pack}}}\}$ be a maximal $\rho_0$-separated subset of $B_2^r(R_z)$ ($\rho_0 = \frac{A}{4L}$). By maximality, the balls $B_2^r(c_m, \rho_0)$ cover $B_2^r(R_z)$ (so $\mathcal{N}(B_2^r(R_z), \|\cdot\|_2, \rho_0) \le K_{\mathrm{pack}}$), while the half-radius balls $B_2^r(c_m, \rho_0/2)$ are pairwise disjoint and contained in $B_2^r(0, R_z + \rho_0/2)$. Comparing $r$-dimensional volumes gives
$$\mathcal{N}(B_2^r(R_z), \|\cdot\|_2, \rho_0) \le K_{\mathrm{pack}}, \qquad K_{\mathrm{pack}}\,\omega_r\Bigl(\frac{\rho_0}{2}\Bigr)^r \le \omega_r\Bigl(R_z + \frac{\rho_0}{2}\Bigr)^r \implies K_{\mathrm{pack}} \le \Bigl(1 + \frac{2 R_z}{\rho_0}\Bigr)^r = \Bigl(1 + \frac{8 L R_z}{A}\Bigr)^r.$$

**Step 3 (Per-Ball Oscillation & Exact Rademacher Double-Counting for Any Cover).** Let $\{B_2^r(c_m, \rho_0)\}_{m=1}^{K_{\mathrm{cov}}}$ be *any* cover of $B_2^r(R_z)$ (or of $z([0,1]^d)$). Assign each $i \in \{1, \dots, N\}$ to one ball $m(i) \in \{1, \dots, K_{\mathrm{cov}}\}$ with $\|z(x_i) - c_{m(i)}\|_2 \le \frac{A}{4L}$, partitioning $\{1,\dots,N\}$ into sets $S_m := \{i : m(i) = m\}$ of sizes $n_m := |S_m|$ ($\sum_{m=1}^{K_{\mathrm{cov}}} n_m = N$). For any $\sigma$ and any $i \in S_m$, the $L$-Lipschitz property forces
$$\bigl|\Phi_\sigma(z(x_i)) - \Phi_\sigma(c_m)\bigr| \le L\|z(x_i) - c_m\|_2 \le \frac{A}{4}.$$
Let $n_m^+(\sigma) := |\{i \in S_m : \sigma_i = +1\}|$ and $n_m^-(\sigma) := |\{i \in S_m : \sigma_i = -1\}|$ ($n_m^+ + n_m^- = n_m$).
- If $\Phi_\sigma(c_m) \le 0$, every $i \in S_m$ with $\sigma_i = +1$ has $\Phi_\sigma(z(x_i)) \le \frac{A}{4}$, so $|A\sigma_i - \Phi_\sigma(z(x_i))| \ge A - \frac{A}{4} = \frac{3A}{4} \ge \frac{A}{2}$ (squared error $\ge \frac{A^2}{4}$). (If one compares against the ball empirical mean $\bar{\Phi}_{\sigma,m}$ instead of $\Phi_\sigma(c_m)$, diameter $\frac{A}{2L}$ still yields $\ge \frac{A}{2}$, hence $\ge \frac{A^2}{4}$.)
- Symmetrically, if $\Phi_\sigma(c_m) > 0$, every $i \in S_m$ with $\sigma_i = -1$ has $\Phi_\sigma(z(x_i)) > -\frac{A}{4}$, so $|A\sigma_i - \Phi_\sigma(z(x_i))| > \frac{3A}{4} \ge \frac{A}{2}$.

Hence, for every $\sigma$, at least $\min(n_m^+(\sigma), n_m^-(\sigma))$ points in $S_m$ incur squared error $\ge \frac{A^2}{4}$. Since $n_m^+ n_m^- = \min(n_m^+, n_m^-)\max(n_m^+, n_m^-) \le n_m \min(n_m^+, n_m^-)$, summing over $\sigma \in \{\pm 1\}^N$ and double-counting pairs $(i,j) \in S_m^2$ yields
$$n_m \sum_{\sigma \in \{\pm 1\}^N} \min\bigl(n_m^+(\sigma), n_m^-(\sigma)\bigr) \ge \sum_{\sigma} n_m^+(\sigma) n_m^-(\sigma) = \sum_{\substack{i,j \in S_m \\ i \ne j}} \sum_{\sigma} \mathbf{1}\{\sigma_i = +1, \sigma_j = -1\} = n_m(n_m - 1) 2^{N-2}.$$
Dividing by $n_m$ (for $n_m \ge 1$; note for $n_m = 0$ the bound $(n_m - 1)2^{N-2} \le 0$ holds trivially as well!) gives
$$\sum_{\sigma \in \{\pm 1\}^N} \sum_{i \in S_m} \bigl(A\sigma_i - \Phi_\sigma(z(x_i))\bigr)^2 \ge (n_m - 1)\, 2^{N-2} \cdot \frac{A^2}{4} = (n_m - 1)\, 2^N \frac{A^2}{16}.$$
Summing over all $m = 1, \dots, K_{\mathrm{cov}}$ and using $\sum_{m=1}^{K_{\mathrm{cov}}} (n_m - 1) = N - K_{\mathrm{cov}}$:
$$(N - K_{\mathrm{cov}}) 2^N \frac{A^2}{16} \le \sum_{\sigma}\sum_{i=1}^N \bigl(A\sigma_i - \Phi_\sigma(z(x_i))\bigr)^2 \le 2^N N \frac{A^2}{64}.$$
Dividing by $2^N A^2 > 0$ gives $\frac{N - K_{\mathrm{cov}}}{16} \le \frac{N}{64}$, i.e., $K_{\mathrm{cov}} \ge \frac{3}{4}N \ge \frac{N}{2} = \frac{\lfloor \Omega \rfloor^d}{2}$, proving (1)(i). Applying this to a minimal cover (which has size $\mathcal{N} \le K_{\mathrm{pack}}$) combines with Step 2 to prove (1)(ii): $(1 + \frac{8 L R_z}{A})^r \ge \mathcal{N} \ge \frac{3}{4}N \ge \frac{3}{2} > 1$ (since $N = \lfloor \Omega \rfloor^d \ge 2$). In particular, $(1 + \frac{8 L R_z}{A})^r > 1$ forces both $r \ge 1$ and $R_z > 0$ strictly, ensuring $\log_2(1 + \frac{8 L R_z}{A}) > 0$ and $\log_2(L R_z \sqrt{r}) \in \mathbb{R}$ are well-defined without $0/0$ or $0\cdot(-\infty)$ even at $N = 2$. Parts (2)–(4) follow immediately by taking logarithms and substituting into the quantization step bound (if $\frac{1}{2}\log_2\frac{1}{\varepsilon} + \log_2(L R_z \sqrt{r}) \ge 0$, using $r \ge r_{\mathrm{eff}} > 0$; otherwise $r b \ge 0 > r_{\mathrm{eff}}(\frac{1}{2}\log_2\frac{1}{\varepsilon} + \log_2(L R_z \sqrt{r}))$). $\square$

*(Remark: Note that Step 3 proves the sharper threshold $K_{\mathrm{cov}} \ge \frac{3}{4}N$—which guarantees $\frac{3}{4}N > 1$ for all $N \ge 2$—whereas `lipschitz_covering_lower_bound` in `CompInflation.lean` formalizes the clean $\frac{N}{2}$ threshold under the same hypothesis.)*

### 3.3 R3 Failure Modes and Counterexamples

1. **Counterexample R3-CE1 (Necessity of Norm Bound $R_z < \infty$):**
   If $\|z(x)\|_2$ is unbounded (or $R_z$ scales with $\Omega$), setting $z(x) = \Gamma x \in \mathbb{R}^d$ with $\Gamma := \frac{\sup_q \|\nabla T_q\|_\infty}{L}$ and $\Phi_q(z) := T_q(z/\Gamma)$ makes $\Phi_q$ $L$-Lipschitz and achieves $r = d = d_{\mathrm{info}}$ and $\mathrm{MSE} = 0$. Bounded $R_z$ (or bounded codebook size $2^B$) is strictly required.
2. **Counterexample R3-CE2 (Necessity of Multi-Query Uncertainty $Q$ or Lipschitz Encoder):**
   For a *single* oscillatory target $T_\nu(u) = A\sin(2\pi\nu u)$ ($\nu \gg L$), setting $z(u) := \frac{A}{L}\sin(2\pi\nu u) \in \mathbb{R}^1$ and $\Phi(z) := L z$ achieves $r = 1$, $R_z = A/L$, and $\mathrm{MSE} = 0$ with zero $\log\nu$ overhead. The R3 tax requires either multi-query unpredictability ($|Q| = 2^N$) or an $L_z$-Lipschitz encoder ($L \cdot L_z \ll \|\nabla T\|$).
3. **Counterexample R3-CE3 (High-Gradient Spike on Negligible Measure):**
   On $U \sim \mathrm{Unif}[0,1]$, $T(u) = \exp\!\bigl(-\frac{(u - 1/2)^2}{2\sigma^2}\bigr)$ with $\sigma = 10^{-6}$ has $\|\nabla T\|_\infty = (\sigma\sqrt{e})^{-1} \approx 6.07 \times 10^5 \gg L = 1$, yet the trivial constant readout $\Phi \equiv 0$ ($r = 0, L = 0, B = 0$) achieves $\mathrm{MSE} = \int_0^1 e^{-(u-1/2)^2/\sigma^2} du \le \sigma\sqrt{\pi} \approx 1.77 \times 10^{-6}$. Bulk $L^2$ oscillation—not pointwise gradient spikes—drives the $L^2$ tax.

---

## 4. R4: Shallow ReLU Collapse Boundary

Let $\sigma(t) := \max(0, t)$. Consider caching $z(X) = \hat{X} \in \mathbb{R}^{d_{\mathrm{info}}}$ ($r = d_{\mathrm{info}}$) and replacing the linear readout with a ReLU network of width $W$ and depth $D$ (where depth $D = 2$ means one hidden ReLU layer of width $W$ plus an affine output layer: $g(u) = c_0 + \sum_{j=1}^W a_j \sigma(w_j^\top u + b_j)$).

### 4.1 Depth-2 ReLU Collapse & Exact Polynomial/Frequency Floors on $[0,1]$

**Theorem R4.1 (Depth-2 ReLU Upper and Lower Bounds on $[0,1]$).**
1. **(Upper Bound: Depth-2 Collapse on $U^k$, $k \ge 2$)** *For every integer $M \ge 1$ and $k \ge 2$, the continuous piecewise-linear interpolant $g_k$ of $f(u) = u^k$ on the uniform grid $u_i = i/M$ ($0 \le i \le M$) is realized by a depth-2 ReLU network of width $W = M$ ($r = 1$) with Lipschitz constant $L_{\mathrm{NN}} = \max |g_k'| \le k$, satisfying*
   $$\|U^k - g_k\|_{L^\infty([0,1])} \le \frac{k(k-1)}{8 M^2}, \qquad \mathrm{MSE}(U^k, g_k) \le \frac{k^2(k-1)^2}{120 M^4}\Bigl(\frac{1}{2k-3} + \frac{1}{M}\Bigr).$$
   *Hence width $W = O(k^{3/4} \varepsilon^{-1/4})$ collapses $d_{\mathrm{comp}} = K$ back to $r = 1 = d_{\mathrm{info}}$.*

2. **(Exact Width-$W$ Lower Bound on $U^2$)** *Every depth-2 ReLU network $g(u) = c_0 + \sum_{j=1}^W a_j \sigma(w_j u + b_j)$ ($W \in \mathbb{N}$) satisfies*
   $$\|U^2 - g\|_{L^\infty([0,1])} \ge \frac{1}{8(W+1)^2}, \qquad \mathrm{MSE}(U^2, g) \ge \frac{1}{180(W+1)^4}.$$
   *In particular, achieving $\mathrm{MSE}(U^2, g) \le \varepsilon$ at depth $D = 2$ forces $W \ge (180\,\varepsilon)^{-1/4} - 1$.*

3. **(High-Frequency Target $T_\nu(u) = \sin(2\pi\nu u)$ at Depth $D = 2$, $2\nu \in \mathbb{N}$)** *Here $\int_0^1 |T_\nu''(u)|\,du = 8\pi\nu^2$. Uniform interpolation with $W = M$ units achieves $\|T_\nu - g\|_{L^\infty} \le \frac{\pi^2 \nu^2}{2 M^2}$ and $\mathrm{MSE} \le \frac{2\pi^4 \nu^4}{15 M^4}$, so $W = O(\nu \varepsilon^{-1/4})$ suffices. Conversely, any depth-2 width-$W$ ReLU network incurs*
   $$\|T_\nu - g\|_{L^\infty([0,1])} \ge \begin{cases} \frac{1}{2}, & W < 2\nu, \\ \frac{\sqrt{2}\pi^2 \nu^2}{32(W+2\nu)^2}, & \forall W \ge 0, \end{cases} \qquad \mathrm{MSE}(T_\nu, g) \ge \frac{\pi^4 \nu^4}{2880(W+2\nu)^4},$$
   *proving $W = \Theta(\nu \varepsilon^{-1/4})$ is tight as $\varepsilon \to 0$.*

*Proof.*
**(1)** Let $h = 1/M$, $u_i = i h$, and let $s_i := \frac{u_{i+1}^k - u_i^k}{h} \in [0, k]$ be the chord slope on $[u_i, u_{i+1}]$. Since $g_k(0) = 0$, $g_k(u) = s_0 \sigma(u) + \sum_{i=1}^{M-1}(s_i - s_{i-1})\sigma(u - u_i)$ uses $M$ ReLUs. By the linear-interpolation remainder formula, for $u \in [u_i, u_{i+1}]$,
$$|u^k - g_k(u)| \le \frac{1}{2}(u - u_i)(u_{i+1} - u)\max_{\xi \in [u_i, u_{i+1}]} k(k-1)\xi^{k-2} \le \frac{k(k-1)}{2}\Bigl(\frac{i+1}{M}\Bigr)^{k-2}(u - u_i)(u_{i+1} - u).$$
Maximizing $(u-u_i)(u_{i+1}-u) \le \frac{h^2}{4}$ gives the $L^\infty$ bound $\frac{k(k-1)}{8 M^2}$. Squaring and integrating over $[u_i, u_{i+1}]$ using $\int_0^h t^2(h-t)^2\, dt = \frac{h^5}{30}$ gives
$$\mathrm{MSE}(U^k, g_k) \le \frac{k^2(k-1)^2}{120 M^4}\cdot \frac{1}{M}\sum_{j=1}^M \Bigl(\frac{j}{M}\Bigr)^{2k-4} \le \frac{k^2(k-1)^2}{120 M^4}\Bigl(\int_0^1 t^{2k-4}\,dt + \frac{1}{M}\Bigr) = \frac{k^2(k-1)^2}{120 M^4}\Bigl(\frac{1}{2k-3} + \frac{1}{M}\Bigr).$$

**(2)** Each term $a_j \sigma(w_j u + b_j)$ is affine everywhere except possibly at a single hinge $u = -b_j/w_j$ (when $w_j \ne 0$). Collecting the distinct hinges inside $(0,1)$ (at most $W$ points) together with $0$ and $1$ produces a monotone grid $0 = u_0 \le u_1 \le \dots \le u_{W+1} = 1$ (padding duplicate endpoints if fewer than $W$ interior hinges occur) partitioning $[0,1]$ into $W + 1$ intervals $[u_{i-1}, u_i]$ of lengths $h_i = u_i - u_{i-1} \ge 0$ ($\sum_{i=1}^{W+1} h_i = 1$) on each of which $g(u) = \alpha_i u + \beta_i$ is affine.
- *$L^\infty$ floor*: On any interval $[a, b]$ ($a \le b$) and any affine $\alpha u + \beta$, let $d(u) := u^2 - (\alpha u + \beta)$ and $u_{\mathrm{mid}} := \frac{a+b}{2}$. Because the discrete second difference annihilates affine functions,
  $$d(a) - 2 d(u_{\mathrm{mid}}) + d(b) = a^2 - 2\Bigl(\frac{a+b}{2}\Bigr)^2 + b^2 = \frac{(b-a)^2}{2}.$$
  By the triangle inequality, $\frac{(b-a)^2}{2} \le 4\max_{u \in \{a, u_{\mathrm{mid}}, b\}}|d(u)|$, so $\max_{[a,b]}|d| \ge \frac{(b-a)^2}{8}$. Since $\sum_{i=1}^{W+1} h_i = 1$, pigeonhole yields some $i$ with $h_i \ge \frac{1}{W+1}$, giving $\|U^2 - g\|_{L^\infty([0,1])} \ge \frac{1}{8(W+1)^2}$.
- *$L^2$ floor*: On $[u_{i-1}, u_i]$ ($h_i > 0$), substitute $u = \frac{u_{i-1}+u_i}{2} + \frac{h_i}{2} s$ ($s \in [-1,1]$). Since $s^2 - \frac{1}{3}$ is $L^2([-1,1])$-orthogonal to $\{1, s\}$,
  $$\int_{u_{i-1}}^{u_i} (u^2 - \alpha_i u - \beta_i)^2\, du \ge \frac{h_i}{2}\int_{-1}^1 \Bigl(\frac{h_i^2}{4}\Bigl(s^2 - \frac{1}{3}\Bigr)\Bigr)^2 ds = \frac{h_i^5}{32}\cdot\frac{8}{45} = \frac{h_i^5}{180}.$$
  Summing over $i = 1, \dots, W+1$ and applying Jensen's inequality to the convex function $t \mapsto t^5$ gives
  $$\mathrm{MSE}(U^2, g) \ge \frac{1}{180}\sum_{i=1}^{W+1} h_i^5 \ge \frac{W+1}{180}\Bigl(\frac{1}{W+1}\sum_{i=1}^{W+1} h_i\Bigr)^5 = \frac{1}{180(W+1)^4}.$$

**(3)** For $T_\nu(u) = \sin(2\pi\nu u)$ ($2\nu \in \mathbb{N}$), $|T_\nu''(u)| = 4\pi^2\nu^2 |\sin(2\pi\nu u)|$, so $\int_0^1 |T_\nu''| du = 8\pi\nu^2$ and $\|T_\nu''\|_\infty = 4\pi^2\nu^2$. On each uniform subinterval $[u_i, u_{i+1}]$ of length $h = 1/M$, $|T_\nu(u) - g(u)| \le \frac{1}{2}(u-u_i)(u_{i+1}-u)\|T_\nu''\|_\infty = 2\pi^2\nu^2(u-u_i)(u_{i+1}-u)$, giving $\|T_\nu - g\|_{L^\infty} \le \frac{4\pi^2\nu^2}{8 M^2} = \frac{\pi^2\nu^2}{2 M^2}$. Squaring and integrating via $\int_0^h t^2(h-t)^2\,dt = \frac{h^5}{30}$ gives per-interval $L^2$ error $\le \frac{1}{4}\cdot\frac{1}{30}\|T_\nu''\|_\infty^2 h^5 = \frac{1}{120}(4\pi^2\nu^2)^2 M^{-5} = \frac{(2\pi^2\nu^2)^2}{30 M^5} = \frac{2\pi^4\nu^4}{15 M^5}$, and summing over all $M$ intervals yields $\mathrm{MSE}(T_\nu, g) \le \frac{2\pi^4\nu^4}{15 M^4}$. For the lower bounds: if $W < 2\nu$, at least one of the $2\nu$ half-periods $I_k = [\frac{k}{2\nu}, \frac{k+1}{2\nu}]$ contains no interior hinge, so on $I_k$, $|d(a) - 2d(u_{\mathrm{mid}}) + d(b)| = |0 - 2(-1)^k + 0| = 2$, forcing $\|T_\nu - g\|_{L^\infty} \ge 2/4 = 1/2$. In general, consider the set $E := \{u \in [0,1] : |\sin(2\pi\nu u)| \ge 1/\sqrt{2}\}$, a union of $2\nu$ intervals of total measure $1/2$. Subdividing $E$ at the $\le W$ hinges of $g$ yields $M' \le W + 2\nu$ subintervals $J_\ell = [a_\ell, b_\ell]$ of lengths $h_\ell$ ($\sum h_\ell = 1/2$) on each of which $g$ is affine and $|T_\nu''(u)| \ge c_2 := 2\sqrt{2}\pi^2\nu^2$ with constant sign.
- Using $f(m - h/2) - 2f(m) + f(m + h/2) = \int_0^{h/2}(\frac{h}{2} - s)(f''(m+s) + f''(m-s))\,ds$ ($\int_0^{h/2} 2(\frac{h}{2}-s)\,ds = \frac{h^2}{4}$), we have $\max_{J_\ell}|T_\nu - g| \ge \frac{c_2 h_\ell^2}{16}$; with $\max_\ell h_\ell \ge \frac{1}{2(W+2\nu)}$, this gives $\|T_\nu - g\|_{L^\infty} \ge \frac{\sqrt{2}\pi^2\nu^2}{32(W+2\nu)^2}$.
- For $L^2$, project $T_\nu$ onto the local normalized Legendre quadratic $\tilde{P}_{2, J_\ell}(u) = \sqrt{\frac{5}{h_\ell}} P_2(s) = \sqrt{\frac{5}{h_\ell}}\frac{d^2}{ds^2}[\frac{(s^2-1)^2}{8}]$ ($u = m_\ell + \frac{h_\ell}{2}s$). Integrating by parts twice yields
  $$\int_{J_\ell} (T_\nu - \alpha_\ell u - \beta_\ell)^2\, du \ge \Bigl(\int_{J_\ell} T_\nu\,\tilde{P}_{2, J_\ell}\, du\Bigr)^2 = \Bigl(\frac{\sqrt{5}\,h_\ell^{5/2}}{64}\int_{-1}^1 T_\nu''\Bigl(m_\ell + \frac{h_\ell}{2}s\Bigr)(1-s^2)^2\, ds\Bigr)^2 \ge \frac{c_2^2 h_\ell^5}{720}.$$
  Summing over $\ell = 1,\dots,M'$ and applying Jensen ($(\sum h_\ell)^5 = 2^{-5}$) gives $\mathrm{MSE}(T_\nu, g) \ge \frac{8\pi^4\nu^4}{720 \cdot 32 (W+2\nu)^4} = \frac{\pi^4\nu^4}{2880(W+2\nu)^4}$. $\square$

---

### 4.2 Deep ReLU Collapse: Yarotsky (2017) Squaring & Exponential Depth Tradeoff

**Theorem R4.2 (Deep ReLU Collapse — Yarotsky Construction).**
*Define the hat function $g(x) := 2\sigma(x) - 4\sigma(x - 1/2) + 2\sigma(x - 1)$ and $s$-fold iterates $g_s := g^{\circ s}$ ($g_0(x) = x$). On $[0,1]$, the partial sum*
$$f_m(x) := x - \sum_{s=1}^m 4^{-s} g_s(x)$$
*satisfies:*
1. **$0 \le f_m(x) - x^2 \le 4^{-(m+1)}$ for all $x \in [0,1]$**;
2. **For any $\delta \in (0, 1)$, setting $m := \max\bigl(1, \lceil \frac{1}{2}\log_2(1/\delta)\rceil - 1\bigr)$ (so $4^{-(m+1)} \le \delta$ and $D = m + 1 = \max\bigl(2, \lceil \frac{1}{2}\log_2(1/\delta)\rceil\bigr)$, with $D = \lceil \frac{1}{2}\log_2(1/\delta)\rceil \ge 2$ whenever $\delta \in (0, 1/4)$), $f_m$ is realized by a ReLU network of depth $D = m + 1$ and width $\le 4$, using exactly $8$ nonzero weights/biases for $m = 1$ and $15m - 7$ nonzero weights/biases for $m \ge 2$ (hence $\le 15m$ for all $m \ge 1$; for $\delta \in [1/4, 1)$, $f_0(x) = x$ already achieves error $\le 1/4 \le \delta$ at depth $D = 1$ with $1$ weight)**;
3. **Via $\mathrm{mult}_m(x,y) := \mathrm{clamp}_{[0,1]}\!\bigl(f_m(\frac{x+y}{2}) - f_m(\frac{|x-y|}{2})\bigr)$ and binary exponentiation, $U^k$ is approximated on $[0,1]$ to $L^\infty$ error $\delta$ in depth $D = O\bigl(\log k \cdot \log(k/\delta)\bigr)$ and width $O(1)$ ($\le 12$), collapsing the depth-2 polynomial width $\Omega(\varepsilon^{-1/4})$ to $\mathrm{polylog}(1/\varepsilon)$ total parameters.**

*Proof.*
**(1)** We prove by induction on $m \ge 0$ that $f_m$ is the continuous piecewise-linear interpolant of $x^2$ at the dyadic nodes $k 2^{-m}$ ($0 \le k \le 2^m$). For $m = 0$, $f_0(x) = x$ interpolates $x^2$ at $\{0, 1\}$. For the inductive step, $g_m(x)$ vanishes at all multiples of $2^{-(m-1)}$, equals $1$ at all odd multiples of $2^{-m}$, and is linear on every $[k 2^{-m}, (k+1) 2^{-m}]$. On each interval $I_k = [k 2^{-(m-1)}, (k+1) 2^{-(m-1)}]$ of length $h = 2^{-(m-1)}$, the midpoint $x_{\mathrm{mid}} = (k + \frac{1}{2})2^{-(m-1)}$ satisfies
$$f_{m-1}(x_{\mathrm{mid}}) - x_{\mathrm{mid}}^2 = \frac{1}{2}\bigl(k^2 + (k+1)^2\bigr)h^2 - \Bigl(k + \frac{1}{2}\Bigr)^2 h^2 = \frac{h^2}{4} = 4^{-m},$$
so $f_m(x_{\mathrm{mid}}) = f_{m-1}(x_{\mathrm{mid}}) - 4^{-m} g_m(x_{\mathrm{mid}}) = x_{\mathrm{mid}}^2$. Since $f_m$ is affine on each half-interval of length $2^{-m}$ and agrees with $x^2$ at all nodes $k 2^{-m}$, convexity of $x^2$ yields $0 \le f_m(x) - x^2 \le \frac{(2^{-m})^2}{4} = 4^{-(m+1)}$.

**(2)** Layer 1 computes $h_{1,1} = \sigma(x)$, $h_{1,2} = \sigma(x - 1/2)$, $h_{1,3} = \sigma(x - 1)$ (width 3; $3$ weights $+ 2$ biases $= 5$ nonzero parameters), from which $g_1(x) = 2h_{1,1} - 4h_{1,2} + 2h_{1,3}$ and $f_1(x) = h_{1,1} - \frac{1}{4}g_1(x) = \frac{1}{2}h_{1,1} + h_{1,2} - \frac{1}{2}h_{1,3} \ge 0$ are affine combinations. For $m = 1$, the output affine layer computes $f_1$ with $3$ nonzero weights, yielding $5 + 3 = 8$ nonzero weights/biases and depth $D = 2$. For $m \ge 2$, Layer 2 takes $h_1 \in \mathbb{R}^3$ and computes $\sigma(g_1)$ ($3$ weights), $\sigma(g_1 - 1/2)$ ($3$ weights $+ 1$ bias), $\sigma(g_1 - 1)$ ($3$ weights $+ 1$ bias), and $\sigma(f_1) = f_1$ ($3$ weights), totaling $3 + 4 + 4 + 3 = 14$ nonzero parameters. Inductively, hidden layer $s$ ($3 \le s \le m$) takes $h_{s-1} \in \mathbb{R}^4$ and computes $\sigma(g_{s-1})$, $\sigma(g_{s-1}-1/2)$, $\sigma(g_{s-1}-1)$ (each using $3$ weights from $h_{s-1,1..3}$, plus $0, 1, 1$ biases), and $\sigma(f_{s-1}) = \sigma(h_{s-1,4} - 4^{-(s-1)} g_{s-1}) = f_{s-1}$ (since $f_{s-1} \ge x^2 \ge 0$, using $4$ weights from all $4$ units of $h_{s-1}$), totaling $3 + 4 + 4 + 4 = 15$ nonzero parameters per layer. The output affine layer computes $f_m = h_{m,4} - 4^{-m} g_m$ using $4$ nonzero weights, giving depth $D = m + 1$ and total nonzero weights/biases $5 + 14 + 15(m - 2) + 4 = 15m - 7 \le 15m$ for $m \ge 2$. Choosing $m = \max(1, \lceil \frac{1}{2}\log_2(1/\delta)\rceil - 1)$ for $\delta \in (0, 1)$ ensures $m + 1 \ge \frac{1}{2}\log_2(1/\delta)$ and hence error $\le 4^{-(m+1)} \le \delta$ (while for $\delta \in [1/4, 1)$, $f_0(x) = x$ at depth $1$ already satisfies $0 \le x - x^2 \le 1/4 \le \delta$).

**(3)** For $x, y \in [0,1]$, $\frac{x+y}{2}, \frac{|x-y|}{2} \in [0,1]$ (with $|t| = \sigma(t) + \sigma(-t)$), and $\mathrm{clamp}_{[0,1]}(t) = \sigma(t) - \sigma(t-1)$ never increases distance to $xy \in [0,1]$, so $|\mathrm{mult}_m(x,y) - xy| \le 2\cdot 4^{-(m+1)}$. Moreover, for $x, \tilde{x}, y, \tilde{y} \in [0,1]$, $|\mathrm{mult}_m(\tilde{x}, \tilde{y}) - xy| \le |\tilde{x}-x| + |\tilde{y}-y| + 2\cdot 4^{-(m+1)}$. Binary exponentiation computes $u^k$ in at most $2\lfloor \log_2 k \rfloor$ chained multiplications; by induction on the multiplication DAG, the accumulated error for $u^k$ is at most $2(k-1)4^{-(m+1)}$. Choosing $4^{m+1} \ge 2k/\delta$ ($m = O(\log(k/\delta))$) yields error $\le \delta$ in depth $D = O(\log k \cdot \log(k/\delta))$ and constant width $O(1)$. $\square$

---

### 4.3 Multi-Dimensional $d_{\mathrm{info}} > 1$, Hard Cache-Bit Floor, and Failure Modes

1. **Barron vs. Sobolev Regimes ($d_{\mathrm{info}} = d > 1$, Depth $D = 2$):**
   - *Barron functions* ($\|f\|_{\mathcal{B}} := \int_{\mathbb{R}^d} \|\omega\|_1 |\hat{f}(\omega)|\, d\omega \le C$, where $\hat{f}$ is the Fourier transform of an extension to $\mathbb{R}^d$) admit a dimension-free width rate $\mathrm{MSE} \le O(C^2 / W)$ at depth $D = 2$ (Maurey–Barron random sampling), though $C$ itself grows rapidly with frequency/oscillation.
   - *General Sobolev balls $W^{s,\infty}([0,1]^d)$* ($s$-smooth functions with unit norm) suffer the curse of dimensionality at *any* fixed or bounded-complexity ReLU architecture: achieving $L^\infty$ error $\delta$ with continuous parameter selection (or bounded-bit ReLU weights) requires $P = \Omega(\delta^{-d/s})$ parameters (DeVore–Howard–Micchelli; $P\log P = \Omega(\delta^{-d/s})$ for arbitrary ReLU graphs via VC-dimension, Yarotsky 2017).

2. **Theorem R4.3 (Unbreakable Information-Theoretic Cache-Bit Floor).**
   *Let $X \sim \mathrm{Unif}([0,1]^d)$ ($d = d_{\mathrm{info}}$) and $T_q(X) = X_q$ ($q = 1, \dots, d$). For **any** $B$-bit quantized cache $\hat{z}$ ($|\mathcal{C}| \le 2^B$) and **any** measurable readout $\Phi_q$ (arbitrary ReLU width $W$ and depth $D$), achieving $\max_{1 \le q \le d}\mathrm{MSE}(X_q, \Phi_q(\hat{z})) \le \varepsilon$ forces*
   $$B \ge \max\!\left(\frac{d_{\mathrm{info}}}{2}\log_2\frac{1}{2\pi e \varepsilon},\; \frac{d_{\mathrm{info}}}{2}\log_2\frac{1}{\varepsilon} - \log_2\bigl(2 \omega_{d_{\mathrm{info}}} (2 d_{\mathrm{info}})^{d_{\mathrm{info}}/2}\bigr)\right) = \frac{d_{\mathrm{info}}}{2}\log_2\frac{1}{\varepsilon} - O(d_{\mathrm{info}}).$$
   *Proof.* Immediate from Theorem R2.2(1) applied to $T(X) = X \sim \mathrm{Unif}([0,1]^d)$, which has $h_2(X) = 0$ and density $f_X \equiv 1$ ($M_T = 1$), since $\Phi(\hat{z}(X))$ takes at most $2^B$ values in $\mathbb{R}^d$. $\square$

3. **Failure Mode R4-FM1 (Norm-Constrained / Weight-Bounded ReLU Reverts to R3):**
   In Theorem R4.1(3), depth-2 collapse of $\sin(2\pi\nu u)$ requires ReLU weights with Lipschitz constant $L_{\mathrm{NN}} \ge 2\pi\nu$ (and total variation / path norm $\sum_j |a_j| |w_j| \ge 8\pi\nu^2$). If the ReLU readout is constrained to bounded Lipschitz norm $L_{\mathrm{NN}} \le L \ll \nu$, Theorem R3.2 immediately re-applies and forces cache dimension $r$ or radius $R_z$ inflation.

---

## 5. R5: Scope Boundary & Lean 4 Formalization Alignment

The Lean 4 formalization in `MathLab.Projects.LateQueryFrontier.CompInflation` (`CompInflation.lean`) targets the core structural obstructions with **no custom axioms** (`[propext, Classical.choice, Quot.sound]` only) and **no `MeasureTheory` dependency**. The table below delineates the exact boundary between what is formally verified by the Lean 4 kernel and what is proved analytically in this monograph:

| Mathematical Result | Status | Lean Theorem Name (`CompInflation.lean`) | Scope / Idealization Boundary |
|---|---|---|---|
| **R2: Subspace-rank irreducible error** ($\sum_{i=1}^K \|T_i - \sum_j W_{ij}\varphi_j\|^2 \ge \varepsilon_0 > 0$ for $r < K$) | **Lean-Verified** | `linear_readout_irreducible_error` | Formalized in arbitrary real `InnerProductSpace ℝ H` via `Matrix.posDef_gram_of_linearIndependent`, spectral theorem, rank-nullity, and Pythagoras ($\varepsilon_0 = \lambda_{\min}(G)$). |
| **R2: Monomial linear independence on $[0,1]$** ($\{t \mapsto t^{k+1}\}_{k=0}^{K-1}$) | **Lean-Verified** | `monomial_linearIndependent_Icc` | Formalized via `Matrix.det_vandermonde` on $K$ distinct nodes in $(0,1]$. |
| **R2: Empirical $L^2$ monomial irreducible error** ($r < K$) | **Lean-Verified** | `monomial_empirical_L2_irreducible_error` | Instantiates `linear_readout_irreducible_error` on `EuclideanSpace ℝ (Fin K)` over $K$ distinct nodes in $(0,1]$ (discrete empirical measure rather than Lebesgue $L^2[0,1]$). |
| **R2: Ky Fan tail bound, affine covariance $\Sigma$, Legendre $c_{kj}$, $\det(\Sigma)$, Hilbert $\lambda_K(G)$ bounds, Shannon EPI & scalar-quantization bit-rate inflation** | **Analytic (Proved in `THEORY.md`)** | — | Proved in full in Section 2 using continuous $L^2([0,1])$, Rodrigues/Beta integration, Laplace integral, Shannon differential entropy, and ball-volume packing. |
| **R3: Lipschitz metric distortion & radius bound** ($|T(x) - T(x')| - 2\delta \le L\|z(x)-z(x')\| \le 2 L R_z$) | **Lean-Verified** | `lipschitz_radius_inflation` | Formalized for any `NormedAddCommGroup E` and $L \ge 0$ without assuming $R_z$ compactness or differentiability. |
| **R3: Multi-query $L$-Lipschitz $L^2$ covering bound** ($K_{\mathrm{cov}} \ge N/2$) | **Lean-Verified** | `lipschitz_covering_lower_bound` | Formalized for finite point sets `z : Fin N → E` covered by $K_{\mathrm{cov}}$ balls of radius $A/(4L)$ via exact `Finset` Rademacher double-counting (`n_m^+ n_m^-`). |
| **R3: Smooth bump family $T_\sigma$, Fubini point extraction, Euclidean ball-volume ratio $(1 + 8 L R_z / A)^r \ge \mathcal{N}$, chaotic map scaling** | **Analytic (Proved in `THEORY.md`)** | — | Proved in full in Section 3 using $C_c^\infty((0,1)^d)$ / PL bumps, Fubini over $\tilde{C}_{\mathbf{k}}$, and $r$-dimensional Lebesgue volume packing. |
| **R4: 3-point affine floor on $[u_0, u_1]$** ($\ge (u_1 - u_0)^2 / 8$ vs. $u^2$) | **Lean-Verified** | `affine_quadratic_interval_floor` | Formalized directly via discrete second-difference algebraic identity at $\{u_0, \frac{u_0+u_1}{2}, u_1\}$. |
| **R4: Piecewise-affine partition floor on $[0,1]$** ($\ge \frac{1}{8(W+1)^2}$) | **Lean-Verified** | `piecewise_affine_quadratic_floor` | Formalizes the monotone $(W+1)$-interval telescoping pigeonhole bound + `affine_quadratic_interval_floor`. |
| **R4: Depth-2 width-$W$ ReLU $L^\infty$ floor on $[0,1]$** ($\ge \frac{1}{8(W+1)^2}$) | **Lean-Verified** | `relu_depth2_quadratic_floor` | Reduces arbitrary depth-2 width-$W$ ReLU sums `c₀ + ∑ aⱼ max(0, wⱼ u + bⱼ)` to `piecewise_affine_quadratic_floor` by sorting interior breakpoints. |
| **R4: Depth-2 $L^2$ floor $\frac{1}{180(W+1)^4}$, $U^k$ & $\sin(2\pi\nu u)$ bounds, Yarotsky depth-$D$ collapse, Barron/Sobolev & bit-rate floor** | **Analytic (Proved in `THEORY.md`)** | — | Proved in full in Section 4 via Legendre $\tilde{P}_2$ projection, Jensen's inequality, dyadic tent iteration, and Theorem R2.2(1). |
