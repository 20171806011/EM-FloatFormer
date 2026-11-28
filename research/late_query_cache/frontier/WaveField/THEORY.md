# Evanescent-Mode Polylogarithmic Scaling, Near-Field Probe Jumps, and Failure Boundaries in Late-Query Wave & Electromagnetic Caching

**Conjecture 1 — Formal Mathematical & Physical Analysis (`THEORY.md`)**

---

## Executive Summary

Consider an unknown physical source or linear medium $X$ supported inside a ball $B_a \subset \mathbb{R}^d$ ($d \in \{2, 3\}$, radius $a > 0$), whose radiated Laplace ($k = 0$), scalar Helmholtz ($k > 0$), or vector Maxwell ($k > 0$) field $u = \mathcal{S} X$ is compressed into a fixed-length $b$-bit late-query cache $E(X) \in \{0,1\}^b$ prior to revealing exterior query locations $x \in \mathbb{R}^d \setminus B_R$ ($R > a$, contraction ratio $\rho := a/R \in (0, 1)$). Achieving outer-sphere mean-squared error $\mathrm{MSE}(u, \hat{u}; S_R) \le \varepsilon$ exhibits a sharp **two-regime transition** governed by the propagating degrees of freedom $N_{\text{DOF}}^{\mathbb{R}}(ka)$ at moderate $\varepsilon$ and by geometric evanescent-mode decay $\sigma_\ell \asymp \ell^{-\beta} \rho^\ell$ as $\varepsilon \to 0$:

1. **Propagating Regime ($\varepsilon \gtrsim \varepsilon^*$, cutoff $\ell_{\text{cut}} \lesssim ka$):**
   $$b_Q(\varepsilon) = \frac{N_{\text{DOF}}^{\mathbb{R}}(ka)}{2} \log_2\!\left(\frac{1}{\varepsilon}\right) + O\!\left(N_{\text{DOF}}(ka)\log N_{\text{DOF}}(ka)\right),$$
   linear in $\log_2(1/\varepsilon)$, with $N_{\text{DOF}}(ka) = \Theta(ka)$ in 2D and $\Theta((ka)^2)$ in 3D. When $R \gg a$ and $kR \gg 1$, an intermediate caustic regime ($ka \lesssim \ell \lesssim kR$) bridges the plateau and the evanescent tail with Lambert-$W$ cutoff $\ell_c \sim \frac{\ln(1/\varepsilon)}{2 W(\ln(1/\varepsilon)/(eka))}$.
2. **Evanescent Regime ($\varepsilon \ll \varepsilon^*$, $\log_2(1/\varepsilon^*) = \Theta(kR \log_2(R/a))$):**
   Geometric decay $\sigma_\ell = \Theta(\ell^{-\beta}\rho^\ell)$ breaks any finite-dimensional $O(\log(1/\varepsilon))$ bound while remaining strictly polylogarithmic:
   - **2D ($m$ real modes per order $\ell \ge 1$):**
     $$b_Q(\varepsilon) = \frac{m}{8\log_2(R/a)} \left(\log_2 \frac{1}{\varepsilon}\right)^2 + O\!\left(\log\frac{1}{\varepsilon} \cdot \log\log\frac{1}{\varepsilon}\right).$$
     For full real 2D multipoles ($\cos\ell\theta, \sin\ell\theta$, $m = 2$), the exact leading constant is $\frac{1}{4\log_2(R/a)}$; Conjecture 1's denominator $8\log_2(R/a)$ corresponds to $m = 1$ real mode per order (e.g., even-parity / cosine-only or single-quadrature sector).
   - **3D ($m_0(2\ell+1)$ real modes per order):**
     $$b_Q(\varepsilon) = \frac{m_0}{24(\log_2(R/a))^2} \left(\log_2 \frac{1}{\varepsilon}\right)^3 + O\!\left(\left(\log\frac{1}{\varepsilon}\right)^2 \log\log\frac{1}{\varepsilon}\right),$$
     yielding $\frac{(\log_2(1/\varepsilon))^3}{24(\log_2(R/a))^2}$ for 3D real scalar fields ($m_0 = 1$), $\frac{(\log_2(1/\varepsilon))^3}{12(\log_2(R/a))^2}$ for 3D real Maxwell (TE+TM, $m_0 = 2$), and $\frac{(\log_2(1/\varepsilon))^3}{6(\log_2(R/a))^2}$ for 3D complex Maxwell phasors ($m_0 = 4$).
3. **Active Near-Field Probe Jump (R2):** Reducing the effective query radius from $R_1$ to $R_2 \in (a, R_1)$ strictly lowers $\log_2(R/a)$ ($0 < \log_2(R_2/a) < \log_2(R_1/a)$), producing a discontinuous upward jump in the leading polylog coefficient by factor $\left(\frac{\log_2(R_1/a)}{\log_2(R_2/a)}\right)^{d-1} > 1$, degrading fixed-budget MSE to $\varepsilon^{\alpha_d + o(1)}$ ($\alpha_d = (\log_2(R_2/a)/\log_2(R_1/a))^{(d-1)/d}$ under optimal $R_2$-reallocation, or $\varepsilon^{\log_2(R_2/a)/\log_2(R_1/a)}$ under non-adaptive $R_1$-truncation), and diverging as $(R_2 - a)^{-(d-1)}$ as $R_2 \to a^+$.
4. **Failure Boundaries (R3):** We establish rigorous breakdown mechanisms and explicit counterexamples under (i) nonlinear unknown high-contrast permittivity $\epsilon(r)$ (whispering-gallery Mie resonances with $|b_\ell| = 1$ at $\ell > kR$ and Mandache logarithmic inverse instability), (ii) non-Gaussian/heavy-tail source priors (infinite-variance Cauchy, finite-variance ultra-heavy-tail forcing $b_Q(\varepsilon) = \Omega(1/\varepsilon)$, and sparse spike-and-slab breaking the Gaussian lower bound), and (iii) worst-case $L^\infty(S_R)$ or super-directive queries ($\Theta(N_{\text{act}})$ Dirichlet-kernel amplification and $\rho^{-2\ell}$ inward-continuation blowup).

---

## 1. Problem Formulation & Physical Operator Architecture

### 1.1 Geometric Domain, Exterior Monotonicity, and Error Metric

Let $d \in \{2, 3\}$. Fix a source radius $a > 0$ and an outer query radius $R > a$, and define the geometric contraction ratio and logarithmic separation
$$\rho := \frac{a}{R} \in (0, 1), \qquad \frac{R}{a} = \rho^{-1} > 1, \qquad \log_2\!\left(\frac{R}{a}\right) > 0.$$
Let $B_a := \{y \in \mathbb{R}^d : |y| \le a\}$ denote the closed source ball and $S_r := \{x \in \mathbb{R}^d : |x| = r\}$ denote the sphere of radius $r$. An unknown source/excitation $X$ supported in $B_a$ generates an exterior field $u = \mathcal{S} X$ satisfying one of the linear elliptic/wave PDEs in $\mathbb{R}^d \setminus B_a$:
- **Laplace ($k = 0$):** $\Delta u = 0$ in $\mathbb{R}^d \setminus B_a$ with exterior regularity ($u(x) = O(1)$ as $|x|\to\infty$ in $d=2$; $u(x) \to 0$ in $d=3$).
- **Scalar Helmholtz ($k > 0$):** $(\Delta + k^2) u = 0$ in $\mathbb{R}^d \setminus B_a$ with the Sommerfeld radiation condition $\lim_{r\to\infty} r^{(d-1)/2}(\partial_r u - iku) = 0$.
- **Vector Maxwell ($k > 0$, $d = 3$):** $\nabla \times \mathbf{E} = i\omega \mu_0 \mathbf{H}$, $\nabla \times \mathbf{H} = -i\omega\varepsilon_0 \mathbf{E}$ in $\mathbb{R}^3 \setminus B_a$ ($k = \omega\sqrt{\mu_0\varepsilon_0}$) with the Silver–Müller radiation condition.

**Why bounding MSE on $S_R$ controls all outer radii $r \ge R$.** For every radiating/regular exterior solution, each multipole mode at order $\ell$ propagates outward from $S_R$ to $S_r$ ($r \ge R$) with radial attenuation ratio strictly bounded by $1$:
- For Laplace ($k=0$), surface-averaged spherical harmonics scale as $(R/r)^\ell \le 1$ ($d=2$) and $(R/r)^{\ell+1} \le 1$ ($d=3$).
- For Helmholtz/Maxwell ($k>0$), by Nicholson's integral formula in 2D,
  $$|H_\ell^{(1)}(x)|^2 = J_\ell(x)^2 + Y_\ell(x)^2 = \frac{8}{\pi^2}\int_0^\infty K_0(2x\sinh t)\cosh(2\ell t)\,dt,$$
  which is strictly decreasing in $x > 0$ since $K_0$ is strictly decreasing; in 3D, $|h_\ell^{(1)}(x)|^2 = \sum_{m=0}^\ell \frac{(2m)!(\ell+m)!}{2^{2m}(m!)^2(\ell-m)!} x^{-2m-2}$ is a finite polynomial in $x^{-2}$ with strictly positive coefficients, hence strictly decreasing in $x > 0$. Thus $|H_\ell^{(1)}(kr)| \le |H_\ell^{(1)}(kR)|$ and $|h_\ell^{(1)}(kr)| \le |h_\ell^{(1)}(kR)|$ for all $r \ge R$.

Consequently, outer-sphere control on $S_R$ uniformly dominates all exterior spheres $S_r$ ($r \ge R$). We measure reconstruction distortion via the normalized outer-sphere MSE:
$$\mathrm{MSE}(u, \hat{u}; S_R) := \mathbb{E}\!\left[\frac{1}{|S_R|}\int_{S_R} |u(x) - \hat{u}(x)|^2\, dS(x)\right] \le \varepsilon,$$
or equivalently the unnormalized $L^2(S_R)$ error $\mathbb{E}[\|u - \hat{u}\|_{L^2(S_R)}^2] \le \varepsilon$ (the factor $|S_R|$ rescales the modal variance constant $c$ only and leaves all leading polylogarithmic coefficients invariant). Taking an $L^2(S_1)$-orthonormal spherical/circular harmonic basis $\{Y_\ell^m\}$ with real surface measure $d\Omega(\hat{x})$, Parseval's identity gives:
$$\|u - \hat{u}\|_{L^2(S_R)}^2 = \sum_{\ell \ge \ell_0}\sum_{m=1}^{m_\ell} |y_{\ell,m} - \hat{y}_{\ell,m}|^2, \qquad y_{\ell,m} := \langle u|_{S_R}, R^{-(d-1)/2} Y_\ell^m(\cdot/R) \rangle_{L^2(S_R)}.$$

### 1.2 Fixed-Length Late-Query Cache & Connection to EM-FloatFormer

**Definition 1.1 (Fixed-Length Late-Query Cache).** A $b$-bit late-query cache consists of an encoder $E : \mathcal{X} \to \{0, 1\}^b$ (executed on the source/scatterer $X$ before queries arrive) and a query decoder $D : \{0, 1\}^b \times \mathcal{Q} \to \mathbb{R}$ (or $\mathbb{C}$, $\mathbb{C}^3$). In the outer-sphere formulation, decoding at all $\hat{x} \in S_R$ produces a function $\hat{u} = \tilde{D}(E(X)) \in L^2(S_R)$ taking at most $2^b$ distinct values. We define:
- **Stochastic prior regime ($X \sim \mu$, e.g. Gaussian source):** $b_Q(\varepsilon)$ is the minimum integer $b \ge 0$ such that there exist $(E, \tilde{D})$ with $\mathbb{E}_{X \sim \mu}[\|u_X - \tilde{D}(E(X))\|_{L^2(S_R)}^2] \le \varepsilon$.
- **Deterministic ellipsoid regime ($X \in \mathcal{B}$, $\mathcal{S}(\mathcal{B}) = \mathcal{E} \subset L^2(S_R)$):** $b_Q(\varepsilon) = \lceil H_{\sqrt{\varepsilon}}(\mathcal{E}) \rceil$, where $H_\delta(\mathcal{E}) := \log_2 N_\delta(\mathcal{E})$ is the Kolmogorov $\delta$-entropy (base-2 logarithm of the minimum external $\delta$-covering number of $\mathcal{E}$ in $L^2(S_R)$).

**Connection to Electromagnetic Scattering Surrogates (EM-FloatFormer).** Neural wave/EM surrogates such as EM-FloatFormer map an input geometry/source $X$ supported in $B_a$ into a fixed-size latent representation (token sequence or KV-cache of $b$ quantized bits) prior to late-arriving spatial field queries $x \in \mathbb{R}^d \setminus B_R$. By Theorem 1.1–5.1 below:
1. At engineering tolerances ($\varepsilon \gtrsim \varepsilon^*$), a fixed latent width scaling as $O(N_{\text{DOF}}(ka)\log(1/\varepsilon))$ suffices because evanescent modes lie below $\sqrt{\varepsilon}$.
2. As $\varepsilon \to 0$ (or as queries approach $S_a$), evanescent modes cross above $\sqrt{\varepsilon}$ and force the required latent cache bit budget $b_Q(\varepsilon)$ to grow polylogarithmically as $(\log_2(1/\varepsilon))^d / (\log_2(R/a))^{d-1}$ ($d \in \{2, 3\}$), making any fixed-dimension $O(N_{\text{DOF}}\log(1/\varepsilon))$ latent memory architecture information-theoretically impossible at arbitrarily high precision.
3. Conversely, for linear scatterers/sources at fixed outer radius $R > a$, the complexity does **not** blow up exponentially in $1/\varepsilon$ (unlike general non-analytic inverse problems), remaining polylogarithmic—provided the hypotheses analyzed in Section 7 hold.

---

## 2. Canonical Abstract Theorem (Gaussian Reverse Water-Filling & Deterministic Kolmogorov $\sqrt{\varepsilon}$-Entropy)

Let $\{y_{\ell,m}\}_{\ell \ge \ell_0,\, 1 \le m \le m_\ell}$ be real modal coefficients indexed by order $\ell \ge \ell_0 \in \{0, 1\}$ with multiplicity $m_\ell \in \mathbb{N}_+$ satisfying either:
- **2D scaling class:** $m_\ell = m$ (constant $m \in \mathbb{N}_+$ for $\ell \ge 1$; any finite $m_0 \ge 0$ at $\ell = 0$), or
- **3D scaling class:** $m_\ell = m_0(2\ell + 1)$ ($m_0 \in \mathbb{N}_+$ for $\ell \ge \ell_0$).

Assume there exist $\ell_1 \ge \max(\ell_0, 1)$, constants $0 < c_1 \le c_2$, geometric base $\rho = a/R \in (0, 1)$, and arbitrary polynomial exponents $\beta_1, \beta_2 \in \mathbb{R}$ (positive, zero, or negative) such that the modal singular values $\sigma_\ell > 0$ satisfy:
$$c_1 \ell^{-\beta_1} \rho^\ell \le \sigma_\ell \le c_2 \ell^{-\beta_2} \rho^\ell \qquad \forall\, \ell \ge \ell_1, \tag{2.1}$$
and $\sigma_\ell > 0$ is finite for $\ell_0 \le \ell < \ell_1$. Write $\lambda_\ell := \sigma_\ell^2$.

### 2.1 Finite-Mode Reverse Water-Filling Optimality (Lagrangian Lower Bound)

**Theorem 2.1 (Exact Finite-Dimensional Reverse Water-Filling Optimality).** *Let $U$ be any finite index set, let $\lambda_i > 0$ for all $i \in U$, and let $\theta > 0$ be any water-filling threshold. For any distortion vector $(D_i)_{i \in U}$ satisfying $0 < D_i \le \lambda_i$ for all $i \in U$ and $\sum_{i \in U} D_i \le \sum_{i \in U} \min(\lambda_i, \theta)$, we have:*
$$\frac{1}{2}\sum_{i \in U} \log_2\!\left(\frac{\lambda_i}{D_i}\right) \ge \frac{1}{2}\sum_{i \in U} \log_2\!\left(\frac{\lambda_i}{\min(\lambda_i, \theta)}\right) + \frac{1}{2\theta \ln 2}\left(\sum_{i \in U} \min(\lambda_i, \theta) - \sum_{i \in U} D_i\right) \ge \frac{1}{2}\sum_{i \in U} \log_2\!\left(\frac{\lambda_i}{\min(\lambda_i, \theta)}\right). \tag{2.2}$$
*Moreover, equality holds at $D_i^* = \min(\lambda_i, \theta)$, which satisfies $0 < D_i^* \le \lambda_i$ (constructive non-vacuity).*

*Proof.* Set $D_i^* := \min(\lambda_i, \theta) > 0$. Using the elementary concavity inequality $\ln x \le x - 1$ for all $x > 0$:
- **Case 1 ($\lambda_i > \theta$):** Here $D_i^* = \theta$. Applying $\ln(D_i/\theta) \le D_i/\theta - 1$ gives
  $$\ln\!\left(\frac{\lambda_i}{D_i}\right) - \ln\!\left(\frac{\lambda_i}{D_i^*}\right) = -\ln\!\left(\frac{D_i}{\theta}\right) \ge 1 - \frac{D_i}{\theta} = \frac{D_i^* - D_i}{\theta}.$$
- **Case 2 ($\lambda_i \le \theta$):** Here $D_i^* = \lambda_i$, so $\ln(\lambda_i / D_i^*) = 0$. Since $D_i \le \lambda_i \le \theta$, we have $\lambda_i - D_i \ge 0$ and $\lambda_i^{-1} \ge \theta^{-1}$. Applying $\ln(D_i/\lambda_i) \le D_i/\lambda_i - 1$ yields
  $$\ln\!\left(\frac{\lambda_i}{D_i}\right) - \ln\!\left(\frac{\lambda_i}{D_i^*}\right) = -\ln\!\left(\frac{D_i}{\lambda_i}\right) \ge \frac{\lambda_i - D_i}{\lambda_i} \ge \frac{\lambda_i - D_i}{\theta} = \frac{D_i^* - D_i}{\theta}.$$

Summing both cases over $i \in U$ and dividing by $2\ln 2 > 0$ yields (2.2) identically. $\square$

### 2.2 Single-Shot Gaussian Rate-Distortion Bounds

Suppose the modes $y_{\ell,m} \sim \mathcal{N}(0, \lambda_\ell)$ are mutually independent.

**Theorem 2.2 (Single-Shot Gaussian Upper and Lower Bounds).** *Let $\varepsilon > 0$ be small enough that at least one mode satisfies $\lambda_{\ell} \ge \varepsilon$.*
1. **(Lower Bound):** *Any fixed-length $b$-bit cache achieving $\mathrm{MSE} \le \varepsilon$ satisfies, for every threshold $\theta > 0$ and finite $U$ with $\sum_{i \in U}\min(\lambda_i, \theta) \ge \varepsilon$ (in particular for $\theta = \varepsilon$ whenever $\max_{i \in U} \lambda_i \ge \varepsilon$, or for the exact water-filling threshold $\theta^*$ solving $\sum_i \min(\lambda_i, \theta^*) = \varepsilon$):*
   $$b \ge \frac{1}{2}\sum_{i \in U} \log_2\!\left(\frac{\lambda_i}{\min(\lambda_i, \theta)}\right). \tag{2.3}$$
2. **(Single-Shot Achievability Upper Bound):** *Let $L_+ \ge \ell_1$ be any cutoff such that $\sum_{\ell > L_+} m_\ell \lambda_\ell \le \varepsilon/4$, let $N_{\text{act}} := \sum_{\ell=\ell_0}^{L_+} m_\ell$, set $\theta_q := \frac{\varepsilon}{4 N_{\text{act}}}$, and let $\mathcal{A} := \{(\ell, m) : \ell \le L_+,\, \lambda_\ell > \theta_q\}$. Then there exists a single-shot fixed-length $b$-bit scalar quantizer achieving $\mathrm{MSE} \le \varepsilon$ with:*
   $$b \le 1 + \sum_{i \in \mathcal{A}} \left(\frac{1}{2}\log_2\!\left(\frac{\lambda_i}{\theta_q}\right) + \log_2(2M)\right), \qquad M := \max\!\left(2,\, \sqrt{4\ln\!\left(\frac{4\sqrt{6}\,N_{\text{act}}\,\lambda_{\max}}{\varepsilon}\right)}\right), \tag{2.4}$$
   *where $\lambda_{\max} := \max_{\ell \le L_+} \lambda_\ell$. In particular, since $\log_2(1/\theta_q) = \log_2(1/\varepsilon) + \log_2(4N_{\text{act}})$ and $\log_2(2M) = O(\log\log(1/\varepsilon) + \log N_{\text{act}})$, the overhead over $\frac{1}{2}\sum_{i \in \mathcal{A}}\log_2(\lambda_i/\varepsilon)$ is at most $O(N_{\text{act}}(\log N_{\text{act}} + \log\log(1/\varepsilon)))$.*

*Proof.*
**(1) Lower Bound.** Let $W = E(Y) \in \{0,1\}^b$ and $\hat{Y} = \tilde{D}(W)$. Fix any finite $U$ with $\sum_{i \in U} \min(\lambda_i, \theta) \ge \varepsilon$. Let $D_i := \mathbb{E}[(y_i - \hat{y}_i)^2]$; we must have $\sum_{i \in U} D_i \le \varepsilon$ and $D_i > 0$ for all $i \in U$ (otherwise $I(y_i; \hat{y}_i) = +\infty$, impossible for finite $b$). Because $W$ takes at most $2^b$ discrete values,
$$b \ge H(W) \ge I(Y_U; W) \ge I(Y_U; \hat{Y}_U) = \sum_{i \in U} \left(h(y_i) - h(y_i \mid Y_{U,<i}, \hat{Y}_U)\right) \ge \sum_{i \in U} \left(h(y_i) - h(y_i \mid \hat{y}_i)\right),$$
using independence of $\{y_i\}$ ($h(y_i \mid Y_{U,<i}) = h(y_i)$) and conditioning reduces entropy. By the maximum-entropy property of the Gaussian distribution for fixed second moment $\mathbb{E}[(y_i - \hat{y}_i)^2 \mid \hat{y}_i]$, Jensen's inequality gives $h(y_i \mid \hat{y}_i) = h(y_i - \hat{y}_i \mid \hat{y}_i) \le \frac{1}{2}\log_2(2\pi e D_i)$, while $h(y_i) = \frac{1}{2}\log_2(2\pi e \lambda_i)$. Setting $\tilde{D}_i := \min(D_i, \lambda_i) \in (0, \lambda_i]$ satisfies $\sum_{i \in U} \tilde{D}_i \le \sum_{i \in U} D_i \le \varepsilon \le \sum_{i \in U}\min(\lambda_i, \theta)$ and $\frac{1}{2}\log_2(\lambda_i/\tilde{D}_i) = \frac{1}{2}\log_2^+(\lambda_i/D_i) \le I(y_i; \hat{y}_i)$. Applying Theorem 2.1 to $(\tilde{D}_i)_{i \in U}$ gives (2.3). (Notice that for $\theta = \varepsilon$, if $\exists i_0 \in U$ with $\lambda_{i_0} \ge \varepsilon$, then $\sum_{i \in U}\min(\lambda_i, \varepsilon) \ge \min(\lambda_{i_0}, \varepsilon) = \varepsilon$ holds automatically.)

**(2) Achievability Upper Bound.** For all $i \notin \mathcal{A}$, set $\hat{y}_i = 0$. The uncoded error is
$$\sum_{\ell > L_+} m_\ell \lambda_\ell + \sum_{i \in \{\ell \le L_+\}\setminus \mathcal{A}} \lambda_i \le \frac{\varepsilon}{4} + N_{\text{act}}\,\theta_q = \frac{\varepsilon}{4} + \frac{\varepsilon}{4} = \frac{\varepsilon}{2}.$$
For each active mode $i \in \mathcal{A}$ (which satisfies $\lambda_i > \theta_q$), truncate to $[-A_i, A_i]$ with $A_i := M\sqrt{\lambda_i}$, and partition $[-A_i, A_i]$ into $K_i := \lceil 2A_i / \Delta \rceil = \lceil M\sqrt{\lambda_i/\theta_q}\rceil$ uniform sub-intervals of width $\Delta_i' = 2A_i / K_i \le \Delta := 2\sqrt{\theta_q}$, reconstructing each bin to its midpoint. Since $M \ge 2$ and $\lambda_i > \theta_q$, we have $M\sqrt{\lambda_i/\theta_q} > 2$, so $K_i \le 2M\sqrt{\lambda_i/\theta_q}$, and the outermost bin midpoint $A_i - \Delta_i'/2 \ge A_i - \sqrt{\theta_q} > A_i/2 > 0$ has the same sign as $y_i$ and smaller magnitude than $|y_i|$ whenever $|y_i| > A_i$. Hence $|y_i - \hat{y}_i|^2 \le (\Delta_i'/2)^2 \le \theta_q$ on $\{|y_i| \le A_i\}$, and $|y_i - \hat{y}_i|^2 \le y_i^2$ on $\{|y_i| > A_i\}$. For $Z = y_i/\sqrt{\lambda_i} \sim \mathcal{N}(0, 1)$, Cauchy–Schwarz and the Gaussian Chernoff bound $\mathbb{P}(|Z| > M) \le 2e^{-M^2/2}$ give:
$$\mathbb{E}[y_i^2 \mathbf{1}_{|y_i| > A_i}] = \lambda_i \mathbb{E}[Z^2 \mathbf{1}_{|Z|>M}] \le \lambda_i (\mathbb{E}[Z^4])^{1/2} \mathbb{P}(|Z|>M)^{1/2} \le \lambda_i \sqrt{3}\sqrt{2e^{-M^2/2}} = \sqrt{6}\,\lambda_i e^{-M^2/4} \le \frac{\varepsilon}{4N_{\text{act}}}$$
by choice of $M$. Summing over $i \in \mathcal{A}$ yields quantization + clipping error at most $|\mathcal{A}|(\theta_q + \frac{\varepsilon}{4N_{\text{act}}}) \le \frac{\varepsilon}{4} + \frac{\varepsilon}{4} = \frac{\varepsilon}{2}$, so total $\mathrm{MSE} \le \varepsilon$. Encoding the joint index in $\prod_{i \in \mathcal{A}} \{0, \dots, K_i - 1\}$ requires $b = \lceil \sum_{i \in \mathcal{A}} \log_2 K_i \rceil$ bits, yielding (2.4). $\square$

### 2.3 Deterministic Analytic Ellipsoids (Kolmogorov $\sqrt{\varepsilon}$-Entropy)

Consider the deterministic analytic ellipsoid of energy $E_0 > 0$:
$$\mathcal{E} := \left\{ y = (y_{\ell,m}) : \sum_{\ell \ge \ell_0}\sum_{m=1}^{m_\ell} \frac{y_{\ell,m}^2}{\sigma_\ell^2} \le E_0^2 \right\}.$$

**Theorem 2.3 (Kolmogorov $\sqrt{\varepsilon}$-Entropy Bounds on $\mathcal{E}$).** *Let $\delta := \sqrt{\varepsilon} > 0$.*
1. **(Volume Packing Lower Bound):** *For any finite subset $I$ of modes,*
   $$H_{\sqrt{\varepsilon}}(\mathcal{E}) \ge \sum_{i \in I} \frac{1}{2}\log_2^+\!\left(\frac{E_0^2 \sigma_i^2}{\varepsilon}\right). \tag{2.5}$$
2. **(Hypercube-Lattice Covering Upper Bound):** *Let $L_+ \ge \ell_1$ be any cutoff such that $\sup_{\ell > L_+} E_0^2 \sigma_\ell^2 \le \varepsilon/2$, and let $K := N_{\text{act}} = \sum_{\ell=\ell_0}^{L_+} m_\ell$. Then:*
   $$H_{\sqrt{\varepsilon}}(\mathcal{E}) \le \sum_{i=1}^K \frac{1}{2}\log_2^+\!\left(\frac{E_0^2 \sigma_i^2}{\varepsilon}\right) + \frac{K}{2}\log_2(12\pi e) + 1. \tag{2.6}$$

*Proof.*
**(1) Lower Bound.** Let $I_+ := \{i \in I : E_0 \sigma_i > \sqrt{\varepsilon}\}$ with $k := |I_+|$. If $k = 0$ the bound is $0$. Otherwise, let $\mathcal{E}_+ := \{y \in \mathbb{R}^{I_+} : \sum_{i \in I_+} y_i^2 / (E_0 \sigma_i)^2 \le 1\} \subset \mathcal{E}$ (padding remaining coordinates by $0$). Any external $\delta$-covering $\mathcal{C} \subset \ell^2$ of $\mathcal{E}$ projects orthogonally onto $\mathbb{R}^{I_+}$ (a $1$-Lipschitz map) to a $\delta$-covering $\mathcal{C}_+$ of $\mathcal{E}_+$ in $\mathbb{R}^k$. Comparing Euclidean $k$-volumes:
$$|\mathcal{C}| \cdot \mathrm{Vol}_k(\delta B_2^k) \ge |\mathcal{C}_+| \cdot \mathrm{Vol}_k(\delta B_2^k) \ge \mathrm{Vol}_k(\mathcal{E}_+) = \mathrm{Vol}_k(B_2^k) \prod_{i \in I_+} (E_0 \sigma_i).$$
Canceling $\mathrm{Vol}_k(B_2^k) > 0$ and taking $\log_2$ gives $H_{\sqrt{\varepsilon}}(\mathcal{E}) \ge \sum_{i \in I_+} \log_2(E_0 \sigma_i / \sqrt{\varepsilon}) = \sum_{i \in I} \frac{1}{2}\log_2^+(E_0^2\sigma_i^2/\varepsilon)$.

**(2) Upper Bound.** For any $y \in \mathcal{E}$, truncating modes $\ell > L_+$ incurs tail squared error
$$\sum_{\ell > L_+, m} y_{\ell,m}^2 \le \left(\sup_{\ell > L_+} \sigma_\ell^2\right) \sum_{\ell > L_+, m} \frac{y_{\ell,m}^2}{\sigma_\ell^2} \le \left(\sup_{\ell > L_+} E_0^2 \sigma_\ell^2\right) \le \frac{\varepsilon}{2}.$$
Tile the $K$-dimensional active subspace $\mathbb{R}^K$ with cubes $z + [-s/2, s/2]^K$ ($z \in s\mathbb{Z}^K$) of side $s := \sqrt{\frac{2\varepsilon}{K}}$. For any $y_K \in \mathbb{R}^K$, its nearest lattice point $z \in s\mathbb{Z}^K$ satisfies $\|y_K - z\|_2^2 \le K(s/2)^2 = \varepsilon/2$, giving total squared error $\le \varepsilon/2 + \varepsilon/2 = \varepsilon$.

Let $N$ be the number of lattice centers $z \in s\mathbb{Z}^K$ whose cube $Q_z := z + [-s/2, s/2]^K$ intersects $\mathcal{E}_K := \{y_K \in \mathbb{R}^K : \sum_{i=1}^K y_i^2/(E_0\sigma_i)^2 \le 1\}$. For any $x \in Q_z$ and witness $y \in Q_z \cap \mathcal{E}_K$, we have $\|x - y\|_\infty \le s$, hence $\|x - y\|_2 \le s\sqrt{K} = \sqrt{2\varepsilon}$. Defining weights $w_i := \sqrt{E_0^2\sigma_i^2 + 2\varepsilon}$, the triangle inequality gives
$$\left(\sum_{i=1}^K \frac{x_i^2}{w_i^2}\right)^{1/2} \le \left(\sum_{i=1}^K \frac{y_i^2}{w_i^2}\right)^{1/2} + \left(\sum_{i=1}^K \frac{(x_i - y_i)^2}{w_i^2}\right)^{1/2} \le \left(\sum_{i=1}^K \frac{y_i^2}{E_0^2\sigma_i^2}\right)^{1/2} + \frac{\|x - y\|_2}{\sqrt{2\varepsilon}} \le 1 + 1 = 2.$$
Hence all $N$ interior-disjoint cubes $Q_z$ (each of volume $s^K$) lie inside the $2$-dilated ellipsoid $\mathcal{E}_K^{(w)}(2) := \{x : \sum_{i=1}^K x_i^2/w_i^2 \le 4\}$. Using the elementary Gaussian-ball volume bound (valid for all $K \ge 1$ without Stirling):
$$(2\pi)^{K/2} = \int_{\mathbb{R}^K} e^{-\|x\|_2^2/2}\,dx \ge \int_{\sqrt{K} B_2^K} e^{-K/2}\,dx = \left(\frac{K}{e}\right)^{K/2}\mathrm{Vol}_K(B_2^K) \implies \mathrm{Vol}_K(B_2^K) \le \left(\frac{2\pi e}{K}\right)^{K/2},$$
we obtain
$$N \cdot s^K \le 2^K \mathrm{Vol}_K(B_2^K) \prod_{i=1}^K w_i \implies N \le \left(\frac{K}{2\varepsilon}\right)^{K/2} 2^K \left(\frac{2\pi e}{K}\right)^{K/2} \prod_{i=1}^K \sqrt{E_0^2\sigma_i^2 + 2\varepsilon} = (4\pi e)^{K/2}\prod_{i=1}^K \sqrt{\frac{E_0^2\sigma_i^2}{\varepsilon} + 2}.$$
Finally, using $x + 2 \le 3\max(x, 1)$ for all $x \ge 0$, we have $\frac{1}{2}\log_2(x+2) \le \frac{1}{2}\log_2^+(x) + \frac{1}{2}\log_2 3$. Taking $\lceil \log_2 N \rceil \le \log_2 N + 1$ gives (2.6). $\square$

### 2.4 Polynomial Prefactor Invariance Theorem

**Theorem 2.4 (Invariance to Polynomial Prefactors).** *Under the two-sided envelope (2.1) with arbitrary $\beta_1, \beta_2 \in \mathbb{R}$ and $0 < c_1 \le c_2$, let $L_0(\varepsilon) := \frac{\log_2(1/\varepsilon)}{2\log_2(R/a)}$. Then both the single-shot Gaussian bit-rate $b_Q(\varepsilon)$ and the deterministic ellipsoid Kolmogorov entropy $\lceil H_{\sqrt{\varepsilon}}(\mathcal{E})\rceil$ satisfy, as $\varepsilon \to 0^+$:*
- **2D scaling class ($m_\ell = m$ for $\ell \ge 1$):**
  $$b_Q(\varepsilon) = \frac{m}{8\log_2(R/a)}\left(\log_2 \frac{1}{\varepsilon}\right)^2 + O\!\left(\log\frac{1}{\varepsilon}\cdot\log\log\frac{1}{\varepsilon}\right). \tag{2.7}$$
- **3D scaling class ($m_\ell = m_0(2\ell+1)$):**
  $$b_Q(\varepsilon) = \frac{m_0}{24(\log_2(R/a))^2}\left(\log_2 \frac{1}{\varepsilon}\right)^3 + O\!\left(\left(\log\frac{1}{\varepsilon}\right)^2 \log\log\frac{1}{\varepsilon}\right). \tag{2.8}$$
*In particular, the leading polylogarithmic coefficients $\frac{m}{8\log_2(R/a)}$ (2D) and $\frac{m_0}{24(\log_2(R/a))^2}$ (3D) are invariant to $c_1, c_2, \beta_1, \beta_2, E_0$, and the finite set of low orders $\ell < \ell_1$.*

*Proof.* Set $\tilde{c}_1 := c_1$ in the Gaussian case ($E_0 := 1$) or $\tilde{c}_1 := E_0 c_1$ in the ellipsoid case, and $\tilde{c}_2 := c_2$ (resp. $E_0 c_2$).
1. **Cutoff $L_-$ for Lower Bound:** Let $L_- \ge \ell_1$ be the largest integer with $\tilde{c}_1^2 L_-^{-2\beta_1} \rho^{2 L_-} \ge \varepsilon$ (for $\varepsilon$ small enough that $L_- \ge \ell_1$, noting that if $\beta_1 < 0$ then $\ell^{-2\beta_1}\rho^{2\ell}$ is strictly decreasing for $\ell > |\beta_1|/\ln(1/\rho)$). By definition of $L_-$, $\tilde{c}_1^2 (L_- + 1)^{-2\beta_1} \rho^{2(L_- + 1)} < \varepsilon$, so taking $\log_2$ gives
   $$2 L_- \log_2(R/a) = \log_2(1/\varepsilon) - 2\beta_1 \log_2 L_- + O(1) \implies L_- = L_0(\varepsilon) - O(\log L_0(\varepsilon)).$$
   For every $\ell_1 \le \ell \le L_-$, we have $\frac{\lambda_\ell}{\varepsilon} \ge \left(\frac{\ell}{L_- + 1}\right)^{-2\beta_1} \rho^{2(\ell - L_- - 1)}$, hence
   $$\frac{1}{2}\log_2\!\left(\frac{\lambda_\ell}{\varepsilon}\right) \ge (L_- + 1 - \ell)\log_2(R/a) - |\beta_1|\log_2(L_- + 1).$$
   By Theorems 2.2(1) (at $\theta = \varepsilon$) and 2.3(1), summing over $\ell_1 \le \ell \le L_-$ yields
   $$b_Q(\varepsilon) \ge \log_2(R/a) \sum_{\ell=\ell_1}^{L_-} m_\ell (L_- - \ell) - O\!\left(\log L_0 \sum_{\ell=\ell_1}^{L_-} m_\ell\right).$$
2. **Cutoff $L_+$ for Upper Bound:** For $\ell > L_+ \ge \max(\ell_1, |\beta_2|/\ln(1/\rho))$, we have $m_\ell \lambda_\ell \le m_\ell \tilde{c}_2^2 \ell^{-2\beta_2} \rho^{2\ell}$, which decays geometrically with ratio $\le \frac{1+\rho}{2} < 1$ once $L_+$ is sufficiently large. Thus $\sum_{\ell > L_+} m_\ell \lambda_\ell \le C' L_+^{d-1 + 2\max(0, -\beta_2)} \rho^{2L_+}$. Choosing $L_+$ as the smallest integer making this tail $\le \varepsilon/4$ (and $\sup_{\ell > L_+} \tilde{c}_2^2 \ell^{-2\beta_2}\rho^{2\ell} \le \varepsilon/2$) gives
   $$2 L_+ \log_2(R/a) = \log_2(1/\varepsilon) + O(\log L_0(\varepsilon)) \implies L_+ = L_0(\varepsilon) + O(\log L_0(\varepsilon)).$$
   For $\ell_1 \le \ell \le L_+$, $\frac{1}{2}\log_2^+(\lambda_\ell/\varepsilon) \le (L_+ - \ell)\log_2(R/a) + O(\log L_0)$. Adding the subleading overhead $O(N_{\text{act}}\log L_0)$ (Gaussian Theorem 2.2(2)) or $O(N_{\text{act}})$ (ellipsoid Theorem 2.3(2)) and the $O(\ell_1 \log(1/\varepsilon))$ contribution from finitely many orders $\ell < \ell_1$, we get:
   $$b_Q(\varepsilon) \le \log_2(R/a) \sum_{\ell=\ell_1}^{L_+} m_\ell (L_+ - \ell) + O\!\left(N_{\text{act}}(L_+)\log L_0\right).$$
3. **Evaluation via Section 3:** Since $L_\pm = L_0 + O(\log L_0)$ and $N_{\text{act}}(L_\pm) = \Theta(L_0^{d-1})$, applying the exact sums of Proposition 3.1–3.2 gives (2.7) and (2.8). $\square$

---

## 3. Exact Bit-Allocation Sums & 2D/3D Constant Reconciliation

### 3.1 Exact Closed-Form Identities

**Proposition 3.1 (3D Spherical Harmonic Bit-Allocation Sum).** *For all integers $L \ge 0$,*
$$\sum_{\ell=0}^L (2\ell + 1)(L - \ell) = \frac{L(L + 1)(2L + 1)}{6}, \tag{3.1}$$
*and consequently,*
$$\frac{L^3}{3} \le \sum_{\ell=0}^L (2\ell + 1)(L - \ell) \le \frac{(L + 1)^3}{3}. \tag{3.2}$$

*Proof.* Expanding $(2\ell+1)(L-\ell) = (2\ell+1)L - 2\ell^2 - \ell$, or substituting $k = L - \ell$ and inducting on $L$: for $L = 0$, $0 = 0$; advancing $L \to L+1$ adds $\sum_{\ell=0}^L (2\ell+1) = (L+1)^2$, and
$$\frac{L(L+1)(2L+1)}{6} + (L+1)^2 = \frac{(L+1)(2L^2 + 7L + 6)}{6} = \frac{(L+1)(L+2)(2L+3)}{6}.$$
Furthermore, $2L^3 \le L(L+1)(2L+1) = 2L^3 + 3L^2 + L \le 2L^3 + 6L^2 + 6L + 2 = 2(L+1)^3$, proving (3.2) upon dividing by $6$. *(Remark: $\frac{L(L+1)(2L+1)}{6}$ agrees with direct evaluation $0, 1, 5, 14, 30$ at $L = 0, 1, 2, 3, 4$, whereas $\frac{L(L+1)(L+2)}{3}$ would give $2$ at $L=1$ and $8$ at $L=2$.)* $\square$

Substituting $L_0 = \frac{\log_2(1/\varepsilon)}{2\log_2(R/a)}$ into $\log_2(R/a) \cdot \frac{L_0^3}{3}$ (for $m_0$ real copies of $2\ell+1$ modes/order) gives:
$$m_0 \log_2(R/a) \cdot \frac{1}{3}\left(\frac{\log_2(1/\varepsilon)}{2\log_2(R/a)}\right)^3 = \frac{m_0}{24(\log_2(R/a))^2}\left(\log_2 \frac{1}{\varepsilon}\right)^3,$$
matching Conjecture 1's 3D constant $24 = 3 \cdot 2^3$ identically for $m_0 = 1$.

**Proposition 3.2 (2D Multipole Bit-Allocation Sums).** *For all $L \ge 1$ and $m \in \mathbb{N}_+$:*
$$\sum_{\ell=1}^L m(L - \ell) = m\,\frac{L(L - 1)}{2}, \qquad \sum_{\ell=0}^L (L - \ell) = \frac{L(L+1)}{2}, \qquad L + \sum_{\ell=1}^L 2(L - \ell) = L^2. \tag{3.3}$$
*Proof.* Since $\sum_{\ell=1}^L (L - \ell) = \sum_{j=0}^{L-1} j = \frac{L(L-1)}{2}$, multiplying by $m$ gives the first identity; adding $L$ at $\ell = 0$ gives the remaining two. $\square$

### 3.2 Transparent Reconciliation of the 2D Factor-of-2 ($8\log_2(R/a)$ vs $4\log_2(R/a)$)

Substituting $L_0 = \frac{\log_2(1/\varepsilon)}{2\log_2(R/a)}$ into $m\frac{L_0^2}{2}\log_2(R/a)$ yields:
$$m\frac{\log_2(R/a)}{2} \left(\frac{\log_2(1/\varepsilon)}{2\log_2(R/a)}\right)^2 = \frac{m}{8\log_2(R/a)}\left(\log_2\frac{1}{\varepsilon}\right)^2.$$
This reconciles Conjecture 1's 2D constant transparently:
- **$m = 1$ real mode/order:** gives $\frac{(\log_2(1/\varepsilon))^2}{8\log_2(R/a)}$, matching Conjecture 1's literal formula (realized physically by an even-parity / cosine-only sector $a_\ell \cos\ell\theta$, or a single real quadrature).
- **$m = 2$ real modes/order ($\cos\ell\theta, \sin\ell\theta$, or $1$ complex mode/order):** gives $\frac{(\log_2(1/\varepsilon))^2}{4\log_2(R/a)}$. The factor-of-2 difference between $8$ and $4$ arises because each real Gaussian mode has rate $\frac{1}{2}\log_2(\lambda_\ell/\theta) = (L-\ell)\log_2(R/a)$, so two independent real modes per order double the budget relative to $m=1$.
- **$m = 4$ real modes/order (complex 2D phasor source, $e^{\pm i\ell\theta}$):** gives $\frac{(\log_2(1/\varepsilon))^2}{2\log_2(R/a)}$.

### 3.3 Complete Summary Table Across PDEs and Mode Conventions

| Physical Setting | Order $\ell$ Range | Real DOFs / Order $m_\ell$ | Degree $d$ | Propagating $N_{\text{DOF}}^{\mathbb{R}}(ka)$ | Exact Leading Evanescent Coefficient |
|---|---|---|---|---|---|
| 2D 1-mode / even-parity sector | $\ell \ge 1$ | $1$ | $2$ | $ka$ | $\dfrac{(\log_2(1/\varepsilon))^2}{8\log_2(R/a)}$ |
| **2D Real Scalar Laplace / Helmholtz** | $\ell \ge 1$ | $2$ | $2$ | $2ka$ | $\dfrac{(\log_2(1/\varepsilon))^2}{4\log_2(R/a)}$ |
| 2D Complex Scalar Helmholtz Phasor | $\ell \ge 1$ | $4$ | $2$ | $4ka$ | $\dfrac{(\log_2(1/\varepsilon))^2}{2\log_2(R/a)}$ |
| **3D Real Scalar Laplace / Helmholtz** | $\ell \ge 0$ | $2\ell+1$ | $3$ | $(ka)^2$ | $\dfrac{(\log_2(1/\varepsilon))^3}{24(\log_2(R/a))^2}$ |
| **3D Complex Scalar / Real Maxwell (TE+TM)** | $\ell \ge 1$ (or $\ge 0$) | $2(2\ell+1)$ | $3$ | $2(ka)^2$ | $\dfrac{(\log_2(1/\varepsilon))^3}{12(\log_2(R/a))^2}$ |
| 3D Complex Vector Maxwell Phasor (TE+TM) | $\ell \ge 1$ | $4(2\ell+1)$ | $3$ | $4(ka)^2$ | $\dfrac{(\log_2(1/\varepsilon))^3}{6(\log_2(R/a))^2}$ |

---

## 4. Specialized PDE Derivations (Laplace, Scalar Helmholtz, Vector Maxwell)

We now verify the envelope (2.1) rigorously for each physical PDE operator.

### 4.1 Laplace Potential Fields ($k = 0$)

1. **2D Laplace ($d = 2$):**
   - *Boundary-to-boundary Dirichlet operator $T : L^2(S_a) \to L^2(S_R)$:* Writing $u(r,\theta) = c_0 + \sum_{\ell=1}^\infty (a/r)^\ell (c_\ell\cos\ell\theta + d_\ell\sin\ell\theta)$ in the $L^2(S_r)$-orthonormal basis $\left\{\frac{1}{\sqrt{2\pi r}}, \frac{\cos\ell\theta}{\sqrt{\pi r}}, \frac{\sin\ell\theta}{\sqrt{\pi r}}\right\}$, we obtain exact singular values $\sigma_0 = \sqrt{R/a}$ (multiplicity $1$) and $\sigma_\ell = \sqrt{R/a}\,\rho^\ell$ (multiplicity $m = 2$, $\ell \ge 1$; or exact $\sigma_\ell = \rho^\ell$ under normalized surface average).
   - *Volumetric logarithmic potential $(V_2 f)(x) = \frac{1}{2\pi}\int_{B_a} \ln\frac{1}{|x-y|} f(y)\,dy$, $V_2 : L^2(B_a) \to L^2(S_R)$:* For $|x| = R > |y| = r'$,
     $$\frac{1}{2\pi}\ln\frac{1}{|x-y|} = \frac{1}{2\pi}\ln\frac{1}{R} + \frac{1}{2\pi}\sum_{\ell=1}^\infty \frac{1}{\ell}\left(\frac{r'}{R}\right)^\ell \cos(\ell(\theta_x - \theta_y)).$$
     Expanding $f(r', \theta')$ in radial-angular modes, $\int_0^{2\pi} \cos(\ell(\theta_x - \theta'))\frac{\cos\ell\theta'}{\sqrt{\pi}}\,d\theta' = \sqrt{\pi}\cos\ell\theta_x$ and $\|r'^\ell\|_{L^2([0,a], r'\,dr')} = \frac{a^{\ell+1}}{\sqrt{2\ell+2}}$, yielding exact singular values ($\ell \ge 1$, multiplicity $2$):
     $$\sigma_\ell(V_2) = \frac{a\sqrt{R}}{2\ell\sqrt{2\ell+2}}\left(\frac{a}{R}\right)^\ell = \Theta(\ell^{-3/2}\rho^\ell) \qquad (\beta = 3/2). \tag{4.1}$$

2. **3D Laplace ($d = 3$):**
   - *Boundary-to-boundary operator $T : L^2(S_a) \to L^2(S_R)$:* $u(r,\hat{x}) = \sum_{\ell=0}^\infty \sum_{m=-\ell}^\ell u_{\ell m}(a/r)^{\ell+1} Y_\ell^m(\hat{x})$ gives $\sigma_\ell = (R/a)\rho^{\ell+1} = \rho^\ell$ (multiplicity $2\ell+1$).
   - *Volumetric Newton potential $(V_3 f)(x) = \frac{1}{4\pi}\int_{B_a}\frac{f(y)}{|x-y|}\,dy$, $V_3 : L^2(B_a) \to L^2(S_R)$:* Using the multipole expansion $\frac{1}{4\pi|x-y|} = \sum_{\ell=0}^\infty \sum_{m=-\ell}^\ell \frac{1}{2\ell+1}\frac{r'^\ell}{R^{\ell+1}} Y_\ell^m(\hat{x})\overline{Y_\ell^m(\hat{y})}$ and $\|r'^\ell\|_{L^2([0,a], r'^2 dr')} = \frac{a^{\ell+3/2}}{\sqrt{2\ell+3}}$, the exact singular values ($\ell \ge 0$, multiplicity $2\ell+1$) are:
     $$\sigma_\ell(V_3) = \frac{a^{3/2}}{(2\ell+1)\sqrt{2\ell+3}}\left(\frac{a}{R}\right)^\ell = \Theta(\ell^{-3/2}\rho^\ell) \qquad (\beta = 3/2). \tag{4.2}$$

### 4.2 Scalar Helmholtz Fields ($k > 0$)

1. **2D Scalar Helmholtz ($(\Delta + k^2)u = -f$):**
   The radiating Green's function $G_2(x,y) = \frac{i}{4}H_0^{(1)}(k|x-y|)$ admits **Graf's addition theorem** for $|x| = R > |y| = r'$:
   $$G_2(x,y) = \frac{i}{4}H_0^{(1)}(kR)J_0(kr') + \frac{i}{2}\sum_{\ell=1}^\infty H_\ell^{(1)}(kR)J_\ell(kr')\cos(\ell(\theta_x - \theta_y)).$$
   For the radiation operator $S_2 : L^2(B_a) \to L^2(S_R)$, using Lommel's integral $\int_0^a J_\ell(kr)^2 r\,dr = \frac{a^2}{2}\!\left[J_\ell'(ka)^2 + \left(1 - \frac{\ell^2}{(ka)^2}\right)J_\ell(ka)^2\right]$, the singular value at order $\ell$ is:
   $$\sigma_\ell^{(2D)} = \frac{\pi\sqrt{R}}{2}|H_\ell^{(1)}(kR)|\left(\int_0^a J_\ell(kr)^2 r\,dr\right)^{1/2}. \tag{4.3}$$
   *(Real vs. complex source DOF note: For real-valued source $f \in L^2(B_a; \mathbb{R})$ mapping to either the complex phasor $u$ or physical real field $\mathrm{Re}(u)$, the two angular modes $\cos\ell\theta, \sin\ell\theta$ share the single complex phase $H_\ell^{(1)}(kR)$, giving $m = 2$ real DOFs per order $\ell \ge 1$. For a complex current phasor $f \in L^2(B_a; \mathbb{C})$, $\mathrm{Re}(f)$ and $\mathrm{Im}(f)$ decouple, giving $m = 4$ real DOFs per order.)*

   By Debye's uniform Bessel/Hankel asymptotics:
   - **Oscillatory / Propagating ($\ell \le (1-\eta)ka$):** $J_\ell(kr) \sim \sqrt{\frac{2}{\pi\sqrt{k^2r^2-\ell^2}}}\cos(\cdots)$ on $[\ell/k, a]$ and $|H_\ell^{(1)}(kR)| \sim \sqrt{\frac{2}{\pi\sqrt{(kR)^2-\ell^2}}}$. Averaging $\cos^2$ yields $\int_0^a J_\ell(kr)^2 r\,dr \sim \frac{\sqrt{(ka)^2-\ell^2}}{\pi k^2}$, so
     $$\sigma_\ell^{(2D)} \sim \frac{\sqrt{R}}{\sqrt{2}\,k}\left(\frac{(ka)^2-\ell^2}{(kR)^2-\ell^2}\right)^{1/4} = \Theta(k^{-1}a^{1/2}),$$
     a uniformly bounded propagating plateau of $N_{\text{DOF}}^{\mathbb{R}}(ka) = 2ka + O((ka)^{1/3})$ real modes (real source) or $4ka$ (complex source).
   - **Deep Evanescent ($\ell \gg kR$):** Debye's large-order asymptotics give
     $$J_\ell(kr) \sim \frac{1}{\sqrt{2\pi\ell}}\left(\frac{ekr}{2\ell}\right)^\ell, \qquad H_\ell^{(1)}(kR) \sim -i\sqrt{\frac{2}{\pi\ell}}\left(\frac{2\ell}{ekR}\right)^\ell.$$
     The factorial-type factors $(2\ell/e)^\ell$ and $k^\ell$ cancel identically:
     $$\sigma_\ell^{(2D)} \sim \frac{\pi\sqrt{R}}{2}\sqrt{\frac{2}{\pi\ell}}\left(\frac{2\ell}{ekR}\right)^\ell \frac{a}{\sqrt{2\pi\ell}\sqrt{2\ell+2}}\left(\frac{eka}{2\ell}\right)^\ell = \frac{a\sqrt{R}}{2\ell\sqrt{2\ell+2}}\left(\frac{a}{R}\right)^\ell = \Theta(\ell^{-3/2}\rho^\ell), \tag{4.4}$$
     recovering the exact 2D Laplace Newton-potential singular value (4.1) to leading constant.

2. **3D Scalar Helmholtz ($(\Delta + k^2)u = -f$):**
   The radiating Green's function $G_3(x,y) = \frac{e^{ik|x-y|}}{4\pi|x-y|} = \frac{ik}{4\pi}h_0^{(1)}(k|x-y|)$ expands via **Gegenbauer's addition theorem** ($|x|=R>|y|=r'$) as:
   $$G_3(x,y) = ik\sum_{\ell=0}^\infty \sum_{m=-\ell}^\ell h_\ell^{(1)}(kR)j_\ell(kr') Y_\ell^m(\hat{x})\overline{Y_\ell^m(\hat{y})}.$$
   The singular values of $S_3 : L^2(B_a) \to L^2(S_R)$ (multiplicity $2\ell+1$ over $\mathbb{R}$ for real $f$, or $2(2\ell+1)$ for complex $f$) are:
   $$\sigma_\ell^{(3D)} = kR |h_\ell^{(1)}(kR)|\left(\int_0^a j_\ell(kr)^2 r^2\,dr\right)^{1/2}. \tag{4.5}$$
   - **Propagating ($\ell \le (1-\eta)ka$):** Using $j_\ell(x) = \sqrt{\frac{\pi}{2x}}J_{\ell+1/2}(x)$ and $h_\ell^{(1)}(x) = \sqrt{\frac{\pi}{2x}}H_{\ell+1/2}^{(1)}(x)$, we have $kR|h_\ell^{(1)}(kR)| \sim (1 - (\ell/(kR))^2)^{-1/4}$ and the radial integral $\int_0^a j_\ell(kr)^2 r^2\,dr = \frac{\pi}{2k}\int_0^a J_{\ell+1/2}(kr)^2 r\,dr \sim \frac{\pi}{2k}\int_{\ell/k}^a \frac{r\,dr}{\pi\sqrt{k^2 r^2 - \ell^2}} = \frac{\sqrt{(ka)^2-\ell^2}}{2k^3}$, whose square root is $\frac{((ka)^2-\ell^2)^{1/4}}{\sqrt{2}\,k^{3/2}}$. Hence
     $$\sigma_\ell^{(3D)} \sim \frac{1}{\sqrt{2}\,k^{3/2}}\left(\frac{(ka)^2-\ell^2}{1-(\ell/(kR))^2}\right)^{1/4} = \Theta(k^{-1}a^{1/2}) \quad (\text{dimension } \text{Length}^{3/2}),$$
     giving $N_{\text{DOF}}^{\mathbb{R}}(ka) = \sum_{\ell=0}^{\lfloor ka\rfloor}(2\ell+1) = (ka)^2 + O((ka)^{4/3})$ real modes (or $2(ka)^2$ for complex $f$).
   - **Deep Evanescent ($\ell \gg kR$):** Using $j_\ell(kr) \sim \frac{(kr)^\ell}{(2\ell+1)!!}$ and $h_\ell^{(1)}(kR) \sim -i\frac{(2\ell-1)!!}{(kR)^{\ell+1}}$, the double factorials cancel to $(2\ell+1)^{-1}$:
     $$\sigma_\ell^{(3D)} \sim kR \cdot \frac{(2\ell-1)!!}{(kR)^{\ell+1}} \cdot \frac{k^\ell a^{\ell+3/2}}{(2\ell+1)!!\sqrt{2\ell+3}} = \frac{a^{3/2}}{(2\ell+1)\sqrt{2\ell+3}}\left(\frac{a}{R}\right)^\ell = \Theta(\ell^{-3/2}\rho^\ell), \tag{4.6}$$
     recovering the exact 3D Laplace Newton-potential singular value (4.2).

### 4.3 Vector Maxwell Fields ($k > 0$, $d = 3$)

For time-harmonic currents $\mathbf{J}$ supported in $B_a$, the scattered/radiated electric field in $\mathbb{R}^3 \setminus B_a$ is $\mathbf{E}(x) = i\omega\mu_0 \int_{B_a} \overline{\overline{\mathbf{G}}}_e(x, y)\cdot \mathbf{J}(y)\,dy$, where $\overline{\overline{\mathbf{G}}}_e(x,y) = (\overline{\overline{\mathbf{I}}} + k^{-2}\nabla_x\nabla_x)G_3(x,y)$. Define the $L^2(S_1; \mathbb{C}^3)$-orthonormal vector spherical harmonics for $\ell \ge 1$, $|m| \le \ell$ (note **no $\ell=0$ radiating monopole exists**):
$$\mathbf{X}_{\ell m}(\hat{x}) := \frac{\nabla_{S^2} Y_\ell^m(\hat{x}) \times \hat{x}}{\sqrt{\ell(\ell+1)}}, \qquad \hat{x}\times\mathbf{X}_{\ell m}(\hat{x}) = \frac{\nabla_{S^2}Y_\ell^m(\hat{x})}{\sqrt{\ell(\ell+1)}}, \qquad \mathbf{Y}_{\ell m}(\hat{x}) := Y_\ell^m(\hat{x})\hat{x},$$
and the Hansen vector spherical wavefunctions ($\alpha = 0$ regular $z_\ell^{(0)} = j_\ell$; $\alpha = 1$ outgoing $z_\ell^{(1)} = h_\ell^{(1)}$):
$$\mathbf{M}_{\ell m}^{(\alpha)}(kx) := z_\ell^{(\alpha)}(kr)\mathbf{X}_{\ell m}(\hat{x}) \;\;(\text{TE}), \qquad \mathbf{N}_{\ell m}^{(\alpha)}(kx) := \frac{\sqrt{\ell(\ell+1)}}{kr}z_\ell^{(\alpha)}(kr)\mathbf{Y}_{\ell m}(\hat{x}) + \frac{[kr\,z_\ell^{(\alpha)}(kr)]'}{kr}(\hat{x}\times\mathbf{X}_{\ell m}(\hat{x})) \;\;(\text{TM}).$$
For $|x| = R > |y|$, the dyadic Green's function decomposes orthogonally across $\ell \ge 1, |m| \le \ell$ and TE/TM polarizations:
$$\overline{\overline{\mathbf{G}}}_e(x,y) = ik \sum_{\ell=1}^\infty \sum_{m=-\ell}^\ell \left[ \mathbf{M}_{\ell m}^{(1)}(kx) \otimes \overline{\mathbf{M}_{\ell m}^{(0)}(ky)} + \mathbf{N}_{\ell m}^{(1)}(kx) \otimes \overline{\mathbf{N}_{\ell m}^{(0)}(ky)} \right]. \tag{4.7}$$
- **Multiplicity:** For $\ell \ge 1$, TE and TM contribute $2(2\ell+1)$ complex modes $= 4(2\ell+1)$ real DOFs for complex current phasors $\mathbf{J} \in L^2(B_a; \mathbb{C}^3)$ (or $2(2\ell+1)$ real DOFs for real current distributions). Since the missing $\ell=0$ mode only changes $N_{\text{act}}$ by $-1$ (or $-2$), Proposition 3.1 applies up to an $O(L)$ constant shift, preserving the leading $(\log_2(1/\varepsilon))^3$ coefficient.
- **Evanescent spectrum ($\ell \gg kR$, fixed $k > 0$):**
  - **TE ($\mathbf{M}$) branch:** Identical radial norms to 3D scalar Helmholtz, giving $\sigma_\ell^{\text{TE}} \asymp \ell^{-3/2} \rho^\ell$ ($\beta = 3/2$).
  - **TM ($\mathbf{N}$) branch:** For $\ell \gg x$, $[x h_\ell^{(1)}(x)]'/x \sim -\frac{\ell}{x}h_\ell^{(1)}(x)$ and $[y j_\ell(y)]'/y \sim \frac{\ell}{y}j_\ell(y)$, so $\|\mathbf{N}_{\ell m}^{(1)}(kR\hat{x})\|_{L^2(S_R)} \sim \frac{\sqrt{2}\ell}{kR}R|h_\ell^{(1)}(kR)|$ and $\|\mathbf{N}_{\ell m}^{(0)}(ky)\|_{L^2(B_a)} \sim \frac{\sqrt{2}\ell}{k}\|j_\ell(kr)\|_{L^2([0,a], dr)}$. Multiplying by $k$ yields:
    $$\sigma_\ell^{\text{TM}} \sim \frac{2\ell^2}{k^2 R} \frac{\sqrt{a}}{(2\ell+1)\sqrt{2\ell+1}}\left(\frac{a}{R}\right)^\ell = \Theta\!\left(\frac{\sqrt{a}\,\ell^{1/2}}{k^2 R}\rho^\ell\right) \qquad (\beta = -1/2,\ \text{dimension } \text{Length}^{3/2}). \tag{4.8}$$
  Because Theorem 2.4 holds for **negative as well as positive $\beta$**, both $\sigma_\ell^{\text{TE}} = \Theta(\ell^{-3/2}\rho^\ell)$ and $\sigma_\ell^{\text{TM}} = \Theta(\ell^{1/2}\rho^\ell)$ satisfy (2.1) at fixed $k > 0$ and obey the exact same polylog degree-$3$ scaling!

---

## 5. Two-Regime Scaling & Propagating-to-Evanescent Crossover

For wave fields ($k > 0$) with $ka \gg 1$, the spectrum $\sigma_\ell$ traverses up to three regimes as $\varepsilon$ decreases:

1. **Propagating Plateau ($\varepsilon \gtrsim \varepsilon^*$, $\ell_{\text{cut}} \lesssim ka$):**
   All $N_{\text{DOF}}^{\mathbb{R}}(ka)$ oscillatory modes ($\ell \le ka$) have $\sigma_\ell \asymp \bar{\sigma}_{\text{prop}}$, while modes $\ell > ka + O((ka)^{1/3})$ decay rapidly below $\sqrt{\varepsilon/N_{\text{DOF}}}$. By Theorem 2.1–2.3 restricted to the propagating subspace,
   $$b_Q(\varepsilon) = \frac{N_{\text{DOF}}^{\mathbb{R}}(ka)}{2}\log_2\!\left(\frac{1}{\varepsilon}\right) + O\!\left(N_{\text{DOF}}(ka)\log N_{\text{DOF}}(ka)\right). \tag{5.1}$$

2. **Intermediate Caustic Regime ($ka \lesssim \ell \lesssim kR$, present when $R/a \gg 1$ and $kR \gg 1$):**
   For $ka < \ell < kR$, the outer Hankel factor remains oscillatory ($|H_\ell^{(1)}(kR)| \sim ((kR)^2-\ell^2)^{-1/4}$) while the inner Bessel factor decays super-exponentially:
   $$\sigma_\ell \asymp \ell^{-1} \left(\frac{eka}{2\ell}\right)^\ell.$$
   Setting $\sigma_{\ell_c}^2 \approx \varepsilon$ gives $2\ell_c \ln\!\left(\frac{2\ell_c}{eka}\right) \approx \ln(1/\varepsilon)$, whose exact solution via the principal Lambert-$W$ function ($u \ln u = z \implies \ln u = W(z)$) is:
   $$\ell_c(\varepsilon) \sim \frac{\ln(1/\varepsilon)}{2\, W\!\left(\frac{\ln(1/\varepsilon)}{eka}\right)}, \tag{5.2}$$
   yielding intermediate sub-polylog bit growth $b_Q(\varepsilon) \asymp \ell_c(\varepsilon)^{d-1}\ln(1/\varepsilon) = \Theta\!\left(\frac{(\ln(1/\varepsilon))^d}{[W(\ln(1/\varepsilon)/(eka))]^{d-1}}\right)$ until $\ell_c$ reaches $kR$.

3. **Evanescent Crossover Precision $\varepsilon^*$ ($\ell_c \gtrsim kR$, $\varepsilon \ll \varepsilon^*$):**
   - When $R/a = O(1)$ (no wide caustic gap), equating the propagating rate (5.1) with the leading evanescent law (Theorem 2.4) locates the crossover precision $\varepsilon^*$ explicitly:
     - **2D ($m$ real modes/order, $N_{\text{DOF}}^{\mathbb{R}}(ka) = m\cdot ka$):**
       $$\frac{m\cdot ka}{2}\log_2\!\left(\frac{1}{\varepsilon^*}\right) = \frac{m(\log_2(1/\varepsilon^*))^2}{8\log_2(R/a)} \implies \log_2\!\left(\frac{1}{\varepsilon^*}\right) = 4\,ka\log_2\!\left(\frac{R}{a}\right). \tag{5.3}$$
     - **3D ($m_0(2\ell+1)$ real modes/order, $N_{\text{DOF}}^{\mathbb{R}}(ka) = m_0(ka)^2$):**
       $$\frac{m_0(ka)^2}{2}\log_2\!\left(\frac{1}{\varepsilon^*}\right) = \frac{m_0(\log_2(1/\varepsilon^*))^3}{24(\log_2(R/a))^2} \implies \log_2\!\left(\frac{1}{\varepsilon^*}\right) = 2\sqrt{3}\,ka\log_2\!\left(\frac{R}{a}\right). \tag{5.4}$$
   - When $R/a \gg 1$, the physical spectral crossover where $\ell_c$ exits the caustic regime ($\ell_c = kR$) occurs at $\sigma_{kR}^2 \sim (ea/(2R))^{2kR}$, i.e., $\log_2(1/\varepsilon^*) \approx 2kR \log_2\!\left(\frac{2R}{ea}\right)$. In both cases, $\log_2(1/\varepsilon^*) = \Theta(kR \log_2(R/a))$: once $\varepsilon \ll \varepsilon^*$, geometric $\rho^\ell$ decay dominates and forces $b_Q(\varepsilon)$ into the evanescent polylog branch of degree $d \in \{2, 3\}$.

---

## 6. Active Near-Field Probe Jump (Requirement R2)

Suppose an initial query regime probes outside radius $R_1 > a$, and a subsequent observation or near-field probe deployment reduces the effective outer boundary radius from $R_1$ to $R_2$ with $a < R_2 < R_1$.

### 6.1 Strict Denominator Monotonicity & Discontinuous Coefficient Jump

**Theorem 6.1 (Strict Near-Field Coefficient Jump).** *For $0 < a < R_2 < R_1$,*
$$0 < \log_2\!\left(\frac{R_2}{a}\right) < \log_2\!\left(\frac{R_1}{a}\right). \tag{6.1}$$
*Consequently, the leading evanescent polylog coefficient $C_d(R) := \frac{c_d}{(\log_2(R/a))^{d-1}}$ ($c_2 = m/8$ in 2D, $c_3 = m_0/24$ in 3D) increases **strictly and discontinuously** as $R_{\text{eff}}$ drops from $R_1$ to $R_2$:*
- **2D ($d=2$, $c_2 = m/8$):**
  $$\Delta C_{\text{2D}} := C_{\text{2D}}(R_2) - C_{\text{2D}}(R_1) = \frac{m\,\log_2(R_1/R_2)}{8\,\log_2(R_1/a)\log_2(R_2/a)} > 0, \qquad \frac{C_{\text{2D}}(R_2)}{C_{\text{2D}}(R_1)} = \frac{\log_2(R_1/a)}{\log_2(R_2/a)} > 1. \tag{6.2}$$
- **3D ($d=3$, $c_3 = m_0/24$):**
  $$\Delta C_{\text{3D}} := C_{\text{3D}}(R_2) - C_{\text{3D}}(R_1) = \frac{m_0}{24}\frac{\left(\log_2(R_1/a) + \log_2(R_2/a)\right)\log_2(R_1/R_2)}{(\log_2(R_1/a))^2(\log_2(R_2/a))^2} > 0, \qquad \frac{C_{\text{3D}}(R_2)}{C_{\text{3D}}(R_1)} = \left(\frac{\log_2(R_1/a)}{\log_2(R_2/a)}\right)^2 > 1. \tag{6.3}$$

*Proof.* Since $0 < a < R_2 < R_1$, dividing by $a > 0$ gives $1 < R_2/a < R_1/a$, and strict monotonicity of $\log_2$ yields $0 < \log_2(R_2/a) < \log_2(R_1/a)$. Inverting and using $\log_2(R_1/a) - \log_2(R_2/a) = \log_2(R_1/R_2) > 0$ gives (6.2) and (6.3). The discontinuity occurs because $C_d$ depends on the *infimum* effective probe radius $R_{\text{eff}} = \inf_{q \in \mathcal{Q}_{\text{obs}}} |x_q|$: an online architecture calibrated to $R_1$ switches discontinuously from $C_d(R_1)$ to $C_d(R_2)$ the instant near-field queries at radius $R_2$ are admitted. $\square$

### 6.2 Fixed-Budget MSE Degradation

Suppose the cache bit budget $B = b_Q(\varepsilon; R_1) \sim C_d(R_1)(\log_2(1/\varepsilon))^d$ is held fixed when the query sphere contracts to $R_2 \in (a, R_1)$:
1. **Optimal $R_2$-reallocation (re-encoding $X$ at budget $B$ for $S_{R_2}$):** Equating $C_d(R_2)(\log_2(1/\varepsilon_{\text{opt}}))^d \sim C_d(R_1)(\log_2(1/\varepsilon))^d$ gives
   $$\log_2\!\left(\frac{1}{\varepsilon_{\text{opt}}}\right) \sim \left(\frac{\log_2(R_2/a)}{\log_2(R_1/a)}\right)^{\frac{d-1}{d}} \log_2\!\left(\frac{1}{\varepsilon}\right) \implies \varepsilon_{\text{opt}} = \varepsilon^{\alpha_d + o(1)}, \qquad \alpha_d := \left(\frac{\log_2(R_2/a)}{\log_2(R_1/a)}\right)^{\frac{d-1}{d}} \in (0, 1). \tag{6.4}$$
2. **Non-adaptive $R_1$-truncated cache (no re-encoding):** An $R_1$-optimal cache discards all modes $\ell > L(R_1) \sim \frac{\log_2(1/\varepsilon)}{2\log_2(R_1/a)}$. Evaluated on $S_{R_2}$, the first omitted mode $\ell = L(R_1) + 1$ has variance $\lambda_\ell(R_2) = \Theta(\ell^{-2\beta}(a/R_2)^{2\ell})$, yielding severe MSE degradation:
   $$\varepsilon_{\text{trunc}} \ge \lambda_{L(R_1)+1}(R_2) = 2^{-2L(R_1)\log_2(R_2/a) + O(\log L(R_1))} = \varepsilon^{\frac{\log_2(R_2/a)}{\log_2(R_1/a)} + o(1)} \gg \varepsilon_{\text{opt}} \gg \varepsilon. \tag{6.5}$$

### 6.3 Near-Source Singularity ($R_2 \to a^+$) vs. Isolated Point Query

As $R_2 \to a^+$, $\log_2(R_2/a) = \frac{R_2 - a}{a\ln 2} + O((R_2 - a)^2)$, so
$$C_d(R_2) \sim c_d\left(\frac{a\ln 2}{R_2 - a}\right)^{d-1} \xrightarrow{R_2 \to a^+} +\infty. \tag{6.6}$$
At $R_2 = a$ ($\rho = 1$), geometric decay vanishes ($\sigma_\ell \asymp \ell^{-\beta}$), collapsing polylogarithmic scaling into polynomial $O(\varepsilon^{-1/(2\beta)})$ bits (or $+\infty$ for boundary data with $\beta = 0$). **Crucial distinction:** a *single isolated point query* at a fixed known point $x_0 \in S_{R_2}$ is a rank-1 linear functional $q_{x_0}(X) = u(x_0)$ of bounded variance $\sigma_{x_0}^2 < \infty$, requiring only $\frac{1}{2}\log_2(\sigma_{x_0}^2/\varepsilon) + O(\log\log(1/\varepsilon))$ bits. The polylogarithmic jump of Theorem 6.1 occurs specifically when the *effective late-query shell* (arbitrary angular queries on $S_{R_2}$) moves from $R_1$ to $R_2$.

---

## 7. Failure Modes & Explicit Counterexamples (Requirement R3)

The evanescent polylogarithmic bounds of Sections 2–5 rely on three structural assumptions: (A1) a **linear** source-to-field map $\mathcal{S}$ with fixed exterior Green's function (no unknown high-contrast multiple scattering), (A2) **Gaussian** (or sub-Gaussian / bounded-ellipsoid) source priors, and (A3) **spherical-average $L^2(S_R)$** query distortion rather than worst-case $L^\infty(S_R)$ or super-directive queries. Violating any of (A1)–(A3) produces provable breakdown.

### 7.1 Failure Mode 1: Nonlinear Multiple Scattering (Unknown Permittivity $\epsilon(r)$)

Consider electromagnetic scattering by a sphere $B_a$ containing an unknown relative permittivity $\varepsilon_r(r) = \epsilon(r)/\epsilon_0 \ge 1$ under plane-wave illumination $\mathbf{E}^{\text{inc}}$. The map $\epsilon \mapsto \mathbf{E}^{\text{scat}}|_{S_R}$ is **nonlinear** (via the Lippmann–Schwinger resolvent $(\mathbf{I} - k^2 \mathbf{G}_e (\varepsilon_r - 1))^{-1}$).

**Counterexample 7.1 (Whispering-Gallery Mie Resonance Breakdown of $ka$-Calibration).**
Let $B_a$ be a homogeneous dielectric sphere of radius $a$ with unknown refractive index $n = \sqrt{\varepsilon_r} > 1$, illuminated by a unit plane wave at wavenumber $k$. In Mie theory, the TE scattered multipole coefficient at order $\ell \ge 1$ is
$$b_\ell(n) = \frac{U_\ell(n)}{U_\ell(n) + i\, V_\ell(n)}, \qquad \begin{cases} U_\ell(n) := S_\ell(nka) S_\ell'(ka) - n S_\ell'(nka) S_\ell(ka) \in \mathbb{R}, \\ V_\ell(n) := S_\ell(nka) C_\ell'(ka) - n S_\ell'(nka) C_\ell(ka) \in \mathbb{R}, \end{cases}$$
where $S_\ell(x) := x j_\ell(x)$ and $C_\ell(x) := x y_\ell(x)$ are the Riccati–Bessel functions. Whenever the dielectric contrast $n$ hits a whispering-gallery resonance pole $V_\ell(n) = 0$ (which occurs densely in $n$ across the window $kR < \ell < nka$), the scattering coefficient **saturates unitarity**:
$$|b_\ell(n)| = 1,$$
while the scattered field amplitude on $S_R$ ($\ell > kR$) carries the outgoing Hankel factor:
$$|b_\ell(n)\, h_\ell^{(1)}(kR)| = |h_\ell^{(1)}(kR)| \sim \frac{(2\ell - 1)!!}{(kR)^{\ell+1}} \sim \frac{1}{\sqrt{2}\,kR}\left(\frac{2\ell}{e kR}\right)^\ell \gg 1! \tag{7.1}$$
*Explicit verified witness:* At $ka = 2.0$, $kR = 3.0$, and multipole order $\ell = 8$ (strictly in the "evanescent" range $\ell = 4ka > kR$), solving $V_8(n) = 0$ numerically yields a sharp TE resonance at $n_* \approx 5.81661694$ ($\varepsilon_r \approx 33.83$), where:
$$|b_8(n_*)| = 1.000000, \qquad |h_8^{(1)}(3.0)| = 140.0601 \implies |b_8(n_*) h_8^{(1)}(3.0)| = 140.0601.$$
(Energy conservation in the far field $kr \to \infty$ is preserved because $|h_\ell^{(1)}(kr)| \sim (kr)^{-1}$; the $140\times$ amplification on $S_R$ is stored reactive/near-field cavity energy.)

**Consequences for Late-Query Caching:**
1. **Collapse of $ka$-calibrated propagating plateau:** The effective propagating cutoff expands from $ka$ to $n_{\max} ka = ka\sqrt{\varepsilon_{r,\max}}$. If $\varepsilon_r$ has unbounded support ($n_{\max} \to \infty$), no $ka$-calibrated polylog bound holds at all.
2. **Non-ellipsoidal manifold & Fano sensitivity:** Resonance widths decay super-exponentially as $\Delta n_{\text{res}} \sim (eka/(2\ell))^{2\ell}$, yielding quality factors $Q_\ell \sim (2\ell/(eka))^{2\ell}$ and extreme sensitivity $\|\partial_n \mathbf{E}^{\text{scat}}\|_{L^2(S_R)} \sim Q_\ell |h_\ell^{(1)}(kR)|$.
3. **Mandache (2001) logarithmic inverse-scattering instability:** When late queries probe scattering across multiple incident angles or near-field interactions that implicitly depend on $\epsilon(r)$, Mandache's theorem shows that the forward map $\epsilon \mapsto \Lambda_\epsilon$ has only **logarithmic stability** ($\|\epsilon_1 - \epsilon_2\|_{L^\infty} \lesssim (\log(1/\delta))^{-s}$), so recovering $\epsilon$-dependent multi-illumination responses to $\mathrm{MSE} \le \varepsilon$ is incompatible with any linear ellipsoid rate-distortion guarantee.

### 7.2 Failure Mode 2: Non-Gaussian / Heavy-Tail Source Priors

Even for a purely linear Laplace/Helmholtz operator ($\beta = 3/2$) and a **single** monopole mode $\ell = 0$, departing from Gaussianity breaks the polylogarithmic scaling in both directions:

1. **Counterexample 7.2A (Infinite-Variance Prior, e.g., Cauchy or Student-$t_\nu$ with $\nu \le 2$):**
   Let $y_0$ have a Cauchy or Student-$t_\nu$ ($\nu \le 2$) prior, so $\mathbb{E}[y_0^2] = +\infty$. For **any** finite $b$-bit codebook $\mathcal{C} = \{\hat{y}^{(1)}, \dots, \hat{y}^{(2^b)}\}$ with $M := \max_j |\hat{y}^{(j)}| < \infty$, on the event $|y_0| \ge 2M$ we have $\min_j |y_0 - \hat{y}^{(j)}|^2 \ge y_0^2/4$, so
   $$\mathbb{E}\!\left[\min_{j} |y_0 - \hat{y}^{(j)}|^2\right] \ge \frac{1}{4}\mathbb{E}[y_0^2 \mathbf{1}_{|y_0| \ge 2M}] = +\infty \implies b_Q(\varepsilon) = +\infty \quad \forall\,\varepsilon < \infty.$$

2. **Counterexample 7.2B (Finite-Variance Ultra-Heavy-Tail Prior Forcing $b_Q(\varepsilon) = \Omega(1/\varepsilon)$):**
   Let $y_0$ have symmetric density supported on $|y| \ge e^e$:
   $$p(y) = \frac{c}{|y|^3 (\ln|y|)(\ln\ln|y|)^2}\,\mathbf{1}_{|y| \ge e^e}, \qquad c := \left(2\int_{e^e}^\infty \frac{dy}{y^3(\ln y)(\ln\ln y)^2}\right)^{-1} < \infty.$$
   Check the second moment: since $\frac{d}{dy}\left(-\frac{1}{\ln\ln y}\right) = \frac{1}{y(\ln y)(\ln\ln y)^2}$,
   $$\mathbb{E}[y_0^2] = 2c\int_{e^e}^\infty \frac{dy}{y(\ln y)(\ln\ln y)^2} = 2c < \infty.$$
   (Rescaling $y_0 \leftarrow y_0 / \sqrt{2c}$ matches $\mathbb{E}[y_0^2] = 1$, identical to $\mathcal{N}(0, 1)$!)
   Now let $\mathcal{C} \subset \mathbb{R}$ be **any** fixed-length $b$-bit codebook of size $M = 2^b$. Consider the pairwise interior-disjoint dyadic intervals $I_k := [2^k, 2^{k+1}]$ for integers $k \ge k_0 := \lceil e / \ln 2\rceil \ge 4$ (so $2^{k_0} > e^e$), and their middle halves $J_k := [\frac{5}{4}2^k, \frac{7}{4}2^k]$ of length $2^{k-1}$.
   - If $\mathcal{C} \cap (2^k, 2^{k+1}) = \emptyset$, then every $y \in J_k$ has distance $\ge 2^{k-2}$ to $\mathcal{C}$, while on $J_k$ we have $y \le 2^{k+1}$, $\ln y \le (k+1)\ln 2$, and $\ln\ln y \le \ln((k+1)\ln 2) \le \ln(k+1)$. Hence each empty $I_k$ contributes MSE at least
     $$\int_{J_k} (2^{k-2})^2 p(y)\,dy \ge 2^{2k-4} \cdot 2^{k-1} \cdot \frac{c}{2^{3k+3}(k+1)\ln 2\,(\ln(k+1))^2} = \frac{c_0}{(k+1)(\ln(k+1))^2}, \qquad c_0 := \frac{c}{256\ln 2} > 0.$$
   - Since $|\mathcal{C}| = M$, at most $M$ open intervals $(2^k, 2^{k+1})$ with $k \ge k_1 := \max(k_0, 2M)$ contain a codeword. Because $\frac{c_0}{(k+1)(\ln(k+1))^2}$ is strictly decreasing in $k$, the worst-case (smallest) sum over the empty indices in $\{k \ge k_1\}$ occurs if the $M$ occupied intervals are the first $M$ indices $k_1, \dots, k_1 + M - 1$. Therefore:
     $$\mathrm{MSE} \ge \sum_{k = k_1 + M}^{\infty} \frac{c_0}{(k+1)(\ln(k+1))^2} \ge c_0 \int_{k_1 + M + 1}^\infty \frac{dx}{x(\ln x)^2} = \frac{c_0}{\ln(k_1 + M + 1)}.$$
   - For all $\varepsilon < \frac{c_0}{\ln(2k_0)}$, the bound $\mathrm{MSE} \le \varepsilon$ forces $k_1 + M + 1 > 2k_0$, hence $M > k_0/2$ (since $k_0 \ge 4$), so $k_1 = 2M$ and $k_1 + M + 1 = 3M + 1$. Thus $\ln(3M + 1) \ge c_0/\varepsilon$, i.e.,
     $$b_Q(\varepsilon) = \log_2 M \ge \frac{c_0}{\ln 2}\cdot \frac{1}{\varepsilon} - O(1) = \Omega\!\left(\frac{1}{\varepsilon}\right), \tag{7.2}$$
     an **exponential blowup** over $\log_2(1/\varepsilon)$ despite finite variance and an identical covariance ellipsoid to the Gaussian prior!

3. **Counterexample 7.2C (Sparse Spike-and-Slab Breaking the Gaussian Lower Bound):**
   Let $y_{\ell,m} = \rho^\ell z_{\ell,m}$ with independent discrete variables
   $$z_{\ell,m} = \begin{cases} +\rho^{-\ell} & \text{with prob. } p_\ell/2, \\ 0 & \text{with prob. } 1 - p_\ell, \\ -\rho^{-\ell} & \text{with prob. } p_\ell/2, \end{cases} \qquad p_\ell := \rho^{2\ell}.$$
   Every mode has $\mathbb{E}[y_{\ell,m}^2] = \rho^{2\ell} \cdot (p_\ell \rho^{-2\ell}) = \rho^{2\ell}$, matching the Gaussian covariance spectrum identically! Yet:
   - **Variable-length lossless entropy:** $H(y_{\ell,m}) = H_2(p_\ell) + p_\ell = O(\ell\rho^{2\ell})$, so $\sum_{\ell,m} H(y_{\ell,m}) = O(\sum_\ell \ell^{d-1}\rho^{2\ell}) = O(1)$ expected bits even at $\varepsilon = 0$.
   - **Fixed-length lossy cache $b_Q(\varepsilon)$:** Truncate at $L = \Theta(\log(1/\varepsilon))$ so $\sum_{\ell > L} m_\ell \rho^{2\ell} \le \varepsilon/2$, leaving $N_{\text{act}} = O((\log(1/\varepsilon))^{d-1})$ modes. The number $S$ of nonzero spikes in $\ell \le L$ is a sum of independent Bernoullis with mean $\mu = \sum_{\ell \le L} m_\ell \rho^{2\ell} \le C_\rho < \infty$. Since each spike has $|y_{\ell,m}| = 1$, encoding only the positions and signs of up to $r = \lceil 2\ln(1/\varepsilon)/\ln\ln(1/\varepsilon)\rceil$ spikes (overflow error $\mathbb{E}[S\mathbf{1}_{S>r}] \le \varepsilon/2$ by Chernoff) takes
     $$b_Q(\varepsilon) \le \left\lceil \log_2 \sum_{j=0}^r \binom{N_{\text{act}}}{j} 2^j \right\rceil \le r \log_2(2N_{\text{act}} + 1) + 1 = O\!\left(\frac{\log(1/\varepsilon)}{\log\log(1/\varepsilon)} \cdot \log\log(1/\varepsilon)\right) = O\!\left(\log \frac{1}{\varepsilon}\right),$$
     strictly breaking the $\Omega((\log_2(1/\varepsilon))^d)$ Gaussian lower bound!

### 7.3 Failure Mode 3: Worst-Case vs. Average-Case Outer-Sphere Queries

1. **Pointwise $L^\infty(S_R)$ vs. Spherical-Average $L^2(S_R)$ Amplification ($\Theta(N_{\text{act}})$ Factor):**
   Let $\hat{u}$ be any $L^2(S_R)$-optimal reconstruction supported on active modes $\ell \le L$ ($N_{\text{act}} = 2L + 1$ in 2D; $N_{\text{act}} = (L+1)^2$ in 3D) with surface-averaged $L^2$ error $\frac{1}{|S_R|}\int_{S_R} |e|^2 dS \le \varepsilon$. By Cauchy–Schwarz and the spherical harmonic addition theorem, for every $\hat{x}_0 \in S_1$:
   $$|e(\hat{x}_0)|^2 = \left|\sum_{\ell=0}^L \sum_{m} e_{\ell,m} |S_1|^{1/2} Y_\ell^m(\hat{x}_0)\right|^2 \le \left(\sum_{\ell,m} |e_{\ell,m}|^2\right) \left(|S_1|\sum_{\ell=0}^L \sum_m |Y_\ell^m(\hat{x}_0)|^2\right) = N_{\text{act}} \cdot \varepsilon.$$
   **Constructive realizability inside $\mathcal{E}$:** Fix $\hat{x}_0 \in S_1$ (say the north pole $\hat{e}_z$ in 3D, where $Y_\ell^m(\hat{e}_z) = \sqrt{\frac{2\ell+1}{4\pi}}\delta_{m,0}$), and let $L' := L - \left\lceil \frac{\ln(C_0 L/(1-\rho^2))}{2\ln(1/\rho)}\right\rceil \sim L$ (for a sufficiently large constant $C_0 > 0$; explicitly, $C_0 \ge 16$ suffices across all finite $L \ge 10$), so $N_{\text{act}}' = \Theta(N_{\text{act}})$. Aligning the residual $e = u - \hat{u}$ with the **Dirichlet reproducing kernel** $K_{L'}(\hat{x}, \hat{x}_0) := \sum_{\ell=0}^{L'}\sum_m Y_\ell^m(\hat{x})\overline{Y_\ell^m(\hat{x}_0)}$ (which satisfies $\|K_{L'}(\cdot,\hat{x}_0)\|_{L^2(S_1)}^2 = K_{L'}(\hat{x}_0,\hat{x}_0) = N_{\text{act}}'/|S_1|$):
   $$e(\hat{x}) := |S_1| \sqrt{\frac{\varepsilon}{N_{\text{act}}'}} \, K_{L'}(\hat{x}, \hat{x}_0)$$
   satisfies $\frac{1}{|S_1|}\int_{S_1} |e|^2 d\Omega = \varepsilon$ AND belongs to $\frac{1}{\sqrt{2}}\mathcal{E}$ (since the surface-averaged variance satisfies $\tilde{\sigma}_L^2 \gtrsim \varepsilon/(E_0^2 N_{\text{act}})$, giving $\sum_{\ell \le L'} \frac{\varepsilon m_\ell / N_{\text{act}}'}{\tilde{\sigma}_\ell^2} \lesssim \frac{m_{L'}}{N_{\text{act}}'(1-\rho^2)} \frac{\varepsilon}{\tilde{\sigma}_{L'}^2} \le \frac{1}{2} E_0^2$ by choice of $L'$), yet achieves the maximal constructive focal amplification:
   $$\|e\|_{L^\infty(S_R)}^2 = |e(\hat{x}_0)|^2 = N_{\text{act}}' \,\varepsilon = \begin{cases} \Theta\!\left(\log(1/\varepsilon)\right)\cdot \varepsilon & (d = 2), \\ \Theta\!\left((\log(1/\varepsilon))^2\right)\cdot \varepsilon & (d = 3). \end{cases} \tag{7.3}$$
   *(Refinement for coordinate-wise scalar quantizers: If a quantizer enforces $|e_{\ell,m}| \le \sqrt{\varepsilon/N_{\text{act}}}$ per mode rather than total $L^2$-ball error $\varepsilon$, then at $\hat{e}_z$ only the $L+1$ zonal modes $m=0$ are nonzero, capping $|e(\hat{e}_z)|^2$ at $\Theta(L\varepsilon)$ in 3D as well as 2D; however, any general $L^2(S_R)$-ball $\sqrt{\varepsilon}$-covering permits the full Dirichlet residual (7.3) with $\Theta(L^2\varepsilon)$ in 3D.)*

2. **Re-provisioning Cost for Uniform $L^\infty(S_R)$ Guarantees:**
   To enforce worst-case pointwise squared error $\|u - \hat{u}\|_{L^\infty(S_R)}^2 \le \varepsilon$ across all angles on $S_R$, it suffices to tighten the surface-averaged $L^2$-MSE target to $\varepsilon' := \varepsilon / N_{\text{act}}(\varepsilon)$ (plus tail error). Replacing $\log_2(1/\varepsilon)$ by $\log_2(1/\varepsilon) + \log_2 N_{\text{act}}(\varepsilon) = \log_2(1/\varepsilon) + (d-1)\log_2 L + O(1)$ increases the bit budget by:
   $$\Delta b_{\infty} = \Theta\!\left(N_{\text{act}}(\varepsilon) \log N_{\text{act}}(\varepsilon)\right) = \begin{cases} \Theta\!\left(\log(1/\varepsilon)\log\log(1/\varepsilon)\right) & (d = 2), \\ \Theta\!\left((\log(1/\varepsilon))^2 \log\log(1/\varepsilon)\right) & (d = 3), \end{cases}$$
   which is subleading as $\varepsilon \to 0$ but substantial at finite precision.

3. **Super-Directive / Inward-Continuation Query Functionals ($\rho^{-2\ell}$ Amplification & Regime-Dependent Divergence):**
   If a late query requests a super-directive beam pattern or inward analytic continuation $q(u) = \int_{S_R} w(\hat{x}) u(R\hat{x})\,dS$ to an interior shell $r = a' \in [a, R)$ whose harmonic weights grow as $|w_{\ell,m}| \asymp (R/a')^\ell$ (in particular $a' = a \implies |w_{\ell,m}| \asymp \rho^{-\ell}$), any $L^2(S_R)$-calibrated cache with per-mode distortion $D_\ell \sim \varepsilon/N_{\text{act}}$ ($\ell \le L$) and uncoded tail $\lambda_\ell \asymp \ell^{-2\beta}\rho^{2\ell}$ ($\ell > L$) incurs uncoded-tail query error $\sum_{\ell > L} m_\ell |w_\ell|^2 \lambda_\ell \asymp \sum_{\ell > L} m_\ell \ell^{-2\beta} (a/a')^{2\ell}$ (and at $a < a' < R$ triggers the near-field jump of Theorem 6.1 from $R$ to $a'$). At $a' = a$, the tail behavior depends sharply on $\beta$:
   - **Boundary-to-boundary ($\beta = 0$) and Maxwell TM ($\beta = -1/2$), i.e., $\beta \le (d-1)/2$:** Since $m_\ell \asymp \ell^{d-2}$, $\sum_{\ell > L} m_\ell \ell^{-2\beta} = +\infty$ at $a' = a$, so the uncoded tail diverges outright.
   - **Volumetric scalar potentials ($\beta = 3/2 > (d-1)/2$, and Maxwell TE):** At $a' = a$, the $L^2(S_a)$ continuation tail is finite, $\sum_{\ell > L} m_\ell \ell^{-3} = \Theta(L^{d-4}) = \Theta((\log(1/\varepsilon))^{d-4}) \gg \varepsilon$ ($O((\log(1/\varepsilon))^{-2})$ in 2D and $O((\log(1/\varepsilon))^{-1})$ in 3D), producing a polylogarithmically slow error that still violates $\mathrm{MSE} \le \varepsilon$ as $\varepsilon \to 0$. True $+\infty$ divergence for volumetric scalar potentials occurs either for any strict interior radius $a' < a$, where $\sum_{\ell > L} m_\ell \ell^{-3}(a/a')^{2\ell} = +\infty$, or at $a' = a$ for super-directive weights $|w_\ell| \asymp \sqrt{m_\ell}\,\rho^{-\ell}$ across modes in 3D (giving $\sum_{\ell > L} m_\ell^2 \ell^{-3} \asymp \sum_{\ell > L} \ell^{-1} = +\infty$) or inward normal-derivative evaluation $\partial_r u(a\hat{x}_0)$ (whose radial derivative adds a factor $\ell/a$, giving $\sum_{\ell > L} \ell^{-1} = +\infty$ in 2D and $\sum_{\ell > L} 1 = +\infty$ in 3D).

---

## 8. Cross-Reference to Machine-Checked Formalization (`WavePolylog.lean`)

The load-bearing algebraic and information-theoretic core of Sections 2, 3, and 6 is formalized in `WavePolylog.lean` (Lean 4 + Mathlib4 v4.36.0, zero `sorry`, standard axioms `[propext, Classical.choice, Quot.sound]` only):
- **Multipole sums (Section 3):** Exact identities and sharp bounds for both $\sum_{\ell=0}^L (2\ell+1)(L-\ell) = \frac{L(L+1)(2L+1)}{6} \in [\frac{L^3}{3}, \frac{(L+1)^3}{3}]$ and 2D $m$-mode / $1$-mode / full circular harmonic sums.
- **Reverse water-filling optimality (Theorem 2.1):** Derivative-free Lagrangian inequality via `Real.log_le_sub_one_of_pos`, constructive feasibility witness $D_i^* = \min(\lambda_i, \theta)$, non-asymptotic geometric-decay polylog lower bounds in 2D and 3D, and the *truncated* (`0..L`) active-mode geometric-decay upper bounds (`evanescent_3d_upper_bound_truncated`, `evanescent_2d_full_upper_bound_truncated`), while the infinite-tail ($\ell > L$) single-shot achievability bound is proved analytically in Section 2.2 above.
- **Active near-field probe jump (Theorem 6.1):** Strict positivity and monotonicity $0 < \log_2(R_2/a) < \log_2(R_1/a)$ for $0 < a < R_2 < R_1$, inverse jump inequalities, additive gaps, and exact multiplicative jump ratios.
