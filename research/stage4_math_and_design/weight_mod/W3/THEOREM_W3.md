# Track W3 (Round 2 Converged): Anti-Forgetting Fisher/Nullspace Geometry, Two-Tier Token-Position Gating, Complex-Symmetrized `W2` Compatibility, Early-GDN & Stop-Grad Resolution (`4B→9B→27B`, `|V|=248,320`) & Calibrated Objective Engineering

**EM-FloatFormer Stage 4 · Weight-Modification Track W3 (Round 1 + Round 2 Cross-Track Integration)**  
[CPU PyTorch Verifier](verify_w3.py) · [Verified Quantitative Results](results_w3.json) · [Multimodal Integration Plan](../../../../docs/MULTIMODAL_INTEGRATION_PLAN.md) · [Stage 3D Host Connector](../../../stage3d_qwen_signed_phase/qwen_connector.py) · [W1 GDN Dynamics](../W1/THEOREM_W1.md) · [W2 Equivariant Cayley](../W2/THEOREM_W2.md) · [W4 Systems & VRAM](../W4/THEOREM_W4.md) · [M2 Forensics](../../M2/THEOREM_M2.md) · [M5 Dual Cache](../../M5/THEOREM_M5.md) · [M6 Execution Lock](../../M6/THEOREM_M6.md)

> **Executive Thesis (Round 2 Converged with `W1, W2, W4, M1–M6`).** If Stage 4 triggers direct backbone weight modification on `Qwen3.5-4B`, `Qwen3.5-9B`, or `Qwen3.8-27B` (all verified at exact vocabulary **`|V| = 248,320`** matching `W4` and `smoke_audit.json` to `0.0000%`), **zero off-task regression and `< 10^{-3}` nat mixed-turn fluency protection are constructive mathematical invariants**:
> 1. **Off-task (ordinary text & vision-text without EM payloads)**: A structural out-of-band binary modality gate $g_{\mathrm{EM}}(x) \in \{0,1\}$ bypassing $\Delta W$ completely guarantees **bit-exact zero regression** (`PureKL = 0.000e+00`, `max_abs_logit_diff = 0.000e+00`, `host_sha256` preserved verbatim).
> 2. **On-task (mixed multi-turn EM conversations where $g_{\mathrm{EM}}(x)=1$)**: Modality gating *alone* fails mixed-turn fluency (`MixKL = 3.687e-1` nats in Round 2; `7.518e-1` in Round 1). Upgrading Round 1's sequence-gated hybrid (`MixKL = 5.052e-3` nats) to **Theorem 4.3 Two-Tier Token-Position Gating** ($m_{\mathrm{EM}}(t) \in \{0,1\}$ on $K, V, \beta, \mathrm{FFN}$ + sequence-gated $P_{\perp, C}$ & `MixKL` anchor on $Q, O, \alpha$) cuts `MixKL` to **`1.191e-3`** standalone (`900×` below unconstrained) and **`8.984e-4` nats (`<= 1.0e-3` trust-region target!, `1,193×` below unconstrained)** when paired with M1/W2's Quotient Wrapper (`G1 = 61.81% / 1.0167`, `Q2_gain = +7.64 pp`, `C4_max = 0.000e+00`).
> 3. **Exact `|V| = 248,320` Architecture Lock & Resolution of Tensions A (`W1`) & B (`W4`)**:
>    - Exact parameter parity with `W4`: **`4,205,751,296`** (`4B`, `tie_word_embeddings=True`), **`8,953,803,264`** (`9B`, `False`), **`26,895,998,464`** (`27B`, `False`). Even untied `lm_head` alone holds `1.017B–1.271B` params (`2.02M` params for a single $r=8$ LoRA!), reinforcing strict `lm_head` freezing.
>    - **Tension B (`W4 Theorem 1`)**: Quantified honestly — freezing layers `1..10` (4B/9B) or `1..18` (27B) saves **0% backward activations** if the Layer-0 connector is co-tuned (`Option B2`), and realizes the full **`-12,110.5 MiB` (4B) / `-14,176.4 MiB` (9B) / `-3,862.6 MiB` (27B Mixed-NF4)** VRAM savings and **`-17.8%` to `-19.8%` step-time savings** strictly under **`Option B1` (Connector-Frozen Stop-Grad after Layer 10 / 18)** — which is our converged default.
>    - **Tension A (`W1` Early-GDN Slow-Head Starvation)**: Naive early `dt_bias` warm-starting without strict prose write/read isolation (`leaky_beta`) triggers a **`90×` `MixKL` explosion** (`1.256e-1` nats) and `-9.0 pp` `G1` drop on `L_gap = 32`. Our zero-parameter **W1+W3 Dual-Track Nullspace-Filtered Slow-Head Highway** ($m_{\mathrm{EM}}(t)$ strict $\beta_{\mathrm{prose}} = 0$ + $P_{\perp, C}$-filtered query readout on Fisher-selected lowest-energy heads) lifts early `L_gap = 32` GDN retention **`77,919×`** (`7e-6 → 0.77919`), preserves `Option B1` `stop_grad` (`0` early trainable params), maintains `PureKL = 0.000e+00`, and cuts `MixKL` **`107×`** vs `leaky_beta` (`1.172e-3` nats).
> 4. **Symmetry Compatibility with `W2` & Shortened `69 → 96`-Step Pilot**:
>    - **Complex-Symmetrized Nullspace $P_{\perp, C}^{\mathbb{C}}$** achieves **`||[P_{\perp, C}^{\mathbb{C}}, J_d]||_F = 0.000e+00`** (vs `3.376` for real PCA), exact Loewner bound $C_{\mathrm{gen}} \preceq 2 C_{\mathrm{gen}}^{\mathbb{C}}$, **`860.11×`** KL suppression (`> 100×`), and exact compatibility with `W2`'s Nullspace-Restricted Complex-Cayley Woodbury adapter $Q^\perp$ (`||Q^\top Q - I||_F = 1.58e-15`, `||[Q, J_d]||_F = 0.0`, `||(Q-I)(I-P_C)||_F = 2.88e-15`, **`729.83×`** KL cut).
>    - Setting **`λ_orbit = 0`** under M1/W2's Quotient Wrapper drops the `rot1` forward pass (`5.77 s` vs `5.91 s/seed`), enforces exact `C4_max = 0.000e+00`, achieves the **highest `G1` (`61.81%`) & `G1–G4` (`55.12%`)**, and reaches **`99.3%` of Step-160 `G1` by Step 96** (`61.37% / 1.0239`, `MixKL = 9.002e-4`), supporting W4 in shortening Phase B from `192 → 96` steps (with Step 69 as the early GO/NO-GO gate).
> 5. **Performance Refactor**: `verify_w3.py` refactored from `947.39 s` (Round 1) to **`132.33 s`** (`<= 160 s`, **`7.16×` faster**), exit code `0`, all integrity assertions green.

---

## 1. Exact Second-Order General-Domain KL Forgetting Bound & Nullspace Geometry

Let $P_W(y \mid x)$ denote the frozen pretrained conditional next-token (or autoregressive sequence) distribution of the host LM (`Qwen3.5-4B / 9B / 3.8-27B`) on general text and multimodal benchmarks $x \sim \mathcal{D}_{\mathrm{gen}}$, with vectorized parameters $\theta = \mathrm{vec}(W) \in \mathbb{R}^P$ and perturbation $\Delta\theta = \mathrm{vec}(\Delta W)$. Define the population general-domain KL forgetting functional:
$$
D(\Delta\theta) := \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{gen}}}\!\Big[D_{\mathrm{KL}}\big(P_W(\cdot \mid x)\,\big\|\,P_{W+\Delta W}(\cdot \mid x)\big)\Big].
$$

### Theorem 1 — Exact Fisher–Rao KL Expansion, Generalized Self-Concordance Bracket, and Trust-Region Certification

Assume $\theta \mapsto \log P_\theta(y \mid x)$ is $C^3$ on the closed ball $B(\theta, r) = \{\theta' : \|\theta' - \theta\|_2 \le r\}$ for every $(x, y)$ in the finite vocabulary support ($|V| = 248,320$). Define the **True Fisher Information Matrix** (sampling $y$ from the *model* $P_W(\cdot \mid x)$, **not** empirical labels):
$$
F_{\mathrm{gen}} := \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{gen}},\, y \sim P_W(\cdot \mid x)}\!\Big[\nabla_\theta \log P_W(y \mid x)\,\nabla_\theta \log P_W(y \mid x)^\top\Big] \succeq 0.
$$

Then:
1. **Exact Second-Order Expansion with Integral Remainder.** For all $\|\Delta\theta\|_2 \le r$,
   $$
   D(\Delta\theta) = \frac{1}{2}\,\Delta\theta^\top F_{\mathrm{gen}}\,\Delta\theta + R_3(\Delta\theta), \qquad
   R_3(\Delta\theta) = \frac{1}{2}\int_0^1 (1-t)^2 \left.\frac{d^3}{ds^3} D(s\Delta\theta)\right|_{s=t}\,dt.
   $$
   Under a third-derivative operator bound $M_3 := \sup_{s \in [0,1],\,\|u\|_2=1} |\nabla^3 D(s\Delta\theta)[u,u,u]|$, the cubic remainder satisfies $|R_3(\Delta\theta)| \le \frac{M_3}{6}\|\Delta\theta\|_2^3$.

2. **Exact Non-Asymptotic Softmax Self-Concordance Bracket (Linear-Logit / Local-Tangent Form).** Write $P_\theta(y \mid x) = \mathrm{softmax}(z_\theta(x))_y$ with log-partition $\psi(z) = \log\sum_y e^{z_y}$, and let the logit perturbation be $\delta z(x) = z_{W+\Delta W}(x) - z_W(x) = v(x) + \rho_2(x)$, where $v(x) = J_z(x)\Delta\theta$ is linear in $\Delta\theta$ and $\rho_2(x)$ is the nonlinear multi-layer logit curvature remainder ($\|\rho_2(x)\|_{\mathrm{osc}} \le \frac{L_z}{2}\|\Delta\theta\|_2^2$). Define the oscillation seminorm $\mathrm{osc}(u) := \max_y u_y - \min_y u_y$. Whenever $\rho_2 \equiv 0$ (exact for final readout perturbations or linearized NTK tangents) and $\sup_x \mathrm{osc}(v(x)) \le \gamma$:
   $$
   \frac{e^{-\gamma} + \gamma - 1}{\gamma^2}\,\Delta\theta^\top F_{\mathrm{gen}}\,\Delta\theta
   \;\le\; D(\Delta\theta) \;\le\;
   \frac{e^{\gamma} - \gamma - 1}{\gamma^2}\,\Delta\theta^\top F_{\mathrm{gen}}\,\Delta\theta.
   $$
   For general nonlinear deep perturbations with $\sup_x \mathrm{osc}(\delta z(x)) \le \tilde{\gamma} := \gamma + \frac{L_z}{2}\|\Delta\theta\|_2^2$, replacing $v$ by $\delta z$ gives the exact nonlinear bound:
   $$
   D(\Delta\theta) \le \frac{e^{\tilde{\gamma}} - \tilde{\gamma} - 1}{\tilde{\gamma}^2}\,\mathbb{E}_x\big[\operatorname{Var}_{P_W(\cdot\mid x)}(\delta z(x))\big]
   \le \frac{e^{\tilde{\gamma}} - \tilde{\gamma} - 1}{\tilde{\gamma}^2}\left(\sqrt{\Delta\theta^\top F_{\mathrm{gen}}\Delta\theta} + \frac{L_z}{4}\|\Delta\theta\|_2^2\right)^2.
   $$

3. **Explicit Spectral Trust-Region Condition ($\le 10^{-3}$ nats).** Let $\varepsilon_{\mathrm{KL}} = 10^{-3}$ nats. If $\Delta\theta$ satisfies the coupled **Fisher–Oscillation Trust Region**:
   $$
   \tilde{\gamma} \le 0.25 \quad\text{and}\quad
   \Delta\theta^\top F_{\mathrm{gen}}\,\Delta\theta \le \frac{2\eta\,\varepsilon_{\mathrm{KL}}}{2\phi(\tilde{\gamma})} \approx 1.65\times 10^{-3}\;\;(\eta = 0.90), \quad
   \frac{L_z}{4}\|\Delta\theta\|_2^2 \le 0.05\sqrt{\Delta\theta^\top F_{\mathrm{gen}}\Delta\theta},
   $$
   where $\phi(\gamma) := (e^\gamma - \gamma - 1)/\gamma^2$ ($\phi(0.25) = 0.5444$), then $D(\Delta\theta) \le 10^{-3}$ nats is strictly certified. By Pinsker's inequality and Jensen's inequality, this implies:
   - Perplexity multiplier $\exp(D(\Delta\theta)) \le e^{0.001} = 1.0010005$ ($< 0.10\%$ perplexity increase);
   - Mean total-variation shift $\mathbb{E}_x[\mathrm{TV}(P_W, P_{W+\Delta W})] \le \sqrt{D(\Delta\theta)/2} \le 0.02236$.

**Proof.** At $s = 0$, $D(0) = 0$. Differentiating $D(s\Delta\theta) = \mathbb{E}_x\sum_y P_W(y\mid x)[\log P_W(y\mid x) - \log P_{W+s\Delta W}(y\mid x)]$:
$$
D'(0) = -\mathbb{E}_x\sum_y P_W(y\mid x)\,\nabla_\theta \log P_W(y\mid x)^\top\Delta\theta
= -\mathbb{E}_x\Big[\nabla_\theta\!\Big(\sum_y P_W(y\mid x)\Big)^\top \Delta\theta\Big] = 0.
$$
At second order, using $\nabla_\theta^2 \log P = P^{-1}\nabla_\theta^2 P - (\nabla_\theta\log P)(\nabla_\theta\log P)^\top$ and $\sum_y \nabla_\theta^2 P_W(y\mid x) = \nabla_\theta^2(1) = 0$:
$$
D''(0) = -\mathbb{E}_{x, y\sim P_W}[\Delta\theta^\top \nabla_\theta^2\log P_W(y\mid x)\,\Delta\theta]
= \mathbb{E}_{x, y\sim P_W}[(\nabla_\theta\log P_W(y\mid x)^\top\Delta\theta)^2] = \Delta\theta^\top F_{\mathrm{gen}}\Delta\theta.
$$
Taylor's theorem with integral remainder yields part (1). For part (2), fix $x$ and set $g(s) := \psi(z_0 + s v) - \psi(z_0) - s\,\mathbb{E}_{p_0}[v]$ so $g(0)=g'(0)=0$ and $g(1) = D_{\mathrm{KL}}(p_0 \| \mathrm{softmax}(z_0 + v))$. Writing $p_s = \mathrm{softmax}(z_0 + sv)$ and centered random variable $U_s = v_Y - \mathbb{E}_{p_s}[v_Y]$, we have $g''(s) = \operatorname{Var}_{p_s}(v) = \mathbb{E}_{p_s}[U_s^2]$ and $g'''(s) = \mathbb{E}_{p_s}[U_s^3]$. Since $U_s \in [-\mathrm{osc}(v), +\mathrm{osc}(v)]$,
$$
|g'''(s)| \le \mathbb{E}_{p_s}[|U_s|\cdot U_s^2] \le \mathrm{osc}(v)\,g''(s) \le \gamma\,g''(s)
\quad (\text{constant } 1 \text{ is tight as } p_s \to 0).
$$
By Grönwall's inequality on $\log g''(s)$, $e^{-\gamma s} g''(0) \le g''(s) \le e^{\gamma s} g''(0)$ for all $s\in[0,1]$. Integrating $g(1) = \int_0^1 (1-s) g''(s)\,ds$ yields $\frac{e^{-\gamma}+\gamma-1}{\gamma^2}g''(0) \le g(1) \le \frac{e^\gamma-\gamma-1}{\gamma^2}g''(0)$. Taking $\mathbb{E}_x$ proves the linear-logit bracket; applying Minkowski's inequality and Popoviciu's bound $\operatorname{Var}_{p_0}(\rho_2) \le \frac{1}{4}\mathrm{osc}(\rho_2)^2$ yields the nonlinear bound. $\square$

> [!NOTE]
> **True Fisher vs Empirical Fisher.** All K-FAC factors and trust-region checks in Track W3 strictly sample $y \sim P_W(\cdot\mid x)$ (True Fisher) so $\nabla^2 D(0) = F_{\mathrm{gen}}$ holds identically even on high-entropy text.

---

### Theorem 2 — Exact Geometry Comparison: Isotropic $L_2$ vs Diagonal EWC vs K-FAC vs Activation-Covariance Nullspace Projection

Consider a single linear layer $y = W x \in \mathbb{R}^{d_{\mathrm{out}}}$ ($x \in \mathbb{R}^{d_{\mathrm{in}}}$, $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$) with perturbation $\Delta W$ and backpropagated score gradient $g := \nabla_y \log P_W(Y \mid X)$ ($Y \sim P_W(\cdot\mid X)$). Using column-major vectorization, $\nabla_{\mathrm{vec}(W)} \log P_W = x \otimes g$, and the **exact** single-layer Fisher quadratic form is $Q_{\mathrm{exact}}(\Delta W) := \frac{1}{2}\,\mathbb{E}_{X, Y\sim P_W}[(g^\top \Delta W x)^2]$.
Let $C_{\mathrm{gen}} := \mathbb{E}[x x^\top] = \sum_{j=1}^{d_{\mathrm{in}}} \sigma_j u_j u_j^\top$ ($\sigma_1 \ge \dots \ge \sigma_{d_{\mathrm{in}}} \ge 0$), top-$k$ basis $U_k = [u_1,\dots,u_k]$, and tail nullspace projector $P_{\perp, C} := I - U_k U_k^\top$ ($\sigma_{k+1} \le \epsilon$).

| Geometry Proxy | Quadratic / Constraint Form | Exact Relationship to $Q_{\mathrm{exact}}(\Delta W)$ & Failure Mode | Storage per Layer |
|---|---|---|---|
| **Isotropic $L_2$** | $\frac{\lambda}{2}\|\Delta W\|_F^2$ | Distortion ratio equals full Fisher condition number $\kappa(F_\ell) \gg 10^6$; blind to text vs EM directions. | $O(1)$ |
| **Diagonal EWC** | $\frac{1}{2}\sum_{i,j}\mathbb{E}[g_i^2 x_j^2]\,\Delta W_{ij}^2$ | **Not rotationally invariant.** Underestimates correlated-subspace forgetting by up to factor **$d_{\mathrm{in}}$** (`2560×–5120×` in Qwen!) while falsely penalizing true zero-KL nullspace directions (`1/d` floor). | $O(d_{\mathrm{in}} d_{\mathrm{out}})$ |
| **K-FAC Kronecker Fisher** | $Q_{\mathrm{KFAC}}(\Delta W) = \frac{1}{2}\mathrm{tr}(\Delta W^\top G_{\mathrm{gen}} \Delta W C_{\mathrm{gen}})$ | Exact whenever $(x, g)$ satisfy fourth-order decoupling; for rank-$r$ LoRA $\Delta W = s B A_{\mathrm{eff}}$, evaluates in exact $O(d^2 r)$ via $\frac{s^2}{2}\sum_{i,j}(B^\top G_{\mathrm{gen}} B)_{ij}(A_{\mathrm{eff}} C_{\mathrm{gen}} A_{\mathrm{eff}}^\top)_{ij}$. | $O(d_{\mathrm{in}}^2 + d_{\mathrm{out}}^2)$ |
| **Input Nullspace Projection** | $\Delta W = B \tilde{A} P_{\perp, C}$ | Directly annihilates top-$k$ activation pathways: $\mathbb{E}[\|\Delta W x\|_2^2] \le \sigma_{k+1}\|\Delta W\|_F^2$. Bounds **both** $Q_{\mathrm{KFAC}}$, non-Kronecker $Q_{\mathrm{exact}}$, and all higher-order terms $R_3, R_4, \dots$ simultaneously! | $O(k d_{\mathrm{in}})$ via low-rank $U_k$ |

**Key Structural Guarantees (Theorem 2):**
1. **Diagonal EWC Rotation-Non-Invariance Counterexample**: For $C_{\mathrm{gen}} = \sigma^2 u u^\top$ ($u = \frac{1}{\sqrt{d}}\mathbf{1}$) and $G_{\mathrm{gen}} = I$, aligned perturbation $\Delta W_\parallel = e_1 u^\top$ yields $Q_{\mathrm{exact}}/Q_{\mathrm{diag}} = d$ (`32.0×` in proxy, `2560×–5120×` in Qwen), while nullspace perturbation $\Delta W_\perp = e_1 w^\top$ ($w \perp u$) gives $Q_{\mathrm{exact}} = 0$ yet $Q_{\mathrm{diag}} = \frac{\sigma^2}{2d} > 0$.
2. **Zero-Independence Source-Annihilation Bound**: For any $\Delta W = B \tilde{A} P_{\perp, C}$, $\mathbb{E}[\|\Delta W x\|_2^2] = \mathrm{tr}(\Delta W C_{\mathrm{gen}}\Delta W^\top) \le \sigma_{k+1}\|\Delta W\|_F^2$, so by Cauchy–Schwarz $Q_{\mathrm{exact}}(\Delta W) \le \frac{1}{2}(\sup_x \|g(x)\|_2^2)\,\sigma_{k+1}\|\Delta W\|_F^2$ with zero Kronecker-independence assumption; when $\sigma_{k+1} = 0$, all higher-order terms $R_3, R_4, \dots$ vanish identically.
3. **Asymmetric Input-Nullspace / Output-Unrestricted Readout Lemma**: Right-sided projection $\Delta W = B \tilde{A} P_{\perp, C}$ (leaving $B$ unrestricted) protects general-text inputs while keeping late-layer readout full-rank along M6's candidate Jacobian $M_{\mathrm{cand}} = \Pi C_{\mathrm{cand}} E_{\mathrm{vocab}} D_{\mathrm{answer}} B$.

---

## 2. Modality-Gated Zero-Interference Weight Deltas & Theorem 4.3 Two-Tier Token-Position Gating

### Theorem 3 — Bit-Exact Zero-Regression Invariant (Off-Task General Text & Vision)

Define the out-of-band structural modality gate $g_{\mathrm{EM}}(x) := \mathbb{I}[\text{authorized EM payload attached in context } x] \in \{0, 1\}$ (tied to `qwen_connector.py`'s structured tensor payload, never regex text matching). Implementing every adapted projection as a conditional skip branch ($h_{\mathrm{out}} = W_0 x$ when $g_{\mathrm{EM}} = 0$; $W_0 x + \Delta W x$ when $g_{\mathrm{EM}} = 1$) invokes the exact unmodified base kernels on unmodified $W_0$ whenever $g_{\mathrm{EM}} = 0$, guaranteeing:
$$
\max_{x \in \mathcal{D}_{\mathrm{gen}}} \|z_{W_0 + \Delta W_{\mathrm{eff}}}(x) - z_{W_0}(x)\|_\infty = 0.000\times 10^0, \qquad
D_{\mathrm{KL}}(P_{W_0} \| P_{W_0+\Delta W_{\mathrm{eff}}}) = 0.000\times 10^0,
$$
and bit-identical `host_sha256`. $\square$

### Theorem 4 — The Mixed Multi-Turn Hard Case & Theorem 4.3 Two-Tier Token-Position Gate (`m_EM(t)`)

In multi-turn EM conversations (`g_EM(x) = 1` across all post-EM prose tokens $t > t_{\mathrm{EM}}$), decompose post-EM prose input as $x_t = x_t^{\mathrm{ling}} + x_t^{\mathrm{EM}}$:

1. **Failure of Modality-Gating Alone**: Unconstrained $\Delta W$ under $g_{\mathrm{EM}} = 1$ perturbs $x_t^{\mathrm{ling}}$ on every prose token, exploding mixed-turn drift to `MixKL = 3.687e-1` nats (`+0.4573` NLL drift) despite `PureKL = 0.000e+00` (Section 6.2).
2. **Round-1 Sequence-Gated Hybrid (`g_EM = 1` + $P_{\perp, C}$ + K-FAC + `MixKL` Anchor)**: Shrinks direct linguistic leakage by $\sigma_{k+1}/\sigma_1$, dropping `MixKL` to `5.052e-3` nats (`212×` below unconstrained), but leaves residual prose drift above `1.0e-3` nats because $\Delta W_K, \Delta W_V, \Delta W_\beta, \Delta W_{\mathrm{FFN}}$ still fire on prose positions $t \notin \mathcal{I}_{\mathrm{EM}}$.
3. **Theorem 4.3 — Two-Tier Token-Position Gate (`m_EM(t)`) + Path-2 Cross-Attention Damping (Round 2 Verified!)**:
   Split adapted projections inside every selected layer under $g_{\mathrm{EM}}(x) = 1$ using the causal per-token EM indicator $m_{\mathrm{EM}}(t) := \mathbb{I}[t \in \mathcal{I}_{\mathrm{EM}}] \in \{0, 1\}$:
   - **Tier 1 (Token-Position Gated by $m_{\mathrm{EM}}(t)$ on $K, V, \beta, \mathrm{FFN}$)**:
     $$
     \Delta h_t^{\text{Tier-1}} = m_{\mathrm{EM}}(t)\,\Delta W\, x_t.
     $$
     On every prose token ($m_{\mathrm{EM}}(t) = 0$), $\Delta K_t = \Delta V_t = \Delta \beta_t = \Delta\mathrm{FFN}_t \equiv 0$ **identically**, while EM tokens ($m_{\mathrm{EM}}(t) = 1$) retain full unconstrained capacity (no input $P_{\perp, C}$ bottleneck!). To suppress *Path-2 cross-attention leakage* (prose queries $Q_t$ attending to modified EM keys/values $K_s, V_s$, $s \in \mathcal{I}_{\mathrm{EM}}$), apply light output-only prose curvature damping $\frac{\lambda_{\mathrm{out}}}{2}\mathrm{tr}(\Delta W^\top G_{\mathrm{gen}} \Delta W)$ on Tier-1 $K/V$ plus the ramped mixed-turn KL anchor $\lambda_{\mathrm{mix}}(t)\,\mathcal{L}_{\mathrm{KL\text{-}mix}}$.
   - **Tier 2 (Sequence-Gated $g_{\mathrm{EM}} = 1$ + Input Nullspace $P_{\perp, C}$ + K-FAC on $Q, O, \alpha$)**:
     Allows post-EM answer query and reasoning tokens to read EM evidence while attenuating prose token perturbations by $\sigma_{k+1}/\sigma_1$.
   - **Round-2 Empirical Closure**: Slashes `MixKL` from `5.052e-3` down to **`1.191e-3`** standalone (`900×` cut vs unconstrained, `G1 = 60.16%`, `Q2_gain = +13.02 pp`) and **`8.984e-4` nats (`<= 1.0e-3` target!, `1,193×` cut vs unconstrained)** under M1/W2 Quotient Wrapping (`G1 = 61.81%`, `Q2_gain = +7.64 pp`, `PureKL = 0.000e+00`).

---

## 3. Generalized-Eigenvalue Layer Selection & Exact `|V| = 248,320` Scaling Ledger (`4B → 9B → 27B`)

### Theorem 5 — Generalized Rayleigh Quotient & Exact Kronecker Factorization

Maximizing local per-example EM Fisher curvature gain per unit general-domain KL forgetting yields the **Generalized Rayleigh Quotient** $\Lambda_{\max}^{(\ell)} := \lambda_{\max}((F_{\mathrm{gen}}^{(\ell)} + \tau I)^{-1/2} F_{\mathrm{EM}}^{(\ell)} (F_{\mathrm{gen}}^{(\ell)} + \tau I)^{-1/2})$. Under SPD K-FAC factors $\tilde{C}_{\mathrm{gen}}^{(\ell)} = C_{\mathrm{gen}}^{(\ell)} + \tau_C I$, $\tilde{G}_{\mathrm{gen}}^{(\ell)} = G_{\mathrm{gen}}^{(\ell)} + \tau_G I$:
$$
\Lambda_{\max}^{(\ell)}
= \lambda_{\max}\!\left((\tilde{C}_{\mathrm{gen}}^{(\ell)})^{-1/2} C_{\mathrm{EM}}^{(\ell)} (\tilde{C}_{\mathrm{gen}}^{(\ell)})^{-1/2}\right)
\cdot
\lambda_{\max}\!\left((\tilde{G}_{\mathrm{gen}}^{(\ell)})^{-1/2} G_{\mathrm{EM}}^{(\ell)} (\tilde{G}_{\mathrm{gen}}^{(\ell)})^{-1/2}\right),
$$
reducing eigensolve cost from $O((d_{\mathrm{in}} d_{\mathrm{out}})^3)$ to $O(d_{\mathrm{in}}^3 + d_{\mathrm{out}}^3)$ (numerically verified to **`1.59e-12`** spectral error in `verify_w3.py`). $\square$

### Round-2 Mandatory Architecture Correction (`|V| = 248,320`, Exact 1:1 Integer Parity with `W4` & `smoke_audit.json`)

> [!IMPORTANT]
> **Correction from Round 1 (`151,936 → 248,320`, `1.6344×` larger vocabulary).** Live checkpoint telemetry audited by `W4` against `smoke_audit.json` (`4,205,751,296` exact parameters, `0.0000%` error) and official Hugging Face configs verified that **`Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.8-27B` ALL have `vocab_size = 248,320`**, `full_attention_interval = 4` (3:1 GDN:FA), `attn_head_dim = 256`, `lin_k_head_dim = lin_v_head_dim = 128`, `lin_conv_kernel = 4`, with **`tie_word_embeddings = True` on 4B** and **`False` (untied `lm_head`) on 9B and 27B**. Because $|V| = 248,320$ inflates `lm_head` logit Fisher and backward softmax activation memory by **`1.6344×`** (`lm_head` alone holds `635.7M` / `1.017B` / `1.271B` parameters, so even a single $r=8$ LoRA on `lm_head` would add `2.01M–2.03M` parameters!), freezing `embed_tokens` and `lm_head` is strictly mandatory across all three scales — even on untied 9B and 27B!

| Host Model (Pinned HF Rev) | Exact Host Params (`|V|=248,320`, BF16 MiB) | `tie_emb` / `lm_head` Params (`r=8` avoided) | Depth $L$ (DN/FA) · Width $d$ | Selected 1-Based Layers (`r=8`) | W3 Standalone LoRA / Unified `W1+W2+W3` Host (`M1+W123` Cumul.) | Param Cut vs All-Layer Attn | LoRA+Opt / Low-Rank $U_k$ MiB |
|---|---:|---|---|---|---:|---:|---:|
| **`Qwen3.5-4B`** (`851bf6e8`) | **`4,205,751,296`** (`8,021.83`) | `True` / `635.70M` (`2.007M`) | `32` (`24/8`) · `2560` | **FA**: `12,16,20,24,28` · **DN**: `11,15,19,23` | `1,623,040` (`0.039%`) / **`998,478`** (`1,486,878`) | **3.34×** | `27.86` / `39.38` |
| **`Qwen3.5-9B`** (`c2022362`) | **`8,953,803,264`** (`17,078.02`) | `False` / `1,017.12M` (`2.019M`) | `32` (`24/8`) · `4096` | **FA**: `12,16,20,24,28` · **DN**: `11,15,19,23` | `2,016,256` (`0.023%`) / **`1,334,862`** (`2,024,478`) | **3.32×** | `34.61` / `100.76` |
| **`Qwen3.8-27B`** (`37561816`) | **`26,895,998,464`** (`51,300.05`) | `False` / `1,271.40M` (`2.028M`) | `64` (`48/16`) · `5120` | **FA**: `20,24,28,32,36,40,48,56` · **DN**: `19,23,27,31,35,39` | `4,147,456` (`0.015%`) / **`2,553,870`** (`3,377,630`) | **4.21×** | `71.20` / `245.00` |

*(Note: All three exact host parameter counts match `W4/results_w4.json` to **0 integer discrepancy**; `W3 Standalone LoRA` reflects full rank-8 Q/K/V/O + GDN adapters on the selected layers, while `Unified W1+W2+W3` uses W2's `2×`-compressed Complex-Cayley `O_proj` and Schur-irrep `V_proj`, keeping 4B & 9B cumulative `M1+W123` params strictly below M6's `2.185M` Tier-P1 cap!)*

---

## 4. Cross-Track Theoretical Resolutions (Round 2 Theorems 8 & 9)

### Theorem 8 — Joint Resolution of Tension A (`W1` Early-GDN Slow-Head Starvation) & Tension B (`W4` Stop-Grad Backward VRAM Illusion)

#### Part A (Tension B — Honest Accounting of `Option B1` vs `Option B2`, Synchronized with `W4 Theorem 1`)
Freezing early layers `1..L_stop` (`L_stop = 10` on 4B/9B, `18` on 27B) **does NOT save backward activations or step time** if the Layer-0 EM connector $E_\theta(x)$ remains trainable in Phase B (`Option B2`, co-tuned at `0.15 η_A`), because chain-rule backpropagation to Layer 0 must retain activations and traverse Jacobians through all frozen layers `L_stop .. 1` (**`0.0 MiB` saved**). The `31.25%` (4B/9B) / `28.12%` (27B) activation-memory reduction and `17.8%–19.8%` step-time speedup are realized **if and only if $E_{\theta_A}$ is strictly frozen after Phase A (`Option B1: Connector-Frozen Stop-Grad`)**, detaching hidden states `h_{L_stop}.detach()` after layer `L_stop` (verified to `stop_grad_cache_parity_err = 0.0` in `verify_w3.py`).

| Host Model & `W4` Parity Config | `L_stop` (Frozen Depth) | **Option B1: Connector-Frozen `stop_grad` (Default)** Peak VRAM / Saved | **Option B2: Co-Tuned Connector (`0.15 η_A`)** Peak VRAM / Saved | B1 Step-Time Reduction | Early-GDN Compatibility |
|---|---:|---:|---:|---:|---|
| **`Qwen3.5-4B`** (BF16 `kv_train mb=16`, Pattern-A `sub=8`) | `10` (`31.25%`) | **`40,988.51 MiB`** (**`-12,110.52 MiB`**) | `53,099.03 MiB` (`0.0 MiB`) | **`-19.82%`** | `0`-param W1+W3 slow-head highway preserves `stop_grad`! |
| **`Qwen3.5-9B`** (BF16 `kv_train mb=16`, Pattern-A `sub=8`) | `10` (`31.25%`) | **`54,708.32 MiB`** (**`-14,176.36 MiB`**) | `68,884.68 MiB` (`0.0 MiB`) | **`-19.82%`** | `0`-param W1+W3 slow-head highway preserves `stop_grad`! |
| **`Qwen3.8-27B`** (Mixed MLP-NF4 + FP32 Cayley + ckpt `mb=16`) | `18` (`28.12%`) | **`46,353.92 MiB`** (**`-3,862.60 MiB`**) | `50,216.52 MiB` (`0.0 MiB`) | **`-17.84%`** | `0`-param W1+W3 slow-head highway preserves `stop_grad`! |

**Decision Rule**: Default Phase B to **Option B1 (`Connector-Frozen Stop-Grad`)** — saving `11.83 GiB` on 4B and `13.84 GiB` on 9B (`mb=16`) while keeping 9B comfortably inside M6's `65 GiB` ceiling (`54.71 GiB` vs `68.88 GiB` under B2!). Unlock **Option B2** (or W4's Mode-2 early DN exception on layers `3, 7`) *only* if Phase-A + Option-B1 diagnostics show residual Layer-0 feature drift.

#### Part B (Tension A — Resolving Early-GDN Slow-Head Starvation Without Violating `MixKL <= 1e-3` or `Option B1`)
In frozen early GDN layers `1..L_stop` (`[1,2,3,5,6,7,9,10]` on 4B/9B; `[0,1,2]` in our 8-layer proxy), pretrained text decay $\alpha_{\mathrm{prose}} < 1$ causes exponential state decay across intervening prose (`L_gap = 32` early retention drops to `7.0e-6` in Section 6.3!). However, our Round-2 ablation proves a critical **Double Interference Trap** if W1's `dt_bias` warm-start is applied naively:
1. **Leaky-Beta State-Norm Trap**: Shifting $\alpha_{\mathrm{prose}} \to 0.995$ without strictly zeroing $\beta_{\mathrm{prose}}$ lets intervening prose tokens accumulate without decay inside $S_t^{(h_s)}$, exploding `MixKL` **`90×`** (`1.400e-3 → 1.256e-1` nats) and crashing `G1` by **`-9.0 pp`** (`59.77% → 50.78%`).
2. **Prose Read-Back Trap**: Even when $\beta_{\mathrm{prose}} = 0$ strictly, hijacking a pretrained prose head $h_s$ in frozen early layers deletes that head's prose context and lets unmodified prose queries $q_t^{(h_s)}$ read frozen EM state $S_{\mathrm{EM}}$, still incurring $O(10^{-1})$ `MixKL`.
3. **Exact Zero-Parameter Solution — W1+W3 Dual-Track Nullspace-Filtered Slow-Head Highway**:
   Select the **lowest Task-A Fisher energy** GDN head(s) $h_s = \arg\min_h \mathrm{tr}(F_{\mathrm{gen}, O}^{(h)})$ (`2` of `32` value heads on 4B/9B; `2` of `48` on 27B; `1` of `4` in proxy) and augment the untouched base recurrence with a **zero-parameter shadow slow-EM track** active solely when $g_{\mathrm{EM}}(x) = 1$:
   $$
   o_t^{(h_s)} = o_t^{\mathrm{base},(h_s)} + g_{\mathrm{EM}}(x)\,\gamma_{\mathrm{res}}\,\Big[S_t^{\mathrm{EM},(h_s)}\,\big(W_Q^{(h_s)} P_{\perp, C} h_t\big)\Big],
   $$
   where $S_t^{\mathrm{EM},(h_s)}$ reuses the exact frozen $(W_K^{(h_s)}, W_V^{(h_s)}, W_Q^{(h_s)})$ weights (`0` extra trainable parameters $\implies$ **100% compatible with Option B1 `stop_grad`!**) with W1's slow prose decay $\alpha_{\mathrm{slow}} = \sigma(a_t + \Delta_{\mathrm{warm}})$ ($\Delta_{\mathrm{warm}} = 5.5$) and W3's strict Tier-1 mask $\beta_t^{\mathrm{EM}} = m_{\mathrm{EM}}(t)\,\beta_t$ (**zero prose writes**).
   - **Verified Properties (Section 6.3)**: Off-task `PureKL = 0.000e+00`; on `L_gap = 32`, early GDN retention jumps **`77,919×`** (`7e-6 → 0.779193`), mid GDN retention jumps **`>100,000×`** (`4e-6 → 0.544170`), and `MixKL` drops **`107×`** vs `leaky_beta` to **`1.172e-3` nats** (even *lower* than `frozen_early_no_warmgate`'s `1.400e-3`!).
   - **Honest 3:1 Hybrid Nuance**: Because Qwen's 3:1 architecture inserts Full-Attention anchors every 4th layer (Layer 3 in proxy; Layers `4, 8, 12` in Qwen), at moderate gaps (`L_gap <= 32`) the residual stream at `t ∈ {0..3}` passes causally into Layer-3 Full-Attention with $O(1)$ distance, allowing `frozen_early_no_warmgate` to retain `G1 = 59.77%` (though losing `Q2` fresh-pilot contrast at `-3.12 pp`). The zero-param `clean_dual_nullspace` highway is therefore deployed cleanly for long-context (`L_gap >= 64..128`) multi-turn GDN regimes without touching `Option B1` `stop_grad` or blowing up `MixKL`. $\square$

---

### Theorem 9 — `W2` Symmetry Compatibility: Complex-Symmetrized Nullspace $P_{\perp, C}^{\mathbb{C}}$, Nullspace-Restricted Complex-Cayley $Q^\perp$, and Quotient $\lambda_{\mathrm{orbit}} = 0$

Real PCA nullspace projection $P_{\perp, C}$ generically breaks the $SO(2)$ complex-structure commutation $[P_{\perp, C}, J_d] = 0$ required by `W2`'s Complex-Cayley orthogonal adapters (`||[P_{\perp, C}, J_d]||_F = 3.376` in Section 6.1!), where $J_d = I_{d/2} \otimes \begin{psmallmatrix}0 & -1\\ 1 & 0\end{psmallmatrix}$ ($J_d^\top = -J_d$, $J_d^2 = -I_d$, $d = 2m$).

1. **Complex-Symmetrized Activation Covariance & Exact Commutation.** Define the $J_d$-symmetrized activation covariance:
   $$
   C_{\mathrm{gen}}^{\mathbb{C}} := \frac{1}{2}\big(C_{\mathrm{gen}} + J_d^\top C_{\mathrm{gen}} J_d\big) \succeq 0.
   $$
   Then $[C_{\mathrm{gen}}^{\mathbb{C}}, J_d] = 0$ identically, so $C_{\mathrm{gen}}^{\mathbb{C}}$ is isomorphic to an $m \times m$ Hermitian PSD matrix $H_{\mathrm{gen}} = A + i B \succeq 0$ ($A = (C_{\mathrm{gen}}^{\mathbb{C}})_{0::2, 0::2}$, $B = (C_{\mathrm{gen}}^{\mathbb{C}})_{1::2, 0::2}$). Let $U_c \in \mathbb{C}^{m \times k_c}$ contain the top-$k_c$ eigenvectors of $H_{\mathrm{gen}}$, set $P_C^{\mathrm{Herm}} := I_m - U_c U_c^*$, and lift back to $\mathbb{R}^{d \times d}$ as $P_{\perp, C}^{\mathbb{C}}$. Then:
   - **Exact $SO(2)$ Doublet Pairing**: $[P_{\perp, C}^{\mathbb{C}}, J_d] = 0$ identically (verified `||[P_{\perp, C}^{\mathbb{C}}, J_d]||_F = 0.000e+00 < 10^{-12}`!).
   - **Exact Factor-2 Loewner Spectral Bound**: Because $C_{\mathrm{gen}} \succeq 0 \implies J_d^\top C_{\mathrm{gen}} J_d \succeq 0$,
     $$
     C_{\mathrm{gen}} \preceq C_{\mathrm{gen}} + J_d^\top C_{\mathrm{gen}} J_d = 2\,C_{\mathrm{gen}}^{\mathbb{C}}
     \quad\Longleftrightarrow\quad
     \lambda_{\max}\!\big(C_{\mathrm{gen}} - 2\,C_{\mathrm{gen}}^{\mathbb{C}}\big) \le 0
     $$
     (verified $\lambda_{\max} = -7.01\times 10^{-4} \le 0$). Consequently, for every complex-nullspace-projected perturbation $\Delta W = \Delta\tilde{W}\, P_{\perp, C}^{\mathbb{C}}$,
     $$
     \mathbb{E}_{x\sim\mathcal{D}_{\mathrm{gen}}}\!\big[\|\Delta W x\|_2^2\big]
     = \mathrm{tr}\!\big(\Delta W C_{\mathrm{gen}} \Delta W^\top\big)
     \le 2\,\mathrm{tr}\!\big(\Delta W C_{\mathrm{gen}}^{\mathbb{C}} \Delta W^\top\big)
     \le 2\,\lambda_{k_c + 1}(H_{\mathrm{gen}})\,\|\Delta W\|_F^2.
     $$
     Thus enforcing exact $SO(2)$ compatibility costs at most a universal factor of `2` in tail spectral leakage — still yielding **`860.11×` exact KL suppression** (`> 100×` by `8.6×`!) in Section 6.1.

2. **Nullspace-Restricted Complex-Cayley Woodbury Adapter ($Q^\perp$, Exact `W2 + W3` Fusion).** Parameterize complex low-rank generators $U, V \in \mathbb{C}^{m \times r_c}$ strictly inside the Hermitian nullspace via $U_\perp = P_C^{\mathrm{Herm}} U$, $V_\perp = P_C^{\mathrm{Herm}} V$, form $A_c = [U_\perp, -V_\perp]$, $B_c = [V_\perp, U_\perp] \in \mathbb{C}^{m \times 2r_c}$ (so $S_\perp = A_c B_c^* = U_\perp V_\perp^* - V_\perp U_\perp^*$ is strictly skew-Hermitian with $S_\perp U_c = 0$), and apply `W2`'s $2r_c \times 2r_c$ Woodbury Cayley transform:
   $$
   Q_c^\perp := \left(I_m - \frac{1}{2} S_\perp\right)\left(I_m + \frac{1}{2} S_\perp\right)^{-1}
   = I_m - A_c\!\left(I_{2r_c} + \frac{1}{2} B_c^* A_c\right)^{-1} B_c^*.
   $$
   Lifting $Q_c^\perp$ to real $Q^\perp \in \mathbb{R}^{d \times d}$ and setting $W_{\mathrm{eff}} = W_0 Q^\perp$ ($\Delta W = W_0 (Q^\perp - I_d)$) guarantees **all four invariants simultaneously**:
   1. Exact orthogonality: $(Q^\perp)^\top Q^\perp = I_d$ (verified `1.58e-15`), preserving `W2`'s unit gradient norm across depth 32–64!
   2. Exact $SO(2)$ commutation: $[Q^\perp, J_d] = 0$ (verified `0.000e+00`)!
   3. Exact identity on the dominant general-text subspace: because $B_c^* U_c = 0$, $(Q^\perp - I_d)(I_d - P_{\perp, C}^{\mathbb{C}}) = 0$ identically (verified **`2.88e-15 < 10^{-12}`**!), yielding **`729.83×`** Task-A KL reduction at matched Frobenius norm!
   4. Strictly FP32 (`W4` proved NF4 quantization inflates Cayley orthogonality defect `20,162×`; FP32 `8×8` Woodbury adds only `0.0049%` FLOPs on 4B).

3. **Quotient Wrapper Synergy ($\lambda_{\mathrm{orbit}} = 0$).** When wrapped in `M1/W2`'s exact $C_4$ Quotient Canonicalizer (canonicalizing $t_1$ to sector $0$ before host forward and unrolling output probabilities by $a_1$), outer $C_4$ equivariance holds identically (`C4_max = 0.000e+00`), making the explicit orbit-consistency loss redundant. Setting **$\lambda_{\mathrm{orbit}} = 0$** removes the `rot1` counterfactual pass per step, speeds up training (`5.77 s` vs `5.91 s/seed`), and *improves* generalization (`G1 = 61.81%` vs `59.81%`, `G1–G4 = 55.12%` vs `53.15%`, `MixKL = 8.984e-4 <= 1.0e-3` nats!). $\square$

---

## 5. Objective, Distillation & Curriculum Optimization (Theorems 6 & 7)

### Theorem 6 — Exact Rao–Blackwell Gradient Variance Reduction & Calibration Optimality of Physical Posterior KL Distillation

Comparing noisy hard-label CE $\ell_{\mathrm{CE}}(\theta; X, Y) = -\log q_\theta(Y \mid X)$ against physical posterior KL distillation $\ell_{\mathrm{KL}}(\theta; X) = D_{\mathrm{KL}}(p^*(X) \| q_\theta(X))$:
1. **Identical Population Minimizer**: $\mathbb{E}_{Y\mid X}[\ell_{\mathrm{CE}}] = H(p^*(X)) + \ell_{\mathrm{KL}}$ and $\mathbb{E}_{Y\mid X}[g_{\mathrm{CE}}] = g_{\mathrm{KL}}$.
2. **Exact Pythagorean Rao–Blackwell Decomposition**:
   $$
   \operatorname{Cov}(g_{\mathrm{CE}}) = \operatorname{Cov}(g_{\mathrm{KL}}) + \mathbb{E}_X\!\big[J_f(\theta; X)^\top (\mathrm{diag}(p^*(X)) - p^*(X)p^*(X)^\top) J_f(\theta; X)\big].
   $$
3. **Zero-Variance Fixed Point**: At $q_{\theta^*} = p^*$, $\operatorname{Cov}(g_{\mathrm{KL}}(\theta^*)) = 0$ identically, whereas hard-label CE retains an $O(1)$ irreducible Bayes Brier noise floor ($\approx 0.5098$ at Stage 3D's 2 dB `G1` regime), verified at **`5.693×` (Step 16)** and **`16.935×` (Step 160, 4.38% Pythagorean residual)** in Round 2 (and `113.732×` at Step 240 in Round 1). On real SDR captures (Pluto/CTTC), anchor a causal off-fold physical teacher with $(1-\alpha)\ell_{\mathrm{KL}} + \alpha \ell_{\mathrm{CE}}$ ($\alpha \in [0.1, 0.2]$). $\square$

### Theorem 7 — Composite Calibrated, Equivariant & Counterfactual-Slot Objective (`λ_orbit = 0` Under Quotient Wrapper)

$$
\mathcal{L}_{\mathrm{W3}}(\theta)
= \mathbb{E}_X\big[D_{\mathrm{KL}}(p_{\mathrm{phys}}(X) \| q_\theta(X))\big]
+ \lambda_{\mathrm{norm}}\,\mathbb{E}_X\Big[\big(\|\Pi s_\theta(X)\|_2 - \|\Pi\log p_{\mathrm{phys}}(X)\|_2\big)^2\Big]
+ \lambda_{\mathrm{cf}}\,\mathcal{L}_{\mathrm{cf\text{-}margin}}
+ \lambda_{\mathrm{prose}}\,\mathcal{L}_{\mathrm{prose\text{-}inv}}
+ \lambda_{\mathrm{orb}}\,\mathcal{L}_{\mathrm{orbit}}
+ \lambda_F\,\mathcal{R}_{\mathrm{KFAC}}
+ \lambda_{\mathrm{mix}}(t)\,\mathcal{L}_{\mathrm{KL\text{-}mix}},
$$
with **$\lambda_{\mathrm{orb}} = 0$** whenever M1/W2's Quotient Wrapper is active (`C4_max = 0.000e+00`), and linearly ramped mixed-turn KL anchor $\lambda_{\mathrm{mix}}(t)$ (`0.25 → 1.0`) so early steps preserve fast EM gradient signal while late cosine-annealed steps pin `MixKL <= 1.0e-3` nats.

---

## 6. Round-2 CPU PyTorch Verification (`verify_w3.py` → `results_w3.json`)

Executed cleanly via:
```powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W3/verify_w3.py
```
- **Runtime & Integrity**: **`132.33 s`** (`<= 160 s` mandate; **`7.16×` speedup** over Round 1's `947.39 s`), exit code `0`, `torch 2.12.1+cpu`, `numpy 2.3.5`, `scipy 1.17.0`, `stop_grad_cache_parity_err = 0.0`.
- **Host & Two-Phase Setup**: 8-layer exact 3:1 Hybrid (`[DN, DN, DN, FA, DN, DN, DN, FA]`, `d_model = 32` = `16` complex doublets, `n_heads = 4`, `d_ff = 64`, `V = 64`, tied `lm_head`, exact parallel WY Gated-DeltaNet). Pretrained 140 steps on Task A (`NLL = 2.9887` vs uniform `3.8712`, top-12 lexical embedding energy `98.84%`, `host_sha256 = 73d19809c5f809bb...`), followed by authentic **Phase-A Connector Alignment** (`45` steps on the 100% frozen host, then strictly frozen for Option B1 Phase B: `G1 = 24.22% / 1.4111`, `Q2_valid = 23.96%`, `Q2_fresh = 18.23%`, reproducing Stage 3D's frozen-host wall; Exact Bayes Oracle: `G1 = 63.80% / 0.9502`, `G3 = 68.23% / 0.8528`, `Q2_valid = 60.94%`, `Q2_fresh = 71.88%`, gain `+10.94 pp`).

### 6.1 Block 1 & Block 2 — Self-Concordance, 5-Way Proxy Distortion, Complex-Symmetrized $P_{\perp, C}^{\mathbb{C}}$ & Cayley $Q^\perp$

1. **Kronecker Spectral Identity (Theorem 5)**: `|lambda_full - lambda_C * lambda_G| = 1.59e-12` (`< 1e-11`).
2. **Bach Self-Concordance Bracket (Theorem 1)**: Holds **`5/5`** across $\|\Delta W\|_F \in [0.01, 0.60]$ (at `0.10`, $\gamma = 0.1232$: exact KL `1.9091e-4`, Fisher quad `1.9080e-4`, **`0.0614%` rel error**; at `0.30`, $\gamma = 0.3696$: exact KL `1.7203e-3`, Fisher quad `1.7172e-3`, **`0.1804%` rel error**).
3. **5-Way Geometry & `W2` Complex-Symmetrized Nullspace Audit at Matched $\|\Delta W\|_F = 0.35$ (`layer_0.q_proj`)**:

| Perturbation Direction ($\|\Delta W\|_F = 0.35$) | Exact Task-A KL Drift | KL Cut vs $u_1$ | K-FAC Proxy | Diag-EWC Proxy | $SO(2)$ Comm. Defect $\|[M, J_d]\|_F$ |
|---|---:|---:|---:|---:|---:|
| Top general-text eigenvector $u_1$ | `9.279e-4` | `1.00×` | `6.221e-4` | `3.178e-4` (`2.92×` under) | — |
| Random rank-4 perturbation | `1.722e-4` | `5.39×` | `1.251e-4` | `9.290e-5` | — |
| Real PCA Nullspace $P_{\perp, C}$ (rank-4) | `4.290e-7` | **`2,162.91×`** | `3.321e-7` | `3.113e-6` (`7.26×` false) | **`3.376258`** (breaks `W2`!) |
| **Complex-Symmetrized $P_{\perp, C}^{\mathbb{C}}$ (rank-4, Thm 9)** | **`1.079e-6`** | **`860.11×`** (`>100×`!) | `8.351e-7` | `2.744e-6` | **`0.000e+00`** (exact!) |
| **Complex-Cayley Nullspace $Q^\perp$ (`W2+W3`, Thm 9)** | **`1.271e-6`** | **`729.83×`** (`>100×`!) | `9.285e-7` | `3.947e-6` | **`0.000e+00`** (orth `1.58e-15`, top-id `2.88e-15`!) |

- **Exact Loewner Spectral Verification**: $\lambda_{\max}(C_{\mathrm{gen}} - 2 C_{\mathrm{gen}}^{\mathbb{C}}) = -7.013\times 10^{-4} \le 0$.
- **Block 2 Generalized Rayleigh Ranking**: Layer 0 (`9,148.4`, early ingestion, owned by Phase-A connector); post-early ranking `L7 (FA: 227.44, max nullspace 19.33/32) > L6 (DN: 226.25) > L5 (DN: 132.30) > L4 (103.64) > L3 (FA: 85.15) > L1 (81.55) > L2 (76.58)`. Selected post-early hybrid layers: **`[3, 5, 6, 7]`** (`stop_layer = 3`, `14,912` params vs `30,400` all-layer LoRA = `51%` cut).

---

### 6.2 Block 3 (Axis 1) — Anti-Forgetting Architecture Comparison (Seed `202611`, Step 160, Matched `bayes_kl_plus_cf_cal` Loss)

| Method (`host_sha256` preserved: ALL `True`) | Params | Time | Task-A Pure KL (`NLLd` / `max`) | Task-A Mixed KL (`NLLd`) | `G1` Acc / NLL | `G3` Acc / NLL | `Q2` Valid → Fresh (`Gain`) | `Q2` Indep NLL ($\log 4=1.386$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Phase-A Frozen Host ($\Delta W = 0$)* | `0` | — | `0.000e+00` (`0` / `0.00`) | `0.000e+00` (`+0.00000`) | `24.22% / 1.4111` | `24.48% / 1.4003` | `23.96% → 18.23%` (`-5.73 pp`) | `1.4732` |
| *Exact Physical Bayes Oracle* | — | — | — | — | `63.80% / 0.9502` | `68.23% / 0.8528` | `60.94% → 71.88%` (`+10.94 pp`) | `1.3863` |
| **1. Unconstrained LoRA (All 8 layers)** | `30,400` | `7.8s` | `1.138e+00` (`+1.322` / `6.85`) | `1.072e+00` (`+1.24863`) | `59.90% / 1.0142` | `65.10% / 0.9413` | `59.38% → 63.02%` (`+3.65 pp`) | `1.3828` |
| **2. Layer-Selected Gen-Fisher LoRA** | `14,912` | `5.0s` | `2.196e-01` (`+0.296` / `4.84`) | `2.625e-01` (`+0.34144`) | `57.55% / 1.0500` | `62.50% / 0.9727` | `57.29% → 62.50%` (`+5.21 pp`) | `1.4657` |
| **3. Nullspace-Projected LoRA ($P_{\perp, C}$)** | `14,912` | `4.7s` | `1.306e-02` (`+0.012` / `3.38`) | `4.760e-02` (`+0.06582`) | `60.42% / 1.0250` | `64.06% / 0.9275` | `51.56% → 58.33%` (`+6.77 pp`) | `1.4520` |
| **4a. Modality-Gated Unconstrained (Thm 4)** | `14,912` | `4.4s` | **`0.000e+00`** (`0` / **`0.00`**) | `3.687e-01` (`+0.45726`) | `58.59% / 1.0536` | `62.50% / 0.9612` | `54.69% → 64.06%` (`+9.38 pp`) | `1.4516` |
| **4b. Sequence-Gated Nullspace Hybrid (R1)** | `14,912` | `5.6s` | **`0.000e+00`** (`0` / **`0.00`**) | `5.052e-03` (`+0.01023`) | `60.94% / 1.0173` | `66.15% / 0.9169` | `55.73% → 64.58%` (`+8.85 pp`) | `1.4762` |
| **5. Two-Tier Token-KV Gate (R2 Thm 4.3)** | `14,912` | `5.9s` | **`0.000e+00`** (`0` / **`0.00`**) | **`1.191e-03`** (`+0.00303`) | **`60.16% / 1.0427`** | **`71.35% / 0.9281`** | **`52.08% → 65.10%`** (**`+13.02 pp`**) | `1.4480` |
| **5+Q. Two-Tier + Quotient (`λ_orb=0`, 3-sd)** | `14,912` | `5.8s` | **`0.000e+00`** (`0` / **`0.00`**) | **`8.984e-04`** (**`<= 1e-3`!**) | **`61.81% / 1.0167`** | `G1–G4: 55.12%` | **`+7.64 pp`** (`+0.1458` nats) | — |

---

### 6.3 Block 3b — Long-Gap (`L_gap = 32`) `W1` Slow-Head GDN Resolution Study (Tension A, 90 Steps, Seed `202611`)

| Arm (`0` Extra Early Trainable Params, `Option B1` `stop_grad` Preserved) | Early GDN `L=32` Retention | Mid GDN `L=32` Retention | Task-A Pure KL | Task-A Mixed KL (`NLLd`) | `G1` Acc / NLL | `Q2` Fresh−Valid |
|---|---:|---:|---:|---:|---:|---:|
| **1. Frozen Early (No Warmgate)** | `0.000007` | `0.000004` | `0.000e+00` | `1.400e-03` (`+0.00147`) | `59.77% / 1.0420` | `-3.12 pp` |
| **2. Naive Early Warmgate (`leaky_beta`)** | `0.683536` | `0.000000` | `0.000e+00` | **`1.256e-01`** (**`+0.15413`, `90×` spike!**) | **`50.78% / 1.1966`** (`-9.0 pp`!) | `+0.00 pp` (`22.9%` collapsed!) |
| **3. W1+W3 Dual-Track Nullspace Highway (Thm 8B)** | **`0.779193`** (**`77,919×`**) | **`0.544170`** (**`136,042×`**) | **`0.000e+00`** | **`1.172e-03`** (**`107×` cleaner!**) | `53.91% / 1.1213` | `-0.52 pp` (`44.3% / 43.8%`, `+21 pp` vs Arm 2) |

*(Fisher-selected lowest-energy slow heads per GDN layer: `{L0: h0, L1: h3, L2: h2, L4: h3, L5: h0, L6: h0}`. Arm 2 demonstrates the catastrophic `MixKL` & `Q2` collapse caused by leaky prose writes; Arm 3 completely eliminates the `MixKL` spike while restoring `77,919×` early GDN retention with zero extra trainable parameters.)*

---

### 6.4 Block 4 (Axis 2) — Loss Optimization, Step-69 vs Step-96 vs Step-160 Convergence & Quotient `λ_orbit = 0` (3 Seeds: `202611, 202622, 202633`, Fixed Theorem 4.3 Two-Tier Gate)

**Rao–Blackwell Gradient Variance Audit (`seed = 202611`):**
- **Step 16**: $\mathbb{E}[\|g_{\mathrm{CE}}\|_2^2] = 1.0058$ vs $\mathbb{E}[\|g_{\mathrm{KL}}\|_2^2] = 0.1767$ (Bayes noise `0.7980`, Pythagorean residual `3.09%`) $\implies$ **`5.693×` variance reduction**.
- **Step 160**: $\mathbb{E}[\|g_{\mathrm{CE}}\|_2^2] = 5.0423$ vs $\mathbb{E}[\|g_{\mathrm{KL}}\|_2^2] = 0.2978$ (Bayes noise `4.5236`, Pythagorean residual `4.38%`) $\implies$ **`16.935×` variance reduction**.

| Objective Mode (3-Seed Mean ± SD, `PureKL = 0.000e+00` all runs) | Time / Seed | **Step-69 `G1` Acc / NLL** (`Q2g`) | **Step-96 `G1` Acc / NLL** (`MixKL` / `Q2g`) | **Step-160 `G1` Acc / NLL** (`G1–G4`) | **Step-160 ECE / $T^*$** | **Step-160 Work-F Flip / Gain** | **Step-160 $C_4$ Max Defect** | **Step-160 `MixKL` / `Q2g` (NLL)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Hard-Label 1-Hot CE** | `5.36s` | `50.52% ± 5.20 / 1.2248` (`-1.04 pp`) | `53.56% ± 2.35 / 1.1673` (`5.96e-4` / `-0.87 pp`) | `53.47% ± 2.17 / 1.1508` (`50.95%`) | `0.1174` / `0.640` | `+0.779` / `0.451` | `4.35e-1` | `4.129e-4` / `+1.91 pp` (`+0.030`) |
| **B. Exact Bayes-KL Only** | `5.67s` | `55.64% ± 0.99 / 1.1096` (`+0.69 pp`) | `55.47% ± 2.96 / 1.1007` (`5.64e-4` / `+6.25 pp`) | `56.42% ± 3.28 / 1.0895` (`51.30%`) | `0.1106` / `0.687` | `+1.010` / `0.575` | `4.09e-1` | `4.317e-4` / `+6.60 pp` (`+0.124`) |
| **C. Bayes-KL + CF-Cal (`λ_orb=0.10`)** | `5.91s` | `56.34% ± 2.87 / 1.0669` (`+4.86 pp`) | `59.03% ± 2.28 / 1.0553` (`1.46e-3` / `+5.90 pp`) | `59.81% ± 1.08 / 1.0490` (`53.15%`) | **`0.0745`** / **`0.881`** | **`+1.394`** / **`0.738`** | `4.75e-1` | `1.218e-3` / `+7.47 pp` (**`+0.174`**) |
| **D. Bayes-KL + CF-Cal + Quotient (`λ_orb=0`)** | **`5.77s`** | **`59.20% ± 3.79 / 1.0575`** (**`+6.42 pp`**) | **`61.37% ± 1.97 / 1.0239`** (**`9.00e-4`** / **`+6.77 pp`**) | **`61.81% ± 1.28 / 1.0167`** (**`55.12%`**) | `0.0789` / `0.877` | `+1.066` / `0.651` | **`0.000e+00`** | **`8.984e-4`** / **`+7.64 pp`** (`+0.146`) |

**Quantitative Recommendation to Track `W4` on Shortening Phase B (`192 → 69–96` Steps inside Modal `$12.88`):**
1. **Step 69 is a decisive GO/NO-GO pilot gate (NOT yet full convergence)**: Under Mode D (`Quotient + λ_orbit=0`), Step 69 already lifts `G1` by **`+8.68 pp`** over 1-hot CE (`59.20%` vs `50.52%`, NLL `-0.1673` nats) and achieves `+6.42 pp` `Q2` gain, **but** still sits `2.61 pp / +0.0408` nats short of Step 160 (`MixKL = 1.418e-3`).
2. **Step 96 captures `99.3%` of Step-160 convergence AND already clears `MixKL <= 1.0e-3`!** Between Step 96 and Step 160 under Mode D, `G1` moves only `+0.44 pp` (`61.37% → 61.81%`), `G1` NLL moves only `-0.0072` nats (`1.0239 → 1.0167`), and `MixKL` is **already `9.002e-4 <= 1.0e-3` nats at Step 96** (`8.984e-4` at Step 160)!
3. **Converged Schedule Recommendation for `W4`**: Keep **Step 69** as M6's early futility/GO-NO-GO checkpoint, and **safely shorten Phase-B training from `192 → 96` steps** (`50%` compute & dollar savings in Phase B!) — reserving `160–192` steps only if a live checkpoint's Step-96 validation NLL slope exceeds `0.005 nats / 24 steps`.

---

## 7. Reproducibility & Scope Boundaries

- **Formally Guaranteed on Any Host (`Qwen3.5-4B / 9B / 3.8-27B`, `|V| = 248,320`)**: Fisher–Rao KL expansion & Bach self-concordance bracket (Theorem 1), Diagonal-EWC distortion & input-nullspace KL bound (Theorem 2), bit-exact `0.000e+00` off-task zero regression & `host_sha256` preservation (Theorem 3), Kronecker Generalized Rayleigh factorization (Theorem 5), Rao–Blackwell gradient variance reduction (Theorem 6), `Option B1` `stop_grad` VRAM boundary (Theorem 8A), and `Complex-Symmetrized P_{\perp, C}^{\mathbb{C}}` + `Q^\perp` exact $SO(2)$/orthogonality/Loewner bounds (Theorem 9).
- **Verified in CPU 3:1 Hybrid Proxy (`132.33 s <= 160 s`) & Pending Live A100 Measurement on Real Checkpoints**: Empirical activation spectra $\{\sigma_j^{(\ell)}\}$ of `Qwen3.5-4B / 9B / 3.8-27B` on Chinese/multimodal text (`1`-pass K-FAC probe, `< 20 s` on A100), live `L_gap = 64..128` verification of the zero-param `W1+W3` dual-track slow-head highway, and paired M6 `6B/6C` general-capability regression gates.
