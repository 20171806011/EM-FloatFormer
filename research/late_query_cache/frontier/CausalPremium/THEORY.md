# Observational vs. Interventional Causal Cache Premium in Late-Query Caching (Conjecture 3)

**Target Module:** `MathLab.LateQueryFrontier.CausalPremium`  
**Deliverables:** `THEORY.md` (analytic, measure-theoretic, topological, and statistical theory) and `CausalPremium.lean` (Lean 4.36.0 + Mathlib4 v4.36.0 kernel-verified core theorems).

---

## 1. Formal SCM Late-Query Framework

### 1.1 Acyclic Structural Causal Models and Late-Query Families
Let $\mathcal{M} = (V, U, F, P_U)$ be an acyclic Structural Causal Model (SCM) associated with a directed acyclic graph (DAG) $G$ over endogenous variables $V = (V_1, \dots, V_m) \in \prod_{i=1}^m \mathcal{V}_i$, independent or jointly distributed exogenous variables $U = (U_1, \dots, U_m) \sim P_U$, and measurable structural equations
$$V_i = f_i(\mathrm{Pa}_G(V_i), U_i), \qquad i \in \{1, \dots, m\}.$$
Because $G$ is acyclic, the system admits a unique reduced-form solution $V = \Phi_{\mathcal{M}}(U)$, inducing the observational law $P_V = (\Phi_{\mathcal{M}})_\# P_U$. For any intervention $\mathrm{do}(A = a)$ on a treatment block $A \subseteq V$ taking values in $\mathcal{A}$, the mutilated SCM $\mathcal{M}_{A \leftarrow a}$ replaces the structural equations for $A$ with the constant assignment $A := a$, inducing the interventional law $P_{V; \mathrm{do}(A=a)}$ and counterfactual random variables $V_{A \leftarrow a} = \Phi_{\mathcal{M}_{A \leftarrow a}}(U)$ on the same underlying exogenous probability space $(U, P_U)$.

In **late-query caching**, a system observes an initial feature projection
$$X := \Pi_{\mathrm{obs}}(V) \in \mathcal{X} \subseteq \mathbb{R}^n$$
at caching time (with topological support $\mathcal{X}_0 := \mathrm{supp}(P_X) \subseteq \mathbb{R}^n$), stores a compressed representation $C = T(X) \in \mathbb{R}^d$, and later answers downstream queries concerning an outcome $Y \in \mathbb{R}^{d_Y}$ (assumed integrable under both $P$ and $P_{\mathrm{do}(A=a)}$):
- **Observational late-query family** $\mathcal{Q}_{\mathrm{obs}}$:
  $$\mathcal{Q}_{\mathrm{obs}} := \big\{ f^{\mathrm{obs}} : \mathcal{X}_0 \to \mathbb{R}^{d_Y} \big\}, \qquad f^{\mathrm{obs}}(x) := \mathbb{E}_P[Y \mid X = x].$$
  (More generally, a finite or indexed collection $\{f_q^{\mathrm{obs}}\}_{q \in \mathcal{I}_{\mathrm{obs}}}$ of conditional expectations given $X$.)
- **Interventional / counterfactual late-query family** $\mathcal{Q}_{\mathrm{do}}$: parameterized by target interventions $a \in \mathcal{A}$ (ranging over an admissible non-singleton set $\mathcal{A}$).

### 1.2 Exact Distinction: Counterfactual Query vs. Pre-Treatment Interventional Query
A frequent formal pitfall when $A \in X$ (or descendants of $A$ appear in $X$) is writing $\mathbb{E}[Y \mid X = x, \mathrm{do}(A = a)]$ literally. Under the mutilated distribution $P_{\mathcal{M}_{A \leftarrow a}}$, the event $\{A = a_{\mathrm{obs}}\}$ has probability $0$ whenever $a_{\mathrm{obs}} \neq a$, so conditioning simultaneously on $X = x$ and $\mathrm{do}(A = a)$ is measure-theoretically ill-posed unless interpreted as one of two distinct, well-posed objects:

1. **Counterfactual query** (well-defined for arbitrary $X = \Pi_{\mathrm{obs}}(V)$):
   $$f_a^{\mathrm{cf}}(x) := \mathbb{E}\big[Y_{A \leftarrow a} \mid X = x\big] = \int \big(\Phi_{\mathcal{M}_{A \leftarrow a}}(u)\big)_Y \, dP(u \mid X = x).$$
   Here the factual observation $X = x$ updates the exogenous posterior $P(U \mid X = x)$, which is then propagated through the mutilated model $\mathcal{M}_{A \leftarrow a}$.
2. **Pre-treatment interventional query** (well-defined for the non-descendant subvector $X_{\mathrm{pre}} \subseteq V \setminus \mathrm{De}_G(A)$, e.g., $X_{\mathrm{pre}} = Z$ when $X = (A, Z)$):
   $$f_a^{\mathrm{do}}(x_{\mathrm{pre}}) := \mathbb{E}\big[Y \mid X_{\mathrm{pre}} = x_{\mathrm{pre}}, \mathrm{do}(A = a)\big] = \mathbb{E}\big[Y_{A \leftarrow a} \mid X_{\mathrm{pre}} = x_{\mathrm{pre}}\big].$$

**Theorem 1.1 (Coincidence and Divergence of Counterfactual and Pre-Treatment Interventional Queries).**  
Suppose $X = (A, Z)$ where $Z \subseteq V \setminus \mathrm{De}_G(A)$ satisfies the backdoor criterion relative to $(A, Y)$ in $G$, and decompose $Y_{A \leftarrow a} = h(a, Z, U_Y)$ where $U_Y$ collects the remaining exogenous/unobserved factors affecting $Y$.
1. *(Coincidence under conditional unconfoundedness)*: If $U_Y \perp A \mid Z$ (in particular, if $W \perp (A, Z)$ in additive-noise models $Y = \mu(A, Z) + W$), then for $P_X$-almost every $x = (a_{\mathrm{obs}}, z)$ and all $a \in \mathcal{A}$,
   $$f_a^{\mathrm{cf}}(a_{\mathrm{obs}}, z) = f_a^{\mathrm{do}}(z).$$
   *Proof:* Because $Z$ is non-descendant of $A$, $Z_{A \leftarrow a} = Z$. By conditional independence $U_Y \perp A \mid Z$, the conditional law satisfies $P(U_Y \mid A = a_{\mathrm{obs}}, Z = z) = P(U_Y \mid Z = z)$. Integrating $h(a, z, U_Y)$ against both posteriors yields $\mathbb{E}[h(a, z, U_Y) \mid A = a_{\mathrm{obs}}, Z = z] = \mathbb{E}[h(a, z, U_Y) \mid Z = z] = f_a^{\mathrm{do}}(z)$. $\square$
2. *(Divergence under unobserved confounding)*: If an unobserved confounder $U$ influences both $A$ and $Y$ (e.g., $(U, \eta)$ zero-mean jointly Gaussian with $\eta \perp U$, $\mathrm{Var}(U) > 0, \mathrm{Var}(\eta) > 0$, $A = U + \eta$, $Y = \alpha A + \beta U$ with $\beta \neq 0$, $X = A$), then $f_a^{\mathrm{cf}}(a_{\mathrm{obs}}) = \alpha a + \beta \mathbb{E}[U \mid A = a_{\mathrm{obs}}] = \alpha a + \beta \frac{\mathrm{Var}(U)}{\mathrm{Var}(U)+\mathrm{Var}(\eta)} a_{\mathrm{obs}}$ depends nontrivially on $a_{\mathrm{obs}}$, whereas $f_a^{\mathrm{do}}(\emptyset) = \alpha a + \beta \mathbb{E}[U]$ is constant in $a_{\mathrm{obs}}$. $\square$

---

### 1.3 Minimal Sufficient Cache Dimensions and Regularity of Encoders

For any family of measurable functions $\mathcal{Q} = \{f_q : \mathcal{X}_0 \to \mathbb{R}^{k_q}\}_{q \in \mathcal{I}}$ and a prescribed encoder class $\mathcal{T}(\mathcal{X}_0, \mathbb{R}^d)$, define the **minimal sufficient cache dimension**
$$d_{\mathcal{T}}(\mathcal{Q}) := \min \Big\{ d \in \mathbb{N} : \exists T \in \mathcal{T}(\mathcal{X}_0, \mathbb{R}^d),\; \forall q \in \mathcal{I},\; \exists g_q : T(\mathcal{X}_0) \to \mathbb{R}^{k_q},\; f_q = g_q \circ T \Big\},$$
with the convention $\min \emptyset = +\infty$ (for $d=0$, $\mathbb{R}^0 = \{0\}$, so $d_{\mathcal{T}}(\mathcal{Q}) = 0$ iff every $f_q$ is constant on $\mathcal{X}_0$).

**Proposition 1.2 (Pathology of Unrestricted Measurable Encoders vs. Brouwer Stabilization).**  
1. *Measurable collapse*: If $\mathcal{T} = \mathcal{T}_{\mathrm{meas}}$ is the class of all Borel-measurable maps, then because every Polish space $\mathbb{R}^n$ is Borel-isomorphic to $\mathbb{R}$, any query family on $\mathcal{X}_0 \subseteq \mathbb{R}^n$ admits a 1D measurable injection $T : \mathbb{R}^n \to \mathbb{R}$ (e.g., binary digit interleaving), forcing $d_{\mathrm{meas}}(\mathcal{Q}) \le 1$ identically and erasing all dimensional structure.
2. *Continuous and linear stabilization*: Restricting to continuous encoders $\mathcal{T}_{\mathrm{cont}}$ (or linear encoders $\mathcal{T}_{\mathrm{lin}}$ when $\mathcal{X}_0$ spans a subspace $V_X = \mathrm{span}(\mathcal{X}_0)$ with nonempty relative interior in $V_X$) restores true dimensionality. Specifically, let $M : V_X \to \mathbb{R}^k$ be any linear map, and suppose $M$ factors through a continuous map $T : U \to \mathbb{R}^d$ on a nonempty open set $U \subseteq V_X$, i.e., $M|_U = g \circ T$ for some $g : T(U) \to \mathbb{R}^k$ (no continuity required of $g$). Choosing a linear complement $W \subseteq V_X$ of $\ker(M)$ so $\dim W = \mathrm{rank}(M)$ and $M|_W : W \xrightarrow{\sim} \mathrm{im}(M)$ is a linear isomorphism, a small open ball $B \subseteq W$ translated into $U$ injects continuously via $T$ into $\mathbb{R}^d$: if $T(w_1) = T(w_2)$ then $M(w_1) = g(T(w_1)) = g(T(w_2)) = M(w_2) \implies w_1 = w_2$. By **Brouwer's Invariance of Domain**, a continuous injection from a nonempty open subset of $\mathbb{R}^{\mathrm{rank}(M)}$ into $\mathbb{R}^d$ can exist only if $d \ge \mathrm{rank}(M)$. Since $T = M$ (composed with a linear isomorphism $\mathrm{im}(M) \cong \mathbb{R}^{\mathrm{rank}(M)}$) achieves $d = \mathrm{rank}(M)$, we obtain
   $$d_{\mathrm{cont}}(M) = d_{\mathrm{lin}}(M) = \mathrm{rank}(M).$$

In particular, for $\mathcal{Q}_{\mathrm{obs}}$ and $\mathcal{Q}_{\mathrm{do}}$ we write
$$d_{\mathrm{obs}} := d_{\mathcal{T}}(\mathcal{Q}_{\mathrm{obs}}), \qquad d_{\mathrm{do}} := d_{\mathcal{T}}(\mathcal{Q}_{\mathrm{do}}), \qquad d_{\mathrm{obs} \cup \mathrm{do}} := d_{\mathcal{T}}(\mathcal{Q}_{\mathrm{obs}} \cup \mathcal{Q}_{\mathrm{do}}).$$

---

### 1.4 Definition, Universal Bounds, and All Five Sign Regimes of $\Delta_d^{\mathrm{causal}}$

**Definition 1.3 (Causal Cache Premium).**  
$$\Delta_d^{\mathrm{causal}} := d_{\mathrm{obs} \cup \mathrm{do}} - d_{\mathrm{obs}}.$$

**Proposition 1.4 (Universal Dimension Bounds).**  
Whenever $d_{\mathrm{obs}}, d_{\mathrm{do}} < \infty$ and $\mathcal{T}$ is closed under Cartesian pairing $(T_1, T_2) \in \mathcal{T}(\mathcal{X}_0, \mathbb{R}^{d_1 + d_2})$ (satisfied by both $\mathcal{T}_{\mathrm{lin}}$ and $\mathcal{T}_{\mathrm{cont}}$):
$$\max(d_{\mathrm{obs}}, d_{\mathrm{do}}) \le d_{\mathrm{obs} \cup \mathrm{do}} \le d_{\mathrm{obs}} + d_{\mathrm{do}}, \qquad 0 \le \Delta_d^{\mathrm{causal}} \le d_{\mathrm{do}}.$$
*Proof:* Any sufficient cache $T$ for $\mathcal{Q}_{\mathrm{obs}} \cup \mathcal{Q}_{\mathrm{do}}$ is sufficient for $\mathcal{Q}_{\mathrm{obs}}$ and $\mathcal{Q}_{\mathrm{do}}$ separately, giving $d_{\mathrm{obs} \cup \mathrm{do}} \ge \max(d_{\mathrm{obs}}, d_{\mathrm{do}})$ and $\Delta_d^{\mathrm{causal}} \ge 0$. Conversely, pairing minimal caches $(T_{\mathrm{obs}}, T_{\mathrm{do}})$ of dimensions $d_{\mathrm{obs}}$ and $d_{\mathrm{do}}$ yields a valid cache of dimension $d_{\mathrm{obs}} + d_{\mathrm{do}}$ for $\mathcal{Q}_{\mathrm{obs}} \cup \mathcal{Q}_{\mathrm{do}}$, giving $d_{\mathrm{obs} \cup \mathrm{do}} \le d_{\mathrm{obs}} + d_{\mathrm{do}}$ and $\Delta_d^{\mathrm{causal}} \le d_{\mathrm{do}}$. $\square$

Notably, $d_{\mathrm{do}} \le d_{\mathrm{obs}}$ does **not** imply $\Delta_d^{\mathrm{causal}} = 0$, nor does $A \notin X$ imply $\Delta_d^{\mathrm{causal}} = 0$. All five feasible sign regimes of $(d_{\mathrm{do}} - d_{\mathrm{obs}},\, \Delta_d^{\mathrm{causal}})$ are realized by valid acyclic SCMs:

| Regime | $d_{\mathrm{do}}$ vs $d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}}$ | Explicit Acyclic SCM Witness ($W \perp (A,Z)$, $\mathbb{E}[W]=0$, $\mathrm{Var}(\eta)>0$) | Dimensions $(d_{\mathrm{obs}}, d_{\mathrm{do}}, d_{\mathrm{obs}\cup\mathrm{do}})$ |
|---|---|---|---|---|
| 1 | $d_{\mathrm{do}} < d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} = 0$ | $Z \sim \mathcal{N}(0,1)$, $A = Z+\eta$, $Y = \alpha A + W$ ($\alpha \neq 0$), $X = (A,Z)$ | $(1, 0, 1)$, $\Delta = 0$ |
| 2 | $d_{\mathrm{do}} < d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} > 0$ | $Z \sim \mathcal{N}(0,1)$, $A = (A_1, A_2) = (Z+\eta_1, Z+\eta_2)$, $Y = (A_1+Z+W_1,\, A_2+Z+W_2)$, $X = (A_1, A_2, Z)$ | $(2, 1, 3)$, $\Delta = 1$ |
| 3 | $d_{\mathrm{do}} = d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} = 0$ | $Z \sim \mathcal{N}(0,1)$, $A = Z+\eta$, $Y = \alpha A + \beta Z + W$ ($\alpha+\beta \neq 0, \beta \neq 0$), $X = Z$ ($A \notin X$) | $(1, 1, 1)$, $\Delta = 0$ |
| 4 | $d_{\mathrm{do}} = d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} > 0$ | Canonical SCM: $Z \sim \mathcal{N}(0,1)$, $A = Z+\eta$, $Y = \alpha A + \beta Z + W$ ($\alpha, \beta \neq 0$), $X = (A, Z)$ | $(1, 1, 2)$, $\Delta = 1$ |
| 5a | $d_{\mathrm{do}} > d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} > 0$ | Nonlinear interaction: $Z = (Z_1, Z_2) \sim \mathcal{N}(0, I_2)$, $A = Z_1 + \eta$, $Y = A Z_1 + Z_2 + W$, $X = (A, Z_1, Z_2)$ | $(1, 2, 3)$, $\Delta = 2$ |
| 5b | $d_{\mathrm{do}} > d_{\mathrm{obs}}$ | $\Delta_d^{\mathrm{causal}} > 0$ | Pre-treatment unfaithful cancellation ($A \notin X$): $X = Z$, $A = Z + \eta$, $Y = A - Z + W$ ($\alpha=1, \beta=-1$) | $(0, 1, 1)$, $\Delta = 1$ |

*(Verification of Regime 2:* $f^{\mathrm{obs}}(A_1,A_2,Z) = (A_1+Z, A_2+Z)^\top$ has rank $2$; $f_a^{\mathrm{do}}(Z) = (a_1+Z, a_2+Z)^\top$ varies with $Z$ along $(1,1)^\top$, which has rank $1$; stacked $(A_1+Z, A_2+Z, Z)$ is invertible in $(A_1, A_2, Z)$ and has rank $3$, so $\Delta_d^{\mathrm{causal}} = 3 - 2 = 1$ even though $d_{\mathrm{do}} = 1 < 2 = d_{\mathrm{obs}}$! In Section 2.3 we also exhibit an $A \notin X$ example with $d_{\mathrm{obs}} = d_{\mathrm{do}} = 1$ and $\Delta_d^{\mathrm{causal}} = 1$ using a 2D covariate $X = (Z_1, Z_2)$.*)

---

## 2. Linear-Gaussian and Multi-Node Linear SCM Theorems

### 2.1 General Linear Kernel, Row-Space, and Factorization Theorems
Suppose the support $\mathcal{X}_0 = \mathrm{supp}(P_X)$ spans the linear subspace $V_X := \mathrm{span}(\mathcal{X}_0) = \mathrm{im}(\Sigma_{XX}) \subseteq \mathbb{R}^n$ (for Gaussian or non-degenerate linear SCMs, $\mathcal{X}_0 = V_X$ is an affine/linear subspace), and the observational and interventional query families take the affine-linear form
$$f^{\mathrm{obs}}(x) = C_{\mathrm{obs}} x + b_{\mathrm{obs}}, \qquad f_a^{\mathrm{do}}(x) = C_{\mathrm{do}} x + h_{\mathrm{do}}(a), \qquad x \in V_X,$$
where $C_{\mathrm{obs}} : V_X \to W_{\mathrm{obs}}$ and $C_{\mathrm{do}} : V_X \to W_{\mathrm{do}}$ are linear maps (for multi-target queries $a \in \mathcal{A}$, if $f_a^{\mathrm{do}}(x) = C_{\mathrm{do}}(a) x + h_{\mathrm{do}}(a)$, we stack a basis of $\{C_{\mathrm{do}}(a)\}_{a \in \mathcal{A}}$ into a finite-dimensional linear map $C_{\mathrm{do}}$).

**Theorem 2.1 (Kernel, Row-Space, and Factorization Characterization — Lean-Verified in `CausalPremium.lean`).**  
Let $T_{\mathrm{obs}} : E \to_{\ell} W_{\mathrm{obs}}$ and $T_{\mathrm{do}} : E \to_{\ell} W_{\mathrm{do}}$ be linear maps over a field $K$ with $\dim_K E < \infty$, and define $T_{\mathrm{joint}} := (T_{\mathrm{obs}}, T_{\mathrm{do}}) : E \to_{\ell} W_{\mathrm{obs}} \times W_{\mathrm{do}}$, $d_{\mathrm{obs}} := \mathrm{rank}(T_{\mathrm{obs}})$, $d_{\mathrm{do}} := \mathrm{rank}(T_{\mathrm{do}})$, $d_{\mathrm{obs}\cup\mathrm{do}} := \mathrm{rank}(T_{\mathrm{joint}})$, and $\Delta_d^{\mathrm{causal}} := d_{\mathrm{obs}\cup\mathrm{do}} - d_{\mathrm{obs}}$. Then:
1. **Kernel intersection & monotonicity**:
   $$\ker(T_{\mathrm{joint}}) = \ker(T_{\mathrm{obs}}) \cap \ker(T_{\mathrm{do}}), \qquad d_{\mathrm{obs}} \le d_{\mathrm{obs}\cup\mathrm{do}}, \qquad d_{\mathrm{do}} \le d_{\mathrm{obs}\cup\mathrm{do}}.$$
2. **Exact premium formulas** (via rank-nullity on $E$ and on $T_{\mathrm{do}}|_{\ker T_{\mathrm{obs}}}$):
   $$\Delta_d^{\mathrm{causal}} = \dim\big(\ker T_{\mathrm{obs}}\big) - \dim\big(\ker T_{\mathrm{obs}} \cap \ker T_{\mathrm{do}}\big) = \dim\left(\frac{\ker T_{\mathrm{obs}}}{\ker T_{\mathrm{obs}} \cap \ker T_{\mathrm{do}}}\right) = \mathrm{rank}\big(T_{\mathrm{do}}\big|_{\ker T_{\mathrm{obs}}}\big) \le d_{\mathrm{do}}.$$
   Over $K = \mathbb{R}$ (with matrices $C_{\mathrm{obs}}, C_{\mathrm{do}}$), duality $\mathrm{row}(M) = (\ker M)^\perp$ equivalently yields
   $$\Delta_d^{\mathrm{causal}} = \mathrm{rank}\begin{pmatrix} C_{\mathrm{obs}} \\ C_{\mathrm{do}} \end{pmatrix} - \mathrm{rank}(C_{\mathrm{obs}}) = \dim\left(\frac{\mathrm{row}(C_{\mathrm{obs}}) + \mathrm{row}(C_{\mathrm{do}})}{\mathrm{row}(C_{\mathrm{obs}})}\right).$$
3. **Causal invariance equivalence**: The following are equivalent:
   $$\Delta_d^{\mathrm{causal}} = 0 \iff \ker(T_{\mathrm{obs}}) \subseteq \ker(T_{\mathrm{do}}) \iff \exists L : W_{\mathrm{obs}} \to_{\ell} W_{\mathrm{do}},\; L \circ T_{\mathrm{obs}} = T_{\mathrm{do}} \iff \exists g : W_{\mathrm{obs}} \to W_{\mathrm{do}},\; \forall x,\, T_{\mathrm{do}}(x) = g(T_{\mathrm{obs}}(x)).$$
   Over $\mathbb{R}^n$, this is equivalent to $\mathrm{row}(C_{\mathrm{do}}) \subseteq \mathrm{row}(C_{\mathrm{obs}})$.
4. **Strict premium witness equivalence**:
   $$\Delta_d^{\mathrm{causal}} > 0 \iff \exists x \in E,\; T_{\mathrm{obs}}(x) = 0 \;\land\; T_{\mathrm{do}}(x) \neq 0.$$
   (Intuitively, $\Delta_d^{\mathrm{causal}} > 0$ iff there exists an observational cache collision direction $x \in \ker(T_{\mathrm{obs}})$ that strictly perturbs the interventional prediction $T_{\mathrm{do}}(x)$.)

---

### 2.2 Sharp Canonical 2-Node Linear-Gaussian SCM Example
Consider the canonical confounded linear-Gaussian SCM:
$$Z \sim \mathcal{N}(0, 1), \qquad A = Z + \eta \quad (\eta \sim \mathcal{N}(0, \sigma_\eta^2)), \qquad Y = \alpha A + \beta Z + W \quad (\mathbb{E}[W] = 0),$$
with $(\eta, W, Z)$ mutually independent, $\sigma_\eta^2 = \mathrm{Var}(\eta) > 0$, and initial observation $X = (A, Z)^\top \in \mathbb{R}^2$.

1. **Support and Gaussian conditionals**: The covariance of $X = (A, Z)^\top$ is $\Sigma_{XX} = \begin{pmatrix} 1+\sigma_\eta^2 & 1 \\ 1 & 1 \end{pmatrix}$ with $\det \Sigma_{XX} = \sigma_\eta^2 > 0$, so $V_X = \mathcal{X}_0 = \mathbb{R}^2$. Since $W \perp (A, Z)$ and $\mathbb{E}[W] = 0$:
   $$f^{\mathrm{obs}}(A, Z) = \mathbb{E}[Y \mid A, Z] = \alpha A + \beta Z \qquad \Longrightarrow \quad C_{\mathrm{obs}} = \begin{pmatrix} \alpha & \beta \end{pmatrix}.$$
   Under intervention $\mathrm{do}(A = a)$, $Y_{A \leftarrow a} = \alpha a + \beta Z + W$. Because $W \perp (A, Z)$ (no unobserved confounding between $A$ and $Y$), Theorem 1.1 guarantees $f_a^{\mathrm{cf}}(A, Z) = f_a^{\mathrm{do}}(Z) = \alpha a + \beta Z$, whose $X$-dependent linear component is
   $$C_{\mathrm{do}} = \begin{pmatrix} 0 & \beta \end{pmatrix}.$$

2. **Exact joint matrix, invertibility, and 1D $\to$ 2D premium**: Stacking $(C_{\mathrm{obs}}, C_{\mathrm{do}})$ gives
   $$M_{\alpha,\beta} := \begin{pmatrix} \alpha & \beta \\ 0 & \beta \end{pmatrix}, \qquad \det(M_{\alpha,\beta}) = \alpha\beta.$$
   - When $\alpha \neq 0$ and $\beta \neq 0$, $M_{\alpha,\beta}$ is invertible with explicit inverse $M_{\alpha,\beta}^{-1}(u_1, u_2) = \big(\frac{u_1 - u_2}{\alpha}, \frac{u_2}{\beta}\big)$, so $d_{\mathrm{obs}} = 1$, $d_{\mathrm{do}} = 1$, $d_{\mathrm{obs}\cup\mathrm{do}} = 2$, and $\Delta_d^{\mathrm{causal}} = 1$. Every 1D sufficient observational statistic $T_{\mathrm{obs}}(A,Z) = \alpha A + \beta Z$ fails to reconstruct $\beta Z$ (along $(-\beta, \alpha) \in \ker C_{\mathrm{obs}}$, $C_{\mathrm{do}}(-\beta, \alpha) = \alpha\beta \neq 0$), forcing an expansion to the 2D sufficient statistic $(\alpha A + \beta Z, \beta Z) \sim (A, Z)$.
   - **Parameter edge cases**: If $\alpha = 0$ (no causal effect of $A$, so $C_{\mathrm{obs}} = C_{\mathrm{do}} = (0, \beta)$) or $\beta = 0$ (no confounding path from $Z$ to $Y$, so $C_{\mathrm{do}} = (0, 0)$), then $\ker C_{\mathrm{obs}} \subseteq \ker C_{\mathrm{do}}$ and $\Delta_d^{\mathrm{causal}} = 0$.

3. **Quantitative bi-Lipschitz equivalence** (for $\alpha \neq 0, \beta \neq 0$):
   - *$\ell_1$-norm bounds*: With $c_{\ell_1} := \frac{1}{2}\min(|\alpha|, |\beta|) > 0$ and $C_{\ell_1} := \max(|\alpha|, 2|\beta|) > 0$,
     $$c_{\ell_1} \big(|\Delta a| + |\Delta z|\big) \le |\alpha \Delta a + \beta \Delta z| + |\beta \Delta z| \le C_{\ell_1} \big(|\Delta a| + |\Delta z|\big).$$
   - *Euclidean ($\ell_2^2$) quadratic bounds*: Using $(\alpha \Delta a + \beta \Delta z)^2 \le 2\alpha^2 \Delta a^2 + 2\beta^2 \Delta z^2$ and $\Delta a^2 + \Delta z^2 = \frac{(u_1 - u_2)^2}{\alpha^2} + \frac{u_2^2}{\beta^2} \le \big(\frac{2}{\alpha^2} + \frac{1}{\beta^2}\big)(u_1^2 + u_2^2)$, we obtain explicit constants $c_{\mathrm{sq}} := \big(\frac{2}{\alpha^2} + \frac{1}{\beta^2}\big)^{-1} > 0$ and $C_{\mathrm{sq}} := \max(2\alpha^2, 3\beta^2) > 0$ such that
     $$c_{\mathrm{sq}}(\Delta a^2 + \Delta z^2) \le (\alpha \Delta a + \beta \Delta z)^2 + (\beta \Delta z)^2 \le C_{\mathrm{sq}}(\Delta a^2 + \Delta z^2).$$
     (Exact squared singular values of $M_{\alpha,\beta}$—the eigenvalues of $M_{\alpha,\beta}^\top M_{\alpha,\beta} = \begin{pmatrix} \alpha^2 & \alpha\beta \\ \alpha\beta & 2\beta^2 \end{pmatrix}$—are $\sigma_{\max,\min}^2 = \frac{1}{2}\big(\alpha^2 + 2\beta^2 \pm \sqrt{\alpha^4 + 4\beta^4}\big) > 0$, since $(\alpha^2 + 2\beta^2)^2 - 4\alpha^2\beta^2 = \alpha^4 + 4\beta^4$.)
   - *Converse failure*: If $\alpha = 0$ or $\beta = 0$, no positive anti-Lipschitz lower bound $c > 0$ can hold (witnessed along $(1,0)-(0,0)$ when $\alpha=0$, and $(0,1)-(0,0)$ when $\beta=0$).

4. **Positivity-violation edge case ($\mathrm{Var}(\eta) = 0$)**: If $\sigma_\eta^2 = 0$, then $A = Z$ a.s. and the support collapses to the 1D diagonal $V_X = \{(z, z) : z \in \mathbb{R}\}$. Restricting $C_{\mathrm{obs}}, C_{\mathrm{do}}$ to $V_X$ via $z \mapsto (z, z)$ gives $(C_{\mathrm{obs}}|_D)(z) = (\alpha + \beta)z$ and $(C_{\mathrm{do}}|_D)(z) = \beta z$:
   - If $\alpha + \beta \neq 0$, then $\ker(C_{\mathrm{obs}}|_D) = \{0\}$ and $\Delta_d^{\mathrm{causal}} = 0$ (though $\alpha$ and $\beta$ are observationally non-identifiable due to collinearity).
   - If $\alpha + \beta = 0$ and $\beta \neq 0$ (unfaithful cancellation on $V_X$), then $C_{\mathrm{obs}}|_D = 0$ ($d_{\mathrm{obs}} = 0$) while $C_{\mathrm{do}}|_D \neq 0$ ($d_{\mathrm{do}} = 1$), yielding $\Delta_d^{\mathrm{causal}} = 1$.

---

### 2.3 Multi-Node Linear SCMs, Structural Jacobians, and Backdoor Subspace Orthogonality

Let $V = B V + \varepsilon$ be a general multi-node linear SCM over $V \in \mathbb{R}^m$ with strictly lower-triangular direct-effect matrix $B \in \mathbb{R}^{m \times m}$ (after topological ordering) and $\mathbb{E}[\varepsilon] = 0$, $\Sigma_\varepsilon = \mathrm{Cov}(\varepsilon)$. The observational reduced form is
$$V = R \varepsilon, \qquad R := (I - B)^{-1}, \qquad \Sigma_{VV} = R \Sigma_\varepsilon R^\top.$$
For an intervention $\mathrm{do}(A = a)$ on a treatment block $A = P_A V \in \mathbb{R}^{d_A}$ (where $P_A \in \mathbb{R}^{d_A \times m}$ is the coordinate selector matrix onto $A$, analogous to $E_Y \in \mathbb{R}^{d_Y \times m}$, and $Q_A := I_m - P_A^\top P_A \in \mathbb{R}^{m \times m}$ is the complementary coordinate projector onto $V \setminus A$), the mutilated structural equations satisfy $V_{A \leftarrow a} = Q_A B V_{A \leftarrow a} + Q_A \varepsilon + P_A^\top a$, giving the **interventional reduced form**
$$V_{A \leftarrow a} = R_A \big(P_A^\top a + Q_A \varepsilon\big), \qquad R_A := (I - Q_A B)^{-1}.$$
Thus the total causal effect Jacobian of $A$ on outcome block $Y \in \mathbb{R}^{d_Y}$ is $\Theta_{YA} = \frac{\partial}{\partial a} \mathbb{E}[Y_{A \leftarrow a}] = E_Y R_A P_A^\top \in \mathbb{R}^{d_Y \times d_A}$.

#### Case A: Observed Backdoor Adjustment Block $X = (A, Z)$
Suppose $X = (A, Z) \in \mathbb{R}^{d_A} \times \mathbb{R}^{d_Z}$ has full support $\mathbb{R}^{d_A + d_Z}$, where $Z \subseteq V \setminus \mathrm{De}_G(A)$ satisfies the backdoor criterion relative to $(A, Y)$ and $\varepsilon_Y$ (and all latent paths into $Y$ conditional on $Z$) is unconfounded with $A$ given $Z$. By Gaussian linearity and backdoor adjustment,
$$f^{\mathrm{obs}}(A, Z) = \Theta_{YA} A + \Theta_{YZ} Z, \qquad f_a^{\mathrm{do}}(Z) = f_a^{\mathrm{cf}}(A, Z) = \Theta_{YA} a + \Theta_{YZ} Z,$$
so the linear query maps are the block coprod $C_{\mathrm{obs}} = (\Theta_{YA} \;\; \Theta_{YZ})$ and $C_{\mathrm{do}} = (0 \;\; \Theta_{YZ})$.

**Definition 2.2 (Minimal Linear Backdoor Adjustment Subspace).**  
The **minimal linear backdoor adjustment subspace** in the covariate space $\mathbb{R}^{d_Z}$ is $\mathcal{S}_{\mathrm{adj}} := \mathrm{row}(\Theta_{YZ}) = (\ker \Theta_{YZ})^\perp \subseteq \mathbb{R}^{d_Z}$, with minimal interventional dimension $d_{\mathrm{do}} = \dim(\mathcal{S}_{\mathrm{adj}}) = \mathrm{rank}(\Theta_{YZ})$.

**Theorem 2.3 (Backdoor Subspace Intersection & Orthogonal Decomposition — Lean-Verified).**  
1. *Block-range intersection formula* (`backdoor_joint_rank`, `backdoor_causalPremium_eq_inf_range`, `backdoor_causalInvariance_iff`): Applying the invertible shear $(u, v) \mapsto (u - v, v)$ on $\mathbb{R}^{d_Y} \times \mathbb{R}^{d_Y}$ maps $\begin{pmatrix} \Theta_{YA} & \Theta_{YZ} \\ 0 & \Theta_{YZ} \end{pmatrix}$ to $\begin{pmatrix} \Theta_{YA} & 0 \\ 0 & \Theta_{YZ} \end{pmatrix}$. Hence
   $$d_{\mathrm{obs}\cup\mathrm{do}} = \mathrm{rank}(\Theta_{YA}) + \mathrm{rank}(\Theta_{YZ}), \qquad d_{\mathrm{obs}} = \dim\big(\mathrm{im}(\Theta_{YA}) + \mathrm{im}(\Theta_{YZ})\big),$$
   $$\Delta_d^{\mathrm{causal}} = \dim\big(\mathrm{im}(\Theta_{YA}) \cap \mathrm{im}(\Theta_{YZ})\big) = \dim\left(\frac{\{0\} \times \mathcal{S}_{\mathrm{adj}}}{(\{0\} \times \mathcal{S}_{\mathrm{adj}}) \cap \mathrm{row}(C_{\mathrm{obs}})}\right).$$
   In particular, **causal invariance holds ($\Delta_d^{\mathrm{causal}} = 0$) if and only if $\mathrm{im}(\Theta_{YA}) \cap \mathrm{im}(\Theta_{YZ}) = \{0\}$** (disjoint outcome-effect images). For scalar $Y$ ($d_Y = 1$), $\mathrm{im}(\Theta_{YA}) \cap \mathrm{im}(\Theta_{YZ})$ is $\{0\}$ unless both $\Theta_{YA} \neq 0$ and $\Theta_{YZ} \neq 0$, in which case $\Delta_d^{\mathrm{causal}} = 1$.
2. *Inner-product orthogonal subspace decomposition* (`causalPremium_eq_finrank_orthogonal_residual`, `causalPremium_eq_zero_iff_subspace_le`, `causalPremium_eq_doDim_of_isOrtho`): On any finite-dimensional real inner product space $E$ (e.g. $V_X \subseteq \mathbb{R}^n$), define the observational and interventional sufficient subspaces $\mathcal{S}_{\mathrm{obs}} := (\ker C_{\mathrm{obs}})^\perp$ and $\mathcal{S}_{\mathrm{do}} := (\ker C_{\mathrm{do}})^\perp$. Then
   $$(\ker C_{\mathrm{joint}})^\perp = \mathcal{S}_{\mathrm{obs}} + \mathcal{S}_{\mathrm{do}}, \qquad \Delta_d^{\mathrm{causal}} = \dim\big(\mathcal{S}_{\mathrm{obs}}^\perp \cap (\mathcal{S}_{\mathrm{obs}} + \mathcal{S}_{\mathrm{do}})\big),$$
   $$\Delta_d^{\mathrm{causal}} = 0 \iff \mathcal{S}_{\mathrm{do}} \subseteq \mathcal{S}_{\mathrm{obs}}, \qquad \Delta_d^{\mathrm{causal}} = d_{\mathrm{do}} \iff \mathcal{S}_{\mathrm{obs}} \cap \mathcal{S}_{\mathrm{do}} = \{0\} \;\;(\text{in particular when } \mathcal{S}_{\mathrm{obs}} \perp \mathcal{S}_{\mathrm{do}}).$$

#### Case B: General Observation $X = H V$ (Including $A \notin X$) and Mediators $M \in X$
1. **Purely pre-treatment observations ($A \notin X$)**: Suppose $X = Z \in \mathbb{R}^{d_Z}$ with $A = \Gamma_{AZ} Z + \eta$ ($\eta \perp Z$) and $Y = \Theta_{YA} A + \Theta_{YZ} Z + W$ ($W \perp (A,Z)$). Then
   $$C_{\mathrm{obs}} = \Theta_{YA} \Gamma_{AZ} + \Theta_{YZ}, \qquad C_{\mathrm{do}} = \Theta_{YZ}.$$
   Whenever $d_Z \ge 2$, $\Delta_d^{\mathrm{causal}} > 0$ generically even though $A \notin X$! *Concrete 2D witness:* $Z = (Z_1, Z_2)^\top \sim \mathcal{N}(0, I_2)$, $A = Z_1 + Z_2 + \eta$, $Y = A + Z_1 + W$. Then $f^{\mathrm{obs}}(Z) = 2 Z_1 + Z_2$ ($d_{\mathrm{obs}} = 1$) while $f_a^{\mathrm{do}}(Z) = a + Z_1$ ($d_{\mathrm{do}} = 1$), so $d_{\mathrm{obs}\cup\mathrm{do}} = 2$ and $\Delta_d^{\mathrm{causal}} = 1$ (realizing $d_{\mathrm{obs}} = d_{\mathrm{do}} = 1, \Delta_d^{\mathrm{causal}} = 1$ with $A \notin X$).
2. **Mediators in $X$ ($X = (A, Z, M)$)**: If $M = B_{MA} A + B_{MZ} Z + \varepsilon_M$ mediates $A \to M \to Y$ via $Y = B_{YA} A + B_{YZ} Z + B_{YM} M + \varepsilon_Y$ (with independent noises), the observational conditional expectation is $f^{\mathrm{obs}}(A, Z, M) = B_{YA} A + B_{YZ} Z + B_{YM} M$. Under the counterfactual query $Y_{A \leftarrow a} \mid (A, Z, M)$, observing $(A, Z, M)$ identifies $\varepsilon_M = M - B_{MA} A - B_{MZ} Z$, so $M_{A \leftarrow a} = M + B_{MA}(a - A)$ and
   $$f_a^{\mathrm{cf}}(A, Z, M) = \big(B_{YA} + B_{YM} B_{MA}\big) a - B_{YM} B_{MA} A + B_{YZ} Z + B_{YM} M.$$
   Comparing $C_{\mathrm{obs}} = (B_{YA},\; B_{YZ},\; B_{YM})$ and $C_{\mathrm{do}}^{\mathrm{cf}} = (-B_{YM} B_{MA},\; B_{YZ},\; B_{YM})$, their difference is $C_{\mathrm{obs}} - C_{\mathrm{do}}^{\mathrm{cf}} = (B_{YA} + B_{YM} B_{MA},\; 0,\; 0) = (\Theta_{YA},\; 0,\; 0)$. Thus whenever the total causal effect $\Theta_{YA} \neq 0$, the counterfactual cache separates the total effect from the mediator residual, again incurring $\Delta_d^{\mathrm{causal}} > 0$ unless $\mathrm{row}(\Theta_{YA} \;\; 0 \;\; 0) \subseteq \mathrm{row}(C_{\mathrm{obs}})$ (i.e., the total-effect image $\Theta_{YA} A$ is already reconstructible from $C_{\mathrm{obs}}(A, Z, M)$; for scalar $A$, $\Theta_{YA} \neq 0$ implies this reduces to $A$ being reconstructible).

---

## 3. Smooth Nonlinear SCMs: Local Jacobian Theorem & Global Counterexamples

Let $\mathcal{U} \subseteq \mathbb{R}^n$ be an open subset of $\mathcal{X}_0$, and suppose $f_{\mathrm{obs}} : \mathcal{U} \to \mathbb{R}^{k_1}$ and $f_{\mathrm{do}}(\cdot, a) : \mathcal{U} \to \mathbb{R}^{k_2}$ are $C^1$ maps for each $a \in \mathcal{A}$. Define the pointwise Jacobian row spaces
$$\mathcal{J}_{\mathrm{obs}}(x) := \mathrm{row}\big(D_x f_{\mathrm{obs}}(x)\big) \subseteq \mathbb{R}^n, \qquad \mathcal{J}_{\mathrm{do}}(x) := \mathrm{span}_{a \in \mathcal{A}}\, \mathrm{row}\big(D_x f_{\mathrm{do}}(x, a)\big) \subseteq \mathbb{R}^n.$$

**Theorem 3.1 (Local Jacobian Obstruction & Constant-Rank Criterion).**  
1. *Differential obstruction (Lean-verified in `CausalPremium.lean` as `jacobian_causalPremium_eq_zero_of_local_factor` and `no_local_diff_factor_of_jacobian_premium_pos`)*: If around $x_0 \in \mathcal{U}$ a query $q_{\mathrm{do}}$ admits a locally differentiable factorization $q_{\mathrm{do}}(x) = h(q_{\mathrm{obs}}(x))$ in a neighborhood of $x_0$, then the chain rule forces $D_{x_0} q_{\mathrm{do}} = D_{q_{\mathrm{obs}}(x_0)} h \circ D_{x_0} q_{\mathrm{obs}}$, whence $\ker(D_{x_0} q_{\mathrm{obs}}) \subseteq \ker(D_{x_0} q_{\mathrm{do}})$ and the Jacobian premium vanishes: $\Delta_d^{\mathrm{causal}}(D_{x_0} q_{\mathrm{obs}}, D_{x_0} q_{\mathrm{do}}) = 0$. Conversely, a positive Jacobian premium $\Delta_d^{\mathrm{causal}}(D_{x_0} q_{\mathrm{obs}}, D_{x_0} q_{\mathrm{do}}) > 0$ proves that **no** locally differentiable factorization $q_{\mathrm{do}} = h \circ q_{\mathrm{obs}}$ exists near $x_0$.
2. *Local constant-rank characterization (analytic)*: Suppose $\Omega \subseteq \mathcal{U}$ is an open neighborhood on which $\dim \mathcal{J}_{\mathrm{obs}}(x) \equiv r_1$ and $\dim(\mathcal{J}_{\mathrm{obs}}(x) + \mathcal{J}_{\mathrm{do}}(x)) \equiv r_{12}$ are constant, with $r_{12}$ attained by a finite tuple $F_{\mathrm{do}}(x) = (f_{\mathrm{do}}(x, a_1), \dots, f_{\mathrm{do}}(x, a_m))$. Then on a sufficiently small open sub-neighborhood $\Omega' \subseteq \Omega$, the Constant Rank Theorem provides local $C^1$ charts in which $f_{\mathrm{obs}}$ and $(f_{\mathrm{obs}}, F_{\mathrm{do}})$ are coordinate projections onto $\mathbb{R}^{r_1}$ and $\mathbb{R}^{r_{12}}$. By Brouwer's Invariance of Domain applied in those charts, the local continuous minimal cache dimensions satisfy $d_{\mathrm{obs}}^{\mathrm{local}}(\Omega') = r_1$, $d_{\mathrm{obs}\cup\mathrm{do}}^{\mathrm{local}}(\Omega') = r_{12}$, and
   $$\Delta_d^{\mathrm{local}}(\Omega') = \dim\big(\mathcal{J}_{\mathrm{obs}}(x) + \mathcal{J}_{\mathrm{do}}(x)\big) - \dim\big(\mathcal{J}_{\mathrm{obs}}(x)\big),$$
   with local causal invariance $\Delta_d^{\mathrm{local}}(\Omega') = 0 \iff \ker(D_x f_{\mathrm{obs}}(x)) \subseteq \ker(D_x f_{\mathrm{do}}(x, a))$ for all $x \in \Omega', a \in \mathcal{A}$.

### 3.2 Three Global Counterexamples Where Local Jacobian Rank Fails Globally

Global continuous cache dimension $d_{\mathrm{cont}}$ can strictly exceed the local constant Jacobian rank due to three distinct mechanisms:

1. **Counterexample 3.2 (Connected Domain with Disconnected Fibers / Branching Quotient).**  
   Consider the connected U-shaped open domain $\mathcal{U} := \{(x,y) \in \mathbb{R}^2 : x < -1 \;\lor\; y > 1 \;\lor\; y < -1\}$ and $C^\infty$ queries
   $$f_{\mathrm{obs}}(x, y) = x, \qquad f_{\mathrm{do}}(x, y) = \begin{cases} e^{-1/(x+1)^2} & \text{if } x > -1 \text{ and } y > 1, \\ 0 & \text{otherwise.} \end{cases}$$
   At every $(x, y) \in \mathcal{U}$, $\nabla f_{\mathrm{do}}(x, y)$ is a scalar multiple of $\nabla f_{\mathrm{obs}}(x, y) = (1, 0)^\top$, so $\mathcal{J}_{\mathrm{do}}(x,y) \subseteq \mathcal{J}_{\mathrm{obs}}(x,y)$ and $\Delta_d^{\mathrm{local}} \equiv 0$ everywhere! Yet on $\{x > -1\}$, each level set $f_{\mathrm{obs}} = x$ splits into two disconnected branches ($y > 1$ vs. $y < -1$) where $f_{\mathrm{do}}$ takes distinct values $e^{-1/(x+1)^2} \neq 0$. The image $(f_{\mathrm{obs}}, f_{\mathrm{do}})(\mathcal{U}) \subseteq \mathbb{R}^2$ is homeomorphic to a **Y-graph** (triod) with a degree-3 vertex at $(-1,0)$, which cannot embed continuously into $\mathbb{R}^1$ (removing the branch point disconnects the Y-graph into 3 components, whereas removing any point from an interval in $\mathbb{R}^1$ leaves at most 2). Thus $d_{\mathrm{obs}}^{\mathrm{global}} = 1$, $d_{\mathrm{obs}\cup\mathrm{do}}^{\mathrm{global}} = 2$, and $\Delta_d^{\mathrm{global}} = 1 > 0 = \Delta_d^{\mathrm{local}}$.

2. **Counterexample 3.3 (Global Topological Obstruction: Immersion vs. Embedding of $S^1$).**  
   Let $\mathcal{U} = \mathbb{R}^2 \setminus \{(0,0)\}$, $X = (x_1, x_2)$, $f_{\mathrm{obs}}(x) = x_1 / \|x\|_2$, and $f_{\mathrm{do}}(x) = x_2 / \|x\|_2$ (or radial phase queries factoring through the unit circle $S^1$). Everywhere on $\mathcal{U}$, the joint Jacobian has constant rank $1$, so the local joint dimension is $1$. However, the quotient space $(f_{\mathrm{obs}}, f_{\mathrm{do}})(\mathcal{U}) = S^1$ has no continuous injection into $\mathbb{R}^1$ (Borsuk–Ulam / Intermediate Value Theorem), forcing $d_{\mathrm{obs}\cup\mathrm{do}}^{\mathrm{global}} = 2 > 1$.

3. **Counterexample 3.4 (Non-Constant Rank Singularity in Multiplicative Interaction).**  
   In $Y = A Z + W$ over $X = (A, Z) \in \mathbb{R}^2$ ($W \perp (A, Z)$), $(f_{\mathrm{obs}}, f_{\mathrm{do}}) = (A Z,\, Z)$ has Jacobian $J(A, Z) = \begin{pmatrix} Z & A \\ 0 & 1 \end{pmatrix}$ of rank $2$ on $\{Z \neq 0\}$, but drops to rank $1$ along the singular hyperplane $\{Z = 0\}$, collapsing the entire line $\{(A, 0) : A \in \mathbb{R}\}$ to $(0, 0)$. Hence global bi-Lipschitz equivalence fails despite generic rank $2$.

---

## 4. Unobserved Confounders, Proxies, and Active Intervention ($\mathrm{do}(A = a_{\mathrm{probe}})$)

Now suppose an unobserved confounder $U$ induces confounding between $A$ and $Y$, while $X$ contains observed proxies $P$ (generated e.g. by $P \leftarrow U$) and possibly factual $A$. After observing $X$, suppose the agent performs **one post-query active intervention** $\mathrm{do}(A = a_{\mathrm{probe}})$ and observes the probe outcome $Y_{\mathrm{probe}} := Y_{A \leftarrow a_{\mathrm{probe}}}$, yielding the augmented observation $\tilde{X} := (X, Y_{\mathrm{probe}})$.

### 4.1 Positive Results: Monotonicity, Unit-Level Deconfounding, and Population Identification

1. **Linear Side-Information Monotonicity (Lean-Verified as `causalPremium_sideInfo_mono`)**:  
   When the probe supplies an additional linear functional $T_{\mathrm{probe}} : E \to_{\ell} W_{\mathrm{probe}}$ of the latent/observational state alongside $T_{\mathrm{obs}}$, appending $T_{\mathrm{probe}}$ shrinks the observational kernel from $\ker(T_{\mathrm{obs}})$ to $\ker(T_{\mathrm{obs}}) \cap \ker(T_{\mathrm{probe}})$, strictly decreasing or preserving the causal cache premium:
   $$\Delta_d^{\mathrm{causal}}(T_{\mathrm{obs}} \times T_{\mathrm{probe}},\, T_{\mathrm{do}}) \le \Delta_d^{\mathrm{causal}}(T_{\mathrm{obs}},\, T_{\mathrm{do}}),$$
   with collapse to $0$ iff $\ker(T_{\mathrm{obs}}) \cap \ker(T_{\mathrm{probe}}) \le \ker(T_{\mathrm{do}})$. (Verified in `CausalPremium.lean` by `probe_collapse_before` $= 1$ and `probe_collapse_after` $= 0$.)

2. **Theorem 4.1 (Deterministic / Injective Nonlinear Unit-Level Deconfounding & Cache Collapse).**  
   Suppose $Y_{A \leftarrow a} = g_Y(a, U) + W$ with $W \equiv 0$ (noiseless outcome mechanism) and $u \mapsto g_Y(a_{\mathrm{probe}}, u)$ injective on $\mathrm{supp}(U)$ (e.g., $g_Y(a_{\mathrm{probe}}, u) = a_{\mathrm{probe}} u + u^2$ for $u > 0, a_{\mathrm{probe}} > 0$, verified in Lean by `probe_nonlinear_inj`). Then:
   - *(Exact posterior collapse)*: $Y_{\mathrm{probe}} = g_Y(a_{\mathrm{probe}}, U)$ uniquely determines $U$ pathwise, so $\mathrm{Var}(U \mid X, Y_{\mathrm{probe}}) = 0$.
   - *(Strict reduction of $\Delta_d^{\mathrm{causal}}$)*: Consider $U > 0$, noisy proxies $X = P = (P_1, P_2)$ of $U$ (such that $\mathbb{E}[U \mid P]$ and $\mathbb{E}[U^2 \mid P]$ are functionally independent 1D statistics of $P$, e.g. $P_1 = U + \varepsilon_1$ and $P_2$ modulates conditional noise scale so $\mathrm{Var}(U \mid P)$ varies independently of $\mathbb{E}[U \mid P]$), $A = U + \eta$ ($\mathbb{E}[\eta \mid U, P] = 0$), and $Y = A U + U^2$.
     - *Before probing*: $f^{\mathrm{obs}}(P) = \mathbb{E}[U^2 + \eta U + U^2 \mid P] = 2 \mathbb{E}[U^2 \mid P]$ ($d_{\mathrm{obs}} = 1$), whereas $f_a^{\mathrm{do}}(P) = a \mathbb{E}[U \mid P] + \mathbb{E}[U^2 \mid P]$ requires both $\mathbb{E}[U \mid P]$ and $\mathbb{E}[U^2 \mid P]$ ($d_{\mathrm{do}} = 2$), giving $d_{\mathrm{obs}\cup\mathrm{do}} = 2$ and $\Delta_d^{\mathrm{causal}}(X) = 1$.
     - *After one probe $a_{\mathrm{probe}} > 0$*: Because $u \mapsto a_{\mathrm{probe}} u + u^2$ is injective on $u > 0$, $Y_{\mathrm{probe}}$ identifies $U$ exactly. Hence $f^{\mathrm{obs}}(\tilde{X}) = 2 U^2$ ($d_{\mathrm{obs}} = 1$) and $f_a^{\mathrm{do}}(\tilde{X}) = a U + U^2$ both factor continuously through the single scalar $U = g_Y(a_{\mathrm{probe}}, \cdot)^{-1}(Y_{\mathrm{probe}})$ ($d_{\mathrm{obs}\cup\mathrm{do}} = 1$). Thus one active probe collapses $\Delta_d^{\mathrm{causal}}$ from $1$ to $0$!

3. **Proposition 4.2 (Population Structural Identification under Known Zero Mean).**  
   In a linear confounded SCM $Y = \alpha A + \beta U + W$ ($\mathbb{E}[W] = 0$) where $\mathbb{E}[U] = 0$ is known and we observe only one noisy proxy $P = U + \varepsilon_P$, the parameter $\alpha$ is observationally underidentified from $(P, A, Y)$ alone unless $\varepsilon_P$ variance is known. Performing one population-level active probe $\mathrm{do}(A = a_{\mathrm{probe}})$ with $a_{\mathrm{probe}} \neq 0$ yields $\mathbb{E}[Y \mid \mathrm{do}(A = a_{\mathrm{probe}})] = \alpha a_{\mathrm{probe}} + \beta \mathbb{E}[U] = \alpha a_{\mathrm{probe}}$, uniquely identifying $\alpha = \mathbb{E}[Y \mid \mathrm{do}(A = a_{\mathrm{probe}})] / a_{\mathrm{probe}}$.

### 4.2 Five Explicit Failure Counterexamples for a Single Active Probe $\mathrm{do}(A = a_{\mathrm{probe}})$

Single-probe active deconfounding fails under any of the following five structural obstructions:

1. **Counterexample 4.3 (Outcome Noise Masking $U$).**  
   Let $U \sim \mathcal{N}(0, 1)$, $P = U + \varepsilon_P$ ($\varepsilon_P \sim \mathcal{N}(0, \sigma_P^2)$), $A = U + \eta$ ($\eta \sim \mathcal{N}(0, \sigma_\eta^2)$), and $Y = \alpha A + \beta U + W$ with $W \sim \mathcal{N}(0, \sigma_W^2)$, $\sigma_W^2 > 0$, $\alpha, \beta \neq 0$. Under a post-query active probe $\mathrm{do}(A = a_{\mathrm{probe}})$ with fresh independent probe noise $W_{\mathrm{probe}} \sim \mathcal{N}(0, \sigma_W^2)$ ($W_{\mathrm{probe}} \perp (U, P, A, W)$, and $(U, \varepsilon_P, \eta, W, W_{\mathrm{probe}})$ mutually independent), $Y_{\mathrm{probe}} = \alpha a_{\mathrm{probe}} + \beta U + W_{\mathrm{probe}}$ yields
   $$\mathrm{Var}(U \mid Y_{\mathrm{probe}}) = \frac{\sigma_W^2}{\beta^2 + \sigma_W^2} > 0,$$
   so $U$ is never exactly recovered at the unit level. On $\tilde{X} = (A, P, Y_{\mathrm{probe}})$ (because $A = U + \eta$ also informs $U$), the exact conditional expectations are
   $$f^{\mathrm{obs}}(\tilde{X}) = \alpha A + \beta \mathbb{E}[U \mid A, P, Y_{\mathrm{probe}}], \qquad f_a^{\mathrm{cf}}(\tilde{X}) = \alpha a + \beta \mathbb{E}[U \mid A, P, Y_{\mathrm{probe}}],$$
   which still have linearly independent coefficients on $(A,\, \mathbb{E}[U \mid A, P, Y_{\mathrm{probe}}])$ (the weight on $Y_{\mathrm{probe}} - \alpha a_{\mathrm{probe}}$ in $\mathbb{E}[U \mid A, P, Y_{\mathrm{probe}}]$ is nonzero whenever $\beta \neq 0$ and $\sigma_W^2 < \infty$), leaving $\Delta_d^{\mathrm{causal}}(\tilde{X}) = 1$. Note that in this linear-Gaussian model with factual $A$ omitted ($\tilde{X}_{\mathrm{pre}} = (P, Y_{\mathrm{probe}})$), both $f^{\mathrm{obs}}(\tilde{X}_{\mathrm{pre}}) = (\alpha + \beta)\mathbb{E}[U \mid P, Y_{\mathrm{probe}}]$ and $f_a^{\mathrm{do}}(\tilde{X}_{\mathrm{pre}}) = \alpha a + \beta \mathbb{E}[U \mid P, Y_{\mathrm{probe}}]$ depend only on the single scalar $\mathbb{E}[U \mid P, Y_{\mathrm{probe}}]$; however, $\sigma_W^2 > 0$ breaks unit-level recovery ($\mathrm{Var}(U \mid Y_{\mathrm{probe}}) > 0$) and breaks nonlinear moment collapse such as $Y = A U + U^2 + W$ ($Y_{\mathrm{probe}} = a_{\mathrm{probe}} U + U^2 + W_{\mathrm{probe}}$) over $\tilde{X}_{\mathrm{pre}} = (P, Y_{\mathrm{probe}})$, where $\mathbb{E}[U \mid P, Y_{\mathrm{probe}}]$ and $\mathbb{E}[U^2 \mid P, Y_{\mathrm{probe}}]$ remain functionally independent and $\Delta_d^{\mathrm{causal}}(\tilde{X}_{\mathrm{pre}}) = 1$.

2. **Counterexample 4.4 (Zero-Sensitivity / Non-Monotone Effect Modification at $a_{\mathrm{probe}}$).**  
   - *Zero-sensitivity probe point* (`probe_zero_sensitivity_fail` in Lean): For $Y = \alpha A + (A - a_{\mathrm{probe}}) U$, probing at $A = a_{\mathrm{probe}}$ yields $Y_{\mathrm{probe}} = \alpha a_{\mathrm{probe}}$, completely annihilating $U$ ($I(U; Y_{\mathrm{probe}}) = 0$), even though $U$ drives $Y_{A \leftarrow a}$ for every $a \neq a_{\mathrm{probe}}$.
   - *Symmetric / even-function sign collapse*: For $Y = (A + 1) U + (U^2 - 1)$ with $U \sim \mathcal{N}(0, 1)$ and $a_{\mathrm{probe}} = -1$, $Y_{-1} = U^2 - 1$ destroys $\mathrm{sgn}(U)$, so $\mathbb{E}[U \mid Y_{-1}] = 0$ and $\mathrm{Var}(U \mid Y_{-1}) = \mathrm{Var}(U) = 1$, leaving $U$ completely unrecovered for any $a \neq -1$.

3. **Counterexample 4.5 (Confounder Dimension Deficit & Unknown Baseline Intercept).**  
   - *Unit-level 2D confounder deficit* (`probe_two_confounder_fail_premium` and `probe_2d_deficit_fail` in Lean): Let $U = (U_1, U_2) \sim \mathcal{N}(0, I_2)$ and $Y = U_1 + A U_2$ ($W = 0$). A scalar probe $Y_{\mathrm{probe}} = U_1 + a_{\mathrm{probe}} U_2$ provides only one linear constraint in two unknowns, leaving the 1D kernel $\{(u_1, u_2) : u_1 + a_{\mathrm{probe}} u_2 = 0\}$ completely unidentifiable and causing prediction collisions for every $a \neq a_{\mathrm{probe}}$. In the linear side-information model `(a, u₁, u₂) ↦ (a + u₁ + u₂, u₁)` with rank-1 probe `u₁`, `causalPremium` remains $1 > 0$.
   - *Population-level unknown baseline intercept*: In $Y = \alpha A + \beta U + W$ when $\mu_0 := \beta \mathbb{E}[U]$ is unknown, a single probe mean $\mathbb{E}[Y \mid \mathrm{do}(A = a_{\mathrm{probe}})] = \alpha a_{\mathrm{probe}} + \mu_0$ gives one equation in two unknowns $(\alpha, \mu_0)$, failing to identify $\alpha$ without a second intervention level.

4. **Counterexample 4.6 (Retained Factual Treatment $A \in X$).**  
   Suppose $W = 0$, $Y = \alpha A + \beta U$ ($\alpha, \beta \neq 0$), and $X = (A, P)$. Even though a noiseless probe $Y_{\mathrm{probe}} = \alpha a_{\mathrm{probe}} + \beta U$ recovers $\beta U = Y_{\mathrm{probe}} - \alpha a_{\mathrm{probe}}$ with zero posterior variance, on $\tilde{X} = (A, P, Y_{\mathrm{probe}})$ we still have
   $$f^{\mathrm{obs}}(\tilde{X}) = \alpha A + (Y_{\mathrm{probe}} - \alpha a_{\mathrm{probe}}), \qquad f_a^{\mathrm{cf}}(\tilde{X}) = \alpha a + (Y_{\mathrm{probe}} - \alpha a_{\mathrm{probe}}).$$
   Thus $d_{\mathrm{obs}} = 1$, $d_{\mathrm{do}} = 1$, $d_{\mathrm{obs}\cup\mathrm{do}} = 2$, and **$\Delta_d^{\mathrm{causal}}(\tilde{X}) = 1$**! Removing unobserved confounding does **not** eliminate $\Delta_d^{\mathrm{causal}}$ whenever the factual treatment $A$ remains part of the cached observation $X$ and $\alpha \neq 0, \beta \neq 0$.

5. **Counterexample 4.7 (Collider / Post-Treatment Proxy Bias & Destructive Probing).**  
   - *Collider bias*: If $P$ is a post-treatment common child $A \to P \leftarrow U$ (e.g. $P = A + U + \varepsilon_P$), conditioning on $(A, P)$ opens the collider path $A \to P \leftarrow U$, inducing spurious dependence between $A$ and $U$ even if $A \perp U$ marginally.
   - *Destructive / state-perturbing probe*: If applying $\mathrm{do}(A = a_{\mathrm{probe}})$ shifts the unit's latent state from $U$ to $U' = U + \delta(a_{\mathrm{probe}})$ (e.g. unobserved carry-over), then $Y_{\mathrm{probe}} = g_Y(a_{\mathrm{probe}}, U')$ fails to match the pre-probe state $U$ unless $\delta(a_{\mathrm{probe}})$ is known and invertible.

---

## 5. Honest Scope Boundary: Lean 4 Kernel-Verified vs. Analytic Results

All theorems in `CausalPremium.lean` are verified directly by the Lean 4.36.0 kernel (using only standard axioms `propext`, `Classical.choice`, `Quot.sound`, with zero `sorry` or custom axioms). Analytic, measure-theoretic, and global topological arguments remain in `THEORY.md` with explicit boundaries:

| `THEORY.md` Section / Result | Status | Exact Lean 4 Theorem(s) in `CausalPremium.lean` (or Reason Analytic) |
|---|---|---|
| §1.1–1.2 SCM structural equations, $f_a^{\mathrm{cf}}$ vs $f_a^{\mathrm{do}}$, Theorem 1.1 | Analytic / Measure-theoretic | Not in Lean (requires conditional expectations & DAG $d$-separation measure theory). |
| §1.3 Proposition 1.2 (Measurable Peano collapse; Brouwer $d_{\mathrm{cont}} = \mathrm{rank}$) | Analytic / Topological | Not in Lean (requires Brouwer's Invariance of Domain in $\mathbb{R}^n$). |
| §1.4 Proposition 1.4 & 5 sign regimes | Hybrid | Linear inequalities `obsDim_le_jointDim`, `doDim_le_jointDim`, `obsDim_add_causalPremium`, `causalPremium_le_doDim` are **Lean-verified**; SCM examples are analytic. |
| §2.1 Theorem 2.1 (Kernel/rank/factorization equivalence) | **Lean-verified** | `ker_jointMap`, `obsDim_le_jointDim`, `doDim_le_jointDim`, `obsDim_add_causalPremium`, `causalPremium_eq_finrank_ker_sub`, `causalPremium_eq_finrank_map_ker`, `causalPremium_le_doDim`, `causalPremium_eq_zero_iff_ker_le`, `ker_le_iff_exists_factor`, `ker_le_iff_exists_set_factor`, `causalPremium_eq_zero_iff_exists_factor`, `causalPremium_pos_iff_exists_witness`. |
| §2.2 Canonical 2D SCM ($\alpha A + \beta Z$, $\beta Z$), $\ell_1$ & $\ell_2^2$ bi-Lipschitz, edge cases, $\mathrm{Var}(\eta)=0$ diagonal | **Lean-verified** (algebraic core) | `canonJointMap_bijective_iff`, `canonJointMap_bilipschitz`, `canonJointMap_sq_norm_bilipschitz`, `canonJointMap_not_antilipschitz_of_eq_zero`, `canonCausalPremium_eq_one_iff`, `canonCausalPremium_eq_zero_iff`, `canonDiag_causalPremium_eq_zero`, `canonDiag_unfaithful_premium_eq_one`. (Gaussian $\mathbb{E}[Y\mid A,Z]$ derivation is analytic.) |
| §2.3 Theorem 2.3(1) Multi-node backdoor block-range intersection | **Lean-verified** | `range_coprod_eq_sup`, `backdoor_joint_rank`, `backdoor_causalPremium_eq_inf_range`, `backdoor_causalInvariance_iff`. (Reduced-form $(I-B)^{-1}$ & backdoor $d$-separation are analytic.) |
| §2.3 Theorem 2.3(2) Inner-product orthogonal subspace decomposition | **Lean-verified** | `jointSubspace_eq_sup`, `obsDim_eq_finrank_obsSubspace`, `doDim_eq_finrank_doSubspace`, `jointDim_eq_finrank_sup`, `causalPremium_eq_finrank_orthogonal_residual`, `causalPremium_eq_zero_iff_subspace_le`, `causalPremium_eq_doDim_iff_disjoint`, `causalPremium_eq_doDim_of_isOrtho`. |
| §3.1 Theorem 3.1(1) Local Fréchet Jacobian obstruction | **Lean-verified** | `hasFDerivAt_joint_query`, `jacobian_causalPremium_eq_zero_of_local_factor`, `no_local_diff_factor_of_jacobian_premium_pos`. |
| §3.1(2) & §3.2 Constant Rank Theorem & 3 global counterexamples | Analytic / Differential topology | Not in Lean (requires Constant Rank Theorem, Y-graph vs $\mathbb{R}^1$ non-embeddability, and $S^1$ topology). |
| §4.1–4.2 Active-probe monotonicity, collapse witness, & algebraic failure witnesses | **Lean-verified** (algebraic core) | `causalPremium_sideInfo_mono`, `causalPremium_sideInfo_eq_zero_iff`, `probe_collapse_before`, `probe_collapse_after`, `probe_two_confounder_fail_premium`, `probe_nonlinear_inj`, `probe_zero_sensitivity_fail`, `probe_2d_deficit_fail`. (Noisy Gaussian variance, conditional expectation collapse on $P=(P_1,P_2)$, & collider $d$-separation are analytic.) |
