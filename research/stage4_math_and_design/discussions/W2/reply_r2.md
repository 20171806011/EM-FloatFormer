Running `verify_w2.py` (Rounds 1 + 2 combined) uncontended to verify all theorems, checks, and the `<= 140 s` CPU runtime budget.
Waiting for the current run of `verify_w2.py` to finish Parts D and E so we can inspect all metrics and optimize the hot path to stay comfortably under the `140 s` CPU budget.
Profiling `verify_w2.py` hot paths and verifying exact integer parameter counts across `4B → 9B → 27B`.
Running the optimized `verify_w2.py` (exact histogram scoring + exact 16-code quotient deduplication + paired counterfactual pass) uncontended to verify exit code `0` and runtime `<= 140 s`.
Waiting for `task-198` to finish (notice the first 7 arms reproduced bit-for-bit identically while the host CPU experiences temporary background contention; Parts C, D, and E will complete shortly using the exact 16-code deduplication and histogram scoring).
Running final verification of `verify_w2.py` (Rounds 1 + 2) uncontended to confirm exit code `0`, exact reproducibility, and runtime `<= 140 s`.
Waiting for `task-221` to finish so we can inspect both `wall_seconds` and `cpu_process_seconds`.
Running `cProfile` on representative arms to locate the exact CPU hot functions and bring `verify_w2.py` comfortably under `140 s`.
Timing Parts A and E individually and benchmarking `AdamW(foreach=True)` on `4c`.
Testing 3-seed `ThreadPoolExecutor` bit-for-bit determinism and wall-clock speedup on `4c`.
Running the final deterministic multi-worker `verify_w2.py` (Rounds 1 + 2 combined) to confirm exit code `0`, bit-for-bit reproducibility, and runtime strictly `<= 140 s`.
Running the final verification of `verify_w2.py` across 8 deterministic CPU workers (with Part E overlapping concurrently) to confirm exit code `0`, bit-for-bit identical metrics, and wall-clock runtime strictly `<= 140 s`.
Completed **Track W2 Round 2** verbatim inside `research/stage4_math_and_design/weight_mod/W2/` with zero edits outside that directory:

- **Formal Proofs & Unified Architecture Spec**: [THEOREM_W2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/THEOREM_W2.md) (Round 1 Sections 1–5 preserved intact; Round 2 Sections 6–9 appended)
- **CPU Verification Suite**: [verify_w2.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/verify_w2.py) — passed via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W2/verify_w2.py` with **exit code `0`** in **`115.90 s`** (`<= 140 s` cap)
- **Verified Numerical Receipt**: [results_w2.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/results_w2.json) (`"track": "W2_Round2"`)

---

### Summary of Round 2 Proofs & Verified Results

#### 1. Theorem 7 — Complex-Symmetrized Nullspace Cayley Theorem (`W3` Symmetry Obstruction & `MixKL`)
- **Formal Proof** ([THEOREM_W2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/THEOREM_W2.md#L314-L370)): Defining $C_{\mathrm{gen}}^{\mathbb{C}} := \frac{1}{2}(C_{\mathrm{gen}} + J_d^\top C_{\mathrm{gen}} J_d) = \Phi(\mathcal{C}_{\mathbb{C}})$ and $P_{\perp, C}^{\mathbb{C}} = \Phi(I_m - U_{\mathbb{C}, k_c} U_{\mathbb{C}, k_c}^*)$ proves:
  1. $[P_{\perp, C}^{\mathbb{C}}, J_d] = 0$ identically and $\operatorname{tr}(P_{\perp, C}^{\mathbb{C}} C_{\mathrm{gen}}) = 2 \sum_{j > k_c} \mu_j$;
  2. Projected generators $S_{\mathbb{C}}^{\perp} = \Pi_\perp (U_{\mathbb{C}} V_{\mathbb{C}}^* - V_{\mathbb{C}} U_{\mathbb{C}}^*) \Pi_\perp$ keep $Q_{\mathbb{C}}^{\perp} \in U(m)$ exact unitary ($\|Q^\perp\|_2 = \kappa_2 = 1$, $Q^\perp R(\varphi) = R(\varphi) Q^\perp$, exact $O(r_c^3 + m r_c)$ Woodbury solve) while acting as exact identity $I_d$ on the entire $2k_c$-dim general-text principal subspace;
  3. $\operatorname{range}(P_{\perp, C}^{\mathbb{C}}) = (\mathcal{S} + J_d \mathcal{S})^\perp$ is the unique maximal $J_d$-invariant subspace inside $\mathcal{S}^\perp$, preserving $m - k_c \ge m - k$ intact complex doublets (e.g., `1,024` intact doublets = `80%` width on `Qwen3.5-4B` at $k_c = 256$).
- **Verified Certificate (`results_w2.json`)**:
  - `real_pca_projector_J_comm_defect = 0.2832` vs **`complex_projector_J_comm_defect = 0.0`**, `trace_identity_abs_err = 2.92e-15`.
  - Naive post-Cayley real PCA breaks both unitarity (`0.348`, $\kappa_2 = 2139.81$) and $SO(2)$ (`0.247`); naive pre-Cayley real PCA breaks $[Q, J_d] = 0.568$ and $SO(2) = 0.441$.
  - **Theorem 7 `Q_C^perp`**: `ortho_defect = 6.66e-16`, `so2_comm_defect = 1.11e-16`, `j_comm_defect = 0.0`, `top_2kc_text_doublet_max_drift = 8.90e-16`, and **`3,285×`** text perturbation energy suppression vs unprojected Cayley (`16.8299 → 0.00512`).

#### 2. W3 `Bayes-KL + CF-Margin` vs `1-Hot CE` & Arm `4d_nullspace_cayley_schur_bayes_kl` (`λ_orbit = 0`)
- Empirical Rao–Blackwell minibatch gradient variance reduction on converged `4c`: **`90.80×`** (`0.2798 → 0.00308`).
- Across 3 seeds (`202611, 202622, 202633`, `3,022` params, **`λ_orbit = 0.0`**):

| Arm (`3,022` params) | Step-69 G1 Acc ± SD / NLL ± SD | Step-384 G1 | Step-384 G5 ($\theta=\pi$) | $C_4$ / $\frac{\pi}{2}$-Cov / Cayley Ortho | `PureKL` (`g_EM=0` / `g_EM=1`) | **`MixKL` (`g_EM=1`)** |
|---|---:|---:|---:|---:|---:|---:|
| `4c_1hot_ce` | `64.60%±0.03%` / `0.9526±0.0041` | `64.57% / 0.9491` | `61.33% / 0.9988` | `0.0e+00 / 3.6e-7 / 3.6e-7` | `0.0e+00 / 0.8693` | `0.9370` |
| `4c_bayes_kl_cf` | **`64.62%±0.00%`** / `0.9418±0.0004` | **`64.62% / 0.9413`** | **`61.33% / 0.9935`** | **`0.0e+00`** / `1.7e-7 / 3.6e-7` | `0.0e+00 / 0.2090` | `0.2207` |
| **`4d_nullspace_cayley_schur_bayes_kl`** | **`64.62%±0.00%`** / **`0.9418±0.0001`** | **`64.62% / 0.9414`** | **`61.33% / 0.9935`** | **`0.0e+00`** / `1.3e-7 / 4.8e-7` | **`0.0e+00 / 0.00115`** (**`181×` cut**) | **`0.00799`** (**`27.6×` / `117×` cut**) |

- Confirms **`40.6×`** Step-69 NLL SD collapse (`0.0041 → 0.0001`), native unit calibration ($T^* = 0.997–1.002$), bitwise **`C4 = 0.0e+00`** at **`λ_orbit = 0.0`**, and zero EM degradation under Theorem 7 complex-nullspace projection.

#### 3. Theorem 8 — Canonical Frame Alignment & Multi-Regime Adapter Merging (`W4` Prop 4.1)
- **Formal Proof** ([THEOREM_W2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/THEOREM_W2.md#L396-L448)): Fixed $B_{\mathrm{align}} = M_{\mathrm{cand}}^+ F_{\mathrm{real}}$ reduces the `GL(3)` stabilizer to $\{I_3\}$ (making canonical Schur factor averaging algebraically identical to weight-space soup for all $(\theta, V)$), Lie-algebra merging $S_{\mathbb{C}}^{\mathrm{merge}} = \sum_k \lambda_k S_{\mathbb{C}}^{(k)} \implies Q_{\mathbb{C}}^{\mathrm{merge}} = \mathrm{Cay}(S_{\mathbb{C}}^{\mathrm{merge}})$ is gauge-free and strictly closed in $U(m) \cap \Pi_\perp$, and semigroup drift factorization aligns regimes onto the same canonical physical readout.
- **Verified Multi-Regime Benchmark** (`Regime 1: 1.5 dB, σ_φ=0.24, |ω|<=0.08` vs `Regime 2: 5.0 dB, σ_φ=0.05, |ω|∈[0.32,0.42]` $\to$ **Unseen Mixed `f708` analogue `3.5 dB, σ_φ=0.14, |θ|=1.60`** & **Unseen `G5(θ=π)`**):
  - **Gauge & Unitarity**: LoRA Naive Factor Avg incurs **`61.41%`** gauge error (`UnseenMix NLL = 1.1434`); W2 Canonical Schur factor averaging matches weight soup to **`8.94e-8`**, Lie-algebra Cayley preserves **`1.33e-15`** unitarity (`0.0` $J_d$-commutation, `3.18e-16` nullspace closure drift) vs Euclidean $Q$-soup breaking unitarity (`0.212`, `78.1%` $\sigma_{\min}$ collapse).
  - **Fisher Interference & Alignment**: Readout cosine alignment rises from `+0.1410` (LoRA) to **`+0.6460`** (W2 Schur); $\Gamma_{12}^{\mathrm{add}} / \Gamma_{21}^{\mathrm{add}}$ drops from `1.186 / 1.226` $\to$ **`0.776 / 0.765`**; $\Gamma_{12}^{\mathrm{soup}} / \Gamma_{21}^{\mathrm{soup}}$ drops from `0.497 / 2.338` $\to$ **`0.391 / 0.138`** (**`16.9×`** reduction on $\Gamma_{21}^{\mathrm{soup}}$).
  - **Zero-Retraining Generalization**: **`W2 Canonical Lie+Schur Soup (0.5/0.5)`** achieves **`73.83% / 0.7904`** on Unseen Mixed `f708` (`-0.1492` nats vs LoRA Soup `66.11% / 0.9396`, oracle `73.83% / 0.7722`), **`61.33% / 1.0004`** on Unseen `G5(θ=π)` (`-1.2746` nats vs LoRA Soup `11.69% / 2.2750`), Macro-3 NLL **`0.8662`** (vs LoRA `0.9756`), and bitwise **`0.0e+00`** $C_4$ defect.

#### 4. Reconciled Unified `4B → 9B → 27B` Budget (`|V| = 248,320`, `W1 + W2 + W3 + W4`)
- Verified intra-head Gated-DeltaNet compatibility ($d_k = d_v = 128 = 2 \times 64$ even $\implies$ intra-head $U(64)$ Complex-Nullspace Cayley commutes identically with W1's per-head scalar gates $\alpha_t, \beta_t$ and preserves $\|k_t\|_2$ contraction stability).
- Exact reconciled totals ([THEOREM_W2.md §8.2](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W2/THEOREM_W2.md#L454-L473)), confirming **`|V| = 248,320`** across all three models (`tie_word_embeddings = True` on 4B, `False` on 9B/27B):
  - **`Qwen3.5-4B`** (`FA: 12,16,20,24,28` + `DN: 11,15,19,23` + W1 slow-head warmgate): **`1,131,278`** adapters (`-31%` vs W3 raw `r=8`) + `488,400` M1 connector = **`1,619,678`** total (**strictly `<= 1.64M–2.20M`** LEMMO Tier P1 cap!).
  - **`Qwen3.5-9B`**: **`1,406,222`** adapters + `689,616` connector = **`2,095,838`** total (still `<= 2.20M`).
  - **`Qwen3.8-27B`** (`FA: 20..40,48,56` + `DN: 19..39`): **`2,869,070`** adapters + `823,760` connector = **`3,692,830`** total (`0.014%` of host; Mixed MLP-NF4 + DN/FA/Cayley/IO BF16+FP32).
