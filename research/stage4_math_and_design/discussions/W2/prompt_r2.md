# Track W2 — Round 2 Cross-Track Integration & Resolution

**Your Track:** `W2` (Parameterization & Optimization Geometry: Complex-Cayley, Schur-Irrep Dynamic Hyper-LoRA, Quotient Wrapping, LEMMO Parity)
**Persistent Thread:** Keep full continuity with your Round 1 derivations in [THEOREM_W2.md](../../weight_mod/W2/THEOREM_W2.md) and [verify_w2.py](../../weight_mod/W2/verify_w2.py).

---

## Verified Converged Findings from `M1–M6` Round 2 and Peer Tracks `W1`, `W3`, `W4`

Your Round-1 `4c_quotient_cayley_schur_hyper_lora` (`3,022` params) achieved **`64.60% / 0.9526` already at Step 69** and **`64.57% / 0.9491` (`G1`), `61.33% / 0.9988` (`G5(θ=π)` vs `15.2%–19.5%` static PEFT!)** with exact bitwise `0.000e+00` $C_4 \times C_4$ defect, depth-32 Cayley gradient ratio `0.99999988` vs LoRA `503.61`, and Minimal Rank `r >= 3` (`WorkF = 0.9743`). Now confront and unify your geometry with the verified Round-1/Round-2 results of `M1–M6` and `W1, W3, W4`:

1. **Peer Track `W3` — The Complex-Nullspace Symmetry Obstruction & Bayes-KL Variance Breakthrough**:
   - `W3` proved that while an out-of-band modality gate `g_EM ∈ {0, 1}` gives bit-exact `PureKL = 0.000e+00`, mixed multi-turn EM conversations require input-side activation-covariance nullspace projection $\Delta W = B \tilde{A} P_{\perp, C}$ (`80.2×` `MixKL` reduction, `508×` mixed NLL drift reduction).
   - **Critical Mathematical Challenge from `W3` to `W2`**: Ordinary real PCA eigenvectors $U_k$ of $C_{\mathrm{gen}} = \mathbb{E}[xx^\top]$ do **NOT** commute with W2's complex-doublet structure matrix $J_d = I_m \otimes \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$! Therefore naive projection $P_{\perp, C} S P_{\perp, C}$ breaks $SO(2)$-equivariance (`[Q, J_d] ≠ 0`), **destroying Complex-Cayley's $2\times$ parameter savings and unitarity** unless the nullspace itself is complex-symmetrized!
   - **Optimization Challenge (`Bayes-KL + CF-Margin` vs `1-Hot CE`)**: `W3` proved that replacing 1-hot CE with **Physical Posterior KL Distillation + Counterfactual Slot Margin** slashes converged gradient variance **`113.732×`** (and lifts noisy-CE Step-69 G1 `+16.96 pp`). Also note W3 asks: when W2 applies **Quotient-Wrapped Weight Modification** (exact `0.000e+00` outer $C_4$ defect), `λ_orbit` is redundant on single-turn Q1–Q3 and can be set to `0` there (saving step compute) and reserved for uncanonicalized multi-turn Stage 5–6 prompts.

2. **Peer Track `W1` — Long-Context Routing Gap (`L_gap = 64 -> OOD@128`) & GDN Decay**:
   - `W1` proved that frozen `M1` / `Prefix-Only` collapses whenever intervening prose tokens reach `L_gap = 64` (`G1: 60.40% -> 43.41%`, `Q2_valid: 59.28% -> 27.64%`, GDN retention `0.0057`), and `Attn-LoRA` alone breaks on unseen `OOD@128` (`30.32%`, seed 1 `14.84%`) and loses `-1.95 pp` on distractors! Only **`DeltaNet + Attn LoRA (warmgate)`** preserves `G1 = 64.36%`, `Q2 = 65.04% -> 72.07%`, `+0.20 pp` distractor immunity, and `OOD@128 = 50.93%` (`50.7×–1,576×` GDN retention).
   - **Question for `W2`**: Reconcile W2's Round-1 budget (`Full-Attn-only Cayley: 680K` on 4B) with `W1` + `W3` (`Full-Attn 12,16,20,24,28 + paired GDN 11,15,19,23` + slow-head GDN gate preservation — still `<= 2.20M` total on 4B!), and verify whether `Schur-Irrep` + `Complex-Cayley` integrates cleanly inside selected GDN `qkv/out` as well as FullAttn!

3. **Peer Track `W4` — Canonical Gauge Eliminates LoRA Factor-Averaging & Merging Interference (`Γ_12 = 2.27`)**:
   - `W4` verified **Proposition 4.1**: naive LoRA factor averaging $(B_1+B_2)(A_1+A_2)$ is corrupted by arbitrary `GL(r)` gauge ambiguity, and even in $\Delta W$ space, near-orthogonal domain adapters ($\cos = -0.0908$) suffered asymmetric Fisher cross-interference (`Γ_12 = 2.268` vs `Γ_21 = 0.223`), penalizing `1.0×` Task Arithmetic (`1.0655` NLL vs `Linear Soup 1.0089`).
   - **Question for `W2`**: In your **`4c_quotient_cayley_schur_hyper_lora`**, the left output basis $B_{\text{align}} = M_{\text{cand}}^+ F_{\text{real}}$ is **fixed canonically** and the complex Cayley/Schur cores transform equivariantly! Prove analytically AND verify numerically whether merging two domain adapters (`Regime 1: low-SNR high-jitter` and `Regime 2: higher-SNR large-CFO`) trained in W2's canonical Schur/Cayley basis dramatically lowers cross-regime interference (`Γ_12`) and improves zero-retraining generalization on unseen mixed regimes (e.g. `f708_snr10` analogue)!
   - Also check `W4` telemetry parity: confirm **`|V| = 248,320`** across `4B/9B/27B` in your exact parameter/head table (`tie_word_embeddings` noted cleanly for 4B vs 9B/27B).

---

## Round 2 Action Items for Track W2 (Execute Strictly Inside `weight_mod/W2/`)

1. **Formal Proofs & Additions in `THEOREM_W2.md`** (append as clean Round 2 sections without breaking Round 1 proofs):
   - **Theorem 7 — Complex-Symmetrized Nullspace Cayley Theorem**: Given general-text activation covariance $C_{\text{gen}} \in \mathbb{R}^{2m \times 2m}$, define its **$U(m)$-invariant complexification**
     $$C_{\text{gen}}^{\mathbb{C}} := \frac{1}{2}\!\left(C_{\text{gen}} + J_d^\top C_{\text{gen}} J_d\right) = \Phi(\mathcal{C}_{\mathbb{C}}), \qquad \mathcal{C}_{\mathbb{C}} \in \mathbb{C}^{m \times m}, \;\mathcal{C}_{\mathbb{C}} = \mathcal{C}_{\mathbb{C}}^*,$$
     and construct the exact rank-$k_c$ complex-nullspace projector $P_{\perp, C}^{\mathbb{C}} = \Phi(I_m - U_{\mathbb{C}, k_c} U_{\mathbb{C}, k_c}^*)$. Prove that:
     1. $P_{\perp, C}^{\mathbb{C}} J_d = J_d P_{\perp, C}^{\mathbb{C}}$ **identically**;
     2. Restricting skew-Hermitian Cayley generators to $S_{\mathbb{C}}^{\perp} := \Pi_{\perp} (U_{\mathbb{C}} V_{\mathbb{C}}^* - V_{\mathbb{C}} U_{\mathbb{C}}^*) \Pi_{\perp}$ keeps $Q_{\mathbb{C}}^{\perp} \in U(m)$ **exact unitary** ($\|Q\|_2 = 1, \kappa_2(Q) = 1$, $Q R(\varphi) = R(\varphi) Q$) while acting as the **exact identity $I_d$ on the entire $2k_c$-dimensional general-text principal subspace** ($\Delta Q\, x = 0 \;\forall x \in \mathrm{span}(U_{\mathbb{C}, k_c})$ -> exact **`PureKL = 0.000`** with `g_EM` AND **`MixKL` suppressed identically on text doublets!**);
     3. Prove the non-trivial subspace survival bound showing why real $k$-PCA breaks $J_d$ invariance while complex-symmetrized $\Pi_{\perp}$ preserves $m - k_c$ intact complex doublets!
   - **Theorem 8 — Canonical Frame Alignment & Gauge-Free Adapter Merging**: Prove that fixing $B_{\text{align}}$ and constraining updates to `Complex-Cayley + Schur-Irrep` eliminates `GL(r)` factor gauge ambiguity (resolving W4 Prop 4.1) and allows factor-aligned merging / Soup / TIES across SDR regimes with lower cross-interference $\Gamma_{12}$.
   - **Reconciled Unified `4B → 9B → 27B` Budget (`|V|=248,320`)**: Reconcile `W2 FullAttn-Cayley/Schur` + `W1 GDN slow-head warmgate/selected DN` + `W3 Layer Selection` strictly within `<= 2.20M` params on 4B (`<= 1.64M–2.20M`) alongside LEMMO Tier P1/P2 attribution.

2. **Implement & Verify in `verify_w2.py` (`<= 140 s` CPU Wall-Clock!!)** and Persist to `results_w2.json`:
   - **Keep runtime fast**: Already `110.67 s` uncontended — keep Round 2 additions lean so total runtime remains **`<= 140 seconds`**!
   - **Add & verify 3 concrete Round 2 checks**:
     1. **Theorem 7 Certificate**: Compare real PCA projector $P_{\perp, C}$ (breaks `[P, J_d] != 0` and breaks unitarity unless re-inverted) vs Complex-Symmetrized Nullspace Cayley $Q_{\mathbb{C}}^{\perp}$: verify `[Q_perp, J_d] < 1e-6`, `||Q_perp^T Q_perp - I|| < 1e-6`, exact identity on top-$k_c$ text subspace (`max_drift < 1e-7`), and measure `G1/G3/G5(θ=π)` + `MixKL` text drift when integrated into **`4d_nullspace_cayley_schur_bayes_kl`**!
     2. **W3 `Bayes-KL` vs `1-Hot CE` inside `4c` / `4d`**: Compare Step-69 pilot convergence variance across the 3 seeds (`202611, 202622, 202633`) and confirm `λ_orbit = 0` parity under Quotient Wrapping (`C4 = 0.0e+00`).
     3. **Multi-Regime Adapter Merging in Canonical Basis (W4 Prop 4.1 Check)**: Train two fast regime adapters (`Regime 1` & `Regime 2`, 96–128 steps each on surrogate) comparing standard unconstrained LoRA vs `W2 Complex-Cayley + Schur-Irrep` on `Γ_12` cross-interference, `cos_sim`, and zero-retraining unseen mixed-capture NLL!

3. **Final Verification & Reporting**:
   - Execute `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W2/verify_w2.py`, require **exit code `0`** and runtime `<= 140 s`.
   - Summarize all Round 2 additions and exact verified metrics concisely. Zero edits outside `weight_mod/W2/`.
