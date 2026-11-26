# Track M3 (Round 1): Minimal Bayesian Sufficiency for Variable Pilot Multisets (Q2 / Q3_after_pilot) & Idempotent Pooling

You are **Track M3**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and diagnosing Stage 3D's Q2 reference-fusion collapse).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M3/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M3/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```

## 1. Grounded Empirical Context (Read These Files Read-Only First)
Inspect only the relevant parts of:
1. `research/stage3c/A/q2_reader.py` and `research/stage3c/A/Q2.md` (and `q2_s17_kernel.json` keys/shapes)
2. `research/stage3_ministral_cache/schema.py` (`Reference`, `dedup_references`, `matched_sum`, `ROOTS`, `Q2_PROMPT`, `Q3_PROMPT_REPEAT_AFTER_PILOT`)
3. `research/stage3_ministral_cache/ministral_connector.py` (`build_ref_feature`, lines 29–61)
4. `research/stage3d_qwen_signed_phase/data_stage3d.py` (`build_q2_arm_references`, `build_stage3d_prompt`, `Q2_BLIND_ARMS`, `generate_paired_cfo_blocks`)
5. `research/stage3d_qwen_signed_phase/qwen_connector.py` (`SignedPhaseCacheConnector.encode_refs`, `build_prefix_embeds`)
6. `research/stage3d_qwen_signed_phase/recompute_all.py` (`work_a_q2_analysis`)
7. `research/stage3d_qwen_signed_phase/results/summary_master.json` (`work_a_q2_c4`) and `research/stage3d_qwen_signed_phase/results/blind_q2_scores.npz`

### Verified Stage 3D Q2 Empirical Facts (`C4_cfo_ramp`: `omega=+0.35, h=ell=2, tp=4.0, snr=2dB, sigma_phi=0.10`, 512 mothers)
- Physical analytical `work_a_physical_reference` (`q2_reader.py`):
  - `none`: `25.00%` Acc (`1.3863` NLL)
  - `valid` (1 valid pilot): `66.41%` Acc (`0.8192` NLL)
  - `duplicate_copy_dedup`: EXACT `66.41%` (`0.8192`)
  - `independent_only`: EXACT `25.00%` (`1.3863`)
  - `valid_plus_independent`: EXACT `66.41%` (`0.8192`)
  - `fresh_resample` (2 independent valid pilots at `tp=4.0`): `74.22%` Acc (`0.6632` NLL), i.e., **+7.81 pp gain**!
- Stage 3D Qwen3.5-4B `phase` arm (`blind_q2_scores.npz`):
  - `none`: `24.80%` (`1.3928` NLL)
  - `valid`: `28.78%` (`1.3889` NLL) — barely above chance (`+3.98 pp` vs physical `+41.41 pp`!)
  - `duplicate_copy_raw` (no dedup): drops `-2.47 pp` (`28.78% -> 26.30%`)
  - `valid_plus_independent`: **drops `-4.23 pp`** (`28.78% -> 24.54%`)!
  - `fresh_resample`: **drops `-2.34 pp`** (`28.78% -> 26.43%`), **VIOLATING Blackwell information monotonicity**!
- Stage 3D `zero` arm is even worse (`valid 26.63% -> fresh_resample 25.78%`).

## 2. Mathematical Questions You Must Answer Rigorously
1. **Exact Minimal Bayesian Sufficiency for Heterogeneous Reference Multisets $\mathcal{R}$**:
   Consider two windows `(y1, y2)` at `t = (0, h)` (encoded as `z1, z2` under `equivariant_b8_s17`), unknown transmitted symbol $X_1 \in \mathbb{Z}_4$, shared nuisance carrier phase $\Phi(0) = \varphi_0 + G \frac{\pi}{2}$ (note $G \in \mathbb{Z}_4$ is the unknown carrier quadrant ambiguity when $y_1 = i^{X_1 + G}/\sqrt{3} + w_1$, $p_m = i^{G} e^{i\phi(t_{p,m})}/\sqrt{3} + \eta_m$, and $y_3 = i^{X_1+G} e^{i\phi(h+\ell)}/\sqrt{3} + w_3$!), and a variable multiset $\mathcal{R} = \{(id_m, c_m, y_m, t_{p,m}, \text{snr}_m)\}_{m=1}^M$:
   - First, carefully audit `q2_reader.py` + `schema.py` + `generate_paired_cfo_blocks` for **BOTH** pinned common $t_p$ AND heterogeneous multi-timestamp $t_{p,1} \neq t_{p,2}$!
   - **Theorem 1 (Same-$t_p$ Sufficiency & SNR-Weighted matched sum)**: Under common timestamp $t_{p,m} \equiv t_p$ and Gaussian complex matched sums $T_m$, prove that after deduplicating by `acquisition_id` and keeping only `shares_channel=True` (`valid` set $V$), the **2D real vector (1 complex number)** $S = \sum_{m \in V} w_m T_m$ with $w_m = \frac{\sqrt{N_s} A_m}{\sigma_m^2}$ ($\propto \sqrt{\text{SNR}_m}$ at fixed amplitude or matching AWGN likelihood) is a **minimal sufficient statistic** for $(X_1, X_3)$, AND derive the exact phase-marginalized closed-form posterior kernel for $P(X_1 \mid Z_1, Z_2, S, \omega, t_p, \sigma_\phi)$!
   - **Theorem 2 (Multi-$t_p$ Wiener-Bridge Insufficiency of Single Sum & Exact Precision-Weighted Sufficient Family)**: When $\sigma_\phi > 0$, $\omega \neq 0$, and pilots arrive at **distinct** timestamps $\{t^{(1)}, \dots, t^{(K)}\}$ (crucial for Stage 4 real-data Pluto/CTTC multi-pilot tracking!), prove that a single vector $S = \sum_m w_m T_m e^{-i\omega t_{p,m}}$ is **strictly insufficient** because $\phi(t)$ is an Ornstein-Uhlenbeck / Brownian motion with non-rank-1 increments $\Delta\phi_m \sim \mathcal{N}(\omega t_{p,m}, \sigma_\phi^2 t_{p,m})$! Derive the **exact Gaussian-Process / Wiener-Bridge likelihood ratio** (comparing exact Gaussian quadrature / Brownian covariance precision factor vs naive single-sum collapsing, and characterizing the approximation error as $O(\sigma_\phi^2 \Delta t)$ vs exact!).
2. **Root-Cause Failure Analysis of Stage 3D `blind_q2_scores.npz` & Variable-Length Prompt Pathology**:
   - Inspect `blind_q2_scores.npz` empirically (`q2__z1, q2__z2, q2__labels, q2__matched_valid1, q2__matched_valid2, q2__matched_indep, q2__work_a_ref_probs, q2__raw_ll`).
   - Show quantitatively why `build_ref_feature` + sequential `<REF><REF>` tokens fails:
     1. In `Valid + Independent` or `Fresh Resample`, appending a second `<REF>` shifts all downstream token positions (`prefix_len` changes!) in Qwen's RoPE + DeltaNet recurrent state, causing out-of-distribution hidden-state degradation (`valid+indep` `-4.23 pp`, `fresh` `-2.34 pp`)!
     2. `build_ref_feature` passes raw `(Re(T), Im(T), |T|, s0..s2, c_m, t_p, snr)` in global coordinates instead of canonicalized relative phase $\tilde{T}_m = T_m \cdot i^{-\text{turn}(z_1)} \cdot e^{-i\omega t_{p,m}}$ (breaking $\mathbb{Z}_4$-equivariance and forcing a 2-layer `ref_mlp` to learn complex exponentiation inside frozen LM attention)!
     3. Without structural `acquisition_id` deduplication + `shares_channel` masking inside the numerical pooling module itself, duplicate prompts and distractor tokens corrupt the LM!
3. **Design Stage 4's Fixed-Slot `RefSufficientPool` Operator**:
   - Specify an exact, deterministic, differentiable module `RefSufficientPool` (plus compatibility interface with Track M1's `EquivariantEvidenceConnector`) guaranteeing:
     1. **Exact `acquisition_id` idempotence** (with explicit conflict assertion matching `schema.py:dedup_references`),
     2. **Exact distractor nullification** (`c_m = False` contributes identically zero gradient and zero activation change),
     3. **Exact fixed token count** inside the frozen LM prompt regardless of whether $M = 0, 1, 2, \dots, 64$ references are supplied (eliminating prompt-length RoPE/DeltaNet shift!),
     4. **Exact $\mathbb{Z}_4$ joint equivariance** under `(y1, y2, refs) -> i^r (y1, y2, refs)`,
     5. **Exact Blackwell Fisher-information monotonicity** on same-$t_p$ and strict improvement on multi-$t_p$ Brownian drift!
4. **Mandatory CPU Empirical Verification (`verify_m3.py`)**:
   - Write `research/stage4_math_and_design/M3/verify_m3.py` using `uv run --python 3.12 ...`:
     - First load REAL `research/stage3d_qwen_signed_phase/results/blind_q2_scores.npz` and verify `q2_reader.py`-style exact analytical Bayes fusion vs Stage 3D's failed Q2 scores, plus run a **paired CPU surrogate model** on all `512` real `blind_q2_scores.npz` mothers (split cleanly by cross-validation/heldout folds or trained on separate `NS_CFO_TRAIN` C4 data and tested on `512` `blind_q2_scores.npz` mothers!) proving:
       - `max_diff(valid, duplicate_copy_raw) == 0.0` (Exact idempotence),
       - `max_diff(valid, valid_plus_independent) == 0.0` (Exact distractor immunity),
       - `max_rot_diff == 0.0` (Exact $\mathbb{Z}_4$ equivariance),
       - Strict Blackwell monotonicity: `none (~25%) < valid (>= 65%) < fresh_resample (>= 73%)`!
     - Also run a multi-$t_p$ Wiener-bridge numerical experiment (`t_p = (1.0, 7.0)` under $\sigma_\phi \in \{0.05, 0.20, 0.40\}$) showing exact multi-timestamp posterior NLL beating naive single-sum pooling!
     - Save all outputs to `research/stage4_math_and_design/M3/results_m3.json`.

## 3. Deliverables Required from Track M3
1. `research/stage4_math_and_design/M3/THEOREM_M3.md`
2. `research/stage4_math_and_design/M3/verify_m3.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M3/results_m3.json`
4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact Stage 3D diagnostics, theorems proved, `verify_m3.py` benchmarks, and interface questions for M1/M2/M4 Round 2.
