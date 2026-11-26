# Track M1 (Round 1): Group-Theoretic Phase-Coupled Equivariant Architecture & Extrapolation Theory

You are **Track M1**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and fixing the root mathematical bottlenecks observed in **Stage 3D**).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M1/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M1/` (especially do NOT touch `research/matched_cache`, `research/stage3_ministral_cache`, `research/stage3b_math`, `research/stage3c`, `research/stage3d_qwen_signed_phase`).
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
  (Omit `--with torch` if only numpy/scipy are needed; test with `uv run` so all scripts run cleanly and reproducibly.)

## 1. Grounded Empirical Context (Read These Files Read-Only First)
Inspect only the relevant slices of these files inside the repo:
1. `research/matched_cache/learned.py` (`equivariant_b8_s17`, `input_features`, `fit`, `Learned.encode`)
2. `research/stage3c/G/quotient.py` (`canonical_input`, `rotate`, `one_call`)
3. `research/stage3_ministral_cache/schema.py` (`Q1`, `Q2`, `Q3` choices/prompts, `matched_sum`, `sector_of_complex`, `ROOTS`)
4. `research/stage3d_qwen_signed_phase/data_stage3d.py` (`public_phase_slots`, `TRAIN_TIMINGS`, `TRAIN_OMEGAS`, `DEV_OMEGAS`, `BLIND_CELLS`, `generate_paired_cfo_blocks`)
5. `research/stage3d_qwen_signed_phase/qwen_connector.py` (`SignedPhaseCacheConnector`, `build_prefix_embeds`, `format_chat_parts`)
6. `research/stage3d_qwen_signed_phase/recompute_all.py` (`compute_16way_group_avg`, `compute_representative_readout`, `work_g_rotation_and_controls`)
7. `research/stage3d_qwen_signed_phase/results/summary_master.json` (inspect summary stats for `G1..G4`, `work_f_primary_040`, `work_g_primary_040`)

### Verified Stage 3D Empirical Facts to Explain & Overcome
- Stage 3D trained `SignedPhaseCacheConnector` (2,184,964 trainable params, `<W1><W2><PHASE>` concatenated as additive/separate embedding blocks before frozen `Qwen/Qwen3.5-4B`) strictly on `|\omega| <= 0.25` (`TRAIN_OMEGAS = (-0.25, -0.10, 0.0, 0.10, 0.25)`) and `h+\ell <= 5` (`TRAIN_TIMINGS = ((2,2,4.0), (2,3,1.0))`), i.e. cumulative angle `|\omega|(h+\ell) <= 1.25 rad < \pi/2`.
- On blind 2x2 CFO test grid (`snr_db = 2.0`, `sigma_phi = 0.10`):
  - `G2_interp_both` (`|\omega|=0.20, h=\ell=2, |\theta|=0.80 rad`): `phase` Acc `33.88%` (NLL `1.3358`) vs `zero` Acc `36.36%` (NLL `1.3081`), `direct_s17_table` Acc `46.78%` (NLL `1.0222`). Notice `zero` beats `phase`!
  - `G4_extrap_freq` (`|\omega|=0.40, h=\ell=1, |\theta|=0.80 rad`): `phase` Acc `33.29%` (NLL `1.3347`) vs `zero` Acc `34.96%` (NLL `1.3098`), `direct_s17_table` Acc `47.75%` (NLL `1.0159`). Again `zero` beats `phase`!
  - `G1_main_extrap_both` (`|\omega|=0.40, h=\ell=2, |\theta|=1.60 rad`): `phase` Acc `23.80%` (NLL `1.4452`) vs `zero` Acc `19.41%` (NLL `1.4809`), while `direct_s17_table` JUMPS UP to `64.62%` (NLL `0.9411`)!
  - `G3_extrap_angle` (`|\omega|=0.20, h=\ell=4, |\theta|=1.60 rad`): `phase` Acc `22.68%` (NLL `1.4410`) vs `zero` Acc `19.74%` (NLL `1.4781`), while `direct_s17_table` JUMPS UP to `62.11%` (NLL `0.9796`)!
- **Work F** (`G1 ±0.40`): `phase` `orientation_gain = +0.0098`, `swapped_minus_matched_nll = +0.0184`, factorial `metadata_M_sq = 1.854e-4 << wave_W_sq = 1.120e-3` (whereas physical `cal_ref` has `M_sq = 0.2016`).
- **Work G** (`G1`): Raw host violates $\mathbb{Z}_4$-equivariance (`rot01` mean max-abs diff `0.0591`, `rot11` `0.1263`, worst `0.4817`); post-hoc offline 16-way group averaging `Pf` lowers NLL (`1.4452 -> 1.4113`) yet DROPS Acc (`23.80% -> 17.05%`)!

## 2. Mathematical Questions You Must Answer Rigorously
1. **Exact $(\mathbb{Z}_4 \times \mathbb{Z}_4) \rtimes S^1$ Decomposition & Why `direct_s17` Jumps to 64.6% While Stage 3D Collapses <25%**:
   - Prove the exact symmetries of `(z1, z2) = (4 a + u, 4 b + v)` (`a, b \in \mathbb{Z}_4`, `u, v \in \{0..3\}` canonical inner states) across **all tasks** (`Q1`, `Q2`, `Q3_repeat`, `Q3_indep`, `Q3_after_pilot`) under continuous CFO $\omega$ and Brownian phase diffusion $\sigma_\phi$.
   - Derive mathematically why `direct_s17_table` exhibits non-monotonic "boundary dip at $|\theta| \approx \pi/4$" (`~47%` at `0.80 rad`) and "quarter-turn resonance peak near $|\theta| \approx \pi/2$" (`~64.6%` at `1.60 rad`), and prove why ANY estimator restricted to unrotated sector predictions or local polynomial extrapolation in raw logits MUST fall below random `25%` at $|\theta| = 1.60 > \pi/2$.
2. **Structural Impossibility / Failure Proof of Additive Token-Concatenation Connector (`SignedPhaseCacheConnector`)**:
   - Analyze `SignedPhaseCacheConnector`: `(e_w1(z1,z2), e_w2(z1,z2), e_phase(cos(omega*t), sin(omega*t)))` injected as separate tokens into a frozen LLM.
   - Prove the exact structural failure modes: (i) broken $\mathbb{Z}_4$ quotient equivariance (16x sample inefficiency & directional collapse), (ii) rank-1 / bilinear blind spot when phase slots live at separate attention positions from discrete cache embeddings inside a **frozen** LLM (cannot synthesize exact `exp(i * omega * t)` group rotation across unseen cumulative phase $|\theta| \in (\pi/2, \pi]$ when trained only on $|\theta| \le 1.25$), and (iii) why naive post-hoc `Pf` probability averaging improves convex NLL by Jensen's inequality yet pushes top-1 Accuracy DOWN from `23.80% -> 17.05%`!
   - Quantify the exact effective rank / Frobenius spectral mismatch on the Stage 3D physical reference (`cal_ref` of `G1`)!
3. **Exact Stage 4 Equivariant + Evidence-Coupled Connector Architecture (`EquivariantEvidenceConnector`)**:
   - Design an end-to-end differentiable, frozen-LLM-compatible connector that:
     - Enforces **exact** `0.000e+00` $\mathbb{Z}_4$-equivariance on all tasks by quotient canonicalization (`canonical_input` + cyclic logit rotation `rotate` or equivariant spectral mixing on canonical `(u, v)` + explicit task conditioning so `Q3_indep` remains flat uniform!),
     - Directly models continuous $SO(2) = S^1$ phase evolution via exact complex/phasor equivariant features $(\cos(\omega \Delta t), \sin(\omega \Delta t), \exp(-\frac{1}{2}\sigma_\phi^2 \Delta t), \dots)$ **multiplicatively coupled** to canonical inner-state evidence **inside the trainable connector** before emitting soft tokens (and optionally via a lightweight equivariant logit residual / direct physical-evidence head or frozen-LM equivariant readout),
     - Uses parameter budget comparable to or smaller than Stage 3D (`<= 2.2M` params, fast convergence inside Modal free credits).
   - Explicitly specify clean interfaces for **Track M3 (Q2/Q3_after_pilot `RefSufficientPool`)** and **Track M4 (Real Pluto/CTTC CFO+SNR inputs)**!
4. **Mandatory CPU Empirical Verification (`verify_m1.py`)**:
   - Write `research/stage4_math_and_design/M1/verify_m1.py` (runnable strictly with `uv run --python 3.12 ...`).
   - Load the REAL `equivariant_b8_s17.npz` encoder (`load_s17_encoder`) and generate REAL synthetic paired CFO data via `generate_paired_cfo_blocks` for `TRAIN_OMEGAS/TIMINGS` (`|\omega| <= 0.25, h+\ell <= 5`) and `G1..G4`.
   - Build a clean frozen-host CPU surrogate (e.g. frozen random Transformer/MLP linear readout host with frozen token embeddings + tiny trainable connector trained ONLY on `|\omega| <= 0.25, h+\ell <= 5`) comparing:
     (A) Stage 3D-style Additive Uncoupled Connector (`AdditivePhaseBaseline`),
     (B) Stage 4 Equivariant + Phase-Coupled Evidence Connector (`EquivariantPhaseCoupled`), and
     (C) Exact Analytical Quotient Oracle (`direct_s17`).
   - Verify empirically:
     - Exact $\mathbb{Z}_4$ equivariance defect on `B` (`<= 1e-6`, ideally `0.0`) vs broken equivariance on `A`;
     - Exact `G1..G4` extrapolation: `B` must cleanly extrapolate past $\pi/2$ on `G1` & `G3` (achieving **> 55% Acc** and matching `direct_s17` closely, while `A` collapses around `<= 25%`!);
     - Reproduce and verify the `Pf` NLL-improves / Acc-drops lemma numerically!

## 3. Deliverables Required from Track M1
1. `research/stage4_math_and_design/M1/THEOREM_M1.md` (comprehensive, mathematically self-contained document with formal Theorems, Proofs, Stage 3D diagnostics, Stage 4 Connector Spec, and interface contract for M3/M4).
2. `research/stage4_math_and_design/M1/verify_m1.py` (executed by you via `uv run` with zero errors).
3. `research/stage4_math_and_design/M1/results_m1.json` (generated by `verify_m1.py`, recording all verified tables, eigenvalues/ranks, equivariance diffs, `G1..G4` extrapolation metrics, and `Pf` verification numbers).
4. A crisp summary in your final message stating exact theorems proved, key numbers from `results_m1.json`, and explicit design/interface questions for M2/M3/M4 Round 2.
