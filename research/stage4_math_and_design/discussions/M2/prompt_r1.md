# Track M2 (Round 1): Information-Theoretic Bregman/KL Decomposition, Calibration Forensics & Stage 4 Statistical Gate

You are **Track M2**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and rigorously dissecting **Stage 3D**'s empirical score arrays).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M2/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M2/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B <script.py>
  ```

## 1. Grounded Empirical Context (Read These Files Read-Only First)
Inspect only the relevant sections of:
1. `research/stage3d_qwen_signed_phase/recompute_all.py` (lines 49–856: exact definitions of `softmax_temperature`, `score_vectors`, `compute_16way_group_avg`, `compute_representative_readout`, `evaluate_cell_paired`, `work_f_sign_probe`, `work_g_rotation_and_controls`, `work_a_q2_analysis`)
2. `research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json` (lines 37–216: locked temperatures `temp_q3`, `temp_q2`, dev pilot stats, Work E sample size calculation `G=131`)
3. `research/stage3d_qwen_signed_phase/results/summary_master.json` (stats for `G1..G4`, `work_f_primary_040`, `work_g_primary_040`, `work_a_q2_c4`)
4. `research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz` and `blind_q2_scores.npz` (exact saved raw log-likelihoods `(2, 3, 2, 256, 4)` etc. across all Stage 3D arms, seeds, and cells!)
5. `research/stage3c/work_results/E/verify.py` (if helpful for comparing Stage 3C's sample size lock formula)

### Verified Stage 3D Empirical Facts Requiring Forensic Dissection
- `G1_main_extrap_both` (131 blocks = 8,384 mothers):
  - `phase`: Acc `23.80%`, NLL `1.4452`, Brier `0.7845`, ECE `0.1186` (`temp_q3 = 0.5099, 0.8489, 0.6575`).
  - `zero`: Acc `19.41%`, NLL `1.4809`, Brier `0.8010`, ECE `0.1827` (`temp_q3 = 1.0414, 0.7827, 0.6733`).
  - Paired `phase - zero`: `dAcc = +4.39 pp [+4.11, +4.68]`, `dNLL = +0.0357 [+0.0330, +0.0383]` (`pass_both = True` against `lo95 > +0.02`), YET BOTH arms are WORSE than uniform random guessing (`25.00%` Acc, `ln 4 = 1.3863` NLL), while `direct_s17_table` achieves `64.62%` Acc / `0.9411` NLL!
- `G2` & `G4` (cumulative angle `0.80 rad < \pi/2`): `zero` BEATS `phase` (`G2`: `36.36% / 1.3081` vs `33.88% / 1.3358`; `G4`: `34.96% / 1.3098` vs `33.29% / 1.3347`)!
- `Work G` (`G1`): Offline 16-way group averaging `Pf` lowers NLL (`phase 1.4452 -> 1.4113`; `zero 1.4809 -> 1.4391`) yet DROPS Acc (`phase 23.80% -> 17.05%`; `zero 19.41% -> 16.43%`), whereas `F0` drops NLL similarly on phase (`1.4112`) with `20.88%` Acc!
- `Work F` (`G1 ±0.40`): `phase` has `orientation_gain = +0.0098` (only ~1% of ideal `1.0`) and `swapped_minus_matched_nll = +0.0184`, yet `zero` has `orientation_gain = +0.0050` and `swapped_minus_matched_nll = +0.0074` (how can a zero-phase arm have nonzero orientation gain `+0.0050`?! Hint: check how prompt text encodes `omega = ±0.40` even in the `zero` arm, and check `seed_mean` vs `direct`!).

## 2. Mathematical & Forensic Questions You Must Answer Rigorously
1. **Exact Information-Theoretic Bregman / KL Decomposition of `ΔNLL` (Applied to REAL `blind_q3_scores.npz`)**:
   - Let $p^*(Y \mid c)$ be the exact empirical/analytical Bayes oracle posterior (`full_iq_probs` / `direct_table_probs` / empirical class frequencies over codes `(s, z)`). Derive exact algebraic identities decomposing expected cross-entropy and $\Delta \text{NLL} = \text{NLL}(q_0) - \text{NLL}(q_1)$ into:
     1. **Bayes Irreducible Entropy** $H(Y \mid \dots)$,
     2. **Calibration / Temperature-Scaling Mismatch Gap** (oracle optimal scalar $T^*$ per cell vs dev-locked $T_{\text{dev}}$ — e.g. compute exact post-hoc oracle-$T^*$ recalibrated NLL & ECE on `G1..G4`!),
     3. **Global Marginal Prior Bias** ($\text{KL}(\bar{y} \| \bar{q})$),
     4. **Equivariance-Breaking Penalty** (KL gap between raw `p` and group-averaged `Pf`),
     5. **Genuine Signed CFO Mutual Information / Resolution Gain** (how much of `G1`'s `+0.0357` NLL gain is real Bayes resolution vs simply lower overconfidence / lower ECE `0.1186 vs 0.1827` or sign-text response?!), and explain quantitatively why `zero` BEATS `phase` on `G2` (`0.80 rad`) and `G4` (`0.80 rad`)!
   - Run this exact decomposition on **ALL** cells of `research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz` (`G1`, `G2`, `G3`, `G4`, `H_c4_035`) + `blind_q2_scores.npz`!
2. **Exact Convexity Proof & Threshold Characterization of the `Pf` Paradox (NLL Drops While Acc Collapses)**:
   - Under any finite group $G$ acting equivariantly on observations/codes and labels (`\mathbb{Z}_4 \times \mathbb{Z}_4` in EM-FloatFormer):
     - Prove that for strictly proper log score, group symmetrization $P_f = \frac{1}{|G|} \sum_{g \in G} \rho(g)^{-1} q(g \cdot x)$ ALWAYS satisfies $\mathbb{E}[\text{NLL}(P_f)] \le \mathbb{E}[\text{NLL}(q)]$ with exact equality deficit $\frac{1}{|G|} \sum_{g \in G} \text{KL}(P_f \| q^{(g)})$!
     - Construct necessary and sufficient geometric conditions under which top-1 Accuracy strictly DECREASES under `Pf`, and empirically measure on `G1..G4` of `blind_q3_scores.npz` how logit-space geometric averaging (geometric mean / log-probability group averaging followed by normalization!) behaves compared to probability-space `Pf`! (Does geometric group averaging fix or change the Acc/NLL behavior on `G1..G4`? Check!)
3. **Formalizing Stage 4's Uncheatable Three-Part Composite Statistical Gate & Hierarchical Sample-Size / Power Law**:
   - Stage 3D's `dual_practical_threshold` ONLY checked paired `phase - zero > +0.02`, allowing two sub-uniform (`23.80% < 25%`, `1.4452 > ln 4`) predictors to pass!
   - Design the formal **Stage 4 Statistical Gate** covering BOTH synthetic extrapolation and real-data Pluto (`G=16`) / CTTC (`G=72` clusters across `9` sessions):
     1. **Absolute Oracle & Baseline Dominance**: Must beat trivial uniform/zero priors AND achieve strictly positive Bayes Normalized Information Gain $\text{NIG} = \frac{H(Y) - \text{NLL}(q)}{H(Y) - \text{NLL}(\text{oracle})} \ge \eta_{\min}$ AND exceed chance + random-label permutation bounds.
     2. **Calibration & Structural Invariance Gate**: Post-dev-calibration ECE bound, Brier bound, and max equivariance violation / idempotence bound (`\le 10^{-5}`).
     3. **Hierarchical Cluster + Seed Inference**: Derive exact Type-I error and power formulas under the Variance Components model $D_{b, s} = \mu + u_b + \nu_s + e_{b, s}$ (mother correlation inside blocks + multi-seed connector variance `s \in \{1..3\}` + finite clusters $G \in \{16, 72, 131\}$), derive exact Holm-Bonferroni / Intersection-Union (IUT) Type-I protection, and compute required block counts $B$ for Stage 4 synthetic & real-data tests given Stage 3D's measured variance components!
4. **Mandatory CPU Empirical Verification (`verify_m2.py`)**:
   - Write `research/stage4_math_and_design/M2/verify_m2.py` (using `uv run --python 3.12 ...`) that loads REAL `blind_q3_scores.npz` & `blind_q2_scores.npz` (and checks exact agreement with `summary_master.json`), computes the complete KL/Bregman decomposition across all 5 cells (`G1..G4`, `H_c4_035`), compares arithmetic vs geometric (logit) group averaging, runs Monte Carlo + noncentral-$t$ power verification for the variance-components Stage 4 gate, and outputs `research/stage4_math_and_design/M2/results_m2.json`.

## 3. Deliverables Required from Track M2
1. `research/stage4_math_and_design/M2/THEOREM_M2.md`
2. `research/stage4_math_and_design/M2/verify_m2.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M2/results_m2.json`
4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact decomposed Stage 3D numbers, the arithmetic vs geometric `Pf` finding, and exact gate/sample-size recommendations for M1/M3/M4 Round 2.
