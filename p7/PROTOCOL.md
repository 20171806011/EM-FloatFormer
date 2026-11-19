# Phase 7 Protocol: Multi-Hypothesis C*-Equivariant Evidence Routing & Held-Out Waveform Consistency

- **Branch**: `research/p7-multihypothesis-routing`
- **Base Commit (immutable)**: `a4e6684830f68e6dfe715470dafb4ff95972e508`
- **Archived P6 Main Commit**: `63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`
- **Historical P4 Reference Commit**: `8380084c6dece64119ab72f9c3c1c212e154fb17`
- **Registration Status**: Within-session prospective freeze executed prior to any formal evaluation or checkpoint fitting; not external public preregistration.

## 1. Scientific Questions Unified Under One State

On linear time-varying single-antenna Regime R2 (`20 x 32` 16-QAM OFDM, rate-1/2 `(128, 64)` LDPC, 9 dB SNR, three physical multipath rays), one `C*`-equivariant multi-hypothesis receiver (`EM7HypothesisReceiver`) answers three coupled questions from one physical/code posterior state `(G_m, pi_m, x_m, v_m)`:

1. **What was sent?** Mixture bit posterior LLR `logsumexp_m(log pi_m + log p_m(b|y))`, coded LDPC block error rate (BLER), and proper bit cross-entropy (BCE).
2. **What should be measured elsewhere?** Held-out known-pilot complex waveform prediction `pred_y = sum_m pi_m G_m x_m`, Law-of-Total-Variance predictive covariance `pred_var = sum_m pi_m (v_m^pred + |mu_m^pred - pred_y|^2)`, and exact scale-invariant mixture Gaussian log-likelihood loss `holdout_nll` on masked held-out probes (with strict zero-leakage isolation verified by test).
3. **Which judgments are certain vs. split and revisable?** Exact decomposition of symbol uncertainty `Var(x_k|y) = E_m[v_k^(m)] + Var_m(mu_k^(m))` and per-bit disagreement `bit_disagreement = sum_m pi_m (p_1^(m) - p_1^mix)^2`, evaluated sequentially from Pass 1 (`C1_row9_erased`) to Pass 2 (`C5_holdout_update` after revealing 4 row-9 probe pilots).
4. **Architecture Showdown:** Parameter-matched (`906` real trainable parameters), identically initialized, identically trained 2D separable Hermitian Axial Attention (`EM7-Axial`, Freq -> Time global reach) vs. 3-tap Local Convolution (`EM7-Local`, strictly local radius) under dynamically roving disambiguating pilot windows (`C4a_roving_window`).

## 2. Frozen Receiver Specification

- **Hypotheses**:
  - Multi-hypothesis arms (`EM7-Axial`, `EM7-Local`, `EM7-Fixed`): `M = 4` Doppler sign-octant branches `(+,+), (+,-), (-,+), (-,-)` over dominant rays 0 and 1, with sector-constrained Fisher-trust bank updates.
  - Single-hypothesis ablation (`EM7-SingleHyp-Axial`): `M = 1`, identical axial backbone and `906` parameter slots.
- **EM Unroll & Readout**:
  - 3 inner EM steps using vectorized full-operator `diagonal_ici_demap` + expected weighted amplitude fit `weighted_amplitude_solve` + sector chirp refinement.
  - Final readout: exact full-frequency `cavity_demap` using the step-2 LDPC extrinsic prior (`make_code_prior`) and refined effective operator `G_bm`, followed by log-space branch evidence aggregation `logits_pi = gamma_phys * ell_phys + gamma_const * ell_const + delta_route_acc` (`gamma_init = 12.0`, `delta_route` scaled by `gamma_init`), and zero-parameter `step-0` exact identity (`EM7-Axial == EM7-Local == EM7-Fixed` at initialization).
  - Canonical Observed-Pilot Gauge: `pilot_gauge` performs full `C* = R_+ x U(1)` normalization strictly over valid observed pilots (`h_hat` and `pred_y` are restored by complex `scale_c`, and `pred_var` by `|scale_c|^2`).
- **Observation Metadata Access**:
  - `EM7-Axial`, `EM7-Local`, `EM7-Fixed`, `EM7-SingleHyp-Axial`, and `P6-Aware-Consistent` all receive the identical known binary receive-gain mask `gain`.
  - Classical `Waveform-DD` does NOT accept erased-pilot mask metadata and therefore is reported strictly as a descriptive legacy classical baseline, never an equal-metadata comparator on erased conditions.

## 3. Frozen Formal Disjoint Cohort & Training Recipe (`N = 10` Pairs)

- **Seed Base**: `SEED_BASE = 702611190000` (completely disjoint from Dev `901..903`, P6 `main`, and `p6_followup`/`p6_consistency`).
- **Pairs**: `pair_idx = 1..10`.
- **Shared Initialization Seed**: `SEED_BASE + 900000 + pair_idx` (identical for `EM7-Axial`, `EM7-Local`, `EM7-SingleHyp-Axial` within pair `i`).
- **Training Stream** (`40` warm-cosine Adam steps, `B = 8`, `lr = 0.004 -> 0.0004`, `lambda_hold = 0.05`, `lambda_nmse = 0.005`, `anchor_l2 = 5e-5`, `grad_clip = 1.0`, cycling `C0_intact, C1_row9_erased, C2_row0_erased, C3_scattered_half, C4a_roving_window`):
  - Waveform seed: `SEED_BASE + 10000 * pair_idx + step` (`step = 0..39`).
  - Mask/probe seed base: `SEED_BASE + 200000 + 1000 * pair_idx`.
- **Held-Out Formal Evaluation Stream**:
  - `4` batches × `B = 8` bursts per pair (`32` bursts = `576` LDPC codewords per pair -> **`5,760` blocks / `737,280` coded bits** pooled across all 10 pairs per condition).
  - Eval waveform seed: `SEED_BASE + 500000 + 1000 * pair_idx + batch` (`batch = 0..3`).
  - Condition mask seed: `SEED_BASE + 700000 + 1000 * pair_idx + batch`.
- **Archived P6 Comparator**:
  - `P6-Aware-Consistent` wraps frozen formal P6 R2 `parity` checkpoints `pair06..pair15` (`p6/results/formal/A/r2/pair{pair_idx+5:02d}/parity.pt`) inside `ObservationAwareReceiver(use_variance=True, effective=True, cavity=True)`.

## 4. Six Benchmark Conditions

1. `C0_intact`: Full pilot grid (`gain == 1`).
2. `C1_row9_erased`: Interior pilot symbol row 9 erased (`gain[:, row9_pilots] = 0`).
3. `C2_row0_erased`: Initial pilot symbol row 0 erased (`gain[:, row0_pilots] = 0`).
4. `C3_scattered_half`: Checkerboard 50% of pilots erased.
5. `C4a_roving_window`: Dynamic ambiguity test — row 9 AND `{0 or 19}` erased except a randomly roving 2-subcarrier window in each burst, breaking single-tap local symmetry.
6. `C5_holdout_update`: Sequential evidence test sharing Pass 1 masks with `C1_row9_erased`, holding out 4 random row-9 pilots as unscored probes (`holdout_nll`, `holdout_nmse`, `holdout_cal_ratio`), then revealing those 4 known pilots in Pass 2 (`prob_shift`, `flip_rate`, `delta_block_errors`, `delta_bce`).

## 5. Pre-Registered Statistical Inference (`N = 10`, Exact 1,024-State Sign Test + Holm-Bonferroni)

All primary confirmatory hypotheses operate at the pair level (`N = 10` independent checkpoint+waveform clusters) via exhaustive exact two-sided `2^10 = 1,024`-state paired sign-flip permutation tests, with step-down Holm-Bonferroni adjustment applied **separately inside each of the 4 registered families** at familywise $\alpha = 0.05$. Positive difference indicates improvement for the stated claim.

### Family 1: Multi-Hypothesis Calibration & Ambiguity Resolution (`C1_row9_erased`, 4 tests)
1. `<0.01` bin MSE/Var ratio reduction: `EM7-SingleHyp-Axial - EM7-Axial`
2. `<0.01` bin MSE/Var ratio reduction: `P6-Aware-Consistent - EM7-Axial`
3. Coded block-error reduction: `EM7-SingleHyp-Axial - EM7-Axial`
4. Coded block-error reduction: `P6-Aware-Consistent - EM7-Axial`

### Family 2: Held-Out Waveform Predictive Consistency (`C5_holdout_update`, 3 tests)
5. Held-out pilot mixture NLL reduction: `EM7-SingleHyp-Axial - EM7-Axial`
6. Held-out pilot mixture NLL reduction: `EM7-Fixed - EM7-Axial`
7. Held-out pilot NMSE reduction: `EM7-SingleHyp-Axial - EM7-Axial`

### Family 3: Selective Disagreement & Evidence Revision (`C5_holdout_update` Pass 1 -> Pass 2, 3 tests)
8. `EM7-Axial` selective probability shift: `high_disagreement_prob_shift - low_disagreement_prob_shift`
9. `EM7-Axial` selective hard-flip rate: `high_disagreement_flip_rate - low_disagreement_flip_rate`
10. `EM7-Axial` sequential coded block-error reduction: `pass1_block_errors - pass2_block_errors`

### Family 4: Dynamic Evidence Routing Architecture Showdown (`C4a_roving_window` & `C5`, 4 tests)
11. `C4a` coded block-error reduction: `EM7-Fixed - EM7-Axial`
12. `C4a` coded block-error reduction: `EM7-Local - EM7-Axial`
13. `C4a` proper bit-BCE reduction: `EM7-Local - EM7-Axial`
14. `C5` held-out pilot NLL reduction: `EM7-Local - EM7-Axial`

### Descriptive Endpoints
`C0_intact`, `C2_row0_erased`, `C3_scattered_half`, classical `Waveform-DD`, CFR NMSE (dB), pre-decoding/post-decoding BER, clean noiseless probe NMSE, full 7-bin reliability curves `[0, .01) .. [.8, inf)`, split-vs-certain (`d>0.01` vs `d<1e-4`) shift ratios, and raw LLR shifts are declared descriptive and reported completely without selective suppression.

## 6. Verification & Release Criteria

- **Checkpoint Replay (`p7/replay.py`)**: All `30/30` formal checkpoints replay batch 0 (`C1, C4a, C5`) with `0` integer block/bit diffs and `< 1e-12` BCE/NLL float discrepancy.
- **Formal `C*` Symmetry Audit (`p7/audit_gauge.py`)**: All `30` trained checkpoints × `5` pilot conditions × `27` scale transforms (`4,050` checks) verified at `threshold = 1e-12` and `0` flips; any exception retained and disclosed verbatim.
- **Isolated Single-Thread Latency (`p7/latency.py`)**: `torch.set_num_threads(1)`, B1 & B8 on `C0` & `C4a`, raw median ms/batch reported honestly without speedup extrapolations.
- **Cross-Phase Regression**: Independent fresh subprocess execution of `p7/tests`, `p6_consistency`, `p6/tests`, `p5b/tests`, `p5/tests`, and `publication.verify_p4`, plus empty `git diff --name-only a4e6684`.
