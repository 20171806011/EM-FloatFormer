# Phase 9 (`p9`) Prospective Formal Protocol

> **Registration status**: Within-session prospective freeze recorded to `p9/results/freeze.json` prior to evaluating ANY formal seed (`SEED_P9_FORMAL_ID = 930000000000` or `SEED_P9_FORMAL_OOD = 940000000000`). Branched cleanly from P8-C commit `ce2ff50e81f6b529ec0311393da0013b0a85b801` on isolated branch `research/p9-cross-waveform-trust-gating`; also registers sibling `p8_manifold` commit `e76c61eef26151985963460b0ab03684cd562e3e`.

## 1. Registered Baselines & Verified Pre-Formal Decisions

1. **P8-C baseline verification**: Replayed clusters `1` and `10` (64 links, 512 cells) from `ce2ff50`: `0` integer/decision diffs, `max_float_diff = 8.527e-11` (pure Windows vs Linux OpenBLAS ULP). All `9/9` `p8/tests` and `7/7` `p8c/tests` passed.
2. **P9-A decoder selection (from `p9a_dev_attribution.json`, `SEED_P9_DEV`)**: Selected `strong_large` (`Turbo MMSE-PIC`, `pic_scale=0.75`, `bp_iters=20`, OSD-2/3 `~2,641` valid candidates scored by exact 32×32 time-varying single-carrier waveform metric). On dev at 9 dB `true_csi`, `strong_large` lowered BLER from `38.57%` (`native_ldpc`) to `2.73%` (`-35.84 pp`, `1.73 s / 1k blocks`). Under estimated channels, plug-in noise-whitened waveform scoring (`noise_r = n0 + sum_ray var_ray`) is used and explicitly documented as an approximate plug-in metric.
3. **P9-B classical gate selection (from `p9b_dev_gating.json` & combined dev preview)**:
   - Best fixed strategy on `sparse` 3-regime equal-weight: `always_inherit` (`63.11%` under `strong_large`, `79.26%` under `native_ldpc`).
   - Strongest classical pre-update gate on `sparse` 3-regime equal-weight: `jeffreys_student_t` (`54.86%` under `strong_large`, `77.18%` under `native_ldpc`, `0.00 pp` continuous protection cost). `pre_bayes_factor` is also evaluated side-by-side.
4. **P9-C frozen neural weights & ensemble rule (from `p9c_dev_gate.json`, `SEED_P9_TRAIN`, 3 seeds `901, 902, 903`)**:
   - Exact real parameter parity: **`2,426`** real params across `transport_attn`, `abs_no_transport`, `local`, and `no_history`.
   - Frozen deployment rule (locked before formal scoring): **3-seed equal-weight logit ensemble** (`mean(logit_w_901, logit_w_902, logit_w_903)`), with every individual seed (`901, 902, 903`) ALSO scored and reported verbatim (never multiplying seeds × links in statistical tests!).
   - All 12 weight files (`p9/results/p9c_dev_ckpts/*.pt`) are SHA-256 hashed in `p9/results/freeze.json` prior to formal execution.

## 2. Disjoint Seed Namespaces & Formal Sample Sizes

| Split | Seed Base | Structure | Common Blocks |
| :--- | ---: | :--- | ---: |
| Training (frozen) | `920000000000` | `840` links (3 ID regimes, `p9c_dev.py`) | Train only |
| Development (frozen) | `910000000000` | `64` links/regime (`p9a/p9b`), `48`/regime (`p9c`) | Dev only |
| **Formal P9-A (SNR sweep)** | `930000000000` | `G = 16` clusters × `32` links = `512` links × `4` SNRs (`6, 9, 12, 15 dB`) | **`8,192` blocks / SNR** |
| **Formal P9-B/C ID** | `930000000000` | `G = 16` clusters × `32` links = `512` links × `3` equal-weight ID regimes (`continuous`, `partial_shift`, `full_reset`) | **`8,192` / regime (`24,576` total)** |
| **Formal P9-B/C OOD** | `940000000000` | `G = 16` clusters × `16` links = `256` links × `2` OOD regimes (`ood_strong_shift`, `ood_midframe_shift`) + `donor_mismatch` | **`4,096` / OOD regime** |

## 3. Pre-Registered Statistical Inference & Claims

### Primary 4-Contrast Holm-Bonferroni Family (Exact `2^16 = 65,536` Sign-Flip Permutation, Two-Sided `α = 0.05`, Directional Alignment Required, `df = 15`)

| # | Contrast ID | Comparison (Positive = Claimed Improvement) | Cohort & Decoder |
|---:|:---|:---|:---|
| 1 | `decoder` | `native_ldpc - strong_large` common BLER | 9 dB `true_csi` `continuous` (`512` links, `8,192` blocks) |
| 2 | `classical_gate` | `always_inherit - jeffreys_student_t` common BLER | 9 dB `sparse` 3-ID equal-weight (`1,536` links, `strong_large`) |
| 3 | `attention_vs_classical` | `jeffreys_student_t - transport_attn` common BLER | 9 dB `sparse` 3-ID equal-weight (`1,536` links, `strong_large`, frozen 3-seed ensemble) |
| 4 | `attention_vs_local` | `local - transport_attn` common BLER | 9 dB `sparse` 3-ID equal-weight (`1,536` links, `strong_large`, exact `2,426 == 2,426` params) |

*(All four contrasts are ALSO reported descriptively under `native_ldpc`, per-regime, per-seed, and on OOD regimes!)*

### Separate Engineering Non-Inferiority Gate
- Fixed code rate, frame length, and `16` common core codewords: test whether `sparse / transport_attn / strong_large` minus `dense / always_clear / strong_large` has one-sided 95% paired $t$ upper bound (`df = 15`) strictly `< +0.01` (`+1.00 pp`).
- Evaluated separately on **`continuous`** (direct P8-C engineering benchmark) and on **`3-ID equal-weight`** (mixed reliability stress test), alongside net correctly delivered info bits per frame (including the 2 extra sparse codewords).

## 4. Independent Audit & Replay Requirements
- Zero hardcoded `N0` across all production `p9/*.py` modules (`p9.independent_ref.check_no_hardcoded_n0`).
- Independent recomputation of all cluster/pooled metrics, exact `65,536`-state Holm-Bonferroni p-values, and non-inferiority bounds directly from per-link JSONL via `p9/audit.py`.
- Fresh single-worker replay of pre-selected clusters **`1` and `16`** verifying exact field-level parity.
- Every claim in `p9/P9_REPORT.md` labeled verbatim as **支持 (Supported)**, **反证 (Refuted)**, or **证据不足 (Insufficient Evidence)**.
