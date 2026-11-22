# EM-FloatFormer Real-Data SDR OFDM: Test Infrastructure & Verification Specification

**Document**: `research/realdata_sdr_ofdm/tests/TEST_INFRA.md`  
**Scope**: Real-Data SDR OFDM (`rikluost/sdr_ofdm_dataset_v1`) End-to-End (`F12`) Test Architecture  
**Repository**: `EM-FloatFormer` (working branch `research/realdata-sdr-ofdm`)  
**Master Entry Point**: `research/realdata_sdr_ofdm/tests/test_realdata_e2e.py`  
**Test Suite Size**: Exactly **132** Test Cases (Tier 1: 55, Tier 2: 55, Tier 3: 15, Tier 4: 7)  
**Standard Environment**: Python 3.12+, NumPy 1.26+, SciPy 1.13+, PyTorch 2.x (`torch.set_num_threads(1)`), `pytest` / `unittest`

---

## 1. Executive Summary & Testing Philosophy

The `research/realdata_sdr_ofdm/tests/` E2E suite verifies the end-to-end application of EM-FloatFormer receiver principles (global reference & content matching, multi-hypothesis posterior fusion, and lightweight $C^*$-equivariant trainable attention) to real over-the-air ADALM-Pluto SDR 16-QAM CP-OFDM data (`rikluost/sdr_ofdm_dataset_v1`, with `ADALM-PLUTO QPSK` `10.17632/k72b4pnc5g.1` documented as backup).

### Core Testing Pillars
1. **Opaque-Box & Interface Independence**:
   - Exercises public module contracts (`RealOFDMFrame`, `apply_stress_perturbation`, `ReceiverOutput`, `run_frame`, `eval.evaluator`, CLI scripts, and JSON/Markdown deliverables) with zero reliance on private internals.
2. **Physical, Mathematical & Statistical Invariant-Driven Assertions**:
   - Enforces exact SDR OFDM grid partitioning (`14 x 128 = 1792` REs: `14` pilots, `14` DC, `364` guards, `1400` payload REs = `5600` bits/TTI).
   - Enforces **non-Gray natural-binary `mapping_table(4)`** (`[-3, -1, +1, +3] / sqrt(10)` indexed `8*b0+4*b1+2*b2+b3`, positive LLR = bit `1`) via an independent reference oracle in `e2e/harness.py`.
   - Enforces $C^* = \mathbb{R}_{>0} \times U(1)$ gauge equivariance across 20–24 octaves, Parseval energy conservation, simplex & total-covariance identities, Section 5.1 scoring-value correlation guard, exact mixture likelihood log-sum-exp, leak-free Train/Dev/Test firewalls, and `p12/audit.py` paired inference.
   - Zero trivial facade assertions (`assert True`, `pass`, or `assert x is not None`).
3. **File-Existence-Aware Progressive Testability & Contract Enforcement**:
   - `MilestoneResolver` in `e2e/harness.py` checks file existence on disk (`RESEARCH_ROOT / ...`) BEFORE importing.
   - Unimplemented M2–M5 modules/artifacts raise informative `unittest.SkipTest("[PROGRESSIVE EXECUTION] ...")` notices.
   - Broken imports/syntax errors inside existing `.py` files propagate loudly, and `MilestoneResolver.get_callable` / `get_first_callable` raise `AssertionError` (`[CONTRACT VIOLATION]`) when an existing on-disk module omits or exposes a non-callable contract symbol (zero false-pass guarantee).
4. **Offline-First + Optional Live Codespace Validation**:
   - Every test runs deterministically offline using seeded synthetic `RealOFDMFrame` and `CustomDataset` `.pth` fixtures.
   - When `/tmp/tw_realdata` (`EMFF_REALDATA_ROOT`) exists on disk (Codespace environment), F3/F5 tests additionally verify live raw SHA-256 checksums, dataset lengths (`load_raw_sdr_ofdm`), and cross-split payload hash disjointness.
5. **Strict Repository Isolation, Windows OpenMP & Deterministic Reproducibility**:
   - All test artifacts live strictly inside `research/realdata_sdr_ofdm/tests/`.
   - Deterministic `E2E_SEED_BASE = 1_360_000_000_000` namespace disjoint from P1–P12 seeds (`<= 1_269_999_999_999`), pinned `torch.set_num_threads(1)`.
   - On Windows (`sys.platform == "win32"`), `test_realdata_e2e.py` sets `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` prior to importing `torch` so `torch.set_num_threads(1)` paired with SciPy/NumPy inside `p12.audit` does not trigger `OMP: Error #15` (`libiomp5md.dll`); Linux Codespace execution is unaffected.

---

## 2. Directory Layout

```text
research/realdata_sdr_ofdm/tests/
├── __init__.py
├── TEST_INFRA.md                         # This infrastructure specification
├── TEST_READY.md                         # Readiness & execution attestation
├── test_realdata_e2e.py                  # Master unittest & pytest aggregator (132 tests)
└── e2e/
    ├── __init__.py
    ├── harness.py                        # MilestoneResolver, non-Gray 16-QAM oracle, synthetic builders, stats oracles
    ├── test_tier1_feature_coverage.py    # Tier 1: F1–F11 happy-path feature coverage (55 tests)
    ├── test_tier2_boundary_corner.py     # Tier 2: F1–F11 boundary & corner cases (55 tests)
    ├── test_tier3_cross_feature.py       # Tier 3: Pairwise cross-feature interactions P01–P15 (15 tests)
    └── test_tier4_real_workloads.py      # Tier 4: End-to-end workload scenarios S01–S07 (7 tests)
```

---

## 3. Canonical Execution Commands

1. **Full E2E Test Suite via `unittest` (132 tests)**:
   ```bash
   python -B -m unittest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
2. **Full E2E Test Suite via `pytest` (132 tests)**:
   ```bash
   python -B -m pytest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
3. **Feature-Filtered Execution (`unittest -k`)**:
   ```bash
   python -B -m unittest -k "f01" research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
   *(Replace `"f01"` with `"f02"` .. `"f11"` to isolate any feature across Tier 1, Tier 2, and Tier 3.)*

---

## 4. Feature Inventory & 4-Tier Test Allocation Matrix

| Feature ID | Feature Name | Target Milestone | Primary Module / Artifact | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Total |
|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **F1** | Git Sync & Branching | M1 | Git branch `research/realdata-sdr-ofdm` | 5 | 5 | P01 | S07 | 12 |
| **F2** | Data Source Catalog | M1 | `DATA_SOURCES.md` | 5 | 5 | P01, P02 | S07 | 13 |
| **F3** | Remote Acquisition & Probe | M1 | `manifests/raw_checksums.json`, `scripts/` | 5 | 5 | P02, P03 | S01 | 13 |
| **F4** | Unified Data Adapter & Non-Gray 16-QAM | M1 | `adapters/sdr_ofdm_adapter.py`, `modulation.py` | 5 | 5 | P03, P04, P05, P08 | S01, S03, S05 | 17 |
| **F5** | Leak-Free Partition & Protocol | M1 | `manifests/splits.json`, `PROTOCOL.md` | 5 | 5 | P04, P06 | S07 | 13 |
| **F6** | Classical Receiver Baseline | M2 | `receivers/classical.py` | 5 | 5 | P05, P06, P07, P09, P12 | S01, S02, S03, S05 | 19 |
| **F7** | Unified Evaluator & Stress Perturbations | M2 | `eval/evaluator.py`, `p12/audit.py` | 5 | 5 | P07, P08, P10, P14, P15 | S01, S02, S06 | 18 |
| **F8** | Group A: Global Reference & Content Matching | M3 | `receivers/group_a_reference.py` | 5 | 5 | P09, P10, P11 | S01, S03, S04, S05 | 17 |
| **F9** | Group B: Hard Selection vs Posterior Fusion | M3 | `receivers/group_b_fusion.py` | 5 | 5 | P11, P12, P13 | S01, S02, S03, S04, S05 | 18 |
| **F10** | Group C: Lightweight Trainable Module | M4 | `receivers/group_c_trainable.py` | 5 | 5 | P13, P14 | S01, S05 | 14 |
| **F11** | Full Codespace Run, Figures & Delivery | M5 | `results/`, `figures/`, `REPORT.md` | 5 | 5 | P15 | S07 | 12 |
| **Total** | **All Features (F1–F11)** | M1–M5 | **132 Unique E2E Test Cases** | **55** | **55** | **15** | **7** | **132** |

---

## 5. Physical & Mathematical Invariants Enforced

| Invariant | Mathematical Identity / Formulation | Acceptance Bound | Features |
|:---|:---|:---|:---|
| **SDR OFDM Grid Partition** | $14\text{ pilots} + 14\text{ DC} + 364\text{ guard} + 1400\text{ payload} = 1792$ REs | Exact integer equality | F3, F4 |
| **Non-Gray 16-QAM (`mapping_table(4)`)** | $C[8b_0+4b_1+2b_2+b_3] = \frac{(-3+2q_I)+j(-3+2q_Q)}{\sqrt{10}}$, $\mathbb{E}\|C\|^2=1$ | $\text{atol} < 10^{-6}$, 0 noiseless bit errors | F4, F6 |
| **AWGN LLR Calibration** | $\Lambda_k = \log \frac{\sum_{c:b_k=1}\exp(-\|z-c\|^2/n_0)}{\sum_{c:b_k=0}\exp(-\|z-c\|^2/n_0)}$ | $\text{ECE} < 0.03$, exact-ref match $< 10^{-4}$ | F4 |
| **$C^*$ Gauge Equivariance** | $H_{\text{est}}(cY) = c H_{\text{est}}(Y)$, $\text{LLR}(cY, \|c\|^2 n_0) = \text{LLR}(Y, n_0)$ | $\max \|\Delta\text{LLR}\| < 10^{-3}$, 0 bit flips ($2^{-12}..2^{12}$) | F4, F6, F8, F10 |
| **Parseval Stress Physics** | Time-domain IFFT $\to$ CFO/phase $\to$ FFT preserves energy; `extra_awgn_db` is added-noise SNR in dB ($\sigma^2_{\text{add}} = p_{\text{active}} \cdot 10^{-\text{extra\_awgn\_db}/10}$, so monotonic stress sweeps descend from larger to smaller dB, e.g., `None -> 12.0 -> 6.0` or `None -> 10.0 -> 5.0 -> 0.0`) | $\|E_{\text{out}}/E_{\text{in}} - 1\| < 10^{-4}$; strict EVM monotonicity | F4, F7 |
| **Simplex & Total Covariance** | $\sum_r \alpha_r = 1$, $\Sigma = \sum_r \alpha_r \Sigma_r + \sum_r \alpha_r \|m_r - m\|^2$ | $\text{atol} < 10^{-6}$, $\text{MSE}(m) \le \text{MSE}(m_{\text{MAP}})$ | F8, F9 |
| **Exact Mixture Likelihood** | $g(c) = \sum_r \alpha_r g_r(c)$ in log-sum-exp (no LLR averaging) | Matches analytic log-mixture; $\alpha=e_{r^*}$ collapses to MAP | F9 |
| **Anti-Leakage Firewall** | $\text{Train} \cap \text{Test} = \emptyset$ (frame IDs, group IDs, payload SHA-256) | Exact empty set intersection; `.fit()` rejects `"test"` in both `train` & `dev` | F5, F10 |
| **Exact Paired Audit & Replay** | $2^G$ sign-flip test (`p12/audit.py`), Holm step-down, exact replay | `max_float_diff == 0.0`, `mismatches == 0` | F7, F11 |

---

## 6. M1–M4 Interface Contracts Verified by the E2E Suite

- **M1 (`adapters.sdr_ofdm_adapter`, `adapters.modulation`)**:
  - `RealOFDMFrame` (with `meta` keys `s0`, `guard_band_power`, `pilot_snr_db`, `pilot_rms`, `dc_power`, `heldout_pilot_folds`, `tx_bits_sha256`, `rx_grid_sha256`), `load_raw_sdr_ofdm(path, *, max_frames=None)`, `build_frame(...)`, `build_leak_free_splits(...)`, `apply_stress_perturbation(...)`, and `exact_llr` / `maxlog_llr` / `mapping_table(4)`.
- **M2 Classical (`receivers.classical`)**:
  - `CLASSICAL_METHODS = ("ls_linear_zf", "lmmse_1d", "lmmse_2d_tracked")` (aliases `"ls_zf"`, `"lmmse"`), `ClassicalConfig`, `ClassicalReceiver`, `ReceiverOutput`, `run_frame(frame, *, method="lmmse", ...) -> ReceiverOutput` (`h_est` / `h_var` allowed in `(14, 102)` or `(14, 128)`, normalized to active band via `extract_active_h`), plus dev calibration helpers (`calibrate_on_dev` / `calibrate_classical`) rejecting `"test"`/`"guard"` frames with `ValueError`.
- **M2 Evaluator (`eval.evaluator`)**:
  - `evaluate_receiver` / `evaluate_frames` / `evaluate_outputs(pairs)`, `compare_arms` / `paired_comparison` / `evaluate_paired(base_pairs, cand_pairs)` (returning `p_two_sided`, `ties`, and mandatory `"verdict"` in `{"支持", "反证", "证据不足"}` via `p12.audit.verdict_superiority`), `compare_multiple_arms`, `profile_receiver_pinned`, `run_stress_suite`, and `subblock_group_ids`, implementing `PROTOCOL.md` §5 observable held-out pilot formulas ($z_{\text{obs}} = y_p/p$, $v_{\text{eff}} = \hat{\sigma}^2_H + \text{est\_n0}/|p|^2$), exact integer payload counts, and engineering cost/coverage accounting.
- **M3 Group A (`receivers.group_a_reference`)**:
  - `run_frame(frame, context_pool=..., mode="local" | "global_uniform" | "global_reliability" | "global_content_attn", local_k=..., temperature=...) -> ReceiverOutput` with `aux_info["weights"]` and `aux_info["n_values_accessed"]`; optional `calibrate_group_a_temperature(dev_frames, pools) -> float`.
- **M3 Group B (`receivers.group_b_fusion`)**:
  - `run_frame(frame, mode=...)` and candidate-branch entry point (`fuse_candidate_branches` / `run_branches` / `run_frame(frame, *, branches=..., branch_weights=..., mode="hard_map" | "posterior_mean_plugin" | "exact_mixture_likelihood")`) implementing exact log-mixture $\log\sum_r \alpha_r g_r(c)$ (`nongray_16qam_mixture_llr_ref`).
- **M4 Group C (`receivers.group_c_trainable`)**:
  - `GroupCTrainableReceiver` / `GroupCReceiver(seed=...)` exposing `.fit(train_frames, dev_frames, epochs=...)` (rejecting `"test"` frames in both `train_frames` and `dev_frames`), `.run_frame(frame)`, `.save(path)`, and `@classmethod .load(path)`.
