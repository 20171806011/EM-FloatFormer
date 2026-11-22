# EM-FloatFormer Real-Data SDR OFDM: Test Readiness Attestation

**Document**: `research/realdata_sdr_ofdm/tests/TEST_READY.md`  
**Status**: **READY — M1+M2 LIVE VERIFIED (`79 passed, 53 skipped, 0 failed, 0 errors`, Exit Code `0`)**  
**Repository**: `EM-FloatFormer` (branch `research/realdata-sdr-ofdm`)  
**Master Test Runner**: `research/realdata_sdr_ofdm/tests/test_realdata_e2e.py`  
**Total Discovered Test Cases**: Exactly **132**  
**Timestamp (UTC)**: 2026-11-21T23:50:00Z

---

## 1. Master Runner Commands & Verification Scope

1. **`unittest` full suite** (executed with `KMP_DUPLICATE_LIB_OK` unset prior to launch):
   ```bash
   python -B -m unittest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
   - **Verified Result**: `Ran 132 tests in 55.858s` — `OK (skipped=53)` (**79 passed, 53 skipped, 0 failed, 0 errors**, exit code `0`). All 79 active M1+M2 tests pass; all 53 M3–M5 tests skip cleanly via `MilestoneResolver`.
2. **`pytest` full suite** (executed with `KMP_DUPLICATE_LIB_OK` unset prior to launch):
   ```bash
   python -B -m pytest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
   - **Verified Result**: `132 collected` — `79 passed, 53 skipped in 57.88s` (**0 failed, 0 errors**, exit code `0`).
3. **Feature-filtered `unittest` execution** (executed with `KMP_DUPLICATE_LIB_OK` unset prior to launch):
   ```bash
   python -B -m unittest -k "f01" research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
   ```
   - **Verified Result**: `Ran 11 tests in 2.283s` — `OK` (exit code `0`).

Both `unittest` and `pytest` discover the exact same **132** opaque-box tests across Tiers 1–4 (`test_realdata_e2e.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically on `win32` before importing `torch`).

---

## 2. Iteration 3 Hardening & M5 Gating Summary

- **Iteration 3 Test Fixes Verified**:
  - **`T1-F6.2`** (`e2e/test_tier1_feature_coverage.py`): Dual-regime high-SNR sanity check (Regime A `flat_channel=True` with non-trivial `phase_offset` & `gauge_scale=2.5*exp(0.7j)` at 30 dB + Regime B `n_taps=4` multipath bounded above Perfect-CSI Oracle floor). **Live verified: PASS**.
  - **`T3-P05`** (`e2e/test_tier3_cross_feature.py`): Dual-regime non-Gray `mapping_table(4)` alignment vs Gray demapping and inverted LLR polarity. **Live verified: PASS**.
  - **`T3-P08`** (`e2e/test_tier3_cross_feature.py`): `extra_awgn_db` (`added_awgn_snr_db`) sweep `(None, 12.0, 6.0)` with strict EVM monotonicity and non-decreasing BER. **Live verified: PASS**.
  - **`T4-S03`** (`e2e/test_tier4_real_workloads.py`): Multi-axis physical stress sweep `(None, 12.0, 6.0)` with strict monotonicity `evms_c[0] < evms_c[1] < evms_c[2]` (presently skipped on M3 `group_a_reference`).
  - **`T1-F11.1`, `T2-F11.2`, `T2-F11.3` M5 Gating Fix**: Added `MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")` immediately before `MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)` in `e2e/test_tier1_feature_coverage.py` (`T1-F11.1`) and `e2e/test_tier2_boundary_corner.py` (`T2-F11.2`, `T2-F11.3`). Because M2 (`scripts/run_baselines.py`) legitimately populates `results/baselines/`, gating on `REPORT.md` ensures all F11 tests stay progressively skipped until M5 completes. **Live verified: clean `SKIPPED` (`0` premature failures)**.
  - **`TEST_INFRA.md`**: Documents `extra_awgn_db` sign convention (§5) and concrete M2 `receivers.classical` & `eval.evaluator` symbols (§6).

---

## 3. 4-Tier Coverage & Progressive Execution Summary

| Tier | Tier Name | Test File | Scope | Count | Current Progressive Status |
|:----:|:----------|:----------|:------|:-----:|:--------------------------|
| **Tier 1** | **Feature Coverage** | `e2e/test_tier1_feature_coverage.py` | 5 isolated tests per feature (F1–F11) | **55** | 35 Passed (M1 F1–F5 + M2 F6–F7), 20 Skipped (M3–M5 F8–F11), 0 Failed |
| **Tier 2** | **Boundary & Corner Cases** | `e2e/test_tier2_boundary_corner.py` | 5 boundary/corner tests per feature (F1–F11) | **55** | 35 Passed (M1 F1–F5 + M2 F6–F7), 20 Skipped (M3–M5 F8–F11), 0 Failed |
| **Tier 3** | **Cross-Feature Pairwise** | `e2e/test_tier3_cross_feature.py` | 15 pairwise seams (`T3-P01`..`T3-P15`) | **15** | 8 Passed (`P01–P08`), 7 Skipped (`P09–P15`), 0 Failed |
| **Tier 4** | **Real-World Workloads** | `e2e/test_tier4_real_workloads.py` | 7 end-to-end workloads (`T4-S01`..`T4-S07`) | **7** | 1 Passed (`S06`), 6 Skipped (`S01–S05`, `S07`), 0 Failed |
| **Total** | **All 4 Tiers** | `test_realdata_e2e.py` | **Full E2E Suite (`F12`)** | **132** | **79 Passed, 53 Skipped, 0 Failed, 0 Errors (Exit 0)** |

---

## 4. Feature Allocation Matrix (F1–F11)

| # | Feature | Target Milestone | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Status at M1+M2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **F1** | Git Sync & Branching | M1 | 5 | 5 | P01 | S07 | Live Passed (11/11 active + S07 gated on M5) |
| **F2** | Data Source Catalog | M1 | 5 | 5 | P01, P02 | S07 | Live Passed |
| **F3** | Remote Acquisition & Probe | M1 | 5 | 5 | P02, P03 | S01 | Live Passed |
| **F4** | Unified Data Adapter & Non-Gray 16-QAM | M1 | 5 | 5 | P03, P04, P05, P08 | S01, S03, S05 | Live Passed (M1+M2 tests) |
| **F5** | Leak-Free Partition & Protocol | M1 | 5 | 5 | P04, P06 | S07 | Live Passed (M1+M2 tests) |
| **F6** | Classical Receiver Baseline | M2 | 5 | 5 | P05–P07, P09, P12 | S01–S03, S05 | Live Passed (all 13 active M1+M2 tests) |
| **F7** | Unified Evaluator & Stress | M2 | 5 | 5 | P07, P08, P10, P14, P15 | S01, S02, S06 | Live Passed (all 13 active M1+M2 tests) |
| **F8** | Group A: Reference & Content Attn | M3 | 5 | 5 | P09–P11 | S01, S03–S05 | Primed (`SkipTest` via `MilestoneResolver`) |
| **F9** | Group B: Hard vs Posterior Fusion | M3 | 5 | 5 | P11–P13 | S01–S05 | Primed (`SkipTest` via `MilestoneResolver`) |
| **F10** | Group C: Lightweight Trainable | M4 | 5 | 5 | P13, P14 | S01, S05 | Primed (`SkipTest` via `MilestoneResolver`) |
| **F11** | Full Codespace Run & Delivery | M5 | 5 | 5 | P15 | S07 | Primed (`SkipTest` via `MilestoneResolver` on `REPORT.md` / `figures/`) |

---

## 5. Sign-Off Criteria

1. All **79** active M1+M2 E2E tests (`F1–F7`) pass with `0` failures and `0` errors across both `unittest` and `pytest` (`79 passed, 53 skipped, 0 failed, 0 errors`, exit code `0`).
2. `python -B -m unittest -k "f01" research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v` exits `0` running the `11` F1-tagged tests (`Ran 11 tests ... OK`).
3. Once M3–M5 deliver `receivers/group_a_reference.py`, `receivers/group_b_fusion.py`, `receivers/group_c_trainable.py`, top-level `results/*summary*.json`, `figures/`, and `REPORT.md`, all 132 tests automatically un-skip and must achieve `132 passed, 0 skipped, 0 failed, 0 errors`.
