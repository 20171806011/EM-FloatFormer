# Handoff Report: `explorer_m3_1` — E2E Test Contracts for Milestone 3 (Stage P4-B: Features F13–F17)

## 1. Observation

1. **Master E2E Baseline Run (`tests/test_p4_e2e.py`)**:
   - Command: `python -m unittest tests/test_p4_e2e.py` in ``.
   - Observed result: `Ran 253 tests in 17.108s`, `OK (skipped=97)` (156 passed, 97 skipped, 0 failures, 0 errors, exit code `0`).
2. **`MilestoneResolver` Behavior (`tests/e2e/harness.py:41-69`)**:
   - `get_module(module_name, required_milestone)` executes `importlib.import_module(f"p4.{module_name}")`.
   - `get_callable(module_name, func_name, required_milestone)` and `get_class(...)` check `hasattr(mod, func_name)` and return `getattr(mod, func_name)`, raising `unittest.SkipTest` on missing module or attribute.
   - All **65** Milestone 3 E2E tests resolve directly from `module_name="slot_2d"` (`p4.slot_2d`):
     - Tier 1 (`tests/e2e/test_tier1_feature_coverage.py:736-939`): 25 tests (`f13`–`f17`, 5 each).
     - Tier 2 (`tests/e2e/test_tier2_boundary_corner.py:631-826`): 25 tests (`f13`–`f17`, 5 each).
     - Tier 3 (`tests/e2e/test_tier3_cross_feature.py:109-115, 136-153, 236-287`): 8 tests (`P04`, `P06`, `P07`, `P15`, `P16`, `P17`, `P18`, `P19`).
     - Tier 4 (`tests/e2e/test_tier4_real_workloads.py:47-80, 97-128, 171-180`): 7 tests (`S01`, `S02`, `S03`, `S05`, `S06`, `S07`, `S11`). Note `S07` (`test_t4_s07_ood_channel_generalization_workload`) is explicitly tagged `"Milestone M3 (S07)"` and resolves `run_ood_generalization` from `slot_2d`.
3. **Exact `p4.slot_2d` Symbol Surface Required by E2E Tests (50 public symbols)**:
   - **F13**: `SlotSimulator2D` (attrs `.n_symb`, `.n_sc`, `.pilot_ratio`; method `.generate_empty_grid(batch_size)` -> `(B, n_symb, n_sc)`), `get_dmrs_mask(n_symb=14, n_sc=32)`, `get_slot_masks(n_symb=14, n_sc=32)`, `get_custom_dmrs_mask(pilot_coords=...)`, `generate_dmrs(batch_size=4, num_pilots=32)`.
   - **F14**: `simulate_slot_channel(batch_size=..., fd=..., cfo=..., return_h_time=...)`, `measure_ici_leakage(fd=0.05)`, `measure_cfo_phase_drift(cfo=...)`, `estimate_cfo_cp(cfo_true=0.03)`, `compute_doppler_spectrum(speed_kmh=500.0, fc=3.5e9)`, `check_phase_continuity()`.
   - **F15**: `LDPCMatrix(codeword_len=128)` (attr `.h_matrix` shape `(64, 128)`), `encode_ldpc(u)` -> `(c, h_mat)`, `decode_min_sum(llrs, h_mat, max_iter=..., return_iters=...)` -> `u_hat` or `(u_hat, iters)`, `correct_single_bit()`, `measure_coding_gain()`.
   - **F16**: `compute_2d_correlation(n_symb=14, n_sc=32, fd=0.03)` -> `(448, 448)`, `lmmse_2d_interpolate(batch_size=2, n_symb=14, n_sc=32)` -> `(2, 14, 32)`, `compare_1d_vs_2d_lmmse(fd=0.03)` -> `(nmse_1d, nmse_2d)`, `idd_pa_2d(batch_size=2, iterations=2)`, `idd_pa_2d_cleanup()`, `lmmse_2d_flat()`, `lmmse_2d_singular(noise_var=1e-12)`, `idd_pa_2d_linear()`, `idd_pa_2d_clipping(A=1.0)`, `check_edge_extrapolation()`.
   - **F17**: `check_block_error(u, u_hat)` -> `int`, `evaluate_bler_curve(snrs)` -> `List[float]`, `run_block_counter(num_blocks=10)` -> `(int, int)`, `check_doppler_bler_sensitivity()` -> `(bler_low_fd, bler_high_fd)`, `evaluate_ood_profile_bler(profile="sparse12")`, `evaluate_bler_point(snr_db=...)`, `simulate_single_block()`, `check_integer_accumulator(count=1_000_000)`, `compute_ci(errors=5, total=100)`.
   - **Tier 3**: `sim_multipath_cfo(fd=0.03, cfo=0.05)`, `decode_calibrated_ldpc(snr_db=15.0)`, `idd_pa_2d_slot(iterations=2)`, `sim_slot_doppler_cfo(fd=0.05, cfo=0.05)`, `estimate_slot_2d_lmmse()`, `eval_high_doppler_bler(fd_hz=600.0)`, `trace_bler_waterfall()`, `eval_ood_baselines_bler()`.
   - **Tier 4**: `run_hst_workload(speed_kmh=300.0, fc=3.0e9, pa_A=1.25, snr_db=15.0)`, `run_urban_macro_workload(n_sc=64, profile="long12", pa_A=2.25, num_slots=20, snr_db=18.0)`, `run_cfo_drift_workload(cfo=0.06, n_symb=14, snr_db=20.0)`, `run_ultra_sparse_workload(comb=4, pilot_symbol=2)`, `run_multi_impairment_stress()`, `run_ood_generalization()`, `run_mission_link(total_bits=100_000, num_slots=50, snr_db=15.0)`.
4. **Critical LLR Sign Convention Observed**:
   - `p4/classical_baselines.py:143` defines `P(b_i = 1 | z) = sigmoid(llr[..., i])`.
   - `tests/e2e/test_tier1_feature_coverage.py:842-844` passes `llrs = (2.0 * c.float() - 1.0) * 10.0` to `decode_min_sum(llrs, h_mat)` and asserts `torch.equal(u_hat, u)`. Thus `llr > 0` corresponds to bit `1` and `llr < 0` corresponds to bit `0`.
   - `p4/em_floatformer.py:651-653` uses `3.5e-2` variance floor and `20.0 * torch.tanh(llrs64 / 20.0)` soft-LLR clipping.
5. **Governance & Existing Exports**:
   - `p4/__init__.py:1-134` exports M1 and M2 symbols in `__all__`.
   - `tests/e2e/test_tier1_feature_coverage.py:1114-1164` and `tests/e2e/test_tier2_boundary_corner.py:1004-1031` enforce `MANIFEST_SHA256.json` byte-exact immutability, forbid stray root files/dirs outside `p4/` and `tests/`, and verify `p4/results/` exists.

## 2. Logic Chain

1. Because `MilestoneResolver` directly imports `p4.slot_2d` and resolves all 50 M3 callables/classes by exact attribute name, implementing `p4/slot_2d.py` with all 50 exact signatures and return structures is both necessary and sufficient to unlock all 65 M3 E2E tests (leaving only the 32 M4 tests skipped).
2. Because `test_t1_f15_min_sum_soft_ldpc_decoding` feeds `(2 * c - 1) * 10.0` into `decode_min_sum`, any Min-Sum implementation using the opposite sign convention (`log P(0)/P(1)`) would invert every decoded bit; negating at the Min-Sum boundary or formulating Min-Sum directly with `llr > 0 => 1` guarantees compatibility with both `calibrated_16qam_llrs` and `EMFloatFormer`.
3. Because `S07` evaluates an unseen 16-tap exponentially decaying channel (delays `0..15`), while `p4/simulator.py:30-36` `PROFILES` only defines up to 12 taps (`long12`, `sparse12`), `SlotSimulator2D` must support 16-tap exponential PDPs internally while keeping `cp >= 16` (`DEFAULT_CP = 16` satisfies `15 <= 16`).
4. Because `T1.F17.3`, `T2.F17.4`, `S01`, and `S11` explicitly assert `isinstance(..., int)` for error/block counters, those counters must be cast to Python `int` (not `torch.Tensor` or `np.int64`).
5. Detailed per-symbol contracts, shapes, dtypes, and edge cases are documented in `explorer_m3_1/analysis.md`.

## 3. Caveats

- Only E2E test files (`tests/test_p4_e2e.py`, `tests/e2e/*`), `p4/__init__.py`, `p4/simulator.py`, `p4/classical_baselines.py` (lines 1-466), `p4/em_floatformer.py` (lines 560-690), and `TEST_READY.md` were read in detail; `p4/tests/test_p4_0_simulator.py`, `p4/tests/test_p4_a_equivariance.py`, and `p4/results/p4a_comparative_benchmark.*` were not inspected (delegated to peer M3 explorers).

## 4. Conclusion

- Full contract specification for Milestone 3 (`p4/slot_2d.py` + `p4/__init__.py` exports + `p4/results/p4b_coded_bler.{json,csv}`) is extracted and verified against all 6 E2E files in `explorer_m3_1/analysis.md`.
- Every M3 E2E test (25 Tier 1 + 25 Tier 2 + 8 Tier 3 + 7 Tier 4 = **65 tests**) maps to exact function/class names, arguments, shapes, dtypes, and quantitative invariants in `p4.slot_2d`.
- Implementing these 50 symbols cleanly (with vectorized LDPC Min-Sum, `llr > 0 => 1` sign convention, continuous sample-level CFO/Doppler, 16-tap OOD PDP support, and `EMFloatFormer2D` with $C^*$ equivariance) will bring E2E pass count from `156/253` to `221/253` (0 failures, 0 errors, only 32 M4 skips remaining).

## 5. Verification Method

1. Inspect the full contract matrix at `explorer_m3_1/analysis.md`.
2. Re-run the baseline E2E suite before M3 edits:
   - `python -m unittest tests/test_p4_e2e.py` -> expected `Ran 253 tests`, `OK (skipped=97)`.
3. After implementing M3 (`p4/slot_2d.py`, `p4/__init__.py`, `p4/tests/test_p4_b_2d_fec.py`), verify:
   - `python -m unittest tests/test_p4_e2e.py -v` -> expected `Ran 253 tests`, `OK (skipped=32)` (0 failures, 0 errors, all 65 M3 tests + all 156 M1/M2/F22 tests passing).
   - `python -m unittest -k "f13 or f14 or f15 or f16 or f17" tests/test_p4_e2e.py -v` -> all 50 Tier 1+2 M3 tests pass with 0 skips.
   - `python -m unittest p4/tests/test_p4_0_simulator.py p4/tests/test_p4_a_equivariance.py p4/tests/test_p4_b_2d_fec.py -v` -> exit code 0.
   - Invalidation condition: if any M3 E2E test skips, fails, or errors, check `analysis.md` Section 2 against `p4/slot_2d.py` for missing attribute/argument or LLR sign mismatch.
