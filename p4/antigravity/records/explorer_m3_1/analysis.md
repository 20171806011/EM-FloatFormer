# E2E Test Contract Analysis for Milestone 3 (Stage P4-B: Features F13–F17)

**Author**: `explorer_m3_1`
**Date**: 2026-11-17T20:23:00Z
**Scope**: Complete extraction of every API contract, signature, tensor shape, dtype, mathematical bound, invariant, and edge case from `tests/test_p4_e2e.py`, `tests/e2e/harness.py`, and Tiers 1–4 (`tests/e2e/test_tier1_feature_coverage.py`, `test_tier2_boundary_corner.py`, `test_tier3_cross_feature.py`, `test_tier4_real_workloads.py`).

---

## 1. Baseline E2E Execution & `MilestoneResolver` Mechanics

### 1.1 Current Baseline Execution (`python -m unittest tests/test_p4_e2e.py`)
- Executed in ``:
  - `Ran 253 tests in 17.108s` -> `OK (skipped=97)` (`156 passed, 97 skipped, 0 failures, 0 errors`, exit code `0`).
- **M3 Test Distribution (65 total M3 tests, all currently skipped)**:
  - Tier 1 (`TestTier1FeatureCoverage`): 25 tests (`test_t1_f13_*` .. `test_t1_f17_*`, 5 per feature)
  - Tier 2 (`TestTier2BoundaryCorner`): 25 tests (`test_t2_f13_*` .. `test_t2_f17_*`, 5 per feature)
  - Tier 3 (`TestTier3CrossFeature`): 8 tests (`P04`, `P06`, `P07`, `P15`, `P16`, `P17`, `P18`, `P19`)
  - Tier 4 (`TestTier4RealWorkloads`): 7 tests (`S01`, `S02`, `S03`, `S05`, `S06`, `S07`, `S11`)
  - Remaining 32 skips belong to Milestone M4 (`p4.ablation_probing`, `p4.confirmatory_protocol`).

### 1.2 `MilestoneResolver` & Package Exports
- `tests/e2e/harness.py:41-69`:
  - `MilestoneResolver.get_module(module_name, required_milestone)` executes `importlib.import_module(f"p4.{module_name}")`.
  - `MilestoneResolver.get_callable(module_name, func_name, required_milestone)` checks `hasattr(mod, func_name)` and returns `getattr(mod, func_name)`.
  - `MilestoneResolver.get_class(module_name, class_name, required_milestone)` delegates to `get_callable`.
- **Direct M3 Target Module**: ALL 65 M3 E2E tests resolve directly from `p4.slot_2d` (`module_name="slot_2d"`).
- **`p4/__init__.py`**: Currently exports all M1 (`simulator`, `classical_baselines`) and M2 (`em_floatformer`) symbols in `__all__` (`p4/__init__.py:1-134`). Should re-export all public M3 classes and functions from `p4.slot_2d` in `__all__` for package consistency (per `SCOPE.md`).
- **Governance & Isolation Constraints** (`test_t1_f22_*`, `test_t2_f22_*`):
  - All new code must live strictly under `p4/` or `tests/` (never stray `.py` files or unrecognized dirs in `PROJECT_ROOT`).
  - Result artifacts (`p4/results/p4b_coded_bler.json`, `p4/results/p4b_coded_bler.csv`) must be placed under `p4/results/`.
  - `MANIFEST_SHA256.json` historical files, `exploratory/`, M1/M2 modules, and `tests/e2e/*` must remain untouched.

---

## 2. Complete Symbol & Signature Matrix for `p4.slot_2d` (50 Public Symbols)

### 2.1 Feature F13: 2D Slot Simulator & Sparse DMRS (5 Tier 1 + 5 Tier 2 tests)

| Symbol | Kind | Exact Call Sites in E2E Tests | Expected Parameters / Defaults | Return Type / Shape / Dtype / Assertions |
|---|---|---|---|---|
| `SlotSimulator2D` | Class | T1.F13.1: `SlotSimulator2D(n_symb=14, n_sc=32)`<br>T2.F13.1: `SlotSimulator2D(n_symb=1, n_sc=32)`<br>T2.F13.2: `SlotSimulator2D(n_symb=28, n_sc=32)`<br>T2.F13.3: `SlotSimulator2D(n_symb=14, n_sc=128)`<br>T2.F13.5: `SlotSimulator2D(n_symb=14, n_sc=32, pilot_ratio=1.0)` | `n_symb: int = 14, n_sc: int = 32, cp: int = 16, profile: str = "equal4", fd: float = 0.0, cfo: float = 0.0, pilot_ratio: Optional[float] = None, ...` | Attributes: `.n_symb`, `.n_sc`, `.pilot_ratio`.<br>Method `generate_empty_grid(batch_size: int) -> torch.Tensor` returning `torch.complex64` zeros of shape `(batch_size, n_symb, n_sc)`. Must handle `pilot_ratio=1.0` (empty payload mask) without crash. |
| `get_dmrs_mask` | Func | T1.F13.2: `get_dmrs_mask(n_symb=14, n_sc=32)`<br>T1.F13.3: `get_dmrs_mask(n_symb=14, n_sc=32)` | `n_symb: int = 14, n_sc: int = 32, ...` | `torch.Tensor` (`dtype=torch.bool`, `shape=(n_symb, n_sc)`).<br>For `(14, 32)`: `mask[2].any() == True`, `mask[11].any() == True`, `mask[0].any() == False`; `float(mask.sum()) / float(mask.numel()) <= 0.10` (comb-2 on symbols 2 & 11 yields `32/448 = 7.14%`). |
| `get_slot_masks` | Func | T1.F13.4: `get_slot_masks(n_symb=14, n_sc=32)` | `n_symb: int = 14, n_sc: int = 32, ...` | `Tuple[torch.Tensor, torch.Tensor]` (`(pilot_mask, data_mask)`), both `torch.bool` of shape `(n_symb, n_sc)`. Must satisfy `not (pilot_mask & data_mask).any()` and `(pilot_mask | data_mask).all()`. |
| `get_custom_dmrs_mask` | Func | T2.F13.4: `get_custom_dmrs_mask(pilot_coords=[(2, 0)])` | `pilot_coords: Sequence[Tuple[int, int]], n_symb: int = 14, n_sc: int = 32` | `torch.Tensor` (`dtype=torch.bool`, `shape=(n_symb, n_sc)`). For `[(2, 0)]`, `int(mask.sum()) == 1` and `mask[2, 0] == True`. |
| `generate_dmrs` | Func | T1.F13.5: `generate_dmrs(batch_size=4, num_pilots=32)` | `batch_size: int, num_pilots: int, seed: Optional[int] = ..., generator: Optional[torch.Generator] = None` | `torch.Tensor` (`dtype=torch.complex64`, `shape=(batch_size, num_pilots)`). Pseudo-random Gold-sequence QPSK with unit energy: `torch.allclose(pilots.abs() ** 2, torch.ones_like(pilots.abs()), atol=1e-5)`. |

### 2.2 Feature F14: Continuous Doppler ICI & CFO (5 Tier 1 + 5 Tier 2 tests)

| Symbol | Kind | Exact Call Sites in E2E Tests | Expected Parameters / Defaults | Return Type / Shape / Dtype / Assertions |
|---|---|---|---|---|
| `simulate_slot_channel` | Func | T1.F14.1: `simulate_slot_channel(batch_size=2, fd=0.03, cfo=0.0)`<br>T1.F14.5: `simulate_slot_channel(batch_size=2, fd=0.03, cfo=0.03)`<br>T2.F14.1: `simulate_slot_channel(batch_size=1, fd=0.0, cfo=0.0, return_h_time=True)`<br>T2.F14.2: `simulate_slot_channel(batch_size=1, cfo=0.5)` | `batch_size: int = 1, n_symb: int = 14, n_sc: int = 32, cp: int = 16, profile: str = "equal4", fd: float = 0.0, cfo: float = 0.0, snr_db: Optional[float] = None, return_h_time: bool = False, generator: Optional[torch.Generator] = None` | Default (`return_h_time=False`): `torch.complex64` tensor of shape `(batch_size, n_symb, n_sc)` with zero NaNs.<br>When `return_h_time=True`: returns time-varying channel tensor with symbol/time axis at dim 1 (e.g. `(batch_size, n_symb, ...)`). When `fd=0.0, cfo=0.0`: `(h_evol[:, 1:] - h_evol[:, :-1]).abs().max() < 1e-6`. |
| `measure_ici_leakage` | Func | T1.F14.2: `measure_ici_leakage(fd=0.05)` | `fd: float = 0.05, n_sc: int = 32, cp: int = 16, ...` | `float > 1e-4` measuring off-diagonal frequency leakage / ICI power ratio under continuous Doppler. |
| `measure_cfo_phase_drift` | Func | T1.F14.3: `measure_cfo_phase_drift(cfo=0.04)`<br>T2.F14.3: `measure_cfo_phase_drift(cfo=-0.04)` | `cfo: float = 0.04, ...` | `float` within `±0.005` (`places=2`) of `cfo` (supports both positive and negative `cfo`, recovering normalized CFO from continuous sample-level phase accumulation $\theta[n] = 2\pi \epsilon n / N$). |
| `estimate_cfo_cp` | Func | T1.F14.4: `estimate_cfo_cp(cfo_true=0.03)` | `cfo_true: float = 0.03, ...` (or optional `rx_time` tensor + `cfo_true` fallback) | `float` within `±0.005` (`places=2`) of `cfo_true`. Uses CP correlation $\hat{\epsilon} = \frac{1}{2\pi} \arg(\sum y^*[m] y[m+N])$. |
| `compute_doppler_spectrum` | Func | T2.F14.4: `compute_doppler_spectrum(speed_kmh=500.0, fc=3.5e9)` | `speed_kmh: float = 500.0, fc: float = 3.5e9, ...` | Non-`None` (e.g. dict or tensor containing `fd_hz`, normalized `fd`, Jakes spectrum $S(f) = (\pi f_d \sqrt{1 - (f/f_d)^2})^{-1}$). |
| `check_phase_continuity` | Func | T2.F14.5: `check_phase_continuity()` | `cfo: float = 0.04, n_symb: int = 14, n_sc: int = 32, cp: int = 16` | `bool` (`True`), verifying global sample index $n \in [0, N_{\text{symb}}(N+N_{\text{cp}})-1]$ produces continuous phase evolution across symbol+CP boundaries with zero discontinuous jumps. |

### 2.3 Feature F15: Native Outer Channel Coding (Rate-1/2 LDPC) (5 Tier 1 + 5 Tier 2 tests)

| Symbol | Kind | Exact Call Sites in E2E Tests | Expected Parameters / Defaults | Return Type / Shape / Dtype / Assertions |
|---|---|---|---|---|
| `LDPCMatrix` | Class | T1.F15.1: `LDPCMatrix(codeword_len=128)` | `codeword_len: int = 128, rate: float = 0.5, seed: int = ...` | Attribute `.h_matrix`: binary `{0, 1}` tensor of shape `(64, 128)` (`codeword_len // 2, codeword_len`). Must support systematic/structured encoding with $H G^T \equiv 0 \pmod 2$. |
| `encode_ldpc` | Func | T1.F15.2: `c, h_mat = encode_ldpc(u)` (`u` shape `(4, 64)`)<br>T1.F15.3/4: `encode_ldpc(u)` (`(2, 64)`, `(1, 64)`)<br>T2.F15.1: `encode_ldpc(torch.zeros(1, 64, dtype=torch.int64))`<br>T2.F15.2: `encode_ldpc(torch.ones(1, 64, dtype=torch.int64))`<br>T2.F15.4: `encode_ldpc(torch.zeros(1, 64))` (float)<br>T2.F15.5: `encode_ldpc(torch.empty(0, 0))` | `u: torch.Tensor` | Raises `ValueError` (or `RuntimeError`/`AssertionError`) if `u.numel() == 0` or `u.shape[-1] == 0`.<br>Returns `(c, h_mat)`: `c` has shape `(B, 2*K)` (`int64` binary `{0, 1}`), `h_mat` has shape `(K, 2*K)`.<br>Must satisfy `torch.remainder(c.float() @ h_mat.t().float(), 2.0) == 0`, and `u == 0` -> `c == 0`. |
| `decode_min_sum` | Func | T1.F15.3: `u_hat = decode_min_sum(llrs, h_mat)`<br>T1.F15.4: `_, iters = decode_min_sum(llrs, h_mat, return_iters=True)`<br>T2.F15.4: `_, iters = decode_min_sum(noise_llrs, h_mat, max_iter=15, return_iters=True)` | `llrs: torch.Tensor, h_mat: torch.Tensor, max_iter: int = 20, return_iters: bool = False, alpha: float = 0.75` | **CRITICAL LLR SIGN**: Tests pass `llrs = (2.0 * c.float() - 1.0) * 10.0`, meaning **`llr > 0` => bit `1`, `llr < 0` => bit `0`** (matching `calibrated_16qam_llrs`).<br>Returns `u_hat` (`int64`, shape `(B, K)`) when `return_iters=False`; returns `(u_hat, iters: int)` when `return_iters=True`.<br>Clean LLRs -> `iters == 1` and `torch.equal(u_hat, u)`; unconverged noise LLRs -> `iters == max_iter` (`15`). |
| `correct_single_bit` | Func | T2.F15.3: `recovered = correct_single_bit()` | `() -> bool` | Returns `True`, verifying Min-Sum corrects a single flipped bit in 1 iteration. |
| `measure_coding_gain` | Func | T1.F15.5: `gain_db = measure_coding_gain()` | `() -> float` | Returns `float > 3.0` (dB SNR coding gain of Rate-1/2 LDPC soft decoding over uncoded at target error rate). |

### 2.4 Feature F16: 2D Classical Baselines (5 Tier 1 + 5 Tier 2 tests)

| Symbol | Kind | Exact Call Sites in E2E Tests | Expected Parameters / Defaults | Return Type / Shape / Dtype / Assertions |
|---|---|---|---|---|
| `compute_2d_correlation` | Func | T1.F16.1: `compute_2d_correlation(n_symb=14, n_sc=32, fd=0.03)` | `n_symb: int = 14, n_sc: int = 32, fd: float = 0.03, cp: int = 16, profile: str = "equal4"` | Tensor of shape `(n_symb * n_sc, n_symb * n_sc)` = `(448, 448)`, representing $R_{\text{time}} \otimes R_{\text{freq}}$. |
| `lmmse_2d_interpolate` | Func | T1.F16.2: `lmmse_2d_interpolate(batch_size=2, n_symb=14, n_sc=32)` | `batch_size: int = 2, n_symb: int = 14, n_sc: int = 32, ...` | `torch.complex64` tensor of shape `(batch_size, n_symb, n_sc)` estimated from sparse DMRS REs. |
| `compare_1d_vs_2d_lmmse` | Func | T1.F16.3: `compare_1d_vs_2d_lmmse(fd=0.03)` | `fd: float = 0.03, ...` | `(nmse_1d: float, nmse_2d: float)` with `nmse_2d < nmse_1d`. |
| `idd_pa_2d` | Func | T1.F16.4: `idd_pa_2d(batch_size=2, iterations=2)` | `batch_size: int = 2, iterations: int = 2, ...` | Non-`None` result (dict or dataclass with `h_hat`, `llrs`, `hard_bits`, `history`, etc.). |
| `idd_pa_2d_cleanup` | Func | T1.F16.5: `idd_pa_2d_cleanup()` | `() -> bool` | `True`, verifying 2D IDD-PA improves LLR/MSE accuracy on payload REs under PA distortion. |
| `lmmse_2d_flat` | Func | T2.F16.1: `lmmse_2d_flat()` | `() -> torch.Tensor` | Tensor of shape `(B, N_symb, N_sc)` under static `fd=0` AWGN satisfying `(h_est[:, 1:] - h_est[:, :-1]).abs().max() < 1e-4`. |
| `lmmse_2d_singular` | Func | T2.F16.2: `lmmse_2d_singular(noise_var=1e-12)` | `noise_var: float = 1e-12` | `bool` (`True`), verifying covariance regularization prevents numerical singularity as `noise_var -> 0`. |
| `idd_pa_2d_linear` | Func | T2.F16.3: `idd_pa_2d_linear()` | `() -> float` | `float` distortion power `~ 0.0` (`places=5`) under linear PA (`A=None, alpha=0.0`). |
| `idd_pa_2d_clipping` | Func | T2.F16.4: `idd_pa_2d_clipping(A=1.0)` | `A: float = 1.0` | `bool` (`True`), verifying stability / non-divergence under severe clipping `A=1.0`. |
| `check_edge_extrapolation` | Func | T2.F16.5: `check_edge_extrapolation()` | `() -> float` | `float < 0.15`, NMSE of 2D-LMMSE channel extrapolation at slot edges (symbols 0 and 13). |

### 2.5 Feature F17: Coded BLER Evaluation (5 Tier 1 + 5 Tier 2 tests)

| Symbol | Kind | Exact Call Sites in E2E Tests | Expected Parameters / Defaults | Return Type / Shape / Dtype / Assertions |
|---|---|---|---|---|
| `check_block_error` | Func | T1.F17.1: `check_block_error(u, u_correct)` / `check_block_error(u, u_wrong)` | `u: torch.Tensor, u_hat: torch.Tensor` | `int` (`0` if all bits equal, `1` if any bit differs). |
| `evaluate_bler_curve` | Func | T1.F17.2: `evaluate_bler_curve(snrs)` (`snrs = [5.0, 10.0, 15.0]`) | `snrs: Sequence[float]` | `List[float]` of length `len(snrs)` with `blers[i] >= blers[i + 1]` (use common random numbers / shared seed for monotonic SNR response). |
| `run_block_counter` | Func | T1.F17.3: `errs, total = run_block_counter(num_blocks=10)` | `num_blocks: int = 10, ...` | `(int, int)` — **strictly Python `int`**, not `np.int64` or `torch.Tensor`. |
| `check_doppler_bler_sensitivity` | Func | T1.F17.4: `bler_low_fd, bler_high_fd = check_doppler_bler_sensitivity()` | `() -> Tuple[float, float]` | `(bler_low_fd, bler_high_fd)` with `bler_low_fd <= bler_high_fd`. |
| `evaluate_ood_profile_bler` | Func | T1.F17.5: `evaluate_ood_profile_bler(profile="sparse12")` | `profile: str = "sparse12", ...` | `float >= 0.0`. |
| `evaluate_bler_point` | Func | T2.F17.1: `evaluate_bler_point(snr_db=40.0)`<br>T2.F17.2: `evaluate_bler_point(snr_db=-15.0)` | `snr_db: float, ...` | `float`: strictly `0.0` at `40.0` dB, strictly `1.0` at `-15.0` dB. |
| `simulate_single_block` | Func | T2.F17.3: `simulate_single_block()` | `(...) -> int` | Python `int` in `{0, 1}`. |
| `check_integer_accumulator` | Func | T2.F17.4: `check_integer_accumulator(count=1_000_000)` | `count: int = 1_000_000` | Python `int` equal to `count` (`1_000_000`). |
| `compute_ci` | Func | T2.F17.5: `low, high = compute_ci(errors=5, total=100)` | `errors: int, total: int, alpha: float = 0.05` | `(low: float, high: float)` Clopper-Pearson interval satisfying `0.0 <= low < high <= 1.0`. |

### 2.6 Tier 3 Cross-Feature Callables in `p4.slot_2d` (8 tests)

| Test ID | Pair | Callable & Exact Invocation | Return & Assertion |
|---|---|---|---|
| `P04` | F2 + F14 | `sim_multipath_cfo(fd=0.03, cfo=0.05)` | Tensor, `not torch.isnan(slot).any()` |
| `P06` | F5 + F15 | `decode_calibrated_ldpc(snr_db=15.0)` | `float` BER `< 0.05` (calibrated 16-QAM soft LLRs -> Rate-1/2 LDPC Min-Sum decoder) |
| `P07` | F6 + F16 | `idd_pa_2d_slot(iterations=2)` | `Dict` with `res["converged"] == True` |
| `P15` | F13 + F14 | `sim_slot_doppler_cfo(fd=0.05, cfo=0.05)` | Tensor of shape `(14, 32)` |
| `P16` | F13 + F16 | `estimate_slot_2d_lmmse()` | Tensor of shape `(14, 32)` |
| `P17` | F14 + F17 | `eval_high_doppler_bler(fd_hz=600.0)` | `float` BLER `< 0.5` |
| `P18` | F15 + F17 | `trace_bler_waterfall()` | Sequence/list of length `5` (SNR grid `[0, 5, 10, 15, 20]` dB) |
| `P19` | F16 + F17 | `eval_ood_baselines_bler()` | `(bler_lmmse: float, bler_idd: float)`, both `>= 0.0` (on held-out `sparse12` profile) |

### 2.7 Tier 4 Real-World Workload Callables in `p4.slot_2d` (7 tests)

| Test ID | Scenario | Callable & Exact Invocation | Return Dict Keys & Assertions |
|---|---|---|---|
| `S01` | High-Speed Train | `run_hst_workload(speed_kmh=300.0, fc=3.0e9, pa_A=1.25, snr_db=15.0)` | `res["bler"] < 0.05`, `res["nonfinite_count"] == 0`, `isinstance(res["bit_errors"], int)` |
| `S02` | Urban Macrocell | `run_urban_macro_workload(n_sc=64, profile="long12", pa_A=2.25, num_slots=20, snr_db=18.0)` | `res["nmse"] < 0.05`, `res["bler"] < 0.01` |
| `S03` | Extreme CFO Drift | `run_cfo_drift_workload(cfo=0.06, n_symb=14, snr_db=20.0)` | `res["residual_phase_deg"] < 3.0`, `res["quadrant_flips"] == 0` |
| `S05` | Ultra-Sparse DMRS | `run_ultra_sparse_workload(comb=4, pilot_symbol=2)` | `res["stable"] == True`, `res["gain_over_ls_db"] >= 4.0` (overhead ~1.78% <= 5%) |
| `S06` | Multi-Impairment Stress | `run_multi_impairment_stress()` | `res["outputs_finite"] == True`, `res["p_value"] < 0.05` |
| `S07` | OOD Delay Generalization | `run_ood_generalization()` | `res["shortcut_collapse"] == False`, `res["ber_gap"] <= 0.005` (unseen 16-tap exponential channel) |
| `S11` | Mission-Critical Coded Link | `run_mission_link(total_bits=100_000, num_slots=50, snr_db=15.0)` | `res["bler"] < 0.01`, `isinstance(res["total_bit_errors"], int)`, `res["exit_code"] == 0` |

---

## 3. Physical Invariants, Cross-Milestone Interactions & Subtle Edge Cases

1. **LLR Sign & Calibration Contract (`p4/classical_baselines.py` + `p4/em_floatformer.py` ↔ `p4/slot_2d.py`)**:
   - In `calibrated_16qam_llrs`: `P(b_i = 1 | z) = sigmoid(llr)` -> **`llr > 0` denotes bit `1`**, **`llr < 0` denotes bit `0`**.
   - Confirmed by `test_t1_f15_min_sum_soft_ldpc_decoding`: `llrs = (2.0 * c.float() - 1.0) * 10.0`.
   - `decode_min_sum` MUST adhere to this exact sign convention (`c_hat = (llr_post > 0)`).
   - `EMFloatFormer` preserves the `3.5e-2` variance floor and `20.0 * torch.tanh(llrs64 / 20.0)` soft-LLR saturation bound (`p4/em_floatformer.py:651-653`) so high-SNR soft LLRs remain unclipped for Rate-1/2 LDPC soft decoding.
2. **2D Channel Profiles & CP Sizing**:
   - `p4/simulator.py:30-36` defines `PROFILES` up to 12 taps (`long12`, `sparse12`, max delay 11).
   - `S07` requires an **unseen 16-tap exponentially decaying channel** (delays `0..15`), so `SlotSimulator2D` must support 16-tap exponential PDPs cleanly with `cp >= 16` (`DEFAULT_CP = 16` suffices since `15 <= 16`).
3. **Sample-Level CFO Continuity & Sign**:
   - Continuous CFO phase must evolve using global sample index `n` across the entire slot (including CP samples), not reset per OFDM symbol (`T2.F14.5`).
   - Negative CFO (`cfo=-0.04`) must produce negative drift (`T2.F14.3`), and `estimate_cfo_cp` relies on CP correlation across `N_sc` sample lag ($\hat{\epsilon} = \frac{1}{2\pi} \arg(\sum y^*[m] y[m+N])$), valid for $|\epsilon| < 0.5$ (`T2.F14.2` tests `cfo=0.5` finiteness).
4. **DMRS Sparsity & Edge Cases**:
   - Default `get_dmrs_mask(14, 32)` and `(14, 64)`: comb-2 on symbols `[2, 11]` = `7.14% <= 10%` overhead.
   - `T2.F13.1/2/3/4/5` test `n_symb=1`, `n_symb=28`, `n_sc=128`, single pilot RE `[(2, 0)]`, and `pilot_ratio=1.0` (0 data REs).
   - `S05` tests comb-4 on single symbol `2` (`8/448 = 1.7857% <= 5%`).
5. **M3 Deliverables Outside E2E Tests (`SCOPE.md`)**:
   - 2D EM-FloatFormer model (`EMFloatFormer2D`) preserving strict $C^*$ equivariance (`< 1e-5` LLR err, 0 flips across $r \in [2^{-12}, 2^{12}]$ and random phases).
   - Unit test suite `p4/tests/test_p4_b_2d_fec.py`.
   - Evaluation artifacts `p4/results/p4b_coded_bler.json` and `p4/results/p4b_coded_bler.csv` comparing 2D EM-FloatFormer, 2D-LMMSE, and 2D IDD-PA across in-distribution and OOD Doppler and delay spreads.
   - Vectorized PyTorch LDPC + 2D slot execution so `S11` (`100_000` bits over 50 slots) and the entire 65-test M3 suite execute fast on 4-thread CPU.
