# Milestone 2 (Stage P4-A) Iteration 2 Test & Governance Audit (`explorer_m2_r2_3`)

**Summary:** Audited `p4/tests/test_p4_a_equivariance.py` (especially lines 498–656), `p4/__init__.py`, `p4/results/`, `MANIFEST_SHA256.json`, and Stage P4-0 preservation. Identified **6 concrete test gaps/weak assertions** in `p4/tests/test_p4_a_equivariance.py` (including an exact precision nuance on `challenger_m2_3`'s 50-step `seed=11` `30 dB` `256`-frame check) plus **2 governance/artifact details** (`p4a_comparative_benchmark.{json,csv}` requiring regeneration after `run_p4a_comparative_benchmarks` RNG seeding fix, and `tests/e2e/test_tier2_boundary_corner.py` overwriting `p4/results/latency_cpu.csv` and `timing.json` at runtime).

---

## 1. Observation

### 1.1 Audit of `p4/tests/test_p4_a_equivariance.py` (Lines 18–56, 498–656)
Inspected `p4\tests\test_p4_a_equivariance.py`:
1. **Imports (`lines 18–56`)**: `run_p4a_comparative_benchmarks` is **not imported**. `ComplexConv1d` is **not imported** (only `LocalEquivConv16QAM` and `TinyJoint16QAM` are imported at `lines 27, 29`).
2. **`test_fresh_seed11_and_trained_depth2_c_star_equivariance` (`lines 499–555`)**:
   - Part 1 (`lines 500–525`): tests fresh `seed=11` `EMFloatFormer(depth=2)` on `batch_size=8, n=32, snr_db=20.0, pa_condition="strong"` (`seed=1101`).
   - Part 2 (`lines 527–555`): trains `EMFloatFormer(depth=2)` (`seed=11`) for **40 steps** with split LRs (`1e-2` backbone, `3e-2` `gamma_eff`), `batch_size=16`, `snr_db=(10.0, 25.0)` cycling `("strong", "mild", "clean")`, and evaluates on the same `batch_size=8`, `snr_db=20.0`, `pa_condition="strong"` frame `fr`.
   - **Does NOT test** `challenger_m2_3/handoff.md:100-117`'s exact diagnostic: 50 Adam steps (`lr=1e-2`) on a single `batch_size=16, snr_db=18.0, pa_condition="strong", generator=99` frame, evaluated on `256` frames at `snr_db=30.0, pa_condition="strong", generator=31415`.
   - **Empirical measurement of `challenger_m2_3`'s exact 50-step `seed=11` diagnostic** (executed in read-only Python):
     - `r=2^-12, phi=2.7183`: `err_c64=6.676e-06 (flips=0)`, `err_c128=8.349e-14 (flips=0)`
     - `r=0.13, phi=-1.7`: `err_c64=9.537e-06 (flips=0)`, `err_c128=9.770e-14 (flips=0)`
     - `r=3.7, phi=0.95`: `err_c64=9.060e-06 (flips=0)`, `err_c128=1.013e-13 (flips=0)`
     - `r=1023.5, phi=2.5356`: `err_c64=6.676e-06 (flips=0)`, `err_c128=8.527e-14 (flips=0)`
     - `r=4096.0, phi=-0.6076`: `err_c64=1.907e-05 (flips=0)`, `err_c128=9.237e-14 (flips=0)`
     (Consistent with `reviewer_m2_r1_1/handoff.md:29`, where `err_c64=1.907e-05` is solely caller `complex64` rounding on `fr.yp * c` / `fr.yd * c`, while exact `complex128` scaling has `err_c128 <= 1.013e-13`.)
3. **`TestM2CalibrationAndBaselines.test_calibrated_llr_dynamic_range_ece_and_equivariance_high_snr` (`lines 564–602`)**:
   - Trains `EMFloatFormer(depth=2)` for 80 steps, checks `ECE < 0.05` across `("clean", "mild", "strong") x (15.0, 20.0, 30.0)`, `max|LLR| > 15.0` at `("clean", 30.0)`, and `< 1e-5` `llrs`/`h_hat` equivariance with `0` flips in `complex64` across all 9 cells.
4. **`TestM2CalibrationAndBaselines.test_joint_loss_mixed_pa_resolution_and_scale_stability` (`lines 604–633`)**:
   - Only calls `loss_fn._resolve_pa_A` on two cases (`lines 607–612`): `fr_clean` (`pa_condition="clean"`) -> `None`, and `fr_mild` (`A=2.25`) -> `2.25`.
   - **Untested `_resolve_pa_A` / `_normalize_pa_A` paths (`p4/em_floatformer.py:825-853`)**:
     - Explicit `A=float("inf")` -> `None`
     - Explicit `A="clean"` -> `None`, `A="mild"` -> `2.25`, `A="strong"` -> `1.25` (plus case/whitespace e.g. `" Strong "`)
     - `frame.A = float("inf")` -> `None`
     - `frame.pa_condition = "strong"` (without `frame.A`, since `generate_frame` returns `FrameData` with `hasattr(fr, "A") == False` and `hasattr(fr, "pa_condition") == False`) -> `1.25`
     - Plain `FrameData` (neither `A` nor `pa_condition`) -> default `1.25`
     - Invalid string `A="invalid"` or `frame.pa_condition = "bad"` -> `ValueError`
     - `pa_residual_loss(..., A=None)` and `A=float("inf")` linear-PA branch (`p4/em_floatformer.py:863-864`).
5. **`TestM2CalibrationAndBaselines.test_comparative_baseline_models_equivariance_and_params` (`lines 635–657`)**:
   - `LocalEquivConv16QAM` (`lines 636–649`): only checks parameter parity (`< 5%` of `EMFloatFormer`) and untrained `llrs` diff `< 1e-5` + `flips == 0` for a single `(r=3.7, phi=0.95)` on `4x32` `20 dB`.
     - **Missing**: `ComplexConv1d` unit test (`bias is None`, `W(c Z) == c W(Z)`), `LocalEquivConv16QAM` **trained** 24-octave equivariance (`llrs` `< 1e-5`, `flips == 0`, `h_hat` rel err `< 1e-5`), **step-1 non-zero finite gradients** on all parameters under `JointPhysicalLoss`, and 1D `(32,)` input support.
     - Empirical read-only run of 20-step mixed-PA trained `LocalEquivConv16QAM` across `(2^-12, 0.13, 3.7, 1023.5, 2^12)` passed with `llr_err <= 2.861e-06 < 1e-5`, `flips = 0`, `h_rel <= 1.246e-07 < 1e-5`; step-1 gradients had `[]` missing/zero/non-finite tensors; 1D `(32,)` input produced `llrs` shape `(1, 32, 4)`.
   - `TinyJoint16QAM` (`lines 651–656`): checks `(4, 32)` batch, `llrs` shape `(4, 32, 4)`, `isfinite(llrs)`, and `p.grad is not None and torch.isfinite(p.grad).all()`.
     - **Weak/Missing**: does **NOT** check non-zero gradients (`float(p.grad.abs().sum()) > 0.0`), does **NOT** test **1D input** `(32,)` (supported at `p4/em_floatformer.py:1505-1510`, verified returning `llrs` `(1, 32, 4)` and `h_hat` `(1, 32)`), and does **NOT** assert `torch.isfinite(out_tiny["h_hat"]).all()` or `torch.isfinite(out_tiny["dist_est"]).all()`. Empirical check confirmed step-1 `TinyJoint16QAM` has `[]` zero/missing/non-finite parameter gradients.
6. **Missing Benchmark Tests (lines 498–656)**:
   - Zero tests for `run_p4a_comparative_benchmarks` RNG-perturbation determinism (`torch.randn(17)` before calling). Note: as `reviewer_m2_r1_2` documented, `p4/em_floatformer.py:1661-1692` currently instantiates all 3 neural models *before* `_train_comparative_model` calls `torch.manual_seed(seed)` at `line 1600`, so `run_p4a_comparative_benchmarks` must first be fixed to seed before model construction and regenerate `p4/results/p4a_comparative_benchmark.{json,csv}`.
   - Zero tests for `p4/results/p4a_comparative_benchmark.{json,csv}` schema, row-for-row parity, and physical invariants.
     - Empirical audit of current `p4/results/p4a_comparative_benchmark.{json,csv}`: 75 rows (`3 PA x 5 SNR x 5 methods`), `0` JSON/CSV mismatches, `Non-monotonic BER: []` (all 15 `(pa, method)` curves monotonic), `param_counts = {'EMFloatFormer': 14090, 'LocalEquivConv16QAM': 14198, 'TinyJoint16QAM': 1880, 'Bussgang-LMMSE': 0, 'DD-PA-Cancel': 0}`, `ECE < 0.05` holds across all 60 rows of the 4 calibrated methods (`EMFloatFormer`, `LocalEquivConv16QAM`, `Bussgang-LMMSE`, `DD-PA-Cancel`), while uncalibrated heuristic `TinyJoint16QAM` has `ECE >= 0.05` at `10.0 dB` (`mild: 0.05520`, `strong: 0.08135`).

### 1.2 Audit of `p4/__init__.py`
Inspected `p4\__init__.py` (`lines 1–134`):
- `__all__` exports `61` symbols (`21` M1 symbols at `lines 73–91` + `40` M2 symbols at `lines 93–133`).
- `python -c "import p4; all(hasattr(p4, name) for name in p4.__all__)"` -> `All __all__ resolve: True`.
- All M2 comparative symbols (`ComplexConv1d`, `LocalEquivConv16QAM`, `TinyJoint16QAM`, `compute_ece`, `run_p4a_comparative_benchmarks`) are exported (`lines 64–68, 129–133`).
- Public symbol in `p4/em_floatformer.py` not in `p4.__all__`: `['RealAttention']` (`p4/em_floatformer.py:360-405`, used internally by `BidirectionalBlock` when `use_hermitian_attention=False`).

### 1.3 Audit of Git Status, `MANIFEST_SHA256.json`, `p4/results/`, and Stage P4-0 Preservation
Executed in ``:
1. `git status --short`:
   ```text
   ?? .agents/
   ?? PROJECT.md
   ?? TEST_INFRA.md
   ?? TEST_READY.md
   ?? p4/
   ?? tests/
   ```
   `git diff --name-only`: empty (0 modified tracked files).
2. `MANIFEST_SHA256.json`: `115` entries, `0` missing, `114/115` match byte-exact under LF (`\r\n` -> `\n`) normalization; sole diff `README.md` (`d4604ef0...` vs `39e8f1ea...`) is pre-P4 commit `f878a73` explicitly exempted in `tests/e2e/test_tier1_feature_coverage.py:1128-1130`.
3. Stage P4-0 source/test files (`p4/` directory listing):
   - `simulator.py` (`19737 B`, `2026/11/15 23:13:57`)
   - `classical_baselines.py` (`16497 B`, `2026/11/15 23:35:02`)
   - `benchmark_latency.py` (`6688 B`, `2026/11/15 23:18:32`)
   - `tests/test_p4_0_simulator.py` (`17347 B`, `2026/11/15 23:35`)
   All untouched since `2026-11-15`.
4. `p4/results/` contents:
   - `p4a_comparative_benchmark.csv` (`11420 B`, `2026/11/17 19:12:19`)
   - `p4a_comparative_benchmark.json` (`31271 B`, `2026/11/17 19:12:19`)
   - `latency_cpu.csv` (`932 B`, `2026/11/17 19:17:36`)
   - `timing.json` (`2567 B`, `2026/11/17 19:17:36`)
5. **Cause of `latency_cpu.csv` / `timing.json` timestamp (`19:17:36`)**: `tests/e2e/test_tier2_boundary_corner.py:382,390,396,402` (and `test_tier1_feature_coverage.py:448,457`) call `run_benchmarks(...)` with default `output_dir="p4/results"`, overwriting `p4/results/latency_cpu.csv` and `p4/results/timing.json` with `batch_size=1, threads=1, warmup_runs=1, timed_runs=3` whenever `tests.test_p4_e2e` runs.

---

## 2. Logic Chain

1. Because `p4/tests/test_p4_a_equivariance.py` lines 498–656 lack tests for `run_p4a_comparative_benchmarks` RNG determinism and `p4/results/p4a_comparative_benchmark.{json,csv}` parity, the unseeded model instantiation bug in `p4/em_floatformer.py:1661-1692` (Observation 1.1.6) was not caught by the unit test suite.
2. Adding a test for `challenger_m2_3`'s exact 50-step `seed=11` `256`-frame `30 dB` `strong` PA check directly with naive `complex64` `< 1e-5` assertions would fail at `(r=4096.0, phi=-0.6076)` with `err_c64=1.907e-05` due to caller `complex64` input rounding, whereas `complex128` input scaling gives `<= 1.013e-13` and `0` flips (Observation 1.1.2). Thus the test must assert `< 1e-5` (or `< 1e-11`) in `complex128` and `flips == 0` / `h_rel < 1e-5` in `complex64`.
3. `TestM2CalibrationAndBaselines` currently leaves `ComplexConv1d`, trained 24-octave `LocalEquivConv16QAM` equivariance, `LocalEquivConv16QAM` step-1 non-zero gradients, `TinyJoint16QAM` 1D input + non-zero step-1 gradients, and `JointPhysicalLoss._resolve_pa_A` edge cases (`inf`, `"clean"`, `"mild"`, `"strong"`, `None`/fallback, invalid string `ValueError`) uncovered or weakly asserted (Observations 1.1.4, 1.1.5).
4. Governance and Stage P4-0 preservation are clean in `git` and `MANIFEST_SHA256.json` (Observations 1.3.1–1.3.3), except (a) `p4a_comparative_benchmark.{json,csv}` must be regenerated after fixing `run_p4a_comparative_benchmarks` seeding, and (b) `p4/results/latency_cpu.csv` & `timing.json` are overwritten at runtime by the read-only E2E suite `tests/e2e/test_tier2_boundary_corner.py` (Observation 1.3.5).

---

## 3. Caveats

- All checks ran on CPU (`torch.set_num_threads(4)`). GPU / FP16 / BF16 behavior is out of Milestone 2 scope (Milestone 4).
- `p4/em_floatformer.py:733` (`evaluate_noise_data_feedback`) and `:759` (`evaluate_perfect_pilot_demapping`) also construct `EMFloatFormer(depth=1)` before any `torch.manual_seed` (minor diagnostic hygiene item noted by `reviewer_m2_r1_2`).
- `tests/e2e/*` is read-only per `SCOPE.md:29`, so `test_tier2_boundary_corner.py`'s overwrite of `p4/results/latency_cpu.csv` / `timing.json` cannot be patched in E2E tests; if canonical Stage P4-0 `batch_sizes=(1,64), threads=4, warmup=25, runs=100` latency artifacts are required after E2E runs, `python -m p4.benchmark_latency` must be executed as the final step (or the Forensic Auditor must expect `tests.test_p4_e2e` to touch those two files).

---

## 4. Conclusion

### 4.1 Recommended Test Additions in `p4/tests/test_p4_a_equivariance.py`
1. **Imports (`lines 18–56`)**: Import `ComplexConv1d` and `run_p4a_comparative_benchmarks` from `p4.em_floatformer` (plus `csv`, `json`, `Path`).
2. **Exact `challenger_m2_3` 50-step `seed=11` `256`-frame `30 dB` `strong` PA check (after `line 555`)**:
   - Reproduce `torch.manual_seed(11)`, `EMFloatFormer(depth=2)`, `Adam(lr=1e-2)`, `JointPhysicalLoss(1.0, 0.25)`, 50 steps on `generate_frame(16, 32, snr_db=18.0, pa_condition="strong", generator=manual_seed(99))`, evaluate on `generate_frame(256, 32, snr_db=30.0, pa_condition="strong", generator=manual_seed(31415))` across `[(2.0**-12, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)]`:
     - Assert `complex128` inputs (`fr.yp.to(torch.complex128) * c128`, etc.) achieve `llr_err < 1e-11` (`< 1e-5`), `flips == 0`, `h_rel < 1e-11`.
     - Assert `complex64` inputs achieve `flips == 0` and `h_rel < 1e-5` (noting caller `complex64` rounding yields `~1.9e-5` at `r=4096.0`).
3. **Complete `_resolve_pa_A` & `pa_residual_loss` linear-PA edge cases (`lines 604–613`)**:
   - Assert `_resolve_pa_A(plain_frame)` == `1.25` (when `FrameData` lacks `A` and `pa_condition`).
   - Assert `_resolve_pa_A(fr, A=float("inf")) is None`, `_resolve_pa_A(fr, A="clean") is None`, `_resolve_pa_A(fr, A="mild") == 2.25`, `_resolve_pa_A(fr, A=" Strong ") == 1.25`.
   - Assert `fr_inf.A = float("inf")` -> `None`; `fr_strong.pa_condition = "strong"` (no `A`) -> `1.25`.
   - Assert `self.assertRaises(ValueError, loss_fn._resolve_pa_A, fr, A="unknown")` and `ValueError` on unknown `frame.pa_condition`.
   - Assert `loss_fn.pa_residual_loss(yd, h_est, x_est, A=float("inf"))` equals `A=None` and is finite `>= 0`.
4. **Harden `LocalEquivConv16QAM` (`ComplexConv1d`) & `TinyJoint16QAM` tests (`lines 635–657`)**:
   - Test `ComplexConv1d`: `self.assertIsNone(conv.bias)` and `W(c * z) == c * W(z)` homogeneity (`atol=1e-5`) for `c = 3.7 * exp(1j * 0.95)`.
   - Test `LocalEquivConv16QAM`:
     - Step-1 non-zero finite gradients on **every** parameter (`p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0`).
     - 1D `(32,)` input returns `(1, 32, 4)` `llrs` and `(1, 32)` `h_hat`.
     - **Trained** (e.g. 20 steps mixed-PA) `LocalEquivConv16QAM` across 24 octaves (`2**-12, 0.13, 3.7, 1023.5, 2**12`) achieves `llr_err < 1e-5`, `flips == 0`, and `h_rel < 1e-5`.
   - Test `TinyJoint16QAM`:
     - Strengthen `line 656` to also assert `float(p.grad.abs().sum()) > 0.0` on every parameter (`bad_grads == []`).
     - Assert `torch.isfinite(out_tiny["h_hat"]).all()` and `torch.isfinite(out_tiny["dist_est"]).all()`.
     - Add 1D input `(32,)` test (`tiny(fr.yp[0], fr.yd[0], fr.noise_var[0])` -> `llrs` `(1, 32, 4)`, `h_hat` `(1, 32)`, `dist_est` `(1, 32)`).
5. **New test: `run_p4a_comparative_benchmarks` RNG-perturbation determinism** (after `line 656`):
   - Call `r1 = run_p4a_comparative_benchmarks(train_steps=2, batch_size=4, eval_frames=8, snrs_db=(15.0,), pa_conditions=("strong",), seed=2026, output_dir=None)`.
   - Perturb global RNG with `torch.randn(17)`.
   - Call `r2 = run_p4a_comparative_benchmarks(train_steps=2, batch_size=4, eval_frames=8, snrs_db=(15.0,), pa_conditions=("strong",), seed=2026, output_dir=None)`.
   - Assert `r1["train_final_scale_ratio"] == r2["train_final_scale_ratio"]` and `r1["rows"] == r2["rows"]` (exact equality on `bit_errors`, `ber`, `nmse`, `bce`, `ece`, `scale_ratio`).
6. **New test: `p4/results/p4a_comparative_benchmark.{json,csv}` schema, parity & physical sanity** (after `line 656`):
   - Load both files from `Path("p4/results")`.
   - Assert top-level JSON metadata (`stage == "P4-A"`, `profile == "equal4"`, `n_sc == 32`, `train_steps == 240`, `batch_size == 32`, `eval_frames == 256`, `seed == 2026`, `num_threads == 4`, `len(rows) == 75`).
   - Assert exact CSV header and row-for-row string parity with JSON `rows`.
   - Assert `total_bits == 32768`, `0 <= bit_errors <= 32768`, `ber == bit_errors / total_bits`, finite metrics, `ECE < 0.05` on all rows for `{"EMFloatFormer", "LocalEquivConv16QAM", "Bussgang-LMMSE", "DD-PA-Cancel"}` (explicitly excluding uncalibrated heuristic `TinyJoint16QAM` which has `ECE ~ 0.055–0.081` at `10 dB`), and monotonic non-increasing BER across `10..30 dB` for every `(pa_condition, method)`.

### 4.2 Pre-Sign-Off Package / Governance Actions
1. **Optional `p4/__init__.py` hygiene**: Consider whether `RealAttention` (`p4/em_floatformer.py:360`) should be added to `p4/__init__.py` or intentionally kept internal (currently `61/61` `__all__` entries resolve cleanly).
2. **Benchmark regeneration (Blocking per `reviewer_m2_r1_2`)**: After seeding `torch.manual_seed(seed + idx)` before constructing each model in `run_p4a_comparative_benchmarks`, regenerate `p4/results/p4a_comparative_benchmark.{json,csv}` so saved artifacts reproduce bit-exact after `torch.randn(17)`.
3. **Stage P4-0 latency artifact hygiene**: After running `tests.test_p4_e2e` (which overwrites `p4/results/latency_cpu.csv` and `timing.json` via `test_tier2_boundary_corner.py`), decide whether to re-run `python -m p4.benchmark_latency` to restore canonical `(batch_sizes=(1,64), threads=4, warmup=25, runs=100)` artifacts before Forensic Auditor sign-off.

---

## 5. Verification Method

From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
git diff --name-only
```
And verify `MANIFEST_SHA256.json` + JSON/CSV parity + RNG-perturbed benchmark exact match:
```powershell
python -c "
import json, torch
from pathlib import Path
from p4.em_floatformer import run_p4a_comparative_benchmarks
torch.randn(17)
saved = json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8'))
repro = run_p4a_comparative_benchmarks(output_dir=None)
assert saved['train_final_scale_ratio'] == repro['train_final_scale_ratio']
assert max(abs(a['ber'] - b['ber']) for a, b in zip(saved['rows'], repro['rows'])) == 0.0
"
```
**Invalidation conditions**: Any failure in the three unittest suites, any new gap in `p4/tests/test_p4_a_equivariance.py`, any non-empty `git diff --name-only`, or any mismatch in `MANIFEST_SHA256.json` beyond the pre-existing `README.md` entry.
