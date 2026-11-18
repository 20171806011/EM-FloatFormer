# Milestone 2 Iteration 2 Implementation Handoff (`worker_m2_r2_1`)

## 1. Observation

### 1.1 Changes Applied to `p4/em_floatformer.py`
1. **Factory RNG Seeding (`reviewer_m2_r1_2` Finding 1, `explorer_m2_r2_2`)**:
   - Added `Callable` to `from typing import ...` (`p4/em_floatformer.py:17`).
   - Refactored `_train_comparative_model` (`p4/em_floatformer.py:1605-1623`) to accept `model_or_factory: Union[nn.Module, Callable[[], nn.Module]]`, invoke `torch.manual_seed(seed)`, and then instantiate `model = model_or_factory if isinstance(model_or_factory, nn.Module) else model_or_factory()`.
   - Updated `run_p4a_comparative_benchmarks` (`p4/em_floatformer.py:1683-1714`) to pass factory lambdas:
     - `lambda: EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=n_sc)` with `seed=seed`
     - `lambda: LocalEquivConv16QAM(width=14, kernel_size=5, depth=2, ffn_dim=12)` with `seed=seed + 1`
     - `lambda: TinyJoint16QAM(width=16, taps=8)` with `seed=seed + 2`
   - Left all comparative hyperparameters, learning rates, schedules, batch sizes, and step counts unchanged.
2. **Helper & Diagnostic RNG Seeding (`reviewer_m2_r1_2` Finding 2, `explorer_m2_r2_2`)**:
   - Added explicit `torch.manual_seed(...)` before model/block instantiation in `evaluate_noise_data_feedback` (`seed=42`), `evaluate_perfect_pilot_demapping` (`seed=99`), and all 15 unseeded audit helpers (`audit_single_scale` `42`, `audit_single_phase` `42`, `verify_loss_gradient_invariance` `42`, `audit_loss_scale_invariance` `42`, `train_step_joint_loss` `42`, `audit_gain_equivariance` `42`, `audit_phase_equivariance` `42`, `audit_c_star_equivariance` `123`, `audit_channel_equivariance` `999`, `sweep_phases_dense` `42`, `check_irrational_phases` `42`, `check_noise_quad_scaling` `42`, `verify_gauge_attention_joint` `77`, `audit_full_equivariance` `88`, `verify_bidirectional_green_prop` `123`).
3. **LLR Variance Floor Calibration (`explorer_m2_r2_1`)**:
   - Updated `sigma_eff2` floor from `2.5e-2` to `3.5e-2` while preserving `20.0 * torch.tanh(llrs64 / 20.0)` in both `EMFloatFormer.forward` (`p4/em_floatformer.py:653-655`) and `LocalEquivConv16QAM.forward` (`p4/em_floatformer.py:1487-1489`).

### 1.2 Regenerated Comparative Benchmark Artifacts & Reproducibility Check
- Executed `run_p4a_comparative_benchmarks()` with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)` to regenerate `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv`:
  - `train_final_scale_ratio`: `{'EMFloatFormer': 0.9793996016184489, 'LocalEquivConv16QAM': 0.9988971432050069, 'TinyJoint16QAM': 0.9782894730567933}`
  - `rows`: `75` (`3 PA conditions x 5 SNRs x 5 methods`).
- Perturbed global RNG with `torch.randn(19)` prior to calling `run_p4a_comparative_benchmarks(output_dir=None)` and verified `saved['train_final_scale_ratio'] == repro['train_final_scale_ratio']` and `saved['rows'] == repro['rows']` -> printed `EXACT_REPRO_MATCH_100PCT` (exit code 0).
- Verified RNG-perturbation determinism (`torch.randn(13)` vs `torch.randn(29)`) across all 17 diagnostic/audit helpers -> printed `ALL_17_HELPERS_DETERMINISTIC` (exit code 0).

### 1.3 Test Gap Closures in `p4/tests/test_p4_a_equivariance.py`
Added/updated tests covering all 6 gaps from `explorer_m2_r2_3/handoff.md`:
1. `TestF12NumericalEquivariance.test_challenger_m2_3_50step_seed11_30db_strong_256frames`: Exact 50-step `seed=11` `256`-frame `30 dB` `strong` PA diagnostic asserting `< 1e-5` / `0` flips in `complex64` AND `< 1e-11` / `0` flips in `complex128`.
2. `TestM2CalibrationAndBaselines.test_joint_loss_mixed_pa_resolution_and_scale_stability`: Full `_resolve_pa_A` (`float("inf")`, `"clean"`, `"mild"`, `" Strong "`, plain `FrameData` default `1.25`, `frame.A = float("inf")`, `frame.pa_condition = "strong"`, invalid string `ValueError` on both `A` and `frame.pa_condition`) and `pa_residual_loss(..., A=float("inf"))` linear-PA path.
3. `TestM2CalibrationAndBaselines.test_comparative_baseline_models_equivariance_and_params`: Added `ComplexConv1d` (`bias is None`, `W(c*z) == c*W(z)`), `LocalEquivConv16QAM` (1D `(32,)` input, step-1 non-zero finite gradients on all params, 20-step trained 24-octave equivariance `< 1e-5` / `0` flips), and `TinyJoint16QAM` (1D `(32,)` input, finite `h_hat`/`dist_est`, step-1 non-zero finite gradients `abs().sum() > 0` on all params).
4. `TestM2CalibrationAndBaselines.test_rng_perturbation_determinism_benchmarks_and_diagnostics`: Verified RNG-perturbation determinism (`torch.randn(17)` / `torch.randn(53)`) for `evaluate_noise_data_feedback`, `evaluate_perfect_pilot_demapping`, and `run_p4a_comparative_benchmarks`.
5. `TestM2CalibrationAndBaselines.test_saved_p4a_comparative_benchmark_artifacts_parity_and_invariants`: Verified JSON metadata, 75-row CSV/JSON exact string parity, `ECE < 0.05` on the 4 calibrated methods (`EMFloatFormer`, `LocalEquivConv16QAM`, `Bussgang-LMMSE`, `DD-PA-Cancel`), strict monotonic non-increasing BER on the 4 calibrated methods, and `ber(30 dB) < ber(10 dB)` / `<= +5e-4` floor bound for uncalibrated `TinyJoint16QAM` (whose fixed `+0.03` regularizer hits a `~0.0116` floor at 25–30 dB `clean`: `0.0115966796875 -> 0.01171875`, 4 bits out of 32,768 across independent eval seeds `202623` vs `202624`).

### 1.4 Final Verification Outputs (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`)
- `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 2.089s — OK** (exit code 0).
- `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 53 tests in 71.266s — OK** (exit code 0).
- `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 25.055s — OK (skipped=97)** (all 156 active M1/M2/F22 tests passed; exit code 0).
- `python -m unittest p4.tests.test_gauge_stress_adversarial -v`: **Ran 5 tests in 23.904s — OK** (exit code 0).
- `git diff --name-only`: empty (exit code 0; 0 tracked historical P1–P3 / `exploratory/*` / `MANIFEST_SHA256.json` files modified).

---

## 2. Logic Chain
1. Moving `torch.manual_seed(seed)` before `model_or_factory()` inside `_train_comparative_model` and passing factory lambdas from `run_p4a_comparative_benchmarks` decouples all three benchmarked neural models (`EMFloatFormer`, `LocalEquivConv16QAM`, `TinyJoint16QAM`) from ambient global RNG state and assigns each its intended distinct seed (`seed`, `seed + 1`, `seed + 2`).
2. Seeding `evaluate_noise_data_feedback`, `evaluate_perfect_pilot_demapping`, and all 15 audit helpers before model construction eliminates every remaining unseeded global RNG dependency in `p4/em_floatformer.py`.
3. Calibrating `sigma_eff2` floor to `3.5e-2` with `20.0 * torch.tanh(llrs64 / 20.0)` in both `EMFloatFormer.forward` and `LocalEquivConv16QAM.forward` resolves `challenger_m2_3`'s exact 50-step `seed=11` `complex64` diagnostic (`< 1e-5`, `0` flips) while maintaining `complex128` accuracy (`< 1e-11`), low `ECE` (`< 0.05` across all 9 cells), and high-SNR LLR dynamic range (`> 15.0`).
4. Regenerating `p4/results/p4a_comparative_benchmark.{json,csv}` and passing all 4 verification suites confirms end-to-end correctness and exact reproducibility.

---

## 3. Caveats
- All tests and benchmarks ran on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`.
- As documented by `explorer_m2_r2_3`, running `tests.test_p4_e2e` invokes `tests/e2e/test_tier2_boundary_corner.py`, which updates `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime; neither file was manually modified by `worker_m2_r2_1`.
- With `TinyJoint16QAM` now receiving `seed + 2` (`2028`) instead of accidentally inheriting `seed + 1` (`2027`), its fixed `+ 0.03` regularizer plateaus around `BER ~ 0.0116` at 25–30 dB `clean` (`0.011597` vs `0.011719` across independent eval seeds `202623` vs `202624`). Because comparative hyperparameters and architectures are frozen without favoring hacks, `test_saved_p4a_comparative_benchmark_artifacts_parity_and_invariants` asserts strict monotonic BER on the 4 calibrated methods and `ber(30 dB) < ber(10 dB)` + `<= +5e-4` floor tolerance on `TinyJoint16QAM`.

---

## 4. Conclusion
All 5 Milestone 2 Iteration 2 requirements are complete and verified:
- `p4/em_floatformer.py`: Factory RNG seeding, full helper RNG seeding (zero global RNG dependence), and `3.5e-2` LLR variance floor calibration (`20.0 * tanh`).
- `p4/results/p4a_comparative_benchmark.{json,csv}`: Regenerated under 4 threads and verified 100% bit-exact reproducible under global RNG perturbation.
- `p4/tests/test_p4_a_equivariance.py`: All 6 test gaps closed; all 53 tests pass.
- All 4 verification suites (`test_p4_0_simulator`, `test_p4_a_equivariance`, `test_p4_e2e`, `test_gauge_stress_adversarial`) pass with exit code 0, and `git diff --name-only` is empty.

---

## 5. Verification Method
From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -m unittest p4.tests.test_gauge_stress_adversarial -v
python -c "
import json, torch
from pathlib import Path
from p4.em_floatformer import run_p4a_comparative_benchmarks
torch.set_num_threads(4)
torch.randn(19)
saved = json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8'))
repro = run_p4a_comparative_benchmarks(output_dir=None)
assert saved['train_final_scale_ratio'] == repro['train_final_scale_ratio']
assert saved['rows'] == repro['rows']
print('EXACT_REPRO_MATCH_100PCT')
"
git diff --name-only
```
**Invalidation conditions**: Any failure in the 4 unittest suites, any mismatch in the RNG-perturbed benchmark replay, or any tracked historical file appearing in `git diff --name-only`.
