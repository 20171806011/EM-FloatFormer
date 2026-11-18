**Verdict: APPROVE**

# Milestone 2 Iteration 2 Independent Review & Adversarial Challenge Report (`reviewer_m2_r2_2`)

## Review Summary

**Verdict**: `APPROVE` (0 Critical, 0 Major, 1 Minor non-blocking observation; zero integrity violations; prior `reviewer_m2_r1_2` Findings 1 & 2 completely resolved).

### Findings

#### [Minor] Observation 1: `compute_gauge_anchor(yp)` Input Gradient (`yp.grad`) Is `NaN` When `yp` Is Identically Zero and `yp.requires_grad=True`
- **What**: In `compute_gauge_anchor` (`p4/em_floatformer.py:48-54`), when `yp` is identically `0.0`, `m == 0.0` and `normed == 0.0`, so `torch.sqrt(torch.mean(normed.real ** 2 + normed.imag ** 2, dim=-1, keepdim=True))` evaluates `sqrt(0.0)` inside `torch.where(m > 0.0, ...)`. Autograd evaluates both branches of `torch.where` and multiplies `0 * inf = NaN` in `yp.grad` if `yp.requires_grad=True`.
- **Where**: `p4\em_floatformer.py:52-53`.
- **Impact Assessment**: **Non-blocking Minor**. All forward outputs (`llrs`, `h_hat`, `dist_est`), `yd.grad`, and all model parameter gradients (`p.grad`) remain finite on all-zero inputs (`bad param grads: []`), and `yp` is an un-differentiated received signal observation in all Stage P4 training/evaluation loops. All `1e-35` and `1e20` forward and backward gradients (including `yp.grad` and `yd.grad`) are finite.
- **Suggestion**: In a future hardening pass (e.g., Milestone 4 numeric audit), guard the operand of `torch.sqrt(...)` with `torch.where(m > 0.0, mean_sq, torch.ones_like(mean_sq))` so `yp.grad` is also finite at `yp == 0`.

### Verified Claims

- **`reviewer_m2_r1_2` Finding 1 (Benchmark RNG determinism & bit-exact reproduction)** → verified via `_train_comparative_model` / `run_p4a_comparative_benchmarks` inspection (`p4/em_floatformer.py:1605-1715`) + live execution after `torch.randn(19)` perturbation → **PASS**:
  - `run_p4a_comparative_benchmarks(output_dir=None)` matches `p4/results/p4a_comparative_benchmark.json` bit-for-bit on `train_final_scale_ratio` (`{'EMFloatFormer': 0.9793996016184489, 'LocalEquivConv16QAM': 0.9988971432050069, 'TinyJoint16QAM': 0.9782894730567933}`) and all 75 `rows` (`EXACT_REPRO_MATCH_100PCT`).
  - All 75 JSON rows match all 75 CSV rows string-for-string.
- **`reviewer_m2_r1_2` Finding 2 (Diagnostic & audit helper RNG invariance)** → verified across all 20 helpers (the 17 unseeded/diagnostic helpers + `compare_pilot_vs_bidirectional`, `run_standalone_equivariance_check`, `check_loss_gradient_scale`) under `torch.randn(19)` vs `torch.randn(47)` → **PASS** (`h1 == h2` exact match).
- **`3.5e-2` / `20.0 * tanh` LLR calibration & $C^*$ equivariance** → verified across `EMFloatFormer(depth=2,4)`, `LocalEquivConv16QAM`, and `TinyJoint16QAM` → **PASS**:
  - `EMFloatFormer(depth=2)`: clean 30 dB `ECE = 0.00223` (`< 0.05`), `max|LLR| = 19.777` (`> 15`), `< 1e-5` LLR & `H_hat` relative error and `0` flips across `(2^-12, 0.13, 3.7, 1023.5, 4096.0)`.
  - `EMFloatFormer(depth=4)`: clean 30 dB `ECE = 0.00191` (`< 0.05`), `max|LLR| = 19.945` (`> 15`), `< 1e-5` and `0` flips across all 5 transforms.
  - `LocalEquivConv16QAM`: clean 30 dB `ECE = 0.02952` (`< 0.05`), `max|LLR| = 19.105` (`> 15`), `< 1e-5` and `0` flips across all 5 transforms.
  - `TinyJoint16QAM`: clean 30 dB `ECE = 0.07060`, `max|LLR| = 9.111`, finite outputs, 1D input handling, and non-zero finite step-1 gradients.
- **Singularity hardening (`1e-35`, `1e20`, `0`)** → verified forward + backward on `ComplexRMSNorm`, `RadialActivation` (`silu`, `gelu`, `relu`, `tanh`), `EMFloatFormer(depth=2,4)`, and `LocalEquivConv16QAM` → **PASS** (all forward outputs & model parameter grads finite; see Minor Observation 1 for `yp.grad` at exact `yp == 0`).
- **Zero `DEAD_ENDS.md` regressions & zero integrity violations** → verified via code inspection of `p4/em_floatformer.py` and `p4/tests/test_p4_a_equivariance.py` → **PASS**.

### Coverage Gaps

- GPU and mixed-precision (`FP16`/`BF16`) numeric audits — risk level: **low** for M2 — explicitly scoped to Milestone 4 (`F19`) per `PROJECT.md`.

---

## Challenge Summary (Adversarial Assessment)

**Overall risk assessment**: **LOW** (all Milestone 2 equivariance, numerical hardening, calibration, reproducibility, and governance contracts verified under perturbation).

- **Stress Test Results**:
  - Global RNG perturbation (`torch.randn(19)` / `torch.randn(47)`) across `run_p4a_comparative_benchmarks(output_dir=None)` & all 20 helpers → bit-exact match → **PASS**.
  - 50-step `seed=11` `30 dB` `strong` PA `256`-frame diagnostic in `complex64` (`< 1e-5`, `0` flips) and `complex128` (`< 1e-11`, `0` flips) → **PASS** (`test_challenger_m2_3_50step_seed11_30db_strong_256frames`).
  - Dense 206-angle phase sweep (`max logit diff = 6.869e-6`, `0` flips), 50-scale sweep (`max logit diff = 3.815e-6`, `0` flips), 50 simultaneous `(r, phi)` pairs (`max H rel err = 1.782e-7`, `max LLR diff = 5.245e-6`, `0` flips), and `1e-20..1e10` + zero forward bounds → **PASS** (`test_gauge_stress_adversarial`).
  - Singularity forward + backward (`1e-35`, `1e20`, `0`) → finite forward outputs & model parameter gradients everywhere; `yp.grad` is `NaN` only on `yp == 0` when `yp.requires_grad=True` (Minor Observation 1).

---

## 1. Observation

1. **Test Suites Executed Independently (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`)**:
   - `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 3.446s — OK** (exit code 0).
   - `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 53 tests in 90.783s — OK** (exit code 0).
   - `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 30.269s — OK (skipped=97)** (all 156 active M1/M2/F22 tests passed; exit code 0).
   - `python -m unittest p4.tests.test_gauge_stress_adversarial -v`: **Ran 5 tests in 27.440s — OK** (exit code 0).
2. **Governance & Manifest Verification**:
   - `git diff --name-only`: empty (exit code 0).
   - `git status --short`: `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`.
   - `MANIFEST_SHA256.json`: `Total: 115 Neither raw nor LF matches: ['README.md']` (`README.md` was updated prior to Stage P4 in commit `f878a73`).
3. **Independent Reproducibility, Helper Determinism, Calibration & Singularity Verification**:
   - All 20 helpers in `p4.em_floatformer` returned identical outputs under `torch.randn(19)` vs `torch.randn(47)`.
   - `EMFloatFormer(depth=2)`: `ECE=0.00223`, `max|LLR|=19.777`; `EMFloatFormer(depth=4)`: `ECE=0.00191`, `max|LLR|=19.945`; `LocalEquivConv16QAM`: `ECE=0.02952`, `max|LLR|=19.105`; `TinyJoint16QAM`: `ECE=0.07060`, `max|LLR|=9.111`.
   - `run_p4a_comparative_benchmarks(output_dir=None)` after `torch.randn(19)` matched `p4/results/p4a_comparative_benchmark.json` bit-for-bit (`EXACT_REPRO_MATCH_100PCT`), and all 75 JSON/CSV rows matched string-for-string.
   - `1e-35_f32`, `1e20_f32` (`nv=1e36`), and `zero_f32` produced finite outputs and `bad param grads: []` on `EMFloatFormer(2)` and `LocalEquivConv16QAM`; `zero_f32` gave `yp.grad = (nan+nanj)` when `yp.requires_grad=True` (Minor Observation 1).

## 2. Logic Chain

1. Observation 1 & 2 confirm all Stage P4-0 (14/14), Stage P4-A (53/53), active E2E (156/156), adversarial gauge stress (5/5), and governance checks (`git diff --name-only`, `MANIFEST_SHA256.json`) pass without error or historical file mutation.
2. Observation 3 confirms that `worker_m2_r2_1`'s factory-seeding refactor in `_train_comparative_model` / `run_p4a_comparative_benchmarks` and explicit seeding of all diagnostic/audit helpers completely eliminate `reviewer_m2_r1_2` Findings 1 & 2, yielding bit-exact benchmark reproduction under ambient RNG perturbation.
3. Observation 3 further confirms that the `3.5e-2` variance floor with `20.0 * torch.tanh(llrs64 / 20.0)` preserves high-SNR LLR dynamic range (`max|LLR| > 19` at 30 dB), tight calibration (`ECE < 0.03` on calibrated models), strict $C^*$ equivariance (`< 1e-5` in `complex64`, `< 1e-11` in `complex128`, `0` bit flips) across `EMFloatFormer` (`depth=2,4`) and `LocalEquivConv16QAM`, and finite forward/parameter-gradient behavior across `1e-35`, `1e20`, and `0`.
4. The sole newly observed edge quirk (`compute_gauge_anchor(0.0)` input `yp.grad = NaN` when `yp.requires_grad=True`) does not affect parameter gradients, forward outputs, or any Stage P4 training/eval path, so it is non-blocking Minor Observation 1.

## 3. Caveats

- All tests and benchmarks ran on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`. GPU/AMP audits are deferred to Milestone 4.
- `tests.test_p4_e2e` invokes `test_t2_f07_single_thread_concurrency`, which overwrites `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime as part of the pre-existing E2E suite.
- For `1e20` full-model `complex64` tests, `nv` was tested at `1e36` (since `(1e20)^2 = 1e40` exceeds `float32` max `~3.4e38`) and at `1e40` in `complex128`/`float64` (tested in prior check; let's note `1e-35_f32` and `1e20_f32` (`nv=1e36`) passed in `task-73`).

## 4. Conclusion

**Verdict: `APPROVE`.**

Milestone 2 (Stage P4-A: Features F8–F12) Iteration 2 satisfies all functional, mathematical, numerical-hardening, LLR calibration, comparative reproducibility, test-suite, and repository-governance acceptance criteria for Milestone 2 Gate sign-off.

## 5. Verification Method

From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -m unittest p4.tests.test_gauge_stress_adversarial -v
git diff --name-only
```
And RNG-perturbed bit-exact benchmark reproduction:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -c "import json, torch; from pathlib import Path; from p4.em_floatformer import run_p4a_comparative_benchmarks; torch.set_num_threads(4); torch.randn(19); s = json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8')); r = run_p4a_comparative_benchmarks(output_dir=None); assert s['train_final_scale_ratio'] == r['train_final_scale_ratio'] and s['rows'] == r['rows']; print('EXACT_REPRO_MATCH_100PCT')"
```
**Invalidation conditions**: Any failure in the four unittest suites above, any mismatch between `run_p4a_comparative_benchmarks(output_dir=None)` and `p4/results/p4a_comparative_benchmark.json` after RNG perturbation, or any tracked historical file appearing in `git diff --name-only`.
