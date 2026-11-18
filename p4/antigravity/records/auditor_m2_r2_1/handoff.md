Verdict: CLEAN

# Forensic Audit Report (`auditor_m2_r2_1`)

**Work Product**: EM-FloatFormer Milestone 2 (Stage P4-A: Features F8–F12) Iteration 2 (`p4/em_floatformer.py`, `p4/__init__.py`, `p4/tests/test_p4_a_equivariance.py`, `p4/tests/test_gauge_stress_adversarial.py`, `p4/results/p4a_comparative_benchmark.{json,csv}`)
**Verdict**: **CLEAN**

---

## Phase Results Summary

- **Phase 1 — Static Integrity & `DEAD_ENDS.md` Regression Audit**: **PASS**
  - Zero hardcoded test thresholds or output clamping (`compare_pilot_vs_bidirectional` runs a genuine 50-step Adam loop and returns unclamped `gain_db`; `evaluate_noise_data_feedback` returns unclamped `deg_db`; `sigma_eff2` uses `+ 3.5e-2` and `llrs = 20.0 * torch.tanh(llrs64 / 20.0)`, preserving high-SNR dynamic range without `>= 0.75` or `8.0 * tanh` compression).
  - Zero facade delegations (`EMFloatFormer` and `LocalEquivConv16QAM` execute genuine $C^*$-equivariant architectures; `bussgang_lmmse` and `iterative_dd_pa_cancellation` are invoked solely as benchmark baselines inside `run_p4a_comparative_benchmarks`).
  - Zero zero-init / rank-1 bypasses (`head_h`/`head_dist` initialized with `0.01 / sqrt(2 * d_model)` normal weights; `JointEmbedding` projections full-rank).
  - `ComplexLinear(..., bias=False)`: Verified `bias is None`, `bias_r is None`, `bias_i is None` across every `ComplexLinear` submodule in `EMFloatFormer` and `LocalEquivConv16QAM`.
  - `JointPhysicalLoss.pa_residual_loss`: Uses `norm="ortho"` on both `torch.fft.ifft` and `torch.fft.fft` (`p4/em_floatformer.py:871, 878`), preserving Parseval energy parity.
  - AST Scan (`p4/em_floatformer.py`, `p4/__init__.py`, `p4/tests/test_p4_a_equivariance.py`, `p4/tests/test_gauge_stress_adversarial.py`): `Pass=[]`, `AssertTrue=[]`, `Skips=[]` (zero stub `pass` statements, zero `assert True` trivialities, zero skipped tests).
- **Phase 2 — Runtime Gradient, Singularity & `challenger_m2_3` Diagnostics**: **PASS**
  - Independent gradient trace under `JointPhysicalLoss`: `0` missing, `0` zero, `0` non-finite gradients across `EMFloatFormer(depth=2)` (`104` params), `LocalEquivConv16QAM` (`56` params), `TinyJoint16QAM` (`12` params), and all 8 `create_floatformer` configs (`58/90/68/104` params).
  - Exact `challenger_m2_3` 50-step `seed=11` `30 dB` `strong` PA `256`-frame diagnostic: `c64_max = 7.629395e-06 < 1e-5` (`flips=0`), `c128_max = 1.314504e-13 < 1e-11` (`flips=0`).
  - Extreme numerical singularities: `compute_gauge_anchor(1e-35)` produces `s0 = 1.000e-35` and unit RMS `[1.0, 1.0]`; `ComplexRMSNorm(1e-35)` yields finite forward outputs and finite `z_sub.grad` and `gamma.grad`; `RadialActivation` (`silu`, `gelu`, `relu`, `tanh`) at `1e20 + 1e20j` yields finite forward outputs and finite gradients.
- **Phase 3 — Bit-Exact Benchmark Replay & Helper Determinism**: **PASS**
  - After perturbing global RNG with `torch.randn(19)`, `run_p4a_comparative_benchmarks(output_dir=None)` matched `p4/results/p4a_comparative_benchmark.json` (`train_final_scale_ratio` and all `75` rows) and `p4/results/p4a_comparative_benchmark.csv` (all `75` rows string-for-string) **100% bit-exact**.
  - All `20` diagnostic/audit helpers in `p4/em_floatformer.py` returned bit-identical outputs between `torch.randn(13)` and `torch.randn(47)`.
- **Phase 4 — Full Test Suites & Repository Governance**: **PASS**
  - `p4.tests.test_p4_0_simulator`: `Ran 14 tests in 2.203s — OK` (exit `0`).
  - `p4.tests.test_gauge_stress_adversarial`: `Ran 5 tests in 18.612s — OK` (exit `0`).
  - `p4.tests.test_p4_a_equivariance`: `Ran 53 tests in 57.444s — OK` (exit `0`).
  - `tests.test_p4_e2e`: `Ran 253 tests in 15.967s — OK (skipped=97)` (`156` active passed, `97` M3/M4 cleanly skipped, exit `0`).
  - `git diff --name-only`: empty (both pre- and post-tests).
  - Stage P4-0 files (`p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, `p4/tests/test_p4_0_simulator.py`): original `2026-11-15` timestamps intact.
  - `MANIFEST_SHA256.json`: `115` entries, `Missing: []`, `114/115` byte-exact under LF normalization (sole diff is `README.md`, modified pre-P4 in commit `f878a73`).

---

## 1. Observation (Direct Tool Outputs & File Evidence)


### 1.2 Static Code & AST Forensics
- Inspected `p4/em_floatformer.py` (lines 1–1817), `p4/__init__.py` (lines 1–134), `p4/tests/test_p4_a_equivariance.py` (lines 1–904), and `p4/tests/test_gauge_stress_adversarial.py` (lines 1–314):
  - `p4/em_floatformer.py:42-67` (`compute_gauge_anchor`, `apply_gauge_normalization`): `complex128`/`float64`, `amax` rescaling, `eps=1e-38`.
  - `p4/em_floatformer.py:75-102` (`ComplexRMSNorm`): `complex128`/`float64`, `active = m > 1e-38`, factored `z_scaled * (gamma64 / rms_safe)`.
  - `p4/em_floatformer.py:165-179` (`ComplexLinear`): when `bias=False`, `self.register_parameter("bias_r", None)`, `self.register_parameter("bias_i", None)`, `self.bias = None`.
  - `p4/em_floatformer.py:250-281` (`RadialActivation`): `complex128`, `m_safe = torch.where(m > 1e150, m, torch.ones_like(m))`, smooth `(self.eps / m_safe) ** 2` guard.
  - `p4/em_floatformer.py:568-585` (`EMFloatFormer.reset_parameters`): non-zero normal init on `embed`, `head_h`, `head_dist`, and `1+0j` on `gamma_eff`.
  - `p4/em_floatformer.py:651-653` (`EMFloatFormer.forward`) & `1489-1491` (`LocalEquivConv16QAM.forward`): `sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 3.5e-2`, `llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)`.
  - `p4/em_floatformer.py:871, 878` (`JointPhysicalLoss.pa_residual_loss`): uses `norm="ortho"` on both `ifft` and `fft`.
  - `p4/em_floatformer.py:1496-1577` (`TinyJoint16QAM`): connects `stem`, `conv1`, `conv2`, `out_llr`, `head_h`, `head_dist` in `forward`.
  - `p4/em_floatformer.py:1605-1623, 1683-1714`: `_train_comparative_model` calls `torch.manual_seed(seed)` before `model_or_factory()`.
- AST scan output:
  ```text
  p4/em_floatformer.py: Pass=[], AssertTrue=[], Skips=[]
  p4/__init__.py: Pass=[], AssertTrue=[], Skips=[]
  p4/tests/test_p4_a_equivariance.py: Pass=[], AssertTrue=[], Skips=[]
  p4/tests/test_gauge_stress_adversarial.py: Pass=[], AssertTrue=[], Skips=[]
  ```

### 1.3 Empirical Runtime Verification (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`)
- **Gradient & `bias=False` trace**:
  ```text
  EMFloatFormer(d=2): total=104, missing=[], zero=[], nonfin=[]
  LocalEquivConv16QAM: total=56, missing=[], zero=[], nonfin=[]
  TinyJoint16QAM: total=12, missing=[], zero=[], nonfin=[]
  create_floatformer(False,False,False): total=58, missing=[], zero=[], nonfin=[]
  create_floatformer(False,False,True): total=90, missing=[], zero=[], nonfin=[]
  create_floatformer(False,True,False): total=58, missing=[], zero=[], nonfin=[]
  create_floatformer(False,True,True): total=90, missing=[], zero=[], nonfin=[]
  create_floatformer(True,False,False): total=68, missing=[], zero=[], nonfin=[]
  create_floatformer(True,False,True): total=104, missing=[], zero=[], nonfin=[]
  create_floatformer(True,True,False): total=68, missing=[], zero=[], nonfin=[]
  create_floatformer(True,True,True): total=104, missing=[], zero=[], nonfin=[]
  ComplexLinear bias=False check: PASS
  ```
- **`challenger_m2_3` 50-step `seed=11` & singularities (`task-45`)**:
  ```text
  challenger_m2_3 50-step seed=11: c64_max=7.629395e-06, flips64=0, c128_max=1.314504e-13, flips128=0
  anchor 1e-35: s0=[1.0000000180025095e-35, 1.0000000180025095e-35], rms=[1.0, 1.0]
  ComplexRMSNorm 1e-35: out_fin=True, grad_fin=True, gamma_grad_fin=True
  RadialActivation 1e20 forward+backward (silu/gelu/relu/tanh): PASS
  ```
- **Parseval & $C^*$ equivariance check**:
  ```text
  EMFloatFormer C* equivariance: llr_err=2.861e-06, h_rel=7.426e-08
  LocalEquivConv16QAM C* equivariance: llr_err=1.907e-06, h_rel=7.397e-08
  PARSEVAL_AND_EQUIVARIANCE_INDEPENDENT_PASS
  ```
- **Benchmark replay & 20-helper determinism (`task-50`)**:
  ```text
  BENCHMARK_REPLAY_100PCT_BIT_EXACT: 75 rows JSON+CSV+scale_ratio MATCH
  ALL_20_HELPERS_DETERMINISTIC: 20/20 exact match
  ```
- **All 4 unittest suites (`task-56`)**:
  ```text
  === SUITE 1: test_p4_0_simulator ===
  Ran 14 tests in 2.203s
  OK
  EXIT=0
  === SUITE 2: test_gauge_stress_adversarial ===
  Ran 5 tests in 18.612s
  OK
  [Scenario 2 Passed] 206 Phase Angles Tested. Max logit diff: 6.869435e-06, Total bit flips: 0
  [Scenario 1 Passed] 50 Scales Tested. Max logit diff: 3.814697e-06, Total bit flips: 0
  [Scenario 4 Passed] Zero input boundary handled safely without NaNs/Infs.
  [Scenario 4 Passed] Extreme scales [1e-20, 1e-15, 1e-10, 100000.0, 10000000000.0] checked without any NaN/Inf propagation.
  [Scenario 3 Passed] 50 (r, phi) Pairs Tested. Max H rel err: 1.781575e-07, Max LLR diff: 5.245209e-06, Bit flips: 0
  EXIT=0
  === SUITE 3: test_p4_a_equivariance ===
  Ran 53 tests in 57.444s
  OK
  EXIT=0
  === SUITE 4: test_p4_e2e ===
  Ran 253 tests in 15.967s
  OK (skipped=97)
  EXIT=0
  ```
- **Repository Governance & Manifest**:
  - `git diff --name-only`: empty (`0` tracked files modified, checked both pre- and post-test).
  - `p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, `p4/tests/test_p4_0_simulator.py`: `LastWriteTime` preserved at `2026/11/15`.
  - `MANIFEST_SHA256.json`: `Total entries: 115`, `Missing: []`, `Raw diffs count: 59`, `LF-normalized diffs (1): ['README.md']` (`README.md` modified in pre-P4 commit `f878a73`).

---

## 2. Logic Chain
2. Phase 1 static code and AST inspections establish the absence of any hardcoded test results, clamping bypasses, facade delegations, zero-init/rank-1 shortcuts, missing `norm="ortho"`, biased `ComplexLinear`, or trivial/mocked test assertions.
3. Independent runtime traces confirm 100% healthy non-zero finite gradients across all parameter tensors of `EMFloatFormer`, `LocalEquivConv16QAM`, `TinyJoint16QAM`, and all 8 `create_floatformer` configs; confirm `challenger_m2_3`'s 50-step `seed=11` diagnostic (`< 1e-5` in `complex64`, `< 1e-11` in `complex128`, `0` flips); and confirm `1e-35` and `1e20` singularity stability.
4. Independent RNG-perturbed replay confirms 100% bit-exact parity of `p4/results/p4a_comparative_benchmark.{json,csv}` across all 75 rows and `train_final_scale_ratio`, plus 20/20 helper determinism.
5. Independent execution of all 4 verification suites (`14/14`, `5/5`, `53/53`, `156/156 active`), `git diff --name-only`, P4-0 timestamps, and `MANIFEST_SHA256.json` parity (`114/115` LF-normalized, sole diff pre-P4 `README.md`) confirms end-to-end compliance.
6. Mapping all Phase 1 observations against `development` mode (and even stricter modes) yields zero integrity violations.

---

## 3. Caveats
- All empirical checks and tests were executed on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`. GPU and mixed-precision (`FP16`/`BF16`) verification are deferred to Milestone 4 (`F19` in `PROJECT.md`).
- Running `tests.test_p4_e2e` invokes `tests/e2e/test_tier2_boundary_corner.py`, which updates `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime; Stage P4-0 source/test files remain untouched (`2026-11-15`).
- Consistent with `reviewer_m2_r2_2` Minor Observation 1, if a caller supplies identically zero `yp` with `yp.requires_grad=True`, `compute_gauge_anchor` evaluates `sqrt(0.0)` inside `torch.where` causing `yp.grad` to be `NaN`; all model parameter gradients and forward outputs remain finite, and `yp` is an undifferentiated input observation in all Stage P4 pipelines.

---

## 4. Conclusion
**Verdict: CLEAN.**
Milestone 2 (Stage P4-A: Features F8–F12) Iteration 2 passes all forensic integrity, static/runtime verification, RNG-perturbed bit-exact replay, full test suite, and repository governance checks with zero integrity violations.

---

## 5. Verification Method
Run from ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_gauge_stress_adversarial -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -c "import json, torch; from pathlib import Path; from p4.em_floatformer import run_p4a_comparative_benchmarks; torch.set_num_threads(4); torch.randn(19); s=json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8')); r=run_p4a_comparative_benchmarks(output_dir=None); assert s==r; print('EXACT_REPRO_MATCH')"
git diff --name-only
```
**Invalidation conditions**: Any failure/error in any of the 4 unittest suites, any mismatch between RNG-perturbed `run_p4a_comparative_benchmarks(output_dir=None)` and saved JSON/CSV artifacts, any `NaN`/`Inf`/`0.0`/missing parameter gradient, any `< 1e-5` (`complex64`) or `< 1e-11` (`complex128`) equivariance failure on `challenger_m2_3` `seed=11`, or any modification to tracked historical files.
