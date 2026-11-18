**Verdict: APPROVE**

# Milestone 2 Independent Review & Adversarial Challenge Report (`reviewer_m2_r2_1`)

## Review Summary

**Verdict**: `APPROVE` — All Iteration 1 blocking issues (`reviewer_m2_r1_2` Finding 1 & Finding 2) are cleanly resolved, bit-for-bit reproducibility of `p4/results/p4a_comparative_benchmark.{json,csv}` under global RNG perturbation is independently verified, the `3.5e-2` LLR variance floor + `20.0 * torch.tanh(llrs64 / 20.0)` bound satisfies both `complex64` (`< 1e-5`, 0 flips) and `complex128` (`< 1e-11`, 0 flips) on `challenger_m2_3`'s 50-step `seed=11` diagnostic as well as calibration (`ECE < 0.05`), zero `DEAD_ENDS.md` regressions or integrity violations exist, and all 4 test suites plus governance checks pass.

### Findings

- **Zero Critical or Major Findings.**
- **[Note] Minor Line-Citation Drift in `worker_m2_r2_1/handoff.md`**: `worker_m2_r2_1/handoff.md:17` cites lines `653-655` and `1487-1489` for the `3.5e-2` floor and `20.0 * tanh`, whereas in `p4/em_floatformer.py` the exact lines are `651-653` (`EMFloatFormer.forward`) and `1489-1491` (`LocalEquivConv16QAM.forward`). Code implementation itself is exact.

### Verified Claims (`worker_m2_r2_1` & Scope Items)

- **Factory RNG Seeding (`_train_comparative_model` & `run_p4a_comparative_benchmarks`)** → verified via source inspection (`p4/em_floatformer.py:1605-1623, 1681-1714`) + live execution after `torch.randn(19)` → **PASS** (exact dict & CSV parity across all 75 rows and `train_final_scale_ratio`).
- **Deterministic Seeding in `evaluate_noise_data_feedback`, `evaluate_perfect_pilot_demapping`, and Audit Helpers** → verified via source inspection (`p4/em_floatformer.py:113, 136, 679, 735, 762, 912, 960, 998, 1034, 1061, 1089, 1121, 1164, 1189, 1213, 1231, 1257, 1288`) + live RNG-perturbation test (`torch.randn(13)` vs `torch.randn(29)`) across all 20 helpers → **PASS** (`ALL_20_HELPERS_DETERMINISTIC_PASS`).
- **`3.5e-2` LLR Variance Floor + `20.0 * torch.tanh(llrs64 / 20.0)`** → verified in both `EMFloatFormer.forward` (`p4/em_floatformer.py:651-653`) and `LocalEquivConv16QAM.forward` (`p4/em_floatformer.py:1489-1491`) + `p4.tests.test_p4_a_equivariance` (`test_challenger_m2_3_50step_seed11_30db_strong_256frames`, `test_calibrated_llr_dynamic_range_ece_and_equivariance_high_snr`) → **PASS**.
- **Zero `DEAD_ENDS.md` Regressions or Integrity Issues** → verified via line-by-line inspection of `p4/em_floatformer.py` → **PASS**:
  - `compare_pilot_vs_bidirectional` (`lines 673-726`): genuine 50-step training loop, unclamped `gain_db`, no delegation to `iterative_dd_pa_cancellation`.
  - `evaluate_noise_data_feedback` (`lines 729-753`): unclamped `deg_db`.
  - `JointPhysicalLoss.pa_residual_loss` (`lines 859-886`): uses `norm="ortho"` on both `ifft` and `fft` (`lines 871, 878`).
  - `ComplexLinear` (`lines 157-201`): enforces `self.bias = None` when `bias=False` (`line 178`).
  - `ComplexRMSNorm` (`lines 75-102`): `complex128` factored normalization `z_scaled * (gamma64 / rms_safe)` with `active = m > 1e-38` (`lines 91-102`).
  - `RadialActivation` (`lines 250-281`): `complex128` with `m > 1e150` rescaling and smooth `(self.eps / m_safe) ** 2` zero guard (`lines 263-281`).
  - `EMFloatFormer.reset_parameters` (`lines 568-584`): non-zero `0.01 / sqrt(2*d_model)` init on `head_h` and `head_dist`, full-rank `JointEmbedding`.
- **All 53 Tests in `p4/tests/test_p4_a_equivariance.py` Genuine** → verified by inspecting `lines 1-904` (no `pass` stubs, no `assert True`, genuine physical/gradient/determinism assertions including `challenger_m2_3` 50-step `seed=11`, `_resolve_pa_A` edge cases, `ComplexConv1d`/`LocalEquivConv16QAM`/`TinyJoint16QAM`, RNG-perturbation determinism, and 75-row JSON/CSV parity) → **PASS**.
- **Stage P4-0 & `MANIFEST_SHA256.json` Immutability** → verified via `git diff --name-only`, file timestamps on P4-0 deliverables (`2026-11-15`), and SHA256 audit of all 115 `MANIFEST_SHA256.json` entries → **PASS**.

### Coverage Gaps

- **GPU & Mixed-Precision (`FP16`/`BF16`) Execution** — risk level: **low** (for M2 CPU sign-off) — recommendation: **accept risk for M2**, explicitly scheduled for Milestone 4 (`F19` in `PROJECT.md`).

---

## Challenge Summary (Adversarial Assessment)

**Overall risk assessment**: **LOW**.

- **RNG Perturbation Stress Test**: Verified `run_p4a_comparative_benchmarks(output_dir=None)` after `torch.randn(19)` and all 20 diagnostic/audit helpers between `torch.randn(13)` and `torch.randn(29)`. Zero ambient RNG leakage remains.
- **24-Octave / Singularity / 50-Step `seed=11` Stress**: Verified via `test_gauge_stress_adversarial` (50 scales across `[2^-12, 2^12]` max diff `3.81e-06` / `0` flips; 206 angles max diff `6.87e-06` / `0` flips; 50 simultaneous `(r, phi)` pairs max H rel err `1.78e-07`, max LLR diff `5.25e-06` / `0` flips; `1e-20` to `1e+10` and all-zero finite) and `test_challenger_m2_3_50step_seed11_30db_strong_256frames` (`complex64 < 1e-5` & `complex128 < 1e-11`, `0` flips).
- **`TinyJoint16QAM` High-SNR Plateau Assessment**: At `clean` `25 -> 30 dB`, `TinyJoint16QAM` (`seed=2028`, `1,880` params) yields `380` vs `384` bit errors out of `32,768` (`0.011597` vs `0.011719`, `+1.22e-4` across independent eval seeds `202623` vs `202624`) while dropping from `0.137451` at `10 dB`. This is genuine physical behavior of its fixed `+ 0.03` regularizer (`p4/em_floatformer.py:1533, 1560`) rather than a regression or hack.

---

## 1. Observation

1. **Test Suites Executed Independently (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`)**:
   - `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 2.636s — OK** (exit code 0).
   - `python -m unittest p4.tests.test_gauge_stress_adversarial -v`: **Ran 5 tests in 20.334s — OK** (exit code 0).
   - `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 53 tests in 88.663s — OK** (exit code 0).
   - `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 20.087s — OK (skipped=97)** (156 active M1/M2/F22 tests passed, 97 M3/M4 skipped, exit code 0).
2. **Bit-Exact Benchmark Reproducibility & 20-Helper Determinism**:
   - Calling `run_p4a_comparative_benchmarks(output_dir=None)` after `torch.randn(19)` with `torch.set_num_threads(4)` matched `p4/results/p4a_comparative_benchmark.json` (`saved_json == repro`) and `p4/results/p4a_comparative_benchmark.csv` across all 75 rows (`BENCHMARK_REPRO_AND_CSV_PARITY_BIT_EXACT_PASS`, exit code 0).
   - Calling all 20 diagnostic/audit helpers (`audit_single_scale`, `audit_single_phase`, `compare_pilot_vs_bidirectional`, `evaluate_noise_data_feedback`, `evaluate_perfect_pilot_demapping`, `verify_loss_gradient_invariance`, `audit_loss_scale_invariance`, `train_step_joint_loss`, `check_loss_gradient_scale`, `audit_gain_equivariance`, `audit_phase_equivariance`, `audit_c_star_equivariance`, `audit_channel_equivariance`, `run_standalone_equivariance_check`, `sweep_phases_dense`, `check_irrational_phases`, `check_noise_quad_scaling`, `verify_gauge_attention_joint`, `audit_full_equivariance`, `verify_bidirectional_green_prop`) between `torch.randn(13)` and `torch.randn(29)` returned identical outputs (`ALL_20_HELPERS_DETERMINISTIC_PASS`, exit code 0).
3. **Governance & Immutability**:
   - `git diff --name-only`: empty (exit code 0).
   - `git status --short`: `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`.
   - `p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, `p4/tests/test_p4_0_simulator.py`: `LastWriteTime` preserved at `2026-11-15`.
   - `MANIFEST_SHA256.json`: 115 entries, `Missing: []`, 114/115 byte-exact under LF normalization; sole entry in `Diffs` is `README.md`, modified in pre-Stage-P4 commit `f878a73`.

---

## 2. Logic Chain

1. Observation 2 proves that refactoring `_train_comparative_model` to accept a model factory after `torch.manual_seed(seed)` and seeding all 20 diagnostic/audit helpers eliminates the unseeded global RNG dependency identified in `reviewer_m2_r1_2`.
2. Observation 1 (`53/53` in `test_p4_a_equivariance` and `5/5` in `test_gauge_stress_adversarial`) confirms that the `3.5e-2` LLR variance floor + `20.0 * torch.tanh(llrs64 / 20.0)` bound satisfies both `complex64` (`< 1e-5`, 0 flips) and `complex128` (`< 1e-11`, 0 flips) equivariance on trained 50-step `seed=11` `depth=2` models while maintaining `ECE < 0.05` and high-SNR LLR dynamic range (`> 15.0`).
3. Observation 1 (`14/14` in `test_p4_0_simulator`, `156/156` active in `tests.test_p4_e2e`) and Observation 3 (`git diff --name-only` empty, 114/115 `MANIFEST_SHA256.json` parity, preserved `2026-11-15` P4-0 timestamps) confirm full Stage P4-0 and historical P1–P3 immutability with zero `DEAD_ENDS.md` regressions or integrity violations.
4. Therefore, all Milestone 2 (Stage P4-A: F8–F12) acceptance criteria for Iteration 2 are satisfied.

---

## 3. Caveats

- All tests and benchmarks were executed on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`. GPU and mixed-precision (`FP16`/`BF16`) verification remain scoped to Milestone 4 (`F19`).
- Running `tests.test_p4_e2e` invokes `tests/e2e/test_tier2_boundary_corner.py`, which updates `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime (`threads=1, batch_size=1, timed_runs=3`).

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 2 (Stage P4-A: Features F8–F12) Iteration 2 is approved for Gate sign-off:
- `p4/em_floatformer.py` implements deterministic factory and helper seeding, `3.5e-2` LLR floor + `20.0 * tanh` calibration, and full $C^*$-equivariant numerical hardening with zero `DEAD_ENDS.md` regressions.
- `p4/tests/test_p4_a_equivariance.py` (53 tests), `p4/tests/test_gauge_stress_adversarial.py` (5 tests), `p4/tests/test_p4_0_simulator.py` (14 tests), and `tests/test_p4_e2e.py` (156 active passed, 97 M3/M4 skipped) all pass with exit code 0.
- `p4/results/p4a_comparative_benchmark.{json,csv}` is 100% bit-for-bit reproducible under global RNG perturbation, and historical P1–P3 / Stage P4-0 artifacts remain intact.

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
assert saved == repro
print('EXACT_REPRO_MATCH_100PCT')
"
git diff --name-only
```
**Invalidation conditions**: Any failure in the 4 unittest suites, any mismatch between `run_p4a_comparative_benchmarks(output_dir=None)` and `p4/results/p4a_comparative_benchmark.{json,csv}` after RNG perturbation, or any tracked historical file appearing in `git diff --name-only`.
