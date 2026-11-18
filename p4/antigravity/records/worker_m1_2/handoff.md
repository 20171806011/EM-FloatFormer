# Handoff Report: Milestone 1 (Stage P4-0) Remediation

**Worker:** Worker 2 (`worker_m1_2`)
**Role:** Implementer / QA / Specialist
**Working Directory:** ``
**State Directory:** `worker_m1_2`
**Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines — Iteration 2 Remediation)
**Date:** 2026-11-15

---

## 1. Observation

### 1.1 Root Cause Verification of Flaky Test
- In `p4/tests/test_p4_0_simulator.py`:
  Prior implementation of `test_f5_llr_sign_matches_hard_slicing` tested `calibrated_16qam_llrs(z, sigma2, method="exact")` against `hard_slice_16qam(z)` with `self.assertTrue(torch.all(hard_from_llr == hard_from_slice))`.
- As mathematically proven in `reviewer_m1_1/handoff.md`, the exact Bayesian posterior LLR incorporates all 4 Gaussian constellation tails:
  $$LLR(b_1) = \ln \frac{e^{-(x-pts_1)^2/(2\sigma^2)} + e^{-(x-pts_2)^2/(2\sigma^2)}}{e^{-(x-pts_0)^2/(2\sigma^2)} + e^{-(x-pts_3)^2/(2\sigma^2)}}$$
  Because $pts_1 = -1/\sqrt{10}$ is closer to the decision midpoint $x = 2/\sqrt{10}$ than $pts_0 = -3/\sqrt{10}$, $e^{-(x-pts_1)^2/(2\sigma^2)} > e^{-(x-pts_0)^2/(2\sigma^2)}$, shifting the true posterior zero-crossing outward by $\delta \approx 2.6 \times 10^{-5}$ ($x^* \approx 0.632482$ vs. Voronoi threshold $0.632456$).
- On continuous unseeded Gaussian noise with 6400 real coordinates, an unseeded coordinate had an $\approx 15\% - 20\%$ probability of falling in that boundary strip, triggering intermittent unit test failure.

### 1.2 Benchmark Artifact Mutation in Unit Test
- In `p4/tests/test_p4_0_simulator.py`:
  `test_f7_benchmark_artifacts_generation` previously executed `run_benchmarks(..., output_dir="p4/results", warmup_runs=5, timed_runs=10)`.
- This directly mutated `p4/results/latency_cpu.csv` and `p4/results/timing.json`, clobbering the high-precision 100-run production benchmark data with 10-run unit test data.

### 1.3 Truncation Risk in `delay_est_dft`
- In `p4/classical_baselines.py`:
  `delay_est_dft` previously had signature `def delay_est_dft(yp: torch.Tensor, taps: int = 8, n: int = DEFAULT_N) -> torch.Tensor:`.
  If called with $N=64$ frequency bins without explicitly setting `n=64`, `F.pad` padded to $32$ instead of $64$, truncating the output tensor.

---

## 2. Logic Chain

1. **Test Stabilization (`test_f5_llr_sign_matches_hard_slicing`)**:
   - For `method="maxlog"`, the log-sum-exp is approximated by the minimum Euclidean distance, strictly reproducing the Voronoi partitioning at boundary $2.0 / \sqrt{10}$. Thus, `maxlog` has a 100% mathematical guarantee of matching `hard_slice_16qam` on all coordinates across 100% of seeds: `self.assertTrue(torch.all(hard_from_llr == hard_from_slice))`.
   - For `method="exact"`, the test asserts `agreement > 0.999` (`self.assertGreater(agreement, 0.999)`), accurately reflecting the continuous Bayesian tail contribution without false-positive failures.
   - Seed repeatability is reinforced using `self.generator`.

2. **Benchmark Isolation (`test_f7_benchmark_artifacts_generation`)**:
   - Refactored `test_f7_benchmark_artifacts_generation` to execute inside `tempfile.TemporaryDirectory() as tmp_dir`, writing temporary test artifacts exclusively to `tmp_dir`.
   - Unit tests verify artifact creation, non-emptiness, schema adherence, and latency positivity without touching `p4/results/`.
   - Re-ran production benchmark harness: `python -m p4.benchmark_latency --warmup 25 --runs 100` generating clean production artifacts `p4/results/latency_cpu.csv` and `p4/results/timing.json` with `warmup_runs: 25` and `timed_runs: 100` on 4 CPU threads.

3. **Dynamic Dimension Handling (`delay_est_dft`)**:
   - Changed signature to `def delay_est_dft(yp: torch.Tensor, taps: int = 8, n: Optional[int] = None) -> torch.Tensor:`.
   - Added dynamic shape resolution: `n = yp.shape[-1] if n is None else n`.
   - Supports arbitrary subcarrier dimensions (e.g. $N=32, 64, 128$) transparently without requiring caller boilerplate.

4. **10-Run Stress Verification**:
   - Executed 10 consecutive programmatic test runs: `python -m unittest p4/tests/test_p4_0_simulator.py -v`.
   - Result: 10/10 runs PASSED with 0 failures (140/140 tests passed, 0 flaky tests).
   - Confirmed `p4/results/timing.json` maintained `warmup_runs: 25, timed_runs: 100` before and after all 10 runs.

5. **Historical Integrity**:
   - `git status --short` verified that zero historical files (P1–P3, exploratory, project manifests) were modified.

---

## 3. Caveats

- **Test Isolation**: The isolation implemented in `test_f7_benchmark_artifacts_generation` protects `p4/results/` from unit test overwrites. If external audit scripts (such as `tests/e2e/test_tier1_feature_coverage.py`) are run with explicit `--output-dir p4/results`, they will write to `p4/results/`. For production releases, running `python -m p4.benchmark_latency --warmup 25 --runs 100` generates the authoritative 100-run dataset.
- **Hardware Profile**: Latency benchmarks were executed on an Intel x86_64 host with PyTorch 2.12.1+cpu on 4 CPU threads.

---

## 4. Conclusion

All requested remediations for Milestone 1 (Stage P4-0) have been implemented and verified:
1. `p4/tests/test_p4_0_simulator.py`: `test_f5_llr_sign_matches_hard_slicing` is 100% stable; `test_f7_benchmark_artifacts_generation` is isolated via `tempfile.TemporaryDirectory()`.
2. `p4/classical_baselines.py`: `delay_est_dft` dynamically resolves `n = yp.shape[-1]` when `n is None`.
3. `p4/results/`: Clean production latency benchmark artifacts (`latency_cpu.csv` and `timing.json`) generated with 25 warmup and 100 timed iterations.
4. Test suite reliability: 10/10 consecutive executions passed cleanly with zero failures.

---

## 5. Verification Method

To independently verify this remediation, run the following commands:

1. **Verify 10 Consecutive Full Test Suite Runs**:
   ```powershell
   cd
   python -c "
   import subprocess, sys
   for i in range(1, 11):
       res = subprocess.run([sys.executable, '-m', 'unittest', 'p4/tests/test_p4_0_simulator.py', '-v'], capture_output=True, text=True)
       if res.returncode != 0 or 'Ran 14 tests' not in res.stderr or 'OK' not in res.stderr:
           print(f'Run {i} failed: {res.stderr}')
           sys.exit(1)
       print(f'Run {i}/10 PASSED (OK)')
   print('ALL 10 CONSECUTIVE TEST RUNS PASSED 10/10 OK!')
   "
   ```

2. **Verify Static Parity Check (< 2e-7 Relative Error)**:
   ```powershell
   python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"
   ```

3. **Verify Production Benchmark Artifacts & Isolation**:
   ```powershell
   python -c "
   import json
   with open('p4/results/timing.json') as f:
       d = json.load(f)
   res = d['results'][0]
   assert res['warmup_runs'] == 25 and res['timed_runs'] == 100, f'Unexpected: {res}'
   print('Production benchmark verified: 25 warmup, 100 runs on 4 threads.')
   "
   ```

4. **Verify Historical File Cleanliness**:
   ```powershell
   git status --short
   ```
   *Expected*: Zero tracked historical files modified.
