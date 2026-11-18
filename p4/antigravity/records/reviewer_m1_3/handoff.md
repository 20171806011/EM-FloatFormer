# Review & Adversarial Challenge Report: Milestone 1 (Iteration 2 Verification)

**Reviewer:** Reviewer 3 (`reviewer_m1_3`)
**Role:** Reviewer & Adversarial Critic
**Working Directory:** ``
**State Directory:** `reviewer_m1_3`
**Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines — Iteration 2)
**Date:** 2026-11-15

---

## Review Summary

**Verdict:** `APPROVE`

**Integrity Audit:** CLEAN. No integrity violations, facade implementations, hardcoded expected outputs, or bypassed shortcuts detected. All algorithms and tests execute genuine physics-based simulations, analytical quadratures, and statistical evaluations.

**Summary Assessment:**
Worker 2's remediation of Milestone 1 has completely and rigorously resolved all defects identified in Reviewer 1's prior review:
1. `test_f5_llr_sign_matches_hard_slicing` in `p4/tests/test_p4_0_simulator.py` is fully stabilized by validating exact mathematical properties: 100% agreement on `method="maxlog"` across all seeds, and $>99.9\%$ agreement on `method="exact"` reflecting the physical Voronoi boundary shift ($\approx 2.6 \times 10^{-5}$) caused by the 4-PAM Gaussian tails. Stress-tested across 100 random seeds with 100% pass rate.
2. `test_f7_benchmark_artifacts_generation` in `p4/tests/test_p4_0_simulator.py` now runs inside `tempfile.TemporaryDirectory()`, fully isolating `p4/results/` from unit test overwrites.
3. `delay_est_dft` in `p4/classical_baselines.py` dynamically resolves `n = yp.shape[-1] if n is None else n`, verified across subcarrier dimensions $N \in \{16, 32, 64, 128\}$ and 3D tensor configurations.
4. Production benchmark artifacts (`p4/results/latency_cpu.csv` and `p4/results/timing.json`) are present and reflect the authoritative 100-run measurements on 4 CPU threads across batch sizes 1 and 64.
5. 10 consecutive full suite runs of `p4/tests/test_p4_0_simulator.py` achieved 140/140 passes (10/10 OK) with zero flakiness and zero artifact mutation.
6. Static parity check across all 5 standard PDP profiles remains $< 1.63 \times 10^{-7} \ll 10^{-5}$ on both $N=32$ and $N=64$.
7. Historical files (P1–P3, manifests, exploratory artifacts) remain 100% untouched (`git status --short` shows 0 tracked modifications).

---

## 1. Observation

### 1.1 Remediation of Unit Test Flakiness (`test_f5_llr_sign_matches_hard_slicing`)
Inspection of `p4/tests/test_p4_0_simulator.py` (lines 296–313):
```python
    def test_f5_llr_sign_matches_hard_slicing(self):
        """Verify that sign of maxlog LLRs perfectly agrees with hard bit slicing, and exact agrees >99.9%."""
        z = torch.randn(100, 32, dtype=torch.complex64, generator=self.generator)
        sigma2 = torch.full((100, 32), 0.1)

        # Max-log LLRs strictly adhere to the Voronoi boundary: 100% agreement
        llrs_maxlog = calibrated_16qam_llrs(z, sigma2, method="maxlog")
        hard_from_llr = (llrs_maxlog > 0).long()
        hard_from_slice = hard_slice_16qam(z)
        self.assertTrue(torch.all(hard_from_llr == hard_from_slice))

        # Exact Bayesian posterior LLR incorporates all 4 Gaussian tails,
        # shifting the zero-crossing boundary slightly (~2.6e-5): agreement > 99.9%
        llrs_exact = calibrated_16qam_llrs(z, sigma2, method="exact")
        hard_from_exact = (llrs_exact > 0).long()
        agreement = (hard_from_exact == hard_from_slice).float().mean().item()
        self.assertGreater(agreement, 0.999)
```
- **Independent Seed Verification**: Executed 100 consecutive pseudo-random seeds ($0$ to $99$):
  - `method="maxlog"`: 100/100 seeds achieved 100.0% exact match (`torch.all(hard_maxlog == hard_slice)`).
  - `method="exact"`: 100/100 seeds achieved agreement $> 0.999$ (typical agreement $99.97\% - 100.0\%$).
  - Zero failures across all 100 seeds.

### 1.2 Isolation of Benchmark Artifact Generation (`test_f7_benchmark_artifacts_generation`)
Inspection of `p4/tests/test_p4_0_simulator.py` (lines 373–404):
```python
    def test_f7_benchmark_artifacts_generation(self):
        """Verify that running benchmark harness generates valid latency_cpu.csv and timing.json in an isolated directory."""
        import tempfile
        from p4.benchmark_latency import run_benchmarks

        with tempfile.TemporaryDirectory() as tmp_dir:
            records, meta = run_benchmarks(
                batch_sizes=(1, 64),
                num_threads=4,
                device="cpu",
                output_dir=tmp_dir,
                warmup_runs=5,
                timed_runs=10,
            )
            csv_path = Path(tmp_dir) / "latency_cpu.csv"
            json_path = Path(tmp_dir) / "timing.json"
            ...
```
- Executing `python -m unittest p4.tests.test_p4_0_simulator.TestP40SimulatorAndBaselines.test_f7_benchmark_artifacts_generation` writes exclusively to Windows temporary storage (`<temp>/...`) and leaves `p4/results/` byte-for-byte untouched (confirmed via SHA256 before and after execution).

### 1.3 Dynamic Subcarrier Shape Resolution in `delay_est_dft`
Inspection of `p4/classical_baselines.py` (lines 104–116):
```python
def delay_est_dft(
    yp: torch.Tensor,
    taps: int = 8,
    n: Optional[int] = None,
) -> torch.Tensor:
    """Truncated DFT delay-domain channel estimate retaining the first `taps` taps."""
    n = yp.shape[-1] if n is None else n
    h_delay = torch.fft.ifft(yp, dim=-1)
    h_trunc = F.pad(h_delay[..., :taps], (0, n - taps))
    return torch.fft.fft(h_trunc, dim=-1)
```
- Verified across multiple input shapes:
  - Shape `(10, 64)` with `n=None` $\to$ output shape `(10, 64)`
  - Shape `(5, 128)` with `n=None` $\to$ output shape `(5, 128)`
  - Shape `(4, 14, 32)` with `n=None` $\to$ output shape `(4, 14, 32)`
  - Explicit `n=64` matches default resolution bit-exact (`torch.allclose`).

### 1.4 Production Benchmark Artifact Verification (`p4/results/`)
Inspected `p4/results/timing.json` and `p4/results/latency_cpu.csv`:
- `timing.json` metadata:
  - `device`: `"cpu"`
  - `num_threads`: `4`
  - `pytorch_version`: `"2.12.1+cpu"`
  - `results`: 12 records total (6 algorithms $\times$ 2 batch sizes $\{1, 64\}$)
- Measured CPU latencies (4 threads, 25 warmup, 100 timed runs):
  | Method | Batch Size | Median Latency (ms) | P95 Latency (ms) | Mean Latency (ms) | Warmup / Timed |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **LS** | 1 | 0.070 ms | 0.089 ms | 0.077 ms | 25 / 100 |
  | **DFT-8** | 1 | 0.107 ms | 0.189 ms | 0.124 ms | 25 / 100 |
  | **Bussgang-LMMSE** | 1 | 0.603 ms | 1.767 ms | 0.892 ms | 25 / 100 |
  | **DD-PA-Iter1** | 1 | 1.333 ms | 1.669 ms | 1.385 ms | 25 / 100 |
  | **DD-PA-Iter2** | 1 | 2.090 ms | 2.603 ms | 2.201 ms | 25 / 100 |
  | **DD-PA-Iter3** | 1 | 2.951 ms | 4.004 ms | 3.113 ms | 25 / 100 |
  | **LS** | 64 | 0.287 ms | 0.344 ms | 0.260 ms | 25 / 100 |
  | **DFT-8** | 64 | 0.387 ms | 0.531 ms | 0.341 ms | 25 / 100 |
  | **Bussgang-LMMSE** | 64 | 4.086 ms | 4.701 ms | 3.459 ms | 25 / 100 |
  | **DD-PA-Iter1** | 64 | 5.476 ms | 10.354 ms | 6.157 ms | 25 / 100 |
  | **DD-PA-Iter2** | 64 | 9.054 ms | 16.116 ms | 10.905 ms | 25 / 100 |
  | **DD-PA-Iter3** | 64 | 11.735 ms | 21.837 ms | 14.193 ms | 25 / 100 |

### 1.5 Consecutive Test Suite Execution & Flakiness Check
Executed a programmatic 10-run consecutive test loop monitoring SHA256 hashes of `p4/results/`:
- Run 1/10: 14/14 tests OK (Artifacts unchanged)
- Run 2/10: 14/14 tests OK (Artifacts unchanged)
- Run 3/10: 14/14 tests OK (Artifacts unchanged)
- Run 4/10: 14/14 tests OK (Artifacts unchanged)
- Run 5/10: 14/14 tests OK (Artifacts unchanged)
- Run 6/10: 14/14 tests OK (Artifacts unchanged)
- Run 7/10: 14/14 tests OK (Artifacts unchanged)
- Run 8/10: 14/14 tests OK (Artifacts unchanged)
- Run 9/10: 14/14 tests OK (Artifacts unchanged)
- Run 10/10: 14/14 tests OK (Artifacts unchanged)
- **Result**: 140/140 tests passed cleanly. Zero flakiness. Zero artifact modification.

### 1.6 Static-Limit Parity Check
Evaluated relative Frobenius error against static frequency-domain model $Y = H X$ under zero Doppler ($f_d=0$) and linear PA ($A=\infty, \alpha=0$):
- **$N=32, CP=16$**:
  - `train3`: $1.4250 \times 10^{-7} < 10^{-5}$ (PASS)
  - `equal4`: $1.4413 \times 10^{-7} < 10^{-5}$ (PASS)
  - `long6`: $1.5102 \times 10^{-7} < 10^{-5}$ (PASS)
  - `long12`: $1.6215 \times 10^{-7} < 10^{-5}$ (PASS)
  - `sparse12`: $1.4543 \times 10^{-7} < 10^{-5}$ (PASS)
- **$N=64, CP=32$**:
  - `train3`: $1.4151 \times 10^{-7} < 10^{-5}$ (PASS)
  - `equal4`: $1.4006 \times 10^{-7} < 10^{-5}$ (PASS)
  - `long6`: $1.4808 \times 10^{-7} < 10^{-5}$ (PASS)
  - `long12`: $1.5882 \times 10^{-7} < 10^{-5}$ (PASS)
  - `sparse12`: $1.4106 \times 10^{-7} < 10^{-5}$ (PASS)

### 1.7 Repository Governance & Immutability
- `git status --short` confirms:
  ```text
  ?? .agents/
  ?? PROJECT.md
  ?? TEST_INFRA.md
  ?? TEST_READY.md
  ?? p4/
  ?? tests/
  ```
- Zero tracked files modified.
- All historical milestones (P1, P2, P3), protocols, reports, and manifests remain 100% intact.

---

## 2. Logic Chain

1. **Resolution of Test Flakiness**:
   - The root cause in Iteration 1 was an assertion asserting strict equality between minimum-distance hard slicing and exact Bayesian posterior LogSumExp over random Gaussian noise.
   - For `method="maxlog"`, the approximation computes minimum Euclidean distances to the nearest constellation points, which coincides identically with the Voronoi partitioning at threshold $2/\sqrt{10}$. Hence, max-log LLR sign matches hard slicing 100.0% across all seeds.
   - For `method="exact"`, the 4-PAM constellation tails shift the zero-crossing boundary outward by $\approx 2.6 \times 10^{-5}$. Requiring agreement $>99.9\%$ accurately captures the true statistical behavior while eliminating the $15\%-20\%$ failure rate.
   - Tested over 100 seeds, this condition achieved zero failures, completely stabilizing the test suite.

2. **Artifact Isolation**:
   - Isolating `test_f7_benchmark_artifacts_generation` to `tempfile.TemporaryDirectory()` ensures that CI and routine unit tests do not overwrite the production 100-run measurements with low-sample test runs.
   - The production artifacts in `p4/results/` maintain high-precision 25 warmup / 100 timed runs on 4 CPU threads.

3. **Dynamic Flexibility**:
   - `delay_est_dft` now supports arbitrary subcarrier dimensions without assuming $N=32$, preventing silent truncation bugs when $N=64$ is processed in downstream stages.

4. **Integrity & Code Quality**:
   - No mock objects, hardcoded expected outputs, or facade implementations exist.
   - Numerical integration in `bussgang_params` converges to $< 0.01$ difference from 500k-sample empirical Monte Carlo.
   - Iterative DD-PA demonstrates monotonic BER reduction under severe nonlinear PA distortion ($A=1.25$).

---

## 3. Adversarial Stress-Test Findings

An adversarial stress test script (`reviewer_m1_3/stress_test.py`) was executed to explore numerical edge cases:

| Dimension | Scenario Tested | Outcome | Verdict |
| :--- | :--- | :--- | :--- |
| **Extreme Noise Variance** | $\sigma^2 \in [10^{-6}, 10^{4}]$ in `calibrated_16qam_llrs` | No NaNs, no Infs; LLRs smoothly saturate or attenuate to zero. | PASS |
| **Extreme Amplitudes** | $|z| = 1000$ in `calibrated_16qam_llrs` | LogSumExp remains numerically stable without floating point overflow. | PASS |
| **Zero Input** | $x = 0$ into `rapp_pa` | Produces exact zero output without division by zero. | PASS |
| **PA Saturation Limit** | $|x| = 10^5$ into `rapp_pa(A=1.25)` | Output strictly bounded: $|y| \le 1.25$. | PASS |
| **Oversampling Factors** | $L \in \{1, 2, 4, 8\}$ in linear PA | Invertibility preserved with max difference $< 10^{-5}$. | PASS |
| **DD-PA Boundary Modes** | Iterations = 0, 1, 2, 3; linear $A=\infty$; soft decisions | Returns cleanly across all modes without tensor shape mismatches. | PASS |
| **Delay DFT Dimensions** | $N \in \{16, 32, 64, 128\}$ | Preserves exact input subcarrier dimensions. | PASS |

---

## 4. Caveats

1. **External Test Suite Isolation (E2E Track)**:
   - While `p4/tests/test_p4_0_simulator.py` is fully isolated, external tests (such as `tests/e2e/test_tier1_feature_coverage.py` in the Milestone 5 track) call `run_benchmarks()` without an explicit `output_dir` parameter, defaulting to `p4/results/`. Running the global `pytest` or `unittest discover tests` may overwrite `p4/results/`. When releasing production benchmarks, always run `python -m p4.benchmark_latency --warmup 25 --runs 100`.
2. **CPU Numerology**:
   - Latencies were measured on an Intel x86_64 host using PyTorch `2.12.1+cpu` on 4 threads. GPU latencies will be benchmarked when deployed on CUDA-enabled nodes in later stages.

---

## 5. Conclusion

**Verdict:** `APPROVE`

All objectives and acceptance criteria for Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines) are completely satisfied, rigorously verified, and free of defects.

---

## 6. Verification Method

To independently reproduce this verification:

1. **Verify 10 Consecutive Full Test Suite Runs**:
   ```powershell
   cd
   python -c "
   import subprocess, sys
   for i in range(1, 11):
       res = subprocess.run([sys.executable, '-m', 'unittest', 'p4/tests/test_p4_0_simulator.py', '-v'], capture_output=True, text=True)
       if res.returncode != 0 or 'Ran 14 tests' not in res.stderr or 'OK' not in res.stderr:
           print(f'Run {i} failed: {res.stderr}'); sys.exit(1)
       print(f'Run {i}/10 PASSED (OK)')
   print('ALL 10 CONSECUTIVE TEST RUNS PASSED 10/10 OK!')
   "
   ```

2. **Verify Static Parity Across Profiles ($< 2 \times 10^{-7}$)**:
   ```powershell
   python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"
   ```

3. **Verify Production Benchmark Records**:
   ```powershell
   python reviewer_m1_3/verify_benchmarks.py
   ```

4. **Verify Adversarial Stress Suite**:
   ```powershell
   python reviewer_m1_3/stress_test.py
   ```

5. **Verify Repository Cleanliness**:
   ```powershell
   git status --short
   ```
