# Review & Adversarial Challenge Report: Milestone 1 (Stage P4-0)

**Reviewer:** Reviewer 1 (`reviewer_m1_1`)
**Role:** Reviewer & Adversarial Critic
**Working Directory:** ``
**State Directory:** `reviewer_m1_1`
**Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)
**Date:** 2026-11-15

---

## Review Summary

**Verdict:** `REQUEST_CHANGES`

**Summary Assessment:**
The mathematical derivations, physical simulation mechanics, and core algorithms implemented in `p4/` (Oversampled Rapp PA, Dynamic Continuous Doppler Channel, Bussgang-LMMSE, Calibrated Soft LLRs, Iterative DD-PA, and Latency Benchmarking) are of exceptionally high theoretical fidelity and rigor. However, independent adversarial stress-testing revealed a critical defect in the unit test suite: `test_f5_llr_sign_matches_hard_slicing` in `p4/tests/test_p4_0_simulator.py` is flaky and fails intermittently (20% failure rate across 10 consecutive full suite runs) due to an uncalibrated assertion that ignores the mathematical boundary difference between exact Bayesian posterior LogSumExp and minimum-distance hard slicing. Additionally, running the test suite mutates and clobbers official benchmark results. These two issues must be resolved before Milestone 1 can be formally approved.

---

## 1. Observation

### 1.1 Direct Test Suite Execution & Flakiness
Executing `python -m unittest p4/tests/test_p4_0_simulator.py -v` (with working directory set to ``) produced intermittent test failures:

```text
======================================================================
FAIL: test_f5_llr_sign_matches_hard_slicing (p4.tests.test_p4_0_simulator.TestP40SimulatorAndBaselines.test_f5_llr_sign_matches_hard_slicing)
Verify that sign of LLRs perfectly agrees with hard bit slicing.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "p4\tests\test_p4_0_simulator.py", line 303, in test_f5_llr_sign_matches_hard_slicing
    self.assertTrue(torch.all(hard_from_llr == hard_from_slice))
AssertionError: tensor(False) is not true

----------------------------------------------------------------------
Ran 14 tests in 6.683s

FAILED (failures=1)
```

Across 10 consecutive programmatic test suite runs without fixing the seed:
- 8 runs passed (`OK`)
- 2 runs failed (`FAILED (failures=1)` on `test_f5_llr_sign_matches_hard_slicing`)
- Exact failure output: `Done 10 runs: 2 failures out of 10`

### 1.2 Mathematical Investigation of the Boundary Mismatch
Direct numerical analysis of the failing condition in `test_f5_llr_sign_matches_hard_slicing`:
In `p4/tests/test_p4_0_simulator.py` (lines 296–304):
```python
    def test_f5_llr_sign_matches_hard_slicing(self):
        """Verify that sign of LLRs perfectly agrees with hard bit slicing."""
        z = torch.randn(100, 32, dtype=torch.complex64)
        sigma2 = torch.full((100, 32), 0.1)
        llrs = calibrated_16qam_llrs(z, sigma2, method="exact")
        hard_from_llr = (llrs > 0).long()
        hard_from_slice = hard_slice_16qam(z)
        self.assertTrue(torch.all(hard_from_llr == hard_from_slice))
```
- In `hard_slice_16qam(z)`, bit 1 (inner/outer) threshold is hardcoded to $2.0 / \sqrt{10} \approx 0.6324555$.
- In `calibrated_16qam_llrs(z, sigma2, method="exact")`, the 4 PAM points are $pts = [-3, -1, 1, 3] / \sqrt{10}$.
- For a sample $x = 2.0 / \sqrt{10} + \delta$:
  * At $\delta = 0.0$: $x = 0.632456$, $llr = +0.000335$, `sign(llr) = 1`, but `hard_slice = 0`.
  * At $\delta = 0.000010$: $x = 0.632466$, $llr = +0.000209$, `sign(llr) = 1`, but `hard_slice = 0`.
  * At $\delta = 0.000020$: $x = 0.632476$, $llr = +0.000082$, `sign(llr) = 1`, but `hard_slice = 0`.
  * Zero-crossing of exact LLR occurs at $\delta \approx 0.000026$ ($x \approx 0.632482$).
- For any continuous random variable $z \sim \mathcal{CN}(0, I)$, with $3200$ complex values ($6400$ real dimensions), the probability that at least one coordinate falls in this $2.6 \times 10^{-5}$ boundary strip is $\approx 15\% - 20\%$.

### 1.3 Benchmark Artifact Clobbering
In `p4/tests/test_p4_0_simulator.py` (lines 368–376):
```python
        test_dir = "p4/results"
        records, meta = run_benchmarks(
            batch_sizes=(1, 64),
            num_threads=4,
            device="cpu",
            output_dir=test_dir,
            warmup_runs=5,
            timed_runs=10,
        )
```
- Running the unit test suite automatically overwrites the production benchmark files (`p4/results/latency_cpu.csv` and `p4/results/timing.json`) with low-sample test results (warmup=5, runs=10), erasing the 100-run measurements produced by `python -m p4.benchmark_latency --warmup 25 --runs 100`.

### 1.4 Hardcoded Default in `delay_est_dft`
In `p4/classical_baselines.py` (lines 104–116):
```python
def delay_est_dft(
    yp: torch.Tensor,
    taps: int = 8,
    n: int = DEFAULT_N,
) -> torch.Tensor:
    h_delay = torch.fft.ifft(yp, dim=-1)
    h_trunc = F.pad(h_delay[..., :taps], (0, n - taps))
    return torch.fft.fft(h_trunc, dim=-1)
```
- If called with an $N=64$ tensor without explicitly specifying `n=64`, `F.pad` pads to $32$ instead of $64$, truncating the tensor to shape `(B, 32)`.

### 1.5 Verified Passing Features
All other core requirements were independently verified and passed:
1. **Static Parity Check ($f_d=0, A=\infty$)**:
   - `train3`: $1.47 \times 10^{-7} < 10^{-5}$
   - `equal4`: $1.43 \times 10^{-7} < 10^{-5}$
   - `long6`: $1.52 \times 10^{-7} < 10^{-5}$
   - `long12`: $1.66 \times 10^{-7} < 10^{-5}$
   - `sparse12`: $1.48 \times 10^{-7} < 10^{-5}$
   - $N=64$: $1.51 \times 10^{-7} < 10^{-5}$
2. **Doppler ICI & Continuous FIR Filtering**:
   - Off-diagonal ICI power grows monotonically ($f_d=0: <10^{-6}$, $f_d=0.02: 1.2 \times 10^{-3}$, $f_d=0.08: 1.8 \times 10^{-2}$).
   - Time-domain FIR matches analytical transfer matrix output to $< 10^{-5}$.
3. **Bussgang Decomposition & LMMSE**:
   - 1D numerical quadrature over Rayleigh PDF matches 500k-sample Monte Carlo ($A=2.25$: $\gamma = 0.9121 + 0.2770j$, diff $< 0.001$; $A=1.25$: $\gamma = 0.7711 + 0.2257j$, diff $< 0.001$).
   - Bussgang-LMMSE achieves strictly lower MSE (0.11688 < 0.11794) and lower LLR BCE (0.14516 < 0.16815) than linear LMMSE under $A=1.25$.
4. **Iterative DD-PA Error Reduction**:
   - Monotonic BER reduction: Iter 0 = 0.0498 $\to$ Iter 1 = 0.0419 (15.8% reduction) $\to$ Iter 2 = 0.0414.
5. **Historical Immutability**:
   - `git status --short` confirms 0 modified tracked files. Historical milestones P1-P3, exploratory files, and `MANIFEST_SHA256.json` are 100% untouched.

---

## 2. Logic Chain

1. **Root Cause of Unit Test Flakiness**:
   - For Gray-coded 16-QAM, PAM constellation points are located at $[-3, -1, +1, +3] / \sqrt{10}$.
   - Minimum-distance hard slicing compares $|x|$ against the midpoint between inner and outer points ($2.0 / \sqrt{10}$).
   - The exact Bayesian posterior LLR $LLR(b_1) = \ln \frac{e^{-d_1} + e^{-d_2}}{e^{-d_0} + e^{-d_3}}$ incorporates the tails of all 4 points.
   - Because $pts_1$ (distance 3) is closer to $x=2$ than $pts_0$ (distance 5), $e^{-d_1} > e^{-d_0}$, so the numerator exceeds the denominator at the Voronoi boundary $x = 2/\sqrt{10}$.
   - Consequently, the exact zero-crossing is shifted outward to $x^* \approx 0.632482$.
   - Asserting `torch.all(hard_from_llr == hard_from_slice)` on continuous unseeded Gaussian noise tests a mathematical falsehood for `method="exact"`.
   - Conversely, for `method="maxlog"`, the piece-wise linear approximation strictly uses the Voronoi boundary ($2.0 / \sqrt{10}$), resulting in 100.0% agreement across 100% of random seeds.

2. **Impact on Project Milestones**:
   - A flaky test in the core test suite will trigger intermittent failures during CI, downstream milestone handoffs (e.g. M2, M3), and M5 E2E regression testing.
   - A unit test should never mutate production benchmark artifacts in `p4/results/`.
   - These are concrete quality issues with trivial fixes that must be resolved before proceeding.

---

## 3. Caveats

1. **Non-Malicious Origin**: This is not an intentional facade or integrity violation; the algorithmic implementation is genuine, mathematically sound, and performs exceptionally well. The failure is an oversight in unit test design (unseeded sampling across an asymmetric Bayesian decision boundary).
2. **Hardware Constraints**: Latencies were measured on 4 CPU threads (Intel x86_64, PyTorch 2.12.1+cpu). GPU latencies will scale with tensor parallelism when deployed on CUDA hardware.

---

## 4. Conclusion & Actionable Remedies

**Verdict:** `REQUEST_CHANGES`

Worker 1 must apply the following specific changes:

### Required Change 1: Fix Flaky Unit Test in `p4/tests/test_p4_0_simulator.py`
In `test_f5_llr_sign_matches_hard_slicing` (lines 296–304):
- Either test `method="maxlog"`, which is mathematically guaranteed to match `hard_slice_16qam` 100% across all seeds:
  ```python
  def test_f5_llr_sign_matches_hard_slicing(self):
      """Verify that sign of maxlog LLRs perfectly agrees with hard bit slicing."""
      z = torch.randn(100, 32, dtype=torch.complex64, generator=self.generator)
      sigma2 = torch.full((100, 32), 0.1)
      llrs = calibrated_16qam_llrs(z, sigma2, method="maxlog")
      hard_from_llr = (llrs > 0).long()
      hard_from_slice = hard_slice_16qam(z)
      self.assertTrue(torch.all(hard_from_llr == hard_from_slice))

      # And verify that exact LLRs agree on >99.9% of samples due to boundary shift
      llrs_exact = calibrated_16qam_llrs(z, sigma2, method="exact")
      agreement = ((llrs_exact > 0).long() == hard_from_slice).float().mean().item()
      self.assertGreater(agreement, 0.999)
  ```

### Required Change 2: Isolate Benchmark Artifact Generation in Unit Tests
In `test_f7_benchmark_artifacts_generation` (lines 364–395 of `p4/tests/test_p4_0_simulator.py`):
- Change `test_dir` to a temporary directory (e.g. using `import tempfile; tempfile.mkdtemp()`) so that running `unittest` does not overwrite the official production benchmark files in `p4/results/`.
- Re-run `python -m p4.benchmark_latency --warmup 25 --runs 100` to ensure `p4/results/latency_cpu.csv` and `p4/results/timing.json` contain the full 100-run production data.

### Required Change 3 (Recommended): Robust $N$ inference in `delay_est_dft`
In `p4/classical_baselines.py` (line 104):
- Update `delay_est_dft(yp: torch.Tensor, taps: int = 8, n: Optional[int] = None)` to automatically default to `n = yp.shape[-1] if n is None else n`.

---

## 5. Verification Method

Once Worker 1 completes the modifications, independent verification will consist of:

1. **Verify 100% Clean Pass of Full Test Suite Across Multiple Seeds**:
   ```powershell
   cd
   python -m unittest p4/tests/test_p4_0_simulator.py -v
   ```
   *Expectation*: 14 tests run, 14 pass, exit code 0 (`OK`), with zero intermittent flakiness.

2. **Verify Static Parity Check**:
   ```powershell
   python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"
   ```
   *Expectation*: Relative error $< 2 \times 10^{-7}$ across all profiles.

3. **Verify Benchmark Artifact Integrity**:
   Verify that `p4/results/latency_cpu.csv` retains the full 100-run measurements and is not modified by running unit tests.
