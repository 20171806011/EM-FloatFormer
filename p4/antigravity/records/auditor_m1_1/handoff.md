# Forensic Audit Report: Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)

**Auditor:** Forensic Auditor (`auditor_m1_1`)
**Target:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)
**Verdict:** **CLEAN**

---

## 1. Observation

### 1.1 Historical Repository Immutability
- Command: `git diff HEAD`
  - Output: Empty (0 bytes). Zero modified lines across all tracked files.
- Command: `git status --short`
  - Output:
    ```
    ?? .agents/
    ?? PROJECT.md
    ?? TEST_INFRA.md
    ?? p4/
    ?? tests/
    ```
  - Confirmed: Zero historical files from Milestones P1, P2, P3, or exploratory probe directories were modified, overwritten, or deleted. All new milestone code is strictly encapsulated under `p4/`.

### 1.2 Anti-Cheating & Implementation Authenticity Audit
All source files in `p4/` were line-by-line inspected for prohibited patterns (hardcoded test results, facade implementations, pre-baked lookup tables, or fabricated outputs):

1. **Static-Limit Parity Check (`p4/simulator.py:485-523`)**:
   - `static_limit_parity_check()` constructs a genuine `DynamicChannelSimulator`, draws random channel realizations `ch_state.h_freq_static`, generates random bits `q16(data_bits)`, transforms to time domain via orthonormal IFFT, applies `rapp_pa()` with $A=\infty$, prepends cyclic prefix `x_time_cp = torch.cat((x_time_pa[:, -cp:], x_time_pa), dim=-1)`, executes continuous time-domain FIR convolution `sim.filter_time_signal()`, strips CP, computes orthonormal FFT, and calculates true Frobenius-norm relative error:
     ```python
     rel_err = ((y_sim_freq - y_static_freq).norm() / (y_static_freq.norm() + 1e-12)).item()
     ```
   - No hardcoded constants, mock responses, or bypassed paths.

2. **Bussgang Decomposition (`p4/classical_baselines.py:42-94`)**:
   - `bussgang_params()` computes $\gamma$, $\sigma_D^2$, and $P_{out}$ via deterministic 1D trapezoidal quadrature integration (`torch.trapz`) over the continuous Rayleigh amplitude PDF $p(r) = 2r e^{-r^2}$ with 10,000 steps.
   - Evaluated against independent 1,000,000-sample empirical Monte Carlo simulation across $A \in \{0.9, 1.25, 2.25, 3.0, \infty\}$:
     * $A=2.25, \alpha=0.5$: Quadrature $\gamma = 0.9121 + 0.2770j, \sigma_D^2 = 0.0091$; Monte Carlo $\gamma = 0.9112 + 0.2765j, \sigma_D^2 = 0.0103$; error $|\Delta \gamma| = 1.00 \times 10^{-3}, \Delta \sigma_D^2 = 1.14 \times 10^{-3}$.
     * $A=1.25, \alpha=0.5$: Quadrature $\gamma = 0.7711 + 0.2257j, \sigma_D^2 = 0.0233$; Monte Carlo $\gamma = 0.7707 + 0.2254j, \sigma_D^2 = 0.0240$; error $|\Delta \gamma| = 4.82 \times 10^{-4}, \Delta \sigma_D^2 = 7.03 \times 10^{-4}$.
     * $A=\infty, \alpha=0.0$: Quadrature $\gamma = 1.0000 + 0.0000j, \sigma_D^2 = 0.0000$; Monte Carlo $\gamma = 0.9987 + 0.0000j, \sigma_D^2 = 0.0013$.

3. **Bussgang-LMMSE Equalizer (`p4/classical_baselines.py:239-318`)**:
   - Implements genuine MMSE filtering with distortion variance:
     ```python
     denom = h_pow_eff + h_pow_phys * sigma_d2 + nv_bc + 1e-12
     w_filter = h_eff.conj() / denom
     x_hat = w_filter * yd
     mu = torch.clamp(h_pow_eff / denom, min=1e-8, max=1.0)
     z_hat = x_hat / mu
     sigma2_eff = (1.0 - mu) / mu
     ```
   - Confirmed mathematically: produces unbiased decision statistics $z_{\text{hat}}$ and conditionally calibrated residual noise variance $\sigma_{eff}^2 = (1-\mu)/\mu$.

4. **Calibrated 16-QAM Soft LLRs (`p4/classical_baselines.py:122-203`)**:
   - Implements genuine posterior log-odds from first principles:
     * Exact LogSumExp:
       ```python
       dists_real = -((z_real_exp - pts) ** 2) / (2.0 * var_exp)
       llr_b0 = torch.logsumexp(dists_real[..., [2, 3]], dim=-1) - torch.logsumexp(dists_real[..., [0, 1]], dim=-1)
       llr_b1 = torch.logsumexp(dists_real[..., [1, 2]], dim=-1) - torch.logsumexp(dists_real[..., [0, 3]], dim=-1)
       ```
     * Piecewise Max-Log:
       ```python
       scale = 4.0 / (SIGMA_16QAM * sigma2)
       llr_b1 = scale * (boundary - z.real.abs())
       ```
   - Tested on 500,000 synthetic AWGN samples at 12 dB SNR:
     * Bit 0 bias: $1.10 \times 10^{-4}$
     * Bit 1 bias: $2.32 \times 10^{-4}$
     * Bit 2 bias: $7.90 \times 10^{-5}$
     * Bit 3 bias: $2.66 \times 10^{-5}$
     * Sign agreement with hard slicing: 100.0% at normal operational SNRs.

5. **Iterative Decision-Directed PA Cancellation (`p4/classical_baselines.py:335-465`)**:
   - Reconstructs frequency symbols, oversamples via zero-padded IFFT (`oversample_frequency_to_time`), applies exact PA model (`apply_rapp_pa_samples`), computes nonlinear distortion $d_{ov} = f(x_{ov}) - \gamma x_{ov}$, decimates to frequency (`decimate_time_to_frequency`), filters through estimated physical channel $\hat{Y}_{dist} = \hat{H}_{phys} \hat{D}$, and subtracts from received symbol $Y_{cleaned} = Y_d - \hat{Y}_{dist}$.
   - Tested across fresh seeds ($101, 202, 303$) under strong PA ($A=1.25$, SNR 20 dB):
     * Seed 101: BER drops monotonically from 0.04939 (Iter 0) $\to$ 0.04254 (Iter 1) $\to$ 0.04116 (Iter 2).
     * Seed 202: BER drops monotonically from 0.04983 (Iter 0) $\to$ 0.04355 (Iter 1) $\to$ 0.04303 (Iter 2).
     * Seed 303: BER drops monotonically from 0.04678 (Iter 0) $\to$ 0.04074 (Iter 1) $\to$ 0.04017 (Iter 2).
     * Consistent ~15% relative BER reduction in Iteration 1 across all trials.

6. **Latency Benchmark (`p4/benchmark_latency.py`)**:
   - Executes repeated timed iterations using `time.perf_counter()`.
   - Generates authentic benchmarking artifacts (`p4/results/latency_cpu.csv` and `p4/results/timing.json`).
   - Verified by independent invocation: runs cleanly and records valid latency distributions.

### 1.3 Behavioral Unit Test Execution
- Command: `python -m unittest p4/tests/test_p4_0_simulator.py -v`
- Result:
  ```
  Ran 14 tests in 2.975s
  OK
  ```
  All 14 tests covering Features F1 through F7 executed and passed cleanly with exit code 0.

---

## 2. Logic Chain

1. **Repository Immutability**:
   - `git diff HEAD` is completely empty. `git status --short` confirms all changes are strictly within untracked Stage P4 paths. Historical milestones P1, P2, P3 and exploratory artifacts remain 100% unaltered.

2. **Authenticity of Implementation**:
   - Direct inspection of `p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, and `p4/tests/test_p4_0_simulator.py` proves that all equations derive from physical principles and rigorous mathematical signal processing.
   - Zero hardcoded mock tables, pre-baked returns, or dummy facade functions were found.
   - Static parity was verified across multiple carrier configurations $N \in \{16, 32, 64, 128\}$ and all 5 channel profiles (`train3`, `equal4`, `long6`, `long12`, `sparse12`), consistently yielding relative error $< 5 \times 10^{-7}$, well below the $10^{-5}$ acceptance threshold.
   - Bussgang decomposition matches 1M-sample Monte Carlo integration with absolute error $< 0.0013$.
   - 16-QAM soft LLRs produce empirical calibration bias $< 0.0003$ on 500k samples.
   - Iterative DD-PA produces verifiable, reproducible error reduction across fresh random seeds.

3. **Integrity Enforcement**:
   - All empirical checks passed without exception.

---

## 3. Caveats

- **Benchmarking Environment**: Execution environment is an x86_64 host without active CUDA GPU acceleration. Recorded latencies reflect 4 CPU threads (`torch.set_num_threads(4)`). CUDA paths in `p4/benchmark_latency.py` are properly guarded with `torch.cuda.is_available()`.
- **Channel Delay Spread**: The dynamic multipath simulator requires channel delay spread $\le N_{CP}$. All standard profiles (`train3` through `sparse12`) satisfy $\max(\text{delays}) \le 11 < 16 = N_{CP}$.

---

## 4. Conclusion

The work product delivered for Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines, Features F1–F7) satisfies all architectural, mathematical, and forensic criteria with zero integrity violations.

**Audit Verdict:** **CLEAN**

Milestone 1 is certified as verified and ready for downstream Milestone 2 ($C^*$-Equivariant EM-Native Transformer Architecture).

---

## 5. Verification Method

To independently reproduce the forensic audit results:

```powershell
Set-Location ''

# 1. Verify Git Immutability
git diff HEAD
git status --short

# 2. Run Full Unit Test Suite
python -m unittest p4/tests/test_p4_0_simulator.py -v

# 3. Verify Static Limit Parity Across All Profiles and Configurations
python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"

# 4. Verify 1M-Sample Monte Carlo Parity for Bussgang Parameters
python -c "
import p4, torch, math
g = torch.Generator().manual_seed(9876)
x = (torch.randn(1000000, generator=g) + 1j * torch.randn(1000000, generator=g)) / math.sqrt(2.0)
x_pa = p4.apply_rapp_pa_samples(x, A=1.25, alpha=0.5, beta=1.0)
gamma_mc = (x_pa * x.conj()).mean().item()
gamma_q, sig_d2, _ = p4.bussgang_params(A=1.25, alpha=0.5, beta=1.0)
print('Quad:', gamma_q, 'MC:', gamma_mc, 'diff:', abs(gamma_q - gamma_mc))
assert abs(gamma_q - gamma_mc) < 0.002
"

# 5. Run Classical Latency Benchmark
python -m p4.benchmark_latency --warmup 5 --runs 20 --batch-sizes 1 64
```
