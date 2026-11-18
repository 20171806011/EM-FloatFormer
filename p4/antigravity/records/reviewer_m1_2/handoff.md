# Reviewer 2 Quality & Adversarial Review Report: Milestone 1 (Stage P4-0)

**Reviewer:** Reviewer 2 (`reviewer_m1_2`)
**Role:** Reviewer & Adversarial Critic
**Working Directory:** ``
**State Directory:** `reviewer_m1_2`
**Target Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)
**Date:** 2026-11-15

---

## Review Summary

**Verdict**: **APPROVE**
**Integrity Audit**: PASS (Zero hardcoded bypasses, zero facade logic, zero fabricated outputs, 100% genuine independent verification).
**Adversarial Challenge Risk Assessment**: LOW (System demonstrates full mathematical convergence, 24-octave scale invariance, and rigorous boundary-condition stability).

---

## 1. Observation

### 1.1 Implementation & Code Layout Inspection
An exhaustive audit of the files in `p4/` was conducted:
1. `p4/simulator.py`:
   - `rapp_am_am` (lines 89-101): Rapp AM-AM amplitude transfer function $g(r) = r / (1 + (r/A)^{2p})^{1/(2p)}$ with $p=2.0$. Exact linear identity when $A \in \{\text{None}, \infty\}$.
   - `rapp_am_pm` (lines 103-112): Saleh-extended AM-PM phase rotation $\Phi(r) = \alpha r^2 / (1 + (r/\beta)^2)$. Strictly evaluates to 0 when $\alpha = 0$.
   - `rapp_pa` (lines 183-210): Implements oversampled PA processing by frequency zero-padding with scale factor $\sqrt{L}$, transforming via IFFT, applying point-wise nonlinear distortion via `torch.polar(g_r, theta)`, transforming back via FFT, extracting central $N$ bins with decimation factor $1/\sqrt{L}$, and returning to Nyquist time domain.
   - `DynamicChannelSimulator` (lines 225-379): Continuous time-domain FIR convolution with sample-level Doppler phase drift $h_l[n] = h_{l, 0} \exp(j 2\pi f_{d, l} n / N)$.
   - `static_limit_parity_check` (lines 485-523): Frobenius-norm relative error comparison against static $Y = H X$.
2. `p4/classical_baselines.py`:
   - `bussgang_params` (lines 41-93): Deterministic 1D numerical quadrature using `torch.trapz` over $10^4$ steps of the Rayleigh amplitude PDF $p(r) = 2 r e^{-r^2}$, calculating attenuation factor $\gamma$ and distortion variance $\sigma_D^2 = P_{out} - |\gamma|^2$.
   - `calibrated_16qam_llrs` (lines 122-202): Exact posterior log-likelihood ratios via numerically stable `torch.logsumexp` as well as piecewise max-log linear approximation, taking conditionally unbiased symbols $Z = \hat{X}/\mu$ and effective variance $\sigma_{eff}^2 = (1 - \mu)/\mu$.
   - `bussgang_lmmse` (lines 239-317): Bussgang-LMMSE equalization accounting for distortion variance $\sigma_D^2$ and physical channel de-embedding $H_{phys} = H_{raw} / \gamma$. Conditional mean factor $\mu = |H_{eff}|^2 / \text{denom}$ clamped to $[10^{-8}, 1.0]$.
   - `iterative_dd_pa_cancellation` (lines 335-465): 2-3 iteration decision-directed time-domain distortion estimation, oversampled filtering, decimation, subtraction from received data subcarriers $Y_{cleaned} = Y_d - H_{phys} \hat{D}$, and variance decay.
3. `p4/benchmark_latency.py`:
   - Standardized harness testing batch sizes (1, 64) with 4 CPU threads across LS, DFT-8, Bussgang-LMMSE, and DD-PA (1, 2, 3 iters).
4. `p4/tests/test_p4_0_simulator.py`:
   - 14 comprehensive unit tests verifying features F1 through F7.

### 1.2 Independent Reproduction & Test Suite Execution
- **Unit Test Execution**:
  Command: `python -m unittest p4/tests/test_p4_0_simulator.py -v`
  Result: 14 tests run in 7.100s, 14 passed, exit code 0 (`OK`).
- **Benchmark Latency Execution**:
  Command: `python -m p4.benchmark_latency --warmup 5 --runs 10`
  Result: Completed with exit code 0. Generated `p4/results/latency_cpu.csv` and `p4/results/timing.json`.
- **Static Limit Parity Reproduction**:
  Command: `python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"`
  - `train3`: $1.46 \times 10^{-7} < 10^{-5}$
  - `equal4`: $1.48 \times 10^{-7} < 10^{-5}$
  - `long6`: $1.48 \times 10^{-7} < 10^{-5}$
  - `long12`: $1.65 \times 10^{-7} < 10^{-5}$
  - `sparse12`: $1.45 \times 10^{-7} < 10^{-5}$
  All profiles achieve $< 2 \times 10^{-7}$ relative error, exceeding the contractual threshold by nearly two orders of magnitude.

### 1.3 Repository Governance & Historical Immutability
- `git diff HEAD`: 0 lines modified.
- `git status --short`:
  ```text
  ?? .agents/
  ?? PROJECT.md
  ?? TEST_INFRA.md
  ?? p4/
  ?? tests/
  ```
  Zero modified tracked files. Historical milestones (P1-P3, exploratory) remain 100% immutable and untouched.

---

## 2. Logic Chain

1. **Integrity & Authenticity Audit**:
   - Examination of `p4/classical_baselines.py` and `p4/simulator.py` confirms that all physical constants, quadrature calculations, and LLR equations are derived from first principles.
   - There are no hardcoded lookup tables substituting for real calculation. Quadrature integration numerically integrates the Rayleigh amplitude PDF over $10^4$ discrete steps and is validated against 500k-sample Monte Carlo simulations.
   - The test assertions test true statistical behavior: LLR calibration bias on synthetic AWGN ($< 0.01$), Expected Calibration Error ($< 0.015$), and monotonic BER reduction in DD-PA.
   - Conclusion: No integrity violations detected.

2. **Mathematical Correctness of Bussgang-LMMSE & Soft LLR**:
   - By Bussgang's theorem, $f_{PA}(x) = \gamma x + d$ with $\mathbb{E}[x^* d] = 0$.
   - In OFDM with $N \ge 32$, time-domain transmit samples approximate a circularly symmetric complex Gaussian process $x \sim \mathcal{CN}(0, 1)$.
   - The pilot observation is $Y_p = H (\gamma X_p + D_p) + W_p$. For QPSK pilots with $|X_p|^2 = 1$, $Y_p / X_p = \gamma H + H D_p X_p^* + W_p X_p^* \approx \gamma H = H_{eff}$.
   - Thus, dividing by $\gamma$ yields the unbiased physical channel $H_{phys} = H_{eff} / \gamma$.
   - For data transmission, $Y_d = H_{eff} s_d + H_{phys} d + W$. The total disturbance variance is $\text{denom} = |H_{eff}|^2 + |H_{phys}|^2 \sigma_D^2 + \sigma_W^2$.
   - The MMSE filter $W = H_{eff}^* / \text{denom}$ yields conditional expectation $\mathbb{E}[\hat{X} | s_d] = \mu s_d$, where $\mu = |H_{eff}|^2 / \text{denom} \in (0, 1]$.
   - Scaling by $1/\mu$ produces the conditionally unbiased decision variable $Z = \hat{X} / \mu = s_d + v$, with exact residual variance $\sigma_{eff}^2 = (1 - \mu)/\mu$.
   - The LLR calculation for Gray-coded 16-QAM correctly partitions the 4-PAM constellations along real and imaginary axes, with distance weighting $- (z_{dim} - s_m)^2 / (2 \cdot \frac{\sigma_{eff}^2}{2}) = - (z_{dim} - s_m)^2 / \sigma_{eff}^2$.
   - In max-log mode, the piecewise linear segments $\text{scale} \cdot (\text{boundary} - |z|)$ and $\text{scale} \cdot z$ exactly match the high-SNR asymptotic log-odds.

3. **Adversarial Boundary & Stress-Testing**:
   An independent adversarial stress-test suite (`adversarial_stress_test.py`) was executed covering 8 critical failure modes:
   - *Zero Noise ($\sigma_W^2 = 0$)*: Matrix inversion denominator contains `1e-12` jitter, preventing singularity. Zero NaNs or Infs generated.
   - *Extreme Low SNR ($\text{SNR} = -60\text{ dB}, \sigma_W^2 = 10^6$)*: Evaluates without numerical overflow; with true channel knowledge, $\mu \to 0$ and mean $|LLR| \to 6.5 \times 10^{-4} \approx 0$ (uninformative log-odds).
   - *Deep Channel Fade ($H \to 0$)*: Division by zero is completely averted; LLRs remain stable.
   - *Extreme PA Saturation Limits*: $A \to \infty$ returns exact identity $\gamma=1, \sigma_D^2=0$; $A = 0.05$ (extreme clipping) converges stably ($\gamma = 0.0426 + 0.0112j, \sigma_D^2 = 0.0005$).
   - *Extreme Doppler ($f_d = 0.5$)*: Continuous Doppler phase evolution remains bounded and stable.
   - *Extreme Constellation Coordinates ($|z| = 1000$)*: `logsumexp` avoids numerical underflow/overflow (exact LLR bounded in $[-25290, +25290]$ without NaN).
   - *Decision-Directed Soft Slicing*: `soft_slice_16qam` provides a closed-form posterior expectation $\mathbb{E}[S | LLR] \in \mathbb{C}$ that integrates seamlessly with DD-PA iterations.
   - *Scaling & Dynamic Range Invariance*: Scaling $Y$, $Y_p$, and $\sqrt{nv}$ by $r \in [2^{-12}, 2^{+12}]$ yields **0 bit flips** across the entire 24-octave range and maximum relative LLR difference $< 4.4 \times 10^{-4}$ under FP32/torch.complex64.
   - *Global Phase $U(1)$ Invariance*: Rotating inputs by arbitrary phase angle $e^{j\theta}$ yields **0 bit flips** and maximum LLR difference $< 1.3 \times 10^{-4}$.

---

## 3. Caveats

1. **CPU Benchmark Context**: The benchmarking results were recorded on an Intel x86_64 CPU running PyTorch 2.12.1+cpu (4 threads). Latency numbers reflect CPU vectorization. When deployed on GPU tensor cores, parallel throughput will scale significantly for large batch sizes ($B \ge 64$).
2. **Channel Taps within Cyclic Prefix**: Current FIR simulation requires maximum channel delay $\max(d_l) \le N_{CP} = 16$. This is fully satisfied by all standard profiles (`train3` max delay 2, `long12` max delay 11, `sparse12` max delay 11). If longer channel delays are tested in Stage P4-B/C, inter-symbol interference from previous OFDM frames will need to be modeled.

---

## 4. Conclusion

- No integrity violations or hardcoded facades were found.
- Mathematical formulations for Bussgang decomposition, Bussgang-LMMSE, and calibrated 16-QAM soft LLRs are correct and verified.
- Numerical stability, 24-octave scale invariance, and boundary conditions passed all tests.
- Historical milestone files remain 100% untouched.

**Explicit Verdict: APPROVE**. The codebase is verified ready for Milestone 2 (Stage P4-A: $C^*$-Equivariant EM-Native Transformer Architecture).

---

## 5. Verification Method

To independently reproduce and verify this review from ``:

1. **Execute Unit Test Suite**:
   ```powershell
   python -m unittest p4/tests/test_p4_0_simulator.py -v
   ```
   *Expected outcome*: 14 tests run, 14 pass, exit code 0 (`OK`).

2. **Verify Static Limit Parity Across Profiles**:
   ```powershell
   python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"
   ```
   *Expected outcome*: Relative errors $< 2 \times 10^{-7}$ for all profiles.

3. **Execute Latency Benchmark Harness**:
   ```powershell
   python -m p4.benchmark_latency --warmup 5 --runs 10
   ```
   *Expected outcome*: Completes with exit code 0 and verifies `p4/results/latency_cpu.csv` and `p4/results/timing.json`.

4. **Run Adversarial Stress-Test Suite**:
   ```powershell
   python reviewer_m1_2/adversarial_stress_test.py
   ```
   *Expected outcome*: 8/8 adversarial stress tests pass cleanly with zero failures.

5. **Verify Repository Immutability**:
   ```powershell
   git status --short
   git diff HEAD
   ```
   *Expected outcome*: `git diff HEAD` returns empty; no historical files are modified.
