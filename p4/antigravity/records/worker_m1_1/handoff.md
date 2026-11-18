# Stage P4-0 Handoff Report: Dynamic Channel Simulation & Nonlinear Classical Baselines

**Author:** Worker 1 (`worker_m1_1`)
**Role:** Implementer / QA / Specialist
**Working Directory:** ``
**State Directory:** `worker_m1_1`
**Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)
**Date:** 2026-11-15

---

## 1. Observation

### 1.1 Codebase & Historical State
- The historical codebase (Milestones P1, P2, P3, and exploratory runs) lacked:
  1. Time-evolving Doppler channels and Inter-Carrier Interference (ICI) simulation.
  2. AM-PM phase distortion and frequency oversampling for nonlinear power amplifiers.
  3. Bussgang-LMMSE equalization accounting for distortion variance $\sigma_D^2$.
  4. Calibrated 16-QAM soft LLR posterior log-odds calculations.
  5. Iterative Decision-Directed PA Cancellation (DD-PA).
- Cryptographic manifests (`MANIFEST_SHA256.json`) lock historical milestones. `git diff --name-only` confirmed zero modified tracked files, maintaining 100% historical immutability.

### 1.2 Delivered Artifacts in `p4/`
All deliverables were authored exclusively under the dedicated `p4/` directory:
- `p4/__init__.py`: Package initialization and exports.
- `p4/simulator.py`: Dynamic time-domain channel simulator with cyclic prefix, continuous Doppler phase evolution $h_l[n] = h_{l,0} e^{j 2\pi f_{d,l} n / N}$, FIR convolution, oversampled Rapp PA with AM-AM ($p=2, A \in \{1.25, 2.25\}$) and AM-PM phase distortion ($\Phi(r) = \frac{\alpha r^2}{1 + (r/\beta)^2}$), and static parity check.
- `p4/classical_baselines.py`: Bussgang decomposition via deterministic 1D quadrature, DFT-8 delay estimation, Bussgang-LMMSE equalizer, calibrated 16-QAM soft LLRs (exact logsumexp and piecewise maxlog), and 2-3 iteration DD-PA canceller.
- `p4/benchmark_latency.py`: Latency benchmarking harness for CPU (4 threads) and GPU across LS, DFT-8, Bussgang-LMMSE, and DD-PA (1, 2, 3 iters) for batch sizes 1 and 64.
- `p4/tests/__init__.py`: Test package init.
- `p4/tests/test_p4_0_simulator.py`: Unit test suite covering all features F1 through F7.
- `p4/results/latency_cpu.csv` and `p4/results/timing.json`: Benchmark outputs.

### 1.3 Measured Verification Results
1. **Static Parity Check ($f_d=0, A=\infty$)**:
   - `train3`: relative error $1.46 \times 10^{-7} < 10^{-5}$
   - `equal4`: relative error $1.45 \times 10^{-7} < 10^{-5}$
   - `long6`: relative error $1.54 \times 10^{-7} < 10^{-5}$
   - `long12`: relative error $1.63 \times 10^{-7} < 10^{-5}$
   - `sparse12`: relative error $1.49 \times 10^{-7} < 10^{-5}$
   - $N=64$: relative error $1.51 \times 10^{-7} < 10^{-5}$
2. **Doppler ICI**:
   - $f_d = 0$: transfer matrix off-diagonal power is $< 10^{-6}$.
   - $f_d = 0.02$: off-diagonal power is $1.2 \times 10^{-3} > 0$.
   - $f_d = 0.08$: off-diagonal power is $1.8 \times 10^{-2}$ (monotonic growth).
3. **Bussgang Decomposition**:
   - Quadrature vs 500,000-sample Monte Carlo:
     * $A=2.25, \alpha=0.5$: $\gamma = 0.9121 + 0.2770j$, $\sigma_D^2 = 0.0091$ (MC: $\gamma = 0.9122 + 0.2769j$, diff $< 0.001$).
     * $A=1.25, \alpha=0.5$: $\gamma = 0.7711 + 0.2257j$, $\sigma_D^2 = 0.0233$ (MC: $\gamma = 0.7712 + 0.2256j$, diff $< 0.001$).
4. **Bussgang-LMMSE vs Naive Linear LMMSE ($A=1.25$, SNR 18 dB)**:
   - Symbol MSE: Bussgang $0.11688 < 0.11794$ Naive Linear.
   - Calibrated LLR BCE: Bussgang $0.14516 < 0.16815$ Naive Linear (0.023 BCE reduction).
5. **16-QAM Soft LLR Calibration**:
   - Empirical bit probability vs predicted probability bias $< 0.001$ across all 4 bits.
   - Expected Calibration Error (ECE) $< 0.009 < 0.015$.
   - Sign of LLR matches hard slicing 100.0%.
6. **Iterative DD-PA Error Reduction ($A=1.25$, SNR 20 dB)**:
   - Iter 0 (Bussgang-LMMSE): BER = 0.0498, MSE = 0.2361
   - Iter 1 (DD-PA): BER = 0.0419, MSE = 0.2228 (15.8% BER reduction)
   - Iter 2 (DD-PA): BER = 0.0414, MSE = 0.2237
7. **Classical Latency Benchmark (CPU 4 threads, $N=32$)**:
   - Batch 1: LS = 0.072 ms, DFT-8 = 0.103 ms, Bussgang-LMMSE = 1.633 ms, DD-PA Iter 1 = 1.674 ms, Iter 2 = 2.176 ms, Iter 3 = 3.013 ms.
   - Batch 64: LS = 0.168 ms, DFT-8 = 0.169 ms, Bussgang-LMMSE = 2.306 ms, DD-PA Iter 1 = 6.150 ms, Iter 2 = 9.355 ms, Iter 3 = 14.907 ms.
8. **Unit Test Suite**:
   - Command: `python -m unittest p4/tests/test_p4_0_simulator.py -v`
   - Output: 14 tests run, 14 passed, exit code 0 (`OK`).

---

## 2. Logic Chain

1. **Static Parity Derivation**:
   - With cyclic prefix of length $N_{CP} \ge \max(d_l)$, the continuous-time linear FIR convolution of duration $N + N_{CP}$ maps to cyclic convolution over the payload window $n \in [N_{CP}, N_{CP} + N - 1]$.
   - Under $f_d = 0$, $h_l[n] = h_{l, 0}$ is constant in time.
   - Taking the orthonormal $N$-point FFT after CP removal diagonalizes the cyclic convolution: $Y[k] = H[k] X[k] + W[k]$.
   - Oversampling via frequency zero-padding with scale factor $\sqrt{L}$ preserves Nyquist-rate samples exactly ($x_{ov}[l \cdot L] = x_{nyq}[l]$).
   - Under linear PA ($A=\infty, \alpha=0$), decimation in frequency returns the identical spectrum up to complex64 floating-point precision ($1.4 \times 10^{-7} < 10^{-5}$), proving structural fidelity.

2. **Bussgang Decomposition and Channel Scaling**:
   - When pilot symbols pass through the transmitter PA, the observed channel from pilots is $Y_p / X_p = \gamma_p H + H D_p + W_p$.
   - Because $N=32$ subcarriers yield an approximately complex Gaussian time signal, $\gamma_p \approx \gamma_{data} = \gamma$.
   - Dividing the truncated delay estimate by $\gamma$ yields the unbiased physical channel estimate $\hat{H}_{phys} = \hat{H}_{LS} / \gamma$.
   - The effective channel seen by data is $H_{eff} = \gamma \hat{H}_{phys}$.
   - The MMSE filter $W = \frac{H_{eff}^*}{|H_{eff}|^2 + |\hat{H}_{phys}|^2 \sigma_D^2 + \sigma_W^2}$ maintains zero phase bias ($\arg(W H_{eff}) = 0$) and minimizes symbol MSE.
   - Normalizing by $\mu = W H_{eff}$ gives the conditionally unbiased observation $\hat{Z} = \hat{X} / \mu$ with calibrated disturbance variance $\sigma_{eff}^2 = (1 - \mu) / \mu$.

3. **Iterative Distortion Cancellation**:
   - In Iteration 0, tentative 16-QAM decisions $\hat{S}^{(0)}$ have ~5% BER under strong PA ($A=1.25$).
   - Transforming $\hat{S}^{(0)}$ to oversampled time domain, applying the exact Rapp PA equations, and subtracting the linear component yields the reconstructed distortion $\hat{d}_{ov} = f_{PA}(\hat{x}_{ov}) - \gamma \hat{x}_{ov}$.
   - Decimating to frequency and filtering through the estimated physical channel $\hat{H}_{phys}$ produces the received nonlinear intermodulation interference $\hat{Y}_{dist} = \hat{H}_{phys} \hat{D}$.
   - Subtraction $Y_{cleaned} = Y_d - \hat{Y}_{dist}$ suppresses in-band clipping distortion, reducing residual distortion variance $\sigma_{D, i}^2$.
   - Re-equalization achieves monotonic error reduction (BER from 4.98% to 4.14%), confirming the physical mechanism.

---

## 3. Caveats

1. **Hardware Environment**: The benchmarking machine is an Intel x86_64 CPU running PyTorch 2.12.1+cpu without CUDA hardware. Latencies reflect 4 CPU threads (`torch.set_num_threads(4)`). If executed on GPU, execution times will scale according to CUDA tensor core batch parallelism.
2. **Inter-Symbol Interference (ISI)**: Channel delay spreads up to $N_{CP} = 16$ are strictly supported without ISI. If delay spread exceeds $N_{CP}$, additional inter-symbol interference from preceding OFDM frames would occur.
3. **Doppler Channel Matrix Complexity**: For full slot simulation in Milestone 3, fast time-varying Doppler channels can be accelerated using band-diagonal transfer matrices.

---

## 4. Conclusion

All requirements for Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines, Features F1–F7) are completely implemented, validated, and passing all tests:
- F1 (Oversampled Rapp PA), F2 (Dynamic Multipath Simulator with Doppler), and F3 (Static Parity $< 10^{-5}$) are operational in `p4/simulator.py`.
- F4 (Bussgang-LMMSE), F5 (Calibrated 16-QAM Soft LLRs), and F6 (Iterative DD-PA Cancellation) are operational in `p4/classical_baselines.py`.
- F7 (Standardized Latency Benchmark) is operational in `p4/benchmark_latency.py`, with artifacts recorded in `p4/results/latency_cpu.csv` and `p4/results/timing.json`.
- Historical repository files are 100% untouched.

The codebase is fully primed for downstream Milestones (Stage P4-A: $C^*$-Equivariant EM-Native Transformer Architecture).

---

## 5. Verification Method

To independently verify this milestone from ``:

1. **Run Full Unit Test Suite**:
   ```powershell
   python -m unittest p4/tests/test_p4_0_simulator.py -v
   ```
   *Expected result*: 14 tests run, 14 pass, exit code 0 (`OK`).

2. **Verify Static Limit Parity Check Across All Profiles**:
   ```powershell
   python -c "import p4; [print(p, p4.static_limit_parity_check(profile=p)) for p in p4.PROFILES]"
   ```
   *Expected result*: Relative errors $< 2 \times 10^{-7}$ for all profiles (`train3`, `equal4`, `long6`, `long12`, `sparse12`).

3. **Run Latency Benchmark Harness**:
   ```powershell
   python -m p4.benchmark_latency --warmup 25 --runs 100
   ```
   *Expected result*: Completes with exit code 0 and updates `p4/results/latency_cpu.csv` and `p4/results/timing.json`.

4. **Verify Repository Governance Immutability**:
   ```powershell
   git status --short
   ```
   *Expected result*: Zero modified tracked files; all deliverables isolated inside `p4/`.
