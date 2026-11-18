# Stage P4-0 Survey Report: Historical Baselines, Channel Simulators, and Classical Nonlinear Receivers

**Author:** Explorer 1 (Historical Baseline & Simulator Specialist, `teamwork_explorer_survey_1`)
**Date:** 2026-11-15
**Working Directory:** ``
**State Directory:** `teamwork_explorer_survey_1`
**Target Milestone:** Stage P4-0 (Requirement R1: Dynamic Channel Simulation & Nonlinear Classical Baselines)

---

## Executive Summary

This survey comprehensively investigates the `EM-FloatFormer` codebase, historical milestones (Stages P1, P2, P3, and prior exploratory probes), execution environment, channel simulator implementations, and classical receiver baselines to establish the exact technical baseline and gaps for **Stage P4-0 (Requirement R1)**.

### Core Survey Findings
2. **Historical Milestone Governance**: P1 (P1-A to P1-E), P2 (P2-A to P2-E + supplement), and P3 (P3-0/A, P3-A2, P3-B, P3-C, P3-DE surrogate) are frozen and audited via dedicated SHA256 manifests (`MANIFEST_SHA256.json`, `P3_A2_PUBLIC_SHA256_MANIFEST.json`, `P2_SUPPLEMENT_MANIFEST_SHA256.json`, etc.). To strictly preserve historical immutability, **all Stage P4 deliverables must reside in a dedicated phase directory (`p4/`)** without modifying any P1–P3 or exploratory files.
3. **Channel Simulator & PA Gaps**:
   - *Existing Simulator*: `p3b_time_link.py` and `exploratory/16qam_joint_pa_probe/probe.py` implement static CP insertion/removal ($N=32, CP=16$) and static causal FIR convolution.
   - *Missing Doppler/ICI*: Time-varying channels and Doppler-induced Inter-Carrier Interference (ICI) are **completely absent** across all existing code.
   - *Missing AM-PM & Oversampling*: Existing Rapp models in `p3b_time_link.py` and `probe.py` implement only memoryless AM-AM magnitude compression at the Nyquist sampling rate ($N=32$). AM-PM phase distortion is non-existent ($\Phi \equiv 0$), and critically sampled nonlinear clipping introduces severe artificial in-band aliasing.
4. **Classical Baseline Gaps**:
   - *Bussgang-LMMSE*: **Completely missing**. Existing receivers use standard linear LS, DFT tap truncation (DFT-8, DFT-12), hard/soft thresholding, or linear Gaussian LMMSE without Bussgang decomposition.
   - *16-QAM Soft LLRs*: **Completely missing**. Existing code uses hard quadrant slicers (`hard_bits`) or heuristic uncalibrated linear logits (`TinyJoint`). No mathematically calibrated soft LLR computation exists.
   - *Iterative DD-PA Cancellation*: **Completely missing**. Only a 1-pass decision-directed channel estimation pseudo-pilot for QPSK was present in P3-B (`dd_threshold`). No decision-directed PA distortion reconstruction and subtraction exists.
5. **Stage P4-0 Action Plan**: R1 requires delivering a unified time-domain dynamic simulator (with Doppler, ICI, CP, and oversampled Rapp AM-AM / AM-PM) and strong nonprivileged classical baselines (Bussgang-LMMSE, calibrated 16-QAM soft LLRs, and 2–3 iteration DD-PA cancellation) inside `p4/`.

---

## 1. Repository Structure, Environment, and Dependencies

### 1.1 Directory Tree Overview
The repository root contains 81 files plus several result and exploratory subdirectories:
- `ofdm_baseline.py`, `p1b_*.py`, `p1c_*.py`, `p1d_*.py`, `p1e_*.py`: P1 exploratory and confirmatory stages.
- `p2_*.py`, `test_p2_*.py`: P2 residual learning, U(1) equivariance, and supplement.
- `p3_*.py`, `p3b_*.py`, `p3c_*.py`, `test_p3*.py`: P3 delay-domain, time-link, and precision audits.
- `exploratory/16qam_joint_pa_probe/`: Initial 16-QAM and Rapp PA probe (`probe.py`, `timing.py`, `audit.py`).
- `exploratory/p3de_tdl_surrogate/`: Standalone 3GPP TDL profile table approximation.
- `p1*_results/`, `p2*_results/`, `p3*_results/`: Stored CSVs, JSON audits, and PNG figures from past stages.
- `MANIFEST_SHA256.json`: Root cryptographic manifest tracking P1/P2 source and reports.

### 1.2 Python & PyTorch Runtime
A live probe of the execution environment confirms:
- **Python**: `3.12.7 | packaged by Anaconda, Inc. | (main, Oct 4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]`
- **PyTorch**: `2.12.1+cpu`
- **CUDA Availability**: `False` (Device count: 0)
- **Host CPU Threads**: 10 threads
- **Benchmarking Convention**: Previous stages (`p1e`, `p2_run`, `p3b_timing_extended`, `timing.py`) enforce `torch.set_num_threads(4)` for reproducible CPU wall-clock latency reporting.

### 1.3 Dependency Manifests
- `requirements.txt`: `torch>=2.1,<3`, `matplotlib>=3.8,<4`
- `requirements_p3c.txt`: `torch>=2.10,<2.11`, `numpy>=2.3`, `matplotlib>=3.9`
- `requirements_p3de_surrogate.txt`: Standard PyTorch/NumPy dependencies.
All necessary dependencies (`torch`, `numpy`, `matplotlib`) are installed and functional. External packages like NVIDIA Sionna are not installed (and cannot be installed due to isolated network access). Stage P4 must remain completely standalone and self-contained within PyTorch.

### 1.4 Windows Platform Nuances (CRLF vs LF)
When running unit tests like `test_p2_integrity.py` and `test_p2_supplement_independent.py`, strict byte-level SHA256 checks on markdown protocol files (e.g., `P2_PROTOCOL.md`) failed because git on Windows checked out files with `\r\n` line terminators rather than Linux `\n`. For Stage P4, test scripts checking protocol immutability must either normalize line endings (`p.read_bytes().replace(b'\r\n', b'\n')`) or use explicit binary-mode verification with awareness of CRLF.

---

## 2. Historical Milestones and Governance

### 2.1 Stage P1: Foundations and Baselines (P1-A to P1-E)
- **Signal Model**: 32 subcarriers ($N=32$), all-subcarrier known pilots ($X_p[k] = 1$), one QPSK data symbol ($X_d[k] \in \{\pm 1 \pm j\}/\sqrt{2}$), static channel per symbol pair.
- **Key Findings**:
  - `ofdm_baseline.py` established pilot-only Least Squares (LS), DFT truncation (`taps=3`), and LMMSE with Gaussian prior covariance.
  - Circular CNNs with log-amplitude/phase representations outperformed small Transformers on held-out 4-tap channels (`equal4`, 6.57% vs 7.43% BER at 10 dB), but failed to outperform LS on 6-tap channels (`long6`, 8.56% vs 8.22% BER).
  - P1-E 5-seed confirmatory replication yielded exact Holm-adjusted $p=0.125$ (not statistically significant at $\alpha = 0.05$).

### 2.2 Stage P2: Physics-Guided Residual Learning and U(1) Equivariance (P2-A to P2-E)
- **Architecture Evolution**:
  - `PointResidual`: Pointwise LS residual (negative control).
  - `LocalResidual`: Circular frequency-domain CNN with LS residual skip connection.
  - `DelayUnfold`: Trainable real Wiener-like shrinkage in delay domain.
  - `EquivComplexLocal`: Complex-linear circular FIR with real invariant energy-gating (`features = [log1p(|yp|^2), log10(nv)]`), achieving exact $U(1)$ common-phase equivariance:
    $$m(e^{j\phi} Y_p, e^{j\phi} Y_d) = e^{j\phi} m(Y_p, Y_d)$$
- **Results**: On unseen channels at 10 dB, `res_equiv_local` (2,060 params) achieved 6.35% BER on `equal4` and 7.37% on `long6`, significantly better than P1 circular CNN (6.57% / 8.32%), but still inferior to fixed DFT-8 (5.39% / 5.34%) and privileged matched LMMSE (4.90% / 5.08%).

### 2.3 Stage P3: Causal Mechanisms, Impairments, and Precision Limits (P3-0 to P3-C)
- **P3-0 / P3-A & P3-A2**:
  - Explored 985-parameter `EquivDelayConv` (`p3_models.py`, `p3_a_mechanism.py`).
  - Factorial study of positional coordinate inputs vs 10% retention floor (`FactorialDelayConv`, Model D). Discovered that removing absolute delay coordinates and retaining a 10% floor dropped `sparse12` BER from 12.29% to 6.55%, but classical DFT-threshold achieved 4.92%.
  - *Codebase Artifact Note*: `p3_a.py` is archived inside `P3-0_A_source_and_results.zip`. Several tests (`test_p3b_link.py`, `test_p3c_precision.py`) attempt `from p3_a import dft_threshold`, which errors if `p3_a.py` is not unpacked. Stage P4 must not depend on `p3_a.py` or unpacked zip archives.
- **P3-B (Time-Link & Impairments)**:
  - First introduction of time-domain processing with cyclic prefix (`tx_cp`, $N=32, CP=16$) and causal FIR filtering (`p3b_time_link.py`).
  - Modeled 4 synthetic impairment axes: PA (mild/strong), impulse noise, CPE (phase drift), and ADC quantization.
  - Receiver `DataResidualHead` consumed both data and pilot observations, but neural models failed to beat the classical envelope across 135 scenario cells.
- **P3-C (Precision & Coherent Scale Generalization Audit)**:
  - Re-evaluated frozen P3-B weights under FP16/BF16 CPU autocast and IQ storage quantization.
  - **Critical Discovery**: Evaluated coherent gain scaling by $2^{\pm 12}$ (`GAIN_EXPONENTS = (-12, 0, 12)`). All prior neural architectures suffered catastrophic scale-generalization collapse even in FP32 because their input encodings and nonlinear activations lacked scale equivariance/invariance.
  - This directly motivates Requirement R2: strict $C^* = \mathbb{R}_{>0} \times U(1)$ gauge invariance!

### 2.4 Exploratory 16-QAM PA Probe (`exploratory/16qam_joint_pa_probe/`)
- First trial of Gray-coded 16-QAM in the repository (`q16` in `probe.py`).
- Implemented memoryless PA magnitude compression ($A=2.25$ mild, $A=1.25$ strong).
- Evaluated a 1,812-parameter `TinyJoint` CNN predicting 4 uncalibrated logits.
- **Documented Negative/Limited Result**: Neural demapper improved over train-calibrated DFT-8 slicer by only 0.19 percentage points (18.61% vs 18.80% at 10 dB equal4/strong PA). The report explicitly stated:
  > *"Full nonlinear-PA-aware iterative correction / decoding and a fully data-aided probabilistic 16-QAM receiver were not implemented. Therefore the suggested definitive 'beat DFT-Thresh plus classical clipping repair, otherwise end the project' test has NOT been completed."*

### 2.5 Historical Immutability and Governance Protocol
Historical stages are sealed. `MANIFEST_SHA256.json`, `P2_SUPPLEMENT_MANIFEST_SHA256.json`, `P3_A2_PUBLIC_SHA256_MANIFEST.json`, and commit histories ensure no past data, reports, or protocols are altered. Stage P4 deliverables must be strictly segregated into `p4/`.

---

## 3. Detailed Technical Analysis: Channel Simulators and PA Models

### 3.1 Existing Implementations vs Requirements

| Feature | Existing Repository Implementation | Stage P4-0 Requirement (R1) | Technical Gap |
|---|---|---|---|
| **Cyclic Prefix (CP)** | $N=32, CP=16$ static concatenation in `p3b_time_link.py:tx_cp` | Time-domain CP insertion and removal with support for dynamic delay spread | Existing CP logic is static; needs seamless integration into dynamic pipeline |
| **Multipath Generation** | Static discrete FIR taps in `causal_linear_filter` (`p3b_time_link.py`) | Continuous-time / oversampled delay spread with fractional or multi-tap profiles | Needs proper interpolation or fractional tap support for time-domain simulation |
| **Doppler & ICI** | **None** (Taps are constant across the entire OFDM symbol) | Time-evolving path gains $h_l(t)$ creating realistic Inter-Carrier Interference (ICI) | **Complete gap**. Must implement Doppler phase evolution $e^{j 2\pi f_d t}$ across symbol samples |
| **Rapp PA Model** | Magnitude compression $z / (1 + (|z|/A)^4)^{0.25}$ (`probe.py:pa_magnitude`) | Oversampled Rapp PA with AM-AM ($p=2$) and AM-PM phase distortion | **AM-PM is completely missing**; **Oversampling is completely missing** |
| **Sampling Rate / Aliasing** | Nyquist-rate critical sampling ($N=32$ samples/symbol) | Oversampled time domain (factor $L \ge 4$, e.g. $L=4$ or $L=8$) to prevent aliased distortion | **Complete gap**. Nyquist clipping causes unphysical in-band aliasing of out-of-band intermod products |
| **Parity with Static Limit** | Audited in `probe.py:validate` for static zero-noise limit | Parity verification: with $f_d=0$ and linear PA, matches static matrix product $Y = H X + W$ to $<10^{-5}$ | Must build formal unit test verifying exact numerical parity |

### 3.2 Mathematical Formulation of the Required Dynamic Channel Simulator

#### 3.2.1 Doppler Evolution and Time-Domain Convolution
Let $N=32$ be the subcarrier count, $N_{CP}=16$ the cyclic prefix length, and $L \ge 4$ the oversampling factor ($N_{ov} = L \cdot N$).
Let $T_{sym} = (N + N_{CP}) T_s$ be the total symbol duration, where $T_s$ is the Nyquist sample period.
For each multipath tap $l \in \{1, \dots, L_{taps}\}$ with delay $\tau_l$ and power $p_l$:
1. Tap initial gains $h_{l, 0} \sim \mathcal{CN}(0, p_l)$.
2. In the presence of maximum Doppler frequency $f_d$, the tap gain evolves continuously across sample index $n \in \{0, \dots, N + N_{CP} - 1\}$:
   $$h_l[n] = h_{l, 0} \cdot \exp\left(j 2\pi f_{d, l} \frac{n T_s}{N}\right) \quad \text{or Jakes sum-of-sinusoids}$$
   where $f_{d, l} \in [-f_d, f_d]$ denotes the Doppler shift for path $l$.
3. The time-domain filtered signal is:
   $$y_{time}[n] = \sum_l h_l[n] x_{time}[n - d_l] + w[n]$$
4. At receiver, discard the first $N_{CP}$ samples and take the $N$-point FFT:
   $$Y[k] = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} y_{time}[n + N_{CP}] \exp\left(-j \frac{2\pi k n}{N}\right)$$
5. **Static Limit Verification**: When $f_d = 0$, $h_l[n] = h_{l, 0}$ is constant. Since $N_{CP} \ge \max(d_l)$, the linear convolution becomes cyclic convolution, yielding the exact diagonal system $Y[k] = H[k] X[k] + W[k]$, where $H = \text{FFT}(h)$.

#### 3.2.2 Oversampled Rapp PA Model with AM-AM and AM-PM
1. **Oversampling**: Input frequency-domain symbol $X$ ($N=32$) is zero-padded in the frequency domain to $N_{ov} = L \cdot N$ ($L=4 \implies N_{ov}=128$) and transformed via $N_{ov}$-point IFFT to obtain $x_{ov}[m]$ with proper power normalization.
2. **AM-AM Distortion**:
   $$g(r) = \frac{r}{\left(1 + \left(\frac{r}{A}\right)^{2p}\right)^{\frac{1}{2p}}}, \quad p = 2$$
   with saturation amplitude $A \in \{1.25, 2.25\}$.
3. **AM-PM Distortion**:
   $$\Phi(r) = \frac{\alpha r^q}{1 + \left(\frac{r}{\beta}\right)^q}$$
   where parameters $\alpha, \beta, q$ introduce realistic amplitude-dependent phase rotation (e.g., standard Rapp/Saleh extension with $\alpha \approx 0.5, \beta \approx 1.5, q \approx 2$). When linear, $\Phi(r) = 0$.
4. **Distorted Output**:
   $$x_{pa, ov}[m] = g(|x_{ov}[m]|) \cdot \exp\left(j \left(\arg(x_{ov}[m]) + \Phi(|x_{ov}[m]|)\right)\right)$$
5. **Downsampling / Low-Pass Filtering**: Out-of-band spectral regrowth is filtered out or transformed via $N_{ov}$-point FFT, selecting the central $N$ subcarriers, followed by $N$-point IFFT to return to Nyquist rate with cyclic prefix insertion.

---

## 4. Detailed Technical Analysis: Classical Baselines & Latency Harness

### 4.1 Classical Receiver Gaps

| Baseline | Codebase Status | Stage P4-0 Requirement (R1) | Required Implementation |
|---|---|---|---|
| **Bussgang-LMMSE** | **Missing** | Effective linear gain $\gamma$ & distortion variance $\sigma_D^2$ incorporated into LMMSE | Derive and compute $\gamma(A)$ and $\sigma_D^2(A)$; construct MMSE filter $\frac{\gamma^* H^*}{|\gamma H|^2 + |H|^2 \sigma_D^2 + \sigma_W^2}$ |
| **16-QAM Soft LLR** | **Missing** | Exact / max-log LLR calculation with noise/distortion variance calibration | Implement separable PAM-4 LLR expressions scaled by $\frac{2\mu}{\sigma_v^2}$; output calibrated soft bit probabilities |
| **Iterative DD-PA Cancellation** | **Missing** | 2–3 iterations of decision-directed time-domain distortion estimation & subtraction | Transmit reconstruction $\hat{x} \to f_{PA}(\hat{x}) - \gamma \hat{x} \to \hat{H} \hat{D} \to$ subtract from $Y \to$ re-equalize |
| **Latency Benchmark** | Exists for old models (`p3b_timing_extended.py`, `timing.py`) | Benchmarks all P4 classical baselines on CPU (and GPU if available) for batch 1 & 64 | Build unified harness measuring warmup, median, P95 latency in ms |

### 4.2 Mathematical Formulation of Classical Baselines

#### 4.2.1 Bussgang Decomposition for Nonlinear PA
By Bussgang's theorem, for OFDM transmit signal $x$ passing through nonlinear PA $f(x)$:
$$f(x) = \gamma x + d$$
where $\mathbb{E}[x d^*] = 0$.
The complex attenuation factor $\gamma$ is:
$$\gamma = \frac{\mathbb{E}[f(x) x^*]}{\mathbb{E}[|x|^2]}$$
The total PA output power is $P_{out} = \mathbb{E}[|f(x)|^2]$, so the uncorrelated nonlinear distortion variance is:
$$\sigma_D^2 = P_{out} - |\gamma|^2 \sigma_x^2$$
For normalized inputs ($\sigma_x^2 = 1$):
- At $A = 2.25$ (mild PA): $\gamma \approx 0.942$, $P_{out} \approx 0.895$, $\sigma_D^2 \approx 0.008$.
- At $A = 1.25$ (strong PA): $\gamma \approx 0.808$, $P_{out} \approx 0.670$, $\sigma_D^2 \approx 0.017$.

#### 4.2.2 Bussgang-LMMSE Channel Equalization
Under frequency-domain channel $H$, the received data signal is:
$$Y_d[k] = H[k] (\gamma X_d[k] + D[k]) + W[k] = \gamma H[k] X_d[k] + \left(H[k] D[k] + W[k]\right)$$
The effective channel is $H_{eff}[k] = \gamma H[k]$, and the total disturbance variance is:
$$\sigma_{eff}^2[k] = |H[k]|^2 \sigma_D^2 + \sigma_W^2$$
The Bussgang-LMMSE equalized symbol $\hat{X}_d[k]$ is:
$$\hat{X}_d[k] = \frac{\gamma^* H^*[k]}{|\gamma H[k]|^2 + |H[k]|^2 \sigma_D^2 + \sigma_W^2} Y_d[k]$$
The equivalent AWGN model after equalization has conditional mean $\mu_k X_d[k]$ with:
$$\mu_k = \frac{|\gamma H[k]|^2}{|\gamma H[k]|^2 + |H[k]|^2 \sigma_D^2 + \sigma_W^2}$$
and residual error variance $\sigma_{v, k}^2 = \mu_k (1 - \mu_k)$.

#### 4.2.3 Calibrated 16-QAM Soft LLR Computation
For Gray-coded 16-QAM with constellation points $s = (s_I + j s_Q)/\sqrt{10}$, $s_I, s_Q \in \{-3, -1, +1, +3\}$:
- Bit 0 ($b_0$): sign of real part ($s_I > 0$)
- Bit 1 ($b_1$): magnitude of real part ($|s_I| < 2$)
- Bit 2 ($b_2$): sign of imag part ($s_Q > 0$)
- Bit 3 ($b_3$): magnitude of imag part ($|s_Q| < 2$)

For an equalized observation $\hat{z} = \hat{X}_d / \mu$, the noise variance is $\sigma^2 = \sigma_v^2 / \mu^2$.
Using the exact piece-wise max-log formulation (normalized by $\sqrt{10}$):
$$LLR(b_0) = \frac{2 \sqrt{10}}{\sigma^2} \cdot \text{Re}(\hat{z}) \quad \text{(or piecewise linear form)}$$
$$LLR(b_1) = \frac{\sqrt{10}}{\sigma^2} \cdot \left( \frac{2}{\sqrt{10}} - |\text{Re}(\hat{z})| \right) \quad \text{(calibrated magnitude boundary)}$$
Analogously for bits 2 and 3 using $\text{Im}(\hat{z})$.
This ensures that the soft LLR directly outputs well-calibrated log-odds:
$$P(b_i = 1 | \hat{z}) = \frac{1}{1 + e^{-LLR(b_i)}} = \sigma(LLR(b_i))$$
minimizing the cross-entropy loss against ground truth bits.

#### 4.2.4 Iterative Decision-Directed PA Cancellation (2–3 Iterations)
The algorithm proceeds as:
1. **Iteration 0**:
   - Compute initial channel estimate $\hat{H}$ from pilots.
   - Equalize via Bussgang-LMMSE: $\hat{X}^{(0)} = \text{Equalize}(Y_d, \hat{H}, \sigma_D^2, \sigma_W^2)$.
   - Compute tentative symbol decisions $\hat{S}^{(0)} = \text{HardSlice}(\hat{X}^{(0)})$ (or soft expectation $\mathbb{E}[S | LLR]$).
2. **Iteration $i \in \{1, 2, 3\}$**:
   - Time-domain reconstruction of transmit symbol: $\hat{x}^{(i-1)} = \text{IFFT}(\hat{S}^{(i-1)})$.
   - Apply oversampled PA model to estimate the nonlinear distortion:
     $$\hat{d}_{time}^{(i-1)} = f_{PA}(\hat{x}^{(i-1)}) - \gamma \hat{x}^{(i-1)}$$
   - Transform to frequency domain: $\hat{D}^{(i-1)} = \text{FFT}(\hat{d}_{time}^{(i-1)})$.
   - Filter through channel estimate: $\hat{Y}_{dist}^{(i-1)} = \hat{H} \cdot \hat{D}^{(i-1)}$.
   - Subtract estimated distortion from received payload:
     $$Y_{cleaned}^{(i)} = Y_d - \hat{Y}_{dist}^{(i-1)}$$
   - Re-equalize cleaned signal with reduced distortion variance $\sigma_{D, i}^2 \ll \sigma_D^2$:
     $$\hat{X}^{(i)} = \frac{H^*}{|H|^2 + \sigma_{D, i}^2 |H|^2 + \sigma_W^2} Y_{cleaned}^{(i)}$$
   - Re-calculate calibrated LLRs and update symbol decisions $\hat{S}^{(i)}$.
3. **Termination**: Return final soft LLRs, hard bit predictions, and iteration history.

---

## 5. Technical Gaps and Specifications for Stage P4-0

### 5.1 Architecture of Stage P4 Directory Layout
To satisfy Acceptance Criterion:
> *"All new deliverables are housed in dedicated phase directories without modifying or overwriting historical P1-P3 or exploratory artifacts."*

Stage P4 must be housed in `p4/` at the repository root:
```text

└── p4/
    ├── __init__.py
    ├── config.py                 # P4 global configurations, seeds, numerology
    ├── simulator.py              # Time-domain dynamic simulator (Doppler, ICI, CP, oversampled Rapp PA)
    ├── classical_baselines.py    # Bussgang-LMMSE, soft LLR, Iterative DD-PA cancellation
    ├── benchmark_latency.py      # Latency benchmarking harness (CPU/GPU, batch 1 and 64)
    ├── test_p4_0_integrity.py    # Unit tests for parity with static limits, Bussgang, LLR, DD-PA
    └── results/                  # P4-0 output CSVs, JSON audits, BER curves
```

### 5.2 Exact Acceptance Checklist for Stage P4-0
- [ ] **CP & Dynamic Doppler PARITY**:
  - `simulator.py` verifies parity with static frequency-domain matrix multiplication $Y = H X + W$ when Doppler $f_d = 0$ and PA is linear ($A \to \infty$), with maximum complex error $< 10^{-5}$.
- [ ] **Oversampled Rapp PA**:
  - Implements $L \ge 4$ oversampling, Rapp AM-AM ($p=2$), and Rapp AM-PM distortion.
  - Supports mild ($A=2.25$) and strong ($A=1.25$) saturation.
- [ ] **Bussgang-LMMSE**:
  - Correctly evaluates $\gamma$ and $\sigma_D^2$.
  - Delivers lower BER than uncalibrated linear LMMSE under strong PA distortion ($A=1.25$).
- [ ] **Calibrated 16-QAM Soft LLRs**:
  - Computes exact/max-log bit LLRs.
  - Verified against synthetic AWGN with zero empirical calibration bias.
- [ ] **Iterative DD-PA Cancellation**:
  - Executes 2 to 3 iterations.
  - Monotonically reduces distortion and outperforms 1-pass Bussgang-LMMSE under strong PA distortion.
- [ ] **Latency Benchmarking**:
  - Benchmarks LS, DFT-8, Bussgang-LMMSE, and DD-PA (1, 2, 3 iterations) on CPU (4 threads) for batch sizes 1 and 64.
  - Outputs `latency_cpu.csv` and `timing.json`.

---

## 6. Conclusion
The repository has a rigorous tradition of empirical honesty, negative result disclosure, and immutable phase manifests. The gaps for Stage P4-0 are clearly identified, mathematically formulated, and completely solvable within standard PyTorch on the available CPU runtime without external closed libraries. All deliverables will be housed in `p4/`, preserving full backward compatibility and repository integrity.
