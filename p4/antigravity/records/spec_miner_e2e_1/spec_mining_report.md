# EM-FloatFormer Stage P4: Comprehensive Specification Mining Report for 4-Tier E2E Testing

**Author:** `teamwork_spec_miner` (EM-FloatFormer Stage P4 E2E Testing Track)
**Date:** 2026-11-15
**Working Directory:** ``
**State Directory:** `spec_miner_e2e_1`
**Target Test Suite:** `tests/test_p4_e2e.py` (Opaque-Box E2E Test Suite)

---

## Executive Summary & Specification Architecture

This specification document provides the authoritative, mathematically rigorous, and verifiable foundation for the **Stage P4 End-to-End (E2E) Test Suite**. Operating under strict opaque-box testing principles, this document formalizes the 22 inventoried features (**F1 through F22**) across Milestones M1 through M4, mapping each feature to a 4-tier testing hierarchy:
- **Tier 1: Feature Coverage** (Isolated feature mechanics, $\ge 5$ test cases per feature, $\ge 110$ tests total)
- **Tier 2: Boundary & Corner Conditions** (Extreme dynamic ranges, clipping limits, high Doppler, zero SNR, $\ge 5$ tests per feature, $\ge 110$ tests total)
- **Tier 3: Cross-Feature Combinations** (Pairwise physical and algorithmic interaction tests, $\ge 22$ test cases)
- **Tier 4: Real-World Workload Scenarios** (End-to-end cellular-grade slot transmission, $\ge 11$ full-pipeline scenarios)

### Physical Testing Invariants
1. **Opaque-Box Nonprivileged Decoupling**: The test harness tests only public tensor interfaces, CLI scripts, and observable outputs (channel estimates $\hat{H}$, bit log-odds $L$, decoded bit vectors $\hat{u}$). No test may peek at ground-truth channel $H_{\text{true}}$ or transmitted bits $b_{\text{true}}$ inside the receiver pipeline.
2. **Exact $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ Symmetry**: Scale factors $r \in [2^{-12}, 2^{12}]$ (72 dB dynamic range) and phase shifts $e^{j\phi}$ must yield zero bit decision flips and $< 10^{-5}$ relative output error.
3. **Statistical Reproducibility**: Exact integer error accounting ($E_{\text{bits}}, E_{\text{blocks}} \in \mathbb{Z}_{\ge 0}$) and deterministic 1024-state sign-flip permutation tests with Holm-Bonferroni adjustment.
4. **Historical Immutability**: All P1–P3 records, exploratory folders, and manifests must maintain bitwise SHA256 integrity.

---

## Features Discovered


| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| **F1** | RF / Impairment | Oversampled Rapp PA | Memoryless Rapp AM-AM ($p=2, A \in \{1.25, 2.25\}$) & AM-PM at $L \ge 4$ oversampling | $x_{time} \in \mathbb{C}^{B \times N}$, $A \in \mathbb{R}_{>0}$, $p, \alpha, \beta, L$ | $y_{pa} \in \mathbb{C}^{B \times N}$ | Raises `ValueError` if $L < 4$, $A \le 0$, or $p \le 0$ | `sub_orch_m1/SCOPE.md`, `Survey 1 §3.2.2` |
| **F2** | Channel / Physics | Dynamic Multipath Simulator | Time-variant FIR convolution with continuous Doppler ICI and CP insertion/removal | $x \in \mathbb{C}^{B \times N}$, profile, $f_d, N_{CP}$ | $y_{time} \in \mathbb{C}^{B \times (N+N_{CP})}$, $Y \in \mathbb{C}^{B \times N}$ | Raises `AssertionError` if tap delay exceeds $N_{CP}$ | `PROJECT.md:34`, `Survey 1 §3.2.1` |
| **F4** | Classical Baseline | Bussgang Decomposition & LMMSE | Bussgang attenuation $\gamma$, distortion $\sigma_D^2$, and Bussgang-LMMSE equalization | $Y_p, Y_d, \hat{H}, \sigma_w^2, A$ | $\hat{X}_d \in \mathbb{C}^{B \times N_d}, \gamma \in \mathbb{R}, \sigma_D^2 \in \mathbb{R}$ | Raises `ValueError` on invalid $A$ or singular matrix | `PROJECT.md:36`, `Survey 1 §4.2.1-2` |
| **F5** | Classical Baseline | Calibrated 16-QAM Soft LLRs | True posterior log-likelihood ratios for 16-QAM under effective distortion-plus-noise variance | $\hat{X}_d \in \mathbb{C}^{B \times N_d}, \mu, \sigma_v^2$ | $L \in \mathbb{R}^{B \times N_d \times 4}$ | NaN/Inf on $\sigma_v^2 \le 0$; out-of-range bounds | `PROJECT.md:37`, `Survey 1 §4.2.3` |
| **F6** | Classical Baseline | Iterative DD-PA Cancellation | 2-3 iteration decision-directed time-domain distortion reconstruction, channel filtering, and subtraction | $Y_p, Y_d, \hat{H}, \sigma_w^2, A, K$ | $\hat{H}^{(K)}, L^{(K)}, \hat{b} \in \{0,1\}^{B \times N_d \times 4}$ | Divergence warning if residual error increases | `PROJECT.md:38`, `Survey 1 §4.2.4` |
| **F7** | Performance / Benchmark | Classical Latency Benchmark | Standardized CPU (4 threads) and GPU latency measurement harness for baselines | Batch sizes $\{1, 64\}$, model handles | CSV/JSON records: median, P95 (ms) | Fails if timing is non-monotonic or execution errors | `PROJECT.md:39`, `Survey 1 §4.2` |
| **F8** | Neural Architecture | Exact $C^*$ Gauge Normalization | Pilot RMS gauge anchor $s_0 = \sqrt{\frac{1}{N}\sum \|Y_p\|^2}$ + internal Complex RMSNorm | $Y_p, Y_d \in \mathbb{C}^{B \times N}, \sigma_w^2 \in \mathbb{R}^B$ | $\bar{Y}_p, \bar{Y}_d, \bar{\sigma}_w^2$ (dimensionless) | Zero-division guard via relative epsilon $\epsilon_{\text{rel}}$ | `PROJECT.md:40`, `Survey 2 §2.2` |
| **F9** | Neural Architecture | Hermitian Green's Attention | Complex Linear without bias ($b=0$), complex relative RoPE, radial activation | $Z \in \mathbb{C}^{B \times N \times D}$ | $Z_{\text{out}} \in \mathbb{C}^{B \times N \times D}$ | Raises `AssertionError` if bias exists or phase rotates | `PROJECT.md:41`, `Survey 2 §2.3` |
| **F10** | Neural Architecture | Bidirectional Pilot-Data Attention | Mutual cross-attention exchange between channel estimate stream and payload demapping | $H^{(l)} \in \mathbb{C}^{B \times N \times D}, X_d^{(l)} \in \mathbb{C}^{B \times N_d \times D}$ | $H^{(l+1)}, X_d^{(l+1)}$ | Dimension mismatch if feature widths disagree | `PROJECT.md:42`, `Survey 2 §2.4` |
| **F11** | Training / Loss | Joint Physical Consistency Loss | Joint objective: calibrated LLR BCE + normalized channel NMSE + forward PA residual consistency | $L, b_{\text{true}}, \hat{H}, H_{\text{true}}, Y_d, \hat{X}_d$ | $\mathcal{L}_{\text{joint}} \in \mathbb{R}_{\ge 0}$ (scalar) | Scale drift failure if NMSE/Residual lacks scaling | `PROJECT.md:43`, `Survey 2 §3` |
| **F12** | Verification / Invariance | Numerical Equivariance Verification | Strict $C^*$ invariance unit tests across $r \in [2^{-12}, 2^{12}]$ and $\phi \in [-\pi, \pi]$ | Transformed tensors $(c Y_p, c Y_d, \|c\|^2 \sigma_w^2)$ | Invariance metrics $\epsilon_H, \epsilon_L, N_{\text{flip}}$ | Assert error if $\epsilon_H \ge 10^{-5}$ or $N_{\text{flip}} > 0$ | `PROJECT.md:44`, `Survey 2 §4` |
| **F13** | 2D / Grid Structure | 2D Slot Simulator & Sparse DMRS | Multi-symbol slots ($N_{\text{symb}}=14, N_{\text{sc}}=32/64$) with sparse DMRS ($\le 10\%$) | $B, N_{\text{symb}}, N_{\text{sc}}$, channel profile | $Y_{\text{grid}} \in \mathbb{C}^{B \times 14 \times N_{\text{sc}}}$, DMRS mask | Overhead error if pilot count $> 10\%$ of grid | `PROJECT.md:45`, `Survey 3 §2.1-2` |
| **F14** | Channel / Dynamic | Continuous Doppler ICI & CFO | High-Doppler time-varying fading & sample-level carrier frequency offset phase drift | Grid $X$, $f_d T_{\text{sym}} \in [0.01, 0.05]$, $\epsilon_{\text{cfo}}$ | Received grid $Y$, ICI matrix $H_{s}[k, k']$ | Spectrum leak or phase wrap if improperly evaluated | `PROJECT.md:46`, `Survey 3 §2.3-4` |
| **F15** | Channel Coding | Native Outer Channel Coding | Native PyTorch Rate-1/2 LDPC (5G NR BG2) or Conv code ($K=7$) with soft decoding | Bits $u \in \{0, 1\}^K$, LLRs $L \in \mathbb{R}^N$ | Codewords $c \in \{0, 1\}^N$, decoded $\hat{u} \in \{0, 1\}^K$ | Parity failure if $H \hat{c}^T \ne 0 \pmod 2$ | `PROJECT.md:47`, `Survey 3 §3.2` |
| **F16** | 2D Classical Baseline | 2D Classical Baselines | Separable 2D-LMMSE channel estimation & 2D iterative decision-directed PA cancellation | 2D grid $Y$, DMRS observations, $R_{\text{time}}, R_{\text{freq}}$ | Full grid channel $\hat{H}_{s, k}$, soft LLRs | Singular correlation matrix if noise regularizer missing | `PROJECT.md:48`, `Survey 3 §3.4` |
| **F17** | Performance / BLER | Coded BLER Evaluation | Coded Block Error Rate evaluation across OOD Doppler and delay spreads | Ground-truth $u$, decoded $\hat{u}$ | $\text{BLER} \in [0.0, 1.0]$ | Monotonicity violation if BLER increases with SNR | `PROJECT.md:49`, `Survey 3 §3.3` |
| **F18** | Probing / Causal | Factorial Ablation Matrix | Orthogonal $2^3$ ablation isolating $C^*$ gauge, Green's attention, and joint propagation | 8 model configurations, evaluation frames | Main effects $\text{ME}(F_i)$, interactions $\text{INT}(F_i, F_j)$ | Rank deficiency if arms share weights incorrectly | `PROJECT.md:50`, `Survey 3 §4` |
| **F19** | Numeric / Stability | Numeric Stability & Precision Audit | FP32, BF16, and FP16 execution across 24 octaves of dynamic range $r \in [2^{-12}, 2^{12}]$ | Model, inputs at $r \in \{2^{-12}, \dots, 2^{12}\}$, dtypes | NaN/Inf counts, $\max \|L_{\text{half}} - L_{\text{fp32}}\|$ | Assert fail on any NaN/Inf or flip rate $> 0.05\%$ | `PROJECT.md:51`, `Survey 3 §5` |
| **F20** | Confirmatory Protocol | Pre-Frozen Confirmatory Protocol | Pre-declared protocol over 10 fresh seeds ($74001–74010$) with exact integer bit accounting | 10 seeds, test conditions | Integer error tables $E_{\text{bits}}, E_{\text{blocks}}$ | Integrity violation if seed in historical set | `PROJECT.md:52`, `Survey 3 §6.1-3` |
| **F21** | Statistical Testing | Sign-Flip Permutation Tests | Two-sided 1024-state exact permutation test with Holm-Bonferroni step-down adjustment | Paired per-seed deltas $\delta_s \in \mathbb{R}^{10}$ | Exact $p$-values, Holm-adjusted $p_{\text{Holm}}$ | Assert fail if $p_{\text{exact}} \notin [1/1024, 1.0]$ | `PROJECT.md:53`, `Survey 3 §6.4` |
| **F22** | Repository Governance | Repository Governance & Immutability | Bitwise SHA256 integrity verification of all historical P1-P3 files and manifests | File paths in `MANIFEST_SHA256.json` | Hash match report, exit code 0 | Non-zero exit code if any historical file differs | `PROJECT.md:54`, `Survey 3 §6.5` |

---

## Edge Cases Discovered

| # | Feature | Input / Condition | Observed / Required Behavior |
|---|---------|-------------------|------------------------------|
| **E1** | F1 (Rapp PA) | Input amplitude $r = 0.0$ | Output exactly $0.0$; no NaN from $0/0$ in phase or magnitude |
| **E2** | F1 (Rapp PA) | Extreme amplitude $r \to \infty$ ($r = 10^6$) | Output magnitude strictly saturates to $A$; phase approaches $\alpha \beta^2$ |
| **E3** | F1 (Rapp PA) | Linear limit $A \to \infty$ | Output exactly equals input ($g(r) = r, \Phi(r) = 0$); zero distortion |
| **E4** | F2 (Simulator) | Zero Doppler ($f_d = 0$) | Channel impulse response is strictly time-invariant across all symbol samples |
| **E5** | F2 (Simulator) | Delay spread equals $N_{CP}$ | Last tap at delay $N_{CP}-1$; cyclic convolution remains intact without ISI |
| **E6** | F3 (Parity) | Zero noise ($W = 0$) and linear PA | Frequency-domain observation $Y[k]$ equals $H[k] X[k]$ with error $< 10^{-6}$ |
| **E7** | F4 (Bussgang) | Extreme clipping $A = 0.5$ | Attenuation $\gamma$ drops significantly ($\gamma < 0.6$), distortion variance $\sigma_D^2$ rises |
| **E8** | F5 (Soft LLR) | Infinite SNR ($\sigma^2 \to 0$) | LLR magnitudes saturate cleanly to clamp threshold ($\pm 50.0$), no Inf/NaN |
| **E9** | F5 (Soft LLR) | Zero SNR ($\sigma^2 \to \infty$) | All LLRs collapse symmetrically to $0.0$, yielding maximal uncertainty |
| **E10** | F6 (IDD-PA) | Clean signal without PA ($A = \infty$) | Iterations 1-3 reconstruct zero distortion ($\hat{D} = 0$), output identical to Iter 0 |
| **E11** | F8 ($C^*$ Norm) | Minimum gain $r = 2^{-12} \approx 2.44 \times 10^{-4}$ | Global anchor $s_0$ scales by $2^{-12}$; internal tokens maintain RMS $\equiv 1.0$ |
| **E12** | F8 ($C^*$ Norm) | Maximum gain $r = 2^{12} = 4096.0$ | Internal tokens maintain RMS $\equiv 1.0$; zero floating-point overflow |
| **E13** | F9 (Hermitian) | Common phase rotation $\phi = \pi$ | Attention affinity matrix $S_{m,n}$ is strictly invariant; output rotates by $-1$ |
| **E14** | F11 (Joint Loss)| Extreme channel gain $r = 2^{12}$ | $\mathcal{L}_{\text{NMSE}}$ and $\mathcal{L}_{\text{PA-res}}$ remain numerically identical to $r = 1.0$ |
| **E15** | F12 (Equiv) | Combined extreme scale and phase ($c = 2^{12} e^{j 3\pi/4}$) | Zero hard bit flips ($N_{\text{flip}} \equiv 0$); relative channel error $< 10^{-5}$ |
| **E16** | F14 (Doppler) | High Doppler ($f_d T_{\text{sym}} = 0.05$) | Off-diagonal ICI energy exceeds $10\%$ of diagonal energy; 1D receivers fail |
| **E17** | F14 (CFO) | Non-zero CFO without compensation | Constellation steadily rotates across the 14 symbols by $2\pi \epsilon (N_{\text{sc}}+N_{\text{cp}})/N_{\text{sc}}$ |
| **E18** | F15 (Outer FEC) | All-zero input bits $u = \mathbf{0}$ | Encoded codeword is all-zero $\mathbf{0}$; decoder converges in 1 iteration |
| **E19** | F15 (Outer FEC) | Extreme negative LLRs ($L = -100$) | Hard decisions decode to all-ones $\mathbf{1}$; no numerical overflow |
| **E20** | F18 (Ablations)| Completely ablated Arm 8 (real, decoupled, no gauge) | Fails catastrophically at $r = 2^{-12}$, establishing the baseline necessity |
| **E21** | F19 (Precision)| FP16 execution at $r = 2^{12}$ without gauge norm | Immediate overflow to `+inf` and `NaN` (validating failure mechanism) |
| **E22** | F21 (Sign-Flip)| All paired differences strictly zero ($\delta_s \equiv 0$) | Exact permutation $p$-value returns $1.0$ |
| **E23** | F21 (Sign-Flip)| All paired differences strictly positive ($\delta_s > 0$) | Exact permutation $p$-value returns $2 \times 2^{-10} = 2/1024 \approx 0.00195$ |

---

## Detailed Specifications for Feature Inventory (F1 to F22)

### Milestone 1: Simulator & Nonlinear Classical Baselines (F1–F7)

#### Feature F1: Oversampled Rapp Power Amplifier (AM-AM & AM-PM)
- **Mathematical Formulation**:
  Let $x_{\text{time}} \in \mathbb{C}^{B \times N}$ be the Nyquist time-domain signal. To prevent aliasing of out-of-band intermodulation distortion products, $x_{\text{time}}$ is oversampled by factor $L \ge 4$ via frequency-domain zero-padding:
  $$X_{\text{ov}}[k] = \begin{cases} \sqrt{L} \cdot \text{FFT}(x_{\text{time}})[k], & 0 \le k < N/2 \\ 0, & N/2 \le k < L \cdot N - N/2 \\ \sqrt{L} \cdot \text{FFT}(x_{\text{time}})[k - (L-1)N], & L \cdot N - N/2 \le k < L \cdot N \end{cases}$$
  $$x_{\text{ov}}[m] = \text{IFFT}_{L \cdot N}(X_{\text{ov}})[m]$$
  Let $r[m] = |x_{\text{ov}}[m]|$. The distorted signal $x_{\text{pa, ov}}[m]$ is:
  $$x_{\text{pa, ov}}[m] = g(r[m]) \cdot \exp\left(j \left(\arg(x_{\text{ov}}[m]) + \Phi(r[m])\right)\right)$$
  - **AM-AM (Rapp Model, $p=2$)**:
    $$g(r) = \frac{r}{\left(1 + \left(\frac{r}{A}\right)^{2p}\right)^{\frac{1}{2p}}} = \frac{r}{\left(1 + \left(\frac{r}{A}\right)^4\right)^{0.25}}, \quad A \in \{1.25, 2.25\}$$
  - **AM-PM Phase Distortion**:
    $$\Phi(r) = \frac{\alpha r^2}{1 + (r / \beta)^2}, \quad \text{with } \alpha = 0.5, \beta = 1.0 \quad (\Phi(r) \equiv 0 \text{ when linear})$$
  - **Decimation**: $x_{\text{pa, ov}}$ is filtered with an ideal anti-aliasing brickwall filter and downsampled back to $N$ subcarriers.
- **Tensors**:
  - Input: `x_time`: `(B, N)`, dtype `torch.complex64` (mean power $\mathbb{E}[|x|^2] \approx 1$).
  - Output: `y_pa`: `(B, N)`, dtype `torch.complex64`.
  - Parameters: `knee=2.0`, `A` in $\{1.25, 2.25, \infty\}$, `alpha=0.5`, `beta=1.0`, `oversample=4`.
- **Acceptance Criteria & Tolerances**:
  - In linear limit ($A \to \infty, \alpha=0$): $\|y_{\text{pa}} - x_{\text{time}}\|_\infty < 10^{-6}$.
  - Peak output envelope: $\max |y_{\text{pa}}| \le A \cdot (1 + 10^{-5})$.
  - Spectral regrowth verification: Out-of-band ACPR (Adjacent Channel Power Ratio) $\ge 15$ dB for $A=1.25$ before filtering.
- **Tier 2 Boundary Conditions**:
  - $A = 1.25$ (severe clipping), $A = 2.25$ (mild clipping), $A = 10^6$ (linear), $p=1$ vs $p=4$, input amplitude $r=0$.
- **Tier 3 Cross-Feature Interaction**:
  - F1 + F2 (Rapp PA + Dynamic Multipath Doppler), F1 + F6 (Rapp PA + Iterative Cancellation).
- **Tier 4 Scenario Context**:
  - Standard cellular uplink with $A=1.25$ under 16-QAM constellation.

---

#### Feature F2: Dynamic Time-Domain Multipath Simulator
- **Mathematical Formulation**:
  $N=32$ (or 64), cyclic prefix length $N_{\text{CP}}=16$. Multipath delay taps $l \in \{0, \dots, L_{\text{taps}}-1\}$ with delay indices $d_l \in [0, N_{\text{CP}}-1]$ and power $p_l$ ($\sum p_l = 1$).
  For maximum Doppler frequency $f_d$, the continuous path gain $h_l[n]$ evolves across sample index $n \in \{0, \dots, N + N_{\text{CP}} - 1\}$:
  $$h_l[n] = h_{l, 0} \cdot \exp\left(j 2\pi f_{d, l} \frac{n T_s}{N}\right), \quad f_{d, l} \in [-f_d, f_d]$$
  Transmit signal with CP: $x_{\text{cp}}[n] = x_{\text{time}}[(n - N_{\text{CP}}) \bmod N]$ for $n \in \{0, \dots, N + N_{\text{CP}} - 1\}$.
  Time-domain causal convolution:
  $$y_{\text{time}}[n] = \sum_{l=0}^{L_{\text{taps}}-1} h_l[n] \cdot x_{\text{cp}}[n - d_l] + w[n], \quad w[n] \sim \mathcal{CN}(0, \sigma_w^2 / N)$$
  Receiver discards the first $N_{\text{CP}}$ samples and computes $N$-point FFT:
  $$Y[k] = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} y_{\text{time}}[n + N_{\text{CP}}] \cdot e^{-j 2\pi k n / N}$$
- **Tensors**:
  - Inputs: `x_time`: `(B, N)`, profile: `List[float]`, `fd`: `float`, `snr_db`: `float`.
  - Outputs: `y_time`: `(B, N + N_CP)`, `Y`: `(B, N)`, `H_diag`: `(B, N)`.
- **Acceptance Criteria & Tolerances**:
  - Time-domain energy conservation: $\mathbb{E}[\|y_{\text{time}}\|^2] = \mathbb{E}[\|x_{\text{time}}\|^2] + \sigma_w^2 \cdot \frac{N+N_{\text{CP}}}{N} \pm 2\%$.
  - Doppler ICI leakage: When $f_d T_{\text{sym}} \ge 0.03$, inter-carrier interference energy on adjacent subcarriers $\ge -25$ dBc.
- **Tier 2 Boundary Conditions**:
  - $f_d = 0$ (static), $f_d T_{\text{sym}} = 0.05$ (extreme high speed), delay spread $= N_{\text{CP}}-1$, single-path flat fading, SNR $= -\infty$ dB ($\sigma_w^2 \to \infty$).
- **Tier 3 Interaction**:
  - F2 + F14 (Time-domain multipath + continuous 2D slot Doppler drift).
- **Tier 4 Scenario Context**:
  - Vehicular speed $v = 120$ km/h ($f_d = 300$ Hz) at 5 GHz carrier.

---

#### Feature F3: Static-Limit Parity Check
- **Mathematical Formulation**:
  When Doppler is zero ($f_d = 0$) and the power amplifier is linear ($A \to \infty, \Phi(r) = 0$):
  $$h_l[n] \equiv h_{l, 0} \implies H[k] = \sum_{l} h_{l, 0} e^{-j 2\pi k d_l / N}$$
  Since $N_{\text{CP}} \ge \max(d_l)$, linear convolution is identical to cyclic convolution:
  $$Y_{\text{sim}}[k] = H[k] \cdot X[k] + W[k]$$
  Relative parity error metric:
  $$\epsilon_{\text{parity}} = \frac{\max_k |Y_{\text{sim}}[k] - (H[k] X[k] + W[k])|}{\max_k |Y_{\text{sim}}[k]| + 10^{-12}}$$
- **Tensors**:
  - Input: $X \in \mathbb{C}^{B \times N}$, $h_{\text{taps}} \in \mathbb{C}^{B \times L_{\text{taps}}}$, $W \in \mathbb{C}^{B \times N}$.
  - Output: $\epsilon_{\text{parity}} \in \mathbb{R}$.
- **Acceptance Criteria & Tolerances**:
  - Strict parity threshold: $\epsilon_{\text{parity}} < 1.0 \times 10^{-5}$ under FP32.
- **Tier 2 Boundary Conditions**:
  - Test on all canonical delay profiles: `train3` (taps 0,1,2), `equal4` (taps 0,1,2,3), `long6` (taps 0..5), `sparse12` (taps 0,4,11).
- **Tier 3 Interaction**:
  - F3 + F8 ($C^*$ gauge normalization under static limit).
- **Tier 4 Scenario Context**:
  - Zero-Doppler line-of-sight static baseline calibration.

---

#### Feature F4: Bussgang Decomposition & LMMSE Equalization
- **Mathematical Formulation**:
  For Gaussian/OFDM signal $x$ passing through nonlinear PA $f(x)$:
  $$f(x) = \gamma x + d, \quad \mathbb{E}[x d^*] = 0$$
  Attenuation factor:
  $$\gamma = \frac{\mathbb{E}[f(x) x^*]}{\mathbb{E}[|x|^2]} \in \mathbb{R}_{>0}$$
  Total output power $P_{\text{out}} = \mathbb{E}[|f(x)|^2]$, uncorrelated distortion variance:
  $$\sigma_D^2 = P_{\text{out}} - \gamma^2 \sigma_x^2$$
  Effective channel: $H_{\text{eff}}[k] = \gamma H[k]$. Total disturbance variance: $\sigma_{\text{eff}}^2[k] = |H[k]|^2 \sigma_D^2 + \sigma_W^2$.
  Bussgang-LMMSE filter:
  $$W[k] = \frac{\gamma^* H^*[k]}{|\gamma H[k]|^2 + |H[k]|^2 \sigma_D^2 + \sigma_W^2}$$
  Equalized symbol: $\hat{X}_d[k] = W[k] Y_d[k]$.
  Equivalent post-equalization model: $\hat{X}_d[k] = \mu_k X_d[k] + v_k$, where $\mu_k = W[k] \gamma H[k] \in [0, 1]$ and $\sigma_{v, k}^2 = \mu_k (1 - \mu_k)$.
- **Tensors**:
  - Inputs: `Yp`: `(B, N_p)`, `Yd`: `(B, N_d)`, `nv`: `(B,)`, `A`: `float`.
  - Outputs: `H_hat`: `(B, N)`, `X_hat`: `(B, N_d)`, `gamma`: `float`, `sigma_D2`: `float`.
- **Acceptance Criteria & Tolerances**:
  - Parameter values for normalized input ($\sigma_x^2 = 1$):
    - $A=2.25 \implies \gamma \in [0.935, 0.945]$, $\sigma_D^2 \in [0.007, 0.009]$.
    - $A=1.25 \implies \gamma \in [0.800, 0.815]$, $\sigma_D^2 \in [0.015, 0.019]$.
  - Superiority: Bussgang-LMMSE MSE $\le$ Standard linear LMMSE MSE $- 0.3$ dB at 20 dB SNR ($A=1.25$).
- **Tier 2 Boundary Conditions**:
  - $A \to \infty$ ($\gamma \to 1.0, \sigma_D^2 \to 0$), $A = 1.0$ (extreme non-linearity), $\sigma_W^2 = 0$ (distortion-limited regime).
- **Tier 3 Interaction**:
  - F4 + F6 (Bussgang LMMSE initialization for Iterative DD-PA cancellation).
- **Tier 4 Scenario Context**:
  - 16-QAM transmission at 15 dB SNR with $A=1.25$.

---

#### Feature F5: Calibrated 16-QAM Soft LLRs
- **Mathematical Formulation**:
  Gray-coded 16-QAM constellation $\mathcal{S}_{16}$ normalized to unit energy $E_s = 1$:
  $$s = \frac{s_I + j s_Q}{\sqrt{10}}, \quad s_I, s_Q \in \{-3, -1, +1, +3\}$$
  Bit labeling:
  - $b_0 = \mathbb{I}(s_I > 0)$
  - $b_1 = \mathbb{I}(|s_I| < 2)$
  - $b_2 = \mathbb{I}(s_Q > 0)$
  - $b_3 = \mathbb{I}(|s_Q| < 2)$
  For unbiased equalized observation $\hat{z} = \hat{X}_d / \mu_k$ with effective noise variance $\sigma_k^2 = \sigma_{v, k}^2 / \mu_k^2$:
  Piecewise Max-Log LLR equations:
  $$L(b_0) = \frac{4}{\sqrt{10} \sigma_k^2} \text{Re}(\hat{z})$$
  $$L(b_1) = \frac{4}{\sqrt{10} \sigma_k^2} \left( \frac{2}{\sqrt{10}} - |\text{Re}(\hat{z})| \right)$$
  $$L(b_2) = \frac{4}{\sqrt{10} \sigma_k^2} \text{Im}(\hat{z})$$
  $$L(b_3) = \frac{4}{\sqrt{10} \sigma_k^2} \left( \frac{2}{\sqrt{10}} - |\text{Im}(\hat{z})| \right)$$
- **Tensors**:
  - Input: `X_hat`: `(B, N_d)`, `mu`: `(B, N_d)`, `sigma_v2`: `(B, N_d)`.
  - Output: `LLRs`: `(B, N_d, 4)`, dtype `torch.float32`.
- **Acceptance Criteria & Tolerances**:
  - Reliability calibration: $\mathbb{E}[b_i | L(b_i)] = \frac{1}{1 + e^{-L(b_i)}} = \sigma(L(b_i))$.
  - Calibration bias: Expected Calibration Error (ECE) $< 0.02$ on synthetic AWGN.
  - Sign correctness: $(L(b_i) > 0) \equiv b_i$ matches hard minimum distance slicer.
- **Tier 2 Boundary Conditions**:
  - Boundary crossing points $|\text{Re}(\hat{z})| = 2/\sqrt{10} \approx 0.63245$ (where $L(b_1) = 0$).
  - $\sigma_k^2 \to 0$ (clamp to $\pm 50$), $\sigma_k^2 \to \infty$ ($L \to 0$).
- **Tier 3 Interaction**:
  - F5 + F15 (Soft LLRs directly driving native LDPC Min-Sum decoder).
- **Tier 4 Scenario Context**:
  - Demapping under combined multipath fading, PA clipping, and AWGN at 10 dB SNR.

---

#### Feature F6: Iterative Decision-Directed PA Cancellation (IDD-PA)
- **Mathematical Formulation**:
  1. **Iteration 0**:
     $$\hat{H}^{(0)} = \text{LS/LMMSE}(Y_p), \quad \hat{X}^{(0)} = \text{Bussgang-Equalize}(Y_d, \hat{H}^{(0)}), \quad \hat{S}^{(0)} = \text{HardSlice}(\hat{X}^{(0)})$$
  2. **Iteration $i \in \{1, \dots, K\}$** ($K \in \{2, 3\}$):
     - Time-domain reconstruction: $\hat{x}_{\text{time}}^{(i-1)} = \text{IFFT}(\hat{S}^{(i-1)})$.
     - Oversampled PA distortion estimation:
       $$\hat{d}_{\text{time}}^{(i-1)} = f_{\text{PA}}(\hat{x}_{\text{time}}^{(i-1)}) - \gamma \hat{x}_{\text{time}}^{(i-1)}$$
     - Frequency-domain distortion: $\hat{D}^{(i-1)} = \text{FFT}(\hat{d}_{\text{time}}^{(i-1)})$.
     - Distorted channel spillover: $\hat{Y}_{\text{dist}}^{(i-1)} = \hat{H}^{(i-1)} \odot \hat{D}^{(i-1)}$.
     - Subtraction:
       $$Y_{\text{cleaned}}^{(i)} = Y_d - \hat{Y}_{\text{dist}}^{(i-1)}$$
     - Re-equalization with updated residual distortion variance $\sigma_{D, i}^2 \approx \sigma_D^2 \cdot \text{SER}^{(i-1)}$:
       $$\hat{X}^{(i)} = \text{Equalize}(Y_{\text{cleaned}}^{(i)}, \hat{H}^{(i-1)}, \sigma_{D, i}^2)$$
     - Re-compute soft LLRs and update symbol decisions $\hat{S}^{(i)}$.
- **Tensors**:
  - Inputs: `Yp`: `(B, N_p)`, `Yd`: `(B, N_d)`, `nv`: `(B,)`, `iterations`: `int` ($2$ or $3$).
  - Outputs: `H_hat`: `(B, N)`, `llrs`: `(B, N_d, 4)`, `hard_bits`: `(B, N_d, 4)`.
- **Acceptance Criteria & Tolerances**:
  - Convergence: $\text{BER}(\text{Iter } 2) \le \text{BER}(\text{Iter } 1) \le \text{BER}(\text{Iter } 0)$.
  - Gain at $A=1.25$: At least $0.5$ dB SNR gain at $\text{BER} = 10^{-2}$ over Iter 0.
- **Tier 2 Boundary Conditions**:
  - Number of iterations $K \in \{1, 2, 3, 5\}$; $A = 1.25$ (strong) vs $A = 2.25$ (mild); initial decisions 100% erroneous.
- **Tier 3 Interaction**:
  - F6 + F16 (Iterative cancellation extended to 2D time-frequency grids).
- **Tier 4 Scenario Context**:
  - 3-iteration cancellation in cell-edge transmission.

---

#### Feature F7: Classical Latency Benchmark
- **Mathematical Formulation**:
  Measure execution time $\tau$ across $M_{\text{trials}} = 100$ iterations after $M_{\text{warmup}} = 20$ iterations:
  $$\tau_{\text{median}} = \text{Median}(\{\tau_m\}), \quad \tau_{\text{P95}} = \text{Quantile}_{0.95}(\{\tau_m\})$$
  Harness standardizes on 4 CPU threads: `torch.set_num_threads(4)`.
- **Tensors**:
  - Evaluates batch sizes $B \in \{1, 64\}$ across baselines: LS, DFT-8, Bussgang-LMMSE, IDD-PA (1, 2, 3 iter).
  - Outputs: CSV and JSON format metrics.
- **Acceptance Criteria & Tolerances**:
  - Reproducibility: Coefficient of variation $\sigma_\tau / \mu_\tau < 0.15$ across trials.
  - Non-zero monotonic latency: $\tau(\text{IDD-PA-3}) > \tau(\text{IDD-PA-1}) > \tau(\text{Bussgang-LMMSE}) > \tau(\text{LS})$.
- **Tier 2 Boundary Conditions**:
  - Single batch $B=1$ (latency-critical) vs $B=64$ (throughput-critical); CPU vs GPU execution.
- **Tier 3 Interaction**:
  - F7 + F19 (Latency measurement under mixed precision BF16/FP16).
- **Tier 4 Scenario Context**:
  - Frame budget constraint: execution time $< 10$ ms per slot on CPU.

---

### Milestone 2: $C^*$-Equivariant EM-Native Transformer (F8–F12)

#### Feature F8: Exact $C^*$ Gauge Normalization
- **Mathematical Formulation**:
  The gauge group $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ acts on electromagnetic signals via $c = r e^{j\phi} \in \mathbb{C}^*$:
  $$(Y_p, Y_d, \sigma^2) \mapsto (c Y_p, c Y_d, |c|^2 \sigma^2)$$
  1. **Global Pilot Energy Gauge Anchor**:
     $$s_0 = \sqrt{\frac{1}{N_p} \sum_{k=1}^{N_p} |Y_p[k]|^2}$$
     Under transformation: $s_0 \mapsto |c| s_0 = r s_0$.
     Define scale-free dimensionless representations:
     $$\bar{Y}_p = \frac{Y_p}{s_0}, \quad \bar{Y}_d = \frac{Y_d}{s_0}, \quad \bar{\sigma}^2 = \frac{\sigma^2}{s_0^2}$$
     Note: $\bar{\sigma}^2$ is **strictly invariant** to $c$, and $\bar{Y}$ transforms purely under $\mathrm{U}(1)$ ($\bar{Y} \mapsto e^{j\phi} \bar{Y}$).
  2. **Internal Complex RMSNorm**:
     For hidden activations $Z \in \mathbb{C}^{B \times N \times D}$:
     $$\text{RMS}(Z)_{b,n} = \sqrt{\frac{1}{D} \sum_{d=1}^D |Z_{b,n,d}|^2 + \epsilon_{\text{rel}} \cdot \langle |Z|^2 \rangle}$$
     where $\epsilon_{\text{rel}} = 10^{-6}$ is dimensionless. With learnable real radial channel gain $\boldsymbol{\gamma} \in \mathbb{R}^D$:
     $$\text{CRMSNorm}(Z) = \left( \frac{Z}{\text{RMS}(Z)} \right) \odot \boldsymbol{\gamma}$$
  3. **Channel Recovery**:
     Physical channel output: $\hat{H} = s_0 \cdot \bar{H}$, guaranteeing $\hat{H}' = c \cdot \hat{H}$.
- **Tensors**:
  - Inputs: `Yp`: `(B, N_p)`, `Yd`: `(B, N_d)`, `nv`: `(B,)`.
  - Dimensionless tokens: `(B, N, D)`, dtype `torch.complex64`.
  - Scale factor: `s0`: `(B, 1)`, dtype `torch.float32`.
- **Acceptance Criteria & Tolerances**:
  - Scale range: Tested across $r \in [2^{-12}, 2^{12}]$ ($72$ dB dynamic range).
  - Channel estimate equivariance: $\frac{\|\hat{H}(c Y) - c \hat{H}(Y)\|_\infty}{|c| (\|\hat{H}\|_\infty + 10^{-12})} < 1.0 \times 10^{-5}$.
  - Bit logit invariance: $\|L(c Y) - L(Y)\|_\infty < 1.0 \times 10^{-5}$.
  - Zero bit decision flips: $N_{\text{flip}} = \sum |(L' > 0) \ne (L > 0)| \equiv 0$.
- **Tier 2 Boundary Conditions**:
  - Extreme minimum scale $r = 2^{-12} \approx 2.44 \times 10^{-4}$, extreme maximum scale $r = 2^{12} = 4096$, random angles $\phi \in [-\pi, \pi]$.
- **Tier 3 Interaction**:
  - F8 + F14 ($C^*$ gauge normalization under continuous CFO phase drift).
- **Tier 4 Scenario Context**:
  - Wide dynamic range cellular link with shadow fading up to $\pm 36$ dB.

---

#### Feature F9: Hermitian Green's Propagator Attention
- **Mathematical Formulation**:
  Models the electromagnetic Green's propagator $G(m, n) = \sum_l \alpha_l e^{-j 2\pi (m-n)\Delta f \tau_l}$.
  1. **Complex Linear Projections Without Bias**:
     $W_Q, W_K, W_V \in \mathbb{C}^{D_{\text{out}} \times D_{\text{in}}}$ with $b \equiv 0$:
     $$Q = X W_Q^T, \quad K = X W_K^T, \quad V = X W_V^T \implies W(c X) = c W(X)$$
  2. **Complex Relative-Phase RoPE**:
     Rotary frequencies $\theta_d = 10000^{-2d / D_h}$ for head dimension $d \in \{0, \dots, D_h-1\}$:
     $$\tilde{Q}_{m,d} = Q_{m,d} e^{j m \theta_d}, \quad \tilde{K}_{n,d} = K_{n,d} e^{j n \theta_d}$$
  3. **Hermitian Propagator Attention Matrix**:
     Complex inner product:
     $$\langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}} = \sum_{d=1}^{D_h} \tilde{Q}_{m,d} \tilde{K}_{n,d}^* = \sum_{d=1}^{D_h} Q_{m,d} K_{n,d}^* e^{j (m - n)\theta_d}$$
     Affinities:
     $$S_{m,n} = \frac{1}{\sqrt{D_h}} \text{Re}\left( \langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}} \right)$$
     Softmax: $A_{m,n} = \frac{\exp(S_{m,n})}{\sum_l \exp(S_{m,l})}$.
     Output: $O_m = \sum_n A_{m,n} V_n$.
  4. **Radial Activation**:
     $$f_{\text{radial}}(z) = \text{SiLU}(|z|) \cdot \frac{z}{|z| + 10^{-6}}$$
     Preserves exact $\mathrm{U}(1)$ phase angle: $f(e^{j\phi} z) = e^{j\phi} f(z)$.
- **Tensors**:
  - `Q, K, V`: `(B, n_heads, N, D_h)`, dtype `torch.complex64`.
  - `A`: `(B, n_heads, N, N)`, dtype `torch.float32`.
  - `O`: `(B, N, D)`, dtype `torch.complex64`.
- **Acceptance Criteria & Tolerances**:
  - Affinity matrix invariance: $\|A(e^{j\phi} X) - A(X)\|_\infty < 1.0 \times 10^{-6}$.
  - Bias check: All projection modules must have `bias is None` or `bias=False`.
  - Radial phase preservation: $|\arg(f(z)) - \arg(z)| < 10^{-6}$ for all $z \ne 0$.
- **Tier 2 Boundary Conditions**:
  - Co-located subcarriers $m=n$ ($\Delta k = 0$), maximum frequency separation $\Delta k = N_{\text{sc}}-1$, $z = 0$, $D_h \in \{16, 32\}$.
- **Tier 3 Interaction**:
  - F9 + F10 (Hermitian attention used within bidirectional cross-attention).
- **Tier 4 Scenario Context**:
  - Multipath propagation across 64 subcarriers with severe delay spread.

---

#### Feature F10: Bidirectional Pilot-Data Attention
- **Mathematical Formulation**:
  Under nonlinear PA distortion, inter-carrier distortion $D_{\text{nonlinear}}$ couples pilot and data subcarriers:
  - **Pilot $\to$ Data (Channel & Phase Propagation)**:
    Data stream queries the pilot stream to interpolate the continuous channel transfer function $H[k]$:
    $$X_d^{(l+1)} = \text{Propagate}\left(X_d^{(l)}, \, \text{CrossAttn}(X_d^{(l)}, H^{(l)})\right)$$
  - **Data $\to$ Pilot (Distortion & ICI Cancellation Feedback)**:
    Tentative data decisions predict nonlinear time-domain envelope $\hat{x}_{\text{time}}$ and feedback distortion to pilot stream:
    $$H^{(l+1)} = \text{Propagate}\left(H^{(l)}, \, \text{CrossAttn}(H^{(l)}, X_d^{(l+1)})\right)$$
  - Formulates a fully differentiable analog to Iterative Decision-Directed PA Cancellation.
- **Tensors**:
  - Pilot stream $H^{(l)} \in \mathbb{C}^{B \times N_p \times D}$.
  - Data stream $X_d^{(l)} \in \mathbb{C}^{B \times N_d \times D}$.
- **Acceptance Criteria & Tolerances**:
  - Gradient flow: Non-zero gradients backpropagate bidirectionally: $\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial H^{(0)}} \ne 0$ and $\frac{\partial \mathcal{L}_{\text{NMSE}}}{\partial X_d^{(0)}} \ne 0$.
  - Distortion rejection: Pilot channel error in presence of $A=1.25$ data is reduced by $\ge 1.0$ dB compared to decoupled pilot-only estimation.
- **Tier 2 Boundary Conditions**:
  - Ratio of pilots to data: $1:1$ (equal), $1:3$ (sparse), $1:7$ (ultrasparse).
- **Tier 3 Interaction**:
  - F10 + F11 (Bidirectional attention trained under joint consistency loss).
- **Tier 4 Scenario Context**:
  - 14-symbol slot with $7.14\%$ sparse DMRS.

---

#### Feature F11: Joint Physical Consistency Loss
- **Mathematical Formulation**:
  $$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{NMSE}} \mathcal{L}_{\text{NMSE}} + \lambda_{\text{PA}} \mathcal{L}_{\text{PA-res}} + \lambda_{\text{delay}} \mathcal{L}_{\text{delay}}$$
  Nominal weights: $\lambda_{\text{NMSE}} = 1.0, \lambda_{\text{PA}} = 0.25, \lambda_{\text{delay}} = 0.10$.
  1. **Calibrated LLR BCE**:
     $$\mathcal{L}_{\text{BCE}} = -\frac{1}{4 N_d} \sum_{k=1}^{N_d} \sum_{i=0}^3 \left[ b_{k,i} \ln \sigma(L_{k,i}) + (1 - b_{k,i}) \ln (1 - \sigma(L_{k,i})) \right]$$
  2. **Normalized Channel NMSE (Scale-Invariant)**:
     $$\mathcal{L}_{\text{NMSE}} = \frac{\sum_{k=1}^N |H_{\text{true}}[k] - \hat{H}[k]|^2}{\sum_{k=1}^N |H_{\text{true}}[k]|^2 + \epsilon_H}$$
  3. **PA Residual Consistency**:
     Reconstruct soft transmitted symbols: $\hat{X}_d[k] = \sum_{s \in \mathcal{S}_{16}} s \cdot \mathbb{P}(X_k = s | L_k)$.
     Forward PA synthesis: $\hat{X}_{\text{PA}} = \text{FFT}(f_{\text{Rapp}}(\text{IFFT}(\hat{X}_d)))$.
     $$\mathcal{L}_{\text{PA-res}} = \frac{\|Y_d - \hat{H} \odot \hat{X}_{\text{PA}}\|_2^2}{\|Y_d\|_2^2 + \epsilon_Y}$$
  4. **Delay-Domain Prior**:
     $$\hat{h}_{\text{time}} = \text{IFFT}(\hat{H}), \quad \mathcal{L}_{\text{delay}} = \frac{\sum_{k = N_{\text{CP}}}^{N-1} |\hat{h}_{\text{time}}[k]|^2}{\sum_{k=0}^{N-1} |\hat{h}_{\text{time}}[k]|^2 + \epsilon_h}$$
- **Tensors**:
  - Inputs: `L`: `(B, N_d, 4)`, `b`: `(B, N_d, 4)`, `H_hat`: `(B, N)`, `H_true`: `(B, N)`, `Yd`: `(B, N_d)`.
  - Output: `loss`: scalar `torch.float32`.
- **Acceptance Criteria & Tolerances**:
  - Scale invariance: For $Y \mapsto c Y$, $|\mathcal{L}_{\text{joint}}(c Y) - \mathcal{L}_{\text{joint}}(Y)| < 1.0 \times 10^{-5}$.
  - Delay loss sanity: For ideal channel with taps inside CP, $\mathcal{L}_{\text{delay}} < 10^{-6}$.
- **Tier 2 Boundary Conditions**:
  - Perfect channel estimate ($\mathcal{L}_{\text{NMSE}} \to 0$); completely random LLRs ($\mathcal{L}_{\text{BCE}} \approx \ln 2 \approx 0.6931$).
- **Tier 3 Interaction**:
  - F11 + F15 (Joint loss trained model evaluated under outer channel coding).
- **Tier 4 Scenario Context**:
  - Full end-to-end gradient updates during training.

---

#### Feature F12: Numerical Equivariance Verification
- **Mathematical Formulation**:
  Formal test vector transformation:
  $$c = r \cdot e^{j\phi}, \quad r \in [2^{-12}, 2^{12}], \quad \phi \in [-\pi, \pi]$$
  $$(Y_p', Y_d', {\sigma'}^2) = (c Y_p, c Y_d, r^2 \sigma^2)$$
  Assertions:
  $$\epsilon_H = \frac{\|\hat{H}' - c \hat{H}\|_\infty}{|c| (\|\hat{H}\|_\infty + 10^{-12})} < 10^{-5}$$
  $$\epsilon_L = \|L' - L\|_\infty < 10^{-5}$$
  $$N_{\text{flip}} = \sum |(L' > 0) \ne (L > 0)| \equiv 0$$
- **Tensors**:
  - Batch size $B=64$, randomly sampled gains and phases.
- **Acceptance Criteria & Tolerances**:
  - 100% pass across all 4 channel profiles (`train3`, `equal4`, `long6`, `sparse12`).
  - Bit flip count $N_{\text{flip}}$ must be **strictly 0**.
- **Tier 2 Boundary Conditions**:
  - Corner scales $r = 2^{-12}, 2^{-6}, 2^0, 2^6, 2^{12}$; phase angles $\phi \in \{0, \pi/4, \pi/2, \pi, -\pi/2\}$.
- **Tier 3 Interaction**:
  - F12 + F19 (Equivariance verification under mixed precision BF16 and FP16).
- **Tier 4 Scenario Context**:
  - Continuous runtime assertion hook during E2E verification.

---

### Milestone 3: 2D Time-Frequency Generalization & Coded Transmission (F13–F17)

#### Feature F13: 2D Slot Simulator & Sparse DMRS
- **Mathematical Formulation**:
  Multi-symbol slot grid: $N_{\text{symb}} = 14$ consecutive OFDM symbols, $N_{\text{sc}} = 32$ or $64$ subcarriers.
  Total Resource Elements: $M_{\text{slot}} = N_{\text{symb}} \times N_{\text{sc}} = 448$ (or $896$).
  - **DMRS Pattern**:
    - DMRS symbols: $s \in \{2, 11\}$ (for high Doppler tracking).
    - Frequency comb-2: $k \in \{0, 2, 4, \dots, N_{\text{sc}}-2\}$.
    - Total pilot REs: $2 \times (N_{\text{sc}} / 2) = N_{\text{sc}}$ REs.
    - Pilot overhead: $N_{\text{sc}} / (14 N_{\text{sc}}) = 1/14 \approx 7.14\% \le 10\%$.
  - **Pilot Sequences**: QPSK Gold sequence or pseudo-random symbols with unit average energy.
- **Tensors**:
  - `grid`: `(B, 14, N_sc)`, dtype `torch.complex64`.
  - `dmrs_mask`: `(14, N_sc)`, dtype `torch.bool`.
- **Acceptance Criteria & Tolerances**:
  - Pilot overhead: $\frac{\sum \text{dmrs\_mask}}{14 \times N_{\text{sc}}} \le 0.10$.
  - Correct masking: Payload data mapped only to REs where `dmrs_mask == False`.
- **Tier 2 Boundary Conditions**:
  - Single-symbol DMRS at $s=2$ ($3.57\%$ overhead) vs dual-symbol DMRS at $s \in \{2, 11\}$ ($7.14\%$ overhead); $N_{\text{sc}} \in \{32, 64\}$.
- **Tier 3 Interaction**:
  - F13 + F16 (Sparse DMRS interpolation via 2D-LMMSE).
- **Tier 4 Scenario Context**:
  - Standard 5G NR slot configuration with 14 symbols.

---

#### Feature F14: Continuous Doppler ICI & Carrier Frequency Offset (CFO)
- **Mathematical Formulation**:
  1. **Continuous Doppler ICI Matrix**:
     For symbol $s$, time sample $n \in \{0, \dots, N_{\text{sc}}-1\}$:
     $$h_l[t_s(n)] = h_{l, 0} \cdot e^{j 2\pi f_d \cos(\theta_l) (s(N_{\text{sc}}+N_{\text{cp}}) + N_{\text{cp}} + n) T_s / N_{\text{sc}}}$$
     Transfer matrix across subcarriers:
     $$H_s[k, k'] = \frac{1}{N_{\text{sc}}} \sum_{n=0}^{N_{\text{sc}}-1} \left( \sum_{l=0}^{L-1} h_l[t_s(n)] e^{-j 2\pi k' d_l / N_{\text{sc}}} \right) e^{-j 2\pi (k - k') n / N_{\text{sc}}}$$
  2. **Carrier Frequency Offset (CFO)**:
     Normalized CFO $\epsilon = \Delta f_{\text{cfo}} / \Delta f_{\text{sub}}$. Sample-level phase rotation:
     $$y_s[n] \mapsto y_s[n] \cdot \exp\left( j 2\pi \epsilon \frac{s(N_{\text{sc}} + N_{\text{cp}}) + N_{\text{cp}} + n}{N_{\text{sc}}} \right)$$
     Produces both intra-symbol attenuation $\text{sinc}(\epsilon)$, adjacent ICI leakage, and inter-symbol phase drift $\Delta \phi_{\text{sym}} = 2\pi \epsilon \frac{N_{\text{sc}}+N_{\text{cp}}}{N_{\text{sc}}}$.
- **Tensors**:
  - Inputs: `grid_tx`: `(B, 14, N_sc)`, `fd_Tsym`: `float`, `cfo_eps`: `float`.
  - Outputs: `grid_rx`: `(B, 14, N_sc)`.
- **Acceptance Criteria & Tolerances**:
  - In linear limit ($\epsilon=0, f_d=0$): $H_s[k, k']$ is strictly diagonal ($H_s[k, k'] = 0$ for $k \ne k'$).
  - High Doppler ICI: For $f_d T_{\text{sym}} = 0.03$, total off-diagonal power $\sum_{k \ne k'} |H_s[k, k']|^2 \ge -20$ dB relative to main diagonal.
  - Phase roll: Symbol-to-symbol phase roll across DMRS symbols matches $2\pi \epsilon \frac{9(N_{\text{sc}}+N_{\text{cp}})}{N_{\text{sc}}} \pm 10^{-4}$.
- **Tier 2 Boundary Conditions**:
  - $\epsilon \in \{0.0, 0.01, 0.05, 0.10\}$; $f_d T_{\text{sym}} \in \{0.0, 0.01, 0.03, 0.05\}$.
- **Tier 3 Interaction**:
  - F14 + F8 ($C^*$ gauge normalization under high Doppler and CFO).
- **Tier 4 Scenario Context**:
  - High-speed train channel ($v=300$ km/h, $f_d=750$ Hz) with 5% CFO.

---

#### Feature F15: Native Outer Channel Coding (Rate-1/2 FEC)
- **Mathematical Formulation**:
  Native PyTorch implementation of Rate-1/2 FEC with soft decoding:
  - **Option A: 5G NR LDPC (Base Graph 2)**:
    - Parity check matrix $H_{\text{ldpc}} \in \{0, 1\}^{M \times N_{\text{code}}}$, rate $R = 1/2$.
    - Codeword length $N_{\text{code}} = 384$ bits, information payload $K_{\text{code}} = 192$ bits.
    - Soft Min-Sum Algorithm (NMSA with scale $\alpha_{\text{ms}} = 0.75$):
      $$L_{q \to c}^{(i)} = L_q^{(0)} + \sum_{c' \in \mathcal{M}(q) \setminus \{c\}} L_{c' \to q}^{(i-1)}$$
      $$L_{c \to q}^{(i)} = \alpha_{\text{ms}} \cdot \left( \prod_{q' \in \mathcal{N}(c) \setminus \{q\}} \text{sign}(L_{q' \to c}^{(i)}) \right) \cdot \min_{q' \in \mathcal{N}(c) \setminus \{q\}} |L_{q' \to c}^{(i)}|$$
    - Early stopping: Parity condition $H_{\text{ldpc}} \hat{c}^T \equiv 0 \pmod 2$.
  - **Option B: Rate-1/2 Convolutional Code ($K=7$, generators $[133, 171]_8$)**:
    - Trellis-based Max-Log-MAP soft decoding.
- **Tensors**:
  - Inputs: `info_bits`: `(B, K_code)`, `llrs`: `(B, N_code)`.
  - Outputs: `coded_bits`: `(B, N_code)`, `decoded_bits`: `(B, K_code)`, `parity_satisfied`: `(B,)` `bool`.
- **Acceptance Criteria & Tolerances**:
  - Parity check: Valid codewords must satisfy $H \cdot c^T = 0 \pmod 2$.
  - Waterfall gain: Over AWGN at rate 1/2, soft decoding delivers $\ge 4.5$ dB coding gain over uncoded transmission at $\text{BER} = 10^{-3}$.
- **Tier 2 Boundary Conditions**:
  - Codeword length $N_{\text{code}} \in \{192, 384, 768\}$; all-zero bits; all-one bits; iterations $I_{\text{max}} \in \{1, 5, 10, 15\}$.
- **Tier 3 Interaction**:
  - F15 + F5 (Calibrated LLRs fed to soft LDPC decoder).
- **Tier 4 Scenario Context**:
  - Outer channel decoding across a complete 14-symbol slot payload.

---

#### Feature F16: 2D Classical Baselines (2D-LMMSE & 2D IDD-PA)
- **Mathematical Formulation**:
  1. **Separable 2D-LMMSE**:
     Time-frequency correlation: $R_{\text{tf}} = R_{\text{time}} \otimes R_{\text{freq}}$, with:
     $$R_{\text{time}}[s, s'] = J_0(2\pi f_d T_{\text{sym}} (s - s')), \quad R_{\text{freq}}[k, k'] = \sum_l p_l e^{-j 2\pi (k - k') d_l / N_{\text{sc}}}$$
     Linear MMSE 2D interpolation from DMRS locations $\mathcal{P}$:
     $$\hat{H}_{\text{grid}} = R_{\text{all}, \mathcal{P}} \left( R_{\mathcal{P}, \mathcal{P}} + \sigma_w^2 I \right)^{-1} Y_{\mathcal{P}}$$
  2. **2D Iterative Decision-Directed PA Cancellation (2D IDD-PA)**:
     - Equalize grid using $\hat{H}_{\text{grid}}^{(0)}$.
     - Tentatively slice 16-QAM symbols across all payload REs.
     - Per-symbol time-domain reconstruction: $\hat{x}_s = \text{IFFT}(\hat{S}_s)$.
     - Reconstruct PA distortion $\hat{d}_{s} = f_{\text{PA}}(\hat{x}_s) - \gamma \hat{x}_s$.
     - Subtract $\hat{H}_{s, k} \hat{D}_{s, k}$ from received grid and re-estimate channel and symbols.
- **Tensors**:
  - Input: `grid_rx`: `(B, 14, N_sc)`, `dmrs_mask`: `(14, N_sc)`.
  - Outputs: `H_est_2d`: `(B, 14, N_sc)`, `llrs_2d`: `(B, N_data_res, 4)`.
- **Acceptance Criteria & Tolerances**:
  - Interpolation NMSE: 2D-LMMSE channel estimation NMSE $\le -15$ dB at 20 dB SNR ($f_d T_{\text{sym}} \le 0.02$).
  - 2D IDD-PA gain: 2D IDD-PA reduces uncoded BER by $\ge 15\%$ relative to 2D-LMMSE at $A=1.25$.
- **Tier 2 Boundary Conditions**:
  - Single DMRS symbol vs 2 DMRS symbols; Doppler $f_d T_{\text{sym}} \in \{0.0, 0.01, 0.05\}$.
- **Tier 3 Interaction**:
  - F16 + F17 (2D baselines benchmarked against EM-FloatFormer for coded BLER).
- **Tier 4 Scenario Context**:
  - Classical reference receiver in high-mobility scenario.

---

#### Feature F17: Coded Block Error Rate (BLER) Evaluation
- **Mathematical Formulation**:
  A block error occurs if any decoded bit in information block $b$ differs from ground truth:
  $$\text{BlockError}_b = \mathbb{I}\left( \hat{u}_b \ne u_b \right)$$
  $$\text{BLER} = \frac{1}{B_{\text{total}}} \sum_{b=1}^{B_{\text{total}}} \text{BlockError}_b$$
  Evaluated across SNR grid $\text{SNR} \in \{0, 5, 10, 15, 20\}$ dB under:
  - In-distribution channel conditions ($f_d = 50$ Hz, `train3`).
  - Out-of-Distribution (OOD) Doppler ($f_d = 300, 600$ Hz).
  - Out-of-Distribution (OOD) Delay profiles (`long12`, `sparse12`).
- **Tensors**:
  - Input: `u_true`: `(B, K)`, `u_dec`: `(B, K)`.
  - Output: `bler`: `float`, `n_block_errors`: `int`, `n_total_blocks`: `int`.
- **Acceptance Criteria & Tolerances**:
  - Monotonicity: $\text{BLER}(\text{SNR}_2) \le \text{BLER}(\text{SNR}_1)$ for $\text{SNR}_2 > \text{SNR}_1$.
  - Target: EM-FloatFormer achieves lower BLER than 2D-LMMSE and 2D IDD-PA across OOD Doppler conditions.
- **Tier 2 Boundary Conditions**:
  - Zero SNR (BLER $\approx 1.0$), High SNR 25 dB (BLER $< 10^{-3}$), extreme OOD Doppler.
- **Tier 3 Interaction**:
  - F17 + F20 (BLER evaluated across 10 pre-frozen confirmatory seeds).
- **Tier 4 Scenario Context**:
  - Primary performance metric for Stage P4-B.

---

### Milestone 4: Mechanism Probing, Precision Auditing & Governance (F18–F22)

#### Feature F18: Factorial Ablation Matrix ($2^3$ Orthogonal Design)
- **Mathematical Formulation**:
  Isolates three orthogonal architectural mechanisms:
  - Factor 1 ($F_1$): Exact $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ Gauge Invariance (RMS anchor + complex RMSNorm).
  - Factor 2 ($F_2$): Complex Hermitian Green's Propagator Attention (Complex linear, RoPE, radial activation).
  - Factor 3 ($F_3$): Bidirectional Joint Pilot-Data Propagation.
  Full $2^3 = 8$ model arms:
  $$\text{Arm } 1 (+F_1, +F_2, +F_3), \quad \text{Arm } 2 (-F_1, +F_2, +F_3), \quad \dots, \quad \text{Arm } 8 (-F_1, -F_2, -F_3)$$
  All 8 models are parameter-matched ($\approx 25\text{k} - 35\text{k}$ params).
  Causal Main Effects via linear contrast:
  $$\text{ME}(F_k) = \frac{1}{4} \sum_{i: F_k = +} \text{BER}(\text{Arm}_i) - \frac{1}{4} \sum_{j: F_k = -} \text{BER}(\text{Arm}_j)$$
- **Tensors**:
  - Model factory: `create_floatformer(use_gauge_norm=bool, use_hermitian_attn=bool, use_joint_prop=bool)`.
- **Acceptance Criteria & Tolerances**:
  - Parameter matching: All 8 models must have parameter counts within $\pm 10\%$ of each other.
  - Causal significance: $\text{ME}(F_1) < 0$, $\text{ME}(F_2) < 0$, $\text{ME}(F_3) < 0$ (all 3 mechanisms provide statistically significant marginal improvements).
- **Tier 2 Boundary Conditions**:
  - Test all 8 arms at normal gain ($r = 1.0$) and extreme gain ($r = 2^{-12}$).
- **Tier 3 Interaction**:
  - F18 + F19 (Factorial matrix tested across FP32, BF16, and FP16).
- **Tier 4 Scenario Context**:
  - Causal mechanistic validation for Stage P4-C.

---

#### Feature F19: Numeric Stability & Mixed-Precision Audit
- **Mathematical Formulation**:
  Audits execution across numeric representations:
  1. `fp32`: Standard IEEE 754 single precision.
  2. `bf16`: BFloat16 mixed precision (`torch.autocast(device_type="cpu", dtype=torch.bfloat16)`).
  3. `fp16`: Float16 mixed precision (`torch.autocast(device_type="cpu", dtype=torch.float16)`).
  Evaluated across 24 octaves of coherent channel gain scaling:
  $$r \in \{ 2^{-12}, 2^{-8}, 2^{-4}, 2^0, 2^4, 2^8, 2^{12} \} \quad (72 \text{ dB})$$
  Metrics:
  $$N_{\text{nan}} = \sum \mathbb{I}(\text{isnan}(L) \lor \text{isinf}(L)) \equiv 0$$
  $$\Delta L_{\text{max}} = \max |L_{\text{half}} - L_{\text{fp32}}| < 0.15$$
  $$\text{FlipRate} = \frac{\sum |(L_{\text{half}} > 0) \ne (L_{\text{fp32}} > 0)|}{N_{\text{total\_bits}}} < 0.05\%$$
- **Tensors**:
  - Input: standard batches scaled by $r$.
  - Outputs: LLRs in FP32, BF16, FP16.
- **Acceptance Criteria & Tolerances**:
  - Zero NaNs / Infs across all scales and dtypes: $N_{\text{nan}} \equiv 0$.
  - Dynamic range robustness: No collapse at $r = 2^{\pm 12}$ (unlike Stage P3-C).
- **Tier 2 Boundary Conditions**:
  - Dynamic range boundaries $r = 2^{-12}$ and $r = 2^{12}$; half-precision minimum subnormals.
- **Tier 3 Interaction**:
  - F19 + F15 (Mixed precision audit on outer FEC soft decoding).
- **Tier 4 Scenario Context**:
  - Hardware deployment efficiency audit.

---

#### Feature F20: Pre-Frozen Confirmatory Protocol
- **Mathematical Formulation**:
  1. Pre-declared protocol committed in `P4D_PROTOCOL.md` and hashed to `P4D_FREEZE.json`.
  2. Exactly 10 fresh, unpolluted seeds:
     $$\mathcal{S}_{\text{P4D}} = \{ 74001, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010 \}$$
     Disjoint from all historical runs (P1: 101–808; P2: 1201–2210; P3: 46001–48105; probes: 59101–61010).
  3. Exact integer accounting:
     $$E_{\text{bits}} \in \mathbb{Z}_{\ge 0}, \quad N_{\text{bits}} \in \mathbb{Z}_{> 0}, \quad E_{\text{blocks}} \in \mathbb{Z}_{\ge 0}, \quad N_{\text{blocks}} \in \mathbb{Z}_{> 0}$$
     No floating-point rounding of raw error tallies.
- **Tensors**:
  - Output CSV: `raw_by_seed.csv` with integer error tallies.
- **Acceptance Criteria & Tolerances**:
  - Seed set equality: $\text{Set}(\text{executed\_seeds}) \equiv \mathcal{S}_{\text{P4D}}$.
  - Protocol hash: SHA256 of `P4D_PROTOCOL.md` matches `P4D_FREEZE.json`.
- **Tier 2 Boundary Conditions**:
  - Protocol execution with partial or interrupted seeds (must fail cleanly and resume).
- **Tier 3 Interaction**:
  - F20 + F21 (Confirmatory seed error tables fed directly to sign-flip permutation tests).
- **Tier 4 Scenario Context**:
  - Final confirmatory replication run for Stage P4-D.

---

#### Feature F21: Two-Sided Sign-Flip Permutation Tests with Holm Adjustment
- **Mathematical Formulation**:
  Paired performance contrast across 10 confirmatory seeds $s \in \mathcal{S}_{\text{P4D}}$:
  $$\delta_s = \text{BLER}(\text{Baseline})_s - \text{BLER}(\text{EM-FloatFormer})_s$$
  Test statistic: $T_{\text{obs}} = \left| \sum_{s=1}^{10} \delta_s \right|$.
  Under null hypothesis $H_0: \delta_s \overset{d}{=} -\delta_s$, there are $2^{10} = 1024$ equally likely sign configurations $\mathbf{s} \in \{-1, +1\}^{10}$.
  Exact two-sided $p$-value:
  $$p_{\text{exact}} = \frac{1}{1024} \sum_{\mathbf{s} \in \{-1, +1\}^{10}} \mathbb{I}\left( \left| \sum_{s=1}^{10} s_i \delta_s \right| \ge T_{\text{obs}} - 10^{-10} \right)$$
  Holm-Bonferroni step-down adjustment across $M$ endpoints:
  $$p_{(k)}^{\text{Holm}} = \max_{j \le k} \min\left(1, (M - j + 1) p_{(j)}\right)$$
  Statistical significance criterion: $p_{(k)}^{\text{Holm}} < 0.05$.
- **Tensors**:
  - Input: $\boldsymbol{\delta} \in \mathbb{R}^{10}$.
  - Outputs: $T_{\text{obs}} \in \mathbb{R}_{\ge 0}$, $p_{\text{exact}} \in [0.0009765625, 1.0]$, $p_{\text{Holm}} \in [0.0, 1.0]$.
- **Acceptance Criteria & Tolerances**:
  - Exact state space: Evaluates all 1024 permutations without Monte Carlo approximation.
  - Discrete granularity: Minimum non-zero $p$-value is exactly $2 / 1024 \approx 0.001953125$.
- **Tier 2 Boundary Conditions**:
  - All $\delta_s \equiv 0$ ($p = 1.0$); all $\delta_s > 0$ with equal magnitude ($p = 2/1024$); one huge outlier dominating sum.
- **Tier 3 Interaction**:
  - F21 + F17 (Permutation testing on coded BLER endpoints).
- **Tier 4 Scenario Context**:
  - Confirmatory statistical hypothesis testing.

---

#### Feature F22: Repository Governance & Immutability Verification
- **Mathematical Formulation**:
  Pre-execution and post-execution cryptographic audit:
  $$\forall \text{file } f \in \text{HistoricalManifest}, \quad \text{SHA256}(f) \equiv \text{ManifestHash}(f)$$
  Ensures that historical stages (P1–P3), exploratory files, and manifests are completely untouched.
  Stage P4 deliverables are strictly restricted to:
  - Source package: `p4/`
  - Test suites: `tests/test_p4_e2e.py`, `p4/tests/`
  - Results: `p4/results/`, `p4b_results/`, `p4c_results/`, `p4d_results/`
  - Protocols & Reports: `P4*_PROTOCOL.md`, `P4*_REPORT.md`
- **Tensors / CLI**:
  - Command: `python -m p4.tests.test_p4_c_d_governance` or audit script.
  - Output: Exit code 0 on success; non-zero on hash mismatch.
- **Acceptance Criteria & Tolerances**:
  - Bitwise equality: 0 modified historical files, 0 deleted historical files.
- **Tier 2 Boundary Conditions**:
  - Tampering test: Modifying a single character in `p3b_time_link.py` must immediately trigger an audit failure.
- **Tier 3 Interaction**:
  - F22 + All (Governance audit run before and after the entire test suite).
- **Tier 4 Scenario Context**:
  - Final milestone sign-off gate.

---

## 4-Tier Test Suite Specification Matrix

The E2E test suite in `tests/test_p4_e2e.py` implements the full 4-tier testing hierarchy:

```text
tests/test_p4_e2e.py
├── TestTier1FeatureCoverage (22 features * 5 tests >= 110 tests)
│   ├── test_tier1_f01_rapp_pa_isolated_1..5
│   ├── test_tier1_f02_dynamic_multipath_isolated_1..5
│   ├── ...
│   └── test_tier1_f22_governance_isolated_1..5
├── TestTier2BoundaryStress (22 features * 5 tests >= 110 tests)
│   ├── test_tier2_f01_rapp_pa_extremes_1..5
│   ├── test_tier2_f02_multipath_corner_cases_1..5
│   ├── ...
│   └── test_tier2_f22_tamper_detection_1..5
├── TestTier3CrossFeatureInteractions (>= 22 pairwise interaction tests)
│   ├── test_tier3_01_rapp_pa_with_doppler_ici (F1 + F2)
│   ├── test_tier3_02_bussgang_with_soft_llr (F4 + F5)
│   ├── test_tier3_03_idd_pa_with_bussgang_lmmse (F6 + F4)
│   ├── test_tier3_04_cstar_gauge_with_hermitian_attention (F8 + F9)
│   ├── test_tier3_05_hermitian_attention_with_bidirectional_coupling (F9 + F10)
│   ├── test_tier3_06_joint_loss_with_gauge_norm (F11 + F8)
│   ├── test_tier3_07_equivariance_with_bidirectional_model (F12 + F10)
│   ├── test_tier3_08_2d_slot_with_continuous_doppler (F13 + F14)
│   ├── test_tier3_09_2d_slot_with_sparse_dmrs (F13 + F16)
│   ├── test_tier3_10_cfo_with_2d_lmmse_tracking (F14 + F16)
│   ├── test_tier3_11_soft_llr_with_outer_ldpc_decoder (F5 + F15)
│   ├── test_tier3_12_outer_ldpc_with_2d_idd_pa (F15 + F16)
│   ├── test_tier3_13_coded_bler_with_ood_doppler (F17 + F14)
│   ├── test_tier3_14_coded_bler_with_ood_delay (F17 + F2)
│   ├── test_tier3_15_factorial_matrix_under_extreme_gain (F18 + F8)
│   ├── test_tier3_16_numeric_stability_under_bf16_fp16 (F19 + F8)
│   ├── test_tier3_17_mixed_precision_with_hermitian_attention (F19 + F9)
│   ├── test_tier3_18_confirmatory_protocol_with_integer_accounting (F20 + F17)
│   ├── test_tier3_19_sign_flip_permutation_with_confirmatory_seeds (F21 + F20)
│   ├── test_tier3_20_holm_adjustment_with_multiple_endpoints (F21 + F17)
│   ├── test_tier3_21_governance_with_artifact_manifest (F22 + F20)
│   └── test_tier3_22_end_to_end_pipeline_cross_coupling (F1 + F8 + F15)
└── TestTier4RealWorldScenarios (>= 11 realistic full-pipeline communication scenarios)
    ├── test_tier4_scenario_01_vehicular_high_doppler
    ├── test_tier4_scenario_02_severe_pa_clipping
    ├── test_tier4_scenario_03_deep_fading_sparse_multipath
    ├── test_tier4_scenario_04_carrier_frequency_offset_drift
    ├── test_tier4_scenario_05_ultrasparse_dmrs_tracking
    ├── test_tier4_scenario_06_wide_dynamic_range_shadowing
    ├── test_tier4_scenario_07_cell_edge_low_snr
    ├── test_tier4_scenario_08_high_throughput_indoor
    ├── test_tier4_scenario_09_mixed_impairment_stress
    ├── test_tier4_scenario_10_full_slot_coded_ldpc_transmission
    └── test_tier4_scenario_11_confirmatory_10_seed_replication
```

### Quantitative Test Count Summary
- **Tier 1 (Feature Coverage)**: $22 \text{ features} \times 5 \text{ tests} = 110 \text{ test cases}$.
- **Tier 2 (Boundary & Corner Cases)**: $22 \text{ features} \times 5 \text{ tests} = 110 \text{ test cases}$.
- **Tier 3 (Pairwise Cross-Feature Interactions)**: $22 \text{ interaction test cases}$.
- **Tier 4 (Real-World Scenarios)**: $11 \text{ realistic workload scenarios}$.
- **Total Test Cases**: $110 + 110 + 22 + 11 = 253 \text{ test cases}$.

---

## Detailed Specifications for Tier 1: Isolated Feature Coverage (>=5 Tests Per Feature)

For each feature F1 to F22, the test suite implements 5 distinct isolated test cases verifying basic mechanics, interfaces, and expected transformations:

1. **F1 (Oversampled Rapp PA)**:
   - `test_tier1_f01_1_oversampling_interpolation`: Verifies frequency-domain zero-padding increases sampling rate by $L=4$ without energy loss.
   - `test_tier1_f01_2_am_am_compression`: Verifies magnitude compression $g(r)$ matches Rapp formula for $A=2.25$.
   - `test_tier1_f01_3_am_pm_phase_shift`: Verifies phase rotation $\Phi(r)$ matches formula and increases monotonically with $r$.
   - `test_tier1_f01_4_linear_passthrough`: Verifies that $A=\infty, \alpha=0$ produces identity output.
   - `test_tier1_f01_5_downsampling_decimation`: Verifies filtering and decimation returns tensor of original dimension $(B, N)$.
2. **F2 (Dynamic Multipath Simulator)**:
   - `test_tier1_f02_1_cp_insertion_removal`: Verifies cyclic prefix samples match end of symbol.
   - `test_tier1_f02_2_multipath_delay_profile`: Verifies channel impulse response power matches profile.
   - `test_tier1_f02_3_doppler_phase_evolution`: Verifies continuous phase rotation across sample index $n$.
   - `test_tier1_f02_4_time_domain_fir_causality`: Verifies output at sample $n$ depends only on $x[n - d_l]$ for $d_l \ge 0$.
   - `test_tier1_f02_5_awgn_addition`: Verifies added noise variance matches specified SNR.
3. **F3 (Static-Limit Parity Check)**:
   - `test_tier1_f03_1_zero_doppler_linear_pa`: Verifies static limit parity error $< 10^{-5}$ on `train3`.
   - `test_tier1_f03_2_equal4_profile_parity`: Verifies parity holds on 4-tap equal power profile.
   - `test_tier1_f03_3_long6_profile_parity`: Verifies parity holds on 6-tap profile.
   - `test_tier1_f03_4_sparse12_profile_parity`: Verifies parity holds on sparse 12-tap profile.
   - `test_tier1_f03_5_random_channel_seed_parity`: Verifies parity across multiple random realizations.
4. **F4 (Bussgang Decomposition & LMMSE)**:
   - `test_tier1_f04_1_gamma_computation_mild_pa`: Verifies $\gamma \approx 0.94$ for $A=2.25$.
   - `test_tier1_f04_2_gamma_computation_strong_pa`: Verifies $\gamma \approx 0.81$ for $A=1.25$.
   - `test_tier1_f04_3_distortion_variance_positivity`: Verifies $\sigma_D^2 > 0$ and scales inversely with $A$.
   - `test_tier1_f04_4_lmmse_filter_construction`: Verifies filter coefficients $W[k]$ incorporate $\gamma$ and $\sigma_D^2$.
   - `test_tier1_f04_5_equalization_mse`: Verifies equalized symbol MSE is lower than simple ZF.
5. **F5 (Calibrated 16-QAM Soft LLRs)**:
   - `test_tier1_f05_1_llr_shape_and_dtype`: Verifies output tensor shape $(B, N_d, 4)$ and float32 dtype.
   - `test_tier1_f05_2_sign_accuracy`: Verifies sign of LLR matches hard decision on known bits.
   - `test_tier1_f05_3_magnitude_boundary_b1_b3`: Verifies zero crossing of $b_1, b_3$ at $|z| = 2/\sqrt{10}$.
   - `test_tier1_f05_4_noise_scaling_inverse_variance`: Verifies LLR magnitude scales proportionally to $1/\sigma^2$.
   - `test_tier1_f05_5_calibration_probability`: Verifies $\sigma(LLR)$ produces valid probability in $[0, 1]$.
6. **F6 (Iterative DD-PA Cancellation)**:
   - `test_tier1_f06_1_initialization_iteration_0`: Verifies iteration 0 matches Bussgang-LMMSE.
   - `test_tier1_f06_2_distortion_reconstruction`: Verifies reconstructed distortion $\hat{d}$ is non-zero under $A=1.25$.
   - `test_tier1_f06_3_frequency_subtraction`: Verifies cleaned payload $Y_{\text{cleaned}} = Y_d - \hat{H}\hat{D}$.
   - `test_tier1_f06_4_two_iteration_execution`: Verifies 2-iteration loop executes cleanly.
   - `test_tier1_f06_5_three_iteration_execution`: Verifies 3-iteration loop executes cleanly.
7. **F7 (Classical Latency Benchmark)**:
   - `test_tier1_f07_1_cpu_thread_configuration`: Verifies `torch.get_num_threads()` equals 4.
   - `test_tier1_f07_2_batch_1_timing`: Verifies benchmark runs and outputs positive latency for $B=1$.
   - `test_tier1_f07_3_batch_64_timing`: Verifies benchmark runs and outputs positive latency for $B=64$.
   - `test_tier1_f07_4_csv_output_generation`: Verifies `latency_cpu.csv` is written with valid columns.
   - `test_tier1_f07_5_json_output_generation`: Verifies `timing.json` contains valid metadata.
8. **F8 (Exact $C^*$ Gauge Normalization)**:
   - `test_tier1_f08_1_pilot_rms_calculation`: Verifies $s_0$ matches true RMS of pilot subcarriers.
   - `test_tier1_f08_2_dimensionless_normalization`: Verifies $\bar{Y}_p$ has RMS exactly equal to 1.0.
   - `test_tier1_f08_3_noise_variance_rescaling`: Verifies $\bar{\sigma}^2 = \sigma^2 / s_0^2$ is invariant to scale.
   - `test_tier1_f08_4_c_rmsnorm_phase_preservation`: Verifies Complex RMSNorm preserves complex phase angles.
   - `test_tier1_f08_5_channel_output_rescaling`: Verifies output channel is multiplied back by $s_0$.
9. **F9 (Hermitian Green's Attention)**:
   - `test_tier1_f09_1_complex_linear_zero_bias`: Verifies projection layers have no additive bias.
   - `test_tier1_f09_2_complex_rope_rotation`: Verifies relative rotary phase modulation on Query and Key.
   - `test_tier1_f09_3_hermitian_inner_product`: Verifies real affinity score cancels global common phase.
   - `test_tier1_f09_4_softmax_probability_distribution`: Verifies attention weights sum to 1.0 along key axis.
   - `test_tier1_f09_5_radial_activation_homogeneity`: Verifies radial activation preserves phase and scales magnitude.
10. **F10 (Bidirectional Pilot-Data Attention)**:
    - `test_tier1_f10_1_pilot_to_data_cross_attention`: Verifies data queries attending to pilot keys.
    - `test_tier1_f10_2_data_to_pilot_cross_attention`: Verifies pilot queries attending to data keys.
    - `test_tier1_f10_3_dual_stream_state_update`: Verifies both streams update state across layers.
    - `test_tier1_f10_4_gradient_backprop_dual_path`: Verifies loss gradients propagate into both streams.
    - `test_tier1_f10_5_parameter_count_budget`: Verifies total parameter count $\le 16,000$.
11. **F11 (Joint Physical Consistency Loss)**:
    - `test_tier1_f11_1_bce_loss_computation`: Verifies multi-bit BCE against ground truth bits.
    - `test_tier1_f11_2_normalized_nmse_computation`: Verifies scale-invariant NMSE.
    - `test_tier1_f11_3_pa_residual_synthesis`: Verifies forward PA model reconstruction and residual loss.
    - `test_tier1_f11_4_delay_domain_prior`: Verifies delay energy penalty outside CP length.
    - `test_tier1_f11_5_joint_weighted_sum`: Verifies weighted sum with nominal hyperparameters.
12. **F12 (Numerical Equivariance Verification)**:
    - `test_tier1_f12_1_scale_invariance_channel_estimate`: Verifies $\epsilon_H < 10^{-5}$ across scales.
    - `test_tier1_f12_2_phase_invariance_bit_logits`: Verifies $\epsilon_L < 10^{-5}$ across random phases.
    - `test_tier1_f12_3_zero_hard_bit_flips`: Verifies $N_{\text{flip}} \equiv 0$ on random inputs.
    - `test_tier1_f12_4_attention_matrix_constancy`: Verifies $\|A' - A\|_\infty < 10^{-6}$.
    - `test_tier1_f12_5_all_channel_profiles`: Verifies equivariance on all 4 canonical profiles.
13. **F13 (2D Slot Simulator & Sparse DMRS)**:
    - `test_tier1_f13_1_grid_dimensions_14_symbols`: Verifies grid shape `(B, 14, N_sc)`.
    - `test_tier1_f13_2_sparse_dmrs_mask_pattern`: Verifies DMRS mask on symbols 2 and 11, comb-2.
    - `test_tier1_f13_3_pilot_overhead_bound`: Verifies pilot overhead $\le 10\%$.
    - `test_tier1_f13_4_payload_data_mapping`: Verifies data RE count equals $M_{\text{slot}} - N_{\text{dmrs}}$.
    - `test_tier1_f13_5_qpsk_pilot_symbols`: Verifies pilot constellation energy is 1.0.
14. **F14 (Continuous Doppler ICI & CFO)**:
    - `test_tier1_f14_1_doppler_ici_matrix_structure`: Verifies off-diagonal terms when $f_d > 0$.
    - `test_tier1_f14_2_continuous_time_fading`: Verifies channel changes smoothly across consecutive symbols.
    - `test_tier1_f14_3_cfo_sample_phase_drift`: Verifies phase accumulation across time samples.
    - `test_tier1_f14_4_cfo_inter_symbol_roll`: Verifies symbol-to-symbol phase roll equals theoretical angle.
    - `test_tier1_f14_5_high_speed_ici_energy`: Verifies ICI power increases quadratically with $f_d$.
15. **F15 (Native Outer Channel Coding)**:
    - `test_tier1_f15_1_rate_half_encoding`: Verifies codeword length is exactly $2 \times$ info length.
    - `test_tier1_f15_2_parity_check_matrix_validity`: Verifies $H c^T \equiv 0 \pmod 2$.
    - `test_tier1_f15_3_soft_llr_min_sum_decoding`: Verifies Min-Sum soft decoding recovers bits from noisy LLRs.
    - `test_tier1_f15_4_early_stopping_parity`: Verifies decoding terminates early when syndrome is zero.
    - `test_tier1_f15_5_awgn_waterfall_gain`: Verifies soft decoding outperforms hard decoding by $> 1.5$ dB.
16. **F16 (2D Classical Baselines)**:
    - `test_tier1_f16_1_separable_correlation_matrices`: Verifies $R_{\text{time}}$ and $R_{\text{freq}}$ structure.
    - `test_tier1_f16_2_2d_lmmse_interpolation`: Verifies channel estimation on non-pilot REs.
    - `test_tier1_f16_3_2d_lmmse_equalization`: Verifies equalized grid symbols.
    - `test_tier1_f16_4_2d_idd_pa_tentative_slicing`: Verifies 2D tentative 16-QAM symbol slicing.
    - `test_tier1_f16_5_2d_idd_pa_distortion_subtraction`: Verifies distortion reconstruction and subtraction in 2D.
17. **F17 (Coded BLER Evaluation)**:
    - `test_tier1_f17_1_block_error_counting`: Verifies block error flag triggers if single bit differs.
    - `test_tier1_f17_2_bler_calculation`: Verifies $\text{BLER} = E_{\text{blocks}} / N_{\text{blocks}}$.
    - `test_tier1_f17_3_snr_grid_evaluation`: Verifies BLER computation across SNR grid.
    - `test_tier1_f17_4_in_distribution_benchmark`: Verifies baseline vs model comparison on `train3`.
    - `test_tier1_f17_5_ood_doppler_benchmark`: Verifies baseline vs model comparison at $f_d = 300$ Hz.
18. **F18 (Factorial Ablation Matrix)**:
    - `test_tier1_f18_1_arm_1_full_configuration`: Verifies initialization of Arm 1 ($+F_1, +F_2, +F_3$).
    - `test_tier1_f18_2_arm_2_no_gauge_configuration`: Verifies initialization of Arm 2 ($-F_1, +F_2, +F_3$).
    - `test_tier1_f18_3_arm_3_no_green_configuration`: Verifies initialization of Arm 3 ($+F_1, -F_2, +F_3$).
    - `test_tier1_f18_4_arm_4_no_joint_configuration`: Verifies initialization of Arm 4 ($+F_1, +F_2, -F_3$).
    - `test_tier1_f18_5_main_effect_linear_contrast`: Verifies Yates contrast vector computation.
19. **F19 (Numeric Stability & Precision Audit)**:
    - `test_tier1_f19_1_fp32_baseline_forward`: Verifies clean FP32 execution without NaNs.
    - `test_tier1_f19_2_bf16_autocast_forward`: Verifies BF16 autocast execution.
    - `test_tier1_f19_3_fp16_autocast_forward`: Verifies FP16 autocast execution.
    - `test_tier1_f19_4_extreme_gain_forward`: Verifies forward pass at $r = 2^{-12}$ and $r = 2^{12}$.
    - `test_tier1_f19_5_logit_deviation_metric`: Verifies calculation of $\max |L_{\text{half}} - L_{\text{fp32}}|$.
20. **F20 (Pre-Frozen Confirmatory Protocol)**:
    - `test_tier1_f20_1_seed_list_completeness`: Verifies 10 pre-frozen seeds $\{74001, \dots, 74010\}$.
    - `test_tier1_f20_2_seed_disjointness`: Verifies seeds do not overlap with any historical stage.
    - `test_tier1_f20_3_integer_bit_error_tally`: Verifies $E_{\text{bits}}$ is exact integer.
    - `test_tier1_f20_4_integer_block_error_tally`: Verifies $E_{\text{blocks}}$ is exact integer.
    - `test_tier1_f20_5_raw_csv_structure`: Verifies `raw_by_seed.csv` columns and format.
21. **F21 (Sign-Flip Permutation Tests)**:
    - `test_tier1_f21_1_paired_delta_vector`: Verifies computation of paired differences $\delta_s$.
    - `test_tier1_f21_2_1024_state_enumeration`: Verifies complete traversal of $2^{10}$ sign patterns.
    - `test_tier1_f21_3_test_statistic_calculation`: Verifies $T_{\text{obs}} = |\sum \delta_s|$.
    - `test_tier1_f21_4_exact_p_value_formula`: Verifies $p$-value computation.
    - `test_tier1_f21_5_holm_step_down_adjustment`: Verifies Holm-Bonferroni monotonic step-down adjustment.
22. **F22 (Repository Governance & Immutability)**:
    - `test_tier1_f22_1_manifest_loading`: Verifies `MANIFEST_SHA256.json` loads correctly.
    - `test_tier1_f22_2_p1_files_sha256`: Verifies hashes of all `p1*.py` files.
    - `test_tier1_f22_3_p2_files_sha256`: Verifies hashes of all `p2*.py` files.
    - `test_tier1_f22_4_p3_files_sha256`: Verifies hashes of all `p3*.py` files.
    - `test_tier1_f22_5_exploratory_files_sha256`: Verifies hashes of exploratory scripts.

---

## Detailed Specifications for Tier 2: Boundary & Corner Cases (>=5 Tests Per Feature)

1. **F1 (Rapp PA)**:
   - `test_tier2_f01_1_zero_amplitude`: $r=0 \implies g(0)=0, \Phi(0)=0$.
   - `test_tier2_f01_2_extreme_saturation`: $r = 10^5 \implies g(r) \to A$.
   - `test_tier2_f01_3_severe_clipping_knee_1_25`: Verify peak output $\le 1.25$.
   - `test_tier2_f01_4_mild_clipping_knee_2_25`: Verify peak output $\le 2.25$.
   - `test_tier2_f01_5_infinite_saturation`: $A = 10^{12} \implies y = x$.
2. **F2 (Dynamic Multipath Simulator)**:
   - `test_tier2_f02_1_zero_doppler`: $f_d = 0 \implies$ time-invariant taps.
   - `test_tier2_f02_2_maximum_doppler_limit`: $f_d T_{\text{sym}} = 0.05 \implies$ severe ICI.
   - `test_tier2_f02_3_single_tap_flat_fading`: $L_{\text{taps}} = 1 \implies$ no frequency selectivity.
   - `test_tier2_f02_4_maximum_delay_spread`: Tap at delay $N_{\text{CP}}-1$.
   - `test_tier2_f02_5_infinite_noise_snr_minus_30db`: $\sigma_w^2 \gg P_{\text{sig}}$.
3. **F3 (Static-Limit Parity Check)**:
   - `test_tier2_f03_1_zero_noise_parity`: $W = 0 \implies \epsilon < 10^{-6}$.
   - `test_tier2_f03_2_extreme_high_snr_50db`: Parity error dominates noise floor.
   - `test_tier2_f03_3_single_active_subcarrier`: Only 1 tone active.
   - `test_tier2_f03_4_all_subcarriers_coherent`: In-phase subcarrier alignment.
   - `test_tier2_f03_5_different_fft_sizes`: Test $N=32$ and $N=64$.
4. **F4 (Bussgang Decomposition & LMMSE)**:
   - `test_tier2_f04_1_zero_noise_distortion_limit`: $\sigma_W^2 = 0 \implies$ filter bounded by $\sigma_D^2$.
   - `test_tier2_f04_2_infinite_noise_limit`: $\sigma_W^2 \to \infty \implies W[k] \to 0$.
   - `test_tier2_f04_3_deep_channel_fade`: $|H[k]| < 10^{-4} \implies W[k] \to 0$ without overflow.
   - `test_tier2_f04_4_extreme_pa_knee_0_75`: Highly compressed PA.
   - `test_tier2_f04_5_perfect_linear_pa`: $A = \infty \implies \gamma = 1.0, \sigma_D^2 = 0$.
5. **F5 (Calibrated 16-QAM Soft LLRs)**:
   - `test_tier2_f05_1_zero_noise_infinite_llrs`: $\sigma^2 \to 0 \implies$ clamp to $\pm 50.0$.
   - `test_tier2_f05_2_infinite_noise_zero_llrs`: $\sigma^2 \to \infty \implies L \to 0$.
   - `test_tier2_f05_3_exact_constellation_centroid_hit`: $\hat{z} = (1+j)/\sqrt{10}$.
   - `test_tier2_f05_4_decision_boundary_re_exact_2_over_sqrt10`: $L(b_1) \equiv 0.0$.
   - `test_tier2_f05_5_extreme_outlier_observation`: $|\hat{z}| = 100.0 \implies$ smooth saturation.
6. **F6 (Iterative DD-PA Cancellation)**:
   - `test_tier2_f06_1_linear_pa_zero_residual`: $A = \infty \implies \hat{D} \equiv 0$.
   - `test_tier2_f06_2_all_decisions_wrong_iteration_stability`: No numeric divergence.
   - `test_tier2_f06_3_single_iteration_mode`: $K=1$.
   - `test_tier2_f06_4_ten_iteration_stress`: $K=10$ stable convergence.
   - `test_tier2_f06_5_zero_snr_cancellation`: Cancellation remains bounded under heavy noise.
7. **F7 (Classical Latency Benchmark)**:
   - `test_tier2_f07_1_single_thread_stress`: `torch.set_num_threads(1)`.
   - `test_tier2_f07_2_max_thread_stress`: Maximum hardware threads.
   - `test_tier2_f07_3_single_sample_batch_1`: $B=1$ minimal latency.
   - `test_tier2_f07_4_large_batch_256`: $B=256$ throughput scaling.
   - `test_tier2_f07_5_consecutive_runs_drift`: Drift between runs $< 5\%$.
8. **F8 (Exact $C^*$ Gauge Normalization)**:
   - `test_tier2_f08_1_minimum_scale_2_minus_12`: $r = 2^{-12} \implies$ stable unit RMS.
   - `test_tier2_f08_2_maximum_scale_2_plus_12`: $r = 2^{12} \implies$ stable unit RMS, no overflow.
   - `test_tier2_f08_3_subcarrier_zero_energy`: One subcarrier zero, global $s_0$ valid.
   - `test_tier2_f08_4_pure_phase_rotation_pi`: $c = e^{j\pi} = -1 \implies$ exact negation.
   - `test_tier2_f08_5_random_phase_sweep_100_angles`: Uniform sweep over $[-\pi, \pi]$.
9. **F9 (Hermitian Green's Attention)**:
   - `test_tier2_f09_1_identical_query_key_zero_distance`: $m=n \implies$ maximal affinity.
   - `test_tier2_f09_2_orthogonal_tokens`: $\text{Re}(\langle Q, K \rangle) = 0 \implies$ uniform attention.
   - `test_tier2_f09_3_maximum_relative_distance`: $|m - n| = N_{\text{sc}}-1$.
   - `test_tier2_f09_4_zero_input_activation`: $Z=0 \implies f(0) = 0$.
   - `test_tier2_f09_5_large_magnitude_token`: $|Z| = 10^4 \implies$ stable radial scaling.
10. **F10 (Bidirectional Pilot-Data Attention)**:
    - `test_tier2_f10_1_zero_pilot_tokens`: Error handling if pilot stream empty.
    - `test_tier2_f10_2_single_pilot_token`: Extreme sparse pilot scenario.
    - `test_tier2_f10_3_equal_pilot_data_tokens`: $N_p = N_d = N/2$.
    - `test_tier2_f10_4_extreme_noisy_data_feedback`: Data stream SNR $-10$ dB.
    - `test_tier2_f10_5_zero_data_feedback_ablation`: Cross-attention weight zeroed.
11. **F11 (Joint Physical Consistency Loss)**:
    - `test_tier2_f11_1_zero_channel_error`: $\hat{H} = H \implies \mathcal{L}_{\text{NMSE}} = 0$.
    - `test_tier2_f11_2_zero_bit_error`: $L$ perfectly matches $b \implies \mathcal{L}_{\text{BCE}} \to 0$.
    - `test_tier2_f11_3_zero_pa_residual`: Reconstructed signal matches received.
    - `test_tier2_f11_4_all_energy_outside_cp`: Impulse response taps in forbidden delay zone.
    - `test_tier2_f11_5_infinite_scale_loss_invariance`: Multiply inputs by $2^{12}$.
12. **F12 (Numerical Equivariance Verification)**:
    - `test_tier2_f12_1_octave_scale_grid_24_octaves`: $r \in [2^{-12}, 2^{12}]$.
    - `test_tier2_f12_2_quadrant_phase_corners`: $\phi \in \{0, \pi/2, \pi, -\pi/2\}$.
    - `test_tier2_f12_3_small_phase_perturbation`: $\phi = 10^{-6}$ rad.
    - `test_tier2_f12_4_combined_worst_case_scale_and_phase`: $c = 2^{12} e^{j 0.99\pi}$.
    - `test_tier2_f12_5_zero_bit_flip_assertion_1000_trials`: Exact 0 flip count.
13. **F13 (2D Slot Simulator & Sparse DMRS)**:
    - `test_tier2_f13_1_single_dmrs_symbol_s2`: Only symbol 2 contains pilots ($3.57\%$).
    - `test_tier2_f13_2_dual_dmrs_symbols_s2_s11`: Both symbols 2 and 11 active ($7.14\%$).
    - `test_tier2_f13_3_comb_4_sparse_pilot`: Ultra-sparse comb-4 allocation ($1.78\%$).
    - `test_tier2_f13_4_narrowband_slot_16_subcarriers`: $N_{\text{sc}} = 16$.
    - `test_tier2_f13_5_wideband_slot_64_subcarriers`: $N_{\text{sc}} = 64$.
14. **F14 (Continuous Doppler ICI & CFO)**:
    - `test_tier2_f14_1_zero_doppler_zero_cfo`: Flat static slot.
    - `test_tier2_f14_2_extreme_cfo_fractional_half_bin`: $\epsilon = 0.50$ (maximum ambiguity).
    - `test_tier2_f14_3_extreme_doppler_fd_tsym_0_05`: $5\%$ subcarrier spacing Doppler spread.
    - `test_tier2_f14_4_negative_cfo_drift`: $\epsilon = -0.05$.
    - `test_tier2_f14_5_combined_max_doppler_and_cfo`: $f_d T_{\text{sym}} = 0.05$ and $\epsilon = 0.10$.
15. **F15 (Native Outer Channel Coding)**:
    - `test_tier2_f15_1_all_zeros_codeword`: Valid all-zero input.
    - `test_tier2_f15_2_all_ones_information_vector`: Valid all-one input.
    - `test_tier2_f15_3_single_bit_error_correction`: Soft decoder corrects 1 flipped bit in 1 iteration.
    - `test_tier2_f15_4_maximum_iterations_exhausted`: Decoder reaches $I_{\text{max}} = 15$ on heavy noise.
    - `test_tier2_f15_5_extreme_llrs_clamp`: $L = \pm 1000.0 \implies$ no overflow.
16. **F16 (2D Classical Baselines)**:
    - `test_tier2_f16_1_2d_lmmse_zero_noise`: $\sigma_w^2 \to 0$ interpolation stability.
    - `test_tier2_f16_2_2d_lmmse_heavy_noise`: $\sigma_w^2 \to \infty$ regularized fallback.
    - `test_tier2_f16_3_2d_idd_pa_single_dmrs`: Interpolation with only symbol 2 DMRS.
    - `test_tier2_f16_4_high_doppler_2d_lmmse_mismatch`: Doppler mismatch between true and assumed $f_d$.
    - `test_tier2_f16_5_severe_clipping_2d_idd_pa`: $A = 1.25$ under 2D slot.
17. **F17 (Coded BLER Evaluation)**:
    - `test_tier2_f17_1_bler_zero_snr_limit`: $\text{BLER} \ge 0.99$ at $-10$ dB SNR.
    - `test_tier2_f17_2_bler_high_snr_limit`: $\text{BLER} = 0.0$ at $+30$ dB SNR.
    - `test_tier2_f17_3_single_block_transmission`: $N_{\text{blocks}} = 1$.
    - `test_tier2_f17_4_thousand_block_statistics`: $N_{\text{blocks}} = 1000$.
    - `test_tier2_f17_5_extreme_ood_delay_sparse12`: Out-of-distribution delay profile.
18. **F18 (Factorial Ablation Matrix)**:
    - `test_tier2_f18_1_all_ablations_at_r_2_minus_12`: Arm 2 collapse vs Arm 1 survival.
    - `test_tier2_f18_2_arm_8_baseline_at_extreme_gain`: Demonstrates failure of standard Transformer.
    - `test_tier2_f18_3_isolated_gauge_contribution`: Contrast Arm 1 vs Arm 2.
    - `test_tier2_f18_4_isolated_green_contribution`: Contrast Arm 1 vs Arm 3.
    - `test_tier2_f18_5_isolated_joint_contribution`: Contrast Arm 1 vs Arm 4.
19. **F19 (Numeric Stability & Precision Audit)**:
    - `test_tier2_f19_1_fp16_extreme_underflow_r_2_minus_12`: FP16 at minimum subnormal.
    - `test_tier2_f19_2_fp16_extreme_overflow_r_2_plus_12`: FP16 at maximum exponent ($65504$).
    - `test_tier2_f19_3_bf16_wide_exponent_range`: BF16 handling dynamic range.
    - `test_tier2_f19_4_zero_input_tensor_stability`: $Y = 0 \implies$ no NaN in softmax or norm.
    - `test_tier2_f19_5_denormalized_subnormal_handling`: Inputs in subnormal float range.
20. **F20 (Pre-Frozen Confirmatory Protocol)**:
    - `test_tier2_f20_1_interrupted_run_resumption`: Resume from partial seed CSV.
    - `test_tier2_f20_2_duplicate_seed_rejection`: Protocol rejects duplicated seeds.
    - `test_tier2_f20_3_zero_error_seed`: Seed with $E_{\text{bits}} = 0$.
    - `test_tier2_f20_4_all_error_seed`: Seed with $100\%$ error.
    - `test_tier2_f20_5_checksum_tampering_rejection`: Modifying `P4D_PROTOCOL.md` aborts execution.
21. **F21 (Sign-Flip Permutation Tests)**:
    - `test_tier2_f21_1_all_deltas_zero`: $\boldsymbol{\delta} = \mathbf{0} \implies p = 1.0$.
    - `test_tier2_f21_2_all_deltas_strictly_positive`: $\delta_s > 0 \implies p = 2/1024$.
    - `test_tier2_f21_3_single_huge_outlier`: $\delta_1 = 1000, \delta_{2..10} = -1$.
    - `test_tier2_f21_4_symmetric_deltas`: Five $+1$ and five $-1 \implies T_{\text{obs}} = 0, p = 1.0$.
    - `test_tier2_f21_5_holm_adjustment_monotonicity`: Verify $p_{(k)}^{\text{Holm}} \ge p_{(k-1)}^{\text{Holm}}$.
22. **F22 (Repository Governance & Immutability)**:
    - `test_tier2_f22_1_single_byte_flip_detection`: Flip 1 byte in copy $\implies$ fails audit.
    - `test_tier2_f22_2_missing_file_detection`: Missing historical file triggers non-zero exit code.
    - `test_tier2_f22_3_unexpected_new_file_in_root`: File created in root triggers isolation alert.
    - `test_tier2_f22_4_manifest_sha256_recomputation`: Recompute SHA256 of entire repo matches manifest.
    - `test_tier2_f22_5_read_only_filesystem_execution`: Verify tests pass on read-only mounted codebase.

---

## Detailed Specifications for Tier 3: Pairwise Cross-Feature Interactions (>=22 Tests)

1. **Test 3.01: F1 (Rapp PA) + F2 (Doppler Multipath)**: Time-domain Rapp PA distortion passing through dynamic Doppler multipath channel; verifies that time-varying channel spreads non-linear intermodulation products across adjacent subcarriers.
2. **Test 3.02: F4 (Bussgang LMMSE) + F5 (Soft LLRs)**: Equalized symbols from Bussgang-LMMSE fed directly to calibrated 16-QAM soft LLR demapper; verifies probability calibration ECE $< 0.02$.
3. **Test 3.03: F6 (IDD-PA) + F4 (Bussgang LMMSE)**: Bussgang-LMMSE initializes Iterative DD-PA cancellation; verifies monotonic reduction of residual distortion across 3 iterations.
4. **Test 3.04: F8 ($C^*$ Gauge) + F9 (Hermitian Attention)**: Dimensionless gauge-normalized tokens fed into complex linear RoPE attention; verifies exact scale and phase cancellation.
5. **Test 3.05: F9 (Hermitian Attention) + F10 (Bidirectional Coupling)**: Bidirectional cross-attention instantiated with Hermitian Green's Propagator attention; verifies dual-stream convergence without phase warping.
6. **Test 3.06: F11 (Joint Loss) + F8 ($C^*$ Gauge)**: Multi-task physical consistency loss evaluated on gauge-normalized model; verifies $\mathcal{L}_{\text{joint}}$ is scale-invariant across $r \in [2^{-12}, 2^{12}]$.
7. **Test 3.07: F12 (Equivariance) + F10 (Bidirectional Propagator)**: Full bidirectional propagator architecture subjected to formal equivariance test suite; verifies $N_{\text{flip}} \equiv 0$ and $\epsilon_H < 10^{-5}$.
8. **Test 3.08: F13 (2D Slot) + F14 (Doppler ICI & CFO)**: 14-symbol slot simulated with continuous time-varying Doppler fading and sample CFO; verifies phase drift across symbols.
9. **Test 3.09: F13 (2D Slot) + F16 (2D-LMMSE Baseline)**: Sparse DMRS grid interpolated via 2D Wiener filter ($R_{\text{time}} \otimes R_{\text{freq}}$); verifies channel estimation MSE $< -15$ dB.
10. **Test 3.10: F14 (CFO) + F16 (2D IDD-PA)**: Continuous CFO phase drift tracked and compensated within 2D Iterative Decision-Directed cancellation loop.
11. **Test 3.11: F5 (Soft LLRs) + F15 (Native LDPC Decoder)**: Calibrated soft LLRs decoded by native PyTorch Rate-1/2 LDPC Min-Sum decoder; verifies valid parity syndrome $H \hat{c}^T = 0 \pmod 2$.
12. **Test 3.12: F15 (Outer FEC) + F16 (2D IDD-PA)**: 2D IDD-PA soft LLRs decoded through outer LDPC; verifies coded BLER reduction across iterations.
13. **Test 3.13: F17 (Coded BLER) + F14 (High Doppler)**: Coded BLER evaluated under vehicular Doppler ($f_d = 300$ Hz); benchmarks EM-FloatFormer against 2D-LMMSE.
14. **Test 3.14: F17 (Coded BLER) + F2 (OOD Delay Spreads)**: Coded BLER evaluated under out-of-distribution delay profile `sparse12`; tests generalization beyond training.
15. **Test 3.15: F18 (Factorial Matrix) + F8 ($C^*$ Gauge)**: All 8 factorial arms evaluated under extreme coherent gain $r = 2^{-12}$; verifies that only arms with $+F_1$ survive.
16. **Test 3.16: F19 (Mixed Precision) + F8 ($C^*$ Gauge)**: BF16 and FP16 execution tested across 24 octaves of dynamic range; verifies zero NaNs and zero arithmetic overflow.
17. **Test 3.17: F19 (Mixed Precision) + F9 (Hermitian Attention)**: Complex attention projections and softmax evaluated under CPU autocast BF16; verifies max logit deviation $< 0.15$.
18. **Test 3.18: F20 (Confirmatory Protocol) + F17 (BLER Accounting)**: 10 pre-frozen confirmatory seeds executed with exact integer block error logging into `raw_by_seed.csv`.
19. **Test 3.19: F21 (Sign-Flip Tests) + F20 (Confirmatory Seeds)**: Exact 1024-state sign-flip permutation tests executed on paired seed deltas from the confirmatory seeds.
20. **Test 3.20: F21 (Holm Adjustment) + F17 (Multi-Endpoint BLER)**: Step-down Holm adjustment applied across 3 primary endpoints: in-distribution BLER, OOD Doppler BLER, and OOD Delay BLER.
21. **Test 3.21: F22 (Governance) + F20 (Protocol Freeze)**: SHA256 integrity verification executed immediately before and after the 10-seed confirmatory run.
22. **Test 3.22: Full End-to-End Pipeline (F1 + F8 + F15)**: Complete pipeline from TX outer FEC $\to$ 16-QAM $\to$ oversampled Rapp PA $\to$ dynamic multipath $\to$ RX $C^*$ gauge norm $\to$ Green's attention $\to$ soft LDPC decoding $\to$ bit recovery.

---

## Detailed Specifications for Tier 4: Real-World Workload Scenarios (>=11 Scenarios)

1. **Scenario 1: High-Speed Vehicular Highway Link**:
   - $N_{\text{symb}} = 14, N_{\text{sc}} = 32$, carrier 5.0 GHz, terminal speed $v = 120$ km/h ($f_d = 555$ Hz, $f_d T_{\text{sym}} \approx 0.03$).
   - Profile: `long6`, Rapp PA $A = 2.25$, DMRS at symbols 2 and 11, SNR $= 15$ dB.
   - Evaluates tracking of continuous time-varying fading and ICI mitigation.
2. **Scenario 2: Severe Nonlinear PA Saturation (Cell-Edge Uplink)**:
   - $A = 1.25$ (severe compression), $\alpha = 0.5, \beta = 1.0$, SNR $= 12$ dB.
   - 16-QAM constellation, Profile `equal4`.
   - Evaluates nonlinear distortion cancellation and soft LLR calibration.
3. **Scenario 3: Deep Fading Sparse Multipath**:
   - Delay profile `sparse12` (energy concentrated at delay taps 0, 4, 11).
   - Frequency-selective nulls spanning up to $25$ dB depth across subcarriers.
   - Evaluates Green's propagator attention interpolation across deep spectral notches.
4. **Scenario 4: High Carrier Frequency Offset (CFO) Drift**:
   - Normalized CFO $\epsilon = 0.08$ ($8\%$ subcarrier spacing), $f_d = 100$ Hz.
   - Continuous phase accumulation across 14 symbols.
   - Evaluates receiver phase tracking and constellation alignment.
5. **Scenario 5: Ultra-Sparse DMRS Pilot Allocation**:
   - Single DMRS symbol at index $s=2$, frequency comb-2 ($3.57\%$ pilot overhead).
   - Speed $v = 60$ km/h, SNR $= 18$ dB.
   - Evaluates bidirectional data-to-pilot attention in sustaining channel tracking with minimal pilots.
6. **Scenario 6: Wide Dynamic Range Shadowing (72 dB Scale)**:
   - Signal levels sweeping through $r \in \{2^{-12}, 2^{-6}, 1.0, 2^6, 2^{12}\}$.
   - SNR maintained at $15$ dB relative to channel gain.
   - Evaluates $C^*$ gauge normalization in preventing scale drift and float overflow.
7. **Scenario 7: Cell-Edge Low-SNR Regime**:
   - SNR $= 0$ dB, strong multipath `long6`, Rapp PA $A = 1.25$.
   - Rate-1/2 LDPC soft decoding.
   - Evaluates robust convergence of message passing and joint physical consistency loss.
8. **Scenario 8: High-Throughput Clean Line-of-Sight**:
   - SNR $= 25$ dB, static Rician channel ($K_{\text{factor}} = 10$ dB), linear PA ($A = \infty$).
   - Evaluates asymptotic error-free transmission ($\text{BLER} < 10^{-4}$) and parity with classical ZF/LMMSE.
9. **Scenario 9: Combined Multi-Impairment Extreme Stress**:
   - Simultaneous worst-case impairments: $A = 1.25$, $f_d T_{\text{sym}} = 0.04$, $\epsilon = 0.05$, SNR $= 10$ dB, profile `sparse12`.
   - Comprehensive robustness stress test of the full EM-FloatFormer architecture.
10. **Scenario 10: Complete Full-Slot Coded 5G NR Transmission**:
    - Full slot (14 symbols $\times$ 64 subcarriers $= 896$ REs).
    - 5G NR LDPC Base Graph 2, codeword length 384 bits, 16-QAM Gray mapping.
    - End-to-end information bit error tracking and throughput calculation.
11. **Scenario 11: 10-Seed Confirmatory Replication Protocol**:
    - Pre-frozen seeds $\mathcal{S}_{\text{P4D}} = \{74001, \dots, 74010\}$.
    - 600 training steps, batch size 64.
    - Evaluates primary endpoints with exact integer error accounting and 1024-state sign-flip permutation tests.

---

## Conclusion & Implementation Directives for Test Engineers

1. **Test Runner Entry Point**:
   ```powershell
   python -m unittest tests/test_p4_e2e.py -v
   ```
2. **Progressive Testability**:
   - Tier 1 and Tier 2 tests for Features F1–F7 must execute and pass as soon as Milestone M1 (`p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`) is delivered.
   - M2 features (F8–F12) can be verified standalone via mock tensors and interface contracts.
   - All tests must maintain strict isolation and exit with code 0.
3. **Immutability Contract**:
   - No historical file outside `p4/` and `tests/test_p4_e2e.py` may be edited. SHA256 verification in F22 guards this invariant.
