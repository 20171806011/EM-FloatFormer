# Comprehensive Survey Report: 2D Generalization, Probing & Replication (Stages P4-B, P4-C, P4-D)

**Agent**: Explorer 3 (`teamwork_explorer_survey_3`)
**Role**: 2D Generalization, Probing & Replication Specialist
**Working Directory**: ``
**Date**: 2026-11-15
**Milestone**: Survey Phase — Technical Scoping, Architecture Audit, and Replication Protocol

---

## Executive Summary

This survey provides a comprehensive architectural, mathematical, and algorithmic audit of the `EM-FloatFormer` repository in support of **Stage P4** of the research roadmap. Specifically, it establishes the factual baseline for:
1. **2D Time-Frequency Generalization and Coded Transmission (Stage P4-B / Requirement R3)**: Transitioning from static 1D single-symbol QPSK to multi-symbol slots ($N_{\text{symb}}=14$ or $7$), sparse 5G-NR-style DMRS pilots ($\le 10\%$ overhead), continuous Doppler Inter-Carrier Interference (ICI), Carrier Frequency Offset (CFO), calibrated Gray 16-QAM soft LLR calculation, and outer channel coding (LDPC / Convolutional) with soft decoding and coded Block Error Rate (BLER) benchmarking against 2D-LMMSE and Iterative Decision-Directed PA Cancellation (IDD-PA).
2. **Factorial Mechanism Probing (Stage P4-C / Requirement R4)**: Designing a rigorous $2^3$ factorial ablation study isolating (i) $C^* = \mathbb{R}_{>0} \times U(1)$ gauge invariance, (ii) Complex Hermitian Green's Propagator Attention (vs standard real attention), and (iii) Bidirectional joint pilot-data propagation.
3. **Numeric Stability and Mixed-Precision Audits (Stage P4-C / Requirement R4)**: Resolving the root causes of the P3-C dynamic-range collapse under coherent gain scaling ($r \in [2^{-12}, 2^{+12}]$) through exact $C^*$ gauge normalization, dimensionless stabilizers, and verified CPU/GPU mixed-precision (FP32 vs BF16/FP16) execution.
4. **Pre-Frozen Confirmatory Replication Protocol (Stage P4-D / Requirement R4)**: Establishing an uncorrupted confirmatory replication framework over 10 fresh seeds, full integer error and block accounting, exhaustive two-sided sign-flip permutation tests ($2^{10} = 1024$ states), Holm step-down family-wise error rate control, and strict directory-level repository governance guaranteeing historical P1–P3 data immutability.

---

## 1. Architectural & Historical Code Audit (P1 through P3)

An exhaustive code and report inspection reveals that the entire existing codebase up to Stage P3 is fundamentally constrained to single-symbol, static 1D channels with uncoded QPSK:

| Historical Stage | Channel & Signal Model | Model Architecture | Pilots & Grid Structure | Key Findings & Limitations |
|---|---|---|---|---|
| **Stage P1** (`p1b`, `p1c`, `p1d`, `p1e`) | 32 subcarriers ($N=32$), $CP=16$, static 3/4/6-tap multipath, AWGN, uncoded QPSK. | Circular CNN, MLP-Mixer, Token-MLP, standard real Transformer (`RepresentationTransformer`). | 1 pilot symbol (all 32 tones) + 1 data symbol (32 tones). | Real Transformer failed out-of-distribution (OOD) against Circular CNN due to lack of phase equivariance and overfitting to absolute delay positions. |
| **Stage P2** (`p2_models`, `p2_run`, `p2_symmetry`, `p2_numeric`) | 32 subcarriers, static multipath, AWGN, uncoded QPSK. | Real delay-tap shrinkage (`DelayUnfold`), 9-tap complex circular FIR (`EquivComplexLocal`). | 1 pilot symbol + 1 data symbol (full 32 tones). | First proof of $U(1)$ phase equivariance. However, 8-tap DFT smoothing and LMMSE still beat neural receivers on unseen profiles (`P2E_REPORT.md:37`). |
| **Stage P3-A / A2** (`p3_models`, `p3_a_mechanism`) | 32 subcarriers, static multipath (train3, equal4, long6, long12, sparse12), AWGN. | Delay-domain convolutional gate (`EquivDelayConv`, 985 params). | 32 pilot tones; explicitly enforces `del yd` (no data symbols used for channel estimation). | $2 \times 2$ factorial ablation proved position-shortcut deletion and residual floor reduce long-delay collapse, but model was trained on bit BCE, yielding poor channel MSE ($\sim 0.40$). |
| **Stage P3-B** (`p3b_time_link`, `p3b_receivers`) | 32 subcarriers, static multipath with single nonidealities: Rapp PA ($p=4$), impulse, CPE, ADC. | Frozen P3-A2 D backbone + 1D CNN residual logit head (`DataResidualHead`, 2,090 params). | 1 pilot symbol (32 tones) + 1 data symbol (30 scored QPSK tones, 2 anchors at subcarriers 0, 8). | **No neural model beat the classical DSP envelope** across 135 cells (`P3B_REPORT.md:9`). Crucially: no Bussgang-LMMSE or nonlinear inverse was implemented. |
| **Stage P3-C** (`p3c_precision`, `test_p3c_precision`) | Same P3-B link; CPU mixed precision audit; coherent gain stress test ($r = 2^{-12}, 2^0, 2^{12}$). | Reused frozen P3-B heads. Real Conv1d executed with BF16/FP16 autocast; FFT remained FP32. | Same 1 pilot + 1 data symbol. | **Catastrophic collapse at extreme gains** ($r = 2^{\pm 12}$) even in pure FP32 (BER $\approx 50\%$) due to unnormalized inputs and hardcoded dimensional stabilizers (`+ 1e-2`). |
| **Exploratory Probes** (`16qam_joint_pa_probe`, `p3de_tdl_surrogate`) | 16-QAM under Rapp PA ($n=3$ seeds); static TDL-A/C/D profile surrogate ($n=10$ seeds). | Small 1,812-param 16-QAM CNN residual slicer; P3-A2 D on TDL tables. | Static single-symbol frames. | No probability-calibrated LLRs; no iterative clipping cancellation; no Doppler; Sionna 2.2 package unavailable offline (`P3DE_SURROGATE_REPORT.md:81`). |

**Conclusion of Code Audit**:
There is currently **zero existing code** in the repository for:
- Multi-symbol time-varying slots (only 1 pilot + 1 payload symbol exists).
- Sparse DMRS pilots (only 100% full-symbol pilots or 2 data anchors exist).
- Continuous Doppler ICI or Carrier Frequency Offset (CFO).
- Outer channel coding (LDPC, Polar, Convolutional) and soft decoding.
- Coded Block Error Rate (BLER) evaluation.
- 2D-LMMSE or Iterative Decision-Directed PA Cancellation (IDD-PA).
- $C^* = \mathbb{R}_{>0} \times U(1)$ gauge normalization and Hermitian Green's Propagator Attention.

All of these are essential deliverables that must be systematically engineered in Stage P4.

---

## 2. Topic 1: Multi-Symbol Slot Structures, Sparse DMRS, High-Doppler ICI, and CFO (Stage P4-B)

### 2.1 2D Time-Frequency Slot Structure
In modern cellular standards (5G NR, 3GPP TS 38.211), transmission is organized into 2D time-frequency resource grids rather than isolated 1D symbols:
- **Slot Configuration**: $N_{\text{symb}} = 14$ consecutive OFDM symbols per slot (normal cyclic prefix), with $N_{\text{sc}} = 32$ or $64$ subcarriers.
- **Resource Element (RE)**: Each coordinate $(s, k)$ corresponds to OFDM symbol index $s \in \{0, \dots, N_{\text{symb}}-1\}$ and subcarrier index $k \in \{0, \dots, N_{\text{sc}}-1\}$.
- **Total REs per Slot**: $M_{\text{slot}} = N_{\text{symb}} \times N_{\text{sc}} = 14 \times 32 = 448$ REs (or $14 \times 64 = 896$ REs).

### 2.2 Sparse DMRS Pilot Patterns
Unlike P1–P3 where pilots occupied 100% of an entire OFDM symbol ($50\%$ total slot overhead), Stage P4-B requires sparse DMRS:
- **DMRS Symbol Allocation**: Standard single-symbol DMRS located at symbol index $s_{\text{dmrs}} = 2$ (and optionally an additional symbol at $s_{\text{dmrs}} = 11$ for high-Doppler tracking).
- **Frequency Subcarrier Comb**: Frequency comb-2 allocation on DMRS symbols:
  $$\mathcal{P}_{\text{dmrs}} = \left\{ (s, k) \;\middle|\; s \in \{2, 11\}, \; k \in \{0, 2, 4, \dots, N_{\text{sc}}-2\} \right\}$$
- **Pilot Overhead**: For a $14 \times 32$ slot, 2 DMRS symbols $\times 16$ pilot subcarriers $= 32$ pilot REs out of 448 total REs ($7.14\%$ pilot overhead).
- **Pilot Symbols**: Pseudo-random QPSK sequence $X_p[s, k] \in \{ (\pm 1 \pm j)/\sqrt{2} \}$ generated via Gold sequence or standard cryptographically seeded PRNG.

### 2.3 Time-Varying Dynamic Channel & High-Doppler ICI
In dynamic propagation environments, user terminal motion induces a Doppler shift $f_d = \frac{v f_c}{c}$.
- **Continuous Channel Fading Model**:
  The continuous-time time-variant channel impulse response across delay taps $\tau_l$ is given by Clarke / Jakes sum-of-sinusoids or autoregressive Gauss-Markov evolution:
  $$h_l(t) = \sum_{m=1}^{M_{\text{paths}}} c_{l, m} e^{j (2 \pi f_d \cos(\theta_{l, m}) t + \phi_{l, m})}$$
  where $\theta_{l, m} \sim \mathcal{U}[0, 2\pi)$ and $\sum_l \mathbb{E}[|h_l(t)|^2] = 1$.
- **Sample-Level Continuous Convolution**:
  Let $x(n)$ be the concatenated time-domain transmitted signal including cyclic prefixes across all $N_{\text{symb}}$ symbols:
  $$x_s(n) = \frac{1}{\sqrt{N_{\text{sc}}}} \sum_{k=0}^{N_{\text{sc}}-1} X_s[k] e^{j 2 \pi k n / N_{\text{sc}}}, \quad n = -N_{\text{cp}}, \dots, N_{\text{sc}}-1$$
  The received time-domain sample at absolute time index $t = s(N_{\text{sc}} + N_{\text{cp}}) + N_{\text{cp}} + n$ is computed via explicit causal time-variant FIR:
  $$y_s[n] = \sum_{l=0}^{L-1} h_l[t] x(t - \tau_l) + w_s[n]$$
- **High-Doppler Inter-Carrier Interference (ICI)**:
  When $f_d T_{\text{sym}} > 0.01$, the channel changes *within* the duration of an OFDM symbol ($n = 0, \dots, N_{\text{sc}}-1$). Taking the receiver FFT yields:
  $$Y_s[k] = \frac{1}{\sqrt{N_{\text{sc}}}} \sum_{n=0}^{N_{\text{sc}}-1} y_s[n] e^{-j 2 \pi k n / N_{\text{sc}}} = \sum_{k'=0}^{N_{\text{sc}}-1} H_s[k, k'] X_s[k'] + W_s[k]$$
  where the 2D transfer matrix $H_s[k, k']$ is:
  $$H_s[k, k'] = \frac{1}{N_{\text{sc}}} \sum_{n=0}^{N_{\text{sc}}-1} \left( \sum_{l=0}^{L-1} h_l[t_s(n)] e^{-j 2 \pi k' \tau_l / N_{\text{sc}}} \right) e^{-j 2 \pi (k - k') n / N_{\text{sc}}}$$
  - Diagonal entries $H_s[k, k] = \bar{H}_s[k]$ represent the average effective subcarrier gain.
  - Off-diagonal entries $H_s[k, k']$ ($k' \ne k$) represent inter-carrier interference (ICI) leakage, creating an irreducible error floor for conventional 1D single-tap equalizers!

### 2.4 Carrier Frequency Offset (CFO) Modeling & Compensation
Oscillator mismatch between transmitter and receiver causes continuous phase drift:
- **Baseband CFO**: Let $\Delta f_{\text{cfo}}$ be the carrier frequency offset, and $\epsilon = \Delta f_{\text{cfo}} / \Delta f_{\text{sub}}$ be the normalized CFO.
- **Time-Domain Phase Rotation**: Each time sample $t$ is multiplied by $e^{j 2 \pi \epsilon t / N_{\text{sc}}}$.
  - Within symbol $s$: $y_s[n] \mapsto y_s[n] e^{j 2 \pi \epsilon (s(N_{\text{sc}}+N_{\text{cp}}) + N_{\text{cp}} + n) / N_{\text{sc}}}$.
  - Demodulated frequency symbol experiences both an attenuation $\text{sinc}(\epsilon)$, an ICI leakage from adjacent bins, and a symbol-to-symbol phase roll of $2 \pi \epsilon \frac{N_{\text{sc}} + N_{\text{cp}}}{N_{\text{sc}}}$.
- **Estimation & Tracking**:
  - Fractional CFO estimated via Cyclic Prefix correlation:
    $$\hat{\epsilon}_{\text{cp}} = \frac{1}{2\pi} \arg \left( \sum_{n=0}^{N_{\text{cp}}-1} y_s[n + N_{\text{sc}}] y_s^*[n] \right)$$
  - Residual CFO tracked across multiple DMRS symbols via cross-symbol correlation:
    $$\Delta \hat{\phi} = \arg \left( \sum_{k \in \mathcal{P}} Y_{s_2}[k] Y_{s_1}^*[k] X_p^*[s_2, k] X_p[s_1, k] \right)$$
  - Classical receivers de-rotate by $e^{-j 2 \pi \hat{\epsilon} t / N_{\text{sc}}}$; neural receivers must learn robust continuous phase tracking or ingest de-rotated baseband samples.

---

## 3. Topic 2: Outer Channel Coding, Soft Demapping & BLER Benchmarking (Stage P4-B)

### 3.1 Calibrated 16-QAM Soft LLR Calculation
Under nonlinear Rapp PA distortion, the received constellation is deformed by AM-AM magnitude compression and additive distortion noise $d$:
$$y = \alpha x + d + w, \quad d \sim \mathcal{CN}(0, \sigma_d^2)$$
For Gray-coded 16-QAM with constellation points $x = (I + j Q) / \sqrt{10}$, $I, Q \in \{\pm 1, \pm 3\}$ carrying 4 bits $[b_0, b_1, b_2, b_3]$:
- Exact Max-Log LLR for bit $i$:
  $$L(b_i) \approx \frac{1}{\sigma_{\text{eff}}^2} \left[ \min_{x \in \mathcal{X}: b_i=1} |y - \alpha x|^2 - \min_{x \in \mathcal{X}: b_i=0} |y - \alpha x|^2 \right]$$
  where $\sigma_{\text{eff}}^2 = \sigma_d^2 + \sigma_w^2$ is the total effective noise plus distortion variance.
- For standard Gray mapping, this simplifies to piece-wise linear demapping:
  $$L(b_0) = \frac{4 \alpha}{\sqrt{10} \sigma_{\text{eff}}^2} \text{Re}\{\hat{x}\}$$
  $$L(b_1) = \frac{4 \alpha}{\sqrt{10} \sigma_{\text{eff}}^2} \left( -\left| \text{Re}\{\hat{x}\} \right| + \frac{2 \alpha}{\sqrt{10}} \right)$$
  $$L(b_2) = \frac{4 \alpha}{\sqrt{10} \sigma_{\text{eff}}^2} \text{Im}\{\hat{x}\}$$
  $$L(b_3) = \frac{4 \alpha}{\sqrt{10} \sigma_{\text{eff}}^2} \left( -\left| \text{Im}\{\hat{x}\} \right| + \frac{2 \alpha}{\sqrt{10}} \right)$$
- **Crucial Requirement**: If $\alpha$ (Bussgang gain) and $\sigma_{\text{eff}}^2$ are not properly calibrated, LLR magnitudes are scaled incorrectly, causing outer channel decoders to fail. The EM-Native Transformer must directly emit calibrated LLRs trained under Binary Cross-Entropy with true bit labels!

### 3.2 Outer Forward Error Correction (FEC) Architecture
Because external packages like Sionna or pyldpc are unavailable in the isolated environment, Stage P4-B must feature an efficient, self-contained, native PyTorch FEC module:
- **FEC Code Option 1: Vectorized 5G-NR Compliant LDPC**:
  - Parity-check matrix $H \in \{0, 1\}^{M \times N}$ derived from 5G NR Base Graph 2 (rate $R = 1/2$, lifting size $Z = 16$ or $Z = 24$).
  - Codeword length $N_{\text{code}} = 320$ to $480$ bits, information payload $K_{\text{code}} = 160$ to $240$ bits.
  - Soft Decoder: Min-Sum Algorithm (MSA) or Normalized Min-Sum (NMSA with factor $\alpha_{\text{ms}} = 0.75$) implemented directly in vectorized PyTorch tensor operations.
  - Variable-to-check and check-to-variable message passing runs for $I_{\text{max}} = 8$ to $15$ iterations with early parity check stopping ($H \hat{c}^T = 0 \pmod 2$).
- **FEC Code Option 2: Rate-1/2 Convolutional Code ($K=7$, [133, 171] octal)**:
  - Standard cellular / Wi-Fi convolutional code with constraint length $K=7$.
  - Soft Decoder: Vectorized Max-Log-MAP (BCJR) soft decoder or Soft-Output Viterbi Algorithm (SOVA).
- **Validation**: The FEC implementation must be independently verified against theoretical BPSK AWGN waterfall curves to guarantee zero algorithmic bugs.

### 3.3 Coded Block Error Rate (BLER) Evaluation
- For each slot, data payload bits are encoded into code blocks and mapped onto the data REs.
- At the receiver, equalized soft LLRs are fed into the soft decoder to obtain estimated information bits $\hat{u}$.
- **Block Error Event**: A block is declared in error if $\hat{u} \ne u$.
- $\text{BLER} = \frac{\sum_{t=1}^{N_{\text{blocks}}} \mathbb{I}(\hat{u}_t \ne u_t)}{N_{\text{blocks}}}$.
- BLER must be evaluated across an SNR grid ($0, 5, 10, 15, 20$ dB) across:
  - In-distribution channel conditions (e.g. low Doppler $f_d = 50$ Hz, moderate delay spread).
  - Out-of-Distribution (OOD) Doppler ($f_d = 300$ Hz and $f_d = 600$ Hz).
  - Out-of-Distribution (OOD) Delay Spreads (long12, sparse12).

### 3.4 Classical Baselines for P4-B
Stage P4-B benchmarks the EM-Native Transformer against two rigorously defined classical baselines:
1. **2D-LMMSE Channel Estimator**:
   - Computes separable 2D time-frequency correlation matrix $R_{tf} = R_{\text{time}} \otimes R_{\text{freq}}$, where $R_{\text{time}}[s, s'] = J_0(2 \pi f_d T_{\text{sym}} (s - s'))$ and $R_{\text{freq}}[k, k'] = \sum_l p_l e^{-j 2 \pi (k - k') \tau_l / N_{\text{sc}}}$.
   - Linear minimum mean square error interpolation from DMRS positions $\mathcal{P}$:
     $$\hat{H} = R_{all, \mathcal{P}} \left( R_{\mathcal{P}, \mathcal{P}} + \sigma_w^2 I \right)^{-1} Y_{\mathcal{P}}$$
   - Followed by LMMSE equalization and calibrated 16-QAM soft LLR demapping.
2. **Iterative Decision-Directed PA Cancellation (IDD-PA)**:
   - **Iteration 0**: Initial 2D-LMMSE channel estimate $\hat{H}^{(0)}$ from DMRS; compute tentative 16-QAM symbol decisions $\hat{X}^{(0)}$.
   - **Iteration 1 & 2**:
     1. Pass tentative decisions $\hat{X}^{(m)}$ through the Rapp PA model to compute nonlinear distortion estimate $\hat{D}^{(m)} = g(\hat{X}^{(m)}) - \alpha \hat{X}^{(m)}$.
     2. Subtract reconstructed nonlinear distortion from received grid: $\tilde{Y}^{(m)} = Y - \hat{H}^{(m)} \hat{D}^{(m)}$.
     3. Use tentative decisions as virtual pilots across all data REs to re-estimate and track time-varying channel $\hat{H}^{(m+1)}$.
     4. Re-equalize and decode soft LLRs.

---

## 4. Topic 3: Factorial Ablation Study Design (Stage P4-C)

To scientifically determine whether the proposed EM-Native Transformer's advantages stem from its specific physics-informed inductances rather than raw parameter capacity, Stage P4-C conducts an orthogonal $2^3$ factorial ablation study.

### 4.1 The Three Isolated Mechanisms

#### Factor 1: Exact $C^* = \mathbb{R}_{>0} \times U(1)$ Gauge Invariance ($F_1$)
- **Active ($+F_1$)**: Exact $C^*$ gauge normalization:
  - Input IQ grid is normalized by global slot root-mean-square amplitude $S_{\text{ref}} = \sqrt{\frac{1}{M} \sum |Y|^2}$ with a dimensionless stabilizer $\epsilon = 10^{-12} \cdot \sigma_w^2 / P_y$.
  - All feature projections maintain zero bias ($b = 0$).
  - Invariance under $Y \mapsto c Y$ ($c = r e^{j \theta} \in \mathbb{C}^*$):
    $$\text{Logits}(c Y, |c|^2 \sigma_w^2) = \text{Logits}(Y, \sigma_w^2), \quad \hat{H}(c Y, |c|^2 \sigma_w^2) = c \hat{H}(Y, \sigma_w^2)$$
- **Ablated ($-F_1$)**: Standard LayerNorm / RMSNorm with learnable affine scaling and biases, breaking $C^*$ gauge equivariance.

#### Factor 2: Complex Hermitian Green's Propagator Attention ($F_2$)
- **Active ($+F_2$)**:
  - Complex linear projections $W_Q, W_K, W_V \in \mathbb{C}^{d \times d}$ without real biases.
  - Hermitian inner product attention: $A = \frac{1}{\sqrt{d}} Q K^H$.
  - Complex 2D Relative Phase RoPE encoding relative time and frequency distances:
    $$R(\Delta s, \Delta k) = e^{j (\Delta s \cdot \omega_{\text{time}} + \Delta k \cdot \omega_{\text{freq}})}$$
  - Radial $U(1)$-preserving activation: $f(z) = \text{SiLU}(|z|) \frac{z}{|z|}$.
- **Ablated ($-F_2$)**: Standard real-valued Multihead Attention (concatenating real and imaginary parts into $2d$ real channels with standard dot-product attention $Q K^T$, real LayerNorm, and real GELU).

#### Factor 3: Bidirectional Joint Pilot-Data Propagation ($F_3$)
- **Active ($+F_3$)**:
  - Joint token representation where pilot REs and data REs are processed simultaneously in a single transformer grid.
  - Multi-head self-attention allows data REs to attend to pilots (channel propagation) and pilots to attend to data (decision-directed feedback and nonlinear distortion tracking).
- **Ablated ($-F_3$)**: Decoupled pipeline (pilot-only channel estimator followed by separate equalization and residual logit head, with no cross-attention from data to pilots, matching P3-A/P3-B).

### 4.2 Full $2^3$ Factorial Matrix

All 8 models are parameter-matched ($\approx 25\text{k} - 35\text{k}$ parameters), trained with identical optimizers, loss schedules, and paired seed initializations:

| Arm ID | $C^*$ Gauge ($F_1$) | Green's Attn ($F_2$) | Joint Prop ($F_3$) | Description / Control Purpose |
|---|:---:|:---:|:---:|---|
| **Arm 1 (Full)** | $+$ | $+$ | $+$ | Full EM-Native Propagator Transformer. |
| **Arm 2 ($-F_1$)** | $-$ | $+$ | $+$ | Isolates marginal necessity of $C^*$ gauge normalization. |
| **Arm 3 ($-F_2$)** | $+$ | $-$ | $+$ | Isolates marginal contribution of complex Hermitian Green attention vs real attention. |
| **Arm 4 ($-F_3$)** | $+$ | $+$ | $-$ | Isolates marginal contribution of joint pilot-data propagation vs decoupled pipeline. |
| **Arm 5 ($-F_1, -F_2$)** | $-$ | $-$ | $+$ | Joint propagation with standard real attention and standard norm. |
| **Arm 6 ($-F_1, -F_3$)** | $-$ | $+$ | $-$ | Complex Green's attention in decoupled pipeline without gauge invariance. |
| **Arm 7 ($-F_2, -F_3$)** | $+$ | $-$ | $-$ | Real attention in decoupled pipeline with gauge invariance. |
| **Arm 8 (Baseline)** | $-$ | $-$ | $-$ | Standard real Transformer baseline in decoupled pipeline. |

### 4.3 Statistical Estimation of Causal Main Effects & Interactions
Using Yates' algorithm or standard linear contrast vectors over $n=10$ seeds:
- **Main Effect of $C^*$ Gauge Invariance**:
  $$\text{ME}(F_1) = \frac{1}{4} \sum_{i \in \{1, 3, 4, 7\}} \text{BER}(\text{Arm}_i) - \frac{1}{4} \sum_{j \in \{2, 5, 6, 8\}} \text{BER}(\text{Arm}_j)$$
- **Main Effect of Green's Attention**:
  $$\text{ME}(F_2) = \frac{1}{4} \sum_{i \in \{1, 2, 4, 6\}} \text{BER}(\text{Arm}_i) - \frac{1}{4} \sum_{j \in \{3, 5, 7, 8\}} \text{BER}(\text{Arm}_j)$$
- **Main Effect of Joint Propagation**:
  $$\text{ME}(F_3) = \frac{1}{4} \sum_{i \in \{1, 2, 3, 5\}} \text{BER}(\text{Arm}_i) - \frac{1}{4} \sum_{j \in \{4, 6, 7, 8\}} \text{BER}(\text{Arm}_j)$$
- **Two-way Interactions**: $\text{INT}(F_1, F_2)$, $\text{INT}(F_1, F_3)$, $\text{INT}(F_2, F_3)$ to test for super-additive or overlapping inductive biases.

---

## 5. Topic 4: Numeric Stability and Mixed-Precision Audits (Stage P4-C)

### 5.1 Root-Cause Analysis of P3-C Coherent Gain Failure
In `P3C_REPORT.md:68`, coherent gain scaling ($r = 2^{\pm 12}$) caused total collapse even under 32-bit floating point (`FP32`):
- **Diagnostic Cause 1: Hardcoded Dimensional Stabilizers**:
  In `p3b_receivers.py:27`:
  $$\text{equalized}(y, h, \sigma_w^2) = \frac{y h^*}{|h|^2 + \sigma_w^2 + 10^{-2}}$$
  When the signal was attenuated by $r = 2^{-12}$, signal power and noise variance scaled by $r^2 = 2^{-24} \approx 5.96 \times 10^{-8}$. The hardcoded constant $10^{-2}$ was $167,000\times$ larger than the actual signal and noise power! The denominator was completely dominated by $10^{-2}$, causing equalized symbols to collapse to near-zero.
- **Diagnostic Cause 2: Unnormalized Feature Encodings**:
  In `p3_models.py:45`: $\log_{10}(\sigma_w^2)$ was directly fed to linear layers. When $\sigma_w^2$ scaled by $2^{-24}$, $\log_{10}(\sigma_w^2)$ shifted by $-7.22$, driving neurons far outside their trained activation distribution.
- **Diagnostic Cause 3: Dynamic Range Limits in Half Precision**:
  FP16 has a maximum representable value of $65,504$ and minimum subnormal value of $5.96 \times 10^{-8}$. An unnormalized power value scaled by $r^2 = 2^{+24} = 1.67 \times 10^7$ causes immediate overflow (`+inf`), yielding NaNs throughout gradient backpropagation or softmax.

### 5.2 Resolution via Exact $C^*$ Gauge Normalization
The EM-Native Transformer resolves this fundamentally:
1. **Dimensionless Scale Extraction**:
   $$S_{\text{ref}} = \sqrt{\frac{1}{M_{\text{slot}}} \sum_{s=0}^{N_{\text{symb}}-1} \sum_{k=0}^{N_{\text{sc}}-1} |Y_s[k]|^2}$$
2. **Dimensionless Normalized Grid**:
   $$\tilde{Y}_s[k] = \frac{Y_s[k]}{S_{\text{ref}}}, \quad \tilde{\sigma}_w^2 = \frac{\sigma_w^2}{S_{\text{ref}}^2}$$
   By construction, $\frac{1}{M_{\text{slot}}} \sum |\tilde{Y}_s[k]|^2 \equiv 1$ regardless of whether the physical received signal was $10^{-6}$ V or $10^6$ V!
3. **Dimensionless Relative Stabilizer**:
   Any numerical stabilizer $\epsilon$ added in denominators is scaled proportionally to $\tilde{\sigma}_w^2$:
   $$\epsilon_{\text{stable}} = 10^{-7} \cdot \max(\tilde{\sigma}_w^2, 10^{-6})$$
   ensuring complete scale invariance across any scale $r \in [2^{-12}, 2^{+12}]$.

### 5.3 Mixed-Precision Audit Protocol
- **Evaluation Formats**:
  1. `fp32`: Standard IEEE 754 single precision (baseline).
  2. `bf16`: BFloat16 mixed-precision (8 exponent bits, 7 mantissa bits) via `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`.
  3. `fp16`: Float16 mixed-precision (5 exponent bits, 10 mantissa bits) via `torch.autocast(device_type="cpu", dtype=torch.float16)`.
- **Dynamic Range Grid**:
  $r \in \{ 2^{-12}, 2^{-8}, 2^{-4}, 2^0, 2^4, 2^8, 2^{12} \}$ (spans $72$ dB of scale variation).
- **Mandatory Assertion Checks**:
  - $0$ nonfinite logits (`NaN` / `Inf`).
  - Max absolute logit deviation: $\max |L_{\text{half}} - L_{\text{fp32}}| < 0.15$.
  - Total hard bit decision flip rate $< 0.05\%$.
  - Unbroken BLER and BER performance across the entire dynamic range.

---

## 6. Topic 5: Pre-Frozen Confirmatory Protocol & Governance (Stage P4-D)

Stage P4-D serves as the final confirmatory validation stage, enforcing the scientific standards established in P1-E and P2-E.

### 6.1 Pre-Frozen Protocol & Cryptographic Integrity
1. Before running training or evaluation for Stage P4-D, the complete experimental protocol must be committed to `P4D_PROTOCOL.md`.
2. The protocol specifies:
   - Fixed list of 10 seeds.
   - Exact network architecture, checkpoint SHA256 hashes, learning rate schedules, batch sizes (64), and training steps (600).
   - Predeclared primary endpoints and statistical hypotheses.
3. The cryptographic SHA256 hash of `P4D_PROTOCOL.md` is frozen into `P4D_FREEZE.json`. Any post-hoc parameter adjustments invalidate the trial.

### 6.2 Ten Fresh Training Seeds
To prevent data snooping and seed selection bias, 10 completely fresh seeds are predeclared, disjoint from all historical runs:
- **Historical Seeds**:
  - P1-C/D: 101, 202, 303
  - P1-E: 404, 505, 606, 707, 808
  - P2-C: 1201, 1202, 1203
  - P2-E: 2201–2210
  - P3-A: 46001–46005
  - P3-A2: 47101–47105
  - P3-B: 48101–48105
  - 16QAM probe: 59101–59103
  - P3DE surrogate: 61001–61010
- **Predeclared Fresh Seeds for Stage P4-D**:
  $$\mathcal{S}_{\text{P4D}} = \{ 74001, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010 \}$$

### 6.3 Integer Error Accounting
Following P1-E and P2-E standards, floating-point rounding errors must not corrupt the statistical evidence:
- **Bit Error Count**: $E_{\text{bits}} = \sum \mathbb{I}(\hat{b} \ne b) \in \mathbb{Z}_{\ge 0}$.
- **Block Error Count**: $E_{\text{blocks}} = \sum \mathbb{I}(\hat{u} \ne u) \in \mathbb{Z}_{\ge 0}$.
- **Total Tested Bits / Blocks**: $N_{\text{bits}}, N_{\text{blocks}} \in \mathbb{Z}_{> 0}$.
- All per-seed records stored as exact integers in `p4d_results/raw_by_seed.csv`.

### 6.4 Two-Sided Sign-Flip Permutation Tests with Holm Step-Down Adjustment
- **Replication Unit**: Training seed ($N = 10$).
- **Paired Contrast**:
  For primary endpoint $m$ (e.g. BLER of EM-FloatFormer vs 2D-LMMSE or IDD-PA at 10 dB on OOD Doppler):
  $$\delta_s = \text{BLER}(\text{Baseline})_s - \text{BLER}(\text{EM-FloatFormer})_s, \quad s \in \mathcal{S}_{\text{P4D}}$$
- **Test Statistic**: $T_{\text{obs}} = \left| \sum_{s=1}^{10} \delta_s \right|$.
- **Exact Two-Sided Permutation Test**:
  Under the exchangeable null hypothesis $H_0: \delta_s \overset{d}{=} -\delta_s$, there are $2^{10} = 1024$ equally likely sign configurations $\mathbf{s} \in \{-1, +1\}^{10}$.
  $$p_{\text{exact}} = \frac{1}{1024} \sum_{\mathbf{s} \in \{-1, +1\}^{10}} \mathbb{I}\left( \left| \sum_{s=1}^{10} s_i \delta_s \right| \ge T_{\text{obs}} - 10^{-10} \right)$$
- **Holm Step-Down Adjustment**:
  For $M$ predeclared primary endpoints with sorted p-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$:
  $$p_{(k)}^{\text{Holm}} = \max_{j \le k} \min\left(1, (M - j + 1) p_{(j)}\right)$$
  Statistical significance requires $p_{(k)}^{\text{Holm}} < 0.05$.

### 6.5 Repository Directory Structure & Historical Governance
To ensure that all previous historical phases (P1–P3) remain immutable:
1. `MANIFEST_SHA256.json` records SHA256 hashes of all existing repository files. An automated pre-execution and post-execution audit script verifies that zero historical files were modified or deleted.
2. All Stage P4 deliverables must be segregated into dedicated Stage P4 directories and files:
   - Source files: `p4_*.py`
   - Test suites: `test_p4_*.py`
   - Results directories: `p4b_results/`, `p4c_results/`, `p4d_results/`
   - Protocols & Reports: `P4B_PROTOCOL.md`, `P4B_REPORT.md`, `P4C_PROTOCOL.md`, `P4C_REPORT.md`, `P4D_PROTOCOL.md`, `P4D_REPORT.md`.

---

## 7. Topic 6: Comprehensive Technical Gap Matrix (Requirements R3, R4)

The table below catalogs every technical gap between the existing repository state and the Stage P4-B/C/D requirements, defining the exact feature specifications needed for implementation:

| Module / Area | Baseline Code State | Required State for P4-B/C/D | Exact Feature Requirements & Implementation Scope |
|---|---|---|---|
| **2D Slot Link** | `p3b_time_link.py` (1 pilot + 1 payload symbol, 32 tones, static FIR) | Multi-symbol slot ($N_{\text{symb}}=14$, $N_{\text{sc}}=32$ or $64$), time-varying FIR | 1. Implement `build_slot_grid(batch, n_symb, n_sc, profile, doppler_fd, cfo_eps)`.<br>2. Vectorized time-variant sample-level FIR convolution.<br>3. Continuous sample-level phase accumulation for CFO. |
| **DMRS Inserter** | None (100% pilot symbol or 2 anchors) | 5G-NR-like sparse DMRS (symbol 2 & 11, comb-2, $\le 10\%$ overhead) | 1. Implement `insert_dmrs(grid, pilot_symbols, pattern='comb2')`.<br>2. Extract DMRS RE mask and payload data mask. |
| **16-QAM Mod/Demod** | QPSK only; toy uncalibrated 16-QAM slicer in exploratory probe | Calibrated Gray 16-QAM mapper and Max-Log soft LLR demapper | 1. Gray bit-to-symbol mapper $\mathcal{X}_{16\text{QAM}}$.<br>2. Calibrated max-log LLR demapping with Bussgang $\alpha$ and effective distortion variance $\sigma_{\text{eff}}^2$. |
| **Outer FEC & BLER** | None (uncoded BER only) | Native Rate-1/2 LDPC or Convolutional soft decoding; BLER tracking | 1. Implement self-contained PyTorch Rate-1/2 5G NR LDPC Min-Sum decoder or Convolutional Max-Log-MAP.<br>2. Add parity check verifier and exact block error counter. |
| **2D-LMMSE Baseline** | 1D frequency LMMSE on 32 subcarriers | Separable 2D Wiener filter ($R_{\text{time}} \otimes R_{\text{freq}}$) from sparse DMRS | 1. Implement separable 2D correlation matrix generator.<br>2. Implement 2D Wiener interpolation to estimate full grid $\hat{H}_{s, k}$. |
| **IDD-PA Baseline** | None (`P3B_REPORT.md:102`) | Iterative Decision-Directed PA Cancellation (2–3 iterations) | 1. Implement iterative loop reconstructing nonlinear distortion $d = g(\hat{X}) - \alpha \hat{X}$.<br>2. Subtract $H d$ and re-estimate channel using tentative payload decisions. |
| **C* Gauge Equivariance** | Broken (P3-C collapsed at $r = 2^{\pm 12}$) | Strict $C^* = \mathbb{R}_{>0} \times U(1)$ equivariance across $[2^{-12}, 2^{+12}]$ | 1. Slot RMS normalization $S_{\text{ref}} = \sqrt{\frac{1}{M} \sum \|Y\|^2}$.<br>2. Dimensionless scaled stabilizers $\epsilon = 10^{-12} \cdot \sigma_w^2 / P_y$.<br>3. Zero bias ($b = 0$) in all linear layers. |
| **Hermitian Attention** | Standard real attention in P1-C; 1D Conv in P2/P3 | Complex Linear ($W \in \mathbb{C}^{d \times d}$), Hermitian $Q K^H$, 2D Relative Phase RoPE | 1. Implement `ComplexLinear(in_dim, out_dim, bias=False)`.<br>2. Implement `HermitianPropagatorAttention` with complex relative RoPE for $(\Delta s, \Delta k)$.<br>3. Radial activation $\text{SiLU}(\|z\|) \frac{z}{\|z\|}$. |
| **Joint Pilot-Data Prop** | Disjoint (`del yd` in `p3_models.py`) | Bidirectional joint pilot-data propagation | 1. Unified 2D token embedding with pilot/data indicator tokens.<br>2. Bidirectional multi-head cross attention between pilots and data. |
| **Factorial Ablations** | $2 \times 2$ (position vs floor on 1D conv) | $2^3$ orthogonal factorial design isolating $C^*$, Green Attn, Joint Prop | 1. Modular architecture flags: `--ablate-cstar`, `--ablate-green`, `--ablate-joint`.<br>2. Paired initialization and seed-wise effect decomposition script. |
| **Numeric & Precision** | CPU real Conv1d autocast only; FFT FP32 | Verified FP32, BF16, and FP16 execution across $r \in [2^{-12}, 2^{+12}]$ | 1. Implement precision audit runner with finite/flip assertion hooks.<br>2. CPU/GPU latency benchmark harness. |
| **Confirmatory Protocol** | 5 seeds in P1E/P3B; 10 in P2E | Pre-frozen 10 fresh seeds, integer error tracking, 2-sided sign-flip + Holm | 1. Commit `P4D_PROTOCOL.md` and freeze SHA256 in `P4D_FREEZE.json`.<br>2. 10 fresh seeds ($74001-74010$).<br>3. Exact 1024-state sign-flip permutation runner with Holm correction. |
| **Historical Governance** | Ad-hoc checks | Strict read-only governance of P1–P3 records | 1. Pre- and post-run verification against `MANIFEST_SHA256.json`.<br>2. Strict file output path containment in `p4*_results/`. |

---

## 8. Recommendations for Experimental Execution

1. **Phase Staging**:
   - **Stage P4-0**: Implement the time-domain dynamic simulator, Rapp PA, Bussgang-LMMSE, and IDD-PA classical baselines. Benchmark latencies on CPU/GPU.
   - **Stage P4-A**: Implement the C*-gauge-invariant EM-Native Propagator Transformer architecture. Verify $C^*$ equivariance under unit tests before training.
   - **Stage P4-B**: Deploy 2D multi-symbol slot generation, sparse DMRS, high-Doppler ICI, CFO, 16-QAM calibrated soft demapping, and outer FEC soft decoding. Benchmark BLER against 2D-LMMSE and IDD-PA across OOD Doppler and delay spreads.
   - **Stage P4-C**: Execute the $2^3$ factorial ablation study and numeric stability / mixed-precision audit across $[2^{-12}, 2^{+12}]$.
   - **Stage P4-D**: Freeze `P4D_PROTOCOL.md` with 10 fresh seeds ($74001–74010$) and execute the final confirmatory replication with sign-flip permutation tests and Holm adjustment.
2. **Computational Feasibility**:
   - Slot size: $14 \times 32 = 448$ tokens. With 2 layers of complex attention and model width $d=32$, a single forward pass takes $\approx 1.2$ ms on CPU, easily executing within reasonable computational budgets.
   - Exact sign-flip permutation test over 10 seeds requires testing $2^{10} = 1024$ combinations, running in $< 5$ milliseconds in pure Python.
3. **Repository Integrity**:
   - Absolutely no existing files in the repository root or subdirectories must be altered. All P4 scripts must write exclusively to `p4/` or dedicated prefixed directories.
