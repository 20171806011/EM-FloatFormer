# EM-FloatFormer Stage P4-A Technical Survey Report
## Architecture, C* Equivariance, Hermitian Green's Attention & Physical Consistency

- **Date:** 2026-11-15
- **Author:** Explorer 2 (Architecture & Invariance Specialist)
- **Scope:** Technical survey and architectural specification for Milestone 2 / Stage P4-A (Requirement R2)

---

## 1. Survey of Existing Architectures in Codebase

A systematic audit was conducted across all existing neural receiver and transformer implementations from Phase 1 through Phase 3, as well as the exploratory 16-QAM non-linear distortion probe:

### 1.1 Phase 1 Baseline: `RepresentationTransformer` & `CircularCNN` (`p1b_representation.py`, `p1c_architecture.py`)
- **Structure**: Real-valued feedforward architectures operating on concatenated real features.
  - Input features (dim=11): Cartesian components $(y_r, y_i)$ and amplitude $|y|$ of pilot and data, LS-equalized symbol $z = y_d y_p^* / (|y_p|^2 + \sigma^2 + 0.01)$ components $(z_r, z_i)$, positional embeddings $(\cos \omega_k, \sin \omega_k)$, and log noise variance $\log_{10}(\sigma^2)$.
  - Transformer: PyTorch standard `nn.TransformerEncoder` (2 layers, 4 heads, $d=48$, `dim_feedforward=96`, GELU), followed by a real linear projection to 2 bit logits with a residual skip connection $3.0 \times [z_r, z_i]$.
  - CNN: 1D circular convolution along subcarriers (3 layers, kernel size 5, circular padding, LayerNorm, GELU).
  - Parameter budget: ~38,000 parameters.
- **Critical Limitations**:
  - **No $U(1)$ phase equivariance**: In `P2C_REPORT.md` and `p2_symmetry.py`, rotating received symbols by $e^{j\phi}$ resulted in logit shifts up to $6.612$ and massive bit decision flips.
  - **No scale equivariance ($\mathbb{R}_{>0}$)**: Handled dynamic range poorly; vulnerable to scale shifts.
  - **Empirical performance**: CNN outperformed the Transformer across multi-tap profiles, revealing that unconstrained real self-attention was prone to overfitting subcarrier correlations without physical inductive bias.

### 1.2 Phase 2: $U(1)$ Phase-Equivariant Receivers (`p2_models.py`, `p2_symmetry.py`)
- **`EquivComplexLocal` (2,060 parameters)**:
  - Formulated as a complex FIR filter with real kernels $w_r, w_i \in \mathbb{R}^{1 \times 1 \times 9}$ (circular convolution) and an amplitude-dependent gating network.
  - Channel update: $\hat{H} = y_p + \sigma(\text{logit\_gate}) \cdot g \odot (\text{FIR}(y_p) - y_p)$, where gate $g$ takes $[ \log(1 + |y_p|^2), \log_{10}(\sigma^2) ]$.
  - Detection: Zero-forcing / MMSE physical equalization $z = y_d \hat{H}^* / (|\hat{H}|^2 + \sigma^2 + 10^{-2})$.
  - Verified in `p2_symmetry.py`: Phase rotation $e^{j\phi}$ produces identical equalized decisions ($\Delta \text{logit} < 2.87 \times 10^{-6}$, 0 bit flips).
  - **Limitation**: Not scale-equivariant. Feature inputs like $\log(1 + |y_p|^2)$ and the fixed regularizer $+10^{-2}$ in the equalizer denominator break down when power is scaled.
- **`DelayUnfold` (1,225 parameters)**:
  - Delay-domain real Wiener-like shrinkage: $t = \mathcal{F}^{-1}(y_p)$, $P_k = |t_k|^2$.
  - Real MLP estimates a real shrinkage factor $g_k \in [0, 1]$ applied to complex delay taps: $\hat{t} = t \odot g$.
  - Strictly $U(1)$ equivariant because $|e^{j\phi} t|^2 = |t|^2$.

### 1.3 Phase 3: Delay-Domain Convolutions & Factorial Invariants (`p3_models.py`, `p3_a_mechanism.py`)
- **`EquivDelayConv` (985 parameters)**:
  - Delay-domain circular CNN computing real shrinkage $g \in [0, 1]^N$.
  - Input features: $\log(1 + P_k / \sigma_t^2)$, $\log(1 + P_k / \bar{P})$, $\log(1 + \bar{P} / \sigma_t^2)$, $\log_{10}(\sigma^2)$, and positional $\cos(2\pi k/N), \sin(2\pi k/N)$.
  - **P3-A2 Discovery**: Absolute delay positional coordinates caused severe shortcut learning. Removing them (`use_position=False`) while maintaining a retention floor (`floor=0.10`) reduced BER on out-of-distribution `sparse12` from $12.29\%$ to $6.55\%$.
  - **Limitation**: Strictly pilot-only. It explicitly discards data: `del yd`. Cannot model data-dependent nonlinear PA distortion.

### 1.4 Phase 3-B & Exploratory 16-QAM Joint Demapper (`p3b_receivers.py`, `exploratory/16qam_joint_pa_probe/probe.py`)
- **`DataResidualHead` (`p3b_receivers.py`)**:
  - Downstream 1D CNN head receiving $y_d$, channel $\hat{H}$ (from frozen backbone), and noise variance. Unidirectional information flow ($\text{Pilot} \to \text{Channel} \to \text{Data Head}$).
  - Negative result in P3-B: Did not beat classical DFT-8 or LMMSE baselines in any evaluated scenario.
- **`TinyJoint` (`exploratory/16qam_joint_pa_probe/probe.py`, 1,812–2,600 parameters)**:
  - First attempt at joint 16-QAM demapping under memoryless Rapp PA distortion ($A=1.25$ strong, $A=2.25$ mild).
  - Small circular CNN taking equalized symbol $z = y_d \hat{h}^* / (|\hat{h}|^2 + \sigma^2 + 0.03)$, raw $y_p, y_d$, and pilot channel estimate $\hat{h}$.
  - Outputs 4 bit logits via heuristic base + delta: $\text{base}(z) \times 5.0 + 0.5 \Delta$.
  - Defeated by a classical training-calibrated decision threshold; lacked probability-calibrated LLR outputs and possessed no $U(1)$ or scale equivariance.

### 1.5 Summary of Existing Architectural Spectrum
| Architecture | Complexity | Domain | $U(1)$ Phase Equiv | $\mathbb{R}_{>0}$ Scale Equiv | Data-Pilot Bidirectional | 16-QAM Calibrated |
|---|---|---|:---:|:---:|:---:|:---:|
| `RepresentationTransformer` (P1) | ~38k params | Frequency (Real) | ❌ | ❌ | Partial (concat) | ❌ (QPSK only) |
| `CircularCNN` (P1) | ~38k params | Frequency (Real) | ❌ | ❌ | Partial (concat) | ❌ (QPSK only) |
| `EquivComplexLocal` (P2) | 2,060 params | Frequency (Complex) | ✅ | ❌ | ❌ (Pilot only) | ❌ (QPSK only) |
| `DelayUnfold` (P2) | 1,225 params | Delay (Complex) | ✅ | ❌ | ❌ (Pilot only) | ❌ (QPSK only) |
| `FactorialDelayConv` (P3-A2) | 985 params | Delay (Complex) | ✅ | ❌ | ❌ (Pilot only) | ❌ (QPSK only) |
| `DataResidualHead` (P3-B) | ~9.5k params | Frequency (Real) | ❌ | ❌ | ❌ (Unidirectional) | ❌ (QPSK only) |
| `TinyJoint` (Exploratory) | 1,812 params | Frequency (Real) | ❌ | ❌ | Partial (1D CNN) | ❌ (Uncalibrated) |
| **EM-Native Propagator** (P4-A Target) | ~8k–15k params | Complex Dual-Domain | **Strict ✅** | **Strict ✅** | **Bidirectional ✅** | **Calibrated ✅** |

---

## 2. Mathematical Formulation & Implementation of $C^*$ Equivariance

### 2.1 The Gauge Group $C^* = \mathbb{R}_{>0} \times U(1)$
Let $\mathbb{C}^* = \mathbb{C} \setminus \{0\}$. Any non-zero complex scalar $c \in \mathbb{C}^*$ uniquely factorizes as:
$$c = r \cdot e^{j\phi}, \quad r \in \mathbb{R}_{>0}, \quad e^{j\phi} \in U(1)$$
In an electromagnetic induction / radio communication channel, if the overall channel attenuation, antenna gain, or AGC changes by $r$, and the local oscillator or carrier phase shifts by $\phi$, the physical received frequency-domain signals and noise variance transform according to the group action:
$$g \cdot (Y_p, Y_d, \sigma^2) = \left(c Y_p, \, c Y_d, \, |c|^2 \sigma^2\right) = \left(r e^{j\phi} Y_p, \, r e^{j\phi} Y_d, \, r^2 \sigma^2\right)$$

An equivariant neural receiver must satisfy two physical symmetry laws:
1. **Channel Equivariance**:
   $$\Phi_H\left(c Y_p, c Y_d, |c|^2 \sigma^2\right) = c \cdot \Phi_H\left(Y_p, Y_d, \sigma^2\right)$$
2. **Bit Logit Invariance**:
   $$\Phi_L\left(c Y_p, c Y_d, |c|^2 \sigma^2\right) = \Phi_L\left(Y_p, Y_d, \sigma^2\right)$$

### 2.2 Exact $C^*$ Gauge Normalization Mechanism
To eliminate scale drift and catastrophic failure under wide dynamic ranges ($r \in [2^{-12}, 2^{+12}]$), gauge normalization is applied at both global and layer levels:

1. **Global Pilot Energy Gauge Anchor**:
   At the network entry point, compute the gauge scalar $s_0 \in \mathbb{R}_{>0}$ strictly from the received pilot symbol:
   $$s_0 = \sqrt{\frac{1}{N} \sum_{k=1}^N |Y_p[k]|^2}$$
   Under transformation $(Y_p, Y_d, \sigma^2) \mapsto (c Y_p, c Y_d, |c|^2 \sigma^2)$, $s_0$ transforms as $s_0 \mapsto |c| s_0 = r s_0$.
   Define dimensionless gauge-normalized inputs:
   $$\bar{Y}_p = \frac{Y_p}{s_0}, \quad \bar{Y}_d = \frac{Y_d}{s_0}, \quad \bar{\sigma}^2 = \frac{\sigma^2}{s_0^2}$$
   - Notice: $\bar{\sigma}^2$ is **strictly invariant** to $c$: $\frac{|c|^2 \sigma^2}{(r s_0)^2} = \frac{r^2 \sigma^2}{r^2 s_0^2} = \frac{\sigma^2}{s_0^2}$.
   - $\bar{Y}_p$ and $\bar{Y}_d$ have RMS power exactly equal to 1, and transform purely under $U(1)$: $\bar{Y} \mapsto e^{j\phi} \bar{Y}$.
   The physical channel estimate is recovered at the output via:
   $$\hat{H} = s_0 \cdot \bar{H}, \quad \text{where } \bar{H} = \bar{\Phi}_H\left(\bar{Y}_p, \bar{Y}_d, \bar{\sigma}^2\right)$$
   Since $\bar{H}(e^{j\phi} \bar{Y}_p, e^{j\phi} \bar{Y}_d, \bar{\sigma}^2) = e^{j\phi} \bar{H}$, we have:
   $$\hat{H}' = (r s_0) \cdot (e^{j\phi} \bar{H}) = c \cdot \hat{H}$$
   This guarantees exact global $\mathbb{C}^*$ equivariance.

2. **Internal Complex RMSNorm**:
   For hidden layer representations $Z \in \mathbb{C}^{B \times N \times D}$:
   $$\text{RMS}(Z)_{b,n} = \sqrt{\frac{1}{D} \sum_{d=1}^D |Z_{b,n,d}|^2 + \epsilon_{\text{rel}} \cdot \langle |Z|^2 \rangle}$$
   where $\epsilon_{\text{rel}}$ is a dimensionless relative tolerance (e.g. $10^{-6}$), preserving exact scale homogeneity.
   With learnable real radial channel gain $\boldsymbol{\gamma} \in \mathbb{R}^D$:
   $$\text{CRMSNorm}(Z) = \left( \frac{Z}{\text{RMS}(Z)} \right) \odot \boldsymbol{\gamma}$$
   Under $Z \mapsto c Z$, $\text{RMS}(c Z) = |c| \text{RMS}(Z) = r \text{RMS}(Z)$, hence:
   $$\text{CRMSNorm}(c Z) = \frac{r e^{j\phi} Z}{r \text{RMS}(Z)} \odot \boldsymbol{\gamma} = e^{j\phi} \text{CRMSNorm}(Z)$$
   This normalizes magnitude while preserving exact phase rotation.

### 2.3 Hermitian Green's Propagator Attention
The attention mechanism is formulated to mirror the physical Green's function propagator of electromagnetic wave induction across subcarriers:
$$G(m, n) = \sum_l \alpha_l e^{-j 2\pi (m - n) \Delta f \tau_l}$$
Key structural requirements:

1. **Complex Linear Projections Without Bias**:
   For token tensor $X \in \mathbb{C}^{B \times N \times D_{\text{in}}}$, complex linear projections for Query, Key, and Value matrices are defined with $W = W_r + j W_i \in \mathbb{C}^{D_{\text{out}} \times D_{\text{in}}}$:
   $$Q = X W_Q^T, \quad K = X W_K^T, \quad V = X W_V^T, \quad \text{with } b \equiv 0$$
   Explicitly in real arithmetic:
   $$\text{Re}(Q) = \text{Re}(X) W_{Q,r}^T - \text{Im}(X) W_{Q,i}^T, \quad \text{Im}(Q) = \text{Re}(X) W_{Q,i}^T + \text{Im}(X) W_{Q,r}^T$$
   **Zero-bias condition ($b=0$)**: A non-zero additive bias breaks both phase and scale equivariance ($W(c X) + b \ne c(WX + b)$). With $b=0$, complex linearity is strict: $W(c X) = c W X$ for all $c \in \mathbb{C}^*$.

2. **Complex Relative-Phase RoPE (Rotary Position Embedding)**:
   In OFDM, inter-carrier coupling is a function of subcarrier frequency separation $(m - n)\Delta f$.
   For head channel dimension $d \in \{0, \dots, D_h - 1\}$, assign rotary base frequencies $\theta_d = \theta_0^{-2d / D_h}$.
   The position phase for subcarrier index $m$ is $\psi_{m,d} = m \cdot \theta_d$.
   Applying complex RoPE to query and key is complex phasor multiplication:
   $$\tilde{Q}_{m,d} = Q_{m,d} \cdot e^{j \psi_{m,d}}, \quad \tilde{K}_{n,d} = K_{n,d} \cdot e^{j \psi_{n,d}}$$
   The Hermitian inner product between query at subcarrier $m$ and key at subcarrier $n$ is:
   $$\langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}} = \sum_{d=1}^{D_h} \tilde{Q}_{m,d} \tilde{K}_{n,d}^* = \sum_{d=1}^{D_h} Q_{m,d} K_{n,d}^* \cdot e^{j(m - n)\theta_d}$$
   - **Phase Cancellation**: If received tokens rotate by $e^{j\phi}$, $Q \mapsto e^{j\phi} Q$ and $K \mapsto e^{j\phi} K$. The inner product satisfies:
     $$\langle \tilde{Q}'_m, \tilde{K}'_n \rangle_{\mathbb{C}} = (e^{j\phi})(e^{-j\phi}) \langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}} = \langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}}$$
     The global phase strictly cancels!
   - **Relative Phase Invariance**: The spatial relationship depends solely on the relative index $(m - n)$, preserving shift invariance across frequency bands.

3. **Hermitian Propagator Attention Matrix**:
   The scalar attention affinity $S_{m,n}$ is obtained from the real part of the complex inner product:
   $$S_{m,n} = \frac{1}{\sqrt{D_h}} \text{Re}\left( \langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}} \right)$$
   For self-attention with shared query/key weights ($W_Q = W_K$), $S_{m,n} = S_{n,m}$, reflecting the physical reciprocity of Green's functions ($G(m, n) = G^*(n, m)$).
   Softmax is computed over the key dimension $n$:
   $$A_{m,n} = \frac{\exp(S_{m,n})}{\sum_{l=1}^N \exp(S_{m,l})}$$
   Since $S_{m,n}$ is strictly invariant to $c \in \mathbb{C}^*$, the attention weights $A_{m,n} \in [0, 1]$ are strictly invariant.
   The aggregated output is:
   $$O_m = \sum_{n=1}^N A_{m,n} V_n \implies O_m \mapsto \sum_{n=1}^N A_{m,n} (c V_n) = c O_m$$
   Subsequent output projection $W_O$ (complex linear without bias) preserves $c$-equivariance.

4. **$U(1)$-Preserving Radial Activation Function**:
   Standard elementwise activations (GELU, SiLU) operate on real coordinates independently, warping the complex phase angle $\arg(z)$.
   To maintain $U(1)$ equivariance, activations must act along the radial magnitude:
   $$f_{\text{radial}}(z) = g(|z|) \cdot \frac{z}{|z| + \epsilon_{\text{rel}} \cdot \bar{r}}$$
   where $g: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ is a real-valued non-linear scaling function (e.g. $g(u) = \text{SiLU}(u)$ or $g(u) = u \cdot \sigma(u - \theta)$).
   Equivariance check:
   $$f_{\text{radial}}\left(e^{j\phi} z\right) = g\left(|e^{j\phi} z|\right) \cdot \frac{e^{j\phi} z}{|e^{j\phi} z|} = g(|z|) \frac{z}{|z|} e^{j\phi} = e^{j\phi} f_{\text{radial}}(z)$$
   By placing Complex RMSNorm before $f_{\text{radial}}$, the magnitude $|z|$ is dimensionless and scale-free, ensuring full $\mathbb{C}^*$ equivariance.

### 2.4 Bidirectional Pilot-Data Attention
Under nonlinear PA distortion (Rapp AM-AM/AM-PM), frequency-domain signals experience Bussgang attenuation $\alpha$ and inter-carrier distortion $D_{\text{nonlinear}}$:
$$Y_d = H \odot (\alpha X_d + D_{\text{nonlinear}}) + Z$$
- **Pilot $\to$ Data (Channel & Phase Propagation)**:
  Pilots provide anchor phase and channel delay profile estimates via Green's propagator attention, allowing data tokens to be equalized without noise enhancement.
- **Data $\to$ Pilot (Distortion & ICI Cancellation)**:
  Tentative demapped data symbols $\hat{X}_d$ predict the nonlinear time-domain envelope $\hat{x}_{\text{time}} = \mathcal{F}^{-1}(\hat{X}_d)$ and its PA distortion $\hat{D}_{\text{nonlinear}}$.
  Cross-attention from data tokens to pilot tokens feeds back the estimated nonlinear interference, enabling the channel estimator to subtract out distortion leaking into pilot subcarriers.
- **Dual-Stream Propagator Block**:
  Let $H^{(l)} \in \mathbb{C}^{N \times D}$ and $X_d^{(l)} \in \mathbb{C}^{N \times D}$ be the layer-$l$ channel and data representations:
  $$H^{(l+1)} = \text{Propagate}\left(H^{(l)}, \, \text{CrossAttn}(H^{(l)}, X_d^{(l)})\right)$$
  $$X_d^{(l+1)} = \text{Propagate}\left(X_d^{(l)}, \, \text{CrossAttn}(X_d^{(l)}, H^{(l+1)})\right)$$
  This iterative bidirectional exchange forms a fully differentiable analog to classical Iterative Decision-Directed PA Cancellation.

---

## 3. Joint Physical Consistency Objective

Training must not optimize solely for uncalibrated bit decisions. A three-component joint physical consistency objective is formulated:

### 3.1 Calibrated 16-QAM LLR BCE Loss ($\mathcal{L}_{\text{BCE}}$)
For 16-QAM constellation with Gray labeling, each subcarrier $k$ conveys 4 bits $b_{k} \in \{0, 1\}^4$.
The network outputs real Log-Likelihood Ratios: $L_{k,i} \in \mathbb{R}$ ($i=0,1,2,3$).
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{4N} \sum_{k=1}^N \sum_{i=0}^3 \left[ b_{k,i} \ln \sigma(L_{k,i}) + (1 - b_{k,i}) \ln (1 - \sigma(L_{k,i})) \right]$$
- **Calibration Condition**: Soft decoders require true log-posterior ratios:
  $$L_i = \ln \frac{\mathbb{P}(b_i = 1 | Y)}{\mathbb{P}(b_i = 0 | Y)}$$
  The output demapper layer is parameterized as a scaled distance to learned constellation centroids:
  $$L_{k,i} = \frac{1}{\tau(\bar{\sigma}^2)} \cdot \Delta_{k,i}(z_k)$$
  where $\tau(\bar{\sigma}^2)$ is a calibrated, noise-dependent temperature scale. Expected Calibration Error (ECE) is audited during validation to verify probability alignment.

### 3.2 Normalized Channel NMSE Loss ($\mathcal{L}_{\text{NMSE}}$)
The model outputs an explicit physical channel estimate $\hat{H} \in \mathbb{C}^N$.
The normalized mean squared error loss is defined as:
$$\mathcal{L}_{\text{NMSE}} = \frac{\|H_{\text{true}} - \hat{H}\|_2^2}{\|H_{\text{true}}\|_2^2 + \epsilon_H} = \frac{\sum_{k=1}^N |H_{\text{true}}[k] - \hat{H}[k]|^2}{\sum_{k=1}^N |H_{\text{true}}[k]|^2 + \epsilon_H}$$
- **Scale Invariance Property**: Under arbitrary channel gain $r \in [2^{-12}, 2^{+12}]$, both numerator and denominator scale as $r^2$. Thus $\mathcal{L}_{\text{NMSE}}$ and its parameter gradients are strictly scale-invariant, eliminating gradient explosion or vanishing across extreme SNR/gain ranges.

### 3.3 PA Residual Consistency Loss ($\mathcal{L}_{\text{PA-res}}$)
Without privileged channel access at inference, physical consistency is enforced by matching the received data against the forward model:
1. Reconstruct soft transmitted constellation symbols from calibrated LLRs:
   $$\hat{X}_d[k] = \sum_{s \in \mathcal{S}_{16}} s \cdot \mathbb{P}(X_k = s | L_k)$$
2. Synthesize time-domain signal and apply memoryless Rapp PA model:
   $$\hat{x}_{\text{time}} = \text{IFFT}(\hat{X}_d), \quad \hat{x}_{\text{PA}} = f_{\text{Rapp}}(\hat{x}_{\text{time}}), \quad \hat{X}_{\text{PA}} = \text{FFT}(\hat{x}_{\text{PA}})$$
3. Compute normalized residual between received data and reconstructed forward response:
   $$\mathcal{L}_{\text{PA-res}} = \frac{\|Y_d - \hat{H} \odot \hat{X}_{\text{PA}}\|_2^2}{\|Y_d\|_2^2 + \epsilon_Y}$$

### 3.4 Delay-Domain Support Regularizer ($\mathcal{L}_{\text{delay}}$)
To enforce the physical multipath prior that channel impulse responses must lie within the cyclic prefix length $L_{\text{CP}} = 16$:
$$\hat{h}_{\text{time}} = \text{IFFT}(\hat{H}), \quad \mathcal{L}_{\text{delay}} = \frac{\sum_{k = L_{\text{CP}}}^{N-1} |\hat{h}_{\text{time}}[k]|^2}{\sum_{k=0}^{N-1} |\hat{h}_{\text{time}}[k]|^2 + \epsilon_h}$$

### 3.5 Total Joint Objective
$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{NMSE}} \mathcal{L}_{\text{NMSE}} + \lambda_{\text{PA}} \mathcal{L}_{\text{PA-res}} + \lambda_{\text{delay}} \mathcal{L}_{\text{delay}}$$
Nominal weighting hyperparameters: $\lambda_{\text{NMSE}} = 1.0$, $\lambda_{\text{PA}} = 0.25$, $\lambda_{\text{delay}} = 0.10$.

---

## 4. Equivariance Unit Testing Methodology

To satisfy the Acceptance Criterion:
> *"Unit tests verify strict C* = R_{>0} x U(1) equivariance of EM-Native Propagator Attention, demonstrating numerical invariance under arbitrary channel gain scaling (r in [2^-12, 2^+12]) and random phase rotation."*

A standalone, deterministic test suite (`test_p4a_equivariance.py`) is specified.

### 4.1 Test Vector Generation
For batch size $B=64$, generate test frames across channel profiles (`train3`, `equal4`, `long6`, `sparse12`):
1. Sample scale exponents $u \sim \text{Uniform}(-12, +12)$ and compute $r = 2^u$.
2. Sample phase angles $\phi \sim \text{Uniform}(-\pi, \pi)$.
3. Form transformation scalar $c = r \cdot e^{j\phi} \in \mathbb{C}^*$.
4. Transform inputs:
   $$Y_p' = c \cdot Y_p, \quad Y_d' = c \cdot Y_d, \quad {\sigma'}^2 = |c|^2 \sigma^2 = r^2 \sigma^2$$

### 4.2 Numerical Equivariance Metrics & Invalidation Thresholds
Execute forward pass on both original and transformed inputs:
$$(\hat{H}, L) = \Phi(Y_p, Y_d, \sigma^2), \quad (\hat{H}', L') = \Phi(Y_p', Y_d', {\sigma'}^2)$$

| Target Observable | Theoretical Relation | Metric Formulation | Numerical Assertion (FP32) |
|---|---|---|---|
| **Channel Estimate $\hat{H}$** | $\hat{H}' = c \cdot \hat{H}$ | $\epsilon_H = \frac{\|\hat{H}' - c \hat{H}\|_\infty}{|c| \cdot (\|\hat{H}\|_\infty + 10^{-12})}$ | $\epsilon_H < 1.0 \times 10^{-5}$ |
| **Bit LLRs $L$** | $L' = L$ | $\epsilon_L = \|L' - L\|_\infty$ | $\epsilon_L < 1.0 \times 10^{-5}$ |
| **Hard Bit Decisions** | $\hat{b}' = \hat{b}$ | $N_{\text{flip}} = \sum |(L' > 0) \ne (L > 0)|$ | **$N_{\text{flip}} \equiv 0$ (Zero flips)** |
| **Attention Matrix $A$** | $A' = A$ | $\epsilon_A = \|A' - A\|_\infty$ | $\epsilon_A < 1.0 \times 10^{-6}$ |
| **Complex RMSNorm** | $\text{Norm}(c Z) = e^{j\phi} \text{Norm}(Z)$ | $\epsilon_{\text{norm}} = \frac{\|\text{Norm}(c Z) - e^{j\phi} \text{Norm}(Z)\|_\infty}{\|\text{Norm}(Z)\|_\infty + 10^{-12}}$ | $\epsilon_{\text{norm}} < 1.0 \times 10^{-6}$ |
| **Complex Linear** | $W(c X) = c W(X)$ | $\epsilon_W = \frac{\|W(c X) - c W(X)\|_\infty}{|c| \cdot (\|W(X)\|_\infty + 10^{-12})}$ | $\epsilon_W < 1.0 \times 10^{-6}$ |
| **Complex RoPE** | $\text{RoPE}(c Q) = c \text{RoPE}(Q)$ | $\epsilon_{\text{RoPE}} = \frac{\|\text{RoPE}(c Q) - c \text{RoPE}(Q)\|_\infty}{|c| \cdot (\|\text{RoPE}(Q)\|_\infty + 10^{-12})}$ | $\epsilon_{\text{RoPE}} < 1.0 \times 10^{-6}$ |
| **Radial Activation** | $f(e^{j\phi} z) = e^{j\phi} f(z)$ | $\epsilon_{\text{act}} = \frac{\|f(e^{j\phi} z) - e^{j\phi} f(z)\|_\infty}{\|f(z)\|_\infty + 10^{-12}}$ | $\epsilon_{\text{act}} < 1.0 \times 10^{-6}$ |

---

## 5. Technical Gaps & Exact Feature Requirements for Stage P4-A (Requirement R2)

### 5.1 Technical Gap Inventory
1. **Core Complex Transformer Modules Missing**:
   - PyTorch contains no native `ComplexLinear` without bias, no complex relative RoPE, and no complex attention layers.
   - All previous transformer experiments in this repository (`p1b_representation.py`, `p1c_architecture.py`) used real-valued abstractions that completely lack $C^*$ equivariance.
2. **Scale Equivariance ($\mathbb{R}_{>0}$) Absent Across All Historical Models**:
   - `EquivComplexLocal` and `DelayUnfold` from P2 were $U(1)$ equivariant only. Under $r = 2^{-12}$ and $r = 2^{+12}$, they failed catastrophically due to fixed stabilizer epsilons and un-normalized power features.
   - P4-A must implement exact pilot gauge anchoring and internal complex RMSNorm.
3. **Absence of Calibrated 16-QAM LLR Demapper**:
   - Codebase currently only supports hard QPSK sign slicing ($z_r > 0, z_i > 0$) or the heuristic, uncalibrated 16-QAM base+delta in `TinyJoint`.
   - A calibrated soft LLR head with noise-dependent temperature scaling must be implemented.
4. **Unidirectional vs. Bidirectional Interaction**:
   - P3-A discarded data (`del yd`); P3-B passed channel to data unidirectionally.
   - Bidirectional cross-attention coupling channel refinement and PA distortion cancellation is completely unbuilt.
5. **Absence of Multi-Task Physical Consistency Loss**:
   - Models currently train purely on bit BCE. The normalized NMSE, PA residual consistency, and delay support losses need complete differentiable implementations.
6. **Lack of Automated $C^*$ Equivariance Unit Tests**:
   - Existing `p2_symmetry.py` only tests 3 phase angles at $r=1.0$. A complete unit test suite covering $r \in [2^{-12}, 2^{+12}]$ and random phases must be constructed.

### 5.2 Exact Feature Specifications for Stage P4-A Implementation
To implement Milestone 2 / Stage P4-A cleanly and modularly, the following deliverables are required:

1. **`p4_complex_ops.py`**:
   - `ComplexLinear(in_features, out_features, bias=False)`: Complex matrix multiplication without bias.
   - `ComplexRMSNorm(dim, eps=1e-6)`: Scale-invariant, phase-preserving RMS normalization.
   - `ComplexRoPE(head_dim, n_sc)`: Rotary position embeddings applied via complex multiplication.
   - `RadialActivation(kind='silu')`: $U(1)$-equivariant radial scaling activation $g(|z|) \frac{z}{|z|}$.
2. **`p4_propagator_transformer.py`**:
   - `HermitianGreenAttention(embed_dim, num_heads)`: Complex multi-head attention computing real symmetric affinity scores from complex inner products.
   - `BidirectionalPropagatorBlock`: Paired pilot and data streams with self-attention and cross-attention.
   - `EMNativePropagatorTransformer`: Full architecture integrating gauge normalization, $L=2$ or $3$ propagator blocks, channel output head $\hat{H}$, and calibrated 16-QAM soft LLR demapping head.
   - Target parameter budget: Parameter-matched to ~12,000–16,000 parameters.
3. **`p4_losses.py`**:
   - `CalibratedLLRBCELoss`: Multi-bit binary cross-entropy with temperature calibration.
   - `NormalizedNMSELoss`: Scale-invariant channel NMSE $\frac{\|H - \hat{H}\|^2}{\|H\|^2 + \epsilon}$.
   - `PAResidualConsistencyLoss`: Differentiable forward PA synthesis and received data residual calculation.
   - `JointPhysicalConsistencyLoss`: Weighted sum of all three objectives plus delay support constraint.
4. **`test_p4a_equivariance.py`**:
   - Automated unit tests executing the protocol described in Section 4 across $r \in [2^{-12}, 2^{+12}]$ and $\phi \in [-\pi, \pi)$, asserting zero bit flips and errors $< 10^{-5}$.
5. **`p4_baselines_comparative.py`**:
   - Parameter-matched `EquivComplexLocal_16QAM` (local convolution baseline).
   - Parameter-matched `TinyJoint_16QAM` (exploratory CNN baseline).
   - Comparative evaluation framework evaluating under clean and Rapp PA ($A=1.25, 2.25$) conditions.
6. **Factorial Ablation Protocol**:
   - Plan factorial ablations isolating:
     - Factor 1: $C^*$ gauge normalization (Enabled vs. Disabled / Un-normalized).
     - Factor 2: Attention type (Hermitian Green's Propagator Attention vs. Local Equivariant Conv vs. Real Transformer).
     - Factor 3: Pilot-data interaction (Bidirectional vs. Unidirectional Pilot-only vs. Decoupled).

---
*Report completed by Explorer 2.*
