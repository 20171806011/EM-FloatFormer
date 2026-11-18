# Handoff Report — Explorer 2 (Architecture & Invariance Specialist)

- **Date:** 2026-11-15T23:01:00Z
- **Author:** Explorer 2 (`teamwork_explorer`)
- **Deliverables:**
  - `report.md`: Detailed architecture survey, mathematical formulations, loss definitions, equivariance unit testing methodology, and Stage P4-A gap analysis.
  - `handoff.md`: 5-component self-contained handoff.

---

## 1. Observation

1. **Phase 1 Real Transformer Architecture**:
   - In `p1b_representation.py:53-80` and `p1c_architecture.py:59-64`, `RepresentationTransformer` concatenates 11 real features:
     ```python
     feature = torch.cat((
         iq_features(yp, self.encoding), iq_features(yd, self.encoding),
         torch.stack((z.real, z.imag, ang.cos(), ang.sin(), nv.log10()), dim=-1)
     ), dim=-1)
     ```
     and processes them via standard real PyTorch `nn.TransformerEncoderLayer(d_model=48, nhead=4, activation='gelu')`.
   - In `P2C_REPORT.md:11`, under common phase rotation $e^{j\phi}$, P1 CNN/log and Transformer exhibited a maximum logit shift of $6.612$, demonstrating a total lack of phase symmetry.
2. **Phase 2 $U(1)$ Phase-Equivariant FIR and Failure under Scale**:
   - In `p2_models.py:111-137`, `EquivComplexLocal` achieved $U(1)$ phase equivariance using complex circular convolutions and real gating:
     ```python
     ar = F.conv1d(a, self.wr) - F.conv1d(b, self.wi)
     ai = F.conv1d(a, self.wi) + F.conv1d(b, self.wr)
     k = yp.abs().square()
     feat = torch.stack((torch.log1p(k), nv.log10().expand_as(k)), dim=1)
     ```
     The gating feature $\log(1 + |y_p|^2)$ and equalization regularizer $+10^{-2}$ depend on un-normalized power.
   - In `P3C_REPORT.md:58-69`, artificial coherent gain stress test ($r \in [2^{-12}, 2^{+12}]$) demonstrated sharp collapse even under FP32:
     At $2^{-12}$, BER was $49.961\%$; at $2^{+12}$, BER was $54.887\%$ (chance level / total failure), proving that $U(1)$ phase equivariance without $\mathbb{R}_{>0}$ scale equivariance fails across dynamic ranges.
3. **Phase 3 Delay Convolution & Invariant Shortcuts**:
   - In `p3_models.py:30-55`, `EquivDelayConv` computes delay shrinkage $g \in [0, 1]$ applied as $\hat{H} = \mathcal{F}(\mathcal{F}^{-1}(y_p) \odot g)$. It explicitly drops data: `del yd`.
   - In `p3_a_mechanism.py:58-106`, factorial ablation of `EquivDelayConv` demonstrated that removing absolute positional inputs $\cos(2\pi k/N), \sin(2\pi k/N)$ improved BER on `sparse12` from $12.29\%$ to $6.55\%$, proving that absolute frequency/delay positional shortcuts harm generalization.
4. **Phase 3-B Residual Head & Exploratory 16-QAM Demapper**:
   - In `p3b_receivers.py:54-77`, `DataResidualHead` only operates unidirectionally ($\hat{H} \to y_d$ residual), and in `P3B_REPORT.md:21-44`, failed to beat the classical baseline envelope in all 135 scenario cells.
   - In `exploratory/16qam_joint_pa_probe/probe.py:83-103`, `TinyJoint` performs 16-QAM demapping using a 1,812-parameter circular CNN atop DFT-8 channel estimates. It outputs uncalibrated binary logits via `base + 0.5 * delta`, and in `exploratory/16qam_joint_pa_probe/REPORT.md:23-35`, was defeated by a classical training-calibrated decision threshold (18.609% vs 18.800% at strong PA).
5. **Nonlinear PA Model**:
   - In `p3b_time_link.py:93-97` and `exploratory/16qam_joint_pa_probe/probe.py:34-39`, Rapp memoryless PA ($p=2$) is implemented as:
     ```python
     def rapp(x, severity):
         A = (2.0 if severity == 'mild' else 1.2) / math.sqrt(N)
         return x / (1 + (x.abs() / A)**4).pow(0.25)
     ```

---

## 2. Logic Chain

1. **Step 1 (Root Cause of Scale Failure)**:
   Observation 1 and Observation 2 prove that all previous models in the repository are completely vulnerable to scale variations because they include un-normalized power features ($\log(1 + |y|^2)$), un-scaled noise variance features ($\log_{10}(\sigma^2)$), and fixed additive constants ($+0.01$ or $+10^{-7}$). When the channel gain scales by $r \in [2^{-12}, 2^{+12}]$, signals either get dwarfed by stabilizer constants or saturate non-linearities, resulting in random chance-level decoding (BER $\approx 50\%$).
2. **Step 2 (Necessity of Exact $C^*$ Gauge Normalization)**:
   The group of non-zero complex scalar transformations is $C^* = \mathbb{R}_{>0} \times U(1)$. To ensure invariance across 24 octaves of dynamic range ($2^{-12}$ to $2^{+12}$) and arbitrary phase shifts, the receiver must be gauge-normalized by an input reference scale $s_0 = \sqrt{\frac{1}{N}\sum |Y_p|^2}$ and internal Complex RMSNorm. This removes the scale $r$ before non-linear processing and maps the signal to a scale-free unit-RMS manifold where only $U(1)$ phase rotates.
3. **Step 3 (Mathematical Structure of Hermitian Green's Attention)**:
   In electromagnetics, subcarrier interaction is dictated by the Green's function propagator $G(m, n) = \sum_l \alpha_l e^{-j 2\pi (m-n)\Delta f \tau_l}$.
   To maintain $C^*$ equivariance while implementing this physical mechanism:
   - Linear projections must be Complex Linear without bias ($b=0$), ensuring $W(c X) = c W(X)$.
   - Position must be encoded as relative phase difference via Complex RoPE ($e^{j(m-n)\theta}$), avoiding absolute positional shortcuts (Observation 3).
   - Attention affinities must be real symmetric projections of the complex inner product $\text{Re}(\langle \tilde{Q}_m, \tilde{K}_n \rangle_{\mathbb{C}})$, causing global phase $e^{j\phi}$ to cancel out completely and guaranteeing invariant attention weights.
   - Activations must be radial ($f(z) = g(|z|) \frac{z}{|z|}$), preserving complex phase angles $\arg(z)$.
4. **Step 4 (Necessity of Bidirectional Interaction)**:
   Under nonlinear PA distortion (Observation 5), transmitted data creates severe inter-carrier interference and distortion $D_{\text{nonlinear}}$ that spills into pilot subcarriers. A unidirectional receiver (Observation 4) cannot cancel this distortion. Bidirectional pilot-data cross-attention enables data tokens to feedback estimated clipping distortion to the pilot channel estimator, and the refined channel estimator to feedforward equalized channel references to data demapping.
5. **Step 5 (Joint Physical Consistency Loss)**:
   Training solely on bit BCE (Observation 4) allows internal representations to drift in scale and produce uncalibrated logits that degrade downstream soft channel decoding. Enforcing a joint objective with normalized channel NMSE ($\mathcal{L}_{\text{NMSE}}$), PA residual consistency ($\mathcal{L}_{\text{PA-res}}$), and calibrated LLR BCE ensures both physical fidelity and optimal soft decisions.

---

## 3. Caveats

- **Scope boundaries**: This survey is strictly read-only and analytical. No code implementation or training was executed during this survey turn.
- **Hardware assumptions**: As verified in `P3C_REPORT.md:19-22`, native CPU PyTorch 2.10 does not support `torch.fft` on `torch.float16` or `torch.bfloat16`. All complex operations in P4-A must run in `torch.complex64` (FP32 real components), while mixed precision remains restricted to real convolutions/linear layers.
- **Rapp PA scope**: The current codebase only implements memoryless Rapp PA (AM-AM distortion). If AM-PM phase distortion or memory effects are introduced in Milestone 1 (R1), the PA forward synthesis model in $\mathcal{L}_{\text{PA-res}}$ must match that parameterization.

---

## 4. Conclusion

1. **Existing Architectural State**: The repository transitions from non-equivariant real-valued models (P1) to $U(1)$-phase equivariant delay/frequency convolutions (P2/P3) and an exploratory 16-QAM CNN (TinyJoint). None of these architectures possess scale equivariance ($\mathbb{R}_{>0}$), complex attention, or bidirectional pilot-data interaction.
2. **Stage P4-A Specifications Defined**:
   - The architecture for Milestone 2 / Stage P4-A must be an **EM-Native Propagator Transformer** featuring:
     1. Exact $C^*$ gauge normalization (pilot RMS gauge anchor + complex RMSNorm).
     2. Hermitian Green's Propagator Attention (Complex Linear without bias, complex relative-phase RoPE, and $U(1)$-preserving radial activation).
     3. Bidirectional pilot-data cross-attention.
     4. Calibrated 16-QAM soft LLR demapper.
   - The training objective must be the **Joint Physical Consistency Objective**:
     $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{NMSE}} + \lambda_2 \mathcal{L}_{\text{PA-res}} + \lambda_3 \mathcal{L}_{\text{delay}}$.
3. **Equivariance Verification Protocol Established**: Unit tests must verify that scaling received signals by $r \in [2^{-12}, 2^{+12}]$ and rotating by $\phi \in [-\pi, \pi)$ yields channel relative error $< 10^{-5}$, logit error $< 10^{-5}$, and **strictly zero hard bit flips** ($N_{\text{flip}} \equiv 0$).

---

## 5. Verification Method

To independently verify the facts and conclusions documented in this report:

1. **Inspect Historical Scale Collapse Evidence**:
   Read `P3C_REPORT.md`, lines 58–69.
   Verify that under coherent gain $r = 2^{-12}$ and $r = 2^{+12}$, FP32 BER is recorded as $49.961\%$ and $54.887\%$.
2. **Inspect Positional Shortcut Evidence**:
   Read `P3_A_MECHANISM_REPORT.md`, Section 3.
   Verify that removing positional inputs dropped `sparse12` BER from $12.29\%$ to $6.55\%$.
3. **Inspect Phase Equivariance Test in P2**:
   Read `p2_symmetry.py`, lines 18–43.
   Verify that P2 tested only $r=1$ and phase angles $0.3, 1.2, 2.7$ radians without scale variation.
4. **Inspect Exploratory 16-QAM Baseline**:
   Read `exploratory\16qam_joint_pa_probe\probe.py`, lines 83–103.
   Verify that `TinyJoint` uses an un-normalized 1D CNN with uncalibrated logits and heuristic base addition.
5. **Validate Survey Report Artifacts**:
   Inspect `teamwork_explorer_survey_2\report.md` for complete mathematical and feature specifications.

---
*Handoff completed by Explorer 2.*
