# Project: EM-FloatFormer Stage P4

## Architecture
Stage P4 advances EM-FloatFormer to an electromagnetic-induction native Transformer algorithm under 16-QAM and nonlinear power amplifier (PA) distortion.

### Physical & Mathematical Foundations
1. **$C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ Gauge Group Equivariance**:
   - Electromagnetic signals undergo arbitrary channel gain scaling $r \in \mathbb{R}_{>0}$ and common phase rotation $e^{j\phi} \in \mathrm{U}(1)$.
   - Global gauge anchoring via pilot RMS: $s_0 = \sqrt{\frac{1}{N}\sum |Y_p|^2}$, mapping inputs to scale-free unit-RMS representations.
   - Internal Complex RMSNorm preserving complex phase angles: $\text{cRMSNorm}(Z) = \frac{Z}{\sqrt{\frac{1}{2 d}\sum (|X|^2 + |Y|^2) + \epsilon}} \odot \gamma_c$.
2. **Hermitian Green's Propagator Attention**:
   - Subcarrier interactions are governed by the electromagnetic Green's propagator $G(m, n) = \sum_l \alpha_l e^{-j 2\pi (m-n)\Delta f \tau_l}$.
   - Complex Linear without bias ($b=0$): guarantees $W(c Z) = c W(Z)$ for all $c \in \mathbb{C}^*$.
   - Complex relative-phase RoPE: relative subcarrier distance phase modulation $R(m-n)$, eliminating harmful absolute coordinate shortcuts.
   - Attention affinities: real projections of complex inner products $\text{Re}(\langle Q_m, K_n \rangle_{\mathbb{C}})$, naturally canceling global phase $e^{j\phi}$.
   - Radial activation: $f(Z) = g(|Z|) \frac{Z}{|Z|}$, strictly preserving $\mathrm{U}(1)$ phase angles.
3. **Bidirectional Pilot-Data Interaction**:
   - Compensates for nonlinear PA inter-carrier distortion ($D_{\text{nonlinear}}$) spilling between payload and pilot subcarriers via cross-attention iterations.
4. **Joint Physical Consistency Objective**:
   - $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{NMSE}} + \lambda_2 \mathcal{L}_{\text{PA-res}}$
   - Calibrated 16-QAM LLR BCE, normalized scale-invariant channel NMSE, and PA forward residual consistency.
5. **2D Time-Frequency & Coded Representation**:
   - 2D time-varying slot grids ($N_{\text{symb}} \times N_{\text{sc}}$) with sparse DMRS pilots ($\le 10\%$), Doppler ICI, and sample-level CFO phase drift.
   - Native self-contained outer forward error correction (Rate-1/2 LDPC/Conv) with soft decoding to evaluate true coded BLER.

---

## Feature Inventory
Every feature from Stage P4 requirements (R1–R4) is cataloged and assigned to a milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Oversampled Rapp PA | Rapp AM-AM ($p=2, A \in \{1.25, 2.25\}$) and AM-PM phase distortion at $L \ge 4$ oversampling | M1 | R1 (Stage P4-0) |
| F2 | Dynamic Multipath Simulator | Time-domain dynamic multipath channel with CP insertion/removal and continuous Doppler ICI | M1 | R1 (Stage P4-0) |
| F3 | Static-Limit Parity Check | Exact parity check with static frequency-domain matrix model under zero Doppler and linear PA ($\epsilon < 10^{-5}$) | M1 | R1 (Stage P4-0) |
| F4 | Bussgang Decomposition & LMMSE | Calculation of Bussgang gain $\gamma$, distortion $\sigma_D^2$, and Bussgang-LMMSE equalization | M1 | R1 (Stage P4-0) |
| F5 | Calibrated 16-QAM Soft LLRs | True posterior log-likelihood ratios for 16-QAM under effective distortion-plus-noise variance | M1 | R1 (Stage P4-0) |
| F6 | Iterative DD-PA Cancellation | 2-3 iteration decision-directed time-domain distortion reconstruction, channel filtering, and subtraction | M1 | R1 (Stage P4-0) |
| F7 | Classical Latency Benchmark | Reproducible execution latency measurement on CPU (4 threads) / GPU | M1 | R1 (Stage P4-0) |
| F8 | Exact $C^*$ Gauge Normalization | Pilot RMS gauge anchor + internal Complex RMSNorm ensuring invariance across $r \in [2^{-12}, 2^{12}]$ | M2 | R2 (Stage P4-A) |
| F9 | Hermitian Green's Attention | Complex Linear without bias, complex relative-phase RoPE, and $\mathrm{U}(1)$-preserving radial activation | M2 | R2 (Stage P4-A) |
| F10 | Bidirectional Pilot-Data Attention | Mutual cross-attention between pilot channel estimates and payload demapping | M2 | R2 (Stage P4-A) |
| F11 | Joint Physical Consistency Loss | Joint training objective: calibrated LLR BCE + normalized channel NMSE + forward PA residual consistency | M2 | R2 (Stage P4-A) |
| F12 | Numerical Equivariance Verification | Unit tests verifying strict $C^*$ invariance under arbitrary scale $r \in [2^{-12}, 2^{12}]$ and random phases ($\epsilon < 10^{-5}$, zero bit flips) | M2 | R2 (Stage P4-A) |
| F13 | 2D Slot Simulator & Sparse DMRS | Multi-symbol time-varying slots ($N_{\text{symb}}=14, N_{\text{sc}}=32/64$) with sparse DMRS pilots ($\le 10\%$ overhead) | M3 | R3 (Stage P4-B) |
| F14 | Continuous Doppler ICI & CFO | High-Doppler time-varying fading and continuous sample-level carrier frequency offset phase drift | M3 | R3 (Stage P4-B) |
| F15 | Native Outer Channel Coding | Self-contained native PyTorch Rate-1/2 FEC (LDPC / Convolutional) with soft decoding | M3 | R3 (Stage P4-B) |
| F16 | 2D Classical Baselines | Separable 2D-LMMSE channel estimation and 2D iterative decision-directed PA cancellation | M3 | R3 (Stage P4-B) |
| F17 | Coded BLER Evaluation | Coded block error rate evaluation against 2D classical baselines across out-of-distribution Doppler and delay spreads | M3 | R3 (Stage P4-B) |
| F18 | Factorial Ablation Matrix | Orthogonal $2^3$ ablation isolating $C^*$ gauge normalization, complex attention, and joint pilot-data propagation | M4 | R4 (Stage P4-C) |
| F19 | Numeric Stability & Precision Audit | Stress testing in FP32, BF16, and FP16 across 24 octaves of dynamic range ($r \in [2^{-12}, 2^{12}]$) | M4 | R4 (Stage P4-C) |
| F20 | Pre-Frozen Confirmatory Protocol | Locked protocol across 10 fresh seeds ($74001–74010$) with exact integer bit/block error accounting | M4 | R4 (Stage P4-D) |
| F21 | Sign-Flip Permutation Tests | Two-sided 1024-state exact permutation tests with Holm-Bonferroni step-down adjustment | M4 | R4 (Stage P4-D) |
| F22 | Repository Governance & Immutability | SHA256 integrity verification ensuring historical P1-P3 records and manifests are 100% untouched | M4 | R4 (Stage P4-D) |
| F23 | Opaque-Box E2E Test Suite | Comprehensive 4-tier test suite verifying functional and physical requirements as an end user | M5 | E2E Track |
| F24 | White-Box Adversarial Hardening | Tier 5 adversarial stress testing searching for uncovered edge cases and numerical limits | M5 | E2E Track |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Stage P4-0 Simulator & Classical Baselines | F1–F7: Dynamic time-domain channel simulator, oversampled Rapp PA (AM-AM & AM-PM), static-limit parity, Bussgang-LMMSE, calibrated soft LLRs, Iterative DD-PA cancellation, latency benchmarks | none | DONE |
| M2 | Stage P4-A $C^*$-Equivariant EM-Native Transformer | F8–F12: $C^*$ gauge normalization, Hermitian Green's attention, relative-phase RoPE, radial activation, bidirectional attention, joint loss training, equivariance unit tests | M1 | DONE (`p4/em_floatformer.py`, `p4/tests/test_p4_a_equivariance.py`, `p4/results/p4a_comparative_benchmark.{json,csv}`, 53/53 unit + 156/253 active E2E pass, Gate PASS) |
| M3 | Stage P4-B 2D Generalization & Coded BLER | F13–F17: 2D slot simulator, sparse DMRS, continuous Doppler ICI & CFO, native outer FEC with soft decoding, 2D-LMMSE & IDD-PA baselines, OOD BLER evaluation | M1, M2 | Measurements, independent replay and the versioned qualification are tracked in the [research status](p4/STAGE3_RESULTS.md). |
| M4 | Stage P4-C/D Mechanism Probing & Confirmatory Replication | F18–F22: $2^3$ factorial ablations, numeric stability & mixed-precision audits, pre-frozen 10-seed confirmatory replication, sign-flip tests with Holm adjustment, historical SHA256 audit | M2, M3 | The [formal protocol](p4/P4D_PROTOCOL.md) specifies the primary 2D receiver, 80 one-dimensional fits and the two Holm families. See the [research status](p4/STAGE3_RESULTS.md) for completed runs. |
| M5 | Stage P4 Final: 100% E2E Verification & Adversarial Hardening | F23–F24: Pass 100% opaque-box E2E test suite (Tiers 1-4) and complete white-box Tier 5 adversarial coverage hardening | M1, M2, M3, M4 | The [research status](p4/STAGE3_RESULTS.md) records the executable E2E and adversarial results. |

---

The [initial continuation report](p4/P4_CONTINUATION_REPORT.md) records the Gaussian receiver and original gate. The [first receiver follow-up](p4/P4B_RECEIVER_FOLLOWUP.md) and [research status](p4/STAGE3_RESULTS.md) record the subsequent experiments. The prior 92-file P4 revision, its original tests and data are available in [the pre-stage3 snapshot](p4/revisions/pre_stage3_p4.zip), to be extracted over a separate checkout of the pre-P4 historical tree for exact replay.

## Interface Contracts

### M1 (Simulator & Baselines) ↔ M2 (EM-Native Transformer)
- **Signal Tensors**:
  - Received signal: `y: torch.Tensor`, dtype `torch.complex64`, shape `(B, N)` where $N=32$ (or 64).
  - Pilot observation: `yp: torch.Tensor`, dtype `torch.complex64`, shape `(B, N_p)`.
  - Transmit data ground truth: `bits: torch.Tensor`, dtype `torch.int64`, shape `(B, N_d * 4)` for 16-QAM.
  - Channel frequency response ground truth: `h: torch.Tensor`, dtype `torch.complex64`, shape `(B, N)`.
  - Noise variance: `nv: torch.Tensor`, dtype `torch.float32`, shape `(B,)`.
- **Rapp PA Interface**:
  - Function: `rapp_pa(x_time, knee=2.0, A=2.25, alpha=0.5, beta=1.0, oversample=4) -> y_time`
  - AM-AM: $g(r) = \frac{r}{(1 + (r/A)^{2p})^{1/(2p)}}$
  - AM-PM: $\Phi(r) = \frac{\alpha r^2}{1 + (r/\beta)^2}$
- **Classical Baseline Outputs**:
  - Function: `bussgang_lmmse(yp, yd, nv, A) -> (h_hat, llrs, hard_bits)`
  - Function: `iterative_dd_pa_cancellation(yp, yd, nv, A, iterations=2) -> (h_hat, llrs, hard_bits)`

### M2 (EM-Native Transformer) ↔ M3 (2D Generalization)
- **Model Architecture Call**:
  - Forward: `model(yp, yd, nv) -> (h_hat, llrs, pa_dist_est)`
  - Output `llrs`: calibrated posterior log-odds for bits, shape `(B, N_d, 4)`.
  - Dimensionless handling: input scale $s_0$ is internally removed; internal states are scale-free unit RMS; output channel $\hat{H}$ is re-scaled by $s_0 / s_{\text{pilot\_tx}}$.
- **2D Extension**:
  - 2D token grid: `(B, N_symb, N_sc, 2)` or `(B, N_symb * N_sc)` complex tokens.
  - Pilot mask: boolean tensor `dmrs_mask: torch.Tensor`, shape `(N_symb, N_sc)`.

### M2/M3 ↔ M4 (Probing & Confirmatory Replication)
- **Ablation Model Factory**:
  - `create_floatformer(use_gauge_norm=True, use_hermitian_attention=True, use_joint_pilot_data=True)`
- **Confirmatory Seeds**:
  - Pre-frozen seed list: $\mathcal{S}_{\text{P4D}} = [74001, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010]$.
  - Bit error accounting: exact integers `total_bit_errors: int`, `total_bits: int`, `total_block_errors: int`, `total_blocks: int`.

### All ↔ M5 (E2E Verification)
- **CLI Commands**:
  - Parity test: `python -m unittest p4/tests/test_simulator.py -v`
  - Equivariance test: `python -m unittest p4/tests/test_equivariance.py -v`
  - Baseline execution: `python -m p4.classical_baselines --condition all`
  - Confirmatory replication: `python -m p4.confirmatory_protocol --execute`
  - All exit with code 0 on success.

---

## Code Layout
All new deliverables for Stage P4 are strictly isolated inside the `p4/` package to preserve historical P1-P3 integrity.

```text
EM-FloatFormer/
├── p4/                                      # Dedicated Stage P4 package
│   ├── __init__.py
│   ├── simulator.py                         # F1, F2, F3: Dynamic time-domain channel & Rapp PA
│   ├── classical_baselines.py               # F4, F5, F6: Bussgang-LMMSE, soft LLR, IDD-PA
│   ├── benchmark_latency.py                 # F7: CPU/GPU latency benchmark
│   ├── em_floatformer.py                    # F8, F9, F10, F11: C*-equivariant Transformer & losses
│   ├── slot_2d.py                           # F13, F14, F15, F16, F17: 2D slot, DMRS, FEC & BLER
│   ├── ablation_probing.py                  # F18, F19: 2^3 ablations & dynamic range audits
│   ├── confirmatory_protocol.py             # F20, F21, F22: 10 seeds, sign-flip tests, Holm adj
│   └── tests/
│       ├── __init__.py
│       ├── test_p4_0_simulator.py           # Unit tests for M1
│       ├── test_p4_a_equivariance.py        # Unit tests for M2
│       ├── test_p4_b_2d_fec.py              # Unit tests for M3
│       └── test_p4_c_d_governance.py        # Unit tests for M4
├── tests/
│   └── test_p4_e2e.py                       # F23: Comprehensive E2E test suite (Tiers 1-4)
├── exploratory/                             # Historical exploratory runs (IMMUTABLE, READ-ONLY)
├── p1*.py, p2*.py, p3*.py                   # Historical milestones (IMMUTABLE, READ-ONLY)
├── MANIFEST_SHA256.json                     # Historical integrity manifest (IMMUTABLE, READ-ONLY)
└── PROJECT.md                               # Living project blueprint & status tracker
```
