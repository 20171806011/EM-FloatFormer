# EM-FloatFormer Real-OTA OFDM Theory & Stress Verification Protocol (`PROTOCOL.md`)

- **Workspace Branch**: `research/realdata-ofdm-theory-and-stress`
- **Repository Baseline Commit SHA**: `f566f9b8b0ddfb468a4cb3fd64588bfa46c87212` (forked from `research/p12-first-principles-diagnostics`)
- **Theoretical Anchors**:
  1. `research/content_addressing_scalar/REPORT.md` — Proposition A (fixed-$k$ local lower bound $L_k$), Proposition B (exact content-addressing risk $R_\alpha = v_1 + D\,\mathbb{E}\|e_J - \alpha\|_2^2$), Margin Upper Bound $U$, and §5.1 Eq. (9) shared-observation correlation bias.
  2. `research/posterior_fusion/README.md`, `pages/01.md`–`12.md`, `validation/validate.py` — Channel MSE projection identity (§3), total covariance decomposition $\Sigma = \Sigma_{\text{within}} + \Sigma_{\text{between}}$ (§4.1), and MSE-vs-BER decoupling / mixture-posterior marginalized bit LLR (§5.1–§5.2).
  3. `p11/manifold_gate.py`, `p12/offgrid_receiver.py`, `p12/P12_CONSISTENCY_FIX.md` — Conjugate-symmetric $U(1)$ parallel-transport connection $P_{s',s} = P_{s,s'}^*$, $C^*$-invariant burst-noise variance guard, and leak-free evaluation discipline.

---

## 1. Real OTA Dataset Identity & Empirically Verified Physical Grid

- **Source Repository**: `https://github.com/rikluost/sdr_ofdm_dataset_v1`
- **Source Commit SHA**: `6f9821d6390d212884fea3d6621267ebe70c7246`
- **Data File**: `ofdm_valset_4_8_sdr.pth`
  - **File Size**: `27,684,430` bytes (26.40 MiB)
  - **SHA-256**: `e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058`
  - **Measured TTI Count**: `742` TTIs (435 MHz carrier, real-world indoor/outdoor pedestrian mobility SDR over-the-air measurements; reported SINR range `5.10`–`30.40` dB, median `13.40` dB).
- **Empirically Verified OFDM Grid**:
  - `N_FFT = 128`, `N_SYM = 14`, `CP = 6` samples, `DC = 64`.
  - Active subcarriers (including DC): indices `13..114` (102 grid points); active non-DC subcarriers (`USED_SCS`): `[13..63, 65..114]` (`101` subcarriers).
  - Full Pilot Set (`14` pilots) on OFDM symbol `2` (`PILOT_SYM = 2`): FFT subcarrier indices `[13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 114]`, transmitted pilot sequence `0.7 * (±1 ± 1j)` (QPSK constant-modulus signature verified across all 742 TTIs).
  - Data REs: `14 × 101 - 14 = 1,400` resource elements in **row-major `(symbol, subcarrier)`** order, carrying `5,600` coded/uncoded bits per TTI (`labels [1400, 4]`).
  - **16-QAM Bit-to-Symbol Mapping**: Empirically inverted on high-SINR TTIs as **Natural Binary (Offset Binary)** — NOT 3GPP Gray coding:
    $$(b_0, b_1) \mapsto \frac{2(2b_0 + b_1) - 3}{\sqrt{10}} \in \left\{\frac{-3,-1,+1,+3}{\sqrt{10}}\right\}\text{ (I)}, \qquad (b_2, b_3) \mapsto \frac{2(2b_2 + b_3) - 3}{\sqrt{10}}\text{ (Q)}.$$
    Verified on Dev and Top-64 high-SINR TTIs (0.000000 BER across 63/64 top-SINR TTIs; real hardware sync-loss burst at TTIs `655..665` explicitly documented and kept intact in full Test reporting with an additional sensitivity column excluding TTIs `655..665`).

---

## 2. Leak-Free Contiguous Temporal Block Split (`splits.json`)

Because 435 MHz pedestrian fading exhibits positive temporal autocorrelation in both residual multipath shape and SINR up to lag ~15 TTIs (and contiguous hardware sync bursts up to 11 TTIs), random TTI splitting leaks channel state across train/dev/test. We enforce **strict contiguous temporal blocks separated by `20` discarded guard TTIs**:

| Split | TTI Index Range | TTI Count | Block Structure | Usage |
| :--- | :--- | ---: | :--- | :--- |
| **Train** | `0 .. 339` | `340` | 17 contiguous blocks × 20 TTIs | Train-only historical pilot reference bank ($M=16$), Group C neural training |
| **Guard 1** | `340 .. 359` | `20` | Excluded buffer | Strictly never read by any stage |
| **Dev** | `360 .. 481` | `122` | 6 contiguous blocks | Mapping check, Group A/B hyperparameter & temperature calibration, Group C early stopping & LLR temperature calibration |
| **Guard 2** | `482 .. 501` | `20` | Excluded buffer | Strictly never read by any stage |
| **Test** | `502 .. 741` | `240` | **12 contiguous blocks × 20 TTIs** (`test_b00 .. test_b11`) | Final frozen paired evaluation (`1,344,000` bits per condition) |

- **Cluster-Level Statistical Inference**: All uncertainty intervals and significance tests on Test treat the **`G = 12` contiguous temporal blocks** as the independent clustering units:
  - 95% confidence intervals via cluster Student-$t$ ($df = 11$).
  - Exact two-sided and one-sided **$2^{12} = 4,096$-state sign-flip permutation test** (minimum achievable two-sided $p = 2/4096 = 0.0004883$), plus block Win/Tie/Loss counts.

---

## 3. Experimental Groups & Controlled Physical Stress Matrix

### 3.1 Five Standardized RF Conditions (Raw OTA Strictly Separate from Stressed)

Each condition uses deterministic per-`(TTI, condition)` seeding `condition_seed(tti, cond, salt=20261121)`:
1. **`C0_raw_ota`**: Unmodified real OTA `pdsch_iq [14, 128]`, 14 pilots.
2. **`C1_awgn_stress`**: Controlled wideband complex circular Gaussian noise added to active/guard grid (`SNR_add = 6 dB` relative to measured active power; supplemented by a full 0/3/6/9/12 dB drop sweep).
3. **`C2_cfo_doppler`**: True time-domain continuous CFO/Doppler drift via exact unitary `IFFT(128) -> CP(6) -> exp(j 2π ε (n - n_ref)/128) -> CP removal -> FFT(128)`, with $|\varepsilon| \in [0.025, 0.055]$ subcarrier spacings (random sign, anchored at pilot symbol 2), inducing real inter-carrier interference (ICI) and severe symbol 4–13 phase rotation.
4. **`C3_phase_jump_jitter`**: Abrupt post-pilot oscillator phase jump at $s_{\text{jump}} \in \{5, 6, 7, 8\}$ with $\Delta\phi \in \{\pm 45^\circ, \pm 90^\circ, \pm 135^\circ, 180^\circ\}$ plus random-walk Wiener phase jitter ($\sigma_\phi = 0.04$ rad/sym).
5. **`C4_nbi_sparse7`**: Pilot sparsification to 7 pilots `[13, 29, 45, 61, 77, 93, 114]` combined with localized narrowband burst interference (8 contiguous active subcarriers × 5 contiguous post-pilot OFDM symbols at `+10 dB` INR).

### 3.2 Group A (`group_a_content_addressing.py`)
- **A1 (Real OTA Sparse-Pilot Historical Reference Bank, $M=16$ Train-only)**: Evaluates 14-pilot, 7-pilot, and 4-pilot densities comparing single-TTI LS-linear, LS-spline, fixed local-$k$ ($k=4$), global uniform ($M=16$), inverse-distance weighting, Dev-calibrated content-addressing softmax, and over-cold/over-hot controls on held-out real pilot MSE, oracle channel NMSE, and 16-QAM BER/SER.
- **A2 (Direct Theorem Formula Verification)**: Disjoint odd/even OFDM symbol real-channel innovation split verifying Proposition A ($L_k$), Proposition B ($R_\alpha = v_1 + D\,\mathbb{E}\|e_J-\alpha\|_2^2$), Margin Upper Bound ($U$), and §5.1 Eq. (9) shared-observation bias ($\Delta_{\text{corr}}$).

### 3.3 Group B (`group_b_posterior_fusion.py`)
- Shared condition-agnostic `R = 17` physical hypothesis bank (static LS, 8 linear CFO/Doppler slopes, 7 post-pilot phase jumps, 1 Train-only content-addressed static channel) with blind $C^*$-invariant NBI variance guard and HMM-smoothed blind 16-QAM marginal likelihood posterior.
- Paired comparison across `B0` (classical single-TTI), `B0b` (+ NBI guard), `B1` (MAP hard select), `B2a` (posterior linear channel fusion, within-var only), `B2b` (posterior linear channel fusion + exact total variance $\Sigma_{\text{within}}+\Sigma_{\text{between}}$), and `B3` (full mixture posterior directly inside 16-QAM bit LLR).

### 3.4 Group C (`group_c_transport_attention.py`)
- Frozen shared `R = 17` frontend + byte-identical blind features (`4` $U(1)$-equivariant complex channels + `10` $C^*$-invariant scalar channels) + identical differentiable mixture 16-QAM LLR output head.
- Exact parameter-matched comparison across **3 independent seeds** (`20261121, 20261122, 20261123`):
  - **Role A (`PhysParallelTransportNet`)**: `12,308` trainable real parameters; symmetric parallel-transport connection $P_{s',s} = P_{s,s'}^*$ and $U(1)$-equivariant continuous residual phase readout.
  - **Role B1 (`MatchedRealTransformer`)**: `12,415` trainable real parameters (`+0.87%`); standard 2-axis real Transformer with relative position bias.
  - **Role B2 (`MatchedRealMLP`)**: `12,419` trainable real parameters (`+0.90%`); context-pooled residual real MLP.
- Single-threaded execution (`OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=1`), peak RSS `< 2.5 GB`, early stopping and single-scalar temperature calibration strictly on Dev, full paired evaluation on Test.
