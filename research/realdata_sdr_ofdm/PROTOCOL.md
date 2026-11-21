# Experimental Protocol (`research/realdata_sdr_ofdm`)

## 1. Study Scope & Research Questions

This protocol governs all experiments applying EM-FloatFormer receiver algorithms to real ADALM-Pluto SDR 16-QAM CP-OFDM measurements (`rikluost/sdr_ofdm_dataset_v1`, `742` TTIs, `5,600` payload bits/TTI):

- **RQ1 (Group A — Global Reference & Content Matching)**: Holding the downstream equalizer/demapper fixed, how do expanding the reference pool from local single-TTI pilots to a multi-TTI global reference pool (uniform weighting vs. classical reliability weighting vs. content-matched softmax attention) affect held-out pilot calibration and payload BER/FER?
- **RQ2 (Group B — Hard Selection vs. Posterior Fusion)**: Under a shared candidate branch bank (delay/CFO/phase/reference hypotheses), how do hard MAP branch selection, weighted posterior-mean channel plug-in, and exact candidate mixture likelihood decoding ($\sum_r \alpha_r p(y \mid c, H_r, \sigma^2_r)$) compare in payload accuracy and uncertainty calibration?
- **RQ3 (Group C — Lightweight Trainable / Attention Module)**: Does a lightweight trainable PyTorch residual/attention module (fitted strictly on `train` and selected on `dev`, keeping the candidate bank and LLR demapper fixed where possible) improve over tuned classical and non-parametric global baselines on clean and stress-perturbed real SDR data?

---

## 2. Data Provenance, Signal Geometry & Known Errata

- **Dataset**: `ofdm_valset_4_8_sdr.pth` (`27,684,430` bytes, SHA-256 `e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058`, git blob SHA-1 `9b1aee8673ec48f42f52dc365fb01cc8b3e2b741`), `742` TTIs, reported `sinr` in `[5.1, 30.4] dB` (mean `14.04 dB`, median `13.40 dB`, std `6.05 dB`).
- **Grid & Normalization**: Centered post-DFT grid `(S=14, N_fft=128)` with DC at bin `64`, active band `13..114` (`102` bins), `14` QPSK pilots at symbol `Sp=2` (`|pilot|^2 = 0.98`, NOT `1.0`), `1400` 16-QAM payload REs (`5600` bits/TTI) in non-Gray natural-binary `mapping_table(4)` order. Each TTI is scaled by blind observable active-band RMS `s0 = sqrt(mean(|rx[pilot | data]|^2))` so active RE mean power is `1.0`.
- **Blind Noise Variance (`est_n0`)**: Estimated strictly from passband guard bins `SAFE_GUARD_FFT_INDICES = (10, 11, 117, 118)` inside the `825 kHz` Butterworth LPF flat passband (`|k - 64| < 55`, avoiding attenuated outer bins `0..8, 120..127` and active-edge leakage bins `12, 115, 116`), with automatic fallback to 2nd-difference pilot variance (`pilot_diff2_n0`) if guard power `<= 1e-10`.
- **Known Errata Handled**: Non-Gray `mapping_table(4)` bit mapping; `|pilot|^2 = 0.98`; `CP=7` (`config.py`) vs `CP=6` (README); broken companion `equalize_ZF` unwrap replaced by exact complex division / LMMSE.

---

## 3. Partition & Anti-Leakage Firewall (`manifests/splits.json`)

### 3.1 Empirical Diagnostics (All 742 TTIs)
- **Payload Deduplication**: `742 / 742` unique `tx_bits_sha256` hashes (`0` duplicate payloads).
- **Temporal Autocorrelation**:
  - `sinr` lag-`1..8`: `[0.171, 0.039, 0.145, 0.067, 0.157, 0.135, 0.084, 0.021]` (rapid decay due to per-TTI randomized TX gain).
  - `h_pilot` coherent inner-product correlation lag-`1..32`: decays slowly from `0.847` (lag `1`) to `0.756` (lag `16`) and `0.736` (lag `32`), indicating persistent RF/SDR hardware channel structure across the capture session.
- **Guard Buffer Selection**: Because `h_pilot` coherence exceeds `0.10` through lag `32`, guard width saturates at `G = 16` frames per block boundary (`min_cross_split_distance = 17`).

### 3.2 Stratified Contiguous Macro-Block Assignment
The `742` frames are divided into `12` contiguous macro-blocks (`block_00..block_11`), with `16` guard frames stripped at each of the `11` boundaries (`176` guard TTIs excluded from all training, tuning, reference pools, and evaluation), and macro-blocks stratified by block-median SINR:

| Split | Block IDs | TTIs | SINR Min / Median / Mean / Max (dB) | Allowed Usage |
|---|---|---:|---|---|
| **Train** | `block_00, 02, 04, 06, 07, 08, 11` | 336 | `5.1 / 12.75 / 13.68 / 30.2` | Model fitting, empirical covariance priors, Group A reference pool |
| **Dev** | `block_03, 05` | 92 | `5.1 / 14.40 / 14.49 / 28.8` | Hyperparameter selection (`n0` scale, temperature $\tau$, regularization, checkpoints) |
| **Test** | `block_01, 09, 10` | 138 | `5.1 / 12.35 / 13.72 / 30.4` | **Frozen evaluation only** (zero tuning, calibration, or model selection) |
| **Guard** | 11 boundary buffers (`G=16`) | 176 | `5.1 / 14.30 / 14.74 / 29.4` | **Strictly excluded** from all training, tuning, reference pools, and test |

### 3.3 Anti-Leakage Rules
1. **Zero Test Leakage**: `test` labels (`tx_bits`, `tx_symbols`), `test` `reported_sinr_db`, and `test` aggregate statistics must NEVER inform `s0`, `est_n0`, LMMSE covariance, softmax temperatures, calibration curves, or model checkpoints.
2. **Reference Pool Firewall (Group A)**: Reference pools for `test` evaluation must be constructed strictly from `train` (or causal/observable pilot-only intra-split context as explicitly labeled), never from `dev`/`guard` or using `tx_bits`.
3. **Held-Out Pilot Evaluation**: Pilot prediction & uncertainty calibration use deterministic 4-fold interior pilot CV (`[1, 5, 9, 12]` in fold 0; endpoints `0, 13` always observed).

---

## 4. Method Input, Context & Candidate Contracts

All receivers implement `run_frame(frame: RealOFDMFrame, context_pool=None, ...) -> ReceiverOutput`:
- **Inputs**: Observable `frame.rx_grid`, `frame.pilot_mask`, `frame.dc_mask`, `frame.data_mask`, `frame.pilot_values`, `frame.est_n0` (NEVER `frame.tx_bits` or `frame.tx_symbols`).
- **Group A**: Shared fixed context budget `C` across local pilot reference, global uniform average, global classical reliability weighting, and content-matched softmax attention (guarding scoring-value correlation and calibrating temperature strictly on `dev`).
- **Group B**: Shared candidate branch bank (delay/CFO/phase/reference hypotheses) with fixed truncation/pruning rule comparing (1) hard MAP branch, (2) weighted posterior-mean plug-in, and (3) exact candidate Gaussian mixture log-likelihood decoding.
- **Group C**: Lightweight PyTorch module trained strictly on `train` (validated on `dev`), logging architecture, parameter count, optimizer/seed, training wall-clock time, and library versions, keeping the candidate bank and LLR demapper matched to classical/Group B baselines.

---

## 5. Metrics, Event Definitions & Statistical Inference

- **Held-Out Pilot Metrics**:
  - `pilot_nmse`: $\sum |H_{\text{true}} - \hat{H}|^2 / \sum |H_{\text{true}}|^2$ on held-out pilot REs (using pilot LS observations $y_p / p$ as noisy ground truth or noise-corrected variance).
  - `pilot_cal_ratio`: $\text{mean}(|y_p/p - \hat{H}|^2 / (\hat{\sigma}^2_H + \sigma^2_{\text{pilot,obs}}))$.
  - `pilot_nll`: complex Gaussian NLL on held-out pilot observations.
- **Payload Metrics**:
  - `evm_pct`: $100 \sqrt{\sum |\hat{x}_{\text{eq}} - x_{\text{tx}}|^2 / \sum |x_{\text{tx}}|^2}$ (%).
  - `sym_mse`: $\text{mean}(|\hat{x}_{\text{eq}} - x_{\text{tx}}|^2)$.
  - `ser`: symbol error rate over `1,400` payload REs/TTI (`193,200` symbols on `test`).
  - `ber`: uncoded bit error rate over `5,600` payload bits/TTI (`772,800` bits on `test`), always reporting exact integer `bit_errors / total_bits`.
  - `fer`: frame/TTI error rate; **event definition**: `>= 1` bit error among the `5,600` payload bits of a TTI (`total_frames = 138` on `test`).
- **Paired Inference**:
  - Report exact error-count deltas relative to the anchor classical baseline.
  - Compute `group_id`-clustered (`block_01, block_09, block_10` or sub-block clusters) bootstrap 95% CIs and exact sign-flip permutation p-values with Holm step-down correction (`p12/audit.py`).
- **Failure & Coverage Accounting**: Track `success_count`, `fail_count`, `timeout_count`, exception messages, and `coverage_pct`.

---

## 6. Stress Perturbation & Resource Profiling Protocol

- **Stress Perturbations** (`apply_stress_perturbation`): Reproducible physical perturbations via `ifftshift -> ifft(norm="ortho") -> continuous sample CFO/phase drift -> Parseval AWGN -> fft(norm="ortho") -> fftshift`, keeping `tx_bits` and `tx_symbols` bit-identical:
  - `extra_awgn_db`: `[null, 10.0, 5.0, 0.0]` (`added_awgn_snr_db` relative to active-band power $p_{\text{active}}$: `null`/`None` = clean baseline with $\sigma^2_{\text{add}} = 0.0$; numeric injects Parseval AWGN of variance $\sigma^2_{\text{add}} = p_{\text{active}} \cdot 10^{-\text{extra\_awgn\_db}/10}$, ordered clean $\to$ mild $\to$ moderate $\to$ severe; note `0.0` injects $\sigma^2_{\text{add}} = p_{\text{active}}$, NOT clean)
  - `cfo_norm`: `[0.0, 0.02, 0.05]`
  - `phase_drift_rad`: `[0.0, 0.3, 0.6]`
- **Consistent Engineering Cost Profiling**:
  - Device: `cpu`, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`, batch size `1`, `8` warmup frames before timing.
  - Report `mean_ms_per_frame`, `p95_ms_per_frame`, `peak_rss_mb`, `n_candidates`, `param_count`, and `train_time_s`.
