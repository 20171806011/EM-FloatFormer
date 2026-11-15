# EM-FloatFormer: P3-D / P3-E standards-bridge contingency protocol

**Frozen before training or held-out testing, 2026-11-15.** Purely synthetic, passive civilian QPSK/OFDM only. This is a **surrogate experiment**, not a completed NVIDIA Sionna 2.2 or 3GPP compliance benchmark.

## Mandatory scope separation

- The original P3-D goal is true Sionna 2.2 TR38.901 TDL/CDL simulation, Sionna's LMMSEChannelEstimator, hardware-independent timing, and then an independent 10-seed P3-E confirmatory replication. **This runtime has no Sionna installed**; a 2026-11-15 installation attempt failed due to unavailable DNS. Consequently **formal P3-D and formal P3-E are not satisfied** and must remain explicitly marked incomplete until a real Sionna-backed run is performed.
- To use available compute productively, perform a separate **independent static TDL-profile simulation** using the TDL-A, TDL-C and TDL-D **V19.2** parameter arrays from NVIDIA Sionna 2.2, `https://github.com/NVlabs/sionna/tree/v2.2.0/src/sionna/phy/channel/tr38901/models/v19_2` (data copyright NVIDIA, Apache-2.0). The independent simulator has different sampling, state evolution and filtering from the package. It **must not** be named a Sionna run.
- Source SHA256: `p3de_profile_bridge.py` = `dd803782a086ba822725bfa6868d1a2b8320a4503fee5168cec4688e5372cfc6`; `p3de_experiment.py` = `9a6fb2b5436ac9c1743ddbf36638a83f029dcc0e9349432dcec92d17d68b82b3`. No changes to these source files are allowed between freezing and training/evaluation; if correctness bugs require changes, stop, log the deviation, and rerun under a separately frozen revision.

## Standalone surrogate physics

- One static SISO block per independent synthetic frame, **32 known all-carrier QPSK pilot tones**, one uncoded QPSK payload symbol (64 scored bits), ideal time/frequency synchronization, exact known reported thermal-noise variance. No real SDR, radar, active RF response, or live transmissions.
- Frequency spacing 30 kHz, 32 carriers, a 16-sample cyclic-prefix **assumption**, fractional physical-path delays and powers from the official TDL profile tables, tap powers normalized to unity; `H[k]` built by independent harmonic summation `sum_l alpha_l exp(-2πi f_k tau_l)`. TDL-A/C use independently drawn complex Gaussian path coefficients; TDL-D's first LoS component has fixed power with random phase and the rest are Gaussian. This is a deliberately narrow **static approximation**, not a time-domain convolution, standard-compliant fading simulator or Sionna-generated channel.
- The maximum modelled path delays for the chosen spreads remain shorter than the assumed CP. Transmitted QPSK symbols and noisy pilot/data observations are synthetic; the receiver cannot access the true channel.
- Training scenarios: TDL-A with 500 ns spread and TDL-C with 500 ns spread, alternating across training steps. Held-out scenarios: TDL-A 500 ns (within training profile family), TDL-C 1000 ns (new delay spread), TDL-D 750 ns (new LoS profile). Evaluate at SNR 0,10,20 dB. No post-test profile/parameter adjustments.

## Neural models and classical comparators

- Exactly two lightweight pre-existing model families, each newly trained from scratch: P3-A2 `A_pos_0` (985 nominal parameters, absolute delay inputs, no floor) and `D_no_pos_10` (985 nominal, no absolute delay inputs, 10% gain retention floor). These share identical initial state within a seed, minibatches, and training regime. They are trained only on the surrogate data; **no archived P2-E/P3-A/P3-B/P3-C weights are retrained, overwritten or re-used**.
- Ten new training seeds **61001–61010**, distinct from all preceding research; **300 optimizer steps/model/seed**, batch64, alternating train scenarios, uniformly random SNR 0–20dB, AdamW lr0.002, weight_decay0.0001, gradient clipping norm1, QPSK BCE objective, CPU PyTorch 2.10.0. No tuning after observing evaluation.
- Fixed nonprivileged traditional controls on the same observations: pilot LS, DFT8, DFT12, noise-only 0.05 expected false-tap threshold, LMMSE covariance averaged from the two **known training population profiles**. The test-profile covariance LMMSE is **privileged** and true instantaneous `H` is a **nondeployable diagnostic**, never an equal-information comparator. Note fractional delays challenge the integer-delay DFT8/DFT12 structural priors.

## Prespecified evaluation and statistics

- Ten independently trained models per arm; **2048 held-out frames per seed/scenario/SNR** on distinct PRNG streams, identical frames for all estimators: `10 × 2048 × 64 = 1,310,720` bits per method/scenario/SNR. Sampling unit is independent **training seed**, not bits or subcarriers.
- Two primary surrogate contrasts at 10dB, **D versus the *fixed*, nonprivileged hard-threshold estimator**, separately on held-out TDL-C/1000ns and TDL-D/750ns. Define paired delta = `BER(Threshold)-BER(D)` in percentage points; positive favors D. Use exhaustive `2^10` two-sided sign flips, Holm adjustment for the two endpoints, and descriptive t(9) confidence intervals. Do not change any decisions to achieve p < 0.05; negative/null results are valid.
- Secondary: A vs D, LS/DFT8/DFT12/Train-LMMSE, privileged Test-LMMSE, true channel, all 0/10/20dB profiles, channel MSE, CPU model-forward time. Never select the minimum classical method per test cell and call it a deployable model.
- Validate input generation: normalized profile powers, Hermitian nonnegative covariance, exact integer-delay convolution theorem check, CP-length envelope, no payload-truth leakage, finite gradients. After training, archive model SHA256, per-seed raw integer errors, analytic summaries, same-code checkpoint replay, seed-wise inference and timing. Confirm historical P2-E/P3-A/P3-A2/P3-B/P3-C reports unchanged.

## Completion and stop conditions

- The **surrogate** is complete only after all training, all nine scenario/SNR combinations, independent raw-count reconciliation, checkpoint replays, and honest result report. It cannot be substituted for a Sionna 2.2 TDL/CDL benchmark.
- **Do not mark formal P3-D or formal P3-E as completed**, and do not claim research project Phase 3 completed, unless Sionna 2.2 genuine execution and its agreed full comparison/confirmation gates have actually passed. Disclose any unavailable features: dynamic fading, full time-domain RF, standard pilot layouts/coding, Sionna-native LMMSE/3GPP CDL, GPU latency and hardware energy.
- Archive local source, reports, CSV, figures and model weights. Publish source, protocol, raw results and a clear **blocker/scope addendum** to the existing `20171806011/EM-FloatFormer` repository under a distinct `exploratory/p3de_tdl_surrogate/` directory. Never rewrite earlier frozen protocols or results.