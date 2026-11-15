# EM-FloatFormer P3-B scope addendum

**Status:** P3-B0/B1/B2/B3/B4 completed as **exploratory, receive-only synthetic civilian QPSK/OFDM**. P3-C/D/E not started. P2-E/P3-A/P3-A2 frozen experiments remain unchanged.

- Pre-run protocol: `P3B_PROTOCOL.md`, committed before model fitting (`440c4aba5881306e29165727dee03cb7f0dc1550`).
- All five model seeds, 30 newly trained heads, nine clean/single-factor conditions, 0/10/20 dB, five channel profiles and all fixed baseline methods were evaluated. 9,450 raw measurements and 1,890 aggregate table rows; exact checkpoint replay 210/210.
- Main negative finding: learned residual heads do not beat the fixed classical ensemble envelope; no claim about absolute optimality of any estimator or future nonlinear receivers. Finite-seed empirical data do not establish broad scientific laws.
- Added missing privileged oracle-context references afterward, in separate CSV; original frozen primary data were not rewritten or retuned. Inference code uses only observed pilot, payload, public anchors, prior learned during training, and reported total mean noise.
- Actual RF-EVM, FLOP counts, activation memory, real radio latency, ADC electronics, full CFO/ICI, continuous multipath fading, coding and Sionna/3GPP benchmark remain untested. The saved scale-aligned symbol MSE is *not* EVM. PA-nonlinearity-aware classical baselines were not implemented. No new parameter sweeps or extra training are supported by these results.
- Full model weights are in the separate full local research archive; GitHub release may include source, CSV, checksum and plots. Reproduction requires the previously frozen P3-A2 D checkpoint weights or retraining with its public code/protocol.
