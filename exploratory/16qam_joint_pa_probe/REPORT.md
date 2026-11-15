# EM-FloatFormer — exploratory 16-QAM pilot/data receiver probe

**Date:** 2026-11-15. **Scope:** synthetic, passive civilian 32-tone OFDM, random QPSK known pilot + Gray 16-QAM data, 10 dB, memoryless PA magnitude compression and ordinary Gaussian thermal noise. **Not Sionna 2.2, not P3-D confirmation, not military/radar/EW or intentional interference.** Original P1/P2/P3 benchmark data and frozen reports are unchanged.

## Scientific question and what was actually built

The previous P3-B `DataResidualHead` **already consumed both received data and pilot-derived channel information**, so simply letting the model read data is not a novel step. The present trial extends the toy scenario to 16-QAM, for which amplitude calibration matters, and uses an additional small CNN data-aware residual demapper. The 1,812-parameter CNN consumes received pilot observations, received data, a DFT-8 pilot-channel estimate and reported noise variance, and predicts four **un-calibrated binary logits** per 16-QAM symbol. These are **not independently verified probability-calibrated LLRs**. The model does **not** jointly learn a new physical channel estimator: the channel estimate itself is still DFT-8.

Only ordinary transmitter PA saturation is modeled; **no burst narrowband interference, active jamming, avoidance algorithm, real signals or hardware** are part of the experiment.

## Protocol and execution provenance

The local protocol was written before formal 300-step training. A 1-step **smoke test mistakenly used the originally planned test RNG family**; before the formal training this was corrected to a disjoint held-out family and recorded in the protocol. The smoke results were never included in the formal dataset and model choices remained fixed. This is **not** an externally timestamped preregistration. Formal protocol SHA256 and code SHA256 are in `formal_results/audit.json`. No substantive adjustments occurred after the formal run.

- Three novel training seeds: `59101,59102,59103`; identical neural initialization per seed, two regimes `clean_train` versus `mixed_train`; 300 AdamW optimization steps, batch=64, learning rate .002, gradient clipping 1.
- Same training profile `train3=(.72,.20,.08)` for both regimes. Held-out profiles: train3, four equal taps, long6. Conditions: clean, PA mild, PA strong. Same observed test samples per method in each cell. SNR=10 dB referenced to uncompressed average symbol power.
- For each method and scenario: 3 seeds × 768 frames × 32 tones × 4 bits = **294,912 scored bits**, **73,728 16-QAM symbols**.
- Comparison methods: DFT-8 channel estimate with original 16-QAM slicer; same DFT-8 with threshold calibrated from *training-only* labels; hard-threshold DFT channel estimator with fixed slicer; two 1,812-param data-aware neural heads.
- Historical QPSK BER values are not directly comparable with these 16-QAM BER values.

## Measured results — pooled BER (%) at 10 dB; lower is better

| Profile | Condition | DFT-8 fixed | DFT-8 calibrated (train data) | DFT-threshold fixed | Neural clean-trained | Neural mixed-trained |
|---|---|---:|---:|---:|---:|---:|
| train3 | clean | 14.673 | 13.865 | 13.980 | 13.645 | 13.601 |
| train3 | mild PA | 15.775 | 14.924 | 15.146 | 14.750 | 14.740 |
| train3 | strong PA | 19.905 | 18.901 | 19.246 | 18.721 | 18.730 |
| equal4 | clean | 14.539 | 13.697 | 14.026 | 13.579 | 13.547 |
| equal4 | mild PA | 15.476 | 14.533 | 14.985 | 14.412 | 14.406 |
| equal4 | strong PA | 19.760 | 18.800 | 19.357 | 18.628 | 18.609 |
| long6 | clean | 14.668 | 13.834 | 14.763 | 13.692 | 13.691 |
| long6 | mild PA | 15.444 | 14.605 | 15.569 | 14.419 | 14.376 |
| long6 | strong PA | 19.687 | 18.643 | 20.031 | 18.422 | 18.436 |

At equal4/strong PA the mixed neural demapper has BER **18.609%** versus training-calibrated conventional slicer **18.800%** (a *0.192 BER percentage-point* improvement). The three seedwise differences have the same direction, but `n=3` is not a confirmatory inference: the smallest possible two-sided exact sign test p-value is 0.25. The effect size is small; no claim of meaningful superiority is justified from this alone. Mixed training does not consistently outperform clean training.

## Post-hoc fairness sensitivity: equal classical calibration frames

The original classical threshold calibration used 3,072 independent train3 frames per seed, compared with 19,200 training frames per neural head. After observing the formal results, a **separately labeled sensitivity check** fitted the single classical magnitude threshold on exactly **19,200 train3 frames per seed**, drawn from the neural mixed arm's original training RNG and impairment sequence. This is training-only data with no held-out labels, but its addition is *post-hoc*, not part of the frozen primary comparison.

| Profile | Strong-PA original 3,072-frame threshold BER | Strong-PA 19,200-frame threshold BER | Mixed-neural strong-PA BER |
|---|---:|---:|---:|
| train3 | 18.901% | 18.874% | 18.730% |
| equal4 | 18.800% | 18.788% | 18.609% |
| long6 | 18.643% | 18.619% | 18.436% |

The neural demapper's directional margin remains modest after matching calibration-frame counts, but even this comparison does **not** exhaust strong classical alternatives. Full nonlinear-PA-aware iterative correction / decoding and a fully data-aided probabilistic 16-QAM receiver were not implemented. Therefore the suggested definitive **“beat DFT-Thresh plus classical clipping repair, otherwise end the project”** test has **NOT** been completed.

## Runtime costs

Measured PyTorch CPU single-frame inference on the same environment (includes DFT8 for both, excludes data generation and real hardware):

| Method | Median latency, batch1 | Median latency, batch64 |
|---|---:|---:|
| Joint CNN bit-logit decoder | 0.283 ms | 0.655 ms |
| DFT-8 + fixed or calibrated slicer | ~0.050 ms | ~0.085 ms |
| Hard-threshold DFT + fixed slicer | 0.058 ms | 0.101 ms |

Neural decoding is approximately 5.7× slower in this CPU forward path for one frame, with no clear engineering tradeoff established.

## Validation

- Unit checks passed: exact Gray 16-QAM encode/decode in the noiseless limit, magnitude compression monotonicity / no amplitude expansion, clean zero-noise frequency response, finite network output/gradients, <3,000 trainable parameters.
- Independent explicit CP-based causal FIR convolution matches the equivalent `H × FFT(PA(IFFT(X)))` generator to `<2×10⁻⁶` max complex error in tested samples.
- Three independently reloaded neural checkpoint sets; **135/135 error/bit/symbol-error count rows matched exactly** against a second pass of held-out test generation.
- **45/45 aggregate rows reconciled** from original per-seed CSV; model/protocol SHA256 matching; no test truth enters the receivers.
- Results reflect the same codebase, same CPU and simulation only; not an independent lab or RF benchmark.

## Scientific conclusion and next gate

This narrow toy experiment suggests a small data-aware residual CNN can improve a DFT-8 + fixed or train-calibrated 16-QAM slicer by a fraction of a BER percentage point on some conventional nonlinear-PA settings. It **does not** establish a new general RF receiver, robust performance under real electronics, statistically confirmed improvement, or superiority over strong model-based clipping-repair algorithms. No Sionna 3GPP or full P3-D confirmatory testing has been performed. Avoid tuning into intentional interference scenarios. The appropriate next scientific gate would be a separately frozen head-to-head with properly matched strong nonlinear-distortion-aware classical demodulators under a non-adversarial civilian channel model, not a victory declaration.

Raw reproducible evidence: `probe.py`, `audit.py`, `timing.py`, `calibration_sensitivity.py`, `formal_results/ber_by_seed.csv`, `formal_results/ber_summary.csv`, `formal_results/equal_budget_calibration.csv`, `formal_results/audit.json`, `formal_results/independent_audit.json` and `formal_results/timing.json`. Six model checkpoint files are included in the private/complete archive, not necessarily in a lightweight public release.
