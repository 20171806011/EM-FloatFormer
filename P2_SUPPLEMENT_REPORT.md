# EM-FloatFormer — P2 plan-alignment supplementary audit results

**Completed: 2026-11-15.** **New exploratory experiments** under a separate protocol with SHA256 `1f7cc522d6f44a7a1102d68addab514a502fc31717a84a5a3d9d4a1019ae698a`; local freeze git commit `682f868f7a8bdca70a83943b8a611723c3c9850c` recorded *before any S1/S2/S3 training*. The GitHub publication is later than execution, so this is **not** an externally timestamped preregistration. The original P2-E frozen protocol, report, endpoints, seeds and conclusion are **unchanged**. No active transmission, radar, military systems or hardware control.

## Plan → gap → evidence closure

| Original P2 scope | Confirmed prior omission | New evidence | Conclusion |
|---|---|---|---|
| P2-B separate RMS-only/local normalized IQ | P2-B had only `raw`, `log`, frame-wide `dual`, not independent RMS-only or local 5-tone window | `p2_supplement.py` S1; `s1_results/ber_by_seed.csv`, `ber_summary.csv` | **Performed**, six paired representations, including global and local RMS-only controls. |
| P2-D sparse pilot densities | Existing OFDM observation had 32/32 known pilot tones | S2, masked pilot 32/16/8/4, 4 classical controls; masks/linear algebra independently checked | **Performed**, limited to evenly spaced pilots, fixed per-tone pilot power. |
| Independent mixed-channel training route | Original P2 A–E train3 only; matched-test prior not information-matched | S3 separate 400-step 3-seed train3 vs mixture arms and finite-training covariance estimates | **Performed**, 2 architectures × 2 regimes; held-out late3/eight_tap. |

## Frozen supplementary design and provenance

- New seeds: S1 `34001/34002/34003`, S2 `35001/35002/35003`, S3 `36001/36002/36003`, **none overlap** P2-E `2201..2210`. Each arm is trained from scratch for **400** steps with batch64 (shorter than the old 600-step P2-E runs and **not comparable as a confirmatory replication**). AdamW lr0.002, weight decay 1e-4, gradient clip 1.0, SNR 0–20 dB.
- New evaluation: **512 independent frames per seed/scenario**, 32 uncoded data QPSK carriers × 2 = 64 bits/frame, so **98,304 pooled bits** per arm/scenario across 3 seeds. Evaluation SNR 0/10/20 dB; tables below highlight 10 dB. Paired receiver predictions use identical sample frames per scenario.
- Logged raw integer error counts: **S1 162 rows, S2 540 rows, S3 486 rows**. Saved **48 checkpoint and training-prior files** with SHA256; files excluded from lightweight GitHub publication but included in full offline archive. No confirmatory p-values, no post-test tuning or added training steps.
- Earlier P2-E used ten fresh seeds and Holm-corrected exact tests: **its original two endpoints and p=0.00390625 remain intact**, without these supplements changing their interpretation. The narrower frozen finding remains that P2-E phase-equivariant model beat the earlier P1 CNN on those synthetic test profiles; DFT8 still won against the new neural model.

## S1 — RMS-only vs explicit log-scale side channel

BER at 10 dB, %, lower is better:

| Profile | Raw | Log/phase | Global RMS-only | Global RMS+log | Local 5-tone RMS-only | Local RMS+log |
|---|---:|---:|---:|---:|---:|---:|
| train3 | 5.332% | 5.428% | 4.919% | 4.944% | 5.769% | 5.334% |
| equal4 | 8.061% | 7.169% | 6.624% | 6.497% | 7.127% | 7.557% |
| long6 | 8.875% | 8.011% | 8.896% | 8.855% | 7.748% | 8.184% |

**Frozen paired contrasts** (mean of 3 seed-wise differences, in BER **percentage points**; positive `RMS-only − dual` means the log side path helped):

| Test profile | Contrast | Mean pp | Per-seed pp |
|---|---|---:|---|
| equal4 | `rms_global_only` − `dual_global` | +0.127 | +0.223, +0.177, -0.018 |
| long6 | `rms_global_only` − `dual_global` | +0.041 | +0.150, +0.150, -0.177 |
| equal4 | `rms_local_only` − `dual_local` | -0.430 | -0.500, -0.278, -0.513 |
| long6 | `rms_local_only` − `dual_local` | -0.435 | -0.412, -0.226, -0.668 |

**Interpretation:** Adding log RMS to frame-wide RMS normalized IQ helped slightly in the two specified OOD profiles, but with the local five-tone normalizer the log side branch **hurt** on both. At long6 the local-RMS-only branch beat local dual by about -0.435 pp under the defined sign convention. This is **not** a universal lossless representation result: RMS-only sets one channel to zero, auxiliary features and residual scaling contain other magnitude information, and changing neighborhoods changes inductive biases. All six models have the same 35,667 parameter topology/initialization per seed, but the supplemental training used new 400-step budgets (not the original P2-B 600). `s1_results/s1_ablation_10dB.png` plots all six representations.

## S2 — pilot density and fair sparse-pilot baselines

BER at 10 dB, %, lower is better:

| Profile | Known pilots | Circular sparse-LS interpolation | DFT8 + fixed ridge | Train3-prior LMMSE | Learned residual on train3-prior LMMSE | Test-profile LMMSE **privileged** |
|---|---:|---:|---:|---:|---:|---:|
| equal4 | 32 | 8.296% | 5.606% | 14.496% | 14.569% | 5.049% |
| equal4 | 16 | 7.668% | 6.564% | 14.793% | 14.841% | 5.616% |
| equal4 | 8 | 9.707% | 8.297% | 15.351% | 15.375% | 6.640% |
| equal4 | 4 | 24.167% | 28.207% | 16.726% | 16.633% | 8.341% |
| long6 | 32 | 8.234% | 5.541% | 13.898% | 13.935% | 5.333% |
| long6 | 16 | 7.800% | 6.504% | 14.145% | 14.193% | 6.088% |
| long6 | 8 | 11.753% | 8.206% | 14.776% | 14.775% | 7.218% |
| long6 | 4 | 22.381% | 26.872% | 19.044% | 18.911% | 15.410% |

**Interpretation:** Classical estimates and learned residual use only the selected pilot observations. The neural model uses the train3-prior LMMSE estimate as its base; it does not see missing pilot observations. This helps on the training profile but remains highly biased on unknown four- and six-tap laws: its performance is close to the **mismatched** train3-prior LMMSE, often much worse than a generic DFT8 with 8–32 pilots. With just 4 pilots, an 8-tap channel is underdetermined and DFT8-ridge loses substantial accuracy. The matched-test-prior LMMSE in the last column knows the test profile and is not equal-information for OOD. Dropping pilot tones also drops total pilot energy at fixed per-tone power: this is **not** a constant-total-training-energy comparison. The baseline uses complex periodic interpolation, a fixed-ridge delay-domain estimator, a train-prior posterior and a privileged matched-profile posterior. 0/20 dB results and **train3** results are preserved in CSV. No trained model is claimed to be optimal.

## S3 — mixed channel information on both sides

BER at 10 dB, %, lower better. `mixture` training population = 1/4 each train3, equal4, long6, two_tap. `late3` and `eight_tap` were held out:

| Test profile | CNN train3 | CNN mixture | Equivariant train3 | Equivariant mixture | Empirical LMMSE train3 prior | Empirical LMMSE mixture prior | Fixed DFT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train3 | 5.477% | 5.603% | 8.913% | 8.910% | 4.777% | 5.099% | 5.382% |
| equal4 | 6.429% | 5.715% | 8.547% | 8.623% | 13.108% | 4.927% | 5.241% |
| long6 | 8.506% | 6.442% | 8.904% | 8.995% | 13.356% | 5.251% | 5.403% |
| two_tap | 5.319% | 5.509% | 8.799% | 8.749% | 4.733% | 5.042% | 5.434% |
| late3 | 5.433% | 5.377% | 8.545% | 8.577% | 4.656% | 4.969% | 5.333% |
| eight_tap | 11.885% | 7.778% | 9.010% | 9.138% | 16.637% | 8.681% | 5.597% |

**Predeclared held-out architecture contrasts:** `train3 BER − mixture BER` (positive = mixed training helps), percentage points; 3 new seeds, descriptive only:

| Architecture | Unseen profile | Mean pp | Per-seed pp |
|---|---|---:|---|
| `res_equiv_local` | late3 | -0.033 | -0.031, -0.040, -0.027 |
| `res_equiv_local` | eight_tap | -0.128 | -0.107, -0.095, -0.183 |
| `p1_cnn_log` | late3 | +0.056 | +0.073, +0.015, +0.079 |
| `p1_cnn_log` | eight_tap | +4.107 | +3.683, +4.620, +4.016 |

- Mixing training profiles improved CNN on long6 and entirely unseen eight_tap, but not on every profile: for example two_tap CNN could worsen. This should not be confused with extrapolation from a train3-only model, because `equal4` and `long6` **are included in mixture training**.
- The 2,060-parameter equivariant model did **not** gain comparable performance in the mixed-information route under the frozen **400-step** budget. It remained near 8.5–9.1% on several profiles, including train3. Possible optimization underfitting is a hypothesis, not experimentally proven; we did not increase training steps after viewing outcomes.
- Traditional empirical covariance LMMSE was estimated from the **same 25,600 training pilot frames per seed/regime**, with average noise variance subtracted and a fixed PSD projection. Thus it uses the same declared training distribution and no test-specific covariance. It improved markedly on mixture-member equal4/long6; it remained mismatched on held-out eight_tap. Its parametric family (linear covariance estimator) and supervision differ from CNN, so equal exposure does not imply identical statistical capacity. `matched_test_privileged` is separately reported only in raw evidence.
- A fixed DFT8 delay-support prior (known support 0–7, no per-profile power prior) remains a strong classical benchmark. On held-out eight_tap, it outperformed all the newly trained neural models. There is no generalized neural-algorithm superiority claim.

## Independent technical checks, CPU cost and scope boundaries

- `python -m unittest test_p1d_invariants test_p1e_integrity test_p2_integrity test_p2_supplement_independent -v` returned **35/35 PASS**. Includes 13 new independent supplement tests: masked-pilot nonleakage even after corrupting all hidden pilot samples by large values; exact sparse-pilot interpolation at observed tones; independent Fourier/tap-space sparse posterior vs full covariance computation; DFT ridge normal-equation residual; 32-pilot posterior vs original code; RMS channel isolation; matched bits and noise in mixed/train3 streams; reconstructed BER error-count bookkeeping and three separate checkpoint replays. Mathematical checks are numerical within stated tolerances.
- CPU latency (PyTorch 2.10 CPU; batch1, 100 repetitions after warmup). These times exclude hardware, FFT/ADC, protocol, signal acquisition, host transfers and end-to-end RF latency; batch64 and p95 in `p2_supplement/latency_cpu.csv`:

| Branch | Method | Median ms | p95 ms |
|---|---|---:|---:|
| s1 | rms_global_only | 0.498 | 0.875 |
| s1 | dual_global | 0.482 | 0.754 |
| s1 | rms_local_only | 0.499 | 0.688 |
| s1 | dual_local | 0.498 | 0.647 |
| s2 | sparse_lmmse_train3 | 0.032 | 0.036 |
| s2 | sparse_dft8_ridge | 0.064 | 0.123 |
| s2 | lmmse_plus_equiv_neural | 0.287 | 0.561 |
| s3 | res_equiv_local | 0.195 | 0.265 |
| s3 | p1_cnn_log | 0.464 | 0.661 |
| s3 | mixture_empirical_lmmse | 0.051 | 0.081 |

- Deterministic replay verifies 6 S1 methods at equal4/10dB, a masked-pilot learned S2 method at long6/10dB, and 4 S3 methods at eight_tap/10dB for one new seed per branch (11 integer-count cells). It is **not** an independent codebase or physical replication. No tests on live RF, sample-rate variation, coding, real oscillators, 5G specifications, asynchronous multi-frame fading or active emission.
- **Local-only freeze provenance:** the signed chronological implication cannot be independently established from the later GitHub upload; it is an internal freeze commit before execution, with printed content hash, not a third-party preregistration timestamp.
- S1 seed34001 checkpoints completed before a process-level time limit; separate training runtime for that seed was not retained, so per-seed timing is blank in `training_registry.csv`. Its checkpoint protocol SHA and saved-count replay passed. 34002/34003 and all other training timings are preserved.

## Scope conclusion

The three omissions are now **substantively tested**, not assumed successful. They produce mixed or negative results, reinforcing that representation, pilot availability, channel-prior information and optimization budget can dominate architectural comparisons. The original P2-E freeze and conclusions stand separate. The final scope has progressed from original P2-A–E (full-pilot/train3) to **P2-S1/S2/S3 independently logged exploratory supplements**; neither set establishes superior general RF intelligence or an operational receiver.

Evidence: `P2_SCOPE_ALIGNMENT_AUDIT.md`, `P2_SUPPLEMENT_PROTOCOL.md`, `p2_supplement.py`, `test_p2_supplement_independent.py`, `p2_supplement/*_results/*.csv`, new training-prior SHA256 manifest, and original P2-E files unchanged.
