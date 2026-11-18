# Phase 6 development study — Iteration 3

## Scientific decision

<!-- BEGIN SCIENTIFIC DECISION -->
**Keep the formal protocol unlocked.** On dev pairs 901–903, the new trajectory solver repaired genuine sign errors, and both requested R3 handles reduced pooled decoding failures. R1 missed its PA target and the cap did not resolve the late BCE–BLER divergence. At the common final update, Local-Parity had fewer failures in all three regimes.

| Regime | Words | Fixed + parity | Iteration 2 Axial-Parity (48) | Iteration 3 Axial-Parity (80) | Iteration 3 Local-Parity (80) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | 384 | 29 | 24 | 29 | 28 |
| R2 | 1,728 | 429 | 373 | 378 | 370 |
| R3 | 384 | 65 | 58 | 57 | 55 |

**R1 PA.** Full Axial-Parity had 29 failures against the target of 20 or fewer. Keeping forward PA emission and switching off receive-side gain adjustment and dewarping gave 27. Suppressing the entire learned PA residual gave 26, one fewer failure on each pair than the full model. The bounded dewarp controls the size of any single constellation correction. Its decoded cost on these frames persisted. These are interventions on the fitted receiver.

**R2 trajectory.** Mean absolute wrapped row-phase error fell across all three passes, from 0.1385 rad initially to 0.1360, 0.1128 and 0.1048. Final learned spans exceeded 0.5 rad on 56/288 burst–rays, up from 1/288 in Iteration 2. Measuring the sign of the net slope against simulator truth identified ten wrong-sign initial rays. Attention repaired five, Local repaired four, and neither introduced a new sign error. All 50 rays with initial mid-sample mismatch spans in (0.5, 3] rad already had the correct sign. Their attention phase MAE fell from 0.1892 to 0.0895 rad.

Suppressing the R2 trajectory raised Axial-Parity failures from 378 to 417. Suppressing only the soft bank raised them to 436. Omitting the third pass gave 388. The enabled receiver saved 51 words versus Fixed + parity and 68 versus Waveform-DD. It finished eight words behind matched Local and 49 above the previously supplied 329-word true-step-1 oracle. The shared bank also operates in Local, so the ablation establishes a solver benefit and cannot assign it to time attention alone. Without parity feedback, Axial-Pure beat Local-Pure by 13 words in R2 (388 versus 401) and three in R3 (57 versus 60).

**R3 spatial filtering.** Suppressing fractional delays raised Axial-Parity from 57 to 65 failures. Suppressing the dedicated spatial filter raised it to 59, with pair differences of +3, 0 and −1. Disabling the entire antenna mixing branch also gave 59. Against the simulator's true steering direction, filtering reduced pooled off-mode amplitude RMS to 0.580 times its pre-filter value. Spatial Green Pearson rose from −0.00029 to 0.00584 (shuffled 0.00043). Direct joint Pearson fell from 0.24217 to 0.23658, below the 0.26870 uniform axis-support control.

**Training objective.** Capped R2 Axial-Parity went from 358 failures at update 24 to 376 at 48 and 378 at 80 while raw validation BCE fell from 0.23464 to 0.23083 and 0.22925. The matched uncapped arm gave 358, 380 and 373 at the same checkpoints. In R3, capped and uncapped Axial-Parity both gave 56, 55 and 57 failures at updates 24, 48 and 80. The shared early improvement and subsequent rebound occurred under both losses. At update 80, mean absolute raw R3 logits were 11.19 with the cap and 11.16 without it. Aggregate confidence offers little evidence that the cap reduced inference-time confidence on these frames. All Iteration 3 primary comparisons use update 80. [Iteration 2 comparison and oracle provenance](DEV_ITER2.md)
<!-- END SCIENTIFIC DECISION -->

## Design and evaluation

The study uses development pairs 901, 902, 903. Each pair and regime has its own training seed and its own stream-6 evaluation draw of 32 bursts. All arms and checkpoints use the same evaluation draw within each pair and regime. Stream 5 supplies training batches. The R3 Green probe uses stream 8 with 8 bursts per pair and stream 9 for column shuffles. This is a repeated evaluation on development data. Formal cohorts remain unevaluated.

Attention and Local each use 856 trainable real parameters, width eight, 3 weight-tied EM passes, the same input features, physical solver and training batches. Their head tensors share an initialization within each pair and regime. The centered head subtracts a frozen copy of its initial map. Attention mixes all positions on each axis. Local mixes three adjacent positions. The parity arms add the same code-based soft-symbol feedback. NoGauge exposes absolute receive amplitude to its head.

Each fit uses Adam, batch size 4, gradient-norm limit 1, four warmup steps to learning rate 0.0045 and cosine decay to one tenth of peak over 80 updates. The primary comparison uses the prespecified final checkpoint. Training computes payload BCE on `10 × tanh(LLR / 10)`, adds 0.04 times training-frame CFR NMSE, and adds 5e-05 times the squared parameter displacement from the initial map. The uncapped parity control changes only the logits used in training BCE. Every validation metric and final LDPC decode uses raw LLRs.

**R1.** Each symbol contributes all-tone DC and centered soft-symbol power-contrast projections. A linear burst fit yields the bounded PA endpoints. The contrast projection rejects a frequency-constant field on each row. The DC projection retains that field. The forward waveform incorporates the nominal PA and learned residual. Equalization applies 20% of the estimated linear Bussgang residual to the channel and predicts a dewarp from bounded posterior symbols, with an additive magnitude limit of 0.04 constellation units.

**R2.** The first learned pass sees residuals from soft payload and known pilots. A per-symbol projection onto the delay dictionary drives a continuous sample-time phase update. Later passes also fit soft-symbol taps and use their coherence with a public chirp bank to estimate a gated trajectory residual. Each step accumulates its phase correction into the current rotation.

**R3.** The antenna readout controls a Hermitian projection of each fitted antenna-amplitude vector onto its estimated ULA steering mode before waveform prediction and CFR reconstruction. A signed strength allows shrinkage or expansion. Estimated ray features enter through a learned embedding. R2 and R3 set the learned common-phase increment to zero.

## Receiver comparison

BLER counts failed LDPC words after normalized min-sum decoding. Pre-BER uses hard coded-bit decisions before decoding. Post-BER uses decoded information bits. CFR NMSE pools channel error and truth energy across pairs before conversion to dB. The Genie-PA baseline receives the realized PA parameters.

### R1 at 12 dB

| Receiver | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ words vs fixed + parity | Δ words vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 40 / 384 | 10.42% | 9.15% | 0.88% | -10.12 | 0.22218 | +11 | — |
| Fixed + parity | 29 / 384 | 7.55% | 8.11% | 0.54% | -10.08 | 0.18750 | 0 | — |
| Axial pure | 40 / 384 | 10.42% | 9.18% | 0.86% | -10.40 | 0.21743 | +11 | +3 |
| Local pure | 37 / 384 | 9.64% | 9.02% | 0.75% | -10.33 | 0.21336 | +8 | — |
| Axial parity | 29 / 384 | 7.55% | 7.40% | 0.54% | -10.08 | 0.17631 | 0 | +1 |
| Local parity | 28 / 384 | 7.29% | 7.35% | 0.57% | -10.06 | 0.17485 | -1 | — |
| Axial no-gauge | 38 / 384 | 9.90% | 9.17% | 0.90% | -10.43 | 0.21731 | +9 | — |
| Nominal Bussgang LMMSE | 94 / 384 | 24.48% | 10.87% | 2.23% | -11.63 | 0.25730 | +65 | — |
| Nominal IDD PA | 81 / 384 | 21.09% | 10.64% | 2.24% | -11.72 | 0.27372 | +52 | — |
| Pilot-CPE-DD | 40 / 384 | 10.42% | 9.17% | 0.86% | -10.24 | 0.22261 | +11 | — |
| Genie-PA (privileged) | 38 / 384 | 9.90% | 9.10% | 0.75% | -11.53 | 0.21814 | +9 | — |

### R2 at 9 dB

| Receiver | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ words vs fixed + parity | Δ words vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 591 / 1728 | 34.20% | 12.21% | 2.19% | -14.77 | 0.27376 | +162 | — |
| Fixed + parity | 429 / 1728 | 24.83% | 11.56% | 1.45% | -18.47 | 0.25352 | 0 | — |
| Axial pure | 388 / 1728 | 22.45% | 11.38% | 1.27% | -20.36 | 0.24895 | -41 | -13 |
| Local pure | 401 / 1728 | 23.21% | 11.41% | 1.33% | -20.02 | 0.25003 | -28 | — |
| Axial parity | 378 / 1728 | 21.88% | 10.12% | 1.24% | -20.82 | 0.22925 | -51 | +8 |
| Local parity | 370 / 1728 | 21.41% | 10.04% | 1.24% | -20.73 | 0.22804 | -59 | — |
| Axial no-gauge | 399 / 1728 | 23.09% | 11.44% | 1.31% | -19.91 | 0.25031 | -30 | — |
| Tap phase slope | 1727 / 1728 | 99.94% | 42.24% | 42.76% | 2.61 | 2.63391 | +1298 | — |
| 2D-LMMSE | 1220 / 1728 | 70.60% | 15.67% | 8.15% | -10.77 | 0.33416 | +791 | — |
| Waveform-DD | 446 / 1728 | 25.81% | 11.54% | 1.56% | -18.49 | 0.25433 | +17 | — |

### R3 at 7 dB

| Receiver | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ words vs fixed + parity | Δ words vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 71 / 384 | 18.49% | 8.57% | 2.10% | -12.52 | 0.20197 | +6 | — |
| Fixed + parity | 65 / 384 | 16.93% | 7.84% | 1.85% | -12.90 | 0.18169 | 0 | — |
| Axial pure | 57 / 384 | 14.84% | 7.96% | 1.59% | -13.46 | 0.18216 | -8 | -3 |
| Local pure | 60 / 384 | 15.62% | 7.93% | 1.57% | -13.47 | 0.18158 | -5 | — |
| Axial parity | 57 / 384 | 14.84% | 6.07% | 1.53% | -13.40 | 0.14770 | -8 | +2 |
| Local parity | 55 / 384 | 14.32% | 6.11% | 1.36% | -13.31 | 0.14715 | -10 | — |
| Axial no-gauge | 58 / 384 | 15.10% | 7.96% | 1.63% | -13.36 | 0.18371 | -7 | — |
| 2D-LMMSE | 83 / 384 | 21.61% | 9.47% | 2.53% | -10.25 | 0.21338 | +18 | — |
| Waveform-DD | 65 / 384 | 16.93% | 7.97% | 2.02% | -12.89 | 0.18231 | 0 | — |

## Training trajectory and the BCE cap

Each checkpoint cell gives failed words / validation BLER / raw validation BCE on the same frame draw. A positive 24→48 change means additional failed words. All checkpoints belong to one continuous schedule per fit.

### R1

| Receiver | Training BCE | Update 0 | Update 24 | Update 48 | Update 64 | Update 80 | Δ words 24→48 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | capped | 40 / 10.42% / 0.22218 | 71 / 18.49% / 0.26171 | 41 / 10.68% / 0.21844 | 44 / 11.46% / 0.21939 | 40 / 10.42% / 0.21743 | -30 |
| Local pure | capped | 40 / 10.42% / 0.22218 | 84 / 21.88% / 0.27130 | 38 / 9.90% / 0.21829 | 43 / 11.20% / 0.21382 | 37 / 9.64% / 0.21336 | -46 |
| Axial parity | capped | 29 / 7.55% / 0.18750 | 29 / 7.55% / 0.18917 | 28 / 7.29% / 0.17475 | 27 / 7.03% / 0.17565 | 29 / 7.55% / 0.17631 | -1 |
| Local parity | capped | 29 / 7.55% / 0.18750 | 45 / 11.72% / 0.20183 | 31 / 8.07% / 0.17670 | 26 / 6.77% / 0.17526 | 28 / 7.29% / 0.17485 | -14 |
| Axial no-gauge | capped | 40 / 10.42% / 0.22218 | 70 / 18.23% / 0.25603 | 43 / 11.20% / 0.21563 | 41 / 10.68% / 0.21678 | 38 / 9.90% / 0.21731 | -27 |
| Axial parity | uncapped | 29 / 7.55% / 0.18750 | 30 / 7.81% / 0.19079 | 29 / 7.55% / 0.17559 | 28 / 7.29% / 0.17570 | 29 / 7.55% / 0.17651 | -1 |
| Local parity | uncapped | 29 / 7.55% / 0.18750 | 47 / 12.24% / 0.20546 | 32 / 8.33% / 0.17749 | 27 / 7.03% / 0.17569 | 29 / 7.55% / 0.17558 | -15 |

### R2

| Receiver | Training BCE | Update 0 | Update 24 | Update 48 | Update 64 | Update 80 | Δ words 24→48 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | capped | 591 / 34.20% / 0.27376 | 409 / 23.67% / 0.25077 | 392 / 22.69% / 0.24951 | 383 / 22.16% / 0.24906 | 388 / 22.45% / 0.24895 | -17 |
| Local pure | capped | 591 / 34.20% / 0.27376 | 433 / 25.06% / 0.25264 | 407 / 23.55% / 0.25157 | 393 / 22.74% / 0.25001 | 401 / 23.21% / 0.25003 | -26 |
| Axial parity | capped | 429 / 24.83% / 0.25352 | 358 / 20.72% / 0.23464 | 376 / 21.76% / 0.23083 | 367 / 21.24% / 0.22985 | 378 / 21.88% / 0.22925 | +18 |
| Local parity | capped | 429 / 24.83% / 0.25352 | 364 / 21.06% / 0.23429 | 377 / 21.82% / 0.22960 | 361 / 20.89% / 0.22884 | 370 / 21.41% / 0.22804 | +13 |
| Axial no-gauge | capped | 591 / 34.20% / 0.27376 | 408 / 23.61% / 0.25142 | 408 / 23.61% / 0.25111 | 391 / 22.63% / 0.25039 | 399 / 23.09% / 0.25031 | 0 |
| Axial parity | uncapped | 429 / 24.83% / 0.25352 | 358 / 20.72% / 0.23576 | 380 / 21.99% / 0.23071 | 369 / 21.35% / 0.22991 | 373 / 21.59% / 0.22930 | +22 |
| Local parity | uncapped | 429 / 24.83% / 0.25352 | 369 / 21.35% / 0.23570 | 376 / 21.76% / 0.23000 | 365 / 21.12% / 0.22887 | 366 / 21.18% / 0.22837 | +7 |

### R3

| Receiver | Training BCE | Update 0 | Update 24 | Update 48 | Update 64 | Update 80 | Δ words 24→48 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | capped | 71 / 18.49% / 0.20197 | 65 / 16.93% / 0.18834 | 63 / 16.41% / 0.18318 | 55 / 14.32% / 0.18282 | 57 / 14.84% / 0.18216 | -2 |
| Local pure | capped | 71 / 18.49% / 0.20197 | 59 / 15.36% / 0.18494 | 59 / 15.36% / 0.18268 | 58 / 15.10% / 0.18320 | 60 / 15.62% / 0.18158 | 0 |
| Axial parity | capped | 65 / 16.93% / 0.18169 | 56 / 14.58% / 0.15453 | 55 / 14.32% / 0.14992 | 56 / 14.58% / 0.14803 | 57 / 14.84% / 0.14770 | -1 |
| Local parity | capped | 65 / 16.93% / 0.18169 | 59 / 15.36% / 0.15413 | 56 / 14.58% / 0.14865 | 55 / 14.32% / 0.14675 | 55 / 14.32% / 0.14715 | -3 |
| Axial no-gauge | capped | 71 / 18.49% / 0.20197 | 61 / 15.89% / 0.18659 | 61 / 15.89% / 0.18459 | 54 / 14.06% / 0.18458 | 58 / 15.10% / 0.18371 | 0 |
| Axial parity | uncapped | 65 / 16.93% / 0.18169 | 56 / 14.58% / 0.15476 | 55 / 14.32% / 0.14990 | 57 / 14.84% / 0.14863 | 57 / 14.84% / 0.14801 | -1 |
| Local parity | uncapped | 65 / 16.93% / 0.18169 | 60 / 15.62% / 0.15404 | 57 / 14.84% / 0.14866 | 55 / 14.32% / 0.14702 | 55 / 14.32% / 0.14720 | -3 |

### Validation confidence

Each cell gives mean absolute raw LLR, percentage of coded bits with signed correct margin above 6, and percentage with signed margin below −6. Individual decoder failure mechanisms would require codeword-level interventions.

| Regime | Receiver | Training BCE | Update 0 | Update 24 | Update 48 | Update 64 | Update 80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial parity | capped | 9.09 / 47.19% / 0.08% | 9.09 / 47.39% / 0.12% | 9.15 / 47.91% / 0.07% | 9.15 / 47.90% / 0.07% | 9.14 / 47.76% / 0.10% |
| R1 | Local parity | capped | 9.09 / 47.19% / 0.08% | 9.10 / 47.22% / 0.18% | 9.19 / 48.17% / 0.09% | 9.19 / 48.16% / 0.09% | 9.18 / 48.11% / 0.10% |
| R1 | Axial parity | uncapped | 9.09 / 47.19% / 0.08% | 9.06 / 47.22% / 0.12% | 9.13 / 47.74% / 0.07% | 9.13 / 47.74% / 0.08% | 9.12 / 47.67% / 0.10% |
| R1 | Local parity | uncapped | 9.09 / 47.19% / 0.08% | 9.08 / 47.06% / 0.22% | 9.15 / 47.89% / 0.09% | 9.15 / 47.91% / 0.08% | 9.14 / 47.86% / 0.10% |
| R2 | Axial parity | capped | 6.06 / 33.10% / 0.01% | 5.90 / 32.12% / 0.01% | 5.99 / 32.78% / 0.01% | 5.97 / 32.63% / 0.01% | 5.97 / 32.66% / 0.01% |
| R2 | Local parity | capped | 6.06 / 33.10% / 0.01% | 5.92 / 32.21% / 0.01% | 5.99 / 32.79% / 0.01% | 5.96 / 32.59% / 0.01% | 5.97 / 32.63% / 0.01% |
| R2 | Axial parity | uncapped | 6.06 / 33.10% / 0.01% | 5.87 / 31.87% / 0.01% | 5.98 / 32.73% / 0.01% | 5.95 / 32.50% / 0.01% | 5.96 / 32.53% / 0.01% |
| R2 | Local parity | uncapped | 6.06 / 33.10% / 0.01% | 5.88 / 31.95% / 0.01% | 5.98 / 32.66% / 0.01% | 5.95 / 32.47% / 0.01% | 5.96 / 32.49% / 0.01% |
| R3 | Axial parity | capped | 10.68 / 50.81% / 0.05% | 11.07 / 52.88% / 0.04% | 11.12 / 53.33% / 0.05% | 11.20 / 53.76% / 0.05% | 11.19 / 53.76% / 0.05% |
| R3 | Local parity | capped | 10.68 / 50.81% / 0.05% | 11.08 / 52.96% / 0.05% | 11.14 / 53.42% / 0.04% | 11.22 / 53.90% / 0.04% | 11.21 / 53.85% / 0.05% |
| R3 | Axial parity | uncapped | 10.68 / 50.81% / 0.05% | 11.01 / 52.65% / 0.04% | 11.09 / 53.20% / 0.04% | 11.17 / 53.60% / 0.05% | 11.16 / 53.62% / 0.05% |
| R3 | Local parity | uncapped | 10.68 / 50.81% / 0.05% | 11.05 / 52.78% / 0.04% | 11.10 / 53.29% / 0.04% | 11.21 / 53.83% / 0.04% | 11.19 / 53.73% / 0.05% |

## Handle interventions on all development pairs

Each row intervenes on the final capped checkpoint without retraining. Pair cells give failed words and the change from the same pair’s full receiver in parentheses. A positive change means the enabled component prevented failures in this intervention. Interacting effects need not add. PA residual off retains the public nominal PA. Forward PA residual only keeps learned emission and disables the learned receive-side gain and dewarp. The third-pass intervention evaluates the first two passes of the trained three-pass receiver.

### R1

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 9 (0) | 14 (0) | 6 (0) | 29 | 0 | +0.00000 |
| Axial parity | PA residual off | 8 (-1) | 13 (-1) | 5 (-1) | 26 | -3 | -0.00005 |
| Axial parity | Forward PA residual only | 8 (-1) | 13 (-1) | 6 (0) | 27 | -2 | +0.00004 |
| Axial parity | Learned common phase off | 9 (0) | 15 (+1) | 4 (-2) | 28 | -1 | -0.00152 |
| Axial parity | PA residual and common phase off | 8 (-1) | 15 (+1) | 3 (-3) | 26 | -3 | -0.00146 |
| Axial parity | General CFR residual off | 10 (+1) | 12 (-2) | 6 (0) | 28 | -1 | +0.01296 |
| Axial parity | All handles off | 7 (-2) | 18 (+4) | 4 (-2) | 29 | 0 | +0.01119 |
| Local parity | Full receiver | 9 (0) | 14 (0) | 5 (0) | 28 | 0 | +0.00000 |
| Local parity | PA residual off | 8 (-1) | 13 (-1) | 3 (-2) | 24 | -4 | -0.00013 |
| Local parity | Forward PA residual only | 9 (0) | 15 (+1) | 4 (-1) | 28 | 0 | +0.00000 |
| Local parity | Learned common phase off | 9 (0) | 14 (0) | 4 (-1) | 27 | -1 | -0.00028 |
| Local parity | PA residual and common phase off | 8 (-1) | 13 (-1) | 4 (-1) | 25 | -3 | -0.00026 |
| Local parity | General CFR residual off | 10 (+1) | 14 (0) | 7 (+2) | 31 | +3 | +0.01431 |
| Local parity | All handles off | 7 (-2) | 18 (+4) | 4 (-1) | 29 | +1 | +0.01265 |

### R2

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 148 (0) | 118 (0) | 112 (0) | 378 | 0 | +0.00000 |
| Axial parity | Learned common phase off | 148 (0) | 118 (0) | 112 (0) | 378 | 0 | +0.00000 |
| Axial parity | Per-ray trajectory off | 159 (+11) | 124 (+6) | 134 (+22) | 417 | +39 | +0.00686 |
| Axial parity | Soft chirp-bank residual off | 168 (+20) | 132 (+14) | 136 (+24) | 436 | +58 | +0.00482 |
| Axial parity | Omit third EM pass | 147 (-1) | 119 (+1) | 122 (+10) | 388 | +10 | +0.00239 |
| Axial parity | General CFR residual off | 148 (0) | 109 (-9) | 114 (+2) | 371 | -7 | +0.01599 |
| Axial parity | All handles off | 163 (+15) | 127 (+9) | 139 (+27) | 429 | +51 | +0.02428 |
| Local parity | Full receiver | 142 (0) | 114 (0) | 114 (0) | 370 | 0 | +0.00000 |
| Local parity | Learned common phase off | 142 (0) | 114 (0) | 114 (0) | 370 | 0 | +0.00000 |
| Local parity | Per-ray trajectory off | 159 (+17) | 120 (+6) | 136 (+22) | 415 | +45 | +0.00679 |
| Local parity | Soft chirp-bank residual off | 161 (+19) | 134 (+20) | 127 (+13) | 422 | +52 | +0.00397 |
| Local parity | Omit third EM pass | 146 (+4) | 120 (+6) | 122 (+8) | 388 | +18 | +0.00250 |
| Local parity | General CFR residual off | 146 (+4) | 117 (+3) | 114 (0) | 377 | +7 | +0.01730 |
| Local parity | All handles off | 163 (+21) | 127 (+13) | 139 (+25) | 429 | +59 | +0.02549 |

### R3

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 11 (0) | 21 (0) | 25 (0) | 57 | 0 | +0.00000 |
| Axial parity | PA residual off | 11 (0) | 21 (0) | 25 (0) | 57 | 0 | +0.00018 |
| Axial parity | Learned common phase off | 11 (0) | 21 (0) | 25 (0) | 57 | 0 | +0.00000 |
| Axial parity | Per-ray trajectory off | 12 (+1) | 18 (-3) | 24 (-1) | 54 | -3 | +0.00073 |
| Axial parity | Path-amplitude filter off | 14 (+3) | 21 (0) | 24 (-1) | 59 | +2 | +0.00257 |
| Axial parity | Entire antenna mixing branch off | 15 (+4) | 21 (0) | 23 (-2) | 59 | +2 | +0.00285 |
| Axial parity | Fractional delays off | 16 (+5) | 24 (+3) | 25 (0) | 65 | +8 | +0.01222 |
| Axial parity | General CFR residual off | 13 (+2) | 20 (-1) | 24 (-1) | 57 | 0 | +0.01867 |
| Axial parity | All handles off | 15 (+4) | 24 (+3) | 26 (+1) | 65 | +8 | +0.03399 |
| Local parity | Full receiver | 11 (0) | 20 (0) | 24 (0) | 55 | 0 | +0.00000 |
| Local parity | PA residual off | 13 (+2) | 21 (+1) | 24 (0) | 58 | +3 | +0.00032 |
| Local parity | Learned common phase off | 11 (0) | 20 (0) | 24 (0) | 55 | 0 | +0.00000 |
| Local parity | Per-ray trajectory off | 12 (+1) | 18 (-2) | 24 (0) | 54 | -1 | +0.00075 |
| Local parity | Path-amplitude filter off | 14 (+3) | 21 (+1) | 24 (0) | 59 | +4 | +0.00274 |
| Local parity | Entire antenna mixing branch off | 14 (+3) | 21 (+1) | 23 (-1) | 58 | +3 | +0.00268 |
| Local parity | Fractional delays off | 17 (+6) | 24 (+4) | 23 (-1) | 64 | +9 | +0.01171 |
| Local parity | General CFR residual off | 12 (+1) | 20 (0) | 24 (0) | 56 | +1 | +0.01870 |
| Local parity | All handles off | 15 (+4) | 24 (+4) | 26 (+2) | 65 | +10 | +0.03454 |

## R2 trajectory depth

The correction span is the range across symbol rows of accumulated sample-averaged phase correction for one burst and ray. The largest span and counts pool all three pairs. The p95 column lists each pair’s 95th percentile in pair order.

| Receiver | Pass | Max increment magnitude | Max accumulated magnitude | Max symbol span | p95 span (901/902/903) | Spans >0.5 / total |
| --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 1 | 0.3603 | 0.3603 | 0.5182 | 0.2603 / 0.2459 / 0.2848 | 1 / 288 |
| Axial parity | 2 | 6.7047 | 6.6299 | 12.7873 | 0.7802 / 0.6109 / 0.6658 | 46 / 288 |
| Axial parity | 3 | 6.8198 | 10.9524 | 21.1768 | 0.9755 / 0.8898 / 0.7083 | 56 / 288 |
| Local parity | 1 | 0.3571 | 0.3571 | 0.5385 | 0.2501 / 0.2604 / 0.2612 | 1 / 288 |
| Local parity | 2 | 6.7234 | 6.6771 | 12.8110 | 0.7498 / 0.6549 / 0.6515 | 46 / 288 |
| Local parity | 3 | 6.7511 | 10.9706 | 21.1978 | 0.9683 / 0.9127 / 0.7141 | 58 / 288 |

Truth comparisons use the simulator after receiver inference. Mean absolute wrapped row error is in radians and allows one constant phase per burst and ray.

| Receiver | Pair | Burst-rays | Initial | Pass 1 | Pass 2 | Pass 3 |
| --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 96 | 0.15711 | 0.15535 | 0.13139 | 0.11885 |
| Axial parity | 902 | 96 | 0.13910 | 0.13542 | 0.11215 | 0.10898 |
| Axial parity | 903 | 96 | 0.11923 | 0.11719 | 0.09494 | 0.08643 |
| Local parity | 901 | 96 | 0.15711 | 0.15484 | 0.13103 | 0.11912 |
| Local parity | 902 | 96 | 0.13910 | 0.13391 | 0.11363 | 0.10989 |
| Local parity | 903 | 96 | 0.11923 | 0.11484 | 0.09292 | 0.08625 |

The initial mismatch span is the range of the unwrapped phase difference between the initial bank trajectory and truth. The (0.5, 3] and >3 groups partition the >0.5 group. A large mismatch can arise from an off-grid or wrong-sign trajectory. The diagnostic does not independently label their causes. Subgroup means weight each burst-ray equally.

| Receiver | Initial mismatch span (rad) | Counts (901/902/903) | Total | Initial error | Pass 1 | Pass 2 | Pass 3 | Lower final error than initial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | >0.5 | 24 / 21 / 17 | 62 | 0.43508 | 0.42385 | 0.34624 | 0.32036 | 57 |
| Axial parity | (0.5, 3] | 19 / 17 / 14 | 50 | 0.18927 | 0.17711 | 0.10002 | 0.08945 | 50 |
| Axial parity | >3 | 5 / 4 / 3 | 12 | 1.45927 | 1.45196 | 1.37213 | 1.28248 | 7 |
| Local parity | >0.5 | 24 / 21 / 17 | 62 | 0.43508 | 0.42185 | 0.34408 | 0.32160 | 57 |
| Local parity | (0.5, 3] | 19 / 17 / 14 | 50 | 0.18927 | 0.17545 | 0.09785 | 0.08891 | 50 |
| Local parity | >3 | 5 / 4 / 3 | 12 | 1.45927 | 1.44852 | 1.37005 | 1.29115 | 7 |

### Chirp direction against simulator truth

Net slope in radians per useful OFDM-symbol duration between the first and last mid-sample. True slope uses integrated simulator Doppler. Wrong sign means the fitted and true slopes have opposite signs. Phase MAE removes one constant angle per burst-ray and uses wrapped mid-sample errors.

The arrows compare initialization with the final pass. Repairs count wrong-to-correct sign changes. Regressions count correct-to-wrong changes. The zero-slope column accounts for indeterminate signs. Groups overlap. Mid-sample errors and spans here use the midpoint of each symbol. The preceding table uses symbol-averaged phasors.

| Receiver | Subset | Burst-rays | Wrong signs | Zero slope | Repaired | Regressed | Net slope MAE | Aligned phase MAE (rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | All rays | 288 | 10 → 5 | 0 → 0 | 5 | 0 | 0.03901 → 0.02159 | 0.13847 → 0.10475 |
| Axial parity | Initially wrong sign | 10 | 10 → 5 | 0 → 0 | 5 | 0 | 0.86071 → 0.48436 | 1.50385 → 1.42943 |
| Axial parity | Initially correct sign | 278 | 0 → 0 | 0 → 0 | 0 | 0 | 0.00945 → 0.00495 | 0.08936 → 0.05710 |
| Axial parity | Initial span (0.5, 3] rad | 50 | 0 → 0 | 0 → 0 | 0 | 0 | 0.02036 → 0.00918 | 0.18924 → 0.08946 |
| Axial parity | Initial span >3 rad | 12 | 10 → 5 | 0 → 0 | 5 | 0 | 0.74463 → 0.41629 | 1.45922 → 1.28241 |
| Axial parity | Initially wrong sign and span (0.5, 3] rad | 0 | 0 → 0 | 0 → 0 | 0 | 0 | — → — | — → — |
| Local parity | All rays | 288 | 10 → 6 | 0 → 0 | 4 | 0 | 0.03901 → 0.02179 | 0.13847 → 0.10509 |
| Local parity | Initially wrong sign | 10 | 10 → 6 | 0 → 0 | 4 | 0 | 0.86071 → 0.49079 | 1.50385 → 1.44074 |
| Local parity | Initially correct sign | 278 | 0 → 0 | 0 → 0 | 0 | 0 | 0.00945 → 0.00492 | 0.08936 → 0.05704 |
| Local parity | Initial span (0.5, 3] rad | 50 | 0 → 0 | 0 → 0 | 0 | 0 | 0.02036 → 0.00912 | 0.18924 → 0.08891 |
| Local parity | Initial span >3 rad | 12 | 10 → 6 | 0 → 0 | 4 | 0 | 0.74463 → 0.42154 | 1.45922 → 1.29108 |
| Local parity | Initially wrong sign and span (0.5, 3] rad | 0 | 0 → 0 | 0 → 0 | 0 | 0 | — → — | — → — |

| Receiver | Pair | Burst-rays | Initial wrong | Final wrong | Repaired | Regressed |
| --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 96 | 4 | 2 | 2 | 0 |
| Axial parity | 902 | 96 | 4 | 2 | 2 | 0 |
| Axial parity | 903 | 96 | 2 | 1 | 1 | 0 |
| Local parity | 901 | 96 | 4 | 2 | 2 | 0 |
| Local parity | 902 | 96 | 4 | 2 | 2 | 0 |
| Local parity | 903 | 96 | 2 | 2 | 0 | 0 |

## R3 fractional delay and Green-kernel measurements

Fractional-delay MAE uses the movable rays and is expressed in Nyquist samples. The nominal value measures the error at zero learned offset.

| Receiver | Pair | Nominal delay MAE | Pass 1 | Pass 2 | Pass 3 |
| --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 0.07243 | 0.05205 | 0.04850 | 0.04863 |
| Axial parity | 902 | 0.06977 | 0.04596 | 0.04898 | 0.04301 |
| Axial parity | 903 | 0.08040 | 0.05395 | 0.05787 | 0.05118 |
| Local parity | 901 | 0.07243 | 0.05297 | 0.04889 | 0.04912 |
| Local parity | 902 | 0.06977 | 0.04707 | 0.04842 | 0.04340 |
| Local parity | 903 | 0.08040 | 0.05299 | 0.05872 | 0.05082 |

### Signed antenna filter and off-mode residuals

Each stage compares the same waveform fit immediately before and after the branch. The estimated rank-one projector is held fixed. A second projection uses the true normalized ULA steering vector after inference. RMS ratios pool squared cross-antenna residual magnitudes.

Positive gain moves amplitudes toward the estimated mode. Negative gain moves them away. An off-mode RMS ratio below one means that the residual decreased relative to the indicated steering mode. The pooled waveform line counts all passes for each burst-ray and excludes the initial redemap.

| Receiver | Filter application | Applications | Positive / negative / zero | Mean gain | Gain p05 / p50 / p95 | Estimated-mode RMS ratio | True-mode RMS ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Pilot redemap | 384 | 381 / 3 / 0 | 0.76214 | 0.26048 / 0.84822 / 0.93138 | 0.30220 | 0.56558 |
| Axial parity | Waveform solve 1 | 384 | 381 / 3 / 0 | 0.76214 | 0.26048 / 0.84822 / 0.93138 | 0.30628 | 0.58880 |
| Axial parity | Waveform solve 2 | 384 | 380 / 4 / 0 | 0.74085 | 0.22803 / 0.83145 / 0.92529 | 0.34680 | 0.57699 |
| Axial parity | Waveform solve 3 | 384 | 381 / 3 / 0 | 0.74026 | 0.22320 / 0.82847 / 0.92362 | 0.34468 | 0.57192 |
| Axial parity | All waveform solves | 1152 | 1142 / 10 / 0 | 0.74775 | 0.23662 / 0.83833 / 0.92744 | 0.33194 | 0.57990 |
| Local parity | Pilot redemap | 384 | 380 / 4 / 0 | 0.75962 | 0.25265 / 0.84069 / 0.94671 | 0.32670 | 0.56891 |
| Local parity | Waveform solve 1 | 384 | 380 / 4 / 0 | 0.75962 | 0.25265 / 0.84069 / 0.94671 | 0.33958 | 0.59684 |
| Local parity | Waveform solve 2 | 384 | 383 / 1 / 0 | 0.75265 | 0.25603 / 0.82857 / 0.94521 | 0.33835 | 0.57871 |
| Local parity | Waveform solve 3 | 384 | 384 / 0 / 0 | 0.75501 | 0.28824 / 0.83335 / 0.94549 | 0.33671 | 0.57467 |
| Local parity | All waveform solves | 1152 | 1147 / 5 / 0 | 0.75576 | 0.25640 / 0.83580 / 0.94607 | 0.33829 | 0.58433 |

| Receiver | Pair, all waveform solves | Applications | Positive gain | Mean gain | Estimated-mode RMS ratio | True-mode RMS ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 384 | 97.92% | 0.72232 | 0.33885 | 0.58008 |
| Axial parity | 902 | 384 | 99.48% | 0.77093 | 0.31714 | 0.58163 |
| Axial parity | 903 | 384 | 100.00% | 0.75000 | 0.33956 | 0.57822 |
| Local parity | 901 | 384 | 99.74% | 0.70447 | 0.35937 | 0.58320 |
| Local parity | 902 | 384 | 99.74% | 0.83688 | 0.26810 | 0.57001 |
| Local parity | 903 | 384 | 99.22% | 0.72593 | 0.37522 | 0.59735 |

### Green-kernel correlations

Pearson correlation compares the measured attention weights with the evaluation-only conditional propagation Green kernel. Overall values use off-diagonal token pairs. Axis-only values restrict both tokens to one physical axis. Pair means average finite burst/pass coefficients, then the overall mean weights each pair equally. Shuffled values average eight stream-9 key permutations. The direct parallel joint uses the measured line supports of the added branches. The historical product composes their weights.

| Receiver | Kernel | Initial | Trained | Δ | Trained shuffled | Initial axis-only | Trained axis-only | Trained (901/902/903) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | Historical product | 0.06652 | 0.05575 | -0.01078 | 0.00064 | — | — | 0.05041 / 0.06960 / 0.04723 |
| Axial pure | Time | -0.00918 | -0.00885 | +0.00032 | -0.00203 | 0.33812 | 0.30074 | -0.01906 / 0.00964 / -0.01714 |
| Axial pure | Frequency | 0.10479 | 0.09091 | -0.01389 | 0.00385 | -0.05265 | -0.04295 | 0.08395 / 0.09090 / 0.09787 |
| Axial pure | Direct parallel joint | 0.24231 | 0.24513 | +0.00282 | 0.00177 | — | — | 0.22700 / 0.28627 / 0.22212 |
| Axial pure | Spatial | -0.00106 | 0.00351 | +0.00458 | -0.00005 | -0.00728 | 0.00220 | 0.01416 / 0.00471 / -0.00833 |
| Axial parity | Historical product | 0.06563 | 0.07642 | +0.01080 | 0.00237 | — | — | 0.05786 / 0.10504 / 0.06636 |
| Axial parity | Time | -0.00961 | -0.01058 | -0.00097 | -0.00056 | 0.33567 | 0.31570 | -0.01823 / 0.00824 / -0.02175 |
| Axial parity | Frequency | 0.10355 | 0.12426 | +0.02071 | 0.00463 | -0.05388 | -0.04278 | 0.09496 / 0.13352 / 0.14430 |
| Axial parity | Direct parallel joint | 0.24217 | 0.23658 | -0.00559 | 0.00185 | — | — | 0.22093 / 0.27294 / 0.21588 |
| Axial parity | Spatial | -0.00029 | 0.00584 | +0.00614 | 0.00043 | -0.00535 | 0.00806 | 0.02485 / 0.00030 / -0.00762 |

Uniform weights on the same three axial lines give direct-joint Pearson 0.26870 over 24 probe bursts (901 0.24605, 902 0.28659, 903 0.27347). Each burst contributes 2256 off-diagonal entries, including 1872 outside all three lines. The uniform historical product is constant and its Pearson coefficient is undefined. The uniform-support control quantifies alignment available from geometry alone. The antenna interventions measure decoding impact.

## Single-thread receiver timing

Cells are median wall milliseconds per forward on the timing pair’s stream-6 inputs. The harness shuffles method order within each round and excludes its warmup. Simulation, final LDPC decoding and scoring occur outside the timed forward.

| Receiver | R1 B1 | R1 B64 | R2 B1 | R2 B64 | R3 B1 | R3 B64 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed | 4.13 | 31.13 | 5.00 | 100.52 | 4.18 | 52.82 |
| Fixed + parity | 4.73 | 37.45 | 5.76 | 135.98 | 4.96 | 60.87 |
| Axial pure | 13.78 | 224.41 | 21.41 | 938.82 | 24.83 | 991.33 |
| Local pure | 12.61 | 165.48 | 15.51 | 583.48 | 17.88 | 607.24 |
| Axial parity | 15.39 | 240.78 | 21.72 | 964.58 | 25.80 | 1027.26 |
| Local parity | 13.75 | 174.70 | 17.03 | 607.88 | 19.41 | 618.77 |
| Axial no-gauge | 13.26 | 219.56 | 18.46 | 870.75 | 23.24 | 953.24 |
| Nominal IDD PA | 4.15 | 41.62 | — | — | — | — |
| Pilot-CPE-DD | 3.81 | 41.42 | — | — | — | — |
| Genie-PA (privileged) | 4.01 | 53.06 | — | — | — | — |
| Nominal Bussgang LMMSE | 1.18 | 16.47 | — | — | — | — |
| 2D-LMMSE | — | — | 2.62 | 127.61 | 1.24 | 20.06 |
| Tap phase slope | — | — | 2.03 | 37.32 | — | — |
| Waveform-DD | — | — | 11.08 | 291.03 | 6.09 | 83.30 |

Host busy is the percentage of total host CPU capacity in use across all processes. Background subtracts benchmark-process CPU from host busy. Each range covers the timed rounds for the specified batch. The load was observed during measurement.

| Regime | Batch | Pair | Timed rounds | Torch threads | Interop threads | Logical affinity | Host busy range | Background range | Background median | Quiet wait (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 1 | 901 | 5 | 1 | 1 | [1] | 20.3–27.0% | 10.9–18.8% | 11.5% | 0 |
| R1 | 64 | 901 | 5 | 1 | 1 | [1] | 15.5–25.0% | 7.3–16.2% | 12.6% | 0 |
| R2 | 1 | 901 | 5 | 1 | 1 | [1] | 9.3–19.5% | 0.0–10.3% | 3.4% | 0 |
| R2 | 64 | 901 | 5 | 1 | 1 | [1] | 13.4–25.3% | 5.6–17.7% | 8.2% | 0 |
| R3 | 1 | 901 | 5 | 1 | 1 | [1] | 27.6–33.3% | 18.4–26.0% | 20.7% | 0 |
| R3 | 64 | 901 | 5 | 1 | 1 | [1] | 13.5–24.2% | 5.5–15.9% | 6.6% | 0 |

## Verification

The combined verification receipt records **PASS**. P6 ran 21 tests with 0 skipped. The archived P5 and P5-B suites ran 15 and 13 tests. Their original test bodies used 30 distinct development frame configurations through the fixture adapter. `publication.verify_p4` returned exit code 0.

Initialization matched the P6 fixed solver bit-for-bit in 30/30 comparisons. The maximum archived initial LLR/CFR difference was 0. Across 15 regime/arm combinations, the active real gradient count ranged from 856 to 856 of 856 trainable components. The audit checked 75 tensor-wise finite-difference directions. The largest absolute directional error was 4.69e-09.

The final gauge-enabled checkpoints passed 360/360 condition checks at complex gains spanning 1e−35 to 1e20. There were 0 coded flips across 4,313,088 decisions. Maximum absolute grid ΔLLR was 3.162e-13 (R2, pair 903, Local pure, `intact`). Maximum relative CFR L2 difference was 3.292e-15. The gauge test excludes the intentional NoGauge control.

Implementation signature `6706ebe6548833e6889a7dff66d80d019d1c218ac6079324eb89b11b69004700`. [Verification receipt](results/iter3_checks/summary.json). [Full metrics, checkpoints, per-pair measurements and timing rounds](results/dev_iter3.json).
