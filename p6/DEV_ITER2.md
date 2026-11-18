# Phase 6 development iteration 2

## Development decision

**Stop before protocol freeze.** Three-step attention with parity improved on fixed physics with parity in every regime. It tied matched local parity in R1, reduced failures in R2 and had one additional failure in R3. Several physical and decoding targets remain open on these three development pairs.

| Regime | Total words | Fixed + parity failures | Axial parity | Local parity | Axial − fixed | Axial − local |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 384 | 29 | 24 | 24 | -5 | 0 |
| R2 | 1728 | 429 | 373 | 383 | -56 | -10 |
| R3 | 384 | 65 | 58 | 57 | -7 | +1 |

**R1.** On the trained attention arm, suppressing the PA correction moved the failure count from 24 to 20. Suppressing common phase alone gave 28 and suppressing both gave 25. These interventions interact. The user's three-step joint true-PHN/true-PA run recorded 19 failures, leaving a 5-word gap for the fully enabled receiver.

**R2.** Suppressing the per-ray phase update moved attention from 373 to 390 failures. Suppressing the general CFR residual gave 399. The phase formula allows unbounded accumulation. The largest final absolute correction was 0.362 rad and 1/288 burst–ray trajectories spanned more than 0.5 rad across symbols. The supplied three-step nearest-bank and true-step-1 diagnostics recorded 362 and 329 failures. The attention receiver finished 11 and 44 words above them, respectively. Those oracle figures and the R1 oracle come from the user's earlier measurements.

**R3.** Suppressing fractional delays moved attention from 58 to 69 failures. Suppressing the entire antenna branch gave 57. Trained spatial Pearson was 0.15367, slightly below its 0.15419 initialization. The trained direct joint (0.34217), historical product (0.39788) and spatial readouts all exceeded 0.084. The strong initialization and the branch suppression results limit what the correlations establish about learning and decoded benefit.

**BCE and decoding.** Between updates 24 and 48 in R2, attention BCE fell from 0.24033 to 0.23736 and failed words moved from 374 to 373. Local BCE also fell (0.23937 to 0.23605) while its failed words rose from 376 to 383. R3 attention BCE fell from 0.15384 to 0.14948 while failures rose from 53 to 58. These divergences persisted after removing free per-resource-element variance. Every primary comparison uses the scheduled final update.

## Receiver and fit

The primary comparison uses three waveform steps and 48 Adam updates. Pairs 901, 902 and 903 use stream 5 for fitting and stream 6 for validation. Stream 8 supplies R3 probe bursts and stream 9 supplies permutation seeds.

All five learned arms use 856 trainable real parameters and a frozen copy of the initial operator. Attention and local start with identical tensors. Their width-eight branches share full-frame features, estimated per-ray CFR inputs, physical handles, objective, training batches and optimizer schedule. The local arm applies three-tap depthwise mixing. The attention arm uses Hermitian coherence magnitude, rotary positions on time and frequency, and phase-steered values on the antenna branch. The no-gauge arm exposes absolute receive units to the learned head.

An initial operator pass uses pilot residuals and the initial channel to refine the first equalizer. Later trajectory updates accumulate in sample time. The inverse PA correction uses the reciprocal bounded Laguerre AM-AM/AM-PM gain on the oversampled equalized waveform. Its coefficients interpolate between learned burst endpoints. R3 uses bounded fractional delays and a dedicated spatial CFR correction. The receiver uses the physical variance.

The training loss is payload BCE plus 0.02 times training-frame CFR NMSE. Batch size is 4. The global gradient norm limit is 1. Four warmup steps reach LR 0.0045, followed by cosine decay to one tenth of peak LR. Training truth is used for the NMSE loss. Inference receives the observation.

## Three-step receiver comparison

Each regime pools 96 validation bursts, one draw of 32 for each pair. BLER counts failed LDPC words after public normalized min-sum decoding. Pre-BER counts coded-bit errors before decoding. CFR NMSE pools squared channel error and truth energy before conversion to dB. Payload BCE is the unmodified mean over coded bits. Genie-PA receives the realized PA parameters. In R1 the channel estimate includes fitted oscillator phase and removes the nominal coherent PA gain.

### R1 at 12 dB

| Receiver | Failed / total | BLER | Pre-BER | NMSE (dB) | BCE | Δ errors vs fixed + parity | Δ errors vs matched local |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 40 / 384 | 10.42% | 9.15% | -10.12 | 0.22218 | +11 | — |
| Fixed + parity | 29 / 384 | 7.55% | 8.11% | -10.08 | 0.18750 | 0 | — |
| Axial pure | 37 / 384 | 9.64% | 8.98% | -10.16 | 0.21473 | +8 | -3 |
| Local pure | 40 / 384 | 10.42% | 9.03% | -10.13 | 0.21395 | +11 | — |
| Axial parity | 24 / 384 | 6.25% | 7.64% | -9.99 | 0.17769 | -5 | 0 |
| Local parity | 24 / 384 | 6.25% | 7.62% | -9.95 | 0.17702 | -5 | — |
| Axial no-gauge | 40 / 384 | 10.42% | 9.20% | -10.16 | 0.21643 | +11 | — |
| Nominal Bussgang LMMSE | 94 / 384 | 24.48% | 10.87% | -11.63 | 0.25730 | +65 | — |
| Nominal IDD PA | 81 / 384 | 21.09% | 10.64% | -11.72 | 0.27372 | +52 | — |
| Pilot-CPE-DD | 40 / 384 | 10.42% | 9.17% | -10.24 | 0.22261 | +11 | — |
| Genie-PA (privileged) | 38 / 384 | 9.90% | 9.10% | -11.53 | 0.21814 | +9 | — |

### R2 at 9 dB

| Receiver | Failed / total | BLER | Pre-BER | NMSE (dB) | BCE | Δ errors vs fixed + parity | Δ errors vs matched local |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 591 / 1728 | 34.20% | 12.21% | -14.77 | 0.27376 | +162 | — |
| Fixed + parity | 429 / 1728 | 24.83% | 11.56% | -18.47 | 0.25352 | 0 | — |
| Axial pure | 422 / 1728 | 24.42% | 11.54% | -19.34 | 0.25207 | -7 | -3 |
| Local pure | 425 / 1728 | 24.59% | 11.58% | -19.23 | 0.25257 | -4 | — |
| Axial parity | 373 / 1728 | 21.59% | 10.66% | -20.66 | 0.23736 | -56 | -10 |
| Local parity | 383 / 1728 | 22.16% | 10.52% | -20.45 | 0.23605 | -46 | — |
| Axial no-gauge | 426 / 1728 | 24.65% | 11.62% | -19.13 | 0.25320 | -3 | — |
| Tap phase slope | 1727 / 1728 | 99.94% | 42.24% | 2.61 | 2.63391 | +1298 | — |
| 2D-LMMSE | 1220 / 1728 | 70.60% | 15.67% | -10.77 | 0.33416 | +791 | — |
| Waveform-DD | 446 / 1728 | 25.81% | 11.54% | -18.49 | 0.25433 | +17 | — |

### R3 at 7 dB

| Receiver | Failed / total | BLER | Pre-BER | NMSE (dB) | BCE | Δ errors vs fixed + parity | Δ errors vs matched local |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 71 / 384 | 18.49% | 8.57% | -12.52 | 0.20197 | +6 | — |
| Fixed + parity | 65 / 384 | 16.93% | 7.84% | -12.90 | 0.18169 | 0 | — |
| Axial pure | 60 / 384 | 15.62% | 8.31% | -12.69 | 0.18878 | -5 | -2 |
| Local pure | 62 / 384 | 16.15% | 8.26% | -12.71 | 0.18857 | -3 | — |
| Axial parity | 58 / 384 | 15.10% | 6.09% | -12.88 | 0.14948 | -7 | +1 |
| Local parity | 57 / 384 | 14.84% | 6.01% | -12.75 | 0.14793 | -8 | — |
| Axial no-gauge | 62 / 384 | 16.15% | 8.32% | -12.65 | 0.19081 | -3 | — |
| 2D-LMMSE | 83 / 384 | 21.61% | 9.47% | -10.25 | 0.21338 | +18 | — |
| Waveform-DD | 65 / 384 | 16.93% | 7.97% | -12.89 | 0.18231 | 0 | — |

The following differences subtract the comparator from the attention receiver. Rate differences use percentage points.

| Regime | Receiver | Comparator | Δ failed words | Δ coded errors | Δ BLER (pp) | Δ Pre-BER (pp) | Δ NMSE (dB) | Δ BCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial pure | Fixed + parity | +8 | +428 | +2.08 | +0.87 | -0.081 | +0.02723 |
| R1 | Axial pure | Local pure | -3 | -25 | -0.78 | -0.05 | -0.028 | +0.00079 |
| R1 | Axial parity | Fixed + parity | -5 | -233 | -1.30 | -0.47 | +0.086 | -0.00981 |
| R1 | Axial parity | Local parity | 0 | +9 | +0.00 | +0.02 | -0.045 | +0.00067 |
| R2 | Axial pure | Fixed + parity | -7 | -51 | -0.41 | -0.02 | -0.870 | -0.00145 |
| R2 | Axial pure | Local pure | -3 | -100 | -0.17 | -0.05 | -0.111 | -0.00050 |
| R2 | Axial parity | Fixed + parity | -56 | -1999 | -3.24 | -0.90 | -2.194 | -0.01617 |
| R2 | Axial parity | Local parity | -10 | +296 | -0.58 | +0.13 | -0.215 | +0.00131 |
| R3 | Axial pure | Fixed + parity | -5 | +234 | -1.30 | +0.48 | +0.204 | +0.00709 |
| R3 | Axial pure | Local pure | -2 | +27 | -0.52 | +0.05 | +0.022 | +0.00021 |
| R3 | Axial parity | Fixed + parity | -7 | -857 | -1.82 | -1.74 | +0.020 | -0.03221 |
| R3 | Axial parity | Local parity | +1 | +43 | +0.26 | +0.09 | -0.131 | +0.00155 |

## Solver depth and training progress

Each count uses the same stream-6 B32 draw for each pair. Iteration 1 used 64 updates and constant LR 0.003. The depth comparison within iteration 2 holds the optimizer, batches and update count fixed. The cross-iteration differences include both model and training changes.

| Regime | Receiver | Iter 1, 2 steps | Iter 2, 2 steps | Iter 1, 3 steps | Iter 2, 3 steps | Iter 2 Δ (3−2) | 3 steps Δ (Iter 2−1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Fixed + parity | 32 | 32 | 29 | 29 | -3 | 0 |
| R1 | Axial parity | 32 | 31 | 26 | 24 | -7 | -2 |
| R1 | Local parity | 31 | 33 | 25 | 24 | -9 | -1 |
| R2 | Fixed + parity | 439 | 439 | 429 | 429 | -10 | 0 |
| R2 | Axial parity | 425 | 384 | 418 | 373 | -11 | -45 |
| R2 | Local parity | 427 | 383 | 418 | 383 | 0 | -35 |
| R3 | Fixed + parity | 66 | 66 | 65 | 65 | -1 | 0 |
| R3 | Axial parity | 55 | 56 | 57 | 58 | +2 | +1 |
| R3 | Local parity | 57 | 60 | 57 | 57 | -3 | 0 |

Three-step parity training traces show pooled validation failures / payload BCE. Every receiver uses the final scheduled update.

| Regime | Receiver | Update 0 | Update 24 | Update 48 |
| --- | --- | --- | --- | --- |
| R1 | Axial parity | 29 / 0.18750 | 27 / 0.18290 | 24 / 0.17769 |
| R1 | Local parity | 29 / 0.18750 | 27 / 0.17935 | 24 / 0.17702 |
| R2 | Axial parity | 429 / 0.25352 | 374 / 0.24033 | 373 / 0.23736 |
| R2 | Local parity | 429 / 0.25352 | 376 / 0.23937 | 383 / 0.23605 |
| R3 | Axial parity | 65 / 0.18169 | 53 / 0.15384 | 58 / 0.14948 |
| R3 | Local parity | 65 / 0.18169 | 57 / 0.15242 | 57 / 0.14793 |

## Physical-handle suppression

Each entry is the change in failed words after suppressing one family at every pass of the trained three-step receiver, without refitting. A positive difference means the enabled family saved words on these frames. Each full-model replay is checked and removing all handles is checked against fixed + parity. The whole-antenna intervention disables the branch in both live and frozen maps and preserves the scaling of the remaining branches. The dedicated antenna correction intervention leaves antenna inputs to the other readouts active.

### R1 paired interventions

| Suppressed family (attention / local) | 901 | 902 | 903 | Pooled Δ errors |
| --- | --- | --- | --- | --- |
| PA waveform correction | -2 / 0 | 0 / 0 | -2 / 0 | -4 / 0 |
| Common phase | 0 / +1 | +5 / +5 | -1 / -1 | +4 / +5 |
| PA + common phase | -2 / 0 | +4 / +5 | -1 / -1 | +1 / +4 |
| General CFR residual | +1 / +2 | 0 / +1 | -1 / 0 | 0 / +3 |
| All handles | -3 / -1 | +9 / +7 | -1 / -1 | +5 / +5 |

### R2 paired interventions

| Suppressed family (attention / local) | 901 | 902 | 903 | Pooled Δ errors |
| --- | --- | --- | --- | --- |
| Common phase | +8 / +2 | -1 / -5 | 0 / +1 | +7 / -2 |
| Per-ray trajectory phase | +7 / +6 | +4 / -1 | +6 / +10 | +17 / +15 |
| General CFR residual | +5 / +2 | +15 / +8 | +6 / +13 | +26 / +23 |
| All handles | +18 / +15 | +21 / +13 | +17 / +18 | +56 / +46 |

### R3 paired interventions

| Suppressed family (attention / local) | 901 | 902 | 903 | Pooled Δ errors |
| --- | --- | --- | --- | --- |
| PA waveform correction | +3 / 0 | -2 / +3 | +2 / +3 | +3 / +6 |
| Common phase | +2 / -1 | -3 / +3 | -2 / 0 | -3 / +2 |
| Per-ray trajectory phase | 0 / -2 | +1 / +1 | +1 / 0 | +2 / -1 |
| Fractional delay | +3 / +4 | +6 / +7 | +2 / 0 | +11 / +11 |
| Whole antenna branch | 0 / -2 | -4 / 0 | +3 / +1 | -1 / -1 |
| Dedicated antenna CFR correction | 0 / -1 | -3 / 0 | +2 / +1 | -1 / 0 |
| General CFR residual | +1 / -1 | -1 / 0 | +1 / 0 | +1 / -1 |
| All handles | +2 / 0 | 0 / +4 | +5 / +4 | +7 / +8 |

### R2 residual trajectory range

Angles are in radians. A trajectory is one burst and one ray. Its symbol span is the maximum minus minimum of the accumulated residual phase across symbol centers. Amplitude and span summarize the learned update. Phase recovery requires comparison to the simulated trajectory.

| Receiver | Pass (0-based) | Max absolute increment | Max absolute accumulated | Max symbol span | Spans above 0.5 rad |
| --- | --- | --- | --- | --- | --- |
| Axial parity | 0 | 0.097 | 0.097 | 0.134 | 0 / 288 |
| Axial parity | 1 | 0.171 | 0.255 | 0.364 | 0 / 288 |
| Axial parity | 2 | 0.117 | 0.362 | 0.545 | 1 / 288 |
| Local parity | 0 | 0.119 | 0.119 | 0.154 | 0 / 288 |
| Local parity | 1 | 0.194 | 0.312 | 0.454 | 0 / 288 |
| Local parity | 2 | 0.137 | 0.449 | 0.650 | 1 / 288 |

## R3 Green-kernel alignment

Pearson correlations compare each probe map to the magnitude of the conditional propagation covariance. Each arm uses eight stream-8 bursts per pair, all three passes, 48 coordinates and 2,256 off-diagonal entries per map. The shuffle uses eight stream-9 key permutations per pass. Means are taken within pair and then across pairs. Axis-only Pearson restricts pairs to the selected axis with both other coordinates fixed.

The direct parallel joint uses the unsigned mean branch weight on its actual axis lines. The historical product composes the three measured maps in F→T→R order for comparison with iteration 1. Iteration 2 applies its branches in parallel. The individual-axis overall columns retain the historical expansion across the full set of coordinate pairs. These probes omit the value projection, per-channel value gain, residual and nonlinear readout.

| Receiver | Kernel | Iter 1 trained | Iter 2 initial | Iter 2 trained | Initial shuffled | Trained shuffled | Trained axis-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | Direct parallel joint | — | 0.34101 | 0.34165 | 0.00148 | 0.00150 | — |
| Axial pure | Historical product | 0.01918 | 0.39523 | 0.39724 | -0.00211 | -0.00213 | — |
| Axial pure | Spatial | 0.01565 | 0.15372 | 0.15372 | -0.00171 | -0.00155 | 0.71403 |
| Axial pure | Time | -0.00654 | -0.01530 | -0.01424 | -0.00241 | -0.00220 | 0.37982 |
| Axial pure | Frequency | 0.03127 | 0.29388 | 0.29323 | 0.00152 | 0.00114 | 0.52858 |
| Axial parity | Direct parallel joint | — | 0.34112 | 0.34217 | 0.00149 | 0.00149 | — |
| Axial parity | Historical product | 0.02546 | 0.39625 | 0.39788 | -0.00213 | -0.00185 | — |
| Axial parity | Spatial | 0.01384 | 0.15419 | 0.15367 | -0.00166 | -0.00138 | 0.71190 |
| Axial parity | Time | -0.00659 | -0.01577 | -0.01631 | -0.00303 | -0.00276 | 0.37253 |
| Axial parity | Frequency | 0.04892 | 0.29489 | 0.29582 | 0.00161 | 0.00151 | 0.53872 |

Uniform mixing on the same three axis lines gives direct-joint Pearson 0.26870 (0.24605, 0.28659 and 0.27347 on pairs 901, 902 and 903 respectively). 1,872 of the 2,256 off-diagonal entries lie outside all three lines and carry zero direct weight. Relative to this support-matched control, trained pure attention changes direct-joint Pearson by +0.07294 and trained parity attention by +0.07347. Uniform mixing gives a constant historical product and hence an undefined Pearson coefficient.

## Training and inference time

Training seconds include stream-5 simulation, forward/backward and optimizer updates. Whole-job means also include validation and any applicable probes, suppressions and checkpoint writes. Each fit worker used one PyTorch thread.

| Steps | Regime | Receiver | Fit 901 (s) | Fit 902 (s) | Fit 903 (s) | Mean fit (s) | Mean job (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | R1 | Axial pure | 4.22 | 4.13 | 4.73 | 4.36 | 5.04 |
| 3 | R1 | Local pure | 3.46 | 3.89 | 3.47 | 3.61 | 4.09 |
| 3 | R1 | Axial parity | 4.41 | 4.86 | 3.38 | 4.22 | 6.42 |
| 3 | R1 | Local parity | 3.67 | 3.71 | 2.57 | 3.32 | 4.67 |
| 3 | R1 | Axial no-gauge | 3.46 | 4.47 | 3.09 | 3.68 | 4.23 |
| 3 | R2 | Axial pure | 8.59 | 7.84 | 7.98 | 8.14 | 10.06 |
| 3 | R2 | Local pure | 5.59 | 4.34 | 5.28 | 5.07 | 6.53 |
| 3 | R2 | Axial parity | 8.26 | 9.41 | 9.14 | 8.94 | 15.49 |
| 3 | R2 | Local parity | 8.11 | 6.09 | 7.23 | 7.14 | 11.56 |
| 3 | R2 | Axial no-gauge | 9.51 | 6.95 | 6.90 | 7.78 | 9.72 |
| 3 | R3 | Axial pure | 12.42 | 10.98 | 7.55 | 10.32 | 12.43 |
| 3 | R3 | Local pure | 6.72 | 7.32 | 4.98 | 6.34 | 7.46 |
| 3 | R3 | Axial parity | 9.33 | 8.50 | 7.55 | 8.46 | 15.53 |
| 3 | R3 | Local parity | 6.43 | 5.24 | 5.75 | 5.81 | 10.22 |
| 3 | R3 | Axial no-gauge | 8.75 | 7.64 | 6.30 | 7.56 | 9.13 |
| 2 | R1 | Axial parity | 1.92 | 2.01 | 1.96 | 1.97 | 2.32 |
| 2 | R1 | Local parity | 1.60 | 2.07 | 1.61 | 1.76 | 2.07 |
| 2 | R2 | Axial parity | 4.98 | 4.36 | 4.43 | 4.59 | 6.07 |
| 2 | R2 | Local parity | 3.39 | 4.10 | 3.04 | 3.51 | 4.68 |
| 2 | R3 | Axial parity | 4.56 | 5.36 | 4.77 | 4.90 | 5.99 |
| 2 | R3 | Local parity | 3.14 | 3.65 | 3.30 | 3.36 | 4.09 |

Forward latency is median wall time in milliseconds on pair 901 after 1 warmup and 5 timed forwards, with method order shuffled on every repetition. Torch used 1 compute thread and 1 inter-op thread. Simulation, metric scoring and final LDPC decoding fall outside the timer.

| Receiver | R1 B1 | R1 B64 | R2 B1 | R2 B64 | R3 B1 | R3 B64 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed | 4.6 | 49.1 | 5.5 | 142.0 | 4.8 | 69.0 |
| Fixed + parity | 4.7 | 47.6 | 7.0 | 184.5 | 5.0 | 81.9 |
| Axial pure | 13.6 | 312.5 | 22.2 | 1196.3 | 26.1 | 1438.3 |
| Local pure | 10.9 | 185.3 | 17.7 | 761.6 | 16.8 | 734.8 |
| Axial parity | 15.7 | 383.3 | 23.8 | 1237.4 | 26.9 | 1212.4 |
| Local parity | 14.6 | 196.9 | 18.9 | 805.9 | 17.7 | 770.3 |
| Axial no-gauge | 13.4 | 262.4 | 22.0 | 1145.6 | 23.8 | 1234.7 |
| Axial parity, 2 steps | 9.1 | 338.7 | 15.8 | 830.8 | 16.6 | 807.0 |
| Local parity, 2 steps | 8.1 | 230.0 | 12.1 | 517.3 | 15.2 | 644.9 |
| Nominal Bussgang LMMSE | 1.7 | 48.4 | — | — | — | — |
| Nominal IDD PA | 4.0 | 89.7 | — | — | — | — |
| Pilot-CPE-DD | 5.6 | 93.2 | — | — | — | — |
| Genie-PA (privileged) | 4.5 | 79.0 | — | — | — | — |
| Tap phase slope | — | — | 3.2 | 49.9 | — | — |
| 2D-LMMSE | — | — | 2.9 | 166.6 | 1.3 | 33.9 |
| Waveform-DD | — | — | 12.0 | 385.3 | 6.7 | 121.8 |

The timing harness screened for project fits and audits before measurement and before each round. Background percentages measure total host busy CPU after subtracting the benchmark process's CPU time, including operating-system and other application activity. Each range and median covers timed rounds at both batch sizes. Unreadable Python command lines limit worker identification.

| Regime | Pinned logical CPU / core / efficiency class | Background before run | Background range | Background median | Identified project workers | Unreadable Python processes |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 0 / 0 / 1 | 33.1% | 21.2–40.4% | 33.6% | 0 | 3 |
| R2 | 0 / 0 / 1 | 33.1% | 2.6–23.2% | 15.0% | 0 | 3 |
| R3 | 0 / 0 / 1 | 33.1% | 2.7–43.8% | 20.6% | 0 | 3 |

### Low-background timing supplement

Both three-step parity receivers used the same pair-901 observations and checkpoints as the first timing. The harness admitted a sample only when measured background CPU was at most 5% of the full host during both a preceding 0.5-second window and the timed group. The order of eligible cells was shuffled each cycle. Each sample is mean wall time per forward over 16 forwards at B1 or 2 at B64. Admission depends on measured CPU activity and project-process scans. Each regime had a 90-second search budget after warmup.

| Regime | Batch | Receiver | Attempts | Pre-window rejected | Timed group rejected | Other rejected | Accepted / required | Accepted group means (ms) | Median (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | B1 | Axial parity | 7 | 7 | 0 | 0 | 0/3 | — | — |
| R1 | B1 | Local parity | 8 | 8 | 0 | 0 | 0/3 | — | — |
| R1 | B64 | Axial parity | 8 | 8 | 0 | 0 | 0/3 | — | — |
| R1 | B64 | Local parity | 7 | 7 | 0 | 0 | 0/3 | — | — |
| R2 | B1 | Axial parity | 8 | 6 | 2 | 0 | 0/3 | — | — |
| R2 | B1 | Local parity | 8 | 8 | 0 | 0 | 0/3 | — | — |
| R2 | B64 | Axial parity | 7 | 7 | 0 | 0 | 0/3 | — | — |
| R2 | B64 | Local parity | 8 | 8 | 0 | 0 | 0/3 | — | — |
| R3 | B1 | Axial parity | 8 | 7 | 1 | 0 | 0/3 | — | — |
| R3 | B1 | Local parity | 7 | 6 | 1 | 0 | 0/3 | — | — |
| R3 | B64 | Axial parity | 8 | 7 | 1 | 0 | 0/3 | — | — |
| R3 | B64 | Local parity | 7 | 7 | 0 | 0 | 0/3 | — | — |

| Regime | Pinned logical CPU / core | Search (s) | Total accepted | Accepted group background CPU | Unreadable Python processes | Search outcome |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 2 / 2 | 91.7 | 0 | — | 3 | Window search ended |
| R2 | 2 / 2 | 91.7 | 0 | — | 3 | Window search ended |
| R3 | 2 / 2 | 92.9 | 0 | — | 3 | Window search ended |

0/12 cells reached the required 3 accepted groups. A dash in the median column marks incomplete coverage. Individual accepted group means are retained for those cells. Even a qualifying interval permits background activity up to the specified limit. [Per-attempt CPU and timing records](results/iter2_quiet_latency.json)

## Invariant and checkpoint checks

The pre-fit gate passed 13 P6 tests, 13 P5-B tests and 15 P5 tests. The unchanged P5/P5-B tests used redirected development fixtures. The P4 publication verifier exited successfully.

Initialized receivers matched their P6 fixed counterparts bit-for-bit in 30/30 depth/regime/arm checks. The maximum initial absolute LLR or CFR difference from the archived solver was 0. All 856 real parameters had finite, nonzero gradient components in each of 15 regime/arm combinations. All 75 tensor-wise directional finite differences passed, with maximum absolute error 3.73e-09 at half-width 1e−7.

Restored three-step gauge-enabled checkpoints passed 360/360 gain/condition checks. Across 4,313,088 coded decisions there were 0 flips. The required absolute grid LLR difference was <1e−12. The largest was 3.2e-13 for R1, pair 902, Axial pure, condition `last_pilot_row`. Maximum relative CFR L2 difference was 2.89e-15. Absolute receive gains ranged from 1e−35 to 1e20 with independent phases. Fixtures covered pilot erasures, all-zero observations, zero noise and ambiguous pilot phases. No-gauge is the explicit amplitude ablation.

The following post-inference truth comparisons average equally across the three pairs. R2 measures mean absolute wrapped trajectory error per symbol in radians after fitting one constant phase origin per burst and ray. R3 measures mean absolute delay error in Nyquist samples over the three movable rays.

| Regime | Receiver | Initial | After pass 0 | After pass 1 | After pass 2 |
| --- | --- | --- | --- | --- | --- |
| R2 | Axial parity | 0.1385 | 0.1365 | 0.1320 | 0.1328 |
| R2 | Local parity | 0.1385 | 0.1359 | 0.1310 | 0.1326 |
| R3 | Axial parity | 0.0742 | 0.0604 | 0.0495 | 0.0543 |
| R3 | Local parity | 0.0742 | 0.0602 | 0.0485 | 0.0543 |

[Iteration 2 measurements and receipts](results/dev_iter2.json) · [Iteration 1 measurements](results/dev_iter1.json)
