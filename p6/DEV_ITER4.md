# Phase 6 development study — Iteration 4

## Scientific decision

**Keep the formal protocol open. The common 48-update recipe misses development targets.** The primary comparison uses the same update for all five fitted receivers in all three regimes.

| Regime | LDPC words | Fixed + parity | Iteration 3 Axial-Parity (80) | Iteration 4 Axial-Parity (48) | Iteration 4 Local-Parity (48) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | 384 | 29 | 29 | 25 | 24 |
| R2 | 1,728 | 429 | 378 | 341 | 342 |
| R3 | 384 | 65 | 57 | 51 | 52 |

**R1.** Axial-Parity gave 25/384 failures. Suppressing both learned PA coefficients gave 25, keeping only their forward contribution gave 25, and suppressing learned common phase gave 26. The full model exceeded both the 23-word upper target and the stricter 20-word target. The PA intervention tied the full receiver on each pair. The same Axial-Parity trace reached 22 at update 24 and 24 at updates 32 and 64. Only the preselected update 48 counts toward the primary comparison.

**R2.** Axial-Parity reached 341/1,728. Matched Local-Parity gave 342. Without parity, Attention gave 457 and Local 459. Both pure arms regressed from Iteration 3’s 388 and 401. Attention corrected 9 of 10 initially wrong signs and introduced 0 wrong signs. Aligned midpoint phase MAE went from 0.1385 to 0.0639 rad. Suppressing the bank gave 412 failed words. Suppressing the general CFR residual gave 362. The parity advantage was six words on pair 901, a tie on 902 and a five-word loss on 903. Attention rose from 340 at update 32 to 345 at 64. Local had fewer failures at both of those checkpoints.

**R3.** Axial-Parity gave 51/384 failures against 52 for Local-Parity and the supplied 53-word post-Iteration-3 reference. Suppressing the complex spatial amplitude correction gave 47, improving the count on every pair. Suppressing fractional delays gave 64. Without parity, Attention and Local gave 56 and 53 failures. The direct joint Green correlation went from 0.35196 to 0.35311 against the 0.26870 uniform control. The spatial correlation went from 0.16446 to 0.16394. Final steering cosine was 0.97987 for Attention and 0.98005 for Local. The structured ray prior already accounts for most of the trained joint correlation. The new spatial correction showed a decoding cost and Local retained a slight lead in final steering agreement.

## Design and scope

Pairs 901, 902 and 903 each supply one independent fit per arm and regime. Within a pair all arms share training batches and one stream-6 validation draw of 32 bursts at the same SNR (12, 9 and 7 dB for R1, R2 and R3). Stream 5 supplies initialization and training. Each R3 Green probe uses eight stream-8 bursts and stream-9 column permutations. The same development draws appeared in prior iterations. Differences in pooled counts alone cannot establish generalization to unseen pairs.

The two mixers use the same 856 trainable real scalars, input fields, initialization, three tied waveform passes, centered physical readouts, Adam updates and minibatches. Attention uses global axis mixing with Hermitian affinity. Local uses three-tap axial convolutions. Width is eight. The four warmup updates reach 0.0045 and cosine decay reaches 0.00045 at update 48. Updates 49–64 continue at 0.00045 on the same optimizer state. Training uses B4, unit gradient-norm clipping, BCE on 10 tanh(LLR/10), 0.005 times raw CFR NMSE and 0.00005 times squared displacement from initialization. Every validation score and final decoder consumes raw logits. A common 48-update endpoint was chosen before the fits. The 32- and 64-update rows provide within-trajectory comparisons.

Training times are per-fit wall times with concurrent CPU workers. The separate latency process used one logical CPU, with intra-op and inter-op thread counts both set to one. It ran after the project's training, intervention and gauge workers exited. Other host activity was present during measurement. The timing tables give the measured load and the JSON retains individual wall-time and process-CPU samples.

The first pass masks the input residual to known pilots. Each PA coefficient uses a frequency-centered constellation-power contrast. The quintic contrast also removes its linear projection onto the cubic contrast. The learned time-domain PA increment is orthogonal to each symbol waveform in the complex inner product. The receive-side adjustment is bounded at 0.04 constellation units. Each regime bounds the general CFR residual at 0.06. R2 alone enables trajectory updates. Its later passes refine soft-symbol LS taps with a bounded backbone path cue, use 160 times normalized coherence in the chirp softmax and accumulate the gated phase residual. R3 retains fixed per-ray Doppler and learned fractional delays. Its additive complex antenna correction comes from the spatial branch projected onto the delay dictionary and demodulated across time. Both Q and K receive the same normalized, unscrambled physical ray coordinates plus separate learned residuals. The additive R3 correction precedes waveform prediction and CFR reconstruction.

The architectural and training changes were applied together. Handle suppressions and the cue-only replay use the selected weights, so causal attribution is limited to those interventions. Earlier 48-step spot-check results supplied in the request are historical comparators.

## Verification

P6 passed 28/28 with 0 skipped. The unchanged P5 and P5-B test bodies passed 15/15 and 13/13 through the dev fixture adapter. The default publication P4 verifier exited 0 after checking archived receipts. The adapter preserved the archived assertions and redirected their waveform draws to allowed development seeds. P6 assertions that previously required a DC response in the cubic PA readout and the external ULA projector now exercise zero-DC PA coefficients and the additive branch-derived antenna correction. New tests cover waveform orthogonality, same-coordinate ray injection and one-pass antenna reach.

All 45 actual first minibatches had 856/856 active real scalar gradients. All 45 selected-depth initialization checks matched their corresponding fixed receiver bit-for-bit. The 75 tensor-direction checks had maximum absolute error 9.712e-10.

The four gauge-enabled trained variants passed 360/360 condition checks over gains spanning 1e−35 to 1e20, with 0 changed payload coded-bit decisions out of 21,565,440. Maximum absolute ΔLLR was 6.288e-13 and maximum relative CFR L2 error was 7.047e-15. Each intact, erased or zero-noise condition uses the full 27-magnitude grid with seeded phases. NoGauge is the intentional amplitude-sensitive control. [Verification receipt](results/iter4_checks/summary.json).

Training implementation `6ab21d87ba3a03558a9e576c54877c7289865709489319b0d1c9b87d0ca45853`. The machine-readable result includes per-pair scores, interventions, checkpoints’ digests, probe observations and timing samples. [Full development results](results/dev_iter4.json). [Iteration 3](DEV_ITER3.md).

## R2 backbone cue intervention

Only the learned path cue is removed from chirp scoring. The soft taps, sharp bank, phase gate and other learned handles continue to operate.

| Receiver | Full failures | Cue off | Full sign repairs | Cue off |
| --- | ---: | ---: | ---: | ---: |
| Axial-Parity | 341 | 339 | 9 | 9 |
| Local-Parity | 342 | 344 | 9 | 9 |

Removing the cue preserved all nine sign repairs in each receiver. In Attention, adding the cue reduced midpoint phase MAE on the initially wrong rays from 0.3511 to 0.3303 radians and added two failed words. The sign repairs survived this intervention in the shared solver.

## Stream-6 receiver comparison

The pair columns count failed LDPC words. BLER pools failed and total words. Pre-BER uses hard coded bits and post-BER uses decoded information bits. NMSE divides summed channel-error energy by summed true-channel energy. BCE uses raw payload LLRs. The Genie-PA control receives the realized PA parameters.

### R1 at 12 dB

| Receiver | 901 | 902 | 903 | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ failures vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 12 | 21 | 7 | 40 / 384 | 10.42% | 9.15% | 0.88% | -10.12 | 0.22218 | — |
| Fixed + parity | 7 | 18 | 4 | 29 / 384 | 7.55% | 8.11% | 0.54% | -10.08 | 0.18750 | — |
| Axial pure | 15 | 20 | 8 | 43 / 384 | 11.20% | 9.17% | 1.08% | -10.16 | 0.22263 | +1 |
| Local pure | 13 | 20 | 9 | 42 / 384 | 10.94% | 9.13% | 0.98% | -10.11 | 0.22068 | — |
| Axial parity | 8 | 13 | 4 | 25 / 384 | 6.51% | 7.96% | 0.53% | -10.05 | 0.18525 | +1 |
| Local parity | 8 | 12 | 4 | 24 / 384 | 6.25% | 7.93% | 0.56% | -9.91 | 0.18528 | — |
| Axial no-gauge | 11 | 18 | 7 | 36 / 384 | 9.38% | 9.17% | 0.82% | -10.19 | 0.22151 | — |
| Nominal Bussgang LMMSE | 34 | 35 | 25 | 94 / 384 | 24.48% | 10.87% | 2.23% | -11.63 | 0.25730 | — |
| Nominal IDD PA | 29 | 36 | 16 | 81 / 384 | 21.09% | 10.64% | 2.24% | -11.72 | 0.27372 | — |
| Pilot-CPE-DD | 12 | 20 | 8 | 40 / 384 | 10.42% | 9.17% | 0.86% | -10.24 | 0.22261 | — |
| Genie-PA (privileged) | 10 | 20 | 8 | 38 / 384 | 9.90% | 9.10% | 0.75% | -11.53 | 0.21814 | — |

### R2 at 9 dB

| Receiver | 901 | 902 | 903 | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ failures vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 225 | 178 | 188 | 591 / 1728 | 34.20% | 12.21% | 2.19% | -14.77 | 0.27376 | — |
| Fixed + parity | 163 | 127 | 139 | 429 / 1728 | 24.83% | 11.56% | 1.45% | -18.47 | 0.25352 | — |
| Axial pure | 178 | 139 | 140 | 457 / 1728 | 26.45% | 11.63% | 1.58% | -17.42 | 0.25696 | -2 |
| Local pure | 174 | 143 | 142 | 459 / 1728 | 26.56% | 11.64% | 1.59% | -17.50 | 0.25679 | — |
| Axial parity | 131 | 104 | 106 | 341 / 1728 | 19.73% | 10.88% | 1.11% | -22.15 | 0.23918 | -1 |
| Local parity | 137 | 104 | 101 | 342 / 1728 | 19.79% | 10.87% | 1.10% | -22.22 | 0.23908 | — |
| Axial no-gauge | 171 | 135 | 139 | 445 / 1728 | 25.75% | 11.61% | 1.55% | -17.68 | 0.25592 | — |
| Tap phase slope | 576 | 576 | 575 | 1727 / 1728 | 99.94% | 42.24% | 42.76% | 2.61 | 2.63391 | — |
| 2D-LMMSE | 428 | 406 | 386 | 1220 / 1728 | 70.60% | 15.67% | 8.15% | -10.77 | 0.33416 | — |
| Waveform-DD | 171 | 129 | 146 | 446 / 1728 | 25.81% | 11.54% | 1.56% | -18.49 | 0.25433 | — |

### R3 at 7 dB

| Receiver | 901 | 902 | 903 | Failed / total | BLER | Pre-BER | Post-BER | NMSE (dB) | Raw BCE | Δ failures vs matched Local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 17 | 27 | 27 | 71 / 384 | 18.49% | 8.57% | 2.10% | -12.52 | 0.20197 | — |
| Fixed + parity | 15 | 24 | 26 | 65 / 384 | 16.93% | 7.84% | 1.85% | -12.90 | 0.18169 | — |
| Axial pure | 12 | 17 | 27 | 56 / 384 | 14.58% | 7.83% | 1.53% | -13.10 | 0.18163 | +3 |
| Local pure | 12 | 17 | 24 | 53 / 384 | 13.80% | 7.86% | 1.53% | -13.11 | 0.18183 | — |
| Axial parity | 10 | 17 | 24 | 51 / 384 | 13.28% | 7.05% | 1.47% | -13.47 | 0.16229 | -1 |
| Local parity | 10 | 18 | 24 | 52 / 384 | 13.54% | 6.99% | 1.44% | -13.50 | 0.16192 | — |
| Axial no-gauge | 14 | 17 | 27 | 58 / 384 | 15.10% | 7.97% | 1.63% | -13.03 | 0.18384 | — |
| 2D-LMMSE | 19 | 29 | 35 | 83 / 384 | 21.61% | 9.47% | 2.53% | -10.25 | 0.21338 | — |
| Waveform-DD | 15 | 23 | 27 | 65 / 384 | 16.93% | 7.97% | 2.02% | -12.89 | 0.18231 | — |

## Validation along the shared learning-rate schedule

Each cell gives pooled failures / raw validation BCE. Each checkpoint uses the same stream-6 validation bursts as its pair’s primary checkpoint. Update 48 ends the cosine decay. Updates 49–64 continue from that checkpoint at the terminal learning rate.

| Regime | Receiver | Update 0 | Update 24 | Update 32 | Update 48 | Update 64 |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial pure | 40 / 0.22218 | 51 / 0.23240 | 44 / 0.22415 | 43 / 0.22263 | 44 / 0.21978 |
| R1 | Local pure | 40 / 0.22218 | 51 / 0.23752 | 42 / 0.22221 | 42 / 0.22068 | 41 / 0.21798 |
| R1 | Axial parity | 29 / 0.18750 | 22 / 0.18346 | 24 / 0.18442 | 25 / 0.18525 | 24 / 0.18361 |
| R1 | Local parity | 29 / 0.18750 | 29 / 0.18965 | 26 / 0.18963 | 24 / 0.18528 | 26 / 0.18361 |
| R1 | Axial no-gauge | 40 / 0.22218 | 68 / 0.24744 | 55 / 0.23755 | 36 / 0.22151 | 41 / 0.21840 |
| R2 | Axial pure | 591 / 0.27376 | 471 / 0.25923 | 469 / 0.25871 | 457 / 0.25696 | 452 / 0.25672 |
| R2 | Local pure | 591 / 0.27376 | 479 / 0.25986 | 466 / 0.25798 | 459 / 0.25679 | 454 / 0.25661 |
| R2 | Axial parity | 429 / 0.25352 | 341 / 0.24071 | 340 / 0.23995 | 341 / 0.23918 | 345 / 0.23909 |
| R2 | Local parity | 429 / 0.25352 | 350 / 0.24060 | 337 / 0.24015 | 342 / 0.23908 | 339 / 0.23895 |
| R2 | Axial no-gauge | 591 / 0.27376 | 457 / 0.25790 | 451 / 0.25666 | 445 / 0.25592 | 440 / 0.25569 |
| R3 | Axial pure | 71 / 0.20197 | 52 / 0.18498 | 51 / 0.18254 | 56 / 0.18163 | 54 / 0.18151 |
| R3 | Local pure | 71 / 0.20197 | 51 / 0.18441 | 51 / 0.18251 | 53 / 0.18183 | 52 / 0.18165 |
| R3 | Axial parity | 65 / 0.18169 | 51 / 0.16527 | 51 / 0.16327 | 51 / 0.16229 | 50 / 0.16209 |
| R3 | Local parity | 65 / 0.18169 | 51 / 0.16495 | 50 / 0.16286 | 52 / 0.16192 | 50 / 0.16155 |
| R3 | Axial no-gauge | 71 / 0.20197 | 53 / 0.18695 | 58 / 0.18487 | 58 / 0.18384 | 56 / 0.18346 |

### Training time by pair

Cumulative wall seconds through update 48 include training-frame generation and optimization. The timer pauses during validation, probing and checkpoint writes.

| Regime | Receiver | 901 (s) | 902 (s) | 903 (s) |
| --- | --- | --- | --- | --- |
| R1 | Axial pure | 4.10 | 4.69 | 5.03 |
| R1 | Local pure | 3.46 | 3.90 | 4.67 |
| R1 | Axial parity | 4.56 | 5.01 | 5.32 |
| R1 | Local parity | 3.66 | 4.45 | 5.14 |
| R1 | Axial no-gauge | 4.32 | 4.46 | 5.44 |
| R2 | Axial pure | 10.82 | 11.74 | 14.25 |
| R2 | Local pure | 6.87 | 9.05 | 9.88 |
| R2 | Axial parity | 11.81 | 14.25 | 13.90 |
| R2 | Local parity | 7.96 | 9.50 | 11.71 |
| R2 | Axial no-gauge | 9.60 | 13.66 | 20.17 |
| R3 | Axial pure | 25.91 | 43.84 | 50.77 |
| R3 | Local pure | 18.11 | 31.21 | 39.69 |
| R3 | Axial parity | 31.23 | 48.76 | 43.97 |
| R3 | Local parity | 22.96 | 35.47 | 32.10 |
| R3 | Axial no-gauge | 32.84 | 46.45 | 29.63 |

## Handle interventions on the selected checkpoints

Each pair cell gives its failure count and the change from its full receiver. A positive change means suppression increased failures. All rows reuse the same weights and frames. The PA-off intervention retains the nominal PA. The forward-only row disables the bounded receive-side PA adjustment.

### R1

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 8 (0) | 13 (0) | 4 (0) | 25 | 0 | +0.00000 |
| Axial parity | PA residual off | 8 (0) | 13 (0) | 4 (0) | 25 | 0 | +0.00003 |
| Axial parity | Forward PA residual only | 8 (0) | 13 (0) | 4 (0) | 25 | 0 | +0.00003 |
| Axial parity | Learned common phase off | 8 (0) | 14 (+1) | 4 (0) | 26 | +1 | -0.00164 |
| Axial parity | PA residual and common phase off | 8 (0) | 14 (+1) | 4 (0) | 26 | +1 | -0.00163 |
| Axial parity | General CFR residual off | 9 (+1) | 14 (+1) | 4 (0) | 27 | +2 | +0.00467 |
| Axial parity | All handles off | 7 (-1) | 18 (+5) | 4 (0) | 29 | +4 | +0.00225 |
| Local parity | Full receiver | 8 (0) | 12 (0) | 4 (0) | 24 | 0 | +0.00000 |
| Local parity | PA residual off | 8 (0) | 12 (0) | 4 (0) | 24 | 0 | +0.00004 |
| Local parity | Forward PA residual only | 8 (0) | 12 (0) | 4 (0) | 24 | 0 | +0.00002 |
| Local parity | Learned common phase off | 8 (0) | 14 (+2) | 4 (0) | 26 | +2 | -0.00173 |
| Local parity | PA residual and common phase off | 8 (0) | 14 (+2) | 4 (0) | 26 | +2 | -0.00173 |
| Local parity | General CFR residual off | 9 (+1) | 14 (+2) | 4 (0) | 27 | +3 | +0.00436 |
| Local parity | All handles off | 7 (-1) | 18 (+6) | 4 (0) | 29 | +5 | +0.00222 |

### R2

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 131 (0) | 104 (0) | 106 (0) | 341 | 0 | +0.00000 |
| Axial parity | Learned common phase off | 131 (0) | 104 (0) | 106 (0) | 341 | 0 | +0.00000 |
| Axial parity | Per-ray trajectory off | 154 (+23) | 120 (+16) | 129 (+23) | 403 | +62 | +0.01008 |
| Axial parity | Soft chirp-bank residual off | 159 (+28) | 121 (+17) | 132 (+26) | 412 | +71 | +0.00585 |
| Axial parity | Omit third EM pass | 147 (+16) | 105 (+1) | 107 (+1) | 359 | +18 | +0.00210 |
| Axial parity | General CFR residual off | 141 (+10) | 111 (+7) | 110 (+4) | 362 | +21 | +0.00383 |
| Axial parity | All handles off | 163 (+32) | 127 (+23) | 139 (+33) | 429 | +88 | +0.01435 |
| Local parity | Full receiver | 137 (0) | 104 (0) | 101 (0) | 342 | 0 | +0.00000 |
| Local parity | Learned common phase off | 137 (0) | 104 (0) | 101 (0) | 342 | 0 | +0.00000 |
| Local parity | Per-ray trajectory off | 155 (+18) | 121 (+17) | 129 (+28) | 405 | +63 | +0.01008 |
| Local parity | Soft chirp-bank residual off | 162 (+25) | 124 (+20) | 126 (+25) | 412 | +70 | +0.00555 |
| Local parity | Omit third EM pass | 145 (+8) | 108 (+4) | 105 (+4) | 358 | +16 | +0.00219 |
| Local parity | General CFR residual off | 144 (+7) | 113 (+9) | 105 (+4) | 362 | +20 | +0.00392 |
| Local parity | All handles off | 163 (+26) | 127 (+23) | 139 (+38) | 429 | +87 | +0.01444 |

### R3

| Receiver | Intervention | 901 | 902 | 903 | Pooled failures | Δ failures | Δ raw BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | Full receiver | 10 (0) | 17 (0) | 24 (0) | 51 | 0 | +0.00000 |
| Axial parity | PA residual off | 10 (0) | 17 (0) | 24 (0) | 51 | 0 | +0.00003 |
| Axial parity | Learned common phase off | 10 (0) | 17 (0) | 24 (0) | 51 | 0 | +0.00000 |
| Axial parity | Per-ray trajectory off | 10 (0) | 17 (0) | 24 (0) | 51 | 0 | +0.00000 |
| Axial parity | Additive antenna/path amplitude correction off | 8 (-2) | 16 (-1) | 23 (-1) | 47 | -4 | +0.00119 |
| Axial parity | Entire antenna mixing branch off | 9 (-1) | 16 (-1) | 23 (-1) | 48 | -3 | +0.00169 |
| Axial parity | Fractional delays off | 16 (+6) | 23 (+6) | 25 (+1) | 64 | +13 | +0.01243 |
| Axial parity | General CFR residual off | 9 (-1) | 18 (+1) | 24 (0) | 51 | 0 | +0.00463 |
| Axial parity | All handles off | 15 (+5) | 24 (+7) | 26 (+2) | 65 | +14 | +0.01940 |
| Local parity | Full receiver | 10 (0) | 18 (0) | 24 (0) | 52 | 0 | +0.00000 |
| Local parity | PA residual off | 10 (0) | 18 (0) | 24 (0) | 52 | 0 | +0.00004 |
| Local parity | Learned common phase off | 10 (0) | 18 (0) | 24 (0) | 52 | 0 | +0.00000 |
| Local parity | Per-ray trajectory off | 10 (0) | 18 (0) | 24 (0) | 52 | 0 | +0.00000 |
| Local parity | Additive antenna/path amplitude correction off | 9 (-1) | 16 (-2) | 25 (+1) | 50 | -2 | +0.00090 |
| Local parity | Entire antenna mixing branch off | 9 (-1) | 16 (-2) | 23 (-1) | 48 | -4 | +0.00129 |
| Local parity | Fractional delays off | 16 (+6) | 23 (+5) | 24 (0) | 63 | +11 | +0.01254 |
| Local parity | General CFR residual off | 8 (-2) | 18 (0) | 24 (0) | 50 | -2 | +0.00518 |
| Local parity | All handles off | 15 (+5) | 24 (+6) | 26 (+2) | 65 | +13 | +0.01977 |

## R2 chirp direction and phase error

Direction uses the sign of the net slope between first and last symbol midpoints. Repairs count wrong-to-correct signs and regressions count correct-to-wrong signs. Phase MAE removes one constant angle for each burst and ray. Simulator truth enters these comparisons after inference.

| Receiver | Pair | Burst-rays | Initial wrong | Final wrong | Repairs | Regressions | Initial midpoint MAE (rad) | Final midpoint MAE (rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 96 | 4 | 1 | 3 | 0 | 0.15710 | 0.07258 |
| Axial parity | 902 | 96 | 4 | 0 | 4 | 0 | 0.13910 | 0.05928 |
| Axial parity | 903 | 96 | 2 | 0 | 2 | 0 | 0.11921 | 0.05984 |
| Local parity | 901 | 96 | 4 | 1 | 3 | 0 | 0.15710 | 0.07373 |
| Local parity | 902 | 96 | 4 | 0 | 4 | 0 | 0.13910 | 0.06169 |
| Local parity | 903 | 96 | 2 | 0 | 2 | 0 | 0.11921 | 0.05907 |

Subsets overlap. Net slope is in radians per useful OFDM-symbol duration.

| Receiver | Subset | Burst-rays | Wrong signs | Zero slopes | Repairs | Regressions | Net slope MAE | Midpoint phase MAE (rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial parity | All rays | 288 | 10 → 1 | 0 → 0 | 9 | 0 | 0.03901 → 0.00797 | 0.13847 → 0.06390 |
| Axial parity | Initially wrong sign | 10 | 10 → 1 | 0 → 0 | 9 | 0 | 0.86071 → 0.10618 | 1.50385 → 0.33032 |
| Axial parity | Initially correct sign | 278 | 0 → 0 | 0 → 0 | 0 | 0 | 0.00945 → 0.00443 | 0.08936 → 0.05432 |
| Axial parity | Initial span (0.5, 3] rad | 50 | 0 → 0 | 0 → 0 | 0 | 0 | 0.02036 → 0.00832 | 0.18924 → 0.08598 |
| Axial parity | Initial span >3 rad | 12 | 10 → 1 | 0 → 0 | 9 | 0 | 0.74463 → 0.09708 | 1.45922 → 0.33898 |
| Axial parity | Initially wrong, span (0.5, 3] rad | 0 | 0 → 0 | 0 → 0 | 0 | 0 | — | — |
| Local parity | All rays | 288 | 10 → 1 | 0 → 0 | 9 | 0 | 0.03901 → 0.00806 | 0.13847 → 0.06483 |
| Local parity | Initially wrong sign | 10 | 10 → 1 | 0 → 0 | 9 | 0 | 0.86071 → 0.10833 | 1.50385 → 0.34486 |
| Local parity | Initially correct sign | 278 | 0 → 0 | 0 → 0 | 0 | 0 | 0.00945 → 0.00445 | 0.08936 → 0.05476 |
| Local parity | Initial span (0.5, 3] rad | 50 | 0 → 0 | 0 → 0 | 0 | 0 | 0.02036 → 0.00799 | 0.18924 → 0.08623 |
| Local parity | Initial span >3 rad | 12 | 10 → 1 | 0 → 0 | 9 | 0 | 0.74463 → 0.09880 | 1.45922 → 0.35115 |
| Local parity | Initially wrong, span (0.5, 3] rad | 0 | 0 → 0 | 0 → 0 | 0 | 0 | — | — |

The per-pass comparison averages the sample-time phasor within each row before measuring wrapped phase error in radians.

| Receiver | Pair | Burst-rays | Initial | Pass 1 | Pass 2 | Pass 3 |
| --- | --- | --- | --- | --- | --- | --- |
| Axial parity | 901 | 96 | 0.15711 | 0.15660 | 0.10006 | 0.07258 |
| Axial parity | 902 | 96 | 0.13910 | 0.13870 | 0.10090 | 0.05927 |
| Axial parity | 903 | 96 | 0.11923 | 0.11763 | 0.07446 | 0.05984 |
| Local parity | 901 | 96 | 0.15711 | 0.15615 | 0.10064 | 0.07373 |
| Local parity | 902 | 96 | 0.13910 | 0.13927 | 0.10191 | 0.06168 |
| Local parity | 903 | 96 | 0.11923 | 0.11669 | 0.07414 | 0.05905 |

## R3 Green-kernel correlations

Each overall Pearson coefficient uses 2256 off-diagonal entries. The 24 probe bursts come from stream 8. The shuffled control averages eight stream-9 key permutations. The direct joint averages the three axis transports at their physical supports. The historical product composes those transports. Axis-only correlations restrict both tokens to the same line along the measured axis.

| Receiver | Kernel | Initial | Trained | Trained shuffled | Direct uniform | Initial axis-only | Trained axis-only | Trained 901 | Trained 902 | Trained 903 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Axial pure | Historical product | 0.50873 | 0.51010 | 0.00039 | — | — | — | 0.45323 | 0.51790 | 0.55918 |
| Axial pure | Time | -0.02438 | -0.02400 | -0.00206 | — | 0.75689 | 0.75508 | -0.02441 | -0.02352 | -0.02408 |
| Axial pure | Frequency | 0.34373 | 0.34384 | 0.00323 | — | 0.77325 | 0.79838 | 0.27819 | 0.39360 | 0.35971 |
| Axial pure | Direct parallel joint | 0.35194 | 0.35294 | 0.00182 | 0.26870 | — | — | 0.33130 | 0.36529 | 0.36223 |
| Axial pure | Spatial | 0.16428 | 0.16410 | -0.00159 | — | 0.85811 | 0.86041 | 0.18461 | 0.14702 | 0.16066 |
| Axial parity | Historical product | 0.50959 | 0.51211 | 0.00060 | — | — | — | 0.45448 | 0.52209 | 0.55975 |
| Axial parity | Time | -0.02474 | -0.02533 | -0.00240 | — | 0.75847 | 0.76218 | -0.02509 | -0.02597 | -0.02493 |
| Axial parity | Frequency | 0.34444 | 0.34527 | 0.00346 | — | 0.77907 | 0.80707 | 0.27769 | 0.39835 | 0.35977 |
| Axial parity | Direct parallel joint | 0.35196 | 0.35311 | 0.00181 | 0.26870 | — | — | 0.33144 | 0.36556 | 0.36234 |
| Axial parity | Spatial | 0.16446 | 0.16394 | -0.00161 | — | 0.85906 | 0.86043 | 0.18435 | 0.14674 | 0.16072 |

The historical uniform product is constant, so its Pearson correlation is undefined. The matched direct-joint uniform control has the following correlations.

| 901 | 902 | 903 | Mean |
| --- | --- | --- | --- |
| 0.24605 | 0.28659 | 0.27347 | 0.26870 |

### R3 spatial steering on validation frames

At each fit, steering cosine is the absolute normalized inner product of the fitted antenna vector and its true ULA direction. Remote phase cosine measures relative-phase agreement for the three non-adjacent antenna pairs. The off-mode ratio divides the RMS amplitude outside the true steering mode after the correction by its value before the correction. The pooled waveform row includes each pass once and excludes the pilot redemap.

| Receiver | Stage | Burst-ray applications | Steering before | Steering after | Remote phase before | Remote phase after | Off-mode RMS ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed + parity | pilot redemap | 384 | 0.93473 | 0.93473 | 0.87365 | 0.87365 | 1.00000 |
| Fixed + parity | waveform solve 1 | 384 | 0.97293 | 0.97293 | 0.95080 | 0.95080 | 1.00000 |
| Fixed + parity | waveform solve 2 | 384 | 0.97808 | 0.97808 | 0.96262 | 0.96262 | 1.00000 |
| Fixed + parity | waveform solve 3 | 384 | 0.97873 | 0.97873 | 0.96446 | 0.96446 | 1.00000 |
| Fixed + parity | all waveform solves | 1152 | 0.97658 | 0.97658 | 0.95929 | 0.95929 | 1.00000 |
| Axial parity | pilot redemap | 384 | 0.93473 | 0.93421 | 0.87365 | 0.87176 | 0.98237 |
| Axial parity | waveform solve 1 | 384 | 0.97398 | 0.97387 | 0.95167 | 0.95135 | 0.98268 |
| Axial parity | waveform solve 2 | 384 | 0.97992 | 0.97966 | 0.96574 | 0.96510 | 0.98600 |
| Axial parity | waveform solve 3 | 384 | 0.98030 | 0.97987 | 0.96647 | 0.96563 | 0.98771 |
| Axial parity | all waveform solves | 1152 | 0.97807 | 0.97780 | 0.96129 | 0.96069 | 0.98522 |
| Local parity | pilot redemap | 384 | 0.93473 | 0.93476 | 0.87365 | 0.87316 | 0.97986 |
| Local parity | waveform solve 1 | 384 | 0.97409 | 0.97410 | 0.95181 | 0.95173 | 0.98064 |
| Local parity | waveform solve 2 | 384 | 0.97996 | 0.97983 | 0.96569 | 0.96554 | 0.98308 |
| Local parity | waveform solve 3 | 384 | 0.98028 | 0.98005 | 0.96638 | 0.96617 | 0.98452 |
| Local parity | all waveform solves | 1152 | 0.97811 | 0.97799 | 0.96129 | 0.96115 | 0.98256 |

| Receiver | Stage | Antennas 0, 2 | Antennas 0, 3 | Antennas 1, 3 |
| --- | --- | --- | --- | --- |
| Fixed + parity | waveform solve 3 | 0.96786 → 0.96786 | 0.96319 → 0.96319 | 0.96233 → 0.96233 |
| Fixed + parity | all waveform solves | 0.96278 → 0.96278 | 0.95921 → 0.95921 | 0.95589 → 0.95589 |
| Axial parity | waveform solve 3 | 0.96750 → 0.96695 | 0.96611 → 0.96540 | 0.96581 → 0.96455 |
| Axial parity | all waveform solves | 0.96396 → 0.96330 | 0.96154 → 0.96109 | 0.95838 → 0.95769 |
| Local parity | waveform solve 3 | 0.96726 → 0.96695 | 0.96658 → 0.96653 | 0.96531 → 0.96502 |
| Local parity | all waveform solves | 0.96373 → 0.96327 | 0.96193 → 0.96205 | 0.95822 → 0.95813 |

## Single-thread receiver timing

Cells are median wall milliseconds per forward on the timing pair’s stream-6 inputs. The harness shuffles method order within each round and excludes its warmup. Simulation, final LDPC decoding and scoring occur outside the timed forward.

| Receiver | R1 B1 | R1 B64 | R2 B1 | R2 B64 | R3 B1 | R3 B64 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed | 5.32 | 31.95 | 5.46 | 117.98 | 4.00 | 56.96 |
| Fixed + parity | 4.88 | 45.18 | 6.35 | 155.87 | 6.27 | 73.80 |
| Axial pure | 15.58 | 253.36 | 23.28 | 1165.60 | 30.25 | 1134.73 |
| Local pure | 12.63 | 213.96 | 17.72 | 655.77 | 22.72 | 709.28 |
| Axial parity | 16.44 | 270.47 | 24.37 | 1107.35 | 25.37 | 1111.71 |
| Local parity | 14.33 | 187.00 | 18.51 | 767.82 | 21.19 | 692.53 |
| Axial no-gauge | 14.45 | 235.63 | 20.56 | 986.46 | 23.31 | 1057.59 |
| Nominal IDD PA | 3.65 | 44.06 | — | — | — | — |
| Pilot-CPE-DD | 4.43 | 44.64 | — | — | — | — |
| Genie-PA (privileged) | 4.56 | 57.41 | — | — | — | — |
| Nominal Bussgang LMMSE | 1.21 | 22.50 | — | — | — | — |
| 2D-LMMSE | — | — | 2.74 | 134.42 | 1.31 | 21.16 |
| Tap phase slope | — | — | 3.00 | 39.55 | — | — |
| Waveform-DD | — | — | 10.74 | 321.44 | 6.94 | 83.38 |

Host busy is the percentage of total host CPU capacity in use across all processes. Background subtracts benchmark-process CPU from host busy. Each range covers the timed rounds for the specified batch. The load was observed during measurement.

| Regime | Batch | Pair | Timed rounds | Torch threads | Interop threads | Logical affinity | Host busy range | Background range | Background median | Quiet wait (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 1 | 901 | 5 | 1 | 1 | [1] | 12.3–92.2% | 3.1–86.3% | 10.6% | 0 |
| R1 | 64 | 901 | 5 | 1 | 1 | [1] | 13.6–29.7% | 5.7–22.2% | 15.2% | 0 |
| R2 | 1 | 901 | 5 | 1 | 1 | [1] | 17.4–54.3% | 8.1–48.1% | 17.9% | 0 |
| R2 | 64 | 901 | 5 | 1 | 1 | [1] | 28.2–40.5% | 20.7–33.0% | 28.3% | 0 |
| R3 | 1 | 901 | 5 | 1 | 1 | [1] | 21.0–38.2% | 12.0–31.6% | 22.2% | 0 |
| R3 | 64 | 901 | 5 | 1 | 1 | [1] | 29.8–42.2% | 22.9–35.5% | 28.9% | 0 |
