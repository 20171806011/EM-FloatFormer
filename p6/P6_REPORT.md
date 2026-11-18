# Phase 6 held-out evaluation

On the ten prospective pairs per regime, Axial-Parity recorded 171/3,200 R1, 2,805/14,400 R2 and 444/3,200 R3 failed LDPC words. 

Positive Holm-significant gains at familywise α = 0.05 occurred in 3/3 equal-information classical comparisons and 2/3 fixed-physics-plus-parity comparisons. The tables retain ties, unfavorable effects and their two-sided tests.

**One formal numerical requirement failed.** On R2 pair 11 with degenerate pilots, Axial-Pure reached a maximum absolute LLR difference of 1.0160761121369433 × 10⁻¹² under complex rescaling. The limit was strictly below 10⁻¹². Axial-Pure passed 299/300 conditions and Axial-Parity passed 300/300 with a maximum of 4.36984 × 10⁻¹³. All 35,942,400 coded-bit comparisons preserved the decision. Both development audits had passed 360/360 before the freeze. The complete formal audit retains this failure. [Individual failing case](results/formal/A/r2/pair11/audit_gauge.json) · [Formal audit](results/trained_gauge.json)

## Design and attribution

Each of the 50 regime/pair combinations received five receiver fits. Each fit's held-out measurement used ten independently seeded Stream 7 draws of eight bursts. A pair is the unit of inference. BLER and BER pool integer counts, while CFR NMSE divides pooled error energy by pooled true-channel energy before conversion to dB. The final decoder is shared by all arms. Genie-PA receives privileged PA parameters.

The development selection scored 25, 337 and 47 failed words. It corrected 9/10 initially wrong R2 signs. The formal model and protocol were frozen before scoring. [Mathematical derivations and ablations](DEV_ITER5_MATH.md) · [Frozen protocol](PROTOCOL.md) · [SHA-256 inventory](results/freeze.json)

A positive table entry below is a pooled reduction in failed words for the second receiver relative to the first. The fixed-receiver parity comparison estimates the iterative parity contribution on shared waveform physics. Adding learned handles on top of Fixed+Parity measures their aggregate contribution, while matched learned arms isolate the change of mixer.

| Comparison | R1 | R2 | R3 |
| --- | --- | --- | --- |
| Fixed → Fixed+Parity | +148 | +1131 | +60 |
| Fixed+Parity → Axial-Parity | +0 | +718 | +83 |
| Local-Parity → Axial-Parity | +5 | -13 | -2 |
| Axial-Pure → Axial-Parity | +134 | +941 | +35 |

In **R1**, iterative parity in the fixed receiver reduced failures from 319 to 171. Adding learned handles left the pooled count at 171. Axial-Parity scored 171 against 315 for Pilot-CPE-DD, 284 for privileged Genie-PA and 724 for nominal IDD-PA. On the legacy R1 pairs, Fixed+Parity achieved 165 and Axial-Parity 188.

In **R2**, adding learned handles to Fixed+Parity saved 718 words. With the same Axial-Parity weights, suppressing the chirp bank alone raised failures from 2,805 to 3,471. Suppressing the phase handle raised them to 3,362. These interventions share a phase-estimation pathway, so adding their effects would double-count part of the gain. On first-pilot-row-erased development frames the information-damped update cost decoded words despite its lower numerical sensitivity. [R2 conditioning and erasure study](results/dev_math/chirp/R2_MATH.md)

In **R3**, adding learned handles to Fixed+Parity saved 83 words. Turning off spatial shrinkage cost 15 words and turning off delay corrections cost 67, for **each** mixer at its own frozen weights. Axial-Parity's direct Green correlation was 0.384273 at initialization and 0.384895 after training. The prospective pair-matched uniform control averaged 0.320608. Most of the measured joint alignment was already present at initialization. The three matched Axial-Parity versus Local-Parity BLER tests all had Holm p = 1. With ten pairs per regime these comparisons establish neither a lower BLER for Attention nor equivalence between mixers. [Interventions](results/interventions.json) · [Paired inference](results/summary.json)

## Prospective inference

For block errors, differences are comparator minus named arm. For Green, differences are named correlation minus comparator. Each exact two-sided sign test discards tied pairs and counts wins and losses with equal probability under the null. Holm adjustment operates separately within the four registered families. With 10 non-tied pairs the smallest raw p is 1/512. Exact fractions accompany decimals; all pair differences are in the machine-readable summary.
### Classical comparators

| Regime | Arm or correlation | Comparator | W/T/L | Mean paired difference | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial-Parity | Pilot-CPE-DD | 10/0/0 | +14.4 | 0.001953125 (1/512) | 0.005859375 (3/512) |
| R2 | Axial-Parity | Waveform-DD | 10/0/0 | +74.5 | 0.001953125 (1/512) | 0.005859375 (3/512) |
| R3 | Axial-Parity | Waveform-DD | 10/0/0 | +9.3 | 0.001953125 (1/512) | 0.005859375 (3/512) |

### Fixed+Parity

| Regime | Arm or correlation | Comparator | W/T/L | Mean paired difference | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial-Parity | Fixed+Parity | 5/2/3 | +0 | 0.7265625 (93/128) | 0.7265625 (93/128) |
| R2 | Axial-Parity | Fixed+Parity | 10/0/0 | +71.8 | 0.001953125 (1/512) | 0.005859375 (3/512) |
| R3 | Axial-Parity | Fixed+Parity | 10/0/0 | +8.3 | 0.001953125 (1/512) | 0.005859375 (3/512) |

### Structural comparisons

| Regime | Arm or correlation | Comparator | W/T/L | Mean paired difference | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Axial-Parity | Local-Parity | 6/1/3 | +0.5 | 0.5078125 (65/128) | 1 |
| R2 | Axial-Parity | Local-Parity | 4/1/5 | -1.3 | 1 | 1 |
| R3 | Axial-Parity | Local-Parity | 3/3/4 | -0.2 | 1 | 1 |
| R1 | Axial-Pure | Local-Pure | 2/2/6 | -1.1 | 0.2890625 (37/128) | 1 |
| R2 | Axial-Pure | Local-Pure | 4/1/5 | -2.5 | 1 | 1 |
| R3 | Axial-Pure | Local-Pure | 5/3/2 | +0.7 | 0.453125 (29/64) | 1 |
| R1 | Axial-Parity | Axial-Pure | 10/0/0 | +13.4 | 0.001953125 (1/512) | 0.0234375 (3/128) |
| R2 | Axial-Parity | Axial-Pure | 10/0/0 | +94.1 | 0.001953125 (1/512) | 0.0234375 (3/128) |
| R3 | Axial-Parity | Axial-Pure | 9/0/1 | +3.5 | 0.021484375 (11/512) | 0.21484375 (55/256) |
| R1 | Axial-Pure | Axial-NoGauge | 4/1/5 | -0.7 | 1 | 1 |
| R2 | Axial-Pure | Axial-NoGauge | 1/0/9 | -8 | 0.021484375 (11/512) | 0.21484375 (55/256) |
| R3 | Axial-Pure | Axial-NoGauge | 5/3/2 | +0.8 | 0.453125 (29/64) | 1 |

### R3 Green correlations

| Regime | Arm or correlation | Comparator | W/T/L | Mean paired difference | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| R3 | Direct joint | Shuffled keys | 10/0/0 | +0.386283 | 0.001953125 (1/512) | 0.0078125 (1/128) |
| R3 | Direct joint | Pair-specific uniform support | 10/0/0 | +0.0642871 | 0.001953125 (1/512) | 0.0078125 (1/128) |
| R3 | Historical product, trained | Historical product, initial | 10/0/0 | +0.00508677 | 0.001953125 (1/512) | 0.0078125 (1/128) |
| R3 | Spatial axis-only | Historical P5 reference 0.084 | 10/0/0 | +0.808253 | 0.001953125 (1/512) | 0.0078125 (1/128) |

## Mathematical model

The fixed training endpoint was 48 warm-cosine Adam updates for both mixers. The frozen configuration used R2 chirp mode `fisher_trust`, R1 bridge `False`, R1 differential PA `False`, R3 IRC `False`, R3 ULA-frame contraction `True`, and objective `bce`. The learned operator width was eight. The three waveform passes share parameters, and the centered map gives exactly zero physical increments at initialization.

For a bank with template phases θ and weights w = softmax(βc), the gradient of a predicted phase with respect to score cᵢ is βwᵢ(θᵢ − Σwⱼθⱼ). Large differences between unwrapped chirps can amplify a small coherence perturbation in a late waveform pass. Development compared final-pass bank conditioning and a continuous chirp correction while checking sign repair and decoded blocks. The R1 comparisons tested a sample-clock phase bridge and differential PA calibration. The R3 comparisons tested a pilot-innovation precision matrix and a ULA-frame contraction. For the latter, adjacent-sensor lag estimates a steering phasor; conjugating the branch contraction by the resulting diagonal phase matrix preserves that estimated steering mode. The objective comparison used the same candidate loss and optimizer for both mixers. [Derivations, ablations and invariant checks](DEV_ITER5_MATH.md)
## Prospective cohort A

### R1 at 12 dB

The per-pair column follows pair order 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 38, 33, 21, 29, 35, 30, 32, 24, 31, 32 | 305 / 3,200 | 9.531% | 8.924% | 0.657% | -10.299 |
| Local-Pure | 39, 32, 21, 28, 33, 30, 26, 23, 29, 33 | 294 / 3,200 | 9.188% | 8.924% | 0.633% | -10.241 |
| Axial-Parity | 22, 23, 20, 11, 14, 20, 17, 14, 16, 14 | 171 / 3,200 | 5.344% | 7.852% | 0.301% | -10.341 |
| Local-Parity | 23, 26, 20, 10, 12, 21, 18, 15, 15, 16 | 176 / 3,200 | 5.500% | 7.856% | 0.298% | -10.317 |
| Axial-NoGauge | 40, 34, 20, 29, 34, 27, 25, 30, 26, 33 | 298 / 3,200 | 9.312% | 8.955% | 0.649% | -10.310 |
| Fixed | 46, 31, 26, 28, 30, 37, 25, 35, 33, 28 | 319 / 3,200 | 9.969% | 8.996% | 0.685% | -10.136 |
| Fixed+Parity | 23, 24, 16, 13, 14, 23, 17, 12, 18, 11 | 171 / 3,200 | 5.344% | 7.954% | 0.296% | -10.270 |
| Nominal Bussgang LMMSE | 95, 81, 81, 62, 70, 81, 73, 78, 79, 61 | 761 / 3,200 | 23.781% | 10.721% | 2.186% | -11.511 |
| Nominal IDD-PA | 93, 76, 76, 60, 73, 73, 72, 76, 72, 53 | 724 / 3,200 | 22.625% | 10.386% | 2.117% | -11.505 |
| Pilot-CPE-DD | 46, 31, 25, 28, 29, 34, 24, 37, 35, 26 | 315 / 3,200 | 9.844% | 8.990% | 0.669% | -10.244 |
| Genie-PA (privileged) | 44, 34, 19, 22, 27, 31, 25, 30, 27, 25 | 284 / 3,200 | 8.875% | 8.874% | 0.612% | -12.031 |

### R2 at 9 dB

The per-pair column follows pair order 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 394, 387, 373, 424, 340, 387, 355, 371, 378, 337 | 3,746 / 14,400 | 26.014% | 11.760% | 1.627% | -17.030 |
| Local-Pure | 395, 392, 373, 416, 338, 376, 356, 374, 368, 333 | 3,721 / 14,400 | 25.840% | 11.751% | 1.621% | -17.064 |
| Axial-Parity | 280, 301, 291, 303, 247, 288, 274, 280, 278, 263 | 2,805 / 14,400 | 19.479% | 10.928% | 1.090% | -21.854 |
| Local-Parity | 278, 282, 288, 302, 254, 292, 277, 276, 280, 263 | 2,792 / 14,400 | 19.389% | 10.920% | 1.070% | -21.973 |
| Axial-NoGauge | 388, 391, 357, 408, 329, 383, 353, 362, 364, 331 | 3,666 / 14,400 | 25.458% | 11.711% | 1.584% | -17.258 |
| Fixed | 502, 456, 477, 499, 450, 444, 463, 480, 449, 434 | 4,654 / 14,400 | 32.319% | 12.256% | 2.194% | -14.610 |
| Fixed+Parity | 380, 345, 347, 388, 341, 355, 351, 368, 326, 322 | 3,523 / 14,400 | 24.465% | 11.596% | 1.513% | -18.100 |
| Tap phase slope | 1439, 1439, 1440, 1440, 1439, 1440, 1438, 1438, 1440, 1439 | 14,392 / 14,400 | 99.944% | 42.315% | 42.686% | 2.601 |
| 2D-LMMSE | 1005, 1008, 1024, 1043, 1035, 1023, 1015, 998, 1021, 989 | 10,161 / 14,400 | 70.562% | 15.830% | 8.343% | -10.551 |
| Waveform-DD | 381, 355, 341, 387, 324, 358, 363, 379, 341, 321 | 3,550 / 14,400 | 24.653% | 11.555% | 1.564% | -18.138 |

### R3 at 7 dB

The per-pair column follows pair order 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 42, 49, 24, 46, 37, 55, 62, 53, 59, 52 | 479 / 3,200 | 14.969% | 7.552% | 1.748% | -13.517 |
| Local-Pure | 42, 51, 24, 47, 38, 56, 62, 57, 58, 51 | 486 / 3,200 | 15.188% | 7.545% | 1.751% | -13.518 |
| Axial-Parity | 36, 51, 23, 43, 36, 43, 58, 51, 54, 49 | 444 / 3,200 | 13.875% | 6.772% | 1.631% | -13.752 |
| Local-Parity | 36, 49, 23, 45, 35, 46, 58, 52, 51, 47 | 442 / 3,200 | 13.812% | 6.744% | 1.656% | -13.741 |
| Axial-NoGauge | 42, 51, 24, 48, 39, 54, 62, 57, 60, 50 | 487 / 3,200 | 15.219% | 7.595% | 1.783% | -13.426 |
| Fixed | 48, 59, 32, 62, 48, 68, 67, 68, 74, 61 | 587 / 3,200 | 18.344% | 8.294% | 2.112% | -12.384 |
| Fixed+Parity | 44, 56, 29, 53, 46, 56, 61, 59, 68, 55 | 527 / 3,200 | 16.469% | 7.647% | 1.907% | -12.718 |
| 2D-LMMSE | 56, 71, 42, 74, 55, 71, 79, 71, 81, 74 | 674 / 3,200 | 21.062% | 9.176% | 2.554% | -10.459 |
| Waveform-DD | 46, 55, 29, 57, 42, 63, 63, 60, 69, 53 | 537 / 3,200 | 16.781% | 7.681% | 2.004% | -12.710 |

## Legacy cohort B

### R1 at 12 dB

The per-pair column follows pair order 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 32, 26, 28, 23, 28, 29, 31, 22, 37, 28 | 284 / 3,200 | 8.875% | 8.851% | 0.567% | -10.629 |
| Local-Pure | 34, 26, 29, 25, 25, 29, 36, 26, 34, 28 | 292 / 3,200 | 9.125% | 8.897% | 0.610% | -10.538 |
| Axial-Parity | 24, 18, 24, 18, 12, 16, 21, 15, 24, 16 | 188 / 3,200 | 5.875% | 7.743% | 0.309% | -10.627 |
| Local-Parity | 25, 18, 20, 18, 11, 19, 20, 15, 23, 12 | 181 / 3,200 | 5.656% | 7.749% | 0.303% | -10.576 |
| Axial-NoGauge | 31, 26, 30, 21, 27, 29, 27, 23, 35, 24 | 273 / 3,200 | 8.531% | 8.837% | 0.530% | -10.616 |
| Fixed | 35, 32, 31, 27, 23, 30, 27, 24, 37, 31 | 297 / 3,200 | 9.281% | 8.909% | 0.604% | -10.442 |
| Fixed+Parity | 17, 17, 20, 14, 13, 15, 20, 12, 24, 13 | 165 / 3,200 | 5.156% | 7.840% | 0.275% | -10.560 |
| P5-B pure | 33, 35, 25, 24, 23, 25, 24, 29, 37, 29 | 284 / 3,200 | 8.875% | 8.815% | 0.557% | -10.544 |
| P5-B parity | 17, 17, 23, 18, 15, 19, 19, 17, 26, 21 | 192 / 3,200 | 6.000% | 8.052% | 0.328% | -10.653 |
| P5-B local | 35, 26, 27, 25, 22, 26, 24, 22, 35, 29 | 271 / 3,200 | 8.469% | 8.774% | 0.544% | -10.662 |
| P5-B local parity | 19, 16, 24, 18, 15, 19, 20, 17, 28, 21 | 197 / 3,200 | 6.156% | 8.047% | 0.335% | -10.679 |
| P5-B no gauge | 33, 25, 31, 25, 21, 26, 24, 23, 38, 28 | 274 / 3,200 | 8.562% | 8.779% | 0.533% | -10.548 |
| P5-B physics | 36, 27, 30, 27, 24, 33, 28, 25, 38, 32 | 300 / 3,200 | 9.375% | 8.910% | 0.597% | -10.559 |
| P5-B physics parity | 26, 19, 20, 19, 16, 21, 20, 15, 27, 20 | 203 / 3,200 | 6.344% | 8.112% | 0.358% | -10.685 |
| P5 Unrolled EM | 36, 28, 38, 34, 27, 29, 42, 32, 36, 35 | 337 / 3,200 | 10.531% | 9.117% | 0.671% | -10.466 |
| Nominal Bussgang LMMSE | 80, 66, 82, 67, 61, 55, 83, 80, 74, 56 | 704 / 3,200 | 22.000% | 10.540% | 1.947% | -11.826 |
| Nominal IDD-PA | 77, 65, 78, 63, 54, 59, 73, 81, 76, 49 | 675 / 3,200 | 21.094% | 10.232% | 1.886% | -11.886 |
| Pilot-CPE-DD | 36, 26, 28, 28, 23, 32, 29, 25, 39, 31 | 297 / 3,200 | 9.281% | 8.917% | 0.593% | -10.558 |
| Genie-PA (privileged) | 31, 25, 32, 25, 17, 26, 27, 30, 33, 29 | 275 / 3,200 | 8.594% | 8.801% | 0.542% | -12.149 |

### R2 at 9 dB

The per-pair column follows pair order 1, 2, 3, 4, 5.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 397, 383, 333, 374, 344 | 1,831 / 7,200 | 25.431% | 11.656% | 1.557% | -17.275 |
| Local-Pure | 397, 383, 335, 380, 329 | 1,824 / 7,200 | 25.333% | 11.655% | 1.539% | -17.278 |
| Axial-Parity | 295, 285, 256, 298, 247 | 1,381 / 7,200 | 19.181% | 10.865% | 1.013% | -22.368 |
| Local-Parity | 298, 287, 252, 290, 247 | 1,374 / 7,200 | 19.083% | 10.857% | 1.010% | -22.299 |
| Axial-NoGauge | 389, 375, 329, 371, 338 | 1,802 / 7,200 | 25.028% | 11.627% | 1.525% | -17.476 |
| Fixed | 480, 484, 470, 474, 433 | 2,341 / 7,200 | 32.514% | 12.190% | 2.176% | -14.652 |
| Fixed+Parity | 365, 380, 329, 363, 335 | 1,772 / 7,200 | 24.611% | 11.549% | 1.467% | -18.137 |
| P5-B pure | 385, 391, 339, 373, 333 | 1,821 / 7,200 | 25.292% | 11.631% | 1.529% | -18.215 |
| P5-B parity | 346, 365, 325, 325, 298 | 1,659 / 7,200 | 23.042% | 11.404% | 1.356% | -19.583 |
| P5-B local | 387, 386, 334, 378, 341 | 1,826 / 7,200 | 25.361% | 11.625% | 1.555% | -18.268 |
| P5-B local parity | 355, 362, 325, 329, 299 | 1,670 / 7,200 | 23.194% | 11.405% | 1.364% | -19.616 |
| P5-B no gauge | 382, 394, 337, 354, 335 | 1,802 / 7,200 | 25.028% | 11.626% | 1.535% | -18.175 |
| P5-B physics | 449, 440, 422, 438, 397 | 2,146 / 7,200 | 29.806% | 12.001% | 1.967% | -15.447 |
| P5-B physics parity | 371, 380, 329, 372, 339 | 1,791 / 7,200 | 24.875% | 11.555% | 1.498% | -17.921 |
| P5 Unrolled EM | 410, 414, 387, 386, 363 | 1,960 / 7,200 | 27.222% | 11.758% | 1.678% | -17.709 |
| Tap phase slope | 1440, 1439, 1437, 1439, 1440 | 7,195 / 7,200 | 99.931% | 42.403% | 42.781% | 2.591 |
| 2D-LMMSE | 981, 1014, 1035, 1038, 997 | 5,065 / 7,200 | 70.347% | 15.808% | 8.265% | -10.451 |
| Waveform-DD | 376, 376, 329, 357, 339 | 1,777 / 7,200 | 24.681% | 11.467% | 1.521% | -18.272 |

### R3 at 7 dB

The per-pair column follows pair order 1, 2, 3, 4, 5.

| Receiver | Failed words by pair | Pooled errors / words | BLER | Pre-BER | Post-BER | CFR NMSE, dB |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | 49, 38, 32, 69, 48 | 236 / 1,600 | 14.750% | 7.312% | 1.569% | -13.807 |
| Local-Pure | 49, 39, 33, 68, 49 | 238 / 1,600 | 14.875% | 7.334% | 1.562% | -13.806 |
| Axial-Parity | 43, 33, 29, 64, 45 | 214 / 1,600 | 13.375% | 6.571% | 1.412% | -14.036 |
| Local-Parity | 43, 33, 29, 62, 45 | 212 / 1,600 | 13.250% | 6.574% | 1.426% | -14.013 |
| Axial-NoGauge | 49, 38, 34, 71, 48 | 240 / 1,600 | 15.000% | 7.393% | 1.587% | -13.724 |
| Fixed | 56, 46, 45, 79, 68 | 294 / 1,600 | 18.375% | 8.162% | 1.951% | -12.608 |
| Fixed+Parity | 46, 42, 38, 73, 54 | 253 / 1,600 | 15.812% | 7.463% | 1.671% | -12.976 |
| P5-B pure | 56, 47, 43, 73, 64 | 283 / 1,600 | 17.688% | 8.063% | 1.905% | -12.629 |
| P5-B parity | 49, 44, 39, 71, 58 | 261 / 1,600 | 16.312% | 7.521% | 1.716% | -12.940 |
| P5-B local | 57, 47, 43, 74, 65 | 286 / 1,600 | 17.875% | 8.050% | 1.902% | -12.616 |
| P5-B local parity | 49, 44, 38, 74, 59 | 264 / 1,600 | 16.500% | 7.513% | 1.729% | -12.924 |
| P5-B no gauge | 57, 47, 42, 75, 65 | 286 / 1,600 | 17.875% | 8.076% | 1.923% | -12.625 |
| P5-B physics | 57, 45, 44, 77, 65 | 288 / 1,600 | 18.000% | 8.133% | 1.931% | -12.620 |
| P5-B physics parity | 47, 43, 38, 73, 54 | 255 / 1,600 | 15.937% | 7.563% | 1.706% | -12.946 |
| P5 Unrolled EM | 67, 53, 53, 85, 73 | 331 / 1,600 | 20.688% | 8.539% | 2.285% | -11.194 |
| 2D-LMMSE | 66, 61, 53, 93, 78 | 351 / 1,600 | 21.938% | 9.077% | 2.526% | -10.488 |
| Waveform-DD | 49, 40, 40, 77, 59 | 265 / 1,600 | 16.562% | 7.487% | 1.832% | -12.972 |

## R3 Green alignment

Each of the ten R3 probe pairs contributes eight stream-8 bursts. At each waveform pass the overall Pearson coefficient uses 2,256 off-diagonal token pairs. The direct map averages the three parallel axial transports on their supports. The historical product composes the three measured axis maps. The shuffled control averages eight stream-9 key permutations. The pair-matched uniform-support mean was 0.320608 on cohort A and 0.268702 on development.

Spatial axis-only correlation restricts endpoints to the same antenna line. The comparison against 0.084 uses a fixed historical P5 reference with different support. It describes distance from that reference and cannot establish a paired superiority claim against the P5 spatial operator.

| Receiver | Map | Initial | Trained | Trained shuffled | Initial on-axis | Trained on-axis |
| --- | --- | --- | --- | --- | --- | --- |
| Axial-Pure | Direct joint | 0.384341 | 0.384649 | -0.001385 | — | — |
| Axial-Pure | Historical product | 0.579192 | 0.584270 | -0.000582 | — | — |
| Axial-Pure | Spatial | 0.129506 | 0.130712 | 0.000638 | 0.885386 | 0.892148 |
| Axial-Pure | Time | -0.018572 | -0.018608 | 0.000017 | 0.782369 | 0.779380 |
| Axial-Pure | Frequency | 0.476630 | 0.481494 | -0.001614 | 0.830999 | 0.867632 |
| Axial-Parity | Direct joint | 0.384273 | 0.384895 | -0.001388 | — | — |
| Axial-Parity | Historical product | 0.580500 | 0.585587 | -0.000596 | — | — |
| Axial-Parity | Spatial | 0.129673 | 0.130708 | 0.000668 | 0.885311 | 0.892253 |
| Axial-Parity | Time | -0.018242 | -0.018140 | 0.000477 | 0.782380 | 0.780421 |
| Axial-Parity | Frequency | 0.477398 | 0.481771 | -0.001695 | 0.831858 | 0.869833 |

## Fixed-weight interventions

Each row omits the named handle at inference on the ten prospective pairs. Positive changes in failed words indicate that the intervention worsened decoding. Multiple handles can affect the same block.

| Regime | Receiver | Intervention | Failed / total | Change |
| --- | --- | --- | --- | --- |
| R1 | Axial-Parity | full | 171 / 3,200 | +0 |
| R1 | Axial-Parity | cpe | 165 / 3,200 | -6 |
| R1 | Axial-Parity | radial | 171 / 3,200 | +0 |
| R1 | Axial-Parity | radial and cpe | 166 / 3,200 | -5 |
| R1 | Axial-Parity | delta h | 180 / 3,200 | +9 |
| R1 | Axial-Parity | dewarp | 171 / 3,200 | +0 |
| R1 | Axial-Parity | all | 171 / 3,200 | +0 |
| R1 | Local-Parity | full | 176 / 3,200 | +0 |
| R1 | Local-Parity | cpe | 167 / 3,200 | -9 |
| R1 | Local-Parity | radial | 176 / 3,200 | +0 |
| R1 | Local-Parity | radial and cpe | 165 / 3,200 | -11 |
| R1 | Local-Parity | delta h | 174 / 3,200 | -2 |
| R1 | Local-Parity | dewarp | 175 / 3,200 | -1 |
| R1 | Local-Parity | all | 171 / 3,200 | -5 |
| R2 | Axial-Parity | full | 2,805 / 14,400 | +0 |
| R2 | Axial-Parity | phase | 3,362 / 14,400 | +557 |
| R2 | Axial-Parity | bank | 3,471 / 14,400 | +666 |
| R2 | Axial-Parity | delta h | 2,944 / 14,400 | +139 |
| R2 | Axial-Parity | all | 3,523 / 14,400 | +718 |
| R2 | Local-Parity | full | 2,792 / 14,400 | +0 |
| R2 | Local-Parity | phase | 3,357 / 14,400 | +565 |
| R2 | Local-Parity | bank | 3,468 / 14,400 | +676 |
| R2 | Local-Parity | delta h | 2,965 / 14,400 | +173 |
| R2 | Local-Parity | all | 3,523 / 14,400 | +731 |
| R3 | Axial-Parity | full | 444 / 3,200 | +0 |
| R3 | Axial-Parity | radial | 440 / 3,200 | -4 |
| R3 | Axial-Parity | delays | 511 / 3,200 | +67 |
| R3 | Axial-Parity | Spatial | 459 / 3,200 | +15 |
| R3 | Axial-Parity | spatial operator | 458 / 3,200 | +14 |
| R3 | Axial-Parity | irc | 444 / 3,200 | +0 |
| R3 | Axial-Parity | ula | 458 / 3,200 | +14 |
| R3 | Axial-Parity | delta h | 448 / 3,200 | +4 |
| R3 | Axial-Parity | all | 527 / 3,200 | +83 |
| R3 | Local-Parity | full | 442 / 3,200 | +0 |
| R3 | Local-Parity | radial | 446 / 3,200 | +4 |
| R3 | Local-Parity | delays | 509 / 3,200 | +67 |
| R3 | Local-Parity | Spatial | 457 / 3,200 | +15 |
| R3 | Local-Parity | spatial operator | 459 / 3,200 | +17 |
| R3 | Local-Parity | irc | 442 / 3,200 | +0 |
| R3 | Local-Parity | ula | 456 / 3,200 | +14 |
| R3 | Local-Parity | delta h | 449 / 3,200 | +7 |
| R3 | Local-Parity | all | 527 / 3,200 | +85 |

## Single-thread latency

Each entry reports the median and interquartile interval in milliseconds for serial receiver forwards. Host background CPU includes unrelated work observed during the timed rounds.

All study worker pools had exited. The timed process used one intra-op and one inter-op thread pinned to logical CPU 0. Median background CPU across the six cells ranged from 13.3% to 20.0%. The scanner identified zero concurrent project workers and recorded three Python processes whose working directories it could not inspect. A subsequent Windows process query placed all three in service session 0 under agy-bin.exe, with creation times on the preceding day. [Round-level timings and process scans](results/latency.json)

| Regime | Batch | Method | Median ms | Q1–Q3 ms | Median background CPU |
| --- | --- | --- | --- | --- | --- |
| R1 | 1 | Axial-Pure | 22.361 | 20.575–24.014 | 13.3% |
| R1 | 1 | Local-Pure | 18.264 | 15.990–20.291 | 13.3% |
| R1 | 1 | Axial-Parity | 21.418 | 21.013–24.605 | 13.3% |
| R1 | 1 | Local-Parity | 19.649 | 17.660–20.916 | 13.3% |
| R1 | 1 | Axial-NoGauge | 18.989 | 17.355–19.977 | 13.3% |
| R1 | 1 | Fixed | 11.156 | 10.575–11.975 | 13.3% |
| R1 | 1 | Fixed+Parity | 14.183 | 12.748–14.716 | 13.3% |
| R1 | 1 | Nominal Bussgang LMMSE | 1.760 | 1.644–3.299 | 13.3% |
| R1 | 1 | Nominal IDD-PA | 6.582 | 5.515–7.323 | 13.3% |
| R1 | 1 | Pilot-CPE-DD | 5.343 | 4.882–5.940 | 13.3% |
| R1 | 1 | Genie-PA (privileged) | 6.050 | 5.618–7.298 | 13.3% |
| R1 | 64 | Axial-Pure | 350.001 | 336.274–387.899 | 19.4% |
| R1 | 64 | Local-Pure | 232.588 | 224.501–298.785 | 19.4% |
| R1 | 64 | Axial-Parity | 341.602 | 318.094–404.951 | 19.4% |
| R1 | 64 | Local-Parity | 239.071 | 224.831–286.384 | 19.4% |
| R1 | 64 | Axial-NoGauge | 307.275 | 297.614–380.314 | 19.4% |
| R1 | 64 | Fixed | 100.160 | 92.190–109.728 | 19.4% |
| R1 | 64 | Fixed+Parity | 125.121 | 110.638–146.282 | 19.4% |
| R1 | 64 | Nominal Bussgang LMMSE | 27.057 | 24.297–28.995 | 19.4% |
| R1 | 64 | Nominal IDD-PA | 60.138 | 53.878–84.810 | 19.4% |
| R1 | 64 | Pilot-CPE-DD | 54.674 | 52.062–58.292 | 19.4% |
| R1 | 64 | Genie-PA (privileged) | 75.106 | 74.379–85.380 | 19.4% |
| R2 | 1 | Axial-Pure | 37.006 | 29.498–43.068 | 16.4% |
| R2 | 1 | Local-Pure | 20.638 | 20.095–25.400 | 16.4% |
| R2 | 1 | Axial-Parity | 31.995 | 30.448–35.354 | 16.4% |
| R2 | 1 | Local-Parity | 23.894 | 22.678–29.546 | 16.4% |
| R2 | 1 | Axial-NoGauge | 27.943 | 25.507–28.596 | 16.4% |
| R2 | 1 | Fixed | 7.187 | 7.180–8.842 | 16.4% |
| R2 | 1 | Fixed+Parity | 10.512 | 9.486–11.961 | 16.4% |
| R2 | 1 | Tap phase slope | 2.412 | 2.335–3.438 | 16.4% |
| R2 | 1 | 2D-LMMSE | 3.278 | 2.851–4.575 | 16.4% |
| R2 | 1 | Waveform-DD | 14.277 | 11.784–18.449 | 16.4% |
| R2 | 64 | Axial-Pure | 1162.412 | 1085.768–1323.547 | 16.4% |
| R2 | 64 | Local-Pure | 836.771 | 777.398–978.434 | 16.4% |
| R2 | 64 | Axial-Parity | 1213.252 | 1190.588–1430.164 | 16.4% |
| R2 | 64 | Local-Parity | 822.239 | 745.859–966.470 | 16.4% |
| R2 | 64 | Axial-NoGauge | 1110.809 | 1030.475–1263.499 | 16.4% |
| R2 | 64 | Fixed | 155.132 | 150.626–170.202 | 16.4% |
| R2 | 64 | Fixed+Parity | 206.503 | 197.557–219.479 | 16.4% |
| R2 | 64 | Tap phase slope | 43.345 | 42.349–50.627 | 16.4% |
| R2 | 64 | 2D-LMMSE | 143.324 | 141.237–183.700 | 16.4% |
| R2 | 64 | Waveform-DD | 360.220 | 354.544–380.891 | 16.4% |
| R3 | 1 | Axial-Pure | 31.120 | 29.535–34.063 | 13.9% |
| R3 | 1 | Local-Pure | 23.061 | 22.708–25.816 | 13.9% |
| R3 | 1 | Axial-Parity | 33.367 | 32.409–33.797 | 13.9% |
| R3 | 1 | Local-Parity | 26.914 | 25.906–29.271 | 13.9% |
| R3 | 1 | Axial-NoGauge | 29.312 | 28.313–43.797 | 13.9% |
| R3 | 1 | Fixed | 10.088 | 9.454–10.629 | 13.9% |
| R3 | 1 | Fixed+Parity | 11.323 | 10.935–13.216 | 13.9% |
| R3 | 1 | 2D-LMMSE | 1.456 | 1.432–1.517 | 13.9% |
| R3 | 1 | Waveform-DD | 7.766 | 7.630–8.507 | 13.9% |
| R3 | 64 | Axial-Pure | 1371.364 | 1273.462–1542.792 | 20.0% |
| R3 | 64 | Local-Pure | 839.525 | 751.590–899.786 | 20.0% |
| R3 | 64 | Axial-Parity | 1306.500 | 1260.352–1566.965 | 20.0% |
| R3 | 64 | Local-Parity | 892.787 | 864.586–1013.153 | 20.0% |
| R3 | 64 | Axial-NoGauge | 1339.168 | 1196.633–1450.722 | 20.0% |
| R3 | 64 | Fixed | 113.912 | 103.434–135.952 | 20.0% |
| R3 | 64 | Fixed+Parity | 141.096 | 122.438–150.072 | 20.0% |
| R3 | 64 | 2D-LMMSE | 27.864 | 25.465–30.756 | 20.0% |
| R3 | 64 | Waveform-DD | 115.019 | 96.542–119.157 | 20.0% |

## Verification and benchmark provenance

Checkpoint replay covered 250 checkpoints across 50 pairs. Frame and codeword digests matched on 50/50 pairs, integer scores on 50/50 and complete score dictionaries on 50/50. The complete replay pass flag was True. [Replay details](results/checkpoint_replay.json)

Trained Axial-Pure and Axial-Parity passed 599/600 gain and pilot conditions, with 0 coded-bit flips in 35,942,400 comparisons, maximum absolute ΔLLR 1.0160761e-12 and maximum relative CFR error 2.9347489e-14. The strict ΔLLR bound was 1e-12. [Trained audit](results/trained_gauge.json)

Cohort B archive checks passed on 20/20 pairs. The six archived integer fields cover pre-decoder coded-bit errors and denominator, post-decoder information-bit errors and denominator, and block errors and denominator. Every archived P5-B frame and codeword digest is compared. The P5 Unrolled EM and P5-B entries in the cohort B tables use restored archived weights.

P4 supplies the coding, waveform and baseline lineage. P4 used different experimental designs, including two-dimensional slots and sparse DMRS. P6 shares the P5/P5-B simulator draws and legacy seeds in cohort B. Paired numerical comparisons with P4's published blocks would lack a common realization. [P4 evidence](../p4/P4_COMPLETION_EVIDENCE.md)

Checkout passed = True. P6 ran 38 tests (pass = True), P5 ran 15 and P5-B ran 13 (legacy pass = True). The publication P4 verifier exited 0. Protected checkout pass = True. [Checkout](results/checkout_verification.json)

On the twenty legacy pairs, P6 Axial-Parity recorded 188, 1,381 and 214 failed words in R1, R2 and R3. The archived P5 Unrolled EM recorded 337, 1,960 and 331. P5-B Parity recorded 192, 1,659 and 261. These counts share the archived draws and are descriptive comparisons. P4's separate 10 dB in-distribution experiment found 3,266/4,160 failures for its EM receiver and 3,211/4,160 for its fixed-prior IDD-PA, Holm p = 0.0234375. Its waveform and impairment design differ from P6, which prevents a paired cross-study BLER estimate. [P4 study](../p4/STAGE3_RESULTS.md)

The frozen checkout command completed the P6 tests and then encountered a standard-library discovery error because the archived test directories lack a package initializer. Verification resumed using each original test directory as its own discovery root. Both archived suites ran on their original fixtures. The 617 tracked-file hashes, 303 protected P5/P5-B file hashes, development firewall and frozen source manifest matched after evaluation. [Checkout receipt and recovery note](results/checkout_verification.json)
