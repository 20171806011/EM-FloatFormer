# P2-D — Frozen-weight synthetic robustness matrix (completed, exploratory)

Five frozen P2 candidates, trained on train3 with 600 steps in P2-A/B/C, evaluated on the same held-out scenario draws. No retraining or test-specific tuning. Three exploratory seeds 1201/1202/1203, 768 frames per seed/scenario (147,456 data bits per method/condition). Baselines: LS, fixed 8-tap DFT, train-prior LMMSE, matched-test-prior LMMSE (privileged). Data cover five multipath power-delay profiles × SNR 0/10/20 dB, plus BF16 storage of IQ, ±6 dB errors in **reported** noise variance, and ±0.35 rad random data-only phase error on train3/equal4/long6. These are simplified receive-side measurement perturbations, not full oscillators or RF hardware.

## Unperturbed 10-dB BER (%), pooled over three seeds

| Receiver | Train3 | Equal4 | Long6 | Two tap | Late-energy three tap |
|---|---:|---:|---:|---:|---:|
| P2 phase-equivariant complex FIR | 5.653 | 6.281 | 7.326 | 5.240 | 5.487 |
| P2 dual-path local residual | 5.118 | 6.318 | 8.411 | 4.712 | 5.016 |
| P2 delay unfolded shrinkage | 7.060 | 7.452 | 7.314 | 6.698 | 7.385 |
| P1 CNN/log retrained | 5.497 | 6.554 | 8.301 | 4.989 | 5.324 |
| LS | 8.129 | 8.074 | 7.940 | 7.770 | 7.703 |
| **Fixed DFT-8** | 5.511 | **5.356** | **5.309** | 5.144 | 5.204 |
| Matched-profile LMMSE (**privileged prior**) | **4.875** | **4.942** | **5.069** | **4.405** | **4.597** |

No learned model wins universally. Dual-path local network is best on some easier/shorter profiles, but fails long6. Equivariant FIR is the most consistent neural candidate across **both** predeclared unknown-path-length endpoints (equal4/long6) while keeping train3 degradation vs CNN under 2 percentage points. DFT8 still outperforms even that candidate, demonstrating the importance of a generic but valid known-delay-support assumption.

## Representative P2 phase-equivariant FIR vs P1 CNN/log and DFT8, 10 dB

| Condition / profile | Equivariant FIR | P1 CNN/log | Fixed DFT8 |
|---|---:|---:|---:|
| equal4 normal | 6.281 | 6.554 | 5.356 |
| equal4 BF16 input storage | 6.371 | 6.620 | 5.480 |
| equal4 reported noise +6 dB | 6.300 | 6.958 | 5.343 |
| equal4 reported noise -6 dB | 6.222 | 6.312 | 5.334 |
| equal4 data-only phase error | 7.088 | 7.324 | 6.069 |
| long6 normal | 7.326 | 8.301 | 5.309 |
| long6 BF16 input storage | 7.345 | 8.351 | 5.371 |
| long6 reported noise +6 dB | 7.279 | 9.090 | 5.373 |
| long6 reported noise -6 dB | 7.309 | 7.375 | 5.273 |
| long6 data-only phase error | 8.162 | 9.293 | 6.089 |

BF16 affects input storage only, not network arithmetic. Noise-report perturbations can appear to improve a method under a finite training distribution; do not interpret as calibration advice. A common rotation of pilot+data is a mathematical symmetry, but an independent data-only phase drift **breaks** that symmetry and worsens BER.

## P2-E freeze selection (written before final seeds 2201–2210)

**Single frozen primary P2 candidate: `res_equiv_local`**, because it beats new P1 CNN/log on both exploratory unknown profiles without >2 BER-point train3 regression, has an exact U(1) symmetry by construction, and requires only 2,060 parameters. All other exploratory variants are excluded from P2-E confirmatory primary comparisons. P2-E two primary endpoints:

`delta_seed = BER(P1 CNN/log) − BER(P2 equivariant FIR)` on **equal4** and **long6**, each at **10 dB**, for **10 fresh training seeds** and 2048 new frames/seed/scenario. Positive delta favors P2. Do two-sided exact paired sign-flip randomization tests across 10 seeds and Holm correction for two endpoints; report the paired effect, per-seed deltas and descriptive t(9) intervals. This is a new-seed generalization study using the **same simulator and code**, not independent laboratory/SDR implementation. Also compare P2 with the fixed DFT8 classical benchmark as contextual external-validity check, without post hoc p-value hunting. Train both P2 and P1 for 600 steps on the exact same training frame stream per seed. No changing architectural or training settings after final tests.

Evidence: `p2d_results/ber_by_seed.csv`, `ber_summary.csv`, `latency_cpu.csv`, `run_config.json`. Limitations: short 32-tone, full-pilot, perfect synchronization synthetic test; no sampling-rate changes, real SDR, channel coding, actual hardware power or physical propagation measurements.
