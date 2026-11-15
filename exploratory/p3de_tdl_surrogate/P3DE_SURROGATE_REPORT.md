# EM-FloatFormer — P3-D / P3-E standards-bridge **surrogate** results

**2026-11-15 · Completed for the standalone surrogate only · Formal NVIDIA Sionna 2.2 P3-D/P3-E NOT completed.**

## Evidence scope and provenance

The independent CPU Python simulation uses the **TDL-A/C/D power-delay parameter values** from NVIDIA Sionna 2.2 (TR 38.901 V19.2) as **input tables**, but **does not import, execute or validate against the Sionna 2.2 package**. Installation of `sionna-no-rt==2.2.0` was attempted but unavailable because PyPI hostname resolution failed in the isolated container. This study cannot be called a 3GPP-certified, Sionna-validated or real RF receiver experiment.

The [separate frozen protocol](P3DE_SURROGATE_PROTOCOL.md) was committed to the original repository **before training**. Existing P2-E and P3-A/A2/B/C frozen results were not overwritten. New code maintains passive, memory-only, civilian QPSK/OFDM scope; there is no intentional interference, radar, emission control or real equipment.

### Modelled link

32 subcarriers spaced by 30kHz, one all-subcarrier known pilot, one static uncoded QPSK data symbol, complex Gaussian thermal noise, analytical fractional-delay frequency response with TDL powers and delays. The 16-sample CP is an **assumption checked for maximum path delay**, not an executed time-domain CP/FFT pipeline. TDL-D has an approximate independent specular LoS component; no Doppler, motion, shadowing, full TDL/CDL polarisation, NR scheduling or Sionna receiver grid.

10 new training seeds **61001–61010** trained two P3-A2 architecture families (A with location and no gain floor; D with location removed and 10% gain floor), 300 AdamW steps × batch64 per model, alternating TDL-A/TDL-C at 500ns and SNR uniform 0–20dB; **20 independent checkpoint files**. They share initialization and exact training frames within each seed. Testing uses disjoint seeded draws, 2048 frames per seed × 64 bits = **1,310,720 bits per method/scenario/SNR**. Independent training seeds are the replication units, not individual bits.

## Full held-out 10dB results

| Receiver | TDL-A 500ns (train family) | TDL-C 1000ns (unseen spread) | TDL-D 750ns (unseen LoS) |
|---|---:|---:|---:|
| `A_pos_0` | 5.483% | 6.118% | 0.894% |
| `D_no_pos_10` | 5.604% | 6.256% | 0.912% |
| `LS` | 7.959% | 8.003% | 2.283% |
| `DFT8` | 6.568% | 7.171% | 0.960% |
| `DFT12` | 7.025% | 7.455% | 1.117% |
| `Threshold` | 6.505% | 7.476% | 0.954% |
| `Train-LMMSE` | 4.876% | 6.017% | 0.894% |
| `Test-LMMSE` **(privileged)** | 4.832% | 5.240% | 0.786% |
| `TrueH` **(true H, inaccessible)** | 4.320% | 4.369% | 0.635% |

`Train-LMMSE` uses an analytic covariance computed solely from the **two known training distributions**, which gives it an explicit distributional model; `Test-LMMSE` knows the held-out test covariance and is privileged. DFT8 and DFT12 assume integer-delay support, which is not matched exactly by the fractional-delay profiles. No method accesses instantaneous H except the labelled unreachable diagnostic.

![Recorded surrogate BER](p3de_results/ber_10db.png)

## Frozen surrogate primary comparisons, 10dB

Prespecified delta = `BER(Threshold) - BER(D)` (percentage points). Positive favors D; two-sided exhaustive 10-seed sign-flip tests, Holm-adjusted for two endpoints.

| Scenario | Mean difference (pp) | Two-sided descriptive t(9) 95% interval (pp) | Exact two-sided p | Holm p |
|---|---:|---|---:|---:|
| test_C_1000ns | +1.2198 | [+1.1632, +1.2764] | 0.001953 | 0.003906 |
| test_D_750ns | +0.0423 | [+0.0194, +0.0652] | 0.007812 | 0.007812 |

Seed differences for `test_C_1000ns` (pp): +1.1833, +1.2672, +1.1909, +1.2634, +1.0612, +1.2573, +1.3664, +1.1864, +1.1925, +1.2291.
Seed differences for `test_D_750ns` (pp): +0.0572, +0.0534, +0.0221, +0.0572, +0.0519, +0.0664, +0.0557, +0.0664, -0.0397, +0.0320.

**Interpretation:** D improves relative to the *fixed hard-threshold* estimator on these specific static table-driven scenarios. This is **not** a general neural victory: on both held-out 10dB profiles the **nonprivileged training-mixture LMMSE matches or improves on D**, and model A has lower pooled BER than D. The static surrogate cannot establish that any of these differences persist in Sionna 2.2 dynamic fading, realistic pilot patterns, or hardware.

## Other SNRs and interpretation

| Profile | SNR | A | D | Threshold | Train-LMMSE | Test-LMMSE (privileged) |
|---|---:|---:|---:|---:|---:|---:|
| train_A_500ns | 0dB | 24.147% | 24.926% | 27.252% | 22.832% | 22.776% |
| train_A_500ns | 20dB | 0.809% | 0.812% | 1.088% | 0.594% | 0.578% |
| test_C_1000ns | 0dB | 24.608% | 25.380% | 27.932% | 23.542% | 23.293% |
| test_C_1000ns | 20dB | 0.954% | 0.900% | 1.224% | 1.434% | 0.634% |
| test_D_750ns | 0dB | 18.739% | 18.891% | 18.542% | 18.819% | 18.014% |
| test_D_750ns | 20dB | 0.007% | 0.007% | 0.007% | 0.010% | 0.004% |

Reported complex channel MSE must not be interpreted as an accurate physical channel estimate if the neural model is trained only with bit BCE: at 10dB, D has high channel MSE despite competitive QPSK BER (channel magnitude affects MSE but not positive-real-scale QPSK sign decisions). Full channel-MSE columns are in raw CSV.

## CPU model-forward latency, exact tested process

| Method | Single frame median | Batch64 median |
|---|---:|---:|
| A_pos_0 | 0.2209ms | 0.5349ms |
| D_no_pos_10 | 0.2173ms | 0.5440ms |
| Threshold | 0.0185ms | 0.0330ms |
| DFT8 | 0.0078ms | 0.0115ms |

These timings include only PyTorch CPU model/channel-estimate forward calls, not simulator generation, complete OFDM receive pipeline, transport, host/device transfer, energy or GPU inference. Model and baseline pipelines perform different arithmetic. A lower BER alone is not evidence of a better engineering choice.

## Independent audit

- Surrogate profile mathematical checks passed: normalized powers, positive-semidefinite Hermitian covariance, CP-path-envelope, cyclic integer-delay convolution identity. Independent NumPy analytic covariance cross-check max absolute error **3.47e-08**; 10dB observed pilot-noise variance **0.09996** (target 0.1).
- **20/20** checkpoint SHA256 files matched, **180** exact endpoint integer error+bit counts reproduced for all 10 seeds, **80** complex-common-phase numerical checks passed; 810 raw per-seed BER records reconciled to 81 summaries. This is a same-runtime replay, not independent hardware or Sionna replication.
- Training and held-out data RNGs are disjoint; within-scenario received observations are paired among neural and classical methods. The tables and p-values were calculated from saved raw integer error counts, with no post-test hyperparameter updates.

## Formal P3-D / P3-E status and explicit blockers

**Unfinished (do not mark complete):** true Sionna 2.2 execution; validated 3GPP TDL-A/C/D and CDL time evolution with Doppler; Sionna `LMMSEChannelEstimator`; standard multi-symbol resource-grid pilot patterns and coding; complete receiver-link timing/GPU benchmark. The failed `pip install sionna-no-rt==2.2.0` cannot be resolved in this internet-isolated runtime, so a Sionna-backed P3-E ten-seed confirmation does not exist. The above 10-seed tests confirm only the **narrow standalone surrogate endpoints**; they are separate from the proposed official P3-E.

If a compatible Sionna environment is available, first run a new Sionna-specific pilot/data channel and estimator-equivalence audit, freeze separate endpoints/compute plan before inspecting held-out outcomes, and then undertake formal P3-D and P3-E with new seeds. Do not recycle these surrogate test seeds as a formal confirmation set.

## Reproduction

```bash
python -m pip install torch numpy scipy matplotlib
python p3de_profile_bridge.py
python p3de_experiment.py --mode smoke
python p3de_experiment.py --mode train --out p3de_results
python p3de_experiment.py --mode eval --out p3de_results
python p3de_experiment.py --mode finalize --out p3de_results
python p3de_independent_audit.py
python p3de_make_report.py
```

No arbitrary changes to models, steps, profiles or fixed classical thresholds were performed after frozen surrogate protocol publication. Original P1/P2/P3-A/B/C GitHub history remains unchanged.
