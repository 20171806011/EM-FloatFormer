# EM-FloatFormer P3-0 / P3-A — mathematical audit and delay-domain model evaluation

**2026-11-15 · Completed · exploratory · synthetic receive-only civilian QPSK/OFDM only.**

## Protocol and frozen scope

- Frozen on GitHub before P3-A training: [`P3_0_A_PROTOCOL.md`](P3_0_A_PROTOCOL.md), commit `bce622eddc0d8277c64073033c1248eb035a3803`. This code/data report was generated *after* the frozen protocol; no later training-budget changes or new test conditions were introduced.
- Model code: `p3_models.py`; training/evaluation: `p3_a.py`; independent tests: `test_p3_0_a.py` and `p3_a_audit.py`. The archived P2-E confirmatory results are unchanged. P3-B/C/D/E have **not** been started.
- Five *new* training seeds 46001–46005, 600 AdamW steps per model, batch64, train3 only, 0–20 dB randomly sampled SNR; all three neural model families receive identical training frames per seed. Exactly 15 newly trained checkpoints. Decoder supervision is QPSK BCE, **not** channel MSE.
- Five profiles × 0/10/20dB, 768 held-out frames per seed × 64 bits/frame => **245,760 bits per receiver per scenario**, same simulated frames across estimators; 5 seeds are the statistical replication units, **not** the 245,760 bits.
- All tests remain 32-tone full-pilot, one static pilot/data pair, AWGN, ideal synchronization, no coding, no genuine ADC or RF hardware. All results exploratory, no confirmatory superiority or preregistered p-value claims.

## P3-0 mathematical result and reproducibility

Let `D=diag(1,...,1,0,...,0)` be the length-32 first-eight-delay mask. The physical DFT-8 estimator is `FFT(D*IFFT(Ypilot))`. By the discrete convolution theorem this is circular convolution of the 32-tone pilot observation with a frequency-domain kernel `k=FFT(diag(D))/32`. The exact kernel has **25 nonzero coefficients**. Its nearest kernel whose lag support is restricted to `{-4,-3,-2,-1,0,1,2,3,4}` has relative coefficient L2 error **0.3029641990831974**. The independent double-precision circular convolution identity passed to `1e-12` numerical tolerance.

**Important scope:** this is a restriction of the **fixed linear nine-tap convolution component**, not a lower bound on the entire `res_equiv_local` network because that network includes nonlinear input-dependent invariant gates. The older `res_unfold` already contained an IFFT/real shrinkage/FFT path; P3-A is not the first attempt at a delay-domain learned representation.

The P3 model performs `t=IFFT(Ypilot)` and `Hhat=FFT[t * sigmoid(g_theta(invariants))]`, where the real gate is computed by two circular 1-D convolutions of invariant amplitude/noise/delay-index features. For any `u=exp(i phi)`, the gate is unchanged by multiplying `Ypilot` by `u`, hence `Hhat(uYpilot)=uHhat(Ypilot)`. The QPSK equalized output is invariant when both pilot and data are rotated by `u`; this is **common rotation** equivariance, not data-only CFO/CPE invariance.

Pre-training audit: 9/9 math/architecture tests passed, including finite gradients, delay-domain cross-check and theoretical false-threshold crossing test. Trained checkpoint phase-audit: **45/45 cases** (5 seeds × 3 learned models × 3 angles) with zero changed hard bits. Maximum channel equivariance error: `3.64e-07`; maximum logit difference `4.9e-06` (floating-point).

## BER results at 10 dB

Values are **BER % (lower is better)**, pooled across five new seeds on 245,760 bits per receiver/profile. Correct-test LMMSE has unavailable privileged profile statistics. DFT-8 support assumption is valid only through delay 7; DFT-12 knows maximum support12. Threshold methods use only the reported noise variance and a prespecified noise-only false-crossings rate 0.05; no profile-specific tuning.

| Method | train3 | equal4 | long6 | long12 | sparse12 |
|---|---:|---:|---:|---:|---:|
| P3 new equivariant delay CNN (985 params) | 4.863% | 5.393% | 6.442% | 11.411% | 12.016% |
| P2 delay unfolded (722 params) | 7.162% | 7.753% | 7.622% | 8.065% | 7.907% |
| P2 equivariant frequency FIR (2,060 params) | 5.426% | 6.272% | 7.466% | 10.688% | 10.720% |
| LS | 8.000% | 8.056% | 8.073% | 8.148% | 8.000% |
| DFT-8 (assumes delay <= 7) | 5.399% | 5.389% | 5.359% | 9.583% | 12.436% |
| DFT-12 (known extended support) | 5.854% | 5.906% | 5.867% | 5.930% | 5.809% |
| DFT-threshold hard (no support prior) | 4.924% | 5.162% | 5.633% | 7.204% | 4.857% |
| DFT-threshold soft (no support prior) | 5.398% | 5.892% | 6.823% | 9.510% | 5.379% |
| LMMSE train3 population covariance | 4.781% | 14.406% | 13.570% | 20.044% | 22.791% |
| LMMSE correct test covariance (privileged) | 4.781% | 4.926% | 5.142% | 5.901% | 4.714% |

## Prespecified exploratory contrasts

We report new-minus-comparator differences in **BER percentage points**. Negative means the P3 model is better. Both long6 and long12 are evaluated at 10dB without any selection of checkpoint or baseline after observing the test.

| Profile | Comparator | Mean new − comparator (pp) | Five paired-seed differences (pp) |
|---|---|---:|---|
| long6 | res_unfold | -1.179 | -0.840, -0.631, -2.244, -1.300, -0.881 |
| long6 | res_equiv_local | -1.024 | -1.257, -0.659, -1.465, -0.844, -0.893 |
| long6 | dft8 | +1.084 | +1.123, +1.243, +0.513, +1.298, +1.241 |
| long6 | dft_thresh_hard | +0.810 | +0.806, +1.064, +0.193, +1.050, +0.936 |
| long6 | dft_thresh_soft | -0.380 | -0.476, -0.004, -0.930, -0.146, -0.346 |
| long12 | res_unfold | +3.346 | +3.233, +3.542, +2.264, +3.963, +3.727 |
| long12 | res_equiv_local | +0.722 | +0.264, +0.614, +0.216, +1.428, +1.088 |
| long12 | dft8 | +1.828 | +1.672, +1.945, +0.818, +2.490, +2.216 |
| long12 | dft_thresh_hard | +4.207 | +4.191, +4.213, +3.060, +4.818, +4.751 |
| long12 | dft_thresh_soft | +1.901 | +1.872, +2.132, +0.832, +2.417, +2.250 |

Despite consistent directional differences on some comparisons, n=5 is an exploratory sample. No multiple-comparison-confirmatory inference is made. No hyperparameter changes followed the tests.

## Additional 0/20-dB stress results

| Profile | SNR | P3 delay | P2 delay | DFT-8 | DFT-threshold hard | DFT-12 |
|---|---:|---:|---:|---:|---:|---:|
| train3 | 0 | 23.152% | 27.567% | 25.042% | 25.606% | 26.490% |
| train3 | 20 | 0.592% | 1.016% | 0.653% | 0.579% | 0.725% |
| equal4 | 0 | 23.949% | 27.985% | 25.186% | 27.036% | 26.525% |
| equal4 | 20 | 1.097% | 0.984% | 0.628% | 0.577% | 0.683% |
| long6 | 0 | 24.956% | 28.195% | 25.102% | 28.724% | 26.423% |
| long6 | 20 | 1.742% | 0.988% | 0.652% | 0.642% | 0.712% |
| long12 | 0 | 28.092% | 28.796% | 26.884% | 32.181% | 26.336% |
| long12 | 20 | 5.690% | 0.953% | 5.431% | 0.696% | 0.670% |
| sparse12 | 0 | 26.977% | 28.209% | 28.269% | 25.660% | 26.519% |
| sparse12 | 20 | 8.012% | 1.029% | 9.030% | 0.581% | 0.697% |

## Channel MSE is NOT the same objective as BER

At train3/10dB the P3 model BER is 4.863%, but its **channel MSE is 0.41320**, compared with **0.01345** for the hard-threshold DFT. The P3 model is trained to classify bits, and shrinking pilot-based channel amplitudes can preserve approximate phase/bit decisions while changing channel amplitudes substantially. A low BER is **not** evidence for high-fidelity complex channel reconstruction.

To inspect the failure mode without retraining, one stored model (seed46001) was evaluated with a diagnostic output-hook at 10dB: mean shrinkage gate over early delays0–2 ≈0.410; delays8–11 ≈0.002; beyond12 ≈0.005 in train3. This is a *descriptive single-seed diagnostic* and explains why the model can suppress delayed paths it did not encounter during training. It is not a causal proof or an average across seeds.

## CPU inference measurements (single environment)

| Method | Batch1 median (ms) | Batch1 p95 (ms) | Batch64 median (ms) | Params |
|---|---:|---:|---:|---:|
| p3_equiv_delay | 0.2394 | 0.3481 | 0.5849 | 985 |
| res_unfold | 0.1237 | 0.1729 | 0.3535 | 722 |
| res_equiv_local | 0.1675 | 0.2902 | 0.6893 | 2060 |
| ls | 0.0234 | 0.0254 | 0.0428 | 0 |
| dft8 | 0.0389 | 0.0521 | 0.0668 | 0 |
| dft_thresh_hard | 0.0486 | 0.0723 | 0.0866 | 0 |
| dft_thresh_soft | 0.0560 | 0.0842 | 0.0935 | 0 |

These are **PyTorch CPU receiver-forward** times, excluding SDR hardware, protocol framing and real-time RF sample handling. Parameters or latency alone are not proof of production efficiency.

## Integrity audit

- 15/15 new checkpoints checksum-verified. 825/825 raw per-seed method × condition cells reconciled with 165 aggregate BER rows; **220/220 integer (errors,bits) pairs** independently recomputed from saved checkpoints without retraining. All original counts and profiles saved in `p3a_results/`.
- 45 trained-model U(1) phase checks across all 5 seeds passed with zero hard-decision changes. The `test_p3_0_a.py` unit suite passed 9/9 cases.
- Training RNG family and held-out evaluation family are disjoint. All receivers evaluated the same simulated frames within each scenario. `p3a_results/independent_audit.json` contains SHA256 of new model files and checkpoints.
- Both conventional threshold variants preserve the frozen threshold `T=sqrt(noise_var/32 * ln(32/0.05))`. Under noise alone the threshold-crossing test agrees with its theoretical expected 0.05 false crossings/frame.

## Interpretation and stopping point

**P3-A partially supports a narrower hypothesis:** better delay representation can improve BER over the two newly retrained P2 neural models on train3/equal4/long6, but it still fails to beat DFT-8 and the adaptive classical threshold baseline on long6. It can degrade sharply on unseen long12 and sparse12; the matched LMMSE reference uses extra privileged test distribution information. A model which performs well on the training family does not automatically possess general structural delay invariance.

**P3-A is complete as an exploratory experiment, not a general communications breakthrough.** No P3-B/C/D/E experiment was started. Next hypotheses would require a separately frozen protocol and new data; do not retrofit new training distributions to these frozen comparisons.

## Reproduction

```bash
python -m pip install -r requirements.txt
python -m unittest test_p3_0_a -v
python p3_a.py --mode train --seeds 46001 46002 46003 46004 46005 --steps 600 --batch 64 --ckpt p3a_checkpoints --out p3a_results
python p3_a.py --mode eval --seeds 46001 46002 46003 46004 46005 --steps 600 --batch 64 --frames 768 --ckpt p3a_checkpoints --out p3a_results
python p3_a.py --mode verify --seed 46001 --frames 768 --ckpt p3a_checkpoints --out p3a_results
python p3_a_audit.py
python p3_a_make_report.py
```

Saved checkpoint binaries are excluded from public GitHub; they are included in the separate full archive. The published source, raw error counts, settings and checksums permit re-training. This is a **same-implementation**, not independent-codebase/hardware, replication.