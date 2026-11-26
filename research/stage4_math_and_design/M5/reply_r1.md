# M5 Round 1 · Decision for the Stage 4→6 cache

Carry a **16-bit canonical candidate cache with the exact 4-bit s17 prefix** into Round 2. Keep 12 bits as a cost ablation and continuous matched complex evidence plus residual energy as the physical oracle. This preserves the old reader and gives event and acquisition heads access to calibrated phase, radial evidence, and residual quality. The 24-bit format is specified for future comparison.

## Exact results

- For a specified observation law, the minimal shared statistic is the quotient of old IQ by equality of answer posteriors and the required candidate predictive kernels. Under matched, white complex Gaussian noise it reduces to the complex matched sums when variance is known. A rival with a private variance change also requires residual energy under the stated span conditions.
- Under log loss, the exact per-query compression penalty is $I(D_q;Y\mid Z,A_q)$. The finite-bit joint frontier is a partition optimization over one shared code. Its numerical values depend on the full fading, coherence, noise, and hypothesis laws. In the coherent noiseless example the thresholds are 2 bits for Q1 and 4 bits for both questions **per pair**. The noisy extension gives the exact Q2 penalty for a rotation-invariant Q1 cache at any specified pilot SNR and coherence lag.
- One-window EIG is the weighted Jensen–Shannon divergence between the rival predictive laws. Maximum EIG strictly improves expected posterior NLL over each baseline exactly when that baseline selects an inferior candidate with positive probability. This result assumes the true predictive law. Accuracy requires a separate argument, such as Gaussian Blackwell dominance.

## CPU evidence

The pinned Python 3.12, NumPy 2.3.5 and SciPy 1.17.0 replay loaded the actual s17 checkpoint, evaluated 16,384 old-task mothers and 8,192 separate blind mothers, and verified continuous $(T,E)$ likelihoods against raw IQ. Pooled NLLs are in nats.

| Cache | Q1 | Q2 | Event and jump location | Post-reveal, max EIG |
|---|---:|---:|---:|---:|
| 2 bits/window | 1.1453 | 0.8616 | 1.5396 | 0.4619 |
| s17, 4 bits/window | 0.9970 | 0.7966 | 1.2788 | 0.3677 |
| 8 bits/window | 0.9648 | 0.7856 | 1.0615 | 0.3614 |
| 12 bits/window | 0.9479 | 0.7536 | 0.9944 | 0.3545 |
| 16 bits/window | 0.9456 | 0.7493 | 0.9669 | 0.3544 |
| Continuous / Full IQ | 0.9438 | 0.7425 | 0.8934 | 0.3538 |

The 16-bit cache recovers **0.312 nat** of joint event/location NLL versus s17, with a pointwise paired 95% interval of **0.302–0.321**, and leaves **0.074 nat** to the continuous oracle. Conditional purely smooth-versus-jump NLL improves by **0.0104 nat** pooled and is effectively tied under selective fading. The actual s17 network merges two strict interior points whose four-label posteriors differ by **0.480 total variation**. The G1 Stage 3D result remains a narrow near-sufficiency case. Its physical cache gap is **0.0121 nat**, and the host's additional readout gap is **0.5041 nat**. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)

On the 2,048 ambiguous selective-fade mothers, **max EIG gives 90.63% accuracy / 0.2369 NLL**, random gives 71.63% / 0.5141, and nearest gives 70.21% / 0.5714. The committed forecasts imply 0.4542 nat expected gain and yield 0.4395 nat realized gain. Every max-EIG selection in that profile is time 4. A fixed time-4 policy would tie there. The synthetic run tests the value of reference geometry and leaves the incremental gain from recording-specific adaptation for a stronger design. Pluto/CTTC, host transfer and LEMMO performance remain live evaluation questions. [Full results](results_m5.json)

## Round 2

- **M1** freezes the wire format and a reusable likelihood interface. Keep physical modules shared across Qwen 4B/9B/27B and fit a host-specific adapter.
- **M3** preserves identity and time at pooling. Deduplicate repeated acquisitions and use time-indexed likelihood factors for cross-time evidence.
- **M4** measures phase, gain and residual calibration on capture-disjoint Pluto and CTTC tuples. Include DC/spur diagnostics if aggregate residual energy fails calibration.
- **M6** freezes noninferiority margins, actual storage and reveal budgets, and a fixed-best development control. Make competing candidates informative on different recordings before drawing conclusions about adaptive selection.

[Theorems and interface](THEOREM_M5.md) · [CPU replay](verify_m5.py) · [Raw results](results_m5.json)
