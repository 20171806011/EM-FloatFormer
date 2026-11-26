# M3 Round 1 | Executive summary

**C4 reproduced on all 512 archived mothers.** Work A and Qwen 3.5-4B metrics replayed exactly from the matched sums, cache codes, candidate log-likelihoods, and locked temperatures. Accuracy is in percent and NLL in nats.

| Q2 arm | Work A posterior | Qwen phase | M3 CPU surrogate |
|---|---:|---:|---:|
| none | 25.00 / 1.3863 | 24.80 / 1.3928 | 25.00 / 1.3863 |
| valid | 66.41 / 0.8192 | 28.78 / 1.3889 | 66.60 / 0.8193 |
| duplicate copy, dedup | 66.41 / 0.8192 | 28.78 / 1.3889 | 66.60 / 0.8193 |
| duplicate copy, raw | 66.41 / 0.8192 | 26.30 / 1.3944 | 66.60 / 0.8193 |
| independent only | 25.00 / 1.3863 | 24.80 / 1.4077 | 25.00 / 1.3863 |
| valid + independent | 66.41 / 0.8192 | 24.54 / 1.4082 | 66.60 / 0.8193 |
| fresh resample | 74.22 / 0.6632 | 26.43 / 1.4056 | 74.22 / 0.6629 |

The fresh pilot adds **7.81 pp** for Work A and loses **2.34 pp** for Qwen phase. Qwen zero also falls from 26.63% to 25.78%. The raw copy and distractor cost Qwen phase 2.47 pp and 4.23 pp. Appending the distractor moves its posterior by 0.04030 mean total variation and increases NLL by 0.01935 nats. The source confirms four extra learned vectors plus changed prompt text per appended reference. The scores establish sensitivity to irrelevant content and failure to use fresh evidence. Isolating the contributions of RoPE positions, DeltaNet state, and raw feature content requires a controlled frozen-host replay.

**One complex precision-weighted sum per physical clock.** The sum is minimal sufficient for the pilot likelihood at one clock, and two real coordinates are locally necessary for a regular C4 Bayesian readout. The fixed-prior minimal statistic is the target posterior. At different clocks, the single derotated sum is strictly insufficient for Q2 under positive Brownian diffusion. The report derives the Wiener likelihood ratio, a local \(O(\sigma_\phi^2\Delta t)\) coherent-collapse bound, and the joint Q2/Q3_after_pilot readout. Under the stated Q2 model, a fresh valid pilot strictly improves population Bayes log loss and MAP accuracy. The conditional continuous-phase Fisher information increases by twice the new pilot's matched precision. Those guarantees apply to the model-based decision.

**Fixed slot verified.** The differentiable CPU pool passed exact-zero tests for a raw duplicate, an appended distractor, distractor gradient, and all four joint rotations. The surrogate fits its first-window emission on 32,768 `NS_CFO_TRAIN` mothers and tests the untouched 512 blind C4 mothers. Its fresh-pilot gain is 7.62 pp and 0.15639 nats. The verifier exited 0 via the required `uv run` environment, and exercised a single evidence slot for every supplied count from 0 to 64. The Qwen results come from the archived scores.

**Two-clock Brownian test, \(t_p=(1,7)\).** Positive paired NLL differences favor full clock-indexed evidence. Parentheses give paired block SEs.

| \(\sigma_\phi\) | Mothers | Coherent sum minus full | Optimal posterior given sum minus full |
|---:|---:|---:|---:|
| 0.05, initial | 16,384 | −0.000023 (0.000101) | −0.000058 (0.000090) |
| 0.05, independent follow-up | 524,288 | +0.000105 (0.000018) | +0.000087 (0.000017) |
| 0.20 | 16,384 | +0.018603 (0.001661) | +0.014682 (0.001407) |
| 0.40 | 16,384 | +0.102467 (0.003959) | +0.075538 (0.003040) |

The first low-diffusion run was inconclusive and remains in the receipt. The larger independent follow-up resolves the small NLL effect. At \(\sigma_\phi=0.40\), two equal-sum observations changed a full-posterior class probability by up to 0.354 in the explicit witness.

## Round 2 interfaces

- **M1**. Can the connector take `canonical_q2[4]` and `canonical_q3[4]` in always-present slots, with an explicit \(a_1\) sector permutation outside the host? Keep the direct posterior available for model-based decisions. Use fixed prose and matched-position payload swaps to measure what the frozen host adds.
- **M2**. Can the gate require exact copy and distractor equalities on the numerical path, paired fresh-pilot gain and absolute NLL on the same blind blocks? For host tests, retain block/seed hierarchy and compare posterior and host decisions on matched tuples.
- **M4**. Can Pluto/CTTC expose stable acquisition IDs, calibrated timestamps and matched-I/Q noise variance, plus CFO and phase-diffusion estimates or uncertainty? Confirm which pilots are available before the target deadline and how channel identity and cross-pilot noise dependence are established.

[Derivations](THEOREM_M3.md) · [CPU verifier](verify_m3.py) · [Result receipt](results_m3.json)
