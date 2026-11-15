# P2-B — matched local-residual representations (completed, exploratory)

P2-A diagnosed failure of direct unconstrained residual CNN under long6; P2-B tests three floating-point input representations with **identical 35,667-parameter residual-CNN architecture**, same initial parameters per seed, same training batches, same loss/optimizer, and the same three exploratory seeds (1201–1203). 600 steps per model and 768 new test frames/seed/condition; all test scenarios are paired.

## 10 dB pooled BER (%)

| Representation | Train3 | Unseen equal4 | Unseen long6 |
|---|---:|---:|---:|
| raw IQ + magnitude | 5.429 | 7.609 | 8.787 |
| log-amplitude/phase | 5.505 | 7.025 | 8.283 |
| **dual path: RMS-normalized IQ + log RMS** | **5.118** | **6.318** | 8.411 |
| fixed 8-tap DFT | 5.511 | 5.356 | **5.309** |
| LS | 8.129 | 8.074 | 7.940 |

The dual path wins the three *local residual* comparisons on train3/equal4, but not on long6, where log/phase does better. Neither solves the six-tap failure, which is 0.471 BER percentage points worse than LS for the dual-path case. This is neither a proof of superior architecture nor of universal floating-point precision benefits. This ablation also does not control for a completely different train-time loss or wider hyperparameter search, and all results are exploratory.

## Mathematical and storage checks

For each frame, set `s=(mean_k |z_k|^2 + eps^2)^1/2` and encode `(Re z/s, Im z/s, log s)`; reconstruct as `s*(u_r+i u_i)`. With positive s this is algebraically invertible in exact real arithmetic. Across 4096 x 32 synthetic, log-uniform amplitude complex values in float64: median relative roundtrip error `4.25e-16`, maximum `1.17e-15`. After FP16 storage of three channels, median relative roundtrip error was `0.316%`, but **maximum relative error exceeded 100%**, due to tiny components under a frame RMS dominated by large values. All reconstructions stayed finite, while only 98.607% of raw IQ pairs remained finite under FP16 storage in this stress test. This verifies overflow avoidance **in this chosen input distribution**, not information-theoretic superiority, global high dynamic-range precision or robust RF decoding.

A uniform complex scaling of z by a>0 keeps normalized IQ approximately unchanged when epsilon is negligible, and shifts log s by log a. Physical complex addition remains linear in unnormalized IQ, not in normalized/log coordinates.

Evidence: `p2b_results/ber_by_seed.csv`, `p2b_results/ber_summary.csv`, `p2b_results/numeric_roundtrip.csv`, original `p2_numeric.py`; P2-B checkpoints are locally archived only.

**Decision:** retain dual encoding as a conditional representation candidate; move to separately verified mathematical U(1) phase-equivariance in P2-C and later frozen-weight P2-D, not to unverified architectural escalation. The strong classical DFT-8 baseline is essential and retained.
