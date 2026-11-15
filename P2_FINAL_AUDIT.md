# Phase P2 closeout audit — 2026-11-15

P2-A, P2-B, P2-C, P2-D, P2-E sequential completion: the initially frozen project protocol, separate phase reports, P2-E selection freeze, CSV integer-error evidence, saved checkpoint hashes, model replays, mathematically independent LMMSE expression, numerical symmetry tests, CPU latency and 10-fresh-seed results are retained.

## Confirmatory decision

**One** candidate (`res_equiv_local`) vs **one** control (newly retrained `p1_cnn_log`), **two** frozen endpoints (equal4/long6, 10 dB). Both reported corrected two-sided exact p values = 0.00390625, favoring P2 on synthetic channels. In particular the 2,060-parameter complex FIR has lower BER than the 38,018-parameter baseline CNN under the specific held-out profiles. No matched-parameter controlled experiment separates the effects of architecture and capacity. The 8-tap DFT baseline remains more accurate, so no assertion of overall superiority or broad novelty is justified. Candidate selection followed exploratory P2-D, hence the final confirmatory claim is restricted to the frozen comparisons, not to all unseen real-world conditions.

## Evidence verification

- P2-E source train seeds 2201..2210 are disjoint from exploratory 1201..1203.
- 540 unique per-seed/profile/SNR/method entries, each 131,072 bits, 3 profiles × 3 SNR × 10 seeds × 6 methods.
- Checkpoint replay: 18/18 original error-count/bit-count cells match seed2201 on 10-dB train3/equal4/long6.
- Phase equivariance: trained estimate max complex error 2.4e-7, decoded logit max error 1.91e-6, 0 hard decision changes in tested common rotations.
- Matched LMMSE matrix posterior vs independently expressed delay-domain Wiener shrinkage: maximum numeric difference 3.41e-6.
- 22 unit tests from P1 and P2 passed in local Python 3.13 / PyTorch 2.10 CPU environment.
- Explicit limitations: not independent scientific implementation, not hardware, 32 tones, one full known pilot per payload, static within pair, uncoded QPSK, ideal synchronization/noise; comparison on narrow synthetic profiles. CPU timing excludes full receiver I/O; do not present any of these numbers as system latency.

No P3 experiment initiated. New features are not claimed to provide electronic warfare, active interference, real radar, or hardware transmission capabilities.
