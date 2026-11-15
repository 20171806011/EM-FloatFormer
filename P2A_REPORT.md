# P2-A — Three small physical-residual estimators (completed, exploratory)

Protocol frozen in [`P2_PROTOCOL.md`](P2_PROTOCOL.md), hash `04d54fac6008426487d030208721fb5a9f98c03d44f5b2039cd3e69f101cf087`, before P2 training.

## Runs

Three new training seeds 1201/1202/1203, 600 AdamW steps, batch 64, original P1 synthetic 32-tone uncoded civilian QPSK/OFDM, train3 profile only. Each test cell evaluated on 768 new frames/seed (147,456 bits combined). Controls: newly retrained P1 CNN/log, pilot LS, DFT-8, train-prior LMMSE, and matched-test-prior LMMSE (privileged). Three learned residual designs: (1) pointwise MLP (9847 params), (2) circular CNN residual (35667 params), (3) delay-domain unfolded shrinkage (722 params). P1 CNN/log: 38018 params. No ground-truth H enters learned model forward path.

At 10 dB, pooled BER percent:

| Receiver | Train3 | OOD equal4 | OOD long6 |
|---|---:|---:|---:|
| P1 CNN/log, retrained | 5.497 | 6.554 | 8.301 |
| P2 pointwise residual | 8.252 | 8.205 | 8.067 |
| P2 local CNN residual | **5.429** | 7.609 | 8.787 |
| P2 delay-domain unfolded shrinkage | 7.060 | 7.452 | **7.314** |
| Pilot LS | 8.129 | 8.074 | 7.940 |
| DFT-8 fixed support | 5.511 | **5.356** | **5.309** |
| LMMSE train3 prior | **4.875** | 14.460 | 13.543 |
| LMMSE *matched test prior* | **4.875** | **4.942** | **5.069** |

- The delay-shrinkage candidate improves long6 relative to the P1 CNN and LS in aggregate, but **does not improve equal4** and its seed1202 performance on long6 is 8.148%, tied with LS, so no guaranteed robustness.
- P2 local CNN regression out of distribution suggests unconstrained correction can do harm; the model is trained through decoded-bit BCE and learns in a different function class from direct-logit P1 CNN. Different 600 vs P1 historical 750 steps, and no direct-logit head here; do not mistake this comparison for isolated attribution of a single bias.
- Even a simple fixed 8-tap DFT estimator outperforms all newly trained networks on both OOD profiles. It assumes the channel impulse-response support is within the first eight delays, which is true in this simulator and is additional useful engineering information relative to pilot-only LS. In this benchmark a reliable classical physical estimator is substantially stronger.
- Matched-prior LMMSE uses privileged profile prior; cannot claim equal-information superiority.

**Decision after P2-A:** no broad neural victory. Proceed with the prespecified representation ablation P2-B to investigate whether the local residual's failure is representational; test exact phase symmetry independently in P2-C. Do *not* add attention, MoE or diffusion merely to force improvement. The low-parameter unfolded estimator is the provisional physics candidate for future tests, subject to P2-D review.

Evidence: `p2a_results/ber_by_seed.csv`, `p2a_results/ber_summary.csv`, `p2a_results/run_config.json`, `p2a_results/latency_cpu.csv`, `p2_checkpoints/model_*_seed120*.pt` (local archive, excluded from lightweight public repo).

Limits: small 32-tone simulation with one all-subcarrier pilot and one data symbol, uncoded QPSK; not a 3GPP or SDR benchmark; only 3 exploratory seeds; no confirmatory test here.
