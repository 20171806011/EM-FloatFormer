# P3-A2 Scope Addendum — separate, exploratory mechanism study

**2026-11-15 · Completed.** This is a follow-up to, not a revision of, frozen P3-0/P3-A, P2-E or P2 scope-supplement results.

## Why the follow-up was needed

The original P3 985-parameter delay-domain model retained two absolute-position inputs (sine/cosine of the tap index) and was allowed to suppress individual tap amplitudes almost to zero. A one-seed descriptive gate probe suggested late-tap suppression, but did not isolate whether location features, zero-floor shrinkage, or their interaction was responsible. The pre-frozen P3-A2 2×2 design tests both factors without altering the rest of the model.

## What has actually been evaluated

- Four paired small networks: position on/off × fixed retention floor 0/10%; no attention, diffusion, new training objectives, real radio, or active transmission.
- Five new training seeds 47101–47105, identical initial weights and stochastic training frames across all four arms per seed; **600 updates per arm**.
- Five held-out synthetic path profiles at 0/10/20 dB, 768 frames/seed/cell, sharing observations with LS, fixed DFT8, DFT12 (extended support reference), hard/soft noise-only DFT thresholds, train-prior LMMSE and privileged test-prior LMMSE.
- BER, complex-channel MSE, NMSE, post-hoc positive-scale-aligned MSE, per-tap gain distributions, true-path energy retention, CPU timing, original counts and full checkpoint hash validation.

## Key result and limits

In **sparse12 at 10dB**, BER fell from **12.291%** (A, original structure, freshly retrained) to **7.180%** (B, no-position), **8.763%** (C, 10% floor) and **6.549%** (D, both); the classical hard-threshold DFT still achieved **4.919%**. In **long12**, A was **11.495%**, B **8.412%**, C **9.614%**, D **7.757%** and hard-threshold DFT **7.031%**. On the original train3 distribution, D instead had a small BER regression relative to A (5.067% vs 4.929%).

The 2×2 factorial interaction is **positive** on average for both primary profiles, indicating overlapping rather than purely additive benefits. This is within the specific P3 architecture, synthetic generator and fixed training procedure: it does not establish a universal explanation of radio-channel generalization, statistical significance or an operational receiver. Five seed replications are exploratory; no retroactive updating of old confirmatory P2-E results.

## Integrity / location of evidence

Original protocol: [`P3_A_MECHANISM_PROTOCOL.md`](P3_A_MECHANISM_PROTOCOL.md), committed to GitHub **before P3-A2 training**, SHA256 `2955db97ee5203240cb76f30e068f0d12e4bf140909ecd8cdff9a4ae70aadc06`, commit `c52a9913baeb54b1ddc47c649301724913f3dc88`.

Source: `p3_a_mechanism.py` and `test_p3_a_mechanism.py`; independent record and checkpoint audit: `p3_a_mechanism_audit.py`; figures/report: `p3_a_mechanism_make_report.py`, [`P3_A_MECHANISM_REPORT.md`](P3_A_MECHANISM_REPORT.md); raw data and model-output checks in `p3a2_results/` and the public P3-A2 source/result ZIP. All 20 trained state dictionaries are preserved in the separate full local archive, not checked into public GitHub.

**Status:** The P3-A2 mechanism supplement is complete; P3-B/P3-C/P3-D/P3-E have not been started in this follow-up.

**Byte-exact freeze detail:** the original local protocol snapshot (LF ending) SHA256 is `2955db97ee5203240cb76f30e068f0d12e4bf140909ecd8cdff9a4ae70aadc06`; the pre-training GitHub blob (same content without final LF) SHA256 is `db3721cb5e3511f21d3ea4adafd92810becd2cd3ce0f375eb42057a2e94aab68`. The GitHub frozen protocol was not changed. The replay scripts now recognize these two exact byte variants without relaxing substantive protocol integrity.
