# P2 final scope addendum — supplement to, not replacement of, P2-E

**2026-11-15 | completed under separate exploration protocol**

The original P2-E frozen confirmation (10 fresh training seeds, two held-out 10-dB endpoints equal4/long6, exact two-sided paired seed sign-flip and Holm correction with adjusted p=0.00390625 for both) remains unchanged. Its candidate, training, evidence, `P2E_FREEZE.json`, `P2E_REPORT.md`, and `P2_FINAL_AUDIT.md` have **not** been altered. The new P2-S experiments do **not** augment the original confirmatory sample size or significance tests.

## Original scope omissions and status

| Originally proposed | Before supplement | After supplement |
|---|---|---|
| P2-B: RMS-only/local RMS IQ control, isolating log-side channel | Missing: only raw / log phase / frame RMS + log | **Exploratory S1 complete**: six encodings under matched architectures; local/global RMS-only compared with matching log-scale arm |
| P2-D: different known-pilot densities and meaningful baselines | Missing: full 32-tone pilot only | **Exploratory S2 complete**: 32,16,8,4 active known pilot tones, periodically spaced, with mask-safe neural/LS/regularized DFT8/learned-prior LMMSE and test-prior privileged reference |
| Standalone mixed-channel training information route for classical and neural methods | Missing: only train3-trained models; test-matched LMMSE had privileged prior | **Exploratory S3 complete**: train3 vs four-profile mixture, 2 architectures, covariance estimated from the same 25,600 training pilot frames per seed/regime; held-out late3 and eight_tap |

Detailed protocol and data: [`P2_SUPPLEMENT_PROTOCOL.md`](P2_SUPPLEMENT_PROTOCOL.md), [`P2_SUPPLEMENT_REPORT.md`](P2_SUPPLEMENT_REPORT.md), raw per-seed counts in `p2_supplement/s{1,2,3}_results/`, code in `p2_supplement.py`, independent checks in `test_p2_supplement_independent.py`.

## Interpretation boundaries

The supplements had three new seeds each and 400 training steps per arm, whereas the frozen P2-E had ten distinct seeds and 600 steps. All supplement findings are **descriptive and exploratory**, with no significance threshold/confirmation claims. Training-data diversity improved some CNN tests, but harmed or did not improve others; plain classical estimators with suitable structural priors remained competitive or superior. Sparse pilot density significantly changes identifiability and total pilot energy under the fixed pilot-tone-power convention. Neither line covers real SDR data, sample-rate variation, a full RF chain, coded radio protocols, moving channels, hardware/energy measurement or independent physical replication.

The local protocol was committed as `682f868f7a8bdca70a83943b8a611723c3c9850c` before supplementary trials and SHA256 `1f7cc522d6f44a7a1102d68addab514a502fc31717a84a5a3d9d4a1019ae698a`; it was not externally timestamped until published with the results. All supplementary seed families and test distributions are disjoint from P2-E. P2-E original files were bytewise SHA256 verified after new experiments.

**Final status:** P2-A–E original completed; original planned omitted axes S1/S2/S3 now tested and documented; no universal neural superiority found, no new P3 initiated.
