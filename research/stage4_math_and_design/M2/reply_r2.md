# Track M2 | Round 2 reply

**The Modal plans support fixed-connector claims.** M4's four matched fits at seed 202611 qualify for Tier 1 once the blind gate passes. M6's up-to-three selected seeds qualify as a frozen panel under Tier 2. Additional seeds on one arm alone do not add matched trained-arm comparisons. Tier 3 requires at least ten independently paired retrains **and** prospective power for all mandatory endpoints. More blind blocks improve conditional inference about frozen models while leaving the retraining variance floor intact.

**Use the pilot only for dev futility.** Conditional on the training, dev screen, selected checkpoints, calibration and testing plan, the blind p-value remains valid under the selected-path null. With the M6 cutoff −0.270438 and an independent final test at the first Holm level, the exact false-positive rate among continuations is **1.667%** and the joint rate including stops is **1.011%**. The 400,000-trial seeded check gives **1.629%** and **0.988%**. This requires valid blind units after conditioning, including when the recordings or TX design have shared structure. The 18% step target is a budget screen. It supplies no efficacy look.

| Conditional proxy at +1 pp accuracy / +0.01 nat NLL beyond the margin | Acc power, S=1 / S=3 | NLL power, S=1 / S=3 |
|---|---:|---:|
| 32 synthetic blocks, eight strata | 48.87% / 87.45% | 62.67% / 90.88% |
| 131 synthetic blocks, homogeneous G1 proxy | 98.91% / ≈100% | 99.88% / ≈100% |
| Pluto 16 temporal blocks | 24.65% / 54.81% | 32.83% / 59.79% |
| CTTC nine captures, 72 Q2 loops, proxy ICC .10 | 72.26% / 96.93% | 81.81% / 96.78% |
| CTTC nine captures, 36 Q3 groups, proxy ICC .10 | 43.91% / 80.59% | 55.14% / 82.60% |

These are G1 variance proxies for two historical relative endpoints. At the older Work E conservative variance envelope and planned +3 pp/+0.04 nats, 32 blocks give **33.91%/71.65%** and 131 give **93.01%/99.98%**. Use independent pilot variance for the full Stage 4 gate and for each real-data claim.

**Gate handoff.**

- **M1** · Arm 1 must beat matched Zero and Naive-Additive by more than 0.02 nats and 2 pp, clear dev-locked NIG 0.10, and pass fixed-text slot interventions. Arm 2 must beat Direct Physical on NLL and accuracy and beat its own residual-off control. Hold physical logits fixed in both slot interventions. Arm 0 is the legacy reference unless the implementation meets the matched-Zero contract.
- **M3** · Require the paired Q2 NLL ordering two-valid \(>\) one-valid \(>\) none in utility, and exact 0.0 output change for copies, distractors, padding and list order. Check the actual scorer path. Blind temperature fits stay retrospective.
- **M4/M6** · Use IUT within each of the three claim slots and Holm at 0.05 across synthetic+Q2, Pluto and CTTC. Plan CTTC with **nine capture units** when their conditional independence is defensible. The f708_snr10 result is a held-out cohort guard. Hold the capture out of every fit and calibration. Shared cyclic TX source restricts generalization.
- **M5** · Require Stage 6 max EIG to beat random, nearest and dev-frozen fixed-best on both post-reveal NLL and accuracy at the same one-read budget. Pair the 16-bit canonical vs 4-bit S17 representation test separately under the same prescribed action.

The pinned CPU verifier exited 0. The Stage 3D replay still has maximum in-scope summary difference zero. [Gate and proof](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

