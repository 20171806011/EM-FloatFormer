# Independent finite-channel audit

The exact oracle sums integer masses over the transmitted symbols, common rotation, receiver errors, pilot errors and each possible path of drift increments. It fits lookup tables directly from these masses. A separate simulator draws chronological samples and applies those tables. This directory depends only on the Python standard library and imports no other implementation from the study.

All states are elements of the additive group modulo four. The first lookup receives only the frozen cache and predicts the transmitted relative phase. The second lookup receives the same cache and the newly observed pilot and predicts the first transmitted symbol. Ties go to the smallest label. Neither lookup can read the raw pair again. Every first-symbol posterior before the pilot is checked for exact uniformity.

The tested cache choices are the observed difference (2 bits), the difference and the parity of the first observation (3 bits), and the two original discrete observations (4 bits). Each table entry is a **probability of a correct symbol answer** under the Bayes decoder for that specified cache.

| Condition | Relative phase, all three caches | First symbol, 2 bits | First symbol, 3 bits | First symbol, 4 bits |
| --- | ---: | ---: | ---: | ---: |
| Ideal shared pilot | 1 | .25 | .5 | 1 |
| Ideal data and independent uniform pilot | 1 | .25 | .25 | .25 |
| Unequal data noise, gap 0 | .7396 | .25 | .4404 | .8212 |
| Unequal data noise, gap 1 | .7396 | .25 | .417552 | .752656 |
| Unequal data noise, gap 4 | .7396 | .25 | .364182 | .592546 |
| One uniform drift increment, unequal data noise | .7396 | .25 | .25 | .25 |
| Uniform data and pilot errors | .25 | .25 | .25 | .25 |

The unequal-noise cases use data symbol error probabilities .07 and .21, pilot error probability .12 and per-step slip probability .09, with each nonzero offset equiprobable. A uniform increment uses .75. Across gaps the first lookup retains exactly the same success probability because it runs before pilot reveal. A stale uniform pilot can have the same one-sample marginal as a useful pilot. Their different dependence on the data phase produces markedly different follow-up risks.

The ideal model has exactly 64 equally weighted atoms. Seven simulation conditions used 60,000 independent frames apiece, sharing each draw across the three cache choices. Across the 42 comparisons between simulated and exact probabilities the largest absolute difference was .00420. Exact rational outputs and raw simulation counts are in [results.json](results.json). A fixed .012 simulation tolerance serves as a reproducibility check. All model-level equality checks use integer counts or fractions.

To reproduce in this directory, run `python3 check.py > reproduced.json` and `diff -u results.json reproduced.json` (Python 3.9 or newer).
