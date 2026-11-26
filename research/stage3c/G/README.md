# One model call per representative, with the quarter-turn law enforced

Use the Stage 3 frozen equivariant encoder `equivariant_b8_s17`. Its index has the form `4r+i`, with a quadrant r and an inner index i. These results assume the four independent, uniform transmitted-symbol hypotheses (at each independent draw) and circular noise of the simulator. For the repeat protocol the first symbol is sent again in window three. Public oscillator parameters can be arbitrary within that model.

## Exact reduction

Let (a,b) rotate the first and second data symbols by a and b quarter turns. Also rotate the repeated third measurement by a when that is the target. Keep the common channel, independently noisy pilots and their acquisition IDs fixed. This intervention preserves the joint experiment law. It takes the pair of encoder indices to `(4((r1+a)%4)+i, 4((r2+b)%4)+j)`. Because the transformation is invertible and the target rotates by a known number of classes, the entire conditional probability vector rotates by that same amount. Conditioning on continuous pilots defines this identity for almost every pilot value.

| Four-class target | Shift of the physical answer under (a,b) |
|---|---:|
| Q1 relative transmitted symbol | b−a |
| Q2 first transmitted symbol given a valid pilot | a |
| Q3 received sector on retransmission | a |

The fresh independent uniform third symbol has a uniform sector posterior. Q2 with no valid reference is uniform under the specified independent gauge. In checking a host, score these controls on its unmodified outputs as well.

Every pair of codes can be rotated uniquely to (i,j) with both quadrants zero. Train the candidate reader on this reduced input, shifting the training labels or probability targets back by the appropriate amount. At inference call the reader on (i,j) and rotate the vector back using the original quadrants. The public parameters and pilot do not change. For any choice of network parameters the resulting outputs obey the identities. Conversely every equivariant vector-valued readout can be reconstructed from its values on the 16 representatives. The class thus contains the matched Bayes rule. This is an architecture restriction with no Bayes risk cost under the stated symmetry. It does not prove that a particular trained reader recovers that rule. A canonicalized host alone cannot verbalize the original class without the inverse transformation. Report the external class permutation as part of the system.

This is a quarter turn of the individual transmitted labels. A change in the receiver's coordinate frame would rotate the pilots too and obeys different rules. Arbitrary continuous phase rotations after encoding are not covered. If the input contains symbol-dependent priors or side information, prove that those inputs transform appropriately before using the reduction.

## Which loss improvement is guaranteed?

For an existing reader f at D, one can instead evaluate all 16 rotated cache pairs, undo each output permutation and average the probabilities. Write P f for this average. For invariant sampling and a convex proper loss such as log loss or the Brier score, Jensen's inequality and invariance give risk(P f) ≤ risk(f). For the multiclass Brier score the reduction is exactly E||f−P f||². Indeed P is the group-average orthogonal projection in L² of the invariant design measure; the conditional target is fixed by P and the Brier excess above Bayes risk is squared distance to that target. This population statement need not hold in an unbalanced finite subset. Accuracy can deteriorate: in canonical coordinates let the same true class have probability .501 on fifteen orbit predictions and .001 on one, with the complement going to class 1. Fifteen of sixteen decisions are initially right; the mean favors class 1 everywhere.

Using only the representative of an existing host gives an equivariant rule in a single forward pass. Its proper-score improvement over that host is unguaranteed: one systematically poor representative can propagate its error over its entire orbit. A fair single-call comparison trains the reduced reader on the reduced task. In a short campaign retain the published host unchanged for mechanism probes; compute the 16-way projection offline after enumerating its raw responses. This provides a paired proper-score baseline and avoids attributing an imposed symmetry to the host.

## Evaluation cost with many independent mothers

For a frozen deterministic candidate scorer and a *fixed* prompt, context and public metadata, no-pilot Q1/Q3 probabilities depend on the mother only through the two integer codes. Evaluate at most 256 distinct inputs and gather scores by pair for every held-out mother. Sum losses on actual labels, preserving mother, seed and block IDs for paired intervals. For projected models the already-enumerated 256 rows suffice to compute all 16 rotations at each state; no extra GPU calls are needed. A trained canonical single-pass reader needs at most 16 distinct input evaluations per fixed question and condition. For example, 8,256 mothers require at most 256 input scores per frozen host arm for a no-pilot question at one fixed condition, before considering batching. That is a reduction of at least 32.25× in distinct input evaluations compared with evaluating each mother separately. This is not a measured wall-time speedup.

Key the tables separately by checkpoint, template, candidate strings/order, physical metadata, numerical precision and any other effective input. Verify scores on a small paired sample before caching, with deterministic scoring and fixed padding/position handling. Full-precision late pilot vectors, free chat histories and randomized decoding can distinguish mothers that share the same two codes. For Q2-valid/fresh use each actual reference or a separately justified sufficient statistic. Do not apply the no-pilot lookup to that endpoint.

## CPU check

Run from the repository root:

```sh
python research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
```

`quotient.py` implements the canonical transform and the one-call interface. The check exhausts 12,288 combinations of state, question and group element. It uses the published Q1/Q3 table, an artificial fixed-reference Q2 reader, and a deliberately biased scoring rule. The example risk calculation weights the cache pairs uniformly; the recorded values are a test of the identities, not a measurement of host performance or the population frequency of cache pairs.
