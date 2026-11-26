# C. Optimal four-candidate readout and faithful language output

## Setup and scope

Let `D=(Z,M,A,q)` denote exactly what a receiver may use when answering. Here `Z` is the frozen pair of window indices (8 bits total), `M` the public condition and timings, `A` the pilot observations actually revealed by that turn, and `q` the question and protocol. Let `Y` be the corresponding one of four physical labels. All expectations are under the specified test law. For a frozen host plus connector, `r(D)` is its normalized four-candidate score vector. First consider arbitrary four-probability readouts, then identify the additional restrictions imposed by the host. These results apply independently at each reveal stage.

## 1. What would optimize each metric?

Write `p_i(D)=Pr(Y=i | D)`. At fixed D, choosing label j succeeds with probability p_j. Therefore any maximizer of p gives the exact optimal accuracy, and the overall ceiling is `E max_i p_i`. For natural-log loss the Bayes risk is `H(Y|D)`. Indeed,

    E[-log r_Y(D)] - H(Y|D) = E KL(p(D) || r(D)).

This follows by conditioning on D and subtracting the two four-term sums. A zero score on a positive-probability label incurs infinite loss. For the multiclass Brier sum over four coordinates, expanding the square gives excess risk `E ||r-p||_2^2`. Thus a matched, exact posterior followed by argmax is optimal for hard decisions, and transmitting its full four entries achieves both proper-score optima. A deterministic renderer can turn the selected class ID into its canonical Chinese label losslessly. This is a constructive reference interface, with the physical computation taking place in the posterior solver.

For the actual candidate-scoring path, let L_i be the sum of autoregressive log probabilities of the tokens in candidate i. The code computes `r=softmax(L/T)` for T>0. If p_i>0, pointwise equality r=p requires and is equivalent to `L_i=T log p_i+c(D)` for all i. Choosing one global temperature preserves every argmax and only rescales log-odds. It cannot repair wrong rankings. Attainability of those scores by a frozen host plus an approximately 1.99M-parameter connector is an independent model-class question. With estimated posterior p-hat, its score is an achieved reference. Claim a Bayes ceiling only when p is known exactly or its approximation error is bounded.

An error bound for approximate readout follows with a=argmax p and b=argmax r:

    p_a-p_b <= |p_a-r_a|+|p_b-r_b| <= ||p-r||_1   (a != b).

For equal labels the regret is zero. Pinsker's inequality (natural logs) and Jensen then bound expected accuracy regret by `sqrt(2 E KL(p||r))`. Score improvements can coexist with unchanged accuracy when the winning class stays fixed.

## 2. Which information does a shortcut discard?

If a system exposes only S=f(D), its best posterior is `p_S=E[p(D)|S]`. Conditioning the log-loss identity twice gives

    E[-log r_Y(S)] - H(Y|D)
      = I(Y;D|S) + E KL(p_S || r(S)).

This separates information lost by the bottleneck from error in reading it. Jensen's inequality gives `E max_i p_i(D) >= E max_i (p_S)_i`; equality holds exactly when, within almost every S fiber, one class is a maximizer almost surely (ties allowed). A current MAP-label cache therefore preserves that single turn's best accuracy, while losing confidence and potentially harming later pilot updates. A marginally calibrated constant class-frequency vector can have poor discrimination.

With latent branch B, the matched posterior is the mixture `sum_b Pr(B=b|D) Pr(Y|B=b,D)`. Hard branch selection can lose even the winning class. Take branch weights (0.51,0.49) and label distributions (0.40,0.35,0.15,0.10) and (0.05,0.80,0.10,0.05). The more likely branch selects class 0. The mixture selects class 1; their conditional accuracies are 0.2285 and 0.5705. Hard selection ties the mixture on accuracy if its chosen label also maximizes the mixture. Other branches may still matter for probability scores.

A sequential counterexample uses two equally likely observations with binary posteriors (0.60,0.40) and (0.90,0.10) (embed both into four classes with two zeros). The current winning label is 0 in both. A later pilot event has likelihoods (0.20,0.60) conditional on the two possible labels and is conditionally independent of the original observation given the label. Its likelihood ratio of 3 flips only the first case to class 1. A receiver storing just the old winner cannot distinguish them. This example isolates the loss from throwing away confidence; in the actual shared-phase simulator, even an old symbol posterior need not suffice for arbitrary future pilots. The retained statistic must preserve the relevant joint law with the pilot.

## 3. Candidate choice and Chinese answer measure different events

Let b be the top candidate class. Let g be the class parsed from unconstrained generated text, with unparseable outputs counted as wrong. For paired records with identical truths,

    Acc(g) - Acc(b) = Pr(g correct, b wrong) - Pr(g wrong, b correct).

The difference can have either sign. Report both transitions, unparseable rate, candidate/generation agreement, and physical-label accuracy; agreement alone cannot detect a shared wrong label. If decoded text is conditionally randomized given D, its accuracy is the expectation of a convex combination of the p_i and cannot exceed the matched Bayes ceiling. Independently scoring explanations requires its own ground truth. Label agreement by itself tests only verbalization.

In the archived run, the four strings enter the candidate scorer as separate continuations. The free generator is asked to pick among options without displaying those strings, and the demo selects a mother on which the analytic pilot helps if possible. A showcased failure establishes existence, not its prevalence. For a matched follow-up, use the same displayed options and prompt for the scorer and free decoder, freeze the parse rule and invalid-answer handling on development data, and evaluate every selected blind mother. Keep unrestricted output and deterministic rendering as separate arms. Also replay the original prompt on the same mothers to measure the prompt change. Compare normalized sequence-likelihood scores within each host; report tokenizer-dependent candidate lengths, temperature, and option order so wording effects can be investigated separately from physics. A fixed, balanced permutation of option labels per mother is a diagnostic for answer-ID bias; consistently remap both predictions and truths.

For the CFO study, compare four arms on identical mothers: an exact or error-controlled same-information posterior when available; the archived calibrated lookup readout; host candidate scores; and parsed free answers. Evaluate Q3 repeat and Q2 valid/fresh separately with accuracy and log loss for probabilistic arms, and cluster paired differences by mother across rotations. Preserve text-only and shuffled-cache controls. A host supplied the reference posterior directly is an additional test of rendering; it cannot establish that the connector learned to compute the physics. Numerical phase features derived entirely from public M change representational convenience, with no change to the unrestricted Bayes ceiling. For example if s=f(M), conditioning on (D,s) gives exactly p(D).

## 4. Training on zero systematic drift leaves its direction unidentified

Consider two softmax readouts with logits `v(D0) + sin(angle) * u` and `v(D0) - sin(angle) * u` for the same vector u and all remaining features D0. If every training case has zero deterministic CFO, both make identical predictions on the entire training support. At angle pi/2, selecting v=0 and u=(1,0,-1,0) makes their predictions favor classes 0 and 2 respectively. Hence empirical risk on that support cannot identify even the sign of the deterministic response for a model class containing both readouts. Stochastic phase diffusion with zero deterministic drift leaves the supplied CFO feature constant. Separate zero-shot tests of a fixed physical rotation from representation tests in which text and phasor variants both train on the same range of nonzero CFO, with fresh held-out rates.

Primary implementation at commit `69fbd29a`: [candidate scoring](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/ministral_connector.py#L410-L491), [supervised objective](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/train_connector.py#L475-L482), [prompt assembly and case selection](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/chat_demo.py#L31-L78).

Run `python check_c.py` in this directory. The script enumerates finite probability tables and transition counts with standard Python. It performs no host inference.
