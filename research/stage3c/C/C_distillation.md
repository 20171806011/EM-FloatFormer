# C. When should a host connector learn from a numerical posterior?

Baseline: [Stage 3B, cbf3d2c0](https://github.com/20171806011/EM-FloatFormer/tree/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3b_math). The [inherited candidate loss](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3_ministral_cache/train_connector.py#L465-L482) uses the true class attached to each simulated mother. This study changes the training target only. The exported 3B teacher covers the fixed C4, s17, pre-pilot Q1 and repeat-Q3 tasks. Other offsets, SNRs and post-pilot questions require their own matched teacher or hard targets.

## Exact teacher

Let D denote all information available at inference, including cache, permissible metadata, question and revealed evidence. Let p(D) be the test-law conditional distribution of four answers. For one population objective, assume training and test have that same conditional law on the support in question. Let E be the one-hot true simulator answer, and rθ the connector-plus-host candidate distribution. Conditional expected hard-label cross-entropy equals H(p,rθ). The unrestricted optimal r is p; under model restrictions, the objective is the log-loss projection onto that model class. An exact teacher supplies p in training, preserving this population objective for *every* θ. A new support or a different task weighting may have a different optimum.

Fix θ. Write J for the Jacobian of four pre-softmax candidate scores with respect to θ (any frozen scoring temperature is included in J). The hard gradient is Jᵀ(r−E); the exact soft gradient is Jᵀ(r−p). By the law of total covariance, the covariance removed per training example is

    E_D [Jᵀ(diag p − p pᵀ)J],

a positive semidefinite matrix. Variability across inputs remains. This calculation concerns the gradient before clipping, with identical inputs and deterministic model states (or matched independent model randomness included when conditioning). The reduction does not by itself establish faster convergence, lower final error, a larger task-specific gain, or improved top-one accuracy for the trained network.

## Approximate teacher and an optimal finite-cell tradeoff

Condition on an estimated table q produced using independent simulation or cross-fitting. Write tα=(1−α)E+αq, with 0≤α≤1. The population target given D is mα=(1−α)p+αq. Its gradient bias is αJᵀ(p−q); conditional hard-label gradient variance is multiplied by (1−α)². Replacing E with the teacher's *winning label* loses the conditional probability information; it also substitutes a point mass where the teacher remains uncertain.

For n iid hard labels at a fixed D and fixed θ/J, define

    V = tr[Jᵀ(diag p − p pᵀ)J],   B = ||Jᵀ(q−p)||².

The squared Euclidean error of the averaged gradient, in expectation relative to the true population gradient at D, is (1−α)²V/n + α²B. The centered label noise has mean zero so the cross term vanishes. The minimizer is α*=V/(V+nB) if the denominator is positive. If B=V=0 all mixtures tie. This weight optimizes a local gradient-MSE criterion for the given number of independent labels and the chosen parameter coordinates. It is neither an identified constant for the real teacher nor a universal weight for accuracy. Rotations of one mother cannot increase this n as if they were independent labels.

If independent validation bounds V≤v and B≤b, α=v/(v+nb) minimizes the *upper bound* (1−α)²v/n+α²b (nonzero denominator). One distribution-free choice when ||J||op≤L and ||q−p||₁≤ε is v=L² and b=L²ε²; this can be conservative. Shared Monte Carlo integration error in q persists when many training records reuse the same approximate table. Fitting q using their own answers voids the independence assumption. Agreement between independent integrations is a numerical diagnostic, not a certified bound on B.

A way to estimate B for repeated observations of the same D is to split independent labels into two means e₁,e₂. Conditional on q and J fixed independently of both halves, the scalar product of Jᵀ(q−e₁) and Jᵀ(q−e₂) has expectation B. It can be negative in finite samples. For a real neural connector the Jacobian and code counts change across the run. A practical alternative is to fix a small grid of mixture weights in advance, select on hard-label development NLL and report the frozen choice on new mothers together with hard and pure-teacher endpoints. This compares actual learning objectives while avoiding a claim of oracle-weight attainment.

## Calibration, classification and a failure case

For an unrestricted fit of mα, test excess log loss is KL(p||mα) ≤ α KL(p||q) by convexity (finite RHS when q is positive on the support of p). The true winner remains optimal if the gap to the runner-up exceeds 2α||q−p||∞. Both statements refer to the population target. The learned model has its own fitting and optimization errors. There is no implication that lower gradient MSE must improve accuracy.

Consider the true four-class distribution (0.48,0.42,0.07,0.03) and biased teacher (0.24,0.64,0.08,0.04). The true winner changes from class 0 to 1 in the mixture when α>3/23 ≈0.130435. Pure teacher training with an unrestricted learner loses 6 percentage points of population accuracy and 0.137823 nats in log loss. Direct differentiation gives weights above this threshold for both n=8 and n=32 in logit coordinates (J=I).

We also enumerated every possible training count vector for n=8 and n=32, fixing the cell and its true distribution. We fitted the convex combination of empirical frequencies and the numerical teacher. The Brier excess risk is exactly the squared distance to p. The following values average over all training histograms, and score the predictions against a fresh draw from p.

| n and objective | Hard target | Mixture at oracle α* | Teacher alone |
|---|---:|---:|---:|
| 8, excess Brier | 0.073425 | 0.043411 | 0.106200 |
| 8, accuracy regret | 0.021757 | 0.046741 | 0.060000 |
| 32, excess Brier | 0.018356 | 0.015651 | 0.106200 |
| 32, accuracy regret | 0.019533 | 0.032490 | 0.060000 |

The example verifies metric dependence even with an oracle blend weight. An exact teacher and a closer teacher are included as controls in [the complete output](results.json). Run `python check.py` from this directory. The standard-library enumeration verifies the formula for every reported mixture using 165 and 6,545 multinomial count vectors. All rows are finite-cell calculations; they contain no Qwen or Ministral measurements.

## Paired connector trial and estimands

Use exactly the same permissible student inputs for every arm. Never give the teacher vector, its winner, or the answer at inference. Teacher probabilities are targets, as in a supervised loss. Share frozen encoder, initialization within a host/seed, selected minibatch order, task sampling weights, optimization budget, public metadata, prompt and choice encoding, and all holdout mothers. Independent seeds across training, teacher integration/calibration, development and evaluation are needed. Balance and tag rotated views by original mother. Tune calibration temperatures by the same validation procedure for every arm and report pre- and post-calibration scores.

| Arm | Training on teacher-supported items | What the paired contrast estimates |
|---|---|---|
| H | True hard answer | Baseline candidate NLL and accuracy for this optimizer, budget and host |
| S | Full numerical q | S−H estimates the change caused by replacing hard supervision with this computed teacher; includes both variance removal and teacher bias |
| M | (1−α)E+αq | M−H gives the end-to-end effect of the preselected blending procedure; M−S compares adding label correction to pure distillation |
| W (diagnostic) | One-hot teacher argmax | W−S measures the consequence of discarding the teacher's non-winning probabilities, holding teacher source fixed |

For configurations lacking a validated q, retain hard supervision identically in all arms and publish results separately. A calibration table fitted at an evaluation frequency uses that frequency's physics; tag whether the angle was seen during student training, available only to the analytic baseline, or held out entirely from learning. Run paired comparisons separately per task and frequency, including the zero-drift control and sign reversal. Use independent mother blocks for uncertainty and multiple paired training seeds to measure training variability. Score candidate accuracy and NLL against independent hard labels. Optional disagreement fields can compare to q; evaluating solely on that teacher would favor its reproduction by construction. A separate experiment supplying q to the host at test time evaluates verbalization and should be labeled accordingly.
