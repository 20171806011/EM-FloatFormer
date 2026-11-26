# Prospective sample size for the next host campaign

Base: `cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0`. This document specifies a future experiment. The executed work is the planning calculation and the support-grid checks.

## One primary comparison

Within frozen Qwen3.5-4B compare matched CFO-supervised connectors with real phasors versus zeros in the same numeric input slots. Both arms receive identical text metadata, same 8-bit encoder, budgets, draws and per-seed initializations. Use the Q3 repeated-symbol task, evaluated before revealing a pilot. The primary condition is the outside-rate/outside-angle cell below at both signs, 2 dB and Brownian σ=0.1. Average the two signs and the three predefined rotations equally; never treat these rows as independent captures.

Predeclare gain as control NLL minus phase-arm NLL (nats), and phase-arm accuracy minus control accuracy. A practical win requires both lower endpoints of two-sided 95% paired intervals to exceed 0.02 nats and 2 percentage points, respectively. This intersection-union rule controls a one-sided false-positive probability of 0.025 for the joint claim under valid single-endpoint intervals. A pass on one outcome only is reported descriptively. This campaign does not infer benefits across arbitrary pretrained models.

Ministral can run the same comparison on the same draws as a named replication. Report the paired difference of the two improvement estimates and its interval before interpreting a host interaction. The design is powered for the Qwen within-host comparison; the cross-host interaction has its own variance. The historical no-CFO-trained checkpoint and its physics/prior probes are separate frozen-weight contrasts.

## What a sample buys

Use separate independently seeded 64-mother balanced blocks; share primitives across both arms, signs, rotations, timing cells and hosts for each source mother. The individual P(correct) difference on a row is D ∈ {−1,0,1}. Write q=P(only one arm correct), δ=E[D]. Then Var(D)=q−δ². For the per-mother mean M of any number of dependent paired rows, Jensen gives E[M²] ≤ mean(q), so q provides a conservative variance even with shared rotations. Reusing measurements increases precision only to the extent that the aggregated contrasts vary less.

For block size m=64 suppose the within-block average covariance of mother contrasts is bounded by ρ times the marginal variance bound v. Then Var(block mean) ≤ v(1+63ρ)/64; use v=q for hit gain and v=s² for mother-averaged NLL gain. Actual balanced-block dependence may differ from this working assumption. Independent block means G_b are the analysis units. With B blocks, report the empirical mean and t interval using sd(G_b)/√B, accompanied by a bootstrap that resamples whole blocks with all descendants. For fixed seed-averaged estimands, average paired per-seed contrasts inside each source block first.

The standard-library calculator uses the conservative normal planning formula

    B = ceil[var(block mean) × max{((z.975 + z.95)/(δ−margin))², (z.975/target half-width)²}]

and takes the maximum across endpoints with a 32-block floor. Targets are a true gain of 0.05 / 0.06, practical margins of 0.02 / 0.02 and maximum 95% half-widths of 0.02 / 0.03 for accuracy/NLL. At q≤0.35, ρ≤0.01 and s≤0.50, the design selects **129 blocks = 8,256 source mothers**. Approximate expected half-widths are 1.63 pp and 0.0138 nats. Prospective marginal powers under the stated alternative are 95.04% and 99.99%; the union bound gives at least 90% if each reaches 95%. These are model-based planning approximations, not guaranteed coverage or measured host gains. Under q=0.50 and ρ=0.03 the same design rule selects 326 blocks. The previously proposed 512 mothers represent eight balanced blocks and are suitable here for exploratory cells.

Run `python design.py` to reproduce [the machine-readable output](results.json). If an independent developmental pilot is available, calculate both paired block-mean SDs on a sacrificial design grid disjoint from the final test seeds. Supply conservative upper estimates with `--pilot-block-sd HIT NLL`; the calculator takes the maximum of each pilot variance and its planning variance. Restrict this adaptation to development, freeze the resulting B before reading the holdout, and report the achieved width whatever it is. For a still-sparse pilot, retain the conservative sensitivity case as a cost scenario. A smaller completed run can report estimates and attained intervals without a powered-success declaration.

## Generalization and allocation

Use the Stage 3B F matched-supervision train support: |ω| ∈ {0,0.10,0.25}, total Q3 lags in {4,5} and therefore training angle magnitudes {0,0.4,0.5,1,1.25}. All angles here are in radians and below π. This 2×2 holdout changes whether the *rate* and the *effective Q3 rotation* are inside or beyond their respective training ranges independently.

| Absolute rate | Absolute Q3 angle | h | ell | tₚ | Generalization cell |
|---:|---:|---:|---:|---:|---|
| 0.20 | 0.8 | 2 | 2 | 4 | Inside both ranges, unseen combination |
| 0.20 | 1.6 | 4 | 4 | 8 | Inside rate, outside angle |
| 0.40 | 0.8 | 1 | 1 | 2 | Outside rate, inside angle |
| 0.40 | 1.6 | 2 | 2 | 4 | Outside both, primary cell |

Evaluate ± rates. Use all B independent blocks in the primary cell, eight predetermined blocks (the first eight under the locked generation order) for each exploratory cell; the same eight blocks receive all four settings and the signed/invariance probes. Additional rows change the evaluation count, not the source-mother count. For one connector seed pair the primary uses 8,256 × 2 signs × 3 rotations × 2 arms = 99,072 Q3 candidate-scoring calls interpreted as observation–arm rows before batching, each with four continuations. Separate same-seed negative controls and failed invariance rates are descriptive; upgrading them to confirmatory endpoints requires their own margin and variance estimate. Examine rotation/hold as pointwise deviations and report the fraction of source blocks/mothers failing a prespecified tolerance. For the independent-X3 uniform control score probabilistic loss since top-one ties can move accuracy without changing information.

Different durations also change accumulated Brownian variance σ²t. Equal ωt therefore need not give equal posterior or accuracy. Compare each arm against its own matched same-cache physics reference. An auxiliary time-scaling check may hold σ²t and all time ratios fixed; label it separately. Only the Q3 product appears in this grid; Q1 and revealed-pilot Q2 have different time dependencies.

### Optional predeclared invariance reliability claim

In the primary cell, independently preselect one uniformly random sentinel mother per block before model scoring. Evaluate the fixed phase-feature system on the same rotations already scored: Q3-repeat probabilities should hold under rotation (0,1) and roll by one class under (1,1). Mark one failure per sentinel if any class probability differs from its required transform by more than the historical absolute tolerance 0.20, for either sign or any of the three prespecified connector seeds. This tests conditional reliability at that tolerance; an exact symmetry claim would require a mathematical proof. The 129 independent sentinel indicators admit an exact binomial test of failure probability ≥5% against <5%. At one-sided α=.05, up to two failures pass. Zero failures give an exact upper bound of 2.30%; zero among eight blocks would give 31.23%. Fifty-nine independent sentinels with zero failures are sufficient for a 5% ceiling. Use this as a separately labelled endpoint with its own claim, or require it jointly with both gains for an all-criteria declaration; the gain power calculation covers only the two gains. All-mother descriptive rates and cluster intervals can also be reported. Checkpoints are held fixed for this inference.

## Lock and reporting

Freeze full weight revisions, encoder, host-template and tokenizer versions, seed identifiers, scorer and probability normalization, trial budget, generator, hyperparameter choices, primary grid and analysis scripts prior to test. Draw a new test namespace unused by Stage 3/3B or model selection, independent from training and development. Record a hash of its seed manifest before holdout access; separately seeded blocks can use `SeedSequence([campaign_test_seed,block_id])`. Exclude scenario IDs from model input or keep the same public labels across arms. The Stage 3B +0.35 C4 anchor can be reported separately from the new grid.

Use three fixed paired connector training seeds for a seed-averaged result and show all three gains. Reuse the same mothers across seeds; repeating checkpoints or changing rotations creates no new test blocks. Waveform-block intervals condition on the pretrained checkpoints and the three fitted connectors. Report seed dispersion separately, including the range. Three seeds provide little support for population inference over retraining randomness. One pretrained checkpoint per host permits only checkpoint-specific replication, independent of the number of generated waveforms.
