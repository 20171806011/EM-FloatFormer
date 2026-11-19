# Observation and reliability consistency: prospective mechanism study

Base: `bb9396f08132bb5f586ff728dcfa4e7872201c7b` (soft-information research branch).
Published P6 main remains `63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`.
Historical P4 reference is `8380084c6dece64119ab72f9c3c1c212e154fb17`.
This is an inference-only follow-up, not a new trained stage or a promotion of defaults.

## Ordered question

1. Separate missing **receive equations** from known **transmitted pilots**.
2. Measure whether the supplied soft means and complex variances describe their actual errors.
3. Use one effective frequency propagation model for the expected amplitude fit and the final Gaussian cavity likelihood.

Engineering uses only the previously reused development checkpoints/frames 901–903.
Run the four-arm observation-by-variance comparison first, inspect calibration,
then run the full mechanism chain. Freeze critical code, these hypotheses,
seeds, weights and score rules before any new fresh scoring. This is a
within-session prospective freeze, not public independent preregistration.

## Observation model and information access

Keep the five archived interventions: intact, first pilot row multiplied by
0.4, first pilot row erased, row 9 erased, and half the scattered pilots erased.
The archived corruption multiplies **signal and noise together**, after
simulation. For known positive gain, invert it and retain the original noise
variance. For gain zero, delete that receive-frequency likelihood equation.
Do not delete the corresponding transmit pilot: its mean is the known pilot,
its variance is zero, and codeword/payload layout is unchanged.

All `aware_*` methods receive the exact corruption-gain map. Legacy methods
do not receive it. This is privileged diagnostic/control metadata, not a
deployable detector, not a mathematical performance upper bound, and not
evidence of superiority over a blind receiver. Within-aware comparisons have
the same metadata. For first_attenuated, every aware arm must reproduce its
intact output within floating tolerance and retain the same hard decisions.

Observation-aware initialization uses only observed pilot subdictionaries,
with retained-tone fraction as row weight. Gauge uses observed pilots. Fisher
reliability retains the full **transmit** pilot aperture as its nominal
reference. Missing neural observation features are imputed by the current
diagonal mean and have zero innovation; this is a frozen-head input heuristic,
not an added measurement. Imputed features never enter likelihoods. The old
no-observed-nonzero-pilot output guard is retained; it is conservative and is
not a theorem that ICI cannot convey pilot information through other bins.

## Seven fixed arms

| Method | Known receive map | Variance in amplitude fit | Fit includes learned diagonal residual | Soft/final likelihood |
|---|---|---|---|---|
| legacy_mean | No | No | No | Original scalar |
| legacy_variance | No | Yes | No | Original scalar |
| aware_mean | Yes | No | No | Original scalar |
| aware_variance | Yes | Yes | No | Original scalar |
| aware_effective | Yes | Yes | Yes, including variance RHS cross-term | Original scalar |
| aware_cavity_mean | Yes | No | Yes | Full-frequency Gaussian cavity |
| aware_consistent | Yes | Yes | Yes | Full-frequency Gaussian cavity |

All arms share each archived parity checkpoint, three passes, original Trust
phase rule and learned weights. No profile-Hessian phase update, training,
calibration-temperature fit, new head, or post-fresh arm selection is allowed.
The cavity mean-only arm is a diagnostic ablation, not a claim of correct EM.

The cavity arms approximate other symbols as proper independent complex
Gaussians. Code priors use only check-node extrinsic messages from the prior
pass's channel LLRs. The message is captured before that pass's amplitude
update (after the special step-zero re-demap), so it has one outer-pass lag.
Re-demaps within a pass reuse one prior; final readout uses the last prior and
latest effective operator. Own-symbol priors are not added to the final
channel LLR. The original max-log readout is retained. This changes both the
interference likelihood and prior schedule, so its gain cannot be attributed
solely to a scalar variance correction. Loopy updates reuse the observation;
no exact joint-posterior or statistically independent-message claim is made.

## Fresh cohort and endpoints

R2 only: 20 OFDM symbols, 32 subcarriers, one receive antenna, three nominal
rays, 16-QAM, SNR 9 dB, public rate-1/2 128-bit LDPC code. Checkpoints 6–15.
For cluster i=1..10 and batch j=0..3 use seed `814611190000 + 1000*i + j`, B8.
All seven methods and five conditions score the same 320 underlying bursts.
Each method/condition has 5,760 codewords and 737,280 coded bits. Conditions
share waveforms and are not independent cohorts. No new channel family is
tested; archived weights are reused. No classical receiver is added to this
mechanism-only study, so it establishes no classical-baseline ranking.

Save every batch's integer coded-bit, decoded-information-bit and block error
counts and denominators, CFR squared error/energy, source/checkpoint identity,
simulator observation/payload hashes, solver diagnostics and calibration.

Two declared primary tests, both on scattered_erased:

1. Observation-by-variance interaction, per cluster:
   `(legacy_variance - legacy_mean) - (aware_variance - aware_mean)` block errors.
   Positive means observation handling reduces the variance penalty.
2. Consistency addition: `aware_variance - aware_consistent` block errors.
   Positive means the consistent receiver improves at the same metadata access.

Use two-sided exact enumeration of all 1,024 paired sign assignments and Holm
correction over these two tests. The exchangeability/symmetry assumption is
explicit. Ten checkpoint/waveform clusters, not 5,760 individual codewords,
are the inferential units. Report all ten differences, wins/ties/losses, mean
percentage-point effect and approximate unadjusted pointwise t95 interval.
Other conditions, BER, information BER, NMSE, BCE, bins, and ablations are
descriptive. A favorable interaction does not itself mean a good receiver;
failure to reject is not equivalence. Do not alter the test family after results.

## Soft-information calibration

Record the actual three mean/variance pairs used by the physical amplitude
updates, including the re-demap before the first update. Ground truth enters
the evaluation function only. Exclude all transmit pilots. Use fixed variance
bins `[0,.01), [.01,.05), [.05,.1), [.1,.2), [.2,.4), [.4,.8), [.8,inf)`.
Keep count, sum predicted complex variance, and sum actual squared mean error
per bin/step/batch. Report their aggregate ratio descriptively. It is not a
proper scoring rule, a proof of posterior correctness, or a causal mediation
test. Also save uncapped stable positive-one logistic BCE of final channel
LLRs. Numerical saturation used only for reported zero noise is not applied
to normal-SNR BCE. Do not fit temperatures to fresh labels.

## Validation and precision

Independent explicit-DFT, finite-QAM-enumeration and augmented-LS checks
validate weighted moments and learned-residual cross terms. Explicit
leave-one-out Gaussian inverses validate cavity equations; own-symbol prior
cancellation is tested. Integration checks cover legacy equivalence, receive
gain inversion, receive-mask invariance, known TX pilots, empty observations,
truth-free inputs, immutable weights and C* transformations.

Full numerical audit: five aware methods, three development checkpoints,
five conditions, 27 gains (1e-35, powers 2^-12 through 2^12, 1e20), B1,
independent deterministic phases. Require finite outputs, no payload hard
flips, maximum grid LLR difference <1e-12 and relative CFR L2 <1e-12. Record
every failure without relaxing the gate. Float64/complex128 CPU only; the
external archived LDPC decoder internally uses float32. Cavity computational
noise/variance floor events are counted and reported; zero-noise robustness
does not imply the exact noiseless Bayesian limit.

Time receiver inference alone on one CPU thread, B1 and B8, two warmups and
ten cyclically interleaved measurements, with no other study jobs. Exclude
simulation, calibration scoring and final external LDPC decode. Use actual
legacy receivers for timing rather than the moment-tracing adapter. Preserve
raw timings and report scope and overhead honestly. No accelerator claim.

An independent agent reviews formulas and result aggregation, verifies hashes
and replays fresh clusters 1 and 10 in this same environment. This is same-
environment independent audit/re-execution, not external independent
reproduction. Preserve all inherited directories and publish this follow-up
as a separate stacked research branch/draft pull request, including negative
and mixed findings.
