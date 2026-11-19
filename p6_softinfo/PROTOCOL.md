# P6 soft-information and full-waveform profile protocol

This opt-in study extends `c699680dd84560747ddf40ecacd59821987bf545` without
changing published P4/P5/P5B/P6 or the previous follow-up. Published main is
`63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`. P4 historical reference is
`8380084c6dece64119ab72f9c3c1c212e154fb17`.

## Fixed hypotheses and scope

R2 only: linear three-ray sample-time waveform, two-dimensional OFDM and the
archived 16-QAM/LDPC receiver. Reuse all ten published parity checkpoint
pairs 06--15; do not retrain. The mathematical candidate is fixed as
`variance_profile`: carry diagonal conditional symbol covariance in all
three amplitude solves, then perform two joint full-waveform profile phase
updates during the final solve. Its settings are fixed by the source hash.
Development pairs 901--903 are reused engineering evidence, never fresh
confirmation. We will retain adverse findings rather than select a new
winner from held-out scores.

Factorial controls are Trust (original), variance only, profile only, and
the combined candidate. Additional anchors are the previous Smooth chart,
matched-capacity local parity model, fixed physical/parity model,
waveform-DD classical solver and 2-D LMMSE classical solver. All see identical
observations, pilots, reported noise and decoder; truth is used only after
inference to score output. Parameter counts and checkpoint weights stay
unchanged. Refinement adds computation; compute budgets are not matched.

## Prospective fresh cohort

Before any fresh run, write `FREEZE.json` with exact source, protocol and
checkpoint SHA-256 digests, environment, and seed schedule. This is an
in-session prospective freeze, not an externally published preregistration.
For index i=1,...,10 use archived pair i+5 and four independent B8 draws
with seed `713611190000 + 1000*i + j`, j=0,...,3, R2 at 9 dB. This namespace
is disjoint from the previous follow-up's `602611190000` base. Record hashes
of each simulated observation and payload. Every method receives every
draw in every condition.

Five conditions are intact, first pilot row amplitude multiplied by 0.4,
first pilot row zeroed, interior pilot row 9 zeroed, and interleaved half of
pilot REs zeroed. These are digital unflagged observation interventions:
pilot mask, nominal pilot symbols and reported noise remain unchanged.
They do not constitute a new physical channel family. The classical
priors match the intact public model, not the added corruption process.

Each condition has 320 bursts, 5,760 LDPC blocks, 737,280 coded bits and
368,640 information bits per method. Do not count these correlated blocks
as thousands of independent experimental replicates.

## Inference and reporting

The primary endpoint is decoded block-error count. Six predeclared tests
compare combined `variance_profile` against Trust and against Smooth on
intact, first-row erasure, and interior-row erasure. Apply two-sided paired
sign-flip enumeration to ten checkpoint/waveform cluster differences and
Holm adjustment across exactly these six tests. The test requires
exchangeable/symmetric independent null cluster signs. Report signed
effects, wins/ties/losses, raw counts and approximate pointwise unadjusted
Student-t 95% intervals. A corrected significant adverse effect is a loss,
not evidence of successful optimization. No noninferiority conclusion
follows from nonsignificance.

Other conditions, ablations, raw coded BER, information BER and channel
NMSE are descriptive. No new candidate will be selected after fresh data.
Preserve every cluster result; reject incomplete or duplicated grids.

## Mathematical and numerical validation

Independently enumerate finite constellations and small random waveforms
to check moments and the expected-Gram identity; test profiled derivatives
and objective decrease. Check the R2 no-intervention copy bit for bit
against frozen P6. Check that weights remain unchanged. The Gaussian
amplitude-evidence and normalized bit-probability-mixture helpers receive
algebraic tests only: they are not an end-to-end eight-mode BER result.

On reused development frames, audit complex gain transformations including
the archived extreme scale grid and erased/zero/noise degeneracies. Retain
strict 1e-12 failures and hard-decision differences. A changed discrete
line-search decision may expose finite-precision sensitivity. Record such
failures rather than relaxing the inherited threshold retrospectively.
The production study uses float64/complex128; no float32 deployment claim.

Measure receiver-only latency on one CPU/runtime, with a single Torch
thread, no other experiment workers, warmups, repeated interleaved method
orders and raw timings. Exclude simulation and final LDPC decoding.

An independent agent in this same environment will audit source identity,
all raw grids and selected independent replays. This is not external
independent reproduction and does not reproduce training. Publish source,
full raw results, adverse numerical findings and limitations on a research
branch/draft PR; do not promote the candidate to main by phase naming.
