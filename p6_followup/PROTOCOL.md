# P6 mathematical follow-up: protocol fixed before fresh scoring

Date: 2026-11-19. Base commit: `63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`.

This is a limited, prospective-in-session **inference intervention**, not a new
formal phase, external preregistration, or independent reproduction of training.
The old P6 source, checkpoints, protocols and results remain unchanged. The
existing development record already motivated sign-conditioned corrections.
The novel intervention is the continuous squared-confidence chart gate derived
in `MATH.md`; the hard sign chart is an existing P6 exploratory control.

## Questions and fixed intervention

Can a parameter-free chart gate restore global Doppler sign correction when
pilot-aperture information is weak, while avoiding the hard chart's discontinuity
at equal sign scores? Its final R2 chirp update is

\[
r=q r_{bank}+(1-q)[(1-d^2)u+[d]_+^2(m_+-\theta)+[-d]_+^2(m_--\theta)],
\quad d=\tanh((\log Z_+-\log Z_-)/2).
\]

All bank templates, beta=160, the pilot information coefficient q, bounded tangent,
learned fields, three waveform iterations, and LDPC operations are inherited.
No hyperparameter sweep or extra fit is planned. A separate optional exact R2
emission simplification removes a mathematically inverse FFT pair. It is
**disabled in all primary efficacy comparisons** and measured separately.

## Integration/development and freeze

Use archived final P6 development parity checkpoints 901, 902 and 903, stream 6,
B8, for integration and regression only. These are reused development samples,
not confirmatory evidence. Unit tests cover sign ties, gradients, convex weights,
full-information equality, finite zero-input behavior and phase invariance.

Before the first fresh score, write `FREEZE.json` containing this protocol,
follow-up runtime sources, inherited Python sources, all selected checkpoint
SHA-256 values, the seed/condition/method design and runtime versions. Refuse fresh
scoring if these hashes change. If a bug invalidates scoring, retain the failed
record and explicitly disclose a new freeze; never silently replace evidence.
The manifest is a local session freeze, not proof of public prior registration.

## Fresh paired cohort

R2 only: the public super-aliased Doppler ensemble, 20 OFDM symbols x 32 carriers,
one receive antenna, three rays, 16-QAM, rate-1/2 (128,64) LDPC, Es/N0=9 dB.
Use all ten archived formal-cohort-A **Axial-Parity** checkpoints, pairs 6 through
15, plus their matched **Local-Parity** checkpoints. Index i in 1..10 maps to
checkpoint pair i+5. For batch j in 0..3 draw B8 with

`seed = 602611190000 + 1000*i + j`.

This fresh namespace is separate from the inherited P5/P6 train/validation/test
streams. Each receiver sees exactly the same simulated waveform and payload for
its paired comparison. There are 320 independent bursts, 5,760 LDPC blocks and
737,280 coded bits per condition; the statistical unit is the ten
checkpoint/waveform clusters, **not individual correlated blocks**.

Five conditions are fixed in advance. Change received pilot entries only; keep
the nominal pilot mask, known pilot symbols, noise declaration and payload fixed:

1. `intact`: unmodified observations.
2. `first_attenuated`: multiply received pilots on row 0 by 0.4.
3. `first_erased`: set received pilots on row 0 to zero.
4. `interior_erased`: set received pilots on row 9 to zero.
5. `scattered_erased`: zero pilots where `(row + carrier//2) % 2 == 0`.

These are **unflagged digital observation corruptions**, including the noise in
the selected entries. They are not a newly simulated fading/dropout process or
evidence of generalization to an unknown channel family. Even intact pilots can
yield q<1 in weak fades; intact equality must be measured, not assumed.

Seven arms: frozen Axial-Parity `trust`; proposed `smooth`; existing exploratory
`hard_chart` with the same Axial weights; frozen `local_parity`; `fixed_parity`
(three steps, no trained correction); classical `waveform_dd`; and `lmmse_2d`.
Both classical methods use the same public channel model available to P6.
The LMMSE covariance is matched to the **original channel prior**, not to the
added unflagged corruption. Fixed+Parity is a fixed physical receiver and should
not be mislabeled a new matched-prior classical optimum. No receiver receives
realized channel, bits or truth at inference. Truth is used only after inference
to score channel error and Doppler sign diagnostics.

## Outcomes and statistics

The primary family contains exactly two two-sided comparisons: smooth versus
trust BLER for `first_erased` and `interior_erased`. For each, aggregate all four
batches within each index; enumerate all 2^10 sign flips of paired block-error
differences, then apply Holm correction across these two tests at alpha=0.05.
This randomization reference assumes paired-null exchangeability/symmetric
differences; report the ten differences and wins/ties/losses. A positive effect
means fewer errors under smooth. Report absolute percentage-point effects and
relative error changes, with seed-cluster uncertainty labeled as approximate.

All five conditions and all seven arms remain in the report, including adverse
outcomes. BER, post-LDPC information BER, channel NMSE, sign repair/regression,
hard-chart and classical/local comparisons are descriptive, not extra primary
success opportunities. No statistical equivalence or attention superiority is
inferred from nonsignificance.

## Numerical, runtime and reproducibility checks

Use CPU float64/complex128, Torch 2.12.1, one intra-op and one inter-op thread,
deterministic algorithms; record actual dependency versions and hardware.
Test the complete receiver under y -> c*y and noise -> |c|^2*noise, including
extreme magnitudes, randomized phases, missing pilots, degenerate pilots, zero
field and zero declared noise. Reuse P6 strict thresholds (<1e-12 LLR and relative
CFR discrepancy, no coded hard flips); retain and report every failed case.
These tests do not establish a uniform floating-point theorem or FP32/FP16 safety.

Time B1, B8 and B64 receiver calls on this machine, interleaving methods and
retaining samples after warmup. Exclude simulation and final external LDPC decode;
include internal parity and context dispatch. Compare original, smooth, and
smooth+optional exact emission separately. Do not compare absolute times to the
historical workstation as a claimed speedup. Do not run other experiment jobs
during timing.

An independent agent will review the code/statistics and replay preselected
fresh indices 1 and 10 in a separate process, checking frame hashes and integer
counts. This is independent checkpoint-inference checking in the same environment,
not external-lab reproduction, fresh training or a second hardware validation.

## Reporting decision

Publish the implementation, derivation, protocol, raw per-batch counts, hashes,
tests and measured limitations regardless of the direction of results. A new
directory or phase label is never evidence that the model wins. Broad P4/P6
claims about channel generalization, matched classical optimality, Hermitian
attention benefit and low precision require their own experiments.
