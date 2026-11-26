# Continuous sign-chart refinement after P6

This follow-up preserves the frozen P6 sources. It studies the final R2 chirp
update's behavior when received pilots provide weak temporal support. The
derivation below is a proposed operator property; decoding gains require
separate measurements. It does not reclassify P6 development measurements as
new confirmatory results.

## Motivation and scope

P6 uses a coarse bank twice. Both calls can repair a wrong chirp sign. Its
final rule combines a global phase residual with a local two-coordinate
Gauss–Newton tangent using observed pilot information `q`. A small `q`
strongly suppresses the global correction even when the remaining received
samples favor a different sign. P6's recorded sign-chart intervention
motivates preserving that possibility, while its refitting results caution
against assuming a new decoding advantage.

See the frozen [R2 derivation](../p6/results/dev_math/chirp/R2_MATH.md) and
[development study](../p6/DEV_ITER5_MATH.md). Neither a single local derivative
nor a favorable fixed-weight intervention proves a system-level benefit.

## Proposed map

Let `c_j` be the existing coherence score and `theta_j` the existing centered,
unwrapped trajectory. Partition the public bank into increasing and decreasing
trajectories, `S_+` and `S_-`. Each current group contains 81 templates. Keep the
existing coefficient 160 and define

\[
\ell_j=160c_j,\quad
\log Z_s=\operatorname{LSE}_{j\in S_s}\ell_j,\quad
m_s=\sum_{j\in S_s}\operatorname{softmax}(\ell|_{S_s})_j\theta_j.
\]

The signed difference of the normalized group masses is

\[
d=\tanh\!\left[\tfrac12(\log Z_+-\log Z_-)\right].
\]

Set

\[
a_+=\max(d,0)^2,\qquad a_-=\max(-d,0)^2.
\]

For the existing bounded tangent `u` and current trajectory `theta`, replace
only the weak-information term by

\[
r_{\rm weak}=(1-d^2)u+a_+(m_+-\theta)+a_-(m_--\theta).
\]

The complete final-pass residual is

\[
r_{\rm new}=q\,r_{\rm bank}+(1-q)r_{\rm weak},
\]

where `q` and the tempered bank residual at coefficient `40+120q` are the
unchanged P6 formulas. The first bank call is unchanged. There are no added
trainable parameters, temperatures, or inputs.

## Properties and their assumptions

**Convex weights.** Since `-1 <= d <= 1`, the tangent and two chart weights
are nonnegative and sum to one. A sign tie returns the existing tangent;
strong preference approaches the corresponding conditional global residual.
At `q=1` the formula reduces to the frozen final-pass bank update.

**Conditional C1 regularity.** For finite logits and nonempty sign groups,
group log-sum-exp and conditional means are smooth. Squared ReLU is C1:
both its value and first derivative vanish at zero. Therefore this chart
blend is C1 across equal sign masses, conditional on its other inputs.
Ordinary hard chart selection is generally discontinuous there. The new map
is generally not C2. This statement does not assert C1 regularity of every
operation in the receiver: phase unwrapping, the unchanged information
estimate, and other existing branches need their own assumptions.

**Gate sensitivity.** With `p_+ = (1+d)/2`, differentiating the two group
masses gives

\[
\|\nabla_\ell d\|_1=1-d^2,\qquad
\|\nabla_\ell d^2\|_1=2|d|(1-d^2)\leq\frac{4}{3\sqrt3}.
\]

Derivatives with respect to coherence acquire the existing factor 160.
The full residual also depends on within-chart trajectories, the tangent,
and `q`; the bound above is not a global Lipschitz constant for the residual
or decoder. Improved floating-point stability must be measured end to end.

**Global complex-gain compatibility.** Consider the complete normalized
receiver under `y -> c y` and `noise_variance -> |c|^2 noise_variance`, with
nonzero complex `c`. Its existing gauge removes the gain magnitude. Under
the remaining common phase, estimated taps and charge-one field cues rotate
together. Correlation magnitudes, pilot information, conditional phase
means, and the tangent innovations are unchanged in exact arithmetic. The
new scalar weights preserve this transformation law. The complete CFR is
equivariant and LLRs are invariant if the inherited operations satisfy their
existing assumptions. The raw residual by itself is **not** scale invariant:
its fixed ridges rely on the upstream normalization. Finite-precision error
and phase-chart edge cases still require explicit audit.

## Interpretation limits

Coherence scores are not calibrated log likelihoods. The squared mass
difference is a score-confidence weight, not a guarantee of Bayesian
confidence. A confidently wrong sign remains possible. Low pilot information
and global sign ambiguity are distinct quantities. The convex and C1
properties guarantee neither lower BLER nor monotonic convergence. The rule
cannot improve a stress input through this modification when the existing
pilot-information estimate saturates at `q=1`.

Fresh waveform tests with archived weights measure an inference modification.
Independent training requires new fits and separate reporting. Any new
evaluation must keep its selection data, held-out data, integer denominators,
and numerical failures explicit.
