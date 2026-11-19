# Conditional uncertainty and a joint waveform profile objective

## What is and is not new

P6 already implements complex Hermitian attention, gauge normalization,
sample-time waveform reconstruction, ridge amplitude fitting and local
tap-based centered chirp/Gauss--Newton updates. We retain them. This study
adds conditional symbol second moments to the full-waveform normal matrix
and optionally optimizes a fixed-posterior full-waveform profile objective
over all ray phase coordinates jointly. It does not invent complex-valued
processing, variable projection, or LDPC extrinsic information.

## Posterior moments

For the existing separable 16-QAM approximate posterior q, carry
`m=E_q[x]` and `v=E_q[|x-m|^2]`. The posterior uses precisely the archived
channel likelihood and, when enabled, its one-round extrinsic check
messages. Pilots are deterministic: v=0. Uninformed payload without parity
has mean zero and variance one. Parity creates inter-symbol dependence;
retaining only diagonal v is a mean-field approximation, not the full
codeword posterior covariance.

For white sample noise and fixed nonlinear phase parameters theta,

    E_q ||y-D(x,theta)a||^2
      = ||y-D(m,theta)a||^2 + a^H C_v(theta) a,

where `C_v=E_q[(D(x)-D(m))^H(D(x)-D(m))]`. The conditional right-hand side
is `D(m)^H y`. The regularized update is

    a* = (D(m)^H D(m) + C_v + Lambda)^(-1) D(m)^H y.

Here Lambda is exactly the existing receiver ridge, not a newly claimed
matched Bayesian amplitude prior. This equation optimizes the stated
fixed-q surrogate. The approximate demapper, neural corrections and outer
loop do not imply exact EM or monotonic likelihood/BER.

## Exact R2 expected-Gram correction under diagonal q

Write `b_pk` for the frequency delay basis, `r_tmp` for the sample ray
rotation, and `c_t` for CPE. With unitary IFFT:

    D_tmp = c_t r_tmp IFFT(x_tk b_pk)_m.

For independent frequency errors with variances v_tk, its correction is

    (C_v)_pq = sum_t |c_t|^2
       [(1/N) sum_k v_tk conjugate(b_pk) b_qk]
       [sum_m conjugate(r_tmp) r_tmq].

Retain off-diagonal ray interactions. This is Hermitian PSD up to floating
roundoff, vanishes at v=0, and includes within-symbol rotations/ICI. The
expression does not extend unchanged through R1/R3 nonlinear PA models.

## Full profile refinement

Hold m,v, delay basis, ridge, CPE and learned channel residual fixed while
adding two centered polynomial phase coordinates per ray, lifted to sample
time using the archived interpolation. With three rays there are six real
coordinates. Eliminate complex amplitudes by the preceding solve, then
differentiate the full profile objective through that solve. Use the full
6x6 real Hessian, retaining cross-ray blocks; this is not the older
per-ray tap-space local update or a mislabeled Schur Gauss--Newton matrix.
Let g be its gradient and H its Hessian. With
`s=max(max|eigenvalue(H)|, ||g||_2)` (s=1 only when both are exactly zero),
the shift is `max(0, 1e-6*s-lambda_min(H))`. This preserves all cross-ray
blocks and gives a positive-definite direction matrix. The step cap is
0.25 peak radians per iteration, at most two iterations in the final
receiver solve. Armijo constant is 1e-4 and the eight trial scales are
1 through 1/128 by halves. The objective is recomputed with
the same moments and all theta-dependent covariance terms for every trial.
An accepted step must meet the fixed-objective decrease rule.

This is an inference-only refinement with detached inputs; we do not claim
to train through it. A local phase chart does not solve the joint global
Doppler-sign ambiguity. Fixed-surrogate descent does not guarantee BER,
BLER or true likelihood improvement.

As in frozen P6, the learned `delta_h` residual is retained in subsequent
demapping but is outside this physical waveform fitting objective. Final
LLR variance/calibration also remains the archived demapper approximation;
the added symbol covariance is used in amplitude fitting and profile
refinement, not a newly derived exact output-bit uncertainty model.

## Evidence and bit mixture helpers

For deterministic x, a proper zero-mean circular Gaussian amplitude prior
Gamma and proper complex Gaussian noise R yield C=R+D Gamma D^H. The
marginal negative log likelihood includes `y^H C^-1 y + logdet(C)` (plus
normalization); the minimized ridge residual alone is not evidence.
The helper tests this identity against a dense covariance calculation.
The public P6 amplitude distribution is not exactly this Gaussian prior.

If valid branch weights pi_s and positive-one bit LLRs L_si are available,

    L_i = log sum_s pi_s sigmoid(L_si)
          - log sum_s pi_s sigmoid(-L_si).

The tested helper performs normalized probability mixing stably. We do not
create a calibrated eight-sign posterior by softmaxing coherence or the
soft-q profile objective. These helpers are code with mathematical tests;
this study has no full eight-mode receiver experiment.

## Symmetry and precision

Under y->g y, noise variance->|g|^2 noise variance, existing gauge
normalization makes the internal coordinates charge-neutral; returned
channel estimates transform by g. The new covariance is built from those
normalized quantities. Hermitian attention is inherited without change.
Exact algebraic equivariance and finite-precision tolerance are separate
claims. Line searches, ill-conditioned solves and machine rounding must
be audited. All reported results use float64/complex128.
