# Mathematical scope

## Received observations versus transmitted symbols

The linear R2 frequency model is `y = G x + n`, with proper complex noise
covariance `N0 I`. The archived digital corruption is `y' = A y`, where A is
a known diagonal gain in the diagnostic arms. For nonzero gain, `A^-1 y'`
has the original noise law; for zero gain there is no receive equation.
Represent retained rows by binary W. A known transmit pilot still has
`m_k = pilot_k, v_k = 0` even if receive row k is absent. Under ICI, its
transmitted contribution can reach other retained output rows.

This study does not model physical attenuation before independent receiver
noise, nor infer unknown gains. For physical `y'=A G x+n`, positive-gain
inversion would produce a different noise covariance and need whitening.

## Expected weighted amplitude fit

For fixed delay/phase handles, ray p has frequency operator

`B_p = cpe F diag(rot_p) F^H diag(basis_p)`.

The effective operator for receive antenna r is

`G_r(a) = sum_p a_pr B_p + E_r`, with `E_r = diag(delta_h_r)`.

For the supplied diagonal symbol belief, `E[x]=m`,
`E[(x-m)(x-m)^H]=V=diag(v)`, minimize

`J(a) = E_q ||sqrt(W)(y-G(a)x)||^2 + sum_r a_r^H Lambda a_r`.

Let design column p be `d_p=B_p m`. The normal correction and RHS correction
are respectively

`C_pq = sum_(t,f,k) w_tf v_tk conj(B_tp,fk) B_tq,fk`,

`c_pr = sum_(t,k) w_tk v_tk conj(B_tp,kk) delta_h_rtk`.

Then

`(D^H W D + C + Lambda) a_r = D^H W (y_r-E_r m) - c_r`.

The subtraction sign follows from differentiating the cross term between
`sum a_p B_p e` and `E e`. Removing `E m` from the observed mean without the
covariance cross-term would optimize a different objective. Direct objective
evaluation uses `sum w v |G(a)_fk|^2`, which retains the learned-residual
square and cross terms as one nonnegative quantity. This covariance identity
is exact for the specified diagonal q and linear R2 model; q itself remains
approximate. It does not extend unchanged through nonlinear R1/R3 PA models.

## Gaussian cavity likelihood

The old scalar clean-up uses `clean=y-(G-D)m` and demaps with `N0/|D|^2`.
But exactly `clean=D x+(G-D)(x-m)+n`. Ignoring the remaining interference
can misstate reliability even if the diagonal channel estimate improves.
Since m is inferred from y, merely adding a plug-in interference variance
does not establish independence or a true posterior.

Here other-symbol interference uses a separately constructed approximate
Gaussian prior. Let `Gw=sqrt(W)G`, `yw=sqrt(W)y`, and prior moments `(m,v)`:

`C = N0 I + Gw diag(v) Gw^H`,

`r = yw - Gw m`, `t = Gw^H C^-1 r`,

`s_k = [Gw^H C^-1 Gw]_kk`.

For an informed column, the own-symbol cavity parameters are

`z_k = m_k + t_k/s_k`,

`tau_k = 1/s_k - v_k`.

By a rank-one inverse identity, these equal the Gaussian likelihood obtained
by removing that symbol's own prior contribution before inversion. Its own
prior mean and variance cancel. A finite-QAM posterior is then formed from
`exp(-|z_k-x_k|^2/tau_k)` times the separately supplied bit prior. Its first
and complex second central moments feed the expected physical fit. Known
TX pilot means and zero variances are reinserted explicitly.

Final channel LLRs use the inherited max-log Gray 16-QAM readout from z,tau,
excluding the own-symbol code prior. The external LDPC decoder can then add
code constraints without directly receiving them twice. Prior messages from
other symbols and repeated outer use of y remain loopy approximations.
The method omits cross-symbol covariance and improper-complex pseudo-
covariance, so it is not exact QAM/LDPC inference or an exact global EM scheme.
No uncertainty over the learned/estimated G itself is integrated out.

## Gauge and finite precision

Under `y -> alpha y`, `N0 -> |alpha|^2 N0`, physical G and h should transform
by alpha, while dimensionless symbol beliefs and LLRs are invariant. Receive
gain metadata is unchanged. Valid-pilot RMS gauge and covariance-normalized
matrix solves preserve this relation in exact arithmetic. Zero likelihood
columns return zero channel information and preserve the prior.

Gaussian solves normalize by the larger of noise power and maximum weighted
row energy. An internal `64*N*eps` normalized noise floor protects singular
zero/extremely-low-noise cases; tau has a dimensionless
`eps*max(v,1)` cancellation floor. Events are logged. These computational
conventions, and the inherited finite sign saturation at reported N0=0,
make no claim of exact noiseless inference. C* numerical tolerances are
checked without inferring FP32/BF16/FP16 performance from FP64 results.
