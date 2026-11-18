# Conditioning the R2 chirp update

The chirp bank makes two decisions in a three-pass receiver. It reads soft-symbol LS taps at EM indices 1 and 2. Both reads can repair a wrong Doppler sign. Replacing the second bank read with a local continuous fit reduced sign repairs from 9/10 to 5/10 after retraining. The last pass still needs a basin-selection mechanism on some bursts.

## Derivative of the global target

For coherence scores \(c_j\), weights \(w_j=\mathrm{softmax}(\beta c)_j\), and centered, unwrapped template angles \(\theta_{jt}\), the target is \(m_t=\sum_j w_j\theta_{jt}\). Differentiating gives

\[
\frac{\partial m_t}{\partial c_j}=\beta w_j(\theta_{jt}-m_t),\qquad
L_t=\|\nabla_c m_t\|_1=\beta\sum_j w_j|\theta_{jt}-m_t|.
\]

Sharpness and posterior angular spread both matter. At the archived R2/901 first-pilot-row erasure coordinate (burst index 1, ray index 2, symbol index 19, third EM pass), \(\beta=160\), the template range was \([-7.163,7.163]\) rad and \(L_t=178.406\). The measured coherence perturbation had maximum magnitude \(1.610\times10^{-15}\). The Jacobian predicted a target shift of \(-1.833\times10^{-13}\) rad, against \(-1.794\times10^{-13}\) measured. Lowering the temperature coefficient reduces the upper bound on \(L_t\) at a fixed template range. The posterior can broaden enough to reverse the improvement at an individual operating point.

Moving the same mixture directly to the circle also has a conditioning cost. For \(z_t=\sum_jw_j e^{i\theta_{jt}}\), the phase derivative is \(\beta w_j\sin(\theta_{jt}-\arg z_t)/|z_t|\). Opposite phasors can make the denominator arbitrarily small. A chirp's winding direction also carries information over the sequence of symbol times. These observations motivate explicit sign components and a bounded tangent update rather than an unconditional circular average.

## Continuous refinement on a phase trajectory

Let \(z_{rt}\) be the estimated complex tap for one ray and antenna after derotation by the current chirp. The nuisance gain for each antenna is \(a_r=T^{-1}\sum_t z_{rt}\). Let \(D\) contain two orthonormal, centered columns proportional to \(t\) and \(t^2-\overline{t^2}\) on a symmetric clock. The local cost is \(\sum_{rt}|z_{rt}-a_r\exp(iD_t b)|^2\). The columns' zero means make the phase coefficients orthogonal to the free complex gain in the linearized fit. At \(b=0\) the descent half-gradient in the time coordinate is \(s_t=\sum_r \mathrm{Im}(z_{rt}a_r^*)\). The profiled Gauss–Newton normal matrix is \(I\sum_r|a_r|^2\). The damped update is

\[
u=\frac{DD^\top s}{\sum_r |a_r|^2+0.001+0.02\sum_r\overline{|z_{rt}|^2}}.
\]

The receiver limits the peak of this polynomial by dividing by \(\sqrt{1+(\|u\|_\infty/r)^2}\), with \(r=1.57\) rad in the local chart. Separate antenna phase rotations cancel in \(z_{rt}a_r^*\). The positive ridge defines the update at zero tap power. On a sign-conditioned bank we also tested applying this step after the bank chose its sign, with \(r\) set by half a cell of the public Cartesian endpoint-frequency grid.

The sign-conditioned bank chooses the sign with greater log-sum-exp mass over its 81 public templates, then averages angles within that component. This removes cross-sign averaging from the terminal sum. The discrete decision is piecewise constant. Its fixed-weight benefit for Axial-Parity did not survive fresh fitting.

## Relative information from observed pilots

For a pilot-bearing OFDM symbol \(t\), \(n_t\) counts pilot sites, \(P_t\) is their mean observed power across sites and antennas, and \(\nu\) is the observed noise variance, in normalized units. Define \(w_t=n_t\mathrm{clip}(P_t/\nu-1,0,1)\). At zero noise, a row receives full weight if its pilot energy is positive. This row-level reliability can remain saturated under partial tone dropout. For the centered clock \(s_t\in[-1,1]\), take \(d_t=(s_t,s_t^2)^\top\). Profiling a constant phase gives

\[
F(w)=\sum_t w_t(d_t-\bar d_w)(d_t-\bar d_w)^\top,\qquad
q=\lambda_{\min}\!\left(F(n)^{-1/2}F(w)F(n)^{-1/2}\right).
\]

Cholesky whitening gives the same generalized eigenvalues. Clamp \(q\) to \([0,1]\) for roundoff and use zero for nominal rank below two. R2 pilot rows are 0, 9, 10 and 19. Dropping an outer row leaves little information in one direction of the two-coordinate chirp fit and \(q=0.001842\). Dropping row 9 gives \(q=0.664825\). The receiver uses received samples, pilot coordinates and noise, with common complex gain canceling in the power ratio.

The Fisher-temperature candidate uses \(\beta_2=40+120q\) at the last pass. Its historical-coordinate \(L_t\) rose to 220.417, with measured \(\Delta m_t=-2.176\times10^{-13}\) despite \(\beta_2=40.221\). Its fresh-checkpoint R2 audit passed 120/120 conditions, with zero flips and maximum \(|\Delta\mathrm{LLR}|=7.008\times10^{-13}\).

The information-damped refinement uses \(q(m-\theta_{\rm current})+(1-q)u\). Its partial Jacobian with respect to the bank coherence scores at fixed tangent and \(q\) is \(q\nabla_c m\). The tangent term responds to changes in the taps through a separate derivative. On the same erased input, the bank partial row norm was 0.405924. Measured shifts were \(-3.96\times10^{-16}\) for the weighted bank contribution and \(+2.54\times10^{-15}\) for the weighted tangent, yielding a total residual change of \(+2.15\times10^{-15}\) rad. The traced end-to-end maximum LLR difference on that half-gain draw was \(4.85\times10^{-13}\). The other receiver operations contribute additional sensitivity.

## Matched development ablations

Each cell uses the original 48-update warm-cosine schedule, B4 shared training draws, B32 validation and all three development pairs. Each count is failed LDPC words out of 1,728. The first number in each cell belongs to Axial-Parity and the second to Local-Parity. The old-weight columns change only the late-pass update on the archived Iteration 5 checkpoints.

| Third-pass update | Old weights, errors | Fresh fits, errors | Fresh repaired signs out of 10 |
|---|---:|---:|---:|
| Original global bank, 160 | 337 / 339 | 337 / 339 (Iteration 5) | 9 / 9 |
| Global bank, 40 | 347 / 349 | 349 / 348 | 9 / 9 |
| Two-coordinate local tangent | 381 / 379 | 370 / 366 | 5 / 5 |
| ¼ original bank + ¾ local tangent | 369 / 362 | 367 / 373 | 5 / 5 |
| Bank conditioned on sign | 336 / 339 | 343 / 344 | 9 / 9 |
| Bank conditioned on sign, followed by tangent | 336 / 347 | 340 / 346 | 9 / 9 |
| Fisher temperature | 337 / 339 | 337 / 339 | 9 / 9 |
| Information-damped bank and tangent | 337 / 339 | 337 / 339 | 9 / 9 |

Neither a favorable oracle count nor these three reused pairs isolates quantization as the cause of the remaining wrong decisions. Preserving both bank reads was important on this development set.

## Performance under attenuated and scattered pilots

These measurements reuse the three Iteration 5 Axial-Parity checkpoints. Only the designated received pilot samples change; all payload samples and underlying simulator truth are shared. Temperature and trust use the same pilot information. This stress test measures decoded failures directly, independently of the gain audit that compares transformed and untransformed decisions.

| Pilot intervention | Mean relative information | Original failures | Fisher temperature | Information-damped update |
|---|---:|---:|---:|---:|
| Intact | 1.0000 | 337 | 337 | 337 |
| First pilot row ×0.8 | 1.0000 | 337 | 337 | 337 |
| First pilot row ×0.6 | 1.0000 | 331 | 331 | 331 |
| First pilot row ×0.4 | 0.5123 | 330 | 329 | 353 |
| First pilot row ×0.2 | 0.0018 | 325 | 341 | 379 |
| First pilot row erased | 0.0018 | 510 | 519 | 559 |
| Inner pilot row erased | 0.6648 | 611 | 605 | 673 |
| Interleaved half of pilot tones erased | 1.0000 | 345 | 345 | 345 |
| Staggered quarter of pilot tones erased | 1.0000 | 478 | 478 | 478 |

On intact validation bursts all three operators agree. At 0.4 attenuation, the per-burst information ranges from 0.0166 to 1. The row-level estimate correctly distinguishes a missing endpoint from a missing inner row, but it remains saturated on the two scattered-tone masks. The information-damped update costs 49 decoded words for complete first-row erasure and 62 for inner-row erasure relative to the original bank. The stricter gain invariance and smaller historical-coordinate bank derivative come with a measurable decoding cost on those inputs. This receiver has shown no improvement in R2 intact dev BLER.

On the four gauge-enabled arms, information damping passed 120/120 full-grid R2 tests with archived Iteration 5 weights, then 120/120 with fresh development weights. Each set covered 14,929,920 payload decisions with zero flips. Both maxima were \(6.182\times10^{-13}\). These satisfy the \(10^{-12}\) gate and exceed the aspirational \(4\times10^{-13}\).

A supplementary fixed-weight ablation changes just the \(1-q\) term, using the posterior-mean trajectory conditioned on the MAP sign, in place of the local tangent. With the first pilot row erased, failed words fell from 559 to 509 and repairs rose from 8/15 to 13/15 initial sign errors. At first-row amplitude 0.2, failures fell from 379 to 325 and repairs rose from 10/13 to 13/13. At inner-row erasure the count fell from 673 to 608 despite identical sign outcomes (19/42 repaired and no new wrong signs). Restoring the late sign decision accounts for part of the effect. The substantial inner-row difference also implicates features of the estimated trajectory beyond its net sign. The intact count and repairs were 337 and 9/10 in both variants. The extra sign-chart intervention was interpretive and had no independent retraining, full symmetry validation or role in selecting the frozen recipe.

[Old and fresh scores and gauges](summary.json) · [Fisher-temperature derivative](conditioning.json) · [Information-damped derivative](trust/conditioning.json) · [Original versus temperature stress](aperture.json) · [Original versus damped stress](trust/aperture.json) · [Sign-chart intervention](chart_stress.json)
