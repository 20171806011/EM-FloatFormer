# Phase 6 mathematical development study

**Development gate passed.** The fresh Axial-Parity model scored 25/337/47, with nine R2 sign repairs and a direct R3 Green correlation of 0.352446. The selected receiver passed the complete 360-condition symmetry audit on freshly trained weights and on the previous iteration?s checkpoints.

## Development design

All fits use pairs 901–903 and the unchanged development firewall. Each arm gets the same real parameters, initial tensors, three EM passes, stream-5 B4 minibatches, 48 Adam updates with four warm-up steps, peak LR 0.0045 and minimum LR 0.00045. All counts use the original B32 stream-6 draw. The optional R3 Green probe uses B8 on stream 8 and permutations on stream 9.

Before the new fits, the development acceptance criteria were 25, 338 and 49 failed words in R1, R2 and R3 respectively, at least nine R2 sign repairs, and mean direct R3 Green correlation at least 0.34 and above the matched uniform control. Every trained gauge-enabled checkpoint must pass every original strict gauge condition and all initialization, first-gradient and regression checks must pass. Lower goals of 23, 335 and 47 words and a maximum absolute LLR change below 4e-13 are recorded separately. The development pairs have already been used for several rounds of model selection.

## Conditioning of the chirp update

For candidate scores $c_j$, let $w_j=\exp(\beta c_j)/\sum_k\exp(\beta c_k)$ and $m=\sum_jw_ja_j$, where $a_j$ is one lifted phase coordinate. Then

$$\frac{\partial m}{\partial c_j}=\beta w_j(a_j-m),\qquad L_{\infty\to|\cdot|}=\beta\sum_j w_j|a_j-m|.$$

A large template span only becomes influential when competing candidates with different coordinates have appreciable weights.

On the previous R2/901 erasure at the final EM pass, (L=178.406), the largest coherence perturbation was 1.61e-15, and the Jacobian predicted −1.833e-13 rad versus −1.794e-13 observed. The score is gauge invariant in exact arithmetic. Finite-precision perturbations of the score can still be amplified. An unconstrained circular mean also has a singularity when its resultant vanishes. This motivates keeping the global basin search at the first bank call and comparing a tempered second call with a continuous local fit to the measured complex paths. Endpoint errors under oracle interventions neither identify the source of every remaining word error nor guarantee an attainable reduction.

For the final-pass alternative, the pilot coordinate rows are $x_t=(t,t^2)$ and each row has an information weight $w_t$ proportional to pilot count times observed signal-to-noise reliability, clipped to the nominal range. Profile out constant phase to obtain $F=\sum_t w_t(x_t-\bar x_w)(x_t-\bar x_w)^T$. In the coordinates whitened by the nominal pilot design $F_0$, take $\lambda=\lambda_{\min}(F_0^{-1/2}F F_0^{-1/2})$, clipped to $[0,1]$. This is an observed design-support proxy. It uses the receiver grid and noise variance, which transform together under complex gain. The least-supported linear combination of slope and curvature determines the confidence.

Tempering alone at $\beta=40+120\lambda$ passed the targeted numerical audit, but broadened the posterior enough to raise the local derivative on the historical cell from 178.406 to 220.417. The selected update is $\lambda r_{\mathrm{bank}}+(1-\lambda)r_{\mathrm{GN}}$. The local tangent profiles one free complex amplitude per antenna, uses centered orthonormal linear/quadratic columns, positive ridge and a bounded chart radius. At the historical erasure, $\lambda=0.001842$ reduces the partial bank derivative to 0.4059. The combined residual perturbation was $2.15\times10^{-15}$ rad. This is a local measurement including the tangent term; the partial bank derivative alone is not a bound on the complete decoder.

| R2 final-pass rule | Refitted Axial-Parity errors | Refitted sign repairs |
|---|---:|---:|
| Original sharp bank | 337 | 9/10 |
| Fixed temperature coefficient 40 | 349 | 9/10 |
| Local tangent alone | 370 | 5/10 |
| ¼ original bank + ¾ tangent | 367 | 5/10 |
| Conditional bank within the selected sign | 343 | 9/10 |
| Conditional bank and within-basin tangent | 340 | 9/10 |
| Fisher temperature | 337 | 9/10 |
| Fisher temperature and information-weighted tangent | 337 | 9/10 |

The second bank call still rescues wrong signs on some bursts. On the fully observed validation set the selected rule preserves the previous Axial total. On additional fixed-weight first-row erasure stress, its 559 failures exceed the original bank's 510 and the temperature-only rule's 519. For the interior-row dropout the corresponding counts were 673, 611 and 605. Conditioning the estimated trajectory improves numerical stability under complex rescaling, while the cost of losing a possible global sign correction is visible under missing pilots. The [R2 derivations, sign diagnostics and stress comparisons](results/dev_math/chirp/R2_MATH.md) retain the full alternatives.

## Geometry of the training loss

For $u=10\tanh(l/10)$ and $p=\sigma(u)$ the capped BCE gradient with respect to the raw logit is $(p-y)\operatorname{sech}^2(l/10)$. One finite alternative used

$$L_\mathrm{bit}=\tfrac12[\mathrm{BCE}(u,y)+2(p-y)^2],\quad
\frac{\partial L_\mathrm{bit}}{\partial l}=\frac{1+4p(1-p)}2 (p-y)\operatorname{sech}^2(l/10).$$

The multiplier ranges from 0.5 to 1. It preserves first-order weight at the decision boundary and reduces weight on confident predictions. Both losses on the predicted probability are proper scoring rules. The real scalar objective differentiates through the real and imaginary components of complex parameters. The Adam preconditioner, NMSE coefficient 0.005 and squared-displacement coefficient 0.00005 are identical across arms.

Test this one common loss on both parity arms and all three regimes against the Iteration 5 geometry. Retain it for integration only if Axial-Parity has no increase in any regime's pooled failures, no loss of sign repairs, and a strict improvement in one or more regimes. Otherwise retain the original loss. Independently evaluate physical candidates against the same original objective, then refit the integrated selection and repeat its audits. Held-out performance plays no part in selection.

The Brier mixture gave Axial-Parity 24/339/49 and Local-Parity 23/343/49, with 9/10 R2 repairs in both arms. Original capped BCE gave 25/337/49 and 24/339/49. The two-word R2 cost fails the selection rule. The historical improvement at update 24 on R1 is compatible with a different decoding optimum on these three pairs. It does not establish a general late-step overfitting mechanism. [Objective fits](results/dev_math/loss_control/)

## R1 phase and radial calibration

Let $d_t=\arg(c_{t+1}c_t^*)$ and $\tau=(n-(N-1)/2)/(N+\mathrm{CP})$. Conditional on adjacent midpoint phases in a local chart, a Brownian bridge has linear conditional mean. Relative to the present midpoint, it is $\tau d_{t-1}$ on the left and $\tau d_t$ on the right, with a flat exterior beyond the observed midpoints. The tested sample-clock phasor multiplies this excursion by $s^2/(s^2+E[\sigma^2])$, using the energy of the centered learned phase readout. Its exact zero at initialization preserves Fixed. A second alternative estimated the slope of the observed complex waveform by a local posterior step. After profiling the row's constant phase, define $\tilde\tau=\tau-\sum|p|^2\tau/\sum|p|^2$ for predicted waveform $p$. The correction uses $\sum\tilde\tau\operatorname{Im}(p^*y)$ divided by $\sum\tilde\tau^2|p|^2+\nu/(2V)$ with $V=(N+\mathrm{CP})E[\sigma^2]$.

The existing per-row orthogonalization removes a single complex linear gain from each waveform. Within-row pilot/data radial contrasts can survive that projection. To test the lost component, the alternative restored $(\gamma_t-\gamma_P)x$ using a burst-wide coefficient $\gamma_P$ estimated on observed pilots. This tests a particular calibration mechanism without exposing realized PA or phase noise to the receiver.

| R1 candidate | Axial-Parity | Local-Parity |
|---|---:|---:|
| Prior | 25 | 24 |
| Bridge | 27 | 24 |
| Pilot-referenced PA | 26 | 24 |
| Both | 27 | 23 |
| Data-conditioned tangent bridge | 25 | 24 |

The tangent suppression tied every pair in both arms. The R1 decoder retains the original physics. [Factorial results](results/dev_math/pa/summary.json) · [Tangent ablation](results/dev_math/pa/tangent_summary.json)

## R3 array geometry

For pilot innovations $e_k=y_k-h_kx_k$ form $C=\sum e_ke_k^H/K$. The IRC alternative uses $\Sigma=\nu I+\alpha M(C-\nu I)M$, with the learned Hermitian contraction $M$ and $0\leq\alpha<1/2$. Its eigenvalues are bounded below by $(1-\alpha)\nu$. For $Q=\nu\Sigma^{-1}$ the distortionless estimate is $h^HQy/(h^HQh)$ with modeled variance $\nu/(h^HQh)$. A solve for $Qh-h$ gives an exactly zero increment at zero handle. In correlated residuals estimated from the same receive grid, this is a plug-in estimate, not an oracle covariance.

The steered alternative works directly on each LS ray amplitude $a$. Let $\zeta=\sum_ra_{r+1}a_r^*/|\sum_ra_{r+1}a_r^*|$, with a deterministic fallback at zero lag. $D=\mathrm{diag}(1,\zeta,\ldots,\zeta^{R-1})$ demodulates that ray to broadside under $D^H$. Since $M\mathbf1=\mathbf1$, $DM^2D^H$ preserves $D\mathbf1$. Squaring $M$ damps orthogonal eigenmodes with eigenvalues below one. The convex strength is the adjacent-lag coherence times $q/(q+1/\sqrt K)$, where $q$ is the learned spatial strength divided by its 0.18 cap and $K$ counts pilots. All phases and strengths come from the observation and its fitted rays.

| R3 matched comparison | Axial full / spatial off | Local full / spatial off |
|---|---:|---:|
| Previous DC smoother | 49 / 49 | 49 / 49 |
| IRC | 48 / 49 | 47 / 49 |
| Steered array smoother | 47 / 49 | 48 / 49 |

Direct joint Green was 0.352338 for IRC and 0.352446 for steered smoothing. With the steered smoother, full-grid spatial Green was 0.164721, final steering cosine reached 0.983847 versus 0.980301 with the handle suppressed, and Axial error counts were 8/15/24 versus 9/16/24. Select steered smoothing on the primary Axial outcome; Local performed best under IRC. The selected direct joint exceeds the 0.34 gate and the 0.268702 development control. It is slightly below the original 0.352660. The two mechanisms were compared separately. The per-pass gradient observations retain both physical units and dimensionless multiplicative derivatives; gradients of delay measured in samples and shrinkage measured as a fraction are not directly comparable. [IRC and gradient evidence](results/dev_math/spatial/) · [ULA ablations](results/dev_math/spatial/ula/summary.json)

On the third pass of the trained Axial receiver, the delay RMS and corresponding objective-gradient RMS were 0.05244 samples and 0.02637 per sample; for the spatial-strength readout they were 0.05103 and 0.002230. The operator-entry gradient RMS was 0.01274 under the steered smoother and 0.000394 under IRC. These are observed sensitivities on development training batch zero, averaged across three fits.

## Integrated selection

| Final trained receiver | R1 / 384 | R2 / 1,728 | R3 / 384 |
|---|---:|---:|---:|
| Axial-Pure | 43 | 458 | 49 |
| Local-Pure | 42 | 461 | 50 |
| Axial-Parity | 25 | 337 | 47 |
| Local-Parity | 24 | 339 | 48 |
| Axial-NoGauge | 36 | 448 | 53 |
| Fixed+Parity | 29 | 429 | 65 |
| Equal-information classical comparator | 40 | 446 | 65 |

All five arms received fresh 48-update fits on all nine regime/pair combinations. Initialization produced bit-identical LLRs and CFRs relative to both the current and archived fixed receivers in 45/45 cases. All 856 trainable real scalars had finite nonzero gradients on the first actual training minibatch in all 45 fits. The final Axial-Parity readout repaired 9/10 initially wrong R2 signs. R3 direct joint Green was 0.35244555. A further stress-only control replaced the low-information tangent with the previously tested sign-conditioned bank on the same weights. First-row erasure fell from 559 to 509 failed words, and repairs rose from 8/15 to 13/15. This exploratory control uses a hard sign decision and has no independent training or full numerical gate. The selected receiver uses the smooth information-weighted tangent. [Final recipe](results/dev_math/final/recipe.json) · [Fit and gradient receipts](results/dev_math/final/selected/) · [Stress ablation](results/dev_math/chirp/chart_stress.json)

The selected receiver passed 360/360 trained gain and pilot checks on the new checkpoints and 360/360 on the Iteration 5 checkpoints reloaded into the revised forward model. Each suite compared 21,565,440 coded-bit decisions with zero flips. The maximum absolute LLR difference was 6.181721801112872e-13 in both, meeting the strict 1e-12 criterion and exceeding the aspirational 4e-13 target. Maximum relative CFR error was 9.887e-15. The worst LLR case was R2/901, Axial-Parity, first pilot row erased.

The P6 suite passed 38/38. The unchanged P5 and P5-B test bodies passed 15/15 and 13/13 using the guarded development fixture adapter. The default P4 verifier exited 0. Pre-development SHA-256 inventories covered all 138 P5 and 165 P5-B files as opaque bytes; their protected contents and the development firewall were unchanged at gate verification. No formal waveform was generated during development. [Machine-readable gate](results/dev_math/gate.json) ? [Regression and integrity checks](results/dev_math/checks/summary.json)
