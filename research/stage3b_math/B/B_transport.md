# Known carrier offset before and after an 8-bit waveform cache

The Stage 3 pair of three-sample windows has two 4-bit indices. Once they have been recorded, a known carrier frequency offset (CFO) cannot in general be removed by relabeling these indices. At the actual C4 ramp frequency we find an explicit collision in the published encoder. The Bayes-optimal alternative for a fixed cache is to condition on its actual cells and the known offset. A system allowed to compensate before encoding has a different observation and must be evaluated in its own arm.

## Signal, target and sign

At [base 69fbd29](https://github.com/20171806011/EM-FloatFormer/tree/69fbd29acdb4eba7961d56176693786b6181477a), see `research/stage3_ministral_cache/{schema,data,analytic}.py` and `research/matched_cache/learned.py`. In the simulator each latent window orientation is multiplied by $e^{+i(\omega t+B_t)}$, where $B_0=0$ is a Brownian phase with independent increments of variance $\sigma^2\Delta t$. Noise is independent circular complex Gaussian. The offset is public. Code recovery uses $E(e^{-i\omega t}Y)$; prediction of a future **received** sector uses $+\omega t$ in the forward observation law. The C4 condition has $\omega=0.35$ rad/step, $h=2$, $\ell=2$, $t_p=4$, $\sigma=0.10$ and 2 dB. Its first, second, third and pilot deterministic phases are $0,0.70,1.40,1.40$ rad. At coincident pilot and third-window times the residual latent phase is shared but the measurement noises are independent.

## When can a rotation act directly on a finite cache?

Write $R_\alpha y=e^{i\alpha}y$ and let $E$ have finitely many possible values. A deterministic map $F_\alpha$ satisfies $F_\alpha(E(y))=E(R_\alpha y)$ on a specified input domain if and only if every nonempty code cell is mapped into a single code cell. Necessity follows by applying the equality to two points in one cell. Sufficiency follows by assigning their common destination as the output for this cell. For a probabilistic almost-sure claim, replace the cell condition by a point-mass conditional law of $E(R_\alpha Y)$ given each positive-probability source code. Invertibility on a rotation-invariant domain and finiteness imply that any exact deterministic action there permutes the occupied codes: the image of the action contains every occupied code.

For the equivariant seed-17 encoder the nonzero window orientation is a quadrant index $r$ and the canonical learned index is $c\in\{0,1,2,3\}$. The code is $4r+c$. For $\alpha=k\pi/2$, $F_\alpha(4r+c)=4((r+k)\bmod4)+c$. The canonicalized samples stay the same, so the learned subcell also stays the same. The zero window has probability zero and is omitted from this action.

For any $\alpha/(2\pi)$ irrational there is no such rule acting everywhere on nonzero windows for this encoder. Suppose there were. Its permutation has finite order $m$. Then the entire sequence $R_{jm\alpha}y$, $j=0,1,\ldots$, would have the same code as $y$. Choose $y$ with nonzero sum. The sequence of anchor angles is dense on the circle and enters each quadrant, contradicting the orientation contained in the code. Both nonzero Stage 3 C4 offsets satisfy this irrationality condition. The criterion also covers rational angles, for which the partition must be checked; an arbitrary non-quarter-turn does not follow from the encoder's quarter-turn symmetry.

**Concrete C4 collision.** For three identical complex samples $y_j=e^{i\theta}/\sqrt3$, the pinned encoder gives the following values. Both points in each pair have the same original code.

| $\omega t$ (rad) | $\theta$ (rad) | $E(y)$ | $E(e^{-i\omega t}y)$ |
|---:|---:|---:|---:|
|0.70|0.201718474|0|3|
|0.70|6.073796930|0|14|
|1.40|6.044651295|0|12|
|1.40|0.250805859|0|14|

## Decision-optimal use of the fixed observation

Let $S$ contain the transmitted labels, shared gauge and residual phases at all distinct times. Let $u_j=(X_j+G)\bmod4$, $s_u=i^u$, and $\rho$ be the linear SNR. For an encoder taking a complete three-coordinate window, the cache likelihood is

$$
L(z\mid u,b,\theta)=\int_{E(y)=z}\prod_{a=1}^3
\frac{\rho}{\pi}\exp\left[-\rho\left|y_a-3^{-1/2}s_u e^{i(b+\theta)}\right|^2\right]d^2y_a .
$$

For clarity the integral is over $\mathbb C^3$ with the product Lebesgue measure. Multiply the prior on $S$ by the two cell likelihoods with $\theta_1=0$, $\theta_2=\omega h$ and each unique relevant pilot's raw likelihood at $\theta_p=\omega t_p$. Normalize and integrate the task conditional $p(a\mid S)$ against this posterior. For Q1/Q2 the conditional is an indicator of the relative/original transmitted symbol. For Q3-repeat it is the probability that the noisy future observation around $s_{u_1}e^{i(B_{h+\ell}+\omega(h+\ell))}$ falls in sector $a$. This formula retains dependence between $B_h$, $B_{h+\ell}$ and $B_{t_p}$. For Q3 with a fresh independent uniform symbol, quarter-turn symmetry instead yields exactly four equal probabilities. These decisions assume the public generative law, raw references where provided, and a fixed encoder.

For categorical log loss the excess conditional risk of any predicted probability vector $q$ is $D_{\rm KL}(p\Vert q)$ where $p$ is the conditional distribution above. Thus $p$ minimizes log loss and its argmax minimizes symbol error. A trained zero-phase posterior lookup table gives neither the entire angle-dependent likelihood nor a general proof of Bayes optimality. Numerical cell integration is a practical approximation with its own error.

When raw pilot samples are available, multiplication by $e^{-i\omega t_p}$ removes the deterministic drift while preserving the circular noise law. For the already-quantized second data window, replacing the index by its most likely de-rotated index discards the other possible cells. Even a randomized transition reproducing the *marginal* transformed-code law need not preserve its joint law with the answer. Supply the original code to a phase-aware readout, e.g. an interpolated likelihood table with error checks. The sine and cosine of the public angles are valid input features but do not themselves implement this posterior. For arbitrary time order, include all relevant times and known drift/noise parameters. Positive and negative offset contrasts must change the simulated waveform and the corresponding public metadata together.

## Reproducible numerical checks

Run `python <path-to-this-directory>/check_b.py --repo-root <checkout-of-the-pinned-base-or-its-descendant>`. The script reads the original seed-17 weights and asserts the Git blob identifier. Only NumPy and the baseline's learned encoder are required.

* With a fixed independent seed 261126 and 65,536 single windows at each time, exact quarter-turn code rotation passed for every sampled window. All 16 occupied source codes split under each fractional de-rotation. The best deterministic mapping fitted and scored on these same rows necessarily made 29,836 / 65,536 code errors for 0.70 rad and 18,283 / 65,536 for 1.40 rad. These counts measure the impossibility of exact re-encoding on these samples; they do not measure downstream prediction error. At time 4 this is a diagnostic applying the encoder to an analogous window; the Stage 3 pilot is actually delivered as a reference and the third window is withheld.
* In a separate analytic sign check, condition on initial orientation $u_1=0$ only. Integrating the sector probabilities over $B_4$ gives $[0.155117,0.763482,0.069863,0.011538]$ (64-point Gauss-Hermite). A 250,000-draw simulation yielded $[0.156348,0.762116,0.069708,0.011828]$. Reversing the sign predicts sector 3 in place of sector 1. This is a conditional physics check, not a host benchmark.
