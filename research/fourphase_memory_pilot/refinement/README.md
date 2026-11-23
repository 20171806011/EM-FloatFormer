# Finite precision for the delayed-pilot decision

Use the [continuous waveform and posterior](../continuous/README.md). Store the two-bit relative MAP label $H$ before the query. Also store $b$ bits for each of two posterior sign probabilities formed from the first received sample. The total message budget is $B=2+2b$ bits. This construction preserves the full-I/Q first decision exactly. For the second decision its excess symbol-error risk over full I/Q is at most $2^{1-b}$, uniformly over the pilot gap, pilot observation, and data observation under this model.

**Proof.** Rotate $Y_1$ and the constellation by $+\pi/4$. The noiseless real and imaginary coordinates are independent uniform signs times $1/\sqrt{2}$. Their Gaussian likelihoods factor. In label order $0,1,2,3$, the coordinate signs are respectively $(++),(-+),(--),(+-)$. Write $u$ and $v$ for the posterior positive-sign probabilities of the first and second coordinates. They determine all four first-symbol label probabilities $a_1$ by the independent Bernoulli product law.

Partition $[0,1]$ into $K=2^b$ equally wide cells and include 1 in the last cell. Store each cell index and reconstruct its midpoint. Each sign probability changes by at most $1/(2K)$. For any label, its probability is a product of two factors in $[0,1]$. Changing them one at a time bounds the absolute probability error by $1/K$. The posterior of $X_1$ is the convolution of this vector with the received pilot's normalized phase likelihood. This convex mixing preserves the same bound per label.

Let $p$ and $\tilde p$ be the full and reconstructed posteriors of $X_1$. Take $x^*$ and $\hat x$ to be their respective maxima (any fixed rule resolves ties). The conditional excess symbol-error risk satisfies

$$
0\le p(x^*)-p(\hat x)
\le [p(x^*)-\tilde p(x^*)]+[\tilde p(\hat x)-p(\hat x)]
\le 2/K .
$$

Expectation gives the stated risk guarantee. The worst-case bound applies for any $b\ge1$; the actual risk can be much smaller. The midpoint decoder is an explicit achievable rule. A decoder using the exact conditional posterior given each quantizer cell could also use this rule and hence has optimal risk at least as good.

## Reproducible prediction

Run from this directory with Python 3 and NumPy:

~~~sh
OPENBLAS_NUM_THREADS=1 python run_refinement.py
~~~

The independent seed 124479 produces 40,000 frames per SNR at gap 4 and step standard deviation 0.25 radians. The simulator shares the input draws across budgets and computes $p(x^*)-p(\hat x)$ for each frame using the [reference likelihood](../continuous/run.py). The CSV reports mean conditional regret, its Monte Carlo standard error, its largest sampled value, the theoretical upper bound, and the corresponding mean full and compressed conditional risks. The seed differs from the broader gap sweep. The midpoint decoder also differs from the exactly calibrated sector decoder at four bits.

| Total bits | Excess SER at 2 dB (percentage points) | Excess SER at 8 dB (percentage points) |
|---:|---:|---:|
| 4 | 0.47625 | 0.02056 |
| 6 | 0.14450 | 0.00609 |
| 8 | 0.04566 | 0.00237 |
| 10 | 0.01344 | 0.00106 |
| 12 | 0.00298 | 0.00018 |

The script checks the sign-posterior factorization, the per-label posterior error bound, and the conditional risk bounds. The monotone mean trend in the table pertains to these paired draws. The proof bounds every decoder in this specified sequence separately.
