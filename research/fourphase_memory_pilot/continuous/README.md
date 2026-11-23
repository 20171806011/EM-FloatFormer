# A delayed noisy reference for a continuous QPSK waveform

Two queries use symbol-error loss. The data pair lies inside one constant-phase interval. Phase diffusion acts **between that interval and the reference**. Within-pair diffusion would also change the first answer and requires a different observation law.

## Observation and posterior

Let the exponents $X_1,X_2,G$ be independent uniform elements of $\mathbb Z_4$. Write $s_u=i^u$, $U_j=X_j+G$, and $R=X_2-X_1$; arithmetic on these labels is modulo four. Independent circular complex Gaussian noises have mean zero and mean squared magnitude $1/\rho$. All three observations have unit signal energy and the same $E_s/N_0=\rho$. Before either question form a fixed message from the pair $Y_j=s_{U_j}+N_j$. Discard the waveform for bounded-message trials. Query 1 asks for $R$. Query 2 asks for $X_1$ after revealing the received sample of a known unit pilot,
$P=s_G e^{iD}+N_p$, with $D\sim N(0,v)$, $v=L\sigma^2$, independent of all symbols and noises. The sine and cosine wrap this random walk displacement onto the circle. An independent-reference control uses a uniform $D$ on the circle. Pilot receipt is free for all decoders and occurs strictly after Query 1.

The change of variables $(X_1,X_2,G)\mapsto(U_1,U_2,G)$ preserves the uniform product law. Therefore the posterior after pilot receipt factorizes into the data-symbol distributions $a_1,a_2$ and a phase distribution $b$. Explicitly,

$$
a_j(u)\propto \exp\{2\rho\,\Re(Y_j\overline{s_u})\},\qquad
b(g)\propto \mathbb E_D\exp\{2\rho\,\Re(P\overline{s_g}e^{-iD})\}.
$$

Normalize each vector over its four labels. The marginal posteriors for the questions are

$$
p_R(r)=\sum_u a_1(u)a_2(u+r),\qquad
p_X(x)=\sum_g b(g)a_1(x+g).
$$

Marginal MAP separately minimizes each symbol error. Before the pilot $b(g)=1/4$ and $X_1$ is uniform given even the full waveform. The independent-reference control gives this same posterior after the pilot. Before observing any pilot the first posterior already has its final value, since both received data symbols share $G$.

**Gap ordering.** For $v_2\ge v_1$, rotating a pilot generated at $v_1$ by an independent Gaussian angle of variance $v_2-v_1$ gives the pilot channel at $v_2$. Circular independent noise keeps its conditional distribution under this rotation. A decoder receiving the shorter-gap pilot could perform this randomization itself. Thus the optimal second-query risk is nondecreasing in gap for any fixed data cache and the full waveform. As $v\to\infty$, the wrapped Gaussian becomes uniform, since its nonconstant circular Fourier coefficients decay as $e^{-k^2v/2}$, giving second-query error $3/4$. With $\rho\to\infty$ and zero drift, both full-IQ decisions are exact. At any finite positive SNR, the four likelihoods have overlapping support and even full I/Q admits no guarantee of zero symbol error.

## Specific nested caches

Let $H=\arg\max_r p_R(r)$ and $J=\arg\max_u\Re(Y_1\overline{s_u})$. Fixed-order ties give measurable actions. At finite positive SNR, ties in these two decisions have probability zero. For $H$, each pairwise posterior difference is a real analytic function of the continuous pair and is nonzero for some pair of large-amplitude samples aligned to favor one of the two labels. Its zero set has Lebesgue measure zero.

| Bits | Stored before query | First answer | Posterior used after reference |
|---|---|---|---|
| 2 | $H$ | $H$ | Uniform on the four $X_1$ labels |
| 3 | $H,\lfloor J/2\rfloor$ | $H$ | Convolve the two-sector first-symbol posterior with $b$ as above |
| 4 | $H,J$ | $H$ | Convolve the four-sector first-symbol posterior with $b$ as above |

Condition on $U_1,Y_1,G,P$. The distribution of $Y_2$ remains invariant under multiplication by $i$, and this multiplication advances $H$ by one modulo four almost surely. Thus $H$ is conditionally uniform, independent of the entire conditioning tuple. In particular $H$ gives no additional second-query information for any of these three caches; it does encode the optimal decision for the first query. This is a property of the specified relative-only cache. A different 2-bit cache could instead prioritize the later absolute query.

For $J$ the error kernel on cyclic offsets is $t=((1-q)^2,q(1-q),q^2,q(1-q))$, $q=\tfrac12\operatorname{erfc}(\sqrt{\rho/2})$. Rotate axes by 45 degrees: the two coordinate signs of a QPSK point each flip independently with probability $q$. Since $U_1$ is uniform, $\Pr(U_1=u\mid J=j)=t_{j-u}$. For the 3-bit cache each sector in the reported pair is equally likely. Average the two corresponding posteriors. These are exact decoder posteriors for these cache designs under the stated model. The 4-bit construction keeps the first readout optimal while discarding some amplitude and sub-sector phase evidence used to combine the first sample with an uncertain pilot. The finite-SNR global minimum over encoders of each budget remains open.

## Numerical check

Run from this directory with Python 3 and NumPy:

```sh
OPENBLAS_NUM_THREADS=1 python run.py
```

The default seed is 20261123. Each of the 12 conditions uses the same 60,000 independently drawn latent triples and base noise samples, for paired comparisons across conditions. The drift standard deviation per step is 0.25 radians. Gap zero uses the exact ordinary pilot likelihood. At positive finite gaps the implementation integrates the phase likelihood on 128 equispaced circular nodes, summing wrapped-normal images past 10 standard deviations and stabilizing the AWGN likelihood by a common exponential shift. The maximum pilot posterior change on the first 2,048 samples per condition after doubling the mesh was below $5\times10^{-16}$. This is a numerical convergence check at the tabulated settings. A separate 64-state sum checks both full posteriors on 16 draws per condition. A rotation check on every frame verifies the relative encoder changes as specified.

The CSV lists symbol error fractions, individual pointwise 95% Wilson intervals, and a pointwise paired normal interval for the mean hard-4-bit minus full-IQ error. A negative estimate of this difference can occur by finite-sample variation. The 2-bit second-query risk is recorded analytically to avoid sampling tie-breaking noise. In the independent condition other decoders always return label zero for the tied absolute query, giving the same realized sample error.

At 8 dB, with the same first-query SER of 2.372% throughout, full-IQ second-query SER rose from 2.273% at gap zero to 18.270%, 45.420%, and 69.090% at gaps 4, 16, and 64. At 2 dB and gap 4, preserving full I/Q reduced second-query SER from 42.278% for the 4-bit cache to 41.927%, a paired difference of 0.352 percentage points (95% interval 0.160–0.544 points). Both have identical first decisions on every frame. These are synthetic operating points; the interval quantifies Monte Carlo uncertainty.
