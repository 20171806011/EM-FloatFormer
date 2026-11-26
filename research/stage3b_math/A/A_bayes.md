# Bayes-optimal reading of a fixed 8-bit waveform cache under known carrier offset

## Observation and estimands

Use the single-mother law of Stage 3 commit [69fbd29](https://github.com/20171806011/EM-FloatFormer/tree/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache). Window centers are $t_1=0,t_2=h,t_3=h+\ell$. Each cache index has 16 values. Each data window contains three complex samples. Conditioning information $I$ comprises the public condition and question, the two frozen indices $(z_1,z_2)$, and the references revealed by query time. The receiver makes a per-mother decision. Pooling other mothers to decode a balanced test block would change the problem.

Let $G,X_1,X_2$ be independent uniform integers modulo four. Write $U=X_1+G$ and $V=X_2+G$ modulo four. Then $(U,V,G)$ is also an independent uniform triple. Put $b=\pi/2$. In the continuous CFO condition, $\Phi_t=\omega t+\sigma W_t$, with $W_t$ a standard Brownian motion and $\Phi_0=0$. Conditioned on apparent phase $\theta$, each sample is $e^{i\theta}/\sqrt{3}$ plus independent circular complex Gaussian noise of variance $1/\rho$. For fixed frozen encoder $E$ define its **cell likelihood**

$$
F_z(\theta)=\Pr(E(Y)=z\mid\theta).
$$

This is a six-real-dimensional Gaussian integral over the preimage of $z$ for the learned encoder. Define $S_k(\theta)$ as the probability that the matched sum $\sum Y_n/\sqrt3$ falls in sector $k$. For example $S_0(\theta)=\Phi_{\rm N}(a\cos(\theta+\pi/4))\Phi_{\rm N}(a\sin(\theta+\pi/4))$, where $a=\sqrt{2\rho}$ and $\Phi_{\rm N}$ is the standard normal CDF. Other sectors are quarter-turn shifts.

## Matched-information theorem

For any path over the sorted unique observation times, form the following unnormalized measure, then normalize it over $u,v,g$ and the shared path:

$$
dM=\tfrac1{64}F_{z_1}(bu)F_{z_2}(bv+\Phi_h)\,L_R(g,\Phi)\,dP_{\omega,\sigma}(\Phi).
$$

If the measurement interface reveals the pilot samples, for distinct valid acquisitions $j$, with matched sums $s_j$ at times $p_j$, their Gaussian likelihood is proportional to

$$
L_R(g,\Phi)=\prod_j \exp\{2\rho_j\operatorname{Re}(s_j e^{-i(bg+\Phi_{p_j})})\}.
$$

A literal duplicate with the same ID contributes once. Two independently sampled pilots with the same physical time and state have two independent noise draws and one shared latent phase. An independent-channel reference contributes a factor independent of the target and cancels. Before any valid pilot, use $L_R=1$. The host implementation rounds public pilot features to float32. For exact finite-precision matching replace $L_R$ by the likelihood of the observed quantization cell in the pilot-feature map (and count numerical metadata at its exposed precision). The no-pilot Q1 and Q3 calculations below use only exact cache indices and the public configured condition.

With $\mathbb E_M$ denoting expectation under the normalized measure:

| Query | Optimal distribution on $k=0,1,2,3$ |
|---|---|
| Q1, transmitted relative phase | $\mathbb E_M[\mathbf1\{v-u=k\bmod4\}]$ |
| Q2, original first symbol | $\mathbb E_M[\mathbf1\{u-g=k\bmod4\}]$ |
| Q3, noisy third-window sector with retransmission | $\mathbb E_M[S_k(bu+\Phi_{h+\ell})]$ |
| Q3, noisy third-window sector with independent uniform symbol | $1/4$ for every $k$ |

**Proof.** Apply independence of window noise conditional on the path and the three symbols, grouping acquisition copies as above. The change of variables from $(X_1,X_2,G)$ to $(U,V,G)$ preserves the uniform law. The resulting joint density of observations and latent variables is proportional to $dM$. Marginalize the indicators for Q1 and Q2. Conditional on the path and $U$, the repeated third transmission has a new independent noise draw, hence the kernel $S$. Averaging $S_k$ over the four quarter turns of a fresh uniform symbol gives $1/4$ pointwise for every path and $g$. Before an informative pilot, $G$ remains uniform and independent of $U$, so Q2 is uniform. This proof also applies to discrete phase walks by replacing the path measure with a finite sum.

For a categorical target, write $p_k(I)$ for the corresponding posterior. The minimum misclassification risk is $1-\mathbb E\max_k p_k(I)$, attained by any MAP rule. The minimum expected NLL is $H(K\mid I)$, attained by emitting the posterior. For another distribution $q$, excess NLL is $\mathbb E\,D_{\rm KL}(p\|q)$ by subtracting the conditional cross entropies. Conditional 0–1 excess for choosing $d$ is $\max_k p_k-p_d$. These are lower bounds for the specified cache and permitted information. Optimizing the encoder itself or increasing its budget is a different optimization.

## Where the published direct comparator approximates the posterior

The [baseline table readout](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/analytic.py) uses a static four-class lookup table. For Q1 under continuous drift it mixes quarter-turn permutations of this table using the probability mass of $\Phi_h$ in each sector. The likelihood for an offset strictly inside a sector generally varies continuously with the offset. For Q3 before a reference it integrates the third-phase kernel against the *prior* drift law and the first cached index. The exact path posterior also includes $\bar F_{z_2}(\Phi_h)=\frac14\sum_vF_{z_2}(bv+\Phi_h)$. Although this factor is quarter-turn-periodic, it need not be constant. It updates the predictive distribution of $\Phi_{h+\ell}$ when phase fluctuations are shared. The same factor can update Q2 with a pilot correlated with $\Phi_h$. Q3 simplifies when the path is deterministic or when the second index provides no information about that path. For perfect coherent conditions, a table matched to the current SNR and exact population emission probabilities is sufficient for the standard no-pilot Q1 and Q3. Finite learned calibration tables are estimates even in this case.

## Numerical check on the actual C4 condition

Use $h=\ell=2$, $\omega=0.35$ rad/step, $\sigma=0.10$ rad/$\sqrt{\rm step}$ and 2 dB. Each phase increment is independently $N(0.7,0.02)$. For the original hand_b8 the rotated matched sum has independent Gaussian coordinates. Each emission probability is a product of two differences of normal CDFs at its original, fixed bin edges. Sum over **all 256 pairs** of indices and integrate the two increments with Gauss–Hermite quadrature. Orders 24 and 36 agree within $3\times10^{-12}$ in each reported risk.

| Fixed hand_b8 population | Phase-resolved MAP accuracy | Stage 3 direct accuracy | Phase-resolved NLL | Stage 3 direct NLL |
|---|---:|---:|---:|---:|
| Q1 relative phase | 56.6183% | 48.2662% | 0.96071912 | 1.03917377 |
| Q3 repeated third sector | 63.3991% | 63.3991% | 0.95717757 | 0.95718133 |

This gives a counterexample to exactness of the old direct rule on the original simulator and a fixed published 8-bit encoder. Q1 gains **8.35 percentage points**; the Q3 correction here is visible primarily in probabilities (maximum absolute change 0.00221).

For the actual learned equivariant_b8_s17 checkpoint, estimate $F$ by self-normalized Gaussian importance sampling. Each integration seed draws 60,000 independent noise windows at cardinal phase zero. The encoder's exact quarter-turn symmetry supplies the other three proposal components. Compute the likelihood ratio to their equally weighted mixture using the matched sum. The emission integrals for this encoder are numerical estimates. Two runs use seeds 41117 and 41129, with order-20 Gaussian quadrature. The corresponding integrated Q1 accuracies are 60.66% and 60.57% for the phase-resolved rule, compared with 49.86% and 50.03% for the original rule. Independent forward data below test the resulting decisions without reusing integration draws.

| Frozen learned checkpoint, 131,072 independent new mothers, seed 79561 | Phase-resolved reader | Stage 3 direct reader | Paired difference |
|---|---:|---:|---:|
| Q1 relative phase | 60.6079% | 50.2304% | +10.3775 pp (paired SE 0.1406 pp) |
| Q3 repeated third sector | 63.6215% | 63.6215% | 0 different decisions on these mothers |

Both integration seeds give the same achieved scores on that one forward data set. These are direct-reader simulations, separate from the earlier 128-mother host benchmark. They do not determine either host's Q3 accuracy. If improving Q1 is the goal, tune the reader against phase-resolved cell likelihoods. For this Q3 condition the original and phase-resolved direct decisions already agree in the forward check, making host transfer the more diagnostic intervention.

Run with Python, NumPy and SciPy from a checkout containing the pinned Stage 3 weights:

~~~sh
python path/to/check_a.py
python path/to/check_a.py --matched-dir research/matched_cache --per-basis 60000
~~~
