# A finite cache strictly increases the Bayes error of the delayed absolute query

Consider the continuous observation model in [the AWGN study](../continuous/README.md). Data and pilot signal-to-noise ratios may differ. Write them as ρd and ρp, both strictly positive and finite. The common data rotation G and transmitted symbols X₁,X₂ are independent, uniform four-phase labels. Let Uⱼ = Xⱼ + G and sᵤ = iᵘ. Before the query, the receiver sees Yⱼ = s(Uⱼ) + Nⱼ. Afterwards it receives P = s(G) exp(iD) + Nₚ. The noises are independent circular complex Gaussians with mean squared magnitudes 1/ρd,1/ρd,1/ρp. The independent angle D is normal with mean zero and finite variance v ≥ 0, including v = 0. The phase is constant across the two data samples. The question is to recover X₁ under symbol 0–1 loss.

**Theorem.** Fix ρd, ρp, v as above and a finite M ≥ 1. There exists δ(M,ρd,ρp,v) > 0 such that every pre-pilot data encoder with at most M states, followed by any decoder using the state and the unquantized pilot, has absolute-symbol error at least the full-I/Q Bayes error plus δ. The conclusion includes private randomness and independent shared randomness. The first relative question may be dropped entirely. Conversely, finite caches can approach the full-I/Q Bayes error arbitrarily closely as their size grows.

### Proof

**1. Factorization.** U₁,U₂,G are mutually independent uniform labels, so P is independent of the two data observations. Write aᵤ(y) for the posterior of U₁ given y = Y₁ and b₍g₎(p) for the posterior of G given p. For X₁ = U₁ − G its posterior given full I/Q and pilot is

    Qₓ(y,p) = Σg aₓ₊g(y) b₍g₎(p).

The second data observation has no effect on this posterior. It can still enter the tested encoder. A full-I/Q decision with access to (Y₁,Y₂,P) obtains conditional success maxₓ Qₓ. Choosing another label x incurs excess error maxₓ′ Qₓ′ − Qₓ.

**2. Distinct data samples can require different pilot-dependent decisions.** Let m(y) = Σᵤ aᵤ(y) iᵘ. Write y = α+iβ and κd=2ρd. The posterior weights are proportional to exp(κd Re(y i⁻ᵘ)), giving

    m(y) = [sinh(κd α) + i sinh(κd β)] / [cosh(κd α) + cosh(κd β)].

In particular, with α>0 held fixed, the phase φ(y)=arg m(y) can take any value strictly between 0 and π/4 by varying β between 0 and α. Put κp=2ρp and consider a received pilot p=t exp(iθ). Apart from factors common to all g, its likelihood is the expectation of exp[κp t Re(exp(iθ) i⁻ᵍ exp(−iD))]. Expanding at t=0 gives the constant 1 and linear coefficient κp exp(−v/2) Re(exp(iθ) i⁻ᵍ). The remainder is O(t²) uniformly in θ and g because the cosine has magnitude at most one. The four linear coefficients sum to zero. After normalization and convolution,

    Qₓ(y,t exp(iθ)) = 1/4
        + t [κp exp(−v/2)/4] Re(exp(−iθ) i⁻ˣ m(y)) + O(t²).

All scores are continuous in (y,p). For nonzero m, the first-order coefficient has a unique maximizing label when φ(y)−θ lies strictly inside the sector of angular radius π/4 around that label. Given two data values with phases 0<φⱼ<φₖ<π/4, choose θ=(φⱼ+φₖ)/2−π/4. At the first value the unique linear-order winner is label 0, and at the second it is label 1. The linear gaps are positive for finite v. At a sufficiently small **positive** t these two winners therefore have positive exact posterior margins over every other label. By continuity this holds on open neighborhoods of each data point and a common open pilot disk.

**3. Quantifiers and positive-probability events.** Choose M+1 first-data centers whose m-phases lie at distinct points in (0,π/4). For each pair j<k apply step 2. There are finitely many such pairs, so choose disjoint data disks Bⱼ small enough for all pairwise conclusions to hold simultaneously, and pilot disks Cⱼₖ on which the two corresponding winners differ. Shrink them so all margins are bounded below by a common η>0. The Gaussian mixture densities are strictly positive on every disk, hence

    b = minⱼ Pr(Y₁∈Bⱼ) > 0,       c = minⱼ<k Pr(P∈Cⱼₖ) > 0.

First condition on any independent shared random seed. For each Bⱼ, some state zⱼ satisfies Pr(Y₁∈Bⱼ, Z=zⱼ) ≥ b/M, allowing an arbitrary randomized encoder of both observations. Two disks j,k must select the same state z. Pilot and encoder variables are independent. For any p in their common pilot disk, the decoder of (z,p) therefore sees the same input on subpopulations of masses at least b/M from each disk, yet their unique optimal labels differ. Even a randomized decoder has error above full-I/Q Bayes by at least η times the smaller mass in the integral over data: its probabilities of choosing those two distinct labels sum to at most one. Integrating over Cⱼₖ gives excess risk at least ηbc/M. This bound is independent of the encoder and the conditioned seed, so averaging proves the claim. The proof gives existence of a positive bound for fixed parameters. It does not estimate a useful numerical magnitude or its sharp scaling.

**4. Approximation.** Quantize each of the four aᵤ(Y₁) to a grid with coordinate error at most ε using finitely many labels. At pilot receipt compute the four convolved scores with the quantized values. Since b(p) is a probability vector, every score changes by at most ε, and an argmax of the approximate scores has conditional excess risk at most 2ε. Taking ε to zero proves the second assertion. The relative-query MAP label can additionally be recorded in two bits if both decisions are required.

The ideal four-bit result uses four discrete noiseless rotation labels. At finite positive SNR, neighboring data phases can prescribe different decisions for a later continuous pilot. The proof applies to every finite wrapped-normal drift variance, however small its first harmonic. In the independent-reference limit the first symbol is uniform even given full I/Q, so a one-state cache attains the same absolute error of 3/4. A noiseless pilot with zero drift also lies outside the hypotheses; retaining the MAP label of U₁ then suffices for the absolute query.

The script `verify_continuous_converse.py` illustrates strict margins at finite nonzero pilot amplitude in the zero-drift case. Open sets and positive probability for every finite budget follow from the proof.
