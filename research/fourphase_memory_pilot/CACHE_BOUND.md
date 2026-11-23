# A bound for continuous I/Q when the first answer must remain optimal

Use the common-data-phase, independent uniform QPSK payloads and independent circular AWGN of the [continuous study](continuous/README.md). The pilot is observed only after the cache is frozen. Let H be the full-data maximum-posterior relative label for the first question, A the transmitted first symbol, and P the later pilot observation. The following applies whenever H is uniform and independent of the pair (A,P), and A is uniform independently of P. It allows an arbitrary encoder into M states, including one trained using both tasks.

**Claim.** Let F be the first answer decoded from that state. The best accuracy S for recovering A later satisfies

$$ S \le \min\{S_{\rm full},\ M/16+\Pr(F\ne H)\}. $$

In particular, if the full-data Bayes relative decision is unique almost surely and the encoder preserves its exact first-query risk, then a two-bit cache gives absolute accuracy exactly 1/4 even with the delayed pilot (with optimal decoding). Three bits give an upper bound of 1/2. These statements apply at finite data SNR. Whether the three-bit bound is attained depends on the observation model.

**Why the symmetry holds.** Conditional on the first payload, its observation, the initial phase and the pilot, the second data observation has a law invariant under multiplication by i. That multiplication cyclically permutes the four relative posterior probabilities and rotates their unique maximizer. Conditional H is therefore uniform, so it is independent of (A,P). At positive finite SNR, ties are null: differences of posterior scores before normalization are real-analytic functions of the four real data coordinates. For each pair of labels one can concentrate the likelihood on the first of those labels by taking two large received amplitudes along corresponding constellation directions; hence the differences are not identically zero. Their zero sets have Lebesgue measure zero under the AWGN density.

**Proof.** Fix protocol randomness independent of the observations. Each state z supplies one first answer f(z) and one later answer a(z,p). On the event that F=H and the later answer is correct, at least one of the M events H=f(z), A=a(z,P) must hold. Conditional on P each such event has probability 1/16. The union bound gives the second term after adding disagreement probability. Averaging over independent randomness gives the same inequality for randomized protocols. A readout of the full data and pilot can emulate the complete protocol, which gives the other upper bound. For a Bayes-optimal first answer, almost-sure uniqueness gives F=H almost surely. A constant guess already attains the resulting two-bit limit.

**Near-optimal first answers.** Let m(Y) be the gap between the largest and second-largest full-data relative posterior probabilities, and let ε be the excess symbol error of F on the first question. Conditional on the data, a disagreement with H costs at least m(Y) times its probability. Thus for every t>0,

$$ \Pr(F\ne H)\le \Pr(m(Y)\le t)+\varepsilon/t . $$

For two bits this bounds the possible follow-up improvement over guessing by the same quantity. A representation may deliberately give up some first-question accuracy on ambiguous samples. In finite runs a small observed difference from full-data first-question accuracy alone does not certify exact optimality.

The bound separates cache limits from the physical information limit. An independent pilot yields S_full=1/4 at every memory size. With an informative pilot, full-I/Q accuracy can exceed 1/2 and this inequality already excludes every three-bit cache that preserves the first Bayes decision. The discrete optimal frontier and the continuous achievable caches have separate proofs.
