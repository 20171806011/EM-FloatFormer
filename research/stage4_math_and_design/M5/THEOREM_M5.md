# Track M5 · One cache for relations, events, and active evidence

Round 1 studies a fixed pre-question cache and a single sealed candidate observation. The mathematical claims specify the observation law they require. The CPU experiment uses matched synthetic IQ. Pluto, CTTC, and the host comparison remain separate tests. [Long-term research plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md) · [Integration plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md)

## 1. Minimal sufficiency for the task union

Let $M$ denote the metadata available to every arm before encoding, including the window addresses, sample clock, calibration, and definitions of eligible observations. Condition on it below. The encoder reads old IQ $Y$ once, producing $Z=f(Y)$ before the question, reference $A$, or candidate values arrive. $U$ denotes the latent physical state, including symbols, phase, gain, noise, and the trajectory model $H,\tau$. Candidate windows are sealed recordings. If they overlap old samples, the joint law must describe that dependence.

**Theorem 1.** For a fixed model and prior, form

$$
\mathcal K(y)=
\left(
\{P(D_q\mid y)\}_{q\in Q1,Q3},\
\{P(D_q\mid y,A=u)\}_{q\in Q2,Q3_{\rm after\ pilot},u},\
\{P(H,\tau\mid y)\}_{q},\
\{P(H,Y_a\mid y)\}_{q,a\in\mathcal C}
\right).
$$

The families range over all permitted addresses, reference contexts, and candidate sets. Take $\tau=\varnothing$ for hypotheses without a jump, and replace $H$ by $(H,\tau)$ in the last term when the design target includes localization. Under a common dominating reference measure with positive conditional densities, identify reference kernels up to common null sets. For singular or deterministic side information, use the supported contexts and corresponding equality of kernels. The equivalence $y\sim y'$ iff their required kernels agree defines the minimal statistic. $\mathcal K$ is a representation of that quotient.

**Proof.** Equal kernels yield the same optimal decisions, candidate predictive distributions, posterior updates, and acquisition values. Conversely, strict propriety of log loss requires each conditional answer distribution to be preserved to attain full-information risk. Preserving the full Stage 6 commitment requires the rival probabilities and conditional predictive distributions, which multiply to give each candidate's joint kernel. Thus the kernels must factor through any exactly sufficient cache. Requiring only the winning action can induce a coarser quotient. More than one prior or readout law requires the joint map over those laws.

If $A,Y_a$ are conditionally independent of old observations given $U$, $P(U\mid y)$ suffices for all these readouts. It can be larger than $\mathcal K$ when distinct latent states give the same answers and predictive mixtures. To preserve all priors and decisions on $U$, the likelihood-ratio criterion identifies $y,y'$ iff $p(y\mid U)/p(y'\mid U)$ is constant in $U$ on common positive support. Comparing all two-state priors gives the necessity. [Kullback and Leibler, 1951](https://doi.org/10.1214/aoms/1177729694)

### 1.1 Exact Gaussian experiment

In window $k$, let $Y_k=m_k(U)s_k+\epsilon_k$, where $\|s_k\|=1$, $m_k=\alpha_k i^{X_k+G_k}e^{i\phi_k}$, and $\epsilon_k\sim{\cal CN}(0,v_k I_{n_k})$ with independent innovations conditional on the state. Define $T_k=s_k^*Y_k$ and $E_k=\|(I-s_ks_k^*)Y_k\|^2$. Then

$$
\log p(Y_k\mid U)=-n_k\log(\pi v_k)-\frac{|T_k-m_k(U)|^2+E_k}{v_k}.
$$

With calibrated covariance, first prewhiten and use the normalized whitened steering vector. For known $v_k$, $E_k$ cancels from every likelihood ratio, including when fading is latent. Thus the matched sums suffice. They are minimal for the entire likelihood experiment when the mean-vector contrasts span both real axes at each address, with independent variation sufficient to span the complete address space. In the log ratio of two observations the only state-dependent term is $2\sum_k\Re[(T_k-T_k')^*m_k(U)]/v_k$. Its constancy in $U$ forces $T_k=T_k'$ under the span condition.

If rival models may change $v_k$, retain $T_k$ and $E_k$. They are minimal for the likelihood experiment under independent identifying mean and inverse-variance contrasts at each address. A single unknown global variance may require only one pooled residual energy. An unknown DC term or spur in a fixed template subspace can require its own complex projection. Equal total residual energies in different nuisance directions can produce different likelihood ratios. Unrestricted covariance alternatives, intra-window CFO, arbitrary waveform forecasts, and windows with correlated noise can increase the sufficient dimension, up to raw IQ in a sufficiently rich model family.

### 1.2 Noisy generalization of the 2-bit versus 4-bit pair

Let $X_1,X_2,G_1$ be independent uniform on $\mathbb Z_4$, with $G_2=G_1+J$, $P(J)=\kappa\delta_0+(1-\kappa){\rm Unif}_4$. One concrete lag law is $\kappa=e^{-d/\ell_c}$. Put $U_k=X_k+G_k$. For independent Gaussian noise, known nonzero gains and $v_k>0$, the window posterior is

$$
p_k(j;t)=\frac{\exp(2\Re[t^*\alpha_k i^j]/v_k)}
{\sum_{l=0}^3\exp(2\Re[t^*\alpha_k i^l]/v_k)}.
$$

Write $u_4$ for the uniform vector and $c_{12}(d_0)=\sum_jp_1(j)p_2(j+d_0)$. The Q1 posterior is $u_4+\kappa(c_{12}-u_4)$. For a known-symbol late pilot $A$, let the pilot channel label $G_p$ have the analogous coherence $\kappa_p=e^{-d_p/\ell_c}$ with $G_1$, with its increment independent of $J$ and the symbols. The pilot gives a posterior $p_p(j;A)$ by the same Gaussian formula. The Q2 posterior is $u_4+\kappa_p(c_{p1}-u_4)$, where $c_{p1}(x)=\sum_j p_p(j;A)p_1(x+j)$.

**Exact Q1-only tradeoff.** Common quarter turns leave the Q1 posterior unchanged and cycle the Q2 posterior for fixed $A$. The old IQ is independent of the pilot under this symmetric submodel because the two data labels are independent and uniform. Every common-rotation-invariant cache, including the exact Q1 posterior, consequently gives $P(X_1\mid Z,A)=u_4$. Its Q2 excess log risk is exactly

$$
\Delta_2=E_{T_1,A}D_{\rm KL}\big(u_4+\kappa_p(c_{p1}-u_4)\,\|\,u_4\big).
$$

This integral gives a gap at the actual gains, old SNR, pilot SNR $\rho_p=|\alpha_p|^2/v_p$, and lag. It is positive for $\kappa_p>0$ and two informative Gaussian legs, since the conditional posterior differs on open sets of positive probability. It is zero for an independent reference or a zero-SNR leg. As $\kappa_p\to0$, it is $(\kappa_p^2/2)E\sum_j(c_{p1}(j)-1/4)^2/(1/4)+O(\kappa_p^3)$. It tends to $\log4$ with noiseless coherent legs. Conditional on calibrated fading gains, average this formula over their law. With hidden gains use the hierarchical likelihood in the kernel. The rotation identity still holds if that likelihood has the common fourfold symmetry. Strictness requires informative phase after marginalization. An independent uniform continuous phase per window can erase symbol-label information at arbitrarily high amplitude.

**Noisy minimality.** For $\kappa,\kappa_p>0$ and the nonzero known gains above, Q1 and Q2 together recover $(T_1,T_2)$ almost surely. Send $A$ along each of four pilot rays to infinity. By continuity, identical Q2 kernels must agree at every finite pilot value and hence in all four limits, where $p_p$ becomes a point mass. This recovers $p_1$. Q1 then recovers $p_2$ by circular deconvolution. With unnormalized weights $(e^a,e^b,e^{-a},e^{-b})$, the first nontrivial Fourier coefficient of $p_1$ vanishes only at $a=b=0$, and the second only at $|a|=|b|$. These exceptions have Gaussian probability zero. Finally, opposite-class probability ratios give both real coordinates of $\alpha_k^*T_k$. The union therefore requires a continuously valued quotient in this subfamily. A single-task optimum need not discard all absolute evidence, so the exact invariant-cache penalty is not a universal penalty for every Q1-optimized design.

In the coherent noiseless limit $D=U_2-U_1$ and $U_1$ are independent uniform four-way variables. For a $b$-bit **pair** cache with a perfect $G_1$ reference,
$$
H(D\mid Z)+H(X_1\mid Z,G_1)
=H(D\mid Z)+H(U_1\mid Z)\ge(4-b)_+\log 2.
$$
This follows by subadditivity and $H(Z)\le b\log2$, and is attained for integer $b\le4$ by retaining $b$ individual bits of $D,U_1$. Q1 alone needs 2 bits for exact recovery and the pair of queries needs 4. More generally, a finite noiseless experiment needs $\lceil\log_2 N_q\rceil$ or $\lceil\log_2 N_\cup\rceil$ bits, where $N$ counts distinct required kernels on supported old observations. These counts also apply to any finite multihypothesis event and forecast experiment. [Noiseless example in the plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md)

## 2. Exact operational rate–risk frontier

Let $\mathcal P_b$ be the measurable partitions of old $Y$ into at most $2^b$ cells, $b=\lfloor LB\rfloor$. If the cache must store exactly $B$ individually addressable bits at each window, restrict to product partitions with at most $2^B$ cells per address. Index bits and variable allocation belong in the ledger. For $P\in\mathcal P_b$, write $Z=P(Y)$. For non-acquisition queries let $D_q$ be the answer and $A_q$ the permitted side information, with $A_q$ empty for Q1 and Stage 5.

$$
\begin{aligned}
R_q(P)&=H(D_q\mid Z,A_q),\\
R_q(P)-R_q(Y)&=I(D_q;Y\mid Z,A_q),\\
R_6(P)&=E_Z\min_{a\in\mathcal C}K_Z(a),&
K_z(a)&=E_{Y_a\mid z}H(P(H\mid z,Y_a)).
\end{aligned}
$$

**Theorem 2.** The upper-closed achievable joint region, allowing limits, is $\overline{\bigcup_{P\in\mathcal P_b}\prod_q[R_q(P),\infty)}$. A single-task optimum is $\inf_{P\in\mathcal P_b}R_q(P)$. A weighted tradeoff minimizes $\sum_q\lambda_q R_q(P)$ over one shared partition. Individual infima can come from different partitions, and weighted sums need not recover every point of a nonconvex frontier.

**Proof.** On each cell and revealed reference, the conditional distribution minimizes log risk. Subtract the full-data risk by the entropy chain rule or integrate the posterior KL in each cell. In Stage 6 the same argument applies to each action separately, followed by the cellwise minimum. For explicit evaluation, set $p(d,a,C)=\int_C p(d,a,y)\,dy$; the Q2 risk is $-\sum_C\sum_d\int p(d,a,C)\log[p(d,a,C)/p(a,C)]\,da$. For Stage 6 perform the same calculation with $p(h,y_a,C)$ for each candidate and take the minimum **inside** the sum over cells. A separate forecast objective can add cross-entropy in a declared observation space. This is an exact finite-rate partition problem. A discretized observation experiment reduces to a finite partition search.

The optimum depends on the full prior and conditional laws. Varying the hypothesis prior or making all rival emissions identical can change the frontier at the same $B,\rho_p,d_p$. Fading, old SNR, covariance and trajectory priors cannot be inferred from those three arguments. In the nondegenerate noisy example, even the single-task Q1 excess is strictly positive at any finite alphabet size $m$. Choose $m+1$ sufficiently separated support points of the continuously varying posterior, and positive-mass disjoint neighborhoods around them. With only $m$ reproduction distributions, one neighborhood must remain at a positive distance from every reproduction, yielding a positive uniform lower bound by Pinsker's inequality.

Define $K_y(a)$ as above using old IQ. With optimal actions $a_Z,a_Y$ from each information set, the active excess decomposes exactly as
$$
R_6(P)-R_6(Y)=I(H;Y\mid Z,a_Z,Y_{a_Z})+E[K_Y(a_Z)-K_Y(a_Y)].
$$
Hold the cache action fixed for the first subtraction, and then compare actions at the full-IQ information set. The first term is compression loss under identical selection, and the second is the regret from the change in selection.

This also characterizes sufficiency when only the terminal Stage 6 log risk is scored. Both terms vanish iff the cache admits a selection that is full-data optimal almost surely and the selected post-reveal posterior factors through the cache and reveal. The full predictive commitment in Theorem 1 preserves more information.

For balanced binary hypotheses and 0–1 loss, the exact discriminability loss is $\frac12[\mathrm{TV}(P_0^Y,P_1^Y)-\mathrm{TV}(P_0^Z,P_1^Z)]$. Bayes accuracy is $(1+\mathrm{TV})/2$, and replacing the IQ densities by their cell masses replaces $\int |p_0-p_1|$ by $\sum_C|\int_C(p_0-p_1)|$. The triangle inequality is strict exactly when a cache cell contains positive mass from both strict signs of the likelihood difference. Hence log risk can increase while accuracy stays equal if within-cell likelihood changes leave the winning hypothesis unchanged.

### 2.1 Continuous trajectories and identifiable events

If $\phi(t+d)-\phi(t)=\omega d+\sigma_\phi\sqrt d\,N\pmod{2\pi}$, the $n$-th circular moment propagates by $e^{-n^2\sigma_\phi^2d/2}e^{in\omega d}$. A jump multiplies the moment by $e^{inJ}$ after $\tau$. An independent uniform reset sets every nonzero circular moment to zero. Combine pilot likelihoods at their time nodes and compute $p(H,\tau\mid Z,A)$ by marginalizing state, slope, diffusion, jump amplitude, gain, and noise under each model. A noise burst can widen a likelihood while leaving its mean unchanged. Under the Gaussian model the residual energy carries additional variance evidence whenever there are residual degrees of freedom.

Two endpoints alone can agree under an unconstrained smooth slope and jump. Unknown uniform QPSK payloads can hide quarter-turn jumps even with noiseless IQ. If the rival old and candidate distributions coincide, no cache or host can distinguish them and their EIG is zero. State these boundaries separately from the performance of a learned reader.

## 3. Audit of s17 and the shared cache format

The released s17 training generator draws coherent unit-energy QPSK with **additive Gaussian noise** at 2 dB. It uses three complex samples with mean magnitude $1/\sqrt3$ and fits to analytic noisy-label posteriors. The b8 name means eight bits for the pair. The turn comes from the sum, with a first-nonzero-sample fallback if that sum is zero. The encoder rotates the three samples before the six-input neural network and four-center assignment. The learned inner label is a codebook index. An angular quantizer cannot in general be assumed to produce that index. Stage 3C shifts Q1 by the difference of turns and Q2/Q3-repeat by the first turn. Its Q2 kernel deduplicates acquisition IDs, ignores known independent references, and is calibrated for its pinned pilot time and SNR. The signed-CFO Stage 3D evaluates s17 outside the encoder's coherent training condition. [Encoder and training](../../matched_cache/learned.py) · [Training driver](../../matched_cache/train.py) · [Paired generator](../../fourphase_paired_questions/paired.py) · [Stage 3C quotient](../../stage3c/G/quotient.py) · [Stage 3C Q2 reader](../../stage3c/A/q2_reader.py) · [Stage 3D data](../../stage3d_qwen_signed_phase/data_stage3d.py)

On Stage 3D G1, physical s17 reaches 64.62% / 0.94113 nat and Full IQ reaches 64.64% / 0.92904 nat. The observed **cache gap** is about 0.01209 nat. The phase host reaches 23.80% / 1.4452 nat, a **readout gap** of 0.5041 nat behind physical s17. This comparison refers to 2 dB Q3-repeat, signed $\omega=\pm0.40$, target diffusion variance 0.04 and the published decoders. It measures NLL and accuracy. The numbers give no CPU latency measurement. For Bayes readers the exact criterion for $\epsilon$-near-sufficiency is $I(D_q;Y\mid Z,A_q)\le\epsilon$. [Stage 3D G1 report](../../stage3d_qwen_signed_phase/REPORT.md)

For a decoded matched sum $\hat T$, a conservative bound makes the precision tradeoff explicit. Let $e_U$ be the error of the surrogate log likelihood up to an additive constant shared across states. Conditional posterior KL to the surrogate is $E_p e_U+\log E_p e^{-e_U}\le(\sup_U e_U-\inf_U e_U)^2/8$ by Hoeffding's lemma. Its expected value upper-bounds the Bayes task loss, by data processing and the optimality of the cell decoder. The Gaussian mean terms give $e_U=2\sum_k \Re[(T_k-\hat T_k)^*m_k(U)]/v_k$ at fixed variance. Coarse rounding can therefore be more costly at higher precision. If angle error is at most $h/2$ and radial error at most $\epsilon_r$ on a bounded unsaturated bin, $|T-\hat T|\le\epsilon_r+|T|h/2$. Saturation and covariance terms must be handled separately.

**What s17 permanently merges.** Opposite means have likelihood ratio $\exp(4\Re[T^*\alpha]/v)$. Two interior points of one s17 cell with distinct projection on $\alpha$ give different likelihood ratios. A strict neural assignment margin ensures an open neighborhood around each point inside the same cell. Gaussian positivity and strict propriety then imply positive Bayes loss when a query separates these posteriors, including an informative phase pilot under the conditions of §1.2. The experiment verifies such a collision with the pinned weights. The same argument holds when an event or forecast contrast varies within a cell. In the variance experiment a circular pure-noise window has a uniform 2-bit turn at every variance, whereas radial or residual energy gives positive class information. s17 can retain some quality information through its learned index, but cannot preserve a continuously varying likelihood ratio. Bounds for the actual Pluto and CTTC acquisitions require their measured calibration and held-out data. The $-3$ to $+24$ dB and $>10$ dB gain variations in this report are synthetic stress conditions.

### 3.1 Bit-exact extension

Retain the original 2-bit turn and 2-bit learned inner index as a prefix. Add canonical phase, a calibrated log radial/reliability field, and residual quality. The Gaussian reference implementation uses $r=|T|/\sqrt{v_{\rm nom}}$, a unit canonical phasor with angle $\arg T-\pi\,\mathrm{turn}/2$ in $[-\pi/4,\pi/4)$, $\log(1+r)$ radial bins, and $E/v_{\rm nom}$ residual bins. The likelihood concentration is $2|\alpha T|/v$ if the gain and variance are known. The radial field alone does not determine both unknown gain and unknown variance. Carry calibrated estimates as metadata or additional bits, or infer them jointly using the cached evidence. Keep original observation units for amplitude forecasts.

| Payload bits/window | Turn | Legacy inner | Canonical angle | Log radial/reliability | Residual quality |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0 | 0 | 0 | 0 |
| 4 | 2 | 2 | 0 | 0 | 0 |
| 8 | 2 | 2 | 2 | 1 | 1 |
| 12 | 2 | 2 | 4 | 3 | 1 |
| 16 | 2 | 2 | 6 | 4 | 2 |
| 24 | 2 | 2 | 8 | 8 | 4 |

Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.

Version the header with time origin/sample rate, template, gain and noise convention, calibration, quantizer edges, format ID, and capture/sample-span mapping. Record whether quality indicates clipping, noise, or a typed nuisance projection such as DC/spur. When required include irregular addresses, masks and per-window calibration bits in the budget. The CPU benchmark holds the address map and nominal amplitude/noise profile common to all readers and counts only the cache payload.

### 3.2 Stage 4→6 connector contract

| Component | Shared physical interface |
|---|---|
| EquivariantEvidenceConnector | Decode complex evidence, time, precision, quality, and masks. Return physical likelihood factors or calibrated candidate probabilities. Transform the reference and trajectory factors consistently under shared or per-window coordinate changes. |
| RefSufficientPool | Accept immutable acquisition ID, association status or probability, time, complex measurement and calibration. Deduplicate repeated renderings. A common latent phase and independent co-temporal Gaussian samples allow a weighted complex sum. Cross-time or correlated observations require likelihood factors/messages at the right nodes. |
| Stage 5 event head | Return a joint posterior over smooth, jump and interval, independent trajectory, and receiver/noise alternatives. Keep conditional state mixtures per rival for forecasting. Report interval coverage as well as maximum-probability times. |
| Stage 6 EIG selector | Consume the posterior, public candidate descriptors, and reveal handle. Commit one candidate ID, prior weights, predictive mixture for each rival, and a receipt. Append one physical factor after reveal and compute the updated law. |

The same numeric modules can serve Ministral 3B and Qwen 4B/9B/27B. Adapt only the host embedding width, special tokens or chat template and language readout calibration. Empirical transfer still requires fitting and testing each adapter. The 9B to Qwen3.8-27B comparison changes generation as well as scale. For LEMMO, report its native instruction-conditioned workflow and a separate matched-access comparison where supported. Use the same old observations, stored/effective precision, reveal allowance, questions and acquisition splits. To attribute a gap to encoding, also match training data, trainable parameters and optimization budget. Report context, latency, old IQ rereads, extra reads and unsupported tasks. LEMMO's published workflow uses Qwen3.5-4B and an instruction-conditioned retrieval stage, with LoRA during joint fine-tuning. [Host plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md) · [LEMMO, Sections 3 and B](https://arxiv.org/html/2611.09476v1)

## 4. One-window Bayes EIG

At a fixed cache $z$, set $\pi_h=P(H=h\mid z)$, $f_{ha}(x)=\int p(Y_a=x\mid u,h,z)P(du\mid z,h)$, and $m_a(x)=\sum_h\pi_h f_{ha}(x)$. Drop $z$ inside the likelihood only under the required conditional independence. Predictor mixtures may have unequal amplitudes, phases, and uncertainty under the rivals.

**Theorem 3.** The exact one-window expected improvement in Bayes log score is

$$
V(a,z)=I(H;Y_a\mid z)
=\sum_h\pi_h\int f_{ha}(x)\log[f_{ha}(x)/m_a(x)]\,dx
=h_b(\pi_0)-E_{m_a}h_b(\pi_0f_{0a}(Y_a)/m_a(Y_a))
$$

for two rivals, with categorical entropy for more than two. It is weighted Jensen–Shannon divergence, zero iff all positive-weight predictions coincide and at most $H(H\mid z)$. Fusion uses $P(H=h\mid z,x)=\pi_h f_{ha}(x)/m_a(x)$. The expected realized gain $\log[f_{Ha}(x)/m_a(x)]$ gives the integral. This proves that the selector minimizing $K_z(a)$ maximizes $V$. For two simple equal-variance complex Gaussian predictions, let $\delta=|\mu_1-\mu_0|^2/v$ and $L=\delta+\sqrt{2\delta}W$, $W\sim N(0,1)$. Then

$$
V=-\pi_0E\log(\pi_0+\pi_1e^{-L})-\pi_1E\log(\pi_1+\pi_0e^{-L}).
$$

Under either true hypothesis, the log likelihood ratio in its direction has the law of $L$, which proves the scalar formula. Composite trajectories use the full predictive mixture; mean separation alone is insufficient. [Lindley, 1956](https://projecteuclid.org/euclid.aoms/1177728069)

Before reveal commit the action and both normalized forecast densities in a fixed observation space, together with prior weights. Log-score the mixture by $-\log m_a(x)$ and, when the generative label is known, its conditional density by $-\log f_{Ha}(x)$. A true-versus-rival score difference is a diagnostic. The mixture log score is strictly proper for the marginal prediction. Separate conditional forecast validation requires labeled or experimentally identified rivals. Use natural acquisitions with observable held-out targets and controlled interventions for labeled rivals. Declare the scored projection. Residuals that are common to the rivals cancel in their posterior update.

**Strict dominance.** For a randomized baseline $r(a\mid z)$, the conditional expected posterior-NLL advantage is exactly $\sum_a r(a\mid z)[V^*(z)-V(a,z)]$. For nearest it is $V^*(z)-V(a_{\rm near},z)$. Each baseline is strictly worse in expectation iff it assigns positive probability to a strictly inferior action on a positive-measure set of caches. Uniform random needs one inferior action there. Nearest must itself be inferior. These statements assume the true predictive law and equal read costs. With a certified uniform evaluation error $\max_a|\hat V(a,z)-V(a,z)|\le\epsilon$, the estimated maximizer incurs at most $2\epsilon$ regret in true EIG, by applying the error bound at both maximizers.

Under 0–1 loss the optimal success probability is $\int\max(\pi_0 f_{0a},\pi_1 f_{1a})=(1+\|\pi_0f_{0a}-\pi_1f_{1a}\|_1)/2$. It can rank actions differently from EIG. A sufficient structural condition for dominance for both losses is Blackwell ordering. If a candidate observation can simulate the other by an $H$-independent garbling, the informative observation can emulate every policy for the garbled one. Data processing gives strict EIG improvement when $H$ and the better observation remain dependent after seeing the garbling. Strict accuracy improvement requires the garbling to mix positive-probability predictive regions on opposite sides of the decision boundary. Two simple equal-variance Gaussian experiments are ordered by standardized mean separation: project onto the difference of means to get a scalar $X=\pm d/2+N(0,1)$; simulate the smaller separation by scaling and adding independent Gaussian noise of the remaining variance. [Blackwell, 1951](https://digicoll.lib.berkeley.edu/record/112749)

- **Change-point split.** Conditional on the jump model, suppose $Y_a$ probes only $S_a=1\{\tau\le t_a\}$ through an identical, binary-input symmetric channel at every candidate. Then $I(\tau;Y_a\mid z,H=\mathrm{jump})=I(S_a;Y_a\mid z,H=\mathrm{jump})$. In the noiseless case it is $h_b(P(\tau\le t_a\mid z,H=\mathrm{jump}))$. For distinct symmetric noisy densities it is strictly concave in the split mass because mixture entropy is strictly concave along the segment. Symmetry puts the maximum at a posterior median. An available split closer to $1/2$ is strictly better. If the objective is to distinguish the model class $H$, use the class-conditional mixtures. The median split is a theorem for this localization channel.
- **Anti-aliasing.** For known pilot symbols, phases at all old times agree for two simple slopes with shared anchored phase at $t_0$ when $\Delta\omega(t_k-t_0)\in2\pi\mathbb Z$. A known-symbol candidate that violates this relation gives positive EIG at positive finite precision and interior weights. With old times $0,2,4$, slope difference $\pi$, and alternatives at 6 and 1, the first candidate remains aliased and the second gives opposite phase. From the anchor at 4, nearest chooses 6. A uniform unknown QPSK symbol at the candidate also masks differences in $(\pi/2)\mathbb Z$, so the candidate needs a known symbol or a cross-window constraint. Posterior phase diffusion and uncertain association can erase otherwise useful separation.
- **Smooth versus jump.** Two trajectories can match at $0,D$ when the smooth slope is $\omega$ and the jump model uses $J$ at $\tau$ and slope $\omega-J/D$. Their phase difference at a candidate $t$ is $J(1\{\tau\le t\}-t/D)$. At fixed calibrated noise, an interior window can have strictly larger $2|\alpha|\left|\sin[J(1\{\tau\le t\}-t/D)/2]\right|/\sqrt v$ than nearest. For simple rivals this gives strict dominance by the Gaussian ordering above. With uncertain $\tau$ and fading, integrate the mixtures.

## 5. CPU verification

[verify_m5.py](verify_m5.py) loads pinned s17 weights and generates three-sample complex windows with two known old pilots and two hidden payload symbols. Q1 reads relative payload phase. Q2 adds a valid late reference at the time origin. The Stage 5 prior gives equal mass to smooth, jump, independent and variance-burst models. It enumerates two smooth slopes, three jump intervals, a reset to independently drawn phase and slope, and a local sixfold noise variance. Old and candidate phases lie on a 128-point circle grid. The simulator uses that finite trajectory law without continuous drift diffusion. Stage 6 separately draws smooth/jump mothers with equal model priors and commits four candidate predictive mixtures before opening the selected recording per policy. A smooth/jump pair can have identical old endpoint means. The model permits post-jump fading, with class-dependent amplitude for some candidates. The maximum old gain swing in the selective fade profile is $20\log_{10}(1.20/0.23)=14.35$ dB, across old SNRs of $-3$ to $+24$ dB.

Each of four profiles contributes 4,096 mothers to the old tasks and 2,048 other mothers to Stage 6. The s17 emission calibration uses 32,768 samples per canonical phase and distinct gain/variance configuration in a profile, an independent seed stream, C4 symmetry, and a half-count per cell. The canonical reader integrates Gaussian polar cell probabilities and uses the gamma CDF for the residual bins. The exact $(T,E)$ log likelihood was checked against a direct calculation from raw IQ over all simulated windows, phases and noise states. Stage 6 quadrature uses the continuously valued candidate matched sum and verifies an independent finer spacing on the first 64 mothers per method and profile. Candidate noise residuals have a common law at fixed address. The JSON keeps true and rival forecast scores, Bayes posterior NLL and accuracy, expected and observed NLL gains, action frequencies, paired intervals, and examples with the predictive distributions recorded before reveal. Because candidate noise scales and units affect differential log density, the raw observation NLL should be interpreted with the candidate and compared to the rival on the same reveal. [Full results](results_m5.json)

### 5.1 Achieved rate–risk points

All values are nats, pooled with equal profile weight. Stage 5 joint includes the event and jump interval. Stage 6 is the posterior model-label NLL under maximum EIG. The table evaluates specified encoder/readers. The exact optimized partition frontier of Theorem 2 is a different quantity.

| Bits/window | Q1 | Q2 | Stage 5 joint | Stage 6 post-reveal |
|---:|---:|---:|---:|---:|
| 2 | 1.1453 | 0.8616 | 1.5396 | 0.4619 |
| 4 (s17) | 0.9970 | 0.7966 | 1.2788 | 0.3677 |
| 8 | 0.9648 | 0.7856 | 1.0615 | 0.3614 |
| 12 | 0.9479 | 0.7536 | 0.9944 | 0.3545 |
| 16 | 0.9456 | 0.7493 | 0.9669 | 0.3544 |
| Continuous canonical / Full IQ | 0.9438 | 0.7425 | 0.8934 | 0.3538 |

The 16-bit gain over s17 is 0.3119 nat for event and location, paired pointwise 95% interval $[0.3023,0.3214]$. The remaining gap to Full IQ is 0.0735 nat. For the conditional smooth-versus-jump task the pooled NLL is 0.5240 (s17), 0.5136 (16 bit), and 0.5131 (Full IQ). For localization given a true jump it is 0.9711, 0.9382, and 0.9379, respectively. The recovery varies with regime. At +24 dB conditional binary NLL falls from 0.0300 with s17 to nearly zero with 16 bits. Under selective fading it is 0.6808 versus 0.6820, and both the 16-bit and continuous results are around 0.68. The observed low-SNR conditional localization remains near $\log3$ for all readers. Thus the larger joint event gain should not be read as an equally large binary-discrimination gain.

The actual s17 encoder emits code 1 for two real, in-cell matched sums of magnitude 0.0200 and 0.5833 in the 2 dB Gaussian witness. The exact four-label posterior changes by 0.4804 total variation. The JSON also includes two IQ vectors sharing their matched sum and orthogonal energy that give distinct s17 codes. This establishes the need for an explicit legacy prefix even with a continuous canonical Gaussian cache.

### 5.2 Sealed one-window selection

With the 16-bit cache, EIG, random, and nearest attain pooled NLLs 0.3544, 0.4497 and 0.4772, and accuracies 81.18%, 73.34% and 71.63%. The comparison below conditions on the same ambiguous-case stratum for all methods, defined for evaluation only by the old Full IQ smooth posterior in $[0.1,0.9]$. All 2,048 cases qualify in each of the three listed profiles. The +24 dB profile has no qualifying cases and shows a ceiling tie between policies at 16 bits.

| Ambiguous profile | Max EIG NLL / accuracy | Random NLL / accuracy | Nearest NLL / accuracy |
|---|---:|---:|---:|
| −3 dB | 0.6374 / 61.82% | 0.6742 / 55.96% | 0.6928 / 53.03% |
| +2 dB | 0.5434 / 72.27% | 0.6107 / 65.77% | 0.6445 / 63.28% |
| Selective fade | 0.2369 / 90.63% | 0.5141 / 71.63% | 0.5714 / 70.21% |

In selective fade, max EIG gains 0.2772 nat over random, paired pointwise 95% interval $[0.2500,0.3043]$, and 0.3345 nat over nearest, interval $[0.3037,0.3652]$. Expected EIG is 0.4542 nat and realized gain is 0.4395 nat. Their realized-minus-expected difference has a paired interval $[-0.0405,0.0112]$. The already committed true and rival predictive observation NLLs are 0.5289 and 4.0492 in the same observation units. The largest checked change in EIG under grid refinement is $1.77\times 10^{-6}$ nat.

In the selective-fade profile time 4 wins all 2,048 selections. At −3 and +2 dB it wins 2,024 and 2,012, respectively. This is strong evidence for the choice of window geometry against the two requested baselines in this simulation. A matched fixed-time policy at time 4 exactly ties the active policy in selective fade. Use candidate geometries with competing informative windows and a fixed-best-on-development control to isolate the value of record-specific adaptation in Round 2.

### 5.3 Rerun and interpretation

Run from the repository root with the pinned NumPy and SciPy environment. The sandbox run used copies of the pre-existing package cache inside M5 and the same pins.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py
~~~

Pointwise intervals are paired by mother and conditional on this specified simulator and approximate s17 calibration. The cache risks use common nominal gain/noise metadata, unlike an acquisition where these quantities must be estimated. Independent signal-session data are needed to establish held-out calibration, transfer, and latency. The blind simulation enforces one release per policy within a common process. A field implementation should isolate the sealed candidate store from the predictor and record the commit and reveal separately.

## 6. Round 2 handoff

- **M1.** Freeze the factored physical evidence interface, 16-bit candidate encoding, and exact legacy prefix. Preserve C4 transformation laws on candidate probabilities, and transform dynamics and reference factors consistently. Use matched old IQ to measure the separate cache and host readout gaps. Keep the continuous physical decoder as an oracle.
- **M3.** Freeze acquisition ID, timestamp, association probability, calibration, and noise-dependence fields. Verify idempotence for reused sources. Compare co-temporal pooling and cross-time messages against the joint likelihood. Keep the posterior mixture available for pre-reveal forecasts.
- **M4.** Use session- or acquisition-disjoint tuples. Calibrate SNR, gain and quality from allowed old data or a separate calibration split. Stratify natural, injected, and independent acquisitions, including low SNR and high fading. Test typed DC/spur projections when a scalar residual fails. Use a separately labeled oracle-calibration arm where needed.
- **M6.** Freeze risk margins by query and capture stratum, candidate-access rules, and held-out proof of sealing before training larger hosts. Compare 12 and 16 bits with all storage and read costs accounted for. Include the continuous ceiling, ambiguous-case coverage, a fixed-best development baseline, forecast calibration, step-by-step I/O and latency. Preserve unidentifiable and tied cases in the report.

## 7. Round 2 · Exact value of adapting to the recording

Let $M$ be everything visible to a baseline that may choose a constant window within each public stratum. The candidate set $\mathcal C(M)$ is finite and nonempty. Define the utility vector $U(Z,M)$ by $U_a(Z,M)=I(H;Y_a\mid Z=z,M=m)$, and let $S(U)=\arg\max_{a\in\mathcal C(M)}U_a$. All utilities are expected under the committed predictive experiment. A static policy may use $M$ and independent randomness. Conditional on $Z,M$, both static and adaptive policies use the same Bayes update on the selected reveal.

**Theorem 4.** The adaptive advantage over the population-best static policy in expected post-reveal NLL is

$$
\Delta_M =
E\!\left[\max_a U_a-\max_a E(U_a\mid M)\right]
=E\!\left[\min_a E(\max_b U_b-U_a\mid M)\right].
$$

This gain is strictly positive iff on a positive-measure set of public strata every candidate is strictly suboptimal with positive conditional probability. Equivalently, no $M$-measurable fixed rule can select a member of $S(U)$ almost surely. Proof follows by subtracting the two expected posterior entropies, which have a common old-data entropy term. The nonnegative conditional regrets have zero expectation precisely when at least one of their finitely many entries vanishes. Thus the result also compares against every dev-fitted static policy that operates without blind-record cache information.

There is an exact conditional mutual-information characterization. Minimize $I(A;U\mid M)$ over all randomized EIG-maximizing action rules, meaning $A\in S(U)$ almost surely. Then

$$
\Delta_M>0
\quad\Longleftrightarrow\quad
\inf_{A\in S(U)} I(A;U\mid M)>0.
$$

If a static maximizer exists, use it to attain zero conditional mutual information. For the converse put $p_m=\min_a P(a\notin S(U)\mid M=m)$. The joint law of $(A,U)$ gives the event $A\notin S(U)$ probability zero. The product of its conditional marginals gives that event probability at least $p_m$, for every admissible action law. Conditional Pinsker therefore gives $I(A;U\mid M)\ge 2E[p_M^2]$. This also handles arbitrary tie-breaking. With a unique maximizer almost surely, the infimum is simply $I(a^*;U\mid M)=H(a^*\mid M)$, and positivity means that the winning window varies within a public stratum. Mutual information with the realized noisy observation or a single candidate's utility is not the criterion. For example, two equally likely utility vectors $(1,1,0)$ and $(1,0,1)$ allow a tie-breaking rule with positive mutual information and zero gain over the first constant.

For accuracy replace $U_a$ by $V_a=E[\max_h P(H=h\mid Z,M,Y_a)\mid Z,M]$ and maximize the new utility. Max EIG gives the corresponding guarantee when it also maximizes $V_a$, for example with Blackwell-ordered simple Gaussian candidates and distinct standardized mean separations. There is no general accuracy dominance from EIG or its mutual information condition. For a balanced binary label, a reveal-perfectly-with-probability-.4 erasure channel has EIG $.4\log 2$ and accuracy .70. A binary symmetric channel with crossover .2 has smaller EIG $\log 2-h_b(.2)$ and higher accuracy .80. Round 2 tests accuracy directly for the composite models below.

### 7.1 Multi-geometry construction

Each public family has five old windows, four sealed candidates, and 16 equiprobable physical states ordered by hidden geometry $g=0,1,2,3$ and then receive turn $r=0,1,2,3$. The two equally likely rivals have exactly the same old-window distributions conditional on state. Thus every old cache, including full IQ, has hypothesis probabilities $(.5,.5)$. The old evidence can locate the informative window while leaving the hypothesis unresolved.

| Public family | Old and candidate geometry | What the old cache estimates | Rival prediction at the informative address |
|---|---|---|---|
| Jump interval | Old times $0,2,4,6,8$. Candidate times $1,3,5,7$. A $\pi$ step occurs in interval $g$. | Which two old windows bracket the step, with a common unknown receive turn. | H0 steps at $t_g-0.5$ and H1 at $t_g+0.5$. Only candidate $g$ lies between the two jumps. |
| Selective fade | Four independent old scout measurements share the respective candidate gains. The fifth old pilot anchors the receive phase. | Which scout saw the clear channel. Its gain is 1.35 and the other gains are .12 to .24. Candidate gain is withheld until reveal. | H0 and H1 have opposite received phases. All candidates have the same thermal variance, so the clear channel has the strongest contrast. |
| CFO alias | Old times $210j$, $j=0,\ldots,4$. Candidate times $210/p_a$ for $p_a=(2,3,5,7)$. | The phase progression of the hidden slope catalogue modulo the old grid. | H1 adds $2\pi/p_g$ to the H0 slope. It aliases on all old times and all candidates except $g$. |

Observations are known-symbol matched pilots with conditional independent proper complex thermal noise. The old decoder integrates the actual v1 phase/radial cells. Residual quality is ancillary in these three old likelihoods. The s17 inner tag is retained and counted in the payload. The decoder uses the phase/radial fields to compute state weights, without the hidden geometry or realized candidate amplitude. Each state with fully resolved geometry has a different strictly informative candidate. With noisy old observations, EIG integrates the remaining state uncertainty. The same catalogue and conditional likelihood power both forecasting and the one-reveal pool. Generalizing the catalogue to continuous trajectories or real RF is an external validation step.

The 2,048 dev and 4,096 test mothers per family have separate seeded streams. Dev chooses one fixed candidate per public family by its mean realized post-reveal NLL and freezes the choice. Dev mean predicted EIG is saved as a diagnostic. The stronger retrospective oracle chooses the best constant per family **separately for each endpoint** using test outcomes. The evaluator pairs separate policy replicas on the same sealed bank. Each replica commits its action and predictive laws before receiving its one matched observation. Uniform random, nearest to the last old timestamp, and fixed-dev all share the same downstream posterior update.

The CPU gate requires positive simultaneous lower confidence endpoints for NLL and accuracy against the three operational controls and the retrospective oracle. The paired stratified t intervals use Bonferroni over 30 elementary contrasts. Six are pooled operational-control endpoint comparisons. Twenty-four cover every constant within every family and endpoint. Taking the minimum fixed-policy bound within each family and averaging covers the oracle's retrospective selection. These are large-sample intervals for independently simulated mothers, with equal family weights. For real data, perform inference at independent capture/session units. A joint superiority claim is an intersection-union test across required endpoints, with the maximum endpoint p-value. Use Holm across separately claimed host comparisons, and fix margins and calibration on a disjoint dev split. CPU verification uses zero superiority margins and a frozen physical policy. It provides no population-of-retrains inference. [M2 statistical design](../M2/THEOREM_M2.md)

| Family | Test selections for IDs 0, 1, 2, 3 | Adaptive NLL | Adaptive accuracy |
|---|---|---:|---:|
| Jump interval | 1,001 / 1,017 / 1,066 / 1,012 | 0.04742 | 98.29% |
| Selective fade | 941 / 1,037 / 1,061 / 1,057 | 0.00324 | 99.88% |
| CFO alias | 1,042 / 1,005 / 1,020 / 1,029 | 0.09401 | 96.44% |

Each family has positive simultaneous lower bounds on both endpoints against every constant. Pooled expected EIG is 0.64184 nat and the realized NLL gain from the pre-reveal prior is 0.64492 nat. The largest observed grid-refinement change is below $2.0\times10^{-6}$ nat on the registered quadrature diagnostic subset. [Round 2 results](results_m5.json) · [Executive summary](reply_r2.md)

## 8. Round 2 wire lock · m5-c16-v1-le

Serialize each record as exactly one unsigned little-endian 16-bit word $w$. The bit numbers refer to its integer representation.

| Bits | Field | Encoding |
|---:|---|---|
| 15–14 | turn | Historical receive quarter-turn, $0\ldots3$. |
| 13–12 | s17_inner | Historical learned codebook index, $0\ldots3$. |
| 11–6 | canon_angle | 64 uniform cells on $[-\pi/4,\pi/4)$ in the turn coordinate. Decode a representative at the bin center or integrate its cell in the likelihood. |
| 5–2 | log_radial | Code 0 is erasure. Codes 1–15 represent increasing cells in $\log(1+|T|/\sqrt{\nu_{\rm cal}})$. The interior edges are $j\log(25)/16$ for $j=2,\dots,15$. An exact edge tie stays below the edge. The highest cell saturates. |
| 1–0 | resid_quality | Bin of $E/\nu_{\rm cal}$ with interior cutpoints $.75,\sqrt{12},16$. Use an observation law calibrated for the actual residual degrees of freedom. |

The erasure predicate is exactly $((w\gg2)\mathbin{\&}15)=0$. A writer sets the entire lower 12 bits to zero when the matched sum is zero or the window is unavailable. Use the actual legacy nibble whenever a historical triplet exists, including the encoder's zero-sum fallback. For an unavailable triplet the prefix is the zero placeholder. The old 4-bit reader needs its original eligibility mask to distinguish that case. A byte fixture is integer hexadecimal A80C, serialized as bytes 0C A8. The suffix may be discarded to export the exact legacy nibble $w\gg12$. A common physical quarter-turn increments only the top two bits modulo four for a valid window and rotates all physical likelihoods in the common receive frame. If erasure is informative in a physical acquisition, include its likelihood explicitly.

Version 1 keeps M4's field widths and all higher radial edges. The existing M4 Stage 4 shard keeps an out-of-band validity byte and uses all 16 radial codes. Its v0-to-v1 conversion maps valid old radial code 0 to 1, leaves other valid codes alone, and clears the 12-bit tail for unavailable or zero-pivot rows. Thus v1 merges the two lowest v0 cells and the rest of the packet needs no re-quantization. Phase eligibility must already include the availability and zero-pivot check at the causal CPU boundary. The Round 1 rates in §5 use v0. Version 1 is the self-contained wire used in the new multi-geometry run. The M1 v0 decoder must explicitly select the v1 interpretation when given a v1 word, including its erasure mask and merged lowest radial cell. [M4 shard wire contract](../M4/THEOREM_M4.md#24-cpu-shard-wire-contract--m4real_dual_cachev2)

### 8.1 One CPU pass and one innovation path

Let $R-\hat d$ denote the allowed raw samples after causal DC removal, $q$ the steering vector, $K$ the calibrated noise covariance, and $\hat a$ the positive gain scale. Under the calibrated scalar Gaussian model,

$$
T=\frac{q^*K^{-1}(R-\hat d)}{\hat a\, q^*K^{-1}q},
\qquad \nu_{\rm cal}=(\hat a^2q^*K^{-1}q)^{-1}.
$$

Compute physical residual quality from the measured whitened orthogonal projection before synthesizing a triplet. One normalized convention for $d_\perp>0$ orthogonal complex degrees of freedom sets $E/\nu_{\rm cal}$ to twice the whitened residual energy per degree of freedom. This agrees with the tested three-sample convention with $d_\perp=2$. A scalar-only frontend masks the quality input through its calibration contract. Count any observation-dependent per-window calibration or quality-availability data in the ledger. A fixed header may carry capture-level units, calibration, clocks, pilot symbols and template IDs. Declare calibrated units for all candidate densities and retain permitted amplitude-scale information when the query requires absolute amplitude.

| Stage / owner | Exact boundary |
|---|---|
| 4 / M4 causal SDR extraction | Compute timing, gain, DC, $T$, physical residual and $K$ from authorized stencils. Form the SNR-preserving R3 triplet $eT+B\eta$, $e=\mathbf1/\sqrt3$, with the frozen independent ancillary stream. Encode its legacy nibble once and append phase, log radial and **measured** residual quality. Synthetic R3 energy carries zero acquisition information. Export the 4-bit view as a projection. Log raw support, clocks and IDs that contributed to calibration. The 20 dB 0/8192 versus 19.89% SER result is M4's forced-normalization surrogate control. |
| 4–5 / M1 EquivariantEvidenceConnector | Consume either legacy or v1 physical fields with an explicit format mask. Return calibrated time/phase likelihoods in one receive frame. Keep receive rotation distinct from transmit relabeling. Physical receive rotations preserve Q1/Q2 TX labels and permute Q3 receive sectors. Stage 5 passes the conditional state mixture per rival, including geometry, phase, ordered jump times, shared CFO and uncertainty. |
| 4–6 / M3 RefSufficientPool | Accept immutable acquisition ID, association status, timestamp in the common clock, $K_{\rm ref}$, matched $T$, and known pilot root, with provenance. Prior IDs from the causal ledger are already conditioned. Identical reappearances are idempotent, conflicting content is rejected, and known independent references are neutral. For an overlapping raw stencil use the conditional innovation law. |
| 6 / selector and scorer | Request candidate-specific predictive mixtures from the phase-context state. Commit the one selected law under each rival. Pass the selected reveal through M3 using the same likelihood, clock, association, data units and ID. A reveal already in the causal ledger violates the sealed selection contract. |

Choose a physical anchor at or before the earliest modeled clock. For nonnegative lags $d_i$, the unwrapped phase covariance prior for shared Gaussian CFO uncertainty and Brownian diffusion is

$$
C_{ij}=P_{\phi_0}+C_{\phi_0,\omega}(d_i+d_j)
+v_\omega d_i d_j+\sigma_\phi^2\min(d_i,d_j).
$$

The mean is $\hat\phi_0+\hat\omega d_i$. Condition the joint law on permitted old evidence before evaluating predictions. Treat shared CFO as one latent variable. If a correlated raw pair $(R_o,R_a)$ is Gaussian conditional on state $u$, the fresh conditional factor has mean $\mu_a+K_{ao}K_{oo}^{-1}(R_o-\mu_o)$ and covariance $K_{aa}-K_{ao}K_{oo}^{-1}K_{oa}$. The old law normalizes the joint exactly once. If only quantized $Z$ retains old support, use $p(R_a\mid Z,u)=\int p(R_a\mid r_o,u)p(dr_o\mid Z,u)$ or an exact equivalent innovation state. Substituting a point estimate of the old observation into the Schur formula is generally an approximation. In Round 2 the candidate thermal innovations are conditionally independent. The vectorized scalar pool checks pre/post likelihood identity, exact deduplication, and neutral prior/distractor IDs. The verifier cross-checks selected actual reveals through M3's shared pool. A separate correlated Gaussian fixture checks the Schur expression against the joint-minus-old log density. The latent CFO/diffusion and hardware noise covariances play distinct roles. [M1 Round 2 brief](../discussions/M1/prompt_r2.md) · [M3 shared pool](../M3/ref_pool.py) · [M4 causal frontend](../M4/THEOREM_M4.md)

## 9. Stage 6 common benchmark contract · m5-stage6-blind-v1

The exact shared UTF-8 question is:

> 只根据已缓存的观测和候选窗口说明，在四个窗口中选择一个补充参考。揭示前提交窗口编号、H0与H1的先验概率及各自对该窗口的预测分布。只读取所选窗口一次，随后给出H0与H1的后验概率。

1. **Input.** The old recording enters once before the question. Supply five immutable v1 words in timestamp order (80 payload bits), the registered public family and hypothesis definitions in §7.1, and four candidate descriptors in ID order 0–3. Public metadata include the catalogue of possible means, timestamps, known roots, and calibrated thermal variances. Actual geometry, phase, candidate values and measured candidate gain remain sealed. Use identical rows and feature bytes for each host and baseline. Use an independent training namespace and the published dev/test split, with disjoint physical captures for subsequent SDR work. Deviations require a new manifest. A real experiment can publish a different state catalogue, version and hash while keeping the field and scoring semantics.
2. **Before reveal.** Submit candidate_id, prior of length 2, and conditional_weights of shape 2 by 16. The mixture weights use the ordered physical states in §7.1. Values are finite and nonnegative. The scorer validates and normalizes rows whose sums differ from 1 by at most $10^{-6}$ and treats other rows as protocol failures. The fixed basis defines normalized densities in common complex matched-sum units. The broker binds the chosen ID, old-cache digest, catalogue hash and normalized output to an opaque evaluation-row ID in a SHA-256 receipt. Host serialization uses length-prefixed fields in that order, UTF-8 identifiers and little-endian binary64 numeric entries. The CPU replay uses a sealed policy handle in the process. The deployed broker must isolate unselected I/Q from model processes.
3. **Reveal and output.** Release exactly one immutable acquisition to the shared RefPool. Return posterior of length 2 in H0,H1 order. Score NLL and argmax correctness with exact ties assigned to H0. Score the committed prior, mixture observation density, and true-hypothesis conditional density when labels are available. A malformed row contributes zero accuracy and infinite NLL to the confirmatory endpoint and triggers a gate failure. Report diagnostic forecast means and failure counts separately. Evaluate selector gains with a downstream decoder shared by all actions. Report end-to-end host comparisons as a separate claim.
4. **Gate.** Run uniform random with a fixed evaluator seed, nearest to the last old timestamp, fixed-best-on-dev by public family, and a physical max-EIG reference. Lowest candidate ID breaks ties. For fixed-best report the dev lock and retrospective best-constant sensitivity. For each claimed host adaptive gain, require strictly positive paired NLL and accuracy improvements against all three operational controls. Use an intersection-union test across required endpoints, mother-level pairs on independent synthetic captures, and capture/session-level pairs for held-out real observations. Apply Holm across separate host claims. A single trained adapter per host supports a frozen-connector conditional claim.

The selector contrast fixes the representation at 16 bits. A separate 16-bit versus 4-bit S17 claim fixes the reveal action and reader calibration opportunity, pairs the same recordings, and tests the two endpoints on that matched action. Preserve the valid legacy prefix in that contrast. Keep each claim's adaptive and representation effects attributable to its own intervention. [M2 Stage 6 gate](../M2/THEOREM_M2.md#8-stage-6-compatibility)

| Host | Frozen matched arm |
|---|---|
| Qwen3.5-4B | Stage 3D revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Fit a host-specific input/output adapter using shared training rows. [Pinned revision](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) |
| Qwen3.5-9B | Revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Same packet, physical heads, query, hypotheses and read budget. [Pinned revision](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) |
| Qwen3.8-27B | Revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0. Treat transfer as a size and generation change. [Revision history](https://huggingface.co/Qwen/Qwen3.8-27B/commits/main) |
| LEMMO | Implement a matched-access adapter through the same cached evidence and output boundary once the checkpoint is obtained and hashed. Code revision ebcc1ac9a6d3a658e62eb73b77b326e3dd2dfc0a. Report native recognition and description under their own raw-IQ access ledger. Mark unsupported matched tasks until the adapter qualifies. [LEMMO paper](https://arxiv.org/html/2611.09476v1) · [Pinned code](https://github.com/N1rv4n4/LEMMO/tree/ebcc1ac9a6d3a658e62eb73b77b326e3dd2dfc0a) |

Match the posterior target, training and validation rows, precision, optimizer budget and trainable-parameter cap for attribution. Use the common M1 ceiling of 2.2 million trainable connector parameters. Record actual counts and steps. Freeze each tokenizer, prompt adapter, weights, calibration and decoding configuration before test. Record total storage including per-record metadata, old IQ reads, extra reads, tokens, latency and memory. The contract reuses the evidence representation and physical modules across hosts. Host qualification, the matched LEMMO adapter and held-out SDR measurements remain to be run.
