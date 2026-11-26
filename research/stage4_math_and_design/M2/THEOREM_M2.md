# Track M2 | Stage 3D score forensics and the Stage 4 gate

Round 1. Losses and information are in nats. Positive paired differences favor phase, so the accuracy contrast is phase minus zero and the NLL contrast is zero minus phase. Each saved mother carries its signs and rotations together. Each of the three connector seeds has equal weight. All calibration fitted on blind labels is retrospective.

## 1. Which Bayes risk is being estimated?

For a population law \(P\), write \(p(c)=P(Y=\cdot\mid C=c)\) and \(L(q)=\mathbb E[-\log q_Y(C)]\). With finite support and positive model probabilities,

\[
L(q)=H(Y\mid C)+\mathbb E_C D_{\rm KL}\!\left(p(C)\Vert q(C)\right).
\]

This follows by adding and subtracting \(-\log p_Y(C)\) inside the conditional expectation. It is the Bregman decomposition for negative Shannon entropy. If there is a finer observation \(X\) and \(C\) is a function of \(X\), then

\[
L(q)=H(Y\mid X)+I(Y;X\mid C)+\mathbb E_C D_{\rm KL}(p(C)\Vert q(C)).
\]

The population entropy, the information lost by coarsening, and the conditional forecast error depend on the named observation. In Q3, \(C=(S,Z)\) is the physical/told CFO sign and the shared S17 code pair at a fixed condition. Both arms are functions of \(C\) and the fitted seed \(J\). Under the empirical law \(J\) is uniform and independent of \((C,Y)\). Therefore \(H(Y\mid C,J)=H(Y\mid C)\). The same Bayes term cancels exactly in the arm contrast. In particular the Q3 input-resolution difference is zero. Improving a forecast can still use existing information more effectively.

The verifier uses three distinct reference laws.

- **Empirical signed-code posterior.** Entropy of blind counts pooled over the prescribed three rotations, separately by \((S,Z)\). It gives an exact replay identity and an in-sample Bayes risk for that empirical law. Estimation and scoring reuse the labels. Sparse codes and especially H make the plug-in conditional MI optimistic as population evidence.
- **S17 direct readout.** A physical predictive baseline using encoder representatives and known condition. Its realized loss is scored on the same mothers. The code does not prove that integrating the population distribution over a quantizer cell equals scoring the representative.
- **Known-parameter full IQ.** A finer physical reference. Its mean soft entropy and realized NLL differ in finite samples. The identity \(L_{\rm realized}(q)=\overline{H(r)}+\overline{D_{\rm KL}(r\Vert q)}+\overline{\sum_y(r_y-1_{Y=y})\log q_y}\) holds with the sign of the last term as written, for any saved reference \(r\). The last term has population conditional mean zero if \(r\) is the true posterior on the observation set. The JSON reports it for both references.

The empirical CFO information is \(I(Y;S\mid Z)=H(Y\mid Z)-H(Y\mid S,Z)\). The code counts supply both conditional entropies. This is the information available to a Bayes decoder with the sign metadata, independent of which connector is used. Additional diagnostics use the score vector \(Q\) as a covariate. Then \(L(Q)=H(Y)-I(Y;Q)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). With the fixed feature \(A\) equal to the argmax set of \(Q\), the identity is \(L(Q)=H(Y\mid A)-I(Y;Q\mid A)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). Argmax resolution reports a coarse, model-dependent predictive association. Exact-float score-vector groups can identify a code or even a pilot row, so their empirical resolution can be high for a poor decoder. Neither diagnostic identifies a phase-slot mechanism.

### Scalar calibration and marginal bias

For logits \(\ell_i\), inverse temperature \(\beta\ge 0\), and empirical weights \(w_i\), let

\[
f(\beta)=\sum_i w_i\big[\log\sum_j e^{\beta \ell_{ij}}-\beta\ell_{iY_i}\big].
\]

Then \(f'=\sum_i w_i(\mathbb E_{q_\beta(\cdot\mid i)}\ell_{ij}-\ell_{iY_i})\) and \(f''=\sum_i w_i\operatorname{Var}_{q_\beta(\cdot\mid i)}\ell_{ij}\ge0\). If \(f'(0)\ge0\), \(\beta^*=0\) and the optimal scalar forecast is the uniform vector. Otherwise solve \(f'(\beta^*)=0\) for an interior optimum when present. The gap at the lock is exactly

\[
f(\beta_d)-f(\beta^*)=
\mathbb E_i D_{\rm KL}(q_{\beta^*}(\cdot\mid i)\Vert q_{\beta_d}(\cdot\mid i))
+(\beta_d-\beta^*)f'(\beta^*).
\]

The boundary derivative term generally remains at \(T^*=\infty\). A finite positive \(T\) preserves each row's ranking, so ECE and NLL can change while accuracy stays fixed. At infinity the four-way tie awards fractional accuracy \(1/4\). Reported oracle fits are label-optimal scalar temperatures, separately for each arm and seed. The JSON also fits a single common raw temperature per arm and cell. ECE uses the same pooled ten top-confidence bins as Stage 3D. ECE is neither proper nor an additive component of log risk.

For any fixed predictor \(q\), define two joints \(P(c,y)=w(c)p_y(c)\) and \(\widetilde Q(c,y)=w(c)q_y(c)\), where \(c\) includes seed when fitting multiple connectors. The chain rule in reverse order yields

\[
\mathbb E_C D_{\rm KL}(p\Vert q)
=D_{\rm KL}(\bar y\Vert\bar q)
+\mathbb E_{Y\sim\bar y}D_{\rm KL}\big(P(C\mid Y)\Vert\widetilde Q(C\mid Y)\big).
\]

The marginal term measures global prior bias. The second term measures conditional allocation error, encompassing misranking, miscalibration, and other errors. A learned intercept adjustment need not realize the marginal term as a standalone reduction in NLL.

### The reported additive path

Let \(q_d\) use the locked temperatures, \(q_*\) use the post-hoc fitted temperatures and \(Pq_*\) denote the arithmetic projection. For the empirical Q3 design the reported identity is

\[
L(q_d)=H(Y\mid S,Z)+[L(q_d)-L(q_*)]+[L(q_*)-L(Pq_*)]
+D_{\rm KL}(\bar y\Vert\overline{Pq_*})
+\mathbb E_Y D_{\rm KL}(P(C\mid Y)\Vert\widetilde Q_{Pq_*}(C\mid Y)).
\]

Take zero minus phase termwise. Empirical \(H(Y\mid S,Z)\) cancels, and the conditional mismatch difference may be signed. This path specifies the order of the two interventions. Temperature and group averaging do not commute. At locked \(T\), the JSON additionally gives the raw marginal KL, the arithmetic group gap, and its complete-orbit counterpart. In the actual design the projection bracket includes a distributional correction and can have either sign. A causal allocation of the gain to phase slots and an orthogonal temperature/projection/prior ANOVA do not follow from the log-loss identity.

For Q2 each control has one physical sign, and the host score depends on row-specific references. The file records selected matched reference summaries and physical predictions but has no counterfactual sign or full rotated forecast orbit for each reference acquisition. \(H(Y\mid Z)\) from the no-pilot code is reported as an empirical anchor. The exact Q2 path on every control uses the Work A soft reference \(r\) and writes the locked loss as \(\overline{H(r)}+\text{temperature gap}+\text{reference marginal KL}+\text{reference conditional KL}+\text{realized label residual}\). The last three terms are evaluated at \(q_*\). Raw-duplicate and deduplicated controls share the same Work A physical reader under logical acquisition deduplication. A row-ID-only empirical posterior is the point-mass label and gives a tautological zero entropy. Full-vector forecast grouping has a related finite-sample problem. A Q2 CFO MI or group penalty would require additional crossed scores.

## 2. Group averaging and the top-1 loss

Let \(R_gq(x)=\rho(g)^{-1}q(gx)\) for a label permutation representation. Assume the joint law of \((X,Y)\) is invariant under \((x,y)\mapsto(gx,\rho(g)y)\) and all evaluated probabilities are positive. Then \(L(R_gq)=L(q)\). For \(m=Pq=|G|^{-1}\sum_g R_gq\), define \(J_y(x)=\log m_y(x)-|G|^{-1}\sum_g\log (R_gq)_y(x)\). Pointwise \(J_y\ge0\) by strict concavity of \(\log\). Hence

\[
L(q)-L(Pq)=\mathbb E_X\sum_y p_y(X)J_y(X)\ge0.
\]

Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \(L(q)-|G|^{-1}\sum_g L(R_gq)\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.

The proposed \(|G|^{-1}\sum_g D_{\rm KL}(Pq\Vert R_gq)\) is \(\sum_y m_y J_y\) pointwise. It corresponds to self-weighting by the pool. It equals the log-loss gain if \(\mathbb E[(p-m)\cdot J]=0\), in particular if \(p=m\). It is not a general equality even under exact invariance. For an invariant binary example, take two equiprobable observations interchanged by a group of order two with identity label action and \(p=(1,0)\) on both. The forecasts are \((0.8,0.2)\) and \((0.4,0.6)\). The pool is \((0.6,0.4)\), and \(J_1=\log(0.6/\sqrt{0.32})\), \(J_2=\log(0.4/\sqrt{0.12})\) differ. The loss gain equals \(J_1\). The reverse-KL mean equals \(0.6J_1+0.4J_2\).

For geometric pooling let \(h_y=\exp(|G|^{-1}\sum_g\log(R_gq)_y)\), \(Z_h=\sum_y h_y\), and \(q_{\rm geo}=h/Z_h\). Componentwise AM–GM gives \(Z_h\le1\), and

\[
L(q_{\rm geo})=|G|^{-1}\sum_g L(R_gq)+\mathbb E_X\log Z_h.
\]

Thus both pools weakly improve invariant-design log risk over the raw model. Their ordering depends on the label law and the forecasts. Normalization makes the geometric pool the minimizer of \(\sum_g D_{\rm KL}(r\Vert R_gq)\) over \(r\). It is equivariant, and applying it again returns the same distribution. The arithmetic pool minimizes \(\sum_g D_{\rm KL}(R_gq\Vert r)\) and is also equivariant and idempotent.

For each target \(y\), the closed top-1 region is the intersection of halfspaces \(r_y-r_k\ge0\) over \(k\ne y\). For the arithmetic path, the exact margin to competitor \(k\) at the barycenter is the mean of aligned margins. A unique raw correct decision is lost iff a mean margin is negative or hits a tie boundary. A unique correct decision is gained iff all mean margins are positive after a raw error or tie. The fractional-tie hit function at a row is

\[
h_y(r)=\frac{1[\min_{k\ne y}(r_y-r_k)\ge0]}{|\arg\max_j r_j|}.
\]

Over a target distribution, strict accuracy decline is equivalent to \(\mathbb E[(h_y(q)-h_y(Pq))_+]>\mathbb E[(h_y(Pq)-h_y(q))_+]\), including ties. At population level this is the same as \(\mathbb E_X[p(X)\cdot(d(q)-d(Pq))]>0\), where \(d\) is uniform on the maximizer set. For geometric pooling the corresponding competitor margins are averages of \(\log r_y-\log r_k\). Log score depends on the probability assigned to the true class, while top-1 depends on which side of the decision planes the pool lies.

## 3. Stage 4 frozen statistical gate

A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.

1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.

### Exact type-I control and power under a balanced Gaussian model

For an oriented block contrast \(D_{bs}\) let \(D_{bs}=\mu+u_b+\nu_s+e_{bs}\), with independent Gaussian families of variances \(\sigma_u^2,\sigma_\nu^2,\sigma_e^2\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \(m\) mothers in a block have common variance \(v\) and exchangeable within-block correlation \(\rho\), a block mean contributes \(v[1+(m-1)\rho]/m\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.

For \(B\) blocks and \(S\) seeds,

\[
V=\operatorname{Var}(\bar D)=\sigma_u^2/B+\sigma_\nu^2/S+\sigma_e^2/(BS).
\]

Orthogonal row, column, and residual contrasts yield mutually independent ANOVA terms \(MS_B/(\sigma_e^2+S\sigma_u^2)\sim\chi^2_{B-1}/(B-1)\), \(MS_S/(\sigma_e^2+B\sigma_\nu^2)\sim\chi^2_{S-1}/(S-1)\), and \(MS_E/\sigma_e^2\sim\chi^2_{(B-1)(S-1)}/[(B-1)(S-1)]\), independent of \(\bar D\). The corresponding method-of-moments component estimates are \((MS_B-MS_E)/S,(MS_S-MS_E)/B,MS_E\). The reported planning plug-in truncates negative estimates to zero and retains the raw estimates. \(MS_S(S-1)/(B\chi^2_{S-1,0.05})\) is a conservative 95% upper bound for \(\sigma_\nu^2\). Three seeds give only two independent seed degrees of freedom. Large \(B\) cannot remove the \(\sigma_\nu^2/S\) floor.

For the **frozen seeds**, set \(v_F=\sigma_u^2+\sigma_e^2/S\). The ordinary one-sided block t-test has exact boundary size \(\alpha\) for the fixed-seed mean and power \(1-F_{\mathrm{nct}}(t_{B-1,1-\alpha};B-1,d\sqrt{B/v_F})\) at true excess \(d=\mu-\delta\) above the margin, with the mean interpreted conditionally. If one applies this t-test to a population-of-seeds claim, the cutoff for the exact unconditional noncentral t is rescaled by \(\sqrt{(v_F/B)/V}\); the nominal type-I error can then be badly inflated. [SciPy noncentral t definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)

For **random seeds**, a finite-level test with unknown Gaussian variance components uses
\[
 W=(MS_B+MS_S)/(BS)=A X+C Y,\quad
 A=(\sigma_u^2+\sigma_e^2/S)/B,\quad C=(\sigma_\nu^2+\sigma_e^2/B)/S,
\]
where \(X,Y\) are the independent normalized row and column chi-squares. Reject when \(\bar D-\delta>k\sqrt W\) with \(k=t_{\min(B-1,S-1),1-\alpha}\). At the boundary the exact rejection probability, also the power formula at excess \(d\), is
\[
\pi(d)=\mathbb E_{X,Y}\overline\Phi\!\left(\frac{k\sqrt{A X+C Y}-d}{\sqrt V}\right).
\]
To prove size control, note \(A+C=V+\sigma_e^2/(BS)\ge V\) and \(x\mapsto\overline\Phi(k\sqrt x)\) decreases and is convex for \(x>0\). Jensen implies \(\pi(0)\le\frac{A}{A+C}\Pr(t_{B-1}>k)+\frac{C}{A+C}\Pr(t_{S-1}>k)\le\alpha\). The supremum \(\alpha\) is attained as the smaller-df variance component dominates. With known component ratios use the exact nuisance-specific mixture cutoff. The unconstrained method-of-moments estimate \((MS_B+MS_S-MS_E)/(BS)\) can be negative and has a signed chi-square mixture law. Plugging it into a t distribution would not be an exact test.

For real data under the same Gaussian model with \(J_0\) independent sessions of \(m\) nested clusters and separate session and cluster effects, the session means have
\[
V=\sigma_{\rm session}^2/J_0+\sigma_{\rm cluster}^2/(J_0m)+\sigma_\nu^2/S+
\sigma_e^2/(J_0mS).
\]
Use the same formulas with \(B=J_0\), \(\sigma_u^2=\sigma_{\rm session}^2+\sigma_{\rm cluster}^2/m\), and \(\sigma_e^2\) replaced by \(\sigma_e^2/m\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \(B\) equal-weight cluster means in \([l,h]\), reject when the empirical excess exceeds \((h-l)\sqrt{\log(1/\alpha)/(2B)}\). For fixed weights \(a_b\) summing to one, replace \(1/B\) by \(\sum_b a_b^2\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.

For endpoint nulls \(H_{0j}\) and a joint claim, the null is their union. Valid p-values give \(p_{\rm IUT}=\max_j p_j\), since with at least one true null the event \(p_{\rm IUT}\le a\) is a subset of rejection of that true null. The exact joint rejection probability is the joint tail of the endpoint statistics, equal to the product of marginal powers only under independence. Its universal lower bound under the alternative is \(1-\sum_j(1-\pi_j)\). Across separately reportable claims, sort the IUT p-values and apply Holm's thresholds \(\alpha/(M-i+1)\). If a first true null is rejected, its p-value is at most \(\alpha/m_0\) where \(m_0\) is the number of true claim nulls. The union bound over those true nulls gives strong FWER \(\le\alpha\) under arbitrary dependence. This is a finite-sample guarantee conditional on valid endpoint tests. Under independence with all nulls uniform, the probability of any rejection is exactly \(1-(1-\alpha/M)^M\). [Holm (1979)](https://www.jstor.org/stable/4615733)

For simple variance planning at known components, if \(z=z_{1-\alpha}+z_{1-\beta}\), power \(1-\beta\) needs
\[
B\ \ge\ \frac{\sigma_u^2+\sigma_e^2/S}{d^2/z^2-\sigma_\nu^2/S}
\]
with positive denominator. At fixed \(S\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \(\alpha=0.05/3\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.




## 4. Replayed empirical results

The verifier loaded the real blind bundles. It replayed all published G1–G4 cell entries, primary and historical Work F, Work G, and Work A Q2 with maximum absolute difference zero. It checked the NPZ temperatures against both the summary and the protocol lock. H has a published Work F entry and an older positive-world lookup anchor. Its pooled two-sign metrics below are recomputed from its NPZ arrays. Q3 tables are \((2,3,2,256,4)\) per cell; Q2 scores are \((2,3,7,512,4)\). NLL clipping never affected the scored true-label probabilities used in the decompositions.

### Q3 Bayes terms and the paired decomposition

| Cell | Blocks | Blind \(H(Y\mid S,Z)\) | Blind \(I(Y;S\mid Z)\) | Independent physical calibration MI | Direct NLL | Full-IQ NLL |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 131 | 0.925633 | 0.372815 | 0.363112 | 0.941126 | 0.929038 |
| G2 | 8 | 0.824877 | 0.292945 | 0.176866 | 1.022230 | 1.019346 |
| G3 | 8 | 0.734110 | 0.427375 | 0.344010 | 0.979568 | 0.953916 |
| G4 | 8 | 0.804198 | 0.271874 | 0.179398 | 1.015851 | 1.003738 |
| H | 16 | 0.820699 | 0.424661 | 0.350032 | 0.950133 | 0.942628 |

The independent estimate evaluates \(I(Y;S\mid Z)\) in the joint defined by \(\frac12\,\texttt{cal\_mass}(s,z)\,\texttt{cal\_ref}(y\mid s,z)\). The reference uses 65,536 independent calibration mothers per sign, quarter-turn pooling of conditional counts, and a half-count per class. Blind counts in the eight-block cells are substantially sparser. The independent estimates show available signed information at 0.80 rad as well as 1.60 rad. They quantify a property of the physical decoding problem.

The next table decomposes \(\Delta L=L_{\rm zero}-L_{\rm phase}\). The Bayes term is zero in every row. Prior is the marginal chain-rule term after fitting then projecting.

| Cell | Locked \(\Delta L\) | Temperature | Projection at \(T^*\) | Prior | Reverse conditional KL | \(\Delta L\) at \(T^*\) |
|---|---:|---:|---:|---:|---:|---:|
| G1 | +0.03565000 | +0.03654337 | +0.00012375 | +0.00000005 | −0.00101717 | −0.00089337 |
| G2 | −0.02775961 | −0.00004351 | +0.00341462 | −0.00008845 | −0.03104227 | −0.02771611 |
| G3 | +0.03708294 | +0.03751254 | −0.00001311 | −0.00000091 | −0.00041559 | −0.00042960 |
| G4 | −0.02491655 | +0.00143469 | +0.00443245 | +0.00002490 | −0.03080859 | −0.02635124 |
| H | +0.01808982 | +0.02178504 | +0.00023216 | −0.00000148 | −0.00392590 | −0.00369522 |

**G1.** For phase, the five-term path is \(0.92563349+0.05975673+0.00036415+0.00000192+0.45947227=1.44522855\). The zero path is \(0.92563349+0.09630010+0.00048790+0.00000197+0.45845510=1.48087856\). Scalar mismatch accounts for 102.51% of the locked lead along this path. Before calibration or projection, marginal KLs are \(0.00008076\) and \(0.00035846\), respectively. Raw phase and zero empirical-code NIGs are −0.12794 and −0.20533. The S17 direct reference scores 64.62% versus 23.80% and 19.41%. Phase's lower ECE is consistent with the confidence mismatch. It does not measure the share of the NLL lead. Using the argmax set, the empirical resolutions are 0.01162 and 0.02104 nats for phase and zero. Using the exact score vector, both have 0.46066 nats of in-sample resolution because both distinguish the signed codes. Those scores can encode information that the decoded predictions use poorly. With sign probabilities marginalized conditional on code, the NLL cost is 0.007704 for phase and 0.002672 for zero. The difference, 0.005032 nats, is sensitivity to that specific sign-discarding operation. It is distinct from Bayes MI and from the factorial intervention.

**G2 and G4.** Raw zero accuracy leads by 2.485 and 1.671 pp. Finite \(T^*\) preserves those rankings and a fitted loss lead of 0.02772 and 0.02635 nats remains. The reverse-conditional term gives zero advantages of 0.03104 and 0.03081 nats after projection and fitting. The global prior and scalar terms are small. Raw zero's coarsened argmax resolution also exceeds phase's by 0.01268 and 0.01877 nats. Projection actually reverses the top-1 ordering in those cells while leaving zero with lower probabilistic loss. Thus the locked reversal is explained at the score level by conditional probability allocation and raw decision ordering. The five saved conditions do not establish an architectural transition at \(\pi/2\). Check the angle effect by holding \(|\omega|\), horizons, and diffusion fixed or by crossing them prospectively with common noise and more independent seeds.

### Q3 scalar recalibration

Each cell gives locked NLL / ECE and retrospective seedwise \(T^*\) NLL / ECE. Temperature triples follow seeds 202611, 202622, 202633.

| Cell | Arm | Locked NLL / ECE | Fitted NLL / ECE | \(T^*\) |
|---|---|---|---|---|
| G1 | phase | 1.445229 / .118604 | 1.385472 / .025608 | ∞, 10.144, 5.720 |
| G1 | zero | 1.480879 / .182737 | 1.384578 / .051229 | ∞, 6.453, 6.087 |
| G2 | phase | 1.335826 / .022970 | 1.334544 / .011354 | .548, 1.008, .769 |
| G2 | zero | 1.308067 / .023020 | 1.306828 / .017347 | 1.220, .916, .711 |
| G3 | phase | 1.441047 / .121695 | 1.385166 / .033546 | 9.890, 8.240, 5.042 |
| G3 | zero | 1.478130 / .179196 | 1.384737 / .052938 | ∞, 6.842, 5.996 |
| G4 | phase | 1.334667 / .027630 | 1.334066 / .018397 | .547, .945, .743 |
| G4 | zero | 1.309751 / .032534 | 1.307715 / .021834 | 1.535, .831, .762 |
| H | phase | 1.416987 / .096744 | 1.380927 / .018175 | 2.405, 3.271, 2.455 |
| H | zero | 1.435077 / .142909 | 1.377232 / .049863 | 7.968, 2.836, 2.339 |

### Q3 pooling on the saved test design

Each entry is accuracy (%) / NLL. Both pools reduce NLL in all ten rows. The two 1.60-rad cells lose accuracy with either pool, while the two 0.80-rad cells gain accuracy with either. H follows the accuracy-decline pattern for both arms.

| Cell | Arm | Raw | Arithmetic | Geometric |
|---|---|---|---|---|
| G1 | phase | 23.80 / 1.445229 | 17.05 / 1.411312 | 16.36 / 1.411586 |
| G1 | zero | 19.41 / 1.480879 | 16.43 / 1.439122 | 16.60 / 1.445854 |
| G2 | phase | 33.88 / 1.335826 | 40.95 / 1.306349 | 40.43 / 1.299371 |
| G2 | zero | 36.36 / 1.308067 | 38.31 / 1.276082 | 38.51 / 1.273466 |
| G3 | phase | 22.68 / 1.441047 | 18.39 / 1.410222 | 18.68 / 1.410409 |
| G3 | zero | 19.74 / 1.478130 | 18.20 / 1.441579 | 17.64 / 1.447584 |
| G4 | phase | 33.29 / 1.334667 | 40.79 / 1.311247 | 40.53 / 1.304942 |
| G4 | zero | 34.96 / 1.309751 | 37.76 / 1.278018 | 37.40 / 1.274705 |
| H | phase | 25.76 / 1.416987 | 22.02 / 1.384622 | 22.74 / 1.382330 |
| H | zero | 23.17 / 1.435077 | 21.63 / 1.395866 | 21.63 / 1.399059 |

On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.

### Work F mechanism and averaging convention

\(\omega\) is formatted with explicit sign in the public prompt. The same condition is used to build the prompt for each connector. Only the six slot entries become zero in zero. At a fixed code, the zero-arm signed text remains the varying input to its deterministic table. The physical world used for evaluation also changes which codes and labels are sampled. Work F quantifies these by comparing the signed tables on the same codes within each world.

G1 seed-mean orientation gains are +0.00981980 and +0.00504776. Their linear dependence on the probabilities makes them equal to the corresponding gains on the seed-average table up to rounding. NLL is nonlinear. Seed-mean swapped penalties are +0.01836983 and +0.00737184. Evaluating the arithmetic seed-ensemble table yields +0.01615141 and +0.00795537. The physical reference has orientation gain 1 by definition. Phase per-seed gains are −0.00147, +0.02104, +0.00989. The zero-arm gains are +0.00184, +0.01295, +0.00035. An orientation gain is a weighted projection coefficient onto the reference odd direction. It does not measure a fraction of the Bayes information or the Stage 3D arm contrast. The direct readout is a separate physical predictor with access to the known condition.

On all the saved G1 rotations of the phase checkpoint, a text-only flip raises NLL by +0.00323146 and a slot-only flip by +0.01723982, holding the other channel matched. A joint flip raises it by +0.01857416. The difference in differences is −0.00189712. On rotation zero these are +0.00302035, +0.01679462, +0.01836983, and −0.00144514. The interventions show sensitivity to physically aligned slots in the phase checkpoint, alongside a text response, in the tested models. They compare flipped slots to matched slots within a checkpoint. Training with the phase arm and training with the zero arm is a distinct intervention. A shared-development retraining comparison with a locked recalibration and independent seeds is needed to attribute persistent NLL gains to richer slot use.

### Q2 controls

\(H(Y)=\log4\), while the base-code in-sample \(H(Y\mid Z)=0.809758\) with 183 occupied codes. The Work A no-reference reader yields uniform on these rows. The gap illustrates the distinction between sparsely conditioned, label-reusing estimates and physical predictive references.

| Control | Work A NLL | Locked phase / zero NLL | Fitted phase / zero NLL | \(\Delta\) temperature | Locked \(\Delta\) |
|---|---:|---|---|---:|---:|
| none | 1.386294 | 1.392846 / 1.393822 | 1.386272 / 1.386294 | +.000953 | +.000976 |
| valid | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
| duplicate dedup | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
| duplicate raw | .819207 | 1.394420 / 1.413341 | 1.384193 / 1.385808 | +.017307 | +.018922 |
| independent only | 1.386294 | 1.407714 / 1.420522 | 1.386294 / 1.386071 | +.013031 | +.012807 |
| valid plus independent | .819207 | 1.408232 / 1.418281 | 1.386244 / 1.385545 | +.010748 | +.010049 |
| fresh resample | .663189 | 1.405640 / 1.406397 | 1.385318 / 1.382645 | +.003429 | +.000757 |

For valid, the fitted residual phase lead \(+0.00159367\) is \(+0.00028827\) reference-prior KL, \(+0.00158510\) reverse KL, and \(−0.00027971\) realized-reference label residual. For fresh resample, the remaining lead reverses to \(−0.00267249\). The Work A reference gains 0.156018 from fresh vs valid, while the phase connector is 0.016757 worse at locked \(T\). Q2 realized NIG is undefined for the uniform Work A controls because the denominator is zero. All Q2 controls and arms have complete numerical paths, ECE, Brier, scalar temperatures, and soft reference diagnostics in the JSON.




## 5. Power replay and Round 2 lock proposal

The legacy pilot replay gives endpoint minima 131 and 54 blocks for accuracy and NLL, and powers 0.950512 and 0.999906 at 131. Its scenario SDs are 0.094414 and 0.079795. The crossed G1 means are 0.043926 accuracy and 0.035650 NLL. The fitted variance components on the real block-by-seed matrix are as follows.

| Contrast | Block variance | Seed variance | Interaction variance | Seed SD estimate | 95% upper bound on seed SD |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.0000758565 | 0.0003711426 | 0.0005829523 | 0.01927 | 0.08557 |
| NLL | 0.0001213570 | 0.0004670843 | 0.0003618383 | 0.02161 | 0.09571 |

At the two planning excesses \(d=(0.03,0.04)\), one-sided \(\alpha=0.05/3\) and marginal power 0.95, the synthetic fixed-seed block minima for accuracy/NLL are G1 7/5, G2 11/7, G3 8/6, G4 11/4 and H 9/6. These are conditional t minima for hypothetical positive effects. Observed G2 and G4 effects favor zero and fail the positive comparative target. G3's accuracy-only random-seed minimum is 42 under its zero-truncated seed variance estimate from eight blocks. Three seeds do not give a viable 95% crossed NLL power design in any cell under the plug-in scenarios. The small conditional minima do not override the frozen synthetic budget or planning from independent upper variance bounds.

G1-proxy sample sizes and powers follow. CTTC uses session means across eight clusters per session. Its ICC parameter is the assumed fraction of the proxy block variance shared at session level, with no additional session-by-seed effect in this sensitivity run. “Minimum” means the first integer yielding at least 95% marginal power under the specified test and effects.

| Setting | Available independent units | Three fixed seeds, minimum units accuracy/NLL | Three random seeds, power accuracy/NLL | Ten random seeds, minimum units accuracy/NLL | Ten random seeds, power accuracy/NLL |
|---|---:|---|---|---|---|
| Synthetic G1 proxy | 131 blocks | 7 / 5 | .224 / .300 | 15 / 7 | .981 / .998 |
| Pluto proxy | 16 captures | 7 / 5 | .136 / .234 | 15 / 7 | .953 / .994 |
| 72 independent clusters sensitivity | 72 clusters | 7 / 5 | .212 / .293 | 15 / 7 | .978 / .998 |
| CTTC proxy, ICC .10 | 9 sessions | 4 / 3 | .208 / .287 | 7 / 5 | .973 / .997 |
| CTTC proxy, ICC .25 | 9 sessions | 4 / 4 | .202 / .280 | 7 / 5 | .970 / .996 |

At the actual G1 excesses over \(0.02\), the three-fixed-seed G1 or Pluto-proxy minima are 10 and 17 blocks/captures, and the CTTC proxy at ICC .10 gives 4 and 6 sessions. For three random seeds at the planning excesses, the ceiling even with known components and infinitely many blocks is 0.71537 and 0.85941. The conservative finite-level test tends to 0.23839 and 0.30978. These ceilings are consequences of the fitted seed variance and fixed effect-size assumptions. The high potential sampling uncertainty in that variance makes the ten-seed numbers planning proxies. Even with ten seeds, the modeled power statements apply only to the two relative endpoints.

The verifier drew 200,000 Monte Carlo replicates per scenario from the independent normal and chi-square sufficient statistics. G1 at 131 blocks gives the following analytic/Monte Carlo boundary rejection rates at nominal \(0.01667\).

| Endpoint | Fixed-seed t | Crossed test | Block-only t with random seeds |
|---|---|---|---|
| Accuracy | .01667 / .01654 | .01161 / .01168 | .39171 / .39145 |
| NLL | .01667 / .01716 | .01280 / .01277 | .40810 / .40951 |

For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.

G1's same-mother Work F fixed-seed 95% block intervals for the all-rotation interventions are \([0.002804,0.003659]\) for text-only flip and \([0.016552,0.017927]\) for slot-only flip. The seed-wise slot effects are \(-0.000641,+0.035214,+0.017147\). A pooled interval over mothers cannot establish a stable retraining effect.

### Round 2

- **M1** should lock matched-text phase, text-only, joint-blind and independently permuted phase-slot controls, with a signed-slot intervention at fixed text and matched waveform. Cross \(|\omega|\) and accumulated angle independently where the simulator allows it, preserve the analytic S17 and full-IQ anchors, and keep sign/rotation/mother identifiers together for inference.
- **M3** should score the native equivariant forecast against the raw, arithmetic, geometric and canonical-representative paths on a common development split. Calibrate on development data that span the prospective regimes and freeze the choice before blind scoring. Lock all absolute endpoints before comparing phase and zero. The G1 raw and pooled paths fail absolute accuracy, and the blind-fitted paths have below-chance accuracy and negligible NIG.
- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.

### Reproduction

Run from the repository root with the pinned environment.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py
~~~

The run used Python 3.12.7, NumPy 2.3.5 and SciPy 1.17.0. The default uv cache was read-only under the workspace sandbox. The run used an offline cache under M2 populated from the already available pinned wheels. The verifier reads the Stage 3D artifacts and writes its result adjacent to the script.



## 6. Round 2 claim scope and budget lock

This section sets the Round 2 promotion rules. Sections 1–5 retain the retrospective Stage 3D replay and its original power assumptions. Freeze the applicable branch, training mixture, checkpoints, comparators, row manifest, scoring and weights before Stage 4 blind scoring. Use fresh synthetic generator namespaces. Stage 3D serves as development evidence for this protocol.

| Tier | Admissible conclusion | Eligibility |
|---|---|---|
| 1. Single frozen connector per trained arm | At seed 202611, these checkpoint/readout pairs satisfy the gate on the registered target population, conditional on training and dev selection. | M4's four matched 384-step fits and one prespecified blind evaluation after dev eligibility. An incomplete M6 pilot gives engineering evidence. A one-seed confirmatory fallback requires its complete arm roster, trained checkpoints and plan to be selected on dev before blind access. |
| 2. Frozen connector panel | The paired mean loss over the selected \(S=2\) or \(3\) frozen seed pairs satisfies the gate, conditional on the selected checkpoints. | M6 continuation targets \(S=3\). Freeze the panel and average losses or hits within block. Report each seed. A deployed probability ensemble has its own estimand and must be frozen before scoring. Extra survivor-only seeds may be evaluated against deterministic controls. Restrict matched trained-arm contrasts to their common preselected paired seeds. |
| 3. Population of retrains | The locked training recipe has expected benefit over fresh training runs and the registered blind sampling process. | Require at least ten independently paired retrains and adequate prospective all-endpoint power under the crossed model or a validated alternative. Ten is the present protocol floor based on G1, with no universal power guarantee. Fewer seeds are ineligible for Tier 3 under this protocol. |

M4 estimates \(\$8.44073416\), including four single-seed connector fits at \(\$2.86786560\). M6's alternative envelope is \(\$9.50\) against the historical \(\$12.88\) balance, including \(\$0.491218\) for the \(\le173/960\)-step pilot and \(\$5.101110\) for continuation to three seeds. These are alternative envelopes. Reprice any combined plan against the single \(\$9.50\) ceiling before release. Mixed synthetic/real fitting and synthetic-only fitting with real transfer are distinct frozen recipes. Both budget routes target zero net spend under the account guard. This verification used local CPU. [M4 compute envelope](../M4/THEOREM_M4.md#3-frozen-compute-envelope-and-dispatch) · [M6 pilot and budget](../M6/THEOREM_M6.md)

### Conditional validity after the 18% dev screen

Let \(\mathcal F\) contain training and dev data and randomness, pilot decision, selected checkpoints, seed panel, calibration, comparator selection, margins, sampling and scoring plans. Set \(A=\{\text{continue and eligible}\}\in\mathcal F\). The selected-path target is the conditional expected blind gain \(\mu_{\mathcal F}\). Under the balanced additive model from §3,
\[
 \mu_{\mathcal F}=\mu+\overline{\nu}_{\mathcal F},\qquad
 \operatorname{Var}(\bar D\mid\mathcal F)=(\sigma_u^2+\sigma_e^2/S)/B.
\]
This assumes conditional independence of blind block innovations from the development selection. The frozen seed effects remain in the conditional mean. Crossed inference includes \(\sigma_\nu^2/S\) in the sampling variance. Dev selection can also change the sampled seed population, so a Tier 3 design must lock that population and account for specified training failures.

Require \(P(p_{\rm blind}\le a\mid\mathcal F=f)\le a\) at every eligible \(f\) with \(\mu_f\le\delta\). The one-sided Student test attains equality at the boundary under its Gaussian and independence conditions. A sharp-null randomization test uses its justified conditional exchangeability orbit. Give a stopped claim \(p=1\). For any \(a<1\), iterated expectation yields
\[
 P(A,\ \mu_{\mathcal F}\le\delta,\ p_{\rm blind}\le a)
 \le a P(A,\ \mu_{\mathcal F}\le\delta).
\]
With exact continuous boundary p-values this gives conditional size \(a\) among boundary-null survivors and joint false-promotion probability \(aP(A)\). Conditional Holm over the true selected-path claims then gives unconditional FWER at most 0.05. Dev choices may govern futility and select a fully specified plan. The confirmatory rejection is made once on the blind data. All three claim slots remain in the family.

M6's \(t=0.18\) is a planning input. The training fraction \(173/960\) does not determine information time. The \(-0.270438\) cutoff came from the correlated two-look model and may be frozen as a dev-only heuristic, with a single confirmatory blind look. For an independent alternative planning model, take \(Z_d\mid\mu\sim N(\sqrt t\mu,1)\), \(Z_b\mid\mu\sim N(\mu,1)\) with independent innovations and prior \(\mu\sim N(0,4)\). The conditional mean and variance of \(\mu\) are \(\sqrt t\,z/(t+1/4)\) and \(1/(t+1/4)\). The cutoff for predicted \(P(Z_b>z_{.95}\mid Z_d)\ge0.10\) is
\[
 \big(z_{.95}-z_{.90}\sqrt{1+1/(t+1/4)}\big)(t+1/4)/\sqrt t.
\]
Either cutoff obeys the conditional proof if the blind test is valid. The threshold controls resource allocation. Its usefulness and projected continuation power require dev calibration.

Disjoint row IDs are one part of the split. The proof also needs conditional test validity when dev and blind share a campaign, hardware state or deterministic TX source. Hold out the entire designated recording where required and justify independence given genuinely fixed conditioning variables. Outcome-linked score reports, variance, pathology and troubleshooting on blind rows can compromise selection. Keep timing and parity smoke tests on dev. Blind scorer failures follow the frozen fallback or stop rule. In the counterexample that uses the same standardized test score to screen and test, conditional error among survivors is \(\min(a,P(A))/P(A)\).

## 7. Round 2 formal gate table

Let \(q_X\) be a frozen dev-calibrated forecast. Orient paired differences as \(D_L(X,Y)=L(q_Y)-L(q_X)\) and \(D_A(X,Y)=A(q_X)-A(q_Y)\) on identical cases. \({\rm LB}(D)>d\) means a one-sided lower confidence bound above the margin at the claim's tested Holm level. Use equivalent component p-values in the testing tree. Every applicable row is mandatory. Exact controls and fixed-cohort guards get p-value 0 on pass and 1 on failure for IUT bookkeeping, with no population inference attached to the guard.

### Arm roles and causal contrasts

The four priced arms are **matched Zero**, **matched Naive-Additive**, **Arm 1 Host-Pure Equivariant**, and **Arm 2 Physical+HostResidual**. M1's **Arm 0** names the legacy separate-token Stage 3D-style reference. Identify it with the matched Zero comparator only if it satisfies the matching contract. Otherwise retain legacy Arm 0 as contextual replay and use a separate matched Zero implementation. Direct S17/Direct Physical and known-parameter Full IQ are analytic references. Full IQ is a finer-information synthetic diagnostic. Real raw-I/Q references obey the causal filtration and may be approximate.

| Gate row | Exact contrast or eligibility control |
|---|---|
| Arm matching | Match allowed observations and deadlines, frozen host, signed text, candidate strings, quotient/lift when isolating fusion, calibration opportunity, score floor, training manifest, optimizer and step budget. Zero uses the prescribed coupled-phase input ablation. Naive-Additive shares the inputs with a centered additive code/phase score in canonical coordinates. Verify its mixed code-by-phase term is zero. Changes to the quotient, head or training recipe create a package comparison. A fusion attribution requires the matched ablation. |
| Common absolute gate | Require NLL improvement \(>0\) over uniform and the dev prior. Require accuracy \(>1/K+0.02\) and improvement \(>0\) over the frozen prior classifier. Test both skill statistics against a justified locked random-label null. Require paired Brier improvement \(>0.005\) over the dev-selected lowest-Brier baseline from uniform, dev prior and matched Zero, and \(>0\) over every required baseline in that set apart from the candidate. Require a valid blind ECE upper bound \(<0.05\) on ten frozen top-label bins. Require native equivariance and projection idempotence max defect \(\le10^{-5}\) on the declared group support. |
| NIG | Lock a same-information oracle or independently fitted physical reference. Establish a positive denominator analytically or with a valid paired lower bound as an additional IUT endpoint, and require \({\rm LB}(0.90H(Y)-L(q_X)+0.10L(q_{\rm ref}))>0\). A balanced synthetic target law has \(H(Y)=\log K\). For real data predeclare the target class law or carry entropy uncertainty jointly. A causal approximate reader yields reference-normalized wording. An indefensible denominator makes the gate ineligible. |
| Arm 1 host-pure extrapolation | Against each of Zero and Naive-Additive require \({\rm LB}(D_L)>0.02\) nats and \({\rm LB}(D_A)>0.02\) on the locked extrapolation mixture, plus common absolute/NIG gates. Score the frozen host's candidate outputs with quotient alignment and scalar calibration. Coupled physical features can enter the soft tokens. Verify the output graph and absence of a direct physical logit path. |
| Arm 1 fixed-text intervention | At the same checkpoint, waveform/cache, signed text, scorer and scalar, require \({\rm LB}(D_L(\mathrm{matched},\mathrm{slot\ swap}))>0\) and \({\rm LB}(D_L(\mathrm{matched},\mathrm{slot\ shuffle}))>0\). The signed swap flips the slots. The shuffle is a separately seeded, frozen mapping of slot values across signs within prespecified absolute-CFO, lag, SNR and availability strata, without labels. Record its effective mismatch fraction and the locked treatment of strata that cannot be shuffled. Keep signs/rotations with their mothers. Audit canonical centered code-by-phase mixed contrasts with the additive arm as the zero-interaction control. This establishes a response to the supplied slot. Waveform-based CFO estimation is a different hypothesis. |
| Arm 2 host residual value | With \(q_2=\operatorname{softmax}((\ell_{\rm phys}+\gamma\tanh\ell_{\rm host})/T_2)\), require \({\rm LB}(D_L)>0.02\) nats and \({\rm LB}(D_A)>0\) against independently dev-calibrated Direct Physical with identical evidence and representation. Also require \({\rm LB}(D_L)>0\) versus the same-stack residual-off forecast \(\operatorname{softmax}(\ell_{\rm phys}/T_2)\). Apply both slot interventions only in the host residual branch. Require positive NLL lower bounds for both while \(\ell_{\rm phys}\) is bit-identical and \(\gamma,T_2\) remain fixed. Apply common absolute/NIG gates to confirmatory Arm 2 release. This tests incremental host value over the specified physical reader. Arm 1 evidence must come from Arm 1. |
| Calibration | Freeze one dev-fitted scalar per checkpoint/arm and task or predeclared stratum, using the same predeclared search range and objective. Hash the calibrator, priors and adapter with the scores. Reuse the treatment's scalar for its ablations. A synthetic-to-real zero-shot route transfers the synthetic-dev fit and permits only registered causal sample state. A real-dev fit is a separate real-calibrated route and excludes the held-out capture. Choose the confirmatory route on dev. Oracle blind temperature and calibration sensitivity are retrospective diagnostics. |
| Q2 gain | Under the same frozen RefSufficientPool scorer require \({\rm LB}(L_{\rm one}-L_{\rm two})>0.02\) nats and \({\rm LB}(L_{\rm none}-L_{\rm one})>0\). One uses the registered first past release. Two includes the distinct second valid release. Q2 two-valid clears its absolute and reference-normalized gates. Report Direct Q2, accuracy/SER, bit NLL and EVM as secondary comparisons. |
| Q2 exact controls | Valid must give byte-identical preprocessing, fixed slots, masks, positions, candidate context and deterministic scorer outputs after appending a raw duplicate ID, independent-channel record or empty padding, or permuting the reference list. Require 0.0 max probability change, TV and KL for each pair. Verify canonical reuse at the real API boundary where used. Validate ID and samples before deduplication and fail conflicting IDs. A genuinely fresh acquisition with a new ID enters the two-valid efficacy test; superiority is expected across cases. |

With metadata \(M\) and authorized references \(R\) held fixed, a deterministic waveform cache \(Z\) and host forecast \(Q\) satisfy \(I(Y;Q\mid R,M)\le I(Y;Z\mid R,M)\le I(Y;W\mid R,M)\). The Stage 3D loss ladder compares finite readouts and includes their regret. Conditional entropy gaps require Bayes readouts at each named information set. The residual-off and fixed-physical interventions compare architectures at the same information set. [M1 arm design](../discussions/M1/prompt_r2.md) · [M3 reference contract](../discussions/M3/prompt_r2.md) · [M6 DPI](../M6/THEOREM_M6.md)



### Three-claim testing tree

Freeze one primary route and checkpoint panel per claim before blind access. Claim 1 defaults to Arm 1. A preregistered Arm 2 substitution requires the complete residual-value, absolute, Q2 and structural endpoints and yields an Arm 2 claim. Both arms can instead be required jointly by taking the maximum over both full endpoint lists. Independent confirmatory releases for either arm require extra hypothesis slots and revised multiplicity before acquisition.

| Claim | Mandatory endpoints and scope | Unit |
|---|---|---|
| 1. Synthetic extrapolation + Q2 | Require the locked extrapolation route's arm-specific, absolute, NIG, calibration and structural endpoints. Also require the Q2 ordered gain, absolute and exact controls. Use the M4 32-block equal-weight mixture of eight signed large-angle settings with four blocks per setting, or preregister a 131-block design with the same target mixture weights and the exact per-stratum weights/counts. Region-wide efficacy means the prespecified average. Per-setting claims are descriptive unless included in the IUT. The 131-block homogeneous G1 curve below is a power proxy, rather than a replay of that stratified design. | Independent generator blocks. Keep a mother's repeats and controls paired. Use the fixed-stratum weighted contrast where applicable. |
| 2. Pluto | Require the common absolute/calibration/structure and ordered Q2 gate on the natural rows. Release pilots 15 and 18. Include every registered lock failure with the common uniform fallback. Report lock coverage, stress results on 140 base frames, and Q1/Q3 diagnostics. | 2,240 frames in 16 chronological blocks from one campaign, with copies nested. Condition on campaign. Exact inference needs a valid conditional block model or a valid design randomization. If shared temporal dependence invalidates the model, report the campaign descriptively. |
| 3. CTTC + f708_snr10 | Require the same paired natural-data Q2, absolute and exact controls, pilots 126 and 127, and the single-capture transfer guard on f708_snr10. Require \(L_{\rm two}<L_{\rm one}<L_{\rm none}\) and \(L_{\rm two}<L_{\rm prior}\) as fixed-cohort point inequalities there, complete reporting and zero integrity failures. Exclude this capture from checkpoint fitting, calibration and nuisance/reader/prior/hyperparameter tuning. Its causal sync and authorized pilots remain permitted. | 1,440 packets in 72 loops nested in nine captures, and 720 Q3 pairs in 36 disjoint loop-pair groups nested in the same captures. Average paired effects within capture, then weight captures equally. Report the held-out capture separately, plus leave-out results for the other eight. The holdout's p=0/1 bookkeeping is a cohort guard, with no standalone population test. |

Real primary Q2 uses paired symbol NLL for its ordered-evidence contrasts. Q1 relative-turn, real Q3 retransmission and stress metrics are secondary unless a full confirmatory endpoint set is appended to the relevant slot before scoring. Preserve every failed-lock row and apply the same fallback rules across arms. Validate identities across the regular and fallback paths. At each release, audit raw support and timestamp using the locked causal alignment and without consulting target labels or unreleased references.

CTTC replays one public cyclic TX sequence. Disjoint roles/splits prevent direct observation reuse. The common fixed test range \([334096,455000)\) can still induce source-ordinal dependence across captures. Scope the primary cohort estimand to the registered captures and fixed source. Justify independent capture innovations given the locked conditioning before nine-unit inference. If common residual randomness remains across captures and there is no valid design-based alternative, report descriptive results. Show source-ordinal aggregation, capture leave-one-out and capture-by-source sensitivity. The 36/72 nested groups contribute to capture means. They do not supply independent capture or new-source draws. [M4 benchmark and split](../M4/THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)

For independent equal-weight Gaussian units with common variance the homogeneous t has \(B-1\) df. In the balanced 32-block synthetic model, fitting eight fixed cell means with common residual variance gives \(32-8=24\) df and the same \(\sigma/\sqrt{32}\) standard error for their average. For unequal allocational counts \(n_h\) with fixed weights \(w_h\), the common-variance standard error is \(\sigma\sqrt{\sum_h w_h^2/n_h}\), with \(B-H\) residual df. For independent Gaussian capture means with fixed heterogeneous means and common variance, the sample variance denominator is noncentral chi-square and independent of the grand mean. It is stochastically larger than the central denominator, giving a conservative upper-tail test at the null boundary. Independent bounded unit means allow the Hoeffding test in §3 for mean margins if Gaussian assumptions are unsupported. Valid random-label p-values require sanctioned assignments under the relevant sharp null. Preserve rotations, repeats, captures and source restrictions in the exchangeability orbit. Use a registered randomized negative control when the existing label design lacks one. A paired sign-flip test needs actual randomization or a justified joint sign-symmetry null. A weak mean inequality alone supplies no such exchangeability.

For each claim \(j\), put \(p_j=\max_{k\in K_j}p_{jk}\) over mandatory tests and guards. The claim null is their union, so \(p_j\) is super-uniform whenever any endpoint null is true, conditional on the dev lock. Apply Holm to \((p_1,p_2,p_3)\) at \(0.05/3,\ 0.05/2,\ 0.05\), stopping at the first failure. Conditional on \(\mathcal F\), the first rejected true claim, if any, meets a threshold at most \(0.05/m_0\), where \(m_0\) counts true claims. The union bound gives conditional FWER at most 0.05 under arbitrary claim dependence. Averaging over \(\mathcal F\) retains the bound. Keep discontinued, ineligible and missing claims as p=1 in their original slots. An IUT divides no alpha inside a claim. Reporting a set of intervals with simultaneous coverage is a separate requirement.

## 8. Stage 6 compatibility

Freeze one task, one pre-question representation, one admissible action set per recording, and exactly one sealed read of equal cost. Give max EIG, uniform random, nearest and fixed best on dev the same available candidates, reader family, calibration opportunity and cutoff. Each policy commits its action and pre-reveal forecast before receiving the read. Seal nonchosen evidence from the policy. The blinded evaluator pairs policies by recording with recorded counterfactual acquisitions when valid. Otherwise preassign an action design with sufficient support and use its registered design-based estimator. Freeze tie breaking. Nearest uses the last authorized anchor with a causal distance rule. Fixed best on dev selects a single constant action (or a prespecified mask-availability fallback rule) by dev-only realized posterior NLL and freezes it. It cannot select based on blind performance or hidden ambiguity.

For the adaptive acquisition claim, use \(D_L=L_{\rm baseline}-L_{\rm max\,EIG}\) and \(D_A=A_{\rm max\,EIG}-A_{\rm baseline}\) on blinded post-reveal predictions, with both one-sided lower bounds strictly positive for **each** of random, nearest and fixed best on dev. Use one IUT on those six endpoints at the claim's registered level. Stratify or nest trial units at the recording level. A pre-reveal proper-score and seal audit tests commitment and integrity separately. For true post-reveal Bayes NLL with equal read costs, \(V(z,a)=I(Y;R_a\mid Z=z)\). The adaptive-versus-best-static value is
\[
 \mathbb E_Z\max_a V(Z,a)-\max_a\mathbb E_Z V(Z,a).
\]
With a finite common action set this is strict iff every fixed action is strictly suboptimal on a set of positive probability. For varying masks, replace each constant by an eligible frozen availability-only policy and apply the analogous criterion to that policy class. Merely beating random and nearest does not exclude a static optimum. Accuracy has its own Bayes decision value and its own empirical contrast.

Treat 16-bit canonical versus 4-bit S17 as a distinct representation contrast paired on identical source records, existing evidence, target and prescribed reveal action. Use the same calibrated reader family and development opportunity and count the actual storage bits. Require positive NLL gain and positive accuracy gain if making a joint superiority claim. Require exact S17 prefix reconstruction and all registered symmetry checks. Repeat the full policy comparison at fixed 16-bit budget for an adaptive claim, so representation and action choice can be attributed separately. Full IQ is the diagnostic ceiling after matching allowed causal inputs. Predeclare both Stage 6 claim slots and use Holm within that family if claiming them separately. An optional portfolio-wide 0.05 rule opens the independent Stage 6 confirmatory family only after all three Stage 4 claims pass. If Stage 4 contains a true claim null, reaching Stage 6 already entails a Stage 4 error. If all Stage 4 nulls are false, conditional Stage 6 family control applies after selection. Without this sequential rule, report the families at their separate levels and refrain from a joint 0.05 claim. [M5 action and cache design](../discussions/M5/prompt_r2.md)



## 9. Round 2 verification and conditional power

The replay exercises two independently generated dev/blind statistics, with a dev-only illustrative switch between 24-df single-seed and 130-df three-seed blind pivots. The switch is a test fixture for selection validity. The actual cost route must be decided and frozen by the pilot rule. At the M6 legacy cutoff, the analytic boundary continuation probability is 0.606588. The recalculated independent predictive example uses dev cutoff −0.701565 and continues with probability 0.758525 at its dev boundary. Both admit a final exact-size blind test.

| Dev-only screen | Blind level | Conditional size exact / MC | Joint false promotion exact / MC |
|---|---:|---:|---:|
| M6 cutoff −0.270438 | .016667 | .016667 / .016290 | .010110 / .009880 |
| M6 cutoff −0.270438 | .05 | .050000 / .050256 | .030329 / .030480 |
| Independent predictive example | .016667 | .016667 / .016601 | .012642 / .012598 |
| Independent predictive example | .05 | .050000 / .050431 | .037926 / .038270 |

The run uses 400,000 deterministic seeded draws, checks both selected branches against their exact t boundary size and repeats all Round 1 replay assertions. The numbers illustrate the conditional theorem. They are not estimates of practical continuation on unseen Stage 4 pilots.

For power, let \(d=\mu_{\mathcal F}-\delta\) and \(s_F\) be the conditional per-unit SD. Use
\[
 P_{\rm pass}(d)=\Pr\!\left[t_{\mathrm{df}}\big(d\sqrt B/s_F\big)>t_{\mathrm{df},1-a}\right].
\]
The receipt gives the full excess grid, an inversion at 95% marginal power, and the power at the historical G1 effects. It uses \(a=0.05/3\), the conservative first Holm level. It reports conditional power among eligible runs. Joint promotion power also depends on continuation and the full endpoint law. The M4 32-block row uses 24 residual df. The 131-block row is the homogeneous G1 proxy with 130 df. Compute a new variance and df from the exact stratified 131-block allocation if that option is locked. [SciPy noncentral t definition](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.nct.html)

For CTTC with \(m\) loops or pairs per capture, use the sensitivity proxy \(s_F^2=\sigma_u^2[\rho+(1-\rho)/m]+\sigma_e^2/(mS)\), after conditioning on the frozen seed panel. The model uses \(m=8\) for Q2 and \(m=4\) for Q3 and retains eight df. An extra session-by-seed term would add a component to \(s_F^2\). Use actual real dev covariance for deployment planning. The main table assumes 0.10 for the fraction of common group variance shared within captures. The JSON covers 0, 0.10 and 0.25, a direct historical-panel SD sensitivity, and the legacy Work E high-variance sensitivity. The latter uses a single aggregate variance envelope for both seed counts because Work E never decomposed it.

| Design | Blind df | G1 proxy accuracy power at +1 pp excess, S=1 / S=3 | G1 proxy NLL power at +0.01 nat excess, S=1 / S=3 | Excess needed for 95% power, acc pp S=1 / S=3 | Excess needed for 95% power, NLL nats S=1 / S=3 |
|---|---:|---:|---:|---:|---:|
| Synthetic 32, eight fixed strata | 24 | 48.87% / 87.45% | 62.67% / 90.88% | 1.799 / 1.152 | .01541 / .01090 |
| Synthetic 131, G1 homogeneous proxy | 130 | 98.91% / ≈100% | 99.88% / ≈100% | .854 / .547 | .00731 / .00517 |
| Pluto 16 chronological blocks | 15 | 24.65% / 54.81% | 32.83% / 59.79% | 2.627 / 1.682 | .02250 / .01592 |
| CTTC nine captures, eight Q2 loops, \(\rho=.10\) | 8 | 72.26% / 96.93% | 81.81% / 96.78% | 1.398 / .942 | .01249 / .00947 |
| CTTC nine captures, four Q3 loop-pair groups, \(\rho=.10\) | 8 | 43.91% / 80.59% | 55.14% / 82.60% | 1.934 / 1.268 | .01689 / .01236 |

The old Work E variance envelope gives only 33.91%/71.65% accuracy/NLL power on 32 blocks and 93.01%/99.98% on 131 at the old planned excesses of +3 pp and +0.04 nat, using the new first Holm level. Pluto under that transferred envelope gives 17.08%/39.22%. These comparisons explain why the 32-block plan can produce a scoped confirmatory result if it passes yet deserves a wider uncertainty forecast. The G1 component proxy and the Work E envelope both originate in synthetic Phase-vs-Zero. Neither measures Stage 4's candidate-vs-additive, Arm 2, Q2 real, ECE, or random-label test power. Prospectively acquire development variance and meaningful effect targets for all required endpoints. If a typed gate remains infeasible under its budget, report its outcome as inconclusive or descriptive and retain the Holm slot.

For the random-label checks, archive the assignment law and compute the upper-tail randomization p-value by the exact legal assignment orbit or the plus-one Monte Carlo rule in §3. A diagnostic shuffle of deterministic real labels is insufficient without a valid exchangeability argument. The existing split proof and CPU replay do not establish real-capture exchangeability or Stage 4 gate completion.

