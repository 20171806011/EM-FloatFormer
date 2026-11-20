**I recommend keeping the reset estimator’s joint GL3 posterior alive through every wave.** The `[3,128]` tracker is a useful compatibility output, but projecting into it discards precisely the cross-ray and common-Doppler dependence that Wave 1 needs. For the inherited and shifted branches, the existing tracker can be retained with a small change to its design matrix. All three branches need an immutable starting state, an explicit record of the observations it already contains, and a branch-specific refit operation.

Three source findings affect the proposed performance claims.

- The standalone payload rebuild achieved **29.00%** pooled `full_reset` BLER. **28.86%** belongs to the receiver that switched using the two validation codewords. The per-row common-phase oracle gave 40/512 and 26/512 errors on two prespecified subsets. Those results establish the value of phase information, but do not establish that two decoded neighbors can supply all of it. [Reconstruction results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/README.md:58>)
- CLI-2 uses its original branch states even **during** Wave 1 for sign splitting and branch-specific residuals. After a successful wave it refits those original priors. Replacing only `initial_gate_out` therefore fails at both stages. [Wave-1 branch selection](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:1625>) · [Refit and final readout](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:1874>)
- Under the frozen folds, **row 11 is a Fold 0 target, row 9 is a validation payload, and neither is an auxiliary row**. Under strict whole-fold isolation, Fold 0 can decode row 11 for its score but cannot feed that word back into the state used for its other targets. Fold 1 can use row 11 as a virtual pilot. This changes how much of the historical wavefront gain can be carried into P11. [Frozen fold definitions](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/prior/strong16.py:21>)

## (a) A scale-equivariant, B-only reset posterior

### The likelihood and the scale mismatch

Write \(n=n_0>0\). For a fixed joint trajectory \(j\), let \(d_0\in\mathbb C^{32}\) and \(D_w=[d_1,d_2]\in\mathbb C^{32\times2}\) be its anchor pilot columns, including delay and rotation about `ORIGIN`. Introduce a physical power scale \(s>0\) and set
\[
a=\sqrt{sP_0},\qquad g_0=a e^{i\theta},\quad
g_w\sim\mathcal{CN}(0,sP_w),\qquad P_w=\operatorname{diag}(0.17,0.08).
\]
Integrating out \(g_w\) gives \(y\mid\theta,j,s\sim\mathcal{CN}(d_0ae^{i\theta},R_s)\), where
\[
R_s=nI+D_w(sP_w)D_w^H,\quad b_s=d_0^HR_s^{-1}y,\quad
\kappa_s=2a|b_s|.
\]
The phase-dependent part of the likelihood is \(\exp[\kappa_s\cos(\theta-\arg b_s)]\). With \(A(\kappa)=I_1(\kappa)/I_0(\kappa)\),
\[
m_{0,s}=a\,e^{i\arg b_s} A(\kappa_s),\qquad
v_{0,s}=sP_0[1-A(\kappa_s)^2].
\]
The **complete** trajectory log likelihood is
\[
L_{j,s}(y)=-N\log\pi-\log\det R_s-y^HR_s^{-1}y
-a^2 d_0^HR_s^{-1}d_0+\log I_0(\kappa_s).
\]
This is the quantity needed for model comparison. The original estimator legitimately drops terms common to its own trajectory normalization. Restoring those terms matters when comparing branches with different implementations or units. Compute \(\log I_0(\kappa)=\kappa+\log i0e(\kappa)\). Use zero mean at \(b_s=0\), and compute \(v_0\) as \(sP_0(1-A)(1+A)\), with a large-\(\kappa\) asymptotic fallback if the ratio rounds to one. [Anchor implementation](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/estimator.py:43>) · [Bessel integral identities](https://dlmf.nist.gov/10.32) · [Large-argument expansions](https://dlmf.nist.gov/10.40)

Suppose the simulator is unscaled, so the generating value is \(s=1\). Dividing by the Frame-A gauge and then using fixed canonical `POWERS` amounts to fitting the physical prior with \(s=s_A\). The exact distortions at a given node are
\[
\frac{\kappa_{s_A}}{\kappa_1}
=\sqrt{s_A}\frac{|d_0^HR_{s_A}^{-1}y|}{|d_0^HR_1^{-1}y|},\qquad
\frac{v_{0,s_A}}{v_{0,1}}
=s_A\frac{1-A(\kappa_{s_A})^2}{1-A(\kappa_1)^2}.
\]
The weak-ray shrinkage changes too. In general even \(\arg b_s\) can change.

For scale, I evaluated the following **illustrative orthogonal-column case** at 9 dB, with \(\|d_0\|^2=32\) and \(|d_0^Hy|=32\sqrt{0.75}\). These are equation values, not observed Frame-A fade quantiles.

| Assumed \(s\) | \(\kappa_s\) | \(I_1/I_0\) | Physical \(|m_0|\) | Physical \(v_0\) |
|---:|---:|---:|---:|---:|
| 0.5 | 269.604 | 0.998144 | 0.611236 | 0.001391 |
| 1 | 381.278 | 0.998688 | 0.864889 | 0.001967 |
| 2 | 539.208 | 0.999072 | 1.223609 | 0.002782 |

The ratio is already close to one throughout. The material distortion comes from imposing the wrong **circle radius**. A highly concentrated phase posterior can coexist with a badly biased gain magnitude.

### Gauge choice

A coordinate normalization and a physical power prior are separate choices. I recommend the following implementation.

1. **Use B-only noise units for reset computation.** Let \(u_B\) be the unit phase of the largest-magnitude anchor pilot observation times the conjugate of its known pilot symbol, with the lowest sample index breaking ties. Set
   \[
   c_B=\sqrt n\,u_B,\qquad y_c=y/c_B,\qquad n_c=1,\qquad \rho=s/n.
   \]
   In these units the dominant radius is \(\sqrt{\rho P_0}\) and the weak covariance is \(\rho P_w\). This uses the supplied noise variance at every SNR. An equivalent implementation can use \(c=\sqrt{\hat s_B}u_B\), provided it transforms **both** the noise and the gain prior.
2. **Infer the B power scale from the anchor.** The Type-II estimate is
   \[
   \hat\rho=\arg\max_{\rho>0}\operatorname{LSE}_j
      [\log\pi_j+L_{j,\rho}(y_c)],\qquad \hat s_B=n\hat\rho .
   \]
   Use a bounded, dimensionless log-scale search with deterministic tie handling. The marginal can attain its supremum at zero for weak observations, so an unrestricted strictly positive maximizer cannot be assumed. A simpler plug-in is
   \[
   E_B=\|y_{10}\|^2/N,\qquad
   \hat s_B=\max(E_B-n,\epsilon E_B,\epsilon n).
   \]
   It is cheap and homogeneous. It treats the two random weak-ray powers as information about the global scale. Even with orthogonal pilot columns and no receiver noise, their summed power has standard deviation \(s\sqrt{0.17^2+0.08^2}\simeq0.188s\). Type-II fitting can exploit the dominant circle separately.
3. **For the primary three-class evidence score, integrate over scale.** Choose a proper, frozen dimensionless prior on \(\rho\) and positive quadrature weights \(\eta_l\). Compute
   \[
   L_j(A)=\operatorname{LSE}_l[\log\eta_l+L_{j,\rho_l}(y_{10,c})].
   \]
   Average the conditional gain moments with the corresponding posterior scale weights. This avoids treating an anchor-fitted maximum as pre-anchor predictive evidence. A concrete development choice is log-uniform \(\rho\in[1/4,64]\), with log-scale quadrature refined until doubling the order has negligible effect on evidence differences and Wave-1 decisions. That prior is a proposed receiver choice. It is not part of the locked channel generator. The B-energy and Type-II versions are useful cheaper ablations.
4. **Keep the Frame-A gauge for the inherited prior.** Build it from A’s pilot cells as in P9, then convert its physical mean and variance to the same B scoring units. For `full_reset`, A and B are drawn independently. A’s power estimate is statistically extraneous to the B-only radius, and feeding it into the rebuild fails the existing `assert_isolation`, which sets all of `y_a` to `50-150j`. The two branches can use different information while being compared in common units. [Frame-A gauge and prior construction](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/gating.py:89>) · [Independent reset draw](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/channel.py:266>) · [Isolation test](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/eval_rebuild.py:68>)

For nonzero anchor observations, \(c_B(\alpha y,|\alpha|^2n)=\alpha c_B(y,n)\). The canonical observations, scale posterior, trajectory weights, branch choices and bits therefore agree in exact arithmetic. Return
\[
m=c_Bm_c,\qquad \Sigma=|c_B|^2\Sigma_c,\qquad
E[gg^H]=|c_B|^2 E[g_cg_c^H].
\]
Probe means obey the first rule and probe variances the second. A score on \(M\) complex samples evaluated in local units converts to physical log density by subtracting \(M\log|c|^2\). Evaluating all branches in the same units cancels this term.

At an all-zero anchor there is no equivariant unit phase that can be selected from that anchor. Use \(u_B=1\) as a computational convention and retain rotation-equivariant complex arithmetic. The anchor-only reset means are then zero. Reject nonfinite authorized inputs or nonpositive \(n\). Avoid a fixed additive floor in physical units. The `1e-300` floor in the old A gauge breaks homogeneity if it activates. An exact-arithmetic invariance proof also cannot guarantee a \(10^{-12}\) floating-point bound at every resampling boundary or decision tie.

## (b) Epoch transport, tracker projection and persistence

### Exact transport and an explicitly optimal projection

For GL trajectory \(k\), put \(\beta_k=(f_{e,k}-f_{s,k})/59.96875\) and \(F_k(t)=f_{s,k}t+\tfrac12\beta_kt^2\), in cycles. With \(O=45.984375\), \(E=30\),
\[
d_k=\exp\{2\pi i[F_k(E)-F_k(O)]\}
=\exp\{2\pi i[(f_{s,k}+\beta_kO)(E-O)+\tfrac12\beta_k(E-O)^2]\}.
\]
For joint node \(j=(k_0,k_1,k_2)\), use \(D_j=\operatorname{diag}(d_{k_0},d_{k_1},d_{k_2})\). Then
\[
m_j^E=D_jm_j^O,\quad \Sigma_j^E=D_j\Sigma_j^OD_j^H,\quad
S_j^E=D_j S_j^O D_j^H,\quad w_j^E=w_j^O.
\]
Phase transport does not change the mass of a trajectory. Map its **global signed start/end frequencies** to the legacy grid. The tracker itself adds `CHIRP * epoch` when evaluating the local frequency. Mapping the local frequency at \(O\) as if it were `FREQ` would apply that shift twice. [GL3 geometry](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/estimator.py:27>) · [Tracker state and evolution](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p8/tracker.py:12>)

There is no loss-independent optimal projection onto `[3,128]`. For an epoch-moment-preserving export, define a **sign-preserving bilinear kernel** \(K_{kh}\) on the surrounding two start and two end magnitudes of the tracker grid. Its rows sum to one. For ray \(a\),
\[
\begin{aligned}
q_{ah}&=\sum_j w_j K_{k_a(j),h},\\
m_{ah}&=q_{ah}^{-1}\sum_j w_j K_{k_a(j),h}m^E_{ja},\\
v_{ah}&=q_{ah}^{-1}\sum_j w_j K_{k_a(j),h}
  \big[\Sigma^E_{j,aa}+|m^E_{ja}-m_{ah}|^2\big].
\end{aligned}
\]
Return `State(30, log(q), m, v)`. For fixed \(K\), these values minimize expected negative log score over a categorical grid distribution and a conditional circular Gaussian gain. They preserve the marginal epoch mean and second moment exactly. For zero-mass cells use `log_weight=-inf`, finite filler gain and strictly positive filler variance, or mask them during conditioning.

If the objective is waveform \(L^2\) error instead, the assignment can differ. For normalized sample weights \(\lambda_t\) and unit-modulus epoch-anchored evolutions \(u_h,u_k\), set \(z_{hk}=\sum_t\lambda_t u_h(t)^*u_k(t)\). The best single-grid approximation to \(g u_k\) uses coefficient \(z_{hk}g\), with loss \(|g|^2(1-|z_{hk}|^2)\). It need not preserve the gain at epoch 30. Neither projection preserves the full joint law. In particular, independent ray banks can create combinations of three trajectories that had negligible joint posterior mass.

### The preferred persistent reset state

Store the reset state as up to \(J=160\) **joint** trajectory indices `[J,3]`, normalized base masses `[J]`, anchor log evidences `[J]`, and conditional gain moments `[J,3]`, `[J,3,3]`. Keep the scale quadrature or its sufficient statistics internally. Obtain the within-node covariance directly during fitting. Reconstructing it later as `second - outer(mean,mean.conj())` wastes precision.

There is a cheap exact gain update for every retained node. Stack the known anchor and the distinct accepted virtual words in \(D=[d_0,D_w]\in\mathbb C^{Q\times3}\), \(Q=32(1+|V|)\). Let \(W\) be the diagonal per-sample precision. For one scale point, in consistent units, set
\[
\begin{aligned}
C_w&=[(sP_w)^{-1}+D_w^H W D_w]^{-1},\\
c&=C_wD_w^HWd_0,\qquad b=C_wD_w^HWy,\\
K&=W-WD_wC_wD_w^HW .
\end{aligned}
\]
Use the phase formula in (a) with \(b_s=d_0^HKy\). The posterior is
\[
m_j=\begin{bmatrix}m_0\\ b-cm_0\end{bmatrix},\qquad
\Sigma_j=
\begin{bmatrix}0&0\\0&C_w\end{bmatrix}
+v_0\begin{bmatrix}1\\-c\end{bmatrix}
       \begin{bmatrix}1\\-c\end{bmatrix}^{H}.
\]
For evidence, the \(R\) of part (a) is now \(W^{-1}+D_w(sP_w)D_w^H\). Use the determinant lemma and a \(2\times2\) Cholesky solve. There is no need to materialize \(K\) or a \(Q\times Q\) inverse. At multiple scales, average central covariances with the within-plus-between formula.

Let \(w_j^0\) be the frozen anchor-plus-auxiliary trajectory masses. After accepting the set \(V\), compute
\[
\boxed{\quad
\log w_j(V)=\mathrm{normalize}_j\{
  \log w_j^0+L_j(A\cup V)-L_j(A)\}.\quad}
\]
Here \(L_j\) includes scale marginalization if enabled. Recompute the gain moments from \(A\cup V\). The ratio is the predictive increment for the virtual observations. At \(V=\varnothing\) it is exactly one. Replaying a version with the same ledger gives the same state. The within-node update is exact for declared known words and their assigned Gaussian precisions. The overall receiver still uses a compressed quadrature, a frozen auxiliary composite score, and decision-directed words.

This also gives a safe legacy alternative. If the immutable projected base already includes the anchor, call a generalized `condition_sc_weighted(projected_base, virtual_rows_only, ...)`. If it starts before the anchor, condition on `anchor + virtual_rows`. Simply subtracting an anchor log weight while retaining anchor-updated means and variances double-counts gain information. Moreover, the auxiliary score uses the **anchor-conditioned** second moment, so subtracting the pilot log likelihood alone cannot recover an independent auxiliary-only posterior. [Estimator weighting](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/estimator.py:136>) · [Weighted conditioner](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:723>)

### Stable variances and the probe

For ray \(a\), with propagated conditional mean \(m_{ja}(t)\),
\[
\mu_a(t)=\sum_j w_jm_{ja}(t),\qquad
v_a(t)=\sum_j w_j[\Sigma_{j,aa}(t)+|m_{ja}(t)-\mu_a(t)|^2].
\]
Use this both for individual wave rows and for final frame readout. The sibling repair has the needed stable scalar form. The sibling turbo row extractor still subtracts two second moments. [Stable repair moments](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:237>) · [Turbo row moments](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:957>)

For probe sample \(n'\), let \(x\in\mathbb C^3\) contain the three *known transmitted* delayed probe symbols. Let \(m_j,\Sigma_j\) now be propagated to that sample. Use
\[
p_j=x^Tm_j,\qquad p=\sum_jw_jp_j,\qquad
v_p=n_0+\sum_jw_j\big[x^T\Sigma_jx^*+|p_j-p|^2\big].
\]
Evaluate the quadratic from the positive covariance factors above where possible. At every additional mixture layer, including folds, add the between-component term. Add receiver noise **once**. This gives `mix_probe_var >= n0` in every version, including version zero when no row unlocks. It retains cross-ray terms that `p9.gating.readout_sc` omits under its independent-ray approximation. Empirical calibration additionally requires probe NLL and a noisy-probe squared-error-to-predicted-variance check. The floor alone cannot establish calibration. [Existing probe readout](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/gating.py:328>)

## (c) What Wave 1 can identify

A row-wide rotation sends \(H_r\mapsto e^{i\phi_r}H_r\) and leaves \(C_r=H_rH_r^H+nI\) exactly unchanged. A physical common Doppler increment sends \(C_r\mapsto D_r C_rD_r^H\). Along the unit direction \((1,1,1)/\sqrt3\), its derivative is
\[
C'_{nm}=\frac{2\pi i}{\sqrt3}\frac{n-m}{N}C_{nm}.
\]
The row-center time cancels. For the local proper-Gaussian covariance score the curvature is \(\sum_r\operatorname{tr}(C_r^{-1}C'_rC_r^{-1}C'_r)\). The reported **2185** is the median relative-to-common ratio over 48 reset links with true gains and slopes held fixed. It is neither a universal condition number nor an estimate of the Wave-1 decoder’s posterior odds. [Phase identity](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/phase_check.py:88>) · [Curvature calculation](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/frequency_sensitivity.py:18>)

**Premature common-mode lock is possible.** The estimator resamples from `logpilot + .75*coarse` and applies the exact covariance score only on the retained support. Importance correction cannot recover an omitted mode. The eight-link comparison against all 5832 nodes tested reconstructed waveform error. It did not test the tail probability of a missing mode changing a parity-certified decision. [Compression implementation](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/estimator.py:145>) · [Quadrature comparison](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/README.md:73>)

I would make **mode coverage** the first change and temperature the second.

- Split proposals by actual signed Ray-0 frequency and, if needed, by common frequency at \(O\). Allocate retained samples within materially weighted strata and carry their original proposal mass in the importance correction. The GL3 code interleaves signs in the flattened bank. CLI-2’s assumption that the first half of the array has positive frequency is valid only for the legacy tracker. When splitting a joint posterior, condition **all three rays** on the split event.
- Start with \(\beta=1\). For a registered ablation, bin the relative frequencies and slopes as \(u\), and denote common-mode coordinates by \(c\). Preserve the full auxiliary posterior mass of each relative bin \(W(u)\), while replacing its conditional law for Wave-1 proposals by
  \[
  W_\beta(c\mid u)\propto W_{\rm pilot}(c\mid u)
                   \exp[\beta\,\ell_{\rm aux}(u,c)],\quad 0\le\beta\le1.
  \]
  Define bins and aggregation before looking at outcomes. Generate these proposals **before** modes are discarded. Rescore proposed words against the registered predictive model. There is no analytic \(\beta\) that maximizes correct unlocks from the curvature ratio. Choose it on independent development links using correct Wave-1 unlocks at a bounded false-accept rate, with \(\beta=1\) as the comparator.
- For ambiguous rows, take the union of the strong decoder’s candidates across active modes and the generator-neighbor checks. Deduplicate words and check parity. For a word \(x\), a common score across modes is
  \[
  \ell(x)=\operatorname{LSE}_b\left[
       \log p_b-\log\det V_b(x)
       -(y-H_bx)^HV_b(x)^{-1}(y-H_bx)\right].
  \]
  A diagonal \(V_b\) is a practical waveform score. Include its log determinant when it varies with mode or word. Compare the best and second **distinct words** after combining modes. Two modes explaining the same word are supporting evidence for that word. The existing `_scan_best_and_runner_up_u64` certifies parity and finds a runner-up on its finite search set. [Candidate scan](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:207>) · [Existing split-mode scoring](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:1556>)

**The coherent information appears once a word is available.** Write \(z_n=(Hx)_n\), with the relative channel held fixed and diagonal variance \(v_n\). The conditional ML phase correction is
\[
\hat\phi=\arg\sum_n z_n^*y_n/v_n .
\]
For a shared phase intercept and common Doppler correction, let \(e_n=|z_n|^2/v_n,\ u_n=t_n-O\). The local information matrix, summed over usable known or virtual observations, is
\[
J=2\sum_n e_n
\begin{bmatrix}1\\2\pi u_n\end{bmatrix}
\begin{bmatrix}1&2\pi u_n\end{bmatrix}.
\]
Eliminating the phase intercept leaves \(2(2\pi)^2\sum_n e_n(u_n-\bar u_e)^2\) for frequency. The neighbor centers are \(O-1.5\) and \(O+1.5\). For example, a common error of \(0.01\) cycles per useful symbol produces opposite center-phase errors of about \(0.0942\) rad. This is first-order coherent sensitivity even when row covariance has very little common-frequency curvature. The result is local and conditional on the correct word and a shared trajectory law. Two neighbors cannot determine arbitrary unrelated \(\phi_r\) on every other row.

**Acceptance requires more than 4.5 nats.** CLI-2 gives zero weight at or below 4.5 for BP convergence and 6.5 otherwise. Its weight is
\[
\alpha_r=0.75\left[1-e^{-(m-m_{\min})/5.5}\right]_+
               w_{\rm fit}w_{\rm resid}.
\]
With both guards equal to one, \(\alpha_r\ge0.08\) requires \(m\ge5.1204\), or \(7.1204\) without convergence. The residual guard uses the larger of \(\sum|e_n|^2/\sum v_n\) and \(\operatorname{mean}(|e_n|^2/v_n)\), and vanishes at 3.5. For a joint state use the projected known-word variance from (b). Freeze each accepted word and its pre-update weight, then use
\[
n_r^{\rm eff}=n_0/\alpha_r+0.15n_0.
\]
Evaluate both neighbors against the same pre-wave snapshot before committing either. This limits self-reinforcement. It cannot guarantee zero error propagation. [Weight rule](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:428>) · [Predictive residual](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:886>)

The saved JT run reports false assimilations of **16/1005 continuous, 7/286 partial shift, and 14/823 full reset**. Its evaluator compares truth with the *final* `decoded_info` for accepted rows, while the receiver can subsequently re-decode a row. P11 should measure false assimilation against the **word actually injected**. Also report the fraction of links with a correct Wave-1 unlock and subsequent target BLER conditional on a correct, incorrect, or absent first unlock. [Stored development results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/results/p10a2_dev.json>) · [Existing false-assimilation calculation](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/p10a2_dev.py:155>)

## (d) Persistent partial-shift repair and comparable regime scores

### Put the shift in the branch dynamics

Represent an inherited branch by `(prior_at_30, changed_ray, delta, onset_row)`. For \(t=\texttt{clock(T,start_row=T)}[r,n]\) and \(\tau_b=30+1.5b\), define
\[
M_a(t)=\exp\!\left(2\pi i\,\mathbf1_{a=a_*}\delta(t-\tau_b)_+\right).
\]
For no change, \(M_a=1\). For the boundary shift \(b=0\) this is the sibling bank’s `exp(2j*pi*delta*TIMES_REL)`. For \(b=6\), \(\tau_b=39\). All useful samples in rows 0–5 follow the inherited trajectory. Rows 6–18 accrue phase from time 39. The simulator uses this same phase-continuous law. The existing 74-entry bank has 72 single-ray shifts and only boundary onset. A blind bank containing both candidate onsets has \(2+3\cdot24\cdot2=146\) entries. If row 6 is supplied as a design candidate, label the experiment as testing inference among known possible onsets. For generalization to unseen onset locations, freeze a broader onset prior independently of the OOD test. [Simulator onset law](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/channel.py:86>) · [Existing bank](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:69>)

The minimal conditioner change is to multiply the delayed symbol vector by \(M_a\) **before computing its precision-weighted conjugate**:
\[
D_{ah,q}=R_{h,q}\underbrace{M_a(t_q)x_{r_q}[(n_q-d_a)\bmod N]}_{\widetilde x_{a,q}} .
\]
Use \(\widetilde x\) in the initial response, matched filter and reconstructed response at every coordinate sweep. Since \(|M_a|=1\), the existing scalar energy
\(\sum_q |x_q|^2/n_q^{\rm eff}\) is preserved. Explicitly, given the residual excluding ray \(a\),
\[
v'_{ah}=(v_{ah}^{-1}+D_{ah}^HWD_{ah})^{-1},\quad
h_{ah}=m_{ah}/v_{ah}+D_{ah}^HW e_{-a},\quad m'_{ah}=v'_{ah}h_{ah},
\]
\[
\log w'_{ah}=\mathrm{normalize}_h\{
 \log w_{ah}+\log(v'_{ah}/v_{ah})-|m_{ah}|^2/v_{ah}+v'_{ah}|h_{ah}|^2\}.
\]
These are the Gaussian coordinate updates used by the factored tracker. Four sweeps are a finite coordinate-ascent approximation to joint inference. Refit from the immutable pre-anchor prior on `anchor + accepted_rows`, once per distinct row. [Weighted update equations in code](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/turbo_assimilate.py:771>)

Use the **same** dynamics object in pilot evidence, auxiliary scoring, sign-split trials, row prediction, residual checks, virtual conditioning, final decoding and probe readout. A deterministic rotation multiplies a marginal mean by \(M_a(t)\), preserves its variance, and multiplies cross covariance by \(M_a(t)M_b(t')^*\). The old `refit_rotated_state` returns arrays of moments and discards its internal posterior state. Its return value cannot implement persistence. [Existing rotated refit](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:321>)

### Score the same evidence under all three classes

For fold \(f\), use the same anchor \(A=\{10\}\), auxiliary rows \(U_f\), reference units, and auxiliary likelihood temperature \(\lambda\) for all branches. Start with \(\lambda=1\). For class configuration \(h\),
\[
S_h=\log p_h(y_A\mid\text{authorized history})
+\lambda\sum_{r\in U_f}\log\mathcal{CN}(y_r;0,C_{hr\mid A}),
\qquad
C_{hr\mid A}=nI+E_h[H_rH_r^H\mid y_A].
\]
For a retained reset trajectory mixture, evaluate the product of row scores **inside** the trajectory sum. A shared trajectory across five rows matters. This is a composite likelihood because uncertain gains are shared and coded QAM is not Gaussian. For the inherited tracker, its pre-anchor moment-matched Gaussian pilot likelihood and coordinate-conditioned auxiliary covariance give a corresponding declared approximation. The sibling bank currently uses pre-B inherited moments on payload for no-change and shift, but anchor-conditioned clear moments on payload for reset. That asymmetry changes the comparison. [Existing pre-update pilot score](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:620>) · [Existing payload-arm construction](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:783>)

For exact second-moment assembly within the declared model, let \(P_{aj}\) map an input word to delayed, trajectory-rotated samples for ray \(a\). For reset node \(j\),
\[
C_{jr}=nI+\sum_{a,b} S_{j,ab}\,P_{aj}P_{bj}^H .
\]
This includes the lag-2 and lag-4 cross terms. Because all delays are even, a parity permutation reduces the solve to two \(16\times16\) Hermitian systems. With an independent-ray tracker the same-ray terms require its second moments, and the distinct-ray terms factor into its means. The second moment follows from \(E[xx^H]=I\). The associated Gaussian density is the scoring approximation. [Exact payload covariance assembly](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/estimator.py:116>)

**Keep the proposal normalizer for the reset class.** Write \(b_j=\log\pi_j+L_j(A)\), \(c_j\) for the coarse score, \(e_j\) for the exact auxiliary score, and \(n_j\) for midpoint resampling counts out of \(M=160\). With the original single systematic proposal,
\[
S_R\simeq
\operatorname{LSE}_j(b_j+0.75c_j)
+\operatorname{LSE}_{j:n_j>0}
 \left[\log(n_j/M)+\lambda e_j-0.75c_j\right].
\]
The second term alone suffices to normalize posterior trajectory weights. It gives the wrong **class evidence** if the first term is omitted. For stratified mode allocation, replace \(n_j/M\) by \(Q_g n_j/M_g\), where \(Q_g\) is the original proposal probability of stratum \(g\) and \(M_g\) its assigned count. Include common likelihood constants consistently across branches.

Use top-level priors \(1/3,1/3,1/3\). Integrate over shift configurations:
\[
B_C=S_C+\log(1/3),\quad
B_S=\log(1/3)+\operatorname{LSE}_{h\in S}[\log\pi(h\mid S)+S_h],\quad
B_R=S_R+\log(1/3).
\]
Taking a maximum over 72 or 144 shifts without a multiplicity adjustment favors false repairs. If refining \(\delta\), keep the refinement inside its selected grid interval, require the local quadratic to be concave, and retain the discrete value at boundaries or failed fits. Recompute the refined branch’s score and state with that same \(\delta\).

For continuous protection, use the fold-local inherited/JT wavefront as the incumbent. Register a switch statistic such as \(T_f=\max(B_S,B_R)-B_C\). Calibrate a threshold from independent continuous links using the distribution of \(\max_f T_f\), then apply that **fixed threshold to each fold separately**. This controls the opportunity for either fold to switch without allowing one fold’s observations to steer the other at inference. A rank-based calibration with exchangeable null links can control the marginal false-switch probability at its chosen level. Its resolution is limited by the calibration sample size. If the threshold is crossed, retain the selected repair or reset state through subsequent waves.

The strong isolation schedule also restricts the incumbent. Thus inheriting its code cannot guarantee its historical 5.18% BLER. The arithmetic target is informative: at continuous \(5.18\%\) and reset \(10.00\%\), partial shift must reach **23.82%** for a 13% average, or **14.82%** for a 10% average. These are materially different engineering targets.

## (e) Fold firewall, unified output and runtime

### Exact row ownership

| Fold | Auxiliary rows used by initializer | Core rows scored by this fold | Available fresh virtual rows under whole-fold isolation |
|---|---|---|---|
| 0 | 2, 5, 8, 13, 16 | 1, 3, 4, 6, 7, **11**, 12, 14, 15, 17, 18 | **0, 9** |
| 1 | 1, 4, 7, 12, 15 | 2, 5, 8, 13, 16 | **0, 3, 6, 9, 11, 14, 17, 18** |

The last column excludes the fold’s targets, anchor, probe and auxiliary factors already consumed. Intersect each scheduled ring with this column. Thus Fold 0’s first ring is `(9,)` and Fold 1’s is `(9,11)`. A Fold 0 certificate for row 11 cannot be sent to Fold 1 either: its construction depends on `AUX1`, which is exactly Fold 1’s target set.

There is a design choice concerning `VALID=(0,9)`. **For the proposed primary, use these two extra payloads for decision-directed assimilation after class selection on anchor plus aux.** Keep the original validation-gated receiver as a separate comparator. The two words then provide no independent validation of the refined primary state. If the experiment requires both words to remain untouched validation observations, remove both from the last column. Fold 0 then has no fresh virtual helper at all. A more expensive extension can reuse an auxiliary row by removing its covariance factor, rebuilding a proposal that excludes it, and replacing that factor with a certified known-word factor. Multiplying both factors double-uses its waveform. The old `code_check.py` also returns a finite-list score on the two validation words, with both arms scoring the same deduplicated list. Its statistic is not a free additional independent likelihood after those words have been assimilated. [Validation score](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p10/payload_reconstruction/code_check.py:11>)

Implement the firewall at the interface level:

- `estimate_payload_posterior` receives copies or read-only views of `y10[32]`, `y_aux[5,32]`, their authorized indices and `n0`. It has no `y_a`, target-array, `VALID` or probe handle. For its pilot-only mode, omit `y_aux` entirely. Its moments for **all predicted times** can then be bitwise unchanged under target and A poisoning.
- The class scorer additionally receives A pilot cells and the same fold-local B evidence. Each wave sees only its eligible helper observations. Core target waveforms enter a separate final row decoder, after the state for that fold is frozen. Target channel estimates must be invariant to poisoning the entire target set. Decoded bits and LLRs for target \(r\) necessarily depend on its own \(y_r\).
- Never copy or normalize a complete B array and then derive an evidence statistic from it. Use integer-indexed allowed rows first. The existing `compute_cov_bank` copies all payload rows, and `readout_sc` reads and checks all data rows. Both need row-restricted entry points inside the fold. Build probe predictions only from the known transmitted probe sequence. Set pilot/probe LLR slots to finite zeros. [Bank reads](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/cov_repair.py:713>) · [Full-frame readout reads](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/gating.py:348>)

### One `AssimilationResult`

Use the owning fold’s frozen prediction, LLRs and final `strong_large` decision for each of the 16 core rows. Assign extra rows 0 and 9 a fixed output owner, for example Fold 0. For an extra row that was accepted, publish its frozen decision and pre-injection readout for that owner. For the probe, use a fixed equal-weight predictive mixture of the two fold distributions:
\[
p=\tfrac12(p_0+p_1),\qquad
v=\tfrac12(v_0+v_1)+\tfrac14|p_0-p_1|^2 .
\]
Use the same mixture law for probe-row ray moments. This is a declared ensemble of two predictors that share observations. Multiplying their posteriors as independent evidence would be unjustified. For a three-branch implementation define `history_weight` explicitly as the weight on branches carrying A history, including the shifted inherited branch; use the same definition for `probe_weight`.

Maintain a detailed internal ledger `(fold, row, injected_word, alpha, margin, version)`. The public flat result can carry a deterministic unique union of accepted rows in wave/row order. For each accepted row, report the weight and margin from one fixed representative accepting fold, for example the lower fold index. For other rows use zero weight and the output owner’s decoder margin. `decoded_info` always carries the owner’s delivered bits. For a core row used as a helper by the other fold, its delivered bits can differ from the injected word. **Use the internal ledger to evaluate false assimilation.** The flat contract cannot represent both words, or two different fold-specific weights on one row. The internal implementation can return `(result, trace)` to the evaluator while `run_p11_primary(obs)` returns just `result`. This preserves the public signature and keeps truth outside inference.

Final arrays are `mix_mean_ray: complex128[3,20,32]`, `mix_var_ray: float64[3,20,32]`, `mix_llrs: float64[20,32,4]`, probe mean/variance `[32]`, `history_weight[20]`, and binary `decoded_info[r]: [64]`. Report `n_rounds` as the number of scheduled rings processed across the common schedule, and `elapsed_ms` as actual end-to-end time. Exclude timing from deterministic comparisons. [P11 result contract](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p11/contract.py:337>)

**There is a literal audit compatibility issue.** The external auditor checks `isinstance(base, p10.contract.AssimilationResult)`. P11 currently defines a distinct dataclass. An otherwise perfect P11 output fails that check before the numerical comparisons. For direct compatibility, share one canonical class or return a compatibility subtype recognized by both packages, and verify both `isinstance` checks under the actual imports. If the P11 contract must stay independently packaged, run a P11-native copy of the independent audit with the P11 type, and also run the unmodified historical auditor through an explicit field-for-field adapter. Report which route passed. The two worktrees have overlapping `p10` package names, so import identity matters. [External audit type check](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/independent_ref.py:972>)

### Implementation pseudocode

```python
# p11/payload_rebuild.py

def estimate_payload_posterior(y10, aux_rows, y_aux, n0, config):
    validate_sparse_anchor_and_five_distinct_aux(aux_rows, y10, y_aux, n0)
    gauge = anchor_phase(y10) * sqrt(n0)
    y_anchor = y10 / gauge
    y_fit = y_aux / gauge
    trajectories, log_gl_prior = gl3_joint_bank()             # [5832, 3]

    # Exact circle/weak-Gaussian update for each scale and trajectory.
    # Marginalize scale before constructing each node's auxiliary covariance.
    anchor = fit_known_blocks_by_node(
        trajectories, rows=(10,), words=(PILOT[10],),
        received=(y_anchor,), noises=(1.0,), scales=config.scale_quadrature,
    )
    b = log_gl_prior + anchor.log_evidence                    # [5832]
    coarse = fft_diagonal_score(y_fit, aux_rows, anchor.second)
    proposal = make_proposal(b + 0.75 * coarse)
    idx, represented_mass = stratified_midpoints(
        proposal, budget=160, strata=config.mode_strata,
    )
    exact = parity_block_cov_score(
        y_fit, aux_rows, trajectories[idx], anchor.second[idx], noise=1.0,
    )
    correction = log(represented_mass) + config.aux_temp * exact - 0.75 * coarse[idx]
    class_log_evidence = logsumexp(b + 0.75 * coarse) + logsumexp(correction)

    return ResetBase(
        gauge=gauge, node_ids=trajectories[idx],
        logw0=log_softmax(correction), anchor=anchor.subset(idx),
        log_evidence=class_log_evidence, aux_rows=aux_rows,
    )


def refit_reset(base, virtual_ledger):
    if not virtual_ledger:
        return base.anchor_state_with_weights(base.logw0)
    blocks = unique_anchor_and_virtual_blocks(base, virtual_ledger)
    post = fit_known_blocks_by_node(base.node_ids, **blocks)
    logw = log_softmax(base.logw0 + post.log_evidence - base.anchor.log_evidence)
    return JointState(base.node_ids, logw, post.mean, post.covariance)
```

Here `represented_mass` means \(Q_g n_j/M_g\), **not** the post-correction weight. `fit_known_blocks_by_node` caches the anchor Gram matrices and adds virtual \(D^HWD,D^HWy,y^HWy\) and noise log determinants. Its zero-virtual path must reproduce initialization. The public wrapper around the low-level array API supplies only the authorized anchor and auxiliary rows.

```python
# p11/wavefront_pipeline.py

def run_p11_primary(obs):
    result, _ = run_p11_with_trace(obs)
    return result


def run_p11_with_trace(obs):
    validate_metadata(obs, training=(10,), probe=19, finite_positive_n0=True)
    a_prior = inherited_prior_from_A_pilot_cells(obs)
    fold_outputs = []

    for f in (0, 1):
        fit = read_only_anchor_and_aux(obs, AUX[f])
        reset = estimate_payload_posterior(**fit, config=FROZEN)
        classes = score_no_change_shifts_reset(a_prior, reset, fit, FROZEN)
        branch = choose_with_frozen_continuous_threshold(classes, FROZEN)

        # The branch owns immutable dynamics and a refit convention.
        # Reset subtracts anchor evidence. Inherit/shift starts pre-anchor.
        ledger = {}
        state = branch.refit(ledger)
        helpers = DATA_ROWS - set(TARGETS[f]) - set(AUX[f])

        for wave in FINE_SCHEDULE:
            pending = {}
            for r in rows_in_wave_order(wave, helpers):
                prior_prediction = branch.predict(state, r)
                decision = parity_margin_decode(
                    obs.y_b[r], prior_prediction, config=FROZEN,
                )
                accepted = assess_preupdate_word(decision, prior_prediction, obs.y_b[r])
                if accepted:
                    pending[r] = freeze_word_weight_margin_and_readout(accepted)
            if pending:
                ledger.update(pending)
                state = branch.refit(ledger)

        predicted = branch.predict(state, TARGETS[f])
        delivered = strong_large_on_each_owner_row(obs, TARGETS[f], predicted)
        probe = branch.predict_known_word(state, row=19, tx=PILOT[19])
        fold_outputs.append(save_fold(delivered, predicted, probe, ledger, state))

    return assemble_owner_rows_and_probe_mixture(fold_outputs)
```

`DATA_ROWS` here excludes 10 and 19. All caches that depend on observations must be link- and fold-local. Geometry-only dictionaries can be shared. Keep `complex128/float64`, deterministic sorting and tie rules, log-domain normalization, Hermitian Cholesky checks, and signed-frequency masks. For comparison with the locked decoder, run the specified large-list, 20-iteration, PIC-enabled final decoder on every core owner row. CLI-2’s fast-lock shortcuts otherwise change the delivered decoder on some rows.

### Formal size and acceptance

At \(G=16\), the three-ID run contains **1,536 links, 3,072 fold fits, and 24,576 scored core blocks**. Pool the 11 and five owner-row error counts before computing a link BLER. Equal weighting of the two fold percentages would overweight the five-row fold.

For the original trajectory budget, the coarse stage visits \(3072\cdot5832\cdot5\cdot16=1,433,272,320\) trajectory/row/half-bin combinations. The exact payload stage has at most \(3072\cdot160\cdot5\cdot2=\mathbf{4,915,200}\) complex \(16\times16\) factorizations. One packed fold tensor at maximum support occupies about 6.25 MiB before temporaries. Factor once for both the quadratic and log determinant. Marginalizing scale before auxiliary scoring multiplies the small pilot fits by the scale quadrature order and leaves the number of payload factorizations unchanged. Budget separately for the shift bank, mode-specific candidate lists and repeated wave fits. Historical JT timings were about 211, 214 and 161 ms/link on the three ID regimes. Those are P10 measurements. A P11 wall-time estimate requires a timed end-to-end development batch. [Historical timings](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/results/p10a2_dev.json>) · [Formal cluster protocol](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/PROTOCOL.md:23>)

I would require four checks before interpreting formal BLER:

1. **State identities.** Zero virtual rows reproduce initialization. Replaying the same ledger is idempotent. Shift \(\delta=0\) reproduces ordinary dynamics. At \(b=6\), pre-onset rotation is exactly one. Compare the joint probe quadratic against an independent dense calculation, including a case with nonzero cross-ray covariance.
2. **Isolation.** Poison `y_a`, targets, `VALID`, and probe for the standalone rebuild. Check its support, weights and moments. For each composed fold, poison all that fold’s targets and compare the frozen pre-target model, helper ledger and branch selection. Separately poison row 19 with the auditor’s finite sentinels and require exact zero changes in scientific outputs and absence from all four row containers.
3. **Equivariance and calibration.** Run the independent audit’s two complex scales, compare all seven `GateOutput` fields in their correct units, compare bits and accepted order exactly, and demand measured maximum error **strictly below** \(10^{-12}\). Include no-unlock, shift-active, mode-split and post-virtual cases. Check noisy-probe calibration and actual injected-word false assimilation separately. [Independent audit comparisons](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-p10/p10/independent_ref.py:1123>)
4. **Paired non-inferiority.** For each of the 16 continuous clusters compute \(d_g=\mathrm{BLER}_{P11,g}-\mathrm{BLER}_{dense,g}\) on the same 32 links and 16 core words. Require
   \[
   \bar d+1.75305\,s_d/\sqrt{16}<0.01.
   \]
   The historical \(-2.35\) pp upper bound used four development clusters. It cannot certify P11. Freeze the scale rule, temperature, switch threshold, helper policy and candidate budget before opening the P11 formal seeds. [NI definition](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p9/PROTOCOL.md:40>) · [P11 seed namespaces](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer/p11/contract.py:59>)

I inspected the requested files in both worktrees, checked the saved development records, and evaluated the illustrative Bessel values and operation counts. I did not change files or run receiver simulations. The phase transport, anchor-evidence cancellation, persistent shift law and probe noise floor follow from the equations above. **The 10–13% three-ID BLER, the continuous non-inferiority bound, the Wave-1 unlock rate and the floating-point audit threshold remain P11 measurements to establish.**