# M3 | Pilot sufficiency, conditional innovations, and fixed-slot pooling

Round 2 extends the independent-pilot result in Section 3 to correlated complex receiver noise, shared CFO uncertainty, and an already-conditioned causal state. The version 2 interface and Stage 6 use are in Sections 7–10.

## 1. Stage 3 release audit

The C4 generator sets the continuous drift to zero at the first window. It draws independent Gaussian increments with mean \(\omega\,dt\) and variance \(\sigma_\phi^2dt\) on the grid \(\{0,h,h+\ell,t_p\}\). Its two valid pilots share the same physical phase and have independent complex AWGN. At \(h=\ell=2,\ t_p=4\), the pilot clock equals the withheld-window clock. Each generator call takes one scalar \(t_p\). The schema can carry a different timestamp on each reference, but the seven C4 arms never exercise this case. The released C4 reader accepts only an ID, a channel Boolean and a matched sum. Its fitted encoder emissions and quadrature are tied to \(t_p=4,\ \omega=+0.35,\ \sigma_\phi=0.10,\ \mathrm{SNR}=2\,\mathrm{dB}\). [Generator and arms](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Schema](../../stage3_ministral_cache/schema.py) · [Reader](../../stage3c/A/q2_reader.py) · [Stage 3C derivation](../../stage3c/A/Q2.md)

A copy has the same acquisition ID and samples. A fresh acquisition has a new ID and independent noise conditional on the shared path. The schema checks duplicate channel flags, timestamps within \(10^{-12}\), and exact sample equality. The weighted pool also checks amplitude and noise variance, which affect the likelihood. Stage 3D's distractor has a separate random quadrant, phase, and noise. Its density cancels when calculating the target posterior. These results condition on public, exogenous acquisition design and correct channel declarations.

Q2 targets the original transmitted \(X_1\). The repeat protocol has \(X_3=X_1\). Q3_after_pilot targets the **sector of the noisy apparent matched sum** of the withheld third observation. This is a different random variable and requires the phase posterior at \(h+\ell\). [Prompt and sector definitions](../../stage3_ministral_cache/schema.py) · [Label construction](../../stage3d_qwen_signed_phase/train_eval_stage3d.py)

## 2. Theorem 1 | Same-clock sufficiency and minimality

Write \(b=\pi/2\), \(\Theta_t=\varphi_0+\omega t+B_t\), \(B_t=\sigma_\phi W_t\). The physical carrier angle is \(bG+\Theta_t\). C4 takes \(\varphi_0=0\) in its chosen frame. With independent uniform \(X_1,X_2,G\), the variables \(U=X_1+G,\ V=X_2+G,\ G\) are independent and uniform modulo four. For \(n_m\) pilot samples of per-sample amplitude \(A_m>0\), define

\[
T_m=n_m^{-1/2}\sum_n y_{mn},\qquad \alpha_m=\sqrt{n_m}A_m,\qquad
T_m\mid G,\Theta_{t_{p,m}}\sim
\mathcal{CN}\!\big(\alpha_m e^{i(bG+\Theta_{t_{p,m}})},\sigma_m^2\big).
\]

Here \(\sigma_m^2=\mathbb E|\eta_m|^2\) is complex matched-noise variance. The orthogonal sample coordinates are ancillary under the white circular model. Known nonwhite noise calls for the corresponding whitened matched inner product.

Deduplicate by ID and retain only valid acquisitions. Expanding the density at a common time \(p\) gives

\[
S=\sum_{m\in V_{\rm ref}}\frac{\alpha_m}{\sigma_m^2}T_m,\qquad
\prod_{m\in V_{\rm ref}}p(T_m\mid g,\theta)
=C(\{T_m\},\mathrm{design})\exp\{2\Re[S e^{-i(bg+\theta)}]\}.
\]

With matched power SNR \(\rho_m=\alpha_m^2/\sigma_m^2\), the weight is \(\rho_m/\alpha_m\). It scales with **power SNR** at fixed amplitude. It scales as \(\sqrt{\rho_m}\) at fixed noise standard deviation, or when the observation is standardized to \(T_m/\sigma_m\). In C4, \(n_m=3,\ \alpha_m=1,\ \rho_m=10^{0.2}\), so the reader uses \(2\rho_m\sum_m T_m\) in the exponent.

For fixed nonempty design, the ratio of densities of two datasets is constant in physical phase precisely when \(\Re[(S-S')e^{-i(bg+\theta)}]\) is constant. Comparing antipodal phases at both quarter-turn axes forces \(S=S'\), even when only four phase values are available. The likelihood-ratio criterion proves that **one complex number is minimal sufficient for the physical pilot likelihood**. Factoring and integrating proves sufficiency for \((X_1,X_3)\) and for the joint transmitted-label and withheld-sector target.

For a fixed prior and discrete target, the exact minimal Bayesian statistic is the target posterior vector, up to a bijection on its image. A statistic preserves all target posteriors exactly if that vector is measurable from it. Likelihood minimality before nuisance marginalization does not by itself imply global injectivity of the target posterior map. Two real coordinates are locally necessary at C4 among differentiable compressions. Define \(A_1(z_1)=\sum_u i^u P(U=u\mid z_1)\) and \(M_1(z_2)=\mathbb E[e^{i\Theta_p}\mid z_2]\). At zero pilot evidence, the first Fourier moment of the \(X_1\) posterior has linear term \(A_1 M_1\bar S\). In fact the likelihood expansion has linear term \(S i^{-g}\overline{M_1}+\bar S i^g M_1\), and averaging it against \(i^{-g}\) leaves \(\bar S M_1\). The resulting real Jacobian has rank two when \(A_1M_1\ne0\), as checked for the C4 table. A differentiable statistic through one real dimension cannot have that rank. For deterministic phase, the carrier log-odds between opposite roots recover \(S\) globally whenever the circulant mapping from \(G\) to \(X_1\) posterior has all its Fourier modes nonzero. With empty valid design the pilot statistic is constant.

For S17 at the pinned initial phase, the circulant mapping is invertible in the underlying Gaussian model. In the first-window turn-zero region, \(\Re T>|\Im T|\) almost surely. On any code cell of positive probability within that region, the density for \(U=0\) exceeds the density for \(U=2\) pointwise because \(\Re T>0\). Also, the sum of densities for \(U=0,2\) exceeds the sum for \(U=1,3\) by the strict increase of \(\cosh\) in \(|\Re T|\) versus \(|\Im T|\). Integrate over the code cell to obtain the nonzero first and second harmonics of \(a_{z_1}\); the third is the conjugate of the first. The other turns just permute the same reasoning. Thus the normalized carrier posterior is itself a minimal Q2 statistic. The argument identifies that statistic even in the Brownian case, without using a global injectivity assumption for \(S\mapsto (L_g/\sum L_g)_g\). The first phase moment is nonzero for at least one \(z_2\) since its unconditional value has magnitude \(e^{-\sigma_\phi^2p/2}\).

### Exact Q2 posterior

Let \(F_z(\alpha)=P(E(Y)=z\mid \text{apparent phase }\alpha)\) for the relevant window SNR. Set

\[
a_{z_1}(u)=\frac{F_{z_1}(\varphi_0+bu)}{\sum_vF_{z_1}(\varphi_0+bv)},\qquad
m_z(\theta)=\tfrac14\sum_vF_z(\theta+bv).
\]

The second code updates the phase law via \(m_{z_2}\) and leaves prior \(G\) uniform. For \(p>0\) and \(d=\min(h,p)\), the bridge is

\[
\Theta_h\mid\Theta_p=s
\sim\mathcal N\!\left(\varphi_0+\omega h+
(d/p)(s-\varphi_0-\omega p),\ \sigma_\phi^2(h-d^2/p)\right).
\]

At \(p=0\) or zero diffusion use the deterministic pilot phase. At \(p=h\) the conditional second-window phase is a point mass. Define \(d\eta_{z_2}(s)\propto dP(\Theta_p=s)\,\mathbb E[m_{z_2}(\Theta_h)\mid\Theta_p=s]\). Then

\[
L_g=\int e^{2\Re(S e^{-i(bg+s)})}\,d\eta_{z_2}(s),\qquad
P(X_1=k\mid z_1,z_2,S)=
\frac{\sum_g a_{z_1}(k+g)L_g}{\sum_g L_g}.
\]

Equivalently the phase-marginalized kernel has the absolutely convergent series
\[
L_g=\sum_{j\in\mathbb Z} I_{|j|}(2|S|)e^{ij(\arg S-bg)}
\int e^{-ijs}\,d\eta_{z_2}(s),
\]
where \(I_j\) is the modified Bessel function. Without \(z_2\), the phase moment in this series equals \(e^{-ij(\varphi_0+\omega p)-j^2\sigma_\phi^2p/2}\). No valid pilots makes the Q2 posterior uniform. The C4 reader evaluates the formula with calibrated code emissions and Gauss–Hermite nodes, so its saved probabilities are exact for that discrete quadrature model.

## 3. Theorem 2 | Distinct clocks require a Wiener bridge

The audited generator is Brownian motion with drift. An OU model would use its own Gaussian mean, covariance and bridges. At sorted distinct pilot clocks \(t_j\), define the precision-weighted family

\[
S_j=\sum_{m:t_{p,m}=t_j}\frac{\alpha_m}{\sigma_m^2}T_m,\qquad
D_j=e^{-i(\varphi_0+\omega t_j)}S_j,\qquad
P_j=\sum_{m:t_{p,m}=t_j}\frac{\alpha_m^2}{\sigma_m^2}.
\]

The family \((t_j,D_j)_j\) is sufficient given design. The extra \(P_j\) gives the complex precision at that clock and is useful for estimating the observation model or defining a coarsened likelihood. At known parameters it already influences the full posterior through the weights in \(D_j\).

For positive distinct clocks put \(C_{jk}=\sigma_\phi^2\min(t_j,t_k)\), \(d_j=t_j-t_{j-1}\), \(t_0=v_0=0\). Independent innovations give the full-rank density

\[
\mathcal N(v;0,C)\propto
\exp\!\left[-\frac{1}{2\sigma_\phi^2}\sum_j\frac{(v_j-v_{j-1})^2}{d_j}\right],
\qquad \det C=\sigma_\phi^{2K}\prod_j d_j .
\]

The precision is tridiagonal. Its diagonal is \((d_j^{-1}+\mathbf1_{j<K}d_{j+1}^{-1})/\sigma_\phi^2\) and its adjacent off-diagonal is \(-1/(\sigma_\phi^2d_{j+1})\). A pilot at zero uses a deterministic node. Set \(c_h=(\sigma_\phi^2\min(h,t_j))_j\). The conditional \(B_h\mid v\) has mean \(c_h^\top C^{-1}v\) and variance \(\sigma_\phi^2h-c_h^\top C^{-1}c_h\). Between two adjacent clocks this is linear interpolation with variance \(\sigma_\phi^2(h-t_j)(t_{j+1}-h)/(t_{j+1}-t_j)\).

The exact conditional likelihood ratio for carrier labels \(g,g'\) is \(L_g/L_{g'}\), where

\[
L_g=\int_{\mathbb R^K}\mathcal N(v;0,C)\,
\mathbb E[m_{z_2}(\varphi_0+\omega h+B_h)\mid v]\,
\exp\{2\sum_j\Re[D_j e^{-i(bg+v_j)}]\}\,dv .
\]

Use the Q2 convolution above. Gauss–Hermite quadrature in independent innovations evaluates this integral for short histories. Brownian independent increments also permit forward/backward integration with wrapped Gaussian transitions on the circle. A Fourier calculation integrates each multidimensional harmonic with multiplier \(e^{-n^\top C n/2}\), augmenting the path with \(h\) if needed. Finite grids, finite Fourier expansions, quadrature, and estimation of encoder emissions introduce distinct errors.

**Minimality for the physical likelihood.** A parameter-independent density ratio implies \(\sum_j\Re[(D_j-D'_j)e^{-i(bg+v_j)}]\) is constant. The phase vector has full support, so vary each \(v_j\) independently to force \(D_j=D'_j\). A deterministic zero-time node is resolved by comparing the four \(g\) values. At zero diffusion all residual phases coincide and the derotated sum is sufficient.

**Strict insufficiency of the derotated sum for Q2.** Fix two distinct clocks with all other pilot sums zero, and consider \((D_1,D_2)=(\epsilon d,-\epsilon d)\) against the origin. The first-order change in the first \(X_1\) Fourier moment is
\[
\epsilon A_1\bar d \big[\mathbb E(e^{iB_{t_1}}\mid z_2)-\mathbb E(e^{iB_{t_2}}\mid z_2)\big].
\]
For at least one \(z_2\) of positive probability the bracket is nonzero, since averaging it over \(z_2\) yields \(e^{-\sigma_\phi^2t_1/2}-e^{-\sigma_\phi^2t_2/2}\ne0\). An informative \(z_1\) has \(A_1\ne0\). Continuity and positive Gaussian observation density extend the difference to positive conditional measure along equal-sum fibers for an open set of totals. Thus even the optimal posterior based on the sum has strictly worse expected log loss. The same proof works at \(\omega=0\). A joint target containing \(X_1\) inherits the loss through its \(X_1\) marginal.

### Error of a common-phase approximation

The coherent approximation freezes residual pilot phases at the earliest clock and retains the correct \(Z_2\) model given that clock. For two clocks, conditional on \(v_1\), the increment has variance \(\xi=\sigma_\phi^2(t_2-t_1)\). For \(q=D_2e^{-i(bg+v_1)}\) and \(f(\delta)=\exp(2\Re(qe^{-i\delta}))\), suppressing the \(Z_2\) factor gives
\[
\mathbb E f(\delta)/f(0)=1+\tfrac{\xi}{2}\big[-2\Re q+4(\Im q)^2\big]+O(\xi^2).
\]

The first correction already depends on the late pilot separately. For the joint \(Z_2\) likelihood compare the Gaussian covariances of \((B_h,B_{t_1},\ldots,B_{t_K})\) before and after freezing. Each entry changes by at most \(\sigma_\phi^2(t_K-t_1)\). Differentiating Gaussian expectations along the convex covariance interpolation gives half the covariance change contracted with the expected Hessian of the integrand, by Gaussian integration by parts. A positive diagonal regularization and continuity handle singular covariances. At bounded pilot sums, the integrand and its derivatives through second order are bounded. Consequently the integrated likelihood changes by \(O(\sigma_\phi^2\Delta t)\) at fixed \(K\). Log-likelihood ratios and normalized posteriors obey the same local order on compact sets with denominators bounded away from zero. This is an error bound for coherent collapse on bounded evidence. The expected NLL gap depends on signal strength, discrimination, and model mismatch.

There is also an optimal **coarsened-data** Bayes comparator. Conditional on \(v,g\), \(D_{\rm sum}\) is complex Gaussian with mean \(i^g\sum_jP_je^{iv_j}\) and variance \(\sum_jP_j\). Phase marginalization retains the squared modulus of that conditional mean in the exponent. A coherent single-clock model adds a likelihood approximation on top of discarding the individual clock sums.

### Q3_after_pilot posterior

Add the withheld clock \(\tau=h+\ell\). Let \(H_q(u,s)\) be the probability that \(\alpha_3 i^u e^{is}+\epsilon_3\) falls in apparent sector \(q\), given \(u,s\), using the withheld-window noise law. For Gaussian noise it is a two-dimensional sector integral. With \(\mathcal W_g(v)=\exp\{2\sum_j\Re[D_je^{-i(bg+v_j)}]\}\), the unnormalized joint posterior is

\[
J(k,q)=\sum_g a_{z_1}(k+g)\,
\mathbb E\!\left[m_{z_2}(\Theta_h)\mathcal W_g(B_{t_1},\ldots,B_{t_K})
H_q(k+g,\Theta_\tau)\right].
\]

Normalize over \(k,q\). Summing over \(q\) gives Q2 and summing over \(k\) gives Q3_after_pilot. The second-window emission and third-window sector emission generally require the same joint bridge. The carrier-only vector \(L_g\) need not retain the pilot update of \(B_\tau\). Coincident pilot and prediction clocks share the latent phase, with independent observation noises. In a causal receiver, only pilots available by the prediction deadline enter the calculation.

The normalized entries of \(J\) give an exact minimal Bayesian statistic for this joint target at fixed data codes and design. For the pair of transmitted symbols in the repeat protocol, the posterior is supported on \(X_3=X_1\) and the four Q2 probabilities suffice.

## 4. Stage 3D reference-fusion diagnosis

The replay uses all 512 mothers and three saved host seeds with locked temperatures. Accuracy is in percent and NLL in nats.

| Arm | Work A Acc / NLL | Qwen phase Acc / NLL |
|---|---:|---:|
| none | 25.00 / 1.3863 | 24.80 / 1.3928 |
| valid | 66.41 / 0.8192 | 28.78 / 1.3889 |
| duplicate copy, dedup | 66.41 / 0.8192 | 28.78 / 1.3889 |
| duplicate copy, raw | 66.41 / 0.8192 | 26.30 / 1.3944 |
| independent only | 25.00 / 1.3863 | 24.80 / 1.4077 |
| valid + independent | 66.41 / 0.8192 | 24.54 / 1.4082 |
| fresh resample | 74.22 / 0.6632 | 26.43 / 1.4056 |

Relative to valid, raw copy costs 2.47 pp, the distractor costs 4.23 pp, and fresh costs 2.34 pp. Appending the distractor changes the phase host posterior by 0.04030 mean total variation and adds 0.01935 nats. A fresh pilot changes it by 0.03727 mean TV and adds 0.01676 nats. The zero host drops from 26.63% to 25.78% on fresh. Work A gains 40 correct decisions net and reduces NLL by 0.15602 on fresh. The verification reconstructs the raw duplicate arm with the physical reader, since the archive has six physical arms for seven host arms. [Results](results_m3.json) · [Archived scoring rules](../../stage3d_qwen_signed_phase/recompute_all.py)

**Position and recurrence.** The prompt renders the ID, timestamp and Boolean as text. Each `<REF>` expands to four learned vectors. The smoke audit records 253 prefix tokens for valid Q2. A second reference adds four embedding vectors and changes the text, moving downstream positions and changing relative distances for attention pairs across the insertion. Distances between two jointly shifted downstream positions stay fixed. DeltaNet consumes additional input and state transitions. The saved blind data lack per-arm prefix traces and hidden states. The training arms already include two fresh pilots, so an unseen length alone is an unsupported explanation. A causal test needs fixed prose, fixed-position payload swaps, and a position-only attention intervention. A separate neutral insertion with a recurrent-state control can test state evolution. [Prompt](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Prefix assembly](../../stage3d_qwen_signed_phase/qwen_connector.py) · [Smoke audit](../../stage3d_qwen_signed_phase/results/smoke_audit.json) · [RoPE](https://arxiv.org/abs/2104.09864) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

**Coordinates and filtering.** `build_ref_feature` includes \((\Re T,\Im T,|T|,\Re y_0,\Im y_0,\ldots,\Re y_2,\Im y_2,c,t_p,\mathrm{SNR}_{dB}/10)\). Four global I/Q pairs rotate together. Magnitude and metadata are invariant. `ref_mlp` and window embeddings lack quarter-turn tying, the explicit multiplication by \(i^{-\mathrm{turn}(z_1)}e^{-i\omega t_p}\), and numeric pooling. The learned mapping therefore relies on cross-token interactions for these operations, with no guarantee that it recovers them. The saved scores do not establish an expressivity impossibility. Deduplication is prompt/feature-list filtering when enabled. The raw duplicate bypasses it. Independent-channel embeddings reach the host along with their status tags, so the flag alone supplies no zero-influence guarantee. [Reference features](../../stage3_ministral_cache/ministral_connector.py) · [Projection](../../stage3d_qwen_signed_phase/qwen_connector.py)

The archived first valid matched sums have RMS magnitude 1.27735. Their global I/Q pair moves by RMS 1.80644 under a joint quarter-turn because \(|iT-T|^2=2|T|^2\). The canonicalized pool had zero movement for that action. The normalized absolute cross moments of the raw matched sums are 0.62189 for the two valid pilots and 0.00420 for valid and independent, consistent with the generator's common signal and separately drawn distractor. [Result receipt](results_m3.json)

## 5. Fixed-slot interface and guarantees

1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
2. **Keep exact evidence upstream.** Use sign swaps for the integer rotation \(i^{-a_1}\) on I/Q before floating-point reduction. Apply continuous derotation once per clock. Pack `Pooled(iq[B,64,2], times, precision[64], gap[B], family1[B], family2[B])` with zero padding excluded from the reader. The prototype batches one shared public design. A production wrapper can batch by shared design or use a padded per-row mask. Declare \(\omega,\sigma_\phi,h,\tau\) and the noise model as public inputs. The continuous path is differentiable for fixed metadata. ID groupings, channel flags, and clock equality are discrete.
3. **Expose fixed task evidence to M1.** Compute `canonical_q2[4]`, `canonical_q3[4]`, and optionally `joint[4,4]` from the reader. Provide a separate frame \(a_1\) for external sector relabeling. Use one always-present task evidence marker and fixed projection, with identical prose and no raw IDs, counts, flags, or clock lists. A multitask prompt can reserve both markers. The CPU prototype returns a single four-component Q2 slot. A projection to Qwen dimensions and an intervention on the frozen host remain separate integration work.
4. **Use the model posterior for guaranteed decisions.** Arbitrary host scores or trainable residual logits can change posterior calibration and rankings even at fixed length. Keep the physical posterior available as the decision output and the frozen-host readout as an experimental comparator.

**Rotation.** Under the joint coordinate action, \(a_i\mapsto a_i+r,\ G\mapsto G+r\), and \(X_1,u_1,u_2\) are invariant. Thus \(i^{-a_1}T_m\) and \(a_2-a_1\) are unchanged. Use \(G-a_1,U-a_1\) in the canonical kernel. Q2 is invariant and Q3 global sector probabilities permute by \(r\) through an explicit external index rotation. The exact-zero CPU verification covers Q2 and pooled evidence in all four frames. Qwen outputs were replayed from the archive.

**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)

For Q2 with these same nondegeneracy conditions, strict population MAP improvement also follows. At zero old evidence, choose a new pilot value giving a nonuniform posterior. Quarter-turning **only that new pilot** produces the four label rotations of that posterior, because the old likelihood is constant in \(G\) there. The intersection of the four sets of maximizers is empty. The strict gaps excluding each label from at least one such maximizer set persist in neighborhoods of the new values and of zero old evidence. The conditional new pilot has positive density in every neighborhood. Therefore the expected new maximum posterior probability strictly exceeds the maximum of the conditional mean on a positive-measure set of old observations. This strengthens the sample-level claim to a population statement for the specified model. Accuracy can still go down on individual mothers or a finite test set, and the result supplies no guarantee for a learned host. For Q3 alone, strict log-loss improvement requires its conditional sector distribution to change.

**Fisher information.** A discrete quadrant or symbol has no ordinary differentiable-parameter Fisher information. For continuous deterministic pilot phase \(\theta\), the score \(2\Re[(T-\mu)^*\partial_\theta\mu]/\sigma_m^2\) has variance \(2\alpha_m^2/\sigma_m^2\). The conditional phase information at clock \(j\) is exactly \(2P_j\), with strict increments for positive precision. For a continuous global phase parameter in a marginalized Brownian model the old score is the conditional expectation of the augmented score. The gain in Fisher information is the expected conditional variance, which is nonnegative and strict exactly when that variance is positive.

## 6. CPU replay and paired experiments

Run from the repository root.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M3/verify_m3.py
~~~

The script replays the archived physical and host probabilities. A fixed-slot CPU surrogate fits \(a_{z_1}\) on 32,768 distinct `NS_CFO_TRAIN` C4 mothers with quarter-turn tying and a symmetric half-count prior. Its second-window phase emission comes from separate Stage 3C-style importance calibration. The 512 blind labels enter only scoring. The multi-clock experiment uses 16,384 new paired mothers at \(t_p=(1,7)\), \(h=2\), \(\omega=+0.35\) and 2 dB, with common draws across methods and diffusion settings. It compares the full Brownian model, the optimal coarsened-data posterior under the same model, and a coherent common-phase approximation. The first low-diffusion observed-label result was inconclusive. A single independent follow-up at 524,288 mothers resolved that contrast, with the initial result retained. Conditional model NLL in the receipt averages over the full fitted posterior for each observation and complements the independently observed-label NLL. The main experiment uses order 40 per pilot innovation, checks order 64 on the 128 highest-pilot-magnitude cases, and takes the second-window emission from 513 periodic knots with order 20 for the inner bridge. The follow-up uses order 12 per innovation and checks against order 40 on 128 high-magnitude cases. [Script](verify_m3.py) · [Result receipt](results_m3.json) · [Benchmarks and Round 2 questions](reply_r1.md)

## 7. Theorem 2 extension | Conditioning on causal history

Let \(z\) denote the existing cache and other authorized context, \(\Theta=(G,\boldsymbol\phi,\omega,\ldots)\) the latent state, and \(P\) the **retained Gaussian noise boundary** of \(\mathcal F^0\). Specifically, use the conditional model

\[
\begin{bmatrix}Y_P\\Y_N\end{bmatrix}
\Bigm|\Theta,z
\sim\mathcal{CN}\!\left(
\begin{bmatrix}\mu_P(\Theta,z)\\\mu_N(\Theta,z)\end{bmatrix},
\begin{bmatrix}K_{PP}&K_{PN}\\K_{NP}&K_{NN}\end{bmatrix}\right).
\]

Here \(N\) contains only distinct valid IDs that did not contribute to \(\mathcal F^0\). Conditioning on \(Y_P,z,\Theta\) must screen off the remaining old observations from \(Y_N\), or the supplied Gaussian boundary model must already incorporate them. Any calibration field obtained from old observations is scoped to that conditioning model. A sample mean can include a known QPSK pilot root \(d_m\), a gain, and a residual offset, for example \(\mu_m=c_m+\alpha_m d_m i^G e^{i\phi(t_m)}\). An ISI model can use a row with multiple coefficients if it mixes more than one phasor. Circular proper complex noise has \(K=\mathbb E[\epsilon\epsilon^\dagger\mid\Theta,z]\). For noncircular Gaussian noise, apply the same block formulas to the stacked real/imaginary vector with its full real covariance. Gaussianity and the choice of boundary are model assumptions to verify on calibration data. [Gaussian conditioning reference](https://gaussianprocess.org/gpml/chapters/)

### Schur-complement factor and sufficient statistic

For positive definite \(K\), put \(A=K_{NP}K_{PP}^{-1}\) and \(S=K_{NN}-AK_{PN}\). The conditional innovation factor is

\[
p(y_N\mid\mathcal F^0,\Theta)=
\frac{\exp\{-(y_N-\mu_N-A(y_P-\mu_P))^\dagger S^{-1}
                (y_N-\mu_N-A(y_P-\mu_P))\}}
     {\pi^{|N|}\det S}.
\tag{7.1}
\]

Proof. The centered vector \(\epsilon_N-A\epsilon_P\) has cross covariance \(K_{NP}-AK_{PP}=0\) with \(\epsilon_P\). Proper joint Gaussianity implies independence. Its covariance is \(S\). This also proves factorization of the full block density as \(p(y_P\mid\Theta,z)\,p(y_N\mid\mathcal F^0,\Theta)\). Compute with Cholesky and solves. If \(K_{NP}=0\), the old noise samples do not enter (7.1). Correlation among new samples alone still requires \(K_{NN}\) in (7.1). A phase posterior, gain estimate and ledger without the correlated noise boundary are generally insufficient. For a scalar counterexample take two possible \(y_P\) with the same compressed prior, a model in which \(Y_P\) is independent of \(\Theta\), and \(A\ne0\). The two conditional means differ by \(A\) times the difference in \(y_P\).

For fixed \(A,S\) and affine means \(\mu=c+M u(\Theta)\), write

\[
r=y_N-c_N-A(y_P-c_P),\quad W=M_N-AM_P,\quad
q=W^\dagger S^{-1}r,\quad J=W^\dagger S^{-1}W.
\]

The log factor is
\[
2\Re(u^\dagger q)-u^\dagger J u
- r^\dagger S^{-1}r-\log\det S-|N|\log\pi .
\tag{7.2}
\]
Thus \(q\) is sufficient for the latent likelihood ratios given design, with \(J\) supplied by the design. The minimal likelihood-ratio statistic is the real projection of \(S^{-1/2}r\) onto the real span of the differences \(S^{-1/2}W[u(\Theta)-u(\Theta_*)]\). Indeed the density ratio for two datasets depends on \(\Theta\) exactly through the real inner products of their whitened difference with those mean differences. When those differences span the modeled phasor coordinates over the reals and \(W\) is full column rank, \(q\) is minimal up to an invertible change of coordinates. With rank deficiency one retains only this span. The same-clock independent case reduces to Round 1. With correlated clocks or samples the Gram penalty may include cross-clock terms of the phase state.

Preserve \(r^\dagger S^{-1}r,\log\det S\) and the observation-space measure for normalized predictive densities. Terms common to nodes cancel in a posterior for one design, but are required for marginal log scoring. When noise, affine offsets, or association depend on a hypothesis, use the appropriate (7.1) with each law before normalizing, including its quadratic and log determinant. A single per-clock precision-weighted sum is generally insufficient across those laws. Side information from orthogonal residuals or different observation support must also be modeled if its conditional law differs between hypotheses. Known pilot derotation is an invertible diagonal transform of the observations **and** the means and covariance. Version 2 puts \(d_m\) in the means and leaves \(K\) in the original matched coordinates.

### Anti-double-counting theorem

Let \(L_0\) be the provenance ledger of all IDs consumed when constructing \(\pi_0(d\Theta)=P(d\Theta\mid\mathcal F^0)\). Deduplicate by ID and metadata before inference. On later reveal, use only \(\mathrm{IDs}(\mathcal R)\setminus(L_0\cup L_{\rm added})\) as new evidence. Already observed samples can enter the **conditioning boundary** of (7.1) but add no new likelihood term. Formally, the conditional law of a literal copy given \(\mathcal F^0\) is a point mass at its recorded value, for every \(\Theta\). The conditional likelihood ratio for that copy is one at the observed value. Sequential conditionals and a joint block update are equivalent by the chain rule:

\[
p(y_A,y_B\mid\mathcal F^0,\Theta)
=p(y_A\mid\mathcal F^0,\Theta)p(y_B\mid\mathcal F^0,y_A,\Theta).
\]

A registry ledger is a causal requirement. Substituting a new ID for an old observation defeats its semantics. M4 must authenticate observation identity before inference and distinguish copying from correlated new acquisitions. Supplying only lineage IDs can safely suppress copies, but a correlated boundary value or its equivalent conditional forecast is still needed. The implementation refuses a forecast that needs a retired boundary payload. It accepts retirement when the corresponding regression coefficient is exactly zero.

**Effect of recounting.** In a linear Gaussian model with initial precision \(\Lambda_0\), recording \(y=L\theta+\epsilon\) at covariance \(R\) contributes \(Q=L^\dagger R^{-1}L\succeq0\) once. Treating the same sample as an independent second draw gives posterior covariances \(P=(\Lambda_0+Q)^{-1}\) and \(P'=(\Lambda_0+2Q)^{-1}\). Write the difference as \(P^{1/2}[I-(I+P^{1/2}QP^{1/2})^{-1}]P^{1/2}\). This is positive semidefinite, with a strict variance reduction along \(v\) precisely when \(Q^{1/2}Pv\ne0\). Recounting can change the mean as well. Under the true law, the excess conditional posterior log loss of any wrong reported density is the forward KL from the correct posterior. It is strictly positive in expectation when the posteriors differ on a positive-probability event. For the proper complex scalar witness with prior variance \(1.25\) and noise variance \(0.4\), the correct and repeated variances are \(0.3030303\) and \(0.1724138\), and the expected forward KL is \(0.2981335\) nats. The KL uses \(\log(P'/P)+P/P'-1+|\mu-\mu'|^2/P'\), averaged over the one true draw. In nonlinear or categorical inference, harmful repeat counting can increase a scalar posterior variance. Prior odds \(9:1\) and likelihood ratio \(1:3\) give posterior odds \(3:1\) after one recording and \(1:1\) after recounting. The KL regret statement survives that case.

A real fresh acquisition Blackwell-dominates the smaller experiment by the projection that discards it. Strict expected log-loss improvement for target \(A_{\rm task}\) requires positive \(I(A_{\rm task};Y_{\rm new}\mid\mathcal F)\), which is an assertion about its conditional innovation. A distinct ID alone cannot certify that information gain.

## 8. Shared CFO and Brownian bridge from the causal anchor

Take a causal anchor \(t_0\) and physical lags \(s_j=(t_j-t_0)\ge0\) in the declared clock units. Conditional on the causal anchor distribution, write
\[
\phi(t_j)=\hat\phi_0+\hat\omega s_j+\epsilon_0+\epsilon_\omega s_j+\sigma_\phi W_{s_j},
\qquad
(\epsilon_0,\epsilon_\omega)\sim\mathcal N\!\left(0,
\begin{bmatrix}P_{\phi,0}&c_{\phi,\omega}\\c_{\phi,\omega}&v_\omega\end{bmatrix}\right).
\]
The Brownian innovations after the anchor are independent of its state. The pre-reveal covariance conditional on this Gaussian anchor model is
\[
C_{jk}=\sigma_\phi^2\min(s_j,s_k)+v_\omega s_js_k+
P_{\phi,0}+c_{\phi,\omega}(s_j+s_k).
\tag{8.1}
\]
The anchored phase variance and the anchor–CFO cross term cancel in \(\operatorname{Var}(\phi(t_j)-\phi(t_i))=\sigma_\phi^2|s_j-s_i|+v_\omega(s_j-s_i)^2\). Distinct edges share \(v_\omega\Delta s_1\Delta s_2\) even when Brownian increments on those edges are disjoint. Brownian overlap contributes \(\sigma_\phi^2\) times oriented overlap length. Thus replacing all transition variances with their marginal values and multiplying independent transition kernels loses shared CFO. For positive distinct clocks and \(\sigma_\phi>0\), with the initial phase fixed, the inverse is
\[
C^{-1}=C_B^{-1}-
\frac{C_B^{-1}s\,s^\top C_B^{-1}}{v_\omega^{-1}+s^\top C_B^{-1}s}
\quad(v_\omega>0).
\]
The zero-CFO-variance limit is \(C_B^{-1}\). The general Gaussian anchor uses \(C=C_B+[{\bf 1},s]P[{\bf 1},s]^\top\). One can use a two-column update or sample the joint anchor/rate, sharing **one rate quadrature dimension** across all times and propagating the Brownian path conditionally. Coincident clocks share a node and deterministic coordinates can be sliced out. The code uses calibrated integer ticks and subtracts the epoch before conversion to limit loss of short-lag precision.

Marginalize the common path against (7.1). In the joint Gaussian phase-prior example, let \(c=(\operatorname{Cov}(\phi(t_3),\phi(t_j)))_j\) be formed from (8.1). Conditional on the pilot phase vector \(v\), the target phase has mean \(m_3+c^\top C^{-1}(v-m)\) and variance \(C_{33}-c^\top C^{-1}c\). This is valid for a target earlier than some pilots and for a target later than them. Coincident target and pilot clocks use the same phase. If the posterior over \((\phi_0,\omega)\) is wrapped, multimodal, or hypothesis-dependent, retain a weighted node bank. When an old clock predates the causal anchor, the adapter must supply its joint conditional path with the future clocks or the corresponding conditional noise-boundary forecast. A pilot with \(t_p>t_3\) can inform a retrospective posterior with a jointly modeled target phase. A live Q3 prediction locked before its availability excludes it. The release deadline uses the latest raw sample in an observation's stencil, which can follow its nominal symbol time.

Keep thermal/residual receiver covariance conditional on the latent trajectory distinct from uncertainty already carried by the trajectory nodes. Adding the shared CFO residual again inside \(K_{\rm ref}\) counts the same uncertainty twice. A separately validated observation-level Gaussian approximation can marginalize such uncertainty into \(K\), in which case remove the corresponding latent variation. Neighboring CTTC pilot indices alone give no numerical \(K\). Estimate conditional \(K\) on allowed calibration data, alongside any residual ISI or spur mean model, and compare predictive coverage on held-out data. [M4 timing and data contract](../M4/THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)

## 9. Frozen RefSufficientPool v2 contract

| Boundary | Fields and semantics |
|---|---|
| Causal snapshot | M4 supplies the already-conditioned node law and a complete history ledger for all observations used in \(\mathcal F^0\), including baseline sync, old pilot, and relevant cache lineage. `PhaseContext` carries `anchor_tick`, `ticks_per_unit`, `phi0_hat`, `omega_hat`, `v_omega`, `diffusion_var_per_unit`, `p_phi0` and `cov_phi0_omega`. Carry gain and offsets from \((H_0,d_0,\ldots)\) through calibrated node means. Use the measured noise boundary or a fully conditioned model that can supply (7.1). |
| Acquisition and calibration | `ClockMeta` has salted `acquisition_id`, provenance handle, physical and maximum-availability ticks, authorized raw support, role and calibration ID, and `known_root` in \(C_4\) where known. A probe with latent symbol can use `None` and include that symbol in the node law. A reveal adds the complex `matched` value in its calibrated units and the authorized `shares_channel` state. Raw IQ and orthogonal energy enter the wider observation model if needed for sufficiency. The implemented likelihood scores one complex matched observation per ID. |
| Node law | `NodeBank(phase_grid[H,N,C], log_weights_by_turn[H,4,N], physical_ticks, hypothesis_ids)`. The posterior log-weights normalize **globally** over hypotheses, turns, and nodes. \(G\) is in the canonical receiver frame. The Gaussian adapter provides `means[H,4,N,D]` and proper complex `K[H,D,D]` indexed by catalog ID. \(D\) covers the modeled noise boundary and candidates, and \(C\) includes target clocks if needed. More general state-indexed covariance or non-Gaussian laws require a corresponding normalized likelihood adapter. |
| Numerical update | `start(history, as_of)` imports the conditioned state without reading its old likelihood again. `innovation(state, ids, deadline)` returns the sorted, normalized conditional law for fresh IDs. `update(state, records, as_of)` validates, masks, then merges uniquely. The implementation recomputes the accepted block relative to the immutable prior in sorted-ID order. Reordering or partitioning the same acquisitions yields identical evaluated tensors in this deterministic implementation. The difference in `log_increment` values is the incremental log evidence. |
| M1 task interface | `task_readout` evaluates `joint[4,4]` from normalized node-wise joint target probabilities, `canonical_q2[4]` by row marginal, and `canonical_q3[4]` by column marginal. If withheld noise shares the observed noise boundary, the target table must use its corresponding conditional target distribution. Keep the receiver frame \(a_1\) external and permute only the Q3 RX sectors under a common receive-gauge shift. A TX relabel is a separate action. M1 maps either S17 or its 16-bit canonical packet to its own window features, and appends the common fixed reference slots. |
| Fixed wire payload | `fixed_slots` returns a `uint8[4]` header `[version=2,inner1,inner2,relative_turn]` and little-endian `float32[2,4]` probabilities in always-present Q2/Q3 slots. Keep the full-precision posterior and optional joint on the sidecar. Prompt text, marker positions, attention mask and text token IDs stay fixed for a task template at all set sizes. Pass neither provenance nor reference count to the host. Use deterministic projections in a pinned execution mode. Byte equality is for equal semantic inputs on a fixed implementation, independent of the numerical accuracy of candidate integration. |

Pluto uses Copy 1 pilot ordinals `15,18` with sync `0:13` in \(\mathcal F^0\). CTTC uses `126,127` with sync `0:6` in \(\mathcal F^0\). These are source ordinals, while conditioning uses calibrated physical clocks and availability. The ledger records every consumed ID even when its sample is retired. Verify metadata and an exact canonical payload digest on old repeats. Conflicting duplicate data, roots, calibration or provenance fail validation. An unlisted fresh valid ID is an error, and a known history ID without its ledger entry is an error. For duplicate digest use canonical binary64 I/Q with signed zeros folded. The independent mask is appropriate only for a known distractor conditionally irrelevant to both the target and the valid-pilot noise after context. For unknown association use a latent association likelihood. Typed transport padding is represented by `None` and discarded before aggregation. The CPU implementation supports 64 distinct accepted IDs, with the host slot count independent of that limit. Degree of shared noise and timestamp validity in the actual captures remain M4 measurements. [Executable schema](ref_pool.py)

## 10. Stage 6 pre-reveal mixture and post-reveal update

Let \(w_{hgn}=P(H=h,G=g,n\mid z,\mathcal F)\) be the current node weights, including previous authorized acquisitions if any. For a candidate \(a\), call the same innovation law with only registered metadata, conditioning on the already visible boundary. For each active hypothesis with \(\pi_h=\sum_{g,n}w_{hgn}>0\) define

\[
f_{ha}(x)=\frac{\sum_{g,n}w_{hgn}\,p_a(x\mid h,g,n,\mathcal F)}{\pi_h},
\qquad m_a(x)=\sum_h\pi_h f_{ha}(x).
\]

These densities normalize on the declared scored observation space. From (7.1) this remains true for a correlated candidate. With action metadata fixed and chosen from visible evidence, the expected information gain and reveal update are

\[
\operatorname{EIG}(a\mid z,\mathcal F)
=\sum_h\pi_h\int f_{ha}(x)\log\frac{f_{ha}(x)}{m_a(x)}\,dx
=\mathbb E_{m_a}D_{\rm KL}(P(H\mid x,a,\mathcal F)\Vert\pi),
\qquad
w'_{hgn}=\frac{w_{hgn}p_a(x\mid h,g,n,\mathcal F)}{m_a(x)}.
\]

Summing \(w'_{hgn}\) over \(g,n\) recovers \(\pi_h f_{ha}(x)/m_a(x)\). Averaging this posterior over \(m_a\) recovers the old \(\pi_h\). An already consumed candidate has zero EIG and is excluded from the action set. The full node mixture, including the log determinant and normalized conditional densities, is part of the pre-reveal commitment. An arbitrary additive log-score offset can be benign for rankings but invalidates the predictive marginal. [Lindley, *On a Measure of the Information Provided by an Experiment*](https://doi.org/10.1214/aoms/1177728069)

Commit the causal state identifier, calibration version, candidate metadata, forecast laws, selected ID and deadline before release. Choose one action, with any policy randomness independent of sealed samples. An external vault enforces one transactional read per query and logs commitment and reveal separately. `forecast` never receives candidate values. `reveal` checks its committed ID, state and deadline and updates through the same conditional Gaussian law. Its immutable snapshots by themselves do not enforce a global read budget on external storage. A physical target earlier than the chosen probe is valid for retrospective Stage 6 inference, with a query deadline that permits that probe. A live forward Q3 evaluation uses its earlier target cutoff. Future, stale, wrong-provenance and consumed candidate IDs are handled before host construction.

### Replay and calibrated comparison

The default `verify_m3.py` command runs the Round 2 checks and replays the archived C4 scores, with the already-recorded slow Round 1 simulations preserved in the JSON. Use `--rerun-round1` to repeat those simulations explicitly. The tests exercise the Schur factor against a full block density, the pinned C4 adapter on real blind data, a two-clock CFO/Brownian simulation with conditional thermal correlation, typed representation and a deterministic prefix adapter, and a sealed-query fixture. The byte checks include text IDs, positions, masks and embedding bytes on the fixture. Actual Qwen embedding execution and measured hardware innovation laws remain integration tests. [Round 2 results](results_m3.json) · [Executive summary](reply_r2.md)

For the matched Q2 contrast, Arm 1 replays the archived native Qwen phase candidate logits and the exact dev-locked temperatures checked against the training receipts. Arm 2 applies a bounded offline residual, `softmax(log q2_reader + 0.15*tanh(centered locked host logits))`. The residual cap comes from M1, with no new fitting on the blind data. Physical `q2_reader` uses its independent S17 calibration. Scores are paired by mother across the same reference arms, averaging seed contrasts before taking block SEs. This is an archived native-logit comparison and an offline residual arithmetic test. A causal claim about M1’s new host-pure and physical-residual architectures additionally requires actual frozen-host forwards with the version 2 slots, dev-locked calibration for each model, and matched-position slot interventions. [M1 ablation design](../discussions/M1/prompt_r2.md) · [M2 calibration gate](../discussions/M2/prompt_r2.md)
