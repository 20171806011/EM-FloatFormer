# M1 Round 1 — Quotient equivariance and signed phase transport

EM-FloatFormer Stage 4 · [CPU verifier](verify_m1.py) · [Numerical evidence](results_m1.json)

Stage 4 makes the quarter-turn output action an explicit permutation and transports evidence by the signed physical phase before sector readout. The permutation law holds independently of the frozen host. The transport accounts for the Stage 3D direct-table recovery near a quarter turn. The Qwen failure is empirical. Separate token positions can interact, and neither a universal below-chance bound for extrapolation nor a fixed sample-complexity penalty follows from the architecture.

## 1. Exact discrete symmetry

Use \(\mathbb Z_4\) labels and roots \(i^j\) in the four-choice order. Q1 asks for \(X_2-X_1\), Q2 asks for \(X_1\), and Q3 asks for the sector of the third matched-filter output. The sectors are \(I_k=[-\pi/4,\pi/4)+k\pi/2\) modulo \(2\pi\). With \(T=\sum_{m=1}^3Y_m/\sqrt3\), window times \(0,h,\tau=h+\ell\), and \(\rho=10^{\mathrm{SNR}_{\mathrm{dB}}/10}\), the Stage 3D generator has

\[
\phi(t)=\omega t+\sigma_\phi B_t,\qquad
T_j=i^{X_j+G}e^{i\phi(t_j)}+\epsilon_j,\qquad
\epsilon_j\sim\mathcal{CN}(0,\rho^{-1}).
\]

Here \(B\) is standard Brownian motion, \(G\) is the uniform QPSK channel turn, and \(X_1,X_2\) and the fresh \(X_3\) are independent and uniform. A shared-channel pilot transmits a known root (root 1 in Stage 3D), so \(T_p=i^Ge^{i\phi(t_p)}+\epsilon_p\). Observation noises are independent within a mother. The balanced blocks sample this marginal law, and the two CFO signs share innovations and noises for paired comparison. The posterior below conditions on one mother's observables. [Schema](../../stage3_ministral_cache/schema.py) · [Generator](../../stage3d_qwen_signed_phase/data_stage3d.py)

### Theorem 1 — Encoder and conditional choice actions

Write \(Z_1=4a+u,\ Z_2=4b+v\), where \(u,v\in\{0,1,2,3\}\). For \(g=(r,s)\in H=C_4\times C_4\), rotating the two windows by \(i^r,i^s\) induces \((a,b,u,v)\mapsto(a+r,b+s,u,v)\). With pilot and metadata fixed, the true conditional choice law obeys \(P(gx)=R_{\chi(g)}P(x)\), where \((R_jp)_k=p_{k-j}\).

| Task | Shift \(\chi(r,s)\) | Target |
|---|---:|---|
| `q1` | \(s-r\) | Relative transmitted turn |
| `q2` | \(r\) | First transmitted turn, fixed pilot frame |
| `q3_repeat` | \(r\) | Received sector of a retransmission of \(X_1\) |
| `q3_indep` | \(0\) | Received sector of a fresh independent symbol, with exactly uniform conditional probabilities |
| `q3_after_pilot` | \(r\) | Received repeat sector given the revealed pilots |

**Proof.** The encoder takes the sector of the sum of the three samples, falling back to the first nonzero sample on a zero sum. The half-open convention gives \(\operatorname{sector}(i^rT)=\operatorname{sector}(T)+r\) for nonzero \(T\), including boundary rays. Undoing the selected turn gives the same canonical I/Q features after rotation. The fitted MLP and center assignment therefore return the same inner index. The all-zero window is fixed and cannot carry a nontrivial turn. It has probability zero under the stated noise. A hardware interface can mark it as an erasure. [Encoder](../../matched_cache/learned.py)

For the joint law, relabel \(X_1\mapsto X_1+r,\ X_2\mapsto X_2+s\), and rotate noises in windows 1, 2, and the repeat window by their designated turns. Keep \(G,\phi\), and the pilots fixed. The uniform source and circular noise laws preserve the joint measure. The three nonuniform label shifts then follow by subtraction, by reading \(X_1\), or by taking the sector. This yields the conditional identities on the support. For the fresh symbol, condition further on \(G\) and the entire phase path. The new symbol and noise are still independent, and mixing the four roots makes its received law quarter-turn invariant. Marginalization preserves the uniform sector law. \(\square\)

To allow a common pilot rotation \(r_p\), relabel \(G\mapsto G+r_p\), \(X_1\mapsto X_1+r-r_p\) and \(X_2\mapsto X_2+s-r_p\). Rotate all shared pilots by \(i^{r_p}\) and the repeat window by \(i^r\). Q2 then shifts by \(r-r_p\), while Q1 and the repeat tasks retain their shifts. In particular, Q2 is invariant under a joint receiver-coordinate rotation \(r=s=r_p\). A common pilot turn respects the shared channel and phase path.

### Theorem 2 — Quotient factorization

Fix an invariant context \(c\), including signed CFO, timestamps, and references in a fixed frame. A four-score map is \(H\)-equivariant if and only if

\[
f(4a+u,4b+v;c)=R_{\chi(a,b)}F(u,v;c).
\]

**Proof.** Every code orbit intersects the canonical slice once. Applying equivariance to that representative proves necessity. The shift rule is additive, so \(R_jR_k=R_{j+k}\) proves sufficiency. A deterministic frozen host can be part of \(F\), because its inputs are identical across the orbit. Normalize in the canonical order and gather by integer indices to obtain bitwise equality for identical canonical computations. With an explicit common pilot frame, canonicalize that evidence and use the extended Q2 shift. For Q3 independent, return exact quarters. \(\square\) [Stage 3C canonicalization and readout](../../stage3c/G/quotient.py)

There are 256 pairs and 16 representatives. Arbitrary probability tables have \(256(4-1)=768\) degrees of freedom per context. The quotient has \(16(4-1)=48\). This is an exact factor of 16 in table degrees of freedom and gives the pooling available in balanced table fitting. Sample complexity for a connector also depends on the source law, hypothesis class, initialization, and objective.

## 2. Continuous phase transport and the sector dip

### Theorem 3 — Circular phase evolution before the sector map

Angle translation by \(\alpha\in S^1\simeq SO(2)\) multiplies \(m_n=\mathbb E[e^{in\varphi}]\) by \(e^{in\alpha}\). For an edge of duration \(d\ge0\), the Gaussian characteristic function gives

\[
U_n(d)=e^{in\omega d}e^{-n^2\sigma_\phi^2d/2},\qquad
m_n(t+d)=U_n(d)m_n(t).
\]

Independent increments yield \(U_n(d_1+d_2)=U_n(d_1)U_n(d_2)\) for constant parameters. Diffusion is an irreversible convolution semigroup for \(\sigma_\phi>0\). To undo a known deterministic drift use the conjugate phasor. Diffusion marginalization continues to use damping.

The quarter-turn actions commute with continuous translations wherever both are realized as rotations of the underlying continuous variables. Conjugation therefore gives the trivial \(S^1\to\operatorname{Aut}(H)\) action. The corresponding semidirect product is a direct product, possibly with a kernel if two factors rotate the same variable. Any continuous action \(S^1\to\operatorname{Aut}(H)\) is trivial because the domain is connected and the codomain is finite. The same reasoning precludes a nontrivial continuous action on the finite codes or choices by permutations. A CFO change is an \(\mathbb R\) flow on the joint edge-phase torus, with edge phases \(\Delta\omega\,d\). It has a common frequency period only if the active durations are commensurate. [Group-action background](https://proceedings.mlr.press/v48/cohenc16.html)

### Theorem 4 — Circulant Q3 pushforward

Let \(\Phi\) be the standard normal CDF, \(\kappa=\sqrt{2\rho}\), and
\(c=\Phi(\kappa\cos(\phi+\pi/4)),\ s=\Phi(\kappa\sin(\phi+\pi/4))\).
For source root 1 the response is

\[
B(\phi)=(cs,(1-c)s,(1-c)(1-s),c(1-s)),\qquad
K(\theta,V)=\mathbb E_{\xi\sim N(0,1)}B(\theta+\sqrt V\,\xi).
\]

With \(A_z(j)\) the adopted posterior on the first apparent transmitted root \(i^{X_1+G}\), the Stage 3D direct S17 table is

\[
q_k(z;\theta,V)=\sum_{j=0}^3 A_z(j)K_{k-j}(\theta,V),
\qquad \theta=\omega(h+\ell),\quad V=\sigma_\phi^2(h+\ell).
\]

**Proof.** Orthogonal projection along the two sector boundaries yields independent normals with the indicated means after standardization. The sign pairs in label order are \(++,-+,--,+-\). Independence gives the four products. Integrate the Brownian angle and use circular observation noise to shift by the source root. The resulting map is a cyclic convolution, diagonalized by the \(C_4\) DFT. Since \(B(\phi+j\pi/2)=R_jB(\phi)\), it gives \(q(z;\theta+j\pi/2,V)=R_jq(z;\theta,V)\) at fixed \(A_z,V,\rho\). \(\square\) [Stage 3D analytical readout](../../stage3_ministral_cache/analytic.py)

The verifier's \(C\) evaluates this integral by 64-point Gauss–Hermite quadrature. Stage 3D evaluates it on a wrapped grid. The formula is exact for the adopted \(A_z\) and marginal target phase distribution, with numerical integration in both implementations. \(A_z\) is a fitted conditional table, and this readout leaves out phase information from \(Z_2\). The archived `cal_ref` is a separate independently simulated two-code reference, pooled by quarter-turn symmetry with half-count smoothing.

For the exact population Bayes rule based on both codes and references, write \(S_1=X_1+G,\ S_2=X_2+G\). Define \(A_u(j)=\Pr(S_1=j\mid Z_1=u)\) in the canonical frame, and \(L_v(j,\psi)=\Pr(Z_2=v\mid S_2=j,\phi_2=\psi)\) from the quantizer regions and observation law. A shared-reference likelihood \(L_P(G,\boldsymbol\phi)\) uses the joint phase path. The canonical joint posterior is proportional to

\[
A_u(S_1)\,L_v(S_2,\phi_2)\,L_P(G,\boldsymbol\phi)\,dP_\omega(\boldsymbol\phi),
\]

using uniform counting measures for \(S_2,G\) and the Brownian prior \(P_\omega\). Q1 marginalizes \(S_2-S_1\); Q2 marginalizes \(S_1-G\). Q3 repeat integrates \(B_k(\phi_\tau+S_1\pi/2)\). For post-pilot repeat include \(L_P\) in the same posterior. A pilot can inform target phase even after averaging over \(G\). With \(L_P=1\), Q2 is uniform because \(G\) remains uniform and independent. Summing \(L_v(S_2,\phi_2)\) over \(S_2\) makes it \(C_4\)-periodic; harmonics with index divisible by four can therefore update phase even though the second symbol is fresh.

### Corollary — Why \(\pi/4\) and \(\pi/2\) give different accuracies

With a known source and \(V=0\), write the angle relative to the nearest sector center as \(\delta\in[0,\pi/4]\) by symmetry. The MAP sector confidence is

\[
a(\delta)=\Phi(\kappa\cos(\pi/4+\delta))\Phi(\kappa\sin(\pi/4+\delta)).
\]

It strictly decreases for \(0<\delta<\pi/4\). Indeed, \(\lambda(x)=\Phi'(x)/\Phi(x)\) decreases on \(x\ge0\) by \(\lambda'(x)=-x\lambda(x)-\lambda(x)^2\). The log derivative of \(a\) is \(\kappa[-\sin(\alpha)\lambda(\kappa\cos\alpha)+\cos(\alpha)\lambda(\kappa\sin\alpha)]<0\) for \(\alpha=\pi/4+\delta\). At the boundary the dominant pair ties at \(\tfrac12\Phi(\kappa)\) per sector. At the center the maximum is \(\Phi(\kappa/\sqrt2)^2\). At 2 dB the two maxima are about \(0.481\) and \(0.803\) before source uncertainty and diffusion.

The Stage 3D evaluation angles \(0.80\) and \(1.60\) are close to \(\pi/4\) and \(\pi/2\), respectively. G3 has twice the phase variance of G1 at the same angle; G4 has half the phase variance of G2 at its angle. The verifier also varies only \(\theta\), holding \(V=0.04\), SNR and the W1 code mixture fixed. It computes the code-mass-weighted maximum of the convolved probabilities. This isolates the angular decision effect from sampling and variance differences across cells. The absolute accuracy under source uncertainty follows from the mixture and the measured data.

The unrotated S17 MAP decision uses \(\arg\max_jA_z(j)\) at every \(\theta\). Its conditional hit rate at \(1.60\) is the corresponding unshifted coordinate of \(q(z;1.60,V)\). A universal below-25% claim is false. A uniform fixed decision is exactly at chance. A fixed code-dependent decision chosen to suit an informative target angle can beat chance there, even as a degree-zero polynomial. Further, the logits \(\log q_k(z;\theta,V)\) are continuous on a compact angle interval for finite SNR. Polynomial logits can approximate them uniformly and yield the same decisions wherever the approximation error is less than half the logit margin. They can also interpolate any finite angle grid exactly. Restricted training gives no assurance that extrapolation finds those coefficients. The chance comparison concerns specific estimators.

At fixed SNR each \(B_k\) is globally \(L\)-Lipschitz with \(L\le\sqrt{2\rho/\pi}\). Differentiate the two CDF factors and bound the sum of absolute sine and cosine by \(\sqrt2\). Coupling the same standard normal draw in both kernels bounds component errors by

\[
L\left(\lvert\widehat\theta-\theta\rvert+\sqrt{2/\pi}\,\bigl\lvert\sqrt{\widehat V}-\sqrt V\bigr\rvert\right).
\]

The row-stochastic kernel also contracts source-posterior \(\ell_1\) error. If the final probability error is at most \(\varepsilon\) per class, an oracle margin larger than \(2\varepsilon\) preserves the decision. These bounds apply past \(\pi/2\) and give no universal accuracy floor for poor SNR, weak codes, or misestimated phase.

## 3. Stage 3D diagnostics and the scope of separation

The release has 2,184,964 trainable connector parameters and frozen Qwen3.5-4B weights. Its pair MLP already couples codes via \(e_1,e_2,e_1-e_2,e_1e_2\). The phase branch maps six slots at \(h,h+\ell,t_p\) to separate tokens. The chat prompt interleaves the embeddings with text, including the numerical condition metadata. The `zero` arm zeros the six slots while retaining public prompt metadata. The signed-CFO train cells have \(\lvert\omega\rvert\le0.25,\ h+\ell\le5,\ \lvert\theta\rvert\le1.25\). The full Stage 3D run also includes zero-CFO replay C1–C3 with its own timings. The CPU surrogate uses only the ten signed-CFO training settings. [Connector and prefix assembly](../../stage3d_qwen_signed_phase/qwen_connector.py) · [Dataset](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Training construction](../../stage3d_qwen_signed_phase/train_eval_stage3d.py)

| Blind cell | Cumulative angle magnitude | Qwen phase acc / NLL | Qwen zero acc / NLL | Direct S17 acc / NLL |
|---|---:|---:|---:|---:|
| G1 | 1.60 | 23.80% / 1.4452 | 19.41% / 1.4809 | 64.62% / 0.9411 |
| G2 | 0.80 | 33.88% / 1.3358 | 36.36% / 1.3081 | 46.78% / 1.0222 |
| G3 | 1.60 | 22.68% / 1.4410 | 19.74% / 1.4781 | 62.11% / 0.9796 |
| G4 | 0.80 | 33.29% / 1.3347 | 34.96% / 1.3098 | 47.75% / 1.0159 |

Work F gives orientation gain \(+0.00982\) and mean swapped-minus-matched NLL \(+0.01837\) across phase seeds. The phase seed-averaged table has mean-square metadata effect \(1.85378\cdot10^{-4}\), waveform effect \(1.11977\cdot10^{-3}\), and interaction \(6.36253\cdot10^{-5}\). The physical calibration has metadata effect \(0.201603\), over a thousand times the host's mean-square value. Work G has rot01 and rot11 mean maximum coordinate differences of \(0.05914\) and \(0.12627\), with worst tested discrepancy \(0.48167\) at rot01. These measurements establish weak response in the correct signed direction and broken quotient equivariance for the trained host. [Stage 3D summary](../../stage3d_qwen_signed_phase/results/summary_master.json)

### Theorem 5 — Additive centered-logit obstruction

Suppose centered logits satisfy \(L(z,\phi)=U(z)+V(\phi)\). The signed phase contrast across codes has rank at most one and the mixed finite difference in \((z,\phi)\) vanishes. If the same readout is exactly \(H\)-equivariant for a task with a surjective output shift, the contrast is zero.

**Proof.** Every row of \(L(z,\phi_+)-L(z,\phi_-)\) is the same vector \(V(\phi_+)-V(\phi_-)\). The mixed difference subtracts two identical rows. Equivariance under all four rolls then makes the common row invariant, hence constant across classes. Centering removes that constant. Multiplicative transport has code-dependent signed contrasts when the states carry differing moments affected by phase. \(\square\)

Separate token emission does not satisfy the theorem's final-logit hypothesis. A frozen attention head can form \(q(z)^\mathsf Tk(\phi)\) with rank up to the projected key dimension. A fixed projection using two coordinates can form \(x(z)\cos\phi+y(z)\sin\phi\), and a second gives the other component of a complex rotation. The host also has nonlinearities and a numerical text stream. For the released fit, the output architecture enforces neither an equivariant code action nor a continuous transport law. The fit's measured extrapolation failure does not establish an expressivity limit of frozen hosts. [Scaled dot-product attention](https://arxiv.org/abs/1706.03762)

#### Work F spectrum

For code pair \(z\), use \(D_z=p_+(z)-p_-(z)\) from `cal_ref` and \(w_z=\min(m_+(z),m_-(z))/\sum_{z'}\min(m_+(z'),m_-(z'))\) from the archived conditional code masses. Compute \(M=\operatorname{diag}(\sqrt w)D\) and the Gram spectrum of \(M^\mathsf TM\). The row sum is zero. In a cyclic orbit of a row, a nonzero frequency-1 component spans a real plane and a nonzero frequency-2 component adds the Nyquist line. The verifier reconstructs the half-count ratios and finds both on the positive-weight orbit of code pair index 0: rows \(0,64,128,192\). Thus the **archived calibration table has rank three**. This is a statement about the finite reference table, with its calibration sampling error.

| Contrast | Nonzero Gram eigenvalues | Frobenius norm | Stable rank | Effective rank |
|---|---|---:|---:|---:|
| Physical `cal_ref` | \(0.405035,\ 0.401909,\ 6.20094\cdot10^{-5}\) | 0.898335 | 1.99244 | 2.00149 |
| Qwen phase, mean probability table | \(5.76490\cdot10^{-4},\ 2.43383\cdot10^{-4},\ 1.80399\cdot10^{-4}\) | 0.031627 | 1.73511 | 2.63904 |

With \(p_i=\lambda_i/\sum_j\lambda_j\), the effective rank is \(\exp(-\sum_i p_i\log p_i)\) and the stable rank is \(\sum_i\lambda_i/\lambda_1\). An arbitrary rank-one matrix would retain at most \(50.19\%\) of the physical Frobenius energy and leave a relative Frobenius residual of \(0.70576\). The host phase contrast has \(3.52\%\) of the physical Frobenius norm and relative Frobenius error \(0.99076\). Work F's orientation gain is its normalized Frobenius projection onto \(M\). Its factorial effect instead uses a paired-mother measure and a signed contrast scaled by \(1/2\). A probability-spectrum rank is different from attention rank or additive-logit rank, since softmax is nonlinear.

### Theorem 6 — Reynolds averaging and NLL

For probability predictions set \((Pf)(x)=\lvert H\rvert^{-1}\sum_{g\in H}R_{-\chi(g)} f(gx)\). This is an idempotent projection onto equivariant functions. For an invariant joint test law, or a finite test evaluated uniformly on every full orbit,

\[
\mathbb E[-\log (Pf)_Y(X)]\le\mathbb E[-\log f_Y(X)].
\]

**Proof.** Reindex the group sum after replacing \(x\) by \(hx\), then factor out \(R_{\chi(h)}\). An already equivariant \(f\) is fixed. For the inequality, apply convexity of \(-\log\) to the average of \(f_{y+\chi(g)}(gx)\). Its upper bound is the mean raw transformed loss. Invariance of the test law gives the displayed result after integration. With a finite incomplete rotation set the empirical comparison can have either direction. This proof uses log loss on positive probabilities. \(\square\)

Top-1 accuracy can fall on a complete orbit. Use aligned target 0. At three quarters of the orientations, assign the correctly classified vector \((0.35,0.34,0.16,0.15)\), and at the other quarter assign \((0.01,0.97,0.01,0.01)\). The average is \((0.265,0.4975,0.1225,0.115)\). Accuracy drops from \(75\%\) to zero while NLL drops from \(1.93866\) to \(1.32803\). Work G reproduces these metric directions on its three sampled rotations: NLL \(1.44523\to1.41131\), and accuracy \(23.80\%\to17.05\%\). The paired 95% interval for \(P f\) minus raw accuracy is \([-7.09,-6.41]\) percentage points. The verifier replays the calculation from the released raw score tables and locked temperatures. Symmetry projection alone does not restore the signed physical decision. [Stage 3D projection and metrics](../../stage3d_qwen_signed_phase/recompute_all.py)

## 4. Stage 4 `EquivariantEvidenceConnector`

1. **Canonicalize the inputs.** Accept integer `codes[B,2]`, a task enum, `PhaseContext`, and optional M3 evidence. Compute \((u,v)\) and the task's output shift. The branch sees only canonical inputs, invariant context, and fixed-frame pilot evidence. Return exact uniform probabilities immediately for `q3_indep`, and for `q2` when there is no usable shared pilot.
2. **Couple evidence and phase.** Start with canonical source distributions, initializing from the frozen S17 table. Propagate suitable circular moments with \(U_n(d)\) on each edge. For instance, transport \(m_{1,n}\) forward to the repeat target and de-rotate the second-window evidence with the known signed mean, while marginalizing uncertainty with nonnegative diffusion. Feed the real and imaginary components, source probabilities, and physical head features into the **same** trainable token projector. For exact continuous likelihood work, integrate a positive density or use Fourier harmonics appropriate to the sector and pilot maps. A first-moment feature alone cannot preserve a multimodal likelihood.
3. **Emit packed soft tokens.** Use a fixed 128-wide evidence vector and a 16-wide task embedding. Map through `Linear(144,128)`, GELU, and `Linear(128,1024)`, then reshape to eight 128-wide tokens. Apply a shared `Linear(128,2560)` and `LayerNorm(2560)`. Use a fixed task prompt and canonical candidate order in the existing Qwen four-candidate scoring interface. Train through the input embeddings with host weights frozen. Equivalent items can share an evaluation when they have identical canonical evidence and context.
4. **Use a bounded physical readout.** For Q3 repeat, the tested CPU implementation interprets canonical host scores as \(\delta=0.15\tanh(\ell_{\rm host})\), then transports the source \(\operatorname{softmax}(\log A_u+\delta)\) through the AWGN/Brownian kernel. A host bypass returns the direct S17 baseline. Q1 reads a relative root difference, Q2 reads a root relative to the common pilot frame, and post-pilot repeat integrates over the target phase conditional on the M3 factors. For an exact transport law as \(\theta\) changes with upstream evidence fixed, keep the source reweighting independent of target transport, or use the bypass. The tested residual reads coupled phase features, so it can change the source adjustment as \(\theta\) changes. The discrete \(H\) law holds for the composite in either case.
5. **Normalize once and lift.** Gather canonical probabilities into the physical choice order. For bitwise table parity, evaluate each distinct representative once at fixed context and reuse its canonical vector. Independently evaluated FP32 rows may differ by roundoff due to batch shape or kernels. Apply the same construction for candidate logits when used. Unrestricted free-text generation is outside the guarantee.

The design budgets 486,740 trainable parameters, counting two \(4\times16\) code embeddings, the \(5\times16\) task embedding, the projector and normalization, and an optional \(128\to4\) head. This leaves room under 2.2M for M3 calibration. A physical-head-first schedule permits a host-free starting point, but frozen-Qwen convergence and Modal free-credit runtime need measurement. The CPU verifier uses a shared frozen MLP and B's physical S17 prior. The A/B result combines quotient sharing, coupled tokens, and the physical head.

### M3 interface — `RefSufficientPool`

- Per acquisition accept `acquisition_id`, `shares_channel`, `timestamp`, `snr_linear`, known pilot root or sequence, and `matched_T` in the declared receiver frame. Deduplicate by acquisition identity before combining likelihoods. Keep fresh independently acquired pilots, including those at a shared timestamp. Independent-channel rows have no shared-channel likelihood factor under the stated model.
- Return `valid_mask[B]`, `pilot_frame_turn[B]` (zero under the fixed external frame), and `log_likelihood(phase_nodes, channel_turns) -> [...,4]` or equivalent factors grouped by timestamp. Equal-time independent AWGN pilots with known roots \(d_i\) reduce to \(S_p=\sum_i \rho_i T_{p,i}\overline{d_i}\) and \(\log L_p(g,\phi)=2\Re[S_p i^{-g}e^{-i\phi}]+C\). Retain relative likelihood scales across phase nodes and channel turns. Differently timed pilots use a shared Brownian chain in chronological order, which handles \(t_p>t_3\) without assigning a winding number to a wrapped observation.
- Q2 uses \(S_1-G\) with output shift \(a-r_p\). Post-pilot repeat uses the conditional target phase with shift \(a\). M1 owns the choice ordering and final gather. An isolated marginal on \(G\) discards dependencies needed by other tasks, and a single mean phasor can discard the sector modes.

### M4 interface — `PhaseContext` for Pluto and CTTC

- Supply `t1`, `t2`, `t3` and per-pilot timestamps in a common time unit and clock basis. Supply signed `omega_rad_per_unit` under \(T(t)\propto e^{+i\omega(t-t_1)}\), `diffusion_var_per_unit`, `snr_linear` with matched-filter normalization, and the frame and tracking state. Synthetic mapping is `t1=0`, `t2=h`, `t3=h+ell` and variance rate \(\sigma_\phi^2\). CFO in Hz uses \(\omega=2\pi f_{\rm CFO}\) with time in seconds. If the front end already de-rotates I/Q, pass the residual under the same sign convention.
- Supply an optional posterior on shared CFO and its estimate age or covariance if available. For independent Gaussian rate uncertainty of variance \(v_\omega\), the single-edge attenuation includes \(e^{-n^2v_\omega d^2/2}\). The same uncertainty is correlated across edges. Keep a shared latent or a joint law for multi-window and pilot inference. A rate uncertainty uses \(d^2\), whereas Brownian phase variance uses \(d\).
- Expose observation confidence and erasures, and specify any mismatch to the three-sample encoder, matched-filter normalization, circular noise, or drift law. Code-space equivariance is independent of the hardware label accuracy.

## 5. CPU verification and Round 2 decisions

Run from the repository root.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py
~~~

The verifier uses the production loader and fingerprint, regenerates the paired mothers, and matches all blind seeds, code pairs, and repeat targets to the archive. It uses 20,480 signed-CFO training rows, 16 fixed epochs, three optimizer seeds, and a single shared frozen MLP architecture with identical weights and fixed text embedding. A has 6,936 trainable parameters and B has 6,520. B uses a frozen physical prior while A learns from the inputs only. Scores use fractional ties. Paired intervals average signs and rotations within each 64-mother block. The Q1, Q2, and pilot-task tests exercise the quotient lift with synthetic canonical scores. Trainable end-to-end metrics are for Q3 repeat, with Q3 independent evaluated as a uniform control.

| Blind cell | A acc / NLL | B acc / NLL | C acc / NLL | Unrotated S17 MAP acc |
|---|---:|---:|---:|---:|
| G1 | 62.40% / 1.0432 | 64.62% / 0.9418 | 64.62% / 0.9411 | 15.25% |
| G2 | 44.59% / 1.0527 | 46.78% / 1.0219 | 46.78% / 1.0222 | 41.80% |
| G3 | 59.06% / 1.1186 | 62.11% / 0.9796 | 62.11% / 0.9796 | 16.60% |
| G4 | 46.10% / 1.0748 | 47.75% / 1.0167 | 47.75% / 1.0159 | 43.07% |

On G1–G4 the physical oracle matches archived direct S17 probabilities within \(1.23\cdot10^{-15}\). At variance \(0.04\) and 2 dB, the fixed-mixture expected accuracy is \(45.72\%\) at \(\pi/4\), \(46.11\%\) at \(0.80\), \(64.63\%\) at \(\pi/2\), and \(64.59\%\) at \(1.60\). At \(1.60\) the unrotated decision has expected accuracy \(15.43\%\).

Each B seed exceeds 55% on G1 and G3. B has zero absolute probability defect when reusing canonical representatives, and at most \(5.96\cdot10^{-8}\) on ungrouped batched evaluation. The five task lifts and joint Q2 coordinate check all have zero defect. A breaks equivariance by up to \(0.77851\) but exceeds chance on G1 and G3. The requested A collapse is therefore unreproduced on this surrogate. The Stage 3D Qwen run and the unrotated S17 MAP control both show below-chance accuracy in those cells. B's top-1 parity with the physical-head-only oracle leaves the contribution of the frozen host unresolved.

- **M2** · Which matched Qwen ablation will isolate quotient sharing, the physical head, and token coupling? Should physical-head-only versus physical-head-plus-host-residual be the primary paired contrast?
- **M3** · Can `RefSufficientPool` supply a joint timestamped likelihood with a declared common pilot frame, including duplicate IDs, different pilot SNRs, and \(t_p>t_3\)?
- **M4** · Which CFO sign, time base, and residual convention does the hardware maintain, and can it provide uncertainty for the same CFO latent across all edges?

## Round 2 Convergence

Section 5 reports the original Round 1 CPU comparison. Round 2 uses a newly specified isotropic candidate projection and the matched architecture below. The regenerated JSON records the Round 2 run and the replayed historical Stage 3D evidence.

### 6.1 Four primary arms and the causal contrasts

Let \(f\) be the frozen host's ordered candidate logits, \(E_\theta\) the trainable three-slot projector, \(A_u\) the S17 first-window source prior, \(K\) the physical sector kernel, and \(R_\chi\) the task-specific output permutation. Each learned arm shares the same host, feature budget, training examples, initial parameter draws, minibatch order, optimizer, and number of updates within seed.

| Arm | Input to the host | Scored Q3 repeat distribution |
|---|---|---|
| 0 · Stage 3D-style separate | Raw S17 code pair. Code-only W1/W2 tokens and a separate phase token. | \(\operatorname{softmax}(f(E_\theta))\) |
| 1 · Equivariant Host-Pure | Canonical pair. W1/W2 token evidence contains products of source moments and signed phase propagators. | \(R_\chi\operatorname{softmax}(f(E_\theta))\) |
| 2 · Equivariant Physical+HostResidual | Same prefix architecture as Arm 1. The candidate logits are source corrections. | \(R_\chi\left[\operatorname{softmax}(\log A_u+\gamma\tanh f(E_\theta))K\right]\), \(\gamma=0.15\) |
| 3 · Physical only | Host-free independent control. | \(R_\chi[A_uK]\) |

Arm 1 can consume the frozen S17 probabilities as **token features**. Its answer scores come from the host, and it gives identical outputs even if the unused physical kernel is replaced by NaNs. Arm 0 also receives the same fixed S17 evidence in its code-only token branch. It reproduces Stage 3D's modality separation, with a matched CPU backbone and input features. The archived Qwen run remains a separate historical measurement.

The two additional Host-Pure training configurations cross **quotient on/off** with **products on/off**. They isolate the quotient effect at either product setting and the coupling effect at either quotient setting. All four use the same allocated parameters. Quotient models activate only the canonical rows of an otherwise equal-size embedding table. This intentional tying is part of the intervention. A separate Arm 1 retrain zeros the numeric phase slots and their products, with public numeric metadata still present in the fixed surrogate text embedding. This is the matched zero-slot learning control.

For fixed-text interventions use the same trained model, packet, prompt, positions, numeric text, and true kernel. Swap the entire six-slot vector between the two signed CFO contexts. In a second intervention, use a seeded balanced permutation of the sign slots that swaps both members of exactly half the paired mothers. All rotations of one mother inherit its slot assignment. For Arm 2, only the host residual changes; the transport used in \(K\) remains physical. The within-model \(\gamma=0\) control recovers Arm 3 to floating-point tolerance and is exactly slot-invariant. Direct Arm 2 versus Arm 3 comparison measures the **net effect of adding the residual**. The slot tests measure the role of the **signed numeric slot within that residual**. Both are needed because a learned slot can be active without improving the analytic baseline.

The verifier fits one scalar temperature per arm and seed on the disjoint \(\omega=\pm0.18\) development split. It reuses the matched temperature under all interventions and reports raw and calibrated likelihoods, accuracy, Brier score, ECE, and reference-normalized NIG. Arm 3 stays at \(T=1\). For the within-Arm-2 residual contrast the verifier also applies the same dev temperature to both sides, so the calibration difference does not enter that contrast. Paired intervals average the two signs, rotations, and fixed optimizer seeds inside each independent 64-mother block. These CPU endpoints and intervals do not constitute M2's joint promotion gate on a real host.

### 6.2 Candidate geometry and transfer to the true frozen host

The shared CPU surrogate has three 24-wide soft tokens, a frozen metadata text feature map, and a fixed nonlinear 64-wide MLP body. If \(H_4\in\mathbb R^{4\times3}\) is an orthonormal basis of the centered candidate subspace and \(Q\in\mathbb R^{64\times3}\) is a random orthonormal frame, set \(W_{\rm out}=H_4Q^\mathsf T\). Then \(W_{\rm out}W_{\rm out}^\mathsf T=I_4-\tfrac14\mathbf1\mathbf1^\mathsf T\). The three centered directions have equal readout gain. This fixes the output geometry without aligning the host's input Jacobian or tokens to a pretrained physical decoder.

The verifier differentiates the actual centered frozen logits with respect to the concatenated soft tokens on the 16 representatives at each dev and blind context. It records the singular spectrum, third singular value, distance to the centered direct-S17 logit target under the unconstrained tangent fit, and the minimum perturbation norm to reach the projected target. It also evaluates the native logit margin between the oracle winner and runner-up. At a token radius \(r=0.10\), the linearized pairwise margin range is

\[
\left[m-r\lVert J_{\rm winner}-J_{\rm runner}\rVert_2,\,
      m+r\lVert J_{\rm winner}-J_{\rm runner}\rVert_2\right].
\]

A positive lower endpoint preserves that *pairwise* preference in the tangent model. A nonpositive upper endpoint marks it unreachable at that radius. On dev only, a second diagnostic composes each token Jacobian with the shared final token projections (weights and biases). Stack the resulting maps \(J_iD_\theta E_i\) with uniform representative/context weights. The singular spectrum and \(\|(I-AA^+)d\|/\|d\|\) show whether a single shared change can reach the locked score contrasts locally. The reported tangent fit is unconstrained in that last projection and supplies no nonlinear or OOD guarantee. For Arm 2 these diagnostics apply to the **native host scores before transport**. The native scores are correction coordinates, so direct-S17 score alignment there is an engineering probe, not an optimal target for the residual.

For Qwen, use the same four-candidate order, prompt, frozen checkpoint, answer positions, and score rule in both host arms. Differentiate the full candidate log-likelihood including log-normalizers and all candidate tokens. M6's \(\Pi C_{\rm cand}E_{\rm vocab}D_{\rm answer}B\) is a host-specific local map. Calculate it at the deployed prompt and cache state; the CPU simplex projection is a surrogate control. Require matched, swapped, and shuffled slots at fixed text, compare to physical-only under paired scoring, and record candidate margins and useful singular directions before attributing any gain to Qwen. Historical locked-temperature advantages are insufficient evidence of a signed-slot gain, given M2's recalibration results. [M2 gate and Stage 3D forensics](../M2/THEOREM_M2.md) · [M6 Jacobian and host audit](../M6/THEOREM_M6.md)

### 6.3 Dual cache contract

| Packet | Word layout | Canonical evidence |
|---|---|---|
| Legacy | 4-bit \(4a+u\) | `turn=a`, `s17_inner=u`, absence indicator, canonical S17 evidence |
| Extended | `[2b turn \| 2b inner \| 6b angle \| 4b log-radial \| 2b quality]` | The same S17 evidence plus canonical angle, radial and quality bins with a presence bit and validity mask |

Use a versioned format per batch or shard. In the 16-bit word \(w\), `s17 = w >> 12`, `turn = w >> 14`, `angle = (w >> 6) & 63`, `radial = (w >> 2) & 15`, and `quality = w & 3`. A quarter turn adds \(r\,2^{14}\) modulo \(2^{16}\), changing the turn field alone. Legacy quarter turns add \(4r\) modulo 16. The encoder writes both from the original window; lossy polar fields cannot reconstruct the learned inner index. A zero payload can be valid, so erasure is a separate flag.

The shared connector embeds the inner state and its S17 moments, then appends optional polar and quality features with explicit presence and saturation flags. It transports the optional canonical complex evidence through the same phasor calculation. The sample implementation uses an angle midpoint and a bounded log-radial summary for tokens. In a calibrated **physical** reader, treat open bins as intervals and use \(P(w\mid\text{latent state},\text{calibration})\) or a valid factorization conditional on the S17 prefix. Multiplying two marginal likelihoods from the same waveform would double-count evidence. Until a 16-bit likelihood or source adapter is fit, Arm 2 can use its bit-exact S17 fallback and still accept the richer token features. The ablation trains only on 4-bit input. The 16-bit verifier checks compatibility and rotation, leaving its accuracy benefit unmeasured.

For a 2,560-wide frozen host, a 160-wide fused feature vector including task conditioning, `Linear(160,128)`, `Linear(128,1024)`, a shared `Linear(128,2560)` and `LayerNorm(2560)`, a \(16\times16\) code embedding, and a \(5\times16\) task embedding total **488,400** trainable parameters. The eight emitted tokens can occupy fixed per-arm marker slots under the same token-count budget. Phase propagation, calibrated likelihoods, and the answer permutation do not add trainable parameters. Keep the header's signed rate convention, time unit, quantizer edges, gain and variance conventions, and acquisition/validity mask. Count metadata and addresses when comparing actual read and storage budgets. [M5 packet and decoder design](../M5/THEOREM_M5.md)

### 6.4 Gauge and M3 joint factor

The synthetic intervention relabels the **actual transmitted choices** per window while leaving the external pilot frame fixed. A hardware receive-gauge change is a common change of receiver coordinates for the **same** transmitted symbols. A TX chart relabeling adds \(t\in C_4\) to every TX root index, including known pilots, while the channel coordinate absorbs the opposite shift. These are distinct operations under the signal law \(Y=i^{X+G}e^{i\phi}+\epsilon\).

| Action | Data and channel transformation | Q1 TX diff | Q2 TX label | Q3 repeat / after-pilot RX sector | Q3 independent distribution |
|---|---|---:|---:|---:|---|
| Synthetic \((r_1,r_2)\), fixed pilot | Rotate each received window accordingly; shift \(X_1,X_2\) by \(r_1,r_2\); leave \(G\) and pilot fixed | \(r_2-r_1\) | \(r_1\) | \(r_1\) | Uniform |
| RX only \(r_{\rm rx}\) | Rotate all receiver-coordinate observations and references; \(G\mapsto G+r_{\rm rx}\); TX labels fixed | \(0\) | \(0\) | \(r_{\rm rx}\) | Uniform |
| TX chart only \(t\) | Known and unknown TX labels \(X\mapsto X+t\); \(G\mapsto G-t\); receiver data fixed | \(0\) | \(t\) | \(0\) | Uniform |
| Joint TX+RX \((t,r_{\rm rx})\) | Rotate receiver observations by \(r_{\rm rx}\); relabel TX by \(t\); \(G\mapsto G+r_{\rm rx}-t\) | \(0\) | \(t\) | \(r_{\rm rx}\) | Uniform |

The entries are additive output shifts. A realized independent sector label in an RX rotation can roll even though the predictive vector remains uniform. CFO and timestamps are unchanged by a constant gauge. A phase jump, changing an LO slope, or physically applying a new transmitter waveform defines a different intervention.

For real receive-gauge evaluation, anchor at the first valid window turn \(a_1\). Rotate receiver evidence and pilots by \(i^{-a_1}\) with exact quarter-turn sign swaps. Retain the gap \((a_2-a_1)\bmod4\), quality, and other invariant data required by the task. If the pivot is invalid, the connector must use a registered valid alternative with consistent task semantics or emit a declared fallback. M3's normalized \(J[k,q]\) has TX-label rows \(k\) and receiver-canonical sector columns \(q\). Under a common RX gauge change \(J\) stays invariant. Under the TX relabeling \(J'[k,q]=J[k-t,q]\) if M3 runs in the relabeled TX chart. Take

\[
q_{\rm Q2}(k)=\sum_qJ[k,q],\qquad
q_{\rm Q3,rx}(q)=\bigl(R_{a_1}\sum_kJ[k,\cdot]\bigr)_q .
\]

Q2 stays in the TX frame with a zero receive-frame lift. If M3 stays in a fixed TX basis, apply the display chart shift \(t\) once. If M3 already reports in the relabeled basis, the same roll is already included. When `valid_mask` is false for usable pilot evidence, the registered uniform-QPSK protocol falls back to uniform Q2 and to the no-pilot repeat posterior for Q3. Arm 1 may pass \(J\) or its marginals as fixed-size tokens with a valid mask and still use only host scores. In a physical/residual joint extension, weight the base \(J\) by \(\exp(\delta_2(k)+\delta_3(q))\) with bounded canonical terms and normalize once before marginalizing. Arm 3 uses \(J\) directly. If \(J\) already incorporates the caches and pilots, a second multiplication by separate cache or pilot factors would reuse data. [M3 joint posterior and coordinate action](../M3/THEOREM_M3.md)

### 6.5 Shared CFO, correlated phase, and R3

Let \(\Omega\sim N(\widehat\omega,v_\omega)\) be shared across all windows and pilot clocks, with Brownian diffusion rate \(v_\phi\). For a vector of signed intervals \(e\) with length \(d_e\), write \(\boldsymbol\Phi=\Omega\boldsymbol d+\sqrt{v_\phi}\,\Delta\boldsymbol B\). Let \(C_{ef}\) be the length of the overlap of intervals \(e,f\), times the product of their orientation signs. Gaussian integration gives the exact joint circular characteristic function

\[
\mathbb E e^{i\boldsymbol n^\mathsf T\boldsymbol\Phi}
=\exp\!\left(i\widehat\omega\,\boldsymbol n^\mathsf T\boldsymbol d
-\tfrac12v_\phi\boldsymbol n^\mathsf TC\boldsymbol n
-\tfrac12v_\omega(\boldsymbol n^\mathsf T\boldsymbol d)^2\right).
\]

For one edge the harmonic damping is \(e^{-n^2(v_\phi|d|+v_\omega d^2)/2}\). For disjoint Brownian edges the CFO covariance is still \(v_\omega d_e d_f\). Multiplying separately marginalized single-edge kernels would discard that term. An equivalent numerical joint implementation takes one Gauss–Hermite draw \(\omega_j\) for the entire Brownian chain, evaluates M3's unnormalized \(\widetilde J_{\omega_j}\) including the shared pilot likelihood at its timestamp, sums with quadrature weights, and normalizes **after** the sum. If the consumer has only normalized \(J_{\omega_j}\), also require its evidence normalizer \(Z_{\omega_j}\) to form the weighted mixture. Disjoint metadata or previously conditioned observations may supply \((\widehat\omega,v_\omega)\); if the same pilot updated this rate posterior, condition on it once in a shared model. Clock order allows \(t_p>t_3\) when that pilot was observed by the declared prediction deadline. This is a release constraint, independent of the existence of a mathematical smoother.

For Pluto/CTTC, M4's SNR-preserving R3 lift is \(Y=eT+B\eta\) with \(e=(1,1,1)/\sqrt3\), \(B^*e=0\), \(B^*B=I\), and an independent ancillary proper Gaussian \(\eta\) in the complement. Hence \(e^*Y=T\), preserving the actual normalized matched scalar and its SNR. Pass estimated post-gain SNR and its calibration separately. Share the realized ancillary draws across arms. In a gauge replay rotate the realized window and any receiver-coordinate complex nuisance or calibration factors consistently. Rate, scalar SNR, symbol clock, and capture IDs stay fixed. The software verification uses synthetic matched sums and realized lifts, with actual M5 packing and the two trained CPU arms. Pluto and CTTC capture performance remains unmeasured. [M4 R3 lift and receiver evaluation contract](../M4/THEOREM_M4.md)

### 6.6 Round 2 verification results

The prescribed `uv run` command exited **0** on Python 3.12.7, NumPy 2.3.5, SciPy 1.17.0, and PyTorch 2.12.1+cpu. Each learned configuration had **9,064 trainable parameters**, 20,480 training rows per seed, and 16 epochs. Scalar temperatures used 4,096 independent dev rows. The final verifier uses FP64 to measure the Gram matrix of the stored FP32 readout.

| Cell | Arm 0 acc / NLL | Arm 1 acc / NLL | Arm 2 acc / NLL | Arm 3 acc / NLL |
|---|---:|---:|---:|---:|
| G1 | 62.16% / 1.04094 | 64.60% / 0.97853 | 64.62% / 0.94222 | 64.62% / 0.94113 |
| G2 | 45.21% / 1.05384 | 46.81% / 1.02343 | 46.78% / 1.02174 | 46.78% / 1.02223 |
| G3 | 58.80% / 1.07606 | 61.91% / 1.01301 | 62.11% / 0.98098 | 62.11% / 0.97957 |
| G4 | 45.18% / 1.05790 | 46.97% / 1.02097 | 47.75% / 1.01639 | 47.75% / 1.01585 |

- **Host-Pure suffices for over 55% on G1 and G3 in every seed.** It reaches 0.916 and 0.918 reference-normalized NIG in those cells. Its mean ten-bin ECE across seeds ranges from 0.0119 to 0.0402 across the four cells. Arms 1 and 2 have exact zero defect with canonical reuse and maxima \(1.49\times10^{-7}\) and \(5.96\times10^{-8}\) on the ungrouped path. Arm 0's maximum defect is 0.84783, although it also exceeds 55% on G1 and G3.
- **Quotient sharing accounts for the consistent score improvement in the matched factorial.** Adding the quotient improves dev-calibrated NLL in every cell at both product settings. On G1 the paired changes are \(-0.07101\) nats with separate inputs and \(-0.06804\) with coupling. Adding the product terms at fixed quotient increases NLL by \(+0.00860\) \([+0.00733,+0.00988]\) on G1 and \(+0.00702\) \([-0.00057,+0.01462]\) on G3. The products show no consistent incremental proper-score gain in this surrogate.
- **Host-Pure uses the signed slots.** On G1, sign swap and balanced shuffle increase Arm 1's NLL by \(2.0172\) and \(1.0072\) nats. On G3 the increases are \(1.9229\) and \(0.9677\). The zero-slot retrain retains the public text and falls to 15.38% and 16.80% on G1 and G3. The quotient-separate control still passes 55% in both cells, so the numeric slot matters even though explicit multiplication is dispensable on this host.
- **The residual has no measured incremental gain over the physical control.** Its top-1 accuracy is identical to Arm 3 across G1–G4. On G1 it increases dev-calibrated NLL by \(+0.00109\) nats \([+0.00050,+0.00169]\) against Arm 3 at \(T=1\), or \(+0.00085\) \([+0.00028,+0.00143]\) against \(\gamma=0\) at the same temperature. The intervals for its sign-swap and shuffle effects include zero in all four cells. The physical kernel drives the extrapolated decision in this arm, and the data do not establish a useful signed-slot effect in its residual.
- **Local reachability and learned alignment are different observations.** All tested candidate Jacobians have centered rank three. The dev last-projection stacks have rank 192 on 64 contexts/representative rows, the maximum after candidate centering. For Arm 1 on G1, seedwise minimum third singular values range from 0.177 to 0.200, median native oracle winner/runner-up margins from 0.522 to 0.785, and native oracle top-1 agreement from 96.9% to 100%. The margin certificates apply only to the stated local linearization. The weaker native Arm 2 margins are measured in source-correction coordinates and do not diagnose its final Q3 accuracy.
- **The interface checks passed on synthetic data.** All 65,536 values preserve the 4-bit prefix and invariant suffix fields under every quarter turn. Both trained arms have zero observed common-gauge defect for both input formats in the R3 fixture. The matched-sum error is \(9.04\times10^{-16}\). The shared-CFO Gauss–Hermite check agrees with the closed form within \(1.90\times10^{-15}\); an independently marginalized rate on each edge changes the fixture's joint moment by 0.415. M3's joint marginal and gauge checks use a normalized fixture. They are not learned or hardware posterior performance tests.

For the next frozen-Qwen comparison, keep the quotient and the physical-only control fixed. Carry the quotient-separate comparator alongside Host-Pure so the effect of explicit multiplication can be measured. Treat Physical+HostResidual as an experimental increment on Arm 3 and require a held-out gain under the same dev calibration and fixed-text controls. M2's simultaneous gate, Qwen candidate geometry, 16-bit predictive gains, and hardware accuracy remain separate measurements. Intervals here are descriptive paired block intervals for the fixed three seeds. [Round 2 numerical receipt](results_m1.json) · [Handoff](reply_r2.md)
