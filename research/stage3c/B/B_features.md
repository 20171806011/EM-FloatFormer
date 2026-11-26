# Signed carrier-offset features for variable window times

**Result.** For the known, constant carrier frequency offset (CFO) and Brownian drift in Stage 3, a compact complete description of the latent phase law at a time $t_j$ is the pair $(m_j,v_j)=(e^{i\omega t_j},\sigma^2t_j)$. Keep the role of each time attached to its pair. The sine and cosine of the phase at the prediction time alone are insufficient in general. Even all three endpoint phasors can be insufficient without diffusion times. An uncertain offset requires integrating a common offset over the whole trajectory.

The base is [cbf3d2c](https://github.com/20171806011/EM-FloatFormer/tree/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0). The simulation in [data.py](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3_ministral_cache/data.py) uses $0,h,T=h+\ell,t_p$, a common initial cardinal rotation $i^G$, and distinct noise for each acquisition. Condition metadata in [schema.py](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3_ministral_cache/schema.py) specifies SNR, drift family, and times. This note treats the Brownian and deterministic limits. Different drift families must supply their own transition law.

## Exact metadata theorem

Let $t_j\ge0$, $\omega\in\mathbb R$, $0\le\sigma<\infty$, and $U_j=\exp(i[\omega t_j+\sigma W(t_j)])$ with a single standard Brownian motion starting at zero. For any integer multi-index $k$, its circular moment is

$$
\mathbb E\prod_j U_j^{k_j}
= \prod_j m_j^{k_j}\exp\!\left[-\tfrac12\sum_{a,b} k_a k_b\min(v_a,v_b)\right].
$$

Proof. The unwrapped Brownian phases are jointly Gaussian with covariance $\sigma^2\min(t_a,t_b)=\min(v_a,v_b)$. Apply their characteristic function. Equality of all these moments gives the same law on the torus (trigonometric polynomials are dense in continuous functions). Conversely, the first marginal moment is $m_j e^{-v_j/2}$. At finite $v_j$ its magnitude determines $v_j$ and its direction determines $m_j$. Therefore two configurations have the same **latent rotor law** on the specified roles if and only if their corresponding pairs agree. This exact characterization concerns the full latent law. Particular encoders, protocols and losses may discard further distinctions.

The encoder, measurement-noise law, symbol priors, and reference protocol must be the same in the comparison. Integrating their common observation and target kernel against identical rotor laws gives identical joint laws for the observed cache $Z$, any authorized pilot measurements $P$, and the answer $A$. Hence $p(A\mid Z,P)$ is identical wherever defined. Equivalently, on sorted distinct times, use each adjacent rotor $m_b\overline{m_a}$ and each incremental variance $v_b-v_a$. At repeated times reuse one latent state. Two fresh readings then have independent measurement noise; one repeated acquisition still counts once.

### Task-dependent feature selection

| Question under the Stage 3 message prior | Relevant noninitial times for an exact general readout | Reason |
|---|---|---|
| Q1: relative symbol from both cached windows, no pilot | $h$ | Only the process through the second observation enters the relative-symbol likelihood. |
| Q2: initial symbol, with same-channel pilot | $h,t_p$ | Second-window cells can inform residual phase jointly with the pilot. When $t_p<h$, chronology runs through the pilot first. |
| Q2: no useful pilot | None for the answer | Uniform symbols and the uniform initial cardinal gauge leave four equiprobable labels. |
| Q3: third window repeats the first symbol, before the pilot | $h,T$ | The second window may inform the shared residual drift en route to the third. |
| Q3: repeat, after a shared pilot | $h,T,t_p$ | Inference uses the joint process over all supplied observations and the prediction time. |
| Q3: fresh independent uniform third symbol | None for its sector probabilities | Uniform cardinal orientation and circular measurement noise give four equal sectors, also conditional on cache and pilot. |

Interface recommendation: with timings and diffusion already given, one signed, unwrapped scalar $\omega$ suffices for a deterministic front end to compute every pair. For a numerical connector, use the real and imaginary parts of each relevant phasor and $v_j$ (or retain the times and $\sigma$). These redundancies aid implementation; the number of network inputs is not a lower bound. Keep the positive sign for a future received sector. A known deterministic drift can be removed with $e^{-i\omega t_p}$ from a raw pilot; integer cache indices generally cannot be so rotated. For the published C4 ramp the three phasor angles are $(0.70,1.40,1.40)$ rad, and $(v_h,v_T,v_p)=(0.02,0.04,0.04)$ rad². Under another timing these numbers can all change.

## Three exact scope checks

1. **Identical unwrapped total angle, variable times.** Use noiseless cardinal signals and $\sigma=0$, $\omega=\pi/2$, $T=4$. Take $(h,\ell,t_p)=(1,3,1)$ and $(2,2,2)$. Condition on the same first window, second window and pilot, all centered on $+1$. These observations have positive probability under uniform labels. The pinned equivariant encoder preserves their quadrant in each code. The Q1 and Q2 labels are respectively $(3,1)$ and $(2,2)$ (units of a quarter-turn); direct enumeration is in the check. Both cases share even the unwrapped total $2\pi$. This collision concerns omitting the other times. If $T>0$, all times are public, and the *unwrapped* total phase is supplied, division by $T$ recovers $\omega$.
2. **A wrapped total can alias even with all times retained.** Fix $(h,T,t_p)=(1,4,2)$. For $\omega=0$ and $\omega=\pi/2$, $e^{i\omega T}=1$, yet the same observed windows and pilot imply Q1/Q2 labels $(0,0)$ and $(3,2)$. A specified offset range over which $\omega\mapsto e^{i\omega T}$ is injective avoids this particular alias. Retaining signed endpoint phasors directly handles wraps without reconstructing the unwrapped slope.
3. **Identical triple of signed angles, different diffusion.** Use $h=0$, $t_p=T$, $\sigma=0.2$, noiseless measurements and Q3-repeat before the pilot. The first cache identifies received initial orientation $+1$ exactly in this limit. At $T=2$, $\omega=0.05$ and $T=200$, $\omega=0.0005$, the three deterministic angles $(0,0.1,0.1)$ agree, while both initial caches are independent of future Brownian increments. The future angle is $N(0.1,0.04T)$ modulo $2\pi$. Summing normal-CDF differences over periodic sector copies gives sector-0 probabilities **0.991436** and **0.258204**. The omitted tails in the check have total mass below $2\Phi(-10)$. These settings stress the mathematical family; the long interval was not part of the historical host benchmark.

## Uncertain or absent offset

Let the public data about the offset induce a prior $\pi(d\omega)$ and let $O$ denote observed caches, prompt and relevant pilot. The optimal rule uses

$$
p(a\mid O)=\int p(a\mid O,\omega)\,
  \frac{p(O\mid\omega)\pi(d\omega)}{\int p(O\mid w)\pi(dw)}.
$$

For known magnitude $\omega_0$ and hidden sign there are two **whole-path** hypotheses, with the sign posterior updated by $O$. Independent sign draws at each time would assign mass to trajectories forbidden by the constant-offset model. For an independent Gaussian offset $N(\mu,\tau^2)$ and public times, the joint unwrapped mean is $\mu t_j$ and covariance gains the rank-one term $\tau^2t_at_b$ in addition to Brownian covariance. This is an equivalent analytic marginalization when the model assumptions hold. If a nominal offset is missing and no prior or decision criterion is specified, there is no unique Bayes answer over offset values. A magnitude-only feature equals the known-sign solution only where the two posteriors agree, or for classification where their optimal actions coincide in the relevant mixture.

**Controlled cost of withholding the sign.** Take $h=0$, $T=1$, no revealed pilot, no Brownian drift, and give the receiver the true initial received orientation $+1$. Let $\omega=\pm\pi/2$ with equal prior, independent of all measurements at $t=0$. For Q3-repeat take circular measurement SNR 2 dB. Each sector is the intersection of two half-planes with orthogonal normals: given phasor angle $\theta$, its probability is

$$
M_k(\theta)=\Phi\!\left(\sqrt{2\rho}\cos(\theta-k\pi/2+\pi/4)\right)
             \Phi\!\left(\sqrt{2\rho}\cos(\theta-k\pi/2-\pi/4)\right).
$$

The two laws put probability 0.802765 on different sectors. Observing the sign gives Bayes accuracy **80.2765%**. With hidden sign the optimal mixture puts 0.406793 on each of those two sectors, giving **40.6793%**. Log loss increases by **0.506436 nats** (the equal-mixture Jensen–Shannon divergence). This isolates the value of knowing the sign for this controlled conditional question. The actual frozen host receives no oracle initial orientation. Neither these values nor the exact collisions predict a measured host score.

Reproduce the checks with `python check_b.py` from this folder with Python 3; no third-party packages are needed. The checked-in `results.json` contains the enumerations and probability vectors.

