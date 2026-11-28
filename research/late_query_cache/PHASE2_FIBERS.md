# Phase 2 A2 — A compact nonlinear sensor with a strict tangent-index gap

## Statement

Let $(x,y,z)$ be independent uniform variables on the compact unit cube. Put
$s=\ln(1+x)$, $v=\ln(1+y)$, and $L=\ln 2$. In $(s,v,z)$ coordinates the domain is $[0,L]^2\times[0,1]$ with probability density $e^{s+v}\in[1,4]$. Let both the question $t$ and the sensing action $a$ range over $[0,1]$. The demand and the single noiseless real observation are
$$
T_t=z-t(1+x)=z-te^s,\qquad O_a=h_a(x,y)=\ln(1+y)-a\ln(1+x)=v-as .
$$
Every $h_a$ is smooth with nonvanishing differential. Each $T_t$ is affine in the original coordinates. The encoder sends a common fixed-length $b$-bit message $M$ before the question. The receiver chooses $a=a(M,t)$ and makes a measurable estimate from $(M,t,O_a)$. Write $b_{\rm act}(\varepsilon)$ for the minimum $b$ giving $\sup_{t\in[0,1]}E[(\widehat T_t-T_t)^2]\le\varepsilon$. Write $b_{\rm pre}$ for the corresponding minimum if the sensing action may depend on $M$ but must be chosen before $t$. Both codes still give the answer *after* learning $t$. For a known question $t$ write $b_t$.

The tangent computation at every base point, with the point and the sensor family given to the tangent problem, needs exactly one real linear message coordinate. The true high-resolution exponents, using base-two logarithms for bits, are
$$
\lim_{\varepsilon\downarrow0}\frac{b_t(\varepsilon)}{\log_2(1/\varepsilon)}=\tfrac12,\qquad
\lim_{\varepsilon\downarrow0}\frac{b_{\rm act}(\varepsilon)}{\log_2(1/\varepsilon)}=\tfrac34,\qquad
\lim_{\varepsilon\downarrow0}\frac{b_{\rm pre}(\varepsilon)}{\log_2(1/\varepsilon)}=1 .
$$
The first estimate holds uniformly in $t$. A2's universal equality between the tangent vector index rate and the global exponent is therefore false. Even the two endpoint questions already attain the two joint exponents. Allowing the entire continuum preserves them. The model is in the noiseless nonlinear-fiber part of A, distinct from the Gaussian coordinate-neighbor part.

## Why the tangent computation gives one dimension

At a fixed $(s,v,z)$ send the linear form $\omega=dz-e^s\,dv$ on the tangent perturbation. For every $t$, taking $a=t$ gives
$$
dT_t=\omega+e^s\,dh_t,\qquad dh_a=dv-a\,ds .
$$
No sensor supplies any $dz$ component, while every demand contains $dz$ with coefficient one. Varying $z$ alone proves that zero-dimensional linear compression cannot work, even for vector or block versions with an independent $z$ direction in every block. One tangent dimension corresponds to coefficient $1/2$ for mean-square precision. The common form displayed here satisfies $\omega\wedge d\omega=-e^s dz\wedge ds\wedge dv\ne0$. This identifies a local nonintegrability obstruction for this particular tangent description. The all-code converse below establishes the stronger operational assertion, including arbitrary discontinuous encoders and different first actions for different messages.

## Attainable code on all questions and explicit bit bound

Take $r=\sqrt\varepsilon$, $0<r<1$, and let $s_0=\sqrt{r/2}\lfloor s/\sqrt{r/2}\rfloor$. Broadcast its index and a midpoint quantization $\widehat W$ of
$$
W=z-e^{s_0}v\in[-2L,1]
$$
with interval lengths at most $r$. The number of combined messages is at most
$$
 N_{\rm act}(r)\le (1+L\sqrt{2/r})\,(1+(1+2L)/r).
$$
A fixed-length encoding needs the ceiling of its base-two logarithm. On question $t$ take action $t$ and return
$$
\widehat W+e^{s_0}O_t+t e^{s_0}(s_0-1).
$$
The signed output error equals $(\widehat W-W)+t[e^s-e^{s_0}-e^{s_0}(s-s_0)]$. The exponential has second derivative at most $2$ here. The Taylor remainder contributes at most $(s-s_0)^2\le r/2$, and quantization contributes at most $r/2$. The bound $|\widehat T_t-T_t|\le r$ holds for every source value and all questions simultaneously. Thus $b_{\rm act}\le \frac34\log_2(1/\varepsilon)+O(1)$.

## Converse against all adaptive codes

Use $r=2\sqrt{\varepsilon}$ for the converse and suppose $0<r\le L^2/8$. Under the asserted risk requirement, the event on which the two endpoint errors both have absolute value at most $r$ has probability at least $1/2$ by Markov and the union bound. To bound its mass for one message, ignore the encoder cell and upper bound the whole region on which the corresponding two decoders would be accurate.

For this message let $a,b$ be the two actions used for questions $0,1$, and let $f,g$ be the corresponding scalar answer functions. The two $z$-constraints are
$$
 |z-f(v-as)|\le r,\qquad |z-e^s-g(v-bs)|\le r .
$$
Their intersection has $z$-length at most $2r$ and is empty unless $|f(v-as)-g(v-bs)-e^s|\le2r$. Let $\mu$ denote the *Lebesgue area* of this last event in $[0,L]^2$ in coordinates $(s,v)$. Since the source density is at most $4$, the probability accounted for by this message is at most $8r\mu$.

**Equal actions.** If $a=b$, change variables to $(s,k=v-as)$ with unit Jacobian. For given $k$, the inequality restricts $e^s$ to an interval of length $4r$. Since $(e^s)'\ge1$, its allowed $s$-length is at most $4r$. The $k$ range is contained in $[-L,L]$. Thus $\mu\le8Lr$, and the one-message probability is at most $64Lr^2$.

**Two different actions, uniformly even as they approach each other.** Put $d=b-a\ne0$, $D=|d|$, $k=v-as$ and $\ell=v-bs$. The Jacobian magnitude of this change is $D$, $s=(k-\ell)/d$, and $k,\ell\in[-L,L]$. Write $E(k,\ell)$ for the good area event with its original parallelogram support included. Then
$$
\mu=D^{-1}\int\!\int 1_E(k,\ell)\,d\ell\,dk,\qquad
\mu^2\le\frac{2L}{D^2}\iiint 1_E(k,\ell)1_E(k,\ell')\,dk\,d\ell\,d\ell'.
$$
The second step is Cauchy–Schwarz and Tonelli. For a fixed pair $\ell\ne\ell'$, simultaneous incidence requires $|\ell-\ell'|\le LD$ and restricts $k$ to an interval of length at most $LD$ by the strip constraints. It also requires
$$
\left|g(\ell)-g(\ell')+e^{(k-\ell)/d}-e^{(k-\ell')/d}\right|\le4r .
$$
On the common strip the derivative in $k$ of the exponential difference has constant nonzero sign and absolute value at least $|\ell-\ell'|/D^2$: both exponents lie in $[0,L]$, where the exponential derivative is at least one. The mean-value theorem therefore bounds admissible $k$-length by $\min(LD,8rD^2/|\ell-\ell'|)$. The diagonal is a null set. Integrating $\ell$ over length at most $2L$, and using both signs of $\ell'-\ell$, gives
$$
\mu^2\le 8L^2\int_0^L\min(L,8r/u)\,du
=64L^2r\left[1+\ln\!\left(\frac{L^2}{8r}\right)\right].
$$
The factors $D^2$ cancel exactly. In particular a code cannot evade the bound by picking two almost identical actions depending on the bit budget. Combining both action cases, summing over $2^b$ messages, and using the good-event probability gives
$$
2^b\ge \frac{1}{128L\,r^{3/2}\sqrt{1+\ln(L^2/(8r))}} .
$$
With $r=2\sqrt{\varepsilon}$ the lower bound is $\frac34\log_2(1/\varepsilon)-\frac12\log_2\log(1/\varepsilon)-O(1)$ bits. This proves the claimed strict slope. A question-independent random seed, or private random seeds independent of the source, cannot improve this exponent: condition on all seeds for the two tests, apply the uniform one-message upper bounds, and average. The two individual MSE guarantees still give the same unconditional good-event lower bound.

## Removing the logarithmic loss by one-dimensional Fourier cancellation

The elementary pair argument gives the exponent with no oscillatory-integral input. For these particular exponential fibers one can also get a bounded remainder in the message length. In the different-action case enlarge the transformed parallelogram to the rectangle $k,\ell\in[-L,L]$ intersected with the strip $0\le(k-\ell)/d\le L$. Extend $f,g$ measurably outside their original ranges if necessary. For frequency $\lambda\ne0$, let
$$
I(\lambda)=D^{-1}\iint_{[-L,L]^2}
 1_{[0,L]}((k-\ell)/d)
 e^{i\lambda[f(k)-g(\ell)-\exp((k-\ell)/d)]}\,dk\,d\ell .
$$
The normalized strip kernel $K_{\lambda,d}(w)=D^{-1}1_{[0,L]}(w/d)\exp(-i\lambda e^{w/d})$ depends only on $w=k-\ell$. Its ordinary unnormalized Fourier transform is
$$
\widehat K_{\lambda,d}(\xi)=\int_0^L \exp[-i(\lambda e^s+\xi ds)]\,ds .
$$
This formula holds for either sign of $d$; a negative $d$ reverses the integral endpoints and contributes the absolute Jacobian. The integration phase $\varphi(s)=\lambda e^s+\xi ds$ has monotone first derivative and $|\varphi''|\ge|\lambda|$. The interval where $|\varphi'|\le\sqrt{|\lambda|}$ has length at most $2/\sqrt{|\lambda|}$. On each of the at most two remaining intervals, integration by parts bounds the boundary contribution by $2/\sqrt{|\lambda|}$ and the total variation of $1/\varphi'$ by $1/\sqrt{|\lambda|}$. Consequently $\|\widehat K_{\lambda,d}\|_\infty\le8/\sqrt{|\lambda|}$ uniformly over $d$. By the $L^2$ convolution multiplier bound from Plancherel, applied to the two unit-modulus functions supported in $[-L,L]$ (each has $L^2$ norm $\sqrt{2L}$),
$$
 |I(\lambda)|\le16L/\sqrt{|\lambda|}.
$$
This step only uses measurability of the two answer functions. All smoothness is in the sensing law.

Use $1_{\{|u|\le2r\}}\le e^{1/2}\exp[-u^2/(8r^2)]$ and the Gaussian Fourier identity
$$
e^{-u^2/(8r^2)}
=\sqrt{2/\pi}\,r\int_{\mathbb R}e^{-2r^2\lambda^2}e^{i\lambda u}\,d\lambda.
$$
Fubini applies since the enlarged strip has finite normalized measure and the frequency weight is integrable. Bounding $\int_{\mathbb R}e^{-2w^2}|w|^{-1/2}\,dw\le5$ by splitting at $|w|=1$ gives
$$
 \mu\le C_L\sqrt r,\qquad C_L=80 e^{1/2}L\sqrt{2/\pi}.
$$
The same estimate covers the equal-action case for $r\le1$. With $r=2\sqrt{\varepsilon}$, the one-message probability bound $8C_Lr^{3/2}$ now implies $2^b\ge (16 C_L r^{3/2})^{-1}$. Combining it with the attainable code proves the stronger fixed-length result
$$
 b_{\rm act}(\varepsilon)=\tfrac34\log_2(1/\varepsilon)+O(1).
$$

## Curvature dichotomy for a class of smooth demands

With the same cube, sensors, and message order, replace $T_t$ by $z-tF(s)$ where $F\in C^2([0,L])$ is fixed. Every tangent problem still has dimension one, using $\omega_F=dz-F'(s)\,dv$. The active global rate has an exact dichotomy:
$$
 b_{\rm act}(\varepsilon)=
 \begin{cases}
  \frac12\log_2(1/\varepsilon)+O(1) & F\text{ affine},\\
  \frac34\log_2(1/\varepsilon)+O(1) & F\text{ nonaffine}.
 \end{cases}
$$
If $F(s)=As+B$, send $z-Av$ quantized to $O(\sqrt\varepsilon)$ and use $O_t$ to return its quantized value plus $AO_t-tB$. The independent $z$ direction supplies the corresponding lower bound.

For nonaffine $F$, put $M_2=\|F''\|_\infty>0$. With $r=\sqrt\varepsilon$ use an anchor grid of step $\sqrt{r/M_2}$ and quantize $W=z-F'(s_0)v$ with maximum error $r/2$. The reconstruction
$$
 \widehat W+F'(s_0)O_t+t\,[F'(s_0)s_0-F(s_0)]
$$
has absolute error at most $r$ by Taylor's theorem. The same $O(r^{-3/2})$ message bound follows, with constants depending on $F$.

There exists a closed nondegenerate subinterval $J\subset(0,L)$ where $|F''|\ge\kappa>0$ and $|F'|\ge c>0$. In fact $F''$ is continuous and nonzero somewhere in the interior, and $F'$ is strictly monotone on a small interval around such a point. Let $p_J=P(s\in J)>0$ and set $r=2\sqrt{\varepsilon/p_J}$. The event within that patch on which both endpoint answers are $r$-accurate has mass at least $p_J/2$. In the Fourier proof replace $1_{[0,L]}$ in the strip kernel by $1_J$ and $e^s$ by $F(s)$. The transform becomes $\int_J e^{-i(\lambda F(s)+\xi d s)}\,ds$; the same monotone-derivative integration by parts yields a bound $8/\sqrt{\kappa|\lambda|}$. In the equal-action case use the inverse slope bound $1/c$ on $J$. The one-message probability on this fixed patch is again at most $C_{F,J}r^{3/2}$. Summing over messages proves the converse. The constants need not be uniform when $F$ itself changes with precision or approaches an affine function.

## Question timing, and what remains open

If $a$ is chosen before the question, the equal-actions estimate applies to *every* codeword: $2^b\ge1/(128Lr^2)$ for $r=2\sqrt\varepsilon$. For the reverse bound quantize both $z$ and $x$ with cells of length $\sqrt{\varepsilon}$ and use $T_t=z-t(1+x)$. The maximum error is at most $\sqrt\varepsilon$ uniformly. This proves exponent $1$ with identical allowed measurement choices, side-information quality, and cube distribution. For one known question, encode its scalar target in the interval $[-2,1]$ at resolution $O(\sqrt\varepsilon)$. Conversely a single decoder for a fixed codeword and a fixed question allows a $z$-interval of length at most $2r$, since every sensor is independent of $z$ given $x,y$. The uniformly distributed $z$ alone forces the matching one-dimensional exponent.

The active encoder uses its coarse parameter to choose a local scalar reference and its question-dependent sensor to cancel the first-order error. With an identical pre-question action, the two endpoint residuals force resolution of a second continuous quantity. This last comparison shows that the actions and the side information influence the operational rate on this compact source.

The sharper Fourier bound leaves a bounded additive bit uncertainty in the joint active problem. The optimal constant and finite-resolution optimal code remain undetermined. The proof does not classify general smooth fibers or quantify the result when this particular sensor is perturbed by additional observation noise. The continuous incidence bound uses Cauchy–Schwarz, Tonelli and mean-value estimates. The bounded-remainder improvement additionally uses the Gaussian Fourier inversion identity and the $L^2$ Plancherel multiplier bound. Formal-check coverage is listed in the main Phase 2 report. The written continuum argument here does not by itself constitute Lean kernel verification.

Related prior work: Hua Sun and Syed A. Jafar, [*On the Capacity of Computation Broadcast*](https://arxiv.org/abs/1903.07597), develops two-receiver computation broadcast with dependent demands and side information, including the significance of nonlinear structure. No priority claim for a new general abstraction is intended.
