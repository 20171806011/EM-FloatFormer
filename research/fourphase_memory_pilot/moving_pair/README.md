# When phase evolves inside the data pair

This extension separates two clocks. The first data symbol arrives at time 0, the second at time $h$, and the known unit pilot at time $h+L$. The relative-phase question uses the two data observations before the pilot arrives. The first-symbol question uses the cached data and the observed pilot.

## Model and risks

All sums are on $C_4$. Let $X_1,X_2,G$ be independent uniform labels. A single phase-walk increment has mass $1-d$ at zero and $d/3$ at each nonzero value. Let $U$ and $V$ be the sums of disjoint blocks of $h$ and $L$ independent increments, respectively. The two data errors $E_1,E_2$ follow the same symmetric family with parameter $e$. The pilot error $E_p$ has parameter $e_p$. All errors, symbols and starting phase are independent before the observations are formed.

$$
Y_1=X_1+G+E_1,\qquad Y_2=X_2+G+U+E_2,
\qquad P=G+U+V+E_p.
$$

Write $k(s)=1-4s/3$. For a displacement with reliability $c$, its probability is $(1+3c)/4$ at zero and $(1-c)/4$ at each nonzero position. Define $b(c)=1-\max\{(1+3c)/4,(1-c)/4\}$. The minimum marginal probability of a wrong four-way label, with all the allowed observed data, is

| Question | Reliability | Bayes error |
|---|---|---|
| $R=X_2-X_1$, before the pilot | $c_R=k(e)^2 k(d)^h$ | $b(c_R)$ |
| $X_1$, after the pilot | $c_A=k(e)k(e_p)k(d)^{h+L}$ | $b(c_A)$ |

A zero-length increment block has reliability one. When $c\ge0$, the Bayes error is $3(1-c)/4$. The tables include an anti-correlated step as a check on the more general max formula.

**Proof.** A symmetric increment with reliability $c$ has probability vector $c\delta_0+(1-c)u$ for uniform vector $u$. Circular convolution obeys $u*u=u$ and $u*\delta_0=u$, so independent increments multiply their reliabilities. Negation preserves this symmetric distribution.

For fixed $G,U,V,E_1,E_2,E_p$, the fresh uniform symbols give each $(Y_1,Y_2)$ probability $1/16$. Conditioning on the data therefore leaves the marginal law of the phase increments and errors unchanged. The difference between $Y_2-Y_1$ and $R$ is $U+E_2-E_1$, which has reliability $c_R$.

For fixed increments and errors, the remaining three uniform variables $(X_1,X_2,G)$ map bijectively to $(Y_1,Y_2,P)$. Hence the increments and errors also retain their prior joint law after all three observations. The difference between $Y_1-P$ and $X_1$ is $E_1-U-V-E_p$, giving $c_A$. These conditional laws depend only on the centers $Y_2-Y_1$ and $Y_1-P$, respectively, so maximizing their probability gives the two reported risks. In particular $Y_2$ adds no information about $X_1$ given $(Y_1,P)$ under this independent uniform payload model. Coding or a known second symbol would alter that premise.

## Memory and gap checks

Storing the four-valued $Y_2-Y_1$ attains the full-observation relative risk with two bits. The same cache gives accuracy $1/4$ for $X_1$ after any pilot in this model. For fixed $X_1$ and pilot-generating variables, the independent uniform $X_2$ makes the cached difference uniform, and $P$ alone is independent of $X_1$. Storing $(Y_2-Y_1,Y_1)$ uses four bits and attains both full-observation marginal risks. This is a statement about risk attainment in the discrete observation model. For $c_R,c_A>0$, the maxima in both posteriors are unique. Attaining both risks then requires recovering the two MAP answers on every positive-probability observation. At $P=0$ these answers are $(Y_2-Y_1,Y_1)$, covering all 16 input pairs. Thus four bits remain necessary for joint attainment of the marginal Bayes risks under positive reliability, even when the risks themselves exceed zero.

For $e=0.06$, $e_p=0.03$, $d=0.12$, holding $h+L=8$ yields:

| $(h,L)$ | Relative-label error | First-symbol error |
|---|---:|---:|
| $(0,8)$ | 11.520% | 58.581% |
| $(4,4)$ | 43.395% | 58.581% |
| $(8,0)$ | 59.265% | 58.581% |

At equal age of the first symbol, timing the second symbol differently changes the relative-phase risk. At fixed $h$, increasing $L$ has no effect on the already answered relative-phase question. If $0<d<3/4$, as $L$ tends to infinity the first-symbol error approaches $3/4$. With $d=3/4$ a single intervening step makes the pilot independent of the initial phase. With $d=0$, the phase remains fixed for any interval.

## Reproduction

Run `python3 verify.py` from this directory. The output is saved in [results.csv](results.csv). The standard-library program builds the phase transition probabilities by rational convolution, enumerates all symbol, phase, offset and error states, and makes each Bayes decision from the resulting observation table. It checks the derived risks by exact equality and verifies row-by-row that removing $Y_2$ leaves the first-symbol posterior unchanged. Only the displayed decimal results are rounded. All results here concern the synthetic four-state phase process.
