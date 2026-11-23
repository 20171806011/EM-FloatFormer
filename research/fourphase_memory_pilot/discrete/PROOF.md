# Exact cache tradeoff with quadrant noise and a drifting reference

Two independently uniform symbols and a uniform common rotation take values in the additive group of integers modulo four. The data observations are

$$Y_i=X_i+G+E_i,\qquad P=G+D_L+E_p,\qquad D_L=\sum_{j=1}^L V_j.$$

The data pair shares a rotation. The reference arrives after the pair and the rotation then has accumulated $L$ independent increments. Each data error, pilot error and increment is mutually independent and independent of the symbols and initial rotation. Their respective laws are $K_{e_1},K_{e_2},K_{e_p},K_d$, where $K_a(0)=1-a$ and $K_a(1)=K_a(2)=K_a(3)=a/3$. All four parameters lie in $[0,3/4]$. These categorical observations model quadrant errors. The step parameter $d$ describes symmetric slips, including jumps by two quadrants.

A cache is formed from the pair before any query and has at most $M$ states. The first decoder uses only this cache to estimate $R=X_2-X_1$. The second uses that same cache and the newly received unit-pilot observation $P$ to estimate $X_1$. Both scores are per-query probabilities of an exact symbol decision. No waveform rereading or separate state is available. Results below concern marginal scores, which do not require independent query errors.

## Reduction to a 4 by 4 grid

Write $U=Y_2-Y_1$, $V=Y_1$ and $B=D_L+E_p$. Conditional on any errors of positive probability, $(X_1,X_2,G)\mapsto(U,V,P)$ is a bijection. Inverting gives $G=P-B$, $X_1=V-P+B-E_1$ and $R=U+E_1-E_2$. Thus $U,V,P$ are independent uniform variables, jointly independent of the errors, even though $X_1$ and $R$ share the noise $E_1$. The cache sees exactly $(U,V)$. Before the pilot, even the uncompressed observation has absolute-symbol success $1/4$.

Write $t(a)=1-4a/3$. The law $K_a=t(a)\delta_0+(1-t(a))\mathrm{Unif}$ is invariant under sign reversal. In convolution, its coefficient on the point mass multiplies because the uniform law absorbs convolution with any probability law on the group. Consequently, the relative and absolute error residuals have coefficients

$$k_R=t(e_1)t(e_2),\qquad k_A=t(e_1)t(e_p)t(d)^L.$$

Use $t(d)^0=1$ even at $d=3/4$. For a fixed encoder let $h_U=\sum_z \max_u \Pr(U=u,Z=z)$ and similarly define $h_V$. The optimal first and second success probabilities are

$$S_R=(1-k_R)/4+k_R h_U,\qquad S_A=(1-k_A)/4+k_A h_V.$$

Indeed, conditional on $z$, the relative posterior is the uniform law mixed with the posterior of $U$. Conditional on $(z,p)$, the absolute posterior is the uniform law mixed with the posterior of $V$ shifted by $-p$. The mixing weights above are nonnegative, so a modal estimate of the corresponding grid coordinate maximizes each posterior. In particular no Bayes-optimal decoding advantage is omitted by using the pilot only to translate the absolute label. Distinct point estimates may be equally good at ties.

## Exact budget frontier for four through sixteen cache states

For $4\le M\le16$,

$$h_U\le1,\qquad h_V\le1,\qquad h_U+h_V\le 1+M/16.$$

For a deterministic frontier code the top edge consists of the pairs

$$(h_U,h_V)=((M+j)/16,(16-j)/16),\qquad j=0,\ldots,16-M.$$

A randomized encoder can attain every intermediate point. If $k_R,k_A>0$, the affine images of these points are exactly the respective deterministic and randomized Pareto frontiers. If either coefficient vanishes, its query has success $1/4$ for every representation. The other query's full-observation optimum can still be attained with four states. If both vanish every representation gives $(1/4,1/4)$.

**Upper bound.** A modal pair for each cache state is a point in the observed $(U,V)$ grid. Correctness of both coordinate guesses forces the transmitted point to equal that state's modal pair. A uniform grid point belongs to this set of at most $M$ pairs with probability at most $M/16$, also for a stochastic encoder. The probability of getting at least one coordinate correct is at most one. Adding these two bounds gives the stated sum bound for the latent grid coordinates. Independent shared randomness can be conditioned on and cannot break the bound.

**Construction.** Mark all four diagonal cells and any $M-4$ additional distinct cells. Each marked cell sends its own index. Of the remaining $16-M$ cells, send $j$ to any marked cell in the same row and the rest to any marked cell in the same column. All unmarked cells thereby recover exactly one coordinate when decoded at their marked center. Every row and column has a center. This assignment yields the stated pair, and the sum bound implies that switching to modal guesses cannot improve either coordinate. Splitting any one unmarked cell probabilistically between its row-center and column-center fills the line segment with the same decoder. Deterministic successes have denominator 16. Every deterministic score below the edge is dominated by at least one such boundary point, by the three inequalities above and $h_U,h_V\ge1/4$. This proves tightness.

## Predictions with optimal first-query accuracy

For $k_R>0$, attaining the first query's full-observation optimum requires preserving the observed relative quadrant and fixes $h_U=1$. Among such codes, the optimal absolute accuracy is attained at $h_V=M/16$. For a two-, three- or four-bit cache ($M=4,8,16$) the absolute accuracy gain from revealing the pilot is respectively $0$, $k_A/4$, or $3k_A/4$. Under independent pilot rotation ($e_p=3/4$ suffices) even the complete categorical pair yields absolute error $3/4$. With $0<d<3/4$ and an informative data/pilot pair, the recoverable gain decays exactly as $t(d)^L$. For zero drift it remains constant. Cache size four bits recovers all information in these categorical observations.

At $e_1=e_2=0.10$, $e_p=0.05$, $d=0.06$, relative symbol error at the selected endpoint stays $0.186667$. Corresponding absolute symbol errors are

| Delay in increments | Two bits | Three bits | Four bits |
|---:|---:|---:|---:|
| 0 | 0.750000 | 0.547778 | 0.143333 |
| 1 | 0.750000 | 0.563956 | 0.191867 |
| 4 | 0.750000 | 0.605129 | 0.315388 |
| 16 | 0.750000 | 0.696736 | 0.590208 |
| 64 | 0.750000 | 0.749027 | 0.747080 |

These numerical examples use equal data-error rates. The proof permits unequal rates. The fixed relative error as the gap grows follows from putting all drift *after* both observations. If a further independent symmetric increment $W$ with law $K_b$ is inserted between them, with $Y_2=X_2+G+W+E_2$ and $P=G+W+D_L+E_p$, the conditional-bijection argument still applies. Both $k_R$ and $k_A$ acquire the factor $t(b)$. The table uses $b=0$.

## Replay

From this directory run

```sh
python3 verify.py > replay.csv
diff -u risks.csv replay.csv
```

For each scenario and gap the script builds the pilot-offset distribution by circular convolution, then enumerates all six underlying categorical variables $(X_1,X_2,G,E_1,E_2,B)$ with integer probability weights. For each cache it sums over hidden states before maximizing each decoder's accessible posterior. It checks equality to the formulas with exact rational arithmetic; only the displayed CSV error columns are rounded. The file includes all 207 tested allocations across nine settings, including unequal data noise, complete data noise, independent reference and coherent noiseless reception. The test verifies the constructions numerically; the converse above covers all encoders.
