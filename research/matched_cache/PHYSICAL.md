# Fixed-budget hand encoder

The frozen three-sample window has unit symbol energy and complex noise variance $1/\rho$, with $\rho=10^{0.2}$. For a four-phase received symbol $U$ uniform on the four roots of unity, a sufficient statistic is $T=(Y_1+Y_2+Y_3)/\sqrt 3$. Rotating $T$ by $\pi/4$ puts the four means at $(\pm\mu,\pm\mu)$, with $\mu=1/\sqrt 2$. The signs are independent and equiprobable. Conditional on each sign the observed coordinate is Gaussian with variance $\sigma^2=1/(2\rho)$.

For either coordinate $t$, Bayes' rule gives $p(+\mid t)=1/(1+\exp(-2\mu t/\sigma^2))$. Divide this probability into $J=2$ equal-width bins for a four-bit pair, or $J=4$ bins for an eight-bit pair. The corresponding cuts on $t$ are $(\sigma^2/2\mu)\log(j/(J-j))$, $1\le j<J$. For real and imaginary bin indices $r,s$, the stored window state is the integer $Jr+s$. A window has $J^2$ states, and two windows together have $J^4$ states, exactly $2^4$ or $2^8$. Only the pair of state indices survives until reference reveal.

For a bin $[a,b)$, put $w_\pm=\Phi((b\mp\mu)/\sigma)-\Phi((a\mp\mu)/\sigma)$. The decoded sign probability is $w_+/(w_++w_-)$. Multiplying the two decoded sign probabilities or their complements gives the four probabilities of $U$, in root order $1,i,-1,-i$. These probabilities integrate the full observation likelihood over the cell under the frozen simulator, so the readout is Bayes-optimal for this particular partition. The protocol compares it to independently fitted partitions.

Multiplication by $i$ maps the coordinates to $(-t_I,t_R)$, permuting the labels by one step. Off bin boundaries this maps bin indices $(r,s)$ to $(J-1-s,r)$. The code uses symmetric cuts and reflected sign probabilities. A sample exactly on a cut goes to the upper bin. Exact ties form a probability-zero set under the continuous model, including the special all-zero sample. Uniform equivariance on ties is not claimed.

Run `python physical.py` to verify that every code occurs on 4,096 independently generated windows, check the probability-vector rotation for each window, and verify all cell centroids by independent numerical integration of the full-posterior expectation.
