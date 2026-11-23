"""Finite-cache decisions scored by conditional Bayes regret."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'continuous'))
from run import ROOTS, difference_posterior, posterior_iq, posterior_pilot


def midpoints(p, cells):
    index = np.minimum(cells - 1, np.floor(p * cells).astype(int))
    return (index + 0.5) / cells


def product_labels(real_plus, imag_plus):
    # Label order after rotating by +pi/4: (++), (-+), (--), (+-).
    return np.stack([real_plus * imag_plus, (1-real_plus) * imag_plus,
                     (1-real_plus) * (1-imag_plus), real_plus * (1-imag_plus)], axis=1)


def experiment(n, seed):
    rng = np.random.default_rng(seed)
    x1,x2,g = rng.integers(4, size=(3,n))
    def complex_noise():
        return (rng.normal(size=n) + 1j*rng.normal(size=n)) / np.sqrt(2)
    n1, n2, npilot = complex_noise(), complex_noise(), complex_noise()
    drift = 0.5 * rng.normal(size=n)  # L=4, sigma=0.25 radians per step.
    for db in [2, 8]:
        snr = 10**(db/10)
        y1 = ROOTS[(x1+g) % 4] + n1/np.sqrt(snr)
        y2 = ROOTS[(x2+g) % 4] + n2/np.sqrt(snr)
        p = ROOTS[g]*np.exp(1j*drift) + npilot/np.sqrt(snr)
        a1, a2 = posterior_iq(y1,snr), posterior_iq(y2,snr)
        pg = posterior_pilot(p,snr,0.25,128)
        h = difference_posterior(a1,a2).argmax(axis=1)
        rel_ser = np.mean(h != (x2-x1) % 4)
        real_plus = a1[:,0] + a1[:,3]
        imag_plus = a1[:,0] + a1[:,1]
        np.testing.assert_allclose(a1, product_labels(real_plus,imag_plus), atol=2e-15)
        exact = difference_posterior(pg,a1)
        best = exact.max(axis=1)
        full_risk = (1-best).mean()
        for bits_per_coord in range(1,6):
            cells = 2**bits_per_coord
            approx_a = product_labels(midpoints(real_plus,cells), midpoints(imag_plus,cells))
            approx = difference_posterior(pg,approx_a)
            guess = approx.argmax(axis=1)
            regret = best - exact[np.arange(n), guess]
            assert regret.min() >= 0
            assert regret.max() <= 2/cells + 1e-14
            assert np.max(np.abs(exact-approx)) <= 1/cells + 1e-14
            yield [db, 4, n, seed, bits_per_coord, 2+2*bits_per_coord, rel_ser, full_risk,
                   full_risk+regret.mean(), regret.mean(), regret.std(ddof=1)/np.sqrt(n),
                   regret.max(), 2/cells]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=124479)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.csv'))
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('use at least two samples')
    with args.output.open('w', newline='') as f:
        out = csv.writer(f,lineterminator='\n')
        out.writerow(['snr_db','gap','n','seed','bits_per_sign_probability','total_cache_bits',
                      'realized_relative_ser','full_conditional_risk_mean','cache_conditional_risk_mean',
                      'conditional_regret_mean','mean_standard_error','sample_max_regret','proven_bound'])
        for row in experiment(args.samples,args.seed):
            out.writerow([format(x,'.12g') if isinstance(x,(float,np.floating)) else x for x in row])
