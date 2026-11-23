"""QPSK waveform, bounded caches, and a delayed noisy pilot. Python 3 + NumPy."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOTS = np.array([1, 1j, -1, -1j])
Z95 = 1.959963984540054


def posterior_iq(y, snr):
    log_weight = 2 * snr * (y[:, None] * ROOTS.conj()).real
    weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
    return weight / weight.sum(axis=1, keepdims=True)


def posterior_pilot(p, snr, variance, grid):
    if variance == 0:
        return posterior_iq(p, snr)
    if np.isinf(variance):
        return np.full((len(p), 4), 0.25)
    # Positive quadrature on a uniform circular mesh. Wrap Gaussian tails to 10 sd.
    phi = (2 * np.pi / grid) * np.arange(grid)
    displacement = (phi[:, None] - np.angle(ROOTS) + np.pi) % (2 * np.pi) - np.pi
    images = math.ceil((10 * math.sqrt(variance) + np.pi) / (2 * np.pi))
    prior = sum(np.exp(-(displacement + k * 2 * np.pi) ** 2 / (2 * variance))
                for k in range(-images, images + 1))
    prior /= prior.sum(axis=0)
    directions = np.exp(-1j * phi)
    posterior = np.empty((len(p), 4))
    for start in range(0, len(p), 768):
        batch = p[start:start + 768]
        log_lik = 2 * snr * (batch[:, None] * directions).real
        lik = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))
        weights = lik @ prior
        posterior[start:start + len(batch)] = weights / weights.sum(axis=1, keepdims=True)
    return posterior


def difference_posterior(a, b):
    """Independent circular variables A, B. Return Pr(B-A=k) for k=0,1,2,3."""
    return np.stack([(a * np.roll(b, -k, axis=1)).sum(axis=1) for k in range(4)], axis=1)


def hard_posterior(y, snr):
    hard = np.argmax((y[:, None] * ROOTS.conj()).real, axis=1)
    q = math.erfc(math.sqrt(snr / 2)) / 2
    trans = np.array([(1 - q)**2, q*(1-q), q**2, q*(1-q)])
    full = trans[(hard[:, None] - np.arange(4)) % 4]
    first = 2 * (hard // 2)
    half = (trans[(first[:, None] - np.arange(4)) % 4] +
            trans[(first[:, None] + 1 - np.arange(4)) % 4]) / 2
    return full, half


def check_latent_enumeration(a, b, pg):
    # Independently accumulate Pr(X1=x1,X2=x2,G=g|Y1,Y2,P).
    r = np.zeros_like(a)
    x = np.zeros_like(a)
    for x1 in range(4):
        for x2 in range(4):
            for g in range(4):
                w = a[:, (x1 + g) % 4] * b[:, (x2 + g) % 4] * pg[:, g]
                r[:, (x2-x1) % 4] += w
                x[:, x1] += w
    np.testing.assert_allclose(r, difference_posterior(a, b), atol=2e-14)
    np.testing.assert_allclose(x, difference_posterior(pg, a), atol=2e-14)


def err_interval(errors):
    n = len(errors)
    p = np.mean(errors)
    center = (p + Z95**2 / (2*n)) / (1 + Z95**2 / n)
    width = Z95 * math.sqrt(p*(1-p)/n + Z95**2/(4*n*n)) / (1 + Z95**2/n)
    return [p, max(0., center-width), min(1., center+width)]


def experiment(n, seed, grid):
    rng = np.random.default_rng(seed)
    def standard_noise():
        return (rng.normal(size=n) + 1j*rng.normal(size=n)) / np.sqrt(2)
    x1, x2, g = rng.integers(4, size=(3,n))
    n1, n2, npilot = standard_noise(), standard_noise(), standard_noise()
    drift_draw = rng.normal(size=n)
    null_theta = rng.uniform(-np.pi,np.pi,size=n)
    rows = []
    for db in [2, 8]:
        snr = 10**(db/10)
        y1 = ROOTS[(x1+g)%4] + n1/np.sqrt(snr)
        y2 = ROOTS[(x2+g)%4] + n2/np.sqrt(snr)
        a, b = posterior_iq(y1,snr), posterior_iq(y2,snr)
        rprob = difference_posterior(a,b)
        rhat = rprob.argmax(axis=1)
        rel_errors = rhat != (x2-x1)%4
        a_hard, a_half = hard_posterior(y1,snr)
        # Verify encoder equivariance used to establish independence of the R cache.
        rotated_r = difference_posterior(a, posterior_iq(1j*y2,snr)).argmax(axis=1)
        np.testing.assert_array_equal(rotated_r, (rhat+1)%4)
        for gap in [0, 1, 4, 16, 64, 'independent']:
            variance = np.inf if gap == 'independent' else gap * 0.25**2
            theta = null_theta if np.isinf(variance) else np.sqrt(variance)*drift_draw
            p = ROOTS[g]*np.exp(1j*theta) + npilot/np.sqrt(snr)
            pg = posterior_pilot(p,snr,variance,grid)
            ng = min(n,2048)
            pg_fine = posterior_pilot(p[:ng],snr,variance,2*grid)
            grid_delta = np.abs(pg[:ng]-pg_fine).max()
            assert grid_delta < 2e-8, (grid, db, gap, grid_delta)
            check_latent_enumeration(a[:16], b[:16], pg[:16])
            # On the null control all absolute actions tie. Fix action 0.
            if np.isinf(variance):
                full = hard = half = np.zeros(n, dtype=int)
            else:
                full = difference_posterior(pg,a).argmax(axis=1)
                hard = difference_posterior(pg,a_hard).argmax(axis=1)
                half = difference_posterior(pg,a_half).argmax(axis=1)
            ef, eh = full != x1, hard != x1
            d = eh.astype(float) - ef.astype(float)
            gain = d.mean()
            hw = Z95*d.std(ddof=1)/np.sqrt(n)
            rows.append([db, gap, n, *err_interval(rel_errors),
                         *err_interval(ef), *err_interval(eh), 0.75,
                         gain, gain-hw, gain+hw, grid_delta, *err_interval(half != x1)])
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--grid', type=int, default=128)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('results.csv'))
    args = ap.parse_args()
    if args.samples < 2 or args.grid < 32 or args.grid % 4:
        ap.error('use >=2 samples and a multiple-of-four grid of at least 32')
    rows = experiment(args.samples, args.seed, args.grid)
    with args.output.open('w', newline='') as f:
        out = csv.writer(f, lineterminator='\n')
        out.writerow(['snr_db','gap','n','rel_map_ser','rel_low','rel_high',
                      'x1_full_ser','full_low','full_high','x1_4bit_ser','4bit_low','4bit_high',
                      'x1_relonly_exact_ser','4bit_minus_full_ser','paired_low','paired_high',
                      'max_posterior_mesh_change_first_2048','x1_3bit_ser','3bit_low','3bit_high'])
        for row in rows:
            out.writerow([format(x,'.12g') if isinstance(x,(float,np.floating)) else x for x in row])
