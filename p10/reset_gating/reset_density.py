"""Predictive pilot density for an independent P9 channel reset.

The dominant ray has a fixed modulus and uniform phase. The other two complex
gains are Gaussian. Only the frequency/chirp integral is numerical, using a
fixed scrambled Sobol net under the draw_channel distribution. No old-state
estimate or unknown received data is used.
"""
from functools import lru_cache
import numpy as np
from scipy.special import i0e, logsumexp
from scipy.stats import qmc

from p8.link import clock, DELAYS, LINK_DURATION, N, POWERS, T
from p8c.link import PILOTS

SOBOL_SEED = 63268
M = 8  # 2**8 = 256 equally weighted frequency-curve draws


def _atoms(rows, m, scramble_seed):
    u = qmc.Sobol(d=9, scramble=True, seed=scramble_seed).random_base2(m)
    signs = np.where(u[:, :3] < 0.5, -1., 1.)
    start = signs * (0.045 + 0.035 * u[:, 3:6])
    end = signs * (0.045 + 0.035 * u[:, 6:9])
    chirp = (end - start) / LINK_DURATION
    times = clock(T, start_row=T)[list(rows)].ravel()
    pilots = PILOTS[list(rows)]
    delayed = np.stack([np.roll(pilots, int(d), axis=1).ravel() for d in DELAYS])
    phase = start[:, :, None] * times + .5 * chirp[:, :, None] * times**2
    return (np.sqrt(POWERS)[None, :, None] * delayed * np.exp(2j*np.pi*phase)).swapaxes(1, 2)


@lru_cache(maxsize=5)
def _cache(rows, n0, m, scramble_seed):
    atom = _atoms(rows, m, scramble_seed)
    dominant = atom[:, :, 0]
    weak = atom[:, :, 1:]
    weak_h = weak.conj().swapaxes(-1,-2)
    small = weak_h @ weak + n0 * np.eye(2)
    inv_small = np.linalg.inv(small)
    dim = dominant.shape[-1]
    logdet = np.linalg.slogdet(small)[1] + (dim-2)*np.log(n0)
    projected = (weak_h @ dominant[...,None])
    inv_dom = (dominant - (weak @ (inv_small @ projected))[...,0]) / n0
    dom_energy = (dominant.conj() * inv_dom).sum(axis=1).real
    const = -dim*np.log(np.pi) - logdet - dom_energy
    return weak_h, inv_small, inv_dom.conj(), const


def _scores(y, rows, n0, m=M, scramble_seed=SOBOL_SEED):
    weak_h, inv_small, match, const = _cache(rows, n0, m, scramble_seed)
    proj = weak_h @ y
    recovered = (proj.conj() * (inv_small @ proj[...,None])[...,0]).sum(axis=1).real
    residual = (np.vdot(y, y).real - recovered) / n0
    radius = 2*np.abs(match @ y)
    return const - residual + radius + np.log(i0e(radius))


def reset_logpdf(obs):
    """Absolute complex log density on the anchor pilot, including constants."""
    rows = tuple(obs.training_rows)
    if rows != (10,) or not np.isfinite(obs.n0) or obs.n0 <= 0:
        raise ValueError("Expected the sparse P9 layout and finite positive N0")
    y = np.asarray(obs.y_b[list(rows)]).ravel()
    scores = _scores(y, rows, float(obs.n0))
    return float(logsumexp(scores) - np.log(scores.size))


def self_test():
    """Compare the Woodbury/Bessel expression to a dense phase quadrature."""
    from p9.channel import simulate_link
    obs, _ = simulate_link(962420260003, 9., 'full_reset', 'sparse')
    rows, n0 = tuple(obs.training_rows), float(obs.n0)
    y = obs.y_b[list(rows)].ravel()
    atom = _atoms(rows,M,SOBOL_SEED)
    scores = _scores(y,rows,n0)
    angles = np.exp(2j*np.pi*np.arange(2048)/2048)
    deviations = []
    for k in (0,31,217):
        strong,weak = atom[k,:,0],atom[k,:,1:]
        covariance = n0*np.eye(N) + weak @ weak.conj().T
        residual = y[:,None] - strong[:,None]*angles
        quadratic = (residual.conj() * np.linalg.solve(covariance,residual)).sum(axis=0).real
        phase_integral = logsumexp(-quadratic) - np.log(angles.size)
        expected = -N*np.log(np.pi)-np.linalg.slogdet(covariance)[1] + phase_integral
        deviations.append(abs(expected - scores[k]))
    assert max(deviations) < 1e-9
    physical = reset_logpdf(obs)
    rotating = _scores(y*np.exp(.77j),rows,n0)
    assert np.allclose(scores,rotating,atol=2e-11,rtol=0.)
    return dict(dense_phase_max_error=float(max(deviations)), logpdf=physical)


if __name__ == '__main__':
    print(self_test())
