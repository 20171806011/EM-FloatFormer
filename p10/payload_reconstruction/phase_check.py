"""Evaluator-only error metrics and identities for the P9 sparse B channel.

The receiver must not import hidden states. These routines may be called with
truth.h_b_true only AFTER producing receiver estimates from public observations.
"""
import json
from itertools import combinations
import numpy as np


def _weighted_angle_sums(estimate, truth, weight, scale=1.0):
    """Weighted squared circular error; missing estimate is charged pi where truth has positive weight."""
    angle = np.angle(estimate * truth.conj())
    angle = np.where(abs(estimate) > 0, angle, np.pi)
    return float(np.sum(weight * (angle / scale)**2)), float(np.sum(weight))


def _measurement(estimate, truth, duration):
    weight = abs(truth)**2
    phase = _weighted_angle_sums(estimate, truth, weight)
    est_advance = estimate[..., -1] * estimate[..., 0].conj()
    true_advance = truth[..., -1] * truth[..., 0].conj()
    freq = _weighted_angle_sums(est_advance, true_advance, abs(true_advance), 2*np.pi*duration)
    return phase, freq


def metrics(mu, true_gains, rows=None):
    """Flattened sufficient counts, errors and weights. Sum numerators/denominators across links.

    absolute_phase and rel_phase are per-sample phase RMSE in radians. Each
    pair i<j is weighted by |g_i*g_j|^2, each single ray by |g_i|^2.
    *_freq = circular frequency RMSE in cycles/N samples. It compares phase
    advance between first and last retained sample of EACH row, no inter-row
    gap, wrapped modulo N/(N-1) cycles/N. For constant chirp this estimates
    the physical instantaneous frequency at the row midpoint, modulo that rate.
    The *_rel_* metrics exactly cancel a simultaneous time-dependent phase on
    all predicted rays, even one that varies arbitrarily from sample to sample.
    """
    a = np.asarray(mu, dtype=complex)
    b = np.asarray(true_gains, dtype=complex)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 3 or a.shape[2] < 2:
        raise ValueError('mu and true_gains must be [3, number_of_rows, N>=2]')
    if rows is not None:
        a, b = a[:, list(rows), :], b[:, list(rows), :]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.shape[1] == 0:
        raise ValueError('Empty rows or nonfinite channel')
    duration = (a.shape[-1] - 1)/a.shape[-1]
    rel_a = np.stack([a[i]*a[j].conj() for i,j in combinations(range(3),2)])
    rel_b = np.stack([b[i]*b[j].conj() for i,j in combinations(range(3),2)])
    pairs = dict(zip(('abs_phase','abs_freq'),_measurement(a,b,duration)))
    pairs.update(zip(('rel_phase','rel_freq'),_measurement(rel_a,rel_b,duration)))
    out = {}
    for k, (sse, w) in pairs.items():
        out[k+'_sse'], out[k+'_weight'] = sse, w
        out[k+'_rmse'] = float(np.sqrt(sse/w)) if w else None
    return out


def self_test():
    from p8.link import clock
    from p8c.link import PILOTS
    from p9.channel import build_sc_channel_matrices, simulate_link
    from p9.contract import N, T, DELAYS

    obs, truth = simulate_link(981120000051,9.,'full_reset','sparse')
    g = truth.h_b_true
    rows = [0,2,5,8,10,13,16,19]
    h = build_sc_channel_matrices(g, rows)
    c = h @ h.conj().swapaxes(-1,-2)
    n = np.arange(N)
    all_lag_error = 0.
    expected_pairs = {0:[(0,0),(1,1),(2,2)], 2:[(1,0),(2,1)], 4:[(2,0)],
                      N-2:[(0,1),(1,2)],N-4:[(0,2)]}
    for k in range(N):
        pairs=[(i,j) for i,di in enumerate(DELAYS) for j,dj in enumerate(DELAYS) if (di-dj)%N == k]
        assert pairs == expected_pairs.get(k,[])
        pred = np.zeros((len(rows),N), complex)
        for i,j in pairs:
            pred += g[i,rows,:]*np.roll(g[j,rows,:],k,axis=-1).conj()
        all_lag_error = max(all_lag_error, float(np.max(abs(c[:,n,(n-k)%N]-pred))))
    assert all_lag_error < 1e-12
    # More than a global phase: an arbitrary row-wide phase, set to one on
    # the only pilot, gives identical within-row population covariance and
    # identical noiseless pilot on an unrestricted slow-time channel family.
    # This example does not assert the transformed channel remains a P9 chirp.
    row_phase = np.exp(1j*np.linspace(-1.,1.,T))
    pilot_row = obs.training_rows[0]
    row_phase /= row_phase[pilot_row]
    alt = build_sc_channel_matrices(g * row_phase[None,:,None], rows)
    idx=rows.index(pilot_row)
    amb_cov = float(np.max(abs(c - alt @ alt.conj().swapaxes(-1,-2))))
    amb_pilot = float(np.max(abs(h[idx] @ PILOTS[pilot_row] - alt[idx] @ PILOTS[pilot_row])))
    off_pilot = float(np.max(abs(h[0]-alt[0])))
    assert amb_cov < 1e-12 and amb_pilot < 1e-12 and off_pilot > 1e-3
    # For a genuine common frequency shift, the within-row sample phases
    # are NOT equal. Covariance transforms by D C D^H. The difference of n and
    # its cyclic partner must use signed sample clocks, not always the lag k.
    times = clock(T,start_row=T)
    df=.03
    d=np.exp(2j*np.pi*df*(times - times[pilot_row].mean()))
    cross=build_sc_channel_matrices(g*d[None,:,:], rows)
    actual = cross @ cross.conj().swapaxes(-1,-2)
    pred = d[rows,:,None]*c*d[rows,None,:].conj()
    doppler_law=float(np.max(abs(actual-pred)))
    delta_fraction=float(np.linalg.norm(actual-c)/np.linalg.norm(c))
    assert doppler_law < 1e-12 and delta_fraction>1e-4
    zero=metrics(g,g)
    assert all(zero[k+'_rmse']<1e-12 for k in ('rel_freq','abs_freq','rel_phase','abs_phase'))
    common=metrics(g*d[None,:,:],g,rows)
    assert common['rel_phase_rmse']<1e-12 and common['rel_freq_rmse']<1e-12
    assert abs(common['abs_freq_rmse']-df)<1e-12
    return dict(all_lag_maxerr=all_lag_error, same_cov_maxerr=amb_cov,
                same_pilot_maxerr=amb_pilot, distinct_channel_max=off_pilot,
                physical_common_doppler_cov_law_maxerr=doppler_law,
                physical_common_doppler_cov_change_fraction=delta_fraction,
                imposed_common_freq=df,recovered_abs_drift_rmse=common['abs_freq_rmse'])


if __name__=='__main__':
    print(json.dumps(self_test(),indent=2))
