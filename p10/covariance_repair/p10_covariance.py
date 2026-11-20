"""Development diagnostic on the pinned EM-FloatFormer P9 waveform."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from p8.coding import decode, generator, parity_check
from p8.link import clock
from p8c.link import PILOTS
from p9.channel import readout_frame_b, sample_moments_b, simulate_link, track_frame_a
from p9.contract import CORE_ROWS, DELAYS, FRAME_DURATION, N, PROBE_ROW, T
from p9.gating import build_candidates, gate_jeffreys_student_t


AUX = (2, 5, 8, 13, 16)
TARGET = [r for r in CORE_ROWS if r not in AUX]
TARGET_INFO = [CORE_ROWS.index(r) for r in TARGET]
DELTAS = [sign * d for sign in (-1, 1) for d in (.025, .035, .045, .055, .065, .075)]
BANK = [(-1, 0.)] + [(r, d) for r in range(3) for d in DELTAS]
CLASS = np.array([r for r, d in BANK])
LOG_PRIOR = np.log([.5] + [.5 / (len(BANK) - 1)] * (len(BANK) - 1))
TIMES = clock(T, start_row=T) - FRAME_DURATION
SEED0 = 960261120000
REGIMES = ('partial_shift', 'continuous')
METHODS = ('pilot', 'diagonal', 'covariance', 'p9_classic')


def gf2_rank(matrix):
    a = matrix.copy()
    rank = 0
    for column in range(a.shape[1]):
        candidates = np.flatnonzero(a[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        a[[pivot, rank]] = a[[rank, pivot]]
        eliminate = np.flatnonzero(a[rank + 1:, column]) + rank + 1
        a[eliminate] ^= a[rank]
        rank += 1
    return rank


def verify_code():
    g = generator()
    groups = [list(range(i, i + 4)) for i in range(0, 128, 4)]
    ranks1 = [gf2_rank(g[:, a]) for a in groups]
    ranks2 = [gf2_rank(g[:, a + b]) for i, a in enumerate(groups) for b in groups[:i]]
    sign_mask = np.tile([1, 0, 1, 0], 32).astype(np.uint8)
    return dict(single_full_rank=ranks1.count(4), pair_full_rank=ranks2.count(8),
                sign_mask_syndrome_weight=int(np.sum((parity_check() @ sign_mask) % 2)))


def predict(obs, *, return_inputs=False):
    mean, var = sample_moments_b(track_frame_a(obs.y_a, obs.n0))
    bank = np.broadcast_to(mean, (len(BANK), *mean.shape)).copy()
    for j, (ray, delta) in enumerate(BANK[1:], 1):
        bank[j, ray] *= np.exp(2j * np.pi * delta * TIMES)
    assert len(obs.training_rows) == 1 and np.isnan(obs.y_b[PROBE_ROW]).all()
    rows = [obs.training_rows[0], *AUX, PROBE_ROW]
    n = np.arange(N)
    matrices = np.zeros((len(BANK), len(rows), N, N), complex)
    for ray, delay in enumerate(DELAYS):
        matrices[:, :, n, (n-delay) % N] = bank[:, ray, rows, :]

    # Diagonal approximation to uncertainty of the A tracker. A pilot score
    # changes only candidate weights in this diagnostic, not within-candidate gains.
    noise = obs.n0 + var[:, rows].sum(axis=0)
    pilot_pred = matrices[:, 0] @ PILOTS[rows[0]]
    pilot_score = LOG_PRIOR - (abs(obs.y_b[rows[0]] - pilot_pred)**2 / noise[0]).sum(axis=-1)

    payload_matrices = matrices[:, 1:-1]
    cov = payload_matrices @ payload_matrices.conj().swapaxes(-1, -2)
    cov[:, :, n, n] += noise[None, 1:-1]
    z = np.broadcast_to(obs.y_b[list(AUX)], (len(BANK), len(AUX), N))
    solution = np.linalg.solve(cov, z[..., None])[..., 0]
    sign, logdet = np.linalg.slogdet(cov)
    assert np.allclose(sign, 1.)
    scores = -(z.conj() * solution).sum(axis=-1).real - logdet
    diagonal = cov.diagonal(axis1=-2, axis2=-1).real
    diagonal_scores = -(abs(z)**2 / diagonal + np.log(diagonal)).sum(axis=-1)
    weights = [softmax(s) for s in (pilot_score, pilot_score + diagonal_scores.sum(axis=1),
                                    pilot_score + scores.sum(axis=1))]
    np.testing.assert_allclose(weights[0], weights[1], rtol=1e-10, atol=1e-12)
    probe_by_hypothesis = matrices[:, -1] @ PILOTS[PROBE_ROW]
    predictions = {}
    for name, w in zip(METHODS, weights):
        mixed = np.einsum('j,jrtn->rtn', w, bank)
        # Phase-only transforms leave each within-branch variance and second moment unchanged.
        mixed_var = var + np.maximum(abs(mean)**2 - abs(mixed)**2, 0.)
        if name == 'diagonal':
            bits = predictions['pilot']['bits']
        else:
            llrs, _, _ = readout_frame_b(mixed, mixed_var, obs.y_b, obs.training_rows, obs.n0)
            bits = decode(llrs[TARGET].reshape(len(TARGET), 128))
        masses = [w[CLASS == r].sum() for r in (-1, 0, 1, 2)]
        predictions[name] = dict(probe=w @ probe_by_hypothesis, bits=bits,
                                 chosen=int(np.argmax(masses)) - 1)
    classic = gate_jeffreys_student_t(obs, *build_candidates(obs))
    predictions['p9_classic'] = dict(probe=classic.mix_probe_pred,
                                     bits=decode(classic.mix_llrs[TARGET].reshape(len(TARGET), 128)),
                                     chosen=None)
    if return_inputs:
        return predictions, weights, classic
    return predictions


def score(regime, seed):
    obs, truth = simulate_link(seed, 9., regime, 'sparse')
    result = predict(obs)  # receiver-facing call has no hidden channel, class, symbols or bits
    truth_class = -1 if truth.changed_ray is None else truth.changed_ray
    probe = truth.y_b_probe_clean
    row = dict(seed=seed, regime=regime, energy=float(np.sum(abs(probe)**2)), target_count=len(TARGET))
    for name, model in result.items():
        errors = (model['bits'] != truth.b_info[TARGET_INFO]).any(axis=1)
        row[name] = dict(probe_sse=float(np.sum(abs(model['probe'] - probe)**2)),
                         correct_class=None if model['chosen'] is None else model['chosen'] == truth_class,
                         block_errors=errors.astype(int).tolist())
    return row


def summarize(records):
    summary = {}
    for reg in REGIMES:
        rows = [r for r in records if r['regime'] == reg]
        total_energy = sum(r['energy'] for r in rows)
        stats = {}
        for name in METHODS:
            m = [r[name] for r in rows]
            errors = np.array([v['block_errors'] for v in m])
            stats[name] = dict(nmse=sum(v['probe_sse'] for v in m) / total_energy,
                               correct=sum(v['correct_class'] for v in m) if name != 'p9_classic' else None,
                               block_errors=int(errors.sum()), bler=float(errors.mean()))
        boot = np.random.default_rng(128).integers(len(rows), size=(4000, len(rows)))
        def per_link(field, name):
            return np.array([r[name][field] for r in rows])
        intervals = {}
        for other in ('pilot','p9_classic'):
            delta = per_link('probe_sse', other) - per_link('probe_sse', 'covariance')
            nmse_diff = delta[boot].sum(1) / np.array([r['energy'] for r in rows])[boot].sum(1)
            bler_diff = (per_link('block_errors', other) - per_link('block_errors', 'covariance')).mean(1)
            intervals[other] = dict(nmse=np.quantile(nmse_diff,[.025,.975]).tolist(),
                                    bler=np.quantile(bler_diff[boot].mean(1),[.025,.975]).tolist())
        summary[reg] = dict(links=len(rows), target_blocks=len(rows)*len(TARGET), methods=stats,
                            bootstrap_95_reference_minus_covariance=intervals)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--links', type=int, default=64)
    p.add_argument('--out', type=Path, default=Path('p10_cov_results.json'))
    args = p.parse_args()
    audit = verify_code()
    assert audit == dict(single_full_rank=32, pair_full_rank=496, sign_mask_syndrome_weight=32)
    records = [score(regime, SEED0+i) for regime in REGIMES for i in range(args.links)]
    summary = summarize(records)
    args.out.write_text(json.dumps(dict(code_audit=audit, summary=summary, records=records), indent=2)+'\n')
    print(json.dumps(dict(code_audit=audit, summary=summary), indent=2))
