"""One-shot conditional repair evaluation on a second, disjoint development seed range."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from p8.coding import decode
from p8.tracker import State
from p9.channel import (_training_dictionary, SWEEPS, build_sc_channel_matrices,
                        condition_frame_b, readout_frame_b, sample_moments_b, simulate_link, track_frame_a)
from p9.contract import N
from p10_covariance import AUX, BANK, CLASS, TARGET, TARGET_INFO, TIMES, predict

VALIDATION = (0, 9)  # Additional data words in P9 sparse, disjoint from five selection and eleven target rows.
NEW_SEED0 = 960261121000
CONFIDENCE = .90
MODES = 2


def refit(obs, state, ray, delta):
    rows = obs.training_rows
    design, _, energy = _training_dictionary(state.epoch, rows)
    design = design.copy()
    rotation = np.exp(2j * np.pi * delta * TIMES)
    design[ray] *= rotation[list(rows)].ravel()[None, :]
    conjugate = design.conj()
    y = obs.y_b[list(rows)].ravel()
    noise = obs.n0
    v = state.variance / (1 + state.variance * energy / noise)
    lw = state.log_weight.copy()
    mu = state.mean.copy()
    signal = np.einsum('lh,lhq->lq', np.exp(lw) * mu, design)
    natural_prior = state.mean / state.variance
    base = state.log_weight + np.log(v / state.variance) - abs(state.mean)**2 / state.variance
    for _ in range(SWEEPS):
        for r in range(len(mu)):
            matched = conjugate[r] @ (y - signal.sum(0) + signal[r])
            natural = natural_prior[r] + matched / noise
            mu[r] = v[r] * natural
            evidence = base[r] + v[r] * abs(natural)**2
            lw[r] = evidence - logsumexp(evidence)
            signal[r] = (np.exp(lw[r]) * mu[r]) @ design[r]
    means, variances = sample_moments_b(State(state.epoch, lw, mu, v))
    means[ray] *= rotation
    return means, variances


def validation_score(obs, mu, v):
    h = build_sc_channel_matrices(mu, rows=VALIDATION)
    cov = h @ h.conj().swapaxes(-1, -2)
    n = np.arange(N)
    cov[:, n, n] += obs.n0 + v[:,list(VALIDATION)].sum(0)
    y = obs.y_b[list(VALIDATION)]
    solved = np.linalg.solve(cov, y[...,None])[...,0]
    return float(-np.linalg.slogdet(cov)[1].sum() - np.sum(y.conj()*solved).real)


def select_and_repair(obs, weights, old, base, state):
    candidate_mass = [weights[CLASS == r].sum() for r in range(3)]
    ray = int(np.argmax(candidate_mass))
    if candidate_mass[ray] < CONFIDENCE:
        return base, base, False, False
    ix = np.flatnonzero(CLASS == ray)
    top = ix[np.argsort(weights[ix])[-MODES:]]
    w = weights[top] / weights[top].sum()
    branches = [refit(obs, state, *BANK[j]) for j in top]
    mu = sum(q * m for q, (m, _) in zip(w, branches))
    var = sum(q * (v + abs(m-mu)**2) for q,(m,v) in zip(w, branches))
    if validation_score(obs, mu, var) <= validation_score(obs, old.mix_mean_ray, old.mix_var_ray):
        return base, base, True, False
    llrs, probe, _ = readout_frame_b(mu, var, obs.y_b, obs.training_rows, obs.n0)
    bits = decode(llrs[TARGET].reshape(len(TARGET),128))
    return base, dict(probe=probe, bits=bits), True, True


def evaluate():
    all_records, summary = [], {}
    assert set(VALIDATION).isdisjoint(set(TARGET)|set(AUX))
    for regime in ('partial_shift','continuous','full_reset'):
        rows = []
        for i in range(64):
            obs, truth = simulate_link(NEW_SEED0+i, 9., regime, 'sparse')
            predictions, weights, old = predict(obs,return_inputs=True)
            state = track_frame_a(obs.y_a, obs.n0)
            if i == 0:
                check = sample_moments_b(condition_frame_b(state, obs.y_b, obs.training_rows, obs.n0))
                trial = refit(obs, state, 0, 0.)
                for expected, actual in zip(check, trial):
                    np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-10)
            base = predictions['p9_classic']
            ref, result, proposed, used = select_and_repair(obs, weights[-1], old, base, state)
            _, pilot_repair, pilot_proposed, pilot_used = select_and_repair(obs, weights[0], old, base, state)
            row = dict(seed=NEW_SEED0+i,regime=regime,proposed=proposed,used=used,pilot_proposed=pilot_proposed,pilot_used=pilot_used,
                       energy=float(np.sum(abs(truth.y_b_probe_clean)**2)))
            for key, model in [('p9_classic',ref),('conditional_repair',result),('pilot_routing_repair',pilot_repair)]:
                err = (model['bits'] != truth.b_info[TARGET_INFO]).any(axis=1)
                row[key] = dict(errors=err.astype(int).tolist(),
                                sse=float(np.sum(abs(model['probe'] - truth.y_b_probe_clean)**2)))
            rows.append(row)
        all_records.extend(rows)
        stats = dict(links=len(rows), target_blocks=len(rows)*len(TARGET),
                     proposed=sum(r['proposed'] for r in rows), used=sum(r['used'] for r in rows))
        for key in ('p9_classic','conditional_repair','pilot_routing_repair'):
            stats[key] = dict(nmse=sum(r[key]['sse'] for r in rows)/sum(r['energy'] for r in rows),
                              errors=sum(sum(r[key]['errors']) for r in rows))
        d = np.array([np.array(r['p9_classic']['errors'])-r['conditional_repair']['errors'] for r in rows])
        b = np.random.default_rng(128).integers(len(rows),size=(4000,len(rows)))
        stats.update(rescued=int((d>0).sum()), newly_wrong=int((d<0).sum()),
                     gain_ci=np.quantile(d.mean(1)[b].mean(1),[.025,.975]).tolist())
        dp = np.array([np.array(r['pilot_routing_repair']['errors'])-r['conditional_repair']['errors'] for r in rows])
        stats['pilot_minus_cov_gain_ci'] = np.quantile(dp.mean(1)[b].mean(1),[.025,.975]).tolist()
        stats['pilot_used'] = sum(r['pilot_used'] for r in rows)
        summary[regime] = stats
    return dict(summary=summary,records=all_records)


if __name__=='__main__':
    result=evaluate()
    print(json.dumps(result['summary'],indent=2))
    Path('p10_repair_results.json').write_text(json.dumps(result,indent=2)+'\n')
