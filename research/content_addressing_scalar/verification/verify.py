"""Check closed-form risks for content-based retrieval from independent complex values."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def binomial_mass(d, prob):
    return np.array([math.comb(d, h) * prob**h * (1 - prob)**(d - h)
                     for h in range(d + 1)])


def margin_probability(d, rho, m, gap, direction):
    """Condition on true-key distance; all distractor distances are independent fair binomials."""
    good = binomial_mass(d, rho)
    distractor = binomial_mass(d, 0.5)
    probabilities = []
    for h in range(d + 1):
        if direction > 0:
            tail = distractor[h + gap:].sum()
        else:
            tail = distractor[:max(0, h - gap + 1)].sum()
        probabilities.append(tail**(m - 1))
    return float(np.clip(np.dot(good, probabilities), 0.0, 1.0))


def softmax(logits):
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def posterior(distances, d, rho):
    if rho == 0:
        compatible = distances == 0
        return compatible / compatible.sum(axis=1, keepdims=True)
    # The joint key density cancels. The binary channel likelihood is
    # proportional to (rho / (1-rho))**distance for every candidate.
    return softmax(distances * (math.log(rho) - math.log1p(-rho)))


def policy_parameters(name, d, rho):
    if name == 'bayes':
        return 1, math.inf if rho == 0 else d / 2 * (math.log1p(-rho) - math.log(rho))
    return (-1 if name == 'reversed_cold' else 1), (0.02 if name == 'hot' else 80.0)


def attention(distances, d, direction, beta):
    scores = direction * (1.0 - 2.0 * distances / d)
    if math.isinf(beta):
        best = scores == scores.max(axis=1, keepdims=True)
        return best / best.sum(axis=1, keepdims=True)
    return softmax(beta * scores)


def cn(rng, n, variance):
    return math.sqrt(variance / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def draw(rng, count, m, d, rho, v0, noise):
    idx = rng.integers(m, size=count)
    rows = np.arange(count)
    keys = rng.integers(0, 2, size=(count, m, d), dtype=np.int8)
    query = keys[rows, idx] ^ (rng.random((count, d)) < rho)
    distances = np.count_nonzero(keys != query[:, None, :], axis=2)
    theta = cn(rng, count, v0)
    values = cn(rng, (count, m), v0 + noise)
    values[rows, idx] = theta + cn(rng, count, noise)
    return idx, distances, theta, values


def mean_se(samples):
    return float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(len(samples)))


def finite_exact(m, k, d, rho, v0, gain, policy):
    if (d + 1)**m > 10000:
        return {}
    # Exchangeability permits J=0 for symmetric global errors and permits
    # multiplying global posterior concentration by k/M for the fixed window.
    distances = np.array(list(itertools.product(range(d + 1), repeat=m)))
    law = (binomial_mass(d, rho)[distances[:, 0]]
           * np.prod(binomial_mass(d, 0.5)[distances[:, 1:]], axis=1))
    p = posterior(distances[law > 0], d, rho)
    a = attention(distances[law > 0], d, *policy_parameters(policy, d, rho))
    law = law[law > 0]
    concentration = law @ (p**2).sum(axis=1)
    return {'att_exact': float(v0 + gain * (law @ ((a**2).sum(axis=1) - 2*a[:, 0]))),
            'local_exact': float(v0 - gain * k / m * concentration),
            'global_exact': float(v0 - gain * concentration)}


def check(condition, label, failures):
    if not condition:
        failures.append(label)


def run_scenario(cfg, trials, batch_size, failures):
    name, m, k, d, rho = (cfg[key] for key in ('name', 'M', 'k', 'd', 'rho'))
    v0, noise, gap = (cfg[key] for key in ('v0', 'sigma2', 'gap_bits'))
    lam = v0 / (v0 + noise)
    gain = v0 * lam
    v1 = v0 * noise / (v0 + noise)
    oracle_local = v0 - gain * k / m
    rng = np.random.Generator(np.random.PCG64(cfg['seed']))
    history = {name: [] for name in cfg['policies']}
    common_measurements = []
    for start in range(0, trials, batch_size):
        n = min(batch_size, trials - start)
        j, h, target, values = draw(rng, n, m, d, rho, v0, noise)
        rows = np.arange(n)
        p = posterior(h, d, rho)
        # Actual estimators receive only observable distances and the permitted values.
        local_loss = abs(lam * (p[:, :k] * values[:, :k]).sum(axis=1) - target)**2
        global_loss = abs(lam * (p * values).sum(axis=1) - target)**2
        hard_loss = abs(lam * values[rows, h.argmin(axis=1)] - target)**2
        active = np.argsort(-p, axis=1, kind='stable')[:, :k]
        ac_weights = np.take_along_axis(p, active, axis=1)
        ac_values = np.take_along_axis(values, active, axis=1)
        adaptive_loss = abs(lam * (ac_weights * ac_values).sum(axis=1) - target)**2
        local_risk = v0 - gain * (p[:, :k]**2).sum(axis=1)
        global_risk = v0 - gain * (p**2).sum(axis=1)
        common_measurements.append(np.column_stack((local_loss, global_loss, hard_loss, adaptive_loss,
                                                    local_loss-local_risk, global_loss-global_risk,
                                                    local_risk, global_risk)))
        for policy in cfg['policies']:
            direction, beta = policy_parameters(policy, d, rho)
            a = attention(h, d, direction, beta)
            att_loss = abs(lam * (a * values).sum(axis=1) - target)**2
            # J enters only this evaluative comparison and the margin count.
            target_weight = a[rows, j]
            conditional_risk = v1 + gain * (1 - 2 * target_weight + (a**2).sum(axis=1))
            signed_h = direction * h
            true_distance = signed_h[rows, j].copy()
            signed_h[rows, j] = d + 1
            margin = signed_h.min(axis=1) - true_distance >= gap
            history[policy].append(np.column_stack((att_loss, conditional_risk, att_loss - conditional_risk,
                                                    att_loss - local_loss, margin)))
    common = np.concatenate(common_measurements)
    local, glob, hard, adaptive, local_err, glob_err, local_theory, glob_theory = common.T
    for error, label in [(local_err, 'local'), (glob_err, 'global')]:
        bias, se = mean_se(error)
        check(abs(bias) <= 6 * se + 1e-12, f'{name}: {label} formula', failures)
    rows_out = []
    for policy, data in history.items():
        losses, theory, residual, paired, sampled_margin = np.concatenate(data).T
        direction, beta = policy_parameters(policy, d, rho)
        prob = margin_probability(d, rho, m, gap, direction)
        b = (m-1) * math.exp(-beta * 2 * gap / d)
        r = b / (1+b)
        ub = v1 + 2 * gain * ((1-prob) + prob * r*r)
        err, err_se = mean_se(residual)
        diff, diff_se = mean_se(paired)
        check(abs(err) <= 6 * err_se + 1e-12, f'{name}/{policy}: attention formula', failures)
        check(abs(sampled_margin.mean() - prob) <= 6*math.sqrt(prob*(1-prob)/trials) + 1e-12,
              f'{name}/{policy}: margin', failures)
        exact = finite_exact(m, k, d, rho, v0, gain, policy)
        if exact:
            for x, stat in [(losses, 'att_exact'), (local, 'local_exact'), (glob, 'global_exact')]:
                average, se = mean_se(x)
                check(abs(average - exact[stat]) <= 6 * se + 1e-12, f'{name}/{policy}: {stat}', failures)
        rows_out.append({'scenario':name, 'policy':policy, 'M':m, 'k':k, 'd':d, 'rho':rho,
                         'beta':beta, 'v0':v0, 'sigma2':noise, 'margin_exact':prob,
                         'margin_mc':sampled_margin.mean(), 'delta':2*gap/d, 'bound':ub,
                         'oracle_local':oracle_local, 'certified':ub < oracle_local,
                         'att_mc':losses.mean(), 'att_formula_mc':theory.mean(),
                         'formula_residual':err, 'formula_residual_se':err_se,
                         'local_mc':local.mean(), 'global_mc':glob.mean(),
                         'local_formula_mc':local_theory.mean(), 'global_formula_mc':glob_theory.mean(),
                         'local_formula_residual_se':mean_se(local_err)[1],
                         'global_formula_residual_se':mean_se(glob_err)[1],
                         'hard_mc':hard.mean(), 'adaptive_mc':adaptive.mean(),
                         'att_minus_local_mc':diff, 'att_minus_local_se':diff_se,
                         'att_exact':None, 'local_exact':None, 'global_exact':None, **exact})
    return rows_out


def dependent_example(cfg, n, failures):
    rng = np.random.Generator(np.random.PCG64(cfg['seed']))
    v0, noise, t = (cfg[key] for key in ('v0', 'sigma2', 'threshold_in_mean_powers'))
    w = v0 + noise
    lam, gain, v1 = v0 / w, v0**2 / w, v0 * noise / w
    j = rng.integers(2, size=n)
    rows = np.arange(n)
    first = rng.integers(2, size=n)
    keys = np.column_stack((first, 1-first))
    query = keys[rows, j]
    target = cn(rng, n, v0)
    values = cn(rng, (n, 2), w)
    values[rows, j] = target + cn(rng, n, noise)
    matched = (keys == query[:, None]).argmax(axis=1)
    skip = abs(values[rows, matched])**2 / w > t
    selected = np.where(skip, 1-matched, matched)
    losses = abs(lam * values[rows, selected] - target)**2
    nominal = v1 + 2 * gain * (selected != j)
    actual_exact = v1 + gain * (t + 2) * math.exp(-t)
    naive_exact = v1 + 2 * gain * math.exp(-t)
    actual, se = mean_se(losses)
    excess, excess_se = mean_se(losses - nominal)
    check(abs(actual - actual_exact) <= 6*se + 1e-12, 'dependent example: actual law', failures)
    check(abs(excess - (actual_exact - naive_exact)) <= 6*excess_se + 1e-12,
          'dependent example: gap', failures)
    return {'actual_mc':actual, 'actual_se':se, 'actual_exact':actual_exact,
            'naive_mc':float(nominal.mean()), 'naive_exact':naive_exact,
            'gap_mc':excess, 'gap_se':excess_se, 'gap_exact':actual_exact-naive_exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = (HERE / 'PLAN.json').read_bytes()
    plan = json.loads(plan_bytes)
    trials = plan['trials']
    rows, failures = [], []
    for spec in plan['scenarios']:
        rows.extend(run_scenario({**plan['defaults'], **spec}, trials, plan['batch_size'], failures))
    dependent = dependent_example(plan['correlated_counterexample'], trials, failures)
    args.out.mkdir(exist_ok=True, parents=True)
    with (args.out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    review = {'trials_per_scenario':trials, 'numpy':np.__version__, 'python':platform.python_version(),
              'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(), 'correlated_example':dependent,
              'failed_sanity_checks':failures}
    (args.out / 'checks.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
