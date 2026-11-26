"""Replay the M4 CPU checks, with no radio downloads or Modal calls by default."""

import argparse
import ast
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import betaln, digamma, gammainc, gammaln, logsumexp, ndtr, ndtri, xlogy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.matched_cache.learned import ROOTS, assign, forward, input_features, load
from research.stage4_math_and_design.M3 import ref_pool as m3

WEIGHT_PATH = ROOT / 'research/matched_cache/weights/equivariant_b8_s17.npz'
LOCK_PATH = ROOT / 'research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json'
PLUTO_REF = 'origin/research/realdata-pluto-qpsk'
CTTC_REF = 'origin/research/realdata-cttc-indoor-qpsk'
V0 = 10 ** (-0.2)
E = np.ones(3) / np.sqrt(3)
BASIS = np.column_stack(([1, -1, 0], np.array([1, 1, -2]) / np.sqrt(3))) / np.sqrt(2)
SEED = 404_017
HOLDOUT_CAPTURE = 'f708_snr10_QPSK'
RADIAL_EDGES = np.r_[np.expm1(np.linspace(0, np.log(25), 17))[:-1], np.inf]
QUALITY_EDGES = np.r_[0.0, np.geomspace(0.75, 16, 3), np.inf]


def branch_text(ref, folder, filename):
    return subprocess.check_output(
        ['git', 'show', f'{ref}:research/{folder}/{filename}'], cwd=ROOT, encoding='utf-8'
    )


def literal_constants(source, names):
    """Read selected branch literals and NumPy index constructors without importing branch code."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
        elif isinstance(node, ast.AnnAssign):
            name = getattr(node.target, 'id', None)
        else:
            continue
        if name in names:
            value = node.value
            if isinstance(value, ast.Call):
                # These are the only constructors used by the requested branch constants.
                function = ast.unparse(value.func)
                if function == 'np.concatenate':
                    args = [[np.arange(*ast.literal_eval(ast.Tuple(elts=item.args, ctx=ast.Load())))
                             for item in value.args[0].elts]]
                else:
                    args = [ast.literal_eval(arg) for arg in value.args]
                functions = {'slice': slice, 'np.arange': np.arange, 'np.array': np.array,
                             'np.concatenate': np.concatenate}
                values[name] = functions[function](*args)
            else:
                values[name] = ast.literal_eval(value)
    assert set(names) <= values.keys(), set(names) - values.keys()
    return values


def noise(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def gaussian_kl(mu, cov, mu0, cov0):
    delta = mu - mu0
    return float((np.linalg.slogdet(cov0)[1] - np.linalg.slogdet(cov)[1]
                  - len(mu) + np.trace(np.linalg.solve(cov0, cov))
                  + np.vdot(delta, np.linalg.solve(cov0, delta))).real)


def centered_t_kl(nu, m, trace_ratio, log_det_ratio):
    if nu <= 1:
        return math.inf
    return float(gammaln(nu + m) - gammaln(nu) - m * np.log(nu) + log_det_ratio
                 - (nu + m) * (digamma(nu + m) - digamma(nu))
                 + nu / (nu - 1) * trace_ratio)


def assert_mc(values, expected, sd_multiple=6):
    se = values.std(ddof=1) / np.sqrt(len(values))
    error = abs(values.mean() - expected)
    assert error <= sd_multiple * se + 1e-10, (values.mean(), expected, se)
    return {'monte_carlo': float(values.mean()), 'mc_se': float(se), 'exact': float(expected)}


def verify_distributions(rng):
    n = 120_000
    index = np.arange(3)
    covariance = V0 * (0.45 ** abs(index[:, None] - index)) / 1.2 ** 2
    mean = (np.exp(0.4j) + 0.26 * np.exp(0.7j)) * E / 1.2 + 0.13j * E
    baseline = V0 * np.eye(3)
    draws = mean + noise(rng, (n, 3)) @ np.linalg.cholesky(covariance).T
    residual = draws - mean
    q = np.einsum('bi,ij,bj->b', residual.conj(), np.linalg.inv(covariance), residual).real
    log_ratio = (3 * np.log(V0) - np.linalg.slogdet(covariance)[1] - q
                 + np.sum(abs(draws - E) ** 2, axis=1) / V0)
    full = assert_mc(log_ratio, gaussian_kl(mean, covariance, E, baseline))

    # If every tap is divided by the same scale, squared radius / nu is beta-prime(m, nu).
    t_rows = []
    for m in (1, 3):
        for nu in (1, 2, 3, 5, 10, 30, 100):
            # u = q / (q + nu) follows Beta(m, nu), including the Rayleigh case.
            def beta_pdf(u):
                return np.exp((m - 1) * np.log(u) + (nu - 1) * np.log1p(-u) - betaln(m, nu))

            integral = quad(beta_pdf, 0, 1, epsabs=2e-10)[0]
            log_moment = quad(lambda u: -np.log1p(-u) * beta_pdf(u), 0, 1, epsabs=2e-10)[0]
            np.testing.assert_allclose([integral, log_moment], [1, digamma(nu + m) - digamma(nu)], atol=2e-9)
            row = {'m': m, 'nu': nu, 'integral': integral, 'log_moment': log_moment}
            if nu == 1:
                row['kl_to_nominal'] = 'infinite'
                # For m=1, tail is exactly 1/(1+q), so truncated mean grows as log(1+R).
                if m == 1:
                    upper = 1000
                    truncated = quad(lambda r: 1 / (1 + r), 0, upper)[0]
                    np.testing.assert_allclose(truncated, np.log1p(upper), atol=1e-9)
                    row['E_min_q_1000'] = truncated
            else:
                constant = gammaln(nu + m) - gammaln(nu) - m * np.log(nu)
                radial_kl = quad(lambda u: (constant + (nu + m) * np.log1p(-u)
                                            + nu * u / (1 - u)) * beta_pdf(u), 0, 1, epsabs=2e-10)[0]
                exact = centered_t_kl(nu, m, m, 0)
                matched = centered_t_kl(nu, m, m * (nu - 1) / nu, m * np.log(nu / (nu - 1)))
                np.testing.assert_allclose(radial_kl, exact, atol=1e-8)
                assert 0 <= matched <= exact
                row.update(kl_to_nominal=exact, kl_to_moment_matched_gaussian=matched, quadrature=radial_kl)
            t_rows.append(row)
    return {'correlated_complex_gaussian_kl': full, 'student_t_radial': t_rows}


def posterior_and_mi(weights, labels):
    log_prob = weights - logsumexp(weights, axis=1, keepdims=True)
    nll = -log_prob[np.arange(len(labels)), labels]
    return float(np.log(4) - nll.mean()), nll


def channel_information(labels, codes, code_count):
    counts = np.bincount(code_count * labels + codes, minlength=4 * code_count).reshape(4, code_count)
    p = counts[:, counts.sum(axis=0) > 0] / counts.sum(axis=1, keepdims=True)
    prior = np.full(4, 0.25)
    row_term = xlogy(p, p).sum(axis=1)
    for step in range(500):
        divergence = row_term - p @ np.log(prior @ p)
        lower, upper = prior @ divergence, divergence.max()
        if step == 0:
            uniform_info = lower
        if upper - lower < 1e-6:
            break
        log_prior = np.log(prior) + divergence
        prior = np.exp(log_prior - logsumexp(log_prior))
    return float(uniform_info), [float(lower), float(upper)], counts.tolist()


def encoder_metrics(encoder, windows, labels):
    x, turns = input_features(windows, encoder.equivariant)
    hidden, prediction = forward(x, encoder.weights)
    codes = turns * len(encoder.centers) + assign(prediction, encoder.centers)
    table = encoder.table[codes]
    uniform_mi, capacity, counts = channel_information(labels, codes, len(encoder.table))
    return {
        'ser': float(np.mean(table.argmax(axis=1) != labels)),
        'frozen_table_nll_nats': float(-np.log(table[np.arange(len(labels)), labels]).mean()),
        'code_uniform_mi_nats': uniform_mi,
        'code_capacity_empirical_bracket_nats': capacity,
        'quadrant_local_counts_by_label': counts,
        'hidden_fraction_abs_gt_0p99': float((abs(hidden) > 0.99).mean()),
        'used_local_codes': int(len(np.unique(codes % len(encoder.centers)))),
    }


def lift(x, ancillary):
    return x[:, None] * E + ancillary @ BASIS.T


def pack_dual_cache(encoder, t, measured_residual, nominal_variance, ancillary):
    """M5 16-bit layout. `measured_residual` comes from the radio taps before R3 lift."""
    if not (np.all(np.isfinite(t)) and np.all(t != 0) and np.all(np.isfinite(measured_residual))
            and np.all(measured_residual >= 0) and np.all(np.isfinite(nominal_variance))
            and np.all(nominal_variance > 0)):
        raise ValueError('Unavailable windows must use the separate validity mask')
    legacy = encoder.encode(lift(t, ancillary)).astype(np.uint16)
    turn = legacy >> 2
    local = (np.angle(t * ROOTS[turn].conj()) + np.pi / 4) / (np.pi / 2)
    angle = np.clip(np.floor(64 * local), 0, 63).astype(np.uint16)
    radius = np.searchsorted(RADIAL_EDGES[1:-1], abs(t) / np.sqrt(nominal_variance)).astype(np.uint16)
    quality = np.searchsorted(QUALITY_EDGES[1:-1], measured_residual / nominal_variance).astype(np.uint16)
    packed = ((legacy << 12) | (angle << 6) | (radius << 2) | quality).astype('<u2')
    np.testing.assert_array_equal(packed >> 12, legacy)
    return packed


def polar_id(packet):
    """Drop only the legacy inner bits from the packed word."""
    return ((packet >> 14) << 12) | (packet & 0x0fff)


def polar_noise_emissions(mean, multiplier, tol_order):
    """P(angle, radius, quality | class) for unit nominal thermal variance."""
    node, weight = np.polynomial.legendre.leggauss(tol_order)
    width = 2 * np.pi / 256
    angle = -np.pi / 4 + (np.arange(256)[:, None] + (1 + node) / 2) * width
    radius = RADIAL_EDGES / np.sqrt(multiplier)
    mean = mean / np.sqrt(multiplier)
    radius = np.clip(radius, max(0, abs(mean) - 11), abs(mean) + 11)
    result = np.zeros((256, 16))
    lateral = 4 * abs(mean) * np.sin((angle - np.angle(mean)) / 2) ** 2
    for i, (lo, hi) in enumerate(zip(radius[:-1], radius[1:])):
        edges = np.linspace(lo, hi, 1 + max(1, math.ceil(hi - lo)))
        dr = np.diff(edges)[:, None] / 2
        r = (edges[1:, None] + edges[:-1, None]) / 2 + dr * node
        radial_weight = (dr * weight * r).ravel()
        distance = (r.ravel() - abs(mean)) ** 2 + lateral[..., None] * r.ravel()
        result[:, i] = (np.exp(-distance) @ radial_weight) @ (weight * width / (2 * np.pi))
    mass_quality = np.diff(gammainc(2, QUALITY_EDGES / multiplier))
    probability = (result[..., None] * mass_quality).ravel()
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-9)
    return probability


def ceiling_readout(log_likelihood, qpsk, burst):
    logp = (log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True)).reshape(-1, 2, 4)
    index = np.arange(len(qpsk))
    by_tx = logsumexp(logp, axis=1)
    by_burst = logsumexp(logp, axis=2)
    return np.column_stack((-by_tx[index, qpsk], -by_burst[index, burst], -logp[index, burst, qpsk]))


def verify_dual_cache(encoder):
    rng = np.random.default_rng(SEED + 2)
    profiles = [('Pluto_fade', db, .0095 * 10 ** ((db + 2) / 20), 0.0) for db in (-2, 3, 8)]
    profiles += [('CTTC_spur', db, 1.0, 0.26) for db in (-3, 10, 20)]
    multipliers = np.array([1.0, 4.0])
    field = np.arange(1 << 16, dtype=np.uint16)
    state8 = np.arange(8)
    results = []
    max_quadrature_change = 0.0
    for profile, snr_db, gain, spur in profiles:
        nominal = 10 ** (-snr_db / 10)
        mean = np.exp(0.23j) + spur * np.exp(0.8j)
        distributions = {name: [] for name in ('s17', 'polar_quality', 'full16')}
        for multiplier in multipliers:
            v = nominal * multiplier
            t_cal = mean + np.sqrt(v) * noise(rng, 65_536)
            e_cal = v * rng.gamma(2, 1, len(t_cal))
            calibration = pack_dual_cache(encoder, t_cal, e_cal, nominal, np.sqrt(V0) * noise(rng, (len(t_cal), 2)))
            legacy_count = np.bincount(calibration >> 12, minlength=16)
            legacy = (legacy_count + 0.5) / (len(calibration) + 8)
            polar = polar_noise_emissions(mean / np.sqrt(nominal), multiplier, 14)
            coarser = polar_noise_emissions(mean / np.sqrt(nominal), multiplier, 10)
            max_quadrature_change = max(max_quadrature_change, float(abs(polar - coarser).sum()))
            inner_count = np.bincount(calibration, minlength=1 << 16) + 0.5
            by_polar = np.bincount(polar_id(field), weights=inner_count, minlength=1 << 14)
            full16 = polar[polar_id(field)] * inner_count / by_polar[polar_id(field)]
            np.testing.assert_allclose(np.bincount(polar_id(field), weights=full16), polar, atol=2e-16)
            for name, density, quarter in [('s17', legacy, 4), ('polar_quality', polar, 4096), ('full16', full16, 16384)]:
                distributions[name].extend(np.roll(density, quarter * s) for s in range(4))

        # Independent blind draws for the declared mixture. Raw gain and raw thermal noise
        # are both represented before applying the causal gain normalization.
        joint_label = np.tile(state8, 4096)
        qpsk, burst = joint_label % 4, joint_label // 4
        variance = nominal * multipliers[burst]
        raw = gain * mean * ROOTS[qpsk] + gain * np.sqrt(variance) * noise(rng, len(qpsk))
        t = raw / gain
        residual = variance * rng.gamma(2, 1, len(qpsk))
        ancillary = np.sqrt(V0) * noise(rng, (len(qpsk), 2))
        pack = pack_dual_cache(encoder, t, residual, nominal, ancillary)
        rotated = pack_dual_cache(encoder, 1j * t, residual, nominal, 1j * ancillary)
        np.testing.assert_array_equal(rotated, (((pack >> 14) + 1) % 4 << 14) | (pack & 0x3fff))
        views = {'s17': pack >> 12, 'polar_quality': polar_id(pack), 'full16': pack}
        scores = {name: ceiling_readout(np.log(np.maximum(np.asarray(distributions[name])[:, indices].T, 1e-280)),
                                        qpsk, burst) for name, indices in views.items()}
        distance = abs(t[:, None] - mean * ROOTS[state8 % 4]) ** 2 + residual[:, None]
        state_variance = nominal * multipliers[state8 // 4]
        scores['continuous_bayes'] = ceiling_readout(-distance / state_variance - 3 * np.log(state_variance), qpsk, burst)
        names = ('symbol', 'variance_state', 'joint')
        gap = scores['s17'] - scores['full16']
        se = gap.std(axis=0, ddof=1) / np.sqrt(len(qpsk))
        for name, value in scores.items():
            if name != 'continuous_bayes':
                regret = value - scores['continuous_bayes']
                assert np.all(regret.mean(axis=0) >= -6 * regret.std(axis=0, ddof=1) / np.sqrt(len(qpsk)) - 1e-9)
        results.append({'profile': profile, 'snr_db': snr_db, 'gain': gain, 'spur_amplitude': spur,
                        'nominal_raw_variance': gain ** 2 * nominal,
                        'nll_nats': {name: dict(zip(names, value.mean(axis=0))) for name, value in scores.items()},
                        's17_minus_full16_nll_nats': dict(zip(names, gap.mean(axis=0))),
                        'paired_mc_standard_error_nats': dict(zip(names, se))})
    assert max_quadrature_change < 2e-8
    assert all(cell['s17_minus_full16_nll_nats']['joint'] > 0.07 for cell in results)
    return {'calibration_draws_per_base_state': 65_536, 'blind_draws_per_profile': 32_768,
            'max_polar_quadrature_l1_order_10_vs_14': max_quadrature_change,
            's17_reader': 'Monte Carlo cell emissions with exact C4 shifts',
            'full16_reader': 'physical polar/quality integration times Monte Carlo inner conditional emissions',
            'scope': 'Declared synthetic mixture; finite-emission decoder NLL estimates, continuous Bayes reference',
            'cells': results}


def frozen_encoder_sweep(rng, encoder):
    labels = np.tile(np.arange(4), 2048)
    symbols = ROOTS[labels]
    idx = np.arange(3)
    colored = 0.65 ** abs(idx[:, None] - idx)
    steering = np.array([0.73, 1, 0.86])
    steering /= np.sqrt(steering @ np.linalg.solve(colored, steering))
    gls = np.linalg.solve(colored, steering)
    thermal = noise(rng, (len(labels), 3)) @ np.linalg.cholesky(colored).T
    ancillary = np.sqrt(V0) * noise(rng, (len(labels), 2))
    extra = noise(rng, len(labels))
    phase_spur = 2 * np.pi * 1503.0 / 500_000.0 * np.arange(len(labels))
    rows = []
    for snr_db in (-2, 0, 2, 6, 10, 15, 20):
        variance = 10 ** (-snr_db / 10)
        oracle_scalar = symbols + np.sqrt(variance) * (thermal @ gls)
        information, _ = posterior_and_mi(-abs(oracle_scalar[:, None] - ROOTS) ** 2 / variance, labels)
        for delta in (0, 0.05, 0.15, 0.30):
            v = variance / (1 + delta) ** 2
            alpha = min(1.0, np.sqrt(V0 / v))
            for dc in (0, 0.15, 0.30):
                for spur in (0, 0.26):
                    signal = symbols * (1 + spur * np.exp(1j * phase_spur))
                    raw = signal[:, None] * steering + dc * np.exp(0.65j) * E + np.sqrt(variance) * thermal
                    scalar = (raw @ gls) / (1 + delta)
                    matched = scalar[:, None] * E
                    calibrated_t = alpha * scalar + np.sqrt(max(0.0, V0 - alpha ** 2 * v)) * extra
                    calibrated = lift(calibrated_t, ancillary)
                    normalized = lift(scalar, ancillary)
                    np.testing.assert_allclose(calibrated @ E, calibrated_t, atol=1e-12)
                    np.testing.assert_allclose(normalized @ E, scalar, atol=1e-12)
                    views = {'R1_colored_3tap': raw / ((1 + delta) * np.linalg.norm(steering)),
                             'R2_repeated_GLS': matched, 'R3_lift_physical_SNR': normalized,
                             'R3_lift_calibrated_SNR': calibrated}
                    metrics = {name: encoder_metrics(encoder, x, labels) for name, x in views.items()}
                    if dc == spur == 0:
                        assert all(item['code_uniform_mi_nats'] < information + 0.045 for item in metrics.values())
                    rows.append({'physical_snr_db': snr_db, 'delta': delta, 'dc': dc, 'spur': spur,
                                 'normalized_noise_variance': v, 'calibration_alpha': alpha,
                                 'clean_scalar_bayes_mi_nats_mc': information, 'maps': metrics})
    # The zero limit preserves the quadrant for nonzero inputs until the exact zero tie.
    ray = lift(np.array([1 + 0.07j]), np.array([[0.1j, -0.1]]))
    x, _ = input_features(ray, True)
    w1, b1, w2, b2 = encoder.weights
    projection = (x @ w1)[0]
    assert np.all(projection != 0)
    signs = np.sign(projection)[None, :]
    limiting_scores = np.exp(signs @ w2 + b2 - logsumexp(signs @ w2 + b2, axis=1, keepdims=True))
    saturation_code = int(assign(limiting_scores, encoder.centers)[0])
    zero_prediction = forward(np.zeros_like(x), encoder.weights)[1]
    zero_code = int(assign(zero_prediction, encoder.centers)[0])
    log_centers = np.log(encoder.centers)
    pairwise_low = log_centers[zero_code] - np.delete(log_centers, zero_code, axis=0)
    pairwise_high = log_centers[saturation_code] - np.delete(log_centers, saturation_code, axis=0)
    margin_low = (pairwise_low @ zero_prediction[0]).min()
    margin_high = (pairwise_high @ limiting_scores[0]).min()
    w2_norm = np.linalg.norm(w2, ord=2)
    assert min(margin_low, margin_high) > 0
    low_radius = margin_low / (np.linalg.norm(projection) * w2_norm * np.linalg.norm(pairwise_low, axis=1).max())
    high_constant = 2 * np.sqrt(len(projection)) * w2_norm * np.linalg.norm(pairwise_high, axis=1).max()
    high_radius = (abs(b1).max() + max(0, np.log(high_constant / margin_high) / 2)) / abs(projection).min()
    low_codes = encoder.encode(1e-10 * ray * ROOTS[:, None])
    high_codes = encoder.encode(1e8 * ray * ROOTS[:, None])
    np.testing.assert_array_equal(low_codes, zero_code + len(encoder.centers) * np.arange(4))
    np.testing.assert_array_equal(high_codes, saturation_code + len(encoder.centers) * np.arange(4))
    assert encoder.encode(low_radius * ray)[0] == zero_code
    assert encoder.encode(high_radius * ray)[0] == saturation_code
    np.testing.assert_array_equal(encoder.table.argmax(axis=1), np.arange(16) // 4)
    rotated = encoder.encode(1j * normalized)
    np.testing.assert_array_equal(rotated, (encoder.encode(normalized) + len(encoder.centers)) % len(encoder.table))
    basis = np.column_stack((E, BASIS))
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-13)
    return {'sweep_examples_per_cell': len(labels), 'cells': rows,
            'ray_check': {'low_gain_codes': low_codes.tolist(), 'high_gain_codes': high_codes.tolist(),
                          'certified_low_ray_gain_upper': float(low_radius), 'certified_high_ray_gain_lower': float(high_radius),
                          'exact_zero_code': int(encoder.encode(np.zeros((1, 3)))[0])},
            'c4_equivariance_checked_on_rows': len(labels), 'frozen_table_map_always_equals_turn': True}


def nuisance_curves(rng, encoder):
    labels = np.tile(np.arange(4), 8192)
    signal = ROOTS[labels]
    ancillary = np.sqrt(V0) * noise(rng, (len(labels), 2))
    n = np.sqrt(V0) * noise(rng, len(labels))
    rows = []
    for kind, values in [('gain_delta', (-0.30, -0.15, -0.05, 0, 0.05, 0.15, 0.30)),
                         ('coherent_spur', (0, 0.10, 0.26, 0.40)),
                         ('cfo_rad_per_symbol', (0, 0.005, 0.015, 0.020, 0.05)),
                         ('student_t_nu', (1, 2, 3, 5, 10, 30))]:
        for value in values:
            for lag in ((4, 25, 126) if kind == 'cfo_rad_per_symbol' else (0,)):
                scale = 1.0
                rotation = 1.0
                model_v = V0
                if kind == 'gain_delta':
                    scale = 1 / (1 + value)
                    model_v = V0 * scale ** 2
                if kind == 'coherent_spur':
                    rotation = 1 + value * np.exp(0.65j + 2j * np.pi * 1503 * 25 / 500_000)
                if kind == 'cfo_rad_per_symbol':
                    rotation = np.exp(1j * value * lag)
                mu = scale * rotation
                if kind == 'student_t_nu':
                    tau = rng.gamma(value, 1 / value, len(labels))
                    t = signal + n / np.sqrt(tau)
                    q = abs(t[:, None] - ROOTS) ** 2 / V0
                    likelihood = -(value + 1) * np.log1p(q / value)
                    joint = centered_t_kl(value, 1, 1, 0)
                else:
                    t = scale * (signal * rotation + n)
                    likelihood = -abs(t[:, None] - mu * ROOTS) ** 2 / model_v
                    joint = gaussian_kl(np.array([mu]), np.array([[model_v]]), np.array([1]), np.array([[V0]]))
                    log_ratio = np.log(V0 / model_v) - abs(t - mu * signal) ** 2 / model_v + abs(t - signal) ** 2 / V0
                    assert_mc(log_ratio, joint)
                mi, oracle_nll = posterior_and_mi(likelihood, labels)
                _, nominal_nll = posterior_and_mi(-abs(t[:, None] - ROOTS) ** 2 / V0, labels)
                regret_samples = nominal_nll - oracle_nll
                regret_se = regret_samples.std(ddof=1) / np.sqrt(len(labels))
                assert -6 * regret_se - 1e-10 <= regret_samples.mean() <= joint + 6 * regret_se + 1e-10
                rows.append({'kind': kind, 'value': value, 'lag_symbols': lag, 'mean_real_imag': [float(np.real(mu)), float(np.imag(mu))],
                             'conditional_joint_kl_nats': joint if np.isfinite(joint) else 'infinite',
                             'scalar_oracle_uniform_mi_nats_mc': mi,
                             'nominal_posterior_regret_nats_mc': float(regret_samples.mean()),
                             'nominal_posterior_regret_mc_se': float(regret_se),
                             'frozen_lift': encoder_metrics(encoder, lift(t, ancillary), labels)})
    # With uniform hidden quarter-turn CFO, the four marginal likelihoods coincide exactly.
    phase = ROOTS[rng.integers(0, 4, len(labels))]
    t = phase * signal + n
    phase_mixture = -abs(t[:, None, None] - (ROOTS[:, None] * ROOTS[None, :])) ** 2 / V0
    mi, _ = posterior_and_mi(logsumexp(phase_mixture, axis=2), labels)
    assert abs(mi) < 1e-12
    scale_values = np.array([0.7, 1.3])
    scale_id = rng.integers(0, 2, len(labels))
    t = (signal + n) / scale_values[scale_id]
    variances = V0 / scale_values ** 2
    # Both signal and receiver noise are divided. The unknown gain induces a noncentral mixture.
    components = (-abs(t[:, None, None] - ROOTS[None, :, None] / scale_values) ** 2 / variances
                  - np.log(np.pi * variances))
    mixture = logsumexp(components, axis=2) - np.log(len(scale_values))
    latent_mi, latent_nll = posterior_and_mi(mixture, labels)
    revealed = components[np.arange(len(labels)), :, scale_id]
    revealed_mi, _ = posterior_and_mi(revealed, labels)
    nominal_true = -abs(t - signal) ** 2 / V0 - np.log(np.pi * V0)
    mixture_true = mixture[np.arange(len(labels)), labels]
    conditional_true = revealed[np.arange(len(labels)), labels]
    joint_kl = float((mixture_true - nominal_true).mean())
    scale_penalty = float((conditional_true - mixture_true).mean())
    expected = np.mean([gaussian_kl(np.array([1 / g]), np.array([[v]]), np.array([1]), np.array([[V0]]))
                        for g, v in zip(scale_values, variances)])
    check = assert_mc(conditional_true - nominal_true, expected)
    assert 0 <= joint_kl and 0 <= scale_penalty and latent_mi <= revealed_mi
    _, nominal_nll = posterior_and_mi(-abs(t[:, None] - ROOTS) ** 2 / V0, labels)
    assert 0 <= (nominal_nll - latent_nll).mean() <= joint_kl
    bounded_mixture = {'denominator_values_equiprobable': scale_values.tolist(),
                       'conditional_kl': check, 'mixture_kl_nats_mc': joint_kl,
                       'I_scale_observation_given_symbol_nats_mc': scale_penalty,
                       'hidden_scale_uniform_mi_nats_mc': latent_mi,
                       'known_scale_uniform_mi_nats_mc': revealed_mi,
                       'frozen_lift': encoder_metrics(encoder, lift(t, ancillary), labels)}
    return {'examples_per_cell': len(labels), 'conditional_curves': rows,
            'random_bounded_gain_mixture': bounded_mixture,
            'hidden_uniform_quarter_turn_mi_exact_nats': 0,
            'hidden_uniform_quarter_turn_encoder': encoder_metrics(encoder, lift(t, ancillary), labels)}


def guarded_intervals(ranges, circumference=None):
    ordered = sorted(ranges)
    gaps = [b[0] - a[1] for a, b in zip(ordered, ordered[1:])]
    if circumference is not None:
        gaps.append(ordered[0][0] + circumference - ordered[-1][1])
    assert min(gaps) >= 0
    return min(gaps)


def cubic_samples(center):
    return range(math.floor(center) - 1, math.floor(center) + 3)


def symbol_support(k, sps, fractional, offsets, base=0):
    return set().union(*(cubic_samples(base + sps * k + fractional + d) for d in offsets))


def audit_local_support(unit, holdout, allowed_nuisance, refs, targets, sps, offsets):
    assert set(refs).isdisjoint(holdout) and set(targets).isdisjoint(holdout)
    assert set(refs).isdisjoint(allowed_nuisance) and set(targets).isdisjoint(allowed_nuisance)
    max_support = 0
    for frac in (-0.5, -0.001, 0, 0.5, 0.999):
        blocked = set().union(*(symbol_support(k, sps, frac, offsets) for k in holdout))
        # Certify the union of the R1 sub-symbol and R2 central tap reads.
        for k in (*refs, *targets):
            support = symbol_support(k, sps, frac, offsets)
            assert not blocked.intersection(support), (unit, k, frac)
            max_support = max(max_support, len(support))
    return max_support


@dataclass(frozen=True)
class PhaseContext:
    """Synthetic receipt uses exact integer symbol clocks and a separate raw cutoff."""

    t1: int
    t2: int
    t3: int
    omega_hat: float
    v_omega: float
    diffusion_var_per_unit: float
    snr_linear: tuple[float, float]


@dataclass(frozen=True)
class Acquisition:
    acquisition_id: str
    unit: str
    split: str
    slot: int
    timestamp: int
    raw_support: tuple[int, int]
    matched_T: complex
    tx_root: int | None
    shares_channel: bool = True


def synthetic_acquisition(unit, split, slot, sps, raw_base, rng):
    support = (raw_base + sps * slot - 1, raw_base + sps * slot + 3)
    identity = hmac.new(b'M4 deterministic fixture salt', f'{unit}/{support}'.encode(), hashlib.sha256).hexdigest()
    return Acquisition(identity, unit, split, slot, slot, support, complex(noise(rng, 1)[0]), slot % 4)


def project_ref_pool(presented, allowed, conditioned, cutoff, r_rx, covariance, tx_anchor=0):
    """Project authorized new evidence into two deterministic, canonical slots."""
    accepted = {}
    for row in presented:
        if row is None:
            continue
        uid = row.acquisition_id
        previous = conditioned.get(uid, accepted.get(uid))
        if previous is not None:
            if row != previous:
                raise ValueError('Conflicting reuse of a physical acquisition ID')
            continue
        if not row.shares_channel:
            if uid in allowed:
                raise ValueError('Conflicting role on an authorized acquisition')
            continue
        if allowed.get(uid) != row or row.raw_support[1] > cutoff:
            raise ValueError('Observation unavailable at the committed cutoff')
        lo, hi = row.raw_support
        if any(max(lo, x.raw_support[0]) < min(hi, x.raw_support[1]) for x in conditioned.values()):
            raise ValueError('Baseline raw overlap requires the conditional raw-block contract')
        accepted[uid] = row
    model_order = sorted(allowed, key=lambda uid: (allowed[uid].timestamp, uid))
    ordered = [accepted[uid] for uid in model_order if uid in accepted]
    slots = np.zeros((2, 4), dtype='<f4')
    mask = np.zeros(2, dtype=np.uint8)
    k = np.zeros((2, 2, 2), dtype='<f4')
    where = [model_order.index(row.acquisition_id) for row in ordered]
    for i, row in enumerate(ordered):
        t = row.matched_T * ROOTS[r_rx].conjugate()
        slots[i] = [row.timestamp, t.real, t.imag, (row.tx_root - tx_anchor) % 4]
        mask[i] = 1
    conditioned_k = covariance[np.ix_(where, where)]
    k[:len(where), :len(where), 0] = conditioned_k.real
    k[:len(where), :len(where), 1] = conditioned_k.imag
    # Signed zero carries no information and has two byte representations in IEEE 754.
    slots[slots == 0] = 0
    k[k == 0] = 0
    return mask.tobytes() + slots.tobytes() + k.tobytes()


def conditional_gaussian(residual, covariance, boundary):
    """Conditional innovation and Schur covariance after conditioning on prefix blocks."""
    cross = covariance[boundary:, :boundary]
    leading = covariance[:boundary, :boundary]
    return (residual[boundary:] - cross @ np.linalg.solve(leading, residual[:boundary]),
            covariance[boundary:, boundary:] - cross @ np.linalg.solve(leading, cross.conj().T))


def gaussian_log_density(residual, covariance):
    return float(-np.vdot(residual, np.linalg.solve(covariance, residual)).real
                 - np.linalg.slogdet(covariance)[1] - len(residual) * np.log(np.pi))


def gauge_forecast(ctx, data, pilots, cov):
    """Exact finite carrier/QPSK predictor for the rotation fixture."""
    g = ROOTS * (0.9 + 0.17j)
    s = ROOTS
    t1, t2, t3 = ctx.t1, ctx.t2, ctx.t3
    a1 = g[:, None] * s[None, :]
    a2 = a1 * np.exp(1j * ctx.omega_hat * (t2 - t1))
    a3 = a1 * np.exp(1j * ctx.omega_hat * (t3 - t1))
    tref = np.array([p.timestamp for p in pilots])
    r = np.array([p.matched_T for p in pilots])
    d = ROOTS[np.array([p.tx_root for p in pilots])]
    residual = r[None, :] - g[:, None] * (d * np.exp(1j * ctx.omega_hat * (tref - t1)))
    l_ref = -np.einsum('bi,ij,bj->b', residual.conj(), np.linalg.inv(cov), residual).real
    ll = (l_ref[:, None, None] - np.abs(data[0] - a1)[:, :, None] ** 2 * ctx.snr_linear[0]
          - np.abs(data[1] - a2)[:, None, :] ** 2 * ctx.snr_linear[1])
    weights = np.exp(ll - logsumexp(ll))
    tx = np.arange(4)
    diffs = (tx[None, :] - tx[:, None]) % 4
    q1 = np.bincount(diffs.ravel(), weights=weights.sum(axis=0).ravel(), minlength=4)
    q2 = weights.sum(axis=(0, 2))
    p = ndtr((a3.real + a3.imag) * np.sqrt(ctx.snr_linear[0]))
    q = ndtr((a3.real - a3.imag) * np.sqrt(ctx.snr_linear[0]))
    sector = np.stack((p * q, p * (1 - q), (1 - p) * (1 - q), (1 - p) * q), axis=-1)
    q3 = (weights.sum(axis=2)[..., None] * sector).sum(axis=(0, 1))
    return q1, q2, q3


def verify_m3_pool_receipt():
    """Run the peer's v2 pool and serializer on the two real pilot time layouts."""
    tick = 1 << 20
    max_error = 0.0
    for sites in ((12, 15, 18), (5, 126, 127)):
        clocks = [m3.ClockMeta(f'acquisition-{j}', 'M4-synthetic', t * tick, t * tick + 1, complex(ROOTS[j]),
                              'cal-v2', role='history' if j == 0 else 'reference') for j, t in enumerate(sites)]
        phase = m3.PhaseContext(0, tick, .002, .0002, .013, p_phi0=.02, cov_phi0_omega=.001)
        phase_mean, phase_cov = phase.path_moments([c.physical_tick for c in clocks])
        assert np.linalg.eigvalsh(phase_cov).min() > 0
        phase_nodes = np.stack((phase_mean - .1, phase_mean + .1))
        cube = ROOTS[None, :, None, None] * np.exp(1j * phase_nodes)[None, None, :, :] * ROOTS[None, None, None, :3]
        ij = np.arange(3)
        cov = .4 ** abs(ij[:, None] - ij) * np.exp(.13j * (ij[:, None] - ij))
        obs = [m3.Observation(clock, y) for clock, y in zip(clocks, (.8 + .2j, .3 + .9j, -.7 + .2j))]
        log_base = -abs(obs[0].matched - cube[..., 0]) ** 2 / cov[0, 0].real
        nodes = m3.NodeBank(('shared-cfo',), tuple(c.physical_tick for c in clocks), phase_nodes[None], log_base)
        pool = m3.RefSufficientPool(nodes, clocks, cube, cov)
        cutoff = (sites[-1] + 20) * tick
        baseline = pool.start(obs[:1], cutoff)
        one = pool.update(baseline, obs[1:2], cutoff)
        two = pool.update(one, obs[2:], cutoff)
        all_at_once = pool.update(baseline, [obs[2], None, obs[0], obs[1], obs[2]], cutoff)
        np.testing.assert_array_equal(two.log_weights_by_turn, all_at_once.log_weights_by_turn)
        assert pool.update(two, obs, cutoff) is two
        y = np.array([o.matched for o in obs])
        residual = y - cube
        full = -np.einsum('...i,ij,...j->...', residual.conj(), np.linalg.inv(cov), residual).real
        expected = full - logsumexp(full)
        max_error = max(max_error, float(np.max(abs(two.log_weights_by_turn - expected))))
        canonical = np.exp(two.log_weights_by_turn).sum(axis=(0, 2))
        repeated = np.exp(all_at_once.log_weights_by_turn).sum(axis=(0, 2))
        wire_a = m3.fixed_slots((0, 1, 2), canonical, np.roll(canonical, 1))
        wire_b = m3.fixed_slots((0, 1, 2), repeated, np.roll(repeated, 1))
        assert all(a.tobytes() == b.tobytes() for a, b in zip(wire_a, wire_b))
    assert max_error < 1e-12
    return {'peer_wire_version': m3.WIRE_VERSION, 'tested_layouts': 2, 'max_posterior_log_error': max_error}


def verify_frontend_contract(encoder, splits, anchors):
    rng = np.random.default_rng(SEED + 3)
    preconditions = ('train', 'dev', 'test')
    schedule = [('Pluto_850', name, 20, 7560 * splits['pluto']['splits'][name]['eval_start'],
                 13, (15, 18), ((21, 26),), (34, 42, 160))
                for name in preconditions]
    source_ranges = splits['cttc']['split_source_ranges']
    for anchor in anchors:
        for loop in range(8):
            for split in preconditions:
                lo, hi = source_ranges[split]['target']
                stride = (hi - lo - 256) // (12 if split == 'dev' else 20)
                base = 4 * (loop * 500_000 + lo + 128 + loop * stride)
                next_repeat_or_private_future = 500140 if loop % 2 == 0 else 190
                schedule.append((anchor['capture_id'], split, 4, base, 6, (126, 127),
                                 ((40, 48), (200, 208)), (140, 148, next_repeat_or_private_future)))
    synth_ids = set()
    bad_checks, prefix_checks, max_gauge = 0, 0, 0.0
    for unit, split, sps, base, n_sync, ref_slots, holdout, times in schedule:
        cutoff = base + sps * times[1] + 3
        sync = [synthetic_acquisition(unit, split, k, sps, base, rng) for k in range(n_sync)]
        refs = [synthetic_acquisition(unit, split, k, sps, base, rng) for k in ref_slots]
        probes = [replace(synthetic_acquisition(unit, split, k, sps, base, rng), tx_root=None)
                  for span in holdout for k in range(*span)]
        windows = [replace(synthetic_acquisition(unit, split, k, sps, base, rng), tx_root=None) for k in times]
        for row in sync + refs + probes + windows:
            assert row.acquisition_id not in synth_ids
            synth_ids.add(row.acquisition_id)
        f0 = {row.acquisition_id: row for row in sync + windows[:2]}
        allowed = {row.acquisition_id: row for row in refs}
        assert not set(allowed) & set(f0)
        assert max(r.raw_support[1] for r in refs + sync) <= cutoff < windows[-1].raw_support[0]
        covariance = np.array([[.5, .17 - .04j], [.17 + .04j, .6]])
        ancillary = np.sqrt(V0) * noise(rng, (2, 2))
        data = np.array([row.matched_T for row in windows[:2]])
        measured_residual = np.array([.7, 1.1])
        packed = pack_dual_cache(encoder, data, measured_residual, .5, ancillary)
        r_rx = int(packed[0] >> 14)
        prefix = project_ref_pool(refs, allowed, f0, cutoff, r_rx, covariance)
        px_none = project_ref_pool([], allowed, f0, cutoff, r_rx, covariance)
        px_one = project_ref_pool(refs[:1], allowed, f0, cutoff, r_rx, covariance)
        assert prefix != px_one and px_one != px_none
        distractor = replace(probes[0], shares_channel=False)
        replay = [None, refs[1], sync[0], windows[0], refs[0], distractor, refs[1]]
        assert project_ref_pool(replay, allowed, f0, cutoff, r_rx, covariance) == prefix
        prefix_checks += 1
        # Poison every scorer-only observation and reevaluate from the locked allowlist.
        vault = {row.acquisition_id: row for row in refs + probes + windows[2:]}
        for row in probes + windows[2:]:
            vault[row.acquisition_id] = replace(row, matched_T=complex(1e6, -1e6))
        revised = [vault[uid] for uid in allowed]
        assert project_ref_pool(revised, allowed, f0, cutoff, r_rx, covariance) == prefix
        bad_rows = (probes[0], windows[2], replace(refs[0], split='other'),
                    replace(refs[0], matched_T=99j), replace(refs[0], shares_channel=False),
                    replace(refs[0], acquisition_id=windows[2].acquisition_id))
        for bad, deadline in [(row, cutoff) for row in bad_rows] + [(refs[0], refs[0].raw_support[1] - 1)]:
            try:
                project_ref_pool([bad], allowed, f0, deadline, r_rx, covariance)
            except ValueError:
                bad_checks += 1
            else:
                raise AssertionError('Forbidden reference was accepted')

        # Receive and transmit actions exercise distinct physical reparameterizations.
        ctx = PhaseContext(*times, 0.02 if sps == 20 else -.002, .0002, .013, (2., 3.))
        gauge = gauge_forecast(ctx, data, refs, covariance)
        for turn in range(4):
            rotated_data = data * ROOTS[turn]
            rotated_refs = [replace(row, matched_T=row.matched_T * ROOTS[turn]) for row in refs]
            rotated_f0 = {uid: replace(row, matched_T=row.matched_T * ROOTS[turn]) for uid, row in f0.items()}
            rotated_allowed = {row.acquisition_id: row for row in rotated_refs}
            rotated = pack_dual_cache(encoder, rotated_data, measured_residual, .5, ancillary * ROOTS[turn])
            np.testing.assert_array_equal(rotated, ((((packed >> 14) + turn) % 4) << 14) | (packed & 0x3fff))
            assert project_ref_pool(rotated_refs, rotated_allowed, rotated_f0, cutoff,
                                    (r_rx + turn) % 4, covariance) == prefix
            rx = gauge_forecast(ctx, rotated_data, rotated_refs, covariance)
            tx_refs = [replace(row, tx_root=(row.tx_root + turn) % 4) for row in refs]
            tx_allowed = {row.acquisition_id: row for row in tx_refs}
            assert project_ref_pool(tx_refs, tx_allowed, f0, cutoff, r_rx, covariance, tx_anchor=turn) == prefix
            tx = gauge_forecast(ctx, data, tx_refs, covariance)
            for actual, expected in zip(rx + tx, (gauge[0], gauge[1], np.roll(gauge[2], turn),
                                                   gauge[0], np.roll(gauge[1], turn), gauge[2])):
                max_gauge = max(max_gauge, float(abs(actual - expected).max()))
    assert max_gauge < 1e-12

    # Colored noise shared with F0 and between releases. Sequential innovations equal batch likelihood.
    positions = np.arange(4)
    covariance = 0.43 ** abs(positions[:, None] - positions) * np.exp(0.11j * (positions[:, None] - positions))
    residual = noise(rng, 4)
    both, k_both = conditional_gaussian(residual, covariance, 2)
    second, k_second = conditional_gaussian(residual, covariance, 3)
    incremental = gaussian_log_density(both[:1], k_both[:1, :1]) + gaussian_log_density(second, k_second)
    joint = gaussian_log_density(residual, covariance) - gaussian_log_density(residual[:2], covariance[:2, :2])
    np.testing.assert_allclose(incremental, joint, atol=1e-14)
    unconditioned = gaussian_log_density(residual[2:], covariance[2:, 2:])
    assert abs(unconditioned - joint) > 1e-3
    # The same latent CFO perturbs both pilot clocks, including times earlier than t1.
    clocks = np.array([-4., -1., 8., 126.])
    brownian = .013 * np.minimum(abs(clocks[:, None]), abs(clocks)) * (clocks[:, None] * clocks > 0)
    phase_cov = brownian + .0002 * np.outer(clocks, clocks)
    np.testing.assert_allclose(np.diag(phase_cov), .013 * abs(clocks) + .0002 * clocks ** 2)
    assert phase_cov[0, 2] < 0 and np.linalg.eigvalsh(phase_cov).min() > 0

    capture_rows = [{'capture': a['capture_id'], 'band_mhz': a['freq_mhz'], 'nominal_snr_db': a['nom_snr_db'],
                     'session_loops': 8, 'repeat_loop_pairs': 4, 'test_packets': 160, 'q3_repeat_rows': 80,
                     'stress_packets': 120, 'all_tuning_excluded': a['capture_id'] == HOLDOUT_CAPTURE} for a in anchors]
    assert len(capture_rows) == 9 and sum(r['all_tuning_excluded'] for r in capture_rows) == 1
    assert sum(r['test_packets'] for r in capture_rows) == splits['cttc']['counts']['test_target']
    assert sum(r['q3_repeat_rows'] for r in capture_rows) == splits['cttc']['repeat_candidate_rows']
    raw_and_ref_units = 7096 + sum(splits['cttc']['counts'].values())
    stress_units = splits['pluto']['stress_rows'] + splits['cttc']['stress_rows']
    unique_window_bound = 3 * (raw_and_ref_units + stress_units)
    assert unique_window_bound * 2 < 2_000_000
    return {'schema': 'm4.real_dual_cache.v2', 'scope': 'synthetic contract fixtures plus archived split manifest',
            'wire': {'packet_dtype': '<u2', 'packet_fields_high_to_low': [2, 2, 6, 4, 2],
                     'legacy_shift_bits': 12, 'ticks_per_symbol': 1 << 20, 'tick_dtype': '<i8',
                     'phase_fields': ['t1', 't2', 't3', 'omega_hat', 'v_omega', 'diffusion_var_per_unit', 'snr_linear'],
                     'phase_units': 'radians per symbol with positive receiver rotation',
                     'ref_noise_covariance': 'proper complex post-normalization K_ref; CFO uncertainty separate',
                     'radial_internal_edges': RADIAL_EDGES[1:-1].tolist(), 'quality_internal_edges': QUALITY_EDGES[1:-1].tolist(),
                     'gauge_output_shifts_q1_q2_q3': [0, 0, 'r_rx'],
                     'tx_relabel_output_shifts_q1_q2_q3': [0, 'r_tx', 0],
                     'canonical_gauges': ['W1 receive turn', 'explicit TX label-basis anchor']},
            'pluto_frames_in_archive': 7096, 'cttc_capture_units': capture_rows,
            'coverage': 'all enrolled rows retained; uniform fallback on common erasure; locked subset secondary',
            'probability_gauge_max_abs_defect': max_gauge, 'fixture_bundles': len(schedule),
            'byte_exact_refpool_fixture_bundles': prefix_checks,
            'rejected_forbidden_ref_attempts': bad_checks, 'distinct_acquisition_ids': len(synth_ids),
            'schur_v_joint_abs_log_likelihood_error': abs(incremental - joint),
            'unconditioned_v_joint_abs_log_likelihood_error': abs(unconditioned - joint),
            'shared_cfo_phase_covariance': phase_cov.tolist(),
            'dual_window_count_conservative_cap': unique_window_bound, 'packed_bytes_cap': unique_window_bound * 2,
            'packed_bytes_with_optional_separate_nibbles_cap': unique_window_bound * 2 + (unique_window_bound + 1) // 2,
            'measured_real_frontend_verified': False}


def audit_splits(anchors):
    pluto = 'realdata_pluto_qpsk'
    p = literal_constants(branch_text(PLUTO_REF, pluto, 'prepare_benchmarks.py'), ('SPLITS',))['SPLITS']
    f = literal_constants(branch_text(PLUTO_REF, pluto, 'waveform_frontend.py'),
                          ('SPS', 'N_SYM', 'RX_SAMP', 'SYNC_SLICE', 'PILOT_SLICE', 'PROBE_SLICE', 'PAYLOAD_SLICE'))
    sf = {name: set(range(*f[name].indices(f['N_SYM']))) for name in
          ('SYNC_SLICE', 'PILOT_SLICE', 'PROBE_SLICE', 'PAYLOAD_SLICE')}
    assert sum(map(len, sf.values())) == len(set.union(*sf.values())) == 126
    frame_sets = [(v['warmup_start'], v['eval_end']) for v in p.values()]
    assert guarded_intervals(frame_sets) == 100 and max(b for _, b in frame_sets) <= 7096
    p_counts = {s: v['eval_end'] - v['eval_start'] for s, v in p.items()}
    assert p_counts == {'train': 3000, 'dev': 1000, 'test': 2240}
    assert all(v['eval_start'] - v['warmup_start'] == 200 for v in p.values())
    p_refs = (15, 18)
    assert set(p_refs) <= sf['PILOT_SLICE']
    p_test = range(p['test']['eval_start'], p['test']['eval_end'])
    p_pairs = [(30 + f % 67, 1 if (f + f // 16) % 2 == 0 else 8) for f in p_test]
    p_targets = {k for start, lag in p_pairs for k in (start, start + lag)}
    assert p_targets <= sf['PAYLOAD_SLICE'] and max(p_refs) < min(p_targets)
    p_support = audit_local_support('Pluto', sf['PROBE_SLICE'], sf['SYNC_SLICE'],
                                    p_refs, p_targets, f['SPS'], (-2, 0, 2))
    p_base = f['N_SYM'] * f['SPS']
    assert f['RX_SAMP'] == p_base * 3
    # If the trigger phase allows it, the two designated copies are complete before the raw cutoff.
    p_latest_valid_copy1_anchor = f['RX_SAMP'] - 1 - max(symbol_support(max(p_targets), 20, 0.5, (-2, 0, 2), p_base))
    assert p_latest_valid_copy1_anchor >= p_base - 1
    p_stress_count = len(p_test[::16]) * 7

    c = literal_constants(branch_text(CTTC_REF, 'realdata_cttc_qpsk', 'adapter.py'),
                          ('SPLIT_SRC_RANGES', 'SOURCE_N_SYMS', 'PKT_LEN', 'PAD_SYMS', 'SPS_NOM',
                           'SYNC_IDX', 'PILOT_IDX', 'HOLDOUT_IDX'))
    pk, cycle, pad = c['PKT_LEN'], c['SOURCE_N_SYMS'], c['PAD_SYMS']
    ranges = c['SPLIT_SRC_RANGES']
    split_guard = guarded_intervals([rg for roles in ranges.values() for rg in roles.values()], cycle)
    assert split_guard == 4096
    c_known = set(c['SYNC_IDX']) | set(c['PILOT_IDX'])
    c_holdout = set(c['HOLDOUT_IDX'])
    c_payload = set(range(pk)) - c_known - c_holdout
    assert [len(s) for s in (c_known, c_holdout, c_payload)] == [10, 16, 230]
    c_refs = (126, 127)
    c_pairs = [(136 + (17 * j + 5 * loop) % 48, 1 if (j + loop) % 2 == 0 else 8)
               for loop in range(8) for j in range(20)]
    c_targets = {k for start, lag in c_pairs for k in (start, start + lag)}
    assert set(c_refs) <= c_known and c_targets <= c_payload and max(c_refs) < min(c_targets)
    c_support = audit_local_support('CTTC', c_holdout, set(c['SYNC_IDX']), c_refs, c_targets, c['SPS_NOM'], (-1, 0, 1))
    starts = {}
    bit_footprints = {}
    for split, roles in ranges.items():
        for role, (lo, hi) in roles.items():
            count = (12 if split == 'dev' else 20) if role == 'target' else 8
            stride = (hi - lo - pk) // count
            assert stride >= 2048
            pos = [lo + i * stride + 128 for i in range(count)]
            assert all(lo <= k and k + pk <= hi for k in pos)
            starts[split, role] = pos
            # Union EVEN and ODD raw TX-bit footprints, including edge bits shared by parities.
            bit_footprints[split, role] = {i % (2 * cycle) for k in pos for i in range(2 * k, 2 * (k + pk) + 1)}
    assert sum(map(len, bit_footprints.values())) == len(set.union(*bit_footprints.values()))
    ids = set()
    receiver_min_gap = math.inf
    ct_counts = dict.fromkeys((f'{sp}_{role}' for sp, role in starts), 0)
    for anchor in anchors:
        scale = anchor['sps_eff']
        origin = anchor['anchor_rx_pos'] - anchor['anchor_src_sym'] * scale
        while origin < 2_900_000:
            origin += cycle * scale
        # Recreate the branch file_2 selection and include source support of its 256-tap causal RRC.
        rx_ranges = []
        for loop in range(8):
            for (sp, role), positions in starts.items():
                for k in positions:
                    start = math.floor(origin + (loop * cycle + k) * scale) - pad * c['SPS_NOM']
                    end = start + (pk + 2 * pad) * c['SPS_NOM']
                    assert start >= 255 and end < anchor['n_samples']
                    rx_ranges.append((start - 255, end))
                    uid = (anchor['capture_id'], loop, sp, role, k)
                    assert uid not in ids
                    ids.add(uid)
                    ct_counts[f'{sp}_{role}'] += 1
        receiver_min_gap = min(receiver_min_gap, guarded_intervals(rx_ranges))
    assert len(anchors) == 9 and ct_counts == {
        'train_target': 1440, 'train_ref': 576, 'dev_target': 864, 'dev_ref': 576,
        'test_target': 1440, 'test_ref': 576}
    test_stress = [i for i in range(ct_counts['test_target']) if i % 4 != 3]
    assert len(test_stress) == 1080
    assert len({i // 20 for i in test_stress}) == 72
    assert len(starts['test', 'target']) * 4 * len(anchors) == 720  # disjoint adjacent loop pairs
    for k1, lag in (*p_pairs, *c_pairs):
        assert k1 < k1 + lag < k1 + 126 < k1 + cycle
    return {'pluto': {'splits': p, 'eval_counts': p_counts, 'frame_guard': 100,
                      'test_clusters': 16, 'frames_per_test_cluster': 140, 'stress_rows': p_stress_count,
                      'refs': p_refs, 'maximum_cubic_union_samples': p_support,
                      'repeat_candidate_rows': p_counts['test']},
            'cttc': {'split_source_ranges': ranges, 'cyclic_symbol_guard': split_guard, 'counts': ct_counts,
                     'test_session_loop_clusters': 72, 'repeat_disjoint_loop_pairs': 36,
                     'repeat_candidate_rows': 720, 'stress_rows': len(test_stress),
                     'minimum_acquisition_gap_samples_after_rcv_support_expansion': receiver_min_gap,
                     'source_bit_footprints_pairwise_disjoint_even_plus_odd': True,
                     'refs': c_refs, 'maximum_cubic_union_samples': c_support},
            'scope': 'Static split and window audit using branch literals and saved CTTC geometry. '
                     'Fresh causal synchronization, frontend provenance and real IQ are pending.'}


def verify_manifests(check_http):
    pluto = 'realdata_pluto_qpsk'
    cttc = 'realdata_cttc_qpsk'
    p = json.loads(branch_text(PLUTO_REF, pluto, 'metadata/data_manifest.json'))
    p_code = literal_constants(branch_text(PLUTO_REF, pluto, 'download_data.py'),
                               ('DATASET_ID', 'DATASET_VERSION', 'EXPECTED_FILENAME', 'EXPECTED_SIZE', 'EXPECTED_SHA256'))
    assert [p[k] for k in ('filename', 'size_bytes', 'sha256')] == [p_code[k] for k in
                                                                        ('EXPECTED_FILENAME', 'EXPECTED_SIZE', 'EXPECTED_SHA256')]
    m_id, version = p_code['DATASET_ID'], p_code['DATASET_VERSION']
    m_api = f'https://data.mendeley.com/public-api/datasets/{m_id}'
    assert p['public_api_metadata_url'] == f'{m_api}?version={version}'
    assert p['public_api_files_url'] == f'{m_api}/files?folder_id=root&version={version}'
    assert re.fullmatch(rf'https://data\.mendeley\.com/public-files/datasets/{m_id}/files/[a-f0-9-]{{36}}/file_downloaded', p['download_url'])
    assert p['sha256_matches_mendeley_api']
    assert all(re.fullmatch(r'[a-f0-9]{64}', p[k]) for k in ('sha256', 'metadata_sha256', 'files_metadata_sha256'))
    c_code = literal_constants(branch_text(CTTC_REF, cttc, 'download_cttc.py'), ('ZENODO_RECORD_ID', 'MANIFEST_FILES'))
    c = json.loads(branch_text(CTTC_REF, cttc, 'data_manifest.json'))
    record = c_code['ZENODO_RECORD_ID']
    assert c['zenodo_record_id'] == record and c['api_url'] == f'https://zenodo.org/api/records/{record}'
    archived = {f['filename']: f for f in c['files']}
    assert set(archived) == {f['filename'] for f in c_code['MANIFEST_FILES']}
    for entry in c_code['MANIFEST_FILES']:
        saved = archived[entry['filename']]
        assert entry['md5'] == saved['md5_official_verified']
        assert entry.get('extracted_files') == saved.get('extracted')
        assert all(entry[k] == saved[k] for k in ('size_bytes', 'sha256'))
    c['files'] = c_code['MANIFEST_FILES']
    api = f'https://zenodo.org/api/records/{record}/files'
    for entry in c['files']:
        assert re.fullmatch(r'[a-zA-Z0-9_.]+', entry['filename'])
        assert re.fullmatch(r'[a-f0-9]{32}', entry['md5']) and re.fullmatch(r'[a-f0-9]{64}', entry['sha256'])
        for member in entry.get('extracted_files', {}).values():
            assert re.fullmatch(r'[a-f0-9]{64}', member['sha256'])
        entry['download_url'] = f'{api}/{entry["filename"]}/content'
        assert urlsplit(entry['download_url']).netloc == 'zenodo.org'
    capture_files = [entry for entry in c['files'] if entry['role'] == 'capture']
    assert len(capture_files) == 9 and len(c['files']) == 11
    c['files_api_url'] = api
    c['capture_archives_bytes'] = sum(f['size_bytes'] for f in capture_files)
    c['all_selected_download_bytes'] = sum(f['size_bytes'] for f in c['files'])
    c['selected_extracted_bytes'] = sum(meta['size_bytes'] for f in c['files']
                                        for name, meta in f.get('extracted_files', {}).items() if name != 'file_1.dat')
    status = 'not_requested; pinned manifests compared with branch downloader constants'
    if check_http:
        status = {}
        for name, url in [('mendeley_files', p['public_api_files_url']), ('zenodo_files', api)]:
            try:
                with urlopen(url, timeout=6) as response:
                    status[name] = {'http_status': response.status, 'metadata_reachable': True}
            except OSError as error:
                status[name] = {'metadata_reachable': False, 'detail': str(error)}
    return {'pluto': p, 'cttc': c, 'total_download_bytes': p['size_bytes'] + c['all_selected_download_bytes'],
            'live_http': status}


def verify_budget(splits, lock):
    report = LOCK_PATH.with_name('REPORT.md').read_text(encoding='utf-8')
    for string in ('$17.12', '$30.00', '$12.88', '$0.00', '1,975.7', '23.88', '52,972.5', '45,409.9'):
        assert string in report
    arms = ('Zero', 'Naive-Additive', 'Equivariant-Host-Pure', 'Equivariant-Physical+Residual')
    steps, train_batch, pilot_steps = 384, 32, 69
    rate = {'A100-80GB': Decimal('0.000694'), 'A100-40GB': Decimal('0.000583'),
            'L4': Decimal('0.000222'), 'A10': Decimal('0.000306'), 'CPU': Decimal(0)}
    cpu, mem = Decimal('0.0000131'), Decimal('0.00000222')
    pluto = splits['pluto']
    cttc = splits['cttc']
    stress = splits['pluto']['stress_rows'] + splits['cttc']['stress_rows']
    synthetic = 32 * 64
    intervention_rows = 2 * (4 * synthetic + 2240 + 1440)
    eval_counts = {'pluto_raw': len(arms) * (pluto['eval_counts']['test'] * 4 + pluto['repeat_candidate_rows'] * 2),
                   'cttc_raw': len(arms) * (cttc['counts']['test_target'] * 4 + cttc['repeat_candidate_rows'] * 2),
                   'real_stress_q2': len(arms) * stress, 'synthetic_blind': len(arms) * synthetic * 5,
                   'dev_calibration': len(arms) * 3000, 'slot_and_text_interventions': intervention_rows}
    assert eval_counts == {'pluto_raw': 53760, 'cttc_raw': 28800, 'real_stress_q2': 8240,
                           'synthetic_blind': 40960, 'dev_calibration': 12000, 'slot_and_text_interventions': 23744}
    pilot_eval_rows = len(arms) * 1000
    pilot_train_seconds = math.ceil(len(arms) * pilot_steps * Decimal('2.50'))
    pilot_eval_seconds = math.ceil(pilot_eval_rows * Decimal('0.030'))
    pilot_seconds = 350 + pilot_train_seconds + pilot_eval_seconds
    assert pilot_seconds == 1160
    continuation_steps = len(arms) * (steps - pilot_steps)
    training_seconds = math.ceil(continuation_steps * Decimal('2.50'))
    eval_seconds = {key: math.ceil(value * Decimal('0.030')) for key, value in eval_counts.items()}
    tasks = [('initial_smoke_and_setup', 'A100-80GB', 350, 2, 16),
             ('tier2_pilot_train', 'A100-80GB', pilot_train_seconds, 2, 16),
             ('tier2_pilot_dev', 'A100-80GB', pilot_eval_seconds, 2, 16),
             ('continuation_if_go', 'A100-80GB', training_seconds, 2, 16)]
    tasks += [(key, 'A100-80GB', seconds, 2, 16) for key, seconds in eval_seconds.items()]
    tasks += [('remaining_setup_and_commit', 'A100-80GB', 750, 2, 16),
              ('gpu_retry_reserve', 'A100-80GB', 900, 2, 16),
              ('ingest_and_verify', 'CPU', 1800, 1, 8), ('frontend_and_physical', 'CPU', 7200, 4, 16),
              ('audit_and_aggregation', 'CPU', 1800, 2, 8)]
    rows = []
    total = Decimal(0)
    for label, gpu, seconds, cores, gib in tasks:
        usd = seconds * (rate[gpu] + cpu * cores + mem * gib)
        rows.append({'phase': label, 'resource': gpu, 'seconds_cap': seconds, 'physical_cores_cap': cores,
                     'gib_cap': gib, 'usd_cap': str(usd)})
        total += usd
    free, past, target = Decimal('30.00'), Decimal('17.12'), Decimal('9.50')
    a100 = rate['A100-80GB'] + 2 * cpu + 16 * mem
    original_a100 = rate['A100-80GB'] + 2 * cpu + 12 * mem
    assert a100 == Decimal('0.00075572') and original_a100 == Decimal('0.00074684')
    # Independent identity across all time caps, then a locked total for accidental scope changes.
    gpu_seconds = sum(row[2] for row in tasks if row[1] != 'CPU')
    assert total == gpu_seconds * rate['A100-80GB'] + cpu * sum(row[2] * row[3] for row in tasks) + mem * sum(row[2] * row[4] for row in tasks)
    assert 650 * a100 == Decimal('0.49121800')
    assert pilot_seconds * a100 == Decimal('0.87663520')
    assert gpu_seconds == 10987 and total == Decimal('9.07079564') and total < target < free - past
    peer = json.loads((HERE.parent / 'M6/results_m6.json').read_text(encoding='utf-8'))['modal_budget']
    peer_labels = dict(zip(
        ('ingest_and_verify', 'frontend_and_physical', 'audit_and_aggregation', 'initial_smoke_and_setup',
         'tier2_pilot_train', 'tier2_pilot_dev', 'continuation_if_go', 'dev_calibration', 'synthetic_blind',
         'pluto_raw', 'cttc_raw', 'real_stress_q2', 'slot_and_text_interventions', 'remaining_setup_and_commit', 'gpu_retry_reserve'),
        ('cpu_ingest_weight_staging', 'cpu_causal_frontend_dual_cache_pool', 'cpu_audit_aggregate', 'a100_initial_smoke_setup',
         'pilot_train_four_matched_arms', 'pilot_disjoint_dev', 'resume_same_four_arms', 'post_fit_disjoint_calibration_dev',
         'blind_synthetic_core', 'blind_pluto_raw', 'blind_cttc_raw', 'blind_real_stress', 'blind_matched_slot_text_controls',
         'a100_remaining_setup_teardown', 'explicit_retry_reserve')))
    assert {r['task'] for r in peer['rows']} == set(peer_labels.values())
    peer_rows = {r['task']: r for r in peer['rows']}
    for row in rows:
        match = peer_rows[peer_labels[row['phase']]]
        assert (row['resource'], row['seconds_cap'], row['physical_cores_cap'], row['gib_cap'], Decimal(row['usd_cap'])) == (
            match['device'], match['max_billed_seconds'], match['max_physical_cores'], match['max_gib'], Decimal(match['max_usd']))
    assert sum(eval_counts.values()) + pilot_eval_rows == sum(r['logical_eval_rows'] or 0 for r in peer_rows.values()) == 171504
    assert Decimal(peer['scheduled_usd_with_retry']) == total and Decimal(peer['cap_usd']) == target
    holdout_rows = len(arms) * (160 * 4 + 80 * 2)
    assert holdout_rows == 3200 and eval_counts['cttc_raw'] == holdout_rows * 9
    extra_pair = math.ceil(2 * 2 * steps * Decimal('2.50')) * a100
    assert total + extra_pair > target  # excludes optional additional scoring, so already infeasible
    # Engineering/futility only. This checks the planning reference, not a live dev outcome.
    info = 0.18
    posterior_variance = 1 / (.25 + info)
    prediction_sd = np.sqrt(1 - info + (1 - info) ** 2 * posterior_variance)
    slope = np.sqrt(info) * (1 + (1 - info) * posterior_variance)
    # Verify M6's reference cut by evaluating the exact Gaussian tail with its rounded z and boundary.
    np.testing.assert_allclose(ndtr((slope * -.270438 - 1.645088) / prediction_sd), .10, atol=1e-7)
    # Round 2's illustrative independent dev/blind model spends no alpha at the pilot.
    v = 1 / (.25 + .6 ** 2)
    dev_cutoff = (ndtri(1 - .05 / 3) + ndtri(.10) * np.sqrt(1 + v)) / (.6 * v)
    np.testing.assert_allclose(dev_cutoff, .0467962, atol=1e-7)
    small_smoke = {gpu: int(Decimal('0.10') / (rate[gpu] + 2 * cpu + 16 * mem)) for gpu in ('L4', 'A10')}
    return {'official_pricing_url': 'https://modal.com/pricing', 'usd_per_second': {k: str(v) for k, v in rate.items()},
            'physical_core_usd_per_second': str(cpu), 'gib_usd_per_second': str(mem), 'rows': rows,
            'a100_2c_16GiB_usd_per_second': str(a100), 'round1_a100_2c_12GiB_usd_per_second': str(original_a100),
            'm6_4000s_l4_to_a100_cost_increment': str(4000 * (rate['A100-80GB'] - rate['L4'])),
            'small_card_optional_receipt_seconds_under_0p10': small_smoke,
            'stage3d_model_revision': lock['hf_sha'], 'stage3d_host_sha256': lock['host_sha256'],
            'stage3d_locked_training': lock['locked_hyperparameters'],
            'stage3d_run_wall_seconds': [2884.0, 3026.2, 3217.8, 2908.7, 3059.1, 2983.0],
            'stage3d_observed': {'training_seconds_per_step': 1.9757, 'eval_seconds_per_row': 0.02388,
                                 'train_peak_mb': 52972.5, 'eval_peak_mb': 45409.9},
            'stage4_eval_rows': eval_counts, 'stage4_training_examples_per_arm': steps * train_batch,
            'arms': arms, 'training_seed': 202611, 'steps_per_arm': steps, 'inference_scope': 'fixed connectors conditional on seed 202611',
            'm6_round2_ledger_matched_row_for_row': True,
            'pilot': {'scope': 'same single seed for all four arms', 'max_steps_per_arm': pilot_steps,
                      'max_total_steps': pilot_steps * len(arms), 'previous_m6_pilot_steps': 173,
                      'previous_m6_optimizer_denominator': 960, 'previous_m6_pilot_usd': str(650 * a100),
                      'planned_arm_denominator': steps, 'dev_rows_cap': pilot_eval_rows, 'futility_probability_floor': .1,
                      'pilot_including_smoke_usd_cap': str(pilot_seconds * a100),
                      'previous_planning_reference_z': -.270438, 'previous_planning_reference_information_fraction': info,
                      'current_independent_proxy_cutoff': float(dev_cutoff),
                      'conditional_independent_blind_decision_only': True, 'live_result': 'pending'},
            'continuation_steps_if_go': continuation_steps,
            'heldout_f708_snr10_raw_rows_included': holdout_rows, 'heldout_f708_snr10_stress_rows_included': 120 * len(arms),
            'extra_two_seeds_for_two_arms_training_only_usd': str(extra_pair),
            'stage4_a100_seconds_cap': gpu_seconds, 'stage4_scheduled_usd_cap': str(total),
            'stage4_scheduled_without_retry_usd': str(total - 900 * a100),
            'stage4_total_usd_cap': str(target), 'unallocated_reserve_below_hard_cap_usd': str(target - total),
            'stage3d_free_at_report_usd': str(free - past), 'stage3d_reported_billed_usd': '0.00',
            'required_net_spend_guard_usd': '0.00',
            'hard_cap_headroom_to_historical_remaining_usd': str(free - past - target),
            'headroom_to_historical_remaining_usd': str(free - past - total),
            'active_balance_or_modal_execution_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-http', action='store_true', help='Optionally probe small public metadata endpoints.')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12) and np.__version__ == '2.3.5' and scipy.__version__ == '1.17.0'
    lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    digest = hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest()
    assert digest == lock['encoder_sha256']
    encoder = load(WEIGHT_PATH)
    assert encoder.equivariant and encoder.table.shape == (16, 4) and encoder.centers.shape == (4, 4)
    print('Verified frozen s17 identity and pinned Python/NumPy/SciPy', flush=True)
    anchors = json.loads(branch_text(CTTC_REF, 'realdata_cttc_qpsk', 'results/capture_anchors.json'))
    splits = audit_splits(anchors)
    manifests = verify_manifests(args.check_http)
    budget = verify_budget(splits, lock)
    print('Verified static splits, historical manifests and Modal arithmetic', flush=True)
    rng = np.random.default_rng(SEED)
    distributions = verify_distributions(rng)
    sweep = frozen_encoder_sweep(rng, encoder)
    curves = nuisance_curves(rng, encoder)
    dual_cache = verify_dual_cache(encoder)
    frontend = verify_frontend_contract(encoder, splits, anchors)
    frontend['m3_pool_integration'] = verify_m3_pool_receipt()
    result = {'status': 'pass', 'round': 2, 'seed': SEED, 'python': sys.version.split()[0], 'numpy': np.__version__,
              'scipy': scipy.__version__, 'scope': 'synthetic CPU replay, historical metadata and prospective budget',
              'encoder_sha256': digest, 'nominal_complex_noise_variance': V0,
              'ref_commits': {ref: subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
                              for ref in (PLUTO_REF, CTTC_REF)},
              'distributions': distributions, 'frozen_encoder_sweep': sweep, 'nuisance_curves': curves,
              'dual_cache_ceiling_surrogates': dual_cache, 'causal_frontend_contract': frontend,
              'index_audit': splits, 'dataset_manifests': manifests, 'modal_budget': budget}
    path = HERE / 'results_m4.json'
    staged = HERE / '.results_m4.json.tmp'
    staged.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staged.replace(path)
    print(f'PASS {len(sweep["cells"])} mapping cells, {len(curves["conditional_curves"])} nuisance cells, '
          f'{len(dual_cache["cells"])} dual-cache profiles, {frontend["fixture_bundles"]} causal bundles, {path.name}')


if __name__ == '__main__':
    main()
