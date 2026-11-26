"""Seeded CPU replay for the M5 cache and one-window blind test."""
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
from math import gcd
import os
from pathlib import Path
import runpy
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import scipy
from scipy.special import gammainc, logsumexp, xlogy
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MATCHED = REPO / "research" / "matched_cache"
sys.path[:0] = [str(MATCHED), str(REPO)]
from learned import forward, input_features, load
from research.stage4_math_and_design.M3 import ref_pool as m3_pool


SEED = 50261126
N_TEST = 4096
N_BLIND = 2048
N_CAL = 32768
PHASES = 128
QUARTER = PHASES // 4
UNIT = np.exp(2j * np.pi * np.arange(PHASES) / PHASES)
OLD = np.array([0, 2, 6, 10])
CANDIDATE = np.array([11, 1, 4, 8])
TAU = np.array([2.5, 4.5, 7.5])
BRANCH_NAMES = ("smooth", "jump_2.5", "jump_4.5", "jump_7.5", "independent", "burst")
LAYOUTS = {"2": (0, 0, 0), "8": (2, 1, 1), "12": (4, 3, 1), "16": (6, 4, 2), "24": (8, 8, 4)}
METHODS = ("2", "4_s17", "8", "12", "16", "canonical_continuous", "full_iq")
EXPECTED_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
WIRE_ID = "m5-c16-v1-le"
ADAPTIVE_DEV = 2048
ADAPTIVE_TEST = 4096
ADAPTIVE_FAMILIES = ("jump_interval", "selective_fade", "cfo_alias")
ADAPTIVE_CONTROLS = ("random_window", "nearest_window", "fixed_best_on_dev")
FROZEN_QUERY = (
    "只根据已缓存的观测和候选窗口说明，在四个窗口中选择一个补充参考。"
    "揭示前提交窗口编号、H0与H1的先验概率及各自对该窗口的预测分布。"
    "只读取所选窗口一次，随后给出H0与H1的后验概率。"
)
HOST_REVISIONS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "N1rv4n4/LEMMO_code_only": "ebcc1ac9a6d3a658e62eb73b77b326e3dd2dfc0a",
}


PROFILES = (
    ("flat_minus3", [-3] * 4, [1] * 4, [-3, 0, 2, -1], [1] * 4, -3),
    ("flat_plus2", [2] * 4, [1] * 4, [2, 2, 2, 2], [1] * 4, 2),
    ("flat_plus24", [24] * 4, [1] * 4, [5, 5, 5, 5], [1] * 4, 12),
    ("selective_fade", [14, -3, 3, 24], [1, .23, .42, 1.20], [7, 1, 7, 3], [1, .4, .85, .6], 0),
)


def normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def parameters(snr, amplitude):
    amplitude = np.array(amplitude, dtype=float)
    return amplitude, amplitude**2 / 10**(np.array(snr) / 10)


def phase_index(theta):
    return np.rint(theta * PHASES / (2 * np.pi)).astype(int) % PHASES


def make_tracks():
    slopes = (.18, .34)
    jump = 2.2
    tracks = []
    for omega in slopes:
        tracks.append((0, omega * OLD, omega * CANDIDATE, 1., -1, 1 / 8))
        for label, tau in enumerate(TAU, 1):
            old = (omega - jump / 10) * OLD + jump * (OLD >= tau)
            new = (omega - jump / 10) * CANDIDATE + jump * (CANDIDATE >= tau)
            tracks.append((label, old, new, np.where(CANDIDATE >= tau, .78, 1.), -1, 1 / 24))
        for offset in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            for omega_second in slopes:
                old = np.where(OLD < 5, omega * OLD, offset + omega_second * (OLD - 5))
                new = np.where(CANDIDATE < 5, omega * CANDIDATE, offset + omega_second * (CANDIDATE - 5))
                tracks.append((4, old, new, 1., -1, 1 / 64))
        for burst in (1, 2):
            tracks.append((5, omega * OLD, omega * CANDIDATE, 1., burst, 1 / 16))

    branch, old_phase, new_phase, gain, burst, prior = [], [], [], [], [], []
    for label, old, new, ratio, burst_window, weight in tracks:
        for g in range(4):
            branch.append(label)
            old_phase.append((phase_index(old) + QUARTER * g) % PHASES)
            new_phase.append((phase_index(new) + QUARTER * g) % PHASES)
            gain.append(np.broadcast_to(ratio, CANDIDATE.shape))
            burst.append(np.where(np.arange(len(OLD)) == burst_window, 6., 1.))
            prior.append(weight / 4)
    prior = np.array(prior)
    np.testing.assert_allclose(prior.sum(), 1.)
    return tuple(map(np.array, (branch, old_phase, new_phase, gain, burst, prior)))


def sample_old(rng, count, prior, phase, burst, amplitude, variance, ref_db):
    state = rng.choice(len(prior), count, p=prior)
    symbols = rng.integers(0, 4, (count, 2))
    shifts = np.zeros((count, len(OLD)), dtype=int)
    shifts[:, 1:3] = QUARTER * symbols
    mean = amplitude * UNIT[(phase[state] + shifts) % PHASES]
    iq = mean[..., None] / np.sqrt(3) + np.sqrt((variance * burst[state])[..., None]) * normal(rng, mean.shape + (3,))
    ref_variance = 10**(-ref_db / 10)
    late_reference = UNIT[phase[state, 0]] + np.sqrt(ref_variance) * normal(rng, count)
    return state, symbols, iq, late_reference, ref_variance


def features(iq):
    t = iq.sum(axis=-1) / np.sqrt(3)
    residual = np.sum(abs(iq - t[..., None] / np.sqrt(3))**2, axis=-1)
    return t, residual


def radial_edges(bits):
    if bits == 0:
        return np.array([0., np.inf])
    edges = np.expm1(np.linspace(0, np.log(25), 2**bits + 1))
    edges[-1] = np.inf
    return edges


WIRE_RADIAL_EDGES = np.r_[0., radial_edges(4)[2:]]


def quality_edges(bits):
    if bits == 0:
        return np.array([0., np.inf])
    if bits == 1:
        return np.array([0., 4., np.inf])
    return np.r_[0., np.geomspace(.75, 16., 2**bits - 1), np.inf]


def encode(iq, t, residual, variance, encoder):
    original = encoder.encode(iq.reshape(-1, 3)).reshape(t.shape)
    _, turns = input_features(iq.reshape(-1, 3), True)
    turns = turns.reshape(t.shape)
    np.testing.assert_array_equal(original // 4, turns)
    local_angle = (np.angle(t * (1j)**(-turns)) + np.pi / 4) / (np.pi / 2)
    packets = {"4_s17": original}
    bins = {}
    for method, (angle_bits, radial_bits, quality_bits) in LAYOUTS.items():
        angle = np.clip(np.floor(local_angle * 2**angle_bits).astype(int), 0, 2**angle_bits - 1)
        radius = np.searchsorted(radial_edges(radial_bits)[1:-1], abs(t) / np.sqrt(variance))
        quality = np.searchsorted(quality_edges(quality_bits)[1:-1], residual / variance)
        bins[method] = (turns * 2**angle_bits + angle, radius, quality)
        extra = radial_bits + quality_bits
        tail = (angle << extra) | (radius << quality_bits) | quality
        packets[method] = turns if method == "2" else (original << (angle_bits + extra)) | tail
        if method != "2":
            np.testing.assert_array_equal(packets[method] >> (angle_bits + extra), original)
        assert np.all((packets[method] >= 0) & (packets[method] < 2**int(method)))
        prefix = packets[method] if method == "2" else packets[method] >> (angle_bits + extra + 2)
        unpacked = (
            (prefix << angle_bits) | ((packets[method] >> extra) & (2**angle_bits - 1)),
            (packets[method] >> quality_bits) & (2**radial_bits - 1),
            packets[method] & (2**quality_bits - 1),
        )
        np.testing.assert_array_equal(unpacked, bins[method])
        bins[method] = unpacked
    return packets, bins


def s17_emissions(encoder, rng, amplitude, variance):
    """Independent Monte Carlo cell probabilities, with exact C4 transfer."""
    base = np.empty((QUARTER, 16))
    for phase in range(QUARTER):
        y = amplitude * UNIT[phase] / np.sqrt(3) + np.sqrt(variance) * normal(rng, (N_CAL, 3))
        counts = np.bincount(encoder.encode(y), minlength=16)
        base[phase] = (counts + .5) / (N_CAL + 8)
    return np.concatenate([np.roll(base, 4 * g, axis=1) for g in range(4)])


def polar_emissions(angle_bits, radial_bits, mean_amp, noise_ratio, order=18, radial_boundaries=None):
    """Integrate the complex Gaussian probability of each packed polar cell."""
    node, weight = np.polynomial.legendre.leggauss(order)
    # Scaled radii farther than 11 standard complex deviations have negligible mass.
    edges = radial_edges(radial_bits) if radial_boundaries is None else radial_boundaries
    r_edge = edges / np.sqrt(noise_ratio)
    r_edge = np.clip(r_edge, max(0., mean_amp - 11), mean_amp + 11)
    angles = 4 * 2**angle_bits
    width = 2 * np.pi / angles
    symmetry = gcd(PHASES, angles)
    base_phases = PHASES // symmetry
    angle_shift = angles // symmetry
    result = np.zeros((base_phases, angles, len(r_edge) - 1))
    phase = np.angle(UNIT[:base_phases])[:, None, None]
    radial_node = (r_edge[:-1, None] + r_edge[1:, None] + (r_edge[1:, None] - r_edge[:-1, None]) * node) / 2
    radial_weight = (r_edge[1:, None] - r_edge[:-1, None]) * weight * radial_node / 2
    subdivisions = max(1, int(np.ceil(mean_amp * width / 2)))
    shifts = ((np.arange(subdivisions)[:, None] + (1 + node) / 2) / subdivisions).ravel()
    angular_node = -np.pi / 4 + (np.arange(angles)[:, None] + shifts) * width
    angular_weight = np.tile(weight, subdivisions) * width / (2 * np.pi * subdivisions)
    for k, (r, rw) in enumerate(zip(radial_node, radial_weight)):
        # Subdivide broad radial or angular bins for concentrated, high-SNR profiles.
        if (r_edge[k + 1] - r_edge[k]) > 2:
            edges = np.linspace(r_edge[k], r_edge[k + 1], 9)
            r = ((edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[:, None] * node / 2).ravel()
            rw = (np.diff(edges)[:, None] * weight / 2).ravel() * r
        distance = (r - mean_amp)**2 + 2 * r * mean_amp * (1 - np.cos(angular_node - phase))[..., None]
        result[:, :, k] = (np.exp(-distance) @ rw) @ angular_weight
    np.testing.assert_allclose(result.sum(axis=(1, 2)), 1., atol=5e-6, rtol=0)
    base = np.log(np.maximum(result, 1e-280))
    return np.concatenate([np.roll(base, shift * angle_shift, axis=1) for shift in range(symmetry)])


def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
    count = len(t)
    values = np.empty((count, len(OLD), 2, PHASES))
    emissions = {}
    for window in range(len(OLD)):
        for noise_id, multiplier in enumerate((1., 6.)):
            variance = v[window] * multiplier
            if method in ("full_iq", "canonical_continuous"):
                # Common constants independent of the trajectory cancel in Bayes.
                distance = abs(t[:, window, None] - a[window] * UNIT)**2
                values[:, window, noise_id] = (
                    -(distance + residual[:, window, None]) / variance - 3 * np.log(variance)
                )
            elif method == "4_s17":
                condition = (a[window], variance)
                if condition not in emissions:
                    rng = np.random.default_rng([SEED, 47, key, window, noise_id])
                    emissions[condition] = s17_emissions(encoder, rng, *condition)
                emission = emissions[condition]
                values[:, window, noise_id] = np.log(emission[:, packets[method][:, window]].T)
            else:
                angle_bits, radial_bits, quality_bits = LAYOUTS[method]
                polar = polar_emissions(angle_bits, radial_bits, a[window] / np.sqrt(variance), multiplier)
                gamma_mass = np.diff(gammainc(2, quality_edges(quality_bits) / multiplier))
                angle, radius, quality = (value[:, window] for value in bins[method])
                values[:, window, noise_id] = polar[:, angle, radius].T + np.log(gamma_mass[quality, None])
    return values


def infer(windows, old_phase, burst, prior, reference=None, ref_variance=None):
    """Exact finite-state Bayes once the observation-cell likelihoods are supplied."""
    noise = (burst != 1).astype(int)
    logits = np.log(prior)[None, :].copy() + windows[:, 0, noise[:, 0], old_phase[:, 0]]
    logits += windows[:, 3, noise[:, 3], old_phase[:, 3]]
    symbol_ll = []
    for window in (1, 2):
        angles = (old_phase[:, window, None] + np.arange(4) * QUARTER) % PHASES
        ll = windows[:, window, noise[:, window, None], angles]
        logits += logsumexp(ll, axis=-1) - np.log(4)
        symbol_ll.append(np.exp(ll - logsumexp(ll, axis=-1, keepdims=True)))
    states = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    if reference is None:
        return states
    left, right = symbol_ll
    q1 = np.stack([np.sum(states * (left * np.roll(right, -delta, axis=-1)).sum(axis=-1), axis=1)
                   for delta in range(4)], axis=1)
    pilot_ll = -abs(reference[:, None] - UNIT[old_phase[:, 0]])**2 / ref_variance
    post_logits = logits + pilot_ll
    pilot_states = np.exp(post_logits - logsumexp(post_logits, axis=1, keepdims=True))
    q2 = np.einsum("ns,nsc->nc", pilot_states, left)
    return states, q1, q2


def score_rows(prob, target):
    np.testing.assert_allclose(prob.sum(axis=1), 1., rtol=0, atol=2e-10)
    return -np.log(np.maximum(prob[np.arange(len(target)), target], 1e-280)), prob.argmax(axis=1) == target


def score(loss, hit):
    return {"nll": float(np.mean(loss)), "accuracy": float(np.mean(hit)), "n": len(loss)}


def paired_ci(diff):
    mean = float(diff.mean())
    half_width = float(1.96 * np.std(diff, ddof=1) / np.sqrt(len(diff)))
    return {"mean": mean, "pointwise_95ci": [mean - half_width, mean + half_width]}


def components(new_phase, new_gain, branch):
    """Return the distinct component means and state-to-component maps per candidate."""
    groups = []
    for k in range(len(CANDIDATE)):
        by_class = []
        for h in (0, 1):
            mask = (branch == 0) if h == 0 else ((branch >= 1) & (branch <= 3))
            key = np.column_stack((new_phase[mask, k], new_gain[mask, k]))
            unique, inverse = np.unique(key, axis=0, return_inverse=True)
            indicator = np.eye(len(unique))[inverse]
            means = UNIT[unique[:, 0].astype(int)] * unique[:, 1]
            by_class.append((mask, indicator, means))
        groups.append(by_class)
    return groups


def density(z, means, variance):
    return -abs(z[..., None] - means)**2 / variance - np.log(np.pi * variance)


def integrate_eig(weights, means, variance, step):
    # Integrate on a Cartesian midpoint grid in noise-standardized coordinates.
    means = [mu / np.sqrt(variance) for mu in means]
    extent = max(np.max(abs(mu)) for mu in means) + 6.5
    count = int(np.ceil(2 * extent / step))
    step = 2 * extent / count
    axis = -extent + (np.arange(count) + .5) * step
    z = (axis[:, None] + 1j * axis).ravel()
    densities = [np.exp(density(z, mu, 1.)).T for mu in means]
    pi = [weight.sum(axis=1) for weight in weights]
    eig = np.zeros(len(pi[0]))
    for start in range(0, len(eig), 128):
        chunk = slice(start, start + 128)
        joint = [weight[chunk] @ grid for weight, grid in zip(weights, densities)]
        mix = joint[0] + joint[1]
        integrand = sum(xlogy(p, p) - xlogy(p, mix) - xlogy(p, prior[chunk, None])
                        for p, prior in zip(joint, pi))
        eig[chunk] = integrand.sum(axis=1) * step**2
    return eig


def digest_arrays(*arrays):
    digest = hashlib.sha256()
    for data in arrays:
        digest.update(np.ascontiguousarray(data).tobytes())
    return digest.hexdigest()


def pick_before_reveal(states, groups, a, v, actions):
    forecasts = []
    eig = []
    checks = []
    for k, by_class in enumerate(groups):
        joint = [states[:, mask] @ indicator for mask, indicator, _ in by_class]
        means = [a[k] * mean for _, _, mean in by_class]
        forecasts.append((joint, means))
        eig.append(integrate_eig(joint, means, v[k], .40))
        count = 64
        fine = integrate_eig([w[:count] for w in joint], means, v[k], .25)
        checks.append(float(np.max(abs(fine - eig[-1][:count]))))
    eig = np.column_stack(eig)
    assert np.isfinite(eig).all() and eig.min() > -1e-9
    np.testing.assert_array_less(eig, np.log(2) + 1e-8)
    selected = {**actions, "max_EIG_window": eig.argmax(axis=1)}
    rows = np.arange(len(states))
    assert np.all(eig[rows, selected["max_EIG_window"]] >= eig[rows, selected["nearest_window"]])
    receipt = digest_arrays(*selected.values(), v, *(arr for joint, means in forecasts for arr in (*joint, *means)))
    return forecasts, eig, selected, receipt, max(checks)


class SealedCandidates:
    """Record once and release only one selected observation per mother and policy."""

    def __init__(self, iq):
        self._iq = iq
        self._used = set()
        self._committed = {}

    def commit(self, policy, receipt, selected):
        assert receipt and not self._used and policy not in self._committed
        assert selected.shape == (len(self._iq),)
        assert np.all((selected >= 0) & (selected < self._iq.shape[1]))
        self._committed[policy] = (receipt, selected.copy())

    def reveal(self, policy, receipt):
        assert policy not in self._used
        expected, selected = self._committed[policy]
        assert receipt == expected
        self._used.add(policy)
        return self._iq[np.arange(len(selected)), selected]


def score_revealed(forecasts, eig, actions, receipt, candidates, method, v, truth, ambiguous):
    results, rows_by_policy = {}, {}
    count = len(truth)
    for policy, selected in actions.items():
        revealed = candidates.reveal((method, policy), receipt)
        observed, _ = features(revealed)
        prior = np.zeros((count, 2))
        conditional_log = np.zeros((count, 2))
        for k, (joint, means) in enumerate(forecasts):
            use = selected == k
            for h in (0, 1):
                sub = joint[h][use]
                p_h = sub.sum(axis=1)
                prior[use, h] = p_h
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_weight = np.log(sub) - np.log(p_h[:, None])
                # For a numerically zero class, the unnormalized ratios have no
                # posterior effect. Supply a finite prediction for trace scoring.
                log_weight[p_h == 0.] = -np.log(len(means[h]))
                conditional_log[use, h] = logsumexp(log_weight + density(observed[use], means[h], v[k]), axis=1)
        with np.errstate(divide="ignore"):
            log_joint = np.log(prior) + conditional_log
        marginal = logsumexp(log_joint, axis=1)
        posterior = np.exp(log_joint - marginal[:, None])
        after, hit = score_rows(posterior, truth)
        before, _ = score_rows(prior, truth)
        rows = np.arange(count)
        selected_eig = eig[rows, selected]
        realized = before - after
        result = score(after, hit)
        result.update(
            nll_before=float(before.mean()),
            expected_eig=float(selected_eig.mean()),
            observed_nll_gain=float(realized.mean()),
            observed_minus_expected_gain=paired_ci(realized - selected_eig),
            precommitted_mixture_obs_nll=float(-marginal.mean()),
            precommitted_true_h_obs_nll=float(-conditional_log[rows, truth].mean()),
            precommitted_rival_obs_nll=float(-conditional_log[rows, 1 - truth].mean()),
            selected_counts=dict(zip(CANDIDATE.tolist(), np.bincount(selected, minlength=len(CANDIDATE)).tolist())),
        )
        result["common_ambiguous_subset"] = (
            {**score(after[ambiguous], hit[ambiguous]),
             "nll_before": float(before[ambiguous].mean()),
             "expected_eig": float(selected_eig[ambiguous].mean()),
             "observed_nll_gain": float(realized[ambiguous].mean())}
            if np.any(ambiguous) else {"n": 0}
        )
        k = selected[0]
        joint, means = forecasts[k]
        result["example"] = {
            "candidate_time": int(CANDIDATE[k]),
            "prior": prior[0].tolist(),
            "per_h_predictive_mixtures": [
                {"weights": (joint[h][0] / prior[0, h]).tolist() if prior[0, h] > 0 else
                 np.full(len(means[h]), 1 / len(means[h])).tolist(),
                 "means_re_im": np.column_stack((means[h].real, means[h].imag)).tolist(),
                 "shared_component_complex_variance": float(v[k])}
                for h in (0, 1)
            ],
            "revealed_re_im": [float(observed[0].real), float(observed[0].imag)],
            "posterior": posterior[0].tolist(),
        }
        results[policy] = result
        rows_by_policy[policy] = after
    results["paired_nll_max_eig_minus_baseline"] = {
        baseline: paired_ci(rows_by_policy["max_EIG_window"] - rows_by_policy[baseline])
        for baseline in ("random_window", "nearest_window")
    }
    if np.any(ambiguous):
        results["paired_ambiguous_nll_max_eig_minus_baseline"] = {
            baseline: paired_ci((rows_by_policy["max_EIG_window"] - rows_by_policy[baseline])[ambiguous])
            for baseline in ("random_window", "nearest_window")
        }
    return results, rows_by_policy["max_EIG_window"]


def run_profile(key, profile, tracks, encoder):
    name, snr, amp, new_snr, new_amp, ref_db = profile
    branch, old_phase, new_phase, new_gain, burst, prior = tracks
    a, v = parameters(snr, amp)
    ca, cv = parameters(new_snr, new_amp)
    rng = np.random.default_rng([SEED, 63, key])
    state, symbols, iq, reference, ref_v = sample_old(rng, N_TEST, prior, old_phase, burst, a, v, ref_db)
    binary = branch <= 3
    binary_prior = prior[binary] / prior[binary].sum()
    rng_b = np.random.default_rng([SEED, 81, key])
    b_state, _, b_iq, _, _ = sample_old(
        rng_b, N_BLIND, binary_prior, old_phase[binary], burst[binary], a, v, ref_db
    )
    b_state = np.flatnonzero(binary)[b_state]
    b_truth = (branch[b_state] > 0).astype(int)
    new_mean = ca * new_gain[b_state] * UNIT[new_phase[b_state]]
    sealed = SealedCandidates(new_mean[..., None] / np.sqrt(3) + np.sqrt(cv[..., None])
                              * normal(rng_b, new_mean.shape + (3,)))
    # No Q2 reference is visible to the active selection arm.
    all_iq = np.concatenate((iq, b_iq))
    t, residual = features(all_iq)
    packets, bins = encode(all_iq, t, residual, v, encoder)
    groups = components(new_phase, new_gain, branch)
    actions = {"random_window": rng_b.integers(0, len(CANDIDATE), N_BLIND),
               "nearest_window": np.full(N_BLIND, np.argmin(abs(CANDIDATE - OLD[-1])))}
    outcomes, losses, committed = {}, {}, {}
    branch_targets = branch[state]
    q2_target = symbols[:, 0]
    q1_target = (symbols[:, 1] - q2_target) % 4
    true_binary = branch_targets <= 3
    for method in METHODS:
        print(f"{name}: {method}", flush=True)
        if method == "full_iq":
            # Direct raw-IQ likelihood check against the factorized (T, E) branch.
            for noise_id, multiplier in enumerate((1., 6.)):
                var = (v * multiplier)[None, :, None]
                raw = -np.sum(abs(all_iq[..., None] - a[None, :, None, None] * UNIT / np.sqrt(3))**2, axis=2)
                raw = raw / var - 3 * np.log(var)
                np.testing.assert_allclose(raw, cache_ll[:, :, noise_id], atol=5e-11, rtol=1e-12)
            continue
        ll = window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key)
        if method == "canonical_continuous":
            cache_ll = ll
        states, q1, q2 = infer(ll[:N_TEST], old_phase, burst, prior, reference, ref_v)
        event = states @ np.eye(len(BRANCH_NAMES))[branch]
        model = np.column_stack((event[:, 0], event[:, 1:4].sum(axis=1), event[:, 4:]))
        pair = model[true_binary, :2]
        pair /= pair.sum(axis=1, keepdims=True)
        jump_rows = (branch_targets >= 1) & (branch_targets <= 3)
        jump_prob = event[jump_rows, 1:4]
        jump_prob /= jump_prob.sum(axis=1, keepdims=True)
        model_target = np.choose(branch_targets, [0, 1, 1, 1, 2, 3])
        scored = {
            "Q1": score_rows(q1, q1_target),
            "Q2": score_rows(q2, q2_target),
            "stage5_model": score_rows(model, model_target),
            "stage5_event_and_location": score_rows(event, branch_targets),
            "stage5_smooth_vs_jump_given_pair": score_rows(pair, (branch_targets[true_binary] > 0).astype(int)),
            "stage5_jump_location_given_jump": score_rows(jump_prob, branch_targets[jump_rows] - 1),
        }
        outcomes[method] = {task: score(*value) for task, value in scored.items()}
        losses[method] = {task: value[0] for task, value in scored.items()}
        b_weights = infer(ll[N_TEST:], old_phase[binary], burst[binary], binary_prior)
        states_for_selection = np.zeros((N_BLIND, len(branch)))
        states_for_selection[:, binary] = b_weights
        committed[method] = pick_before_reveal(states_for_selection, groups, ca, cv, actions)
        if method == "canonical_continuous":
            # Common scoring stratum only. The selector never receives this mask.
            smooth_mass = states_for_selection[:, branch == 0].sum(axis=1)
            ambiguous = (.1 <= smooth_mass) & (smooth_mass <= .9)
    # Every policy and budget has committed its action and predictive law before any reveal.
    for method, (_, _, selected, receipt, _) in committed.items():
        for policy, indices in selected.items():
            sealed.commit((method, policy), receipt, indices)
    for method, precommit in committed.items():
        forecasts, eig, selected, receipt, integration_gap = precommit
        policy_result, post_loss = score_revealed(
            forecasts, eig, selected, receipt, sealed, method, cv, b_truth, ambiguous
        )
        outcomes[method]["stage6"] = {
            "commit_sha256": receipt,
            "quadrature_max_abs_delta_position_0_to_63": integration_gap,
            **policy_result,
        }
        losses[method]["stage6_post_max_eig"] = post_loss
    outcomes["full_iq"] = outcomes["canonical_continuous"]
    losses["full_iq"] = losses["canonical_continuous"]
    return {"profile": name, "old_snr_db": snr, "old_amplitude": amp, "reference_snr_db": ref_db,
            "candidate_snr_db_nominal": new_snr, "candidate_amplitude_nominal": new_amp,
            "metrics": outcomes}, losses


def collision_witness(encoder):
    # Both samples are strictly within one winning neural code cell on the real ray.
    amplitudes = np.geomspace(.02, 20., 300)
    encoded = encoder.encode(np.repeat(amplitudes[:, None] / np.sqrt(3), 3, axis=1))
    best = None
    for code in np.unique(encoded):
        group = amplitudes[encoded == code]
        logits = 2 * 10**.2 * group[:, None] * np.array([1, 0, -1, 0])
        posterior = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        difference = abs(posterior[-1] - posterior[0]).sum() / 2
        if best is None or difference > best["posterior_tv"]:
            best = {"code": int(code), "matched_sum_amplitude": [float(group[0]), float(group[-1])],
                    "posterior_at_pair": posterior[[0, -1]].tolist(), "posterior_tv": float(difference)}
    assert best["posterior_tv"] > .05
    selected = np.repeat(np.array(best["matched_sum_amplitude"])[:, None] / np.sqrt(3), 3, axis=1)
    canonical, turns = input_features(selected, True)
    _, soft = forward(canonical, encoder.weights)
    assignment = np.sort(soft @ np.log(encoder.centers).T, axis=1)
    best["strict_neural_assignment_margins"] = (assignment[:, -1] - assignment[:, -2]).tolist()
    assert turns.tolist() == [0, 0] and min(best["strict_neural_assignment_margins"]) > 0
    # Check all nearby Cartesian corners in 6 real dimensions for the witness.
    corners = np.array(list(np.ndindex(2, 2, 2, 2, 2, 2))) * 2 - 1
    perturb = 1e-5 * (corners[:, :3] + 1j * corners[:, 3:])
    np.testing.assert_array_equal(encoder.encode((selected[:, None] + perturb).reshape(-1, 3)), best["code"])
    return best


def legacy_inner_witness(encoder):
    """Permutations can preserve both Gaussian statistics while changing the learned code."""
    for phase in (0., .3, -.3):
        for radius in np.geomspace(.2, 3., 20):
            for delta in (.2, .6, 1.):
                distortions = [list(permutations((delta, -delta, 0))),
                               list(permutations((1j * delta, -1j * delta, 0)))]
                vectors = radius * np.exp(1j * phase) / np.sqrt(3) + np.concatenate(distortions)
                codes = encoder.encode(vectors)
                differences = np.flatnonzero(codes != codes[0])
                if differences.size:
                    pair = vectors[[0, differences[0]]]
                    t, e = features(pair)
                    np.testing.assert_allclose(t[0], t[1], atol=1e-14)
                    np.testing.assert_allclose(e, 2 * delta**2, atol=1e-14)
                    return {"matched_sum_re_im": [float(t[0].real), float(t[0].imag)],
                            "residual_energy": 2 * delta**2,
                            "codes": encoder.encode(pair).tolist(),
                            "iq_re_im": np.stack((pair.real, pair.imag), axis=-1).tolist()}
    raise AssertionError("Expected the legacy inner index to use more than matched sum and residual energy")


def canonical_word(legacy, t, physical_residual, variance, available):
    """Pack causal calibrated evidence with an optional legacy triplet code."""
    legacy, t, physical_residual, variance, available = np.broadcast_arrays(
        legacy, t, physical_residual, variance, available
    )
    if np.any((legacy < 0) | (legacy > 15)):
        raise ValueError("Legacy prefix must fit four bits")
    if np.any(available & (~np.isfinite(t) | ~np.isfinite(physical_residual)
                          | (physical_residual < 0) | ~np.isfinite(variance) | (variance <= 0))):
        raise ValueError("Available evidence must have a finite calibrated matched statistic")
    keep = available & (t != 0)
    phase = np.angle(np.where(keep, t * (1j)**(-(legacy // 4)), 1.))
    angle = np.clip(np.floor(64 * (phase + np.pi / 4) / (np.pi / 2)), 0, 63).astype(int)
    scale = np.where(available, variance, 1.)
    radial = 1 + np.searchsorted(WIRE_RADIAL_EDGES[1:-1], abs(np.where(keep, t, 0.)) / np.sqrt(scale))
    quality = np.searchsorted(quality_edges(2)[1:-1], np.where(keep, physical_residual, 0.) / scale)
    tail = (angle << 6) | (radial << 2) | quality
    return ((legacy.astype(int) << 12) | np.where(keep, tail, 0)).astype("<u2")


def m4_shard_to_wire(words, available):
    """Map a v0 shard plus measured phase eligibility to the self-contained v1 word."""
    words = np.asarray(words, dtype="<u2")
    radial = (words >> 2) & 15
    valid = words | ((radial == 0).astype("<u2") << 2)
    return np.where(available, valid, words & 0xf000).astype("<u2")


def wire_check(encoder):
    iq = np.array([[1, 1, 1], [1j, 1j, 1j], [1, -1, 0], [0, 0, 0]], dtype=complex)
    legacy = encoder.encode(iq)
    t, e = features(iq)
    available = np.array([True, True, True, False])
    legacy[-1] = 0  # Missing raw window has no historical triplet.
    word = canonical_word(legacy, t, e, 1., available)
    np.testing.assert_array_equal(word >> 12, legacy)
    np.testing.assert_array_equal(((word >> 2) & 15) == 0, ~available | (t == 0))
    assert np.all((word[2:] & 4095) == 0)
    rotated = canonical_word(encoder.encode(iq[:2] * 1j), t[:2] * 1j, e[:2], 1., True)
    np.testing.assert_array_equal(rotated & 0x3fff, word[:2] & 0x3fff)
    np.testing.assert_array_equal(rotated >> 14, ((word[:2] >> 14) + 1) % 4)
    assert canonical_word(10, -1 + 0j, 0., 1., True).tobytes() == bytes.fromhex("0ca8")
    # Check the peer's actual R3 packer, including all v0 radial and quality edges.
    m4_path = HERE.parent / "M4" / "verify_m4.py"
    m4 = runpy.run_path(str(m4_path))
    edge = radial_edges(4)[1:-1]
    r = np.r_[.01, np.repeat(edge, 3) * np.tile([1 - 1e-10, 1, 1 + 1e-10], len(edge)), 25.]
    t4 = np.tile(r + 0j, 4)
    e4 = np.repeat([.2, .9, 5., 17.], len(r))
    ancillary = np.zeros((len(t4), 2), complex)
    packed4 = m4["pack_dual_cache"](encoder, t4, e4, np.ones_like(e4), ancillary)
    eligible4 = np.arange(len(t4)) % 7 != 0
    expected = canonical_word(packed4 >> 12, t4, e4, 1., eligible4)
    np.testing.assert_array_equal(m4_shard_to_wire(packed4, eligible4), expected)
    np.testing.assert_array_equal(m4_shard_to_wire(np.arange(65536), True) >> 12,
                                  np.arange(65536) >> 12)
    return {
        "wire_id": WIRE_ID, "byte_order": "little", "bits_msb_first": [2, 2, 6, 4, 2],
        "valid_radial_code": [1, 15], "erasure_radial_code": 0, "erased_tail": 0,
        "radial_edges_with_infinite_upper_encoded_as_null": [*WIRE_RADIAL_EDGES[:-1].tolist(), None],
        "fixture_hex": word.tobytes().hex(),
        "m4_shard_conformance_fixture_words": len(t4),
        "m4_shard_code_sha256": hashlib.sha256(m4_path.read_bytes()).hexdigest(),
        "m4_mapping": "merge valid v0 radial codes 0 and 1 into v1 code 1; erased rows clear the tail",
    }


@dataclass(frozen=True)
class Geometry:
    name: str
    old_time: np.ndarray
    candidate_time: np.ndarray
    old_mean: np.ndarray
    candidate_mean: np.ndarray
    old_variance: float
    candidate_variance: float


def geometry_catalogue():
    """Return old means [window, state] and sealed means [action, H, state]."""
    old_time = np.arange(0, 9, 2)
    candidate_time = np.arange(1, 8, 2)
    gauge = (1j)**np.arange(4)
    jump_old = np.where(old_time[:, None] > candidate_time, -1., 1.)
    jump_offsets = np.array([-.5, .5])
    jump_new = np.where(
        candidate_time[:, None, None] > candidate_time[None, None, :] + jump_offsets[None, :, None],
        -1., 1.,
    )
    fade_gain = np.full((4, 4), .12) + .06 * (np.arange(4)[:, None] + np.arange(4)) % .15
    np.fill_diagonal(fade_gain, 1.35)
    fade_old = np.vstack([fade_gain, np.ones(4)])
    fade_new = np.stack((fade_gain, -fade_gain), axis=1)
    primes = np.array([2, 3, 5, 7])
    alias_old_time = 210 * np.arange(5)
    alias_time = 210 // primes
    base = 2 * np.pi * np.arange(4) / (4 * 210)
    alias_old = (1j)**((np.arange(5)[:, None] * np.arange(4)) % 4)
    alias_new = np.exp(1j * alias_time[:, None, None] * (
        base + np.arange(2)[:, None] * (2 * np.pi / primes)
    ))
    aliased = alias_time[:, None] % primes == 0
    alias_new[:, 1] = np.where(aliased, alias_new[:, 0], alias_new[:, 1])
    np.testing.assert_array_equal(~aliased, np.eye(4, dtype=bool))

    def turns(array):
        return (array[..., None] * gauge).reshape(*array.shape[:-1], 16)

    return (
        Geometry("jump_interval", old_time, candidate_time, turns(jump_old), turns(jump_new), .22, .45),
        Geometry("selective_fade", old_time, candidate_time, turns(fade_old), turns(fade_new), .14, .40),
        Geometry("cfo_alias", alias_old_time, alias_time, turns(alias_old), turns(alias_new), .29, .50),
    )


def geometry_posterior(words, geometry):
    """Integrate each observed polar cell using only the 16-bit cache."""
    angle = ((words >> 14) << 6) | ((words >> 6) & 63)
    radial = (words >> 2) & 15
    means = geometry.old_mean
    amplitude = np.round(abs(means), 12)
    phase = phase_index(np.angle(means))
    log_weights = np.zeros((len(words), means.shape[1]))
    for amp in np.unique(amplitude):
        emission = polar_emissions(6, 4, amp / np.sqrt(geometry.old_variance), 1.,
                                   radial_boundaries=WIRE_RADIAL_EDGES)
        for w in range(means.shape[0]):
            states = np.flatnonzero(amplitude[w] == amp)
            probability = emission[phase[w, states][None], angle[:, w, None],
                                   np.maximum(radial[:, w, None].astype(int) - 1, 0)]
            log_weights[:, states] += np.where(radial[:, w, None] > 0, probability, 0.)
    return np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))


def geometry_split(geometry, encoder, family_id, size, split):
    rng = np.random.default_rng([SEED, 62, family_id, split])
    state = rng.integers(0, 16, size)
    truth = rng.integers(0, 2, size)
    old = (geometry.old_mean[:, state].T[..., None] / np.sqrt(3)
           + np.sqrt(geometry.old_variance) * normal(rng, (size, len(geometry.old_time), 3)))
    t, e = features(old)
    legacy = encoder.encode(old.reshape(-1, 3)).reshape(t.shape)
    word = canonical_word(legacy, t, e, geometry.old_variance, True)
    np.testing.assert_array_equal(word >> 12, legacy)
    selected_mean = geometry.candidate_mean[:, truth, state].T
    candidates = SealedCandidates(selected_mean[..., None] / np.sqrt(3)
                                  + np.sqrt(geometry.candidate_variance) * normal(rng, (*selected_mean.shape, 3)))
    random_action = rng.integers(0, len(geometry.candidate_time), size)
    return word, truth, state, candidates, random_action


def geometry_utilities(weights, geometry):
    joint = [weights / 2, weights / 2]
    value, delta = [], []
    for means in geometry.candidate_mean:
        coarse = integrate_eig(joint, means, geometry.candidate_variance, .36)
        fine = integrate_eig([p[:64] for p in joint], means, geometry.candidate_variance, .23)
        value.append(coarse)
        delta.append(np.max(abs(fine - coarse[:64])))
    return np.column_stack(value), float(max(delta))


@dataclass(frozen=True)
class ReferenceBatch:
    acquisition_id: np.ndarray
    shares_channel: np.ndarray
    timestamp: np.ndarray
    K_ref: float
    matched_T: np.ndarray

    def __post_init__(self):
        for name in ("acquisition_id", "shares_channel", "timestamp", "matched_T"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def fingerprint(self):
        return digest_arrays(self.acquisition_id, self.shares_channel,
                             self.timestamp, np.float64(self.K_ref), self.matched_T)


class OneRevealRefPool:
    """M3-compatible scalar adapter for k=1 conditionally independent innovations."""

    def __init__(self, weights, prior_ids):
        self.weights = weights
        self.prior_ids = prior_ids
        self.accepted_id = None
        self.fingerprint = None

    def ingest(self, ref, means):
        used = np.any(ref.acquisition_id[:, None] == self.prior_ids, axis=1)
        use = ref.shares_channel & ~used
        if self.accepted_id is not None:
            if (np.any(use) or np.any(ref.acquisition_id == self.accepted_id)) and ref.fingerprint() != self.fingerprint:
                raise ValueError("Conflicting or second reveal")
            return self.result
        pi = np.full((len(use), 2), .5)
        conditional = np.zeros_like(pi)
        if np.any(use):
            with np.errstate(divide="ignore"):
                log_weight = np.log(self.weights[use, None])
            conditional[use] = logsumexp(
                log_weight + density(ref.matched_T[use, None], means[use], ref.K_ref), axis=-1
            )
        marginal = logsumexp(np.log(pi) + conditional, axis=1)
        posterior = np.exp(np.log(pi) + conditional - marginal[:, None])
        if np.any(use):
            self.accepted_id = ref.acquisition_id
            self.fingerprint = ref.fingerprint()
            self.result = (posterior, conditional, marginal)
        return posterior, conditional, marginal


def correlated_innovation_fixture():
    """Check the raw Gaussian conditional factor against full-minus-old density."""
    dt = np.array([.7, 1.8, 3.2])
    cfo_variance, diffusion_variance = .031, .072
    phase_covariance = (cfo_variance * np.outer(dt, dt)
                        + diffusion_variance * np.minimum.outer(dt, dt))
    np.testing.assert_allclose(phase_covariance - diffusion_variance * np.minimum.outer(dt, dt),
                               cfo_variance * np.outer(dt, dt))
    # Illustrative conditional receiver noise covariance, separate from latent phase.
    cov = np.array([[1, .2 + .1j, -.1j], [.2 - .1j, 1.1, .25], [.1j, .25, .8]])
    observed = np.array([.1 + .4j, .2 - .1j, -.3 + .3j])
    means = np.array([[1., .7 + .2j, .4 + .2j], [-1., -.5 + .4j, -.7 - .1j]])
    old = cov[:2, :2]
    transfer = np.linalg.solve(old, cov[:2, 2]).conj()
    variance = float((cov[2, 2] - transfer @ cov[:2, 2]).real)
    conditional_mean = means[:, 2] + (observed[:2] - means[:, :2]) @ transfer

    def log_density(x, mean, covariance):
        error = x - mean
        quadratic = np.einsum("bi,ij,bj->b", error.conj(), np.linalg.inv(covariance), error)
        return -quadratic.real - np.linalg.slogdet(covariance)[1] - len(x) * np.log(np.pi)

    direct = log_density(observed, means, cov) - log_density(observed[:2], means[:, :2], old)
    error = float(np.max(abs(direct - density(observed[2], conditional_mean, variance))))
    assert error < 1e-12
    return {"schur_residual_complex_variance": variance, "conditional_log_density_max_abs_error": error,
            "shared_cfo_diffusion_covariance": phase_covariance.tolist()}


def reveal_and_score(geometry, candidates, selection, receipt, weights, truth):
    revealed = candidates.reveal(receipt, receipt)
    matched_t, _ = features(revealed)
    ids = selection + len(geometry.old_time)
    refs = ReferenceBatch(ids, np.ones(len(selection), dtype=bool), geometry.candidate_time[selection],
                          geometry.candidate_variance, matched_t)
    pool = OneRevealRefPool(weights, np.arange(len(geometry.old_time)))
    means = geometry.candidate_mean[selection]
    old = ReferenceBatch(np.zeros_like(ids), refs.shares_channel, np.zeros_like(ids), refs.K_ref, matched_t)
    independent = ReferenceBatch(ids + 100, ~refs.shares_channel, refs.timestamp, refs.K_ref, matched_t)
    np.testing.assert_array_equal(pool.ingest(old, means)[0], .5)
    np.testing.assert_array_equal(pool.ingest(independent, means)[0], .5)
    posterior, log_cond, log_marginal = pool.ingest(refs, means)
    for repeat in (refs, old, independent):
        np.testing.assert_array_equal(pool.ingest(repeat, means)[0], posterior)
    try:
        pool.ingest(ReferenceBatch(ids, refs.shares_channel, refs.timestamp, refs.K_ref, matched_t + 1), means)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting samples accepted for a revealed acquisition ID")
    rows = np.arange(len(selection))
    with np.errstate(divide="ignore"):
        committed_log_cond = logsumexp(
            np.log(weights[:, None]) + density(matched_t[:, None], means, refs.K_ref), axis=-1
        )
    np.testing.assert_array_equal(log_cond, committed_log_cond)
    return (*score_rows(posterior, truth), log_cond[rows, truth], log_cond[rows, 1 - truth], log_marginal,
            matched_t, posterior, log_cond)


def check_m3_reveal(geometry, weights, selection, sample, posterior, log_cond, log_marginal):
    """Replay selected records through Track M3's actual shared Gaussian pool."""
    ticks = tuple(map(int, geometry.candidate_time))
    clocks = tuple(m3_pool.ClockMeta(str(k), f"m5-probe-{k}", tick, tick, 1, WIRE_ID, role="probe")
                   for k, tick in enumerate(ticks))
    means = geometry.candidate_mean.reshape(4, 2, 4, 4).transpose(1, 3, 2, 0)
    covariance = geometry.candidate_variance * np.eye(4)
    deadline = int(max(max(ticks), geometry.old_time[-1]))
    errors = []
    for row in range(8):
        with np.errstate(divide="ignore"):
            log_weights = np.broadcast_to(np.log(weights[row].reshape(4, 4).T / 2), (2, 4, 4))
        nodes = m3_pool.NodeBank(("H0", "H1"), ticks, np.angle(means[:, 0]), log_weights)
        pool = m3_pool.RefSufficientPool(nodes, clocks, means, covariance)
        initial = pool.start((), deadline)
        clock = clocks[selection[row]]
        forecast = pool.forecast(initial, clock.acquisition_id, deadline)
        observation = m3_pool.Observation(clock, complex(sample[row]))
        after = pool.reveal(initial, forecast, observation, deadline)
        assert pool.update(after, (observation,), deadline) is after
        ref_cond, ref_marginal, ref_log_post = forecast.log_densities([sample[row]])
        pool_post = np.exp(logsumexp(after.log_weights_by_turn, axis=(-1, -2)))
        errors.append(max(np.max(abs(ref_cond - log_cond[row])), abs(ref_marginal - log_marginal[row]),
                          np.max(abs(np.exp(ref_log_post) - posterior[row])),
                          np.max(abs(pool_post - posterior[row]))))
    error = float(max(errors))
    assert error < 2e-12
    return error


def commit_geometry(sealed, actions, words, weights, geometry):
    receipts = {}
    for policy, selected in actions.items():
        receipt = digest_arrays(np.frombuffer(policy.encode("utf-8"), dtype=np.uint8), words, selected,
                                weights, geometry.candidate_mean[selected], np.float64(geometry.candidate_variance))
        sealed.commit(receipt, receipt, selected)
        receipts[policy] = receipt
    return receipts


def run_geometry(family_id, geometry, encoder):
    print(f"adaptive {geometry.name}", flush=True)
    dev_words, dev_truth, _, dev_sealed, _ = geometry_split(geometry, encoder, family_id, ADAPTIVE_DEV, 0)
    dev_weights = geometry_posterior(dev_words, geometry)
    dev_eig, _ = geometry_utilities(dev_weights, geometry)
    dev_actions = {str(k): np.full(ADAPTIVE_DEV, k) for k in range(len(geometry.candidate_time))}
    dev_receipts = commit_geometry(dev_sealed, dev_actions, dev_words, dev_weights, geometry)
    dev_nll = [
        reveal_and_score(geometry, dev_sealed, action, dev_receipts[k], dev_weights, dev_truth)[0].mean()
        for k, action in dev_actions.items()
    ]
    fixed_dev = int(np.argmin(dev_nll))
    words, truth, state, sealed, random_action = geometry_split(geometry, encoder, family_id, ADAPTIVE_TEST, 1)
    weights = geometry_posterior(words, geometry)
    eig, grid_delta = geometry_utilities(weights, geometry)
    n = len(words)
    rows = np.arange(n)
    nearest = int(np.argmin(abs(geometry.candidate_time - geometry.old_time[-1])))
    constants = {f"constant_{k}": np.full(n, k) for k in range(len(geometry.candidate_time))}
    actions = {
        "max_EIG_window": eig.argmax(axis=1),
        "random_window": random_action,
        "nearest_window": np.full(n, nearest),
        "fixed_best_on_dev": np.full(n, fixed_dev),
        **constants,
    }
    receipts = commit_geometry(sealed, actions, words, weights, geometry)
    metrics, outcomes = {}, {}
    for policy, selected in actions.items():
        losses, hits, log_true, log_rival, log_marginal, observations, posterior, log_cond = reveal_and_score(
            geometry, sealed, selected, receipts[policy], weights, truth
        )
        if policy == "max_EIG_window":
            m3_error = check_m3_reveal(geometry, weights, selected, observations, posterior, log_cond, log_marginal)
        expected = eig[rows, selected]
        realized = np.log(2.) - losses
        outcomes[policy] = np.column_stack((-losses, hits))
        metrics[policy] = {
            **score(losses, hits), "commit_sha256": receipts[policy],
            "selected_counts": np.bincount(selected, minlength=len(constants)).tolist(),
            "precommitted_true_h_observation_nll": float(-log_true.mean()),
            "precommitted_rival_observation_nll": float(-log_rival.mean()),
            "precommitted_mixture_observation_nll": float(-log_marginal.mean()),
            "expected_eig": float(expected.mean()), "realized_gain": float(realized.mean()),
            "realized_minus_expected": paired_ci(realized - expected),
        }
    adaptive = actions["max_EIG_window"]
    conditional_means = eig.mean(axis=0)
    expected_regret = eig[rows, adaptive].mean() - conditional_means.max()
    assert expected_regret > 0 and np.unique(adaptive).size == len(constants)
    assert np.max(np.abs(eig - np.clip(eig, 0, np.log(2)))) < 1e-8
    return {
        "family": geometry.name, "old_times": geometry.old_time.tolist(),
        "candidate_times": geometry.candidate_time.tolist(),
        "candidate_complex_variance": geometry.candidate_variance,
        "fixed_dev_candidate_index": fixed_dev, "nearest_candidate_index": nearest,
        "dev_mean_conditional_eig_by_candidate": dev_eig.mean(axis=0).tolist(),
        "dev_post_reveal_nll_by_candidate": dev_nll,
        "dev_constant_precommit_sha256": dev_receipts,
        "test_mean_conditional_eig_by_candidate": conditional_means.tolist(),
        "conditional_eig_adaptive_advantage_over_best_constant": float(expected_regret),
        "argmax_matches_latent_geometry_rate": float(np.mean(adaptive == state // 4)),
        "max_quadrature_refinement_difference_first_64": grid_delta,
        "m3_shared_pool_eight_selected_rows_max_abs_error": m3_error,
        "policy_metrics": metrics,
    }, outcomes


def simultaneous_intervals(by_family, controls):
    """Paired bounded-sample scores with large-sample stratified t intervals."""
    family_count = len(by_family)
    sample_size = len(by_family[0][1]["max_EIG_window"])
    endpoints = ("nll_gain_nats", "accuracy_gain")
    fixed_actions = tuple(k for k in by_family[0][1] if k.startswith("constant_"))
    total_intervals = len(endpoints) * (len(controls) + family_count * len(fixed_actions))
    critical = student_t.ppf(1 - .05 / (2 * total_intervals), sample_size - 1)

    def contrasts(action, datasets):
        paired = [scores["max_EIG_window"] - scores[action] for scores in datasets]
        mean = np.mean([d.mean(axis=0) for d in paired], axis=0)
        se = np.sqrt(np.sum([d.var(axis=0, ddof=1) / len(d) for d in paired], axis=0)) / len(paired)
        return {key: {"gain": float(m), "simultaneous_95ci": [float(m - critical*s), float(m + critical*s)]}
                for key, m, s in zip(endpoints, mean, se)}

    fixed_by_family = {
        result["family"]: {action: contrasts(action, [scores]) for action in fixed_actions}
        for result, scores in by_family
    }
    comparisons = {policy: contrasts(policy, [scores for _, scores in by_family]) for policy in controls}
    oracle_chosen = {key: {} for key in endpoints}
    oracle_score, oracle_gain = {}, {}
    for index, key in enumerate(endpoints):
        per_family = []
        for result, scores in by_family:
            family = result["family"]
            choice = max(fixed_actions, key=lambda a: scores[a][:, index].mean())
            oracle_chosen[key][family] = int(choice.rsplit("_", 1)[1])
            per_family.append(scores[choice][:, index])
        oracle_score[key] = float(np.concatenate(per_family).mean() * (-1 if index == 0 else 1))
        # Every endpoint-wise fixed choice is covered simultaneously, including the test oracle.
        per_family = [[item[key] for item in fixed.values()] for fixed in fixed_by_family.values()]
        oracle_gain[key] = {
            "gain": float(np.mean([min(item["gain"] for item in group) for group in per_family])),
            "simultaneous_95ci": [
                float(np.mean([min(item["simultaneous_95ci"][side] for item in group) for group in per_family]))
                for side in (0, 1)
            ],
        }
    comparisons["fixed_best_oracle"] = oracle_gain
    passed = all(entry["simultaneous_95ci"][0] > 0
                 for comparison in comparisons.values() for entry in comparison.values())
    assert passed, "Adaptive selection did not clear the frozen joint gate"
    return {
        "interval_method": "paired stratified t approximation, two-sided Bonferroni over all elementary contrasts",
        "bonferroni_elementary_intervals": total_intervals, "critical_value": float(critical),
        "scope": "independent synthetic mothers, fixed simulator and fixed physical reader; equal family weights",
        "pooled_adaptive_advantages": comparisons,
        "adaptive_vs_every_constant_by_family": fixed_by_family,
        "oracle_fixed_candidate_per_family_and_endpoint": oracle_chosen,
        "oracle_fixed_nll_and_accuracy": {"nll": oracle_score[endpoints[0]], "accuracy": oracle_score[endpoints[1]]},
        "strict_nll_and_accuracy_gate_passed": passed,
    }


def multi_geometry_adaptive_suite(encoder):
    catalogue = geometry_catalogue()
    by_family = [run_geometry(i, model, encoder) for i, model in enumerate(catalogue)]
    report = simultaneous_intervals(by_family, ADAPTIVE_CONTROLS)
    policies = ("max_EIG_window", *ADAPTIVE_CONTROLS)
    pooled = {}
    for policy in policies:
        draws = np.concatenate([scores[policy] for _, scores in by_family])
        pooled[policy] = score(-draws[:, 0], draws[:, 1])
    pooled["fixed_best_oracle"] = report["oracle_fixed_nll_and_accuracy"]
    catalogue_receipt = digest_arrays(*(arr for g in catalogue for arr in
                                        (g.old_time, g.candidate_time, g.old_mean, g.candidate_mean,
                                         np.float64(g.old_variance), np.float64(g.candidate_variance))))
    return {
        "contract": {
            "id": "m5-stage6-blind-v1", "packet": WIRE_ID, "bits_per_old_window": 16, "read_budget": 1,
            "query_utf8": FROZEN_QUERY, "public_families": ADAPTIVE_FAMILIES,
            "matched_candidate_space": "complex matched sum in declared capture units",
            "catalogue_sha256": catalogue_receipt, "candidate_id_order": [0, 1, 2, 3],
            "mixture_node_order": "geometry index then receiver turn; 16 nodes per hypothesis",
            "before_reveal_fields": ["candidate_id", "prior[2]", "conditional_weights[2,16]"],
            "after_reveal_fields": ["posterior[2]"],
            "hosts": ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.8-27B", "LEMMO_matched_adapter"],
            "revision_pins": HOST_REVISIONS,
        },
        "design": {
            "seed": SEED, "rng_namespace": 62, "dev_mothers_per_family": ADAPTIVE_DEV,
            "test_mothers_per_family": ADAPTIVE_TEST,
            "dev_lock": "minimize realized post-reveal NLL on dev within each public family; lowest ID breaks ties",
            "oracle_definition": "retrospective best fixed candidate per public family and per endpoint",
            "nearest_anchor": "latest old timestamp, smallest candidate ID breaks ties",
            "old_hypothesis_probability": [.5, .5],
        },
        "refpool": {
            "k1_reveal": "one immutable acquisition per policy; exact duplicate and prior-ID invariance checked",
            "correlated_gaussian_fixture": correlated_innovation_fixture(),
            "m3_shared_pool_code_sha256": hashlib.sha256(Path(m3_pool.__file__).read_bytes()).hexdigest(),
        },
        "aggregate_post_reveal": pooled,
        "per_family": [result for result, _ in by_family],
        "paired_gate": report,
    }


def main():
    checkpoint = MATCHED / "weights" / "equivariant_b8_s17.npz"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA
    encoder = load(checkpoint)
    assert encoder.equivariant and len(encoder.centers) == 4
    wire = wire_check(encoder)
    adaptive = multi_geometry_adaptive_suite(encoder)
    witness = collision_witness(encoder)
    inner_witness = legacy_inner_witness(encoder)
    tracks = make_tracks()
    all_profiles, all_losses = [], []
    for key, profile in enumerate(PROFILES):
        profile_result, losses = run_profile(key, profile, tracks, encoder)
        all_profiles.append(profile_result)
        all_losses.append(losses)
    tasks = tuple(all_losses[0]["full_iq"])
    pooled = {method: {task: np.concatenate([p[method][task] for p in all_losses]) for task in tasks}
              for method in METHODS}
    comparison = {method: {task: paired_ci(pooled[method][task] - pooled["full_iq"][task]) for task in tasks}
                  for method in METHODS}
    table = {method: {task: float(pooled[method][task].mean()) for task in tasks} for method in METHODS}
    pareto_tasks = ("Q1", "Q2", "stage5_event_and_location", "stage6_post_max_eig")
    rates = np.array([2, 4, 8, 12, 16, np.inf, np.inf])
    objectives = np.array([[table[m][q] for q in pareto_tasks] for m in METHODS])
    finite = rates < np.inf
    nondominated = [
        method for i, method in enumerate(METHODS) if finite[i] and not np.any(
            (rates <= rates[i]) & np.all(objectives <= objectives[i], axis=1)
            & ((rates < rates[i]) | np.any(objectives < objectives[i], axis=1))
        )
    ]
    result = {
        "protocol": {
            "seed": SEED, "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "mothers_per_profile_old": N_TEST, "mothers_per_profile_blind": N_BLIND,
            "s17_calibration_samples_per_canonical_phase_per_distinct_gain_variance_within_profile": N_CAL,
            "s17_checkpoint_sha256": digest, "sample_phase_grid": PHASES,
            "old_times": OLD.tolist(), "payload_windows": OLD[1:3].tolist(),
            "candidate_times": CANDIDATE.tolist(), "jump_times": TAU.tolist(),
            "event_location_labels": BRANCH_NAMES, "phase_slope_grid": [.18, .34],
            "cache_extensions_angle_radius_quality_bits": LAYOUTS,
            "compact_reader": "integrated physical cells; packed legacy inner bits available but ignored",
            "full_iq": "distinct raw likelihood verified against canonical (T,residual) factorization",
            "cache_and_candidate_visibility": "all candidate decisions and predictive densities committed before first reveal",
            "common_ambiguity_subset": "old Full IQ smooth posterior in [0.1, 0.9]; evaluation only",
        },
        "s17_radial_collision_witness": witness,
        "s17_inner_not_determined_by_matched_sum_and_residual_energy": inner_witness,
        "nll_pareto_table": table,
        "finite_pointwise_nondominated_including_rate": nondominated,
        "paired_excess_nll_vs_full_iq": comparison,
        "paired_nll_16_minus_s17": {task: paired_ci(pooled["16"][task] - pooled["4_s17"][task]) for task in tasks},
        "paired_profile_nll_16_minus_s17": {
            p["profile"]: {task: paired_ci(loss["16"][task] - loss["4_s17"][task]) for task in tasks}
            for p, loss in zip(all_profiles, all_losses)
        },
        "per_profile": all_profiles,
        "round2_packet": wire,
        "multi_geometry_adaptive_suite": adaptive,
    }
    output = HERE / "results_m5.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.name}", flush=True)


if __name__ == "__main__":
    main()
