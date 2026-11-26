"""CPU verification of the prior-aware pool, with optional full Round 1 replay."""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.special import logsumexp, roots_hermitenorm, softmax
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "research" / "stage3c" / "A")]

from check_q2 import fit_likelihood, make_bridge
from q2_reader import read
from ref_pool import (
    ClockMeta, GaussianLaw, NodeBank, Observation, PhaseContext,
    RefSufficientPool as ConditionalRefPool, conditional_gaussian, fixed_slots,
    sample_digest, task_readout,
)
from research.matched_cache.learned import load
from research.stage3d_qwen_signed_phase import data_stage3d as data
from research.stage3d_qwen_signed_phase.recompute_all import (
    Q2_LOGICAL_REF_IDX,
    score_vectors,
    work_a_q2_analysis,
)


RHO = 10**0.2
OMEGA = 0.35
H = 2
SIGMA = 0.10
TP = 4.0
CLASSES = np.arange(4)
ROOTS = data.ROOTS
MAX_PILOTS = 64


def rot90(iq, turn):
    """Use sign changes and swaps so the canonical frame is exact in floating point."""
    x, y = iq.unbind(dim=-1)
    orbit = torch.stack((iq, torch.stack((-y, x), -1), -iq, torch.stack((y, -x), -1)), dim=1)
    return orbit[torch.arange(len(iq), device=iq.device), turn % 4]


def as_iq(values):
    return torch.from_numpy(np.stack((values.real, values.imag), axis=-1))


@dataclass(frozen=True)
class Pilot:
    acquisition_id: int
    shares_channel: bool
    t_p: float
    variance: float
    iq: torch.Tensor  # (B, 2) for matched sums or (B, Ns, 2) for raw pilot samples
    matched_amplitude: float = 1.0  # sqrt(Ns) times the per-sample amplitude


@dataclass(frozen=True)
class Pooled:
    iq: torch.Tensor  # (B, 64, 2), derotated and in z1's quarter-turn frame
    times: tuple
    precision: torch.Tensor  # (64,), sum of |matched amplitude|^2 / variance per clock
    gap: torch.Tensor  # (B,), the z2 turn relative to the z1 turn
    family1: torch.Tensor
    family2: torch.Tensor


class RefSufficientPool(nn.Module):
    """Route a shared public design across rows, then pool live floating-point values."""

    def forward(self, z1, z2, refs, omega):
        if len(refs) > MAX_PILOTS:
            raise ValueError("More than 64 supplied references")
        t1, t2 = z1 // 4, z2 // 4
        by_id = {}
        for ref in sorted(refs, key=lambda x: (x.acquisition_id, x.t_p)):
            if not math.isfinite(ref.t_p) or ref.t_p < 0:
                raise ValueError("Expected a finite nonnegative pilot timestamp")
            prev = by_id.get(ref.acquisition_id)
            if prev is None:
                by_id[ref.acquisition_id] = ref
                continue
            if (ref.shares_channel != prev.shares_channel
                    or abs(ref.t_p - prev.t_p) > 1e-12
                    or ref.variance != prev.variance
                    or ref.matched_amplitude != prev.matched_amplitude):
                raise ValueError(f"Conflicting metadata for acquisition_id={ref.acquisition_id}")
            if not torch.equal(ref.iq.detach(), prev.iq.detach()):
                raise ValueError(f"Conflicting samples for acquisition_id={ref.acquisition_id}")
        by_time = {}
        precision = {}
        for acquisition_id in sorted(by_id):
            ref = by_id[acquisition_id]
            if not ref.shares_channel:
                continue
            if not (math.isfinite(ref.variance) and ref.variance > 0
                    and math.isfinite(ref.matched_amplitude) and ref.matched_amplitude > 0):
                raise ValueError("Invalid valid-pilot noise variance or amplitude")
            if (ref.iq.ndim not in (2, 3) or ref.iq.shape[0] != len(z1)
                    or ref.iq.shape[-1] != 2 or not torch.isfinite(ref.iq).all()):
                raise ValueError("Expected finite (B, 2) or (B, Ns, 2) pilot I/Q")
            canon = rot90(ref.iq.to(dtype=torch.float64), -t1)
            if canon.ndim == 3:
                canon = canon.sum(dim=1) / math.sqrt(canon.shape[1])
            weight = ref.matched_amplitude / ref.variance
            by_time.setdefault(ref.t_p, []).append(weight * canon)
            precision[ref.t_p] = precision.get(ref.t_p, 0.0) + weight * ref.matched_amplitude
        times = tuple(sorted(by_time))
        base = torch.zeros((len(z1), 2), dtype=torch.float64, device=z1.device)
        clock_iq = []
        for time in times:
            sx, sy = torch.stack(by_time[time]).sum(dim=0).unbind(dim=-1)
            angle = torch.as_tensor(omega, dtype=base.dtype, device=base.device) * time
            c, s = angle.cos(), angle.sin()
            clock_iq.append(torch.stack((sx * c + sy * s, sy * c - sx * s), dim=-1))
        clock_iq.extend([base] * (MAX_PILOTS - len(times)))
        power = base.new_tensor([precision[t] for t in times] + [0.0] * (MAX_PILOTS - len(times)))
        return Pooled(torch.stack(clock_iq, dim=1), times, power, (t2 - t1) % 4, z1 % 4, z2 % 4)


class FixedSlotQ2(nn.Module):
    """Canonical same-clock Bayes readout. Four probabilities form one fixed slot."""

    def __init__(self, apparent, phase_nodes, weights_given_family2):
        super().__init__()
        self.register_buffer("apparent", torch.from_numpy(apparent))
        self.register_buffer("phase", torch.from_numpy(phase_nodes))
        self.register_buffer("log_weights", torch.from_numpy(np.log(weights_given_family2)))
        self.register_buffer("shift", torch.from_numpy((CLASSES[:, None] + CLASSES) % 4))

    def forward(self, evidence):
        if not evidence.times:
            return evidence.iq[:, :1, :1].expand(-1, 1, 4) + 0.25
        if evidence.times != (TP,):
            raise ValueError("This calibration uses the pinned 4.0 pilot clock")
        s = evidence.iq[:, 0]
        direction = torch.stack((self.phase.cos(), self.phase.sin()), -1)
        perp = torch.stack((-direction[:, 1], direction[:, 0]), -1)
        x = (2 * s @ direction.T).unsqueeze(-1)
        y = (2 * s @ perp.T).unsqueeze(-1)
        ll = torch.cat((x, y, -x, -y), dim=-1)
        g = torch.softmax(torch.logsumexp(ll + self.log_weights[evidence.family2, :, None], dim=1), -1)
        a = self.apparent[evidence.family1][:, self.shift]
        return (g[:, None, :] * a).sum(dim=-1).unsqueeze(1)


def metrics(probs, labels):
    row = score_vectors(probs, labels)
    return {key: row[key] for key in ("n", "acc", "nll", "brier", "ece")}


def compare(left, right, labels, block_size=64):
    a, b = score_vectors(left, labels), score_vectors(right, labels)
    result = {}
    for metric, key in (("acc", "per_item_hit"), ("nll", "per_item_nll")):
        delta = b[key] - a[key]
        blocks = delta.reshape(-1, block_size).mean(axis=1)
        result[metric] = {
            "delta": float(delta.mean()),
            "block_se": float(blocks.std(ddof=1) / np.sqrt(len(blocks))),
        }
        if metric == "acc":
            result[metric].update(improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()))
    return result


def make_arms(m1, m2, mi):
    p1 = Pilot(0, True, TP, 1 / RHO, as_iq(m1))
    p2 = Pilot(1, True, TP, 1 / RHO, as_iq(m2))
    indep = Pilot(1, False, TP, 1 / RHO, as_iq(mi))
    return dict(none=(), valid=(p1,), duplicate_copy_dedup=(p1, p1),
                duplicate_copy_raw=(p1, p1), independent_only=(indep,),
                valid_plus_independent=(p1, indep), fresh_resample=(p1, p2))


def audit_archive(bundle, kernel, summary, smoke):
    y = bundle["q2__labels"].astype(int)
    z1, z2 = (bundle[f"q2__z{k}"].astype(int) for k in (1, 2))
    assert len(y) == 512 and np.array_equal(np.bincount(y), [128] * 4)
    m1, m2, mi = (bundle[f"q2__matched_{p}"] for p in ("valid1", "valid2", "indep"))
    arms = make_arms(m1, m2, mi)
    iq = {ref: v for ref, v in (("valid1", m1), ("valid2", m2), ("indep", mi))}
    source_samples = {0: m1, 1: m2}
    measured = {}
    for arm, refs in arms.items():
        arrays = [(r, source_samples[r.acquisition_id] if r.shares_channel else mi) for r in refs]
        measured[arm] = np.asarray([read(kernel, a, b, [
            (r.acquisition_id, r.shares_channel, vals[i]) for r, vals in arrays
        ]) for i, (a, b) in enumerate(zip(z1.tolist(), z2.tolist()))])
    saved = bundle["q2__work_a_ref_probs"]
    physical_err = max(float(np.max(np.abs(measured[arm] - saved[idx])))
                       for arm, idx in Q2_LOGICAL_REF_IDX.items())
    assert physical_err < 1e-14
    replay = work_a_q2_analysis(bundle, bundle["temps_q2"], data.CONNECTOR_SEEDS)
    max_summary_diff = max(abs(replay["arms"][host]["by_arm"][arm][metric]
                               - summary["arms"][host]["by_arm"][arm][metric])
                           for host in ("phase", "zero") for arm in arms for metric in ("acc", "nll"))
    assert max_summary_diff < 1e-14
    raw = bundle["q2__raw_ll"]
    by_arm = {arm: i for i, arm in enumerate(data.Q2_BLIND_ARMS)}
    assert np.array_equal(raw[:, :, by_arm["valid"]], raw[:, :, by_arm["duplicate_copy_dedup"]])
    pilot_stats = {
        name: {"mean_abs": float(np.abs(v).mean()), "rms_abs": float(np.sqrt(np.mean(np.abs(v)**2)))}
        for name, v in iq.items()
    }
    normprod = lambda a, b: np.mean(a * b.conj()) / np.sqrt(np.mean(abs(a)**2) * np.mean(abs(b)**2))
    replay_info = {
        "arrays": {key: {"shape": list(bundle[key].shape), "dtype": str(bundle[key].dtype)}
                   for key in bundle.files},
        "labels": np.bincount(y).tolist(),
        "code_counts": {"z1": np.bincount(z1, minlength=16).tolist(),
                        "z2": np.bincount(z2, minlength=16).tolist()},
        "temperatures": bundle["temps_q2"].tolist(),
        "pilot_moments": pilot_stats,
        "pilot_normalized_cross_moment_abs": {
            "valid1_valid2": float(abs(normprod(m1, m2))),
            "valid1_indep": float(abs(normprod(m1, mi))),
        },
        "physical": {arm: metrics(p, y) for arm, p in measured.items()},
        "physical_fresh_minus_valid": compare(measured["valid"], measured["fresh_resample"], y),
        "physical_saved_max_diff": physical_err,
        "host_saved_summary_max_diff": max_summary_diff,
        "host": replay["arms"],
        "host_prefix_audit": {"saved_valid_smoke_length": smoke["scorer_parity"]["Q2_valid"]["prefix_len_tokens"],
                              "learned_tokens_per_ref": 4, "added_ref_text": True,
                              "blind_prefix_lengths_stored": False,
                              "hidden_state_or_position_ablation_available": False},
    }
    return replay_info, (torch.from_numpy(z1), torch.from_numpy(z2), y, arms, measured)


def fit_apparent(encoder):
    blocks = 512
    train, _ = data.generate_paired_cfo_blocks(data.NS_CFO_TRAIN, 0, blocks, H, 2, TP, (OMEGA,),
                                               split_prefix="m3_cfo_train")
    batch = train[OMEGA]
    code = encoder.encode(batch.y1)
    u = (batch.x1 + batch.g) % 4
    counts = np.bincount((code % 4) * 4 + (u - code // 4) % 4, minlength=16).reshape(4, 4)
    a = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 2.0)
    fit = {"namespace": data.NS_CFO_TRAIN, "blind_namespace": data.NS_BLIND_Q2_C4,
           "training_blocks": blocks, "training_mothers": len(code), "orbit_counts": counts.tolist(),
           "z2_phase_calibration": {"seed": 41117, "proposal_windows": 60000},
           "blind_fitting": False}
    return a, fit


def check_pool(pool, reader, blind, pinned):
    z1, z2, y, arms, measured = blind
    pooled = {arm: pool(z1, z2, refs, OMEGA) for arm, refs in arms.items()}
    fitted = {arm: reader(e)[:, 0].numpy() for arm, e in pooled.items()}
    ref_err = float(max(np.max(np.abs(pinned(e)[:, 0].numpy() - measured[arm]))
                        for arm, e in pooled.items()))
    assert ref_err < 1e-13
    checks = {
        "max_diff_valid_duplicate_copy_raw": float(np.max(np.abs(fitted["valid"] - fitted["duplicate_copy_raw"]))),
        "max_diff_valid_valid_plus_independent": float(np.max(np.abs(fitted["valid"] - fitted["valid_plus_independent"]))),
        "max_diff_none_independent_only": float(np.max(np.abs(fitted["none"] - fitted["independent_only"]))),
        "max_rot_diff": 0.0,
        "max_rot_diff_pooled": 0.0,
    }
    for r in range(4):
        turn = torch.full_like(z1, r)
        for arm, refs in arms.items():
            rotated = tuple(replace(ref, iq=rot90(ref.iq, turn)) for ref in refs)
            e_rot = pool((z1 + 4*r) % 16, (z2 + 4*r) % 16, rotated, OMEGA)
            checks["max_rot_diff_pooled"] = max(checks["max_rot_diff_pooled"],
                                                (e_rot.iq - pooled[arm].iq).abs().max().item())
            checks["max_rot_diff"] = max(checks["max_rot_diff"],
                                          np.max(np.abs(reader(e_rot)[:, 0].numpy() - fitted[arm])).item())
    assert checks == dict.fromkeys(checks, 0.0)

    valid, distractor = arms["valid_plus_independent"]
    live = replace(valid, iq=valid.iq.clone().requires_grad_())
    masked = replace(distractor, iq=distractor.iq.clone().requires_grad_())
    objective = lambda refs: reader(pool(z1, z2, refs, OMEGA))[..., 0].sum()
    g_valid, = torch.autograd.grad(objective((live,)), (live.iq,))
    g_both, g_masked = torch.autograd.grad(objective((live, masked)), (live.iq, masked.iq),
                                          allow_unused=True, materialize_grads=True)
    assert torch.equal(g_valid, g_both) and torch.count_nonzero(g_masked) == 0

    # Differentiate the unchanged raw-sample API and exercise padded clock capacity.
    raw = replace(live, iq=(live.iq.detach()[:, None, :] / math.sqrt(3))
                  .expand(-1, 3, -1).clone().requires_grad_())
    g_raw, = torch.autograd.grad(objective((raw,)), (raw.iq,))
    assert torch.isfinite(g_raw).all() and torch.count_nonzero(g_raw) > 0
    n2 = z1[:2]
    base = replace(valid, iq=valid.iq[:2])
    many = tuple(replace(base, acquisition_id=i, t_p=float(i), variance=(i+1)/RHO)
                 for i in range(MAX_PILOTS))
    full = pool(n2, z2[:2], many, OMEGA)
    permuted = pool(n2, z2[:2], many[::-1], OMEGA)
    assert torch.equal(full.iq, permuted.iq) and torch.equal(full.precision, permuted.precision)
    assert full.iq.shape == (2, 64, 2) and len(full.times) == 64
    for count in range(MAX_PILOTS + 1):
        copies = (base,) * count
        slots = reader(pool(n2, z2[:2], copies, OMEGA))
        target = fitted["valid" if count else "none"][:2]
        assert slots.shape == (2, 1, 4)
        assert np.allclose(slots[:, 0].numpy(), target, rtol=2e-15, atol=2e-15)
    hetero = replace(base, acquisition_id=1, variance=3/RHO, matched_amplitude=0.75)
    unequal = pool(n2, z2[:2], (base, hetero), 0.0)
    expected = (RHO + 0.75 / (3/RHO)) * rot90(base.iq, -(n2 // 4))
    assert torch.allclose(unequal.iq[:, 0], expected, rtol=2e-15, atol=2e-15)
    bad = (replace(valid, shares_channel=False), replace(valid, t_p=TP + 0.1),
           replace(valid, variance=2*valid.variance), replace(valid, iq=valid.iq + 1))
    for conflict in bad:
        try:
            pool(z1, z2, (valid, conflict), OMEGA)
        except ValueError as exc:
            assert "Conflicting" in str(exc)
        else:
            raise AssertionError("Acquisition conflict accepted")

    scores = {arm: metrics(p, y) for arm, p in fitted.items()}
    assert scores["none"]["acc"] == 0.25
    assert 0.65 <= scores["valid"]["acc"] < scores["fresh_resample"]["acc"]
    assert scores["fresh_resample"]["acc"] >= 0.73
    return {"metrics": scores, "fresh_minus_valid": compare(fitted["valid"], fitted["fresh_resample"], y),
            "checks": checks, "distractor_grad_max": float(g_masked.abs().max()),
            "valid_grad_diff_when_appending_distractor": float((g_valid - g_both).abs().max()),
            "pinned_pool_max_diff_vs_reader": ref_err, "fixed_slots": 1, "slot_components": 4,
            "tested_reference_counts": list(range(MAX_PILOTS + 1)),
            "tested_clock_capacity": len(full.times), "conflict_assertions": len(bad)}


def bridge_weights(c1, c2, times, sigma, z2_phase_emission, inner_order=20):
    lo, hi = times
    d = (H-lo) / (hi-lo)
    mean = OMEGA*H + (1-d)*c1 + d*c2
    var = sigma**2 * (H-lo) * (hi-H) / (hi-lo)
    nodes, weights = roots_hermitenorm(inner_order)
    values = z2_phase_emission(mean[:, None] + np.sqrt(var)*nodes)
    return np.einsum("qnf,n->qf", values, weights / np.sqrt(2*np.pi))


def multi_post(s, families, a, times, sigma, z2_phase_emission, order):
    """Exact path, exact conditional-on-sum density, and coherent anchor approximation."""
    nodes, w = roots_hermitenorm(order)
    w /= np.sqrt(2*np.pi)
    b1 = np.repeat(nodes * sigma * np.sqrt(times[0]), order)
    b2 = b1 + np.tile(nodes * sigma * np.sqrt(times[1]-times[0]), order)
    weights = (w[:, None] * w).ravel()
    emission = bridge_weights(b1, b2, times, sigma, z2_phase_emission)
    log_prior = np.log(weights[:, None] * emission)
    e1, e2 = np.exp(-1j*b1), np.exp(-1j*b2)
    basis = ROOTS.conj()
    # S_total | phases has complex variance 2*rho and mean rho*(exp(i*b1)+exp(i*b2))*i^g.
    collapsed_mean_penalty = -0.5 * RHO * np.abs(e1+e2)**2
    methods = ("exact", "bayes_given_sum", "naive_common_early_clock")
    out = {key: np.empty((len(s), 4)) for key in methods}
    first, second = s.T
    for start in range(0, len(s), 128):
        sl = slice(start, start+128)
        total = (first[sl]+second[sl])[:, None]
        evidence = (first[sl, None]*e1 + second[sl, None]*e2,
                    0.5*total*(e1+e2),
                    total*e1)
        prior = log_prior[:, families[sl, 1]].T
        # The naive model uses the correct joint (B_1, Z2) marginal.
        for key, ev in zip(methods, evidence):
            base = prior if key != "bayes_given_sum" else prior + collapsed_mean_penalty
            marginal = logsumexp(base[..., None] + 2*(ev[..., None] * basis).real, axis=1)
            g = np.exp(marginal - logsumexp(marginal, axis=1, keepdims=True))
            table = a[families[sl, 0]][:, (CLASSES[:, None]+CLASSES) % 4]
            out[key][sl] = np.einsum("ng,nkg->nk", g, table)
    return out


def multi_draws(encoder, times, n, seed):
    rng = np.random.default_rng(seed)
    labels, x2, g = rng.integers(4, size=(3, n))
    steps = np.diff([0, times[0], H, times[1]])
    unit_walk = np.cumsum(rng.normal(size=(n, 3)) * np.sqrt(steps), axis=1)
    noise = (rng.normal(size=(4, n, 3)) + 1j*rng.normal(size=(4, n, 3))) / np.sqrt(2*RHO)
    z1 = encoder.encode(ROOTS[(labels+g) % 4, None]/np.sqrt(3) + noise[0])
    return labels, x2, g, unit_walk, noise, z1


def multi_evidence(pool, encoder, draws, sigma, times):
    labels, x2, g, unit_walk, noise, z1 = draws
    b1, bh, b2 = (sigma*unit_walk).T
    z2 = encoder.encode((ROOTS[(x2+g) % 4]*np.exp(1j*(OMEGA*H+bh)))[:, None]/np.sqrt(3) + noise[1])
    p1 = (ROOTS[g]*np.exp(1j*(OMEGA*times[0]+b1)))[:, None]/np.sqrt(3) + noise[2]
    p2 = (ROOTS[g]*np.exp(1j*(OMEGA*times[1]+b2)))[:, None]/np.sqrt(3) + noise[3]
    pilots = (Pilot(12, True, times[0], 1/RHO, as_iq(p1)),
              Pilot(80, True, times[1], 1/RHO, as_iq(p2)))
    pooled = pool(torch.from_numpy(z1), torch.from_numpy(z2), pilots, OMEGA)
    canonical = pooled.iq[:, :2].numpy()
    sums = canonical[..., 0] + 1j*canonical[..., 1]
    families = np.stack((z1 % 4, z2 % 4), axis=1)
    return sums, families, labels


def multi_test(encoder, a, marginal):
    times = (1.0, 7.0)
    n = 16384
    seed = 431670
    draws = multi_draws(encoder, times, n, seed)
    # Only the family matters because the second transmitted symbol is uniform.
    angle_grid = np.linspace(-np.pi/4, np.pi/4, 513)
    emission_grid = marginal(angle_grid)[:, :4]

    def z2_phase_emission(angle):
        folded = (angle + np.pi/4) % (np.pi/2) - np.pi/4
        return np.stack([np.interp(folded, angle_grid, emission_grid[:, f])
                         for f in range(4)], axis=-1)

    out = {"pilot_times": times, "mothers_per_sigma": n, "seed": seed,
           "phase_emission_grid_knots": len(angle_grid), "quad_order": 40,
           "crosscheck_order": 64, "by_sigma": {}}
    pool = RefSufficientPool()
    for sigma in (0.05, 0.20, 0.40):
        s, families, labels = multi_evidence(pool, encoder, draws, sigma, times)
        probs = multi_post(s, families, a, times, sigma, z2_phase_emission, 40)
        hard = np.argsort(np.abs(s).sum(axis=1))[-128:]
        check = multi_post(s[hard], families[hard], a, times, sigma, z2_phase_emission, 64)
        err = {name: float(np.max(np.abs(probs[name][hard] - check[name]))) for name in probs}
        assert max(err.values()) < 2e-5
        naive_gap = compare(probs["exact"], probs["naive_common_early_clock"], labels)
        sum_gap = compare(probs["exact"], probs["bayes_given_sum"], labels)
        print(f"sigma={sigma:.2f}: observed naive gap {naive_gap['nll']['delta']:+.8f}, "
              f"observed coarsened gap {sum_gap['nll']['delta']:+.8f}, quadrature {err}", flush=True)
        # Conditional NLL integrates the label against the full fitted-model posterior.
        # It complements the noisy observed-label average, especially at low diffusion.
        full = probs["exact"]
        log_full = np.log(full)
        model_nll = {key: float(-np.mean(np.sum(full*np.log(p), axis=1))) for key, p in probs.items()}
        model_gaps = {}
        for key in ("bayes_given_sum", "naive_common_early_clock"):
            kl = (full*(log_full - np.log(probs[key]))).sum(axis=1)
            blocks = kl.reshape(-1, 64).mean(axis=1)
            model_gaps[key] = {"delta": float(kl.mean()),
                               "block_se": float(blocks.std(ddof=1)/np.sqrt(len(blocks)))}
            assert model_gaps[key]["delta"] > 0
        # Exhibit equal-total pilot vectors with distinct full posterior and identical coarsened posterior.
        witness_s = np.array([[1+1j, -1-1j], [-1-1j, 1+1j]])
        witness_fam = np.repeat(families[:1], 2, axis=0)
        witness = multi_post(witness_s, witness_fam, a, times, sigma, z2_phase_emission, 40)
        witness_diff = float(np.max(np.abs(witness["exact"][0]-witness["exact"][1])))
        assert witness_diff > 0 and np.array_equal(witness["bayes_given_sum"][0], witness["bayes_given_sum"][1])
        out["by_sigma"][str(sigma)] = {
            "increment_var": sigma**2*(times[1]-times[0]),
            "metrics": {key: metrics(p, labels) for key, p in probs.items()},
            "naive_minus_exact": naive_gap, "bayes_given_sum_minus_exact": sum_gap,
            "full_model_conditional_nll": model_nll,
            "full_model_conditional_nll_gaps": model_gaps,
            "quadrature_max_diff_top_128_by_pilot_magnitude": err,
            "witness_equal_sum_posterior_max_diff": witness_diff,
        }
    # One independent low-diffusion follow-up at fixed N after the first noisy estimate.
    # Process in chunks to bound the padded clock tensor and quadrature allocations.
    sigma = 0.05
    low_n = 524288
    low_seed = 431671
    batch = 8192
    low_rng = np.random.default_rng(low_seed)
    batch_probs, batch_labels = {}, []
    largest_error = 0.0
    for _ in range(low_n // batch):
        low = multi_draws(encoder, times, batch, int(low_rng.integers(2**32)))
        s, families, labels = multi_evidence(pool, encoder, low, sigma, times)
        probs = multi_post(s, families, a, times, sigma, z2_phase_emission, 12)
        hard = np.argsort(np.abs(s).sum(axis=1))[-2:]
        check = multi_post(s[hard], families[hard], a, times, sigma, z2_phase_emission, 40)
        largest_error = max(largest_error, max(float(np.max(np.abs(probs[key][hard]-check[key]))) for key in probs))
        batch_labels.append(labels)
        for key, value in probs.items():
            batch_probs.setdefault(key, []).append(value)
    assert largest_error < 2e-7
    labels = np.concatenate(batch_labels)
    probs = {key: np.concatenate(values) for key, values in batch_probs.items()}
    followup = {
        "sigma": sigma, "mothers": low_n, "seed": low_seed, "batch_size": batch, "quad_order": 12,
        "quadrature_max_diff_order_40": largest_error,
        "metrics": {key: metrics(p, labels) for key, p in probs.items()},
        "naive_minus_exact": compare(probs["exact"], probs["naive_common_early_clock"], labels),
        "bayes_given_sum_minus_exact": compare(probs["exact"], probs["bayes_given_sum"], labels),
    }
    print(f"Low-diffusion follow-up: observed naive gap {followup['naive_minus_exact']['nll']['delta']:+.8f}, "
          f"observed coarsened gap {followup['bayes_given_sum_minus_exact']['nll']['delta']:+.8f}", flush=True)
    out["independent_low_diffusion_followup"] = followup
    return out


def runtime_versions():
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "device": "cpu"}


def rerun_round1(blind, kernel):
    weights_path = REPO / "research" / "matched_cache" / "weights" / "equivariant_b8_s17.npz"
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == kernel["checkpoint_sha256"]
    encoder = load(weights_path)
    assert encoder.equivariant and len(encoder.centers) == 4
    a, fit = fit_apparent(encoder)
    kernel_a = np.asarray(kernel["u_given_z1"])
    moment = np.exp(1j*np.asarray(kernel["pilot_phase_nodes"]))
    kernel_eta = np.asarray(kernel["weights_given_z2"])
    fit["min_abs_pinned_first_symbol_harmonic"] = float(np.abs(kernel_a @ ROOTS).min())
    fit["min_abs_pinned_first_pilot_phase_moment"] = float(np.abs(kernel_eta @ moment).min())
    assert min(fit["min_abs_pinned_first_symbol_harmonic"],
               fit["min_abs_pinned_first_pilot_phase_moment"]) > 0
    _, marginal = fit_likelihood(encoder, 60000, 41117)
    phase, phase_weights, _ = make_bridge(marginal, 20)
    surrogate = FixedSlotQ2(a, phase-OMEGA*TP, phase_weights[:4])
    pinned = FixedSlotQ2(np.asarray(kernel["u_given_z1"])[:4],
                         np.asarray(kernel["pilot_phase_nodes"])-OMEGA*TP,
                         np.asarray(kernel["weights_given_z2"])[:4])
    surrogate_scores = check_pool(RefSufficientPool(), surrogate, blind, pinned)
    print("Passed blind fixed-slot and disjoint-calibration checks.", flush=True)
    multi = multi_test(encoder, a, marginal)
    print("Passed paired two-timestamp Brownian checks.", flush=True)
    return {"runtime": runtime_versions(), "surrogate_fit": fit, "surrogate": surrogate_scores,
            "multi_timestamp": multi}


def normalize_log(log_w):
    return log_w - logsumexp(log_w)


def direct_density(values, mean, covariance):
    """Independent full-block Gaussian oracle for the Schur test."""
    delta = np.asarray(values)-mean
    v = np.einsum("...i,ij,...j->...", delta.conj(), np.linalg.inv(covariance), delta).real
    return -v - len(values)*np.log(np.pi) - np.linalg.slogdet(covariance)[1]


def rejected(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected rejection containing {message}")


def scenario(clocks, frame=0):
    times = tuple(sorted({c.physical_tick for c in clocks} | {4}))
    scaled = np.asarray(times) / max(times)
    angles = np.array(((0.4*scaled, 0.4*scaled + 0.12),
                       (0.4*scaled + 1.7*(scaled > 0), 0.4*scaled + 1.7*(scaled > 0)-0.16)))
    phase = np.stack([angles[..., times.index(c.physical_tick)] for c in clocks], axis=-1)
    root = np.array([c.known_root for c in clocks])
    means = ROOTS[None, :, None, None] * np.exp(1j*phase[:, None]) * root
    loading = np.resize(np.array([0.30+0.16j, -0.39+0.34j, 0.45-0.07j, 0.09+0.18j]), len(clocks))
    k = np.diag(np.linspace(0.21, 0.49, len(clocks))) + np.outer(loading, loading.conj())
    k = (k + k.conj().T)/2
    k_h = np.stack((k, k+np.diag(np.linspace(0.03, 0.08, len(clocks)))))
    prior = normalize_log(np.log(np.array([0.55, 0.45])[:, None, None]
                                 * np.tile([0.25, 0.25, 0.25, 0.25], (2, 1))[..., None]
                                 * np.array([0.47, 0.53])))
    return times, angles, means, k_h, prior


def clock_row(name, tick, root=1, available=None, role="reference"):
    return ClockMeta(name, "synthetic:"+name, tick, tick if available is None else available, root, "locked-v2",
                     raw_support=((tick, tick+1),), role=role)


def build_given_history(clocks, y0, frame=0):
    times, phase, means, k, prior = scenario(clocks)
    old_law = GaussianLaw(means[..., :1], k[:, :1, :1])
    old = Observation(clocks[0], y0*ROOTS[frame])
    base = normalize_log(prior + old_law.logpdf([y0]))
    return ConditionalRefPool(NodeBank(("smooth", "jump"), times, phase, base), clocks, means, k, frame), old, prior


def q_readout_fixture(pool):
    # A normalized target table with a categorical apparent first symbol and noisy future sector.
    family = np.array([0.70, 0.13, 0.05, 0.12])
    target = np.full((*pool.initial.shape, 4, 4), 0.02/16)
    angles = pool.phase_grid[..., pool.physical_ticks.index(4)]
    for h, n in np.ndindex(angles.shape):
        for g, u in np.ndindex(4, 4):
            sector = int(np.floor(angles[h, n] * (2/np.pi) + u + 0.5)) % 4
            target[h, g, n, (u-g) % 4, sector] += 0.98*family[u]
    return target


def prefix_fixture(state, joint):
    q2, q3, grid = task_readout(state, joint)
    header, slots = fixed_slots((2, 3, 1), q2, q3)
    # Deterministic fixed-position prefix stand-in with four text and two evidence tokens.
    text_ids = np.array([5, 0, 0, 11, 21, 8], dtype="<i4")
    position_ids = np.arange(6, dtype="<i4")
    attention_mask = np.ones(6, dtype=np.uint8)
    embeds = np.arange(24, dtype="<f4").reshape(6, 4)/16
    embeds[1:3] = slots
    byte_arrays = (header, slots, text_ids, position_ids, attention_mask, embeds)
    return tuple(a.tobytes() for a in byte_arrays), (q2, q3, grid)


def check_correlated_innovations():
    clocks = (clock_row("sync", 0), clock_row("pilot-A", 1, 1j),
              clock_row("pilot-B", 7, -1), clock_row("candidate", 6, -1j))
    y0 = 1.15+0.22j
    pool, old, unconditional = build_given_history(clocks, y0)
    state = pool.start((old,), as_of=0)
    first = Observation(clocks[1], 0.21+0.64j)
    second = Observation(clocks[2], -0.75-0.48j)
    noise = Observation(clock_row("unrelated", 2), 17-81j, shares_channel=False)
    expected = pool.update(state, (first, second), as_of=7)
    assert pool.update(state, (old, None, noise, None), as_of=7) is state
    assert pool.update(expected, (old, first, second, noise, None), as_of=7) is expected
    rejected(lambda: pool.update(state, (replace(old, matched=-old.matched),), 7), "consumed acquisition")
    rejected(lambda: pool.update(state, (first, replace(first, matched=-first.matched)), 7), "Conflicting copies")
    rejected(lambda: pool.update(state, (replace(first, clock=replace(first.clock, provenance="wrong")),), 7),
             "Conflicting metadata")
    rejected(lambda: pool.update(state, (Observation(clock_row("uncalibrated", 1), 1),), 7), "provenance")
    rejected(lambda: pool.update(state, (second,), 4), "unavailable")
    retired = pool.start((replace(old, matched=None, digest=sample_digest(y0)),), as_of=0)
    assert pool.update(retired, (old,), 7) is retired
    rejected(lambda: pool.update(retired, (first,), 7), "correlated Gaussian boundary")
    rejected(lambda: pool.update(retired, (replace(old, matched=-old.matched),), 7), "consumed acquisition")
    # Retirement is allowed at exactly zero regression coefficient.
    diag_k = pool.covariance * np.eye(len(clocks))
    zero_cross = ConditionalRefPool(NodeBank(pool.hypothesis_ids, pool.physical_ticks,
                                            pool.phase_grid, state.base), clocks, pool.means, diag_k)
    without_boundary = zero_cross.start((replace(old, matched=None, digest=sample_digest(y0)),), 0)
    with_boundary = zero_cross.start((old,), 0)
    assert zero_cross.update(without_boundary, (first,), 7).log_weights_by_turn.tobytes() == (
        zero_cross.update(with_boundary, (first,), 7).log_weights_by_turn.tobytes())

    full = np.stack([direct_density([y0, first.matched, second.matched],
                                    pool.means[h, ..., :3], pool.covariance[h, :3, :3]) for h in range(2)])
    old_only = GaussianLaw(pool.means[..., :1], pool.covariance[:, :1, :1]).logpdf([y0])
    ordered_ids, innovation = pool.innovation(state, ("pilot-B", "pilot-A"), 7)
    assert ordered_ids == ("pilot-A", "pilot-B")
    innovation_ll = innovation.logpdf([first.matched, second.matched])
    factor_error = float(np.max(np.abs(full - old_only - innovation_ll)))
    state_error = float(np.max(np.abs(normalize_log(unconditional+full) - expected.log_weights_by_turn)))
    assert max(factor_error, state_error) < 1e-13

    # Expansion in the Gaussian sufficient statistics W* S^-1 r and W* S^-1 W.
    k = pool.covariance[0, :3, :3]
    reg = k[1:, :1] @ np.linalg.inv(k[:1, :1])
    design = np.diag([c.known_root for c in clocks[:3]])
    w = design[1:] - reg @ design[:1]
    s = innovation.covariance[0]
    r = np.array([first.matched, second.matched])-reg[:, 0]*y0
    path = pool.means[0, ..., :3] / np.diag(design)
    natural = w.conj().T @ np.linalg.solve(s, r)
    gram = w.conj().T @ np.linalg.solve(s, w)
    kern = 2*np.real(path.conj() @ natural) - np.einsum("...i,ij,...j->...", path.conj(), gram, path).real
    common = direct_density(r, np.zeros_like(r), s)
    suff_error = float(np.max(np.abs(kern + common - innovation_ll[0])))
    assert suff_error < 1e-13

    # Gaussian overcount and a categorical example with increasing posterior variance.
    p0, noise_var = 1.25, 0.4
    p, wrong = (1/(1/p0+m/noise_var) for m in (1, 2))
    regret = np.log(wrong/p)+p/wrong-1+((p-2*wrong)/noise_var)**2*(p0+noise_var)/wrong
    assert wrong < p and regret > 0
    type_vars = {"correct": p, "recounted": wrong, "expected_forward_kl": float(regret),
                 "discrete_variance_once": 0.75*0.25, "discrete_variance_twice": 0.25}

    joint = q_readout_fixture(pool)
    bytes0, p0s = prefix_fixture(expected, joint)
    variants = [(first, second, first), (first, noise, second), (None, first, second, None),
                (second, first), (None, noise, second, first, old, None, first), (first, second, *(None,)*62)]
    kl, tv = [], []
    for refs in variants:
        state_v = pool.update(state, refs, 7)
        payload, p_v = prefix_fixture(state_v, joint)
        assert payload == bytes0
        for a, b in zip(p0s, p_v):
            kl.append(float(np.sum(a * (np.log(a)-np.log(b)))))
            tv.append(float(0.5*np.abs(a-b).sum()))
    sequential = pool.update(pool.update(state, (second,), 7), (first,), 7)
    assert prefix_fixture(sequential, joint)[0] == bytes0
    after_first = pool.update(state, (first,), 7)
    conditional_second = pool.forecast(after_first, second.acquisition_id, 7)
    _, predicted_evidence, _ = conditional_second.log_densities([second.matched])
    conditional_error = float(abs(predicted_evidence - (expected.log_increment-after_first.log_increment)))
    assert conditional_error < 2e-14
    assert prefix_fixture(pool.reveal(after_first, conditional_second, second, 7), joint)[0] == bytes0
    for frame in range(4):
        rotated, old_r, _ = build_given_history(clocks, y0, frame)
        result = rotated.update(rotated.start((old_r,), 0), (
            replace(first, matched=first.matched*ROOTS[frame]), replace(second, matched=second.matched*ROOTS[frame])
        ), 7)
        assert prefix_fixture(result, joint)[0] == bytes0

    hardware = {}
    for name, stop, ticks in (("Pluto", 13, (15, 18)), ("CTTC", 6, (126, 127))):
        sync = tuple(Observation(clock_row(f"{name}:sync:{j}", j, role="history"), y0) for j in range(stop))
        hw_clocks = (sync[-1].clock, *(clock_row(f"{name}:pilot:{tick}", tick, root)
                                       for tick, root in zip(ticks, (1j, -1))))
        hw, _, _ = build_given_history(hw_clocks, y0)
        base = hw.start(sync, as_of=stop-1)
        refs = (Observation(hw_clocks[1], first.matched), Observation(hw_clocks[2], second.matched))
        rejected(lambda: hw.update(hw.start(sync[:-1], stop-1), (sync[-1],), ticks[-1]),
                 "History ID missing")
        updated = hw.update(base, refs, as_of=ticks[-1])
        assert hw.update(base, sync, as_of=ticks[-1]) is base
        assert hw.update(updated, (*refs, *sync), as_of=ticks[-1]) is updated
        hardware[name] = {"baseline_ids": len(base.history), "new_ids": len(updated.added),
                          "pilot_ticks": ticks, "test_fixture_only": True}
    return {
        "full_vs_conditional_log_likelihood_max_diff": factor_error,
        "full_vs_conditional_log_posterior_max_diff": state_error,
        "sufficient_quadratic_max_diff": suff_error,
        "forecast_after_first_vs_joint_increment_diff": conditional_error,
        "gaussian_recount_witness": type_vars,
        "no_new_records_identical_state": True, "byte_identical_prefix_fixture": True,
        "byte_variants": len(variants), "bytes_per_array": [len(b) for b in bytes0],
        "max_kl": max(kl), "max_tv": max(tv), "byte_identical_reversed_sequential_reveals": True,
        "byte_identical_canonical_joint_rotations": 4, "hardware_roles": hardware,
    }


def q_turn_from_law(law, values, weights):
    ell = law.logpdf(values)
    return softmax(logsumexp(ell+weights, axis=(-1, -3)), axis=-1)


def gaussian_nodes(mean, cov, order):
    x, w = roots_hermitenorm(order)
    innovations = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    return mean + innovations @ np.linalg.cholesky(cov).T, (w[:, None]*w).ravel()/(2*np.pi)


def check_cfo_brownian():
    times = (1, 7)
    ctx = PhaseContext(0, 1, 0.35, 0.024, 0.0484)
    mean, cov = ctx.path_moments(times)
    lag = times[1]-times[0]
    difference_variance = cov[0, 0]+cov[1, 1]-cov[0, 1]-cov[1, 0]
    assert np.isclose(difference_variance, ctx.diffusion_var_per_unit*lag + ctx.v_omega*lag**2)
    v = np.asarray(times, dtype=float)
    c_b = ctx.diffusion_var_per_unit*np.minimum.outer(v, v)
    q = np.linalg.inv(c_b) @ v
    woodbury = np.linalg.inv(c_b) - np.outer(q, q)/(1/ctx.v_omega+v@q)
    sr_error = float(np.max(np.abs(woodbury-np.linalg.inv(cov))))
    assert sr_error < 2e-13
    _, anchored = replace(ctx, p_phi0=0.18, cov_phi0_omega=-0.03).path_moments((0, *times))
    assert np.isclose(anchored[0, -1], 0.18 - 0.03*times[-1])
    epoch = 2**55
    _, precise = replace(ctx, anchor_tick=epoch).path_moments([epoch+t for t in times])
    assert np.array_equal(precise, cov)

    k = np.array([[.55, .13+.08j, .09-.03j],
                  [.13-.08j, .55, .21+.09j],
                  [.09+.03j, .21-.09j, .58]])
    y0 = .47+.13j
    g_prior = softmax(-np.abs(y0-ROOTS)**2/k[0, 0].real)
    pilot_symbols = np.array([1j, -1])
    design = np.r_[1, pilot_symbols]
    projection = np.exp(-1j*mean)*pilot_symbols.conj()/k.diagonal().real[1:]

    def hypothesis_law(path):
        all_angles = np.column_stack((np.zeros(len(path)), path))
        means = ROOTS[None, :, None, None]*(design*np.exp(1j*all_angles))[None, None]
        return conditional_gaussian(k[None], means, [1, 2], [0], [y0])

    seed, n = 431672, 4096
    rng = np.random.default_rng(seed)
    g = rng.choice(4, n, p=g_prior)
    actual = rng.multivariate_normal(mean, cov, size=n)
    dynamic = hypothesis_law(actual)
    noise = (rng.normal(size=(n, 2))+1j*rng.normal(size=(n, 2)))/np.sqrt(2)
    samples = dynamic.mean[0, g, np.arange(n)] + noise @ np.linalg.cholesky(dynamic.covariance[0]).T
    def solve(order, sample):
        grid, w = gaussian_nodes(mean, cov, order)
        full = hypothesis_law(grid)
        summed = full.project(projection)
        coherent_grid = np.column_stack((grid[:, 0], grid[:, 0]+mean[1]-mean[0]))
        naive = hypothesis_law(coherent_grid).project(projection)
        w_full = np.log(g_prior[None, :, None]*w)
        cfo_free_grid, w_b = gaussian_nodes(mean, c_b, order)
        return {
            "full": q_turn_from_law(full, sample, w_full),
            "bayes_given_sum": q_turn_from_law(summed, (sample @ projection)[:, None], w_full),
            "naive_coherent_sum": q_turn_from_law(naive, (sample @ projection)[:, None], w_full),
            "brownian_only": q_turn_from_law(hypothesis_law(cfo_free_grid), sample,
                                            np.log(g_prior[None, :, None]*w_b)),
        }
    # Chunked to keep the full two-clock lattice and six complex work arrays bounded.
    parts = [solve(32, samples[i:i+128]) for i in range(0, n, 128)]
    p = {key: np.concatenate([x[key] for x in parts]) for key in parts[0]}
    hi = np.argsort(np.abs(samples).sum(1))[-64:]
    check = solve(48, samples[hi])
    quad_error = max(float(np.max(np.abs(p[key][hi]-check[key]))) for key in p)
    scores = {key: metrics(val, g) for key, val in p.items()}
    comp = {key: compare(p["full"], val, g) for key, val in p.items() if key != "full"}
    assert comp["naive_coherent_sum"]["nll"]["delta"] > 0.01
    assert comp["bayes_given_sum"]["nll"]["delta"] > 0.01
    assert comp["brownian_only"]["nll"]["delta"] > 0
    assert quad_error < 2e-4
    thermal = dynamic.covariance[0]
    return {"n": n, "seed": seed, "clocks": times, "covariance": cov.tolist(),
            "diffusion_var_per_unit": ctx.diffusion_var_per_unit, "v_omega": ctx.v_omega,
            "difference_variance": float(difference_variance), "sherman_morrison_max_diff": sr_error,
            "thermal_conditional_covariance_abs": float(abs(thermal[0, 1])),
            "thermal_conditional_correlation_abs": float(abs(thermal[0, 1])/np.sqrt(thermal[0,0].real*thermal[1,1].real)),
            "quadrature_order": 32, "check_order": 48, "quadrature_max_prob_diff": quad_error,
            "target": "carrier quarter-turn G given fixed F0 sync IQ",
            "metrics": scores, "paired_against_full": comp}


def check_stage6():
    clocks = (clock_row("sync", 0), clock_row("alias", 2, 1j, role="probe"),
              clock_row("probe", 7, -1, 9, role="probe"))
    times, angles, means, k, prior = scenario(clocks)
    # Same conditional law under each hypothesis at the aliased candidate, including calibrated K.
    means[1, ..., :2] = means[0, ..., :2]
    k[1] = k[0]
    y0 = 1.31+0.19j
    weights = normalize_log(prior + GaussianLaw(means[..., :1], k[:, :1, :1]).logpdf([y0]))
    open_phase = angles.copy()
    open_phase[1, :, :2] = open_phase[0, :, :2]
    pool = ConditionalRefPool(NodeBank(("smooth", "jump"), times, open_phase, weights), clocks, means, k)
    state = pool.start((Observation(clocks[0], y0),), 0)
    rejected(lambda: pool.forecast(state, "probe", 7), "unavailable")
    rejected(lambda: pool.forecast(state, "sync", 9), "already consumed")
    # The fixture exposes metadata to the predictor and hides measured IQ until a committed read.
    forecasts = {c.acquisition_id: pool.forecast(state, c.acquisition_id, 9) for c in clocks[1:]}
    eig = {key: forecast.eig(24) for key, forecast in forecasts.items()}
    assert abs(eig["alias"]) < 3e-15 and eig["probe"] > 0.1
    key = max(eig, key=eig.get)
    committed = forecasts[key]
    vault = {"alias": -0.7+0.24j, "probe": 0.60-0.61j}
    reading = Observation(clocks[2], vault.pop(key))
    rejected(lambda: pool.reveal(state, committed, reading, 8), "unavailable")
    rejected(lambda: pool.reveal(state, committed, reading, 10), "commitment")
    after = pool.reveal(state, committed, reading, 9)
    rejected(lambda: pool.reveal(after, committed, reading, 9), "commitment")
    assert pool.update(after, (reading,), 9) is after
    threshold = 4
    assert reading.clock.physical_tick > threshold and len(vault) == 1

    f, m, posterior = committed.log_densities([reading.matched])
    log_h0 = logsumexp(state.log_weights_by_turn, axis=(-2, -1))
    posterior_error = float(np.max(np.abs(posterior-logsumexp(after.log_weights_by_turn, axis=(-2, -1)))))
    z_error = float(abs(m - (after.log_increment-state.log_increment)))
    assert max(posterior_error, z_error) < 2e-14

    axis = np.linspace(-6, 6, 181)
    grid = (axis[:, None]+1j*axis).reshape(-1, 1)
    area = (axis[1]-axis[0])**2
    log_f_grid, log_m_grid, log_post_grid = committed.log_densities(grid)
    mass_h = np.exp(log_f_grid).sum(axis=0)*area
    mass_m = np.exp(log_m_grid).sum()*area
    marginalized = np.sum(np.exp(log_post_grid+log_m_grid[..., None]), axis=0)*area
    grid_eig = np.sum(np.exp(log_post_grid+log_m_grid[..., None])*(log_post_grid-log_h0))*area
    eig_error = abs(grid_eig-eig[key])
    assert np.max(np.abs(mass_h-1)) < 2e-11 and abs(mass_m-1) < 2e-11
    assert np.max(np.abs(marginalized-np.exp(log_h0))) < 2e-11 and eig_error < 3e-5
    return {"candidates": len(forecasts), "eig_nats": eig, "selected": key,
            "post_target_clock": threshold, "selected_physical_tick": reading.clock.physical_tick,
            "last_reveal_tick": reading.clock.available_tick, "observed_reads": 1,
            "hypothesis_density_mass": mass_h.tolist(), "mixture_density_mass": float(mass_m),
            "prior_h": np.exp(log_h0).tolist(), "posterior_h": np.exp(posterior).tolist(),
            "forecast_log_conditional_at_reveal": f.tolist(), "forecast_log_marginal_at_reveal": float(m),
            "pre_post_log_evidence_max_diff": z_error, "pre_post_log_posterior_max_diff": posterior_error,
            "quadrature_vs_grid_eig_diff": float(eig_error)}


def check_pinned_c4_adapter(blind, kernel):
    z1, z2, _, arms, archived = blind
    idx1, idx2 = z1.numpy(), z2.numpy()
    clocks = (clock_row("c4:pilot1", 4), clock_row("c4:pilot2", 4))
    phase = np.asarray(kernel["pilot_phase_nodes"])[None, :, None]
    means = np.repeat(ROOTS[None, :, None, None]*np.exp(1j*phase[:, None]), 2, axis=-1)
    k = np.eye(2)/kernel["rho"]
    a = np.asarray(kernel["u_given_z1"])
    phase_w = np.asarray(kernel["weights_given_z2"])
    pools = [ConditionalRefPool(NodeBank(("C4",), (4,), phase,
                                         np.log(np.repeat(w[None, None, :]/4, 4, axis=1))),
                                 clocks, means, k) for w in phase_w]
    initial = [pool.start((), 0) for pool in pools]
    y = [ref.iq[:, 0].numpy()+1j*ref.iq[:, 1].numpy() for ref in arms["fresh_resample"]]
    errors = dict(none=0.0, valid=0.0, fresh_resample=0.0)
    rel = (CLASSES[:, None]+CLASSES) % 4
    for i, (a1, a2) in enumerate(zip(idx1, idx2)):
        row = tuple(Observation(c, v[i]) for c, v in zip(clocks, y))
        for count, name in enumerate(errors):
            state = pools[a2].update(initial[a2], row[:count], 4)
            carrier = np.exp(logsumexp(state.log_weights_by_turn, axis=(0, 2)))
            p = a[a1, rel] @ carrier
            errors[name] = max(errors[name], float(np.max(np.abs(p-archived[name][i]))))
    assert max(errors.values()) < 2e-14
    return {"mothers": len(idx1), "max_prob_diff_vs_q2_reader": errors}


def contrast_native_scores(bundle, blind, result_dir):
    gamma = 0.15  # M1 bounded residual budget, fixed before this comparison.
    locked = bundle["temps_q2"][0]
    y = blind[2]
    physical = blind[4]
    raw = bundle["q2__raw_ll"][0]
    for seed, temp in zip(data.CONNECTOR_SEEDS, locked):
        dev = json.loads((result_dir/f"train_summary_phase_s{seed}.json").read_text(encoding="utf-8"))
        assert dev["temp_q2"] == temp
    out = {"original_dev_locked_temperatures": locked.tolist(), "residual_limit": gamma,
           "scope": "offline contrast with native Stage 3D candidate scores; no Round 2 Qwen forward pass",
           "additional_temperature_or_blind_fit": False, "by_arm": {}}
    for idx, arm in enumerate(data.Q2_BLIND_ARMS):
        logits = raw[:, idx]/locked[:, None, None]
        centered = logits - logits.mean(-1, keepdims=True)
        native = softmax(logits, axis=-1)
        p0 = physical[arm]
        residual = softmax(np.log(p0)+gamma*np.tanh(centered), axis=-1)
        notes = {}
        for label, candidate in (("arm1_archived_native_host", native), ("arm2_offline_physical_residual", residual)):
            notes[label] = metrics(candidate.reshape(-1, 4), np.tile(y, len(locked)))
            base = score_vectors(p0, y)
            augmented = score_vectors(candidate.reshape(-1, 4), np.tile(y, len(locked)))
            d = augmented["per_item_nll"].reshape(len(locked), -1).mean(axis=0)-base["per_item_nll"]
            block = d.reshape(-1, 64).mean(axis=1)
            notes[label]["delta_nll_vs_physical"] = float(block.mean())
            notes[label]["delta_nll_block_se"] = float(block.std(ddof=1)/np.sqrt(len(block)))
        out["by_arm"][arm] = {"q2_reader": metrics(p0, y), **notes}
    return out


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-round1", action="store_true", help="also refresh the slower Round 1 simulations")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = HERE / "results_m3.json"
    receipt = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result_dir = REPO / "research" / "stage3d_qwen_signed_phase" / "results"
    kernel_path = REPO / "research" / "stage3c" / "A" / "q2_s17_kernel.json"
    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "summary_master.json").read_text(encoding="utf-8"))["work_a_q2_c4"]
    smoke = json.loads((result_dir / "smoke_audit.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "blind_q2_scores.npz", allow_pickle=False) as bundle:
        archive, blind = audit_archive(bundle, kernel, summary, smoke)
        contrast = contrast_native_scores(bundle, blind, result_dir)
    receipt.update(condition=summary["condition"], stage3d=archive,
                   kernel_fields={k: list(np.shape(v)) if isinstance(v, list) else v for k, v in kernel.items()})
    innovation = check_correlated_innovations()
    print("Passed Schur, provenance, and byte-identity checks.", flush=True)
    multi = check_cfo_brownian()
    print("Passed shared-CFO and Brownian verification.", flush=True)
    stage6 = check_stage6()
    print("Passed Stage 6 predictive and reveal verification.", flush=True)
    receipt["round2"] = {
        "runtime": runtime_versions(),
        "innovation": innovation, "shared_cfo_multiclock": multi, "stage6": stage6,
        "c4_adapter": check_pinned_c4_adapter(blind, kernel),
        "dev_locked_contrasts": contrast, "full_round1_replayed": args.rerun_round1,
    }
    receipt["round2"]["core_seconds"] = time.perf_counter()-started
    if args.rerun_round1:
        receipt.update(rerun_round1(blind, kernel))
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
