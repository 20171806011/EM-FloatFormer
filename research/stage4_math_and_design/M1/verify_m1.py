"""Stage 4 M1 matched-host ablations and physical interface checks."""

import json
import math
from fractions import Fraction
from pathlib import Path
import platform
import sys
from typing import NamedTuple

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS, DEV_OMEGAS, NS_CFO_DEV, NS_CFO_TRAIN, TRAIN_OMEGAS, TRAIN_TIMINGS,
    generate_paired_cfo_blocks, load_s17_encoder, public_phase_slots,
)
from research.stage3d_qwen_signed_phase.recompute_all import (
    compute_16way_group_avg, compute_representative_readout, paired_t_ci,
    score_vectors, softmax_temperature, work_f_sign_probe, work_g_rotation_and_controls,
)
from research.stage4_math_and_design.M5.verify_m5 import encode as m5_encode, quality_edges, radial_edges

STAGE3D_RESULTS = ROOT / "research/stage3d_qwen_signed_phase/results"
MODEL_SEEDS = (202611, 202622, 202633)
TRAIN_BLOCKS = 32
DEV_BLOCKS = 16
EPOCHS = 16
BATCH_SIZE = 256
LEARNING_RATE = 0.01
WIDTH = 24
HIDDEN = 48
HOST_SEED = 123_457
SHUFFLE_SEED = 407_913
RESIDUAL_LIMIT = 0.15
MARGIN_RADIUS = 0.10
TASKS = ("q1", "q2", "q3_repeat", "q3_indep", "q3_after_pilot")
ARM0 = "arm0_separate_nonquotient"
ARM1 = "arm1_equivariant_host_pure"
ARM2 = "arm2_equivariant_physical_residual"
ARM3 = "arm3_physical_only"
CONTROL_Q = "control_quotient_separate"
CONTROL_C = "control_nonquotient_coupled"
CONTROL_ZERO = "control_arm1_zero_slots_refit"


def lift_scores(canonical, codes, task, pilot_turn=0, receive_gauge=False, tx_chart_turn=0):
    """Map normalized canonical scores into the task's physical option order."""
    a, b = (codes // 4).unbind(dim=-1)
    if task == "q1":
        shift = torch.zeros_like(a) if receive_gauge else b - a
    elif task == "q2":
        shift = torch.zeros_like(a) + tx_chart_turn if receive_gauge else a - pilot_turn
    elif task in ("q3_repeat", "q3_after_pilot"):
        shift = a
    elif task == "q3_indep":
        return torch.full_like(canonical, 0.25)
    else:
        raise ValueError(f"Unknown task {task}")
    index = (torch.arange(4, device=canonical.device) - shift[:, None]) % 4
    return canonical.gather(-1, index)


class SectorTransport(nn.Module):
    """Integrate the AWGN sector map against the signed Brownian propagator."""

    def __init__(self):
        super().__init__()
        x, w = np.polynomial.hermite.hermgauss(64)
        self.register_buffer("nodes", torch.from_numpy(x))
        self.register_buffer("weights", torch.from_numpy(w / math.sqrt(math.pi)))

    def forward(self, theta, variance, snr_db):
        angle = theta[..., None] + (2.0 * variance[..., None]).sqrt() * self.nodes
        scale = (2.0 * (10.0 ** (snr_db[..., None] / 10.0))).sqrt()
        c = torch.special.ndtr(scale * (angle + math.pi / 4).cos())
        s = torch.special.ndtr(scale * (angle + math.pi / 4).sin())
        base = torch.stack((c * s, (1 - c) * s, (1 - c) * (1 - s), c * (1 - s)), dim=-1)
        sector = (self.weights[:, None] * base).sum(dim=-2)
        return torch.stack([sector.roll(j, dims=-1) for j in range(4)], dim=-2)


class EvidencePacket(NamedTuple):
    codes: torch.Tensor
    tail: torch.Tensor
    valid: torch.Tensor


def decode_windows(words, bits, valid=None):
    """Decode the bit-exact S17 prefix and optional invariant M5 fields."""
    if bits not in (4, 16) or words.is_floating_point() or words.is_complex() or words.dtype == torch.bool:
        raise ValueError("Expected integer 4-bit or 16-bit window words")
    words = words.to(torch.int64)
    if words.ndim != 2 or words.shape[-1] != 2 or bool(((words < 0) | (words >= 2**bits)).any()):
        raise ValueError("Expected two in-range words per item")
    valid = torch.ones_like(words, dtype=torch.bool) if valid is None else valid
    if valid.shape != words.shape or valid.dtype != torch.bool:
        raise ValueError("Validity mask must be boolean and match the window words")
    words = torch.where(valid, words, 0)
    if bits == 4:
        tail = torch.zeros((*words.shape, 6), dtype=torch.float32, device=words.device)
        return EvidencePacket(words, tail, valid)
    angle = (((words >> 6) & 63) + 0.5) * (math.pi / 128) - math.pi / 4
    # Summarize bins by log-midpoint, with a separate saturation indicator.
    # Open-ended cells remain intervals in a calibrated likelihood reader.
    rad = (words >> 2) & 15
    qual = words & 3
    log_edges = torch.as_tensor(np.log1p(radial_edges(4)[:-1]), dtype=torch.float32, device=words.device)
    log_edges = torch.cat((log_edges, log_edges.new_tensor([math.log(25)])))
    log_rad = (log_edges[rad] + log_edges[rad + 1]) / 2
    quality_low = torch.as_tensor(np.log1p(quality_edges(2)[:-1]), dtype=torch.float32, device=words.device)[qual]
    tail = torch.stack((angle.cos(), angle.sin(), log_rad, quality_low,
                        (rad == 15) | (qual == 3), torch.ones_like(angle)), dim=-1)
    return EvidencePacket(words >> 12, torch.where(valid[..., None], tail, 0), valid)


class FrozenHost(nn.Module):
    """Frozen MLP and a random regular-simplex candidate projection."""

    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.register_buffer("text", torch.randn(1, 1, WIDTH) * 0.2)
            self.body = nn.Sequential(nn.Linear(4 * WIDTH, 64), nn.GELU())
            frame, r = torch.linalg.qr(torch.randn(64, 3, dtype=torch.float64))
            frame = frame * torch.diagonal(r).sign()
            helmert = torch.tensor(
                [[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=torch.float64,
            ) / torch.tensor([2, 6, 12], dtype=torch.float64).sqrt()
            self.register_buffer("w_out", (helmert @ frame.T).float())
        self.requires_grad_(False)
        self.eval()

    def forward(self, tokens, public):
        # The immutable prompt embedding includes true public metadata. A
        # counterfactual replaces the numeric phase slots, leaving this fixed.
        text_numbers = public[:, :6] * public.new_tensor([0.25, 0.25, 0.25, 2., 5., 0.25])
        text = self.text + F.pad(text_numbers, (0, WIDTH - 6))[:, None, :]
        return F.linear(self.body(torch.cat((tokens, text), dim=1).flatten(1)), self.w_out)


class EquivariantEvidenceConnector(nn.Module):
    """Shared token architecture for the 2×2 factorial and the residual arm."""

    def __init__(self, host, prior, *, quotient, coupled, residual=False, zero_slots=False):
        super().__init__()
        self.host = host
        self.quotient = quotient
        self.coupled = coupled
        self.residual = residual
        self.zero_slots = zero_slots
        self.register_buffer("prior", torch.as_tensor(prior[:4], dtype=torch.float32))
        powers = ROOTS[None, :] ** np.arange(1, 4)[:, None]
        self.register_buffer("fourier_basis", torch.from_numpy(powers).to(torch.complex64))
        self.code_embedding = nn.Embedding(16, 16)
        self.window_hidden = nn.Linear(64 + 2 * (4 + 6 + 1 + 6 + 2), HIDDEN)
        self.window_out = nn.Linear(HIDDEN, 2 * WIDTH)
        self.phase_hidden = nn.Linear(6, HIDDEN)
        self.phase_out = nn.Linear(HIDDEN, WIDTH)

    def prefix(self, packet, meta, slots):
        codes, tail, valid = packet
        inner = codes % 4
        ids = inner if self.quotient else codes
        first, second = self.code_embedding(ids).unbind(dim=1)
        paired = torch.cat((first, second, first - second, first * second), dim=-1)
        prior = self.prior[inner]
        prior = torch.where(valid[..., None], prior, 0.25)
        if self.zero_slots:
            slots = torch.zeros_like(slots)
        n = torch.arange(1, 4, device=meta.device)
        h, ell, _, _, sigma, _, rate_var = meta[:, 6:].unbind(dim=1)
        dt = torch.stack((h + ell, h), dim=-1)
        variance = sigma[:, None].square() * dt + rate_var[:, None] * dt.square()
        base = torch.stack((torch.complex(slots[:, 2], slots[:, 3]),
                            torch.complex(slots[:, 0], -slots[:, 1])), dim=-1)
        phasors = torch.exp(-0.5 * variance[..., None] * n.square()) * base[..., None].pow(n)
        source = prior.to(phasors.dtype) @ self.fourier_basis.T
        polar = torch.complex(tail[..., 0], tail[..., 1]) * tail[..., 2].expm1()
        product = torch.view_as_real(phasors * source).flatten(-2)
        polar_product = torch.view_as_real(phasors[..., 0] * polar)
        interaction = torch.cat((product, polar_product), dim=-1)
        if not self.coupled:
            interaction = torch.zeros_like(interaction)
        evidence = torch.cat((prior, tail, valid[..., None], interaction), dim=-1)
        wave_hidden = F.gelu(self.window_hidden(torch.cat((paired, evidence.flatten(1)), dim=-1)))
        phase_hidden = F.gelu(self.phase_hidden(slots))
        waves = self.window_out(wave_hidden).reshape(-1, 2, WIDTH)
        tokens = torch.cat((waves, self.phase_out(phase_hidden)[:, None, :]), dim=1)
        return tokens, wave_hidden, phase_hidden

    def canonical_readout(self, packet, meta, kernel, slots, gamma=RESIDUAL_LIMIT):
        tokens, _, _ = self.prefix(packet, meta, slots)
        logits = self.host(tokens, meta[:, 6:])
        if not self.residual:
            return logits.softmax(dim=-1)
        source = self.prior[packet.codes[:, 0] % 4]
        source = torch.where(packet.valid[:, :1], source, 0.25)
        corrected = (source.log() + gamma * logits.tanh()).softmax(dim=-1)
        return torch.bmm(corrected[:, None, :], kernel).squeeze(1)

    def predict(self, packet, meta, kernel, slots=None, gamma=RESIDUAL_LIMIT, receive_gauge=False):
        used_slots = meta[:, :6] if slots is None else slots
        canonical = self.canonical_readout(packet, meta, kernel, used_slots, gamma)
        if not self.quotient:
            return canonical
        return lift_scores(canonical, packet.codes, "q3_repeat", receive_gauge=receive_gauge)

    def forward(self, codes, meta, kernel):
        return self.predict(decode_windows(codes, 4), meta, kernel)

    def orbit_table(self, meta, kernel, slots, gamma=RESIDUAL_LIMIT):
        if not self.quotient:
            return self.predict(decode_windows(ALL_CODES, 4), meta.expand(256, -1),
                                kernel.expand(256, -1, -1), slots.expand(256, -1), gamma)
        canonical = self.canonical_readout(decode_windows(INNER_CODES, 4), meta.expand(16, -1),
                                           kernel.expand(16, -1, -1), slots.expand(16, -1), gamma)
        inner_index = (ALL_CODES[:, 0] % 4) * 4 + ALL_CODES[:, 1] % 4
        return lift_scores(canonical[inner_index], ALL_CODES, "q3_repeat")


def metadata(cond, rate_var=0.0):
    slots = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p)
    scalars = (cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db, rate_var)
    return torch.tensor([*slots, *scalars], dtype=torch.float32)


def physical_kernel(transport, theta, variance, snr_db):
    return transport(*(torch.as_tensor(x, dtype=torch.float64) for x in (theta, variance, snr_db)))


def condition_kernel(transport, cond, rate_var=0.0):
    dt = cond.h + cond.ell
    return physical_kernel(transport, cond.omega * dt, dt * cond.sigma_phi**2 + rate_var * dt**2, cond.snr_db)


def encode_view(encoder, mother, r1, r2):
    y1, y2 = mother.rotated_view(r1, r2)
    return np.column_stack((encoder.encode(y1), encoder.encode(y2)))


def draw_split(encoder, transport, namespace, blocks, omegas, split):
    columns = ([], [], [], [])
    seeds = []
    for i, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        paired, block_seeds = generate_paired_cfo_blocks(
            namespace, i, blocks, h, ell, tp, omegas, split_prefix=split,
        )
        seeds.extend(block_seeds)
        for omega in omegas:
            mother = paired[omega]
            n = len(mother.x1)
            values = (
                torch.from_numpy(encode_view(encoder, mother, 0, 0)),
                metadata(mother.condition).expand(n, -1),
                condition_kernel(transport, mother.condition).float().expand(n, -1, -1),
                torch.from_numpy(mother.k3_repeat),
            )
            for column, value in zip(columns, values):
                column.append(value)
    return tuple(torch.cat(column) for column in columns), seeds


def fit(model, training, seed):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )
    rng = torch.Generator().manual_seed(seed)
    n = len(training[-1])
    for _ in range(EPOCHS):
        for index in torch.randperm(n, generator=rng).split(BATCH_SIZE):
            z, meta, kernel, y = (column[index] for column in training)
            loss = F.nll_loss(model(z, meta, kernel).clamp_min(1e-12).log(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    assert torch.isfinite(loss)
    model.eval()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scores(p, labels):
    scored = score_vectors(np.asarray(p).reshape(-1, 4), np.asarray(labels).ravel())
    return {k: scored[k] for k in ("acc", "nll", "brier", "ece", "tie_count")}


def orbit_defect(p, shift_rule=lambda r, s: r):
    """Return the maximum probability defect over all codes and all 16 actions."""
    q = np.asarray(p).reshape(-1, 16, 16, 4)
    return float(max(
        np.abs(np.roll(q, (-r * 4, -s * 4), axis=(1, 2))
               - np.roll(q, shift_rule(r, s), axis=-1)).max()
        for r in range(4) for s in range(4)
    ))


def blind_data(cell, encoder, transport, archive, cid, train_seeds):
    # Stage 3D blind cell generation uses cell_idx=0 for each separate namespace.
    omegas = (-cell.abs_omega, cell.abs_omega)
    paired, seeds = generate_paired_cfo_blocks(
        cell.namespace, 0, cell.default_blocks, cell.h, cell.ell, cell.t_p,
        omegas, cell.snr_db, cell.sigma_phi, split_prefix=cid,
    )
    assert set(seeds).isdisjoint(train_seeds)
    raw_codes, labels, independent, metas, kernels = [], [], [], [], []
    for omega in omegas:
        mother = paired[omega]
        z = encode_view(encoder, mother, 0, 0)
        y_rep = np.stack([
            sector_of_complex(matched_sum(mother.y3_repeat * ROOTS[r])) for r, _ in ROTATIONS
        ])
        y_ind = np.stack([
            sector_of_complex(matched_sum(mother.y3_indep * ROOTS[r])) for r, _ in ROTATIONS
        ])
        for r, s in [(r, s) for r in range(4) for s in range(4)]:
            expected = (z + np.array([4 * r, 4 * s])) % 16
            np.testing.assert_array_equal(encode_view(encoder, mother, r, s), expected)
            np.testing.assert_array_equal(
                sector_of_complex(matched_sum(mother.y3_repeat * ROOTS[r])), (y_rep[0] + r) % 4,
            )
        raw_codes.append(np.stack([(z + [4 * r, 4 * s]) % 16 for r, s in ROTATIONS]))
        labels.append(y_rep)
        independent.append(y_ind)
        metas.append(metadata(mother.condition))
        kernels.append(condition_kernel(transport, mother.condition))
    code_pairs = np.stack(raw_codes)
    index = code_pairs @ [16, 1]
    targets = np.stack(labels)
    np.testing.assert_array_equal(index, archive[f"{cid}__codes"])
    np.testing.assert_array_equal(targets, archive[f"{cid}__targets"])
    np.testing.assert_array_equal(seeds, archive[f"{cid}__block_seeds"])
    return index, targets, np.stack(independent), torch.stack(metas), torch.stack(kernels)


def physical_oracle(prior, kernels):
    """Exact source-posterior pushforward, with numerical integration of the kernel."""
    z1, _ = ALL_CODES.T.numpy()
    canonical = np.einsum("zi,sik->szk", prior[z1 % 4], kernels.numpy())
    return np.take_along_axis(canonical, (np.arange(4) - z1[:, None] // 4)[None] % 4, axis=-1)


def lookup(table, indices):
    return table[np.arange(len(table))[:, None, None], indices]


def predict_table(model, meta, kernel):
    return torch.stack([
        model(ALL_CODES, m.expand(256, -1), k.float().expand(256, -1, -1))
        for m, k in zip(meta, kernel)
    ]).numpy()


def predict_orbit_tables(model, meta, kernel, slots=None, gamma=RESIDUAL_LIMIT):
    slots = meta[:, :6] if slots is None else slots
    return torch.stack([model.orbit_table(m, k.float(), s, gamma) for m, k, s in zip(meta, kernel, slots)]).numpy()


def logit_lift_checks():
    # Interface test using nonuniform canonical evidence. The pilot gauge changes
    # both the external Q2 lift and the designated output permutation.
    canonical = torch.arange(64, dtype=torch.float32).reshape(16, 4).softmax(dim=-1)
    canonical = canonical[(ALL_CODES[:, 0] % 4) * 4 + ALL_CODES[:, 1] % 4]
    rules = {"q1": lambda r, s: s - r, "q2": lambda r, s: r,
             "q3_repeat": lambda r, s: r, "q3_indep": lambda r, s: 0,
             "q3_after_pilot": lambda r, s: r}
    defects = {task: orbit_defect(lift_scores(canonical, ALL_CODES, task).numpy(), rules[task])
               for task in TASKS}
    q2 = lift_scores(canonical, ALL_CODES, "q2")
    defects["q2_joint_coordinate"] = max(
        (lift_scores(canonical, (ALL_CODES + torch.tensor([4 * r, 4 * r])) % 16,
                     "q2", pilot_turn=r) - q2).abs().max().item()
        for r in range(4)
    )
    assert max(defects.values()) == 0.0
    return defects


def spectrum(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    e = s**2
    total = float(e.sum())
    fraction = e[e > 0] / total if total else np.empty(0)
    threshold = max(matrix.shape) * np.finfo(matrix.dtype).eps * s.max()
    return {
        "gram_eigenvalues_descending": e.tolist(),
        "singular_values_descending": s.tolist(),
        "rank_svd": int((s > threshold).sum()), "rank_threshold": float(threshold),
        "stable_rank": total / float(e.max()) if total else 0.0,
        "effective_rank_entropy_of_gram_energy": float(np.exp(-np.dot(fraction, np.log(fraction))))
        if total else 0.0,
        "frobenius_norm": math.sqrt(total),
        "best_rank1_relative_frobenius_residual": math.sqrt(max(0.0, 1 - float(e.max()) / total))
        if total else 0.0,
    }


def cal_rank_certificate(cal, overlap):
    """Certify the rank of the archived half-count table before float weighting."""
    # Each archived probability is (2*count+1)/(2*row_count+4), with
    # 65,536 total calibration mothers per CFO sign. Rationals in this bounded
    # denominator range are separated by much more than float64 rounding.
    def exact_row(z):
        return [Fraction(float(p)).limit_denominator(2 * (65_536 + 2)) for p in z]

    for code in range(64):
        orbit = [(code + 64 * r) % 256 for r in range(4)]
        if not np.all(overlap[orbit] > 0):
            continue
        diff = [[q - p for p, q in zip(exact_row(cal[0, z]), exact_row(cal[1, z]))] for z in orbit]
        for r, row in enumerate(diff):
            assert row == [diff[0][(k - r) % 4] for k in range(4)]
            assert sum(row) == 0
        d = diff[0]
        real, imag, nyquist = d[0] - d[2], d[1] - d[3], d[0] - d[1] + d[2] - d[3]
        if (real or imag) and nyquist:
            return {"rank": 3, "code_index": code, "positive_weight_orbit_code_indices": orbit,
                    "frequency_1_real_and_imag": [str(real), str(imag)], "frequency_2": str(nyquist)}
    raise AssertionError("No rank-three cyclic subspace witness in the physical reference")


def stage3d_diagnostics(archive, saved):
    cid = next(iter(BLIND_CELLS))
    read = lambda suffix: archive[f"{cid}__{suffix}"]
    raw, temps, seeds = read("raw_ll"), archive["temps_q3"], archive["seeds"]
    codes, labels = read("codes"), read("targets")
    cal, mass = read("cal_ref"), read("cal_mass")
    work_f = work_f_sign_probe(
        raw_ll_table=raw, temps=temps, codes_rot0=codes[:, 0], targets_rot0=labels[:, 0],
        cal_ref=cal, cal_mass=mass, seeds=seeds,
        contra_text_flipped=read("contra_text_flipped_raw_ll"),
        contra_phase_flipped=read("contra_phase_flipped_raw_ll"),
    )
    work_g = work_g_rotation_and_controls(
        raw, temps, codes, labels, read("rot10_codes"), read("rot10_targets"),
        read("indep_raw_ll"), read("indep_targets"), read("shuffled_perm"), seeds,
    )
    overlap = np.minimum(*mass)
    sqrt_weight = np.sqrt(overlap / overlap.sum())[:, None]
    d_ref = sqrt_weight * (cal[1] - cal[0])
    physical = spectrum(d_ref)
    np.testing.assert_allclose(physical["frobenius_norm"]**2, work_f["cal_ref_odd_norm_sq"], atol=1e-14)
    centered_ref = d_ref - sqrt_weight * (sqrt_weight * d_ref).sum(axis=0)
    mismatch = {"physical": physical, "physical_row_mean_removed": spectrum(centered_ref),
                "physical_calibration_rank_certificate": cal_rank_certificate(cal, overlap), "host": {}}
    for arm, name in enumerate(("phase", "zero")):
        tables = np.stack([softmax_temperature(raw[arm, i], temps[arm, i]) for i in range(len(seeds))])
        by_seed = sqrt_weight[None] * (tables[:, 1] - tables[:, 0])
        average = by_seed.mean(axis=0)
        gain = float(np.sum(average * d_ref) / np.sum(d_ref**2))
        mismatch["host"][name] = {
            "seed_mean_table_contrast": spectrum(average),
            "per_seed": {str(seed): spectrum(d) for seed, d in zip(seeds, by_seed)},
            "orientation_gain": gain,
            "relative_frobenius_error_to_physical": float(np.linalg.norm(average - d_ref) / np.linalg.norm(d_ref)),
            "relative_frobenius_norm": float(np.linalg.norm(average) / np.linalg.norm(d_ref)),
        }
        np.testing.assert_allclose(gain, saved["work_f_primary_040"]["arms"][name]["seed_mean_orientation_gain"])
        for key in ("raw_host", "offline_group_avg_16way", "offline_representative_F0"):
            for metric in ("acc", "nll"):
                np.testing.assert_allclose(work_g["arms"][name][key][metric],
                                           saved["work_g_primary_040"]["arms"][name][key][metric], atol=1e-12)
    return work_f, work_g, mismatch


def pf_counterexample():
    """One complete orbit with three good groups and a confident wrong group."""
    good = np.array([0.35, 0.34, 0.16, 0.15])
    bad = np.array([0.01, 0.97, 0.01, 0.01])
    a = ALL_CODES[:, 0].numpy() // 4
    raw = np.stack([np.roll(bad if r == 3 else good, r) for r in a])
    projected = compute_16way_group_avg(raw)
    idempotence = float(np.max(np.abs(compute_16way_group_avg(projected) - projected)))
    f0_defect = orbit_defect(compute_representative_readout(raw))
    raw_score, pf_score = scores(raw, a), scores(projected, a)
    assert pf_score["nll"] < raw_score["nll"] and pf_score["acc"] < raw_score["acc"]
    assert orbit_defect(projected) < 1e-14 and idempotence < 1e-14 and f0_defect == 0.0
    return {"raw": raw_score, "pf": pf_score, "pf_defect": orbit_defect(projected),
            "pf_idempotence_residual": idempotence, "F0_defect": f0_defect,
            "aligned_projected_probabilities": projected[0].tolist()}


def boundary_scan(prior, g1_indices, transport):
    # Hold the diffusion and W1 code mixture fixed. These are conditional model
    # accuracies, distinct from observed hits on each blind cell.
    weights = np.bincount((g1_indices[:, 0] // 16 % 4).ravel(), minlength=4)
    weights = weights / weights.sum()
    out = {}
    for theta in (0.0, math.pi / 4, 0.80, math.pi / 2, 1.60, math.pi):
        kernel = physical_kernel(transport, theta, 0.04, 2.0).numpy()
        probs = prior[:4] @ kernel
        unrotated = prior[:4].argmax(axis=-1)
        out[f"{theta:.12f}"] = {
            "center_source_kernel": kernel[0].tolist(),
            "expected_oracle_accuracy": float(weights @ probs.max(axis=-1)),
            "expected_unrotated_MAP_accuracy": float(weights @ probs[np.arange(4), unrotated]),
        }
    return out


def lock_temperature(model, dev):
    codes, meta, kernel, targets = dev
    with torch.no_grad():
        probabilities = model(codes, meta, kernel).numpy()
    logits = np.log(np.clip(probabilities.astype(np.float64), 1e-12, 1.0))
    target_logit = logits[np.arange(len(targets)), targets.numpy()]

    def objective(log_t):
        scale = np.exp(-log_t)
        return float(np.mean(logsumexp(logits * scale, axis=1) - target_logit * scale))

    fit_result = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20)), method="bounded")
    assert fit_result.success
    return {"temperature": float(np.exp(fit_result.x)), "unit_T_dev_nll": objective(0.0),
            "calibrated_dev_nll": float(fit_result.fun)}


def apply_temperature(probabilities, temperature):
    return softmax_temperature(np.log(np.clip(probabilities.astype(np.float64), 1e-12, 1.0)), temperature)


def paired_change(before, after, y):
    """After minus before, averaging seeds, rotations, and signs inside block."""
    def item_scores(arms):
        return [score_vectors(p.reshape(-1, 4), y.ravel()) for p in arms]

    before_scores, after_scores = item_scores(before), item_scores(after)
    result = {}
    for key in ("per_item_hit", "per_item_nll"):
        diff = np.stack([b[key] - a[key] for a, b in zip(before_scores, after_scores)])
        blocks = diff.reshape((-1, *y.shape)).mean(axis=(0, 1, 2)).reshape(-1, 64)
        result[{"per_item_hit": "acc", "per_item_nll": "nll"}[key]] = paired_t_ci(blocks.mean(axis=1))
    return result


def mean_scores(probabilities, targets, oracle_nll):
    rows = [scores(p, targets) for p in probabilities]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("acc", "nll", "brier", "ece")}
    mean["reference_normalized_nig"] = (math.log(4) - mean["nll"]) / (math.log(4) - oracle_nll)
    mean["nig_eta_0p10_linear_point"] = 0.90 * math.log(4) - mean["nll"] + 0.10 * oracle_nll
    return mean, rows


def benchmark_cell(cid, data, encoder, archive, saved, models, calibrations):
    indices, labels, independent, meta, kernels = data
    oracle = physical_oracle(encoder.table, kernels)
    expected = archive[f"{cid}__direct_table_probs"]
    oracle_on_mothers = lookup(oracle, indices)
    parity = float(np.max(np.abs(oracle_on_mothers - expected)))
    np.testing.assert_allclose(oracle_on_mothers, expected, atol=1e-9, rtol=0)
    oracle_score = scores(oracle_on_mothers, labels)
    np.testing.assert_allclose(oracle_score["acc"], saved["cells"][cid]["analytic_same_mothers"]["direct_s17_table"]["acc"])
    unrotated_prior = encoder.table[(indices // 16)]
    uniform = lift_scores(torch.zeros_like(torch.from_numpy(oracle[0])), ALL_CODES, "q3_indep")[None]
    uniform = np.repeat(uniform.numpy(), len(meta), axis=0)
    uniform_on_mothers = lookup(uniform, indices)
    result = {
        "mothers_per_sign": labels.shape[-1], "C_direct_s17": oracle_score,
        "C_archive_max_abs_probability_difference": parity, "C_equivariance_defect": orbit_defect(oracle),
        "unrotated_s17_MAP": scores(unrotated_prior, labels), "models": {},
        "q3_indep_zero_tv": {
            **scores(uniform_on_mothers, independent),
            "max_total_variation_from_uniform": float(np.max(0.5 * np.abs(uniform_on_mothers - 0.25).sum(axis=-1))),
        },
    }
    shuffle_swap = np.zeros(labels.shape[-1], dtype=bool)
    shuffler = np.random.default_rng(SHUFFLE_SEED + BLIND_CELLS[cid].namespace)
    shuffle_swap[shuffler.permutation(len(shuffle_swap))[:len(shuffle_swap) // 2]] = True
    fit_scores, raw_scores = {}, {}
    for name, arms in models.items():
        raw_batch_tables = [predict_table(arm, meta, kernels) for arm in arms]
        model_tables = [predict_orbit_tables(arm, meta, kernels) for arm in arms] if arms[0].quotient else raw_batch_tables
        defects = [orbit_defect(tbl) for tbl in model_tables]
        temps = [cal["temperature"] for cal in calibrations[name]]
        raw = [lookup(tbl, indices) for tbl in model_tables]
        fitted = [apply_temperature(prob, t) for prob, t in zip(raw, temps)]
        raw_scores[name], fit_scores[name] = raw, fitted
        raw_mean, raw_metrics = mean_scores(raw, labels, oracle_score["nll"])
        cal_mean, cal_metrics = mean_scores(fitted, labels, oracle_score["nll"])
        result["models"][name] = {
            "unit_T": raw_mean, "dev_calibrated": cal_mean,
            "per_seed": {str(seed): {"unit_T": a, "dev_calibrated": b, "equivariance_defect": defect}
                         for seed, a, b, defect in zip(MODEL_SEEDS, raw_metrics, cal_metrics, defects)},
            "maximum_equivariance_defect": max(defects),
            "dev_calibrated_probability_equivariance_defect": max(
                orbit_defect(apply_temperature(tbl, t)) for tbl, t in zip(model_tables, temps)
            ),
        }
        if arms[0].quotient:
            result["models"][name]["ungrouped_batch_maximum_equivariance_defect"] = max(
                orbit_defect(tbl) for tbl in raw_batch_tables
            )
        if name not in (ARM0, ARM1, ARM2):
            continue
        swapped = [lookup(predict_orbit_tables(arm, meta, kernels, meta[:, :6].flip(0)), indices)
                   for arm in arms]
        shuffled = [np.where(shuffle_swap[None, None, :, None], s, p) for p, s in zip(raw, swapped)]
        result["models"][name]["fixed_text_interventions"] = {}
        for key, intervention in (("sign_swap", swapped), ("balanced_shuffle", shuffled)):
            cal_intervention = [apply_temperature(p, t) for p, t in zip(intervention, temps)]
            result["models"][name]["fixed_text_interventions"][key] = {
                "dev_calibrated": mean_scores(cal_intervention, labels, oracle_score["nll"])[0],
                "unit_T_intervention_minus_matched": paired_change(raw, intervention, labels),
                "dev_cal_intervention_minus_matched": paired_change(fitted, cal_intervention, labels),
            }
        if name == ARM2:
            zero = [lookup(predict_orbit_tables(arm, meta, kernels, gamma=0.0), indices) for arm in arms]
            zero_swapped = [lookup(predict_orbit_tables(arm, meta, kernels, meta[:, :6].flip(0), gamma=0.0), indices)
                            for arm in arms]
            np.testing.assert_array_equal(zero, zero_swapped)
            np.testing.assert_allclose(zero, [oracle_on_mothers] * len(arms), atol=2e-7, rtol=0)
            cal_zero = [apply_temperature(p, t) for p, t in zip(zero, temps)]
            result["models"][name]["bypass_diagnostics"] = {
                "gamma_zero_max_abs_diff_from_arm3": float(np.max(np.abs(np.asarray(zero) - oracle_on_mothers))),
                "gamma_zero_slot_swap_defect": float(np.max(np.abs(np.asarray(zero) - np.asarray(zero_swapped)))),
                "unit_T_host_increment_arm2_minus_gamma_zero": paired_change(zero, raw, labels),
                "common_dev_T_host_increment_arm2_minus_gamma_zero": paired_change(cal_zero, fitted, labels),
            }
    references = [oracle_on_mothers] * len(MODEL_SEEDS)
    contrasts = {
        "arm1_minus_arm0": (ARM0, ARM1),
        "quotient_effect_with_separate": (ARM0, CONTROL_Q),
        "quotient_effect_with_coupling": (CONTROL_C, ARM1),
        "coupling_effect_without_quotient": (ARM0, CONTROL_C),
        "coupling_effect_with_quotient": (CONTROL_Q, ARM1),
        "arm1_minus_zero_slots_refit": (CONTROL_ZERO, ARM1),
        "arm2_minus_arm1": (ARM1, ARM2),
        "arm1_minus_arm3": (ARM3, ARM1),
        "arm2_minus_arm3": (ARM3, ARM2),
    }
    result["paired_contrasts"] = {
        key: {
            "unit_T": paired_change(raw_scores.get(a, references), raw_scores[b], labels),
            "dev_calibrated": paired_change(fit_scores.get(a, references), fit_scores[b], labels),
        } for key, (a, b) in contrasts.items()
    }
    result["slot_shuffle"] = {
        "seed": SHUFFLE_SEED + BLIND_CELLS[cid].namespace, "mothers_switched": int(shuffle_swap.sum()),
        "mothers_total": len(shuffle_swap), "policy": "Swap both signs for the same half of paired mother indices",
    }
    return result


def centered(values):
    return values - values.mean(axis=-1, keepdims=True)


def jacobian_readout(model, meta, target, *, stack=False):
    """Native centered candidate scores, before any physical output path."""
    codes = INNER_CODES.repeat(len(meta), 1)
    meta_rows = meta.repeat_interleave(len(INNER_CODES), dim=0)
    public = meta_rows[:, 6:]
    with torch.no_grad():
        tokens, wave_hidden, phase_hidden = model.prefix(decode_windows(codes, 4), meta_rows, meta_rows[:, :6])
        values = centered(model.host(tokens, public).numpy().astype(np.float64))
    target = centered(np.log(target.reshape(-1, 4)))

    def read(candidate_tokens, text):
        logits = model.host(candidate_tokens[None], text[None])[0]
        return logits - logits.mean()

    j = torch.func.vmap(torch.func.jacrev(read))(tokens, public).reshape(-1, 4, 3 * WIDTH).detach().numpy()
    j = j.astype(np.float64)
    sv = np.linalg.svd(j, compute_uv=False)
    numerical_rank = (sv > sv[:, :1] * 1e-6).sum(axis=-1)
    top, runner = np.argsort(target, axis=-1)[:, -2:][:, ::-1].T
    row = np.arange(len(target))
    oracle_margin = target[row, top] - target[row, runner]
    chosen_margin = values[row, top] - values[row, runner]
    v = j[row, top] - j[row, runner]
    reach = np.linalg.norm(v, axis=-1) * MARGIN_RADIUS
    target_delta = target - values
    delta_min = np.einsum("nki,ni->nk", np.linalg.pinv(j, rcond=1e-6), target_delta)
    remaining = target_delta - np.einsum("nik,nk->ni", j, delta_min)
    result = {
        "rows": len(j), "rank_min": int(numerical_rank.min()), "rank_max": int(numerical_rank.max()),
        "median_singular_values_descending": np.median(sv, axis=0).tolist(),
        "minimum_third_singular_value": float(sv[:, 2].min()),
        "native_margin_median_against_oracle_runner_up": float(np.median(chosen_margin)),
        "oracle_margin_median": float(np.median(oracle_margin)),
        "native_top1_agreement_with_oracle": float(np.mean(values.argmax(axis=-1) == top)),
        "linearized_radius": MARGIN_RADIUS,
        "fraction_oracle_pairwise_margin_positive_throughout_linearized_ball": float(np.mean(chosen_margin > reach)),
        "fraction_oracle_pairwise_margin_unreachable_in_linearized_ball": float(np.mean(chosen_margin + reach <= 0)),
        "median_min_token_norm_to_fit_centered_target_linearly": float(np.median(np.linalg.norm(delta_min, axis=-1))),
        "local_unconstrained_relative_projection_residual": float(np.linalg.norm(remaining) / np.linalg.norm(target_delta)),
    }
    if stack:
        # Compose the token Jacobian with the two *shared* final projection
        # Jacobians. Use a uniformly weighted representative/context design.
        wave_h = np.pad(wave_hidden.numpy().astype(np.float64), ((0, 0), (0, 1)), constant_values=1)
        phase_h = np.pad(phase_hidden.numpy().astype(np.float64), ((0, 0), (0, 1)), constant_values=1)
        n = len(j)
        wave = (j[..., :2 * WIDTH, None] * wave_h[:, None, None, :]).reshape(n * 4, -1)
        phase = (j[..., 2 * WIDTH:, None] * phase_h[:, None, None, :]).reshape(n * 4, -1)
        a = np.concatenate((wave, phase), axis=1) / math.sqrt(n)
        d = target_delta.ravel() / math.sqrt(n)
        lam, u = np.linalg.eigh(a @ a.T)
        keep = lam > lam[-1] * 1e-10
        coeff = u[:, keep].T @ d
        result["shared_last_projection_tangent"] = {
            "rows": a.shape[0], "columns": a.shape[1], "rank": int(keep.sum()),
            "retained_singular_values_descending": np.sqrt(lam[keep][::-1]).tolist(),
            "relative_projection_residual": float(np.linalg.norm(d - u[:, keep] @ coeff) / np.linalg.norm(d)),
            "min_weight_norm_for_projected_target": float(np.linalg.norm(coeff / np.sqrt(lam[keep]))),
        }
    return result


def candidate_diagnostics(models, contexts, prior, host):
    w_out = host.w_out.numpy().astype(np.float64)
    gram = w_out @ w_out.T
    projector = np.eye(4) - np.ones((4, 4)) / 4
    np.testing.assert_allclose(gram, projector, atol=1e-7, rtol=0)
    out = {"host_out_projection_gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
           "host_out_projection_gram_max_abs_defect": float(np.max(np.abs(gram - projector))), "contexts": {}}
    representative_indices = (INNER_CODES @ torch.tensor([16, 1])).numpy()
    for name, (meta, kernels) in contexts.items():
        target = physical_oracle(prior, kernels)[:, representative_indices]
        out["contexts"][name] = {
            arm: {str(seed): jacobian_readout(m, meta, target, stack=(name == "dev"))
                  for seed, m in zip(MODEL_SEEDS, models[arm])} for arm in (ARM0, ARM1, ARM2)
        }
    return out


ALL_CODES = torch.cartesian_prod(torch.arange(16), torch.arange(16))
INNER_CODES = torch.cartesian_prod(torch.arange(4), torch.arange(4))


def joint_phase_moment(edges, harmonics, rate_mean, phase_var, rate_var):
    """Joint moment of signed Brownian edge increments with one latent CFO."""
    start, end = np.asarray(edges, dtype=float).T
    durations = end - start
    lo, hi = np.minimum(start, end), np.maximum(start, end)
    overlap = np.maximum(0, np.minimum.outer(hi, hi) - np.maximum.outer(lo, lo))
    brownian = overlap * np.sign(durations[:, None] * durations[None, :])
    cov = phase_var * brownian + rate_var * np.outer(durations, durations)
    harmonics = np.asarray(harmonics)
    return np.exp(1j * rate_mean * (harmonics @ durations) - 0.5 * (harmonics @ cov @ harmonics)), cov


def pilot_joint_marginals(joint, valid, no_pilot_q3):
    """Read the TX-label and canonical RX-sector marginals of normalized M3 J."""
    q2 = torch.where(valid[:, None], joint.sum(dim=-1), 0.25)
    q3 = torch.where(valid[:, None], joint.sum(dim=-2), no_pilot_q3)
    return q2, q3


def interface_checks(encoder, model, residual_model, transport):
    with torch.no_grad():
        words = torch.arange(65536).unsqueeze(-1).expand(-1, 2)
        packet16 = decode_windows(words, 16)
        assert torch.equal(packet16.codes, words >> 12)
        assert bool(torch.isfinite(packet16.tail).all())
        for r in range(4):
            rotated = decode_windows((words + (r << 14)) % 65536, 16)
            assert torch.equal(rotated.tail, packet16.tail)
            assert torch.equal(rotated.codes, (packet16.codes + 4 * r) % 16)
        missing = decode_windows(words[:2], 16, torch.zeros((2, 2), dtype=torch.bool))
        assert torch.count_nonzero(missing.codes) == 0 and torch.count_nonzero(missing.tail) == 0

        h, ell, tp = TRAIN_TIMINGS[0]
        mothers, _ = generate_paired_cfo_blocks(NS_CFO_DEV, 0, 1, h, ell, tp, DEV_OMEGAS)
        mother = mothers[DEV_OMEGAS[0]]
        t = np.stack((matched_sum(mother.y1), matched_sum(mother.y2)), axis=-1)
        rng = np.random.default_rng(1907)
        variance = 10**(-mother.condition.snr_db / 10)
        noise = math.sqrt(variance / 2) * (rng.normal(size=(*t.shape, 2))
                                            + 1j * rng.normal(size=(*t.shape, 2)))
        basis = np.stack(([1, -1, 0] / np.sqrt(2), [1, 1, -2] / np.sqrt(6)), axis=-1)
        ancillary = noise @ basis.T
        lifted = t[..., None] / math.sqrt(3) + ancillary
        energy = np.sum(np.abs(ancillary)**2, axis=-1)
        err = float(np.max(np.abs(matched_sum(lifted) - t)))
        assert err < 2e-14
        m = metadata(mother.condition, rate_var=0.0025).expand(len(t), -1)
        k = condition_kernel(transport, mother.condition, rate_var=0.0025).float().expand(len(t), -1, -1)
        packed_base, _ = m5_encode(lifted, t, energy, variance, encoder)

        def formats(packed):
            return {bits: decode_windows(torch.from_numpy(packed[key]), bits)
                    for bits, key in ((4, "4_s17"), (16, "16"))}

        original = formats(packed_base)
        torch.testing.assert_close(original[4].codes, original[16].codes, atol=0, rtol=0)
        assert torch.equal(model.predict(original[4], m, k), model.predict(original[4], m, torch.full_like(k, torch.nan)))
        checks = {}
        for name, arm in ((ARM1, model), (ARM2, residual_model)):
            by_format = {}
            for bits, original_packet in original.items():
                pred = arm.predict(original_packet, m, k, receive_gauge=True)
                defects = []
                for r in range(4):
                    packed, _ = m5_encode(lifted * ROOTS[r], t * ROOTS[r], energy, variance, encoder)
                    packet = formats(packed)[bits]
                    assert torch.equal(packet.tail, original_packet.tail)
                    assert torch.equal(packet.codes, (original_packet.codes + 4 * r) % 16)
                    rotated = arm.predict(packet, m, k, receive_gauge=True)
                    defects.append(float((rotated - pred.roll(r, dims=-1)).abs().max()))
                by_format[str(bits)] = max(defects)
                assert by_format[str(bits)] <= 1e-6
            checks[name] = by_format

        fixture = torch.arange(1, 33, dtype=torch.float64).reshape(2, 4, 4)
        fixture /= fixture.sum(dim=(-1, -2), keepdim=True)
        q2, q3 = pilot_joint_marginals(fixture, torch.tensor([True, False]), k[:2, 0].double())
        assert torch.equal(q2[0], fixture[0].sum(dim=-1)) and torch.equal(q3[0], fixture[0].sum(dim=-2))
        assert torch.equal(q2[1], torch.full_like(q2[1], 0.25)) and torch.equal(q3[1], k[1, 0].double())
        code = original[4].codes[:2]
        # A nonuniform joint factor with TX rows and RX-canonical columns.
        q1 = torch.softmax(torch.tensor([[0.0, 1.1, -1.7, 0.1], [1.6, -0.8, 0.4, 0.0]]), dim=-1)
        canonical = dict(q1=q1, q2=q2, q3_repeat=q3, q3_indep=q3, q3_after_pilot=q3)
        gauge_defects = {}
        for task, score in canonical.items():
            base = lift_scores(score, code, task, receive_gauge=True)
            error = 0.0
            for tx in range(4):
                for r in range(4):
                    reply = lift_scores(score, (code + 4 * r) % 16, task, receive_gauge=True, tx_chart_turn=tx)
                    shift = tx if task == "q2" else r if task in ("q3_repeat", "q3_after_pilot") else 0
                    error = max(error, float((reply - base.roll(shift, dims=-1)).abs().max()))
            gauge_defects[task] = error
        assert max(gauge_defects.values()) == 0.0

    edges = [(0, 2), (2, 4), (0, 4)]
    n, omega, v_phi, v_omega = np.array([1, -2, 1]), 0.23, 0.01, 0.04
    expected, covariance = joint_phase_moment(edges, n, omega, v_phi, v_omega)
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    weight = weights / math.sqrt(math.pi)
    rates = omega + np.sqrt(2 * v_omega) * nodes
    brownian = np.sqrt(2 * v_phi * 2) * nodes
    ep1 = 2 * rates[:, None, None] + brownian[None, :, None]
    ep2 = 2 * rates[:, None, None] + brownian[None, None, :]
    gh = np.einsum("i,j,k,ijk->", weight, weight, weight,
                   np.exp(1j * ((n[0] + n[2]) * ep1 + (n[1] + n[2]) * ep2)))
    independent_rate_cov = covariance - v_omega * (np.outer([2, 2, 4], [2, 2, 4]) - np.diag([4, 4, 16]))
    wrong = np.exp(1j * omega * (n @ [2, 2, 4]) - 0.5 * (n @ independent_rate_cov @ n))
    assert abs(gh - expected) < 1e-12 and abs(wrong - expected) > 0.05
    return {
        "all_65536_prefixes_and_four_turns_checked": True,
        "host_pure_probabilities_independent_of_kernel_even_if_nan": True,
        "m3_marginals_and_missing_fallback_checked": True,
        "m3_rx_tx_action_max_probability_defect": gauge_defects,
        "synthetic_r3_matched_filter_max_abs_difference": err,
        "synthetic_r3_packet_model_rx_gauge_defects": checks,
        "hardware_capture_scored": False, "extended_packet_weights_trained": False,
        "shared_cfo_joint_moment": {"edges": edges, "harmonics": n.tolist(), "covariance": covariance.tolist(),
                                    "closed_form_re_im": [expected.real, expected.imag],
                                    "gauss_hermite_abs_error": float(abs(gh - expected)),
                                    "spuriously_independent_rate_abs_error": float(abs(wrong - expected))},
    }


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, digest = load_s17_encoder(ROOT)
    transport = SectorTransport()
    archive_file = STAGE3D_RESULTS / "blind_q3_scores.npz"
    saved = json.loads((STAGE3D_RESULTS / "summary_master.json").read_text(encoding="utf-8"))
    training, train_seeds = draw_split(encoder, transport, NS_CFO_TRAIN, TRAIN_BLOCKS, TRAIN_OMEGAS, "m1_train")
    dev, dev_seeds = draw_split(encoder, transport, NS_CFO_DEV, DEV_BLOCKS, DEV_OMEGAS, "m1_dev")
    assert set(train_seeds).isdisjoint(dev_seeds)
    configurations = {
        ARM0: dict(quotient=False, coupled=False), ARM1: dict(quotient=True, coupled=True),
        ARM2: dict(quotient=True, coupled=True, residual=True),
        CONTROL_Q: dict(quotient=True, coupled=False), CONTROL_C: dict(quotient=False, coupled=True),
        CONTROL_ZERO: dict(quotient=True, coupled=True, zero_slots=True),
    }
    models = {key: [] for key in configurations}
    calibrations = {key: [] for key in models}
    host = FrozenHost(HOST_SEED)
    original_host = {key: value.clone() for key, value in host.state_dict().items()}
    for seed in MODEL_SEEDS:
        for key, config in configurations.items():
            torch.manual_seed(seed)
            arm = EquivariantEvidenceConnector(host, encoder.table, **config)
            fit(arm, training, seed)
            models[key].append(arm)
            calibrations[key].append(lock_temperature(arm, dev))
        assert all(torch.equal(value, original_host[key]) for key, value in host.state_dict().items())
        print(f"Trained the six matched CPU connectors for seed {seed}", flush=True)

    dev_meta = dev[1][::DEV_BLOCKS * 64]
    dev_kernels = dev[2][::DEV_BLOCKS * 64].double()
    contexts = {"dev": (dev_meta, dev_kernels)}
    with np.load(archive_file, allow_pickle=False) as archive, torch.no_grad():
        work_f, work_g, mismatch = stage3d_diagnostics(archive, saved)
        cells = {}
        for cid, cell in BLIND_CELLS.items():
            data = blind_data(cell, encoder, transport, archive, cid, [*train_seeds, *dev_seeds])
            contexts[cid] = (data[3], data[4])
            cells[cid] = benchmark_cell(cid, data, encoder, archive, saved, models, calibrations)
            if cell.namespace == BLIND_CELLS["G1_main_extrap_both"].namespace:
                scan = boundary_scan(encoder.table, data[0], transport)
            print(cid, ", ".join(f"Arm {i}={100 * cells[cid]['models'][k]['dev_calibrated']['acc']:.2f}%"
                                 for i, k in enumerate((ARM0, ARM1, ARM2))),
                  f"Arm 3={100 * cells[cid]['C_direct_s17']['acc']:.2f}%", flush=True)

    geometry = candidate_diagnostics(models, contexts, encoder.table, host)
    interfaces = interface_checks(encoder, models[ARM1][0], models[ARM2][0], transport)
    extrapolation = [cells[cid] for cid in ("G1_main_extrap_both", "G3_extrap_angle")]
    structural = {
        "arm1_arm2_grouped_defect_zero": all(
            row["models"][k]["maximum_equivariance_defect"] == 0.0
            for row in cells.values() for k in (ARM1, ARM2)
        ),
        "arm1_arm2_ungrouped_defect_at_most_1e_6": all(
            row["models"][k]["ungrouped_batch_maximum_equivariance_defect"] <= 1e-6
            for row in cells.values() for k in (ARM1, ARM2)
        ),
        "host_frozen_and_same_trainable_parameter_count": (
            all(not p.requires_grad for p in host.parameters())
            and len({count_trainable(v[0]) for v in models.values()}) == 1
        ),
    }
    observed = {
        k: {
            "G1_G3_seedmean_above_55_pct_in_both": all(row["models"][k]["dev_calibrated"]["acc"] > 0.55
                                                     for row in extrapolation),
            "G1_G3_each_seed_above_55_pct_in_both": all(
                s["dev_calibrated"]["acc"] > 0.55 for row in extrapolation for s in row["models"][k]["per_seed"].values()
            ),
            "all_cell_mean_ece_point_below_0p05": all(row["models"][k]["dev_calibrated"]["ece"] <= 0.05
                                                    for row in cells.values()),
        } for k in (ARM0, ARM1, ARM2)
    }
    out = {
        "round": 2, "physical_control": {"name": ARM3, "cell_result_key": "C_direct_s17", "temperature": 1.0},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__, "device": "cpu"},
        "train": {"s17_sha256": digest, "namespace": NS_CFO_TRAIN, "blocks_per_timing": TRAIN_BLOCKS,
                  "rows": len(training[-1]), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                  "learning_rate": LEARNING_RATE, "timings": TRAIN_TIMINGS, "omegas": TRAIN_OMEGAS,
                  "model_seeds": MODEL_SEEDS, "host_seed": HOST_SEED, "configurations": configurations,
                  "trainable_parameters": {k: count_trainable(v[0]) for k, v in models.items()},
                  "log_prior_residual_limit": RESIDUAL_LIMIT},
        "dev": {"namespace": NS_CFO_DEV, "blocks_per_timing": DEV_BLOCKS, "omegas": DEV_OMEGAS,
                "rows": len(dev[-1]), "temperature_by_arm": {
                    k: dict(zip(map(str, MODEL_SEEDS), v)) for k, v in calibrations.items()
                }},
        "verified_structural_checks": structural, "observed_performance_thresholds": observed,
        "all_task_quotient_lift_defects": logit_lift_checks(), "interface_checks": interfaces,
        "candidate_jacobian": geometry,
        "cells": cells, "boundary_scan_fixed_variance_0p04": scan,
        "stage3d_saved_cells": {cid: {"arms": value["arms"], "analytic_same_mothers": value["analytic_same_mothers"]}
                                for cid, value in saved["cells"].items()},
        "work_f_recomputed": work_f, "work_g_recomputed": work_g,
        "work_f_overlap_weighted_signed_contrast_spectra": mismatch,
        "pf_complete_orbit_counterexample": pf_counterexample(),
    }
    output = Path(__file__).with_name("results_m1.json")
    output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output, flush=True)
    print(json.dumps(structural), flush=True)
    print(json.dumps(observed), flush=True)
    assert all(structural.values())


if __name__ == "__main__":
    main()
