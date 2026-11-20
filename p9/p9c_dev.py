"""P9-C Self-Contained 3-Seed Dev Training, Evaluation & Cost-Profiling Script.

Completely self-contained using p9/contract.py + p9/neural_gate.py + p8/p8c + PyTorch
(zero dependency on p9/channel.py or p9/gating.py).

Trains all 4 parameter-matched TemporalTrustGate variants ('transport_attn',
'abs_no_transport', 'local', 'no_history') across 3 seeds (901, 902, 903) on
strictly disjoint training trajectories (SEED_P9_TRAIN) using identical data,
optimizer, steps, and joint loss (payload_bce + 0.05 * probe_nll), then evaluates
on disjoint dev trajectories (SEED_P9_DEV + 300000: 4 clusters x 12 links = 48
links/regime) across all 3 ID regimes + 2 OOD regimes against classical baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import tracemalloc
from typing import Dict, List, Sequence, Tuple

# Enforce single-threaded deterministic execution before heavy math work.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

from p8 import link as ofdm_link
from p8.coding import decode as ldpc_decode
from p8.coding import qam_llrs
from p8.link import EVEN, clock, complex_normal, draw_channel, stream, traffic
from p8.tracker import State, advance, evolution, observed_taps, prior, update
from p8c import link as sc_link
from p8c.link import PILOTS as SC_PILOTS
from p8c.link import extra_rows
from p8c.receiver import DELAYED_INDEX, SAMPLES, SWEEPS, _training_dictionary, sample_moments
from p9.contract import (
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    N,
    OFDM_PILOT_ROWS,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    SC_LAYOUTS,
    SEED_P9_DEV,
    SEED_P9_TRAIN,
    T,
    CandidateSummary,
    GateOutput,
    HiddenTruth,
    PublicObservation,
    n0_from_snr,
)
from p9.neural_gate import (
    COMPLEX,
    REAL,
    VARIANTS,
    GateFeatures,
    TemporalTrustGate,
    classical_pre_update_gate,
    compose_gate_output,
    extract_gate_features,
    mix_bit_llrs_torch_from_logit,
)


DEV_SEEDS = (901, 902, 903)
DEV_SEED_BASE = SEED_P9_DEV + 300000
TRAIN_LINKS_PER_REGIME = 280          # 280 x 3 ID regimes = 840 training links (disjoint SEED_P9_TRAIN)
DEV_CLUSTERS = 4
DEV_LINKS_PER_CLUSTER = 12            # 4 x 12 = 48 links per regime (240 total across 5 regimes)

TRAIN_STEPS = 420
BATCH_PER_REGIME = 12                 # batch size 36 = 12 per ID regime
LR = 3.0e-3
MIN_LR = 1.5e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 2.0
PROBE_LOSS_WEIGHT = 0.05

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = RESULTS_DIR / "p9c_dev_ckpts"
OUTPUT_JSON = RESULTS_DIR / "p9c_dev_gate.json"


def configure_determinism() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Self-Contained Multi-Regime Physical Simulator (Explicit n0, ZERO hardcoded N0)
# ---------------------------------------------------------------------------


def _eval_frame_b_ray_gains(
    channel_a: ofdm_link.Channel,
    regime: str,
    seed: int,
) -> Tuple[np.ndarray, object, int | None, float, int | None]:
    """Compute exact time-domain ray gains h_b_true [3, T, N] on Frame B for each P9 regime."""
    time_b = clock(T, start_row=T)  # [T, N] in useful-symbol units starting at Frame B
    rng_reg = stream(seed, 8)

    if regime == "continuous":
        phase = 2.0 * np.pi * (
            channel_a.frequency[:, None, None] * time_b[None]
            + 0.5 * channel_a.chirp[:, None, None] * (time_b[None] ** 2)
        )
        gains = channel_a.gain[:, None, None] * np.exp(1j * phase)
        return gains, channel_a, None, 0.0, None

    if regime == "full_reset":
        channel_b = draw_channel(stream(seed, 5))
        phase = 2.0 * np.pi * (
            channel_b.frequency[:, None, None] * time_b[None]
            + 0.5 * channel_b.chirp[:, None, None] * (time_b[None] ** 2)
        )
        gains = channel_b.gain[:, None, None] * np.exp(1j * phase)
        return gains, channel_b, None, 0.0, 0

    if regime in ("partial_shift", "ood_strong_shift", "ood_midframe_shift"):
        changed_ray = int(rng_reg.integers(0, len(DELAYS)))
        if regime == "partial_shift":
            shift_row = 0
        elif regime == "ood_strong_shift":
            shift_row = 0
        else:  # ood_midframe_shift
            shift_row = 6

        # Exact phase-continuous Doppler shift starting at t_shift = start of shift_row's CP.
        t_shift = (T + shift_row) * (N + CP) / N
        phase = 2.0 * np.pi * (
            channel_a.frequency[:, None, None] * time_b[None]
            + 0.5 * channel_a.chirp[:, None, None] * (time_b[None] ** 2)
        )
        phi_at_shift = 2.0 * np.pi * (
            channel_a.frequency[changed_ray] * t_shift
            + 0.5 * channel_a.chirp[changed_ray] * (t_shift ** 2)
        )
        f_at_shift = float(channel_a.frequency[changed_ray] + channel_a.chirp[changed_ray] * t_shift)
        old_sign = 1.0 if f_at_shift >= 0.0 else -1.0

        if regime in ("partial_shift", "ood_midframe_shift"):
            # Jump changed_ray to a fresh, physically IN-DISTRIBUTION Doppler trajectory
            # inside ±[0.045, 0.080] (opposite sign so delta_freq is decisive and clear stays in-band).
            new_sign = -old_sign
            f_new_start = float(rng_reg.uniform(0.045, 0.080) * new_sign)
            f_new_end = float(rng_reg.uniform(0.045, 0.080) * new_sign)
            rem_dur = max((2 * T - (T + shift_row)) * (N + CP) / N, 1.0)
            c_new = (f_new_end - f_new_start) / rem_dur
            delta_freq = f_new_start - f_at_shift
        else:  # ood_strong_shift: push changed_ray strictly OUTSIDE the ID ±[0.045, 0.080] bank
            df_mag = float(rng_reg.uniform(0.060, 0.110))
            delta_freq = old_sign * df_mag
            f_new_start = f_at_shift + delta_freq
            c_new = float(channel_a.chirp[changed_ray])

        dt_after = time_b - t_shift
        mask_after = dt_after > 0.0
        phase_after = phi_at_shift + 2.0 * np.pi * (f_new_start * dt_after + 0.5 * c_new * (dt_after ** 2))
        phase[changed_ray] = np.where(mask_after, phase_after, phase[changed_ray])
        gains = channel_a.gain[:, None, None] * np.exp(1j * phase)

        shifted_freq = channel_a.frequency.copy()
        shifted_chirp = channel_a.chirp.copy()
        shifted_freq[changed_ray] = f_new_start
        shifted_chirp[changed_ray] = c_new
        ch_b = ofdm_link.Channel(channel_a.gain.copy(), shifted_freq, shifted_chirp)
        return gains, ch_b, changed_ray, float(delta_freq), shift_row

    raise ValueError(f"Unsupported P9-C regime: {regime}")


def simulate_p9_link(
    seed: int,
    regime: str,
    snr_db: float = PRIMARY_SNR_DB,
    budget: str = "sparse",
) -> Tuple[PublicObservation, HiddenTruth]:
    """Generate one strictly separated (PublicObservation, HiddenTruth) OFDM(A) -> CP-SC(B) link."""
    n0 = n0_from_snr(snr_db)
    training_rows = SC_LAYOUTS[budget]

    channel_a = draw_channel(stream(seed, 0))
    a_words = traffic(stream(seed, 1))
    a_clean = ofdm_link.propagate(ofdm_link.tx_grid(a_words, OFDM_PILOT_ROWS), channel_a, start_row=0)
    a_noise = np.fft.fft(np.sqrt(n0) * complex_normal(stream(seed, 2), (T, N)), norm="ortho")
    y_a = a_clean + a_noise

    words_full = traffic(stream(seed, 3))
    b_words = ofdm_link.Traffic(words_full.info[:18], words_full.code[:18], words_full.symbols[:18])
    tx_b = sc_link.tx_grid(b_words, training_rows)

    h_b_true, channel_b, changed_ray, delta_freq, shift_row = _eval_frame_b_ray_gains(channel_a, regime, seed)
    # Time-domain CP-SC convolution with exact per-sample time-varying ray gains.
    tx_delayed = np.transpose(tx_b[:, DELAYED_INDEX], (1, 0, 2))  # [3, T, N]
    y_b_clean = (h_b_true * tx_delayed).sum(axis=0)
    b_noise = np.sqrt(n0) * complex_normal(stream(seed, 4), (T, N))
    y_b_noisy = y_b_clean + b_noise

    # Mask withheld probe row to NaN in PublicObservation.
    y_b_pub = y_b_noisy.copy()
    y_b_pub[PROBE_ROW] = np.nan + 1j * np.nan

    obs = PublicObservation(
        y_a=y_a,
        y_b=y_b_pub,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        budget=budget,
    )
    truth = HiddenTruth(
        regime=regime,
        channel_a=channel_a,
        channel_b=channel_b,
        changed_ray=changed_ray,
        delta_freq=delta_freq,
        shift_row=shift_row,
        b_info=b_words.info,
        b_code=b_words.code,
        b_symbols=b_words.symbols,
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_clean[PROBE_ROW].copy(),
        y_b_full_clean=y_b_clean,
        h_b_true=h_b_true,
    )
    return obs, truth


# ---------------------------------------------------------------------------
# Explicit-n0 Classical Physical Branch Summaries (Strict Pre-Update Likelihood!)
# ---------------------------------------------------------------------------


def _pre_update_training_stats(
    prior_state: State,
    y_b_safe: np.ndarray,
    training_rows: tuple,
    n0: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Exact strict pre-update joint complex Gaussian predictive loglik, mean, and variance."""
    rows = tuple(int(r) for r in training_rows)
    y_vec = y_b_safe[list(rows)].reshape(-1)
    m = len(y_vec)
    design, _, _ = _training_dictionary(prior_state.epoch, rows)  # [3, H, M]
    weights = np.exp(prior_state.log_weight)                      # [3, H]
    mu_ray_vec = np.einsum("lh,lhm->lm", weights * prior_state.mean, design)  # [3, M]
    mu_y = mu_ray_vec.sum(axis=0)                                             # [M]

    second = np.maximum(weights * (np.abs(prior_state.mean) ** 2 + prior_state.variance), 0.0)
    weighted_d = (np.sqrt(second)[:, :, None] * design).reshape(-1, m)
    # Exact Hermitian covariance Sigma[i, j] = sum_k d[k, i] * conj(d[k, j]) - sum_l mu[l, i] * conj(mu[l, j]).
    cov = weighted_d.T @ weighted_d.conj() - mu_ray_vec.T @ mu_ray_vec.conj()
    cov = 0.5 * (cov + cov.conj().T)
    cov[np.diag_indices_from(cov)] += max(n0, 1e-300)

    chol = np.linalg.cholesky(cov)
    diff = y_vec - mu_y
    z = np.linalg.solve(chol, diff)
    log_det = 2.0 * float(np.sum(np.log(np.real(np.diag(chol)))))
    loglik = float(-m * math.log(math.pi) - log_det - float(np.real(np.vdot(z, z))))
    pred = mu_y.reshape(len(rows), N)
    var = np.maximum(np.real(np.diag(cov)), 1e-300).reshape(len(rows), N)
    return loglik, pred, var


def _condition_n0(prior_state: State, y_b_safe: np.ndarray, rows: tuple, n0: float) -> State:
    """Explicit-n0 coordinate ascent conditioning (generalizes p8c/receiver.py without hardcoded N0)."""
    rows = tuple(int(r) for r in rows)
    if not rows:
        return prior_state
    y = y_b_safe[list(rows)].reshape(-1)
    design, conjugate, energy = _training_dictionary(prior_state.epoch, rows)
    n0_safe = max(float(n0), 1e-300)
    post_var = prior_state.variance / (1.0 + prior_state.variance * energy / n0_safe)
    log_weight = prior_state.log_weight.copy()
    mean = prior_state.mean.copy()
    per_ray = np.einsum("lh,lhq->lq", np.exp(log_weight) * mean, design)
    initial_precision_mean = prior_state.mean / prior_state.variance
    evidence_base = (
        prior_state.log_weight
        + np.log(post_var / prior_state.variance)
        - np.abs(prior_state.mean) ** 2 / prior_state.variance
    )
    for _ in range(SWEEPS):
        for ray in range(len(DELAYS)):
            residual = y - per_ray.sum(axis=0) + per_ray[ray]
            matched = conjugate[ray] @ residual
            natural_mean = initial_precision_mean[ray] + matched / n0_safe
            mean[ray] = post_var[ray] * natural_mean
            evidence = evidence_base[ray] + post_var[ray] * (np.abs(natural_mean) ** 2)
            log_weight[ray] = evidence - logsumexp(evidence)
            per_ray[ray] = (np.exp(log_weight[ray]) * mean[ray]) @ design[ray]
    return State(prior_state.epoch, log_weight, mean, post_var)


def _readout_n0(
    mean_ray: np.ndarray,
    var_ray: np.ndarray,
    y_b_safe: np.ndarray,
    rows: tuple,
    n0: float,
    probe_row: int = PROBE_ROW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit-n0 CP-SC probe prediction & data-block LMMSE soft bit LLRs."""
    rows = tuple(int(r) for r in rows)
    n0_safe = max(float(n0), 1e-300)
    known = SC_PILOTS[probe_row, DELAYED_INDEX]
    prediction = (mean_ray[:, probe_row] * known).sum(axis=0)
    pred_var = n0_safe + (var_ray[:, probe_row] * (np.abs(known) ** 2)).sum(axis=0)

    data = np.array([r for r in range(T) if r not in (*rows, probe_row)], dtype=int)
    llrs = np.zeros((T, N, 4), dtype=np.float64)
    if not len(data):
        return llrs, prediction, pred_var

    noise = n0_safe + var_ray[:, data].sum(axis=0)
    design = np.zeros((len(data), N, N), dtype=np.complex128)
    for ray, delayed in enumerate(DELAYED_INDEX):
        design[:, SAMPLES, delayed] = mean_ray[ray, data]
    adjoint = design.conj().swapaxes(-1, -2) / noise[:, None, :]
    cov = np.linalg.inv(np.eye(N, dtype=np.complex128) + adjoint @ design)
    estimate = (cov @ (adjoint @ y_b_safe[data, :, None]))[..., 0]
    uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
    llrs[data] = qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(len(data), N, 4)
    return llrs, prediction, pred_var


def _build_branch_summary(
    name: str,
    prior_state: State,
    obs: PublicObservation,
    y_b_safe: np.ndarray,
) -> CandidateSummary:
    pre_ll, pre_pred, pre_var = _pre_update_training_stats(prior_state, y_b_safe, obs.training_rows, obs.n0)
    post_state = _condition_n0(prior_state, y_b_safe, obs.training_rows, obs.n0)
    mean_ray, var_ray = sample_moments(post_state)
    llrs, probe_pred, probe_var = _readout_n0(
        mean_ray, var_ray, y_b_safe, obs.training_rows, obs.n0, probe_row=obs.probe_row
    )
    return CandidateSummary(
        name=name,
        pre_train_loglik=pre_ll,
        pre_train_pred=pre_pred,
        pre_train_var=pre_var,
        mean_ray=mean_ray,
        var_ray=var_ray,
        probe_pred=probe_pred,
        probe_var=probe_var,
        llrs=llrs,
    )


def build_candidates(obs: PublicObservation) -> Tuple[CandidateSummary, CandidateSummary]:
    """Construct ('inherit', 'clear') CandidateSummary pair strictly from PublicObservation."""
    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j

    tap_noise_a = (float(obs.n0) + 1.0 - float(np.sinc(0.0625) ** 2)) / len(EVEN)
    state_a = update(prior(), OFDM_PILOT_ROWS, observed_taps(obs.y_a, OFDM_PILOT_ROWS), row_noise=tap_noise_a)
    prior_inh = advance(state_a, FRAME_DURATION)
    prior_clr = advance(prior(), FRAME_DURATION)

    cand_inh = _build_branch_summary("inherit", prior_inh, obs, y_b_safe)
    cand_clr = _build_branch_summary("clear", prior_clr, obs, y_b_safe)
    return cand_inh, cand_clr


# ---------------------------------------------------------------------------
# Precomputed Dataset Caching & Batched Training
# ---------------------------------------------------------------------------


@dataclass
class CachedLink:
    seed: int
    regime: str
    obs: PublicObservation
    truth: HiddenTruth
    cand_inh: CandidateSummary
    cand_clr: CandidateSummary
    feats_full: GateFeatures
    feats_nohist: GateFeatures


def _payload_row_order(training_rows: tuple) -> List[int]:
    """Order of the 18 payload rows matching HiddenTruth.b_info / b_code ([:16] core, [16:] extra)."""
    return list(CORE_ROWS) + list(extra_rows(training_rows))


def build_dataset(
    seed_regime_pairs: Sequence[Tuple[int, str]],
    snr_db: float = PRIMARY_SNR_DB,
    budget: str = "sparse",
) -> List[CachedLink]:
    items: List[CachedLink] = []
    for seed, regime in seed_regime_pairs:
        obs, truth = simulate_p9_link(seed, regime, snr_db=snr_db, budget=budget)
        cand_inh, cand_clr = build_candidates(obs)
        feats_full = extract_gate_features(obs, cand_inh, cand_clr)
        feats_nohist = extract_gate_features(obs, cand_clr, cand_clr)
        items.append(
            CachedLink(
                seed=seed,
                regime=regime,
                obs=obs,
                truth=truth,
                cand_inh=cand_inh,
                cand_clr=cand_clr,
                feats_full=feats_full,
                feats_nohist=feats_nohist,
            )
        )
    return items


def _stack_training_tensors(items: Sequence[CachedLink], use_nohist: bool) -> Dict[str, torch.Tensor]:
    feats_list = [it.feats_nohist if use_nohist else it.feats_full for it in items]
    payload_rows = _payload_row_order(items[0].obs.training_rows)
    return {
        "scalar_feats": torch.from_numpy(np.stack([f.scalar_feats for f in feats_list])).to(REAL),
        "complex_feats": torch.from_numpy(np.stack([f.complex_feats for f in feats_list])).to(COMPLEX),
        "t_norm": torch.from_numpy(np.stack([f.t_norm for f in feats_list])).to(REAL),
        "ev_base": torch.from_numpy(np.stack([f.ev_base for f in feats_list])).to(REAL),
        "base_logit": torch.from_numpy(np.stack([f.base_logit for f in feats_list])).to(REAL),
        "llr_inh": torch.from_numpy(np.stack([it.cand_inh.llrs[payload_rows] for it in items])).to(REAL),
        "llr_clr": torch.from_numpy(np.stack([it.cand_clr.llrs[payload_rows] for it in items])).to(REAL),
        "b_code": torch.from_numpy(np.stack([it.truth.b_code.reshape(len(payload_rows), N, 4) for it in items])).to(REAL),
        "probe_pred_inh": torch.from_numpy(np.stack([it.cand_inh.probe_pred for it in items])).to(COMPLEX),
        "probe_pred_clr": torch.from_numpy(np.stack([it.cand_clr.probe_pred for it in items])).to(COMPLEX),
        "probe_var_inh": torch.from_numpy(np.stack([it.cand_inh.probe_var for it in items])).to(REAL),
        "probe_var_clr": torch.from_numpy(np.stack([it.cand_clr.probe_var for it in items])).to(REAL),
        "probe_target": torch.from_numpy(np.stack([it.truth.y_b_probe_noisy for it in items])).to(COMPLEX),
        "n0": torch.tensor([it.obs.n0 for it in items], dtype=REAL),
        "payload_rows": torch.tensor(payload_rows, dtype=torch.long),
    }


def _joint_loss_batch(
    w: torch.Tensor,
    logit: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint training loss: payload_bce + 0.05 * probe_nll (strictly C*-invariant)."""
    payload_rows = batch["payload_rows"]
    logit_pay = logit[:, payload_rows, None, None]  # [B, 18, 1, 1]
    llr_inh = batch["llr_inh"][idx]                 # [B, 18, 32, 4]
    llr_clr = batch["llr_clr"][idx]                 # [B, 18, 32, 4]
    bits = batch["b_code"][idx]                     # [B, 18, 32, 4]

    mix_llr = mix_bit_llrs_torch_from_logit(logit_pay, llr_inh, llr_clr)
    signed = 2.0 * bits - 1.0
    payload_bce = F.softplus(-signed * mix_llr).mean()

    wp = w[:, PROBE_ROW, None]                      # [B, 1]
    mu1 = batch["probe_pred_inh"][idx]              # [B, 32]
    mu0 = batch["probe_pred_clr"][idx]              # [B, 32]
    v1 = batch["probe_var_inh"][idx]                # [B, 32]
    v0 = batch["probe_var_clr"][idx]                # [B, 32]
    y_p = batch["probe_target"][idx]                # [B, 32]
    n0 = batch["n0"][idx, None].clamp_min(1e-300)   # [B, 1]

    mix_mu = wp * mu1 + (1.0 - wp) * mu0
    mix_var = (wp * v1 + (1.0 - wp) * v0 + wp * (1.0 - wp) * (mu1 - mu0).abs().square()).clamp_min(1e-300)
    err2 = (y_p - mix_mu).abs().square()
    probe_nll = (torch.log(mix_var / n0) + err2 / mix_var).mean()

    total = payload_bce + PROBE_LOSS_WEIGHT * probe_nll
    return total, payload_bce, probe_nll


def train_variant(
    variant: str,
    seed: int,
    train_tensors: Dict[str, torch.Tensor],
    regime_indices: Dict[str, np.ndarray],
    steps: int = TRAIN_STEPS,
) -> Tuple[TemporalTrustGate, Dict[str, float]]:
    """Train one TemporalTrustGate variant for one seed under identical trajectory & budget schedule."""
    configure_determinism()
    torch.manual_seed(seed)
    model = TemporalTrustGate(variant=variant, d_model=12, n_heads=4, n_layers=1)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=MIN_LR)

    # Identical minibatch trajectory schedule across all 4 variants for a given seed!
    batch_rng = np.random.default_rng(np.random.SeedSequence([SEED_P9_TRAIN, seed, 777]))

    t0 = time.perf_counter()
    final_loss = 0.0
    final_bce = 0.0
    final_nll = 0.0

    model.train()
    for _ in range(steps):
        chosen = []
        for reg in REGIMES_ID:
            pool = regime_indices[reg]
            chosen.append(batch_rng.choice(pool, size=BATCH_PER_REGIME, replace=False))
        idx_np = np.concatenate(chosen)
        idx = torch.from_numpy(idx_np).long()

        opt.zero_grad(set_to_none=True)
        w, logit = model.forward_tensors(
            train_tensors["scalar_feats"][idx],
            train_tensors["complex_feats"][idx],
            train_tensors["t_norm"][idx],
            train_tensors["ev_base"][idx],
            train_tensors["base_logit"][idx],
        )
        loss, bce, nll = _joint_loss_batch(w, logit, train_tensors, idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP,
        )
        opt.step()
        sched.step()
        final_loss = float(loss.detach())
        final_bce = float(bce.detach())
        final_nll = float(nll.detach())

    train_s = time.perf_counter() - t0
    model.eval()
    return model, {
        "train_seconds": round(train_s, 3),
        "final_loss": round(final_loss, 6),
        "final_payload_bce": round(final_bce, 6),
        "final_probe_nll": round(final_nll, 6),
    }


def fit_calibrated_classical_gate(
    train_tensors: Dict[str, torch.Tensor],
    regime_indices: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """Fit a 2-parameter scalar (tau, bias) on pre-update Bayes factor using the exact same training set."""
    configure_determinism()
    log_tau = torch.zeros(1, dtype=REAL, requires_grad=True)
    bias = torch.zeros(1, dtype=REAL, requires_grad=True)
    opt = torch.optim.Adam([log_tau, bias], lr=0.05)
    all_idx = torch.from_numpy(np.concatenate([regime_indices[r] for r in REGIMES_ID])).long()
    base_l = train_tensors["base_logit"][all_idx]
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        logit = torch.exp(log_tau) * base_l + bias
        w = torch.sigmoid(logit)
        loss, _, _ = _joint_loss_batch(w, logit, train_tensors, all_idx)
        loss.backward()
        opt.step()
    return float(torch.exp(log_tau).detach()), float(bias.detach())


# ---------------------------------------------------------------------------
# Evaluation & Computational Cost Profiling
# ---------------------------------------------------------------------------


def _score_gate_output(gate_out: GateOutput, item: CachedLink) -> Dict[str, float]:
    training_rows = item.obs.training_rows
    core_llrs = gate_out.mix_llrs[list(CORE_ROWS)].reshape(16, 128)
    core_wrong = ldpc_decode(core_llrs) != item.truth.b_info[:16]
    core_block_errors = int(core_wrong.any(axis=1).sum())

    free = list(extra_rows(training_rows))
    extra_block_errors = 0
    if free:
        extra_llrs = gate_out.mix_llrs[free].reshape(len(free), 128)
        extra_wrong = ldpc_decode(extra_llrs) != item.truth.b_info[16 : 16 + len(free)]
        extra_block_errors = int(extra_wrong.any(axis=1).sum())

    # Core payload bit BCE (nats).
    signed_core = 2.0 * item.truth.b_code[:16].astype(np.float64) - 1.0
    bit_bce = float(np.mean(np.logaddexp(0.0, -signed_core * core_llrs)))

    y_noisy = item.truth.y_b_probe_noisy
    y_clean = item.truth.y_b_probe_clean
    err_noisy2 = np.abs(y_noisy - gate_out.mix_probe_pred) ** 2
    err_clean2 = np.abs(y_clean - gate_out.mix_probe_pred) ** 2
    var_sum = float(gate_out.mix_probe_var.sum())
    nll_sum = float(np.sum(np.log(gate_out.mix_probe_var / item.obs.n0) + err_noisy2 / gate_out.mix_probe_var))

    w = gate_out.history_weight
    return {
        "core_blocks": 16,
        "core_errors": core_block_errors,
        "all_blocks": 16 + len(free),
        "all_errors": core_block_errors + extra_block_errors,
        "bit_bce": bit_bce,
        "probe_sse_noisy": float(err_noisy2.sum()),
        "probe_energy_noisy": float((np.abs(y_noisy) ** 2).sum()),
        "probe_sse_clean": float(err_clean2.sum()),
        "probe_energy_clean": float((np.abs(y_clean) ** 2).sum()),
        "probe_var_sum": var_sum,
        "probe_nll_sum": nll_sum,
        "probe_samples": len(y_noisy),
        "mean_w": float(w.mean()),
        "w_early": float(w[:6].mean()),
        "w_late": float(w[11:19].mean()),
        "w_probe": float(gate_out.probe_weight),
    }


def _aggregate_regime_scores(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    n_links = len(scores)
    core_blocks = sum(s["core_blocks"] for s in scores)
    core_errors = sum(s["core_errors"] for s in scores)
    all_blocks = sum(s["all_blocks"] for s in scores)
    all_errors = sum(s["all_errors"] for s in scores)
    probe_sse_noisy = sum(s["probe_sse_noisy"] for s in scores)
    probe_energy_noisy = sum(s["probe_energy_noisy"] for s in scores)
    probe_sse_clean = sum(s["probe_sse_clean"] for s in scores)
    probe_energy_clean = sum(s["probe_energy_clean"] for s in scores)
    probe_var_sum = sum(s["probe_var_sum"] for s in scores)
    probe_nll_sum = sum(s["probe_nll_sum"] for s in scores)
    probe_samples = sum(s["probe_samples"] for s in scores)
    return {
        "links": n_links,
        "core_blocks": core_blocks,
        "core_errors": core_errors,
        "bler_core": core_errors / core_blocks,
        "all_blocks": all_blocks,
        "all_errors": all_errors,
        "bler_all18": all_errors / all_blocks,
        "correct_blocks_per_link_core": (core_blocks - core_errors) / n_links,
        "correct_blocks_per_link_all18": (all_blocks - all_errors) / n_links,
        "bit_bce": float(np.mean([s["bit_bce"] for s in scores])),
        "probe_nmse_noisy": probe_sse_noisy / probe_energy_noisy,
        "probe_nmse_clean": probe_sse_clean / probe_energy_clean,
        "probe_cal_ratio": probe_sse_noisy / probe_var_sum,
        "probe_nll": probe_nll_sum / probe_samples,
        "mean_w": float(np.mean([s["mean_w"] for s in scores])),
        "w_early": float(np.mean([s["w_early"] for s in scores])),
        "w_late": float(np.mean([s["w_late"] for s in scores])),
        "w_probe": float(np.mean([s["w_probe"] for s in scores])),
    }


def evaluate_gate_fn(gate_fn, dev_by_regime: Dict[str, List[CachedLink]]) -> Dict[str, object]:
    per_regime: Dict[str, Dict[str, float]] = {}
    for reg, items in dev_by_regime.items():
        scores = [_score_gate_output(gate_fn(it), it) for it in items]
        per_regime[reg] = _aggregate_regime_scores(scores)

    def _equal_summary(reg_list: Sequence[str]) -> Dict[str, float]:
        return {
            "bler_core": float(np.mean([per_regime[r]["bler_core"] for r in reg_list])),
            "bler_all18": float(np.mean([per_regime[r]["bler_all18"] for r in reg_list])),
            "correct_blocks_per_link_core": float(np.mean([per_regime[r]["correct_blocks_per_link_core"] for r in reg_list])),
            "correct_blocks_per_link_all18": float(np.mean([per_regime[r]["correct_blocks_per_link_all18"] for r in reg_list])),
            "bit_bce": float(np.mean([per_regime[r]["bit_bce"] for r in reg_list])),
            "probe_nmse_noisy": float(np.mean([per_regime[r]["probe_nmse_noisy"] for r in reg_list])),
            "probe_nmse_clean": float(np.mean([per_regime[r]["probe_nmse_clean"] for r in reg_list])),
            "probe_cal_ratio": float(np.mean([per_regime[r]["probe_cal_ratio"] for r in reg_list])),
            "probe_nll": float(np.mean([per_regime[r]["probe_nll"] for r in reg_list])),
        }

    return {
        "id_equal_mean": _equal_summary(REGIMES_ID),
        "ood_equal_mean": _equal_summary(REGIMES_OOD),
        "per_regime": per_regime,
    }


def profile_end_to_end_cost(
    gate_callable,
    sample_items: Sequence[CachedLink],
    correct_blocks_per_link_core: float,
    correct_blocks_per_link_all18: float,
) -> Dict[str, float]:
    """Measure single-thread end-to-end CPU latency (ms/link), gate-only latency, peak memory (MB),
    and CPU ms per correctly decoded block via tracemalloc & perf_counter.
    """
    configure_determinism()
    # Warm-up 2 links.
    for it in sample_items[:2]:
        ci, cc = build_candidates(it.obs)
        go = gate_callable(it.obs, ci, cc)
        _ = ldpc_decode(go.mix_llrs[list(CORE_ROWS)].reshape(16, 128))

    # Pass 1: clean single-thread CPU latency WITHOUT tracemalloc instrumentation overhead.
    gate_times_ms: List[float] = []
    e2e_times_ms: List[float] = []
    for it in sample_items:
        t_start = time.perf_counter()
        ci, cc = build_candidates(it.obs)
        t_gate_start = time.perf_counter()
        go = gate_callable(it.obs, ci, cc)
        t_gate_end = time.perf_counter()
        rows_pay = _payload_row_order(it.obs.training_rows)
        _ = ldpc_decode(go.mix_llrs[rows_pay].reshape(len(rows_pay), 128))
        t_end = time.perf_counter()
        gate_times_ms.append((t_gate_end - t_gate_start) * 1000.0)
        e2e_times_ms.append((t_end - t_start) * 1000.0)

    # Pass 2: peak Python allocation memory via tracemalloc over a full link evaluation pass.
    tracemalloc.start()
    tracemalloc.reset_peak()
    for it in sample_items:
        ci, cc = build_candidates(it.obs)
        go = gate_callable(it.obs, ci, cc)
        rows_pay = _payload_row_order(it.obs.training_rows)
        _ = ldpc_decode(go.mix_llrs[rows_pay].reshape(len(rows_pay), 128))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    e2e_mean = float(np.mean(e2e_times_ms))
    e2e_median = float(np.median(e2e_times_ms))
    gate_mean = float(np.mean(gate_times_ms))
    gate_median = float(np.median(gate_times_ms))
    return {
        "n_profile_links": len(sample_items),
        "gate_latency_mean_ms": round(gate_mean, 4),
        "gate_latency_median_ms": round(gate_median, 4),
        "e2e_latency_mean_ms": round(e2e_mean, 4),
        "e2e_latency_median_ms": round(e2e_median, 4),
        "peak_memory_mb": round(peak_bytes / (1024.0 * 1024.0), 4),
        "cpu_ms_per_correct_block_core": round(e2e_mean / max(correct_blocks_per_link_core, 1e-9), 4),
        "cpu_ms_per_correct_block_all18": round(e2e_mean / max(correct_blocks_per_link_all18, 1e-9), 4),
    }


def _oracle_branch_gate(item: CachedLink) -> GateOutput:
    """Diagnostic-only upper bound picking the better branch (lower coded bit error) per row."""
    w = np.zeros(T, dtype=np.float64)
    payload_rows = _payload_row_order(item.obs.training_rows)
    for k, r in enumerate(payload_rows):
        b = item.truth.b_code[k]
        e_inh = int(np.count_nonzero((item.cand_inh.llrs[r].reshape(128) > 0) != b))
        e_clr = int(np.count_nonzero((item.cand_clr.llrs[r].reshape(128) > 0) != b))
        w[r] = 1.0 if e_inh <= e_clr else 0.0
    # Probe row oracle: lower probe squared error.
    p_inh = float(np.sum(np.abs(item.truth.y_b_probe_clean - item.cand_inh.probe_pred) ** 2))
    p_clr = float(np.sum(np.abs(item.truth.y_b_probe_clean - item.cand_clr.probe_pred) ** 2))
    w[PROBE_ROW] = 1.0 if p_inh <= p_clr else 0.0
    return compose_gate_output(w, item.cand_inh, item.cand_clr, probe_row=item.obs.probe_row)


def run_dev_study() -> Dict[str, object]:
    configure_determinism()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()

    # 1. Construct strictly disjoint train & dev seed lists.
    train_pairs: List[Tuple[int, str]] = []
    for r_idx, reg in enumerate(REGIMES_ID):
        for i in range(TRAIN_LINKS_PER_REGIME):
            seed = SEED_P9_TRAIN + 100000 * (r_idx + 1) + i
            train_pairs.append((seed, reg))

    dev_pairs_by_regime: Dict[str, List[Tuple[int, str]]] = {}
    all_regimes_eval = REGIMES_ID + REGIMES_OOD
    for r_idx, reg in enumerate(all_regimes_eval):
        pairs = []
        for c in range(1, DEV_CLUSTERS + 1):
            for i in range(DEV_LINKS_PER_CLUSTER):
                seed = DEV_SEED_BASE + 10000 * (r_idx + 1) + 1000 * c + i
                pairs.append((seed, reg))
        dev_pairs_by_regime[reg] = pairs

    train_seed_set = {s for s, _ in train_pairs}
    dev_seed_set = {s for pairs in dev_pairs_by_regime.values() for s, _ in pairs}
    assert train_seed_set.isdisjoint(dev_seed_set), "Train and dev seed sets must be strictly disjoint!"

    print(
        f"[P9-C] Generating datasets: {len(train_pairs)} train links "
        f"({TRAIN_LINKS_PER_REGIME}/regime x 3 ID), "
        f"{len(dev_seed_set)} dev links ({DEV_CLUSTERS}x{DEV_LINKS_PER_CLUSTER}=48/regime x 5 regimes)...",
        flush=True,
    )
    t_gen = time.perf_counter()
    train_items = build_dataset(train_pairs, snr_db=PRIMARY_SNR_DB, budget="sparse")
    dev_by_regime = {
        reg: build_dataset(pairs, snr_db=PRIMARY_SNR_DB, budget="sparse")
        for reg, pairs in dev_pairs_by_regime.items()
    }
    print(f"[P9-C] Dataset & branch precomputation done in {time.perf_counter() - t_gen:.2f}s.", flush=True)

    train_tensors_full = _stack_training_tensors(train_items, use_nohist=False)
    train_tensors_nohist = _stack_training_tensors(train_items, use_nohist=True)
    regime_indices = {
        reg: np.array([i for i, it in enumerate(train_items) if it.regime == reg], dtype=np.int64)
        for reg in REGIMES_ID
    }

    # 2. Evaluate classical baselines & calibrated 2-param pre-update Bayes baseline.
    cal_tau, cal_bias = fit_calibrated_classical_gate(train_tensors_full, regime_indices)
    print(f"[P9-C] Calibrated 2-param classical pre-update Bayes: tau={cal_tau:.4f}, bias={cal_bias:.4f}", flush=True)

    def _always_inherit_fn(obs, ci, cc):
        return compose_gate_output(np.ones(T, dtype=np.float64), ci, cc, probe_row=obs.probe_row)

    def _always_clear_fn(obs, ci, cc):
        return compose_gate_output(np.zeros(T, dtype=np.float64), ci, cc, probe_row=obs.probe_row)

    def _classical_cal_fn(obs, ci, cc):
        raw_bf = float(ci.pre_train_loglik) - float(cc.pre_train_loglik)
        clipped = float(np.clip(raw_bf, -35.0, 35.0))
        logit = cal_tau * clipped + cal_bias
        w_val = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
        return compose_gate_output(np.full(T, w_val, dtype=np.float64), ci, cc, probe_row=obs.probe_row)

    classical_eval = {
        "always_inherit": evaluate_gate_fn(lambda it: _always_inherit_fn(it.obs, it.cand_inh, it.cand_clr), dev_by_regime),
        "always_clear": evaluate_gate_fn(lambda it: _always_clear_fn(it.obs, it.cand_inh, it.cand_clr), dev_by_regime),
        "classical_pre_update_bayes": evaluate_gate_fn(
            lambda it: classical_pre_update_gate(it.obs, it.cand_inh, it.cand_clr), dev_by_regime
        ),
        "classical_pre_update_calibrated": evaluate_gate_fn(
            lambda it: _classical_cal_fn(it.obs, it.cand_inh, it.cand_clr), dev_by_regime
        ),
        "oracle_branch_per_row": evaluate_gate_fn(_oracle_branch_gate, dev_by_regime),
    }
    classical_eval["classical_pre_update_calibrated"]["fitted_params"] = {"tau": cal_tau, "bias": cal_bias}

    for cname in ("always_inherit", "always_clear", "classical_pre_update_bayes", "classical_pre_update_calibrated", "oracle_branch_per_row"):
        id_m = classical_eval[cname]["id_equal_mean"]
        ood_m = classical_eval[cname]["ood_equal_mean"]
        print(
            f"  [Baseline {cname:32s}] ID BLER={id_m['bler_core']*100:5.2f}% "
            f"(BCE={id_m['bit_bce']:.4f}, NMSE={id_m['probe_nmse_noisy']:.4f}, Cal={id_m['probe_cal_ratio']:.2f}x) | "
            f"OOD BLER={ood_m['bler_core']*100:5.2f}%",
            flush=True,
        )

    # 3. Train & evaluate all 4 neural variants across 3 seeds.
    neural_results: Dict[str, Dict[str, object]] = {}
    trained_models_seed0: Dict[str, TemporalTrustGate] = {}

    for variant in VARIANTS:
        tensors_v = train_tensors_nohist if variant == "no_history" else train_tensors_full
        per_seed: Dict[str, object] = {}
        param_count = None
        for seed in DEV_SEEDS:
            model, train_info = train_variant(variant, seed, tensors_v, regime_indices, steps=TRAIN_STEPS)
            param_count = model.real_parameters()
            if seed == DEV_SEEDS[0]:
                trained_models_seed0[variant] = model

            ckpt_path = CKPT_DIR / f"{variant}_seed{seed}.pt"
            torch.save(
                {
                    "variant": variant,
                    "seed": seed,
                    "real_parameters": param_count,
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )

            eval_out = evaluate_gate_fn(lambda it, m=model: m.gate(it.obs, it.cand_inh, it.cand_clr), dev_by_regime)
            per_seed[str(seed)] = {
                "train": train_info,
                "eval": eval_out,
                "ckpt": str(ckpt_path.relative_to(Path(__file__).parent)),
            }
            id_m = eval_out["id_equal_mean"]
            ood_m = eval_out["ood_equal_mean"]
            print(
                f"  [{variant:16s} seed={seed}] params={param_count} ({train_info['train_seconds']:.2f}s) | "
                f"ID BLER={id_m['bler_core']*100:5.2f}% (BCE={id_m['bit_bce']:.4f}, "
                f"NMSE={id_m['probe_nmse_noisy']:.4f}, Cal={id_m['probe_cal_ratio']:.2f}x) | "
                f"OOD BLER={ood_m['bler_core']*100:5.2f}%",
                flush=True,
            )

        # Aggregate mean & std across the 3 seeds.
        def _mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            keys = dicts[0].keys()
            return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

        def _std_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            keys = dicts[0].keys()
            return {k: float(np.std([d[k] for d in dicts], ddof=1)) for k in keys}

        id_list = [per_seed[str(s)]["eval"]["id_equal_mean"] for s in DEV_SEEDS]
        ood_list = [per_seed[str(s)]["eval"]["ood_equal_mean"] for s in DEV_SEEDS]
        reg_mean = {
            reg: _mean_dict([per_seed[str(s)]["eval"]["per_regime"][reg] for s in DEV_SEEDS])
            for reg in all_regimes_eval
        }
        neural_results[variant] = {
            "real_parameters": param_count,
            "mean_id_equal": _mean_dict(id_list),
            "std_id_equal": _std_dict(id_list),
            "mean_ood_equal": _mean_dict(ood_list),
            "std_ood_equal": _std_dict(ood_list),
            "mean_per_regime": reg_mean,
            "per_seed": per_seed,
        }

    # 4. Single-thread CPU latency, peak memory (tracemalloc), and CPU ms / correctly decoded block.
    profile_links = [dev_by_regime[r][i] for r in REGIMES_ID for i in range(6)]  # 18 balanced ID links
    cost_profile: Dict[str, Dict[str, float]] = {}

    cost_profile["classical_pre_update_bayes"] = {
        "real_parameters": 0,
        **profile_end_to_end_cost(
            classical_pre_update_gate,
            profile_links,
            classical_eval["classical_pre_update_bayes"]["id_equal_mean"]["correct_blocks_per_link_core"],
            classical_eval["classical_pre_update_bayes"]["id_equal_mean"]["correct_blocks_per_link_all18"],
        ),
    }
    cost_profile["classical_pre_update_calibrated"] = {
        "real_parameters": 2,
        **profile_end_to_end_cost(
            _classical_cal_fn,
            profile_links,
            classical_eval["classical_pre_update_calibrated"]["id_equal_mean"]["correct_blocks_per_link_core"],
            classical_eval["classical_pre_update_calibrated"]["id_equal_mean"]["correct_blocks_per_link_all18"],
        ),
    }
    for variant in VARIANTS:
        m0 = trained_models_seed0[variant]
        id_mean = neural_results[variant]["mean_id_equal"]
        cost_profile[variant] = {
            "real_parameters": m0.real_parameters(),
            **profile_end_to_end_cost(
                m0.gate,
                profile_links,
                id_mean["correct_blocks_per_link_core"],
                id_mean["correct_blocks_per_link_all18"],
            ),
        }

    # Verify exact integer parameter parity across all 4 neural variants.
    param_counts = {v: neural_results[v]["real_parameters"] for v in VARIANTS}
    assert len(set(param_counts.values())) == 1, f"Parameter parity broken: {param_counts}"

    report = {
        "meta": {
            "task": "P9-C Dev Gate Study (9 dB sparse OFDM->SC)",
            "snr_db": PRIMARY_SNR_DB,
            "budget": "sparse",
            "seeds": list(DEV_SEEDS),
            "train_seed_base": SEED_P9_TRAIN,
            "dev_seed_base": DEV_SEED_BASE,
            "train_links_per_regime": TRAIN_LINKS_PER_REGIME,
            "dev_clusters": DEV_CLUSTERS,
            "dev_links_per_cluster": DEV_LINKS_PER_CLUSTER,
            "dev_links_per_regime": DEV_CLUSTERS * DEV_LINKS_PER_CLUSTER,
            "regimes_id": list(REGIMES_ID),
            "regimes_ood": list(REGIMES_OOD),
            "hyperparams": {
                "d_model": 12,
                "n_heads": 4,
                "n_layers": 1,
                "steps": TRAIN_STEPS,
                "batch_size": BATCH_PER_REGIME * len(REGIMES_ID),
                "optimizer": "AdamW",
                "lr": LR,
                "min_lr": MIN_LR,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "probe_loss_weight": PROBE_LOSS_WEIGHT,
            },
            "parameter_parity": param_counts,
            "total_wall_seconds": round(time.perf_counter() - t_all, 2),
        },
        "classical_baselines": classical_eval,
        "neural_variants": neural_results,
        "cost_profile": cost_profile,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n[P9-C] Saved clean JSON report to {OUTPUT_JSON} ({report['meta']['total_wall_seconds']:.2f}s total).", flush=True)
    return report


if __name__ == "__main__":
    run_dev_study()
