"""P9-B: Statistical Inference & Cross-Waveform History-Trust Gating.

Self-contained explicit-N0 classical candidate builder, strict pre-update full-covariance
marginal predictive likelihood, symmetric 2-fold out-of-sample split-block predictive
scoring, C*-invariant Jeffreys divergence & Student-t hybrid partial-change gating,
probability-space branch mixing (with exact within + between total variance), and
locked branch-merged whole-word codeword scoring.

Strictly obeys p9/contract.py boundaries:
  - ZERO hardcoded N0 (all noise scaling derives from obs.n0 / explicit n0 argument).
  - ZERO access to obs.y_b[PROBE_ROW] anywhere in candidate building or gating.
  - Strict pre-update predictive likelihoods computed BEFORE conditioning on Frame B training.
  - Full float64/complex128 precision and C* = R+ x U(1) scale/phase invariance.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from p8.coding import qam_llrs
from p8.link import DELAYS, EVEN, N, PILOTS as OFDM_PILOTS, POWERS, T
from p8.tracker import State, advance, evolution, observed_taps, prior, update
from p8c.link import PILOTS as SC_PILOTS
from p8c.receiver import DELAYED_INDEX, SAMPLES, SWEEPS, sample_moments
from p9.contract import (
    FRAME_DURATION,
    OFDM_PILOT_ROWS,
    PROBE_ROW,
    CandidateSummary,
    GateOutput,
    PublicObservation,
    jeffreys_complex_gaussian,
    mix_bit_llrs,
)


_SINC_MID_SQ = float(np.sinc(0.0625) ** 2)
_ICI_UNIT_POWER = float(1.0 - _SINC_MID_SQ)
_ALL_SAMPLES = np.arange(N, dtype=np.int64)
_ALL_SAMPLES.setflags(write=False)


def _validate_rows(rows: tuple[int, ...], probe_row: int = PROBE_ROW) -> tuple[int, ...]:
    """Validate distinct Frame-B training rows strictly excluding the withheld probe row."""
    out = tuple(int(r) for r in rows)
    if len(set(out)) != len(out):
        raise ValueError("Training rows must be distinct")
    if any(r < 0 or r >= T or r == probe_row for r in out):
        raise ValueError(f"Training rows must lie in [0, {T - 1}] and exclude probe_row={probe_row}")
    return out


def _validate_sample_indices(sample_indices: Optional[np.ndarray]) -> tuple[int, ...]:
    if sample_indices is None:
        return tuple(range(N))
    idx = tuple(int(i) for i in np.asarray(sample_indices, dtype=np.int64).ravel())
    if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= N for i in idx):
        raise ValueError(f"sample_indices must be non-empty distinct integers in [0, {N - 1}]")
    return idx


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic sigmoid in float64."""
    arr = np.asarray(x, dtype=np.float64)
    pos = arr >= 0.0
    out = np.empty_like(arr, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ez = np.exp(arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return float(out) if np.ndim(x) == 0 else out


def _scale_state(state: State, complex_scale: complex | float) -> State:
    """Scale a tracker State's complex means by alpha and variances by |alpha|^2."""
    c = complex(complex_scale)
    s2 = float(abs(c) ** 2)
    return State(
        epoch=float(state.epoch),
        log_weight=np.asarray(state.log_weight, dtype=np.float64).copy(),
        mean=np.asarray(state.mean, dtype=np.complex128) * c,
        variance=np.asarray(state.variance, dtype=np.float64) * s2,
    )


def estimate_canonical_gauge(obs: PublicObservation) -> tuple[float, complex]:
    """Estimate full C* = R+ x U(1) canonical gauge (power_scale, unit_phase) strictly from Frame A pilots.

    NEVER inspects obs.y_b (neither training rows, data rows, nor PROBE_ROW).
    Under (y_a, y_b, n0) -> (alpha * y_a, alpha * y_b, |alpha|^2 * n0) for any alpha = r*e^{i*theta} in C*:
      - power_scale -> r^2 * power_scale
      - unit_phase  -> e^{i*theta} * unit_phase
    so (y / (sqrt(power_scale) * unit_phase), n0 / power_scale) is strictly C*-invariant to ~1e-15.
    """
    n0 = float(obs.n0)
    if not np.isfinite(n0) or n0 < 0.0:
        raise ValueError(f"obs.n0 must be finite and non-negative, got {obs.n0}")
    sel = np.ix_(OFDM_PILOT_ROWS, EVEN)
    pilots_a = np.asarray(obs.y_a, dtype=np.complex128)[sel]
    if not np.isfinite(pilots_a).all():
        raise ValueError("obs.y_a pilot cells contain non-finite values")
    ref_pilots = OFDM_PILOTS[sel]
    mags = np.abs(pilots_a)
    peak = float(np.max(mags))
    if peak > 0.0:
        p_unit = pilots_a / peak
        unit_energy = float(np.mean(np.abs(p_unit) ** 2))
        e_a = (peak * peak) * unit_energy
        proj = complex(np.sum(p_unit * ref_pilots.conj()))
        if abs(proj) > 1e-12:
            unit_phase = proj / abs(proj)
        else:
            dom = int(np.argmax(mags))
            dom_vec = complex(p_unit.ravel()[dom] * ref_pilots.ravel()[dom].conj())
            unit_phase = dom_vec / abs(dom_vec) if abs(dom_vec) > 0.0 else 1.0 + 0.0j
    else:
        e_a = 0.0
        unit_phase = 1.0 + 0.0j
    power_scale = float(max(e_a - n0, 0.05 * e_a, 0.05 * n0, 1e-300))
    return power_scale, complex(unit_phase)


def estimate_power_scale(obs: PublicObservation) -> float:
    """Estimate homogeneous degree-2 channel power scale strictly from Frame A public pilots."""
    power_scale, _ = estimate_canonical_gauge(obs)
    return power_scale


def build_prior_states(obs: PublicObservation) -> tuple[State, State]:
    """Construct pre-B-training prior bank states (inherit, clear) at Frame B's epoch.

    Strictly precedes any access to obs.y_b. Uses explicit obs.n0 and full C* = R+ x U(1)
    canonical gauge conditioning so Frame A tracking and public common prior are strictly
    C*-equivariant across arbitrary (alpha * y, |alpha|^2 * n0) scaling and phase rotations.
    """
    n0 = float(obs.n0)
    power_scale, unit_phase = estimate_canonical_gauge(obs)
    complex_scale = complex(np.sqrt(power_scale) * unit_phase)
    n0_canon = n0 / power_scale

    y_a_canon = np.asarray(obs.y_a, dtype=np.complex128) / complex_scale
    tap_noise_canon = (n0_canon + _ICI_UNIT_POWER) / float(len(EVEN))
    tap_obs_canon = observed_taps(y_a_canon, OFDM_PILOT_ROWS)

    state_a_canon = update(prior(), OFDM_PILOT_ROWS, tap_obs_canon, row_noise=tap_noise_canon)
    state_inh_canon = advance(state_a_canon, FRAME_DURATION)
    state_clr_canon = advance(prior(), FRAME_DURATION)

    return _scale_state(state_inh_canon, complex_scale), _scale_state(state_clr_canon, complex_scale)


@lru_cache(maxsize=32)
def _training_design_cached(
    epoch: float,
    rows: tuple[int, ...],
    sample_idx: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-gain per-sample responses [3, H, M], conjugate, and energies [3, H]."""
    idx = np.asarray(sample_idx, dtype=np.int64)
    rotations = evolution(float(epoch))[:, list(rows)][:, :, idx]  # [H, R, S]
    pilots = SC_PILOTS[list(rows)]  # [R, N]
    delayed = np.stack([np.roll(pilots, int(d), axis=-1)[:, idx] for d in DELAYS], axis=0)  # [3, R, S]
    design = np.ascontiguousarray(
        (delayed[:, None, :, :] * rotations[None, :, :, :]).reshape(len(DELAYS), rotations.shape[0], -1),
        dtype=np.complex128,
    )
    conj = np.ascontiguousarray(design.conj(), dtype=np.complex128)
    energy = np.ascontiguousarray(np.sum(np.abs(design) ** 2, axis=-1), dtype=np.float64)
    design.setflags(write=False)
    conj.setflags(write=False)
    energy.setflags(write=False)
    return design, conj, energy


def predictive_distribution(
    state: State,
    rows: tuple[int, ...],
    n0: float,
    sample_indices: Optional[np.ndarray] = None,
    probe_row: int = PROBE_ROW,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pre-update marginal predictive mean [M] and full Hermitian covariance [M, M].

    For each ray l in {0, 1, 2}, marginalizes over the ray's Doppler/chirp bank
    (log_weight, mean, variance) via p8.tracker.evolution(state.epoch), accounting for
    all exact intra-ray correlations across samples q in 0..M-1:

        mu_l   = sum_h w_{l,h} * mean_{l,h} * d_{l,h}
        Cov_l  = sum_h w_{l,h} * (|mean_{l,h}|^2 + var_{l,h}) * (d_{l,h} d_{l,h}^H) - mu_l mu_l^H
        Sigma  = n0 * I_M + sum_l Cov_l
    """
    rows = _validate_rows(rows, probe_row=probe_row)
    sample_idx = _validate_sample_indices(sample_indices)
    n0 = float(n0)
    if not np.isfinite(n0) or n0 < 0.0:
        raise ValueError(f"n0 must be finite and non-negative, got {n0}")

    design, conj, _ = _training_design_cached(float(state.epoch), rows, sample_idx)
    m_dim = design.shape[-1]

    weights = np.exp(np.asarray(state.log_weight, dtype=np.float64))  # [3, H]
    mean = np.asarray(state.mean, dtype=np.complex128)  # [3, H]
    var = np.asarray(state.variance, dtype=np.float64)  # [3, H]

    mu_total = np.zeros(m_dim, dtype=np.complex128)
    cov_total = np.zeros((m_dim, m_dim), dtype=np.complex128)
    second = (np.abs(mean) ** 2) + var  # [3, H]

    for ray in range(len(DELAYS)):
        d_ray = design[ray]  # [H, M]
        dc_ray = conj[ray]  # [H, M]
        mu_ray = (weights[ray] * mean[ray]) @ d_ray  # [M]
        a_ray = weights[ray] * second[ray]  # [H]
        # (d_ray.T * a_ray) @ d_ray.conj() has (p, q) = sum_h d_{h,p} * a_h * conj(d_{h,q})
        gram = (d_ray.T * a_ray[None, :]) @ dc_ray
        cov_ray = gram - np.outer(mu_ray, mu_ray.conj())
        mu_total += mu_ray
        cov_total += cov_ray

    cov_total[np.diag_indices(m_dim)] += n0
    # Enforce exact Hermitian symmetry numerically
    cov_total = 0.5 * (cov_total + cov_total.conj().T)
    return mu_total, cov_total


def complex_gaussian_loglik(y: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Exact proper complex Gaussian log-density log CN(y; mu, cov) via scale-safe Cholesky."""
    y_vec = np.asarray(y, dtype=np.complex128).ravel()
    mu_vec = np.asarray(mu, dtype=np.complex128).ravel()
    cov_mat = np.asarray(cov, dtype=np.complex128)
    m_dim = y_vec.size
    if mu_vec.size != m_dim or cov_mat.shape != (m_dim, m_dim):
        raise ValueError(f"Shape mismatch: y={y_vec.shape}, mu={mu_vec.shape}, cov={cov_mat.shape}")
    if not (np.isfinite(y_vec).all() and np.isfinite(mu_vec).all() and np.isfinite(cov_mat).all()):
        raise ValueError("Non-finite input to complex_gaussian_loglik")

    # Scale-canonicalize cov to O(1) diagonal to prevent underflow/overflow under extreme C* scaling
    diag_real = np.real(np.diag(cov_mat))
    scale = float(np.mean(np.maximum(diag_real, 0.0)))
    if scale <= 0.0:
        raise ValueError("Covariance matrix must have strictly positive diagonal")

    cov_unit = 0.5 * (cov_mat + cov_mat.conj().T) / scale
    try:
        chol = np.linalg.cholesky(cov_unit)
    except np.linalg.LinAlgError:
        # Tiny relative jitter only if exact n0 == 0 makes covariance numerically semi-definite
        jitter = 1e-12
        chol = np.linalg.cholesky(cov_unit + jitter * np.eye(m_dim, dtype=np.complex128))

    resid_unit = (y_vec - mu_vec) / np.sqrt(scale)
    whitened = solve_triangular(chol, resid_unit, lower=True, check_finite=False)
    quad = float(np.sum(np.abs(whitened) ** 2))
    logdet_unit = float(2.0 * np.sum(np.log(np.real(np.diag(chol)))))
    logdet = logdet_unit + m_dim * float(np.log(scale))
    return float(-m_dim * np.log(np.pi) - logdet - quad)


def condition_sc(
    prior_state: State,
    y_b: np.ndarray,
    rows: tuple[int, ...],
    n0: float,
    sample_indices: Optional[np.ndarray] = None,
    sweeps: int = SWEEPS,
    probe_row: int = PROBE_ROW,
) -> State:
    """Explicit-N0 coordinate ascent on independent ray factors over Frame B training samples.

    Supports full training blocks (sample_indices=None) and half-block splits for
    out-of-sample cross-prediction. NEVER indexes y_b[probe_row].
    """
    rows = _validate_rows(rows, probe_row=probe_row)
    sample_idx = _validate_sample_indices(sample_indices)
    n0 = float(n0)
    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError(f"condition_sc requires finite positive n0, got {n0}")
    if not rows:
        return prior_state

    idx_arr = np.asarray(sample_idx, dtype=np.int64)
    y_obs = np.asarray(y_b, dtype=np.complex128)[np.ix_(list(rows), idx_arr)].reshape(-1)
    if not np.isfinite(y_obs).all():
        raise ValueError("Training observations in y_b contain non-finite values")

    # Canonicalize scale to O(1) for numerical precision and exact C*-equivariance
    ref_var = max(float(np.mean(prior_state.variance)), n0, 1e-300)
    amp_scale = float(np.sqrt(ref_var))
    n0_c = n0 / ref_var
    y_c = y_obs / amp_scale
    prior_mean_c = np.asarray(prior_state.mean, dtype=np.complex128) / amp_scale
    prior_var_c = np.asarray(prior_state.variance, dtype=np.float64) / ref_var
    prior_lw = np.asarray(prior_state.log_weight, dtype=np.float64)

    design, conj, energy = _training_design_cached(float(prior_state.epoch), rows, sample_idx)
    post_var_c = prior_var_c / (1.0 + prior_var_c * energy / n0_c)
    log_weight = prior_lw.copy()
    mean_c = prior_mean_c.copy()
    per_ray = np.einsum("lh,lhq->lq", np.exp(log_weight) * mean_c, design)
    initial_precision_mean = prior_mean_c / prior_var_c
    evidence_base = (
        prior_lw
        + np.log(post_var_c / prior_var_c)
        - (np.abs(prior_mean_c) ** 2) / prior_var_c
    )

    for _ in range(int(sweeps)):
        for ray in range(len(DELAYS)):
            residual = y_c - per_ray.sum(axis=0) + per_ray[ray]
            matched = conj[ray] @ residual
            natural_mean = initial_precision_mean[ray] + matched / n0_c
            mean_c[ray] = post_var_c[ray] * natural_mean
            evidence = evidence_base[ray] + post_var_c[ray] * (np.abs(natural_mean) ** 2)
            log_weight[ray] = evidence - logsumexp(evidence)
            per_ray[ray] = (np.exp(log_weight[ray]) * mean_c[ray]) @ design[ray]

    return State(
        epoch=float(prior_state.epoch),
        log_weight=log_weight,
        mean=mean_c * amp_scale,
        variance=post_var_c * ref_var,
    )


def readout_sc(
    mean_ray: np.ndarray,
    var_ray: np.ndarray,
    y_b: np.ndarray,
    rows: tuple[int, ...],
    n0: float,
    probe_row: int = PROBE_ROW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit-N0 CP-SC withheld-probe prediction and data-block LMMSE soft bit LLRs.

    NEVER indexes y_b[probe_row].
    """
    rows = _validate_rows(rows, probe_row=probe_row)
    n0 = float(n0)
    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError(f"readout_sc requires finite positive n0, got {n0}")

    mean_ray = np.asarray(mean_ray, dtype=np.complex128)
    var_ray = np.asarray(var_ray, dtype=np.float64)

    known = SC_PILOTS[probe_row, DELAYED_INDEX]  # [3, N]
    probe_pred = (mean_ray[:, probe_row] * known).sum(axis=0)  # [N]
    probe_var = n0 + (var_ray[:, probe_row] * (np.abs(known) ** 2)).sum(axis=0)  # [N]

    data_rows = np.array([r for r in range(T) if r not in (*rows, probe_row)], dtype=np.int64)
    llrs = np.zeros((T, N, 4), dtype=np.float64)
    if len(data_rows) == 0:
        return llrs, probe_pred, probe_var

    y_data = np.asarray(y_b, dtype=np.complex128)[data_rows]
    if not np.isfinite(y_data).all():
        raise ValueError("Data rows in y_b contain non-finite values")

    # Canonicalize scale to O(1) so linear solve is identically conditioned under any C* scaling
    ref_var = max(float(np.mean(np.abs(mean_ray[:, data_rows]) ** 2)), n0, 1e-300)
    amp_scale = float(np.sqrt(ref_var))
    mean_d = mean_ray[:, data_rows] / amp_scale
    var_d = var_ray[:, data_rows] / ref_var
    y_d = y_data / amp_scale
    n0_c = n0 / ref_var

    noise = n0_c + var_d.sum(axis=0)  # [len(data_rows), N]
    design = np.zeros((len(data_rows), N, N), dtype=np.complex128)
    for ray, delayed in enumerate(DELAYED_INDEX):
        design[:, SAMPLES, delayed] = mean_d[ray]
    adjoint = design.conj().swapaxes(-1, -2) / noise[:, None, :]
    eye_n = np.eye(N, dtype=np.complex128)[None, :, :]
    cov = np.linalg.inv(eye_n + adjoint @ design)
    estimate = (cov @ (adjoint @ y_d[:, :, None]))[..., 0]
    uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
    llrs[data_rows] = qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(len(data_rows), N, 4)
    return llrs, probe_pred, probe_var


def _build_one_candidate(
    name: str,
    prior_state: State,
    obs: PublicObservation,
    power_scale: Optional[float] = None,
    unit_phase: Optional[complex] = None,
) -> CandidateSummary:
    """Strict 2-stage pipeline: (1) PRE-UPDATE predictive scoring, then (2) Frame B conditioning.

    Runs all linear algebra in the Frame-A C* canonical gauge (power_scale, unit_phase)
    and transforms equivariant outputs back to physical units, guaranteeing <1e-13
    C* invariance/equivariance across arbitrary (alpha * y, |alpha|^2 * n0).
    """
    rows = _validate_rows(obs.training_rows, probe_row=obs.probe_row)
    n0 = float(obs.n0)
    if power_scale is None or unit_phase is None:
        power_scale, unit_phase = estimate_canonical_gauge(obs)
    complex_scale = complex(np.sqrt(power_scale) * unit_phase)
    n0_c = n0 / power_scale

    # Canonicalize prior_state and y_b (strictly excluding probe_row)
    prior_c = _scale_state(prior_state, 1.0 / complex_scale)
    y_b_c = np.empty_like(obs.y_b, dtype=np.complex128)
    valid_rows = [r for r in range(T) if r != obs.probe_row]
    y_b_c[valid_rows] = np.asarray(obs.y_b, dtype=np.complex128)[valid_rows] / complex_scale
    y_b_c[obs.probe_row] = complex(np.nan, np.nan)
    y_train_c = y_b_c[list(rows)]  # [R, N]

    # STAGE 1: STRICT PRE-UPDATE PREDICTIVE DISTRIBUTION (ZERO Frame B conditioning!)
    mu_tr_c, cov_tr_c = predictive_distribution(
        prior_c, rows, n0=n0_c, sample_indices=None, probe_row=obs.probe_row
    )
    ll_c = complex_gaussian_loglik(y_train_c.reshape(-1), mu_tr_c, cov_tr_c)
    pre_train_loglik = float(ll_c - y_train_c.size * np.log(power_scale))
    pre_train_pred = (mu_tr_c * complex_scale).reshape(len(rows), N)
    pre_train_var = (np.maximum(np.real(np.diag(cov_tr_c)), 1e-300) * power_scale).reshape(len(rows), N)

    # STAGE 2: CONDITION ON FRAME B TRAINING ROWS & READ OUT DATA LLRs + WITHHELD PROBE
    post_c = condition_sc(
        prior_c, y_b_c, rows, n0=n0_c, sample_indices=None, probe_row=obs.probe_row
    )
    mean_ray_c, var_ray_c = sample_moments(post_c)
    llrs, probe_pred_c, probe_var_c = readout_sc(
        mean_ray_c, var_ray_c, y_b_c, rows, n0=n0_c, probe_row=obs.probe_row
    )

    return CandidateSummary(
        name=name,
        pre_train_loglik=pre_train_loglik,
        pre_train_pred=np.asarray(pre_train_pred, dtype=np.complex128),
        pre_train_var=np.asarray(pre_train_var, dtype=np.float64),
        mean_ray=np.asarray(mean_ray_c * complex_scale, dtype=np.complex128),
        var_ray=np.asarray(var_ray_c * power_scale, dtype=np.float64),
        probe_pred=np.asarray(probe_pred_c * complex_scale, dtype=np.complex128),
        probe_var=np.asarray(probe_var_c * power_scale, dtype=np.float64),
        llrs=np.asarray(llrs, dtype=np.float64),
    )


def build_candidates_with_priors(
    obs: PublicObservation,
) -> tuple[State, State, CandidateSummary, CandidateSummary]:
    """Build both pre-update prior states and full CandidateSummary objects ('inherit', 'clear')."""
    power_scale, unit_phase = estimate_canonical_gauge(obs)
    state_inh_prior, state_clr_prior = build_prior_states(obs)
    cand_inh = _build_one_candidate("inherit", state_inh_prior, obs, power_scale, unit_phase)
    cand_clr = _build_one_candidate("clear", state_clr_prior, obs, power_scale, unit_phase)
    return state_inh_prior, state_clr_prior, cand_inh, cand_clr


def build_candidates(obs: PublicObservation) -> tuple[CandidateSummary, CandidateSummary]:
    """Explicit-N0 self-contained candidate builder returning (cand_inherit, cand_clear).

    NEVER reads obs.y_b[obs.probe_row]. Computes exact full-covariance pre-update
    predictive log-likelihoods strictly BEFORE conditioning on obs.training_rows.
    """
    _, _, cand_inh, cand_clr = build_candidates_with_priors(obs)
    return cand_inh, cand_clr


# --- Information-Geometric Diagnostics (p8_manifold Construction B, C*-invariant) ---

def pre_update_jeffreys(cand_inh: CandidateSummary, cand_clr: CandidateSummary) -> np.ndarray:
    """Exact C*-invariant pre-update Jeffreys divergence D_J(p_inherit || p_clear) on training rows."""
    return jeffreys_complex_gaussian(
        cand_inh.pre_train_pred,
        cand_inh.pre_train_var,
        cand_clr.pre_train_pred,
        cand_clr.pre_train_var,
    )


def post_update_jeffreys_rays(
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    n0: float,
) -> np.ndarray:
    """Exact C*-invariant post-update per-ray Jeffreys divergence across all [3, T, N] cells."""
    n0 = float(n0)
    # Include scale-equivariant physical per-ray observation noise floor n0 * POWERS[l]
    floor = np.maximum(n0 * POWERS[:, None, None], 1e-300)
    return jeffreys_complex_gaussian(
        cand_inh.mean_ray,
        cand_inh.var_ray + floor,
        cand_clr.mean_ray,
        cand_clr.var_ray + floor,
    )


def post_update_jeffreys_rows(
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    n0: float,
) -> np.ndarray:
    """Row-aggregated post-update C*-invariant Jeffreys divergence [T] across all T=20 rows."""
    dj_rays = post_update_jeffreys_rays(cand_inh, cand_clr, n0)  # [3, T, N]
    return np.mean(np.sum(dj_rays, axis=0), axis=-1)  # [T]


# --- Split-Block Out-of-Sample Predictive Scoring ---

def split_block_predictive_scores(
    obs: PublicObservation,
    state_inh_prior: State,
    state_clr_prior: State,
) -> tuple[float, float]:
    """Symmetric 2-fold out-of-sample split-training predictive log-likelihoods.

    Fold 1: condition on samples [0..15] of training rows, score held-out [16..31].
    Fold 2: condition on samples [16..31] of training rows, score held-out [0..15].
    Strictly out-of-sample and C*-invariant (<1e-13) in (score_inh - score_clr).
    """
    rows = _validate_rows(obs.training_rows, probe_row=obs.probe_row)
    n0 = float(obs.n0)
    power_scale, unit_phase = estimate_canonical_gauge(obs)
    complex_scale = complex(np.sqrt(power_scale) * unit_phase)
    n0_c = n0 / power_scale
    y_b_c = np.full_like(obs.y_b, complex(np.nan, np.nan), dtype=np.complex128)
    y_b_c[list(rows)] = np.asarray(obs.y_b, dtype=np.complex128)[list(rows)] / complex_scale
    st_inh_c = _scale_state(state_inh_prior, 1.0 / complex_scale)
    st_clr_c = _scale_state(state_clr_prior, 1.0 / complex_scale)

    half_a = np.arange(0, N // 2, dtype=np.int64)
    half_b = np.arange(N // 2, N, dtype=np.int64)
    y_tr_c = y_b_c[list(rows)]  # [R, N]
    y_a_c = y_tr_c[:, half_a].reshape(-1)
    y_b_half_c = y_tr_c[:, half_b].reshape(-1)
    m_total = y_a_c.size + y_b_half_c.size
    jacobian_shift = float(-m_total * np.log(power_scale))

    def _branch_score_canon(prior_state_c: State) -> float:
        st_a = condition_sc(
            prior_state_c, y_b_c, rows, n0=n0_c, sample_indices=half_a, probe_row=obs.probe_row
        )
        mu_b, cov_b = predictive_distribution(
            st_a, rows, n0=n0_c, sample_indices=half_b, probe_row=obs.probe_row
        )
        ll_ab = complex_gaussian_loglik(y_b_half_c, mu_b, cov_b)

        st_b = condition_sc(
            prior_state_c, y_b_c, rows, n0=n0_c, sample_indices=half_b, probe_row=obs.probe_row
        )
        mu_a, cov_a = predictive_distribution(
            st_b, rows, n0=n0_c, sample_indices=half_a, probe_row=obs.probe_row
        )
        ll_ba = complex_gaussian_loglik(y_a_c, mu_a, cov_a)
        return float(ll_ab + ll_ba)

    sc_inh_c = _branch_score_canon(st_inh_c)
    sc_clr_c = _branch_score_canon(st_clr_c)
    return float(sc_inh_c + jacobian_shift), float(sc_clr_c + jacobian_shift)


# --- Exact Branch Mixture Helper ---

def combine_candidates(
    history_weight: np.ndarray,
    probe_weight: float,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
) -> GateOutput:
    """Combine 'inherit' and 'clear' branches via exact probability-space LLR mixing
    and exact total variance (within + between) on both probe predictions and ray moments.
    """
    w_rows = np.clip(np.asarray(history_weight, dtype=np.float64), 0.0, 1.0)
    if w_rows.shape != (T,):
        raise ValueError(f"history_weight must have shape ({T},), got {w_rows.shape}")
    w_probe = float(np.clip(probe_weight, 0.0, 1.0))

    mix_llrs = mix_bit_llrs(w_rows[:, None, None], cand_inh.llrs, cand_clr.llrs)

    mix_probe_pred = w_probe * cand_inh.probe_pred + (1.0 - w_probe) * cand_clr.probe_pred
    probe_between = w_probe * (1.0 - w_probe) * (np.abs(cand_inh.probe_pred - cand_clr.probe_pred) ** 2)
    mix_probe_var = w_probe * cand_inh.probe_var + (1.0 - w_probe) * cand_clr.probe_var + probe_between

    w3 = w_rows[None, :, None]
    mix_mean_ray = w3 * cand_inh.mean_ray + (1.0 - w3) * cand_clr.mean_ray
    ray_between = w3 * (1.0 - w3) * (np.abs(cand_inh.mean_ray - cand_clr.mean_ray) ** 2)
    mix_var_ray = w3 * cand_inh.var_ray + (1.0 - w3) * cand_clr.var_ray + ray_between

    return GateOutput(
        history_weight=w_rows,
        probe_weight=w_probe,
        mix_llrs=np.asarray(mix_llrs, dtype=np.float64),
        mix_probe_pred=np.asarray(mix_probe_pred, dtype=np.complex128),
        mix_probe_var=np.asarray(mix_probe_var, dtype=np.float64),
        mix_mean_ray=np.asarray(mix_mean_ray, dtype=np.complex128),
        mix_var_ray=np.asarray(mix_var_ray, dtype=np.float64),
    )


# --- Five Classical Gating Policies ---

def gate_always_inherit(cand_inh: CandidateSummary, cand_clr: CandidateSummary) -> GateOutput:
    """Fixed baseline: always trust inherited Frame A history (w = 1.0)."""
    return combine_candidates(np.ones(T, dtype=np.float64), 1.0, cand_inh, cand_clr)


def gate_always_clear(cand_inh: CandidateSummary, cand_clr: CandidateSummary) -> GateOutput:
    """Fixed baseline: always discard Frame A history and re-estimate on Frame B (w = 0.0)."""
    return combine_candidates(np.zeros(T, dtype=np.float64), 0.0, cand_inh, cand_clr)


def gate_pre_bayes_factor(
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    prior_odds: float = 1.0,
    temp: float = 1.0,
) -> GateOutput:
    """Exact pre-update marginal predictive Bayes factor gate.

    w = sigmoid(temp * (cand_inh.pre_train_loglik - cand_clr.pre_train_loglik) + log(prior_odds))
    """
    if prior_odds <= 0.0:
        raise ValueError(f"prior_odds must be strictly positive, got {prior_odds}")
    delta_ll = float(cand_inh.pre_train_loglik) - float(cand_clr.pre_train_loglik)
    w = float(_sigmoid(float(temp) * delta_ll + float(np.log(prior_odds))))
    return combine_candidates(np.full(T, w, dtype=np.float64), w, cand_inh, cand_clr)


def gate_split_predictive(
    obs: PublicObservation,
    state_inh_prior: State,
    state_clr_prior: State,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    temp: float = 1.0,
) -> GateOutput:
    """Symmetric 2-fold out-of-sample split-training predictive Bayes factor gate."""
    score_inh, score_clr = split_block_predictive_scores(obs, state_inh_prior, state_clr_prior)
    w = float(_sigmoid(float(temp) * (score_inh - score_clr)))
    return combine_candidates(np.full(T, w, dtype=np.float64), w, cand_inh, cand_clr)


def _complex_student_t_loglik(
    z: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Proper circular complex Student-t log density (up to nu-only constant, cancels in LLR).

    For complex scalar z with scale parameter v > 0 and nu > 0 real degrees of freedom
    (equivalent to 2D isotropic real Student-t with covariance (v/2) I_2):
        log p_t(z | mu, v; nu) = const(nu) - log(v) - (nu/2 + 1) * log(1 + 2 * |z - mu|^2 / (nu * v))
    """
    v = np.maximum(np.asarray(var, dtype=np.float64), 1e-300)
    u = (np.abs(np.asarray(z, dtype=np.complex128) - np.asarray(mu, dtype=np.complex128)) ** 2) / v
    return -np.log(v) - (0.5 * nu + 1.0) * np.log1p(2.0 * u / nu)


def gate_jeffreys_student_t(
    obs: PublicObservation,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    nu: float = 4.0,
) -> GateOutput:
    """Hybrid partial-change gate combining:
      1. Orthogonal ZC per-ray pre-update Student-t log-likelihood ratios (protects continuous
         dominant rays when a minor ray shifts in Regime 2 partial_shift),
      2. Pre-update per-sample Student-t innovation ratio + robust tempered Bayes factor,
      3. C*-invariant pre-update & post-update Jeffreys divergence D_J(p_inherit || p_clear)
         for row-adaptive extrapolation-risk tempering across all T=20 rows.

    Strictly C*-invariant (<1e-12), strictly pre-update on training innovations, and
    strictly isolated from obs.y_b[PROBE_ROW].
    """
    nu = float(nu)
    if nu <= 0.0:
        raise ValueError(f"Student-t degrees of freedom nu must be > 0, got {nu}")
    rows = _validate_rows(obs.training_rows, probe_row=obs.probe_row)
    n0 = float(obs.n0)
    y_tr = np.asarray(obs.y_b, dtype=np.complex128)[list(rows)]  # [R, N]
    n_rows = len(rows)

    # Exact scale-equivariant channel power scale recovered from clear pre-update variance
    # (exact identity: mean(cand_clr.pre_train_var) == n0 + power_scale * sum(POWERS))
    power_scale = max(float(np.mean(cand_clr.pre_train_var)) - n0, 1e-12 * n0, 1e-300)

    # 1. Per-ray orthogonal Zadoff-Chu matched-filter pre-update Student-t LLR & outlier weights
    pilots_tr = SC_PILOTS[list(rows)]  # [R, N]
    ray_llr = np.zeros(len(DELAYS), dtype=np.float64)
    ray_u = np.zeros(len(DELAYS), dtype=np.float64)
    ray_inlier = np.zeros(len(DELAYS), dtype=np.float64)

    excess_inh_row = np.maximum(np.mean(cand_inh.pre_train_var, axis=-1) - n0, 0.0)  # [R]
    # Add small scale-equivariant floor for intra-block Doppler mismatch on inherit projection
    v_noise_proj = n0 / float(N)
    for ray_idx, delay in enumerate(DELAYS):
        p_del = np.roll(pilots_tr, int(delay), axis=-1)  # [R, N], unit modulus, orthogonal across rays
        z_ray = np.mean(p_del.conj() * y_tr, axis=-1)  # [R]
        mu_inh_ray = np.mean(p_del.conj() * cand_inh.pre_train_pred, axis=-1)  # [R]
        mu_clr_ray = np.zeros(n_rows, dtype=np.complex128)

        p_l = float(POWERS[ray_idx])
        v_clr_ray = np.full(
            n_rows,
            power_scale * p_l * _SINC_MID_SQ + v_noise_proj + power_scale * _ICI_UNIT_POWER / float(N),
            dtype=np.float64,
        )
        v_inh_ray = (
            excess_inh_row * p_l * _SINC_MID_SQ
            + v_noise_proj
            + (excess_inh_row + 0.05 * power_scale * p_l) * _ICI_UNIT_POWER / float(N)
        )
        v_inh_ray = np.maximum(v_inh_ray, 1e-12 * power_scale)

        ll_inh = _complex_student_t_loglik(z_ray, mu_inh_ray, v_inh_ray, nu=nu)
        ll_clr = _complex_student_t_loglik(z_ray, mu_clr_ray, v_clr_ray, nu=nu)
        ray_llr[ray_idx] = float(np.sum(ll_inh - ll_clr))

        u_ray = float(np.mean((np.abs(z_ray - mu_inh_ray) ** 2) / v_inh_ray))
        ray_u[ray_idx] = u_ray
        ray_inlier[ray_idx] = float(nu / (nu + u_ray))

    # Exact factorized per-ray Student-t marginal Bayes factor (unweighted sum across rays:
    # stronger rays already carry higher SNR in v_clr/v_inh, while minor rays still vote
    # honestly against full_reset; Student-t heavy tails keep 1 shifted minor ray from
    # vetoing 2 intact rays in Regime 2 partial_shift).
    ray_llr_sum = float(np.sum(ray_llr))
    ray_consistency = float(np.sum(POWERS * ray_inlier))  # in (0, 1], null E ~ nu / (nu + 1)

    # 2. Per-sample pre-update Student-t LLR (normalized to effective ray dimension)
    samp_ll_inh = _complex_student_t_loglik(y_tr, cand_inh.pre_train_pred, cand_inh.pre_train_var, nu=nu)
    samp_ll_clr = _complex_student_t_loglik(y_tr, cand_clr.pre_train_pred, cand_clr.pre_train_var, nu=nu)
    samp_llr_eff = float(len(DELAYS) * np.mean(samp_ll_inh - samp_ll_clr) * n_rows)

    # 3. Robust Student-t-saturated full-covariance pre-update Bayes factor:
    # captures intra-block Doppler slope & full multi-ray coherence while preventing
    # unbounded quadratic Gaussian veto on a single minor-ray shift.
    bf_raw = float(cand_inh.pre_train_loglik) - float(cand_clr.pre_train_loglik)
    bf_scale = nu * float(n_rows)
    bf_robust = float(np.sign(bf_raw) * bf_scale * np.log1p(abs(bf_raw) / bf_scale))

    # Base link-level logit combining exact per-ray Student-t Bayes factor + tempered full-cov BF
    base_logit = ray_llr_sum + 0.35 * bf_robust + 0.15 * samp_llr_eff

    # 4. C*-invariant Jeffreys divergence row-adaptive modulation (p8_manifold Construction B)
    dj_pre = float(np.mean(pre_update_jeffreys(cand_inh, cand_clr)))
    dj_post_rows = post_update_jeffreys_rows(cand_inh, cand_clr, n0=n0)  # [T]
    dj_train_ref = float(np.mean(dj_post_rows[list(rows)]))
    dj_excess = np.maximum(dj_post_rows - dj_train_ref, 0.0) / (1.0 + dj_pre + dj_train_ref)

    # Activate extrapolation-risk tempering strictly when ray consistency drops below the
    # theoretical Student-t null expectation nu / (nu + 1) (at E[u_l] = 1):
    # - On continuous links: ray_consistency >= null_ref -> mismatch_signal = 0 (zero cost!)
    # - On partial_shift / ood_midframe_shift: rows farthest from training (highest dj_excess)
    #   smoothly temper toward clear where wrong-Doppler drift is largest.
    null_ref = nu / (nu + 1.0)
    mismatch_signal = float(np.clip(1.0 - ray_consistency / null_ref, 0.0, 1.0))
    row_logits = base_logit - (2.0 * mismatch_signal) * dj_excess
    w_rows = np.asarray(_sigmoid(row_logits), dtype=np.float64)
    w_probe = float(w_rows[obs.probe_row])

    return combine_candidates(w_rows, w_probe, cand_inh, cand_clr)


# --- Locked Branch-Merged Whole-Word Codeword Scoring Helper ---

def score_codeword_mixture(
    gate_out: GateOutput,
    y_b_row: np.ndarray,
    row_idx: int,
    n0: float,
    codewords_symbols: np.ndarray,
) -> np.ndarray | float:
    """Locked branch-merged whole-word log-likelihood for candidate 16-QAM codewords.

    Evaluates exact Gaussian predictive log-likelihood log p(y_b_row | x(c), gate_out)
    using the merged effective ray means and total (within + between) ray variances
    at Frame B row `row_idx`:

        y_hat[n]  = sum_l mix_mean_ray[l, row_idx, n] * x[(n - delay_l) % N]
        var_y[n]  = n0 + sum_l mix_var_ray[l, row_idx, n] * |x[(n - delay_l) % N]|^2
        score(x)  = -sum_n [ log(pi * var_y[n]) + |y_b_row[n] - y_hat[n]|^2 / var_y[n] ]

    When mix_var_ray == 0 (known-channel oracle), reduces identically to -N*log(pi*n0)
    - ||y - H x||^2 / n0. Score differences and argmax are strictly C*-invariant.

    Args:
        gate_out: GateOutput containing mix_mean_ray [3, T, N] and mix_var_ray [3, T, N].
        y_b_row: Received time-domain CP-SC block [N] complex128.
        row_idx: Frame B row index in [0, T - 1].
        n0: Explicit positive complex noise variance.
        codewords_symbols: Candidate 16-QAM symbol block(s), shape [N] or [K, N].

    Returns:
        Scalar float if codewords_symbols is 1D [N], else float64 array of shape [K].
    """
    r = int(row_idx)
    if r < 0 or r >= T:
        raise ValueError(f"row_idx must be in [0, {T - 1}], got {row_idx}")
    n0 = float(n0)
    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError(f"n0 must be finite and positive, got {n0}")

    y_row = np.asarray(y_b_row, dtype=np.complex128).ravel()
    if y_row.shape != (N,) or not np.isfinite(y_row).all():
        raise ValueError(f"y_b_row must be finite complex array of shape ({N},)")

    cw = np.asarray(codewords_symbols, dtype=np.complex128)
    single = cw.ndim == 1
    if single:
        cw = cw[None, :]
    if cw.ndim != 2 or cw.shape[1] != N:
        raise ValueError(f"codewords_symbols must have shape ({N},) or (K, {N}), got {cw.shape}")

    mean_t = np.asarray(gate_out.mix_mean_ray[:, r, :], dtype=np.complex128)  # [3, N]
    var_t = np.maximum(np.asarray(gate_out.mix_var_ray[:, r, :], dtype=np.float64), 0.0)  # [3, N]

    # DELAYED_INDEX has shape [3, N] -> delayed_x has shape [K, 3, N]
    delayed_x = cw[:, DELAYED_INDEX]
    y_hat = np.einsum("ln,kln->kn", mean_t, delayed_x)  # [K, N]
    var_y = n0 + np.einsum("ln,kln->kn", var_t, np.abs(delayed_x) ** 2)  # [K, N]
    var_y = np.maximum(var_y, 1e-300)
    resid2 = np.abs(y_row[None, :] - y_hat) ** 2
    scores = -np.sum(np.log(np.pi * var_y) + resid2 / var_y, axis=-1)  # [K]
    return float(scores[0]) if single else np.asarray(scores, dtype=np.float64)
