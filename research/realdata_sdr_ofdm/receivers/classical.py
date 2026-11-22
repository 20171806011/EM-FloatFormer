"""Classical OFDM receiver baselines for realdata_sdr_ofdm (F6).

Implements:
- ``ReceiverOutput`` and ``ClassicalConfig`` dataclasses
- ``ls_linear_zf`` (alias ``ls_zf``): companion-aligned LS pilot estimation +
  piecewise-linear Re/Im frequency interpolation + exact complex division ZF +
  calibrated non-Gray 16-QAM LLRs
- ``lmmse_1d``: scale-equivariant timing-detrended 1D frequency LMMSE +
  regularized MMSE equalization + calibrated exact LLRs
- ``lmmse_2d_tracked`` (alias ``lmmse``, default): 2D Jakes temporal + frequency
  LMMSE anchored at Sp=2 + blind Bayesian/DD phase-drift tracking outward from s=2
- ``calibrate_on_dev`` (alias ``calibrate_classical``): strict dev-only grid
  calibration raising ``ValueError`` on any ``test`` or ``guard`` frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.special import j0 as bessel_j0

from research.realdata_sdr_ofdm.adapters.modulation import (
    bits_to_symbols,
    constellation_points,
    exact_llr,
)
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    ACTIVE_END,
    ACTIVE_START,
    CP_LEN,
    N_ACTIVE,
    N_BITS,
    N_DATA_RES,
    N_FFT,
    N_SYMB,
    PILOT_ACTIVE_INDICES,
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    QM,
    RealOFDMFrame,
    build_heldout_pilot_masks,
)

CLASSICAL_METHODS: tuple[str, ...] = (
    "ls_linear_zf",
    "lmmse_1d",
    "lmmse_2d_tracked",
)

_METHOD_ALIASES: dict[str, str] = {
    "ls_linear_zf": "ls_linear_zf",
    "ls_zf": "ls_linear_zf",
    "lmmse_1d": "lmmse_1d",
    "lmmse_2d_tracked": "lmmse_2d_tracked",
    "lmmse": "lmmse_2d_tracked",
}


@dataclass
class ReceiverOutput:
    """Canonical output contract shared across all M2/M3/M4 receivers."""

    h_est: np.ndarray
    h_var: np.ndarray
    eq_symbols: np.ndarray
    bit_llrs: np.ndarray
    hard_bits: np.ndarray
    hard_symbols: np.ndarray
    heldout_pilot_pred: Optional[np.ndarray] = None
    heldout_pilot_var: Optional[np.ndarray] = None
    aux_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassicalConfig:
    """Hyperparameters for classical receivers (tunable strictly on dev)."""

    tau_rms: float = 1.5
    n_taps: int = 5
    n0_scale: float = 1.0
    diag_reg: float = 1e-4
    fd_norm: float = 0.0
    detrend_timing: bool = False
    phase_track: bool = True
    phase_track_max_slope: float = 0.50
    phase_track_prior_weight: float = 2.0
    phase_track_min_gain: float = 8.0
    time_var_growth: float = 0.0
    cpe_refine_alpha: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_method_name(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown classical receiver method {method!r}; expected one of "
            f"{sorted(_METHOD_ALIASES.keys())}"
        )
    return _METHOD_ALIASES[key]


def _exp_pdp_freq_cov(
    k1: np.ndarray,
    k2: np.ndarray,
    *,
    tau_rms: float,
    n_taps: int,
    centered: bool = False,
) -> np.ndarray:
    """Unit-diagonal Hermitian PSD frequency correlation matrix for exponential PDP."""
    taps = np.arange(max(int(n_taps), 1), dtype=np.float64)
    trms = max(float(tau_rms), 1e-3)
    p = np.exp(-taps / trms)
    p /= np.sum(p)
    eff_taps = (taps - float(np.sum(taps * p))) if centered else taps
    dk = np.asarray(k1, dtype=np.float64)[:, None] - np.asarray(k2, dtype=np.float64)[None, :]
    phase = -2.0 * np.pi * dk[..., None] * eff_taps[None, None, :] / float(N_FFT)
    return np.sum(p[None, None, :] * np.exp(1j * phase), axis=-1).astype(np.complex128)


def _estimate_timing_slope(
    k_obs: np.ndarray,
    h_p: np.ndarray,
) -> float:
    """Blind $C^*$-invariant frequency phase-slope estimate from adjacent observed pilots."""
    if len(k_obs) < 2:
        return 0.0
    k_arr = np.asarray(k_obs, dtype=np.float64)
    h_arr = np.asarray(h_p, dtype=np.complex128)
    if np.any(np.diff(k_arr) < 0.0):
        order = np.argsort(k_arr, kind="mergesort")
        k_arr = k_arr[order]
        h_arr = h_arr[order]
    dk = np.diff(k_arr).astype(np.float64)
    diffs = h_arr[1:] * np.conj(h_arr[:-1])
    min_dk = float(np.min(dk))
    mask = dk <= max(min_dk, 8.0) + 1e-9
    if not np.any(mask):
        mask = np.ones_like(dk, dtype=bool)
    weighted = np.sum(diffs[mask])
    mag = float(np.abs(weighted))
    if mag <= 1e-30 or not np.isfinite(mag):
        return 0.0
    mean_dk = float(np.mean(dk[mask]))
    if mean_dk <= 0.0:
        return 0.0
    return float(np.angle(weighted) / mean_dk)


def _estimate_sym2_channel(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    obs_pilot_indices: Sequence[int],
    n0_norm: float,
    *,
    method: str,
    config: ClassicalConfig,
    target_active_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate channel and posterior variance at symbol Sp=2 in normalized gauge.

    Returns (h_est_norm, h_var_norm, sigma_h2_norm).
    """
    obs_idx = np.unique(np.asarray(obs_pilot_indices, dtype=np.int64))
    if obs_idx.size == 0:
        raise ValueError("obs_pilot_indices must be non-empty")
    k_obs = np.asarray([PILOT_ACTIVE_INDICES[i] for i in obs_idx], dtype=np.float64)
    fft_obs = [PILOT_FFT_INDICES[i] for i in obs_idx]
    y_p = rx_norm[PILOT_SYMBOL_IDX, fft_obs]
    p_val = np.asarray(pilot_values[obs_idx], dtype=np.complex128)
    p_pow = np.abs(p_val) ** 2
    h_p = y_p / p_val

    k_target = (
        np.arange(N_ACTIVE, dtype=np.float64)
        if target_active_indices is None
        else np.asarray(target_active_indices, dtype=np.float64)
    )

    n0_eff = max(float(config.n0_scale) * float(n0_norm), 0.0)
    sigma2_obs = n0_eff / np.maximum(p_pow, 1e-30)
    mean_hp2 = float(np.mean(np.abs(h_p) ** 2))
    mean_sig2 = float(np.mean(sigma2_obs))
    # Strictly positive floor in normalized space (active RMS == 1.0 unless all-zero)
    sigma_h2 = max(mean_hp2 - mean_sig2, 0.05 * mean_hp2, 1e-6 * max(1.0, mean_sig2))

    if method == "ls_linear_zf":
        h_re = np.interp(k_target, k_obs, h_p.real)
        h_im = np.interp(k_target, k_obs, h_p.imag)
        h_1d = (h_re + 1j * h_im).astype(np.complex128)
        # Analytic linear-interpolation observation variance across active bins
        idx_hi = np.searchsorted(k_obs, k_target, side="right")
        idx_lo = np.clip(idx_hi - 1, 0, len(k_obs) - 1)
        idx_hi = np.clip(idx_hi, 0, len(k_obs) - 1)
        span = np.maximum(k_obs[idx_hi] - k_obs[idx_lo], 1.0)
        alpha = np.where(idx_hi == idx_lo, 0.0, (k_target - k_obs[idx_lo]) / span)
        var_1d = ((1.0 - alpha) ** 2) * sigma2_obs[idx_lo] + (alpha ** 2) * sigma2_obs[idx_hi]
        var_floor = max(1e-6 * sigma_h2, 1e-6 * mean_sig2, 1e-12)
        var_1d = np.maximum(var_1d, var_floor).astype(np.float64)
        return h_1d, var_1d, sigma_h2

    # LMMSE 1D (also foundation for LMMSE 2D tracked)
    if mean_hp2 <= 1e-28:
        # Zero-pilot resilience: return zero channel and strictly positive prior variance
        h_1d = np.zeros(len(k_target), dtype=np.complex128)
        var_1d = np.full(len(k_target), max(sigma_h2, mean_sig2, 1e-6), dtype=np.float64)
        return h_1d, var_1d, sigma_h2

    omega_f = _estimate_timing_slope(k_obs, h_p) if config.detrend_timing else 0.0
    h_p_detrend = h_p * np.exp(-1j * omega_f * k_obs) if config.detrend_timing else h_p

    r_pp_unit = _exp_pdp_freq_cov(
        k_obs,
        k_obs,
        tau_rms=config.tau_rms,
        n_taps=config.n_taps,
        centered=bool(config.detrend_timing),
    )
    r_tp_unit = _exp_pdp_freq_cov(
        k_target,
        k_obs,
        tau_rms=config.tau_rms,
        n_taps=config.n_taps,
        centered=bool(config.detrend_timing),
    )

    r_pp = sigma_h2 * r_pp_unit
    r_tp = sigma_h2 * r_tp_unit
    reg = float(config.diag_reg) * sigma_h2
    a_mat = r_pp + np.diag(sigma2_obs + reg)
    # Hermitian ridge guard for numerical stability
    a_mat = 0.5 * (a_mat + a_mat.conj().T)

    alpha_vec = np.linalg.solve(a_mat, h_p_detrend)
    h_detrend = r_tp @ alpha_vec
    h_1d = (
        (h_detrend * np.exp(1j * omega_f * k_target)) if config.detrend_timing else h_detrend
    ).astype(np.complex128)

    # Posterior covariance diagonal: sigma_h2 - Re(diag(R_tp A^{-1} R_pt))
    w_t = np.linalg.solve(a_mat, r_tp.conj().T)  # (n_obs, n_target)
    post_red = np.real(np.sum(r_tp * w_t.T, axis=1))
    var_floor = max(1e-4 * mean_sig2, 1e-6 * sigma_h2, 1e-12)
    var_1d = np.maximum(sigma_h2 - post_red, var_floor).astype(np.float64)
    return h_1d, var_1d, sigma_h2


def _equalize_and_demap(
    y_data: np.ndarray,
    h_data: np.ndarray,
    v_data: np.ndarray,
    n0_norm: float,
    *,
    method: str,
    config: ClassicalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equalize payload REs and compute calibrated non-Gray 16-QAM LLRs."""
    n0_cal = max(float(config.n0_scale) * float(n0_norm), 0.0)
    h_abs2 = np.abs(h_data) ** 2

    if method == "ls_linear_zf":
        # Exact complex division ZF with zero-channel guard; safe if h == 0
        safe_mask = h_abs2 > 1e-24
        eq_symbols = np.where(
            safe_mask,
            y_data / np.where(safe_mask, h_data, 1.0 + 0.0j),
            0.0 + 0.0j,
        ).astype(np.complex128)
        n_eff = n0_cal + np.maximum(v_data, 0.0)
        sigma2_eff = np.where(
            safe_mask,
            np.maximum(n_eff / np.maximum(h_abs2, 1e-24), 1e-12),
            1e12,
        ).astype(np.float64)
        z_for_llr = eq_symbols
    else:
        n_eff = n0_cal + np.maximum(v_data, 0.0)
        denom = h_abs2 + n_eff
        safe_denom = np.maximum(denom, 1e-30)
        x_mmse = np.where(
            denom > 0.0,
            (np.conj(h_data) * y_data) / safe_denom,
            0.0 + 0.0j,
        ).astype(np.complex128)
        mu = np.clip(h_abs2 / safe_denom, 1e-8, 1.0)
        z_for_llr = np.where(h_abs2 > 1e-30, x_mmse / mu, 0.0 + 0.0j).astype(np.complex128)
        sigma2_eff = np.where(
            h_abs2 > 1e-30,
            np.maximum((1.0 - mu) / mu, 1e-12),
            1e12,
        ).astype(np.float64)
        eq_symbols = x_mmse

    bit_llrs = exact_llr(z_for_llr, sigma2_eff, qm=QM)
    hard_bits = (bit_llrs > 0.0).astype(np.uint8)
    hard_symbols = bits_to_symbols(hard_bits, qm=QM)
    return eq_symbols, bit_llrs, hard_bits, hard_symbols


def _track_phase_2d(
    rx_active: np.ndarray,
    data_mask_active: np.ndarray,
    h_2d: np.ndarray,
    v_2d: np.ndarray,
    n0_norm: float,
    *,
    config: ClassicalConfig,
) -> tuple[np.ndarray, float]:
    """Blind $C^*$-invariant linear phase-drift + optional CPE tracking anchored at Sp=2."""
    if not config.phase_track or float(np.max(np.abs(h_2d))) <= 1e-14:
        return h_2d, 0.0

    # Extract payload REs and their symbol indices (0..13)
    sym_grid = np.broadcast_to(np.arange(N_SYMB, dtype=np.float64)[:, None], (N_SYMB, N_ACTIVE))
    s_data = sym_grid[data_mask_active]
    y_data = rx_active[data_mask_active]
    h_data = h_2d[data_mask_active]
    v_data = v_2d[data_mask_active]

    n0_cal = max(float(config.n0_scale) * float(n0_norm), 0.0)
    n_eff = n0_cal + np.maximum(v_data, 0.0)
    h_abs2 = np.abs(h_data) ** 2
    denom = np.maximum(h_abs2 + n_eff, 1e-30)
    mu = np.clip(h_abs2 / denom, 1e-8, 1.0)
    z_base = np.where(h_abs2 > 1e-30, ((np.conj(h_data) * y_data) / denom) / mu, 0.0 + 0.0j)
    sigma2_eff = np.where(h_abs2 > 1e-30, np.maximum((1.0 - mu) / mu, 1e-12), 1e12)

    # Focus scoring on non-symbol-2 REs with highest effective SNR (top 50%, min 200)
    off_mask = s_data != float(PILOT_SYMBOL_IDX)
    z_off = z_base[off_mask]
    s_off = s_data[off_mask] - float(PILOT_SYMBOL_IDX)
    sig2_off = sigma2_eff[off_mask]

    # Select top 50% lowest-noise REs for crisp, fast phase likelihood evaluation
    k_keep = max(min(650, len(z_off)), len(z_off) // 2)
    top_idx = np.argpartition(sig2_off, k_keep - 1)[:k_keep]
    z_sub = z_off[top_idx]
    s_sub = s_off[top_idx]
    sig2_sub = np.clip(sig2_off[top_idx], 1e-12, 1e12)

    pts = constellation_points(QM)  # (16,)
    # Marginal log-likelihood logsumexp(-|z*exp(-j*omega*s) - pts|^2 / sig2)
    inv_sig2 = (1.0 / sig2_sub)[:, None]
    max_slope = float(config.phase_track_max_slope)
    prior_w = float(config.phase_track_prior_weight)

    def _score_slopes(omegas: np.ndarray) -> np.ndarray:
        scores = np.empty(len(omegas), dtype=np.float64)
        for idx, om in enumerate(omegas):
            z_rot = z_sub * np.exp(-1j * float(om) * s_sub)
            sq = np.abs(z_rot[:, None] - pts[None, :]) ** 2
            ll = -sq * inv_sig2
            m = np.max(ll, axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.sum(np.exp(np.clip(ll - m, -700.0, 0.0)), axis=1))
            scores[idx] = float(np.sum(lse)) - 0.5 * prior_w * (float(om) / max(max_slope, 1e-6)) ** 2
        return scores

    coarse_grid = np.linspace(-max_slope, max_slope, 41, dtype=np.float64)
    coarse_scores = _score_slopes(coarse_grid)
    best_om = float(coarse_grid[int(np.argmax(coarse_scores))])

    # Fine local refinement around best coarse slope
    step = float(coarse_grid[1] - coarse_grid[0])
    fine_grid = np.linspace(
        max(-max_slope, best_om - step),
        min(max_slope, best_om + step),
        17,
        dtype=np.float64,
    )
    fine_scores = _score_slopes(fine_grid)
    best_idx = int(np.argmax(fine_scores))
    best_om = float(fine_grid[best_idx])
    best_score = float(fine_scores[best_idx])
    zero_score = float(_score_slopes(np.array([0.0], dtype=np.float64))[0])
    if best_score - zero_score < float(config.phase_track_min_gain):
        best_om = 0.0

    sym_offsets = np.arange(N_SYMB, dtype=np.float64) - float(PILOT_SYMBOL_IDX)
    theta_sym = best_om * sym_offsets
    h_rot = h_2d * np.exp(1j * theta_sym[:, None])

    # Optional per-symbol DD CPE residual refinement outward from Sp=2
    cpe_alpha = float(config.cpe_refine_alpha)
    if cpe_alpha > 0.0:
        h_refined = h_rot.copy()
        for s_idx in list(range(PILOT_SYMBOL_IDX + 1, N_SYMB)) + list(range(PILOT_SYMBOL_IDX - 1, -1, -1)):
            dmask_s = data_mask_active[s_idx]
            if not np.any(dmask_s):
                continue
            y_s = rx_active[s_idx, dmask_s]
            h_s = h_refined[s_idx, dmask_s]
            v_s = v_2d[s_idx, dmask_s]
            h_s2 = np.abs(h_s) ** 2
            den_s = np.maximum(h_s2 + n0_cal + v_s, 1e-30)
            mu_s = np.clip(h_s2 / den_s, 1e-8, 1.0)
            z_s = np.where(h_s2 > 1e-30, ((np.conj(h_s) * y_s) / den_s) / mu_s, 0.0 + 0.0j)
            pts_idx = np.argmin(np.abs(z_s[:, None] - pts[None, :]) ** 2, axis=1)
            x_hat = pts[pts_idx]
            w_s = h_s2 / den_s
            acc = np.sum(w_s * y_s * np.conj(h_s * x_hat))
            if np.abs(acc) > 1e-30:
                dphi = float(np.angle(acc))
                h_refined[s_idx] *= np.exp(1j * cpe_alpha * dphi)
        h_rot = h_refined

    return h_rot.astype(np.complex128), best_om


def _run_2d_core(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    data_mask_active: np.ndarray,
    obs_pilot_indices: Sequence[int],
    n0_norm: float,
    *,
    method: str,
    config: ClassicalConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (14, 102) normalized channel estimate & posterior variance."""
    h_1d, var_1d, sigma_h2 = _estimate_sym2_channel(
        rx_norm,
        pilot_values,
        obs_pilot_indices,
        n0_norm,
        method=method,
        config=config,
    )
    if method in ("ls_linear_zf", "lmmse_1d"):
        h_2d = np.broadcast_to(h_1d[None, :], (N_SYMB, N_ACTIVE)).copy()
        v_2d = np.broadcast_to(var_1d[None, :], (N_SYMB, N_ACTIVE)).copy()
        return h_2d, v_2d, 0.0

    # lmmse_2d_tracked: Jakes temporal prior anchored at Sp=2
    dt = np.abs(np.arange(N_SYMB, dtype=np.float64) - float(PILOT_SYMBOL_IDX))
    arg = 2.0 * np.pi * max(float(config.fd_norm), 0.0) * dt * (float(N_FFT + CP_LEN) / float(N_FFT))
    r_t = bessel_j0(arg).astype(np.float64)
    h_2d = r_t[:, None] * h_1d[None, :]
    extra_t = dt * max(float(config.time_var_growth), 0.0) * sigma_h2
    v_raw = sigma_h2 * (1.0 - r_t[:, None] ** 2) + (r_t[:, None] ** 2) * var_1d[None, :] + extra_t[:, None]
    var_floor = np.maximum(var_1d[None, :], 1e-12)
    v_2d = np.maximum(v_raw, var_floor).astype(np.float64)

    rx_active = rx_norm[:, ACTIVE_START:ACTIVE_END]
    h_tracked, omega_t = _track_phase_2d(
        rx_active,
        data_mask_active,
        h_2d,
        v_2d,
        n0_norm,
        config=config,
    )
    return h_tracked, v_2d, omega_t


def _validate_pilot_indices(
    indices: Sequence[Any],
    label: str = "heldout_pilot_indices",
) -> list[int]:
    """Validate pilot index sequence: unique integers strictly in [0, N_PILOTS - 1] (order-preserving)."""
    out: list[int] = []
    seen: set[int] = set()
    n_pilots = len(PILOT_FFT_INDICES)
    for raw in indices:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
            raise ValueError(
                f"{label} entries must be integers in [0, {n_pilots - 1}], got {raw!r}"
            )
        pi = int(raw)
        if not (0 <= pi < n_pilots):
            raise ValueError(
                f"{label} index out of range [0, {n_pilots - 1}]: {raw!r}"
            )
        if pi not in seen:
            seen.add(pi)
            out.append(pi)
    return out


def _compute_cv_fold_scale(
    rx_grid: np.ndarray,
    pilot_mask: np.ndarray,
    data_mask: np.ndarray,
    est_n0: float,
    heldout_indices: Sequence[int],
) -> float:
    """Compute per-fold RMS normalization scale strictly excluding held-out pilots."""
    p_mask = np.asarray(pilot_mask, dtype=bool)
    d_mask = np.asarray(data_mask, dtype=bool)
    heldout_mask = np.zeros_like(p_mask, dtype=bool)
    for pi_int in _validate_pilot_indices(heldout_indices, "heldout_pilot_indices"):
        heldout_mask[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[pi_int]] = True
    cv_active_mask = (p_mask & ~heldout_mask) | (d_mask & ~heldout_mask)
    if np.any(cv_active_mask):
        s_rms_cv = float(np.sqrt(np.mean(np.abs(rx_grid[cv_active_mask]) ** 2)))
        if not np.isfinite(s_rms_cv):
            raise ValueError("Non-finite active RMS power in held-out CV")
    else:
        s_rms_cv = 0.0
    return s_rms_cv if s_rms_cv > 0.0 else max(math.sqrt(est_n0), 1.0)


def _compute_heldout_cv(
    frame: RealOFDMFrame,
    rx_grid: np.ndarray,
    est_n0: float,
    *,
    method: str,
    config: ClassicalConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute held-out pilot predictions & strictly positive variances in original gauge.

    Uses per-fold observable normalization ``s_scale_cv`` excluding held-out pilot REs
    so held-out predictions and variances have zero leakage from mutated held-out pilots.
    """
    explicit_idx = frame.meta.get("heldout_pilot_indices") if isinstance(frame.meta, dict) else None
    if explicit_idx is not None and len(explicit_idx) > 0:
        eval_idx = _validate_pilot_indices(explicit_idx, "heldout_pilot_indices")
        eval_set = set(eval_idx)
        obs_idx = [i for i in range(14) if i not in eval_set]
        s_scale_cv = _compute_cv_fold_scale(
            rx_grid, frame.pilot_mask, frame.data_mask, est_n0, eval_idx
        )
        rx_norm_cv = rx_grid / s_scale_cv
        n0_norm_cv = est_n0 / (s_scale_cv ** 2)
        v_floor_cv = max(1e-30 * (float(s_scale_cv) ** 2), 1e-300)
        target_active = np.asarray([PILOT_ACTIVE_INDICES[i] for i in eval_idx], dtype=np.int64)
        h_pred_norm, h_var_norm, _ = _estimate_sym2_channel(
            rx_norm_cv,
            frame.pilot_values,
            obs_idx,
            n0_norm_cv,
            method=method,
            config=config,
            target_active_indices=target_active,
        )
        pred = (h_pred_norm * s_scale_cv).astype(np.complex128)
        var = np.maximum(h_var_norm * (s_scale_cv ** 2), v_floor_cv).astype(np.float64)
        return pred, var, eval_idx

    folds = None
    if isinstance(frame.meta, dict):
        folds = frame.meta.get("heldout_pilot_folds")
    if not folds:
        folds = build_heldout_pilot_masks(4)

    collected: dict[int, tuple[complex, float]] = {}
    for fold in folds:
        h_idx = _validate_pilot_indices(fold["heldout_pilot_indices"], "heldout_pilot_indices")
        if not h_idx:
            raise ValueError("heldout_pilot_indices in fold must be non-empty")
        h_set = set(h_idx)
        o_raw = _validate_pilot_indices(fold["obs_pilot_indices"], "obs_pilot_indices")
        o_idx = sorted(i for i in o_raw if i not in h_set)
        s_scale_cv = _compute_cv_fold_scale(
            rx_grid, frame.pilot_mask, frame.data_mask, est_n0, h_idx
        )
        rx_norm_cv = rx_grid / s_scale_cv
        n0_norm_cv = est_n0 / (s_scale_cv ** 2)
        v_floor_cv = max(1e-30 * (float(s_scale_cv) ** 2), 1e-300)
        target_active = np.asarray([PILOT_ACTIVE_INDICES[i] for i in h_idx], dtype=np.int64)
        h_pred_norm, h_var_norm, _ = _estimate_sym2_channel(
            rx_norm_cv,
            frame.pilot_values,
            o_idx,
            n0_norm_cv,
            method=method,
            config=config,
            target_active_indices=target_active,
        )
        for pos, pi in enumerate(h_idx):
            if pi not in collected:
                collected[pi] = (
                    complex(h_pred_norm[pos] * s_scale_cv),
                    max(float(h_var_norm[pos] * (s_scale_cv ** 2)), v_floor_cv),
                )

    if not collected:
        raise ValueError("heldout_pilot_folds produced no held-out pilots")

    eval_idx = sorted(collected.keys())
    pred = np.asarray([collected[i][0] for i in eval_idx], dtype=np.complex128)
    var = np.asarray([collected[i][1] for i in eval_idx], dtype=np.float64)
    return pred, var, eval_idx


def _failed_output(err_msg: str, method: str, latency_ms: float) -> ReceiverOutput:
    bit_llrs = np.zeros((N_DATA_RES, QM), dtype=np.float64)
    hard_bits = (bit_llrs > 0.0).astype(np.uint8)
    hard_symbols = bits_to_symbols(hard_bits, qm=QM)
    return ReceiverOutput(
        h_est=np.zeros((N_SYMB, N_ACTIVE), dtype=np.complex128),
        h_var=np.ones((N_SYMB, N_ACTIVE), dtype=np.float64),
        eq_symbols=np.zeros(N_DATA_RES, dtype=np.complex128),
        bit_llrs=bit_llrs,
        hard_bits=hard_bits,
        hard_symbols=hard_symbols,
        heldout_pilot_pred=None,
        heldout_pilot_var=None,
        aux_info={
            "n_candidates": 1,
            "branch_weights": [1.0],
            "latency_ms": float(latency_ms),
            "failed": True,
            "error": str(err_msg),
            "param_count": 0,
            "method": method,
            "heldout_pilot_indices": [],
        },
    )


def run_frame(
    frame: RealOFDMFrame,
    context_pool: Any = None,
    *,
    method: str = "lmmse",
    config: Optional[ClassicalConfig] = None,
    return_heldout_cv: bool = True,
) -> ReceiverOutput:
    """Run a classical OFDM receiver on one RealOFDMFrame without touching labels."""
    del context_pool  # Classical single-TTI baseline ignores context_pool
    t0 = time.perf_counter()
    canon_method = normalize_method_name(method)
    cfg = ClassicalConfig() if config is None else config

    try:
        rx_grid = np.asarray(frame.rx_grid, dtype=np.complex128)
        if rx_grid.shape != (N_SYMB, N_FFT):
            raise ValueError(f"Expected rx_grid shape {(N_SYMB, N_FFT)}, got {rx_grid.shape}")
        if not np.all(np.isfinite(rx_grid)):
            raise ValueError("rx_grid contains non-finite (NaN/Inf) values")

        est_n0 = float(frame.est_n0)
        if not np.isfinite(est_n0) or est_n0 < 0.0:
            raise ValueError(f"Invalid est_n0: {frame.est_n0!r}")

        pilot_values = np.asarray(frame.pilot_values, dtype=np.complex128)
        if pilot_values.shape != (len(PILOT_FFT_INDICES),):
            raise ValueError(
                f"Expected pilot_values shape {(len(PILOT_FFT_INDICES),)}, got {pilot_values.shape}"
            )
        if not np.all(np.isfinite(pilot_values)):
            raise ValueError("pilot_values contains non-finite (NaN/Inf) values")
        if not np.all(np.abs(pilot_values) > 0.0):
            raise ValueError("pilot_values contains zero-magnitude entries")

        active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
        s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
        if not np.isfinite(s_rms):
            raise ValueError("Non-finite active RMS power")
        if s_rms > 0.0:
            s_scale = s_rms
        else:
            s_scale = max(math.sqrt(est_n0), 1.0)

        rx_norm = rx_grid / s_scale
        n0_norm = est_n0 / (s_scale ** 2)

        data_mask_active = np.asarray(frame.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
        all_pilots = list(range(14))
        h_2d_norm, v_2d_norm, omega_t = _run_2d_core(
            rx_norm,
            pilot_values,
            data_mask_active,
            all_pilots,
            n0_norm,
            method=canon_method,
            config=cfg,
        )

        rx_active = rx_norm[:, ACTIVE_START:ACTIVE_END]
        y_data = rx_active[data_mask_active]
        h_data = h_2d_norm[data_mask_active]
        v_data = v_2d_norm[data_mask_active]

        eq_symbols, bit_llrs, hard_bits, hard_symbols = _equalize_and_demap(
            y_data,
            h_data,
            v_data,
            n0_norm,
            method=canon_method,
            config=cfg,
        )

        h_est = (h_2d_norm * s_scale).astype(np.complex128)
        h_var = np.maximum(
            v_2d_norm * (s_scale ** 2),
            max(1e-30 * (s_scale ** 2), 1e-300),
        ).astype(np.float64)
        if not (np.all(np.isfinite(h_est)) and np.all(np.isfinite(h_var))):
            raise ValueError("Channel estimate contains non-finite (NaN/Inf) values")
        if not (np.all(np.isfinite(eq_symbols)) and np.all(np.isfinite(bit_llrs))):
            raise ValueError("Equalized symbols or bit_llrs contain non-finite (NaN/Inf) values")

        heldout_pred: Optional[np.ndarray] = None
        heldout_var: Optional[np.ndarray] = None
        heldout_indices: list[int] = []
        if return_heldout_cv:
            heldout_pred, heldout_var, heldout_indices = _compute_heldout_cv(
                frame,
                rx_grid,
                est_n0,
                method=canon_method,
                config=cfg,
            )
            if not (np.all(np.isfinite(heldout_pred)) and np.all(np.isfinite(heldout_var))):
                raise ValueError("Held-out pilot predictions contain non-finite (NaN/Inf) values")

        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        aux_info: dict[str, Any] = {
            "n_candidates": 1,
            "branch_weights": [1.0],
            "latency_ms": float(latency_ms),
            "failed": False,
            "error": None,
            "param_count": 0,
            "method": canon_method,
            "omega_t_rad_per_sym": float(omega_t),
            "heldout_pilot_indices": heldout_indices,
        }
        return ReceiverOutput(
            h_est=h_est,
            h_var=h_var,
            eq_symbols=eq_symbols,
            bit_llrs=bit_llrs,
            hard_bits=hard_bits,
            hard_symbols=hard_symbols,
            heldout_pilot_pred=heldout_pred,
            heldout_pilot_var=heldout_var,
            aux_info=aux_info,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        return _failed_output(f"{type(exc).__name__}: {exc}", canon_method, latency_ms)


class ClassicalReceiver:
    """Stateful wrapper holding method & dev-calibrated ClassicalConfig."""

    def __init__(
        self,
        method: str = "lmmse_2d_tracked",
        config: Optional[ClassicalConfig] = None,
    ) -> None:
        self.method = normalize_method_name(method)
        self.config = ClassicalConfig() if config is None else config

    def run_frame(
        self,
        frame: RealOFDMFrame,
        context_pool: Any = None,
        *,
        return_heldout_cv: bool = True,
    ) -> ReceiverOutput:
        return run_frame(
            frame,
            context_pool=context_pool,
            method=self.method,
            config=self.config,
            return_heldout_cv=return_heldout_cv,
        )


def _dev_objective(
    frames: Sequence[RealOFDMFrame],
    method: str,
    config: ClassicalConfig,
) -> tuple[float, float, float]:
    """Compute (ber, pilot_nll, sym_mse) strictly on dev_frames."""
    bit_err = 0
    tot_bits = 0
    sq_sym = 0.0
    tot_sym = 0
    nlls: list[float] = []
    for fr in frames:
        out = run_frame(fr, method=method, config=config, return_heldout_cv=True)
        if out.aux_info.get("failed", False):
            return 1.0, 1e9, 1e9
        bit_err += int(np.sum(out.hard_bits != fr.tx_bits))
        tot_bits += int(fr.tx_bits.size)
        sq_sym += float(np.sum(np.abs(out.eq_symbols - fr.tx_symbols) ** 2))
        tot_sym += int(fr.tx_symbols.size)
        if out.heldout_pilot_pred is not None and out.heldout_pilot_var is not None:
            idx = out.aux_info.get("heldout_pilot_indices") or []
            if len(idx) == len(out.heldout_pilot_pred) and len(idx) > 0:
                fft_idx = [PILOT_FFT_INDICES[i] for i in idx]
                y_p = fr.rx_grid[PILOT_SYMBOL_IDX, fft_idx]
                p_val = fr.pilot_values[idx]
                z_obs = y_p / p_val
                sig2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)
                act_mask = np.asarray(fr.pilot_mask | fr.data_mask, dtype=bool)
                act_pow = float(np.mean(np.abs(fr.rx_grid[act_mask]) ** 2)) if np.any(act_mask) else 0.0
                n0_val = float(fr.est_n0)
                s2_scale = act_pow if (np.isfinite(act_pow) and act_pow > 0.0) else (
                    n0_val if (np.isfinite(n0_val) and n0_val > 0.0) else 1.0
                )
                v_eff = np.maximum(out.heldout_pilot_var + sig2_obs, max(1e-30 * s2_scale, 1e-300))
                err = out.heldout_pilot_pred - z_obs
                nlls.append(float(np.mean(np.log(np.pi * v_eff) + (np.abs(err) ** 2) / v_eff)))
    ber = float(bit_err / max(tot_bits, 1))
    sym_mse = float(sq_sym / max(tot_sym, 1))
    pilot_nll = float(np.mean(nlls)) if nlls else 0.0
    return ber, pilot_nll, sym_mse


def calibrate_on_dev(
    dev_frames: Sequence[RealOFDMFrame],
    *,
    methods: Sequence[str] = CLASSICAL_METHODS,
    n0_scales: Sequence[float] = (0.7, 1.0, 1.3),
    tau_rms_grid: Sequence[float] = (1.5, 2.5, 4.0),
    n_taps_grid: Sequence[int] = (5, 7, 9),
    fd_norm_grid: Sequence[float] = (0.0, 0.002, 0.005),
    cpe_refine_grid: Sequence[float] = (0.0,),
) -> dict[str, ClassicalConfig]:
    """Tune classical receiver hyperparameters strictly on dev frames (zero test leakage).

    Raises ValueError if `dev_frames` is empty or contains any `test` or `guard` split frame.
    """
    if not dev_frames:
        raise ValueError("dev_frames must be non-empty")
    for fr in dev_frames:
        sp = getattr(fr, "split", "")
        if str(sp).strip().lower() in ("test", "guard"):
            raise ValueError(
                f"Anti-leakage violation: calibrate_on_dev received frame {fr.frame_id} with split={sp!r}"
            )

    tuned: dict[str, ClassicalConfig] = {}
    for m_raw in methods:
        m = normalize_method_name(m_raw)
        if m == "ls_linear_zf":
            best_score = (float("inf"), float("inf"), float("inf"))
            best_cfg = ClassicalConfig()
            for n0s in n0_scales:
                cfg = ClassicalConfig(n0_scale=float(n0s))
                ber, nll, smse = _dev_objective(dev_frames, m, cfg)
                score = (ber, smse, nll)
                if score < best_score:
                    best_score = score
                    best_cfg = cfg
            tuned[m] = best_cfg
        elif m == "lmmse_1d":
            best_score = (float("inf"), float("inf"), float("inf"))
            best_cfg = ClassicalConfig()
            for nt in n_taps_grid:
                for trms in tau_rms_grid:
                    for n0s in n0_scales:
                        cfg = ClassicalConfig(n_taps=int(nt), tau_rms=float(trms), n0_scale=float(n0s))
                        ber, nll, smse = _dev_objective(dev_frames, m, cfg)
                        score = (ber, nll, smse)
                        if score < best_score:
                            best_score = score
                            best_cfg = cfg
            tuned[m] = best_cfg
        else:
            # Inherit tuned frequency prior from lmmse_1d when available, then tune temporal/phase parameters
            base_nt = tuned["lmmse_1d"].n_taps if "lmmse_1d" in tuned else 7
            base_tau = tuned["lmmse_1d"].tau_rms if "lmmse_1d" in tuned else 2.5
            base_n0s = tuned["lmmse_1d"].n0_scale if "lmmse_1d" in tuned else 1.0
            best_score = (float("inf"), float("inf"), float("inf"))
            best_cfg = ClassicalConfig(n_taps=base_nt, tau_rms=base_tau, n0_scale=base_n0s)
            for nt in (base_nt,) if "lmmse_1d" in tuned else n_taps_grid:
                for trms in (base_tau,) if "lmmse_1d" in tuned else tau_rms_grid:
                    for n0s in (base_n0s,) if "lmmse_1d" in tuned else n0_scales:
                        for fd in fd_norm_grid:
                            for cpe_a in cpe_refine_grid:
                                cfg = ClassicalConfig(
                                    n_taps=int(nt),
                                    tau_rms=float(trms),
                                    n0_scale=float(n0s),
                                    fd_norm=float(fd),
                                    phase_track=True,
                                    cpe_refine_alpha=float(cpe_a),
                                )
                                ber, nll, smse = _dev_objective(dev_frames, m, cfg)
                                score = (ber, smse, nll)
                                if score < best_score:
                                    best_score = score
                                    best_cfg = cfg
            tuned[m] = best_cfg
    return tuned


calibrate_classical = calibrate_on_dev

__all__ = [
    "CLASSICAL_METHODS",
    "ClassicalConfig",
    "ClassicalReceiver",
    "ReceiverOutput",
    "calibrate_classical",
    "calibrate_on_dev",
    "normalize_method_name",
    "run_frame",
]
