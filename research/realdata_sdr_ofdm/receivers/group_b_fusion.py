"""Group B receivers: hard selection vs. posterior mean plug-in vs. exact mixture likelihood (F9).

Implements a shared physical multi-hypothesis candidate bank and three controlled
decision rules on identical candidate branches `(m_r, v_r)` and posterior weights
`alpha_r`:
  1. ``hard_map``: deterministic MAP single-branch selection ``r* = argmax_r alpha_r``
     followed by MMSE equalization and calibrated exact LLRs.
  2. ``posterior_mean_plugin`` (alias ``weighted_plugin``): orthogonal-projection
     posterior mean ``m_bar = sum_r alpha_r m_r`` with exact Total Variance
     ``v_tot = sum_r alpha_r (v_r + |m_r - m_bar|^2)`` plugged into single-Gaussian
     MMSE equalization and exact LLRs.
  3. ``exact_mixture_likelihood`` (alias ``mixture_likelihood``): exact log-sum-exp
     symbol mixture likelihood
     ``log g(s_m) = LSE_{r: alpha_r > 0} (log alpha_r - log(pi * sigma2_r) - |y - m_r s_m|^2 / sigma2_r)``
     where ``sigma2_r = n0_cal + v_r`` (``include_constellation_var=False`` by default,
     matching ``classical.py`` to ``< 1e-11`` in the single-branch limit while preserving
     exact 24-octave C*-equivariance via observable gauge normalization before variance
     clamping) and exact bitwise LLRs ``LSE_{b=1} log g - LSE_{b=0} log g``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np

from research.realdata_sdr_ofdm.adapters.modulation import (
    _bit_index_groups,
    bits_to_symbols,
    constellation_points,
)
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    ACTIVE_END,
    ACTIVE_START,
    N_ACTIVE,
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
from research.realdata_sdr_ofdm.receivers.classical import (
    ClassicalConfig,
    ReceiverOutput,
    _equalize_and_demap,
    _estimate_sym2_channel,
)

GROUP_B_MODES: tuple[str, ...] = (
    "hard_map",
    "posterior_mean_plugin",
    "exact_mixture_likelihood",
)

_MODE_ALIASES: dict[str, str] = {
    "hard_map": "hard_map",
    "posterior_mean_plugin": "posterior_mean_plugin",
    "weighted_plugin": "posterior_mean_plugin",
    "exact_mixture_likelihood": "exact_mixture_likelihood",
    "mixture_likelihood": "exact_mixture_likelihood",
}


def normalize_group_b_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown Group B mode {mode!r}; expected one of {sorted(_MODE_ALIASES.keys())}"
        )
    return _MODE_ALIASES[key]


@dataclass
class CandidateBranch:
    """Single physical channel-estimation hypothesis in the Group B bank."""

    name: str
    h_est: np.ndarray
    h_var: np.ndarray
    log_score: float = 0.0
    omega_t: float = 0.0
    pdp_params: tuple[int, float] = (11, 1.0)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupBConfig:
    """Hyperparameters for Group B shared candidate bank & fusion (calibrated on dev)."""

    pdp_hypotheses: tuple[tuple[int, float], ...] = ((7, 0.8), (11, 1.0), (11, 2.0))
    n0_scale: float = 0.7
    diag_reg: float = 1e-4
    detrend_timing: bool = True
    phase_track_max_slope: float = 0.50
    phase_track_prior_weight: float = 2.0
    beta: float = 0.5
    nonzero_om_penalty: float = 4.0
    omega_min_diff: float = 1e-3
    include_constellation_var: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pdp_hypotheses"] = [[int(nt), float(tr)] for nt, tr in self.pdp_hypotheses]
        return d

    def classical_cfg_for(self, n_taps: int, tau_rms: float) -> ClassicalConfig:
        return ClassicalConfig(
            tau_rms=float(tau_rms),
            n_taps=int(n_taps),
            n0_scale=float(self.n0_scale),
            diag_reg=float(self.diag_reg),
            detrend_timing=bool(self.detrend_timing),
            phase_track=False,
        )


def _validate_frame_inputs(frame: RealOFDMFrame) -> tuple[np.ndarray, float, np.ndarray]:
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
    return rx_grid, est_n0, pilot_values


def _extract_active_slice(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.shape == (N_SYMB, N_ACTIVE):
        return a
    if a.shape == (N_SYMB, N_FFT):
        return a[:, ACTIVE_START:ACTIVE_END]
    raise ValueError(f"Expected {name} shape {(N_SYMB, N_ACTIVE)} or {(N_SYMB, N_FFT)}, got {a.shape}")


def _pdp_pilot_cv_ll(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    obs_idx: Sequence[int],
    n0_norm: float,
    *,
    cls_cfg: ClassicalConfig,
) -> float:
    """Compute averaged 4-fold held-out pilot CV log-likelihood on observable pilots only."""
    obs_set = set(int(i) for i in obs_idx)
    folds = build_heldout_pilot_masks(4)
    fold_lls: list[float] = []
    for fold in folds:
        h_sub = [int(i) for i in fold["heldout_pilot_indices"] if int(i) in obs_set]
        if not h_sub:
            continue
        o_sub = [int(i) for i in obs_idx if int(i) not in h_sub]
        if not o_sub:
            continue
        target_active = np.asarray([PILOT_ACTIVE_INDICES[i] for i in h_sub], dtype=np.int64)
        h_pred, v_pred, _ = _estimate_sym2_channel(
            rx_norm,
            pilot_values,
            o_sub,
            n0_norm,
            method="lmmse_1d",
            config=cls_cfg,
            target_active_indices=target_active,
        )
        fft_h = [PILOT_FFT_INDICES[i] for i in h_sub]
        y_h = rx_norm[PILOT_SYMBOL_IDX, fft_h]
        p_val = pilot_values[h_sub]
        z_obs = y_h / p_val
        sig2_obs = max(float(cls_cfg.n0_scale) * float(n0_norm), 0.0) / np.maximum(np.abs(p_val) ** 2, 1e-30)
        v_eff = np.maximum(np.maximum(v_pred, 0.0) + sig2_obs, 1e-12)
        err2 = np.abs(h_pred - z_obs) ** 2
        ll_fold = float(np.sum(-np.log(math.pi * v_eff) - err2 / v_eff))
        fold_lls.append(ll_fold)
    if not fold_lls:
        return 0.0
    return float(np.mean(fold_lls))


def _blind_marginal_ll_subset(
    y_sub: np.ndarray,
    s_sub: np.ndarray,
    h_sub_base: np.ndarray,
    v_sub: np.ndarray,
    n0_cal: float,
    omega: float,
    pts: np.ndarray,
) -> float:
    """Blind 16-QAM observation-space marginal log-likelihood on off-pilot RE subset."""
    h_rot = h_sub_base * np.exp(1j * float(omega) * s_sub)
    sigma2 = np.maximum(n0_cal + np.maximum(v_sub, 0.0), 1e-12)
    diff2 = np.abs(y_sub[:, None] - h_rot[:, None] * pts[None, :]) ** 2
    ll = -np.log(math.pi * sigma2)[:, None] - diff2 / sigma2[:, None]
    m_max = np.max(ll, axis=1, keepdims=True)
    lse = m_max[:, 0] + np.log(np.sum(np.exp(np.clip(ll - m_max, -700.0, 0.0)), axis=1))
    return float(np.sum(lse))


def _prepare_blind_omega_subset(
    rx_active: np.ndarray,
    data_mask_active: np.ndarray,
    h_1d: np.ndarray,
    v_1d: np.ndarray,
    n0_cal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select top-50% lowest-noise off-pilot REs for blind CFO/phase-drift scoring."""
    sym_grid = np.broadcast_to(np.arange(N_SYMB, dtype=np.float64)[:, None], (N_SYMB, N_ACTIVE))
    h_2d = np.broadcast_to(h_1d[None, :], (N_SYMB, N_ACTIVE))
    v_2d = np.broadcast_to(v_1d[None, :], (N_SYMB, N_ACTIVE))

    s_data = sym_grid[data_mask_active]
    y_data = rx_active[data_mask_active]
    h_data = h_2d[data_mask_active]
    v_data = v_2d[data_mask_active]

    off_mask = s_data != float(PILOT_SYMBOL_IDX)
    y_off = y_data[off_mask]
    s_off = s_data[off_mask] - float(PILOT_SYMBOL_IDX)
    h_off = h_data[off_mask]
    v_off = v_data[off_mask]

    h_abs2 = np.abs(h_off) ** 2
    sig2_eff = np.where(
        h_abs2 > 1e-30,
        np.maximum((n0_cal + np.maximum(v_off, 0.0)) / np.maximum(h_abs2, 1e-30), 1e-12),
        1e12,
    )
    k_keep = max(min(650, len(y_off)), len(y_off) // 2)
    top_idx = np.argpartition(sig2_eff, k_keep - 1)[:k_keep]
    return y_off[top_idx], s_off[top_idx], h_off[top_idx], v_off[top_idx]


def _find_omega_peak(
    rx_active: np.ndarray,
    data_mask_active: np.ndarray,
    h_1d_anchor: np.ndarray,
    v_1d_anchor: np.ndarray,
    n0_cal: float,
    cfg: GroupBConfig,
) -> float:
    """Coarse (41) -> fine (17) blind marginal-likelihood peak search for temporal phase drift."""
    if float(np.max(np.abs(h_1d_anchor))) <= 1e-14:
        return 0.0
    y_sub, s_sub, h_sub, v_sub = _prepare_blind_omega_subset(
        rx_active, data_mask_active, h_1d_anchor, v_1d_anchor, n0_cal
    )
    pts = constellation_points(QM)
    max_slope = max(float(cfg.phase_track_max_slope), 1e-6)
    prior_w = float(cfg.phase_track_prior_weight)
    sigma2 = np.maximum(n0_cal + np.maximum(v_sub, 0.0), 1e-12)
    inv_sig2 = (1.0 / sigma2)[:, None]
    log_norm = (-np.log(math.pi * sigma2))[:, None]
    y_col = y_sub[:, None]
    pts_row = pts[None, :]

    def _score_om(om: float) -> float:
        h_rot = h_sub * np.exp(1j * om * s_sub)
        diff2 = np.abs(y_col - h_rot[:, None] * pts_row) ** 2
        ll = log_norm - diff2 * inv_sig2
        m_max = np.max(ll, axis=1, keepdims=True)
        lse = m_max[:, 0] + np.log(np.sum(np.exp(np.clip(ll - m_max, -700.0, 0.0)), axis=1))
        return float(np.sum(lse)) - 0.5 * prior_w * (om / max_slope) ** 2

    coarse_grid = np.linspace(-max_slope, max_slope, 41, dtype=np.float64)
    coarse_scores = np.array([_score_om(float(om)) for om in coarse_grid], dtype=np.float64)
    best_om = float(coarse_grid[int(np.argmax(coarse_scores))])
    step = float(coarse_grid[1] - coarse_grid[0])

    fine_grid = np.linspace(
        max(-max_slope, best_om - step),
        min(max_slope, best_om + step),
        17,
        dtype=np.float64,
    )
    fine_scores = np.array([_score_om(float(om)) for om in fine_grid], dtype=np.float64)
    return float(fine_grid[int(np.argmax(fine_scores))])


def _build_bank_normalized(
    frame: RealOFDMFrame,
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    s_scale: float,
    obs_idx: Sequence[int],
    *,
    cfg: GroupBConfig,
    include_temporal_branches: bool = True,
) -> tuple[list[CandidateBranch], np.ndarray]:
    """Build shared candidate bank in normalized gauge and return (branches_orig_gauge, alpha)."""
    pdp_list = list(cfg.pdp_hypotheses)
    if not pdp_list:
        raise ValueError("GroupBConfig.pdp_hypotheses must be non-empty")

    n0_cal = max(float(cfg.n0_scale) * float(n0_norm), 0.0)
    data_mask_active = np.asarray(frame.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
    rx_active = rx_norm[:, ACTIVE_START:ACTIVE_END]

    pdp_h1d: list[np.ndarray] = []
    pdp_v1d: list[np.ndarray] = []
    pdp_cv_ll: list[float] = []
    for n_taps, tau_rms in pdp_list:
        cls_cfg = cfg.classical_cfg_for(n_taps, tau_rms)
        h_1d, v_1d, _ = _estimate_sym2_channel(
            rx_norm,
            pilot_values,
            obs_idx,
            n0_norm,
            method="lmmse_1d",
            config=cls_cfg,
        )
        cv_ll = _pdp_pilot_cv_ll(rx_norm, pilot_values, obs_idx, n0_norm, cls_cfg=cls_cfg)
        pdp_h1d.append(h_1d)
        pdp_v1d.append(v_1d)
        pdp_cv_ll.append(cv_ll)

    if include_temporal_branches:
        anchor_idx = min(1, len(pdp_list) - 1)
        omega_peak = _find_omega_peak(
            rx_active,
            data_mask_active,
            pdp_h1d[anchor_idx],
            pdp_v1d[anchor_idx],
            n0_cal,
            cfg,
        )
        if abs(omega_peak) < float(cfg.omega_min_diff):
            omega_list = [0.0]
        else:
            omega_list = [0.0, omega_peak]
    else:
        omega_list = [0.0]

    sym_offsets = np.arange(N_SYMB, dtype=np.float64) - float(PILOT_SYMBOL_IDX)
    pts = constellation_points(QM)
    max_slope = max(float(cfg.phase_track_max_slope), 1e-6)
    prior_w = float(cfg.phase_track_prior_weight)

    branches: list[CandidateBranch] = []
    raw_scores: list[float] = []
    s_scale2 = s_scale * s_scale

    for b_idx, (n_taps, tau_rms) in enumerate(pdp_list):
        h_1d = pdp_h1d[b_idx]
        v_1d = pdp_v1d[b_idx]
        v_2d_norm = np.broadcast_to(v_1d[None, :], (N_SYMB, N_ACTIVE)).copy()

        ll0 = 0.0
        if len(omega_list) > 1:
            y_sub, s_sub, h_sub, v_sub = _prepare_blind_omega_subset(
                rx_active, data_mask_active, h_1d, v_1d, n0_cal
            )
            ll0 = _blind_marginal_ll_subset(y_sub, s_sub, h_sub, v_sub, n0_cal, 0.0, pts)

        for om in omega_list:
            if om == 0.0:
                h_2d_norm = np.broadcast_to(h_1d[None, :], (N_SYMB, N_ACTIVE)).copy()
                delta_ll_om = 0.0
            else:
                rot = np.exp(1j * float(om) * sym_offsets)[:, None]
                h_2d_norm = (h_1d[None, :] * rot).astype(np.complex128)
                ll_om = _blind_marginal_ll_subset(y_sub, s_sub, h_sub, v_sub, n0_cal, float(om), pts)
                delta_ll_om = (
                    (ll_om - ll0)
                    - 0.5 * prior_w * (float(om) / max_slope) ** 2
                    - float(cfg.nonzero_om_penalty)
                )

            score_r = float(pdp_cv_ll[b_idx] + delta_ll_om)
            raw_scores.append(score_r)
            branches.append(
                CandidateBranch(
                    name=f"pdp_t{n_taps}_r{tau_rms:.1f}_om{om:+.4f}",
                    h_est=(h_2d_norm * s_scale).astype(np.complex128),
                    h_var=np.maximum(v_2d_norm * s_scale2, 1e-30 * s_scale2).astype(np.float64),
                    log_score=score_r,
                    omega_t=float(om),
                    pdp_params=(int(n_taps), float(tau_rms)),
                    meta={
                        "pdp_cv_ll": float(pdp_cv_ll[b_idx]),
                        "delta_ll_omega": float(delta_ll_om),
                    },
                )
            )

    scores_arr = np.asarray(raw_scores, dtype=np.float64) * float(cfg.beta)
    scores_arr -= float(np.max(scores_arr))
    scores_arr = np.clip(scores_arr, -700.0, 0.0)
    w = np.exp(scores_arr)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        alpha = np.full(len(branches), 1.0 / float(len(branches)), dtype=np.float64)
    else:
        alpha = w / w_sum
    return branches, alpha


def build_candidate_bank(
    frame: RealOFDMFrame,
    *,
    context_pool: Any = None,
    config: Optional[GroupBConfig] = None,
) -> tuple[list[CandidateBranch], np.ndarray]:
    """Build shared Group B physical candidate bank and posterior weights (zero label access)."""
    del context_pool
    cfg = GroupBConfig() if config is None else config
    rx_grid, est_n0, pilot_values = _validate_frame_inputs(frame)
    active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
    s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
    if not np.isfinite(s_rms):
        raise ValueError("Non-finite active RMS power")
    s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
    rx_norm = rx_grid / s_scale
    n0_norm = est_n0 / (s_scale ** 2)
    return _build_bank_normalized(
        frame,
        rx_norm,
        pilot_values,
        n0_norm,
        s_scale,
        list(range(14)),
        cfg=cfg,
        include_temporal_branches=True,
    )


build_candidate_branches = build_candidate_bank


def _normalize_branch_weights(weights: Sequence[float], n_branches: int) -> np.ndarray:
    if n_branches < 1:
        raise ValueError("branches must be non-empty")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (n_branches,):
        raise ValueError(f"Expected branch_weights length {n_branches}, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("branch_weights contains non-finite values")
    if np.any(w < -1e-12):
        raise ValueError(f"branch_weights contains negative entries: min={float(np.min(w))}")
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("branch_weights must sum to a strictly positive finite value")
    return (w / w_sum).astype(np.float64)


def _compute_heldout_cv_from_fused(
    frame: RealOFDMFrame,
    h_est: np.ndarray,
    h_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    explicit_idx = frame.meta.get("heldout_pilot_indices") if isinstance(frame.meta, dict) else None
    if explicit_idx is not None and len(explicit_idx) > 0:
        eval_idx = [int(i) for i in explicit_idx]
    else:
        eval_idx = list(range(1, 13))
    act_idx = [PILOT_ACTIVE_INDICES[i] for i in eval_idx]
    pred = np.asarray(h_est[PILOT_SYMBOL_IDX, act_idx], dtype=np.complex128)
    var = np.maximum(np.asarray(h_var[PILOT_SYMBOL_IDX, act_idx], dtype=np.float64), 1e-30)
    return pred, var, eval_idx


def _cv_fold_normalized_inputs(
    frame: RealOFDMFrame,
    rx_grid: np.ndarray,
    est_n0: float,
    obs_idx: Sequence[int],
) -> tuple[np.ndarray, float, float]:
    """Compute fold-local observable normalization strictly excluding held-out pilot REs."""
    obs_pilot_mask = np.zeros_like(frame.pilot_mask, dtype=bool)
    if len(obs_idx) > 0:
        fft_obs: list[int] = []
        for i in obs_idx:
            pi_int = int(i)
            if not (0 <= pi_int < len(PILOT_FFT_INDICES)):
                raise ValueError(
                    f"Observed pilot index out of range [0, {len(PILOT_FFT_INDICES) - 1}]: {i!r}"
                )
            fft_obs.append(PILOT_FFT_INDICES[pi_int])
        obs_pilot_mask[PILOT_SYMBOL_IDX, fft_obs] = True
    p_mask = np.asarray(frame.pilot_mask, dtype=bool)
    d_mask = np.asarray(frame.data_mask, dtype=bool)
    obs_pilot_mask &= p_mask
    cv_active_mask = obs_pilot_mask | (d_mask & ~p_mask)
    if np.any(cv_active_mask):
        s_rms_cv = float(np.sqrt(np.mean(np.abs(rx_grid[cv_active_mask]) ** 2)))
        if not np.isfinite(s_rms_cv):
            raise ValueError("Non-finite CV active RMS power")
    else:
        s_rms_cv = 0.0
    s_scale_cv = s_rms_cv if s_rms_cv > 0.0 else max(math.sqrt(est_n0), 1.0)
    rx_norm_cv = rx_grid / s_scale_cv
    n0_norm_cv = est_n0 / (s_scale_cv ** 2)
    return rx_norm_cv, n0_norm_cv, s_scale_cv


def _compute_heldout_cv_group_b(
    frame: RealOFDMFrame,
    rx_grid: np.ndarray,
    pilot_values: np.ndarray,
    est_n0: float,
    *,
    mode: str,
    cfg: GroupBConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Leak-free held-out pilot CV building candidate banks strictly on non-heldout pilots."""
    explicit_idx = frame.meta.get("heldout_pilot_indices") if isinstance(frame.meta, dict) else None
    if explicit_idx is not None and len(explicit_idx) > 0:
        eval_idx = [int(i) for i in explicit_idx]
        eval_set = set(eval_idx)
        obs_idx = [i for i in range(14) if i not in eval_set]
        rx_norm_cv, n0_norm_cv, s_scale_cv = _cv_fold_normalized_inputs(
            frame, rx_grid, est_n0, obs_idx
        )
        branches, alpha = _build_bank_normalized(
            frame,
            rx_norm_cv,
            pilot_values,
            n0_norm_cv,
            s_scale_cv,
            obs_idx,
            cfg=cfg,
            include_temporal_branches=False,
        )
        v_floor = 1e-30 * (s_scale_cv ** 2)
        h_agg, v_agg = _aggregate_h_only(branches, alpha, mode=mode, min_var=v_floor)
        act_idx = [PILOT_ACTIVE_INDICES[i] for i in eval_idx]
        pred = np.asarray(h_agg[PILOT_SYMBOL_IDX, act_idx], dtype=np.complex128)
        var = np.maximum(np.asarray(v_agg[PILOT_SYMBOL_IDX, act_idx], dtype=np.float64), v_floor)
        return pred, var, eval_idx

    folds = frame.meta.get("heldout_pilot_folds") if isinstance(frame.meta, dict) else None
    if not folds:
        folds = build_heldout_pilot_masks(4)

    collected: dict[int, tuple[complex, float]] = {}
    for fold in folds:
        h_idx = [int(i) for i in fold["heldout_pilot_indices"]]
        h_set = set(h_idx)
        o_idx = [int(i) for i in fold["obs_pilot_indices"] if int(i) not in h_set]
        rx_norm_cv, n0_norm_cv, s_scale_cv = _cv_fold_normalized_inputs(
            frame, rx_grid, est_n0, o_idx
        )
        branches, alpha = _build_bank_normalized(
            frame,
            rx_norm_cv,
            pilot_values,
            n0_norm_cv,
            s_scale_cv,
            o_idx,
            cfg=cfg,
            include_temporal_branches=False,
        )
        v_floor = 1e-30 * (s_scale_cv ** 2)
        h_agg, v_agg = _aggregate_h_only(branches, alpha, mode=mode, min_var=v_floor)
        for pi in h_idx:
            if pi not in collected:
                ai = PILOT_ACTIVE_INDICES[pi]
                collected[pi] = (
                    complex(h_agg[PILOT_SYMBOL_IDX, ai]),
                    max(float(v_agg[PILOT_SYMBOL_IDX, ai]), v_floor),
                )

    eval_idx = sorted(collected.keys())
    pred = np.asarray([collected[i][0] for i in eval_idx], dtype=np.complex128)
    var = np.asarray([collected[i][1] for i in eval_idx], dtype=np.float64)
    return pred, var, eval_idx


def _aggregate_h_only(
    branches: Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]],
    alpha: np.ndarray,
    *,
    mode: str,
    min_var: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    active_indices = [r for r in range(len(branches)) if float(alpha[r]) > 0.0]
    if mode == "hard_map":
        r_star = int(np.argmax(alpha))
        br = branches[r_star]
        m_r = _extract_active_slice(br.h_est if isinstance(br, CandidateBranch) else br[0], "h_est")
        v_r = _extract_active_slice(br.h_var if isinstance(br, CandidateBranch) else br[1], "h_var")
        return np.asarray(m_r, dtype=np.complex128), np.maximum(np.asarray(v_r, dtype=np.float64), float(min_var))

    h_bar = np.zeros((N_SYMB, N_ACTIVE), dtype=np.complex128)
    v_within = np.zeros((N_SYMB, N_ACTIVE), dtype=np.float64)
    m_cache: dict[int, np.ndarray] = {}
    for r in active_indices:
        br = branches[r]
        m_r = np.asarray(
            _extract_active_slice(br.h_est if isinstance(br, CandidateBranch) else br[0], "h_est"),
            dtype=np.complex128,
        )
        v_r = np.maximum(
            np.asarray(
                _extract_active_slice(br.h_var if isinstance(br, CandidateBranch) else br[1], "h_var"),
                dtype=np.float64,
            ),
            0.0,
        )
        m_cache[r] = m_r
        h_bar += float(alpha[r]) * m_r
        v_within += float(alpha[r]) * v_r

    v_between = np.zeros((N_SYMB, N_ACTIVE), dtype=np.float64)
    for r in active_indices:
        v_between += float(alpha[r]) * (np.abs(m_cache[r] - h_bar) ** 2)
    v_tot = np.maximum(v_within + v_between, float(min_var))
    return h_bar, v_tot


def fuse_candidate_branches(
    frame: RealOFDMFrame,
    *,
    branches: Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]],
    branch_weights: Sequence[float],
    mode: str = "exact_mixture_likelihood",
    include_constellation_var: bool = False,
    n0_scale: Optional[float] = None,
    config: Optional[GroupBConfig] = None,
    return_heldout_cv: bool = True,
) -> ReceiverOutput:
    """Fuse caller-supplied candidate branches under exact Group B decision rules.

    CRITICAL INVARIANTS:
    - Branches with ``alpha_r <= 0.0`` are strictly skipped BEFORE touching ``m_r`` or ``v_r``.
    - Observable ``s_scale`` normalizes ``(y, m_r, v_r, est_n0)`` BEFORE applying any ``1e-12``
      variance floor, guaranteeing exact 24-octave C*-equivariance.
    - ``include_constellation_var=False`` (default) uses ``sigma2_r = max(n0_cal + v_r, 1e-12)``
      so ``R=1`` reduces identically to ``classical.py`` (``< 1e-11`` LLR diff, ``0`` bit flips).
    """
    t0 = time.perf_counter()
    canon_mode = normalize_group_b_mode(mode)
    if n0_scale is not None:
        eff_n0_scale = float(n0_scale)
    elif config is not None:
        eff_n0_scale = float(config.n0_scale)
    else:
        eff_n0_scale = 1.0

    rx_grid, est_n0, _ = _validate_frame_inputs(frame)
    alpha = _normalize_branch_weights(branch_weights, len(branches))
    active_indices = [r for r in range(len(branches)) if float(alpha[r]) > 0.0]

    active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
    s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
    if not np.isfinite(s_rms):
        raise ValueError("Non-finite active RMS power")
    s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
    inv_s = 1.0 / s_scale
    inv_s2 = inv_s * inv_s

    rx_norm = rx_grid * inv_s
    n0_norm = est_n0 * inv_s2
    n0_cal = max(eff_n0_scale * n0_norm, 0.0)
    data_mask_active = np.asarray(frame.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
    rx_active = rx_norm[:, ACTIVE_START:ACTIVE_END]
    y_data = rx_active[data_mask_active]

    # Extract & validate ONLY active branches (alpha_r > 0) — NEVER touch zero-weight branches!
    m_norm_active: dict[int, np.ndarray] = {}
    v_norm_active: dict[int, np.ndarray] = {}
    branch_names: list[str] = []
    for r, br in enumerate(branches):
        branch_names.append(br.name if isinstance(br, CandidateBranch) else f"branch_{r}")
        if float(alpha[r]) <= 0.0:
            continue
        m_raw = _extract_active_slice(br.h_est if isinstance(br, CandidateBranch) else br[0], f"branches[{r}].h_est")
        v_raw = _extract_active_slice(br.h_var if isinstance(br, CandidateBranch) else br[1], f"branches[{r}].h_var")
        m_arr = np.asarray(m_raw, dtype=np.complex128)
        v_arr = np.asarray(v_raw, dtype=np.float64)
        if not np.all(np.isfinite(m_arr)):
            raise ValueError(f"Active branch {r} h_est contains non-finite values")
        if np.iscomplexobj(v_arr) or not np.all(np.isfinite(v_arr)) or np.any(v_arr < 0.0):
            raise ValueError(f"Active branch {r} h_var contains non-finite, complex, or negative values")
        m_norm_active[r] = m_arr * inv_s
        v_norm_active[r] = np.maximum(v_arr * inv_s2, 0.0)

    selected_branch: Optional[int] = None
    between_var_mean = 0.0
    cls_demap_cfg = ClassicalConfig(n0_scale=eff_n0_scale)

    if canon_mode == "hard_map":
        r_star = int(np.argmax(alpha))
        selected_branch = r_star
        h_2d_norm = m_norm_active[r_star]
        v_2d_norm = np.maximum(v_norm_active[r_star], 1e-30)
        eq_symbols, bit_llrs, hard_bits, hard_symbols = _equalize_and_demap(
            y_data,
            h_2d_norm[data_mask_active],
            v_2d_norm[data_mask_active],
            n0_norm,
            method="lmmse_1d",
            config=cls_demap_cfg,
        )
    else:
        # Both posterior_mean_plugin and exact_mixture_likelihood report exact posterior mean + Total Variance
        h_2d_norm = np.zeros((N_SYMB, N_ACTIVE), dtype=np.complex128)
        v_within = np.zeros((N_SYMB, N_ACTIVE), dtype=np.float64)
        for r in active_indices:
            w_r = float(alpha[r])
            h_2d_norm += w_r * m_norm_active[r]
            v_within += w_r * v_norm_active[r]

        v_between = np.zeros((N_SYMB, N_ACTIVE), dtype=np.float64)
        for r in active_indices:
            v_between += float(alpha[r]) * (np.abs(m_norm_active[r] - h_2d_norm) ** 2)
        between_var_mean = float(np.mean(v_between[data_mask_active])) * (s_scale ** 2)
        v_2d_norm = np.maximum(v_within + v_between, 1e-30)

        eq_symbols, plugin_llrs, plugin_bits, plugin_syms = _equalize_and_demap(
            y_data,
            h_2d_norm[data_mask_active],
            v_2d_norm[data_mask_active],
            n0_norm,
            method="lmmse_1d",
            config=cls_demap_cfg,
        )

        if canon_mode == "posterior_mean_plugin":
            bit_llrs = plugin_llrs
            hard_bits = plugin_bits
            hard_symbols = plugin_syms
        else:  # exact_mixture_likelihood
            pts = constellation_points(QM)  # (16,)
            idx1, idx0 = _bit_index_groups(QM)
            pts_pow = np.abs(pts) ** 2
            log_terms = np.empty((len(active_indices), len(y_data), len(pts)), dtype=np.float64)

            for pos, r in enumerate(active_indices):
                m_d = m_norm_active[r][data_mask_active]
                v_d = v_norm_active[r][data_mask_active]
                diff2 = np.abs(y_data[:, None] - m_d[:, None] * pts[None, :]) ** 2
                if include_constellation_var:
                    sigma2_rm = np.maximum(n0_cal + v_d[:, None] * pts_pow[None, :], 1e-12)
                    log_g_r = -np.log(math.pi * sigma2_rm) - diff2 / sigma2_rm
                else:
                    sigma2_r = np.maximum(n0_cal + v_d, 1e-12)
                    log_g_r = -np.log(math.pi * sigma2_r[:, None]) - diff2 / sigma2_r[:, None]
                log_terms[pos] = math.log(float(alpha[r])) + log_g_r

            m_max = np.max(log_terms, axis=0, keepdims=True)
            log_mix = m_max[0] + np.log(
                np.sum(np.exp(np.clip(log_terms - m_max, -700.0, 0.0)), axis=0)
            )

            ll1 = log_mix[:, idx1]
            m1 = np.max(ll1, axis=-1, keepdims=True)
            lse1 = m1[..., 0] + np.log(np.sum(np.exp(np.clip(ll1 - m1, -700.0, 0.0)), axis=-1))

            ll0 = log_mix[:, idx0]
            m0 = np.max(ll0, axis=-1, keepdims=True)
            lse0 = m0[..., 0] + np.log(np.sum(np.exp(np.clip(ll0 - m0, -700.0, 0.0)), axis=-1))

            bit_llrs = (lse1 - lse0).astype(np.float64)
            hard_bits = (bit_llrs > 0.0).astype(np.uint8)
            hard_symbols = bits_to_symbols(hard_bits, qm=QM)

    h_est = (h_2d_norm * s_scale).astype(np.complex128)
    h_var = np.maximum(v_2d_norm * (s_scale ** 2), 1e-30 * (s_scale ** 2)).astype(np.float64)
    if not (np.all(np.isfinite(h_est)) and np.all(np.isfinite(h_var))):
        raise ValueError("Fused channel estimate contains non-finite values")
    if not (np.all(np.isfinite(eq_symbols)) and np.all(np.isfinite(bit_llrs))):
        raise ValueError("Equalized symbols or bit_llrs contain non-finite values")

    heldout_pred: Optional[np.ndarray] = None
    heldout_var: Optional[np.ndarray] = None
    heldout_indices: list[int] = []
    if return_heldout_cv:
        heldout_pred, heldout_var, heldout_indices = _compute_heldout_cv_from_fused(frame, h_est, h_var)

    pos_a = alpha[alpha > 0.0]
    branch_entropy = float(-np.sum(pos_a * np.log(pos_a)))
    latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)

    aux_info: dict[str, Any] = {
        "mode": canon_mode,
        "decision_rule": canon_mode,
        "n_candidates": int(len(branches)),
        "retained_count": int(len(active_indices)),
        "branch_weights": [float(x) for x in alpha.tolist()],
        "branch_names": branch_names,
        "selected_branch": selected_branch,
        "between_var_mean": float(between_var_mean),
        "branch_entropy": branch_entropy,
        "include_constellation_var": bool(include_constellation_var),
        "pilot_aux_dependencies": "caller-supplied candidate branches",
        "latency_ms": float(latency_ms),
        "failed": False,
        "error": None,
        "param_count": 0,
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


def _failed_output(err_msg: str, mode: str, latency_ms: float) -> ReceiverOutput:
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
            "mode": mode,
            "decision_rule": mode,
            "n_candidates": 1,
            "retained_count": 1,
            "branch_weights": [1.0],
            "branch_names": ["failed"],
            "selected_branch": 0 if mode == "hard_map" else None,
            "latency_ms": float(latency_ms),
            "failed": True,
            "error": str(err_msg),
            "param_count": 0,
            "heldout_pilot_indices": [],
        },
    )


def run_frame(
    frame: RealOFDMFrame,
    context_pool: Any = None,
    *,
    mode: str = "exact_mixture_likelihood",
    branches: Optional[Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]]] = None,
    branch_weights: Optional[Sequence[float]] = None,
    include_constellation_var: Optional[bool] = None,
    config: Optional[GroupBConfig] = None,
    return_heldout_cv: bool = True,
    **kwargs: Any,
) -> ReceiverOutput:
    """Run Group B multi-hypothesis receiver on one RealOFDMFrame (zero label access)."""
    del context_pool, kwargs
    t0 = time.perf_counter()
    canon_mode = normalize_group_b_mode(mode)
    cfg = GroupBConfig() if config is None else config
    eff_inc_cvar = (
        bool(cfg.include_constellation_var)
        if include_constellation_var is None
        else bool(include_constellation_var)
    )

    try:
        if branches is not None:
            if branch_weights is None:
                weights_use: Sequence[float] = [1.0 / float(len(branches))] * len(branches)
            else:
                weights_use = branch_weights
            out = fuse_candidate_branches(
                frame,
                branches=branches,
                branch_weights=weights_use,
                mode=canon_mode,
                include_constellation_var=eff_inc_cvar,
                n0_scale=(1.0 if config is None else float(cfg.n0_scale)),
                config=config,
                return_heldout_cv=return_heldout_cv,
            )
            out.aux_info["latency_ms"] = max((time.perf_counter() - t0) * 1000.0, 1e-6)
            return out

        rx_grid, est_n0, pilot_values = _validate_frame_inputs(frame)
        active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
        s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
        if not np.isfinite(s_rms):
            raise ValueError("Non-finite active RMS power")
        s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
        rx_norm = rx_grid / s_scale
        n0_norm = est_n0 / (s_scale ** 2)

        built_branches, built_alpha = _build_bank_normalized(
            frame,
            rx_norm,
            pilot_values,
            n0_norm,
            s_scale,
            list(range(14)),
            cfg=cfg,
            include_temporal_branches=True,
        )
        out = fuse_candidate_branches(
            frame,
            branches=built_branches,
            branch_weights=built_alpha,
            mode=canon_mode,
            include_constellation_var=eff_inc_cvar,
            n0_scale=float(cfg.n0_scale),
            config=cfg,
            return_heldout_cv=False,
        )

        if return_heldout_cv:
            pred, var, eval_idx = _compute_heldout_cv_group_b(
                frame,
                rx_grid,
                pilot_values,
                est_n0,
                mode=canon_mode,
                cfg=cfg,
            )
            if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(var))):
                raise ValueError("Held-out pilot predictions contain non-finite values")
            out.heldout_pilot_pred = pred
            out.heldout_pilot_var = var
            out.aux_info["heldout_pilot_indices"] = eval_idx

        out.aux_info["omega_t_candidates"] = sorted(set(float(b.omega_t) for b in built_branches))
        out.aux_info["pilot_aux_dependencies"] = (
            "14 QPSK pilots (Sp=2) + blind top-50% off-pilot marginal LL (zero labels)"
        )
        out.aux_info["latency_ms"] = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        return out
    except Exception as exc:  # noqa: BLE001
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        return _failed_output(f"{type(exc).__name__}: {exc}", canon_mode, latency_ms)


def _scores_to_alpha(raw_scores: np.ndarray, beta: float) -> np.ndarray:
    s = np.asarray(raw_scores, dtype=np.float64) * float(beta)
    s = s - float(np.max(s))
    s = np.clip(s, -700.0, 0.0)
    w = np.exp(s)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return np.full(len(s), 1.0 / float(len(s)), dtype=np.float64)
    return w / w_sum


def calibrate_group_b_on_dev(
    dev_frames: Sequence[RealOFDMFrame],
    *,
    betas: Sequence[float] = (0.25, 0.5, 1.0),
    nonzero_om_penalties: Sequence[float] = (2.0, 4.0, 8.0),
    n0_scales: Sequence[float] = (0.7,),
    base_config: Optional[GroupBConfig] = None,
) -> tuple[GroupBConfig, dict[str, Any]]:
    """Calibrate Group B candidate-bank tempering & penalty strictly on dev frames."""
    if not dev_frames:
        raise ValueError("dev_frames must be non-empty")
    for fr in dev_frames:
        sp = str(getattr(fr, "split", "")).strip().lower()
        if sp in ("test", "guard"):
            raise ValueError(
                f"Anti-leakage violation: calibrate_group_b_on_dev received frame {fr.frame_id} with split={fr.split!r}"
            )

    base = GroupBConfig() if base_config is None else base_config
    best_score = (float("inf"), float("inf"), float("inf"))
    best_cfg = base
    grid_records: list[dict[str, Any]] = []

    for n0s in n0_scales:
        # Precompute candidate banks and base scores once per n0_scale (invariant to beta & nonzero_om_penalty)
        proto_cfg = GroupBConfig(
            pdp_hypotheses=base.pdp_hypotheses,
            n0_scale=float(n0s),
            diag_reg=base.diag_reg,
            detrend_timing=base.detrend_timing,
            phase_track_max_slope=base.phase_track_max_slope,
            phase_track_prior_weight=base.phase_track_prior_weight,
            beta=base.beta,
            nonzero_om_penalty=0.0,
            omega_min_diff=base.omega_min_diff,
            include_constellation_var=base.include_constellation_var,
        )
        cached_items: list[dict[str, Any]] = []
        for fr in dev_frames:
            try:
                rx_grid, est_n0, pilot_values = _validate_frame_inputs(fr)
                active_mask = np.asarray(fr.pilot_mask | fr.data_mask, dtype=bool)
                s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
                if not np.isfinite(s_rms):
                    raise ValueError("Non-finite active RMS power")
                s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
                rx_norm = rx_grid / s_scale
                n0_norm = est_n0 / (s_scale ** 2)

                built_branches, _ = _build_bank_normalized(
                    fr,
                    rx_norm,
                    pilot_values,
                    n0_norm,
                    s_scale,
                    list(range(14)),
                    cfg=proto_cfg,
                    include_temporal_branches=True,
                )
                base_scores = np.asarray([float(br.log_score) for br in built_branches], dtype=np.float64)
                is_nonzero_om = np.asarray([1.0 if float(br.omega_t) != 0.0 else 0.0 for br in built_branches], dtype=np.float64)

                # Precompute heldout CV banks (temporal_branches=False => independent of penalty)
                explicit_idx = fr.meta.get("heldout_pilot_indices") if isinstance(fr.meta, dict) else None
                cv_banks: list[tuple[list[int], list[CandidateBranch], np.ndarray, float]] = []
                if explicit_idx is not None and len(explicit_idx) > 0:
                    eval_idx = [int(i) for i in explicit_idx]
                    eval_set = set(eval_idx)
                    obs_idx = [i for i in range(14) if i not in eval_set]
                    rx_norm_cv, n0_norm_cv, s_scale_cv = _cv_fold_normalized_inputs(
                        fr, rx_grid, est_n0, obs_idx
                    )
                    cv_br, _ = _build_bank_normalized(
                        fr,
                        rx_norm_cv,
                        pilot_values,
                        n0_norm_cv,
                        s_scale_cv,
                        obs_idx,
                        cfg=proto_cfg,
                        include_temporal_branches=False,
                    )
                    cv_banks.append(
                        (
                            eval_idx,
                            cv_br,
                            np.asarray([float(b.log_score) for b in cv_br], dtype=np.float64),
                            s_scale_cv,
                        )
                    )
                else:
                    folds = fr.meta.get("heldout_pilot_folds") if isinstance(fr.meta, dict) else None
                    if not folds:
                        folds = build_heldout_pilot_masks(4)
                    for fold in folds:
                        h_idx = [int(i) for i in fold["heldout_pilot_indices"]]
                        h_set = set(h_idx)
                        o_idx = [int(i) for i in fold["obs_pilot_indices"] if int(i) not in h_set]
                        rx_norm_cv, n0_norm_cv, s_scale_cv = _cv_fold_normalized_inputs(
                            fr, rx_grid, est_n0, o_idx
                        )
                        cv_br, _ = _build_bank_normalized(
                            fr,
                            rx_norm_cv,
                            pilot_values,
                            n0_norm_cv,
                            s_scale_cv,
                            o_idx,
                            cfg=proto_cfg,
                            include_temporal_branches=False,
                        )
                        cv_banks.append(
                            (
                                h_idx,
                                cv_br,
                                np.asarray([float(b.log_score) for b in cv_br], dtype=np.float64),
                                s_scale_cv,
                            )
                        )

                all_eval = sorted({pi for h_idx, _, _, _ in cv_banks for pi in h_idx})
                fft_idx = [PILOT_FFT_INDICES[i] for i in all_eval]
                y_p = fr.rx_grid[PILOT_SYMBOL_IDX, fft_idx]
                p_val = fr.pilot_values[all_eval]
                z_obs = y_p / p_val
                sig2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)

                cached_items.append(
                    {
                        "failed": False,
                        "fr": fr,
                        "s_scale": s_scale,
                        "built_branches": built_branches,
                        "base_scores": base_scores,
                        "is_nonzero_om": is_nonzero_om,
                        "cv_banks": cv_banks,
                        "all_eval": all_eval,
                        "z_obs": z_obs,
                        "sig2_obs": sig2_obs,
                    }
                )
            except Exception:  # noqa: BLE001
                cached_items.append({"failed": True})

        for beta in betas:
            for pen in nonzero_om_penalties:
                cand_cfg = GroupBConfig(
                    pdp_hypotheses=base.pdp_hypotheses,
                    n0_scale=float(n0s),
                    diag_reg=base.diag_reg,
                    detrend_timing=base.detrend_timing,
                    phase_track_max_slope=base.phase_track_max_slope,
                    phase_track_prior_weight=base.phase_track_prior_weight,
                    beta=float(beta),
                    nonzero_om_penalty=float(pen),
                    omega_min_diff=base.omega_min_diff,
                    include_constellation_var=base.include_constellation_var,
                )
                bit_err = 0
                tot_bits = 0
                sq_sym = 0.0
                tot_sym = 0
                nlls: list[float] = []
                failed = False
                for item in cached_items:
                    if item["failed"]:
                        failed = True
                        break
                    fr = item["fr"]
                    s_scale = item["s_scale"]
                    alpha = _scores_to_alpha(
                        item["base_scores"] - float(pen) * item["is_nonzero_om"],
                        float(beta),
                    )
                    out = fuse_candidate_branches(
                        fr,
                        branches=item["built_branches"],
                        branch_weights=alpha,
                        mode="exact_mixture_likelihood",
                        include_constellation_var=bool(cand_cfg.include_constellation_var),
                        n0_scale=float(n0s),
                        config=cand_cfg,
                        return_heldout_cv=False,
                    )
                    bit_err += int(np.sum(out.hard_bits != fr.tx_bits))
                    tot_bits += int(fr.tx_bits.size)
                    sq_sym += float(np.sum(np.abs(out.eq_symbols - fr.tx_symbols) ** 2))
                    tot_sym += int(fr.tx_symbols.size)

                    collected: dict[int, tuple[complex, float]] = {}
                    for h_idx, cv_br, cv_sc, s_scale_cv in item["cv_banks"]:
                        cv_alpha = _scores_to_alpha(cv_sc, float(beta))
                        v_floor = 1e-30 * (s_scale_cv ** 2)
                        h_agg, v_agg = _aggregate_h_only(
                            cv_br, cv_alpha, mode="exact_mixture_likelihood", min_var=v_floor
                        )
                        for pi in h_idx:
                            if pi not in collected:
                                ai = PILOT_ACTIVE_INDICES[pi]
                                collected[pi] = (
                                    complex(h_agg[PILOT_SYMBOL_IDX, ai]),
                                    max(float(v_agg[PILOT_SYMBOL_IDX, ai]), v_floor),
                                )
                    eval_idx = item["all_eval"]
                    p_pred = np.asarray([collected[i][0] for i in eval_idx], dtype=np.complex128)
                    p_var = np.asarray([collected[i][1] for i in eval_idx], dtype=np.float64)
                    v_eff = np.maximum(p_var + item["sig2_obs"], 1e-30)
                    err = p_pred - item["z_obs"]
                    nlls.append(float(np.mean(np.log(np.pi * v_eff) + (np.abs(err) ** 2) / v_eff)))

                if failed:
                    continue
                ber = float(bit_err / max(tot_bits, 1))
                sym_mse = float(sq_sym / max(tot_sym, 1))
                pilot_nll = float(np.mean(nlls)) if nlls else 0.0
                grid_records.append(
                    {
                        "n0_scale": float(n0s),
                        "beta": float(beta),
                        "nonzero_om_penalty": float(pen),
                        "ber": ber,
                        "pilot_nll": pilot_nll,
                        "sym_mse": sym_mse,
                    }
                )
                score = (ber, sym_mse, pilot_nll)
                if score < best_score:
                    best_score = score
                    best_cfg = cand_cfg

    return best_cfg, {
        "best_config": best_cfg.to_dict(),
        "best_ber": best_score[0],
        "best_sym_mse": best_score[1],
        "best_pilot_nll": best_score[2],
        "grid_records": grid_records,
    }


class GroupBFusionReceiver:
    """Convenience stateful wrapper for Group B modes & dev-calibrated GroupBConfig."""

    def __init__(
        self,
        mode: str = "exact_mixture_likelihood",
        config: Optional[GroupBConfig] = None,
    ) -> None:
        self.mode = normalize_group_b_mode(mode)
        self.config = GroupBConfig() if config is None else config

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
            mode=self.mode,
            config=self.config,
            return_heldout_cv=return_heldout_cv,
        )


__all__ = [
    "CandidateBranch",
    "GROUP_B_MODES",
    "GroupBConfig",
    "GroupBFusionReceiver",
    "build_candidate_bank",
    "build_candidate_branches",
    "calibrate_group_b_on_dev",
    "fuse_candidate_branches",
    "normalize_group_b_mode",
    "run_frame",
]
