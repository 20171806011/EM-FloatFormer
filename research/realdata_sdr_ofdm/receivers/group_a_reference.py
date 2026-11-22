"""Group A receivers: reference scope expansion vs. content-matched selection (F8).

Implements a strictly leak-free, C*-equivariant reference-pool framework on top of
the M1 adapter (`RealOFDMFrame`) and M2 classical channel estimator (`classical.py`):
- Disjoint Section 5.1 Eq. (9) pilot partitioning (`KEY_PILOT_INDICES` vs. `VAL_PILOT_INDICES`)
  so key similarity scoring and key-only phase alignment share zero thermal noise with
  reference value estimation.
- Four controlled comparison arms sharing an identical reference pool budget `M`:
  1. ``local`` (alias ``local_ref``): accesses only the first ``min(local_k, M)`` pool
     entries with uniform weights ``1/k`` (remaining ``M-k`` entries receive ``0.0``).
  2. ``global_uniform``: accesses all ``M`` pool entries with uniform weights ``1/M``.
  3. ``global_reliability``: accesses all ``M`` pool entries weighted inversely by
     observable pilot noise variance (never touching ``data_mask`` or labels).
  4. ``global_content_attn`` (aliases ``global_content_match``, ``content_match``):
     accesses all ``M`` pool entries via C*-invariant normalized complex coherence on
     disjoint key pilots with dev-calibrated softmax temperature ``tau``.
- Exact within + between reference Total Variance followed by precision-weighted Bayes
  fusion with the query frame's own 1D LMMSE channel estimate, MMSE equalization, and
  calibrated non-Gray 16-QAM exact LLRs.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np

from research.realdata_sdr_ofdm.adapters.modulation import bits_to_symbols
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
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
    _estimate_timing_slope,
    _exp_pdp_freq_cov,
    _track_phase_2d,
    _validate_pilot_indices,
)

GROUP_A_MODES: tuple[str, ...] = (
    "local",
    "global_uniform",
    "global_reliability",
    "global_content_attn",
)

_MODE_ALIASES: dict[str, str] = {
    "local": "local",
    "local_ref": "local",
    "global_uniform": "global_uniform",
    "global_reliability": "global_reliability",
    "global_content_attn": "global_content_attn",
    "global_content_match": "global_content_attn",
    "content_match": "global_content_attn",
}

KEY_PILOT_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12)
VAL_PILOT_INDICES: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13)


def normalize_group_a_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown Group A mode {mode!r}; expected one of {sorted(_MODE_ALIASES.keys())}"
        )
    return _MODE_ALIASES[key]


@dataclass
class GroupAConfig:
    """Hyperparameters for Group A receivers (calibrated strictly on dev)."""

    temperature: float = 0.10
    local_k: int = 4
    max_pool_size: Optional[int] = 64
    prior_scale: float = 1.0
    n_taps: int = 5
    tau_rms: float = 1.5
    n0_scale: float = 1.0
    diag_reg: float = 1e-4
    detrend_timing: bool = False
    phase_track: bool = False
    reg_floor: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_classical_config(self) -> ClassicalConfig:
        return ClassicalConfig(
            tau_rms=float(self.tau_rms),
            n_taps=int(self.n_taps),
            n0_scale=float(self.n0_scale),
            diag_reg=float(self.diag_reg),
            detrend_timing=bool(self.detrend_timing),
            phase_track=bool(self.phase_track),
        )


def _validate_non_test_frames(frames: Sequence[RealOFDMFrame], caller: str) -> None:
    if not frames:
        raise ValueError(f"{caller} requires a non-empty sequence of frames")
    for fr in frames:
        sp = str(getattr(fr, "split", "")).strip().lower()
        if sp in ("test", "guard"):
            raise ValueError(
                f"Anti-leakage violation: {caller} received frame {fr.frame_id} with split={fr.split!r}"
            )


def build_reference_pool(
    frames: Sequence[RealOFDMFrame],
    *,
    max_pool_size: Optional[int] = 64,
    classical_cfg: Optional[ClassicalConfig] = None,
) -> list[RealOFDMFrame]:
    """Construct a stratified, strictly leak-free reference pool from train/dev frames.

    Raises ValueError if `frames` is empty or contains any `test` or `guard` split frame.
    """
    del classical_cfg  # Pool stores frames cleanly; LMMSE estimation uses active config dynamically.
    _validate_non_test_frames(frames, "build_reference_pool")

    n_total = len(frames)
    if max_pool_size is None or int(max_pool_size) >= n_total:
        return list(frames)

    budget = int(max_pool_size)
    if budget <= 0:
        raise ValueError(f"max_pool_size must be positive when provided, got {max_pool_size!r}")

    # Stratify evenly across group_id preserving deterministic ordering
    groups: dict[str, list[RealOFDMFrame]] = {}
    for fr in frames:
        groups.setdefault(str(fr.group_id), []).append(fr)

    group_keys = list(groups.keys())
    n_groups = len(group_keys)
    base_alloc = budget // n_groups
    rem = budget % n_groups
    selected: list[RealOFDMFrame] = []
    used_ids: set[int] = set()

    # First pass: proportional / balanced stratified pick
    for idx_g, gkey in enumerate(group_keys):
        g_list = groups[gkey]
        take = min(len(g_list), base_alloc + (1 if idx_g < rem else 0))
        if take <= 0:
            continue
        pos_arr = np.linspace(0, len(g_list) - 1, num=take, dtype=int)
        for pos in pos_arr:
            fr_pick = g_list[int(pos)]
            if id(fr_pick) not in used_ids:
                selected.append(fr_pick)
                used_ids.add(id(fr_pick))

    # Top up round-robin if some groups were smaller than quota
    if len(selected) < budget:
        for fr in frames:
            if id(fr) not in used_ids:
                selected.append(fr)
                used_ids.add(id(fr))
                if len(selected) >= budget:
                    break

    return selected[:budget]


def _partition_key_val_pilots(obs_idx: Sequence[int]) -> tuple[list[int], list[int]]:
    """Strictly disjoint Section 5.1 Eq. (9) partition of observed pilots."""
    obs = sorted(set(int(i) for i in obs_idx))
    if len(obs) == 0:
        raise ValueError("Observed pilot set must be non-empty")
    if len(obs) == 1:
        return list(obs), list(obs)
    key_idx = obs[0::2]
    val_idx = obs[1::2]
    return key_idx, val_idx


@lru_cache(maxsize=256)
def _cached_pdp_cov(
    obs_tuple: tuple[int, ...],
    target_tuple: Optional[tuple[int, ...]],
    tau_rms: float,
    n_taps: int,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_obs = np.asarray([PILOT_ACTIVE_INDICES[i] for i in obs_tuple], dtype=np.float64)
    k_target = (
        np.arange(N_ACTIVE, dtype=np.float64)
        if target_tuple is None
        else np.asarray(target_tuple, dtype=np.float64)
    )
    r_pp_unit = _exp_pdp_freq_cov(
        k_obs,
        k_obs,
        tau_rms=float(tau_rms),
        n_taps=int(n_taps),
        centered=bool(centered),
    )
    r_tp_unit = _exp_pdp_freq_cov(
        k_target,
        k_obs,
        tau_rms=float(tau_rms),
        n_taps=int(n_taps),
        centered=bool(centered),
    )
    k_obs.flags.writeable = False
    k_target.flags.writeable = False
    r_pp_unit.flags.writeable = False
    r_tp_unit.flags.writeable = False
    return k_obs, k_target, r_pp_unit, r_tp_unit


def _estimate_from_hp_subset(
    h_p: np.ndarray,
    obs_idx: Sequence[int],
    sigma2_obs: np.ndarray,
    *,
    config: ClassicalConfig,
    target_active_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """1D LMMSE channel estimate from pre-extracted LS pilot vector h_p."""
    obs_arr = np.asarray([int(i) for i in obs_idx], dtype=np.int64)
    if obs_arr.size == 0:
        raise ValueError("obs_idx must be non-empty")
    if obs_arr.size > 1 and np.any(np.diff(obs_arr) <= 0):
        obs_arr, uniq_idx = np.unique(obs_arr, return_index=True)
        h_p = np.asarray(h_p, dtype=np.complex128)[uniq_idx]
        sigma2_obs = np.asarray(sigma2_obs, dtype=np.float64)[uniq_idx]
    obs_tuple = tuple(int(i) for i in obs_arr.tolist())
    target_tuple = (
        None
        if target_active_indices is None
        else tuple(int(x) for x in target_active_indices)
    )
    k_obs, k_target, r_pp_unit, r_tp_unit = _cached_pdp_cov(
        obs_tuple,
        target_tuple,
        float(config.tau_rms),
        int(config.n_taps),
        bool(config.detrend_timing),
    )
    mean_hp2 = float(np.mean(np.abs(h_p) ** 2))
    mean_sig2 = float(np.mean(sigma2_obs))
    sigma_h2 = max(mean_hp2 - mean_sig2, 0.05 * mean_hp2, 1e-6 * max(1.0, mean_sig2))

    if mean_hp2 <= 1e-28:
        h_1d = np.zeros(len(k_target), dtype=np.complex128)
        var_1d = np.full(len(k_target), max(sigma_h2, mean_sig2, 1e-6), dtype=np.float64)
        return h_1d, var_1d, sigma_h2

    omega_f = _estimate_timing_slope(k_obs, h_p) if config.detrend_timing else 0.0
    h_p_detrend = h_p * np.exp(-1j * omega_f * k_obs) if config.detrend_timing else h_p

    r_pp = sigma_h2 * r_pp_unit
    r_tp = sigma_h2 * r_tp_unit
    reg = float(config.diag_reg) * sigma_h2
    a_mat = r_pp + np.diag(sigma2_obs + reg)
    a_mat = 0.5 * (a_mat + a_mat.conj().T)

    rhs = np.column_stack([h_p_detrend, r_tp.conj().T])
    sol = np.linalg.solve(a_mat, rhs)
    alpha_vec = sol[:, 0]
    w_t = sol[:, 1:]

    h_detrend = r_tp @ alpha_vec
    h_1d = (
        (h_detrend * np.exp(1j * omega_f * k_target)) if config.detrend_timing else h_detrend
    ).astype(np.complex128)

    post_red = np.real(np.sum(r_tp * w_t.T, axis=1))
    var_floor = max(1e-4 * mean_sig2, 1e-6 * sigma_h2, 1e-12)
    var_1d = np.maximum(sigma_h2 - post_red, var_floor).astype(np.float64)
    return h_1d, var_1d, sigma_h2


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


def _estimate_group_a_sym2(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    s_scale: float,
    obs_idx: Sequence[int],
    context_pool: Optional[Sequence[RealOFDMFrame]],
    *,
    mode: str,
    local_k: int,
    temperature: float,
    cfg: GroupAConfig,
    cls_cfg: ClassicalConfig,
    target_active_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float], int, list[int], list[int]]:
    """Estimate (14, 102) or target-indexed channel via query LMMSE + Group A pool fusion.

    Strictly reads ONLY `pilot_mask` REs of `context_pool` frames (never `data_mask`,
    `tx_bits`, `tx_symbols`, or `reported_sinr_db`) and enforces disjoint key/value pilots.
    """
    obs_sorted = sorted(set(int(i) for i in obs_idx))
    key_idx, val_idx = _partition_key_val_pilots(obs_sorted)
    h_q, v_q, sigma_h2_q = _estimate_sym2_channel(
        rx_norm,
        pilot_values,
        obs_sorted,
        n0_norm,
        method="lmmse_1d",
        config=cls_cfg,
        target_active_indices=target_active_indices,
    )

    if context_pool is None or len(context_pool) == 0:
        return h_q, v_q, sigma_h2_q, [], 0, key_idx, val_idx

    pool_list = list(context_pool)
    m_pool = len(pool_list)

    # Extract query key pilot LS vector strictly from KEY_PILOT_INDICES
    key_fft = [PILOT_FFT_INDICES[i] for i in key_idx]
    y_q_key = rx_norm[PILOT_SYMBOL_IDX, key_fft]
    pv_q_key = pilot_values[key_idx]
    k_q = y_q_key / pv_q_key
    norm_kq = float(np.linalg.norm(k_q))

    # Extract pool pilot-only observations in query's s_scale gauge (NEVER read data_mask!)
    pilot_fft_all = list(PILOT_FFT_INDICES)
    val_fft = [PILOT_FFT_INDICES[i] for i in val_idx]
    inv_s_scale = 1.0 / s_scale
    inv_s_scale2 = inv_s_scale * inv_s_scale

    k_pool_list: list[np.ndarray] = []
    h_val_p_list: list[np.ndarray] = []
    sig2_val_list: list[np.ndarray] = []
    mean_sig2_obs = np.empty(m_pool, dtype=np.float64)
    inner_vals = np.empty(m_pool, dtype=np.complex128)
    cos_mag = np.empty(m_pool, dtype=np.float64)

    for j, p_fr in enumerate(pool_list):
        p_grid = np.asarray(p_fr.rx_grid, dtype=np.complex128)
        p_pv = np.asarray(p_fr.pilot_values, dtype=np.complex128)
        p_n0 = float(p_fr.est_n0)
        if not (np.all(np.isfinite(p_grid[PILOT_SYMBOL_IDX, pilot_fft_all])) and np.isfinite(p_n0) and p_n0 >= 0.0):
            raise ValueError(f"Pool frame {getattr(p_fr, 'frame_id', j)} has invalid pilot/n0 observations")
        if p_pv.shape != (len(PILOT_FFT_INDICES),) or not np.all(np.isfinite(p_pv)) or not np.all(np.abs(p_pv) > 0.0):
            raise ValueError(f"Pool frame {getattr(p_fr, 'frame_id', j)} has invalid pilot_values")

        y_j_key = p_grid[PILOT_SYMBOL_IDX, key_fft] * inv_s_scale
        k_j = y_j_key / p_pv[key_idx]
        k_pool_list.append(k_j)

        y_j_val = p_grid[PILOT_SYMBOL_IDX, val_fft] * inv_s_scale
        h_j_val_p = y_j_val / p_pv[val_idx]
        h_val_p_list.append(h_j_val_p)

        n0_j_eff = max(float(cfg.n0_scale) * (p_n0 * inv_s_scale2), 0.0)
        sig2_all_j = n0_j_eff / np.maximum(np.abs(p_pv[obs_sorted]) ** 2, 1e-30)
        mean_sig2_obs[j] = float(np.mean(sig2_all_j))
        sig2_val_list.append(n0_j_eff / np.maximum(np.abs(p_pv[val_idx]) ** 2, 1e-30))

        # C*-invariant cosine magnitude & key-only complex inner product
        inner_j = complex(np.vdot(k_j, k_q))
        inner_vals[j] = inner_j
        norm_kj = float(np.linalg.norm(k_j))
        denom_norm = norm_kj * norm_kq
        if denom_norm > 1e-30:
            cos_mag[j] = float(min(abs(inner_j) / denom_norm, 1.0))
        else:
            cos_mag[j] = 0.0

    # Compute normalized pool weights of exact length M
    if mode == "local":
        k_eff = min(max(int(local_k), 1), m_pool)
        w = np.zeros(m_pool, dtype=np.float64)
        w[:k_eff] = 1.0 / float(k_eff)
        n_values_accessed = int(k_eff)
    elif mode == "global_uniform":
        w = np.full(m_pool, 1.0 / float(m_pool), dtype=np.float64)
        n_values_accessed = int(m_pool)
    elif mode == "global_reliability":
        # Weight inversely by observable pilot noise variance (never touching data_mask)
        prec = 1.0 / np.maximum(mean_sig2_obs, float(cfg.reg_floor))
        prec_sum = float(np.sum(prec))
        if not np.isfinite(prec_sum) or prec_sum <= 0.0:
            w = np.full(m_pool, 1.0 / float(m_pool), dtype=np.float64)
        else:
            w = prec / prec_sum
        n_values_accessed = int(m_pool)
    else:  # global_content_attn
        tau = float(temperature)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
        scale_d = math.sqrt(float(max(len(key_idx), 1)))
        logits = (scale_d * cos_mag) / tau
        logits = logits - float(np.max(logits))
        logits = np.clip(logits, -700.0, 0.0)
        exp_l = np.exp(logits)
        exp_sum = float(np.sum(exp_l))
        if not np.isfinite(exp_sum) or exp_sum <= 0.0:
            w = np.full(m_pool, 1.0 / float(m_pool), dtype=np.float64)
        else:
            w = exp_l / exp_sum
        n_values_accessed = int(m_pool)

    # Estimate aligned value channels ONLY for entries with positive weight w[j] > 0
    k_target_len = N_ACTIVE if target_active_indices is None else len(target_active_indices)
    aligned_h = np.zeros((m_pool, k_target_len), dtype=np.complex128)
    val_v = np.zeros((m_pool, k_target_len), dtype=np.float64)
    active_j = [j for j in range(m_pool) if float(w[j]) > 0.0]

    for j in active_j:
        h_val_j, v_val_j, _ = _estimate_from_hp_subset(
            h_val_p_list[j],
            val_idx,
            sig2_val_list[j],
            config=cls_cfg,
            target_active_indices=target_active_indices,
        )
        inner_j = inner_vals[j]
        mag_in = abs(inner_j)
        u_j = (inner_j / mag_in) if mag_in > 1e-30 else (1.0 + 0.0j)
        aligned_h[j] = u_j * h_val_j
        val_v[j] = v_val_j

    # Aggregate weighted posterior mean & exact Total Variance (within + between)
    w_col = w[:, None]
    h_ref = np.sum(w_col * aligned_h, axis=0)
    v_within = np.sum(w_col * val_v, axis=0)
    v_between = np.sum(w_col * (np.abs(aligned_h - h_ref[None, :]) ** 2), axis=0)
    v_ref = np.maximum(float(cfg.prior_scale) * (v_within + v_between), 1e-30)

    # Precision-weighted Bayes fusion with query frame's own 1D LMMSE estimate
    inv_vq = 1.0 / np.maximum(v_q, 1e-30)
    inv_vref = 1.0 / v_ref
    v_post = np.maximum(1.0 / (inv_vq + inv_vref), 1e-30).astype(np.float64)
    h_post = (v_post * (h_q * inv_vq + h_ref * inv_vref)).astype(np.complex128)

    return h_post, v_post, sigma_h2_q, [float(x) for x in w.tolist()], n_values_accessed, key_idx, val_idx


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


def _compute_heldout_cv_group_a(
    frame: RealOFDMFrame,
    rx_grid: np.ndarray,
    pilot_values: np.ndarray,
    est_n0: float,
    context_pool: Optional[Sequence[RealOFDMFrame]],
    *,
    mode: str,
    local_k: int,
    temperature: float,
    cfg: GroupAConfig,
    cls_cfg: ClassicalConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Leak-free held-out pilot CV masking eval_idx from both query LMMSE and key matching."""
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
        target_active = np.asarray([PILOT_ACTIVE_INDICES[i] for i in eval_idx], dtype=np.int64)
        h_pred_norm, h_var_norm, _, _, _, _, _ = _estimate_group_a_sym2(
            rx_norm_cv,
            pilot_values,
            n0_norm_cv,
            s_scale_cv,
            obs_idx,
            context_pool,
            mode=mode,
            local_k=local_k,
            temperature=temperature,
            cfg=cfg,
            cls_cfg=cls_cfg,
            target_active_indices=target_active,
        )
        pred = (h_pred_norm * s_scale_cv).astype(np.complex128)
        v_floor_cv = max(1e-30 * (s_scale_cv ** 2), 1e-300)
        var = np.maximum(h_var_norm * (s_scale_cv ** 2), v_floor_cv).astype(np.float64)
        return pred, var, eval_idx

    folds = frame.meta.get("heldout_pilot_folds") if isinstance(frame.meta, dict) else None
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
        target_active = np.asarray([PILOT_ACTIVE_INDICES[i] for i in h_idx], dtype=np.int64)
        h_pred_norm, h_var_norm, _, _, _, _, _ = _estimate_group_a_sym2(
            rx_norm_cv,
            pilot_values,
            n0_norm_cv,
            s_scale_cv,
            o_idx,
            context_pool,
            mode=mode,
            local_k=local_k,
            temperature=temperature,
            cfg=cfg,
            cls_cfg=cls_cfg,
            target_active_indices=target_active,
        )
        v_floor_cv = max(1e-30 * (s_scale_cv ** 2), 1e-300)
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


def _failed_output(
    err_msg: str,
    mode: str,
    latency_ms: float,
    pool_len: int = 0,
) -> ReceiverOutput:
    bit_llrs = np.zeros((N_DATA_RES, QM), dtype=np.float64)
    hard_bits = (bit_llrs > 0.0).astype(np.uint8)
    hard_symbols = bits_to_symbols(hard_bits, qm=QM)
    weights = [1.0 / float(pool_len)] * pool_len if pool_len > 0 else []
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
            "weights": weights,
            "branch_weights": list(weights),
            "n_values_accessed": 0,
            "n_candidates": max(pool_len, 1),
            "latency_ms": float(latency_ms),
            "failed": True,
            "error": str(err_msg),
            "param_count": 0,
            "heldout_pilot_indices": [],
        },
    )


def run_frame(
    frame: RealOFDMFrame,
    context_pool: Optional[Sequence[RealOFDMFrame]] = None,
    *,
    mode: str = "global_content_attn",
    local_k: Optional[int] = None,
    temperature: Optional[float] = None,
    config: Optional[GroupAConfig] = None,
    return_heldout_cv: bool = True,
    **kwargs: Any,
) -> ReceiverOutput:
    """Run Group A reference-pool receiver on one RealOFDMFrame (zero label access)."""
    del kwargs  # Reserved for forward compatibility
    t0 = time.perf_counter()
    canon_mode = normalize_group_a_mode(mode)
    cfg = GroupAConfig() if config is None else config
    pool_len = len(context_pool) if context_pool is not None else 0

    try:
        raw_local_k = cfg.local_k if local_k is None else local_k
        if isinstance(raw_local_k, (bool, np.bool_)) or not isinstance(raw_local_k, (int, np.integer)):
            raise ValueError(f"local_k must be a positive integer, got {raw_local_k!r}")
        eff_local_k = int(raw_local_k)
        if eff_local_k < 1:
            raise ValueError(f"local_k must be >= 1, got {eff_local_k!r}")

        raw_temp = cfg.temperature if temperature is None else temperature
        if isinstance(raw_temp, (bool, np.bool_)):
            raise ValueError(f"temperature must be finite and > 0, got {raw_temp!r}")
        eff_temp = float(raw_temp)
        if not np.isfinite(eff_temp) or eff_temp <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {eff_temp!r}")

        if isinstance(cfg.prior_scale, (bool, np.bool_)):
            raise ValueError(f"prior_scale must be finite and > 0, got {cfg.prior_scale!r}")
        prior_scale = float(cfg.prior_scale)
        if not np.isfinite(prior_scale) or prior_scale <= 0.0:
            raise ValueError(f"prior_scale must be finite and > 0, got {cfg.prior_scale!r}")

        if isinstance(cfg.reg_floor, (bool, np.bool_)):
            raise ValueError(f"reg_floor must be finite and > 0, got {cfg.reg_floor!r}")
        reg_floor = float(cfg.reg_floor)
        if not np.isfinite(reg_floor) or reg_floor <= 0.0:
            raise ValueError(f"reg_floor must be finite and > 0, got {cfg.reg_floor!r}")

        if context_pool is not None and len(context_pool) > 0:
            _validate_non_test_frames(context_pool, "group_a_reference.run_frame(context_pool)")

        rx_grid, est_n0, pilot_values = _validate_frame_inputs(frame)
        active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
        s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
        if not np.isfinite(s_rms):
            raise ValueError("Non-finite active RMS power")
        s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)

        rx_norm = rx_grid / s_scale
        n0_norm = est_n0 / (s_scale ** 2)
        cls_cfg = cfg.to_classical_config()
        data_mask_active = np.asarray(frame.data_mask[:, 13:115], dtype=bool)
        all_pilots = list(range(14))

        h_1d_norm, v_1d_norm, _, weights, n_accessed, key_idx, val_idx = _estimate_group_a_sym2(
            rx_norm,
            pilot_values,
            n0_norm,
            s_scale,
            all_pilots,
            context_pool,
            mode=canon_mode,
            local_k=eff_local_k,
            temperature=eff_temp,
            cfg=cfg,
            cls_cfg=cls_cfg,
        )

        h_2d_norm = np.broadcast_to(h_1d_norm[None, :], (N_SYMB, N_ACTIVE)).copy()
        v_2d_norm = np.broadcast_to(v_1d_norm[None, :], (N_SYMB, N_ACTIVE)).copy()
        omega_t = 0.0
        if cfg.phase_track:
            rx_active = rx_norm[:, 13:115]
            h_2d_norm, omega_t = _track_phase_2d(
                rx_active,
                data_mask_active,
                h_2d_norm,
                v_2d_norm,
                n0_norm,
                config=cls_cfg,
            )

        rx_active = rx_norm[:, 13:115]
        y_data = rx_active[data_mask_active]
        h_data = h_2d_norm[data_mask_active]
        v_data = v_2d_norm[data_mask_active]

        eq_symbols, bit_llrs, hard_bits, hard_symbols = _equalize_and_demap(
            y_data,
            h_data,
            v_data,
            n0_norm,
            method="lmmse_1d",
            config=cls_cfg,
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
            heldout_pred, heldout_var, heldout_indices = _compute_heldout_cv_group_a(
                frame,
                rx_grid,
                pilot_values,
                est_n0,
                context_pool,
                mode=canon_mode,
                local_k=eff_local_k,
                temperature=eff_temp,
                cfg=cfg,
                cls_cfg=cls_cfg,
            )
            if not (np.all(np.isfinite(heldout_pred)) and np.all(np.isfinite(heldout_var))):
                raise ValueError("Held-out pilot predictions contain non-finite (NaN/Inf) values")

        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        aux_info: dict[str, Any] = {
            "mode": canon_mode,
            "weights": weights,
            "branch_weights": list(weights),
            "n_values_accessed": int(n_accessed),
            "n_candidates": max(pool_len, 1),
            "key_pilot_indices": list(key_idx),
            "val_pilot_indices": list(val_idx),
            "temperature": float(eff_temp),
            "local_k": int(eff_local_k),
            "omega_t_rad_per_sym": float(omega_t),
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
    except Exception as exc:  # noqa: BLE001
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        return _failed_output(f"{type(exc).__name__}: {exc}", canon_mode, latency_ms, pool_len=pool_len)


def _resolve_dev_pools(
    dev_frames: Sequence[RealOFDMFrame],
    pools: Sequence[RealOFDMFrame] | Sequence[Sequence[RealOFDMFrame]],
) -> list[list[RealOFDMFrame]]:
    """Validate pools (single pool or per-frame pools) and enforce anti-leakage."""
    if not pools:
        raise ValueError("pools must be non-empty")

    first = pools[0]
    if isinstance(first, RealOFDMFrame):
        flat_pool = list(pools)  # type: ignore[arg-type]
        _validate_non_test_frames(flat_pool, "calibrate_group_a_temperature(pools)")
        resolved: list[list[RealOFDMFrame]] = []
        for fr in dev_frames:
            sub = [p for p in flat_pool if p is not fr]
            resolved.append(sub if sub else flat_pool)
        return resolved

    if len(pools) != len(dev_frames):
        raise ValueError(
            f"Per-frame pools length {len(pools)} != dev_frames length {len(dev_frames)}"
        )
    per_frame: list[list[RealOFDMFrame]] = []
    for idx, p_seq in enumerate(pools):
        p_list = list(p_seq)  # type: ignore[arg-type]
        _validate_non_test_frames(p_list, f"calibrate_group_a_temperature(pools[{idx}])")
        fr = dev_frames[idx]
        sub = [p for p in p_list if p is not fr]
        per_frame.append(sub if sub else p_list)
    return per_frame


def _eval_group_a_dev(
    dev_frames: Sequence[RealOFDMFrame],
    per_frame_pools: Sequence[Sequence[RealOFDMFrame]],
    *,
    mode: str,
    cfg: GroupAConfig,
) -> tuple[float, float, float]:
    bit_err = 0
    tot_bits = 0
    sq_sym = 0.0
    tot_sym = 0
    nlls: list[float] = []
    for fr, pool in zip(dev_frames, per_frame_pools):
        out = run_frame(fr, context_pool=pool, mode=mode, config=cfg, return_heldout_cv=True)
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


def calibrate_group_a_temperature(
    dev_frames: Sequence[RealOFDMFrame],
    pools: Sequence[RealOFDMFrame] | Sequence[Sequence[RealOFDMFrame]],
    *,
    temps: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.50, 1.0, 2.0),
    config: Optional[GroupAConfig] = None,
) -> float:
    """Calibrate softmax temperature strictly on dev frames (zero test/guard leakage)."""
    _validate_non_test_frames(dev_frames, "calibrate_group_a_temperature")
    per_frame_pools = _resolve_dev_pools(dev_frames, pools)
    base_cfg = GroupAConfig() if config is None else config

    best_score = (float("inf"), float("inf"), float("inf"))
    best_tau = float(base_cfg.temperature)
    for tau in temps:
        t_val = float(tau)
        if not np.isfinite(t_val) or t_val <= 0.0:
            raise ValueError(f"Candidate temperature must be positive and finite, got {tau!r}")
        cand_cfg = GroupAConfig(**{**base_cfg.to_dict(), "temperature": t_val})
        ber, nll, smse = _eval_group_a_dev(
            dev_frames,
            per_frame_pools,
            mode="global_content_attn",
            cfg=cand_cfg,
        )
        score = (ber, nll, smse)
        if score < best_score:
            best_score = score
            best_tau = t_val
    return float(best_tau)


def _prepare_sym2_components(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    s_scale: float,
    obs_idx: Sequence[int],
    context_pool: Optional[Sequence[RealOFDMFrame]],
    *,
    cfg: GroupAConfig,
    cls_cfg: ClassicalConfig,
    target_active_indices: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Precompute query & pool pilot-only components independent of (temperature, prior_scale)."""
    obs_sorted = sorted(set(int(i) for i in obs_idx))
    key_idx, val_idx = _partition_key_val_pilots(obs_sorted)
    h_q, v_q, sigma_h2_q = _estimate_sym2_channel(
        rx_norm,
        pilot_values,
        obs_sorted,
        n0_norm,
        method="lmmse_1d",
        config=cls_cfg,
        target_active_indices=target_active_indices,
    )
    if context_pool is None or len(context_pool) == 0:
        return {
            "empty": True,
            "h_q": h_q,
            "v_q": v_q,
            "sigma_h2_q": sigma_h2_q,
            "key_idx": key_idx,
            "val_idx": val_idx,
        }

    pool_list = list(context_pool)
    m_pool = len(pool_list)
    key_fft = [PILOT_FFT_INDICES[i] for i in key_idx]
    y_q_key = rx_norm[PILOT_SYMBOL_IDX, key_fft]
    pv_q_key = pilot_values[key_idx]
    k_q = y_q_key / pv_q_key
    norm_kq = float(np.linalg.norm(k_q))

    pilot_fft_all = list(PILOT_FFT_INDICES)
    val_fft = [PILOT_FFT_INDICES[i] for i in val_idx]
    inv_s_scale = 1.0 / s_scale
    inv_s_scale2 = inv_s_scale * inv_s_scale

    k_target_len = N_ACTIVE if target_active_indices is None else len(target_active_indices)
    aligned_h = np.zeros((m_pool, k_target_len), dtype=np.complex128)
    val_v = np.zeros((m_pool, k_target_len), dtype=np.float64)
    cos_mag = np.empty(m_pool, dtype=np.float64)

    for j, p_fr in enumerate(pool_list):
        p_grid = np.asarray(p_fr.rx_grid, dtype=np.complex128)
        p_pv = np.asarray(p_fr.pilot_values, dtype=np.complex128)
        p_n0 = float(p_fr.est_n0)
        if not (np.all(np.isfinite(p_grid[PILOT_SYMBOL_IDX, pilot_fft_all])) and np.isfinite(p_n0) and p_n0 >= 0.0):
            raise ValueError(f"Pool frame {getattr(p_fr, 'frame_id', j)} has invalid pilot/n0 observations")
        if p_pv.shape != (len(PILOT_FFT_INDICES),) or not np.all(np.isfinite(p_pv)) or not np.all(np.abs(p_pv) > 0.0):
            raise ValueError(f"Pool frame {getattr(p_fr, 'frame_id', j)} has invalid pilot_values")

        y_j_key = p_grid[PILOT_SYMBOL_IDX, key_fft] * inv_s_scale
        k_j = y_j_key / p_pv[key_idx]
        y_j_val = p_grid[PILOT_SYMBOL_IDX, val_fft] * inv_s_scale
        h_j_val_p = y_j_val / p_pv[val_idx]

        n0_j_eff = max(float(cfg.n0_scale) * (p_n0 * inv_s_scale2), 0.0)
        sig2_val_j = n0_j_eff / np.maximum(np.abs(p_pv[val_idx]) ** 2, 1e-30)

        inner_j = complex(np.vdot(k_j, k_q))
        norm_kj = float(np.linalg.norm(k_j))
        denom_norm = norm_kj * norm_kq
        cos_mag[j] = float(min(abs(inner_j) / denom_norm, 1.0)) if denom_norm > 1e-30 else 0.0

        h_val_j, v_val_j, _ = _estimate_from_hp_subset(
            h_j_val_p,
            val_idx,
            sig2_val_j,
            config=cls_cfg,
            target_active_indices=target_active_indices,
        )
        mag_in = abs(inner_j)
        u_j = (inner_j / mag_in) if mag_in > 1e-30 else (1.0 + 0.0j)
        aligned_h[j] = u_j * h_val_j
        val_v[j] = v_val_j

    return {
        "empty": False,
        "m_pool": m_pool,
        "h_q": h_q,
        "v_q": v_q,
        "sigma_h2_q": sigma_h2_q,
        "key_idx": key_idx,
        "val_idx": val_idx,
        "cos_mag": cos_mag,
        "aligned_h": aligned_h,
        "val_v": val_v,
    }


def _fuse_sym2_content_attn(
    comp: dict[str, Any],
    *,
    temperature: float,
    prior_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if comp["empty"]:
        return comp["h_q"], comp["v_q"]
    m_pool = int(comp["m_pool"])
    scale_d = math.sqrt(float(max(len(comp["key_idx"]), 1)))
    logits = (scale_d * comp["cos_mag"]) / float(temperature)
    logits = logits - float(np.max(logits))
    logits = np.clip(logits, -700.0, 0.0)
    exp_l = np.exp(logits)
    exp_sum = float(np.sum(exp_l))
    w = np.full(m_pool, 1.0 / float(m_pool), dtype=np.float64) if (not np.isfinite(exp_sum) or exp_sum <= 0.0) else (exp_l / exp_sum)

    w_col = w[:, None]
    aligned_h = comp["aligned_h"]
    val_v = comp["val_v"]
    h_ref = np.sum(w_col * aligned_h, axis=0)
    v_within = np.sum(w_col * val_v, axis=0)
    v_between = np.sum(w_col * (np.abs(aligned_h - h_ref[None, :]) ** 2), axis=0)
    v_ref = np.maximum(float(prior_scale) * (v_within + v_between), 1e-30)

    inv_vq = 1.0 / np.maximum(comp["v_q"], 1e-30)
    inv_vref = 1.0 / v_ref
    v_post = np.maximum(1.0 / (inv_vq + inv_vref), 1e-30).astype(np.float64)
    h_post = (v_post * (comp["h_q"] * inv_vq + h_ref * inv_vref)).astype(np.complex128)
    return h_post, v_post


def calibrate_group_a_on_dev(
    dev_frames: Sequence[RealOFDMFrame],
    context_pool: Sequence[RealOFDMFrame],
    *,
    temps: Sequence[float] = (0.05, 0.15, 0.30, 0.50, 1.0, 2.0, 5.0),
    prior_scales: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    base_config: Optional[GroupAConfig] = None,
) -> tuple[GroupAConfig, dict[str, Any]]:
    """Calibrate Group A prior_scale & temperature strictly on dev frames."""
    _validate_non_test_frames(dev_frames, "calibrate_group_a_on_dev")
    per_frame_pools = _resolve_dev_pools(dev_frames, context_pool)
    base = GroupAConfig() if base_config is None else base_config
    cls_cfg = base.to_classical_config()

    # Precompute invariant query/pool LMMSE components once per dev frame when phase_track=False
    precomputed: list[dict[str, Any]] = []
    can_fast_path = not bool(base.phase_track)
    if can_fast_path:
        for fr, pool in zip(dev_frames, per_frame_pools):
            try:
                rx_grid, est_n0, pilot_values = _validate_frame_inputs(fr)
                active_mask = np.asarray(fr.pilot_mask | fr.data_mask, dtype=bool)
                s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
                if not np.isfinite(s_rms):
                    raise ValueError("Non-finite active RMS power")
                s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
                rx_norm = rx_grid / s_scale
                n0_norm = est_n0 / (s_scale ** 2)
                data_mask_active = np.asarray(fr.data_mask[:, 13:115], dtype=bool)
                rx_active = rx_norm[:, 13:115]
                y_data = rx_active[data_mask_active]

                main_comp = _prepare_sym2_components(
                    rx_norm,
                    pilot_values,
                    n0_norm,
                    s_scale,
                    list(range(14)),
                    pool,
                    cfg=base,
                    cls_cfg=cls_cfg,
                )

                explicit_idx = fr.meta.get("heldout_pilot_indices") if isinstance(fr.meta, dict) else None
                cv_folds: list[tuple[list[int], float, dict[str, Any]]] = []
                if explicit_idx is not None and len(explicit_idx) > 0:
                    eval_idx = _validate_pilot_indices(explicit_idx, "heldout_pilot_indices")
                    eval_set = set(eval_idx)
                    obs_idx = [i for i in range(14) if i not in eval_set]
                    s_scale_cv = _compute_cv_fold_scale(
                        rx_grid, fr.pilot_mask, fr.data_mask, est_n0, eval_idx
                    )
                    rx_norm_cv = rx_grid / s_scale_cv
                    n0_norm_cv = est_n0 / (s_scale_cv ** 2)
                    t_act = np.asarray([PILOT_ACTIVE_INDICES[i] for i in eval_idx], dtype=np.int64)
                    cv_folds.append(
                        (
                            eval_idx,
                            s_scale_cv,
                            _prepare_sym2_components(
                                rx_norm_cv,
                                pilot_values,
                                n0_norm_cv,
                                s_scale_cv,
                                obs_idx,
                                pool,
                                cfg=base,
                                cls_cfg=cls_cfg,
                                target_active_indices=t_act,
                            ),
                        )
                    )
                else:
                    folds = fr.meta.get("heldout_pilot_folds") if isinstance(fr.meta, dict) else None
                    if not folds:
                        folds = build_heldout_pilot_masks(4)
                    for fold in folds:
                        h_idx = _validate_pilot_indices(fold["heldout_pilot_indices"], "heldout_pilot_indices")
                        if not h_idx:
                            raise ValueError("heldout_pilot_indices in fold must be non-empty")
                        h_set = set(h_idx)
                        o_raw = _validate_pilot_indices(fold["obs_pilot_indices"], "obs_pilot_indices")
                        o_idx = sorted(i for i in o_raw if i not in h_set)
                        s_scale_cv = _compute_cv_fold_scale(
                            rx_grid, fr.pilot_mask, fr.data_mask, est_n0, h_idx
                        )
                        rx_norm_cv = rx_grid / s_scale_cv
                        n0_norm_cv = est_n0 / (s_scale_cv ** 2)
                        t_act = np.asarray([PILOT_ACTIVE_INDICES[i] for i in h_idx], dtype=np.int64)
                        cv_folds.append(
                            (
                                h_idx,
                                s_scale_cv,
                                _prepare_sym2_components(
                                    rx_norm_cv,
                                    pilot_values,
                                    n0_norm_cv,
                                    s_scale_cv,
                                    o_idx,
                                    pool,
                                    cfg=base,
                                    cls_cfg=cls_cfg,
                                    target_active_indices=t_act,
                                ),
                            )
                        )

                # Precompute heldout pilot targets
                all_eval = sorted({pi for h_idx, _, _ in cv_folds for pi in h_idx})
                fft_idx = [PILOT_FFT_INDICES[i] for i in all_eval]
                y_p = fr.rx_grid[PILOT_SYMBOL_IDX, fft_idx]
                p_val = fr.pilot_values[all_eval]
                z_obs = y_p / p_val
                sig2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)

                precomputed.append(
                    {
                        "failed": False,
                        "fr": fr,
                        "s_scale": s_scale,
                        "n0_norm": n0_norm,
                        "data_mask_active": data_mask_active,
                        "y_data": y_data,
                        "main_comp": main_comp,
                        "cv_folds": cv_folds,
                        "all_eval": all_eval,
                        "z_obs": z_obs,
                        "sig2_obs": sig2_obs,
                    }
                )
            except Exception:  # noqa: BLE001
                precomputed.append({"failed": True})

    best_score = (float("inf"), float("inf"), float("inf"))
    best_cfg = base
    grid_records: list[dict[str, Any]] = []

    for ps in prior_scales:
        for tau in temps:
            t_val = float(tau)
            ps_val = float(ps)
            if not np.isfinite(t_val) or t_val <= 0.0:
                raise ValueError(f"Candidate temperature must be positive and finite, got {tau!r}")
            if not np.isfinite(ps_val) or ps_val <= 0.0:
                raise ValueError(f"Candidate prior_scale must be positive and finite, got {ps!r}")
            cand_cfg = GroupAConfig(
                **{**base.to_dict(), "prior_scale": ps_val, "temperature": t_val}
            )
            if not can_fast_path:
                ber, nll, smse = _eval_group_a_dev(
                    dev_frames,
                    per_frame_pools,
                    mode="global_content_attn",
                    cfg=cand_cfg,
                )
            else:
                bit_err = 0
                tot_bits = 0
                sq_sym = 0.0
                tot_sym = 0
                nlls: list[float] = []
                any_fail = False
                for item in precomputed:
                    if item["failed"]:
                        any_fail = True
                        break
                    fr = item["fr"]
                    s_scale = item["s_scale"]
                    h_1d_norm, v_1d_norm = _fuse_sym2_content_attn(
                        item["main_comp"], temperature=t_val, prior_scale=ps_val
                    )
                    h_2d_norm = np.broadcast_to(h_1d_norm[None, :], (N_SYMB, N_ACTIVE))
                    v_2d_norm = np.broadcast_to(v_1d_norm[None, :], (N_SYMB, N_ACTIVE))
                    dm = item["data_mask_active"]
                    eq_symbols, _, hard_bits, _ = _equalize_and_demap(
                        item["y_data"],
                        h_2d_norm[dm],
                        v_2d_norm[dm],
                        item["n0_norm"],
                        method="lmmse_1d",
                        config=cls_cfg,
                    )
                    bit_err += int(np.sum(hard_bits != fr.tx_bits))
                    tot_bits += int(fr.tx_bits.size)
                    sq_sym += float(np.sum(np.abs(eq_symbols - fr.tx_symbols) ** 2))
                    tot_sym += int(fr.tx_symbols.size)

                    collected: dict[int, tuple[complex, float]] = {}
                    for h_idx, s_scale_cv, fold_comp in item["cv_folds"]:
                        hp_n, hv_n = _fuse_sym2_content_attn(
                            fold_comp, temperature=t_val, prior_scale=ps_val
                        )
                        v_floor_cv = max(1e-30 * (s_scale_cv ** 2), 1e-300)
                        for pos, pi in enumerate(h_idx):
                            if pi not in collected:
                                collected[pi] = (
                                    complex(hp_n[pos] * s_scale_cv),
                                    max(float(hv_n[pos] * (s_scale_cv ** 2)), v_floor_cv),
                                )
                    eval_idx = item["all_eval"]
                    p_pred = np.asarray([collected[i][0] for i in eval_idx], dtype=np.complex128)
                    p_var = np.asarray([collected[i][1] for i in eval_idx], dtype=np.float64)
                    v_eff = np.maximum(p_var + item["sig2_obs"], max(1e-30 * (s_scale ** 2), 1e-300))
                    err = p_pred - item["z_obs"]
                    nlls.append(float(np.mean(np.log(np.pi * v_eff) + (np.abs(err) ** 2) / v_eff)))

                if any_fail:
                    ber, nll, smse = 1.0, 1e9, 1e9
                else:
                    ber = float(bit_err / max(tot_bits, 1))
                    smse = float(sq_sym / max(tot_sym, 1))
                    nll = float(np.mean(nlls)) if nlls else 0.0

            grid_records.append(
                {
                    "prior_scale": ps_val,
                    "temperature": t_val,
                    "ber": ber,
                    "pilot_nll": nll,
                    "sym_mse": smse,
                }
            )
            score = (ber, nll, smse)
            if score < best_score:
                best_score = score
                best_cfg = cand_cfg

    return best_cfg, {
        "best_config": best_cfg.to_dict(),
        "best_ber": best_score[0],
        "best_pilot_nll": best_score[1],
        "best_sym_mse": best_score[2],
        "grid_records": grid_records,
    }


calibrate_temperature = calibrate_group_a_temperature
fit_temperature = calibrate_group_a_temperature


class GroupAReferenceReceiver:
    """Convenience stateful wrapper for Group A modes & dev-calibrated GroupAConfig."""

    def __init__(
        self,
        mode: str = "global_content_attn",
        config: Optional[GroupAConfig] = None,
    ) -> None:
        self.mode = normalize_group_a_mode(mode)
        self.config = GroupAConfig() if config is None else config

    def run_frame(
        self,
        frame: RealOFDMFrame,
        context_pool: Optional[Sequence[RealOFDMFrame]] = None,
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
    "GROUP_A_MODES",
    "GroupAConfig",
    "GroupAReferenceReceiver",
    "KEY_PILOT_INDICES",
    "VAL_PILOT_INDICES",
    "build_reference_pool",
    "calibrate_group_a_on_dev",
    "calibrate_group_a_temperature",
    "calibrate_temperature",
    "fit_temperature",
    "normalize_group_a_mode",
    "run_frame",
]
