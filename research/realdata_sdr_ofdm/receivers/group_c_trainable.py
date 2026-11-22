"""Group C receiver: lightweight C*-equivariant branch-weight attention module (F10 / R3-C).

Reuses EM-FloatFormer PyTorch primitives from ``p4.em_floatformer``
(``ComplexLinear(bias=False)``, ``ComplexRMSNorm``, ``RadialActivation``,
``hermitian_attention_scores``) and ``p4.slot_2d`` (``ComplexRoPE2D``), plus
``ManifoldTrustGate``-style frozen ``initial_map`` step-0 centering and
Hermitian/quadratic C*-invariant readout (without importing ``p9``/``p11``).

Per R3-C and ``PROTOCOL.md`` §4:
- Candidate generation and branch estimators are kept fixed via Group B's
  ``group_b_fusion._build_bank_normalized`` (with optional fixed Group A
  ``global_content_attn`` branch built strictly from ``train`` pilot-only pool).
- Backend decoding is kept fixed via Group B's ``exact_mixture_likelihood``
  demapper (``fuse_candidate_branches``).
- Only the lightweight (~2.4k real params, float64/complex128, zero BatchNorm)
  C*-equivariant attention network ``BranchWeightAttentionNet`` predicting
  simplex branch weights ``alpha_r`` is trained strictly on ``train``/``dev``.
"""

from __future__ import annotations

import copy
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from p4.em_floatformer import (
    ComplexLinear,
    ComplexRMSNorm,
    RadialActivation,
    hermitian_attention_scores,
)
from p4.slot_2d import ComplexRoPE2D
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
    _estimate_sym2_channel,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    GroupAConfig,
    _estimate_from_hp_subset,
    _partition_key_val_pilots,
    _validate_non_test_frames,
    build_reference_pool,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    CandidateBranch,
    GroupBConfig,
    _aggregate_h_only,
    _build_bank_normalized,
    _cv_fold_normalized_inputs,
    _extract_active_slice,
    _scores_to_alpha,
    _validate_frame_inputs,
    fuse_candidate_branches,
)

GROUP_C_MODES: tuple[str, ...] = (
    "group_c_branch_attn",
    "group_c_hybrid_attn",
    "group_c_pilot_nll_only",
)

N_SCALAR_FEATS = 7
N_COMPLEX_FEATS = 3


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def _default_group_b_config() -> GroupBConfig:
    """Dev-calibrated Group B configuration (beta=0.25, nonzero_om_penalty=8.0, n0_scale=0.7)."""
    return GroupBConfig(
        pdp_hypotheses=((7, 0.8), (11, 1.0), (11, 2.0)),
        n0_scale=0.7,
        diag_reg=1e-4,
        detrend_timing=True,
        phase_track_max_slope=0.50,
        phase_track_prior_weight=2.0,
        beta=0.25,
        nonzero_om_penalty=8.0,
        omega_min_diff=1e-3,
        include_constellation_var=False,
    )


def _default_group_a_config() -> GroupAConfig:
    """Dev-calibrated Group A configuration (temperature=0.05, prior_scale=0.5, n0_scale=0.7)."""
    return GroupAConfig(
        temperature=0.05,
        local_k=4,
        max_pool_size=64,
        prior_scale=0.5,
        n_taps=11,
        tau_rms=1.0,
        n0_scale=0.7,
        diag_reg=1e-4,
        detrend_timing=True,
        phase_track=False,
        reg_floor=1e-8,
    )


@dataclass
class GroupCConfig:
    """Configuration for Group C lightweight C*-equivariant branch-weight attention."""

    d_c: int = 8
    d_s: int = 12
    n_heads: int = 2
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 16
    bce_weight: float = 1.0
    pilot_nll_weight: float = 0.05
    prior_reg_weight: float = 1e-3
    include_global_branch: bool = True
    max_pool_size: int = 64
    n0_scale: float = 0.7
    architecture: str = "CStarBranchWeightAttentionNet"
    group_b_cfg: GroupBConfig = field(default_factory=_default_group_b_config)
    group_a_cfg: GroupAConfig = field(default_factory=_default_group_a_config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_c": int(self.d_c),
            "d_s": int(self.d_s),
            "n_heads": int(self.n_heads),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "bce_weight": float(self.bce_weight),
            "pilot_nll_weight": float(self.pilot_nll_weight),
            "prior_reg_weight": float(self.prior_reg_weight),
            "include_global_branch": bool(self.include_global_branch),
            "max_pool_size": int(self.max_pool_size),
            "n0_scale": float(self.n0_scale),
            "architecture": str(self.architecture),
            "group_b_cfg": self.group_b_cfg.to_dict(),
            "group_a_cfg": self.group_a_cfg.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupCConfig":
        gb_raw = dict(data.get("group_b_cfg", {}))
        if "pdp_hypotheses" in gb_raw:
            gb_raw["pdp_hypotheses"] = tuple((int(nt), float(tr)) for nt, tr in gb_raw["pdp_hypotheses"])
        ga_raw = dict(data.get("group_a_cfg", {}))
        return cls(
            d_c=int(data.get("d_c", 8)),
            d_s=int(data.get("d_s", 12)),
            n_heads=int(data.get("n_heads", 2)),
            lr=float(data.get("lr", 3e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            epochs=int(data.get("epochs", 40)),
            batch_size=int(data.get("batch_size", 16)),
            bce_weight=float(data.get("bce_weight", 1.0)),
            pilot_nll_weight=float(data.get("pilot_nll_weight", 0.05)),
            prior_reg_weight=float(data.get("prior_reg_weight", 1e-3)),
            include_global_branch=bool(data.get("include_global_branch", True)),
            max_pool_size=int(data.get("max_pool_size", 64)),
            n0_scale=float(data.get("n0_scale", 0.7)),
            architecture=str(data.get("architecture", "CStarBranchWeightAttentionNet")),
            group_b_cfg=GroupBConfig(**gb_raw) if gb_raw else _default_group_b_config(),
            group_a_cfg=GroupAConfig(**ga_raw) if ga_raw else _default_group_a_config(),
        )


def _build_pilot_pool_summary(
    frames: Sequence[RealOFDMFrame],
    *,
    max_pool_size: int = 64,
    group_a_cfg: Optional[GroupAConfig] = None,
) -> dict[str, Any]:
    """Extract compact (< 30 KB) pilot-only C*-normalized summary from train frames."""
    _validate_non_test_frames(frames, "GroupCTrainableReceiver._build_pilot_pool_summary")
    cfg = _default_group_a_config() if group_a_cfg is None else group_a_cfg
    selected = build_reference_pool(frames, max_pool_size=max_pool_size, classical_cfg=cfg.to_classical_config())

    m_pool = len(selected)
    pilot_fft = list(PILOT_FFT_INDICES)
    h_p_unit = np.zeros((m_pool, len(pilot_fft)), dtype=np.complex128)
    sig2_unit = np.zeros((m_pool, len(pilot_fft)), dtype=np.float64)
    frame_ids: list[int] = []
    group_ids: list[str] = []

    for j, fr in enumerate(selected):
        rx_grid, est_n0, pilot_values = _validate_frame_inputs(fr)
        y_p = rx_grid[PILOT_SYMBOL_IDX, pilot_fft]
        h_p = y_p / pilot_values
        n0_eff = max(float(cfg.n0_scale) * float(est_n0), 0.0)
        sig2_p = n0_eff / np.maximum(np.abs(pilot_values) ** 2, 1e-30)

        mean_hp2 = float(np.mean(np.abs(h_p) ** 2))
        mean_sig2 = float(np.mean(sig2_p))
        sig_h2 = max(mean_hp2 - mean_sig2, 0.05 * mean_hp2)
        s_pool = math.sqrt(sig_h2) if sig_h2 > 0.0 else 1.0
        h_p_unit[j] = h_p / s_pool
        sig2_unit[j] = sig2_p / (s_pool * s_pool)
        frame_ids.append(int(fr.frame_id))
        group_ids.append(str(fr.group_id))

    return {
        "h_p_unit": h_p_unit,
        "sig2_unit": sig2_unit,
        "frame_ids": frame_ids,
        "group_ids": group_ids,
        "_val_cache": {},
    }


def _get_pool_val_estimates(
    pool_summary: dict[str, Any],
    obs_idx: Sequence[int],
    *,
    group_a_cfg: GroupAConfig,
    target_active_indices: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Lazily compute and cache per-pool-frame unit-scale 1D LMMSE value estimates."""
    obs_tuple = tuple(int(i) for i in obs_idx)
    target_tuple = (
        None if target_active_indices is None else tuple(int(x) for x in target_active_indices)
    )
    cache_key = (
        obs_tuple,
        target_tuple,
        int(group_a_cfg.n_taps),
        float(group_a_cfg.tau_rms),
        float(group_a_cfg.diag_reg),
        bool(group_a_cfg.detrend_timing),
    )
    cache = pool_summary.setdefault("_val_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    h_p_unit_all = np.asarray(pool_summary["h_p_unit"], dtype=np.complex128)
    sig2_unit_all = np.asarray(pool_summary["sig2_unit"], dtype=np.float64)
    m_pool = int(h_p_unit_all.shape[0])
    _, val_idx = _partition_key_val_pilots(obs_tuple)
    k_target_len = N_ACTIVE if target_active_indices is None else len(target_active_indices)
    cls_cfg = group_a_cfg.to_classical_config()

    h_val_all = np.zeros((m_pool, k_target_len), dtype=np.complex128)
    v_val_all = np.zeros((m_pool, k_target_len), dtype=np.float64)
    for j in range(m_pool):
        h_j, v_j, _ = _estimate_from_hp_subset(
            h_p_unit_all[j, val_idx],
            val_idx,
            sig2_unit_all[j, val_idx],
            config=cls_cfg,
            target_active_indices=target_active_indices,
        )
        h_val_all[j] = h_j
        v_val_all[j] = v_j

    h_val_all.setflags(write=False)
    v_val_all.setflags(write=False)
    cache[cache_key] = (h_val_all, v_val_all)
    return h_val_all, v_val_all


def _estimate_global_branch_from_summary(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    obs_idx: Sequence[int],
    pool_summary: Optional[dict[str, Any]],
    *,
    group_a_cfg: GroupAConfig,
    exclude_frame_id: Optional[int] = None,
    target_active_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute C*-equivariant Group A global_content_attn estimate strictly from pilot-only summary."""
    cls_cfg = group_a_cfg.to_classical_config()
    h_q, v_q, _ = _estimate_sym2_channel(
        rx_norm,
        pilot_values,
        obs_idx,
        n0_norm,
        method="lmmse_1d",
        config=cls_cfg,
        target_active_indices=target_active_indices,
    )

    if pool_summary is None:
        return h_q, v_q

    h_p_unit_all = np.asarray(pool_summary["h_p_unit"], dtype=np.complex128)
    if h_p_unit_all.shape[0] == 0:
        return h_q, v_q

    h_val_all, v_val_all = _get_pool_val_estimates(
        pool_summary,
        obs_idx,
        group_a_cfg=group_a_cfg,
        target_active_indices=target_active_indices,
    )
    pool_fids = pool_summary.get("frame_ids", [])

    if exclude_frame_id is not None and len(pool_fids) == len(h_p_unit_all):
        keep = [idx for idx, fid in enumerate(pool_fids) if int(fid) != int(exclude_frame_id)]
        if keep:
            h_p_unit_all = h_p_unit_all[keep]
            h_val_all = h_val_all[keep]
            v_val_all = v_val_all[keep]

    m_pool = int(h_p_unit_all.shape[0])
    if m_pool == 0:
        return h_q, v_q

    key_idx, _ = _partition_key_val_pilots(obs_idx)
    key_fft = [PILOT_FFT_INDICES[i] for i in key_idx]
    obs_fft = [PILOT_FFT_INDICES[i] for i in obs_idx]

    h_q_obs = rx_norm[PILOT_SYMBOL_IDX, obs_fft] / pilot_values[obs_idx]
    n0_q_eff = max(float(group_a_cfg.n0_scale) * float(n0_norm), 0.0)
    sig2_q_obs = n0_q_eff / np.maximum(np.abs(pilot_values[obs_idx]) ** 2, 1e-30)
    mean_hq2 = float(np.mean(np.abs(h_q_obs) ** 2))
    if mean_hq2 <= 1e-28:
        return h_q, v_q

    sig_hq2 = max(mean_hq2 - float(np.mean(sig2_q_obs)), 0.05 * mean_hq2)
    s_q = math.sqrt(sig_hq2)

    k_q = rx_norm[PILOT_SYMBOL_IDX, key_fft] / pilot_values[key_idx]
    norm_kq = float(np.linalg.norm(k_q))
    if norm_kq <= 1e-30:
        return h_q, v_q

    k_pool = h_p_unit_all[:, key_idx]
    inner_vals = np.einsum("mj,j->m", np.conj(k_pool), k_q)
    norm_kp = np.linalg.norm(k_pool, axis=1)
    denom = norm_kp * norm_kq
    cos_mag = np.where(denom > 1e-30, np.minimum(np.abs(inner_vals) / np.maximum(denom, 1e-30), 1.0), 0.0)

    tau = max(float(group_a_cfg.temperature), 1e-6)
    scale_d = math.sqrt(float(max(len(key_idx), 1)))
    logits = (scale_d * cos_mag) / tau
    logits = np.clip(logits - float(np.max(logits)), -700.0, 0.0)
    exp_l = np.exp(logits)
    exp_sum = float(np.sum(exp_l))
    w = (exp_l / exp_sum) if (np.isfinite(exp_sum) and exp_sum > 0.0) else np.full(m_pool, 1.0 / float(m_pool))

    mag_in = np.abs(inner_vals)
    u_vec = np.where(mag_in > 1e-30, inner_vals / np.maximum(mag_in, 1e-30), 1.0 + 0.0j)
    aligned_h = (u_vec * s_q)[:, None] * h_val_all
    val_v = (s_q * s_q) * v_val_all

    w_col = w[:, None]
    h_ref = np.sum(w_col * aligned_h, axis=0)
    v_within = np.sum(w_col * val_v, axis=0)
    v_between = np.sum(w_col * (np.abs(aligned_h - h_ref[None, :]) ** 2), axis=0)
    v_ref = np.maximum(float(group_a_cfg.prior_scale) * (v_within + v_between), 1e-30)

    inv_vq = 1.0 / np.maximum(v_q, 1e-30)
    inv_vref = 1.0 / v_ref
    v_post = np.maximum(1.0 / (inv_vq + inv_vref), 1e-30).astype(np.float64)
    h_post = (v_post * (h_q * inv_vq + h_ref * inv_vref)).astype(np.complex128)
    return h_post, v_post


def _global_branch_pilot_cv_ll(
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    obs_idx: Sequence[int],
    n0_norm: float,
    pool_summary: Optional[dict[str, Any]],
    *,
    group_a_cfg: GroupAConfig,
    exclude_frame_id: Optional[int] = None,
) -> float:
    """Compute 4-fold pilot CV log-likelihood of the global branch strictly inside obs_idx."""
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
        h_pred, v_pred = _estimate_global_branch_from_summary(
            rx_norm,
            pilot_values,
            n0_norm,
            o_sub,
            pool_summary,
            group_a_cfg=group_a_cfg,
            exclude_frame_id=exclude_frame_id,
            target_active_indices=target_active,
        )
        fft_h = [PILOT_FFT_INDICES[i] for i in h_sub]
        z_obs = rx_norm[PILOT_SYMBOL_IDX, fft_h] / pilot_values[h_sub]
        sig2_obs = max(float(group_a_cfg.n0_scale) * float(n0_norm), 0.0) / np.maximum(
            np.abs(pilot_values[h_sub]) ** 2, 1e-30
        )
        v_eff = np.maximum(np.maximum(v_pred, 0.0) + sig2_obs, 1e-12)
        err2 = np.abs(h_pred - z_obs) ** 2
        fold_lls.append(float(np.sum(-np.log(math.pi * v_eff) - err2 / v_eff)))
    return float(np.mean(fold_lls)) if fold_lls else 0.0


def _build_group_c_bank_normalized(
    frame: RealOFDMFrame,
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    s_scale: float,
    obs_idx: Sequence[int],
    *,
    cfg: GroupCConfig,
    pool_summary: Optional[dict[str, Any]],
    include_temporal_branches: bool,
    exclude_frame_id: Optional[int] = None,
) -> tuple[list[CandidateBranch], np.ndarray, np.ndarray]:
    """Build Group B fixed candidate bank (+ optional Group A global branch) in normalized gauge.

    Returns (branches_orig_gauge, prior_logits, prior_alpha).
    """
    branches, _ = _build_bank_normalized(
        frame,
        rx_norm,
        pilot_values,
        n0_norm,
        s_scale,
        obs_idx,
        cfg=cfg.group_b_cfg,
        include_temporal_branches=include_temporal_branches,
    )

    if cfg.include_global_branch and pool_summary is not None and len(pool_summary.get("h_p_unit", ())) > 0:
        h_g_1d, v_g_1d = _estimate_global_branch_from_summary(
            rx_norm,
            pilot_values,
            n0_norm,
            obs_idx,
            pool_summary,
            group_a_cfg=cfg.group_a_cfg,
            exclude_frame_id=exclude_frame_id,
        )
        g_cv_ll = _global_branch_pilot_cv_ll(
            rx_norm,
            pilot_values,
            obs_idx,
            n0_norm,
            pool_summary,
            group_a_cfg=cfg.group_a_cfg,
            exclude_frame_id=exclude_frame_id,
        )
        h_2d_norm = np.broadcast_to(h_g_1d[None, :], (N_SYMB, N_ACTIVE)).copy()
        v_2d_norm = np.broadcast_to(v_g_1d[None, :], (N_SYMB, N_ACTIVE)).copy()
        s_scale2 = s_scale * s_scale
        branches.append(
            CandidateBranch(
                name="global_content_attn",
                h_est=(h_2d_norm * s_scale).astype(np.complex128),
                h_var=np.maximum(v_2d_norm * s_scale2, 1e-30 * s_scale2).astype(np.float64),
                log_score=float(g_cv_ll),
                omega_t=0.0,
                pdp_params=(int(cfg.group_a_cfg.n_taps), float(cfg.group_a_cfg.tau_rms)),
                meta={"pdp_cv_ll": float(g_cv_ll), "delta_ll_omega": 0.0, "is_global_ref": True},
            )
        )

    raw_scores = np.asarray([float(br.log_score) for br in branches], dtype=np.float64)
    beta = float(cfg.group_b_cfg.beta)
    prior_logits = np.clip(beta * raw_scores - float(np.max(beta * raw_scores)), -700.0, 0.0)
    prior_alpha = _scores_to_alpha(raw_scores, beta)
    return branches, prior_logits.astype(np.float64), prior_alpha.astype(np.float64)


def _extract_network_features(
    frame: RealOFDMFrame,
    rx_norm: np.ndarray,
    pilot_values: np.ndarray,
    n0_norm: float,
    s_scale: float,
    obs_idx: Sequence[int],
    branches: Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]],
    prior_logits: Optional[np.ndarray],
    *,
    cfg: GroupCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract strictly C*-invariant scalars, U(1)-equivariant pilot tensors, and normalized bank.

    Strictly accesses ONLY ``obs_idx`` pilots (never held-out pilots) and zero labels/SINR.
    Returns:
        scalar_feats: (R, N_SCALAR_FEATS) float64
        complex_feats: (R, P, N_COMPLEX_FEATS) complex128
        pilot_coords: (P,) float64
        prior_logits_out: (R,) float64
        m_data_norm: (R, N_DATA_RES) complex128
        v_data_norm: (R, N_DATA_RES) float64
    """
    r_count = len(branches)
    if r_count < 1:
        raise ValueError("branches must be non-empty")

    obs_list = [int(i) for i in obs_idx]
    obs_fft = [PILOT_FFT_INDICES[i] for i in obs_list]
    obs_act = [PILOT_ACTIVE_INDICES[i] for i in obs_list]
    pilot_coords = np.asarray(obs_act, dtype=np.float64)

    inv_s = 1.0 / s_scale
    inv_s2 = inv_s * inv_s

    y_p_norm = rx_norm[PILOT_SYMBOL_IDX, obs_fft]
    pv_obs = pilot_values[obs_list]
    h_ls_p = (y_p_norm / pv_obs).astype(np.complex128)  # (P,)

    n0_cal = max(float(cfg.n0_scale) * float(n0_norm), 0.0)
    sig2_p = n0_cal / np.maximum(np.abs(pv_obs) ** 2, 1e-30)

    data_mask_active = np.asarray(frame.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
    rx_active = rx_norm[:, ACTIVE_START:ACTIVE_END]
    y_data = rx_active[data_mask_active]  # (D,)

    pts = constellation_points(QM)  # (16,)
    p_len = len(obs_list)

    complex_feats = np.zeros((r_count, p_len, N_COMPLEX_FEATS), dtype=np.complex128)
    m_data_norm = np.zeros((r_count, len(y_data)), dtype=np.complex128)
    v_data_norm = np.zeros((r_count, len(y_data)), dtype=np.float64)

    if prior_logits is None:
        if all(isinstance(b, CandidateBranch) for b in branches):
            raw_sc = np.asarray([float(b.log_score) for b in branches], dtype=np.float64)  # type: ignore[union-attr]
            beta = float(cfg.group_b_cfg.beta)
            p_logits = np.clip(beta * raw_sc - float(np.max(beta * raw_sc)), -700.0, 0.0)
        else:
            p_logits = np.zeros(r_count, dtype=np.float64)
    else:
        p_logits = np.asarray(prior_logits, dtype=np.float64).ravel()

    prior_centered = p_logits - float(np.mean(p_logits))
    max_slope = max(float(cfg.group_b_cfg.phase_track_max_slope), 1e-6)

    pilot_nlls = np.zeros(r_count, dtype=np.float64)
    blind_lls = np.zeros(r_count, dtype=np.float64)
    log_var_ratios = np.zeros(r_count, dtype=np.float64)
    is_nz_om = np.zeros(r_count, dtype=np.float64)
    abs_om = np.zeros(r_count, dtype=np.float64)
    is_glob = np.zeros(r_count, dtype=np.float64)

    # Subsample off-pilot data REs deterministically for fast blind_data_ll feature
    stride_d = max(1, len(y_data) // 256)
    y_sub = y_data[::stride_d]

    for r, br in enumerate(branches):
        m_raw = _extract_active_slice(br.h_est if isinstance(br, CandidateBranch) else br[0], f"branches[{r}].h_est")
        v_raw = _extract_active_slice(br.h_var if isinstance(br, CandidateBranch) else br[1], f"branches[{r}].h_var")
        m_arr = np.asarray(m_raw, dtype=np.complex128) * inv_s
        v_arr = np.maximum(np.asarray(v_raw, dtype=np.float64) * inv_s2, 1e-12)

        m_p = m_arr[PILOT_SYMBOL_IDX, obs_act]
        v_p = v_arr[PILOT_SYMBOL_IDX, obs_act]
        complex_feats[r, :, 0] = m_p
        complex_feats[r, :, 1] = h_ls_p - m_p
        complex_feats[r, :, 2] = h_ls_p

        v_eff_p = np.maximum(v_p + sig2_p, 1e-12)
        err2_p = np.abs(h_ls_p - m_p) ** 2
        pilot_nlls[r] = float(np.mean(np.log(math.pi * v_eff_p) + err2_p / v_eff_p))

        m_d = m_arr[data_mask_active]
        v_d = v_arr[data_mask_active]
        m_data_norm[r] = m_d
        v_data_norm[r] = v_d

        m_sub = m_d[::stride_d]
        v_sub = v_d[::stride_d]
        sig2_sub = np.maximum(n0_cal + v_sub, 1e-12)
        diff2 = np.abs(y_sub[:, None] - m_sub[:, None] * pts[None, :]) ** 2
        ll = -np.log(math.pi * sig2_sub)[:, None] - diff2 / sig2_sub[:, None]
        m_max = np.max(ll, axis=1, keepdims=True)
        lse = m_max[:, 0] + np.log(np.sum(np.exp(np.clip(ll - m_max, -700.0, 0.0)), axis=1))
        blind_lls[r] = float(np.mean(lse))

        log_var_ratios[r] = math.log1p(float(np.mean(v_d)) / max(n0_cal, 1e-6))
        if isinstance(br, CandidateBranch):
            om = float(br.omega_t)
            is_nz_om[r] = 1.0 if abs(om) > 1e-12 else 0.0
            abs_om[r] = min(abs(om) / max_slope, 2.0)
            is_glob[r] = 1.0 if bool(br.meta.get("is_global_ref", False)) else 0.0

    pilot_nll_rel = np.arcsinh((pilot_nlls - float(np.mean(pilot_nlls))) * 0.5)
    blind_ll_rel = np.arcsinh((blind_lls - float(np.mean(blind_lls))) * 2.0)

    scalar_feats = np.stack(
        [
            np.clip(prior_centered, -20.0, 20.0),
            is_nz_om,
            abs_om,
            is_glob,
            np.clip(log_var_ratios, 0.0, 20.0),
            pilot_nll_rel,
            blind_ll_rel,
        ],
        axis=-1,
    ).astype(np.float64)

    return scalar_feats, complex_feats, pilot_coords, p_logits, m_data_norm, v_data_norm


class _BranchAttentionBackbone(nn.Module):
    """U(1)-equivariant spectral pilot encoder + permutation-equivariant cross-branch attention."""

    def __init__(
        self,
        d_c: int = 8,
        d_s: int = 12,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        if d_c % n_heads != 0:
            raise ValueError(f"d_c ({d_c}) must be divisible by n_heads ({n_heads})")
        self.d_c = int(d_c)
        self.d_s = int(d_s)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_c // self.n_heads

        # 1. Pilot-axis spectral encoder (coordinate-aware, bias=False)
        self.c_in = ComplexLinear(N_COMPLEX_FEATS, self.d_c, bias=False)
        self.p_norm1 = ComplexRMSNorm(self.d_c)
        self.p_wq = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_wk = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_wv = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_wout = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_rope = ComplexRoPE2D(self.head_dim, base_t=16.0, base_f=32.0)
        with torch.no_grad():
            self.p_wk.weight_r.copy_(self.p_wq.weight_r + 0.05 * torch.randn_like(self.p_wq.weight_r))
            self.p_wk.weight_i.copy_(self.p_wq.weight_i + 0.05 * torch.randn_like(self.p_wq.weight_i))

        self.p_norm2 = ComplexRMSNorm(self.d_c)
        self.p_ff1 = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_act = RadialActivation("silu")
        self.p_ff2 = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.p_pool_proj = ComplexLinear(self.d_c, 2, bias=False)

        # 2. Scalar branch stream (NO BatchNorm)
        self.s_in = nn.Linear(N_SCALAR_FEATS, self.d_s)
        self.s_mod = nn.Linear(self.d_s, self.d_c, bias=False)

        # 3. Cross-branch permutation-equivariant Hermitian attention
        self.b_norm1 = ComplexRMSNorm(self.d_c)
        self.b_wq = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.b_wk = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.b_wv = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.b_wout = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.b_s_bias = nn.Linear(self.d_s, self.n_heads, bias=False)

        self.b_norm2 = ComplexRMSNorm(self.d_c)
        self.b_ff1 = ComplexLinear(self.d_c, self.d_c, bias=False)
        self.b_ff2 = ComplexLinear(self.d_c, self.d_c, bias=False)

        self.s_up1 = nn.Linear(2 * self.d_s + self.n_heads, self.d_s)
        self.s_up2 = nn.Linear(self.d_s, self.d_s)

        # 4. Strictly C*-invariant Hermitian & quadratic readout (p11-style)
        self.w_read_ref = ComplexLinear(self.d_c, N_COMPLEX_FEATS, bias=False)
        self.w_read_bilin = ComplexLinear(self.d_c, 4, bias=False)
        self.head_1 = nn.Linear(self.d_s + N_COMPLEX_FEATS + 2, self.d_s)
        self.head_2 = nn.Linear(self.d_s, 2)

        self.to(torch.float64)

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        pilot_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (raw_delta [B, R], raw_log_tau [B, R])."""
        bsz, n_br, n_p, _ = complex_feats.shape
        x_p = self.c_in(complex_feats.reshape(bsz * n_br, n_p, N_COMPLEX_FEATS))
        xn = self.p_norm1(x_p)

        q = self.p_wq(xn).view(bsz * n_br, n_p, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.p_wk(xn).view(bsz * n_br, n_p, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.p_wv(xn).view(bsz * n_br, n_p, self.n_heads, self.head_dim).transpose(1, 2)

        zeros_t = torch.zeros_like(pilot_coords)
        ph = self.p_rope.get_phasors(zeros_t, pilot_coords).to(q.dtype).view(1, 1, n_p, self.head_dim)
        q_rot = q * ph
        k_rot = k * ph
        v_rot = v * ph

        p_scores = hermitian_attention_scores(q_rot, k_rot)
        p_attn = F.softmax(p_scores, dim=-1).to(v_rot.dtype)
        ctx = (torch.matmul(p_attn, v_rot) * ph.conj()).transpose(1, 2).contiguous().view(bsz * n_br, n_p, self.d_c)
        x_p = x_p + self.p_wout(ctx)

        xn2 = self.p_norm2(x_p)
        x_p = x_p + self.p_ff2(self.p_act(self.p_ff1(xn2)))

        # Strictly U(1)-invariant pooling weights over variable-length observed pilots
        pool_proj = self.p_pool_proj(self.p_norm2(x_p))
        pool_score = (pool_proj.real ** 2 + pool_proj.imag ** 2).sum(dim=-1)  # (B*R, P)
        gamma = F.softmax(pool_score, dim=-1).view(bsz, n_br, n_p, 1).to(x_p.dtype)

        h_r = torch.sum(gamma * x_p.view(bsz, n_br, n_p, self.d_c), dim=2)  # (B, R, d_c)
        ref_r = torch.sum(gamma * complex_feats, dim=2)  # (B, R, C_c)

        # Scalar branch stream modulating complex representation via strictly C*-invariant factor
        s = F.silu(self.s_in(scalar_feats))  # (B, R, d_s)
        mod = (1.0 + torch.tanh(self.s_mod(s))).to(h_r.dtype)
        h_r = h_r * mod

        # Cross-branch permutation-equivariant Hermitian attention (no RoPE across branches)
        hn = self.b_norm1(h_r)
        bq = self.b_wq(hn).view(bsz, n_br, self.n_heads, self.head_dim).transpose(1, 2)
        bk = self.b_wk(hn).view(bsz, n_br, self.n_heads, self.head_dim).transpose(1, 2)
        bv = self.b_wv(hn).view(bsz, n_br, self.n_heads, self.head_dim).transpose(1, 2)

        b_scores = hermitian_attention_scores(bq, bk)
        key_bias = self.b_s_bias(s).transpose(1, 2).unsqueeze(2)  # (B, H, 1, R)
        b_scores = b_scores + key_bias
        b_attn = F.softmax(b_scores, dim=-1)  # (B, H, R, R)
        b_ctx = torch.matmul(b_attn.to(bv.dtype), bv).transpose(1, 2).contiguous().view(bsz, n_br, self.d_c)
        h_r = h_r + self.b_wout(b_ctx)
        h_r = h_r + self.b_ff2(self.p_act(self.b_ff1(self.b_norm2(h_r))))

        # Permutation-equivariant scalar update combining local s, global mean s, and mean attention affinity
        s_mean = s.mean(dim=1, keepdim=True).expand_as(s)
        attn_diag = b_attn.diagonal(dim1=-2, dim2=-1).transpose(1, 2)  # (B, R, H)
        s = s + self.s_up2(F.silu(self.s_up1(torch.cat([s, s_mean, attn_diag], dim=-1))))

        # Hermitian & quadratic strictly C*-invariant readout
        u_ref = self.w_read_ref(h_r)  # (B, R, C_c)
        inv_ref = torch.asinh((ref_r.conj() * u_ref).real)
        u_b = self.w_read_bilin(h_r)  # (B, R, 4)
        inv_bilin = torch.asinh((u_b[..., :2].conj() * u_b[..., 2:]).real)

        head_in = torch.cat([s, inv_ref, inv_bilin], dim=-1)
        out = self.head_2(F.silu(self.head_1(head_in)))  # (B, R, 2)
        return out[..., 0], out[..., 1]


class BranchWeightAttentionNet(nn.Module):
    """C*-equivariant branch-weight predictor centered around frozen step-0 ``initial_map``."""

    def __init__(self, d_c: int = 8, d_s: int = 12, n_heads: int = 2) -> None:
        super().__init__()
        self.backbone = _BranchAttentionBackbone(d_c=d_c, d_s=d_s, n_heads=n_heads)
        self.initial_map = copy.deepcopy(self.backbone)
        for param in self.initial_map.parameters():
            param.requires_grad_(False)

    def real_parameters(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def compute_initial_offsets(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        pilot_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.initial_map(
                scalar_feats.detach(),
                complex_feats.detach(),
                pilot_coords.detach(),
            )

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        pilot_coords: torch.Tensor,
        prior_logits: torch.Tensor,
        init_offsets: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (alpha [B, R], logits [B, R], delta_logit [B, R])."""
        raw_delta, raw_log_tau = self.backbone(scalar_feats, complex_feats, pilot_coords)
        if init_offsets is None:
            delta_0, log_tau_0 = self.compute_initial_offsets(scalar_feats, complex_feats, pilot_coords)
        else:
            delta_0, log_tau_0 = init_offsets
        delta_logit = raw_delta - delta_0
        log_tau = (raw_log_tau - log_tau_0).mean(dim=-1, keepdim=True)
        tau = 2.0 * torch.sigmoid(torch.clamp(log_tau, -8.0, 8.0))
        logits = tau * prior_logits + delta_logit
        alpha = F.softmax(logits, dim=-1)
        return alpha, logits, delta_logit


def _compute_log_g_torch(
    y_data: torch.Tensor,
    m_data: torch.Tensor,
    v_data: torch.Tensor,
    n0_cal: torch.Tensor,
    pts_t: torch.Tensor,
) -> torch.Tensor:
    """Compute exact per-branch 16-QAM symbol log-likelihoods ``log_g`` of shape ``(B, R, D, 16)``."""
    sigma2 = torch.clamp(n0_cal[:, None, None] + v_data, min=1e-12)  # (B, R, D)
    diff = y_data[:, None, :, None] - m_data[:, :, :, None] * pts_t[None, None, None, :]
    diff2 = diff.real ** 2 + diff.imag ** 2
    return -torch.log(math.pi * sigma2)[..., None] - diff2 / sigma2[..., None]


def differentiable_mixture_llrs_torch(
    alpha: torch.Tensor,
    log_g: torch.Tensor,
    idx1_t: torch.Tensor,
    idx0_t: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable PyTorch mirror of Group B ``exact_mixture_likelihood`` bit LLRs."""
    log_alpha = torch.log(torch.clamp(alpha, min=1e-300))
    log_terms = log_alpha[:, :, None, None] + log_g  # (B, R, D, 16)
    log_mix = torch.logsumexp(log_terms, dim=1)  # (B, D, 16)
    lse1 = torch.logsumexp(log_mix[:, :, idx1_t], dim=-1)  # (B, D, 4)
    lse0 = torch.logsumexp(log_mix[:, :, idx0_t], dim=-1)  # (B, D, 4)
    return lse1 - lse0


def _failed_output(
    err_msg: str,
    *,
    architecture: str,
    param_count: int,
    seed: int,
    train_time_s: float,
    latency_ms: float,
) -> ReceiverOutput:
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
            "mode": "group_c_trainable",
            "architecture": str(architecture),
            "param_count": int(param_count),
            "seed": int(seed),
            "train_time_s": float(train_time_s),
            "n_candidates": 1,
            "branch_weights": [1.0],
            "prior_weights": [1.0],
            "versions": _runtime_versions(),
            "heldout_pilot_indices": [],
            "latency_ms": float(latency_ms),
            "failed": True,
            "error": str(err_msg),
        },
    )


class GroupCTrainableReceiver:
    """Group C lightweight C*-equivariant branch-weight attention receiver (F10 / R3-C)."""

    def __init__(
        self,
        config: Optional[GroupCConfig] = None,
        *,
        seed: int = 20261121,
        context_pool: Optional[Sequence[RealOFDMFrame]] = None,
    ) -> None:
        self.config: GroupCConfig = GroupCConfig() if config is None else copy.deepcopy(config)
        self.seed: int = int(seed)
        self.train_time_s: float = 0.0
        self.best_epoch: int = 0
        self.history: list[dict[str, Any]] = []
        self.pool_summary: Optional[dict[str, Any]] = None

        if context_pool is not None and len(context_pool) > 0 and self.config.include_global_branch:
            self.pool_summary = _build_pilot_pool_summary(
                context_pool,
                max_pool_size=self.config.max_pool_size,
                group_a_cfg=self.config.group_a_cfg,
            )

        self._init_network(self.seed)

    def _init_network(self, seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        self.net = BranchWeightAttentionNet(
            d_c=self.config.d_c,
            d_s=self.config.d_s,
            n_heads=self.config.n_heads,
        )
        self.net.eval()

    @property
    def param_count(self) -> int:
        return int(self.net.real_parameters())

    def _predict_branch_weights(
        self,
        scalar_feats: np.ndarray,
        complex_feats: np.ndarray,
        pilot_coords: np.ndarray,
        prior_logits: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            s_t = torch.from_numpy(scalar_feats).to(torch.float64).unsqueeze(0)
            c_t = torch.from_numpy(complex_feats).to(torch.complex128).unsqueeze(0)
            k_t = torch.from_numpy(pilot_coords).to(torch.float64)
            p_t = torch.from_numpy(prior_logits).to(torch.float64).unsqueeze(0)
            alpha_t, _, _ = self.net(s_t, c_t, k_t, p_t)
            alpha = alpha_t.squeeze(0).cpu().numpy().astype(np.float64)
        alpha = np.maximum(alpha, 0.0)
        a_sum = float(np.sum(alpha))
        if not np.isfinite(a_sum) or a_sum <= 0.0:
            return np.full(len(prior_logits), 1.0 / float(len(prior_logits)), dtype=np.float64)
        return (alpha / a_sum).astype(np.float64)

    def _precompute_split(
        self,
        frames: Sequence[RealOFDMFrame],
        *,
        include_cv_folds: bool,
        exclude_self_from_pool: bool,
        pts_t: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Precompute fixed candidate banks, network features, and log_g tensors."""
        all_obs = list(range(14))
        items: list[dict[str, Any]] = []
        for fr in frames:
            rx_grid, est_n0, pilot_values = _validate_frame_inputs(fr)
            active_mask = np.asarray(fr.pilot_mask | fr.data_mask, dtype=bool)
            s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
            s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
            rx_norm = rx_grid / s_scale
            n0_norm = est_n0 / (s_scale ** 2)
            n0_cal = max(float(self.config.n0_scale) * float(n0_norm), 0.0)

            excl_id = int(fr.frame_id) if exclude_self_from_pool else None
            branches, prior_logits, _ = _build_group_c_bank_normalized(
                fr,
                rx_norm,
                pilot_values,
                n0_norm,
                s_scale,
                all_obs,
                cfg=self.config,
                pool_summary=self.pool_summary,
                include_temporal_branches=True,
                exclude_frame_id=excl_id,
            )
            sc, cx, p_coords, p_logits, m_d, v_d = _extract_network_features(
                fr,
                rx_norm,
                pilot_values,
                n0_norm,
                s_scale,
                all_obs,
                branches,
                prior_logits,
                cfg=self.config,
            )
            data_mask_active = np.asarray(fr.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
            y_d = rx_norm[:, ACTIVE_START:ACTIVE_END][data_mask_active]

            y_t = torch.from_numpy(y_d).to(torch.complex128).unsqueeze(0)
            m_t = torch.from_numpy(m_d).to(torch.complex128).unsqueeze(0)
            v_t = torch.from_numpy(v_d).to(torch.float64).unsqueeze(0)
            n0_t = torch.tensor([n0_cal], dtype=torch.float64)
            log_g = _compute_log_g_torch(y_t, m_t, v_t, n0_t, pts_t).squeeze(0)  # (R, D, 16)

            cv_items: list[dict[str, Any]] = []
            if include_cv_folds:
                folds = fr.meta.get("heldout_pilot_folds") if isinstance(fr.meta, dict) else None
                if not folds:
                    folds = build_heldout_pilot_masks(4)
                for fold in folds:
                    h_idx = [int(i) for i in fold["heldout_pilot_indices"]]
                    h_set = set(h_idx)
                    o_idx = [int(i) for i in fold["obs_pilot_indices"] if int(i) not in h_set]
                    rx_cv, n0_cv, s_cv = _cv_fold_normalized_inputs(fr, rx_grid, est_n0, o_idx)
                    cv_br, cv_pl, _ = _build_group_c_bank_normalized(
                        fr,
                        rx_cv,
                        pilot_values,
                        n0_cv,
                        s_cv,
                        o_idx,
                        cfg=self.config,
                        pool_summary=self.pool_summary,
                        include_temporal_branches=False,
                        exclude_frame_id=excl_id,
                    )
                    cv_sc, cv_cx, cv_coords, cv_pl_out, _, _ = _extract_network_features(
                        fr,
                        rx_cv,
                        pilot_values,
                        n0_cv,
                        s_cv,
                        o_idx,
                        cv_br,
                        cv_pl,
                        cfg=self.config,
                    )
                    inv_s_cv = 1.0 / s_cv
                    inv_s_cv2 = inv_s_cv * inv_s_cv
                    h_act = [PILOT_ACTIVE_INDICES[i] for i in h_idx]
                    h_fft = [PILOT_FFT_INDICES[i] for i in h_idx]
                    z_obs = (rx_cv[PILOT_SYMBOL_IDX, h_fft] / pilot_values[h_idx]).astype(np.complex128)
                    sig2_obs = max(float(self.config.n0_scale) * float(n0_cv), 0.0) / np.maximum(
                        np.abs(pilot_values[h_idx]) ** 2, 1e-30
                    )
                    m_h = np.stack(
                        [
                            _extract_active_slice(b.h_est, "h_est")[PILOT_SYMBOL_IDX, h_act] * inv_s_cv
                            for b in cv_br
                        ],
                        axis=0,
                    ).astype(np.complex128)
                    v_h = np.stack(
                        [
                            np.maximum(
                                _extract_active_slice(b.h_var, "h_var")[PILOT_SYMBOL_IDX, h_act] * inv_s_cv2,
                                1e-12,
                            )
                            for b in cv_br
                        ],
                        axis=0,
                    ).astype(np.float64)
                    v_eff = np.maximum(v_h + sig2_obs[None, :], 1e-12)
                    err2 = np.abs(z_obs[None, :] - m_h) ** 2
                    log_p_h = (-np.log(math.pi * v_eff) - err2 / v_eff).astype(np.float64)  # (R, H_fold)
                    cv_items.append(
                        {
                            "scalar": torch.from_numpy(cv_sc).to(torch.float64),
                            "complex": torch.from_numpy(cv_cx).to(torch.complex128),
                            "coords": torch.from_numpy(cv_coords).to(torch.float64),
                            "prior_logits": torch.from_numpy(cv_pl_out).to(torch.float64),
                            "log_p_h": torch.from_numpy(log_p_h).to(torch.float64),
                        }
                    )

            items.append(
                {
                    "scalar": torch.from_numpy(sc).to(torch.float64),
                    "complex": torch.from_numpy(cx).to(torch.complex128),
                    "coords": torch.from_numpy(p_coords).to(torch.float64),
                    "prior_logits": torch.from_numpy(p_logits).to(torch.float64),
                    "log_g": log_g,
                    "tx_bits": torch.from_numpy(np.asarray(fr.tx_bits, dtype=np.float64)),
                    "cv_items": cv_items,
                }
            )
        return items

    def precompute_training_splits(
        self,
        train_frames: Sequence[RealOFDMFrame],
        dev_frames: Optional[Sequence[RealOFDMFrame]] = None,
        *,
        include_cv_folds: bool = True,
        context_pool: Optional[Sequence[RealOFDMFrame]] = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """Precompute fixed candidate banks & tensors once across multiple seeds."""
        _validate_non_test_frames(train_frames, "GroupCTrainableReceiver.precompute_training_splits(train)")
        if dev_frames is not None and len(dev_frames) > 0:
            _validate_non_test_frames(dev_frames, "GroupCTrainableReceiver.precompute_training_splits(dev)")
        t_pre = time.perf_counter()
        if self.config.include_global_branch:
            pool_src = context_pool if (context_pool is not None and len(context_pool) > 0) else train_frames
            _validate_non_test_frames(pool_src, "GroupCTrainableReceiver.precompute_training_splits(pool)")
            if self.pool_summary is None:
                self.pool_summary = _build_pilot_pool_summary(
                    pool_src,
                    max_pool_size=self.config.max_pool_size,
                    group_a_cfg=self.config.group_a_cfg,
                )
        else:
            self.pool_summary = None
        pts_t = torch.from_numpy(np.array(constellation_points(QM), copy=True)).to(torch.complex128)
        tr_items = self._precompute_split(
            train_frames,
            include_cv_folds=include_cv_folds,
            exclude_self_from_pool=True,
            pts_t=pts_t,
        )
        dv_items = (
            self._precompute_split(
                dev_frames,
                include_cv_folds=include_cv_folds,
                exclude_self_from_pool=False,
                pts_t=pts_t,
            )
            if (dev_frames is not None and len(dev_frames) > 0)
            else []
        )
        return tr_items, dv_items, float(time.perf_counter() - t_pre)

    def fit(
        self,
        train_frames: Sequence[RealOFDMFrame],
        dev_frames: Optional[Sequence[RealOFDMFrame]] = None,
        epochs: Optional[int] = None,
        *,
        context_pool: Optional[Sequence[RealOFDMFrame]] = None,
        precomputed_train: Optional[list[dict[str, Any]]] = None,
        precomputed_dev: Optional[list[dict[str, Any]]] = None,
        precompute_time_s: float = 0.0,
    ) -> "GroupCTrainableReceiver":
        """Train lightweight Group C network strictly on train_frames / dev_frames (zero test/guard)."""
        _validate_non_test_frames(train_frames, "GroupCTrainableReceiver.fit(train_frames)")
        if dev_frames is not None and len(dev_frames) > 0:
            _validate_non_test_frames(dev_frames, "GroupCTrainableReceiver.fit(dev_frames)")

        t0 = time.perf_counter()
        self._init_network(self.seed)

        if self.config.include_global_branch:
            pool_src = context_pool if (context_pool is not None and len(context_pool) > 0) else train_frames
            _validate_non_test_frames(pool_src, "GroupCTrainableReceiver.fit(context_pool)")
            self.pool_summary = _build_pilot_pool_summary(
                pool_src,
                max_pool_size=self.config.max_pool_size,
                group_a_cfg=self.config.group_a_cfg,
            )
        else:
            self.pool_summary = None

        n_epochs = int(self.config.epochs if epochs is None else epochs)
        if n_epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {n_epochs}")

        pts_t = torch.from_numpy(np.array(constellation_points(QM), copy=True)).to(torch.complex128)
        idx1_np, idx0_np = _bit_index_groups(QM)
        idx1_t = torch.from_numpy(np.array(idx1_np, copy=True)).to(torch.long)
        idx0_t = torch.from_numpy(np.array(idx0_np, copy=True)).to(torch.long)

        need_cv = float(self.config.pilot_nll_weight) > 0.0
        if precomputed_train is not None:
            train_items = precomputed_train
            dev_items = [] if precomputed_dev is None else precomputed_dev
        else:
            train_items = self._precompute_split(
                train_frames,
                include_cv_folds=need_cv,
                exclude_self_from_pool=True,
                pts_t=pts_t,
            )
            dev_items = (
                self._precompute_split(
                    dev_frames,
                    include_cv_folds=need_cv,
                    exclude_self_from_pool=False,
                    pts_t=pts_t,
                )
                if (dev_frames is not None and len(dev_frames) > 0)
                else []
            )

        def _attach_init_offsets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # Cache frozen initial_map offsets batched by (R, coords) so epochs never recompute initial_map
            by_r: dict[int, list[dict[str, Any]]] = {}
            cv_groups: dict[tuple[int, tuple[float, ...]], list[dict[str, Any]]] = {}
            for it in items:
                by_r.setdefault(int(it["scalar"].shape[0]), []).append(it)
                for cv in it["cv_items"]:
                    ckey = (int(cv["scalar"].shape[0]), tuple(float(x) for x in cv["coords"].tolist()))
                    cv_groups.setdefault(ckey, []).append(cv)
            with torch.no_grad():
                for _, grp in by_r.items():
                    sc_b = torch.stack([g["scalar"] for g in grp], dim=0)
                    cx_b = torch.stack([g["complex"] for g in grp], dim=0)
                    d0, lt0 = self.net.compute_initial_offsets(sc_b, cx_b, grp[0]["coords"])
                    for idx_g, g in enumerate(grp):
                        g["init_d0"] = d0[idx_g]
                        g["init_lt0"] = lt0[idx_g]
                for _, cgrp in cv_groups.items():
                    sc_b = torch.stack([c["scalar"] for c in cgrp], dim=0)
                    cx_b = torch.stack([c["complex"] for c in cgrp], dim=0)
                    d0, lt0 = self.net.compute_initial_offsets(sc_b, cx_b, cgrp[0]["coords"])
                    for idx_c, c in enumerate(cgrp):
                        c["init_d0"] = d0[idx_c]
                        c["init_lt0"] = lt0[idx_c]
            return items

        def _batched_cv_nll(items_sub: list[dict[str, Any]]) -> tuple[torch.Tensor, int]:
            cv_groups: dict[tuple[int, tuple[float, ...], int], list[dict[str, Any]]] = {}
            for it in items_sub:
                for cv in it["cv_items"]:
                    ckey = (
                        int(cv["scalar"].shape[0]),
                        tuple(float(x) for x in cv["coords"].tolist()),
                        int(cv["log_p_h"].shape[1]),
                    )
                    cv_groups.setdefault(ckey, []).append(cv)
            if not cv_groups:
                return torch.zeros((), dtype=torch.float64), 0
            nll_sum = torch.zeros((), dtype=torch.float64)
            cnt = 0
            for _, cgrp in cv_groups.items():
                sc_b = torch.stack([c["scalar"] for c in cgrp], dim=0)
                cx_b = torch.stack([c["complex"] for c in cgrp], dim=0)
                pl_b = torch.stack([c["prior_logits"] for c in cgrp], dim=0)
                d0_b = torch.stack([c["init_d0"] for c in cgrp], dim=0)
                lt0_b = torch.stack([c["init_lt0"] for c in cgrp], dim=0)
                lp_b = torch.stack([c["log_p_h"] for c in cgrp], dim=0)  # (B_cv, R, H_fold)
                al_cv, _, _ = self.net(sc_b, cx_b, cgrp[0]["coords"], pl_b, init_offsets=(d0_b, lt0_b))
                log_al = torch.log(torch.clamp(al_cv, min=1e-300))[:, :, None]  # (B_cv, R, 1)
                nll_b = -torch.logsumexp(log_al + lp_b, dim=1).mean(dim=-1)  # (B_cv,)
                nll_sum = nll_sum + nll_b.sum()
                cnt += int(len(cgrp))
            return nll_sum, cnt

        _attach_init_offsets(train_items)
        if dev_items:
            _attach_init_offsets(dev_items)

        params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(n_epochs, 1),
            eta_min=max(float(self.config.lr) * 0.05, 1e-5),
        )

        def _eval_split(items: list[dict[str, Any]]) -> dict[str, float]:
            self.net.eval()
            tot_bce = 0.0
            bit_err = 0
            tot_bits = 0
            with torch.no_grad():
                by_r: dict[int, list[dict[str, Any]]] = {}
                for it in items:
                    by_r.setdefault(int(it["scalar"].shape[0]), []).append(it)
                for _, grp in by_r.items():
                    sc_b = torch.stack([g["scalar"] for g in grp], dim=0)
                    cx_b = torch.stack([g["complex"] for g in grp], dim=0)
                    pl_b = torch.stack([g["prior_logits"] for g in grp], dim=0)
                    lg_b = torch.stack([g["log_g"] for g in grp], dim=0)
                    tb_b = torch.stack([g["tx_bits"] for g in grp], dim=0)
                    d0_b = torch.stack([g["init_d0"] for g in grp], dim=0)
                    lt0_b = torch.stack([g["init_lt0"] for g in grp], dim=0)
                    coords = grp[0]["coords"]
                    alpha, _, _ = self.net(sc_b, cx_b, coords, pl_b, init_offsets=(d0_b, lt0_b))
                    llrs = differentiable_mixture_llrs_torch(alpha, lg_b, idx1_t, idx0_t)
                    bce = float(F.binary_cross_entropy_with_logits(llrs, tb_b, reduction="sum").item())
                    tot_bce += bce
                    bit_err += int(((llrs > 0.0).to(torch.uint8) != tb_b.to(torch.uint8)).sum().item())
                    tot_bits += int(tb_b.numel())
                if need_cv:
                    pnll_sum, pnll_cnt = _batched_cv_nll(items)
                    mean_pnll = float(pnll_sum.item()) / max(pnll_cnt, 1)
                else:
                    mean_pnll = 0.0
            return {
                "ber": float(bit_err / max(tot_bits, 1)),
                "bce": float(tot_bce / max(tot_bits, 1)),
                "pilot_nll": float(mean_pnll),
            }

        # Evaluate step-0 baseline first
        val_items = dev_items if dev_items else train_items
        step0_metrics = _eval_split(val_items)
        best_score = (
            float(step0_metrics["ber"]),
            float(step0_metrics["bce"]) + float(self.config.pilot_nll_weight) * float(step0_metrics["pilot_nll"]),
        )
        best_state = copy.deepcopy(self.net.state_dict())
        self.best_epoch = 0
        self.history = [{"epoch": 0, "val_ber": step0_metrics["ber"], "val_bce": step0_metrics["bce"], "val_pilot_nll": step0_metrics["pilot_nll"]}]

        rng = np.random.default_rng(self.seed)
        n_train = len(train_items)
        batch_sz = max(1, min(int(self.config.batch_size), n_train))
        w_bce = float(self.config.bce_weight)
        w_pnll = float(self.config.pilot_nll_weight)
        w_prior = float(self.config.prior_reg_weight)

        for ep in range(1, n_epochs + 1):
            self.net.train()
            order = rng.permutation(n_train)
            ep_loss = 0.0
            n_seen = 0
            for start in range(0, n_train, batch_sz):
                batch_idx = order[start : start + batch_sz]
                batch_items = [train_items[int(bi)] for bi in batch_idx]
                by_r: dict[int, list[dict[str, Any]]] = {}
                for it in batch_items:
                    by_r.setdefault(int(it["scalar"].shape[0]), []).append(it)

                optimizer.zero_grad()
                batch_total_loss = torch.zeros((), dtype=torch.float64)
                total_b = len(batch_idx)

                for _, grp in by_r.items():
                    sc_b = torch.stack([g["scalar"] for g in grp], dim=0)
                    cx_b = torch.stack([g["complex"] for g in grp], dim=0)
                    pl_b = torch.stack([g["prior_logits"] for g in grp], dim=0)
                    lg_b = torch.stack([g["log_g"] for g in grp], dim=0)
                    tb_b = torch.stack([g["tx_bits"] for g in grp], dim=0)
                    d0_b = torch.stack([g["init_d0"] for g in grp], dim=0)
                    lt0_b = torch.stack([g["init_lt0"] for g in grp], dim=0)
                    coords = grp[0]["coords"]

                    alpha, _, delta_logit = self.net(sc_b, cx_b, coords, pl_b, init_offsets=(d0_b, lt0_b))
                    loss_g = torch.zeros((), dtype=torch.float64)
                    if w_bce > 0.0:
                        llrs = differentiable_mixture_llrs_torch(alpha, lg_b, idx1_t, idx0_t)
                        loss_g = loss_g + w_bce * F.binary_cross_entropy_with_logits(llrs, tb_b)
                    if w_prior > 0.0:
                        loss_g = loss_g + w_prior * torch.mean(delta_logit ** 2)

                    weight_frac = float(len(grp)) / float(total_b)
                    batch_total_loss = batch_total_loss + weight_frac * loss_g

                if w_pnll > 0.0:
                    pnll_sum, pnll_cnt = _batched_cv_nll(batch_items)
                    if pnll_cnt > 0:
                        batch_total_loss = batch_total_loss + w_pnll * (pnll_sum / float(pnll_cnt))

                batch_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                ep_loss += float(batch_total_loss.item()) * total_b
                n_seen += total_b

            scheduler.step()
            val_m = _eval_split(val_items)
            score = (
                float(val_m["ber"]),
                float(val_m["bce"]) + w_pnll * float(val_m["pilot_nll"]),
            )
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(self.net.state_dict())
                self.best_epoch = ep

            self.history.append(
                {
                    "epoch": ep,
                    "train_loss": float(ep_loss / max(n_seen, 1)),
                    "val_ber": float(val_m["ber"]),
                    "val_bce": float(val_m["bce"]),
                    "val_pilot_nll": float(val_m["pilot_nll"]),
                }
            )

        self.net.load_state_dict(best_state, strict=True)
        self.net.eval()
        self.train_time_s = float(time.perf_counter() - t0) + float(precompute_time_s)
        return self

    train_on_splits = fit

    def _compute_heldout_cv(
        self,
        frame: RealOFDMFrame,
        rx_grid: np.ndarray,
        pilot_values: np.ndarray,
        est_n0: float,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Strict per-fold s_scale_cv held-out pilot CV (zero leakage from held-out REs)."""
        explicit_idx = frame.meta.get("heldout_pilot_indices") if isinstance(frame.meta, dict) else None
        if explicit_idx is not None and len(explicit_idx) > 0:
            eval_idx = [int(i) for i in explicit_idx]
            eval_set = set(eval_idx)
            obs_idx = [i for i in range(14) if i not in eval_set]
            rx_cv, n0_cv, s_cv = _cv_fold_normalized_inputs(frame, rx_grid, est_n0, obs_idx)
            cv_br, cv_pl, _ = _build_group_c_bank_normalized(
                frame,
                rx_cv,
                pilot_values,
                n0_cv,
                s_cv,
                obs_idx,
                cfg=self.config,
                pool_summary=self.pool_summary,
                include_temporal_branches=False,
            )
            cv_sc, cv_cx, cv_coords, cv_pl_out, _, _ = _extract_network_features(
                frame,
                rx_cv,
                pilot_values,
                n0_cv,
                s_cv,
                obs_idx,
                cv_br,
                cv_pl,
                cfg=self.config,
            )
            cv_alpha = self._predict_branch_weights(cv_sc, cv_cx, cv_coords, cv_pl_out)
            v_floor = max(1e-12 * (s_cv ** 2), 1e-300)
            h_agg, v_agg = _aggregate_h_only(
                cv_br, cv_alpha, mode="exact_mixture_likelihood", min_var=v_floor
            )
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
            rx_cv, n0_cv, s_cv = _cv_fold_normalized_inputs(frame, rx_grid, est_n0, o_idx)
            cv_br, cv_pl, _ = _build_group_c_bank_normalized(
                frame,
                rx_cv,
                pilot_values,
                n0_cv,
                s_cv,
                o_idx,
                cfg=self.config,
                pool_summary=self.pool_summary,
                include_temporal_branches=False,
            )
            cv_sc, cv_cx, cv_coords, cv_pl_out, _, _ = _extract_network_features(
                frame,
                rx_cv,
                pilot_values,
                n0_cv,
                s_cv,
                o_idx,
                cv_br,
                cv_pl,
                cfg=self.config,
            )
            cv_alpha = self._predict_branch_weights(cv_sc, cv_cx, cv_coords, cv_pl_out)
            v_floor = max(1e-12 * (s_cv ** 2), 1e-300)
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

        eval_idx = sorted(collected.keys())
        pred = np.asarray([collected[i][0] for i in eval_idx], dtype=np.complex128)
        var = np.asarray([collected[i][1] for i in eval_idx], dtype=np.float64)
        return pred, var, eval_idx

    def run_frame(
        self,
        frame: RealOFDMFrame,
        context_pool: Optional[Sequence[RealOFDMFrame]] = None,
        *,
        branches: Optional[Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]]] = None,
        return_heldout_cv: bool = True,
    ) -> ReceiverOutput:
        """Run Group C inference on one RealOFDMFrame (zero tx_bits/tx_symbols/SINR access)."""
        t0 = time.perf_counter()
        self.net.eval()

        try:
            rx_grid, est_n0, pilot_values = _validate_frame_inputs(frame)
            active_mask = np.asarray(frame.pilot_mask | frame.data_mask, dtype=bool)
            s_rms = float(np.sqrt(np.mean(np.abs(rx_grid[active_mask]) ** 2)))
            if not np.isfinite(s_rms):
                raise ValueError("Non-finite active RMS power")
            s_scale = s_rms if s_rms > 0.0 else max(math.sqrt(est_n0), 1.0)
            rx_norm = rx_grid / s_scale
            n0_norm = est_n0 / (s_scale ** 2)

            if context_pool is not None and len(context_pool) > 0 and self.config.include_global_branch:
                active_pool = _build_pilot_pool_summary(
                    context_pool,
                    max_pool_size=self.config.max_pool_size,
                    group_a_cfg=self.config.group_a_cfg,
                )
            else:
                active_pool = self.pool_summary

            all_obs = list(range(14))
            if branches is not None:
                used_branches = list(branches)
                sc, cx, p_coords, p_logits, _, _ = _extract_network_features(
                    frame,
                    rx_norm,
                    pilot_values,
                    n0_norm,
                    s_scale,
                    all_obs,
                    used_branches,
                    None,
                    cfg=self.config,
                )
                prior_alpha = _scores_to_alpha(p_logits, 1.0)
                alpha = self._predict_branch_weights(sc, cx, p_coords, p_logits)
                out = fuse_candidate_branches(
                    frame,
                    branches=used_branches,
                    branch_weights=alpha,
                    mode="exact_mixture_likelihood",
                    include_constellation_var=False,
                    n0_scale=float(self.config.n0_scale),
                    config=self.config.group_b_cfg,
                    return_heldout_cv=return_heldout_cv,
                )
            else:
                used_branches, p_logits, prior_alpha = _build_group_c_bank_normalized(
                    frame,
                    rx_norm,
                    pilot_values,
                    n0_norm,
                    s_scale,
                    all_obs,
                    cfg=self.config,
                    pool_summary=active_pool,
                    include_temporal_branches=True,
                )
                sc, cx, p_coords, p_logits_out, _, _ = _extract_network_features(
                    frame,
                    rx_norm,
                    pilot_values,
                    n0_norm,
                    s_scale,
                    all_obs,
                    used_branches,
                    p_logits,
                    cfg=self.config,
                )
                alpha = self._predict_branch_weights(sc, cx, p_coords, p_logits_out)
                out = fuse_candidate_branches(
                    frame,
                    branches=used_branches,
                    branch_weights=alpha,
                    mode="exact_mixture_likelihood",
                    include_constellation_var=False,
                    n0_scale=float(self.config.n0_scale),
                    config=self.config.group_b_cfg,
                    return_heldout_cv=False,
                )
                if return_heldout_cv:
                    pred, var, eval_idx = self._compute_heldout_cv(frame, rx_grid, pilot_values, est_n0)
                    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(var))):
                        raise ValueError("Held-out pilot predictions contain non-finite values")
                    out.heldout_pilot_pred = pred
                    out.heldout_pilot_var = var
                    out.aux_info["heldout_pilot_indices"] = eval_idx

            # Enforce strict h_var > 0.0 with normalized-gauge floor (C*-equivariant even at 1e-15)
            v_floor = max(1e-12 * (s_scale ** 2), 1e-300)
            out.h_var = np.maximum(np.asarray(out.h_var, dtype=np.float64), v_floor).astype(np.float64)

            latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
            out.aux_info.update(
                {
                    "mode": "group_c_trainable",
                    "architecture": str(self.config.architecture),
                    "param_count": int(self.param_count),
                    "seed": int(self.seed),
                    "train_time_s": float(self.train_time_s),
                    "best_epoch": int(self.best_epoch),
                    "n_candidates": int(len(used_branches)),
                    "branch_weights": [float(x) for x in alpha.tolist()],
                    "prior_weights": [float(x) for x in prior_alpha.tolist()],
                    "omega_t_candidates": sorted(
                        set(float(b.omega_t) for b in used_branches if isinstance(b, CandidateBranch))
                    )
                    or [0.0],
                    "versions": _runtime_versions(),
                    "latency_ms": float(latency_ms),
                    "failed": False,
                    "error": None,
                }
            )
            return out
        except Exception as exc:  # noqa: BLE001
            latency_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
            return _failed_output(
                f"{type(exc).__name__}: {exc}",
                architecture=self.config.architecture,
                param_count=self.param_count,
                seed=self.seed,
                train_time_s=self.train_time_s,
                latency_ms=latency_ms,
            )

    def save(self, path: Union[str, Path]) -> Path:
        """Save lightweight (< 1 MB) Group C checkpoint with weights_only=True compatibility."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        pool_tensors: Optional[dict[str, Any]] = None
        if self.pool_summary is not None:
            pool_tensors = {
                "h_p_unit": torch.from_numpy(np.asarray(self.pool_summary["h_p_unit"], dtype=np.complex128)).cpu(),
                "sig2_unit": torch.from_numpy(np.asarray(self.pool_summary["sig2_unit"], dtype=np.float64)).cpu(),
                "frame_ids": [int(x) for x in self.pool_summary.get("frame_ids", [])],
                "group_ids": [str(x) for x in self.pool_summary.get("group_ids", [])],
            }
        payload = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "seed": int(self.seed),
            "train_time_s": float(self.train_time_s),
            "best_epoch": int(self.best_epoch),
            "param_count": int(self.param_count),
            "versions": _runtime_versions(),
            "state_dict": cpu_state,
            "pool_summary": pool_tensors,
        }
        torch.save(payload, out_path)
        return out_path

    save_checkpoint = save

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GroupCTrainableReceiver":
        """Reload Group C checkpoint with bit-exact inference parity."""
        in_path = Path(path)
        try:
            payload = torch.load(in_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(in_path, map_location="cpu")

        cfg = GroupCConfig.from_dict(dict(payload["config"]))
        obj = cls(config=cfg, seed=int(payload["seed"]))
        obj.net.load_state_dict(payload["state_dict"], strict=True)
        obj.net.eval()
        obj.train_time_s = float(payload.get("train_time_s", 0.0))
        obj.best_epoch = int(payload.get("best_epoch", 0))

        ps = payload.get("pool_summary")
        if isinstance(ps, dict) and "h_p_unit" in ps and "sig2_unit" in ps:
            obj.pool_summary = {
                "h_p_unit": ps["h_p_unit"].detach().cpu().numpy().astype(np.complex128),
                "sig2_unit": ps["sig2_unit"].detach().cpu().numpy().astype(np.float64),
                "frame_ids": [int(x) for x in ps.get("frame_ids", [])],
                "group_ids": [str(x) for x in ps.get("group_ids", [])],
            }
        else:
            obj.pool_summary = None
        return obj

    load_checkpoint = load


GroupCReceiver = GroupCTrainableReceiver


def run_frame(
    frame: RealOFDMFrame,
    context_pool: Optional[Sequence[RealOFDMFrame]] = None,
    *,
    model: Optional[GroupCTrainableReceiver] = None,
    config: Optional[GroupCConfig] = None,
    seed: int = 20261121,
    branches: Optional[Sequence[Union[CandidateBranch, tuple[np.ndarray, np.ndarray]]]] = None,
    return_heldout_cv: bool = True,
) -> ReceiverOutput:
    """Module-level convenience wrapper for Group C inference."""
    rx = model if model is not None else GroupCTrainableReceiver(config=config, seed=seed, context_pool=context_pool)
    return rx.run_frame(
        frame,
        context_pool=context_pool,
        branches=branches,
        return_heldout_cv=return_heldout_cv,
    )


__all__ = [
    "BranchWeightAttentionNet",
    "GROUP_C_MODES",
    "GroupCConfig",
    "GroupCReceiver",
    "GroupCTrainableReceiver",
    "differentiable_mixture_llrs_torch",
    "run_frame",
]
