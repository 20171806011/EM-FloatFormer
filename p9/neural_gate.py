"""P9-C: Strictly C*-Invariant Learnable Temporal History-Trust Attention Gate.

Implements public-only feature extraction and the 4 exact parameter-matched variants:
  1. 'transport_attn'   : Signed U(1) fiber parallel-transport + multi-scale geodesic temporal
                          attention across all T=20 rows with exact C*-invariant Jeffreys bias.
  2. 'abs_no_transport' : Exact integer parameter parity ablation (|q^H k| + real cosine
                          coordinate bias + untransported weights @ v).
  3. 'local'            : Exact integer parameter parity local 3-row (dt in {-1, 0, +1})
                          temporal neighborhood operator (zero non-local reach).
  4. 'no_history'       : Exact same 'transport_attn' architecture & parameter count with
                          Frame-A 'inherit' features masked/replaced by 'clear'.

Strict invariants enforced by construction:
  - ZERO access to hidden truth or obs.y_b[PROBE_ROW] (row 19 observation masked to 0 first).
  - Step-0 centering via frozen initial_map so untrained model matches classical pre-update
    Bayes gate bit-for-bit (< 1e-14) while preserving 100% live gradients from step 0.
  - Strict C* gauge invariance (< 1e-12) on w[T] & mix_llrs and equivariance on mix_probe_pred,
    mix_probe_var, mix_mean_ray, mix_var_ray across arbitrary complex gain scaling alpha in C*.
  - Float64 / Complex128 deterministic arithmetic; NEVER uses torch.inference_mode().
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
from typing import Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from p8c.link import PILOTS as SC_PILOTS
from p8c.receiver import DELAYED_INDEX
from p9.contract import (
    N,
    PROBE_ROW,
    T,
    CandidateSummary,
    GateOutput,
    PublicObservation,
    jeffreys_complex_gaussian,
    mix_bit_llrs,
)


REAL = torch.float64
COMPLEX = torch.complex128
VARIANTS = ("transport_attn", "abs_no_transport", "local", "no_history")

C_SCALAR = 14
C_COMPLEX = 10
BASE_LOGIT_CLIP = 35.0


@dataclass(frozen=True)
class GateFeatures:
    """Strictly public, zero-probe-leakage, C*-canonical per-row features for T rows."""

    t_norm: np.ndarray        # [T] float64 in [-1, 1]
    is_train: np.ndarray      # [T] float64 in {0, 1}
    is_probe: np.ndarray      # [T] float64 in {0, 1}
    scalar_feats: np.ndarray  # [T, C_SCALAR] float64 (strictly C*-invariant)
    complex_feats: np.ndarray # [T, C_COMPLEX] complex128 (U(1)-equivariant after 1/s_mag)
    ev_base: np.ndarray       # [T] float64 (strictly C*-invariant Jeffreys + evidence map)
    base_logit: np.ndarray    # [T] float64 (exact classical pre-update Bayes-factor logit)


def _soft_qam_moments(llrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Gray 16-QAM soft symbol mean and variance from bit LLRs (positive = bit 1)."""
    l = np.clip(np.asarray(llrs, dtype=np.float64), -40.0, 40.0)
    p = 1.0 / (1.0 + np.exp(-l))
    inv_sqrt10 = 1.0 / math.sqrt(10.0)
    pa_r, pb_r = p[..., 0], p[..., 1]
    pa_i, pb_i = p[..., 2], p[..., 3]
    mu_r = inv_sqrt10 * (2.0 * pa_r - 1.0) * (3.0 - 2.0 * pb_r)
    mu_i = inv_sqrt10 * (2.0 * pa_i - 1.0) * (3.0 - 2.0 * pb_i)
    ex2_r = 0.1 * (9.0 - 8.0 * pb_r)
    ex2_i = 0.1 * (9.0 - 8.0 * pb_i)
    var_sym = np.maximum(ex2_r - mu_r * mu_r, 0.0) + np.maximum(ex2_i - mu_i * mu_i, 0.0)
    return mu_r + 1j * mu_i, var_sym


def _bit_entropy_and_tanh(llrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically stable binary entropy in [0, 1] bits and soft bit mean tanh(L/2)."""
    l = np.clip(np.asarray(llrs, dtype=np.float64), -40.0, 40.0)
    al = np.abs(l)
    p_min = 1.0 / (1.0 + np.exp(al))
    ent = (np.log1p(np.exp(-al)) + al * p_min) / math.log(2.0)
    return ent, np.tanh(0.5 * l)


def _soft_sc_predictive(
    mean_ray_norm: np.ndarray,
    var_ray_norm: np.ndarray,
    sym_mean: np.ndarray,
    sym_var: np.ndarray,
    n0_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row soft CP-SC predictive waveform mean & total variance from branch soft symbols."""
    # mean_ray_norm, var_ray_norm: [3, T, N]; sym_mean, sym_var: [T, N]
    # DELAYED_INDEX: [3, N] where DELAYED_INDEX[l, n] = (n - d_l) % N
    x_del = np.transpose(sym_mean[:, DELAYED_INDEX], (1, 0, 2))      # [3, T, N]
    vx_del = np.transpose(sym_var[:, DELAYED_INDEX], (1, 0, 2))      # [3, T, N]
    y_hat = (mean_ray_norm * x_del).sum(axis=0)                      # [T, N]
    ex2_del = np.abs(x_del) ** 2 + vx_del
    v_y = (
        n0_norm
        + (np.abs(mean_ray_norm) ** 2 * vx_del + var_ray_norm * ex2_del).sum(axis=0)
    )
    return y_hat, np.maximum(v_y, 1e-300)


def extract_gate_features(
    obs: PublicObservation,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
) -> GateFeatures:
    """Extract strictly public, C*-invariant/U(1)-equivariant per-row features with ZERO probe leakage.

    obs.y_b[obs.probe_row] is overwritten with 0 before any calculation so row-19 observations
    can never leak into any feature or canonical scale.
    """
    probe_row = int(obs.probe_row)
    train_rows = tuple(int(r) for r in obs.training_rows)
    if probe_row in train_rows:
        raise ValueError("probe_row must never appear in training_rows")

    # Immediately sanitize withheld probe row before touching y_b anywhere.
    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    y_b_safe[probe_row] = 0.0 + 0.0j

    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    is_train = np.zeros(T, dtype=np.float64)
    is_train[list(train_rows)] = 1.0
    is_probe = np.zeros(T, dtype=np.float64)
    is_probe[probe_row] = 1.0
    is_data = 1.0 - is_train - is_probe

    # Canonical purely-real positive gauge scale from global public n0 (zero row coupling!).
    n0 = float(obs.n0)
    if n0 > 0.0:
        s_mag = math.sqrt(n0)
    else:
        tr_energy = float(np.mean(np.abs(y_b_safe[list(train_rows)]) ** 2)) if train_rows else 1.0
        s_mag = math.sqrt(max(tr_energy, 1e-300))
    s_var = s_mag * s_mag
    n0_norm = n0 / s_var if n0 > 0.0 else 0.0

    # Normalize all complex fields by 1/s_mag (leaving pure U(1) phase e^{i phi}) and variances by 1/s_var.
    y_norm = y_b_safe / s_mag
    mu_ray_inh = np.asarray(cand_inh.mean_ray, dtype=np.complex128) / s_mag
    mu_ray_clr = np.asarray(cand_clr.mean_ray, dtype=np.complex128) / s_mag
    var_ray_inh = np.maximum(np.asarray(cand_inh.var_ray, dtype=np.float64) / s_var, 1e-300)
    var_ray_clr = np.maximum(np.asarray(cand_clr.var_ray, dtype=np.float64) / s_var, 1e-300)

    pre_pred_inh = np.asarray(cand_inh.pre_train_pred, dtype=np.complex128) / s_mag
    pre_pred_clr = np.asarray(cand_clr.pre_train_pred, dtype=np.complex128) / s_mag
    pre_var_inh = np.maximum(np.asarray(cand_inh.pre_train_var, dtype=np.float64) / s_var, 1e-300)
    pre_var_clr = np.maximum(np.asarray(cand_clr.pre_train_var, dtype=np.float64) / s_var, 1e-300)

    probe_pred_inh = np.asarray(cand_inh.probe_pred, dtype=np.complex128) / s_mag
    probe_pred_clr = np.asarray(cand_clr.probe_pred, dtype=np.complex128) / s_mag
    probe_var_inh = np.maximum(np.asarray(cand_inh.probe_var, dtype=np.float64) / s_var, 1e-300)
    probe_var_clr = np.maximum(np.asarray(cand_clr.probe_var, dtype=np.float64) / s_var, 1e-300)

    # Exact C*-invariant pre-update Bayes factor (broadcast from training rows only).
    raw_bf = float(cand_inh.pre_train_loglik) - float(cand_clr.pre_train_loglik)
    base_logit_val = float(np.clip(raw_bf, -BASE_LOGIT_CLIP, BASE_LOGIT_CLIP))
    base_logit = np.full(T, base_logit_val, dtype=np.float64)
    n_train_samples = max(len(train_rows) * N, 1)
    pre_ll_broadcast = np.full(T, math.asinh(raw_bf / n_train_samples), dtype=np.float64)

    # Soft symbol moments & LLR diagnostics (zero labels; zeroed on non-data rows).
    llrs_inh = np.asarray(cand_inh.llrs, dtype=np.float64)
    llrs_clr = np.asarray(cand_clr.llrs, dtype=np.float64)
    sym_inh, vsym_inh = _soft_qam_moments(llrs_inh)
    sym_clr, vsym_clr = _soft_qam_moments(llrs_clr)
    ent_inh, tanh_inh = _bit_entropy_and_tanh(llrs_inh)
    ent_clr, tanh_clr = _bit_entropy_and_tanh(llrs_clr)

    llr_conf_inh = is_data * (1.0 - ent_inh.mean(axis=(1, 2)))
    llr_conf_clr = is_data * (1.0 - ent_clr.mean(axis=(1, 2)))
    llr_agree = is_data * (tanh_inh * tanh_clr).mean(axis=(1, 2))

    # Soft data-row predictive distributions (public branch soft symbol means only).
    yhat_inh, vy_inh = _soft_sc_predictive(mu_ray_inh, var_ray_inh, sym_inh, vsym_inh, n0_norm)
    yhat_clr, vy_clr = _soft_sc_predictive(mu_ray_clr, var_ray_clr, sym_clr, vsym_clr, n0_norm)

    # Unified per-row predictive distributions across all T rows:
    #   - training rows: exact pre-update predictions/variances
    #   - data rows:     soft decision-directed predictions/variances
    #   - probe row:     branch withheld probe predictions/variances (zero y_b[probe_row]!)
    pred_inh_all = yhat_inh.copy()
    pred_clr_all = yhat_clr.copy()
    var_inh_all = vy_inh.copy()
    var_clr_all = vy_clr.copy()
    for k, r in enumerate(train_rows):
        pred_inh_all[r] = pre_pred_inh[k]
        pred_clr_all[r] = pre_pred_clr[k]
        var_inh_all[r] = pre_var_inh[k]
        var_clr_all[r] = pre_var_clr[k]
    pred_inh_all[probe_row] = probe_pred_inh
    pred_clr_all[probe_row] = probe_pred_clr
    var_inh_all[probe_row] = probe_var_inh
    var_clr_all[probe_row] = probe_var_clr

    # Observation residuals (STRICTLY masked to 0 on probe_row).
    obs_mask = (1.0 - is_probe)[:, None]  # [T, 1]
    res_inh = (y_norm - pred_inh_all) * obs_mask
    res_clr = (y_norm - pred_clr_all) * obs_mask

    e_inh = (np.abs(res_inh) ** 2 / np.maximum(var_inh_all, 1e-300)).mean(axis=-1) * (1.0 - is_probe)
    e_clr = (np.abs(res_clr) ** 2 / np.maximum(var_clr_all, 1e-300)).mean(axis=-1) * (1.0 - is_probe)
    res_ratio_inh = np.log1p(np.maximum(e_inh, 0.0))
    res_ratio_clr = np.log1p(np.maximum(e_clr, 0.0))

    # Per-row observation Gaussian log-likelihood difference (strictly C*-invariant, 0 on probe_row).
    ll_obs_diff = (
        (-np.log(var_inh_all / var_clr_all) - (np.abs(res_inh) ** 2) / var_inh_all + (np.abs(res_clr) ** 2) / var_clr_all)
        .mean(axis=-1)
        * (1.0 - is_probe)
    )
    res_ll_diff = np.arcsinh(ll_obs_diff)

    # Exact C*-invariant per-row Jeffreys divergence (predictive & ray-level) across all T rows.
    dj_pred_raw = jeffreys_complex_gaussian(pred_inh_all, var_inh_all, pred_clr_all, var_clr_all).mean(axis=-1)
    dj_pred = np.log1p(np.maximum(dj_pred_raw, 0.0))
    dj_ray_raw = jeffreys_complex_gaussian(mu_ray_inh, var_ray_inh, mu_ray_clr, var_ray_clr).mean(axis=(0, 2))
    dj_ray = np.log1p(np.maximum(dj_ray_raw, 0.0))

    # Per-row branch predictive & ray uncertainty log-ratios.
    log_var_ratio_pred = np.clip(
        np.log(np.maximum(var_inh_all.mean(axis=-1), 1e-300) / np.maximum(var_clr_all.mean(axis=-1), 1e-300)),
        -10.0,
        10.0,
    )
    log_var_ratio_ray = np.clip(
        np.log(
            np.maximum(var_ray_inh.sum(axis=0).mean(axis=-1), 1e-300)
            / np.maximum(var_ray_clr.sum(axis=0).mean(axis=-1), 1e-300)
        ),
        -10.0,
        10.0,
    )

    scalar_feats = np.stack(
        (
            t_norm,
            is_train,
            is_probe,
            pre_ll_broadcast,
            res_ratio_inh,
            res_ratio_clr,
            res_ll_diff,
            dj_pred,
            dj_ray,
            log_var_ratio_pred,
            log_var_ratio_ray,
            llr_conf_inh,
            llr_conf_clr,
            llr_agree,
        ),
        axis=-1,
    ).astype(np.float64)

    # Complex U(1)-equivariant per-row field embeddings [T, 10]:
    #   0..2: cand_inh coherent per-row ray gain means (3 rays)
    #   3..5: cand_clr coherent per-row ray gain means (3 rays)
    #   6..7: cand_inh & cand_clr ray-0 matched-filter residual phasors (0 on probe_row)
    #   8..9: cand_inh & cand_clr ray-1 matched-filter residual phasors (0 on probe_row)
    g_inh = mu_ray_inh.mean(axis=-1).T  # [T, 3]
    g_clr = mu_ray_clr.mean(axis=-1).T  # [T, 3]

    # Reference symbol sequence per row: known SC_PILOTS on training rows, soft symbol mean on data rows, 0 on probe.
    x_ref_inh = sym_inh.copy()
    x_ref_clr = sym_clr.copy()
    for r in train_rows:
        x_ref_inh[r] = SC_PILOTS[r]
        x_ref_clr[r] = SC_PILOTS[r]
    x_ref_inh[probe_row] = 0.0 + 0.0j
    x_ref_clr[probe_row] = 0.0 + 0.0j

    rho_inh_0 = (res_inh * x_ref_inh[:, DELAYED_INDEX[0]].conj()).mean(axis=-1)
    rho_clr_0 = (res_clr * x_ref_clr[:, DELAYED_INDEX[0]].conj()).mean(axis=-1)
    rho_inh_1 = (res_inh * x_ref_inh[:, DELAYED_INDEX[1]].conj()).mean(axis=-1)
    rho_clr_1 = (res_clr * x_ref_clr[:, DELAYED_INDEX[1]].conj()).mean(axis=-1)

    complex_feats = np.concatenate(
        (
            g_inh,
            g_clr,
            np.stack((rho_inh_0, rho_clr_0, rho_inh_1, rho_clr_1), axis=-1),
        ),
        axis=-1,
    ).astype(np.complex128)

    # Attention key evidence map (strictly C*-invariant; nonzero even in no_history via is_train/is_probe).
    ev_base = dj_pred + 0.5 * dj_ray + is_train - 2.0 * is_probe

    return GateFeatures(
        t_norm=t_norm,
        is_train=is_train,
        is_probe=is_probe,
        scalar_feats=scalar_feats,
        complex_feats=complex_feats,
        ev_base=ev_base,
        base_logit=base_logit,
    )


def compose_gate_output(
    history_weight: np.ndarray,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    probe_row: int = PROBE_ROW,
) -> GateOutput:
    """Assemble unified GateOutput with exact probability-space LLR mixing and within+between variance."""
    w = np.clip(np.asarray(history_weight, dtype=np.float64).reshape(T), 0.0, 1.0)
    w_probe = float(w[int(probe_row)])

    mix_llrs = mix_bit_llrs(w[:, None, None], cand_inh.llrs, cand_clr.llrs)

    mu_p1 = np.asarray(cand_inh.probe_pred, dtype=np.complex128)
    mu_p0 = np.asarray(cand_clr.probe_pred, dtype=np.complex128)
    v_p1 = np.asarray(cand_inh.probe_var, dtype=np.float64)
    v_p0 = np.asarray(cand_clr.probe_var, dtype=np.float64)
    mix_probe_pred = w_probe * mu_p1 + (1.0 - w_probe) * mu_p0
    mix_probe_var = (
        w_probe * v_p1
        + (1.0 - w_probe) * v_p0
        + w_probe * (1.0 - w_probe) * (np.abs(mu_p1 - mu_p0) ** 2)
    )

    w_ray = w[None, :, None]
    mu_r1 = np.asarray(cand_inh.mean_ray, dtype=np.complex128)
    mu_r0 = np.asarray(cand_clr.mean_ray, dtype=np.complex128)
    v_r1 = np.asarray(cand_inh.var_ray, dtype=np.float64)
    v_r0 = np.asarray(cand_clr.var_ray, dtype=np.float64)
    mix_mean_ray = w_ray * mu_r1 + (1.0 - w_ray) * mu_r0
    mix_var_ray = (
        w_ray * v_r1
        + (1.0 - w_ray) * v_r0
        + w_ray * (1.0 - w_ray) * (np.abs(mu_r1 - mu_r0) ** 2)
    )

    return GateOutput(
        history_weight=w,
        probe_weight=w_probe,
        mix_llrs=mix_llrs,
        mix_probe_pred=mix_probe_pred,
        mix_probe_var=mix_probe_var,
        mix_mean_ray=mix_mean_ray,
        mix_var_ray=mix_var_ray,
    )


def classical_pre_update_gate(
    obs: PublicObservation,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
) -> GateOutput:
    """Exact classical pre-update Bayes-factor gate (equal prior odds, scalar weight broadcast to T)."""
    raw_bf = float(cand_inh.pre_train_loglik) - float(cand_clr.pre_train_loglik)
    clipped = float(np.clip(raw_bf, -BASE_LOGIT_CLIP, BASE_LOGIT_CLIP))
    w_val = 1.0 / (1.0 + math.exp(-clipped))
    return compose_gate_output(np.full(T, w_val, dtype=np.float64), cand_inh, cand_clr, probe_row=obs.probe_row)


def mix_bit_llrs_torch_from_logit(
    logit_w: torch.Tensor,
    llr_inh: torch.Tensor,
    llr_clr: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable log-space probability mixture of branch bit LLRs (zero clamp dead-zone)."""
    # logit_w: [B, T, 1, 1] (or broadcastable); llr_inh, llr_clr: [B, T, N, 4]
    log_w = -F.softplus(-logit_w)
    log_1mw = -F.softplus(logit_w)
    l1 = llr_inh.clamp(-40.0, 40.0)
    l0 = llr_clr.clamp(-40.0, 40.0)
    log_p1_1 = -F.softplus(-l1)
    log_p1_0 = -F.softplus(l1)
    log_p0_1 = -F.softplus(-l0)
    log_p0_0 = -F.softplus(l0)
    log_pm_1 = torch.logaddexp(log_w + log_p1_1, log_1mw + log_p0_1)
    log_pm_0 = torch.logaddexp(log_w + log_p1_0, log_1mw + log_p0_0)
    return log_pm_1 - log_pm_0


# ---------------------------------------------------------------------------
# Differentiable C*-Equivariant / C*-Invariant Neural Temporal Attention Core
# ---------------------------------------------------------------------------


def _cmat(rows: int, cols: int) -> nn.Parameter:
    scale = 1.0 / math.sqrt(2.0 * rows)
    real = torch.randn(rows, cols, dtype=REAL) * scale
    imag = torch.randn(rows, cols, dtype=REAL) * scale
    return nn.Parameter(torch.complex(real, imag))


def _rmat(rows: int, cols: int) -> nn.Parameter:
    return nn.Parameter(torch.randn(rows, cols, dtype=REAL) / math.sqrt(rows))


def _safe_abs(z: torch.Tensor, eps: float = 1e-24) -> torch.Tensor:
    """Smooth zero-safe modulus for already dimensionless/normalized complex tensors."""
    return torch.sqrt(z.real.square() + z.imag.square() + eps)


def _safe_rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable zero-safe RMS magnitude (never differentiates sqrt(0))."""
    p = (x.real.square() + x.imag.square()).mean(dim=dim, keepdim=keepdim)
    safe = torch.where(p > 0.0, p, torch.ones_like(p))
    rms = safe.sqrt()
    unit = torch.where(p > 0.0, x / rms, torch.zeros_like(x))
    return unit, rms


class TemporalTransportLayer(nn.Module):
    """Parameter-matched 1D temporal layer over T=20 rows supporting 3 exact topologies:
      - 'transport_attn'   : Signed Re(q'^H k') + U(1) fiber parallel transport on Q, K, AND V
                             + multi-scale geodesic distance decay + Jeffreys evidence bias.
      - 'abs_no_transport' : Exact same parameter tensors ablated to |q^H k| + real cosine
                             coordinate bias + untransported weights @ v.
      - 'local'            : Exact same parameter tensors restricted strictly to 3-row
                             neighborhood (dt in {-1, 0, +1}, zero non-local reach).
    """

    def __init__(self, variant: str, d_model: int = 12, d_scalar: int = 8, n_heads: int = 4, d_ff: int = 6):
        super().__init__()
        if variant not in ("transport_attn", "abs_no_transport", "local"):
            raise ValueError(f"Unsupported layer variant: {variant}")
        if d_model % n_heads != 0 or d_scalar % n_heads != 0:
            raise ValueError("d_model and d_scalar must both be divisible by n_heads")

        self.variant = variant
        self.d_model = d_model
        self.d_scalar = d_scalar
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_s_head = d_scalar // n_heads

        # Identical parameter tensors across all variants for exact integer parameter parity.
        self.w_q = _cmat(d_model, d_model)
        self.w_k = _cmat(d_model, d_model)
        self.w_v = _cmat(d_model, d_model)
        self.w_o = _cmat(d_model, d_model)

        self.w_sv = _rmat(d_scalar, d_scalar)
        self.w_so = _rmat(d_scalar, d_scalar)

        # Symmetric +/- Doppler passband & true multi-scale geodesic initialization in t_norm in [-1, 1]:
        # Since 1 row step has dt_norm^2 = (2/19)^2 = 0.0111, beta~20..25 gives sharp local (+/-2 rows),
        # beta~2.5 gives medium (+/-6 rows), and beta~0.13 gives global full-frame (20 rows) reach.
        if n_heads == 4:
            init_omega = torch.tensor([-4.8, 4.2, -1.6, 1.4], dtype=REAL)
            init_raw_beta = torch.tensor([25.0, 20.0, 2.5, -2.0], dtype=REAL)
        else:
            init_omega = torch.linspace(-5.0, 5.0, n_heads, dtype=REAL)
            init_raw_beta = torch.linspace(25.0, -2.0, n_heads, dtype=REAL)
        self.omega = nn.Parameter(init_omega)
        self.raw_beta = nn.Parameter(init_raw_beta)
        self.gamma_ev = nn.Parameter(torch.full((n_heads,), 0.5, dtype=REAL))
        self.ev_proj = _rmat(d_scalar, n_heads)

        # C*-equivariant gated complex FFN + C*-invariant scalar FFN.
        self.ff1 = _cmat(d_model, d_ff)
        self.ff2 = _cmat(d_ff, d_model)
        self.s_ff1 = _rmat(d_scalar, d_scalar)
        self.s_fb1 = nn.Parameter(torch.zeros(d_scalar, dtype=REAL))
        self.s_ff2 = _rmat(d_scalar, d_scalar)

    def _split_c(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        return x.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)

    def _merge_c(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Dh] -> [B, T, D]
        return x.transpose(1, 2).flatten(-2)

    def _split_s(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.n_heads, self.d_s_head)).transpose(1, 2)

    def _merge_s(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).flatten(-2)

    def _local_3row(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        vc: torch.Tensor,
        vs: torch.Tensor,
        t_norm: torch.Tensor,
        ev_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strictly local 3-row neighborhood (dt in {-1, 0, +1}) using explicit neighbor indexing."""
        b, h, t_len, _ = q.shape
        device = q.device
        t_idx = torch.arange(t_len, device=device)
        scale = 1.0 / math.sqrt(self.d_head)
        beta = F.softplus(self.raw_beta)  # [H]

        logits_list = []
        valid_list = []
        vc_list = []
        vs_list = []
        for dt in (-1, 0, 1):
            raw_ti = t_idx + dt
            valid = (raw_ti >= 0) & (raw_ti < t_len)          # [T]
            ti = raw_ti.clamp(0, t_len - 1)                   # [T]
            kn = k[:, :, ti, :]                               # [B, H, T, Dh]
            vcn = vc[:, :, ti, :]                             # [B, H, T, Dh]
            vsn = vs[:, :, ti, :]                             # [B, H, T, Dsh]

            d_t = t_norm[:, None, :] - t_norm[:, None, ti]    # [B, 1, T]
            ang = d_t * self.omega[None, :, None]             # [B, H, T]
            u_rel = torch.polar(torch.ones_like(ang), ang)[..., None]  # [B, H, T, 1]
            kn_t = kn * u_rel
            vcn_t = vcn * u_rel

            geo = -beta[None, :, None] * d_t.square()         # [B, H, T]
            sc = (q.conj() * kn_t).sum(dim=-1).real * scale + geo + ev_k[:, :, ti]
            v_bht = valid[None, None, :].expand(b, h, t_len)
            sc = torch.where(v_bht, sc, torch.full_like(sc, -1e9))
            logits_list.append(sc)
            valid_list.append(v_bht)
            vc_list.append(vcn_t)
            vs_list.append(vsn)

        logits = torch.stack(logits_list, dim=-1)             # [B, H, T, 3]
        valid_stack = torch.stack(valid_list, dim=-1)         # [B, H, T, 3]
        weights = logits.softmax(dim=-1)
        weights = torch.where(valid_stack, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-300)

        vc_stack = torch.stack(vc_list, dim=-2)               # [B, H, T, 3, Dh]
        vs_stack = torch.stack(vs_list, dim=-2)               # [B, H, T, 3, Dsh]
        agg_c = (weights[..., None].to(COMPLEX) * vc_stack).sum(dim=-2)
        agg_s = (weights[..., None] * vs_stack).sum(dim=-2)
        return agg_c, agg_s

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h:       [B, T, D] complex128 (U(1)-equivariant, raw amplitudes preserved)
          s:       [B, T, d_s] float64 (C*-invariant)
          t_norm:  [B, T] float64 in [-1, 1]
          ev_base: [B, T] float64 (C*-invariant Jeffreys + training/probe evidence map)
        """
        h_u, _ = _safe_rms(h, dim=-1, keepdim=True)
        q = self._split_c(h_u @ self.w_q)         # [B, H, T, Dh] unit-scale
        k = self._split_c(h_u @ self.w_k)         # [B, H, T, Dh] unit-scale
        vc = self._split_c(h @ self.w_v)          # [B, H, T, Dh] 1-homogeneous raw amplitude!
        vs = self._split_s(s @ self.w_sv)         # [B, H, T, Dsh]

        ev_k = (
            self.gamma_ev[None, :, None] * ev_base[:, None, :]
            + (s @ self.ev_proj).transpose(1, 2)
        )  # [B, H, T]

        if self.variant == "local":
            agg_c, agg_s = self._local_3row(q, k, vc, vs, t_norm, ev_k)
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            dt2 = (t_norm[:, :, None] - t_norm[:, None, :]).square()  # [B, T, T]
            beta = F.softplus(self.raw_beta)                          # [H]
            geo_bias = -beta[None, :, None, None] * dt2[:, None, :, :]

            angles = t_norm[:, None, :] * self.omega[None, :, None]   # [B, H, T]
            rho = torch.polar(torch.ones_like(angles), angles)[..., None]  # [B, H, T, 1]

            if self.variant == "transport_attn":
                # Signed Hermitian inner product + U(1) fiber parallel transport on Q, K, AND V!
                qp = q * rho.conj()
                kp = k * rho.conj()
                vp = vc * rho.conj()
                logits = (qp @ kp.mH).real * scale + geo_bias + ev_k[:, :, None, :]
                weights = logits.softmax(dim=-1)
                vp_real = torch.view_as_real(vp.resolve_conj()).flatten(-2)  # [B, H, T, 2*Dh]
                agg_real = weights @ vp_real
                agg_c = torch.view_as_complex(agg_real.unflatten(-1, (self.d_head, 2))) * rho
                agg_s = weights @ vs
            else:
                # 'abs_no_transport': unsigned |q^H k| + real cosine bias + untransported weights @ v
                pos_bias = 0.25 * (rho @ rho.mH).real
                logits = _safe_abs(q @ k.mH) * scale + pos_bias + geo_bias + ev_k[:, :, None, :]
                weights = logits.softmax(dim=-1)
                vc_real = torch.view_as_real(vc.resolve_conj()).flatten(-2)
                agg_real = weights @ vc_real
                agg_c = torch.view_as_complex(agg_real.unflatten(-1, (self.d_head, 2)))
                agg_s = weights @ vs

        delta_h = self._merge_c(agg_c) @ self.w_o
        delta_s = self._merge_s(agg_s) @ self.w_so

        # Residual + strictly C*-equivariant complex gated FFN.
        h1 = h + delta_h
        h1_u, h1_r = _safe_rms(h1, dim=-1, keepdim=True)
        ff_h = h1_u @ self.ff1
        ff_gate = _safe_abs(ff_h).sigmoid()
        h_out = h1 + 0.5 * ((ff_h * ff_gate) @ self.ff2) * h1_r

        # Residual + strictly C*-invariant scalar FFN.
        s1 = s + delta_s
        s_out = s1 + 0.5 * (F.silu(s1 @ self.s_ff1 + self.s_fb1) @ self.s_ff2)
        return h_out, s_out


class _TemporalTrustBackbone(nn.Module):
    """Core U(1)-equivariant + C*-invariant temporal backbone."""

    def __init__(
        self,
        layer_variant: str,
        d_model: int = 12,
        n_heads: int = 4,
        n_layers: int = 1,
        d_scalar: int = 8,
        d_ff: int = 6,
    ):
        super().__init__()
        self.w_c = _cmat(C_COMPLEX, d_model)
        self.w_s = _rmat(C_SCALAR, d_scalar)
        self.b_s = nn.Parameter(torch.zeros(d_scalar, dtype=REAL))
        self.w_s2c = _rmat(d_scalar, d_model)

        self.layers = nn.ModuleList(
            [
                TemporalTransportLayer(layer_variant, d_model=d_model, d_scalar=d_scalar, n_heads=n_heads, d_ff=d_ff)
                for _ in range(n_layers)
            ]
        )

        # Strictly C*-invariant readout combining Hermitian field contractions + quadratic forms + scalar state.
        self.w_read_ref = _cmat(d_model, 4)
        self.w_read_bilin = _cmat(d_model, 4)
        self.w_head = _rmat(d_scalar + 6, 2)
        self.b_head = nn.Parameter(torch.zeros(2, dtype=REAL))

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = F.silu(scalar_feats @ self.w_s + self.b_s)       # [B, T, d_s] C*-invariant
        mod = 1.0 + torch.tanh(s @ self.w_s2c)               # [B, T, D] C*-invariant real multiplier
        h = (complex_feats @ self.w_c) * mod.to(COMPLEX)     # [B, T, D] strictly U(1)-equivariant!

        for layer in self.layers:
            h, s = layer(h, s, t_norm, ev_base)

        # Reference fields: [g_inh_0, g_clr_0, rho_inh_0, rho_clr_0] (all U(1)-equivariant & nonzero in no_history).
        ref = torch.stack(
            (
                complex_feats[..., 0],
                complex_feats[..., 3],
                complex_feats[..., 6],
                complex_feats[..., 7],
            ),
            dim=-1,
        )  # [B, T, 4]
        u_ref = h @ self.w_read_ref
        inv_ref = torch.asinh((ref.conj() * u_ref).real)     # [B, T, 4] strictly C*-invariant

        u_b = h @ self.w_read_bilin                          # [B, T, 4]
        inv_bilin = torch.asinh((u_b[..., :2].conj() * u_b[..., 2:]).real)  # [B, T, 2] strictly C*-invariant

        head_in = torch.cat((s, inv_ref, inv_bilin), dim=-1) # [B, T, d_s + 6]
        out = head_in @ self.w_head + self.b_head            # [B, T, 2]
        return out[..., 0], out[..., 1]


class TemporalTrustGate(nn.Module):
    """Unified P9-C learnable C*-invariant temporal history-trust gate.

    Centered around a frozen step-0 copy (`initial_map`) so step-0 initialization reproduces
    the exact classical pre-update Bayes gate bit-for-bit (< 1e-14) while maintaining 100%
    active finite nonzero gradients for all trainable parameters from step 0.
    """

    def __init__(
        self,
        variant: str = "transport_attn",
        d_model: int = 12,
        n_heads: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown TemporalTrustGate variant '{variant}'. Expected one of {VARIANTS}.")
        self.variant = variant
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        layer_variant = "transport_attn" if variant == "no_history" else variant
        self.backbone = _TemporalTrustBackbone(
            layer_variant=layer_variant,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        # Frozen step-0 map for exact zero-residual centering (requires_grad=False -> 0 trainable params).
        self.initial_map = copy.deepcopy(self.backbone)
        for p in self.initial_map.parameters():
            p.requires_grad_(False)

    def real_parameters(self) -> int:
        """Exact integer count of trainable real-valued scalar degrees of freedom."""
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def forward_tensors(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable batched forward pass returning (history_weight [B, T], logit [B, T])."""
        raw_delta, raw_log_tau = self.backbone(scalar_feats, complex_feats, t_norm, ev_base)
        with torch.no_grad():
            delta_0, log_tau_0 = self.initial_map(
                scalar_feats.detach(),
                complex_feats.detach(),
                t_norm.detach(),
                ev_base.detach(),
            )
        delta_logit = raw_delta - delta_0
        log_tau = raw_log_tau - log_tau_0
        # Bounded positive attenuation/sharpening gate in (0, 2): exact 1.0 at step 0 (log_tau == 0).
        tau = 2.0 * torch.sigmoid(log_tau)
        logit = tau * base_logit + delta_logit
        w = torch.sigmoid(logit)
        return w, logit

    def extract_features_for_variant(
        self,
        obs: PublicObservation,
        cand_inh: CandidateSummary,
        cand_clr: CandidateSummary,
    ) -> GateFeatures:
        """Extract variant-appropriate features ('no_history' masks/replaces Frame-A inherit by clear)."""
        if self.variant == "no_history":
            return extract_gate_features(obs, cand_clr, cand_clr)
        return extract_gate_features(obs, cand_inh, cand_clr)

    def gate(
        self,
        obs: PublicObservation,
        cand_inh: CandidateSummary,
        cand_clr: CandidateSummary,
    ) -> GateOutput:
        """Inference entry point (uses torch.no_grad(), NEVER torch.inference_mode())."""
        feats = self.extract_features_for_variant(obs, cand_inh, cand_clr)
        with torch.no_grad():
            s_t = torch.from_numpy(feats.scalar_feats).to(REAL).unsqueeze(0)
            c_t = torch.from_numpy(feats.complex_feats).to(COMPLEX).unsqueeze(0)
            t_t = torch.from_numpy(feats.t_norm).to(REAL).unsqueeze(0)
            ev_t = torch.from_numpy(feats.ev_base).to(REAL).unsqueeze(0)
            bl_t = torch.from_numpy(feats.base_logit).to(REAL).unsqueeze(0)
            w_t, _ = self.forward_tensors(s_t, c_t, t_t, ev_t, bl_t)
            w_np = w_t.squeeze(0).cpu().numpy()
        return compose_gate_output(w_np, cand_inh, cand_clr, probe_row=obs.probe_row)
