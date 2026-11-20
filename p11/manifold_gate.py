"""P11: Exact-Parameter-Matched 2D Ray x Time Gauge-Equivariant Manifold Trust Gate.

Implements strictly public, zero-probe-leakage, C*-invariant & ray-permutation-equivariant
per-ray, per-row feature extraction and 4 exact integer-parameter-matched neural variants:
  1. 'em10_transport_attn'   : Global 2D (Ray x Time) gauge-equivariant parallel-transport wave
                               attention across all K_hat * T tokens with explicit signed U(1)
                               phase connection exp(-2j*pi*(s_nu*nu_k*(t-t') + omega_t*(t-t')
                               - (tau_k - tau_k')*df_eff)) + continuous (dt, dtau) geodesic bias.
  2. 'em10_abs_no_transport' : Exact integer parameter parity global 2D (Ray x Time) attention
                               dropping signed U(1) phase transport (|q^H k| + untransported v).
  3. 'em10_local'            : Exact integer parameter parity local 1D temporal kernel (size 3,
                               dt in {-1, 0, +1} within each ray only; ZERO cross-ray attention
                               and ZERO long-range time reach) + pointwise ray MLP.
  4. 'em10_no_history'       : Exact integer parameter parity 'em10_transport_attn' with all
                               Frame-A history inputs strictly masked out.

Strict invariants enforced by construction:
  - ZERO hardcoded N0 anywhere (always uses explicit obs.n0).
  - ZERO access to obs.y_b[PROBE_ROW] (row 19 masked to 0 before any computation).
  - Exact step-0 identity (< 1e-13) to classical_per_ray_gate via frozen initial_map centering.
  - Strict C* = R+ x U(1) invariance (< 1e-12) on w_ray, noise_mult, mix_llrs and equivariance
    on H_mix, noise_mix, mix_probe_pred, mix_probe_var.
  - Exact ray-permutation equivariance (< 1e-12) across unordered discovered ray atoms.
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
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from p11.contract import (
    N,
    OFFGRID_DELAY_MAX,
    PROBE_ROW,
    T,
    OffGridObservation,
    jeffreys_complex_gaussian,
    unitary_dft_matrix,
)
from p11.offgrid_channel import (
    ROW_STEP,
    OffGridBranchBundle,
    OffGridGateResult,
    compose_offgrid_gate_output,
)


REAL = torch.float64
COMPLEX = torch.complex128
VARIANTS = (
    "em10_transport_attn",
    "em10_abs_no_transport",
    "em10_local",
    "em10_no_history",
)

C_SCALAR = 18
C_COMPLEX = 6

# Static constants for fast deterministic frequency-domain differentiable LMMSE + 16-QAM BCE.
_PAM_TORCH = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=REAL) / math.sqrt(10.0)
_F_H_TORCH = torch.from_numpy(np.ascontiguousarray(unitary_dft_matrix(N).conj().T)).to(COMPLEX)


@dataclass(frozen=True)
class ManifoldFeatures:
    """Strictly public, row-local, zero-probe-leakage, C*-canonical, ray-perm-equivariant features."""

    tau_hat: np.ndarray        # [K_hat] float64 continuous ray delays in [0.0, 6.5]
    nu_hat: np.ndarray         # [K_hat] float64 continuous ray Dopplers (Frame-B-only in no_history)
    t_phys: np.ndarray         # [T] float64 exact physical Frame-B row times (0.0, 1.5, ..., 28.5)
    t_norm: np.ndarray         # [T] float64 normalized row coordinates in [-1, 1]
    is_anchor: np.ndarray      # [T] float64 in {0, 1}
    is_probe: np.ndarray       # [T] float64 in {0, 1}
    scalar_feats: np.ndarray   # [K_hat, T, C_SCALAR] float64 strictly C*-invariant
    complex_feats: np.ndarray  # [K_hat, T, C_COMPLEX] complex128 strictly U(1)-equivariant
    ev_base: np.ndarray        # [K_hat, T] float64 strictly C*-invariant key evidence bias
    base_logit: np.ndarray     # [K_hat, T] float64 exact classical per-ray logit (0 in no_history)


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


def extract_manifold_features(
    obs: OffGridObservation,
    bundle: OffGridBranchBundle,
    *,
    mask_history: bool = False,
) -> ManifoldFeatures:
    """Extract strictly public, row-local, C*-canonical, ray-permutation-equivariant features.

    Guarantees:
      1. obs.y_b[obs.probe_row] is overwritten with 0 before any calculation (EXACT 0.0 probe leak).
      2. Row t's features NEVER read obs.y_b[t'] for t' != t (strictly row-local feature map!).
      3. Canonical scale s_mag = sqrt(obs.n0) -> exact C* invariance & zero cross-row scale coupling.
      4. Every feature channel treats ray index k symmetrically -> exact ray-permutation equivariance.
      5. When mask_history=True ('em10_no_history'), ALL Frame-A history inputs are strictly masked out.
    """
    if isinstance(obs.probe_row, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(obs.probe_row, (int, np.integer)):
        raise TypeError(f"obs.probe_row must be an exact integer, got {type(obs.probe_row).__name__}")
    probe_row = int(obs.probe_row)
    train_rows_list: list[int] = []
    for r in obs.training_rows:
        if isinstance(r, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(r, (int, np.integer)):
            raise TypeError(f"obs.training_rows entries must be exact integers, got {type(r).__name__}")
        train_rows_list.append(int(r))
    train_rows = tuple(train_rows_list)
    if probe_row < 0 or probe_row >= T or any(r < 0 or r >= T for r in train_rows) or probe_row in train_rows or not train_rows:
        raise ValueError(f"Invalid training_rows={train_rows} with probe_row={probe_row}")
    r_anchor = train_rows[0]

    if isinstance(obs.n0, (bool, np.bool_, np.timedelta64, np.datetime64, complex, np.complexfloating)):
        raise TypeError(f"obs.n0 must be a positive finite real number, got {type(obs.n0).__name__}")
    n0 = float(obs.n0)
    if n0 <= 0.0 or not math.isfinite(n0):
        raise ValueError(f"Expected positive finite obs.n0, got {n0!r}")
    s_mag = math.sqrt(n0)
    s_var = n0

    # Immediately sanitize withheld probe row before touching obs.y_b anywhere.
    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    y_b_safe[probe_row] = 0.0 + 0.0j
    if not np.all(np.isfinite(y_b_safe)):
        raise ValueError("obs.y_b non-probe rows contain non-finite values")
    if not np.all(np.isfinite(obs.pilot_tx_b)):
        raise ValueError("obs.pilot_tx_b contains non-finite values")
    y_u = y_b_safe / s_mag  # [T, N], unit-noise gauge (pure U(1) phase remains!)

    k_hat = bundle.k_hat
    t_phys = (np.arange(T, dtype=np.float64) * ROW_STEP).astype(np.float64)
    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    is_anchor = np.zeros(T, dtype=np.float64)
    is_anchor[list(train_rows)] = 1.0
    is_probe = np.zeros(T, dtype=np.float64)
    is_probe[probe_row] = 1.0
    is_data = 1.0 - is_anchor - is_probe
    obs_mask = 1.0 - is_probe  # [T], 0 on probe_row

    dist_anchor = (np.arange(T, dtype=np.float64) - float(r_anchor)) / float(T - 1)
    abs_dist_anchor = np.abs(dist_anchor)

    tau_hat = np.asarray(bundle.tau_hat, dtype=np.float64)
    tau_norm = tau_hat / float(OFFGRID_DELAY_MAX)
    d_mats = bundle.d_mats  # [K, N, N]
    d_abs2 = np.abs(d_mats) ** 2  # [K, N, N]

    # Select history-masked vs full Frame-A inputs.
    if mask_history:
        mu_inh_u = bundle.mean_clr / s_mag
        var_inh_u = np.maximum(bundle.var_clr / s_var, 1e-12)
        pre_pred_inh_u = bundle.pre_pred_clr / s_mag
        pre_var_inh_u = np.maximum(bundle.pre_var_clr / s_var, 1e-12)
        llrs_inh = bundle.llrs_clr
        nu_use = np.asarray(bundle.nu_b_proxy, dtype=np.float64)
        atom_snr_use = np.zeros(k_hat, dtype=np.float64)
        base_logit_ray_use = np.zeros(k_hat, dtype=np.float64)
    else:
        mu_inh_u = bundle.mean_inh / s_mag
        var_inh_u = np.maximum(bundle.var_inh / s_var, 1e-12)
        pre_pred_inh_u = bundle.pre_pred_inh / s_mag
        pre_var_inh_u = np.maximum(bundle.pre_var_inh / s_var, 1e-12)
        llrs_inh = bundle.llrs_inh
        nu_use = np.asarray(bundle.nu_hat, dtype=np.float64)
        atom_snr_use = np.asarray(bundle.atoms.atom_snr_a, dtype=np.float64)
        base_logit_ray_use = np.asarray(bundle.base_logit_ray, dtype=np.float64)

    mu_clr_u = bundle.mean_clr / s_mag
    var_clr_u = np.maximum(bundle.var_clr / s_var, 1e-12)
    pre_pred_clr_u = bundle.pre_pred_clr / s_mag
    pre_var_clr_u = np.maximum(bundle.pre_var_clr / s_var, 1e-12)
    llrs_clr = bundle.llrs_clr

    # Soft 16-QAM moments & blind confidence stats (zero labels, zeroed on non-data rows).
    sym_inh, vsym_inh = _soft_qam_moments(llrs_inh)
    sym_clr, vsym_clr = _soft_qam_moments(llrs_clr)
    ent_inh, _ = _bit_entropy_and_tanh(llrs_inh)
    ent_clr, _ = _bit_entropy_and_tanh(llrs_clr)
    llr_conf_inh = is_data * (1.0 - ent_inh.mean(axis=(1, 2)))
    llr_conf_clr = is_data * (1.0 - ent_clr.mean(axis=(1, 2)))
    row_llr_conf_diff = llr_conf_inh - llr_conf_clr  # [T]

    # Reference symbol sequence per row (strictly row-local, 0 on probe_row).
    x_ref_inh = sym_inh.copy()
    x_ref_clr = sym_clr.copy()
    vx_ref_inh = vsym_inh.copy()
    vx_ref_clr = vsym_clr.copy()
    for idx_tr, r in enumerate(train_rows):
        x_ref_inh[r] = obs.pilot_tx_b[idx_tr]
        x_ref_clr[r] = obs.pilot_tx_b[idx_tr]
        vx_ref_inh[r] = 0.0
        vx_ref_clr[r] = 0.0
    x_ref_inh[probe_row] = 0.0 + 0.0j
    x_ref_clr[probe_row] = 0.0 + 0.0j
    vx_ref_inh[probe_row] = 0.0
    vx_ref_clr[probe_row] = 0.0

    # Exact fractionally delayed reference waveforms & propagated symbol variances: [K, T, N].
    u_ref_inh = np.einsum("knm,tm->ktn", d_mats, x_ref_inh)
    u_ref_clr = np.einsum("knm,tm->ktn", d_mats, x_ref_clr)
    vx_del_inh = np.einsum("knm,tm->ktn", d_abs2, vx_ref_inh)
    vx_del_clr = np.einsum("knm,tm->ktn", d_abs2, vx_ref_clr)

    # Per-row soft predictive distributions (data rows), exact pre-update (anchor), exact probe pred (row 19).
    pred_inh = np.sum(mu_inh_u * u_ref_inh, axis=0)  # [T, N]
    pred_clr = np.sum(mu_clr_u * u_ref_clr, axis=0)  # [T, N]
    vy_inh = 1.0 + np.sum(
        (np.abs(mu_inh_u) ** 2) * vx_del_inh + var_inh_u * (np.abs(u_ref_inh) ** 2 + vx_del_inh),
        axis=0,
    )
    vy_clr = 1.0 + np.sum(
        (np.abs(mu_clr_u) ** 2) * vx_del_clr + var_clr_u * (np.abs(u_ref_clr) ** 2 + vx_del_clr),
        axis=0,
    )
    for idx_tr, r in enumerate(train_rows):
        pred_inh[r] = pre_pred_inh_u[idx_tr]
        pred_clr[r] = pre_pred_clr_u[idx_tr]
        vy_inh[r] = pre_var_inh_u[idx_tr]
        vy_clr[r] = pre_var_clr_u[idx_tr]
    pred_inh[probe_row] = np.sum(mu_inh_u[:, probe_row, :] * bundle.u_probe, axis=0)
    pred_clr[probe_row] = np.sum(mu_clr_u[:, probe_row, :] * bundle.u_probe, axis=0)
    vy_inh[probe_row] = 1.0 + np.sum(var_inh_u[:, probe_row, :] * (np.abs(bundle.u_probe) ** 2), axis=0)
    vy_clr[probe_row] = 1.0 + np.sum(var_clr_u[:, probe_row, :] * (np.abs(bundle.u_probe) ** 2), axis=0)

    vy_inh = np.maximum(vy_inh, 1e-12)
    vy_clr = np.maximum(vy_clr, 1e-12)

    # Strictly masked observation residuals (EXACT 0 on probe_row).
    res_inh = (y_u - pred_inh) * obs_mask[:, None]  # [T, N]
    res_clr = (y_u - pred_clr) * obs_mask[:, None]  # [T, N]

    row_res_inh = np.log1p(np.maximum(np.mean((np.abs(res_inh) ** 2) / vy_inh, axis=-1), 0.0)) * obs_mask
    row_res_clr = np.log1p(np.maximum(np.mean((np.abs(res_clr) ** 2) / vy_clr, axis=-1), 0.0)) * obs_mask

    # Blind row-local energy & kurtosis anomaly features (strictly C*-invariant, 0 on probe_row).
    row_e_obs = np.mean(np.abs(y_u) ** 2, axis=-1)  # [T]
    row_e_exp = 1.0 + np.sum(np.mean(np.abs(mu_clr_u) ** 2 + var_clr_u, axis=-1), axis=0)  # [T]
    row_energy_anom = np.log(np.maximum(row_e_obs, 1e-12) / np.maximum(row_e_exp, 1e-12)) * obs_mask
    row_m4 = np.mean(np.abs(y_u) ** 4, axis=-1)
    row_kurt_raw = row_m4 / np.maximum(row_e_obs * row_e_obs, 1e-12) - 2.0
    row_kurt_anom = np.log1p(np.maximum(row_kurt_raw, 0.0)) * obs_mask

    # Per-ray matched-filter residual phasors [K, T] (strictly U(1)-equivariant, 0 on probe_row).
    u_pow_inh = np.maximum(np.sum(np.abs(u_ref_inh) ** 2, axis=-1), 1.0)
    u_pow_clr = np.maximum(np.sum(np.abs(u_ref_clr) ** 2, axis=-1), 1.0)
    rho_inh = (np.sum(u_ref_inh.conj() * res_inh[None, :, :], axis=-1) / u_pow_inh) * obs_mask[None, :]
    rho_clr = (np.sum(u_ref_clr.conj() * res_clr[None, :, :], axis=-1) / u_pow_clr) * obs_mask[None, :]

    # Per-ray, per-row mean gains & variances: [K, T].
    mu_inh_bar = mu_inh_u.mean(axis=-1)
    mu_clr_bar = mu_clr_u.mean(axis=-1)
    var_inh_bar = np.maximum(var_inh_u.mean(axis=-1), 1e-12)
    var_clr_bar = np.maximum(var_clr_u.mean(axis=-1), 1e-12)

    # Interference-cancelled blind observed ray wave (0 on probe_row).
    g_obs_inh = (mu_inh_bar + rho_inh) * obs_mask[None, :]

    # Exact C*-invariant per-ray, per-row Jeffreys divergence & variance log-ratio.
    dj_ray_raw = jeffreys_complex_gaussian(mu_inh_u, var_inh_u, mu_clr_u, var_clr_u).mean(axis=-1)  # [K, T]
    dj_ray = np.log1p(np.maximum(dj_ray_raw, 0.0))
    log_var_ratio_ray = np.clip(np.log(var_inh_bar / var_clr_bar), -10.0, 10.0)

    # Normalized per-ray residual energy ratios (strictly C*-invariant, 0 on probe_row).
    rho_inh_mag = np.log1p((np.abs(rho_inh) ** 2) / np.maximum(var_inh_bar + 1.0 / float(N), 1e-12)) * obs_mask[None, :]
    rho_clr_mag = np.log1p((np.abs(rho_clr) ** 2) / np.maximum(var_clr_bar + 1.0 / float(N), 1e-12)) * obs_mask[None, :]

    # Strictly localized anchor Bayes factor (only nonzero at t in train_rows!).
    anchor_bf_local = np.arcsinh(base_logit_ray_use)[:, None] * is_anchor[None, :]
    log1p_snr_a = np.log1p(np.maximum(atom_snr_use, 0.0))
    nu_norm = np.clip(nu_use / 0.15, -1.5, 1.5)

    # Broadcast [T] and [K] features cleanly to [K, T].
    def _bt(arr: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr[None, :], (k_hat, T))

    def _bk(arr: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr[:, None], (k_hat, T))

    scalar_feats = np.stack(
        (
            _bt(t_norm),
            _bt(is_anchor),
            _bt(is_probe),
            _bt(dist_anchor),
            _bt(abs_dist_anchor),
            _bk(tau_norm),
            _bk(nu_norm),
            _bk(log1p_snr_a),
            anchor_bf_local,
            dj_ray,
            log_var_ratio_ray,
            rho_inh_mag,
            rho_clr_mag,
            _bt(row_res_inh),
            _bt(row_res_clr),
            _bt(row_energy_anom),
            _bt(row_kurt_anom),
            _bt(row_llr_conf_diff),
        ),
        axis=-1,
    ).astype(np.float64)

    complex_feats = np.stack(
        (
            mu_inh_bar,
            mu_clr_bar,
            mu_inh_bar - mu_clr_bar,
            rho_inh,
            rho_clr,
            g_obs_inh,
        ),
        axis=-1,
    ).astype(np.complex128)

    # Attention key evidence map (strictly C*-invariant, row-local, ray-perm equivariant).
    ev_base = (
        dj_ray
        + 1.5 * _bt(is_anchor)
        - 2.5 * _bt(is_probe)
        - 1.5 * np.maximum(_bt(row_energy_anom), 0.0)
        + 0.25 * _bk(log1p_snr_a)
    ).astype(np.float64)

    base_logit = np.broadcast_to(base_logit_ray_use[:, None], (k_hat, T)).astype(np.float64).copy()

    return ManifoldFeatures(
        tau_hat=tau_hat.copy(),
        nu_hat=nu_use.copy(),
        t_phys=t_phys,
        t_norm=t_norm,
        is_anchor=is_anchor,
        is_probe=is_probe,
        scalar_feats=scalar_feats,
        complex_feats=complex_feats,
        ev_base=ev_base,
        base_logit=base_logit,
    )


# ---------------------------------------------------------------------------
# Differentiable 2D (Ray x Time) Parallel-Transport Wave Attention Architecture
# ---------------------------------------------------------------------------


def _cmat(rows: int, cols: int) -> nn.Parameter:
    scale = 1.0 / math.sqrt(2.0 * rows)
    real = torch.randn(rows, cols, dtype=REAL) * scale
    imag = torch.randn(rows, cols, dtype=REAL) * scale
    return nn.Parameter(torch.complex(real, imag))


def _rmat(rows: int, cols: int) -> nn.Parameter:
    return nn.Parameter(torch.randn(rows, cols, dtype=REAL) / math.sqrt(rows))


def _safe_abs(z: torch.Tensor, eps: float = 1e-24) -> torch.Tensor:
    """Smooth zero-safe modulus for dimensionless normalized complex tensors."""
    return torch.sqrt(z.real.square() + z.imag.square() + eps)


def _safe_rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable zero-safe RMS magnitude (never differentiates sqrt(0))."""
    p = (x.real.square() + x.imag.square()).mean(dim=dim, keepdim=keepdim)
    safe = torch.where(p > 0.0, p, torch.ones_like(p))
    rms = safe.sqrt()
    unit = torch.where(p > 0.0, x / rms, torch.zeros_like(x))
    return unit, rms


class ManifoldTransportLayer(nn.Module):
    """Exact-parameter-matched 2D (Ray x Time) layer supporting 3 exact topologies:
      - 'em10_transport_attn'   : Global K_hat * T attention with explicit signed U(1) parallel-
                                  transport phase connection exp(-2j*pi*(s_nu*nu_k*(t-t')
                                  + omega_t*(t-t') - (tau_k - tau_k')*df_eff)) on Q, K, AND V
                                  + continuous relative (dt, dtau) geodesic bias.
      - 'em10_abs_no_transport' : Identical global K_hat * T attention & exact same parameter
                                  tensors, ablated to |q^H k| + real cosine bias + untransported v.
      - 'em10_local'            : Identical parameter tensors restricted strictly to local 1D
                                  temporal kernel (dt in {-1, 0, +1}) within each ray k only
                                  (EXACT ZERO cross-ray coupling and ZERO |dt| > 1 reach!).
    """

    def __init__(
        self,
        variant: str,
        d_model: int = 12,
        d_scalar: int = 12,
        n_heads: int = 4,
        d_ff: int = 12,
    ):
        super().__init__()
        if variant not in ("em10_transport_attn", "em10_abs_no_transport", "em10_local"):
            raise ValueError(f"Unsupported layer variant: {variant}")
        if d_model % n_heads != 0 or d_scalar % n_heads != 0:
            raise ValueError("d_model and d_scalar must both be divisible by n_heads")

        self.variant = variant
        self.d_model = d_model
        self.d_scalar = d_scalar
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_s_head = d_scalar // n_heads

        # Identical parameter tensors across all variants for exact integer real-parameter parity!
        self.w_q = _cmat(d_model, d_model)
        self.w_k = _cmat(d_model, d_model)
        self.w_v = _cmat(d_model, d_model)
        self.w_o = _cmat(d_model, d_model)

        self.w_sv = _rmat(d_scalar, d_scalar)
        self.w_so = _rmat(d_scalar, d_scalar)

        # Head-wise physical parallel-transport connection & multi-scale (dt, dtau) geodesic decay.
        if n_heads == 4:
            init_s_nu = torch.tensor([1.0, 1.0, 0.5, -1.0], dtype=REAL)
            init_omega = torch.tensor([-0.03, 0.03, -0.015, 0.015], dtype=REAL)
            init_df = torch.tensor([-0.12, 0.12, -0.04, 0.04], dtype=REAL)
            init_bt = torch.tensor([8.0, 2.0, -1.5, -3.0], dtype=REAL)
            init_btau = torch.tensor([5.0, 1.0, -1.0, -2.5], dtype=REAL)
        else:
            init_s_nu = torch.linspace(1.0, -1.0, n_heads, dtype=REAL)
            init_omega = torch.linspace(-0.03, 0.03, n_heads, dtype=REAL)
            init_df = torch.linspace(-0.12, 0.12, n_heads, dtype=REAL)
            init_bt = torch.linspace(8.0, -3.0, n_heads, dtype=REAL)
            init_btau = torch.linspace(5.0, -2.5, n_heads, dtype=REAL)

        self.s_nu = nn.Parameter(init_s_nu)
        self.omega_t = nn.Parameter(init_omega)
        self.df_eff = nn.Parameter(init_df)
        self.raw_beta_t = nn.Parameter(init_bt)
        self.raw_beta_tau = nn.Parameter(init_btau)
        self.gamma_ev = nn.Parameter(torch.full((n_heads,), 0.5, dtype=REAL))
        self.ev_proj = _rmat(d_scalar, n_heads)

        # C*-equivariant gated complex FFN + C*-invariant scalar FFN.
        self.ff1 = _cmat(d_model, d_ff)
        self.ff2 = _cmat(d_ff, d_model)
        self.s_ff1 = _rmat(d_scalar, d_scalar)
        self.s_fb1 = nn.Parameter(torch.zeros(d_scalar, dtype=REAL))
        self.s_ff2 = _rmat(d_scalar, d_scalar)

    def _local_1d_time_only(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        vc: torch.Tensor,
        vs: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_hat: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strictly local 1D temporal kernel (size 3: dt in {-1, 0, +1}) independently per ray k.

        ZERO cross-ray attention and ZERO non-local time reach (|dt| > 1)!
        Input shapes: q, k, vc: [B, H, K, T, Dh]; vs: [B, H, K, T, Dsh]; ev_k: [B, H, K, T].
        """
        b, h, k_len, t_len, _ = q.shape
        device = q.device
        t_idx = torch.arange(t_len, device=device)
        scale = 1.0 / math.sqrt(self.d_head)
        beta_t = F.softplus(self.raw_beta_t)      # [H]
        beta_tau = F.softplus(self.raw_beta_tau)  # [H]
        tau_n = tau_hat / float(OFFGRID_DELAY_MAX)  # [B, K]

        logits_list = []
        valid_list = []
        vc_list = []
        vs_list = []

        for dt in (-1, 0, 1):
            raw_ti = t_idx + dt
            valid = (raw_ti >= 0) & (raw_ti < t_len)  # [T]
            ti = raw_ti.clamp(0, t_len - 1)           # [T]
            kn = k[:, :, :, ti, :]                    # [B, H, K, T, Dh]
            vcn = vc[:, :, :, ti, :]                  # [B, H, K, T, Dh]
            vsn = vs[:, :, :, ti, :]                  # [B, H, K, T, Dsh]

            d_tp = t_phys[:, None, None, :] - t_phys[:, None, None, ti]  # [B, 1, 1, T]
            d_tn = t_norm[:, None, None, :] - t_norm[:, None, None, ti]  # [B, 1, 1, T]

            # Pointwise-in-ray phase & local geometric modulation (ZERO dependence on any k' != k!).
            phase = -2.0 * math.pi * (
                (self.s_nu[None, :, None, None] * nu_hat[:, None, :, None] + self.omega_t[None, :, None, None]) * d_tp
                - self.df_eff[None, :, None, None] * tau_n[:, None, :, None] * d_tn
            )  # [B, H, K, T]
            u_qk = torch.polar(torch.ones_like(phase), phase)  # [B, H, K, T]
            # (q * conj(kn)).sum(-1) * u_qk
            dot = (q * kn.conj()).sum(dim=-1)  # [B, H, K, T]
            geo = (
                -beta_t[None, :, None, None] * d_tn.square()
                - beta_tau[None, :, None, None] * (tau_n[:, None, :, None] * d_tn).square()
            )
            sc = (dot * u_qk).real * scale + geo + ev_k[:, :, :, ti]
            v_bhkt = valid[None, None, None, :].expand(b, h, k_len, t_len)
            sc = torch.where(v_bhkt, sc, torch.full_like(sc, -1e9))

            logits_list.append(sc)
            valid_list.append(v_bhkt)
            vc_list.append(vcn * u_qk.conj()[..., None])
            vs_list.append(vsn)

        logits = torch.stack(logits_list, dim=-1)      # [B, H, K, T, 3]
        valid_stack = torch.stack(valid_list, dim=-1)  # [B, H, K, T, 3]
        weights = logits.softmax(dim=-1)
        weights = torch.where(valid_stack, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-300)

        vc_stack = torch.stack(vc_list, dim=-2)        # [B, H, K, T, 3, Dh]
        vs_stack = torch.stack(vs_list, dim=-2)        # [B, H, K, T, 3, Dsh]
        agg_c = (weights[..., None].to(COMPLEX) * vc_stack).sum(dim=-2)  # [B, H, K, T, Dh]
        agg_s = (weights[..., None] * vs_stack).sum(dim=-2)              # [B, H, K, T, Dsh]
        return agg_c, agg_s

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_hat: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h:       [B, K, T, D] complex128 (U(1)-equivariant, raw amplitudes preserved)
          s:       [B, K, T, d_s] float64 (C*-invariant)
          tau_hat: [B, K] float64 continuous delays in [0.0, 6.5]
          nu_hat:  [B, K] float64 continuous Dopplers
          t_phys:  [B, T] float64 physical row times in useful-symbol units
          t_norm:  [B, T] float64 normalized row times in [-1, 1]
          ev_base: [B, K, T] float64 C*-invariant key evidence bias
        """
        b, k_len, t_len, _ = h.shape
        l_tok = k_len * t_len

        h_u, _ = _safe_rms(h, dim=-1, keepdim=True)
        q_kt = (h_u @ self.w_q).unflatten(-1, (self.n_heads, self.d_head)).permute(0, 3, 1, 2, 4)   # [B, H, K, T, Dh]
        k_kt = (h_u @ self.w_k).unflatten(-1, (self.n_heads, self.d_head)).permute(0, 3, 1, 2, 4)
        vc_kt = (h @ self.w_v).unflatten(-1, (self.n_heads, self.d_head)).permute(0, 3, 1, 2, 4)    # 1-homogeneous!
        vs_kt = (s @ self.w_sv).unflatten(-1, (self.n_heads, self.d_s_head)).permute(0, 3, 1, 2, 4)

        ev_k = (
            self.gamma_ev[None, :, None, None] * ev_base[:, None, :, :]
            + (s @ self.ev_proj).permute(0, 3, 1, 2)
        )  # [B, H, K, T]

        if self.variant == "em10_local":
            agg_c_kt, agg_s_kt = self._local_1d_time_only(
                q_kt, k_kt, vc_kt, vs_kt, tau_hat, nu_hat, t_phys, t_norm, ev_k
            )
        else:
            # Flatten unordered (K, T) tokens to L = K * T for global 2D Ray x Time attention.
            ev_flat = ev_k.reshape(b, self.n_heads, l_tok)

            # Token coordinates: each token (k, t) has (tau_k, nu_k, t_phys_t, t_norm_t).
            tau_tok = tau_hat[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok)
            tn_tok = t_norm[:, None, :].expand(b, k_len, t_len).reshape(b, l_tok)

            dt_norm = tn_tok[:, :, None] - tn_tok[:, None, :]                            # [B, L, L]
            dtau_norm = (tau_tok[:, :, None] - tau_tok[:, None, :]) / float(OFFGRID_DELAY_MAX)

            beta_t = F.softplus(self.raw_beta_t)                                         # [H]
            beta_tau = F.softplus(self.raw_beta_tau)                                     # [H]
            geo_bias = (
                -beta_t[None, :, None, None] * dt_norm.square()[:, None, :, :]
                - beta_tau[None, :, None, None] * dtau_norm.square()[:, None, :, :]
            )  # [B, H, L, L]

            # Exact algebraic factorization of the non-separable pairwise U(1) parallel-transport connection:
            #   U_qk[(k,t), (k',t')] = exp(-2j*pi*((s_nu_h*nu_k + omega_t_h)*(t - t') - df_eff_h*(tau_k - tau_k')))
            #                        = e_q[k, t] * e_k[k, t'] * e_tau[k']
            # Requires ONLY O(K*T) trig evaluations instead of O((K*T)^2) while being bit-exact (< 1e-15)!
            rate = self.s_nu[None, :, None] * nu_hat[:, None, :] + self.omega_t[None, :, None]  # [B, H, K]
            ang_q = -2.0 * math.pi * (
                rate[..., None] * t_phys[:, None, None, :]
                - self.df_eff[None, :, None, None] * tau_hat[:, None, :, None]
            )  # [B, H, K, T]
            ang_k = 2.0 * math.pi * (rate[..., None] * t_phys[:, None, None, :])        # [B, H, K, T]
            ang_tau = -2.0 * math.pi * (self.df_eff[None, :, None] * tau_hat[:, None, :])  # [B, H, K]

            e_q = torch.polar(torch.ones_like(ang_q), ang_q)      # [B, H, K, T]
            e_k = torch.polar(torch.ones_like(ang_k), ang_k)      # [B, H, K, T]
            e_tau = torch.polar(torch.ones_like(ang_tau), ang_tau)  # [B, H, K]

            scale = 1.0 / math.sqrt(self.d_head)

            if self.variant == "em10_transport_attn":
                qm = (q_kt * e_q[..., None]).reshape(b, self.n_heads, l_tok, self.d_head)
                km = (k_kt * e_tau.conj()[..., None, None]).reshape(b, self.n_heads, l_tok, self.d_head)
                s_mod = (qm @ km.mH).reshape(b, self.n_heads, k_len, t_len, k_len, t_len)
                qk_real = (s_mod * e_k[:, :, :, None, None, :]).real.reshape(b, self.n_heads, l_tok, l_tok)
                logits = qk_real * scale + geo_bias + ev_flat[:, :, None, :]
                weights = logits.softmax(dim=-1)
                # Value parallel transport: conj(U_qk) = conj(e_q[k,t]) * conj(e_k[k,t']) * conj(e_tau[k'])
                w_rot = (
                    weights.reshape(b, self.n_heads, k_len, t_len, k_len, t_len).to(COMPLEX)
                    * e_k.conj()[:, :, :, None, None, :]
                ).reshape(b, self.n_heads, l_tok, l_tok)
                vm = (vc_kt * e_tau.conj()[..., None, None]).reshape(b, self.n_heads, l_tok, self.d_head)
                agg_c_kt = (w_rot @ vm).reshape(b, self.n_heads, k_len, t_len, self.d_head) * e_q.conj()[..., None]
                vs = vs_kt.reshape(b, self.n_heads, l_tok, self.d_s_head)
                agg_s_kt = (weights @ vs).reshape(b, self.n_heads, k_len, t_len, self.d_s_head)
            else:
                # 'em10_abs_no_transport': unsigned |q^H k| + real cosine bias + untransported v
                q = q_kt.reshape(b, self.n_heads, l_tok, self.d_head)
                k = k_kt.reshape(b, self.n_heads, l_tok, self.d_head)
                vc = vc_kt.reshape(b, self.n_heads, l_tok, self.d_head)
                vs = vs_kt.reshape(b, self.n_heads, l_tok, self.d_s_head)
                s_complex = q @ k.mH
                # Exact cos(phase) = Re(e_q[k,t] * e_k[k,t'] * e_tau[k'])
                u_real = (
                    (e_q[:, :, :, :, None, None] * e_k[:, :, :, None, None, :])
                    * e_tau[:, :, None, None, :, None]
                ).real.reshape(b, self.n_heads, l_tok, l_tok)
                pos_bias = 0.15 * u_real
                logits = _safe_abs(s_complex) * scale + pos_bias + geo_bias + ev_flat[:, :, None, :]
                weights = logits.softmax(dim=-1)
                vc_real = torch.view_as_real(vc.resolve_conj()).flatten(-2)
                agg_c = torch.view_as_complex((weights @ vc_real).unflatten(-1, (self.d_head, 2)))
                agg_c_kt = agg_c.reshape(b, self.n_heads, k_len, t_len, self.d_head)
                agg_s_kt = (weights @ vs).reshape(b, self.n_heads, k_len, t_len, self.d_s_head)

        delta_h = agg_c_kt.permute(0, 2, 3, 1, 4).flatten(-2) @ self.w_o  # [B, K, T, D]
        delta_s = agg_s_kt.permute(0, 2, 3, 1, 4).flatten(-2) @ self.w_so  # [B, K, T, d_s]

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


class _ManifoldTrustBackbone(nn.Module):
    """Core U(1)-equivariant + C*-invariant + ray-permutation-equivariant 2D manifold backbone."""

    def __init__(
        self,
        layer_variant: str,
        d_model: int = 12,
        d_scalar: int = 12,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 12,
    ):
        super().__init__()
        self.w_c = _cmat(C_COMPLEX, d_model)
        self.w_s = _rmat(C_SCALAR, d_scalar)
        self.b_s = nn.Parameter(torch.zeros(d_scalar, dtype=REAL))
        self.w_s2c = _rmat(d_scalar, d_model)

        self.layers = nn.ModuleList(
            [
                ManifoldTransportLayer(
                    layer_variant,
                    d_model=d_model,
                    d_scalar=d_scalar,
                    n_heads=n_heads,
                    d_ff=d_ff,
                )
                for _ in range(n_layers)
            ]
        )

        # Strictly C*-invariant & ray-permutation-equivariant readout.
        self.w_read_ref = _cmat(d_model, 4)
        self.w_read_bilin = _cmat(d_model, 4)
        self.w_head = _rmat(d_scalar + 6, 3)
        self.b_head = nn.Parameter(torch.zeros(3, dtype=REAL))

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_hat: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = F.silu(scalar_feats @ self.w_s + self.b_s)       # [B, K, T, d_s] C*-invariant
        mod = 1.0 + torch.tanh(s @ self.w_s2c)               # [B, K, T, D] C*-invariant real multiplier
        h = (complex_feats @ self.w_c) * mod.to(COMPLEX)     # [B, K, T, D] strictly U(1)-equivariant!

        for layer in self.layers:
            h, s = layer(h, s, tau_hat, nu_hat, t_phys, t_norm, ev_base)

        # Reference fields: [mu_inh, mu_clr, rho_inh, rho_clr] (all U(1)-equivariant & active in no_history).
        ref = torch.stack(
            (
                complex_feats[..., 0],
                complex_feats[..., 1],
                complex_feats[..., 3],
                complex_feats[..., 4],
            ),
            dim=-1,
        )  # [B, K, T, 4]
        u_ref = h @ self.w_read_ref
        inv_ref = torch.asinh((ref.conj() * u_ref).real)     # [B, K, T, 4] strictly C*-invariant

        u_b = h @ self.w_read_bilin                          # [B, K, T, 4]
        inv_bilin = torch.asinh((u_b[..., :2].conj() * u_b[..., 2:]).real)  # [B, K, T, 2]

        head_in = torch.cat((s, inv_ref, inv_bilin), dim=-1) # [B, K, T, d_s + 6]
        out = head_in @ self.w_head + self.b_head            # [B, K, T, 3]
        return out[..., 0], out[..., 1], out[..., 2]


class ManifoldTrustGate(nn.Module):
    """Unified P11 learnable C*-invariant, ray-permutation-equivariant 2D manifold trust gate.

    Centered around a frozen step-0 copy (`initial_map`) so untrained initialization matches
    `classical_per_ray_gate` bit-for-bit (< 1e-13) while maintaining 100% active finite nonzero
    gradients on every trainable parameter from step 0.
    """

    def __init__(
        self,
        variant: str = "em10_transport_attn",
        d_model: int = 12,
        d_scalar: int = 12,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 12,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ManifoldTrustGate variant '{variant}'. Expected one of {VARIANTS}.")
        self.variant = variant
        self.d_model = d_model
        self.d_scalar = d_scalar
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        layer_variant = "em10_transport_attn" if variant == "em10_no_history" else variant
        self.backbone = _ManifoldTrustBackbone(
            layer_variant=layer_variant,
            d_model=d_model,
            d_scalar=d_scalar,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        # Frozen step-0 map for exact zero-residual centering (requires_grad=False -> 0 trainable params).
        self.initial_map = copy.deepcopy(self.backbone)
        for p in self.initial_map.parameters():
            p.requires_grad_(False)

    def real_parameters(self) -> int:
        """Exact integer count of trainable real-valued scalar degrees of freedom."""
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def compute_initial_offsets(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_hat: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute frozen step-0 centering offsets (exact match to inline initial_map pass)."""
        with torch.no_grad():
            return self.initial_map(
                scalar_feats.detach(),
                complex_feats.detach(),
                tau_hat.detach(),
                nu_hat.detach(),
                t_phys.detach(),
                t_norm.detach(),
                ev_base.detach(),
            )

    def forward_tensors(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_hat: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
        base_logit: torch.Tensor,
        is_probe: torch.Tensor,
        init_offsets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable batched forward pass returning (w_ray [B, K, T], logit_ray [B, K, T], noise_mult [B, T])."""
        raw_delta, raw_log_tau, raw_noise = self.backbone(
            scalar_feats, complex_feats, tau_hat, nu_hat, t_phys, t_norm, ev_base
        )
        if init_offsets is None:
            delta_0, log_tau_0, noise_0 = self.compute_initial_offsets(
                scalar_feats, complex_feats, tau_hat, nu_hat, t_phys, t_norm, ev_base
            )
        else:
            delta_0, log_tau_0, noise_0 = init_offsets

        delta_logit = raw_delta - delta_0
        log_tau = raw_log_tau - log_tau_0
        # Permutation-invariant mean over rays for row-level outlier / noise-inflation gate.
        delta_log_noise = (raw_noise - noise_0).mean(dim=1) * (1.0 - is_probe)  # [B, T]
        noise_mult = torch.exp(delta_log_noise.clamp(-1.0, 3.0))                # exact 1.0 at step 0!

        tau = 2.0 * torch.sigmoid(log_tau)                                      # exact 1.0 at step 0!
        logit_ray = tau * base_logit + delta_logit                              # exact base_logit at step 0!
        w_ray = torch.sigmoid(logit_ray)
        return w_ray, logit_ray, noise_mult

    def extract_features_for_variant(
        self,
        obs: OffGridObservation,
        bundle: OffGridBranchBundle,
    ) -> ManifoldFeatures:
        """Extract variant-appropriate features ('em10_no_history' strictly masks Frame-A inputs)."""
        return extract_manifold_features(
            obs,
            bundle,
            mask_history=(self.variant == "em10_no_history"),
        )

    def gate(
        self,
        obs: OffGridObservation,
        bundle: OffGridBranchBundle,
    ) -> OffGridGateResult:
        """Inference entry point (uses torch.no_grad(), NEVER torch.inference_mode())."""
        feats = self.extract_features_for_variant(obs, bundle)
        with torch.no_grad():
            s_t = torch.from_numpy(feats.scalar_feats).to(REAL).unsqueeze(0)
            c_t = torch.from_numpy(feats.complex_feats).to(COMPLEX).unsqueeze(0)
            tau_t = torch.from_numpy(feats.tau_hat).to(REAL).unsqueeze(0)
            nu_t = torch.from_numpy(feats.nu_hat).to(REAL).unsqueeze(0)
            tp_t = torch.from_numpy(feats.t_phys).to(REAL).unsqueeze(0)
            tn_t = torch.from_numpy(feats.t_norm).to(REAL).unsqueeze(0)
            ev_t = torch.from_numpy(feats.ev_base).to(REAL).unsqueeze(0)
            bl_t = torch.from_numpy(feats.base_logit).to(REAL).unsqueeze(0)
            pr_t = torch.from_numpy(feats.is_probe).to(REAL).unsqueeze(0)
            w_t, _, nm_t = self.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
            w_np = w_t.squeeze(0).cpu().numpy()
            nm_np = nm_t.squeeze(0).cpu().numpy()
        return compose_offgrid_gate_output(w_np, nm_np, obs, bundle)


# ---------------------------------------------------------------------------
# Fast Differentiable Off-Grid Bit BCE + Time-Varying Waveform & Probe NLL Loss
# ---------------------------------------------------------------------------


def differentiable_offgrid_loss(
    w_ray: torch.Tensor,
    noise_mult: torch.Tensor,
    *,
    mean_inh_bar: torch.Tensor,     # [B, K, 18] complex128 row-mean inherit ray gains on data rows
    var_inh_bar: torch.Tensor,      # [B, K, 18] float64 row-mean inherit ray vars on data rows
    mean_clr_bar: torch.Tensor,     # [B, K, 18] complex128 row-mean clear ray gains on data rows
    var_clr_bar: torch.Tensor,      # [B, K, 18] float64 row-mean clear ray vars on data rows
    phi_f: torch.Tensor,            # [B, K, N] complex128 unitary fractional-delay freq responses
    y_data_freq: torch.Tensor,      # [B, 18, N] complex128 unitary FFT of received data rows
    b_code: torch.Tensor,           # [B, 18, N, 4] float64 coded bits on data rows
    data_row_idx: torch.Tensor,     # [B, 18] long indices of the 18 data rows in [0..T-1]
    mean_inh_probe: torch.Tensor,   # [B, K, N] complex128 inherit ray gains on probe row
    var_inh_probe: torch.Tensor,    # [B, K, N] float64 inherit ray vars on probe row
    mean_clr_probe: torch.Tensor,   # [B, K, N] complex128 clear ray gains on probe row
    var_clr_probe: torch.Tensor,    # [B, K, N] float64 clear ray vars on probe row
    u_probe: torch.Tensor,          # [B, K, N] complex128 delayed known probe waveforms
    probe_target: torch.Tensor,     # [B, N] complex128 withheld noisy probe target
    n0: torch.Tensor,               # [B] float64 explicit noise variance
    probe_loss_weight: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strictly C*-invariant differentiable training objective: exact freq-domain LMMSE Bit BCE + probe NLL."""
    b, k_len, _ = w_ray.shape
    r_len = data_row_idx.shape[1]

    # Gather per-link data-row weights w_data [B, K, 18] and noise_mult_data [B, 18].
    idx_k = data_row_idx[:, None, :].expand(b, k_len, r_len)
    w_data = torch.gather(w_ray, dim=2, index=idx_k)                       # [B, K, 18]
    nm_data = torch.gather(noise_mult, dim=1, index=data_row_idx)          # [B, 18]

    # Per-ray mixture moments on data rows (exact within + between variance!).
    mix_mu = w_data * mean_inh_bar + (1.0 - w_data) * mean_clr_bar         # [B, K, 18]
    mix_var = (
        w_data * var_inh_bar
        + (1.0 - w_data) * var_clr_bar
        + w_data * (1.0 - w_data) * (mean_inh_bar - mean_clr_bar).abs().square()
    )  # [B, K, 18]

    n0_b = n0[:, None].clamp_min(1e-300)                                   # [B, 1]
    noise_data = (n0_b * nm_data + mix_var.sum(dim=1)).clamp_min(1e-300)   # [B, 18]

    # Exact unitary frequency-domain LMMSE equalizer + deterministic IFFT via static F_H!
    h_freq = torch.einsum("bkr,bkn->brn", mix_mu, phi_f)                   # [B, 18, N]
    h2 = h_freq.real.square() + h_freq.imag.square()
    denom = (h2 + noise_data.unsqueeze(-1)).clamp_min(1e-300)
    w_f = h_freq.conj() / denom
    x_hat = (w_f * y_data_freq) @ _F_H_TORCH.to(device=w_ray.device)       # [B, 18, N]
    post_var = (noise_data.unsqueeze(-1) / denom).mean(dim=-1).clamp(1e-12, 1.0)  # [B, 18]
    gain = 1.0 - post_var

    # Exact differentiable Gray 16-QAM soft bit LLRs (positive = bit 1).
    pam = _PAM_TORCH.to(device=w_ray.device)
    axes = torch.stack((x_hat.real, x_hat.imag), dim=-1)                   # [B, 18, N, 2]
    logp = (
        2.0 * axes.unsqueeze(-1) * pam
        - gain[:, :, None, None, None] * (pam * pam)
    ) / post_var[:, :, None, None, None]
    sign_l = torch.logaddexp(logp[..., 2], logp[..., 3]) - torch.logaddexp(logp[..., 0], logp[..., 1])
    inner_l = torch.logaddexp(logp[..., 1], logp[..., 3]) - torch.logaddexp(logp[..., 0], logp[..., 2])
    llrs = torch.stack((sign_l, inner_l), dim=-1).reshape(b, r_len, N, 4)

    signed = 2.0 * b_code - 1.0
    bit_bce = F.softplus(-signed * llrs).mean()

    # Exact withheld row-19 probe NLL with exact within + between total variance (strictly C*-invariant!).
    wp = w_ray[:, :, PROBE_ROW].unsqueeze(-1)                              # [B, K, 1]
    mix_mu_p = wp * mean_inh_probe + (1.0 - wp) * mean_clr_probe           # [B, K, N]
    mix_var_p = (
        wp * var_inh_probe
        + (1.0 - wp) * var_clr_probe
        + wp * (1.0 - wp) * (mean_inh_probe - mean_clr_probe).abs().square()
    )
    probe_pred = (mix_mu_p * u_probe).sum(dim=1)                           # [B, N]
    probe_var = (n0_b + (mix_var_p * u_probe.abs().square()).sum(dim=1)).clamp_min(1e-300)
    err2 = (probe_target - probe_pred).abs().square()
    probe_nll = (torch.log(probe_var / n0_b) + err2 / probe_var).mean()

    total = bit_bce + float(probe_loss_weight) * probe_nll
    return total, bit_bce, probe_nll


__all__ = [
    "COMPLEX",
    "C_COMPLEX",
    "C_SCALAR",
    "REAL",
    "VARIANTS",
    "ManifoldFeatures",
    "ManifoldTransportLayer",
    "ManifoldTrustGate",
    "differentiable_offgrid_loss",
    "extract_manifold_features",
]
