"""P12 Stage 5 Neural Refiners: Role A (Symmetric Parallel-Transport) vs Role B (Equal-Budget Non-Physics).

Strict architectural, physical, and fairness invariants:
  1. ZERO target-row or probe-row leakage: features are extracted strictly from a target-excluded
     `FoldPosteriorSnapshot` where owned target rows and `probe_row` were masked to `NaN` before
     any inference.
  2. Role A (`SymmetricTransportRefiner`):
       - Exact symmetric unitary parallel-transport connection:
           Phi_ij = pi * (nu_tilde_i + nu_tilde_j) * (t_i - t_j) - 2*pi * df_eff * (tau_i - tau_j)
           P_ij = exp(i * Phi_ij)  ==>  P_ji == conj(P_ij) and |P_ij * P_ji - 1| < 1e-15!
       - Exact U(1) / C* gauge equivariance on complex residual output `delta_g_u` and exact
         C* invariance on `delta_fam_logits [K, T, 4]` and `delta_log_var [K, T]`.
       - Exact ray-permutation equivariance across unordered discovered rays.
  3. Role B (`NonPhysicsTransformerRefiner`):
       - Equal-input, equal-budget, exact-parameter-matched standard real Transformer
         (Multi-Head Self-Attention + LayerNorm + SiLU FFN + real relative positional MLP bias).
       - Removes complex gauge equivariance and symmetric U(1) parallel-transport geometry.
  4. Both Role A and Role B use frozen `initial_map` step-0 zero-centering so at step 0 they
     reproduce the underlying upgraded physics engine to `< 1e-13` while keeping 100% nonzero
     gradients on every trainable parameter.
  5. Pure `float64` / `complex128` deterministic arithmetic; NEVER uses `torch.inference_mode()`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F

from p11.contract import N, OFFGRID_DELAY_MAX, T, fractional_delay_freq_response, unitary_dft_matrix
from p11.offgrid_channel import ROW_STEP
from research.p12_stage5_fusion_neural.consistent_fusion_engine import FoldPosteriorSnapshot

REAL = torch.float64
COMPLEX = torch.complex128

S5_C_SCALAR = 22
S5_C_COMPLEX = 6
N_FAMILIES = 4  # 0=inherit, 1=clear+, 2=clear-, 3=piecewise

_PAM_TORCH = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=REAL) / math.sqrt(10.0)
_F_H_TORCH = torch.from_numpy(np.ascontiguousarray(unitary_dft_matrix(N).conj().T)).to(COMPLEX)


# ---------------------------------------------------------------------------
# 1. Strictly Fold-Isolated Feature & Physics Branch Tensor Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage5FoldFeatures:
    """Strictly fold-isolated inputs & intra-row phase rotations for Role A / Role B."""

    scalar_feats: np.ndarray    # [K, T, S5_C_SCALAR] float64 strictly C*-invariant
    complex_feats: np.ndarray   # [K, T, S5_C_COMPLEX] complex128 strictly U(1)-equivariant (1-homog)
    tau_hat: np.ndarray         # [K] float64
    nu_post: np.ndarray         # [K] float64 posterior mean Doppler per ray
    t_phys: np.ndarray          # [T] float64
    t_norm: np.ndarray          # [T] float64
    ev_base: np.ndarray         # [K, T] float64 strictly C*-invariant key evidence bias
    rot_intra: np.ndarray       # [K, T, N] complex128 unit-modulus intra-row phase profile
    fam_active: np.ndarray      # [K, 4] bool which families exist in this fold


def compute_family_summaries(
    snap: FoldPosteriorSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-ray, per-family `(fam_logw [K,4], fam_mean_u [K,4,T,N], fam_var_u [K,4,T,N], fam_active [K,4])`."""
    k_cnt = snap.lw_cur.shape[0]
    w_all = np.exp(snap.lw_cur - logsumexp(snap.lw_cur, axis=1, keepdims=True))

    fam_logw = np.full((k_cnt, N_FAMILIES), -1e9, dtype=np.float64)
    fam_mean_u = np.zeros((k_cnt, N_FAMILIES, T, N), dtype=np.complex128)
    fam_var_u = np.full((k_cnt, N_FAMILIES, T, N), 1e-6, dtype=np.float64)
    fam_active = np.zeros((k_cnt, N_FAMILIES), dtype=bool)

    for j in range(k_cnt):
        f_j = snap.mode_family[j]
        s_j = snap.mode_sign_pos[j]
        masks = (
            f_j == 0,
            (f_j == 1) & s_j,
            (f_j == 1) & (~s_j),
            f_j == 2,
        )
        for b_i, mask in enumerate(masks):
            if not np.any(mask):
                continue
            fam_active[j, b_i] = True
            lw_sub = snap.lw_cur[j, mask]
            log_mass = float(logsumexp(lw_sub))
            fam_logw[j, b_i] = log_mass
            w_sub = np.exp(lw_sub - log_mass)
            mh_sub = snap.mh_cur[j, mask]
            pv_sub = snap.pv_cur[j, mask]
            rot_sub = snap.rot_bank[j, mask]  # [Hm, T, N]
            m_b = np.einsum("h,htn->tn", w_sub * mh_sub, rot_sub)
            sec_b = float(np.sum(w_sub * (np.abs(mh_sub) ** 2 + pv_sub)))
            v_b = np.maximum(sec_b - (m_b.real ** 2 + m_b.imag ** 2), 1e-12)
            fam_mean_u[j, b_i] = m_b
            fam_var_u[j, b_i] = v_b

    return fam_logw, fam_mean_u, fam_var_u, fam_active


def extract_stage5_fold_features(snap: FoldPosteriorSnapshot) -> Stage5FoldFeatures:
    """Extract strictly target/probe-excluded, C*-canonical 2D (Ray x Time) fold features."""
    k_cnt = snap.lw_cur.shape[0]
    w_all = np.exp(snap.lw_cur - logsumexp(snap.lw_cur, axis=1, keepdims=True))
    fam_logw, fam_mean_u, fam_var_u, fam_active = compute_family_summaries(snap)

    # Posterior mean Doppler & entropy per ray (strictly C*-invariant!)
    nu_post = np.sum(w_all * snap.mode_nu_eff, axis=1).astype(np.float64)
    ent_k = -np.sum(w_all * np.log(np.maximum(w_all, 1e-300)), axis=1) / max(math.log(float(w_all.shape[1])), 1.0)
    fam_prob = np.where(fam_active, np.exp(np.clip(fam_logw, -50.0, 0.0)), 0.0)
    fam_prob /= np.maximum(np.sum(fam_prob, axis=1, keepdims=True), 1e-12)

    t_phys = (np.arange(T, dtype=np.float64) * ROW_STEP).astype(np.float64)
    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    dist_anc = (np.arange(T, dtype=np.float64) - float(snap.r_anchor)) / float(T - 1)
    abs_dist_anc = np.abs(dist_anc)

    is_anchor = np.zeros(T, dtype=np.float64)
    is_anchor[snap.r_anchor] = 1.0
    is_probe = np.zeros(T, dtype=np.float64)
    is_probe[snap.probe_row] = 1.0
    is_target = np.zeros(T, dtype=np.float64)
    is_target[list(snap.t_rows)] = 1.0
    is_aux = np.zeros(T, dtype=np.float64)
    is_aux[list(snap.a_rows)] = 1.0
    is_acc_helper = np.zeros(T, dtype=np.float64)
    if snap.accepted_rows:
        is_acc_helper[list(snap.accepted_rows)] = 1.0
    obs_mask = 1.0 - is_target - is_probe  # Strictly 0 on targets AND probe!

    # Intra-row unit phase profile from posterior mean Doppler (strictly public & C*-invariant!)
    dt_intra = (np.arange(N, dtype=np.float64) - 0.5 * float(N - 1)) / float(N)  # [-0.5, +0.5] symbol
    rot_intra = np.exp(2j * np.pi * nu_post[:, None, None] * dt_intra[None, None, :])
    rot_intra = np.broadcast_to(rot_intra, (k_cnt, T, N)).copy()

    # Row-averaged complex ray moments per family (strictly U(1)-equivariant & 1-homogeneous!)
    # De-rotate intra-row phase before averaging so row-mean phasor has full coherent amplitude!
    mu_mix_bar = np.mean(snap.m_u * rot_intra.conj(), axis=-1)              # [K, T]
    mu_inh_bar = np.mean(fam_mean_u[:, 0] * rot_intra.conj(), axis=-1)      # [K, T]
    mu_clrp_bar = np.mean(fam_mean_u[:, 1] * rot_intra.conj(), axis=-1)     # [K, T]
    mu_clrn_bar = np.mean(fam_mean_u[:, 2] * rot_intra.conj(), axis=-1)     # [K, T]
    mu_pw_bar = np.mean(fam_mean_u[:, 3] * rot_intra.conj(), axis=-1)       # [K, T]
    rho_obs = (snap.obs_matched_resid * obs_mask[None, :]).astype(np.complex128)  # 0 on targets & probe!

    var_mix_bar = np.maximum(np.mean(snap.v_u, axis=-1), 1e-12)
    pow_mix_bar = np.mean(snap.m_u.real ** 2 + snap.m_u.imag ** 2, axis=-1)
    log_var_mix = np.log1p(var_mix_bar)
    log_pow_mix = np.log1p(pow_mix_bar)

    tau_norm = np.asarray(snap.used_delays, dtype=np.float64) / float(OFFGRID_DELAY_MAX)
    nu_norm = np.clip(nu_post / 0.15, -1.5, 1.5)
    log_p_atom = np.log1p(np.maximum(snap.p_atom_u, 0.0))
    reset_flag = np.full((k_cnt, T), 1.0 if snap.delay_reset_triggered else 0.0, dtype=np.float64)

    # Strictly masked observation stats (EXACT 0 on targets and probe!)
    h_alpha = snap.helper_alpha_row * obs_mask
    h_margin = np.log1p(np.maximum(snap.helper_margin_row, 0.0)) * obs_mask
    r_resid = snap.row_resid_row * obs_mask
    r_eanom = snap.row_energy_anom * obs_mask

    def _bt(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a[None, :], (k_cnt, T))

    def _bk(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a[:, None], (k_cnt, T))

    scalar_feats = np.stack(
        (
            _bt(t_norm),
            _bt(dist_anc),
            _bt(abs_dist_anc),
            _bt(is_anchor),
            _bt(is_aux),
            _bt(is_acc_helper),
            _bt(is_target),
            _bt(is_probe),
            _bk(tau_norm),
            _bk(nu_norm),
            _bk(log_p_atom),
            _bk(ent_k),
            _bk(fam_prob[:, 0]),
            _bk(fam_prob[:, 1]),
            _bk(fam_prob[:, 2]),
            _bk(fam_prob[:, 3]),
            log_var_mix,
            log_pow_mix,
            _bt(h_alpha),
            _bt(h_margin),
            _bt(r_resid),
            _bt(r_eanom) + 0.0 * reset_flag,
        ),
        axis=-1,
    ).astype(np.float64)
    # Replace last channel cleanly with explicit reset_flag? Keep r_eanom AND encode reset via fam_prob (inh==0,pw==0)!
    # Notice when delay_reset_triggered=True, fam_prob[:,0]==0 and fam_prob[:,3]==0 identically!

    complex_feats = np.stack(
        (
            mu_mix_bar,
            mu_inh_bar,
            mu_clrp_bar,
            mu_clrn_bar,
            mu_pw_bar,
            rho_obs,
        ),
        axis=-1,
    ).astype(np.complex128)

    # Strictly C*-invariant key evidence bias: emphasizes anchor & accepted helpers, masks targets/probe!
    ev_base = (
        1.8 * _bt(is_anchor)
        + 1.2 * _bt(h_alpha)
        + 0.3 * _bt(is_aux)
        - 2.0 * _bt(is_target)
        - 2.5 * _bt(is_probe)
        - 1.2 * np.maximum(_bt(r_eanom), 0.0)
        + 0.25 * _bk(log_p_atom)
    ).astype(np.float64)

    return Stage5FoldFeatures(
        scalar_feats=scalar_feats,
        complex_feats=complex_feats,
        tau_hat=np.asarray(snap.used_delays, dtype=np.float64).copy(),
        nu_post=nu_post,
        t_phys=t_phys,
        t_norm=t_norm,
        ev_base=ev_base,
        rot_intra=rot_intra,
        fam_active=fam_active,
    )


# ---------------------------------------------------------------------------
# 2. Role A: Exact Symmetric Unitary Parallel-Transport Attention Refiner
# ---------------------------------------------------------------------------


def _cmat(rows: int, cols: int) -> nn.Parameter:
    scale = 1.0 / math.sqrt(2.0 * rows)
    return nn.Parameter(
        torch.complex(
            torch.randn(rows, cols, dtype=REAL) * scale,
            torch.randn(rows, cols, dtype=REAL) * scale,
        )
    )


def _rmat(rows: int, cols: int) -> nn.Parameter:
    return nn.Parameter(torch.randn(rows, cols, dtype=REAL) / math.sqrt(rows))


def _safe_abs(z: torch.Tensor, eps: float = 1e-24) -> torch.Tensor:
    return torch.sqrt(z.real.square() + z.imag.square() + eps)


def _safe_rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    p = (x.real.square() + x.imag.square()).mean(dim=dim, keepdim=True)
    safe = torch.where(p > 0.0, p, torch.ones_like(p))
    rms = safe.sqrt()
    unit = torch.where(p > 0.0, x / rms, torch.zeros_like(x))
    return unit, (rms if keepdim else rms.squeeze(dim))


class SymmetricTransportLayer(nn.Module):
    """2D (Ray x Time) Attention with EXACT Symmetric Unitary Parallel-Transport Connection P_ji = P_ij*."""

    def __init__(
        self,
        d_model: int = 16,
        d_scalar: int = 16,
        n_heads: int = 4,
        d_ff: int = 24,
    ):
        super().__init__()
        if d_model % n_heads != 0 or d_scalar % n_heads != 0:
            raise ValueError("d_model and d_scalar must be divisible by n_heads")
        self.d_model = d_model
        self.d_scalar = d_scalar
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_s_head = d_scalar // n_heads

        self.w_q = _cmat(d_model, d_model)
        self.w_k = _cmat(d_model, d_model)
        self.w_v = _cmat(d_model, d_model)
        self.w_o = _cmat(d_model, d_model)

        self.w_sv = _rmat(d_scalar, d_scalar)
        self.w_so = _rmat(d_scalar, d_scalar)

        # Head-wise symmetric connection parameters
        self.s_nu = nn.Parameter(torch.linspace(1.0, -0.5, n_heads, dtype=REAL))
        self.omega_t = nn.Parameter(torch.linspace(-0.025, 0.025, n_heads, dtype=REAL))
        self.df_eff = nn.Parameter(torch.linspace(-0.12, 0.12, n_heads, dtype=REAL))
        self.raw_beta_t = nn.Parameter(torch.linspace(4.0, -1.5, n_heads, dtype=REAL))
        self.raw_beta_tau = nn.Parameter(torch.linspace(3.0, -1.0, n_heads, dtype=REAL))
        self.gamma_ev = nn.Parameter(torch.full((n_heads,), 0.5, dtype=REAL))
        self.ev_proj = _rmat(d_scalar, n_heads)

        # C*-equivariant complex gated FFN + C*-invariant scalar FFN
        self.ff1 = _cmat(d_model, d_ff)
        self.ff2 = _cmat(d_ff, d_model)
        self.s_ff1 = _rmat(d_scalar, d_ff)
        self.s_fb1 = nn.Parameter(torch.zeros(d_ff, dtype=REAL))
        self.s_ff2 = _rmat(d_ff, d_scalar)

    def compute_symmetric_connection(
        self,
        tau_hat: torch.Tensor,  # [B, K]
        nu_post: torch.Tensor,  # [B, K]
        t_phys: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """Exact symmetric unitary connection P_ij = exp(i * [pi*(nu_i + nu_j)*(t_i - t_j) - 2*pi*df*(tau_i - tau_j)]).

        Guaranteed algebraically: P_ji == conj(P_ij) and P_ij * P_ji == 1 (< 1e-15).
        Returns `P [B, H, L, L]` complex128 where L = K * T.
        """
        b, k_len = tau_hat.shape
        t_len = t_phys.shape[1]
        l_tok = k_len * t_len

        tau_tok = tau_hat[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok)          # [B, L]
        nu_tok = nu_post[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok)           # [B, L]
        t_tok = t_phys[:, None, :].expand(b, k_len, t_len).reshape(b, l_tok)             # [B, L]

        dt = t_tok[:, :, None] - t_tok[:, None, :]                                       # [B, L, L] skew-symmetric
        dtau = tau_tok[:, :, None] - tau_tok[:, None, :]                                 # [B, L, L] skew-symmetric

        # Effective frequency per head and token: nu_tilde [B, H, L]
        nu_tilde = self.s_nu[None, :, None] * nu_tok[:, None, :] + self.omega_t[None, :, None]
        nu_sum = nu_tilde[:, :, :, None] + nu_tilde[:, :, None, :]                       # [B, H, L, L] symmetric!

        phi_sym = (
            math.pi * nu_sum * dt[:, None, :, :]
            - 2.0 * math.pi * self.df_eff[None, :, None, None] * dtau[:, None, :, :]
        )
        return torch.polar(torch.ones_like(phi_sym), phi_sym)  # [B, H, L, L] complex128

    def forward(
        self,
        h: torch.Tensor,        # [B, K, T, D] complex128
        s: torch.Tensor,        # [B, K, T, Ds] float64
        tau_hat: torch.Tensor,  # [B, K]
        nu_post: torch.Tensor,  # [B, K]
        t_phys: torch.Tensor,   # [B, T]
        t_norm: torch.Tensor,   # [B, T]
        ev_base: torch.Tensor,  # [B, K, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k_len, t_len, _ = h.shape
        l_tok = k_len * t_len

        h_u, _ = _safe_rms(h, dim=-1, keepdim=True)
        q = (h_u @ self.w_q).reshape(b, l_tok, self.n_heads, self.d_head).permute(0, 2, 1, 3)     # [B, H, L, Dh]
        k = (h_u @ self.w_k).reshape(b, l_tok, self.n_heads, self.d_head).permute(0, 2, 1, 3)     # [B, H, L, Dh]
        vc = (h @ self.w_v).reshape(b, l_tok, self.n_heads, self.d_head).permute(0, 2, 1, 3)      # 1-homogeneous!
        vs = (s @ self.w_sv).reshape(b, l_tok, self.n_heads, self.d_s_head).permute(0, 2, 1, 3)   # [B, H, L, Dsh]

        # Exact symmetric unitary parallel transport P_ij (transports fiber j -> fiber i)
        p_conn = self.compute_symmetric_connection(tau_hat, nu_post, t_phys)  # [B, H, L, L]

        # Gauge-invariant score: Re( <q_i, P_ij k_j> ) = Re( (q_i^H k_j) * P_ij )
        # Note: (q @ k.mH)[i, j] = sum_d q_{i,d} * conj(k_{j,d}); transporting k_j -> P_ij k_j
        # multiplies conj(k_j) by conj(P_ij), so <P_ij k_j, q_i> = (q @ k.mH) * conj(P_ij),
        # whose real part equals Re( (q @ k.mH) * conj(P_ij) ) = Re( <q_i, P_ij k_j> )!
        scale = 1.0 / math.sqrt(self.d_head)
        qk_transported = ((q @ k.mH) * p_conn.conj()).real * scale  # [B, H, L, L] strictly C*-invariant!

        tau_tok = tau_hat[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok)
        tn_tok = t_norm[:, None, :].expand(b, k_len, t_len).reshape(b, l_tok)
        dt_n = tn_tok[:, :, None] - tn_tok[:, None, :]
        dtau_n = (tau_tok[:, :, None] - tau_tok[:, None, :]) / float(OFFGRID_DELAY_MAX)
        beta_t = F.softplus(self.raw_beta_t)
        beta_tau = F.softplus(self.raw_beta_tau)
        geo_bias = (
            -beta_t[None, :, None, None] * dt_n.square()[:, None, :, :]
            - beta_tau[None, :, None, None] * dtau_n.square()[:, None, :, :]
        )

        ev_k = (
            self.gamma_ev[None, :, None, None] * ev_base[:, None, :, :]
            + (s @ self.ev_proj).permute(0, 3, 1, 2)
        ).reshape(b, self.n_heads, l_tok)

        logits = qk_transported + geo_bias + ev_k[:, :, None, :]
        weights = logits.softmax(dim=-1)  # [B, H, L, L] strictly C*-invariant!

        # Parallel-transport values from fiber j -> fiber i: sum_j A_ij * P_ij * v_j
        agg_c = (weights.to(COMPLEX) * p_conn) @ vc  # [B, H, L, Dh] strictly U(1)/C*-equivariant!
        agg_s = weights @ vs                         # [B, H, L, Dsh] strictly C*-invariant!

        delta_h = agg_c.permute(0, 2, 1, 3).reshape(b, k_len, t_len, self.d_model) @ self.w_o
        delta_s = agg_s.permute(0, 2, 1, 3).reshape(b, k_len, t_len, self.d_scalar) @ self.w_so

        # Residual + strictly C*-equivariant gated complex FFN
        h1 = h + delta_h
        h1_u, h1_r = _safe_rms(h1, dim=-1, keepdim=True)
        ff_h = h1_u @ self.ff1
        ff_gate = _safe_abs(ff_h).sigmoid()
        h_out = h1 + 0.5 * ((ff_h * ff_gate) @ self.ff2) * h1_r

        # Residual + strictly C*-invariant scalar FFN
        s1 = s + delta_s
        s_out = s1 + 0.5 * (F.silu(s1 @ self.s_ff1 + self.s_fb1) @ self.s_ff2)
        return h_out, s_out


class _SymmetricTransportBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        d_scalar: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 24,
    ):
        super().__init__()
        self.w_c = _cmat(S5_C_COMPLEX, d_model)
        self.w_s = _rmat(S5_C_SCALAR, d_scalar)
        self.b_s = nn.Parameter(torch.zeros(d_scalar, dtype=REAL))
        self.w_s2c = _rmat(d_scalar, d_model)

        self.layers = nn.ModuleList(
            [SymmetricTransportLayer(d_model=d_model, d_scalar=d_scalar, n_heads=n_heads, d_ff=d_ff) for _ in range(n_layers)]
        )

        # Strictly C*-invariant readout for family logits (4) + log-var (1)
        self.w_read_ref = _cmat(d_model, 4)
        self.w_read_bilin = _cmat(d_model, 4)
        self.w_head_scalar = _rmat(d_scalar + 6, N_FAMILIES + 1)
        self.b_head_scalar = nn.Parameter(torch.zeros(N_FAMILIES + 1, dtype=REAL))

        # Strictly U(1)/C*-equivariant (1-homogeneous!) complex residual gain readout
        self.w_head_complex = _cmat(d_model, 1)
        self.w_head_cgate = _rmat(d_scalar + 6, 1)
        self.b_head_cgate = nn.Parameter(torch.zeros(1, dtype=REAL))

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_post: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = F.silu(scalar_feats @ self.w_s + self.b_s)
        mod = 1.0 + torch.tanh(s @ self.w_s2c)
        h = (complex_feats @ self.w_c) * mod.to(COMPLEX)  # 1-homogeneous in complex_feats!

        for layer in self.layers:
            h, s = layer(h, s, tau_hat, nu_post, t_phys, t_norm, ev_base)

        ref = torch.stack(
            (complex_feats[..., 0], complex_feats[..., 1], complex_feats[..., 2], complex_feats[..., 5]),
            dim=-1,
        )
        # Scale-invariant normalized complex correlations for scalar head
        h_u, _ = _safe_rms(h, dim=-1, keepdim=True)
        ref_u, _ = _safe_rms(ref, dim=-1, keepdim=True)
        u_ref = h_u @ self.w_read_ref
        inv_ref = torch.asinh((ref_u.conj() * u_ref).real)
        u_b = h_u @ self.w_read_bilin
        inv_bilin = torch.asinh((u_b[..., :2].conj() * u_b[..., 2:]).real)

        head_in = torch.cat((s, inv_ref, inv_bilin), dim=-1)
        out_s = head_in @ self.w_head_scalar + self.b_head_scalar
        raw_fam = out_s[..., :N_FAMILIES]
        raw_lv = out_s[..., N_FAMILIES]

        # Strictly 1-homogeneous U(1)-equivariant complex residual gain: bounded gate * linear(h)
        c_gate = torch.sigmoid(head_in @ self.w_head_cgate + self.b_head_cgate).squeeze(-1)  # [B, K, T]
        raw_dg = (h @ self.w_head_complex).squeeze(-1) * c_gate.to(COMPLEX)                  # [B, K, T] complex128
        return raw_fam, raw_dg, raw_lv


class SymmetricTransportRefiner(nn.Module):
    """Role A: Exact Symmetric Parallel-Transport Gauge-Equivariant Stage-5 Refiner."""

    role: str = "role_a_symmetric_transport"

    def __init__(
        self,
        d_model: int = 16,
        d_scalar: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 24,
    ):
        super().__init__()
        self.cfg = dict(d_model=d_model, d_scalar=d_scalar, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        self.backbone = _SymmetricTransportBackbone(
            d_model=d_model, d_scalar=d_scalar, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
        )
        self.initial_map = copy.deepcopy(self.backbone)
        for p in self.initial_map.parameters():
            p.requires_grad_(False)
        # Dev-calibrated output scales (frozen during training at 1.0, optionally calibrated on dev)
        self.register_buffer("fam_scale", torch.tensor(1.0, dtype=REAL))
        self.register_buffer("res_scale", torch.tensor(1.0, dtype=REAL))

    def real_parameters(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def forward_tensors(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_post: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_fam, raw_dg, raw_lv = self.backbone(scalar_feats, complex_feats, tau_hat, nu_post, t_phys, t_norm, ev_base)
        with torch.no_grad():
            f0, dg0, lv0 = self.initial_map(
                scalar_feats.detach(),
                complex_feats.detach(),
                tau_hat.detach(),
                nu_post.detach(),
                t_phys.detach(),
                t_norm.detach(),
                ev_base.detach(),
            )
        delta_fam = 3.0 * torch.tanh((raw_fam - f0) / 3.0) * self.fam_scale
        # Smooth amplitude-capped residual relative to local complex feature RMS (preserves exact C* equivariance!)
        _, c_rms = _safe_rms(complex_feats, dim=-1, keepdim=False)  # [B, K, T]
        dg_diff = raw_dg - dg0
        dg_mag = _safe_abs(dg_diff)
        cap = 0.65 * (c_rms + 1e-6)
        # Smooth radial tanh preserves U(1) phase and exact 1-homogeneity (up to eps)!
        scale_rad = torch.tanh(dg_mag / cap) * (cap / dg_mag.clamp_min(1e-24))
        delta_g = dg_diff * scale_rad.to(COMPLEX) * self.res_scale.to(COMPLEX)
        delta_lv = 0.8 * torch.tanh((raw_lv - lv0) / 0.8)
        return delta_fam, delta_g, delta_lv


# ---------------------------------------------------------------------------
# 3. Role B: Equal-Budget, Exact-Parameter-Matched Non-Physics Strong Control
# ---------------------------------------------------------------------------


class _NonPhysicsTransformerLayer(nn.Module):
    """Standard Real Pre-LN Multi-Head Self-Attention + SiLU FFN (NO U(1) transport geometry)."""

    def __init__(self, d_hid: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_hid = d_hid
        self.n_heads = n_heads
        self.d_head = d_hid // n_heads
        self.ln1 = nn.LayerNorm(d_hid, dtype=REAL)
        self.qkv = nn.Linear(d_hid, 3 * d_hid, bias=False, dtype=REAL)
        self.out_proj = nn.Linear(d_hid, d_hid, bias=False, dtype=REAL)
        # Standard learned real pairwise relative positional bias (dt, dtau, nu_i, nu_j) -> n_heads
        self.rel_w1 = nn.Linear(4, n_heads * 2, bias=True, dtype=REAL)
        self.rel_w2 = nn.Linear(n_heads * 2, n_heads, bias=False, dtype=REAL)
        self.ln2 = nn.LayerNorm(d_hid, dtype=REAL)
        self.ff1 = nn.Linear(d_hid, d_ff, bias=True, dtype=REAL)
        self.ff2 = nn.Linear(d_ff, d_hid, bias=True, dtype=REAL)

    def forward(self, x: torch.Tensor, rel_feat: torch.Tensor) -> torch.Tensor:
        b, l_tok, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, l_tok, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, L, Dh]
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.d_head)
        rel_bias = self.rel_w2(F.silu(self.rel_w1(rel_feat))).permute(0, 3, 1, 2)  # [B, H, L, L]
        logits = (q @ k.transpose(-1, -2)) * scale + rel_bias
        w = logits.softmax(dim=-1)
        agg = (w @ v).permute(0, 2, 1, 3).reshape(b, l_tok, self.d_hid)
        x = x + self.out_proj(agg)
        x = x + self.ff2(F.silu(self.ff1(self.ln2(x))))
        return x


class _NonPhysicsTransformerBackbone(nn.Module):
    def __init__(self, d_hid: int, n_heads: int, n_layers: int, d_ff: int, extra_pad: int = 0):
        super().__init__()
        # Exact same inputs as Role A: scalar_feats (22) + Re/Im complex_feats (12) + ev_base (1) = 35 real channels
        in_dim = S5_C_SCALAR + 2 * S5_C_COMPLEX + 1
        self.in_proj = nn.Linear(in_dim, d_hid, bias=True, dtype=REAL)
        self.layers = nn.ModuleList([_NonPhysicsTransformerLayer(d_hid, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_hid, dtype=REAL)
        self.head = nn.Linear(d_hid, N_FAMILIES + 3, bias=True, dtype=REAL)  # 4 fam + Re(dg) + Im(dg) + lv
        # Optional tiny extra residual skip weight for exact integer parameter parity with Role A
        self.extra_pad = int(extra_pad)
        if self.extra_pad > 0:
            self.pad_w = nn.Parameter(torch.zeros(self.extra_pad, dtype=REAL))

    def forward(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_post: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k_len, t_len, _ = scalar_feats.shape
        l_tok = k_len * t_len
        x_in = torch.cat(
            (scalar_feats, complex_feats.real, complex_feats.imag, ev_base.unsqueeze(-1)),
            dim=-1,
        ).reshape(b, l_tok, -1)

        tau_tok = tau_hat[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok) / float(OFFGRID_DELAY_MAX)
        nu_tok = nu_post[:, :, None].expand(b, k_len, t_len).reshape(b, l_tok) / 0.15
        tn_tok = t_norm[:, None, :].expand(b, k_len, t_len).reshape(b, l_tok)
        dt = tn_tok[:, :, None] - tn_tok[:, None, :]
        dtau = tau_tok[:, :, None] - tau_tok[:, None, :]
        nu_i = nu_tok[:, :, None].expand(b, l_tok, l_tok)
        nu_j = nu_tok[:, None, :].expand(b, l_tok, l_tok)
        rel_feat = torch.stack((dt, dtau, nu_i, nu_j), dim=-1)  # [B, L, L, 4]

        x = F.silu(self.in_proj(x_in))
        for layer in self.layers:
            x = layer(x, rel_feat)
        out = self.head(self.ln_out(x)).reshape(b, k_len, t_len, N_FAMILIES + 3)
        if self.extra_pad > 0:
            # Tiny differentiable scalar coupling using pad_w (zero-centered via initial_map)
            n_use = min(self.extra_pad, x_in.shape[-1])
            pad_term = (x_in[..., :n_use] * self.pad_w[:n_use]).sum(dim=-1).reshape(b, k_len, t_len)
            out = out.clone()
            out[..., -1] = out[..., -1] + 0.01 * pad_term
        raw_fam = out[..., :N_FAMILIES]
        raw_dg = torch.complex(out[..., N_FAMILIES], out[..., N_FAMILIES + 1])
        raw_lv = out[..., N_FAMILIES + 2]
        return raw_fam, raw_dg, raw_lv


class NonPhysicsTransformerRefiner(nn.Module):
    """Role B: Exact-Parameter-Matched Standard Real Transformer (Equal Input & Budget Control)."""

    role: str = "role_b_nonphysics_transformer"

    def __init__(
        self,
        d_hid: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 44,
        extra_pad: int = 0,
    ):
        super().__init__()
        self.cfg = dict(d_hid=d_hid, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, extra_pad=extra_pad)
        self.backbone = _NonPhysicsTransformerBackbone(
            d_hid=d_hid, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, extra_pad=extra_pad
        )
        self.initial_map = copy.deepcopy(self.backbone)
        for p in self.initial_map.parameters():
            p.requires_grad_(False)
        self.register_buffer("fam_scale", torch.tensor(1.0, dtype=REAL))
        self.register_buffer("res_scale", torch.tensor(1.0, dtype=REAL))

    def real_parameters(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def forward_tensors(
        self,
        scalar_feats: torch.Tensor,
        complex_feats: torch.Tensor,
        tau_hat: torch.Tensor,
        nu_post: torch.Tensor,
        t_phys: torch.Tensor,
        t_norm: torch.Tensor,
        ev_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_fam, raw_dg, raw_lv = self.backbone(scalar_feats, complex_feats, tau_hat, nu_post, t_phys, t_norm, ev_base)
        with torch.no_grad():
            f0, dg0, lv0 = self.initial_map(
                scalar_feats.detach(),
                complex_feats.detach(),
                tau_hat.detach(),
                nu_post.detach(),
                t_phys.detach(),
                t_norm.detach(),
                ev_base.detach(),
            )
        delta_fam = 3.0 * torch.tanh((raw_fam - f0) / 3.0) * self.fam_scale
        _, c_rms = _safe_rms(complex_feats, dim=-1, keepdim=False)
        dg_diff = raw_dg - dg0
        dg_mag = _safe_abs(dg_diff)
        cap = 0.65 * (c_rms + 1e-6)
        scale_rad = torch.tanh(dg_mag / cap) * (cap / dg_mag.clamp_min(1e-24))
        delta_g = dg_diff * scale_rad.to(COMPLEX) * self.res_scale.to(COMPLEX)
        delta_lv = 0.8 * torch.tanh((raw_lv - lv0) / 0.8)
        return delta_fam, delta_g, delta_lv


def build_matched_stage5_models(
    *,
    d_model_a: int = 16,
    d_scalar_a: int = 16,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff_a: int = 24,
) -> tuple[SymmetricTransportRefiner, NonPhysicsTransformerRefiner]:
    """Construct Role A and Role B with EXACT integer real-parameter parity (`P_A == P_B`)."""
    model_a = SymmetricTransportRefiner(
        d_model=d_model_a, d_scalar=d_scalar_a, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff_a
    )
    target_p = model_a.real_parameters()

    # Search best (d_hid, d_ff) <= target_p with small remainder `extra_pad` (<= 35) for EXACT integer parity!
    best_cfg: Optional[tuple[int, int, int]] = None
    best_pad = 10**9
    for d_hid in range(20, 48, n_heads):
        for d_ff in range(16, 96):
            probe = NonPhysicsTransformerRefiner(d_hid=d_hid, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, extra_pad=0)
            p_b = probe.real_parameters()
            diff = target_p - p_b
            if 0 <= diff <= 35 and diff < best_pad:
                best_pad = diff
                best_cfg = (d_hid, d_ff, diff)
    if best_cfg is None:
        raise RuntimeError(f"Could not match Role B real parameter count to Role A ({target_p})")
    d_hid, d_ff, extra_pad = best_cfg
    model_b = NonPhysicsTransformerRefiner(
        d_hid=d_hid, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, extra_pad=extra_pad
    )
    assert model_a.real_parameters() == model_b.real_parameters(), (
        f"Parameter parity violated: Role A={model_a.real_parameters()} vs Role B={model_b.real_parameters()}"
    )
    return model_a, model_b


# ---------------------------------------------------------------------------
# 4. Inference Adapter (Single Model or Multi-Seed Ensemble)
# ---------------------------------------------------------------------------


def infer_stage5_fold_override(
    models: list[nn.Module],
    snap: FoldPosteriorSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one or more Stage-5 models (3-seed ensemble) on a fold snapshot using `torch.no_grad()`.

    Returns `(delta_fam [K, T, 4], delta_g_u [K, T, N], delta_log_var [K, T])` in numpy float64/complex128.
    """
    feats = extract_stage5_fold_features(snap)
    with torch.no_grad():
        s_t = torch.from_numpy(feats.scalar_feats).to(REAL).unsqueeze(0)
        c_t = torch.from_numpy(feats.complex_feats).to(COMPLEX).unsqueeze(0)
        tau_t = torch.from_numpy(feats.tau_hat).to(REAL).unsqueeze(0)
        nu_t = torch.from_numpy(feats.nu_post).to(REAL).unsqueeze(0)
        tp_t = torch.from_numpy(feats.t_phys).to(REAL).unsqueeze(0)
        tn_t = torch.from_numpy(feats.t_norm).to(REAL).unsqueeze(0)
        ev_t = torch.from_numpy(feats.ev_base).to(REAL).unsqueeze(0)

        fam_acc = None
        dg_acc = None
        lv_acc = None
        for m in models:
            df, dg, dlv = m.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t)
            fam_acc = df if fam_acc is None else (fam_acc + df)
            dg_acc = dg if dg_acc is None else (dg_acc + dg)
            lv_acc = dlv if lv_acc is None else (lv_acc + dlv)
        inv_n = 1.0 / float(len(models))
        delta_fam = (fam_acc * inv_n).squeeze(0).cpu().numpy()
        delta_g_bar = (dg_acc * inv_n).squeeze(0).cpu().numpy()  # [K, T] complex128
        delta_lv = (lv_acc * inv_n).squeeze(0).cpu().numpy()     # [K, T] float64

    # Expand per-row complex residual across the 32 samples via public intra-row phase profile
    delta_g_full = delta_g_bar[:, :, None] * feats.rot_intra     # [K, T, N] complex128
    return delta_fam, delta_g_full, delta_lv


# ---------------------------------------------------------------------------
# 5. Strictly C*-Invariant Differentiable Training Loss
# ---------------------------------------------------------------------------


def stage5_differentiable_loss(
    delta_fam: torch.Tensor,         # [B, K, T, 4] float64
    delta_g_bar: torch.Tensor,       # [B, K, T] complex128
    delta_lv: torch.Tensor,          # [B, K, T] float64
    *,
    t_rows_idx: torch.Tensor,        # [B, 6] long owned target row indices
    probe_row_idx: torch.Tensor,     # [B] long probe row index
    rot_intra_t: torch.Tensor,       # [B, K, 6, N] complex128
    rot_intra_p: torch.Tensor,       # [B, K, N] complex128
    fam_logw: torch.Tensor,          # [B, K, 4] float64 (-1e9 for inactive families)
    fam_active: torch.Tensor,        # [B, K, 4] bool
    fam_mean_t: torch.Tensor,        # [B, K, 4, 6, N] complex128
    fam_var_t: torch.Tensor,         # [B, K, 4, 6, N] float64
    fam_mean_p: torch.Tensor,        # [B, K, 4, N] complex128
    fam_var_p: torch.Tensor,         # [B, K, 4, N] float64
    d_mats: torch.Tensor,            # [B, K, N, N] complex128
    phi_f: torch.Tensor,             # [B, K, N] complex128
    h_true_u: torch.Tensor,          # [B, 6, N, N] complex128
    y_target_freq_u: torch.Tensor,   # [B, 6, N] complex128 unitary FFT of y_target_u
    b_code_target: torch.Tensor,     # [B, 6, N, 4] float64
    u_probe: torch.Tensor,           # [B, K, N] complex128
    probe_clean_u: torch.Tensor,     # [B, N] complex128
    w_mse: float = 1.0,
    w_bce: float = 0.60,
    w_probe: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact C*-invariant Stage-5 training loss: normalized channel MSE + mixture bit-BCE + probe NLL."""
    b, k_len, _, _ = delta_fam.shape
    r_len = t_rows_idx.shape[1]

    # Gather target-row neural outputs: [B, K, 6, ...]
    idx_k4 = t_rows_idx[:, None, :, None].expand(b, k_len, r_len, N_FAMILIES)
    df_t = torch.gather(delta_fam, dim=2, index=idx_k4)                          # [B, K, 6, 4]
    idx_k = t_rows_idx[:, None, :].expand(b, k_len, r_len)
    dg_t_bar = torch.gather(delta_g_bar, dim=2, index=idx_k)                     # [B, K, 6]
    dlv_t = torch.gather(delta_lv, dim=2, index=idx_k)                           # [B, K, 6]

    dg_t = dg_t_bar.unsqueeze(-1) * rot_intra_t                                  # [B, K, 6, N]
    vs_t = torch.exp(dlv_t).unsqueeze(-1)                                        # [B, K, 6, 1]

    # Updated per-ray, per-row family weights w_f [B, K, 6, 4] (inactive families strictly masked!)
    lw_raw = fam_logw[:, :, None, :] + df_t                                      # [B, K, 6, 4]
    lw_masked = torch.where(fam_active[:, :, None, :], lw_raw, torch.full_like(lw_raw, -1e9))
    w_f = lw_masked.softmax(dim=-1)                                              # [B, K, 6, 4]

    # Reconstruct refined per-ray posterior mean & variance on the 6 owned target rows
    # fam_mean_t: [B, K, 4, 6, N] -> permute to [B, K, 6, 4, N]
    fm_t = fam_mean_t.permute(0, 1, 3, 2, 4)                                     # [B, K, 6, 4, N]
    fv_t = fam_var_t.permute(0, 1, 3, 2, 4)                                      # [B, K, 6, 4, N]
    mu_ray_t = (w_f.unsqueeze(-1).to(COMPLEX) * fm_t).sum(dim=3) + dg_t          # [B, K, 6, N]
    within_v = (w_f.unsqueeze(-1) * fv_t).sum(dim=3)
    between_v = (w_f.unsqueeze(-1) * (fm_t - (mu_ray_t - dg_t).unsqueeze(3)).abs().square()).sum(dim=3)
    var_ray_t = (within_v + between_v).clamp_min(1e-12) * vs_t                   # [B, K, 6, N]

    # 1) Normalized channel Frobenius MSE (strictly C*-invariant!)
    # H_hat[b, r, n, m] = sum_k mu_ray_t[b, k, r, n] * d_mats[b, k, n, m]
    h_hat_u = torch.einsum("bkrn,bknm->brnm", mu_ray_t, d_mats)                  # [B, 6, N, N]
    diff_h = h_hat_u - h_true_u
    err_frob2 = (diff_h.real.square() + diff_h.imag.square()).sum(dim=(-2, -1))  # [B, 6]
    true_frob2 = (h_true_u.real.square() + h_true_u.imag.square()).sum(dim=(-2, -1))
    loss_mse = (err_frob2 / (true_frob2 + 1.0)).mean()

    # 2) Differentiable Dominant-Ray Family-Mixture Bit-LLR BCE (matches downstream mixture LLR decoding!)
    # Identify dominant ray per fold by posterior signal energy (detached index)
    with torch.no_grad():
        ray_e = (fm_t.real.square() + fm_t.imag.square()).mean(dim=(2, 3, 4))    # [B, K]
        dom_idx = torch.argmax(ray_e, dim=1)                                     # [B]
    b_ar = torch.arange(b, device=delta_fam.device)

    # Row-mean gains for fast unitary freq-domain LMMSE per dominant-ray family m in {0,1,2,3}
    mu_bar_all = mu_ray_t.mean(dim=-1)                                           # [B, K, 6]
    var_bar_all = var_ray_t.mean(dim=-1)                                         # [B, K, 6]
    mu_dom_tot = mu_bar_all[b_ar, dom_idx]                                       # [B, 6]
    var_dom_tot = var_bar_all[b_ar, dom_idx]                                     # [B, 6]
    phi_dom = phi_f[b_ar, dom_idx]                                               # [B, N]

    # Base frequency channel from non-dominant rays: [B, 6, N]
    h_freq_all = torch.einsum("bkr,bkn->brn", mu_bar_all, phi_f)
    h_freq_other = h_freq_all - mu_dom_tot.unsqueeze(-1) * phi_dom.unsqueeze(1)
    var_other = (var_bar_all.sum(dim=1) - var_dom_tot).clamp_min(0.0)            # [B, 6]

    # Dominant ray family means/vars/weights: [B, 6, 4]
    fm_dom_bar = (fm_t[b_ar, dom_idx] + dg_t[b_ar, dom_idx].unsqueeze(2)).mean(dim=-1)
    fv_dom_bar = fv_t[b_ar, dom_idx].mean(dim=-1) * torch.exp(dlv_t[b_ar, dom_idx]).unsqueeze(-1)
    log_w_dom = F.log_softmax(lw_masked[b_ar, dom_idx], dim=-1)                  # [B, 6, 4]

    # Per-family frequency channel: [B, 6, 4, N]
    h_f_m = h_freq_other.unsqueeze(2) + fm_dom_bar.unsqueeze(-1) * phi_dom[:, None, None, :]
    nv_m = (1.0 + var_other.unsqueeze(-1) + fv_dom_bar).clamp_min(1e-12)         # [B, 6, 4]
    h2_m = h_f_m.real.square() + h_f_m.imag.square()
    denom_m = (h2_m + nv_m.unsqueeze(-1)).clamp_min(1e-12)
    wf_m = h_f_m.conj() / denom_m
    x_hat_m = (wf_m * y_target_freq_u.unsqueeze(2)) @ _F_H_TORCH.to(device=delta_fam.device)  # [B, 6, 4, N]
    post_var_m = (nv_m.unsqueeze(-1) / denom_m).mean(dim=-1).clamp(1e-12, 1.0)   # [B, 6, 4]
    gain_m = 1.0 - post_var_m

    pam = _PAM_TORCH.to(device=delta_fam.device)
    axes_m = torch.stack((x_hat_m.real, x_hat_m.imag), dim=-1)                   # [B, 6, 4, N, 2]
    logp_m = (
        2.0 * axes_m.unsqueeze(-1) * pam - gain_m[..., None, None, None] * (pam * pam)
    ) / post_var_m[..., None, None, None]
    sign_l = torch.logaddexp(logp_m[..., 2], logp_m[..., 3]) - torch.logaddexp(logp_m[..., 0], logp_m[..., 1])
    inner_l = torch.logaddexp(logp_m[..., 1], logp_m[..., 3]) - torch.logaddexp(logp_m[..., 0], logp_m[..., 2])
    llrs_m = torch.stack((sign_l, inner_l), dim=-1).reshape(b, r_len, N_FAMILIES, N, 4)

    # Exact differentiable probability-domain mixture bit-LLR (log-sum-exp!)
    log_p1_m = -F.softplus(-llrs_m)
    log_p0_m = -F.softplus(llrs_m)
    lw_b = log_w_dom[..., None, None]
    mix_llrs = (torch.logsumexp(lw_b + log_p1_m, dim=2) - torch.logsumexp(lw_b + log_p0_m, dim=2)).clamp(-40.0, 40.0)
    signed = 2.0 * b_code_target.reshape(b, r_len, N, 4) - 1.0
    loss_bce = F.softplus(-signed * mix_llrs).mean()

    # 3) Calibrated Probe NLL (strictly C*-invariant!)
    p_idx_k4 = probe_row_idx[:, None, None, None].expand(b, k_len, 1, N_FAMILIES)
    df_p = torch.gather(delta_fam, dim=2, index=p_idx_k4).squeeze(2)             # [B, K, 4]
    p_idx_k = probe_row_idx[:, None, None].expand(b, k_len, 1)
    dg_p_bar = torch.gather(delta_g_bar, dim=2, index=p_idx_k).squeeze(2)        # [B, K]
    dlv_p = torch.gather(delta_lv, dim=2, index=p_idx_k).squeeze(2)              # [B, K]
    lw_p = torch.where(fam_active, fam_logw + df_p, torch.full_like(fam_logw, -1e9))
    w_p = lw_p.softmax(dim=-1)
    dg_p = dg_p_bar.unsqueeze(-1) * rot_intra_p
    mu_p = (w_p.unsqueeze(-1).to(COMPLEX) * fam_mean_p).sum(dim=2) + dg_p        # [B, K, N]
    v_p = (
        (w_p.unsqueeze(-1) * fam_var_p).sum(dim=2)
        + (w_p.unsqueeze(-1) * (fam_mean_p - (mu_p - dg_p).unsqueeze(2)).abs().square()).sum(dim=2)
    ).clamp_min(1e-12) * torch.exp(dlv_p).unsqueeze(-1)
    probe_pred = (mu_p * u_probe).sum(dim=1)                                     # [B, N]
    probe_var = (1.0 + (v_p * u_probe.abs().square()).sum(dim=1)).clamp_min(1e-12)
    # Clean probe target + unit noise expectation: E[|y_noisy - pred|^2] = |y_clean - pred|^2 + 1
    p_err2 = (probe_clean_u - probe_pred).abs().square() + 1.0
    loss_probe = (torch.log(probe_var) + p_err2 / probe_var).mean()

    total = float(w_mse) * loss_mse + float(w_bce) * loss_bce + float(w_probe) * loss_probe
    return total, loss_mse, loss_bce, loss_probe


__all__ = [
    "COMPLEX",
    "N_FAMILIES",
    "NonPhysicsTransformerRefiner",
    "REAL",
    "S5_C_COMPLEX",
    "S5_C_SCALAR",
    "Stage5FoldFeatures",
    "SymmetricTransportLayer",
    "SymmetricTransportRefiner",
    "build_matched_stage5_models",
    "compute_family_summaries",
    "extract_stage5_fold_features",
    "infer_stage5_fold_override",
    "stage5_differentiable_loss",
]
