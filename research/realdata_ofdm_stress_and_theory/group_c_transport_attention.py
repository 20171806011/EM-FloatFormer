"""Group C: Physics-Structured Parallel-Transport Attention (Role A) vs Matched Non-Physics Baselines (Role B).

Strictly frozen frontend:
  - All three models consume the BYTE-IDENTICAL R=17 shared physical hypothesis bank & blind features
    produced by `hypothesis_frontend.build_shared_hypothesis_bundle` (zero label leakage, zero test leakage).
  - All three models output the identical physical interface:
      (delta_logw [B, 14, 17], delta_phi [B, 14, 17], log_nv_scale [B, 14, 1])
    feeding the EXACT SAME differentiable mixture-posterior 16-QAM bit LLR layer!

Controlled architectural comparison (exact trainable real-parameter parity within ~1%):
  - Role A (`PhysParallelTransportAttn`):
      * Strictly U(1)-equivariant complex channel stream (zero complex biases!) + C*-invariant scalar stream
      * Symmetric parallel-transport temporal attention across the 14 OFDM symbols with exact
        conjugate-symmetric U(1) connection P_{s,s'} = exp(-j * Omega_h * (s - s')), P_{s',s} = P_{s,s'}^*
        applied to Q, K, AND V + continuous temporal geodesic bias
      * Cross-hypothesis content-addressing attention across all R=17 hypotheses
      * U(1)-equivariant continuous off-grid phase residual readout via complex phasor atan2 +
        C*-invariant weight & variance-calibration readouts
      * Step-0 zero-residual anchoring so training starts smoothly from the physical baseline
  - Role B1 (`MatchedRealTransformer`):
      * Identical inputs (4 complex features unpacked to 8 real channels + 10 scalar = 18 real channels)
      * Standard real-valued 2-axis Transformer (temporal self-attention with learned relative position bias
        + cross-hypothesis self-attention + real FFN), NO parallel-transport connection, NO U(1) equivariance
      * Parameter-matched to Role A
  - Role B2 (`MatchedRealMLP`):
      * Identical 18 real channels + global temporal & hypothesis context pooling + residual SiLU MLP
      * Parameter-matched to Role A

Strict experimental protocol:
  - 3 independent training seeds: 20261121, 20261122, 20261123
  - Training strictly on Train split (TTIs 0..339) across all 5 conditions (C0 x1 + C1..C4 x2 = 3060 grids)
  - Early stopping strictly on Dev split (TTIs 360..481, 610 grids) minimizing Dev Bit NLL
  - Post-hoc single-scalar LLR temperature calibration fitted strictly on Dev
  - Final paired evaluation on frozen Test split (TTIs 502..741, 12 contiguous blocks x 20 TTIs, 1200 grids)
  - Reports parameter counts, train time, single-thread CPU inference latency, peak RSS (MB),
    raw & calibrated NLL/ECE, BER/SER across all 5 conditions, across-seed mean +- std,
    and 12-block exact sign-flip tests!
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

from hypothesis_frontend import (
    HYP_LOG_PRIOR,
    JUMP_RAD_GRID,
    OMEGA_GRID,
    R_HYP,
    build_shared_hypothesis_bundle,
    build_train_reference_bank,
    hmm_smooth_log_posterior,
)
from ofdm_data import (
    BIT_ONE_IDX,
    BIT_ZERO_IDX,
    CONST16,
    DATA_COORDS_101,
    DATA_MASK_101,
    N_SYM,
    PILOT_SYM,
    USED_SCS,
    RealOFDMDataset,
    build_oracle_channel,
    build_temporal_splits,
    compute_bit_metrics,
    exact_sign_flip_test,
    load_real_ofdm_dataset,
    save_json,
)
from stress_channel import STRESS_CONDITIONS, apply_stress_condition, condition_seed

SEEDS: Tuple[int, ...] = (20261121, 20261122, 20261123)
MAX_DELTA_PHI: float = math.pi / 4.0  # +-45 deg continuous off-grid residual phase range

# Precomputed torch constants for fast exact differentiable mixture 16-QAM LLR layer
_CONST16_RE = torch.tensor(CONST16.real, dtype=torch.float32)  # [16]
_CONST16_IM = torch.tensor(CONST16.imag, dtype=torch.float32)  # [16]
_CONST16_ABS2 = _CONST16_RE.square() + _CONST16_IM.square()    # [16]
_BIT_ONE_T = torch.tensor(BIT_ONE_IDX, dtype=torch.long)       # [4, 8]
_BIT_ZERO_T = torch.tensor(BIT_ZERO_IDX, dtype=torch.long)     # [4, 8]
_DATA_RS_T = torch.tensor(DATA_COORDS_101[:, 0], dtype=torch.long)  # [1400]
_DATA_KS_T = torch.tensor(DATA_COORDS_101[:, 1], dtype=torch.long)  # [1400]

# Exact hypothesis phase table [17, 14] to reconstruct h_bank on-the-fly (<5x RAM!)
def _build_hyp_phase_table() -> Tuple[torch.Tensor, torch.Tensor]:
    s_idx = np.arange(N_SYM, dtype=np.float64)
    dt_s = s_idx - float(PILOT_SYM)
    non_p = (np.arange(N_SYM) != PILOT_SYM).astype(np.float64)
    ph = np.zeros((R_HYP, N_SYM), dtype=np.float64)
    for i, om in enumerate(OMEGA_GRID):
        ph[1 + i] = om * dt_s
    off = 1 + len(OMEGA_GRID)
    for i, jr in enumerate(JUMP_RAD_GRID):
        ph[off + i] = jr * non_p
    return torch.tensor(np.cos(ph), dtype=torch.float32), torch.tensor(np.sin(ph), dtype=torch.float32)

_HYP_COS_RS, _HYP_SIN_RS = _build_hyp_phase_table()  # [17, 14] each


def count_real_params(model: nn.Module) -> int:
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)


def peak_rss_mb() -> float:
    # Linux getrusage ru_maxrss is in kilobytes
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


# ---------------------------------------------------------------------------
# Fast Exact Differentiable Mixture 16-QAM Bit LLR Layer (Shared by ALL Roles)
# ---------------------------------------------------------------------------


def differentiable_mixture_llrs_compact(
    y_re_d: torch.Tensor,      # [B, 1400] Re(y) on the 1400 data REs
    y_im_d: torch.Tensor,      # [B, 1400] Im(y) on the 1400 data REs
    hls_re_d: torch.Tensor,    # [B, 1400] Re(h_ls) gathered at DATA_KS
    hls_im_d: torch.Tensor,    # [B, 1400] Im(h_ls) gathered at DATA_KS
    hca_re_d: torch.Tensor,    # [B, 1400] Re(h_ca) gathered at DATA_KS
    hca_im_d: torch.Tensor,    # [B, 1400] Im(h_ca) gathered at DATA_KS
    nv_d: torch.Tensor,        # [B, 1400] float32 NBI-guarded noise var on data REs
    log_weights_sr: torch.Tensor,  # [B, 14, 17] float32 normalized log-posterior over R
    delta_phi_sr: torch.Tensor,    # [B, 14, 17] float32 continuous residual phase (rad)
    log_nv_scale_s: torch.Tensor,  # [B, 14, 1] float32 per-symbol log noise variance calibration
) -> torch.Tensor:
    """Memory-lean, bit-exact differentiable mixture-posterior 16-QAM bit LLR layer on 1400 data REs.

    Reconstructs the R=17 rotated channel bank on-the-fly directly on the 1400 data REs,
    keeping peak RAM < 600 MB while matching `mixture_16qam_llrs` to float32 precision (<2e-5).
    """
    b_sz = y_re_d.shape[0]
    rs = _DATA_RS_T  # [1400]

    # Effective per-RE inverse noise variance
    nv_scale_d = torch.exp(log_nv_scale_s.clamp(-1.0, 2.0)[:, rs, 0])  # [B, 1400]
    inv_nv_d = 1.0 / (nv_d * nv_scale_d).clamp_min(1e-8)               # [B, 1400]

    # Base complex products y * conj(h_ls) and y * conj(h_ca): [B, 1400]
    yh_ls_re = y_re_d * hls_re_d + y_im_d * hls_im_d
    yh_ls_im = y_im_d * hls_re_d - y_re_d * hls_im_d
    h2_ls = hls_re_d.square() + hls_im_d.square()

    yh_ca_re = y_re_d * hca_re_d + y_im_d * hca_im_d
    yh_ca_im = y_im_d * hca_re_d - y_re_d * hca_im_d
    h2_ca = hca_re_d.square() + hca_im_d.square()

    # Build [B, 17, 1400] unrotated (y * conj(h_base)) & |h_base|^2: r=0..15 use h_ls, r=16 uses h_ca
    # Total phase applied to hypothesis r at symbol s is hyp_phase[r, s] + delta_phi[b, s, r]!
    # So rotation on y * conj(H_r) is exp(-j * (hyp_phase[r, s] + delta_phi[b, s, r]))!
    dphi_rs = delta_phi_sr.permute(0, 2, 1)  # [B, 17, 14]
    cos_dp = torch.cos(dphi_rs)
    sin_dp = torch.sin(dphi_rs)
    # exp(j*(hyp + dp)) = (cos_h cos_dp - sin_h sin_dp) + j (sin_h cos_dp + cos_h sin_dp)
    cos_tot = (_HYP_COS_RS[None, :, :] * cos_dp - _HYP_SIN_RS[None, :, :] * sin_dp)[:, :, rs]  # [B, 17, 1400]
    sin_tot = (_HYP_SIN_RS[None, :, :] * cos_dp + _HYP_COS_RS[None, :, :] * sin_dp)[:, :, rs]  # [B, 17, 1400]

    yh_base_re = torch.cat([yh_ls_re.unsqueeze(1).expand(b_sz, R_HYP - 1, 1400), yh_ca_re.unsqueeze(1)], dim=1)
    yh_base_im = torch.cat([yh_ls_im.unsqueeze(1).expand(b_sz, R_HYP - 1, 1400), yh_ca_im.unsqueeze(1)], dim=1)
    h2_d = torch.cat([h2_ls.unsqueeze(1).expand(b_sz, R_HYP - 1, 1400), h2_ca.unsqueeze(1)], dim=1)

    # Rotate by exp(-j * tot_phase): (a + jb)(c - jd) = (ac + bd) + j(bc - ad)
    zre_d = yh_base_re * cos_tot + yh_base_im * sin_tot  # [B, 17, 1400]
    zim_d = yh_base_im * cos_tot - yh_base_re * sin_tot  # [B, 17, 1400]
    lw_d = log_weights_sr.permute(0, 2, 1)[:, :, rs]     # [B, 17, 1400]

    quad = (
        h2_d.unsqueeze(-1) * _CONST16_ABS2.view(1, 1, 1, 16)
        - 2.0 * (
            zre_d.unsqueeze(-1) * _CONST16_RE.view(1, 1, 1, 16)
            + zim_d.unsqueeze(-1) * _CONST16_IM.view(1, 1, 1, 16)
        )
    )  # [B, 17, 1400, 16]
    log_joint = lw_d.unsqueeze(-1) - quad * inv_nv_d.view(b_sz, 1, 1400, 1)
    log_pm = torch.logsumexp(log_joint, dim=1)  # [B, 1400, 16]

    llrs = []
    for b in range(4):
        s1 = torch.logsumexp(log_pm.index_select(-1, _BIT_ONE_T[b]), dim=-1)
        s0 = torch.logsumexp(log_pm.index_select(-1, _BIT_ZERO_T[b]), dim=-1)
        llrs.append(s1 - s0)
    return torch.stack(llrs, dim=-1)  # [B, 1400, 4]


# ---------------------------------------------------------------------------
# Role A: Physics-Structured Symmetric Parallel-Transport Attention
# ---------------------------------------------------------------------------


def _init_cmat(in_d: int, out_d: int) -> Tuple[nn.Parameter, nn.Parameter]:
    scale = 1.0 / math.sqrt(2.0 * in_d)
    wr = nn.Parameter(torch.randn(in_d, out_d, dtype=torch.float32) * scale)
    wi = nn.Parameter(torch.randn(in_d, out_d, dtype=torch.float32) * scale)
    return wr, wi


def _cmul_lin(xr: torch.Tensor, xi: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bias-free complex linear map (xr + j xi) @ (wr + j wi): strictly U(1)-equivariant!"""
    return xr @ wr - xi @ wi, xr @ wi + xi @ wr


def _crms_norm(xr: torch.Tensor, xi: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """U(1)-equivariant complex RMS normalization."""
    p = (xr.square() + xi.square()).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(p + eps)
    return xr / rms, xi / rms, rms


class SymmetricParallelTransportBlock(nn.Module):
    """Exact U(1) gauge-equivariant temporal parallel-transport attention + cross-hypothesis content attention.

    Parallel-transport connection across OFDM symbols s, s' in {0..13}:
      P^{(h)}_{s, s'} = exp(-j * Omega_h * (s - s'))
    Explicitly satisfies conjugate symmetry P_{s', s} = conj(P_{s, s'}) and unitarity |P| = 1.
    """

    def __init__(self, d_c: int = 16, d_s: int = 24, n_heads: int = 4):
        super().__init__()
        assert d_c % n_heads == 0 and d_s % n_heads == 0
        self.d_c = d_c
        self.d_s = d_s
        self.n_heads = n_heads
        self.dh_c = d_c // n_heads
        self.dh_s = d_s // n_heads

        # Complex Q, K, V, O projections (strictly zero-bias -> exact U(1) equivariance!)
        self.wq_r, self.wq_i = _init_cmat(d_c, d_c)
        self.wk_r, self.wk_i = _init_cmat(d_c, d_c)
        self.wv_r, self.wv_i = _init_cmat(d_c, d_c)
        self.wo_r, self.wo_i = _init_cmat(d_c, d_c)

        # Scalar projections for temporal attention
        self.w_sq = nn.Linear(d_s, d_s, bias=False)
        self.w_sk = nn.Linear(d_s, d_s, bias=False)
        self.w_sv = nn.Linear(d_s, d_s, bias=False)
        self.w_so = nn.Linear(d_s, d_s, bias=False)

        # Head-wise symmetric parallel-transport connection rates Omega_h & geodesic decay beta_h
        init_om = torch.tensor([-0.24, -0.08, 0.08, 0.24], dtype=torch.float32)[:n_heads]
        init_bt = torch.tensor([4.0, 1.0, -1.0, -3.0], dtype=torch.float32)[:n_heads]
        self.omega_h = nn.Parameter(init_om)
        self.raw_beta_h = nn.Parameter(init_bt)
        self.alpha_c_score = nn.Parameter(torch.full((n_heads,), 1.0, dtype=torch.float32))
        self.alpha_s_score = nn.Parameter(torch.full((n_heads,), 0.5, dtype=torch.float32))

        # Cross-hypothesis content-addressing attention (along R=17)
        self.hq = nn.Linear(d_s, d_s, bias=False)
        self.hk = nn.Linear(d_s, d_s, bias=False)
        self.hv = nn.Linear(d_s, d_s, bias=False)
        self.ho = nn.Linear(d_s, d_s, bias=False)
        self.h_temp = nn.Parameter(torch.ones(n_heads, dtype=torch.float32))

        # U(1)-equivariant gated complex FFN + C*-invariant scalar FFN
        self.cff1_r, self.cff1_i = _init_cmat(d_c, d_c)
        self.cff2_r, self.cff2_i = _init_cmat(d_c, d_c)
        self.sff = nn.Sequential(
            nn.LayerNorm(d_s),
            nn.Linear(d_s, d_s * 2),
            nn.SiLU(),
            nn.Linear(d_s * 2, d_s),
        )
        # Scalar-to-complex U(1)-equivariant real modulation gate
        self.s2c_gate = nn.Linear(d_s, d_c)

    def parallel_transport_matrix(self, s_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos P, sin P) of shape [H, S, S] where P_{s,s'} = exp(-j * omega_h * (s - s'))."""
        idx = torch.arange(s_len, device=device, dtype=torch.float32)
        dt = idx[:, None] - idx[None, :]  # [S, S], antisymmetric: dt[s', s] = -dt[s, s']
        ang = -self.omega_h[:, None, None] * dt[None, :, :]  # [H, S, S]
        return torch.cos(ang), torch.sin(ang)

    def forward(
        self, hr: torch.Tensor, hi: torch.Tensor, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hr, hi: [B, S, R, Dc]; s: [B, S, R, Ds]."""
        b, s_len, r_len, _ = hr.shape
        h_cnt = self.n_heads

        # --- Stage 1: Temporal Parallel-Transport Attention along S=14 (for each r independently) ---
        # Reshape [B, S, R, D] -> [B*R, S, D]
        hr_br = hr.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_c)
        hi_br = hi.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_c)
        s_br = s.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_s)

        hur, hui, _ = _crms_norm(hr_br, hi_br)
        qr, qi = _cmul_lin(hur, hui, self.wq_r, self.wq_i)
        kr, ki = _cmul_lin(hur, hui, self.wk_r, self.wk_i)
        vr, vi = _cmul_lin(hr_br, hi_br, self.wv_r, self.wv_i)

        # Split heads -> [BR, H, S, dh_c]
        def _sh(x, dh):
            return x.view(b * r_len, s_len, h_cnt, dh).permute(0, 2, 1, 3)

        qr, qi = _sh(qr, self.dh_c), _sh(qi, self.dh_c)
        kr, ki = _sh(kr, self.dh_c), _sh(ki, self.dh_c)
        vr, vi = _sh(vr, self.dh_c), _sh(vi, self.dh_c)

        # Complex inner product <q_s, k_{s'}> = sum_d (qr_s kr_{s'} + qi_s ki_{s'}) + j(qi_s kr_{s'} - qr_s ki_{s'})
        # Wait: Re<q_s, P_{s,s'} k_{s'}> where <a, b> = a^H b = conj(a) * b:
        # Let dot = conj(q_s) * k_{s'} = (qr kr + qi ki) + j (qr ki - qi kr)
        # Then Re( conj(q_s) * P_{s,s'} k_{s'} ) = Re(dot) * cosP - Im(dot) * sinP!
        dot_re = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)  # [BR, H, S, S]
        dot_im = qr @ ki.transpose(-1, -2) - qi @ kr.transpose(-1, -2)  # [BR, H, S, S]
        cos_p, sin_p = self.parallel_transport_matrix(s_len, hr.device)  # [H, S, S]
        c_score = (dot_re * cos_p[None, :, :, :] - dot_im * sin_p[None, :, :, :]) / math.sqrt(self.dh_c)

        # Scalar QK score + continuous geodesic temporal bias
        sq = _sh(self.w_sq(s_br), self.dh_s)
        sk = _sh(self.w_sk(s_br), self.dh_s)
        sv = _sh(self.w_sv(s_br), self.dh_s)
        s_score = (sq @ sk.transpose(-1, -2)) / math.sqrt(self.dh_s)

        idx = torch.arange(s_len, device=hr.device, dtype=torch.float32)
        dt_norm = (idx[:, None] - idx[None, :]) / float(max(s_len - 1, 1))
        beta = F.softplus(self.raw_beta_h)  # [H]
        geo_bias = -beta[None, :, None, None] * (dt_norm.square())[None, None, :, :]

        logits = (
            self.alpha_c_score[None, :, None, None] * c_score
            + self.alpha_s_score[None, :, None, None] * s_score
            + geo_bias
        )
        attn = F.softmax(logits, dim=-1)  # [BR, H, S, S]

        # Parallel transport of complex values: sum_{s'} A_{s,s'} P_{s,s'} v_{s'}
        w_re = attn * cos_p[None, :, :, :]
        w_im = attn * sin_p[None, :, :, :]
        agg_r = w_re @ vr - w_im @ vi  # [BR, H, S, dh_c]
        agg_i = w_re @ vi + w_im @ vr  # [BR, H, S, dh_c]
        agg_s = attn @ sv              # [BR, H, S, dh_s]

        agg_r = agg_r.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_c)
        agg_i = agg_i.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_c)
        agg_s = agg_s.permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_s)

        dhr, dhi = _cmul_lin(agg_r, agg_i, self.wo_r, self.wo_i)
        ds_t = self.w_so(agg_s)

        hr1 = hr + dhr.view(b, r_len, s_len, self.d_c).permute(0, 2, 1, 3)
        hi1 = hi + dhi.view(b, r_len, s_len, self.d_c).permute(0, 2, 1, 3)
        s1 = s + ds_t.view(b, r_len, s_len, self.d_s).permute(0, 2, 1, 3)

        # --- Stage 2: Cross-Hypothesis Content-Addressing Attention along R=17 ---
        s_bs = s1.reshape(b * s_len, r_len, self.d_s)
        hq = self.hq(s_bs).view(b * s_len, r_len, h_cnt, self.dh_s).permute(0, 2, 1, 3)
        hk = self.hk(s_bs).view(b * s_len, r_len, h_cnt, self.dh_s).permute(0, 2, 1, 3)
        hv = self.hv(s_bs).view(b * s_len, r_len, h_cnt, self.dh_s).permute(0, 2, 1, 3)
        h_logits = (hq @ hk.transpose(-1, -2)) * (F.softplus(self.h_temp)[None, :, None, None] / math.sqrt(self.dh_s))
        h_attn = F.softmax(h_logits, dim=-1)
        h_agg = (h_attn @ hv).permute(0, 2, 1, 3).reshape(b * s_len, r_len, self.d_s)
        s2 = s1 + self.ho(h_agg).view(b, s_len, r_len, self.d_s)

        # --- Stage 3: U(1)-equivariant gated complex FFN + C*-invariant scalar FFN ---
        s_out = s2 + 0.5 * self.sff(s2)
        mod = 1.0 + 0.5 * torch.tanh(self.s2c_gate(s_out))  # strictly real C*-invariant multiplier
        h1u_r, h1u_i, h1_rms = _crms_norm(hr1, hi1)
        ff1_r, ff1_i = _cmul_lin(h1u_r, h1u_i, self.cff1_r, self.cff1_i)
        gate_c = torch.sigmoid(torch.sqrt(ff1_r.square() + ff1_i.square() + 1e-8))
        ff2_r, ff2_i = _cmul_lin(ff1_r * gate_c, ff1_i * gate_c, self.cff2_r, self.cff2_i)
        hr_out = (hr1 + 0.5 * ff2_r * h1_rms) * mod
        hi_out = (hi1 + 0.5 * ff2_i * h1_rms) * mod
        return hr_out, hi_out, s_out


class PhysParallelTransportNet(nn.Module):
    """Role A: Lightweight symmetric parallel-transport & U(1)-equivariant content-addressing receiver head."""

    def __init__(self, d_c: int = 16, d_s: int = 24, n_heads: int = 4):
        super().__init__()
        self.d_c = d_c
        self.d_s = d_s
        # Input embeddings
        self.s_in = nn.Sequential(nn.Linear(10, d_s), nn.SiLU())
        self.s2c_in = nn.Linear(d_s, d_c)
        self.wc_in_r, self.wc_in_i = _init_cmat(4, d_c)

        self.block = SymmetricParallelTransportBlock(d_c=d_c, d_s=d_s, n_heads=n_heads)

        # U(1)-equivariant complex readout -> continuous residual phase correction delta_phi
        self.w_phi_r, self.w_phi_i = _init_cmat(d_c, 1)
        self.phi_mix_logit = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.phi_gate_head = nn.Linear(d_s + 4, 1)

        # C*-invariant complex-to-scalar readout: Re(conj(c_ref) * u_inv)
        self.w_inv_r, self.w_inv_i = _init_cmat(d_c, 4)
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_s + 4),
            nn.Linear(d_s + 4, d_s),
            nn.SiLU(),
            nn.Linear(d_s, 2),  # [delta_logw, raw_log_nv_per_sr]
        )
        # Small near-zero init on final linear layers for clean step-0 anchoring
        nn.init.zeros_(self.out_head[-1].weight)
        nn.init.zeros_(self.out_head[-1].bias)
        nn.init.zeros_(self.phi_gate_head.weight)
        nn.init.constant_(self.phi_gate_head.bias, -2.5)  # starts near zero residual phase

    def forward(
        self,
        c_re: torch.Tensor,        # [B, 14, 17, 4]
        c_im: torch.Tensor,        # [B, 14, 17, 4]
        s_feats: torch.Tensor,     # [B, 14, 17, 10]
        base_logw: torch.Tensor,   # [B, 14, 17]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.s_in(s_feats)
        mod0 = 1.0 + 0.5 * torch.tanh(self.s2c_in(s))
        hr, hi = _cmul_lin(c_re, c_im, self.wc_in_r, self.wc_in_i)
        hr, hi = hr * mod0, hi * mod0

        hr, hi, s = self.block(hr, hi, s)

        # C*-invariant readout features: asinh(Re(conj(c_in) * (h @ W_inv)))
        ur, ui = _cmul_lin(hr, hi, self.w_inv_r, self.w_inv_i)  # [B, 14, 17, 4]
        inv_c = torch.asinh(c_re * ur + c_im * ui)              # strictly U(1)-invariant!
        feat_inv = torch.cat([s, inv_c], dim=-1)                # [B, 14, 17, d_s + 4]

        out_w_nv = self.out_head(feat_inv)
        delta_logw = out_w_nv[..., 0]
        log_w = F.log_softmax(base_logw + delta_logw, dim=-1)
        # Hypothesis-weighted symbol-level log variance calibration
        w_prob = torch.exp(log_w).detach()
        log_nv_scale = (w_prob * out_w_nv[..., 1]).sum(dim=-1, keepdim=True)  # [B, 14, 1]

        # Strictly U(1)-equivariant continuous off-grid phase readout!
        # Combine raw coherent soft-DF phasor c[..., 0] with parallel-transported complex readout u_phi
        pr, pi = _cmul_lin(hr, hi, self.w_phi_r, self.w_phi_i)  # [B, 14, 17, 1]
        pr, pi = pr.squeeze(-1), pi.squeeze(-1)
        # Normalize z_corr phasor safely
        z0r, z0i = c_re[..., 0], c_im[..., 0]
        z0_mag = torch.sqrt(z0r.square() + z0i.square() + 1e-6)
        p_mag = torch.sqrt(pr.square() + pi.square() + 1e-6)
        mix = torch.sigmoid(self.phi_mix_logit)
        tot_r = (1.0 - mix) * (z0r / z0_mag) + mix * (pr / p_mag)
        tot_i = (1.0 - mix) * (z0i / z0_mag) + mix * (pi / p_mag)
        raw_angle = torch.atan2(tot_i, tot_r + 1e-6 * torch.where(tot_r >= 0, 1.0, -1.0))  # (-pi, pi)
        phi_gate = torch.sigmoid(self.phi_gate_head(feat_inv).squeeze(-1))
        delta_phi = MAX_DELTA_PHI * torch.tanh(raw_angle / MAX_DELTA_PHI) * phi_gate
        return log_w, delta_phi, log_nv_scale


# ---------------------------------------------------------------------------
# Role B1: Parameter-Matched Standard Non-Physics Real Transformer
# ---------------------------------------------------------------------------


class MatchedRealTransformer(nn.Module):
    """Role B1: Standard real-valued 2-axis Transformer with matched parameter budget & identical inputs."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads

        self.in_proj = nn.Sequential(nn.Linear(18, d_model), nn.SiLU())
        # Temporal real multi-head self-attention (no complex parallel transport!)
        self.t_norm = nn.LayerNorm(d_model)
        self.t_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.t_out = nn.Linear(d_model, d_model, bias=False)
        # Standard learned relative position bias table for dt in [-(S-1)..+(S-1)]
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, 2 * N_SYM - 1, dtype=torch.float32))

        # Cross-hypothesis real multi-head self-attention
        self.r_norm = nn.LayerNorm(d_model)
        self.r_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.r_out = nn.Linear(d_model, d_model, bias=False)

        # Standard real FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        # Standard real output head -> [delta_logw, raw_delta_phi, raw_log_nv]
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        c_re: torch.Tensor,
        c_im: torch.Tensor,
        s_feats: torch.Tensor,
        base_logw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s_len, r_len, _ = c_re.shape
        x_in = torch.cat([s_feats, c_re, c_im], dim=-1)  # [B, 14, 17, 18]
        x = self.in_proj(x_in)

        # Temporal real self-attention along S=14
        xt = self.t_norm(x).permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_model)
        qkv = self.t_qkv(xt).view(b * r_len, s_len, 3, self.n_heads, self.dh)
        q, k, v = qkv[:, :, 0].permute(0, 2, 1, 3), qkv[:, :, 1].permute(0, 2, 1, 3), qkv[:, :, 2].permute(0, 2, 1, 3)
        idx = torch.arange(s_len, device=x.device)
        rel_idx = idx[:, None] - idx[None, :] + (s_len - 1)  # [S, S]
        bias = self.rel_pos_bias[:, rel_idx]                 # [H, S, S]
        att = F.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + bias[None, :, :, :], dim=-1)
        out_t = self.t_out((att @ v).permute(0, 2, 1, 3).reshape(b * r_len, s_len, self.d_model))
        x = x + out_t.view(b, r_len, s_len, self.d_model).permute(0, 2, 1, 3)

        # Hypothesis real self-attention along R=17
        xr = self.r_norm(x).reshape(b * s_len, r_len, self.d_model)
        qkvr = self.r_qkv(xr).view(b * s_len, r_len, 3, self.n_heads, self.dh)
        qr, kr, vr = qkvr[:, :, 0].permute(0, 2, 1, 3), qkvr[:, :, 1].permute(0, 2, 1, 3), qkvr[:, :, 2].permute(0, 2, 1, 3)
        attr = F.softmax((qr @ kr.transpose(-1, -2)) / math.sqrt(self.dh), dim=-1)
        out_r = self.r_out((attr @ vr).permute(0, 2, 1, 3).reshape(b * s_len, r_len, self.d_model))
        x = x + out_r.view(b, s_len, r_len, self.d_model)

        x = x + 0.5 * self.ffn(x)
        out = self.out_head(self.out_norm(x))
        log_w = F.log_softmax(base_logw + out[..., 0], dim=-1)
        delta_phi = MAX_DELTA_PHI * torch.tanh(out[..., 1])
        w_prob = torch.exp(log_w).detach()
        log_nv_scale = (w_prob * out[..., 2]).sum(dim=-1, keepdim=True)
        return log_w, delta_phi, log_nv_scale


# ---------------------------------------------------------------------------
# Role B2: Parameter-Matched Non-Physics Context-Pooled Real MLP
# ---------------------------------------------------------------------------


class MatchedRealMLP(nn.Module):
    """Role B2: Standard real-valued context-pooled residual MLP with matched parameter budget."""

    def __init__(self, d_hid: int = 56):
        super().__init__()
        # Input: 18 local + 18 temporal-mean + 18 hypothesis-mean = 54 real features
        self.fc1 = nn.Linear(54, d_hid)
        self.n1 = nn.LayerNorm(d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.n2 = nn.LayerNorm(d_hid)
        self.fc3 = nn.Linear(d_hid, d_hid)
        self.n3 = nn.LayerNorm(d_hid)
        self.out_head = nn.Linear(d_hid, 3)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        c_re: torch.Tensor,
        c_im: torch.Tensor,
        s_feats: torch.Tensor,
        base_logw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_loc = torch.cat([s_feats, c_re, c_im], dim=-1)  # [B, 14, 17, 18]
        x_tmean = x_loc.mean(dim=1, keepdim=True).expand_as(x_loc)
        x_rmean = x_loc.mean(dim=2, keepdim=True).expand_as(x_loc)
        x = torch.cat([x_loc, x_tmean, x_rmean], dim=-1)  # [B, 14, 17, 54]
        h = F.silu(self.n1(self.fc1(x)))
        h = h + 0.5 * F.silu(self.n2(self.fc2(h)))
        h = h + 0.5 * F.silu(self.n3(self.fc3(h)))
        out = self.out_head(h)
        log_w = F.log_softmax(base_logw + out[..., 0], dim=-1)
        delta_phi = MAX_DELTA_PHI * torch.tanh(out[..., 1])
        w_prob = torch.exp(log_w).detach()
        log_nv_scale = (w_prob * out[..., 2]).sum(dim=-1, keepdim=True)
        return log_w, delta_phi, log_nv_scale


def verify_role_a_invariants() -> dict:
    """Explicit unit audit of Role A's exact P_{s',s} = conj(P_{s,s'}) symmetry & U(1) phase equivariance."""
    torch.manual_seed(42)
    model = PhysParallelTransportNet()
    model.eval()
    # 1. Exact conjugate symmetry P_{s', s} = conj(P_{s, s'})
    cos_p, sin_p = model.block.parallel_transport_matrix(N_SYM, torch.device("cpu"))
    p_cplx = torch.complex(cos_p, sin_p)  # [H, S, S]
    sym_err = float(torch.max(torch.abs(p_cplx.transpose(-1, -2) - torch.conj(p_cplx))).item())
    unit_err = float(torch.max(torch.abs(torch.abs(p_cplx) - 1.0)).item())

    # 2. Exact U(1) invariance of log_w & log_nv_scale under arbitrary global phase rotation e^{j*psi}
    c_re = torch.randn(2, N_SYM, R_HYP, 4)
    c_im = torch.randn(2, N_SYM, R_HYP, 4)
    s_f = torch.randn(2, N_SYM, R_HYP, 10)
    blw = F.log_softmax(torch.randn(2, N_SYM, R_HYP), dim=-1)
    psi = 1.2345
    cp, sp = math.cos(psi), math.sin(psi)
    c_re_rot = c_re * cp - c_im * sp
    c_im_rot = c_re * sp + c_im * cp
    with torch.no_grad():
        lw1, _, nv1 = model(c_re, c_im, s_f, blw)
        lw2, _, nv2 = model(c_re_rot, c_im_rot, s_f, blw)
    u1_lw_err = float(torch.max(torch.abs(lw1 - lw2)).item())
    u1_nv_err = float(torch.max(torch.abs(nv1 - nv2)).item())
    assert sym_err < 1e-6 and unit_err < 1e-6 and u1_lw_err < 1e-5 and u1_nv_err < 1e-5
    return {
        "parallel_transport_conj_sym_max_err": sym_err,
        "parallel_transport_unitarity_max_err": unit_err,
        "u1_invariance_logw_max_err": u1_lw_err,
        "u1_invariance_nv_max_err": u1_nv_err,
    }


# ---------------------------------------------------------------------------
# Compact Cached Dataset Tensor Container (Memory < 400 MB)
# ---------------------------------------------------------------------------


@dataclass
class CachedSplitTensors:
    c_re: torch.Tensor       # [N, 14, 17, 4] float32
    c_im: torch.Tensor       # [N, 14, 17, 4] float32
    s_feats: torch.Tensor    # [N, 14, 17, 10] float32
    base_logw: torch.Tensor  # [N, 14, 17] float32
    y_re_d: torch.Tensor     # [N, 1400] float32
    y_im_d: torch.Tensor     # [N, 1400] float32
    hls_re_d: torch.Tensor   # [N, 1400] float32
    hls_im_d: torch.Tensor   # [N, 1400] float32
    hca_re_d: torch.Tensor   # [N, 1400] float32
    hca_im_d: torch.Tensor   # [N, 1400] float32
    nv_d: torch.Tensor       # [N, 1400] float32
    labels: torch.Tensor     # [N, 1400, 4] float32
    meta_tti: List[int]
    meta_cond: List[str]


def build_cached_tensors(
    ds: RealOFDMDataset,
    indices: np.ndarray,
    train_ref_bank: np.ndarray,
    conditions: Tuple[str, ...],
    *,
    ll_temp: float,
    p_stay: float,
    extra_stress_replicas: int = 0,
) -> CachedSplitTensors:
    """Precompute ultra-compact frozen physical frontend tensors (~55 KB/TTI, total < 300 MB RAM)."""
    c_re_l, c_im_l, s_l, blw_l = [], [], [], []
    yre_l, yim_l, hlsr_l, hlsi_l, hcar_l, hcai_l, nvd_l, lbl_l = [], [], [], [], [], [], [], []
    meta_tti, meta_cond = [], []
    rs = DATA_COORDS_101[:, 0]
    ks = DATA_COORDS_101[:, 1]

    for cond in conditions:
        n_rep = 1 if cond == "C0_raw_ota" else (1 + extra_stress_replicas)
        for rep in range(n_rep):
            salt = 20261121 + rep * 999983
            for t in indices:
                pert = apply_stress_condition(
                    ds.iq[t], cond, seed=condition_seed(int(t), cond, split_salt=salt)
                )
                b = build_shared_hypothesis_bundle(
                    pert.iq, pert.pilot_scs, train_ref_bank, ll_temp=ll_temp, use_nbi_guard=True
                )
                log_w_hmm = hmm_smooth_log_posterior(b.sym_ll, HYP_LOG_PRIOR, ll_temp=ll_temp, p_stay=p_stay)

                c_re_l.append(b.cplx_feats.real.astype(np.float32))
                c_im_l.append(b.cplx_feats.imag.astype(np.float32))
                s_l.append(b.scal_feats.astype(np.float32))
                blw_l.append(log_w_hmm.T.astype(np.float32))

                yd = b.y_used[rs, ks]
                hlsd = b.h_ls[ks]
                hcad = b.h_ca[ks]
                yre_l.append(yd.real.astype(np.float32))
                yim_l.append(yd.imag.astype(np.float32))
                hlsr_l.append(hlsd.real.astype(np.float32))
                hlsi_l.append(hlsd.imag.astype(np.float32))
                hcar_l.append(hcad.real.astype(np.float32))
                hcai_l.append(hcad.imag.astype(np.float32))
                nvd_l.append(b.nv_grid[rs, ks].astype(np.float32))
                lbl_l.append(ds.labels[t].astype(np.float32))
                meta_tti.append(int(t))
                meta_cond.append(cond)

    return CachedSplitTensors(
        c_re=torch.from_numpy(np.stack(c_re_l, axis=0)),
        c_im=torch.from_numpy(np.stack(c_im_l, axis=0)),
        s_feats=torch.from_numpy(np.stack(s_l, axis=0)),
        base_logw=torch.from_numpy(np.stack(blw_l, axis=0)),
        y_re_d=torch.from_numpy(np.stack(yre_l, axis=0)),
        y_im_d=torch.from_numpy(np.stack(yim_l, axis=0)),
        hls_re_d=torch.from_numpy(np.stack(hlsr_l, axis=0)),
        hls_im_d=torch.from_numpy(np.stack(hlsi_l, axis=0)),
        hca_re_d=torch.from_numpy(np.stack(hcar_l, axis=0)),
        hca_im_d=torch.from_numpy(np.stack(hcai_l, axis=0)),
        nv_d=torch.from_numpy(np.stack(nvd_l, axis=0)),
        labels=torch.from_numpy(np.stack(lbl_l, axis=0)),
        meta_tti=meta_tti,
        meta_cond=meta_cond,
    )


def forward_llrs_on_slice(model: nn.Module, data: CachedSplitTensors, idx: torch.Tensor) -> torch.Tensor:
    log_w, dphi, log_nv = model(
        data.c_re[idx], data.c_im[idx], data.s_feats[idx], data.base_logw[idx]
    )
    return differentiable_mixture_llrs_compact(
        data.y_re_d[idx],
        data.y_im_d[idx],
        data.hls_re_d[idx],
        data.hls_im_d[idx],
        data.hca_re_d[idx],
        data.hca_im_d[idx],
        data.nv_d[idx],
        log_w,
        dphi,
        log_nv,
    )


def evaluate_model_on_cache(
    model: nn.Module,
    data: CachedSplitTensors,
    batch_size: int = 32,
) -> Tuple[np.ndarray, float]:
    """Run inference over cached split, returning all LLRs [N, 1400, 4] float64 and mean Bit NLL."""
    model.eval()
    n = data.c_re.shape[0]
    llrs_all = []
    with torch.no_grad():
        for st in range(0, n, batch_size):
            ed = min(n, st + batch_size)
            idx = torch.arange(st, ed, dtype=torch.long)
            ll = forward_llrs_on_slice(model, data, idx)
            llrs_all.append(ll.cpu().numpy().astype(np.float64))
    llrs_np = np.concatenate(llrs_all, axis=0)
    lbl_np = data.labels.numpy()
    signed = 2.0 * lbl_np - 1.0
    z = -signed * np.clip(llrs_np, -40.0, 40.0)
    nll = float(np.mean(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))))
    return llrs_np, nll


def fit_dev_temperature(dev_llrs: np.ndarray, dev_labels: np.ndarray) -> float:
    signed = 2.0 * dev_labels - 1.0
    ll = np.clip(dev_llrs, -40.0, 40.0)

    def _obj(temp: float) -> float:
        z = -signed * (ll / temp)
        return float(np.mean(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))))

    opt = minimize_scalar(_obj, bounds=(0.25, 5.0), method="bounded")
    return float(opt.x)


def train_one_model_seed(
    role_name: str,
    model: nn.Module,
    seed: int,
    train_data: CachedSplitTensors,
    dev_data: CachedSplitTensors,
    *,
    epochs: int = 24,
    batch_size: int = 24,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
) -> Tuple[nn.Module, dict]:
    """Train one model seed strictly on Train, early-stop + calibrate temperature strictly on Dev."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    n_tr = train_data.c_re.shape[0]
    best_dev_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0
    history = []

    # Step-0 Dev check
    _, dev_nll0 = evaluate_model_on_cache(model, dev_data, batch_size=32)
    best_dev_nll = dev_nll0
    history.append({"epoch": 0, "train_nll": None, "dev_nll": round(dev_nll0, 5)})

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tr_loss_sum = 0.0
        tr_cnt = 0
        for st in range(0, n_tr, batch_size):
            idx = perm[st : st + batch_size]
            opt.zero_grad(set_to_none=True)
            llrs = forward_llrs_on_slice(model, train_data, idx)
            loss = F.binary_cross_entropy_with_logits(llrs, train_data.labels[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_loss_sum += float(loss.item()) * len(idx)
            tr_cnt += len(idx)
        sched.step()
        tr_nll = tr_loss_sum / float(tr_cnt)
        _, dev_nll = evaluate_model_on_cache(model, dev_data, batch_size=32)
        history.append({"epoch": ep, "train_nll": round(tr_nll, 5), "dev_nll": round(dev_nll, 5)})
        if dev_nll < best_dev_nll - 1e-5:
            best_dev_nll = dev_nll
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0

    # Fit post-hoc scalar temperature calibration on Dev ONLY
    dev_llrs, dev_nll_raw = evaluate_model_on_cache(model, dev_data, batch_size=32)
    t_cal = fit_dev_temperature(dev_llrs, dev_data.labels.numpy())
    dev_bm_cal = compute_bit_metrics(dev_llrs / t_cal, dev_data.labels.numpy().astype(np.int32))

    # Measure single-thread per-TTI neural head inference latency (batch_size=1, 200 TTIs)
    model.eval()
    n_bench = min(200, dev_data.c_re.shape[0])
    with torch.no_grad():
        # warmup
        for i in range(5):
            idx1 = torch.tensor([i], dtype=torch.long)
            _ = model(dev_data.c_re[idx1], dev_data.c_im[idx1], dev_data.s_feats[idx1], dev_data.base_logw[idx1])
        t_inf0 = time.perf_counter()
        for i in range(n_bench):
            idx1 = torch.tensor([i], dtype=torch.long)
            _ = model(dev_data.c_re[idx1], dev_data.c_im[idx1], dev_data.s_feats[idx1], dev_data.base_logw[idx1])
        head_ms_per_tti = 1000.0 * (time.perf_counter() - t_inf0) / float(n_bench)

    info = {
        "role": role_name,
        "seed": seed,
        "trainable_real_params": count_real_params(model),
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "train_time_s": round(train_s, 2),
        "head_infer_ms_per_tti": round(head_ms_per_tti, 3),
        "dev_nll_step0": round(dev_nll0, 5),
        "dev_nll_raw": round(dev_nll_raw, 5),
        "dev_nll_cal": round(dev_bm_cal["nll"], 5),
        "dev_T_cal": round(t_cal, 4),
        "history": history,
    }
    return model, info


def make_role_model(role_name: str, seed: int) -> nn.Module:
    """Instantiate exact parameter-matched Role A, Role B1, or Role B2."""
    torch.manual_seed(seed)
    if role_name == "RoleA_PhysParallelTransport":
        return PhysParallelTransportNet(d_c=16, d_s=24, n_heads=4)
    elif role_name == "RoleB1_MatchedTransformer":
        return MatchedRealTransformer(d_model=32, n_heads=4, d_ff=48)
    elif role_name == "RoleB2_MatchedMLP":
        return MatchedRealMLP(d_hid=64)
    else:
        raise ValueError(f"Unknown role: {role_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--group-b-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    t_start = time.perf_counter()
    inv_audit = verify_role_a_invariants()
    print("Role A exact physics invariants audit:", inv_audit)

    # Verify parameter parity upfront!
    param_counts = {
        r: count_real_params(make_role_model(r, SEEDS[0]))
        for r in ("RoleA_PhysParallelTransport", "RoleB1_MatchedTransformer", "RoleB2_MatchedMLP")
    }
    print("Trainable real parameter counts:", param_counts)

    # Load frozen Dev-calibrated frontend hyperparameters from Group B
    gb = json.loads(args.group_b_json.read_text(encoding="utf-8"))
    ll_temp = float(gb["dev_calibration"]["ll_temp"])
    p_stay = float(gb["dev_calibration"]["p_stay"])

    ds = load_real_ofdm_dataset(args.pth)
    splits = build_temporal_splits(len(ds.sinr_db), ds.sinr_db)
    train_idx = np.array(splits["train"]["tti_indices"], dtype=np.int64)
    dev_idx = np.array(splits["dev"]["tti_indices"], dtype=np.int64)
    test_idx = np.array(splits["test"]["tti_indices"], dtype=np.int64)
    test_blocks = splits["test"]["blocks"]

    train_ref_bank = build_train_reference_bank(ds.iq[train_idx], ds.sinr_db[train_idx], m_refs=16)

    t_fe = time.perf_counter()
    print("Building frozen shared frontend tensors (Train/Dev/Test)...")
    # Train: C0 x 1 + C1..C4 x 2 = 340 * 9 = 3060 grids (strictly Train split only!)
    train_cache = build_cached_tensors(
        ds, train_idx, train_ref_bank, STRESS_CONDITIONS,
        ll_temp=ll_temp, p_stay=p_stay, extra_stress_replicas=1
    )
    dev_cache = build_cached_tensors(
        ds, dev_idx, train_ref_bank, STRESS_CONDITIONS,
        ll_temp=ll_temp, p_stay=p_stay, extra_stress_replicas=0
    )
    test_cache = build_cached_tensors(
        ds, test_idx, train_ref_bank, STRESS_CONDITIONS,
        ll_temp=ll_temp, p_stay=p_stay, extra_stress_replicas=0
    )
    fe_time = time.perf_counter() - t_fe
    print(
        f"Frontend caching done in {fe_time:.1f}s | Train={len(train_cache.meta_tti)}, "
        f"Dev={len(dev_cache.meta_tti)}, Test={len(test_cache.meta_tti)} | RSS={peak_rss_mb():.1f} MB"
    )

    roles = ("RoleA_PhysParallelTransport", "RoleB1_MatchedTransformer", "RoleB2_MatchedMLP")

    # Pre-index Test positions by condition
    cond_to_pos = {
        cond: [i for i, c in enumerate(test_cache.meta_cond) if c == cond]
        for cond in STRESS_CONDITIONS
    }
    # Verify ordering matches test_idx exactly
    for cond, pos in cond_to_pos.items():
        assert [test_cache.meta_tti[i] for i in pos] == test_idx.tolist()

    # Evaluate frozen Group B3 baseline directly through the exact same torch mixture LLR path for parity check!
    with torch.no_grad():
        zeros_phi = torch.zeros_like(test_cache.base_logw)
        zeros_nv = torch.zeros((test_cache.c_re.shape[0], N_SYM, 1), dtype=torch.float32)
        b3_llrs_list = []
        for st in range(0, test_cache.c_re.shape[0], 32):
            ed = min(test_cache.c_re.shape[0], st + 32)
            idx = torch.arange(st, ed, dtype=torch.long)
            b3_llrs_list.append(
                differentiable_mixture_llrs_compact(
                    test_cache.y_re_d[idx], test_cache.y_im_d[idx],
                    test_cache.hls_re_d[idx], test_cache.hls_im_d[idx],
                    test_cache.hca_re_d[idx], test_cache.hca_im_d[idx],
                    test_cache.nv_d[idx], test_cache.base_logw[idx], zeros_phi[idx], zeros_nv[idx]
                ).numpy().astype(np.float64)
            )
        b3_test_llrs = np.concatenate(b3_llrs_list, axis=0)
    b3_t_cal = float(gb["dev_calibration"]["llr_T_cal"]["B3_mixture_likelihood"])

    # Train and evaluate all roles x 3 independent seeds
    seed_runs: Dict[str, List[dict]] = {r: [] for r in roles}
    # Store per-(role, seed, cond, tti_pos) metrics for exact 12-block paired tests
    per_seed_cond_metrics: Dict[str, Dict[int, Dict[str, List[dict]]]] = {r: {} for r in roles}
    per_tti_csv_rows = []

    for role in roles:
        for seed in SEEDS:
            model = make_role_model(role, seed)
            trained_model, train_info = train_one_model_seed(
                role, model, seed, train_cache, dev_cache,
                epochs=18, batch_size=32, lr=3.5e-3, weight_decay=1e-4, patience=5
            )
            test_llrs, _ = evaluate_model_on_cache(trained_model, test_cache, batch_size=32)
            t_cal = float(train_info["dev_T_cal"])

            # Collect per-condition Test metrics
            per_seed_cond_metrics[role][seed] = {}
            cond_summary = {}
            for cond in STRESS_CONDITIONS:
                pos = cond_to_pos[cond]
                tti_list = []
                for local_j, p_i in enumerate(pos):
                    lbl = test_cache.labels[p_i].numpy().astype(np.int32)
                    raw_bm = compute_bit_metrics(test_llrs[p_i], lbl)
                    cal_bm = compute_bit_metrics(test_llrs[p_i] / t_cal, lbl)
                    row = {
                        "tti": int(test_cache.meta_tti[p_i]),
                        "ber": raw_bm["ber"],
                        "ser": raw_bm["ser"],
                        "nll_raw": raw_bm["nll"],
                        "ece_raw": raw_bm["ece"],
                        "nll_cal": cal_bm["nll"],
                        "ece_cal": cal_bm["ece"],
                    }
                    tti_list.append(row)
                    per_tti_csv_rows.append({"role": role, "seed": seed, "condition": cond, **row})
                per_seed_cond_metrics[role][seed][cond] = tti_list
                clean_list = [x for x in tti_list if not (655 <= x["tti"] <= 665)]
                cond_summary[cond] = {
                    "ber": float(np.mean([x["ber"] for x in tti_list])),
                    "ber_excl_sync_burst": float(np.mean([x["ber"] for x in clean_list])),
                    "ser": float(np.mean([x["ser"] for x in tti_list])),
                    "nll_raw": float(np.mean([x["nll_raw"] for x in tti_list])),
                    "ece_raw": float(np.mean([x["ece_raw"] for x in tti_list])),
                    "nll_cal": float(np.mean([x["nll_cal"] for x in tti_list])),
                    "ece_cal": float(np.mean([x["ece_cal"] for x in tti_list])),
                }

            train_info["test_summary"] = cond_summary
            seed_runs[role].append(train_info)
            print(
                f"[{role} | seed={seed}] params={train_info['trainable_real_params']} | "
                f"ep={train_info['best_epoch']}/{train_info['epochs_ran']} ({train_info['train_time_s']}s) | "
                f"infer={train_info['head_infer_ms_per_tti']}ms/TTI | "
                f"Test BER: C0={cond_summary['C0_raw_ota']['ber']:.4f}, C1={cond_summary['C1_awgn_stress']['ber']:.4f}, "
                f"C2={cond_summary['C2_cfo_doppler']['ber']:.4f}, C3={cond_summary['C3_phase_jump_jitter']['ber']:.4f}, "
                f"C4={cond_summary['C4_nbi_sparse7']['ber']:.4f}"
            )

    # Aggregate across 3 seeds (mean +- std) and run 12-block exact sign-flip tests on seed-averaged per-TTI losses!
    role_aggregates = {}
    for role in roles:
        agg = {
            "trainable_real_params": param_counts[role],
            "mean_train_time_s": round(float(np.mean([r["train_time_s"] for r in seed_runs[role]])), 2),
            "mean_head_infer_ms_per_tti": round(float(np.mean([r["head_infer_ms_per_tti"] for r in seed_runs[role]])), 3),
            "conditions": {},
        }
        for cond in STRESS_CONDITIONS:
            cond_d = {}
            for m in ("ber", "ber_excl_sync_burst", "ser", "nll_raw", "ece_raw", "nll_cal", "ece_cal"):
                vals = [r["test_summary"][cond][m] for r in seed_runs[role]]
                cond_d[f"{m}_mean"] = float(np.mean(vals))
                cond_d[f"{m}_std"] = float(np.std(vals, ddof=1))
            agg["conditions"][cond] = cond_d
        role_aggregates[role] = agg

    # Compute B3 per-TTI baseline table for direct paired 12-block comparison with Role A
    b3_per_cond: Dict[str, List[dict]] = {}
    for cond in STRESS_CONDITIONS:
        pos = cond_to_pos[cond]
        lst = []
        for p_i in pos:
            lbl = test_cache.labels[p_i].numpy().astype(np.int32)
            raw_bm = compute_bit_metrics(b3_test_llrs[p_i], lbl)
            cal_bm = compute_bit_metrics(b3_test_llrs[p_i] / b3_t_cal, lbl)
            lst.append({
                "tti": int(test_cache.meta_tti[p_i]),
                "ber": raw_bm["ber"],
                "nll_cal": cal_bm["nll"],
                "ece_cal": cal_bm["ece"],
            })
        b3_per_cond[cond] = lst

    # Paired 12-block exact sign-flip tests (seed-averaged per-TTI metrics: RoleA vs RoleB1, RoleA vs RoleB2, RoleA vs B3)
    block_comparisons = {}
    for cond in STRESS_CONDITIONS:
        block_comparisons[cond] = {}
        # Seed-averaged per-TTI vectors for each role
        role_tti_avg = {}
        for role in roles:
            role_tti_avg[role] = {}
            for metric in ("ber", "nll_cal", "ece_cal"):
                arr = np.mean(
                    [[per_seed_cond_metrics[role][sd][cond][j][metric] for j in range(len(test_idx))] for sd in SEEDS],
                    axis=0,
                )
                role_tti_avg[role][metric] = arr

        b3_avg = {
            metric: np.array([b3_per_cond[cond][j][metric] for j in range(len(test_idx))])
            for metric in ("ber", "nll_cal", "ece_cal")
        }

        comp_pairs = [
            ("RoleA_vs_RoleB1_Transformer", role_tti_avg["RoleB1_MatchedTransformer"], role_tti_avg["RoleA_PhysParallelTransport"]),
            ("RoleA_vs_RoleB2_MLP", role_tti_avg["RoleB2_MatchedMLP"], role_tti_avg["RoleA_PhysParallelTransport"]),
            ("RoleA_vs_GroupB3_PhysMix", b3_avg, role_tti_avg["RoleA_PhysParallelTransport"]),
            ("RoleB1_vs_GroupB3_PhysMix", b3_avg, role_tti_avg["RoleB1_MatchedTransformer"]),
        ]
        for cname, base_d, cand_d in comp_pairs:
            for metric in ("ber", "nll_cal"):
                b_diffs = []
                for blk in test_blocks:
                    bi = [x - 502 for x in blk["tti_indices"]]
                    b_diffs.append(float(np.mean(base_d[metric][bi] - cand_d[metric][bi])))
                block_comparisons[cond][f"{cname}_{metric}"] = exact_sign_flip_test(np.array(b_diffs))

    out = {
        "total_elapsed_s": round(time.perf_counter() - t_start, 2),
        "frontend_cache_time_s": round(fe_time, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "dataset_sha256": ds.sha256,
        "seeds": list(SEEDS),
        "frontend_dev_calibration": {"ll_temp": ll_temp, "p_stay": p_stay},
        "role_a_invariants_audit": inv_audit,
        "trainable_real_params": param_counts,
        "role_aggregates": role_aggregates,
        "block_comparisons_seed_averaged": block_comparisons,
        "seed_runs": seed_runs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, args.out_dir / "group_c_results.json")

    csv_path = args.out_dir / "group_c_per_tti.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tti_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_tti_csv_rows)

    print(f"\nGroup C complete in {out['total_elapsed_s']}s | Peak RSS = {out['peak_rss_mb']} MB")
    for cond in STRESS_CONDITIONS:
        print(f"\n=== Group C Test Summary: {cond} ===")
        for role in roles:
            c = role_aggregates[role]["conditions"][cond]
            print(
                f"  {role:28s}: BER={c['ber_mean']:.5f}+-{c['ber_std']:.5f} | "
                f"NLL_cal={c['nll_cal_mean']:.4f}+-{c['nll_cal_std']:.4f} | "
                f"ECE_cal={c['ece_cal_mean']:.4f}+-{c['ece_cal_std']:.4f}"
            )


if __name__ == "__main__":
    main()
