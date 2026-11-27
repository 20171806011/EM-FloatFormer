"""Equivariant C4-quotient connector, M3 RefSufficientPool bridge, and S17 physical kernel for Stage 4."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache import analytic as s3_analytic
from research.stage3_ministral_cache.schema import ROOTS, Reference, matched_sum
from research.stage3d_qwen_signed_phase.qwen_connector import QwenCacheHost
from research.stage4_math_and_design.M3 import ref_pool as m3_pool
from research.stage4_real_and_surgical.packet_c16 import WIRE_RADIAL_EDGES, quality_edges, radial_edges

ARM_NAMES_STAGE4 = (
    "zero",
    "naive_additive",
    "equivariant_host_pure",
    "equivariant_physical_residual",
)

TASK_TO_ID: Dict[str, int] = {
    "Q1": 0,
    "Q2": 1,
    "Q3_repeat_x1": 2,
    "Q3_independent_x3": 3,
    "Q3_after_pilot": 4,
    "q1": 0,
    "q2": 1,
    "q3_repeat": 2,
    "q3_indep": 3,
    "q3_after_pilot": 4,
}

CANONICAL_TASK_NAME: Dict[str, str] = {
    "Q1": "q1",
    "q1": "q1",
    "Q2": "q2",
    "q2": "q2",
    "Q3_repeat_x1": "q3_repeat",
    "q3_repeat": "q3_repeat",
    "Q3_independent_x3": "q3_indep",
    "q3_indep": "q3_indep",
    "Q3_after_pilot": "q3_after_pilot",
    "q3_after_pilot": "q3_after_pilot",
}

DEFAULT_RESIDUAL_GAMMA = 0.15
CANONICAL_REPLAY_FLOOR = 1e-6


def lift_scores(
    canonical_probs: torch.Tensor,
    codes: torch.Tensor,
    task: str,
    pilot_turn: int | torch.Tensor = 0,
    receive_gauge: bool = False,
    tx_chart_turn: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Apply exact C4 quotient permutation R_{chi(a, b)} (M1 Theorem 2)."""
    tname = CANONICAL_TASK_NAME[task]
    a, b = (codes // 4).unbind(dim=-1)
    if tname == "q1":
        shift = torch.zeros_like(a) if receive_gauge else (b - a)
    elif tname == "q2":
        shift = (torch.zeros_like(a) + tx_chart_turn) if receive_gauge else (a - pilot_turn)
    elif tname in ("q3_repeat", "q3_after_pilot"):
        shift = a
    elif tname == "q3_indep":
        return canonical_probs * 0.0 + 0.25
    else:
        raise ValueError(f"Unsupported task={task}")
    idx = (torch.arange(4, device=canonical_probs.device).unsqueeze(0) - shift.unsqueeze(-1)) % 4
    return canonical_probs.gather(-1, idx)


def unlift_probs(
    probs: torch.Tensor,
    codes: torch.Tensor,
    task: str,
    pilot_turn: int | torch.Tensor = 0,
    receive_gauge: bool = False,
    tx_chart_turn: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Exact inverse of `lift_scores`: map physical-option probs back to C4-canonical coordinates."""
    tname = CANONICAL_TASK_NAME[task]
    a, b = (codes // 4).unbind(dim=-1)
    if tname == "q1":
        shift = torch.zeros_like(a) if receive_gauge else (b - a)
    elif tname == "q2":
        shift = (torch.zeros_like(a) + tx_chart_turn) if receive_gauge else (a - pilot_turn)
    elif tname in ("q3_repeat", "q3_after_pilot"):
        shift = a
    elif tname == "q3_indep":
        return probs
    else:
        raise ValueError(f"Unsupported task={task}")
    idx = (torch.arange(4, device=probs.device).unsqueeze(0) + shift.unsqueeze(-1)) % 4
    return probs.gather(-1, idx)


class PhysicalKernelS17(nn.Module):
    """64-node Gauss-Hermite Brownian phase transport K(theta, V) and direct S17 prior A_u."""

    def __init__(self, prior_table: np.ndarray):
        super().__init__()
        x, w = np.polynomial.hermite.hermgauss(64)
        self.register_buffer("nodes", torch.from_numpy(x).to(torch.float64))
        self.register_buffer("weights", torch.from_numpy(w / math.sqrt(math.pi)).to(torch.float64))
        tbl = np.asarray(prior_table, dtype=np.float64)
        assert tbl.shape == (16, 4) or tbl.shape == (4, 4), f"Unexpected s17 prior shape: {tbl.shape}"
        if tbl.shape == (16, 4):
            canon_tbl = tbl[:4]
            full_tbl = tbl
        else:
            canon_tbl = tbl
            full_tbl = np.concatenate([np.roll(tbl, r, axis=1) for r in range(4)], axis=0)
        self.register_buffer("prior_canon", torch.tensor(canon_tbl, dtype=torch.float32))
        self.register_buffer("prior_full", torch.tensor(full_tbl, dtype=torch.float32))

    def _wrapped_quad_probs_torch(self, theta: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Exact wrapped-Gaussian mass inside QPSK sectors [-pi/4, pi/4) + k*pi/2."""
        th = theta.to(torch.float64)
        var = variance.to(torch.float64)
        b = th.shape[0]
        cos_t = torch.cos(th)
        sin_t = torch.sin(th)
        u = cos_t + sin_t
        v = sin_t - cos_t
        det_out = torch.zeros((b, 4), dtype=torch.float64, device=th.device)
        m1 = (u > 0.0) & (v >= 0.0)
        m2 = (~m1) & (u <= 0.0) & (v > 0.0)
        m3 = (~m1) & (~m2) & (u < 0.0) & (v <= 0.0)
        m0 = (~m1) & (~m2) & (~m3)
        det_out[:, 0] = m0.to(torch.float64)
        det_out[:, 1] = m1.to(torch.float64)
        det_out[:, 2] = m2.to(torch.float64)
        det_out[:, 3] = m3.to(torch.float64)

        safe_sd = torch.sqrt(var.clamp_min(1e-12))
        k_idx = torch.arange(4, device=th.device, dtype=torch.float64)
        lo_base = -0.25 * math.pi + 0.5 * math.pi * k_idx
        hi_base = lo_base + 0.5 * math.pi
        m_vals = torch.arange(-4, 5, device=th.device, dtype=torch.float64)
        lo = lo_base[None, :, None] + 2.0 * math.pi * m_vals[None, None, :]
        hi = hi_base[None, :, None] + 2.0 * math.pi * m_vals[None, None, :]
        z_hi = (hi - th[:, None, None]) / safe_sd[:, None, None]
        z_lo = (lo - th[:, None, None]) / safe_sd[:, None, None]
        diff_mass = (torch.special.ndtr(z_hi) - torch.special.ndtr(z_lo)).sum(dim=-1)
        diff_mass = diff_mass / diff_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        is_det = (var <= 1e-12)[:, None]
        return torch.where(is_det, det_out, diff_mass).to(torch.float32)

    def kernel(self, theta: torch.Tensor, variance: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """Return `(B, 4, 4)` circulant sector pushforward K(theta, V)."""
        th = theta.to(torch.float64)
        var = variance.to(torch.float64).clamp_min(1e-12)
        snr = snr_db.to(torch.float64)
        angle = th[..., None] + torch.sqrt(2.0 * var[..., None]) * self.nodes
        scale = torch.sqrt(2.0 * (10.0 ** (snr[..., None] / 10.0)))
        c = torch.special.ndtr(scale * torch.cos(angle + math.pi / 4.0))
        s = torch.special.ndtr(scale * torch.sin(angle + math.pi / 4.0))
        base = torch.stack((c * s, (1.0 - c) * s, (1.0 - c) * (1.0 - s), c * (1.0 - s)), dim=-1)
        sector = (self.weights[:, None] * base).sum(dim=-2)  # (B, 4)
        return torch.stack([sector.roll(j, dims=-1) for j in range(4)], dim=-2).to(torch.float32)

    def direct_physical_probs(
        self,
        codes: torch.Tensor,
        meta_scalars: torch.Tensor,
        task: str,
        valid_w1: Optional[torch.Tensor] = None,
        m3_q2: Optional[torch.Tensor] = None,
        m3_q3: Optional[torch.Tensor] = None,
        residual_logits: Optional[torch.Tensor] = None,
        gamma: float = DEFAULT_RESIDUAL_GAMMA,
        discrete_d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exact physical or residual-corrected distribution across Q1/Q2/Q3."""
        tname = CANONICAL_TASK_NAME[task]
        b = codes.shape[0]
        if tname == "q3_indep":
            if residual_logits is not None:
                return residual_logits * 0.0 + 0.25
            return torch.full((b, 4), 0.25, device=codes.device, dtype=torch.float32)

        inner1 = codes[:, 0] % 4
        inner2 = codes[:, 1] % 4
        a_u1 = self.prior_canon[inner1]  # (B, 4)
        a_u2 = self.prior_canon[inner2]  # (B, 4)
        if valid_w1 is not None:
            a_u1 = torch.where(valid_w1[:, None], a_u1, torch.full_like(a_u1, 0.25))

        h, ell, _, omega, sigma, snr_db, rate_var = meta_scalars.unbind(dim=-1)
        is_disc = (
            discrete_d.to(device=codes.device, dtype=torch.float32) > 0.0
            if discrete_d is not None
            else torch.zeros(b, dtype=torch.bool, device=codes.device)
        )

        if tname == "q1":
            var12 = sigma.square() * h + rate_var * h.square()
            w_quad2 = self._wrapped_quad_probs_torch(omega * h, var12)
            a_u2_cont = sum(w_quad2[:, k : k + 1] * a_u2.roll(-k, dims=-1) for k in range(4))
            if bool(is_disc.any()):
                d_f = discrete_d.to(device=codes.device, dtype=torch.float32)
                c_h = torch.where(
                    h <= 0.0,
                    torch.ones_like(h),
                    (1.0 - (4.0 / 3.0) * d_f).clamp(0.0, 1.0).pow(h),
                )
                a_u2_disc = c_h[:, None] * a_u2 + (1.0 - c_h[:, None]) * 0.25
                a_u2_unshifted = torch.where(is_disc[:, None], a_u2_disc, a_u2_cont)
            else:
                a_u2_unshifted = a_u2_cont
            canon = torch.stack([(a_u1 * a_u2_unshifted.roll(-c, dims=-1)).sum(dim=-1) for c in range(4)], dim=-1)
            canon = canon / canon.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if residual_logits is not None and gamma != 0.0:
                corr = torch.log(canon.clamp_min(1e-12)) + gamma * residual_logits.tanh()
                canon = corr.softmax(dim=-1)
            return lift_scores(canon, codes, tname)

        if tname == "q2":
            bp = m3_q2 if m3_q2 is not None else torch.full((b, 4), 0.25, device=codes.device, dtype=torch.float32)
            canon_q2 = torch.stack([(bp * a_u1.roll(-c, dims=-1)).sum(dim=-1) for c in range(4)], dim=-1)
            canon_q2 = canon_q2 / canon_q2.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if residual_logits is not None and gamma != 0.0:
                canon_q2 = (torch.log(canon_q2.clamp_min(1e-12)) + gamma * residual_logits.tanh()).softmax(dim=-1)
            return lift_scores(canon_q2, codes, tname)

        # q3_repeat or q3_after_pilot: always start from Window-1 source a_u1
        base_source = a_u1
        if residual_logits is not None and gamma != 0.0:
            source = (torch.log(base_source.clamp_min(1e-12)) + gamma * residual_logits.tanh()).softmax(dim=-1)
        else:
            source = base_source
        if bool(is_disc.any()):
            d_f = discrete_d.to(device=codes.device, dtype=torch.float32)
            c_hl = torch.where(
                (h + ell) <= 0.0,
                torch.ones_like(h),
                (1.0 - (4.0 / 3.0) * d_f).clamp(0.0, 1.0).pow(h + ell),
            )
            source_disc = c_hl[:, None] * source + (1.0 - c_hl[:, None]) * 0.25
            source_eff = torch.where(is_disc[:, None], source_disc, source)
            theta = torch.where(is_disc, torch.zeros_like(omega), omega * (h + ell))
            var = torch.where(
                is_disc, torch.zeros_like(sigma), sigma.square() * (h + ell) + rate_var * (h + ell).square()
            )
        else:
            source_eff = source
            theta = omega * (h + ell)
            var = sigma.square() * (h + ell) + rate_var * (h + ell).square()
        k = self.kernel(theta, var, snr_db)
        canon = torch.bmm(source_eff[:, None, :], k).squeeze(1)
        return lift_scores(canon, codes, tname)


def decode_c16_tail(words: torch.Tensor, bits: int = 4, valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode `(B, 2)` 4-bit or 16-bit packets -> `(codes_4bit, tail_6, valid_mask)`."""
    w = words.to(torch.int64)
    v = torch.ones_like(w, dtype=torch.bool) if valid is None else valid.to(torch.bool)
    w = torch.where(v, w, 0)
    if bits == 4:
        return w, torch.zeros((*w.shape, 6), dtype=torch.float32, device=w.device), v
    if bits != 16:
        raise ValueError(f"Expected bits in (4, 16), got {bits}")
    angle = (((w >> 6) & 63).to(torch.float32) + 0.5) * (math.pi / 128.0) - math.pi / 4.0
    rad = (w >> 2) & 15
    qual = w & 3
    log_edges = torch.as_tensor(np.log1p(radial_edges(4)[:-1]), dtype=torch.float32, device=w.device)
    log_edges = torch.cat((log_edges, log_edges.new_tensor([math.log(25.0)])))
    log_rad = 0.5 * (log_edges[rad] + log_edges[rad + 1])
    qual_low = torch.as_tensor(np.log1p(quality_edges(2)[:-1]), dtype=torch.float32, device=w.device)[qual]
    has_radial = rad > 0
    is_sat = ((rad == 15) | (qual == 3)).to(torch.float32)
    tail = torch.stack(
        (angle.cos(), angle.sin(), log_rad, qual_low, is_sat, has_radial.to(torch.float32)),
        dim=-1,
    )
    eff_valid = v & has_radial
    return (w >> 12), torch.where(eff_valid[..., None], tail, 0.0), eff_valid


def extract_canonical_ref_summary(
    refs: Sequence[Reference],
    batch_size: int,
    device: torch.device,
    omega: float = 0.0,
    sigma_phi: float = 0.0,
    drift_type: str = "cfo_continuous",
    discrete_d: float = 0.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Byte-exact M3-style reference summary immutable to duplicates, distractors, padding, and order.

    Returns `(ref_summary (B, 29), canonical_dedup_feats_list)`.
    """
    seen_ids = set()
    accepted: List[Reference] = []
    for r in refs:
        if r is None:
            continue
        if r.acquisition_id in seen_ids:
            continue
        seen_ids.add(r.acquisition_id)
        if not r.shares_channel:
            continue
        accepted.append(r)

    accepted.sort(key=lambda item: int(item.acquisition_id))
    summary = torch.zeros((batch_size, 29), device=device, dtype=torch.float32)
    if not accepted:
        summary[:, 0:4] = 0.25  # Uniform Q2 pilot posterior
        summary[:, 4:8] = 0.25  # Uniform Q3 auxiliary slot
        return summary, []

    roots_np = np.asarray(ROOTS, dtype=np.complex128)
    t_p = float(accepted[0].t_p)
    feat_stack: List[torch.Tensor] = []

    for r in accepted:
        samp = np.asarray(r.samples, dtype=np.complex128)
        t = matched_sum(samp)
        snr_r = 10.0 ** (float(r.snr_db) / 10.0)
        phase_det = float(getattr(r, "t_p", 0.0))
        t_derot = t * np.exp(-1j * float(omega) * phase_det)
        t_feat = np.stack(
            [
                t_derot.real,
                t_derot.imag,
                np.abs(t_derot),
                np.full(batch_size, phase_det),
                np.full(batch_size, snr_r / 10.0),
            ],
            axis=-1,
        )
        feat_stack.append(torch.from_numpy(t_feat).to(device=device, dtype=torch.float32))

    # Exact canonical posterior bp(g) over pilot symbol G in C4 matching analytic.readout_from_table
    if drift_type == "discrete":
        log_raw = np.zeros((batch_size, 4), dtype=np.float64)
        for r in accepted:
            t_r = matched_sum(np.asarray(r.samples, dtype=np.complex128))
            snr_r = 10.0 ** (float(r.snr_db) / 10.0)
            log_raw += 2.0 * snr_r * (t_r[:, None] * roots_np[None, :].conj()).real
        log_raw -= np.max(log_raw, axis=1, keepdims=True)
        bp_raw = np.exp(log_raw)
        bp_raw /= np.sort(bp_raw, axis=1).sum(axis=1, keepdims=True)
        p_0tp = s3_analytic.discrete_walk_dist(float(discrete_d), int(round(t_p)))
        p_g = np.stack([sum(p_0tp[w] * bp_raw[:, (g + w) % 4] for w in range(4)) for g in range(4)], axis=1)
        p_g /= np.sort(p_g, axis=1).sum(axis=1, keepdims=True)
    else:
        var_p = 0.0 if drift_type == "coherent" else float(t_p) * (float(sigma_phi) ** 2)
        mean_p = 0.0 if drift_type == "coherent" else float(omega) * float(t_p)
        if var_p <= 0.0:
            dir_p = roots_np * np.exp(1j * mean_p)
            log_g = np.zeros((batch_size, 4), dtype=np.float64)
            for r in accepted:
                t_r = matched_sum(np.asarray(r.samples, dtype=np.complex128))
                snr_r = 10.0 ** (float(r.snr_db) / 10.0)
                log_g += 2.0 * snr_r * (t_r[:, None] * dir_p[None, :].conj()).real
            log_g -= np.max(log_g, axis=1, keepdims=True)
            p_g = np.exp(log_g)
            p_g /= np.sort(p_g, axis=1).sum(axis=1, keepdims=True)
        else:
            phi_g, w_p = s3_analytic.wrapped_gaussian_weights(mean_p, var_p, grid=s3_analytic.QUAD_NODES_DEFAULT)
            phases = roots_np[None, :] * np.exp(1j * phi_g)[:, None]
            lw = np.zeros((batch_size, len(phi_g), 4), dtype=np.float64)
            for r in accepted:
                t_r = matched_sum(np.asarray(r.samples, dtype=np.complex128))
                snr_r = 10.0 ** (float(r.snr_db) / 10.0)
                lw += 2.0 * snr_r * np.real(t_r[:, None, None] * np.conj(phases)[None, :, :])
            lw -= np.max(lw, axis=(1, 2), keepdims=True)
            lp = np.exp(lw)
            p_g = np.einsum("g,mgk->mk", w_p, lp)
            p_g /= np.sort(p_g, axis=1).sum(axis=1, keepdims=True)

    p_g_t = torch.from_numpy(p_g).to(device=device, dtype=torch.float32)
    mean_feat = torch.stack(feat_stack, dim=0).mean(dim=0)  # (B, 5)
    max_abs = torch.stack([f[:, 2] for f in feat_stack], dim=0).max(dim=0).values.unsqueeze(-1)
    summary[:, 0:4] = p_g_t
    summary[:, 4:8] = p_g_t
    summary[:, 8] = 1.0
    summary[:, 9] = min(1.0, len(accepted) / 4.0)
    summary[:, 10:15] = mean_feat
    summary[:, 15:16] = max_abs
    summary[:, 16] = float(len(accepted))
    return summary, feat_stack


class Stage4MatchedConnector(nn.Module):
    """Exact 488,400-parameter matched connector (M1 §6.3) across all 4 Stage 4 arms.

    Parameterized for `hidden_size=2560` (Qwen3.5-4B):
      win_emb (16x16=256) + task_emb (5x16=80) + feat_proj (160->128: 20,608)
      + token_head (128->1024: 132,096) + out_proj (128->2560: 330,240)
      + out_norm (LayerNorm 2560: 5,120) = 488,400 trainable parameters.
    """

    def __init__(
        self,
        arm: str,
        prior_table: np.ndarray,
        hidden_size: int = 2560,
        residual_gamma: float = DEFAULT_RESIDUAL_GAMMA,
    ):
        super().__init__()
        if arm not in ARM_NAMES_STAGE4:
            raise ValueError(f"Unknown arm={arm!r}; expected one of {ARM_NAMES_STAGE4}")
        self.arm = arm
        self.hidden_size = int(hidden_size)
        self.quotient = arm in ("equivariant_host_pure", "equivariant_physical_residual")
        self.coupled = arm in ("equivariant_host_pure", "equivariant_physical_residual")
        self.zero_slots = arm == "zero"
        self.residual = arm == "equivariant_physical_residual"
        self.residual_gamma = float(residual_gamma)

        self.kernel_module = PhysicalKernelS17(prior_table)
        self.register_buffer("prior_canon", self.kernel_module.prior_canon.clone())
        self.register_buffer("prior_full", self.kernel_module.prior_full.clone())
        powers = ROOTS[None, :] ** np.arange(1, 4)[:, None]
        self.register_buffer("fourier_basis", torch.from_numpy(powers).to(torch.complex64))

        # Exact M1 §6.3 trainable weights (488,400 params when hidden_size=2560)
        self.win_emb = nn.Embedding(16, 16)
        self.task_emb = nn.Embedding(5, 16)
        self.feat_proj = nn.Linear(160, 128)
        self.token_head = nn.Linear(128, 1024)
        self.out_proj = nn.Linear(128, self.hidden_size)
        self.out_norm = nn.LayerNorm(self.hidden_size)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def connector_config(self) -> dict:
        return {
            "arm": self.arm,
            "hidden_size": self.hidden_size,
            "quotient": self.quotient,
            "coupled": self.coupled,
            "zero_slots": self.zero_slots,
            "residual": self.residual,
            "residual_gamma": self.residual_gamma,
            "trainable_params": self.param_count(),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_fused_features(
        self,
        words: torch.Tensor,
        bits: int,
        task_id: torch.Tensor,
        phase_slots: torch.Tensor,
        meta_scalars: torch.Tensor,
        ref_summary: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the `(B, 160)` fused evidence tensor without exposing raw IQ/labels."""
        codes, tail, eff_valid = decode_c16_tail(words, bits=bits, valid=valid)
        inner = codes % 4
        ids = inner if self.quotient else codes
        w1_e, w2_e = self.win_emb(ids).unbind(dim=1)  # each (B, 16)
        paired = torch.cat((w1_e, w2_e, w1_e - w2_e, w1_e * w2_e), dim=-1)  # (B, 64)
        t_e = self.task_emb(task_id)  # (B, 16)

        prior_w = self.prior_canon[inner] if self.quotient else self.prior_full[codes]  # (B, 2, 4)
        prior_w = torch.where(eff_valid[..., None], prior_w, torch.full_like(prior_w, 0.25))

        slots = torch.zeros_like(phase_slots) if self.zero_slots else phase_slots
        # Coupled physical Fourier phasor propagation (M1 §6.1)
        h, ell, _, _, sigma, _, rate_var = meta_scalars.unbind(dim=-1)
        dt = torch.stack((h + ell, h), dim=-1)
        var = sigma[:, None].square() * dt + rate_var[:, None] * dt.square()
        n = torch.arange(1, 4, device=meta_scalars.device, dtype=torch.float32)
        base = torch.stack(
            (torch.complex(slots[:, 2], slots[:, 3]), torch.complex(slots[:, 0], -slots[:, 1])),
            dim=-1,
        )
        phasors = torch.exp(-0.5 * var[..., None] * n.square()) * base[..., None].pow(n)  # (B, 2, 3)
        source_moments = prior_w.to(phasors.dtype) @ self.fourier_basis.T  # (B, 2, 3)
        product = torch.view_as_real(phasors * source_moments).flatten(-2)  # (B, 2, 6) -> 12
        polar = torch.complex(tail[..., 0], tail[..., 1]) * tail[..., 2].expm1()  # (B, 2)
        polar_prod = torch.view_as_real(phasors[..., 0] * polar)  # (B, 2, 2) -> 4
        if not self.coupled:
            product = torch.zeros_like(product)
            polar_prod = torch.zeros_like(polar_prod)

        ev = torch.cat(
            (
                prior_w.flatten(1),       # 8
                tail.flatten(1),          # 12
                eff_valid.to(torch.float32),  # 2
                product.flatten(1),       # 12
                polar_prod.flatten(1),    # 4
            ),
            dim=-1,
        )  # 38 dims

        scalars_pad = meta_scalars[:, :7]  # 7 dims
        ref_vec = (
            torch.zeros((codes.shape[0], 29), device=codes.device, dtype=torch.float32)
            if ref_summary is None
            else ref_summary[:, :29].clone()
        )
        # Pure C4-Fourier bilinear Q1 & Q2 harmonic products (zero PhysicalKernel integral; zero extra params)
        if self.coupled:
            q1_harm = torch.view_as_real(
                source_moments[:, 0].conj() * (phasors[:, 1] * source_moments[:, 1])
            ).flatten(1)
            bp_m = ref_vec[:, 0:4].to(phasors.dtype) @ self.fourier_basis.T
            q2_harm = torch.view_as_real(source_moments[:, 0] * bp_m.conj()).flatten(1)
            ref_vec[:, 17:23] = q1_harm
            ref_vec[:, 23:29] = q2_harm
        else:
            ref_vec[:, 17:29] = 0.0
        fused = torch.cat((paired, t_e, ev, slots, scalars_pad, ref_vec), dim=-1)  # 64+16+38+6+7+29 = 160
        return fused, codes, eff_valid

    def forward_tokens(
        self,
        words: torch.Tensor,
        bits: int,
        task_id: torch.Tensor,
        phase_slots: torch.Tensor,
        meta_scalars: torch.Tensor,
        ref_summary: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(soft_tokens (B, 8, D), codes (B, 2), eff_valid (B, 2), latent_h (B, 128))`."""
        fused, codes, eff_valid = self.build_fused_features(
            words, bits, task_id, phase_slots, meta_scalars, ref_summary=ref_summary, valid=valid
        )
        h = F.gelu(self.feat_proj(fused))  # (B, 128)
        toks_128 = self.token_head(h).view(-1, 8, 128)  # (B, 8, 128)
        soft_tokens = self.out_norm(self.out_proj(toks_128)) * 0.05  # (B, 8, D)
        return soft_tokens, codes, eff_valid, h

    def build_host_prefix_embeds(
        self,
        host: QwenCacheHost,
        user_prompt: str,
        soft_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject soft tokens at `<W1>/<W2>/<PHASE>/<REF>` and return `(prefix_embeds, m_acc_mask)`.

        `m_acc_mask` is a `(1, L_prefix)` float/bool mask marking 1 on EM prefix tokens and 0 on prose,
        for optional W3 Tier-1 Token-Position Gate without ever masking query tokens.
        """
        b = soft_tokens.shape[0]
        parts = host.format_chat_parts(user_prompt) if host.tokenizer is not None else ["[TEST]", "<W1>", "<W2>", "<PHASE>", "?"]
        embed_layer = host.lm.get_input_embeddings()
        tok_b = soft_tokens.to(dtype=host.dtype)

        w1_t = tok_b[:, 0:2, :]
        w2_t = tok_b[:, 2:4, :]
        ph_t = tok_b[:, 4:6, :]
        ref_t = tok_b[:, 6:8, :]

        chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        for seg in parts:
            if seg == "<W1>":
                chunks.append(w1_t)
                mask_chunks.append(torch.ones((b, 2), device=host.device, dtype=torch.bool))
            elif seg == "<W2>":
                chunks.append(w2_t)
                mask_chunks.append(torch.ones((b, 2), device=host.device, dtype=torch.bool))
            elif seg == "<PHASE>":
                chunks.append(ph_t)
                mask_chunks.append(torch.ones((b, 2), device=host.device, dtype=torch.bool))
            elif seg == "<REF>":
                chunks.append(ref_t)
                mask_chunks.append(torch.ones((b, 2), device=host.device, dtype=torch.bool))
            else:
                if host.tokenizer is None:
                    ids = [1, 2, 3]
                else:
                    ids = host.tokenizer.encode(seg, add_special_tokens=False)
                if ids:
                    t_ids = torch.tensor(ids, device=host.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
                    chunks.append(embed_layer(t_ids))
                    mask_chunks.append(torch.zeros((b, len(ids)), device=host.device, dtype=torch.bool))
        return torch.cat(chunks, dim=1), torch.cat(mask_chunks, dim=1)

    def score_batch(
        self,
        host: QwenCacheHost,
        user_prompt: str,
        choices: Tuple[str, ...],
        task: str,
        words: torch.Tensor,
        bits: int,
        phase_slots: torch.Tensor,
        meta_scalars: torch.Tensor,
        ref_summary: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        scorer_mode: str = "kv_cached",
        temperature: float = 1.0,
        surgical_adapter=None,
        discrete_d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """End-to-end differentiable scoring returning `(log_probs, raw_host_logits, probs, latent_h)`."""
        b = words.shape[0]
        task_id = torch.full((b,), TASK_TO_ID[task], device=words.device, dtype=torch.long)
        soft_tokens, codes, eff_valid, latent_h = self.forward_tokens(
            words, bits, task_id, phase_slots, meta_scalars, ref_summary=ref_summary, valid=valid
        )
        prefix_embeds, m_acc_mask = self.build_host_prefix_embeds(host, user_prompt, soft_tokens)

        ctx = surgical_adapter.activate(g_em=1, m_acc_mask=m_acc_mask) if surgical_adapter is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            if host.tokenizer is None:
                # Mock tiny LM test path without tokenizer
                seq_len = prefix_embeds.shape[1]
                attn = torch.ones((b, seq_len), device=host.device, dtype=torch.long)
                out = host.lm(inputs_embeds=prefix_embeds, attention_mask=attn, use_cache=False)
                raw_logits = out.logits[:, -1, :4].float()
            elif scorer_mode == "kv_cached":
                _, raw_logits = host.score_candidates_kv(prefix_embeds, choices, temperature=1.0)
            elif scorer_mode == "batched_uncached":
                _, raw_logits = host.score_candidates_batched_uncached(prefix_embeds, choices, temperature=1.0)
            else:
                _, raw_logits = host.score_candidates_naive_uncached(prefix_embeds, choices, temperature=1.0)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

        centered_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
        if not self.quotient:
            scaled = centered_logits / float(temperature)
            log_probs = F.log_softmax(scaled, dim=-1)
            return log_probs, centered_logits, log_probs.exp(), latent_h

        if self.residual:
            m3_q2 = ref_summary[:, 0:4] if ref_summary is not None else None
            m3_q3 = ref_summary[:, 4:8] if ref_summary is not None else None
            probs = self.kernel_module.direct_physical_probs(
                codes,
                meta_scalars,
                task,
                valid_w1=eff_valid[:, 0],
                m3_q2=m3_q2,
                m3_q3=m3_q3,
                residual_logits=centered_logits / max(1e-6, float(temperature)),
                gamma=self.residual_gamma,
                discrete_d=discrete_d,
            )
            log_probs = torch.log(probs.clamp_min(1e-12))
            return log_probs, centered_logits, probs, latent_h

        canon_probs = F.softmax(centered_logits / float(temperature), dim=-1)
        probs = lift_scores(canon_probs, codes, task)
        log_probs = torch.log(probs.clamp_min(1e-12))
        return log_probs, centered_logits, probs, latent_h
