"""EM-FloatFormer: Electromagnetic-Induction Native Transformer Architecture for Stage P4-A.

Features Implemented:
- F8: Exact C* = R_{>0} x U(1) Gauge Normalization & Complex RMSNorm
- F9: Hermitian Green's Propagator Attention, Complex Linear (bias=False), Complex RoPE & Radial Activation
- F10: Bidirectional Pilot-Data Cross-Attention & EMFloatFormer Neural Receiver
- F11: Joint Physical Consistency Objective (Calibrated LLR BCE, Normalized NMSE, Forward PA Residual)
- F12: Standalone & Integrated Numerical Equivariance Verification Functions
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from p4.simulator import (
    DEFAULT_N,
    SIGMA_16QAM,
    THRESH_16QAM,
    FrameData,
    generate_frame,
)
from p4.classical_baselines import (
    bussgang_lmmse,
    calibrated_16qam_llrs,
    delay_est_dft,
    iterative_dd_pa_cancellation,
)


# =============================================================================
# Feature F8: Exact C* Gauge Normalization
# =============================================================================

def compute_gauge_anchor(yp: torch.Tensor, eps: float = 1e-38) -> torch.Tensor:
    """Compute global pilot RMS gauge anchor s0 = sqrt(mean(|Yp|^2)).
    
    Uses amax rescaling in float64 to avoid float32 overflow/underflow across
    extreme dynamic ranges (e.g. 1e-35 to 1e20) while guaranteeing s0 > 0.
    """
    yp64 = yp.to(torch.complex128)
    m = yp64.abs().amax(dim=-1, keepdim=True)
    m_safe = torch.where(m > 0.0, m, torch.ones_like(m))
    normed = yp64 / m_safe
    power = torch.mean(normed.real ** 2 + normed.imag ** 2, dim=-1, keepdim=True)
    # The zero row takes the constant branch below. Its unused square root
    # also needs a finite derivative for backpropagation through a mixed batch.
    rms_scaled = torch.sqrt(torch.where(m > 0.0, power, torch.ones_like(power)))
    s0 = torch.where(m > 0.0, torch.clamp(m * rms_scaled, min=eps), torch.full_like(m, eps))
    return s0.to(yp.real.dtype)


def apply_gauge_normalization(
    yp: torch.Tensor,
    yd: torch.Tensor,
    eps: float = 1e-38,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map received pilot and data observations to scale-free unit-RMS representations."""
    s0 = compute_gauge_anchor(yp, eps=eps)
    s0_64 = s0.to(torch.float64)
    yp_bar = (yp.to(torch.complex128) / s0_64).to(yp.dtype)
    yd_bar = (yd.to(torch.complex128) / s0_64).to(yd.dtype)
    return yp_bar, yd_bar, s0


def rescale_channel_estimate(h_bar: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    """Rescale normalized channel estimate back to true physical units via s0 * H_bar."""
    return (h_bar.to(torch.complex128) * s0.to(torch.float64)).to(h_bar.dtype)


class ComplexRMSNorm(nn.Module):
    """Complex RMS Normalization layer preserving complex phase angles.
    
    Uses amax rescaling in float64/complex128 with factored z_scaled * (gamma / rms_scaled)
    so subnormal/small (1e-35) and huge (1e20) inputs remain normalized and free of
    NaN/Inf/collapse in both forward and backward passes.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., d_model) complex64 or complex128
        z64 = z.to(torch.complex128)
        m = z64.abs().amax(dim=-1, keepdim=True)
        active = m > 1e-38
        m_safe = torch.where(active, m, torch.ones_like(m))
        z_scaled = z64 / m_safe
        rms_scaled = torch.sqrt(
            torch.mean(z_scaled.real ** 2 + z_scaled.imag ** 2, dim=-1, keepdim=True) + (self.eps ** 2)
        )
        rms_safe = torch.where(active, rms_scaled, torch.ones_like(rms_scaled))
        gamma64 = self.gamma.to(torch.float64)
        out64 = z_scaled * (gamma64 / rms_safe)
        return out64.to(z.dtype)


def audit_single_scale(scale: float) -> Tuple[float, int]:
    """Audit model equivariance under a single scaling factor.
    
    Compares outputs on (yp, yd, nv) vs (scale*yp, scale*yd, scale^2*nv).
    
    Returns:
        (max_err, flips): Maximum absolute LLR error and number of hard bit decision flips.
    """
    torch.manual_seed(42)
    model = EMFloatFormer()
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out1 = model(yp, yd, nv)
        out2 = model(yp * scale, yd * scale, nv * (scale ** 2))
        max_err = (out1["llrs"] - out2["llrs"]).abs().max().item()
        flips = ((out1["llrs"] > 0) != (out2["llrs"] > 0)).sum().item()

    return max_err, flips


def audit_single_phase(phase: float) -> int:
    """Audit model invariance under a single phase rotation exp(j * phase).
    
    Returns:
        flips: Number of hard bit decision flips.
    """
    torch.manual_seed(42)
    model = EMFloatFormer()
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)
    c = torch.exp(1j * torch.tensor(phase))

    with torch.no_grad():
        out1 = model(yp, yd, nv)
        out2 = model(yp * c, yd * c, nv)
        flips = ((out1["llrs"] > 0) != (out2["llrs"] > 0)).sum().item()

    return flips


# =============================================================================
# Feature F9: Hermitian Green's Propagator Attention
# =============================================================================

class ComplexLinear(nn.Module):
    """Complex linear map without bias satisfying W(c Z) = c W(Z) for all c in C*.
    
    When bias=False, self.bias is strictly None.
    Re(W Z) = X W_r^T - Y W_i^T
    Im(W Z) = X W_i^T + Y W_r^T
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))

        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
            self.bias_i = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(2.0 * max(1, self.in_features))
        nn.init.normal_(self.weight_r, 0.0, std)
        nn.init.normal_(self.weight_i, 0.0, std)
        if self.bias_r is not None:
            nn.init.zeros_(self.bias_r)
            nn.init.zeros_(self.bias_i)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xr = z.real
        xi = z.imag
        real_dtype = xr.dtype
        w_r = self.weight_r.to(real_dtype)
        w_i = self.weight_i.to(real_dtype)
        out_r = F.linear(xr, w_r) - F.linear(xi, w_i)
        out_i = F.linear(xr, w_i) + F.linear(xi, w_r)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r.to(real_dtype)
            out_i = out_i + self.bias_i.to(real_dtype)
        return torch.complex(out_r, out_i)


class ComplexRoPE(nn.Module):
    """Rotary position embedding (RoPE) applied via complex phasor multiplication.
    
    Modulates query and key tokens by relative subcarrier distance:
    Q_tilde = Q * exp(j * m * theta)
    """

    def __init__(self, dim: int, max_len: int = 128, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        theta = base ** (-torch.arange(0, dim, dtype=torch.float32) / max(1, dim))
        self.register_buffer("theta", theta)

    def get_phasor(self, dist: int) -> torch.Tensor:
        """Returns unitary phasor of shape (dim,) with |phasor| == 1."""
        angle = dist * self.theta
        return torch.complex(torch.cos(angle), torch.sin(angle))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (..., seq_len, dim)
        seq_len = x.shape[-2]
        real_dtype = x.real.dtype
        pos = torch.arange(offset, offset + seq_len, device=x.device, dtype=real_dtype)
        angles = pos.unsqueeze(-1) * self.theta.to(real_dtype).unsqueeze(0)  # (seq_len, dim)
        phasors = torch.complex(torch.cos(angles), torch.sin(angles))
        return x * phasors


def hermitian_attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute real projection of complex Hermitian inner product: Re(<Q, K>_C) / sqrt(dim).
    
    Guarantees strict global phase cancellation: Re(<c Q, c K>_C) == Re(<Q, K>_C) for |c| == 1.
    
    Args:
        q: Query tensor (..., N_q, dim) complex64.
        k: Key tensor (..., N_k, dim) complex64.
        
    Returns:
        scores: Real attention scores (..., N_q, N_k) float32.
    """
    dim = q.shape[-1]
    inner = torch.matmul(q, k.conj().transpose(-2, -1))
    return inner.real / math.sqrt(dim)


class RadialActivation(nn.Module):
    """U(1)-equivariant radial activation function f(Z) = g(|Z|) * (Z / |Z|).
    
    Strictly satisfies f(exp(j * phi) * Z) = exp(j * phi) * f(Z).
    Handles zero input with exact zero (no NaN) and large inputs without overflow.
    """

    def __init__(self, kind: str = "silu", eps: float = 1e-12):
        super().__init__()
        self.kind = kind
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z64 = z.to(torch.complex128)
        m = torch.maximum(z64.real.abs(), z64.imag.abs())
        m_safe = torch.where(m > 1e150, m, torch.ones_like(m))
        mag = m_safe * torch.sqrt(
            (z64.real / m_safe) ** 2 + (z64.imag / m_safe) ** 2 + (self.eps / m_safe) ** 2
        )
        if self.kind == "silu":
            g_mag = F.silu(mag)
        elif self.kind == "gelu":
            g_mag = F.gelu(mag)
        elif self.kind == "relu":
            g_mag = F.relu(mag)
        elif self.kind == "tanh":
            g_mag = torch.tanh(mag)
        else:
            g_mag = F.silu(mag)

        scale = g_mag / mag
        return (z64 * scale).to(z.dtype)


class GreenAttention(nn.Module):
    """Hermitian Green's Propagator Attention.
    
    Multi-head complex attention with zero-bias ComplexLinear projections,
    ComplexRoPE relative position modulation on queries, keys, and propagator values,
    and Hermitian inner-product affinities.
    """

    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = ComplexLinear(d_model, d_model, bias=False)
        self.w_k = ComplexLinear(d_model, d_model, bias=False)
        self.w_v = ComplexLinear(d_model, d_model, bias=False)
        self.w_out = ComplexLinear(d_model, d_model, bias=False)
        self.rope = ComplexRoPE(dim=self.head_dim, base=32.0)

        # Initialize w_k close to w_q for an initially local Hermitian Green's kernel
        with torch.no_grad():
            self.w_k.weight_r.copy_(self.w_q.weight_r + 0.05 * torch.randn_like(self.w_q.weight_r))
            self.w_k.weight_i.copy_(self.w_q.weight_i + 0.05 * torch.randn_like(self.w_q.weight_i))

    def compute_weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Computes softmax attention weights from query and key representations."""
        if q.dim() == 3:
            B, N_q, _ = q.shape
            q = q.view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        if k.dim() == 3:
            B, N_k, _ = k.shape
            k = k.view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = hermitian_attention_scores(q, k)
        weights = F.softmax(scores, dim=-1)
        return weights

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        value_context: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        k_in = z if context is None else context
        v_in = k_in if value_context is None else value_context

        B, N_q, _ = z.shape
        N_k = k_in.shape[1]

        q = self.w_q(z).view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k_in).view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v_in).view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)
        v_rot = self.rope(v)

        weights = self.compute_weights(q, k)  # (B, n_heads, N_q, N_k)

        ctx_rot = torch.matmul(weights.to(v_rot.dtype), v_rot)  # (B, n_heads, N_q, head_dim)
        # Un-rotate query positions to realize relative Green's propagator e^{-j(m-n)theta}
        real_dtype = z.real.dtype
        pos_q = torch.arange(0, N_q, device=z.device, dtype=real_dtype)
        ang_q = pos_q.unsqueeze(-1) * self.rope.theta.to(real_dtype).unsqueeze(0)
        unrot = torch.complex(torch.cos(ang_q), -torch.sin(ang_q))
        ctx = ctx_rot * unrot
        ctx = ctx.transpose(1, 2).contiguous().view(B, N_q, self.d_model)

        out = self.w_out(ctx)
        if return_weights:
            return out, weights
        return out


class RealAttention(nn.Module):
    """Real-valued dot-product attention ablation (non-Hermitian, no ComplexRoPE)."""

    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        real_dim = 2 * d_model
        self.w_q = nn.Linear(real_dim, real_dim, bias=True)
        self.w_k = nn.Linear(real_dim, real_dim, bias=True)
        self.w_v = nn.Linear(real_dim, real_dim, bias=True)
        self.w_out = nn.Linear(real_dim, real_dim, bias=True)

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        value_context: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        k_in = z if context is None else context
        v_in = k_in if value_context is None else value_context
        B, N_q, _ = z.shape
        N_k = k_in.shape[1]

        zr = torch.cat([z.real, z.imag], dim=-1)
        kr = torch.cat([k_in.real, k_in.imag], dim=-1)
        vr = torch.cat([v_in.real, v_in.imag], dim=-1)
        rdtype = zr.dtype

        rh = 2 * self.head_dim
        q = F.linear(zr, self.w_q.weight.to(rdtype), self.w_q.bias.to(rdtype)).view(B, N_q, self.n_heads, rh).transpose(1, 2)
        k = F.linear(kr, self.w_k.weight.to(rdtype), self.w_k.bias.to(rdtype)).view(B, N_k, self.n_heads, rh).transpose(1, 2)
        v = F.linear(vr, self.w_v.weight.to(rdtype), self.w_v.bias.to(rdtype)).view(B, N_k, self.n_heads, rh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(rh)
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, N_q, 2 * self.d_model)
        out_r = F.linear(ctx, self.w_out.weight.to(rdtype), self.w_out.bias.to(rdtype))
        out = torch.complex(out_r[..., : self.d_model], out_r[..., self.d_model :])
        if return_weights:
            return out, weights
        return out


# =============================================================================
# Feature F10: Bidirectional Pilot-Data Cross-Attention
# =============================================================================

class JointEmbedding(nn.Module):
    """Dual-stream embedding mapping scalar subcarrier observations to complex token vectors."""

    def __init__(self, n: int = 32, d_model: int = 64):
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.proj_p = ComplexLinear(1, d_model, bias=False)
        self.proj_d = ComplexLinear(1, d_model, bias=False)

    def embed_pilot(self, yp: torch.Tensor) -> torch.Tensor:
        if yp.dim() == 2:
            yp = yp.unsqueeze(-1)
        return self.proj_p(yp)

    def embed_data(self, yd: torch.Tensor) -> torch.Tensor:
        if yd.dim() == 2:
            yd = yd.unsqueeze(-1)
        return self.proj_d(yd)

    def forward(self, yp: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        p_tokens = self.embed_pilot(yp)  # (B, N_p, d_model)
        d_tokens = self.embed_data(yd)   # (B, N_d, d_model)
        return torch.cat([p_tokens, d_tokens], dim=1)  # (B, N_p + N_d, d_model)


class BidirectionalBlock(nn.Module):
    """Dual-stream Transformer block with mutual cross-attention between pilots and data."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        use_gauge_norm: bool = True,
        use_hermitian_attention: bool = True,
        use_joint_pilot_data: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_gauge_norm = use_gauge_norm
        self.use_hermitian_attention = use_hermitian_attention
        self.use_joint_pilot_data = use_joint_pilot_data

        NormCls = ComplexRMSNorm if use_gauge_norm else (lambda _: nn.Identity())
        AttnCls = GreenAttention if use_hermitian_attention else RealAttention

        # Pilot stream
        self.norm_p1 = NormCls(d_model)
        self.self_attn_p = AttnCls(d_model, n_heads=n_heads)
        self.norm_p2 = NormCls(d_model) if use_joint_pilot_data else None
        self.cross_d2p = AttnCls(d_model, n_heads=n_heads) if use_joint_pilot_data else None
        self.norm_p3 = NormCls(d_model)
        self.ffn_p_1 = ComplexLinear(d_model, 2 * d_model, bias=False)
        self.act_p = RadialActivation()
        self.ffn_p_2 = ComplexLinear(2 * d_model, d_model, bias=False)

        # Data stream
        self.norm_d1 = NormCls(d_model)
        self.self_attn_d = AttnCls(d_model, n_heads=n_heads)
        self.norm_d2 = NormCls(d_model) if use_joint_pilot_data else None
        self.cross_p2d = AttnCls(d_model, n_heads=n_heads) if use_joint_pilot_data else None
        self.norm_d3 = NormCls(d_model)
        self.ffn_d_1 = ComplexLinear(d_model, 2 * d_model, bias=False)
        self.act_d = RadialActivation()
        self.ffn_d_2 = ComplexLinear(2 * d_model, d_model, bias=False)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pilot self-attention (preserve subcarrier amplitudes in values)
        p_res = p + self.self_attn_p(self.norm_p1(p), value_context=p)
        # Data self-attention
        d_res = d + self.self_attn_d(self.norm_d1(d), value_context=d)

        if self.use_joint_pilot_data and self.cross_p2d is not None and self.cross_d2p is not None:
            # Cross-attention: channel propagation p -> d
            d_cross = d_res + self.cross_p2d(
                self.norm_d2(d_res), context=self.norm_p1(p_res), value_context=p_res
            )
            # Cross-attention: distortion feedback d -> p
            p_cross = p_res + self.cross_d2p(
                self.norm_p2(p_res), context=self.norm_d1(d_res), value_context=d_res
            )
        else:
            d_cross = d_res
            p_cross = p_res

        # Radial feedforward
        p_out = p_cross + self.ffn_p_2(self.act_p(self.ffn_p_1(self.norm_p3(p_cross))))
        d_out = d_cross + self.ffn_d_2(self.act_d(self.ffn_d_1(self.norm_d3(d_cross))))

        return p_out, d_out


class FloatFormerOutput(dict):
    """Dictionary subclass supporting both dict access out['llrs'] and tuple unpacking (h_hat, llrs, dist_est)."""

    def __init__(self, h_hat: torch.Tensor, llrs: torch.Tensor, dist_est: torch.Tensor):
        super().__init__(h_hat=h_hat, llrs=llrs, dist_est=dist_est)
        self.h_hat = h_hat
        self.llrs = llrs
        self.dist_est = dist_est

    def __iter__(self):
        return iter((self.h_hat, self.llrs, self.dist_est))


class EMFloatFormer(nn.Module):
    """C*-Gauge-Invariant EM-Native Transformer Architecture.
    
    Integrates exact pilot gauge normalization, Hermitian Green's propagator attention,
    bidirectional pilot-data cross-attention, physical channel estimation,
    and calibrated 16-QAM soft LLR calculation.
    """

    def __init__(
        self,
        d_model: int = 12,
        n_heads: int = 2,
        depth: int = 2,
        n_sc: int = 32,
        use_gauge_norm: bool = True,
        use_hermitian_attention: bool = True,
        use_joint_pilot_data: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.n_sc = n_sc
        self.use_gauge_norm = use_gauge_norm
        self.use_hermitian_attention = use_hermitian_attention
        self.use_joint_pilot_data = use_joint_pilot_data

        self.embed = JointEmbedding(n=n_sc, d_model=d_model)
        self.blocks = nn.ModuleList([
            BidirectionalBlock(
                d_model=d_model,
                n_heads=n_heads,
                use_gauge_norm=use_gauge_norm,
                use_hermitian_attention=use_hermitian_attention,
                use_joint_pilot_data=use_joint_pilot_data,
            )
            for _ in range(depth)
        ])
        self.norm_p = ComplexRMSNorm(d_model) if use_gauge_norm else nn.Identity()
        self.norm_d = ComplexRMSNorm(d_model) if use_gauge_norm else nn.Identity()

        # Output heads + complex Bussgang equalization scale
        self.head_h = ComplexLinear(d_model, 1, bias=False)
        self.head_dist = ComplexLinear(d_model, 1, bias=False)
        self.gamma_eff = ComplexLinear(1, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        std_embed = 1.0 / math.sqrt(2.0 * self.d_model)
        std_head = 0.01 / math.sqrt(2.0 * self.d_model)

        nn.init.normal_(self.embed.proj_p.weight_r, 0.0, std_embed)
        nn.init.normal_(self.embed.proj_p.weight_i, 0.0, std_embed)
        nn.init.normal_(self.embed.proj_d.weight_r, 0.0, std_embed)
        nn.init.normal_(self.embed.proj_d.weight_i, 0.0, std_embed)

        nn.init.normal_(self.head_h.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_h.weight_i, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_i, 0.0, std_head)

        with torch.no_grad():
            self.gamma_eff.weight_r.fill_(1.0)
            self.gamma_eff.weight_i.fill_(0.0)

    def forward(
        self,
        yp: torch.Tensor,
        yd: torch.Tensor,
        nv: torch.Tensor,
    ) -> FloatFormerOutput:
        # Preserve caller dtypes at the boundary while executing internal C*-equivariant
        # backbone and LLR calibration in double precision (complex128 / float64) to eliminate
        # complex64 rounding drift across stacked attention / RMSNorm layers.
        out_complex_dtype = yp.dtype
        out_real_dtype = yp.real.dtype

        # Standardize input dimensions
        if yp.dim() == 1:
            yp = yp.unsqueeze(0)
        if yd.dim() == 1:
            yd = yd.unsqueeze(0)
        if nv.dim() == 1:
            nv = nv.unsqueeze(-1)

        yp64 = yp.to(torch.complex128)
        yd64 = yd.to(torch.complex128)
        nv64 = nv.to(torch.float64)

        # 1. Exact C* Gauge Normalization in complex128 / float64
        if self.use_gauge_norm:
            yp_bar, yd_bar, s0 = apply_gauge_normalization(yp64, yd64)
            nv_bar = nv64 / (s0 ** 2)
        else:
            s0 = torch.ones((yp64.shape[0], 1), dtype=torch.float64, device=yp.device)
            yp_bar = yp64
            yd_bar = yd64
            nv_bar = nv64

        # 2. Dual-Stream Token Embedding (complex128)
        p_tok = self.embed.embed_pilot(yp_bar)
        d_tok = self.embed.embed_data(yd_bar)

        # 3. Stacked Bidirectional Propagator Blocks (complex128)
        for block in self.blocks:
            p_tok, d_tok = block(p_tok, d_tok)

        p_out = p_tok + 0.1 * self.norm_p(p_tok)
        d_out = d_tok + 0.1 * self.norm_d(d_tok)

        # 4. Physical Channel Frequency Response Estimation
        h_bar_delta = self.head_h(p_out).squeeze(-1)
        h_bar = yp_bar + h_bar_delta
        h_hat = rescale_channel_estimate(h_bar, s0).to(out_complex_dtype)

        # 5. Nonlinear PA Inter-Carrier Distortion Estimation
        dist_bar = self.head_dist(d_out).squeeze(-1)
        dist_est = rescale_channel_estimate(dist_bar, s0).to(out_complex_dtype)

        # 6. Conditionally Unbiased Physical Equalization & Phase Cancellation in float64/complex128
        h_eff64 = self.gamma_eff(h_bar.unsqueeze(-1)).squeeze(-1)
        h_power = h_eff64.real ** 2 + h_eff64.imag ** 2
        nv_exp = nv_bar if nv_bar.shape == h_power.shape else nv_bar.expand_as(h_power)

        dist_eq = (dist_bar * h_eff64.conj()) / (h_power + 1e-4)
        z_eq = (yd_bar * h_eff64.conj()) / (h_power + 1e-4) - dist_eq

        # 7. Calibrated 16-QAM Soft LLRs (detached variance prevents BCE variance gaming)
        h_power_det = h_power.detach()
        dist_power_det = (dist_bar.real ** 2 + dist_bar.imag ** 2).detach()
        sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 3.5e-2
        llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
        llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)

        return FloatFormerOutput(h_hat=h_hat, llrs=llrs, dist_est=dist_est)


def create_floatformer(
    use_gauge_norm: bool = True,
    use_hermitian_attention: bool = True,
    use_joint_pilot_data: bool = True,
    **kwargs: Any,
) -> EMFloatFormer:
    """Ablation model factory supporting orthogonal toggles for M4 factorial probing."""
    return EMFloatFormer(
        use_gauge_norm=use_gauge_norm,
        use_hermitian_attention=use_hermitian_attention,
        use_joint_pilot_data=use_joint_pilot_data,
        **kwargs,
    )


def compare_pilot_vs_bidirectional() -> float:
    """Compare BER performance of pilot-only LS vs trained EMFloatFormer under strong PA.
    
    Returns:
        gain_db: Unclamped Bit Error Rate gain in dB.
    """
    torch.manual_seed(42)
    train_gen = torch.Generator().manual_seed(101)
    train_frame = generate_frame(
        batch_size=128, n=32, snr_db=18.0, pa_condition="strong", generator=train_gen
    )

    eval_gen = torch.Generator().manual_seed(2026)
    eval_frame = generate_frame(
        batch_size=64, n=32, snr_db=18.0, pa_condition="strong", generator=eval_gen
    )

    # 1. Pilot-only LS baseline without bidirectional PA cancellation
    h_ls = eval_frame.yp
    h_ls_pow = h_ls.real ** 2 + h_ls.imag ** 2
    z_ls = (eval_frame.yd * h_ls.conj()) / (h_ls_pow + 1e-4)
    llrs_ls = calibrated_16qam_llrs(
        z_ls, eval_frame.noise_var / (h_ls_pow + 1e-4) + 1e-4, method="exact"
    )
    bits_ls = (llrs_ls > 0).long()
    ber_pilot = float((bits_ls != eval_frame.bits).float().mean().item())

    # 2. Genuine EMFloatFormer with active bidirectional pilot-data attention
    model = EMFloatFormer(d_model=12, n_heads=2, depth=2)
    model.train()
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "gamma_eff" not in n], "lr": 1e-2},
        {"params": model.gamma_eff.parameters(), "lr": 3e-2},
    ])
    loss_fn = JointPhysicalLoss(lambda_nmse=2.0, lambda_pa=0.0)

    for _ in range(50):
        idx = torch.randperm(128)[:32]
        optimizer.zero_grad()
        out = model(train_frame.yp[idx], train_frame.yd[idx], train_frame.noise_var[idx])
        loss = loss_fn.bce_loss(out["llrs"], train_frame.bits[idx]) + 2.0 * loss_fn.nmse_loss(
            out["h_hat"], train_frame.h_true[idx]
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out_bidi = model(eval_frame.yp, eval_frame.yd, eval_frame.noise_var)
        bits_bidi = (out_bidi["llrs"] > 0).long()
        ber_bidi = float((bits_bidi != eval_frame.bits).float().mean().item())

    gain_db = 10.0 * math.log10(max(ber_pilot, 1e-4) / max(ber_bidi, 1e-4))
    return gain_db


def evaluate_noise_data_feedback() -> float:
    """Evaluate channel estimate degradation when data tokens are pure random noise.
    
    Returns:
        deg_db: Unclamped pilot NMSE degradation in dB (< 1.0 dB).
    """
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    frame = generate_frame(batch_size=16, n=32, snr_db=20.0, generator=g)

    with torch.no_grad():
        out_clean = model(frame.yp, frame.yd, frame.noise_var)
        nmse_clean = ((out_clean["h_hat"] - frame.h_true).abs() ** 2).mean().item()

        # Pure noise data tokens generated deterministically
        noise_r = torch.randn(frame.yd.shape, dtype=torch.float32, generator=g)
        noise_i = torch.randn(frame.yd.shape, dtype=torch.float32, generator=g)
        yd_noise = torch.complex(noise_r, noise_i) * math.sqrt(0.5)
        out_noisy = model(frame.yp, yd_noise, frame.noise_var)
        nmse_noisy = ((out_noisy["h_hat"] - frame.h_true).abs() ** 2).mean().item()

    deg_db = 10.0 * abs(math.log10(max(nmse_noisy, 1e-6) / max(nmse_clean, 1e-6)))
    return deg_db


def evaluate_perfect_pilot_demapping() -> float:
    """Verify that noiseless pilots enable near-optimal AWGN demapping.
    
    Returns:
        ber: Bit Error Rate (< 1e-3).
    """
    torch.manual_seed(99)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(99)
    # High SNR, clean PA condition, zero Doppler
    frame = generate_frame(batch_size=32, n=32, snr_db=50.0, fd=0.0, profile="train3", pa_condition="clean", generator=g)

    with torch.no_grad():
        out = model(frame.yp, frame.yd, frame.noise_var)
        bits_hat = (out["llrs"] > 0).long()
        ber = float((bits_hat != frame.bits).float().mean().item())

    return ber


# =============================================================================
# Feature F11: Joint Physical Consistency Loss
# =============================================================================

def soft_reconstruct_16qam(llrs: torch.Tensor) -> torch.Tensor:
    """Soft constellation symbol reconstruction from bit LLRs for 16-QAM.
    
    Args:
        llrs: (..., N, 4) tensor of posterior bit log-likelihood ratios.
        
    Returns:
        x_soft: (..., N) expected complex constellation symbols.
    """
    p0 = torch.sigmoid(llrs[..., 0])
    p1 = torch.sigmoid(llrs[..., 1])
    p2 = torch.sigmoid(llrs[..., 2])
    p3 = torch.sigmoid(llrs[..., 3])

    e_i = ((2.0 * p0 - 1.0) * (3.0 - 2.0 * p1)) / SIGMA_16QAM
    e_q = ((2.0 * p2 - 1.0) * (3.0 - 2.0 * p3)) / SIGMA_16QAM
    return torch.complex(e_i, e_q)


class JointPhysicalLoss(nn.Module):
    """Multi-task Joint Physical Consistency Objective.
    
    L_joint = L_BCE + lambda_nmse * L_NMSE + lambda_pa * L_PA-res
    """

    def __init__(self, lambda_nmse: float = 1.0, lambda_pa: float = 0.25):
        super().__init__()
        self.lambda_nmse = lambda_nmse
        self.lambda_pa = lambda_pa

    def bce_loss(self, llrs: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Numerically stable BCE loss on bit log-likelihood ratios."""
        return F.binary_cross_entropy_with_logits(llrs, bits.float())

    def nmse_loss(self, h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
        """Scale-invariant normalized mean squared error on channel estimates.
        
        NMSE = ||h_est - h_true||^2 / (||h_true||^2 + eps)
        """
        err_sq = (h_est - h_true).abs() ** 2
        den_sq = h_true.abs() ** 2
        num_sum = err_sq.sum(dim=-1)
        den_sum = den_sq.sum(dim=-1)
        # Protect against zero channel power while preserving exact scale invariance
        den_safe = torch.where(den_sum < 1e-12, torch.full_like(den_sum, 1e-12), den_sum)
        return (num_sum / den_safe).mean()

    @staticmethod
    def _normalize_pa_A(pa_val: Any) -> Optional[float]:
        if pa_val is None:
            return None
        if isinstance(pa_val, str):
            cond = pa_val.strip().lower()
            if cond == "clean":
                return None
            if cond == "mild":
                return 2.25
            if cond == "strong":
                return 1.25
            raise ValueError(f"Unknown pa_condition string: {pa_val!r}")
        a_float = float(pa_val)
        if math.isinf(a_float):
            return None
        return a_float

    def _resolve_pa_A(
        self,
        frame: Any,
        A: Optional[Union[float, str]] = None,
    ) -> Optional[float]:
        if A is not None:
            return self._normalize_pa_A(A)
        if hasattr(frame, "A") and getattr(frame, "A") is not None:
            return self._normalize_pa_A(getattr(frame, "A"))
        if hasattr(frame, "pa_condition") and getattr(frame, "pa_condition") is not None:
            return self._normalize_pa_A(getattr(frame, "pa_condition"))
        return 1.25

    def pa_residual_loss(
        self,
        yd: torch.Tensor,
        h_est: torch.Tensor,
        x_est: torch.Tensor,
        A: Optional[float] = 1.25,
    ) -> torch.Tensor:
        """Forward Rapp PA consistency loss between received data and forward model."""
        if A is None or (isinstance(A, (int, float)) and math.isinf(A)):
            x_pa_freq = x_est
        else:
            # Synthesize time-domain envelope using unitary ortho FFT matching simulator.py
            x_time = torch.fft.ifft(x_est, dim=-1, norm="ortho")
            r2 = x_time.real ** 2 + x_time.imag ** 2
            r = torch.sqrt(r2 + 1e-24)
            two_p = 4.0
            scale = (1.0 + (r / A) ** two_p) ** (-1.0 / two_p)
            phi = 0.5 * r2 / (1.0 + r2)
            x_pa_time = x_time * scale * torch.complex(torch.cos(phi), torch.sin(phi))
            x_pa_freq = torch.fft.fft(x_pa_time, dim=-1, norm="ortho")

        y_model = h_est * x_pa_freq
        num = (yd - y_model).abs() ** 2
        den = yd.abs() ** 2
        num_sum = num.sum(dim=-1)
        den_sum = den.sum(dim=-1)
        den_safe = torch.where(den_sum < 1e-12, torch.full_like(den_sum, 1e-12), den_sum)
        return (num_sum / den_safe).mean()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        frame: Any,
        A: Optional[Union[float, str]] = None,
    ) -> torch.Tensor:
        llrs = out["llrs"]
        h_est = out["h_hat"]

        l_bce = self.bce_loss(llrs, frame.bits)
        h_true = getattr(frame, "h_true", None)
        if h_true is None:
            h_true = frame.yp
        l_nmse = self.nmse_loss(h_est, h_true)

        x_est = soft_reconstruct_16qam(llrs)
        a_resolved = self._resolve_pa_A(frame, A=A)
        l_pa = self.pa_residual_loss(frame.yd, h_est, x_est, A=a_resolved)

        return l_bce + self.lambda_nmse * l_nmse + self.lambda_pa * l_pa


def verify_loss_gradient_invariance() -> float:
    """Verify that ALL parameter gradients are non-zero on step 1 and strictly C*-invariant (< 1e-5)."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    loss_fn = JointPhysicalLoss()
    g = torch.Generator().manual_seed(42)
    frame = generate_frame(batch_size=4, n=32, snr_db=20.0, generator=g)

    # 1. Base gradient
    out1 = model(frame.yp, frame.yd, frame.noise_var)
    loss1 = loss_fn(out1, frame)
    loss1.backward()
    grads1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    if len(grads1) != sum(1 for _ in model.parameters()):
        raise RuntimeError("Missing parameter gradients on step 1")
    zero_grads = sum(1 for g1 in grads1 if float(g1.abs().sum()) == 0.0)
    if zero_grads > 0:
        raise RuntimeError(f"Found {zero_grads} zero-gradient parameter tensors on step 1")

    model.zero_grad()

    # 2. Scaled & phase-rotated gradient with non-power-of-2 c = 3.7 * exp(j * 0.95)
    r = 3.7
    phi = 0.95
    c = r * torch.exp(1j * torch.tensor(phi))
    yp_s = frame.yp * c
    yd_s = frame.yd * c
    nv_s = frame.noise_var * (r ** 2)
    h_true_s = frame.h_true * c

    frame_s = FrameData(
        yp=yp_s,
        yd=yd_s,
        bits=frame.bits,
        h_true=h_true_s,
        noise_var=nv_s,
        pilot_tx=frame.pilot_tx,
        data_tx=frame.data_tx,
    )
    out2 = model(yp_s, yd_s, nv_s)
    loss2 = loss_fn(out2, frame_s)
    loss2.backward()
    grads2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    max_diff = max((g1 - g2).abs().max().item() for g1, g2 in zip(grads1, grads2))
    return max_diff


def audit_loss_scale_invariance() -> float:
    """Audit loss invariance across 24 octaves of input dynamic range (< 1e-5)."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    loss_fn = JointPhysicalLoss()
    g = torch.Generator().manual_seed(42)
    frame = generate_frame(batch_size=4, n=32, snr_db=20.0, generator=g)

    with torch.no_grad():
        out_base = model(frame.yp, frame.yd, frame.noise_var)
        loss_base = loss_fn(out_base, frame).item()
        max_rel_diff = 0.0

        for p in [-12, -6, 0, 6, 12]:
            r = 2.0 ** p
            yp_s = frame.yp * r
            yd_s = frame.yd * r
            nv_s = frame.noise_var * (r ** 2)
            h_true_s = frame.h_true * r

            frame_s = FrameData(
                yp=yp_s,
                yd=yd_s,
                bits=frame.bits,
                h_true=h_true_s,
                noise_var=nv_s,
                pilot_tx=frame.pilot_tx,
                data_tx=frame.data_tx,
            )
            out_s = model(yp_s, yd_s, nv_s)
            loss_s = loss_fn(out_s, frame_s).item()
            rel_diff = abs(loss_s - loss_base) / (abs(loss_base) + 1e-12)
            max_rel_diff = max(max_rel_diff, rel_diff)

    return max_rel_diff


def train_step_joint_loss() -> bool:
    """Execute 1 training step of bidirectional model with joint loss and verify finite gradients on all params."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointPhysicalLoss()
    g = torch.Generator().manual_seed(42)
    frame = generate_frame(batch_size=8, n=32, snr_db=15.0, pa_condition="strong", generator=g)

    optimizer.zero_grad()
    out = model(frame.yp, frame.yd, frame.noise_var)
    loss = loss_fn(out, frame)
    loss.backward()

    for p in model.parameters():
        if p.grad is None:
            return False
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            return False
        if float(p.grad.abs().sum()) == 0.0:
            return False

    optimizer.step()
    return True


def check_loss_gradient_scale() -> float:
    """Verify that parameter gradients under scaling match to within < 1e-5."""
    return verify_loss_gradient_invariance()


# =============================================================================
# Feature F12: Numerical Equivariance Verification Functions
# =============================================================================

def audit_gain_equivariance(octaves: int = 24) -> Tuple[float, int]:
    """Verify model across 24 gain octaves r in [2^-12, 2^12] with max error < 1e-5 and 0 flips."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]
        max_err = 0.0
        flips = 0

        for p in range(-12, 13):
            r = 2.0 ** p
            out_scaled = model(yp * r, yd * r, nv * (r ** 2))
            diff = (out_scaled["llrs"] - base_llrs).abs().max().item()
            max_err = max(max_err, diff)
            flip_count = ((out_scaled["llrs"] > 0) != (base_llrs > 0)).sum().item()
            flips += flip_count

    return max_err, flips


def audit_phase_equivariance(num_phases: int = 16) -> Tuple[float, int]:
    """Verify model across 16 phase circle points with 0 decision flips and max error < 1e-5."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]
        max_err = 0.0
        flips = 0

        for k in range(num_phases):
            phi = 2.0 * math.pi * k / num_phases
            c = torch.exp(1j * torch.tensor(phi))
            out_rot = model(yp * c, yd * c, nv)
            diff = (out_rot["llrs"] - base_llrs).abs().max().item()
            max_err = max(max_err, diff)
            flip_count = ((out_rot["llrs"] > 0) != (base_llrs > 0)).sum().item()
            flips += flip_count

    return max_err, flips


def audit_c_star_equivariance(num_samples: int = 20) -> Tuple[float, int]:
    """Verify simultaneous scale and phase rotation invariance."""
    torch.manual_seed(123)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(123)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]
        max_err = 0.0
        flips = 0

        rg = torch.Generator().manual_seed(1234)
        for _ in range(num_samples):
            u = torch.empty(1).uniform_(-12.0, 12.0, generator=rg).item()
            phi = torch.empty(1).uniform_(-math.pi, math.pi, generator=rg).item()
            r = 2.0 ** u
            c = r * torch.exp(1j * torch.tensor(phi))

            out_c = model(yp * c, yd * c, nv * (r ** 2))
            diff = (out_c["llrs"] - base_llrs).abs().max().item()
            max_err = max(max_err, diff)
            flip_count = ((out_c["llrs"] > 0) != (base_llrs > 0)).sum().item()
            flips += flip_count

    return max_err, flips


def audit_channel_equivariance() -> float:
    """Verify channel estimate satisfies H_hat(c Y) = c H_hat(Y)."""
    torch.manual_seed(999)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(999)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        h_base = out_base["h_hat"]
        r = 3.5
        phi = 1.25
        c = r * torch.exp(1j * torch.tensor(phi))

        out_c = model(yp * c, yd * c, nv * (r ** 2))
        h_c = out_c["h_hat"]
        rel_err = (h_c - c * h_base).abs().max().item() / (r * (h_base.abs().max().item() + 1e-12))

    return rel_err


def run_standalone_equivariance_check() -> bool:
    """Standalone equivariance verification runner."""
    err_gain, flips_gain = audit_gain_equivariance(octaves=12)
    err_phase, flips_phase = audit_phase_equivariance(num_phases=8)
    err_c, flips_c = audit_c_star_equivariance(num_samples=10)
    err_chan = audit_channel_equivariance()

    success = (
        err_gain < 1e-5
        and flips_gain == 0
        and err_phase < 1e-5
        and flips_phase == 0
        and err_c < 1e-5
        and flips_c == 0
        and err_chan < 1e-5
    )
    return success


def sweep_phases_dense(num_points: int = 100) -> float:
    """Sweep 100 uniformly spaced phase angles in [0, 2pi) and verify max error < 1e-5."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((2, 1), 0.01)

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]
        max_err = 0.0

        for k in range(num_points):
            phi = 2.0 * math.pi * k / num_points
            c = torch.exp(1j * torch.tensor(phi))
            out_rot = model(yp * c, yd * c, nv)
            diff = (out_rot["llrs"] - base_llrs).abs().max().item()
            max_err = max(max_err, diff)

    return max_err


def check_irrational_phases() -> int:
    """Verify irrational phase angles pi/7, e, sqrt(2) yield 0 bit flips."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((2, 1), 0.01)
    total_flips = 0

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]

        for phi in [math.pi / 7.0, math.e, math.sqrt(2.0)]:
            c = torch.exp(1j * torch.tensor(phi))
            out_rot = model(yp * c, yd * c, nv)
            flips = ((out_rot["llrs"] > 0) != (base_llrs > 0)).sum().item()
            total_flips += flips

    return total_flips


def check_noise_quad_scaling(r: float = 3.5) -> float:
    """Verify scaling noise variance by r^2 alongside signal by r maintains identical bit LLRs."""
    torch.manual_seed(42)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(42)
    yp = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(2, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((2, 1), 0.02)

    with torch.no_grad():
        out1 = model(yp, yd, nv)
        out2 = model(yp * r, yd * r, nv * (r ** 2))
        diff = (out1["llrs"] - out2["llrs"]).abs().max().item()

    return diff


def verify_gauge_attention_joint() -> bool:
    """P08 (F8+F9): Verify pilot RMS anchoring combined with Hermitian relative-phase attention."""
    torch.manual_seed(77)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(77)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    with torch.no_grad():
        out1 = model(yp, yd, nv)
        r = 3.7
        phi = 0.95
        c = r * torch.exp(1j * torch.tensor(phi))
        out2 = model(yp * c, yd * c, nv * (r ** 2))

        diff = (out1["llrs"] - out2["llrs"]).abs().max().item()
        flips = ((out1["llrs"] > 0) != (out2["llrs"] > 0)).sum().item()
        h_err = (out2["h_hat"] - c * out1["h_hat"]).abs().max().item() / (
            r * (out1["h_hat"].abs().max().item() + 1e-12)
        )

    return (diff < 1e-5) and (flips == 0) and (h_err < 1e-5)


def audit_full_equivariance() -> Tuple[float, int]:
    """P09 (F8+F12): Integrated equivariance verification over 24 octaves and full phase circle."""
    torch.manual_seed(88)
    model = EMFloatFormer(depth=1)
    model.eval()
    g = torch.Generator().manual_seed(88)
    yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
    nv = torch.full((4, 1), 0.01)

    max_err = 0.0
    total_flips = 0

    with torch.no_grad():
        out_base = model(yp, yd, nv)
        base_llrs = out_base["llrs"]

        for p in [-12, -8, -4, 0, 4, 8, 12]:
            r = 2.0 ** p
            for k in range(8):
                phi = 2.0 * math.pi * k / 8.0
                c = r * torch.exp(1j * torch.tensor(phi))
                out_c = model(yp * c, yd * c, nv * (r ** 2))
                diff = (out_c["llrs"] - base_llrs).abs().max().item()
                max_err = max(max_err, diff)
                flips = ((out_c["llrs"] > 0) != (base_llrs > 0)).sum().item()
                total_flips += flips

    return max_err, total_flips


def verify_bidirectional_green_prop() -> bool:
    """P11 (F9+F10): Verify bidirectional cross-attention using Hermitian inner products."""
    torch.manual_seed(123)
    block = BidirectionalBlock(d_model=32, n_heads=2)
    g = torch.Generator().manual_seed(123)
    p = torch.randn(2, 16, 32, dtype=torch.complex64, generator=g, requires_grad=True)
    d = torch.randn(2, 16, 32, dtype=torch.complex64, generator=g, requires_grad=True)

    p_out, d_out = block(p, d)
    if p_out.shape != p.shape or d_out.shape != d.shape:
        return False

    # Check bidirectional gradient propagation
    loss_d = (d_out.abs() ** 2).sum()
    loss_d.backward(retain_graph=True)
    if p.grad is None or float(p.grad.abs().sum()) == 0.0:
        return False

    loss_p = (p_out.abs() ** 2).sum()
    loss_p.backward()
    if d.grad is None or float(d.grad.abs().sum()) == 0.0:
        return False

    return True


# =============================================================================
# Stage P4-A Comparative Baselines & Benchmark Runner (M2.3)
# =============================================================================

class ComplexConv1d(nn.Module):
    """Bias-free circular 1D complex convolution satisfying W(c Z) = c W(Z) for c in C*."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_r = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=torch.float32)
        )
        self.weight_i = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=torch.float32)
        )
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(2.0 * max(1, self.in_channels * self.kernel_size))
        nn.init.normal_(self.weight_r, 0.0, std)
        nn.init.normal_(self.weight_i, 0.0, std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, N) complex64 or complex128
        xr = z.real
        xi = z.imag
        real_dtype = xr.dtype
        pad_left = self.kernel_size // 2
        pad_right = self.kernel_size - 1 - pad_left
        if pad_left > 0 or pad_right > 0:
            xr = F.pad(xr, (pad_left, pad_right), mode="circular")
            xi = F.pad(xi, (pad_left, pad_right), mode="circular")
        w_r = self.weight_r.to(real_dtype)
        w_i = self.weight_i.to(real_dtype)
        out_r = F.conv1d(xr, w_r) - F.conv1d(xi, w_i)
        out_i = F.conv1d(xr, w_i) + F.conv1d(xi, w_r)
        return torch.complex(out_r, out_i)


class _LocalEquivBlock(nn.Module):
    """Dual-stream local circular equivariant convolutional block (no global attention)."""

    def __init__(self, width: int = 14, kernel_size: int = 5, ffn_dim: int = 12):
        super().__init__()
        self.norm_p1 = ComplexRMSNorm(width)
        self.conv_p = ComplexConv1d(width, width, kernel_size=kernel_size)
        self.norm_d1 = ComplexRMSNorm(width)
        self.conv_d = ComplexConv1d(width, width, kernel_size=kernel_size)

        self.norm_p2 = ComplexRMSNorm(width)
        self.mix_d2p = ComplexConv1d(2 * width, width, kernel_size=1)
        self.norm_d2 = ComplexRMSNorm(width)
        self.mix_p2d = ComplexConv1d(2 * width, width, kernel_size=1)

        self.norm_p3 = ComplexRMSNorm(width)
        self.ffn_p1 = ComplexConv1d(width, ffn_dim, kernel_size=1)
        self.act_p = RadialActivation()
        self.ffn_p2 = ComplexConv1d(ffn_dim, width, kernel_size=1)

        self.norm_d3 = ComplexRMSNorm(width)
        self.ffn_d1 = ComplexConv1d(width, ffn_dim, kernel_size=1)
        self.act_d = RadialActivation()
        self.ffn_d2 = ComplexConv1d(ffn_dim, width, kernel_size=1)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # p, d: (B, N, C) complex128; ComplexConv1d expects (B, C, N)
        pn1 = self.norm_p1(p).transpose(1, 2)
        dn1 = self.norm_d1(d).transpose(1, 2)
        p_res = p + self.conv_p(pn1).transpose(1, 2)
        d_res = d + self.conv_d(dn1).transpose(1, 2)

        pn2 = self.norm_p2(p_res).transpose(1, 2)
        dn2 = self.norm_d2(d_res).transpose(1, 2)
        p_cross = p_res + self.mix_d2p(torch.cat([pn2, dn2], dim=1)).transpose(1, 2)
        d_cross = d_res + self.mix_p2d(torch.cat([dn2, pn2], dim=1)).transpose(1, 2)

        pn3 = self.norm_p3(p_cross).transpose(1, 2)
        dn3 = self.norm_d3(d_cross).transpose(1, 2)
        p_out = p_cross + self.ffn_p2(self.act_p(self.ffn_p1(pn3))).transpose(1, 2)
        d_out = d_cross + self.ffn_d2(self.act_d(self.ffn_d1(dn3))).transpose(1, 2)
        return p_out, d_out


class LocalEquivConv16QAM(nn.Module):
    """Parameter-matched (~14.1k params) C*-equivariant local circular Conv1d 16-QAM baseline.

    Uses pilot gauge normalization, ComplexRMSNorm, bias-free ComplexConv1d, and RadialActivation,
    without global Hermitian Green's Propagator Attention.
    """

    def __init__(
        self,
        width: int = 14,
        kernel_size: int = 5,
        depth: int = 2,
        ffn_dim: int = 12,
    ):
        super().__init__()
        self.width = width
        self.kernel_size = kernel_size
        self.depth = depth

        self.proj_p = ComplexConv1d(1, width, kernel_size=3)
        self.proj_d = ComplexConv1d(1, width, kernel_size=3)
        self.blocks = nn.ModuleList([
            _LocalEquivBlock(width=width, kernel_size=kernel_size, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.norm_p = ComplexRMSNorm(width)
        self.norm_d = ComplexRMSNorm(width)

        self.head_h = ComplexConv1d(width, 1, kernel_size=3)
        self.head_dist = ComplexConv1d(width, 1, kernel_size=3)
        self.gamma_eff = ComplexLinear(1, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        std_head = 0.01 / math.sqrt(2.0 * max(1, self.width))
        nn.init.normal_(self.head_h.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_h.weight_i, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_i, 0.0, std_head)
        with torch.no_grad():
            self.gamma_eff.weight_r.fill_(1.0)
            self.gamma_eff.weight_i.fill_(0.0)

    def forward(
        self,
        yp: torch.Tensor,
        yd: torch.Tensor,
        nv: torch.Tensor,
    ) -> FloatFormerOutput:
        out_complex_dtype = yp.dtype
        out_real_dtype = yp.real.dtype

        if yp.dim() == 1:
            yp = yp.unsqueeze(0)
        if yd.dim() == 1:
            yd = yd.unsqueeze(0)
        if nv.dim() == 1:
            nv = nv.unsqueeze(-1)

        yp64 = yp.to(torch.complex128)
        yd64 = yd.to(torch.complex128)
        nv64 = nv.to(torch.float64)

        yp_bar, yd_bar, s0 = apply_gauge_normalization(yp64, yd64)
        nv_bar = nv64 / (s0 ** 2)

        p_tok = self.proj_p(yp_bar.unsqueeze(1)).transpose(1, 2)
        d_tok = self.proj_d(yd_bar.unsqueeze(1)).transpose(1, 2)

        for block in self.blocks:
            p_tok, d_tok = block(p_tok, d_tok)

        p_out = p_tok + 0.1 * self.norm_p(p_tok)
        d_out = d_tok + 0.1 * self.norm_d(d_tok)

        h_bar = yp_bar + self.head_h(p_out.transpose(1, 2)).squeeze(1)
        h_hat = rescale_channel_estimate(h_bar, s0).to(out_complex_dtype)

        dist_bar = self.head_dist(d_out.transpose(1, 2)).squeeze(1)
        dist_est = rescale_channel_estimate(dist_bar, s0).to(out_complex_dtype)

        h_eff64 = self.gamma_eff(h_bar.unsqueeze(-1)).squeeze(-1)
        h_power = h_eff64.real ** 2 + h_eff64.imag ** 2
        nv_exp = nv_bar if nv_bar.shape == h_power.shape else nv_bar.expand_as(h_power)

        dist_eq = (dist_bar * h_eff64.conj()) / (h_power + 1e-4)
        z_eq = (yd_bar * h_eff64.conj()) / (h_power + 1e-4) - dist_eq

        h_power_det = h_power.detach()
        dist_power_det = (dist_bar.real ** 2 + dist_bar.imag ** 2).detach()
        sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 3.5e-2
        llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
        llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)

        return FloatFormerOutput(h_hat=h_hat, llrs=llrs, dist_est=dist_est)


class TinyJoint16QAM(nn.Module):
    """Non-equivariant joint Conv1d 16-QAM baseline inspired by exploratory TinyJoint."""

    def __init__(self, width: int = 16, taps: int = 8):
        super().__init__()
        self.width = width
        self.taps = taps
        self.stem = nn.Conv1d(10, width, 1)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(width, width, 3, padding=1, padding_mode="circular")
        self.out_llr = nn.Conv1d(width, 4, 1)
        self.head_h = nn.Conv1d(width, 2, 1)
        self.head_dist = nn.Conv1d(width, 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for head in (self.out_llr, self.head_h, self.head_dist):
            nn.init.normal_(head.weight, std=0.004)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        yp: torch.Tensor,
        yd: torch.Tensor,
        nv: torch.Tensor,
    ) -> FloatFormerOutput:
        if yp.dim() == 1:
            yp = yp.unsqueeze(0)
        if yd.dim() == 1:
            yd = yd.unsqueeze(0)
        if nv.dim() == 1:
            nv = nv.unsqueeze(-1)

        n = yp.shape[-1]
        taps = min(self.taps, n)
        h = delay_est_dft(yp, taps=taps, n=n)
        nv_bc = nv if nv.shape == h.real.shape else nv.expand_as(h.real)
        z = yd * h.conj() / (h.real ** 2 + h.imag ** 2 + nv_bc + 0.03)

        feat = torch.stack(
            (
                z.real,
                z.imag,
                z.abs(),
                yp.real,
                yp.imag,
                yd.real,
                yd.imag,
                h.real,
                h.imag,
                torch.log10(nv_bc.clamp(min=1e-12)),
            ),
            dim=1,
        ).float()

        y = F.silu(self.stem(feat))
        y = y + 0.4 * F.silu(self.conv1(y))
        y = y + 0.4 * F.silu(self.conv2(y))

        delta_llr = self.out_llr(y).transpose(1, 2)
        dh = self.head_h(y).transpose(1, 2)
        dd = self.head_dist(y).transpose(1, 2)
        h_hat = (h + torch.complex(dh[..., 0], dh[..., 1])).to(yp.dtype)
        dist_est = torch.complex(dd[..., 0], dd[..., 1]).to(yp.dtype)
        dist_eq = dist_est * h.conj() / (h.real ** 2 + h.imag ** 2 + nv_bc + 0.03)
        z_corr = z - dist_eq
        base_llr = (
            torch.stack(
                (
                    z_corr.real.float(),
                    THRESH_16QAM - z_corr.real.abs().float(),
                    z_corr.imag.float(),
                    THRESH_16QAM - z_corr.imag.abs().float(),
                ),
                dim=-1,
            )
            * 5.0
        )
        llrs = base_llr + 0.5 * delta_llr

        return FloatFormerOutput(h_hat=h_hat, llrs=llrs.to(yp.real.dtype), dist_est=dist_est)


def compute_ece(llrs: torch.Tensor, bits: torch.Tensor, n_bins: int = 10) -> float:
    """Compute multi-bin Expected Calibration Error (ECE) over predicted bit confidences."""
    probs = torch.sigmoid(llrs.float()).reshape(-1)
    targets = bits.float().reshape(-1)
    preds = (probs > 0.5).float()
    confs = torch.where(probs >= 0.5, probs, 1.0 - probs)
    accs = (preds == targets).float()

    total = max(1, confs.numel())
    ece = 0.0
    bin_edges = torch.linspace(0.5, 1.0, n_bins + 1, device=confs.device)
    for idx in range(n_bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        count = int(mask.sum().item())
        if count > 0:
            bin_acc = float(accs[mask].mean().item())
            bin_conf = float(confs[mask].mean().item())
            ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def _train_comparative_model(
    model_or_factory: Union[nn.Module, Callable[[], nn.Module]],
    steps: int,
    batch_size: int,
    n_sc: int,
    profile: str,
    seed: int,
    has_gamma_eff: bool = True,
    backbone_lr: float = 5e-3,
    gamma_lr: float = 1.5e-2,
) -> Tuple[nn.Module, List[float]]:
    """Train a neural receiver model under mixed-PA JointPhysicalLoss with fixed seed."""
    torch.manual_seed(seed)
    model = (
        model_or_factory
        if isinstance(model_or_factory, nn.Module)
        else model_or_factory()
    )
    gen = torch.Generator().manual_seed(seed + 1000)
    model.train()

    if has_gamma_eff and hasattr(model, "gamma_eff"):
        optimizer = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "gamma_eff" not in n], "lr": backbone_lr},
            {"params": model.gamma_eff.parameters(), "lr": gamma_lr},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=backbone_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=backbone_lr * 0.1
    )
    loss_fn = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
    pa_cycle = ("strong", "mild", "clean")
    pa_to_A = {"strong": 1.25, "mild": 2.25, "clean": float("inf")}
    scale_ratios: List[float] = []

    for step in range(steps):
        pa_cond = pa_cycle[step % len(pa_cycle)]
        frame = generate_frame(
            batch_size=batch_size,
            n=n_sc,
            profile=profile,
            snr_db=(10.0, 30.0),
            pa_condition=pa_cond,
            generator=gen,
        )
        optimizer.zero_grad()
        out = model(frame.yp, frame.yd, frame.noise_var)
        loss = loss_fn(out, frame, A=pa_to_A[pa_cond])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            h_norm = out["h_hat"].abs().norm(dim=-1)
            t_norm = frame.h_true.abs().norm(dim=-1).clamp(min=1e-12)
            scale_ratios.append(float((h_norm / t_norm).mean().item()))

    model.eval()
    return model, scale_ratios


def run_p4a_comparative_benchmarks(
    train_steps: int = 240,
    batch_size: int = 32,
    eval_frames: int = 256,
    n_sc: int = DEFAULT_N,
    snrs_db: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    pa_conditions: Sequence[str] = ("strong", "mild", "clean"),
    profile: str = "equal4",
    seed: int = 2026,
    output_dir: Optional[Union[str, Path]] = "p4/results",
) -> Dict[str, Any]:
    """Run controlled Stage P4-A comparative benchmarks across 5 methods and save JSON/CSV."""
    torch.set_num_threads(4)

    emff, emff_scales = _train_comparative_model(
        lambda: EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=n_sc),
        steps=train_steps,
        batch_size=batch_size,
        n_sc=n_sc,
        profile=profile,
        seed=seed,
        has_gamma_eff=True,
        backbone_lr=8e-3,
        gamma_lr=2e-2,
    )
    local_conv, local_scales = _train_comparative_model(
        lambda: LocalEquivConv16QAM(width=14, kernel_size=5, depth=2, ffn_dim=12),
        steps=train_steps,
        batch_size=batch_size,
        n_sc=n_sc,
        profile=profile,
        seed=seed + 1,
        has_gamma_eff=True,
        backbone_lr=3e-3,
        gamma_lr=1e-2,
    )
    tiny_joint, tiny_scales = _train_comparative_model(
        lambda: TinyJoint16QAM(width=16, taps=8),
        steps=train_steps,
        batch_size=batch_size,
        n_sc=n_sc,
        profile=profile,
        seed=seed + 2,
        has_gamma_eff=False,
        backbone_lr=5e-3,
    )

    param_counts = {
        "EMFloatFormer": int(sum(p.numel() for p in emff.parameters())),
        "LocalEquivConv16QAM": int(sum(p.numel() for p in local_conv.parameters())),
        "TinyJoint16QAM": int(sum(p.numel() for p in tiny_joint.parameters())),
        "Bussgang-LMMSE": 0,
        "DD-PA-Cancel": 0,
    }
    pa_to_A: Dict[str, Optional[float]] = {"strong": 1.25, "mild": 2.25, "clean": None}
    loss_eval = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for pa_idx, pa_cond in enumerate(pa_conditions):
            A_val = pa_to_A[pa_cond]
            A_label = "inf" if A_val is None else f"{A_val:.2f}"
            for snr_idx, snr_db in enumerate(snrs_db):
                eval_seed = seed * 100 + pa_idx * 10 + snr_idx
                eval_gen = torch.Generator().manual_seed(eval_seed)
                frame = generate_frame(
                    batch_size=eval_frames,
                    n=n_sc,
                    profile=profile,
                    snr_db=float(snr_db),
                    pa_condition=pa_cond,
                    generator=eval_gen,
                )

                out_emff = emff(frame.yp, frame.yd, frame.noise_var)
                out_local = local_conv(frame.yp, frame.yd, frame.noise_var)
                out_tiny = tiny_joint(frame.yp, frame.yd, frame.noise_var)
                res_buss = bussgang_lmmse(frame.yp, frame.yd, frame.noise_var, A=A_val)
                h_dd, llrs_dd, _, _ = iterative_dd_pa_cancellation(
                    frame.yp, frame.yd, frame.noise_var, A=A_val, iterations=2
                )

                method_outputs = {
                    "EMFloatFormer": (out_emff["h_hat"], out_emff["llrs"]),
                    "LocalEquivConv16QAM": (out_local["h_hat"], out_local["llrs"]),
                    "TinyJoint16QAM": (out_tiny["h_hat"], out_tiny["llrs"]),
                    "Bussgang-LMMSE": (res_buss.h_hat, res_buss.llrs),
                    "DD-PA-Cancel": (h_dd, llrs_dd),
                }

                for method_name, (h_hat, llrs) in method_outputs.items():
                    preds = (llrs > 0).long()
                    bit_errors = int((preds != frame.bits).sum().item())
                    total_bits = int(frame.bits.numel())
                    ber = bit_errors / max(1, total_bits)
                    nmse = float(loss_eval.nmse_loss(h_hat, frame.h_true).item())
                    bce = float(loss_eval.bce_loss(llrs, frame.bits).item())
                    ece = compute_ece(llrs, frame.bits, n_bins=10)
                    h_norm = h_hat.abs().norm(dim=-1)
                    t_norm = frame.h_true.abs().norm(dim=-1).clamp(min=1e-12)
                    scale_ratio = float((h_norm / t_norm).mean().item())

                    rows.append({
                        "pa_condition": pa_cond,
                        "A": A_label,
                        "snr_db": float(snr_db),
                        "method": method_name,
                        "params": param_counts[method_name],
                        "bit_errors": bit_errors,
                        "total_bits": total_bits,
                        "ber": ber,
                        "nmse": nmse,
                        "bce": bce,
                        "ece": ece,
                        "scale_ratio": scale_ratio,
                        "eval_seed": eval_seed,
                    })

    summary = {
        "stage": "P4-A",
        "profile": profile,
        "n_sc": n_sc,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "eval_frames": eval_frames,
        "seed": seed,
        "num_threads": torch.get_num_threads(),
        "param_counts": param_counts,
        "train_final_scale_ratio": {
            "EMFloatFormer": float(sum(emff_scales[-15:]) / max(1, len(emff_scales[-15:]))),
            "LocalEquivConv16QAM": float(sum(local_scales[-15:]) / max(1, len(local_scales[-15:]))),
            "TinyJoint16QAM": float(sum(tiny_scales[-15:]) / max(1, len(tiny_scales[-15:]))),
        },
        "rows": rows,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / "p4a_comparative_benchmark.json"
        csv_path = out_path / "p4a_comparative_benchmark.csv"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return summary
