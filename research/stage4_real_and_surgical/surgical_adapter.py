"""Unified W1+W2+W3+W4 Mode-I (Stop-Grad L10) Surgical Host Adapter and anti-forgetting suite.

All surgical perturbations attach via non-mutating PyTorch hooks so the frozen host
weights (`host_sha256`) stay bitwise immutable and `g_EM == 0` bypasses every EM branch
prior to floating-point addition (`PureKL = 0.0`, `max_logit_diff = 0.0`).
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint

QWEN4B_FA_LAYERS = (11, 15, 19, 23, 27)  # 0-based (1-based: 12, 16, 20, 24, 28)
QWEN4B_DN_LAYERS = (10, 14, 18, 22)      # 0-based (1-based: 11, 15, 19, 23)
QWEN4B_STOP_LAYER = 9                    # 0-based (1-based: Layer 10; 1..10 stopped in Mode I)
TIER_P1_MAX_PARAMS = 2_184_964

HELMERT_4X3 = torch.tensor(
    [
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
        [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
        [0.0, -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
        [0.0, 0.0, -3.0 / math.sqrt(12.0)],
    ],
    dtype=torch.float32,
)


def apply_j_d(x: torch.Tensor) -> torch.Tensor:
    """Apply canonical complex structure J_d = I_m kron [[0, -1], [1, 0]] on last dim (even d=2m)."""
    d = x.shape[-1]
    assert d % 2 == 0, f"Expected even dimension for J_d, got {d}"
    pairs = x.view(*x.shape[:-1], d // 2, 2)
    rot = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
    return rot.reshape_as(x)


def compute_complex_nullspace_projector(
    C_gen: torch.Tensor,
    k_c: int,
) -> Dict[str, torch.Tensor]:
    """Compute U(m)-symmetrized C_gen^C = 0.5*(C_gen + J_d^T C_gen J_d) & Hermitian nullspace Pi_perp.

    Returns `U_c` `(m, k_c)` complex64, `Pi_perp` `(m, m)` complex64, `P_perp_real` `(d, d)` float32,
    and `eigenvalues` `(m,)` float32 (descending).
    """
    c_f64 = 0.5 * (C_gen.to(torch.float64) + C_gen.to(torch.float64).transpose(-1, -2))
    d = c_f64.shape[-1]
    assert c_f64.shape == (d, d) and d % 2 == 0
    m = d // 2
    if not (0 <= k_c < m):
        raise ValueError(f"k_c={k_c} must satisfy 0 <= k_c < m={m}")

    # Exact J_d^T C J_d: left-multiply rows by J_d^T (-J_d), right-multiply cols by J_d
    # Since apply_j_d(v) = v @ J_d^T, we use J_d * x on columns and rows cleanly:
    j_mat = torch.zeros((d, d), dtype=torch.float64, device=c_f64.device)
    idx = torch.arange(m, device=c_f64.device)
    j_mat[2 * idx, 2 * idx + 1] = -1.0
    j_mat[2 * idx + 1, 2 * idx] = 1.0
    c_sym = 0.5 * (c_f64 + j_mat.T @ c_f64 @ j_mat)

    a_blk = c_sym[0::2, 0::2]
    b_blk = c_sym[1::2, 0::2]
    h_complex = torch.complex(a_blk, b_blk)
    h_complex = 0.5 * (h_complex + h_complex.conj().T)

    evals, evecs = torch.linalg.eigh(h_complex)
    # Sort descending
    order = torch.argsort(evals, descending=True)
    evals = evals[order].real
    evecs = evecs[:, order]
    u_c = evecs[:, :k_c].to(torch.complex64).contiguous()

    eye_m = torch.eye(m, dtype=torch.complex64, device=c_f64.device)
    pi_perp = eye_m - u_c @ u_c.conj().T

    p_real = torch.zeros((d, d), dtype=torch.float32, device=c_f64.device)
    p_real[0::2, 0::2] = pi_perp.real
    p_real[1::2, 0::2] = pi_perp.imag
    p_real[0::2, 1::2] = -pi_perp.imag
    p_real[1::2, 1::2] = pi_perp.real

    return {
        "U_c": u_c,
        "Pi_perp": pi_perp,
        "P_perp_real": p_real,
        "eigenvalues": evals.to(torch.float32),
        "C_sym": c_sym.to(torch.float32),
    }


def apply_complex_nullspace_fast(x: torch.Tensor, u_c: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply P_{perp, C}^C x in O(d * k_c) time without materializing a d x d matrix."""
    if u_c is None or u_c.numel() == 0:
        return x
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    z = torch.complex(xf[..., 0::2], xf[..., 1::2])
    uc = u_c.to(device=z.device, dtype=torch.complex64)
    # z_proj = z - (z @ uc.conj()) @ uc.T (for Hermitian P = I - uc uc^*)
    coeffs = z @ uc.conj()
    z_perp = z - coeffs @ uc.T
    out = torch.empty_like(xf)
    out[..., 0::2] = z_perp.real
    out[..., 1::2] = z_perp.imag
    return out.to(orig_dtype)


class FP32ComplexNullspaceCayley(nn.Module):
    """Rank-r_c FP32 Complex-Symmetrized Nullspace Cayley adapter (W2 Theorem 7).

    Always computes in FP32 (`torch.complex64`) and initializes V=0 -> Q=I exactly at step 0.
    """

    def __init__(
        self,
        dim: int,
        r_c: int = 4,
        u_c: Optional[torch.Tensor] = None,
        init_std: float = 0.02,
    ):
        super().__init__()
        assert dim % 2 == 0, f"Expected even dim, got {dim}"
        self.dim = int(dim)
        self.m = self.dim // 2
        self.r_c = int(r_c)

        # Master parameters strictly FP32 (never BF16 / NF4)
        self.u_re = nn.Parameter(torch.randn(self.m, self.r_c, dtype=torch.float32) * init_std)
        self.u_im = nn.Parameter(torch.randn(self.m, self.r_c, dtype=torch.float32) * init_std)
        self.v_re = nn.Parameter(torch.zeros(self.m, self.r_c, dtype=torch.float32))
        self.v_im = nn.Parameter(torch.zeros(self.m, self.r_c, dtype=torch.float32))

        if u_c is not None and u_c.numel() > 0:
            self.register_buffer("u_c", u_c.to(torch.complex64).clone())
        else:
            self.u_c = None

    def _projected_uv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.complex(self.u_re.float(), self.u_im.float())
        v = torch.complex(self.v_re.float(), self.v_im.float())
        if self.u_c is not None:
            uc = self.u_c.to(device=u.device, dtype=torch.complex64)
            u = u - uc @ (uc.conj().T @ u)
            v = v - uc @ (uc.conj().T @ v)
        return u, v

    def materialize_unitary(self) -> torch.Tensor:
        """Materialize full complex m x m unitary matrix Q_C^perp (for small dims/tests)."""
        u_perp, v_perp = self._projected_uv()
        a_c = torch.cat((u_perp, -v_perp), dim=1)  # (m, 2 r_c)
        b_c = torch.cat((v_perp, u_perp), dim=1)   # (m, 2 r_c)
        eye_2r = torch.eye(2 * self.r_c, dtype=torch.complex64, device=u_perp.device)
        core = eye_2r + 0.5 * (b_c.conj().T @ a_c)
        solved = torch.linalg.solve(core, b_c.conj().T)
        return torch.eye(self.m, dtype=torch.complex64, device=u_perp.device) - a_c @ solved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Q^perp x strictly via O(m r_c + r_c^3) FP32 Woodbury solve."""
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        z = torch.complex(xf[..., 0::2], xf[..., 1::2])  # (..., m)
        orig_lead = z.shape[:-1]
        z2d = z.reshape(-1, self.m)  # (N, m)

        u_perp, v_perp = self._projected_uv()
        a_c = torch.cat((u_perp, -v_perp), dim=1)  # (m, 2r_c)
        b_c = torch.cat((v_perp, u_perp), dim=1)   # (m, 2r_c)
        eye_2r = torch.eye(2 * self.r_c, dtype=torch.complex64, device=x.device)
        core = eye_2r + 0.5 * (b_c.conj().T @ a_c)  # (2r_c, 2r_c)
        # Note right-action on row vectors: (B^* z)^T = z @ B.conj() -> solve -> @ A^T
        rhs = (z2d @ b_c.conj()).T  # (2r_c, N)
        w = torch.linalg.solve(core, rhs).T  # (N, 2r_c)
        z_out = z2d - w @ a_c.T
        z_out = z_out.reshape(*orig_lead, self.m)

        out = torch.empty_like(xf)
        out[..., 0::2] = z_out.real
        out[..., 1::2] = z_out.imag
        return out.to(orig_dtype)


class NullspaceLoRADelta(nn.Module):
    """Low-rank rank-r LoRA delta with optional P_{perp, C}^C input filter and BF16 support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        u_c: Optional[torch.Tensor] = None,
        out_slice: Optional[Tuple[int, int]] = None,
        interleaved_half_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)
        self.scale = float(alpha) / float(max(1, r))
        if out_slice is not None and interleaved_half_head_dim is not None:
            raise ValueError("Specify at most one of out_slice or interleaved_half_head_dim.")
        self.out_slice = out_slice
        self.interleaved_half_head_dim = int(interleaved_half_head_dim) if interleaved_half_head_dim is not None else None
        if self.interleaved_half_head_dim is not None:
            assert self.out_features % (2 * self.interleaved_half_head_dim) == 0, (
                f"out_features={self.out_features} not divisible by 2*head_dim={2 * self.interleaved_half_head_dim}"
            )
            eff_out = self.out_features // 2
        elif out_slice is not None:
            eff_out = out_slice[1] - out_slice[0]
        else:
            eff_out = self.out_features
        self.lora_a = nn.Parameter(torch.randn(self.r, self.in_features, dtype=torch.float32) * (1.0 / math.sqrt(self.in_features)))
        self.lora_b = nn.Parameter(torch.zeros(eff_out, self.r, dtype=torch.float32))
        if u_c is not None and u_c.numel() > 0:
            self.register_buffer("u_c", u_c.to(torch.complex64).clone())
        else:
            self.u_c = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = apply_complex_nullspace_fast(x, self.u_c).to(torch.float32)
        h = F.linear(xf, self.lora_a)
        delta = F.linear(h, self.lora_b) * self.scale
        return delta.to(x.dtype)


class SchurIrrepReadout(nn.Module):
    """Canonical frame aligned (B_align = M_cand^+ F_real) low-rank readout adapter (W2 Theorem 5/8)."""

    def __init__(
        self,
        hidden_size: int,
        r: int = 11,
        b_align: Optional[torch.Tensor] = None,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.r = int(r)
        self.proj_in = nn.Parameter(torch.randn(self.r, self.hidden_size, dtype=torch.float32) * (init_std / math.sqrt(self.hidden_size)))
        self.proj_coord = nn.Parameter(torch.zeros(3, self.r, dtype=torch.float32))
        if b_align is None:
            # Orthographic fallback until live M_cand^+ F_real calibration
            frame = torch.zeros((self.hidden_size, 3), dtype=torch.float32)
            frame[:3, :3] = torch.eye(3, dtype=torch.float32)
            self.register_buffer("b_align", frame)
        else:
            self.register_buffer("b_align", b_align.to(torch.float32).clone())

    def canonical_coords(self, hidden: torch.Tensor) -> torch.Tensor:
        hf = hidden[:, -1, :].to(torch.float32)
        return F.linear(F.gelu(F.linear(hf, self.proj_in)), self.proj_coord)  # (B, 3)

    def forward_at_positions(self, hidden: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        out = torch.zeros_like(hidden)
        seq_l = hidden.shape[1]
        pos_idx = sorted({p if p >= 0 else seq_l + p for p in positions if 0 <= (p if p >= 0 else seq_l + p) < seq_l})
        if not pos_idx:
            return out
        hf = hidden[:, pos_idx, :].to(torch.float32)
        coords = F.linear(F.gelu(F.linear(hf, self.proj_in)), self.proj_coord)  # (B, K, 3)
        delta = (coords @ self.b_align.T).to(hidden.dtype)
        out[:, pos_idx, :] = delta
        return out

    def forward_last_token_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward_at_positions(hidden, [-1])


def calibrate_b_align_from_host(host) -> torch.Tensor:
    """Compute B_align = pinv(M_cand) @ HELMERT_4X3 with unit-norm columns from host lm_head."""
    out_emb = host.lm.get_output_embeddings()
    if out_emb is None:
        out_emb = host.lm.get_input_embeddings()
    w = out_emb.weight.detach()
    if getattr(host, "tokenizer", None) is None:
        cand_w = w[:4, :].float().cpu()
    else:
        digit_ids = [int(host.tokenizer.encode(d_str, add_special_tokens=False)[-1]) for d_str in ("0", "1", "2", "3")]
        cand_w = w[digit_ids, :].float().cpu()
    m_cand = cand_w - cand_w.mean(dim=0, keepdim=True)  # (4, D)
    b_align = torch.linalg.pinv(m_cand) @ HELMERT_4X3  # (D, 3)
    norms = b_align.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return (b_align / norms).contiguous()


class TwoTierModalityGate:
    """Out-of-band binary g_EM + causal token-position write mask m_EM(t) (W3 Theorem 4.3)."""

    def __init__(self):
        self.g_em: int = 0
        self.m_acc_mask: Optional[torch.Tensor] = None

    def set_state(self, g_em: int, m_acc_mask: Optional[torch.Tensor] = None) -> None:
        self.g_em = int(bool(g_em))
        self.m_acc_mask = m_acc_mask


class ModeIStopGradHook:
    """Detaches layer residual output at `stop_layer` when Mode I stop-grad is enabled (W4 Thm 1 / W3 Thm 8)."""

    def __init__(self, stop_layer: int, gate: TwoTierModalityGate, mode: str = "mode1_stopgrad_l10"):
        self.stop_layer = int(stop_layer)
        self.gate = gate
        self.mode = mode

    def __call__(self, module, args, output):
        if self.gate.g_em == 0 or self.mode != "mode1_stopgrad_l10":
            return output
        if isinstance(output, tuple):
            return (output[0].detach(), *output[1:])
        return output.detach()


def _get_decoder_layers(lm) -> nn.ModuleList:
    if hasattr(lm, "model") and hasattr(lm.model, "language_model") and hasattr(lm.model.language_model, "layers"):
        return lm.model.language_model.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError(f"Could not locate decoder layers on model type {type(lm)}")


class SurgicalHostAdapter(nn.Module):
    """Non-mutating hook-attached W1+W2+W3+W4 adapter suite for Qwen3.5 or mock Qwen2 hosts."""

    def __init__(
        self,
        lm: nn.Module,
        fa_layers: Sequence[int] = QWEN4B_FA_LAYERS,
        dn_layers: Sequence[int] = QWEN4B_DN_LAYERS,
        stop_layer: int = QWEN4B_STOP_LAYER,
        mode: str = "mode1_stopgrad_l10",
        r_c_cayley: int = 4,
        r_lora: int = 8,
        r_schur: int = 11,
        u_c_by_layer: Optional[Dict[int, torch.Tensor]] = None,
        b_align: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mode = mode
        self.stop_layer = int(stop_layer)
        self.gate = TwoTierModalityGate()
        self.handles: List[object] = []

        layers = _get_decoder_layers(lm)
        n_layers = len(layers)
        self.fa_cayleys = nn.ModuleDict()
        self.fa_loras = nn.ModuleDict()
        self.dn_cayleys = nn.ModuleDict()
        self.dn_loras = nn.ModuleDict()
        self.dn_gate_loras = nn.ModuleDict()

        u_map = u_c_by_layer or {}
        d_model = int(getattr(getattr(lm.config, "text_config", lm.config), "hidden_size", 2560))
        self.schur_readout = SchurIrrepReadout(d_model, r=r_schur, b_align=b_align)

        for idx in fa_layers:
            if idx < 0 or idx >= n_layers:
                continue
            blk = layers[idx]
            attn = getattr(blk, "self_attn", None)
            if attn is None:
                continue
            u_c_d = u_map.get(idx)
            # o_proj input Cayley
            o_in = attn.o_proj.in_features
            u_c_o = u_c_d if (u_c_d is not None and u_c_d.shape[0] * 2 == o_in) else None
            self.fa_cayleys[str(idx)] = FP32ComplexNullspaceCayley(o_in, r_c=r_c_cayley, u_c=u_c_o)
            # k_proj & v_proj with Tier-1 token-position mask m_EM(t)
            for pname, masked in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
                proj = getattr(attn, pname)
                u_c_p = u_c_d if (u_c_d is not None and u_c_d.shape[0] * 2 == proj.in_features) else None
                out_slice = None
                interleaved_half_head_dim = None
                if pname == "q_proj" and hasattr(attn, "num_key_value_groups"):
                    # Qwen3.5 gated q_proj outputs per-head interleaved [Q_h, Gate_h] of width 2 * num_heads * head_dim
                    if proj.out_features == 2 * o_in:
                        if hasattr(attn, "head_dim"):
                            interleaved_half_head_dim = int(attn.head_dim)
                        else:
                            out_slice = (0, o_in)
                self.fa_loras[f"{idx}_{pname}"] = NullspaceLoRADelta(
                    proj.in_features,
                    proj.out_features,
                    r=r_lora,
                    u_c=u_c_p,
                    out_slice=out_slice,
                    interleaved_half_head_dim=interleaved_half_head_dim,
                )

        for idx in dn_layers:
            if idx < 0 or idx >= n_layers:
                continue
            blk = layers[idx]
            dn = getattr(blk, "linear_attn", None)
            if dn is None:
                continue
            u_c_d = u_map.get(idx)
            self.dn_cayleys[str(idx)] = FP32ComplexNullspaceCayley(dn.out_proj.in_features, r_c=r_c_cayley, u_c=None)
            # in_proj_qkv: adapt KV slice [key_dim : 2*key_dim+value_dim] with m_EM(t), leaving Q slice clean
            kv_start = dn.key_dim
            kv_end = 2 * dn.key_dim + dn.value_dim
            u_c_in = u_c_d if (u_c_d is not None and u_c_d.shape[0] * 2 == dn.in_proj_qkv.in_features) else None
            self.dn_loras[f"{idx}_in_proj_kv"] = NullspaceLoRADelta(
                dn.in_proj_qkv.in_features,
                dn.in_proj_qkv.out_features,
                r=r_lora,
                u_c=u_c_in,
                out_slice=(kv_start, kv_end),
            )
            # W1 slow-head in_proj_a / in_proj_b low-rank adaptation (full rank <= 4 on gates)
            self.dn_gate_loras[f"{idx}_in_proj_a"] = NullspaceLoRADelta(
                dn.in_proj_a.in_features, dn.in_proj_a.out_features, r=4, u_c=u_c_in
            )
            self.dn_gate_loras[f"{idx}_in_proj_b"] = NullspaceLoRADelta(
                dn.in_proj_b.in_features, dn.in_proj_b.out_features, r=4, u_c=u_c_in
            )

        self._attach_hooks(lm)

    def _masked_write_tensor(self, x: torch.Tensor) -> torch.Tensor:
        m = self.gate.m_acc_mask
        if m is None:
            return x
        if x.shape[0] != m.shape[0]:
            if x.shape[0] % m.shape[0] == 0:
                m = m.repeat_interleave(x.shape[0] // m.shape[0], dim=0)
            else:
                return torch.zeros_like(x)
        seq_l = x.shape[1]
        m_len = m.shape[1]
        if seq_l == m_len:
            return x * m.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        if seq_l < m_len:
            # KV-cached candidate continuation pass: zero EM write mask on all continuation tokens
            return torch.zeros_like(x)
        # Uncached prefix + candidate pass: pad candidate suffix with 0
        pad_len = seq_l - m_len
        zeros = torch.zeros((x.shape[0], pad_len), device=x.device, dtype=x.dtype)
        full_m = torch.cat((m.to(device=x.device, dtype=x.dtype), zeros), dim=1)
        return x * full_m.unsqueeze(-1)

    def _attach_hooks(self, lm: nn.Module) -> None:
        layers = _get_decoder_layers(lm)
        if 0 <= self.stop_layer < len(layers):
            stop_hook = ModeIStopGradHook(self.stop_layer, self.gate, mode=self.mode)
            self.handles.append(layers[self.stop_layer].register_forward_hook(stop_hook))

        for k_str, cay in self.fa_cayleys.items():
            idx = int(k_str)
            attn = layers[idx].self_attn

            def make_pre_cayley(mod):
                def _pre(module, args):
                    if self.gate.g_em == 0:
                        return args
                    x = args[0]
                    return (mod(x), *args[1:])
                return _pre

            self.handles.append(attn.o_proj.register_forward_pre_hook(make_pre_cayley(cay)))

            for pname, masked in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
                lora = self.fa_loras[f"{idx}_{pname}"]
                proj = getattr(attn, pname)

                def make_post_lora(mod, apply_m_acc: bool):
                    def _post(module, args, out):
                        if self.gate.g_em == 0:
                            return out
                        x = args[0]
                        if apply_m_acc:
                            x = self._masked_write_tensor(x)
                        delta = mod(x)
                        hd = getattr(mod, "interleaved_half_head_dim", None)
                        if hd is not None:
                            res = out.clone().view(*out.shape[:-1], -1, 2 * hd)
                            res[..., :hd] = res[..., :hd] + delta.view(*out.shape[:-1], -1, hd)
                            return res.reshape_as(out)
                        if mod.out_slice is not None:
                            s0, s1 = mod.out_slice
                            res = out.clone()
                            res[..., s0:s1] = res[..., s0:s1] + delta
                            return res
                        return out + delta
                    return _post

                self.handles.append(proj.register_forward_hook(make_post_lora(lora, masked)))

        for k_str, cay in self.dn_cayleys.items():
            idx = int(k_str)
            dn = layers[idx].linear_attn

            def make_dn_cay(mod):
                def _pre(module, args):
                    if self.gate.g_em == 0:
                        return args
                    return (mod(args[0]), *args[1:])
                return _pre

            self.handles.append(dn.out_proj.register_forward_pre_hook(make_dn_cay(cay)))

            kv_lora = self.dn_loras[f"{idx}_in_proj_kv"]

            def make_kv_post(mod):
                def _post(module, args, out):
                    if self.gate.g_em == 0:
                        return out
                    x_masked = self._masked_write_tensor(args[0])
                    delta = mod(x_masked)
                    s0, s1 = mod.out_slice
                    res = out.clone()
                    res[..., s0:s1] = res[..., s0:s1] + delta
                    return res
                return _post

            self.handles.append(dn.in_proj_qkv.register_forward_hook(make_kv_post(kv_lora)))

            for gname in ("in_proj_a", "in_proj_b"):
                glora = self.dn_gate_loras[f"{idx}_{gname}"]
                gproj = getattr(dn, gname)

                def make_gate_post(mod):
                    def _post(module, args, out):
                        if self.gate.g_em == 0:
                            return out
                        x_masked = self._masked_write_tensor(args[0])
                        return out + mod(x_masked)
                    return _post

                self.handles.append(gproj.register_forward_hook(make_gate_post(glora)))

        # Attach Schur-Irrep readout delta on last decoder layer output
        if len(layers) > 0:
            def _schur_post(module, args, output):
                if self.gate.g_em == 0:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                if self.gate.m_acc_mask is not None and h.shape[1] < self.gate.m_acc_mask.shape[1]:
                    positions = list(range(min(h.shape[1], 3)))
                else:
                    positions = [-1]
                delta = self.schur_readout.forward_at_positions(h, positions)
                if isinstance(output, tuple):
                    return (h + delta, *output[1:])
                return h + delta

            self.handles.append(layers[-1].register_forward_hook(_schur_post))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @contextmanager
    def activate(self, g_em: int = 1, m_acc_mask: Optional[torch.Tensor] = None):
        prev_g = self.gate.g_em
        prev_m = self.gate.m_acc_mask
        self.gate.set_state(g_em=g_em, m_acc_mask=m_acc_mask)
        try:
            yield
        finally:
            self.gate.set_state(g_em=prev_g, m_acc_mask=prev_m)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UnconstrainedLoRAAdapter(nn.Module):
    """Matched LEMMO Tier-P1 unconstrained LoRA baseline (no K-FAC, no Cayley, no Schur, no g_EM/m_EM gates)."""

    def __init__(
        self,
        lm: nn.Module,
        fa_layers: Sequence[int] = QWEN4B_FA_LAYERS,
        dn_layers: Sequence[int] = QWEN4B_DN_LAYERS,
        stop_layer: int = QWEN4B_STOP_LAYER,
        r_lora: int = 8,
    ):
        super().__init__()
        self.stop_layer = int(stop_layer)
        self.active: bool = True
        self.handles: List[object] = []

        layers = _get_decoder_layers(lm)
        n_layers = len(layers)
        self.fa_loras = nn.ModuleDict()
        self.dn_loras = nn.ModuleDict()

        for idx in fa_layers:
            if idx < 0 or idx >= n_layers:
                continue
            attn = getattr(layers[idx], "self_attn", None)
            if attn is None:
                continue
            o_in = attn.o_proj.in_features
            self.fa_loras[f"{idx}_o_proj"] = NullspaceLoRADelta(attn.o_proj.in_features, attn.o_proj.out_features, r=4)
            for pname in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, pname)
                out_slice = None
                interleaved_half = None
                if pname == "q_proj" and hasattr(attn, "num_key_value_groups") and proj.out_features == 2 * o_in:
                    if hasattr(attn, "head_dim"):
                        interleaved_half = int(attn.head_dim)
                    else:
                        out_slice = (0, o_in)
                self.fa_loras[f"{idx}_{pname}"] = NullspaceLoRADelta(
                    proj.in_features,
                    proj.out_features,
                    r=r_lora,
                    out_slice=out_slice,
                    interleaved_half_head_dim=interleaved_half,
                )

        for idx in dn_layers:
            if idx < 0 or idx >= n_layers:
                continue
            dn = getattr(layers[idx], "linear_attn", None)
            if dn is None:
                continue
            self.dn_loras[f"{idx}_out_proj"] = NullspaceLoRADelta(dn.out_proj.in_features, dn.out_proj.out_features, r=4)
            kv_start = dn.key_dim
            kv_end = 2 * dn.key_dim + dn.value_dim
            self.dn_loras[f"{idx}_in_proj_kv"] = NullspaceLoRADelta(
                dn.in_proj_qkv.in_features, dn.in_proj_qkv.out_features, r=r_lora, out_slice=(kv_start, kv_end)
            )
            self.dn_loras[f"{idx}_in_proj_a"] = NullspaceLoRADelta(dn.in_proj_a.in_features, dn.in_proj_a.out_features, r=4)
            self.dn_loras[f"{idx}_in_proj_b"] = NullspaceLoRADelta(dn.in_proj_b.in_features, dn.in_proj_b.out_features, r=4)

        self._attach_hooks(lm)

    def _attach_hooks(self, lm: nn.Module) -> None:
        layers = _get_decoder_layers(lm)
        if 0 <= self.stop_layer < len(layers):
            def _stop_hook(module, args, output):
                if not self.active:
                    return output
                if isinstance(output, tuple):
                    return (output[0].detach(), *output[1:])
                return output.detach()

            self.handles.append(layers[self.stop_layer].register_forward_hook(_stop_hook))

        def make_post(mod):
            def _post(module, args, out):
                if not self.active:
                    return out
                delta = mod(args[0])
                hd = getattr(mod, "interleaved_half_head_dim", None)
                if hd is not None:
                    res = out.clone().view(*out.shape[:-1], -1, 2 * hd)
                    res[..., :hd] = res[..., :hd] + delta.view(*out.shape[:-1], -1, hd)
                    return res.reshape_as(out)
                if mod.out_slice is not None:
                    s0, s1 = mod.out_slice
                    res = out.clone()
                    res[..., s0:s1] = res[..., s0:s1] + delta
                    return res
                return out + delta
            return _post

        for key, lmod in self.fa_loras.items():
            idx_s, pname = key.split("_", 1)
            proj = getattr(layers[int(idx_s)].self_attn, pname)
            self.handles.append(proj.register_forward_hook(make_post(lmod)))

        for key, lmod in self.dn_loras.items():
            idx_s, pname = key.split("_", 1)
            dn = layers[int(idx_s)].linear_attn
            proj = dn.in_proj_qkv if pname == "in_proj_kv" else getattr(dn, pname)
            self.handles.append(proj.register_forward_hook(make_post(lmod)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @contextmanager
    def activate(self, g_em: int = 1, m_acc_mask: Optional[torch.Tensor] = None):
        # Always active when attached (unconstrained LoRA has no g_EM or m_EM gates)
        del g_em, m_acc_mask
        yield

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def rao_blackwell_bayes_kl_cf_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    cf_swapped_logits: Optional[torch.Tensor] = None,
    lambda_cf: float = 0.25,
    lambda_norm: float = 0.05,
    margin: float = 0.25,
    mix_kl: Optional[torch.Tensor] = None,
    lambda_mix: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Composite W3 Bayes-KL + CF-Margin + norm calibration + ramped MixKL (`λ_orbit=0`)."""
    centered = logits - logits.mean(dim=-1, keepdim=True)
    log_q = F.log_softmax(centered, dim=-1)
    p = target_probs.clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    kl_loss = torch.sum(p * (torch.log(p) - log_q), dim=-1).mean()

    log_p = torch.log(p)
    centered_p = log_p - log_p.mean(dim=-1, keepdim=True)
    norm_loss = ((centered.norm(dim=-1) - centered_p.norm(dim=-1)) ** 2).mean()

    cf_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if cf_swapped_logits is not None:
        # Counterfactual slot swap must separate from factual log_q
        swap_log_q = F.log_softmax(cf_swapped_logits - cf_swapped_logits.mean(dim=-1, keepdim=True), dim=-1)
        sym_kl = 0.5 * torch.sum((log_q.exp() - swap_log_q.exp()) * (log_q - swap_log_q), dim=-1)
        cf_loss = F.relu(margin - sym_kl).mean()

    mix_loss = mix_kl if mix_kl is not None else torch.zeros((), device=logits.device, dtype=logits.dtype)
    total = kl_loss + lambda_norm * norm_loss + lambda_cf * cf_loss + lambda_mix * mix_loss
    return {"total": total, "kl": kl_loss, "norm": norm_loss, "cf": cf_loss, "mix": mix_loss}


def param_budget_audit(
    connector: Optional[nn.Module],
    adapter: Optional[nn.Module],
    frozen_lm: Optional[nn.Module] = None,
    tier_p1_cap: int = TIER_P1_MAX_PARAMS,
) -> dict:
    """Audit parameter counts against LEMMO Tier P1 (<= 2,184,964 on 4B) and host immutability."""
    c_params = sum(p.numel() for p in connector.parameters() if p.requires_grad) if connector is not None else 0
    a_params = adapter.param_count() if adapter is not None else 0
    total = c_params + a_params
    out = {
        "connector_params": int(c_params),
        "surgical_adapter_params": int(a_params),
        "cumulative_params": int(total),
        "tier_p1_cap": int(tier_p1_cap),
        "under_tier_p1_cap": bool(total <= tier_p1_cap),
    }
    if frozen_lm is not None:
        out["host_sha256"] = _compute_host_fingerprint(frozen_lm)
    return out
