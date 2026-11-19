"""C*-equivariant 2D separable Axial Attention vs matched 3-tap LocalConv evidence-routing block."""
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn.functional import pad


REAL = torch.float64
COMPLEX = torch.complex128


def routing_features(h_m, h_bar, y_obs, x_m, pred_m, y_bar, res_obs, anchor_m, path_fields):
    """Build C_in=10 strictly 1-homogeneous complex feature channels [B*M, 1, T, N, 10]."""
    disagreement_h = h_m - h_bar
    disagreement_y = pred_m - y_bar
    base = torch.stack(
        (
            h_m,
            y_obs,
            h_m * x_m[:, None],
            res_obs,
            disagreement_h,
            disagreement_y,
            anchor_m,
        ),
        dim=-1,
    )
    return torch.cat((base, path_fields), dim=-1)


def _matrix(rows, cols):
    return nn.Parameter(torch.randn(rows, cols, dtype=COMPLEX) / math.sqrt(rows))


@lru_cache(maxsize=32)
def _phase_positions(size, width):
    exponent = torch.arange(width - 1, -1, -1, dtype=REAL) / width
    rates = torch.pow(10000., -exponent)
    return torch.arange(size, dtype=REAL)[:, None] * rates


@lru_cache(maxsize=32)
def _neighbors(size, circular):
    idx = torch.arange(size)
    if circular:
        return (idx - 1) % size, (idx + 1) % size
    return (idx - 1).clamp_min(0), (idx + 1).clamp_max(size - 1)


class EvidenceRoutingBlock(nn.Module):
    """2D separable (Freq -> Time) Hermitian attention or parameter-matched 3-tap local mixer."""

    def __init__(self, variant="axial", width=8):
        super().__init__()
        self.width = width
        if variant not in ("axial", "local"):
            raise ValueError(f"Unknown routing variant: {variant}")
        self.local = variant == "local"
        self.embed = _matrix(10, width)
        self.project = _matrix(width, 3 * width)
        self.kernel = nn.Parameter(torch.zeros(3, width, dtype=REAL))
        self.mix = _matrix(2 * width, width)
        self.read = _matrix(width, 5)  # 3 physical residuals + 2 routing readout channels

    def _global_axis(self, q, k, v, axis, gains):
        angles = _phase_positions(q.shape[-2], self.width).to(device=q.device) * gains[1]
        rope = torch.polar(torch.ones_like(angles), angles)
        qr, kr = q * rope, k * rope
        gram = (qr * gains[0]) @ kr.mH
        strength = gram.abs()
        weights = (strength / math.sqrt(self.width)).softmax(-1)
        v_pair = torch.view_as_real((v * gains[2]).resolve_conj()).flatten(-2)
        flat = weights @ v_pair
        mixed = torch.view_as_complex(flat.unflatten(-1, (self.width, 2)))
        return mixed, weights

    def _local_axis(self, v, axis, gains):
        left, right = _neighbors(v.shape[-2], axis == 3)
        left = left.to(device=v.device)
        right = right.to(device=v.device)
        mixed = (
            .25 * gains[0] * v.index_select(-2, left)
            + .5 * gains[1] * v
            + .25 * gains[2] * v.index_select(-2, right)
        )
        return mixed, None

    def _mix_axis(self, state, rays, axis, gains, return_attn=False):
        q, k, v = (state @ self.project).chunk(3, dim=-1)
        if rays is not None:
            physical = math.sqrt(self.width) * rays
            q, k = physical + .08 * q, physical + .08 * k
            v = rays + v
        qm = q.movedim(axis, -2)
        km = k.movedim(axis, -2)
        vm = v.movedim(axis, -2)
        if self.local:
            mixed, attn = self._local_axis(vm, axis, gains)
        else:
            mixed, attn = self._global_axis(qm, km, vm, axis, gains)
        return mixed.movedim(-2, axis), q, k, attn if return_attn else None

    def forward(self, features, return_attn=False):
        power = features.abs().square().mean(-1, keepdim=True)
        scale = torch.where(power > 0, power, 1.).sqrt()
        norm_feat = features / scale
        z = norm_feat @ self.embed

        rays = pad(norm_feat[..., 7:10], (0, self.width - 3))
        ray_e = rays.abs().square().sum(-1, keepdim=True)
        rays = rays / torch.where(ray_e > 0, ray_e, 1.).sqrt()

        gains = 1. + self.kernel
        # Sequential 2D axial composition: Freq (axis=3) then Time (axis=2)
        mix_f, qf, kf, attn_f = self._mix_axis(z, rays, 3, gains, return_attn=return_attn)
        z_ft = z + .5 * mix_f
        mix_t, qt, kt, attn_t = self._mix_axis(z_ft, None, 2, gains, return_attn=return_attn)
        mixed = .5 * (mix_f + mix_t)

        latent = torch.cat((z + mixed, qt * kt.abs().sigmoid() + .5 * kt), dim=-1)
        hidden = z + .5 * mixed + .5 * ((latent * latent.abs().sigmoid()) @ self.mix)
        out = (hidden @ self.read) * scale  # B*M, 1, T, N, 5
        phys_core = out[..., :3]
        spatial_zero = torch.zeros_like(phys_core[..., :1])
        phys_fields = torch.cat((phys_core, spatial_zero), dim=-1)  # B*M, 1, T, N, 4
        route_fields = out[..., 3:5]  # B*M, 1, T, N, 2
        if return_attn:
            return phys_fields, route_fields, {"freq": attn_f, "time": attn_t}
        return phys_fields, route_fields
