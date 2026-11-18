"""Full-frame, charge-one axial operators on a CPU complex128 receive grid."""

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn.functional import pad


def full_frame_features(h, y, x, residual, anchor, pilot_mask, path_fields=None):
    """Build the observed full-frame fields, optionally including estimated rays.

    x is the pilot or current soft transmitted symbol [B,T,N]. A pilot
    indicator enters only as a real modulation of an observed field. The two
    amplitude-contrast channels retain constellation radius through RMS norm.
    """
    ref = x.conj()[:, None]
    contrast = (x.abs().square() - 1)[:, None]
    mask = pilot_mask.to(dtype=h.real.dtype)
    density = mask.mean()
    variance = density * (1 - density)
    indicator = (mask - density) / torch.where(variance > 0, variance, 1.).sqrt()
    matched = y * ref
    observed = torch.stack((h, matched, residual * ref, residual, anchor, h * ref,
                            contrast * h, indicator * h, contrast * matched), dim=-1)
    return observed if path_fields is None else torch.cat((observed, path_fields), dim=-1)


def _matrix(rows, cols):
    return nn.Parameter(torch.randn(rows, cols, dtype=torch.complex128) / math.sqrt(rows))


@lru_cache(maxsize=32)
def _phase_positions(size, width):
    # Order rotary bands from slow to fast across the finite burst.
    exponent = torch.arange(width - 1, -1, -1, dtype=torch.float64) / width
    rates = torch.pow(10000., -exponent)
    return torch.arange(size, dtype=torch.float64)[:, None] * rates


@lru_cache(maxsize=32)
def _neighbors(size, circular):
    idx = torch.arange(size)
    if circular:
        return (idx - 1) % size, (idx + 1) % size
    return (idx - 1).clamp_min(0), (idx + 1).clamp_max(size - 1)


def _hermitian_transport(transport):
    """Project the frame-averaged steered map to a DC-preserving PSD contraction."""
    average = transport.mean((1, 2))
    hermitian = (average + average.mH) / 2
    size = average.shape[-1]
    identity = torch.eye(size, dtype=average.dtype, device=average.device)
    dc = torch.ones_like(identity) / size
    centered = identity - dc
    # The maximum absolute row sum bounds the spectral radius of a Hermitian map.
    bound = hermitian.abs().sum(-1).amax(-1).clamp_min(1)[:, None, None]
    return dc + centered @ ((identity + hermitian / bound) / 2) @ centered


def _local_transport(gains, size):
    """Symmetrize the normalized three taps, using reflecting endpoint diagonals."""
    weights = gains.mean(-1).abs() * gains.new_tensor([.25, .5, .25])
    total = weights.sum()
    # Lazification bounds each neighbour weight by 1/4, so the minimum
    # eigenvalue is nonnegative even for a centre tap of zero.
    edge = (weights[0] + weights[2]) / (4 * torch.where(total > 0, total, 1.))
    diagonal = torch.ones(size - 1, device=gains.device, dtype=gains.dtype)
    neighbors = torch.diag(diagonal, 1) + torch.diag(diagonal, -1)
    return torch.eye(size, device=gains.device, dtype=gains.dtype) + edge * (
        neighbors - torch.diag(neighbors.sum(-1)))


@lru_cache(maxsize=16)
def _probe_coordinates(r, t, n):
    # A readout-only coordinate set matching the P5 Green probe at R=4,T=5.
    # Tokens on the remaining sites participate in every receiver forward.
    if t == 20:
        times = torch.tensor([0, 3, 6, 9, 10, 13, 16, 19])
    else:
        times = torch.arange(0, t, 2)
    return torch.cartesian_prod(torch.arange(r), times, torch.arange(2, n, 8))


def _probe_path(weights, shape):
    """Sample actual axis softmaxes and their two distinct joint summaries.

    The product holds the three measured matrices fixed and composes them
    in the historical F -> T -> R order. The parallel receiver does not
    apply this composition. Its direct branch transport magnitude is the
    mean of the three matrices with zeros outside each axis's support.
    Both readouts exclude pointwise updates and readout projections.
    """
    r, t, n = shape
    coords = _probe_coordinates(r, t, n)
    rx, row, tone = coords.T
    source, target = (None, slice(None)), (slice(None), None)
    ar = (weights[1][:, row[target], tone[target], rx[target], rx[source]]
          if r > 1 else weights[3].new_ones(weights[3].shape[0], len(coords), len(coords)))
    at = weights[2][:, rx[source], tone[target], row[target], row[source]]
    af = weights[3][:, rx[source], row[source], tone[target], tone[source]]
    same_r = rx[target] == rx[source]
    same_t = row[target] == row[source]
    same_f = tone[target] == tone[source]
    parallel = at * (same_r & same_f) + af * (same_r & same_t)
    if r > 1:
        parallel = parallel + ar * (same_t & same_f)
    parallel = parallel / (2 + int(r > 1))
    return coords, ar * at * af, {"space": ar, "time": at, "frequency": af,
                                  "parallel_joint": parallel}


class AxialOperatorBlock(nn.Module):
    """Parallel frequency, time and antenna mixing from the same input.

    Each branch reads the same projected Q/K/V before any axis mixes sites.
    Local mixing is a learnable channelwise three-tap convolution with
    circular frequency padding and endpoint replication in time/space.
    Both mixers have three real coefficients per hidden channel. In global
    mixing they control logit amplitude, RoPE frequency, and value gain.
    R=1 skips the spatial residual, leaving all weights used on F and T.
    """

    def __init__(self, variant="pure", width=8):
        super().__init__()
        self.width = width
        self.local = variant in {"local", "local_pure", "local_parity"}
        self.gauge = variant != "no_gauge"
        self.disable_spatial = False
        self.embed = _matrix(9, width)
        self.project = _matrix(width, 3 * width)
        self.kernel = nn.Parameter(torch.zeros(3, width, dtype=torch.float64))
        self.mix = _matrix(2 * width, width)
        self.read = _matrix(width, 3)

    def _global(self, q, k, v, axis, gains):
        if axis != 1:
            angles = _phase_positions(q.shape[-2], self.width) * gains[1]
            rope = torch.polar(torch.ones_like(angles), angles)
            q, k = q * rope, k * rope
        # Coherence magnitude measures the strength of a path match even
        # when a steering phase would make its real part change sign.
        gram = (q * gains[0]) @ k.mH
        strength = gram.abs()
        weights = (strength / math.sqrt(self.width)).softmax(-1)
        if axis == 1:
            # Steer each incoming complex value into the query's phase.
            # A null correlation has no identifiable phase; choose identity.
            safe = torch.where(strength > 0, strength, 1.)
            steering = torch.where(strength > 0, gram / safe, 1.)
            transport = weights * steering
            return transport @ (v * gains[2]), weights, transport
        v_pair = torch.view_as_real((v * gains[2]).resolve_conj()).flatten(-2)
        flat = weights @ v_pair
        mixed = torch.view_as_complex(flat.unflatten(-1, (self.width, 2)))
        return mixed, weights, None

    def _local(self, v, axis, gains):
        left, right = _neighbors(v.shape[-2], axis == 3)
        mixed = (.25 * gains[0] * v.index_select(-2, left)
                 + .5 * gains[1] * v + .25 * gains[2] * v.index_select(-2, right))
        return mixed, None

    def forward(self, features, probe=False, return_spatial=False):
        observed = features[..., :9]
        if self.gauge:
            power = observed.abs().square().mean(-1, keepdim=True)
            scale = torch.where(power > 0, power, 1.).sqrt()
        else:
            scale = 1.
        z = (observed / scale) @ self.embed
        q, k, v = (z @ self.project).chunk(3, dim=-1)
        if features.shape[-1] > 9:
            # The same coordinate denotes the same estimated physical path at
            # both ends of an edge. Their Hermitian product therefore retains
            # per-ray steering and delay. Normalization removes gauge magnitude.
            rays = features[..., 9:]
            if rays.shape[-1] > self.width:
                raise ValueError("operator width must cover the physical rays")
            if self.gauge:
                energy = rays.abs().square().sum(-1, keepdim=True)
                rays = rays / torch.where(energy > 0, energy, 1.).sqrt()
            rays = pad(rays, (0, self.width - rays.shape[-1]))
            physical = math.sqrt(self.width) * rays
            q, k = physical + .08 * q, physical + .08 * k
            v = rays + v
        weights = {}
        gains = 1 + self.kernel
        branches = {}
        spatial_operator = None
        active = [axis for axis in (1, 2, 3) if z.shape[axis] > 1]
        for axis in active:
            if axis == 1 and self.disable_spatial:
                continue
            value = v.movedim(axis, -2)
            if self.local:
                mixed, attn = self._local(value, axis, gains)
                if return_spatial and axis == 1:
                    spatial_operator = _local_transport(gains, value.shape[-2]).to(value.dtype)
            else:
                mixed, attn, transport = self._global(q.movedim(axis, -2), k.movedim(axis, -2),
                                                      value, axis, gains)
                if return_spatial and axis == 1:
                    spatial_operator = _hermitian_transport(transport)
            branches[axis] = mixed.movedim(-2, axis)
            if probe and attn is not None:
                weights[axis] = attn
        mixed = sum(branches.values()) / max(1, len(active))
        latent = torch.cat((z + mixed, q * k.abs().sigmoid() + .5 * k), dim=-1)
        hidden = z + .5 * mixed + .5 * ((latent * latent.abs().sigmoid()) @ self.mix)
        core = hidden @ self.read
        # One shared read column sees active time/frequency gradients even
        # when R=1. Only this output exclusively reads the antenna branch.
        spatial = (branches[1] @ self.read[:, :1]
                   if 1 in branches else torch.zeros_like(core[..., :1]))
        fields = torch.cat((core, spatial), dim=-1) * scale
        record = (_probe_path(weights, z.shape[1:4])
                  if probe and not self.local and not self.disable_spatial else None)
        return (fields, record, spatial_operator) if return_spatial else (fields, record)
