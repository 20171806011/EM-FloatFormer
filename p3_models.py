#!/usr/bin/env python3
"""P3-A: small common-phase-equivariant delay-domain pilot estimator.

Only synthetic passive civilian QPSK/OFDM. No transmit or adversarial logic.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from p2_models import ResidualBase


class EquivDelayConv(ResidualBase):
    """Real invariant delay-domain gate on complex pilot IFFT taps.

    For any scalar exp(1j*phi), amplitudes and noise stay invariant, so
    channel(exp(1j*phi) * yp) = exp(1j*phi) * channel(yp).
    No true H or data symbol enters channel().
    """
    def __init__(self, nsc=32, width=12):
        super().__init__()
        self.nsc = nsc
        self.stem = nn.Conv1d(6, width, 1)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv1d(width, width, 3, padding=1, padding_mode='circular')
        self.out = nn.Conv1d(width, 1, 1)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def channel(self, yp, yd, nv):
        del yd   # The model MUST NOT use the unknown data symbols as pilots.
        n = yp.shape[-1]
        if n != self.nsc:
            raise ValueError(f'expected {self.nsc} subcarriers, got {n}')
        t = torch.fft.ifft(yp, dim=-1)
        power = t.abs().square()
        mean = power.mean(dim=-1, keepdim=True)
        noise_t = nv / n
        k = torch.arange(n, device=yp.device, dtype=power.dtype)[None, :]
        phase = 2 * math.pi * k / n
        f = torch.stack((
            torch.log1p(power / (noise_t + 1e-7)),
            torch.log1p(power / (mean + 1e-7)),
            torch.log1p(mean / (noise_t + 1e-7)).expand_as(power),
            nv.log10().expand_as(power),
            phase.cos().expand_as(power),
            phase.sin().expand_as(power),
        ), dim=1)
        x = F.silu(self.stem(f))
        x = x + 0.5 * F.silu(self.conv1(x))
        x = x + 0.5 * F.silu(self.conv2(x))
        # Pure real [0,1] shrinkage ensures exact U(1) structure over reals.
        gain = torch.sigmoid(self.out(x)).squeeze(1)
        return torch.fft.fft(t * gain, dim=-1)


def model_factory_p3(name):
    from p2_models import model_factory
    if name == 'p3_equiv_delay':
        return EquivDelayConv(nsc=32)
    if name in ('res_unfold','res_equiv_local'):
        return model_factory(name)
    raise ValueError(f'unknown P3 model: {name}')


if __name__ == '__main__':
    m = EquivDelayConv()
    print('P3-A parameters:', sum(p.numel() for p in m.parameters()))
