#!/usr/bin/env python3
"""Reproducible, receive-only, synthetic civilian OFDM benchmark.

This prototype is intentionally limited to normal Gaussian thermal noise and
fixed multipath channels. It has no radar, emitter-control, jamming or
adversarial-channel capabilities.

OFDM model: one known all-subcarrier pilot symbol, one QPSK data symbol,
static channel within the pair, and a cyclic prefix long enough to eliminate
intersymbol interference. Frequency-domain multiplication is the exact
FFT-domain equivalent of cyclic convolution under these assumptions.
"""
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def complex_noise(shape, noise_var, device):
    scale = (noise_var / 2).sqrt()
    return torch.complex(
        torch.randn(shape, device=device) * scale,
        torch.randn(shape, device=device) * scale,
    )


@dataclass
class Batch:
    yp: torch.Tensor  # observed pilot subcarriers
    yd: torch.Tensor  # observed data subcarriers
    h: torch.Tensor   # true channel, evaluation only
    bits: torch.Tensor  # (batch, subcarriers, 2)
    noise_var: torch.Tensor  # (batch,1)


def make_batch(batch_size, nsc, snr_lo, snr_hi, device, profile=(0.72, 0.20, 0.08)):
    """Independent synthetic random QPSK frames; no files or SDR required."""
    taps = len(profile)
    powers = torch.tensor(profile, device=device, dtype=torch.float32)
    powers = powers / powers.sum()
    hr = torch.randn(batch_size, taps, device=device)
    hi = torch.randn(batch_size, taps, device=device)
    h_time = torch.complex(hr, hi) * (powers / 2).sqrt()[None, :]
    h = torch.fft.fft(h_time, n=nsc, dim=-1)

    bits = torch.randint(0, 2, (batch_size, nsc, 2), device=device)
    symbol = torch.complex(2.0 * bits[..., 0].float() - 1.0,
                           2.0 * bits[..., 1].float() - 1.0) / math.sqrt(2)
    snr_db = torch.empty(batch_size, 1, device=device).uniform_(snr_lo, snr_hi)
    noise_var = 10.0 ** (-snr_db / 10.0)
    # Pilot Xp[k] == 1 for every subcarrier. Receiver knows pilot only.
    yp = h + complex_noise((batch_size, nsc), noise_var, device)
    yd = h * symbol + complex_noise((batch_size, nsc), noise_var, device)
    return Batch(yp, yd, h, bits, noise_var)


def estimate_channel(yp, noise_var, method, taps=3, covariance=None):
    if method == 'ls':
        return yp
    if method == 'dft':
        ht = torch.fft.ifft(yp, dim=-1)
        # Known maximum delay support; does not access true individual taps.
        return torch.fft.fft(F.pad(ht[..., :taps], (0, yp.shape[-1]-taps)), dim=-1)
    if method == 'lmmse':
        n = yp.shape[-1]
        eye = torch.eye(n, dtype=yp.dtype, device=yp.device)
        mat = covariance[None, :, :] + noise_var[:, :, None] * eye[None, :, :]
        # E[H|Ypilot] for Gaussian channel prior (pilot-only)
        y = torch.linalg.solve(mat, yp.unsqueeze(-1)).squeeze(-1)
        return torch.einsum('ij,bj->bi', covariance, y)
    raise ValueError(method)


def covariance_matrix(n, profile, device):
    p = torch.tensor(profile, device=device, dtype=torch.float32)
    p = p / p.sum()
    f = torch.fft.fft(torch.eye(n, len(p), device=device, dtype=torch.complex64), dim=0)
    return (f * p[None, :]) @ f.conj().T


def decisions_from_channel(yd, hhat, noise_var):
    # Regularization avoids division by zero. Positive real denominator does
    # not change QPSK hard-decision boundaries.
    z = yd * hhat.conj() / (hhat.abs().square() + noise_var + 1e-8)
    return torch.stack((z.real > 0, z.imag > 0), dim=-1).long()


class SmallOFDMTransformer(nn.Module):
    def __init__(self, nsc, width=48, depth=2, heads=4):
        super().__init__()
        self.nsc = nsc
        self.in_proj = nn.Linear(9, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            batch_first=True, norm_first=True, dropout=0.0, activation='gelu')
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=depth,
                                          enable_nested_tensor=False)
        self.out_proj = nn.Linear(width, 2)
        self.register_buffer('omega', 2 * math.pi * torch.arange(nsc) / nsc)

    def forward(self, yp, yd, noise_var):
        n = yp.shape[-1]
        if n != self.nsc:
            raise ValueError(f'model expects {self.nsc} subcarriers, received {n}')
        z = yd * yp.conj() / (yp.abs().square() + noise_var + 0.01)
        ang = self.omega[None, :].expand(yp.shape[0], -1)
        nv = noise_var.expand(-1, n)
        inp = torch.stack((yp.real, yp.imag, yd.real, yd.imag,
                           z.real, z.imag, ang.cos(), ang.sin(),
                           nv.log10()), dim=-1)
        x = self.in_proj(inp)
        x = self.net(x)
        # An explicit conventional-estimate skip path accelerates training.
        return self.out_proj(x) + 3.0 * torch.stack((z.real, z.imag), dim=-1)


def run_evaluation(model, nsc, device, n_frames, snrs, profile):
    methods = ('ls', 'dft', 'lmmse', 'oracle', 'transformer')
    cov = covariance_matrix(nsc, profile, device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for snr in snrs:
            errors = {m: 0 for m in methods}
            total = 0
            batches = max(1, math.ceil(n_frames/128))
            for idx in range(batches):
                size = min(128, n_frames - idx*128)
                if size <= 0:
                    break
                b = make_batch(size, nsc, snr, snr, device, profile)
                h_est = {
                    'ls': estimate_channel(b.yp, b.noise_var, 'ls'),
                    'dft': estimate_channel(b.yp, b.noise_var, 'dft', taps=len(profile)),
                    'lmmse': estimate_channel(b.yp, b.noise_var, 'lmmse', covariance=cov),
                    'oracle': b.h,
                }
                for m, h in h_est.items():
                    errors[m] += int((decisions_from_channel(b.yd, h, b.noise_var)
                                      != b.bits).sum().item())
                pred = (model(b.yp, b.yd, b.noise_var) > 0).long()
                errors['transformer'] += int((pred != b.bits).sum().item())
                total += b.bits.numel()
            for m in methods:
                rows.append({'snr_db': snr, 'receiver': m, 'bit_errors': errors[m],
                             'total_bits': total, 'ber': errors[m] / total})
    return rows


def smoke_test(device):
    torch.manual_seed(123)
    b = make_batch(4, 32, 300, 300, device)  # approximately noiseless
    recovered = decisions_from_channel(b.yd, b.yp, b.noise_var)
    assert torch.equal(recovered, b.bits), 'near-noiseless recovery failed'
    model = SmallOFDMTransformer(32).to(device)
    assert model(b.yp, b.yd, b.noise_var).shape == b.bits.shape
    assert torch.isfinite(model(b.yp, b.yd, b.noise_var)).all()
    print('SMOKE TEST PASSED: synth, channel, QPSK demod, model shapes, finite outputs')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--nsc', type=int, default=32)
    ap.add_argument('--eval-frames', type=int, default=500)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='results')
    ap.add_argument('--smoke-only', action='store_true')
    args = ap.parse_args()
    if args.nsc < 8 or args.nsc < 3:
        ap.error('nsc must be >= 8')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smoke_test(device)
    if args.smoke_only:
        return
    torch.manual_seed(args.seed)
    model = SmallOFDMTransformer(args.nsc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    print('device:', device, 'parameters:', sum(p.numel() for p in model.parameters()))
    t0 = time.time()
    model.train()
    for step in range(1, args.steps+1):
        b = make_batch(args.batch_size, args.nsc, 0.0, 20.0, device)
        opt.zero_grad(set_to_none=True)
        logits = model(b.yp, b.yd, b.noise_var)
        loss = F.binary_cross_entropy_with_logits(logits, b.bits.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f'step {step:04d} / {args.steps:04d} | BCE {loss.item():.4f}')
    print(f'train_seconds: {time.time()-t0:.1f}')
    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    tests = [
        ('in_distribution', (0.72, 0.20, 0.08)),
        ('shifted_delay_profile', (0.25, 0.25, 0.25, 0.25)),
    ]
    for label, prof in tests:
        rows = run_evaluation(model, args.nsc, device, args.eval_frames,
                              [0, 5, 10, 15, 20, 25], prof)
        for r in rows:
            r['split'] = label
        all_rows.extend(rows)
    csvpath = os.path.join(args.out, 'ber_results.csv')
    with open(csvpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split','snr_db','receiver',
                                               'bit_errors','total_bits','ber'])
        writer.writeheader()
        writer.writerows(all_rows)
    torch.save({'state_dict': model.cpu().state_dict(),
                'config': {'nsc': args.nsc}, 'seed': args.seed},
               os.path.join(args.out, 'model.pt'))
    with open(os.path.join(args.out, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'Output: {csvpath}')
    print('In-distribution BER at 10 dB:')
    for row in all_rows:
        if row['split'] == 'in_distribution' and row['snr_db'] == 10:
            print(f"  {row['receiver']:11s} {row['ber']:.6f}")
    print('No superiority claim: compare all SNRs and held-out profiles in CSV.')

if __name__ == '__main__':
    main()
