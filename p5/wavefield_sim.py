"""CPU baseband OFDM generator. Only Observation crosses the receiver boundary.

Each seed indexes independent payload, path, PA, oscillator and AWGN streams.
All transforms use unitary scaling. The useful samples of a CP-guarded,
bandlimited OFDM symbol admit exact integer or fractional delays by a DFT
phase ramp. The path gain then varies at every receive sample, producing ICI.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from p4.simulator import (
    complex_randn, decimate_time_to_frequency, oversample_frequency_to_time, q16, qpsk,
)
from p4.slot_2d import encode_ldpc


N = 32
CP = 16
OVERSAMPLE = 2
TWO_PI = 2.0 * math.pi
REAL = torch.float64
COMPLEX = torch.complex128
PILOT_ROWS = {1: (0, 4), 2: (0, 9, 10, 19), 3: (0, 4)}
SYMBOLS = {1: 5, 2: 20, 3: 5}
RECEIVERS = {1: 1, 2: 1, 3: 4}
DELAYS = {1: (0., 2., 4.), 2: (0., 2., 4.), 3: (0., 1.5, 3.5, 5.25)}
POWERS = {1: (.75, .17, .08), 2: (.75, .17, .08), 3: (.48, .28, .16, .08)}


@dataclass(frozen=True)
class Observation:
    y: torch.Tensor             # [burst, antenna, OFDM symbol, subcarrier]
    pilot_mask: torch.Tensor    # [symbol, subcarrier], bool
    pilots: torch.Tensor        # [symbol, subcarrier], zero on payload
    noise_var: torch.Tensor     # [burst], each complex receive sample
    regime: int


@dataclass(frozen=True)
class PAState:
    A: torch.Tensor             # [burst, symbol, 1]
    knee: torch.Tensor
    alpha: torch.Tensor


@dataclass(frozen=True)
class PhysicalTruth:
    delays: torch.Tensor        # [burst, ray], in Nyquist samples
    gains: torch.Tensor         # [burst, ray], complex ray amplitudes
    angles: torch.Tensor        # [burst, ray], broadside radians
    nu_start: torch.Tensor      # [burst, ray], cycles per useful OFDM symbol
    nu_end: torch.Tensor
    ray_evolution: torch.Tensor # [burst, symbol, ray], useful-sample phase mean
    pa: PAState
    phn: torch.Tensor           # [burst, symbol, useful sample], receiver phase
    phn_sigma: torch.Tensor     # [burst], rad per square root of sample


@dataclass(frozen=True)
class Frame:
    obs: Observation
    bits: torch.Tensor          # [burst, payload RE, bit], P4 mapping
    info_bits: torch.Tensor     # [burst, LDPC codeword, 64]
    h_true: torch.Tensor        # CFR diagonal of propagation alone
    truth: PhysicalTruth


def _rng(seed, stream):
    return torch.Generator().manual_seed((int(seed) * 1009 + stream) % (2**63 - 1))


def _uniform(shape, generator, low, high):
    return low + (high - low) * torch.rand(shape, dtype=REAL, generator=generator)


def _phasor(phase):
    return torch.polar(torch.ones_like(phase), phase)


def steer(angles, n_rx):
    """Half-wave ULA, unit Euclidean norm. Shape [..., ray, antenna]."""
    index = torch.arange(n_rx, dtype=REAL, device=angles.device)
    return _phasor(math.pi * angles.to(REAL).sin()[..., None] * index) / math.sqrt(n_rx)


def frequency_basis(delays, n=N):
    bins = torch.fft.fftfreq(n, d=1/n, dtype=REAL, device=delays.device)
    return _phasor((-TWO_PI / n) * delays.to(REAL)[..., None] * bins)


def ray_phase(nu_start, nu_end, time, duration):
    """Integrated linear Doppler versus time in useful-symbol units."""
    return TWO_PI * (nu_start[..., None] * time +
                     (nu_end - nu_start)[..., None] * (time.square() / (2 * duration)))


@lru_cache(maxsize=3)
def pilot_grid(regime):
    t = SYMBOLS[regime]
    mask = torch.zeros((t, N), dtype=torch.bool)
    mask[list(PILOT_ROWS[regime]), ::2] = True
    known = torch.zeros((t, N), dtype=COMPLEX)
    pilot_bits = torch.randint(2, (int(mask.sum()), 2), generator=_rng(900, regime))
    known[mask] = qpsk(pilot_bits).to(COMPLEX)
    # P4's QPSK table is float32; restore exact unit magnitude after promotion.
    known[mask] /= known[mask].abs()
    return mask, known


def pa_grid(x, A, knee, alpha, oversample=OVERSAMPLE):
    """Bandlimited Rapp AM-AM and Saleh AM-PM (beta=1) on the transmit grid.

    Scalars or tensors [burst,symbol,1] broadcast against the time samples.
    Uses P4's baseband oversampling/reconstruction functions verbatim.
    """
    signal = oversample_frequency_to_time(x, oversample)
    radius = signal.abs()
    gain = (1 + (radius / A) ** (2 * knee)) ** (-1 / (2 * knee))
    angle = alpha * radius.square() / (1 + radius.square())
    return decimate_time_to_frequency(signal * gain * _phasor(angle), x.shape[-1], oversample)


def _pa_state(b, t, regime, rng):
    def drifting(low, high):
        ends = _uniform((b, 2), rng, low, high)
        progress = torch.linspace(0, 1, t, dtype=REAL)
        return torch.lerp(ends[:, :1], ends[:, 1:], progress).unsqueeze(-1)

    if regime == 2:
        return PAState(torch.full((b, t, 1), math.inf, dtype=REAL),
                       torch.full((b, t, 1), 2., dtype=REAL), torch.zeros(b, t, 1, dtype=REAL))
    return PAState(drifting(1.05, 2.5), drifting(1.3, 2.7), drifting(-.30, .90))


def _paths(b, regime, generator):
    powers = torch.tensor(POWERS[regime], dtype=REAL)
    delays = torch.tensor(DELAYS[regime], dtype=REAL).expand(b, -1).clone()
    paths = len(powers)
    gains = complex_randn((b, paths), generator=generator, dtype=COMPLEX) * powers.sqrt()
    angles = torch.zeros_like(delays)
    start = torch.zeros_like(delays)
    end = torch.zeros_like(delays)
    if regime in (1, 2):
        gains[:, 0] = powers[0].sqrt() * _phasor(_uniform((b,), generator, -math.pi, math.pi))
    if regime == 2:
        signs = 2 * torch.randint(2, (b, paths), generator=generator) - 1
        start = signs * _uniform((b, paths), generator, .045, .080)
        end = signs * _uniform((b, paths), generator, .045, .080)
    if regime == 3:
        delays[:, 1:] += _uniform((b, paths - 1), generator, -.15, .15)
        centers = _uniform((b, 2), generator, -.8, .8).repeat_interleave(2, dim=-1)
        angles = centers + .10 * torch.randn(b, paths, dtype=REAL, generator=generator)
        start = _uniform((b, paths), generator, -.035, .035)
        end = start.clone()
    return gains, delays, angles, start, end


def propagate(x, delays, gains, angles, nu_start, nu_end, n_rx, cp=CP):
    """Return noiseless post-CP *time* samples, CFR diagonal, and ray phase means.

    Path delays must fit the guard. The time coordinate includes CP intervals.
    Modeling only the retained samples avoids allocating discarded prefixes.
    """
    b, t, n = x.shape
    if bool((delays < 0).any() | (delays > cp).any()):
        raise ValueError("Every path delay must fit the cyclic prefix")
    clock = (torch.arange(t, dtype=REAL)[:, None] * (n + cp) +
             cp + torch.arange(n, dtype=REAL)) / n
    duration = (t * (n + cp) - 1) / n
    phase = ray_phase(nu_start, nu_end, clock.flatten(), duration)
    evolution = _phasor(phase.reshape(b, -1, t, n))
    basis = frequency_basis(delays, n)
    delayed = torch.fft.ifft(x[:, None] * basis[:, :, None], norm="ortho")
    field = delayed * evolution
    spatial = (steer(angles, n_rx) * math.sqrt(n_rx) * gains[..., None]).transpose(-1, -2)
    useful = (spatial @ field.flatten(2)).reshape(b, n_rx, t, n)
    mean_phase = evolution.mean(-1)
    diagonal_rays = mean_phase[..., None] * basis[:, :, None]
    diagonal = (spatial @ diagonal_rays.flatten(2)).reshape(b, n_rx, t, n)
    return useful, diagonal, mean_phase.transpose(1, 2)


def simulate(regime, batch_size, snr_db, seed):
    """Draw paired full bursts. SNR is unit transmit Es divided by known AWGN N0."""
    if regime not in SYMBOLS or batch_size < 1:
        raise ValueError("Expected regime in {1,2,3} and a positive batch size")
    b, t, r = batch_size, SYMBOLS[regime], RECEIVERS[regime]
    mask, pilots = pilot_grid(regime)
    n_data = int((~mask).sum())
    assert n_data % 32 == 0
    info = torch.randint(2, (b, n_data // 32, 64), generator=_rng(seed, 1))
    code, _ = encode_ldpc(info)
    bits = code.reshape(b, n_data, 4)
    grid = pilots.expand(b, -1, -1).clone()
    grid[:, ~mask] = q16(bits).to(COMPLEX)
    pa = _pa_state(b, t, regime, _rng(seed, 2))
    tx = grid if regime == 2 else pa_grid(grid, pa.A, pa.knee, pa.alpha)
    gains, delays, angles, start, end = _paths(b, regime, _rng(seed, 3))
    signal, h_true, ray_evolution = propagate(tx, delays, gains, angles, start, end, r)
    phn = torch.zeros((b, t, N), dtype=REAL)
    sigma = torch.zeros(b, dtype=REAL)
    if regime == 1:
        rng = _rng(seed, 4)
        sigma = _uniform((b,), rng, .012, .032)
        increments = torch.randn(b, t * (N + CP), dtype=REAL, generator=rng) * sigma[:, None]
        increments[:, 0] = 0.
        phn = increments.cumsum(-1).reshape(b, t, N + CP)[:, :, CP:]
        signal = signal * _phasor(phn)[:, None]
    noise_var = torch.full((b,), 10.**(-float(snr_db) / 10.), dtype=REAL)
    noise = complex_randn(signal.shape, generator=_rng(seed, 5), dtype=COMPLEX)
    y = torch.fft.fft(signal + noise * noise_var[:, None, None, None].sqrt(), norm="ortho")
    obs = Observation(y, mask, pilots, noise_var, regime)
    truth = PhysicalTruth(delays, gains, angles, start, end, ray_evolution, pa, phn, sigma)
    return Frame(obs, bits, info, h_true, truth)


def green_kernel(frame, coords):
    """Conditional phase-ensemble |E[H_i H_j*]| of the propagation diagonal.

    Input coordinates [Q,3] are (antenna,symbol,subcarrier). Conditional ray
    powers are realized squared amplitudes, initial independent phases averaged
    out. This evaluation-only target excludes PA, phase noise and AWGN.
    """
    r, t, k = coords.long().unbind(-1)
    truth = frame.truth
    num_rx = frame.obs.y.shape[1]
    spatial = (steer(truth.angles, num_rx) * math.sqrt(num_rx)).transpose(1, 2)
    basis = frequency_basis(truth.delays).transpose(1, 2)
    features = spatial[:, r] * truth.ray_evolution[:, t] * basis[:, k] * truth.gains.abs()[:, None]
    return (features @ features.mH).abs()
