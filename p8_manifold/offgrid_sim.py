"""Phase 8 Off-Grid continuous-delay, variable-path, non-uniform pilot & blind-corruption OFDM simulator."""
from dataclasses import dataclass, replace
import math

import torch

from p4.simulator import complex_randn, q16, qpsk
from p4.slot_2d import _get_default_ldpc, decode_min_sum, encode_ldpc
from p5.wavefield_sim import CP, N, PAState, PhysicalTruth, frequency_basis, propagate, ray_phase
from p7.hypotheses import frequency_operators_fast


REAL = torch.float64
COMPLEX = torch.complex128
T_SYM = 20
N_SUB = N
N_PILOTS = 64
N_PAYLOAD = T_SYM * N_SUB - N_PILOTS  # 576 payload REs = 18 LDPC codewords of 128 bits
P_MAX = 5

OFFGRID_CONDITIONS = (
    "O0_random_sparse",
    "O1_roving_island",
    "O2_blind_burst_erasure",
    "O3_holdout_sequential",
)


@dataclass(frozen=True)
class OffGridObservation:
    y: torch.Tensor              # [B, 1, T, N] complex128
    pilot_mask: torch.Tensor     # [B, T, N] bool (exact 64 TX pilots per burst)
    pilots: torch.Tensor         # [B, T, N] complex128 (known TX pilot symbols, 0 on payload)
    noise_var: torch.Tensor      # [B] float64
    valid_mask: torch.Tensor     # [B, T, N] bool (known observed mask; all True in blind mode)
    blind: bool = False          # True in O2_blind_burst_erasure


@dataclass(frozen=True)
class OffGridFrame:
    obs: OffGridObservation
    bits: torch.Tensor           # [B, 576, 4] int64
    info_bits: torch.Tensor      # [B, 18, 64] int64
    h_true: torch.Tensor         # [B, 1, T, N] complex128 (exact diagonal CFR)
    truth: PhysicalTruth
    n_rays: torch.Tensor         # [B] int64 (active rays per burst in {2,3,4,5})
    condition: str
    probe_mask: torch.Tensor     # [B, T, N] bool (held-out probes in O3, else all False)
    corrupt_mask: torch.Tensor   # [B, T, N] bool (eval-only truth of blind corruptions in O2)
    y_clean_full: torch.Tensor   # [B, 1, T, N] complex128 (unmasked noisy received grid for O3 Pass 2 / probe scoring)


def _rng(seed, stream):
    return torch.Generator().manual_seed((int(seed) * 1009 + stream) % (2**63 - 1))


def _uniform(shape, gen, low, high):
    return low + (high - low) * torch.rand(shape, dtype=REAL, generator=gen)


def _phasor(phase):
    return torch.polar(torch.ones_like(phase), phase)


def _sample_pilot_masks(batch_size, condition, gen):
    """Return (pilot_mask[B,T,N], probe_mask[B,T,N], corrupt_mask[B,T,N], erase_mask[B,T,N], impulse_mask[B,T,N])."""
    pilot_mask = torch.zeros((batch_size, T_SYM, N_SUB), dtype=torch.bool)
    probe_mask = torch.zeros((batch_size, T_SYM, N_SUB), dtype=torch.bool)
    erase_mask = torch.zeros((batch_size, T_SYM, N_SUB), dtype=torch.bool)
    impulse_mask = torch.zeros((batch_size, T_SYM, N_SUB), dtype=torch.bool)

    for b in range(batch_size):
        if condition in ("O0_random_sparse", "O2_blind_burst_erasure"):
            perm = torch.randperm(T_SYM * N_SUB, generator=gen)[:N_PILOTS]
            rows = perm // N_SUB
            cols = perm % N_SUB
            pilot_mask[b, rows, cols] = True
            if condition == "O2_blind_burst_erasure":
                sub_perm = torch.randperm(N_PILOTS, generator=gen)
                er_idx = sub_perm[:8]
                im_idx = sub_perm[8:16]
                erase_mask[b, rows[er_idx], cols[er_idx]] = True
                impulse_mask[b, rows[im_idx], cols[im_idx]] = True
        elif condition in ("O1_roving_island", "O3_holdout_sequential"):
            # Two anchor bands near start (2 rows in [0..3]) & end (2 rows in [16..19]): 24 pilots each = 48
            r_start = int(torch.randint(0, 3, (1,), generator=gen).item())
            r_end = int(torch.randint(16, 19, (1,), generator=gen).item())
            band_a_rows = (r_start, r_start + 1)
            band_b_rows = (r_end, r_end + 1)
            for rr in band_a_rows:
                cols = torch.randperm(N_SUB, generator=gen)[:12]
                pilot_mask[b, rr, cols] = True
            for rr in band_b_rows:
                cols = torch.randperm(N_SUB, generator=gen)[:12]
                pilot_mask[b, rr, cols] = True
            # Roving interior island: 2x8 contiguous (circular in freq) window at random t0 in [6..12], f0 in [0..31]
            t_isl = int(torch.randint(6, 13, (1,), generator=gen).item())
            f_isl = int(torch.randint(0, N_SUB, (1,), generator=gen).item())
            isl_coords = []
            for dt in range(2):
                for df in range(8):
                    rr = t_isl + dt
                    cc = (f_isl + df) % N_SUB
                    pilot_mask[b, rr, cc] = True
                    isl_coords.append((rr, cc))
            if condition == "O3_holdout_sequential":
                # Hold out 8 of the 16 roving-island pilots as zero-leakage probes
                h_perm = torch.randperm(len(isl_coords), generator=gen)[:8]
                for idx in h_perm.tolist():
                    rr, cc = isl_coords[idx]
                    probe_mask[b, rr, cc] = True
        else:
            raise ValueError(f"Unknown Off-Grid condition: {condition}")

        assert int(pilot_mask[b].sum()) == N_PILOTS

    corrupt_mask = erase_mask | impulse_mask
    return pilot_mask, probe_mask, corrupt_mask, erase_mask, impulse_mask


def _sample_offgrid_paths(batch_size, gen):
    """Draw physically identifiable off-grid continuous delays (tau in [0, 6.5]), variable P in {2..5},
    and linear-chirp Doppler (nu in [0.015, 0.048]) placing 16-QAM LDPC right in the 15%-75% BLER waterfall."""
    delays = torch.zeros((batch_size, P_MAX), dtype=REAL)
    gains = torch.zeros((batch_size, P_MAX), dtype=COMPLEX)
    angles = torch.zeros((batch_size, P_MAX), dtype=REAL)
    nu_start = torch.zeros((batch_size, P_MAX), dtype=REAL)
    nu_end = torch.zeros((batch_size, P_MAX), dtype=REAL)
    n_rays = torch.zeros((batch_size,), dtype=torch.int64)

    for b in range(batch_size):
        p_cnt = int(torch.randint(2, P_MAX + 1, (1,), generator=gen).item())
        n_rays[b] = p_cnt
        # Continuous separated off-grid delays inside [0, 6.5] < CP=16
        d0 = float(_uniform((1,), gen, 0.0, 1.0).item())
        # Place remaining p_cnt - 1 delays in [d0 + 0.35, 6.5] with spacing >= 0.35
        span = 6.5 - (d0 + 0.35) - 0.35 * max(p_cnt - 2, 0)
        raw = torch.sort(torch.rand(p_cnt - 1, dtype=REAL, generator=gen)).values * max(span, 0.5)
        rest = (d0 + 0.35) + raw + 0.35 * torch.arange(p_cnt - 1, dtype=REAL)
        d = torch.cat((torch.tensor([d0], dtype=REAL), rest))

        decay = float(_uniform((1,), gen, 0.35, 0.70).item())
        powers = torch.exp(-decay * torch.arange(p_cnt, dtype=REAL))
        powers = powers / powers.sum()

        g = complex_randn((p_cnt,), generator=gen, dtype=COMPLEX) * powers.sqrt()
        # Unit-norm phase on ray 0 (preserving Rayleigh fading on multipath taps 1..P-1)
        g[0] = powers[0].sqrt() * _phasor(_uniform((1,), gen, -math.pi, math.pi))[0]

        signs = 2.0 * torch.randint(2, (p_cnt,), generator=gen).double() - 1.0
        ns = signs * _uniform((p_cnt,), gen, 0.015, 0.048)
        ne = signs * _uniform((p_cnt,), gen, 0.015, 0.048)

        delays[b, :p_cnt] = d
        gains[b, :p_cnt] = g
        nu_start[b, :p_cnt] = ns
        nu_end[b, :p_cnt] = ne

    return gains, delays, angles, nu_start, nu_end, n_rays


def simulate_offgrid(batch_size, snr_db, seed, condition="O0_random_sparse"):
    """Generate an Off-Grid multi-path OFDM batch with exact physical propagation and LDPC codewords."""
    if condition not in OFFGRID_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    pilot_mask, probe_mask, corrupt_mask, erase_mask, impulse_mask = _sample_pilot_masks(
        batch_size, condition, _rng(seed, 10)
    )

    # Generate QPSK pilots per burst (exact unit magnitude)
    pilots = torch.zeros((batch_size, T_SYM, N_SUB), dtype=COMPLEX)
    p_bits = torch.randint(2, (batch_size, N_PILOTS, 2), generator=_rng(seed, 11))
    p_syms = qpsk(p_bits).to(COMPLEX)
    p_syms = p_syms / p_syms.abs()
    for b in range(batch_size):
        pilots[b, pilot_mask[b]] = p_syms[b]

    # Encode 18 LDPC (128,64) codewords -> 576 16-QAM payload symbols per burst
    info = torch.randint(2, (batch_size, N_PAYLOAD // 32, 64), generator=_rng(seed, 1))
    code, _ = encode_ldpc(info)
    bits = code.reshape(batch_size, N_PAYLOAD, 4)

    tx = pilots.clone()
    q_syms = q16(bits).to(COMPLEX)
    for b in range(batch_size):
        tx[b, ~pilot_mask[b]] = q_syms[b]

    gains, delays, angles, nu_start, nu_end, n_rays = _sample_offgrid_paths(batch_size, _rng(seed, 3))
    signal, h_true, ray_evolution = propagate(tx, delays, gains, angles, nu_start, nu_end, n_rx=1, cp=CP)

    noise_var = torch.full((batch_size,), 10.0 ** (-float(snr_db) / 10.0), dtype=REAL)
    noise = complex_randn(signal.shape, generator=_rng(seed, 5), dtype=COMPLEX)
    y_full = torch.fft.fft(signal + noise * noise_var[:, None, None, None].sqrt(), norm="ortho")

    y_obs = y_full.clone()
    valid_mask = torch.ones((batch_size, T_SYM, N_SUB), dtype=torch.bool)
    blind = False

    if condition == "O2_blind_burst_erasure":
        blind = True
        # Corrupt 16 pilots per burst WITHOUT revealing corrupt_mask to receiver
        y_obs[:, 0] = torch.where(erase_mask, torch.zeros_like(y_obs[:, 0]), y_obs[:, 0])
        imp_phase = _phasor(_uniform((batch_size, T_SYM, N_SUB), _rng(seed, 12), -math.pi, math.pi))
        imp_amp = _uniform((batch_size, T_SYM, N_SUB), _rng(seed, 13), 8.0, 12.0)
        y_obs[:, 0] = torch.where(impulse_mask, y_obs[:, 0] + imp_amp * imp_phase, y_obs[:, 0])
    elif condition == "O3_holdout_sequential":
        valid_mask = ~probe_mask
        y_obs[:, 0] = torch.where(valid_mask, y_obs[:, 0], torch.zeros_like(y_obs[:, 0]))

    pa = PAState(
        torch.full((batch_size, T_SYM, 1), math.inf, dtype=REAL),
        torch.full((batch_size, T_SYM, 1), 2.0, dtype=REAL),
        torch.zeros((batch_size, T_SYM, 1), dtype=REAL),
    )
    phn = torch.zeros((batch_size, T_SYM, N_SUB), dtype=REAL)
    sigma = torch.zeros((batch_size,), dtype=REAL)
    truth = PhysicalTruth(delays, gains, angles, nu_start, nu_end, ray_evolution, pa, phn, sigma)
    obs = OffGridObservation(y_obs, pilot_mask, pilots, noise_var, valid_mask, blind=blind)
    return OffGridFrame(
        obs=obs, bits=bits, info_bits=info, h_true=h_true, truth=truth,
        n_rays=n_rays, condition=condition, probe_mask=probe_mask,
        corrupt_mask=corrupt_mask, y_clean_full=y_full,
    )


def true_offgrid_operator(frame):
    """Build exact noiseless propagation operator G_true[B, T, N, N] for OffGridFrame (eval only)."""
    truth = frame.truth
    b, _, t, n = frame.obs.y.shape
    clock = (torch.arange(t, dtype=REAL)[:, None] * (n + CP) + CP + torch.arange(n, dtype=REAL)) / n
    duration = (t * (n + CP) - 1) / n
    phase = ray_phase(truth.nu_start, truth.nu_end, clock.flatten(), duration)
    rot = torch.polar(torch.ones_like(phase), phase).reshape(b, -1, t, n).permute(0, 2, 3, 1)
    basis = frequency_basis(truth.delays, n)
    cpe = torch.ones((b, t), dtype=COMPLEX)
    ops = frequency_operators_fast(basis, rot, cpe)
    return torch.einsum("btpfk,bp->btfk", ops, truth.gains)


def extract_payload_llrs(pilot_mask, bit_llr):
    """Gather per-burst payload LLRs [B, 576, 4] from grid [B, T, N, 4] with per-burst pilot_mask [B, T, N]."""
    b = bit_llr.shape[0]
    return torch.stack([bit_llr[i, ~pilot_mask[i]] for i in range(b)], dim=0)


@torch.no_grad()
def score_offgrid(frame, bit_llr, h_hat):
    """Exact LDPC coded BLER, Pre-BER, Post-BER, proper bit BCE, and CFR NMSE for OffGridFrame."""
    data = extract_payload_llrs(frame.obs.pilot_mask, bit_llr).double()
    if data.shape != frame.bits.shape or h_hat.shape != frame.h_true.shape:
        raise ValueError("OffGrid receiver output shape mismatch")
    if not (torch.isfinite(data).all() and torch.isfinite(h_hat).all()):
        raise ValueError("Nonfinite OffGrid receiver output")
    blocks = data.reshape(*frame.info_bits.shape[:-1], 128)
    decoded = decode_min_sum(blocks, _get_default_ldpc().h_matrix, max_iter=12)
    wrong = decoded != frame.info_bits
    signed_bits = 2.0 * frame.bits.to(device=data.device, dtype=REAL) - 1.0
    bce_sum = float(torch.nn.functional.softplus(-signed_bits * data).sum())
    return dict(
        bit_errors=int(((data > 0) != frame.bits).sum()),
        bits=int(frame.bits.numel()),
        info_errors=int(wrong.sum()),
        info_bits=int(wrong.numel()),
        block_errors=int(wrong.any(-1).sum()),
        blocks=int(wrong.numel() // 64),
        channel_error=float((h_hat - frame.h_true).abs().square().sum()),
        channel_energy=float(frame.h_true.abs().square().sum()),
        bce_sum=bce_sum,
        bit_count=int(data.numel()),
    )


@torch.no_grad()
def calibration_offgrid(frame, bit_llr, traces):
    """Per-burst pilot-mask-aware 7-bin Step-last symbol calibration matching p6_consistency bins."""
    edges = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8)
    b = frame.bits.shape[0]
    pm = frame.obs.pilot_mask
    truth_syms = q16(frame.bits.long()).to(COMPLEX)  # B, 576
    last = traces[-1]
    mean_p = torch.stack([last["mean"][i, ~pm[i]] for i in range(b)], dim=0).to(COMPLEX)
    var_p = torch.stack([last["variance"][i, ~pm[i]] for i in range(b)], dim=0).double()
    sq_err = (mean_p - truth_syms).abs().square()
    bounds = var_p.new_tensor(edges[1:])
    indices = torch.bucketize(var_p.contiguous(), bounds, right=True)
    bins = []
    for idx, lower in enumerate(edges):
        upper = edges[idx + 1] if idx + 1 < len(edges) else None
        sel = indices == idx
        bins.append(
            dict(
                index=idx, lower=lower, upper=upper,
                count=int(sel.sum()),
                sum_predicted_variance=float(var_p[sel].sum()),
                sum_squared_error=float(sq_err[sel].sum()),
            )
        )
    return dict(
        bins=bins,
        total=dict(
            count=int(var_p.numel()),
            sum_predicted_variance=float(var_p.sum()),
            sum_squared_error=float(sq_err.sum()),
        ),
    )
