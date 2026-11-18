"""Stage P4-B: 2D Time-Frequency Slots, Continuous Doppler/CFO, Outer LDPC FEC,
2D Classical Baselines, 2D C*-Equivariant EMFloatFormer2D, and Coded BLER Evaluation.

Features Implemented:
- F13: 2D Slot Simulator & Sparse DMRS (SlotSimulator2D, get_dmrs_mask, get_slot_masks,
       get_custom_dmrs_mask, generate_dmrs)
- F14: Continuous Doppler ICI & CFO (simulate_slot_channel, measure_ici_leakage,
       measure_cfo_phase_drift, estimate_cfo_cp, compute_doppler_spectrum,
       check_phase_continuity)
- F15: Native Outer Channel Coding (Rate-1/2 PEG/IRA-LDPC: LDPCMatrix, encode_ldpc,
       decode_min_sum, correct_single_bit, measure_coding_gain)
- F16: 2D Classical Baselines & EMFloatFormer2D (compute_2d_correlation, lmmse_2d_interpolate,
       compare_1d_vs_2d_lmmse, idd_pa_2d, idd_pa_2d_cleanup, lmmse_2d_flat, lmmse_2d_singular,
       idd_pa_2d_linear, idd_pa_2d_clipping, check_edge_extrapolation, EMFloatFormer2D,
       JointPhysicalLoss2D)
- F17: Coded BLER Evaluation, Tier 3/4 Workloads & Reproducible Benchmarks
       (check_block_error, evaluate_bler_curve, run_block_counter,
        check_doppler_bler_sensitivity, evaluate_ood_profile_bler, evaluate_bler_point,
        simulate_single_block, check_integer_accumulator, compute_ci,
        run_p4b_coded_bler_benchmarks, plus Tier 3 & Tier 4 callables)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from p4.simulator import (
    DEFAULT_CP,
    DEFAULT_N,
    DEFAULT_OVERSAMPLE,
    PROFILES,
    SIGMA_16QAM,
    THRESH_16QAM,
    apply_rapp_pa_samples,
    complex_randn,
    decimate_time_to_frequency,
    hard_bits_16qam,
    oversample_frequency_to_time,
    q16,
    qpsk,
    rapp_pa,
)
from p4.classical_baselines import (
    bussgang_params,
    calibrated_16qam_llrs,
    hard_slice_16qam,
    soft_slice_16qam,
)
from p4.em_floatformer import (
    ComplexLinear,
    ComplexRMSNorm,
    FloatFormerOutput,
    RadialActivation,
    apply_gauge_normalization,
    compute_ece,
    compute_gauge_anchor,
    hermitian_attention_scores,
    rescale_channel_estimate,
    soft_reconstruct_16qam,
)

# ---------------------------------------------------------------------------
# Constants & Extended 2D Channel Profiles
# ---------------------------------------------------------------------------

DEFAULT_N_SYMB = 14
DEFAULT_N_SC = DEFAULT_N  # 32
SPEED_OF_LIGHT = 299_792_458.0
DEFAULT_SCS_HZ = 30_000.0  # 30 kHz 5G NR subcarrier spacing

# Unseen 16-tap exponentially decaying OOD PDP (delays 0..15 <= DEFAULT_CP=16)
_EXP16_RAW = tuple(math.exp(-0.28 * k) for k in range(16))
_EXP16_SUM = sum(_EXP16_RAW)
EXP16_PROFILE: Tuple[float, ...] = tuple(v / _EXP16_SUM for v in _EXP16_RAW)

SLOT_PROFILES: Dict[str, Tuple[float, ...]] = {
    **PROFILES,
    "exp16": EXP16_PROFILE,
}


def _resolve_pa_params(
    pa_condition: Optional[str] = "clean",
    A: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """Resolve (A, alpha) from either explicit kwargs or pa_condition string."""
    if A is not None or alpha is not None:
        a_out = None if (A is None or math.isinf(float(A))) else float(A)
        alpha_out = 0.0 if alpha is None else float(alpha)
        if a_out is not None and alpha is None:
            alpha_out = 0.5
        return a_out, alpha_out
    if pa_condition is None:
        return None, 0.0
    cond = str(pa_condition).strip().lower()
    if cond == "clean":
        return None, 0.0
    if cond == "mild":
        return 2.25, 0.5
    if cond == "strong":
        return 1.25, 0.5
    raise ValueError(f"Unknown pa_condition {pa_condition!r}. Choose 'clean', 'mild', or 'strong'.")


# ===========================================================================
# Feature F13: 2D Slot Simulator & Sparse DMRS
# ===========================================================================

def get_dmrs_mask(
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    pilot_symbols: Optional[Sequence[int]] = None,
    comb: int = 2,
    offset: int = 0,
    pilot_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Construct boolean 2D sparse DMRS mask of shape (n_symb, n_sc).

    Default (n_symb=14, n_sc=32/64): comb-2 DMRS at symbols (2, 11) -> 7.14% <= 10%.
    Supports n_symb in {1, 14, 28}, n_sc in {32, 64, 128}, custom comb, and pilot_ratio=1.0.
    """
    if n_symb <= 0 or n_sc <= 0:
        raise ValueError(f"n_symb and n_sc must be positive, got ({n_symb}, {n_sc})")

    if pilot_ratio is not None:
        if pilot_ratio >= 1.0:
            return torch.ones((n_symb, n_sc), dtype=torch.bool)
        if pilot_ratio <= 0.0:
            return torch.zeros((n_symb, n_sc), dtype=torch.bool)
        total_res = n_symb * n_sc
        n_pilots = max(1, min(total_res, int(round(pilot_ratio * total_res))))
        mask_flat = torch.zeros(total_res, dtype=torch.bool)
        idx = torch.linspace(0, total_res - 1, steps=n_pilots).round().long().unique()
        mask_flat[idx] = True
        return mask_flat.view(n_symb, n_sc)

    mask = torch.zeros((n_symb, n_sc), dtype=torch.bool)
    if pilot_symbols is None:
        if n_symb == 1:
            symbs = [0]
        elif n_symb < 14:
            symbs = [min(2, n_symb - 1)]
        else:
            symbs = []
            for base in range(0, n_symb, 14):
                for s_off in (2, 11):
                    if base + s_off < n_symb:
                        symbs.append(base + s_off)
            if not symbs:
                symbs = [0]
    else:
        symbs = [int(s) for s in pilot_symbols if 0 <= int(s) < n_symb]
        if not symbs:
            symbs = [0]

    comb = max(1, int(comb))
    offset = int(offset) % comb
    for s in symbs:
        mask[s, offset::comb] = True
    return mask


def get_slot_masks(
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    pilot_symbols: Optional[Sequence[int]] = None,
    comb: int = 2,
    offset: int = 0,
    pilot_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint partitioning (pilot_mask, data_mask) of shape (n_symb, n_sc)."""
    pilot_mask = get_dmrs_mask(
        n_symb=n_symb,
        n_sc=n_sc,
        pilot_symbols=pilot_symbols,
        comb=comb,
        offset=offset,
        pilot_ratio=pilot_ratio,
    )
    data_mask = ~pilot_mask
    return pilot_mask, data_mask


def get_custom_dmrs_mask(
    pilot_coords: Sequence[Tuple[int, int]],
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
) -> torch.Tensor:
    """Create a custom boolean DMRS mask from explicit (symbol, subcarrier) coordinates."""
    mask = torch.zeros((n_symb, n_sc), dtype=torch.bool)
    for t_idx, f_idx in pilot_coords:
        if not (0 <= t_idx < n_symb and 0 <= f_idx < n_sc):
            raise ValueError(f"Coordinate ({t_idx}, {f_idx}) out of bounds for ({n_symb}, {n_sc})")
        mask[t_idx, f_idx] = True
    return mask


def _gold_sequence_bits(length: int, c_init: int) -> torch.Tensor:
    """3GPP TS 38.211-style 31-bit Gold pseudo-random binary sequence of length `length`."""
    if length <= 0:
        return torch.zeros(0, dtype=torch.int64)
    n_c = 1600
    total = n_c + length
    x1 = [0] * total
    x2 = [0] * total
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total - 31):
        x1[n + 31] = (x1[n + 3] + x1[n]) & 1
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
    c = [(x1[n + n_c] + x2[n + n_c]) & 1 for n in range(length)]
    return torch.tensor(c, dtype=torch.int64)


def generate_dmrs(
    batch_size: int,
    num_pilots: int,
    seed: int = 2026,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Generate unit-energy pseudo-random Gold-sequence QPSK DMRS symbols (B, num_pilots)."""
    if num_pilots == 0:
        return torch.zeros((batch_size, 0), dtype=torch.complex64, device=device)

    if generator is not None:
        base_seed = int(torch.randint(0, 2**30, (1,), generator=generator).item())
    else:
        base_seed = int(seed)

    # Generate Gold-sequence QPSK pairs per batch entry
    bits_list = []
    for b in range(batch_size):
        c_init = (base_seed + 1009 * b) & 0x7FFFFFFF
        seq = _gold_sequence_bits(2 * num_pilots, c_init=c_init)
        bits_list.append(seq.view(num_pilots, 2))
    bits = torch.stack(bits_list, dim=0).to(device=device)
    return qpsk(bits).to(torch.complex64)


@dataclass
class SlotData:
    """Container for a simulated 2D OFDM slot."""
    y_grid: torch.Tensor        # (B, N_symb, N_sc) received complex64 FFT grid
    yp_sparse: torch.Tensor     # (B, N_p) LS pilot observation (y_p * pilot_tx.conj())
    y_p_raw: torch.Tensor       # (B, N_p) raw received pilot REs
    y_d_raw: torch.Tensor       # (B, N_d) raw received data REs
    pilot_tx: torch.Tensor      # (B, N_p) transmitted unit-energy QPSK DMRS
    data_tx: torch.Tensor       # (B, N_d) transmitted 16-QAM payload symbols
    x_grid: torch.Tensor        # (B, N_symb, N_sc) transmitted frequency grid
    bits: torch.Tensor          # (B, N_d, 4) transmitted coded/uncoded 16-QAM bits
    info_bits: Optional[torch.Tensor]  # (B, n_cw, K) info bits if coded=True
    coded_bits: Optional[torch.Tensor] # (B, n_cw, N_cw) coded bits if coded=True
    h_true: torch.Tensor        # (B, N_symb, N_sc) symbol-midpoint CFR ground truth
    h_time_evol: torch.Tensor   # (B, N_symb, L_taps) tap gains at symbol midpoints
    noise_var: torch.Tensor     # (B, 1) noise variance per complex subcarrier
    pilot_mask: torch.Tensor    # (N_symb, N_sc) bool
    data_mask: torch.Tensor     # (N_symb, N_sc) bool
    rx_time: torch.Tensor       # (B, N_symb * (N_sc + CP)) continuous time-domain signal
    A: Optional[float]          # PA saturation parameter (None for linear)
    alpha: float                # PA AM-PM coefficient
    fd: float                   # Normalized Doppler
    cfo: float                  # Normalized CFO
    profile: str                # Delay spread profile name
    oversample: int = DEFAULT_OVERSAMPLE
    knee: float = 2.0
    beta: float = 1.0


def _draw_slot_taps(
    batch_size: int,
    n_sc: int,
    cp: int,
    profile: str,
    fd: float,
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw multipath initial taps, integer delays, and Jakes Doppler shifts."""
    if profile not in SLOT_PROFILES:
        raise ValueError(f"Unknown channel profile {profile!r}. Choose from {list(SLOT_PROFILES.keys())}")
    pdp = torch.tensor(SLOT_PROFILES[profile], dtype=torch.float32, device=device)
    pdp = pdp / pdp.sum()
    num_taps = len(pdp)
    delays = torch.arange(num_taps, device=device, dtype=torch.int64)
    if int(delays.max().item()) > cp:
        raise ValueError(f"Max channel delay {int(delays.max().item())} exceeds CP length {cp}")

    h_init = complex_randn((batch_size, num_taps), generator=generator, device=device) * pdp.sqrt().unsqueeze(0)
    if fd > 0.0:
        angles = torch.rand((batch_size, num_taps), generator=generator, device=device) * (2.0 * math.pi)
        fd_shifts = float(fd) * torch.cos(angles)
    else:
        fd_shifts = torch.zeros((batch_size, num_taps), dtype=torch.float32, device=device)
    return h_init, delays, fd_shifts


def _apply_continuous_slot_channel(
    x_stream: torch.Tensor,
    h_init: torch.Tensor,
    delays: torch.Tensor,
    fd_shifts: torch.Tensor,
    n_symb: int,
    n_sc: int,
    cp: int,
    cfo: float = 0.0,
    noise_var: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply continuous sample-level time-varying multipath + CFO across entire slot stream.

    Sample index n runs continuously across [0, n_symb * (n_sc + cp) - 1] with zero resets
    across CP or symbol boundaries.
    """
    batch_size, total_len = x_stream.shape
    device = x_stream.device
    sym_len = n_sc + cp
    assert total_len == n_symb * sym_len

    n_grid = torch.arange(total_len, dtype=torch.float64, device=device).view(1, -1)
    y_chan = torch.zeros((batch_size, total_len), dtype=torch.complex128, device=device)
    x_stream64 = x_stream.to(torch.complex128)

    num_taps = h_init.shape[1]
    for l in range(num_taps):
        d_l = int(delays[l].item())
        h0_l = h_init[:, l : l + 1].to(torch.complex128)
        fd_l = fd_shifts[:, l : l + 1].to(torch.float64)
        phase_l = 2.0 * math.pi * fd_l * (n_grid / float(n_sc))
        h_l_n = h0_l * torch.complex(torch.cos(phase_l), torch.sin(phase_l))
        if d_l == 0:
            y_chan += h_l_n * x_stream64
        else:
            y_chan[:, d_l:] += h_l_n[:, d_l:] * x_stream64[:, : total_len - d_l]

    if cfo != 0.0:
        phase_cfo = 2.0 * math.pi * float(cfo) * (n_grid / float(n_sc))
        y_chan = y_chan * torch.complex(torch.cos(phase_cfo), torch.sin(phase_cfo))

    if noise_var is not None:
        nv64 = noise_var.to(torch.float64)
        if nv64.ndim == 1:
            nv64 = nv64.unsqueeze(-1)
        noise = complex_randn(y_chan.shape, generator=generator, device=device, dtype=torch.complex128)
        y_chan = y_chan + noise * torch.sqrt(nv64)

    # Compute exact tap evolution & CFR at each symbol's payload midpoint
    mid_samples = (
        torch.arange(n_symb, dtype=torch.float64, device=device) * float(sym_len)
        + float(cp)
        + float(n_sc) * 0.5
    )  # (N_symb,)
    phase_mid = (
        2.0 * math.pi * fd_shifts.to(torch.float64).unsqueeze(1) * (mid_samples.view(1, n_symb, 1) / float(n_sc))
    )
    h_time_evol = h_init.to(torch.complex128).unsqueeze(1) * torch.complex(
        torch.cos(phase_mid), torch.sin(phase_mid)
    )  # (B, N_symb, L_taps)

    # Static-compatible FFT across delay taps at symbol midpoint
    k_idx = torch.arange(n_sc, dtype=torch.float64, device=device).view(1, 1, 1, n_sc)
    d_idx = delays.to(torch.float64).view(1, 1, num_taps, 1)
    dft_basis = torch.complex(
        torch.cos(-2.0 * math.pi * k_idx * d_idx / float(n_sc)),
        torch.sin(-2.0 * math.pi * k_idx * d_idx / float(n_sc)),
    )
    h_true = (h_time_evol.unsqueeze(-1) * dft_basis).sum(dim=2)  # (B, N_symb, N_sc)

    return y_chan.to(torch.complex64), h_true.to(torch.complex64), h_time_evol.to(torch.complex64)


class SlotSimulator2D:
    """2D Time-Frequency OFDM Slot Simulator with sparse DMRS, Rapp PA, Doppler ICI, and CFO."""

    def __init__(
        self,
        n_symb: int = DEFAULT_N_SYMB,
        n_sc: int = DEFAULT_N_SC,
        cp: int = DEFAULT_CP,
        profile: str = "equal4",
        fd: float = 0.0,
        cfo: float = 0.0,
        pilot_ratio: Optional[float] = None,
        pilot_symbols: Optional[Sequence[int]] = None,
        comb: int = 2,
        oversample: int = DEFAULT_OVERSAMPLE,
        device: Union[str, torch.device] = "cpu",
    ):
        self.n_symb = int(n_symb)
        self.n_sc = int(n_sc)
        self.cp = int(cp)
        self.profile = profile
        self.fd = float(fd)
        self.cfo = float(cfo)
        self.pilot_ratio = pilot_ratio
        self.oversample = int(oversample)
        self.device = device

        if profile not in SLOT_PROFILES:
            raise ValueError(f"Unknown channel profile {profile!r}")

        self.pilot_mask, self.data_mask = get_slot_masks(
            n_symb=self.n_symb,
            n_sc=self.n_sc,
            pilot_symbols=pilot_symbols,
            comb=comb,
            pilot_ratio=pilot_ratio,
        )
        self.pilot_mask = self.pilot_mask.to(device=device)
        self.data_mask = self.data_mask.to(device=device)
        self.num_pilots = int(self.pilot_mask.sum().item())
        self.num_data = int(self.data_mask.sum().item())
        if self.pilot_ratio is None:
            self.pilot_ratio = float(self.num_pilots) / float(self.n_symb * self.n_sc)

    def generate_empty_grid(self, batch_size: int) -> torch.Tensor:
        """Return empty zero complex64 2D grid of shape (batch_size, n_symb, n_sc)."""
        return torch.zeros((batch_size, self.n_symb, self.n_sc), dtype=torch.complex64, device=self.device)

    def generate_slot(
        self,
        batch_size: int = 4,
        snr_db: Union[float, Tuple[float, float], None] = 18.0,
        pa_condition: Optional[str] = "mild",
        A: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: float = 1.0,
        knee: float = 2.0,
        fd: Optional[float] = None,
        cfo: Optional[float] = None,
        profile: Optional[str] = None,
        pilot_mask: Optional[torch.Tensor] = None,
        coded: bool = False,
        ldpc: Optional["LDPCMatrix"] = None,
        generator: Optional[torch.Generator] = None,
    ) -> SlotData:
        """Generate a batch of 2D OFDM slots with optional outer LDPC coding."""
        fd_val = self.fd if fd is None else float(fd)
        cfo_val = self.cfo if cfo is None else float(cfo)
        prof = self.profile if profile is None else profile
        a_val, alpha_val = _resolve_pa_params(pa_condition=pa_condition, A=A, alpha=alpha)

        p_mask = self.pilot_mask if pilot_mask is None else pilot_mask.to(device=self.device, dtype=torch.bool)
        d_mask = ~p_mask
        n_p = int(p_mask.sum().item())
        n_d = int(d_mask.sum().item())

        pilot_tx = generate_dmrs(batch_size=batch_size, num_pilots=n_p, generator=generator, device=self.device)

        info_bits: Optional[torch.Tensor] = None
        coded_bits: Optional[torch.Tensor] = None

        if n_d == 0:
            bits = torch.zeros((batch_size, 0, 4), dtype=torch.int64, device=self.device)
            data_tx = torch.zeros((batch_size, 0), dtype=torch.complex64, device=self.device)
        elif coded:
            ldpc_obj = _get_default_ldpc(128) if ldpc is None else ldpc
            total_bits_slot = n_d * 4
            n_cw = total_bits_slot // ldpc_obj.n
            rem = total_bits_slot % ldpc_obj.n
            if n_cw > 0:
                info_bits = torch.randint(
                    0, 2, (batch_size, n_cw, ldpc_obj.k), generator=generator, device=self.device, dtype=torch.int64
                )
                coded_flat, _ = encode_ldpc(info_bits.reshape(-1, ldpc_obj.k), ldpc_obj)
                coded_bits = coded_flat.view(batch_size, n_cw, ldpc_obj.n)
                flat_all = coded_bits.reshape(batch_size, n_cw * ldpc_obj.n)
            else:
                info_bits = torch.zeros((batch_size, 0, ldpc_obj.k), dtype=torch.int64, device=self.device)
                coded_bits = torch.zeros((batch_size, 0, ldpc_obj.n), dtype=torch.int64, device=self.device)
                flat_all = torch.zeros((batch_size, 0), dtype=torch.int64, device=self.device)
            if rem > 0:
                pad_bits = torch.randint(
                    0, 2, (batch_size, rem), generator=generator, device=self.device, dtype=torch.int64
                )
                flat_all = torch.cat([flat_all, pad_bits], dim=-1)
            bits = flat_all.view(batch_size, n_d, 4)
            data_tx = q16(bits).to(torch.complex64)
        else:
            bits = torch.randint(
                0, 2, (batch_size, n_d, 4), generator=generator, device=self.device, dtype=torch.int64
            )
            data_tx = q16(bits).to(torch.complex64)

        x_grid = self.generate_empty_grid(batch_size)
        if n_p > 0:
            x_grid[:, p_mask] = pilot_tx
        if n_d > 0:
            x_grid[:, d_mask] = data_tx

        # Per-symbol IFFT and oversampled Rapp PA
        x_time = torch.fft.ifft(x_grid, dim=-1, norm="ortho")
        x_pa_time = rapp_pa(
            x_time,
            knee=knee,
            A=a_val,
            alpha=alpha_val,
            beta=beta,
            oversample=self.oversample,
        )

        # CP insertion per symbol and flatten to continuous stream
        if self.cp > 0:
            x_cp = torch.cat([x_pa_time[..., -self.cp :], x_pa_time], dim=-1)
        else:
            x_cp = x_pa_time
        x_stream = x_cp.reshape(batch_size, self.n_symb * (self.n_sc + self.cp))

        # Noise variance
        if snr_db is None:
            noise_var_tensor: Optional[torch.Tensor] = None
            nv_store = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        else:
            if isinstance(snr_db, (tuple, list)):
                lo, hi = float(snr_db[0]), float(snr_db[1])
                snr_vals = lo + (hi - lo) * torch.rand((batch_size, 1), generator=generator, device=self.device)
            else:
                snr_vals = torch.full((batch_size, 1), float(snr_db), dtype=torch.float32, device=self.device)
            noise_var_tensor = 10.0 ** (-snr_vals / 10.0)
            nv_store = noise_var_tensor

        h_init, delays, fd_shifts = _draw_slot_taps(
            batch_size=batch_size,
            n_sc=self.n_sc,
            cp=self.cp,
            profile=prof,
            fd=fd_val,
            generator=generator,
            device=self.device,
        )
        rx_time, h_true, h_time_evol = _apply_continuous_slot_channel(
            x_stream=x_stream,
            h_init=h_init,
            delays=delays,
            fd_shifts=fd_shifts,
            n_symb=self.n_symb,
            n_sc=self.n_sc,
            cp=self.cp,
            cfo=cfo_val,
            noise_var=noise_var_tensor,
            generator=generator,
        )

        # Remove CP and take per-symbol FFT
        rx_sym = rx_time.view(batch_size, self.n_symb, self.n_sc + self.cp)[:, :, self.cp :]
        y_grid = torch.fft.fft(rx_sym, dim=-1, norm="ortho")

        y_p_raw = y_grid[:, p_mask] if n_p > 0 else torch.zeros((batch_size, 0), dtype=torch.complex64, device=self.device)
        y_d_raw = y_grid[:, d_mask] if n_d > 0 else torch.zeros((batch_size, 0), dtype=torch.complex64, device=self.device)
        yp_sparse = y_p_raw * pilot_tx.conj()

        return SlotData(
            y_grid=y_grid,
            yp_sparse=yp_sparse,
            y_p_raw=y_p_raw,
            y_d_raw=y_d_raw,
            pilot_tx=pilot_tx,
            data_tx=data_tx,
            x_grid=x_grid,
            bits=bits,
            info_bits=info_bits,
            coded_bits=coded_bits,
            h_true=h_true,
            h_time_evol=h_time_evol,
            noise_var=nv_store,
            pilot_mask=p_mask,
            data_mask=d_mask,
            rx_time=rx_time,
            A=a_val,
            alpha=alpha_val,
            fd=fd_val,
            cfo=cfo_val,
            profile=prof,
            oversample=self.oversample,
            knee=knee,
            beta=beta,
        )


# ===========================================================================
# Feature F14: Continuous Doppler ICI & CFO
# ===========================================================================

def simulate_slot_channel(
    batch_size: int = 1,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
    profile: str = "equal4",
    fd: float = 0.0,
    cfo: float = 0.0,
    snr_db: Optional[float] = None,
    pa_condition: Optional[str] = "clean",
    return_h_time: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Simulate a 2D slot through the continuous time-varying multipath + Doppler + CFO channel."""
    if generator is None:
        generator = torch.Generator().manual_seed(2026)
    sim = SlotSimulator2D(
        n_symb=n_symb, n_sc=n_sc, cp=cp, profile=profile, fd=fd, cfo=cfo
    )
    slot = sim.generate_slot(
        batch_size=batch_size,
        snr_db=snr_db,
        pa_condition=pa_condition,
        generator=generator,
    )
    if return_h_time:
        return slot.h_time_evol
    return slot.y_grid


def measure_ici_leakage(
    fd: float = 0.05,
    n_symb: int = 1,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
    profile: str = "equal4",
    seed: int = 42,
) -> float:
    """Measure normalized inter-carrier interference (ICI) power leakage under continuous Doppler."""
    gen = torch.Generator().manual_seed(seed)
    k0 = n_sc // 2
    x_grid = torch.zeros((4, n_symb, n_sc), dtype=torch.complex64)
    x_grid[:, :, k0] = 1.0 + 0.0j
    x_time = torch.fft.ifft(x_grid, dim=-1, norm="ortho")
    x_cp = torch.cat([x_time[..., -cp:], x_time], dim=-1)
    x_stream = x_cp.reshape(4, n_symb * (n_sc + cp))

    h_init, delays, fd_shifts = _draw_slot_taps(4, n_sc, cp, profile, fd, gen, "cpu")
    rx_time, _, _ = _apply_continuous_slot_channel(
        x_stream=x_stream,
        h_init=h_init,
        delays=delays,
        fd_shifts=fd_shifts,
        n_symb=n_symb,
        n_sc=n_sc,
        cp=cp,
        cfo=0.0,
        noise_var=None,
    )
    rx_sym = rx_time.view(4, n_symb, n_sc + cp)[:, :, cp:]
    y_grid = torch.fft.fft(rx_sym, dim=-1, norm="ortho")
    total_pow = (y_grid.abs() ** 2).sum(dim=-1)
    sig_pow = y_grid[..., k0].abs() ** 2
    leakage = ((total_pow - sig_pow) / total_pow.clamp(min=1e-12)).mean().item()
    return float(leakage)


def measure_cfo_phase_drift(
    cfo: float = 0.04,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
    seed: int = 42,
) -> float:
    """Measure normalized CFO from continuous sample-level inter-symbol phase accumulation."""
    gen = torch.Generator().manual_seed(seed)
    x_grid = torch.ones((1, n_symb, n_sc), dtype=torch.complex64)
    x_time = torch.fft.ifft(x_grid, dim=-1, norm="ortho")
    x_cp = torch.cat([x_time[..., -cp:], x_time], dim=-1)
    x_stream = x_cp.reshape(1, n_symb * (n_sc + cp))

    h_init = torch.ones((1, 1), dtype=torch.complex64)
    delays = torch.zeros(1, dtype=torch.int64)
    fd_shifts = torch.zeros((1, 1), dtype=torch.float32)

    rx_time, _, _ = _apply_continuous_slot_channel(
        x_stream=x_stream,
        h_init=h_init,
        delays=delays,
        fd_shifts=fd_shifts,
        n_symb=n_symb,
        n_sc=n_sc,
        cp=cp,
        cfo=cfo,
        noise_var=None,
        generator=gen,
    )
    rx_sym = rx_time.view(1, n_symb, n_sc + cp)[:, :, cp:]
    y_grid = torch.fft.fft(rx_sym, dim=-1, norm="ortho")
    # Phase advance between consecutive symbols separated by (n_sc + cp) samples
    inter_sym = (y_grid[:, 1:, :] * y_grid[:, :-1, :].conj()).sum()
    sym_dur = float(n_sc + cp) / float(n_sc)
    est_cfo = float(torch.angle(inter_sym).item()) / (2.0 * math.pi * sym_dur)
    return est_cfo


def estimate_cfo_cp_tensor(
    rx_time: torch.Tensor,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
) -> torch.Tensor:
    """Batched CP-correlation fractional CFO estimator returning (B,) tensor."""
    if rx_time.dim() == 1:
        rx_time = rx_time.unsqueeze(0)
    bsz = rx_time.shape[0]
    sym_view = rx_time.view(bsz, n_symb, n_sc + cp)
    cp_head = sym_view[:, :, :cp]
    cp_tail = sym_view[:, :, n_sc : n_sc + cp]
    corr = (cp_head.conj() * cp_tail).sum(dim=(-2, -1))
    return torch.angle(corr) / (2.0 * math.pi)


def compensate_cfo_time(
    rx_time: torch.Tensor,
    cfo_hat: Union[float, torch.Tensor],
    n_sc: int = DEFAULT_N_SC,
) -> torch.Tensor:
    """Compensate continuous sample-level CFO phase exp(-j 2*pi * cfo_hat * n / n_sc)."""
    if rx_time.dim() == 1:
        rx_time = rx_time.unsqueeze(0)
    total_len = rx_time.shape[-1]
    n_grid = torch.arange(total_len, dtype=torch.float64, device=rx_time.device).view(1, -1)
    if isinstance(cfo_hat, torch.Tensor):
        cfo_t = cfo_hat.to(torch.float64).view(-1, 1)
    else:
        cfo_t = torch.full((rx_time.shape[0], 1), float(cfo_hat), dtype=torch.float64, device=rx_time.device)
    phase = -2.0 * math.pi * cfo_t * (n_grid / float(n_sc))
    comp = rx_time.to(torch.complex128) * torch.complex(torch.cos(phase), torch.sin(phase))
    return comp.to(rx_time.dtype)


def _synchronize_slot(slot: SlotData) -> SlotData:
    """Estimate fractional CFO from received CP samples. Rephase the reference CFR for scoring only."""
    batch_size, n_symb, n_sc = slot.y_grid.shape
    cp = slot.rx_time.shape[-1] // n_symb - n_sc
    if cp == 0:
        return slot
    cfo_hat = estimate_cfo_cp_tensor(slot.rx_time, n_symb=n_symb, n_sc=n_sc, cp=cp)
    rx = compensate_cfo_time(slot.rx_time, cfo_hat, n_sc=n_sc)
    payload = rx.view(batch_size, n_symb, n_sc + cp)[:, :, cp:]
    y = torch.fft.fft(payload, dim=-1, norm="ortho")
    yp = y[:, slot.pilot_mask]
    mid = torch.arange(n_symb, dtype=torch.float64, device=y.device) * (n_sc + cp) + cp + n_sc / 2
    phase = (slot.cfo - cfo_hat.double())[:, None] * mid * (2.0 * math.pi / n_sc)
    rotation = torch.polar(torch.ones_like(phase), phase).to(y.dtype).unsqueeze(-1)
    return replace(slot, rx_time=rx, y_grid=y, yp_sparse=yp * slot.pilot_tx.conj(),
                   y_p_raw=yp, y_d_raw=y[:, slot.data_mask], h_true=slot.h_true * rotation,
                   h_time_evol=slot.h_time_evol * rotation)


def estimate_cfo_cp(
    cfo_true: float = 0.03,
    rx_time: Optional[torch.Tensor] = None,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
    snr_db: float = 30.0,
    seed: int = 42,
) -> float:
    """Estimate normalized CFO via cyclic-prefix correlation."""
    if rx_time is None:
        gen = torch.Generator().manual_seed(seed)
        sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=cp, fd=0.0, cfo=cfo_true)
        slot = sim.generate_slot(batch_size=4, snr_db=snr_db, pa_condition="clean", generator=gen)
        rx_time = slot.rx_time
    est = estimate_cfo_cp_tensor(rx_time, n_symb=n_symb, n_sc=n_sc, cp=cp)
    return float(est.mean().item())


def compute_doppler_spectrum(
    speed_kmh: float = 500.0,
    fc: float = 3.5e9,
    num_freqs: int = 129,
) -> Dict[str, Any]:
    """Compute classical Jakes 2D isotropic Doppler power spectral density."""
    v_ms = float(speed_kmh) / 3.6
    fd_hz = float(fc) * v_ms / SPEED_OF_LIGHT
    fd_norm = fd_hz / DEFAULT_SCS_HZ
    freqs = torch.linspace(-0.99 * fd_hz, 0.99 * fd_hz, steps=num_freqs, dtype=torch.float64)
    ratio = freqs / max(fd_hz, 1e-12)
    psd = 1.0 / (math.pi * max(fd_hz, 1e-12) * torch.sqrt(torch.clamp(1.0 - ratio ** 2, min=1e-6)))
    return {
        "speed_kmh": float(speed_kmh),
        "fc": float(fc),
        "fd_hz": float(fd_hz),
        "fd_norm": float(fd_norm),
        "freqs": freqs,
        "psd": psd,
    }


def check_phase_continuity(
    cfo: float = 0.04,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
) -> bool:
    """Verify continuous sample-level CFO phase evolution across CP and symbol boundaries."""
    total_len = n_symb * (n_sc + cp)
    x_stream = torch.ones((1, total_len), dtype=torch.complex64)
    h_init = torch.ones((1, 1), dtype=torch.complex64)
    delays = torch.zeros(1, dtype=torch.int64)
    fd_shifts = torch.zeros((1, 1), dtype=torch.float32)

    rx_time, _, _ = _apply_continuous_slot_channel(
        x_stream=x_stream,
        h_init=h_init,
        delays=delays,
        fd_shifts=fd_shifts,
        n_symb=n_symb,
        n_sc=n_sc,
        cp=cp,
        cfo=cfo,
        noise_var=None,
    )
    step_phase = torch.angle(rx_time[0, 1:] * rx_time[0, :-1].conj())
    expected_step = 2.0 * math.pi * float(cfo) / float(n_sc)
    max_jump = float((step_phase - expected_step).abs().max().item())
    return bool(max_jump < 1e-4)


# ===========================================================================
# Feature F15: Native Outer Channel Coding (Rate-1/2 PEG/IRA-LDPC)
# ===========================================================================

def _build_girth6_rate_half_h(codeword_len: int = 128, seed: int = 2026) -> torch.Tensor:
    """Construct a deterministic Rate-1/2 (M x N) parity-check matrix H = [H_u | H_p]
    with girth >= 6, full-rank lower-bidiagonal+tail H_p (all cols weight >= 2), and
    PEG-placed info columns H_u (weight 3).
    """
    if codeword_len <= 0 or codeword_len % 2 != 0:
        raise ValueError(f"codeword_len must be a positive even integer, got {codeword_len}")
    n = int(codeword_len)
    k = n // 2
    m = n - k

    h = [[0] * n for _ in range(m)]
    row_deg = [0] * m
    cn_nbrs: List[List[int]] = [[] for _ in range(m)]
    vn_nbrs: List[List[int]] = [[] for _ in range(n)]

    def add_edge(r: int, c: int) -> None:
        h[r][c] = 1
        row_deg[r] += 1
        cn_nbrs[r].append(c)
        vn_nbrs[c].append(r)

    # 1. Invertible H_p on columns [k .. n-1] with every column having weight >= 2:
    #    - Lower bidiagonal for c = 0..m-2: rows (c, c+1)
    #    - Last column c = m-1: 3 rows (0, m//2, m-1) -> odd count => det(H_p) = 1 mod 2!
    for c_local in range(m - 1):
        add_edge(c_local, k + c_local)
        add_edge(c_local + 1, k + c_local)
    tail_rows = sorted({0, m // 2, m - 1})
    if len(tail_rows) % 2 == 0 and m >= 3:
        # Ensure odd number of 1s in the final column so det(H_p) == 1 (mod 2)
        for cand in range(1, m - 1):
            if cand not in tail_rows:
                tail_rows.append(cand)
                break
    for r in tail_rows:
        add_edge(r, k + m - 1)

    # 2. Deterministic PEG placement for info columns [0 .. k-1], weight d_v = 3
    #    Avoid 4-cycles (distance-2 check nodes in bipartite graph) whenever possible.
    gen = torch.Generator().manual_seed(seed)
    dv = 3 if m >= 6 else max(1, m // 2)
    for c in range(k):
        for edge_idx in range(dv):
            if edge_idx == 0:
                # Choose minimum-degree check node deterministically
                min_deg = min(row_deg)
                cands = [r for r in range(m) if row_deg[r] == min_deg and h[r][c] == 0]
                pick_idx = int(torch.randint(0, len(cands), (1,), generator=gen).item())
                add_edge(cands[pick_idx], c)
            else:
                # Forbidden checks that would close a 4-cycle with existing neighbors of c
                forbidden = set(vn_nbrs[c])
                for r0 in vn_nbrs[c]:
                    for c_other in cn_nbrs[r0]:
                        if c_other == c:
                            continue
                        for r1 in vn_nbrs[c_other]:
                            forbidden.add(r1)
                cands = [r for r in range(m) if r not in forbidden]
                if not cands:
                    cands = [r for r in range(m) if h[r][c] == 0]
                min_deg = min(row_deg[r] for r in cands)
                best = [r for r in cands if row_deg[r] == min_deg]
                pick_idx = int(torch.randint(0, len(best), (1,), generator=gen).item())
                add_edge(best[pick_idx], c)

    return torch.tensor(h, dtype=torch.int64)


def _compute_systematic_generator(h_matrix: torch.Tensor) -> torch.Tensor:
    """Compute exact systematic GF(2) generator G = [I_k | P] from H = [H_u | H_p]."""
    m, n = h_matrix.shape
    k = n - m
    h_u = h_matrix[:, :k].clone().tolist()
    h_p = h_matrix[:, k:].clone().tolist()

    # Gauss-Jordan elimination on [H_p | H_u] over GF(2) to obtain [I_m | P^T]
    aug = [h_p[r][:] + h_u[r][:] for r in range(m)]
    for col in range(m):
        pivot = None
        for r in range(col, m):
            if aug[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Parity submatrix H_p is singular over GF(2).")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(m):
            if r != col and aug[r][col] == 1:
                aug[r] = [a ^ b for a, b in zip(aug[r], aug[col])]

    p_t = [aug[r][m:] for r in range(m)]  # m x k
    p_mat = [[p_t[r][c] for r in range(m)] for c in range(k)]  # k x m
    g_rows = []
    for c in range(k):
        id_row = [1 if j == c else 0 for j in range(k)]
        g_rows.append(id_row + p_mat[c])
    return torch.tensor(g_rows, dtype=torch.int64)


class LDPCMatrix:
    """Self-contained Rate-1/2 PEG/IRA-LDPC parity-check & systematic generator container."""

    def __init__(self, codeword_len: int = 128, rate: float = 0.5, seed: int = 2026):
        if abs(rate - 0.5) > 1e-6:
            raise ValueError("Only Rate-1/2 LDPC is supported.")
        self.n = int(codeword_len)
        self.k = self.n // 2
        self.m = self.n - self.k
        self.rate = 0.5
        self.seed = int(seed)
        self.h_matrix = _build_girth6_rate_half_h(codeword_len=self.n, seed=self.seed)
        self.g_matrix = _compute_systematic_generator(self.h_matrix)

        # Precompute vectorized edge-index tables for fast Min-Sum BP
        edges_r, edges_c = torch.nonzero(self.h_matrix, as_tuple=True)
        self._edge_cn = edges_r.long()  # (E,)
        self._edge_vn = edges_c.long()  # (E,)
        self._num_edges = int(edges_r.numel())

        # Check-node padded adjacency: (M, dc_max)
        cn_lists = [torch.nonzero(self.h_matrix[r], as_tuple=False).squeeze(-1).tolist() for r in range(self.m)]
        dc_max = max(len(x) for x in cn_lists)
        cn_pad_vn = torch.full((self.m, dc_max), self.n, dtype=torch.long)  # dummy index = n
        cn_pad_edge = torch.full((self.m, dc_max), self._num_edges, dtype=torch.long)  # dummy edge = E
        edge_lookup = {(int(r), int(c)): idx for idx, (r, c) in enumerate(zip(edges_r.tolist(), edges_c.tolist()))}
        cn_edge_valid = torch.zeros((self.m, dc_max), dtype=torch.bool)
        for r, cols in enumerate(cn_lists):
            for j, c in enumerate(cols):
                cn_pad_vn[r, j] = c
                cn_pad_edge[r, j] = edge_lookup[(r, c)]
                cn_edge_valid[r, j] = True
        self._cn_pad_vn = cn_pad_vn
        self._cn_pad_edge = cn_pad_edge
        self._cn_edge_valid = cn_edge_valid
        self._dc_max = dc_max

        # Reverse mapping from (M, dc_max) valid entries back to edge index 0..E-1
        self._valid_cn_flat = torch.nonzero(cn_edge_valid.reshape(-1), as_tuple=False).squeeze(-1)
        self._valid_edge_idx = cn_pad_edge.reshape(-1)[self._valid_cn_flat]
        # Build permutation so scatter order matches _edge_cn / _edge_vn
        inv_perm = torch.empty(self._num_edges, dtype=torch.long)
        inv_perm[self._valid_edge_idx] = torch.arange(self._num_edges, dtype=torch.long)
        self._edge_to_cn_flat = self._valid_cn_flat[inv_perm]


@lru_cache(maxsize=16)
def _get_default_ldpc(codeword_len: int = 128) -> LDPCMatrix:
    return LDPCMatrix(codeword_len=codeword_len, rate=0.5, seed=2026)


def _get_ldpc_from_h(h_mat: torch.Tensor) -> LDPCMatrix:
    """Resolve an LDPCMatrix matching h_mat (fast path for default H, custom wrap otherwise)."""
    m, n = h_mat.shape
    default_obj = _get_default_ldpc(n) if n % 2 == 0 and m == n // 2 else None
    if default_obj is not None and torch.equal(default_obj.h_matrix.cpu(), h_mat.cpu()):
        return default_obj

    obj = LDPCMatrix.__new__(LDPCMatrix)
    obj.m, obj.n = int(m), int(n)
    obj.k = obj.n - obj.m
    obj.rate = float(obj.k) / float(obj.n)
    obj.seed = 0
    obj.h_matrix = h_mat.cpu().long()
    obj.g_matrix = None
    edges_r, edges_c = torch.nonzero(obj.h_matrix, as_tuple=True)
    obj._edge_cn = edges_r.long()
    obj._edge_vn = edges_c.long()
    obj._num_edges = int(edges_r.numel())
    cn_lists = [torch.nonzero(obj.h_matrix[r], as_tuple=False).squeeze(-1).tolist() for r in range(obj.m)]
    dc_max = max(len(x) for x in cn_lists)
    cn_pad_vn = torch.full((obj.m, dc_max), obj.n, dtype=torch.long)
    cn_pad_edge = torch.full((obj.m, dc_max), obj._num_edges, dtype=torch.long)
    edge_lookup = {(int(r), int(c)): idx for idx, (r, c) in enumerate(zip(edges_r.tolist(), edges_c.tolist()))}
    cn_edge_valid = torch.zeros((obj.m, dc_max), dtype=torch.bool)
    for r, cols in enumerate(cn_lists):
        for j, c in enumerate(cols):
            cn_pad_vn[r, j] = c
            cn_pad_edge[r, j] = edge_lookup[(r, c)]
            cn_edge_valid[r, j] = True
    obj._cn_pad_vn = cn_pad_vn
    obj._cn_pad_edge = cn_pad_edge
    obj._cn_edge_valid = cn_edge_valid
    obj._dc_max = dc_max
    obj._valid_cn_flat = torch.nonzero(cn_edge_valid.reshape(-1), as_tuple=False).squeeze(-1)
    obj._valid_edge_idx = cn_pad_edge.reshape(-1)[obj._valid_cn_flat]
    inv_perm = torch.empty(obj._num_edges, dtype=torch.long)
    inv_perm[obj._valid_edge_idx] = torch.arange(obj._num_edges, dtype=torch.long)
    obj._edge_to_cn_flat = obj._valid_cn_flat[inv_perm]
    return obj


def encode_ldpc(
    u: torch.Tensor,
    ldpc: Optional[LDPCMatrix] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode binary information vector(s) u of shape (..., K) into systematic LDPC codewords (..., 2*K).

    Returns (c, h_matrix) where c is int64 in {0, 1} satisfying H c^T == 0 (mod 2).
    """
    if not isinstance(u, torch.Tensor):
        u = torch.as_tensor(u)
    if u.numel() == 0 or u.ndim == 0 or u.shape[-1] == 0:
        raise ValueError("Input payload tensor `u` must be non-empty with shape (..., K).")

    k = int(u.shape[-1])
    ldpc_obj = _get_default_ldpc(2 * k) if ldpc is None else ldpc
    if k != ldpc_obj.k:
        raise ValueError(f"Expected info length K={ldpc_obj.k}, got {k}")

    orig_shape = u.shape[:-1]
    u_2d = u.reshape(-1, k).long()
    g_mat = ldpc_obj.g_matrix.to(device=u.device, dtype=torch.float32)
    c_2d = torch.remainder(torch.matmul(u_2d.float(), g_mat), 2.0).round().long()
    c = c_2d.view(*orig_shape, ldpc_obj.n)
    return c, ldpc_obj.h_matrix.to(device=u.device)


def decode_min_sum(
    llrs: torch.Tensor,
    h_mat: torch.Tensor,
    max_iter: int = 25,
    alpha: float = 0.75,
    return_iters: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """Vectorized Normalized Min-Sum LDPC soft decoder.

    LLR Sign Convention (matching calibrated_16qam_llrs):
      llr > 0  =>  bit = 1
      llr < 0  =>  bit = 0

    Performs syndrome early stopping per iteration and returns (u_hat, iters) if return_iters=True.
    """
    if llrs.numel() == 0:
        raise ValueError("Empty LLR input to decode_min_sum.")

    ldpc_obj = _get_ldpc_from_h(h_mat)
    m, n, k = ldpc_obj.m, ldpc_obj.n, ldpc_obj.k
    orig_shape = llrs.shape[:-1]
    if llrs.dim() == 1:
        llrs_2d = llrs.unsqueeze(0)
    else:
        llrs_2d = llrs.reshape(-1, n)

    bsz = llrs_2d.shape[0]
    device = llrs.device

    # Convert to standard BP sign-0 convention: L_ch > 0 => bit = 0, L_ch < 0 => bit = 1
    l_ch = -llrs_2d.to(torch.float32)
    h_t = h_mat.to(device=device, dtype=torch.float32).t()

    cn_pad_vn = ldpc_obj._cn_pad_vn.to(device=device)        # (M, dc_max)
    cn_edge_valid = ldpc_obj._cn_edge_valid.to(device=device) # (M, dc_max)
    edge_cn = ldpc_obj._edge_cn.to(device=device)            # (E,)
    edge_vn = ldpc_obj._edge_vn.to(device=device)            # (E,)
    edge_to_cn_flat = ldpc_obj._edge_to_cn_flat.to(device=device)
    num_edges = ldpc_obj._num_edges
    dc_max = ldpc_obj._dc_max

    # Initialize variable-to-check edge messages from channel LLRs
    v2c_edge = l_ch[:, edge_vn]  # (B, E)
    c2v_edge = torch.zeros((bsz, num_edges), dtype=torch.float32, device=device)
    l_post = l_ch.clone()

    # Pre-allocated padding mask tensors for check-node min1/min2
    inf_pad = torch.full((bsz, m, dc_max), 1e9, dtype=torch.float32, device=device)
    valid_exp = cn_edge_valid.unsqueeze(0).expand(bsz, m, dc_max)
    vn_idx_scatter = edge_vn.unsqueeze(0).expand(bsz, num_edges)

    iters_used = 0
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    c_hat_final = (l_ch < 0.0).long()

    for it in range(1, int(max_iter) + 1):
        iters_used = it

        # Populate padded (B, M, dc_max) tensor with current v2c messages
        v2c_grid = torch.zeros((bsz, m * dc_max), dtype=torch.float32, device=device)
        v2c_grid[:, edge_to_cn_flat] = v2c_edge
        v2c_grid = v2c_grid.view(bsz, m, dc_max)

        abs_v = torch.where(valid_exp, v2c_grid.abs(), inf_pad)
        sgn_v = torch.where(valid_exp, torch.where(v2c_grid >= 0.0, 1.0, -1.0), torch.ones_like(v2c_grid))

        # Total sign product per check node
        tot_sgn = torch.prod(sgn_v, dim=-1, keepdim=True)  # (B, M, 1)
        ext_sgn = tot_sgn * sgn_v  # (B, M, dc_max)

        # Top-2 smallest magnitudes per check node
        top2_vals, top2_idx = torch.topk(abs_v, k=2, dim=-1, largest=False)
        min1 = top2_vals[..., 0:1]  # (B, M, 1)
        min2 = top2_vals[..., 1:2]  # (B, M, 1)
        idx1 = top2_idx[..., 0:1]   # (B, M, 1)
        pos = torch.arange(dc_max, device=device).view(1, 1, dc_max)
        ext_mag = torch.where(pos == idx1, min2, min1)

        c2v_grid = float(alpha) * ext_sgn * ext_mag
        c2v_edge_new = c2v_grid.view(bsz, m * dc_max)[:, edge_to_cn_flat]

        # Variable-node posterior update
        sum_c2v = torch.zeros((bsz, n), dtype=torch.float32, device=device)
        sum_c2v.scatter_add_(1, vn_idx_scatter, c2v_edge_new)
        l_post_new = l_ch + sum_c2v

        # Freeze already-converged batch elements
        if done.any():
            keep = (~done).unsqueeze(-1)
            l_post = torch.where(keep, l_post_new, l_post)
            c2v_edge = torch.where(keep, c2v_edge_new, c2v_edge)
        else:
            l_post = l_post_new
            c2v_edge = c2v_edge_new

        c_hat = (l_post < 0.0).long()
        syndrome = torch.remainder(torch.matmul(c_hat.float(), h_t), 2.0)
        newly_done = (syndrome == 0).all(dim=-1)
        c_hat_final = torch.where(done.unsqueeze(-1), c_hat_final, c_hat)
        done = done | newly_done
        if bool(done.all().item()):
            break

        # Update variable-to-check extrinsic messages for next iteration
        v2c_edge = l_post[:, edge_vn] - c2v_edge

    u_hat = c_hat_final[:, :k]
    if llrs.dim() == 1:
        u_hat = u_hat.squeeze(0)
    else:
        u_hat = u_hat.view(*orig_shape, k)

    if return_iters:
        return u_hat, int(iters_used)
    return u_hat


def correct_single_bit(seed: int = 42) -> bool:
    """Verify Normalized Min-Sum recovers a single flipped bit in 1 iteration."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 2, (1, 64), generator=gen, dtype=torch.int64)
    c, h_mat = encode_ldpc(u)
    llrs = (2.0 * c.float() - 1.0) * 20.0
    # Test multiple positions (info, parity, and tail)
    for pos in (0, 31, 64, 95, 127):
        corrupted = llrs.clone()
        corrupted[0, pos] = -corrupted[0, pos]
        u_hat, iters = decode_min_sum(corrupted, h_mat, max_iter=20, return_iters=True)
        if not torch.equal(u_hat, u) or iters != 1:
            return False
    return True


def measure_coding_gain(seed: int = 42) -> float:
    """Measure SNR coding gain (dB) of Rate-1/2 LDPC soft decoding over uncoded 16-QAM
    at target bit error rate ~ 1e-3 over AWGN/Rayleigh calibrated LLRs.
    """
    gen = torch.Generator().manual_seed(seed)
    ldpc = _get_default_ldpc(128)
    n_blocks = 80
    u = torch.randint(0, 2, (n_blocks, ldpc.k), generator=gen, dtype=torch.int64)
    c, h_mat = encode_ldpc(u, ldpc=ldpc)
    bits_16 = c.view(n_blocks, 32, 4)
    x_tx = q16(bits_16)

    # Same noise realization across SNR sweep for monotonic estimation
    noise = complex_randn(x_tx.shape, generator=gen, dtype=torch.complex64)

    def _eval_coded_ber(snr_db: float) -> float:
        nv = 10.0 ** (-snr_db / 10.0)
        z = x_tx + noise * math.sqrt(nv)
        llrs = calibrated_16qam_llrs(z, torch.full_like(z.real, nv), method="exact").view(n_blocks, 128)
        u_hat = decode_min_sum(llrs, h_mat, max_iter=25)
        return float((u_hat != u).float().mean().item())

    def _eval_uncoded_ber(snr_db: float) -> float:
        nv = 10.0 ** (-snr_db / 10.0)
        z = x_tx + noise * math.sqrt(nv)
        hard = hard_bits_16qam(z)
        return float((hard != bits_16).float().mean().item())

    snr_grid = [float(s) for s in range(4, 21)]
    target_ber = 1e-3

    snr_coded = None
    for s in snr_grid:
        if _eval_coded_ber(s) <= target_ber:
            snr_coded = s
            break
    if snr_coded is None:
        snr_coded = snr_grid[-1]

    snr_uncoded = None
    for s in snr_grid:
        if _eval_uncoded_ber(s) <= target_ber:
            snr_uncoded = s
            break
    if snr_uncoded is None:
        snr_uncoded = snr_grid[-1] + 1.0

    return float(snr_uncoded - snr_coded)


# ===========================================================================
# Feature F16: 2D Classical Baselines & EMFloatFormer2D
# ===========================================================================

def compute_2d_correlation(
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    fd: float = 0.03,
    cp: int = DEFAULT_CP,
    profile: str = "equal4",
) -> torch.Tensor:
    """Compute separable 2D time-frequency channel correlation matrix R_2D = R_time (x) R_freq
    of shape (n_symb * n_sc, n_symb * n_sc).
    """
    if profile not in SLOT_PROFILES:
        raise ValueError(f"Unknown channel profile {profile!r}")
    t_idx = torch.arange(n_symb, dtype=torch.float64)
    dt = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).abs()
    sym_dur = float(n_sc + cp) / float(n_sc)
    if fd > 0.0:
        r_t = torch.special.bessel_j0(2.0 * math.pi * float(fd) * dt * sym_dur)
    else:
        r_t = torch.ones((n_symb, n_symb), dtype=torch.float64)

    pdp = torch.tensor(SLOT_PROFILES[profile], dtype=torch.float64)
    pdp = pdp / pdp.sum()
    delays = torch.arange(len(pdp), dtype=torch.float64).view(-1, 1, 1)
    k_idx = torch.arange(n_sc, dtype=torch.float64)
    dk = k_idx.unsqueeze(1) - k_idx.unsqueeze(0)
    ang = -2.0 * math.pi * dk.unsqueeze(0) * delays / float(n_sc)
    r_f = (pdp.view(-1, 1, 1) * torch.complex(torch.cos(ang), torch.sin(ang))).sum(dim=0)

    r_2d = torch.kron(r_t.to(torch.complex128), r_f)
    return r_2d.to(torch.complex64)


def _solve_2d_lmmse_core(
    yp_sparse: torch.Tensor,
    pilot_mask: torch.Tensor,
    noise_var: Union[float, torch.Tensor],
    n_symb: int,
    n_sc: int,
    cp: int = DEFAULT_CP,
    fd: float = 0.03,
    profile: str = "equal4",
    A: Optional[float] = None,
    alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve 2D-LMMSE channel interpolation from sparse DMRS observations yp_sparse (B, N_p)."""
    bsz = yp_sparse.shape[0]
    device = yp_sparse.device
    r_2d = compute_2d_correlation(n_symb=n_symb, n_sc=n_sc, fd=fd, cp=cp, profile=profile).to(
        device=device, dtype=torch.complex128
    )
    p_idx = torch.nonzero(pilot_mask.reshape(-1), as_tuple=False).squeeze(-1)
    n_p = int(p_idx.numel())
    if n_p == 0:
        zeros = torch.zeros((bsz, n_symb, n_sc), dtype=torch.complex64, device=device)
        return zeros, zeros

    r_gp = r_2d[:, p_idx]              # (N_grid, N_p)
    r_pp = r_2d[p_idx][:, p_idx]       # (N_p, N_p)

    gamma, sigma_d2, _ = bussgang_params(A=A, alpha=alpha)
    gamma_abs2 = abs(gamma) ** 2 + 1e-12

    if isinstance(noise_var, torch.Tensor):
        nv_vec = noise_var.reshape(bsz).to(device=device, dtype=torch.float64)
    else:
        nv_vec = torch.full((bsz,), float(noise_var), dtype=torch.float64, device=device)

    reg = torch.clamp((nv_vec + float(sigma_d2)) / gamma_abs2, min=1e-6)  # (B,)
    eye_p = torch.eye(n_p, dtype=torch.complex128, device=device).unsqueeze(0)
    a_mat = r_pp.unsqueeze(0) + reg.view(bsz, 1, 1) * eye_p  # (B, N_p, N_p)
    # Solve (R_pp + reg * I) alpha = yp_sparse -> h_eff = R_gp @ alpha
    alpha_vec = torch.linalg.solve(a_mat, yp_sparse.to(torch.complex128).unsqueeze(-1))  # (B, N_p, 1)
    h_eff = torch.matmul(r_gp.unsqueeze(0), alpha_vec).squeeze(-1)  # (B, N_grid)
    h_eff = h_eff.view(bsz, n_symb, n_sc)
    gamma_c = complex(gamma)
    h_phys = h_eff / gamma_c
    return h_phys.to(torch.complex64), h_eff.to(torch.complex64)


def lmmse_2d_interpolate(
    batch_size: int = 2,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    cp: int = DEFAULT_CP,
    fd: float = 0.03,
    profile: str = "equal4",
    noise_var: Union[float, torch.Tensor] = 0.01,
    yp_sparse: Optional[torch.Tensor] = None,
    pilot_mask: Optional[torch.Tensor] = None,
    A: Optional[float] = None,
    alpha: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """Estimate 2D channel frequency response (B, N_symb, N_sc) via 2D-LMMSE from sparse DMRS."""
    if yp_sparse is None:
        gen = torch.Generator().manual_seed(seed)
        sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=cp, profile=profile, fd=fd)
        snr_db = -10.0 * math.log10(max(float(noise_var), 1e-12))
        slot = sim.generate_slot(batch_size=batch_size, snr_db=snr_db, pa_condition="clean", generator=gen)
        yp_sparse = slot.yp_sparse
        pilot_mask = slot.pilot_mask
        noise_var = slot.noise_var
    elif pilot_mask is None:
        pilot_mask = get_dmrs_mask(n_symb=n_symb, n_sc=n_sc)

    h_phys, _ = _solve_2d_lmmse_core(
        yp_sparse=yp_sparse,
        pilot_mask=pilot_mask,
        noise_var=noise_var,
        n_symb=n_symb,
        n_sc=n_sc,
        cp=cp,
        fd=fd,
        profile=profile,
        A=A,
        alpha=alpha,
    )
    return h_phys


def compare_1d_vs_2d_lmmse(fd: float = 0.03, snr_db: float = 20.0, seed: int = 42) -> Tuple[float, float]:
    """Compare frequency-only 1D-LMMSE (nearest pilot copy in time) vs 2D-LMMSE under Doppler."""
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=fd)
    slot = sim.generate_slot(batch_size=8, snr_db=snr_db, pa_condition="clean", generator=gen)

    # 2D-LMMSE
    h_2d = lmmse_2d_interpolate(
        batch_size=8,
        n_symb=14,
        n_sc=32,
        fd=fd,
        profile="equal4",
        noise_var=slot.noise_var,
        yp_sparse=slot.yp_sparse,
        pilot_mask=slot.pilot_mask,
    )

    # 1D frequency-only LMMSE at each pilot symbol + nearest-symbol copy in time
    r_2d = compute_2d_correlation(n_symb=14, n_sc=32, fd=0.0, profile="equal4").to(torch.complex128)
    r_f = r_2d[:32, :32]  # frequency covariance
    p_cols = torch.nonzero(slot.pilot_mask[2], as_tuple=False).squeeze(-1)
    r_fp = r_f[:, p_cols]
    r_pp = r_f[p_cols][:, p_cols]
    nv_scalar = float(slot.noise_var.mean().item())
    reg = max(1e-6, nv_scalar)
    a_1d = r_pp + reg * torch.eye(len(p_cols), dtype=torch.complex128)

    # Extract pilot observations for symbol 2 and symbol 11
    n_p_sym = len(p_cols)
    yp_s2 = slot.yp_sparse[:, :n_p_sym].to(torch.complex128)
    yp_s11 = slot.yp_sparse[:, n_p_sym:].to(torch.complex128)
    alpha_s2 = torch.linalg.solve(a_1d.unsqueeze(0), yp_s2.unsqueeze(-1))
    alpha_s11 = torch.linalg.solve(a_1d.unsqueeze(0), yp_s11.unsqueeze(-1))
    h_s2 = torch.matmul(r_fp.unsqueeze(0), alpha_s2).squeeze(-1).to(torch.complex64)
    h_s11 = torch.matmul(r_fp.unsqueeze(0), alpha_s11).squeeze(-1).to(torch.complex64)

    h_1d = torch.empty_like(h_2d)
    for t in range(14):
        h_1d[:, t, :] = h_s2 if abs(t - 2) <= abs(t - 11) else h_s11

    den = (slot.h_true.abs() ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)
    nmse_1d = float((((h_1d - slot.h_true).abs() ** 2).sum(dim=(-2, -1)) / den).mean().item())
    nmse_2d = float((((h_2d - slot.h_true).abs() ** 2).sum(dim=(-2, -1)) / den).mean().item())
    return nmse_1d, nmse_2d


def _bussgang_equalize_2d(
    y_grid: torch.Tensor,
    h_phys: torch.Tensor,
    h_eff: torch.Tensor,
    noise_var: torch.Tensor,
    sigma_d2: float,
    llr_method: str = "exact",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D per-RE Bussgang-MMSE equalization and calibrated 16-QAM soft LLR calculation."""
    bsz = y_grid.shape[0]
    nv_bc = noise_var.view(bsz, 1, 1)
    h_pow_eff = h_eff.abs() ** 2
    h_pow_phys = h_phys.abs() ** 2

    denom = h_pow_eff + h_pow_phys * float(sigma_d2) + nv_bc + 1e-12
    w_filter = h_eff.conj() / denom
    x_hat = w_filter * y_grid
    mu = torch.clamp(h_pow_eff / denom, min=1e-8, max=1.0)
    z_hat = x_hat / mu
    sigma2_eff = (1.0 - mu) / mu
    llrs = calibrated_16qam_llrs(z_hat, sigma2_eff, method=llr_method)
    return z_hat, llrs, sigma2_eff


def idd_pa_2d(
    slot_data: Optional[SlotData] = None,
    batch_size: int = 2,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    iterations: int = 2,
    snr_db: float = 18.0,
    A: Optional[float] = 1.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
    fd: float = 0.01,
    cfo: float = 0.0,
    profile: str = "equal4",
    oversample: int = DEFAULT_OVERSAMPLE,
    seed: int = 42,
) -> Dict[str, Any]:
    """2D Iterative Decision-Directed PA Cancellation (IDD-PA) over a multi-symbol slot."""
    if slot_data is None:
        gen = torch.Generator().manual_seed(seed)
        sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=DEFAULT_CP, profile=profile, fd=fd, cfo=cfo)
        slot_data = sim.generate_slot(
            batch_size=batch_size,
            snr_db=snr_db,
            A=A,
            alpha=alpha,
            beta=beta,
            knee=knee,
            generator=gen,
        )
    else:
        batch_size, n_symb, n_sc = slot_data.y_grid.shape
        A = slot_data.A
        alpha = slot_data.alpha
        fd = slot_data.fd
        profile = slot_data.profile

    y_grid = slot_data.y_grid
    p_mask = slot_data.pilot_mask
    d_mask = slot_data.data_mask
    nv = slot_data.noise_var

    h_phys, h_eff = _solve_2d_lmmse_core(
        yp_sparse=slot_data.yp_sparse,
        pilot_mask=p_mask,
        noise_var=nv,
        n_symb=n_symb,
        n_sc=n_sc,
        cp=DEFAULT_CP,
        fd=fd,
        profile=profile,
        A=A,
        alpha=alpha,
    )

    gamma, sigma_d2_init, _ = bussgang_params(A=A, alpha=alpha, beta=beta, knee=knee)
    gamma_c = torch.tensor(gamma, dtype=torch.complex64, device=y_grid.device)

    z_hat, llrs, _ = _bussgang_equalize_2d(y_grid, h_phys, h_eff, nv, sigma_d2_init)
    history: List[Dict[str, Any]] = [
        {"iteration": 0, "z_hat": z_hat, "llrs": llrs, "sigma_d2": sigma_d2_init, "dist_power": 0.0}
    ]

    is_linear = (A is None or math.isinf(float(A))) and alpha == 0.0
    sigma_d2_rem = sigma_d2_init
    dist_power = 0.0

    for it in range(1, int(iterations) + 1):
        if is_linear:
            dist_power = 0.0
            history.append({"iteration": it, "z_hat": z_hat, "llrs": llrs, "sigma_d2": 0.0, "dist_power": 0.0})
            continue

        # Assemble estimated 2D frequency grid: known pilots + soft reconstructed data
        s_hat = torch.zeros_like(y_grid)
        if slot_data.pilot_tx.shape[1] > 0:
            s_hat[:, p_mask] = slot_data.pilot_tx
        if int(d_mask.sum().item()) > 0:
            s_soft = soft_slice_16qam(llrs)
            s_hat[:, d_mask] = s_soft[:, d_mask]

        # Reconstruct oversampled per-symbol PA distortion
        s_ov = oversample_frequency_to_time(s_hat, oversample=oversample)
        s_pa_ov = apply_rapp_pa_samples(s_ov, A=A, alpha=alpha, beta=beta, knee=knee)
        d_ov = s_pa_ov - gamma_c * s_ov
        d_freq = decimate_time_to_frequency(d_ov, n=n_sc, oversample=oversample)
        dist_power = float((d_freq.abs() ** 2).mean().item())

        y_cleaned = y_grid - h_phys * d_freq
        sigma_d2_rem = sigma_d2_rem * 0.35
        z_hat, llrs, _ = _bussgang_equalize_2d(y_cleaned, h_phys, h_eff, nv, sigma_d2_rem)
        history.append(
            {"iteration": it, "z_hat": z_hat, "llrs": llrs, "sigma_d2": sigma_d2_rem, "dist_power": dist_power}
        )

    hard_bits = hard_slice_16qam(z_hat)
    llrs_data = llrs[:, d_mask, :] if int(d_mask.sum().item()) > 0 else torch.zeros((batch_size, 0, 4))
    hard_data = hard_bits[:, d_mask, :] if int(d_mask.sum().item()) > 0 else torch.zeros((batch_size, 0, 4), dtype=torch.int64)
    z_data = z_hat[:, d_mask] if int(d_mask.sum().item()) > 0 else torch.zeros((batch_size, 0), dtype=torch.complex64)

    return {
        "h_hat": h_phys,
        "h_eff": h_eff,
        "llrs": llrs_data,
        "llrs_grid": llrs,
        "hard_bits": hard_data,
        "z_hat": z_data,
        "z_grid": z_hat,
        "dist_power": dist_power,
        "history": history,
        "converged": bool(torch.isfinite(llrs).all().item()),
    }


def idd_pa_2d_cleanup(seed: int = 42) -> bool:
    """Verify 2D IDD-PA improves payload symbol MSE / LLR quality under strong Rapp PA."""
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=0.005)
    slot = sim.generate_slot(batch_size=8, snr_db=22.0, A=1.25, alpha=0.5, generator=gen)
    res = idd_pa_2d(slot_data=slot, iterations=2)
    z0 = res["history"][0]["z_hat"][:, slot.data_mask]
    z2 = res["history"][-1]["z_hat"][:, slot.data_mask]
    mse0 = float(((z0 - slot.data_tx).abs() ** 2).mean().item())
    mse2 = float(((z2 - slot.data_tx).abs() ** 2).mean().item())
    return bool(mse2 < mse0)


def lmmse_2d_flat(seed: int = 42) -> torch.Tensor:
    """Run 2D-LMMSE under static fd=0 AWGN; verify flat channel across symbols."""
    return lmmse_2d_interpolate(
        batch_size=2, n_symb=14, n_sc=32, fd=0.0, profile="equal4", noise_var=1e-4, seed=seed
    )


def lmmse_2d_singular(noise_var: float = 1e-12, seed: int = 42) -> bool:
    """Verify 2D-LMMSE covariance regularization stays numerically nonsingular as noise -> 0."""
    h_est = lmmse_2d_interpolate(
        batch_size=2, n_symb=14, n_sc=32, fd=0.02, profile="equal4", noise_var=noise_var, seed=seed
    )
    return bool(torch.isfinite(h_est).all().item() and h_est.abs().max().item() > 0.0)


def idd_pa_2d_linear(seed: int = 42) -> float:
    """Verify 2D IDD-PA distortion power is zero under linear PA (A=None, alpha=0)."""
    res = idd_pa_2d(batch_size=2, iterations=2, snr_db=20.0, A=None, alpha=0.0, seed=seed)
    return float(res["dist_power"])


def idd_pa_2d_clipping(A: float = 1.0, seed: int = 42) -> bool:
    """Verify 2D IDD-PA remains finite and non-divergent under severe clipping A=1.0."""
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=0.01)
    slot = sim.generate_slot(batch_size=4, snr_db=20.0, A=A, alpha=0.5, generator=gen)
    res = idd_pa_2d(slot_data=slot, iterations=3)
    z0 = res["history"][0]["z_hat"][:, slot.data_mask]
    z_end = res["history"][-1]["z_hat"][:, slot.data_mask]
    mse0 = float(((z0 - slot.data_tx).abs() ** 2).mean().item())
    mse_end = float(((z_end - slot.data_tx).abs() ** 2).mean().item())
    return bool(torch.isfinite(res["llrs"]).all().item() and mse_end <= mse0 * 1.1)


def check_edge_extrapolation(fd: float = 0.02, snr_db: float = 20.0, seed: int = 42) -> float:
    """Measure 2D-LMMSE channel extrapolation NMSE at slot edges (symbols 0 and 13)."""
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=fd)
    slot = sim.generate_slot(batch_size=8, snr_db=snr_db, pa_condition="clean", generator=gen)
    h_est = lmmse_2d_interpolate(
        batch_size=8,
        n_symb=14,
        n_sc=32,
        fd=fd,
        profile="equal4",
        noise_var=slot.noise_var,
        yp_sparse=slot.yp_sparse,
        pilot_mask=slot.pilot_mask,
    )
    edge_est = h_est[:, [0, 13], :]
    edge_true = slot.h_true[:, [0, 13], :]
    num = ((edge_est - edge_true).abs() ** 2).sum(dim=(-2, -1))
    den = (edge_true.abs() ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)
    return float((num / den).mean().item())


# ---------------------------------------------------------------------------
# 2D C*-Equivariant EM-Native Propagator Transformer (EMFloatFormer2D)
# ---------------------------------------------------------------------------

class ComplexRoPE2D(nn.Module):
    """Coordinate-aware 2D complex rotary position embedding for (t, f) grids and sparse DMRS."""

    def __init__(self, head_dim: int, base_t: float = 16.0, base_f: float = 32.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.d_t = self.head_dim // 2
        self.d_f = self.head_dim - self.d_t
        if self.d_t > 0:
            theta_t = float(base_t) ** (-torch.arange(0, self.d_t, dtype=torch.float64) / float(self.d_t))
        else:
            theta_t = torch.zeros(0, dtype=torch.float64)
        theta_f = float(base_f) ** (-torch.arange(0, self.d_f, dtype=torch.float64) / float(max(1, self.d_f)))
        self.register_buffer("theta_t", theta_t)
        self.register_buffer("theta_f", theta_f)

    def get_phasors(self, t_coords: torch.Tensor, f_coords: torch.Tensor) -> torch.Tensor:
        """Compute unitary phasors of shape (N_tok, head_dim) for arbitrary 2D coordinates."""
        t64 = t_coords.to(torch.float64).unsqueeze(-1)
        f64 = f_coords.to(torch.float64).unsqueeze(-1)
        parts = []
        if self.d_t > 0:
            parts.append(t64 * self.theta_t.to(t64.device).unsqueeze(0))
        if self.d_f > 0:
            parts.append(f64 * self.theta_f.to(f64.device).unsqueeze(0))
        angles = torch.cat(parts, dim=-1)
        return torch.complex(torch.cos(angles), torch.sin(angles))


class GreenAttention2D(nn.Module):
    """2D Coordinate-Aware Hermitian Green's Propagator Attention.

    Realizes exact 2D relative-phase Green's propagator e^{-j((t_q-t_k)theta_t + (f_q-f_k)theta_f)}
    and strict U(1) & R_{>0} equivariance for both axial grid self-attention and sparse DMRS cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int = 2):
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
        self.rope = ComplexRoPE2D(self.head_dim)

        with torch.no_grad():
            self.w_k.weight_r.copy_(self.w_q.weight_r + 0.05 * torch.randn_like(self.w_q.weight_r))
            self.w_k.weight_i.copy_(self.w_q.weight_i + 0.05 * torch.randn_like(self.w_q.weight_i))

    def forward(
        self,
        z: torch.Tensor,
        t_q: torch.Tensor,
        f_q: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        value_context: Optional[torch.Tensor] = None,
        t_k: Optional[torch.Tensor] = None,
        f_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_in = z if context is None else context
        v_in = k_in if value_context is None else value_context
        t_k = t_q if t_k is None else t_k
        f_k = f_q if f_k is None else f_k

        bsz, n_q, _ = z.shape
        n_k = k_in.shape[1]

        q = self.w_q(z).view(bsz, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k_in).view(bsz, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v_in).view(bsz, n_k, self.n_heads, self.head_dim).transpose(1, 2)

        ph_q = self.rope.get_phasors(t_q, f_q).to(q.dtype).view(1, 1, n_q, self.head_dim)
        ph_k = self.rope.get_phasors(t_k, f_k).to(k.dtype).view(1, 1, n_k, self.head_dim)

        q_rot = q * ph_q
        k_rot = k * ph_k
        v_rot = v * ph_k

        scores = hermitian_attention_scores(q_rot, k_rot)
        weights = F.softmax(scores, dim=-1).to(v_rot.dtype)
        ctx_rot = torch.matmul(weights, v_rot)
        ctx = (ctx_rot * ph_q.conj()).transpose(1, 2).contiguous().view(bsz, n_q, self.d_model)
        return self.w_out(ctx)


class BidirectionalBlock2D(nn.Module):
    """2D Dual-Stream Propagator Block: pilot self-attn + axial 2D grid self-attn +
    bidirectional sparse-DMRS cross-attention + radial FFN.
    """

    def __init__(self, d_model: int = 16, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Pilot stream
        self.norm_p1 = ComplexRMSNorm(d_model)
        self.self_attn_p = GreenAttention2D(d_model, n_heads=n_heads)
        self.norm_p2 = ComplexRMSNorm(d_model)
        self.cross_g2p = GreenAttention2D(d_model, n_heads=n_heads)
        self.norm_p3 = ComplexRMSNorm(d_model)
        self.ffn_p1 = ComplexLinear(d_model, 2 * d_model, bias=False)
        self.act_p = RadialActivation()
        self.ffn_p2 = ComplexLinear(2 * d_model, d_model, bias=False)

        # Grid stream (axial freq + axial time + pilot cross)
        self.norm_g1_f = ComplexRMSNorm(d_model)
        self.attn_g_f = GreenAttention2D(d_model, n_heads=n_heads)
        self.norm_g1_t = ComplexRMSNorm(d_model)
        self.attn_g_t = GreenAttention2D(d_model, n_heads=n_heads)
        self.norm_g2 = ComplexRMSNorm(d_model)
        self.cross_p2g = GreenAttention2D(d_model, n_heads=n_heads)
        self.norm_g3 = ComplexRMSNorm(d_model)
        self.ffn_g1 = ComplexLinear(d_model, 2 * d_model, bias=False)
        self.act_g = RadialActivation()
        self.ffn_g2 = ComplexLinear(2 * d_model, d_model, bias=False)

    def forward(
        self,
        p_tok: torch.Tensor,
        g_tok: torch.Tensor,
        t_p: torch.Tensor,
        f_p: torch.Tensor,
        t_g: torch.Tensor,
        f_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_symb, n_sc, d_model = g_tok.shape
        device = g_tok.device

        # 1. Pilot self-attention with exact sparse coordinates
        p_res = p_tok + self.self_attn_p(
            self.norm_p1(p_tok), t_q=t_p, f_q=f_p, value_context=p_tok
        )

        # 2. Axial 2D grid self-attention: frequency axis then time axis
        g_f_in = self.norm_g1_f(g_tok).reshape(bsz * n_symb, n_sc, d_model)
        g_f_val = g_tok.reshape(bsz * n_symb, n_sc, d_model)
        t_ax_f = torch.zeros(n_sc, dtype=torch.float64, device=device)
        f_ax_f = torch.arange(n_sc, dtype=torch.float64, device=device)
        g_f_out = self.attn_g_f(g_f_in, t_q=t_ax_f, f_q=f_ax_f, value_context=g_f_val)
        g_mid = g_tok + g_f_out.view(bsz, n_symb, n_sc, d_model)

        g_t_in = self.norm_g1_t(g_mid).transpose(1, 2).reshape(bsz * n_sc, n_symb, d_model)
        g_t_val = g_mid.transpose(1, 2).reshape(bsz * n_sc, n_symb, d_model)
        t_ax_t = torch.arange(n_symb, dtype=torch.float64, device=device)
        f_ax_t = torch.zeros(n_symb, dtype=torch.float64, device=device)
        g_t_out = self.attn_g_t(g_t_in, t_q=t_ax_t, f_q=f_ax_t, value_context=g_t_val)
        g_res = g_mid + g_t_out.view(bsz, n_sc, n_symb, d_model).transpose(1, 2)

        # 3. Bidirectional sparse-DMRS cross-attention
        g_flat = g_res.reshape(bsz, n_symb * n_sc, d_model)
        p_cross = p_res + self.cross_g2p(
            self.norm_p2(p_res),
            t_q=t_p,
            f_q=f_p,
            context=self.norm_g1_f(g_flat),
            value_context=g_flat,
            t_k=t_g,
            f_k=f_g,
        )

        # Feed the updated pilots back into the grid before its readout.
        p_out = p_cross + self.ffn_p2(self.act_p(self.ffn_p1(self.norm_p3(p_cross))))
        g_cross_flat = g_flat + self.cross_p2g(
            self.norm_g2(g_flat),
            t_q=t_g,
            f_q=f_g,
            context=self.norm_p1(p_out),
            value_context=p_out,
            t_k=t_p,
            f_k=f_p,
        )
        g_cross = g_cross_flat.view(bsz, n_symb, n_sc, d_model)
        g_out = g_cross + self.ffn_g2(self.act_g(self.ffn_g1(self.norm_g3(g_cross))))
        return p_out, g_out


class EMFloatFormer2D(nn.Module):
    """2D C*-Gauge-Invariant EM-Native Propagator Transformer for multi-symbol sparse DMRS slots."""

    def __init__(
        self,
        d_model: int = 16,
        n_heads: int = 2,
        depth: int = 2,
        sigma_t: float = 4.5,
        sigma_f: float = 4.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.depth = int(depth)
        self.sigma_t = float(sigma_t)
        self.sigma_f = float(sigma_f)

        self.proj_p = ComplexLinear(1, d_model, bias=False)
        self.proj_g = ComplexLinear(2, d_model, bias=False)
        self.blocks = nn.ModuleList([BidirectionalBlock2D(d_model=d_model, n_heads=n_heads) for _ in range(depth)])
        self.norm_g = ComplexRMSNorm(d_model)

        self.head_h = ComplexLinear(d_model, 1, bias=False)
        self.head_dist = ComplexLinear(d_model, 1, bias=False)
        self.gamma_eff = ComplexLinear(1, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        std_embed = 1.0 / math.sqrt(2.0 * self.d_model)
        std_head = 0.01 / math.sqrt(2.0 * self.d_model)
        nn.init.normal_(self.proj_p.weight_r, 0.0, std_embed)
        nn.init.normal_(self.proj_p.weight_i, 0.0, std_embed)
        nn.init.normal_(self.proj_g.weight_r, 0.0, std_embed)
        nn.init.normal_(self.proj_g.weight_i, 0.0, std_embed)
        nn.init.normal_(self.head_h.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_h.weight_i, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_r, 0.0, std_head)
        nn.init.normal_(self.head_dist.weight_i, 0.0, std_head)
        with torch.no_grad():
            self.gamma_eff.weight_r.fill_(1.0)
            self.gamma_eff.weight_i.fill_(0.0)

    def _compute_rbf_prior(
        self,
        yp_bar: torch.Tensor,
        t_p: torch.Tensor,
        f_p: torch.Tensor,
        t_g: torch.Tensor,
        f_g: torch.Tensor,
        n_symb: int,
        n_sc: int,
    ) -> torch.Tensor:
        """Strictly C*-equivariant real RBF kernel interpolation from sparse DMRS to 2D grid."""
        dt = t_g.unsqueeze(1) - t_p.unsqueeze(0)  # (N_grid, N_p)
        df_raw = (f_g.unsqueeze(1) - f_p.unsqueeze(0)).abs()
        df = torch.minimum(df_raw, float(n_sc) - df_raw)
        dist2 = (dt / self.sigma_t) ** 2 + (df / self.sigma_f) ** 2
        weights = F.softmax(-0.5 * dist2, dim=-1).to(torch.complex128)  # real-valued weights in complex128
        h0_flat = torch.matmul(yp_bar, weights.t())  # (B, N_grid)
        return h0_flat.view(yp_bar.shape[0], n_symb, n_sc)

    def forward(
        self,
        yp_sparse: torch.Tensor,
        y_grid: torch.Tensor,
        pilot_mask: torch.Tensor,
        nv: torch.Tensor,
    ) -> FloatFormerOutput:
        out_complex_dtype = y_grid.dtype
        out_real_dtype = y_grid.real.dtype

        if yp_sparse.dim() == 1:
            yp_sparse = yp_sparse.unsqueeze(0)
        if y_grid.dim() == 2:
            y_grid = y_grid.unsqueeze(0)
        if nv.dim() == 1:
            nv = nv.unsqueeze(-1)

        bsz, n_symb, n_sc = y_grid.shape
        device = y_grid.device

        yp64 = yp_sparse.to(torch.complex128)
        yg64 = y_grid.to(torch.complex128)
        nv64 = nv.to(torch.float64)

        # 1. Exact C* Gauge Normalization in complex128 / float64
        s0 = compute_gauge_anchor(yp64, eps=1e-38)  # (B, 1) float64
        yp_bar = yp64 / s0
        yg_bar = yg64 / s0.unsqueeze(-1)
        nv_bar = nv64 / (s0 ** 2)

        # 2. Coordinate tensors for sparse DMRS and full 2D grid
        p_coords = torch.nonzero(pilot_mask, as_tuple=False)
        t_p = p_coords[:, 0].to(dtype=torch.float64, device=device)
        f_p = p_coords[:, 1].to(dtype=torch.float64, device=device)

        t_grid_2d, f_grid_2d = torch.meshgrid(
            torch.arange(n_symb, dtype=torch.float64, device=device),
            torch.arange(n_sc, dtype=torch.float64, device=device),
            indexing="ij",
        )
        t_g = t_grid_2d.reshape(-1)
        f_g = f_grid_2d.reshape(-1)

        # 3. C*-Equivariant 2D RBF Initial Channel Prior
        h0_bar = self._compute_rbf_prior(yp_bar, t_p, f_p, t_g, f_g, n_symb, n_sc)

        # 4. Dual-stream embeddings & 2D Propagator blocks
        p_tok = self.proj_p(yp_bar.unsqueeze(-1))
        g_in = torch.stack([yg_bar, h0_bar], dim=-1)
        g_tok = self.proj_g(g_in)

        for block in self.blocks:
            p_tok, g_tok = block(p_tok, g_tok, t_p=t_p, f_p=f_p, t_g=t_g, f_g=f_g)

        g_out = g_tok + 0.1 * self.norm_g(g_tok)

        # 5. Physical Channel & Nonlinear PA Distortion Estimates
        h_bar = h0_bar + self.head_h(g_out).squeeze(-1)
        h_hat = (h_bar * s0.unsqueeze(-1)).to(out_complex_dtype)

        dist_bar = self.head_dist(g_out).squeeze(-1)
        dist_est = (dist_bar * s0.unsqueeze(-1)).to(out_complex_dtype)

        # 6. Physical Equalization & Exact M2 LLR Calibration Contract (3.5e-2 floor + 20*tanh)
        h_eff64 = self.gamma_eff(h_bar.unsqueeze(-1)).squeeze(-1)
        h_power = h_eff64.real ** 2 + h_eff64.imag ** 2
        nv_exp = nv_bar.unsqueeze(-1).expand_as(h_power)

        dist_eq = (dist_bar * h_eff64.conj()) / (h_power + 1e-4)
        z_eq = (yg_bar * h_eff64.conj()) / (h_power + 1e-4) - dist_eq

        h_power_det = h_power.detach()
        dist_power_det = (dist_bar.real ** 2 + dist_bar.imag ** 2).detach()
        sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 3.5e-2
        llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
        llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)

        return FloatFormerOutput(h_hat=h_hat, llrs=llrs, dist_est=dist_est)


class JointPhysicalLoss2D(nn.Module):
    """2D Joint Physical Consistency Loss: payload LLR BCE + 2D channel NMSE + 2D PA residual."""

    def __init__(self, lambda_nmse: float = 1.0, lambda_pa: float = 0.25):
        super().__init__()
        self.lambda_nmse = lambda_nmse
        self.lambda_pa = lambda_pa

    def forward(
        self,
        out: FloatFormerOutput,
        slot: SlotData,
        A: Optional[float] = None,
    ) -> torch.Tensor:
        llrs_grid = out["llrs"]       # (B, N_symb, N_sc, 4)
        h_est = out["h_hat"]          # (B, N_symb, N_sc)
        d_mask = slot.data_mask
        p_mask = slot.pilot_mask

        # 1. Payload BCE
        if int(d_mask.sum().item()) > 0:
            llrs_data = llrs_grid[:, d_mask, :]
            l_bce = F.binary_cross_entropy_with_logits(llrs_data, slot.bits.float())
        else:
            l_bce = torch.zeros((), dtype=torch.float32, device=h_est.device)

        # 2. Scale-invariant 2D channel NMSE
        err_sq = (h_est - slot.h_true).abs() ** 2
        den_sq = slot.h_true.abs() ** 2
        num_sum = err_sq.sum(dim=(-2, -1))
        den_sum = den_sq.sum(dim=(-2, -1)).clamp(min=1e-12)
        l_nmse = (num_sum / den_sum).mean()

        # 3. 2D Forward PA residual consistency
        a_val = slot.A if A is None else A
        x_est_grid = torch.zeros_like(slot.y_grid)
        if slot.pilot_tx.shape[1] > 0:
            x_est_grid[:, p_mask] = slot.pilot_tx
        if int(d_mask.sum().item()) > 0:
            x_est_grid[:, d_mask] = soft_reconstruct_16qam(llrs_grid[:, d_mask, :]).to(slot.y_grid.dtype)

        x_time = torch.fft.ifft(x_est_grid, dim=-1, norm="ortho")
        x_pa_time = rapp_pa(x_time, A=a_val, alpha=slot.alpha, oversample=slot.oversample,
                            knee=slot.knee, beta=slot.beta)
        x_pa_freq = torch.fft.fft(x_pa_time, dim=-1, norm="ortho")

        y_model = h_est * x_pa_freq
        pa_num = ((slot.y_grid - y_model).abs() ** 2).sum(dim=(-2, -1))
        pa_den = (slot.y_grid.abs() ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)
        l_pa = (pa_num / pa_den).mean()

        return l_bce + self.lambda_nmse * l_nmse + self.lambda_pa * l_pa


def _train_emfloatformer_2d(
    steps: int = 80,
    batch_size: int = 8,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    profile: str = "equal4",
    seed: int = 2026,
    backbone_lr: float = 8e-3,
    gamma_lr: float = 2e-2,
    *,
    model_factory: Callable[[], EMFloatFormer2D] = EMFloatFormer2D,
) -> EMFloatFormer2D:
    """Train EMFloatFormer2D under mixed-PA, mixed-Doppler JointPhysicalLoss2D."""
    torch.manual_seed(seed)
    model = model_factory()
    gen = torch.Generator().manual_seed(seed + 1000)
    sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=DEFAULT_CP, profile=profile)
    loss_fn = JointPhysicalLoss2D(lambda_nmse=1.0, lambda_pa=0.25)

    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "gamma_eff" not in n], "lr": backbone_lr},
        {"params": model.gamma_eff.parameters(), "lr": gamma_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=backbone_lr * 0.1
    )
    pa_cycle = ("strong", "mild", "clean")
    fd_cycle = (0.0, 0.01, 0.02)

    model.train()
    for step in range(steps):
        pa_cond = pa_cycle[step % len(pa_cycle)]
        fd_val = fd_cycle[(step // len(pa_cycle)) % len(fd_cycle)]
        slot = sim.generate_slot(
            batch_size=batch_size,
            snr_db=(10.0, 26.0),
            pa_condition=pa_cond,
            fd=fd_val,
            cfo=0.0,
            generator=gen,
        )
        slot = _synchronize_slot(slot)
        optimizer.zero_grad()
        out = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        loss = loss_fn(out, slot)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


# ===========================================================================
# Feature F17: Coded BLER Evaluation, Tier 3 & Tier 4 Workloads & Benchmarks
# ===========================================================================

def check_block_error(u: torch.Tensor, u_hat: torch.Tensor) -> int:
    """Return Python int 0 iff all information bits match, else 1."""
    return 0 if torch.equal(u.long(), u_hat.long()) else 1


def _decode_slot_ldpc_blocks(
    llrs_data: torch.Tensor,
    info_bits: torch.Tensor,
    ldpc: LDPCMatrix,
    max_iter: int = 25,
) -> Tuple[int, int, int, int]:
    """Decode all Rate-1/2 LDPC codewords inside a batch of 2D slots and return exact Python ints:
    (bit_errors, total_bits, block_errors, total_blocks).
    """
    bsz, n_cw, k = info_bits.shape
    if n_cw == 0:
        return 0, 0, 0, 0
    needed = n_cw * ldpc.n
    llrs_flat = llrs_data.reshape(bsz, -1)[:, :needed].reshape(bsz * n_cw, ldpc.n)
    u_hat = decode_min_sum(llrs_flat, ldpc.h_matrix, max_iter=max_iter).view(bsz, n_cw, k)

    bit_errors = int((u_hat != info_bits).sum().item())
    total_bits = int(info_bits.numel())
    blk_errs_vec = (u_hat != info_bits).any(dim=-1)
    block_errors = int(blk_errs_vec.sum().item())
    total_blocks = int(bsz * n_cw)
    return bit_errors, total_bits, block_errors, total_blocks


def _evaluate_slot_bler(
    snr_db: float,
    num_slots: int = 4,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    fd: float = 0.01,
    cfo: float = 0.0,
    profile: str = "equal4",
    pa_condition: str = "mild",
    method: str = "idd_pa_2d",
    model: Optional[EMFloatFormer2D] = None,
    seed: int = 2026,
    shared_noise: bool = False,
) -> Dict[str, Any]:
    """Evaluate coded BLER and exact integer error counts over a batch of 2D slots."""
    gen = torch.Generator().manual_seed(seed)
    ldpc = _get_default_ldpc(128)
    sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=DEFAULT_CP, profile=profile, fd=fd, cfo=cfo)
    slot = sim.generate_slot(
        batch_size=num_slots,
        snr_db=snr_db,
        pa_condition=pa_condition,
        coded=True,
        ldpc=ldpc,
        generator=gen,
    )

    slot_proc = _synchronize_slot(slot)

    if method == "emfloatformer2d":
        if model is None:
            raise ValueError("EMFloatFormer2D model instance is required for method='emfloatformer2d'.")
        with torch.no_grad():
            out = model(slot_proc.yp_sparse, slot_proc.y_grid, slot_proc.pilot_mask, slot_proc.noise_var)
        h_hat = out["h_hat"]
        llrs_data = out["llrs"][:, slot_proc.data_mask, :]
    elif method in ("lmmse_2d", "idd_pa_2d", "lmmse_2d_fixed", "idd_pa_2d_fixed"):
        receiver_slot = slot_proc
        if method.endswith("_fixed"):
            receiver_slot = replace(slot_proc, profile="equal4", fd=0.01, A=2.25, alpha=0.5)
        res = idd_pa_2d(slot_data=receiver_slot, iterations=0 if method.startswith("lmmse") else 2)
        h_hat = res["h_hat"]
        llrs_data = res["llrs"]
    else:
        raise ValueError(f"Unknown receiver {method!r}")

    bit_errs, tot_bits, blk_errs, tot_blks = _decode_slot_ldpc_blocks(llrs_data, slot.info_bits, ldpc)
    uncoded_bit_errs = int(((llrs_data > 0).long() != slot.bits).sum().item())
    uncoded_tot_bits = int(slot.bits.numel())

    num = ((h_hat - slot_proc.h_true).abs() ** 2).sum(dim=(-2, -1))
    den = (slot.h_true.abs() ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)
    nmse = float((num / den).mean().item())
    ece = compute_ece(llrs_data, slot.bits, n_bins=10)
    nonfinite = int((~torch.isfinite(llrs_data)).sum().item() + (~torch.isfinite(h_hat)).sum().item())

    return {
        "bler": float(blk_errs) / float(max(1, tot_blks)),
        "ber": float(bit_errs) / float(max(1, tot_bits)),
        "uncoded_ber": float(uncoded_bit_errs) / float(max(1, uncoded_tot_bits)),
        "uncoded_bit_errors": uncoded_bit_errs,
        "total_uncoded_bits": uncoded_tot_bits,
        "bit_errors": int(bit_errs),
        "total_bits": int(tot_bits),
        "block_errors": int(blk_errs),
        "total_blocks": int(tot_blks),
        "nmse": nmse,
        "ece": ece,
        "nonfinite_count": int(nonfinite),
    }


def evaluate_bler_curve(
    snrs: Sequence[float],
    num_blocks: int = 26,
    fd: float = 0.01,
    profile: str = "equal4",
    pa_condition: str = "mild",
    seed: int = 2026,
) -> List[float]:
    """Evaluate coded BLER across an SNR grid using common random numbers for monotonicity."""
    num_slots = max(1, math.ceil(int(num_blocks) / 13))
    blers: List[float] = []
    for snr_db in snrs:
        res = _evaluate_slot_bler(
            snr_db=float(snr_db),
            num_slots=num_slots,
            fd=fd,
            profile=profile,
            pa_condition=pa_condition,
            method="idd_pa_2d",
            seed=seed,  # common random numbers across SNRs
        )
        blers.append(float(res["bler"]))
    return blers


def run_block_counter(num_blocks: int = 10, snr_db: float = 12.0, seed: int = 42) -> Tuple[int, int]:
    """Simulate `num_blocks` coded blocks and return exact Python (int_errors, int_total)."""
    ldpc = _get_default_ldpc(128)
    gen = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 2, (num_blocks, ldpc.k), generator=gen, dtype=torch.int64)
    c, h_mat = encode_ldpc(u, ldpc=ldpc)
    x = q16(c.view(num_blocks, 32, 4))
    nv = 10.0 ** (-float(snr_db) / 10.0)
    z = x + complex_randn(x.shape, generator=gen) * math.sqrt(nv)
    llrs = calibrated_16qam_llrs(z, torch.full_like(z.real, nv), method="exact").view(num_blocks, 128)
    u_hat = decode_min_sum(llrs, h_mat, max_iter=25)

    errs = 0
    for b in range(num_blocks):
        errs += check_block_error(u[b], u_hat[b])
    return int(errs), int(num_blocks)


def check_doppler_bler_sensitivity(snr_db: float = 12.0, seed: int = 42) -> Tuple[float, float]:
    """Compare coded BLER at low Doppler (fd=0.0) vs high Doppler (fd=0.08) with common seed."""
    r_lo = _evaluate_slot_bler(
        snr_db=snr_db, num_slots=2, fd=0.0, profile="equal4", pa_condition="clean", method="lmmse_2d", seed=seed
    )
    r_hi = _evaluate_slot_bler(
        snr_db=snr_db, num_slots=2, fd=0.08, profile="equal4", pa_condition="clean", method="lmmse_2d", seed=seed
    )
    return float(r_lo["bler"]), float(r_hi["bler"])


def evaluate_ood_profile_bler(
    profile: str = "sparse12",
    snr_db: float = 15.0,
    num_slots: int = 2,
    seed: int = 42,
) -> float:
    """Evaluate coded BLER on an OOD multipath delay profile (e.g. 'sparse12')."""
    res = _evaluate_slot_bler(
        snr_db=snr_db,
        num_slots=num_slots,
        fd=0.01,
        profile=profile,
        pa_condition="mild",
        method="idd_pa_2d",
        seed=seed,
    )
    return float(res["bler"])


def evaluate_bler_point(snr_db: float, seed: int = 42) -> float:
    """Evaluate coded BLER at a single SNR point (returns 0.0 at 40 dB, 1.0 at -15 dB)."""
    res = _evaluate_slot_bler(
        snr_db=float(snr_db),
        num_slots=2,
        fd=0.0,
        profile="equal4",
        pa_condition="clean",
        method="lmmse_2d",
        seed=seed,
    )
    return float(res["bler"])


def simulate_single_block(snr_db: float = 15.0, seed: int = 42) -> int:
    """Simulate exactly 1 LDPC block and return Python int 0 or 1."""
    errs, _ = run_block_counter(num_blocks=1, snr_db=snr_db, seed=seed)
    return int(errs)


def check_integer_accumulator(count: int = 1_000_000) -> int:
    """Verify exact 64-bit Python int block/bit error accumulation up to `count`."""
    total: int = 0
    chunk = 10_000
    full_chunks, rem = divmod(int(count), chunk)
    for _ in range(full_chunks):
        total += chunk
    total += rem
    return int(total)


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    log_c = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_c + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _binom_cdf_lower(k_max: int, n: int, p: float) -> float:
    """P(X <= k_max) for X ~ Binomial(n, p)."""
    if k_max < 0:
        return 0.0
    if k_max >= n:
        return 1.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k_max + 1)))


def _binom_cdf_upper(k_min: int, n: int, p: float) -> float:
    """P(X >= k_min) for X ~ Binomial(n, p)."""
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k_min, n + 1)))


def compute_ci(errors: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval [p_low, p_high] with zero external deps."""
    e = int(errors)
    n = int(total)
    if n <= 0:
        raise ValueError("total must be positive.")
    if e < 0 or e > n:
        raise ValueError("errors must be in [0, total].")

    half_alpha = alpha / 2.0
    if e == 0:
        p_low = 0.0
    else:
        lo, hi = 0.0, float(e) / float(n)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _binom_cdf_upper(e, n, mid) < half_alpha:
                lo = mid
            else:
                hi = mid
        p_low = 0.5 * (lo + hi)

    if e == n:
        p_high = 1.0
    else:
        lo, hi = float(e) / float(n), 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _binom_cdf_lower(e, n, mid) < half_alpha:
                hi = mid
            else:
                lo = mid
        p_high = 0.5 * (lo + hi)

    return float(p_low), float(p_high)


# ---------------------------------------------------------------------------
# Tier 3 Cross-Feature Callables in p4.slot_2d
# ---------------------------------------------------------------------------

def sim_multipath_cfo(fd: float = 0.03, cfo: float = 0.05, seed: int = 42) -> torch.Tensor:
    """P04 (F2+F14): Continuous multipath fading combined with sample CFO drift."""
    return simulate_slot_channel(
        batch_size=2, fd=fd, cfo=cfo, generator=torch.Generator().manual_seed(seed)
    )


def decode_calibrated_ldpc(snr_db: float = 15.0, seed: int = 42) -> float:
    """P06 (F5+F15): Calibrated 16-QAM soft LLRs fed into Rate-1/2 LDPC Min-Sum decoder."""
    gen = torch.Generator().manual_seed(seed)
    ldpc = _get_default_ldpc(128)
    n_blk = 16
    u = torch.randint(0, 2, (n_blk, ldpc.k), generator=gen, dtype=torch.int64)
    c, h_mat = encode_ldpc(u, ldpc=ldpc)
    x = q16(c.view(n_blk, 32, 4))
    nv = 10.0 ** (-float(snr_db) / 10.0)
    z = x + complex_randn(x.shape, generator=gen) * math.sqrt(nv)
    llrs = calibrated_16qam_llrs(z, torch.full_like(z.real, nv), method="exact").view(n_blk, 128)
    u_hat = decode_min_sum(llrs, h_mat, max_iter=25)
    return float((u_hat != u).float().mean().item())


def idd_pa_2d_slot(iterations: int = 2, seed: int = 42) -> Dict[str, Any]:
    """P07 (F6+F16): 2D iterative DD-PA cancellation over a multi-symbol slot."""
    return idd_pa_2d(batch_size=2, iterations=iterations, snr_db=20.0, A=1.25, seed=seed)


def sim_slot_doppler_cfo(fd: float = 0.05, cfo: float = 0.05, seed: int = 42) -> torch.Tensor:
    """P15 (F13+F14): 14-symbol slot with sparse DMRS under high Doppler & CFO, shape (14, 32)."""
    out = simulate_slot_channel(
        batch_size=1, n_symb=14, n_sc=32, fd=fd, cfo=cfo, generator=torch.Generator().manual_seed(seed)
    )
    return out.squeeze(0)


def estimate_slot_2d_lmmse(seed: int = 42) -> torch.Tensor:
    """P16 (F13+F16): 2D-LMMSE channel estimation from sparse DMRS REs, shape (14, 32)."""
    return lmmse_2d_interpolate(batch_size=1, n_symb=14, n_sc=32, fd=0.02, seed=seed).squeeze(0)


def eval_high_doppler_bler(fd_hz: float = 600.0, fc: float = 3.0e9, snr_db: float = 16.0, seed: int = 42) -> float:
    """P17 (F14+F17): Evaluate outer coded BLER under high Doppler (600 Hz)."""
    fd_norm = float(fd_hz) / DEFAULT_SCS_HZ
    res = _evaluate_slot_bler(
        snr_db=snr_db, num_slots=4, fd=fd_norm, profile="equal4", pa_condition="mild", method="idd_pa_2d", seed=seed
    )
    return float(res["bler"])


def trace_bler_waterfall(
    snrs: Sequence[float] = (0.0, 5.0, 10.0, 15.0, 20.0),
    seed: int = 42,
) -> List[float]:
    """P18 (F15+F17): Trace coded BLER waterfall across SNR grid [0, 5, 10, 15, 20] dB."""
    return evaluate_bler_curve(snrs, num_blocks=26, fd=0.01, profile="equal4", pa_condition="mild", seed=seed)


def eval_ood_baselines_bler(seed: int = 42) -> Tuple[float, float]:
    """P19 (F16+F17): Benchmark 2D-LMMSE and 2D IDD-PA BLER on held-out sparse12 profile."""
    r_lmmse = _evaluate_slot_bler(
        snr_db=15.0, num_slots=2, fd=0.01, profile="sparse12", pa_condition="strong", method="lmmse_2d", seed=seed
    )
    r_idd = _evaluate_slot_bler(
        snr_db=15.0, num_slots=2, fd=0.01, profile="sparse12", pa_condition="strong", method="idd_pa_2d", seed=seed
    )
    return float(r_lmmse["bler"]), float(r_idd["bler"])


# ---------------------------------------------------------------------------
# Tier 4 Real-World Workload Callables in p4.slot_2d
# ---------------------------------------------------------------------------

def run_hst_workload(
    speed_kmh: float = 300.0,
    fc: float = 3.0e9,
    pa_A: float = 1.25,
    snr_db: float = 15.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """S01: four independent fading slots with continuous sample-level Doppler inside each slot."""
    from .stage3_link_hst import run_hst_workload as receive_hst

    return receive_hst(speed_kmh, fc, pa_A, snr_db, seed,
                       propagation="native", arm="long_slot_spread")


def run_urban_macro_workload(
    n_sc: int = 64,
    profile: str = "long12",
    pa_A: float = 2.25,
    num_slots: int = 20,
    snr_db: float = 18.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """S02: Urban Macrocell (N_sc=64, 12-tap long12, mild PA A=2.25, 20 slots)."""
    if profile in ("long12", "sparse12") and n_sc % 2 == 0 and n_sc // 2 >= 12:
        from .stage3_urban import run_urban_workload

        return run_urban_workload(n_sc, profile, pa_A, num_slots, snr_db, seed)

    pa_cond = "mild" if pa_A >= 2.0 else "strong"
    res = _evaluate_slot_bler(
        snr_db=snr_db,
        num_slots=num_slots,
        n_symb=14,
        n_sc=n_sc,
        fd=0.005,
        cfo=0.0,
        profile=profile,
        pa_condition=pa_cond,
        method="idd_pa_2d",
        seed=seed,
    )
    return {
        "nmse": float(res["nmse"]),
        "bler": float(res["bler"]),
        "bit_errors": int(res["bit_errors"]),
        "total_bits": int(res["total_bits"]),
    }


def run_cfo_drift_workload(
    cfo: float = 0.06,
    n_symb: int = 14,
    snr_db: float = 20.0,
    seed: int = 42,
    doppler_hz: float = 150.0,
) -> Dict[str, Any]:
    """S03: blind synchronization and symbol recovery in the 150 Hz Doppler channel."""
    if n_symb == 14:
        from .stage3_cfo import run_stage3_cfo_workload

        result = run_stage3_cfo_workload(cfo, n_symb, snr_db, seed, doppler_hz)
        return {**result, "iq_sign_errors": result["quadrant_flips"],
                "quadrant_flips": result["wrong_carrier_quadrant_rows"] + result["unobservable_rows"],
                "quadrant_metric": "OFDM symbol rows outside the transmitted sector or without an observable lock"}

    # Retain the classical receiver for alternate slot sizes outside the learned geometry.
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=n_symb, n_sc=32, cp=16, profile="equal4", fd=doppler_hz/DEFAULT_SCS_HZ, cfo=cfo)
    slot = sim.generate_slot(batch_size=8, snr_db=snr_db, pa_condition="clean", generator=gen)

    cfo_est = estimate_cfo_cp_tensor(slot.rx_time, n_symb=n_symb, n_sc=32, cp=16)
    rx_comp = compensate_cfo_time(slot.rx_time, cfo_est, n_sc=32)
    rx_sym = rx_comp.view(8, n_symb, 32 + 16)[:, :, 16:]
    y_comp = torch.fft.fft(rx_sym, dim=-1, norm="ortho")

    # Equalize using pilot observation after CFO compensation
    yp_comp = y_comp[:, slot.pilot_mask] * slot.pilot_tx.conj()
    h_est = lmmse_2d_interpolate(
        batch_size=8,
        n_symb=n_symb,
        n_sc=32,
        fd=doppler_hz/DEFAULT_SCS_HZ,
        profile="equal4",
        noise_var=slot.noise_var,
        yp_sparse=yp_comp,
        pilot_mask=slot.pilot_mask,
    )
    z_eq = y_comp[:, slot.data_mask] / h_est[:, slot.data_mask]
    # Measure residual common phase error across slot relative to transmitted symbols
    phase_err_rad = torch.angle((z_eq * slot.data_tx.conj()).sum(dim=-1))
    residual_phase_deg = float((phase_err_rad.abs() * (180.0 / math.pi)).mean().item())
    # Count I and Q sign errors separately. Noise, fading and channel estimation also affect these decisions.
    quad_flips = int(
        ((z_eq.real > 0) != (slot.data_tx.real > 0)).sum().item()
        + ((z_eq.imag > 0) != (slot.data_tx.imag > 0)).sum().item()
    )
    from .stage3_cfo import phase_and_errors
    equalized_grid = torch.zeros_like(y_comp)
    equalized_grid[:, slot.data_mask] = z_eq
    phase = phase_and_errors(equalized_grid, slot.data_tx, slot.data_mask)
    return {
        "residual_phase_deg": residual_phase_deg,
        "quadrant_flips": phase["wrong_carrier_quadrant_rows"] + phase["unobservable_rows"],
        "wrong_carrier_quadrant_rows": phase["wrong_carrier_quadrant_rows"],
        "unobservable_rows": phase["unobservable_rows"],
        "carrier_quadrant_transitions": phase["carrier_quadrant_transitions"],
        "rows": phase["rows"], "adjacent_row_pairs": phase["adjacent_row_pairs"],
        "iq_sign_errors": quad_flips, "sign_count": phase["sign_total"],
        "quadrant_metric": "OFDM symbol rows outside the transmitted sector or without an observable lock",
        "cfo_est_mean": float(cfo_est.mean().item()),
    }


def run_ultra_sparse_workload(
    comb: int = 4,
    pilot_symbol: int = 2,
    snr_db: float = 20.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """S05: estimate the channel from sparse pilots and posterior payload information."""
    from .stage3_sparse import run_ultra_sparse_workload as estimate

    return estimate(comb=comb, pilot_symbol=pilot_symbol, snr_db=snr_db, seed=seed)


def run_multi_impairment_stress(num_slots: int = 12, seed: int = 42) -> Dict[str, Any]:
    """Measure coded BLER against both matched baselines and retain uncoded BER diagnostics."""
    from .stage3_stress import coded_stress

    coded = coded_stress(num_slots, seed)
    legacy = _run_uncoded_stress(num_slots, seed)
    return {**coded, "outputs_finite": coded["outputs_finite"] and legacy["outputs_finite"],
            "ber_naive": legacy["ber_naive"], "ber_2d": legacy["ber_2d"],
            "uncoded_sign_flip_p": legacy["p_value"]}


def _run_uncoded_stress(num_slots: int, seed: int) -> Dict[str, Any]:
    """S06: Concurrent Rapp AM-AM (1.25), AM-PM (0.5), Doppler (0.03), CFO (0.04), delay spread."""
    gen = torch.Generator().manual_seed(seed)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="long6", fd=0.03, cfo=0.04)
    slot = sim.generate_slot(batch_size=num_slots, snr_db=18.0, A=1.25, alpha=0.5, generator=gen)

    # Naive uncompensated LS baseline (no CFO correction, nearest-pilot LS)
    p_coords = torch.nonzero(slot.pilot_mask, as_tuple=False)
    h_naive = torch.zeros_like(slot.y_grid)
    for k in range(32):
        # nearest pilot in symbol 2
        idx_s2 = torch.nonzero(p_coords[:, 0] == 2, as_tuple=False).squeeze(-1)
        nearest = int(idx_s2[torch.argmin((p_coords[idx_s2, 1] - k).abs())].item())
        h_naive[:, :, k] = slot.yp_sparse[:, nearest].unsqueeze(1)
    z_naive = slot.y_grid[:, slot.data_mask] / (h_naive[:, slot.data_mask] + 1e-6)
    bits_naive = hard_slice_16qam(z_naive)
    err_naive = (bits_naive != slot.bits).reshape(num_slots, -1).sum(dim=-1)

    # 2D receiver: CP-CFO compensation + 2D IDD-PA cancellation
    cfo_hat = estimate_cfo_cp_tensor(slot.rx_time, n_symb=14, n_sc=32, cp=16)
    rx_comp = compensate_cfo_time(slot.rx_time, cfo_hat, n_sc=32)
    rx_sym = rx_comp.view(num_slots, 14, 32 + 16)[:, :, 16:]
    y_comp = torch.fft.fft(rx_sym, dim=-1, norm="ortho")
    yp_comp = y_comp[:, slot.pilot_mask] * slot.pilot_tx.conj()
    slot_comp = SlotData(
        y_grid=y_comp,
        yp_sparse=yp_comp,
        y_p_raw=y_comp[:, slot.pilot_mask],
        y_d_raw=y_comp[:, slot.data_mask],
        pilot_tx=slot.pilot_tx,
        data_tx=slot.data_tx,
        x_grid=slot.x_grid,
        bits=slot.bits,
        info_bits=slot.info_bits,
        coded_bits=slot.coded_bits,
        h_true=slot.h_true,
        h_time_evol=slot.h_time_evol,
        noise_var=slot.noise_var,
        pilot_mask=slot.pilot_mask,
        data_mask=slot.data_mask,
        rx_time=slot.rx_time,
        A=slot.A,
        alpha=slot.alpha,
        fd=slot.fd,
        cfo=slot.cfo,
        profile=slot.profile,
    )
    res_2d = idd_pa_2d(slot_data=slot_comp, iterations=2)
    err_2d = (res_2d["hard_bits"] != slot.bits).reshape(num_slots, -1).sum(dim=-1)

    from .stage3_stress import paired_p
    p_val = paired_p((err_naive - err_2d).tolist())

    finite_ok = bool(
        torch.isfinite(res_2d["llrs"]).all().item() and torch.isfinite(res_2d["h_hat"]).all().item()
    )
    return {
        "outputs_finite": finite_ok,
        "p_value": p_val,
        "ber_naive": float(err_naive.float().sum().item() / slot.bits.numel()),
        "ber_2d": float(err_2d.float().sum().item() / slot.bits.numel()),
    }


def run_ood_generalization(seed: int = 2026) -> Dict[str, Any]:
    """S07: train on equal4; evaluate integer-FEC and fractional-delay uncoded criteria."""
    from .next_neural import TapPhaseNoChannel
    from .stage3_ood import evaluate_fractional

    torch.set_num_threads(4)
    model = _train_emfloatformer_2d(steps=75, batch_size=8, n_symb=14, n_sc=32, profile="equal4",
                                    seed=seed, model_factory=TapPhaseNoChannel)

    # Pilots and payload bits are shared; different tap counts advance the RNG before AWGN is drawn.
    gen_id = torch.Generator().manual_seed(seed + 500)
    gen_ood = torch.Generator().manual_seed(seed + 500)
    sim_id = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=0.01)
    sim_ood = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="exp16", fd=0.01)

    slot_id = _synchronize_slot(sim_id.generate_slot(
        batch_size=32, snr_db=20.0, pa_condition="mild", coded=True, generator=gen_id))
    slot_ood = _synchronize_slot(sim_ood.generate_slot(
        batch_size=32, snr_db=20.0, pa_condition="mild", coded=True, generator=gen_ood))

    ldpc = _get_default_ldpc(128)
    with torch.no_grad():
        out_id = model(slot_id.yp_sparse, slot_id.y_grid, slot_id.pilot_mask, slot_id.noise_var)
        out_ood = model(slot_ood.yp_sparse, slot_ood.y_grid, slot_ood.pilot_mask, slot_ood.noise_var)

    # Score the decoded BER gap separately from the circular subcarrier-shift diagnostic below.
    llrs_id = out_id["llrs"][:, slot_id.data_mask, :]
    llrs_ood = out_ood["llrs"][:, slot_ood.data_mask, :]
    err_id, tot_id, _, _ = _decode_slot_ldpc_blocks(llrs_id, slot_id.info_bits, ldpc)
    err_ood, tot_ood, _, _ = _decode_slot_ldpc_blocks(llrs_ood, slot_ood.info_bits, ldpc)
    ber_id = err_id / float(max(1, tot_id))
    ber_ood = err_ood / float(max(1, tot_ood))
    ber_gap = abs(ber_ood - ber_id)

    # Shortcut check: circularly shifting y_grid along frequency axis with shifted pilot_mask
    # should produce equivariant predictions (relative LLR invariance under frequency shift)
    shift = 4
    y_shift = torch.roll(slot_ood.y_grid, shifts=shift, dims=-1)
    p_shift = torch.roll(slot_ood.pilot_mask, shifts=shift, dims=-1)
    yp_grid = torch.zeros_like(slot_ood.y_grid)
    yp_grid[:, slot_ood.pilot_mask] = slot_ood.yp_sparse
    yp_shift = torch.roll(yp_grid, shifts=shift, dims=-1)[:, p_shift]
    with torch.no_grad():
        out_shift = model(yp_shift, y_shift, p_shift, slot_ood.noise_var)
    llrs_unshifted = torch.roll(out_shift["llrs"], shifts=-shift, dims=-2)
    shift_diff = float((llrs_unshifted - out_ood["llrs"]).abs().mean().item())
    fractional = evaluate_fractional(model, seed=seed + 901000)
    shortcut_collapse = bool(ber_ood > 0.1 or shift_diff > 1.0 or fractional["ber_minus_dft8"] > 0.005)

    return {
        "shortcut_collapse": shortcut_collapse,
        "ber_gap": float(ber_gap),
        "ber_id": float(ber_id),
        "ber_ood": float(ber_ood),
        "shift_diff": shift_diff,
        "fractional": fractional,
        "integer_decoded_errors": {"id": err_id, "ood": err_ood},
        "integer_decoded_bits": {"id": tot_id, "ood": tot_ood},
    }


def run_mission_link(
    total_bits: int = 100_000,
    num_slots: int = 50,
    snr_db: float = 15.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """S11: continuous strong-PA LDPC link with burst interleaving and integer counts."""
    from .stage3_link_workload import run_mission_link as receive_burst

    return receive_burst(total_bits, num_slots, snr_db, seed)


# ---------------------------------------------------------------------------
# Reproducible Stage P4-B Coded BLER Benchmark Runner
# ---------------------------------------------------------------------------

def run_p4b_coded_bler_benchmarks(
    train_steps: int = 120,
    batch_size: int = 8,
    eval_slots: int = 24,
    n_symb: int = DEFAULT_N_SYMB,
    n_sc: int = DEFAULT_N_SC,
    seed: int = 2026,
    output_dir: Optional[Union[str, Path]] = "p4/results",
) -> Dict[str, Any]:
    """Evaluate coded BLER & BER/NMSE/ECE of EMFloatFormer2D, 2D-LMMSE, and 2D-IDD-PA
    across in-distribution and OOD Doppler / delay-spread regimes, saving JSON and CSV.
    """
    torch.set_num_threads(4)
    model = _train_emfloatformer_2d(
        steps=train_steps,
        batch_size=batch_size,
        n_symb=n_symb,
        n_sc=n_sc,
        profile="equal4",
        seed=seed,
    )
    param_count = int(sum(p.numel() for p in model.parameters()))

    regimes = [
        # (regime_name, profile, fd, cfo, pa_condition, snrs)
        ("in_dist_strong", "equal4", 0.01, 0.0, "strong", (10.0, 14.0, 18.0, 22.0)),
        ("in_dist_mild", "equal4", 0.01, 0.0, "mild", (10.0, 14.0, 18.0, 22.0)),
        ("ood_doppler_003", "equal4", 0.03, 0.02, "strong", (12.0, 16.0, 20.0)),
        ("ood_doppler_005", "equal4", 0.05, 0.02, "strong", (12.0, 16.0, 20.0)),
        ("ood_delay_long12", "long12", 0.01, 0.0, "strong", (12.0, 16.0, 20.0)),
        ("ood_delay_sparse12", "sparse12", 0.01, 0.0, "strong", (12.0, 16.0, 20.0)),
        ("ood_delay_exp16", "exp16", 0.01, 0.0, "strong", (12.0, 16.0, 20.0)),
    ]

    methods = [
        ("EMFloatFormer2D", "emfloatformer2d", param_count),
        ("2D-LMMSE", "lmmse_2d", 0),
        ("2D-IDD-PA", "idd_pa_2d", 0),
        ("2D-LMMSE-fixed", "lmmse_2d_fixed", 0),
        ("2D-IDD-PA-fixed", "idd_pa_2d_fixed", 0),
    ]

    rows: List[Dict[str, Any]] = []
    for r_idx, (reg_name, profile, fd, cfo, pa_cond, snrs) in enumerate(regimes):
        for s_idx, snr_db in enumerate(snrs):
            eval_seed = seed * 1000 + r_idx * 37 + s_idx
            for m_label, m_key, p_cnt in methods:
                res = _evaluate_slot_bler(
                    snr_db=float(snr_db),
                    num_slots=eval_slots,
                    n_symb=n_symb,
                    n_sc=n_sc,
                    fd=fd,
                    cfo=cfo,
                    profile=profile,
                    pa_condition=pa_cond,
                    method=m_key,
                    model=model if m_key == "emfloatformer2d" else None,
                    seed=eval_seed,
                )
                rows.append({
                    "regime": reg_name,
                    "profile": profile,
                    "fd": float(fd),
                    "cfo": float(cfo),
                    "pa_condition": pa_cond,
                    "snr_db": float(snr_db),
                    "method": m_label,
                    "params": int(p_cnt),
                    "bit_errors": int(res["bit_errors"]),
                    "total_bits": int(res["total_bits"]),
                    "block_errors": int(res["block_errors"]),
                    "total_blocks": int(res["total_blocks"]),
                    "bler": float(res["bler"]),
                    "ber": float(res["ber"]),
                    "uncoded_ber": float(res["uncoded_ber"]),
                    "uncoded_bit_errors": res["uncoded_bit_errors"],
                    "total_uncoded_bits": res["total_uncoded_bits"],
                    "nonfinite_count": res["nonfinite_count"],
                    "nmse": float(res["nmse"]),
                    "ece": float(res["ece"]),
                    "eval_seed": int(eval_seed),
                })

    summary = {
        "stage": "P4-B",
        "n_symb": n_symb,
        "n_sc": n_sc,
        "cp": DEFAULT_CP,
        "train_profile": "equal4",
        "train_pa_doppler": "All nine combinations of strong/mild/clean PA and Doppler 0/0.01/0.02, in a fixed cycle; CFO=0.",
        "frontend": "All receivers use observation-only CP-correlation frequency-offset correction on every slot. NMSE uses the rephased midpoint CFR.",
        "classical_priors": "2D-LMMSE and 2D-IDD-PA know the generating PDP, Doppler and PA law. Suffix -fixed uses equal4, fd=0.01, A=2.25, alpha=0.5 throughout.",
        "pairing": "All methods in each regime/SNR cell use identical seeded coded slots. Codes sharing one slot share fading. SNR cells use independent seeds.",
        "train_steps": train_steps,
        "batch_size": batch_size,
        "eval_slots": eval_slots,
        "ldpc_code": "Rate-1/2 (64, 128) PEG-IRA",
        "seed": seed,
        "num_threads": torch.get_num_threads(),
        "param_count_emfloatformer2d": param_count,
        "rows": rows,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / "p4b_coded_bler.json"
        csv_path = out_path / "p4b_coded_bler.csv"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage P4-B 2D Coded BLER Benchmarks.")
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--eval-slots", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="p4/results")
    args = parser.parse_args()
    out = run_p4b_coded_bler_benchmarks(
        train_steps=args.train_steps,
        eval_slots=args.eval_slots,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Completed Stage P4-B coded BLER benchmark: {len(out['rows'])} rows written to {args.output_dir}")
