"""Observation masks, held-out probe splits, and dynamic evidence benchmarks for Phase 7."""
from dataclasses import replace

import torch


REAL = torch.float64
CONDITIONS = (
    "C0_intact",
    "C1_row9_erased",
    "C2_row0_erased",
    "C3_scattered_half",
    "C4a_roving_window",
    "C5_holdout_update",
)
PILOT_AUDIT_CONDITIONS = (
    "C0_intact",
    "C1_row9_erased",
    "C2_row0_erased",
    "C3_scattered_half",
    "C4a_roving_window",
)


def _rng(seed):
    return torch.Generator().manual_seed(int(seed) % (2**63 - 1))


def mask_for_condition(pilot_mask, condition, batch_size, seed=0):
    """Return (gain_pass1[B,T,N], gain_pass2[B,T,N]|None, probe_mask[B,T,N]|None)."""
    if pilot_mask.ndim != 2 or pilot_mask.dtype != torch.bool:
        raise ValueError("Expected boolean [T,N] pilot_mask")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Phase 7 condition: {condition}")
    t, n = pilot_mask.shape
    rows = torch.arange(t)[:, None]
    tones = torch.arange(n)[None] // 2
    gain = torch.ones((batch_size, t, n), dtype=REAL)

    if condition == "C0_intact":
        return gain, None, None

    if condition == "C1_row9_erased":
        gain[:, pilot_mask & (rows == 9)] = 0.
        return gain, None, None

    if condition == "C2_row0_erased":
        gain[:, pilot_mask & (rows == 0)] = 0.
        return gain, None, None

    if condition == "C3_scattered_half":
        gain[:, pilot_mask & ((rows + tones) % 2 == 0)] = 0.
        return gain, None, None

    gen = _rng(int(seed) * 104729 + (17 if condition == "C4a_roving_window" else 31))
    if condition == "C4a_roving_window":
        pilot_cols = pilot_mask[9].nonzero().flatten()  # 16 even subcarriers
        for b in range(batch_size):
            other = 0 if int(torch.randint(2, (1,), generator=gen).item()) == 0 else 19
            gain[b, pilot_mask & ((rows == 9) | (rows == other))] = 0.
            k1 = int(torch.randint(len(pilot_cols) - 1, (1,), generator=gen).item())
            k2 = int(torch.randint(len(pilot_cols) - 1, (1,), generator=gen).item())
            gain[b, 9, pilot_cols[k1:k1 + 2]] = 1.
            gain[b, other, pilot_cols[k2:k2 + 2]] = 1.
        return gain, None, None

    # C5_holdout_update: Pass 1 erases row 9 completely; Pass 2 reveals 4 known row-9 pilots.
    gain[:, pilot_mask & (rows == 9)] = 0.
    probe_mask = torch.zeros((batch_size, t, n), dtype=torch.bool)
    pilot_cols = pilot_mask[9].nonzero().flatten()
    for b in range(batch_size):
        perm = torch.randperm(len(pilot_cols), generator=gen)[:4]
        probe_mask[b, 9, pilot_cols[perm]] = True
    gain2 = gain.clone()
    gain2[probe_mask] = 1.
    return gain, gain2, probe_mask


def apply_observation_gain(frame, gain):
    """Apply real receive gain [T,N] or [B,T,N] to frame.obs.y (keeps TX pilot_mask intact)."""
    if gain.ndim == 2:
        g = gain[None, None]
    elif gain.ndim == 3:
        g = gain[:, None]
    else:
        raise ValueError("Expected gain shape [T,N] or [B,T,N]")
    y = frame.obs.y * g.to(device=frame.obs.y.device, dtype=frame.obs.y.real.dtype)
    return replace(frame, obs=replace(frame.obs, y=y))
