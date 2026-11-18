"""Uncoded fractional-delay S07 measurement for an already fitted receiver."""
import torch

from .next_fractional import FD, N, S, CP, delay_geometry, dft8, make_slot, transmit
from .simulator import complex_randn
from .slot_2d import SLOT_PROFILES, SlotSimulator2D


@torch.inference_mode()
def evaluate_fractional(model, seed, batches=8):
    simulator = SlotSimulator2D(n_symb=S, n_sc=N, cp=CP, profile="equal4", fd=FD)
    profile = torch.tensor(SLOT_PROFILES["exp16"])
    powers = (profile / profile.sum()).sqrt()
    delays = delay_geometry(len(profile), fractional=True)
    methods = ("model", "dft8_linear", "dft8_tap_phase")
    counts = {name: 0 for name in methods}
    channel_errors = {name: 0.0 for name in methods}
    bits = slots = 0
    for batch in range(batches):
        base = seed + 4 * batch
        gen = [torch.Generator().manual_seed(base + j) for j in range(4)]
        slot = simulator.generate_slot(batch_size=8, snr_db=20.0, pa_condition="mild",
                                       coded=False, generator=gen[0])
        gains = complex_randn((8, len(profile)), generator=gen[1]) * powers
        dopplers = FD * torch.cos(2 * torch.pi * torch.rand((8, len(profile)), generator=gen[2]))
        noise = complex_randn((8, S * (N + CP)), generator=gen[3], dtype=torch.complex128)
        noise *= slot.noise_var.double().sqrt()
        slot = make_slot(slot, transmit(slot), gains, dopplers, delays, noise, "exp16")
        out = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        inputs = (slot.yp_sparse, slot.y_grid, slot.noise_var)
        for name, (llr, channel) in zip(methods, ((out.llrs, out.h_hat),
                                                  dft8(*inputs, False), dft8(*inputs, True))):
            if not (torch.isfinite(llr).all() and torch.isfinite(channel).all()):
                raise FloatingPointError(f"Nonfinite S07 prediction from {name}")
            counts[name] += int(((llr[:, slot.data_mask] > 0) != slot.bits).sum())
            nmse = (channel - slot.h_true).abs().square().sum((1, 2)) / slot.h_true.abs().square().sum((1, 2))
            channel_errors[name] += float(nmse.double().sum())
        bits += slot.bits.numel()
        slots += len(slot.bits)
    return {"seed": seed, "slots": slots, "uncoded_bits": bits, "bit_errors": counts,
            "channel_nmse": {k: v / slots for k, v in channel_errors.items()},
            "ber_minus_dft8": (counts["model"] - counts["dft8_linear"]) / bits}
