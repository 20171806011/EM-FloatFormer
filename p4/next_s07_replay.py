"""Run the unchanged S07 workload with its prespecified candidate and audit the trained model."""
import cmath
import json
from pathlib import Path

import torch

from p4 import slot_2d
from p4.next_neural import TapPhaseNoChannel
from p4.refit_pilot_prior import per_slot_errors


def run():
    torch.set_num_threads(4)
    model = slot_2d._train_emfloatformer_2d(steps=75, batch_size=8, n_symb=14, n_sc=32,
                                            profile="equal4", seed=2026, model_factory=TapPhaseNoChannel)
    code = slot_2d._get_default_ldpc(128)
    counts = {}
    with torch.inference_mode():
        for profile in ("equal4", "exp16"):
            sim = slot_2d.SlotSimulator2D(n_symb=14, n_sc=32, cp=16, fd=0.01, profile=profile)
            slot = slot_2d._synchronize_slot(sim.generate_slot(
                batch_size=32, coded=True, snr_db=20, pa_condition="mild",
                generator=torch.Generator().manual_seed(2526)))
            base = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
            blocks, bits = per_slot_errors(base["llrs"][:, slot.data_mask], slot, code)
            counts[profile] = {"bit_errors": sum(bits), "bits": slot.info_bits.numel(),
                               "block_errors": sum(blocks), "blocks": slot.info_bits.numel() // code.k}
        # S07 circular shift by four physical tones, including the pilots.
        pilot_grid = torch.zeros_like(slot.y_grid)
        pilot_grid[:, slot.pilot_mask] = slot.yp_sparse
        shift_mask = slot.pilot_mask.roll(4, -1)
        shifted = model(pilot_grid.roll(4, -1)[:, shift_mask], slot.y_grid.roll(4, -1),
                        shift_mask, slot.noise_var)
        shift_error = (shifted.llrs.roll(-4, -2) - base.llrs).abs().mean().item()
        id_rate = counts["equal4"]["bit_errors"] / counts["equal4"]["bits"]
        ood_rate = counts["exp16"]["bit_errors"] / counts["exp16"]["bits"]
        workload = {"shortcut_collapse": ood_rate > .1 or shift_error > 1.,
                    "ber_gap": abs(id_rate - ood_rate), "ber_id": id_rate, "ber_ood": ood_rate,
                    "shift_diff": shift_error}
        checks = {}
        # Check physical input rounding separately from higher precision arithmetic.
        for dtype in (torch.complex64, torch.complex128):
            yp = slot.yp_sparse[:4].to(dtype)
            yg = slot.y_grid[:4].to(dtype)
            nv = slot.noise_var[:4].to(yp.real.dtype)
            mask = slot.pilot_mask
            ref = model(yp, yg, mask, nv)
            rows = []
            for r, phi in ((2**-12, -1.31), (1.0, 2.07), (3.7, .57), (2**12, 1.13)):
                c = r * cmath.exp(1j * phi)
                out = model(yp*c, yg*c, mask, nv*(r*r))
                rows.append({"gain": r, "phase": phi,
                             "max_llr_error": (out.llrs-ref.llrs).abs().max().item(),
                             "decision_flips": int(((out.llrs > 0) != (ref.llrs > 0)).sum()),
                             "relative_channel_error": ((out.h_hat-c*ref.h_hat).norm()/(c*ref.h_hat).norm()).item()})
            checks[str(dtype)] = rows
    result = {"candidate": "TapPhaseNoChannel", "training_steps": 75, "batch_size": 8,
              "training_seed": 2026, "workload": workload, "counts": counts,
              "passed_s07": not workload["shortcut_collapse"] and workload["ber_gap"] <= .005,
              "gauge_checks_on_four_exp16_slots": checks}
    dest = Path(__file__).resolve().parent / "results/next_s07_replay.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"counts": counts, "workload": workload}))


if __name__ == "__main__":
    run()
