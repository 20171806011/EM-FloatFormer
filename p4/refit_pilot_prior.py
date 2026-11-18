"""Paired development experiment on seven fresh 18 dB conditions.

The trained 2D architecture and 120-step schedule are held fixed. The only
architecture change is frequency interpolation of the two comb pilot rows.
This study is separate from the unopened confirmatory seeds.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from p4.pilot_prior_probe import FourierPilotProbe
from p4.slot_2d import (
    EMFloatFormer2D, SlotSimulator2D, _get_default_ldpc, _synchronize_slot,
    _train_emfloatformer_2d, decode_min_sum, idd_pa_2d,
)

CONDITIONS = (
    ("equal4_strong", "equal4", 0.01, 0.00, "strong"),
    ("equal4_mild", "equal4", 0.01, 0.00, "mild"),
    ("doppler003", "equal4", 0.03, 0.02, "strong"),
    ("doppler005", "equal4", 0.05, 0.02, "strong"),
    ("long12", "long12", 0.01, 0.00, "strong"),
    ("sparse12", "sparse12", 0.01, 0.00, "strong"),
    ("exp16", "exp16", 0.01, 0.00, "strong"),
)


def per_slot_errors(llrs, slot, code):
    if not torch.isfinite(llrs).all():
        raise FloatingPointError("Nonfinite LLR in held-out comparison")
    batch, blocks, k = slot.info_bits.shape
    channel = llrs.reshape(batch, -1)[:, :blocks * code.n].reshape(-1, code.n)
    bits = decode_min_sum(channel, code.h_matrix).reshape(batch, blocks, k)
    mismatch = bits != slot.info_bits
    return mismatch.any(-1).sum(-1).tolist(), mismatch.sum((-1, -2)).tolist()


def run():
    folder = Path(__file__).resolve().parent / "results"
    torch.set_num_threads(4)
    baseline = EMFloatFormer2D(d_model=16, n_heads=2, depth=2).eval()
    baseline.load_state_dict(torch.load(folder / "p4b_seed2026_state.pt", weights_only=True))
    swap = FourierPilotProbe(d_model=16, n_heads=2, depth=2).eval()
    swap.load_state_dict(baseline.state_dict())
    refit = _train_emfloatformer_2d(steps=120, batch_size=8, seed=2026, model_factory=FourierPilotProbe)
    pilot_only = FourierPilotProbe(d_model=16, n_heads=2, depth=2).eval()
    with torch.no_grad():
        for head in (pilot_only.head_h, pilot_only.head_dist):
            for p in head.parameters():
                p.zero_()
    models = {"original": baseline, "frozen_swap": swap, "refit": refit,
              "fourier_pilot_only": pilot_only}
    code = _get_default_ldpc(128)
    records = []
    boot = np.random.default_rng(821905)
    with torch.inference_mode():
        for ci, (name, profile, fd, cfo, pa) in enumerate(CONDITIONS):
            sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile=profile, fd=fd, cfo=cfo)
            per_method = {m: {"block_errors": [], "bit_errors": []}
                          for m in (*models, "lmmse_fixed", "lmmse_matched")}
            for batch_index in range(8):
                seed = 821000 + 10 * ci + batch_index
                slot = _synchronize_slot(sim.generate_slot(batch_size=8, snr_db=18, pa_condition=pa,
                    coded=True, ldpc=code, generator=torch.Generator().manual_seed(seed)))
                for m in per_method:
                    if m in models:
                        pred = models[m](slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
                        llrs = pred["llrs"][:, slot.data_mask, :]
                    else:
                        obs = replace(slot, fd=0.01, profile="equal4", A=2.25, alpha=0.5) if m == "lmmse_fixed" else slot
                        llrs = idd_pa_2d(slot_data=obs, iterations=0)["llrs"]
                    failed, bit_errors = per_slot_errors(llrs, slot, code)
                    per_method[m]["block_errors"].extend(failed)
                    per_method[m]["bit_errors"].extend(bit_errors)

            blocks = slot.info_bits.size(1)
            n = len(per_method["original"]["block_errors"])
            delta = (np.asarray(per_method["refit"]["block_errors"])
                     - np.asarray(per_method["original"]["block_errors"])) / blocks
            draws = boot.choice(delta, size=(5000, n), replace=True).mean(axis=1)
            records.append({
                "condition": name, "profile": profile, "fd": fd, "cfo": cfo, "pa": pa,
                "slots": n, "blocks_per_slot": blocks, "info_bits_per_block": code.k,
                "block_errors": {key: sum(v["block_errors"]) for key, v in per_method.items()},
                "bit_errors": {key: sum(v["bit_errors"]) for key, v in per_method.items()},
                "refit_minus_original_bler": float(delta.mean()),
                "paired_slot_bootstrap_95pct": np.quantile(draws, [0.025, 0.975]).tolist(),
                "per_slot": per_method,
            })
    result = {
        "training": "Paired seed 2026, identical initialization, training frames, objective and schedule. 120 steps, batch 8, equal4 training.",
        "eval": "All methods see identical 18 dB LDPC-coded slots. 8 batches of 8 per condition. Seed 821000 + 10 * condition index + batch index.",
        "intervals": "Within-condition descriptive 95% bootstrap percentiles across paired slots, conditional on one fitted seed. No multiple-test adjustment.",
        "pilot_only": "Same Fourier prior, identity gamma, zero channel correction and zero distortion correction. Same soft demapper.",
        "records": records,
    }
    (folder / "p4b_refit_holdout.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    torch.save(refit.state_dict(), folder / "p4b_fourier_refit_state.pt")


if __name__ == "__main__":
    run()
