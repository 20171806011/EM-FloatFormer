"""Replay the P4-B integer counts, then change only the trained model's pilot interpolator."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from p4.slot_2d import (
    EMFloatFormer2D, SlotSimulator2D, _get_default_ldpc, _synchronize_slot,
    _train_emfloatformer_2d, decode_min_sum, get_dmrs_mask, idd_pa_2d,
)


class FourierPilotProbe(EMFloatFormer2D):
    """Fixed comb-2 pilot experiment for the 14 × 32 benchmark grid."""

    def _compute_rbf_prior(self, yp_bar, t_p, f_p, t_g, f_g, n_symb, n_sc):
        # Equally spaced comb observations identify n_sc/2 causal tap bins.
        # Transform each pilot row independently. Since both have the same
        # comb, the original separable Gaussian gives them these same time weights.
        pilot_times = torch.unique_consecutive(t_p)
        expected_f = torch.arange(0, n_sc, 2, device=f_p.device, dtype=f_p.dtype)
        if (n_sc % 2 or len(pilot_times) != 2
                or not torch.equal(f_p, expected_f.repeat(2))
                or not torch.equal(t_p, pilot_times.repeat_interleave(len(expected_f)))):
            raise ValueError("This probe requires two full, zero-offset comb-2 pilot rows.")
        pilot_rows = yp_bar.reshape(len(yp_bar), 2, n_sc // 2)
        taps = torch.fft.ifft(pilot_rows, dim=-1)
        row_cfr = torch.fft.fft(taps, n=n_sc, dim=-1)
        times = torch.arange(n_symb, dtype=t_p.dtype, device=t_p.device)
        distance = (times[:, None] - pilot_times) / self.sigma_t
        weights = torch.softmax(-0.5 * distance.square(), dim=-1)
        return weights.to(row_cfr.dtype) @ row_cfr


def delay_response(original, alternative):
    n_symb, n_sc = 14, 32
    mask = get_dmrs_mask(n_symb, n_sc)
    t_p, f_p = mask.nonzero().T.double()
    t_g, f_g = torch.ones_like(mask).nonzero().T.double()
    rows = []
    for delay in range(n_sc // 2):
        angles = -2 * torch.pi * f_g * delay / n_sc
        truth = torch.polar(torch.ones_like(angles), angles).reshape(n_symb, n_sc)
        pilot = truth[mask].unsqueeze(0)
        row = {"delay_samples": delay}
        for name, model in (("rbf", original), ("fourier", alternative)):
            prior = model._compute_rbf_prior(pilot, t_p, f_p, t_g, f_g, n_symb, n_sc)[0]
            row[name + "_coherent_gain"] = (prior * truth.conj()).mean().real.item()
            row[name + "_nmse"] = (prior - truth).abs().square().mean().item()
        rows.append(row)
    return rows


def count_errors(llrs, slot, code):
    """Recount using a per-codeword Hamming distance and the systematic information bits."""
    truth = slot.info_bits.reshape(-1, code.k)
    coded_len = slot.info_bits.size(1) * code.n
    channel_llrs = llrs.reshape(len(llrs), -1)[:, :coded_len].reshape(-1, code.n)
    decoded = decode_min_sum(channel_llrs, code.h_matrix, max_iter=25)
    distances = torch.count_nonzero(decoded != truth, dim=-1)
    return {
        "block_errors": int(torch.count_nonzero(distances)),
        "total_blocks": len(distances),
        "bit_errors": int(distances.sum()),
        "total_bits": truth.numel(),
        "uncoded_bit_errors": int(torch.count_nonzero((llrs > 0) != slot.bits)),
    }


@torch.inference_mode()
def run_probe(model, archive):
    alternative = FourierPilotProbe(d_model=16, n_heads=2, depth=2).eval()
    alternative.load_state_dict(model.state_dict())
    cells = {}
    for row in archive["rows"]:
        cells.setdefault((row["regime"], row["snr_db"]), {})[row["method"]] = row

    code = _get_default_ldpc(128)
    paired = []
    verified = 0
    for (regime, snr), receivers in cells.items():
        cfg = receivers["EMFloatFormer2D"]
        simulator = SlotSimulator2D(n_symb=archive["n_symb"], n_sc=archive["n_sc"],
                                    cp=archive["cp"], profile=cfg["profile"], fd=cfg["fd"], cfo=cfg["cfo"])
        slot = _synchronize_slot(simulator.generate_slot(
            batch_size=archive["eval_slots"], snr_db=snr, pa_condition=cfg["pa_condition"],
            coded=True, ldpc=code, generator=torch.Generator().manual_seed(cfg["eval_seed"]),
        ))
        for name, reference in receivers.items():
            if name == "EMFloatFormer2D":
                output = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
                llrs = output["llrs"][:, slot.data_mask, :]
            else:
                obs = slot
                if name.endswith("-fixed"):
                    obs = replace(slot, profile="equal4", fd=0.01, A=2.25, alpha=0.5)
                llrs = idd_pa_2d(slot_data=obs, iterations=0 if "LMMSE" in name else 2)["llrs"]
            observed = count_errors(llrs, slot, code)
            if any(observed[k] != reference[k] for k in observed):
                raise ValueError(f"Independent recount disagrees at {regime}, {snr}, {name}: {observed}")
            verified += 1
        output = alternative(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        changes = count_errors(output["llrs"][:, slot.data_mask, :], slot, code)
        denom = slot.h_true.abs().square().sum(dim=(-1, -2))
        nmse = ((output["h_hat"] - slot.h_true).abs().square().sum(dim=(-1, -2)) / denom).mean()
        paired.append({
            "regime": regime, "snr_db": snr, "eval_seed": cfg["eval_seed"],
            "rbf_block_errors": cfg["block_errors"], "rbf_bit_errors": cfg["bit_errors"],
            "rbf_nmse": cfg["nmse"], "fourier_nmse": nmse.item(), **changes,
        })
    return {
        "archive_integer_rows_verified": verified,
        "comparison": "Same trained weights, time interpolation and received slots. Only frequency interpolation changes from Gaussian weights to an inverse/zero-padded Fourier transform of each pilot row. No retraining.",
        "delay_response": delay_response(model, alternative), "paired_cells": paired,
    }


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--retrain", action="store_true", help="reproduce the 120-step P4-B checkpoint")
    args = cli.parse_args()
    folder = Path(__file__).resolve().parent / "results"
    archive = json.loads((folder / "p4b_coded_bler.json").read_text(encoding="utf-8"))
    torch.set_num_threads(archive["num_threads"])
    checkpoint = folder / "p4b_seed2026_state.pt"
    if args.retrain:
        model = _train_emfloatformer_2d(steps=archive["train_steps"], batch_size=archive["batch_size"],
                                       n_symb=archive["n_symb"], n_sc=archive["n_sc"],
                                       profile=archive["train_profile"], seed=archive["seed"])
        torch.save(model.state_dict(), checkpoint)
    else:
        model = EMFloatFormer2D(d_model=16, n_heads=2, depth=2)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    results = run_probe(model.eval(), archive)
    (folder / "p4b_pilot_prior_probe.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Replayed {results['archive_integer_rows_verified']} method/cell records.")
