"""Paired P4-B classical receiver study; lock the design with --seal before evaluating."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from p4.classical_baselines import bussgang_params, calibrated_16qam_llrs
from p4.next_classical import delay_doppler_receive
from p4.refit_pilot_prior import per_slot_errors
from p4.simulator import rapp_pa
from p4.slot_2d import (
    DEFAULT_SCS_HZ, SPEED_OF_LIGHT, SlotSimulator2D, _get_default_ldpc, _synchronize_slot, idd_pa_2d,
)


P4 = Path(__file__).resolve().parent
FOLDER = P4 / "results"
FREEZE = FOLDER / "next_classical_protocol.json"
GATE_SEED = 42
FRESH = [86601, 86602, 86603, 86604]
METHODS = ["legacy", "tap_pilots", "waveform_uncoded_feedback", "waveform_code_feedback",
           "genie_diagonal_reference"]
CONDITIONS = {
    "S01": dict(n_sc=32, profile="equal4", fd=(300 / 3.6) * 3e9 / SPEED_OF_LIGHT / DEFAULT_SCS_HZ,
                snr_db=15.0, pa_condition="strong", gate_slots=4, fresh_slots=16, synchronize=True),
    "S02": dict(n_sc=64, profile="long12", fd=0.005, snr_db=18.0, pa_condition="mild",
                gate_slots=20, fresh_slots=20, synchronize=True),
    "S11": dict(n_sc=80, profile="equal4", fd=0.01, snr_db=15.0, pa_condition="mild",
                gate_slots=50, fresh_slots=50, synchronize=False),
}


def design():
    files = ["next_classical.py", "next_classical_study.py", "slot_2d.py", "simulator.py",
             "classical_baselines.py", "refit_pilot_prior.py"]
    return {
        "study": "P4-B paired classical receiver continuation",
        "conditions": CONDITIONS, "fresh_seeds": FRESH, "historical_gate_seed": GATE_SEED,
        "development_seed": 86590, "methods": METHODS,
        "candidate": "waveform_code_feedback. 3 outer iterations. Jointly fit tap amplitudes and phase rates to code-assisted PA waveforms; no access to transmitted labels or realized taps.",
        "endpoint": "Post-decoder BLER at the unchanged absolute workload thresholds, and paired counts per slot and seed. The unchanged rate-1/2 (128,64) normalized min-sum decoder receives channel LLRs.",
        "guards": "Original simulation batch layouts and synchronization procedures preserved for each gate; no sender interleaving. S02 additionally requires channel NMSE < .05.",
        "diagnostic": "genie_diagonal_reference uses true time-varying taps and transmitted waveform only to remove PA/ICI, then applies the same diagonal soft demapper/decoder to the remaining faded observation. It is not eligible for the gates and is not a lower bound on optimal coded reception.",
        "source_sha256": {n: hashlib.sha256((P4 / n).read_bytes()).hexdigest() for n in files},
    }


def _genie(slot, cp):
    """Exact known-waveform subtraction, leaving a Bussgang linear channel and actual noise."""
    _, ns, nf = slot.y_grid.shape
    t = torch.arange(nf, dtype=torch.float64, device=slot.y_grid.device)
    midpoint_taps = slot.h_time_evol.to(torch.complex128)
    rates = torch.angle((midpoint_taps[:, 1:] * midpoint_taps[:, :-1].conj()).sum(1)) / (nf + cp)
    phase_from_mid = torch.exp(1j * rates[:, None, None, :] * (t - nf * 0.5)[:, None])
    time_taps = midpoint_taps.unsqueeze(2) * phase_from_mid
    payload = torch.fft.ifft(slot.x_grid, norm="ortho")
    pa = rapp_pa(payload, A=slot.A, alpha=slot.alpha, beta=slot.beta,
                 knee=slot.knee, oversample=slot.oversample)
    delayed = torch.stack([pa.roll(i, -1) for i in range(time_taps.shape[-1])], dim=-1)
    predicted = torch.fft.fft((time_taps * delayed).sum(-1), norm="ortho")
    noise = slot.y_grid - predicted
    gamma, _, _ = bussgang_params(A=slot.A, alpha=slot.alpha, beta=slot.beta, knee=slot.knee)
    h = gamma * slot.h_true.to(torch.complex128)
    power = h.abs().square().clamp_min(1e-12)
    equivalent = slot.x_grid + noise * h.conj() / power
    llrs = calibrated_16qam_llrs(equivalent, slot.noise_var.unsqueeze(-1) / power)
    check_noise = (noise.abs().square().mean((-2, -1)) / slot.noise_var.flatten()).tolist()
    return llrs[:, slot.data_mask, :].float(), check_noise


def _receive(slot, name):
    if name == "legacy":
        return idd_pa_2d(slot_data=slot, iterations=2)
    return delay_doppler_receive(
        slot.y_grid, slot.pilot_tx, slot.pilot_mask, slot.noise_var, cp=16,
        profile=slot.profile, A=slot.A, alpha=slot.alpha, beta=slot.beta,
        knee=slot.knee, oversample=slot.oversample,
        rounds=0 if name == "tap_pilots" else 3, code_aided=name == "waveform_code_feedback",
    )


@torch.inference_mode()
def run_cell(scenario, seed):
    cfg = CONDITIONS[scenario]
    num_slots = cfg["gate_slots"] if seed == GATE_SEED else cfg["fresh_slots"]
    sim = SlotSimulator2D(n_symb=14, cp=16, n_sc=cfg["n_sc"], profile=cfg["profile"], fd=cfg["fd"])
    code = _get_default_ldpc(128)
    records = {name: {"failed_blocks_per_slot": [], "failed_bits_per_slot": [],
                      "nmse_per_slot": []} for name in METHODS}
    generator = torch.Generator().manual_seed(seed)
    chunk = 25 if scenario == "S11" else num_slots
    noise_ratios = []
    for start in range(0, num_slots, chunk):
        slot = sim.generate_slot(batch_size=min(chunk, num_slots - start), snr_db=cfg["snr_db"],
                                 pa_condition=cfg["pa_condition"], coded=True, ldpc=code, generator=generator)
        if cfg["synchronize"]:
            slot = _synchronize_slot(slot)
        for name in METHODS:
            if name == "genie_diagonal_reference":
                llrs, ratios = _genie(slot, 16)
                noise_ratios.extend(ratios)
            else:
                res = _receive(slot, name)
                llrs = res["llrs"]
                nmse = ((res["h_hat"] - slot.h_true).abs().square().sum((-2, -1))
                        / slot.h_true.abs().square().sum((-2, -1)))
                records[name]["nmse_per_slot"].extend(nmse.tolist())
            if not torch.isfinite(llrs).all():
                raise FloatingPointError(f"Nonfinite LLR: {name} in {scenario}")
            failed, errors = per_slot_errors(llrs, slot, code)
            records[name]["failed_blocks_per_slot"].extend(failed)
            records[name]["failed_bits_per_slot"].extend(errors)
    blocks = (sim.num_data * 4 // code.n) * num_slots
    for rec in records.values():
        rec["block_errors"] = sum(rec["failed_blocks_per_slot"])
        rec["bit_errors"] = sum(rec["failed_bits_per_slot"])
        rec["bler"] = rec["block_errors"] / blocks
        if rec["nmse_per_slot"]:
            rec["nmse"] = sum(rec["nmse_per_slot"]) / num_slots
    return {"scenario": scenario, "seed": seed, "slots": num_slots, "blocks": blocks,
            "bits": blocks * code.k, "methods": records, "genie_mean_noise_power_ratio": sum(noise_ratios) / num_slots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true", help="write the pre-evaluation design and source identity")
    args = parser.parse_args()
    plan = design()
    if args.seal:
        if FREEZE.exists():
            assert json.loads(FREEZE.read_text()) == plan
        else:
            FREEZE.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        return
    if json.loads(FREEZE.read_text()) != plan:
        raise ValueError("The implemented recipe no longer matches the pre-evaluation design.")
    torch.set_num_threads(2)
    out = []
    for seed in [*FRESH, GATE_SEED]:
        for scenario in CONDITIONS:
            r = run_cell(scenario, seed)
            out.append(r)
            print(seed, scenario, r["blocks"], {k:v["block_errors"] for k,v in r["methods"].items()}, flush=True)
    data = {"recipe_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(), "rows": out}
    (FOLDER / "next_classical_evaluation.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
