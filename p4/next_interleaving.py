"""Paired transmitter control. Change only the placement of QAM symbols within a slot."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from p4.refit_pilot_prior import per_slot_errors
from p4.simulator import rapp_pa
from p4.slot_2d import (DEFAULT_SCS_HZ, SPEED_OF_LIGHT, SlotSimulator2D, _apply_continuous_slot_channel,
                        _draw_slot_taps, _get_default_ldpc, _synchronize_slot, generate_dmrs, idd_pa_2d)

SEEDS = (835301, 835302, 835303, 835304, 835305, 835306, 42)
PERMUTATION_SEED = 825011
HST_DOPPLER = (300 / 3.6) * 3e9 / SPEED_OF_LIGHT / DEFAULT_SCS_HZ
CASES = {
    "S01": (32, HST_DOPPLER, "equal4", 15., "strong", 4, 4),
    "S11": (80, .01, "equal4", 15., "mild", 50, 25),
    "S02": (64, .005, "long12", 18., "mild", 20, 20),
}


def transmit(sim, slot, order, taps, noise_state):
    """Physical-layer rerender from the same taps and AWGN draw for any symbol order."""
    symbols = slot.data_tx[:, order]
    grid = slot.x_grid.clone()
    grid[:, slot.data_mask] = symbols
    tx = torch.fft.ifft(grid, norm="ortho")
    tx = rapp_pa(tx, A=slot.A, alpha=slot.alpha, beta=slot.beta, knee=slot.knee, oversample=slot.oversample)
    tx = torch.cat([tx[..., -sim.cp:], tx], dim=-1).reshape(len(grid), -1)
    rng = torch.Generator().set_state(noise_state)
    rx, h, _ = _apply_continuous_slot_channel(tx, *taps, sim.n_symb, sim.n_sc, sim.cp, sim.cfo, slot.noise_var, rng)
    if not torch.equal(h, slot.h_true):
        raise AssertionError("Pairing failed: the channel realization changed")
    payload = rx.reshape(len(grid), sim.n_symb, sim.n_sc + sim.cp)[..., sim.cp:]
    y = torch.fft.fft(payload, norm="ortho")
    yp = y[:, slot.pilot_mask]
    return replace(slot, data_tx=symbols, x_grid=grid, bits=slot.bits[:, order],
                   rx_time=rx, y_grid=y, y_p_raw=yp, yp_sparse=yp*slot.pilot_tx.conj(),
                   y_d_raw=y[:, slot.data_mask])


def paired_batch(sim, batch, snr, pa, rng, code, order, verify):
    replay = torch.Generator().set_state(rng.get_state())
    slot = sim.generate_slot(batch_size=batch, coded=True, ldpc=code, snr_db=snr, pa_condition=pa, generator=rng)
    # Rewind the generator to the original channel draw without changing the
    # owning simulator. Its pilot, payload and padding draws are verified here.
    pilots = generate_dmrs(batch, sim.num_pilots, generator=replay)
    info = torch.randint(0, 2, slot.info_bits.shape, generator=replay)
    padding = sim.num_data*4 % code.n
    if padding:
        pad_bits = torch.randint(0, 2, (batch, padding), generator=replay)
        assert torch.equal(pad_bits, slot.bits.flatten(1)[:, -padding:])
    assert torch.equal(pilots, slot.pilot_tx) and torch.equal(info, slot.info_bits)
    taps = _draw_slot_taps(batch, sim.n_sc, sim.cp, sim.profile, sim.fd, replay, sim.device)
    noise_state = replay.get_state()
    if verify:
        identity = transmit(sim, slot, torch.arange(sim.num_data), taps, noise_state)
        assert torch.equal(identity.rx_time, slot.rx_time) and torch.equal(identity.y_grid, slot.y_grid)
    permuted = transmit(sim, slot, order, taps, noise_state)
    assert torch.equal(permuted.bits[:, torch.argsort(order)], slot.bits)
    return slot, permuted


@torch.inference_mode()
def run(receiver="legacy"):
    torch.set_num_threads(2)
    rows = []
    code = _get_default_ldpc(128)
    for case, (n_sc, fd, profile, snr, pa, original_slots, original_batch) in CASES.items():
        sim = SlotSimulator2D(n_symb=14, n_sc=n_sc, cp=16, profile=profile, fd=fd)
        order = torch.randperm(sim.num_data, generator=torch.Generator().manual_seed(PERMUTATION_SEED))
        inverse = torch.argsort(order)
        for seed in SEEDS:
            count, chunk = (original_slots, original_batch) if seed == 42 else (16, 4)
            rng = torch.Generator().manual_seed(seed)
            streams = {key: {"blocks_wrong": [], "info_bits_wrong": [], "raw_bits_wrong": 0}
                       for key in ("sequential", "permuted")}
            for offset in range(0, count, chunk):
                first, second = paired_batch(sim, min(chunk, count-offset), snr, pa, rng, code, order, offset == 0)
                for key, slot in (("sequential", first), ("permuted", second)):
                    inp = slot if case == "S11" else _synchronize_slot(slot)
                    if receiver == "waveform":
                        from p4.next_classical import delay_doppler_receive

                        active_order = order if key == "permuted" else None
                        result = delay_doppler_receive(inp.y_grid, inp.pilot_tx, inp.pilot_mask, inp.noise_var,
                            cp=sim.cp, profile=profile, A=inp.A, alpha=inp.alpha, physical_to_source=active_order)
                    else:
                        result = idd_pa_2d(slot_data=inp, iterations=2)
                    llrs = result["llrs"]
                    streams[key]["raw_bits_wrong"] += int(torch.count_nonzero((llrs > 0) != slot.bits))
                    if key == "permuted":
                        llrs = llrs[:, inverse]
                    blocks, bits = per_slot_errors(llrs, slot, code)
                    streams[key]["blocks_wrong"].extend(blocks)
                    streams[key]["info_bits_wrong"].extend(bits)
            rows.append({"case": case, "seed": seed, "slots": count, "blocks": count*(sim.num_data*4//code.n),
                         "bits_per_block": code.k, "raw_bits": count*sim.num_data*4, "methods": streams})
    filename = "next_interleaving_waveform.json" if receiver == "waveform" else "next_interleaving.json"
    target = Path(__file__).resolve().parent / "results" / filename
    result = {"mapping": "Fixed CPU torch.randperm of the QAM data REs, including filler symbols; inverse order before the unchanged LDPC decoder.",
              "permutation_seed": PERMUTATION_SEED, "development_seeds": list(SEEDS[:-1]), "legacy_seed": 42,
              "receiver": receiver, "synchronization": "Original S01/S02/S11 settings.",
              "pair_checks": "Verified common pilots, source bits, channel responses and byte-identical identity-mapping received samples for the first batch of every condition/seed.",
              "rows": rows}
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(rows)} paired condition/seed records to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", choices=["legacy", "waveform"], default="legacy")
    run(parser.parse_args().receiver)
