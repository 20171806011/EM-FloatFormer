"""P4-B development study of a channel head removed throughout training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from p4.pilot_prior_probe import FourierPilotProbe
from p4.refit_pilot_prior import per_slot_errors
from p4.slot_2d import SlotSimulator2D, _get_default_ldpc, _synchronize_slot, _train_emfloatformer_2d

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "next_neural_protocol.json"
FREEZE = RESULTS / "next_neural_freeze.json"


class ZeroChannelHead(nn.Module):
    def forward(self, features):
        return features.new_zeros((*features.shape[:-1], 1))


class FourierNoChannel(FourierPilotProbe):
    """The tap-interpolated channel goes directly into the learned gain and demapper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove after identical initialization so every retained parameter is paired.
        self.head_h = ZeroChannelHead()


class TapPhaseNoChannel(FourierNoChannel):
    """Estimate a single phase slope per resolvable delay using the two comb observations."""

    def _compute_rbf_prior(self, yp_bar, t_p, f_p, t_g, f_g, n_symb, n_sc):
        times = torch.unique_consecutive(t_p)
        expected_f = torch.arange(0, n_sc, 2, device=f_p.device, dtype=f_p.dtype)
        if (n_sc % 2 or len(times) != 2 or not torch.equal(f_p, expected_f.repeat(2))
                or not torch.equal(t_p, times.repeat_interleave(len(expected_f)))):
            raise ValueError("This receiver requires two complete zero-offset comb-2 pilot rows.")
        taps = torch.fft.ifft(yp_bar.reshape(len(yp_bar), 2, n_sc // 2), dim=-1)
        first, second = taps.unbind(dim=1)
        turn = torch.angle(second * first.conj())
        aligned = 0.5 * (first + second * torch.polar(torch.ones_like(turn), -turn))
        noise_proxy = (first.abs() - second.abs()).square().mean(dim=-1, keepdim=True)
        power = 0.5 * (first.abs().square() + second.abs().square())
        gain = (1 - noise_proxy / power.clamp_min(torch.finfo(power.dtype).tiny)).clamp_min(0)
        position = (torch.arange(n_symb, device=t_p.device) - times[0]) / (times[1] - times[0])
        angle = position.view(1, -1, 1) * turn[:, None, :]
        channel_taps = (aligned * gain)[:, None, :] * torch.polar(torch.ones_like(angle), angle)
        return torch.fft.fft(channel_taps, n=n_sc, dim=-1)


ARMS = {"fourier_full": FourierPilotProbe, "fourier_no_channel": FourierNoChannel,
        "tap_phase_no_channel": TapPhaseNoChannel}


def pilot_control(factory):
    model = factory().eval()
    with torch.no_grad():
        for head in (model.head_h, model.head_dist):
            for param in head.parameters():
                param.zero_()
    return model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    files = [PROTOCOL, Path(__file__), HERE/"pilot_prior_probe.py", HERE/"refit_pilot_prior.py",
             HERE/"slot_2d.py", HERE/"simulator.py", HERE/"em_floatformer.py", HERE/"classical_baselines.py"]
    return {path.name: digest(path) for path in files}


@torch.inference_mode()
def evaluate(models, protocol):
    rows = []
    code = _get_default_ldpc(128)
    schedule = protocol["evaluation"]
    for ci, (condition, profile, fd, cfo, pa, snr) in enumerate(protocol["conditions"]):
        sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile=profile, fd=fd, cfo=cfo)
        counts = {key: {"failed_per_slot": [], "wrong_bits_per_slot": []} for key in models}
        shift_differences = {key: [] for key in models}
        for bi in range(schedule["batches"]):
            rng = torch.Generator().manual_seed(schedule["seed_base"] + 10 * ci + bi)
            slot = _synchronize_slot(sim.generate_slot(batch_size=schedule["batch_size"], snr_db=snr,
                pa_condition=pa, coded=True, ldpc=code, generator=rng))
            for key, model in models.items():
                llr = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)["llrs"]
                failed, errors = per_slot_errors(llr[:, slot.data_mask, :], slot, code)
                counts[key]["failed_per_slot"].extend(failed)
                counts[key]["wrong_bits_per_slot"].extend(errors)
                if condition == "s07_ood":
                    shifted_mask = slot.pilot_mask.roll(4, dims=-1)
                    pilot_grid = torch.zeros_like(slot.y_grid)
                    pilot_grid[:, slot.pilot_mask] = slot.yp_sparse
                    shifted = model(pilot_grid.roll(4, dims=-1)[:, shifted_mask], slot.y_grid.roll(4, dims=-1),
                                    shifted_mask, slot.noise_var)["llrs"].roll(-4, dims=-2)
                    shift_differences[key].append(float((shifted - llr).abs().mean()))
        blocks = slot.info_bits.shape[1] * schedule["batches"] * schedule["batch_size"]
        for key, obs in counts.items():
            shift = shift_differences[key]
            rows.append({"condition": condition, "method": key, "blocks": blocks, "bits": blocks * code.k,
                         "failed_blocks": sum(obs["failed_per_slot"]), "wrong_bits": sum(obs["wrong_bits_per_slot"]),
                         "mean_shift_llr": sum(shift) / len(shift) if shift else None, **obs})
    return rows


def run():
    spec = json.loads(PROTOCOL.read_bytes())
    if json.loads(FREEZE.read_bytes()) != source_identity():
        raise RuntimeError("The recipe or source no longer matches the development freeze.")
    threads = spec["training"]["threads"]
    torch.set_num_threads(threads)
    models = {"fourier_pilot_only": pilot_control(FourierPilotProbe),
              "tap_phase_pilot_only": pilot_control(TapPhaseNoChannel)}
    for seed in spec["training_seeds"]:
        for name, factory in ARMS.items():
            key = f"{name}_{seed}"
            path = RESULTS / f"next_neural_{key}.pt"
            if path.exists():
                checkpoint = torch.load(path, weights_only=True)
                if checkpoint["freeze"] != digest(FREEZE):
                    raise RuntimeError(f"Checkpoint {key} belongs to another recipe")
                model = factory().eval()
                model.load_state_dict(checkpoint["weights"])
            else:
                model = _train_emfloatformer_2d(steps=spec["training"]["steps"],
                    batch_size=spec["training"]["batch_size"], profile=spec["training"]["profile"],
                    seed=seed, model_factory=factory)
                tmp = path.with_suffix(".tmp")
                torch.save({"freeze": digest(FREEZE), "weights": model.state_dict()}, tmp)
                tmp.replace(path)
            models[key] = model
            print(f"Loaded or trained {key}", flush=True)
    rows = evaluate(models, spec)
    report = {"freeze": digest(FREEZE), "torch": torch.__version__, "threads": threads, "rows": rows}
    (RESULTS/"next_neural_study.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Scored {len(rows)} method-condition combinations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="record source identity before experiments")
    args = parser.parse_args()
    if args.freeze:
        with FREEZE.open("x", encoding="utf-8") as handle:
            json.dump(source_identity(), handle, indent=2)
    else:
        run()
