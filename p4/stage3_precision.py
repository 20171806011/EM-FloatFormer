"""Paired precision measurements with real low-precision GEMMs on the two-dimensional receiver."""
import copy
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .em_floatformer import ComplexLinear
from .next_neural import TapPhaseNoChannel
from .slot_2d import SlotSimulator2D, _synchronize_slot


class PackedProjection(nn.Module):
    """Frozen complex linear map in a single real GEMM with the specified operand precision."""

    def __init__(self, source, dtype):
        super().__init__()
        real, imag = source.weight_r.detach(), source.weight_i.detach()
        matrix = torch.cat((torch.cat((real, -imag), 1), torch.cat((imag, real), 1)))
        self.register_buffer("matrix", matrix.to(dtype))
        bias = None if source.bias_r is None else torch.cat((source.bias_r.detach(), source.bias_i.detach()))
        self.register_buffer("offset", None if bias is None else bias.to(dtype))

    def forward(self, z):
        pair = torch.cat((z.real, z.imag), dim=-1).to(self.matrix.dtype)
        components = F.linear(pair, self.matrix, self.offset).to(z.real.dtype)
        real, imag = components.chunk(2, dim=-1)
        return torch.complex(real, imag)


def engines():
    source = TapPhaseNoChannel().eval()
    weight = Path(__file__).resolve().parent / "results/next_neural_tap_phase_no_channel_73161.pt"
    source.load_state_dict(torch.load(weight, weights_only=True)["weights"])
    models = {}
    for precision in (torch.float32, torch.bfloat16, torch.float16):
        model = copy.deepcopy(source)
        for name, module in list(model.named_modules()):
            if isinstance(module, ComplexLinear):
                model.set_submodule(name, PackedProjection(module, precision))
        models[str(precision)] = model
    return models


def sample(batch, seed, snr):
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=.01)
    slot = _synchronize_slot(sim.generate_slot(batch_size=batch, snr_db=snr, pa_condition="strong",
                                       generator=torch.Generator().manual_seed(seed)))
    return slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var


@torch.inference_mode()
def audit(num_slots=100, snr_grid=(0, 5, 10, 15, 20)):
    if not snr_grid or num_slots < len(snr_grid):
        raise ValueError("At least one slot per SNR is required.")
    models = engines()
    rows, times = [], defaultdict(list)
    for i, snr in enumerate(snr_grid):
        slots = num_slots // len(snr_grid) + int(i < num_slots % len(snr_grid))
        result = {key: {"flips": 0, "bits": 0, "max_logit_error": 0., "nonfinite": 0} for key in models}
        for offset in range(0, slots, 4):
            batch = min(4, slots-offset)
            args = sample(batch, 934000 + num_slots*i + offset, snr)
            names = list(models)
            j = (i + offset//4) % len(names)
            predictions = {}
            for name in names[j:] + names[:j]:
                if i == 0 and offset == 0:
                    models[name](*args)
                start = perf_counter()
                out = models[name](*args)
                times[name].append((batch, perf_counter()-start))
                result[name]["nonfinite"] += sum(int((~torch.isfinite(v)).sum()) for v in out)
                predictions[name] = out.llrs[:, ~args[2]]
            ref = predictions["torch.float32"]
            for name, pred in predictions.items():
                case = result[name]
                case["flips"] += int(torch.count_nonzero((pred > 0) != (ref > 0)))
                case["bits"] += ref.numel()
                case["max_logit_error"] = max(case["max_logit_error"], float((pred-ref).abs().max()))
        rows.extend({"dtype": name, "snr_db": snr, "slots": slots, **case} for name, case in result.items())
    rates = {name: sum(n for n, _ in t)/sum(seconds for _, seconds in t) for name, t in times.items()}
    flip_rates = {name: sum(r["flips"] for r in rows if r["dtype"] == name) /
                        sum(r["bits"] for r in rows if r["dtype"] == name) for name in models}
    cpu_features = dict(getattr(torch.cpu, "get_capabilities", lambda: {})())
    reduced = [r for r in rows if r["dtype"] != "torch.float32"]
    return {"rows": rows, "flip_rate": max(flip_rates.values()), "flip_rates": flip_rates,
            "flips": sum(r["flips"] for r in reduced), "total_bits": sum(r["bits"] for r in reduced),
            "nan_count": sum(r["nonfinite"] for r in rows), "slots_per_second": rates,
            "median_seconds_per_slot": {k: median(s/n for n, s in v) for k, v in times.items()},
            "bf16_speedup": rates["torch.bfloat16"]/rates["torch.float32"],
            "cpu_capabilities": cpu_features, "threads": torch.get_num_threads(),
            "scope": "Real FP32/BF16/FP16 projection GEMMs. Gauge normalization, FFT, Hermitian scores, "
                     "residual accumulation and final LLR evaluation use wide arithmetic. Timings exclude "
                     "model construction, data generation and scoring."}


@torch.inference_mode()
def shadowing(scales, dtype=torch.bfloat16):
    model = engines()[str(dtype)]
    yp, y, mask, nv = sample(8, 936018, 10.)
    ref = model(yp, y, mask, nv).llrs
    rows = []
    for scale in scales:
        out = model(yp*scale, y*scale, mask, nv*scale**2)
        rows.append({"scale": scale, "max_logit_error": float((ref-out.llrs).abs().max()),
                     "flips": int(torch.count_nonzero((ref > 0) != (out.llrs > 0))),
                     "nan_count": sum(int((~torch.isfinite(v)).sum()) for v in out)})
    bits = len(rows)*ref.numel()
    return {"rows": rows, "max_logit_error": max(r["max_logit_error"] for r in rows),
            "total_bits": bits, "flips": sum(r["flips"] for r in rows),
            "flip_rate": sum(r["flips"] for r in rows)/bits, "nan_count": sum(r["nan_count"] for r in rows)}


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = {"mixed": audit(), "shadowing": shadowing([2.**n for n in range(-12, 13, 4)])}
    path = Path(__file__).resolve().parent / "results/stage3_precision.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(result["mixed"]["flip_rates"], result["mixed"]["bf16_speedup"], result["shadowing"]["max_logit_error"])
