"""Compare the S06 receiver with both matched classical baselines on paired coded slots."""
from collections import Counter
from pathlib import Path

import torch

from .next_neural import TapPhaseNoChannel, pilot_control
from .stage3_link import receive
from .refit_pilot_prior import per_slot_errors
from .slot_2d import SlotSimulator2D, _get_default_ldpc, _synchronize_slot, idd_pa_2d


def paired_p(differences):
    """Exact two-sided randomization over independent integer slot differences."""
    population = Counter({0: 1})
    for difference in differences:
        updated = Counter()
        for total, count in population.items():
            updated[total + difference] += count
            updated[total - difference] += count
        population = updated
    observed = abs(sum(differences))
    return sum(n for total, n in population.items() if abs(total) >= observed) / 2**len(differences)


@torch.inference_mode()
def coded_stress(num_slots, seed):
    if num_slots < 1:
        raise ValueError("At least one slot is required for the paired comparison.")
    result_dir = Path(__file__).resolve().parent / "results"
    weight = result_dir / "next_neural_tap_phase_no_channel_73161.pt"
    model = TapPhaseNoChannel().eval()
    model.load_state_dict(torch.load(weight, weights_only=True)["weights"])
    code = _get_default_ldpc(128)
    simulator = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="long6", fd=.03, cfo=.04)
    generator = torch.Generator().manual_seed(seed)
    pilot = pilot_control(TapPhaseNoChannel)
    methods = ("EMFloatFormer_with_waveform_feedback", "pilot_with_waveform_feedback",
               "EMFloatFormer", "2D-LMMSE", "2D-IDD-PA")
    errors = {method: [] for method in methods}
    info = {method: [] for method in methods}
    total_blocks = 0
    for start in range(0, num_slots, 12):
        slot = _synchronize_slot(simulator.generate_slot(batch_size=min(12, num_slots-start), snr_db=18.0,
                                   coded=True, ldpc=code, A=1.25, alpha=.5, generator=generator))
        total_blocks += slot.info_bits.numel() // code.k
        for method in methods:
            if method.endswith("_with_waveform_feedback"):
                estimate = receive(slot.y_grid, slot.pilot_tx, slot.pilot_mask, slot.noise_var,
                                   cp=16, profile="long6", A=1.25, alpha=.5,
                                   neural=model if method.startswith("EM") else pilot)
                llrs, channel = estimate["llrs"], estimate["h_hat"]
            elif method == "EMFloatFormer":
                pred = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
                llrs, channel = pred.llrs[:, slot.data_mask], pred.h_hat
            else:
                pred = idd_pa_2d(slot_data=slot, iterations=0 if method == "2D-LMMSE" else 2)
                llrs, channel = pred["llrs"], pred["h_hat"]
            if not (torch.isfinite(llrs).all() and torch.isfinite(channel).all()):
                raise FloatingPointError(f"Nonfinite S06 predictions from {method}")
            blocks, bits = per_slot_errors(llrs, slot, code)
            errors[method].extend(blocks)
            info[method].extend(bits)
    neural = errors["EMFloatFormer_with_waveform_feedback"]
    contrasts = {}
    for name in ("2D-LMMSE", "2D-IDD-PA"):
        diff = [b - n for b, n in zip(errors[name], neural)]
        contrasts[name] = {"baseline_minus_model": sum(diff), "two_sided_p": paired_p(diff)}
    first, second = sorted(contrasts, key=lambda k: contrasts[k]["two_sided_p"])
    contrasts[first]["holm"] = min(1., 2 * contrasts[first]["two_sided_p"])
    contrasts[second]["holm"] = max(contrasts[first]["holm"], contrasts[second]["two_sided_p"])
    return {"outputs_finite": True, "p_value": max(c["holm"] for c in contrasts.values()),
            "reduction_vs_both": all(c["baseline_minus_model"] > 0 for c in contrasts.values()),
            "model_training_seed": 73161, "evaluation_seed": seed, "slots": num_slots,
            "primary_receiver": methods[0], "outer_decoder_for_all": "25-iteration min-sum",
            "total_blocks_per_method": total_blocks, "info_bits_per_method": total_blocks * code.k,
            "block_errors_per_slot": errors, "bit_errors_per_slot": info, "contrasts": contrasts}
