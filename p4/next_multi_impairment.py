"""S06 coded BLER panel against the two classical receivers in the written plan."""
import hashlib
import itertools
import json
from pathlib import Path

import torch

from p4.next_neural import FREEZE, TapPhaseNoChannel, pilot_control
from p4.next_classical import delay_doppler_receive
from p4.refit_pilot_prior import per_slot_errors
from p4.slot_2d import SlotSimulator2D, _get_default_ldpc, _synchronize_slot, idd_pa_2d


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent / "results"
    models = {"pilot_only": pilot_control(TapPhaseNoChannel)}
    fits = (73161, 73162, 73163)
    for s in fits:
        weights = torch.load(root / f"next_neural_tap_phase_no_channel_{s}.pt", weights_only=True)
        assert weights["freeze"] == hashlib.sha256(FREEZE.read_bytes()).hexdigest()
        model = TapPhaseNoChannel().eval()
        model.load_state_dict(weights["weights"])
        models[str(s)] = model
    code = _get_default_ldpc(128)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="long6", fd=.03, cfo=.04)
    rows = []
    for s in range(835701, 835713):
        slot = _synchronize_slot(sim.generate_slot(batch_size=8, coded=True, ldpc=code, snr_db=18.,
            A=1.25, alpha=.5, generator=torch.Generator().manual_seed(s)))
        outcomes = {}
        for key in (*models, "2D-LMMSE", "2D-IDD-PA", "waveform_feedback"):
            if key in models:
                pred = models[key](slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
                llrs = pred.llrs[:, slot.data_mask]
            elif key == "waveform_feedback":
                llrs = delay_doppler_receive(slot.y_grid, slot.pilot_tx, slot.pilot_mask, slot.noise_var,
                    cp=16, profile="long6", A=1.25, alpha=.5)["llrs"]
            else:
                llrs = idd_pa_2d(slot_data=slot, iterations=0 if key == "2D-LMMSE" else 2)["llrs"]
            blocks, bits = per_slot_errors(llrs, slot, code)
            outcomes[key] = {"block_errors": blocks, "info_bit_errors": bits}
        rows.append({"seed": s, "slots": len(slot.rx_time), "codewords_per_slot": slot.info_bits.shape[1],
                     "methods": outcomes})
    contrasts = []
    for key in ("2D-LMMSE", "2D-IDD-PA"):
        # Three fits see the same slots; pair at the 12 independent draw groups.
        # Keep the differences as integers in units of three times the count.
        d = [3*sum(r["methods"][key]["block_errors"])
             -sum(sum(r["methods"][str(f)]["block_errors"]) for f in fits) for r in rows]
        observed = abs(sum(d))
        extreme = sum(abs(sum(s*x for s,x in zip(signs,d))) >= observed
                      for signs in itertools.product((-1,1),repeat=len(d)))
        contrasts.append({"comparator":key, "baseline_minus_network_block_counts_times_three": d,
                          "sign_flip_two_sided_p": extreme/2**len(d)})
    ordered = sorted(contrasts,key=lambda c:c["sign_flip_two_sided_p"])
    ordered[0]["holm"] = min(1.,2*ordered[0]["sign_flip_two_sided_p"])
    ordered[1]["holm"] = max(ordered[0]["holm"],ordered[1]["sign_flip_two_sided_p"])
    result = {"study":"Written S06 multi-impairment workload, development only. 18 dB, long6, fd=.03, CFO=.04, strong PA.",
              "receiver_information":"All use the same blind CP correction. Matched classical baselines know the nominal channel/PA statistics.",
              "endpoint":"Paired LDPC BLER. Each of 12 independent test groups includes 8 slots, 13 words per slot. Candidate averages the three prefixed fits on each group. Two-sided sign-flip assumes sign exchangeability under the null; Holm family is the two planned classical comparisons.",
              "additional_comparator":"waveform_feedback is a descriptive strong receiver control outside the primary statistical family.",
              "model_seeds":fits, "rows":rows, "tests":contrasts}
    (root/"next_multi_impairment.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("S06 conditional paired contrasts:",contrasts)


if __name__ == "__main__":
    run()
