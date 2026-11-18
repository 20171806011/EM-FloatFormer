"""S02 paired initializer ablation using the same waveform estimator and soft-list decoder."""
import json

import torch

from p4.next_neural import TapPhaseNoChannel, pilot_control, digest
from p4.slot_2d import _synchronize_slot
from p4.stage3_urban import HERE, NOMINAL, generate, run_receiver, score, run_both_urban_profiles


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.manual_seed(838000)
    model = pilot_control(TapPhaseNoChannel)
    source = HERE/"results"/"stage3_urban.json"
    main = json.loads(source.read_bytes())
    records = []
    for row in main["records"]:
        slot = _synchronize_slot(generate(64,row["profile"],2.25,20,18.,row["seed"]))
        result = run_receiver("link",slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var,
                              NOMINAL,2.25,model)
        control = score(result,slot)
        trained = row["methods"]["link_nominal"]
        records.append({"seed":row["seed"],"profile":row["profile"],"untrained_initializer":control,
                        "trained_minus_untrained_blocks":trained["block_errors"]-control["block_errors"],
                        "trained_minus_untrained_bits":trained["bit_errors"]-control["bit_errors"]})
        print(row["seed"],row["profile"],control["block_errors"],flush=True)
    checks = run_both_urban_profiles()
    replay = {}
    for row in main["records"][-2:]:
        name = row["profile"]
        original = row["methods"]["link_nominal"]
        replay[name] = all(checks[name][key] == original[key] for key in original)
    if not all(replay.values()):
        raise AssertionError("Public S02 entry did not reproduce experimental counts")
    report = {"source_comparison_sha256":digest(source),"entrypoint_seed42_exact":replay,
              "scope":"Identical received waveforms, nominal long12 prior, physical waveform fitting and LDPC soft-list decoder. Only the initial decoder/demapper LLRs change between the trained 73161 and unit-gain, zero-distortion initializers; neither initializer has a trained channel correction.",
              "records":records}
    path = HERE/"results"/"stage3_urban_ablation.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    run()
