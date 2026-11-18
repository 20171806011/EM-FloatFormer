"""Reconcile S05 raw observations and compute paired, draw-cluster bootstrap intervals."""
import json
import math

import numpy as np
import torch

from p4.next_pilots import sparse_slot
from p4.stage3_sparse import load_receiver
from p4.stage3_sparse_study import DRAWS, FREEZE, RESULTS, SEEDS, STEPS, protocol, sha


@torch.inference_mode()
def numeric_gauge_review(seed):
    """Additional diagnostic on the separate unit-test draw, after training."""
    torch.set_num_threads(2)
    model = load_receiver(seed)
    slot = sparse_slot(848888, 4, 2, 20.)
    result = {}
    for dtype in (torch.complex64,torch.complex128):
        y = slot.y_grid.to(dtype)
        nv = slot.noise_var.to(y.real.dtype)
        base = model(y,slot.pilot_tx,slot.pilot_mask,nv)
        rows = []
        for exponent,phase in [(-12,.37),(-5,1.91),(0,math.pi/7),(6,2.718),(12,5.1)]:
            radius = 2.**exponent
            c = torch.polar(torch.tensor(radius,dtype=torch.float64),torch.tensor(phase,dtype=torch.float64))
            moved = model((y.to(torch.complex128)*c).to(dtype),slot.pilot_tx,slot.pilot_mask,nv*radius**2)
            expected = base.h_hat.to(torch.complex128)*c
            rows.append({"power_of_two":exponent,"phase":phase,
                         "llr_max_error":float((moved.llrs-base.llrs).abs().max()),
                         "hard_bit_flips":int(((moved.llrs>0)!=(base.llrs>0)).sum()),
                         "channel_rel_l2":float((moved.h_hat-expected).norm()/expected.norm())})
        result[str(dtype)] = rows
    return result


def analyze():
    raw_path = RESULTS / "stage3_sparse_evaluation.json"
    raw = json.loads(raw_path.read_bytes())
    assert raw["protocol_sha256"] == sha(FREEZE)
    assert json.loads(FREEZE.read_bytes()) == protocol()
    rows = raw["rows"][:-1]
    gate = raw["rows"][-1:]
    assert [row["seed"] for row in rows] == DRAWS and gate[0]["seed"] == 42
    assert all(row["slots"] == 8 and row["bits_per_slot"] == 1760 for row in [*rows, *gate])
    ckpts = {}
    for seed in SEEDS:
        for arm in ("full", "cut"):
            path = RESULTS / f"stage3_sparse_{arm}_{seed}.pt"
            entry = torch.load(path, weights_only=True)
            assert (entry["freeze"], entry["arm"], entry["seed"]) == (sha(FREEZE), arm, seed)
            assert entry["history"] == raw["training"][f"{seed}/{arm}"]
            assert entry["history"]["steps"] == STEPS
            ckpts[path.name] = sha(path)

    def panel(subset, name, field):
        method = lambda r: np.array(r["methods"][name][field], dtype=np.float64)
        return np.array([method(r) for r in subset])

    for subset, summary in ((rows, raw["fresh"]), (gate, raw["legacy_gate"])):
        nearest = panel(subset, "nearest_ls", "nmse_per_slot")
        for method, values in summary.items():
            nmse = panel(subset, method, "nmse_per_slot")
            errors = panel(subset, method, "bit_errors_per_slot")
            assert np.isfinite(nmse).all() and (nmse > 0).all()
            assert np.equal(errors, np.floor(errors)).all() and (errors >= 0).all() and (errors <= 1760).all()
            assert np.isclose(nmse.mean(),values["mean_nmse"],rtol=1e-12)
            assert np.isclose(10*np.log10(nearest.mean()/nmse.mean()), values["gain_db"], atol=1e-12)
            assert np.count_nonzero(10*np.log10(nearest.sum(1)/nmse.sum(1))>=4) == values["draws_at_least_4db"]
            assert errors.sum() == values["bit_errors"]

    # The three fits share all test slots. Average over fits, resample the 16 entire draws jointly.
    rng = np.random.default_rng(869032)
    bootstrap = rng.integers(0, len(rows), (10000,len(rows)))
    def combined(method, field):
        names = [f"{s}/{method}" for s in SEEDS] if method in ("full","frozen_cut","retrained_cut") else [method]
        return np.mean([panel(rows,n,field) for n in names],axis=0).mean(1)

    learned = combined("full","nmse_per_slot")
    learned_errors = combined("full","bit_errors_per_slot")
    contrasts = {}
    for method in ("physics6","frozen_cut","retrained_cut","physics12","legacy_map6","nearest_ls"):
        control = combined(method,"nmse_per_slot")
        control_errors = combined(method,"bit_errors_per_slot")
        gain = 10*np.log10(control.mean()/learned.mean())
        boot_gain = 10*np.log10(control[bootstrap].mean(1)/learned[bootstrap].mean(1))
        delta_ber = (control_errors-learned_errors)/1760
        contrasts[method] = {
            "control_over_full_nmse_db":float(gain),
            "interval_db": np.quantile(boot_gain,[.025,.975]).tolist(),
            "draws_lower_nmse_full":int(np.sum(control > learned)),
            "control_minus_full_ber_percentage_points":float(delta_ber.mean()*100),
            "interval_ber_percentage_points":np.quantile(100*delta_ber[bootstrap].mean(1),[.025,.975]).tolist(),
        }

    stages = {}
    for mode in ("physics6","physics12","full","frozen_cut","retrained_cut"):
        names = [f"{s}/{mode}" for s in SEEDS] if mode in ("full","frozen_cut","retrained_cut") else [mode]
        keys = rows[0]["stages"][names[0]]
        stages[mode] = {
            key:{field.removeprefix("sum_slot_"):
                 float(np.mean([r["stages"][n][key][field] / r["slots"] for r in rows for n in names]))
                 for field in values} for key,values in keys.items()
        }
    result = {
        "raw_file_sha256":sha(raw_path),"protocol_sha256":sha(FREEZE),"checkpoint_sha256":ckpts,
        "independently_reconciled_method_summaries":sum(len(raw[s]) for s in ("fresh","legacy_gate")),
        "clusters":len(rows),"slots_per_cluster":8,"bits_per_fit":len(rows)*8*1760,
        "intervals":"Two-sided percentile 95% paired cluster bootstrap, 10000 draws. Training seeds held fixed, shared slots never counted multiple times as independent. Descriptive, no multiplicity adjustment.",
        "positive_contrast":"Full receiver attains lower NMSE/BER than the named control.",
        "fresh_paired_contrasts":contrasts,"fresh_stage_means":stages,
        "gate_default_first_seed":raw["legacy_gate"][f"{SEEDS[0]}/full"],
        "supplementary_complex_gauge_by_fit":{str(s):numeric_gauge_review(s) for s in SEEDS},
    }
    path = RESULTS/"stage3_sparse_analysis.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(analyze()["fresh_paired_contrasts"],indent=2))
