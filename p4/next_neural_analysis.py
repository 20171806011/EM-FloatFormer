"""Reconcile exact counts and describe paired differences on held-out development slots."""
import json
from itertools import product

import numpy as np

from p4.next_neural import FREEZE, PROTOCOL, RESULTS, digest


def analyze():
    settings = json.loads(PROTOCOL.read_bytes())
    experiment = json.loads((RESULTS / "next_neural_study.json").read_bytes())
    if experiment["freeze"] != digest(FREEZE):
        raise ValueError("Source/recipe identity differs from the experiment")
    rows = experiment["rows"]
    lookup = {(r["condition"], r["method"]): r for r in rows}
    methods = [f"{arm}_{seed}" for seed in settings["training_seeds"] for arm in settings["learned_arms"]]
    methods += settings["pilot_controls"]
    conditions = [c[0] for c in settings["conditions"]]
    if len(lookup) != len(rows) or set(lookup) != set(product(conditions, methods)):
        raise ValueError("Expected one observation per condition and method")
    slots = settings["evaluation"]["batches"] * settings["evaluation"]["batch_size"]
    for r in rows:
        bb = r["failed_per_slot"]
        ib = r["wrong_bits_per_slot"]
        if (len(bb) != slots or len(ib) != slots or r["blocks"] != 13 * slots or r["bits"] != 64 * r["blocks"]
                or sum(bb) != r["failed_blocks"] or sum(ib) != r["wrong_bits"]
                or any(type(b) != int or type(i) != int or not (0 <= b <= 13 and b <= i <= 64*b)
                       for b, i in zip(bb, ib))):
            raise ValueError(f"Invalid integer counts for {r['condition']}, {r['method']}")
    delays = ("long12", "sparse12", "exp16")
    summaries = {}
    for method in methods:
        selected = [lookup[condition, method] for condition in delays]
        id_row, ood_row = [lookup[c, method] for c in ("s07_id", "s07_ood")]
        id_ber = id_row["wrong_bits"] / id_row["bits"]
        ood_ber = ood_row["wrong_bits"] / ood_row["bits"]
        summaries[method] = {
            "ood_block_errors": sum(r["failed_blocks"] for r in selected),
            "ood_blocks": sum(r["blocks"] for r in selected),
            "ood_bit_errors": sum(r["wrong_bits"] for r in selected),
            "ood_bits": sum(r["bits"] for r in selected),
            "s07_id_ber": id_ber, "s07_ood_ber": ood_ber,
            "s07_abs_gap": abs(ood_ber-id_ber), "s07_mean_shift_llr": ood_row["mean_shift_llr"],
        }
    seeds = settings["training_seeds"]

    def paired_interval(first, second):
        # Fit identifiers share slots. Average fits before sampling slot clusters.
        differences = []
        for condition in delays:
            by_fit = []
            for seed in seeds:
                a = first if first in settings["pilot_controls"] else f"{first}_{seed}"
                b = second if second in settings["pilot_controls"] else f"{second}_{seed}"
                av = np.asarray(lookup[condition, a]["failed_per_slot"])
                bv = np.asarray(lookup[condition, b]["failed_per_slot"])
                by_fit.append((av - bv) / 13)
            differences.append(np.mean(by_fit, axis=0))
        rng = np.random.default_rng(846821)
        replications = np.mean([rng.choice(d, size=(5000, slots), replace=True).mean(axis=1)
                                for d in differences], axis=0)
        return {"first": first, "second": second, "mean_bler_difference": float(np.mean(differences)),
                "conditional_slot_bootstrap_95pct": np.quantile(replications, [0.025, 0.975]).tolist()}

    contrasts = [paired_interval(a, b) for a, b in (
        ("fourier_no_channel", "fourier_full"),
        ("tap_phase_no_channel", "fourier_no_channel"),
        ("tap_phase_no_channel", "tap_phase_pilot_only"),
        ("tap_phase_no_channel", "fourier_full"),
        ("tap_phase_pilot_only", "fourier_pilot_only"),
    )]
    return {"freeze": experiment["freeze"], "summaries": summaries, "contrasts": contrasts,
            "interval_scope": "Descriptive paired 95% slot bootstrap, stratified equally over three delay profiles, conditional on the three fitted networks; no multiplicity adjustment or inference over training initializations."}


if __name__ == "__main__":
    out = analyze()
    (RESULTS/"next_neural_summary.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    for name, row in out["summaries"].items():
        print(name, row["ood_block_errors"], row["ood_blocks"], "S07-like abs BER gap", row["s07_abs_gap"])
    for c in out["contrasts"]:
        print(c["first"], "-", c["second"], c["mean_bler_difference"], c["conditional_slot_bootstrap_95pct"])
