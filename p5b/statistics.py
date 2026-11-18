"""Aggregate validated pair records, preserving integer errors and the registered families."""
import csv
import json
import math
from pathlib import Path
from statistics import fmean

from p5.statistics import COUNT_FIELDS, ENERGY_FIELDS, _rates, comparison, holm, kernel_row


OUT = Path(__file__).resolve().parent / "results"
ARCHIVE = OUT.parents[1] / "p5" / "results"
ARMS = ("pure", "parity", "local", "local_parity", "no_gauge")
LABEL = {"pure": "OpEM-PureWavefield", "parity": "OpEM-SoftParity",
         "local": "OpEM-LocalConv (pure)", "local_parity": "OpEM-LocalConv (parity)",
         "no_gauge": "OpEM-NoGauge (pure)", "physics": "Fixed wavefield",
         "physics_parity": "Fixed wavefield + parity", "p5_full": "P5 Unrolled EM",
         "nominal_bussgang_lmmse": "Nominal Bussgang-LMMSE", "nominal_idd_pa": "Nominal IDD-PA",
         "pilot_cpe_dd": "Pilot-CPE-DD", "genie_pa": "Genie-PA (privileged)",
         "tap_phase_slope": "Tap-phase slope", "lmmse_2d": "2D-LMMSE", "waveform_dd": "Waveform-DD"}
CLASSICAL = {1: ("nominal_bussgang_lmmse", "nominal_idd_pa", "pilot_cpe_dd", "genie_pa"),
             2: ("tap_phase_slope", "lmmse_2d", "waveform_dd"), 3: ("lmmse_2d", "waveform_dd")}
PAIRS = {1: 10, 2: 5, 3: 5}
SNR = {1: 12, 2: 9, 3: 7}
PRIMARY = {1: ("pure", "pilot_cpe_dd"), 2: ("parity", "waveform_dd"), 3: ("parity", "waveform_dd")}
ABLATIONS = [("pure", "local"), ("parity", "local_parity"), ("pure", "no_gauge"),
             ("parity", "pure"), ("pure", "physics"), ("parity", "physics_parity")]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def assess_pair(record, regime, pair, methods):
    require((record["regime"], record["pair"], record["snr_db"]) == (regime, pair, SNR[regime]),
            "Pair header mismatch")
    require(set(record["rows"]) == set(methods), f"Receiver inventory mismatch {record['rows'].keys()}")
    n = 80 * (18 if regime == 2 else 4)
    archive = load(ARCHIVE / f"r{regime}/pair{pair:02d}/record.json")["rows"]
    for name, row in record["rows"].items():
        require(all(type(row[k]) is int for k in COUNT_FIELDS), "Noninteger count")
        require((row["blocks"], row["bits"], row["info_bits"]) == (n, 128*n, 64*n), "Wrong denominator")
        require(0 <= row["block_errors"] <= n and row["block_errors"] <= row["info_errors"] <=
                row["block_errors"] * 64 and 0 <= row["bit_errors"] <= 128*n, "Impossible counts")
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in ENERGY_FIELDS), "Invalid energy")
        if name in (*CLASSICAL[regime], "p5_full"):
            reference = archive["full" if name == "p5_full" else name]
            require(all(reference[k] == row[k] for k in COUNT_FIELDS), "Archived counts disagree")


def aggregate():
    raw, pooled, learning, probe = [], [], [], []
    families = {"primary": [], "ablation": [], "kernel": []}
    p5_probe = load(ARCHIVE / "summary.json")["kernel"]
    for regime, size in PAIRS.items():
        timing = {(r["method"], r["batch"]): r for r in load(OUT / f"latency_r{regime}.json")}
        methods = (*ARMS, "physics", "physics_parity", "p5_full", *CLASSICAL[regime])
        records = []
        for pair in range(1, size + 1):
            rec = load(OUT / f"r{regime}/pair{pair:02d}/record.json")
            assess_pair(rec, regime, pair, methods)
            records.append(rec)
            for method in methods:
                row = rec["rows"][method]
                raw.append(dict(regime=regime, pair=pair, snr_db=SNR[regime], method=method,
                                **row, **_rates(row)))
            if regime == 3:
                before = p5_probe[pair-1]
                require(before["pair"] == pair, "P5 probe pair mismatch")
                for method in ("pure", "parity"):
                    result = kernel_row(pair, rec["probe"][method])
                    probe.append(dict(method=method, p5_trained=before["trained"], **result))
        for method in methods:
            data = {k: sum(rec["rows"][method][k] for rec in records) for k in (*COUNT_FIELDS, *ENERGY_FIELDS)}
            one, many = (timing[method, b] for b in (1, 64))
            require(one["params"] == many["params"], "Parameter count differs by batch")
            pooled.append(dict(regime=regime, method=method, pairs=size, snr_db=SNR[regime],
                               **data, **_rates(data), params=one["params"], b1_ms=one["ms"],
                               b64_ms=many["ms"]))
        for method in ARMS:
            traces = [r["trace"][method] for r in records]
            require(all([v["step"] for v in curve] == [0,16,32,48,64] for curve in traces),
                    "Incomplete training trace")
            means = [fmean(point["validation_bce"] for point in pts) for pts in zip(*traces)]
            learning.append(dict(regime=regime, method=method, steps=[0,16,32,48,64],
                                 validation_bce=means, final_lower=sum(t[-1]["validation_bce"] <
                                 t[0]["validation_bce"] for t in traces), pairs=size))
        for family, contrasts in (("primary", [PRIMARY[regime]]), ("ablation", ABLATIONS)):
            for trial, control in contrasts:
                differences = [r["rows"][control]["block_errors"]-r["rows"][trial]["block_errors"]
                               for r in records]
                denom = sum(r["rows"][trial]["blocks"] for r in records)
                families[family].append(dict(arm=trial, **comparison(regime, control, differences,
                                                                     100*sum(differences)/denom)))
    for control in ("initial", "trained_shuffled", "p5_trained"):
        diffs = [r["trained"]-r[control] for r in probe if r["method"] == "pure"]
        families["kernel"].append(dict(arm="trained", **comparison(3, control, diffs, fmean(diffs))))
    for family in families.values():
        holm(family)
    require([len(f) for f in families.values()] == [3, 18, 3], "Incomplete comparison family")
    summary = dict(aggregates=pooled, comparisons=families, validation=learning, kernel=probe)
    save(OUT / "summary.json", summary)
    save(OUT / "r3_kernel.json", dict(scores=probe, comparisons=families["kernel"],
                                     raw_layers="r3/pairNN/record.json under probe"))
    with (OUT / "raw_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    print(f"{len(raw)} seed/method rows, {len(pooled)} pooled rows, 24 paired tests")


if __name__ == "__main__":
    aggregate()
