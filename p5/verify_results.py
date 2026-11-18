"""Reconcile the P5 frozen protocol, per-seed records, checkpoints and published totals."""

import argparse
import csv
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import torch

from p5.study import rng
from p5.unrolled_em_transformer import UnrolledEMTransformer
from p5.wavefield_sim import _rng as simulator_rng


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = {1: 10, 2: 5, 3: 5}
FITTED = ("full", "local", "no_unroll", "no_gauge")
CONTROLS = ("physics", "local", "no_unroll", "no_gauge")
REFERENCE = {1: "nominal_idd_pa", 2: "waveform_dd", 3: "waveform_dd"}
CLASSICAL = {
    1: ("nominal_bussgang_lmmse", "nominal_idd_pa", "pilot_cpe_dd", "genie_pa"),
    2: ("tap_phase_slope", "lmmse_2d", "waveform_dd"),
    3: ("lmmse_2d", "waveform_dd"),
}
WORDS = {1: 4, 2: 18, 3: 4}
COUNT_KEYS = ("bits", "bit_errors", "info_bits", "info_errors", "blocks", "block_errors")
ENERGY_KEYS = ("channel_error", "channel_energy")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(x, y):
    return math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-14)


def seeds_used():
    outer, simulation, internal = set(), set(), set()

    def check(regime, stream, pair=0, index=0):
        expected = regime * 10**8 + stream * 10**6 + pair * 1000 + index
        actual = rng(regime, stream, pair, index)
        require(actual == expected and actual not in outer, f"seed collision or drift: {actual}")
        outer.add(actual)
        if stream not in (4, 9):
            simulation.add(actual)
            for substream in range(1, 6):
                native = simulator_rng(actual, substream).initial_seed()
                require(native not in internal, f"simulator stream collision: {native}")
                internal.add(native)

    for regime, pairs in SEEDS.items():
        # SNR sweeps deliberately reuse the same frames across SNRs.
        for stream, count in ((1, 3), (2, 8), (3, 12)):
            for index in range(count):
                check(regime, stream, index=index)
        for pair in range(1, pairs + 1):
            check(regime, 4, pair)
            check(regime, 6, pair)
            for step in range(192):
                check(regime, 5, pair, step)
            for batch in range(10):
                check(regime, 7, pair, batch)
            if regime == 3:
                check(regime, 8, pair)
                check(regime, 9, pair)
        for size in (1, 64):
            check(regime, 10, index=size)
    require(outer.isdisjoint(internal), "outer and derived seeds overlap")
    return {"unique_outer_seeds": len(outer), "unique_simulator_frame_seeds": len(simulation),
            "unique_frame_substream_seeds": len(internal)}


def count_check(row, regime, bursts, context):
    for name in COUNT_KEYS:
        require(type(row[name]) is int and row[name] >= 0, f"{context}: invalid {name}")
    words = bursts * WORDS[regime]
    require((row["blocks"], row["info_bits"], row["bits"]) == (words, words * 64, words * 128),
            f"{context}: wrong LDPC denominators")
    require(row["bit_errors"] <= row["bits"] and row["info_errors"] <= row["info_bits"] and
            row["block_errors"] <= row["blocks"], f"{context}: errors exceed trials")
    require(row["block_errors"] <= row["info_errors"] <= 64 * row["block_errors"],
            f"{context}: block/bit errors disagree")
    require(all(math.isfinite(row[k]) for k in ENERGY_KEYS) and row["channel_error"] >= 0 and
            row["channel_energy"] > 0, f"{context}: invalid channel energy")


def calibration():
    frozen_files = ("PROTOCOL.md", "study.py", "classical.py", "wavefield_sim.py",
                    "unrolled_em_transformer.py")
    digests = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in frozen_files}
    checked = {}
    for regime in SEEDS:
        c = load(RESULTS / f"calibration_r{regime}.json")
        require(c["regime"] == regime and c["reference"] == REFERENCE[regime] and
                c["sources"] == digests, f"R{regime}: calibration identity")
        def nearest(rows):
            return min(rows, key=lambda row: (abs(Fraction(row["block_errors"], row["blocks"])
                                                    - Fraction(1, 5)), row["snr_db"]))["snr_db"]
        for rows, bursts in ((c["coarse"], 24), (c["fine"], 64), ([c["validation"]], 96)):
            for row in rows:
                require(row["bursts"] == bursts, f"R{regime}: incorrect calibration sample")
                count_check(row, regime, bursts, f"R{regime} calibration")
                require(close(row["bler"], row["block_errors"] / row["blocks"]),
                        f"R{regime}: calibration BLER mismatch")
        center = nearest(c["coarse"])
        selected = nearest(c["fine"])
        valid = c["validation"]
        require([v["snr_db"] for v in c["coarse"]] == list(range(-8, 33, 4)) and
                [v["snr_db"] for v in c["fine"]] == list(range(center - 3, center + 4)) and
                c["snr_db"] == valid["snr_db"] == selected, f"R{regime}: SNR selection mismatch")
        require(.05 <= valid["bler"] <= .35 and c["passed"] is True, f"R{regime}: calibration gate")
        checked[regime] = {"snr_db": selected, "errors": valid["block_errors"],
                           "blocks": valid["blocks"], "bler": valid["bler"]}
    return checked, digests


def records_and_states(cal):
    records = {}
    expected = {RESULTS / f"r{r}" / f"pair{p:02d}" / "record.json"
                for r, n in SEEDS.items() for p in range(1, n + 1)}
    require(set(RESULTS.glob("r*/pair*/record.json")) == expected,
            "missing or unexpected seed records")
    for regime, pairs in SEEDS.items():
        for pair in range(1, pairs + 1):
            folder = RESULTS / f"r{regime}" / f"pair{pair:02d}"
            record = load(folder / "record.json")
            label = f"R{regime} pair {pair:02d}"
            require(record["regime"] == regime and record["pair"] == pair and
                    record["snr_db"] == cal[regime]["snr_db"], f"{label}: identity")
            require(set(record["rows"]) == {*FITTED, "physics", *CLASSICAL[regime]},
                    f"{label}: missing or unexpected receiver")
            for method, row in record["rows"].items():
                count_check(row, regime, 80, f"{label} {method}")
            energy = {row["channel_energy"] for row in record["rows"].values()}
            require(len(energy) == 1, f"{label}: receiver frame energy differs")
            require(set(record["trace"]) == set(FITTED), f"{label}: incomplete trace")
            require({p.stem for p in folder.glob("*.pt")} == set(FITTED), f"{label}: checkpoint set")
            for name in FITTED:
                trace = record["trace"][name]
                require([x["step"] for x in trace] == [0, 48, 96, 144, 192] and
                        all(math.isfinite(x["validation_bce"]) for x in trace) and
                        trace[0]["train_mean_bce"] is None and
                        all(math.isfinite(x["train_mean_bce"]) for x in trace[1:]), f"{label}: {name} trace")
                torch.manual_seed(rng(regime, 4, pair))
                model = UnrolledEMTransformer(variant=name)
                state = torch.load(folder / f"{name}.pt", weights_only=True, map_location="cpu")
                updates = []
                for key, param in model.named_parameters():
                    require(torch.isfinite(state[key]).all().item(), f"{label}: {name}/{key} nonfinite")
                    changed = not torch.equal(param, state[key])
                    if param.requires_grad:
                        updates.append(changed)
                    else:
                        require(not changed, f"{label}: changed frozen reference {name}/{key}")
                require(any(updates), f"{label}: no fitted weight changed in {name}")
                model.load_state_dict(state, strict=True)
            records[regime, pair] = record
    return records


def sign_probability(differences):
    """Enumerate subsets to negate using exact rationals, preserving multiplicities."""
    unsigned = list(map(Fraction, differences))
    original = sum(unsigned)
    partials = [Fraction(0)]
    for item in unsigned:
        partials.extend([v + item for v in partials])
    exceed = sum(abs(original - 2 * v) >= abs(original) for v in partials)
    return Fraction(exceed, len(partials))


def holm(probabilities):
    ordered = sorted(probabilities, key=probabilities.get)
    adjusted = {}
    previous = Fraction(0)
    for position, item in enumerate(ordered):
        previous = max(previous, min(Fraction(1), (len(ordered)-position) * probabilities[item]))
        adjusted[item] = previous
    return adjusted


def kernel_means(record, stage, shuffled=False):
    layers = record["probe"][stage]
    require(len(layers) == 3, "unexpected number of R3 probe layers")
    all_values = []
    for layer in layers:
        matrix = layer["shuffled_corr"] if shuffled else [layer["corr"]]
        require(len(matrix) == (8 if shuffled else 1) and all(len(v) == 8 for v in matrix),
                "R3 probe frame/permutation counts")
        flat = [v for row in matrix for v in row]
        require(all(math.isfinite(x) and abs(x) <= 1.000000000001 for x in flat),
                "invalid kernel correlations")
        all_values.extend(flat)
    return math.fsum(all_values) / len(all_values)


def empirical_rates(row):
    return {"ber": row["bit_errors"] / row["bits"], "bler": row["block_errors"] / row["blocks"],
            "info_ber": row["info_errors"] / row["info_bits"],
            "nmse_db": 10 * math.log10(row["channel_error"] / row["channel_energy"])}


def reconcile_table(raw, pooled, records, cal):
    rows = {(int(row["regime"]), int(row["pair"]), row["method"]): row for row in raw}
    pooled_rows = {(row["regime"], row["method"]): row for row in pooled}
    require(len(rows) == len(raw) and len(pooled_rows) == len(pooled), "duplicate summary/CSV row")
    aggregate = {}
    for (r, p), rec in records.items():
        for arm, item in rec["rows"].items():
            csv_row = rows.pop((r, p, arm))
            require(int(csv_row["snr_db"]) == cal[r]["snr_db"], "raw CSV SNR")
            for field in COUNT_KEYS:
                require(int(csv_row[field]) == item[field], f"raw CSV count {r}/{p}/{arm}/{field}")
            for field, expected in {**{k:item[k] for k in ENERGY_KEYS}, **empirical_rates(item)}.items():
                require(close(float(csv_row[field]), expected), f"raw CSV measurement {r}/{p}/{arm}/{field}")
            total = aggregate.setdefault((r,arm), dict.fromkeys((*COUNT_KEYS,*ENERGY_KEYS),0))
            for key in total:
                total[key] += item[key]
    require(not rows and pooled_rows.keys() == aggregate.keys(), "unmatched table entries")
    for key, total in aggregate.items():
        r, arm = key
        row = pooled_rows[key]
        require(row["pairs"] == SEEDS[r] and row["snr_db"] == cal[r]["snr_db"], "aggregate metadata")
        for field in COUNT_KEYS:
            require(type(row[field]) is int and row[field] == total[field], f"summary count {r}/{arm}/{field}")
        for field, expected in {**{k:total[k] for k in ENERGY_KEYS}, **empirical_rates(total)}.items():
            require(close(row[field], expected), f"summary measurement {r}/{arm}/{field}")
    return aggregate, pooled_rows


def reconcile_statistics(summary, records):
    expected = {"primary": {}, "ablation": {}, "kernel": {}}
    for r, n in SEEDS.items():
        tests = [("primary", REFERENCE[r]), *(("ablation", arm) for arm in CONTROLS)]
        for family, arm in tests:
            deltas = [records[r,p]["rows"][arm]["block_errors"] - records[r,p]["rows"]["full"]["block_errors"]
                      for p in range(1,n+1)]
            denom = 80 * WORDS[r] * n
            expected[family][r,arm] = (deltas, 100 * sum(deltas) / denom)
    kernel_rows = {row["pair"]: row for row in summary["kernel"]}
    require(len(summary["kernel"]) == len(kernel_rows) == SEEDS[3], "kernel pair count")
    probe_values = []
    for p in range(1, SEEDS[3] + 1):
        probe = records[3,p]
        values = {}
        for stage in ("initial", "trained"):
            values[stage] = kernel_means(probe, stage)
            values[stage + "_shuffled"] = kernel_means(probe, stage, shuffled=True)
        require(all(close(kernel_rows[p][k], val) for k,val in values.items()), f"kernel summary pair {p}")
        probe_values.append(values)
    for arm in ("initial", "trained_shuffled"):
        deltas = [v["trained"] - v[arm] for v in probe_values]
        expected["kernel"][3, arm] = deltas, math.fsum(deltas) / len(deltas)
    comparisons = summary["comparisons"]
    require(set(comparisons) == set(expected), "incorrect hypothesis families")
    receipt = {}
    for family, tests in expected.items():
        rows = {(v["regime"],v["comparison"]): v for v in comparisons[family]}
        require(set(rows) == set(tests) and len(rows) == len(comparisons[family]), f"{family}: test identities")
        raw = {}
        for key, (deltas, effect) in tests.items():
            row = rows[key]
            require(len(row["differences"]) == len(deltas) and
                    all(close(x,y) for x,y in zip(deltas,row["differences"])) and close(effect,row["effect"]),
                    f"{family}/{key}: paired differences or pooled effect")
            if family != "kernel":
                require(all(type(v) is int for v in row["differences"]), "paired BLER differences must be integers")
            counts = [sum(x>0 for x in deltas), sum(x==0 for x in deltas), sum(x<0 for x in deltas)]
            require(counts == [row["wins"],row["ties"],row["losses"]], f"{family}/{key}: signs")
            size = 2**len(deltas)
            raw[key] = sign_probability(deltas)
            require(raw[key] == row["p_raw"] and size == row["signflip_total"] and
                    raw[key]*size == row["signflip_tail"], f"{family}/{key}: randomization tail")
        adjusted = holm(raw)
        for key in tests:
            require(adjusted[key] == rows[key]["p_holm"], f"{family}/{key}: Holm correction")
        receipt[family] = len(tests)
    return receipt


def reconcile_timings(summary, pooled, records):
    timing = {row["regime"]: row for row in summary["runtime"]}
    require(set(timing) == set(SEEDS) and len(timing) == len(summary["runtime"]), "runtime group count")
    groups = 0
    dofs = {}
    for arm in FITTED:
        model = UnrolledEMTransformer(variant=arm)
        dofs[arm] = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
    require(len(set(dofs.values())) == 1, "model controls are not parameter-matched")
    for r, n in SEEDS.items():
        spans = [records[r,p] for p in range(1,n+1)]
        durations = timing[r]
        require(close(load(RESULTS / f"calibration_r{r}.json")["seconds"],durations["calibration_seconds"]) and
                close(sum(v["train_seconds"] for v in spans), durations["sum_training_seconds"]) and
                close(sum(v["total_seconds"] for v in spans), durations["sum_pair_wall_seconds"]),
                f"R{r}: runtime totals")
        data = load(RESULTS / f"latency_r{r}.json")
        expected_set = {(arm,batch) for (regime,arm) in pooled if r == regime for batch in (1,64)}
        actual_set = {(v["method"],v["batch"]) for v in data}
        require(expected_set == actual_set and len(data) == len(actual_set), f"R{r}: latency group count")
        for v in data:
            samples = sorted(v["samples_ms"])
            require(len(samples) == 11 and all(math.isfinite(t) and t>0 for t in samples),
                    f"R{r}: invalid latency samples")
            arm, batch = v["method"],v["batch"]
            summary_row = pooled[r,arm]
            require(close(v["q1_ms"], (samples[2]+samples[3])/2) and close(v["q3_ms"],(samples[7]+samples[8])/2)
                    and v["ms"] == samples[5] == summary_row[f"latency_b{batch}_ms"]
                    and v["params"] == dofs.get(arm,0) == summary_row["params"], f"R{r}: {arm} timing/size")
            groups += 1
    return groups


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preflight", action="store_true", help="Check seed spaces and the three calibration records only")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = seeds_used()
    cal, hashes = calibration()
    if args.preflight:
        print(json.dumps({"calibration": cal, "rng": seeds}, indent=2))
        return
    records = records_and_states(cal)
    with (RESULTS / "raw_by_seed.csv").open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    summary = load(RESULTS / "summary.json")
    aggregate, pooled = reconcile_table(raw, summary["aggregates"], records, cal)
    by_regime = {v["regime"]: v for v in summary["calibration"]}
    require(set(by_regime) == set(SEEDS) and len(by_regime) == len(summary["calibration"]), "calibration groups")
    for r, observed in by_regime.items():
        require(observed["reference"] == REFERENCE[r] and
                observed["snr_db"] == cal[r]["snr_db"] and observed["block_errors"] == cal[r]["errors"] and
                observed["blocks"] == cal[r]["blocks"] and close(observed["bler"],cal[r]["bler"]), "calibration summary")
    tests = reconcile_statistics(summary, records)
    timings = reconcile_timings(summary, pooled, records)
    waterfall = {}
    for r, arm in REFERENCE.items():
        row = aggregate[r,arm]
        rate = row["block_errors"] / row["blocks"]
        waterfall[r] = {"reference": arm, "errors": row["block_errors"], "blocks": row["blocks"],
                        "bler": rate, "in_5_to_35_percent": .05 <= rate <= .35}
    receipt = {"verification": "passed", "records": len(records), "strict_checkpoint_loads": 4 * len(records),
               "training_traces": 4 * len(records), "raw_csv_rows": len(raw), "aggregate_rows": len(aggregate),
               "paired_tests_by_family": tests, "latency_groups": timings, "validated_seed_partition": seeds,
               "calibration": cal, "held_out_reference_waterfall": waterfall, "verified_source_sha256": hashes}
    target = RESULTS / "study_verification.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",encoding="utf-8")
    os.replace(tmp,target)
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
