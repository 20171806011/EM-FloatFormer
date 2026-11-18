"""Verify Iteration-1 receipts without importing the simulator or running a model."""

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAIRS = (901, 902, 903)
REGIMES = (1, 2, 3)
SNR = {1: 12., 2: 9., 3: 7.}
ARMS = ("pure", "local", "parity", "local_parity", "no_gauge")
PARITY = ("parity", "local_parity")
DESIGNS = ("references", "main", "depth3", "lr005", "warmcos", "aux", "unclipped")
REFERENCES = {1: ("physics", "physics_parity", "nominal_bussgang_lmmse", "nominal_idd_pa",
                  "pilot_cpe_dd", "genie_pa"),
              2: ("physics", "physics_parity", "tap_phase_slope", "lmmse_2d", "waveform_dd"),
              3: ("physics", "physics_parity", "lmmse_2d", "waveform_dd")}
COUNTS = ("blocks", "block_errors", "bits", "bit_errors", "info_bits", "info_errors",
          "channel_error", "channel_energy", "bce_sum")


def check(ok, context):
    if not ok:
        raise ValueError(context)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite number {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def close(a, b):
    if isinstance(b, int):
        return type(a) is int and a == b
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12)


def seed(r, stream, pair, batch=0):
    return r * 100_000_000 + stream * 1_000_000 + pair * 1000 + batch


def ids():
    files = ["experiment.py", "data.py", "axial_operator.py", "axial_em_transformer.py", "parity.py"]
    out = {}
    for key, names in (("references", files[:2]), ("main", files)):
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode())
            h.update((ROOT / name).read_bytes())
        out[key] = h.hexdigest()
    return out


def design_jobs():
    for d in DESIGNS:
        for r in REGIMES:
            pairs = PAIRS if d in ("main", "depth3", "references") else PAIRS[:1]
            arms = ("references",) if d == "references" else (ARMS if d == "main" else PARITY)
            for p in pairs:
                for a in arms:
                    yield (r, p, d, a), ROOT / "results" / "jobs" / f"r{r}" / str(p) / d / f"{a}.json"


def score_check(row, regime, pairs=1):
    blocks = (576 if regime == 2 else 128) * pairs
    for denom, n in (("blocks", blocks), ("bits", blocks*128), ("info_bits", blocks*64)):
        check(close(row[denom], n), f"R{regime} incorrect {denom}")
    for err, n in (("block_errors", "blocks"), ("bit_errors", "bits"), ("info_errors", "info_bits")):
        check(type(row[err]) is int and 0 <= row[err] <= row[n], f"Invalid {err}")
    check(row["block_errors"] <= row["info_errors"] <= row["block_errors"] * 64, "LDPC counts disagree")
    check(all(math.isfinite(row[k]) and row[k] >= 0 for k in COUNTS) and
          row["channel_energy"] > 0 and row["channel_error"] > 0, "Invalid score sum")


def derived(rows):
    total = {k: sum(x[k] for x in rows) for k in COUNTS}
    return dict(**total, bler=total["block_errors"]/total["blocks"],
                pre_ber=total["bit_errors"]/total["bits"], bce=total["bce_sum"]/total["bits"],
                nmse_db=10*math.log10(total["channel_error"]/total["channel_energy"]))


def compare_scores(observed, expected, context):
    for k, v in expected.items():
        check(close(observed[k], v), f"{context}: inconsistent {k}")


def verify_fit(rec, r, p, d, a, report):
    config = dict(depth=3 if d == "depth3" else 2, lr=.005 if d == "lr005" else .003,
                  clip=None if d == "unclipped" else 1., aux_nmse=.05 if d == "aux" else 0.,
                  schedule="warmcos" if d == "warmcos" else "constant")
    budgets = (64, 96, 128) if d == "main" and a in PARITY else (64,)
    steps = max(budgets)
    check(rec["config"] == config and set(rec["rows"]) == {str(x) for x in budgets}, "Design mismatch")
    check(rec["train_seed_first"] == seed(r, 5, p) and rec["train_seed_last"] == seed(r, 5, p, steps-1)
          and rec["init_seed"] == seed(r, 5, p, 900), "Training seed mismatch")
    check(rec["validation_curve_seed"] == seed(r, 6, p) and rec["train_batch"] == 4 and
          rec["validation_curve_batch_size"] == 8 and rec["scoring_batch_size"] == 32, "Batch mismatch")
    check(rec["parameters"] == report["trainable_real_parameters"] and rec["parameters"] > 0,
          "Unmatched real parameter count")
    trace = rec["trace"]
    check([x["step"] for x in trace] == list(range(0, steps+1, 16)), "Incomplete learning curve")
    check(trace[0]["cumulative_fit_seconds"] == 0 and
          close(trace[-1]["cumulative_fit_seconds"], rec["fit_seconds"]) and
          0 < rec["fit_seconds"] <= rec["wall_seconds"], "Invalid fit time")
    for i, row in enumerate(trace):
        check(row["validation_bce"] >= 0 and row["lr"] > 0, "Invalid validation value")
        if i:
            check(row["cumulative_fit_seconds"] > trace[i-1]["cumulative_fit_seconds"] and
                  0 <= row["mean_preclip_norm"] <= row["max_preclip_norm"] and row["train_mean_bce"] >= 0,
                  "Invalid training trace")
    for b in budgets:
        individual = report["by_pair"][f"r{r}/{d}/{a}/{b}"][str(p)]
        compare_scores(individual["scores"], derived([rec["rows"][str(b)]]), "Pair metrics")
        time = next(x["cumulative_fit_seconds"] for x in trace if x["step"] == b)
        check(close(individual["fit_seconds"], time), "Pair time mismatch")


def verify_probes(records, aggregate):
    check(set(aggregate) == {"pure", "parity"}, "Unexpected R3 probe arms")
    for (r, p, d, a), rec in records.items():
        if r != 3 or d != "main" or a not in aggregate:
            continue
        for stage in ("initial", "trained"):
            layers = rec["probe"][stage]
            check(len(layers) == 2, "Missing probe layer")
            for layer in layers:
                coords = {tuple(x) for x in layer["coords"]}
                check(len(coords) == 48 and len(layer["coords"]) == 48 and
                      all(0 <= i < 4 and 0 <= t < 5 and 0 <= k < 32 for i,t,k in coords),
                      "Invalid R3 probe coordinates")
                check(layer["off_diagonal_entries"] == 48*47, "Wrong Green probe mask")
            for axis in ("joint", "spatial", "time", "frequency"):
                for field in ("overall", "shuffled") if axis == "joint" else ("overall","shuffled","axis_only"):
                    values = []
                    for layer in layers:
                        vals = layer["maps"][axis][field]
                        check(len(vals) == 8, "Missing probe burst")
                        values.extend(v for v in vals if v is not None)
                    check(all(-1.000000000001 <= v <= 1.000000000001 for v in values), "Invalid Pearson")
                    avg = mean(values) if values else None
                    dest = aggregate[a][stage][axis][field]
                    obs = dest["by_pair"][str(p)]
                    check((obs is None and avg is None) or (avg is not None and close(obs, avg)),
                          "Per-pair Green probe mismatch")
                    vals = [v for v in dest["by_pair"].values() if v is not None]
                    check(set(dest["by_pair"]) == {str(k) for k in PAIRS} and
                          (close(dest["mean"], mean(vals)) if vals else dest["mean"] is None),
                          "Pooled Green probe mismatch")


def verify_ancillary(signatures, records):
    info = {}
    inv = read(RESULTS / "invariants.json")
    rows = inv["rows"]
    check(inv["passed"] and inv["matched_capacity"] and len(rows) == 15 and
          {(v["regime"], v["arm"]) for v in rows} == {(r,a) for r in REGIMES for a in ARMS},
          "Missing invariants")
    for v in rows:
        check(v["passed"] and v["gauge"]["passed"] and v["pair"] in PAIRS
              and v["active_real_parameters"] == v["sizes"]["trainable_real"]
              and {a["steps"] for a in v["anchors"]} == {2,3}
              and all(a["passed"] and a["p6_bit_exact"] for a in v["anchors"])
              and all(x["passed"] and x["real_dofs"] == x["real_gradient_entries_nonzero"]
                      for x in v["gradient"]["tensors"]), "Invariant receipt mismatch")
    info["invariant_arm_regimes"] = len(rows)
    legacy = read(RESULTS / "legacy_dev.json")
    for key, n in (("p5",15), ("p5b",13)):
        check(legacy["suites"][key]["passed"] and legacy["suites"][key]["run"] == n, "Legacy suite mismatch")
    check(legacy["passed"] and all(r in REGIMES and s in (5,6) and p in PAIRS and 0 <= i < 1000
                                  for r,s,p,i in legacy["unique_frame_draws"]), "Legacy split mismatch")
    info["dev_redirected_legacy_tests"] = 28
    for name in ("extended_once_updated", "extended_trained"):
        path = RESULTS / f"{name}.json"
        if not path.exists():
            continue
        d = read(path)
        s, rows = d["summary"], d["rows"]
        check(len(rows) == s["checks"] and sum(v["passed"] for v in rows) == s["passed_checks"] and
              all(v["seed"] == seed(v["regime"],6,v["pair"]) and v["pair"] in PAIRS for v in rows)
              and close(s["max_grid_llr_diff"], max(v["max_grid_llr_diff"] for v in rows))
              and s["coded_flips"] == sum(v["coded_flips"] for v in rows), "Extended audit mismatch")
        info[name] = {k:s[k] for k in ("passed","checks","passed_checks","coded_flips","max_grid_llr_diff")}
    path = RESULTS / "handle_ablation.json"
    if path.exists():
        d = read(path)
        check(d["pair"] == 901 and d["stream"] == 6 and d["implementation"] == signatures["main"]
              and d["frames_per_regime"] == 32 and d["steps"] == 64, "Handle experiment mismatch")
        for r in REGIMES:
            for a in PARITY:
                v = d["arms"][str(r)][a]
                check(v["full_replay_passed"] and v["all_removed_fixed_check_passed"]
                      and v["score_seeds"] == [seed(r,6,901)], "Handle replay mismatch")
                compare_scores(v["rows"]["full"], records[r,901,"main",a]["rows"]["64"], "Handle/full")
                compare_scores(v["rows"]["all"], records[r,901,"references","references"]["rows"]["physics_parity"],
                               "Handle/all")
                for key, deltas in v["removed_minus_full"].items():
                    for metric, val in deltas.items():
                        check(close(val, v["rows"][key][metric] - v["rows"]["full"][metric]), "Handle delta mismatch")
        info["matched_handle_receipts"] = 6
    return info


def verify():
    report = read(RESULTS / "dev_iter1.json")
    jobs = dict(design_jobs())
    check(len(jobs) == 96 and len(report["job_records"]) == 96 and
          {p.relative_to(ROOT).as_posix() for p in jobs.values()} == set(report["job_records"]),
          "Incorrect job manifest")
    check(set((RESULTS/"jobs").glob("r*/*/*/*.json")) == set(jobs.values()), "Unexpected job directory contents")
    data = report["data"]
    check(data["pairs"] == list(PAIRS) and [data[k] for k in
          ("train_stream","validation_stream","kernel_frames_stream","kernel_shuffle_seed_stream",
           "frames_per_validation_pair","scoring_batch_size","scoring_batches_per_pair",
           "validation_curve_frames","probe_frames_per_pair","shuffled_permutations_per_layer")] ==
          [5,6,8,9,32,32,1,8,8,8], "Final split metadata mismatch")
    signatures = ids()
    check(report["implementations"] == signatures, "Final model/reference source mismatch")
    records, energies = {}, {}
    groups, curves = defaultdict(list), defaultdict(list)
    for job, path in jobs.items():
        r,p,d,a = job
        rec = records[job] = read(path)
        check(tuple(rec[k] for k in ("regime","pair","design","arm")) == job and rec["snr_db"] == SNR[r]
              and rec["implementation"] == signatures["references" if d == "references" else "main"]
              and rec["score_seeds"] == [seed(r,6,p)], f"Receipt identity/seed mismatch {job}")
        if d == "references":
            check(set(rec["rows"]) == set(REFERENCES[r]), "Missing classical/fixed scores")
        else:
            verify_fit(rec, r, p, d, a, report)
            curves[f"r{r}/{d}/{a}"].append(rec["trace"])
        for key, row in rec["rows"].items():
            score_check(row, r)
            energy = energies.setdefault((r,p), row["channel_energy"])
            check(close(energy, row["channel_energy"]), f"Different target energy {job}")
            label = key if d == "references" else a if d == "main" and key == "64" else f"{d}/{a}/{key}"
            section = "pooled_main" if d == "references" or (d == "main" and key == "64") else "ablations"
            groups[section,r,label].append(row)
            if d == "main" and a in PARITY and key == "64" and p == 901:
                groups["ablations",r,f"main_901_only/{a}/64"].append(row)
    for (section,r,label), raw in groups.items():
        expected = derived(raw)
        score_check(expected, r, len(raw))
        compare_scores(report[section][str(r)][label], expected, f"{section}/{r}/{label}")
    for section in ("pooled_main", "ablations"):
        for r in REGIMES:
            check(set(report[section][str(r)]) == {k for s,r0,k in groups if (s,r0) == (section,r)},
                  "Missing or extra pooled arm")
    check(set(curves) == set(report["curves"]), "Learning curve keys differ")
    for name, raw in curves.items():
        for target, *src in zip(report["curves"][name], *raw, strict=True):
            check(all(x["step"] == target["step"] for x in src), "Learning curve steps disagree")
            for k, v in target.items():
                if k == "step":
                    continue
                vals = [x[k] for x in src if x[k] is not None]
                check(close(v, mean(vals)) if vals else v is None, "Pooled learning curve mismatch")
    provenance = read(RESULTS / "dev_batch32_check.json")
    for r in REGIMES:
        main = report["pooled_main"][str(r)]
        extra = report["ablations"][str(r)]
        for a in PARITY:
            scores = {64: main[a], 96: extra[f"main/{a}/96"], 128: extra[f"main/{a}/128"]}
            for n, row in scores.items():
                times = [report["by_pair"][f"r{r}/main/{a}/{n}"][str(p)]["fit_seconds"] for p in PAIRS]
                check(close(row["mean_fit_seconds_per_pair"], mean(times)), "Pooled fit time mismatch")
            for metric in ("bler", "bce"):
                b = min(scores, key=lambda n: (scores[n][metric], n))
                decision = report["dev_budget_minima"][str(r)][a][metric]
                check(decision["step"] == b and close(decision["value"], scores[b][metric]),
                      "Development checkpoint ranking mismatch")
        fixed = main["physics_parity"]
        compare_scores(fixed, provenance["results"][str(r)]["pooled"], "Baseline B32 check")
        check(fixed["block_errors"] == {1:32, 2:439, 3:66}[r], "Submitted baseline counts disagree")
    verify_probes(records, report["r3_green_probe"])
    depth_fixed = read(RESULTS / "depth3_fixed.json")
    check(depth_fixed == report["depth3_fixed_reference"] and depth_fixed["steps"] == 3
          and depth_fixed["batch_size"] == 32, "Depth-three reference metadata mismatch")
    for r in REGIMES:
        cells = depth_fixed["by_pair"][str(r)]
        for p in PAIRS:
            check(cells[str(p)]["seed"] == seed(r,6,p), "Depth-three reference seed mismatch")
        for arm in ("physics", "physics_parity"):
            values = [cells[str(p)]["rows"][arm] for p in PAIRS]
            for p, value in zip(PAIRS, values):
                score_check(value, r)
                check(close(value["channel_energy"], energies[r,p]), "Depth-three reference frame mismatch")
            compare_scores(depth_fixed["pooled"][str(r)][arm], derived(values), "Depth-three fixed pool")
    latency_rows = 0
    for r in REGIMES:
        d = read(RESULTS/"jobs"/f"latency_r{r}.json")
        check(d["implementation"] == signatures["main"] and d["pair"] == 901 and d["regime"] == r
              and d["rows"] == report["latency_ms"][str(r)] and set(d["rows"]) == {"1","64"}, "Timing mismatch")
        for batch in d["rows"].values():
            check(set(batch) == {*REFERENCES[r], *(f"main/{a}" for a in ARMS),
                                 *(f"depth3/{a}" for a in PARITY)}, "Missing latency arm")
            for v in batch.values():
                vals = v["samples_ms"]
                check(len(vals) == 5 and min(vals) > 0 and close(median(vals), v["median_ms"]), "Latency median")
                latency_rows += 1
    return dict(passed=True, design_receipts=len(records), model_fits=len(records)-9, reference_jobs=9,
                scored_rows=sum(len(rec["rows"]) for rec in records.values()), pooled_score_rows=len(groups),
                scored_development_cells=len(energies), learning_curves=len(curves), latency_rows=latency_rows,
                probe_arm_pairs=6, implementations=signatures, ancillary=verify_ancillary(signatures, records),
                scope="Recomputed from development receipts. This verification did not run any model.")


if __name__ == "__main__":
    try:
        outcome = verify()
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        outcome = dict(passed=False, error=str(error))
    dest = RESULTS / "verification.json"
    temp = dest.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(dest)
    print(json.dumps(outcome))
    raise SystemExit(0 if outcome["passed"] else 1)
