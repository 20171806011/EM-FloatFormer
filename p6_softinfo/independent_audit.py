"""Independent-in-this-environment review of counts, profiling and inference.

The audit uses the same CPU/runtime and existing checkpoints; it does not train
models or establish external reproduction. Its statistic calculation uses an
integer sum-distribution recurrence instead of the main sign-enumeration helper.
Pass --output to retain a new receipt when repeating an already published audit.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import traceback

from p6 import experiment as ex
from p6_softinfo.common import (
    CONDITIONS, METHODS, PRIMARY_BASELINES, PRIMARY_CONDITIONS, RESULTS, ROOT,
    environment, inherited_changes, read, save, sha, verify_freeze,
)
from p6_softinfo.statistics import validate
from p6_softinfo.study import score_pair


COUNT_FIELDS = ("bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks")
ENERGY_FIELDS = ("channel_error", "channel_energy")
PROFILE_METHODS = ("profile", "variance_profile")
MOMENT_METHODS = ("variance", "variance_profile")
REPLAY_INDICES = (1, 10)
RELATIVE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    """Energy agreement: finite values, relative tolerance only, no absolute floor."""
    return (math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.))


def _profile_check(record):
    """Inspect every stored burst, not only means that could conceal failures."""
    totals = defaultdict(Counter)
    failures, strict_increases = [], []
    for row in record["rows"]:
        where = dict(index=record["index"], batch=row["batch_index"],
                     condition=row["condition"], method=row["method"])
        diagnostics = row.get("diagnostics")
        if row["method"] in MOMENT_METHODS:
            require(isinstance(diagnostics, dict), f"Missing moment diagnostics: {where}")
            moments = diagnostics.get("moments", [])
            require([item["step"] for item in moments] == [0, 1, 2],
                    f"Moment solve indices differ: {where}")
            for item in moments:
                variance = item["mean_variance"]
                trace = item["covariance_trace_mean"]
                require(math.isfinite(variance) and 0 <= variance <= 1.8 + 1e-12,
                        f"Invalid Gray 16-QAM conditional variance: {where}")
                require(math.isfinite(trace) and trace >= 0,
                        f"Invalid expected-Gram trace: {where}")
        if row["method"] not in PROFILE_METHODS:
            continue
        require(isinstance(diagnostics, dict), f"Missing profile diagnostics: {where}")
        profile = diagnostics.get("profile")
        require(isinstance(profile, dict), f"Missing profile record: {where}")
        keys = ("objective_before", "objective_after", "accepted_steps", "candidate_evaluations",
                "used_pinv", "invalid_curvature", "max_total_phase_change")
        require(all(isinstance(profile.get(key), list) and len(profile[key]) == 8 for key in keys),
                f"Profile telemetry must retain all B8 bursts: {where}")
        for burst in range(8):
            before, after = profile["objective_before"][burst], profile["objective_after"][burst]
            accepted, evaluations = profile["accepted_steps"][burst], profile["candidate_evaluations"][burst]
            pinv, invalid = profile["used_pinv"][burst], profile["invalid_curvature"][burst]
            phase = profile["max_total_phase_change"][burst]
            require(type(accepted) is int and 0 <= accepted <= 2,
                    f"Invalid accepted-step counter: {where}/{burst}")
            require(type(evaluations) is int and accepted <= evaluations <= 16,
                    f"Invalid line-search counter: {where}/{burst}")
            require(type(pinv) is bool and type(invalid) is bool,
                    f"Fallback flags must be Boolean: {where}/{burst}")
            require(math.isfinite(before) and math.isfinite(after) and before >= 0 and after >= 0,
                    f"Invalid conditional objective: {where}/{burst}")
            require(math.isfinite(phase) and 0 <= phase <= .5 + 1e-12,
                    f"Two capped steps exceed total phase bound: {where}/{burst}")
            counter = totals[row["condition"], row["method"]]
            counter["bursts"] += 1
            counter["accepted_steps"] += accepted
            counter["candidate_evaluations"] += evaluations
            counter["used_pinv"] += int(pinv)
            counter["invalid_curvature"] += int(invalid)
            counter[f"bursts_accepting_{accepted}_steps"] += 1
            if after > before:
                item = dict(where, burst=burst, before=before, after=after,
                            absolute_increase=after-before)
                strict_increases.append(item)
                # Record every strict increase, even if it meets the declared
                # relative numerical agreement tolerance.
                if not close(before, after):
                    failures.append(item)
    return dict(index=record["index"], conditional_objective_nonincrease_within_tolerance=not failures,
                strict_objective_increases=strict_increases, objective_failures=failures,
                totals=[dict(condition=condition, method=method, **counter)
                        for (condition, method), counter in sorted(totals.items())])


def independent_statistics(records, summary):
    """Recompute aggregate integers and six primary tests from raw score cells."""
    grouped = defaultdict(Counter)
    energies = defaultdict(lambda: defaultdict(list))
    cluster_errors = Counter()
    for record in records:
        for row in record["rows"]:
            key = row["condition"], row["method"]
            grouped[key].update({field: row["score"][field] for field in COUNT_FIELDS})
            for field in ENERGY_FIELDS:
                energies[key][field].append(row["score"][field])
            cluster_errors[record["index"], *key] += row["score"]["block_errors"]
    published = {(row["condition"], row["method"]): row for row in summary["aggregates"]}
    grid_exact = set(grouped) == set(published) == set(itertools.product(CONDITIONS, METHODS))
    integers_exact = grid_exact and all(
        published[key][field] == grouped[key][field] for key in grouped for field in COUNT_FIELDS)
    energies_close = grid_exact and all(
        close(published[key][field], math.fsum(energies[key][field]))
        for key in grouped for field in ENERGY_FIELDS)
    tests = []
    for condition, baseline in itertools.product(PRIMARY_CONDITIONS, PRIMARY_BASELINES):
        differences = [cluster_errors[index, condition, baseline] -
                       cluster_errors[index, condition, "variance_profile"] for index in range(1, 11)]
        distribution = Counter({0: 1})
        for difference in differences:
            changed = Counter()
            for partial, multiplicity in distribution.items():
                changed[partial+difference] += multiplicity
                changed[partial-difference] += multiplicity
            distribution = changed
        tail = sum(multiplicity for value, multiplicity in distribution.items()
                   if abs(value) >= abs(sum(differences)))
        total = sum(distribution.values())
        tests.append(dict(condition=condition, baseline=baseline, candidate="variance_profile",
                          seed_block_error_reductions=differences,
                          mean_bler_reduction_pp=100*sum(differences)/5760,
                          signflip_tail=tail, signflip_total=total, p_raw=tail/total,
                          wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
                          losses=sum(d < 0 for d in differences)))
    ordered = sorted(tests, key=lambda row: row["p_raw"])
    for rank, test in enumerate(ordered):
        test["p_holm"] = min(1., max((6-k)*ordered[k]["p_raw"] for k in range(rank+1)))
    primary = {(row["condition"], row["baseline"]): row for row in summary["primary_tests"]}
    matches = []
    for test in tests:
        key = test["condition"], test["baseline"]
        exact_fields = [field for field in test if field != "mean_bler_reduction_pp"]
        exact = key in primary and all(primary[key][field] == test[field] for field in exact_fields)
        effect_close = key in primary and close(primary[key]["mean_bler_reduction_pp"],
                                               test["mean_bler_reduction_pp"])
        matches.append(dict(condition=key[0], baseline=key[1], discrete_statistics_exact=exact,
                            mean_effect_agrees=effect_close))
    return dict(algorithm="independent Counter aggregation and integer sum-distribution recurrence",
                aggregate_rows=len(grouped), integer_fields_per_row=len(COUNT_FIELDS),
                aggregate_integer_counts_exact=integers_exact,
                aggregate_energy_agreement=energies_close, energy_relative_tolerance=RELATIVE_TOLERANCE,
                primary_tests=tests, primary_test_matches=matches,
                passed=integers_exact and energies_close and all(
                    row["discrete_statistics_exact"] and row["mean_effect_agrees"] for row in matches))


def replay(index):
    path = RESULTS / "fresh" / f"pair{index:02d}.json"
    original_digest = sha(path)
    original = read(path)
    repeated = score_pair(index, replay=True)
    key = lambda row: (row["batch_index"], row["condition"], row["method"])
    old = {key(row): row for row in original["rows"]}
    new = {key(row): row for row in repeated["rows"]}
    grid_exact = set(old) == set(new) and len(new) == 4*len(CONDITIONS)*len(METHODS)
    integers, energies, energy_outside = [], [], []
    for cell in sorted(old.keys() & new.keys()):
        for field in COUNT_FIELDS:
            a, b = old[cell]["score"][field], new[cell]["score"][field]
            if a != b:
                integers.append(dict(cell=list(cell), field=field, stored=a, replayed=b))
        for field in ENERGY_FIELDS:
            a, b = old[cell]["score"][field], new[cell]["score"][field]
            if a != b:
                item = dict(cell=list(cell), field=field, stored=a, replayed=b,
                            absolute_difference=abs(a-b))
                energies.append(item)
                if not close(a, b):
                    energy_outside.append(item)
    frames_exact = original["frames"] == repeated["frames"]
    identity_exact = all(original[field] == repeated[field] for field in
                         ("index", "checkpoint_pair", "freeze_sha256", "checkpoints"))
    unchanged = sha(path) == original_digest
    profiling = _profile_check(repeated)
    result = dict(index=index, checkpoint_pair=index+5, result_cells=len(new),
                  integer_fields_checked=len(new)*len(COUNT_FIELDS),
                  energy_fields_checked=len(new)*len(ENERGY_FIELDS),
                  frame_digests_exact=frames_exact, identities_exact=identity_exact,
                  result_grid_exact=grid_exact, original_record_unchanged=unchanged,
                  fresh_record_sha256=original_digest, integer_counts_exact=not integers,
                  integer_differences=integers, energy_floats_exact=not energies,
                  energy_relative_tolerance=RELATIVE_TOLERANCE, energy_absolute_tolerance=0.,
                  energy_differences=energies, energy_agreement=not energy_outside,
                  energy_outside_tolerance=energy_outside,
                  max_energy_absolute_difference=max((row["absolute_difference"] for row in energies), default=0.),
                  profiling=profiling, replay_seconds=repeated["seconds"])
    result["passed"] = (frames_exact and identity_exact and grid_exact and unchanged and not integers
                         and not energy_outside and profiling["conditional_objective_nonincrease_within_tolerance"])
    print(f"Independent replay {index}: passed={result['passed']}; "
          f"integer differences={len(integers)}, energy differences={len(energies)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "independent_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new --output path; do not replace audit evidence")
    ex.setup()
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  reviewer="separate collaborating agent in the same environment",
                  scope="raw-count/statistic review plus separate-process checkpoint inference replay",
                  limitations="same CPU/runtime; no retraining, external laboratory or second-hardware reproduction",
                  training_reproduced=False, preselected_indices=list(REPLAY_INDICES),
                  environment=environment(), auditor_source_sha256=sha(Path(__file__)))
    try:
        result["freeze_sha256"] = verify_freeze()
        files = [RESULTS / "fresh" / f"pair{index:02d}.json" for index in range(1, 11)]
        require(set((RESULTS / "fresh").glob("pair*.json")) == set(files), "Unexpected fresh result files")
        records = [read(path) for path in files]
        require(validate(records) == result["freeze_sha256"], "Whole-cohort validation identity differs")
        result["all_raw_grids_valid"] = True
        result["input_sha256"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
        result["profiling_all_records"] = [_profile_check(record) for record in records]
        result["statistics"] = independent_statistics(records, read(RESULTS / "summary.json"))
        result["summary_sha256"] = sha(RESULTS / "summary.json")
        result["replays"] = [replay(index) for index in REPLAY_INDICES]
        result["inherited_changes"] = inherited_changes()
        result["input_records_unchanged"] = all(sha(ROOT/path) == digest
                                               for path, digest in result["input_sha256"].items())
        result["passed"] = (result["statistics"]["passed"] and not result["inherited_changes"]
                             and result["input_records_unchanged"]
                             and all(row["passed"] for row in result["replays"])
                             and all(row["conditional_objective_nonincrease_within_tolerance"]
                                     for row in result["profiling_all_records"])
                             and result["freeze_sha256"] == verify_freeze())
    except Exception:
        result["passed"] = False
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)
    print("Independent-in-this-environment audit passed", flush=True)


if __name__ == "__main__":
    main()
