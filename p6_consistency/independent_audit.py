"""Independent-in-this-environment arithmetic review and inference replay.

This file is authored separately from the receiver and experimental runner.
It reuses the same CPU/runtime, frozen weights and simulator; it does not
reproduce training or establish external/second-hardware reproduction.
Use a new --output path when repeating an already retained audit.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import traceback

from scipy.special import stdtrit

from p6 import experiment as ex
from p6_consistency.common import (
    CONDITIONS, METHODS, RESULTS, ROOT, environment, inherited_changes,
    read, save, sha, source_hashes, verify_freeze,
)
from p6_consistency.statistics import validate
from p6_consistency.study import score_pair


COUNT_FIELDS = ("bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks")
ENERGY_FIELDS = ("channel_error", "channel_energy")
REPLAY_INDICES = (1, 10)
RELATIVE_TOLERANCE = 1e-12
PRIMARY_COEFFICIENTS = {
    "observation_by_variance_interaction": {
        "legacy_variance": 1, "legacy_mean": -1, "aware_variance": -1, "aware_mean": 1},
    "aware_consistency_addition": {"aware_variance": 1, "aware_consistent": -1},
}
MOMENT_FIELDS = ("sum_predicted_variance", "sum_squared_error")
ERASED_BINS = {"intact": 0, "first_attenuated": 0, "first_erased": 16,
               "interior_erased": 16, "scattered_erased": 32}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(first, second):
    return (math.isfinite(first) and math.isfinite(second) and
            math.isclose(first, second, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.))


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def independent_statistics(records, summary):
    """Manual count sums and integer sum-distribution recurrence, not sign_flip."""
    counts, energies = defaultdict(Counter), defaultdict(lambda: defaultdict(list))
    seed_counts, seed_energies = defaultdict(Counter), defaultdict(lambda: defaultdict(list))
    for record in records:
        for row in record["rows"]:
            key = row["condition"], row["method"]
            seed_key = record["index"], *key
            counts[key].update({field: row["score"][field] for field in COUNT_FIELDS})
            seed_counts[seed_key].update({field: row["score"][field] for field in COUNT_FIELDS})
            for field in ENERGY_FIELDS:
                energies[key][field].append(row["score"][field])
                seed_energies[seed_key][field].append(row["score"][field])
    published = {(row["condition"], row["method"]): row for row in summary["aggregates"]}
    published_seeds = {(row["index"], row["condition"], row["method"]): row for row in summary["by_seed"]}
    grid = set(itertools.product(CONDITIONS, METHODS))
    seed_grid = set(itertools.product(range(1, 11), CONDITIONS, METHODS))
    require(len(summary["aggregates"]) == len(grid) and set(published) == set(counts) == grid,
            "Aggregate grid differs")
    require(len(summary["by_seed"]) == len(seed_grid) and set(published_seeds) == set(seed_counts) == seed_grid,
            "Per-cluster summary grid differs")
    integer_exact = all(counts[key][field] == published[key][field]
                        for key in grid for field in COUNT_FIELDS)
    seed_integer_exact = all(seed_counts[key][field] == published_seeds[key][field]
                             for key in seed_grid for field in COUNT_FIELDS)
    energy_agreement = all(close(math.fsum(energies[key][field]), published[key][field])
                           for key in grid for field in ENERGY_FIELDS)
    seed_energy_agreement = all(close(math.fsum(seed_energies[key][field]), published_seeds[key][field])
                                for key in seed_grid for field in ENERGY_FIELDS)
    rate_agreement = True
    for grouped, published_group, energy_group in ((counts, published, energies),
                                                  (seed_counts, published_seeds, seed_energies)):
        for key, row in grouped.items():
            expected = dict(ber=row["bit_errors"]/row["bits"], info_ber=row["info_errors"]/row["info_bits"],
                            bler=row["block_errors"]/row["blocks"],
                            nmse=math.fsum(energy_group[key]["channel_error"])/
                                 math.fsum(energy_group[key]["channel_energy"]))
            expected["nmse_db"] = 10*math.log10(expected["nmse"]) if expected["nmse"] > 0 else None
            for field, value in expected.items():
                actual = published_group[key][field]
                rate_agreement &= actual is None if value is None else close(actual, value)
    tests = []
    for name, coefficients in PRIMARY_COEFFICIENTS.items():
        differences = [sum(coefficient*seed_counts[index, "scattered_erased", method]["block_errors"]
                           for method, coefficient in coefficients.items()) for index in range(1, 11)]
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
        mean = 100*sum(differences)/5760
        points = [100*difference/576 for difference in differences]
        standard_error = math.sqrt(math.fsum((point-mean)**2 for point in points)/90)
        # Share the recorded runtime's distribution quantile; recompute the
        # standard error directly. Count/sign-flip arithmetic is independent.
        half_width = float(stdtrit(9, .975))*standard_error
        tests.append(dict(name=name, condition="scattered_erased", coefficients=coefficients,
            seed_block_error_contrasts=differences, mean_effect_pp=mean,
            pointwise_t95_effect_pp=[mean-half_width, mean+half_width],
            signflip_tail=tail, signflip_total=total, p_raw=tail/total,
            favorable_clusters=sum(value > 0 for value in differences),
            tied_clusters=sum(value == 0 for value in differences),
            adverse_clusters=sum(value < 0 for value in differences)))
    ordered = sorted(tests, key=lambda row: row["p_raw"])
    for rank, test in enumerate(ordered):
        test["p_holm"] = min(1., max((2-k)*ordered[k]["p_raw"] for k in range(rank+1)))
    published_tests = {row["name"]: row for row in summary["primary_tests"]}
    require(len(summary["primary_tests"]) == 2 and set(published_tests) == set(PRIMARY_COEFFICIENTS),
            "Primary inferential family changed")
    matches = []
    for test in tests:
        target = published_tests[test["name"]]
        exact_fields = [field for field in test if field not in ("mean_effect_pp", "pointwise_t95_effect_pp")]
        discrete = all(test[field] == target[field] for field in exact_fields)
        effect = close(test["mean_effect_pp"], target["mean_effect_pp"])
        interval = all(close(first, second) for first, second in
                       zip(test["pointwise_t95_effect_pp"], target["pointwise_t95_effect_pp"]))
        matches.append(dict(name=test["name"], discrete_statistics_exact=discrete,
                            mean_effect_agrees=effect, pointwise_interval_agrees=interval))
    passed = (integer_exact and seed_integer_exact and energy_agreement and seed_energy_agreement and
              rate_agreement and all(all(row[key] for key in ("discrete_statistics_exact",
                  "mean_effect_agrees", "pointwise_interval_agrees")) for row in matches))
    return dict(algorithm="independent Counter sums and integer sum-distribution recurrence",
                aggregate_rows=len(grid), per_cluster_rows=len(seed_grid),
                aggregate_integer_counts_exact=integer_exact, per_cluster_integer_counts_exact=seed_integer_exact,
                aggregate_energy_agreement=energy_agreement, per_cluster_energy_agreement=seed_energy_agreement,
                reported_rates_agree=rate_agreement, primary_tests=tests, primary_test_matches=matches,
                passed=passed)


def independent_calibration(records, summary):
    """Re-sum raw fixed-bin counts/masses without using the statistics helpers."""
    grouped = defaultdict(list)
    for record in records:
        for row in record["rows"]:
            grouped[row["condition"], row["method"]].append(row["calibration"])
    published = {(row["condition"], row["method"]): row for row in summary["calibration"]}
    require(len(summary["calibration"]) == len(grouped) and set(published) == set(grouped),
            "Calibration summary grid differs")
    failures, checked = [], Counter()

    def check(where, field, actual, expected):
        checked["scalars"] += 1
        if expected is None:
            agreement = actual is None
        elif type(expected) is int:
            checked["integer_counts"] += 1
            agreement = type(actual) is int and actual == expected
        else:
            checked["floating_sums_or_derived_values"] += 1
            agreement = type(actual) in (int, float) and close(actual, expected)
        if not agreement:
            failures.append(dict(where=where, field=field, stored=actual, independently_computed=expected))

    def moment(where, actual, rows):
        count = sum(row["count"] for row in rows)
        variance = math.fsum(row["sum_predicted_variance"] for row in rows)
        error = math.fsum(row["sum_squared_error"] for row in rows)
        expected = dict(count=count, sum_predicted_variance=variance, sum_squared_error=error,
                        mean_predicted_variance=variance/count if count else None,
                        mean_squared_error=error/count if count else None,
                        error_to_variance_ratio=error/variance if variance > 0 else None)
        for field, value in expected.items():
            check(where, field, actual[field], value)

    def bits(where, actual, rows):
        count = sum(row["bit_count"] for row in rows)
        bce = math.fsum(row["bce_sum"] for row in rows)
        absolute = math.fsum(row["sum_abs_llr"] for row in rows)
        expected = dict(bit_count=count, bce_sum=bce, sum_abs_llr=absolute,
                        max_abs_llr=max(row["max_abs_llr"] for row in rows),
                        mean_bce=bce/count, mean_abs_llr=absolute/count)
        for field, value in expected.items():
            check(where, field, actual[field], value)

    for key, rows in grouped.items():
        where = "/".join(key)
        target = published[key]
        require(target["variance_bin_edges"] == [0., .01, .05, .1, .2, .4, .8, None],
                f"Calibration aggregate edges differ: {where}")
        steps = {step["step"]: step for step in target["steps"]}
        require(set(steps) == {0, 1, 2} and len(target["steps"]) == 3,
                f"Calibration aggregate update points differ: {where}")
        for index in range(3):
            inputs = [next(step for step in row["steps"] if step["step"] == index) for row in rows]
            moment(f"{where}/step{index}/total", steps[index]["total"], [step["total"] for step in inputs])
            require(len(steps[index]["bins"]) == 7, f"Wrong aggregate bin count: {where}")
            for bin_index in range(7):
                destination = steps[index]["bins"][bin_index]
                source = [step["bins"][bin_index] for step in inputs]
                require(all(destination[field] == source[0][field] for field in ("index", "lower", "upper")),
                        f"Aggregate calibration bin labels differ: {where}/{index}/{bin_index}")
                moment(f"{where}/step{index}/bin{bin_index}", destination, source)
            if all("channel" in step for step in inputs):
                bits(f"{where}/step{index}/channel", steps[index]["channel"], [step["channel"] for step in inputs])
        bits(where+"/final", target["final"], [row["final"] for row in rows])
    return dict(receiver_condition_rows=len(grouped), checked=dict(checked),
                relative_tolerance=RELATIVE_TOLERANCE, absolute_tolerance=0.,
                failures=failures, passed=not failures)


def telemetry(records):
    """Verify masks, solve/cavity ranges and the positive-gain hard-decision control."""
    totals, attenuated_mismatches = defaultdict(Counter), []
    for record in records:
        cells = {(row["batch_index"], row["condition"], row["method"]): row for row in record["rows"]}
        for row in record["rows"]:
            method, condition = row["method"], row["condition"]
            where = f"{record['index']}/{row['batch_index']}/{condition}/{method}"
            diagnostics = row.get("diagnostics")
            require(isinstance(diagnostics, dict), f"Missing diagnostics: {where}")
            moments = diagnostics.get("moments", [])
            expected = [] if method == "legacy_mean" else [0, 1, 2]
            require([item["step"] for item in moments] == expected, f"Wrong moment steps: {where}")
            for item in moments:
                require(nonnegative(item["mean_variance"]) and item["mean_variance"] <= 1.8+1e-12 and
                        nonnegative(item["covariance_trace_mean"]), f"Invalid moments: {where}")
            if not method.startswith("aware_"):
                continue
            observed = 640-ERASED_BINS[condition]
            require(diagnostics["transmitted_pilots"] == 64 and
                    diagnostics["observed_pilots"] == 64-ERASED_BINS[condition] and
                    diagnostics["observed_frequency_bins"] == observed,
                    f"Receive/TX pilot metadata disagrees: {where}")
            counter = totals[condition, method]
            counter["score_cells"] += 1
            for item in moments:
                require(all(nonnegative(item[field]) for field in ("covariance_rhs_norm", "objective_sum",
                                                                  "uncertainty_energy_sum")),
                        f"Invalid physical fit diagnostics: {where}")
                require(type(item["pinv_batches"]) is int and 0 <= item["pinv_batches"] <= 8,
                        f"Invalid pseudoinverse count: {where}")
                counter["amplitude_updates"] += 8
                counter["pinv_batches"] += item["pinv_batches"]
                if method in ("aware_mean", "aware_cavity_mean"):
                    require(item["covariance_trace_mean"] == item["covariance_rhs_norm"] ==
                            item["uncertainty_energy_sum"] == 0, f"Mean-only fit contains variance: {where}")
                if method == "aware_variance":
                    require(item["covariance_rhs_norm"] == 0, f"Residual-free fit has a covariance RHS: {where}")
            cavity = diagnostics.get("cavity", [])
            require([item["step"] for item in cavity] ==
                    ([0, 1, 2, "final"] if method in ("aware_cavity_mean", "aware_consistent") else []),
                    f"Wrong cavity call points: {where}")
            for item in cavity:
                require(item["valid_observations_by_burst"] == [observed]*8 and
                        len(item["predictive_quadratic_by_burst"]) == 8 and
                        all(nonnegative(value) for value in item["predictive_quadratic_by_burst"]),
                        f"Invalid predictive residual or observation exposure: {where}")
                for field, maximum in (("informed_payload", 4608), ("computational_noise_floor_frames", 160),
                                       ("cavity_variance_clamps", 4608)):
                    require(type(item[field]) is int and 0 <= item[field] <= maximum,
                            f"Invalid cavity counter {field}: {where}")
                require(item["payload_symbols"] == 4608 and
                        all(nonnegative(item[field]) for field in ("mean_abs_channel_llr", "max_abs_channel_llr",
                            "mean_prior_variance", "mean_posterior_variance")), f"Invalid cavity summary: {where}")
                counter["cavity_calls"] += 1
                counter["computational_noise_floor_frames"] += item["computational_noise_floor_frames"]
                counter["cavity_variance_clamps"] += item["cavity_variance_clamps"]
            if condition == "first_attenuated":
                intact = cells[row["batch_index"], "intact", method]
                if any(row["score"][field] != intact["score"][field] for field in COUNT_FIELDS):
                    attenuated_mismatches.append(where)
    return dict(totals=[dict(condition=key[0], method=key[1], **value) for key, value in sorted(totals.items())],
                positive_gain_hard_decisions_match=not attenuated_mismatches,
                positive_gain_mismatches=attenuated_mismatches, passed=not attenuated_mismatches)


def tree_comparison(first, second):
    """Compare all calibration/diagnostic scalars, retaining exactness separately."""
    differences, outside, checked = [], [], Counter()

    def visit(old, new, path):
        if isinstance(old, dict):
            if not isinstance(new, dict) or set(old) != set(new):
                item = dict(path=path, kind="dictionary_keys")
                differences.append(item)
                outside.append(item)
            if isinstance(new, dict):
                for key in sorted(set(old) & set(new)):
                    visit(old[key], new[key], f"{path}/{key}")
        elif isinstance(old, list):
            if not isinstance(new, list) or len(old) != len(new):
                item = dict(path=path, kind="list_length")
                differences.append(item)
                outside.append(item)
            if isinstance(new, list):
                for index, (a, b) in enumerate(zip(old, new)):
                    visit(a, b, f"{path}/{index}")
        else:
            checked["scalars"] += 1
            exact = type(old) is type(new) and old == new
            if type(old) is float:
                checked["floats"] += 1
                agreed = type(new) is float and close(old, new)
            else:
                checked["discrete"] += 1
                agreed = exact
            if not exact:
                item = dict(path=path, stored=old, replayed=new)
                differences.append(item)
                if not agreed:
                    outside.append(item)

    visit(first, second, "")
    return dict(checked=dict(checked), exact=not differences, agreement=not outside,
                differences=differences, outside_tolerance=outside,
                relative_tolerance=RELATIVE_TOLERANCE, absolute_tolerance=0.)


def numerical_receipt():
    """Recheck the retained gate arithmetic; an honest failing gate is not erased."""
    path = RESULTS/"numerical_audit.json"
    record = read(path)
    require(record["freeze_sha256"] == verify_freeze() and
            record["freeze_sha256_after"] == record["freeze_sha256"] and
            record["strict_threshold"] == 1e-12 and record["strict_comparison"] == "less_than",
            "Numerical-audit identity or strict criterion changed")
    expected = set(itertools.product((901, 902, 903), CONDITIONS, METHODS[2:]))
    actual = {(row["pair"], row["condition"], row["mode"]) for row in record["rows"]}
    require(actual == expected and len(record["rows"]) == len(expected) and not record["forward_exceptions"],
            "Numerical transformation cohort incomplete")
    totals, failures = defaultdict(Counter), []
    for row in record["rows"]:
        require(len(row["transformations"]) == 27, "Numerical transformation count differs")
        passed_values = []
        for transform in row["transformations"]:
            passed = (transform["finite"] is True and transform["coded_flips"] == 0 and
                      transform["max_grid_llr_diff"] < 1e-12 and transform["max_relative_cfr_l2"] < 1e-12)
            require(transform["passed"] == passed, "A raw transformation's strict label disagrees")
            passed_values.append(passed)
            totals[row["mode"]]["transformations"] += 1
            totals[row["mode"]]["passed"] += int(passed)
            if not passed:
                failures.append(dict(pair=row["pair"], condition=row["condition"], method=row["mode"],
                                     **transform))
        require(row["passed"] == all(passed_values), "A condition-level numerical gate label disagrees")
    for mode, values in totals.items():
        stored = record["summary_by_mode"][mode]
        require(stored["gain_transformations"] == values["transformations"] and
                stored["passed_transformations"] == values["passed"] and
                stored["failed_transformations"] == values["transformations"]-values["passed"],
                "Numerical aggregate gate counts disagree")
    strict = not failures
    require(record["strict_passed"] == strict, "Numerical overall gate label disagrees")
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                receipt_arithmetic_consistent=True, strict_gate_passed=strict,
                transformations=sum(value["transformations"] for value in totals.values()),
                passed_transformations=sum(value["passed"] for value in totals.values()),
                strict_failed_transformations=failures,
                interpretation="Audit consistency does not turn a strict numerical failure into a receiver pass.")


def replay(index):
    path = RESULTS/"fresh"/f"pair{index:02d}.json"
    original_digest = sha(path)
    original = read(path)
    repeated = score_pair(index, replay=True)
    key = lambda row: (row["batch_index"], row["condition"], row["method"])
    old, new = {key(row): row for row in original["rows"]}, {key(row): row for row in repeated["rows"]}
    grid_exact = set(old) == set(new) and len(new) == 4*len(CONDITIONS)*len(METHODS)
    integer_differences, energy_differences, energy_outside = [], [], []
    for cell in sorted(old.keys() & new.keys()):
        for field in COUNT_FIELDS:
            a, b = old[cell]["score"][field], new[cell]["score"][field]
            if a != b:
                integer_differences.append(dict(cell=list(cell), field=field, stored=a, replayed=b))
        for field in ENERGY_FIELDS:
            a, b = old[cell]["score"][field], new[cell]["score"][field]
            if a != b:
                item = dict(cell=list(cell), field=field, stored=a, replayed=b, absolute_difference=abs(a-b))
                energy_differences.append(item)
                if not close(a, b):
                    energy_outside.append(item)
    metadata = tree_comparison(
        [{"cell": list(cell), "calibration": old[cell]["calibration"], "diagnostics": old[cell]["diagnostics"]}
         for cell in sorted(old)],
        [{"cell": list(cell), "calibration": new[cell]["calibration"], "diagnostics": new[cell]["diagnostics"]}
         for cell in sorted(new)])
    frames_exact = original["frames"] == repeated["frames"]
    identity_exact = all(original[field] == repeated[field] for field in
                         ("index", "checkpoint_pair", "freeze_sha256", "checkpoints"))
    unchanged = sha(path) == original_digest
    result = dict(index=index, checkpoint_pair=index+5, result_cells=len(new),
                  integer_fields_checked=len(new)*len(COUNT_FIELDS), energy_fields_checked=len(new)*len(ENERGY_FIELDS),
                  frame_digests_exact=frames_exact, identities_exact=identity_exact, result_grid_exact=grid_exact,
                  original_record_unchanged=unchanged, fresh_record_sha256=original_digest,
                  integer_counts_exact=not integer_differences, integer_differences=integer_differences,
                  energy_floats_exact=not energy_differences, energy_agreement=not energy_outside,
                  energy_differences=energy_differences, energy_outside_tolerance=energy_outside,
                  energy_relative_tolerance=RELATIVE_TOLERANCE, energy_absolute_tolerance=0.,
                  calibration_and_diagnostics=metadata, replay_seconds=repeated["seconds"])
    result["passed"] = (grid_exact and frames_exact and identity_exact and unchanged and
                         not integer_differences and not energy_outside and metadata["agreement"])
    print(f"Independent replay {index}: passed={result['passed']}; integer differences={len(integer_differences)}; "
          f"energy differences={len(energy_differences)}; calibration/diagnostic differences={len(metadata['differences'])}",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS/"independent_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new --output path; never replace audit evidence")
    ex.setup()
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        reviewer="separate collaborating agent in the same environment",
        scope="raw-grid/count/calibration/statistic review and separate-process frozen-checkpoint inference replay",
        limitations="same CPU/runtime; no retraining, external laboratory, second-hardware or independent-data collection",
        training_reproduced=False, preselected_indices=list(REPLAY_INDICES), environment=environment(),
        auditor_source_sha256=sha(Path(__file__)),
        auditor_development_notes=[
            "An in-memory pre-replay self-check using duplicated development JSON caught a stale hardcoded t9 quantile differing from the recorded SciPy quantile at about 1e-10; the audit now uses stdtrit, while recomputing count contrasts, standard errors and sign-flip distributions separately. No candidate source, formal data or inferential rule changed."])
    try:
        result["freeze_sha256"] = verify_freeze()
        result["critical_sources_before"] = source_hashes()
        files = [RESULTS/"fresh"/f"pair{index:02d}.json" for index in range(1, 11)]
        require(set((RESULTS/"fresh").glob("pair*.json")) == set(files), "Unexpected fresh files")
        records = [read(path) for path in files]
        require(validate(records) == result["freeze_sha256"], "Full-grid identity differs")
        result["all_raw_grids_valid"] = True
        result["input_sha256"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
        summary = read(RESULTS/"summary.json")
        require(summary["freeze_sha256"] == result["freeze_sha256"], "Summary freeze identity differs")
        require({row["path"]: row["sha256"] for row in summary["input_files"]} == result["input_sha256"],
                "Summary source-result identities differ")
        result["summary_sha256"] = sha(RESULTS/"summary.json")
        result["statistics"] = independent_statistics(records, summary)
        result["calibration_arithmetic"] = independent_calibration(records, summary)
        result["telemetry"] = telemetry(records)
        result["numerical_receipt"] = numerical_receipt()
        result["replays"] = [replay(index) for index in REPLAY_INDICES]
        result["critical_sources_after"] = source_hashes()
        result["critical_sources_unchanged"] = result["critical_sources_before"] == result["critical_sources_after"]
        result["inherited_changes"] = inherited_changes()
        result["input_records_unchanged"] = all(sha(ROOT/path) == digest
                                               for path, digest in result["input_sha256"].items())
        result["summary_unchanged"] = sha(RESULTS/"summary.json") == result["summary_sha256"]
        result["numerical_receipt_unchanged"] = sha(ROOT/result["numerical_receipt"]["path"]) == result["numerical_receipt"]["sha256"]
        result["passed"] = (all(result[key]["passed"] for key in ("statistics", "calibration_arithmetic", "telemetry"))
            and all(row["passed"] for row in result["replays"]) and result["critical_sources_unchanged"]
            and not result["inherited_changes"] and result["input_records_unchanged"] and result["summary_unchanged"]
            and result["numerical_receipt_unchanged"]
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
