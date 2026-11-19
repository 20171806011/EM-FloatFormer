"""Validate the fixed cohort and test two predeclared mechanism contrasts.

Known corruption metadata is privileged diagnostic information. The main
interaction concerns how the variance increment changes with the observation
handling; it is neither a blind-receiver comparison nor a causal mediation
proof. Every other contrast is descriptive, including intermediate ablations.
"""

import itertools
import math
import statistics

from scipy.stats import t as student_t

from p6_followup.statistics import _rates, _score_valid, _sha, _sum_scores, sign_flip
from p6_consistency.common import (
    CONDITIONS, METHODS, RESULTS, ROOT, SEED_BASE, read, save, sha, verify_freeze,
)


INFORMATION_CLASSES = {
    "legacy_mean": "unflagged_original_observation",
    "legacy_variance": "unflagged_original_observation",
    "aware_mean": "known_receive_gain_and_erasure_metadata",
    "aware_variance": "known_receive_gain_and_erasure_metadata",
    "aware_effective": "known_receive_gain_and_erasure_metadata",
    "aware_cavity_mean": "known_receive_gain_and_erasure_metadata",
    "aware_consistent": "known_receive_gain_and_erasure_metadata",
}
CHAIN_COMPARISONS = (
    ("legacy_mean", "legacy_variance"),
    ("aware_mean", "aware_variance"),
    ("legacy_mean", "aware_mean"),
    ("legacy_variance", "aware_variance"),
    ("aware_variance", "aware_effective"),
    ("aware_effective", "aware_consistent"),
    ("aware_cavity_mean", "aware_consistent"),
    ("aware_variance", "aware_consistent"),
    ("aware_mean", "aware_consistent"),
)
INTERACTION_COEFFICIENTS = {
    "legacy_variance": 1, "legacy_mean": -1,
    "aware_variance": -1, "aware_mean": 1,
}
CONSISTENCY_COEFFICIENTS = {"aware_variance": 1, "aware_consistent": -1}
VARIANCE_EDGES = [0., .01, .05, .1, .2, .4, .8, None]
MOMENT_FIELDS = ("sum_predicted_variance", "sum_squared_error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _close(first, second):
    return math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-12)


def _bit_valid(row, where):
    require(type(row.get("bit_count")) is int and row["bit_count"] == 18432,
            f"{where}: wrong payload bit count")
    require(all(_nonnegative(row.get(key)) for key in
                ("bce_sum", "sum_abs_llr", "mean_abs_llr", "max_abs_llr")),
            f"{where}: invalid bit calibration summary")
    require(_close(row["mean_abs_llr"], row["sum_abs_llr"]/row["bit_count"]),
            f"{where}: inconsistent mean absolute LLR")
    require(row["max_abs_llr"] <= row["sum_abs_llr"] + 1e-12 and
            row["mean_abs_llr"] <= row["max_abs_llr"] + 1e-12,
            f"{where}: inconsistent maximum absolute LLR")


def _moment_valid(row, where):
    require(type(row.get("count")) is int and 0 <= row["count"] <= 4608,
            f"{where}: invalid payload symbol count")
    require(all(_nonnegative(row.get(key)) for key in MOMENT_FIELDS),
            f"{where}: invalid symbol calibration sum")
    if not row["count"]:
        require(all(row[key] == 0 for key in MOMENT_FIELDS),
                f"{where}: empty calibration bin has nonzero mass")


def _calibration_valid(row, where):
    require(isinstance(row, dict) and row.get("kind") == "evaluation_only_posterior_calibration",
            f"{where}: missing or unrecognized calibration record")
    require(type(row.get("batch_size")) is int and row["batch_size"] == 8 and
            type(row.get("payload_symbols")) is int and row["payload_symbols"] == 4608,
            f"{where}: wrong calibration exposure")
    require(row.get("variance_bin_edges") == VARIANCE_EDGES,
            f"{where}: calibration edges changed")
    steps = row.get("steps", [])
    require(len(steps) == 3 and all(type(step.get("step")) is int for step in steps) and
            {step["step"] for step in steps} == {0, 1, 2},
            f"{where}: expected three distinct M-step points")
    for step in steps:
        label = f"{where}/step{step['step']}"
        total, bins = step["total"], step["bins"]
        _moment_valid(total, label)
        require(total["count"] == 4608 and len(bins) == 7,
                f"{label}: incomplete symbol exposure or bins")
        for index, entry in enumerate(bins):
            require(type(entry.get("index")) is int and entry["index"] == index and
                    entry.get("lower") == VARIANCE_EDGES[index] and
                    entry.get("upper") == VARIANCE_EDGES[index+1],
                    f"{label}: calibration bin labels changed")
            _moment_valid(entry, f"{label}/bin{index}")
            low = entry["lower"] * entry["count"]
            require(entry["sum_predicted_variance"] >= low or
                    _close(entry["sum_predicted_variance"], low),
                    f"{label}: predicted variance below bin lower bound")
            if entry["upper"] is not None:
                high = entry["upper"] * entry["count"]
                require(entry["sum_predicted_variance"] <= high or
                        _close(entry["sum_predicted_variance"], high),
                        f"{label}: predicted variance above bin upper bound")
        require(sum(entry["count"] for entry in bins) == total["count"],
                f"{label}: bin counts do not partition payload")
        require(all(_close(math.fsum(entry[key] for entry in bins), total[key])
                    for key in MOMENT_FIELDS),
                f"{label}: calibration bin sums disagree with total")
        if "channel" in step:
            _bit_valid(step["channel"], label+"/channel")
    _bit_valid(row["final"], where+"/final")


def _sum_moments(rows):
    rows = list(rows)
    result = {"count": sum(row["count"] for row in rows),
              **{key: math.fsum(row[key] for row in rows) for key in MOMENT_FIELDS}}
    count, variance = result["count"], result["sum_predicted_variance"]
    result.update(mean_predicted_variance=variance/count if count else None,
                  mean_squared_error=result["sum_squared_error"]/count if count else None,
                  error_to_variance_ratio=result["sum_squared_error"]/variance if variance > 0 else None)
    return result


def _sum_bits(rows):
    rows = list(rows)
    result = {"bit_count": sum(row["bit_count"] for row in rows),
              **{key: math.fsum(row[key] for row in rows) for key in ("bce_sum", "sum_abs_llr")},
              "max_abs_llr": max(row["max_abs_llr"] for row in rows)}
    result.update(mean_bce=result["bce_sum"]/result["bit_count"],
                  mean_abs_llr=result["sum_abs_llr"]/result["bit_count"])
    return result


def _calibration_aggregate(rows):
    rows = list(rows)
    steps = []
    for index in range(3):
        selected = [next(step for step in row["steps"] if step["step"] == index) for row in rows]
        entry = dict(step=index, total=_sum_moments(step["total"] for step in selected),
                     bins=[dict(index=bin_index, lower=VARIANCE_EDGES[bin_index],
                                upper=VARIANCE_EDGES[bin_index+1],
                                **_sum_moments(step["bins"][bin_index] for step in selected))
                           for bin_index in range(7)])
        require(len({"channel" in step for step in selected}) == 1,
                "Inconsistent availability of intermediate channel calibration")
        if "channel" in selected[0]:
            entry["channel"] = _sum_bits(step["channel"] for step in selected)
        steps.append(entry)
    return dict(variance_bin_edges=VARIANCE_EDGES, steps=steps,
                final=_sum_bits(row["final"] for row in rows))


def validate(records):
    """Reject incomplete grids, changed exposures, seeds or checkpoint receipts."""
    signature = verify_freeze()
    require(len(records) == 10, "Exactly ten paired clusters are required")
    require(all(type(record.get("index")) is int for record in records),
            "Cluster indices must be integers")
    require({record["index"] for record in records} == set(range(1, 11)),
            "Cluster indices must be exactly 1 through 10")
    require(set(METHODS) == set(INFORMATION_CLASSES), "Unregistered receiver family")
    required_cells = set(itertools.product(range(4), CONDITIONS, METHODS))
    manifest = read(ROOT / "FREEZE.json")
    seeds_seen = []
    for record in records:
        index = record["index"]
        require(record.get("freeze_sha256") == signature, f"Cluster {index}: wrong freeze")
        require(type(record.get("checkpoint_pair")) is int and record["checkpoint_pair"] == index+5,
                f"Cluster {index}: wrong archived checkpoint pair")
        expected_path = f"p6/results/formal/A/r2/pair{index+5:02d}/parity.pt"
        checkpoints = record.get("checkpoints", {})
        require(isinstance(checkpoints, dict) and "parity" in checkpoints,
                f"Cluster {index}: absent parity checkpoint provenance")
        receipt = checkpoints["parity"]
        require(receipt.get("path") == expected_path and _sha(receipt.get("sha256")),
                f"Cluster {index}: checkpoint path or digest is malformed")
        require(manifest["checkpoints"].get(expected_path) == receipt["sha256"],
                f"Cluster {index}: checkpoint differs from the study freeze")
        fit = read((ROOT.parent / expected_path).parent / "fit.json")
        require(receipt.get("stored_implementation") == fit["freeze_sha256"],
                f"Cluster {index}: archived checkpoint implementation differs")
        frames = record.get("frames", [])
        require(len(frames) == 4, f"Cluster {index}: expected four B8 frames")
        expected_seeds = [SEED_BASE + 1000*index + batch for batch in range(4)]
        require([frame.get("seed") for frame in frames] == expected_seeds,
                f"Cluster {index}: wrong or reordered fresh seeds")
        for frame in frames:
            require(type(frame["seed"]) is int and frame.get("batch_size") == 8,
                    f"Cluster {index}: invalid seed type or simulator batch size")
            require(_sha(frame.get("y_sha256")) and _sha(frame.get("bits_sha256")),
                    f"Cluster {index}: missing frame fingerprints")
            seeds_seen.append(frame["seed"])
        rows = record.get("rows", [])
        actual = [(row.get("batch_index"), row.get("condition"), row.get("method")) for row in rows]
        require(len(actual) == len(required_cells) and set(actual) == required_cells,
                f"Cluster {index}: duplicate, missing or unregistered score cells")
        for row in rows:
            require(type(row["batch_index"]) is int, f"Cluster {index}: batch index must be an integer")
            _score_valid(row["score"], f"Cluster {index}/{row['batch_index']}/"
                         f"{row['condition']}/{row['method']}")
            _calibration_valid(row.get("calibration"), f"Cluster {index}/{row['batch_index']}/"
                               f"{row['condition']}/{row['method']}")
    require(len(set(seeds_seen)) == 40, "Fresh waveform seeds repeat across clusters")
    return signature


def contrast(by_seed, condition, coefficients, name, kind, direction):
    """A zero-sum count contrast on the same ten equal-exposure clusters."""
    require(sum(coefficients.values()) == 0 and coefficients,
            "A paired count contrast must be nonempty and sum to zero")
    differences, denominators = [], []
    for index in range(1, 11):
        exposure = {by_seed[index, condition, method]["blocks"] for method in coefficients}
        require(len(exposure) == 1, "Receiver exposure differs inside a contrast")
        denominators.append(exposure.pop())
        differences.append(sum(coefficient * by_seed[index, condition, method]["block_errors"]
                               for method, coefficient in coefficients.items()))
    require(len(set(denominators)) == 1, "Cluster exposures differ")
    points = [100 * difference / denominators[0] for difference in differences]
    average = statistics.mean(points)
    half_width = float(student_t.ppf(.975, 9)) * statistics.stdev(points) / math.sqrt(10)
    return dict(
        name=name, kind=kind, condition=condition, coefficients=coefficients,
        direction=direction, seed_block_error_contrasts=differences,
        seed_effect_pp=points, mean_effect_pp=average,
        pointwise_t95_effect_pp=[average-half_width, average+half_width],
        blocks_per_cluster=denominators[0],
        favorable_clusters=sum(difference > 0 for difference in differences),
        tied_clusters=sum(difference == 0 for difference in differences),
        adverse_clusters=sum(difference < 0 for difference in differences),
    )


def summarize(records):
    signature = validate(records)
    by_seed, seed_rows = {}, []
    for record in sorted(records, key=lambda item: item["index"]):
        for condition, method in itertools.product(CONDITIONS, METHODS):
            score = _sum_scores(row["score"] for row in record["rows"]
                                if (row["condition"], row["method"]) == (condition, method))
            by_seed[record["index"], condition, method] = score
            seed_rows.append(dict(index=record["index"], checkpoint_pair=record["checkpoint_pair"],
                                  condition=condition, method=method,
                                  information_class=INFORMATION_CLASSES[method], **score, **_rates(score)))
    aggregates, calibration = [], []
    for condition, method in itertools.product(CONDITIONS, METHODS):
        score = _sum_scores(by_seed[index, condition, method] for index in range(1, 11))
        aggregates.append(dict(condition=condition, method=method,
                               information_class=INFORMATION_CLASSES[method], **score, **_rates(score)))
        calibration.append(dict(condition=condition, method=method,
            **_calibration_aggregate(row["calibration"] for record in records for row in record["rows"]
                                     if (row["condition"], row["method"]) == (condition, method))))
    effects = []
    for condition, (baseline, candidate) in itertools.product(CONDITIONS, CHAIN_COMPARISONS):
        effect = contrast(by_seed, condition, {baseline: 1, candidate: -1},
                          name=f"{candidate}_versus_{baseline}", kind="paired_receiver_difference",
                          direction="positive means candidate has fewer decoded block errors")
        effect.update(baseline=baseline, candidate=candidate,
                      same_information_class=INFORMATION_CLASSES[baseline] == INFORMATION_CLASSES[candidate])
        effects.append(effect)
    interactions = [contrast(
        by_seed, condition, INTERACTION_COEFFICIENTS,
        name="observation_by_variance_interaction", kind="difference_in_differences",
        direction="positive means known-gain-aware handling reduces the block-error increment from variance")
        for condition in CONDITIONS]
    primary = [next(row.copy() for row in interactions if row["condition"] == "scattered_erased")]
    primary.append(contrast(
        by_seed, "scattered_erased", CONSISTENCY_COEFFICIENTS,
        name="aware_consistency_addition", kind="paired_receiver_difference",
        direction="positive means aware_consistent has fewer block errors than aware_variance at the same metadata access"))
    for row in primary:
        row.update(sign_flip(row["seed_block_error_contrasts"]))
    running = 0.
    for rank, row in enumerate(sorted(primary, key=lambda item: item["p_raw"])):
        running = max(running, min(1., (2-rank)*row["p_raw"]))
        row["p_holm"] = running
    return dict(
        schema_version=1, freeze_sha256=signature,
        design=dict(clusters=10, batches_per_cluster=4, bursts_per_batch=8,
                    bursts_per_condition=320, blocks_per_condition=5760,
                    coded_bits_per_condition=737280, information_bits_per_condition=368640,
                    conditions=list(CONDITIONS), methods=list(METHODS),
                    information_classes=INFORMATION_CLASSES,
                    primary_family=[row["name"] for row in primary]),
        aggregates=aggregates, by_seed=seed_rows, effects=effects,
        calibration=calibration, variance_handling_interactions=interactions, primary_tests=primary,
        inference_notes=[
            "The two primary contrasts and all receiver variants are fixed before fresh evaluation; no post-fresh winner selection.",
            "Ten archived-checkpoint/fresh-waveform clusters are the inferential units; blocks or repeated conditions are not independent replicates.",
            "Two-sided enumeration of 1024 paired sign assignments assumes exchangeable/symmetric independent null cluster signs.",
            "Holm adjustment covers exactly the two declared scattered-erasure contrasts. Other conditions, intermediate ablations, BCE, BER, NMSE and calibration are descriptive.",
            "Pointwise Student-t 95% intervals are approximate, unadjusted and based on ten clusters.",
            "Calibration bins, posterior update points and final uncapped channel-LLR BCE are evaluation-only descriptive diagnostics. The error/variance ratio is neither a proper score nor an additional efficacy test; zero-variance or empty cells have null ratios.",
            "Known receive-gain/erasure metadata is privileged diagnostic information. These implementations are not blind receivers or mathematical performance upper bounds.",
            "Legacy-versus-aware differences change information access and observation handling. Same-information comparisons are explicitly marked.",
            "A favorable variance-handling interaction does not by itself establish lower absolute BLER or identify one causal mediator: shared learned weights and intermediate beliefs change the full receiver trajectory.",
            "Nonsignificance does not establish equivalence or noninferiority; no such margin is registered.",
            "Frozen weights are reused. No new training, unseen-channel-family validation, or external independent reproduction is established.",
        ],
    )


def main():
    paths = [RESULTS / "fresh" / f"pair{index:02d}.json" for index in range(1, 11)]
    require(set((RESULTS / "fresh").glob("pair*.json")) == set(paths),
            "Fresh result files are incomplete or unexpected")
    result = summarize([read(path) for path in paths])
    result["input_files"] = [dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)) for path in paths]
    save(RESULTS / "summary.json", result)
    for row in result["primary_tests"]:
        print({field: row[field] for field in
               ("name", "condition", "mean_effect_pp", "favorable_clusters", "tied_clusters",
                "adverse_clusters", "p_raw", "p_holm")}, flush=True)


if __name__ == "__main__":
    main()
