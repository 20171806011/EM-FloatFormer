"""Separate-process inference replay and independently coded count/statistic audit.

This runner was written by a separate collaborating agent after the scoring
implementation. It does not retrain any model and runs in the same environment.
It never overwrites fresh study records or the frozen manifest.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import traceback

from p6 import experiment as ex
from p6_followup.common import BASE, REPO, ROOT, environment, sha, verify_freeze
from p6_followup.study import score_pair


DEST = ROOT / "results" / "independent_audit.json"
COUNTS = ("bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks")
ENERGIES = ("channel_error", "channel_energy")
PRIMARY = ("first_erased", "interior_erased")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def independently_aggregate():
    """Use a sum-distribution recurrence, not the main sign-enumeration helper."""
    files = [ROOT / "results" / "fresh" / f"pair{i:02d}.json" for i in range(1, 11)]
    records = [read(path) for path in files]
    summary = read(ROOT / "results" / "summary.json")
    grouped_counts = defaultdict(lambda: Counter())
    cluster_errors = Counter()
    for expected, record in enumerate(records, 1):
        if record["index"] != expected or record["checkpoint_pair"] != expected + 5:
            raise ValueError("Fresh cluster/checkpoint mapping disagrees")
        for row in record["rows"]:
            key = row["condition"], row["method"]
            grouped_counts[key].update({field: row["score"][field] for field in COUNTS})
            cluster_errors[expected, *key] += row["score"]["block_errors"]
    summary_counts = {(r["condition"], r["method"]): {field: r[field] for field in COUNTS}
                      for r in summary["aggregates"]}
    counts_exact = (set(summary_counts) == set(grouped_counts) and
                    all(dict(grouped_counts[key]) == summary_counts[key] for key in grouped_counts))
    tests = []
    for condition in PRIMARY:
        differences = [cluster_errors[i, condition, "trust"] -
                       cluster_errors[i, condition, "smooth"] for i in range(1, 11)]
        distribution = Counter({0: 1})
        for difference in differences:
            changed = Counter()
            for partial, multiplicity in distribution.items():
                changed[partial + difference] += multiplicity
                changed[partial - difference] += multiplicity
            distribution = changed
        threshold = abs(sum(differences))
        tail = sum(multiplicity for value, multiplicity in distribution.items()
                   if abs(value) >= threshold)
        total = sum(distribution.values())
        tests.append(dict(condition=condition, seed_block_error_reductions=differences,
                          signflip_tail=tail, signflip_total=total, p_raw=tail / total))
    first, second = sorted(tests, key=lambda row: row["p_raw"])
    first["p_holm"] = min(1., 2 * first["p_raw"])
    second["p_holm"] = max(first["p_holm"], second["p_raw"])
    published = {row["condition"]: row for row in summary["primary_tests"]}
    test_matches = {test["condition"]: all(published[test["condition"]][field] == value
                                          for field, value in test.items()) for test in tests}
    return dict(
        aggregation_algorithm="independent Counter accumulation of six integer fields",
        test_algorithm="dynamic-programming integer sum distribution of independent signs",
        aggregate_rows=len(grouped_counts), integer_fields_per_row=len(COUNTS),
        aggregate_integer_counts_exact=counts_exact,
        primary_tests=tests, primary_tests_exact=test_matches,
        input_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        summary_sha256=sha(ROOT / "results" / "summary.json"),
        passed=counts_exact and all(test_matches.values()),
    )


def replay(index):
    original_path = ROOT / "results" / "fresh" / f"pair{index:02d}.json"
    original_digest = sha(original_path)
    original = read(original_path)
    new = score_pair(index, replay=True)
    old_rows = {(row["batch_index"], row["condition"], row["method"]): row for row in original["rows"]}
    new_rows = {(row["batch_index"], row["condition"], row["method"]): row for row in new["rows"]}
    cells_exact = set(old_rows) == set(new_rows) and len(new_rows) == 140
    integer_differences, energy_differences, diagnostic_differences = [], [], []
    for key in sorted(old_rows.keys() & new_rows.keys()):
        before, after = old_rows[key], new_rows[key]
        for field in COUNTS:
            if before["score"][field] != after["score"][field]:
                integer_differences.append(dict(cell=list(key), field=field,
                                                stored=before["score"][field], replayed=after["score"][field]))
        for field in ENERGIES:
            if before["score"][field] != after["score"][field]:
                energy_differences.append(dict(cell=list(key), field=field,
                                               stored=before["score"][field], replayed=after["score"][field]))
        for field in ("signs", "pilot_q_min", "pilot_q_mean", "pilot_q_below_one"):
            if before.get(field) != after.get(field):
                diagnostic_differences.append(dict(cell=list(key), field=field))
    identity_exact = all(original[field] == new[field] for field in
                         ("index", "checkpoint_pair", "freeze_sha256", "checkpoints"))
    frames_exact = original["frames"] == new["frames"]
    unchanged = sha(original_path) == original_digest
    result = dict(
        index=index, checkpoint_pair=index+5, fresh_record_sha256=original_digest,
        frame_digests_exact=frames_exact, identities_exact=identity_exact,
        result_grid_exact=cells_exact, result_cells=len(new_rows),
        integer_fields_checked=len(new_rows) * len(COUNTS),
        integer_counts_exact=not integer_differences, integer_differences=integer_differences,
        energy_fields_checked=len(new_rows) * len(ENERGIES),
        energy_floats_exact=not energy_differences, energy_differences=energy_differences,
        max_energy_absolute_difference=max((abs(row["stored"] - row["replayed"])
                                             for row in energy_differences), default=0.),
        diagnostics_exact=not diagnostic_differences, diagnostic_differences=diagnostic_differences,
        original_record_unchanged=unchanged, replay_seconds=new["seconds"],
    )
    result["passed"] = (frames_exact and identity_exact and cells_exact and unchanged
                         and not integer_differences and not energy_differences
                         and not diagnostic_differences)
    print(f"Independent replay index {index}: passed={result['passed']}, "
          f"integer differences={len(integer_differences)}, "
          f"energy differences={len(energy_differences)}", flush=True)
    return result


def main():
    if DEST.exists():
        raise FileExistsError("Do not overwrite an independent audit receipt")
    ex.setup()
    result = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        reviewer="separate collaborating agent",
        scope="separate-process checkpoint inference and independent statistic calculation",
        independence_limits="same machine and runtime; no external laboratory, fresh training or second-hardware replication",
        training_reproduced=False, preselected_indices=[1, 10],
        environment=environment(),
        auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        result["freeze_sha256"] = verify_freeze()
        result["replays"] = [replay(index) for index in (1, 10)]
        result["statistics"] = independently_aggregate()
        changed = subprocess.check_output(["git", "diff", BASE, "--name-only", "--",
                                           "p4", "p5", "p5b", "p6"], cwd=REPO, text=True).strip()
        result["inherited_changes"] = changed
        result["passed"] = (not changed and all(row["passed"] for row in result["replays"])
                             and result["statistics"]["passed"]
                             and result["freeze_sha256"] == verify_freeze())
    except Exception:
        result["passed"] = False
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        DEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)
    print("Independent replay and count/statistic audit passed", flush=True)


if __name__ == "__main__":
    main()
