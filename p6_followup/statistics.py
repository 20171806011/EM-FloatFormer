"""Validate and aggregate the frozen P6 follow-up's ten paired clusters.

Run from the repository root with ``python -m p6_followup.statistics``.
The inferential unit is an archived-checkpoint/new-waveform cluster, not an
individual LDPC block. All seven receivers see the same corrupted observation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent
FRESH = ROOT / "results" / "fresh"
SUMMARY = ROOT / "results" / "summary.json"
CONDITIONS = (
    "intact", "first_attenuated", "first_erased", "interior_erased",
    "scattered_erased",
)
METHODS = (
    "trust", "smooth", "hard_chart", "local_parity", "fixed_parity",
    "waveform_dd", "lmmse_2d",
)
PRIMARY = ("first_erased", "interior_erased")
COUNT_FIELDS = (
    "bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks",
)
ENERGY_FIELDS = ("channel_error", "channel_energy")
SCORE_FIELDS = (*COUNT_FIELDS, *ENERGY_FIELDS)
DENOMINATORS = {"bits": 18432, "info_bits": 9216, "blocks": 144}
SEED_BASE = 602611190000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _score_valid(score, where):
    _require(isinstance(score, dict), f"{where}: score must be a mapping")
    _require(all(key in score for key in SCORE_FIELDS), f"{where}: missing score fields")
    _require(all(type(score[key]) is int for key in COUNT_FIELDS),
             f"{where}: counts must be exact integers, not booleans or rounded floats")
    for key, expected in DENOMINATORS.items():
        _require(score[key] == expected, f"{where}: wrong {key}, expected {expected}")
    for error, denominator in (("bit_errors", "bits"), ("info_errors", "info_bits"),
                               ("block_errors", "blocks")):
        _require(0 <= score[error] <= score[denominator], f"{where}: invalid {error}")
    _require(score["blocks"] * 128 == score["bits"] and
             score["blocks"] * 64 == score["info_bits"],
             f"{where}: LDPC dimensions disagree")
    _require(score["block_errors"] <= score["info_errors"] <= 64 * score["block_errors"],
             f"{where}: information-bit and block errors disagree")
    for key in ENERGY_FIELDS:
        _require(type(score[key]) in (int, float) and math.isfinite(score[key]),
                 f"{where}: {key} must be a finite real number")
    _require(score["channel_error"] >= 0 and score["channel_energy"] > 0,
             f"{where}: invalid channel energy")


def validate(records):
    """Reject partial cohorts, stale seeds, repeated cells and unequal exposure."""
    _require(len(records) == 10, "Exactly ten fresh clusters are required")
    _require({r.get("index") for r in records} == set(range(1, 11)),
             "Fresh cluster indices must be exactly 1 through 10")
    identities = {r.get("freeze_sha256") for r in records}
    _require(len(identities) == 1 and _sha(next(iter(identities))),
             "All clusters must identify one valid frozen protocol/source manifest")
    required = set(itertools.product(range(4), CONDITIONS, METHODS))
    all_seeds = []
    for record in records:
        index = record["index"]
        _require(type(index) is int, "Cluster indices must be integers")
        _require(record.get("checkpoint_pair") == index + 5,
                 f"Cluster {index}: wrong historical checkpoint pair")
        frames = record.get("frames", [])
        _require(len(frames) == 4, f"Cluster {index}: expected four B8 simulator draws")
        expected_seeds = {SEED_BASE + 1000 * index + batch for batch in range(4)}
        _require({f.get("seed") for f in frames} == expected_seeds,
                 f"Cluster {index}: unregistered waveform seeds")
        for frame in frames:
            _require(type(frame["seed"]) is int and frame.get("batch_size") == 8,
                     f"Cluster {index}: wrong frame seed type or batch size")
            _require(_sha(frame.get("y_sha256")) and _sha(frame.get("bits_sha256")),
                     f"Cluster {index}: missing/invalid frame hashes")
            all_seeds.append(frame["seed"])
        rows = record.get("rows", [])
        _require(len(rows) == len(required), f"Cluster {index}: incomplete result grid")
        actual = [(r.get("batch_index"), r.get("condition"), r.get("method")) for r in rows]
        _require(len(set(actual)) == len(actual) and set(actual) == required,
                 f"Cluster {index}: repeated, missing or unexpected result cells")
        for row in rows:
            _require(type(row["batch_index"]) is int,
                     f"Cluster {index}: batch index must be an integer")
            _score_valid(row["score"], f"Cluster {index}/{row['batch_index']}/"
                         f"{row['condition']}/{row['method']}")
    _require(len(set(all_seeds)) == 40, "Waveform seeds repeat across clusters")
    return next(iter(identities))


def _sum_scores(scores):
    rows = list(scores)
    return {
        **{key: sum(r[key] for r in rows) for key in COUNT_FIELDS},
        **{key: math.fsum(r[key] for r in rows) for key in ENERGY_FIELDS},
    }


def _rates(score):
    ratio = score["channel_error"] / score["channel_energy"]
    return dict(
        ber=score["bit_errors"] / score["bits"],
        info_ber=score["info_errors"] / score["info_bits"],
        bler=score["block_errors"] / score["blocks"],
        nmse=ratio,
        nmse_db=10 * math.log10(ratio) if ratio > 0 else None,
    )


def sign_flip(differences):
    """Enumerate all paired sign assignments; integer arithmetic includes ties.

    Exact enumeration is not an assumption-free test of equal mean error.
    Validity requires exchangeable signs under the paired null (for example,
    independent, symmetric zero-centered cluster differences).
    """
    _require(bool(differences) and all(type(d) is int for d in differences),
             "Sign-flip test requires a nonempty list of integer count differences")
    observed = abs(sum(differences))
    tail = sum(
        abs(sum(sign * d for sign, d in zip(signs, differences))) >= observed
        for signs in itertools.product((-1, 1), repeat=len(differences))
    )
    total = 2 ** len(differences)
    return dict(signflip_tail=tail, signflip_total=total, p_raw=tail / total)


def _effect(condition, by_seed):
    differences, denominator = [], None
    for index in range(1, 11):
        base = by_seed[index, condition, "trust"]
        candidate = by_seed[index, condition, "smooth"]
        _require(base["blocks"] == candidate["blocks"], "Paired denominators differ")
        _require(denominator in (None, base["blocks"]), "Cluster sizes differ")
        denominator = base["blocks"]
        differences.append(base["block_errors"] - candidate["block_errors"])
    percentage_points = [100 * d / denominator for d in differences]
    average = statistics.mean(percentage_points)
    standard_error = statistics.stdev(percentage_points) / math.sqrt(len(differences))
    half_width = float(student_t.ppf(.975, len(differences) - 1)) * standard_error
    return dict(
        condition=condition,
        direction="positive means smooth has fewer block errors than frozen trust",
        seed_block_error_reductions=differences,
        seed_bler_reduction_pp=percentage_points,
        blocks_per_cluster=denominator,
        mean_bler_reduction_pp=average,
        pointwise_t95_bler_reduction_pp=[average - half_width, average + half_width],
        wins=sum(d > 0 for d in differences),
        ties=sum(d == 0 for d in differences),
        losses=sum(d < 0 for d in differences),
    )


def summarize(records):
    freeze_sha256 = validate(records)
    by_seed = {}
    seed_rows = []
    for record in sorted(records, key=lambda row: row["index"]):
        for condition, method in itertools.product(CONDITIONS, METHODS):
            score = _sum_scores(row["score"] for row in record["rows"]
                                if (row["condition"], row["method"]) == (condition, method))
            by_seed[record["index"], condition, method] = score
            seed_rows.append(dict(index=record["index"], checkpoint_pair=record["checkpoint_pair"],
                                  condition=condition, method=method, **score, **_rates(score)))
    aggregates = []
    for condition, method in itertools.product(CONDITIONS, METHODS):
        score = _sum_scores(by_seed[index, condition, method] for index in range(1, 11))
        aggregates.append(dict(condition=condition, method=method, **score, **_rates(score)))
    effects = [_effect(condition, by_seed) for condition in CONDITIONS]
    primary = [{**effect, **sign_flip(effect["seed_block_error_reductions"])}
               for effect in effects if effect["condition"] in PRIMARY]
    running = 0.
    for rank, test in enumerate(sorted(primary, key=lambda row: row["p_raw"])):
        running = max(running, min(1., (len(primary) - rank) * test["p_raw"]))
        test["p_holm"] = running
    return dict(
        schema_version=1,
        freeze_sha256=freeze_sha256,
        design=dict(clusters=10, batches_per_cluster=4, bursts_per_batch=8,
                    bursts_per_condition=320, blocks_per_condition=5760,
                    coded_bits_per_condition=737280, conditions=list(CONDITIONS),
                    methods=list(METHODS), primary_family=list(PRIMARY)),
        aggregates=aggregates,
        by_seed=seed_rows,
        smooth_vs_trust=effects,
        primary_tests=primary,
        inference_notes=[
            "Inferential units are ten archived-checkpoint/new-waveform clusters; LDPC blocks within a burst are not independent replicates.",
            "The two-sided sign-flip distribution enumerates all 1024 assignments and assumes paired-null exchangeability/symmetric independent cluster effects.",
            "Holm adjustment covers exactly the two predeclared erased-row BLER comparisons; other outcomes are descriptive.",
            "Displayed 95% Student-t intervals are pointwise, unadjusted and approximate; only ten clusters are available and normal cluster effects are not established.",
            "The study reuses frozen trained weights on fresh waveforms. It does not reproduce training or establish generalization to unseen channel families.",
            "Pilot corruption is unflagged digital attenuation/erasure. A baseline matched to the original public channel prior is not matched to this added observation corruption.",
            "Failure to reject a difference does not establish equivalence or noninferiority.",
        ],
    )


def main():
    paths = [FRESH / f"pair{index:02d}.json" for index in range(1, 11)]
    unexpected = set(FRESH.glob("pair*.json")) - set(paths)
    _require(not unexpected, f"Unexpected fresh result files: {sorted(unexpected)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = summarize(records)
    result["input_files"] = [dict(path=path.relative_to(ROOT).as_posix(),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for path in paths]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for test in result["primary_tests"]:
        print(f"{test['condition']}: BLER reduction {test['mean_bler_reduction_pp']:.6f} pp; "
              f"wins/ties/losses {test['wins']}/{test['ties']}/{test['losses']}; "
              f"p={test['p_raw']:.8f}, Holm={test['p_holm']:.8f}")


if __name__ == "__main__":
    main()
