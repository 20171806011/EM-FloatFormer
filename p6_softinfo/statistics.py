"""Complete-grid validation and six fixed paired BLER comparisons."""
import itertools
import math
import statistics

from scipy.stats import t as student_t

from p6_followup.statistics import _rates, _score_valid, _sha, _sum_scores, sign_flip
from p6_softinfo.common import (CONDITIONS, METHODS, PRIMARY_BASELINES,
    PRIMARY_CONDITIONS, RESULTS, ROOT, SEED_BASE, read, save, sha, verify_freeze)


def validate(records):
    signature = verify_freeze()
    if len(records) != 10 or {r["index"] for r in records} != set(range(1, 11)):
        raise ValueError("Require all ten distinct paired clusters")
    expected_grid = set(itertools.product(range(4), CONDITIONS, METHODS))
    all_seeds = []
    for record in records:
        index = record["index"]
        if record["freeze_sha256"] != signature or record["checkpoint_pair"] != index+5:
            raise ValueError("Wrong freeze or checkpoint identity")
        if len(record["frames"]) != 4:
            raise ValueError("Wrong frame count")
        seeds = [frame["seed"] for frame in record["frames"]]
        if seeds != [SEED_BASE+1000*index+j for j in range(4)]:
            raise ValueError("Wrong or reordered registered seeds")
        all_seeds.extend(seeds)
        for frame in record["frames"]:
            if frame["batch_size"] != 8 or not all(_sha(frame[key]) for key in ("y_sha256", "bits_sha256")):
                raise ValueError("Wrong frame size or absent observation/payload digest")
        cells = [(r["batch_index"], r["condition"], r["method"]) for r in record["rows"]]
        if len(cells) != len(expected_grid) or set(cells) != expected_grid:
            raise ValueError("Duplicate, missing, or extra result cell")
        for row in record["rows"]:
            _score_valid(row["score"], f"cluster{index}/{row['condition']}/{row['method']}")
    if len(set(all_seeds)) != 40:
        raise ValueError("Repeated fresh simulator seed")
    return signature


def effect(by_seed, condition, baseline, candidate):
    differences, denominators = [], []
    for index in range(1, 11):
        base = by_seed[index, condition, baseline]
        new = by_seed[index, condition, candidate]
        if base["blocks"] != new["blocks"]:
            raise ValueError("Unpaired exposure")
        denominators.append(base["blocks"])
        differences.append(base["block_errors"]-new["block_errors"])
    if len(set(denominators)) != 1:
        raise ValueError("Unequal cluster sizes")
    pp = [100*d/denominators[0] for d in differences]
    mean = statistics.mean(pp)
    half = float(student_t.ppf(.975, 9))*statistics.stdev(pp)/math.sqrt(10)
    return dict(condition=condition, baseline=baseline, candidate=candidate,
                direction="positive means candidate has fewer decoded block errors",
                seed_block_error_reductions=differences, mean_bler_reduction_pp=mean,
                pointwise_t95_bler_reduction_pp=[mean-half, mean+half],
                wins=sum(d>0 for d in differences), ties=sum(d==0 for d in differences),
                losses=sum(d<0 for d in differences), blocks_per_cluster=denominators[0])


def summarize(records):
    signature = validate(records)
    by_seed, seed_rows = {}, []
    for record in sorted(records, key=lambda r: r["index"]):
        for condition, method in itertools.product(CONDITIONS, METHODS):
            score = _sum_scores(row["score"] for row in record["rows"]
                                if (row["condition"], row["method"]) == (condition, method))
            by_seed[record["index"], condition, method] = score
            seed_rows.append(dict(index=record["index"], checkpoint_pair=record["checkpoint_pair"],
                                  condition=condition, method=method, **score, **_rates(score)))
    aggregates = []
    for condition, method in itertools.product(CONDITIONS, METHODS):
        score = _sum_scores(by_seed[i, condition, method] for i in range(1, 11))
        aggregates.append(dict(condition=condition, method=method, **score, **_rates(score)))
    comparisons = [(base, "variance_profile") for base in PRIMARY_BASELINES]
    comparisons += [("trust", "variance"), ("trust", "profile"), ("variance", "variance_profile")]
    effects = [effect(by_seed, condition, base, candidate)
               for condition in CONDITIONS for base, candidate in comparisons]
    primary = [dict(row, **sign_flip(row["seed_block_error_reductions"])) for row in effects
               if row["candidate"] == "variance_profile" and row["baseline"] in PRIMARY_BASELINES
               and row["condition"] in PRIMARY_CONDITIONS]
    if len(primary) != 6:
        raise ValueError("Primary family must contain exactly six tests")
    running = 0.
    for rank, row in enumerate(sorted(primary, key=lambda r: r["p_raw"])):
        running = max(running, min(1., (len(primary)-rank)*row["p_raw"]))
        row["p_holm"] = running
    return dict(schema_version=1, freeze_sha256=signature,
        design=dict(clusters=10, bursts_per_condition=320, blocks_per_condition=5760,
                    coded_bits_per_condition=737280, conditions=list(CONDITIONS), methods=list(METHODS)),
        aggregates=aggregates, by_seed=seed_rows, effects=effects, primary_tests=primary,
        inference_notes=[
          "No post-fresh candidate selection; variance_profile was fixed before fresh scoring.",
          "Ten checkpoint/fresh-waveform clusters are inferential units; blocks within each burst are not independent replicates.",
          "Two-sided exact enumeration of 1024 paired sign assignments assumes independent symmetric/exchangeable null cluster signs.",
          "Holm corrects the six declared candidate comparisons; ablations, BER, NMSE, and remaining conditions are descriptive.",
          "Pointwise t95 effect intervals are approximate, unadjusted, and based on only ten clusters.",
          "Frozen weights are reused; no retraining or new-channel-family generalization is established.",
          "Digital pilot corruption is unflagged; classical priors match the public intact model, not the added corruption.",
          "Surrogate-objective descent is a numerical property, not an error-rate guarantee.",
          "No external independent reproduction is included.",
        ])


def main():
    paths = [RESULTS/"fresh"/f"pair{i:02d}.json" for i in range(1, 11)]
    if set((RESULTS/"fresh").glob("pair*.json")) != set(paths):
        raise ValueError("Incomplete or unexpected fresh result files")
    result = summarize([read(path) for path in paths])
    result["input_files"] = [dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)) for path in paths]
    save(RESULTS/"summary.json", result)
    for row in result["primary_tests"]:
        print({key: row[key] for key in ("condition", "baseline", "mean_bler_reduction_pp", "wins", "ties", "losses", "p_holm")}, flush=True)


if __name__ == "__main__":
    main()
