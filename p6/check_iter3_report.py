"""Audit iteration-3 receipts without importing the simulator or study implementation."""

import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKS = RESULTS / "iter3_checks"
PAIRS = (901, 902, 903)
ALL = ("pure", "local", "parity", "local_parity", "no_gauge")
PARITY = ("parity", "local_parity")
BUDGETS = [0, 24, 48, 64, 80]
FIXED = {"physics", "physics_parity"}
CLASSICAL = {1: {"nominal_bussgang_lmmse", "nominal_idd_pa", "pilot_cpe_dd", "genie_pa"},
             2: {"tap_phase_slope", "lmmse_2d", "waveform_dd"}, 3: {"lmmse_2d", "waveform_dd"}}
FIELDS = ("block_errors", "bit_errors", "bler", "pre_ber", "bce", "nmse_db")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numbers(row, expected, label):
    for key, value in expected.items():
        actual = row[key]
        ok = actual == value if type(value) is int else math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)
        assert ok, (label, key, actual, value)


def pooled(raw, regime):
    total = {k: sum(row[k] for row in raw) for k in raw[0]}
    assert total["blocks"] == (1728 if regime == 2 else 384)
    assert total["bits"] == 128 * total["blocks"] and total["info_bits"] == 64 * total["blocks"]
    return dict(**total, bler=total["block_errors"] / total["blocks"],
                pre_ber=total["bit_errors"] / total["bits"], bce=total["bce_sum"] / total["bits"],
                nmse_db=10 * math.log10(total["channel_error"] / total["channel_energy"]))


def delta(row, base):
    return {k: row[k] - base[k] for k in FIELDS}


def probe_checks(report, jobs):
    arms = ("pure", "parity")
    assert set(report["kernels"]) == set(arms)
    uniform = report["uniform_green_support"]
    assert uniform["stream"] == 8 and uniform["total_bursts"] == 24
    for p in PAIRS:
        point = uniform["parallel_joint"]["by_pair"][str(p)]
        assert point["seed"] == 308_000_000 + p * 1000 and len(point["per_burst"]) == 8
        numbers(point, {"mean": statistics.mean(point["per_burst"])}, "uniform support")
    for arm in arms:
        for stage in ("initial", "trained"):
            summary = report["kernels"][arm][stage]
            assert set(summary) == {"spatial", "time", "frequency", "parallel_joint", "joint"}
            for name, columns in summary.items():
                for column, values in columns.items():
                    by_pair = {}
                    for p in PAIRS:
                        layers = jobs["capped", 3, p, arm]["probe"][stage]
                        assert len(layers) == 3 and all(x["off_diagonal_entries"] == 2256 for x in layers)
                        raw = [x for layer in layers for x in layer["maps"][name][column]]
                        assert len(raw) == 24 and all(x is None or -1.000001 <= x <= 1.000001 for x in raw)
                        finite = [x for x in raw if x is not None]
                        assert finite, (stage, arm, name, column, p)
                        by_pair[str(p)] = statistics.mean(finite)
                    numbers(values["by_pair"], by_pair, (stage, arm, name, column))
                    numbers(values, {"mean": statistics.mean(by_pair.values())}, (stage, arm, name, column))


def verification(report):
    summary = read(CHECKS / "summary.json")
    assert report["verification"]["summary"] == summary
    assert summary["passed"] and summary["implementation"] == report["implementation"]
    suite = read(CHECKS / "p6_suite.json")
    tests = [name for path in (ROOT / "tests").glob("test_*.py")
             for name in re.findall(r"^\s+def (test_\w+)\(", path.read_text(encoding="utf-8"), re.M)]
    assert suite["passed"] and summary["p6_tests"] == suite["run"] == len(tests)
    assert summary["p6_skipped"] == suite["skipped"] == 0 and all(name in suite["log"] for name in tests)
    assert (summary["p5_tests"], summary["p5b_tests"], summary["p4"]["exit"]) == (15, 13, 0)
    assert summary["capacity_matched"] and summary["step1_all_live"]
    assert summary["live_real_per_regime_arm"] == [report["trainable_real_parameters"]] * 15
    assert summary["exact_anchors"] == summary["initialized_anchors"] == 30
    assert summary["max_archived_init_diff"] == 0 and summary["finite_difference_directions"] > 0
    rows = []
    for r in CLASSICAL:
        for p in PAIRS:
            rec = read(CHECKS / f"trained_r{r}_{p}.json")
            assert rec["implementation"] == report["implementation"] and (rec["regime"], rec["pair"]) == (r, p)
            assert len(rec["rows"]) == len({(x["arm"], x["condition"]) for x in rec["rows"]}) == 40
            assert {x["arm"] for x in rec["rows"]} == set(ALL) - {"no_gauge"}
            for x in rec["rows"]:
                assert (x["regime"], x["pair"], x["updates"], x["seed"]) == (r, p, 80, r*10**8+6*10**6+p*1000)
                assert x["scales"][0] == 1e-35 and x["scales"][-1] == 1e20
                assert x["finite"] and x["passed"] and x["coded_flips"] == 0 and x["max_grid_llr_diff"] < 1e-12
            rows.extend(rec["rows"])
    gauge = summary["trained_gauge"]
    assert gauge["passed"] and not gauge["failures"] and gauge["checks"] == gauge["passed_checks"] == len(rows) == 360
    numbers(gauge, {"decisions": sum(x["decisions"] for x in rows), "coded_flips": 0,
                    "max_grid_llr_diff": max(x["max_grid_llr_diff"] for x in rows)}, "trained gauge")


def main():
    if not __debug__:
        raise RuntimeError("This receipt audit requires Python assertions")
    report = read(RESULTS / "dev_iter3.json")
    config = report["config"]
    assert config["checkpoints"] == BUDGETS and config["updates"] == 80 and config["selection"] == "fixed update 80"
    assert config["dev_pairs"] == list(PAIRS) and config["streams"] == dict(train=5, score=6, r3_probe=8, r3_permutation=9)
    numbers(config, dict(cap=10., aux_nmse=.04, anchor_coefficient_on_sum_sq=.5e-4, train_batch=4,
                         validation_batch=32, probe_batch=8, depth=3), "design")
    jobs = {(j["loss"], j["regime"], j["pair"], j["arm"]): j for j in report["by_pair"]}
    expected = {(loss, r, p, a) for loss, arms in (("capped", (*ALL, "references")), ("uncapped", PARITY))
                for r in CLASSICAL for p in PAIRS for a in arms}
    assert set(jobs) == expected and len(report["by_pair"]) == 72
    for (loss, r, p, a), job in jobs.items():
        path = RESULTS / "iter3_jobs" / loss / f"r{r}" / str(p) / f"{a}.json"
        assert job == read(path), path
        assert job["implementation"] == report["implementation"] and job["depth"] == 3
        seed = r * 10**8 + 5 * 10**6 + p * 1000
        assert job["score_seed"] == seed + 10**6
        if a == "references":
            continue
        assert job["parameters"] == report["trainable_real_parameters"]
        assert job["init_seed"] == seed + 900 and (job["first_training_seed"], job["last_training_seed"]) == (seed, seed+79)
        assert job["initial_weights_sha256"] == jobs["capped", r, p, "pure"]["initial_weights_sha256"]
        assert [t["update"] for t in job["trace"]] == BUDGETS
        assert job["trace"][0]["score"] == job["initial_score"] and job["trace"][-1]["score"] == job["score"]
        ref = jobs["capped", r, p, "references"]["scores"]["physics_parity" if a in PARITY else "physics"]
        numbers(job["initial_score"], ref, path)
    for r in CLASSICAL:
        regime = str(r)
        table = report["pooled"][regime]
        assert set(table) == set(ALL) | FIXED | CLASSICAL[r]
        for a in FIXED | CLASSICAL[r]:
            numbers(table[a], pooled([jobs["capped", r, p, "references"]["scores"][a] for p in PAIRS], r), a)
        for loss, arms in (("capped", ALL), ("uncapped", PARITY)):
            for arm in arms:
                group = [jobs[loss, r, p, arm] for p in PAIRS]
                for i, budget in enumerate(BUDGETS):
                    total = pooled([j["trace"][i]["score"] for j in group], r)
                    numbers(report["curves"][loss][regime][arm][str(budget)], total, (loss, r, arm, budget))
                if loss != "capped":
                    continue
                numbers(table[arm], total, (r, arm, 80))
                if arm not in PARITY:
                    continue
                labels = {"all", "full", "delta_h", "cpe"} | {
                    1: {"radial", "radial_and_cpe", "emit_only"}, 2: {"phase", "soft_bank", "third_pass"},
                    3: {"delays", "radial", "phase", "spatial", "spatial_operator"}}[r]
                ablation = report["suppression"][regime][arm]
                assert set(ablation["rows"]) == labels
                for label in labels:
                    rows = [j["suppressions"][label] for j in group]
                    recomputed = pooled(rows, r)
                    numbers(ablation["rows"][label], recomputed, (r, arm, label))
                    numbers(ablation["removed_minus_full"][label], delta(recomputed, total), (r, arm, label))
                    assert ablation["block_delta_by_pair"][label] == {
                        str(j["pair"]): x["block_errors"] - j["score"]["block_errors"] for j, x in zip(group, rows)}
                numbers(ablation["rows"]["all"], table["physics_parity"], (r, arm, "all removed"))
                if r in (2, 3):
                    assert ablation["rows"]["cpe"] == ablation["rows"]["full"]
        for label, a, b in (("pure_minus_local", "pure", "local"), ("parity_minus_local_parity", *PARITY),
                            ("parity_minus_fixed_parity", "parity", "physics_parity")):
            numbers(report["differences"][regime][label], delta(table[a], table[b]), (r, label))
        latency = read(RESULTS / "iter3_jobs" / f"latency_r{r}.json")
        assert report["latency"][regime] == latency and latency["torch_threads"] == latency["torch_interop_threads"] == 1
        for batch in ("1", "64"):
            assert set(latency["rows"][batch]) == set(table)
            for a, obs in latency["rows"][batch].items():
                assert len(obs["wall_ms"]) == latency["repeats"]
                numbers(obs, {"median_ms": statistics.median(obs["wall_ms"])}, (r, a, batch))
    probe_checks(report, jobs)
    verification(report)
    print("PASS: 72 receipts, matched initialization, primary scores, curves, signed ablations, Green probes, timing and numerical gates")


if __name__ == "__main__":
    main()
