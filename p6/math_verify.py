"""Regression verification, numerical gate and immutable formal manifest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch

from p6 import experiment as ex
from p6.math_dev import OUT, path_for, training_sources
from p6.iter5_dev import location as previous_location
from p6.data import PAIRS, SNR
from p6.extended_audit import ARMS as GAUGED, summarize
from p6.run_dev_iter2 import probe_summary
from p5b.diagnostics import SCALES


ROOT = ex.ROOT
FREEZE = ex.RESULTS / "freeze.json"
REPO = ROOT.parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected():
    return {
        p.relative_to(REPO).as_posix(): sha(p)
        for name in ("p5", "p5b")
        for p in sorted((REPO / name).rglob("*"))
        if p.is_file()
    }


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tracked_inventory():
    entries = subprocess.check_output(["git","ls-tree","-r","-z","8380084"], cwd=REPO).split(b"\0")
    files = [entry.split(b"\t",1) for entry in entries if entry]
    paths = [name for _,name in files]
    if any(b"\n" in name or b'"' in name for name in paths):
        raise ValueError("Unexpected name in tracked inventory")
    payload = b"".join(name+b"\n" for name in paths)
    def git_hashes(options):
        return subprocess.check_output(["git","hash-object",*options,"--stdin-paths"],
                                       cwd=REPO,input=payload).splitlines()
    clean, raw = git_hashes([]), git_hashes(["--no-filters"])
    result = {}
    for (header,name), filtered, literal in zip(files,clean,raw,strict=True):
        original = header.split()[2]
        if original not in (filtered,literal):
            raise ValueError(f"Content mismatch against protected commit: {name}")
        path = name.decode("utf-8")
        result[path] = sha(REPO/path)
    return result


def check_tree():
    baseline = ex.read(OUT / "protected_before.json")
    current = protected()
    result = dict(
        head=git("rev-parse", "HEAD"),
        tracked_changes=git("diff", "--name-only", "8380084"),
        staged_changes=git("diff", "--cached", "--name-only", "8380084"),
        tracked=len(baseline["tracked"]),
        protected_files=len(current),
        protected_match=current == baseline["protected"],
        firewall_sha256=sha(ROOT / "data.py"),
        git_status=git("status", "--short"),
    )
    result["passed"] = (
        result["head"] == baseline["head"]
        and not result["tracked_changes"]
        and not result["staged_changes"]
        and result["protected_match"]
        and result["firewall_sha256"] == baseline["firewall"]
    )
    return result


def suite(after):
    from p6 import iter5_audit, legacy_regression

    dest = ex.RESULTS / "checkout" if after else OUT / "checks"
    with patch.object(iter5_audit, "OUT", dest):
        p6_ok = iter5_audit.suite(None)
    if not after:
        legacy = legacy_regression.run()
        ex.save(dest / "legacy.json", legacy)
        tests = {k: v["run"] for k, v in legacy["suites"].items()}
        legacy_ok = legacy["passed"]
    else:
        import io, unittest

        tests, legacy = {}, {}
        for name in ("p5", "p5b"):
            cases = unittest.defaultTestLoader.discover(str(REPO / name / "tests"), top_level_dir=str(REPO))
            log = io.StringIO()
            outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(cases)
            tests[name] = outcome.testsRun
            legacy[name] = dict(passed=outcome.wasSuccessful() and not outcome.skipped, log=log.getvalue())
        ex.save(dest / "legacy.json", legacy)
        legacy_ok = all(r["passed"] for r in legacy.values())
    p4 = subprocess.run(
        [sys.executable, "-B", "-m", "publication.verify_p4"], cwd=REPO, capture_output=True, text=True
    )
    tree = check_tree()
    result = dict(
        p6_tests=ex.read(dest / "audit_suite.json")["run"],
        p6_passed=p6_ok,
        legacy_tests=tests,
        legacy_passed=legacy_ok,
        legacy_fixtures="archived" if after else "development adapter",
        p4_exit=p4.returncode,
        p4_stdout=p4.stdout,
        p4_stderr=p4.stderr,
        tree=tree,
    )
    result["passed"] = (
        p6_ok and tests == {"p5": 15, "p5b": 13} and legacy_ok and p4.returncode == 0 and tree["passed"]
    )
    if after:
        result["frozen_source_verification"] = verify_freeze()
        result["tracked_checkout_sha256_exact"] = tracked_inventory() == ex.read(FREEZE)["tracked_sha256"]
        result["passed"] &= result["tracked_checkout_sha256_exact"]
    ex.save(ex.RESULTS / "checkout_verification.json" if after else dest / "summary.json", result)
    return result["passed"]


def gate():
    recipe = ex.read(OUT / "final/recipe.json")
    records = [ex.read(path_for("selected", r, p, a)) for r in SNR for p in PAIRS for a in ex.ARMS]
    sources = training_sources()
    fair = all(
        r["recipe"] == recipe
        and r["source"] == sources
        and r["parameters"] == 856
        and r["anchor"]["p6_bit_exact"]
        and max(r["anchor"]["archived_max_llr_cfr"]) == 0
        and all(t["finite"] and t["active"] == t["size"] for t in r["gradients"].values())
        and sum(t["active"] for t in r["gradients"].values()) == 856
        for r in records
    )
    fair &= all(
        len({r["initial_sha"] for r in records if (r["regime"], r["pair"]) == (reg, p)}) == 1
        for reg in SNR
        for p in PAIRS
    )
    parity = [[r for r in records if r["regime"] == reg and r["arm"] == "parity"] for reg in SNR]
    counts = [sum(r["score"]["block_errors"] for r in rr) for rr in parity]
    blocks = [sum(r["score"]["blocks"] for r in rr) for rr in parity]
    repairs = sum(r["signs"]["repaired"] for r in parity[1])
    wrong_initial = sum(r["signs"]["initial_wrong"] for r in parity[1])
    regressions = sum(r["signs"]["regressed"] for r in parity[1])
    green = probe_summary(parity[2])["trained"]["parallel_joint"]["overall"]["mean"]
    uniform = ex.read(ex.RESULTS/"dev_iter5/dev_scores.json")["uniform_green"]["parallel_joint"]["mean"]
    audits = {}
    for name in ("old_gauge", "gauge"):
        rows = []
        for reg in SNR:
            for pair in PAIRS:
                for arm in GAUGED:
                    item = ex.read(path_for(name, reg, pair, arm))
                    if item["recipe"] != recipe or item["source"] != sources or len(item["rows"]) != 10:
                        raise ValueError("Unexpected gauge provenance or coverage")
                    expected_ck = (
                        path_for("selected", reg, pair, arm)
                        if name == "gauge"
                        else previous_location(reg, pair, arm, 48, 0.05, False)
                    )
                    if item["checkpoint_sha256"] != sha(expected_ck.with_suffix(".pt")):
                        raise ValueError("Gauge provenance differs from checkpoint")
                    rows.extend(item["rows"])
        a = summarize(rows)
        a["passed"] &= len({(v["regime"], v["pair"], v["arm"], v["condition"]) for v in rows}) == 360
        a["passed"] &= all(v["scales"] == list(SCALES) for v in rows)
        a["failed_cases"] = [
            {k: v[k] for k in ("regime", "pair", "arm", "condition", "max_grid_llr_diff", "coded_flips")}
            for v in rows
            if not v["passed"]
        ]
        audits[name] = a
    checks = ex.read(OUT / "checks/summary.json")
    result = dict(
        recipe=recipe,
        exact_anchors_and_live_gradients=fair,
        fit_count=len(records),
        counts=counts,
        blocks=blocks,
        repairs=repairs,
        initially_wrong=wrong_initial,
        sign_regressions=regressions,
        joint_green=green,
        uniform_green=uniform,
        audits=audits,
        regressions=checks,
    )
    result["passed"] = bool(
        fair
        and blocks == [384, 1728, 384]
        and all(x <= v for x, v in zip(counts, (25, 338, 49)))
        and repairs >= 9
        and wrong_initial == 10
        and green >= 0.34
        and green > uniform
        and all(a["passed"] for a in audits.values())
        and checks["passed"]
    )
    ex.save(OUT / "gate.json", result)
    return result["passed"]


def inputs():
    files = [
        p for p in ROOT.rglob("*.py") if not {"tmp", "__pycache__"}.intersection(p.relative_to(ROOT).parts)
    ]
    files.extend(
        [
            ROOT / "PROTOCOL.md",
            ROOT / "DEV_ITER5_MATH.md",
            OUT / "final/recipe.json",
            OUT / "gate.json",
            OUT / "protected_before.json",
            OUT / "chirp/R2_MATH.md",
        ]
    )
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def freeze():
    if FREEZE.exists() or (ex.RESULTS / "formal").exists():
        raise RuntimeError("Formal freeze or results already exist")
    if not gate() or not check_tree()["passed"]:
        raise RuntimeError("Development gate or protected inventory failed")
    data = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
                         logical_cpus=os.cpu_count(),torch_threads=torch.get_num_threads(),
                         torch_interop_threads=torch.get_num_interop_threads(),
                         deterministic=torch.are_deterministic_algorithms_enabled(),
                         environment={k:os.environ.get(k) for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","MKL_CBWR","ATEN_CPU_CAPABILITY")}),
        head=git("rev-parse", "HEAD"),
        recipe=ex.read(OUT / "final/recipe.json"),
        sources_and_evidence=inputs(),
        protected_sha256=protected(),
        tracked_sha256=tracked_inventory(),
        dev_gate=ex.read(OUT / "gate.json"),
    )
    with FREEZE.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return verify_freeze()


def verify_freeze():
    data = ex.read(FREEZE)
    if (
        not data["dev_gate"]["passed"]
        or data["sources_and_evidence"] != inputs()
        or git("rev-parse", "HEAD") != data["head"]
    ):
        raise RuntimeError("Frozen sources or development gate mismatch")
    return sha(FREEZE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("suite", "checkout", "gate", "freeze"))
    args = ap.parse_args()
    ex.setup()
    result = {
        "suite": lambda: suite(False),
        "checkout": lambda: suite(True),
        "gate": gate,
        "freeze": freeze,
    }[args.action]()
    print(args.action, result)
    sys.exit(0 if result else 1)
