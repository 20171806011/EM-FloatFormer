"""Run the unmodified 253-case suite in a disposable copy to protect signed timing artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    from tests.test_p4_e2e import suite
    from .confirmatory_protocol import _historical_digest
    paths = [path for name in ("p4", "tests") for path in (ROOT/name).rglob("*.py")]
    paths += list((ROOT/"p4/results").glob("*.pt"))
    paths += [ROOT/"PROJECT.md", ROOT/"p4/P4D_PROTOCOL.md", ROOT/"p4/p4cd_factorial_protocol.json"]
    sources = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(paths)}

    class EvidenceResult(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.successful = []

        def addSuccess(self, test):
            super().addSuccess(test)
            self.successful.append(test.id())

    results = EvidenceResult()
    start = time.monotonic()
    suite().run(results)
    return {"suite": "tests.test_p4_e2e.suite", "run": results.testsRun, "passed": len(results.successful),
            "passed_cases": results.successful, "failures": [(t.id(), msg) for t, msg in results.failures],
            "errors": [(t.id(), msg) for t, msg in results.errors],
            "skips": [(t.id(), why) for t, why in results.skipped], "seconds": time.monotonic()-start,
            "sources": sources, "historical_checkout": _historical_digest()}


def isolated(output):
    with tempfile.TemporaryDirectory(prefix="emfloatformer-verification-") as scratch:
        clone = Path(scratch, "repo")
        subprocess.run(["git", "clone", "--quiet", "--local", "--no-hardlinks", str(ROOT), str(clone)],
                       check=True)
        for name in ("p4", "tests"):
            shutil.copytree(ROOT/name, clone/name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.tmp", "*.lock"))
        for name in ("PROJECT.md", "TEST_INFRA.md", "TEST_READY.md"):
            shutil.copy2(ROOT/name, clone/name)
        destination = Path(scratch, "measured.json")
        subprocess.run([sys.executable, "-B", "-m", "p4.stage3_verify", "--inside", str(destination)],
                       cwd=clone, check=True)
        shutil.copyfile(destination, output)
        result = json.loads(output.read_bytes())
        print("Passed", result["passed"], "of", result["run"])
        for case, why in result["failures"] + result["errors"] + result["skips"]:
            print(case, why.strip().splitlines()[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"p4/results/stage3_e2e.json")
    args = parser.parse_args()
    if args.inside:
        args.inside.write_text(json.dumps(evidence(), indent=2, allow_nan=False), encoding="utf-8")
    else:
        isolated(args.output)
