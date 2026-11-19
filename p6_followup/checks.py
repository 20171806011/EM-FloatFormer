"""Record focused new tests and the unchanged P6 regression suite."""
import io
import subprocess
import unittest
from contextlib import ExitStack
import importlib
from unittest.mock import patch

from p6 import experiment as ex, diagnostics
from p6_followup.common import BASE, REPO, ROOT, RESULTS, environment, save, sha


def main():
    ex.setup()
    suites = {}
    for name in ("p6_followup", "p6"):
        stream = io.StringIO()
        tests = unittest.defaultTestLoader.discover(str(REPO / name / "tests"),
                                                  top_level_dir=str(REPO))
        # The inherited full audit writes a receipt. Redirect only that receipt
        # so a regression test cannot overwrite an archived P6 result.
        with ExitStack() as hooks:
            redirect = lambda path, value: save(RESULTS / "p6_regression_numeric.json", value)
            hooks.enter_context(patch.object(diagnostics, "save", redirect))
            if name == "p6":
                # The test imports save by value, separately from diagnostics.
                inherited_test = importlib.import_module("p6.tests.test_p6_invariants")
                hooks.enter_context(patch.object(inherited_test, "save", redirect))
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
        suites[name] = dict(tests=result.testsRun, errors=len(result.errors),
                            failures=len(result.failures), skipped=len(result.skipped),
                            passed=result.wasSuccessful() and not result.skipped, log=stream.getvalue())
    changed = subprocess.check_output(["git", "diff", BASE, "--name-only", "--",
                                       "p4", "p5", "p5b", "p6"], cwd=REPO, text=True).strip()
    files = [ROOT / "checks.py", *sorted((ROOT / "tests").glob("*.py"))]
    report = dict(environment=environment(), suites=suites,
                  source_sha256={p.relative_to(REPO).as_posix(): sha(p) for p in files},
                  inherited_changes=changed, passed=all(s["passed"] for s in suites.values()) and not changed)
    save(RESULTS / "tests.json", report)
    print({name: {k: v for k, v in value.items() if k != "log"} for name, value in suites.items()}, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
