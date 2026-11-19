"""Run new mathematical/integration tests and optionally the frozen P6 suite."""

import argparse
from contextlib import ExitStack
import importlib
import io
import re
import subprocess
import unittest
from unittest.mock import patch

from p6 import diagnostics, experiment as ex
from p6_softinfo.common import BASE, REPO, RESULTS, ROOT, environment, save, sha


def run_suite(tests, inherited=False):
    stream = io.StringIO()
    with ExitStack() as hooks:
        if inherited:
            redirect = lambda path, value: save(RESULTS / "p6_regression_numeric.json", value)
            hooks.enter_context(patch.object(diagnostics, "save", redirect))
            # This test imports save by value; both bindings must be redirected
            # to avoid overwriting the archived P6 numeric-audit receipt.
            imported = importlib.import_module("p6.tests.test_p6_invariants")
            hooks.enter_context(patch.object(imported, "save", redirect))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    return dict(tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                skipped=len(result.skipped), passed=result.wasSuccessful() and not result.skipped,
                log=stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inherited", action="store_true",
                        help="Also run the unchanged 38-test P6 regression suite")
    parser.add_argument("--attempt", help="Keep a named nonfinal attempt instead of tests.json")
    args = parser.parse_args()
    if args.attempt and re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt) is None:
        parser.error("Attempt labels may contain letters, digits, underscores and hyphens only")
    destination = RESULTS / (f"tests_{args.attempt}.json" if args.attempt else "tests.json")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite test evidence: {destination}")
    ex.setup()
    modules = sorted(f"p6_softinfo.{path.stem}" for path in ROOT.glob("test_*.py"))
    suites = {"p6_softinfo": run_suite(unittest.defaultTestLoader.loadTestsFromNames(modules))}
    if args.inherited:
        inherited = unittest.defaultTestLoader.discover(str(REPO / "p6" / "tests"),
                                                       top_level_dir=str(REPO))
        suites["p6"] = run_suite(inherited, inherited=True)
    inherited_changes = subprocess.check_output(
        ["git", "diff", BASE, "--name-only", "--", "p4", "p5", "p5b", "p6", "p6_followup"],
        cwd=REPO, text=True).strip()
    report = dict(environment=environment(), suites=suites, test_modules=modules,
                  source_sha256={path.relative_to(REPO).as_posix(): sha(path)
                                 for path in sorted(ROOT.glob("*.py"))},
                  inherited_changes=inherited_changes,
                  passed=all(value["passed"] for value in suites.values()) and not inherited_changes)
    save(destination, report)
    print({name: {key: value for key, value in suite.items() if key != "log"}
           for name, suite in suites.items()}, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
