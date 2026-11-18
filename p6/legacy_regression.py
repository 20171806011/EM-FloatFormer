"""Run the archived test bodies with every waveform draw redirected to development.

The original fixtures use literal simulation seeds and P5-B pair 1. Patching
the module references in this process preserves every assertion, SNR and
batch size, while preventing the original generators from seeing those seeds.
"""
from contextlib import ExitStack
from io import StringIO
import sys
import unittest
from unittest.mock import patch

from p5.tests import test_p5_invariants, test_p5_physics
from p5b import diagnostics as old_diagnostics
from p5b.tests import test_p5b_invariants
from p6.data import dev_frame, dev_seed


def run():
    seed_roles = {4: (5, 900), 5: (5, 0), 11: (6, 0), 12: (5, 901), 13: (6, 901)}
    generated = {}
    draws = []

    def adapted_rng(regime, stream, pair, batch=0):
        if pair != 1 or stream not in seed_roles:
            raise ValueError("unrecognized archive fixture")
        role, offset = seed_roles[stream]
        seed = dev_seed(regime, role, 901, offset + batch)
        generated[seed] = (role, 901, offset + batch)
        return seed

    def adapted_simulate(regime, batch_size, snr_db, seed):
        if seed in generated:
            stream, pair, index = generated[seed]
        else:
            literal_seeds = (22, 54193, 54195, 54196, 55139, 67168, 83133,
                             96001, 96002, 96003, 96004, 128603, 882506)
            index = literal_seeds.index(seed)
            # The same archived fixture sometimes doubles as train and evaluation
            # in its original invariant. Both uses see the same dev fixture.
            stream, pair, index = 5, 902, 400 + index
        draws.append((regime, stream, pair, index))
        return dev_frame(regime, stream, pair, batch_size, index, snr_db)

    suites = {}
    with ExitStack() as stack:
        for module in (test_p5_physics, test_p5_invariants, test_p5b_invariants, old_diagnostics):
            stack.enter_context(patch.object(module, "simulate", adapted_simulate))
        stack.enter_context(patch.object(test_p5b_invariants, "rng", adapted_rng))
        for name, modules in [("p5", [test_p5_physics, test_p5_invariants]),
                              ("p5b", [test_p5b_invariants])]:
            cases = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
            log = StringIO()
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(cases)
            suites[name] = dict(run=result.testsRun, passed=result.wasSuccessful(), log=log.getvalue())
            print(log.getvalue(), flush=True)
    return dict(suites=suites, unique_frame_draws=sorted(set(draws)),
                passed=all(s["passed"] for s in suites.values()))


if __name__ == "__main__":
    from pathlib import Path
    from p5.study import save
    result = run()
    save(Path(__file__).parent / "results" / "legacy_dev.json", result)
    sys.exit(0 if result["passed"] else 1)
