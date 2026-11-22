"""
Master E2E Test Suite Aggregator for EM-FloatFormer Real-Data SDR OFDM (`F12`).

Aggregates all 4 test tiers and 132 test cases:
- Tier 1: Feature Coverage (55 tests, 5 per feature F1–F11)
- Tier 2: Boundary & Corner Cases (55 tests, 5 per feature F1–F11)
- Tier 3: Cross-Feature Combinations (15 pairwise tests T3-P01..T3-P15)
- Tier 4: Real-World Workload Scenarios (7 workload scenarios T4-S01..T4-S07)

Canonical commands:
    python -B -m unittest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
    python -B -m pytest research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
    python -B -m unittest -k "f01" research/realdata_sdr_ofdm/tests/test_realdata_e2e.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

from research.realdata_sdr_ofdm.tests.e2e.test_tier1_feature_coverage import TestTier1FeatureCoverage
from research.realdata_sdr_ofdm.tests.e2e.test_tier2_boundary_corner import TestTier2BoundaryCorner
from research.realdata_sdr_ofdm.tests.e2e.test_tier3_cross_feature import TestTier3CrossFeature
from research.realdata_sdr_ofdm.tests.e2e.test_tier4_real_workloads import TestTier4RealWorkloads


def suite(loader: unittest.TestLoader | None = None) -> unittest.TestSuite:
    """Build and return the complete 132-test E2E suite (honoring CLI `-k` loader filters)."""
    active_loader = loader or unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(active_loader.loadTestsFromTestCase(TestTier1FeatureCoverage))
    test_suite.addTests(active_loader.loadTestsFromTestCase(TestTier2BoundaryCorner))
    test_suite.addTests(active_loader.loadTestsFromTestCase(TestTier3CrossFeature))
    test_suite.addTests(active_loader.loadTestsFromTestCase(TestTier4RealWorkloads))
    return test_suite


def load_tests(loader, tests, pattern):
    """Unittest `load_tests` protocol hook for test discovery."""
    return suite(loader)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
