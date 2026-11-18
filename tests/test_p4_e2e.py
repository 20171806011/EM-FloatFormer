"""
Master E2E Test Suite Aggregator for EM-FloatFormer Stage P4.

Aggregates all 4 test tiers and 253 test cases:
- Tier 1: Feature Coverage (110 tests, F1 to F22 happy paths)
- Tier 2: Boundary & Corner Cases (110 tests, F1 to F22 extreme conditions)
- Tier 3: Cross-Feature Combinations (22 pairwise interaction tests)
- Tier 4: Real-World Workload Scenarios (11 realistic end-to-end communication scenarios)

Primary execution command:
    python -m unittest tests/test_p4_e2e.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# Ensure torch CPU concurrency is standardized
torch.set_num_threads(4)

# Import tier test case classes
from tests.e2e.test_tier1_feature_coverage import TestTier1FeatureCoverage
from tests.e2e.test_tier2_boundary_corner import TestTier2BoundaryCorner
from tests.e2e.test_tier3_cross_feature import TestTier3CrossFeature
from tests.e2e.test_tier4_real_workloads import TestTier4RealWorkloads


def suite() -> unittest.TestSuite:
    """Build and return the complete 253-test suite."""
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestTier1FeatureCoverage))
    test_suite.addTests(loader.loadTestsFromTestCase(TestTier2BoundaryCorner))
    test_suite.addTests(loader.loadTestsFromTestCase(TestTier3CrossFeature))
    test_suite.addTests(loader.loadTestsFromTestCase(TestTier4RealWorkloads))
    return test_suite


def load_tests(loader, tests, pattern):
    """Unittest load_tests protocol hook for test discovery."""
    return suite()


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
