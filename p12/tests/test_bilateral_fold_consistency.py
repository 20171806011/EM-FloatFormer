"""Geometry-only regression tests for P12's advertised bilateral first waves."""
from __future__ import annotations

import unittest

import numpy as np

from p11.contract import OFFGRID_ANCHOR_ROWS, PROBE_ROW, T
from p12.offgrid_physics import make_roving_bilateral_3fold


class TestBilateralFoldConsistency(unittest.TestCase):
    def assert_valid_layout(self, anchor: int, probe: int) -> None:
        targets, aux = make_roving_bilateral_3fold(anchor, probe)
        data_rows = tuple(r for r in range(T) if r not in (anchor, probe))
        flat_targets = tuple(r for fold in targets for r in fold)
        flat_aux = tuple(r for fold in aux for r in fold)

        self.assertEqual(tuple(len(fold) for fold in targets), (6, 6, 6))
        self.assertEqual(len(flat_targets), len(set(flat_targets)))
        self.assertEqual(set(flat_targets), set(data_rows))
        self.assertEqual(tuple(len(fold) for fold in aux), (5, 5, 5))
        self.assertEqual(len(set(flat_aux)), 15)
        self.assertTrue(set(flat_aux).issubset(data_rows))
        # Target ownership must remain the original interleaved partition.
        self.assertEqual(targets, tuple(data_rows[f::3] for f in range(3)))

        for fold, (target_rows, aux_rows) in enumerate(zip(targets, aux)):
            with self.subTest(anchor=anchor, probe=probe, fold=fold):
                self.assertTrue(set(target_rows).isdisjoint(aux_rows))
                self.assertTrue(set(aux_rows).issubset(targets[(fold + 1) % 3]))
                forbidden = set(target_rows) | set(aux_rows) | {anchor, probe}
                # Mirror the receiver's public-row traversal and stable distance
                # sort, not the construction's retained-donor selection.
                helpers = sorted(
                    [r for r in data_rows if r not in forbidden],
                    key=lambda r: abs(r - anchor),
                )
                self.assertEqual(len(helpers), 7)
                self.assertTrue(set(helpers).isdisjoint(forbidden))
                self.assertTrue(any(r < anchor for r in helpers))
                self.assertTrue(any(r > anchor for r in helpers))
                self.assertLess(min(helpers[:2]), anchor)
                self.assertGreater(max(helpers[:2]), anchor)

    def test_all_distinct_anchor_probe_layouts(self) -> None:
        feasible = infeasible = 0
        for anchor in range(T):
            for probe in range(T):
                if anchor == probe:
                    continue
                with self.subTest(anchor=anchor, probe=probe):
                    data_rows = [r for r in range(T) if r not in (anchor, probe)]
                    n_left = sum(r < anchor for r in data_rows)
                    n_right = sum(r > anchor for r in data_rows)
                    if min(n_left, n_right) < 2:
                        with self.assertRaisesRegex(ValueError, "at least two data rows on each side"):
                            make_roving_bilateral_3fold(anchor, probe)
                        infeasible += 1
                    else:
                        self.assert_valid_layout(anchor, probe)
                        feasible += 1
        self.assertEqual((feasible, infeasible), (300, 80))

    def test_all_advertised_anchor_folds_are_deterministic(self) -> None:
        self.assertEqual(tuple(OFFGRID_ANCHOR_ROWS), (3, 6, 10, 13, 16))
        for anchor in OFFGRID_ANCHOR_ROWS:
            with self.subTest(anchor=anchor):
                self.assert_valid_layout(anchor, PROBE_ROW)
                self.assertEqual(
                    make_roving_bilateral_3fold(anchor, PROBE_ROW),
                    make_roving_bilateral_3fold(np.int64(anchor), np.int64(PROBE_ROW)),
                )

    def test_anchor16_fold1_retains_the_only_right_helper(self) -> None:
        targets, aux = make_roving_bilateral_3fold(16, 19)
        self.assertEqual(aux[1], (2, 5, 8, 11, 14))
        helpers = sorted(
            [r for r in range(T) if r not in set(targets[1]) | set(aux[1]) | {16, 19}],
            key=lambda r: abs(r - 16),
        )
        self.assertEqual(helpers[:2], [15, 18])

    def test_invalid_row_indices_fail_explicitly(self) -> None:
        for bad in (True, np.bool_(False), 3.0, 3.5, "3", None, -1, T):
            with self.subTest(argument="r_anchor", value=bad):
                with self.assertRaisesRegex(ValueError, "r_anchor"):
                    make_roving_bilateral_3fold(bad, PROBE_ROW)
            with self.subTest(argument="probe_row", value=bad):
                with self.assertRaisesRegex(ValueError, "probe_row"):
                    make_roving_bilateral_3fold(10, bad)
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            make_roving_bilateral_3fold(10, 10)


if __name__ == "__main__":
    unittest.main()
