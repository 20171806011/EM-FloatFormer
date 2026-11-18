from fractions import Fraction
import unittest

from p6.formal_report import holm, sign_row


class ExactPairedInference(unittest.TestCase):
    def test_sign_test_excludes_ties_and_counts_two_tails(self):
        def sign(values):
            return sign_row(1, "receiver", "control", dict(enumerate(values)))

        self.assertEqual(sign([0] * 10)["raw_exact"], "1")
        self.assertEqual(sign([1] * 10)["raw_exact"], "1/512")
        row = sign([1] * 8 + [0, -1])
        self.assertEqual((row["wins"], row["ties"], row["losses"]), (8, 1, 1))
        self.assertEqual(row["raw_exact"], "5/128")

    def test_holm_monotone_within_the_family(self):
        raw = [Fraction(1, 32), Fraction(1, 512), Fraction(1, 32), Fraction(1)]
        rows = holm([{"raw_exact": str(v)} for v in raw])
        self.assertEqual(
            [r["holm_exact"] for r in rows], ["3/32", "1/128", "3/32", "1"]
        )
