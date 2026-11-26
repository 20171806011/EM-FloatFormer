"""CPU regression for listed candidate prompts and strict generation metrics."""
import unittest

from replay_choices import parse_choice, prompts, summarize


CHOICES = ("符号0（1）", "符号1（i）", "符号2（-1）", "符号3（-i）")


class ReplayCheck(unittest.TestCase):
    def test_only_prompt_change_is_listing(self):
        variants = prompts("问题", CHOICES)
        listing = "四个选项如下。\n" + "\n".join(CHOICES) + "\n"
        self.assertEqual(variants["omitted"], variants["listed"].replace(listing, ""))
        for choice in CHOICES:
            self.assertEqual(variants["listed"].count(choice), 1)
            self.assertNotIn(choice, variants["omitted"])
        with self.assertRaises(ValueError):
            prompts("问题", [CHOICES[0]] * 4)

    def test_first_line_canonical_label_only(self):
        self.assertEqual(parse_choice(" **符号1(i)**\n说明", CHOICES), 1)
        for reply in ("", "0000", "i", "符号1(i) 或符号2(-1)", "未定\n符号1（i）"):
            self.assertIsNone(parse_choice(reply, CHOICES))

    def test_unparseable_in_denominator_and_paired_counts(self):
        def row(omitted, listed):
            return {"task": "q2", "condition": "c1", "true_index": 1, "original_score": 1,
                    "omitted": {"score_index": 0, "generated_index": omitted},
                    "listed": {"score_index": 1, "generated_index": listed}}
        summary = summarize([row(None, 1), row(1, None), row(0, 1)])["c1/q2"]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["omitted_parseable"], 2)
        self.assertEqual(summary["listed_generation_correct"], 2)
        self.assertEqual(summary["omitted_agreement"], 1)
        self.assertEqual(summary["generation_corrected"], 2)
        self.assertEqual(summary["generation_new_errors"], 1)


if __name__ == "__main__":
    unittest.main()
