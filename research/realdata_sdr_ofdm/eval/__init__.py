"""Unified evaluation harness for research/realdata_sdr_ofdm."""

from research.realdata_sdr_ofdm.eval.evaluator import (
    compare_arms,
    compare_multiple_arms,
    evaluate_frames,
    evaluate_outputs,
    evaluate_paired,
    evaluate_receiver,
    paired_comparison,
    profile_receiver_pinned,
    run_stress_suite,
    subblock_group_ids,
)

__all__ = [
    "compare_arms",
    "compare_multiple_arms",
    "evaluate_frames",
    "evaluate_outputs",
    "evaluate_paired",
    "evaluate_receiver",
    "paired_comparison",
    "profile_receiver_pinned",
    "run_stress_suite",
    "subblock_group_ids",
]
