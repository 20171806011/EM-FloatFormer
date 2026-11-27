"""Strictly causal Pluto & CTTC SDR split validators and `m5-c16-v1-le` dual-cache exporters."""

from __future__ import annotations

import ast
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from research.stage4_real_and_surgical.packet_c16 import WIRE_ID, decode_c16, encode_c16

PLUTO_BRANCH = "origin/research/realdata-pluto-qpsk"
CTTC_BRANCH = "origin/research/realdata-cttc-indoor-qpsk"

PLUTO_MANIFEST: Dict[str, Any] = {
    "splits": {
        "train": {"warmup_start": 0, "eval_start": 200, "eval_end": 3200},
        "dev": {"warmup_start": 3300, "eval_start": 3500, "eval_end": 4500},
        "test": {"warmup_start": 4600, "eval_start": 4800, "eval_end": 7040},
    },
    "eval_counts": {"train": 3000, "dev": 1000, "test": 2240},
    "frame_guard": 100,
    "test_clusters": 16,
    "frames_per_test_cluster": 140,
    "stress_rows": 980,
    "refs": (15, 18),
    "holdout_slice": (21, 26),
    "payload_slice": (26, 126),
    "sync_slice": (0, 13),
    "pilot_slice": (13, 21),
    "sps": 20,
    "n_sym": 126,
    "maximum_cubic_union_samples": 8,
}

CTTC_MANIFEST: Dict[str, Any] = {
    "split_source_ranges": {
        "train": {"target": (0, 160_000), "ref": (164_096, 200_000)},
        "dev": {"target": (204_096, 290_000), "ref": (294_096, 330_000)},
        "test": {"target": (334_096, 455_000), "ref": (459_096, 495_904)},
    },
    "cyclic_circumference": 500_000,
    "cyclic_symbol_guard": 4096,
    "counts": {
        "train_target": 1440,
        "train_ref": 576,
        "dev_target": 864,
        "dev_ref": 576,
        "test_target": 1440,
        "test_ref": 576,
    },
    "test_session_loop_clusters": 72,
    "repeat_disjoint_loop_pairs": 36,
    "repeat_candidate_rows": 720,
    "stress_rows": 1080,
    "minimum_acquisition_gap_samples_after_rcv_support_expansion": 16481,
    "refs": (126, 127),
    "sync_idx": tuple(range(0, 6)),
    "pilot_idx": (126, 127, 254, 255),
    "holdout_idx": tuple(list(range(40, 48)) + list(range(200, 208))),
    "holdout_capture": "f708_snr10_QPSK",
    "sps": 4,
    "pkt_len": 256,
    "maximum_cubic_union_samples": 6,
}


def _guarded_gap(ranges: Sequence[Tuple[int, int]], circumference: int | None = None) -> int:
    ordered = sorted(ranges)
    gaps = [b[0] - a[1] for a, b in zip(ordered, ordered[1:])]
    if circumference is not None:
        gaps.append(ordered[0][0] + circumference - ordered[-1][1])
    return min(gaps)


def validate_pluto_splits(manifest: Dict[str, Any] = PLUTO_MANIFEST) -> dict:
    """Verify strict disjointness and >= 100-frame guard bands across Pluto splits."""
    sp = manifest["splits"]
    spans = [(v["warmup_start"], v["eval_end"]) for v in sp.values()]
    min_gap = _guarded_gap(spans)
    assert min_gap >= manifest["frame_guard"], f"Pluto frame guard violated: {min_gap}"
    for name, s in sp.items():
        assert s["eval_start"] >= s["warmup_start"]
        count = s["eval_end"] - s["eval_start"]
        assert count == manifest["eval_counts"][name]
    assert manifest["test_clusters"] * manifest["frames_per_test_cluster"] == manifest["eval_counts"]["test"]
    # Confirm refs strictly precede holdout & payload
    assert max(manifest["refs"]) < manifest["holdout_slice"][0] < manifest["payload_slice"][0]
    return {"min_gap_frames": min_gap, "test_total": manifest["eval_counts"]["test"], "ok": True}


def validate_cttc_splits(manifest: Dict[str, Any] = CTTC_MANIFEST) -> dict:
    """Verify strict cyclic 4,096-symbol disjointness across CTTC train/dev/test target & ref ranges."""
    sr = manifest["split_source_ranges"]
    all_ranges = [r for split_dict in sr.values() for r in split_dict.values()]
    min_gap = _guarded_gap(all_ranges, circumference=manifest["cyclic_circumference"])
    assert min_gap >= manifest["cyclic_symbol_guard"], f"CTTC cyclic guard violated: {min_gap}"
    # Verify known refs are strictly disjoint from holdout symbols
    assert set(manifest["refs"]).isdisjoint(set(manifest["holdout_idx"]))
    return {"min_cyclic_gap_symbols": min_gap, "holdout_capture": manifest["holdout_capture"], "ok": True}


def _git_show_file(repo_root: Path, rev_path: str) -> str:
    return subprocess.check_output(["git", "show", rev_path], cwd=str(repo_root), text=True, encoding="utf-8")


def validate_branch_literal_parity(repo_root: Path) -> dict:
    """Cross-check live remote branch constants in realdata_pluto_qpsk & realdata_cttc_qpsk without mutating trees."""
    cttc_src = _git_show_file(repo_root, f"{CTTC_BRANCH}:research/realdata_cttc_qpsk/adapter.py")
    pluto_src = _git_show_file(repo_root, f"{PLUTO_BRANCH}:research/realdata_pluto_qpsk/waveform_frontend.py")
    assert "4,096" in cttc_src or "4096" in cttc_src
    assert "(0, 160_000)" in cttc_src and "(334_096, 455_000)" in cttc_src
    assert "N_SYM = 126" in pluto_src and "SPS = 20" in pluto_src
    return {"cttc_branch_checked": True, "pluto_branch_checked": True}


def causal_matched_sum_and_residual(iq_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute canonical 3-sample matched sum `t` and quadratic residual `residual` per window."""
    arr = np.asarray(iq_windows, dtype=np.complex128)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 samples per window, got shape {arr.shape}")
    t = arr.sum(axis=-1) / math.sqrt(3.0)
    residual = np.sum(np.abs(arr - (t[..., None] / math.sqrt(3.0))) ** 2, axis=-1)
    return t, residual


def export_dual_cache_c16(
    iq_windows: np.ndarray,
    encoder,
    noise_variance: np.ndarray | float,
    available: np.ndarray | bool = True,
) -> Dict[str, Any]:
    """Encode raw 3-sample complex windows into bit-exact `s17` + `m5-c16-v1-le` 16-bit dual cache."""
    arr = np.asarray(iq_windows, dtype=np.complex128)
    orig_shape = arr.shape[:-1]
    flat = arr.reshape(-1, 3)
    s17_codes = np.asarray(encoder.encode(flat), dtype=np.int64).reshape(orig_shape)
    t, res = causal_matched_sum_and_residual(arr)
    words_c16 = encode_c16(s17_codes, t, res, noise_variance, available=available)
    dec = decode_c16(words_c16)
    avail_arr = np.broadcast_to(np.asarray(available, dtype=bool), orig_shape)
    # Bit-exact S17 invariant whenever available
    assert np.array_equal(dec["s17"][avail_arr], s17_codes[avail_arr])
    return {
        "wire_id": WIRE_ID,
        "s17_codes": s17_codes,
        "words_c16": words_c16,
        "matched_sum": t,
        "residual_energy": res,
    }
