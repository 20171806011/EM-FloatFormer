"""Offline-capable unit tests for M1 modulation, SDR OFDM adapter, stress hooks, and leak-free splits."""

from __future__ import annotations

import io
import json
import pickle
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from research.realdata_sdr_ofdm.adapters.modulation import (
    bits_to_symbols,
    constellation_bits,
    constellation_log_likelihood,
    constellation_points,
    exact_llr,
    mapping_table,
    maxlog_llr,
    symbols_to_hard_bits,
)
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    N_ACTIVE,
    N_DATA_RES,
    N_FFT,
    N_SYMB,
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    PILOT_VALUES,
    SAFE_GUARD_FFT_INDICES,
    CustomDataset,
    RealOFDMFrame,
    _CompatUnpickler,
    active_rx_grid,
    apply_stress_perturbation,
    build_frame,
    build_heldout_pilot_masks,
    build_leak_free_splits,
    build_masks,
    estimate_blind_n0,
    load_processed_split,
    load_raw_sdr_ofdm,
    pilot_cv_masks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = REPO_ROOT / "research" / "realdata_sdr_ofdm"
MANIFESTS_DIR = RESEARCH_DIR / "manifests"
DEFAULT_CONFIG_PATH = RESEARCH_DIR / "configs" / "default.yaml"
SPLITS_PATH = MANIFESTS_DIR / "splits.json"
PROC_DIR = Path("/tmp/tw_realdata/data_proc")


def _make_synthetic_raw(
    seed: int = 123,
    *,
    noise_var: float = 0.01,
    zero_guards: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, 4), dtype=np.uint8)
    tx_syms = bits_to_symbols(tx_bits, qm=4)

    pilot_mask, dc_mask, data_mask = build_masks()
    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[data_mask] = tx_syms
    tx_grid[pilot_mask] = PILOT_VALUES

    # Smooth frequency-selective channel across active band
    k_norm = np.linspace(-1.0, 1.0, N_FFT)
    h_freq = 0.85 * np.exp(-1j * 0.8 * k_norm) + 0.15 * np.cos(1.7 * k_norm)
    rx_grid = tx_grid * h_freq[None, :]

    if noise_var > 0.0:
        noise = np.sqrt(noise_var / 2.0) * (
            rng.standard_normal((N_SYMB, N_FFT)) + 1j * rng.standard_normal((N_SYMB, N_FFT))
        )
        rx_grid = rx_grid + noise

    if zero_guards:
        rx_grid[:, :13] = 0.0
        rx_grid[:, 115:] = 0.0
    rx_grid[dc_mask] = 0.05 + 0.02j
    # Replicate upstream max-abs normalization
    rx_grid = rx_grid / np.max(np.abs(rx_grid))
    return rx_grid, tx_bits, 15.0


def test_mapping_table_4_nongray_roundtrip() -> None:
    pts = constellation_points(4)
    bits = constellation_bits(4)
    assert len(pts) == 16
    assert abs(float(np.mean(np.abs(pts) ** 2)) - 1.0) < 1e-12

    mapping, demapping = mapping_table(4)
    assert len(mapping) == 16 and len(demapping) == 16

    # Check exact non-Gray natural binary formula: (b0, b1) -> Re, (b2, b3) -> Im
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / np.sqrt(10.0)
    for b0 in range(2):
        for b1 in range(2):
            for b2 in range(2):
                for b3 in range(2):
                    expected = levels[2 * b0 + b1] + 1j * levels[2 * b2 + b3]
                    assert abs(mapping[(b0, b1, b2, b3)] - expected) < 1e-12

    # Non-Gray property: adjacent Re symbols (0,1,0,0) and (1,0,0,0) differ by 2 bits
    s_0100 = mapping[(0, 1, 0, 0)]
    s_1000 = mapping[(1, 0, 0, 0)]
    assert abs(s_0100.imag - s_1000.imag) < 1e-12
    assert abs((s_1000.real - s_0100.real) - 2.0 / np.sqrt(10.0)) < 1e-12

    for qm in (2, 4):
        b_all = constellation_bits(qm)
        s_all = bits_to_symbols(b_all, qm=qm)
        b_rt = symbols_to_hard_bits(s_all, qm=qm)
        assert np.array_equal(b_rt, b_all)

    # Exact LLR and MaxLog LLR agreement vs brute-force 16-point formula, plus sign convention (bit 1 > 0)
    rng = np.random.default_rng(7)
    z = (rng.standard_normal(50) + 1j * rng.standard_normal(50)) * 0.8
    n0_vec = rng.uniform(0.05, 0.5, size=50)
    llr_ex = exact_llr(z, n0_vec, qm=4)
    llr_ml = maxlog_llr(z, n0_vec, qm=4)
    ll = constellation_log_likelihood(z, n0_vec, qm=4, include_norm=True)
    assert ll.shape == (50, 16)
    assert llr_ex.shape == (50, 4)
    assert llr_ml.shape == (50, 4)

    for idx in range(50):
        # Verify Gaussian density formula
        dists = np.abs(z[idx] - pts) ** 2
        expected_ll = -np.log(np.pi * n0_vec[idx]) - dists / n0_vec[idx]
        assert np.allclose(ll[idx], expected_ll, atol=1e-12, rtol=1e-12)
        probs = np.exp(expected_ll)
        for k in range(4):
            p1 = np.sum(probs[bits[:, k] == 1])
            p0 = np.sum(probs[bits[:, k] == 0])
            assert abs(llr_ex[idx, k] - np.log(p1 / p0)) < 1e-11
            ml_ref = np.max(expected_ll[bits[:, k] == 1]) - np.max(expected_ll[bits[:, k] == 0])
            assert abs(llr_ml[idx, k] - ml_ref) < 1e-12

    # Sign convention check on noiseless symbol (1, 0, 1, 0)
    z_sym = bits_to_symbols(np.array([[1, 0, 1, 0]], dtype=np.uint8), qm=4)
    llr_clean = exact_llr(z_sym, 0.05, qm=4)[0]
    assert llr_clean[0] > 0 and llr_clean[1] < 0 and llr_clean[2] > 0 and llr_clean[3] < 0

    # Broadcast (14, 102) and numerical stability at extreme n0 & |z|
    z_2d = np.full((14, 102), 100.0 + 100.0j, dtype=np.complex128)
    n0_2d = np.full((14, 102), 1e-12, dtype=np.float64)
    llr_ext = exact_llr(z_2d, n0_2d, qm=4)
    assert llr_ext.shape == (14, 102, 4)
    assert np.all(np.isfinite(llr_ext))


def test_adapter_geometry_and_masks() -> None:
    pilot_mask, dc_mask, data_mask = build_masks()
    guard_mask = ~(pilot_mask | dc_mask | data_mask)

    assert pilot_mask.shape == (14, 128)
    assert pilot_mask.sum() == 14
    assert np.all(pilot_mask[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)])
    assert dc_mask.sum() == 14
    assert np.all(dc_mask[:, 64])
    assert data_mask.sum() == 1400
    assert guard_mask.sum() == 364

    assert not np.any(pilot_mask & dc_mask)
    assert not np.any(pilot_mask & data_mask)
    assert not np.any(dc_mask & data_mask)
    assert np.all((pilot_mask | dc_mask | data_mask | guard_mask))
    assert not np.any((pilot_mask | dc_mask | data_mask)[:, :13])
    assert not np.any((pilot_mask | dc_mask | data_mask)[:, 115:])

    iq, bits, sinr = _make_synthetic_raw(seed=11)
    fr = build_frame(frame_id=5, iq_raw=iq, labels_raw=bits, sinr_val=sinr, split="train", group_id="block_00")
    assert isinstance(fr, RealOFDMFrame)
    for attr in (
        "frame_id",
        "split",
        "group_id",
        "rx_grid",
        "pilot_mask",
        "dc_mask",
        "data_mask",
        "pilot_values",
        "tx_bits",
        "tx_symbols",
        "reported_sinr_db",
        "est_n0",
        "meta",
    ):
        assert hasattr(fr, attr)
    assert active_rx_grid(fr).shape == (14, N_ACTIVE)

    # Held-out pilot CV covers interior [1..12] exactly once, endpoints 0,13 always observed
    folds = build_heldout_pilot_masks(4)
    covered = []
    for f in folds:
        obs_m, held_m = pilot_cv_masks(pilot_mask, f["heldout_pilot_indices"])
        assert not np.any(obs_m & held_m)
        assert np.array_equal(obs_m | held_m, pilot_mask)
        assert obs_m[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[0]]
        assert obs_m[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[13]]
        covered.extend(f["heldout_pilot_indices"])
    assert sorted(covered) == list(range(1, 13))


def test_rms_gauge_and_pilot_correlation() -> None:
    iq, bits, sinr = _make_synthetic_raw(seed=22, noise_var=0.005)
    fr1 = build_frame(0, iq, bits, sinr)
    iq_scaled = iq * (3.7 * np.exp(1j * 0.9))
    fr2 = build_frame(0, iq_scaled, bits, sinr)

    act1 = fr1.rx_grid[fr1.pilot_mask | fr1.data_mask]
    act2 = fr2.rx_grid[fr2.pilot_mask | fr2.data_mask]
    assert abs(float(np.mean(np.abs(act1) ** 2)) - 1.0) < 1e-12
    assert abs(float(np.mean(np.abs(act2) ** 2)) - 1.0) < 1e-12
    assert abs(fr2.meta["s0"] / fr1.meta["s0"] - 3.7) < 1e-10

    h_p1 = fr1.rx_grid[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)] / PILOT_VALUES
    h_p2 = fr2.rx_grid[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)] / PILOT_VALUES
    coh1 = np.abs(np.sum(h_p1[:-1] * np.conj(h_p1[1:]))) / np.sum(np.abs(h_p1) ** 2)
    coh2 = np.abs(np.sum(h_p2[:-1] * np.conj(h_p2[1:]))) / np.sum(np.abs(h_p2) ** 2)
    assert abs(coh1 - coh2) < 1e-12
    assert coh1 > 0.85


def test_blind_n0_positive_and_label_invariant() -> None:
    iq_low, bits, sinr = _make_synthetic_raw(seed=33, noise_var=0.002)
    iq_high, _, _ = _make_synthetic_raw(seed=33, noise_var=0.08)
    fr_low = build_frame(0, iq_low, bits, sinr)
    fr_high = build_frame(0, iq_high, bits, sinr)
    assert fr_low.est_n0 > 0.0
    assert fr_high.est_n0 > fr_low.est_n0

    # Zeroed guards should trigger pilot_diff2 fallback cleanly
    iq_zg, bits_zg, sinr_zg = _make_synthetic_raw(seed=34, noise_var=0.02, zero_guards=True)
    fr_zg = build_frame(0, iq_zg, bits_zg, sinr_zg)
    assert fr_zg.est_n0 > 0.0
    assert fr_zg.meta["n0_source"] == "pilot_diff2_fallback"

    # Zero label leakage check: tampering tx_bits must NOT change s0 or est_n0
    bits_tampered = 1 - bits
    fr_tamp = build_frame(0, iq_low, bits_tampered, sinr)
    assert fr_tamp.meta["s0"] == fr_low.meta["s0"]
    assert fr_tamp.est_n0 == fr_low.est_n0
    assert np.allclose(fr_tamp.rx_grid, fr_low.rx_grid, atol=0.0, rtol=0.0)


def test_stress_perturbation_invariance() -> None:
    iq, bits, sinr = _make_synthetic_raw(seed=44, noise_var=0.0)
    fr = build_frame(0, iq, bits, sinr)
    rx_orig = fr.rx_grid.copy()
    bits_orig = fr.tx_bits.copy()
    syms_orig = fr.tx_symbols.copy()

    # Zero perturbation exact round-trip
    fr_zero = apply_stress_perturbation(fr, extra_awgn_db=None, cfo_norm=0.0, phase_drift_rad=0.0, seed=42)
    assert np.max(np.abs(fr_zero.rx_grid - fr.rx_grid)) < 1e-12
    assert np.array_equal(fr_zero.tx_bits, bits_orig)
    assert np.allclose(fr_zero.tx_symbols, syms_orig, atol=0.0, rtol=0.0)

    # Parseval AWGN check: extra_awgn_db=10.0 adds sigma2_add = 0.10 to guard est_n0
    fr_awgn1 = apply_stress_perturbation(fr, extra_awgn_db=10.0, seed=42)
    fr_awgn2 = apply_stress_perturbation(fr, extra_awgn_db=10.0, seed=42)
    assert np.array_equal(fr_awgn1.rx_grid, fr_awgn2.rx_grid)
    assert abs((fr_awgn1.est_n0 - fr.est_n0) - 0.10) < 0.035

    # CFO + phase drift leaves tx_bits/tx_symbols invariant and does not mutate input frame
    fr_cfo = apply_stress_perturbation(fr, extra_awgn_db=12.0, cfo_norm=0.03, phase_drift_rad=0.4, seed=9)
    assert np.array_equal(fr.rx_grid, rx_orig)
    assert np.array_equal(fr_cfo.tx_bits, bits_orig)
    assert np.allclose(fr_cfo.tx_symbols, syms_orig, atol=0.0, rtol=0.0)
    assert not np.allclose(fr_cfo.rx_grid, rx_orig)

    # Confirm safe guard indices used by estimate_blind_n0 are populated
    n0_val, info = estimate_blind_n0(fr_awgn1.rx_grid)
    assert len(SAFE_GUARD_FFT_INDICES) == 4
    assert n0_val == fr_awgn1.est_n0
    assert info["n0_source"].startswith("guard_passband")


def test_split_disjointness_guard_and_zero_leakage() -> None:
    # Synthetic 180-frame sequence with correlated SINR/channel and injected cross-block duplicate payloads
    n = 180
    rng = np.random.default_rng(99)
    sinr = 15.0 + 5.0 * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 0.3, size=n)
    # Create autocorrelated h_pilots
    h = np.zeros((n, 14), dtype=np.complex128)
    state = np.ones(14, dtype=np.complex128)
    for i in range(n):
        state = 0.8 * state + 0.2 * (rng.standard_normal(14) + 1j * rng.standard_normal(14))
        h[i] = state
    hashes = [f"hash_{i:04d}" for i in range(n)]
    # Inject duplicate payloads spanning opposite ends of dataset
    hashes[150] = hashes[5]
    hashes[100] = hashes[10]

    sp = build_leak_free_splits(
        sinr_vals=sinr,
        h_pilots=h,
        payload_hashes=hashes,
        dataset_sha256="synthetic_sha",
        n_macro_blocks=12,
        min_guard_frames=4,
        max_guard_frames=16,
    )
    tr, dv, te, gd = set(sp["train_indices"]), set(sp["dev_indices"]), set(sp["test_indices"]), set(sp["guard_indices"])
    assert tr.isdisjoint(dv) and tr.isdisjoint(te) and tr.isdisjoint(gd)
    assert dv.isdisjoint(te) and dv.isdisjoint(gd) and te.isdisjoint(gd)
    assert (tr | dv | te | gd) == set(range(n))
    assert len({hashes[i] for i in tr} & {hashes[i] for i in dv}) == 0
    assert len({hashes[i] for i in tr} & {hashes[i] for i in te}) == 0
    assert len({hashes[i] for i in dv} & {hashes[i] for i in te}) == 0
    assert sp["diagnostics"]["min_cross_split_distance"] >= sp["diagnostics"]["guard_width"] + 1


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="manifests/splits.json not generated yet")
def test_real_splits_manifest_invariants() -> None:
    sp = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    tr = set(sp["train_indices"])
    dv = set(sp["dev_indices"])
    te = set(sp["test_indices"])
    gd = set(sp["guard_indices"])
    n_total = sp["counts"]["total"]
    assert n_total >= 200
    assert tr.isdisjoint(dv) and tr.isdisjoint(te) and tr.isdisjoint(gd)
    assert dv.isdisjoint(te) and dv.isdisjoint(gd) and te.isdisjoint(gd)
    assert (tr | dv | te | gd) == set(range(n_total))
    assert len(tr) > 0 and len(dv) > 0 and len(te) > 0
    assert sp["diagnostics"]["min_cross_split_distance"] >= sp["diagnostics"]["guard_width"] + 1


@pytest.mark.skipif(not (PROC_DIR / "test.npz").exists(), reason="Codespace processed split not present locally")
def test_codespace_processed_test_split_integrity() -> None:
    sp = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    frames = load_processed_split(PROC_DIR, split="test")
    assert len(frames) == len(sp["test_indices"])
    for fr in frames[:10]:
        assert fr.split == "test"
        assert fr.rx_grid.shape == (14, 128)
        assert fr.tx_bits.shape == (1400, 4)
        assert np.all(np.isfinite(fr.rx_grid))
        assert fr.est_n0 > 0.0
        assert np.allclose(fr.tx_symbols, bits_to_symbols(fr.tx_bits, qm=4))


def _write_synthetic_processed_npz(npz_path: Path, n_frames: int = 40, seed: int = 2026) -> None:
    """Write a synthetic processed split .npz matching preprocess_and_save._save_subset schema."""
    rng = np.random.default_rng(seed)
    rx_grids = (
        rng.standard_normal((n_frames, N_SYMB, N_FFT))
        + 1j * rng.standard_normal((n_frames, N_SYMB, N_FFT))
    ).astype(np.complex128)
    tx_bits = rng.integers(0, 2, size=(n_frames, N_DATA_RES, 4), dtype=np.uint8)
    tx_symbols = bits_to_symbols(tx_bits.reshape(-1, 4), qm=4).reshape(n_frames, N_DATA_RES)
    meta_str = json.dumps({"s0": 1.0, "n0_source": "guard_passband_mean"})
    np.savez(
        npz_path,
        frame_ids=np.arange(n_frames, dtype=np.int64),
        splits=np.array(["train"] * n_frames),
        group_ids=np.array(["block_00"] * n_frames),
        rx_grids=rx_grids,
        tx_bits=tx_bits,
        tx_symbols=tx_symbols,
        reported_sinr_db=np.full(n_frames, 15.0, dtype=np.float64),
        est_n0=np.full(n_frames, 0.05, dtype=np.float64),
        s0=np.ones(n_frames, dtype=np.float64),
        meta_json=np.array([meta_str] * n_frames),
    )


def test_manifests_strict_utf8_no_bom() -> None:
    """All manifests/*.json (and unit_metadata.jsonl if present) must parse strictly as UTF-8 without BOM or CRLF."""
    json_files = sorted(MANIFESTS_DIR.glob("*.json"))
    assert len(json_files) >= 3
    for p in json_files:
        raw_bytes = p.read_bytes()
        assert not raw_bytes.startswith(b"\xef\xbb\xbf"), f"Unexpected UTF-8 BOM in {p}"
        assert b"\r\n" not in raw_bytes, f"Unexpected CRLF in {p}"
        parsed = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and len(parsed) > 0

    jsonl_path = MANIFESTS_DIR / "unit_metadata.jsonl"
    if jsonl_path.exists():
        raw_jsonl = jsonl_path.read_bytes()
        assert not raw_jsonl.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" not in raw_jsonl
        for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                rec = json.loads(line)
                assert "frame_id" in rec, f"Missing frame_id at line {line_no}"


def test_default_config_stress_extra_awgn_ordering() -> None:
    """configs/default.yaml stress.extra_awgn_db must start with None (clean) followed by descending SNR."""
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert (
        cfg["dataset"]["expected_sha256"]
        == "e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058"
    )
    awgn_levels = cfg["stress"]["extra_awgn_db"]
    assert awgn_levels[0] is None
    assert awgn_levels == [None, 10.0, 5.0, 0.0]


def test_load_processed_split_no_view_aliasing_and_bounded_memory_synthetic(tmp_path: Path) -> None:
    """Synthetic .npz regression test: load_processed_split copies slices (base is None) with bounded RAM."""
    n_frames = 40
    _write_synthetic_processed_npz(tmp_path / "train.npz", n_frames=n_frames)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    try:
        frames = load_processed_split(tmp_path, split="train")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert len(frames) == n_frames
    for fr in frames:
        assert fr.rx_grid.base is None and fr.rx_grid.flags["OWNDATA"]
        assert fr.tx_bits.base is None and fr.tx_bits.flags["OWNDATA"]
        assert fr.tx_symbols.base is None and fr.tx_symbols.flags["OWNDATA"]

    peak_mb = peak_bytes / (1024.0 * 1024.0)
    # 40 frames are ~2.27 MB raw; old O(N^2) view leak pins ~90.7 MB vs ~5 MB when fixed
    assert peak_mb < 20.0, f"Synthetic load_processed_split peak too high: {peak_mb:.2f} MB"
    assert load_processed_split(tmp_path, split="train", max_frames=0) == []


@pytest.mark.skipif(not (PROC_DIR / "all_frames.npz").exists(), reason="Codespace processed splits not present locally")
def test_codespace_processed_splits_memory_and_no_view_aliasing() -> None:
    """Codespace regression: train+dev+test+all_frames load with base is None and tracemalloc peak < 150 MB."""
    sp = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    try:
        tr = load_processed_split(PROC_DIR, split="train")
        dv = load_processed_split(PROC_DIR, split="dev")
        te = load_processed_split(PROC_DIR, split="test")
        al = load_processed_split(PROC_DIR, split="all_frames")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert len(tr) == len(sp["train_indices"])
    assert len(dv) == len(sp["dev_indices"])
    assert len(te) == len(sp["test_indices"])
    assert len(al) == sp["counts"]["total"]

    for split_frames in (tr, dv, te, al):
        for fr in split_frames:
            assert fr.rx_grid.base is None
            assert fr.tx_bits.base is None
            assert fr.tx_symbols.base is None

    peak_mb = peak_bytes / (1024.0 * 1024.0)
    assert peak_mb < 150.0, f"Codespace load_processed_split tracemalloc peak={peak_mb:.2f} MB >= 150 MB"


def test_validation_edge_cases_and_deepcopy_meta(tmp_path: Path) -> None:
    """Validate n0 < 0, NaN/Inf in blind_n0/build_frame, max_frames < 0, and deepcopy meta isolation."""
    # 1. Negative n0 raises ValueError across modulation LLRs, while n0 == 0.0 stays valid and finite
    z = np.array([0.3 + 0.3j, -0.9 - 0.9j], dtype=np.complex128)
    for bad_n0 in (-1e-6, -10.0, np.array([0.1, -0.01], dtype=np.float64)):
        with pytest.raises(ValueError):
            constellation_log_likelihood(z, bad_n0, qm=4)
        with pytest.raises(ValueError):
            exact_llr(z, bad_n0, qm=4)
        with pytest.raises(ValueError):
            maxlog_llr(z, bad_n0, qm=4)
    assert np.all(np.isfinite(exact_llr(z, 0.0, qm=4)))
    assert np.allclose(exact_llr(z, 0.0, qm=4), exact_llr(z, 1e-12, qm=4))

    # 2. NaN / Inf in estimate_blind_n0 and build_frame raise ValueError
    iq, bits, sinr = _make_synthetic_raw(seed=77)
    for bad_val in (complex(float("nan"), 0.0), complex(float("inf"), 0.0)):
        iq_bad = iq.copy()
        iq_bad[0, 10] = bad_val
        with pytest.raises(ValueError):
            estimate_blind_n0(iq_bad)
        with pytest.raises(ValueError):
            build_frame(0, iq_bad, bits, sinr)
    with pytest.raises(ValueError):
        build_frame(0, iq, bits, float("nan"))

    # 3. max_frames < 0 (or bool) raises ValueError in load_processed_split and load_raw_sdr_ofdm
    _write_synthetic_processed_npz(tmp_path / "test.npz", n_frames=2)
    for bad_max in (-1, -5, True):
        with pytest.raises(ValueError):
            load_processed_split(tmp_path, split="test", max_frames=bad_max)

    empty_pth = tmp_path / "empty.pth"
    torch.save(CustomDataset(), empty_pth)
    for bad_max in (-1, -5, False):
        with pytest.raises(ValueError):
            load_raw_sdr_ofdm(empty_pth, max_frames=bad_max)

    # 4. apply_stress_perturbation deepcopies meta (no shared heldout_pilot_folds aliasing)
    fr = build_frame(0, iq, bits, sinr)
    fr_p = apply_stress_perturbation(fr, extra_awgn_db=10.0, seed=123)
    assert fr_p.meta is not fr.meta
    assert fr_p.meta["heldout_pilot_folds"] is not fr.meta["heldout_pilot_folds"]
    assert fr_p.meta["heldout_pilot_folds"][0] is not fr.meta["heldout_pilot_folds"][0]
    orig_fold0_len = len(fr.meta["heldout_pilot_folds"][0]["heldout_pilot_indices"])
    fr_p.meta["heldout_pilot_folds"][0]["heldout_pilot_indices"].append(999)
    assert len(fr.meta["heldout_pilot_folds"][0]["heldout_pilot_indices"]) == orig_fold0_len


def test_compat_unpickler_allowlist_and_rejection(tmp_path: Path) -> None:
    """_CompatUnpickler must reject disallowed globals (e.g. os.system) while allowing CustomDataset."""
    for payload in (
        b"\x80\x02cos\nsystem\nq\x00.",
        b"\x80\x02cposix\nsystem\nq\x00.",
        b"\x80\x02cbuiltins\neval\nq\x00.",
    ):
        with pytest.raises(pickle.UnpicklingError, match="Disallowed pickle global"):
            _CompatUnpickler(io.BytesIO(payload)).load()

    ds = CustomDataset()
    iq, bits, sinr = _make_synthetic_raw(seed=88)
    ds.add_item(torch.from_numpy(iq.astype(np.complex64)), bits.astype(np.int64), sinr)
    synth_pth = tmp_path / "custom_ds.pth"
    torch.save(ds, synth_pth)

    iq_out, lab_out, sinr_out = load_raw_sdr_ofdm(synth_pth)
    assert len(iq_out) == 1
    assert iq_out[0].shape == (N_SYMB, N_FFT)
    assert np.array_equal(lab_out[0], bits)
    assert float(sinr_out[0]) == sinr
