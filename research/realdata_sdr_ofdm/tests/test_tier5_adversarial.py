"""Tier 5 white-box adversarial coverage & bug-regression tests (F13)."""

from __future__ import annotations

import copy
import json
import math
import sys
import tracemalloc
import types
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.realdata_sdr_ofdm.adapters.modulation import (
    _bit_index_groups,
    _validate_qm,
    bits_to_symbols,
    constellation_log_likelihood,
    exact_llr,
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
    QM,
    CustomDataset,
    RealOFDMFrame,
    apply_stress_perturbation,
    build_frame,
    build_heldout_pilot_masks,
    build_leak_free_splits,
    build_masks,
    compute_temporal_diagnostics,
    estimate_blind_n0,
    load_processed_split,
    load_raw_sdr_ofdm,
    pilot_cv_masks,
    preprocess_and_save,
)
from research.realdata_sdr_ofdm.eval.evaluator import (
    _is_valid_output,
    _sign_flip_stats,
    compare_arms,
    compare_multiple_arms,
    evaluate_receiver,
    profile_receiver_pinned,
    run_stress_suite,
    subblock_group_ids,
)
from research.realdata_sdr_ofdm.receivers.classical import (
    CLASSICAL_METHODS,
    ClassicalConfig,
    ReceiverOutput,
    _estimate_sym2_channel,
    _estimate_timing_slope,
    calibrate_on_dev,
    run_frame as run_classical_frame,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    GROUP_A_MODES,
    VAL_PILOT_INDICES,
    GroupAConfig,
    _partition_key_val_pilots,
    build_reference_pool,
    calibrate_group_a_on_dev,
    calibrate_group_a_temperature,
    run_frame as run_group_a_frame,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    GROUP_B_MODES,
    CandidateBranch,
    GroupBConfig,
    GroupBFusionReceiver,
    calibrate_group_b_on_dev,
    fuse_candidate_branches,
    run_frame as run_group_b_frame,
)
from research.realdata_sdr_ofdm.receivers.group_c_trainable import (
    BranchWeightAttentionNet,
    GroupCConfig,
    GroupCTrainableReceiver,
    _build_pilot_pool_summary,
    _estimate_global_branch_from_summary,
    differentiable_mixture_llrs_torch,
)
from research.realdata_sdr_ofdm.scripts import (
    make_figures,
    run_baselines,
    run_group_a,
    run_group_b,
    run_group_c,
    run_unified_benchmark,
)
from research.realdata_sdr_ofdm.scripts.download_and_verify import (
    CustomDataset as DownloadCustomDataset,
    EXIT_FALLBACK_TRIGGERED,
    _compute_hashes,
    main as download_main,
)
from research.realdata_sdr_ofdm.tests.e2e.harness import nongray_16qam_mixture_llr_ref


def _make_frame(
    seed: int = 123,
    *,
    snr_db: float = 18.0,
    split: str = "dev",
    group_id: str = "blk_00",
    frame_id: int = 0,
    delay_samples: float = 0.0,
) -> RealOFDMFrame:
    rng = np.random.default_rng(int(seed))
    p_m, _, d_m = build_masks()
    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, QM), dtype=np.uint8)
    tx_syms = bits_to_symbols(tx_bits, qm=QM)
    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[d_m] = tx_syms
    tx_grid[p_m] = PILOT_VALUES

    k_idx = np.arange(N_FFT, dtype=np.float64) - 64.0
    h_freq = np.exp(-1j * 2.0 * np.pi * delay_samples * k_idx / float(N_FFT))
    n0 = 10.0 ** (-float(snr_db) / 10.0)
    noise = math.sqrt(n0 / 2.0) * (
        rng.standard_normal((N_SYMB, N_FFT)) + 1j * rng.standard_normal((N_SYMB, N_FFT))
    )
    rx = tx_grid * h_freq[None, :] + noise
    return build_frame(
        frame_id=frame_id,
        iq_raw=rx,
        labels_raw=tx_bits,
        sinr_val=snr_db,
        split=split,
        group_id=group_id,
    )


def _make_valid_output(frame: RealOFDMFrame, *, bit_err: int = 0) -> ReceiverOutput:
    hard_bits = frame.tx_bits.copy()
    if bit_err > 0:
        flat = hard_bits.reshape(-1)
        flat[:bit_err] ^= 1
    llrs = np.where(hard_bits == 1, 5.0, -5.0).astype(np.float64)
    return ReceiverOutput(
        h_est=np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128),
        h_var=np.full((N_SYMB, N_ACTIVE), 0.1, dtype=np.float64),
        eq_symbols=frame.tx_symbols.copy(),
        bit_llrs=llrs,
        hard_bits=hard_bits,
        hard_symbols=frame.tx_symbols.copy(),
        heldout_pilot_pred=np.ones((12,), dtype=np.complex128),
        heldout_pilot_var=np.full((12,), 0.1, dtype=np.float64),
        aux_info={"n_candidates": 1, "latency_ms": 1.0, "failed": False, "error": None},
    )


# ---------------------------------------------------------------------------
# 1. Scripts: run_unified_benchmark, run_{baselines,group_a,b,c}, make_figures, download_and_verify
# ---------------------------------------------------------------------------


def test_resolve_canonical_checkpoint_negative_paths_and_fallback(tmp_path: Path) -> None:
    ckpt = tmp_path / "arm_seed1.pt"
    ckpt.write_bytes(b"canonical-payload")
    good_sha = run_unified_benchmark._sha256_file(ckpt)
    manifest = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError):
        run_unified_benchmark._resolve_canonical_checkpoint("group_c_branch_attn", manifest, tmp_path)

    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "arm": "group_c_branch_attn",
                        "seed": 1,
                        "file": "arm_seed1.pt",
                        "sha256": "0" * 64,
                        "dev_ber": 0.05,
                        "dev_pilot_nll": -0.3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_unified_benchmark._resolve_canonical_checkpoint("group_c_branch_attn", manifest, tmp_path)
    with pytest.raises(KeyError):
        run_unified_benchmark._resolve_canonical_checkpoint("unknown_arm", manifest, tmp_path)

    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "arm": "group_c_branch_attn",
                        "seed": 1,
                        "file": "missing.pt",
                        "sha256": good_sha,
                        "dev_ber": 0.05,
                        "dev_pilot_nll": -0.3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        run_unified_benchmark._resolve_canonical_checkpoint("group_c_branch_attn", manifest, tmp_path)

    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "arm": "group_c_branch_attn",
                        "seed": 1,
                        "file": "arm_seed1.pt",
                        "sha256": good_sha,
                        "dev_ber": 0.05,
                        "dev_pilot_nll": -0.3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    other_ckpt_dir = tmp_path / "other_ckpts"
    other_ckpt_dir.mkdir()
    _, resolved = run_unified_benchmark._resolve_canonical_checkpoint(
        "group_c_branch_attn", manifest, other_ckpt_dir
    )
    assert resolved == ckpt


def test_build_per_frame_record_all_groups_nan_and_sinr_stratified(tmp_path: Path) -> None:
    assert run_unified_benchmark._sinr_bin_label(4.9) == "other"
    assert run_unified_benchmark._sinr_bin_label(31.1) == "other"

    fr = _make_frame(1, snr_db=16.0, split="test", group_id="block_01")
    fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
    out_stub = SimpleNamespace(
        h_est=np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128),
        h_var=np.full((N_SYMB, N_ACTIVE), 0.01, dtype=np.float64),
        eq_symbols=fr.tx_symbols.copy(),
        bit_llrs=np.where(fr.tx_bits == 1, 4.0, -4.0),
        hard_bits=fr.tx_bits.copy(),
        hard_symbols=fr.tx_symbols.copy(),
        heldout_pilot_pred=np.ones(4, dtype=np.complex128),
        heldout_pilot_var=np.full(4, 0.02, dtype=np.float64),
        aux_info={
            "heldout_pilot_indices": [1, 5, 9, 12],
            "n_candidates": 6,
            "latency_ms": 5.0,
            "failed": False,
            "error": None,
            "omega_t_rad_per_sym": 0.01,
            "weights": [0.5, 0.5, 0.0],
            "n_values_accessed": 2,
            "branch_entropy": 0.69,
            "between_var_mean": 0.02,
            "branch_weights": [0.7, 0.3],
            "prior_weights": [0.6, 0.4],
            "omega_t_candidates": [0.0, 0.02],
        },
    )
    r_cls = run_unified_benchmark._build_per_frame_record("lmmse_2d_tracked", "classical", fr, out_stub, param_count=0)
    r_ga = run_unified_benchmark._build_per_frame_record("global_content_attn", "group_a", fr, out_stub, param_count=0)
    r_gc = run_unified_benchmark._build_per_frame_record("group_c_hybrid_attn", "group_c", fr, out_stub, param_count=2698)
    assert "omega_t_rad_per_sym" in r_cls
    assert r_ga["n_values_accessed"] == 2 and r_ga["weight_max"] == 0.5
    assert r_gc["param_count"] == 2698 and len(r_gc["branch_weights"]) == 2
    assert r_cls["pilot_nmse"] > 0.0 and math.isfinite(r_cls["pilot_nll"])

    # NaN / failed outputs must serialize cleanly with allow_nan=False across all scripts
    out_nan = _make_valid_output(fr)
    out_nan.eq_symbols = np.full_like(out_nan.eq_symbols, np.nan)
    out_nan.aux_info["latency_ms"] = float("nan")
    for grp, arm in (
        ("classical", "ls_linear_zf"),
        ("group_a", "global_content_attn"),
        ("group_b", "exact_mixture_likelihood"),
        ("group_c", "group_c_hybrid_attn"),
    ):
        rec_nan = run_unified_benchmark._build_per_frame_record(arm, grp, fr, out_nan, param_count=0)
        assert rec_nan["failed"] is True
        json.dumps(rec_nan, allow_nan=False)

    for idx_m, mod in enumerate((run_baselines, run_group_a, run_group_b, run_group_c)):
        jl = tmp_path / f"per_frame_{idx_m}.jsonl"
        mod._write_per_frame_jsonl(jl, [(fr, out_nan)])
        parsed = json.loads(jl.read_text(encoding="utf-8").strip())
        assert parsed["failed"] is True

    # _compute_sinr_stratified: omits 'other' when in [5, 31], includes 'other' when out-of-range
    fr_in = _make_frame(2, snr_db=15.0, split="test", frame_id=1)
    fr_out = _make_frame(3, snr_db=3.5, split="test", frame_id=2)
    arm_metrics = {a: {"peak_rss_mb": 100.0, "param_count": 0, "train_time_s": 0.0} for a in run_unified_benchmark.ALL_ARMS}
    strat_in = run_unified_benchmark._compute_sinr_stratified(
        [fr_in], {a: [(fr_in, _make_valid_output(fr_in))] for a in run_unified_benchmark.ALL_ARMS}, arm_metrics
    )
    assert all(b["label"] != "other" for b in strat_in["bins"])
    assert "other" not in strat_in["arms"]["lmmse_1d"]

    strat_mix = run_unified_benchmark._compute_sinr_stratified(
        [fr_in, fr_out],
        {a: [(fr_in, _make_valid_output(fr_in)), (fr_out, _make_valid_output(fr_out))] for a in run_unified_benchmark.ALL_ARMS},
        arm_metrics,
    )
    assert any(b["label"] == "other" and b["n_frames"] == 1 for b in strat_mix["bins"])
    assert strat_mix["arms"]["lmmse_1d"]["other"]["n_frames"] == 1


def test_run_unified_benchmark_cli_worker_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["run_unified_benchmark.py", "--worker-arm", "lmmse_1d"])
    with pytest.raises(ValueError, match="--worker-out"):
        run_unified_benchmark.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unified_benchmark.py", "--worker-arm", "unknown_arm", "--worker-out", str(tmp_path / "x.pkl")],
    )
    with pytest.raises(ValueError, match="Unknown arm"):
        run_unified_benchmark.main()


def test_make_figures_subblock_clusters_boot_ci_and_empty_sinr_bins(tmp_path: Path) -> None:
    recs_16 = [
        {
            "frame_id": i,
            "group_id": f"block_{i // 6:02d}",
            "bit_errors": 500 + i * 5,
            "total_bits": 5600,
            "frame_error": 1,
            "evm_pct": 65.0 + i,
            "pilot_nll": 2.0 + 0.05 * i,
        }
        for i in range(16)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clusters = make_figures._subblock_clusters(recs_16, target_g=16)
        assert all(len(c) > 0 for c in clusters)
        for m in ("ber_pct", "fer_pct", "evm_pct", "pilot_nll"):
            lo, hi = make_figures._cluster_boot_ci(recs_16, m, n_boot=200)
            assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    assert not [w for w in caught if "Mean of empty slice" in str(w.message)]

    # m=0 and m > target_g must not crash
    assert make_figures._subblock_clusters([], target_g=16) == []
    with pytest.raises(ValueError, match="empty records"):
        make_figures._cluster_boot_ci([], "ber_pct")
    recs_20 = [dict(recs_16[0], frame_id=i, group_id=f"block_{i:02d}") for i in range(20)]
    clusters_20 = make_figures._subblock_clusters(recs_20, target_g=16)
    assert len(clusters_20) == 20 and all(len(c) > 0 for c in clusters_20)

    # make_fig4_sinr_and_cost with empty SINR bins
    fr = _make_frame(0, snr_db=15.0, split="test")
    out = _make_valid_output(fr)
    arm_pairs = {a: [(fr, out)] for a in run_unified_benchmark.ALL_ARMS}
    arm_metrics = {
        a: {
            "peak_rss_mb": 100.0,
            "param_count": 2698 if a.startswith("group_c") else 0,
            "train_time_s": 0.0,
            "ber": 0.0,
            "mean_ms_per_frame": 1.0,
        }
        for a in run_unified_benchmark.ALL_ARMS
    }
    sinr_strat = run_unified_benchmark._compute_sinr_stratified([fr], arm_pairs, arm_metrics)
    summary_clean = {"sinr_stratified": sinr_strat, "arms": arm_metrics}
    per_frame_by_arm = {
        a: [{"frame_id": 0, "group_id": "block_00", "reported_sinr_db": 15.0, "ber": 0.0}]
        for a in run_unified_benchmark.ALL_ARMS
    }
    make_figures.make_fig4_sinr_and_cost(summary_clean, per_frame_by_arm, tmp_path)
    assert (tmp_path / "fig4_sinr_and_cost.png").exists()
    assert (tmp_path / "fig4_sinr_and_cost.pdf").exists()


def test_make_figures_and_download_verify_cli_and_sha_empty_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_figures.py", "--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "figs")],
    )
    with pytest.raises(FileNotFoundError):
        make_figures.main()

    monkeypatch.setattr(sys, "argv", ["download_and_verify.py", "--probe-n", "5"])
    with pytest.raises(ValueError, match="--probe-n"):
        download_main()

    # Case A: missing raw file -> EXIT_FALLBACK_TRIGGERED
    cfg_missing = tmp_path / "cfg_missing.yaml"
    cfg_missing.write_text(
        f"""dataset:
  name: test_ds
  url: https://example.com/ds.pth
  raw_dir: {(tmp_path / 'raw_missing').as_posix()}
  raw_filename: absent.pth
probe:
  min_tti_count: 200
""",
        encoding="utf-8",
    )
    probe_missing = tmp_path / "probe_missing.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_and_verify.py",
            "--config",
            str(cfg_missing),
            "--raw-dir",
            str(tmp_path / "raw_missing"),
            "--manifest-out",
            str(tmp_path / "man_missing.json"),
            "--probe-out",
            str(probe_missing),
            "--skip-download",
        ],
    )
    assert download_main() == EXIT_FALLBACK_TRIGGERED
    assert json.loads(probe_missing.read_text(encoding="utf-8"))["fallback_triggered"] is True

    # Case B: custom size/blob + mismatched SHA-256 -> SHA-256 check still enforced!
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ds_path = raw_dir / "dataset.pth"
    torch.save(DownloadCustomDataset(), ds_path)
    sz, real_sha256, bsha1 = _compute_hashes(ds_path)

    cfg_sha = tmp_path / "cfg_sha.yaml"
    cfg_sha.write_text(
        f"""dataset:
  name: test_ds
  url: https://example.com/ds.pth
  raw_dir: {raw_dir.as_posix()}
  raw_filename: dataset.pth
  expected_size_bytes: {sz}
  expected_blob_sha1: {bsha1}
  expected_sha256: {'f' * 64}
probe:
  min_tti_count: 200
""",
        encoding="utf-8",
    )
    probe_sha = tmp_path / "probe_sha.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_and_verify.py",
            "--config",
            str(cfg_sha),
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(tmp_path / "man_sha.json"),
            "--probe-out",
            str(probe_sha),
            "--skip-download",
        ],
    )
    assert download_main() == EXIT_FALLBACK_TRIGGERED
    summary_sha = json.loads(probe_sha.read_text(encoding="utf-8"))
    assert summary_sha["fallback_triggered"] is True
    assert any("sha256" in r.lower() for r in summary_sha["fallback_reasons"])

    # Case C: 0-frame dataset with matching checksums -> no IndexError on np.quantile, clean fallback exit 42
    cfg_empty = tmp_path / "cfg_empty.yaml"
    cfg_empty.write_text(
        f"""dataset:
  name: test_ds
  url: https://example.com/ds.pth
  raw_dir: {raw_dir.as_posix()}
  raw_filename: dataset.pth
  expected_size_bytes: {sz}
  expected_blob_sha1: {bsha1}
  expected_sha256: {real_sha256}
probe:
  min_tti_count: 200
""",
        encoding="utf-8",
    )
    probe_empty = tmp_path / "probe_empty.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_and_verify.py",
            "--config",
            str(cfg_empty),
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(tmp_path / "man_empty.json"),
            "--probe-out",
            str(probe_empty),
            "--skip-download",
        ],
    )
    assert download_main() == EXIT_FALLBACK_TRIGGERED
    summary_empty = json.loads(probe_empty.read_text(encoding="utf-8"))
    assert summary_empty["n_tti"] == 0
    assert summary_empty["sinr_stats"] is None
    assert summary_empty["fallback_triggered"] is True


# ---------------------------------------------------------------------------
# 2. Adapters: modulation.py & sdr_ofdm_adapter.py
# ---------------------------------------------------------------------------


def test_modulation_error_paths_and_degenerate_shapes() -> None:
    for bad_qm in (0, 1, 3, -2, 4.0, True):
        with pytest.raises(ValueError):
            _validate_qm(bad_qm)

    with pytest.raises(ValueError):
        bits_to_symbols(np.array(0), qm=4)
    with pytest.raises(ValueError):
        bits_to_symbols(np.zeros((2, 3), dtype=np.uint8), qm=4)
    with pytest.raises(ValueError):
        bits_to_symbols(np.array([[0, 2, 1, 0]], dtype=np.int64), qm=4)
    assert bits_to_symbols(np.zeros((0, 4), dtype=np.uint8), qm=4).shape == (0,)

    with pytest.raises(ValueError):
        symbols_to_hard_bits(np.array([complex(float("nan"), 0.0)]), qm=4)
    assert symbols_to_hard_bits(np.zeros((0,), dtype=np.complex128), qm=4).shape == (0, 4)

    with pytest.raises(ValueError):
        constellation_log_likelihood(np.array([complex(float("inf"), 0.0)]), 0.1)
    with pytest.raises(ValueError):
        constellation_log_likelihood(np.array([0.3 + 0.3j]), float("nan"))
    with pytest.raises(ValueError):
        constellation_log_likelihood(np.array([0.3 + 0.3j]), 0.1, min_n0=0.0)
    with pytest.raises(ValueError):
        constellation_log_likelihood(np.array([0.3 + 0.3j]), 0.1, min_n0=1.0, max_n0=0.5)

    ll_hi = constellation_log_likelihood(np.array([0.3 + 0.3j]), 1e20, max_n0=1e12)
    ll_cap = constellation_log_likelihood(np.array([0.3 + 0.3j]), 1e12, max_n0=1e12)
    np.testing.assert_allclose(ll_hi, ll_cap, atol=0.0)
    assert exact_llr(0.3 + 0.3j, 0.1).shape == (4,)
    assert exact_llr(np.zeros((0,), dtype=np.complex64), 0.1, 4).shape == (0, 4)
    assert maxlog_llr(0.3 + 0.3j, 0.1).shape == (4,)


def test_adapter_unpickler_config_module_and_load_validation(tmp_path: Path) -> None:
    fr = _make_frame(2)
    fake_cfg = types.ModuleType("config")
    fake_cfg.CustomDataset = CustomDataset
    sys.modules["config"] = fake_cfg
    CustomDataset.__module__ = "config"
    try:
        ds = CustomDataset()
        ds.add_item(torch.from_numpy(fr.rx_grid.astype(np.complex64)), fr.tx_bits.astype(np.int64), 15.0)
        assert len(ds) == 1 and len(ds[0]) == 3
        pth = tmp_path / "real_style.pth"
        torch.save(ds, pth)
    finally:
        CustomDataset.__module__ = "research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter"
        sys.modules.pop("config", None)

    iq_l, lab_l, sinr_a = load_raw_sdr_ofdm(pth)
    assert len(iq_l) == 1 and "config" not in sys.modules
    assert lab_l[0].shape == (N_DATA_RES, 4) and float(sinr_a[0]) == 15.0

    with pytest.raises(FileNotFoundError):
        load_raw_sdr_ofdm(tmp_path / "does_not_exist.pth")

    bad_obj_pth = tmp_path / "bad_obj.pth"
    torch.save({"pdsch_iq": []}, bad_obj_pth)
    with pytest.raises(ValueError, match="missing required attribute"):
        load_raw_sdr_ofdm(bad_obj_pth)

    bad_len_ds = CustomDataset()
    bad_len_ds.pdsch_iq.append(fr.rx_grid)
    bad_len_pth = tmp_path / "bad_len.pth"
    torch.save(bad_len_ds, bad_len_pth)
    with pytest.raises(ValueError, match="Mismatched lengths"):
        load_raw_sdr_ofdm(bad_len_pth)

    for bad_iq, bad_lab, bad_sinr in (
        (np.zeros((14, 64), dtype=np.complex64), fr.tx_bits, 10.0),
        (np.full((14, 128), complex(float("nan"), 0.0)), fr.tx_bits, 10.0),
        (fr.rx_grid, np.zeros((10, 4), dtype=np.uint8), 10.0),
        (fr.rx_grid, np.full((1400, 4), 2, dtype=np.uint8), 10.0),
        (fr.rx_grid, fr.tx_bits, float("inf")),
    ):
        d_bad = CustomDataset()
        d_bad.add_item(bad_iq, bad_lab, bad_sinr)
        p_bad = tmp_path / "item_bad.pth"
        torch.save(d_bad, p_bad)
        with pytest.raises(ValueError):
            load_raw_sdr_ofdm(p_bad)


def test_adapter_preprocess_roundtrip_and_edge_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_heldout_pilot_masks(1)
    with pytest.raises(ValueError):
        build_heldout_pilot_masks(13)
    p_m, _, _ = build_masks()
    with pytest.raises(ValueError):
        pilot_cv_masks(p_m, [-1])
    with pytest.raises(ValueError):
        pilot_cv_masks(p_m, [14])
    with pytest.raises(FileNotFoundError):
        load_processed_split(tmp_path, split="missing")

    fr = _make_frame(3)
    rx_imp = fr.rx_grid.copy()
    rx_imp[0, 10] = 20.0 + 0.0j
    n0_imp, info_imp = estimate_blind_n0(rx_imp)
    assert n0_imp > 0.0 and info_imp["n0_source"] == "guard_passband_median"

    with pytest.raises(ValueError):
        estimate_blind_n0(fr.rx_grid, n0_floor=0.0)
    with pytest.raises(ValueError):
        estimate_blind_n0(fr.rx_grid, safe_guard_indices=())

    with pytest.raises(ValueError):
        apply_stress_perturbation(fr, cfo_norm=float("nan"))
    with pytest.raises(ValueError):
        apply_stress_perturbation(fr, extra_awgn_db=float("inf"))
    with pytest.raises(ValueError):
        apply_stress_perturbation(fr, cp_len=-1)
    fr_z = replace(fr, rx_grid=np.zeros_like(fr.rx_grid), est_n0=1e-8)
    fr_zp = apply_stress_perturbation(fr_z, extra_awgn_db=0.0, cfo_norm=0.05, cp_len=0, seed=1)
    assert fr_zp.meta["perturbation"]["sigma2_add"] == 0.0
    assert np.all(np.isfinite(fr_zp.rx_grid))

    diag = compute_temporal_diagnostics(
        np.full(20, 15.0),
        np.ones((20, 14), dtype=np.complex128),
        [f"h_{i}" for i in range(20)],
        max_lag=5,
        min_guard_frames=2,
        max_guard_frames=4,
    )
    assert diag["sinr_autocorr_lags"]["1"] == 0.0 and diag["tau_c"] == 4

    with pytest.raises(ValueError):
        build_leak_free_splits(
            np.full(5, 15.0),
            np.ones((5, 14), dtype=np.complex128),
            [f"h_{i}" for i in range(5)],
            n_macro_blocks=2,
        )

    ds = CustomDataset()
    for i in range(8):
        fi = _make_frame(10 + i, frame_id=i)
        ds.add_item(torch.from_numpy(fi.rx_grid.astype(np.complex64)), fi.tx_bits, 15.0 + i)
    raw_p = tmp_path / "raw.pth"
    torch.save(ds, raw_p)
    sp = {
        "blocks": [
            {"group_id": "b0", "split": "train", "indices": [0, 1, 2]},
            {"group_id": "b1", "split": "dev", "indices": [4, 5]},
            {"group_id": "b2", "split": "test", "indices": [6, 7]},
        ],
        "guard_indices": [3],
    }
    sp_p = tmp_path / "splits.json"
    sp_p.write_text(json.dumps(sp), encoding="utf-8")
    manifest = preprocess_and_save(
        raw_p, sp_p, tmp_path / "proc", tmp_path / "unit_metadata.jsonl", tmp_path / "proc_checksums.json"
    )
    assert manifest["n_total_processed"] == 8
    loaded_te = load_processed_split(tmp_path / "proc", split="test")
    assert [f.frame_id for f in loaded_te] == [6, 7]


def test_preprocess_and_save_empty_guard_split(tmp_path: Path) -> None:
    ds = CustomDataset()
    for i in range(6):
        fi = _make_frame(20 + i, frame_id=i)
        ds.add_item(torch.from_numpy(fi.rx_grid.astype(np.complex64)), fi.tx_bits, 15.0)
    raw_p = tmp_path / "raw.pth"
    torch.save(ds, raw_p)
    sp = {
        "blocks": [
            {"group_id": "b0", "split": "train", "indices": [0, 1]},
            {"group_id": "b1", "split": "dev", "indices": [2, 3]},
            {"group_id": "b2", "split": "test", "indices": [4, 5]},
        ],
        "guard_indices": [],
    }
    sp_p = tmp_path / "splits.json"
    sp_p.write_text(json.dumps(sp), encoding="utf-8")
    preprocess_and_save(raw_p, sp_p, tmp_path / "proc", tmp_path / "meta.jsonl", tmp_path / "chk.json")
    assert load_processed_split(tmp_path / "proc", split="guard") == []


def test_estimate_blind_n0_cstar_equivariance_below_1e10() -> None:
    fr = _make_frame(4, snr_db=15.0)
    n0_1, i1 = estimate_blind_n0(fr.rx_grid, n0_floor=1e-30)
    n0_s, is_ = estimate_blind_n0(1e-6 * fr.rx_grid, n0_floor=1e-30)
    assert i1["n0_source"] == is_["n0_source"]
    assert math.isclose(n0_s / 1e-12, n0_1, rel_tol=1e-6)

    # Also verify impulsive median branch stays scale-equivariant below 1e-10
    rx_imp = fr.rx_grid.copy()
    rx_imp[0, 10] = 20.0 + 0.0j
    n0_imp_1, info_imp_1 = estimate_blind_n0(rx_imp, n0_floor=1e-30)
    n0_imp_s, info_imp_s = estimate_blind_n0(1e-6 * rx_imp, n0_floor=1e-30)
    assert info_imp_1["n0_source"] == "guard_passband_median"
    assert info_imp_s["n0_source"] == "guard_passband_median"
    assert math.isclose(n0_imp_s / 1e-12, n0_imp_1, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. Receivers: classical, group_a_reference, group_b_fusion, group_c_trainable
# ---------------------------------------------------------------------------


def test_classical_cpe_detrend_fd_zero_grid_and_error_paths() -> None:
    fr_delay = _make_frame(101, snr_db=22.0, delay_samples=3.5, split="dev")
    cfg_adv = ClassicalConfig(
        detrend_timing=True,
        fd_norm=0.004,
        time_var_growth=0.01,
        cpe_refine_alpha=0.5,
    )
    o_adv = run_classical_frame(fr_delay, method="lmmse_2d_tracked", config=cfg_adv)
    assert not o_adv.aux_info["failed"]
    assert np.all(np.isfinite(o_adv.bit_llrs))
    assert float(np.mean(o_adv.hard_bits != fr_delay.tx_bits)) < 0.02

    assert _estimate_timing_slope(np.array([0.0]), np.array([1.0 + 0j])) == 0.0
    assert _estimate_timing_slope(np.array([0.0, 8.0]), np.array([0.0 + 0j, 1.0 + 0j])) == 0.0

    fr_z = replace(fr_delay, rx_grid=np.zeros((N_SYMB, N_FFT), dtype=np.complex128), est_n0=0.0)
    for m in CLASSICAL_METHODS:
        oz = run_classical_frame(fr_z, method=m)
        assert not oz.aux_info["failed"]
        assert np.all(np.isfinite(oz.bit_llrs)) and np.all(oz.h_var > 0.0)

    for bad_fr in (
        replace(fr_delay, rx_grid=np.zeros((14, 64), dtype=np.complex128)),
        replace(fr_delay, est_n0=-0.1),
        replace(fr_delay, est_n0=float("nan")),
    ):
        ob = run_classical_frame(bad_fr, method="lmmse_1d")
        assert ob.aux_info["failed"] and bool(ob.aux_info["error"])

    fr_14 = copy.deepcopy(fr_delay)
    fr_14.meta["heldout_pilot_indices"] = list(range(14))
    assert run_classical_frame(fr_14, method="ls_linear_zf").aux_info["failed"]

    tuned = calibrate_on_dev(
        [fr_delay],
        methods=("lmmse_2d_tracked",),
        n0_scales=(1.0,),
        tau_rms_grid=(1.5,),
        n_taps_grid=(5,),
        fd_norm_grid=(0.0, 0.002),
    )
    assert "lmmse_2d_tracked" in tuned


def test_group_a_section51_val_isolation_pool_edges_and_slow_calib() -> None:
    fr_q = _make_frame(201, snr_db=16.0, split="dev", group_id="g0", frame_id=0)
    pool = [_make_frame(210 + i, snr_db=20.0, split="train", group_id=f"g{i % 2}", frame_id=10 + i) for i in range(4)]

    o1 = run_group_a_frame(fr_q, context_pool=pool, mode="global_content_attn", temperature=0.1)
    pool_val_pert = copy.deepcopy(pool)
    val_fft = [PILOT_FFT_INDICES[i] for i in VAL_PILOT_INDICES]
    for p in pool_val_pert:
        p.rx_grid[PILOT_SYMBOL_IDX, val_fft] *= 50.0
    o2 = run_group_a_frame(fr_q, context_pool=pool_val_pert, mode="global_content_attn", temperature=0.1)
    np.testing.assert_array_equal(o1.aux_info["weights"], o2.aux_info["weights"])

    with pytest.raises(ValueError):
        _partition_key_val_pilots([])
    assert _partition_key_val_pilots([3]) == ([3], [3])
    with pytest.raises(ValueError):
        build_reference_pool([])
    with pytest.raises(ValueError):
        build_reference_pool(pool, max_pool_size=0)
    imb = build_reference_pool(
        [pool[0]] + [_make_frame(230 + i, split="train", group_id="g1", frame_id=30 + i) for i in range(5)],
        max_pool_size=4,
    )
    assert len(imb) == 4

    fr_z = replace(fr_q, rx_grid=np.zeros((N_SYMB, N_FFT), dtype=np.complex128), est_n0=0.0)
    for m in GROUP_A_MODES:
        oz = run_group_a_frame(fr_z, context_pool=pool, mode=m)
        assert not oz.aux_info["failed"] and np.all(np.isfinite(oz.bit_llrs))
    o_pt = run_group_a_frame(fr_q, context_pool=pool, mode="global_content_attn", config=GroupAConfig(phase_track=True))
    assert not o_pt.aux_info["failed"]

    assert run_group_a_frame(fr_q, context_pool=pool, temperature=-1.0).aux_info["failed"]
    assert run_group_a_frame(fr_q, context_pool=pool, mode="local", local_k=0).aux_info["failed"]
    bad_pool = copy.deepcopy(pool)
    bad_pool[0].rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[0]] = complex(float("nan"), 0.0)
    assert run_group_a_frame(fr_q, context_pool=bad_pool).aux_info["failed"]

    assert calibrate_group_a_temperature([fr_q], [fr_q], temps=(0.1, 0.5)) in (0.1, 0.5)
    with pytest.raises(ValueError):
        calibrate_group_a_temperature([fr_q], pool, temps=(0.0,))
    cfg_slow, det_slow = calibrate_group_a_on_dev(
        [fr_q], pool, temps=(0.1, 0.5), prior_scales=(1.0,), base_config=GroupAConfig(phase_track=True)
    )
    assert len(det_slow["grid_records"]) == 2 and cfg_slow.phase_track


def test_group_b_constellation_var_oracle_and_error_paths() -> None:
    fr = _make_frame(301, snr_db=16.0, split="test")
    m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
    m1 = np.full((N_SYMB, N_ACTIVE), 0.5 + 0.5j, dtype=np.complex128)
    v0 = np.full((N_SYMB, N_ACTIVE), 0.10, dtype=np.float64)
    v1 = np.full((N_SYMB, N_ACTIVE), 0.20, dtype=np.float64)
    dm = fr.data_mask[:, 13:115]
    y_vec = fr.rx_grid[:, 13:115][dm]

    o_inc = fuse_candidate_branches(
        fr, branches=[(m0, v0), (m1, v1)], branch_weights=[0.7, 0.3], include_constellation_var=True, return_heldout_cv=False
    )
    ref_inc = nongray_16qam_mixture_llr_ref(
        y_vec, [m0[dm], m1[dm]], [v0[dm], v1[dm]], [0.7, 0.3], fr.est_n0, include_constellation_var=True
    )
    np.testing.assert_allclose(o_inc.bit_llrs, ref_inc, atol=1e-12)

    rx_b = GroupBFusionReceiver("exact_mixture_likelihood", GroupBConfig(include_constellation_var=True))
    out_b = rx_b.run_frame(fr)
    assert out_b.h_var.shape == (N_SYMB, N_ACTIVE)
    assert np.all(np.isfinite(out_b.h_var))

    m_nan = np.full((N_SYMB, N_ACTIVE), complex(float("nan"), 0.0), dtype=np.complex128)
    v_neg = np.full((N_SYMB, N_ACTIVE), -1.0, dtype=np.float64)
    o_skip = fuse_candidate_branches(fr, branches=[(m0, v0), (m_nan, v_neg)], branch_weights=[1.0, 0.0])
    assert np.all(np.isfinite(o_skip.bit_llrs))

    o_128 = fuse_candidate_branches(
        fr,
        branches=[(np.ones((N_SYMB, N_FFT), dtype=np.complex128), np.full((N_SYMB, N_FFT), 0.01))],
        branch_weights=[1.0],
    )
    assert o_128.h_est.shape == (N_SYMB, N_ACTIVE)

    for bad_br, bad_w in (
        ([], []),
        ([(m0, v0)], [0.0]),
        ([(m0, v0)], [-0.5]),
        ([(m0, v0)], [float("nan")]),
        ([(m0, v0)], [0.5, 0.5]),
        ([(m_nan, v0)], [1.0]),
        ([(m0, v_neg)], [1.0]),
        ([(np.ones((N_SYMB, 64), dtype=np.complex128), np.ones((N_SYMB, 64)))], [1.0]),
    ):
        with pytest.raises(ValueError):
            fuse_candidate_branches(fr, branches=bad_br, branch_weights=bad_w)
        assert run_group_b_frame(fr, branches=bad_br, branch_weights=bad_w).aux_info["failed"]

    assert run_group_b_frame(fr, config=GroupBConfig(pdp_hypotheses=())).aux_info["failed"]
    fr_z = replace(fr, rx_grid=np.zeros((N_SYMB, N_FFT), dtype=np.complex128), est_n0=0.0)
    for m in GROUP_B_MODES:
        assert not run_group_b_frame(fr_z, mode=m).aux_info["failed"]
    with pytest.raises(ValueError):
        calibrate_group_b_on_dev([])


def test_group_c_precompute_pilot_nll_only_reload_and_edges(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BranchWeightAttentionNet(d_c=7, d_s=12, n_heads=2)
    cfg = GroupCConfig(epochs=1)
    assert GroupCConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()

    tr = [_make_frame(400 + i, split="train", group_id="g0", frame_id=i) for i in range(3)]
    dv = [_make_frame(410 + i, split="dev", group_id="g1", frame_id=10 + i) for i in range(2)]
    te = _make_frame(420, split="test", group_id="g2", frame_id=20)

    with pytest.raises(ValueError):
        GroupCTrainableReceiver(cfg, seed=1).fit(tr, dv, epochs=0)

    pre_rx = GroupCTrainableReceiver(cfg, seed=42)
    tr_it, dv_it, pre_s = pre_rx.precompute_training_splits(tr, dv)
    m_pre = GroupCTrainableReceiver(cfg, seed=42).fit(tr, dv, precomputed_train=tr_it, precomputed_dev=dv_it, precompute_time_s=pre_s)
    m_dir = GroupCTrainableReceiver(cfg, seed=42).fit(tr, dv)
    np.testing.assert_allclose(m_pre.run_frame(te).bit_llrs, m_dir.run_frame(te).bit_llrs, atol=1e-12)

    m_nodev = GroupCTrainableReceiver(GroupCConfig(include_global_branch=False, epochs=1), seed=7).fit(tr, dev_frames=None)
    o_r1 = m_nodev.run_frame(te, branches=[(np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128), np.full((N_SYMB, N_ACTIVE), 0.01))])
    assert not o_r1.aux_info["failed"] and o_r1.aux_info["branch_weights"] == [1.0]
    assert m_nodev.run_frame(te, branches=[]).aux_info["failed"]
    assert m_dir.run_frame(te, context_pool=[te]).aux_info["failed"]

    # pilot_nll_only path + weights_only=True checkpoint reload parity
    cfg_nll = GroupCConfig(bce_weight=0.0, pilot_nll_weight=1.0, epochs=1, lr=1e-3, d_c=8, n_heads=2)
    rec_nll = GroupCTrainableReceiver(config=cfg_nll, seed=42).fit(tr[:2], dev_frames=None)
    assert len(rec_nll.history) == 2
    ck = tmp_path / "gc.pt"
    rec_nll.save(ck)
    payload = torch.load(ck, map_location="cpu", weights_only=True)
    assert "state_dict" in payload
    rec_nll2 = GroupCTrainableReceiver.load(ck)
    np.testing.assert_allclose(rec_nll.run_frame(te).bit_llrs, rec_nll2.run_frame(te).bit_llrs, atol=1e-12)

    bad_ckpt = tmp_path / "corrupt.pt"
    bad_ckpt.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        GroupCTrainableReceiver.load(bad_ckpt)


def test_group_c_runtime_context_pool_heldout_cv_consistency() -> None:
    tr = [_make_frame(430 + i, split="train", group_id="g0", frame_id=i) for i in range(3)]
    te = _make_frame(440, split="test", group_id="g2", frame_id=20)
    te.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
    m_no_pool = GroupCTrainableReceiver(GroupCConfig(include_global_branch=True), seed=1)
    m_no_pool.pool_summary = None
    m_with_pool = GroupCTrainableReceiver(GroupCConfig(include_global_branch=True), seed=1, context_pool=tr)
    m_with_pool.net.load_state_dict(m_no_pool.net.state_dict())

    o_ctx = m_no_pool.run_frame(te, context_pool=tr, return_heldout_cv=True)
    o_self = m_with_pool.run_frame(te, context_pool=None, return_heldout_cv=True)
    np.testing.assert_allclose(o_ctx.heldout_pilot_pred, o_self.heldout_pilot_pred, atol=1e-12)
    np.testing.assert_allclose(o_ctx.heldout_pilot_var, o_self.heldout_pilot_var, atol=1e-12)


def test_group_c_zero_loss_weights_and_zero_alpha_nan_mask() -> None:
    tr = [_make_frame(450 + i, split="train", group_id="g0", frame_id=i) for i in range(2)]
    dv = [_make_frame(460, split="dev", group_id="g1", frame_id=10)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GroupCTrainableReceiver(
            GroupCConfig(bce_weight=0.0, pilot_nll_weight=0.0, prior_reg_weight=0.0, epochs=1),
            seed=1,
        ).fit(tr, dv)
    sched_warns = [w for w in caught if "lr_scheduler.step()" in str(w.message)]
    assert not sched_warns

    idx1, idx0 = _bit_index_groups(4)
    log_g = torch.zeros((1, 2, 8, 16), dtype=torch.float64, requires_grad=True)
    log_g_in = log_g.clone()
    with torch.no_grad():
        log_g_in[0, 1] = float("nan")
    alpha = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    llrs = differentiable_mixture_llrs_torch(
        alpha,
        log_g_in,
        torch.from_numpy(idx1.copy()).long(),
        torch.from_numpy(idx0.copy()).long(),
    )
    assert torch.all(torch.isfinite(llrs))
    llrs.sum().backward()
    assert alpha.grad is not None and torch.all(torch.isfinite(alpha.grad))


# ---------------------------------------------------------------------------
# 4. Evaluator: validation, G=1 sign-flip, denoised, compare_arms, profiler, stress suite
# ---------------------------------------------------------------------------


def test_evaluator_sign_flip_g1_validation_timeout_and_bad_pilot_idx() -> None:
    pos = _sign_flip_stats([5.0])
    neg = _sign_flip_stats([-5.0])
    zero = _sign_flip_stats([0.0])
    assert pos["p_two_sided"] == pytest.approx(1.0)
    assert pos["p_one_sided_pos"] == pytest.approx(0.5)
    assert neg["p_one_sided_pos"] == pytest.approx(1.0)
    assert zero["p_one_sided_pos"] == pytest.approx(1.0)

    fr = _make_frame(500, snr_db=15.0, split="test")
    with pytest.raises(ValueError, match="metric_mode"):
        evaluate_receiver([(fr, _make_valid_output(fr))], metric_mode="bogus_mode")

    out_bad_idx = _make_valid_output(fr)
    out_bad_idx.heldout_pilot_pred = np.ones((4,), dtype=np.complex128)
    out_bad_idx.heldout_pilot_var = np.full((4,), 0.1, dtype=np.float64)
    out_bad_idx.aux_info["heldout_pilot_indices"] = [0, 1, 2, 99]
    res_idx = evaluate_receiver([(fr, out_bad_idx)])
    assert res_idx["fail_count"] == 1 and res_idx["coverage_pct"] == 0.0

    out_cplx_llr = _make_valid_output(fr)
    out_cplx_llr.bit_llrs = out_cplx_llr.bit_llrs.astype(np.complex64)
    assert evaluate_receiver([(fr, out_cplx_llr)])["fail_count"] == 1

    out_nonbin = _make_valid_output(fr)
    out_nonbin.hard_bits = np.full_like(out_nonbin.hard_bits, 2)
    assert evaluate_receiver([(fr, out_nonbin)])["fail_count"] == 1

    out_bad_h = _make_valid_output(fr)
    out_bad_h.h_est = np.ones((1,), dtype=np.complex128)
    assert evaluate_receiver([(fr, out_bad_h)])["fail_count"] == 1

    out_bad_hsym = _make_valid_output(fr)
    out_bad_hsym.hard_symbols = np.ones((1,), dtype=np.complex128)
    assert evaluate_receiver([(fr, out_bad_hsym)])["fail_count"] == 1

    out_half_pilot = _make_valid_output(fr)
    out_half_pilot.heldout_pilot_var = None
    assert evaluate_receiver([(fr, out_half_pilot)])["fail_count"] == 1

    out_to = _make_valid_output(fr)
    out_to.aux_info["timeout"] = True
    res_to = evaluate_receiver([(fr, out_to)])
    assert res_to["timeout_count"] == 1 and res_to["fail_count"] == 1 and res_to["coverage_pct"] == 0.0


def test_evaluator_denoised_all_fail_compare_edges_profiler_and_stress() -> None:
    fr_noisy = _make_frame(505, snr_db=0.0, split="test")
    fr_noisy = replace(fr_noisy, est_n0=100.0)
    out_ok = _make_valid_output(fr_noisy)
    res_den = evaluate_receiver([(fr_noisy, out_ok)], metric_mode="denoised")
    assert res_den["pilot_nmse_denoised"] == 0.0
    assert res_den["pilot_nmse"] == 0.0
    assert res_den["coverage_pct"] == 100.0

    res_fail = evaluate_receiver([(fr_noisy, None)])
    assert res_fail["coverage_pct"] == 0.0 and res_fail["fail_count"] == 1
    assert res_fail["total_bits"] == 0 and res_fail["ber"] == 0.0

    with pytest.raises(ValueError):
        evaluate_receiver([])

    frames = [_make_frame(510 + i, split="test", group_id=f"b{i // 2}", frame_id=i) for i in range(4)]
    outs = [run_classical_frame(f) for f in frames]
    outs_to = [replace(outs[0], aux_info={**outs[0].aux_info, "timeout": True})] + outs[1:]

    ev = evaluate_receiver(list(zip(frames, outs_to)), metric_mode="denoised", peak_rss_mb=123.4)
    assert ev["timeout_count"] == 1 and ev["fail_count"] == 1 and ev["coverage_pct"] == 75.0
    assert ev["peak_rss_mb"] == 123.4
    assert ev["pilot_nmse"] == ev["pilot_nmse_denoised"]

    class _Missing:
        pass

    assert not _is_valid_output(frames[0], _Missing())[0]
    assert not _is_valid_output(frames[0], replace(outs[0], h_var=np.ones((14, 102), dtype=np.complex128)))[0]
    assert not _is_valid_output(frames[0], replace(outs[0], heldout_pilot_pred=np.zeros(0, dtype=np.complex128)))[0]

    pairs = list(zip(frames, outs))
    st_eq = compare_arms(pairs, pairs, metric="ber", n_boot=50)
    assert st_eq["mean_pp"] == 0.0 and st_eq["ties"] == 2 and st_eq["p_two_sided"] == 1.0
    assert compare_arms(pairs, pairs, metric="evm_pct", n_boot=0)["mean_pp"] == 0.0
    assert compare_arms(pairs, pairs, metric="pilot_nmse", n_boot=20)["mean_pp"] == 0.0
    assert compare_multiple_arms({"a": pairs}, anchor_name="a") == {}

    pairs_broken = [(frames[0], None), (frames[1], None), pairs[2], pairs[3]]
    with pytest.raises(ValueError, match="0 valid receiver outputs"):
        compare_arms(pairs, pairs_broken)

    for bad_kwargs in (
        dict(base_pairs=[], cand_pairs=[]),
        dict(base_pairs=pairs[:2], cand_pairs=pairs[:1]),
        dict(base_pairs=[pairs[0]], cand_pairs=[pairs[1]]),
        dict(base_pairs=pairs, cand_pairs=pairs, metric="unknown"),
    ):
        with pytest.raises(ValueError):
            compare_arms(**bad_kwargs)
    with pytest.raises(ValueError):
        compare_multiple_arms({"a": pairs}, anchor_name="missing")

    assert _sign_flip_stats([])["p_two_sided"] == 1.0
    assert subblock_group_ids([], target_g=4) == {}
    assert subblock_group_ids(frames, target_g=1) == {f.frame_id: f.group_id for f in frames}

    tracemalloc.start()
    try:
        prof = profile_receiver_pinned(frames[:2], lambda f: run_classical_frame(f), warmup_frames=0)
        assert tracemalloc.is_tracing()
        assert prof["mean_ms_per_frame"] > 0.0
    finally:
        tracemalloc.stop()

    with pytest.raises(ValueError):
        profile_receiver_pinned([], lambda f: f)
    with pytest.raises(ValueError):
        run_stress_suite([], lambda f: f)

    suite = run_stress_suite(
        frames[:2],
        lambda f: run_classical_frame(f),
        extra_awgn_db=(None, 5.0),
        cfo_norm=(0.0,),
        phase_drift_rad=(0.0,),
        param_count=2698,
        train_time_s=123.45,
    )
    assert len(suite["conditions"]) == 2
    assert suite["conditions"][0]["condition"] == "clean"
    assert suite["clean"]["train_time_s"] == pytest.approx(123.45)
    for cond in suite["conditions"]:
        assert cond["metrics"]["train_time_s"] == pytest.approx(123.45)


# ---------------------------------------------------------------------------
# 5. Iteration 2 Challenger Regression Tests (G1-G5, Gap 1-3, Minor 4)
# ---------------------------------------------------------------------------


def test_cstar_equivariance_heldout_pilot_var_and_evaluator_1e_15() -> None:
    dev_f = _make_frame(601, snr_db=18.0, split="dev", frame_id=1)
    scale = 1e-15
    tiny_f = replace(
        dev_f,
        rx_grid=dev_f.rx_grid * scale,
        est_n0=dev_f.est_n0 * (scale ** 2),
    )

    out_cls_1 = run_classical_frame(dev_f)
    out_cls_t = run_classical_frame(tiny_f)
    np.testing.assert_allclose(
        out_cls_t.heldout_pilot_var, out_cls_1.heldout_pilot_var * (scale ** 2), rtol=1e-4, atol=0.0
    )

    ev_1 = evaluate_receiver([(dev_f, out_cls_1)])
    ev_t = evaluate_receiver([(tiny_f, out_cls_t)])
    assert np.isclose(ev_1["pilot_cal_ratio"], ev_t["pilot_cal_ratio"], rtol=1e-4)

    br_1 = [CandidateBranch(h_est=out_cls_1.h_est.copy(), h_var=out_cls_1.h_var.copy(), name="b0")]
    br_t = [CandidateBranch(h_est=out_cls_1.h_est * scale, h_var=out_cls_1.h_var * (scale ** 2), name="b0")]
    gb_1 = fuse_candidate_branches(dev_f, branches=br_1, branch_weights=[1.0])
    gb_t = fuse_candidate_branches(tiny_f, branches=br_t, branch_weights=[1.0])
    np.testing.assert_allclose(
        gb_t.heldout_pilot_var, gb_1.heldout_pilot_var * (scale ** 2), rtol=1e-4, atol=0.0
    )

    tr = [_make_frame(602 + i, snr_db=18.0, split="train", frame_id=10 + i) for i in range(3)]
    dv = [_make_frame(605 + i, snr_db=18.0, split="dev", frame_id=20 + i) for i in range(2)]
    rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=2, batch_size=2), seed=7)
    rec_c.fit(tr, dv)
    gc_1 = rec_c.run_frame(dev_f, branches=br_1)
    gc_t = rec_c.run_frame(tiny_f, branches=br_t)
    np.testing.assert_allclose(
        gc_t.heldout_pilot_var, gc_1.heldout_pilot_var * (scale ** 2), rtol=1e-4, atol=0.0
    )

    # Also verify Group A heldout CV & benchmark per-frame record at 1e-15
    ga_1 = run_group_a_frame(dev_f, context_pool=tr, mode="global_content_attn")
    tr_t = [replace(p, rx_grid=p.rx_grid * scale, est_n0=p.est_n0 * (scale ** 2)) for p in tr]
    ga_t = run_group_a_frame(tiny_f, context_pool=tr_t, mode="global_content_attn")
    np.testing.assert_allclose(
        ga_t.heldout_pilot_var, ga_1.heldout_pilot_var * (scale ** 2), rtol=1e-4, atol=0.0
    )
    rec_1 = run_unified_benchmark._build_per_frame_record("lmmse_2d_tracked", "classical", dev_f, out_cls_1, param_count=0)
    rec_t = run_unified_benchmark._build_per_frame_record("lmmse_2d_tracked", "classical", tiny_f, out_cls_t, param_count=0)
    assert np.isclose(rec_1["pilot_cal_ratio"], rec_t["pilot_cal_ratio"], rtol=1e-4)


def test_complex_branch_h_var_rejected_across_group_b_and_c() -> None:
    dev_f = _make_frame(610, snr_db=18.0, split="dev", frame_id=1)
    tr = [_make_frame(611 + i, snr_db=18.0, split="train", frame_id=10 + i) for i in range(3)]
    dv = [_make_frame(614 + i, snr_db=18.0, split="dev", frame_id=20 + i) for i in range(2)]
    out_cls = run_classical_frame(dev_f)
    br_cplx = [CandidateBranch(h_est=out_cls.h_est.copy(), h_var=out_cls.h_var + 1j * 2.0, name="cplx")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="complex|real numeric"):
            fuse_candidate_branches(dev_f, branches=br_cplx, branch_weights=[1.0])

        out_gb = run_group_b_frame(dev_f, branches=br_cplx, branch_weights=[1.0])
        assert out_gb.aux_info["failed"] is True

        rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=2, batch_size=2), seed=11)
        rec_c.fit(tr, dv)
        out_gc = rec_c.run_frame(dev_f, branches=br_cplx)
        assert out_gc.aux_info["failed"] is True
    assert not [w for w in caught if "ComplexWarning" in w.category.__name__ or "imaginary part" in str(w.message)]


def test_negative_heldout_pilot_indices_rejected_all_receivers_and_group_a_pool_guard() -> None:
    dev_f = _make_frame(620, snr_db=18.0, split="dev", frame_id=1)
    tr = [_make_frame(621 + i, snr_db=18.0, split="train", frame_id=10 + i) for i in range(3)]
    dv = [_make_frame(624 + i, snr_db=18.0, split="dev", frame_id=20 + i) for i in range(2)]
    te = _make_frame(626, snr_db=18.0, split="test", frame_id=30)

    rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=2, batch_size=2), seed=13)
    rec_c.fit(tr, dv)

    for bad_idx in ([-1], [14], [True]):
        f_bad = replace(dev_f, meta={**dev_f.meta, "heldout_pilot_indices": bad_idx})
        assert run_classical_frame(f_bad).aux_info["failed"] is True
        assert run_group_a_frame(f_bad, context_pool=tr[:2]).aux_info["failed"] is True
        assert run_group_b_frame(f_bad).aux_info["failed"] is True
        assert rec_c.run_frame(f_bad).aux_info["failed"] is True

    # G5: Group A run_frame rejects test/guard frames in context_pool consistently with Group C
    assert run_group_a_frame(dev_f, context_pool=[te]).aux_info["failed"] is True


def test_group_c_self_pool_exclusion_single_frame_pool() -> None:
    f0 = _make_frame(630, snr_db=18.0, split="train", frame_id=0)
    cfg = GroupCConfig(epochs=2, batch_size=2)
    pool_one = _build_pilot_pool_summary([f0], max_pool_size=64, group_a_cfg=cfg.group_a_cfg)
    s0 = float(np.sqrt(np.mean(np.abs(f0.rx_grid[f0.pilot_mask | f0.data_mask]) ** 2)))
    rx_norm = f0.rx_grid / s0
    n0_norm = f0.est_n0 / (s0 ** 2)
    obs_cv = [i for i in range(14) if i not in (1, 5, 9)]
    h_q, _, _ = _estimate_sym2_channel(
        rx_norm,
        f0.pilot_values,
        obs_cv,
        n0_norm,
        method="lmmse_1d",
        config=cfg.group_a_cfg.to_classical_config(),
    )
    h_g, _ = _estimate_global_branch_from_summary(
        rx_norm,
        f0.pilot_values,
        n0_norm,
        obs_cv,
        pool_one,
        group_a_cfg=cfg.group_a_cfg,
        exclude_frame_id=0,
    )
    np.testing.assert_allclose(h_g, h_q, rtol=1e-12, atol=1e-12)


def test_evaluator_timeout_consistency_and_none_aux_resilience(tmp_path: Path) -> None:
    fr0 = _make_frame(640, snr_db=16.0, split="test", group_id="b0", frame_id=0)

    # 1. _is_valid_output must treat timeout=True identically to evaluate_receiver and reject None/object arrays
    out_to = _make_valid_output(fr0)
    out_to.aux_info["timeout"] = True
    assert _is_valid_output(fr0, out_to)[0] is False

    bad_out = _make_valid_output(fr0)
    bad_out.h_est = None  # type: ignore[assignment]
    ok, _ = _is_valid_output(fr0, bad_out)
    assert ok is False

    bad_obj = _make_valid_output(fr0)
    bad_obj.eq_symbols = np.full((N_DATA_RES,), None, dtype=object)
    assert _is_valid_output(fr0, bad_obj)[0] is False

    bad_p = _make_valid_output(fr0)
    bad_p.heldout_pilot_pred = np.full((4,), None, dtype=object)
    bad_p.heldout_pilot_var = np.full((4,), 0.1, dtype=np.float64)
    assert _is_valid_output(fr0, bad_p)[0] is False

    frames = [_make_frame(640 + i, snr_db=16.0, split="test", group_id=f"b{i // 2}", frame_id=i) for i in range(4)]
    base_pairs = [(frames[i], _make_valid_output(frames[i], bit_err=100)) for i in range(4)]
    cand_pairs = [
        (frames[0], replace(_make_valid_output(frames[0], bit_err=2841), aux_info={"timeout": True})),
        (frames[1], replace(_make_valid_output(frames[1], bit_err=2841), aux_info={"timeout": True})),
        (frames[2], _make_valid_output(frames[2], bit_err=100)),
        (frames[3], _make_valid_output(frames[3], bit_err=100)),
    ]
    with pytest.raises(ValueError, match="0 valid"):
        compare_arms(base_pairs, cand_pairs, metric="ber", n_boot=50, seed=0)

    # 2. Explicit None values in aux_info must not crash evaluator or JSONL writers
    none_out = _make_valid_output(fr0)
    none_out.aux_info = {
        "weights": None,
        "branch_weights": None,
        "prior_weights": None,
        "omega_t_candidates": None,
        "n_candidates": None,
        "param_count": None,
        "n_values_accessed": None,
        "latency_ms": None,
        "failed": False,
        "error": None,
    }
    ev = evaluate_receiver([(fr0, none_out)])
    assert ev["success_count"] == 1
    for grp, arm in (
        ("classical", "lmmse_1d"),
        ("group_a", "global_content_attn"),
        ("group_b", "exact_mixture_likelihood"),
        ("group_c", "group_c_branch_attn"),
    ):
        rec = run_unified_benchmark._build_per_frame_record(arm, grp, fr0, none_out, param_count=0)
        assert rec["failed"] is False
        json.dumps(rec, allow_nan=False)
    for idx, mod in enumerate((run_baselines, run_group_a, run_group_b, run_group_c)):
        mod._write_per_frame_jsonl(tmp_path / f"m{idx}.jsonl", [(fr0, none_out)])


def test_download_verify_frame0_corrupt_no_nan_and_non_numeric_sinr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import yaml

    def _prep_cfg(ds: DownloadCustomDataset, name: str) -> tuple[Path, Path, Path, Path]:
        pt = tmp_path / name
        torch.save(ds, pt)
        data = pt.read_bytes()
        cfg = yaml.safe_load(Path("research/realdata_sdr_ofdm/configs/default.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["raw_dir"] = str(tmp_path)
        cfg["dataset"]["raw_filename"] = name
        cfg["dataset"]["expected_size_bytes"] = len(data)
        cfg["dataset"]["expected_sha256"] = hashlib.sha256(data).hexdigest()
        cfg["dataset"]["expected_blob_sha1"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        cfg["dataset"]["companion_files"] = []
        cfg_p = tmp_path / f"{name}.yaml"
        cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return pt, cfg_p, tmp_path / f"{name}.man.json", tmp_path / f"{name}.probe.json"

    # Case A: frame 0 corrupt shape must emit valid RFC-8259 JSON (no NaN) and no RuntimeWarnings
    ds_bad0 = DownloadCustomDataset()
    ds_bad0.pdsch_iq = [np.zeros((2, 14, 10), dtype=np.float32)] + [np.zeros((2, 14, 72), dtype=np.float32)] * 15
    ds_bad0.labels = [np.zeros((2, 14, 72), dtype=np.float32)] * 16
    ds_bad0.sinr = [10.0 + i for i in range(16)]
    _, cfg_a, man_a, probe_a = _prep_cfg(ds_bad0, "bad0.pth")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_and_verify.py",
            "--config",
            str(cfg_a),
            "--raw-dir",
            str(tmp_path),
            "--manifest-out",
            str(man_a),
            "--probe-out",
            str(probe_a),
            "--probe-n",
            "16",
            "--skip-download",
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rc_a = download_main()
    assert rc_a == EXIT_FALLBACK_TRIGGERED
    assert not [x for x in w if issubclass(x.category, RuntimeWarning)]
    json.loads(
        probe_a.read_text(encoding="utf-8"),
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")),
    )

    # Case B: non-numeric sinr must trigger T2 fallback exit 42 without unhandled ValueError
    ds_sinr = DownloadCustomDataset()
    ds_sinr.pdsch_iq = np.zeros((16, 2, 14, 72), dtype=np.float32)
    ds_sinr.labels = np.zeros((16, 2, 14, 72), dtype=np.float32)
    ds_sinr.sinr = ["bad"] * 16
    _, cfg_b, man_b, probe_b = _prep_cfg(ds_sinr, "badsinr.pth")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_and_verify.py",
            "--config",
            str(cfg_b),
            "--raw-dir",
            str(tmp_path),
            "--manifest-out",
            str(man_b),
            "--probe-out",
            str(probe_b),
            "--probe-n",
            "16",
            "--skip-download",
        ],
    )
    assert download_main() == EXIT_FALLBACK_TRIGGERED


def test_make_fig1_zero_metrics_and_negative_pilot_nll_no_ylim_warnings(tmp_path: Path) -> None:
    arms_data = {
        arm: {
            "ber": 0.0,
            "fer": 0.0,
            "evm_pct": 0.0,
            "pilot_nll": -0.5,
        }
        for arm in make_figures.ARM_ORDER
    }
    per_frame_by_arm = {
        arm: [
            {
                "frame_id": i,
                "group_id": f"block_{i:02d}",
                "bit_errors": 0,
                "total_bits": 5600,
                "frame_error": 0,
                "evm_pct": 0.0,
                "pilot_nll": -0.5,
            }
            for i in range(4)
        ]
        for arm in make_figures.ARM_ORDER
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_figures.make_fig1_clean_comparison({"arms": arms_data}, per_frame_by_arm, tmp_path)
    ylim_warns = [w for w in caught if "identical low and high ylims" in str(w.message)]
    assert not ylim_warns
    assert (tmp_path / "fig1_clean_comparison.png").exists()
    assert (tmp_path / "fig1_clean_comparison.pdf").exists()


# ---------------------------------------------------------------------------
# 6. Iteration 3 Challenger Regression Tests (Gap 1, Gap 2, Gap 3)
# ---------------------------------------------------------------------------


def test_classical_heldout_cv_zero_leakage_under_1e9_mutation() -> None:
    fr = _make_frame(701, snr_db=18.0, split="test", frame_id=1)
    fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
    fr_mut = copy.deepcopy(fr)
    for idx in (1, 5, 9, 12):
        fr_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e9 + 1e9j

    for m in CLASSICAL_METHODS:
        o_ref = run_classical_frame(fr, method=m)
        o_mut = run_classical_frame(fr_mut, method=m)
        assert not o_ref.aux_info["failed"]
        assert not o_mut.aux_info["failed"]
        np.testing.assert_allclose(o_mut.heldout_pilot_pred, o_ref.heldout_pilot_pred, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(o_mut.heldout_pilot_var, o_ref.heldout_pilot_var, rtol=0.0, atol=1e-13)

    # Also verify 4-fold default CV (fold 0 holds out [1, 5, 9])
    fr_4f = _make_frame(702, snr_db=18.0, split="test", frame_id=2)
    fr_4f.meta.pop("heldout_pilot_indices", None)
    fr_4f_mut = copy.deepcopy(fr_4f)
    for idx in (1, 5, 9):
        fr_4f_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e9 + 1e9j

    for m in CLASSICAL_METHODS:
        o_ref = run_classical_frame(fr_4f, method=m)
        o_mut = run_classical_frame(fr_4f_mut, method=m)
        assert not o_ref.aux_info["failed"]
        assert not o_mut.aux_info["failed"]
        eval_idx = list(o_ref.aux_info["heldout_pilot_indices"])
        pos = [eval_idx.index(pi) for pi in (1, 5, 9)]
        np.testing.assert_allclose(o_mut.heldout_pilot_pred[pos], o_ref.heldout_pilot_pred[pos], rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(o_mut.heldout_pilot_var[pos], o_ref.heldout_pilot_var[pos], rtol=0.0, atol=1e-13)


def test_adapter_temporal_diagnostics_degenerate_and_pilot_cv_bool() -> None:
    p_m, _, _ = build_masks()
    for bad_idx in ([True], [np.bool_(False)], [1.5]):
        with pytest.raises(ValueError):
            pilot_cv_masks(p_m, bad_idx)  # type: ignore[arg-type]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for n_bad in (0, 1):
            with pytest.raises(ValueError, match="at least 2"):
                compute_temporal_diagnostics(
                    np.full(n_bad, 15.0),
                    np.ones((n_bad, 14), dtype=np.complex128),
                    [f"h{i}" for i in range(n_bad)],
                )
        diag_n2 = compute_temporal_diagnostics(
            np.array([15.0, 16.0]),
            np.ones((2, 14), dtype=np.complex128),
            ["h0", "h1"],
        )
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert diag_n2["n_tti"] == 2
    assert math.isfinite(diag_n2["h_pilot_coherent_corr_lags"]["1"])


def test_group_a_config_bounds_empty_fold_rejection_and_duplicate_pilot_dedup() -> None:
    fr = _make_frame(703, snr_db=18.0, split="dev", frame_id=1)
    tr = [_make_frame(710 + i, snr_db=18.0, split="train", frame_id=10 + i) for i in range(2)]
    dv = [_make_frame(715 + i, snr_db=18.0, split="dev", frame_id=15 + i) for i in range(2)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for bad_cfg in (
            GroupAConfig(prior_scale=0.0),
            GroupAConfig(prior_scale=-1.0),
            GroupAConfig(prior_scale=float("nan")),
            GroupAConfig(reg_floor=0.0),
            GroupAConfig(reg_floor=-1e-8),
            GroupAConfig(local_k=True),  # type: ignore[arg-type]
        ):
            assert run_group_a_frame(fr, context_pool=tr, config=bad_cfg).aux_info["failed"] is True
        assert run_group_a_frame(fr, context_pool=tr, local_k=True).aux_info["failed"] is True  # type: ignore[arg-type]
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=1, batch_size=2), seed=19).fit(tr, dv)
    out_cls = run_classical_frame(fr)
    br_stub = [CandidateBranch(h_est=out_cls.h_est.copy(), h_var=out_cls.h_var.copy(), name="b0")]

    # Empty fold rejection across all 4 receivers (including Group B/C branches path)
    fr_efold = copy.deepcopy(fr)
    fr_efold.meta.pop("heldout_pilot_indices", None)
    fr_efold.meta["heldout_pilot_folds"] = [{"heldout_pilot_indices": [], "obs_pilot_indices": [0, 1, 2]}]
    assert run_classical_frame(fr_efold).aux_info["failed"] is True
    assert run_group_a_frame(fr_efold, context_pool=tr).aux_info["failed"] is True
    assert run_group_b_frame(fr_efold).aux_info["failed"] is True
    assert run_group_b_frame(fr_efold, branches=br_stub, branch_weights=[1.0]).aux_info["failed"] is True
    assert rec_c.run_frame(fr_efold).aux_info["failed"] is True
    assert rec_c.run_frame(fr_efold, branches=br_stub).aux_info["failed"] is True

    # Order-preserving deduplication of duplicate heldout_pilot_indices across all 4 receivers (explicit + fold CV)
    fr_dup_exp = copy.deepcopy(fr)
    fr_dup_exp.meta["heldout_pilot_indices"] = [5, 1, 5, 1]
    for out_d in (
        run_classical_frame(fr_dup_exp),
        run_group_a_frame(fr_dup_exp, context_pool=tr),
        run_group_b_frame(fr_dup_exp),
        run_group_b_frame(fr_dup_exp, branches=br_stub, branch_weights=[1.0]),
        rec_c.run_frame(fr_dup_exp),
        rec_c.run_frame(fr_dup_exp, branches=br_stub),
    ):
        assert not out_d.aux_info["failed"]
        assert out_d.aux_info["heldout_pilot_indices"] == [5, 1]
        assert len(out_d.heldout_pilot_pred) == 2
        assert len(out_d.heldout_pilot_var) == 2

    fr_dup_fold = copy.deepcopy(fr)
    fr_dup_fold.meta.pop("heldout_pilot_indices", None)
    fr_dup_fold.meta["heldout_pilot_folds"] = [
        {"heldout_pilot_indices": [5, 1, 5], "obs_pilot_indices": [0, 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]}
    ]
    for out_f in (
        run_classical_frame(fr_dup_fold),
        run_group_a_frame(fr_dup_fold, context_pool=tr),
        run_group_b_frame(fr_dup_fold),
        run_group_b_frame(fr_dup_fold, branches=br_stub, branch_weights=[1.0]),
        rec_c.run_frame(fr_dup_fold),
        rec_c.run_frame(fr_dup_fold, branches=br_stub),
    ):
        assert not out_f.aux_info["failed"]
        assert out_f.aux_info["heldout_pilot_indices"] == [1, 5]
        assert len(out_f.heldout_pilot_pred) == 2
        assert len(out_f.heldout_pilot_var) == 2


# ---------------------------------------------------------------------------
# 7. Iteration 4 Challenger Regression Tests (Gap 1, Gap 2, Gap 3)
# ---------------------------------------------------------------------------


def test_unsorted_obs_pilot_indices_invariant_across_all_receivers() -> None:
    fr_base = _make_frame(801, snr_db=20.0, split="dev", frame_id=1, delay_samples=3.5)
    tr = [_make_frame(810 + i, snr_db=20.0, split="train", frame_id=10 + i, delay_samples=3.5) for i in range(3)]
    dv = [_make_frame(820 + i, snr_db=20.0, split="dev", frame_id=20 + i, delay_samples=3.5) for i in range(2)]
    rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=1, pilot_nll_weight=0.05), seed=7).fit(tr, dv)

    obs_sorted = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
    obs_perm = [13, 0, 12, 2, 11, 3, 10, 4, 9, 6, 8, 7]

    fr_s = copy.deepcopy(fr_base)
    fr_s.meta.pop("heldout_pilot_indices", None)
    fr_s.meta["heldout_pilot_folds"] = [{"heldout_pilot_indices": [1, 5], "obs_pilot_indices": obs_sorted}]
    fr_p = copy.deepcopy(fr_s)
    fr_p.meta["heldout_pilot_folds"] = [{"heldout_pilot_indices": [1, 5], "obs_pilot_indices": obs_perm}]

    for fn in (
        lambda f: run_classical_frame(f, method="ls_linear_zf"),
        lambda f: run_classical_frame(f, method="lmmse_1d", config=ClassicalConfig(detrend_timing=True)),
        lambda f: run_group_a_frame(f, context_pool=tr, config=GroupAConfig(detrend_timing=True)),
        lambda f: run_group_b_frame(f),
        lambda f: rec_c.run_frame(f),
    ):
        o_s = fn(fr_s)
        o_p = fn(fr_p)
        assert not o_s.aux_info["failed"] and not o_p.aux_info["failed"]
        np.testing.assert_allclose(o_p.heldout_pilot_pred, o_s.heldout_pilot_pred, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(o_p.heldout_pilot_var, o_s.heldout_pilot_var, rtol=1e-12, atol=1e-12)


def test_nan_inf_bounds_rejected_modulation_group_bc_and_splits() -> None:
    for bad_bound in (float("nan"), float("inf"), -1.0, 0.0, True):
        with pytest.raises(ValueError):
            constellation_log_likelihood(np.array([0.3 + 0.3j]), 0.1, min_n0=bad_bound)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            exact_llr(np.array([0.3 + 0.3j]), 0.1, min_n0=bad_bound)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            maxlog_llr(np.array([0.3 + 0.3j]), 0.1, min_n0=bad_bound)  # type: ignore[arg-type]
    for bad_max in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            constellation_log_likelihood(np.array([0.3 + 0.3j]), 0.1, max_n0=bad_max)  # type: ignore[arg-type]

    fr = _make_frame(830, snr_db=18.0, split="dev", frame_id=1)
    tr = [_make_frame(831 + i, snr_db=18.0, split="train", frame_id=10 + i) for i in range(2)]
    dv = [_make_frame(835 + i, snr_db=18.0, split="dev", frame_id=20 + i) for i in range(2)]
    rec_c = GroupCTrainableReceiver(GroupCConfig(epochs=1), seed=11).fit(tr, dv)

    for bad_beta in (float("nan"), float("inf"), 0.0, -0.5, True):
        assert run_group_b_frame(fr, config=GroupBConfig(beta=bad_beta)).aux_info["failed"] is True  # type: ignore[arg-type]

    for bad_score in (float("nan"), -float("inf"), float("inf")):
        br_bad = [
            CandidateBranch(name="b0", h_est=np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128), h_var=np.full((N_SYMB, N_ACTIVE), 0.01), log_score=bad_score),
            CandidateBranch(name="b1", h_est=np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128), h_var=np.full((N_SYMB, N_ACTIVE), 0.01), log_score=0.0),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out_bad = rec_c.run_frame(fr, branches=br_bad)
        assert out_bad.aux_info["failed"] is True
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    sinr_bad = np.linspace(5.0, 25.0, 100)
    sinr_bad[10] = float("nan")
    with pytest.raises(ValueError):
        build_leak_free_splits(sinr_bad, np.ones((100, 14), dtype=np.complex128), [f"h_{i}" for i in range(100)])

    h_bad = np.ones((100, 14), dtype=np.complex128)
    h_bad[0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError):
        compute_temporal_diagnostics(np.linspace(5.0, 25.0, 100), h_bad, [f"h_{i}" for i in range(100)])
