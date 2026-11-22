"""Unit tests for Milestone 5 delivery scripts (`run_unified_benchmark.py` and `make_figures.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import RealOFDMFrame
from research.realdata_sdr_ofdm.scripts import make_figures, run_unified_benchmark


def _synthetic_frame(frame_id: int = 0, sinr_db: float = 12.5) -> RealOFDMFrame:
    rx_grid = np.zeros((14, 128), dtype=np.complex128)
    pilot_mask = np.zeros((14, 128), dtype=bool)
    dc_mask = np.zeros((14, 128), dtype=bool)
    dc_mask[:, 64] = True
    data_mask = np.zeros((14, 128), dtype=bool)
    data_mask[:, 13:115] = True
    data_mask[:, 64] = False
    pilot_mask[2, :14] = True
    return RealOFDMFrame(
        frame_id=frame_id,
        split="test",
        group_id="block_01",
        rx_grid=rx_grid,
        pilot_mask=pilot_mask,
        dc_mask=dc_mask,
        data_mask=data_mask,
        pilot_values=np.ones(14, dtype=np.complex128),
        tx_bits=np.zeros((1400, 4), dtype=np.uint8),
        tx_symbols=np.ones(1400, dtype=np.complex128),
        reported_sinr_db=sinr_db,
        est_n0=0.05,
        meta={},
    )


def test_sinr_bin_label_boundaries() -> None:
    assert run_unified_benchmark._sinr_bin_label(5.1) == "[5,10)dB"
    assert run_unified_benchmark._sinr_bin_label(10.0) == "[10,15)dB"
    assert run_unified_benchmark._sinr_bin_label(15.0) == "[15,20)dB"
    assert run_unified_benchmark._sinr_bin_label(20.0) == "[20,31]dB"
    assert run_unified_benchmark._sinr_bin_label(30.4) == "[20,31]dB"


def test_resolve_canonical_checkpoint_verifies_sha256(tmp_path: Path) -> None:
    ckpt = tmp_path / "arm_seed1.pt"
    ckpt.write_bytes(b"canonical-payload")
    sha = run_unified_benchmark._sha256_file(ckpt)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "arm": "group_c_branch_attn",
                        "seed": 1,
                        "file": "arm_seed1.pt",
                        "sha256": sha,
                        "dev_ber": 0.07,
                        "dev_pilot_nll": -0.3,
                    },
                    {
                        "arm": "group_c_branch_attn",
                        "seed": 2,
                        "file": "arm_seed2.pt",
                        "sha256": "deadbeef",
                        "dev_ber": 0.08,
                        "dev_pilot_nll": -0.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    best, path = run_unified_benchmark._resolve_canonical_checkpoint(
        "group_c_branch_attn", manifest, tmp_path
    )
    assert best["seed"] == 1
    assert path == ckpt


def test_build_per_frame_record_schema() -> None:
    fr = _synthetic_frame(frame_id=42, sinr_db=16.2)
    out = SimpleNamespace(
        h_est=np.ones((14, 102), dtype=np.complex128),
        h_var=np.ones((14, 102), dtype=np.float64) * 0.01,
        eq_symbols=np.ones(1400, dtype=np.complex128),
        bit_llrs=np.ones((1400, 4), dtype=np.float64),
        hard_bits=np.zeros((1400, 4), dtype=np.uint8),
        hard_symbols=np.ones(1400, dtype=np.complex128),
        heldout_pilot_pred=None,
        heldout_pilot_var=None,
        aux_info={"n_candidates": 6, "latency_ms": 12.3, "failed": False, "error": None},
    )
    rec = run_unified_benchmark._build_per_frame_record(
        "exact_mixture_likelihood", "group_b", fr, out, param_count=0
    )
    assert rec["arm"] == "exact_mixture_likelihood"
    assert rec["group"] == "group_b"
    assert rec["sinr_bin"] == "[15,20)dB"
    assert rec["bit_errors"] == 0
    assert rec["total_bits"] == 5600
    assert rec["failed"] is False


def test_make_figures_renders_valid_png_and_pdf(tmp_path: Path) -> None:
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    res_dir.mkdir()

    # Build minimal synthetic unified results matching schema
    arms_clean = {}
    per_frame_lines = []
    stress_arms = {}
    bins = [
        {"label": "[5,10)dB", "lo_db": 5.0, "hi_db": 10.0, "inclusive_hi": False, "n_frames": 4},
        {"label": "[10,15)dB", "lo_db": 10.0, "hi_db": 15.0, "inclusive_hi": False, "n_frames": 4},
        {"label": "[15,20)dB", "lo_db": 15.0, "hi_db": 20.0, "inclusive_hi": False, "n_frames": 4},
        {"label": "[20,31]dB", "lo_db": 20.0, "hi_db": 31.0, "inclusive_hi": True, "n_frames": 4},
    ]
    sinr_arms = {}

    conds = [
        ("clean", None, 0.0, 0.0),
        ("awgn_10.0db", 10.0, 0.0, 0.0),
        ("awgn_5.0db", 5.0, 0.0, 0.0),
        ("awgn_0.0db", 0.0, 0.0, 0.0),
        ("cfo_0.020", None, 0.02, 0.0),
        ("cfo_0.050", None, 0.05, 0.0),
        ("drift_0.30rad", None, 0.0, 0.3),
        ("drift_0.60rad", None, 0.0, 0.6),
    ]

    for idx, arm in enumerate(make_figures.ARM_ORDER):
        ber = 0.12 + 0.001 * idx
        arms_clean[arm] = {
            "ber": ber,
            "fer": 0.80,
            "evm_pct": 65.0,
            "pilot_nll": 2.0 - 0.05 * idx,
            "mean_ms_per_frame": 5.0 + 3.0 * idx,
            "peak_rss_mb": 350.0 + 5.0 * idx,
            "param_count": 2698 if arm.startswith("group_c") else 0,
        }
        sinr_arms[arm] = {b["label"]: {"ber": ber} for b in bins}
        stress_arms[arm] = {
            "conditions": [
                {
                    "condition": c,
                    "extra_awgn_db": aw,
                    "cfo_norm": cf,
                    "phase_drift_rad": dr,
                    "metrics": {"ber": ber + 0.02 * ci},
                }
                for ci, (c, aw, cf, dr) in enumerate(conds)
            ]
        }
        for fid in range(16):
            per_frame_lines.append(
                json.dumps(
                    {
                        "arm": arm,
                        "group": make_figures.ARM_GROUP[arm],
                        "frame_id": fid,
                        "group_id": f"block_{fid // 6:02d}",
                        "reported_sinr_db": 29.1 if fid == 15 else 10.0,
                        "bit_errors": 2800 if fid == 15 else 500,
                        "total_bits": 5600,
                        "ber": 0.5 if fid == 15 else 0.1,
                        "frame_error": 1,
                        "evm_pct": 65.0,
                        "pilot_nll": 2.0,
                    }
                )
            )

    (res_dir / "summary_clean.json").write_text(
        json.dumps({"arms": arms_clean, "sinr_stratified": {"bins": bins, "arms": sinr_arms}}),
        encoding="utf-8",
    )
    (res_dir / "summary_stress.json").write_text(
        json.dumps({"arms": stress_arms}),
        encoding="utf-8",
    )
    (res_dir / "per_frame_all.jsonl").write_text("\n".join(per_frame_lines) + "\n", encoding="utf-8")

    contrast_stub = {
        "ber_g16": {
            "mean_pp": 0.25,
            "boot_ci95_lo_pp": 0.10,
            "boot_ci95_hi_pp": 0.40,
            "error_delta": -1000,
            "wins": 14,
            "ties": 0,
            "losses": 2,
            "p_two_sided": 0.001,
        },
        "ber_g3": {"cluster_diffs_pp": [0.15, 0.25, 0.35]},
    }
    contrast_keys = [
        "local_vs_lmmse_1d",
        "global_uniform_vs_lmmse_1d",
        "global_reliability_vs_lmmse_1d",
        "global_content_attn_vs_lmmse_1d",
        "global_uniform_vs_local",
        "global_content_attn_vs_local",
        "global_content_attn_vs_global_uniform",
        "global_reliability_vs_global_uniform",
        "hard_map_vs_lmmse_1d",
        "posterior_mean_plugin_vs_lmmse_1d",
        "exact_mixture_likelihood_vs_lmmse_1d",
        "posterior_mean_plugin_vs_hard_map",
        "exact_mixture_likelihood_vs_hard_map",
        "exact_mixture_likelihood_vs_posterior_mean_plugin",
        "group_c_hybrid_attn_vs_exact_mixture_likelihood",
        "group_c_hybrid_attn_vs_group_c_branch_attn",
        "group_c_hybrid_attn_vs_group_c_pilot_nll_only",
    ]
    (res_dir / "paired_comparisons_all.json").write_text(
        json.dumps({"pairwise_key_contrasts": {k: contrast_stub for k in contrast_keys}}),
        encoding="utf-8",
    )

    make_figures.make_fig1_clean_comparison(
        json.loads((res_dir / "summary_clean.json").read_text(encoding="utf-8")),
        make_figures._load_per_frame_grouped(res_dir / "per_frame_all.jsonl"),
        fig_dir,
    )
    make_figures.make_fig2_group_a_b_ablation(
        json.loads((res_dir / "paired_comparisons_all.json").read_text(encoding="utf-8")),
        fig_dir,
    )
    make_figures.make_fig3_stress_robustness(
        json.loads((res_dir / "summary_stress.json").read_text(encoding="utf-8")),
        fig_dir,
    )
    make_figures.make_fig4_sinr_and_cost(
        json.loads((res_dir / "summary_clean.json").read_text(encoding="utf-8")),
        make_figures._load_per_frame_grouped(res_dir / "per_frame_all.jsonl"),
        fig_dir,
    )

    for stem in (
        "fig1_clean_comparison",
        "fig2_group_a_b_ablation",
        "fig3_stress_robustness",
        "fig4_sinr_and_cost",
    ):
        png_bytes = (fig_dir / f"{stem}.png").read_bytes()
        pdf_bytes = (fig_dir / f"{stem}.pdf").read_bytes()
        assert len(png_bytes) > 1024 and png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(pdf_bytes) > 1024 and pdf_bytes.startswith(b"%PDF-")
