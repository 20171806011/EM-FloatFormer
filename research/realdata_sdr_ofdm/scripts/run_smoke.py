#!/usr/bin/env python3
"""Build leak-free splits, run 64-frame smoke check, and execute full dataset preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.realdata_sdr_ofdm.adapters.modulation import exact_llr, symbols_to_hard_bits
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    PILOT_ACTIVE_INDICES,
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    PILOT_VALUES,
    active_rx_grid,
    apply_stress_perturbation,
    build_frame,
    build_leak_free_splits,
    load_processed_split,
    load_raw_sdr_ofdm,
    preprocess_and_save,
)


def _clean_ls_zf_ber(frame: Any) -> float:
    h_p = frame.rx_grid[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)] / PILOT_VALUES
    x_act = np.arange(102, dtype=np.float64)
    h_re = np.interp(x_act, np.asarray(PILOT_ACTIVE_INDICES, dtype=np.float64), h_p.real)
    h_im = np.interp(x_act, np.asarray(PILOT_ACTIVE_INDICES, dtype=np.float64), h_p.imag)
    h_102 = np.broadcast_to((h_re + 1j * h_im)[None, :], (14, 102))
    data_mask_102 = frame.data_mask[:, 13:115]
    y_data = active_rx_grid(frame)[data_mask_102]
    h_data = h_102[data_mask_102]
    z_eq = y_data / np.where(np.abs(h_data) > 1e-12, h_data, 1e-12)
    hard_bits = symbols_to_hard_bits(z_eq, qm=4)
    n0_eff = frame.est_n0 / np.maximum(np.abs(h_data) ** 2, 1e-12)
    llrs = exact_llr(z_eq, n0_eff, qm=4)
    assert np.all(np.isfinite(llrs)), "Non-finite LLRs in smoke check"
    return float(np.mean(hard_bits != frame.tx_bits))


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 split builder, smoke test, and full preprocessing.")
    parser.add_argument("--config", default="research/realdata_sdr_ofdm/configs/default.yaml")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--proc-dir", default=None)
    parser.add_argument("--splits-out", default="research/realdata_sdr_ofdm/manifests/splits.json")
    parser.add_argument("--proc-checksums-out", default="research/realdata_sdr_ofdm/manifests/proc_checksums.json")
    parser.add_argument("--metadata-out", default="research/realdata_sdr_ofdm/manifests/unit_metadata.jsonl")
    parser.add_argument("--build-splits", action="store_true")
    parser.add_argument("--smoke-n", type=int, default=0)
    parser.add_argument("--full-preprocess", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds_cfg = cfg["dataset"]
    sp_cfg = cfg["splits"]
    raw_path = Path(args.raw_path or (Path(ds_cfg["raw_dir"]) / ds_cfg["raw_filename"]))
    proc_dir = Path(args.proc_dir or ds_cfg["proc_dir"])
    outputs_dir = Path(ds_cfg["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not (args.build_splits or args.smoke_n > 0 or args.full_preprocess):
        args.build_splits = True
        args.smoke_n = 64
        args.full_preprocess = True

    if args.build_splits:
        raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        iq_list, lab_list, sinr_arr = load_raw_sdr_ofdm(raw_path)
        n = len(iq_list)
        payload_hashes: list[str] = []
        h_pilots = np.empty((n, 14), dtype=np.complex128)
        for i in range(n):
            fr = build_frame(
                frame_id=i,
                iq_raw=iq_list[i],
                labels_raw=lab_list[i],
                sinr_val=float(sinr_arr[i]),
                raw_source_sha256=raw_sha256,
            )
            payload_hashes.append(fr.meta["tx_bits_sha256"])
            h_pilots[i] = fr.rx_grid[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)] / PILOT_VALUES

        splits_manifest = build_leak_free_splits(
            sinr_vals=sinr_arr,
            h_pilots=h_pilots,
            payload_hashes=payload_hashes,
            dataset_sha256=raw_sha256,
            n_macro_blocks=int(sp_cfg.get("n_macro_blocks", 12)),
            ratios=dict(sp_cfg.get("ratios", {"train": 0.60, "dev": 0.15, "test": 0.25})),
            min_guard_frames=int(sp_cfg.get("min_guard_frames", 4)),
            max_guard_frames=int(sp_cfg.get("max_guard_frames", 16)),
            autocorr_threshold=float(sp_cfg.get("autocorr_threshold", 0.10)),
            heldout_pilot_eval_indices=list(sp_cfg.get("heldout_pilot_eval_indices", [1, 5, 9, 12])),
            seed=int(sp_cfg.get("seed", 20261121)),
        )
        splits_out = Path(args.splits_out)
        splits_out.parent.mkdir(parents=True, exist_ok=True)
        splits_out.write_text(json.dumps(splits_manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("SPLITS_BUILT:", json.dumps(
            {
                "counts": splits_manifest["counts"],
                "n_unique_payloads": splits_manifest["diagnostics"]["n_unique_payloads"],
                "n_duplicate_groups": splits_manifest["diagnostics"]["n_duplicate_groups"],
                "tau_c": splits_manifest["diagnostics"]["tau_c"],
                "guard_width": splits_manifest["diagnostics"]["guard_width"],
                "min_cross_split_distance": splits_manifest["diagnostics"]["min_cross_split_distance"],
                "sinr_by_split": splits_manifest["sinr_by_split"],
            },
            indent=2,
        ))

    if args.smoke_n > 0:
        iq_list, lab_list, sinr_arr = load_raw_sdr_ofdm(raw_path, max_frames=args.smoke_n)
        bers = []
        pert_bers = []
        est_n0s = []
        pert_n0s = []
        for i in range(len(iq_list)):
            fr = build_frame(
                frame_id=i,
                iq_raw=iq_list[i],
                labels_raw=lab_list[i],
                sinr_val=float(sinr_arr[i]),
            )
            bers.append(_clean_ls_zf_ber(fr))
            est_n0s.append(fr.est_n0)
            fr_p = apply_stress_perturbation(fr, extra_awgn_db=10.0, cfo_norm=0.02, phase_drift_rad=0.1, seed=100 + i)
            assert np.array_equal(fr_p.tx_bits, fr.tx_bits)
            pert_bers.append(_clean_ls_zf_ber(fr_p))
            pert_n0s.append(fr_p.est_n0)
        smoke_summary = {
            "smoke_n": len(iq_list),
            "mean_ls_zf_ber": float(np.mean(bers)),
            "mean_stressed_ls_zf_ber": float(np.mean(pert_bers)),
            "mean_est_n0": float(np.mean(est_n0s)),
            "mean_stressed_est_n0": float(np.mean(pert_n0s)),
        }
        (outputs_dir / "smoke_summary.json").write_text(
            json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print("SMOKE_OK:", json.dumps(smoke_summary, indent=2))

    if args.full_preprocess:
        proc_manifest = preprocess_and_save(
            raw_path=raw_path,
            splits_manifest_path=args.splits_out,
            proc_dir=proc_dir,
            metadata_jsonl_path=args.metadata_out,
            proc_checksums_path=args.proc_checksums_out,
        )
        test_frames = load_processed_split(proc_dir, split="test")
        assert len(test_frames) > 0, "Empty test split after preprocess"
        print("FULL_PREPROCESS_OK:", json.dumps(
            {
                "n_total_processed": proc_manifest["n_total_processed"],
                "loaded_test_frames": len(test_frames),
                "files": {k: {"size_bytes": v["size_bytes"], "sha256": v["sha256"], "n_frames": v["n_frames"]} for k, v in proc_manifest["files"].items()},
            },
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
