"""Data integrity audit, high-SINR 16-QAM mapping verification, temporal coherence audit, and splits export."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ofdm_data import (
    BITS16,
    CONST16,
    DATA_COORDS_101,
    DATA_COORDS_128,
    N_FFT,
    N_SYM,
    PILOT_SCS_14,
    PILOT_SYM,
    PILOT_TX_14,
    USED_SCS,
    build_oracle_channel,
    build_temporal_splits,
    compute_bit_metrics,
    estimate_noise_var_from_grid,
    exact_16qam_llrs,
    load_real_ofdm_dataset,
    ls_linear_channel,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    ds = load_real_ofdm_dataset(args.pth)
    n = len(ds.sinr_db)
    splits = build_temporal_splits(n, ds.sinr_db)
    save_json(splits, args.out_dir / "splits.json")

    dev_idx = np.array(splits["dev"]["tti_indices"], dtype=np.int64)
    # High-SINR verification on Dev top-16, Dev top-32, Dev top-64, plus Full-dataset top-64
    dev_order = dev_idx[np.argsort(-ds.sinr_db[dev_idx])]
    full_order = np.argsort(-ds.sinr_db)

    def _eval_subset(idxs: np.ndarray, use_3gpp_gray_wrong: bool = False) -> dict:
        bers, sers, nlls = [], [], []
        # Optional wrong Gray mapping comparison to demonstrate why mapping verification is essential
        gray_levels = np.array([-3.0, -1.0, 3.0, 1.0]) / math.sqrt(10.0)
        const_gray = np.array([gray_levels[i // 4] + 1j * gray_levels[i % 4] for i in range(16)])
        per_tti = []
        for t in idxs:
            h_ls = ls_linear_channel(ds.iq[t], PILOT_SCS_14)
            n0 = estimate_noise_var_from_grid(ds.iq[t], h_ls, PILOT_SCS_14)
            y_d = ds.iq[t][:, USED_SCS][DATA_COORDS_101[:, 0], DATA_COORDS_101[:, 1]]
            h_d = h_ls[DATA_COORDS_101[:, 1]]
            if not use_3gpp_gray_wrong:
                llrs = exact_16qam_llrs(y_d, h_d, n0)
                bm = compute_bit_metrics(llrs, ds.labels[t])
                bers.append(bm["ber"])
                sers.append(bm["ser"])
                nlls.append(bm["nll"])
                per_tti.append((int(t), float(ds.sinr_db[t]), bm["ber"]))
            else:
                xeq = y_d / h_d
                pred_c = np.argmin(np.abs(xeq[:, None] - const_gray[None, :]) ** 2, axis=1)
                bers.append(float(np.mean(BITS16[pred_c] != ds.labels[t])))
                sers.append(float(np.mean(pred_c != ds.sym_idx[t])))
                nlls.append(float("nan"))
        return {
            "n_ttis": int(len(idxs)),
            "sinr_min_db": round(float(np.min(ds.sinr_db[idxs])), 2),
            "sinr_max_db": round(float(np.max(ds.sinr_db[idxs])), 2),
            "ber_mean": float(np.mean(bers)),
            "ber_median": float(np.median(bers)),
            "ser_mean": float(np.mean(sers)),
            "nll_mean": float(np.nanmean(nlls)) if not use_3gpp_gray_wrong else None,
            "zero_error_ttis": int(sum(1 for b in bers if b == 0.0)),
            "worst_tti": max(per_tti, key=lambda x: x[2]) if per_tti else None,
        }

    # Empirical constellation centroid check on Dev top-32
    top32_dev = dev_order[:32]
    xeq_all = []
    for t in top32_dev:
        h_ls = ls_linear_channel(ds.iq[t], PILOT_SCS_14)
        y_d = ds.iq[t][:, USED_SCS][DATA_COORDS_101[:, 0], DATA_COORDS_101[:, 1]]
        xeq_all.append(y_d / h_ls[DATA_COORDS_101[:, 1]])
    xeq_cat = np.concatenate(xeq_all, axis=0)
    sym_cat = ds.sym_idx[top32_dev].reshape(-1)
    centroids = []
    max_centroid_err = 0.0
    for c in range(16):
        mu = complex(np.mean(xeq_cat[sym_cat == c]))
        err = abs(mu - CONST16[c])
        max_centroid_err = max(max_centroid_err, float(err))
        centroids.append(
            {
                "bits": f"{(c>>3)&1}{(c>>2)&1}{(c>>1)&1}{c&1}",
                "expected": [round(float(CONST16[c].real), 5), round(float(CONST16[c].imag), 5)],
                "empirical_mean": [round(float(mu.real), 5), round(float(mu.imag), 5)],
                "abs_err": round(float(err), 5),
            }
        )

    # Temporal coherence & sync-anomaly scan across all 742 TTIs
    h_oracle = build_oracle_channel(ds)
    h_norm = h_oracle / np.maximum(np.linalg.norm(h_oracle, axis=1, keepdims=True), 1e-12)
    h_mag = np.abs(h_oracle)
    h_mag_c = h_mag - h_mag.mean(axis=0, keepdims=True)
    h_mag_cn = h_mag_c / np.maximum(np.linalg.norm(h_mag_c, axis=1, keepdims=True), 1e-12)
    coherence = {}
    for lag in (1, 2, 5, 10, 15, 20, 30, 50):
        coherence[str(lag)] = {
            "raw_complex_abs_corr": round(float(np.mean(np.abs(np.sum(h_norm[:-lag].conj() * h_norm[lag:], axis=1)))), 4),
            "centered_mag_corr": round(float(np.mean(np.sum(h_mag_cn[:-lag] * h_mag_cn[lag:], axis=1))), 4),
            "sinr_corr": round(float(np.corrcoef(ds.sinr_db[:-lag], ds.sinr_db[lag:])[0, 1]), 4),
        }

    # Pilot smoothness anomaly detection (captures real OTA sync-loss bursts like TTI 655..665)
    p_dr = []
    for t in range(n):
        hp = ds.iq[t, PILOT_SYM, PILOT_SCS_14] / PILOT_TX_14
        dr = float(np.mean(np.abs(np.diff(hp)) ** 2) / max(float(np.mean(np.abs(hp) ** 2)), 1e-12))
        p_dr.append(dr)
    p_dr_arr = np.array(p_dr)
    sync_anomalies = [int(t) for t in np.where(p_dr_arr > 1.0)[0]]

    audit = {
        "dataset_repo": "https://github.com/rikluost/sdr_ofdm_dataset_v1",
        "dataset_commit_sha": "6f9821d6390d212884fea3d6621267ebe70c7246",
        "pth_file": "ofdm_valset_4_8_sdr.pth",
        "pth_size_bytes": ds.file_size,
        "pth_sha256": ds.sha256,
        "n_ttis": n,
        "sinr_db_summary": {
            "min": round(float(ds.sinr_db.min()), 2),
            "p10": round(float(np.percentile(ds.sinr_db, 10)), 2),
            "median": round(float(np.median(ds.sinr_db)), 2),
            "mean": round(float(np.mean(ds.sinr_db)), 2),
            "p90": round(float(np.percentile(ds.sinr_db, 90)), 2),
            "max": round(float(ds.sinr_db.max()), 2),
        },
        "verified_grid": {
            "n_fft": N_FFT,
            "n_sym": N_SYM,
            "cp_len": 6,
            "dc_sc": 64,
            "active_non_dc_count": len(USED_SCS),
            "active_range_incl_dc": [13, 114],
            "pilot_sym": PILOT_SYM,
            "pilot_scs_14": PILOT_SCS_14.tolist(),
            "pilot_amplitude": "0.7*(+-1 +- 1j)",
            "data_re_count": 1400,
            "data_re_ordering": "row_major_(sym, sc)",
            "qam16_mapping": "natural_binary: (b0,b1)->{-3,-1,+1,+3}/sqrt(10) on I, (b2,b3)->{-3,-1,+1,+3}/sqrt(10) on Q",
        },
        "mapping_verification": {
            "dev_top16_natural_binary": _eval_subset(dev_order[:16], False),
            "dev_top32_natural_binary": _eval_subset(dev_order[:32], False),
            "dev_top64_natural_binary": _eval_subset(dev_order[:64], False),
            "dev_top32_standard_3gpp_gray_mismatch": _eval_subset(dev_order[:32], True),
            "full_top64_natural_binary": _eval_subset(full_order[:64], False),
            "full_top63_excl_tti665": _eval_subset(np.array([t for t in full_order[:64] if t != 665]), False),
            "dev_top32_max_constellation_centroid_error": round(max_centroid_err, 5),
            "dev_top32_centroids": centroids,
        },
        "temporal_coherence": coherence,
        "detected_sync_loss_burst_ttis_pilot_dr_gt_1": sync_anomalies,
    }
    save_json(audit, args.out_dir / "mapping_verification.json")
    print(json.dumps(audit["mapping_verification"], indent=2))
    print("Sync anomalies (pilot diff_ratio > 1.0):", sync_anomalies)


if __name__ == "__main__":
    main()
