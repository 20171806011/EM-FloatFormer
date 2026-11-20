"""P9-A Dev Attribution Study: multi-SNR channel quality x decoder x ML certificate.

Runs on strictly dev-only seeds (SEED_P9_DEV: 4 clusters x 16 links = 64 links =
1,024 common SC codewords per SNR/condition, sparse training budget), across paired
6/9/12/15 dB SNR realizations sharing identical channels, codewords, and unit noise.

Outputs clean JSON to ``p9/results/p9a_dev_attribution.json``.
"""
import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from p9 import channel, decoder
from p9.contract import (
    CORE_ROWS,
    N,
    P8C_BASELINE_COMMIT,
    PRIMARY_SNR_DB,
    SEED_P9_DEV,
    SNR_GRID_DB,
    n0_from_snr,
)


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "p9a_dev_attribution.json"
DEFAULT_CLUSTERS = (1, 2, 3, 4)
DEFAULT_LINKS_PER_CLUSTER = 16


def _paired_counts(base_err: np.ndarray, cand_err: np.ndarray) -> dict[str, Any]:
    """Block-by-block paired contingency table comparing ``cand`` against ``base``."""
    b = np.asarray(base_err, dtype=bool)
    c = np.asarray(cand_err, dtype=bool)
    n = int(len(b))
    rescued = int(np.sum(b & ~c))
    lost = int(np.sum(~b & c))
    both_ok = int(np.sum(~b & ~c))
    both_err = int(np.sum(b & c))
    assert rescued + lost + both_ok + both_err == n
    return {
        "blocks": n,
        "rescued": rescued,
        "lost": lost,
        "both_ok": both_ok,
        "both_err": both_err,
        "net_rescued": rescued - lost,
        "bler_delta_pp": float((c.mean() - b.mean()) * 100.0),
    }


def _evaluate_snr_slice(
    seeds: list[tuple[int, int, int]],
    snr_db: float,
    budget: str = "sparse",
    pic_scale: float = 0.75,
) -> dict[str, Any]:
    """Run paired 2 channel qualities x 3 decoders + ML certificate at one SNR."""
    n_links = len(seeds)
    n_blocks = n_links * len(CORE_ROWS)
    cluster_ids = sorted({c for c, _, _ in seeds})
    cluster_index = {c: idx for idx, c in enumerate(cluster_ids)}
    n_clusters = len(cluster_ids)

    csi_modes = ("true_csi", "est_carry")
    decoders = ("native_ldpc", "strong_small", "strong_large")

    block_err = {csi: {dec: np.zeros(n_blocks, dtype=bool) for dec in decoders} for csi in csi_modes}
    info_err = {csi: {dec: 0 for dec in decoders} for csi in csi_modes}
    cpu_ms = {csi: {dec: 0.0 for dec in decoders} for csi in csi_modes}
    cand_total = {csi: {dec: 0 for dec in decoders} for csi in csi_modes}
    bp_conv_total = {csi: {dec: 0 for dec in decoders} for csi in csi_modes}
    cluster_errs = {
        csi: {dec: [0] * n_clusters for dec in decoders} for csi in csi_modes
    }
    cluster_blocks = [0] * n_clusters

    cert_must_err = np.zeros(n_blocks, dtype=bool)
    cert_by_sf = 0
    cert_by_extra = 0
    cert_invalid_rejected = 0
    cert_checked_total = 0
    cert_time_ms = 0.0

    blk_cursor = 0
    for cluster, _link_idx, seed in seeds:
        c_idx = cluster_index[cluster]
        obs, truth = channel.simulate_link(seed, snr_db=snr_db, regime="continuous", budget=budget)

        # Prepare channel moments for both CSI qualities (est_carry uses strictly public obs!)
        st_a = channel.track_frame_a(obs.y_a, obs.n0)
        post_b = channel.condition_frame_b(st_a, obs.y_b, obs.training_rows, obs.n0)
        est_mean, est_var = channel.sample_moments_b(post_b)

        csi_inputs = {
            "true_csi": (truth.h_b_true, np.zeros_like(truth.h_b_true, dtype=np.float64)),
            "est_carry": (est_mean, est_var),
        }

        true_Hs = channel.build_sc_channel_matrices(truth.h_b_true, CORE_ROWS)
        true_csi_cands: list[list[np.ndarray]] = [[] for _ in range(len(CORE_ROWS))]

        for csi in csi_modes:
            mean_ray, var_ray = csi_inputs[csi]
            Hs = true_Hs if csi == "true_csi" else channel.build_sc_channel_matrices(mean_ray, CORE_ROWS)
            noises = channel.effective_row_noise(var_ray, obs.n0, CORE_ROWS)

            # 1. Native comparator: LMMSE equalization + p8.coding.decode
            t0 = time.perf_counter()
            llrs_mat = np.empty((len(CORE_ROWS), 128), dtype=np.float64)
            for k, r in enumerate(CORE_ROWS):
                llrs_mat[k], _, _ = decoder.lmmse_equalize_row(obs.y_b[r], Hs[k], noises[k])
            native_info = decoder.decode_native_ldpc(llrs_mat)
            cpu_ms[csi]["native_ldpc"] += (time.perf_counter() - t0) * 1000.0

            native_codes = decoder.native_ldpc_codeword(native_info) if csi == "true_csi" else None

            for k, r in enumerate(CORE_ROWS):
                g_idx = blk_cursor + k
                truth_info = truth.b_info[k]
                wrong_bits = int(np.count_nonzero(native_info[k] != truth_info))
                is_err = wrong_bits > 0
                block_err[csi]["native_ldpc"][g_idx] = is_err
                info_err[csi]["native_ldpc"] += wrong_bits
                cand_total[csi]["native_ldpc"] += 1
                if is_err:
                    cluster_errs[csi]["native_ldpc"][c_idx] += 1
                if csi == "true_csi":
                    true_csi_cands[k].append(native_codes[k : k + 1])

                # 2. Strong classical SC decoder: small and large candidate lists
                for dec_name, list_size in (("strong_small", "small"), ("strong_large", "large")):
                    res = decoder.decode_strong_sc(
                        obs.y_b[r],
                        Hs[k],
                        noises[k],
                        list_size=list_size,
                        bp_iters=20,
                        turbo_pic=True,
                        pic_scale=pic_scale,
                    )
                    cpu_ms[csi][dec_name] += res.elapsed_ms
                    cand_total[csi][dec_name] += res.n_candidates
                    bp_conv_total[csi][dec_name] += int(res.bp_converged)
                    w_bits = int(np.count_nonzero(res.info_bits != truth_info))
                    w_err = w_bits > 0
                    block_err[csi][dec_name][g_idx] = w_err
                    info_err[csi][dec_name] += w_bits
                    if w_err:
                        cluster_errs[csi][dec_name][c_idx] += 1
                    if csi == "true_csi":
                        true_csi_cands[k].append(res.candidates_u64)

        # 3. Evaluator-only ML lower-bound certificate on true_csi (receivers NEVER saw c_true!)
        t_cert0 = time.perf_counter()
        for k, r in enumerate(CORE_ROWS):
            g_idx = blk_cursor + k
            cert = decoder.ml_lower_bound_certificate(
                obs.y_b[r],
                true_Hs[k],
                truth.b_code[k],
                extra_candidates=true_csi_cands[k],
            )
            cert_must_err[g_idx] = cert.ml_must_err
            cert_checked_total += cert.n_checked
            cert_invalid_rejected += cert.n_invalid_rejected
            if cert.ml_must_err:
                if cert.witness_source == "single_flip":
                    cert_by_sf += 1
                elif cert.witness_source == "extra_candidate":
                    cert_by_extra += 1
        cert_time_ms += (time.perf_counter() - t_cert0) * 1000.0

        cluster_blocks[c_idx] += len(CORE_ROWS)
        blk_cursor += len(CORE_ROWS)

    # Assemble clean structured metrics
    cells: dict[str, Any] = {}
    for csi in csi_modes:
        cells[csi] = {}
        for dec in decoders:
            errs_n = int(block_err[csi][dec].sum())
            cells[csi][dec] = {
                "blocks": n_blocks,
                "block_errors": errs_n,
                "bler": float(errs_n / n_blocks),
                "info_bits": n_blocks * 64,
                "info_errors": int(info_err[csi][dec]),
                "ber": float(info_err[csi][dec] / (n_blocks * 64)),
                "cpu_ms_total": float(cpu_ms[csi][dec]),
                "cpu_ms_per_1000_blocks": float(cpu_ms[csi][dec] * 1000.0 / n_blocks),
                "avg_candidates_scored": float(cand_total[csi][dec] / n_blocks),
                "bp_converged_rate": (
                    float(bp_conv_total[csi][dec] / n_blocks) if dec != "native_ldpc" else None
                ),
                "cluster_bler": [
                    float(cluster_errs[csi][dec][i] / cluster_blocks[i]) for i in range(n_clusters)
                ],
            }

    paired: dict[str, Any] = {}
    for csi in csi_modes:
        paired[csi] = {
            "strong_small_vs_native": _paired_counts(
                block_err[csi]["native_ldpc"], block_err[csi]["strong_small"]
            ),
            "strong_large_vs_native": _paired_counts(
                block_err[csi]["native_ldpc"], block_err[csi]["strong_large"]
            ),
            "strong_large_vs_small": _paired_counts(
                block_err[csi]["strong_small"], block_err[csi]["strong_large"]
            ),
        }

    # True-CSI attribution & ML certificate decomposition
    nat_err_t = block_err["true_csi"]["native_ldpc"]
    sm_err_t = block_err["true_csi"]["strong_small"]
    lg_err_t = block_err["true_csi"]["strong_large"]
    n_cert_err = int(cert_must_err.sum())
    n_undec = n_blocks - n_cert_err

    cert_summary = {
        "blocks": n_blocks,
        "ml_must_err_count": n_cert_err,
        "ml_lower_bound_rate": float(n_cert_err / n_blocks),
        "undecided_count": n_undec,
        "undecided_rate": float(n_undec / n_blocks),
        "certified_by_single_flip": int(cert_by_sf),
        "certified_by_extra_candidate": int(cert_by_extra),
        "invalid_candidates_rejected": int(cert_invalid_rejected),
        "avg_valid_candidates_checked": float(cert_checked_total / n_blocks),
        "cert_cpu_ms_per_1000_blocks": float(cert_time_ms * 1000.0 / n_blocks),
        "attribution_true_csi": {
            "native_errors": int(nat_err_t.sum()),
            "strong_small_errors": int(sm_err_t.sum()),
            "strong_large_errors": int(lg_err_t.sum()),
            "native_err_and_certified_ml": int(np.sum(nat_err_t & cert_must_err)),
            "strong_large_err_and_certified_ml": int(np.sum(lg_err_t & cert_must_err)),
            "strong_large_err_and_undecided": int(np.sum(lg_err_t & ~cert_must_err)),
            "strong_large_ok_and_certified_ml": int(np.sum(~lg_err_t & cert_must_err)),
            "proven_decoder_suboptimality_rescued_by_large": int(np.sum(nat_err_t & ~lg_err_t)),
            "fraction_of_native_errors_rescued_by_large": (
                float(np.sum(nat_err_t & ~lg_err_t) / nat_err_t.sum())
                if int(nat_err_t.sum()) > 0
                else 0.0
            ),
            "fraction_of_native_errors_certified_ml": (
                float(np.sum(nat_err_t & cert_must_err) / nat_err_t.sum())
                if int(nat_err_t.sum()) > 0
                else 0.0
            ),
        },
        "channel_est_gap_pp": {
            dec: float((cells["est_carry"][dec]["bler"] - cells["true_csi"][dec]["bler"]) * 100.0)
            for dec in decoders
        },
    }

    return {
        "snr_db": float(snr_db),
        "n0": float(n0_from_snr(snr_db)),
        "cells": cells,
        "paired": paired,
        "ml_certificate": cert_summary,
    }


def _run_turbo_pic_ablation(
    seeds: list[tuple[int, int, int]],
    snr_list: tuple[float, ...] = (6.0, 9.0),
    budget: str = "sparse",
    pic_scale: float = 0.75,
) -> dict[str, Any]:
    """Paired dev ablation of Turbo MMSE-PIC (off vs 1 round) on true_csi & est_carry."""
    n_blocks = len(seeds) * len(CORE_ROWS)
    out: dict[str, Any] = {}
    for snr_db in snr_list:
        snr_key = f"{snr_db:.1f}dB"
        out[snr_key] = {}
        for csi in ("true_csi", "est_carry"):
            err_nopic_s = np.zeros(n_blocks, dtype=bool)
            err_pic_s = np.zeros(n_blocks, dtype=bool)
            err_nopic_l = np.zeros(n_blocks, dtype=bool)
            err_pic_l = np.zeros(n_blocks, dtype=bool)
            ms = {"small_nopic": 0.0, "small_pic": 0.0, "large_nopic": 0.0, "large_pic": 0.0}
            cursor = 0
            for _, _, seed in seeds:
                obs, truth = channel.simulate_link(seed, snr_db=snr_db, regime="continuous", budget=budget)
                if csi == "true_csi":
                    mean_ray = truth.h_b_true
                    var_ray = np.zeros_like(mean_ray, dtype=np.float64)
                else:
                    st_a = channel.track_frame_a(obs.y_a, obs.n0)
                    post_b = channel.condition_frame_b(st_a, obs.y_b, obs.training_rows, obs.n0)
                    mean_ray, var_ray = channel.sample_moments_b(post_b)
                Hs = channel.build_sc_channel_matrices(mean_ray, CORE_ROWS)
                noises = channel.effective_row_noise(var_ray, obs.n0, CORE_ROWS)
                for k, r in enumerate(CORE_ROWS):
                    g_idx = cursor + k
                    r_sn = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="small", turbo_pic=False)
                    r_sp = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="small", turbo_pic=True, pic_scale=pic_scale)
                    r_ln = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="large", turbo_pic=False)
                    r_lp = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="large", turbo_pic=True, pic_scale=pic_scale)
                    err_nopic_s[g_idx] = not np.array_equal(r_sn.info_bits, truth.b_info[k])
                    err_pic_s[g_idx] = not np.array_equal(r_sp.info_bits, truth.b_info[k])
                    err_nopic_l[g_idx] = not np.array_equal(r_ln.info_bits, truth.b_info[k])
                    err_pic_l[g_idx] = not np.array_equal(r_lp.info_bits, truth.b_info[k])
                    ms["small_nopic"] += r_sn.elapsed_ms
                    ms["small_pic"] += r_sp.elapsed_ms
                    ms["large_nopic"] += r_ln.elapsed_ms
                    ms["large_pic"] += r_lp.elapsed_ms
                cursor += len(CORE_ROWS)

            out[snr_key][csi] = {
                "small_nopic_bler": float(err_nopic_s.mean()),
                "small_pic_bler": float(err_pic_s.mean()),
                "small_pic_vs_nopic": _paired_counts(err_nopic_s, err_pic_s),
                "large_nopic_bler": float(err_nopic_l.mean()),
                "large_pic_bler": float(err_pic_l.mean()),
                "large_pic_vs_nopic": _paired_counts(err_nopic_l, err_pic_l),
                "cpu_ms_per_1000_blocks": {k: float(v * 1000.0 / n_blocks) for k, v in ms.items()},
            }
    return out


def _run_diagnostics(n_links: int = 16) -> dict[str, Any]:
    """Diagnostic sanity checks: zero-noise 3-ray and single-ray AWGN at 9 dB & 15 dB."""
    n_blocks = n_links * len(CORE_ROWS)
    diag: dict[str, Any] = {}
    for label, kind, snr_db in (
        ("zero_noise_3ray", "zero_noise", 9.0),
        ("awgn_9dB", "awgn", 9.0),
        ("awgn_15dB", "awgn", 15.0),
    ):
        err_nat = err_sm = err_lg = 0
        for link in range(n_links):
            seed = SEED_P9_DEV + 9000 + link
            obs, truth = channel.simulate_diagnostic(seed, kind=kind, snr_db=snr_db, budget="sparse")
            Hs = channel.build_sc_channel_matrices(truth.h_b_true, CORE_ROWS)
            noises = np.full((len(CORE_ROWS), N), obs.n0, dtype=np.float64)
            llrs, _, _ = channel.oracle_readout_b(truth, obs.y_b, obs.training_rows, obs.n0)
            nat_info = decoder.decode_native_ldpc(llrs[list(CORE_ROWS)].reshape(len(CORE_ROWS), 128))
            for k, r in enumerate(CORE_ROWS):
                if not np.array_equal(nat_info[k], truth.b_info[k]):
                    err_nat += 1
                rs = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="small", turbo_pic=True)
                rl = decoder.decode_strong_sc(obs.y_b[r], Hs[k], noises[k], list_size="large", turbo_pic=True)
                if not np.array_equal(rs.info_bits, truth.b_info[k]):
                    err_sm += 1
                if not np.array_equal(rl.info_bits, truth.b_info[k]):
                    err_lg += 1
        diag[label] = {
            "blocks": n_blocks,
            "native_errors": int(err_nat),
            "native_bler": float(err_nat / n_blocks),
            "strong_small_errors": int(err_sm),
            "strong_small_bler": float(err_sm / n_blocks),
            "strong_large_errors": int(err_lg),
            "strong_large_bler": float(err_lg / n_blocks),
        }
    return diag


def run_study(
    clusters: tuple[int, ...] = DEFAULT_CLUSTERS,
    links_per_cluster: int = DEFAULT_LINKS_PER_CLUSTER,
    budget: str = "sparse",
    pic_scale: float = 0.75,
    output_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Execute the complete P9-A Dev Attribution Study and persist clean JSON."""
    decoder.warmup_numba()
    t_start = time.perf_counter()

    seeds = [
        (c, j, int(SEED_P9_DEV + 1000 * c + j))
        for c in clusters
        for j in range(links_per_cluster)
    ]

    snr_results: dict[str, Any] = {}
    for snr_db in SNR_GRID_DB:
        key = f"{snr_db:.1f}dB"
        snr_results[key] = _evaluate_snr_slice(seeds, snr_db=snr_db, budget=budget, pic_scale=pic_scale)

    turbo_ablation = _run_turbo_pic_ablation(seeds, snr_list=(6.0, PRIMARY_SNR_DB), budget=budget, pic_scale=pic_scale)
    diagnostics = _run_diagnostics(n_links=16)
    mini_check = decoder.verify_mini_codebook_ml_certificate(n_trials=240, snr_db=PRIMARY_SNR_DB, seed=910999)

    total_wall_s = time.perf_counter() - t_start
    report = {
        "study": "P9-A Dev Attribution Study (Multi-SNR CSI x Decoder x ML Lower-Bound Certificate)",
        "baseline_commit": P8C_BASELINE_COMMIT,
        "seed_namespace": int(SEED_P9_DEV),
        "clusters": list(clusters),
        "links_per_cluster": int(links_per_cluster),
        "total_links": len(seeds),
        "blocks_per_link": len(CORE_ROWS),
        "blocks_per_cell": len(seeds) * len(CORE_ROWS),
        "budget": budget,
        "snr_grid_db": list(SNR_GRID_DB),
        "primary_snr_db": float(PRIMARY_SNR_DB),
        "decoder_config": {
            "bp_iters": 20,
            "turbo_pic": True,
            "pic_scale": float(pic_scale),
            "small_list_spec": "order-0 (1) + order-1 (64) + top-256 order-2 = 321 (+1 if BP-0 converged)",
            "large_list_spec": "order-0 (1) + order-1 (64) + all 2016 order-2 + 560 order-3 (16 weakest MRB) = 2641 (+1 if BP-0 converged)",
            "waveform_metric_note": (
                "Exact ML metric ||y_r - H_r x(c)||^2 / n0 under true_csi; "
                "plug-in approximate diagonal-variance waveform metric under est_carry."
            ),
        },
        "snr_slices": snr_results,
        "turbo_pic_ablation": turbo_ablation,
        "diagnostics": diagnostics,
        "mini_codebook_verification": mini_check,
        "total_wall_seconds": float(total_wall_s),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"P9-A Dev Attribution Study ({report['total_links']} links = {report['blocks_per_cell']} blocks/cell, "
        f"budget={report['budget']}, wall={report['total_wall_seconds']:.2f}s)"
    )
    print("-" * 112)
    print(
        f"{'SNR':>6s} | {'CSI':>9s} | {'Native BLER':>12s} | {'Small BLER':>12s} | {'Large BLER':>12s} | "
        f"{'Small ms/1k':>11s} | {'Large ms/1k':>11s} | {'ML LB%':>7s} | {'Undec%':>7s}"
    )
    print("-" * 112)
    for snr_key, sl in report["snr_slices"].items():
        cert = sl["ml_certificate"]
        for csi in ("true_csi", "est_carry"):
            c = sl["cells"][csi]
            ml_s = f"{cert['ml_lower_bound_rate']*100:6.2f}%" if csi == "true_csi" else "      -"
            un_s = f"{cert['undecided_rate']*100:6.2f}%" if csi == "true_csi" else "      -"
            print(
                f"{snr_key:>6s} | {csi:>9s} | "
                f"{c['native_ldpc']['bler']*100:6.2f}% ({c['native_ldpc']['block_errors']:3d}) | "
                f"{c['strong_small']['bler']*100:6.2f}% ({c['strong_small']['block_errors']:3d}) | "
                f"{c['strong_large']['bler']*100:6.2f}% ({c['strong_large']['block_errors']:3d}) | "
                f"{c['strong_small']['cpu_ms_per_1000_blocks']:11.1f} | "
                f"{c['strong_large']['cpu_ms_per_1000_blocks']:11.1f} | "
                f"{ml_s:>7s} | {un_s:>7s}"
            )
    print("-" * 112)
    s9 = report["snr_slices"]["9.0dB"]
    p9_t = s9["paired"]["true_csi"]
    p9_e = s9["paired"]["est_carry"]
    c9 = s9["ml_certificate"]
    print(
        f"9 dB true_csi paired: large_vs_native rescued={p9_t['strong_large_vs_native']['rescued']} "
        f"lost={p9_t['strong_large_vs_native']['lost']} net={p9_t['strong_large_vs_native']['net_rescued']:+d}; "
        f"large_vs_small rescued={p9_t['strong_large_vs_small']['rescued']} "
        f"lost={p9_t['strong_large_vs_small']['lost']} net={p9_t['strong_large_vs_small']['net_rescued']:+d}"
    )
    print(
        f"9 dB est_carry paired: large_vs_native rescued={p9_e['strong_large_vs_native']['rescued']} "
        f"lost={p9_e['strong_large_vs_native']['lost']} net={p9_e['strong_large_vs_native']['net_rescued']:+d}; "
        f"large_vs_small rescued={p9_e['strong_large_vs_small']['rescued']} "
        f"lost={p9_e['strong_large_vs_small']['lost']} net={p9_e['strong_large_vs_small']['net_rescued']:+d}"
    )
    print(
        f"9 dB ML certificate: must_err={c9['ml_must_err_count']}/{c9['blocks']} "
        f"({c9['ml_lower_bound_rate']*100:.2f}%), undecided={c9['undecided_count']}/{c9['blocks']} "
        f"({c9['undecided_rate']*100:.2f}%), invalid_rejected={c9['invalid_candidates_rejected']}"
    )
    mini = report["mini_codebook_verification"]
    print(
        f"Mini (16,8) exhaustive check: all_passed={mini['all_passed']}, d_min={mini['d_min']}, "
        f"partial_cert={mini['partial_cert_rate']*100:.2f}% <= exact_ml={mini['exact_ml_rate']*100:.2f}% "
        f"<= lmmse_hard={mini['lmmse_hard_rate']*100:.2f}%, violations={mini['false_cert_violations']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", nargs="+", type=int, default=list(DEFAULT_CLUSTERS))
    parser.add_argument("--links-per-cluster", type=int, default=DEFAULT_LINKS_PER_CLUSTER)
    parser.add_argument("--budget", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--pic-scale", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    report = run_study(
        clusters=tuple(args.clusters),
        links_per_cluster=args.links_per_cluster,
        budget=args.budget,
        pic_scale=args.pic_scale,
        output_path=args.output,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
