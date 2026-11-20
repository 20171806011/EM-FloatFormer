"""Phase 11 Primary Axis Development Benchmark (strictly SEED_P11_DEV only).

Evaluates 6 paired arms across 5 regimes on SEED_P11_DEV (4 clusters x 16 links = 64 links
= 1,024 core blocks per regime, 9.0 dB, explicit n0 = n0_from_snr(9.0)):
  1. sparse_always_inherit            (P9 baseline, strong_large)
  2. sparse_jeffreys_student_t        (P9 classical gate, strong_large)
  3. sparse_payload_rebuild_pilot     (GL3 B-only pilot-only reset comparator)
  4. sparse_payload_rebuild_standalone(GL3 fold-isolated payload_rebuild without wavefront)
  5. p11_primary                      (Chained payload_rebuild + shift_repair + wavefront)
  6. dense_always_clear               (Paired engineering non-inferiority target)

Reports:
  - Pooled 16-core-row BLER per regime and 3-ID equal-weight BLER
  - Continuous non-inferiority (df = n_clusters - 1 one-sided 95% t upper bound < +1.00 pp)
  - Common-phase blind-spot breakdown (`p11_primary` vs `sparse_payload_rebuild_standalone`)
    plus post-hoc `phase_check.metrics` (`abs_phase_rmse`, `rel_phase_rmse`, `abs_freq_rmse`, `rel_freq_rmse`)
  - Ledger-based injected-word false assimilation vs final decoded_info mismatches
  - Wave-1 unlock rate (`correct`, `incorrect`, `absent`) & conditional target BLER
  - Probe calibration (`probe_nmse_noisy/clean`, `probe_cal_ratio`, `probe_nll_per_sample`, `min_probe_var`)
  - Warm ms/link per regime and static AST checks (`check_no_hardcoded_n0`, `inference_mode == []`)

STRICTLY DEV-ONLY: NEVER touches formal seeds. Writes `p11/results/p11_dev.json`.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t as student_t

from p8c.link import extra_rows
from p9.channel import build_sc_channel_matrix, effective_row_noise, simulate_link
from p9.decoder import decode_strong_sc
from p9.gating import (
    build_candidates_with_priors,
    gate_always_clear,
    gate_always_inherit,
    gate_jeffreys_student_t,
)
from p9.independent_ref import check_no_hardcoded_n0
from p10.payload_reconstruction.phase_check import metrics as phase_metrics
from p11.contract import (
    ANCHOR,
    CORE_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    SEED_P11_DEV,
    T,
    HiddenTruth,
    PublicObservation,
    enforce_determinism,
    is_historical_seed,
    n0_from_snr,
)
from p11.payload_rebuild import estimate_payload_posterior
from p11.wavefront_pipeline import (
    AUX_FOLDS,
    FOLD_TARGETS,
    FROZEN_TAU_SWITCH,
    _readout_rows_llrs,
    run_p11_with_trace,
    warmup_p11_numba,
)


DEV_SEED_BASE = SEED_P11_DEV + 200_000
N_CLUSTERS = 4
LINKS_PER_CLUSTER = 16
DEV_REGIMES = REGIMES_ID + REGIMES_OOD
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "p11_dev.json"

ARMS = (
    "sparse_always_inherit",
    "sparse_jeffreys_student_t",
    "sparse_payload_rebuild_pilot",
    "sparse_payload_rebuild_standalone",
    "p11_primary",
    "dense_always_clear",
)


def _new_accum() -> dict:
    return dict(
        links=0,
        core_blocks=0,
        core_errors=0,
        extra_blocks=0,
        extra_errors=0,
        info_bits=0,
        info_errors=0,
        total_assimilations=0,
        false_assimilations_final=0,
        false_assimilations_injected=0,
        w1_links_correct=0,
        w1_links_incorrect=0,
        w1_links_absent=0,
        w1_target_err_correct=0,
        w1_target_blk_correct=0,
        w1_target_err_incorrect=0,
        w1_target_blk_incorrect=0,
        w1_target_err_absent=0,
        w1_target_blk_absent=0,
        probe_sse_noisy=0.0,
        probe_energy_noisy=0.0,
        probe_sse_clean=0.0,
        probe_energy_clean=0.0,
        probe_var_sum=0.0,
        probe_min_var=float("inf"),
        probe_nll_sum=0.0,
        probe_samples=0,
        weight_sum=0.0,
        probe_weight_sum=0.0,
        abs_phase_sse=0.0,
        abs_phase_w=0.0,
        rel_phase_sse=0.0,
        rel_phase_w=0.0,
        abs_freq_sse=0.0,
        abs_freq_w=0.0,
        rel_freq_sse=0.0,
        rel_freq_w=0.0,
        elapsed_ms=0.0,
    )


def _row_to_true_info(truth: HiddenTruth, training_rows: tuple[int, ...]) -> dict[int, np.ndarray]:
    mapping = {int(r): truth.b_info[idx] for idx, r in enumerate(CORE_ROWS)}
    for idx_e, r in enumerate(extra_rows(training_rows)):
        mapping[int(r)] = truth.b_info[len(CORE_ROWS) + idx_e]
    return mapping


def _decode_baseline_strong_large(
    obs: PublicObservation,
    mean_ray: np.ndarray,
    var_ray: np.ndarray,
    llrs: np.ndarray,
) -> tuple[dict[int, np.ndarray], float]:
    t0 = time.perf_counter()
    free_rows = tuple(int(r) for r in extra_rows(obs.training_rows))
    data_rows = tuple(int(r) for r in CORE_ROWS) + free_rows
    decoded: dict[int, np.ndarray] = {}
    for r in data_rows:
        h_r = build_sc_channel_matrix(mean_ray[:, r])
        noise_r = effective_row_noise(var_ray, obs.n0, rows=(r,))[0]
        res = decode_strong_sc(
            obs.y_b[r],
            h_r,
            noise_r,
            list_size="large",
            bp_iters=20,
            turbo_pic=True,
            pic_scale=0.75,
            initial_llrs=llrs[r].reshape(128),
        )
        decoded[r] = res.info_bits
    return decoded, (time.perf_counter() - t0) * 1000.0


def _run_standalone_payload_rebuild(
    obs: PublicObservation,
    *,
    use_payload: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fold-isolated standalone GL3 payload_rebuild comparator (no wavefront assimilation)."""
    t0 = time.perf_counter()
    free_rows = tuple(int(r) for r in extra_rows(obs.training_rows))
    if use_payload:
        p0 = estimate_payload_posterior(obs, AUX_FOLDS[0], use_payload=True)
        p1 = estimate_payload_posterior(obs, AUX_FOLDS[1], use_payload=True)
    else:
        p0 = estimate_payload_posterior(obs, (), use_payload=False)
        p1 = p0

    mean_ray = np.zeros_like(p0.mean)
    var_ray = np.zeros_like(p0.var)
    llrs = np.zeros((T, obs.y_b.shape[1], 4), dtype=np.float64)

    for f_res, owner_rows in ((p0, FOLD_TARGETS[0] + free_rows), (p1, FOLD_TARGETS[1])):
        mean_ray[:, list(owner_rows), :] = f_res.mean[:, list(owner_rows), :]
        var_ray[:, list(owner_rows), :] = f_res.var[:, list(owner_rows), :]
        llrs[list(owner_rows)] = _readout_rows_llrs(
            f_res.mean[:, list(owner_rows), :],
            f_res.var[:, list(owner_rows), :],
            obs.y_b[list(owner_rows)],
            obs.n0,
        )

    probe_pred = 0.5 * (p0.probe_pred + p1.probe_pred)
    probe_var = 0.5 * (p0.probe_var + p1.probe_var) + 0.25 * (np.abs(p0.probe_pred - p1.probe_pred) ** 2)
    decoded, _ = _decode_baseline_strong_large(obs, mean_ray, var_ray, llrs)
    return decoded, mean_ray, var_ray, probe_pred, probe_var, (time.perf_counter() - t0) * 1000.0


def _score_into(
    acc: dict,
    decoded_info: dict[int, np.ndarray],
    mean_ray: np.ndarray,
    probe_pred: np.ndarray,
    probe_var: np.ndarray,
    history_weight: np.ndarray,
    probe_weight: float,
    truth: HiddenTruth,
    training_rows: tuple[int, ...],
    n0: float,
    elapsed_ms: float,
    *,
    assimilated_rows: tuple[int, ...] = (),
    ledger: list[dict] | None = None,
) -> tuple[int, int]:
    pred = np.asarray(probe_pred, dtype=np.complex128)
    pvar = np.asarray(probe_var, dtype=np.float64)
    if not (np.isfinite(pred).all() and np.isfinite(pvar).all() and np.all(pvar >= float(n0) - 1e-12)):
        raise ValueError("Non-finite probe prediction or variance below n0")

    true_map = _row_to_true_info(truth, training_rows)
    core_err = 0
    info_err = 0
    for idx_c, r in enumerate(CORE_ROWS):
        diff = decoded_info[int(r)] != truth.b_info[idx_c]
        if bool(diff.any()):
            core_err += 1
            info_err += int(diff.sum())

    free_rows = tuple(int(r) for r in extra_rows(training_rows))
    ext_err = 0
    for idx_e, r in enumerate(free_rows):
        if not np.array_equal(decoded_info[r], truth.b_info[len(CORE_ROWS) + idx_e]):
            ext_err += 1

    false_ass_final = 0
    for r in assimilated_rows:
        if int(r) == PROBE_ROW or int(r) in training_rows:
            raise AssertionError(f"Illegal assimilated row {r}")
        if not np.array_equal(decoded_info[int(r)], true_map[int(r)]):
            false_ass_final += 1

    false_ass_inj = 0
    tot_inj = 0
    if ledger is not None:
        tot_inj = len(ledger)
        for rec in ledger:
            r = int(rec["row"])
            if not np.array_equal(rec["injected_info"], true_map[r]):
                false_ass_inj += 1

        # Wave-1 conditional diagnostics (wave == 0)
        w1_recs = [rec for rec in ledger if int(rec["wave"]) == 0]
        if not w1_recs:
            acc["w1_links_absent"] += 1
            acc["w1_target_err_absent"] += core_err
            acc["w1_target_blk_absent"] += len(CORE_ROWS)
        elif all(np.array_equal(rec["injected_info"], true_map[int(rec["row"])]) for rec in w1_recs):
            acc["w1_links_correct"] += 1
            acc["w1_target_err_correct"] += core_err
            acc["w1_target_blk_correct"] += len(CORE_ROWS)
        else:
            acc["w1_links_incorrect"] += 1
            acc["w1_target_err_incorrect"] += core_err
            acc["w1_target_blk_incorrect"] += len(CORE_ROWS)

    res_noisy = np.abs(truth.y_b_probe_noisy - pred) ** 2
    res_clean = np.abs(truth.y_b_probe_clean - pred) ** 2

    pm = phase_metrics(mean_ray, truth.h_b_true, rows=CORE_ROWS)
    acc["abs_phase_sse"] += float(pm["abs_phase_sse"])
    acc["abs_phase_w"] += float(pm["abs_phase_weight"])
    acc["rel_phase_sse"] += float(pm["rel_phase_sse"])
    acc["rel_phase_w"] += float(pm["rel_phase_weight"])
    acc["abs_freq_sse"] += float(pm["abs_freq_sse"])
    acc["abs_freq_w"] += float(pm["abs_freq_weight"])
    acc["rel_freq_sse"] += float(pm["rel_freq_sse"])
    acc["rel_freq_w"] += float(pm["rel_freq_weight"])

    acc["links"] += 1
    acc["core_blocks"] += len(CORE_ROWS)
    acc["core_errors"] += core_err
    acc["extra_blocks"] += len(free_rows)
    acc["extra_errors"] += ext_err
    acc["info_bits"] += len(CORE_ROWS) * 64
    acc["info_errors"] += info_err
    acc["total_assimilations"] += tot_inj if ledger is not None else len(assimilated_rows)
    acc["false_assimilations_final"] += false_ass_final
    acc["false_assimilations_injected"] += false_ass_inj
    acc["probe_sse_noisy"] += float(np.sum(res_noisy))
    acc["probe_energy_noisy"] += float(np.sum(np.abs(truth.y_b_probe_noisy) ** 2))
    acc["probe_sse_clean"] += float(np.sum(res_clean))
    acc["probe_energy_clean"] += float(np.sum(np.abs(truth.y_b_probe_clean) ** 2))
    acc["probe_var_sum"] += float(np.sum(pvar))
    acc["probe_min_var"] = min(acc["probe_min_var"], float(np.min(pvar)))
    acc["probe_nll_sum"] += float(np.sum(np.log(pvar / float(n0)) + res_noisy / pvar))
    acc["probe_samples"] += int(len(pred))
    acc["weight_sum"] += float(np.mean(history_weight))
    acc["probe_weight_sum"] += float(probe_weight)
    acc["elapsed_ms"] += float(elapsed_ms)
    return core_err, len(CORE_ROWS)


def _finalize(acc: dict) -> dict:
    links = max(acc["links"], 1)
    core_blocks = max(acc["core_blocks"], 1)
    core_ok = acc["core_blocks"] - acc["core_errors"]
    extra_ok = acc["extra_blocks"] - acc["extra_errors"]
    total_ok = core_ok + extra_ok
    tot_ass = acc["total_assimilations"]
    out = dict(
        links=acc["links"],
        core_blocks=acc["core_blocks"],
        core_errors=acc["core_errors"],
        bler=acc["core_errors"] / core_blocks,
        extra_blocks=acc["extra_blocks"],
        extra_errors=acc["extra_errors"],
        extra_bler=(acc["extra_errors"] / acc["extra_blocks"]) if acc["extra_blocks"] > 0 else 0.0,
        info_ber=acc["info_errors"] / max(acc["info_bits"], 1),
        net_delivered_blocks_per_frame=total_ok / links,
        net_delivered_info_bits_per_frame=(total_ok * 64.0) / links,
        total_assimilations=tot_ass,
        assimilated_rows_per_link=tot_ass / links,
        false_assimilations_final=acc["false_assimilations_final"],
        false_assimilations_injected=acc["false_assimilations_injected"],
        false_assimilation_rate_injected=(acc["false_assimilations_injected"] / tot_ass) if tot_ass > 0 else 0.0,
        probe_nmse_noisy=acc["probe_sse_noisy"] / max(acc["probe_energy_noisy"], 1e-300),
        probe_nmse_clean=acc["probe_sse_clean"] / max(acc["probe_energy_clean"], 1e-300),
        probe_cal_ratio=acc["probe_sse_noisy"] / max(acc["probe_var_sum"], 1e-300),
        probe_min_var=acc["probe_min_var"] if math.isfinite(acc["probe_min_var"]) else 0.0,
        probe_nll_per_sample=acc["probe_nll_sum"] / max(acc["probe_samples"], 1),
        mean_history_weight=acc["weight_sum"] / links,
        mean_probe_weight=acc["probe_weight_sum"] / links,
        abs_phase_rmse=math.sqrt(acc["abs_phase_sse"] / max(acc["abs_phase_w"], 1e-300)),
        rel_phase_rmse=math.sqrt(acc["rel_phase_sse"] / max(acc["rel_phase_w"], 1e-300)),
        abs_freq_rmse=math.sqrt(acc["abs_freq_sse"] / max(acc["abs_freq_w"], 1e-300)),
        rel_freq_rmse=math.sqrt(acc["rel_freq_sse"] / max(acc["rel_freq_w"], 1e-300)),
        ms_per_link=acc["elapsed_ms"] / links,
    )
    if acc["w1_links_correct"] + acc["w1_links_incorrect"] + acc["w1_links_absent"] > 0:
        out["wave1_diagnostics"] = dict(
            links_correct=acc["w1_links_correct"],
            links_incorrect=acc["w1_links_incorrect"],
            links_absent=acc["w1_links_absent"],
            rate_correct=acc["w1_links_correct"] / links,
            target_bler_if_correct=(acc["w1_target_err_correct"] / acc["w1_target_blk_correct"]) if acc["w1_target_blk_correct"] > 0 else None,
            target_bler_if_incorrect=(acc["w1_target_err_incorrect"] / acc["w1_target_blk_incorrect"]) if acc["w1_target_blk_incorrect"] > 0 else None,
            target_bler_if_absent=(acc["w1_target_err_absent"] / acc["w1_target_blk_absent"]) if acc["w1_target_blk_absent"] > 0 else None,
        )
    return out


def _paired_noninferiority_stats(
    sparse_cluster_blers: list[float],
    dense_cluster_blers: list[float],
    margin_pp: float = 1.00,
) -> dict:
    diffs_pp = [100.0 * (s - d) for s, d in zip(sparse_cluster_blers, dense_cluster_blers)]
    g_cnt = len(diffs_pp)
    mean_pp = float(np.mean(diffs_pp))
    if g_cnt >= 2:
        sd_pp = float(np.std(diffs_pp, ddof=1))
        se_pp = sd_pp / math.sqrt(g_cnt)
        t_crit_1s = float(student_t.ppf(0.95, df=g_cnt - 1))
        t_crit_2s = float(student_t.ppf(0.975, df=g_cnt - 1))
        upper_95_1s = mean_pp + t_crit_1s * se_pp
        ci95 = [mean_pp - t_crit_2s * se_pp, mean_pp + t_crit_2s * se_pp]
    else:
        se_pp = 0.0
        upper_95_1s = mean_pp
        ci95 = [mean_pp, mean_pp]
    return dict(
        n_clusters=g_cnt,
        df=max(g_cnt - 1, 0),
        cluster_diffs_pp=diffs_pp,
        mean_diff_pp=mean_pp,
        se_pp=se_pp,
        ci95_2sided_pp=ci95,
        upper_95_1sided_pp=float(upper_95_1s),
        noninferiority_margin_pp=float(margin_pp),
        passes_noninferiority_1pp=bool(upper_95_1s < margin_pp),
        beats_dense_point_est=bool(mean_pp < 0.0),
    )


def run_dev_benchmark(
    snr_db: float = PRIMARY_SNR_DB,
    n_clusters: int = N_CLUSTERS,
    links_per_cluster: int = LINKS_PER_CLUSTER,
) -> dict:
    """Run full P11 dev validation strictly on SEED_P11_DEV."""
    det_info = enforce_determinism(require_env=True)
    warmup_p11_numba()
    t_start = time.perf_counter()
    n0 = n0_from_snr(snr_db)

    # Warmup caches on one scratch seed from SEED_P11_DEV namespace
    warm_seed = DEV_SEED_BASE + 99_999
    if is_historical_seed(warm_seed) is not None:
        raise AssertionError(f"Seed {warm_seed} overlaps historical ranges")
    obs_w, _ = simulate_link(warm_seed, snr_db=snr_db, regime="full_reset", budget="sparse")
    run_p11_with_trace(obs_w)

    acc = {arm: {reg: _new_accum() for reg in DEV_REGIMES} for arm in ARMS}
    cluster_blers = {arm: {reg: [] for reg in DEV_REGIMES} for arm in ARMS}

    for regime in DEV_REGIMES:
        for cluster in range(n_clusters):
            c_err = {arm: 0 for arm in ARMS}
            c_blk = {arm: 0 for arm in ARMS}
            for idx in range(links_per_cluster):
                seed = DEV_SEED_BASE + 1000 * cluster + idx
                if is_historical_seed(seed) is not None:
                    raise AssertionError(f"Seed {seed} overlaps historical ranges")

                obs_s, truth_s = simulate_link(seed, snr_db=snr_db, regime=regime, budget="sparse")
                t_c0 = time.perf_counter()
                _, _, cand_inh, cand_clr = build_candidates_with_priors(obs_s)
                t_cand_ms = (time.perf_counter() - t_c0) * 1000.0

                # Arm 1: sparse_always_inherit
                t_g0 = time.perf_counter()
                g_inh = gate_always_inherit(cand_inh, cand_clr)
                t_g_ms = (time.perf_counter() - t_g0) * 1000.0
                dec_inh, dec_inh_ms = _decode_baseline_strong_large(
                    obs_s, g_inh.mix_mean_ray, g_inh.mix_var_ray, g_inh.mix_llrs
                )
                e1, b1 = _score_into(
                    acc["sparse_always_inherit"][regime],
                    dec_inh, g_inh.mix_mean_ray, g_inh.mix_probe_pred, g_inh.mix_probe_var,
                    g_inh.history_weight, g_inh.probe_weight,
                    truth_s, obs_s.training_rows, n0, t_cand_ms + t_g_ms + dec_inh_ms,
                )
                c_err["sparse_always_inherit"] += e1
                c_blk["sparse_always_inherit"] += b1

                # Arm 2: sparse_jeffreys_student_t
                t_g0 = time.perf_counter()
                g_jt = gate_jeffreys_student_t(obs_s, cand_inh, cand_clr)
                t_g_ms = (time.perf_counter() - t_g0) * 1000.0
                dec_jt, dec_jt_ms = _decode_baseline_strong_large(
                    obs_s, g_jt.mix_mean_ray, g_jt.mix_var_ray, g_jt.mix_llrs
                )
                e2, b2 = _score_into(
                    acc["sparse_jeffreys_student_t"][regime],
                    dec_jt, g_jt.mix_mean_ray, g_jt.mix_probe_pred, g_jt.mix_probe_var,
                    g_jt.history_weight, g_jt.probe_weight,
                    truth_s, obs_s.training_rows, n0, t_cand_ms + t_g_ms + dec_jt_ms,
                )
                c_err["sparse_jeffreys_student_t"] += e2
                c_blk["sparse_jeffreys_student_t"] += b2

                # Arm 3: sparse_payload_rebuild_pilot
                dec_prp, m_prp, _, pp_prp, pv_prp, ms_prp = _run_standalone_payload_rebuild(obs_s, use_payload=False)
                e3, b3 = _score_into(
                    acc["sparse_payload_rebuild_pilot"][regime],
                    dec_prp, m_prp, pp_prp, pv_prp,
                    np.zeros(T, dtype=np.float64), 0.0,
                    truth_s, obs_s.training_rows, n0, ms_prp,
                )
                c_err["sparse_payload_rebuild_pilot"] += e3
                c_blk["sparse_payload_rebuild_pilot"] += b3

                # Arm 4: sparse_payload_rebuild_standalone
                dec_prs, m_prs, _, pp_prs, pv_prs, ms_prs = _run_standalone_payload_rebuild(obs_s, use_payload=True)
                e4, b4 = _score_into(
                    acc["sparse_payload_rebuild_standalone"][regime],
                    dec_prs, m_prs, pp_prs, pv_prs,
                    np.zeros(T, dtype=np.float64), 0.0,
                    truth_s, obs_s.training_rows, n0, ms_prs,
                )
                c_err["sparse_payload_rebuild_standalone"] += e4
                c_blk["sparse_payload_rebuild_standalone"] += b4

                # Arm 5: p11_primary
                res_p11, tr_p11 = run_p11_with_trace(obs_s)
                e5, b5 = _score_into(
                    acc["p11_primary"][regime],
                    res_p11.decoded_info,
                    res_p11.gate_out.mix_mean_ray,
                    res_p11.gate_out.mix_probe_pred,
                    res_p11.gate_out.mix_probe_var,
                    res_p11.gate_out.history_weight,
                    res_p11.gate_out.probe_weight,
                    truth_s, obs_s.training_rows, n0, res_p11.elapsed_ms,
                    assimilated_rows=res_p11.assimilated_rows,
                    ledger=tr_p11["ledger"],
                )
                c_err["p11_primary"] += e5
                c_blk["p11_primary"] += b5

                # Arm 6: dense_always_clear (paired NI target)
                obs_d, truth_d = simulate_link(seed, snr_db=snr_db, regime=regime, budget="dense")
                t_d0 = time.perf_counter()
                _, _, ci_d, cc_d = build_candidates_with_priors(obs_d)
                g_dclr = gate_always_clear(ci_d, cc_d)
                t_dg_ms = (time.perf_counter() - t_d0) * 1000.0
                dec_dclr, dec_d_ms = _decode_baseline_strong_large(
                    obs_d, g_dclr.mix_mean_ray, g_dclr.mix_var_ray, g_dclr.mix_llrs
                )
                e6, b6 = _score_into(
                    acc["dense_always_clear"][regime],
                    dec_dclr, g_dclr.mix_mean_ray, g_dclr.mix_probe_pred, g_dclr.mix_probe_var,
                    g_dclr.history_weight, g_dclr.probe_weight,
                    truth_d, obs_d.training_rows, n0, t_dg_ms + dec_d_ms,
                )
                c_err["dense_always_clear"] += e6
                c_blk["dense_always_clear"] += b6

            for arm in ARMS:
                cluster_blers[arm][regime].append(c_err[arm] / max(c_blk[arm], 1))

    summary = {arm: {reg: _finalize(acc[arm][reg]) for reg in DEV_REGIMES} for arm in ARMS}
    for arm in ARMS:
        id_blers = [summary[arm][r]["bler"] for r in REGIMES_ID]
        ood_blers = [summary[arm][r]["bler"] for r in REGIMES_OOD]
        summary[arm]["_aggregate"] = {
            "id_equal_weight_bler": float(np.mean(id_blers)),
            "ood_equal_weight_bler": float(np.mean(ood_blers)),
            "all5_equal_weight_bler": float(np.mean(id_blers + ood_blers)),
            "mean_id_ms_per_link": float(np.mean([summary[arm][r]["ms_per_link"] for r in REGIMES_ID])),
        }

    ni_continuous = _paired_noninferiority_stats(
        cluster_blers["p11_primary"]["continuous"],
        cluster_blers["dense_always_clear"]["continuous"],
    )

    p11_dir = Path(__file__).resolve().parent
    checked_files = sorted(
        [
            p11_dir / "__init__.py",
            p11_dir / "contract.py",
            p11_dir / "payload_rebuild.py",
            p11_dir / "wavefront_pipeline.py",
            Path(__file__).resolve(),
            p11_dir / "tests" / "test_p11_contract.py",
            p11_dir / "tests" / "test_p11_primary.py",
        ]
    )
    n0_report = check_no_hardcoded_n0(checked_files, strict=True)
    inf_hits: list[str] = []
    for p in checked_files:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "inference_mode") or (
                isinstance(node, ast.Name) and node.id == "inference_mode"
            ):
                inf_hits.append(f"{p.name}:{node.lineno}")
    if inf_hits:
        raise RuntimeError(f"Forbidden inference_mode references: {inf_hits}")

    out = {
        "benchmark": "p11_dev",
        "seed_namespace": "SEED_P11_DEV",
        "dev_seed_base": int(DEV_SEED_BASE),
        "snr_db": float(snr_db),
        "n0": float(n0),
        "n_clusters": int(n_clusters),
        "links_per_cluster": int(links_per_cluster),
        "frozen_tau_switch": float(FROZEN_TAU_SWITCH),
        "determinism": det_info,
        "static_checks": {
            "no_hardcoded_n0": n0_report,
            "inference_mode_hits": inf_hits,
        },
        "paired_noninferiority_continuous_vs_dense": ni_continuous,
        "summary": summary,
        "total_elapsed_s": float(time.perf_counter() - t_start),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return out


def main() -> None:
    out = run_dev_benchmark()
    print(f"Wrote {RESULTS_PATH} in {out['total_elapsed_s']:.1f}s")
    for arm in ARMS:
        agg = out["summary"][arm]["_aggregate"]
        id_str = ", ".join(f"{r}={100*out['summary'][arm][r]['bler']:.2f}%" for r in REGIMES_ID)
        print(
            f"{arm:34s} | 3-ID={100*agg['id_equal_weight_bler']:5.2f}% | {id_str} | "
            f"OOD={100*agg['ood_equal_weight_bler']:5.2f}% | ms/link={agg['mean_id_ms_per_link']:.1f}"
        )
    ni = out["paired_noninferiority_continuous_vs_dense"]
    print(
        f"Continuous NI vs dense_always_clear: diff={ni['mean_diff_pp']:+.2f} pp, "
        f"upper_95_1s={ni['upper_95_1sided_pp']:+.2f} pp (< +1.00 pp: {ni['passes_noninferiority_1pp']})"
    )


if __name__ == "__main__":
    main()
