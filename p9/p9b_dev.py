"""P9-B Dev Evaluation: Classical History-Trust Gating across ID, OOD, and Sanity Regimes.

Self-contained dev generator & evaluator using ONLY p9/contract.py, p9/gating.py, p8, and p8c.
Evaluates all 5 classical gates at 9 dB on sparse (1 training block) and dense (3 training
blocks reference) budgets across:
  - 3 ID regimes:  continuous, partial_shift (boundary phase-continuous 1-ray shift), full_reset
  - 2 OOD regimes: ood_strong_shift (larger boundary shift), ood_midframe_shift (row-6 shift)
  - 1 Sanity:      donor_mismatch (foreign Frame A grafted onto Frame B)

Using dev seeds SEED_P9_DEV + 200000 (4 clusters x 16 links = 64 links = 1,024 core blocks per
regime). Writes clean summary JSON to p9/results/p9b_dev_gating.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from p8 import coding, link as ofdm
from p8c import link as sc
from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    N,
    OFDM_PILOT_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    SC_LAYOUTS,
    SEED_P9_DEV,
    T,
    HiddenTruth,
    PublicObservation,
    n0_from_snr,
)
from p9.gating import (
    build_candidates_with_priors,
    gate_always_clear,
    gate_always_inherit,
    gate_jeffreys_student_t,
    gate_pre_bayes_factor,
    gate_split_predictive,
    readout_sc,
)


DEV_SEED_BASE = SEED_P9_DEV + 200000
N_CLUSTERS = 4
LINKS_PER_CLUSTER = 16
RESULTS_PATH = Path(__file__).parent / "results" / "p9b_dev_gating.json"

GATES = (
    "always_inherit",
    "always_clear",
    "pre_bayes_factor",
    "split_predictive",
    "jeffreys_student_t",
)


def frame_b_ray_gains(
    channel_a: ofdm.Channel,
    channel_b: ofdm.Channel,
    time_b: np.ndarray,
    changed_ray: int | None = None,
    delta_freq: float = 0.0,
    shift_row: int | None = None,
) -> np.ndarray:
    """Compute exact time-domain ray gains h_b_true [3, T, N] on Frame B.

    For single-ray Doppler shifts (partial_shift, ood_strong_shift, ood_midframe_shift),
    applies the exact phase-continuous shift starting at t_s = (T + shift_row) * (N + CP) / N:

        phi_l(t) = 2*pi * (f_l * t + 0.5 * c_l * t^2 + 1_{l==changed_ray} * delta_freq * max(t - t_s, 0))

    At t = t_s, max(t - t_s, 0) == 0, guaranteeing exact complex phase continuity across the
    A/B boundary (shift_row=0, t_s=FRAME_DURATION=30.0) or mid-frame boundary (shift_row=6, t_s=39.0).
    """
    t_arr = np.asarray(time_b, dtype=np.float64)
    base = channel_b if changed_ray is None else channel_a
    phase = (
        base.frequency[:, None, None] * t_arr[None, :, :]
        + 0.5 * base.chirp[:, None, None] * (t_arr[None, :, :] ** 2)
    )
    if changed_ray is not None and delta_freq != 0.0:
        sr = 0 if shift_row is None else int(shift_row)
        t_shift = float((T + sr) * (N + CP) / N)
        phase[int(changed_ray)] += float(delta_freq) * np.maximum(t_arr - t_shift, 0.0)
    return base.gain[:, None, None] * np.exp(2j * np.pi * phase)


def propagate_sc_from_gains(tx_grid: np.ndarray, h_b_true: np.ndarray) -> np.ndarray:
    """Propagate CP-SC time-domain blocks [T, N] through exact ray gains [3, T, N]."""
    grid = np.asarray(tx_grid, dtype=np.complex128)
    rx = np.zeros(grid.shape, dtype=np.complex128)
    for ray_idx, delay in enumerate(DELAYS):
        rx += h_b_true[ray_idx] * np.roll(grid, int(delay), axis=-1)
    return rx


def simulate_link(
    seed: int,
    regime: str,
    snr_db: float = PRIMARY_SNR_DB,
    budget: str = "sparse",
) -> tuple[PublicObservation, HiddenTruth]:
    """Self-contained explicit-N0 OFDM(A) -> CP-SC(B) link simulator for all P9-B regimes.

    Strictly masks obs.y_b[PROBE_ROW] to NaN+1j*NaN before returning PublicObservation.
    """
    if regime not in ALL_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {ALL_REGIMES}")
    if budget not in SC_LAYOUTS:
        raise ValueError(f"Unknown budget {budget!r}; expected one of {tuple(SC_LAYOUTS)}")

    n0 = n0_from_snr(snr_db)
    training_rows = SC_LAYOUTS[budget]

    # Paired base streams matching p8c.link.simulate stream layout
    channel_a = ofdm.draw_channel(ofdm.stream(seed, 0))
    a_words = ofdm.traffic(ofdm.stream(seed, 1))
    a_noise = np.fft.fft(
        np.sqrt(n0) * ofdm.complex_normal(ofdm.stream(seed, 2), (T, N)),
        norm="ortho",
    )
    y_a_base = ofdm.propagate(ofdm.tx_grid(a_words, OFDM_PILOT_ROWS), channel_a) + a_noise

    words_raw = ofdm.traffic(ofdm.stream(seed, 3))
    b_words = ofdm.Traffic(words_raw.info[:18], words_raw.code[:18], words_raw.symbols[:18])
    b_noise = np.sqrt(n0) * ofdm.complex_normal(ofdm.stream(seed, 4), (T, N))

    donor_channel = ofdm.draw_channel(ofdm.stream(seed, 5))
    donor_words = ofdm.traffic(ofdm.stream(seed, 6))
    donor_noise = np.fft.fft(
        np.sqrt(n0) * ofdm.complex_normal(ofdm.stream(seed, 7), (T, N)),
        norm="ortho",
    )
    y_a_donor = ofdm.propagate(ofdm.tx_grid(donor_words, OFDM_PILOT_ROWS), donor_channel) + donor_noise

    reg_rng = ofdm.stream(seed, 8)
    time_b = ofdm.clock(T, start_row=T)

    y_a_used = y_a_base
    channel_b = channel_a
    changed_ray: int | None = None
    delta_freq = 0.0
    shift_row: int | None = None

    if regime == "continuous":
        pass
    elif regime == "partial_shift":
        changed_ray = int(reg_rng.integers(0, len(DELAYS)))
        sign = float(reg_rng.choice([-1.0, 1.0]))
        delta_freq = float(sign * reg_rng.uniform(0.025, 0.055))
        shift_row = 0
    elif regime == "full_reset":
        channel_b = ofdm.draw_channel(ofdm.stream(seed, 9))
        shift_row = 0
    elif regime == "ood_strong_shift":
        changed_ray = int(reg_rng.integers(0, len(DELAYS)))
        sign = float(reg_rng.choice([-1.0, 1.0]))
        delta_freq = float(sign * reg_rng.uniform(0.080, 0.135))
        shift_row = 0
    elif regime == "ood_midframe_shift":
        changed_ray = int(reg_rng.integers(0, len(DELAYS)))
        sign = float(reg_rng.choice([-1.0, 1.0]))
        delta_freq = float(sign * reg_rng.uniform(0.035, 0.070))
        shift_row = 6
    elif regime == "donor_mismatch":
        y_a_used = y_a_donor
        channel_b = channel_a
        shift_row = 0
    else:
        raise AssertionError(f"Unhandled regime: {regime}")

    h_b_true = frame_b_ray_gains(
        channel_a=channel_a,
        channel_b=channel_b,
        time_b=time_b,
        changed_ray=changed_ray,
        delta_freq=delta_freq,
        shift_row=shift_row,
    )
    tx_b = sc.tx_grid(b_words, training_rows)
    y_b_clean = propagate_sc_from_gains(tx_b, h_b_true)
    y_b_noisy = y_b_clean + b_noise

    # Strict withheld probe masking on PublicObservation
    y_b_public = y_b_noisy.copy()
    y_b_public[PROBE_ROW] = complex(np.nan, np.nan)

    obs = PublicObservation(
        y_a=y_a_used,
        y_b=y_b_public,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        budget=budget,
    )
    truth = HiddenTruth(
        regime=regime,
        channel_a=donor_channel if regime == "donor_mismatch" else channel_a,
        channel_b=channel_b,
        changed_ray=changed_ray,
        delta_freq=delta_freq,
        shift_row=shift_row,
        b_info=b_words.info,
        b_code=b_words.code,
        b_symbols=b_words.symbols,
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_clean[PROBE_ROW].copy(),
        y_b_full_clean=y_b_clean,
        h_b_true=h_b_true,
    )
    return obs, truth


def _new_accum() -> dict:
    return dict(
        links=0,
        blocks=0,
        block_errors=0,
        first8_errors=0,
        last8_errors=0,
        extra_blocks=0,
        extra_errors=0,
        coded_bits=0,
        coded_errors=0,
        info_bits=0,
        info_errors=0,
        probe_sse_noisy=0.0,
        probe_energy_noisy=0.0,
        probe_sse_clean=0.0,
        probe_energy_clean=0.0,
        probe_var_sum=0.0,
        probe_nll_sum=0.0,
        probe_samples=0,
        weight_sum=0.0,
        probe_weight_sum=0.0,
        elapsed_sec=0.0,
    )


def _score_into(
    acc: dict,
    llrs: np.ndarray,
    probe_pred: np.ndarray,
    probe_var: np.ndarray,
    history_weight: np.ndarray,
    probe_weight: float,
    truth: HiddenTruth,
    training_rows: tuple[int, ...],
    n0: float,
    elapsed_sec: float,
) -> None:
    if not (np.isfinite(llrs).all() and np.isfinite(probe_pred).all() and np.isfinite(probe_var).all()):
        raise ValueError("Non-finite receiver output")
    if np.any(probe_var <= 0.0):
        raise ValueError("Non-positive probe predictive variance")

    core = llrs[list(CORE_ROWS)].reshape(len(CORE_ROWS), 128)
    dec_core = coding.decode(core)
    wrong_core = dec_core != truth.b_info[:16]
    err_core = wrong_core.any(axis=1)

    extra = list(sc.extra_rows(training_rows))
    extra_err = 0
    if extra:
        dec_extra = coding.decode(llrs[extra].reshape(len(extra), 128))
        extra_err = int((dec_extra != truth.b_info[16 : 16 + len(extra)]).any(axis=1).sum())

    res_noisy = np.abs(truth.y_b_probe_noisy - probe_pred) ** 2
    res_clean = np.abs(truth.y_b_probe_clean - probe_pred) ** 2

    acc["links"] += 1
    acc["blocks"] += len(CORE_ROWS)
    acc["block_errors"] += int(err_core.sum())
    acc["first8_errors"] += int(err_core[:8].sum())
    acc["last8_errors"] += int(err_core[8:].sum())
    acc["extra_blocks"] += len(extra)
    acc["extra_errors"] += extra_err
    acc["coded_bits"] += int(core.size)
    acc["coded_errors"] += int(((core > 0.0) != truth.b_code[:16]).sum())
    acc["info_bits"] += int(wrong_core.size)
    acc["info_errors"] += int(wrong_core.sum())
    acc["probe_sse_noisy"] += float(res_noisy.sum())
    acc["probe_energy_noisy"] += float((np.abs(truth.y_b_probe_noisy) ** 2).sum())
    acc["probe_sse_clean"] += float(res_clean.sum())
    acc["probe_energy_clean"] += float((np.abs(truth.y_b_probe_clean) ** 2).sum())
    acc["probe_var_sum"] += float(probe_var.sum())
    acc["probe_nll_sum"] += float((np.log(probe_var / n0) + res_noisy / probe_var).sum())
    acc["probe_samples"] += int(len(probe_pred))
    acc["weight_sum"] += float(np.mean(history_weight))
    acc["probe_weight_sum"] += float(probe_weight)
    acc["elapsed_sec"] += float(elapsed_sec)


def _finalize(acc: dict) -> dict:
    blocks = max(acc["blocks"], 1)
    links = max(acc["links"], 1)
    extra_blocks = acc["extra_blocks"]
    core_ok = acc["blocks"] - acc["block_errors"]
    extra_ok = extra_blocks - acc["extra_errors"]
    total_ok = core_ok + extra_ok
    return dict(
        links=acc["links"],
        blocks=acc["blocks"],
        block_errors=acc["block_errors"],
        bler=acc["block_errors"] / blocks,
        first8_bler=acc["first8_errors"] / max(acc["links"] * 8, 1),
        last8_bler=acc["last8_errors"] / max(acc["links"] * 8, 1),
        extra_blocks=extra_blocks,
        extra_errors=acc["extra_errors"],
        net_delivered_blocks=total_ok,
        net_delivered_blocks_per_link=total_ok / links,
        net_delivered_info_bits=total_ok * 64,
        coded_ber=acc["coded_errors"] / max(acc["coded_bits"], 1),
        info_ber=acc["info_errors"] / max(acc["info_bits"], 1),
        probe_nmse_noisy=acc["probe_sse_noisy"] / max(acc["probe_energy_noisy"], 1e-300),
        probe_nmse_clean=acc["probe_sse_clean"] / max(acc["probe_energy_clean"], 1e-300),
        probe_cal_ratio=acc["probe_sse_noisy"] / max(acc["probe_var_sum"], 1e-300),
        probe_nll_per_sample=acc["probe_nll_sum"] / max(acc["probe_samples"], 1),
        mean_weight=acc["weight_sum"] / links,
        mean_probe_weight=acc["probe_weight_sum"] / links,
        ms_per_link=1e3 * acc["elapsed_sec"] / links,
    )


def run_dev_eval(
    snr_db: float = PRIMARY_SNR_DB,
    n_clusters: int = N_CLUSTERS,
    links_per_cluster: int = LINKS_PER_CLUSTER,
    budgets: tuple[str, ...] = ("sparse", "dense"),
) -> dict:
    t_start = time.perf_counter()
    n0 = n0_from_snr(snr_db)
    seeds = [
        DEV_SEED_BASE + 1000 * cluster + idx
        for cluster in range(n_clusters)
        for idx in range(links_per_cluster)
    ]

    report: dict = dict(
        meta=dict(
            snr_db=float(snr_db),
            n0=float(n0),
            dev_seed_base=int(DEV_SEED_BASE),
            n_clusters=int(n_clusters),
            links_per_cluster=int(links_per_cluster),
            links_per_regime=len(seeds),
            core_blocks_per_regime=len(seeds) * len(CORE_ROWS),
            regimes_id=list(REGIMES_ID),
            regimes_ood=list(REGIMES_OOD),
            regimes_all=list(ALL_REGIMES),
            gates=list(GATES),
        ),
        budgets={},
    )

    for budget in budgets:
        training_rows = SC_LAYOUTS[budget]
        acc_by_gate_regime = {
            g: {reg: _new_accum() for reg in ALL_REGIMES}
            for g in (*GATES, "oracle_ref")
        }
        # Per-changed-ray breakdown for single-ray shift regimes
        shift_regimes = ("partial_shift", "ood_strong_shift", "ood_midframe_shift")
        acc_by_ray = {
            g: {reg: {f"ray_{r}": _new_accum() for r in range(len(DELAYS))} for reg in shift_regimes}
            for g in (*GATES, "oracle_ref")
        }

        for regime in ALL_REGIMES:
            for seed in seeds:
                obs, truth = simulate_link(seed, regime, snr_db=snr_db, budget=budget)

                t0 = time.perf_counter()
                st_inh_prior, st_clr_prior, cand_inh, cand_clr = build_candidates_with_priors(obs)
                t_cand = time.perf_counter() - t0

                # Gate 1: always_inherit
                tg = time.perf_counter()
                out_inh = gate_always_inherit(cand_inh, cand_clr)
                dt_inh = t_cand + (time.perf_counter() - tg)

                # Gate 2: always_clear
                tg = time.perf_counter()
                out_clr = gate_always_clear(cand_inh, cand_clr)
                dt_clr = t_cand + (time.perf_counter() - tg)

                # Gate 3: pre_bayes_factor
                tg = time.perf_counter()
                out_bf = gate_pre_bayes_factor(cand_inh, cand_clr, prior_odds=1.0, temp=1.0)
                dt_bf = t_cand + (time.perf_counter() - tg)

                # Gate 4: split_predictive
                tg = time.perf_counter()
                out_sp = gate_split_predictive(obs, st_inh_prior, st_clr_prior, cand_inh, cand_clr, temp=1.0)
                dt_sp = t_cand + (time.perf_counter() - tg)

                # Gate 5: jeffreys_student_t
                tg = time.perf_counter()
                out_jt = gate_jeffreys_student_t(obs, cand_inh, cand_clr, nu=4.0)
                dt_jt = t_cand + (time.perf_counter() - tg)

                gate_outputs = (
                    ("always_inherit", out_inh, dt_inh),
                    ("always_clear", out_clr, dt_clr),
                    ("pre_bayes_factor", out_bf, dt_bf),
                    ("split_predictive", out_sp, dt_sp),
                    ("jeffreys_student_t", out_jt, dt_jt),
                )
                for g_name, g_out, dt in gate_outputs:
                    _score_into(
                        acc_by_gate_regime[g_name][regime],
                        g_out.mix_llrs,
                        g_out.mix_probe_pred,
                        g_out.mix_probe_var,
                        g_out.history_weight,
                        g_out.probe_weight,
                        truth,
                        training_rows,
                        n0,
                        dt,
                    )
                    if regime in shift_regimes and truth.changed_ray is not None:
                        _score_into(
                            acc_by_ray[g_name][regime][f"ray_{truth.changed_ray}"],
                            g_out.mix_llrs,
                            g_out.mix_probe_pred,
                            g_out.mix_probe_var,
                            g_out.history_weight,
                            g_out.probe_weight,
                            truth,
                            training_rows,
                            n0,
                            dt,
                        )

                # True-channel oracle diagnostic reference (evaluator-only)
                tg = time.perf_counter()
                o_llrs, o_pred, o_var = readout_sc(
                    truth.h_b_true,
                    np.zeros_like(truth.h_b_true.real, dtype=np.float64),
                    obs.y_b,
                    training_rows,
                    n0=n0,
                    probe_row=PROBE_ROW,
                )
                dt_o = time.perf_counter() - tg
                _score_into(
                    acc_by_gate_regime["oracle_ref"][regime],
                    o_llrs,
                    o_pred,
                    o_var,
                    np.ones(T, dtype=np.float64),
                    1.0,
                    truth,
                    training_rows,
                    n0,
                    dt_o,
                )
                if regime in shift_regimes and truth.changed_ray is not None:
                    _score_into(
                        acc_by_ray["oracle_ref"][regime][f"ray_{truth.changed_ray}"],
                        o_llrs,
                        o_pred,
                        o_var,
                        np.ones(T, dtype=np.float64),
                        1.0,
                        truth,
                        training_rows,
                        n0,
                        dt_o,
                    )

        # Finalize budget tables
        final_per_gate = {}
        for g_name in (*GATES, "oracle_ref"):
            per_reg = {reg: _finalize(acc_by_gate_regime[g_name][reg]) for reg in ALL_REGIMES}
            id_avg_bler = float(np.mean([per_reg[r]["bler"] for r in REGIMES_ID]))
            id_avg_nmse_noisy = float(np.mean([per_reg[r]["probe_nmse_noisy"] for r in REGIMES_ID]))
            id_avg_nmse_clean = float(np.mean([per_reg[r]["probe_nmse_clean"] for r in REGIMES_ID]))
            id_avg_cal = float(np.mean([per_reg[r]["probe_cal_ratio"] for r in REGIMES_ID]))
            ood_avg_bler = float(np.mean([per_reg[r]["bler"] for r in REGIMES_OOD]))
            final_per_gate[g_name] = dict(
                id_3regime_avg_bler=id_avg_bler,
                id_3regime_avg_probe_nmse_noisy=id_avg_nmse_noisy,
                id_3regime_avg_probe_nmse_clean=id_avg_nmse_clean,
                id_3regime_avg_probe_cal_ratio=id_avg_cal,
                ood_2regime_avg_bler=ood_avg_bler,
                per_regime=per_reg,
                per_changed_ray={
                    reg: {rk: _finalize(acc_by_ray[g_name][reg][rk]) for rk in acc_by_ray[g_name][reg]}
                    for reg in shift_regimes
                },
            )

        # Cross-gate comparative deltas (protection cost, recovery, gain vs best fixed)
        inh_cont_bler = final_per_gate["always_inherit"]["per_regime"]["continuous"]["bler"]
        inh_rst_bler = final_per_gate["always_inherit"]["per_regime"]["full_reset"]["bler"]
        inh_mis_bler = final_per_gate["always_inherit"]["per_regime"]["donor_mismatch"]["bler"]
        best_fixed_id_bler = min(
            final_per_gate["always_inherit"]["id_3regime_avg_bler"],
            final_per_gate["always_clear"]["id_3regime_avg_bler"],
        )
        best_fixed_name = (
            "always_inherit"
            if final_per_gate["always_inherit"]["id_3regime_avg_bler"]
            <= final_per_gate["always_clear"]["id_3regime_avg_bler"]
            else "always_clear"
        )
        for g_name in (*GATES, "oracle_ref"):
            g_dict = final_per_gate[g_name]
            g_dict["continuous_protection_cost"] = (
                g_dict["per_regime"]["continuous"]["bler"] - inh_cont_bler
            )
            g_dict["full_reset_recovery"] = (
                inh_rst_bler - g_dict["per_regime"]["full_reset"]["bler"]
            )
            g_dict["donor_mismatch_recovery"] = (
                inh_mis_bler - g_dict["per_regime"]["donor_mismatch"]["bler"]
            )
            g_dict["delta_vs_best_fixed_id"] = g_dict["id_3regime_avg_bler"] - best_fixed_id_bler

        report["budgets"][budget] = dict(
            training_rows=list(training_rows),
            best_fixed_id_policy=best_fixed_name,
            best_fixed_id_bler=best_fixed_id_bler,
            gates=final_per_gate,
        )

    report["meta"]["total_elapsed_sec"] = float(time.perf_counter() - t_start)
    return report


def print_summary(report: dict) -> None:
    meta = report["meta"]
    print(
        f"=== P9-B Dev Gating Summary ({meta['snr_db']:.1f} dB, "
        f"{meta['links_per_regime']} links = {meta['core_blocks_per_regime']} core blks/regime, "
        f"elapsed={meta['total_elapsed_sec']:.1f}s) ==="
    )
    for budget, bdata in report["budgets"].items():
        print(f"\n--- Budget: {budget} (training_rows={bdata['training_rows']}, best_fixed={bdata['best_fixed_id_policy']}) ---")
        header = (
            f"{'Gate':20s} | {'3-ID Avg':>8s} | {'Cont':>7s} | {'PartSh':>7s} | {'FullRst':>7s} | "
            f"{'OODStr':>7s} | {'OODMid':>7s} | {'DonorM':>7s} | {'ProtCost':>8s} | {'RstRecov':>8s}"
        )
        print(header)
        print("-" * len(header))
        for g_name in (*GATES, "oracle_ref"):
            gd = bdata["gates"][g_name]
            pr = gd["per_regime"]
            print(
                f"{g_name:20s} | {100*gd['id_3regime_avg_bler']:7.2f}% | "
                f"{100*pr['continuous']['bler']:6.2f}% | {100*pr['partial_shift']['bler']:6.2f}% | "
                f"{100*pr['full_reset']['bler']:6.2f}% | {100*pr['ood_strong_shift']['bler']:6.2f}% | "
                f"{100*pr['ood_midframe_shift']['bler']:6.2f}% | {100*pr['donor_mismatch']['bler']:6.2f}% | "
                f"{100*gd['continuous_protection_cost']:+7.2f}p | {100*gd['full_reset_recovery']:+7.2f}p"
            )
        print("\nMean History Weight E[w] & Probe Calibration (row 19):")
        header2 = (
            f"{'Gate':20s} | {'E[w]Cnt':>7s} {'E[w]Prt':>7s} {'E[w]Rst':>7s} {'E[w]Str':>7s} {'E[w]Mid':>7s} {'E[w]Dnr':>7s} | "
            f"{'ID NMSEc':>8s} {'ID Cal':>7s}"
        )
        print(header2)
        print("-" * len(header2))
        for g_name in GATES:
            gd = bdata["gates"][g_name]
            pr = gd["per_regime"]
            print(
                f"{g_name:20s} | "
                f"{pr['continuous']['mean_weight']:7.3f} {pr['partial_shift']['mean_weight']:7.3f} "
                f"{pr['full_reset']['mean_weight']:7.3f} {pr['ood_strong_shift']['mean_weight']:7.3f} "
                f"{pr['ood_midframe_shift']['mean_weight']:7.3f} {pr['donor_mismatch']['mean_weight']:7.3f} | "
                f"{gd['id_3regime_avg_probe_nmse_clean']:8.4f} {gd['id_3regime_avg_probe_cal_ratio']:7.3f}"
            )
        print("\nPartial-Shift Per-Changed-Ray Breakdown (BLER / E[w]):")
        for g_name in (*GATES, "oracle_ref"):
            pcr = bdata["gates"][g_name]["per_changed_ray"]["partial_shift"]
            parts = " | ".join(
                f"{rk}: {100*pcr[rk]['bler']:5.1f}% (w={pcr[rk]['mean_weight']:.2f}, n={pcr[rk]['links']})"
                for rk in ("ray_0", "ray_1", "ray_2")
            )
            print(f"  {g_name:20s}: {parts}")


def main() -> None:
    report = run_dev_eval()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_summary(report)
    print(f"\nSaved dev gating results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
