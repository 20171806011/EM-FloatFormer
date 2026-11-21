"""Phase 12 Production Primary-Axis Receiver (`K=2` & `K=3` Symmetric Bilateral Wavefront) & Out-of-Support OOD Simulator.

Hardened for formal Stage 3 G=16 closed-book evaluation:
  - Strict validation of fold partitions (equal lengths, unique ownership covering all 16 CORE_ROWS,
    strict target/AUX/anchor/probe disjointness, zero silent zip truncation).
  - Strict held-out readout for reclaimed extra rows `(0, 9)`: each extra row is read out strictly
    from a fold that NEVER observed it (neither as AUX nor as a helper!) so reclaimed-row goodput
    carries zero post-injection leakage.
  - True Out-of-Support (OOD) primary simulator supporting:
      * `ood_unseen_onset`: integer shift onset strictly outside P11's `{0, 6}` bank (`onset in {3, 14}`)
      * `ood_two_ray_shift`: two distinct rays simultaneously shifting Doppler at Frame-B boundary
      * All standard P9/P11 regimes delegated identically to `p9.channel.simulate_link`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from p8.link import (
    Channel,
    Traffic,
    clock,
    complex_normal,
    draw_channel,
    propagate as propagate_ofdm,
    stream,
    traffic,
    tx_grid as ofdm_tx_grid,
)
from p8c.link import PILOTS as SC_PILOTS, extra_rows, tx_grid as sc_tx_grid
from p9.channel import (
    _ofdm_noise,
    evaluate_ray_gains,
    propagate_sc_from_gains,
    row_start_time_b,
    simulate_link,
)
from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    DELAYS,
    N,
    OFDM_PILOT_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    SC_LAYOUTS,
    T,
    GateOutput,
    HiddenTruth,
    PublicObservation,
    n0_from_snr,
)
from p11.contract import ANCHOR, COMPLEX_DTYPE, REAL_DTYPE, AssimilationResult
from p11.payload_rebuild import (
    DEFAULT_SCALE_QUAD_ORDER,
    _strict_row_int,
    _validate_positive_n0,
    build_frame_a_prior_clean,
    fit_anchor_only,
)
from p11.wavefront_pipeline import (
    DEFAULT_AUX_TEMP,
    FINE_SCHEDULE,
    FROZEN_TAU_SWITCH,
    _readout_rows_llrs,
    _sanitize_y_b,
    run_p11_primary,
    run_p11_with_trace,
)
from p12.contract import (
    P12_FOLD2_AUX,
    P12_FOLD2_TARGETS,
    P12_FOLD3_AUX,
    P12_FOLD3_TARGETS,
)
from p12.primary_folds import (
    CORE_ROW_TO_INFO_IDX,
    SPARSE_EXTRA_ROWS,
    SPARSE_ROW_TO_INFO_IDX,
    _run_generic_fold,
    decode_rows_strong_l,
)

# Out-of-support primary OOD regimes strictly outside P11's registered shift bank ({0, 6} single-ray)
PRIMARY_OOD_UNSEEN_ONSET = "ood_unseen_onset"
PRIMARY_OOD_TWO_RAY_SHIFT = "ood_two_ray_shift"
PRIMARY_UNSEEN_ONSET_ROWS: tuple[int, ...] = (3, 14)
PRIMARY_OOD_EXTRA_REGIMES: tuple[str, ...] = (
    PRIMARY_OOD_UNSEEN_ONSET,
    PRIMARY_OOD_TWO_RAY_SHIFT,
)
PRIMARY_ALL_P12_OOD_REGIMES: tuple[str, ...] = (
    "ood_strong_shift",
    "ood_midframe_shift",
    "donor_mismatch",
    PRIMARY_OOD_UNSEEN_ONSET,
    PRIMARY_OOD_TWO_RAY_SHIFT,
)


# ---------------------------------------------------------------------------
# 1. Exact Multi-Ray / Unseen-Onset Phase-Continuous Primary Simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiRayShiftChannel:
    """Phase-continuous multi-ray shift channel for true out-of-support OOD evaluation."""
    base: Channel
    changed_rays: tuple[int, ...]
    delta_freqs: tuple[float, ...]
    shift_row: int
    shift_time: float


def _eval_multiray_gains(ch: MultiRayShiftChannel, times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=REAL_DTYPE)
    base = ch.base
    gain = np.asarray(base.gain, dtype=COMPLEX_DTYPE)
    freq = np.asarray(base.frequency, dtype=REAL_DTYPE)
    chirp = np.asarray(base.chirp, dtype=REAL_DTYPE)
    expand = (slice(None),) + (None,) * t.ndim
    phase = freq[expand] * t[None, ...] + 0.5 * chirp[expand] * (t[None, ...] ** 2)
    mask = t >= float(ch.shift_time)
    for ray, df in zip(ch.changed_rays, ch.delta_freqs):
        phase[int(ray)] = np.where(mask, phase[int(ray)] + float(df) * (t - float(ch.shift_time)), phase[int(ray)])
    return gain[expand] * np.exp(2j * np.pi * phase)


def simulate_p12_primary_link(
    seed: int,
    snr_db: float = PRIMARY_SNR_DB,
    regime: str = "continuous",
    budget: str = "sparse",
) -> tuple[PublicObservation, HiddenTruth]:
    """Generate one P12 primary link (standard P9/P11 regimes + true out-of-support OOD regimes)."""
    if regime in ALL_REGIMES:
        return simulate_link(seed, snr_db=snr_db, regime=regime, budget=budget)
    if regime not in PRIMARY_OOD_EXTRA_REGIMES:
        raise ValueError(f"Unknown primary regime {regime!r}")
    if budget not in SC_LAYOUTS:
        raise ValueError(f"Unknown budget {budget!r}")

    n0 = float(n0_from_snr(snr_db))
    training_rows = tuple(SC_LAYOUTS[budget])

    channel_0 = draw_channel(stream(seed, 0))
    a_words = traffic(stream(seed, 1))
    unit_noise_a = complex_normal(stream(seed, 2), (T, N))
    words_raw = traffic(stream(seed, 3))
    b_words = Traffic(words_raw.info[:18], words_raw.code[:18], words_raw.symbols[:18])
    unit_noise_b = complex_normal(stream(seed, 4), (T, N))

    rng_s = stream(seed, 9)
    if regime == PRIMARY_OOD_UNSEEN_ONSET:
        changed_ray = int(rng_s.integers(0, len(DELAYS)))
        sign = int(rng_s.choice([-1, 1]))
        df = float(sign * rng_s.uniform(0.035, 0.065))
        s_row = int(rng_s.choice(PRIMARY_UNSEEN_ONSET_ROWS))
        t_shift = row_start_time_b(s_row)
        ch_b: Any = MultiRayShiftChannel(channel_0, (changed_ray,), (df,), s_row, t_shift)
        ch_ray_meta: Optional[int] = changed_ray
        df_meta = df
    else:
        # ood_two_ray_shift: 2 distinct rays simultaneously shift Doppler at Frame-B boundary (row 0)
        pair = np.sort(rng_s.choice(len(DELAYS), size=2, replace=False))
        signs = rng_s.choice([-1, 1], size=2)
        mags = rng_s.uniform(0.035, 0.065, size=2)
        dfs = tuple(float(signs[i] * mags[i]) for i in range(2))
        s_row = 0
        t_shift = row_start_time_b(s_row)
        ch_b = MultiRayShiftChannel(channel_0, (int(pair[0]), int(pair[1])), dfs, s_row, t_shift)
        ch_ray_meta = int(pair[0])
        df_meta = float(dfs[0])

    y_a = propagate_ofdm(ofdm_tx_grid(a_words, OFDM_PILOT_ROWS), channel_0) + _ofdm_noise(unit_noise_a, n0)
    grid_b = sc_tx_grid(b_words, training_rows)
    times_b = clock(T, start_row=T)
    h_b_true = _eval_multiray_gains(ch_b, times_b)
    y_b_full_clean = propagate_sc_from_gains(grid_b, h_b_true)
    y_b_noisy = y_b_full_clean + np.sqrt(n0) * unit_noise_b
    y_b_pub = y_b_noisy.copy()
    y_b_pub[PROBE_ROW] = complex(np.nan, np.nan)

    obs = PublicObservation(
        y_a=np.asarray(y_a, dtype=COMPLEX_DTYPE),
        y_b=y_b_pub,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        budget=budget,
    )
    truth = HiddenTruth(
        regime=regime,
        channel_a=channel_0,
        channel_b=ch_b,
        changed_ray=ch_ray_meta,
        delta_freq=df_meta,
        shift_row=s_row,
        b_info=np.asarray(b_words.info, dtype=np.uint8),
        b_code=np.asarray(b_words.code, dtype=np.uint8),
        b_symbols=np.asarray(b_words.symbols, dtype=COMPLEX_DTYPE),
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_full_clean[PROBE_ROW].copy(),
        y_b_full_clean=y_b_full_clean,
        h_b_true=h_b_true,
    )
    return obs, truth


# ---------------------------------------------------------------------------
# 2. Strict Partition Validation & Held-Out Extra-Row Readout Assignment
# ---------------------------------------------------------------------------


def validate_fold_partition(
    target_folds: Sequence[Sequence[int]],
    aux_folds: Sequence[Sequence[int]],
    *,
    require_global_aux_disjoint: bool = True,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Strictly validate K-fold target and AUX partitions (prevents silent zip truncation or ownership collisions)."""
    if len(target_folds) == 0 or len(target_folds) != len(aux_folds):
        raise ValueError(
            f"target_folds ({len(target_folds)}) and aux_folds ({len(aux_folds)}) must be non-empty and equal length"
        )
    t_norm = tuple(tuple(_strict_row_int(r, "target_row") for r in tf) for tf in target_folds)
    a_norm = tuple(tuple(_strict_row_int(r, "aux_row") for r in af) for af in aux_folds)

    seen_t: set[int] = set()
    seen_a: set[int] = set()
    for f_idx, (tf, af) in enumerate(zip(t_norm, a_norm)):
        if len(tf) == 0:
            raise ValueError(f"Fold {f_idx} has empty target_rows")
        if len(tf) != len(set(tf)):
            raise ValueError(f"Fold {f_idx} has duplicate target_rows: {tf}")
        if len(af) != 5 or len(set(af)) != 5:
            raise ValueError(f"Fold {f_idx} must have 5 distinct aux_rows, got {af}")
        if not set(tf).isdisjoint(af):
            raise ValueError(f"Fold {f_idx} target_rows and aux_rows overlap: {set(tf) & set(af)}")
        if not set(tf).isdisjoint({ANCHOR, PROBE_ROW}) or not set(af).isdisjoint({ANCHOR, PROBE_ROW}):
            raise ValueError(f"Fold {f_idx} touches ANCHOR={ANCHOR} or PROBE_ROW={PROBE_ROW}")
        if not seen_t.isdisjoint(tf):
            raise ValueError(f"Duplicate target ownership across folds: {seen_t & set(tf)}")
        seen_t.update(tf)
        if require_global_aux_disjoint and not seen_a.isdisjoint(af):
            raise ValueError(f"Global AUX overlap in fold {f_idx}: {seen_a & set(af)}")
        seen_a.update(af)

    if seen_t != set(CORE_ROWS):
        raise ValueError(f"Target folds must exactly cover CORE_ROWS={CORE_ROWS}, got {sorted(seen_t)}")
    return t_norm, a_norm


def run_p12_primary_kfold(
    obs: PublicObservation,
    *,
    k_folds: int = 3,
    tau_switch: float = FROZEN_TAU_SWITCH,
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> tuple[AssimilationResult, dict[str, Any]]:
    """Run hardened P12 symmetric bilateral K-fold primary receiver (`k_folds in {2, 3}`).

    Guarantees:
      - Zero target leakage (every core target is masked to NaN during its owning fold's inference & waves).
      - Strict held-out readout for reclaimed extra rows `(0, 9)`: decoded strictly from an isolated
        helper-and-AUX-free fold state (pre-wave `state0` if a row is a helper in all folds, or an
        owning fold where that row was strictly excluded from both AUX and helpers!).
    """
    t0 = time.perf_counter()
    if k_folds == 2:
        t_folds, a_folds = validate_fold_partition(P12_FOLD2_TARGETS, P12_FOLD2_AUX)
    elif k_folds == 3:
        t_folds, a_folds = validate_fold_partition(P12_FOLD3_TARGETS, P12_FOLD3_AUX)
    else:
        raise ValueError(f"Unsupported k_folds={k_folds}; expected 2 or 3")

    n0 = _validate_positive_n0(obs.n0, name="obs.n0")
    probe_row = _strict_row_int(obs.probe_row, "obs.probe_row")
    training_rows = tuple(_strict_row_int(r, "obs.training_rows") for r in obs.training_rows)
    if probe_row != PROBE_ROW or training_rows != (ANCHOR,):
        raise ValueError(f"P12 primary requires sparse training_rows={(ANCHOR,)} and probe_row={PROBE_ROW}")

    y_b_safe = _sanitize_y_b(obs.y_b, probe_row)
    y10_phys = y_b_safe[ANCHOR].copy()
    a_prior = build_frame_a_prior_clean(obs.y_a, n0)
    precomputed_anchor = fit_anchor_only(y10_phys, n0, scale_quad_order=scale_quad_order)
    all_data_rows = tuple(r for r in range(T) if r not in (ANCHOR, PROBE_ROW))

    # For strict zero-leakage extra-row readout of (0, 9):
    # We read out each extra row r_ex ONLY from a fold that excludes r_ex from BOTH its AUX and helpers!
    # Check if we can assign an extra row directly into a fold's target_rows (excluding it from helpers)?
    # Wait: in P12 K=2, row 0 is helper in both folds, row 9 is AUX in Fold 1 & helper in Fold 0;
    # in P12 K=3, row 0 is AUX in Fold 1 & helper in Folds 0,2; row 9 is AUX in Fold 2 & helper in Folds 0,1.
    # Therefore, to guarantee ZERO post-injection leakage for (0, 9) without losing any helper on CORE_ROWS,
    # we decode (0, 9) using the pre-wave (strictly before observing r_ex!) state of the fold where r_ex is NOT in AUX:
    #   - row 9: Fold 0's pre-wave state0 (AUX_0 excludes row 9, Wave-1 hasn't observed row 9 yet!)
    #   - row 0: Fold 0's pre-row-0 wave snapshot (after Wave 1/2/3 on interior rows, strictly BEFORE Wave 4 touches row 0!)
    # Wait — even cleaner: in `_run_generic_fold`, Wave 0 is (9,11), Wave 1 is (8,12), Wave 2 is (4..7,13..16), Wave 3 is (0..3,17,18).
    # So right BEFORE Wave 0 evaluates row 9, Fold 0 has NEVER read row 9! And right BEFORE Wave 3 evaluates row 0,
    # Fold 0 has NEVER read row 0! Reading out row 9 from pre-Wave-0 snapshot and row 0 from pre-Wave-3 snapshot
    # guarantees 100% mathematical zero-leakage (tampering y_b[0] or y_b[9] produces 0.0 diff in its channel prediction!)
    # with ZERO extra compute!
    folds_out: list[dict[str, Any]] = []
    for f_idx, (t_rows, a_rows) in enumerate(zip(t_folds, a_folds)):
        f_res = _run_generic_fold(
            fold_idx=f_idx,
            a_prior=a_prior,
            y10_phys=y10_phys,
            y_b_safe=y_b_safe,
            aux_rows=a_rows,
            target_rows=t_rows,
            all_data_rows=all_data_rows,
            n0=n0,
            precomputed_anchor=precomputed_anchor,
            wave_schedule=FINE_SCHEDULE,
            tau_switch=tau_switch,
            aux_temp=aux_temp,
            scale_quad_order=scale_quad_order,
            bp_iters=bp_iters,
            pic_scale=pic_scale,
        )
        folds_out.append(f_res)

    # Strict held-out readout for extra rows (0, 9):
    # Re-evaluate ONLY the pre-observation fold prediction for row 0 and row 9 cleanly from Fold 0's branch & ledger
    # excluding the queried extra row (and any wave >= its wave) so tampering y_b[r_ex] has 0.0 effect on its prior channel!
    extra_decoded: dict[int, np.ndarray] = {}
    extra_mean: dict[int, np.ndarray] = {}
    extra_var: dict[int, np.ndarray] = {}
    extra_llrs: dict[int, np.ndarray] = {}

    # Fold 0 never has 0 or 9 in AUX_0 for BOTH K=2 ((2,6,7,14,15)) and K=3 ((2,5,8,13,16))!
    assert 0 not in a_folds[0] and 9 not in a_folds[0]
    from p11.payload_rebuild import estimate_payload_posterior_raw, scale_tracker_state, score_branches_common_units, ShiftedBranchDynamics
    from p11.wavefront_pipeline import InheritedOrShiftedBranch, ResetBranch

    f0_aux = a_folds[0]
    y_aux0 = y_b_safe[list(f0_aux)].copy()
    rb0 = estimate_payload_posterior_raw(
        y10=y10_phys, aux_rows=f0_aux, y_aux=y_aux0, n0=n0,
        use_payload=True, aux_temp=aux_temp, scale_quad_order=scale_quad_order,
        precomputed_anchor=precomputed_anchor,
    )
    c_b0 = complex(rb0.gauge)
    ev0 = score_branches_common_units(
        a_prior=a_prior, reset_base=rb0, y10=y10_phys, aux_rows=f0_aux, y_aux=y_aux0, n0=n0, aux_temp=aux_temp, use_parabolic=True
    )
    b0_kind = folds_out[0]["branch"]
    if b0_kind == "reset":
        br0: Any = ResetBranch(rb0, y10_phys)
    elif b0_kind == "shift":
        br0 = InheritedOrShiftedBranch(
            kind="shift",
            dynamics_c=ShiftedBranchDynamics(
                scale_tracker_state(a_prior, 1.0 / c_b0),
                changed_ray=ev0.best_shift.ray, delta_freq=ev0.refined_delta, onset_row=ev0.best_shift.onset_row,
            ),
            gauge=c_b0, n0=n0, y10_c=y10_phys / c_b0,
        )
    else:
        br0 = InheritedOrShiftedBranch(
            kind="continuous",
            dynamics_c=ShiftedBranchDynamics(scale_tracker_state(a_prior, 1.0 / c_b0), changed_ray=None, delta_freq=0.0, onset_row=0),
            gauge=c_b0, n0=n0, y10_c=y10_phys / c_b0,
        )

    from p8.coding import generator, qam
    _g_mat = generator()

    for r_ex in SPARSE_EXTRA_ROWS:
        # Strictly use ONLY Fold 0 ledger entries from waves strictly BEFORE r_ex's wave (never includes r_ex!)
        r_ex_wave = next(wi for wi, w in enumerate(FINE_SCHEDULE) if r_ex in w)
        pre_ledger = {}
        for rec in folds_out[0]["ledger"]:
            if int(rec["wave"]) < r_ex_wave and int(rec["row"]) != r_ex:
                cw = (np.asarray(rec["injected_info"], dtype=np.uint8) @ _g_mat) % 2
                alpha_e = float(rec["alpha"])
                n_eff = n0 * (1.0 / alpha_e + 0.15)
                pre_ledger[int(rec["row"])] = (qam(cw).astype(COMPLEX_DTYPE), y_b_safe[int(rec["row"])].copy(), n_eff)
        st_pre = br0.refit(pre_ledger)
        m_ex, v_ex = br0.predict(st_pre, rows=(r_ex,))
        dec_ex, llr_ex = decode_rows_strong_l(m_ex, v_ex, y_b_safe[r_ex : r_ex + 1], (r_ex,), n0, bp_iters=bp_iters, pic_scale=pic_scale)
        extra_decoded[r_ex] = dec_ex[r_ex]
        extra_mean[r_ex] = m_ex[:, 0, :]
        extra_var[r_ex] = v_ex[:, 0, :]
        extra_llrs[r_ex] = llr_ex[0]

    # Assemble unified AssimilationResult
    mix_mean_ray = np.zeros((len(DELAYS), T, N), dtype=COMPLEX_DTYPE)
    mix_var_ray = np.zeros((len(DELAYS), T, N), dtype=REAL_DTYPE)
    mix_llrs = np.zeros((T, N, 4), dtype=REAL_DTYPE)
    history_weight = np.zeros(T, dtype=REAL_DTYPE)
    decoded_info: dict[int, np.ndarray] = {}

    for f_res in folds_out:
        hw = 0.0 if f_res["branch"] == "reset" else 1.0
        for local_i, r in enumerate(f_res["target_rows"]):
            mix_mean_ray[:, r, :] = f_res["mean_targets"][:, local_i, :]
            mix_var_ray[:, r, :] = f_res["var_targets"][:, local_i, :]
            mix_llrs[r] = f_res["llrs_targets"][local_i]
            history_weight[r] = hw
            decoded_info[r] = f_res["decoded_targets"][r]

    hw0 = 0.0 if folds_out[0]["branch"] == "reset" else 1.0
    for r_ex in SPARSE_EXTRA_ROWS:
        mix_mean_ray[:, r_ex, :] = extra_mean[r_ex]
        mix_var_ray[:, r_ex, :] = extra_var[r_ex]
        mix_llrs[r_ex] = extra_llrs[r_ex]
        history_weight[r_ex] = hw0
        decoded_info[r_ex] = extra_decoded[r_ex]

    mix_llrs[ANCHOR] = 0.0
    mix_llrs[PROBE_ROW] = 0.0

    # Equal-weight K-fold probe mixture (exact within + between)
    p_stack = np.stack([f["probe_pred"] for f in folds_out], axis=0)
    pv_stack = np.stack([f["probe_var"] for f in folds_out], axis=0)
    mix_probe_pred = np.mean(p_stack, axis=0)
    mix_probe_var = np.mean(pv_stack, axis=0) + np.mean(np.abs(p_stack - mix_probe_pred[None, :]) ** 2, axis=0)
    probe_hw = float(np.mean([0.0 if f["branch"] == "reset" else 1.0 for f in folds_out]))
    history_weight[PROBE_ROW] = probe_hw
    history_weight[ANCHOR] = probe_hw

    ledger_all = [rec for f in folds_out for rec in f["ledger"]]
    rep_weights: dict[int, float] = {}
    assim_rows: list[int] = []
    seen_r: set[int] = set()
    for rec in ledger_all:
        r = int(rec["row"])
        if r not in seen_r:
            seen_r.add(r)
            assim_rows.append(r)
            rep_weights[r] = float(rec["alpha"])

    gate_out = GateOutput(
        history_weight=history_weight,
        probe_weight=probe_hw,
        mix_llrs=mix_llrs,
        mix_probe_pred=np.asarray(mix_probe_pred, dtype=COMPLEX_DTYPE),
        mix_probe_var=np.asarray(mix_probe_var, dtype=REAL_DTYPE),
        mix_mean_ray=mix_mean_ray,
        mix_var_ray=mix_var_ray,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = AssimilationResult(
        gate_out=gate_out,
        decoded_info=decoded_info,
        assimilated_rows=tuple(assim_rows),
        row_weights={r: float(rep_weights.get(r, 0.0)) for r in all_data_rows},
        row_margins={},
        n_rounds=len(FINE_SCHEDULE),
        elapsed_ms=float(elapsed_ms),
    )
    trace = {
        "k_folds": int(k_folds),
        "ledger": ledger_all,
        "fold_branches": [str(f["branch"]) for f in folds_out],
        "fold_targets": t_folds,
        "fold_aux": a_folds,
        "fold_state_versions": [int(f["state_version"]) for f in folds_out],
    }
    return result, trace


__all__ = [
    "MultiRayShiftChannel",
    "PRIMARY_ALL_P12_OOD_REGIMES",
    "PRIMARY_OOD_EXTRA_REGIMES",
    "PRIMARY_OOD_TWO_RAY_SHIFT",
    "PRIMARY_OOD_UNSEEN_ONSET",
    "PRIMARY_UNSEEN_ONSET_ROWS",
    "run_p11_primary",
    "run_p11_with_trace",
    "run_p12_primary_kfold",
    "simulate_p12_primary_link",
    "validate_fold_partition",
]
