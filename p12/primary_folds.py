"""Phase 12 Primary-Axis General K-Fold Symmetric Wavefront Engine & Exact GL3 Diagnostics.

Reuses P11's verified C*-equivariant second-order payload reconstruction and parity-certified
CSR branch-and-bound kernels without modifying any `p11/` file, while generalizing to:
  1. Arbitrary whole-fold excluded partitions (P11 11+5, P12 symmetric K=2 8+8, K=3 6+5+5),
     triggering multi-mode union decoding on each fold's first non-empty wave (`state_version == 0`).
  2. Strict four-arm causal sentinel intervention on locked `ResetBase` (true words, blind decoded,
     and accepted-row truth substitution) with zero target leakage.
  3. Post-chain per-target common-phase oracle & forced-true-branch attribution.
  4. Full 5,832-node exact GL3 covariance posterior vs 160-node compression & continuous 6D
     (start_freq, end_freq) optimization from Top-K modes vs ground-truth initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, logsumexp

from p8.coding import qam
from p8c.link import PILOTS as SC_PILOTS, extra_rows
from p9.channel import DELAYED_INDEX, build_sc_channel_matrix
from p9.contract import CORE_ROWS, DELAYS, LINK_DURATION, N, POWERS, PROBE_ROW, T, PublicObservation
from p9.decoder import decode_strong_sc
from p11.contract import ANCHOR, COMPLEX_DTYPE, REAL_DTYPE
from p11.payload_rebuild import (
    DEFAULT_SCALE_QUAD_ORDER,
    JointResetState,
    OUT,
    ResetBase,
    ShiftedBranchDynamics,
    U_REL,
    _bessel_ratio_and_one_minus_sq,
    _compute_scale_quad_terms_numba,
    _marginalize_scale_moments_numba,
    _strict_row_int,
    _validate_positive_n0,
    build_frame_a_prior_clean,
    default_scale_quadrature,
    estimate_payload_posterior_raw,
    fit_anchor_only,
    gl3_frequency_bank,
    parity_block_cov_score,
    project_joint_moments,
    project_joint_moments_canon,
    project_joint_probe,
    refit_reset,
    scale_tracker_state,
    score_branches_common_units,
)
from p11.wavefront_pipeline import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_AUX_TEMP,
    DEFAULT_MIN_ASSIMILATE_WEIGHT,
    DEFAULT_SIGN_SPLIT_THRESH,
    DEFAULT_VIRTUAL_VAR_FLOOR,
    FINE_SCHEDULE,
    FROZEN_TAU_SWITCH,
    InheritedOrShiftedBranch,
    ResetBranch,
    _decode_single_mode_row,
    _decode_wave1_multimode_union,
    _readout_rows_llrs,
    _sanitize_y_b,
    predictive_row_residual_weight,
)

# Row-to-info-index mapping for sparse layout (training_rows = (10,), extra_rows = (0, 9))
SPARSE_EXTRA_ROWS: tuple[int, ...] = tuple(int(r) for r in extra_rows((ANCHOR,)))
CORE_ROW_TO_INFO_IDX: dict[int, int] = {int(r): i for i, r in enumerate(CORE_ROWS)}
SPARSE_ROW_TO_INFO_IDX: dict[int, int] = {
    **CORE_ROW_TO_INFO_IDX,
    **{int(r): len(CORE_ROWS) + i for i, r in enumerate(SPARSE_EXTRA_ROWS)},
}


# ---------------------------------------------------------------------------
# 1. Exact Error Metrics & Per-Row Readout Helpers
# ---------------------------------------------------------------------------


def row_channel_metrics(
    mean_row_phys: np.ndarray,  # [3, N] complex128
    h_true_row: np.ndarray,     # [3, N] complex128 (truth.h_b_true[:, r, :])
) -> dict[str, float]:
    """Compute Ray-0 phase error (deg), optimal common-phase angle (deg), and H Frobenius errors."""
    m = np.asarray(mean_row_phys, dtype=COMPLEX_DTYPE)
    g = np.asarray(h_true_row, dtype=COMPLEX_DTYPE)
    # Ray-0 mean complex alignment across the 32 samples of row r
    inner0 = complex(np.vdot(g[0], m[0]))  # sum conj(g0) * m0
    ray0_phase_deg = float(abs(math.degrees(math.atan2(inner0.imag, inner0.real)))) if abs(inner0) > 0.0 else 180.0

    # Optimal common-phase alignment across all 3 rays: maximize Re(exp(-i*phi) * <g, m>)
    inner_all = complex(np.vdot(g.ravel(), m.ravel()))
    phi_opt = math.atan2(inner_all.imag, inner_all.real) if abs(inner_all) > 0.0 else 0.0
    common_phase_deg = float(abs(math.degrees(phi_opt)))

    # Channel matrix relative Frobenius error (since delays 0,2,4 have orthogonal shifted columns,
    # ||H_hat - H_true||_F^2 = sum_{l,n} |m[l,n] - g[l,n]|^2 exactly!)
    true_norm = math.sqrt(float(np.sum(g.real ** 2 + g.imag ** 2)))
    diff_raw = m - g
    rel_frob_raw = math.sqrt(float(np.sum(diff_raw.real ** 2 + diff_raw.imag ** 2))) / max(true_norm, 1e-30)

    rot_m = m * complex(math.cos(-phi_opt), math.sin(-phi_opt))
    diff_rot = rot_m - g
    rel_frob_phase_aligned = math.sqrt(float(np.sum(diff_rot.real ** 2 + diff_rot.imag ** 2))) / max(true_norm, 1e-30)

    return {
        "ray0_phase_err_deg": ray0_phase_deg,
        "common_phase_err_deg": common_phase_deg,
        "rel_frob_err": rel_frob_raw,
        "rel_frob_after_cp_oracle": rel_frob_phase_aligned,
        "phi_opt_rad": float(-phi_opt),  # multiply m by exp(1j * phi_opt_rad) to align with g
    }


def decode_rows_strong_l(
    mean_rows_phys: np.ndarray,  # [3, R, N]
    var_rows_phys: np.ndarray,   # [3, R, N]
    y_rows_phys: np.ndarray,     # [R, N]
    rows: Sequence[int],
    n0: float,
    *,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Decode `rows` with frozen Strong-L using C*-canonicalized per-row inputs."""
    n0_val = _validate_positive_n0(n0, name="n0")
    s_mag = math.sqrt(n0_val)
    mean_c = np.asarray(mean_rows_phys, dtype=COMPLEX_DTYPE) / s_mag
    var_c = np.asarray(var_rows_phys, dtype=REAL_DTYPE) / n0_val
    y_c = np.asarray(y_rows_phys, dtype=COMPLEX_DTYPE) / s_mag

    llrs = _readout_rows_llrs(mean_c, var_c, y_c, 1.0)  # [R, N, 4]
    decoded: dict[int, np.ndarray] = {}
    for idx, r in enumerate(rows):
        h_r = build_sc_channel_matrix(mean_c[:, idx, :])
        nv_r = 1.0 + var_c[:, idx, :].sum(axis=0)
        dec = decode_strong_sc(
            y_c[idx],
            h_r,
            nv_r,
            list_size="large",
            bp_iters=bp_iters,
            turbo_pic=True,
            pic_scale=pic_scale,
            initial_llrs=llrs[idx].reshape(128),
        )
        decoded[int(r)] = dec.info_bits.copy()
    return decoded, llrs


# ---------------------------------------------------------------------------
# 2. General Whole-Fold Excluded K-Fold Wavefront Engine
# ---------------------------------------------------------------------------


def _run_generic_fold(
    *,
    fold_idx: int,
    a_prior: Any,
    y10_phys: np.ndarray,
    y_b_safe: np.ndarray,
    aux_rows: tuple[int, ...],
    target_rows: tuple[int, ...],
    all_data_rows: tuple[int, ...],
    n0: float,
    precomputed_anchor: tuple[complex, np.ndarray, float, Any],
    wave_schedule: tuple[tuple[int, ...], ...] = FINE_SCHEDULE,
    forced_branch: Optional[str] = None,
    tau_switch: float = FROZEN_TAU_SWITCH,
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    min_assimilate_weight: float = DEFAULT_MIN_ASSIMILATE_WEIGHT,
    virtual_var_floor: float = DEFAULT_VIRTUAL_VAR_FLOOR,
    sign_split_thresh: float = DEFAULT_SIGN_SPLIT_THRESH,
) -> dict[str, Any]:
    """Run one strictly target-excluded fold; triggers multi-mode union on the FIRST non-empty wave."""
    # Firewall: mask all target_rows to NaN inside inference/waves so target leakage is structurally impossible
    y_inf = y_b_safe.copy()
    y_inf[list(target_rows)] = complex(np.nan, np.nan)

    y_aux_phys = y_inf[list(aux_rows)].copy()
    reset_base = estimate_payload_posterior_raw(
        y10=y10_phys,
        aux_rows=aux_rows,
        y_aux=y_aux_phys,
        n0=n0,
        use_payload=True,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
        precomputed_anchor=precomputed_anchor,
    )
    c_b = complex(reset_base.gauge)
    g2 = float(abs(c_b) ** 2)
    y10_c = y10_phys / c_b

    evidence = score_branches_common_units(
        a_prior=a_prior,
        reset_base=reset_base,
        y10=y10_phys,
        aux_rows=aux_rows,
        y_aux=y_aux_phys,
        n0=n0,
        aux_temp=aux_temp,
        use_parabolic=True,
    )
    t_f = max(evidence.b_s, evidence.b_r) - evidence.b_c
    a_prior_c = scale_tracker_state(a_prior, 1.0 / c_b)

    if forced_branch is not None:
        if forced_branch == "reset":
            branch_kind = "reset"
        elif forced_branch == "shift":
            branch_kind = "shift"
        elif forced_branch == "continuous":
            branch_kind = "continuous"
        else:
            raise ValueError(f"Unsupported forced_branch={forced_branch!r}")
    else:
        if t_f > float(tau_switch):
            branch_kind = "shift" if evidence.b_s >= evidence.b_r else "reset"
        else:
            branch_kind = "continuous"

    if branch_kind == "reset":
        branch: Any = ResetBranch(reset_base, y10_phys)
    elif branch_kind == "shift":
        dyn_s = ShiftedBranchDynamics(
            a_prior_c,
            changed_ray=evidence.best_shift.ray,
            delta_freq=evidence.refined_delta,
            onset_row=evidence.best_shift.onset_row,
        )
        branch = InheritedOrShiftedBranch(kind="shift", dynamics_c=dyn_s, gauge=c_b, n0=n0, y10_c=y10_c)
    else:
        dyn_c = ShiftedBranchDynamics(a_prior_c, changed_ray=None, delta_freq=0.0, onset_row=0)
        branch = InheritedOrShiftedBranch(kind="continuous", dynamics_c=dyn_c, gauge=c_b, n0=n0, y10_c=y10_c)

    ledger: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    ledger_records: list[dict[str, Any]] = []
    state = branch.refit(ledger)
    state_version = 0

    forbidden = set(target_rows) | set(aux_rows) | {ANCHOR, PROBE_ROW}
    helpers = tuple(r for r in all_data_rows if r not in forbidden)
    helper_set = set(helpers)

    first_nonempty_wave_done = False
    for wave_idx, wave in enumerate(wave_schedule):
        wave_rows = tuple(r for r in wave if r in helper_set)
        if not wave_rows:
            continue
        use_multimode = (not first_nonempty_wave_done) or (state_version == 0)
        first_nonempty_wave_done = True

        m_wave_phys, v_wave_phys = branch.predict(state, rows=wave_rows)
        pending: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        pending_records: list[dict[str, Any]] = []

        for local_idx, r in enumerate(wave_rows):
            y_row_c = y_inf[r] / c_b
            if use_multimode:
                res, eval_m_c, eval_v_c = _decode_wave1_multimode_union(
                    y_row_c,
                    branch,
                    state,
                    r,
                    bp_iters=bp_iters,
                    pic_scale=pic_scale,
                    sign_split_thresh=sign_split_thresh,
                )
            else:
                eval_m_c = m_wave_phys[:, local_idx, :] / c_b
                eval_v_c = v_wave_phys[:, local_idx, :] / g2
                res, _ = _decode_single_mode_row(
                    y_row_c,
                    eval_m_c,
                    eval_v_c,
                    bp_iters=bp_iters,
                    pic_scale=pic_scale,
                )

            if res.reliability >= float(min_assimilate_weight):
                x_hat = qam(res.codeword).astype(COMPLEX_DTYPE)
                _, w_resid = predictive_row_residual_weight(eval_m_c, eval_v_c, x_hat, y_row_c, 1.0)
                alpha_eff = float(min(DEFAULT_ALPHA_MAX, res.reliability * w_resid))
                if alpha_eff >= float(min_assimilate_weight):
                    n_eff_phys = n0 * (1.0 / alpha_eff + float(virtual_var_floor))
                    pending[r] = (x_hat, y_inf[r].copy(), n_eff_phys)
                    pending_records.append({
                        "fold": int(fold_idx),
                        "wave": int(wave_idx),
                        "use_multimode": bool(use_multimode),
                        "row": int(r),
                        "injected_info": res.info_bits.copy(),
                        "alpha": alpha_eff,
                        "margin": float(res.margin),
                        "state_version": int(state_version),
                        "branch": branch.kind,
                    })

        if pending:
            ledger.update(pending)
            ledger_records.extend(pending_records)
            state = branch.refit(ledger)
            state_version += 1

    # Read out frozen fold channel predictions on target_rows + probe
    mean_full_c, var_full_c = branch.predict_canon(state, rows=None)
    mean_full_phys = np.asarray(mean_full_c * c_b, dtype=COMPLEX_DTYPE)
    var_full_phys = np.asarray(var_full_c * g2, dtype=REAL_DTYPE)
    probe_pred_phys, probe_var_phys = branch.predict_probe(state, probe_row=PROBE_ROW)

    decoded_targets, llrs_targets = decode_rows_strong_l(
        mean_full_phys[:, list(target_rows), :],
        var_full_phys[:, list(target_rows), :],
        y_b_safe[list(target_rows)],
        target_rows,
        n0,
        bp_iters=bp_iters,
        pic_scale=pic_scale,
    )

    return {
        "fold": int(fold_idx),
        "branch": branch.kind,
        "t_f": float(t_f),
        "b_c": float(evidence.b_c),
        "b_s": float(evidence.b_s),
        "b_r": float(evidence.b_r),
        "helpers_eligible": helpers,
        "helpers_accepted": tuple(int(rec["row"]) for rec in ledger_records),
        "state_version": int(state_version),
        "target_rows": tuple(int(r) for r in target_rows),
        "mean_targets": mean_full_phys[:, list(target_rows), :].copy(),
        "var_targets": var_full_phys[:, list(target_rows), :].copy(),
        "llrs_targets": llrs_targets,
        "decoded_targets": decoded_targets,
        "probe_pred": probe_pred_phys,
        "probe_var": probe_var_phys,
        "ledger": ledger_records,
    }


def run_k_fold_wavefront(
    obs: PublicObservation,
    target_folds: Sequence[tuple[int, ...]],
    aux_folds: Sequence[tuple[int, ...]],
    *,
    forced_branch: Optional[str] = None,
    wave_schedule: tuple[tuple[int, ...], ...] = FINE_SCHEDULE,
    tau_switch: float = FROZEN_TAU_SWITCH,
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> dict[str, Any]:
    """Run an arbitrary K-fold whole-fold-excluded wavefront receiver on sparse obs."""
    t0 = time.perf_counter()
    n0 = _validate_positive_n0(obs.n0, name="obs.n0")
    probe_row = _strict_row_int(obs.probe_row, "obs.probe_row")
    if probe_row != PROBE_ROW or tuple(obs.training_rows) != (ANCHOR,):
        raise ValueError("run_k_fold_wavefront requires sparse layout (10,) and probe_row=19")

    y_b_safe = _sanitize_y_b(obs.y_b, probe_row)
    y10_phys = y_b_safe[ANCHOR].copy()
    a_prior = build_frame_a_prior_clean(obs.y_a, n0)
    precomputed_anchor = fit_anchor_only(y10_phys, n0, scale_quad_order=scale_quad_order)
    all_data_rows = tuple(r for r in range(T) if r not in (ANCHOR, PROBE_ROW))

    folds_out: list[dict[str, Any]] = []
    owner_mean: dict[int, np.ndarray] = {}
    owner_var: dict[int, np.ndarray] = {}
    owner_decoded: dict[int, np.ndarray] = {}
    owner_fold_idx: dict[int, int] = {}

    for f_idx, (t_rows, a_rows) in enumerate(zip(target_folds, aux_folds)):
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
            wave_schedule=wave_schedule,
            forced_branch=forced_branch,
            tau_switch=tau_switch,
            aux_temp=aux_temp,
            scale_quad_order=scale_quad_order,
            bp_iters=bp_iters,
            pic_scale=pic_scale,
        )
        folds_out.append(f_res)
        for local_i, r in enumerate(t_rows):
            owner_mean[int(r)] = f_res["mean_targets"][:, local_i, :]
            owner_var[int(r)] = f_res["var_targets"][:, local_i, :]
            owner_decoded[int(r)] = f_res["decoded_targets"][int(r)]
            owner_fold_idx[int(r)] = int(f_idx)

    # Equal-weight K-fold probe mixture (exact within + between)
    p_stack = np.stack([f["probe_pred"] for f in folds_out], axis=0)
    pv_stack = np.stack([f["probe_var"] for f in folds_out], axis=0)
    p_mean = np.mean(p_stack, axis=0)
    p_var = np.mean(pv_stack, axis=0) + np.mean(np.abs(p_stack - p_mean[None, :]) ** 2, axis=0)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "k_folds": len(folds_out),
        "folds": folds_out,
        "owner_mean": owner_mean,
        "owner_var": owner_var,
        "owner_decoded": owner_decoded,
        "owner_fold_idx": owner_fold_idx,
        "probe_pred": p_mean,
        "probe_var": p_var,
        "elapsed_ms": float(elapsed_ms),
    }


def evaluate_post_chain_common_phase_oracle(
    owner_mean: Mapping[int, np.ndarray],
    owner_var: Mapping[int, np.ndarray],
    y_b_safe: np.ndarray,
    h_b_true: np.ndarray,
    rows: Sequence[int],
    n0: float,
    *,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    """Rotate each target row's predicted channel by its optimal scalar common phase & decode with Strong-L."""
    rot_means = np.empty((len(DELAYS), len(rows), N), dtype=COMPLEX_DTYPE)
    raw_vars = np.empty((len(DELAYS), len(rows), N), dtype=REAL_DTYPE)
    metrics_by_row: dict[int, dict[str, float]] = {}

    for idx, r in enumerate(rows):
        m_r = owner_mean[int(r)]
        v_r = owner_var[int(r)]
        met = row_channel_metrics(m_r, h_b_true[:, int(r), :])
        metrics_by_row[int(r)] = met
        rot_means[:, idx, :] = m_r * complex(math.cos(met["phi_opt_rad"]), math.sin(met["phi_opt_rad"]))
        raw_vars[:, idx, :] = v_r

    dec_cp, _ = decode_rows_strong_l(
        rot_means,
        raw_vars,
        y_b_safe[list(rows)],
        rows,
        n0,
        bp_iters=bp_iters,
        pic_scale=pic_scale,
    )
    return dec_cp, metrics_by_row


# ---------------------------------------------------------------------------
# 3. Four-Arm Causal Sentinel Intervention on Locked ResetBase
# ---------------------------------------------------------------------------


def run_sentinel_arms_on_link(
    obs: PublicObservation,
    truth_b_symbols: np.ndarray,  # [18, N]
    truth_b_info: np.ndarray,     # [18, 64]
    truth_h_b: np.ndarray,        # [3, T, N]
    *,
    sentinel_targets: tuple[int, ...],
    sentinel_aux: tuple[int, ...],
    sentinel_arms: Mapping[str, tuple[int, int]],
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    min_assimilate_weight: float = DEFAULT_MIN_ASSIMILATE_WEIGHT,
    virtual_var_floor: float = DEFAULT_VIRTUAL_VAR_FLOOR,
    sign_split_thresh: float = DEFAULT_SIGN_SPLIT_THRESH,
) -> dict[str, Any]:
    """Run locked-ResetBase 4-arm causal sentinel intervention (true words, blind decoded, accepted truth)."""
    n0 = _validate_positive_n0(obs.n0, name="obs.n0")
    y_b_safe = _sanitize_y_b(obs.y_b, PROBE_ROW)

    # Strictly mask target rows while building ResetBase & decoding helpers (structural zero leakage!)
    y_inf = y_b_safe.copy()
    y_inf[list(sentinel_targets)] = complex(np.nan, np.nan)
    y10_phys = y_inf[ANCHOR].copy()
    y_aux_phys = y_inf[list(sentinel_aux)].copy()

    reset_base = estimate_payload_posterior_raw(
        y10=y10_phys,
        aux_rows=sentinel_aux,
        y_aux=y_aux_phys,
        n0=n0,
        use_payload=True,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
    )
    c_b = complex(reset_base.gauge)
    branch = ResetBranch(reset_base, y10_phys)
    state0 = branch.refit({})

    def _eval_state_on_targets(st: JointResetState) -> dict[str, Any]:
        m_t, v_t = project_joint_moments(st, rows=sentinel_targets)
        dec_t, _ = decode_rows_strong_l(
            m_t, v_t, y_b_safe[list(sentinel_targets)], sentinel_targets, n0, bp_iters=bp_iters, pic_scale=pic_scale
        )
        per_row: dict[str, Any] = {}
        for idx, r in enumerate(sentinel_targets):
            info_idx = SPARSE_ROW_TO_INFO_IDX[int(r)]
            err = int(not np.array_equal(dec_t[int(r)], truth_b_info[info_idx]))
            met = row_channel_metrics(m_t[:, idx, :], truth_h_b[:, int(r), :])
            per_row[str(r)] = {
                "block_error": err,
                "ray0_phase_err_deg": met["ray0_phase_err_deg"],
                "common_phase_err_deg": met["common_phase_err_deg"],
                "rel_frob_err": met["rel_frob_err"],
                "rel_frob_after_cp_oracle": met["rel_frob_after_cp_oracle"],
            }
        return {"per_row": per_row}

    # Baseline 0: payload_only (empty ledger) on the exact same ResetBase
    out: dict[str, Any] = {
        "payload_only": _eval_state_on_targets(state0),
        "arms": {},
    }

    # Also compute true_csi on the exact same 4 sentinel target rows
    h_true_t = truth_h_b[:, list(sentinel_targets), :]
    v_zero_t = np.zeros((len(DELAYS), len(sentinel_targets), N), dtype=REAL_DTYPE)
    dec_oracle, _ = decode_rows_strong_l(
        h_true_t, v_zero_t, y_b_safe[list(sentinel_targets)], sentinel_targets, n0, bp_iters=bp_iters, pic_scale=pic_scale
    )
    out["true_csi"] = {
        "per_row": {
            str(r): {
                "block_error": int(not np.array_equal(dec_oracle[int(r)], truth_b_info[SPARSE_ROW_TO_INFO_IDX[int(r)]])),
            }
            for r in sentinel_targets
        }
    }

    # Nominal blind assimilation variance (matching alpha = alpha_max = 0.75)
    n_eff_nominal = n0 * (1.0 / DEFAULT_ALPHA_MAX + float(virtual_var_floor))

    for arm_name, helper_pair in sentinel_arms.items():
        # 1) True helper words with exact noise variance n0
        ledger_true_n0 = {
            int(r): (truth_b_symbols[SPARSE_ROW_TO_INFO_IDX[int(r)]], y_inf[int(r)].copy(), float(n0))
            for r in helper_pair
        }
        st_true_n0 = branch.refit(ledger_true_n0)

        # 2) True helper words with nominal blind assimilation variance n_eff_nominal
        ledger_true_vf = {
            int(r): (truth_b_symbols[SPARSE_ROW_TO_INFO_IDX[int(r)]], y_inf[int(r)].copy(), float(n_eff_nominal))
            for r in helper_pair
        }
        st_true_vf = branch.refit(ledger_true_vf)

        # 3) Blind multi-mode union decoding on the same pre-wave snapshot state0
        ledger_blind: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        ledger_truth_on_acc: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        helper_diag: list[dict[str, Any]] = []
        for r in helper_pair:
            y_row_c = y_inf[int(r)] / c_b
            res, eval_m_c, eval_v_c = _decode_wave1_multimode_union(
                y_row_c,
                branch,
                state0,
                int(r),
                bp_iters=bp_iters,
                pic_scale=pic_scale,
                sign_split_thresh=sign_split_thresh,
            )
            info_idx = SPARSE_ROW_TO_INFO_IDX[int(r)]
            decoded_correct = bool(np.array_equal(res.info_bits, truth_b_info[info_idx]))
            accepted = False
            alpha_eff = 0.0
            if res.reliability >= float(min_assimilate_weight):
                x_hat = qam(res.codeword).astype(COMPLEX_DTYPE)
                _, w_resid = predictive_row_residual_weight(eval_m_c, eval_v_c, x_hat, y_row_c, 1.0)
                alpha_eff = float(min(DEFAULT_ALPHA_MAX, res.reliability * w_resid))
                if alpha_eff >= float(min_assimilate_weight):
                    accepted = True
                    n_eff_phys = n0 * (1.0 / alpha_eff + float(virtual_var_floor))
                    ledger_blind[int(r)] = (x_hat, y_inf[int(r)].copy(), n_eff_phys)
                    ledger_truth_on_acc[int(r)] = (
                        truth_b_symbols[info_idx],
                        y_inf[int(r)].copy(),
                        n_eff_phys,
                    )
            helper_diag.append({
                "row": int(r),
                "decoded_correct": decoded_correct,
                "accepted": accepted,
                "false_accepted": bool(accepted and (not decoded_correct)),
                "alpha": float(alpha_eff),
                "margin": float(res.margin),
            })

        st_blind = branch.refit(ledger_blind) if ledger_blind else state0
        st_truth_acc = branch.refit(ledger_truth_on_acc) if ledger_truth_on_acc else state0

        out["arms"][arm_name] = {
            "helpers": tuple(int(r) for r in helper_pair),
            "helper_diag": helper_diag,
            "true_words_n0": _eval_state_on_targets(st_true_n0),
            "true_words_vfloor": _eval_state_on_targets(st_true_vf),
            "blind_decoded": _eval_state_on_targets(st_blind),
            "true_on_accepted": _eval_state_on_targets(st_truth_acc),
        }

    return out


# ---------------------------------------------------------------------------
# 4. Exact Full 5,832-Node GL3 vs 160-Node Compression & Continuous 6D Refinement
# ---------------------------------------------------------------------------


def _eval_single_continuous_trajectory_log_ev(
    start_freq: np.ndarray,  # [3]
    end_freq: np.ndarray,    # [3]
    y10_c: np.ndarray,       # [N]
    y_aux_c: np.ndarray,     # [R_aux, N]
    aux_rows: tuple[int, ...],
    scales: tuple[np.ndarray, np.ndarray],
    aux_temp: float = 1.0,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate exact anchor log-evidence + exact parity-block payload log-likelihood at arbitrary 6D (start, end).

    Returns `(total_log_ev, anchor_log_ev, aux_log_ev, mean_c [3], cov_c [3, 3])` in canonical c_B units.
    """
    sf = np.asarray(start_freq, dtype=REAL_DTYPE).reshape(3)
    ef = np.asarray(end_freq, dtype=REAL_DTYPE).reshape(3)
    slope = (ef - sf) / float(LINK_DURATION)
    origin = float(45.984375)  # exact ORIGIN = ABS_TIME[10].mean()
    local_o = sf + slope * origin

    # Evaluate exact rot [3, T, N] for this continuous trajectory
    rot = np.exp(
        2j * np.pi * (local_o[:, None, None] * U_REL[None, :, :] + 0.5 * slope[:, None, None] * (U_REL[None, :, :] ** 2))
    ).astype(COMPLEX_DTYPE)

    # 1) Exact anchor block fit (single node J=1) reusing P11's exact Numba scale-quadrature kernel
    rho_arr, log_eta = scales
    rot10 = rot[:, ANCHOR, :]  # [3, N]
    pilot10 = SC_PILOTS[ANCHOR]
    dicts = np.empty((3, N), dtype=COMPLEX_DTYPE)
    for a in range(3):
        dicts[a] = rot10[a] * np.roll(pilot10, int(DELAYS[a]))

    y_wy = float(np.vdot(y10_c, y10_c).real)
    d0_w_d0 = np.array([float(np.vdot(dicts[0], dicts[0]).real)], dtype=REAL_DTYPE)
    d0_w_y = np.array([complex(np.vdot(dicts[0], y10_c))], dtype=COMPLEX_DTYPE)
    v0_r = np.array([complex(np.vdot(dicts[1], dicts[0]))], dtype=COMPLEX_DTYPE)
    v0_i = np.array([complex(np.vdot(dicts[2], dicts[0]))], dtype=COMPLEX_DTYPE)
    vy_r = np.array([complex(np.vdot(dicts[1], y10_c))], dtype=COMPLEX_DTYPE)
    vy_i = np.array([complex(np.vdot(dicts[2], y10_c))], dtype=COMPLEX_DTYPE)
    g11 = np.array([float(np.vdot(dicts[1], dicts[1]).real)], dtype=REAL_DTYPE)
    g12 = np.array([complex(np.vdot(dicts[1], dicts[2]))], dtype=COMPLEX_DTYPE)
    g22 = np.array([float(np.vdot(dicts[2], dicts[2]).real)], dtype=REAL_DTYPE)

    kappa, phase, c0, c1, bw0, bw1, cw_00, cw_10, cw_11, log_base = _compute_scale_quad_terms_numba(
        np.ascontiguousarray(rho_arr, dtype=REAL_DTYPE),
        g11, g12, g22, d0_w_d0, d0_w_y, v0_r, v0_i, vy_r, vy_i,
        y_wy, 0.0, float(-N * math.log(math.pi)),
        float(POWERS[0]), float(POWERS[1]), float(POWERS[2]),
    )
    a_dom = np.sqrt(rho_arr * float(POWERS[0]))[:, None]
    ratio_a, one_minus_a2 = _bessel_ratio_and_one_minus_sq(kappa)
    m0 = a_dom * phase * ratio_a
    v0 = (rho_arr * float(POWERS[0]))[:, None] * one_minus_a2
    m1 = bw0 - c0 * m0
    m2 = bw1 - c1 * m0
    log_i0 = kappa + np.log(i0e(kappa))
    log_joint = log_eta[:, None] + log_base + log_i0  # [L, 1]
    anchor_le = float(logsumexp(log_joint[:, 0]))
    w_scale = np.exp(log_joint - anchor_le)

    mean_out, cov_out = _marginalize_scale_moments_numba(w_scale, m0, m1, m2, v0, c0, c1, cw_00, cw_10, cw_11)
    second = mean_out[0, :, None] * mean_out[0, None, :].conj() + cov_out[0]
    second = 0.5 * (second + second.conj().T)

    # 2) Exact parity-0/1 16x16 Cholesky payload score on aux_rows
    r_cnt = len(aux_rows)
    half = N // 2
    gains_rot = [rot[a, list(aux_rows), :][:, OUT[a]] for a in range(3)]  # each [R, 32]
    cov = np.zeros((r_cnt, 2, half, half), dtype=COMPLEX_DTYPE)
    for a in range(3):
        for b in range(3):
            cross = second[a, b] * gains_rot[a] * gains_rot[b].conj()
            for parity in (0, 1):
                cov[:, parity, OUT[a, parity::2] // 2, OUT[b, parity::2] // 2] += cross[:, parity::2]
    diag_idx = np.arange(half, dtype=np.int64)
    cov[..., diag_idx, diag_idx] += 1.0
    cov = 0.5 * (cov + cov.conj().swapaxes(-1, -2))
    chol = np.linalg.cholesky(cov)
    diag_l = np.real(np.diagonal(chol, axis1=-2, axis2=-1))
    logdet = 2.0 * np.sum(np.log(np.maximum(diag_l, 1e-300)), axis=-1)
    z = np.stack([y_aux_c[:, ::2], y_aux_c[:, 1::2]], axis=1)  # [R, 2, 16]
    whitened = np.linalg.solve(chol, z[..., None])[..., 0]
    quad = np.sum(whitened.real ** 2 + whitened.imag ** 2, axis=-1)
    aux_le = float((-float(half) * math.log(math.pi) - logdet - quad).sum())

    # Uniform physical support penalty for leaving the [0.045, 0.080] sign-consistent box
    total = anchor_le + float(aux_temp) * aux_le
    return total, anchor_le, aux_le, mean_out[0], cov_out[0]


def run_gl3_exact_and_refinement_diag(
    obs: PublicObservation,
    truth_channel_b: Any,
    truth_b_symbols: np.ndarray,
    truth_b_info: np.ndarray,
    truth_h_b: np.ndarray,
    *,
    aux_rows: tuple[int, ...],
    target_rows: tuple[int, ...],
    bilateral_helpers: tuple[int, int] = (9, 12),
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    top_k_refine: int = 5,
) -> dict[str, Any]:
    """Compare compressed 160-node GL3 vs exact 5,832-node GL3 vs continuous Top-K & truth-init refinement."""
    n0 = _validate_positive_n0(obs.n0, name="obs.n0")
    y_b_safe = _sanitize_y_b(obs.y_b, PROBE_ROW)
    y10_phys = y_b_safe[ANCHOR].copy()
    y_aux_phys = y_b_safe[list(aux_rows)].copy()

    precomp = fit_anchor_only(y10_phys, n0, scale_quad_order=scale_quad_order)
    c_b, y10_c, n_c, anchor_fit = precomp
    y_aux_c = y_aux_phys / c_b
    bank = gl3_frequency_bank()
    scales = default_scale_quadrature(scale_quad_order)

    # 1) Compressed 160-node ResetBase (exact P11 algorithm)
    base_160 = estimate_payload_posterior_raw(
        y10=y10_phys,
        aux_rows=aux_rows,
        y_aux=y_aux_phys,
        n0=n0,
        use_payload=True,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
        precomputed_anchor=precomp,
    )

    # 2) Exact full 5,832-node GL3 scoring (ZERO pruning!)
    t_full0 = time.perf_counter()
    all_idx = np.arange(len(bank.nodes), dtype=np.int64)
    # Chunk by 729 nodes to keep memory low and L2 cache warm
    exact_aux_all = np.empty(len(bank.nodes), dtype=REAL_DTYPE)
    chunk = 729
    for start in range(0, len(bank.nodes), chunk):
        end = min(start + chunk, len(bank.nodes))
        exact_aux_all[start:end] = parity_block_cov_score(
            y_aux_c, aux_rows, bank.nodes[start:end], anchor_fit.second[start:end], n_c=n_c
        )
    log_post_unnorm_5832 = bank.log_gl_prior + anchor_fit.log_evidence + float(aux_temp) * exact_aux_all
    log_ev_5832_canon = float(logsumexp(log_post_unnorm_5832))
    logw_5832 = (log_post_unnorm_5832 - log_ev_5832_canon).astype(REAL_DTYPE)
    w_5832 = np.exp(logw_5832)
    full_score_ms = (time.perf_counter() - t_full0) * 1000.0

    # Map base_160.node_ids back to global 0..5831 indices
    # bank.nodes is ordered lexicographically as n0*324 + n1*18 + n2
    idx_160 = base_160.node_ids[:, 0] * 324 + base_160.node_ids[:, 1] * 18 + base_160.node_ids[:, 2]
    retained_exact_mass = float(np.sum(w_5832[idx_160]))
    omitted_exact_mass = float(max(0.0, 1.0 - retained_exact_mass))
    map_idx_5832 = int(np.argmax(log_post_unnorm_5832))
    map_retained = bool(map_idx_5832 in set(int(i) for i in idx_160))

    p_pos_160 = float(np.sum(np.exp(base_160.logw0)[base_160.sign_ray0]))
    p_pos_5832 = float(np.sum(w_5832[bank.sign_ray0]))
    true_sign_ray0 = bool(float(truth_channel_b.frequency[0]) > 0.0)

    # Construct full 5,832 ResetBase to compare target BLER & phase error both before and after true bilateral helpers
    base_5832 = ResetBase(
        gauge=c_b,
        n0=n0,
        aux_rows=aux_rows,
        use_payload=True,
        aux_temp=aux_temp,
        node_ids=bank.nodes.copy(),
        logw0=logw_5832,
        anchor_log_evidence=anchor_fit.log_evidence.copy(),
        anchor_mean=anchor_fit.mean.copy(),
        anchor_cov=anchor_fit.cov.copy(),
        anchor_second=anchor_fit.second.copy(),
        scales=scales,
        sign_ray0=bank.sign_ray0.copy(),
        class_log_evidence=float(log_ev_5832_canon - float((1 + len(aux_rows)) * N) * math.log(float(abs(c_b) ** 2))),
        class_log_evidence_canon=log_ev_5832_canon,
        pilot_loglik=base_160.pilot_loglik,
        mean=base_160.mean,
        var=base_160.var,
        state_clr_proj=base_160.state_clr_proj,
        probe_pred=base_160.probe_pred,
        probe_var=base_160.probe_var,
    )

    ledger_bilat_true = {
        int(r): (truth_b_symbols[SPARSE_ROW_TO_INFO_IDX[int(r)]], y_b_safe[int(r)].copy(), float(n0))
        for r in bilateral_helpers
    }

    def _eval_base(base_obj: ResetBase) -> dict[str, Any]:
        st0 = base_obj.to_joint_state()
        st_b = refit_reset(base_obj, y10_phys, ledger_bilat_true)
        res_pair: dict[str, Any] = {}
        for tag, st in (("no_helper", st0), ("true_bilateral_helpers", st_b)):
            m_t, v_t = project_joint_moments(st, rows=target_rows)
            dec_t, _ = decode_rows_strong_l(m_t, v_t, y_b_safe[list(target_rows)], target_rows, n0)
            errs = sum(
                int(not np.array_equal(dec_t[int(r)], truth_b_info[SPARSE_ROW_TO_INFO_IDX[int(r)]]))
                for r in target_rows
            )
            frob_list = [row_channel_metrics(m_t[:, i, :], truth_h_b[:, int(r), :])["rel_frob_err"] for i, r in enumerate(target_rows)]
            ph_list = [row_channel_metrics(m_t[:, i, :], truth_h_b[:, int(r), :])["ray0_phase_err_deg"] for i, r in enumerate(target_rows)]
            res_pair[tag] = {
                "block_errors": int(errs),
                "blocks": len(target_rows),
                "mean_rel_frob_err": float(np.mean(frob_list)),
                "mean_ray0_phase_err_deg": float(np.mean(ph_list)),
            }
        return res_pair

    eval_160 = _eval_base(base_160)
    eval_5832 = _eval_base(base_5832)

    # 3) Continuous 6D refinement (start_freq[3], end_freq[3]) from Top-K diverse 5,832 modes vs truth init!
    # Pick top-K modes from exact 5,832 posterior, ensuring at least one from each Ray-0 sign
    sorted_nodes = np.argsort(-log_post_unnorm_5832)
    seed_modes: list[int] = []
    seen_triples: list[np.ndarray] = []
    # Force top mode from positive Ray-0 and top mode from negative Ray-0 first
    for sgn_flag in (True, False):
        for nid in sorted_nodes:
            if bool(bank.sign_ray0[int(nid)]) == sgn_flag:
                seed_modes.append(int(nid))
                seen_triples.append(bank.nodes[int(nid)])
                break
    for nid in sorted_nodes:
        if len(seed_modes) >= int(top_k_refine):
            break
        n_trip = bank.nodes[int(nid)]
        if all(np.sum(n_trip != prev) >= 2 for prev in seen_triples):
            seed_modes.append(int(nid))
            seen_triples.append(n_trip)

    def _optimize_from(sf_init: np.ndarray, ef_init: np.ndarray) -> dict[str, Any]:
        signs = np.where(sf_init >= 0.0, 1.0, -1.0)
        x0 = np.concatenate([np.abs(sf_init), np.abs(ef_init)])
        bounds = [(0.045, 0.080)] * 6

        def _neg_obj(x_mag: np.ndarray) -> float:
            sf_v = signs * x_mag[:3]
            ef_v = signs * x_mag[3:]
            tot, _, _, _, _ = _eval_single_continuous_trajectory_log_ev(
                sf_v, ef_v, y10_c, y_aux_c, aux_rows, scales, aux_temp=aux_temp
            )
            return -float(tot)

        init_val = -_neg_obj(x0)
        res_opt = minimize(_neg_obj, x0, method="Powell", bounds=bounds, options={"maxiter": 60, "ftol": 1e-4, "xtol": 1e-4})
        x_best = np.clip(res_opt.x, 0.045, 0.080)
        sf_b = signs * x_best[:3]
        ef_b = signs * x_best[3:]
        tot_b, anc_b, aux_b, _, _ = _eval_single_continuous_trajectory_log_ev(
            sf_b, ef_b, y10_c, y_aux_c, aux_rows, scales, aux_temp=aux_temp
        )
        return {
            "init_log_ev": float(init_val),
            "opt_log_ev": float(tot_b),
            "anchor_log_ev": float(anc_b),
            "aux_log_ev": float(aux_b),
            "start_freq": [float(v) for v in sf_b],
            "end_freq": [float(v) for v in ef_b],
            "nfev": int(res_opt.nfev),
        }

    top_k_results: list[dict[str, Any]] = []
    for nid in seed_modes:
        trip = bank.nodes[nid]
        r_opt = _optimize_from(bank.start_freq[trip], bank.end_freq[trip])
        r_opt["grid_node_id"] = int(nid)
        r_opt["grid_log_ev_no_prior"] = float(anchor_fit.log_evidence[nid] + float(aux_temp) * exact_aux_all[nid])
        top_k_results.append(r_opt)

    best_topk = max(top_k_results, key=lambda d: d["opt_log_ev"])

    # Ground-truth initialization on exact same objective (diagnoses basin search failure!)
    sf_true = np.asarray(truth_channel_b.frequency, dtype=REAL_DTYPE)
    ef_true = sf_true + np.asarray(truth_channel_b.chirp, dtype=REAL_DTYPE) * float(LINK_DURATION)
    truth_at_exact_tot, _, _, _, _ = _eval_single_continuous_trajectory_log_ev(
        sf_true, ef_true, y10_c, y_aux_c, aux_rows, scales, aux_temp=aux_temp
    )
    truth_opt = _optimize_from(sf_true, ef_true)
    truth_opt["exact_truth_log_ev"] = float(truth_at_exact_tot)

    best_grid_log_ev_no_prior = float(np.max(anchor_fit.log_evidence + float(aux_temp) * exact_aux_all))
    best_160_log_ev_no_prior = float(np.max(anchor_fit.log_evidence[idx_160] + float(aux_temp) * exact_aux_all[idx_160]))

    return {
        "full_5832_score_ms": float(full_score_ms),
        "n_nodes_160": int(len(idx_160)),
        "omitted_exact_mass": omitted_exact_mass,
        "map_retained_in_160": map_retained,
        "log_ev_diff_160_minus_5832": float(base_160.class_log_evidence_canon - log_ev_5832_canon),
        "true_sign_ray0_pos": true_sign_ray0,
        "p_true_sign_160": float(p_pos_160 if true_sign_ray0 else 1.0 - p_pos_160),
        "p_true_sign_5832": float(p_pos_5832 if true_sign_ray0 else 1.0 - p_pos_5832),
        "eval_160": eval_160,
        "eval_5832": eval_5832,
        "best_160_node_log_ev": best_160_log_ev_no_prior,
        "best_5832_node_log_ev": best_grid_log_ev_no_prior,
        "best_topk_refined_log_ev": float(best_topk["opt_log_ev"]),
        "truth_init_refined_log_ev": float(truth_opt["opt_log_ev"]),
        "exact_truth_log_ev": float(truth_at_exact_tot),
        # Positive gap = compression loss / discretization loss / basin search loss
        "compression_peak_gap": float(best_grid_log_ev_no_prior - best_160_log_ev_no_prior),
        "grid_discretization_gap": float(best_topk["opt_log_ev"] - best_grid_log_ev_no_prior),
        "basin_search_gap": float(truth_opt["opt_log_ev"] - best_topk["opt_log_ev"]),
        "truth_sign_matched_by_topk": bool(
            (best_topk["start_freq"][0] > 0.0) == true_sign_ray0
        ),
        "top_k_modes": top_k_results,
        "truth_opt": truth_opt,
    }


__all__ = [
    "CORE_ROW_TO_INFO_IDX",
    "SPARSE_EXTRA_ROWS",
    "SPARSE_ROW_TO_INFO_IDX",
    "decode_rows_strong_l",
    "evaluate_post_chain_common_phase_oracle",
    "row_channel_metrics",
    "run_gl3_exact_and_refinement_diag",
    "run_k_fold_wavefront",
    "run_sentinel_arms_on_link",
]
