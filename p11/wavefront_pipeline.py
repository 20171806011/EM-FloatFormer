"""Phase 11 Feature F5: Chained Payload Covariance Reconstruction + Parity-Certified Wavefront Assimilation.

Chains `p11.payload_rebuild` (`ResetBase` GL3 joint posterior, scale quadrature,
persistent `ShiftedBranchDynamics`, comparable common-unit 3-branch scoring) into a
strictly fold-isolated, parity-certified CSR branch-and-bound wavefront loop:
  - Strict whole-fold isolation:
      Fold 0: AUX = (2, 5, 8, 13, 16), TARGETS = (1, 3, 4, 6, 7, 11, 12, 14, 15, 17, 18),
              helpers = (0, 9), Wave 1 = (9,)
      Fold 1: AUX = (1, 4, 7, 12, 15), TARGETS = (2, 5, 8, 13, 16),
              helpers = (0, 3, 6, 9, 11, 14, 17, 18), Wave 1 = (9, 11)
  - Strict `PROBE_ROW = 19` firewall: masked to `NaN + 1j*NaN` on entry, never indexed.
  - Persistent branch state (`ContinuousBranch`, `ShiftBranch`, `ResetBranch`) kept alive
    across all wavefront sweeps (`state_version >= 1`).
  - Wave-1 multi-mode union candidate scoring across signed Ray-0 modes with exact
    word-dependent `V_b(x)` log-evidence LSE.
  - Final frozen readout runs `decode_strong_sc(..., list_size='large', bp_iters=20,
    turbo_pic=True, pic_scale=0.75)` on EVERY core owner row (zero fast-lock skips).
  - Equal-weight 2-fold probe mixture (`v_p >= n0`) and finite `0.0` LLRs on `(10, 19)`.

Strict invariants:
  - ZERO hardcoded N0 anywhere.
  - ZERO `torch.inference_mode()`.
  - Strictly dev/train/e2e namespaces in primary/dev code (never references formal seed pools).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence

from numba import njit
import numpy as np
from scipy.special import logsumexp

from p8.coding import PAM, generator, parity_check, qam, qam_llrs
from p8.tracker import State
from p8c.link import PILOTS as SC_PILOTS, extra_rows
from p9.channel import DELAYED_INDEX, SAMPLES, SWEEPS, build_sc_channel_matrix
from p9.contract import CORE_ROWS, DELAYS, N, PROBE_ROW, T, GateOutput, PublicObservation
from p9.decoder import (
    bp_posterior,
    decode_strong_sc,
    is_valid_codeword,
    pack_codewords,
    unpack_codewords,
    warmup_numba as _warmup_p9_numba,
)
from p11.contract import ANCHOR, COMPLEX_DTYPE, REAL_DTYPE, AssimilationResult
from p11.payload_rebuild import (
    DEFAULT_SCALE_QUAD_ORDER,
    JointResetState,
    ResetBase,
    ShiftedBranchDynamics,
    _strict_row_int,
    b_only_canonical_gauge,
    build_frame_a_prior_clean,
    estimate_payload_posterior_raw,
    fit_anchor_only,
    project_joint_moments,
    project_joint_moments_canon,
    project_joint_probe,
    refit_reset,
    scale_tracker_state,
    score_branches_common_units,
    split_joint_reset_ray0_sign,
    warmup_payload_numba,
)


# ---------------------------------------------------------------------------
# Frozen Fold Partitions, Wavefront Schedule & Calibrated Hyperparameters
# ---------------------------------------------------------------------------

AUX_FOLDS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (2, 5, 8, 13, 16),
    (1, 4, 7, 12, 15),
)
FOLD_TARGETS: tuple[tuple[int, ...], tuple[int, ...]] = (
    tuple(r for r in CORE_ROWS if r not in AUX_FOLDS[0]),
    AUX_FOLDS[0],
)
assert set(FOLD_TARGETS[0]).isdisjoint(FOLD_TARGETS[1])
assert set(FOLD_TARGETS[0] + FOLD_TARGETS[1]) == set(CORE_ROWS)

FINE_SCHEDULE: tuple[tuple[int, ...], ...] = (
    (9, 11),
    (8, 12),
    (4, 5, 6, 7, 13, 14, 15, 16),
    (0, 1, 2, 3, 17, 18),
)

# Frozen reliability & wavefront parameters (all dimensionless & C*-invariant)
DEFAULT_MARGIN_MIN: float = 4.5
DEFAULT_MARGIN_SCALE: float = 5.5
DEFAULT_ALPHA_MAX: float = 0.75
DEFAULT_BP_UNCONVERGED_PENALTY: float = 2.0
DEFAULT_FIT_NU: float = 4.0
DEFAULT_FIT_CUTOFF: float = 3.5
DEFAULT_RESID_NU: float = 4.0
DEFAULT_RESID_CUTOFF: float = 3.5
DEFAULT_MIN_ASSIMILATE_WEIGHT: float = 0.08
DEFAULT_VIRTUAL_VAR_FLOOR: float = 0.15
DEFAULT_SIGN_SPLIT_THRESH: float = 0.05
DEFAULT_AUX_TEMP: float = 1.0

# Continuous-protection threshold on T_f = max(B_S, B_R) - B_C (calibrated strictly on SEED_P11_TRAIN:
# continuous 95th pct = 3.28, partial_shift 10th pct = 7.03, full_reset 10th pct = 30.41)
FROZEN_TAU_SWITCH: float = 3.50

# Precomputed read-only GF(2) & 16-QAM tables (matching p8.coding / p9.decoder)
_H_MATRIX = parity_check()
_G_MATRIX = generator()
_G_MATRIX.setflags(write=False)

_GEN_U64 = np.ascontiguousarray(
    np.packbits(_G_MATRIX.astype(np.uint8), axis=1, bitorder="little").view("<u8")
)
_GEN_U64.setflags(write=False)

_H_U64 = np.ascontiguousarray(
    np.packbits(_H_MATRIX.astype(np.uint8), axis=1, bitorder="little").view("<u8")
)
_H_U64.setflags(write=False)

_NIBBLE_RE = np.ascontiguousarray(
    [PAM[2 * (nib & 1) + ((nib >> 1) & 1)] for nib in range(16)], dtype=REAL_DTYPE
)
_NIBBLE_RE.setflags(write=False)

_NIBBLE_IM = np.ascontiguousarray(
    [PAM[2 * ((nib >> 2) & 1) + ((nib >> 3) & 1)] for nib in range(16)], dtype=REAL_DTYPE
)
_NIBBLE_IM.setflags(write=False)

_EMPTY_U64 = np.empty((0, 2), dtype=np.uint64)
_EMPTY_U64.setflags(write=False)


# ---------------------------------------------------------------------------
# 1. Numba Parity-Certified CSR Branch-and-Bound & Multi-Mode Union Scorers
# ---------------------------------------------------------------------------


@njit
def _parity64(x: np.uint64) -> np.uint64:
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return x & np.uint64(1)


@njit
def _is_valid_u64(c0: np.uint64, c1: np.uint64, h_u64: np.ndarray) -> bool:
    for r in range(h_u64.shape[0]):
        p = _parity64(c0 & h_u64[r, 0]) ^ _parity64(c1 & h_u64[r, 1])
        if p != np.uint64(0):
            return False
    return True


@njit
def _waveform_metric_csr_bounded(
    c0: np.uint64,
    c1: np.uint64,
    y_re: np.ndarray,
    y_im: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    val_re: np.ndarray,
    val_im: np.ndarray,
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
    x_re: np.ndarray,
    x_im: np.ndarray,
    cutoff: float,
) -> float:
    for s in range(16):
        nib0 = int((c0 >> np.uint64(4 * s)) & np.uint64(15))
        x_re[s] = nibble_re[nib0]
        x_im[s] = nibble_im[nib0]
        nib1 = int((c1 >> np.uint64(4 * s)) & np.uint64(15))
        x_re[16 + s] = nibble_re[nib1]
        x_im[16 + s] = nibble_im[nib1]

    total = 0.0
    n_rows = y_re.shape[0]
    for r in range(n_rows):
        acc_re = 0.0
        acc_im = 0.0
        start = row_ptr[r]
        end = row_ptr[r + 1]
        for p in range(start, end):
            j = col_idx[p]
            vr = val_re[p]
            vi = val_im[p]
            xr = x_re[j]
            xi = x_im[j]
            acc_re += vr * xr - vi * xi
            acc_im += vr * xi + vi * xr
        dr = y_re[r] - acc_re
        di = y_im[r] - acc_im
        total += dr * dr + di * di
        if total >= cutoff:
            return total
    return total


@njit
def _scan_best_and_runner_up_u64(
    seed_c0: np.uint64,
    seed_c1: np.uint64,
    cand_u64: np.ndarray,
    gen_u64: np.ndarray,
    h_u64: np.ndarray,
    y_re: np.ndarray,
    y_im: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    val_re: np.ndarray,
    val_im: np.ndarray,
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
):
    x_re = np.empty(32, dtype=np.float64)
    x_im = np.empty(32, dtype=np.float64)

    best_d = np.inf
    best_c0 = np.uint64(0)
    best_c1 = np.uint64(0)
    has_best = False

    second_d = np.inf
    second_c0 = np.uint64(0)
    second_c1 = np.uint64(0)
    n_scored = 0

    if _is_valid_u64(seed_c0, seed_c1, h_u64):
        d = _waveform_metric_csr_bounded(
            seed_c0, seed_c1, y_re, y_im, row_ptr, col_idx, val_re, val_im,
            nibble_re, nibble_im, x_re, x_im, np.inf,
        )
        best_d = d
        best_c0 = seed_c0
        best_c1 = seed_c1
        has_best = True
        n_scored += 1

    if has_best:
        base_c0 = best_c0
        base_c1 = best_c1
        for i in range(gen_u64.shape[0]):
            c0 = base_c0 ^ gen_u64[i, 0]
            c1 = base_c1 ^ gen_u64[i, 1]
            if c0 == best_c0 and c1 == best_c1:
                continue
            n_scored += 1
            d = _waveform_metric_csr_bounded(
                c0, c1, y_re, y_im, row_ptr, col_idx, val_re, val_im,
                nibble_re, nibble_im, x_re, x_im, second_d,
            )
            if d >= second_d:
                continue
            if not _is_valid_u64(c0, c1, h_u64):
                continue
            if d < best_d:
                second_d = best_d
                second_c0 = best_c0
                second_c1 = best_c1
                best_d = d
                best_c0 = c0
                best_c1 = c1
            elif d < second_d:
                second_d = d
                second_c0 = c0
                second_c1 = c1

    for idx in range(cand_u64.shape[0]):
        c0 = cand_u64[idx, 0]
        c1 = cand_u64[idx, 1]
        if has_best and c0 == best_c0 and c1 == best_c1:
            continue
        n_scored += 1
        cutoff = second_d if has_best else np.inf
        d = _waveform_metric_csr_bounded(
            c0, c1, y_re, y_im, row_ptr, col_idx, val_re, val_im,
            nibble_re, nibble_im, x_re, x_im, cutoff,
        )
        if has_best and d >= second_d:
            continue
        if not _is_valid_u64(c0, c1, h_u64):
            continue
        if not has_best or d < best_d:
            if has_best:
                second_d = best_d
                second_c0 = best_c0
                second_c1 = best_c1
            best_d = d
            best_c0 = c0
            best_c1 = c1
            has_best = True
        elif d < second_d:
            second_d = d
            second_c0 = c0
            second_c1 = c1

    if has_best and (best_c0 != seed_c0 or best_c1 != seed_c1):
        for _pass in range(2):
            improved = False
            base_c0 = best_c0
            base_c1 = best_c1
            for i in range(gen_u64.shape[0]):
                c0 = base_c0 ^ gen_u64[i, 0]
                c1 = base_c1 ^ gen_u64[i, 1]
                if c0 == best_c0 and c1 == best_c1:
                    continue
                n_scored += 1
                d = _waveform_metric_csr_bounded(
                    c0, c1, y_re, y_im, row_ptr, col_idx, val_re, val_im,
                    nibble_re, nibble_im, x_re, x_im, second_d,
                )
                if d >= second_d:
                    continue
                if not _is_valid_u64(c0, c1, h_u64):
                    continue
                if d < best_d:
                    second_d = best_d
                    second_c0 = best_c0
                    second_c1 = best_c1
                    best_d = d
                    best_c0 = c0
                    best_c1 = c1
                    improved = True
                elif d < second_d:
                    second_d = d
                    second_c0 = c0
                    second_c1 = c1
            if not improved:
                break

    return best_c0, best_c1, best_d, second_c0, second_c1, second_d, n_scored


@njit
def _score_multimode_union_u64(
    cand_u64: np.ndarray,      # [M, 2] uint64 (deduplicated)
    h_u64: np.ndarray,
    y_re: np.ndarray,          # [32]
    y_im: np.ndarray,          # [32]
    log_p: np.ndarray,         # [B]
    h_re: np.ndarray,          # [B, 32, 32]
    h_im: np.ndarray,          # [B, 32, 32]
    v_ray: np.ndarray,         # [B, 3, 32]
    n_c: float,
    delays: np.ndarray,        # [3]
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
):
    """Score each distinct valid codeword x via exact multi-mode LSE with word-dependent V_b(x)."""
    m_cnt = cand_u64.shape[0]
    b_cnt = log_p.shape[0]
    n_sym = y_re.shape[0]
    log_pi = math.log(math.pi)

    x_re = np.empty(n_sym, dtype=np.float64)
    x_im = np.empty(n_sym, dtype=np.float64)
    x_pow = np.empty(n_sym, dtype=np.float64)
    mode_ll = np.empty(b_cnt, dtype=np.float64)
    mode_fit = np.empty(b_cnt, dtype=np.float64)

    best_ll = -np.inf
    best_c0 = np.uint64(0)
    best_c1 = np.uint64(0)
    best_mode = 0
    best_fit = np.inf

    second_ll = -np.inf
    second_c0 = np.uint64(0)
    second_c1 = np.uint64(0)
    n_valid = 0

    for idx in range(m_cnt):
        c0 = cand_u64[idx, 0]
        c1 = cand_u64[idx, 1]
        if not _is_valid_u64(c0, c1, h_u64):
            continue
        n_valid += 1

        for s in range(16):
            nib0 = int((c0 >> np.uint64(4 * s)) & np.uint64(15))
            xr0 = nibble_re[nib0]
            xi0 = nibble_im[nib0]
            x_re[s] = xr0
            x_im[s] = xi0
            x_pow[s] = xr0 * xr0 + xi0 * xi0

            nib1 = int((c1 >> np.uint64(4 * s)) & np.uint64(15))
            xr1 = nibble_re[nib1]
            xi1 = nibble_im[nib1]
            x_re[16 + s] = xr1
            x_im[16 + s] = xi1
            x_pow[16 + s] = xr1 * xr1 + xi1 * xi1

        d0 = int(delays[0])
        d1 = int(delays[1])
        d2 = int(delays[2])

        max_mll = -np.inf
        win_m = 0
        win_fit = np.inf
        for b in range(b_cnt):
            ll_b = log_p[b]
            chi2_b = 0.0
            for n in range(n_sym):
                acc_re = 0.0
                acc_im = 0.0
                for col in range(n_sym):
                    hr = h_re[b, n, col]
                    hi = h_im[b, n, col]
                    xr = x_re[col]
                    xi = x_im[col]
                    acc_re += hr * xr - hi * xi
                    acc_im += hr * xi + hi * xr
                dr = y_re[n] - acc_re
                di = y_im[n] - acc_im
                r2 = dr * dr + di * di

                i0 = (n - d0) % n_sym
                i1 = (n - d1) % n_sym
                i2 = (n - d2) % n_sym
                v_bn = (
                    n_c
                    + v_ray[b, 0, n] * x_pow[i0]
                    + v_ray[b, 1, n] * x_pow[i1]
                    + v_ray[b, 2, n] * x_pow[i2]
                )
                ratio = r2 / v_bn
                chi2_b += ratio
                ll_b -= math.log(v_bn) + log_pi + ratio

            mode_ll[b] = ll_b
            mode_fit[b] = chi2_b / float(n_sym)
            if ll_b > max_mll:
                max_mll = ll_b
                win_m = b
                win_fit = mode_fit[b]

        sum_exp = 0.0
        for b in range(b_cnt):
            sum_exp += math.exp(mode_ll[b] - max_mll)
        l_x = max_mll + math.log(sum_exp)

        if l_x > best_ll:
            if n_valid > 1:
                second_ll = best_ll
                second_c0 = best_c0
                second_c1 = best_c1
            best_ll = l_x
            best_c0 = c0
            best_c1 = c1
            best_mode = win_m
            best_fit = win_fit
        elif l_x > second_ll:
            second_ll = l_x
            second_c0 = c0
            second_c1 = c1

    return best_c0, best_c1, best_ll, second_c0, second_c1, second_ll, best_mode, best_fit, n_valid


def _prepare_weighted_csr(y_r: np.ndarray, h_r: np.ndarray, noise_r: np.ndarray):
    y = np.asarray(y_r, dtype=COMPLEX_DTYPE)
    h = np.asarray(h_r, dtype=COMPLEX_DTYPE)
    nv = np.asarray(noise_r, dtype=REAL_DTYPE)
    w = 1.0 / np.sqrt(nv)
    y_w = y * w
    a = h * w[:, None]

    nz_mask = (a.real != 0.0) | (a.imag != 0.0)
    row_counts = nz_mask.sum(axis=1).astype(np.int64)
    row_ptr = np.empty(N + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_counts, out=row_ptr[1:])
    nz_rows, nz_cols = np.nonzero(nz_mask)
    col_idx = np.ascontiguousarray(nz_cols.astype(np.int64))
    vals = a[nz_rows, nz_cols]
    val_re = np.ascontiguousarray(vals.real.astype(REAL_DTYPE))
    val_im = np.ascontiguousarray(vals.imag.astype(REAL_DTYPE))
    return (
        np.ascontiguousarray(y_w.real.astype(REAL_DTYPE)),
        np.ascontiguousarray(y_w.imag.astype(REAL_DTYPE)),
        row_ptr,
        col_idx,
        val_re,
        val_im,
    )


def warmup_p11_numba() -> None:
    """Trigger Numba JIT compilation cleanly across P9 and P11 kernels."""
    _warmup_p9_numba()
    warmup_payload_numba()
    dummy_y = np.zeros(N, dtype=COMPLEX_DTYPE)
    dummy_h = np.eye(N, dtype=COMPLEX_DTYPE)
    dummy_nv = np.ones(N, dtype=REAL_DTYPE)
    y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(dummy_y, dummy_h, dummy_nv)
    _scan_best_and_runner_up_u64(
        np.uint64(0),
        np.uint64(0),
        np.zeros((1, 2), dtype=np.uint64),
        _GEN_U64,
        _H_U64,
        y_re,
        y_im,
        row_ptr,
        col_idx,
        val_re,
        val_im,
        _NIBBLE_RE,
        _NIBBLE_IM,
    )
    _score_multimode_union_u64(
        np.zeros((1, 2), dtype=np.uint64),
        _H_U64,
        np.zeros(N, dtype=REAL_DTYPE),
        np.zeros(N, dtype=REAL_DTYPE),
        np.zeros(1, dtype=REAL_DTYPE),
        np.zeros((1, N, N), dtype=REAL_DTYPE),
        np.zeros((1, N, N), dtype=REAL_DTYPE),
        np.ones((1, len(DELAYS), N), dtype=REAL_DTYPE),
        1.0,
        DELAYS,
        _NIBBLE_RE,
        _NIBBLE_IM,
    )


# ---------------------------------------------------------------------------
# 2. Blind Margin, Redescending Weights & Row Readout Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlindMarginDecodeResult:
    info_bits: np.ndarray
    codeword: np.ndarray
    d_best: float
    d_second: float
    margin: float
    fit_ratio: float
    bp_converged: bool
    reliability: float
    candidates_u64: np.ndarray
    n_candidates: int
    elapsed_ms: float


def redescending_residual_weight(
    ratio: float,
    *,
    nu: float = DEFAULT_FIT_NU,
    cutoff: float = DEFAULT_FIT_CUTOFF,
) -> float:
    r = float(ratio)
    if not math.isfinite(r) or r < 0.0:
        return 0.0
    if r <= 1.0:
        return 1.0
    cut = float(cutoff)
    if cut <= 1.0 or r >= cut:
        return 0.0
    excess = r - 1.0
    span = cut - 1.0
    student_factor = float(nu) / (float(nu) + excess * excess)
    tukey_factor = (1.0 - (excess / span) ** 2) ** 2
    return float(max(0.0, min(1.0, student_factor * tukey_factor)))


def reliability_weight_from_margin(
    margin: float,
    bp_converged: bool,
    fit_ratio: float = 1.0,
    *,
    margin_min: float = DEFAULT_MARGIN_MIN,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    bp_unconverged_penalty: float = DEFAULT_BP_UNCONVERGED_PENALTY,
    fit_nu: float = DEFAULT_FIT_NU,
    fit_cutoff: float = DEFAULT_FIT_CUTOFF,
) -> float:
    m = float(margin)
    amax = float(alpha_max)
    if amax <= 0.0 or not math.isfinite(m):
        return 0.0
    eff_min = float(margin_min) if bool(bp_converged) else float(margin_min) + float(bp_unconverged_penalty)
    if m <= eff_min:
        return 0.0
    scale = max(float(margin_scale), 1e-12)
    w_margin = amax * (1.0 - math.exp(-(m - eff_min) / scale))
    w_fit = redescending_residual_weight(fit_ratio, nu=fit_nu, cutoff=fit_cutoff)
    return float(max(0.0, min(amax, w_margin * w_fit)))


def _canonicalize_block_cstar(
    y_r: np.ndarray,
    h_r: np.ndarray,
    noise_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale `(y_r, h_r, noise_r)` locally to O(1) without any physical-unit floor."""
    y = np.asarray(y_r, dtype=COMPLEX_DTYPE).ravel()
    h = np.asarray(h_r, dtype=COMPLEX_DTYPE)
    nv = np.asarray(noise_r, dtype=REAL_DTYPE).ravel()
    nv_mean = float(np.mean(nv))
    h_pow = float(np.mean(np.abs(h) ** 2))
    var_ref = nv_mean if nv_mean > 0.0 else (h_pow if h_pow > 0.0 else 1.0)
    amp_scale = math.sqrt(var_ref)

    h_norm = float(np.linalg.norm(h))
    diag_sum = complex(np.sum(np.diag(h)))
    if h_norm > 0.0 and abs(diag_sum) > 1e-10 * h_norm:
        unit_phase = diag_sum / abs(diag_sum)
    elif h_norm > 0.0:
        h_flat = h.ravel()
        dom_val = complex(h_flat[int(np.argmax(np.abs(h_flat)))])
        unit_phase = dom_val / abs(dom_val) if abs(dom_val) > 0.0 else 1.0 + 0.0j
    else:
        unit_phase = 1.0 + 0.0j

    c_scale = complex(amp_scale * unit_phase)
    return y / c_scale, h / c_scale, nv / var_ref


def decode_with_blind_margin(
    y_r: np.ndarray,
    h_r: np.ndarray,
    noise_r: np.ndarray,
    *,
    list_size: str = "large",
    bp_iters: int = 20,
    turbo_pic: bool = True,
    pic_scale: float = 0.75,
    initial_llrs: Optional[np.ndarray] = None,
) -> BlindMarginDecodeResult:
    t0 = time.perf_counter()
    y_c, h_c, nv_c = _canonicalize_block_cstar(y_r, h_r, noise_r)
    res = decode_strong_sc(
        y_c,
        h_c,
        nv_c,
        list_size=list_size,
        bp_iters=bp_iters,
        turbo_pic=turbo_pic,
        pic_scale=pic_scale,
        initial_llrs=initial_llrs,
    )

    seed_u64 = pack_codewords(res.codeword)
    y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(y_c, h_c, nv_c)
    best_c0, best_c1, d_best, _s0, _s1, d_second, n_scored = _scan_best_and_runner_up_u64(
        seed_u64[0],
        seed_u64[1],
        res.candidates_u64,
        _GEN_U64,
        _H_U64,
        y_re,
        y_im,
        row_ptr,
        col_idx,
        val_re,
        val_im,
        _NIBBLE_RE,
        _NIBBLE_IM,
    )
    codeword = unpack_codewords(np.array([best_c0, best_c1], dtype=np.uint64))
    if not is_valid_codeword(codeword):
        raise RuntimeError("Internal invariant failure: decoded codeword failed parity check")
    margin = float(max(0.0, float(d_second) - float(d_best)))
    fit_ratio = float(d_best) / float(N)
    reliability = reliability_weight_from_margin(margin, res.bp_converged, fit_ratio=fit_ratio)
    return BlindMarginDecodeResult(
        info_bits=codeword[:64].copy(),
        codeword=codeword,
        d_best=float(d_best),
        d_second=float(d_second),
        margin=margin,
        fit_ratio=fit_ratio,
        bp_converged=bool(res.bp_converged),
        reliability=reliability,
        candidates_u64=res.candidates_u64,
        n_candidates=int(n_scored),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def predictive_row_residual_weight(
    mean_ray_row: np.ndarray,
    var_ray_row: np.ndarray,
    x_hat: np.ndarray,
    y_b_row: np.ndarray,
    n0: float,
    *,
    nu: float = DEFAULT_RESID_NU,
    cutoff: float = DEFAULT_RESID_CUTOFF,
) -> tuple[float, float]:
    """Pre-assimilation C*-invariant Student-t/Tukey redescending outlier guard."""
    m_row = np.asarray(mean_ray_row, dtype=COMPLEX_DTYPE)
    v_row = np.maximum(np.asarray(var_ray_row, dtype=REAL_DTYPE), 0.0)
    x = np.asarray(x_hat, dtype=COMPLEX_DTYPE).ravel()
    y = np.asarray(y_b_row, dtype=COMPLEX_DTYPE).ravel()
    n0_val = float(n0)
    if not math.isfinite(n0_val) or n0_val <= 0.0:
        raise ValueError(f"n0 must be finite and strictly positive, got {n0}")

    delayed_x = x[DELAYED_INDEX]
    y_hat = np.sum(m_row * delayed_x, axis=0)
    pred_var = n0_val + np.sum(v_row * (delayed_x.real ** 2 + delayed_x.imag ** 2), axis=0)
    diff = y - y_hat
    resid2 = diff.real ** 2 + diff.imag ** 2
    u_agg = float(np.sum(resid2) / np.sum(pred_var))
    u_mean = float(np.mean(resid2 / pred_var))
    u_stat = max(u_agg, u_mean)
    w_resid = redescending_residual_weight(u_stat, nu=nu, cutoff=cutoff)
    return float(u_stat), float(w_resid)


def _readout_rows_llrs(
    mean_rows: np.ndarray,  # [3, R, N] (in consistent units with y_rows and n0)
    var_rows: np.ndarray,   # [3, R, N]
    y_rows: np.ndarray,     # [R, N]
    n0: float,
) -> np.ndarray:
    r_cnt = len(y_rows)
    if r_cnt == 0:
        return np.zeros((0, N, 4), dtype=REAL_DTYPE)
    n0_val = float(n0)
    ref_var = max(float(np.mean(mean_rows.real ** 2 + mean_rows.imag ** 2)), n0_val)
    amp = math.sqrt(ref_var)
    mean_d = mean_rows / amp
    var_d = var_rows / ref_var
    y_d = y_rows / amp
    n0_c = n0_val / ref_var

    noise = n0_c + var_d.sum(axis=0)
    design = np.zeros((r_cnt, N, N), dtype=COMPLEX_DTYPE)
    for ray, delayed in enumerate(DELAYED_INDEX):
        design[:, SAMPLES, delayed] = mean_d[ray]
    adjoint = design.conj().swapaxes(-1, -2) / noise[:, None, :]
    eye_n = np.eye(N, dtype=COMPLEX_DTYPE)[None, :, :]
    cov = np.linalg.inv(eye_n + adjoint @ design)
    estimate = (cov @ (adjoint @ y_d[:, :, None]))[..., 0]
    uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
    return qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(r_cnt, N, 4)


# ---------------------------------------------------------------------------
# 3. Persistent Fold Branch Adapters (ContinuousBranch, ShiftBranch, ResetBranch)
# ---------------------------------------------------------------------------


class _FoldBranchBase:
    """Common persistent branch contract inside one isolated fold."""
    kind: str  # "continuous", "shift", or "reset"
    history_weight_val: float

    def refit(self, ledger: Mapping[int, tuple[np.ndarray, np.ndarray, float]]) -> Any:
        raise NotImplementedError

    def predict(self, state: Any, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean [3, R, N], var [3, R, N])` in physical units."""
        raise NotImplementedError

    def predict_canon(self, state: Any, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean_c [3, R, N], var_c [3, R, N])` directly in canonical `c_B` units."""
        raise NotImplementedError

    def predict_probe(self, state: Any, probe_row: int = PROBE_ROW) -> tuple[np.ndarray, np.ndarray]:
        """Return `(probe_pred [N], probe_var [N])` in physical units."""
        raise NotImplementedError

    def split_ray0_modes(self, state: Any) -> tuple[float, float, Any, Any]:
        """Split ALL 3 rays by Ray-0 signed frequency (`+` vs `-`)."""
        raise NotImplementedError


class InheritedOrShiftedBranch(_FoldBranchBase):
    """Persistent wrapper over `ShiftedBranchDynamics` in canonical `c_B` units."""

    def __init__(
        self,
        *,
        kind: str,
        dynamics_c: ShiftedBranchDynamics,
        gauge: complex,
        n0: float,
        y10_c: np.ndarray,
        sweeps: int = SWEEPS,
    ) -> None:
        self.kind = kind
        self.history_weight_val = 1.0
        self.dynamics_c = dynamics_c
        self.gauge = complex(gauge)
        self.g2 = float(abs(self.gauge) ** 2)
        self.n0 = float(n0)
        self.y10_c = np.asarray(y10_c, dtype=COMPLEX_DTYPE).ravel()
        self.sweeps = int(sweeps)

    def refit(self, ledger: Mapping[int, tuple[np.ndarray, np.ndarray, float]]) -> State:
        items = sorted(((_strict_row_int(r, "ledger row"), val[0], val[1], float(val[2])) for r, val in ledger.items()), key=lambda t: t[0])
        rows = (ANCHOR,) + tuple(t[0] for t in items)
        if not items:
            tx = SC_PILOTS[ANCHOR : ANCHOR + 1]
            y_c = self.y10_c[None, :]
            noises_c = np.array([1.0], dtype=REAL_DTYPE)
        else:
            tx = np.vstack([SC_PILOTS[ANCHOR : ANCHOR + 1], np.stack([t[1] for t in items], axis=0)])
            y_phys = np.stack([t[2] for t in items], axis=0)
            y_c = np.vstack([self.y10_c[None, :], y_phys / self.gauge])
            noises_c = np.array([1.0] + [t[3] / self.n0 for t in items], dtype=REAL_DTYPE)
        return self.dynamics_c.refit(rows, tx, y_c, noises_c, sweeps=self.sweeps, probe_row=PROBE_ROW)

    def predict_canon(self, state: State, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.dynamics_c.predict(state, rows=rows)

    def predict(self, state: State, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        m_c, v_c = self.predict_canon(state, rows=rows)
        return np.asarray(m_c * self.gauge, dtype=COMPLEX_DTYPE), np.asarray(v_c * self.g2, dtype=REAL_DTYPE)

    def predict_probe(self, state: State, probe_row: int = PROBE_ROW) -> tuple[np.ndarray, np.ndarray]:
        p_c, pv_c = self.dynamics_c.predict_probe(state, n0=1.0, probe_row=probe_row)
        return np.asarray(p_c * self.gauge, dtype=COMPLEX_DTYPE), np.asarray(pv_c * self.g2, dtype=REAL_DTYPE)

    def split_ray0_modes(self, state: State) -> tuple[float, float, State, State]:
        # In `p8.tracker`, indices 0..63 have positive start freq, 64..127 negative.
        lw0 = np.asarray(state.log_weight[0], dtype=REAL_DTYPE)
        half = lw0.shape[0] // 2
        w0 = np.where(np.isfinite(lw0), np.exp(lw0), 0.0)
        p_pos = float(np.sum(w0[:half]))
        p_neg = float(np.sum(w0[half:]))

        lw_pos = np.asarray(state.log_weight, dtype=REAL_DTYPE).copy()
        lw_pos[0, half:] = -np.inf
        if p_pos > 0.0:
            lw_pos[0, :half] -= float(logsumexp(lw_pos[0, :half]))

        lw_neg = np.asarray(state.log_weight, dtype=REAL_DTYPE).copy()
        lw_neg[0, :half] = -np.inf
        if p_neg > 0.0:
            lw_neg[0, half:] -= float(logsumexp(lw_neg[0, half:]))

        st_pos = State(float(state.epoch), lw_pos, state.mean, state.variance)
        st_neg = State(float(state.epoch), lw_neg, state.mean, state.variance)
        return p_pos, p_neg, st_pos, st_neg


class ResetBranch(_FoldBranchBase):
    """Persistent wrapper over `ResetBase` (`GL3` joint posterior) across all wavefront sweeps."""

    def __init__(self, base: ResetBase, y10_phys: np.ndarray) -> None:
        self.kind = "reset"
        self.history_weight_val = 0.0
        self.base = base
        self.y10_phys = np.asarray(y10_phys, dtype=COMPLEX_DTYPE).ravel()

    def refit(self, ledger: Mapping[int, tuple[np.ndarray, np.ndarray, float]]) -> JointResetState:
        return refit_reset(self.base, self.y10_phys, ledger)

    def predict_canon(self, state: JointResetState, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        return project_joint_moments_canon(state, rows=rows)

    def predict(self, state: JointResetState, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        return project_joint_moments(state, rows=rows)

    def predict_probe(self, state: JointResetState, probe_row: int = PROBE_ROW) -> tuple[np.ndarray, np.ndarray]:
        return project_joint_probe(state, probe_row=probe_row)

    def split_ray0_modes(self, state: JointResetState) -> tuple[float, float, JointResetState, JointResetState]:
        return split_joint_reset_ray0_sign(state)


# ---------------------------------------------------------------------------
# 4. Fold-Isolated Wavefront Assimilation & Public Entrypoints
# ---------------------------------------------------------------------------


def _sanitize_y_b(y_b: np.ndarray, probe_row: int) -> np.ndarray:
    arr = np.asarray(y_b, dtype=COMPLEX_DTYPE)
    if arr.shape != (T, N):
        raise ValueError(f"Expected y_b shape {(T, N)}, got {arr.shape}")
    out = np.full((T, N), complex(np.nan, np.nan), dtype=COMPLEX_DTYPE)
    pub = [r for r in range(T) if r != int(probe_row)]
    pub_vals = arr[pub]
    if not np.isfinite(pub_vals).all():
        raise ValueError("Public non-probe rows of obs.y_b contain non-finite values")
    out[pub] = pub_vals
    return out


def _decode_single_mode_row(
    y_row_c: np.ndarray,
    mean_row_c: np.ndarray,
    var_row_c: np.ndarray,
    *,
    bp_iters: int,
    pic_scale: float,
) -> tuple[BlindMarginDecodeResult, np.ndarray]:
    h_r = build_sc_channel_matrix(mean_row_c)
    nv_r = 1.0 + var_row_c.sum(axis=0)
    init_llrs = _readout_rows_llrs(mean_row_c[:, None, :], var_row_c[:, None, :], y_row_c[None, :], 1.0)[0].reshape(128)
    # Fast BP-0 probe for early accept on ultra-clean rows (always followed by strong_large in final owner readout!)
    post0, conv0 = bp_posterior(init_llrs, iterations=bp_iters)
    if bool(conv0):
        c_bp = (post0 > 0.0).astype(np.uint8)
        seed_u64 = pack_codewords(c_bp)
        y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(y_row_c, h_r, nv_r)
        b0, b1, db, _s0, _s1, ds, n_sc = _scan_best_and_runner_up_u64(
            seed_u64[0], seed_u64[1], _EMPTY_U64, _GEN_U64, _H_U64,
            y_re, y_im, row_ptr, col_idx, val_re, val_im, _NIBBLE_RE, _NIBBLE_IM,
        )
        cw_bp = unpack_codewords(np.array([b0, b1], dtype=np.uint64))
        m_bp = float(max(0.0, float(ds) - float(db)))
        fit_bp = float(db) / float(N)
        if is_valid_codeword(cw_bp) and m_bp >= 18.0 and fit_bp <= 1.35:
            rel_bp = reliability_weight_from_margin(m_bp, True, fit_ratio=fit_bp)
            return (
                BlindMarginDecodeResult(
                    info_bits=cw_bp[:64].copy(),
                    codeword=cw_bp,
                    d_best=float(db),
                    d_second=float(ds),
                    margin=m_bp,
                    fit_ratio=fit_bp,
                    bp_converged=True,
                    reliability=rel_bp,
                    candidates_u64=np.array([[b0, b1]], dtype=np.uint64),
                    n_candidates=int(n_sc),
                    elapsed_ms=0.0,
                ),
                init_llrs,
            )

    res = decode_with_blind_margin(
        y_row_c,
        h_r,
        nv_r,
        list_size="large",
        bp_iters=bp_iters,
        turbo_pic=True,
        pic_scale=pic_scale,
        initial_llrs=init_llrs,
    )
    return res, init_llrs


def _decode_wave1_multimode_union(
    y_row_c: np.ndarray,
    branch: _FoldBranchBase,
    state: Any,
    row: int,
    *,
    bp_iters: int,
    pic_scale: float,
    sign_split_thresh: float,
) -> tuple[BlindMarginDecodeResult, np.ndarray, np.ndarray]:
    """Wave-1 multi-mode union candidate scoring across signed Ray-0 modes."""
    p_pos, p_neg, st_pos, st_neg = branch.split_ray0_modes(state)
    c_b = complex(branch.base.gauge) if isinstance(branch, ResetBranch) else branch.gauge
    g2 = float(abs(c_b) ** 2)

    if min(p_pos, p_neg) <= float(sign_split_thresh):
        m_full, v_full = branch.predict(state, rows=(row,))
        m_c = m_full[:, 0, :] / c_b
        v_c = v_full[:, 0, :] / g2
        res, _ = _decode_single_mode_row(y_row_c, m_c, v_c, bp_iters=bp_iters, pic_scale=pic_scale)
        return res, m_c, v_c

    m_pos_p, v_pos_p = branch.predict(st_pos, rows=(row,))
    m_neg_p, v_neg_p = branch.predict(st_neg, rows=(row,))
    m_pos_c = m_pos_p[:, 0, :] / c_b
    v_pos_c = v_pos_p[:, 0, :] / g2
    m_neg_c = m_neg_p[:, 0, :] / c_b
    v_neg_c = v_neg_p[:, 0, :] / g2

    llrs_pos = _readout_rows_llrs(m_pos_c[:, None, :], v_pos_c[:, None, :], y_row_c[None, :], 1.0)[0].reshape(128)
    llrs_neg = _readout_rows_llrs(m_neg_c[:, None, :], v_neg_c[:, None, :], y_row_c[None, :], 1.0)[0].reshape(128)

    h_pos = build_sc_channel_matrix(m_pos_c)
    h_neg = build_sc_channel_matrix(m_neg_c)
    nv_pos = 1.0 + v_pos_c.sum(axis=0)
    nv_neg = 1.0 + v_neg_c.sum(axis=0)

    res_p = decode_strong_sc(
        y_row_c, h_pos, nv_pos, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_pos
    )
    res_n = decode_strong_sc(
        y_row_c, h_neg, nv_neg, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_neg
    )

    # Deduplicated union of parity-valid candidates + 64 1-flips around each mode winner
    seed_p = pack_codewords(res_p.codeword)
    seed_n = pack_codewords(res_n.codeword)
    flips_p = seed_p[None, :] ^ _GEN_U64
    flips_n = seed_n[None, :] ^ _GEN_U64
    all_u64 = np.vstack([seed_p[None, :], seed_n[None, :], flips_p, flips_n, res_p.candidates_u64, res_n.candidates_u64])

    # Fast unique over (c0, c1) pairs using structured view
    dt_pair = np.dtype([("c0", np.uint64), ("c1", np.uint64)])
    uniq_view = np.unique(np.ascontiguousarray(all_u64).view(dt_pair).ravel())
    uniq_u64 = np.ascontiguousarray(uniq_view.view(np.uint64).reshape(-1, 2))

    log_p = np.array([math.log(max(p_pos, 1e-300)), math.log(max(p_neg, 1e-300))], dtype=REAL_DTYPE)
    h_stack = np.stack([h_pos, h_neg], axis=0)
    v_stack = np.stack([v_pos_c, v_neg_c], axis=0)

    b0, b1, best_ll, _s0, _s1, second_ll, win_mode, win_fit, n_valid = _score_multimode_union_u64(
        uniq_u64,
        _H_U64,
        np.ascontiguousarray(y_row_c.real, dtype=REAL_DTYPE),
        np.ascontiguousarray(y_row_c.imag, dtype=REAL_DTYPE),
        log_p,
        np.ascontiguousarray(h_stack.real, dtype=REAL_DTYPE),
        np.ascontiguousarray(h_stack.imag, dtype=REAL_DTYPE),
        np.ascontiguousarray(v_stack, dtype=REAL_DTYPE),
        1.0,
        DELAYS,
        _NIBBLE_RE,
        _NIBBLE_IM,
    )

    cw_best = unpack_codewords(np.array([b0, b1], dtype=np.uint64))
    if not is_valid_codeword(cw_best):
        raise RuntimeError("Internal invariant failure in Wave-1 union scorer")

    margin = float(max(0.0, float(best_ll) - float(second_ll))) if n_valid >= 2 else 0.0
    bp_conv = bool(
        (res_p.bp_converged and np.array_equal(res_p.codeword, cw_best))
        or (res_n.bp_converged and np.array_equal(res_n.codeword, cw_best))
    )
    rel = reliability_weight_from_margin(margin, bp_conv, fit_ratio=float(win_fit))

    m_win_c = m_pos_c if win_mode == 0 else m_neg_c
    v_win_c = v_pos_c if win_mode == 0 else v_neg_c

    return (
        BlindMarginDecodeResult(
            info_bits=cw_best[:64].copy(),
            codeword=cw_best,
            d_best=float(-best_ll),
            d_second=float(-second_ll),
            margin=margin,
            fit_ratio=float(win_fit),
            bp_converged=bp_conv,
            reliability=rel,
            candidates_u64=uniq_u64,
            n_candidates=int(n_valid),
            elapsed_ms=float(res_p.elapsed_ms + res_n.elapsed_ms),
        ),
        m_win_c,
        v_win_c,
    )


def _run_one_fold(
    *,
    fold_idx: int,
    a_prior: State,
    y10_phys: np.ndarray,
    y_b_safe: np.ndarray,
    aux_rows: tuple[int, ...],
    target_rows: tuple[int, ...],
    extra_owner_rows: tuple[int, ...],
    all_data_rows: tuple[int, ...],
    n0: float,
    precomputed_anchor: tuple[complex, np.ndarray, float, Any],
    tau_switch: float,
    aux_temp: float,
    scale_quad_order: int,
    bp_iters: int,
    pic_scale: float,
    min_assimilate_weight: float,
    virtual_var_floor: float,
    sign_split_thresh: float,
) -> dict[str, Any]:
    """Execute one completely isolated fold (`f=0` or `f=1`). Never reads `target_rows` during inference/waves."""
    y_aux_phys = y_b_safe[list(aux_rows)].copy()
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

    if t_f > float(tau_switch):
        if evidence.b_s >= evidence.b_r:
            dyn_s = ShiftedBranchDynamics(
                a_prior_c,
                changed_ray=evidence.best_shift.ray,
                delta_freq=evidence.refined_delta,
                onset_row=evidence.best_shift.onset_row,
            )
            branch: _FoldBranchBase = InheritedOrShiftedBranch(
                kind="shift",
                dynamics_c=dyn_s,
                gauge=c_b,
                n0=n0,
                y10_c=y10_c,
            )
        else:
            branch = ResetBranch(reset_base, y10_phys)
    else:
        dyn_c = ShiftedBranchDynamics(a_prior_c, changed_ray=None, delta_freq=0.0, onset_row=0)
        branch = InheritedOrShiftedBranch(
            kind="continuous",
            dynamics_c=dyn_c,
            gauge=c_b,
            n0=n0,
            y10_c=y10_c,
        )

    ledger: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    fold_ledger_records: list[dict[str, Any]] = []
    row_weights_f: dict[int, float] = {}
    row_margins_f: dict[int, float] = {}
    state = branch.refit(ledger)
    state_version = 0

    # Strict fold helper exclusion: exclude targets, aux, anchor, and probe
    forbidden = set(target_rows) | set(aux_rows) | {ANCHOR, PROBE_ROW}
    helpers = tuple(r for r in all_data_rows if r not in forbidden)
    helper_set = set(helpers)

    for wave_idx, wave in enumerate(FINE_SCHEDULE):
        wave_rows = tuple(r for r in wave if r in helper_set)
        if not wave_rows:
            continue

        # Evaluate ALL rows in this ring against the exact same pre-wave snapshot `state`
        m_wave_phys, v_wave_phys = branch.predict(state, rows=wave_rows)
        pending: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        pending_records: list[dict[str, Any]] = []

        for local_idx, r in enumerate(wave_rows):
            y_row_c = y_b_safe[r] / c_b
            if wave_idx == 0:
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

            row_margins_f[r] = float(res.margin)
            if res.reliability >= float(min_assimilate_weight):
                x_hat = qam(res.codeword).astype(COMPLEX_DTYPE)
                _, w_resid = predictive_row_residual_weight(eval_m_c, eval_v_c, x_hat, y_row_c, 1.0)
                alpha_eff = float(min(DEFAULT_ALPHA_MAX, res.reliability * w_resid))
                if alpha_eff >= float(min_assimilate_weight):
                    n_eff_phys = n0 * (1.0 / alpha_eff + float(virtual_var_floor))
                    pending[r] = (x_hat, y_b_safe[r].copy(), n_eff_phys)
                    row_weights_f[r] = alpha_eff
                    pending_records.append({
                        "fold": int(fold_idx),
                        "wave": int(wave_idx),
                        "row": int(r),
                        "injected_info": res.info_bits.copy(),
                        "injected_codeword": res.codeword.copy(),
                        "alpha": alpha_eff,
                        "margin": float(res.margin),
                        "state_version": int(state_version),
                        "branch": branch.kind,
                    })

        if pending:
            ledger.update(pending)
            fold_ledger_records.extend(pending_records)
            state = branch.refit(ledger)
            state_version += 1

    # Freeze fold state & read out full Frame B predictions + owner rows via strong_large
    mean_full_c, var_full_c = branch.predict_canon(state, rows=None)  # [3, T, N]
    mean_full_phys = np.asarray(mean_full_c * c_b, dtype=COMPLEX_DTYPE)
    var_full_phys = np.asarray(var_full_c * g2, dtype=REAL_DTYPE)
    probe_pred_phys, probe_var_phys = branch.predict_probe(state, probe_row=PROBE_ROW)

    owner_rows = tuple(target_rows) + tuple(extra_owner_rows)
    y_owner_c = y_b_safe[list(owner_rows)] / c_b
    llrs_owner = _readout_rows_llrs(
        mean_full_c[:, list(owner_rows), :],
        var_full_c[:, list(owner_rows), :],
        y_owner_c,
        1.0,
    )

    decoded_owner: dict[int, np.ndarray] = {}
    for idx_o, r in enumerate(owner_rows):
        m_row_c = mean_full_c[:, r, :]
        v_row_c = var_full_c[:, r, :]
        y_row_c = y_owner_c[idx_o]
        h_c_r = build_sc_channel_matrix(m_row_c)
        nv_c_r = 1.0 + v_row_c.sum(axis=0)
        init_llrs_r = llrs_owner[idx_o].reshape(128)
        dec_r = decode_strong_sc(
            y_row_c,
            h_c_r,
            nv_c_r,
            list_size="large",
            bp_iters=bp_iters,
            turbo_pic=True,
            pic_scale=pic_scale,
            initial_llrs=init_llrs_r,
        )
        decoded_owner[r] = dec_r.info_bits.copy()
        if r not in row_margins_f:
            seed_u64 = pack_codewords(dec_r.codeword)
            y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(y_row_c, h_c_r, nv_c_r)
            _, _, d_best, _, _, d_sec, _ = _scan_best_and_runner_up_u64(
                seed_u64[0], seed_u64[1], dec_r.candidates_u64, _GEN_U64, _H_U64,
                y_re, y_im, row_ptr, col_idx, val_re, val_im, _NIBBLE_RE, _NIBBLE_IM,
            )
            row_margins_f[r] = float(max(0.0, float(d_sec) - float(d_best)))

    return {
        "fold": int(fold_idx),
        "branch": branch.kind,
        "history_weight": float(branch.history_weight_val),
        "t_f": float(t_f),
        "b_c": float(evidence.b_c),
        "b_s": float(evidence.b_s),
        "b_r": float(evidence.b_r),
        "best_shift": {
            "ray": evidence.best_shift.ray,
            "delta": float(evidence.refined_delta),
            "onset_row": int(evidence.best_shift.onset_row),
        },
        "state_version": int(state_version),
        "mean_full": mean_full_phys,
        "var_full": var_full_phys,
        "probe_pred": probe_pred_phys,
        "probe_var": probe_var_phys,
        "owner_rows": owner_rows,
        "llrs_owner": llrs_owner,
        "decoded_owner": decoded_owner,
        "row_weights": row_weights_f,
        "row_margins": row_margins_f,
        "ledger": fold_ledger_records,
    }


def run_p11_with_trace(
    obs: PublicObservation,
    *,
    tau_switch: float = FROZEN_TAU_SWITCH,
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    min_assimilate_weight: float = DEFAULT_MIN_ASSIMILATE_WEIGHT,
    virtual_var_floor: float = DEFAULT_VIRTUAL_VAR_FLOOR,
    sign_split_thresh: float = DEFAULT_SIGN_SPLIT_THRESH,
) -> tuple[AssimilationResult, dict[str, Any]]:
    """Run strictly fold-isolated P11 primary pipeline returning `(AssimilationResult, trace)`."""
    t0 = time.perf_counter()

    probe_row = _strict_row_int(obs.probe_row, "obs.probe_row")
    if probe_row != PROBE_ROW:
        raise ValueError(f"P11 primary requires probe_row={PROBE_ROW}, got {probe_row}")
    training_rows = tuple(_strict_row_int(r, "obs.training_rows") for r in obs.training_rows)
    if training_rows != (ANCHOR,):
        raise ValueError(f"P11 primary requires sparse training_rows={(ANCHOR,)}, got {training_rows}")
    from p11.payload_rebuild import _validate_positive_n0

    n0 = _validate_positive_n0(obs.n0, name="obs.n0")

    y_b_safe = _sanitize_y_b(obs.y_b, probe_row)
    y10_phys = y_b_safe[ANCHOR].copy()

    # Frame-A prior (strictly from y_a, never touches y_b)
    a_prior = build_frame_a_prior_clean(obs.y_a, n0)

    # Link-local anchor fit depends ONLY on (y10, n0) — identical for both folds
    precomputed_anchor = fit_anchor_only(y10_phys, n0, scale_quad_order=scale_quad_order)

    free_extra = tuple(_strict_row_int(r, "extra_rows") for r in extra_rows(training_rows))  # (0, 9) in sparse
    all_data_rows = tuple(r for r in range(T) if r not in (*training_rows, probe_row))

    fold0 = _run_one_fold(
        fold_idx=0,
        a_prior=a_prior,
        y10_phys=y10_phys,
        y_b_safe=y_b_safe,
        aux_rows=AUX_FOLDS[0],
        target_rows=FOLD_TARGETS[0],
        extra_owner_rows=free_extra,
        all_data_rows=all_data_rows,
        n0=n0,
        precomputed_anchor=precomputed_anchor,
        tau_switch=tau_switch,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
        bp_iters=bp_iters,
        pic_scale=pic_scale,
        min_assimilate_weight=min_assimilate_weight,
        virtual_var_floor=virtual_var_floor,
        sign_split_thresh=sign_split_thresh,
    )
    fold1 = _run_one_fold(
        fold_idx=1,
        a_prior=a_prior,
        y10_phys=y10_phys,
        y_b_safe=y_b_safe,
        aux_rows=AUX_FOLDS[1],
        target_rows=FOLD_TARGETS[1],
        extra_owner_rows=(),
        all_data_rows=all_data_rows,
        n0=n0,
        precomputed_anchor=precomputed_anchor,
        tau_switch=tau_switch,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
        bp_iters=bp_iters,
        pic_scale=pic_scale,
        min_assimilate_weight=min_assimilate_weight,
        virtual_var_floor=virtual_var_floor,
        sign_split_thresh=sign_split_thresh,
    )

    # Assemble unified AssimilationResult from owning folds + equal-weight 2-fold probe mixture
    mix_mean_ray = np.zeros((len(DELAYS), T, N), dtype=COMPLEX_DTYPE)
    mix_var_ray = np.zeros((len(DELAYS), T, N), dtype=REAL_DTYPE)
    mix_llrs = np.zeros((T, N, 4), dtype=REAL_DTYPE)
    history_weight = np.zeros(T, dtype=REAL_DTYPE)
    decoded_info: dict[int, np.ndarray] = {}
    row_margins: dict[int, float] = {}

    for f_res in (fold0, fold1):
        hw = float(f_res["history_weight"])
        for idx_o, r in enumerate(f_res["owner_rows"]):
            mix_mean_ray[:, r, :] = f_res["mean_full"][:, r, :]
            mix_var_ray[:, r, :] = f_res["var_full"][:, r, :]
            mix_llrs[r] = f_res["llrs_owner"][idx_o]
            history_weight[r] = hw
            decoded_info[r] = f_res["decoded_owner"][r]
            row_margins[r] = float(f_res["row_margins"][r])

    # Anchor & probe ray moments: equal-weight 2-fold mixture with between-fold term
    for r_shared in (ANCHOR, PROBE_ROW):
        m0 = fold0["mean_full"][:, r_shared, :]
        m1 = fold1["mean_full"][:, r_shared, :]
        v0 = fold0["var_full"][:, r_shared, :]
        v1 = fold1["var_full"][:, r_shared, :]
        mix_mean_ray[:, r_shared, :] = 0.5 * (m0 + m1)
        mix_var_ray[:, r_shared, :] = 0.5 * (v0 + v1) + 0.25 * (np.abs(m0 - m1) ** 2)
        history_weight[r_shared] = 0.5 * (fold0["history_weight"] + fold1["history_weight"])

    # Pilot & probe LLRs strictly finite 0.0
    mix_llrs[ANCHOR] = 0.0
    mix_llrs[PROBE_ROW] = 0.0

    # Equal-weight 2-fold probe mixture: p = 0.5*(p0+p1), v = 0.5*(v0+v1) + 0.25*|p0-p1|^2
    p0 = fold0["probe_pred"]
    p1 = fold1["probe_pred"]
    pv0 = fold0["probe_var"]
    pv1 = fold1["probe_var"]
    mix_probe_pred = 0.5 * (p0 + p1)
    mix_probe_var = 0.5 * (pv0 + pv1) + 0.25 * (np.abs(p0 - p1) ** 2)
    probe_weight = 0.5 * (fold0["history_weight"] + fold1["history_weight"])

    # Flat assimilated_rows union & representative weights (lower fold index takes precedence)
    ledger_all = fold0["ledger"] + fold1["ledger"]
    rep_weights: dict[int, float] = {}
    assimilated_rows_list: list[int] = []
    seen_rows: set[int] = set()
    for rec in ledger_all:
        r = _strict_row_int(rec["row"], "ledger.row")
        if r not in seen_rows:
            seen_rows.add(r)
            assimilated_rows_list.append(r)
            rep_weights[r] = float(rec["alpha"])

    row_weights: dict[int, float] = {r: float(rep_weights.get(r, 0.0)) for r in all_data_rows}

    gate_out = GateOutput(
        history_weight=history_weight,
        probe_weight=float(probe_weight),
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
        assimilated_rows=tuple(assimilated_rows_list),
        row_weights=row_weights,
        row_margins=row_margins,
        n_rounds=len(FINE_SCHEDULE),
        elapsed_ms=float(elapsed_ms),
    )

    trace = {
        "ledger": ledger_all,
        "folds": (
            {k: v for k, v in fold0.items() if k not in ("mean_full", "var_full", "llrs_owner", "decoded_owner")},
            {k: v for k, v in fold1.items() if k not in ("mean_full", "var_full", "llrs_owner", "decoded_owner")},
        ),
        "tau_switch": float(tau_switch),
        "aux_temp": float(aux_temp),
    }
    return result, trace


def run_p11_primary(
    obs: PublicObservation,
    *,
    tau_switch: float = FROZEN_TAU_SWITCH,
    aux_temp: float = DEFAULT_AUX_TEMP,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    min_assimilate_weight: float = DEFAULT_MIN_ASSIMILATE_WEIGHT,
    virtual_var_floor: float = DEFAULT_VIRTUAL_VAR_FLOOR,
    sign_split_thresh: float = DEFAULT_SIGN_SPLIT_THRESH,
) -> AssimilationResult:
    """Public P11 primary-axis receiver returning `p11.contract.AssimilationResult`."""
    res, _ = run_p11_with_trace(
        obs,
        tau_switch=tau_switch,
        aux_temp=aux_temp,
        scale_quad_order=scale_quad_order,
        bp_iters=bp_iters,
        pic_scale=pic_scale,
        min_assimilate_weight=min_assimilate_weight,
        virtual_var_floor=virtual_var_floor,
        sign_split_thresh=sign_split_thresh,
    )
    return res


def as_p10_compatible_dict(result: AssimilationResult) -> dict[str, Any]:
    """Field-for-field compatibility adapter dictionary matching `p10.contract.AssimilationResult`."""
    return {
        "gate_out": result.gate_out,
        "decoded_info": result.decoded_info,
        "assimilated_rows": result.assimilated_rows,
        "row_weights": result.row_weights,
        "row_margins": result.row_margins,
        "n_rounds": result.n_rounds,
        "elapsed_ms": result.elapsed_ms,
    }


to_p10_compatible_dict = as_p10_compatible_dict


def to_p10_assimilation_result(result: AssimilationResult, target_cls: type) -> Any:
    """Instantiate an external `AssimilationResult` class (e.g. `p10.contract.AssimilationResult`) cleanly."""
    return target_cls(**as_p10_compatible_dict(result))


__all__ = [
    "AUX_FOLDS",
    "BlindMarginDecodeResult",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_AUX_TEMP",
    "DEFAULT_BP_UNCONVERGED_PENALTY",
    "DEFAULT_FIT_CUTOFF",
    "DEFAULT_FIT_NU",
    "DEFAULT_MARGIN_MIN",
    "DEFAULT_MARGIN_SCALE",
    "DEFAULT_MIN_ASSIMILATE_WEIGHT",
    "DEFAULT_RESID_CUTOFF",
    "DEFAULT_RESID_NU",
    "DEFAULT_SIGN_SPLIT_THRESH",
    "DEFAULT_VIRTUAL_VAR_FLOOR",
    "FINE_SCHEDULE",
    "FOLD_TARGETS",
    "FROZEN_TAU_SWITCH",
    "InheritedOrShiftedBranch",
    "ResetBranch",
    "as_p10_compatible_dict",
    "decode_with_blind_margin",
    "predictive_row_residual_weight",
    "redescending_residual_weight",
    "reliability_weight_from_margin",
    "run_p11_primary",
    "run_p11_with_trace",
    "to_p10_assimilation_result",
    "to_p10_compatible_dict",
    "warmup_p11_numba",
]
