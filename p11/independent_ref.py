"""Independent Phase 11 off-grid physics reference, exact 2^K subset-Bayes oracle, and anti-leakage firewalls.

Completely independent from M2 (p11.payload_rebuild, p11.wavefront_pipeline),
M3 Pillar 2 (p11.offgrid_channel, p11.manifold_gate), PyTorch, and all P8/P9/P10
receiver/channel modules so it serves as an uncoupled blind verification oracle.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, is_dataclass, fields as dc_fields
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from p9.contract import (
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
)
from p9.independent_ref import (
    BLOCK_LEN,
    FORBIDDEN_HIDDEN_ATTRS as P9_FORBIDDEN_HIDDEN_ATTRS,
    TOTAL_BLOCKS,
    TOTAL_SAMPLES,
    PiecewiseDopplerShift,
    check_no_hardcoded_n0,
    row_boundary_time,
    serial_propagate_ab,
    verify_public_observation,
)
from p11.contract import (
    CP,
    FRAME_DURATION,
    HISTORICAL_SEED_RANGES,
    N,
    OFDM_PILOT_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    P11_NAMESPACE_SPAN,
    P11_SEED_RANGES,
    PROBE_ROW,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_FORMAL_ID,
    SEED_P11_FORMAL_OOD,
    SEED_P11_TRAIN,
    T,
    AssimilationResult,
    GateOutput,
    OffGridObservation,
    PublicObservation,
    fractional_delay_freq_response,
    fractional_delay_matrix,
    is_historical_seed,
)


# Historical base seeds (strictly disjoint from P11)
P8_MANIFOLD_BASE_SEED = 802611190000
P8_PREFIX_BASE_SEED = 812611190000
P8C_BASE_SEED = 822611190000
P4_STAGE3_LITERAL_SEED = 9_102_379_953_802

P11_FORBIDDEN_PEER_MODULES = frozenset({
    "p11.payload_rebuild",
    "payload_rebuild",
    "p11.wavefront_pipeline",
    "wavefront_pipeline",
    "p11.offgrid_channel",
    "offgrid_channel",
    "p11.manifold_gate",
    "manifold_gate",
    "torch",
    "p10",
    "p10.per_ray_gate",
    "p10.turbo_assimilate",
    "p10.offgrid_channel",
    "p10.manifold_gate",
    "p10.cov_attention",
    "p10.cov_repair",
    "p10.payload_reconstruction",
    "p10.covariance_repair",
    "p10.reset_gating",
    "p9.channel",
    "p9.gating",
    "p9.decoder",
    "p9.neural_gate",
    "p8.link",
    "p8.coding",
    "p8.tracker",
    "p8c.link",
    "p8",
    "p8c",
})

ALLOWED_OFFGRID_ATTRS = frozenset({
    "y_a",
    "y_b",
    "training_rows",
    "probe_row",
    "n0",
    "pilot_tx_b",
    "pilot_tx_a",
})

FORBIDDEN_OFFGRID_HIDDEN_ATTRS = P9_FORBIDDEN_HIDDEN_ATTRS | frozenset({
    "n_rays",
    "delays",
    "powers",
    "freqs_a",
    "freqs_b",
    "chirps",
    "changed_rays",
    "corrupted_rows",
    "data_rows",
    "h_b_true_mat",
})


def _is_valid_int_seed(item: Any) -> bool:
    """Return True iff `item` is a genuine integer (rejecting bool, timedelta64, datetime64)."""
    if isinstance(item, (bool, np.bool_, np.timedelta64, np.datetime64)):
        return False
    return isinstance(item, (int, np.integer))


def _is_valid_real_number(item: Any) -> bool:
    """Return True iff `item` is a genuine finite real int/float (rejecting bool, timedelta64, datetime64)."""
    if isinstance(item, (bool, np.bool_, np.timedelta64, np.datetime64, complex, np.complexfloating)):
        return False
    if not isinstance(item, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(item))


_FORBIDDEN_SCALAR_TYPES = (bool, np.bool_, np.timedelta64, np.datetime64)
_REAL_ARRAY_KINDS = frozenset({"i", "u", "f"})
_COMPLEX_ARRAY_KINDS = frozenset({"i", "u", "f", "c"})


def _seq_has_forbidden_scalar(obj: Any, *, allow_complex: bool) -> bool:
    if isinstance(obj, _FORBIDDEN_SCALAR_TYPES):
        return True
    if not allow_complex and isinstance(obj, (complex, np.complexfloating)):
        return True
    if isinstance(obj, np.ndarray):
        allowed_kinds = _COMPLEX_ARRAY_KINDS if allow_complex else _REAL_ARRAY_KINDS
        if obj.dtype.kind == "O":
            return any(_seq_has_forbidden_scalar(x, allow_complex=allow_complex) for x in obj.flat)
        return obj.dtype.kind not in allowed_kinds
    if isinstance(obj, (list, tuple)):
        return any(_seq_has_forbidden_scalar(x, allow_complex=allow_complex) for x in obj)
    return False


def _validate_numeric_array(
    val: Any,
    *,
    allow_complex: bool,
    field_name: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if isinstance(val, np.ma.MaskedArray):
        return None, f"{field_name} must not be a MaskedArray"
    if isinstance(val, _FORBIDDEN_SCALAR_TYPES) or isinstance(val, (str, bytes)):
        return None, f"{field_name} has forbidden scalar type {type(val).__name__}"
    if not allow_complex and isinstance(val, (complex, np.complexfloating)):
        return None, f"{field_name} must be real-valued, got {type(val).__name__}"
    if isinstance(val, (list, tuple)) and _seq_has_forbidden_scalar(val, allow_complex=allow_complex):
        return None, f"{field_name} sequence contains bool/timedelta64/datetime64 or invalid scalar"
    try:
        raw = np.asarray(val)
    except Exception as exc:
        return None, f"{field_name} could not be converted to array: {exc}"
    allowed_kinds = _COMPLEX_ARRAY_KINDS if allow_complex else _REAL_ARRAY_KINDS
    if raw.dtype.kind == "O":
        allowed_scalar_types = (
            (int, float, np.integer, np.floating, complex, np.complexfloating)
            if allow_complex
            else (int, float, np.integer, np.floating)
        )
        for elem in raw.flat:
            if _seq_has_forbidden_scalar(elem, allow_complex=allow_complex):
                return None, f"{field_name} object array contains bool/timedelta64/datetime64"
            if not isinstance(elem, allowed_scalar_types):
                return None, f"{field_name} object array contains non-numeric element {type(elem).__name__}"
    elif raw.dtype.kind not in allowed_kinds:
        return None, f"{field_name} has forbidden dtype {raw.dtype}"
    target_dtype = np.complex128 if allow_complex else np.float64
    return np.asarray(raw, dtype=target_dtype), None


@dataclass(frozen=True)
class OffGridPropagationResult:
    """Full diagnostics from independent sample-by-sample off-grid cross-waveform propagation."""
    y_a: np.ndarray                             # [T, N] complex128 (OFDM freq-domain if a_is_freq else time)
    y_b: np.ndarray                             # [T, N] complex128 (CP-SC post-CP time-domain)
    tx_serial: np.ndarray                       # [1920] complex128 transmitted serial stream with CPs
    rx_serial: np.ndarray                       # [1920] complex128 received serial stream before CP removal
    t_serial: np.ndarray                        # [1920] float64 absolute sample clock t_k = k / N
    g_serial_a: np.ndarray                      # [P_a, 960] complex128 exact ray gains on Frame A serial samples
    g_serial_b: np.ndarray                      # [P_b, 960] complex128 exact ray gains on Frame B serial samples
    h_b_useful: np.ndarray                      # [P_b, T, N] complex128 exact ray gains on Frame B useful samples
    h_b_true_mat: np.ndarray                    # [T, N, N] complex128 independent per-row CP-SC channel matrices
    phase_continuity: Optional[list[dict]] = None


# ---------------------------------------------------------------------------
# 1. Independent Time-Domain Dirichlet Fractional-Delay Kernel
# ---------------------------------------------------------------------------

def _independent_signed_k(n: int) -> np.ndarray:
    """Construct centered DFT indices [0, ..., n//2 - 1, -n//2, ..., -1] independently."""
    if not _is_valid_int_seed(n):
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    half = n_int // 2
    if n_int % 2 == 0:
        return np.concatenate((
            np.arange(0, half, dtype=np.float64),
            np.arange(-half, 0, dtype=np.float64),
        ))
    return np.concatenate((
        np.arange(0, half + 1, dtype=np.float64),
        np.arange(-half, 0, dtype=np.float64),
    ))


def _independent_unitary_dft(n: int) -> np.ndarray:
    """Construct unitary N x N DFT matrix independently using exact mod-N angle reduction."""
    if not _is_valid_int_seed(n):
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n_int = int(n)
    idx = np.arange(n_int, dtype=np.int64)
    prod_mod = (idx[:, None] * idx[None, :]) % n_int
    return (
        np.exp(-2j * math.pi * prod_mod.astype(np.float64) / float(n_int)) / math.sqrt(float(n_int))
    ).astype(np.complex128)


def dirichlet_fractional_delay_matrix(delay: float, n: int = N) -> np.ndarray:
    """Independent explicit time-domain Dirichlet fractional-delay cyclic shift operator D_N(delay).

    Computes every entry directly from the trigonometric sum:
        D[n, m] = (1 / N) * sum_{k in k_s} exp(2j * pi * k * (n - m - delay) / N)
    without calling p11.contract.fractional_delay_matrix or any FFT routine.
    """
    if not _is_valid_int_seed(n):
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not _is_valid_real_number(delay):
        raise ValueError(f"delay must be finite real, got {delay!r}")
    tau = float(delay)

    k_s = _independent_signed_k(n_int)
    k_s_int = k_s.astype(np.int64)
    idx = np.arange(n_int, dtype=np.int64)
    diff_mod = (idx[:, None] - idx[None, :]) % n_int
    int_term = ((diff_mod[:, :, None] * k_s_int[None, None, :]) % n_int).astype(np.float64)
    angles = (2.0 * math.pi / float(n_int)) * (int_term - tau * k_s[None, None, :])
    d_mat = np.sum(np.exp(1j * angles), axis=-1) / float(n_int)
    return np.asarray(d_mat, dtype=np.complex128)


def dirichlet_kernel_closed_form(delay: float, n: int = N) -> np.ndarray:
    """Analytic closed-form Dirichlet trigonometric kernel for independent third-way verification.

    For even N: D[n, m] = exp(-i*pi*u/N) * sin(pi*u) / (N * sin(pi*u/N)) (limit 1 when u == 0 mod N).
    For odd  N: D[n, m] = sin(pi*u) / (N * sin(pi*u/N))                  (limit 1 when u == 0 mod N).
    """
    if not _is_valid_int_seed(n):
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not _is_valid_real_number(delay):
        raise ValueError(f"delay must be finite real, got {delay!r}")
    tau = float(delay)

    idx = np.arange(n_int, dtype=np.int64)
    half = n_int // 2
    diff_int = ((idx[:, None] - idx[None, :] + half) % n_int) - half
    u = diff_int.astype(np.float64) - tau

    denom = np.sin(math.pi * u / float(n_int))
    singular = np.abs(denom) < 1e-13
    safe_denom = np.where(singular, 1.0, denom)
    ratio = np.sin(math.pi * u) / (float(n_int) * safe_denom)
    if n_int % 2 == 0:
        out = np.where(singular, 1.0 + 0.0j, np.exp(-1j * math.pi * u / float(n_int)) * ratio)
    else:
        out = np.where(singular, 1.0 + 0.0j, ratio.astype(np.complex128))
    return np.asarray(out, dtype=np.complex128)


def verify_fractional_delay_operator(
    delays: Sequence[float] = (0.0, 0.25, 0.5, 1.37, 2.0, 3.14159265, 4.0, 5.83, 6.5),
    *,
    n: int = N,
    tol: float = 1e-14,
) -> dict:
    """Cross-verify independent Dirichlet kernel against p11.contract, unitarity, duality & integer shifts."""
    n_int = int(n)
    eye_n = np.eye(n_int, dtype=np.complex128)
    f_indep = _independent_unitary_dft(n_int)

    max_contract_err = 0.0
    max_closed_form_err = 0.0
    max_unitary_err = 0.0
    max_duality_err = 0.0
    max_int_shift_err = 0.0

    for raw_tau in delays:
        tau = float(raw_tau)
        d_indep = dirichlet_fractional_delay_matrix(tau, n=n_int)
        d_closed = dirichlet_kernel_closed_form(tau, n=n_int)
        d_contract = fractional_delay_matrix(tau, n=n_int)

        max_contract_err = max(max_contract_err, float(np.max(np.abs(d_indep - d_contract))))
        max_closed_form_err = max(max_closed_form_err, float(np.max(np.abs(d_indep - d_closed))))

        u_left = float(np.max(np.abs(d_indep.conj().T @ d_indep - eye_n)))
        u_right = float(np.max(np.abs(d_indep @ d_indep.conj().T - eye_n)))
        max_unitary_err = max(max_unitary_err, u_left, u_right)

        freq_diag = np.diag(fractional_delay_freq_response(tau, n=n_int))
        duality = float(np.max(np.abs(f_indep @ d_indep @ f_indep.conj().T - freq_diag)))
        max_duality_err = max(max_duality_err, duality)

        if abs(tau - round(tau)) < 1e-15:
            d_int = int(round(tau))
            expected_perm = np.zeros((n_int, n_int), dtype=np.complex128)
            for row in range(n_int):
                expected_perm[row, (row - d_int) % n_int] = 1.0 + 0.0j
            max_int_shift_err = max(max_int_shift_err, float(np.max(np.abs(d_indep - expected_perm))))

    d_a = dirichlet_fractional_delay_matrix(1.37, n=n_int)
    d_b = dirichlet_fractional_delay_matrix(2.89, n=n_int)
    d_ab = dirichlet_fractional_delay_matrix(1.37 + 2.89, n=n_int)
    d_neg = dirichlet_fractional_delay_matrix(-1.37, n=n_int)
    homomorphism_err = float(np.max(np.abs(d_a @ d_b - d_ab)))
    adjoint_err = float(np.max(np.abs(d_neg - d_a.conj().T)))

    passed = (
        max_contract_err < tol
        and max_closed_form_err < tol
        and max_unitary_err < tol
        and max_duality_err < tol
        and max_int_shift_err < tol
        and homomorphism_err < tol
        and adjoint_err < tol
    )
    return {
        "n": n_int,
        "delays_tested": [float(d) for d in delays],
        "max_contract_err": max_contract_err,
        "max_closed_form_err": max_closed_form_err,
        "max_unitary_err": max_unitary_err,
        "max_duality_err": max_duality_err,
        "max_int_shift_err": max_int_shift_err,
        "homomorphism_err": homomorphism_err,
        "adjoint_err": adjoint_err,
        "tol": float(tol),
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# 2. Independent Off-Grid Cross-Waveform Serial Propagator (OFDM A -> CP-SC B)
# ---------------------------------------------------------------------------

def _normalize_offgrid_shifts(
    shifts: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]],
    n_rays: int,
) -> tuple[PiecewiseDopplerShift, ...]:
    if shifts is None:
        return ()
    if isinstance(shifts, PiecewiseDopplerShift):
        items = (shifts,)
    else:
        items = tuple(shifts)
    for s in items:
        if not isinstance(s, PiecewiseDopplerShift):
            raise TypeError(f"Expected PiecewiseDopplerShift, got {type(s)!r}")
        if not (0 <= int(s.ray_index) < int(n_rays)):
            raise ValueError(f"ray_index must be in [0, {n_rays - 1}], got {s.ray_index}")
        if not (math.isfinite(s.delta_freq) and math.isfinite(s.t_shift) and math.isfinite(s.delta_chirp)):
            raise ValueError(f"Non-finite shift parameter in {s}")
    return items


def offgrid_phase_and_gain(
    times: np.ndarray,
    gains: np.ndarray,
    freqs: np.ndarray,
    chirps: Optional[np.ndarray] = None,
    *,
    shifts: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact continuous phase phi_l(t) [rad] and complex gain g_l(t) for arbitrary P rays."""
    t_arr = np.asarray(times, dtype=np.float64)
    g0 = np.asarray(gains, dtype=np.complex128)
    f0 = np.asarray(freqs, dtype=np.float64)
    if g0.ndim != 1 or f0.ndim != 1 or g0.shape != f0.shape or g0.size < 1:
        raise ValueError(f"Expected matching non-empty 1D gains/freqs, got {g0.shape} and {f0.shape}")
    n_rays = int(g0.size)
    c0 = np.zeros(n_rays, dtype=np.float64) if chirps is None else np.asarray(chirps, dtype=np.float64)
    if c0.shape != (n_rays,):
        raise ValueError(f"Expected chirps shape ({n_rays},), got {c0.shape}")
    if not (np.isfinite(t_arr).all() and np.isfinite(g0).all() and np.isfinite(f0).all() and np.isfinite(c0).all()):
        raise ValueError("Non-finite input to offgrid_phase_and_gain")

    norm_shifts = _normalize_offgrid_shifts(shifts, n_rays)
    phi = np.empty((n_rays,) + t_arr.shape, dtype=np.float64)
    for l_idx in range(n_rays):
        phi[l_idx] = 2.0 * math.pi * (f0[l_idx] * t_arr + 0.5 * c0[l_idx] * (t_arr ** 2))

    for s in norm_shifts:
        mask = t_arr >= float(s.t_shift)
        dt = np.where(mask, t_arr - float(s.t_shift), 0.0)
        phi[int(s.ray_index)] += 2.0 * math.pi * (
            float(s.delta_freq) * dt + 0.5 * float(s.delta_chirp) * (dt ** 2)
        )

    g_out = g0.reshape((n_rays,) + (1,) * t_arr.ndim) * np.exp(1j * phi)
    return phi, g_out


def verify_offgrid_phase_continuity(
    gains: np.ndarray,
    freqs: np.ndarray,
    chirps: np.ndarray,
    shift: PiecewiseDopplerShift,
    *,
    probe_dt: float = 1e-7,
    post_sample_times: Optional[np.ndarray] = None,
) -> dict:
    """Verify exact instantaneous phase & complex gain continuity (< 1e-14) at arbitrary t_shift for P rays."""
    g0 = np.asarray(gains, dtype=np.complex128)
    f0 = np.asarray(freqs, dtype=np.float64)
    c0 = np.asarray(chirps, dtype=np.float64)
    l_idx = int(shift.ray_index)
    t_s = float(shift.t_shift)
    df = float(shift.delta_freq)
    dc = float(shift.delta_chirp)

    phi_left = 2.0 * math.pi * (f0[l_idx] * t_s + 0.5 * c0[l_idx] * (t_s ** 2))
    g_left = g0[l_idx] * np.exp(1j * phi_left)

    f_post_anchor = f0[l_idx] + c0[l_idx] * t_s + df
    c_post_anchor = c0[l_idx] + dc
    f_eq = f0[l_idx] + df - dc * t_s
    c_eq = c0[l_idx] + dc
    phi_offset = 2.0 * math.pi * (-df * t_s + 0.5 * dc * (t_s ** 2))
    phi_right = phi_offset + 2.0 * math.pi * (f_eq * t_s + 0.5 * c_eq * (t_s ** 2))
    g_right = g0[l_idx] * np.exp(1j * phi_right)

    phi_eval, g_eval = offgrid_phase_and_gain(np.array([t_s], dtype=np.float64), g0, f0, c0, shifts=shift)
    phase_jump_rad = float(max(abs(phi_eval[l_idx, 0] - phi_left), abs(phi_right - phi_left)))
    gain_jump = float(max(abs(g_eval[l_idx, 0] - g_left), abs(g_right - g_left)))

    h = float(probe_dt)
    t_probe = np.array([t_s - h, t_s, t_s + h], dtype=np.float64)
    phi_probe, _ = offgrid_phase_and_gain(t_probe, g0, f0, c0, shifts=shift)
    f_minus = float((phi_probe[l_idx, 1] - phi_probe[l_idx, 0]) / (2.0 * math.pi * h))
    f_plus = float((phi_probe[l_idx, 2] - phi_probe[l_idx, 1]) / (2.0 * math.pi * h))
    freq_jump_measured = f_plus - f_minus
    freq_jump_err = abs(freq_jump_measured - df)

    if post_sample_times is None:
        post_sample_times = np.arange(TOTAL_SAMPLES, dtype=np.float64) / float(N)
    post_t = np.asarray(post_sample_times, dtype=np.float64)
    post_t = post_t[post_t >= t_s]
    if post_t.size > 0:
        tau = post_t - t_s
        g_reanchored = g_left * np.exp(
            2j * math.pi * (f_post_anchor * tau + 0.5 * c_post_anchor * (tau ** 2))
        )
        _, g_piecewise = offgrid_phase_and_gain(post_t, g0, f0, c0, shifts=shift)
        reanchored_max_gain_err = float(np.max(np.abs(g_piecewise[l_idx] - g_reanchored)))
    else:
        reanchored_max_gain_err = 0.0

    return {
        "ray_index": l_idx,
        "t_shift": t_s,
        "delta_freq": df,
        "delta_chirp": dc,
        "phase_jump_rad": phase_jump_rad,
        "gain_jump": gain_jump,
        "freq_jump_measured": freq_jump_measured,
        "freq_jump_err": freq_jump_err,
        "reanchored_max_gain_err": reanchored_max_gain_err,
        "continuous_ok": bool(
            phase_jump_rad < 1e-14 and gain_jump < 1e-14 and reanchored_max_gain_err < 1e-13
        ),
    }


def independent_offgrid_propagate_ab(
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    gains: np.ndarray,
    delays: np.ndarray,
    freqs_a: np.ndarray,
    chirps: Optional[np.ndarray] = None,
    *,
    a_is_freq: bool = True,
    freqs_b: Optional[np.ndarray] = None,
    shift_row: Optional[int] = None,
    t_shift: Optional[float] = None,
    shifts: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]] = None,
    full_reset_b: Optional[Mapping[str, Any]] = None,
    channel_b_time_origin: str = "global",
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | OffGridPropagationResult:
    """Independent sample-by-sample off-grid cross-waveform propagator across OFDM(A) -> CP-SC(B)."""
    a_arr = np.asarray(a_grid, dtype=np.complex128)
    b_arr = np.asarray(b_grid, dtype=np.complex128)
    if a_arr.shape != (T, N) or b_arr.shape != (T, N):
        raise ValueError(f"Expected ({T}, {N}) grids for A and B, got {a_arr.shape} and {b_arr.shape}")

    g_a0 = np.asarray(gains, dtype=np.complex128)
    d_a = np.asarray(delays, dtype=np.float64)
    f_a0 = np.asarray(freqs_a, dtype=np.float64)
    if g_a0.ndim != 1 or d_a.ndim != 1 or f_a0.ndim != 1:
        raise ValueError("gains, delays, and freqs_a must be 1D arrays")
    p_a = int(g_a0.size)
    if p_a < 1 or d_a.shape != (p_a,) or f_a0.shape != (p_a,):
        raise ValueError(f"Mismatched ray arrays: gains={g_a0.shape}, delays={d_a.shape}, freqs_a={f_a0.shape}")
    c_a0 = np.zeros(p_a, dtype=np.float64) if chirps is None else np.asarray(chirps, dtype=np.float64)
    if c_a0.shape != (p_a,):
        raise ValueError(f"Expected chirps shape ({p_a},), got {c_a0.shape}")

    if full_reset_b is not None and (shifts is not None or freqs_b is not None):
        raise ValueError("Specify either full_reset_b OR piecewise shifts/freqs_b, not both")
    if shifts is not None and freqs_b is not None:
        raise ValueError("Specify either explicit `shifts` OR `freqs_b`, not both")

    f_dft = _independent_unitary_dft(N)
    f_idft = f_dft.conj().T

    a_useful = (a_arr @ f_idft.T) if a_is_freq else a_arr.copy()
    b_useful = b_arr.copy()
    useful_ab = np.vstack((a_useful, b_useful))

    tx_blocks = np.empty((TOTAL_BLOCKS, BLOCK_LEN), dtype=np.complex128)
    for m in range(TOTAL_BLOCKS):
        tx_blocks[m, :CP] = useful_ab[m, N - CP:]
        tx_blocks[m, CP:] = useful_ab[m, :]
    tx_serial = tx_blocks.reshape(TOTAL_SAMPLES)
    t_serial = np.arange(TOTAL_SAMPLES, dtype=np.float64) / float(N)
    split_k = T * BLOCK_LEN

    phase_cont_list: Optional[list[dict]] = None
    if full_reset_b is not None:
        g_b0 = np.asarray(full_reset_b["gains"], dtype=np.complex128)
        d_b = np.asarray(full_reset_b["delays"], dtype=np.float64)
        f_b0 = np.asarray(full_reset_b["freqs"], dtype=np.float64)
        p_b = int(g_b0.size)
        c_b0 = (
            np.zeros(p_b, dtype=np.float64)
            if full_reset_b.get("chirps") is None
            else np.asarray(full_reset_b["chirps"], dtype=np.float64)
        )
        if d_b.shape != (p_b,) or f_b0.shape != (p_b,) or c_b0.shape != (p_b,):
            raise ValueError("Mismatched ray array shapes in full_reset_b")
        t_a = t_serial[:split_k]
        if channel_b_time_origin == "global":
            t_b = t_serial[split_k:]
        elif channel_b_time_origin == "frame_b":
            t_b = t_serial[split_k:] - FRAME_DURATION
        else:
            raise ValueError(f"Unsupported channel_b_time_origin: {channel_b_time_origin!r}")
        _, g_serial_a = offgrid_phase_and_gain(t_a, g_a0, f_a0, c_a0)
        _, g_serial_b = offgrid_phase_and_gain(t_b, g_b0, f_b0, c_b0)
    else:
        eff_shifts: tuple[PiecewiseDopplerShift, ...]
        if freqs_b is not None:
            fb_arr = np.asarray(freqs_b, dtype=np.float64)
            if fb_arr.shape != (p_a,):
                raise ValueError(f"Expected freqs_b shape ({p_a},), got {fb_arr.shape}")
            if t_shift is not None:
                ts_val = float(t_shift)
            elif shift_row is not None:
                ts_val = row_boundary_time(int(shift_row), frame="B")
            else:
                ts_val = float(FRAME_DURATION)
            eff_shifts = tuple(
                PiecewiseDopplerShift(
                    ray_index=l_idx,
                    delta_freq=float(fb_arr[l_idx] - f_a0[l_idx]),
                    t_shift=ts_val,
                )
                for l_idx in range(p_a)
                if abs(float(fb_arr[l_idx] - f_a0[l_idx])) > 0.0
            )
        else:
            eff_shifts = _normalize_offgrid_shifts(shifts, p_a)

        _, g_serial_all = offgrid_phase_and_gain(t_serial, g_a0, f_a0, c_a0, shifts=eff_shifts)
        g_serial_a = g_serial_all[:, :split_k]
        g_serial_b = g_serial_all[:, split_k:]
        d_b = d_a
        p_b = p_a
        if eff_shifts:
            phase_cont_list = [
                verify_offgrid_phase_continuity(g_a0, f_a0, c_a0, s, post_sample_times=t_serial)
                for s in eff_shifts
            ]

    d_kernels_a = np.stack([dirichlet_fractional_delay_matrix(float(tau), n=N) for tau in d_a], axis=0)
    if full_reset_b is None:
        d_kernels_b = d_kernels_a
    else:
        d_kernels_b = np.stack([dirichlet_fractional_delay_matrix(float(tau), n=N) for tau in d_b], axis=0)

    rx_serial = np.zeros(TOTAL_SAMPLES, dtype=np.complex128)
    for m in range(TOTAL_BLOCKS):
        u_m = useful_ab[m]
        is_frame_a = m < T
        d_ker = d_kernels_a if is_frame_a else d_kernels_b
        g_frame = g_serial_a if is_frame_a else g_serial_b
        n_r = p_a if is_frame_a else p_b
        local_block = m if is_frame_a else (m - T)

        for s in range(BLOCK_LEN):
            k_global = m * BLOCK_LEN + s
            k_local = local_block * BLOCK_LEN + s
            pos = (s - CP) % N
            acc = 0.0 + 0.0j
            for l_idx in range(n_r):
                delayed_sample = complex(np.dot(d_ker[l_idx, pos], u_m))
                acc += g_frame[l_idx, k_local] * delayed_sample
            rx_serial[k_global] = acc

    rx_blocks = rx_serial.reshape(TOTAL_BLOCKS, BLOCK_LEN)
    rx_useful = rx_blocks[:, CP:]
    y_a_time = rx_useful[:T]
    y_b_time = rx_useful[T:]
    y_a_out = (y_a_time @ f_dft.T) if a_is_freq else y_a_time

    if not return_details:
        return y_a_out, y_b_time

    h_b_useful = g_serial_b.reshape(p_b, T, BLOCK_LEN)[:, :, CP:]
    h_b_true_mat = np.zeros((T, N, N), dtype=np.complex128)
    for r in range(T):
        h_r = np.zeros((N, N), dtype=np.complex128)
        for l_idx in range(p_b):
            h_r += h_b_useful[l_idx, r, :, None] * d_kernels_b[l_idx]
        h_b_true_mat[r] = h_r

    return OffGridPropagationResult(
        y_a=y_a_out,
        y_b=y_b_time,
        tx_serial=tx_serial,
        rx_serial=rx_serial,
        t_serial=t_serial,
        g_serial_a=g_serial_a,
        g_serial_b=g_serial_b,
        h_b_useful=h_b_useful,
        h_b_true_mat=h_b_true_mat,
        phase_continuity=phase_cont_list,
    )


# ---------------------------------------------------------------------------
# 3. Independent Pure-SciPy Complex Gaussian Marginal Log-Likelihood & 2^K Subset-Bayes
# ---------------------------------------------------------------------------

def independent_complex_gaussian_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Exact proper complex Gaussian log-density via pure SciPy scale-safe Cholesky."""
    y_vec = np.asarray(y, dtype=np.complex128).ravel()
    mu_vec = np.asarray(mu, dtype=np.complex128).ravel()
    cov_mat = np.asarray(cov, dtype=np.complex128)
    m_dim = int(y_vec.size)
    if m_dim < 1 or mu_vec.size != m_dim or cov_mat.shape != (m_dim, m_dim):
        raise ValueError(f"Shape mismatch: y={y_vec.shape}, mu={mu_vec.shape}, cov={cov_mat.shape}")
    if not (np.isfinite(y_vec).all() and np.isfinite(mu_vec).all() and np.isfinite(cov_mat).all()):
        raise ValueError("Non-finite input to independent_complex_gaussian_loglik")

    max_entry = float(np.max(np.abs(cov_mat)))
    herm_asym = float(np.max(np.abs(cov_mat - cov_mat.conj().T)))
    if herm_asym > 1e-9 * max(1.0, max_entry):
        raise ValueError(f"Covariance matrix is not Hermitian (max |Sigma - Sigma^H| = {herm_asym:.3e})")
    cov_herm = 0.5 * (cov_mat + cov_mat.conj().T)

    diag_real = np.real(np.diag(cov_herm))
    if np.any(diag_real <= 0.0):
        raise ValueError(f"Covariance diagonal must be strictly positive, min={float(np.min(diag_real))}")
    scale = float(math.fsum(diag_real.tolist()) / float(m_dim))
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"Invalid covariance trace scale: {scale}")

    cov_unit = cov_herm / scale
    chol_lower = cholesky(cov_unit, lower=True, check_finite=True)

    resid_unit = (y_vec - mu_vec) / math.sqrt(scale)
    whitened = solve_triangular(chol_lower, resid_unit, lower=True, check_finite=True)
    quad = float(np.vdot(whitened, whitened).real)
    diag_l = np.real(np.diag(chol_lower))
    logdet = 2.0 * float(math.fsum(math.log(float(d)) for d in diag_l.tolist())) + float(m_dim) * math.log(scale)
    return float(-float(m_dim) * math.log(math.pi) - logdet - quad)


def ray_predictive_moments(
    log_weight: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-ray marginal predictive means [K, M] and Hermitian covariances [K, M, M]."""
    lw = np.asarray(log_weight, dtype=np.float64)
    mn = np.asarray(mean, dtype=np.complex128)
    vr = np.asarray(variance, dtype=np.float64)
    des = np.asarray(design, dtype=np.complex128)
    if lw.ndim != 2 or mn.shape != lw.shape or vr.shape != lw.shape:
        raise ValueError(f"Expected matching [K, H] bank arrays, got {lw.shape}, {mn.shape}, {vr.shape}")
    k_rays, n_hyp = lw.shape
    if des.ndim != 3 or des.shape[:2] != (k_rays, n_hyp):
        raise ValueError(f"Expected design shape [{k_rays}, {n_hyp}, M], got {des.shape}")
    if np.any(vr < 0.0):
        raise ValueError("Ray bank variance must be non-negative")
    m_dim = int(des.shape[2])

    mu_rays = np.zeros((k_rays, m_dim), dtype=np.complex128)
    cov_rays = np.zeros((k_rays, m_dim, m_dim), dtype=np.complex128)
    for l_idx in range(k_rays):
        w_l = np.exp(lw[l_idx] - float(logsumexp(lw[l_idx])))
        d_l = des[l_idx]
        mu_l = np.sum((w_l * mn[l_idx])[:, None] * d_l, axis=0)
        sec_l = w_l * (np.abs(mn[l_idx]) ** 2 + vr[l_idx])
        gram_l = (d_l.T * sec_l[None, :]) @ d_l.conj()
        cov_l = gram_l - np.outer(mu_l, mu_l.conj())
        cov_l = 0.5 * (cov_l + cov_l.conj().T)
        mu_rays[l_idx] = mu_l
        cov_rays[l_idx] = cov_l
    return mu_rays, cov_rays


def independent_subset_bayes_posteriors(
    y: np.ndarray,
    mu_inherit: np.ndarray,
    cov_inherit: np.ndarray,
    mu_clear: np.ndarray,
    cov_clear: np.ndarray,
    n0: float,
    *,
    subset_priors: Optional[np.ndarray] = None,
    ray_priors: Optional[np.ndarray] = None,
    bit_order: str = "lsb",
) -> dict:
    """Exact 2^K subset-Bayes marginal likelihood & posterior reference oracle."""
    y_vec = np.asarray(y, dtype=np.complex128).ravel()
    m_dim = int(y_vec.size)
    mu_i = np.asarray(mu_inherit, dtype=np.complex128)
    mu_c = np.asarray(mu_clear, dtype=np.complex128)
    cov_i = np.asarray(cov_inherit, dtype=np.complex128)
    cov_c = np.asarray(cov_clear, dtype=np.complex128)

    if mu_i.ndim != 2 or mu_c.shape != mu_i.shape or mu_i.shape[1] != m_dim:
        raise ValueError(f"Expected mu_inherit/mu_clear [K, {m_dim}], got {mu_i.shape} and {mu_c.shape}")
    k_rays = int(mu_i.shape[0])
    if k_rays < 1 or k_rays > 10:
        raise ValueError(f"Exact 2^K enumeration requires 1 <= K <= 10, got K={k_rays}")
    if cov_i.shape != (k_rays, m_dim, m_dim) or cov_c.shape != (k_rays, m_dim, m_dim):
        raise ValueError(f"Expected cov arrays [{k_rays}, {m_dim}, {m_dim}], got {cov_i.shape} and {cov_c.shape}")

    if not _is_valid_real_number(n0) or float(n0) < 0.0:
        raise ValueError(f"n0 must be finite and >= 0, got {n0!r}")
    n0_f = float(n0)
    if bit_order not in ("lsb", "msb"):
        raise ValueError(f"bit_order must be 'lsb' or 'msb', got {bit_order!r}")
    if subset_priors is not None and ray_priors is not None:
        raise ValueError("Specify at most one of subset_priors or ray_priors")

    n_subsets = 1 << k_rays
    masks = np.zeros((n_subsets, k_rays), dtype=np.uint8)
    for s in range(n_subsets):
        for l_idx in range(k_rays):
            shift = l_idx if bit_order == "lsb" else (k_rays - 1 - l_idx)
            masks[s, l_idx] = (s >> shift) & 1

    if subset_priors is not None:
        sp = np.asarray(subset_priors, dtype=np.float64)
        if sp.shape != (n_subsets,) or not np.isfinite(sp).all() or np.any(sp < 0.0) or float(np.sum(sp)) <= 0.0:
            raise ValueError("subset_priors must be non-negative finite probabilities with positive sum")
        sp = sp / float(math.fsum(sp.tolist()))
        log_priors = np.where(sp > 0.0, np.log(np.maximum(sp, 1e-300)), -math.inf)
    elif ray_priors is not None:
        rp = np.asarray(ray_priors, dtype=np.float64)
        if rp.shape != (k_rays,) or not np.isfinite(rp).all() or np.any(rp < 0.0) or np.any(rp > 1.0):
            raise ValueError("ray_priors must lie in [0, 1]")
        log_priors = np.zeros(n_subsets, dtype=np.float64)
        for s in range(n_subsets):
            lp = 0.0
            for l_idx in range(k_rays):
                p_l = float(rp[l_idx]) if masks[s, l_idx] == 1 else (1.0 - float(rp[l_idx]))
                lp = -math.inf if p_l == 0.0 else (lp + math.log(p_l))
            log_priors[s] = lp
    else:
        log_priors = np.full(n_subsets, -float(k_rays) * math.log(2.0), dtype=np.float64)

    noise_eye = n0_f * np.eye(m_dim, dtype=np.complex128)
    subset_logliks = np.empty(n_subsets, dtype=np.float64)
    for s in range(n_subsets):
        mu_s = np.zeros(m_dim, dtype=np.complex128)
        cov_s = noise_eye.copy()
        for l_idx in range(k_rays):
            if masks[s, l_idx] == 1:
                mu_s += mu_i[l_idx]
                cov_s += cov_i[l_idx]
            else:
                mu_s += mu_c[l_idx]
                cov_s += cov_c[l_idx]
        cov_s = 0.5 * (cov_s + cov_s.conj().T)
        subset_logliks[s] = independent_complex_gaussian_loglik(y_vec, mu_s, cov_s)

    log_unnorm = log_priors + subset_logliks
    log_evidence = float(logsumexp(log_unnorm))
    subset_posteriors = np.exp(log_unnorm - log_evidence).astype(np.float64)
    subset_posteriors = subset_posteriors / float(math.fsum(subset_posteriors.tolist()))

    ray_posteriors = np.array(
        [float(math.fsum(subset_posteriors[masks[:, l_idx] == 1].tolist())) for l_idx in range(k_rays)],
        dtype=np.float64,
    )

    return {
        "k_rays": k_rays,
        "n_subsets": n_subsets,
        "bit_order": bit_order,
        "subset_masks": masks,
        "subset_log_priors": log_priors,
        "subset_logliks": subset_logliks,
        "subset_posteriors": subset_posteriors,
        "ray_posteriors": ray_posteriors,
        "log_evidence": log_evidence,
        "all_clear_idx": 0,
        "all_inherit_idx": n_subsets - 1,
        "all_inherit_vs_clear_logbf": float(subset_logliks[n_subsets - 1] - subset_logliks[0]),
    }


# ---------------------------------------------------------------------------
# 4. Strict Runtime Anti-Leakage Firewalls, Seed Disjointness & Static Audits
# ---------------------------------------------------------------------------

def verify_offgrid_observation(obs: Any, *, strict: bool = False) -> dict:
    """Strictly verify p11.contract.OffGridObservation schema, NaN probe masking, and zero hidden truth fields."""
    errors: list[str] = []

    if not isinstance(obs, OffGridObservation):
        errors.append(f"Expected p11.contract.OffGridObservation instance, got {type(obs).__name__}")

    present_attrs = set(getattr(obs, "__dict__", {}).keys())
    forbidden_present = sorted(present_attrs & FORBIDDEN_OFFGRID_HIDDEN_ATTRS)
    if forbidden_present:
        errors.append(f"Forbidden hidden truth attributes attached: {forbidden_present}")
    unexpected_attrs = sorted(present_attrs - ALLOWED_OFFGRID_ATTRS - FORBIDDEN_OFFGRID_HIDDEN_ATTRS)
    if unexpected_attrs:
        errors.append(f"Unexpected non-contract attributes attached: {unexpected_attrs}")

    if not errors:
        y_a = np.asarray(obs.y_a)
        y_b = np.asarray(obs.y_b)
        if y_a.shape != (T, N) or y_a.dtype != np.complex128:
            errors.append(f"y_a must have shape {(T, N)} and dtype complex128, got {y_a.shape}/{y_a.dtype}")
        elif not np.isfinite(y_a).all():
            errors.append("y_a contains non-finite values")

        if not _is_valid_int_seed(obs.probe_row):
            errors.append(f"probe_row must be an exact integer, got {type(obs.probe_row).__name__}")
        elif int(obs.probe_row) != PROBE_ROW:
            errors.append(f"probe_row must be {PROBE_ROW}, got {obs.probe_row}")

        if y_b.shape != (T, N) or y_b.dtype != np.complex128:
            errors.append(f"y_b must have shape {(T, N)} and dtype complex128, got {y_b.shape}/{y_b.dtype}")
        elif _is_valid_int_seed(obs.probe_row):
            p_row = int(obs.probe_row)
            if 0 <= p_row < T:
                probe_slice = y_b[p_row]
                if not (np.isnan(probe_slice.real).all() and np.isnan(probe_slice.imag).all()):
                    errors.append("Withheld probe_row in y_b is NOT strictly masked to NaN+1j*NaN")
                non_probe = np.delete(y_b, p_row, axis=0)
                if not np.isfinite(non_probe).all():
                    errors.append("Non-probe rows in y_b contain non-finite values")

        try:
            tr_raw = tuple(obs.training_rows)
        except TypeError:
            tr_raw = ()
            errors.append("training_rows is not iterable")
        if len(tr_raw) == 0:
            errors.append("training_rows must be non-empty")
        else:
            tr_ints: list[int] = []
            for item in tr_raw:
                if not _is_valid_int_seed(item):
                    errors.append(f"training_rows contains non-integer entry: {item!r} ({type(item).__name__})")
                else:
                    r_val = int(item)
                    if not (0 <= r_val < T):
                        errors.append(f"training_rows entry out of range [0, {T - 1}]: {r_val}")
                    tr_ints.append(r_val)
            if len(set(tr_ints)) != len(tr_ints):
                errors.append(f"training_rows contains duplicates: {tr_ints}")
            if PROBE_ROW in tr_ints or (
                _is_valid_int_seed(obs.probe_row) and int(obs.probe_row) in tr_ints
            ):
                errors.append("probe_row appears inside training_rows")

        if not _is_valid_real_number(obs.n0) or float(obs.n0) <= 0.0:
            errors.append(
                f"n0 must be a strictly positive finite real float > 0.0, got {obs.n0!r} ({type(obs.n0).__name__})"
            )

        pt_b = np.asarray(obs.pilot_tx_b)
        expected_b_shape = (len(tr_raw), N)
        if pt_b.shape != expected_b_shape or pt_b.dtype != np.complex128:
            errors.append(f"pilot_tx_b must have shape {expected_b_shape} and complex128, got {pt_b.shape}/{pt_b.dtype}")
        elif not np.isfinite(pt_b).all():
            errors.append("pilot_tx_b contains non-finite values")

        pt_a = np.asarray(obs.pilot_tx_a)
        expected_a_shape = (len(OFDM_PILOT_ROWS), N // 2)
        if pt_a.shape != expected_a_shape or pt_a.dtype != np.complex128:
            errors.append(f"pilot_tx_a must have shape {expected_a_shape} and complex128, got {pt_a.shape}/{pt_a.dtype}")
        elif not np.isfinite(pt_a).all():
            errors.append("pilot_tx_a contains non-finite values")

    report = {"valid": len(errors) == 0, "errors": errors}
    if strict and errors:
        raise ValueError(f"OffGridObservation firewall check failed: {errors}")
    return report


def verify_p11_public_observation(obs: Any, *, strict: bool = False) -> dict:
    """Wrap p9.independent_ref.verify_public_observation with strict timedelta64/datetime64 guards."""
    errors: list[str] = []
    if not isinstance(obs, PublicObservation):
        errors.append(f"Expected PublicObservation instance, got {type(obs).__name__}")
    else:
        if not _is_valid_int_seed(obs.probe_row):
            errors.append(f"probe_row must be an exact integer, got {type(obs.probe_row).__name__}")
        if not _is_valid_real_number(obs.n0) or float(obs.n0) <= 0.0:
            errors.append(
                f"n0 must be a strictly positive finite real float > 0.0, got {obs.n0!r} ({type(obs.n0).__name__})"
            )
        try:
            for item in obs.training_rows:
                if not _is_valid_int_seed(item):
                    errors.append(f"training_rows contains invalid integer entry: {item!r} ({type(item).__name__})")
        except TypeError:
            errors.append("training_rows is not iterable")

    if not errors:
        base_rep = verify_public_observation(obs, strict=False)
        errors.extend(base_rep.get("errors", []))

    report = {"valid": len(errors) == 0, "errors": errors}
    if strict and errors:
        raise ValueError(f"PublicObservation firewall check failed: {errors}")
    return report


def check_assimilation_result_adapter(
    res: Any,
    *,
    target_cls: Optional[type] = None,
    strict: bool = False,
) -> dict:
    """Validate 7-field AssimilationResult / GateOutput structural parity and optional class adaptation."""
    errors: list[str] = []
    required_fields = (
        "gate_out",
        "decoded_info",
        "assimilated_rows",
        "row_weights",
        "row_margins",
        "n_rounds",
        "elapsed_ms",
    )
    gate_fields = (
        "history_weight",
        "probe_weight",
        "mix_llrs",
        "mix_probe_pred",
        "mix_probe_var",
        "mix_mean_ray",
        "mix_var_ray",
    )

    missing = [f for f in required_fields if not hasattr(res, f)]
    if missing:
        errors.append(f"Missing AssimilationResult fields: {missing}")

    adapted = None
    if not errors:
        g_in = getattr(res, "gate_out")
        g_missing = [f for f in gate_fields if not hasattr(g_in, f)]
        if g_missing:
            errors.append(f"Missing GateOutput fields: {g_missing}")
        else:
            cls_to_build = AssimilationResult if target_cls is None else target_cls
            field_errors: list[str] = []
            arr_hw, e_hw = _validate_numeric_array(
                g_in.history_weight, allow_complex=False, field_name="gate_out.history_weight"
            )
            arr_llrs, e_llrs = _validate_numeric_array(
                g_in.mix_llrs, allow_complex=False, field_name="gate_out.mix_llrs"
            )
            arr_pp, e_pp = _validate_numeric_array(
                g_in.mix_probe_pred, allow_complex=True, field_name="gate_out.mix_probe_pred"
            )
            arr_pv, e_pv = _validate_numeric_array(
                g_in.mix_probe_var, allow_complex=False, field_name="gate_out.mix_probe_var"
            )
            arr_mray, e_mray = _validate_numeric_array(
                g_in.mix_mean_ray, allow_complex=True, field_name="gate_out.mix_mean_ray"
            )
            arr_vray, e_vray = _validate_numeric_array(
                g_in.mix_var_ray, allow_complex=False, field_name="gate_out.mix_var_ray"
            )
            for err in (e_hw, e_llrs, e_pp, e_pv, e_mray, e_vray):
                if err is not None:
                    field_errors.append(err)
            if e_hw is None and arr_hw.shape != (T,):
                field_errors.append(f"gate_out.history_weight must have shape ({T},), got {arr_hw.shape}")
            if e_llrs is None and arr_llrs.shape != (T, N, 4):
                field_errors.append(f"gate_out.mix_llrs must have shape ({T}, {N}, 4), got {arr_llrs.shape}")
            if e_pp is None and arr_pp.shape != (N,):
                field_errors.append(f"gate_out.mix_probe_pred must have shape ({N},), got {arr_pp.shape}")
            if e_pv is None and arr_pv.shape != (N,):
                field_errors.append(f"gate_out.mix_probe_var must have shape ({N},), got {arr_pv.shape}")
            if e_mray is None and (arr_mray.ndim != 3 or arr_mray.shape[0] < 1 or arr_mray.shape[1:] != (T, N)):
                field_errors.append(
                    f"gate_out.mix_mean_ray must have shape (K, {T}, {N}) with K >= 1, got {arr_mray.shape}"
                )
            if e_vray is None and (arr_vray.ndim != 3 or arr_vray.shape[0] < 1 or arr_vray.shape[1:] != (T, N)):
                field_errors.append(
                    f"gate_out.mix_var_ray must have shape (K, {T}, {N}) with K >= 1, got {arr_vray.shape}"
                )
            elif e_mray is None and e_vray is None and arr_vray.shape != arr_mray.shape:
                field_errors.append(
                    f"gate_out.mix_var_ray shape {arr_vray.shape} != mix_mean_ray shape {arr_mray.shape}"
                )
            if not _is_valid_real_number(g_in.probe_weight):
                field_errors.append(
                    f"gate_out.probe_weight must be a finite real number, got {g_in.probe_weight!r} ({type(g_in.probe_weight).__name__})"
                )

            if not _is_valid_int_seed(res.n_rounds) or int(res.n_rounds) < 0:
                field_errors.append(
                    f"n_rounds must be a non-negative integer, got {res.n_rounds!r} ({type(res.n_rounds).__name__})"
                )
            if not _is_valid_real_number(res.elapsed_ms) or float(res.elapsed_ms) < 0.0:
                field_errors.append(
                    f"elapsed_ms must be a finite non-negative real number, got {res.elapsed_ms!r} ({type(res.elapsed_ms).__name__})"
                )

            assim_rows_clean: list[int] = []
            if isinstance(res.assimilated_rows, (str, bytes, Mapping)):
                field_errors.append(f"assimilated_rows must be a sequence, got {type(res.assimilated_rows).__name__}")
            else:
                try:
                    for item in res.assimilated_rows:
                        if not _is_valid_int_seed(item):
                            field_errors.append(
                                f"assimilated_rows contains non-integer entry {item!r} ({type(item).__name__})"
                            )
                        else:
                            assim_rows_clean.append(int(item))
                except TypeError:
                    field_errors.append(f"assimilated_rows is not iterable: {type(res.assimilated_rows).__name__}")

            rw_clean: dict[int, float] = {}
            if not isinstance(res.row_weights, Mapping):
                field_errors.append(f"row_weights must be a Mapping, got {type(res.row_weights).__name__}")
            else:
                for k, v in res.row_weights.items():
                    if not _is_valid_int_seed(k):
                        field_errors.append(f"row_weights key {k!r} ({type(k).__name__}) is not an integer")
                    elif not _is_valid_real_number(v):
                        field_errors.append(
                            f"row_weights[{k!r}]={v!r} ({type(v).__name__}) is not a finite real number"
                        )
                    else:
                        rw_clean[int(k)] = float(v)

            rm_clean: dict[int, float] = {}
            if not isinstance(res.row_margins, Mapping):
                field_errors.append(f"row_margins must be a Mapping, got {type(res.row_margins).__name__}")
            else:
                for k, v in res.row_margins.items():
                    if not _is_valid_int_seed(k):
                        field_errors.append(f"row_margins key {k!r} ({type(k).__name__}) is not an integer")
                    elif not _is_valid_real_number(v):
                        field_errors.append(
                            f"row_margins[{k!r}]={v!r} ({type(v).__name__}) is not a finite real number"
                        )
                    else:
                        rm_clean[int(k)] = float(v)

            di_clean: dict[int, np.ndarray] = {}
            if not isinstance(res.decoded_info, Mapping):
                field_errors.append(f"decoded_info must be a Mapping, got {type(res.decoded_info).__name__}")
            else:
                for k, v in res.decoded_info.items():
                    if not _is_valid_int_seed(k):
                        field_errors.append(f"decoded_info key {k!r} ({type(k).__name__}) is not an integer")
                    else:
                        _, v_err = _validate_numeric_array(
                            v, allow_complex=False, field_name=f"decoded_info[{k!r}]"
                        )
                        if v_err is not None:
                            field_errors.append(v_err)
                        else:
                            di_clean[int(k)] = np.asarray(v)

            errors.extend(field_errors)
            if not errors:
                try:
                    g_adapted = g_in if isinstance(g_in, GateOutput) else GateOutput(
                        history_weight=arr_hw,
                        probe_weight=float(g_in.probe_weight),
                        mix_llrs=arr_llrs,
                        mix_probe_pred=arr_pp,
                        mix_probe_var=arr_pv,
                        mix_mean_ray=arr_mray,
                        mix_var_ray=arr_vray,
                    )
                    adapted = cls_to_build(
                        gate_out=g_adapted,
                        decoded_info=di_clean,
                        assimilated_rows=tuple(assim_rows_clean),
                        row_weights=rw_clean,
                        row_margins=rm_clean,
                        n_rounds=int(res.n_rounds),
                        elapsed_ms=float(res.elapsed_ms),
                    )
                except Exception as exc:
                    errors.append(f"Adapter construction failed for {cls_to_build.__name__}: {exc}")

    report = {
        "valid": len(errors) == 0,
        "is_p11_type": isinstance(res, AssimilationResult),
        "source_type": type(res).__name__,
        "target_type": AssimilationResult.__name__ if target_cls is None else target_cls.__name__,
        "adapted_instance": adapted,
        "errors": errors,
    }
    if strict and errors:
        raise ValueError(f"AssimilationResult adapter check failed: {errors}")
    return report


def _clone_observation(
    obs: PublicObservation | OffGridObservation,
    *,
    y_a: np.ndarray,
    y_b: np.ndarray,
    n0: float,
) -> PublicObservation | OffGridObservation:
    if isinstance(obs, PublicObservation):
        return PublicObservation(
            y_a=y_a,
            y_b=y_b,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=float(n0),
            budget=obs.budget,
        )
    if isinstance(obs, OffGridObservation):
        return OffGridObservation(
            y_a=y_a,
            y_b=y_b,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=float(n0),
            pilot_tx_b=obs.pilot_tx_b.copy(),
            pilot_tx_a=obs.pilot_tx_a.copy(),
        )
    raise TypeError(f"Unsupported observation type: {type(obs).__name__}")


def _scaled_diff(scaled_arr: np.ndarray, base_arr: np.ndarray, divisor: complex | float) -> float:
    pulled = np.asarray(scaled_arr) / divisor
    base = np.asarray(base_arr)
    abs_err = float(np.max(np.abs(pulled - base)))
    norm_ref = max(1.0, float(np.max(np.abs(base)))) if base.size > 0 else 1.0
    return abs_err / norm_ref


def audit_assimilation_anti_leakage(
    assimilate_fn: Callable[[Any], AssimilationResult],
    obs: PublicObservation | OffGridObservation,
    *,
    cstar_tol: float = 1e-12,
    cstar_margin_tol: Optional[float] = None,
    cstar_alphas: Sequence[complex] = (
        2.75 * complex(math.cos(1.17), math.sin(1.17)),
        0.37 * complex(math.cos(-2.09), math.sin(-2.09)),
    ),
    allow_historical_adapter: bool = False,
    strict: bool = False,
) -> dict:
    """Strict runtime anti-leakage & C*-invariance audit for P11 blind wavefront assimilation.

    Binds strictly to `p11.contract.AssimilationResult` by default (rejecting foreign classes
    unless `allow_historical_adapter=True`), rejects `timedelta64`/`datetime64`/`bool` scalar leaks,
    verifies exact 0.0 probe-row poisoning invariance, and checks < cstar_tol C* equivariance
    (with separate `cstar_margin_tol` defaulting to `max(100 * cstar_tol, 1e-10)` for `row_margins`).
    """
    if not _is_valid_real_number(cstar_tol) or float(cstar_tol) < 0.0:
        raise ValueError(f"cstar_tol must be a non-negative finite real number, got {cstar_tol!r}")
    if cstar_margin_tol is None:
        margin_tol = max(100.0 * float(cstar_tol), 1e-10) if float(cstar_tol) > 0.0 else float(cstar_tol)
    else:
        if not _is_valid_real_number(cstar_margin_tol) or float(cstar_margin_tol) < 0.0:
            raise ValueError(f"cstar_margin_tol must be a non-negative finite real number, got {cstar_margin_tol!r}")
        margin_tol = float(cstar_margin_tol)
    if isinstance(obs, PublicObservation):
        obs_check = verify_p11_public_observation(obs, strict=strict)
    elif isinstance(obs, OffGridObservation):
        obs_check = verify_offgrid_observation(obs, strict=strict)
    else:
        raise TypeError(f"Expected PublicObservation or p11.contract.OffGridObservation, got {type(obs).__name__}")

    issues: list[str] = list(obs_check["errors"])
    probe_row = int(obs.probe_row)
    training_set = {int(r) for r in obs.training_rows}
    valid_data_rows = frozenset(r for r in range(T) if r != probe_row and r not in training_set)

    def _coerce_res(raw_res: Any, label: str) -> Optional[AssimilationResult]:
        if isinstance(raw_res, AssimilationResult):
            return raw_res
        if allow_historical_adapter:
            ad = check_assimilation_result_adapter(raw_res, target_cls=AssimilationResult, strict=False)
            if ad["valid"] and isinstance(ad["adapted_instance"], AssimilationResult):
                return ad["adapted_instance"]
            issues.append(f"{label} adapter failed: {ad['errors']}")
            return None
        issues.append(
            f"{label} must return p11.contract.AssimilationResult, got {type(raw_res).__module__}.{type(raw_res).__name__}"
        )
        return None

    base_raw = assimilate_fn(obs)
    base = _coerce_res(base_raw, "assimilate_fn")
    if base is None:
        report = {
            "passed": False,
            "issues": issues,
            "max_probe_poison_diff": math.inf,
            "max_cstar_err": math.inf,
            "probe_excluded": False,
            "all_assimilated_rows_valid": False,
        }
        if strict:
            raise ValueError(f"Assimilation anti-leakage audit failed: {issues}")
        return report

    g_base = base.gate_out
    if not isinstance(g_base, GateOutput):
        issues.append(f"AssimilationResult.gate_out must be GateOutput, got {type(g_base).__name__}")

    base_pred, e_bp = _validate_numeric_array(g_base.mix_probe_pred, allow_complex=True, field_name="mix_probe_pred")
    base_var, e_bv = _validate_numeric_array(g_base.mix_probe_var, allow_complex=False, field_name="mix_probe_var")
    base_llrs, e_bl = _validate_numeric_array(g_base.mix_llrs, allow_complex=False, field_name="mix_llrs")
    base_hw, e_bh = _validate_numeric_array(g_base.history_weight, allow_complex=False, field_name="history_weight")
    base_mray, e_bm = _validate_numeric_array(g_base.mix_mean_ray, allow_complex=True, field_name="mix_mean_ray")
    base_vray, e_bvr = _validate_numeric_array(g_base.mix_var_ray, allow_complex=False, field_name="mix_var_ray")
    for err in (e_bp, e_bv, e_bl, e_bh, e_bm, e_bvr):
        if err is not None:
            issues.append(err)
    if base_hw is not None and base_hw.shape != (T,):
        issues.append(f"gate_out.history_weight must have shape ({T},), got {base_hw.shape}")
    if base_llrs is not None and base_llrs.shape != (T, N, 4):
        issues.append(f"gate_out.mix_llrs must have shape ({T}, {N}, 4), got {base_llrs.shape}")
    if base_pred is not None and base_pred.shape != (N,):
        issues.append(f"gate_out.mix_probe_pred must have shape ({N},), got {base_pred.shape}")
    if base_var is not None and base_var.shape != (N,):
        issues.append(f"gate_out.mix_probe_var must have shape ({N},), got {base_var.shape}")
    if base_mray is not None and (base_mray.ndim != 3 or base_mray.shape[0] < 1 or base_mray.shape[1:] != (T, N)):
        issues.append(f"gate_out.mix_mean_ray must have shape (K, {T}, {N}) with K >= 1, got {base_mray.shape}")
    if base_vray is not None and (base_vray.ndim != 3 or base_vray.shape[0] < 1 or base_vray.shape[1:] != (T, N)):
        issues.append(f"gate_out.mix_var_ray must have shape (K, {T}, {N}) with K >= 1, got {base_vray.shape}")
    elif base_mray is not None and base_vray is not None and base_vray.shape != base_mray.shape:
        issues.append(f"gate_out.mix_var_ray shape {base_vray.shape} != mix_mean_ray shape {base_mray.shape}")

    base_arrays_ok = all(
        arr is not None and np.isfinite(arr).all()
        for arr in (base_pred, base_var, base_llrs, base_hw, base_mray, base_vray)
    )
    if not (base_arrays_ok and _is_valid_real_number(g_base.probe_weight)):
        issues.append("Base assimilation GateOutput contains non-finite or invalid values")
    if base_var is not None and np.isfinite(base_var).all() and np.any(base_var < float(obs.n0) - 1e-12):
        issues.append(f"mix_probe_var fell below noise floor n0={obs.n0}: min={float(np.min(base_var))}")

    assim_tuple = tuple(base.assimilated_rows)
    probe_in_assim = (PROBE_ROW in assim_tuple) or (probe_row in assim_tuple)
    probe_in_decoded = (PROBE_ROW in base.decoded_info) or (probe_row in base.decoded_info)
    probe_in_weights = (PROBE_ROW in base.row_weights) or (probe_row in base.row_weights)
    probe_in_margins = (PROBE_ROW in base.row_margins) or (probe_row in base.row_margins)
    probe_excluded = not (probe_in_assim or probe_in_decoded or probe_in_weights or probe_in_margins)
    if probe_in_assim:
        issues.append(f"PROBE_ROW ({PROBE_ROW}) found inside res.assimilated_rows: {assim_tuple}")
    if probe_in_decoded:
        issues.append(f"PROBE_ROW ({PROBE_ROW}) found inside res.decoded_info")
    if probe_in_weights:
        issues.append(f"PROBE_ROW ({PROBE_ROW}) found inside res.row_weights")
    if probe_in_margins:
        issues.append(f"PROBE_ROW ({PROBE_ROW}) found inside res.row_margins")

    all_assimilated_rows_valid = True
    seen_assim: set[int] = set()
    for r in assim_tuple:
        if not _is_valid_int_seed(r):
            all_assimilated_rows_valid = False
            issues.append(f"Non-integer entry in assimilated_rows: {r!r} ({type(r).__name__})")
            continue
        r_int = int(r)
        if r_int in seen_assim:
            all_assimilated_rows_valid = False
            issues.append(f"Duplicate row {r_int} in assimilated_rows")
        seen_assim.add(r_int)
        if r_int not in valid_data_rows:
            all_assimilated_rows_valid = False
            issues.append(f"Assimilated row {r_int} is NOT a valid public data row (valid={sorted(valid_data_rows)})")

    for r_key, bits in base.decoded_info.items():
        if not _is_valid_int_seed(r_key) or int(r_key) not in valid_data_rows:
            issues.append(f"decoded_info key {r_key!r} is not a valid public data row")
        b_arr, b_err = _validate_numeric_array(bits, allow_complex=False, field_name=f"decoded_info[{r_key!r}]")
        if b_err is not None:
            issues.append(b_err)
        elif b_arr.shape != (64,) or not np.isin(b_arr, (0, 1)).all():
            issues.append(f"decoded_info[{r_key!r}] must be binary shape (64,), got {b_arr.shape}/{np.asarray(bits).dtype}")

    for r_key, w_val in base.row_weights.items():
        if not _is_valid_int_seed(r_key):
            issues.append(f"row_weights key {r_key!r} is not an integer")
        if not (_is_valid_real_number(w_val) and 0.0 <= float(w_val) <= 1.0):
            issues.append(f"row_weights[{r_key!r}]={w_val!r} outside [0, 1]")

    for r_key, m_val in base.row_margins.items():
        if not _is_valid_int_seed(r_key):
            issues.append(f"row_margins key {r_key!r} is not an integer")
        if not _is_valid_real_number(m_val):
            issues.append(f"row_margins[{r_key!r}]={m_val!r} is not a finite real number")

    if not _is_valid_int_seed(base.n_rounds) or int(base.n_rounds) < 0:
        issues.append(f"n_rounds must be a non-negative integer, got {base.n_rounds!r}")
    if not _is_valid_real_number(base.elapsed_ms) or float(base.elapsed_ms) < 0.0:
        issues.append(f"elapsed_ms must be finite non-negative real, got {base.elapsed_ms!r}")

    max_probe_poison_diff = 0.0
    poison_field_diffs: dict[str, Any] = {
        "decoded_info_bit_diffs": 0,
        "assimilated_rows_changed": False,
        "row_weights_diff": 0.0,
        "row_margins_diff": 0.0,
        "n_rounds_changed": False,
        "history_weight_diff": 0.0,
        "probe_weight_diff": 0.0,
        "mix_probe_pred_diff": 0.0,
        "mix_probe_var_diff": 0.0,
        "mix_llrs_diff": 0.0,
        "mix_mean_ray_diff": 0.0,
        "mix_var_ray_diff": 0.0,
    }
    base_assim_ints = [int(r) for r in assim_tuple if _is_valid_int_seed(r)]
    base_di_keys = {int(k) for k in base.decoded_info.keys() if _is_valid_int_seed(k)}
    base_rw_keys = {int(k) for k in base.row_weights.keys() if _is_valid_int_seed(k)}
    base_rm_keys = {int(k) for k in base.row_margins.keys() if _is_valid_int_seed(k)}

    for sentinel in (1e6 - 1e6j, -5e5 + 3e5j):
        y_poison = obs.y_b.copy()
        y_poison[probe_row] = sentinel
        obs_poison = _clone_observation(obs, y_a=obs.y_a.copy(), y_b=y_poison, n0=float(obs.n0))
        p_res = _coerce_res(assimilate_fn(obs_poison), "probe_poison_call")
        if p_res is None:
            max_probe_poison_diff = math.inf
            continue

        p_assim_raw = tuple(p_res.assimilated_rows)
        if (not all(_is_valid_int_seed(r) for r in p_assim_raw)) or ([int(r) for r in p_assim_raw] != base_assim_ints):
            poison_field_diffs["assimilated_rows_changed"] = True
            max_probe_poison_diff = math.inf
            issues.append(
                f"Probe poisoning altered assimilated_rows: {assim_tuple} -> {p_assim_raw}"
            )

        if (
            not isinstance(p_res.decoded_info, Mapping)
            or not all(_is_valid_int_seed(k) for k in p_res.decoded_info.keys())
            or {int(k) for k in p_res.decoded_info.keys()} != base_di_keys
        ):
            poison_field_diffs["decoded_info_bit_diffs"] += 64
            max_probe_poison_diff = math.inf
            issues.append("Probe poisoning altered decoded_info row keys or returned non-int keys")
        else:
            for r_key, b_base in base.decoded_info.items():
                p_bits, p_b_err = _validate_numeric_array(
                    p_res.decoded_info[r_key],
                    allow_complex=False,
                    field_name=f"probe_poison.decoded_info[{r_key!r}]",
                )
                if p_b_err is not None:
                    max_probe_poison_diff = math.inf
                    issues.append(p_b_err)
                    continue
                b_base_arr = np.asarray(b_base)
                if p_bits.shape != b_base_arr.shape:
                    poison_field_diffs["decoded_info_bit_diffs"] += 64
                    max_probe_poison_diff = math.inf
                    issues.append(
                        f"Probe poisoning changed decoded_info[{r_key!r}] shape: {b_base_arr.shape} -> {p_bits.shape}"
                    )
                    continue
                bit_diffs = int(np.count_nonzero(p_bits != b_base_arr))
                if bit_diffs > 0:
                    poison_field_diffs["decoded_info_bit_diffs"] += bit_diffs
                    max_probe_poison_diff = max(max_probe_poison_diff, float(bit_diffs))
                    issues.append(f"Probe poisoning flipped {bit_diffs} decoded_info bits on row {r_key}")

        if (
            not isinstance(p_res.row_weights, Mapping)
            or not all(_is_valid_int_seed(k) for k in p_res.row_weights.keys())
            or {int(k) for k in p_res.row_weights.keys()} != base_rw_keys
        ):
            poison_field_diffs["row_weights_diff"] = math.inf
            max_probe_poison_diff = math.inf
            issues.append("Probe poisoning altered row_weights keys or returned non-int keys")
        else:
            for r_key, w_base in base.row_weights.items():
                if not _is_valid_real_number(p_res.row_weights[r_key]) or not _is_valid_real_number(w_base):
                    poison_field_diffs["row_weights_diff"] = math.inf
                    max_probe_poison_diff = math.inf
                    issues.append(f"Probe poisoning invalid row_weights[{r_key}]={p_res.row_weights[r_key]!r}")
                    continue
                w_diff = abs(float(p_res.row_weights[r_key]) - float(w_base))
                poison_field_diffs["row_weights_diff"] = max(poison_field_diffs["row_weights_diff"], w_diff)
                max_probe_poison_diff = max(max_probe_poison_diff, w_diff)
                if w_diff != 0.0:
                    issues.append(f"Probe poisoning altered row_weights[{r_key}] by {w_diff:.3e} (must be exact 0.0)")

        if (
            not isinstance(p_res.row_margins, Mapping)
            or not all(_is_valid_int_seed(k) for k in p_res.row_margins.keys())
            or {int(k) for k in p_res.row_margins.keys()} != base_rm_keys
        ):
            poison_field_diffs["row_margins_diff"] = math.inf
            max_probe_poison_diff = math.inf
            issues.append("Probe poisoning altered row_margins keys or returned non-int keys")
        else:
            for r_key, m_base in base.row_margins.items():
                if not _is_valid_real_number(p_res.row_margins[r_key]) or not _is_valid_real_number(m_base):
                    poison_field_diffs["row_margins_diff"] = math.inf
                    max_probe_poison_diff = math.inf
                    issues.append(f"Probe poisoning invalid row_margins[{r_key}]={p_res.row_margins[r_key]!r}")
                    continue
                m_diff = abs(float(p_res.row_margins[r_key]) - float(m_base))
                poison_field_diffs["row_margins_diff"] = max(poison_field_diffs["row_margins_diff"], m_diff)
                max_probe_poison_diff = max(max_probe_poison_diff, m_diff)
                if m_diff != 0.0:
                    issues.append(f"Probe poisoning altered row_margins[{r_key}] by {m_diff:.3e} (must be exact 0.0)")

        if not _is_valid_int_seed(p_res.n_rounds) or (
            _is_valid_int_seed(base.n_rounds) and int(p_res.n_rounds) != int(base.n_rounds)
        ):
            poison_field_diffs["n_rounds_changed"] = True
            max_probe_poison_diff = math.inf
            issues.append(f"Probe poisoning altered n_rounds: {base.n_rounds!r} -> {p_res.n_rounds!r}")

        pg = p_res.gate_out
        p_hw, e_phw = _validate_numeric_array(
            pg.history_weight, allow_complex=False, field_name="probe_poison.history_weight"
        )
        p_pred, e_pp = _validate_numeric_array(
            pg.mix_probe_pred, allow_complex=True, field_name="probe_poison.mix_probe_pred"
        )
        p_var, e_pv = _validate_numeric_array(
            pg.mix_probe_var, allow_complex=False, field_name="probe_poison.mix_probe_var"
        )
        p_llrs, e_pl = _validate_numeric_array(
            pg.mix_llrs, allow_complex=False, field_name="probe_poison.mix_llrs"
        )
        p_mray, e_pmr = _validate_numeric_array(
            pg.mix_mean_ray, allow_complex=True, field_name="probe_poison.mix_mean_ray"
        )
        p_vray, e_pvr = _validate_numeric_array(
            pg.mix_var_ray, allow_complex=False, field_name="probe_poison.mix_var_ray"
        )
        for err in (e_phw, e_pp, e_pv, e_pl, e_pmr, e_pvr):
            if err is not None:
                issues.append(err)
        if not _is_valid_real_number(pg.probe_weight):
            poison_field_diffs["probe_weight_diff"] = math.inf
            max_probe_poison_diff = math.inf
            issues.append(f"Probe poisoning produced invalid probe_weight {pg.probe_weight!r}")
        elif _is_valid_real_number(g_base.probe_weight):
            d_pw = abs(float(pg.probe_weight) - float(g_base.probe_weight))
            poison_field_diffs["probe_weight_diff"] = max(poison_field_diffs["probe_weight_diff"], d_pw)
            max_probe_poison_diff = max(max_probe_poison_diff, d_pw)
            if d_pw != 0.0:
                issues.append(f"Probe poisoning altered probe_weight by {d_pw:.3e} (must be exact 0.0)")

        if any(x is None for x in (p_hw, p_pred, p_var, p_llrs, p_mray, p_vray)):
            max_probe_poison_diff = math.inf
        elif not all(np.isfinite(x).all() for x in (p_hw, p_pred, p_var, p_llrs, p_mray, p_vray)):
            max_probe_poison_diff = math.inf
            issues.append("Probe poisoning produced non-finite GateOutput fields")
        elif base_arrays_ok:
            for fname, p_arr, b_arr, diff_key in (
                ("history_weight", p_hw, base_hw, "history_weight_diff"),
                ("mix_probe_pred", p_pred, base_pred, "mix_probe_pred_diff"),
                ("mix_probe_var", p_var, base_var, "mix_probe_var_diff"),
                ("mix_llrs", p_llrs, base_llrs, "mix_llrs_diff"),
                ("mix_mean_ray", p_mray, base_mray, "mix_mean_ray_diff"),
                ("mix_var_ray", p_vray, base_vray, "mix_var_ray_diff"),
            ):
                if p_arr.shape != b_arr.shape:
                    poison_field_diffs[diff_key] = math.inf
                    max_probe_poison_diff = math.inf
                    issues.append(f"Probe poisoning changed {fname} shape: {b_arr.shape} -> {p_arr.shape}")
                    continue
                d_val = float(np.max(np.abs(p_arr - b_arr)))
                poison_field_diffs[diff_key] = max(poison_field_diffs[diff_key], d_val)
                max_probe_poison_diff = max(max_probe_poison_diff, d_val)
                if d_val != 0.0:
                    issues.append(f"Probe poisoning altered {fname} by {d_val:.3e} (must be exact 0.0)")

    max_cstar_err = 0.0
    max_cstar_margin_err = 0.0
    for alpha in cstar_alphas:
        alpha_c = complex(alpha)
        alpha_sq = float(abs(alpha_c) ** 2)
        y_a_s = (alpha_c * obs.y_a).astype(np.complex128)
        y_b_s = (alpha_c * obs.y_b).astype(np.complex128)
        y_b_s[probe_row] = complex(math.nan, math.nan)
        obs_s = _clone_observation(obs, y_a=y_a_s, y_b=y_b_s, n0=alpha_sq * float(obs.n0))
        s_res = _coerce_res(assimilate_fn(obs_s), f"cstar_call(alpha={alpha_c})")
        if s_res is None:
            max_cstar_err = math.inf
            continue

        s_assim_raw = tuple(s_res.assimilated_rows)
        if (not all(_is_valid_int_seed(r) for r in s_assim_raw)) or ([int(r) for r in s_assim_raw] != base_assim_ints):
            max_cstar_err = math.inf
            issues.append(
                f"C* scaling (alpha={alpha_c}) changed assimilated_rows: {assim_tuple} -> {s_assim_raw}"
            )
        if (
            not isinstance(s_res.decoded_info, Mapping)
            or not all(_is_valid_int_seed(k) for k in s_res.decoded_info.keys())
            or {int(k) for k in s_res.decoded_info.keys()} != base_di_keys
        ):
            max_cstar_err = math.inf
            issues.append(f"C* scaling (alpha={alpha_c}) changed decoded_info keys or returned non-int keys")
        else:
            for r_key, b_base in base.decoded_info.items():
                s_bits, s_b_err = _validate_numeric_array(
                    s_res.decoded_info[r_key],
                    allow_complex=False,
                    field_name=f"cstar.decoded_info[{r_key!r}]",
                )
                if s_b_err is not None:
                    max_cstar_err = math.inf
                    issues.append(s_b_err)
                elif not np.array_equal(s_bits, np.asarray(b_base)):
                    max_cstar_err = math.inf
                    issues.append(f"C* scaling (alpha={alpha_c}) altered decoded_info bits on row {r_key}")

        if (
            not isinstance(s_res.row_weights, Mapping)
            or not all(_is_valid_int_seed(k) for k in s_res.row_weights.keys())
            or {int(k) for k in s_res.row_weights.keys()} != base_rw_keys
        ):
            max_cstar_err = math.inf
            issues.append(f"C* scaling (alpha={alpha_c}) changed row_weights keys or returned non-int keys")
        else:
            for r_key, w_base in base.row_weights.items():
                if not _is_valid_real_number(s_res.row_weights[r_key]) or not _is_valid_real_number(w_base):
                    max_cstar_err = math.inf
                    issues.append(f"C* scaling invalid row_weights[{r_key}]={s_res.row_weights[r_key]!r}")
                    continue
                w_err = abs(float(s_res.row_weights[r_key]) - float(w_base))
                max_cstar_err = max(max_cstar_err, w_err)
                if w_err > cstar_tol:
                    issues.append(f"C* scaling altered row_weights[{r_key}] by {w_err:.3e} > {cstar_tol:.3e}")

        if (
            isinstance(s_res.row_margins, Mapping)
            and all(_is_valid_int_seed(k) for k in s_res.row_margins.keys())
            and {int(k) for k in s_res.row_margins.keys()} == base_rm_keys
        ):
            for r_key, m_base in base.row_margins.items():
                if not _is_valid_real_number(s_res.row_margins[r_key]) or not _is_valid_real_number(m_base):
                    max_cstar_margin_err = math.inf
                    issues.append(f"C* scaling invalid row_margins[{r_key}]={s_res.row_margins[r_key]!r}")
                    continue
                m_err = abs(float(s_res.row_margins[r_key]) - float(m_base)) / max(1.0, abs(float(m_base)))
                max_cstar_margin_err = max(max_cstar_margin_err, m_err)
                if m_err > margin_tol:
                    issues.append(f"C* scaling altered row_margins[{r_key}] by {m_err:.3e} > {margin_tol:.3e}")
        else:
            max_cstar_margin_err = math.inf
            issues.append(f"C* scaling (alpha={alpha_c}) changed row_margins keys or returned non-int keys")

        sg = s_res.gate_out
        sg_llrs, e_sl = _validate_numeric_array(sg.mix_llrs, allow_complex=False, field_name="cstar.mix_llrs")
        sg_hw, e_sh = _validate_numeric_array(sg.history_weight, allow_complex=False, field_name="cstar.history_weight")
        sg_pred, e_sp = _validate_numeric_array(sg.mix_probe_pred, allow_complex=True, field_name="cstar.mix_probe_pred")
        sg_var, e_sv = _validate_numeric_array(sg.mix_probe_var, allow_complex=False, field_name="cstar.mix_probe_var")
        sg_mray, e_sm = _validate_numeric_array(sg.mix_mean_ray, allow_complex=True, field_name="cstar.mix_mean_ray")
        sg_vray, e_svr = _validate_numeric_array(sg.mix_var_ray, allow_complex=False, field_name="cstar.mix_var_ray")
        for err in (e_sl, e_sh, e_sp, e_sv, e_sm, e_svr):
            if err is not None:
                max_cstar_err = math.inf
                issues.append(err)
        if not _is_valid_real_number(sg.probe_weight):
            max_cstar_err = math.inf
            issues.append(f"C* scaling invalid probe_weight {sg.probe_weight!r}")
        elif (
            base_arrays_ok
            and _is_valid_real_number(g_base.probe_weight)
            and all(x is not None for x in (sg_llrs, sg_hw, sg_pred, sg_var, sg_mray, sg_vray))
        ):
            errs = {
                "mix_llrs": _scaled_diff(sg_llrs, base_llrs, 1.0),
                "history_weight": _scaled_diff(sg_hw, base_hw, 1.0),
                "probe_weight": abs(float(sg.probe_weight) - float(g_base.probe_weight)),
                "mix_probe_pred": _scaled_diff(sg_pred, base_pred, alpha_c),
                "mix_probe_var": _scaled_diff(sg_var, base_var, alpha_sq),
                "mix_mean_ray": _scaled_diff(sg_mray, base_mray, alpha_c),
                "mix_var_ray": _scaled_diff(sg_vray, base_vray, alpha_sq),
            }
            for fname, ferr in errs.items():
                max_cstar_err = max(max_cstar_err, ferr)
                if ferr > cstar_tol:
                    issues.append(f"C* invariance/equivariance failed on {fname}: {ferr:.3e} > {cstar_tol:.3e}")

    report = {
        "passed": len(issues) == 0,
        "issues": issues,
        "probe_excluded": bool(probe_excluded),
        "all_assimilated_rows_valid": bool(all_assimilated_rows_valid),
        "assimilated_rows": list(assim_tuple),
        "max_probe_poison_diff": float(max_probe_poison_diff),
        "poison_field_diffs": poison_field_diffs,
        "max_cstar_err": float(max_cstar_err),
        "cstar_tol": float(cstar_tol),
        "max_cstar_margin_err": float(max_cstar_margin_err),
        "cstar_margin_tol": float(margin_tol),
    }
    if strict and issues:
        raise ValueError(f"Assimilation anti-leakage audit failed: {issues}")
    return report


def _compare_isolated_outputs(
    base: Any,
    alt: Any,
    path: str,
    tol: float,
    issues: list[str],
    stats: dict[str, float],
) -> None:
    if isinstance(base, np.ndarray) or isinstance(alt, np.ndarray):
        b_arr = np.asarray(base)
        a_arr = np.asarray(alt)
        if b_arr.shape != a_arr.shape:
            issues.append(f"Target isolation shape mismatch at {path}: {b_arr.shape} vs {a_arr.shape}")
            stats["max_diff"] = math.inf
            return
        if b_arr.dtype == np.bool_ or a_arr.dtype == np.bool_:
            if not np.array_equal(b_arr, a_arr):
                issues.append(f"Target isolation bool mismatch at {path}")
                stats["max_diff"] = math.inf
            return
        if np.issubdtype(b_arr.dtype, np.number) and np.issubdtype(a_arr.dtype, np.number):
            if not (np.isfinite(b_arr).all() and np.isfinite(a_arr).all()):
                issues.append(f"Target isolation non-finite array at {path}")
                stats["max_diff"] = math.inf
                return
            diff = float(np.max(np.abs(b_arr - a_arr))) if b_arr.size else 0.0
            scale = max(1.0, float(np.max(np.abs(b_arr)))) if b_arr.size else 1.0
            rel = diff / scale
            stats["max_diff"] = max(stats["max_diff"], rel)
            if rel > tol:
                issues.append(f"Target isolation numeric drift at {path}: rel_diff={rel:.3e} > tol={tol:.3e}")
            return
        if not np.array_equal(b_arr, a_arr):
            issues.append(f"Target isolation array mismatch at {path}")
            stats["max_diff"] = math.inf
        return

    if is_dataclass(base) and is_dataclass(alt) and type(base) is type(alt):
        for f in dc_fields(base):
            _compare_isolated_outputs(
                getattr(base, f.name), getattr(alt, f.name), f"{path}.{f.name}", tol, issues, stats
            )
        return

    if isinstance(base, Mapping) and isinstance(alt, Mapping):
        if set(base.keys()) != set(alt.keys()):
            issues.append(f"Target isolation key mismatch at {path}: {set(base.keys()) ^ set(alt.keys())}")
            stats["max_diff"] = math.inf
            return
        for k in sorted(base.keys(), key=str):
            _compare_isolated_outputs(base[k], alt[k], f"{path}[{k!r}]", tol, issues, stats)
        return

    if isinstance(base, (list, tuple)) and isinstance(alt, (list, tuple)):
        if type(base) is not type(alt) or len(base) != len(alt):
            issues.append(f"Target isolation sequence mismatch at {path}")
            stats["max_diff"] = math.inf
            return
        for i, (bv, av) in enumerate(zip(base, alt)):
            _compare_isolated_outputs(bv, av, f"{path}[{i}]", tol, issues, stats)
        return

    if isinstance(base, (bool, np.bool_)) or isinstance(alt, (bool, np.bool_)):
        if not (isinstance(base, (bool, np.bool_)) and isinstance(alt, (bool, np.bool_)) and bool(base) == bool(alt)):
            issues.append(f"Target isolation bool mismatch at {path}: {base!r} vs {alt!r}")
            stats["max_diff"] = math.inf
        return

    if isinstance(base, (int, np.integer)) and isinstance(alt, (int, np.integer)):
        if int(base) != int(alt):
            issues.append(f"Target isolation int mismatch at {path}: {base!r} vs {alt!r}")
            stats["max_diff"] = math.inf
        return

    if isinstance(base, (float, complex, np.floating, np.complexfloating)) and isinstance(
        alt, (int, float, complex, np.number)
    ):
        diff = abs(complex(base) - complex(alt)) / max(1.0, abs(complex(base)))
        stats["max_diff"] = max(stats["max_diff"], diff)
        if diff > tol:
            issues.append(f"Target isolation scalar mismatch at {path}: rel_diff={diff:.3e} > tol={tol:.3e}")
        return

    if base != alt:
        issues.append(f"Target isolation value mismatch at {path}: {base!r} vs {alt!r}")
        stats["max_diff"] = math.inf


def audit_fold_target_isolation(
    fold_fn: Callable[[PublicObservation], Any],
    obs: PublicObservation,
    target_rows: Sequence[int],
    *,
    check_a_isolation: bool = False,
    tol: float = 1e-12,
    strict: bool = False,
) -> dict:
    """Verify fold-level target-row firewall: poisoning target_rows in y_b (and optionally y_a) leaves pre-target state invariant."""
    verify_p11_public_observation(obs, strict=True)
    t_rows = [int(r) for r in target_rows]
    if not t_rows:
        raise ValueError("target_rows must be non-empty")
    for r in t_rows:
        if r in obs.training_rows or r == obs.probe_row or not (0 <= r < T):
            raise ValueError(f"Invalid target row {r} relative to training={obs.training_rows}, probe={obs.probe_row}")

    base = fold_fn(obs)
    issues: list[str] = []
    stats = {"max_diff": 0.0}

    y_b_poison = obs.y_b.copy()
    y_b_poison[t_rows] = 2.125 - 0.75j
    obs_target_poison = _clone_observation(obs, y_a=obs.y_a.copy(), y_b=y_b_poison, n0=float(obs.n0))
    alt_b = fold_fn(obs_target_poison)
    _compare_isolated_outputs(base, alt_b, "target_poison", float(tol), issues, stats)

    if check_a_isolation:
        y_a_poison = np.full_like(obs.y_a, 50.0 - 150.0j)
        obs_a_poison = _clone_observation(obs, y_a=y_a_poison, y_b=obs.y_b.copy(), n0=float(obs.n0))
        alt_a = fold_fn(obs_a_poison)
        _compare_isolated_outputs(base, alt_a, "y_a_poison", float(tol), issues, stats)

    report = {
        "passed": len(issues) == 0,
        "target_rows": t_rows,
        "check_a_isolation": bool(check_a_isolation),
        "max_diff": float(stats["max_diff"]),
        "tol": float(tol),
        "issues": issues,
    }
    if strict and issues:
        raise ValueError(f"Fold target isolation check failed: {issues}")
    return report


def _as_exact_int_list(seeds: Iterable[Any], pool_name: str) -> list[int]:
    out: list[int] = []
    for item in seeds:
        if not _is_valid_int_seed(item):
            raise TypeError(
                f"Pool {pool_name!r} contains invalid seed (bool, timedelta64, datetime64, or non-int): "
                f"{item!r} ({type(item).__name__})"
            )
        out.append(int(item))
    return out


def check_seed_disjointness(
    train_seeds: Iterable[int],
    dev_seeds: Iterable[int],
    formal_id_seeds: Iterable[int],
    formal_ood_seeds: Iterable[int],
    *,
    extra_pools: Optional[Mapping[str, Iterable[int]]] = None,
    strict: bool = False,
) -> dict:
    """Verify strict pairwise disjointness and intra-pool uniqueness across seed pools."""
    pools_raw: dict[str, Iterable[int]] = {
        "train": train_seeds,
        "dev": dev_seeds,
        "formal_id": formal_id_seeds,
        "formal_ood": formal_ood_seeds,
    }
    if extra_pools is not None:
        for name, values in extra_pools.items():
            if name in pools_raw:
                raise ValueError(f"Duplicate extra pool name: {name!r}")
            pools_raw[name] = values

    pool_lists = {name: _as_exact_int_list(vals, name) for name, vals in pools_raw.items()}
    pool_sets: dict[str, set[int]] = {}
    intra_duplicates: dict[str, list[int]] = {}

    for name, items in pool_lists.items():
        seen: set[int] = set()
        dups: set[int] = set()
        for val in items:
            if val in seen:
                dups.add(val)
            seen.add(val)
        pool_sets[name] = seen
        if dups:
            intra_duplicates[name] = sorted(dups)

    names = list(pool_sets.keys())
    pairwise_overlaps: dict[str, list[int]] = {}
    total_overlap = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            inter = sorted(pool_sets[a_name] & pool_sets[b_name])
            if inter:
                pairwise_overlaps[f"{a_name}__{b_name}"] = inter
                total_overlap += len(inter)

    disjoint = (total_overlap == 0) and (len(intra_duplicates) == 0)
    report = {
        "disjoint": disjoint,
        "counts": {name: len(items) for name, items in pool_lists.items()},
        "unique_counts": {name: len(s) for name, s in pool_sets.items()},
        "intra_pool_duplicates": intra_duplicates,
        "pairwise_overlaps": pairwise_overlaps,
        "total_overlap_count": int(total_overlap),
    }
    if strict and not disjoint:
        raise ValueError(
            f"Seed disjointness violation: overlaps={pairwise_overlaps}, duplicates={intra_duplicates}"
        )
    return report


def default_historical_seed_pools() -> dict[str, list[int]]:
    """Canonical historical seed pools across P1..P10 (including 9e9d93b, EM-FloatFormer-p10, and p4 literal)."""
    p8c = [P8C_BASE_SEED + 1000 * c + j for c in range(1, 11) for j in range(32)]
    p8_manifold = (
        [P8_MANIFOLD_BASE_SEED]
        + [P8_MANIFOLD_BASE_SEED + 1000 * c + j for c in range(1, 11) for j in range(32)]
        + [P8_PREFIX_BASE_SEED]
        + [P8_PREFIX_BASE_SEED + 1000 * c + j for c in range(1, 11) for j in range(32)]
    )
    p9_train = [SEED_P9_TRAIN] + [
        SEED_P9_TRAIN + 100000 * (r_idx + 1) + i
        for r_idx in range(3)
        for i in range(280)
    ]
    p9_dev = (
        [SEED_P9_DEV]
        + [SEED_P9_DEV + 1000 * c + j for c in range(4) for j in range(16)]
        + [SEED_P9_DEV + 9000 + j for j in range(64)]
        + [SEED_P9_DEV + 200000 + 1000 * c + j for c in range(4) for j in range(16)]
        + [
            SEED_P9_DEV + 300000 + 10000 * (r_idx + 1) + 1000 * c + j
            for r_idx in range(6)
            for c in range(4)
            for j in range(12)
        ]
    )
    p9_formal_id = (
        [SEED_P9_FORMAL_ID]
        + [SEED_P9_FORMAL_ID + 1000 * c + j for c in range(1, 17) for j in range(32)]
        + [SEED_P9_FORMAL_ID + 200000 + 1000 * c + j for c in range(1, 17) for j in range(32)]
        + [SEED_P9_FORMAL_ID + 900000 + i for i in range(64)]
    )
    p9_formal_ood = (
        [SEED_P9_FORMAL_OOD]
        + [SEED_P9_FORMAL_OOD + 1000 * c + j for c in range(1, 17) for j in range(32)]
    )
    p10_9e9d93b = (
        [960_000_000_000]
        + [971_120_030_000 + i for i in range(128)]
        + [981_120_020_000 + i for i in range(128)]
        + [981_120_030_000 + i for i in range(128)]
        + [981_120_040_000 + i for i in range(128)]
    )
    p10_external = [
        1_010_000_000_000,
        1_020_000_000_000,
        1_030_000_000_000,
        1_040_000_000_000,
        1_050_000_000_000,
    ]
    return {
        "p8c": sorted(set(p8c)),
        "p8_manifold": sorted(set(p8_manifold)),
        "p9_train": sorted(set(p9_train)),
        "p9_dev": sorted(set(p9_dev)),
        "p9_formal_id": sorted(set(p9_formal_id)),
        "p9_formal_ood": sorted(set(p9_formal_ood)),
        "p10_9e9d93b": sorted(set(p10_9e9d93b)),
        "p10_external": sorted(set(p10_external)),
        "p4_stage3_link_genie_literal": [P4_STAGE3_LITERAL_SEED],
    }


def check_p11_seed_disjointness(
    p11_train_seeds: Iterable[int],
    p11_dev_seeds: Iterable[int],
    p11_formal_id_seeds: Iterable[int],
    p11_formal_ood_seeds: Iterable[int],
    p11_e2e_seeds: Iterable[int] = (),
    *,
    p8c_seeds: Optional[Iterable[int]] = None,
    p8_manifold_seeds: Optional[Iterable[int]] = None,
    p9_train_seeds: Optional[Iterable[int]] = None,
    p9_dev_seeds: Optional[Iterable[int]] = None,
    p9_formal_id_seeds: Optional[Iterable[int]] = None,
    p9_formal_ood_seeds: Optional[Iterable[int]] = None,
    p10_9e9d93b_seeds: Optional[Iterable[int]] = None,
    p10_external_seeds: Optional[Iterable[int]] = None,
    extra_pools: Optional[Mapping[str, Iterable[int]]] = None,
    check_namespace_bounds: bool = True,
    strict: bool = False,
) -> dict:
    """Verify strict pairwise disjointness across ALL 24 historical seed ranges AND all 5 P11 seed pools."""
    hist_defaults = default_historical_seed_pools()
    p11_train_list = _as_exact_int_list(p11_train_seeds, "p11_train")
    p11_dev_list = _as_exact_int_list(p11_dev_seeds, "p11_dev")
    p11_fid_list = _as_exact_int_list(p11_formal_id_seeds, "p11_formal_id")
    p11_food_list = _as_exact_int_list(p11_formal_ood_seeds, "p11_formal_ood")
    p11_e2e_list = _as_exact_int_list(p11_e2e_seeds, "p11_e2e")

    hist_pools: dict[str, Iterable[int]] = {
        "p11_e2e": p11_e2e_list,
        "p8c": hist_defaults["p8c"] if p8c_seeds is None else list(p8c_seeds),
        "p8_manifold": hist_defaults["p8_manifold"] if p8_manifold_seeds is None else list(p8_manifold_seeds),
        "p9_train": hist_defaults["p9_train"] if p9_train_seeds is None else list(p9_train_seeds),
        "p9_dev": hist_defaults["p9_dev"] if p9_dev_seeds is None else list(p9_dev_seeds),
        "p9_formal_id": hist_defaults["p9_formal_id"] if p9_formal_id_seeds is None else list(p9_formal_id_seeds),
        "p9_formal_ood": hist_defaults["p9_formal_ood"] if p9_formal_ood_seeds is None else list(p9_formal_ood_seeds),
        "p10_9e9d93b": hist_defaults["p10_9e9d93b"] if p10_9e9d93b_seeds is None else list(p10_9e9d93b_seeds),
        "p10_external": hist_defaults["p10_external"] if p10_external_seeds is None else list(p10_external_seeds),
        "p4_stage3_link_genie_literal": hist_defaults["p4_stage3_link_genie_literal"],
    }
    if extra_pools is not None:
        for name, vals in extra_pools.items():
            if name in hist_pools or name in ("train", "dev", "formal_id", "formal_ood"):
                raise ValueError(f"Duplicate pool name in extra_pools: {name!r}")
            hist_pools[name] = list(vals)

    base_report = check_seed_disjointness(
        p11_train_list,
        p11_dev_list,
        p11_fid_list,
        p11_food_list,
        extra_pools=hist_pools,
        strict=False,
    )

    namespace_collisions: list[dict] = []
    if check_namespace_bounds:
        p11_named = {
            "p11_train": p11_train_list,
            "p11_dev": p11_dev_list,
            "p11_formal_id": p11_fid_list,
            "p11_formal_ood": p11_food_list,
            "p11_e2e": p11_e2e_list,
        }
        for pool_name, seeds in p11_named.items():
            lo_exp, hi_exp = P11_SEED_RANGES[pool_name]
            for s_val in seeds:
                hist_hit = is_historical_seed(s_val)
                if hist_hit is not None:
                    lo, hi = HISTORICAL_SEED_RANGES[hist_hit]
                    namespace_collisions.append({
                        "p11_pool": pool_name,
                        "seed": s_val,
                        "historical_namespace": hist_hit,
                        "range": [lo, hi],
                    })
                elif not (lo_exp <= s_val < hi_exp):
                    namespace_collisions.append({
                        "p11_pool": pool_name,
                        "seed": s_val,
                        "historical_namespace": f"outside_{pool_name}",
                        "range": [lo_exp, hi_exp],
                    })

    disjoint = bool(base_report["disjoint"] and len(namespace_collisions) == 0)
    report = dict(base_report)
    report["disjoint"] = disjoint
    report["namespace_collisions"] = namespace_collisions
    report["p11_namespaces"] = {
        "SEED_P11_DEV": SEED_P11_DEV,
        "SEED_P11_TRAIN": SEED_P11_TRAIN,
        "SEED_P11_FORMAL_ID": SEED_P11_FORMAL_ID,
        "SEED_P11_FORMAL_OOD": SEED_P11_FORMAL_OOD,
        "SEED_P11_E2E": SEED_P11_E2E,
        "P11_NAMESPACE_SPAN": P11_NAMESPACE_SPAN,
    }
    if strict and not disjoint:
        raise ValueError(
            f"P11 seed disjointness violation: overlaps={report['pairwise_overlaps']}, "
            f"duplicates={report['intra_pool_duplicates']}, namespace_collisions={namespace_collisions}"
        )
    return report


def check_p11_no_hardcoded_n0(
    paths: Iterable[str | Path],
    *,
    strict: bool = False,
) -> dict:
    """Wrap p9.independent_ref.check_no_hardcoded_n0 to enforce zero hardcoded N0 across P11 modules."""
    report = check_no_hardcoded_n0(paths, strict=strict)
    return {
        **report,
        "p11_no_hardcoded_n0_passed": bool(report["clean"]),
    }


def _resolve_package_parts(path: Path) -> list[str]:
    """Infer Python package components for `path` (anchored at p11/p10/p9 if present)."""
    parts = list(path.resolve().parent.parts)
    for root_pkg in ("p11", "p10", "p9"):
        if root_pkg in parts:
            idx = len(parts) - 1 - parts[::-1].index(root_pkg)
            return parts[idx:]
    return [path.resolve().parent.name] if path.resolve().parent.name else []


def _extract_str_arg(call_node: ast.Call, pos: int, kw_name: str) -> Optional[str]:
    if len(call_node.args) > pos:
        arg = call_node.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in call_node.keywords:
        if kw.arg == kw_name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _resolve_dynamic_target(target: str, pkg_val: Optional[str], pkg_parts: list[str]) -> str:
    if not target.startswith("."):
        return target
    level = len(target) - len(target.lstrip("."))
    mod_tail = target[level:]
    if isinstance(pkg_val, str) and pkg_val:
        bits = pkg_val.rsplit(".", level - 1) if level > 1 else [pkg_val]
        base = bits[0] if len(bits) >= level else ""
        return f"{base}.{mod_tail}" if (base and mod_tail) else (base or mod_tail)
    pop_n = level - 1
    base_parts = pkg_parts[: max(0, len(pkg_parts) - pop_n)]
    base = ".".join(base_parts)
    return f"{base}.{mod_tail}" if (base and mod_tail) else (base or mod_tail)


def check_auditor_independence(
    paths: Iterable[str | Path],
    *,
    forbidden_modules: Iterable[str] = P11_FORBIDDEN_PEER_MODULES,
    strict: bool = False,
) -> dict:
    """AST static checker verifying P11-Audit modules NEVER import receiver/channel/ML peer implementations."""
    forbidden = frozenset(str(m) for m in forbidden_modules)
    violations: list[dict] = []
    checked: list[str] = []

    def _is_forbidden(mod_name: str) -> bool:
        if not mod_name:
            return False
        return mod_name in forbidden or any(mod_name.startswith(f + ".") for f in forbidden)

    for raw_path in paths:
        path = Path(raw_path)
        checked.append(str(path))
        pkg_parts = _resolve_package_parts(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        importlib_aliases: set[str] = {"importlib"}
        import_module_aliases: set[str] = {"import_module"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib":
                        importlib_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_aliases.add(alias.asname or alias.name)

        # Fixed-point pass for simple alias assignments: `il = importlib`, `im = il.import_module`, `im2 = im`
        assign_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        changed = True
        while changed:
            changed = False
            for node in assign_nodes:
                val = node.value
                is_il = isinstance(val, ast.Name) and val.id in importlib_aliases
                is_im = (
                    (isinstance(val, ast.Name) and val.id in import_module_aliases)
                    or (
                        isinstance(val, ast.Attribute)
                        and val.attr == "import_module"
                        and isinstance(val.value, ast.Name)
                        and val.value.id in importlib_aliases
                    )
                )
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        if is_il and tgt.id not in importlib_aliases:
                            importlib_aliases.add(tgt.id)
                            changed = True
                        elif is_im and tgt.id not in import_module_aliases:
                            import_module_aliases.add(tgt.id)
                            changed = True

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append({
                            "file": str(path),
                            "line": int(node.lineno),
                            "import": alias.name,
                        })
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level > 0:
                    pop_n = node.level - 1
                    base = pkg_parts[: max(0, len(pkg_parts) - pop_n)]
                    resolved = ".".join(base + ([mod] if mod else []))
                else:
                    resolved = mod
                if _is_forbidden(resolved):
                    violations.append({
                        "file": str(path),
                        "line": int(node.lineno),
                        "import": resolved,
                    })
                else:
                    for alias in node.names:
                        full = f"{resolved}.{alias.name}" if resolved else alias.name
                        if _is_forbidden(full):
                            violations.append({
                                "file": str(path),
                                "line": int(node.lineno),
                                "import": full,
                            })
            elif isinstance(node, ast.Call):
                is_import_module = False
                is_dunder_import = False
                if isinstance(node.func, ast.Name):
                    if node.func.id == "__import__":
                        is_dunder_import = True
                    elif node.func.id in import_module_aliases:
                        is_import_module = True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == "import_module":
                        is_import_module = True
                    elif node.func.attr == "__import__":
                        is_dunder_import = True

                if is_import_module or is_dunder_import:
                    target = _extract_str_arg(node, 0, "name")
                    if target is not None:
                        if is_import_module:
                            pkg_val = _extract_str_arg(node, 1, "package")
                            resolved = _resolve_dynamic_target(target, pkg_val, pkg_parts)
                            if _is_forbidden(resolved):
                                violations.append({
                                    "file": str(path),
                                    "line": int(node.lineno),
                                    "import": resolved,
                                })
                        else:
                            if _is_forbidden(target):
                                violations.append({
                                    "file": str(path),
                                    "line": int(node.lineno),
                                    "import": target,
                                })
                            else:
                                fromlist_node = node.args[3] if len(node.args) > 3 else None
                                if fromlist_node is None:
                                    for kw in node.keywords:
                                        if kw.arg == "fromlist":
                                            fromlist_node = kw.value
                                            break
                                if isinstance(fromlist_node, (ast.Tuple, ast.List)):
                                    for elt in fromlist_node.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            sub = f"{target}.{elt.value}"
                                            if _is_forbidden(sub):
                                                violations.append({
                                                    "file": str(path),
                                                    "line": int(node.lineno),
                                                    "import": sub,
                                                })

    report = {
        "independent": len(violations) == 0,
        "files_checked": checked,
        "forbidden_modules": sorted(forbidden),
        "violations": violations,
    }
    if strict and violations:
        raise ValueError(f"Auditor independence violation detected: {violations}")
    return report


__all__ = [
    "ALLOWED_OFFGRID_ATTRS",
    "FORBIDDEN_OFFGRID_HIDDEN_ATTRS",
    "HISTORICAL_SEED_RANGES",
    "OFFGRID_DELAY_MAX",
    "OFFGRID_DELAY_MIN",
    "OffGridPropagationResult",
    "P4_STAGE3_LITERAL_SEED",
    "P8C_BASE_SEED",
    "P8_MANIFOLD_BASE_SEED",
    "P8_PREFIX_BASE_SEED",
    "P11_FORBIDDEN_PEER_MODULES",
    "PiecewiseDopplerShift",
    "audit_assimilation_anti_leakage",
    "audit_fold_target_isolation",
    "check_assimilation_result_adapter",
    "check_auditor_independence",
    "check_p11_no_hardcoded_n0",
    "check_p11_seed_disjointness",
    "check_seed_disjointness",
    "default_historical_seed_pools",
    "dirichlet_fractional_delay_matrix",
    "dirichlet_kernel_closed_form",
    "independent_complex_gaussian_loglik",
    "independent_offgrid_propagate_ab",
    "independent_subset_bayes_posteriors",
    "offgrid_phase_and_gain",
    "ray_predictive_moments",
    "row_boundary_time",
    "verify_fractional_delay_operator",
    "verify_offgrid_observation",
    "verify_offgrid_phase_continuity",
    "verify_p11_public_observation",
]
