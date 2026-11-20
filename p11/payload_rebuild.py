"""Phase 11 Feature F4: C*-Equivariant Second-Order Payload Covariance Reconstruction.

Implements the mathematically verified P11 primary-axis reset estimator and persistent
three-branch dynamics in `p11/math_verification.md`:
  - Full C* = R_+ x U(1) B-only canonical gauge (`c_B = sqrt(n0) * u_B`, zero physical
    additive floor, all-zero anchor edge handling with `u_B = 1 + 0j`).
  - Dimensionless log-uniform scale quadrature (`rho = s / n0`) marginalized inside exact
    complete log-evidence `L_{j, rho}` (including `-Q*log(pi)` and large-kappa Bessel
    asymptotic fallback for `v0 = rho * P0 * (1 - A) * (1 + A)`).
  - GL3 (`ORDER=3`, 18/ray, 5,832 joint nodes) with explicit signed-frequency masks,
    coarse FFT screening, stratified mode-preserving compression (`J <= 160` with
    `Q_g * n_j / M_g` weighting and full proposal normalizer `S_R`), and single-Cholesky
    parity-0/1 `16x16` block covariance scoring.
  - Persistent `ResetBase` / `JointResetState` with exact `fit_known_blocks_by_node` and
    `refit_reset(base, virtual_ledger)` via `2x2` Woodbury/Cholesky, cancellation-free
    within+between ray moments, joint cross-ray probe quadratic (`v_p >= n0`), and
    exact epoch-30 sign-preserving bilinear projection to `p8.tracker.State [3, 128]`.
  - Persistent `ShiftedBranchDynamics` (`M_a(t) = exp(2*pi*i * 1_{a=a*} * delta * (t - tau_b)_+)`)
    across pilot scoring, auxiliary scoring, weighted conditioning, and readout, plus
    comparable common-unit 3-branch scoring `(B_C, B_S, B_R)` with parabolic delta refinement.

Strict invariants:
  - ZERO hardcoded N0 (no `p8.link.N0`, `p8.tracker.N0`, `p8c.receiver`, or 9 dB literals).
  - ZERO access to `PROBE_ROW` (`y_b[19]`), `y_a`, or target rows inside `estimate_payload_posterior_raw`.
  - Pure `float64` / `complex128` precision; zero `torch.inference_mode()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping, Optional, Sequence

from numba import njit
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import i0e, i1e, logsumexp

from p8.link import EVEN, clock
from p8.tracker import CHIRP, FREQ, State, advance, evolution, observed_taps, prior
from p8c.link import PILOTS as SC_PILOTS
from p9.channel import DELAYED_INDEX, SAMPLES, SWEEPS, update_ofdm_state
from p9.contract import (
    CORE_ROWS,
    DELAYS,
    FRAME_DURATION,
    LINK_DURATION,
    N,
    OFDM_PILOT_ROWS,
    POWERS,
    PROBE_ROW,
    T,
    PublicObservation,
)
from p11.contract import ANCHOR, COMPLEX_DTYPE, REAL_DTYPE


# ---------------------------------------------------------------------------
# Frozen Geometry Constants (no N0, no observation data)
# ---------------------------------------------------------------------------

GL3_ORDER = 3
FINAL_NODES = 160
ABS_TIME: np.ndarray = np.asarray(clock(T, start_row=T), dtype=REAL_DTYPE)
ABS_TIME.setflags(write=False)
ORIGIN: float = float(ABS_TIME[ANCHOR].mean())  # 45.984375
U_REL: np.ndarray = np.asarray(ABS_TIME - ORIGIN, dtype=REAL_DTYPE)
U_REL.setflags(write=False)

IND: np.ndarray = np.arange(N, dtype=np.int64)
IND.setflags(write=False)
PATHS: np.ndarray = np.arange(len(DELAYS), dtype=np.int64)
PATHS.setflags(write=False)
OUT: np.ndarray = np.asarray((IND[None, :] + DELAYS[:, None]) % N, dtype=np.int64)
OUT.setflags(write=False)

_TRACKER_MAGS: np.ndarray = np.linspace(0.045, 0.080, 8, dtype=REAL_DTYPE)
_TRACKER_MAGS.setflags(write=False)
_SINC_MID_SQ: float = float(np.sinc(0.0625) ** 2)
_ICI_UNIT_POWER: float = float(1.0 - _SINC_MID_SQ)

# Frozen shift hypothesis grid (symmetric ID + OOD deltas, boundary b=0 & midframe b=6)
DELTA_STEP: float = 0.010
DELTAS_ID: tuple[float, ...] = (0.025, 0.035, 0.045, 0.055, 0.065, 0.075)
DELTAS_OOD: tuple[float, ...] = (0.085, 0.095, 0.105, 0.115, 0.125, 0.135)
DELTAS_POS: tuple[float, ...] = DELTAS_ID + DELTAS_OOD
DELTAS: tuple[float, ...] = tuple(-d for d in reversed(DELTAS_POS)) + DELTAS_POS
N_DELTAS: int = len(DELTAS)
N_RAYS: int = len(DELAYS)
ONSET_ROWS: tuple[int, ...] = (0, 6)
DEFAULT_SCALE_QUAD_ORDER: int = 49
DEFAULT_SCALE_RHO_BOUNDS: tuple[float, float] = (0.25, 64.0)


# ---------------------------------------------------------------------------
# 1. GL3 Bank, Epoch-30 Projection Kernel & Auxiliary Geometry Caches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GL3Bank:
    nodes: np.ndarray            # [5832, 3] int64
    rot: np.ndarray              # [18, T, N] complex128 centered at ORIGIN
    log_gl_prior: np.ndarray     # [5832] float64
    start_freq: np.ndarray       # [18] float64 global freq at t=0
    end_freq: np.ndarray         # [18] float64 global freq at t=LINK_DURATION
    slope: np.ndarray            # [18] float64 chirp rate
    local_freq_o: np.ndarray     # [18] float64 local freq at ORIGIN
    ray_sign: np.ndarray         # [18] bool (True if positive freq)
    sign_ray0: np.ndarray        # [5832] bool (True if Ray-0 positive freq)
    strata_4: np.ndarray         # [5832] int64 in {0, 1, 2, 3}


@lru_cache(maxsize=1)
def gl3_frequency_bank() -> GL3Bank:
    """Construct the 18-trajectory/ray, 5832-joint-node GL3 bank with explicit sign masks."""
    g, w = np.polynomial.legendre.leggauss(GL3_ORDER)
    values = 0.0625 + 0.0175 * g
    v, e, sign = np.meshgrid(values, values, (-1.0, 1.0), indexing="ij")
    start = np.asarray((v * sign).ravel(), dtype=REAL_DTYPE)
    end = np.asarray((e * sign).ravel(), dtype=REAL_DTYPE)
    wa, wb, _ = np.meshgrid(w / 2.0, w / 2.0, (-1.0, 1.0), indexing="ij")
    prior_1d = np.asarray((wa * wb * 0.5).ravel(), dtype=REAL_DTYPE)

    slope = (end - start) / float(LINK_DURATION)
    local_o = start + slope * ORIGIN
    rot = np.exp(
        2j * np.pi * (local_o[:, None, None] * U_REL[None, :, :] + 0.5 * slope[:, None, None] * (U_REL[None, :, :] ** 2))
    ).astype(COMPLEX_DTYPE)

    n_1d = len(local_o)
    nodes = np.stack(np.meshgrid(*(np.arange(n_1d, dtype=np.int64),) * N_RAYS, indexing="ij"), -1).reshape(-1, N_RAYS)
    log_gl_prior = np.log(prior_1d[nodes]).sum(axis=1).astype(REAL_DTYPE)

    ray_sign = np.asarray(start > 0.0, dtype=bool)
    sign_ray0 = np.asarray(ray_sign[nodes[:, 0]], dtype=bool)
    mag_hi_ray0 = np.asarray(np.abs(local_o[nodes[:, 0]]) >= 0.0625, dtype=bool)
    strata_4 = (sign_ray0.astype(np.int64) * 2) + mag_hi_ray0.astype(np.int64)

    for arr in (nodes, rot, log_gl_prior, start, end, slope, local_o, ray_sign, sign_ray0, strata_4):
        arr.setflags(write=False)

    return GL3Bank(
        nodes=nodes,
        rot=rot,
        log_gl_prior=log_gl_prior,
        start_freq=start,
        end_freq=end,
        slope=slope,
        local_freq_o=local_o,
        ray_sign=ray_sign,
        sign_ray0=sign_ray0,
        strata_4=strata_4,
    )


@lru_cache(maxsize=1)
def epoch30_projection_kernel() -> tuple[np.ndarray, np.ndarray]:
    """Return `(d_transport [18] complex128, K [18, 128] float64)` for exact epoch-30 export.

    Maps global signed `(start_freq, end_freq)` via sign-preserving bilinear interpolation
    onto the surrounding 2 start x 2 end magnitudes of `p8.tracker` `[128]`.
    """
    bank = gl3_frequency_bank()
    dt = float(FRAME_DURATION) - ORIGIN  # 30.0 - 45.984375 = -15.984375
    phase_delta = (bank.start_freq + bank.slope * ORIGIN) * dt + 0.5 * bank.slope * (dt ** 2)
    d_transport = np.exp(2j * np.pi * phase_delta).astype(COMPLEX_DTYPE)

    n_grid = len(FREQ)
    k_mat = np.zeros((len(bank.start_freq), n_grid), dtype=REAL_DTYPE)
    n_mags = len(_TRACKER_MAGS)  # 8

    for k_idx in range(len(bank.start_freq)):
        s_val = float(bank.start_freq[k_idx])
        e_val = float(bank.end_freq[k_idx])
        is_pos = s_val >= 0.0
        half_offset = 0 if is_pos else (n_grid // 2)
        s_mag = abs(s_val)
        e_mag = abs(e_val)

        i0 = int(np.clip(np.searchsorted(_TRACKER_MAGS, s_mag) - 1, 0, n_mags - 2))
        i1 = i0 + 1
        a_s = (s_mag - float(_TRACKER_MAGS[i0])) / float(_TRACKER_MAGS[i1] - _TRACKER_MAGS[i0])
        a_s = float(np.clip(a_s, 0.0, 1.0))

        j0 = int(np.clip(np.searchsorted(_TRACKER_MAGS, e_mag) - 1, 0, n_mags - 2))
        j1 = j0 + 1
        a_e = (e_mag - float(_TRACKER_MAGS[j0])) / float(_TRACKER_MAGS[j1] - _TRACKER_MAGS[j0])
        a_e = float(np.clip(a_e, 0.0, 1.0))

        k_mat[k_idx, half_offset + i0 * n_mags + j0] += (1.0 - a_s) * (1.0 - a_e)
        k_mat[k_idx, half_offset + i0 * n_mags + j1] += (1.0 - a_s) * a_e
        k_mat[k_idx, half_offset + i1 * n_mags + j0] += a_s * (1.0 - a_e)
        k_mat[k_idx, half_offset + i1 * n_mags + j1] += a_s * a_e

    d_transport.setflags(write=False)
    k_mat.setflags(write=False)
    return d_transport, k_mat


@lru_cache(maxsize=8)
def frequency_geometry_cache(rows: tuple[int, ...]) -> dict[tuple[int, int], np.ndarray]:
    """Cached exact FFT-diagonal pair Gram matrices `[18, 18, len(rows)]` for `0 <= b < a < 3`."""
    bank = gl3_frequency_bank()
    rot_sub = bank.rot[:, list(rows), :]
    pair_gram: dict[tuple[int, int], np.ndarray] = {}
    for a in range(N_RAYS):
        ra = rot_sub[:, :, OUT[a]]
        for b in range(a):
            rb = rot_sub[:, :, OUT[b]]
            pg = np.einsum("hrm,krm->hkr", ra, rb.conj()) / float(N)
            pg.setflags(write=False)
            pair_gram[(a, b)] = pg
    return pair_gram


# ---------------------------------------------------------------------------
# 2. B-Only Canonical Gauge, Dimensionless Scale Quadrature & Bessel Moments
# ---------------------------------------------------------------------------


def _strict_row_int(val: Any, name: str = "row") -> int:
    """Strictly accept only non-bool exact integers for row/index/budget parameters."""
    if isinstance(val, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(val, (int, np.integer)):
        raise TypeError(f"{name} must be an exact integer (not bool, float, or timedelta64), got {type(val).__name__}: {val!r}")
    return int(val)


def _validate_positive_n0(n0: Any, *, name: str = "n0") -> float:
    """Strictly validate that `n0` is a finite positive real scalar (rejecting bool, timedelta64, complex)."""
    if isinstance(n0, (bool, np.bool_, np.timedelta64, np.datetime64, complex, np.complexfloating, str, bytes)):
        raise TypeError(f"{name} must be a finite positive real number, got {type(n0).__name__}")
    if isinstance(n0, np.ndarray):
        if n0.ndim != 0 or n0.dtype.kind not in ("i", "u", "f"):
            raise TypeError(f"{name} must be a real scalar, got ndarray dtype={n0.dtype} shape={n0.shape}")
    elif not isinstance(n0, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite positive real number, got {type(n0).__name__}")
    val = float(n0)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {n0!r}")
    return val


def b_only_canonical_gauge(y10: np.ndarray, n0: float) -> tuple[complex, np.ndarray, float]:
    """Compute B-only C*-equivariant canonical gauge `c_B = sqrt(n0) * u_B` without additive floor.

    Uses only `y10` (Frame B anchor row) and `n0`. Never reads `y_a`, targets, or `PROBE_ROW`.
    At an all-zero anchor, uses `u_B = 1.0 + 0.0j` so `y_c = 0` and equivariance holds exactly.
    """
    n0_val = _validate_positive_n0(n0, name="n0")
    y_arr = np.asarray(y10, dtype=COMPLEX_DTYPE).ravel()
    if y_arr.shape != (N,) or not np.isfinite(y_arr).all():
        raise ValueError(f"y10 must be a finite complex128 vector of shape ({N},), got {y_arr.shape}")

    mags = np.abs(y_arr)
    idx_max = int(np.argmax(mags))
    if float(mags[idx_max]) == 0.0:
        u_b = 1.0 + 0.0j
    else:
        vec = complex(y_arr[idx_max] * np.conj(SC_PILOTS[ANCHOR, idx_max]))
        mag_vec = abs(vec)
        u_b = (vec / mag_vec) if mag_vec > 0.0 else (1.0 + 0.0j)

    c_b = complex(math.sqrt(n0_val) * u_b)
    y_c = np.asarray(y_arr / c_b, dtype=COMPLEX_DTYPE)
    return c_b, y_c, 1.0


@lru_cache(maxsize=16)
def default_scale_quadrature(
    n_pts: int = DEFAULT_SCALE_QUAD_ORDER,
    rho_min: float = DEFAULT_SCALE_RHO_BOUNDS[0],
    rho_max: float = DEFAULT_SCALE_RHO_BOUNDS[1],
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen log-uniform Gauss-Legendre quadrature `(rho_l, log_eta_l)` for `rho = s / n0`."""
    n_int = int(n_pts)
    if n_int < 1:
        raise ValueError(f"n_pts must be >= 1, got {n_pts}")
    if rho_min <= 0.0 or rho_max <= rho_min:
        raise ValueError(f"Invalid rho bounds: ({rho_min}, {rho_max})")
    g, w = np.polynomial.legendre.leggauss(n_int)
    lo = math.log(float(rho_min))
    hi = math.log(float(rho_max))
    u = 0.5 * (hi - lo) * g + 0.5 * (hi + lo)
    rho = np.exp(u).astype(REAL_DTYPE)
    log_eta = np.log(0.5 * w).astype(REAL_DTYPE)
    rho.setflags(write=False)
    log_eta.setflags(write=False)
    return rho, log_eta


def _bessel_ratio_and_one_minus_sq(kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute `A(kappa) = I1(kappa)/I0(kappa)` and `(1 - A(kappa)^2)` without cancellation."""
    k = np.asarray(kappa, dtype=REAL_DTYPE)
    a_val = np.zeros_like(k, dtype=REAL_DTYPE)
    one_minus_a2 = np.ones_like(k, dtype=REAL_DTYPE)

    nz = k > 0.0
    small = nz & (k < 1.0e7)
    large = k >= 1.0e7

    if np.any(small):
        ks = k[small]
        ratio = i1e(ks) / i0e(ks)
        a_val[small] = ratio
        one_minus_a2[small] = np.maximum((1.0 - ratio) * (1.0 + ratio), 0.0)

    if np.any(large):
        kl = k[large]
        inv_k = 1.0 / kl
        # DLMF 10.40.1 asymptotic expansion of 1 - I1(k)/I0(k)
        one_minus_a = inv_k * (0.5 + inv_k * (0.125 + inv_k * (0.125 + 0.1953125 * inv_k)))
        ratio_l = 1.0 - one_minus_a
        a_val[large] = ratio_l
        one_minus_a2[large] = np.maximum(one_minus_a * (1.0 + ratio_l), 0.0)

    return a_val, one_minus_a2


# ---------------------------------------------------------------------------
# 3. Exact Multi-Block Node Fitter (`fit_known_blocks_by_node`)
# ---------------------------------------------------------------------------


@njit
def _compute_scale_quad_terms_numba(
    rho_arr: np.ndarray,
    g11: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    d0_w_d0: np.ndarray,
    d0_w_y: np.ndarray,
    v0_r: np.ndarray,
    v0_i: np.ndarray,
    vy_r: np.ndarray,
    vy_i: np.ndarray,
    y_wy: float,
    logdet_w_inv: float,
    const_q_logpi: float,
    p0: float,
    p1: float,
    p2: float,
):
    """Exact 2x2 Cholesky + Woodbury solves over (L, J) in L1-friendly Numba loops."""
    l_cnt = rho_arr.shape[0]
    j_cnt = g11.shape[0]

    kappa = np.empty((l_cnt, j_cnt), dtype=np.float64)
    phase = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    c0 = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    c1 = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    bw0 = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    bw1 = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    cw_00 = np.empty((l_cnt, j_cnt), dtype=np.float64)
    cw_10 = np.empty((l_cnt, j_cnt), dtype=np.complex128)
    cw_11 = np.empty((l_cnt, j_cnt), dtype=np.float64)
    log_base = np.empty((l_cnt, j_cnt), dtype=np.float64)

    for l in range(l_cnt):
        rho = rho_arr[l]
        inv_rp1 = 1.0 / (rho * p1)
        inv_rp2 = 1.0 / (rho * p2)
        rp0 = rho * p0
        a_dom = math.sqrt(rp0)
        logdet_prefix = logdet_w_inv + math.log(rho * p1) + math.log(rho * p2)

        for j in range(j_cnt):
            m00 = g11[j] + inv_rp1
            l00 = math.sqrt(m00 if m00 > 1e-300 else 1e-300)
            g12c = g12[j].conjugate()
            l10 = g12c / l00
            m11 = (g22[j] + inv_rp2) - (l10.real * l10.real + l10.imag * l10.imag)
            l11 = math.sqrt(m11 if m11 > 1e-300 else 1e-300)

            logdet_r = logdet_prefix + 2.0 * (math.log(l00) + math.log(l11))

            zy0 = vy_r[j] / l00
            zy1 = (vy_i[j] - l10 * zy0) / l11
            z00 = v0_r[j] / l00
            z01 = (v0_i[j] - l10 * z00) / l11

            y_r_inv_y = y_wy - (zy0.real * zy0.real + zy0.imag * zy0.imag + zy1.real * zy1.real + zy1.imag * zy1.imag)
            d0_r_inv_d0 = d0_w_d0[j] - (z00.real * z00.real + z00.imag * z00.imag + z01.real * z01.real + z01.imag * z01.imag)
            if d0_r_inv_d0 < 0.0:
                d0_r_inv_d0 = 0.0
            b_s = d0_w_y[j] - (z00.conjugate() * zy0 + z01.conjugate() * zy1)

            c1_v = z01 / l11
            c0_v = (z00 - l10.conjugate() * c1_v) / l00
            bw1_v = zy1 / l11
            bw0_v = (zy0 - l10.conjugate() * bw1_v) / l00

            inv_l11 = 1.0 / l11
            inv_l00 = 1.0 / l00
            l_inv_10 = -l10 * (inv_l00 * inv_l11)

            c0[l, j] = c0_v
            c1[l, j] = c1_v
            bw0[l, j] = bw0_v
            bw1[l, j] = bw1_v
            cw_00[l, j] = inv_l00 * inv_l00 + (l_inv_10.real * l_inv_10.real + l_inv_10.imag * l_inv_10.imag)
            cw_10[l, j] = inv_l11 * l_inv_10
            cw_11[l, j] = inv_l11 * inv_l11

            abs_b = abs(b_s)
            kappa[l, j] = 2.0 * a_dom * abs_b
            phase[l, j] = (b_s / abs_b) if abs_b > 0.0 else (1.0 + 0.0j)
            log_base[l, j] = const_q_logpi - logdet_r - y_r_inv_y - rp0 * d0_r_inv_d0

    return kappa, phase, c0, c1, bw0, bw1, cw_00, cw_10, cw_11, log_base


@njit
def _marginalize_scale_moments_numba(
    w_scale: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    v0: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    cw_00: np.ndarray,
    cw_10: np.ndarray,
    cw_11: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate marginalized mean [J, 3] and within+between cov [J, 3, 3] in a single pass."""
    l_cnt, j_cnt = w_scale.shape
    mean_out = np.zeros((j_cnt, 3), dtype=np.complex128)
    cov_out = np.zeros((j_cnt, 3, 3), dtype=np.complex128)

    for j in range(j_cnt):
        mu0 = 0.0 + 0.0j
        mu1 = 0.0 + 0.0j
        mu2 = 0.0 + 0.0j
        for l in range(l_cnt):
            w = w_scale[l, j]
            mu0 += w * m0[l, j]
            mu1 += w * m1[l, j]
            mu2 += w * m2[l, j]
        mean_out[j, 0] = mu0
        mean_out[j, 1] = mu1
        mean_out[j, 2] = mu2

        c00_r = 0.0
        c11_r = 0.0
        c22_r = 0.0
        c10 = 0.0 + 0.0j
        c20 = 0.0 + 0.0j
        c21 = 0.0 + 0.0j

        for l in range(l_cnt):
            w = w_scale[l, j]
            wv = w * v0[l, j]
            cc0 = c0[l, j]
            cc1 = c1[l, j]
            d0 = m0[l, j] - mu0
            d1 = m1[l, j] - mu1
            d2 = m2[l, j] - mu2

            c00_r += wv + w * (d0.real * d0.real + d0.imag * d0.imag)
            c10 += -wv * cc0 + w * d1 * d0.conjugate()
            c20 += -wv * cc1 + w * d2 * d0.conjugate()
            c11_r += w * cw_00[l, j] + wv * (cc0.real * cc0.real + cc0.imag * cc0.imag) + w * (d1.real * d1.real + d1.imag * d1.imag)
            c21 += w * cw_10[l, j] + wv * cc1 * cc0.conjugate() + w * d2 * d1.conjugate()
            c22_r += w * cw_11[l, j] + wv * (cc1.real * cc1.real + cc1.imag * cc1.imag) + w * (d2.real * d2.real + d2.imag * d2.imag)

        cov_out[j, 0, 0] = c00_r + 0.0j
        cov_out[j, 1, 0] = c10
        cov_out[j, 0, 1] = c10.conjugate()
        cov_out[j, 2, 0] = c20
        cov_out[j, 0, 2] = c20.conjugate()
        cov_out[j, 1, 1] = c11_r + 0.0j
        cov_out[j, 2, 1] = c21
        cov_out[j, 1, 2] = c21.conjugate()
        cov_out[j, 2, 2] = c22_r + 0.0j

    return mean_out, cov_out


@dataclass(frozen=True)
class BlockFitResult:
    log_evidence: np.ndarray   # [J] float64
    mean: np.ndarray           # [J, 3] complex128
    cov: np.ndarray            # [J, 3, 3] complex128
    second: np.ndarray         # [J, 3, 3] complex128


def fit_known_blocks_by_node(
    node_ids: np.ndarray,
    rows: tuple[int, ...],
    words: np.ndarray,
    received: np.ndarray,
    noises: np.ndarray,
    scales: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> BlockFitResult:
    """Exact circle + 2-weak-Gaussian joint node fit on arbitrary known/virtual blocks.

    All inputs (`received`, `noises`) must be in consistent units (typically canonical `c_B`
    units where anchor noise is `1.0`). Evaluates complete log-evidence `L_{j, rho_l}`
    including `-Q * log(pi)` and `2x2` Woodbury/Cholesky, then marginalizes scale `rho`.
    """
    n_idx = np.asarray(node_ids, dtype=np.int64)
    if n_idx.ndim != 2 or n_idx.shape[1] != N_RAYS:
        raise ValueError(f"Expected node_ids [J, 3], got {n_idx.shape}")
    j_cnt = n_idx.shape[0]

    rows_t = tuple(_strict_row_int(r, "rows") for r in rows)
    if len(set(rows_t)) != len(rows_t) or any(r < 0 or r >= T or r == PROBE_ROW for r in rows_t):
        raise ValueError(f"Invalid known/virtual rows {rows_t} (must exclude PROBE_ROW={PROBE_ROW})")
    b_cnt = len(rows_t)
    if b_cnt == 0:
        raise ValueError("fit_known_blocks_by_node requires at least one known block")

    words_arr = np.asarray(words, dtype=COMPLEX_DTYPE)
    recv_arr = np.asarray(received, dtype=COMPLEX_DTYPE)
    noises_arr = np.asarray(noises, dtype=REAL_DTYPE).ravel()
    if words_arr.shape != (b_cnt, N) or recv_arr.shape != (b_cnt, N) or noises_arr.shape != (b_cnt,):
        raise ValueError(f"Shape mismatch for B={b_cnt}: words={words_arr.shape}, recv={recv_arr.shape}, noises={noises_arr.shape}")
    if not (np.isfinite(words_arr).all() and np.isfinite(recv_arr).all() and np.isfinite(noises_arr).all()):
        raise ValueError("Non-finite inputs to fit_known_blocks_by_node")
    if np.any(noises_arr <= 0.0):
        raise ValueError("All block noise variances must be strictly positive")

    if scales is None:
        rho_arr, log_eta = default_scale_quadrature()
    else:
        rho_arr = np.asarray(scales[0], dtype=REAL_DTYPE).ravel()
        log_eta = np.asarray(scales[1], dtype=REAL_DTYPE).ravel()
    l_cnt = len(rho_arr)

    bank = gl3_frequency_bank()
    q_dim = b_cnt * N
    rot_rows = bank.rot[:, list(rows_t), :].reshape(len(bank.local_freq_o), q_dim)  # [18, Q]

    y_flat = recv_arr.reshape(q_dim)
    w_flat = np.repeat(1.0 / noises_arr, N)  # [Q]
    wy_flat = w_flat * y_flat
    y_wy = float(np.vdot(y_flat, wy_flat).real)
    logdet_w_inv = float(N * np.sum(np.log(noises_arr)))
    const_q_logpi = float(-q_dim * math.log(math.pi))

    # Build [3, 18, Q] per-ray dictionaries for `rows_t`
    dicts = np.empty((N_RAYS, rot_rows.shape[0], q_dim), dtype=COMPLEX_DTYPE)
    for a in range(N_RAYS):
        del_a = np.roll(words_arr, int(DELAYS[a]), axis=-1).reshape(q_dim)
        dicts[a] = rot_rows * del_a[None, :]

    # Compute [3, 3, 18, 18] weighted Gram matrices and [3, 18] matched vectors
    d_wy_1d = np.empty((N_RAYS, rot_rows.shape[0]), dtype=COMPLEX_DTYPE)
    for a in range(N_RAYS):
        d_wy_1d[a] = dicts[a].conj() @ wy_flat

    d0_w_d0 = np.sum((dicts[0].real ** 2 + dicts[0].imag ** 2) * w_flat[None, :], axis=1)[n_idx[:, 0]]  # [J]
    d0_w_y = d_wy_1d[0, n_idx[:, 0]]  # [J]
    dw_w_y = np.column_stack([d_wy_1d[1, n_idx[:, 1]], d_wy_1d[2, n_idx[:, 2]]])  # [J, 2]

    g_w0_1 = (dicts[1].conj() * w_flat[None, :]) @ dicts[0].T
    g_w0_2 = (dicts[2].conj() * w_flat[None, :]) @ dicts[0].T
    dw_w_d0 = np.column_stack([g_w0_1[n_idx[:, 1], n_idx[:, 0]], g_w0_2[n_idx[:, 2], n_idx[:, 0]]])  # [J, 2]

    g_11 = (dicts[1].conj() * w_flat[None, :]) @ dicts[1].T
    g_12 = (dicts[1].conj() * w_flat[None, :]) @ dicts[2].T
    g_22 = (dicts[2].conj() * w_flat[None, :]) @ dicts[2].T

    g11 = g_11[n_idx[:, 1], n_idx[:, 1]].real
    g12 = g_12[n_idx[:, 1], n_idx[:, 2]]
    g22 = g_22[n_idx[:, 2], n_idx[:, 2]].real

    p0 = float(POWERS[0])
    p1 = float(POWERS[1])
    p2 = float(POWERS[2])

    kappa, phase, c0, c1, bw0, bw1, cw_00, cw_10, cw_11, log_base = _compute_scale_quad_terms_numba(
        np.ascontiguousarray(rho_arr, dtype=REAL_DTYPE),
        np.ascontiguousarray(g11, dtype=REAL_DTYPE),
        np.ascontiguousarray(g12, dtype=COMPLEX_DTYPE),
        np.ascontiguousarray(g22, dtype=REAL_DTYPE),
        np.ascontiguousarray(d0_w_d0, dtype=REAL_DTYPE),
        np.ascontiguousarray(d0_w_y, dtype=COMPLEX_DTYPE),
        np.ascontiguousarray(dw_w_d0[:, 0], dtype=COMPLEX_DTYPE),
        np.ascontiguousarray(dw_w_d0[:, 1], dtype=COMPLEX_DTYPE),
        np.ascontiguousarray(dw_w_y[:, 0], dtype=COMPLEX_DTYPE),
        np.ascontiguousarray(dw_w_y[:, 1], dtype=COMPLEX_DTYPE),
        y_wy,
        logdet_w_inv,
        const_q_logpi,
        p0,
        p1,
        p2,
    )

    a_dom = np.sqrt(rho_arr * p0)[:, None]
    ratio_a, one_minus_a2 = _bessel_ratio_and_one_minus_sq(kappa)
    m0 = a_dom * phase * ratio_a
    v0 = (rho_arr * p0)[:, None] * one_minus_a2

    m1 = bw0 - c0 * m0
    m2 = bw1 - c1 * m0

    log_i0 = kappa + np.log(i0e(kappa))
    log_l_rho = log_base + log_i0

    # Marginalize scale quadrature rho_l via fast Numba L1-cache accumulation (zero [L, J, 3] einsum)
    log_joint = log_eta[:, None] + log_l_rho  # [L, J]
    log_evidence = logsumexp(log_joint, axis=0).astype(REAL_DTYPE)  # [J]
    w_scale = np.exp(log_joint - log_evidence[None, :])  # [L, J]

    mean_out, cov_out = _marginalize_scale_moments_numba(
        w_scale, m0, m1, m2, v0, c0, c1, cw_00, cw_10, cw_11
    )
    second_out = mean_out[:, :, None] * mean_out[:, None, :].conj() + cov_out
    second_out = 0.5 * (second_out + second_out.conj().swapaxes(-1, -2))

    return BlockFitResult(
        log_evidence=log_evidence,
        mean=mean_out,
        cov=cov_out,
        second=second_out,
    )


# ---------------------------------------------------------------------------
# 4. Coarse Screening, Stratified Compression & Parity-0/1 16x16 Cholesky Score
# ---------------------------------------------------------------------------


def coarse_payload_score(
    y_aux_c: np.ndarray,
    aux_rows: tuple[int, ...],
    second: np.ndarray,
    n_c: float = 1.0,
) -> np.ndarray:
    """FFT-diagonal exact projection score `[5832]` including `-R * N * log(pi)`."""
    bank = gl3_frequency_bank()
    geom = frequency_geometry_cache(tuple(_strict_row_int(r, "aux_rows") for r in aux_rows))
    y_arr = np.asarray(y_aux_c, dtype=COMPLEX_DTYPE)
    n_val = float(n_c)

    energy = np.abs(np.fft.fft(y_arr, axis=-1, norm="ortho")) ** 2
    multiplicity = 2
    half = N // multiplicity
    energy_fold = energy[:, :half] + energy[:, half:]

    trace_sec = np.trace(second, axis1=1, axis2=2).real
    powers = np.empty((len(bank.nodes), len(aux_rows), half), dtype=REAL_DTYPE)
    powers[:] = (n_val + trace_sec)[:, None, None]

    ramp_base = np.arange(half, dtype=REAL_DTYPE) / float(N)
    for (a, b), g in geom.items():
        coef = second[:, a, b, None] * g[bank.nodes[:, a], bank.nodes[:, b]]
        ramp = np.exp(-2j * np.pi * float(DELAYS[a] - DELAYS[b]) * ramp_base)
        powers += 2.0 * (coef[:, :, None] * ramp[None, None, :]).real

    powers = np.maximum(powers, n_val)
    const_term = -float(len(aux_rows) * N) * math.log(math.pi)
    return const_term - (float(multiplicity) * np.log(powers) + energy_fold[None, :, :] / powers).sum(axis=(1, 2))


def stratified_mode_midpoints(
    log_proposal: np.ndarray,
    strata: np.ndarray,
    budget: int = FINAL_NODES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stratified systematic midpoint compression with exact `Q_g * n_j / M_g` mass weights.

    Returns `(idx [J], represented_mass [J], log_proposal_norm)`.
    """
    budget_i = _strict_row_int(budget, "budget")
    if budget_i < 1:
        raise ValueError(f"budget must be >= 1, got {budget_i}")

    lp = np.asarray(log_proposal, dtype=REAL_DTYPE).ravel()
    st = np.asarray(strata, dtype=np.int64).ravel()
    if lp.shape != st.shape or lp.size == 0:
        raise ValueError("log_proposal and strata must be non-empty with identical shape")
    if np.any(np.isnan(lp)) or np.any(lp == np.inf):
        raise ValueError("log_proposal must not contain NaN or +inf")

    log_norm = float(logsumexp(lp))
    if not math.isfinite(log_norm):
        probs = np.full(lp.shape, 1.0 / float(lp.size), dtype=REAL_DTYPE)
    else:
        probs = np.exp(lp - log_norm)
    unique_strata = np.unique(st)

    # Stratum masses Q_g
    q_by_s: dict[int, float] = {}
    idx_by_s: dict[int, np.ndarray] = {}
    for g in unique_strata:
        g_int = int(g)
        mask_idx = np.flatnonzero(st == g_int)
        idx_by_s[g_int] = mask_idx
        q_by_s[g_int] = float(np.sum(probs[mask_idx]))

    active_strata = [g for g in unique_strata if q_by_s[int(g)] > 0.0]
    if not active_strata:
        # Fallback if completely flat/zero
        active_strata = [int(unique_strata[0])]
        q_by_s[int(unique_strata[0])] = 1.0

    # Deterministic largest-remainder allocation summing to `budget_i`, reserving >= 1 node
    # for any active stratum with Q_g >= 1e-12 (>= 4 if Q_g >= 1e-4 and budget permits).
    m_alloc: dict[int, int] = {int(g): 0 for g in active_strata}
    protected = [int(g) for g in active_strata if q_by_s[int(g)] >= 1e-12]
    if not protected:
        protected = [int(max(active_strata, key=lambda g: q_by_s[int(g)]))]

    if len(protected) > budget_i:
        protected = sorted(
            protected,
            key=lambda g: (-float(np.max(lp[idx_by_s[g]])), -q_by_s[g], int(g)),
        )[:budget_i]

    for g in protected:
        m_alloc[g] = 4 if (q_by_s[g] >= 1e-4 and budget_i >= 4 * len(protected)) else 1

    rem = budget_i - sum(m_alloc.values())
    if rem > 0:
        q_sum_act = sum(q_by_s[int(g)] for g in active_strata)
        targets = {int(g): (rem * q_by_s[int(g)] / q_sum_act) for g in active_strata}
        floors = {g: int(math.floor(t)) for g, t in targets.items()}
        for g, fl in floors.items():
            m_alloc[g] += fl
        left = rem - sum(floors.values())
        if left > 0:
            order = sorted(active_strata, key=lambda g: (-(targets[int(g)] - floors[int(g)]), int(g)))
            for g in order[:left]:
                m_alloc[int(g)] += 1

    all_idx: list[np.ndarray] = []
    all_mass: list[np.ndarray] = []
    for g in sorted(m_alloc.keys()):
        m_g = m_alloc[g]
        if m_g <= 0:
            continue
        q_g = q_by_s[g]
        g_nodes = idx_by_s[g]
        p_cond = probs[g_nodes] / q_g
        cdf = np.cumsum(p_cond)
        cdf[-1] = 1.0
        u_pts = (np.arange(m_g, dtype=REAL_DTYPE) + 0.5) / float(m_g)
        local_hits, counts = np.unique(np.searchsorted(cdf, u_pts), return_counts=True)
        local_hits = np.minimum(local_hits, len(g_nodes) - 1)
        all_idx.append(g_nodes[local_hits])
        all_mass.append(q_g * counts.astype(REAL_DTYPE) / float(m_g))

    idx_out = np.concatenate(all_idx).astype(np.int64)
    mass_out = np.concatenate(all_mass).astype(REAL_DTYPE)
    order = np.argsort(idx_out)
    return idx_out[order], mass_out[order], log_norm


def parity_block_cov_score(
    y_aux_c: np.ndarray,
    aux_rows: tuple[int, ...],
    node_ids: np.ndarray,
    second_sub: np.ndarray,
    n_c: float = 1.0,
) -> np.ndarray:
    """Exact parity-0/1 `16x16` Hermitian Cholesky payload log-likelihood `[J]`.

    Includes the exact `-R * N * log(pi)` Gaussian density constant.
    """
    bank = gl3_frequency_bank()
    selected = np.asarray(node_ids, dtype=np.int64)
    rows_list = [_strict_row_int(r, "aux_rows") for r in aux_rows]
    j_cnt = selected.shape[0]
    r_cnt = len(rows_list)
    if r_cnt == 0:
        return np.zeros(j_cnt, dtype=REAL_DTYPE)

    y_arr = np.asarray(y_aux_c, dtype=COMPLEX_DTYPE)
    if y_arr.shape != (r_cnt, N):
        raise ValueError(f"Expected y_aux_c shape ({r_cnt}, {N}), got {y_arr.shape}")
    n_val = float(n_c)

    gains_rot = [bank.rot[selected[:, a]][:, rows_list, :][:, :, OUT[a]] for a in range(N_RAYS)]
    half = N // 2
    cov = np.zeros((j_cnt, r_cnt, 2, half, half), dtype=COMPLEX_DTYPE)
    for a in range(N_RAYS):
        for b in range(N_RAYS):
            cross = second_sub[:, a, b, None, None] * gains_rot[a] * gains_rot[b].conj()
            for parity in (0, 1):
                cov[:, :, parity, OUT[a, parity::2] // 2, OUT[b, parity::2] // 2] += cross[:, :, parity::2]

    diag_idx = np.arange(half, dtype=np.int64)
    cov[..., diag_idx, diag_idx] += n_val
    cov = 0.5 * (cov + cov.conj().swapaxes(-1, -2))

    # Single Hermitian Cholesky factorization per [J, R, 2] 16x16 block
    chol = np.linalg.cholesky(cov)
    diag_l = np.real(np.diagonal(chol, axis1=-2, axis2=-1))
    logdet = 2.0 * np.sum(np.log(np.maximum(diag_l, 1e-300)), axis=-1)

    z = np.stack([y_arr[:, ::2], y_arr[:, 1::2]], axis=1)  # [R, 2, 16]
    z_b = np.broadcast_to(z, cov.shape[:-1])
    whitened = np.linalg.solve(chol, z_b[..., None])[..., 0]
    quad = np.sum(whitened.real ** 2 + whitened.imag ** 2, axis=-1)

    const_half = -float(half) * math.log(math.pi)
    score = const_half - logdet - quad
    return score.sum(axis=(1, 2)).astype(REAL_DTYPE)


# ---------------------------------------------------------------------------
# 5. Persistent ResetBase / JointResetState, Refit & Moment/Probe/State Export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointResetState:
    """Mutable-across-waves joint GL3 posterior state (in canonical `c_B` units internally)."""
    gauge: complex
    n0: float
    node_ids: np.ndarray        # [J, 3] int64
    logw: np.ndarray            # [J] float64
    mean_c: np.ndarray          # [J, 3] complex128
    cov_c: np.ndarray           # [J, 3, 3] complex128
    sign_ray0: np.ndarray       # [J] bool


@dataclass(frozen=True)
class PayloadPosterior:
    """Persistent `ResetBase` + full Frame-B moment / `State [3, 128]` / probe outputs."""
    gauge: complex
    n0: float
    aux_rows: tuple[int, ...]
    use_payload: bool
    aux_temp: float
    node_ids: np.ndarray             # [J, 3] int64
    logw0: np.ndarray                # [J] float64
    anchor_log_evidence: np.ndarray  # [J] float64 (canon units)
    anchor_mean: np.ndarray          # [J, 3] complex128 (canon units)
    anchor_cov: np.ndarray           # [J, 3, 3] complex128 (canon units)
    anchor_second: np.ndarray        # [J, 3, 3] complex128 (canon units)
    scales: tuple[np.ndarray, np.ndarray]
    sign_ray0: np.ndarray            # [J] bool
    class_log_evidence: float        # physical units S_R
    class_log_evidence_canon: float  # canon units S_R
    pilot_loglik: float              # physical units pilot-only log-evidence
    mean: np.ndarray                 # [3, T, N] complex128 (physical units)
    var: np.ndarray                  # [3, T, N] float64 (physical units)
    state_clr_proj: State            # epoch=30.0, [3, 128] (physical units)
    probe_pred: np.ndarray           # [N] complex128 (physical units)
    probe_var: np.ndarray            # [N] float64 (physical units)

    @property
    def scale_alpha(self) -> complex:
        return complex(self.gauge)

    def to_joint_state(self) -> JointResetState:
        return JointResetState(
            gauge=complex(self.gauge),
            n0=float(self.n0),
            node_ids=self.node_ids.copy(),
            logw=self.logw0.copy(),
            mean_c=self.anchor_mean.copy(),
            cov_c=self.anchor_cov.copy(),
            sign_ray0=self.sign_ray0.copy(),
        )


ResetBase = PayloadPosterior


def project_joint_moments_canon(
    jstate: JointResetState,
    rows: Optional[tuple[int, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cancellation-free within+between ray moments `(mean_c [3, R, N], var_c [3, R, N])` in canonical `c_B` units."""
    bank = gl3_frequency_bank()
    row_list = list(range(T)) if rows is None else [_strict_row_int(r, "rows") for r in rows]
    r_cnt = len(row_list)
    w = np.exp(np.asarray(jstate.logw, dtype=REAL_DTYPE))  # [J]

    mean_c = np.empty((N_RAYS, r_cnt, N), dtype=COMPLEX_DTYPE)
    var_c = np.empty((N_RAYS, r_cnt, N), dtype=REAL_DTYPE)

    for a in range(N_RAYS):
        rot_ja = bank.rot[jstate.node_ids[:, a]][:, row_list, :]  # [J, R, N]
        m_ja = jstate.mean_c[:, a, None, None] * rot_ja          # [J, R, N]
        mu_a = np.einsum("j,jrn->rn", w, m_ja)                   # [R, N]
        diff = m_ja - mu_a[None, :, :]
        within_a = float(np.dot(w, np.real(jstate.cov_c[:, a, a])))
        between_a = np.einsum("j,jrn->rn", w, diff.real ** 2 + diff.imag ** 2)
        mean_c[a] = mu_a
        var_c[a] = np.maximum(within_a + between_a, 0.0)

    return mean_c, var_c


def project_joint_moments(
    jstate: JointResetState,
    rows: Optional[tuple[int, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cancellation-free within+between ray moments `(mean [3, R, N], var [3, R, N])` in physical units."""
    mean_c, var_c = project_joint_moments_canon(jstate, rows=rows)
    g = complex(jstate.gauge)
    g2 = float(abs(g) ** 2)
    return np.asarray(mean_c * g, dtype=COMPLEX_DTYPE), np.asarray(var_c * g2, dtype=REAL_DTYPE)


def project_joint_probe(
    jstate: JointResetState,
    probe_row: int = PROBE_ROW,
    tx_probe: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact joint probe quadratic `v_p = n0 + sum_j w_j [x^T Sigma_j x^* + |p_j - p|^2] >= n0`."""
    p_row = _strict_row_int(probe_row, "probe_row")
    tx = SC_PILOTS[p_row] if tx_probe is None else np.asarray(tx_probe, dtype=COMPLEX_DTYPE).ravel()
    if tx.shape != (N,):
        raise ValueError(f"tx_probe must have shape ({N},), got {tx.shape}")

    x = tx[DELAYED_INDEX]  # [3, N]
    bank = gl3_frequency_bank()
    rot_j = bank.rot[jstate.node_ids, p_row, :]  # [J, 3, N]
    m_j_t = jstate.mean_c[:, :, None] * rot_j    # [J, 3, N]
    sig_j_t = (
        jstate.cov_c[:, :, :, None]
        * rot_j[:, :, None, :]
        * rot_j[:, None, :, :].conj()
    )  # [J, 3, 3, N]

    w = np.exp(np.asarray(jstate.logw, dtype=REAL_DTYPE))
    p_j = np.einsum("an,jan->jn", x, m_j_t)       # [J, N]
    p_c = w @ p_j                                 # [N]

    quad_within = np.einsum("an,jabn,bn->jn", x, sig_j_t, x.conj()).real
    quad_within = np.maximum(quad_within, 0.0)
    diff_p = p_j - p_c[None, :]
    between = diff_p.real ** 2 + diff_p.imag ** 2
    v_p_c = 1.0 + (w @ (quad_within + between))

    g = complex(jstate.gauge)
    g2 = float(abs(g) ** 2)
    return np.asarray(p_c * g, dtype=COMPLEX_DTYPE), np.asarray(v_p_c * g2, dtype=REAL_DTYPE)


def project_to_tracker_state(jstate: JointResetState) -> State:
    """Exact epoch-30 transport + sign-preserving bilinear projection onto `p8.tracker.State [3, 128]`."""
    d_transport, k_mat = epoch30_projection_kernel()
    w = np.exp(np.asarray(jstate.logw, dtype=REAL_DTYPE))
    g = complex(jstate.gauge)
    g2 = float(abs(g) ** 2)

    n_grid = len(FREQ)
    lw_out = np.full((N_RAYS, n_grid), -np.inf, dtype=REAL_DTYPE)
    m_out = np.zeros((N_RAYS, n_grid), dtype=COMPLEX_DTYPE)
    v_out = np.empty((N_RAYS, n_grid), dtype=REAL_DTYPE)

    for a in range(N_RAYS):
        k_sub = k_mat[jstate.node_ids[:, a], :]  # [J, 128]
        d_a = d_transport[jstate.node_ids[:, a]]  # [J]
        m_e = jstate.mean_c[:, a] * d_a           # [J] (canon units)
        v_e = np.maximum(np.real(jstate.cov_c[:, a, a]), 0.0)  # [J] (|d_a| == 1)

        wk = w[:, None] * k_sub                   # [J, 128]
        q_h = wk.sum(axis=0)                      # [128]
        pos = q_h > 0.0

        filler_v = max(float(POWERS[a] * g2), float(jstate.n0) * float(POWERS[a]))
        v_out[a, :] = filler_v

        if np.any(pos):
            q_pos = q_h[pos]
            m_pos = (wk[:, pos].T @ m_e) / q_pos
            dm = m_e[:, None] - m_pos[None, :]
            sec_diff = v_e[:, None] + (dm.real ** 2 + dm.imag ** 2)
            v_pos = np.sum(wk[:, pos] * sec_diff, axis=0) / q_pos

            lw_pos = np.log(q_pos)
            lw_pos -= float(logsumexp(lw_pos))
            lw_out[a, pos] = lw_pos
            m_out[a, pos] = m_pos * g
            v_out[a, pos] = np.maximum(v_pos * g2, 1e-30 * float(jstate.n0))

    return State(
        epoch=float(FRAME_DURATION),
        log_weight=lw_out,
        mean=m_out,
        variance=v_out,
    )


def fit_anchor_only(
    y10: np.ndarray,
    n0: float,
    *,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
) -> tuple[complex, np.ndarray, float, BlockFitResult]:
    """Fit the `[5832]` GL3 joint bank on anchor `(10,)` only in canonical `c_B` units."""
    c_b, y10_c, n_c = b_only_canonical_gauge(y10, n0)
    scales = default_scale_quadrature(scale_quad_order)
    bank = gl3_frequency_bank()
    anchor_fit = fit_known_blocks_by_node(
        bank.nodes,
        rows=(ANCHOR,),
        words=SC_PILOTS[ANCHOR : ANCHOR + 1],
        received=y10_c[None, :],
        noises=np.array([n_c], dtype=REAL_DTYPE),
        scales=scales,
    )
    return c_b, y10_c, n_c, anchor_fit


def estimate_payload_posterior_raw(
    y10: np.ndarray,
    aux_rows: tuple[int, ...],
    y_aux: Optional[np.ndarray],
    n0: float,
    *,
    use_payload: bool = True,
    aux_temp: float = 1.0,
    budget: int = FINAL_NODES,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    precomputed_anchor: Optional[tuple[complex, np.ndarray, float, BlockFitResult]] = None,
) -> PayloadPosterior:
    """Strictly raw-array C*-equivariant GL3 reset posterior (zero access to `y_a`, targets, or `PROBE_ROW`)."""
    if precomputed_anchor is None:
        c_b, y10_c, n_c, anchor_fit = fit_anchor_only(y10, n0, scale_quad_order=scale_quad_order)
    else:
        c_b, y10_c, n_c, anchor_fit = precomputed_anchor
    n0_val = float(n0)
    scales = default_scale_quadrature(scale_quad_order)
    bank = gl3_frequency_bank()

    aux_t = tuple(_strict_row_int(r, "aux_rows") for r in aux_rows)
    if use_payload:
        if len(set(aux_t)) != 5 or len(aux_t) != 5 or any(
            r < 0 or r >= T or r in (ANCHOR, PROBE_ROW) for r in aux_t
        ):
            raise ValueError(f"Expected 5 distinct valid auxiliary payload rows excluding (10, 19), got {aux_t}")
        if y_aux is None:
            raise ValueError("y_aux must be provided when use_payload=True")
        y_aux_arr = np.asarray(y_aux, dtype=COMPLEX_DTYPE)
        if y_aux_arr.shape != (len(aux_t), N) or not np.isfinite(y_aux_arr).all():
            raise ValueError(f"Expected finite y_aux shape ({len(aux_t)}, {N}), got {y_aux_arr.shape}")
        y_aux_c = y_aux_arr / c_b
    else:
        if len(set(aux_t)) != len(aux_t) or any(
            r < 0 or r >= T or r in (ANCHOR, PROBE_ROW) for r in aux_t
        ):
            raise ValueError(f"Invalid auxiliary rows {aux_t} for pilot-only mode")
        y_aux_c = np.empty((0, N), dtype=COMPLEX_DTYPE)
    b_log = bank.log_gl_prior + anchor_fit.log_evidence  # [5832]
    pilot_ll_canon = float(logsumexp(b_log))
    pilot_ll_phys = pilot_ll_canon - float(N) * math.log(float(abs(c_b) ** 2))

    if use_payload:
        coarse = coarse_payload_score(y_aux_c, aux_t, anchor_fit.second, n_c=n_c)
        log_prop = b_log + 0.75 * coarse
        idx, rep_mass, log_prop_norm = stratified_mode_midpoints(log_prop, bank.strata_4, budget=budget)
        exact = parity_block_cov_score(y_aux_c, aux_t, bank.nodes[idx], anchor_fit.second[idx], n_c=n_c)
        correction = np.log(rep_mass) + float(aux_temp) * exact - 0.75 * coarse[idx]
        lse_corr = float(logsumexp(correction))
        logw0 = (correction - lse_corr).astype(REAL_DTYPE)
        s_r_canon = log_prop_norm + lse_corr
        n_samples_total = N + len(aux_t) * N
    else:
        idx = np.arange(len(bank.nodes), dtype=np.int64)
        logw0 = (b_log - pilot_ll_canon).astype(REAL_DTYPE)
        s_r_canon = pilot_ll_canon
        n_samples_total = N

    s_r_phys = s_r_canon - float(n_samples_total) * math.log(float(abs(c_b) ** 2))

    node_ids_sub = bank.nodes[idx].copy()
    anchor_le_sub = anchor_fit.log_evidence[idx].copy()
    anchor_mean_sub = anchor_fit.mean[idx].copy()
    anchor_cov_sub = anchor_fit.cov[idx].copy()
    anchor_sec_sub = anchor_fit.second[idx].copy()
    sign_sub = bank.sign_ray0[idx].copy()

    jstate = JointResetState(
        gauge=c_b,
        n0=n0_val,
        node_ids=node_ids_sub,
        logw=logw0,
        mean_c=anchor_mean_sub,
        cov_c=anchor_cov_sub,
        sign_ray0=sign_sub,
    )
    mean_phys, var_phys = project_joint_moments(jstate)
    probe_pred, probe_var = project_joint_probe(jstate, probe_row=PROBE_ROW)
    state_proj = project_to_tracker_state(jstate)

    return PayloadPosterior(
        gauge=c_b,
        n0=n0_val,
        aux_rows=aux_t,
        use_payload=bool(use_payload),
        aux_temp=float(aux_temp),
        node_ids=node_ids_sub,
        logw0=logw0,
        anchor_log_evidence=anchor_le_sub,
        anchor_mean=anchor_mean_sub,
        anchor_cov=anchor_cov_sub,
        anchor_second=anchor_sec_sub,
        scales=scales,
        sign_ray0=sign_sub,
        class_log_evidence=float(s_r_phys),
        class_log_evidence_canon=float(s_r_canon),
        pilot_loglik=float(pilot_ll_phys),
        mean=mean_phys,
        var=var_phys,
        state_clr_proj=state_proj,
        probe_pred=probe_pred,
        probe_var=probe_var,
    )


def estimate_payload_posterior_obs(
    obs: PublicObservation,
    aux: Sequence[int],
    *,
    use_payload: bool = True,
    aux_temp: float = 1.0,
    budget: int = FINAL_NODES,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
) -> PayloadPosterior:
    """PublicObservation wrapper slicing ONLY `obs.y_b[ANCHOR]` and `obs.y_b[aux]`."""
    n0_val = _validate_positive_n0(obs.n0, name="obs.n0")
    train_t = tuple(_strict_row_int(r, "obs.training_rows") for r in obs.training_rows)
    if train_t != (ANCHOR,):
        raise ValueError(f"Expected sparse training_rows={(ANCHOR,)}, got {train_t}")
    p_row = _strict_row_int(obs.probe_row, "obs.probe_row")
    aux_t = tuple(_strict_row_int(r, "aux") for r in aux)
    if any(r < 0 or r >= T or r in (ANCHOR, p_row) for r in aux_t):
        raise ValueError(f"Invalid auxiliary rows {aux_t} (must exclude ANCHOR={ANCHOR} and PROBE_ROW={p_row})")
    if use_payload and (len(set(aux_t)) != 5 or len(aux_t) != 5):
        raise ValueError(f"Expected 5 distinct valid auxiliary payload rows excluding (10, 19), got {aux_t}")
    y10 = np.asarray(obs.y_b[ANCHOR], dtype=COMPLEX_DTYPE).copy()
    y_aux = np.asarray(obs.y_b[list(aux_t)], dtype=COMPLEX_DTYPE).copy() if (use_payload and aux_t) else None
    return estimate_payload_posterior_raw(
        y10=y10,
        aux_rows=aux_t,
        y_aux=y_aux,
        n0=n0_val,
        use_payload=use_payload,
        aux_temp=aux_temp,
        budget=budget,
        scale_quad_order=scale_quad_order,
    )


def estimate_payload_posterior(
    y10_or_obs: np.ndarray | PublicObservation,
    aux_rows: Sequence[int],
    y_aux: Optional[np.ndarray] = None,
    n0: Optional[float] = None,
    *,
    use_payload: bool = True,
    aux_temp: float = 1.0,
    budget: int = FINAL_NODES,
    scale_quad_order: int = DEFAULT_SCALE_QUAD_ORDER,
    gauge_mode: str = "canonical",
) -> PayloadPosterior:
    """Unified entrypoint accepting either `(y10, aux_rows, y_aux, n0)` or `(obs, aux)`."""
    if gauge_mode != "canonical":
        raise ValueError(f"Unsupported gauge_mode={gauge_mode!r}; P11 mandates 'canonical'")
    if isinstance(y10_or_obs, PublicObservation):
        return estimate_payload_posterior_obs(
            y10_or_obs,
            aux_rows,
            use_payload=use_payload,
            aux_temp=aux_temp,
            budget=budget,
            scale_quad_order=scale_quad_order,
        )
    if n0 is None:
        raise ValueError("n0 must be provided when calling estimate_payload_posterior with raw arrays")
    n0_val = _validate_positive_n0(n0, name="n0")
    return estimate_payload_posterior_raw(
        y10=y10_or_obs,
        aux_rows=tuple(_strict_row_int(r, "aux_rows") for r in aux_rows),
        y_aux=y_aux,
        n0=n0_val,
        use_payload=use_payload,
        aux_temp=aux_temp,
        budget=budget,
        scale_quad_order=scale_quad_order,
    )


def refit_reset(
    base: ResetBase,
    y10: np.ndarray,
    virtual_ledger: Mapping[int, tuple[np.ndarray, np.ndarray, float]] | Sequence[tuple[int, np.ndarray, np.ndarray, float]],
) -> JointResetState:
    """Refit persistent `ResetBase` from `A union V`: `logw_j(V) = normalize(logw0 + L_j(A u V) - L_j(A))`.

    `virtual_ledger` maps `row -> (x_hat_symbols [N], y_b_row [N], n_eff_phys)` (or is a sequence
    of `(row, x_hat, y_b_row, n_eff)` tuples). When empty, reproduces `base` identically.
    """
    if isinstance(virtual_ledger, Mapping):
        items = sorted(((_strict_row_int(r, "virtual_ledger row"), val[0], val[1], float(val[2])) for r, val in virtual_ledger.items()), key=lambda t: t[0])
    else:
        items = sorted(((_strict_row_int(t[0], "virtual_ledger row"), t[1], t[2], float(t[3])) for t in virtual_ledger), key=lambda t: t[0])

    if not items:
        return base.to_joint_state()

    rows_v = [t[0] for t in items]
    if len(set(rows_v)) != len(rows_v):
        raise ValueError("Duplicate virtual rows in ledger")
    if any(r < 0 or r >= T or r in (ANCHOR, PROBE_ROW) for r in rows_v):
        raise ValueError(f"Illegal virtual rows {rows_v}: must exclude ANCHOR={ANCHOR} and PROBE_ROW={PROBE_ROW}")

    y10_arr = np.asarray(y10, dtype=COMPLEX_DTYPE).ravel()
    if y10_arr.shape != (N,) or not np.isfinite(y10_arr).all():
        raise ValueError(f"Expected finite y10 shape ({N},)")

    g = complex(base.gauge)
    n0_val = float(base.n0)

    all_rows = (ANCHOR,) + tuple(rows_v)
    words = np.vstack([SC_PILOTS[ANCHOR : ANCHOR + 1], np.stack([np.asarray(t[1], dtype=COMPLEX_DTYPE).ravel() for t in items], axis=0)])
    recv_phys = np.vstack([y10_arr[None, :], np.stack([np.asarray(t[2], dtype=COMPLEX_DTYPE).ravel() for t in items], axis=0)])
    noises_phys = np.array([n0_val] + [float(t[3]) for t in items], dtype=REAL_DTYPE)
    if np.any(noises_phys <= 0.0) or not np.isfinite(noises_phys).all():
        raise ValueError("Virtual block effective noise variances must be finite and strictly positive")

    post = fit_known_blocks_by_node(
        base.node_ids,
        rows=all_rows,
        words=words,
        received=recv_phys / g,
        noises=noises_phys / n0_val,
        scales=base.scales,
    )
    raw_lw = base.logw0 + post.log_evidence - base.anchor_log_evidence
    logw = (raw_lw - float(logsumexp(raw_lw))).astype(REAL_DTYPE)

    return JointResetState(
        gauge=g,
        n0=n0_val,
        node_ids=base.node_ids.copy(),
        logw=logw,
        mean_c=post.mean,
        cov_c=post.cov,
        sign_ray0=base.sign_ray0.copy(),
    )


def split_joint_reset_ray0_sign(jstate: JointResetState) -> tuple[float, float, JointResetState, JointResetState]:
    """Condition ALL 3 rays of `JointResetState` on Ray-0 positive (`True`) vs negative (`False`) frequency."""
    w = np.exp(np.asarray(jstate.logw, dtype=REAL_DTYPE))
    mask_pos = np.asarray(jstate.sign_ray0, dtype=bool)
    mask_neg = ~mask_pos
    p_pos = float(np.sum(w[mask_pos])) if np.any(mask_pos) else 0.0
    p_neg = float(np.sum(w[mask_neg])) if np.any(mask_neg) else 0.0

    def _sub(mask: np.ndarray) -> JointResetState:
        if not np.any(mask):
            return jstate
        lw_sub = jstate.logw[mask]
        lw_sub = (lw_sub - float(logsumexp(lw_sub))).astype(REAL_DTYPE)
        return JointResetState(
            gauge=jstate.gauge,
            n0=jstate.n0,
            node_ids=jstate.node_ids[mask].copy(),
            logw=lw_sub,
            mean_c=jstate.mean_c[mask].copy(),
            cov_c=jstate.cov_c[mask].copy(),
            sign_ray0=jstate.sign_ray0[mask].copy(),
        )

    return p_pos, p_neg, _sub(mask_pos), _sub(mask_neg)


# ---------------------------------------------------------------------------
# 6. Clean Frame-A Prior, Persistent Shift Dynamics & Common-Unit 3-Branch Scorer
# ---------------------------------------------------------------------------


def build_frame_a_prior_clean(y_a: np.ndarray, n0: float) -> State:
    """Construct Frame-A inherited prior State at epoch 30 without any `1e-300` physical floor.

    Strictly isolated from `y_b`. Guarantees exact C*-equivariance across all scales.
    """
    n0_val = _validate_positive_n0(n0, name="n0")
    sel = np.ix_(OFDM_PILOT_ROWS, EVEN)
    pilots_a = np.asarray(y_a, dtype=COMPLEX_DTYPE)[sel]
    if not np.isfinite(pilots_a).all():
        raise ValueError("obs.y_a pilot cells contain non-finite values")
    ref_pilots = SC_PILOTS  # unused here; use p8.link OFDM_PILOTS
    from p8.link import PILOTS as OFDM_PILOTS_MAT

    ref_a = OFDM_PILOTS_MAT[sel]
    mags = np.abs(pilots_a)
    peak = float(np.max(mags))
    if peak > 0.0:
        p_unit = pilots_a / peak
        e_a = (peak * peak) * float(np.mean(np.abs(p_unit) ** 2))
        proj = complex(np.sum(p_unit * ref_a.conj()))
        if abs(proj) > 1e-12:
            u_a = proj / abs(proj)
        else:
            dom = int(np.argmax(mags))
            dv = complex(p_unit.ravel()[dom] * ref_a.ravel()[dom].conj())
            u_a = (dv / abs(dv)) if abs(dv) > 0.0 else (1.0 + 0.0j)
    else:
        e_a = 0.0
        u_a = 1.0 + 0.0j

    power_scale = float(max(e_a - n0_val, 0.05 * e_a, 0.05 * n0_val))
    c_a = complex(math.sqrt(power_scale) * u_a)
    n0_c = n0_val / power_scale
    y_a_c = np.asarray(y_a, dtype=COMPLEX_DTYPE) / c_a
    taps_c = observed_taps(y_a_c, OFDM_PILOT_ROWS)
    state_a_c = update_ofdm_state(prior(), OFDM_PILOT_ROWS, taps_c, n0=n0_c)
    state_b_c = advance(state_a_c, float(FRAME_DURATION))
    return State(
        epoch=float(state_b_c.epoch),
        log_weight=np.asarray(state_b_c.log_weight, dtype=REAL_DTYPE).copy(),
        mean=np.asarray(state_b_c.mean, dtype=COMPLEX_DTYPE) * c_a,
        variance=np.asarray(state_b_c.variance, dtype=REAL_DTYPE) * power_scale,
    )


def scale_tracker_state(state: State, complex_scale: complex | float) -> State:
    c = complex(complex_scale)
    s2 = float(abs(c) ** 2)
    return State(
        epoch=float(state.epoch),
        log_weight=np.asarray(state.log_weight, dtype=REAL_DTYPE).copy(),
        mean=np.asarray(state.mean, dtype=COMPLEX_DTYPE) * c,
        variance=np.asarray(state.variance, dtype=REAL_DTYPE) * s2,
    )


def sample_moments_b_stable_rows(state: State, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Cancellation-free within+between ray moments `[3, R, N]` on `rows` (never subtracts second moments)."""
    row_list = list(range(T)) if rows is None else [_strict_row_int(r, "rows") for r in rows]
    weights = np.exp(np.asarray(state.log_weight, dtype=REAL_DTYPE))  # [3, H]
    rotations = evolution(float(state.epoch))[:, row_list, :].reshape(weights.shape[1], -1)  # [H, R*N]
    m_nodes = np.asarray(state.mean, dtype=COMPLEX_DTYPE)
    v_nodes = np.asarray(state.variance, dtype=REAL_DTYPE)

    mean_flat = (weights * m_nodes) @ rotations  # [3, R*N]
    within = (weights * v_nodes).sum(axis=1)[:, None]  # [3, 1]
    between = np.empty_like(mean_flat, dtype=REAL_DTYPE)
    for ray in range(N_RAYS):
        diff = m_nodes[ray, :, None] * rotations - mean_flat[ray, None, :]
        between[ray] = weights[ray] @ (diff.real ** 2 + diff.imag ** 2)
    var_flat = np.maximum(within + between, 0.0)
    r_cnt = len(row_list)
    return mean_flat.reshape(N_RAYS, r_cnt, N), var_flat.reshape(N_RAYS, r_cnt, N)


@dataclass(frozen=True)
class ShiftedBranchDynamics:
    """Persistent phase-continuous shift dynamics `M_a(t) = exp(2*pi*i * 1_{a=a*} * delta * (t - tau_b)_+)`.

    When `changed_ray is None` or `delta_freq == 0.0`, reduces identically to ordinary continuous
    `p8.tracker.State` dynamics (`M_a(t) == 1`).
    """
    prior_state: State
    changed_ray: Optional[int] = None
    delta_freq: float = 0.0
    onset_row: int = 0

    def __post_init__(self) -> None:
        if self.changed_ray is not None:
            r = _strict_row_int(self.changed_ray, "changed_ray")
            if r < 0 or r >= N_RAYS:
                raise ValueError(f"changed_ray must be in 0..{N_RAYS - 1} or None, got {self.changed_ray}")
        onset_i = _strict_row_int(self.onset_row, "onset_row")
        if onset_i < 0 or onset_i >= T:
            raise ValueError(f"onset_row must be in 0..{T - 1}, got {self.onset_row}")

    @property
    def tau_onset(self) -> float:
        return float(FRAME_DURATION) + 1.5 * float(self.onset_row)

    def shift_rotation(self, rows: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Return `M_{changed_ray}(t)` of shape `[len(rows), N]` (`ones` if unshifted or `t < tau_b`)."""
        row_list = list(range(T)) if rows is None else [_strict_row_int(r, "rows") for r in rows]
        if self.changed_ray is None or float(self.delta_freq) == 0.0:
            return np.ones((len(row_list), N), dtype=COMPLEX_DTYPE)
        t_sub = ABS_TIME[row_list, :]
        dt_pos = np.maximum(t_sub - self.tau_onset, 0.0)
        return np.exp(2j * np.pi * float(self.delta_freq) * dt_pos).astype(COMPLEX_DTYPE)

    def refit(
        self,
        rows: tuple[int, ...],
        tx_blocks: np.ndarray,
        y_blocks: np.ndarray,
        n0_per_block: np.ndarray,
        *,
        sweeps: int = SWEEPS,
        probe_row: int = PROBE_ROW,
    ) -> State:
        """Condition `self.prior_state` on `rows` applying `M_a(t)` to delayed symbols."""
        return condition_sc_weighted_shifted(
            self.prior_state,
            rows,
            tx_blocks,
            y_blocks,
            n0_per_block,
            changed_ray=self.changed_ray,
            delta_freq=self.delta_freq,
            onset_row=self.onset_row,
            sweeps=sweeps,
            probe_row=probe_row,
        )

    def predict(self, post_state: State, rows: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict `[3, R, N]` ray `(mean, var)` applying `M_a(t)` to `changed_ray`."""
        mean, var = sample_moments_b_stable_rows(post_state, rows=rows)
        if self.changed_ray is not None and float(self.delta_freq) != 0.0:
            mean = mean.copy()
            mean[int(self.changed_ray)] *= self.shift_rotation(rows=rows)
        return mean, var

    def predict_probe(
        self,
        post_state: State,
        n0: float,
        probe_row: int = PROBE_ROW,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict withheld probe `(pred [N], var [N])` under persistent dynamics (`>= n0`)."""
        p_row = _strict_row_int(probe_row, "probe_row")
        mean_p, var_p = self.predict(post_state, rows=(p_row,))
        known = SC_PILOTS[p_row, DELAYED_INDEX]  # [3, N]
        pred = (mean_p[:, 0, :] * known).sum(axis=0)
        pvar = float(n0) + (var_p[:, 0, :] * (np.abs(known) ** 2)).sum(axis=0)
        return np.asarray(pred, dtype=COMPLEX_DTYPE), np.asarray(pvar, dtype=REAL_DTYPE)


def condition_sc_weighted_shifted(
    prior_state: State,
    rows: tuple[int, ...],
    tx_blocks: np.ndarray,
    y_blocks: np.ndarray,
    n0_per_block: np.ndarray,
    *,
    changed_ray: Optional[int] = None,
    delta_freq: float = 0.0,
    onset_row: int = 0,
    sweeps: int = SWEEPS,
    probe_row: int = PROBE_ROW,
) -> State:
    """Rank-factored BLAS coordinate ascent with phase-continuous `M_a(t)` on `changed_ray`."""
    p_row = _strict_row_int(probe_row, "probe_row")
    rows_t = tuple(_strict_row_int(r, "rows") for r in rows)
    if len(set(rows_t)) != len(rows_t) or any(r < 0 or r >= T or r == p_row for r in rows_t):
        raise ValueError(f"Invalid rows {rows_t} (must exclude PROBE_ROW={p_row})")
    m_blocks = len(rows_t)
    if m_blocks == 0:
        return prior_state

    tx = np.asarray(tx_blocks, dtype=COMPLEX_DTYPE)
    yb = np.asarray(y_blocks, dtype=COMPLEX_DTYPE)
    n0_arr = np.asarray(n0_per_block, dtype=REAL_DTYPE).ravel()
    if tx.shape != (m_blocks, N) or yb.shape != (m_blocks, N) or n0_arr.shape != (m_blocks,):
        raise ValueError(f"Shape mismatch: tx={tx.shape}, y={yb.shape}, n0={n0_arr.shape}")
    if not (np.isfinite(tx).all() and np.isfinite(yb).all() and np.isfinite(n0_arr).all()):
        raise ValueError("Non-finite inputs to condition_sc_weighted_shifted")
    if np.any(n0_arr <= 0.0):
        raise ValueError("All per-block effective noise variances must be strictly positive")

    # Scale-canonicalize without additive physical floor (`prior_state.variance` and `n0_arr` > 0)
    mean_pvar = float(np.mean(prior_state.variance))
    min_n0 = float(np.min(n0_arr))
    ref_var = max(mean_pvar, min_n0)
    amp_scale = math.sqrt(ref_var)
    prec_c = ref_var / n0_arr
    q_dim = m_blocks * N
    y_flat = (yb / amp_scale).reshape(q_dim)
    prec_flat = np.repeat(prec_c, N)

    prior_mean_c = np.asarray(prior_state.mean, dtype=COMPLEX_DTYPE) / amp_scale
    prior_var_c = np.asarray(prior_state.variance, dtype=REAL_DTYPE) / ref_var
    prior_lw = np.asarray(prior_state.log_weight, dtype=REAL_DTYPE)

    rot = np.ascontiguousarray(
        evolution(float(prior_state.epoch))[:, list(rows_t), :].reshape(-1, q_dim),
        dtype=COMPLEX_DTYPE,
    )
    rot_conj = rot.conj()

    delayed_3d = np.stack([np.roll(tx, int(d), axis=-1) for d in DELAYS], axis=0)  # [3, M, N]
    if changed_ray is not None and float(delta_freq) != 0.0:
        tau_b = float(FRAME_DURATION) + 1.5 * float(onset_row)
        dt_pos = np.maximum(ABS_TIME[list(rows_t), :] - tau_b, 0.0)
        m_rot = np.exp(2j * np.pi * float(delta_freq) * dt_pos)
        delayed_3d[int(changed_ray)] *= m_rot
    delayed = delayed_3d.reshape(N_RAYS, q_dim)
    delayed_conj_prec = delayed.conj() * prec_flat[None, :]

    block_energy = np.sum(tx.real ** 2 + tx.imag ** 2, axis=-1)
    energy_prec_scalar = float(np.dot(block_energy, prec_c))
    post_var_c = prior_var_c / (1.0 + prior_var_c * energy_prec_scalar)

    # Mask zero-mass (-inf log_weight) cells cleanly during coordinate ascent
    finite_mask = np.isfinite(prior_lw)
    safe_lw = np.where(finite_mask, prior_lw, -np.inf)
    log_weight = safe_lw.copy()
    mean_c = prior_mean_c.copy()
    w_init = np.where(finite_mask, np.exp(safe_lw), 0.0)
    per_ray = (w_init * mean_c) @ rot
    per_ray = per_ray * delayed
    total_resp = per_ray.sum(axis=0)

    initial_precision_mean = prior_mean_c / prior_var_c
    evidence_base = np.where(
        finite_mask,
        safe_lw + np.log(post_var_c / prior_var_c) - (prior_mean_c.real ** 2 + prior_mean_c.imag ** 2) / prior_var_c,
        -np.inf,
    )

    for _ in range(int(sweeps)):
        for ray in range(N_RAYS):
            residual = y_flat - total_resp + per_ray[ray]
            matched_prec = rot_conj @ (delayed_conj_prec[ray] * residual)
            natural_mean = initial_precision_mean[ray] + matched_prec
            pv = post_var_c[ray]
            m_ray = pv * natural_mean
            mean_c[ray] = m_ray

            ev = np.where(
                finite_mask[ray],
                evidence_base[ray] + pv * (natural_mean.real ** 2 + natural_mean.imag ** 2),
                -np.inf,
            )
            lse = float(logsumexp(ev))
            lw_ray = np.where(finite_mask[ray], ev - lse, -np.inf)
            log_weight[ray] = lw_ray

            w_ray = np.where(finite_mask[ray], np.exp(lw_ray), 0.0)
            new_pr = ((w_ray * m_ray) @ rot) * delayed[ray]
            total_resp = total_resp - per_ray[ray] + new_pr
            per_ray[ray] = new_pr

    return State(
        epoch=float(prior_state.epoch),
        log_weight=log_weight,
        mean=mean_c * amp_scale,
        variance=post_var_c * ref_var,
    )


@njit
def _score_3ray_cov_numba(
    g_all: np.ndarray,   # [M, 3, 32] complex128
    v_all: np.ndarray,   # [M, 32] float64
    y_all: np.ndarray,   # [M, 32] complex128
) -> np.ndarray:
    """Exact Hermitian Cholesky + O(N^2) triangular solve for 3-ray (0, 2, 4) 32x32 covariance."""
    n_items = g_all.shape[0]
    dim = g_all.shape[2]
    loglik = np.empty(n_items, dtype=np.float64)
    const_term = float(dim) * math.log(math.pi)

    l_mat = np.empty((dim, dim), dtype=np.complex128)
    z = np.empty(dim, dtype=np.complex128)

    for idx in range(n_items):
        for i in range(dim):
            for j in range(i + 1):
                l_mat[i, j] = 0.0 + 0.0j

        for n in range(dim):
            g0 = g_all[idx, 0, n]
            g1 = g_all[idx, 1, n]
            g2 = g_all[idx, 2, n]
            d_val = (
                g0.real * g0.real + g0.imag * g0.imag
                + g1.real * g1.real + g1.imag * g1.imag
                + g2.real * g2.real + g2.imag * g2.imag
                + v_all[idx, n]
            )
            l_mat[n, n] = d_val + 0.0j

            p2 = (n + 2) % dim
            c2 = g0 * g_all[idx, 1, p2].conjugate() + g1 * g_all[idx, 2, p2].conjugate()
            if p2 > n:
                l_mat[p2, n] = c2.conjugate()
            else:
                l_mat[n, p2] = c2

            p4 = (n + 4) % dim
            c4 = g0 * g_all[idx, 2, p4].conjugate()
            if p4 > n:
                l_mat[p4, n] = c4.conjugate()
            else:
                l_mat[n, p4] = c4

        ld = 0.0
        for j in range(dim):
            s_re = l_mat[j, j].real
            for k in range(j):
                lk = l_mat[j, k]
                s_re -= lk.real * lk.real + lk.imag * lk.imag
            if s_re < 1e-300:
                s_re = 1e-300
            diag_j = math.sqrt(s_re)
            l_mat[j, j] = diag_j + 0.0j
            ld += math.log(diag_j)
            inv_dj = 1.0 / diag_j
            for i in range(j + 1, dim):
                acc = l_mat[i, j]
                for k in range(j):
                    acc -= l_mat[i, k] * l_mat[j, k].conjugate()
                l_mat[i, j] = acc * inv_dj

        q_sum = 0.0
        for i in range(dim):
            acc = y_all[idx, i]
            for k in range(i):
                acc -= l_mat[i, k] * z[k]
            zi = acc / l_mat[i, i].real
            z[i] = zi
            q_sum += zi.real * zi.real + zi.imag * zi.imag

        loglik[idx] = -const_term - 2.0 * ld - q_sum
    return loglik


def score_3ray_payload_fast(
    g_batch: np.ndarray,
    v_batch: np.ndarray,
    y_batch: np.ndarray,
) -> np.ndarray:
    """Fast exact batched covariance scoring for `[..., 3, N]` ray gains via Numba Cholesky."""
    g_arr = np.ascontiguousarray(g_batch, dtype=COMPLEX_DTYPE)
    v_arr = np.ascontiguousarray(v_batch, dtype=REAL_DTYPE)
    batch_shape = g_arr.shape[:-2]
    y_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(y_batch, dtype=COMPLEX_DTYPE), (*batch_shape, N)))
    m_tot = int(np.prod(batch_shape)) if batch_shape else 1
    ll = _score_3ray_cov_numba(
        g_arr.reshape(m_tot, N_RAYS, N),
        v_arr.reshape(m_tot, N),
        y_arr.reshape(m_tot, N),
    )
    return ll.reshape(batch_shape)


@dataclass(frozen=True)
class ShiftHypothesis:
    ray: Optional[int]
    delta: float
    onset_row: int


@lru_cache(maxsize=1)
def shift_hypothesis_bank() -> tuple[ShiftHypothesis, ...]:
    """Return 144 shift hypotheses: 3 rays x 24 symmetric deltas x 2 onsets `(0, 6)`."""
    return tuple(
        ShiftHypothesis(ray=r, delta=float(d), onset_row=b)
        for b in ONSET_ROWS
        for r in range(N_RAYS)
        for d in DELTAS
    )


def _per_ray_pilot_moments_stable(
    state: State,
    rows: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Cancellation-free pre-update per-ray mean `[3, M]` and covariance `[3, M, M]`."""
    rot = evolution(float(state.epoch))[:, list(rows), :]  # [H, R, N]
    pilots = SC_PILOTS[list(rows)]  # [R, N]
    delayed = np.stack([np.roll(pilots, int(d), axis=-1) for d in DELAYS], axis=0)  # [3, R, N]
    design = (delayed[:, None, :, :] * rot[None, :, :, :]).reshape(N_RAYS, rot.shape[0], -1)  # [3, H, M]
    m_dim = design.shape[-1]

    lw = np.asarray(state.log_weight, dtype=REAL_DTYPE)
    weights = np.where(np.isfinite(lw), np.exp(lw), 0.0)  # [3, H]
    mean = np.asarray(state.mean, dtype=COMPLEX_DTYPE)
    var = np.asarray(state.variance, dtype=REAL_DTYPE)

    mu_rays = np.zeros((N_RAYS, m_dim), dtype=COMPLEX_DTYPE)
    cov_rays = np.zeros((N_RAYS, m_dim, m_dim), dtype=COMPLEX_DTYPE)
    for ray in range(N_RAYS):
        d_ray = design[ray]  # [H, M]
        mu_r = (weights[ray] * mean[ray]) @ d_ray
        within_r = (d_ray.T * (weights[ray] * var[ray])[None, :]) @ d_ray.conj()
        diff_h = d_ray * mean[ray, :, None] - mu_r[None, :]
        between_r = (diff_h.T * weights[ray][None, :]) @ diff_h.conj()
        cov_r = 0.5 * ((within_r + between_r) + (within_r + between_r).conj().T)
        mu_rays[ray] = mu_r
        cov_rays[ray] = cov_r
    return mu_rays, cov_rays


def _score_one_inherited_hypothesis(
    a_prior_c: State,
    y10_c: np.ndarray,
    aux_rows: tuple[int, ...],
    y_aux_c: np.ndarray,
    *,
    changed_ray: Optional[int],
    delta_freq: float,
    onset_row: int,
    aux_temp: float,
) -> float:
    """Exact common-unit composite score `L_pilot + aux_temp * L_aux` for a single `(C/S)` hypothesis."""
    dyn = ShiftedBranchDynamics(a_prior_c, changed_ray=changed_ray, delta_freq=delta_freq, onset_row=onset_row)
    mu_rays, cov_rays = _per_ray_pilot_moments_stable(a_prior_c, (ANCHOR,))
    mu_base = mu_rays.sum(axis=0)
    cov_base = cov_rays.sum(axis=0)
    cov_base[np.diag_indices(N)] += 1.0

    if changed_ray is not None and float(delta_freq) != 0.0:
        r_idx = int(changed_ray)
        rot_10 = dyn.shift_rotation((ANCHOR,)).reshape(N)
        mu_r = mu_rays[r_idx]
        cov_r = cov_rays[r_idx]
        mu_h = (mu_base - mu_r) + rot_10 * mu_r
        cov_h = (cov_base - cov_r) + rot_10[:, None] * cov_r * rot_10.conj()[None, :]
    else:
        mu_h = mu_base
        cov_h = cov_base
    cov_h = 0.5 * (cov_h + cov_h.conj().T)

    chol = np.linalg.cholesky(cov_h)
    diag_l = np.real(np.diagonal(chol))
    logdet = float(2.0 * np.sum(np.log(np.maximum(diag_l, 1e-300))))
    whitened = solve_triangular(chol, y10_c - mu_h, lower=True, check_finite=False)
    quad = float(np.sum(whitened.real ** 2 + whitened.imag ** 2))
    pilot_ll = -float(N) * math.log(math.pi) - logdet - quad

    post_a = dyn.refit((ANCHOR,), SC_PILOTS[ANCHOR : ANCHOR + 1], y10_c[None, :], np.array([1.0], dtype=REAL_DTYPE))
    m_aux, v_aux = dyn.predict(post_a, rows=aux_rows)  # [3, R, N]
    g_rows = np.transpose(m_aux, (1, 0, 2))            # [R, 3, N]
    nv_rows = 1.0 + v_aux.sum(axis=0)                  # [R, N]
    aux_ll = float(np.sum(score_3ray_payload_fast(g_rows, nv_rows, y_aux_c)))
    return pilot_ll + float(aux_temp) * aux_ll


@dataclass(frozen=True)
class BranchEvidence:
    b_c: float
    b_s: float
    b_r: float
    s_c: float
    s_s_best: float
    s_r: float
    best_shift: ShiftHypothesis
    refined_delta: float
    reset_base: ResetBase
    a_prior: State


@njit
def _within_between_aux_numba(
    w_b: np.ndarray,         # [K, 3, H] float64
    m_phys_b: np.ndarray,    # [K, 3, H] complex128
    v_phys_b: np.ndarray,    # [3, H] float64
    rot_aux_ev: np.ndarray,  # [H, Q] complex128
    mean_flat: np.ndarray,   # [K, 3, Q] complex128
) -> np.ndarray:
    """Compute cancellation-free within+between variance [K, 3, Q] without [K, H, Q] temp tensor."""
    k_cnt, n_rays, h_cnt = w_b.shape
    q_cnt = rot_aux_ev.shape[1]
    var_out = np.empty((k_cnt, n_rays, q_cnt), dtype=np.float64)
    for k in range(k_cnt):
        for ray in range(n_rays):
            within = 0.0
            for h in range(h_cnt):
                within += w_b[k, ray, h] * v_phys_b[ray, h]
            for q in range(q_cnt):
                mu = mean_flat[k, ray, q]
                btw = 0.0
                for h in range(h_cnt):
                    diff = m_phys_b[k, ray, h] * rot_aux_ev[h, q] - mu
                    btw += w_b[k, ray, h] * (diff.real * diff.real + diff.imag * diff.imag)
                tot = within + btw
                var_out[k, ray, q] = tot if tot > 0.0 else 0.0
    return var_out


def score_branches_common_units(
    a_prior: State,
    reset_base: ResetBase,
    y10: np.ndarray,
    aux_rows: tuple[int, ...],
    y_aux: np.ndarray,
    n0: float,
    *,
    aux_temp: float = 1.0,
    use_parabolic: bool = True,
) -> BranchEvidence:
    """Score `(C no-change, S shift, R reset)` on identical anchor `(10,)` + `aux_rows` in common `c_B` units.

    Applies `(1/3, 1/3, 1/3)` top-level priors, uniform `1/144` shift multiplicity prior,
    anchor-conditioned payload scoring across all three classes (no pre-B vs post-B asymmetry),
    and recomputes BOTH state and evidence score `S_h` upon parabolic delta refinement.
    """
    c_b = complex(reset_base.gauge)
    y10_arr = np.asarray(y10, dtype=COMPLEX_DTYPE).ravel()
    y_aux_arr = np.asarray(y_aux, dtype=COMPLEX_DTYPE)
    y10_c = y10_arr / c_b
    y_aux_c = y_aux_arr / c_b
    a_prior_c = scale_tracker_state(a_prior, 1.0 / c_b)

    shift_bank = shift_hypothesis_bank()
    n_shifts = len(shift_bank)
    n_hyps = 1 + n_shifts

    # 1. Pre-anchor pilot marginal log-likelihoods for C (idx 0) and 144 shifts (idx 1..144)
    mu_rays, cov_rays = _per_ray_pilot_moments_stable(a_prior_c, (ANCHOR,))
    mu_base = mu_rays.sum(axis=0)
    cov_base = cov_rays.sum(axis=0)
    cov_base[np.diag_indices(N)] += 1.0
    cov_base = 0.5 * (cov_base + cov_base.conj().T)

    mu_all = np.empty((n_hyps, N), dtype=COMPLEX_DTYPE)
    cov_all = np.empty((n_hyps, N, N), dtype=COMPLEX_DTYPE)
    mu_all[0] = mu_base
    cov_all[0] = cov_base

    t_10 = ABS_TIME[ANCHOR, :]
    # Precompute anchor rotation for every (onset, delta)
    rot_anchor_by_od: dict[tuple[int, float], np.ndarray] = {}
    for b in ONSET_ROWS:
        tau_b = float(FRAME_DURATION) + 1.5 * float(b)
        dt_10 = np.maximum(t_10 - tau_b, 0.0)
        for d in DELTAS:
            rot_anchor_by_od[(b, float(d))] = np.exp(2j * np.pi * float(d) * dt_10).astype(COMPLEX_DTYPE)

    for idx_h, hyp in enumerate(shift_bank, start=1):
        r_idx = int(hyp.ray)
        rot_10 = rot_anchor_by_od[(hyp.onset_row, hyp.delta)]
        mu_r = mu_rays[r_idx]
        cov_r = cov_rays[r_idx]
        mu_all[idx_h] = (mu_base - mu_r) + rot_10 * mu_r
        cov_h = (cov_base - cov_r) + rot_10[:, None] * cov_r * rot_10.conj()[None, :]
        cov_all[idx_h] = 0.5 * (cov_h + cov_h.conj().T)

    chol_p = np.linalg.cholesky(cov_all)
    diag_lp = np.real(np.diagonal(chol_p, axis1=-2, axis2=-1))
    logdet_p = 2.0 * np.sum(np.log(np.maximum(diag_lp, 1e-300)), axis=-1)
    resid_p = y10_c[None, :] - mu_all
    whitened_p = np.linalg.solve(chol_p, resid_p[..., None])[..., 0]
    quad_p = np.sum(whitened_p.real ** 2 + whitened_p.imag ** 2, axis=-1)
    pilot_lls = (-float(N) * math.log(math.pi) - logdet_p - quad_p).astype(REAL_DTYPE)

    # 2. Batched anchor conditioning + anchor-conditioned auxiliary payload scoring
    aux_list = [_strict_row_int(r, "aux_rows") for r in aux_rows]
    r_aux = len(aux_list)
    g_aux_batch = np.empty((n_hyps, r_aux, N_RAYS, N), dtype=COMPLEX_DTYPE)
    v_aux_batch = np.empty((n_hyps, r_aux, N), dtype=REAL_DTYPE)

    # Cache unique (ray, delta, onset) conditioned states; note for anchor=10 only,
    # M_a(t_10) depends on (onset, delta) — still fast (~0.05 ms/hyp or batched!).
    rot_anchor_ev = np.ascontiguousarray(
        evolution(float(a_prior_c.epoch))[:, ANCHOR, :], dtype=COMPLEX_DTYPE
    )  # [128, 32]
    rot_aux_ev = np.ascontiguousarray(
        evolution(float(a_prior_c.epoch))[:, aux_list, :].reshape(len(FREQ), -1), dtype=COMPLEX_DTYPE
    )  # [128, R*32]

    prior_var_c = np.asarray(a_prior_c.variance, dtype=REAL_DTYPE)
    prior_mean_c = np.asarray(a_prior_c.mean, dtype=COMPLEX_DTYPE)
    prior_lw = np.asarray(a_prior_c.log_weight, dtype=REAL_DTYPE)
    mean_pvar = float(np.mean(prior_var_c))
    ref_var = max(mean_pvar, 1.0)
    amp_s = math.sqrt(ref_var)
    prec_c = ref_var / 1.0
    pm_c = prior_mean_c / amp_s
    pv_c = prior_var_c / ref_var
    y10_sc = y10_c / amp_s

    delayed_10_base = np.stack([np.roll(SC_PILOTS[ANCHOR], int(d)) for d in DELAYS], axis=0)  # [3, 32]
    energy_10 = float(np.sum(np.abs(SC_PILOTS[ANCHOR]) ** 2)) * prec_c
    post_var_c = pv_c / (1.0 + pv_c * energy_10)
    init_prec_mean = pm_c / pv_c
    ev_base = prior_lw + np.log(post_var_c / pv_c) - (pm_c.real ** 2 + pm_c.imag ** 2) / pv_c

    # Vectorize across all `n_hyps` hypotheses simultaneously: [H_hyp, 3, 32] delayed pilots
    delayed_all = np.broadcast_to(delayed_10_base[None, :, :], (n_hyps, N_RAYS, N)).copy()
    for idx_h, hyp in enumerate(shift_bank, start=1):
        delayed_all[idx_h, int(hyp.ray)] *= rot_anchor_by_od[(hyp.onset_row, hyp.delta)]
    delayed_conj_prec_all = delayed_all.conj() * prec_c

    # [H_hyp, 3, 128]
    lw_b = np.broadcast_to(prior_lw[None, :, :], (n_hyps, N_RAYS, len(FREQ))).copy()
    mc_b = np.broadcast_to(pm_c[None, :, :], (n_hyps, N_RAYS, len(FREQ))).copy()
    w_init = np.exp(lw_b)
    # per_ray_b: [H_hyp, 3, 32]
    per_ray_b = np.einsum("kra,an->krn", w_init * mc_b, rot_anchor_ev) * delayed_all
    total_resp_b = per_ray_b.sum(axis=1)  # [H_hyp, 32]
    rot_anchor_conj = rot_anchor_ev.conj()

    for _ in range(SWEEPS):
        for ray in range(N_RAYS):
            resid_b = y10_sc[None, :] - total_resp_b + per_ray_b[:, ray, :]  # [H_hyp, 32]
            matched_b = (delayed_conj_prec_all[:, ray, :] * resid_b) @ rot_anchor_conj.T  # [H_hyp, 128]
            nat_m = init_prec_mean[ray][None, :] + matched_b
            pv_r = post_var_c[ray][None, :]
            m_r = pv_r * nat_m
            mc_b[:, ray, :] = m_r
            ev_b = ev_base[ray][None, :] + pv_r * (nat_m.real ** 2 + nat_m.imag ** 2)
            lse_b = logsumexp(ev_b, axis=1, keepdims=True)
            lw_r = ev_b - lse_b
            lw_b[:, ray, :] = lw_r
            new_pr = ((np.exp(lw_r) * m_r) @ rot_anchor_ev) * delayed_all[:, ray, :]
            total_resp_b = total_resp_b - per_ray_b[:, ray, :] + new_pr
            per_ray_b[:, ray, :] = new_pr

    # Compute cancellation-free within+between moments on `aux_rows` for all `n_hyps`
    w_b = np.exp(lw_b)                  # [H_hyp, 3, 128]
    m_phys_b = mc_b * amp_s             # [H_hyp, 3, 128]
    v_phys_b = post_var_c * ref_var     # [3, 128]
    mean_aux_flat = (w_b * m_phys_b) @ rot_aux_ev  # [H_hyp, 3, R*32]
    var_aux_flat = _within_between_aux_numba(w_b, m_phys_b, v_phys_b, rot_aux_ev, mean_aux_flat)

    mean_aux_4d = mean_aux_flat.reshape(n_hyps, N_RAYS, r_aux, N)
    var_aux_4d = var_aux_flat.reshape(n_hyps, N_RAYS, r_aux, N)

    # Apply M_a(t) rotation on aux_rows for shift hypotheses
    t_aux = ABS_TIME[aux_list, :]  # [R, 32]
    for idx_h, hyp in enumerate(shift_bank, start=1):
        tau_b = float(FRAME_DURATION) + 1.5 * float(hyp.onset_row)
        dt_aux = np.maximum(t_aux - tau_b, 0.0)
        mean_aux_4d[idx_h, int(hyp.ray)] *= np.exp(2j * np.pi * float(hyp.delta) * dt_aux)

    g_aux_batch = np.transpose(mean_aux_4d, (0, 2, 1, 3))  # [H_hyp, R, 3, 32]
    v_aux_batch = 1.0 + var_aux_4d.sum(axis=1)             # [H_hyp, R, 32]
    aux_scores_mat = score_3ray_payload_fast(g_aux_batch, v_aux_batch, y_aux_c[None, :, :])  # [H_hyp, R]
    aux_lls = aux_scores_mat.sum(axis=1)

    s_all = pilot_lls + float(aux_temp) * aux_lls
    s_c = float(s_all[0])
    s_shifts = s_all[1:].copy()

    # 3. Parabolic delta refinement on winning shift, recomputing BOTH state and evidence S_h
    best_rel_idx = int(np.argmax(s_shifts))
    best_hyp = shift_bank[best_rel_idx]
    refined_delta = float(best_hyp.delta)

    if use_parabolic:
        # Locate local delta index within same (onset_row, ray) slice of length N_DELTAS
        local_idx = best_rel_idx % N_DELTAS
        half = N_DELTAS // 2
        is_neg_int = 1 <= local_idx <= (half - 2)
        is_pos_int = (half + 1) <= local_idx <= (N_DELTAS - 2)
        if is_neg_int or is_pos_int:
            f0 = float(s_shifts[best_rel_idx])
            dm = float(s_shifts[best_rel_idx - 1]) - f0
            dp = float(s_shifts[best_rel_idx + 1]) - f0
            denom = dm + dp
            if math.isfinite(denom) and denom < -1e-9:
                step = float(np.clip(0.5 * DELTA_STEP * (dm - dp) / denom, -0.5 * DELTA_STEP, 0.5 * DELTA_STEP))
                cand_delta = float(best_hyp.delta + step)
                s_ref = _score_one_inherited_hypothesis(
                    a_prior_c,
                    y10_c,
                    aux_rows,
                    y_aux_c,
                    changed_ray=best_hyp.ray,
                    delta_freq=cand_delta,
                    onset_row=best_hyp.onset_row,
                    aux_temp=aux_temp,
                )
                if math.isfinite(s_ref) and s_ref > f0:
                    s_shifts[best_rel_idx] = s_ref
                    refined_delta = cand_delta

    log_third = math.log(1.0 / 3.0)
    b_c = s_c + log_third
    b_s = log_third + float(logsumexp(s_shifts - math.log(float(n_shifts))))
    s_r = float(reset_base.class_log_evidence_canon)
    b_r = s_r + log_third

    return BranchEvidence(
        b_c=float(b_c),
        b_s=float(b_s),
        b_r=float(b_r),
        s_c=s_c,
        s_s_best=float(s_shifts[best_rel_idx]),
        s_r=s_r,
        best_shift=best_hyp,
        refined_delta=refined_delta,
        reset_base=reset_base,
        a_prior=a_prior,
    )


def warmup_payload_numba() -> None:
    """Trigger Numba JIT compilation cleanly before timing benchmarks."""
    g = np.zeros((1, N_RAYS, N), dtype=COMPLEX_DTYPE)
    v = np.ones((1, N), dtype=REAL_DTYPE)
    y = np.zeros((1, N), dtype=COMPLEX_DTYPE)
    score_3ray_payload_fast(g, v, y)
    w1 = np.ones((1, 1), dtype=REAL_DTYPE)
    z1 = np.zeros((1, 1), dtype=COMPLEX_DTYPE)
    _compute_scale_quad_terms_numba(
        np.ones(1, dtype=REAL_DTYPE),
        np.ones(1, dtype=REAL_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        np.ones(1, dtype=REAL_DTYPE),
        np.ones(1, dtype=REAL_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        np.zeros(1, dtype=COMPLEX_DTYPE),
        1.0, 0.0, 0.0, 0.75, 0.17, 0.08,
    )
    _marginalize_scale_moments_numba(w1, z1, z1, z1, w1, z1, z1, w1, z1, w1)
    wb = np.ones((1, N_RAYS, 1), dtype=REAL_DTYPE)
    mb = np.zeros((1, N_RAYS, 1), dtype=COMPLEX_DTYPE)
    vb = np.ones((N_RAYS, 1), dtype=REAL_DTYPE)
    rot_e = np.ones((1, 1), dtype=COMPLEX_DTYPE)
    mf = np.zeros((1, N_RAYS, 1), dtype=COMPLEX_DTYPE)
    _within_between_aux_numba(wb, mb, vb, rot_e, mf)
    gl3_frequency_bank()
    epoch30_projection_kernel()
    default_scale_quadrature()


__all__ = [
    "ABS_TIME",
    "ANCHOR",
    "BlockFitResult",
    "BranchEvidence",
    "DEFAULT_SCALE_QUAD_ORDER",
    "DELTAS",
    "FINAL_NODES",
    "GL3Bank",
    "GL3_ORDER",
    "JointResetState",
    "N_DELTAS",
    "N_RAYS",
    "ONSET_ROWS",
    "ORIGIN",
    "OUT",
    "PayloadPosterior",
    "ResetBase",
    "ShiftHypothesis",
    "ShiftedBranchDynamics",
    "U_REL",
    "b_only_canonical_gauge",
    "build_frame_a_prior_clean",
    "coarse_payload_score",
    "condition_sc_weighted_shifted",
    "default_scale_quadrature",
    "epoch30_projection_kernel",
    "estimate_payload_posterior",
    "estimate_payload_posterior_obs",
    "estimate_payload_posterior_raw",
    "fit_anchor_only",
    "fit_known_blocks_by_node",
    "frequency_geometry_cache",
    "gl3_frequency_bank",
    "parity_block_cov_score",
    "project_joint_moments",
    "project_joint_moments_canon",
    "project_joint_probe",
    "project_to_tracker_state",
    "refit_reset",
    "sample_moments_b_stable_rows",
    "scale_tracker_state",
    "score_3ray_payload_fast",
    "score_branches_common_units",
    "shift_hypothesis_bank",
    "split_joint_reset_ray0_sign",
    "stratified_mode_midpoints",
    "warmup_payload_numba",
]
