"""P11: Off-Grid Continuous-Delay Cross-Waveform Physics Simulator,
Blind Frame-A Off-Grid Multi-Ray Estimator & Per-Ray Frame-B Branch Builder.

Strictly enforces:
  - Exact unitary centered-DFT fractional cyclic-delay operators from p11.contract.
  - ZERO hardcoded N0 anywhere (always explicit n0 / n0_from_snr(snr_db)).
  - ZERO access to obs.y_b[PROBE_ROW] (masked to 0 before any calculation).
  - Exact end-to-end C* = R+ x U(1) invariance/equivariance (< 1e-12).
  - Exact ray-permutation equivariance (< 1e-12) across unordered discovered ray atoms.
  - Float64 / complex128 arithmetic throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from p8 import link as ofdm_link
from p8.coding import qam_llrs
from p8.link import EVEN, clock, complex_normal, stream, traffic
from p8c.link import PILOTS as SC_PILOTS
from p11.contract import (
    CP,
    FRAME_DURATION,
    LINK_DURATION,
    N,
    OFFGRID_ALL_REGIMES,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    OFFGRID_RAY_COUNTS,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    OFDM_PILOT_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    T,
    GateOutput,
    OffGridObservation,
    OffGridTruth,
    build_offgrid_sc_channel_matrix,
    fractional_delay_freq_response,
    fractional_delay_matrix,
    n0_from_snr,
    signed_subcarrier_indices,
)


K_HAT = 5
MIN_DELAY_SEP = 0.75
ID_RAY_COUNTS = (2, 3, 4)
OOD_FIVE_RAY_COUNT = 5
MIDFRAME_SHIFT_ROWS = tuple(range(5, 15))
BURST_NOISE_MULT = 10.0
BASE_LOGIT_CLIP = 25.0

BLOCK_LEN = N + CP
ROW_STEP = float(BLOCK_LEN) / float(N)  # 1.5 useful symbols per CP-guarded block

# Physical mobile Doppler passband & 4-pilot comb (0, 9, 10, 19) grating-lobe spacing.
NU_MIN_BAND = 0.028
NU_MAX_BAND = 0.145
GRATING_LOBE_DF = 1.0 / 14.55  # ~0.0687285 cycles/symbol for dt = [-14.25, -0.75, +0.75, +14.25]

# --- Precomputed Static Geometry for Frame A Blind Off-Grid Discovery ---
_K_SIGNED = signed_subcarrier_indices(N)
_K_EVEN = _K_SIGNED[EVEN]  # [16] float64
_TIME_A_FULL = clock(T, start_row=0)  # [T, N] float64
_TIME_B_FULL = clock(T, start_row=T)  # [T, N] float64
_T_BAR_PILOT_A = _TIME_A_FULL[list(OFDM_PILOT_ROWS)].mean(axis=-1)  # [4] float64
_T_A_MEAN = float(_T_BAR_PILOT_A.mean())
_DT_PILOT_A = _T_BAR_PILOT_A - _T_A_MEAN  # [-14.25, -0.75, +0.75, +14.25]


def _diric_factor(nu: float | np.ndarray, n: int = N) -> np.ndarray:
    """Exact diagonal FFT attenuation (1/N) * sum_{m=0}^{N-1} exp(2j*pi*nu*(m-(N-1)/2)/N)."""
    x = np.asarray(nu, dtype=np.float64)
    num = np.sin(np.pi * x)
    den = float(n) * np.sin(np.pi * x / float(n))
    safe = np.where(np.abs(den) > 1e-14, den, 1.0)
    return np.where(np.abs(den) > 1e-14, num / safe, 1.0)


_TAU_COARSE = np.linspace(OFFGRID_DELAY_MIN, OFFGRID_DELAY_MAX, 45, dtype=np.float64)
_NU_POS = np.linspace(NU_MIN_BAND, NU_MAX_BAND, 24, dtype=np.float64)
_NU_COARSE = np.concatenate((-_NU_POS[::-1], _NU_POS))  # [48] float64 symmetric mobile passband
_DIRIC_COARSE_SQ = (_diric_factor(_NU_COARSE, n=N) ** 2)[:, None]  # [48, 1] physical ICI weighting
_PHI_F_COARSE = np.exp(-2j * np.pi * np.outer(_K_EVEN, _TAU_COARSE) / float(N)) / math.sqrt(float(len(EVEN)))
_PHI_T_COARSE = np.exp(2j * np.pi * np.outer(_DT_PILOT_A, _NU_COARSE)) / math.sqrt(float(len(OFDM_PILOT_ROWS)))

# Symmetric local (dnu, chirp) hypothesis grid around each Frame-A atom's instantaneous Doppler nu_hat[k]
# (15 primary-lobe hypotheses + 6 grating-lobe backup hypotheses = 21 total per ray!).
_DNU_PRI = np.array([-0.0045, 0.0, 0.0045], dtype=np.float64)
_CHIRP_PRI = np.linspace(-0.0010, 0.0010, 5, dtype=np.float64)
_DG_P, _CG_P = np.meshgrid(_DNU_PRI, _CHIRP_PRI, indexing="ij")
_DNU_PRI_FLAT = _DG_P.ravel()      # [15]
_CHIRP_PRI_FLAT = _CG_P.ravel()    # [15]

_DNU_LOBE = np.array([-0.0025, 0.0025], dtype=np.float64)
_CHIRP_LOBE = np.array([-0.0005, 0.0, 0.0005], dtype=np.float64)
_DG_L, _CG_L = np.meshgrid(_DNU_LOBE, _CHIRP_LOBE, indexing="ij")
_DNU_LOBE_FLAT = _DG_L.ravel()     # [6]
_CHIRP_LOBE_FLAT = _CG_L.ravel()   # [6]
_H_INH = len(_DNU_PRI_FLAT) + len(_DNU_LOBE_FLAT)  # 21

# Symmetric zero-mean Doppler bank for 'clear' Frame-B-only extrapolation away from r_anchor.
_NU_CLR_MAG = np.linspace(0.035, 0.085, 8, dtype=np.float64)
_NU_CLR_BANK = np.concatenate((_NU_CLR_MAG, -_NU_CLR_MAG))  # [16] float64
_EYE_N = np.eye(N, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Exact Off-Grid Continuous-Delay Cross-Waveform Physical Simulator
# ---------------------------------------------------------------------------


def offgrid_ofdm_propagate(
    grid: np.ndarray,
    ray_gains: np.ndarray,
    delays: np.ndarray,
) -> np.ndarray:
    """Exact unitary off-grid OFDM propagation via build_offgrid_sc_channel_matrix."""
    g = np.asarray(grid, dtype=np.complex128)
    gains = np.asarray(ray_gains, dtype=np.complex128)
    d = np.asarray(delays, dtype=np.float64)
    tx_time = np.fft.ifft(g, axis=-1, norm="ortho")
    rx_time = np.empty_like(tx_time, dtype=np.complex128)
    for r in range(g.shape[0]):
        h_r = build_offgrid_sc_channel_matrix(gains[:, r, :], d, n=N)
        rx_time[r] = h_r @ tx_time[r]
    return np.fft.fft(rx_time, axis=-1, norm="ortho")


def offgrid_sc_propagate(
    grid: np.ndarray,
    ray_gains: np.ndarray,
    delays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact unitary off-grid CP-SC propagation returning (y_clean [T, N], H_mat [T, N, N])."""
    g = np.asarray(grid, dtype=np.complex128)
    gains = np.asarray(ray_gains, dtype=np.complex128)
    d = np.asarray(delays, dtype=np.float64)
    n_rows = g.shape[0]
    h_mats = np.empty((n_rows, N, N), dtype=np.complex128)
    rx_time = np.empty((n_rows, N), dtype=np.complex128)
    for r in range(n_rows):
        h_r = build_offgrid_sc_channel_matrix(gains[:, r, :], d, n=N)
        h_mats[r] = h_r
        rx_time[r] = h_r @ g[r]
    return rx_time, h_mats


def _sample_continuous_delays(rng: np.random.Generator, n_rays: int) -> np.ndarray:
    """Uniformly sample ordered continuous delays in [0.0, 6.5] with separation >= 0.75."""
    span = OFFGRID_DELAY_MAX - OFFGRID_DELAY_MIN - (n_rays - 1) * MIN_DELAY_SEP
    if span <= 0.0:
        raise ValueError(f"Cannot pack {n_rays} rays with min_sep={MIN_DELAY_SEP}")
    u = np.sort(rng.uniform(0.0, span, size=n_rays))
    return (OFFGRID_DELAY_MIN + u + np.arange(n_rays, dtype=np.float64) * MIN_DELAY_SEP).astype(np.float64)


def _draw_offgrid_channel(
    rng: np.random.Generator,
    n_rays: int,
    *,
    doppler_scale: float = 1.0,
    delays_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw (delays, powers, gains, freqs, chirps) for an off-grid continuous-delay channel.

    Uses a P8-consistent dominant-LOS exponential power profile (normalized to sum = 1.0) so
    multi-ray CP-SC channels operate in the informative 10%-65% BLER regime at PRIMARY_SNR_DB=9 dB
    rather than deep-null saturation, while keeping secondary rays clearly resolvable & consequential.
    """
    delays = (
        _sample_continuous_delays(rng, n_rays)
        if delays_override is None
        else np.asarray(delays_override, dtype=np.float64).copy()
    )
    idx = np.arange(n_rays, dtype=np.float64)
    raw_powers = np.exp(-0.90 * idx - 0.10 * (delays - delays[0]))
    powers = (raw_powers / float(raw_powers.sum())).astype(np.float64)

    phases = rng.uniform(-np.pi, np.pi, size=n_rays)
    cn = complex_normal(rng, n_rays)
    # Dominant ray constant-modulus, secondary rays mild Rician (K=2, exact E[|g_l|^2] = powers[l]).
    gains = np.sqrt(powers) * (math.sqrt(2.0 / 3.0) * np.exp(1j * phases) + math.sqrt(1.0 / 3.0) * cn)
    gains[0] = math.sqrt(float(powers[0])) * np.exp(1j * float(phases[0]))

    f_lo = 0.035 * float(doppler_scale)
    f_hi = 0.085 * float(doppler_scale)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_rays)
    f_start = rng.uniform(f_lo, f_hi, size=n_rays) * signs
    f_end = rng.uniform(f_lo, f_hi, size=n_rays) * signs
    chirps = ((f_end - f_start) / float(LINK_DURATION)).astype(np.float64)
    return delays, powers, gains.astype(np.complex128), f_start.astype(np.float64), chirps


def eval_piecewise_ray_phase_and_gain(
    times: np.ndarray,
    gains_a: np.ndarray,
    freqs_a: np.ndarray,
    chirps_a: np.ndarray,
    *,
    changed_rays: Sequence[int] = (),
    t_shift: Optional[float] = None,
    freqs_post: Optional[np.ndarray] = None,
    chirps_post: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate exact phase-continuous ray phases phi[P, ...] and gains g[P, ...] at arbitrary times."""
    t_arr = np.asarray(times, dtype=np.float64)
    p = len(gains_a)
    phi = np.empty((p,) + t_arr.shape, dtype=np.float64)
    for l_idx in range(p):
        phi[l_idx] = 2.0 * np.pi * (freqs_a[l_idx] * t_arr + 0.5 * chirps_a[l_idx] * (t_arr ** 2))

    if changed_rays and t_shift is not None and freqs_post is not None and chirps_post is not None:
        ts = float(t_shift)
        dt = t_arr - ts
        mask = dt >= 0.0
        dt_pos = np.where(mask, dt, 0.0)
        for l_idx in changed_rays:
            phi_ts = 2.0 * np.pi * (freqs_a[l_idx] * ts + 0.5 * chirps_a[l_idx] * (ts ** 2))
            phi_after = phi_ts + 2.0 * np.pi * (
                freqs_post[l_idx] * dt_pos + 0.5 * chirps_post[l_idx] * (dt_pos ** 2)
            )
            phi[l_idx] = np.where(mask, phi_after, phi[l_idx])

    g = gains_a.reshape((p,) + (1,) * t_arr.ndim) * np.exp(1j * phi)
    return phi, g


def simulate_offgrid_link(
    seed: int,
    regime: str,
    snr_db: float = PRIMARY_SNR_DB,
    anchor_row: Optional[int] = None,
) -> Tuple[OffGridObservation, OffGridTruth]:
    """Generate one strictly separated (OffGridObservation, OffGridTruth) OFDM(A) -> CP-SC(B) link."""
    if regime not in OFFGRID_ALL_REGIMES:
        raise ValueError(f"Unknown off-grid regime '{regime}'. Expected one of {OFFGRID_ALL_REGIMES}.")
    n0 = n0_from_snr(snr_db)

    rng_ch = stream(seed, 0)
    rng_reg = stream(seed, 8)
    rng_anchor = stream(seed, 9)

    # 1. Sample variable ray count P and continuous off-grid channel geometry.
    if regime == "offgrid_ood_five_ray_fast":
        n_rays = OOD_FIVE_RAY_COUNT
        doppler_scale = 1.5
    else:
        n_rays = int(rng_ch.choice(ID_RAY_COUNTS))
        doppler_scale = 1.0

    delays, powers, gains_a, freqs_a, chirps_a = _draw_offgrid_channel(
        rng_ch, n_rays, doppler_scale=doppler_scale
    )

    # 2. Sample roving sparse training row r_anchor in OFFGRID_ANCHOR_ROWS.
    if anchor_row is None:
        r_anchor = int(rng_anchor.choice(OFFGRID_ANCHOR_ROWS))
    else:
        r_anchor = int(anchor_row)
        if r_anchor == PROBE_ROW or not (0 <= r_anchor < T):
            raise ValueError(f"Invalid anchor_row={r_anchor}")
    training_rows = (r_anchor,)
    data_rows = tuple(r for r in range(T) if r not in (r_anchor, PROBE_ROW))

    # 3. Simulate Frame A (OFDM) on exact continuous delays.
    a_words = traffic(stream(seed, 1))
    tx_grid_a = ofdm_link.tx_grid(a_words, OFDM_PILOT_ROWS)
    _, g_a = eval_piecewise_ray_phase_and_gain(_TIME_A_FULL, gains_a, freqs_a, chirps_a)
    y_a_clean = offgrid_ofdm_propagate(tx_grid_a, g_a, delays)
    a_noise = np.fft.fft(math.sqrt(n0) * complex_normal(stream(seed, 2), (T, N)), axis=-1, norm="ortho")
    y_a = y_a_clean + a_noise
    pilot_tx_a = ofdm_link.PILOTS[np.ix_(list(OFDM_PILOT_ROWS), EVEN)].copy()

    # 4. Build Frame B (CP-SC) transmit grid: 1 roving anchor + 1 withheld probe + 18 LDPC data rows.
    words_full = traffic(stream(seed, 3))
    b_info = words_full.info[: len(data_rows)].copy()
    b_code = words_full.code[: len(data_rows)].copy()
    b_symbols = words_full.symbols[: len(data_rows)].copy()

    tx_grid_b = np.empty((T, N), dtype=np.complex128)
    tx_grid_b[r_anchor] = SC_PILOTS[r_anchor]
    tx_grid_b[PROBE_ROW] = SC_PILOTS[PROBE_ROW]
    tx_grid_b[list(data_rows)] = b_symbols
    pilot_tx_b = SC_PILOTS[list(training_rows)].copy()

    # 5. Evaluate Frame B ray trajectory according to regime.
    freqs_b = freqs_a.copy()
    chirps_b = chirps_a.copy()
    changed_rays: Tuple[int, ...] = ()
    shift_row: Optional[int] = None
    corrupted_rows: Tuple[int, ...] = ()

    if regime == "offgrid_continuous":
        _, g_b = eval_piecewise_ray_phase_and_gain(_TIME_B_FULL, gains_a, freqs_a, chirps_a)

    elif regime == "offgrid_full_reset":
        _, _, gains_b, freqs_b, chirps_b = _draw_offgrid_channel(
            stream(seed, 5), n_rays, doppler_scale=1.0, delays_override=delays
        )
        _, g_b = eval_piecewise_ray_phase_and_gain(_TIME_B_FULL, gains_b, freqs_b, chirps_b)
        changed_rays = tuple(range(n_rays))
        shift_row = 0

    elif regime in (
        "offgrid_partial_shift",
        "offgrid_midframe_shift",
        "offgrid_ood_burst_and_shift",
        "offgrid_ood_five_ray_fast",
    ):
        # Strictly subset of rays (< n_rays) shifts Doppler with exact phase continuity!
        # Weight selection mildly toward strong rays so shifts are physically consequential.
        n_changed = 1 if n_rays == 2 else int(rng_reg.choice([1, 2]))
        changed_arr = np.sort(rng_reg.choice(n_rays, size=n_changed, replace=False))
        changed_rays = tuple(int(x) for x in changed_arr)

        if regime == "offgrid_midframe_shift":
            shift_row = int(rng_reg.choice(MIDFRAME_SHIFT_ROWS))
        else:
            shift_row = 0

        t_shift = float(T + shift_row) * ROW_STEP
        f_lo = 0.035 * doppler_scale
        f_hi = 0.085 * doppler_scale
        rem_dur = max(float(2 * T - (T + shift_row)) * ROW_STEP, 1.0)

        for l_idx in changed_rays:
            f_at_shift = float(freqs_a[l_idx] + chirps_a[l_idx] * t_shift)
            old_sign = 1.0 if f_at_shift >= 0.0 else -1.0
            new_sign = -old_sign
            f_new_start = float(rng_reg.uniform(f_lo, f_hi) * new_sign)
            f_new_end = float(rng_reg.uniform(f_lo, f_hi) * new_sign)
            freqs_b[l_idx] = f_new_start
            chirps_b[l_idx] = (f_new_end - f_new_start) / rem_dur

        _, g_b = eval_piecewise_ray_phase_and_gain(
            _TIME_B_FULL,
            gains_a,
            freqs_a,
            chirps_a,
            changed_rays=changed_rays,
            t_shift=t_shift,
            freqs_post=freqs_b,
            chirps_post=chirps_b,
        )

        if regime == "offgrid_ood_burst_and_shift":
            burst_arr = np.sort(rng_reg.choice(np.array(data_rows, dtype=np.int64), size=2, replace=False))
            corrupted_rows = tuple(int(x) for x in burst_arr)
    else:
        raise ValueError(f"Unhandled off-grid regime: {regime}")

    # 6. Exact CP-SC propagation & additive noise (with 10x noise spike on blind OOD burst rows).
    y_b_clean, h_b_true_mat = offgrid_sc_propagate(tx_grid_b, g_b, delays)
    b_noise = math.sqrt(n0) * complex_normal(stream(seed, 4), (T, N))
    if corrupted_rows:
        b_noise[list(corrupted_rows)] = math.sqrt(BURST_NOISE_MULT) * b_noise[list(corrupted_rows)]
    y_b_noisy = y_b_clean + b_noise

    # Mask withheld probe row strictly to NaN+1j*NaN in PublicObservation.
    y_b_pub = y_b_noisy.copy()
    y_b_pub[PROBE_ROW] = np.nan + 1j * np.nan

    obs = OffGridObservation(
        y_a=y_a,
        y_b=y_b_pub,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=float(n0),
        pilot_tx_b=pilot_tx_b,
        pilot_tx_a=pilot_tx_a,
    )
    truth = OffGridTruth(
        regime=regime,
        n_rays=n_rays,
        delays=delays,
        powers=powers,
        freqs_a=freqs_a,
        freqs_b=freqs_b,
        chirps=chirps_b,
        changed_rays=changed_rays,
        shift_row=shift_row,
        corrupted_rows=corrupted_rows,
        data_rows=data_rows,
        b_info=b_info,
        b_code=b_code,
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_clean[PROBE_ROW].copy(),
        h_b_true_mat=h_b_true_mat,
    )
    return obs, truth


# ---------------------------------------------------------------------------
# Strictly Blind Off-Grid Frame-A Multi-Ray OMP + Continuous GN/RELAX Estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffGridAtomEstimates:
    """Blind Frame-A discovered continuous ray atoms (zero oracle ray count or delays)."""

    tau_hat: np.ndarray        # [K_hat] float64 continuous delays in [0.0, 6.5]
    nu_hat: np.ndarray         # [K_hat] float64 continuous primary Dopplers
    g_a_mid: np.ndarray        # [K_hat] complex128 estimated ray gain at Frame-A midpoint t_a_mean
    v_a_mid: np.ndarray        # [K_hat] float64 marginal estimation variance at t_a_mean
    atom_snr_a: np.ndarray     # [K_hat] float64 strictly C*-invariant Frame-A atom SNR
    p_atom_a: np.ndarray       # [K_hat] float64 shrunk Frame-A prior ray power
    p_tot_a: float             # Frame-A estimated total signal power (scales as |alpha|^2)
    # Exact (dnu, chirp) Frame-A posterior bank [K_hat, H_inh] capturing quadratic chirp phase!
    nu_bank_a: np.ndarray      # [K_hat, H_inh] float64 Doppler hypotheses around nu_hat[k]
    chirp_bank_a: np.ndarray   # [K_hat, H_inh] float64 chirp hypotheses (cycles / sym^2)
    log_w_bank_a: np.ndarray   # [K_hat, H_inh] float64 exact Frame-A posterior log-weights
    mu_bank_a: np.ndarray      # [K_hat, H_inh] complex128 Frame-A posterior complex gain per hypothesis
    var_bank_a: np.ndarray     # [K_hat, H_inh] float64 Frame-A posterior variance per hypothesis


def _atom_vec_a(tau: float, nu: float) -> np.ndarray:
    """Unit-norm 64-element separable (time x even-freq) Frame-A atom referenced at t_a_mean."""
    af = np.exp(-2j * np.pi * _K_EVEN * float(tau) / float(N)) / math.sqrt(float(len(EVEN)))
    at = np.exp(2j * np.pi * _DT_PILOT_A * float(nu)) / math.sqrt(float(len(OFDM_PILOT_ROWS)))
    return np.outer(at, af).reshape(-1)


def _clamp_nu_passband(nu: float, sign_ref: float) -> float:
    """Clamp continuous Doppler into the symmetric mobile passband [NU_MIN_BAND, NU_MAX_BAND]."""
    sgn = 1.0 if sign_ref >= 0.0 else -1.0
    mag = float(np.clip(abs(nu), NU_MIN_BAND, NU_MAX_BAND))
    return sgn * mag


def _refine_atom_2d(
    res_mat: np.ndarray,
    tau_init: float,
    nu_init: float,
    existing_taus: Sequence[float],
    min_sep: float = 0.45,
    n_iters: int = 3,
) -> Tuple[float, float]:
    """Continuous sub-grid 2D Newton refinement maximizing |a_t(nu)^H R a_f(tau)^*|^2."""
    tau = float(tau_init)
    nu = float(nu_init)
    sign_ref = 1.0 if nu >= 0.0 else -1.0
    c_f = -2.0 * np.pi * _K_EVEN / float(N)
    c_t = 2.0 * np.pi * _DT_PILOT_A
    inv_norm = 1.0 / math.sqrt(float(len(EVEN) * len(OFDM_PILOT_ROWS)))

    def _eval_obj(t_val: float, f_val: float) -> Tuple[float, complex, np.ndarray, np.ndarray]:
        af = np.exp(1j * c_f * t_val)
        at = np.exp(1j * c_t * f_val)
        z = complex((at.conj() @ res_mat @ af.conj()) * inv_norm)
        return float(z.real * z.real + z.imag * z.imag), z, at, af

    obj, z, at, af = _eval_obj(tau, nu)

    for _ in range(n_iters):
        at_c = at.conj()
        af_c = af.conj()
        r_af = res_mat @ af_c
        r_daf = res_mat @ (-1j * c_f * af_c)
        r_d2af = res_mat @ (-(c_f * c_f) * af_c)

        z_t = complex((at_c @ r_daf) * inv_norm)
        z_f = complex(((-1j * c_t * at_c) @ r_af) * inv_norm)
        z_tt = complex((at_c @ r_d2af) * inv_norm)
        z_ff = complex(((-(c_t * c_t) * at_c) @ r_af) * inv_norm)
        z_tf = complex(((-1j * c_t * at_c) @ r_daf) * inv_norm)

        zc = z.conjugate()
        g_t = 2.0 * float((zc * z_t).real)
        g_f = 2.0 * float((zc * z_f).real)
        h_tt = 2.0 * float((z_t.conjugate() * z_t + zc * z_tt).real)
        h_ff = 2.0 * float((z_f.conjugate() * z_f + zc * z_ff).real)
        h_tf = 2.0 * float((z_t.conjugate() * z_f + zc * z_tf).real)

        det = h_tt * h_ff - h_tf * h_tf
        if h_tt < -1e-12 and h_ff < -1e-12 and det > 1e-12:
            d_tau = -(h_ff * g_t - h_tf * g_f) / det
            d_nu = -(-h_tf * g_t + h_tt * g_f) / det
        else:
            d_tau = g_t / max(abs(h_tt), 1.0) if h_tt < 0.0 else (0.03 * math.copysign(1.0, g_t) if g_t != 0.0 else 0.0)
            d_nu = g_f / max(abs(h_ff), 1.0) if h_ff < 0.0 else (0.0015 * math.copysign(1.0, g_f) if g_f != 0.0 else 0.0)

        d_tau = float(np.clip(d_tau, -0.15, 0.15))
        d_nu = float(np.clip(d_nu, -0.006, 0.006))
        if abs(d_tau) < 1e-10 and abs(d_nu) < 1e-11:
            break

        improved = False
        for step in (1.0, 0.5):
            cand_t = float(np.clip(tau + step * d_tau, OFFGRID_DELAY_MIN, OFFGRID_DELAY_MAX))
            cand_f = _clamp_nu_passband(nu + step * d_nu, sign_ref)
            if any(abs(cand_t - ot) < min_sep for ot in existing_taus):
                continue
            cand_obj, cand_z, cand_at, cand_af = _eval_obj(cand_t, cand_f)
            if cand_obj > obj * (1.0 + 1e-12):
                tau, nu = cand_t, cand_f
                obj, z, at, af = cand_obj, cand_z, cand_at, cand_af
                improved = True
                break
        if not improved:
            break

    return tau, nu


def _strict_int(val: Any, name: str) -> int:
    if isinstance(val, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(val, (int, np.integer)):
        raise TypeError(f"{name} must be an exact integer, got {type(val).__name__}: {val!r}")
    return int(val)


def _validate_n0(val: Any, name: str = "obs.n0") -> float:
    if isinstance(val, (bool, np.bool_, np.timedelta64, np.datetime64, complex, np.complexfloating, str, bytes)):
        raise TypeError(f"{name} must be a positive finite real number, got {type(val).__name__}")
    out = float(val)
    if out <= 0.0 or not math.isfinite(out):
        raise ValueError(f"Expected positive finite {name}, got {val!r}")
    return out


def discover_frame_a_atoms(
    obs: OffGridObservation,
    k_hat: int = K_HAT,
) -> OffGridAtomEstimates:
    """Strictly blind Frame-A off-grid multi-ray discovery (OMP + 2D Newton + RELAX + multi-lobe bank).

    Uses ONLY obs.y_a, obs.pilot_tx_a, and obs.n0 (NEVER touches obs.y_b or hidden truth).
    Normalizes by sqrt(obs.n0) internally so discovered (tau_hat, nu_hat) and atom_snr_a are
    EXACTLY C*-invariant (< 1e-13) across arbitrary complex gain scaling alpha in C*.
    """
    k_i = _strict_int(k_hat, "k_hat")
    if k_i < 1:
        raise ValueError(f"k_hat must be >= 1, got {k_i}")
    n0 = _validate_n0(obs.n0, "obs.n0")
    s_mag = math.sqrt(n0)
    s_var = n0

    y_p = np.asarray(obs.y_a[np.ix_(list(OFDM_PILOT_ROWS), EVEN)], dtype=np.complex128)  # [4, 16]
    pilot_a = np.asarray(obs.pilot_tx_a, dtype=np.complex128)
    if not np.all(np.isfinite(y_p)):
        raise ValueError("obs.y_a pilot cells contain non-finite values")
    if pilot_a.shape != y_p.shape or not np.all(np.isfinite(pilot_a)):
        raise ValueError(f"obs.pilot_tx_a must be finite with shape {y_p.shape}, got {pilot_a.shape}")
    if np.any(np.abs(pilot_a) == 0.0):
        raise ValueError("obs.pilot_tx_a must be strictly non-zero at all pilot cells")
    h_ls = y_p / pilot_a                                 # [4, 16]
    h_u_raw = h_ls / s_mag                               # Canonical unit-noise scale -> pure U(1) remains
    # Canonical global phase gauge alignment: de-rotate by largest entry's unit phasor so OMP/Newton/RELAX
    # runs on strictly C*-invariant numbers (< 1e-14), then restore exact U(1) phase on complex gain outputs!
    h_flat_raw = h_u_raw.reshape(-1)
    idx_max = int(np.argmax(h_flat_raw.real ** 2 + h_flat_raw.imag ** 2))
    mag_max = float(abs(h_flat_raw[idx_max]))
    u_phase = complex(h_flat_raw[idx_max] / mag_max) if mag_max > 0.0 else (1.0 + 0.0j)
    h_u = h_u_raw * u_phase.conjugate()
    h_u_vec = h_u.reshape(-1)
    n_obs = h_u_vec.size  # 64

    # Stage 1: Greedy 2D OMP + continuous sub-grid Newton refinement.
    taus: list[float] = []
    nus: list[float] = []
    res_u = h_u.copy()
    d_cols: list[np.ndarray] = []

    for _ in range(int(k_hat)):
        surf = (np.abs(_PHI_T_COARSE.conj().T @ res_u @ _PHI_F_COARSE.conj()) ** 2) * _DIRIC_COARSE_SQ
        if taus:
            mask = np.ones(len(_TAU_COARSE), dtype=bool)
            for t_prev in taus:
                mask &= np.abs(_TAU_COARSE - t_prev) >= 0.65
            if np.any(mask):
                surf[:, ~mask] = -1.0

        flat_idx = int(np.argmax(surf))
        i_nu, i_tau = divmod(flat_idx, len(_TAU_COARSE))
        tau_ref, nu_ref = _refine_atom_2d(
            res_u, float(_TAU_COARSE[i_tau]), float(_NU_COARSE[i_nu]), taus, min_sep=0.60, n_iters=3
        )
        taus.append(tau_ref)
        nus.append(nu_ref)
        d_cols.append(_atom_vec_a(tau_ref, nu_ref))

        d_mat = np.column_stack(d_cols)
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
        res_u = (h_u_vec - d_mat @ coeffs).reshape(len(OFDM_PILOT_ROWS), len(EVEN))

    # Stage 2: 1-pass cyclic RELAX super-resolution joint refinement.
    d_mat = np.column_stack(d_cols)
    gram = d_mat.conj().T @ d_mat
    gram[np.diag_indices_from(gram)] += 1e-10
    coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
    for j in range(int(k_hat)):
        res_j = (h_u_vec - (d_mat @ coeffs - coeffs[j] * d_mat[:, j])).reshape(
            len(OFDM_PILOT_ROWS), len(EVEN)
        )
        others = [taus[i] for i in range(int(k_hat)) if i != j]
        tau_j, nu_j = _refine_atom_2d(res_j, taus[j], nus[j], others, min_sep=0.60, n_iters=3)
        taus[j] = tau_j
        nus[j] = nu_j
        d_mat[:, j] = _atom_vec_a(tau_j, nu_j)
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)

    tau_arr = np.asarray(taus, dtype=np.float64)
    nu_arr = np.asarray(nus, dtype=np.float64)

    # Physical unit-gain Frame-A response columns (including exact intra-symbol Dirichlet factor).
    diric_vals = _diric_factor(nu_arr, n=N)
    scale_phys = math.sqrt(float(n_obs)) * diric_vals
    d_phys = d_mat * scale_phys[None, :]  # [64, K_hat]

    # C*-equivariant effective pilot noise variance (scales strictly as |alpha|^2).
    p_ls_u = float(np.mean(np.abs(h_u) ** 2))
    p_tot_u = max(p_ls_u - 1.0, 0.25)
    sigma_h2_u = 1.0 + 0.015 * p_tot_u

    gram_p = d_phys.conj().T @ d_phys
    gram_p[np.diag_indices_from(gram_p)] += 1e-10
    gram_inv = np.linalg.inv(gram_p)
    g_ls_u = gram_inv @ (d_phys.conj().T @ h_u_vec)
    v_ls_u = np.maximum(np.real(np.diag(gram_inv)) * sigma_h2_u, 1e-12)

    atom_snr_a = (np.abs(g_ls_u) ** 2) / v_ls_u
    shrink = np.maximum(atom_snr_a - 5.5, 0.0) / (atom_snr_a + 1.0)
    g_mid_u = shrink * g_ls_u
    p_atom_u = np.maximum(np.abs(g_ls_u) ** 2 - 4.5 * v_ls_u, 0.04 * v_ls_u)
    v_mid_u = np.maximum(shrink * v_ls_u + (1.0 - shrink) * 0.04 * v_ls_u, 1e-12)

    # Stage 3: Exact (dnu, chirp) + Grating-Lobe Backup Frame-A Bayesian Bank (21 hypotheses/ray).
    # Primary lobe (15 dnu x chirp hypotheses) + 6 backup hypotheses on the nearest passband comb
    # grating lobe (nu_0 +/- GRATING_LOBE_DF), weighted by exact Frame-A marginal likelihood + smooth
    # physical Doppler/chirp prior so Frame B's roving anchor resolves BOTH chirp AND comb ambiguity!
    h_inh = _H_INH
    nu_bank_a = np.empty((int(k_hat), h_inh), dtype=np.float64)
    chirp_bank_a = np.empty((int(k_hat), h_inh), dtype=np.float64)
    log_w_bank_a = np.empty((int(k_hat), h_inh), dtype=np.float64)
    mu_bank_u = np.empty((int(k_hat), h_inh), dtype=np.complex128)
    var_bank_u = np.empty((int(k_hat), h_inh), dtype=np.float64)

    full_fit = d_phys @ g_ls_u
    dt_a = _DT_PILOT_A[None, :]           # [1, 4]
    dt_a2 = 0.5 * (dt_a ** 2)             # [1, 4]
    for j in range(int(k_hat)):
        res_j_vec = h_u_vec - (full_fit - g_ls_u[j] * d_phys[:, j])
        res_j_mat = res_j_vec.reshape(len(OFDM_PILOT_ROWS), len(EVEN))
        af_j = np.exp(-2j * np.pi * _K_EVEN * tau_arr[j] / float(N))
        z_t = (res_j_mat @ af_j.conj()) / float(len(EVEN))  # [4]
        v_zt = sigma_h2_u / float(len(EVEN))

        nu0 = float(nu_arr[j])
        # Pick the strongest passband grating lobe (nu0 - GRATING_LOBE_DF or nu0 + GRATING_LOBE_DF)
        lobe_cands = [nu0 - GRATING_LOBE_DF, nu0 + GRATING_LOBE_DF]
        valid_lobes = [nl for nl in lobe_cands if NU_MIN_BAND <= abs(nl) <= NU_MAX_BAND]
        if valid_lobes:
            # Prefer lower-|nu| passband grating lobe (rescues high-lobe misses like -0.127 -> -0.058!)
            nu_sec = min(valid_lobes, key=lambda x: abs(x))
        else:
            nu_sec = nu0 + (0.009 if nu0 >= 0.0 else -0.009)

        cands_nu = np.concatenate((nu0 + _DNU_PRI_FLAT, nu_sec + _DNU_LOBE_FLAT))
        cands_ch = np.concatenate((_CHIRP_PRI_FLAT, _CHIRP_LOBE_FLAT))
        cands_dnu_rel = np.concatenate((_DNU_PRI_FLAT, _DNU_LOBE_FLAT))
        nu_bank_a[j] = cands_nu
        chirp_bank_a[j] = cands_ch

        dir_h = _diric_factor(cands_nu, n=N)
        phase_a = 2.0 * np.pi * (cands_nu[:, None] * dt_a + cands_ch[:, None] * dt_a2)
        u_h = dir_h[:, None] * np.exp(1j * phase_a)  # [21, 4]
        energy_h = np.sum(np.abs(u_h) ** 2, axis=-1)
        p_j = float(p_atom_u[j])
        post_v = p_j / (1.0 + p_j * energy_h / v_zt)
        matched = u_h.conj() @ z_t
        nat_m = matched / v_zt
        post_m = post_v * nat_m

        # Physical Gaussian prior on frequency magnitude, local dnu, and chirp rate.
        prior_h = (
            -0.5 * ((cands_nu / 0.095) ** 2)
            - 0.5 * ((cands_dnu_rel / 0.005) ** 2)
            - 0.5 * ((cands_ch / 0.00045) ** 2)
        )
        ev_h = prior_h + np.log(post_v / p_j) + post_v * (np.abs(nat_m) ** 2)
        log_w_bank_a[j] = ev_h - float(logsumexp(ev_h))
        mu_bank_u[j] = shrink[j] * post_m
        var_bank_u[j] = np.maximum(shrink[j] * post_v + (1.0 - shrink[j]) * 0.04 * v_ls_u[j] + 0.01 * p_j, 1e-12)

    return OffGridAtomEstimates(
        tau_hat=tau_arr,
        nu_hat=nu_arr,
        g_a_mid=(g_mid_u * u_phase * s_mag).astype(np.complex128),
        v_a_mid=(v_mid_u * s_var).astype(np.float64),
        atom_snr_a=atom_snr_a.astype(np.float64),
        p_atom_a=(p_atom_u * s_var).astype(np.float64),
        p_tot_a=float(p_tot_u * s_var),
        nu_bank_a=nu_bank_a,
        chirp_bank_a=chirp_bank_a,
        log_w_bank_a=log_w_bank_a,
        mu_bank_a=(mu_bank_u * u_phase * s_mag).astype(np.complex128),
        var_bank_a=(var_bank_u * s_var).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Per-Ray Frame-B Candidate Branch Builder (Strict Pre-Update & Permutation-Equivariant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffGridBranchBundle:
    """Strictly public per-ray, per-row Frame-B ('inherit', 'clear') summaries & exact subset Bayes stats."""

    atoms: OffGridAtomEstimates
    k_hat: int
    tau_hat: np.ndarray            # [K_hat] float64
    nu_hat: np.ndarray             # [K_hat] float64 (post-anchor refined Doppler for transport connection)
    nu_b_proxy: np.ndarray         # [K_hat] float64 Frame-B-only C*-invariant Doppler proxy (for no_history)
    d_mats: np.ndarray             # [K_hat, N, N] complex128 unitary fractional-delay operators
    phi_f: np.ndarray              # [K_hat, N] complex128 unitary frequency responses
    u_probe: np.ndarray            # [K_hat, N] complex128 delayed known probe waveforms
    mean_inh: np.ndarray           # [K_hat, T, N] complex128 conditioned inherit ray means
    var_inh: np.ndarray            # [K_hat, T, N] float64 conditioned inherit ray variances
    mean_clr: np.ndarray           # [K_hat, T, N] complex128 Frame-B-only clear ray means
    var_clr: np.ndarray            # [K_hat, T, N] float64 Frame-B-only clear ray variances
    pre_pred_inh: np.ndarray       # [M_tr, N] complex128 strict pre-update inherit prediction on training rows
    pre_var_inh: np.ndarray        # [M_tr, N] float64 strict pre-update inherit variance on training rows
    pre_pred_clr: np.ndarray       # [M_tr, N] complex128 strict pre-update clear prediction on training rows
    pre_var_clr: np.ndarray        # [M_tr, N] float64 strict pre-update clear variance on training rows
    pre_ll_inh: float              # exact strict pre-update all-inherit log-likelihood
    pre_ll_clr: float              # exact strict pre-update all-clear log-likelihood
    subset_logliks: np.ndarray     # [2^K_hat] float64 exact strict pre-update subset log-likelihoods
    subset_posteriors: np.ndarray  # [2^K_hat] float64 exact pre-update subset posteriors
    w_ray_classical: np.ndarray    # [K_hat] float64 marginal per-ray classical trust weights
    base_logit_ray: np.ndarray     # [K_hat] float64 clipped logit of w_ray_classical
    llrs_inh: np.ndarray           # [T, N, 4] float64 all-inherit LMMSE bit LLRs
    llrs_clr: np.ndarray           # [T, N, 4] float64 all-clear LMMSE bit LLRs
    llrs_cls: np.ndarray           # [T, N, 4] float64 classical per-ray gate LMMSE bit LLRs
    probe_row: int
    training_rows: Tuple[int, ...]


def _compute_all_row_llrs(
    mean_ray: np.ndarray,
    var_ray: np.ndarray,
    d_mats: np.ndarray,
    y_b_safe: np.ndarray,
    n0: float,
    training_rows: Tuple[int, ...],
    probe_row: int,
    noise_mult: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact H_mix [T, N, N], noise_mix [T, N], and vectorized LMMSE bit LLRs [T, N, 4]."""
    k_hat = mean_ray.shape[0]
    h_mix = np.zeros((T, N, N), dtype=np.complex128)
    for k_idx in range(k_hat):
        h_mix += mean_ray[k_idx, :, :, None] * d_mats[k_idx][None, :, :]

    nm = np.ones(T, dtype=np.float64) if noise_mult is None else np.asarray(noise_mult, dtype=np.float64).reshape(T)
    n0_row = float(n0) * nm
    n0_row[int(probe_row)] = float(n0)
    noise_mix = np.maximum(n0_row[:, None] + var_ray.sum(axis=0), 1e-300)

    skip = set(int(r) for r in training_rows) | {int(probe_row)}
    data = np.array([r for r in range(T) if r not in skip], dtype=np.int64)
    llrs = np.zeros((T, N, 4), dtype=np.float64)
    if data.size == 0:
        return h_mix, noise_mix, llrs

    s_mag = math.sqrt(max(float(n0), 1e-300))
    s_var = s_mag * s_mag
    hd = h_mix[data] / s_mag                                       # [R, N, N] O(1) gauge-normalized
    nv = noise_mix[data] / s_var                                   # [R, N] O(1) gauge-normalized
    y_d = y_b_safe[data] / s_mag                                   # [R, N] O(1) gauge-normalized
    adj = hd.conj().swapaxes(-1, -2) / nv[:, None, :]              # [R, N, N]
    cov = np.linalg.inv(_EYE_N[None, :, :] + adj @ hd)             # [R, N, N]
    est = (cov @ (adj @ y_d[:, :, None]))[..., 0]                  # [R, N]
    var = np.diagonal(cov, axis1=-2, axis2=-1).real                # [R, N]
    llrs[data] = qam_llrs(est.ravel(), var.ravel()).reshape(data.size, N, 4)
    return h_mix, noise_mix, llrs


def build_offgrid_branches(
    obs: OffGridObservation,
    atoms: Optional[OffGridAtomEstimates] = None,
    k_hat: int = K_HAT,
) -> OffGridBranchBundle:
    """Construct strictly public per-ray ('inherit', 'clear') branches & exact 2^K subset Bayes statistics.

    100% blind (uses ONLY OffGridObservation), ZERO probe-row access, exact C* invariance/equivariance,
    and exact ray-permutation equivariance (uses simultaneous damped Jacobi & closed-form joint LMMSE).
    """
    probe_row = _strict_int(obs.probe_row, "obs.probe_row")
    train_rows = tuple(_strict_int(r, "obs.training_rows") for r in obs.training_rows)
    if probe_row < 0 or probe_row >= T or any(r < 0 or r >= T for r in train_rows) or probe_row in train_rows or not train_rows:
        raise ValueError(f"Invalid training_rows={train_rows} with probe_row={probe_row}")

    n0 = _validate_n0(obs.n0, "obs.n0")
    s_mag = math.sqrt(n0)
    s_var = n0

    # Immediately sanitize withheld probe row before touching obs.y_b anywhere.
    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    if y_b_safe.shape != (T, N):
        raise ValueError(f"Expected obs.y_b shape ({T}, {N}), got {y_b_safe.shape}")
    y_b_safe[probe_row] = 0.0 + 0.0j
    if not np.all(np.isfinite(y_b_safe)):
        raise ValueError("obs.y_b non-probe rows contain non-finite values")
    pilot_b = np.asarray(obs.pilot_tx_b, dtype=np.complex128)
    if pilot_b.shape != (len(train_rows), N) or not np.all(np.isfinite(pilot_b)):
        raise ValueError(f"obs.pilot_tx_b must be finite with shape ({len(train_rows)}, {N}), got {pilot_b.shape}")

    if atoms is None:
        atoms = discover_frame_a_atoms(obs, k_hat=k_hat)

    k = len(atoms.tau_hat)
    tau_hat = np.asarray(atoms.tau_hat, dtype=np.float64)

    # Exact unitary fractional-delay operators and frequency responses.
    d_mats = np.stack([fractional_delay_matrix(float(t), n=N) for t in tau_hat], axis=0)  # [K, N, N]
    phi_f = fractional_delay_freq_response(tau_hat, n=N)                                  # [K, N]
    u_probe = np.einsum("knm,m->kn", d_mats, SC_PILOTS[probe_row])                        # [K, N]

    # Normalized training observations & anchor times.
    r_anchor = train_rows[0]
    y_tr_u = (y_b_safe[list(train_rows)] / s_mag).reshape(-1)  # [M_tr * N]
    m_tr = y_tr_u.size
    t_tr = _TIME_B_FULL[list(train_rows)].reshape(-1)          # [M_tr * N]
    t_anchor = float(t_tr.mean())

    u_tr_blocks = np.einsum("knm,rm->krn", d_mats, obs.pilot_tx_b)  # [K, len(train_rows), N]
    u_tr = u_tr_blocks.reshape(k, m_tr)                             # [K, M_tr * N]

    # Frame-B-only C*-invariant intra-anchor Doppler proxy (strictly independent of Frame A, for no_history).
    y_anc_0 = y_b_safe[r_anchor] / s_mag
    u_anc_0 = u_tr_blocks[:, 0, :]
    half = N // 2
    c1 = np.sum(u_anc_0[:, :half].conj() * y_anc_0[None, :half], axis=-1)
    c2 = np.sum(u_anc_0[:, half:].conj() * y_anc_0[None, half:], axis=-1)
    prod_half = c2 * c1.conj()
    nu_b_proxy = np.where(
        np.abs(prod_half) > 1e-18,
        np.clip(np.angle(prod_half) / np.pi, -0.12, 0.12),
        0.02,
    ).astype(np.float64)

    # --- 1. Build Strict Pre-Update Prior Moments on Training Row(s) ---
    p_atom_u = np.maximum(atoms.p_atom_a / s_var, 1e-12)
    p_tot_u = max(float(atoms.p_tot_a) / s_var, 0.25)

    nu_inh_bank = atoms.nu_bank_a                           # [K, H_inh]
    chirp_inh_bank = atoms.chirp_bank_a                     # [K, H_inh]
    log_w_prior_inh = atoms.log_w_bank_a                    # [K, H_inh]
    w_prior_inh = np.exp(log_w_prior_inh)                   # [K, H_inh]
    mu_prior_inh_u = atoms.mu_bank_a / s_mag                # [K, H_inh]
    v_prior_inh_u = np.maximum(atoms.var_bank_a / s_var, 1e-12)

    dt_tr_a = t_tr[None, None, :] - _T_A_MEAN
    rot_tr_inh = np.exp(
        2j * np.pi * (nu_inh_bank[:, :, None] * dt_tr_a + 0.5 * chirp_inh_bank[:, :, None] * (dt_tr_a ** 2))
    )
    design_inh = rot_tr_inh * u_tr[:, None, :]              # [K, H_inh, M]

    # Per-ray pre-update mean & covariance on training row(s) under 'inherit' (m_k = 1):
    mu_1_ray = np.einsum("kh,khm->km", w_prior_inh * mu_prior_inh_u, design_inh)
    sec_1 = np.maximum(w_prior_inh * (np.abs(mu_prior_inh_u) ** 2 + v_prior_inh_u), 0.0)
    cov_1_ray = np.empty((k, m_tr, m_tr), dtype=np.complex128)
    for l_idx in range(k):
        wd = np.sqrt(sec_1[l_idx])[:, None] * design_inh[l_idx]
        c_mat = wd.T @ wd.conj() - np.outer(mu_1_ray[l_idx], mu_1_ray[l_idx].conj())
        cov_1_ray[l_idx] = 0.5 * (c_mat + c_mat.conj().T)

    # Per-ray pre-update mean (0) & covariance under 'clear' (m_k = 0):
    p_clear_u = np.maximum(p_atom_u, 0.10 * p_tot_u / float(k))
    h_clr = len(_NU_CLR_BANK)
    rot_tr_clr = np.exp(2j * np.pi * _NU_CLR_BANK[:, None] * (t_tr[None, :] - t_anchor))
    cov_0_ray = np.empty((k, m_tr, m_tr), dtype=np.complex128)
    for l_idx in range(k):
        d_c = (math.sqrt(float(p_clear_u[l_idx]) / float(h_clr))) * (rot_tr_clr * u_tr[l_idx][None, :])
        c_mat = d_c.T @ d_c.conj()
        cov_0_ray[l_idx] = 0.5 * (c_mat + c_mat.conj().T)

    # --- 2. Exact 2^K Strict Pre-Update Subset Marginal Log-Likelihoods ---
    n_subsets = 1 << k
    bits = ((np.arange(n_subsets, dtype=np.int64)[:, None] >> np.arange(k, dtype=np.int64)[None, :]) & 1).astype(np.float64)
    mu_sub = bits @ mu_1_ray  # [2^K, M]
    cov_sub = (
        np.eye(m_tr, dtype=np.complex128)[None, :, :]
        + np.tensordot(bits, cov_1_ray, axes=([1], [0]))
        + np.tensordot(1.0 - bits, cov_0_ray, axes=([1], [0]))
    )
    cov_sub = 0.5 * (cov_sub + cov_sub.conj().swapaxes(-1, -2))
    chol_sub = np.linalg.cholesky(cov_sub)
    diff_sub = (y_tr_u[None, :] - mu_sub)[..., None]
    z_sub = np.linalg.solve(chol_sub, diff_sub)[..., 0]
    logdet_sub = 2.0 * np.sum(np.log(np.real(np.diagonal(chol_sub, axis1=-2, axis2=-1))), axis=-1)
    quad_sub = np.sum(z_sub.real ** 2 + z_sub.imag ** 2, axis=-1)
    ll_scale_shift = -float(m_tr) * math.log(s_var)
    subset_logliks = (-float(m_tr) * math.log(math.pi) - logdet_sub - quad_sub + ll_scale_shift).astype(np.float64)
    log_post_sub = subset_logliks - float(logsumexp(subset_logliks))
    subset_posteriors = np.exp(log_post_sub).astype(np.float64)

    w_ray_classical = np.empty(k, dtype=np.float64)
    base_logit_ray = np.empty(k, dtype=np.float64)
    for l_idx in range(k):
        mask_1 = bits[:, l_idx] > 0.5
        lp1 = float(logsumexp(log_post_sub[mask_1]))
        lp0 = float(logsumexp(log_post_sub[~mask_1]))
        clipped_logit = float(np.clip(lp1 - lp0, -BASE_LOGIT_CLIP, BASE_LOGIT_CLIP))
        base_logit_ray[l_idx] = clipped_logit
        w_ray_classical[l_idx] = 1.0 / (1.0 + math.exp(-clipped_logit))

    idx_all_1 = n_subsets - 1
    idx_all_0 = 0
    pre_ll_inh = float(subset_logliks[idx_all_1])
    pre_ll_clr = float(subset_logliks[idx_all_0])
    pre_pred_inh = (mu_sub[idx_all_1] * s_mag).reshape(len(train_rows), N)
    pre_var_inh = (np.maximum(np.real(np.diag(cov_sub[idx_all_1])), 1e-12) * s_var).reshape(len(train_rows), N)
    pre_pred_clr = np.zeros((len(train_rows), N), dtype=np.complex128)
    pre_var_clr = (np.maximum(np.real(np.diag(cov_sub[idx_all_0])), 1e-12) * s_var).reshape(len(train_rows), N)

    # --- 3. Post-Anchor Conditioning for 'inherit' (Simultaneous Damped Jacobi -> Exact Ray-Perm Equivariant!) ---
    energy_inh = np.sum(np.abs(design_inh) ** 2, axis=-1)
    post_var_inh_u = v_prior_inh_u / (1.0 + v_prior_inh_u * energy_inh)
    init_prec_mean = mu_prior_inh_u / v_prior_inh_u
    ev_base_inh = (
        log_w_prior_inh
        + np.log(post_var_inh_u / v_prior_inh_u)
        - (np.abs(mu_prior_inh_u) ** 2) / v_prior_inh_u
    )

    log_w_inh = log_w_prior_inh.copy()
    mean_inh_h = mu_prior_inh_u.copy()
    per_ray_tr = np.einsum("kh,khm->km", np.exp(log_w_inh) * mean_inh_h, design_inh)
    design_inh_conj = design_inh.conj()
    damp = 0.65
    for _ in range(8):
        total_tr = per_ray_tr.sum(axis=0)
        residuals = y_tr_u[None, :] - (total_tr[None, :] - per_ray_tr)
        matched = np.einsum("khm,km->kh", design_inh_conj, residuals)
        nat_mean = init_prec_mean + matched
        new_mean = post_var_inh_u * nat_mean
        ev = ev_base_inh + post_var_inh_u * (np.abs(nat_mean) ** 2)
        new_log_w = ev - logsumexp(ev, axis=1, keepdims=True)
        new_per_ray = np.einsum("kh,khm->km", np.exp(new_log_w) * new_mean, design_inh)
        per_ray_tr = damp * new_per_ray + (1.0 - damp) * per_ray_tr
        mean_inh_h = new_mean
        log_w_inh = new_log_w

    w_post_inh = np.exp(log_w_inh)  # [K, H_inh]
    # Posterior-expected instantaneous Doppler at Frame-B midpoint (for parallel-transport connection!)
    t_b_mid = float(_TIME_B_FULL.mean())
    nu_hat_post = np.sum(
        w_post_inh * (nu_inh_bank + chirp_inh_bank * (t_b_mid - _T_A_MEAN)), axis=1
    ).astype(np.float64)

    dt_b_a = (_TIME_B_FULL - _T_A_MEAN).reshape(-1)
    rot_b_inh = np.exp(
        2j * np.pi * (nu_inh_bank[:, :, None] * dt_b_a[None, None, :] + 0.5 * chirp_inh_bank[:, :, None] * (dt_b_a[None, None, :] ** 2))
    )
    mean_inh_u = np.einsum("kh,khq->kq", w_post_inh * mean_inh_h, rot_b_inh).reshape(k, T, N)
    sec_post_inh_u = np.sum(w_post_inh * (np.abs(mean_inh_h) ** 2 + post_var_inh_u), axis=1)
    var_inh_u = np.maximum(sec_post_inh_u[:, None, None] - np.abs(mean_inh_u) ** 2, 1e-12)

    mean_inh = (mean_inh_u * s_mag).astype(np.complex128)
    var_inh = (var_inh_u * s_var).astype(np.float64)

    # --- 4. Post-Anchor Fresh Frame-B-Only 'clear' Conditioning (16-point Doppler Bank, Damped Jacobi!) ---
    # Extracts exact gain AND intra-row Doppler sign observable from y_b[r_anchor] alone (ZERO Frame-A
    # knowledge!), matching P8-C/P9 clear physics while preserving exact ray-permutation equivariance.
    design_clr = rot_tr_clr[None, :, :] * u_tr[:, None, :]  # [K, H_clr, M]
    energy_clr = np.sum(np.abs(design_clr) ** 2, axis=-1)   # [K, H_clr]
    v_prior_clr_u = np.broadcast_to(p_clear_u[:, None], (k, h_clr)).copy()
    post_var_clr_u = v_prior_clr_u / (1.0 + v_prior_clr_u * energy_clr)
    log_w_prior_clr = np.full((k, h_clr), -math.log(float(h_clr)), dtype=np.float64)
    ev_base_clr = log_w_prior_clr + np.log(post_var_clr_u / v_prior_clr_u)

    log_w_clr = log_w_prior_clr.copy()
    mean_clr_h = np.zeros((k, h_clr), dtype=np.complex128)
    per_ray_clr = np.zeros((k, m_tr), dtype=np.complex128)
    design_clr_conj = design_clr.conj()
    for _ in range(8):
        total_clr = per_ray_clr.sum(axis=0)
        residuals = y_tr_u[None, :] - (total_clr[None, :] - per_ray_clr)
        matched = np.einsum("khm,km->kh", design_clr_conj, residuals)
        new_mean = post_var_clr_u * matched
        ev = ev_base_clr + post_var_clr_u * (np.abs(matched) ** 2)
        new_log_w = ev - logsumexp(ev, axis=1, keepdims=True)
        new_per_ray = np.einsum("kh,khm->km", np.exp(new_log_w) * new_mean, design_clr)
        per_ray_clr = damp * new_per_ray + (1.0 - damp) * per_ray_clr
        mean_clr_h = new_mean
        log_w_clr = new_log_w

    w_post_clr = np.exp(log_w_clr)  # [K, H_clr]
    dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)  # [T*N]
    rot_b_clr = np.exp(2j * np.pi * _NU_CLR_BANK[:, None] * dt_b_anc[None, :])  # [H_clr, T*N]
    mean_clr_u = ((w_post_clr * mean_clr_h) @ rot_b_clr).reshape(k, T, N)
    sec_clr_u = np.sum(w_post_clr * (np.abs(mean_clr_h) ** 2 + post_var_clr_u), axis=1)
    var_clr_u = np.maximum(sec_clr_u[:, None, None] - np.abs(mean_clr_u) ** 2, 1e-12)

    mean_clr = (mean_clr_u * s_mag).astype(np.complex128)
    var_clr = (var_clr_u * s_var).astype(np.float64)

    # --- 5. Precompute Branch & Classical-Per-Ray LMMSE LLRs ---
    _, _, llrs_inh = _compute_all_row_llrs(mean_inh, var_inh, d_mats, y_b_safe, n0, train_rows, probe_row)
    _, _, llrs_clr = _compute_all_row_llrs(mean_clr, var_clr, d_mats, y_b_safe, n0, train_rows, probe_row)

    w_cls_3 = w_ray_classical[:, None, None]
    mean_cls = w_cls_3 * mean_inh + (1.0 - w_cls_3) * mean_clr
    var_cls = (
        w_cls_3 * var_inh
        + (1.0 - w_cls_3) * var_clr
        + w_cls_3 * (1.0 - w_cls_3) * (np.abs(mean_inh - mean_clr) ** 2)
    )
    _, _, llrs_cls = _compute_all_row_llrs(mean_cls, var_cls, d_mats, y_b_safe, n0, train_rows, probe_row)

    return OffGridBranchBundle(
        atoms=atoms,
        k_hat=k,
        tau_hat=tau_hat,
        nu_hat=nu_hat_post,
        nu_b_proxy=nu_b_proxy,
        d_mats=d_mats,
        phi_f=phi_f,
        u_probe=u_probe,
        mean_inh=mean_inh,
        var_inh=var_inh,
        mean_clr=mean_clr,
        var_clr=var_clr,
        pre_pred_inh=pre_pred_inh,
        pre_var_inh=pre_var_inh,
        pre_pred_clr=pre_pred_clr,
        pre_var_clr=pre_var_clr,
        pre_ll_inh=pre_ll_inh,
        pre_ll_clr=pre_ll_clr,
        subset_logliks=subset_logliks,
        subset_posteriors=subset_posteriors,
        w_ray_classical=w_ray_classical,
        base_logit_ray=base_logit_ray,
        llrs_inh=llrs_inh,
        llrs_clr=llrs_clr,
        llrs_cls=llrs_cls,
        probe_row=probe_row,
        training_rows=train_rows,
    )


def permute_branch_bundle(bundle: OffGridBranchBundle, perm: Sequence[int]) -> OffGridBranchBundle:
    """Apply an exact ray-index permutation `perm` to an OffGridBranchBundle."""
    p_idx = np.asarray(perm, dtype=np.int64)
    k = bundle.k_hat
    if p_idx.shape != (k,) or set(int(x) for x in p_idx) != set(range(k)):
        raise ValueError(f"Invalid ray permutation {perm!r} for k_hat={k}")

    n_subsets = 1 << k
    old_indices = np.zeros(n_subsets, dtype=np.int64)
    for s_new in range(n_subsets):
        s_old = 0
        for i_new in range(k):
            if (s_new >> i_new) & 1:
                s_old |= 1 << int(p_idx[i_new])
        old_indices[s_new] = s_old

    atoms_p = OffGridAtomEstimates(
        tau_hat=bundle.atoms.tau_hat[p_idx].copy(),
        nu_hat=bundle.atoms.nu_hat[p_idx].copy(),
        g_a_mid=bundle.atoms.g_a_mid[p_idx].copy(),
        v_a_mid=bundle.atoms.v_a_mid[p_idx].copy(),
        atom_snr_a=bundle.atoms.atom_snr_a[p_idx].copy(),
        p_atom_a=bundle.atoms.p_atom_a[p_idx].copy(),
        p_tot_a=bundle.atoms.p_tot_a,
        nu_bank_a=bundle.atoms.nu_bank_a[p_idx].copy(),
        chirp_bank_a=bundle.atoms.chirp_bank_a[p_idx].copy(),
        log_w_bank_a=bundle.atoms.log_w_bank_a[p_idx].copy(),
        mu_bank_a=bundle.atoms.mu_bank_a[p_idx].copy(),
        var_bank_a=bundle.atoms.var_bank_a[p_idx].copy(),
    )
    return OffGridBranchBundle(
        atoms=atoms_p,
        k_hat=k,
        tau_hat=bundle.tau_hat[p_idx].copy(),
        nu_hat=bundle.nu_hat[p_idx].copy(),
        nu_b_proxy=bundle.nu_b_proxy[p_idx].copy(),
        d_mats=bundle.d_mats[p_idx].copy(),
        phi_f=bundle.phi_f[p_idx].copy(),
        u_probe=bundle.u_probe[p_idx].copy(),
        mean_inh=bundle.mean_inh[p_idx].copy(),
        var_inh=bundle.var_inh[p_idx].copy(),
        mean_clr=bundle.mean_clr[p_idx].copy(),
        var_clr=bundle.var_clr[p_idx].copy(),
        pre_pred_inh=bundle.pre_pred_inh.copy(),
        pre_var_inh=bundle.pre_var_inh.copy(),
        pre_pred_clr=bundle.pre_pred_clr.copy(),
        pre_var_clr=bundle.pre_var_clr.copy(),
        pre_ll_inh=bundle.pre_ll_inh,
        pre_ll_clr=bundle.pre_ll_clr,
        subset_logliks=bundle.subset_logliks[old_indices].copy(),
        subset_posteriors=bundle.subset_posteriors[old_indices].copy(),
        w_ray_classical=bundle.w_ray_classical[p_idx].copy(),
        base_logit_ray=bundle.base_logit_ray[p_idx].copy(),
        llrs_inh=bundle.llrs_inh.copy(),
        llrs_clr=bundle.llrs_clr.copy(),
        llrs_cls=bundle.llrs_cls.copy(),
        probe_row=bundle.probe_row,
        training_rows=bundle.training_rows,
    )


# ---------------------------------------------------------------------------
# Mixture Composition & Classical Baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffGridGateResult:
    """Unified output of any classical or neural P11 off-grid per-ray gate."""

    gate_out: GateOutput          # Unified GateOutput (mix_llrs, mix_probe_pred, mix_probe_var, etc.)
    w_ray: np.ndarray             # [K_hat, T] float64 per-ray, per-row history trust weights in [0, 1]
    noise_mult: np.ndarray        # [T] float64 per-row outlier/noise-inflation multiplier (1.0 on probe)
    h_mix: np.ndarray             # [T, N, N] complex128 exact time-varying mixture channel matrices
    noise_mix: np.ndarray         # [T, N] float64 effective per-row sample noise variances


def compose_offgrid_gate_output(
    w_ray: np.ndarray,
    noise_mult: np.ndarray,
    obs: OffGridObservation,
    bundle: OffGridBranchBundle,
) -> OffGridGateResult:
    """Compose per-ray w_ray [K_hat, T] & noise_mult [T] into exact H_mix, noise_mix, LLRs, & probe (pred, var)."""
    k = bundle.k_hat
    probe_row = _strict_int(obs.probe_row, "obs.probe_row")
    n0 = _validate_n0(obs.n0, "obs.n0")

    w_arr_raw = np.asarray(w_ray)
    if w_arr_raw.dtype == np.bool_ or w_arr_raw.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"w_ray must be real numeric, got dtype {w_arr_raw.dtype}")
    w_raw = np.asarray(w_arr_raw, dtype=np.float64).reshape(k, T)
    if not np.all(np.isfinite(w_raw)):
        raise ValueError("w_ray contains non-finite values")

    nm_arr_raw = np.asarray(noise_mult)
    if nm_arr_raw.dtype == np.bool_ or nm_arr_raw.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"noise_mult must be real numeric, got dtype {nm_arr_raw.dtype}")
    nm_raw = np.asarray(nm_arr_raw, dtype=np.float64).reshape(T)
    if not np.all(np.isfinite(nm_raw)) or np.any(nm_raw <= 0.0):
        raise ValueError("noise_mult must be finite and strictly positive")

    w = np.clip(w_raw, 0.0, 1.0)
    nm = np.maximum(nm_raw, 1e-6).copy()
    nm[probe_row] = 1.0

    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    y_b_safe[probe_row] = 0.0 + 0.0j
    if not np.all(np.isfinite(y_b_safe)):
        raise ValueError("obs.y_b non-probe rows contain non-finite values")

    w3 = w[:, :, None]
    mu1 = bundle.mean_inh
    mu0 = bundle.mean_clr
    v1 = bundle.var_inh
    v0 = bundle.var_clr

    mix_mean_ray = w3 * mu1 + (1.0 - w3) * mu0
    mix_var_ray = (
        w3 * v1
        + (1.0 - w3) * v0
        + w3 * (1.0 - w3) * (np.abs(mu1 - mu0) ** 2)
    )

    h_mix, noise_mix, mix_llrs = _compute_all_row_llrs(
        mix_mean_ray,
        mix_var_ray,
        bundle.d_mats,
        y_b_safe,
        n0,
        bundle.training_rows,
        probe_row,
        noise_mult=nm,
    )

    # Exact held-out row-19 probe prediction & total variance (exact within + between + n0!).
    mix_probe_pred = np.sum(mix_mean_ray[:, probe_row, :] * bundle.u_probe, axis=0)
    mix_probe_var = n0 + np.sum(mix_var_ray[:, probe_row, :] * (np.abs(bundle.u_probe) ** 2), axis=0)

    w_row_summary = w.mean(axis=0)
    gate_out = GateOutput(
        history_weight=w_row_summary,
        probe_weight=float(w_row_summary[probe_row]),
        mix_llrs=mix_llrs,
        mix_probe_pred=mix_probe_pred,
        mix_probe_var=mix_probe_var,
        mix_mean_ray=mix_mean_ray,
        mix_var_ray=mix_var_ray,
    )
    return OffGridGateResult(
        gate_out=gate_out,
        w_ray=w,
        noise_mult=nm,
        h_mix=h_mix,
        noise_mix=noise_mix,
    )


def classical_always_inherit(obs: OffGridObservation, bundle: OffGridBranchBundle) -> OffGridGateResult:
    return compose_offgrid_gate_output(
        np.ones((bundle.k_hat, T), dtype=np.float64),
        np.ones(T, dtype=np.float64),
        obs,
        bundle,
    )


def classical_always_clear(obs: OffGridObservation, bundle: OffGridBranchBundle) -> OffGridGateResult:
    return compose_offgrid_gate_output(
        np.zeros((bundle.k_hat, T), dtype=np.float64),
        np.ones(T, dtype=np.float64),
        obs,
        bundle,
    )


def classical_link_gate(obs: OffGridObservation, bundle: OffGridBranchBundle) -> OffGridGateResult:
    """Classical link-level pre-update Bayes-factor gate (single scalar weight broadcast to [K_hat, T])."""
    raw_bf = float(bundle.pre_ll_inh) - float(bundle.pre_ll_clr)
    clipped = float(np.clip(raw_bf, -BASE_LOGIT_CLIP, BASE_LOGIT_CLIP))
    w_val = 1.0 / (1.0 + math.exp(-clipped))
    return compose_offgrid_gate_output(
        np.full((bundle.k_hat, T), w_val, dtype=np.float64),
        np.ones(T, dtype=np.float64),
        obs,
        bundle,
    )


def classical_per_ray_gate(obs: OffGridObservation, bundle: OffGridBranchBundle) -> OffGridGateResult:
    """Exact 2^K subset-Bayes marginal per-ray classical gate (broadcast across T rows)."""
    w_mat = np.broadcast_to(bundle.w_ray_classical[:, None], (bundle.k_hat, T)).copy()
    return compose_offgrid_gate_output(
        w_mat,
        np.ones(T, dtype=np.float64),
        obs,
        bundle,
    )


__all__ = [
    "BASE_LOGIT_CLIP",
    "BURST_NOISE_MULT",
    "GRATING_LOBE_DF",
    "ID_RAY_COUNTS",
    "K_HAT",
    "MIDFRAME_SHIFT_ROWS",
    "MIN_DELAY_SEP",
    "NU_MAX_BAND",
    "NU_MIN_BAND",
    "OOD_FIVE_RAY_COUNT",
    "ROW_STEP",
    "OffGridAtomEstimates",
    "OffGridBranchBundle",
    "OffGridGateResult",
    "build_offgrid_branches",
    "classical_always_clear",
    "classical_always_inherit",
    "classical_link_gate",
    "classical_per_ray_gate",
    "compose_offgrid_gate_output",
    "discover_frame_a_atoms",
    "eval_piecewise_ray_phase_and_gain",
    "offgrid_ofdm_propagate",
    "offgrid_sc_propagate",
    "permute_branch_bundle",
    "simulate_offgrid_link",
]
