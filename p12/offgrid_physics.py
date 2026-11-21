"""Phase 12 Secondary-Axis Off-Grid Mathematical-Physics Engine & Causal Diagnostics.

Implements all P12 Stage 1B first-principles modules strictly without modifying `p11/`:
  1. Paired off-grid simulator supporting arbitrary 4-pilot Frame-A layouts
     (`(0,9,10,19)` locked comb vs `(0,4,6,19)` Chebyshev Pareto optimum) AND explicit
     separation of P11's inherited-delay `offgrid_full_reset` vs true `offgrid_full_reset_new_delays`.
  2. Geometry-aware blind Frame-A atom discovery + multi-lobe bank (computes exact continuous
     ambiguity side-lobe spacing from `pilot_rows_a`, zero comb-hardcoded bias).
  3. Exact Oracle Ladder on any off-grid link:
     - Step 1: exact box-constrained convex QP mixture `min_{0<=w_k<=1} ||H* - H_clr - sum w_k dH_k||_F^2`
     - Step 2: crossed `(est_K5 / est_trueP / true_delays) x (est_traj / true_traj)`
     - Step 3: full 32x32 continuous-delay second-order payload covariance `C_r = sum G_a D(tau_a-tau_b) G_b^H + n0 I`
       + roving-anchor symmetric bilateral 2-fold wavefront B-refit (true words & blind decoded)
  4. Exact Identifiability & Gauge Diagnostics:
     - Projected-Jacobian Schur-complement Fisher matrix `J_eff = B_psi^T (I - P_{B_eta}) B_psi`
     - Global wrong-lobe profiled KL divergence & Pinsker lower bound `(1 - sqrt(D_KL/2))_+ / 2`
     - Symmetric parallel-transport connection `P_ij = exp(i*pi*(nu_i+nu_j)*(t_i-t_j) - i*2*pi*df_eff*(tau_i-tau_j))`
       verifying exact reverse-edge unitarity `|P_ij P_ji - 1| < 1e-15` vs P11's query-only edge phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.special import logsumexp

from p8 import link as ofdm_link
from p8.coding import qam, qam_llrs
from p8.link import EVEN, clock, complex_normal, stream, traffic
from p8c.link import PILOTS as SC_PILOTS
from p9.decoder import decode_strong_sc
from p11.contract import (
    COMPLEX_DTYPE,
    N,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    T,
    OffGridObservation,
    OffGridTruth,
    build_offgrid_sc_channel_matrix,
    fractional_delay_freq_response,
    fractional_delay_matrix,
    n0_from_snr,
    signed_subcarrier_indices,
)
from p11.offgrid_channel import (
    BURST_NOISE_MULT,
    ID_RAY_COUNTS,
    K_HAT,
    MIDFRAME_SHIFT_ROWS,
    NU_MAX_BAND,
    NU_MIN_BAND,
    OOD_FIVE_RAY_COUNT,
    ROW_STEP,
    OffGridAtomEstimates,
    OffGridBranchBundle,
    _CHIRP_LOBE_FLAT,
    _CHIRP_PRI_FLAT,
    _DNU_LOBE_FLAT,
    _DNU_PRI_FLAT,
    _EYE_N,
    _H_INH,
    _K_EVEN,
    _NU_CLR_BANK,
    _NU_COARSE,
    _TAU_COARSE,
    _TIME_A_FULL,
    _TIME_B_FULL,
    _clamp_nu_passband,
    _compute_all_row_llrs,
    _diric_factor,
    _draw_offgrid_channel,
    build_offgrid_branches,
    classical_per_ray_gate,
    compose_offgrid_gate_output,
    discover_frame_a_atoms,
    eval_piecewise_ray_phase_and_gain,
    offgrid_ofdm_propagate,
    offgrid_sc_propagate,
)
from p11.wavefront_pipeline import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_MIN_ASSIMILATE_WEIGHT,
    DEFAULT_VIRTUAL_VAR_FLOOR,
    decode_with_blind_margin,
    redescending_residual_weight,
)
from p12.contract import (
    OFDM_PILOT_COMB,
    OFDM_PILOT_PARETO,
    OFFGRID_REGIME_TRUE_RESET,
    OFFGRID_STAGE1_REGIMES,
)


# ---------------------------------------------------------------------------
# 1. Paired Simulator with Configurable Frame-A Pilot Geometry & True Full Reset
# ---------------------------------------------------------------------------


def simulate_offgrid_link_p12(
    seed: int,
    regime: str,
    snr_db: float = PRIMARY_SNR_DB,
    *,
    pilot_rows_a: Sequence[int] = OFDM_PILOT_COMB,
    anchor_row: Optional[int] = None,
) -> tuple[OffGridObservation, OffGridTruth, dict[str, Any]]:
    """Generate paired (OffGridObservation, OffGridTruth, extra_truth) with configurable Frame-A pilots.

    When `pilot_rows_a == (0,9,10,19)` and `regime != 'offgrid_full_reset_new_delays'`, reproduces
    `p11.offgrid_channel.simulate_offgrid_link` bit-for-bit!
    Returns extra_truth dict containing `delays_a`, `delays_b`, `g_b_true [P, T, N]`, `b_symbols [18, N]`,
    `t_shift`, `pilot_rows_a`, and `anchor_row`.
    """
    if regime not in OFFGRID_STAGE1_REGIMES:
        raise ValueError(f"Unknown Stage 1B regime {regime!r}; expected one of {OFFGRID_STAGE1_REGIMES}")
    p_rows_a = tuple(int(r) for r in pilot_rows_a)
    if len(p_rows_a) != 4 or len(set(p_rows_a)) != 4 or any(r < 0 or r >= T for r in p_rows_a):
        raise ValueError(f"Expected 4 distinct Frame-A pilot rows in 0..{T-1}, got {p_rows_a}")

    n0 = n0_from_snr(snr_db)
    rng_ch = stream(seed, 0)
    rng_reg = stream(seed, 8)
    rng_anchor = stream(seed, 9)

    if regime == "offgrid_ood_five_ray_fast":
        n_rays = OOD_FIVE_RAY_COUNT
        doppler_scale = 1.5
    else:
        n_rays = int(rng_ch.choice(ID_RAY_COUNTS))
        doppler_scale = 1.0

    delays_a, powers_a, gains_a, freqs_a, chirps_a = _draw_offgrid_channel(
        rng_ch, n_rays, doppler_scale=doppler_scale
    )

    if anchor_row is None:
        r_anchor = int(rng_anchor.choice(OFFGRID_ANCHOR_ROWS))
    else:
        r_anchor = int(anchor_row)
    training_rows = (r_anchor,)
    data_rows = tuple(r for r in range(T) if r not in (r_anchor, PROBE_ROW))

    # Frame A (OFDM) on exact continuous delays with configurable 4 pilot rows (identical 64 unit-modulus pilots!)
    a_words = traffic(stream(seed, 1))
    pilot_tx_a = ofdm_link.PILOTS[np.ix_(list(OFDM_PILOT_COMB), EVEN)].copy()
    if p_rows_a == OFDM_PILOT_COMB:
        tx_grid_a = ofdm_link.tx_grid(a_words, p_rows_a)
    else:
        pilot_mask_a = ofdm_link.row_mask(p_rows_a)
        tx_grid_a = np.zeros((T, N), dtype=COMPLEX_DTYPE)
        tx_grid_a[np.ix_(list(p_rows_a), EVEN)] = pilot_tx_a
        tx_grid_a[~pilot_mask_a] = a_words.symbols[:18].ravel()
    _, g_a = eval_piecewise_ray_phase_and_gain(_TIME_A_FULL, gains_a, freqs_a, chirps_a)
    y_a_clean = offgrid_ofdm_propagate(tx_grid_a, g_a, delays_a)
    a_noise = np.fft.fft(math.sqrt(n0) * complex_normal(stream(seed, 2), (T, N)), axis=-1, norm="ortho")
    y_a = y_a_clean + a_noise

    # Frame B (CP-SC) transmit grid
    words_full = traffic(stream(seed, 3))
    b_info = words_full.info[: len(data_rows)].copy()
    b_code = words_full.code[: len(data_rows)].copy()
    b_symbols = words_full.symbols[: len(data_rows)].copy()

    tx_grid_b = np.empty((T, N), dtype=COMPLEX_DTYPE)
    tx_grid_b[r_anchor] = SC_PILOTS[r_anchor]
    tx_grid_b[PROBE_ROW] = SC_PILOTS[PROBE_ROW]
    tx_grid_b[list(data_rows)] = b_symbols
    pilot_tx_b = SC_PILOTS[list(training_rows)].copy()

    # Frame B ray trajectory & delays
    delays_b = delays_a.copy()
    powers_b = powers_a.copy()
    freqs_b = freqs_a.copy()
    chirps_b = chirps_a.copy()
    changed_rays: Tuple[int, ...] = ()
    shift_row: Optional[int] = None
    t_shift_val: Optional[float] = None
    corrupted_rows: Tuple[int, ...] = ()

    if regime == "offgrid_continuous":
        _, g_b = eval_piecewise_ray_phase_and_gain(_TIME_B_FULL, gains_a, freqs_a, chirps_a)

    elif regime == "offgrid_full_reset":
        # Exact P11 behavior: delays_override=delays_a (preserves delays!)
        _, powers_b, gains_b, freqs_b, chirps_b = _draw_offgrid_channel(
            stream(seed, 5), n_rays, doppler_scale=1.0, delays_override=delays_a
        )
        _, g_b = eval_piecewise_ray_phase_and_gain(_TIME_B_FULL, gains_b, freqs_b, chirps_b)
        changed_rays = tuple(range(n_rays))
        shift_row = 0

    elif regime == OFFGRID_REGIME_TRUE_RESET:
        # P12 true full reset: delays_b ALSO independently redrawn!
        delays_b, powers_b, gains_b, freqs_b, chirps_b = _draw_offgrid_channel(
            stream(seed, 5), n_rays, doppler_scale=1.0, delays_override=None
        )
        _, g_b = eval_piecewise_ray_phase_and_gain(_TIME_B_FULL, gains_b, freqs_b, chirps_b)
        changed_rays = tuple(range(n_rays))
        shift_row = 0

    else:
        n_changed = 1 if n_rays == 2 else int(rng_reg.choice([1, 2]))
        changed_arr = np.sort(rng_reg.choice(n_rays, size=n_changed, replace=False))
        changed_rays = tuple(int(x) for x in changed_arr)
        shift_row = int(rng_reg.choice(MIDFRAME_SHIFT_ROWS)) if regime == "offgrid_midframe_shift" else 0
        t_shift_val = float(T + shift_row) * ROW_STEP
        f_lo = 0.035 * doppler_scale
        f_hi = 0.085 * doppler_scale
        rem_dur = max(float(2 * T - (T + shift_row)) * ROW_STEP, 1.0)
        for l_idx in changed_rays:
            f_at_shift = float(freqs_a[l_idx] + chirps_a[l_idx] * t_shift_val)
            new_sign = -1.0 if f_at_shift >= 0.0 else 1.0
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
            t_shift=t_shift_val,
            freqs_post=freqs_b,
            chirps_post=chirps_b,
        )
        if regime == "offgrid_ood_burst_and_shift":
            burst_arr = np.sort(rng_reg.choice(np.array(data_rows, dtype=np.int64), size=2, replace=False))
            corrupted_rows = tuple(int(x) for x in burst_arr)

    y_b_clean, h_b_true_mat = offgrid_sc_propagate(tx_grid_b, g_b, delays_b)
    b_noise = math.sqrt(n0) * complex_normal(stream(seed, 4), (T, N))
    if corrupted_rows:
        b_noise[list(corrupted_rows)] = math.sqrt(BURST_NOISE_MULT) * b_noise[list(corrupted_rows)]
    y_b_noisy = y_b_clean + b_noise

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
        delays=delays_b,
        powers=powers_b,
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
    extra = {
        "delays_a": delays_a.copy(),
        "delays_b": delays_b.copy(),
        "g_b_true": g_b.copy(),
        "b_symbols": b_symbols.copy(),
        "t_shift": t_shift_val,
        "pilot_rows_a": p_rows_a,
        "anchor_row": r_anchor,
        "y_b_noisy_full": y_b_noisy.copy(),
    }
    return obs, truth, extra


# ---------------------------------------------------------------------------
# 2. Geometry-Aware Frame-A Discovery (Fair Across Arbitrary 4-Pilot Layouts)
# ---------------------------------------------------------------------------


def dominant_sidelobe_spacing(pilot_rows_a: Sequence[int]) -> float:
    """Find the strongest in-band temporal ambiguity sidelobe offset delta in [0.02, 0.25]."""
    p_rows = tuple(int(r) for r in pilot_rows_a)
    if p_rows == OFDM_PILOT_COMB:
        return 1.0 / 14.55  # exact P11 constant for comb parity
    t_bar = _TIME_A_FULL[list(p_rows)].mean(axis=-1)
    dt = t_bar - float(t_bar.mean())
    grid = np.linspace(0.02, 0.25, 4601, dtype=REAL_DTYPE)
    # |A(delta)| = |(1/4) sum exp(2j*pi*dt*delta)|
    vals = np.abs(np.mean(np.exp(2j * np.pi * dt[:, None] * grid[None, :]), axis=0))
    # Search interior local maxima to avoid the mainlobe slope at 0.02
    best_delta = 0.07
    best_val = -1.0
    for i in range(1, len(grid) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1] and vals[i] > best_val:
            best_val = float(vals[i])
            best_delta = float(grid[i])
    return best_delta


def _refine_atom_2d_geom(
    res_mat: np.ndarray,
    dt_pilot_a: np.ndarray,
    tau_init: float,
    nu_init: float,
    existing_taus: Sequence[float],
    min_sep: float = 0.60,
    n_iters: int = 3,
) -> tuple[float, float]:
    """2D Newton refinement generalized to arbitrary Frame-A pilot time offsets `dt_pilot_a`."""
    tau = float(tau_init)
    nu = float(nu_init)
    sign_ref = 1.0 if nu >= 0.0 else -1.0
    c_f = -2.0 * np.pi * _K_EVEN / float(N)
    c_t = 2.0 * np.pi * dt_pilot_a
    inv_norm = 1.0 / math.sqrt(float(len(EVEN) * len(dt_pilot_a)))

    def _eval_obj(t_val: float, f_val: float):
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


def discover_frame_a_atoms_geom(
    obs: OffGridObservation,
    *,
    pilot_rows_a: Sequence[int] = OFDM_PILOT_COMB,
    k_hat: int = K_HAT,
    fixed_delays: Optional[np.ndarray] = None,
) -> tuple[OffGridAtomEstimates, float]:
    """Geometry-aware Frame-A blind discovery returning `(OffGridAtomEstimates, t_a_mean)`.

    If `fixed_delays` is supplied (oracle support step), locks `tau_hat = fixed_delays` while still
    blindly estimating Doppler, chirp, gains, and grating-lobe bank solely from Frame-A pilots!
    """
    p_rows = tuple(int(r) for r in pilot_rows_a)
    t_bar_a = _TIME_A_FULL[list(p_rows)].mean(axis=-1)
    t_a_mean = float(t_bar_a.mean())
    dt_pilot_a = t_bar_a - t_a_mean

    if p_rows == OFDM_PILOT_COMB and fixed_delays is None:
        return discover_frame_a_atoms(obs, k_hat=k_hat), t_a_mean

    n0 = float(obs.n0)
    s_mag = math.sqrt(n0)
    s_var = n0
    y_p = np.asarray(obs.y_a[np.ix_(list(p_rows), EVEN)], dtype=COMPLEX_DTYPE)
    pilot_a = np.asarray(obs.pilot_tx_a, dtype=COMPLEX_DTYPE)
    h_ls = y_p / pilot_a
    h_u_raw = h_ls / s_mag
    h_flat_raw = h_u_raw.reshape(-1)
    idx_max = int(np.argmax(h_flat_raw.real ** 2 + h_flat_raw.imag ** 2))
    mag_max = float(abs(h_flat_raw[idx_max]))
    u_phase = complex(h_flat_raw[idx_max] / mag_max) if mag_max > 0.0 else (1.0 + 0.0j)
    h_u = h_u_raw * u_phase.conjugate()
    h_u_vec = h_u.reshape(-1)
    n_obs = h_u_vec.size

    phi_f_coarse = np.exp(-2j * np.pi * np.outer(_K_EVEN, _TAU_COARSE) / float(N)) / math.sqrt(float(len(EVEN)))
    phi_t_coarse = np.exp(2j * np.pi * np.outer(dt_pilot_a, _NU_COARSE)) / math.sqrt(float(len(p_rows)))
    diric_sq = (_diric_factor(_NU_COARSE, n=N) ** 2)[:, None]

    def _atom_vec(tau_v: float, nu_v: float) -> np.ndarray:
        af = np.exp(-2j * np.pi * _K_EVEN * float(tau_v) / float(N)) / math.sqrt(float(len(EVEN)))
        at = np.exp(2j * np.pi * dt_pilot_a * float(nu_v)) / math.sqrt(float(len(p_rows)))
        return np.outer(at, af).reshape(-1)

    if fixed_delays is not None:
        taus = [float(t) for t in np.asarray(fixed_delays, dtype=REAL_DTYPE).ravel()]
        k_use = len(taus)
        nus: list[float] = []
        res_u = h_u.copy()
        d_cols: list[np.ndarray] = []
        for tau_i in taus:
            af_i = np.exp(-2j * np.pi * _K_EVEN * tau_i / float(N)) / math.sqrt(float(len(EVEN)))
            proj_t = res_u @ af_i.conj()  # [4]
            surf_nu = (np.abs(phi_t_coarse.conj().T @ proj_t) ** 2) * diric_sq[:, 0]
            nu_init = float(_NU_COARSE[int(np.argmax(surf_nu))])
            # 1D Newton on nu holding tau fixed
            nu_ref = nu_init
            c_t = 2.0 * np.pi * dt_pilot_a
            sign_ref = 1.0 if nu_ref >= 0.0 else -1.0
            for _ in range(4):
                at = np.exp(1j * c_t * nu_ref) / math.sqrt(float(len(p_rows)))
                z = complex(at.conj() @ proj_t)
                zf = complex(((-1j * c_t) * at.conj()) @ proj_t)
                zff = complex(((-(c_t * c_t)) * at.conj()) @ proj_t)
                gf = 2.0 * float((z.conjugate() * zf).real)
                hff = 2.0 * float((zf.conjugate() * zf + z.conjugate() * zff).real)
                d_nu = float(np.clip(-gf / hff if hff < -1e-12 else 0.0015 * math.copysign(1.0, gf), -0.006, 0.006))
                cand = _clamp_nu_passband(nu_ref + d_nu, sign_ref)
                at_c = np.exp(1j * c_t * cand) / math.sqrt(float(len(p_rows)))
                if abs(at_c.conj() @ proj_t) ** 2 > abs(z) ** 2 * (1.0 + 1e-12):
                    nu_ref = cand
            nus.append(nu_ref)
            d_cols.append(_atom_vec(tau_i, nu_ref))
            d_mat = np.column_stack(d_cols)
            gram = d_mat.conj().T @ d_mat
            gram[np.diag_indices_from(gram)] += 1e-10
            coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
            res_u = (h_u_vec - d_mat @ coeffs).reshape(len(p_rows), len(EVEN))
        # 1 RELAX pass on nus holding fixed_delays
        d_mat = np.column_stack(d_cols)
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
        for j in range(k_use):
            res_j = (h_u_vec - (d_mat @ coeffs - coeffs[j] * d_mat[:, j])).reshape(len(p_rows), len(EVEN))
            af_j = np.exp(-2j * np.pi * _K_EVEN * taus[j] / float(N)) / math.sqrt(float(len(EVEN)))
            proj_t = res_j @ af_j.conj()
            surf_nu = (np.abs(phi_t_coarse.conj().T @ proj_t) ** 2) * diric_sq[:, 0]
            nus[j] = float(_NU_COARSE[int(np.argmax(surf_nu))])
            d_mat[:, j] = _atom_vec(taus[j], nus[j])
            gram = d_mat.conj().T @ d_mat
            gram[np.diag_indices_from(gram)] += 1e-10
            coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
    else:
        k_use = int(k_hat)
        taus = []
        nus = []
        res_u = h_u.copy()
        d_cols = []
        for _ in range(k_use):
            surf = (np.abs(phi_t_coarse.conj().T @ res_u @ phi_f_coarse.conj()) ** 2) * diric_sq
            if taus:
                mask = np.ones(len(_TAU_COARSE), dtype=bool)
                for t_prev in taus:
                    mask &= np.abs(_TAU_COARSE - t_prev) >= 0.65
                if np.any(mask):
                    surf[:, ~mask] = -1.0
            i_nu, i_tau = divmod(int(np.argmax(surf)), len(_TAU_COARSE))
            t_ref, n_ref = _refine_atom_2d_geom(
                res_u, dt_pilot_a, float(_TAU_COARSE[i_tau]), float(_NU_COARSE[i_nu]), taus, min_sep=0.60, n_iters=3
            )
            taus.append(t_ref)
            nus.append(n_ref)
            d_cols.append(_atom_vec(t_ref, n_ref))
            d_mat = np.column_stack(d_cols)
            gram = d_mat.conj().T @ d_mat
            gram[np.diag_indices_from(gram)] += 1e-10
            coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
            res_u = (h_u_vec - d_mat @ coeffs).reshape(len(p_rows), len(EVEN))
        d_mat = np.column_stack(d_cols)
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)
        for j in range(k_use):
            res_j = (h_u_vec - (d_mat @ coeffs - coeffs[j] * d_mat[:, j])).reshape(len(p_rows), len(EVEN))
            others = [taus[i] for i in range(k_use) if i != j]
            t_j, n_j = _refine_atom_2d_geom(res_j, dt_pilot_a, taus[j], nus[j], others, min_sep=0.60, n_iters=3)
            taus[j], nus[j] = t_j, n_j
            d_mat[:, j] = _atom_vec(t_j, n_j)
            gram = d_mat.conj().T @ d_mat
            gram[np.diag_indices_from(gram)] += 1e-10
            coeffs = np.linalg.solve(gram, d_mat.conj().T @ h_u_vec)

    tau_arr = np.asarray(taus, dtype=REAL_DTYPE)
    nu_arr = np.asarray(nus, dtype=REAL_DTYPE)
    diric_vals = _diric_factor(nu_arr, n=N)
    d_phys = d_mat * (math.sqrt(float(n_obs)) * diric_vals)[None, :]

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

    lobe_df = dominant_sidelobe_spacing(p_rows)
    h_inh = _H_INH
    nu_bank_a = np.empty((k_use, h_inh), dtype=REAL_DTYPE)
    chirp_bank_a = np.empty((k_use, h_inh), dtype=REAL_DTYPE)
    log_w_bank_a = np.empty((k_use, h_inh), dtype=REAL_DTYPE)
    mu_bank_u = np.empty((k_use, h_inh), dtype=COMPLEX_DTYPE)
    var_bank_u = np.empty((k_use, h_inh), dtype=REAL_DTYPE)

    full_fit = d_phys @ g_ls_u
    dt_a = dt_pilot_a[None, :]
    dt_a2 = 0.5 * (dt_a ** 2)
    for j in range(k_use):
        res_j_mat = (h_u_vec - (full_fit - g_ls_u[j] * d_phys[:, j])).reshape(len(p_rows), len(EVEN))
        af_j = np.exp(-2j * np.pi * _K_EVEN * tau_arr[j] / float(N))
        z_t = (res_j_mat @ af_j.conj()) / float(len(EVEN))
        v_zt = sigma_h2_u / float(len(EVEN))

        nu0 = float(nu_arr[j])
        valid_lobes = [nl for nl in (nu0 - lobe_df, nu0 + lobe_df) if NU_MIN_BAND <= abs(nl) <= NU_MAX_BAND]
        nu_sec = min(valid_lobes, key=lambda x: abs(x)) if valid_lobes else (nu0 + (0.009 if nu0 >= 0.0 else -0.009))

        cands_nu = np.concatenate((nu0 + _DNU_PRI_FLAT, nu_sec + _DNU_LOBE_FLAT))
        cands_ch = np.concatenate((_CHIRP_PRI_FLAT, _CHIRP_LOBE_FLAT))
        cands_dnu_rel = np.concatenate((_DNU_PRI_FLAT, _DNU_LOBE_FLAT))
        nu_bank_a[j] = cands_nu
        chirp_bank_a[j] = cands_ch

        dir_h = _diric_factor(cands_nu, n=N)
        phase_a = 2.0 * np.pi * (cands_nu[:, None] * dt_a + cands_ch[:, None] * dt_a2)
        u_h = dir_h[:, None] * np.exp(1j * phase_a)
        energy_h = np.sum(np.abs(u_h) ** 2, axis=-1)
        p_j = float(p_atom_u[j])
        post_v = p_j / (1.0 + p_j * energy_h / v_zt)
        nat_m = (u_h.conj() @ z_t) / v_zt
        post_m = post_v * nat_m
        prior_h = (
            -0.5 * ((cands_nu / 0.095) ** 2)
            - 0.5 * ((cands_dnu_rel / 0.005) ** 2)
            - 0.5 * ((cands_ch / 0.00045) ** 2)
        )
        ev_h = prior_h + np.log(post_v / p_j) + post_v * (np.abs(nat_m) ** 2)
        log_w_bank_a[j] = ev_h - float(logsumexp(ev_h))
        mu_bank_u[j] = shrink[j] * post_m
        var_bank_u[j] = np.maximum(shrink[j] * post_v + (1.0 - shrink[j]) * 0.04 * v_ls_u[j] + 0.01 * p_j, 1e-12)

    atoms = OffGridAtomEstimates(
        tau_hat=tau_arr,
        nu_hat=nu_arr,
        g_a_mid=(g_mid_u * u_phase * s_mag).astype(COMPLEX_DTYPE),
        v_a_mid=(v_mid_u * s_var).astype(REAL_DTYPE),
        atom_snr_a=atom_snr_a.astype(REAL_DTYPE),
        p_atom_a=(p_atom_u * s_var).astype(REAL_DTYPE),
        p_tot_a=float(p_tot_u * s_var),
        nu_bank_a=nu_bank_a,
        chirp_bank_a=chirp_bank_a,
        log_w_bank_a=log_w_bank_a,
        mu_bank_a=(mu_bank_u * u_phase * s_mag).astype(COMPLEX_DTYPE),
        var_bank_a=(var_bank_u * s_var).astype(REAL_DTYPE),
    )
    return atoms, t_a_mean


def build_offgrid_branches_geom(
    obs: OffGridObservation,
    atoms: OffGridAtomEstimates,
    *,
    t_a_mean: float,
) -> OffGridBranchBundle:
    """Exact counterpart of `build_offgrid_branches` using the layout's actual `t_a_mean`."""
    if abs(t_a_mean - float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())) < 1e-12:
        return build_offgrid_branches(obs, atoms=atoms, k_hat=len(atoms.tau_hat))

    # Shift atoms' Frame-A bank reference from t_a_mean to default _T_A_MEAN exactly before delegating!
    # Wait: nu(t) = nu0 + ch*(t - t_a_mean) = (nu0 + ch*dt0) + ch*(t - T_A_DEF),
    # phase(t) = phase(T_A_DEF) + ... -> exact phase transport of mu_bank_a & nu_bank_a!
    t_def = float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())
    dt_shift = t_def - float(t_a_mean)
    nu_shifted = atoms.nu_bank_a + atoms.chirp_bank_a * dt_shift
    phase_shift = np.exp(
        2j * np.pi * (atoms.nu_bank_a * dt_shift + 0.5 * atoms.chirp_bank_a * (dt_shift ** 2))
    )
    atoms_shifted = OffGridAtomEstimates(
        tau_hat=atoms.tau_hat.copy(),
        nu_hat=(atoms.nu_hat + atoms.chirp_bank_a[:, 7] * dt_shift).astype(REAL_DTYPE),
        g_a_mid=(atoms.g_a_mid * np.exp(2j * np.pi * atoms.nu_hat * dt_shift)).astype(COMPLEX_DTYPE),
        v_a_mid=atoms.v_a_mid.copy(),
        atom_snr_a=atoms.atom_snr_a.copy(),
        p_atom_a=atoms.p_atom_a.copy(),
        p_tot_a=atoms.p_tot_a,
        nu_bank_a=nu_shifted.astype(REAL_DTYPE),
        chirp_bank_a=atoms.chirp_bank_a.copy(),
        log_w_bank_a=atoms.log_w_bank_a.copy(),
        mu_bank_a=(atoms.mu_bank_a * phase_shift).astype(COMPLEX_DTYPE),
        var_bank_a=atoms.var_bank_a.copy(),
    )
    return build_offgrid_branches(obs, atoms=atoms_shifted, k_hat=len(atoms_shifted.tau_hat))


# ---------------------------------------------------------------------------
# 3. Decoder Helper & Exact Box-Constrained Mixture / Oracle Support×Trajectory
# ---------------------------------------------------------------------------


def score_offgrid_channel_est(
    h_mix: np.ndarray,      # [T, N, N]
    noise_mix: np.ndarray,  # [T, N]
    llrs: np.ndarray,       # [T, N, 4]
    obs: OffGridObservation,
    truth: OffGridTruth,
    *,
    probe_pred: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Decode all 18 data rows with frozen Strong-L and compute channel Frobenius & probe NMSE."""
    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j
    s_mag = math.sqrt(float(obs.n0))
    s_var = float(obs.n0)

    errs = 0
    frob_errs: list[float] = []
    for idx_d, r in enumerate(truth.data_rows):
        dec = decode_strong_sc(
            y_b_safe[r] / s_mag,
            h_mix[r] / s_mag,
            noise_mix[r] / s_var,
            list_size="large",
            bp_iters=20,
            turbo_pic=True,
            pic_scale=0.75,
            initial_llrs=llrs[r].reshape(128),
        )
        if not np.array_equal(dec.info_bits, truth.b_info[idx_d]):
            errs += 1
        dh = h_mix[r] - truth.h_b_true_mat[r]
        hn = float(np.linalg.norm(truth.h_b_true_mat[r]))
        frob_errs.append(float(np.linalg.norm(dh)) / max(hn, 1e-30))

    p_nmse = float("nan")
    if probe_pred is not None:
        pe = probe_pred - truth.y_b_probe_clean
        pn = float(np.sum(truth.y_b_probe_clean.real ** 2 + truth.y_b_probe_clean.imag ** 2))
        p_nmse = float(np.sum(pe.real ** 2 + pe.imag ** 2) / max(pn, 1e-30))

    return {
        "blocks": len(truth.data_rows),
        "block_errors": int(errs),
        "mean_rel_frob_err": float(np.mean(frob_errs)),
        "probe_clean_nmse": p_nmse if math.isfinite(p_nmse) else -1.0,
    }


def solve_oracle_best_box_mixture(
    obs: OffGridObservation,
    bundle: OffGridBranchBundle,
    truth: OffGridTruth,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Step 1: Exact per-row box-constrained QP `min_{0<=w_k<=1} ||H* - H_clr - sum w_k (H_inh,k - H_clr,k)||_F^2`."""
    k = bundle.k_hat
    w_opt = np.zeros((k, T), dtype=REAL_DTYPE)
    for r in range(T):
        h_clr_r = np.zeros((N, N), dtype=COMPLEX_DTYPE)
        d_cols = np.empty((2 * N * N, k), dtype=REAL_DTYPE)
        for j in range(k):
            h_c_j = bundle.mean_clr[j, r, :, None] * bundle.d_mats[j]
            h_i_j = bundle.mean_inh[j, r, :, None] * bundle.d_mats[j]
            h_clr_r += h_c_j
            diff_j = (h_i_j - h_c_j).ravel()
            d_cols[: N * N, j] = diff_j.real
            d_cols[N * N :, j] = diff_j.imag
        resid = (truth.h_b_true_mat[r] - h_clr_r).ravel()
        b_vec = np.concatenate([resid.real, resid.imag])
        sol = lsq_linear(d_cols, b_vec, bounds=(0.0, 1.0), lsmr_tol="auto")
        w_opt[:, r] = np.clip(sol.x, 0.0, 1.0)

    gate_res = compose_offgrid_gate_output(w_opt, np.ones(T, dtype=REAL_DTYPE), obs, bundle)
    sc = score_offgrid_channel_est(
        gate_res.h_mix,
        gate_res.noise_mix,
        gate_res.gate_out.mix_llrs,
        obs,
        truth,
        probe_pred=gate_res.gate_out.mix_probe_pred,
    )
    return w_opt, sc


def fit_known_trajectory_from_blocks(
    delays: np.ndarray,          # [P]
    unit_traj: np.ndarray,       # [P, T, N] complex128 unit-modulus phase trajectories exp(i*phi_p(t))
    known_rows: Sequence[int],
    known_words: np.ndarray,     # [B, N] complex128
    known_y: np.ndarray,         # [B, N] complex128
    known_noises: np.ndarray,    # [B] float64
    prior_ray_var: float,        # scalar per-ray prior variance (C*-equivariant!)
    obs: OffGridObservation,
    truth: OffGridTruth,
) -> dict[str, Any]:
    """Exact Bayesian LMMSE complex-gain fit given known `(delays, unit_traj)` on `known_rows` ONLY."""
    p = len(delays)
    d_mats = np.stack([fractional_delay_matrix(float(d), n=N) for d in delays], axis=0)
    b_cnt = len(known_rows)
    q_dim = b_cnt * N

    # Build design matrix A [Q, P] where column l is unit_traj[l, row, :] * (D(tau_l) @ word_row)
    a_mat = np.empty((q_dim, p), dtype=COMPLEX_DTYPE)
    w_vec = np.repeat(1.0 / np.asarray(known_noises, dtype=REAL_DTYPE), N)
    y_vec = np.asarray(known_y, dtype=COMPLEX_DTYPE).reshape(q_dim)

    for bi, r in enumerate(known_rows):
        sl = slice(bi * N, (bi + 1) * N)
        w_row = known_words[bi]
        for l_idx in range(p):
            a_mat[sl, l_idx] = unit_traj[l_idx, int(r), :] * (d_mats[l_idx] @ w_row)

    prior_prec = np.eye(p, dtype=COMPLEX_DTYPE) / max(float(prior_ray_var), 1e-30)
    gram = (a_mat.conj().T * w_vec[None, :]) @ a_mat + prior_prec
    gram = 0.5 * (gram + gram.conj().T)
    cov_g = np.linalg.inv(gram)
    mu_g = cov_g @ (a_mat.conj().T @ (w_vec * y_vec))

    mean_ray = mu_g[:, None, None] * unit_traj  # [P, T, N]
    # Exact per-sample marginal ray variance = diag(cov_g) since |unit_traj| == 1
    var_g = np.maximum(np.real(np.diag(cov_g)), 1e-18)
    var_ray = np.broadcast_to(var_g[:, None, None], (p, T, N)).copy()

    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j
    h_mix, noise_mix, llrs = _compute_all_row_llrs(
        mean_ray, var_ray, d_mats, y_b_safe, float(obs.n0), obs.training_rows, obs.probe_row
    )
    u_probe = np.einsum("pnm,m->pn", d_mats, SC_PILOTS[obs.probe_row])
    probe_pred = np.sum(mean_ray[:, obs.probe_row, :] * u_probe, axis=0)
    return score_offgrid_channel_est(h_mix, noise_mix, llrs, obs, truth, probe_pred=probe_pred)


# ---------------------------------------------------------------------------
# 4. Full 32x32 Continuous-Delay Payload Covariance + Roving Bilateral 2-Fold Refit
# ---------------------------------------------------------------------------


def make_roving_bilateral_2fold(
    r_anchor: int,
    probe_row: int = PROBE_ROW,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]]:
    """Generate a public-row-only symmetric bilateral 2-fold partition `(T0, T1)`, `(A0, A1)` for any `r_anchor`.

    Strictly satisfies:
      - `T0 ∪ T1 = D` (all 18 non-anchor, non-probe data rows), `T0 ∩ T1 = ∅`, `|T0| = |T1| = 9`
      - `|A0| = |A1| = 5`, `A0 ∩ A1 = ∅`, `A_f ∩ T_f = ∅`
      - Each fold's eligible helpers `H_f = D \\ (T_f ∪ A_f)` (4 helpers/fold) strictly bracket `r_anchor`
        on BOTH sides when `0 < r_anchor < 18`!
    """
    all_d = [r for r in range(T) if r not in (int(r_anchor), int(probe_row))]
    # Sort data rows by distance from anchor, alternating left/right into Fold 0 and Fold 1
    left = sorted([r for r in all_d if r < r_anchor], reverse=True)   # closest left first
    right = sorted([r for r in all_d if r > r_anchor])                # closest right first

    # Assign closest left & closest right to opposite target folds so each fold gets immediate bilateral helpers!
    t0: list[int] = []
    t1: list[int] = []
    for i, r in enumerate(left):
        (t1 if i % 2 == 0 else t0).append(r)  # closest left (i=0) -> T1, so Fold 0 gets it as helper!
    for i, r in enumerate(right):
        (t1 if i % 2 == 0 else t0).append(r)  # closest right (i=0) -> T1, so Fold 0 gets it as bilateral helper!
    # Rebalance sizes to exactly 9 + 9 if needed using farthest rows
    while len(t1) > 9:
        t0.append(t1.pop())
    while len(t0) > 9:
        t1.append(t0.pop())
    t0.sort()
    t1.sort()

    # Pick 5 AUX rows for Fold 0 inside T1 (excluding Fold 0's closest bilateral helpers!),
    # and 5 AUX rows for Fold 1 inside T0 (excluding Fold 1's closest bilateral helpers!).
    # Since A0 ⊂ T1 and A1 ⊂ T0 and T0 ∩ T1 = ∅, global disjointness A0 ∩ A1 = ∅ and A_f ∩ T_f = ∅ hold automatically!
    def _pick_aux_leaving_bilateral_helpers(pool: list[int]) -> tuple[int, ...]:
        # Reserve the closest left and closest right row in `pool` as Wave-1 bilateral helpers,
        # plus 2 far rows (leftmost & rightmost) as Wave-2 bracketing helpers!
        p_left = sorted([r for r in pool if r < r_anchor], reverse=True)
        p_right = sorted([r for r in pool if r > r_anchor])
        reserved: list[int] = []
        if p_left:
            reserved.append(p_left[0])
        if p_right:
            reserved.append(p_right[0])
        if len(p_left) >= 2:
            reserved.append(p_left[-1])
        if len(p_right) >= 2:
            reserved.append(p_right[-1])
        # Fill up to 4 reserved helpers if one side was short
        for r in sorted(pool, key=lambda x: abs(x - r_anchor)):
            if len(reserved) >= 4:
                break
            if r not in reserved:
                reserved.append(r)
        aux = tuple(sorted(r for r in pool if r not in reserved[:4]))
        assert len(aux) == 5
        return aux

    a0 = _pick_aux_leaving_bilateral_helpers(t1)
    a1 = _pick_aux_leaving_bilateral_helpers(t0)
    return (tuple(t0), tuple(t1)), (a0, a1)


def make_roving_bilateral_3fold(
    r_anchor: int,
    probe_row: int = PROBE_ROW,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Return geometry-only folds with bilateral first waves and unchanged row budgets.

    Each fold has six targets, five globally disjoint AUX rows, and seven
    eligible helpers. The receiver's nearest-two helper selection must include
    one row on each side of the anchor; equal-distance ties use ascending rows.
    At least two public data rows are required on each side: if a side has only
    one row, that row's target-owning fold cannot use any helper on that side.
    """
    for name, row in (("r_anchor", r_anchor), ("probe_row", probe_row)):
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise ValueError(f"{name} must be an integer row index, got {row!r}")
        if not 0 <= int(row) < T:
            raise ValueError(f"{name} must be in [0, {T}), got {row!r}")
    r_anchor, probe_row = int(r_anchor), int(probe_row)
    if r_anchor == probe_row:
        raise ValueError("r_anchor and probe_row must be distinct")

    all_d = [r for r in range(T) if r not in (r_anchor, probe_row)]
    n_left = sum(r < r_anchor for r in all_d)
    n_right = sum(r > r_anchor for r in all_d)
    if min(n_left, n_right) < 2:
        raise ValueError(
            "Bilateral 3-fold geometry requires at least two data rows on each side "
            f"of r_anchor={r_anchor}; probe_row={probe_row} leaves {n_left} left and {n_right} right"
        )

    # Mod-3 interleaved targets across ascending data rows -> balanced 6+6+6 across left/right of anchor
    t_folds = tuple(tuple(all_d[i] for i in range(f, len(all_d), 3)) for f in range(3))

    # Keep the cyclic donor assignment: AUX_f is five rows from T_(f+1).
    # The helpers are all six rows of T_(f+2) plus the one retained donor row.
    # Retaining the nearest donor alone can remove every helper on one side
    # (anchor 16, fold 1), or leave the first two helpers on the same side.
    # Scan donor rows in the previous distance order and retain the first one
    # that makes the actual nearest-two wave bilateral. This preserves the old
    # AUX choice whenever it already satisfied that condition, without using
    # observations, channel truth, or evaluation results.
    a_folds_list: list[tuple[int, ...]] = []
    for f in range(3):
        donor = list(t_folds[(f + 1) % 3])
        donor_sorted = sorted(donor, key=lambda r: (abs(r - r_anchor), r))
        other_helpers = t_folds[(f + 2) % 3]
        for retained in donor_sorted:
            helpers = sorted((*other_helpers, retained), key=lambda r: (abs(r - r_anchor), r))
            first, second = helpers[:2]
            if min(first, second) < r_anchor < max(first, second):
                aux_f = tuple(r for r in donor if r != retained)
                break
        else:
            raise ValueError(
                f"No bilateral first wave for anchor={r_anchor}, probe={probe_row}, fold={f}"
            )
        assert len(aux_f) == 5 and set(aux_f).isdisjoint(t_folds[f])
        a_folds_list.append(aux_f)

    a_folds = (a_folds_list[0], a_folds_list[1], a_folds_list[2])
    assert len(set(a_folds[0]) | set(a_folds[1]) | set(a_folds[2])) == 15
    return (t_folds[0], t_folds[1], t_folds[2]), a_folds


def offgrid_cov32_loglik(
    y_aux_u: np.ndarray,      # [R_aux, N] complex128 (normalized by sqrt(n0))
    mean_ray_u: np.ndarray,   # [K, R_aux, N] complex128 (normalized by sqrt(n0))
    var_ray_u: np.ndarray,    # [K, R_aux, N] float64 (normalized by n0)
    d_mats: np.ndarray,       # [K, N, N] complex128 unitary fractional-delay operators
) -> float:
    """Exact full 32x32 Hermitian Cholesky continuous-delay payload Gaussian quasi-log-likelihood.

    Computes `C_r = H_r H_r^H + diag(1 + sum_k v_{k,r})`, where `H_r = sum_k diag(m_{k,r}) D(tau_k)`.
    Algebraically identical to `sum_{a,b} G_{ar} D(tau_a - tau_b) G_{br}^H + diag(1 + sum v)`.
    """
    r_cnt = y_aux_u.shape[0]
    k_cnt = mean_ray_u.shape[0]
    eye_n = _EYE_N
    const_n = -float(N) * math.log(math.pi)
    total = 0.0
    for ri in range(r_cnt):
        h_r = np.zeros((N, N), dtype=COMPLEX_DTYPE)
        for k_idx in range(k_cnt):
            h_r += mean_ray_u[k_idx, ri, :, None] * d_mats[k_idx]
        c_r = h_r @ h_r.conj().T
        c_r[np.diag_indices(N)] += 1.0 + np.sum(var_ray_u[:, ri, :], axis=0)
        c_r = 0.5 * (c_r + c_r.conj().T)
        chol = np.linalg.cholesky(c_r)
        logdet = 2.0 * float(np.sum(np.log(np.maximum(np.real(np.diag(chol)), 1e-300))))
        z = np.linalg.solve(chol, y_aux_u[ri])
        quad = float(np.vdot(z, z).real)
        total += const_n - logdet - quad
    return float(total)


def run_offgrid_cov32_and_wavefront_refit(
    obs: OffGridObservation,
    truth: OffGridTruth,
    extra: Mapping[str, Any],
    *,
    pilot_rows_a: Sequence[int] = OFDM_PILOT_COMB,
    use_aux_cov: bool = True,
    use_wavefront_refit: bool = True,
    helper_mode: str = "blind_decoded",  # "blind_decoded" | "true_words_all" | "true_on_accepted"
    k_hat: int = K_HAT,
    fixed_delays: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Prototype Stage 1B Off-Grid B-Refit: full 32x32 AUX covariance + roving bilateral 2-fold wavefront.

    Strictly respects whole-fold exclusion: each fold masks its 9 target rows before AUX scoring & wavefront!
    """
    n0 = float(obs.n0)
    s_mag = math.sqrt(n0)
    s_var = n0
    r_anchor = int(obs.training_rows[0])

    atoms, t_a_mean = discover_frame_a_atoms_geom(obs, pilot_rows_a=pilot_rows_a, k_hat=k_hat, fixed_delays=fixed_delays)
    bundle = build_offgrid_branches_geom(obs, atoms, t_a_mean=t_a_mean)

    # Track wrong-lobe error on dominant ray before refit
    dom_est_idx = int(np.argmax(atoms.p_atom_a))
    dom_true_f_b_mid = float(truth.freqs_b[0] + truth.chirps[0] * (float(_TIME_B_FULL.mean())))
    dom_nu_err_pre = abs(float(bundle.nu_hat[dom_est_idx]) - dom_true_f_b_mid)

    if (not use_aux_cov) and (not use_wavefront_refit):
        cls_res = classical_per_ray_gate(obs, bundle)
        sc = score_offgrid_channel_est(
            cls_res.h_mix, cls_res.noise_mix, cls_res.gate_out.mix_llrs, obs, truth, probe_pred=cls_res.gate_out.mix_probe_pred
        )
        sc["dom_nu_err_pre"] = dom_nu_err_pre
        sc["dom_nu_err_post"] = dom_nu_err_pre
        sc["wrong_lobe_pre"] = bool(dom_nu_err_pre > 0.03)
        sc["wrong_lobe_post"] = bool(dom_nu_err_pre > 0.03)
        sc["helpers_accepted"] = 0
        sc["false_assimilations"] = 0
        return sc

    # Prepare unified per-ray candidate Doppler/chirp bank per ray (inherit Frame-A 21 hyps + clear 16 B-only hyps = 37)
    t_def = float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())
    dt_shift = t_def - float(t_a_mean)
    nu_inh_bank = atoms.nu_bank_a + atoms.chirp_bank_a * dt_shift  # referenced to t_def
    ch_inh_bank = atoms.chirp_bank_a
    # Map each inherit hyp's instantaneous Doppler at t_anchor
    t_anchor = float(_TIME_B_FULL[r_anchor].mean())
    dt_anc_def = t_anchor - t_def
    nu_inh_at_anc = nu_inh_bank + ch_inh_bank * dt_anc_def  # [K, 21]

    k = bundle.k_hat
    # Combined 37-mode bank per ray: 21 inherit modes + 16 clear modes
    n_inh = nu_inh_bank.shape[1]
    n_clr = len(_NU_CLR_BANK)
    n_tot_h = n_inh + n_clr

    # Precompute phase trajectories on all Frame-B samples relative to t_anchor
    dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)  # [T*N]
    # For each ray k, [37, T, N] unit trajectories referenced at t_anchor
    rot_bank = np.empty((k, n_tot_h, T, N), dtype=COMPLEX_DTYPE)
    nu_mid_bank = np.empty((k, n_tot_h), dtype=REAL_DTYPE)
    t_b_mid = float(_TIME_B_FULL.mean())
    dt_mid_anc = t_b_mid - t_anchor
    for j in range(k):
        # First 21: inherit modes (shifted to anchor phase reference)
        ph_i = 2.0 * np.pi * (
            nu_inh_at_anc[j, :, None] * dt_b_anc[None, :]
            + 0.5 * ch_inh_bank[j, :, None] * (dt_b_anc[None, :] ** 2)
        )
        rot_bank[j, :n_inh] = np.exp(1j * ph_i).reshape(n_inh, T, N)
        nu_mid_bank[j, :n_inh] = nu_inh_at_anc[j] + ch_inh_bank[j] * dt_mid_anc
        # Next 16: clear B-only Doppler modes (zero chirp)
        ph_c = 2.0 * np.pi * (_NU_CLR_BANK[:, None] * dt_b_anc[None, :])
        rot_bank[j, n_inh:] = np.exp(1j * ph_c).reshape(n_clr, T, N)
        nu_mid_bank[j, n_inh:] = _NU_CLR_BANK

    # Prior means/vars at t_anchor for each mode in canonical unit-noise units
    p_atom_u = np.maximum(atoms.p_atom_a / s_var, 1e-12)
    p_tot_u = max(float(atoms.p_tot_a) / s_var, 0.25)
    p_clear_u = np.maximum(p_atom_u, 0.10 * p_tot_u / float(k))

    # Shift atoms.mu_bank_a from t_a_mean -> t_anchor
    dt_anc_a = t_anchor - float(t_a_mean)
    mu_inh_anc_u = (atoms.mu_bank_a / s_mag) * np.exp(
        2j * np.pi * (atoms.nu_bank_a * dt_anc_a + 0.5 * atoms.chirp_bank_a * (dt_anc_a ** 2))
    )
    v_inh_anc_u = np.maximum(atoms.var_bank_a / s_var, 1e-12)

    prior_mu_u = np.zeros((k, n_tot_h), dtype=COMPLEX_DTYPE)
    prior_var_u = np.empty((k, n_tot_h), dtype=REAL_DTYPE)
    prior_logw = np.empty((k, n_tot_h), dtype=REAL_DTYPE)
    for j in range(k):
        w_inh_j = float(np.clip(bundle.w_ray_classical[j], 1e-4, 1.0 - 1e-4))
        prior_mu_u[j, :n_inh] = mu_inh_anc_u[j]
        prior_var_u[j, :n_inh] = v_inh_anc_u[j]
        prior_logw[j, :n_inh] = math.log(w_inh_j) + atoms.log_w_bank_a[j]
        prior_var_u[j, n_inh:] = p_clear_u[j]
        prior_logw[j, n_inh:] = math.log(1.0 - w_inh_j) - math.log(float(n_clr))

    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j
    y_b_u = y_b_safe / s_mag

    row_to_data_idx = {int(r): idx for idx, r in enumerate(truth.data_rows)}
    (t0_rows, t1_rows), (a0_rows, a1_rows) = make_roving_bilateral_2fold(r_anchor, probe_row=obs.probe_row)

    def _fit_jacobi_blocks(
        rows_known: Sequence[int],
        words_known: np.ndarray,   # [B, N]
        noises_u: np.ndarray,      # [B] in unit-noise units
        logw_prior_override: np.ndarray,  # [K, 37]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Damped Jacobi simultaneous multi-mode ray conditioning on arbitrary known/virtual blocks."""
        b_cnt = len(rows_known)
        m_tot = b_cnt * N
        y_k = y_b_u[list(rows_known)].reshape(m_tot)
        w_inv = np.repeat(1.0 / np.asarray(noises_u, dtype=REAL_DTYPE), N)  # [M]

        # Delayed known waveforms u_k [K, B, N]
        u_blocks = np.einsum("knm,bm->kbn", bundle.d_mats, words_known)
        # Design [K, 37, M]
        rot_k = rot_bank[:, :, list(rows_known), :].reshape(k, n_tot_h, m_tot)
        design = rot_k * u_blocks.reshape(k, 1, m_tot)
        energy = np.sum((design.real ** 2 + design.imag ** 2) * w_inv[None, None, :], axis=-1)  # [K, 37]

        post_v = prior_var_u / (1.0 + prior_var_u * energy)
        init_prec_m = prior_mu_u / prior_var_u
        ev_base = logw_prior_override + np.log(post_v / prior_var_u) - (np.abs(prior_mu_u) ** 2) / prior_var_u

        logw_cur = logw_prior_override.copy()
        mean_h = prior_mu_u.copy()
        per_ray_y = np.einsum("kh,khm->km", np.exp(logw_cur) * mean_h, design)
        d_conj_w = design.conj() * w_inv[None, None, :]
        for _ in range(8):
            tot_y = per_ray_y.sum(axis=0)
            res = y_k[None, :] - (tot_y[None, :] - per_ray_y)
            matched = np.einsum("khm,km->kh", d_conj_w, res)
            nat_m = init_prec_m + matched
            new_m = post_v * nat_m
            ev = ev_base + post_v * (np.abs(nat_m) ** 2)
            new_lw = ev - logsumexp(ev, axis=1, keepdims=True)
            new_py = np.einsum("kh,khm->km", np.exp(new_lw) * new_m, design)
            per_ray_y = 0.65 * new_py + 0.35 * per_ray_y
            mean_h = new_m
            logw_cur = new_lw

        w_post = np.exp(logw_cur)
        mean_all = np.einsum("kh,khtn->ktn", w_post * mean_h, rot_bank)
        sec_all = np.sum(w_post * (np.abs(mean_h) ** 2 + post_v), axis=1)
        var_all = np.maximum(sec_all[:, None, None] - np.abs(mean_all) ** 2, 1e-12)
        return mean_all, var_all, logw_cur, mean_h

    def _refine_modes_with_cov32(
        aux_rows: tuple[int, ...],
        mean_init_u: np.ndarray,
        var_init_u: np.ndarray,
        logw_init: np.ndarray,
        mean_h_init: np.ndarray,
    ) -> np.ndarray:
        """Coordinate-wise exact full 32x32 AUX covariance reweighting over top rays' modes."""
        y_aux_u = y_b_u[list(aux_rows)]
        logw_out = logw_init.copy()
        # Refine top-2 strongest rays (captures >90% of off-grid energy and cross-ray interference)
        order = np.argsort(-atoms.p_atom_a)[: min(2, k)]
        cur_mean_aux = mean_init_u[:, list(aux_rows), :].copy()
        cur_var_aux = var_init_u[:, list(aux_rows), :].copy()
        for j in order:
            # Evaluate top-12 modes of ray j + ensure both primary/lobe & positive/negative modes included
            top_m = np.argsort(-logw_out[j])[:12]
            # Always include top-4 clear modes and top-4 inherit modes
            top_i = np.argsort(-logw_out[j, :n_inh])[:6]
            top_c = n_inh + np.argsort(-logw_out[j, n_inh:])[:6]
            cand_h = np.unique(np.concatenate([top_m, top_i, top_c]))
            ll_cov = np.full(n_tot_h, -1e9, dtype=REAL_DTYPE)
            base_m_other = cur_mean_aux.copy()
            base_v_other = cur_var_aux.copy()
            for h_idx in cand_h:
                base_m_other[j] = mean_h_init[j, h_idx] * rot_bank[j, h_idx, list(aux_rows), :]
                base_v_other[j] = 0.05 * p_atom_u[j]
                ll_cov[h_idx] = offgrid_cov32_loglik(y_aux_u, base_m_other, base_v_other, bundle.d_mats)
            # Only apply cov32 delta on evaluated candidates (shift relative to max to avoid underflow)
            eval_delta = ll_cov[cand_h] - float(np.max(ll_cov[cand_h]))
            logw_new_j = np.full(n_tot_h, -1e9, dtype=REAL_DTYPE)
            # Tempered 0.5 factor accounts for non-Gaussian 16-QAM / single-ray approximation variance
            logw_new_j[cand_h] = logw_out[j, cand_h] + 0.5 * eval_delta
            logw_new_j -= float(logsumexp(logw_new_j))
            logw_out[j] = logw_new_j
            w_j = np.exp(logw_new_j)
            cur_mean_aux[j] = np.einsum("h,hrn->rn", w_j * mean_h_init[j], rot_bank[j][:, list(aux_rows), :])
        return logw_out

    owner_h_mix = np.zeros((T, N, N), dtype=COMPLEX_DTYPE)
    owner_noise_mix = np.ones((T, N), dtype=REAL_DTYPE) * n0
    owner_llrs = np.zeros((T, N, 4), dtype=REAL_DTYPE)
    probe_preds: list[np.ndarray] = []
    dom_nu_posts: list[float] = []
    tot_acc = 0
    false_acc = 0

    for t_rows, a_rows in ((t0_rows, a0_rows), (t1_rows, a1_rows)):
        # Initial anchor-only Jacobi fit
        m_u, v_u, lw_cur, mh_cur = _fit_jacobi_blocks(
            (r_anchor,), obs.pilot_tx_b[:1], np.array([1.0], dtype=REAL_DTYPE), prior_logw
        )
        if use_aux_cov:
            lw_cov = _refine_modes_with_cov32(a_rows, m_u, v_u, lw_cur, mh_cur)
            # Re-project moments under AUX-cov updated posterior
            w_cov = np.exp(lw_cov)
            m_u = np.einsum("kh,khtn->ktn", w_cov * mh_cur, rot_bank)
            sec_c = np.sum(w_cov * (np.abs(mh_cur) ** 2 + prior_var_u / (1.0 + prior_var_u * N)), axis=1)
            v_u = np.maximum(sec_c[:, None, None] - np.abs(m_u) ** 2, 1e-12)
            lw_for_refit = lw_cov
        else:
            lw_for_refit = prior_logw

        if use_wavefront_refit:
            forbidden = set(t_rows) | set(a_rows) | {r_anchor, obs.probe_row}
            helpers = sorted([r for r in truth.data_rows if r not in forbidden], key=lambda x: abs(x - r_anchor))
            wave1 = helpers[:2]
            wave2 = helpers[2:]

            known_rows_list: list[int] = [r_anchor]
            known_words_list: list[np.ndarray] = [obs.pilot_tx_b[0].copy()]
            known_noises_list: list[float] = [1.0]

            for wave in (wave1, wave2):
                if not wave:
                    continue
                h_cur, nv_cur, llrs_cur = _compute_all_row_llrs(
                    m_u * s_mag, v_u * s_var, bundle.d_mats, y_b_safe, n0, (r_anchor,), obs.probe_row
                )
                added = False
                for r_h in wave:
                    d_idx = row_to_data_idx[r_h]
                    if helper_mode == "true_words_all":
                        known_rows_list.append(r_h)
                        known_words_list.append(extra["b_symbols"][d_idx].copy())
                        known_noises_list.append(1.0)
                        tot_acc += 1
                        added = True
                        continue

                    res_h = decode_with_blind_margin(
                        y_b_u[r_h],
                        h_cur[r_h] / s_mag,
                        nv_cur[r_h] / s_var,
                        list_size="large",
                        bp_iters=20,
                        turbo_pic=True,
                        pic_scale=0.75,
                        initial_llrs=llrs_cur[r_h].reshape(128),
                    )
                    if res_h.reliability >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                        x_hat = qam(res_h.codeword).astype(COMPLEX_DTYPE)
                        y_hat = (h_cur[r_h] / s_mag) @ x_hat
                        diff = y_b_u[r_h] - y_hat
                        pred_v = nv_cur[r_h] / s_var
                        u_stat = max(
                            float(np.sum(diff.real ** 2 + diff.imag ** 2) / np.sum(pred_v)),
                            float(np.mean((diff.real ** 2 + diff.imag ** 2) / pred_v)),
                        )
                        w_res = redescending_residual_weight(u_stat)
                        alpha_eff = float(min(DEFAULT_ALPHA_MAX, res_h.reliability * w_res))
                        if alpha_eff >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                            n_eff_u = 1.0 / alpha_eff + DEFAULT_VIRTUAL_VAR_FLOOR
                            known_rows_list.append(r_h)
                            if helper_mode == "true_on_accepted":
                                known_words_list.append(extra["b_symbols"][d_idx].copy())
                            else:
                                known_words_list.append(x_hat)
                            known_noises_list.append(n_eff_u)
                            tot_acc += 1
                            if not np.array_equal(res_h.info_bits, truth.b_info[d_idx]):
                                false_acc += 1
                            added = True

                if added:
                    m_u, v_u, lw_cur, mh_cur = _fit_jacobi_blocks(
                        known_rows_list,
                        np.stack(known_words_list, axis=0),
                        np.asarray(known_noises_list, dtype=REAL_DTYPE),
                        lw_for_refit,
                    )

        # Read out fold's owned targets
        h_f, nv_f, llrs_f = _compute_all_row_llrs(
            m_u * s_mag, v_u * s_var, bundle.d_mats, y_b_safe, n0, (r_anchor,), obs.probe_row
        )
        for r_t in t_rows:
            owner_h_mix[r_t] = h_f[r_t]
            owner_noise_mix[r_t] = nv_f[r_t]
            owner_llrs[r_t] = llrs_f[r_t]

        probe_preds.append(np.sum((m_u[:, obs.probe_row, :] * s_mag) * bundle.u_probe, axis=0))
        w_dom = np.exp(lw_cur[dom_est_idx])
        dom_nu_posts.append(float(np.dot(w_dom, nu_mid_bank[dom_est_idx])))

    probe_pred = 0.5 * (probe_preds[0] + probe_preds[1])
    dom_nu_err_post = abs(float(np.mean(dom_nu_posts)) - dom_true_f_b_mid)

    sc = score_offgrid_channel_est(owner_h_mix, owner_noise_mix, owner_llrs, obs, truth, probe_pred=probe_pred)
    sc["dom_nu_err_pre"] = float(dom_nu_err_pre)
    sc["dom_nu_err_post"] = float(dom_nu_err_post)
    sc["wrong_lobe_pre"] = bool(dom_nu_err_pre > 0.03)
    sc["wrong_lobe_post"] = bool(dom_nu_err_post > 0.03)
    sc["helpers_accepted"] = int(tot_acc)
    sc["false_assimilations"] = int(false_acc)
    return sc


# ---------------------------------------------------------------------------
# 5. Identifiability & Connection Audits: Fisher Schur J_eff, Wrong-Lobe KL, P_ij Unitarity
# ---------------------------------------------------------------------------


def compute_offgrid_fisher_schur_diag(
    delays: np.ndarray,
    gains_b_anc: np.ndarray,  # [P] complex128 true ray gains at t_anchor
    freqs_b_anc: np.ndarray,  # [P] float64 instantaneous freq at t_anchor
    chirps_b: np.ndarray,     # [P] float64 chirp
    r_anchor: int,
    n0: float,
    b_symbols_map: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Compute projected-Jacobian Schur-complement Fisher matrix J_eff for psi=(tau, nu) profiling eta=(Re g, Im g, chirp)."""
    p = len(delays)
    t_anc = float(_TIME_B_FULL[r_anchor].mean())

    # Real parameter vector theta = [tau(P), nu(P), Re(g)(P), Im(g)(P), chirp(P)] -> dim 5P
    # psi = theta[:2P], eta = theta[2P:]
    theta0 = np.concatenate([
        np.asarray(delays, dtype=REAL_DTYPE),
        np.asarray(freqs_b_anc, dtype=REAL_DTYPE),
        np.asarray(gains_b_anc.real, dtype=REAL_DTYPE),
        np.asarray(gains_b_anc.imag, dtype=REAL_DTYPE),
        np.asarray(chirps_b, dtype=REAL_DTYPE),
    ])
    dim = len(theta0)
    n_psi = 2 * p

    def _build_h_row(theta: np.ndarray, row: int) -> np.ndarray:
        tau_v = theta[0:p]
        nu_v = theta[p : 2 * p]
        g_v = theta[2 * p : 3 * p] + 1j * theta[3 * p : 4 * p]
        ch_v = theta[4 * p : 5 * p]
        dt = _TIME_B_FULL[row] - t_anc  # [N]
        ph = 2.0 * np.pi * (nu_v[:, None] * dt[None, :] + 0.5 * ch_v[:, None] * (dt[None, :] ** 2))
        gains_row = g_v[:, None] * np.exp(1j * ph)
        return build_offgrid_sc_channel_matrix(gains_row, tau_v, n=N)

    eps = np.concatenate([
        np.full(p, 1e-5),   # tau
        np.full(p, 1e-6),   # nu
        np.full(2 * p, 1e-5),  # Re/Im g
        np.full(p, 1e-7),   # chirp
    ])

    def _row_known_fisher(row: int, x_word: np.ndarray) -> np.ndarray:
        d_mu = np.empty((N, dim), dtype=COMPLEX_DTYPE)
        for i in range(dim):
            tp = theta0.copy(); tp[i] += eps[i]
            tm = theta0.copy(); tm[i] -= eps[i]
            d_mu[:, i] = ((_build_h_row(tp, row) - _build_h_row(tm, row)) @ x_word) / (2.0 * eps[i])
        return (2.0 / float(n0)) * np.real(d_mu.conj().T @ d_mu)

    def _row_cov_fisher(row: int) -> np.ndarray:
        h0 = _build_h_row(theta0, row)
        c0 = h0 @ h0.conj().T + float(n0) * _EYE_N
        c0_inv = np.linalg.inv(0.5 * (c0 + c0.conj().T))
        v_deriv = np.empty((dim, N, N), dtype=COMPLEX_DTYPE)
        for i in range(dim):
            tp = theta0.copy(); tp[i] += eps[i]
            tm = theta0.copy(); tm[i] -= eps[i]
            hp = _build_h_row(tp, row)
            hm = _build_h_row(tm, row)
            dc = (hp @ hp.conj().T - hm @ hm.conj().T) / (2.0 * eps[i])
            v_deriv[i] = c0_inv @ dc
        # J_ij = Re tr( (C^-1 dC_i) (C^-1 dC_j) )
        # = Re sum_{a,b} A[a,b] * B[b,a]
        flat_a = v_deriv.reshape(dim, N * N)
        flat_b = np.swapaxes(v_deriv, 1, 2).reshape(dim, N * N)
        j_mat = np.real(flat_a @ flat_b.T)
        return 0.5 * (j_mat + j_mat.T)

    def _schur_metrics(j_tot: np.ndarray) -> dict[str, float]:
        j_pp = j_tot[:n_psi, :n_psi]
        j_pe = j_tot[:n_psi, n_psi:]
        j_ee = j_tot[n_psi:, n_psi:]
        j_ee_pinv = np.linalg.pinv(0.5 * (j_ee + j_ee.T), rcond=1e-12)
        j_eff = j_pp - j_pe @ j_ee_pinv @ j_pe.T
        j_eff = 0.5 * (j_eff + j_eff.T)
        eigs = np.linalg.eigvalsh(j_eff)
        eigs_tau = np.linalg.eigvalsh(j_eff[:p, :p])
        eigs_nu = np.linalg.eigvalsh(j_eff[p:, p:])
        min_eig = float(np.min(eigs))
        max_eig = float(np.max(eigs))
        return {
            "min_eig_all": min_eig,
            "max_eig_all": max_eig,
            "min_eig_tau": float(np.min(eigs_tau)),
            "min_eig_nu": float(np.min(eigs_nu)),
            "cond_eff": float(max_eig / max(min_eig, 1e-18)),
        }

    (_, _), (a0_rows, _) = make_roving_bilateral_2fold(r_anchor, probe_row=PROBE_ROW)
    # Closest bilateral helper pair around r_anchor
    h_left = r_anchor - 1 if r_anchor - 1 >= 0 else r_anchor + 2
    h_right = r_anchor + 1 if r_anchor + 1 < PROBE_ROW else r_anchor - 2

    j_anc = _row_known_fisher(r_anchor, SC_PILOTS[r_anchor])
    j_5aux = sum(_row_cov_fisher(r) for r in a0_rows)
    j_2bilat = _row_known_fisher(h_left, b_symbols_map[h_left]) + _row_known_fisher(h_right, b_symbols_map[h_right])

    return {
        "n_rays": int(p),
        "anchor_only": _schur_metrics(j_anc),
        "anchor_plus_5aux_cov": _schur_metrics(j_anc + j_5aux),
        "anchor_plus_2bilat_known": _schur_metrics(j_anc + j_2bilat),
        "anchor_plus_5aux_plus_2bilat": _schur_metrics(j_anc + j_5aux + j_2bilat),
    }


def compute_wrong_lobe_profile_kl(
    freqs_a: np.ndarray,
    chirps_a: np.ndarray,
    powers_a: np.ndarray,
    n0: float,
    pilot_rows_a: Sequence[int],
) -> dict[str, float]:
    """Exact single-dominant-ray Frame-A profiled KL divergence & Pinsker lower bound for closest wrong sidelobe."""
    p_rows = tuple(int(r) for r in pilot_rows_a)
    t_bar = _TIME_A_FULL[list(p_rows)].mean(axis=-1)
    dt = t_bar - float(t_bar.mean())
    lobe_df = dominant_sidelobe_spacing(p_rows)

    nu_true = float(freqs_a[0] + chirps_a[0] * float(t_bar.mean()))
    ch_true = float(chirps_a[0])
    # Closest passband wrong sidelobe
    cands = [nu_true - lobe_df, nu_true + lobe_df]
    nu_lobe = min(cands, key=lambda x: abs(x))
    amp2 = float(powers_a[0]) * float(len(EVEN))  # 16 even pilot tones coherent sum

    u_true = _diric_factor(nu_true, n=N) * np.exp(2j * np.pi * (nu_true * dt + 0.5 * ch_true * (dt ** 2)))

    def _kl_at(params: np.ndarray) -> float:
        dnu, ch = float(params[0]), float(params[1])
        nu_c = nu_lobe + dnu
        u_c = _diric_factor(nu_c, n=N) * np.exp(2j * np.pi * (nu_c * dt + 0.5 * ch * (dt ** 2)))
        # Profile optimal complex gain g_hat = (u_c^H u_true) / ||u_c||^2
        denom = float(np.vdot(u_c, u_c).real)
        proj = complex(np.vdot(u_c, u_true)) / max(denom, 1e-30)
        resid = u_true - proj * u_c
        # For CN(mu1, n0 I) vs CN(mu2, n0 I), D_KL = ||mu1 - mu2||^2 / n0
        return float(amp2 * float(np.vdot(resid, resid).real) / float(n0))

    res = minimize(_kl_at, x0=np.array([0.0, 0.0]), method="Powell", bounds=[(-0.015, 0.015), (-0.0015, 0.0015)])
    d_kl = float(max(0.0, res.fun))
    pinsker_lb = 0.5 * max(0.0, 1.0 - math.sqrt(0.5 * d_kl))
    return {
        "lobe_df": float(lobe_df),
        "profile_kl_nats": d_kl,
        "pinsker_error_lb": float(pinsker_lb),
    }


def audit_parallel_transport_connection(
    tau_hat: np.ndarray,
    nu_hat: np.ndarray,
    df_eff: float = 0.125,
) -> dict[str, float]:
    """Audit reverse-edge unitarity `max |U_ij U_ji - 1|` of P11 query-only phase vs P12 symmetric connection."""
    k = len(tau_hat)
    t_rows = _TIME_B_FULL.mean(axis=-1)  # [T]
    # Build K*T tokens (ray r, row t)
    tok_nu = np.repeat(nu_hat, T)
    tok_tau = np.repeat(tau_hat, T)
    tok_t = np.tile(t_rows, k)

    dt = tok_t[:, None] - tok_t[None, :]
    dtau = tok_tau[:, None] - tok_tau[None, :]

    # P11 implemented query-only edge phase: 2*pi * nu_i * (t_i - t_j) - 2*pi * df_eff * (tau_i - tau_j)
    ph_p11 = 2.0 * np.pi * (tok_nu[:, None] * dt - float(df_eff) * dtau)
    u_p11 = np.exp(1j * ph_p11)
    p11_rev_err = float(np.max(np.abs(u_p11 * u_p11.T - 1.0)))

    # P12 symmetric connection: pi * (nu_i + nu_j) * (t_i - t_j) - 2*pi * df_eff * (tau_i - tau_j)
    ph_p12 = np.pi * (tok_nu[:, None] + tok_nu[None, :]) * dt - 2.0 * np.pi * float(df_eff) * dtau
    p_p12 = np.exp(1j * ph_p12)
    p12_rev_err = float(np.max(np.abs(p_p12 * p_p12.T - 1.0)))
    p12_conj_sym_err = float(np.max(np.abs(p_p12.T - p_p12.conj())))

    return {
        "p11_query_only_max_rev_unitarity_err": p11_rev_err,
        "p12_symmetric_max_rev_unitarity_err": p12_rev_err,
        "p12_symmetric_max_conj_sym_err": p12_conj_sym_err,
    }


__all__ = [
    "audit_parallel_transport_connection",
    "build_offgrid_branches_geom",
    "compute_offgrid_fisher_schur_diag",
    "compute_wrong_lobe_profile_kl",
    "discover_frame_a_atoms_geom",
    "dominant_sidelobe_spacing",
    "fit_known_trajectory_from_blocks",
    "make_roving_bilateral_2fold",
    "make_roving_bilateral_3fold",
    "offgrid_cov32_loglik",
    "run_offgrid_cov32_and_wavefront_refit",
    "score_offgrid_channel_est",
    "simulate_offgrid_link_p12",
    "solve_oracle_best_box_mixture",
]
