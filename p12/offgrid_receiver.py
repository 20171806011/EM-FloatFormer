"""Phase 12 Production Secondary-Axis Off-Grid Physics Receiver (Locked Comb `(0,9,10,19)`).

Implements the three Stage 2B first-principles physics upgrades diagnosed by Stage 1B & Codex ultra:
  1. Strict single-counting evidence accounting: incremental full 32x32 continuous-delay AUX
     payload covariance log-likelihood `delta_aux` is stored separately from Frame-A prior and
     combined ONCE with `(anchor U accepted_helpers)` on every wavefront refit (zero anchor
     double counting!).
  2. Wave-1 Numba-accelerated off-grid multi-mode union decoder: partitions dominant-ray posterior
     into coherent trajectory clusters, runs Strong-L on each mode, takes the deduplicated parity-valid
     codeword union + 64 1-flips, and scores exact marginalized predictive LSE log-likelihood.
  3. Unified 57-mode physical trajectory bank per ray:
       - 21 Frame-A inherited `(dnu, chirp)` + grating-lobe modes
       - 20 B-only clear Doppler modes (covering ID `0.035..0.085` + OOD fast `0.105, 0.125`)
       - 16 phase-continuous piecewise midframe-shift modes (`s_row in (6, 8, 11, 14)` x 4 opposite-sign
         post-shift Dopplers) with exact analytic anchor phase transport from Frame A through `t_shift`!
  4. B-only continuous-delay support rediscovery (`discover_b_only_delays`): uses the known B anchor
     pilot waveform (breaking covariance's common-delay shift kernel `H -> H D(s)`) with an explicit
     Occam / BIC evidence threshold to rescue genuine delay-reset links (`offgrid_full_reset_new_delays`)
     without degrading preserved-delay regimes.
  5. Exact C*-invariant per-row excess-energy burst variance guard for `offgrid_ood_burst_and_shift`.

Strict invariants:
  - ZERO hardcoded N0 anywhere.
  - ZERO access to `PROBE_ROW` (`y_b[19]`) or a fold's 9 target rows during inference/waves.
  - Pure `float64` / `complex128` arithmetic; zero `torch.inference_mode()`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence

from numba import njit
import numpy as np
from scipy.special import logsumexp

from p8.coding import qam
from p8c.link import PILOTS as SC_PILOTS
from p9.decoder import decode_strong_sc, is_valid_codeword, pack_codewords, unpack_codewords, warmup_numba as _warmup_p9_numba
from p11.contract import (
    COMPLEX_DTYPE,
    N,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    PROBE_ROW,
    REAL_DTYPE,
    T,
    GateOutput,
    OffGridObservation,
    fractional_delay_matrix,
)
from p11.offgrid_channel import (
    K_HAT,
    ROW_STEP,
    _EYE_N,
    _NU_CLR_BANK,
    _TIME_A_FULL,
    _TIME_B_FULL,
    _compute_all_row_llrs,
    build_offgrid_branches,
    classical_per_ray_gate,
    discover_frame_a_atoms,
)
from p11.wavefront_pipeline import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_MIN_ASSIMILATE_WEIGHT,
    DEFAULT_VIRTUAL_VAR_FLOOR,
    _GEN_U64,
    _H_U64,
    _NIBBLE_IM,
    _NIBBLE_RE,
    _is_valid_u64,
    decode_with_blind_margin,
    redescending_residual_weight,
    reliability_weight_from_margin,
)
from p12.contract import OFDM_PILOT_COMB
from p12.offgrid_physics import make_roving_bilateral_2fold, make_roving_bilateral_3fold, offgrid_cov32_loglik

# Extended B-only clear Doppler bank (16 ID + 4 OOD fast = 20 symmetric modes)
_NU_CLR_EXT_POS = np.concatenate((np.linspace(0.035, 0.085, 8, dtype=REAL_DTYPE), np.array([0.105, 0.125], dtype=REAL_DTYPE)))
_NU_CLR_EXT_BANK = np.concatenate((_NU_CLR_EXT_POS, -_NU_CLR_EXT_POS))  # [20]

# Piecewise phase-continuous midframe shift grid (4 onsets x 4 opposite-sign post-shift speeds = 16 modes)
MID_SHIFT_ONSET_GRID: tuple[int, ...] = (6, 8, 11, 14)
MID_SHIFT_POST_MAGS: tuple[float, ...] = (0.042, 0.056, 0.070, 0.084)

# B-only delay rediscovery coarse grid & Occam log-evidence threshold
_TAU_B_COARSE = np.linspace(OFFGRID_DELAY_MIN, OFFGRID_DELAY_MAX, 66, dtype=REAL_DTYPE)
_D_MATS_B_COARSE: Optional[np.ndarray] = None
DELAY_RESET_OCCAM_NATS: float = 18.0
AUX_COV_TEMP: float = 0.75


def _get_b_coarse_delay_mats() -> np.ndarray:
    global _D_MATS_B_COARSE
    if _D_MATS_B_COARSE is None:
        _D_MATS_B_COARSE = np.stack([fractional_delay_matrix(float(d), n=N) for d in _TAU_B_COARSE], axis=0)
        _D_MATS_B_COARSE.setflags(write=False)
    return _D_MATS_B_COARSE


# ---------------------------------------------------------------------------
# 1. Numba Dense 32x32 Off-Grid Multi-Mode Union Codeword Scorer
# ---------------------------------------------------------------------------


@njit
def _score_offgrid_multimode_u64(
    cand_u64: np.ndarray,   # [M, 2] uint64 deduplicated
    h_u64: np.ndarray,      # [64, 2] uint64
    y_re: np.ndarray,       # [32]
    y_im: np.ndarray,       # [32]
    log_p: np.ndarray,      # [B]
    h_re: np.ndarray,       # [B, 32, 32]
    h_im: np.ndarray,       # [B, 32, 32]
    nv: np.ndarray,         # [B, 32] effective per-sample total noise+channel variance
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
):
    """Exact marginalized LSE predictive likelihood over B dense 32x32 off-grid channel modes."""
    m_cnt = cand_u64.shape[0]
    b_cnt = log_p.shape[0]
    n_sym = y_re.shape[0]
    log_pi = math.log(math.pi)

    x_re = np.empty(n_sym, dtype=np.float64)
    x_im = np.empty(n_sym, dtype=np.float64)
    mode_ll = np.empty(b_cnt, dtype=np.float64)
    mode_fit = np.empty(b_cnt, dtype=np.float64)

    # Precompute constant per-mode log-determinant term
    mode_const = np.empty(b_cnt, dtype=np.float64)
    for b in range(b_cnt):
        c_b = log_p[b]
        for n in range(n_sym):
            c_b -= math.log(nv[b, n]) + log_pi
        mode_const[b] = c_b

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
            x_re[s] = nibble_re[nib0]
            x_im[s] = nibble_im[nib0]
            nib1 = int((c1 >> np.uint64(4 * s)) & np.uint64(15))
            x_re[16 + s] = nibble_re[nib1]
            x_im[16 + s] = nibble_im[nib1]

        max_mll = -np.inf
        win_m = 0
        win_fit = np.inf
        for b in range(b_cnt):
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
                chi2_b += (dr * dr + di * di) / nv[b, n]
            ll_b = mode_const[b] - chi2_b
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


def warmup_p12_offgrid_numba() -> None:
    """Trigger JIT compilation for P9 decoder + P12 off-grid multi-mode union scorer."""
    _warmup_p9_numba()
    _score_offgrid_multimode_u64(
        np.zeros((1, 2), dtype=np.uint64),
        _H_U64,
        np.zeros(N, dtype=REAL_DTYPE),
        np.zeros(N, dtype=REAL_DTYPE),
        np.zeros(1, dtype=REAL_DTYPE),
        np.zeros((1, N, N), dtype=REAL_DTYPE),
        np.zeros((1, N, N), dtype=REAL_DTYPE),
        np.ones((1, N), dtype=REAL_DTYPE),
        _NIBBLE_RE,
        _NIBBLE_IM,
    )


# ---------------------------------------------------------------------------
# 2. B-Only Continuous-Delay Support Discovery & Occam Delay-Reset Gate
# ---------------------------------------------------------------------------


def _anchor_delay_marginal_loglik(
    y_anc_u: np.ndarray,   # [N] unit-noise scaled
    pilot_b: np.ndarray,   # [N] known B anchor pilot
    delays: np.ndarray,    # [K]
    prior_ray_var_u: float,
) -> float:
    """Exact marginal Gaussian log-evidence `log p(y_anc | delays)` under CN(0, prior_ray_var_u I) gains."""
    d_mats = np.stack([fractional_delay_matrix(float(d), n=N) for d in delays], axis=0)
    u_mat = np.einsum("knm,m->nk", d_mats, pilot_b)  # [N, K]
    cov = _EYE_N + float(prior_ray_var_u) * (u_mat @ u_mat.conj().T)
    cov = 0.5 * (cov + cov.conj().T)
    chol = np.linalg.cholesky(cov)
    logdet = 2.0 * float(np.sum(np.log(np.maximum(np.real(np.diag(chol)), 1e-300))))
    z = np.linalg.solve(chol, y_anc_u)
    quad = float(np.vdot(z, z).real)
    return float(-float(N) * math.log(math.pi) - logdet - quad)


def discover_b_only_delays(
    y_anc_u: np.ndarray,   # [N] unit-noise scaled
    pilot_b: np.ndarray,   # [N] known B anchor pilot
    *,
    k_hat: int = K_HAT,
    min_sep: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Strictly B-only continuous-delay discovery (OMP + golden-section refinement + 1-pass RELAX).

    Uses ONLY `(y_anc_u, pilot_b)` -> 100% C*-invariant, zero Frame-A dependence!
    Returns `(tau_b [K], p_atom_b_u [K], p_tot_b_u)`.
    """
    # Canonical phase de-rotation for exact C*-invariance
    idx_m = int(np.argmax(y_anc_u.real ** 2 + y_anc_u.imag ** 2))
    mag_m = float(abs(y_anc_u[idx_m]))
    u_ph = complex(y_anc_u[idx_m] / mag_m) if mag_m > 0.0 else (1.0 + 0.0j)
    y_c = y_anc_u * u_ph.conjugate()

    d_coarse = _get_b_coarse_delay_mats()
    dict_coarse = np.einsum("gnm,m->ng", d_coarse, pilot_b)  # [N, G]
    col_norms = np.maximum(np.linalg.norm(dict_coarse, axis=0), 1e-12)
    dict_unit = dict_coarse / col_norms[None, :]

    def _refine_1d(res_vec: np.ndarray, tau0: float, others: Sequence[float]) -> float:
        lo = max(OFFGRID_DELAY_MIN, float(tau0) - 0.15)
        hi = min(OFFGRID_DELAY_MAX, float(tau0) + 0.15)
        best_t = float(tau0)
        best_v = -1.0
        # 15-point local bracket + 1 zoom pass -> accuracy < 0.002 samples, deterministic & fast
        for span in (0.15, 0.025):
            grid = np.linspace(max(OFFGRID_DELAY_MIN, best_t - span), min(OFFGRID_DELAY_MAX, best_t + span), 13)
            for cand in grid:
                if any(abs(float(cand) - ot) < min_sep for ot in others):
                    continue
                atom = fractional_delay_matrix(float(cand), n=N) @ pilot_b
                v = float(abs(np.vdot(atom, res_vec)) ** 2 / max(float(np.vdot(atom, atom).real), 1e-12))
                if v > best_v:
                    best_v = v
                    best_t = float(cand)
        return best_t

    taus: list[float] = []
    cols: list[np.ndarray] = []
    res = y_c.copy()
    for _ in range(int(k_hat)):
        corr = np.abs(dict_unit.conj().T @ res) ** 2
        if taus:
            mask = np.ones(len(_TAU_B_COARSE), dtype=bool)
            for tp in taus:
                mask &= np.abs(_TAU_B_COARSE - tp) >= min_sep
            if np.any(mask):
                corr[~mask] = -1.0
        t0 = float(_TAU_B_COARSE[int(np.argmax(corr))])
        t_ref = _refine_1d(res, t0, taus)
        taus.append(t_ref)
        cols.append(fractional_delay_matrix(t_ref, n=N) @ pilot_b)
        d_mat = np.column_stack(cols)
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ y_c)
        res = y_c - d_mat @ coeffs

    # 1-pass RELAX refinement
    d_mat = np.column_stack(cols)
    gram = d_mat.conj().T @ d_mat
    gram[np.diag_indices_from(gram)] += 1e-10
    coeffs = np.linalg.solve(gram, d_mat.conj().T @ y_c)
    for j in range(int(k_hat)):
        res_j = y_c - (d_mat @ coeffs - coeffs[j] * d_mat[:, j])
        others = [taus[i] for i in range(int(k_hat)) if i != j]
        t_j = _refine_1d(res_j, taus[j], others)
        taus[j] = t_j
        d_mat[:, j] = fractional_delay_matrix(t_j, n=N) @ pilot_b
        gram = d_mat.conj().T @ d_mat
        gram[np.diag_indices_from(gram)] += 1e-10
        coeffs = np.linalg.solve(gram, d_mat.conj().T @ y_c)

    p_tot_u = max(float(np.mean(np.abs(y_c) ** 2)) - 1.0, 0.25)
    gram_inv = np.linalg.inv(gram)
    v_ls = np.maximum(np.real(np.diag(gram_inv)), 1e-12)
    p_atom_u = np.maximum(np.abs(coeffs) ** 2 - 2.0 * v_ls, 0.05 * p_tot_u / float(k_hat))
    return np.asarray(taus, dtype=REAL_DTYPE), p_atom_u.astype(REAL_DTYPE), float(p_tot_u)


# ---------------------------------------------------------------------------
# 3. Unified P12 Off-Grid Physics Receiver (Zero Double Counting + 57 Modes + Wave-1 Union)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffGridPhysicsResult:
    """Output of the P12 off-grid physical receiver."""
    gate_out: GateOutput
    h_mix: np.ndarray           # [T, N, N] complex128
    noise_mix: np.ndarray       # [T, N] float64
    delay_reset_triggered: bool
    used_delays: np.ndarray     # [K_hat] float64
    delay_ll_gain_nats: float
    helpers_accepted: int
    ledger: tuple[dict[str, Any], ...]
    elapsed_ms: float


def run_p12_offgrid_physics(
    obs: OffGridObservation,
    *,
    k_hat: int = K_HAT,
    use_aux_cov: bool = True,
    use_wavefront: bool = True,
    use_piecewise_midframe: bool = True,
    use_delay_rediscovery: bool = True,
    use_burst_guard: bool = True,
    use_wave1_multimode: bool = True,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> OffGridPhysicsResult:
    """Run the frozen-grade P12 off-grid pure-physics receiver on locked comb `(0,9,10,19)`.

    Strictly respects whole-fold exclusion on every fold: the 9 target rows and PROBE_ROW are
    masked to `NaN` during AUX covariance scoring, Wave-1/2 decoding, and all channel refits.
    """
    t0 = time.perf_counter()
    n0 = float(obs.n0)
    if not math.isfinite(n0) or n0 <= 0.0:
        raise ValueError(f"Expected positive finite obs.n0, got {n0}")
    s_mag = math.sqrt(n0)
    s_var = n0
    r_anchor = int(obs.training_rows[0])
    probe_row = int(obs.probe_row)

    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[probe_row] = 0.0 + 0.0j
    y_b_u = y_b_safe / s_mag

    # Frame-A blind discovery (strictly from y_a, locked comb (0,9,10,19))
    atoms_a = discover_frame_a_atoms(obs, k_hat=k_hat)
    bundle_a = build_offgrid_branches(obs, atoms=atoms_a, k_hat=k_hat)

    # Mask-matched `B-refit off` baseline mode
    if (not use_aux_cov) and (not use_wavefront) and (not use_delay_rediscovery):
        cls_r = classical_per_ray_gate(obs, bundle_a)
        return OffGridPhysicsResult(
            gate_out=cls_r.gate_out,
            h_mix=cls_r.h_mix,
            noise_mix=cls_r.noise_mix,
            delay_reset_triggered=False,
            used_delays=bundle_a.tau_hat.copy(),
            delay_ll_gain_nats=0.0,
            helpers_accepted=0,
            ledger=(),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # --- Step 0: B-only Continuous-Delay Rediscovery & Occam Gate ---
    delay_reset_triggered = False
    delay_ll_gain = 0.0
    p_tot_b_est = max(float(np.mean(np.abs(y_b_u[r_anchor]) ** 2)) - 1.0, 0.25)
    prior_var_delay_test = max(p_tot_b_est / float(k_hat), 0.15)

    if use_delay_rediscovery:
        tau_b, p_atom_b_u, p_tot_b_u = discover_b_only_delays(y_b_u[r_anchor], obs.pilot_tx_b[0], k_hat=k_hat)
        ll_a_anc = _anchor_delay_marginal_loglik(y_b_u[r_anchor], obs.pilot_tx_b[0], bundle_a.tau_hat, prior_var_delay_test)
        ll_b_anc = _anchor_delay_marginal_loglik(y_b_u[r_anchor], obs.pilot_tx_b[0], tau_b, prior_var_delay_test)
        delay_ll_gain = float(ll_b_anc - ll_a_anc)
        if delay_ll_gain > DELAY_RESET_OCCAM_NATS:
            delay_reset_triggered = True

    if delay_reset_triggered:
        used_delays = tau_b
        d_mats = np.stack([fractional_delay_matrix(float(d), n=N) for d in used_delays], axis=0)
        u_probe = np.einsum("knm,m->kn", d_mats, SC_PILOTS[probe_row])
        k = len(used_delays)
        # Pure B-only clear trajectory bank (20 Doppler modes per ray)
        n_inh = 0
        n_clr = len(_NU_CLR_EXT_BANK)
        n_mid = 0
        n_tot_h = n_clr
        t_anchor = float(_TIME_B_FULL[r_anchor].mean())
        dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)
        rot_clr = np.exp(2j * np.pi * (_NU_CLR_EXT_BANK[:, None] * dt_b_anc[None, :])).reshape(n_clr, T, N)
        rot_bank = np.broadcast_to(rot_clr[None, :, :, :], (k, n_clr, T, N)).copy()
        mode_family = np.ones((k, n_tot_h), dtype=np.int64)  # 1=clear
        mode_sign_pos = np.broadcast_to((_NU_CLR_EXT_BANK > 0.0)[None, :], (k, n_tot_h)).copy()

        prior_mu_u = np.zeros((k, n_tot_h), dtype=COMPLEX_DTYPE)
        p_clear_u = np.maximum(p_atom_b_u, 0.12 * p_tot_b_u / float(k))
        prior_var_u = np.broadcast_to(p_clear_u[:, None], (k, n_tot_h)).copy()
        prior_logw = np.full((k, n_tot_h), -math.log(float(n_clr)), dtype=REAL_DTYPE)
        ray_power_order = np.argsort(-p_atom_b_u)
        p_atom_u = p_clear_u
    else:
        used_delays = bundle_a.tau_hat.copy()
        d_mats = bundle_a.d_mats
        u_probe = bundle_a.u_probe
        k = bundle_a.k_hat

        t_a_mean = float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())
        t_anchor = float(_TIME_B_FULL[r_anchor].mean())
        dt_anc_a = t_anchor - t_a_mean
        dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)

        n_inh = atoms_a.nu_bank_a.shape[1]  # 21
        n_clr = len(_NU_CLR_EXT_BANK)       # 20
        n_mid = (len(MID_SHIFT_ONSET_GRID) * len(MID_SHIFT_POST_MAGS)) if use_piecewise_midframe else 0  # 16
        n_tot_h = n_inh + n_clr + n_mid

        rot_bank = np.empty((k, n_tot_h, T, N), dtype=COMPLEX_DTYPE)
        mode_family = np.empty((k, n_tot_h), dtype=np.int64)  # 0=inherit, 1=clear, 2=piecewise_midframe
        mode_sign_pos = np.empty((k, n_tot_h), dtype=bool)

        p_atom_u = np.maximum(atoms_a.p_atom_a / s_var, 1e-12)
        p_tot_u = max(float(atoms_a.p_tot_a) / s_var, 0.25)
        p_clear_u = np.maximum(p_atom_u, 0.10 * p_tot_u / float(k))

        nu_inh_at_anc = atoms_a.nu_bank_a + atoms_a.chirp_bank_a * dt_anc_a
        mu_inh_anc_u = (atoms_a.mu_bank_a / s_mag) * np.exp(
            2j * np.pi * (atoms_a.nu_bank_a * dt_anc_a + 0.5 * atoms_a.chirp_bank_a * (dt_anc_a ** 2))
        )
        v_inh_anc_u = np.maximum(atoms_a.var_bank_a / s_var, 1e-12)

        prior_mu_u = np.zeros((k, n_tot_h), dtype=COMPLEX_DTYPE)
        prior_var_u = np.empty((k, n_tot_h), dtype=REAL_DTYPE)
        prior_logw = np.empty((k, n_tot_h), dtype=REAL_DTYPE)

        rot_clr = np.exp(2j * np.pi * (_NU_CLR_EXT_BANK[:, None] * dt_b_anc[None, :])).reshape(n_clr, T, N)

        for j in range(k):
            # Family 0: Frame-A inherit modes (21)
            ph_i = 2.0 * np.pi * (
                nu_inh_at_anc[j, :, None] * dt_b_anc[None, :]
                + 0.5 * atoms_a.chirp_bank_a[j, :, None] * (dt_b_anc[None, :] ** 2)
            )
            rot_bank[j, :n_inh] = np.exp(1j * ph_i).reshape(n_inh, T, N)
            mode_family[j, :n_inh] = 0
            mode_sign_pos[j, :n_inh] = nu_inh_at_anc[j] > 0.0

            # Family 1: B-only clear modes (20)
            rot_bank[j, n_inh : n_inh + n_clr] = rot_clr
            mode_family[j, n_inh : n_inh + n_clr] = 1
            mode_sign_pos[j, n_inh : n_inh + n_clr] = _NU_CLR_EXT_BANK > 0.0

            # Weight split between inherit / clear / piecewise-midframe (coupled to link-level Frame-A coherence!)
            w_inh_raw = float(np.clip(bundle_a.w_ray_classical[j], 0.02, 0.98))
            link_coherence = float(np.max(bundle_a.w_ray_classical))
            if use_piecewise_midframe:
                # A midframe partial shift originates from a Frame-A-coherent link; full resets have link_coherence ~ 0.
                p_mid_tot = 0.02 + 0.18 * link_coherence
                p_inh_tot = (1.0 - p_mid_tot) * w_inh_raw
                p_clr_tot = (1.0 - p_mid_tot) * (1.0 - w_inh_raw)
            else:
                p_mid_tot = 0.0
                p_inh_tot = w_inh_raw
                p_clr_tot = 1.0 - w_inh_raw

            prior_mu_u[j, :n_inh] = mu_inh_anc_u[j]
            prior_var_u[j, :n_inh] = v_inh_anc_u[j]
            prior_logw[j, :n_inh] = math.log(p_inh_tot) + atoms_a.log_w_bank_a[j]

            prior_var_u[j, n_inh : n_inh + n_clr] = p_clear_u[j]
            # Mildly favor ID passband over fast OOD tails
            clr_prior = np.array([1.0] * 8 + [0.35, 0.35] + [1.0] * 8 + [0.35, 0.35], dtype=REAL_DTYPE)
            clr_prior /= float(clr_prior.sum())
            prior_logw[j, n_inh : n_inh + n_clr] = math.log(p_clr_tot) + np.log(clr_prior)

            # Family 2: Phase-continuous piecewise midframe shift modes (16)
            if use_piecewise_midframe:
                # Anchor Frame-A trajectory at best Frame-A mode for ray j
                h_best_a = int(np.argmax(atoms_a.log_w_bank_a[j]))
                nu_a0 = float(atoms_a.nu_bank_a[j, h_best_a])
                ch_a0 = float(atoms_a.chirp_bank_a[j, h_best_a])
                mu_a0 = complex(atoms_a.mu_bank_a[j, h_best_a]) / s_mag
                v_a0 = float(v_inh_anc_u[j, h_best_a])
                sgn_a = 1.0 if (nu_a0 + ch_a0 * (30.0 - t_a_mean)) >= 0.0 else -1.0

                dt_b_from_a = _TIME_B_FULL - t_a_mean  # [T, N]
                phi_pre_full = 2.0 * np.pi * (nu_a0 * dt_b_from_a + 0.5 * ch_a0 * (dt_b_from_a ** 2))

                m_off = n_inh + n_clr
                idx_m = 0
                for s_row in MID_SHIFT_ONSET_GRID:
                    t_s = float(T + s_row) * ROW_STEP
                    dt_s_a = t_s - t_a_mean
                    phi_at_s = 2.0 * np.pi * (nu_a0 * dt_s_a + 0.5 * ch_a0 * (dt_s_a ** 2))
                    mask_post = _TIME_B_FULL >= t_s
                    dt_post = np.where(mask_post, _TIME_B_FULL - t_s, 0.0)
                    for mag_post in MID_SHIFT_POST_MAGS:
                        nu_post = -sgn_a * float(mag_post)
                        phi_pw = np.where(mask_post, phi_at_s + 2.0 * np.pi * nu_post * dt_post, phi_pre_full)
                        # Value of phase at anchor midpoint (exact phase continuity across t_s!)
                        if t_anchor < t_s:
                            phi_anc = 2.0 * np.pi * (nu_a0 * (t_anchor - t_a_mean) + 0.5 * ch_a0 * ((t_anchor - t_a_mean) ** 2))
                        else:
                            phi_anc = phi_at_s + 2.0 * np.pi * nu_post * (t_anchor - t_s)
                        rot_bank[j, m_off + idx_m] = np.exp(1j * (phi_pw - phi_anc))
                        # Analytic Frame-A gain transport through t_s to t_anchor (with gentle extra variance)
                        prior_mu_u[j, m_off + idx_m] = mu_a0 * complex(math.cos(phi_anc), math.sin(phi_anc))
                        prior_var_u[j, m_off + idx_m] = max(v_a0 + 0.15 * p_atom_u[j], 0.08 * p_clear_u[j])
                        prior_logw[j, m_off + idx_m] = math.log(p_mid_tot / float(n_mid))
                        mode_family[j, m_off + idx_m] = 2
                        mode_sign_pos[j, m_off + idx_m] = nu_post > 0.0
                        idx_m += 1

        ray_power_order = np.argsort(-atoms_a.p_atom_a)

    # Helper: Single-counting damped Jacobi conditioning on arbitrary known/virtual blocks
    def _fit_jacobi_single_count(
        rows_known: Sequence[int],
        words_known: np.ndarray,    # [B, N]
        noises_u: np.ndarray,       # [B]
        y_inf_u: np.ndarray,        # [T, N] with targets & probe masked to NaN!
        eff_prior_logw: np.ndarray, # [K, H] strictly containing ONLY Frame-A + incremental AUX delta (NO anchor!)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        b_cnt = len(rows_known)
        m_tot = b_cnt * N
        y_k = y_inf_u[list(rows_known)].reshape(m_tot)
        w_inv = np.repeat(1.0 / np.asarray(noises_u, dtype=REAL_DTYPE), N)

        u_blocks = np.einsum("knm,bm->kbn", d_mats, words_known)
        rot_k = rot_bank[:, :, list(rows_known), :].reshape(k, n_tot_h, m_tot)
        design = rot_k * u_blocks.reshape(k, 1, m_tot)
        energy = np.sum((design.real ** 2 + design.imag ** 2) * w_inv[None, None, :], axis=-1)

        post_v = prior_var_u / (1.0 + prior_var_u * energy)
        init_prec_m = prior_mu_u / prior_var_u
        ev_base = eff_prior_logw + np.log(post_v / prior_var_u) - (np.abs(prior_mu_u) ** 2) / prior_var_u

        logw_cur = eff_prior_logw.copy()
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
        return mean_all, var_all, logw_cur, mean_h, post_v

    # Helper: Pure incremental full 32x32 AUX covariance log-likelihood delta (ZERO anchor evidence added!)
    def _compute_incremental_aux_cov32_delta(
        aux_rows: tuple[int, ...],
        y_inf_u: np.ndarray,
        mean_anc_u: np.ndarray,
        var_anc_u: np.ndarray,
        lw_anc: np.ndarray,
        mh_anc: np.ndarray,
        pv_anc: np.ndarray,
    ) -> np.ndarray:
        """Return pure incremental `delta_aux [K, H]` from 5 AUX rows ONLY (no anchor log-evidence)."""
        y_aux_u = y_inf_u[list(aux_rows)]
        delta_aux = np.zeros((k, n_tot_h), dtype=REAL_DTYPE)
        cur_m_aux = mean_anc_u[:, list(aux_rows), :].copy()
        cur_v_aux = var_anc_u[:, list(aux_rows), :].copy()

        # Score top-3 strongest rays sequentially via full 32x32 Hermitian Cholesky
        for j in ray_power_order[: min(3, k)]:
            # Evaluate a rich candidate set covering ALL families (inherit, clear+, clear-, piecewise)
            cands_list: list[np.ndarray] = [np.argsort(-lw_anc[j])[:10]]
            if n_inh > 0:
                cands_list.append(np.argsort(-lw_anc[j, :n_inh])[:6])
            cands_list.append(n_inh + np.argsort(-lw_anc[j, n_inh : n_inh + n_clr])[:10])
            if n_mid > 0:
                cands_list.append(n_inh + n_clr + np.arange(n_mid, dtype=np.int64))
            cand_h = np.unique(np.concatenate(cands_list))

            ll_vals = np.empty(len(cand_h), dtype=REAL_DTYPE)
            m_tmp = cur_m_aux.copy()
            v_tmp = cur_v_aux.copy()
            for ci, h_idx in enumerate(cand_h):
                m_tmp[j] = mh_anc[j, h_idx] * rot_bank[j, h_idx, list(aux_rows), :]
                v_tmp[j] = float(pv_anc[j, h_idx])
                ll_vals[ci] = offgrid_cov32_loglik(y_aux_u, m_tmp, v_tmp, d_mats)

            # Shift relative to best evaluated candidate; give unevaluated low-prior modes a conservative floor
            max_ll = float(np.max(ll_vals))
            min_rel = float(np.min(ll_vals - max_ll))
            delta_j = np.full(n_tot_h, AUX_COV_TEMP * (min_rel - 2.0), dtype=REAL_DTYPE)
            delta_j[cand_h] = AUX_COV_TEMP * (ll_vals - max_ll)
            delta_aux[j] = delta_j

            # Update running ray-j mean on AUX for next ray's coordinate sweep (using temp posterior)
            lw_tmp = lw_anc[j] + delta_j
            lw_tmp -= float(logsumexp(lw_tmp))
            w_tmp = np.exp(lw_tmp)
            cur_m_aux[j] = np.einsum("h,hrn->rn", w_tmp * mh_anc[j], rot_bank[j][:, list(aux_rows), :])
            sec_tmp = float(np.sum(w_tmp * (np.abs(mh_anc[j]) ** 2 + pv_anc[j])))
            cur_v_aux[j] = np.maximum(sec_tmp - np.abs(cur_m_aux[j]) ** 2, 1e-12)

        return delta_aux

    # Helper: Off-grid Wave-1 multi-mode union decoder on coherent trajectory clusters of dominant ray
    def _decode_offgrid_wave1_multimode(
        r_h: int,
        y_row_u: np.ndarray,
        mean_u_all: np.ndarray,
        var_u_all: np.ndarray,
        lw_cur: np.ndarray,
        mh_cur: np.ndarray,
        pv_cur: np.ndarray,
    ):
        dom_k = int(ray_power_order[0])
        w_dom = np.exp(lw_cur[dom_k])

        # Partition dominant ray's modes into up to 3 coherent physical clusters:
        #   C0: inherit / pre-shift positive-or-main cluster (family==0)
        #   C1: clear positive (family==1 & pos)
        #   C2: clear negative (family==1 & neg)
        #   C3: piecewise midframe (family==2)
        buckets = [
            mode_family[dom_k] == 0,
            (mode_family[dom_k] == 1) & mode_sign_pos[dom_k],
            (mode_family[dom_k] == 1) & (~mode_sign_pos[dom_k]),
            mode_family[dom_k] == 2,
        ]
        active: list[tuple[float, np.ndarray]] = []
        for mask in buckets:
            if np.any(mask):
                p_m = float(np.sum(w_dom[mask]))
                if p_m >= 0.04:
                    active.append((p_m, mask))
        active.sort(key=lambda t: -t[0])
        active = active[:2]  # Top-2 coherent trajectory clusters

        if (not use_wave1_multimode) or len(active) < 2:
            h_single = np.zeros((N, N), dtype=COMPLEX_DTYPE)
            for j in range(k):
                h_single += mean_u_all[j, r_h, :, None] * d_mats[j]
            nv_single = 1.0 + var_u_all[:, r_h, :].sum(axis=0)
            adj = h_single.conj().T / nv_single[None, :]
            cov = np.linalg.inv(_EYE_N + adj @ h_single)
            est = cov @ (adj @ y_row_u)
            u_var = np.diagonal(cov).real
            from p8.coding import qam_llrs
            llrs_s = qam_llrs(est, u_var).reshape(128)
            res_s = decode_with_blind_margin(
                y_row_u, h_single, nv_single, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_s
            )
            return res_s, h_single, nv_single

        from p8.coding import qam_llrs
        h_modes: list[np.ndarray] = []
        nv_modes: list[np.ndarray] = []
        log_p_list: list[float] = []
        dec_results = []
        p_norm = sum(p for p, _ in active)

        # Base contribution from non-dominant rays
        h_other = np.zeros((N, N), dtype=COMPLEX_DTYPE)
        v_other = np.zeros(N, dtype=REAL_DTYPE)
        for j in range(k):
            if j != dom_k:
                h_other += mean_u_all[j, r_h, :, None] * d_mats[j]
                v_other += var_u_all[j, r_h, :]

        for p_m, mask in active:
            w_sub = np.where(mask, w_dom, 0.0)
            w_sub /= float(np.sum(w_sub))
            m_dom_r = np.einsum("h,hn->n", w_sub * mh_cur[dom_k], rot_bank[dom_k, :, r_h, :])
            sec_dom = float(np.sum(w_sub * (np.abs(mh_cur[dom_k]) ** 2 + pv_cur[dom_k])))
            v_dom_r = np.maximum(sec_dom - np.abs(m_dom_r) ** 2, 1e-12)

            h_m = h_other + m_dom_r[:, None] * d_mats[dom_k]
            nv_m = 1.0 + v_other + v_dom_r
            adj = h_m.conj().T / nv_m[None, :]
            cov = np.linalg.inv(_EYE_N + adj @ h_m)
            est = cov @ (adj @ y_row_u)
            llrs_m = qam_llrs(est, np.diagonal(cov).real).reshape(128)
            dec_m = decode_strong_sc(
                y_row_u, h_m, nv_m, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_m
            )
            h_modes.append(h_m)
            nv_modes.append(nv_m)
            log_p_list.append(math.log(max(p_m / p_norm, 1e-300)))
            dec_results.append(dec_m)

        # Build deduplicated union of candidates + 64 1-flips around each mode winner
        stacks = []
        for dec_m in dec_results:
            sd = pack_codewords(dec_m.codeword)
            stacks.append(sd[None, :])
            stacks.append(sd[None, :] ^ _GEN_U64)
            stacks.append(dec_m.candidates_u64)
        all_u64 = np.vstack(stacks)
        dt_pair = np.dtype([("c0", np.uint64), ("c1", np.uint64)])
        uniq_u64 = np.ascontiguousarray(np.unique(np.ascontiguousarray(all_u64).view(dt_pair).ravel()).view(np.uint64).reshape(-1, 2))

        h_stack = np.stack(h_modes, axis=0)
        nv_stack = np.stack(nv_modes, axis=0)
        b0, b1, best_ll, _, _, sec_ll, win_m, win_fit, n_val = _score_offgrid_multimode_u64(
            uniq_u64,
            _H_U64,
            np.ascontiguousarray(y_row_u.real, dtype=REAL_DTYPE),
            np.ascontiguousarray(y_row_u.imag, dtype=REAL_DTYPE),
            np.asarray(log_p_list, dtype=REAL_DTYPE),
            np.ascontiguousarray(h_stack.real, dtype=REAL_DTYPE),
            np.ascontiguousarray(h_stack.imag, dtype=REAL_DTYPE),
            np.ascontiguousarray(nv_stack, dtype=REAL_DTYPE),
            _NIBBLE_RE,
            _NIBBLE_IM,
        )
        cw = unpack_codewords(np.array([b0, b1], dtype=np.uint64))
        margin = float(max(0.0, float(best_ll) - float(sec_ll))) if n_val >= 2 else 0.0
        bp_conv = any(bool(dm.bp_converged and np.array_equal(dm.codeword, cw)) for dm in dec_results)
        rel = reliability_weight_from_margin(margin, bp_conv, fit_ratio=float(win_fit))

        from p11.wavefront_pipeline import BlindMarginDecodeResult
        res_union = BlindMarginDecodeResult(
            info_bits=cw[:64].copy(),
            codeword=cw,
            d_best=float(-best_ll),
            d_second=float(-sec_ll),
            margin=margin,
            fit_ratio=float(win_fit),
            bp_converged=bp_conv,
            reliability=rel,
            candidates_u64=uniq_u64,
            n_candidates=int(n_val),
            elapsed_ms=0.0,
        )
        return res_union, h_modes[int(win_m)], nv_modes[int(win_m)]

    # --- Step 1: Strictly Target-Excluded Roving Bilateral 3-Fold Execution (6+6+6 targets, 5 AUX, 7 helpers/fold) ---
    t_folds, a_folds = make_roving_bilateral_3fold(r_anchor, probe_row=probe_row)
    all_data_rows = tuple(r for r in range(T) if r not in (r_anchor, probe_row))

    owner_h_mix = np.zeros((T, N, N), dtype=COMPLEX_DTYPE)
    owner_noise_mix = np.ones((T, N), dtype=REAL_DTYPE) * n0
    owner_llrs = np.zeros((T, N, 4), dtype=REAL_DTYPE)
    owner_mean_ray = np.zeros((k, T, N), dtype=COMPLEX_DTYPE)
    owner_var_ray = np.zeros((k, T, N), dtype=REAL_DTYPE)
    probe_preds: list[np.ndarray] = []
    probe_vars: list[np.ndarray] = []
    ledger_records: list[dict[str, Any]] = []
    tot_acc = 0

    for fold_idx, (t_rows, a_rows) in enumerate(zip(t_folds, a_folds)):
        # Structural zero-leakage firewall: mask all 9 target rows + probe_row to NaN!
        y_inf_u = y_b_u.copy()
        y_inf_u[list(t_rows)] = complex(np.nan, np.nan)
        y_inf_u[probe_row] = complex(np.nan, np.nan)

        # Anchor-only initial fit (solely to supply conditional gain means for AUX covariance scoring)
        m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc = _fit_jacobi_single_count(
            (r_anchor,), obs.pilot_tx_b[:1], np.array([1.0], dtype=REAL_DTYPE), y_inf_u, prior_logw
        )

        # Compute pure incremental AUX covariance log-likelihood delta (ZERO anchor evidence inside!)
        if use_aux_cov:
            delta_aux = _compute_incremental_aux_cov32_delta(
                a_rows, y_inf_u, m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc
            )
            eff_prior_logw = prior_logw + delta_aux
            eff_prior_logw -= logsumexp(eff_prior_logw, axis=1, keepdims=True)
            # Re-condition anchor ONCE with eff_prior_logw (zero double counting!)
            m_u, v_u, lw_cur, mh_cur, pv_cur = _fit_jacobi_single_count(
                (r_anchor,), obs.pilot_tx_b[:1], np.array([1.0], dtype=REAL_DTYPE), y_inf_u, eff_prior_logw
            )
        else:
            eff_prior_logw = prior_logw
            m_u, v_u, lw_cur, mh_cur, pv_cur = m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc

        # Roving bilateral 2-wave wavefront assimilation
        if use_wavefront:
            forbidden = set(t_rows) | set(a_rows) | {r_anchor, probe_row}
            helpers = sorted([r for r in all_data_rows if r not in forbidden], key=lambda x: abs(x - r_anchor))
            wave1 = helpers[:2]
            wave2 = helpers[2:]

            known_rows_list: list[int] = [r_anchor]
            known_words_list: list[np.ndarray] = [obs.pilot_tx_b[0].copy()]
            known_noises_list: list[float] = [1.0]

            for w_idx, wave in enumerate((wave1, wave2)):
                if not wave:
                    continue
                added = False
                for r_h in wave:
                    if w_idx == 0:
                        res_h, eval_h_u, eval_nv_u = _decode_offgrid_wave1_multimode(
                            r_h, y_inf_u[r_h], m_u, v_u, lw_cur, mh_cur, pv_cur
                        )
                    else:
                        eval_h_u = np.zeros((N, N), dtype=COMPLEX_DTYPE)
                        for j in range(k):
                            eval_h_u += m_u[j, r_h, :, None] * d_mats[j]
                        eval_nv_u = 1.0 + v_u[:, r_h, :].sum(axis=0)
                        adj = eval_h_u.conj().T / eval_nv_u[None, :]
                        cov_l = np.linalg.inv(_EYE_N + adj @ eval_h_u)
                        est_l = cov_l @ (adj @ y_inf_u[r_h])
                        from p8.coding import qam_llrs
                        llrs_l = qam_llrs(est_l, np.diagonal(cov_l).real).reshape(128)
                        res_h = decode_with_blind_margin(
                            y_inf_u[r_h], eval_h_u, eval_nv_u, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_l
                        )

                    if res_h.reliability >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                        x_hat = qam(res_h.codeword).astype(COMPLEX_DTYPE)
                        y_hat = eval_h_u @ x_hat
                        diff = y_inf_u[r_h] - y_hat
                        resid2 = diff.real ** 2 + diff.imag ** 2
                        u_stat = max(float(np.sum(resid2) / np.sum(eval_nv_u)), float(np.mean(resid2 / eval_nv_u)))
                        w_res = redescending_residual_weight(u_stat)
                        alpha_eff = float(min(DEFAULT_ALPHA_MAX, res_h.reliability * w_res))
                        if alpha_eff >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                            n_eff_u = 1.0 / alpha_eff + DEFAULT_VIRTUAL_VAR_FLOOR
                            known_rows_list.append(r_h)
                            known_words_list.append(x_hat)
                            known_noises_list.append(n_eff_u)
                            tot_acc += 1
                            ledger_records.append({
                                "fold": int(fold_idx),
                                "wave": int(w_idx),
                                "row": int(r_h),
                                "injected_info": res_h.info_bits.copy(),
                                "alpha": float(alpha_eff),
                                "margin": float(res_h.margin),
                            })
                            added = True

                if added:
                    # Single-counting refit from pure `eff_prior_logw` on `(anchor U accepted_helpers)`!
                    m_u, v_u, lw_cur, mh_cur, pv_cur = _fit_jacobi_single_count(
                        known_rows_list,
                        np.stack(known_words_list, axis=0),
                        np.asarray(known_noises_list, dtype=REAL_DTYPE),
                        y_inf_u,
                        eff_prior_logw,
                    )

        # Burst noise guard on each owned target row (strictly C*-invariant excess-energy estimator)
        nm_fold = np.ones(T, dtype=REAL_DTYPE)
        if use_burst_guard:
            for r_t in t_rows:
                # Expected unit-noise total energy E[||y_u||^2] = sum_{k,n} (|m_k|^2 + v_k) + N
                sig_e = float(np.sum(m_u[:, r_t, :].real ** 2 + m_u[:, r_t, :].imag ** 2 + v_u[:, r_t, :]))
                exp_e = sig_e + float(N)
                obs_e = float(np.sum(y_b_u[r_t].real ** 2 + y_b_u[r_t].imag ** 2))
                if obs_e > 1.65 * exp_e:
                    est_mult = max(1.0, (obs_e - sig_e) / float(N))
                    nm_fold[r_t] = float(min(25.0, est_mult))

        h_f, nv_f, llrs_f = _compute_all_row_llrs(
            m_u * s_mag, v_u * s_var, d_mats, y_b_safe, n0, (r_anchor,), probe_row, noise_mult=nm_fold
        )
        for r_t in t_rows:
            owner_h_mix[r_t] = h_f[r_t]
            owner_noise_mix[r_t] = nv_f[r_t]
            owner_llrs[r_t] = llrs_f[r_t]
            owner_mean_ray[:, r_t, :] = m_u[:, r_t, :] * s_mag
            owner_var_ray[:, r_t, :] = v_u[:, r_t, :] * s_var

        p_pred_f = np.sum((m_u[:, probe_row, :] * s_mag) * u_probe, axis=0)
        p_var_f = n0 + np.sum((v_u[:, probe_row, :] * s_var) * (np.abs(u_probe) ** 2), axis=0)
        probe_preds.append(p_pred_f)
        probe_vars.append(p_var_f)

    # Equal-weight K-fold probe mixture (exact within + between)
    p_stack = np.stack(probe_preds, axis=0)
    pv_stack = np.stack(probe_vars, axis=0)
    mix_probe_pred = np.mean(p_stack, axis=0)
    mix_probe_var = np.mean(pv_stack, axis=0) + np.mean(np.abs(p_stack - mix_probe_pred[None, :]) ** 2, axis=0)
    owner_llrs[r_anchor] = 0.0
    owner_llrs[probe_row] = 0.0

    hw_val = 0.0 if delay_reset_triggered else 0.5
    gate_out = GateOutput(
        history_weight=np.full(T, hw_val, dtype=REAL_DTYPE),
        probe_weight=float(hw_val),
        mix_llrs=owner_llrs,
        mix_probe_pred=np.asarray(mix_probe_pred, dtype=COMPLEX_DTYPE),
        mix_probe_var=np.asarray(mix_probe_var, dtype=REAL_DTYPE),
        mix_mean_ray=owner_mean_ray,
        mix_var_ray=owner_var_ray,
    )
    return OffGridPhysicsResult(
        gate_out=gate_out,
        h_mix=owner_h_mix,
        noise_mix=owner_noise_mix,
        delay_reset_triggered=bool(delay_reset_triggered),
        used_delays=used_delays.copy(),
        delay_ll_gain_nats=float(delay_ll_gain),
        helpers_accepted=int(tot_acc),
        ledger=tuple(ledger_records),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


__all__ = [
    "AUX_COV_TEMP",
    "DELAY_RESET_OCCAM_NATS",
    "MID_SHIFT_ONSET_GRID",
    "MID_SHIFT_POST_MAGS",
    "OffGridPhysicsResult",
    "discover_b_only_delays",
    "run_p12_offgrid_physics",
    "warmup_p12_offgrid_numba",
]
