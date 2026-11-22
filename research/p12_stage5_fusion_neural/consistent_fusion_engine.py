"""Consistent (`655de6a`) Off-Grid Posterior Fusion Engine with Content-Addressed Weighting.

Strictly preserves every mathematical consistency invariant of `p12/offgrid_receiver.py` (`655de6a`):
  1. ZERO B-anchor posterior recycling into trajectory family priors (fixed prospective family
     masses `(1/3, 1/3, 1/3)` or `(1/2, 1/2, 0)`; Frame-A-only inherited mode priors; read-only
     base priors & read-only effective priors; each Jacobi refit starts fresh from `eff_prior_logw`
     on the deduplicated set `(anchor U accepted_helpers)`).
  2. Strict bilateral 3-fold target exclusion (`make_roving_bilateral_3fold`): every fold masks
     its 6 owned targets AND `probe_row` to `NaN` during AUX scoring, Wave-1/2 decoding,
     content-addressed weighting, and all channel refits.
  3. Exact `C*` invariance/equivariance and ZERO hardcoded `N0` anywhere.

Upgrades built strictly on top of this zero-leakage physics core:
  - Coherent trajectory-mixture extraction (`build_coherent_row_mixture`): preserves distinct
    phase-coherent physical modes (primary lobe, grating lobe, opposite-sign Dopplers, distinct
    speeds, piecewise midframe onsets, and bimodal 2nd-ray splits) instead of collapsing them
    into a phase-cancelling linear mean channel.
  - Exact probability-domain Mixture Bit-LLR + Exact Marginalized Mixture Codeword MAP decoding
    (`decode_row_mixture_llr`), rigorously following `research/posterior_fusion` §5.2:
      * Target-updated branch posterior `beta_m = softmax(log w_m + gamma_obs * log p(y|H_m, v_m))`
        feeds exact probability-domain bit LLR fusion (log-sum-exp over `beta_m * P(b|y, m)`),
        NEVER linear LLR averaging.
      * Multi-branch codeword union is scored by the exact marginalized mixture likelihood
        `log sum_m w_m CN(y | H_m x(c), diag(v_m))` using the pre-target fold prior `w_m`
        (ZERO double-counting of target observation `y`!).
  - Dev-calibrated Content-Addressed AUX & Helper Reliability Weighting (`ContentAddressingConfig`),
    rigorously following `research/content_addressing_scalar` §3–§6:
      * Content-addressed reliability weights over the 5 AUX rows (down-weighting energy/kurtosis/
        covariance outliers before computing `delta_aux`).
      * Content-addressed cross-helper consistency & calibrated precision weighting over the 7
        legal helpers in each fold, plus optional target-position-queried reweighting using ONLY
        the public target row index `r_t` (ZERO access to `y_b[r_t]`).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from p8.coding import qam, qam_llrs
from p8c.link import PILOTS as SC_PILOTS
from p9.decoder import (
    _CHECK_OFFSETS,
    _EDGE_BITS,
    _bp_sum_product,
    decode_strong_sc,
    is_valid_codeword,
    pack_codewords,
    unpack_codewords,
)
from p11.contract import (
    COMPLEX_DTYPE,
    N,
    PROBE_ROW,
    REAL_DTYPE,
    T,
    OffGridObservation,
    OffGridTruth,
    fractional_delay_matrix,
)
from p11.offgrid_channel import (
    K_HAT,
    ROW_STEP,
    _EYE_N,
    _TIME_A_FULL,
    _TIME_B_FULL,
    _compute_all_row_llrs,
    _strict_int,
    _validate_n0,
    discover_frame_a_atoms,
)
from p11.wavefront_pipeline import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_MIN_ASSIMILATE_WEIGHT,
    DEFAULT_VIRTUAL_VAR_FLOOR,
    BlindMarginDecodeResult,
    _GEN_U64,
    _H_U64,
    _NIBBLE_IM,
    _NIBBLE_RE,
    decode_with_blind_margin,
    redescending_residual_weight,
    reliability_weight_from_margin,
)
from p12.contract import OFDM_PILOT_COMB
from p12.offgrid_physics import make_roving_bilateral_3fold, offgrid_cov32_loglik
from p12.offgrid_receiver import (
    AUX_COV_TEMP,
    DELAY_RESET_OCCAM_NATS,
    FAMILY_PRIOR_WITH_MIDFRAME,
    FAMILY_PRIOR_WITHOUT_MIDFRAME,
    MID_SHIFT_ONSET_GRID,
    MID_SHIFT_POST_MAGS,
    _NU_CLR_EXT_BANK,
    _anchor_delay_marginal_loglik,
    _score_offgrid_multimode_u64,
    discover_b_only_delays,
    warmup_p12_offgrid_numba,
)


# ---------------------------------------------------------------------------
# 1. Configuration & Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentAddressingConfig:
    """Dev-calibrated content-addressing & full-support consistent inference parameters.

    Strictly C*-invariant, zero B-anchor prior recycling, and zero target/probe leakage.
    """

    # Full-support AUX + converged single-count Jacobi (still 100% 655de6a consistent!)
    full_aux_bank: bool = True
    aux_max_rays: int = 4          # Top-4 rays by anchor posterior energy (covers all ID P in {2,3,4})
    jacobi_iters: int = 16
    jacobi_damping: float = 0.72
    b_power_floor_frac: float = 0.25

    # Thresholded AUX content reliability weighting (clean rows = 1.0, burst/outlier rows suppressed)
    aux_cov_temp: float = 0.75
    use_aux_content_weights: bool = True
    aux_energy_thresh: float = 0.42
    aux_energy_penalty: float = 3.0
    aux_kurt_thresh: float = 0.60
    aux_kurt_penalty: float = 0.75

    # Helper burst guard & calibrated target-position-only content query (NEVER reads y_b[r_t]!)
    use_helper_burst_guard: bool = True
    use_target_pos_query: bool = True
    target_pos_temp: float = 0.45


@dataclass(frozen=True)
class MixtureDecodeConfig:
    """16-QAM phase-coherent mixture construction & exact mixture-LLR LDPC decoding parameters."""

    cos_merge_thresh: float = 0.92     # ~23 deg 16-QAM phase-coherent mode merging threshold
    rel_dist_merge_thresh: float = 0.22
    max_dom_clusters: int = 5          # Dominant ray coherent clusters
    enable_second_ray_split: bool = True
    second_ray_energy_ratio: float = 0.18
    second_ray_min_mass: float = 0.14
    max_joint_branches: int = 8        # Joint (ray0 x ray1) coherent hypotheses
    min_branch_mass: float = 0.015
    obs_cov_gamma: float = 0.85        # 32x32 Gaussian marginal likelihood weight for beta_m
    runner_up_beta_thresh: float = 0.035
    max_decode_branches: int = 4       # Up to 4 active coherent branches generating codeword union
    runner_up_list_size: str = "small" # "small" (321 cands, fast & effective) or "large" (2641 cands)
    use_mixture_in_wave1: bool = True
    use_mixture_in_wave2: bool = True


@dataclass
class FoldPosteriorSnapshot:
    """Strictly target-excluded fold state for downstream mixture decoding or Stage 5 neural refiner."""

    fold_idx: int
    t_rows: tuple[int, ...]
    a_rows: tuple[int, ...]
    helper_rows: tuple[int, ...]
    r_anchor: int
    probe_row: int
    delay_reset_triggered: bool
    used_delays: np.ndarray            # [K]
    d_mats: np.ndarray                 # [K, N, N] complex128
    u_probe: np.ndarray                # [K, N] complex128
    rot_bank: np.ndarray               # [K, H, T, N] complex128
    mode_family: np.ndarray            # [K, H] int64 (0=inh, 1=clr, 2=pw)
    mode_sign_pos: np.ndarray          # [K, H] bool
    mode_nu_eff: np.ndarray            # [K, H] float64 instantaneous mid-frame Doppler per mode
    ray_power_order: np.ndarray        # [K] int64
    p_atom_u: np.ndarray               # [K] float64
    prior_mu_u: np.ndarray             # [K, H] complex128
    prior_var_u: np.ndarray            # [K, H] float64
    eff_prior_logw: np.ndarray         # [K, H] float64
    # Fitted fold posterior moments & modes (strictly target/probe excluded!)
    m_u: np.ndarray                    # [K, T, N] complex128
    v_u: np.ndarray                    # [K, T, N] float64
    lw_cur: np.ndarray                 # [K, H] float64
    mh_cur: np.ndarray                 # [K, H] complex128
    pv_cur: np.ndarray                 # [K, H] float64
    # Per-row observed fold-only metadata (EXACT ZERO on t_rows and probe_row!)
    accepted_rows: tuple[int, ...]
    known_words_u: np.ndarray          # [B_known, N] complex128 (anchor + accepted helpers)
    known_noises_u: np.ndarray         # [B_known] float64
    helper_alpha_row: np.ndarray       # [T] float64 (0 on targets/probe/unaccepted)
    helper_margin_row: np.ndarray      # [T] float64 (0 on targets/probe/unaccepted)
    row_resid_row: np.ndarray          # [T] float64 (0 on targets/probe)
    row_energy_anom: np.ndarray        # [T] float64 (0 on targets/probe)
    row_kurt_anom: np.ndarray          # [T] float64 (0 on targets/probe)
    obs_matched_resid: np.ndarray      # [K, T] complex128 U(1)-equivariant (0 on targets/probe)
    aux_weights: np.ndarray            # [5] float64
    # Optional target-position-queried per-target-row posterior overrides (still ZERO target-obs access!)
    per_target_lw: Optional[dict[int, np.ndarray]] = None
    per_target_mh: Optional[dict[int, np.ndarray]] = None
    per_target_pv: Optional[dict[int, np.ndarray]] = None
    per_target_m_u: Optional[dict[int, np.ndarray]] = None
    per_target_v_u: Optional[dict[int, np.ndarray]] = None


@dataclass
class ConsistentFoldsResult:
    """Output of the consistent 3-fold off-grid inference core."""

    folds: tuple[FoldPosteriorSnapshot, ...]
    n0: float
    r_anchor: int
    probe_row: int
    delay_reset_triggered: bool
    used_delays: np.ndarray
    delay_ll_gain_nats: float
    helpers_accepted: int
    ledger: tuple[dict[str, Any], ...]
    aux_reads: int
    helper_key_reads: int
    helper_value_reads: int
    elapsed_ms: float


@dataclass(frozen=True)
class RowMixtureBranches:
    """Coherent joint trajectory hypotheses at one row `r` (in unit-noise gauge)."""

    log_w: np.ndarray          # [M] float64 pre-target fold prior log-weights (sum exp == 1)
    h_u: np.ndarray            # [M, N, N] complex128 branch channel matrices in unit-noise gauge
    nv_u: np.ndarray           # [M, N] float64 effective diagonal noise+within-branch var (unit-noise)
    h_mean_u: np.ndarray       # [N, N] complex128 exact posterior mean channel
    nv_mean_u: np.ndarray      # [N] float64 exact posterior total diagonal variance
    mean_ray_u: np.ndarray     # [K, N] complex128 per-ray posterior mean at row r
    var_ray_u: np.ndarray      # [K, N] float64 per-ray posterior total variance at row r


# ---------------------------------------------------------------------------
# 2. Exact Probability-Domain Mixture Bit-LLR & 32x32 Marginal Log-Likelihood
# ---------------------------------------------------------------------------


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 / (1 + exp(-x))) = -softplus(-x)."""
    arr = np.asarray(x, dtype=REAL_DTYPE)
    return -np.logaddexp(0.0, -arr)


def mix_branch_bit_llrs_exact(
    branch_llrs: np.ndarray,   # [M, D] float64, positive = bit 1
    log_beta: np.ndarray,      # [M] float64, logsumexp(log_beta) == 0
) -> np.ndarray:
    """Exact probability-domain mixture bit-LLR via log-sum-exp (NEVER linear LLR averaging!).

    Computes:
      Lambda_j = log(sum_m beta_m * sigmoid(L_{m,j})) - log(sum_m beta_m * sigmoid(-L_{m,j}))
    """
    llrs = np.asarray(branch_llrs, dtype=REAL_DTYPE)
    lb = np.asarray(log_beta, dtype=REAL_DTYPE).reshape(-1)
    lb = lb - float(logsumexp(lb))
    log_p1 = _log_sigmoid(llrs)
    log_p0 = _log_sigmoid(-llrs)
    num = logsumexp(lb[:, None] + log_p1, axis=0)
    den = logsumexp(lb[:, None] + log_p0, axis=0)
    return np.clip(num - den, -40.0, 40.0).astype(REAL_DTYPE)


def _branch_marginal_gaussian_loglik(
    y_u: np.ndarray,      # [N] complex128
    h_u: np.ndarray,      # [N, N] complex128
    nv_u: np.ndarray,     # [N] float64
) -> float:
    """Exact 32x32 Hermitian Cholesky Gaussian marginal log-likelihood log CN(y | 0, H H^H + diag(nv))."""
    c_mat = h_u @ h_u.conj().T
    c_mat[np.diag_indices(N)] += nv_u
    c_mat = 0.5 * (c_mat + c_mat.conj().T)
    chol = np.linalg.cholesky(c_mat)
    logdet = 2.0 * float(np.sum(np.log(np.maximum(np.real(np.diag(chol)), 1e-300))))
    z = np.linalg.solve(chol, y_u)
    quad = float(np.vdot(z, z).real)
    return float(-float(N) * math.log(math.pi) - logdet - quad)


def _lmmse_row_llrs_unit(
    y_u: np.ndarray,      # [N] complex128
    h_u: np.ndarray,      # [N, N] complex128
    nv_u: np.ndarray,     # [N] float64
) -> np.ndarray:
    """LMMSE equalizer -> Gray 16-QAM bit LLRs [128] in unit-noise gauge."""
    adj = h_u.conj().T / nv_u[None, :]
    cov = np.linalg.inv(_EYE_N + adj @ h_u)
    est = cov @ (adj @ y_u)
    u_var = np.diagonal(cov).real
    return qam_llrs(est, u_var).reshape(128).astype(REAL_DTYPE)


def _compute_burst_mult_row(
    y_row_u: np.ndarray,
    m_ray_u: np.ndarray,   # [K, N]
    v_ray_u: np.ndarray,   # [K, N]
) -> float:
    """Exact P12 C*-invariant per-row excess-energy burst noise variance multiplier."""
    sig_e = float(np.sum(m_ray_u.real ** 2 + m_ray_u.imag ** 2 + v_ray_u))
    exp_e = sig_e + float(N)
    obs_e = float(np.sum(y_row_u.real ** 2 + y_row_u.imag ** 2))
    if obs_e > 1.65 * exp_e:
        return float(min(25.0, max(1.0, (obs_e - sig_e) / float(N))))
    return 1.0


# ---------------------------------------------------------------------------
# 3. Phase-Coherent Trajectory Mixture Extraction
# ---------------------------------------------------------------------------


def _cluster_ray_modes_coherent(
    w_ray: np.ndarray,       # [H] float64 (sum == 1)
    mh_ray: np.ndarray,      # [H] complex128
    pv_ray: np.ndarray,      # [H] float64
    rot_row: np.ndarray,     # [H, N] complex128
    *,
    cos_thresh: float = 0.72,
    rel_dist_thresh: float = 0.40,
    max_clusters: int = 4,
    min_cluster_mass: float = 0.015,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Partition one ray's modes at row `r` into phase-coherent clusters (100% C*-invariant).

    Returns list of `(mass, mean_vec [N], var_vec [N])` sorted by descending mass, normalized to sum=1.
    """
    h_cnt = len(w_ray)
    # Complex gain trajectory vector per mode on this row: g_h [H, N]
    g_h = mh_ray[:, None] * rot_row
    g_norm2 = np.sum(g_h.real ** 2 + g_h.imag ** 2, axis=-1)  # [H]
    g_norm = np.sqrt(np.maximum(g_norm2, 1e-30))

    order = np.argsort(-w_ray)
    assigned = np.zeros(h_cnt, dtype=bool)
    clusters: list[tuple[float, np.ndarray, np.ndarray]] = []

    for seed_h in order:
        if assigned[seed_h]:
            continue
        if len(clusters) >= max_clusters:
            break
        if float(w_ray[seed_h]) < 1e-4 and clusters:
            break

        g0 = g_h[seed_h]
        n0_sq = float(g_norm2[seed_h])
        n0 = float(g_norm[seed_h])
        v0 = float(pv_ray[seed_h]) * float(N)

        # Find all unassigned modes phase-coherent with seed_h on row r
        rem_idx = np.flatnonzero(~assigned)
        g_rem = g_h[rem_idx]
        inner_re = np.sum(g_rem.real * g0.real[None, :] + g_rem.imag * g0.imag[None, :], axis=-1)
        cos_sim = inner_re / np.maximum(g_norm[rem_idx] * n0, 1e-30)
        diff = g_rem - g0[None, :]
        dist2 = np.sum(diff.real ** 2 + diff.imag ** 2, axis=-1)
        rel_dist = dist2 / max(n0_sq + v0, 1e-12)

        # Seed always joins; others join if phase-coherent and amplitude-compatible
        join = (rem_idx == seed_h) | ((cos_sim >= cos_thresh) & (rel_dist <= rel_dist_thresh))
        members = rem_idx[join]
        assigned[members] = True

        w_sub = w_ray[members]
        p_c = float(np.sum(w_sub))
        if p_c <= 0.0:
            continue
        w_norm = w_sub / p_c
        m_c = np.einsum("m,mn->n", w_norm, g_h[members])
        sec_c = float(np.sum(w_norm * (np.abs(mh_ray[members]) ** 2 + pv_ray[members])))
        v_c = np.maximum(sec_c - (m_c.real ** 2 + m_c.imag ** 2), 1e-12)
        clusters.append((p_c, m_c, v_c))

    # Any leftover tail modes get merged into their closest existing cluster (if any) or dropped if negligible
    if not clusters:
        m_all = np.einsum("h,hn->n", w_ray, g_h)
        sec_all = float(np.sum(w_ray * (np.abs(mh_ray) ** 2 + pv_ray)))
        v_all = np.maximum(sec_all - (m_all.real ** 2 + m_all.imag ** 2), 1e-12)
        return [(1.0, m_all, v_all)]

    # Filter tiny clusters (keep at least top-1) and renormalize
    kept = [c for idx, c in enumerate(clusters) if idx == 0 or c[0] >= min_cluster_mass]
    tot = sum(c[0] for c in kept)
    return [(c[0] / tot, c[1], c[2]) for c in kept]


def build_coherent_row_mixture(
    row: int,
    lw_ray: np.ndarray,        # [K, H] log-weights
    mh_ray: np.ndarray,        # [K, H] complex128
    pv_ray: np.ndarray,        # [K, H] float64
    rot_bank: np.ndarray,      # [K, H, T, N] complex128
    d_mats: np.ndarray,        # [K, N, N] complex128
    *,
    cfg: MixtureDecodeConfig = MixtureDecodeConfig(),
    noise_mult: float = 1.0,
    delta_g_row_u: Optional[np.ndarray] = None,     # [K, N] complex128 optional neural residual
    var_scale_row: Optional[np.ndarray] = None,     # [K] float64 optional neural var multiplier
) -> RowMixtureBranches:
    """Build coherent joint trajectory hypotheses `(log_w, H_u, nv_u)` at row `row` (unit-noise gauge)."""
    k_cnt = lw_ray.shape[0]
    w_all = np.exp(lw_ray - logsumexp(lw_ray, axis=1, keepdims=True))

    # Exact per-ray posterior moments at row `row`
    m_ray = np.einsum("kh,khn->kn", w_all * mh_ray, rot_bank[:, :, row, :])  # [K, N]
    sec_ray = np.sum(w_all * (np.abs(mh_ray) ** 2 + pv_ray), axis=1)         # [K]
    v_ray = np.maximum(sec_ray[:, None] - (m_ray.real ** 2 + m_ray.imag ** 2), 1e-12)

    if delta_g_row_u is not None:
        m_ray = m_ray + delta_g_row_u
    if var_scale_row is not None:
        v_ray = np.maximum(v_ray * var_scale_row[:, None], 1e-12)

    # Posterior expected energy per ray (strictly C*-invariant ordering!)
    ray_e = np.mean(m_ray.real ** 2 + m_ray.imag ** 2 + v_ray, axis=-1)
    order = np.argsort(-ray_e)
    dom_0 = int(order[0])

    # Cluster dominant ray
    c0 = _cluster_ray_modes_coherent(
        w_all[dom_0],
        mh_ray[dom_0],
        pv_ray[dom_0],
        rot_bank[dom_0, :, row, :],
        cos_thresh=cfg.cos_merge_thresh,
        rel_dist_thresh=cfg.rel_dist_merge_thresh,
        max_clusters=cfg.max_dom_clusters,
        min_cluster_mass=cfg.min_branch_mass,
    )

    # Optionally cluster 2nd strongest ray if strong and genuinely multi-modal
    split_r1 = False
    dom_1 = int(order[1]) if k_cnt >= 2 else -1
    c1: list[tuple[float, np.ndarray, np.ndarray]] = []
    if (
        cfg.enable_second_ray_split
        and k_cnt >= 2
        and float(ray_e[dom_1]) >= cfg.second_ray_energy_ratio * max(float(ray_e[dom_0]), 1e-12)
    ):
        c1_cand = _cluster_ray_modes_coherent(
            w_all[dom_1],
            mh_ray[dom_1],
            pv_ray[dom_1],
            rot_bank[dom_1, :, row, :],
            cos_thresh=cfg.cos_merge_thresh,
            rel_dist_thresh=cfg.rel_dist_merge_thresh,
            max_clusters=2,
            min_cluster_mass=cfg.second_ray_min_mass,
        )
        if len(c1_cand) >= 2 and c1_cand[1][0] >= cfg.second_ray_min_mass:
            split_r1 = True
            c1 = c1_cand

    # Base contribution of unsplit rays
    split_set = {dom_0, dom_1} if split_r1 else {dom_0}
    h_base = np.zeros((N, N), dtype=COMPLEX_DTYPE)
    v_base = np.zeros(N, dtype=REAL_DTYPE)
    for j in range(k_cnt):
        if j not in split_set:
            h_base += m_ray[j, :, None] * d_mats[j]
            v_base += v_ray[j]

    # Exact overall posterior mean channel & variance
    h_mean = np.zeros((N, N), dtype=COMPLEX_DTYPE)
    for j in range(k_cnt):
        h_mean += m_ray[j, :, None] * d_mats[j]
    nv_mean = np.maximum(float(noise_mult) + np.sum(v_ray, axis=0), 1e-12)

    # Assemble joint branches
    dg0 = delta_g_row_u[dom_0] if delta_g_row_u is not None else 0.0
    vs0 = float(var_scale_row[dom_0]) if var_scale_row is not None else 1.0
    joint: list[tuple[float, np.ndarray, np.ndarray]] = []

    if split_r1:
        dg1 = delta_g_row_u[dom_1] if delta_g_row_u is not None else 0.0
        vs1 = float(var_scale_row[dom_1]) if var_scale_row is not None else 1.0
        for p0, m0, v0 in c0:
            for p1, m1, v1 in c1:
                pw = p0 * p1
                h_m = h_base + (m0 + dg0)[:, None] * d_mats[dom_0] + (m1 + dg1)[:, None] * d_mats[dom_1]
                nv_m = np.maximum(float(noise_mult) + v_base + v0 * vs0 + v1 * vs1, 1e-12)
                joint.append((pw, h_m, nv_m))
    else:
        for p0, m0, v0 in c0:
            h_m = h_base + (m0 + dg0)[:, None] * d_mats[dom_0]
            nv_m = np.maximum(float(noise_mult) + v_base + v0 * vs0, 1e-12)
            joint.append((p0, h_m, nv_m))

    joint.sort(key=lambda t: -t[0])
    joint = [x for i, x in enumerate(joint) if i == 0 or (i < cfg.max_joint_branches and x[0] >= cfg.min_branch_mass)]
    p_sum = sum(x[0] for x in joint)
    log_w = np.array([math.log(max(x[0] / p_sum, 1e-300)) for x in joint], dtype=REAL_DTYPE)
    h_stack = np.stack([x[1] for x in joint], axis=0)
    nv_stack = np.stack([x[2] for x in joint], axis=0)

    return RowMixtureBranches(
        log_w=log_w,
        h_u=h_stack,
        nv_u=nv_stack,
        h_mean_u=h_mean,
        nv_mean_u=nv_mean,
        mean_ray_u=m_ray,
        var_ray_u=v_ray,
    )


# ---------------------------------------------------------------------------
# 4. Row Decoders: Mean-Channel, Hard-Choice, and Exact Mixture-LLR MAP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowDecodeOutcome:
    """Detailed decoding outcome for one target or helper row."""

    info_bits: np.ndarray        # [64] uint8
    codeword: np.ndarray         # [128] uint8
    margin: float
    fit_ratio: float
    bp_converged: bool
    reliability: float
    win_branch: int
    n_candidates: int
    candidates_u64: np.ndarray
    bit_llrs_used: np.ndarray    # [128] float64
    beta_weights: np.ndarray     # [M] float64


def decode_row_mean_channel(
    y_row_u: np.ndarray,
    branches: RowMixtureBranches,
    *,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    list_size: str = "large",
) -> RowDecodeOutcome:
    """Baseline P12 decoder: linearly collapsed mean channel `(h_mean_u, nv_mean_u)`."""
    llrs = _lmmse_row_llrs_unit(y_row_u, branches.h_mean_u, branches.nv_mean_u)
    dec = decode_strong_sc(
        y_row_u,
        branches.h_mean_u,
        branches.nv_mean_u,
        list_size=list_size,
        bp_iters=bp_iters,
        turbo_pic=True,
        pic_scale=pic_scale,
        initial_llrs=llrs,
    )
    return RowDecodeOutcome(
        info_bits=dec.info_bits.copy(),
        codeword=dec.codeword.copy(),
        margin=0.0,
        fit_ratio=float(dec.metric) / float(N),
        bp_converged=bool(dec.bp_converged),
        reliability=0.0,
        win_branch=0,
        n_candidates=int(dec.n_candidates),
        candidates_u64=dec.candidates_u64,
        bit_llrs_used=llrs,
        beta_weights=np.array([1.0], dtype=REAL_DTYPE),
    )


def decode_row_hard_choice(
    y_row_u: np.ndarray,
    branches: RowMixtureBranches,
    *,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    list_size: str = "large",
) -> RowDecodeOutcome:
    """Hard MAP single-trajectory selection on the shared coherent branch bank (always index 0)."""
    m_star = int(np.argmax(branches.log_w))
    h_s = branches.h_u[m_star]
    nv_s = branches.nv_u[m_star]
    llrs = _lmmse_row_llrs_unit(y_row_u, h_s, nv_s)
    dec = decode_strong_sc(
        y_row_u,
        h_s,
        nv_s,
        list_size=list_size,
        bp_iters=bp_iters,
        turbo_pic=True,
        pic_scale=pic_scale,
        initial_llrs=llrs,
    )
    beta = np.zeros(len(branches.log_w), dtype=REAL_DTYPE)
    beta[m_star] = 1.0
    return RowDecodeOutcome(
        info_bits=dec.info_bits.copy(),
        codeword=dec.codeword.copy(),
        margin=0.0,
        fit_ratio=float(dec.metric) / float(N),
        bp_converged=bool(dec.bp_converged),
        reliability=0.0,
        win_branch=m_star,
        n_candidates=int(dec.n_candidates),
        candidates_u64=dec.candidates_u64,
        bit_llrs_used=llrs,
        beta_weights=beta,
    )


def decode_row_mixture_llr(
    y_row_u: np.ndarray,
    branches: RowMixtureBranches,
    *,
    cfg: MixtureDecodeConfig = MixtureDecodeConfig(),
    bp_iters: int = 20,
    pic_scale: float = 0.75,
    compute_margin: bool = False,
) -> RowDecodeOutcome:
    """Exact probability-domain Mixture Bit-LLR + Marginalized Mixture Codeword MAP decoding.

    Strictly follows `research/posterior_fusion` §5.2:
      1. Target-updated branch weights `beta_m propto w_m * p(y | H_m, v_m)^gamma_obs` combine
         per-branch LMMSE bit posteriors in probability space (`mix_branch_bit_llrs_exact`).
      2. Codeword union scoring uses PRE-target fold weights `w_m` inside
         `_score_offgrid_multimode_u64`, preventing any double-counting of `y`!
    """
    m_cnt = len(branches.log_w)
    if m_cnt == 1:
        llrs = _lmmse_row_llrs_unit(y_row_u, branches.h_u[0], branches.nv_u[0])
        if compute_margin:
            res = decode_with_blind_margin(
                y_row_u,
                branches.h_u[0],
                branches.nv_u[0],
                list_size="large",
                bp_iters=bp_iters,
                turbo_pic=True,
                pic_scale=pic_scale,
                initial_llrs=llrs,
            )
            return RowDecodeOutcome(
                info_bits=res.info_bits.copy(),
                codeword=res.codeword.copy(),
                margin=float(res.margin),
                fit_ratio=float(res.fit_ratio),
                bp_converged=bool(res.bp_converged),
                reliability=float(res.reliability),
                win_branch=0,
                n_candidates=int(res.n_candidates),
                candidates_u64=res.candidates_u64,
                bit_llrs_used=llrs,
                beta_weights=np.array([1.0], dtype=REAL_DTYPE),
            )
        dec = decode_strong_sc(
            y_row_u,
            branches.h_u[0],
            branches.nv_u[0],
            list_size="large",
            bp_iters=bp_iters,
            turbo_pic=True,
            pic_scale=pic_scale,
            initial_llrs=llrs,
        )
        return RowDecodeOutcome(
            info_bits=dec.info_bits.copy(),
            codeword=dec.codeword.copy(),
            margin=0.0,
            fit_ratio=float(dec.metric) / float(N),
            bp_converged=bool(dec.bp_converged),
            reliability=0.0,
            win_branch=0,
            n_candidates=int(dec.n_candidates),
            candidates_u64=dec.candidates_u64,
            bit_llrs_used=llrs,
            beta_weights=np.array([1.0], dtype=REAL_DTYPE),
        )

    # Step 1: Per-branch exact 32x32 Gaussian marginal evidence & LMMSE bit LLRs
    cov_ll = np.empty(m_cnt, dtype=REAL_DTYPE)
    branch_llrs = np.empty((m_cnt, 128), dtype=REAL_DTYPE)
    for m in range(m_cnt):
        cov_ll[m] = _branch_marginal_gaussian_loglik(y_row_u, branches.h_u[m], branches.nv_u[m])
        branch_llrs[m] = _lmmse_row_llrs_unit(y_row_u, branches.h_u[m], branches.nv_u[m])

    log_beta = branches.log_w + float(cfg.obs_cov_gamma) * (cov_ll - float(np.max(cov_ll)))
    log_beta = log_beta - float(logsumexp(log_beta))
    beta = np.exp(log_beta)

    # Step 2: Exact probability-domain mixture bit-LLR (NEVER linear LLR averaging!)
    mix_llrs = mix_branch_bit_llrs_exact(branch_llrs, log_beta)

    # Step 3: Fast BP directly on exact mixture bit-LLRs to harvest any converged codeword
    post_mix_arr, conv_mix_arr = _bp_sum_product(
        np.ascontiguousarray(mix_llrs[None, :]), _EDGE_BITS, _CHECK_OFFSETS, int(bp_iters)
    )
    stacks: list[np.ndarray] = []
    bp_conv_cws: list[np.ndarray] = []
    if bool(conv_mix_arr[0]):
        cw_mix_bp = (post_mix_arr[0] > 0.0).astype(np.uint8)
        if is_valid_codeword(cw_mix_bp):
            sd_m = pack_codewords(cw_mix_bp)
            stacks.append(sd_m[None, :])
            stacks.append(sd_m[None, :] ^ _GEN_U64)
            bp_conv_cws.append(cw_mix_bp)

    # Step 4: Select active branches ranked by target-updated posterior beta
    b_order = np.argsort(-beta)
    active_b: list[int] = [int(b_order[0])]
    for idx in b_order[1:]:
        if len(active_b) >= cfg.max_decode_branches:
            break
        if float(beta[idx]) >= cfg.runner_up_beta_thresh:
            active_b.append(int(idx))

    dec_results = []
    for rank, b_idx in enumerate(active_b):
        # Top branch runs Strong-L; if mixture posterior is ambiguous, feed exact mix_llrs
        # on rank 0 AND also branch-specific LLR if top branch isn't dominant.
        l_size = "large" if rank == 0 else cfg.runner_up_list_size
        init_l = mix_llrs if (rank == 0 and float(beta[b_idx]) < 0.90) else branch_llrs[b_idx]
        dec_m = decode_strong_sc(
            y_row_u,
            branches.h_u[b_idx],
            branches.nv_u[b_idx],
            list_size=l_size,
            bp_iters=bp_iters,
            turbo_pic=True,
            pic_scale=pic_scale,
            initial_llrs=init_l,
        )
        dec_results.append(dec_m)
        sd = pack_codewords(dec_m.codeword)
        stacks.append(sd[None, :])
        stacks.append(sd[None, :] ^ _GEN_U64)
        stacks.append(dec_m.candidates_u64)
        if dec_m.bp_converged:
            bp_conv_cws.append(dec_m.codeword)

        # If rank==0 used mix_llrs, also check rank-0 branch-pure BP-0 quickly (costs <0.2ms!)
        if rank == 0 and init_l is mix_llrs:
            p_b0, c_b0 = _bp_sum_product(
                np.ascontiguousarray(branch_llrs[b_idx][None, :]), _EDGE_BITS, _CHECK_OFFSETS, int(bp_iters)
            )
            if bool(c_b0[0]):
                cw_b0 = (p_b0[0] > 0.0).astype(np.uint8)
                if is_valid_codeword(cw_b0):
                    sd0 = pack_codewords(cw_b0)
                    stacks.append(sd0[None, :])
                    stacks.append(sd0[None, :] ^ _GEN_U64)
                    bp_conv_cws.append(cw_b0)

    # Step 5: Exact marginalized mixture log-likelihood scoring over ALL M branches using PRE-target log_w!
    all_u64 = np.vstack(stacks)
    dt_pair = np.dtype([("c0", np.uint64), ("c1", np.uint64)])
    uniq_u64 = np.ascontiguousarray(
        np.unique(np.ascontiguousarray(all_u64).view(dt_pair).ravel()).view(np.uint64).reshape(-1, 2)
    )

    b0, b1, best_ll, _, _, sec_ll, win_m, win_fit, n_val = _score_offgrid_multimode_u64(
        uniq_u64,
        _H_U64,
        np.ascontiguousarray(y_row_u.real, dtype=REAL_DTYPE),
        np.ascontiguousarray(y_row_u.imag, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.log_w, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.h_u.real, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.h_u.imag, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.nv_u, dtype=REAL_DTYPE),
        _NIBBLE_RE,
        _NIBBLE_IM,
    )
    cw = unpack_codewords(np.array([b0, b1], dtype=np.uint64))
    margin = float(max(0.0, float(best_ll) - float(sec_ll))) if n_val >= 2 else 0.0
    bp_conv = any(np.array_equal(c_bp, cw) for c_bp in bp_conv_cws)
    rel = reliability_weight_from_margin(margin, bp_conv, fit_ratio=float(win_fit)) if compute_margin else 0.0

    return RowDecodeOutcome(
        info_bits=cw[:64].copy(),
        codeword=cw,
        margin=margin,
        fit_ratio=float(win_fit),
        bp_converged=bp_conv,
        reliability=float(rel),
        win_branch=int(win_m),
        n_candidates=int(n_val),
        candidates_u64=uniq_u64,
        bit_llrs_used=mix_llrs,
        beta_weights=beta,
    )


def score_hard_choice_on_shared_union(
    y_row_u: np.ndarray,
    branches: RowMixtureBranches,
    uniq_u64: np.ndarray,
) -> np.ndarray:
    """Score the exact same codeword list `uniq_u64` under hard MAP branch only (for §5.3 audit)."""
    m_star = int(np.argmax(branches.log_w))
    b0, b1, *_ = _score_offgrid_multimode_u64(
        uniq_u64,
        _H_U64,
        np.ascontiguousarray(y_row_u.real, dtype=REAL_DTYPE),
        np.ascontiguousarray(y_row_u.imag, dtype=REAL_DTYPE),
        np.zeros(1, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.h_u[m_star : m_star + 1].real, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.h_u[m_star : m_star + 1].imag, dtype=REAL_DTYPE),
        np.ascontiguousarray(branches.nv_u[m_star : m_star + 1], dtype=REAL_DTYPE),
        _NIBBLE_RE,
        _NIBBLE_IM,
    )
    cw = unpack_codewords(np.array([b0, b1], dtype=np.uint64))
    return cw[:64].copy()


# ---------------------------------------------------------------------------
# 5. Consistent 3-Fold Off-Grid Core (`655de6a` + Optional Content Addressing)
# ---------------------------------------------------------------------------


def _weighted_cov32_loglik_per_row(
    y_aux_u: np.ndarray,     # [R_aux, N]
    mean_ray_u: np.ndarray,  # [K, R_aux, N]
    var_ray_u: np.ndarray,   # [K, R_aux, N]
    d_mats: np.ndarray,      # [K, N, N]
    row_weights: np.ndarray, # [R_aux]
) -> float:
    """Weighted full 32x32 Hermitian Cholesky continuous-delay AUX quasi-log-likelihood."""
    r_cnt = y_aux_u.shape[0]
    k_cnt = mean_ray_u.shape[0]
    const_n = -float(N) * math.log(math.pi)
    total = 0.0
    for ri in range(r_cnt):
        w_r = float(row_weights[ri])
        if w_r <= 0.0:
            continue
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
        total += w_r * (const_n - logdet - quad)
    return float(total)


def run_consistent_folds(
    obs: OffGridObservation,
    *,
    k_hat: int = K_HAT,
    use_aux_cov: bool = True,
    use_wavefront: bool = True,
    use_piecewise_midframe: bool = True,
    use_delay_rediscovery: bool = True,
    use_wave1_multimode: bool = True,
    content_cfg: Optional[ContentAddressingConfig] = None,
    wave_mix_cfg: Optional[MixtureDecodeConfig] = None,
    bp_iters: int = 20,
    pic_scale: float = 0.75,
) -> ConsistentFoldsResult:
    """Run the `655de6a`-consistent 3-fold off-grid inference core, returning full fold snapshots.

    When `content_cfg is None` and `wave_mix_cfg is None`, executes the EXACT algorithm of
    `p12/offgrid_receiver.py:run_p12_offgrid_physics` (verified to `0.0` float diff!).
    """
    t0 = time.perf_counter()
    n0 = _validate_n0(obs.n0)
    s_mag = math.sqrt(n0)
    s_var = n0
    probe_row = _strict_int(obs.probe_row, "obs.probe_row")
    training_rows = tuple(_strict_int(r, "obs.training_rows") for r in obs.training_rows)
    if (
        not training_rows
        or probe_row < 0
        or probe_row >= T
        or any(r < 0 or r >= T for r in training_rows)
        or probe_row in training_rows
    ):
        raise ValueError(f"Invalid training_rows={training_rows} with probe_row={probe_row}")
    r_anchor = training_rows[0]

    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    if y_b_safe.shape != (T, N):
        raise ValueError(f"Expected obs.y_b shape ({T}, {N}), got {y_b_safe.shape}")
    y_b_safe[probe_row] = 0.0 + 0.0j
    if not np.all(np.isfinite(y_b_safe)):
        raise ValueError("obs.y_b non-probe rows contain non-finite values")
    pilot_b = np.asarray(obs.pilot_tx_b, dtype=COMPLEX_DTYPE)
    if pilot_b.shape != (len(training_rows), N) or not np.all(np.isfinite(pilot_b)):
        raise ValueError(f"obs.pilot_tx_b must be finite with shape ({len(training_rows)}, {N}), got {pilot_b.shape}")
    y_b_u = y_b_safe / s_mag

    # Frame-A blind discovery (strictly from y_a, locked comb (0,9,10,19))
    atoms_a = discover_frame_a_atoms(obs, k_hat=k_hat)

    # Step 0: B-only delay rediscovery (identical to p12/offgrid_receiver.py)
    delay_reset_triggered = False
    delay_ll_gain = 0.0
    p_tot_b_est = max(float(np.mean(np.abs(y_b_u[r_anchor]) ** 2)) - 1.0, 0.25)
    prior_var_delay_test = max(p_tot_b_est / float(k_hat), 0.15)

    if use_delay_rediscovery:
        tau_b, p_atom_b_u, p_tot_b_u = discover_b_only_delays(y_b_u[r_anchor], obs.pilot_tx_b[0], k_hat=k_hat)
        ll_a_anc = _anchor_delay_marginal_loglik(y_b_u[r_anchor], obs.pilot_tx_b[0], atoms_a.tau_hat, prior_var_delay_test)
        ll_b_anc = _anchor_delay_marginal_loglik(y_b_u[r_anchor], obs.pilot_tx_b[0], tau_b, prior_var_delay_test)
        delay_ll_gain = float(ll_b_anc - ll_a_anc)
        if delay_ll_gain > DELAY_RESET_OCCAM_NATS:
            delay_reset_triggered = True

    t_b_mid = float(_TIME_B_FULL.mean())
    if delay_reset_triggered:
        used_delays = tau_b
        d_mats = np.stack([fractional_delay_matrix(float(d), n=N) for d in used_delays], axis=0)
        u_probe = np.einsum("knm,m->kn", d_mats, SC_PILOTS[probe_row])
        k = len(used_delays)
        n_inh = 0
        n_clr = len(_NU_CLR_EXT_BANK)
        n_mid = 0
        n_tot_h = n_clr
        t_anchor = float(_TIME_B_FULL[r_anchor].mean())
        dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)
        rot_clr = np.exp(2j * np.pi * (_NU_CLR_EXT_BANK[:, None] * dt_b_anc[None, :])).reshape(n_clr, T, N)
        rot_bank = np.broadcast_to(rot_clr[None, :, :, :], (k, n_clr, T, N)).copy()
        mode_family = np.ones((k, n_tot_h), dtype=np.int64)
        mode_sign_pos = np.broadcast_to((_NU_CLR_EXT_BANK > 0.0)[None, :], (k, n_tot_h)).copy()
        mode_nu_eff = np.broadcast_to(_NU_CLR_EXT_BANK[None, :], (k, n_tot_h)).copy()

        prior_mu_u = np.zeros((k, n_tot_h), dtype=COMPLEX_DTYPE)
        p_clear_u = np.maximum(p_atom_b_u, 0.12 * p_tot_b_u / float(k))
        prior_var_u = np.broadcast_to(p_clear_u[:, None], (k, n_tot_h)).copy()
        prior_logw = np.full((k, n_tot_h), -math.log(float(n_clr)), dtype=REAL_DTYPE)
        ray_power_order = np.argsort(-p_atom_b_u)
        p_atom_u = p_clear_u.copy()
    else:
        used_delays = atoms_a.tau_hat.copy()
        d_mats = np.stack([fractional_delay_matrix(float(d), n=N) for d in used_delays], axis=0)
        u_probe = np.einsum("knm,m->kn", d_mats, SC_PILOTS[probe_row])
        k = len(used_delays)

        t_a_mean = float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())
        t_anchor = float(_TIME_B_FULL[r_anchor].mean())
        dt_anc_a = t_anchor - t_a_mean
        dt_b_anc = (_TIME_B_FULL - t_anchor).reshape(-1)

        n_inh = atoms_a.nu_bank_a.shape[1]
        n_clr = len(_NU_CLR_EXT_BANK)
        n_mid = (len(MID_SHIFT_ONSET_GRID) * len(MID_SHIFT_POST_MAGS)) if use_piecewise_midframe else 0
        n_tot_h = n_inh + n_clr + n_mid

        rot_bank = np.empty((k, n_tot_h, T, N), dtype=COMPLEX_DTYPE)
        mode_family = np.empty((k, n_tot_h), dtype=np.int64)
        mode_sign_pos = np.empty((k, n_tot_h), dtype=bool)
        mode_nu_eff = np.empty((k, n_tot_h), dtype=REAL_DTYPE)

        p_atom_u = np.maximum(atoms_a.p_atom_a / s_var, 1e-12)
        p_tot_u = max(float(atoms_a.p_tot_a) / s_var, 0.25)
        if content_cfg is not None and content_cfg.b_power_floor_frac > 0.0:
            p_clear_u = np.maximum(p_atom_u, float(content_cfg.b_power_floor_frac) * max(p_tot_u, p_tot_b_est) / float(k))
        else:
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
        p_inh_tot, p_clr_tot, p_mid_tot = (
            FAMILY_PRIOR_WITH_MIDFRAME if use_piecewise_midframe else FAMILY_PRIOR_WITHOUT_MIDFRAME
        )

        for j in range(k):
            ph_i = 2.0 * np.pi * (
                nu_inh_at_anc[j, :, None] * dt_b_anc[None, :]
                + 0.5 * atoms_a.chirp_bank_a[j, :, None] * (dt_b_anc[None, :] ** 2)
            )
            rot_bank[j, :n_inh] = np.exp(1j * ph_i).reshape(n_inh, T, N)
            mode_family[j, :n_inh] = 0
            mode_sign_pos[j, :n_inh] = nu_inh_at_anc[j] > 0.0
            mode_nu_eff[j, :n_inh] = atoms_a.nu_bank_a[j] + atoms_a.chirp_bank_a[j] * (t_b_mid - t_a_mean)

            rot_bank[j, n_inh : n_inh + n_clr] = rot_clr
            mode_family[j, n_inh : n_inh + n_clr] = 1
            mode_sign_pos[j, n_inh : n_inh + n_clr] = _NU_CLR_EXT_BANK > 0.0
            mode_nu_eff[j, n_inh : n_inh + n_clr] = _NU_CLR_EXT_BANK

            prior_mu_u[j, :n_inh] = mu_inh_anc_u[j]
            prior_var_u[j, :n_inh] = v_inh_anc_u[j]
            prior_logw[j, :n_inh] = math.log(p_inh_tot) + atoms_a.log_w_bank_a[j]

            prior_var_u[j, n_inh : n_inh + n_clr] = p_clear_u[j]
            clr_prior = np.array([1.0] * 8 + [0.35, 0.35] + [1.0] * 8 + [0.35, 0.35], dtype=REAL_DTYPE)
            clr_prior /= float(clr_prior.sum())
            prior_logw[j, n_inh : n_inh + n_clr] = math.log(p_clr_tot) + np.log(clr_prior)

            if use_piecewise_midframe:
                h_best_a = int(np.argmax(atoms_a.log_w_bank_a[j]))
                nu_a0 = float(atoms_a.nu_bank_a[j, h_best_a])
                ch_a0 = float(atoms_a.chirp_bank_a[j, h_best_a])
                mu_a0 = complex(atoms_a.mu_bank_a[j, h_best_a]) / s_mag
                v_a0 = float(v_inh_anc_u[j, h_best_a])
                sgn_a = 1.0 if (nu_a0 + ch_a0 * (30.0 - t_a_mean)) >= 0.0 else -1.0

                dt_b_from_a = _TIME_B_FULL - t_a_mean
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
                        if t_anchor < t_s:
                            phi_anc = 2.0 * np.pi * (
                                nu_a0 * (t_anchor - t_a_mean) + 0.5 * ch_a0 * ((t_anchor - t_a_mean) ** 2)
                            )
                        else:
                            phi_anc = phi_at_s + 2.0 * np.pi * nu_post * (t_anchor - t_s)
                        rot_bank[j, m_off + idx_m] = np.exp(1j * (phi_pw - phi_anc))
                        prior_mu_u[j, m_off + idx_m] = mu_a0 * complex(math.cos(phi_anc), math.sin(phi_anc))
                        prior_var_u[j, m_off + idx_m] = max(v_a0 + 0.15 * p_atom_u[j], 0.08 * p_clear_u[j])
                        prior_logw[j, m_off + idx_m] = math.log(p_mid_tot / float(n_mid))
                        mode_family[j, m_off + idx_m] = 2
                        mode_sign_pos[j, m_off + idx_m] = nu_post > 0.0
                        mode_nu_eff[j, m_off + idx_m] = 0.5 * (nu_a0 + nu_post)
                        idx_m += 1

        ray_power_order = np.argsort(-atoms_a.p_atom_a)

    # Strictly read-only base priors (enforces 655de6a invariant!)
    for arr in (prior_mu_u, prior_var_u, prior_logw):
        arr.setflags(write=False)

    def _fit_jacobi_single_count(
        rows_known: Sequence[int],
        words_known: np.ndarray,
        noises_u: np.ndarray,
        y_inf_u: np.ndarray,
        eff_prior_logw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(set(rows_known)) != len(rows_known):
            raise ValueError("A refit may condition on each known row only once")
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
        n_jac = int(content_cfg.jacobi_iters) if content_cfg is not None else 8
        damp = float(content_cfg.jacobi_damping) if content_cfg is not None else 0.65
        for _ in range(n_jac):
            tot_y = per_ray_y.sum(axis=0)
            res = y_k[None, :] - (tot_y[None, :] - per_ray_y)
            matched = np.einsum("khm,km->kh", d_conj_w, res)
            nat_m = init_prec_m + matched
            new_m = post_v * nat_m
            ev = ev_base + post_v * (np.abs(nat_m) ** 2)
            new_lw = ev - logsumexp(ev, axis=1, keepdims=True)
            new_py = np.einsum("kh,khm->km", np.exp(new_lw) * new_m, design)
            per_ray_y = damp * new_py + (1.0 - damp) * per_ray_y
            mean_h = new_m
            logw_cur = new_lw

        w_post = np.exp(logw_cur)
        mean_all = np.einsum("kh,khtn->ktn", w_post * mean_h, rot_bank)
        sec_all = np.sum(w_post * (np.abs(mean_h) ** 2 + post_v), axis=1)
        var_all = np.maximum(sec_all[:, None, None] - np.abs(mean_all) ** 2, 1e-12)
        return mean_all, var_all, logw_cur, mean_h, post_v

    def _compute_aux_content_weights(
        aux_rows: tuple[int, ...],
        y_inf_u: np.ndarray,
        mean_anc_u: np.ndarray,
        var_anc_u: np.ndarray,
    ) -> np.ndarray:
        """Thresholded C*-invariant content reliability weights over the 5 AUX rows.

        Clean AUX rows keep weight 1.0 (avoiding noise amplification); burst/kurtosis
        outlier rows are smoothly downweighted before computing `delta_aux`.
        """
        r_cnt = len(aux_rows)
        if content_cfg is None or (not content_cfg.use_aux_content_weights):
            return np.ones(r_cnt, dtype=REAL_DTYPE)
        w_out = np.ones(r_cnt, dtype=REAL_DTYPE)
        for idx_a, r_a in enumerate(aux_rows):
            y_a = y_inf_u[r_a]
            obs_e = float(np.mean(y_a.real ** 2 + y_a.imag ** 2))
            exp_e = 1.0 + float(np.sum(np.mean(mean_anc_u[:, r_a, :].real ** 2 + mean_anc_u[:, r_a, :].imag ** 2 + var_anc_u[:, r_a, :], axis=-1)))
            e_anom = math.log(max(obs_e, 1e-12) / max(exp_e, 1e-12))
            e_excess = max(0.0, e_anom - float(content_cfg.aux_energy_thresh))
            m4 = float(np.mean((y_a.real ** 2 + y_a.imag ** 2) ** 2))
            kurt = max(0.0, m4 / max(obs_e * obs_e, 1e-12) - 2.0)
            k_excess = max(0.0, math.log1p(kurt) - float(content_cfg.aux_kurt_thresh))
            w_out[idx_a] = math.exp(-float(content_cfg.aux_energy_penalty) * e_excess - float(content_cfg.aux_kurt_penalty) * k_excess)
        return w_out

    def _compute_incremental_aux_cov32_delta(
        aux_rows: tuple[int, ...],
        y_inf_u: np.ndarray,
        mean_anc_u: np.ndarray,
        var_anc_u: np.ndarray,
        lw_anc: np.ndarray,
        mh_anc: np.ndarray,
        pv_anc: np.ndarray,
        aux_w: np.ndarray,
    ) -> np.ndarray:
        y_aux_u = y_inf_u[list(aux_rows)]
        delta_aux = np.zeros((k, n_tot_h), dtype=REAL_DTYPE)
        cur_m_aux = mean_anc_u[:, list(aux_rows), :].copy()
        cur_v_aux = var_anc_u[:, list(aux_rows), :].copy()
        aux_temp = float(content_cfg.aux_cov_temp) if content_cfg is not None else AUX_COV_TEMP
        use_weighted = content_cfg is not None and content_cfg.use_aux_content_weights
        use_full_bank = content_cfg is not None and content_cfg.full_aux_bank

        if content_cfg is not None and content_cfg.full_aux_bank:
            anc_e = np.mean(
                mean_anc_u[:, r_anchor, :].real ** 2 + mean_anc_u[:, r_anchor, :].imag ** 2 + var_anc_u[:, r_anchor, :],
                axis=-1,
            )
            aux_rays = np.argsort(-anc_e)[: min(int(content_cfg.aux_max_rays), k)]
        else:
            aux_rays = ray_power_order[: min(3, k)]

        for j in aux_rays:
            if use_full_bank:
                # Score ALL clear modes + ALL piecewise modes + top-6 primary & ALL grating-lobe inherit modes
                cands_list: list[np.ndarray] = [np.arange(n_inh, n_inh + n_clr, dtype=np.int64)]
                if n_inh > 0:
                    cands_list.append(np.argsort(-lw_anc[j, : min(15, n_inh)])[:6])
                    if n_inh > 15:
                        cands_list.append(15 + np.arange(n_inh - 15, dtype=np.int64))
                if n_mid > 0:
                    cands_list.append(n_inh + n_clr + np.arange(n_mid, dtype=np.int64))
                cand_h = np.unique(np.concatenate(cands_list))
            else:
                cands_list = [np.argsort(-lw_anc[j])[:10]]
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
                if use_weighted:
                    ll_vals[ci] = _weighted_cov32_loglik_per_row(y_aux_u, m_tmp, v_tmp, d_mats, aux_w)
                else:
                    ll_vals[ci] = offgrid_cov32_loglik(y_aux_u, m_tmp, v_tmp, d_mats)

            max_ll = float(np.max(ll_vals))
            min_rel = float(np.min(ll_vals - max_ll))
            delta_j = np.full(n_tot_h, aux_temp * (min_rel - 2.0), dtype=REAL_DTYPE)
            delta_j[cand_h] = aux_temp * (ll_vals - max_ll)
            delta_aux[j] = delta_j

            lw_tmp = lw_anc[j] + delta_j
            lw_tmp -= float(logsumexp(lw_tmp))
            w_tmp = np.exp(lw_tmp)
            cur_m_aux[j] = np.einsum("h,hrn->rn", w_tmp * mh_anc[j], rot_bank[j][:, list(aux_rows), :])
            sec_tmp = float(np.sum(w_tmp * (np.abs(mh_anc[j]) ** 2 + pv_anc[j])))
            cur_v_aux[j] = np.maximum(sec_tmp - np.abs(cur_m_aux[j]) ** 2, 1e-12)

        return delta_aux

    def _decode_wave1_legacy_p12(
        r_h: int,
        y_row_u: np.ndarray,
        mean_u_all: np.ndarray,
        var_u_all: np.ndarray,
        lw_cur: np.ndarray,
        mh_cur: np.ndarray,
        pv_cur: np.ndarray,
    ):
        """Exact `p12/offgrid_receiver.py:_decode_offgrid_wave1_multimode` for bit-exact baseline parity."""
        dom_k = int(ray_power_order[0])
        w_dom = np.exp(lw_cur[dom_k])
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
        active = active[:2]

        if (not use_wave1_multimode) or len(active) < 2:
            h_single = np.zeros((N, N), dtype=COMPLEX_DTYPE)
            for j in range(k):
                h_single += mean_u_all[j, r_h, :, None] * d_mats[j]
            nv_single = 1.0 + var_u_all[:, r_h, :].sum(axis=0)
            llrs_s = _lmmse_row_llrs_unit(y_row_u, h_single, nv_single)
            res_s = decode_with_blind_margin(
                y_row_u, h_single, nv_single, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_s
            )
            return res_s, h_single, nv_single

        h_modes: list[np.ndarray] = []
        nv_modes: list[np.ndarray] = []
        log_p_list: list[float] = []
        dec_results = []
        p_norm = sum(p for p, _ in active)

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
            llrs_m = _lmmse_row_llrs_unit(y_row_u, h_m, nv_m)
            dec_m = decode_strong_sc(
                y_row_u, h_m, nv_m, list_size="large", bp_iters=bp_iters, turbo_pic=True, pic_scale=pic_scale, initial_llrs=llrs_m
            )
            h_modes.append(h_m)
            nv_modes.append(nv_m)
            log_p_list.append(math.log(max(p_m / p_norm, 1e-300)))
            dec_results.append(dec_m)

        stacks = []
        for dec_m in dec_results:
            sd = pack_codewords(dec_m.codeword)
            stacks.append(sd[None, :])
            stacks.append(sd[None, :] ^ _GEN_U64)
            stacks.append(dec_m.candidates_u64)
        all_u64 = np.vstack(stacks)
        dt_pair = np.dtype([("c0", np.uint64), ("c1", np.uint64)])
        uniq_u64 = np.ascontiguousarray(
            np.unique(np.ascontiguousarray(all_u64).view(dt_pair).ravel()).view(np.uint64).reshape(-1, 2)
        )

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

    # --- Step 1: Strictly Target-Excluded Roving Bilateral 3-Fold Execution ---
    t_folds, a_folds = make_roving_bilateral_3fold(r_anchor, probe_row=probe_row)
    all_data_rows = tuple(r for r in range(T) if r not in (r_anchor, probe_row))

    snapshots: list[FoldPosteriorSnapshot] = []
    ledger_records: list[dict[str, Any]] = []
    tot_acc = 0
    aux_reads = 0
    helper_key_reads = 0
    helper_value_reads = 0

    for fold_idx, (t_rows, a_rows) in enumerate(zip(t_folds, a_folds)):
        # Structural firewall: mask all owned target rows and probe_row to NaN!
        y_inf_u = y_b_u.copy()
        y_inf_u[list(t_rows)] = complex(np.nan, np.nan)
        y_inf_u[probe_row] = complex(np.nan, np.nan)

        m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc = _fit_jacobi_single_count(
            (r_anchor,), obs.pilot_tx_b[:1], np.array([1.0], dtype=REAL_DTYPE), y_inf_u, prior_logw
        )

        aux_w = _compute_aux_content_weights(a_rows, y_inf_u, m_anc_u, v_anc_u)
        if use_aux_cov:
            aux_reads += len(a_rows)
            delta_aux = _compute_incremental_aux_cov32_delta(
                a_rows, y_inf_u, m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc, aux_w
            )
            eff_prior_logw = prior_logw + delta_aux
            eff_prior_logw -= logsumexp(eff_prior_logw, axis=1, keepdims=True)
            eff_prior_logw.setflags(write=False)
            m_u, v_u, lw_cur, mh_cur, pv_cur = _fit_jacobi_single_count(
                (r_anchor,), obs.pilot_tx_b[:1], np.array([1.0], dtype=REAL_DTYPE), y_inf_u, eff_prior_logw
            )
        else:
            eff_prior_logw = prior_logw
            m_u, v_u, lw_cur, mh_cur, pv_cur = m_anc_u, v_anc_u, lw_anc, mh_anc, pv_anc

        forbidden = set(t_rows) | set(a_rows) | {r_anchor, probe_row}
        helpers = tuple(sorted([r for r in all_data_rows if r not in forbidden], key=lambda x: abs(x - r_anchor)))

        known_rows_list: list[int] = [r_anchor]
        known_words_list: list[np.ndarray] = [obs.pilot_tx_b[0].copy()]
        known_noises_list: list[float] = [1.0]

        helper_alpha_row = np.zeros(T, dtype=REAL_DTYPE)
        helper_margin_row = np.zeros(T, dtype=REAL_DTYPE)
        helper_raw_score: dict[int, float] = {}
        helper_alpha_raw: dict[int, float] = {}

        if use_wavefront:
            wave1 = helpers[:2]
            wave2 = helpers[2:]

            for w_idx, wave in enumerate((wave1, wave2)):
                if not wave:
                    continue
                added = False
                for r_h in wave:
                    helper_key_reads += 1
                    helper_value_reads += 1

                    # Burst-aware noise multiplier on helper row r_h (only when content_cfg active)
                    nm_h = (
                        _compute_burst_mult_row(y_inf_u[r_h], m_u[:, r_h, :], v_u[:, r_h, :])
                        if (content_cfg is not None and content_cfg.use_helper_burst_guard)
                        else 1.0
                    )

                    use_mix_this_wave = (
                        wave_mix_cfg is not None
                        and ((w_idx == 0 and wave_mix_cfg.use_mixture_in_wave1) or (w_idx == 1 and wave_mix_cfg.use_mixture_in_wave2))
                    )

                    if use_mix_this_wave:
                        br_h = build_coherent_row_mixture(
                            r_h, lw_cur, mh_cur, pv_cur, rot_bank, d_mats, cfg=wave_mix_cfg, noise_mult=nm_h
                        )
                        out_h = decode_row_mixture_llr(
                            y_inf_u[r_h], br_h, cfg=wave_mix_cfg, bp_iters=bp_iters, pic_scale=pic_scale, compute_margin=True
                        )
                        eval_h_u = br_h.h_u[out_h.win_branch]
                        eval_nv_u = br_h.nv_u[out_h.win_branch]
                        rel_h = out_h.reliability
                        margin_h = out_h.margin
                        cw_h = out_h.codeword
                        info_h = out_h.info_bits
                    elif w_idx == 0:
                        res_h, eval_h_u, eval_nv_u = _decode_wave1_legacy_p12(
                            r_h, y_inf_u[r_h], m_u, v_u, lw_cur, mh_cur, pv_cur
                        )
                        rel_h = res_h.reliability
                        margin_h = res_h.margin
                        cw_h = res_h.codeword
                        info_h = res_h.info_bits
                    else:
                        eval_h_u = np.zeros((N, N), dtype=COMPLEX_DTYPE)
                        for j in range(k):
                            eval_h_u += m_u[j, r_h, :, None] * d_mats[j]
                        eval_nv_u = float(nm_h) + v_u[:, r_h, :].sum(axis=0)
                        llrs_l = _lmmse_row_llrs_unit(y_inf_u[r_h], eval_h_u, eval_nv_u)
                        res_h = decode_with_blind_margin(
                            y_inf_u[r_h],
                            eval_h_u,
                            eval_nv_u,
                            list_size="large",
                            bp_iters=bp_iters,
                            turbo_pic=True,
                            pic_scale=pic_scale,
                            initial_llrs=llrs_l,
                        )
                        rel_h = res_h.reliability
                        margin_h = res_h.margin
                        cw_h = res_h.codeword
                        info_h = res_h.info_bits

                    helper_margin_row[r_h] = float(margin_h)
                    if rel_h >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                        x_hat = qam(cw_h).astype(COMPLEX_DTYPE)
                        y_hat = eval_h_u @ x_hat
                        diff = y_inf_u[r_h] - y_hat
                        resid2 = diff.real ** 2 + diff.imag ** 2
                        u_stat = max(float(np.sum(resid2) / np.sum(eval_nv_u)), float(np.mean(resid2 / eval_nv_u)))
                        w_res = redescending_residual_weight(u_stat)
                        alpha_eff = float(min(DEFAULT_ALPHA_MAX, rel_h * w_res))

                        if alpha_eff >= DEFAULT_MIN_ASSIMILATE_WEIGHT:
                            n_eff_u = (1.0 / alpha_eff + DEFAULT_VIRTUAL_VAR_FLOOR) * float(nm_h)
                            known_rows_list.append(r_h)
                            known_words_list.append(x_hat)
                            known_noises_list.append(n_eff_u)
                            helper_alpha_row[r_h] = float(alpha_eff)
                            tot_acc += 1
                            ledger_records.append({
                                "fold": int(fold_idx),
                                "wave": int(w_idx),
                                "row": int(r_h),
                                "injected_info": info_h.copy(),
                                "alpha": float(alpha_eff),
                                "margin": float(margin_h),
                            })
                            added = True

                if added:
                    m_u, v_u, lw_cur, mh_cur, pv_cur = _fit_jacobi_single_count(
                        known_rows_list,
                        np.stack(known_words_list, axis=0),
                        np.asarray(known_noises_list, dtype=REAL_DTYPE),
                        y_inf_u,
                        eff_prior_logw,
                    )

        # Optional target-position-only content query reweighting (reads ONLY public row index r_t, NEVER y_b[r_t]!)
        per_t_lw: Optional[dict[int, np.ndarray]] = None
        per_t_mh: Optional[dict[int, np.ndarray]] = None
        per_t_pv: Optional[dict[int, np.ndarray]] = None
        per_t_m_u: Optional[dict[int, np.ndarray]] = None
        per_t_v_u: Optional[dict[int, np.ndarray]] = None
        if (
            content_cfg is not None
            and content_cfg.use_target_pos_query
            and len(known_rows_list) >= 3
        ):
            per_t_lw = {}
            per_t_mh = {}
            per_t_pv = {}
            per_t_m_u = {}
            per_t_v_u = {}
            words_arr = np.stack(known_words_list, axis=0)
            base_noises = np.asarray(known_noises_list, dtype=REAL_DTYPE)
            t_q = max(float(content_cfg.target_pos_temp), 1e-4)
            for r_t in t_rows:
                # Target query Q(r_t) depends ONLY on public row index r_t and anchor/helper indices!
                # Closer helpers to r_t (in normalized time) get gently higher precision without dropping far ones.
                dists = np.array([abs(int(r_k) - int(r_t)) / float(T - 1) for r_k in known_rows_list], dtype=REAL_DTYPE)
                # Anchor (index 0) keeps exact unit noise; helpers modulated smoothly via calibrated softmax
                h_d = dists[1:]
                log_q = -h_d / t_q
                w_q = float(len(h_d)) * np.exp(log_q - float(logsumexp(log_q)))
                mult_q = np.clip(0.55 + 0.45 * w_q, 0.40, 1.80)
                q_noises = base_noises.copy()
                q_noises[1:] = base_noises[1:] / mult_q
                mt, vt, lwt, mht, pvt = _fit_jacobi_single_count(
                    known_rows_list, words_arr, q_noises, y_inf_u, eff_prior_logw
                )
                per_t_lw[r_t] = lwt
                per_t_mh[r_t] = mht
                per_t_pv[r_t] = pvt
                per_t_m_u[r_t] = mt[:, r_t, :].copy()
                per_t_v_u[r_t] = vt[:, r_t, :].copy()

        # Compute strictly target/probe-excluded fold-only observable metadata for Stage 5 neural features
        row_resid_row = np.zeros(T, dtype=REAL_DTYPE)
        row_energy_anom = np.zeros(T, dtype=REAL_DTYPE)
        row_kurt_anom = np.zeros(T, dtype=REAL_DTYPE)
        obs_matched_resid = np.zeros((k, T), dtype=COMPLEX_DTYPE)

        obs_rows_fold = set(range(T)) - set(t_rows) - {probe_row}
        known_map = {int(rk): idx_k for idx_k, rk in enumerate(known_rows_list)}
        for r_o in obs_rows_fold:
            y_o = y_inf_u[r_o]
            obs_e = float(np.mean(y_o.real ** 2 + y_o.imag ** 2))
            exp_e = 1.0 + float(np.sum(np.mean(m_u[:, r_o, :].real ** 2 + m_u[:, r_o, :].imag ** 2 + v_u[:, r_o, :], axis=-1)))
            row_energy_anom[r_o] = float(math.log(max(obs_e, 1e-12) / max(exp_e, 1e-12)))
            m4 = float(np.mean((y_o.real ** 2 + y_o.imag ** 2) ** 2))
            row_kurt_anom[r_o] = float(math.log1p(max(0.0, m4 / max(obs_e * obs_e, 1e-12) - 2.0)))
            if r_o in known_map:
                xk = known_words_list[known_map[r_o]]
                u_k = np.einsum("knm,m->kn", d_mats, xk)  # [K, N]
                y_pred = np.sum(m_u[:, r_o, :] * u_k, axis=0)
                res_o = y_o - y_pred
                nv_o = 1.0 + np.sum(v_u[:, r_o, :], axis=0)
                row_resid_row[r_o] = float(math.log1p(max(0.0, float(np.mean((res_o.real ** 2 + res_o.imag ** 2) / nv_o)))))
                # Strictly U(1)-equivariant per-ray matched-filter residual phasor on known rows ONLY!
                u_pow = np.maximum(np.sum(u_k.real ** 2 + u_k.imag ** 2, axis=-1), 1.0)
                obs_matched_resid[:, r_o] = np.sum(u_k.conj() * res_o[None, :], axis=-1) / u_pow

        snapshots.append(
            FoldPosteriorSnapshot(
                fold_idx=int(fold_idx),
                t_rows=tuple(t_rows),
                a_rows=tuple(a_rows),
                helper_rows=helpers,
                r_anchor=int(r_anchor),
                probe_row=int(probe_row),
                delay_reset_triggered=bool(delay_reset_triggered),
                used_delays=used_delays.copy(),
                d_mats=d_mats.copy(),
                u_probe=u_probe.copy(),
                rot_bank=rot_bank,
                mode_family=mode_family,
                mode_sign_pos=mode_sign_pos,
                mode_nu_eff=mode_nu_eff,
                ray_power_order=ray_power_order.copy(),
                p_atom_u=p_atom_u.copy(),
                prior_mu_u=prior_mu_u,
                prior_var_u=prior_var_u,
                eff_prior_logw=eff_prior_logw,
                m_u=m_u,
                v_u=v_u,
                lw_cur=lw_cur,
                mh_cur=mh_cur,
                pv_cur=pv_cur,
                accepted_rows=tuple(known_rows_list[1:]),
                known_words_u=np.stack(known_words_list, axis=0),
                known_noises_u=np.asarray(known_noises_list, dtype=REAL_DTYPE),
                helper_alpha_row=helper_alpha_row,
                helper_margin_row=helper_margin_row,
                row_resid_row=row_resid_row,
                row_energy_anom=row_energy_anom,
                row_kurt_anom=row_kurt_anom,
                obs_matched_resid=obs_matched_resid,
                aux_weights=aux_w,
                per_target_lw=per_t_lw,
                per_target_mh=per_t_mh,
                per_target_pv=per_t_pv,
                per_target_m_u=per_t_m_u,
                per_target_v_u=per_t_v_u,
            )
        )

    return ConsistentFoldsResult(
        folds=tuple(snapshots),
        n0=float(n0),
        r_anchor=int(r_anchor),
        probe_row=int(probe_row),
        delay_reset_triggered=bool(delay_reset_triggered),
        used_delays=used_delays.copy(),
        delay_ll_gain_nats=float(delay_ll_gain),
        helpers_accepted=int(tot_acc),
        ledger=tuple(ledger_records),
        aux_reads=int(aux_reads),
        helper_key_reads=int(helper_key_reads),
        helper_value_reads=int(helper_value_reads),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# 6. End-to-End Evaluation & Scoring Helpers
# ---------------------------------------------------------------------------


def _apply_neural_override_to_fold(
    snap: FoldPosteriorSnapshot,
    row: int,
    delta_fam_logits_row: Optional[np.ndarray] = None,  # [K, 4] float64
    delta_g_row_u: Optional[np.ndarray] = None,         # [K, N] complex128
    delta_log_var_row: Optional[np.ndarray] = None,     # [K] float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply optional Stage-5 neural posterior & residual refinement at row `row`."""
    lw = snap.per_target_lw[row] if (snap.per_target_lw is not None and row in snap.per_target_lw) else snap.lw_cur
    mh = snap.per_target_mh[row] if (snap.per_target_mh is not None and row in snap.per_target_mh) else snap.mh_cur
    pv = snap.per_target_pv[row] if (snap.per_target_pv is not None and row in snap.per_target_pv) else snap.pv_cur

    if delta_fam_logits_row is not None:
        # 4 physical family buckets per ray: 0=inh, 1=clr+, 2=clr-, 3=pw
        k_cnt, h_cnt = lw.shape
        lw_new = lw.copy()
        for j in range(k_cnt):
            f_j = snap.mode_family[j]
            s_j = snap.mode_sign_pos[j]
            b_idx = np.where(f_j == 0, 0, np.where(f_j == 2, 3, np.where(s_j, 1, 2)))
            lw_new[j] = lw[j] + delta_fam_logits_row[j, b_idx]
            lw_new[j] -= float(logsumexp(lw_new[j]))
        lw = lw_new

    vs = np.exp(np.clip(delta_log_var_row, -0.8, 0.8)) if delta_log_var_row is not None else None
    return lw, mh, pv, delta_g_row_u, vs


def evaluate_decoder_mode_on_folds(
    obs: OffGridObservation,
    truth: OffGridTruth,
    folds_res: ConsistentFoldsResult,
    *,
    mode: str = "mixture_llr",   # "mean_channel" | "hard_choice" | "mixture_llr"
    mix_cfg: MixtureDecodeConfig = MixtureDecodeConfig(),
    use_burst_guard: bool = True,
    neural_overrides: Optional[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    # neural_overrides[fold_idx] = (delta_fam [K, T, 4], delta_g_u [K, T, N], delta_log_var [K, T])
    also_score_shared_hard: bool = False,
) -> dict[str, Any]:
    """Decode all 18 data rows from a precomputed `ConsistentFoldsResult` and compute full metrics."""
    t0 = time.perf_counter()
    n0 = float(folds_res.n0)
    s_mag = math.sqrt(n0)
    s_var = n0
    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[folds_res.probe_row] = 0.0 + 0.0j
    y_b_u = y_b_safe / s_mag

    row_to_idx = {int(r): i for i, r in enumerate(truth.data_rows)}
    row_to_fold = {}
    for snap in folds_res.folds:
        for r_t in snap.t_rows:
            row_to_fold[int(r_t)] = snap

    block_errors = 0
    bit_errors = 0
    shared_hard_errors = 0
    true_cw_in_union = 0
    total_candidates = 0
    frob_rel_list: list[float] = []
    mse_ch_list: list[float] = []
    hard_mse_ch_list: list[float] = []
    branch_counts: list[int] = []

    for r_t in truth.data_rows:
        r_i = int(r_t)
        d_idx = row_to_idx[r_i]
        snap = row_to_fold[r_i]

        d_fam = d_g = d_lv = None
        if neural_overrides is not None and snap.fold_idx in neural_overrides:
            ov_fam, ov_g, ov_lv = neural_overrides[snap.fold_idx]
            d_fam = ov_fam[:, r_i, :]
            d_g = ov_g[:, r_i, :]
            d_lv = ov_lv[:, r_i]

        lw, mh, pv, dg_use, vs_use = _apply_neural_override_to_fold(snap, r_i, d_fam, d_g, d_lv)

        # Compute exact pre-burst moments for C*-invariant burst guard
        br_pre = build_coherent_row_mixture(
            r_i, lw, mh, pv, snap.rot_bank, snap.d_mats, cfg=mix_cfg, noise_mult=1.0,
            delta_g_row_u=dg_use, var_scale_row=vs_use,
        )
        nm_r = _compute_burst_mult_row(y_b_u[r_i], br_pre.mean_ray_u, br_pre.var_ray_u) if use_burst_guard else 1.0
        if nm_r != 1.0:
            br = build_coherent_row_mixture(
                r_i, lw, mh, pv, snap.rot_bank, snap.d_mats, cfg=mix_cfg, noise_mult=nm_r,
                delta_g_row_u=dg_use, var_scale_row=vs_use,
            )
        else:
            br = br_pre

        branch_counts.append(len(br.log_w))

        if mode == "mean_channel":
            out = decode_row_mean_channel(y_b_u[r_i], br)
            h_est_phys = br.h_mean_u * s_mag
        elif mode == "hard_choice":
            out = decode_row_hard_choice(y_b_u[r_i], br)
            m_star = int(np.argmax(br.log_w))
            h_est_phys = br.h_u[m_star] * s_mag
        elif mode == "mixture_llr":
            out = decode_row_mixture_llr(y_b_u[r_i], br, cfg=mix_cfg, compute_margin=False)
            h_est_phys = br.h_mean_u * s_mag
        else:
            raise ValueError(f"Unknown decode mode: {mode!r}")

        diff_bits = out.info_bits != truth.b_info[d_idx]
        if bool(diff_bits.any()):
            block_errors += 1
            bit_errors += int(diff_bits.sum())

        total_candidates += int(out.n_candidates)

        if also_score_shared_hard:
            true_u64 = pack_codewords(truth.b_code[d_idx])
            in_list = bool(
                np.any((out.candidates_u64[:, 0] == true_u64[0]) & (out.candidates_u64[:, 1] == true_u64[1]))
            )
            if in_list:
                true_cw_in_union += 1
            sh_info = score_hard_choice_on_shared_union(y_b_u[r_i], br, out.candidates_u64)
            if not np.array_equal(sh_info, truth.b_info[d_idx]):
                shared_hard_errors += 1

        h_true = truth.h_b_true_mat[r_i]
        dh = h_est_phys - h_true
        hn = float(np.linalg.norm(h_true))
        frob_rel_list.append(float(np.linalg.norm(dh)) / max(hn, 1e-30))
        mse_ch_list.append(float(np.sum(dh.real ** 2 + dh.imag ** 2)) / max(float(np.sum(h_true.real ** 2 + h_true.imag ** 2)), 1e-30))
        dh_hard = br.h_u[int(np.argmax(br.log_w))] * s_mag - h_true
        hard_mse_ch_list.append(
            float(np.sum(dh_hard.real ** 2 + dh_hard.imag ** 2)) / max(float(np.sum(h_true.real ** 2 + h_true.imag ** 2)), 1e-30)
        )

    # Probe mixture across the 3 folds (with optional neural refinement at probe_row)
    probe_preds: list[np.ndarray] = []
    probe_vars: list[np.ndarray] = []
    p_row = folds_res.probe_row
    for snap in folds_res.folds:
        d_fam_p = d_g_p = d_lv_p = None
        if neural_overrides is not None and snap.fold_idx in neural_overrides:
            ov_fam, ov_g, ov_lv = neural_overrides[snap.fold_idx]
            d_fam_p = ov_fam[:, p_row, :]
            d_g_p = ov_g[:, p_row, :]
            d_lv_p = ov_lv[:, p_row]
        lw_p, mh_p, pv_p, dg_p, vs_p = _apply_neural_override_to_fold(snap, p_row, d_fam_p, d_g_p, d_lv_p)
        w_p = np.exp(lw_p - logsumexp(lw_p, axis=1, keepdims=True))
        m_p_u = np.einsum("kh,khn->kn", w_p * mh_p, snap.rot_bank[:, :, p_row, :])
        sec_p = np.sum(w_p * (np.abs(mh_p) ** 2 + pv_p), axis=1)
        v_p_u = np.maximum(sec_p[:, None] - (m_p_u.real ** 2 + m_p_u.imag ** 2), 1e-12)
        if dg_p is not None:
            m_p_u = m_p_u + dg_p
        if vs_p is not None:
            v_p_u = np.maximum(v_p_u * vs_p[:, None], 1e-12)
        if mode == "hard_choice":
            # Probe prediction under hard MAP mode of dominant ray
            dom_k = int(snap.ray_power_order[0])
            h_star = int(np.argmax(lw_p[dom_k]))
            m_p_u = m_p_u.copy()
            v_p_u = v_p_u.copy()
            m_p_u[dom_k] = mh_p[dom_k, h_star] * snap.rot_bank[dom_k, h_star, p_row, :] + (dg_p[dom_k] if dg_p is not None else 0.0)
            v_p_u[dom_k] = max(float(pv_p[dom_k, h_star]) * (float(vs_p[dom_k]) if vs_p is not None else 1.0), 1e-12)
        probe_preds.append(np.sum((m_p_u * s_mag) * snap.u_probe, axis=0))
        probe_vars.append(n0 + np.sum((v_p_u * s_var) * (np.abs(snap.u_probe) ** 2), axis=0))

    p_stack = np.stack(probe_preds, axis=0)
    pv_stack = np.stack(probe_vars, axis=0)
    mix_probe_pred = np.mean(p_stack, axis=0)
    mix_probe_var = np.mean(pv_stack, axis=0) + np.mean(np.abs(p_stack - mix_probe_pred[None, :]) ** 2, axis=0)

    pe_clean = mix_probe_pred - truth.y_b_probe_clean
    pn_clean = float(np.sum(truth.y_b_probe_clean.real ** 2 + truth.y_b_probe_clean.imag ** 2))
    probe_clean_nmse = float(np.sum(pe_clean.real ** 2 + pe_clean.imag ** 2) / max(pn_clean, 1e-30))

    pe_noisy = mix_probe_pred - truth.y_b_probe_noisy
    pe_noisy_sq = pe_noisy.real ** 2 + pe_noisy.imag ** 2
    probe_cal_ratio = float(np.sum(pe_noisy_sq) / max(float(np.sum(mix_probe_var)), 1e-30))
    probe_nll = float(np.mean(np.log(np.maximum(mix_probe_var, 1e-300) / n0) + pe_noisy_sq / np.maximum(mix_probe_var, 1e-300)))

    false_a = sum(
        1 for rec in folds_res.ledger if not np.array_equal(rec["injected_info"], truth.b_info[row_to_idx[int(rec["row"])]])
    )

    decode_ms = (time.perf_counter() - t0) * 1000.0
    out_dict: dict[str, Any] = {
        "blocks": len(truth.data_rows),
        "block_errors": int(block_errors),
        "info_bits": len(truth.data_rows) * 64,
        "bit_errors": int(bit_errors),
        "mean_rel_frob_err": float(np.mean(frob_rel_list)),
        "mean_rel_mse_ch": float(np.mean(mse_ch_list)),
        "mean_hard_rel_mse_ch": float(np.mean(hard_mse_ch_list)),
        "probe_clean_nmse": probe_clean_nmse,
        "probe_cal_ratio": probe_cal_ratio,
        "probe_nll_per_sample": probe_nll,
        "delay_reset_triggered": bool(folds_res.delay_reset_triggered),
        "helpers_accepted": int(folds_res.helpers_accepted),
        "false_assimilations": int(false_a),
        "mean_branches": float(np.mean(branch_counts)),
        "mean_candidates": float(total_candidates) / float(len(truth.data_rows)),
        "aux_reads": int(folds_res.aux_reads),
        "helper_key_reads": int(folds_res.helper_key_reads),
        "helper_value_reads": int(folds_res.helper_value_reads),
        "fold_ms": float(folds_res.elapsed_ms),
        "decode_ms": float(decode_ms),
        "elapsed_ms": float(folds_res.elapsed_ms + decode_ms),
    }
    if also_score_shared_hard:
        out_dict["shared_union_hard_errors"] = int(shared_hard_errors)
        out_dict["true_cw_in_union_count"] = int(true_cw_in_union)
    return out_dict


__all__ = [
    "ConsistentFoldsResult",
    "ContentAddressingConfig",
    "FoldPosteriorSnapshot",
    "MixtureDecodeConfig",
    "RowDecodeOutcome",
    "RowMixtureBranches",
    "build_coherent_row_mixture",
    "decode_row_hard_choice",
    "decode_row_mean_channel",
    "decode_row_mixture_llr",
    "evaluate_decoder_mode_on_folds",
    "mix_branch_bit_llrs_exact",
    "run_consistent_folds",
    "score_hard_choice_on_shared_union",
    "warmup_p12_offgrid_numba",
]
