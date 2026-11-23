"""Exact analytic conditional solvers (Discrete C4 & Continuous Circular Quadrature) for Q1, Q2, Q3."""
import math
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.special import erfc, ndtr

from research.stage3_ministral_cache.schema import (
    ConditionSpec,
    N_SAMPLES,
    ROOTS,
    Reference,
    dedup_references,
    matched_sum,
)


QUAD_NODES_DEFAULT = 128


def awgn_sector_kernel(snr: float) -> np.ndarray:
    q = 0.5 * float(erfc(math.sqrt(snr / 2.0)))
    return np.array([(1.0 - q) ** 2, q * (1.0 - q), q ** 2, q * (1.0 - q)], dtype=np.float64)


def cont_sector_probs(theta: np.ndarray, snr: float) -> np.ndarray:
    scale = math.sqrt(2.0 * snr)
    p_plus = ndtr(scale * np.cos(theta + math.pi / 4.0))
    q_plus = ndtr(scale * np.sin(theta + math.pi / 4.0))
    return np.stack(
        [
            p_plus * q_plus,
            (1.0 - p_plus) * q_plus,
            (1.0 - p_plus) * (1.0 - q_plus),
            p_plus * (1.0 - q_plus),
        ],
        axis=-1,
    )


def discrete_walk_dist(d: float, steps: int) -> np.ndarray:
    k = 1.0 - 4.0 * d / 3.0
    c = 1.0 if steps <= 0 else (k ** steps)
    p = np.full(4, (1.0 - c) / 4.0, dtype=np.float64)
    p[0] += c
    return p


def wrapped_gaussian_weights(mean: float, variance: float, grid: int = QUAD_NODES_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    phi = (2.0 * np.pi / grid) * np.arange(grid, dtype=np.float64) - np.pi
    if variance <= 0.0:
        w = np.zeros(grid, dtype=np.float64)
        idx = int(np.argmin(np.abs(((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi)))
        w[idx] = 1.0
        # For exact deterministic nonzero phase mean (e.g. omega*t with sigma=0),
        # return exact angle `[mean]` if caller checks variance==0, else grid.
        return phi, w
    if not np.isfinite(variance):
        return phi, np.full(grid, 1.0 / grid, dtype=np.float64)
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    disp = ((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi
    w = np.zeros(grid, dtype=np.float64)
    for m in range(-n_img, n_img + 1):
        w += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)
    w /= w.sum()
    return phi, w


def wrapped_sector_quad_probs(mean: float, variance: float) -> np.ndarray:
    """Exact mass of wrapped N(mean, variance) inside each QPSK sector [-pi/4, pi/4) + k*pi/2."""
    if variance <= 0.0:
        out = np.zeros(4, dtype=np.float64)
        z = complex(math.cos(mean), math.sin(mean))
        u = z.real + z.imag
        v = z.imag - z.real
        if u > 0 and v >= 0:
            out[1] = 1.0
        elif u <= 0 and v > 0:
            out[2] = 1.0
        elif u < 0 and v <= 0:
            out[3] = 1.0
        else:
            out[0] = 1.0
        return out
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    out = np.zeros(4, dtype=np.float64)
    for k in range(4):
        lo_base = -0.25 * math.pi + 0.5 * math.pi * k
        hi_base = lo_base + 0.5 * math.pi
        mass = 0.0
        for m in range(-n_img, n_img + 1):
            lo = lo_base + 2.0 * math.pi * m
            hi = hi_base + 2.0 * math.pi * m
            mass += float(ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd))
        out[k] = mass
    out /= out.sum()
    return out


def iq_likelihood(windows: np.ndarray, snr: float) -> np.ndarray:
    """Exact 4-class QPSK posterior a_j(u) from 3-sample complex windows (M, 3)."""
    t = matched_sum(windows)
    logw = 2.0 * snr * (t[:, None] * ROOTS[None, :].conj()).real
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / np.sort(w, axis=1).sum(axis=1, keepdims=True)


def diff_posterior(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact sorted-sum circular difference P(right - left = k mod 4)."""
    sums = []
    for k in range(4):
        prod = left * np.roll(right, -k, axis=-1)
        sums.append(np.sort(prod, axis=-1).sum(axis=-1))
    return np.stack(sums, axis=-1)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 1e-300)
    return arr / np.sort(arr, axis=-1).sum(axis=-1, keepdims=True)


def _gaussian_transition_matrix(
    phi_from: np.ndarray,
    phi_to: np.ndarray,
    mean_shift: float,
    variance: float,
) -> np.ndarray:
    """Matrix K[i, j] proportional to wrapped N(phi_to[i] - phi_from[j] - mean_shift; variance)."""
    if variance <= 0.0:
        # Nearest-node delta fallback when dt=0 or sigma=0
        mat = np.zeros((len(phi_to), len(phi_from)), dtype=np.float64)
        for j, pf in enumerate(phi_from):
            disp = np.abs(((phi_to - pf - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi)
            mat[int(np.argmin(disp)), j] = 1.0
        return mat
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    disp = ((phi_to[:, None] - phi_from[None, :] - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi
    mat = np.zeros_like(disp)
    for m in range(-n_img, n_img + 1):
        mat += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)
    mat /= mat.sum(axis=0, keepdims=True)
    return mat


def _pilot_c4_likelihood(
    t_pilot_sum: Optional[np.ndarray],
    snr: float,
    phi_grid: np.ndarray,
) -> Optional[np.ndarray]:
    """L_p(phi_p, g) for pilot sum T_p, shape (M, G, 4), or None if no valid pilots."""
    if t_pilot_sum is None:
        return None
    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)
    lw = 2.0 * snr * np.real(t_pilot_sum[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)
    lw -= np.max(lw, axis=(1, 2), keepdims=True)
    return np.exp(lw)


def _window2_phase_likelihood(t2: np.ndarray, snr: float, phi_grid: np.ndarray) -> np.ndarray:
    """C4-periodic likelihood L_2(phi_2) = (1/4) sum_v exp(2*snr*Re(T_2 * conj(s_v) * exp(-i*phi_2)))."""
    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)
    lw = 2.0 * snr * np.real(t2[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)
    m_lw = np.max(lw, axis=2, keepdims=True)
    lik = np.exp(lw - m_lw).sum(axis=2) * np.exp(m_lw[:, :, 0] - np.max(lw, axis=(1, 2))[:, None])
    return 0.25 * lik  # (M, G)


def readout_from_table(
    table_z1: np.ndarray,
    table_z2: np.ndarray,
    references: Sequence[Reference],
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    grid: int = QUAD_NODES_DEFAULT,
    allow_double_count_unsafe: bool = False,
) -> dict:
    """Exact readout conditioned ONLY on discrete cache posteriors `(a1, a2)` + deduped refs."""
    a1 = np.asarray(table_z1, dtype=np.float64)
    a2 = np.asarray(table_z2, dtype=np.float64)
    n_obs = len(a1)
    snr = cond.snr

    if not allow_double_count_unsafe:
        valid_refs, _, _, _ = dedup_references(references)
    else:
        valid_refs = [r for r in references if r.shares_channel]

    if valid_refs:
        t_p_sum = sum(matched_sum(r.samples) for r in valid_refs)
    else:
        t_p_sum = None

    if cond.drift_type == "coherent":
        q1 = diff_posterior(a1, a2)
        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())
            lw_p -= lw_p.max(axis=1, keepdims=True)
            bp = np.exp(lw_p)
            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = diff_posterior(bp, a1)
        if protocol == "repeat_x1":
            dk = awgn_sector_kernel(snr)
            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)
            q3 = a1 @ kmat
            mu3 = a1 @ ROOTS
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = q3.copy()

    elif cond.drift_type == "discrete":
        k_rel = 1.0 - 4.0 * cond.d / 3.0
        ch = 1.0 if cond.h == 0 else (k_rel ** cond.h)
        chl = 1.0 if (cond.h + cond.ell) == 0 else (k_rel ** (cond.h + cond.ell))
        q1_raw = diff_posterior(a1, a2)
        q1 = ch * q1_raw + (1.0 - ch) / 4.0

        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            p_0tp = discrete_walk_dist(cond.d, int(round(cond.t_p)))
            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())
            lw_p -= lw_p.max(axis=1, keepdims=True)
            bp_raw = np.exp(lw_p)
            bp_raw /= np.sort(bp_raw, axis=1).sum(axis=1, keepdims=True)
            # Propagate by p_0tp: P(g_p - g = w) -> bp[g] = sum_w p_0tp[w] * bp_raw[(g+w)%4]
            bp = np.stack([sum(p_0tp[w] * bp_raw[:, (g + w) % 4] for w in range(4)) for g in range(4)], axis=1)
            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = diff_posterior(bp, a1)

        if protocol == "repeat_x1":
            dk = awgn_sector_kernel(snr)
            pu3 = chl * a1 + (1.0 - chl) / 4.0
            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)
            q3 = pu3 @ kmat
            mu3 = chl * (a1 @ ROOTS)
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = q3.copy()

    else:
        # continuous Brownian (+ omega)
        var2 = float(cond.h) * (cond.sigma_phi ** 2)
        var3 = float(cond.h + cond.ell) * (cond.sigma_phi ** 2)
        varp = float(cond.t_p) * (cond.sigma_phi ** 2)
        mean2 = cond.omega * float(cond.h)
        mean3 = cond.omega * float(cond.h + cond.ell)
        meanp = cond.omega * float(cond.t_p)

        # Q1: phase rotation weights of a2 shifted by phi_2
        w_quad2 = wrapped_sector_quad_probs(mean2, var2)  # mass in quadrant k for phi_2
        # Pr(u2_apparent = v) -> true u2 distribution rotated back: sum_k w_quad2[k] * a2[:, (u2+k)%4]
        a2_unshifted = sum(w_quad2[k] * np.roll(a2, -k, axis=1) for k in range(4))
        q1 = _normalize_rows(diff_posterior(a1, a2_unshifted))

        # Q2: continuous wrapped-Gaussian pilot prior at t_p
        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            phi_g, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)
            # For exact zero-var deterministic offset (omega*t_p)
            if varp <= 0.0:
                dir_p = ROOTS * np.exp(1j * meanp)
                lw = 2.0 * snr * np.real(t_p_sum[:, None] * np.conj(dir_p)[None, :])
                lw -= lw.max(axis=1, keepdims=True)
                bp = np.exp(lw)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            else:
                lp = _pilot_c4_likelihood(t_p_sum, snr, phi_g)  # (M, G, 4)
                bp = np.einsum("g,mgk->mk", w_p, lp)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = _normalize_rows(diff_posterior(bp, a1))

        # Q3 pre-reveal
        if protocol == "repeat_x1":
            phi_g, w_3 = wrapped_gaussian_weights(mean3, var3, grid=grid)
            if var3 <= 0.0:
                theta_u = 0.5 * math.pi * np.arange(4, dtype=np.float64) + mean3
                mc_u = cont_sector_probs(theta_u, snr)  # (4, 4)
                q3 = a1 @ mc_u
                mu3 = (a1 @ ROOTS) * np.exp(1j * mean3)
            else:
                theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_g[None, :]
                mc_grid = cont_sector_probs(theta_grid, snr)  # (4, G, 4)
                q3 = np.einsum("mu,ugk,g->mk", a1, mc_grid, w_3)
                mu3 = (a1 @ ROOTS) * complex(w_3 @ np.exp(1j * phi_g))
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = _normalize_rows(q3.copy())
        # Note: if pilot reveals phase phi_p and t_p == t_3 (e.g. C6 with finite drift or general),
        # for table readout (which only observes pilot sum) under continuous drift t_p and t_3 correlated,
        # compute exact conditional posterior p(phi_3 | P) if t_p > 0 and sigma_phi > 0!
        if protocol == "repeat_x1" and t_p_sum is not None and cond.sigma_phi > 0.0 and cond.t_p > 0.0:
            t_share = min(float(cond.t_p), float(cond.h + cond.ell))
            phi_p, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)
            lp = _pilot_c4_likelihood(t_p_sum, snr, phi_p)  # (M, G, 4)
            lik_p_marginal = 0.25 * lp.sum(axis=2)  # (M, G)
            post_phip = _normalize_rows(lik_p_marginal * w_p[None, :])
            if abs(float(cond.h + cond.ell) - float(cond.t_p)) < 1e-12:
                post_phi3 = post_phip
            elif float(cond.h + cond.ell) > float(cond.t_p):
                dt = float(cond.h + cond.ell) - float(cond.t_p)
                trans = _gaussian_transition_matrix(phi_p, phi_p, cond.omega * dt, dt * cond.sigma_phi ** 2)
                post_phi3 = post_phip @ trans.T
            else:
                # Brownian bridge from phi_p at t_p back to t_3 < t_p
                alpha = float(cond.h + cond.ell) / float(cond.t_p)
                var_br = alpha * (1.0 - alpha) * varp
                # Note E[phi_3 | phi_p] = alpha * phi_p
                disp = ((phi_p[:, None] - alpha * phi_p[None, :] + math.pi) % (2.0 * math.pi)) - math.pi
                sd_br = math.sqrt(max(var_br, 1e-12))
                n_img = max(1, math.ceil((10.0 * sd_br + math.pi) / (2.0 * math.pi)))
                trans = sum(np.exp(-0.5 * ((disp + 2.0 * math.pi * m) / sd_br) ** 2) for m in range(-n_img, n_img + 1))
                trans /= trans.sum(axis=0, keepdims=True)
                post_phi3 = post_phip @ trans.T
            theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_p[None, :]
            mc_grid = cont_sector_probs(theta_grid, snr)
            q3_after = _normalize_rows(np.einsum("mu,ugk,mg->mk", a1, mc_grid, post_phi3))

    return {
        "q1": _normalize_rows(q1),
        "q2": _normalize_rows(q2),
        "q3": _normalize_rows(q3),
        "q3_after_pilot": _normalize_rows(q3_after),
        "mu3": mu3,
    }


def solve_full_iq_known_param(
    y1: np.ndarray,
    y2: np.ndarray,
    references: Sequence[Reference],
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    grid: int = QUAD_NODES_DEFAULT,
    allow_double_count_unsafe: bool = False,
) -> dict:
    """Analytic conditional Bayesian optimal baseline using UNQUANTIZED full complex I/Q and KNOWN cond params."""
    snr = cond.snr
    a1 = iq_likelihood(y1, snr)
    a2 = iq_likelihood(y2, snr)
    if cond.drift_type in ("coherent", "discrete"):
        # For coherent & discrete C4, a1 and a2 are exact sufficient statistics of (Y1, Y2)!
        return readout_from_table(
            a1,
            a2,
            references,
            cond,
            protocol=protocol,
            grid=grid,
            allow_double_count_unsafe=allow_double_count_unsafe,
        )

    # Continuous Brownian (+ CFO): unquantized T2 informs fractional phase phi_2 mod (pi/2) even when Y2 -> i*Y2!
    n_obs = len(y1)
    t2 = matched_sum(y2)
    if allow_double_count_unsafe:
        valid_refs = [r for r in references if r.shares_channel]
    else:
        valid_refs, _, _, _ = dedup_references(references)
    t_p_sum = sum(matched_sum(r.samples) for r in valid_refs) if valid_refs else None

    h = float(cond.h)
    ell = float(cond.ell)
    tp = float(cond.t_p)
    sig2 = cond.sigma_phi ** 2
    phi_grid = (2.0 * math.pi / grid) * np.arange(grid, dtype=np.float64) - math.pi

    # --- Q1 ---
    if h == 0.0:
        q1 = diff_posterior(a1, a2)
    else:
        _, w2_prior = wrapped_gaussian_weights(cond.omega * h, h * sig2, grid=grid)
        # L(r | phi_2, T1, T2) prop sum_{u1} a1[u1] * exp(2*snr*Re(T2 * conj(s_{u1+r}) * exp(-i*phi_2)))
        phases2 = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)
        lw2 = 2.0 * snr * np.real(t2[:, None, None] * np.conj(phases2)[None, :, :])  # (M, G, 4)
        lw2 -= np.max(lw2, axis=(1, 2), keepdims=True)
        w2_unnorm = np.exp(lw2) * w2_prior[None, :, None]  # (M, G, 4)
        b2_avg = w2_unnorm.sum(axis=1)  # marginal over phi_2, proportional to integrated likelihood of u2!
        b2_avg /= np.sort(b2_avg, axis=1).sum(axis=1, keepdims=True)
        q1 = _normalize_rows(diff_posterior(a1, b2_avg))

    # --- Q3 (before pilot) ---
    if protocol == "independent_x3":
        q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        mu3 = np.zeros(n_obs, dtype=np.complex128)
    else:
        # Compute posterior of phi_2 given t2:
        if h == 0.0 or sig2 == 0.0:
            mean3 = cond.omega * (h + ell)
            var3 = (h + ell) * sig2
            _, w3_post = wrapped_gaussian_weights(mean3, var3, grid=grid)
            post_phi3 = np.broadcast_to(w3_post[None, :], (n_obs, grid)).copy()
        else:
            _, w2_prior = wrapped_gaussian_weights(cond.omega * h, h * sig2, grid=grid)
            lik2 = _window2_phase_likelihood(t2, snr, phi_grid)  # (M, G)
            post_phi2 = _normalize_rows(lik2 * w2_prior[None, :])
            if ell == 0.0:
                post_phi3 = post_phi2
            else:
                trans32 = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * ell, ell * sig2)
                post_phi3 = _normalize_rows(post_phi2 @ trans32.T)

        if (h + ell) * sig2 <= 0.0:
            theta_u = 0.5 * math.pi * np.arange(4, dtype=np.float64) + cond.omega * (h + ell)
            mc_u = cont_sector_probs(theta_u, snr)
            q3 = a1 @ mc_u
            mu3 = (a1 @ ROOTS) * np.exp(1j * cond.omega * (h + ell))
        else:
            theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_grid[None, :]
            mc_grid = cont_sector_probs(theta_grid, snr)  # (4, G, 4)
            q3 = np.einsum("mu,ugk,mg->mk", a1, mc_grid, post_phi3)
            mu3 = (a1 @ ROOTS) * (post_phi3 @ np.exp(1j * phi_grid))

    # --- Q2 & Q3_after_pilot ---
    if t_p_sum is None:
        q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        q3_after = q3.copy()
    else:
        # Exact joint over (phi_2, phi_p) given T2 and T_p!
        if h == 0.0 or sig2 == 0.0:
            # No dependence between t2 and phi_p fractional shift when h=0 or sig2=0
            _, w_p = wrapped_gaussian_weights(cond.omega * tp, tp * sig2, grid=grid)
            if tp * sig2 <= 0.0:
                dir_p = ROOTS * np.exp(1j * cond.omega * tp)
                lw = 2.0 * snr * np.real(t_p_sum[:, None] * np.conj(dir_p)[None, :])
                lw -= lw.max(axis=1, keepdims=True)
                bp = np.exp(lw)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
                post_phi3_p = None
            else:
                lp = _pilot_c4_likelihood(t_p_sum, snr, phi_grid)  # (M, G, 4)
                bp = np.einsum("g,mgk->mk", w_p, lp)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
                post_phi3_p = _normalize_rows((0.25 * lp.sum(axis=2)) * w_p[None, :])
            q2 = _normalize_rows(diff_posterior(bp, a1))
            if protocol == "independent_x3" or post_phi3_p is None:
                q3_after = q3.copy()
            else:
                # Propagate phi_p -> phi_3 if h+ell != tp
                t3 = h + ell
                if abs(t3 - tp) < 1e-12:
                    p3_joint = post_phi3_p
                elif t3 > tp:
                    tr = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * (t3 - tp), (t3 - tp) * sig2)
                    p3_joint = _normalize_rows(post_phi3_p @ tr.T)
                else:
                    # Bridge t3 < tp using alpha = t3/tp (here h=0 => t3=ell < tp)
                    alpha = t3 / tp if tp > 0 else 0.0
                    var_br = max(1e-12, alpha * (1.0 - alpha) * tp * sig2)
                    tr = _gaussian_transition_matrix(phi_grid, phi_grid, 0.0, var_br)
                    # Center at alpha * phi_p
                    sd_br = math.sqrt(var_br)
                    n_img = max(1, math.ceil((10.0 * sd_br + math.pi) / (2.0 * math.pi)))
                    disp = ((phi_grid[:, None] - alpha * phi_grid[None, :] + math.pi) % (2.0 * math.pi)) - math.pi
                    tr = sum(np.exp(-0.5 * ((disp + 2.0 * math.pi * m) / sd_br) ** 2) for m in range(-n_img, n_img + 1))
                    tr /= tr.sum(axis=0, keepdims=True)
                    p3_joint = _normalize_rows(post_phi3_p @ tr.T)
                theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_grid[None, :]
                mc_grid = cont_sector_probs(theta_grid, snr)
                q3_after = np.einsum("mu,ugk,mg->mk", a1, mc_grid, p3_joint)
        else:
            # General h > 0, sig2 > 0 (e.g. C3 with tp=2 between 0 and h=4, or tp >= h)
            _, w2_prior = wrapped_gaussian_weights(cond.omega * h, h * sig2, grid=grid)
            lik2 = _window2_phase_likelihood(t2, snr, phi_grid)  # (M, G2)
            post_phi2 = _normalize_rows(lik2 * w2_prior[None, :])
            lp = _pilot_c4_likelihood(t_p_sum, snr, phi_grid)  # (M, Gp, 4)

            if tp <= h:
                # Bridge kernel p(phi_p | phi_2) has mean (tp/h)*phi_2, var = (tp*(h-tp)/h)*sig2
                # Notice deterministic mean omega*tp == (tp/h)*(omega*h), so centered at (tp/h)*phi_2!
                if tp == 0.0:
                    # phi_p = 0 deterministically
                    lp0 = _pilot_c4_likelihood(t_p_sum, snr, np.array([0.0]))[:, 0, :]  # (M, 4)
                    bp = _normalize_rows(lp0)
                elif abs(tp - h) < 1e-12:
                    # phi_p == phi_2 identically
                    bp = np.einsum("mg,mgk->mk", post_phi2, lp)
                else:
                    alpha = tp / h
                    var_br = alpha * (1.0 - alpha) * h * sig2
                    sd_br = math.sqrt(var_br)
                    n_img = max(1, math.ceil((10.0 * sd_br + math.pi) / (2.0 * math.pi)))
                    disp = ((phi_grid[:, None] - alpha * phi_grid[None, :] + math.pi) % (2.0 * math.pi)) - math.pi
                    trans_p2 = sum(np.exp(-0.5 * ((disp + 2.0 * math.pi * m) / sd_br) ** 2) for m in range(-n_img, n_img + 1))
                    trans_p2 /= trans_p2.sum(axis=0, keepdims=True)  # (Gp, G2)
                    prior_p_given_2 = post_phi2 @ trans_p2.T  # (M, Gp)
                    bp = np.einsum("mg,mgk->mk", prior_p_given_2, lp)
            else:
                # tp > h: phi_p = phi_2 + omega*(tp-h) + N(0, (tp-h)*sig2)
                dt_p = tp - h
                trans_p2 = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * dt_p, dt_p * sig2)  # (Gp, G2)
                prior_p_given_2 = post_phi2 @ trans_p2.T  # (M, Gp)
                bp = np.einsum("mg,mgk->mk", prior_p_given_2, lp)

            bp = _normalize_rows(bp)
            q2 = _normalize_rows(diff_posterior(bp, a1))

            if protocol == "independent_x3":
                q3_after = q3.copy()
            else:
                # Joint Markov update for phi_3 given (T2, T_p)
                lik_p = 0.25 * lp.sum(axis=2)  # (M, Gp)
                t3 = h + ell
                if tp <= h:
                    # Both T2 at h and Tp at tp<=h inform phi_2 -> then step h -> t3
                    if tp == 0.0:
                        post_phi2_tp = post_phi2
                    elif abs(tp - h) < 1e-12:
                        post_phi2_tp = _normalize_rows(post_phi2 * lik_p)
                    else:
                        # Likelihood on phi_2 induced by Tp: (lik_p @ trans_p2)
                        lik2_from_p = lik_p @ trans_p2  # (M, G2)
                        post_phi2_tp = _normalize_rows(post_phi2 * lik2_from_p)
                    if ell == 0.0:
                        post_phi3_joint = post_phi2_tp
                    else:
                        tr32 = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * ell, ell * sig2)
                        post_phi3_joint = _normalize_rows(post_phi2_tp @ tr32.T)
                elif abs(tp - t3) < 1e-12:
                    # tp == t3 == h + ell: post_phi3 from T2 multiplied by lik_p!
                    post_phi3_joint = _normalize_rows(post_phi3 * lik_p)
                elif tp < t3:
                    # h < tp < t3: post(phi_p | T2, Tp) -> step to t3
                    post_phip_tp = _normalize_rows(prior_p_given_2 * lik_p)
                    tr3p = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * (t3 - tp), (t3 - tp) * sig2)
                    post_phi3_joint = _normalize_rows(post_phip_tp @ tr3p.T)
                else:
                    # h < t3 < tp: marginalize at t3 along Markov chain h -> t3 -> tp
                    tr32 = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * ell, ell * sig2)
                    trp3 = _gaussian_transition_matrix(phi_grid, phi_grid, cond.omega * (tp - t3), (tp - t3) * sig2)
                    lik3_from_p = lik_p @ trp3  # (M, G3)
                    prior3_from_2 = post_phi2 @ tr32.T  # (M, G3)
                    post_phi3_joint = _normalize_rows(prior3_from_2 * lik3_from_p)
                theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_grid[None, :]
                mc_grid = cont_sector_probs(theta_grid, snr)
                q3_after = _normalize_rows(np.einsum("mu,ugk,mg->mk", a1, mc_grid, post_phi3_joint))

    return {
        "q1": _normalize_rows(q1),
        "q2": _normalize_rows(q2),
        "q3": _normalize_rows(q3),
        "q3_after_pilot": _normalize_rows(q3_after),
        "mu3": mu3,
    }
