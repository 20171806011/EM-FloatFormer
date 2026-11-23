"""Numerical & exhaustive finite-state verification for Section A theory (THEORY.md)."""
import math
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import erfc, ndtr


ROOTS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
TWO_DB_SNR = 10.0 ** 0.2
N_SAMPLES = 3


def sector_of_complex(z: np.ndarray) -> np.ndarray:
    """Canonical half-open QPSK sectors matching research/matched_cache/learned.py."""
    z = np.asarray(z, dtype=np.complex128)
    u = z.real + z.imag
    v = z.imag - z.real
    out = np.zeros(z.shape, dtype=np.int64)
    out[(u > 0.0) & (v >= 0.0)] = 1
    out[(u <= 0.0) & (v > 0.0)] = 2
    out[(u < 0.0) & (v <= 0.0)] = 3
    return out


def awgn_sector_kernel(snr: float) -> np.ndarray:
    q = 0.5 * float(erfc(math.sqrt(snr / 2.0)))
    return np.array([(1.0 - q) ** 2, q * (1.0 - q), q ** 2, q * (1.0 - q)], dtype=np.float64)


def discrete_walk_dist(d: float, steps: int) -> np.ndarray:
    k = 1.0 - 4.0 * d / 3.0
    c = 1.0 if steps == 0 else (k ** steps)
    p = np.full(4, (1.0 - c) / 4.0, dtype=np.float64)
    p[0] += c
    return p


def wrapped_gaussian_grid(mean: float, variance: float, grid: int = 128) -> tuple[np.ndarray, np.ndarray]:
    phi = (2.0 * np.pi / grid) * np.arange(grid, dtype=np.float64) - np.pi
    if variance == 0.0:
        w = np.zeros(grid, dtype=np.float64)
        idx = int(np.argmin(np.abs(((phi - mean + np.pi) % (2 * np.pi)) - np.pi)))
        w[idx] = 1.0
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


def verify_sector_equivariance() -> dict:
    # Boundaries & 10,000 random non-zero points
    rng = np.random.default_rng(20261123)
    z = rng.normal(size=10000) + 1j * rng.normal(size=10000)
    z = np.where(np.abs(z) < 1e-9, 1.0 + 0.1j, z)
    s0 = sector_of_complex(z)
    s1 = sector_of_complex(1j * z)
    assert np.array_equal(s1, (s0 + 1) % 4), "sector(i*z) != (sector(z)+1)%4"
    # Verify exact diagonal half-open boundary intervals using exact grid coordinates
    pts = np.array([1+0j, 1+1j, 1j, -1+1j, -1+0j, -1-1j, -1j, 1-1j], dtype=np.complex128)
    got = sector_of_complex(pts).tolist()
    assert got == [0, 1, 1, 2, 2, 3, 3, 0], f"Unexpected boundary sectors: {got}"
    # Verify exact match between AWGN closed form at constellation points and discrete kernel
    for db in (0.0, 2.0, 8.0):
        snr = 10.0 ** (db / 10.0)
        mc = cont_sector_probs(np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2]), snr)
        dk = awgn_sector_kernel(snr)
        for k in range(4):
            np.testing.assert_allclose(mc[k], np.roll(dk, k), atol=1e-14, rtol=0)
    # Cross-check cont_sector_probs at arbitrary off-grid angle theta=0.37 via 200k MC draws
    sd = math.sqrt(1.0 / (2.0 * TWO_DB_SNR))
    mc_max_err = 0.0
    for theta in (0.37, 1.91, -1.15):
        z_mc = np.exp(1j * theta) + (rng.normal(scale=sd, size=200000) + 1j * rng.normal(scale=sd, size=200000))
        emp = np.bincount(sector_of_complex(z_mc), minlength=4) / len(z_mc)
        exact = cont_sector_probs(np.array([theta]), TWO_DB_SNR)[0]
        mc_max_err = max(mc_max_err, float(np.max(np.abs(emp - exact))))
    assert mc_max_err < 0.005, f"MC mismatch on cont_sector_probs: {mc_max_err}"
    return {"random_points": 10000, "boundary_sectors": got, "mc_cont_sector_max_err": mc_max_err, "sector_equivariance_pass": True}


def verify_discrete_exact_enumeration() -> dict:
    """Compare discrete model against exhaustive 4^6/4^7 finite-state enumeration for Q1, Q2, Q3, and pre-reveal pilot."""
    e = 0.09
    d = 0.08
    h, ell, tp = 2, 2, 5  # pilot at t=5 (h < tp, delta from t2 is 3; test also mid-pilot below)
    ch = (1.0 - 4.0 * d / 3.0) ** h
    cl = (1.0 - 4.0 * d / 3.0) ** ell
    chl = ch * cl
    p_e = np.array([1 - e, e / 3, e / 3, e / 3], dtype=np.float64)
    p_u = discrete_walk_dist(d, h)
    p_v = discrete_walk_dist(d, ell)
    p_dp = discrete_walk_dist(d, abs(tp - h))  # steps from t2=2 to tp=5 is 3

    # Enumerate over (X1, X2, G, U, V, Dp, N1, N2, N3, Np) using dynamic factoring over latent noise first -> exact!
    # Joint over discrete received observations (Y1, Y2, P) in Z_4^3 -> 64 cells
    joint_q1 = np.zeros((4, 4, 4), dtype=np.float64)
    joint_q2 = np.zeros((4, 4, 4, 4), dtype=np.float64)
    joint_q3_rep = np.zeros((4, 4, 4), dtype=np.float64)
    joint_q3_ind = np.zeros((4, 4, 4), dtype=np.float64)
    joint_p_pre = np.zeros((4, 4, 4), dtype=np.float64)

    for g, x1, x2 in itertools.product(range(4), repeat=3):
        prior_sym = 1.0 / 64.0
        for u, v, dp in itertools.product(range(4), repeat=3):
            pw = p_u[u] * p_v[v] * p_dp[dp]
            u1 = (x1 + g) % 4
            u2 = (x2 + g + u) % 4
            u3_rep = (x1 + g + u + v) % 4
            gp = (g + u + dp) % 4
            for n1, n2 in itertools.product(range(4), repeat=2):
                w12 = prior_sym * pw * p_e[n1] * p_e[n2]
                y1 = (u1 + n1) % 4
                y2 = (u2 + n2) % 4
                joint_q1[y1, y2, (x2 - x1) % 4] += w12
                # Q3 repeat: noisy observation K3 = (u3_rep + n3) % 4
                for n3 in range(4):
                    joint_q3_rep[y1, y2, (u3_rep + n3) % 4] += w12 * p_e[n3]
                # Q3 indep: x3 uniform independent of all
                joint_q3_ind[y1, y2, :] += w12 * 0.25
                # Pilot pre-reveal & Q2 post-reveal
                for np_ in range(4):
                    p_obs = (gp + np_) % 4
                    wp = w12 * p_e[np_]
                    joint_p_pre[y1, y2, p_obs] += wp
                    joint_q2[y1, y2, p_obs, x1] += wp

    # Closed-form formulas
    max_q1 = 0.0
    max_q3_rep = 0.0
    max_q3_ind = 0.0
    max_pre_pilot = 0.0
    max_q2 = 0.0
    for y1, y2 in itertools.product(range(4), repeat=2):
        a1 = np.roll(p_e, y1)
        a2 = np.roll(p_e, y2)

        # Q1
        num_q1 = joint_q1[y1, y2]
        got_q1 = num_q1 / num_q1.sum()
        pred_q1 = np.array([ch * sum(a1[u] * a2[(u + r) % 4] for u in range(4)) + (1 - ch) / 4 for r in range(4)])
        max_q1 = max(max_q1, float(np.max(np.abs(got_q1 - pred_q1))))

        # Q3 repeat
        got_q3_rep = joint_q3_rep[y1, y2] / joint_q3_rep[y1, y2].sum()
        pu3 = chl * a1 + (1 - chl) / 4.0
        pred_q3_rep = np.array([sum(pu3[w] * p_e[(k - w) % 4] for w in range(4)) for k in range(4)])
        max_q3_rep = max(max_q3_rep, float(np.max(np.abs(got_q3_rep - pred_q3_rep))))

        # Q3 indep
        got_q3_ind = joint_q3_ind[y1, y2] / joint_q3_ind[y1, y2].sum()
        max_q3_ind = max(max_q3_ind, float(np.max(np.abs(got_q3_ind - 0.25))))

        # Pre-reveal pilot sector uniformity
        got_pre = joint_p_pre[y1, y2] / joint_p_pre[y1, y2].sum()
        max_pre_pilot = max(max_pre_pilot, float(np.max(np.abs(got_pre - 0.25))))

        # Q2 post-reveal (note pilot at tp=5 shares drift U up to t=h=2, so conditioning on y2 helps infer U!)
        p_t2p = discrete_walk_dist(d, tp - h)
        p_g_given_obs = np.zeros((4, 4))  # [p_obs, g]
        for p_obs in range(4):
            num_q2 = joint_q2[y1, y2, p_obs]
            got_q2 = num_q2 / num_q2.sum()
            # Factored Markov posterior over (g, u):
            # P(g, u | y2, p_obs) prop sum_{x2} (1/4) a2(x2+g+u) sum_{dp} p_dp(dp) p_e(p_obs - g - u - dp)
            # Notice sum_{x2} a2(x2+g+u) = 1 identically, so U_2 carries ZERO info about (g, u) in uniform X2!
            # Wait: check exact Markov formula! sum_{x2} a2(x2+g+u) = sum_{u2} a2(u2) = 1.
            # So P(g | y2, p_obs) = sum_u p_u[u] * (p_t2p * p_e)[p_obs - g - u]!
            conv_dp_e = np.array([sum(p_t2p[dp] * p_e[(p_obs - k - dp) % 4] for dp in range(4)) for k in range(4)])
            # Because p_u[u] and p_t2p convolve to discrete_walk_dist(d, tp):
            p_0tp = discrete_walk_dist(d, tp)
            bp = np.array([sum(p_0tp[w] * p_e[(p_obs - g - w) % 4] for w in range(4)) for g in range(4)])
            pred_q2 = np.array([sum(bp[g] * a1[(x1 + g) % 4] for g in range(4)) for x1 in range(4)])
            pred_q2 /= pred_q2.sum()
            max_q2 = max(max_q2, float(np.max(np.abs(got_q2 - pred_q2))))

    assert max_q1 < 3e-14, max_q1
    assert max_q3_rep < 3e-14, max_q3_rep
    assert max_q3_ind < 3e-14, max_q3_ind
    assert max_pre_pilot < 3e-14, max_pre_pilot
    assert max_q2 < 3e-14, max_q2
    return {
        "max_abs_err_q1": max_q1,
        "max_abs_err_q3_repeat": max_q3_rep,
        "max_abs_err_q3_indep": max_q3_ind,
        "max_abs_err_pre_reveal_pilot_uniform": max_pre_pilot,
        "max_abs_err_q2_post_reveal": max_q2,
    }


def verify_relative_cache_counterexample_and_bounds() -> dict:
    """Verify coherent/decoherent bounds & 2-bit relative-only counterexample on 2 dB & high-SNR."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matched_cache"))
    from physical import Hand

    rng = np.random.default_rng(123456)
    n = 40000
    x1 = rng.integers(4, size=n)
    x2 = rng.integers(4, size=n)
    g = rng.integers(4, size=n)
    snr = TWO_DB_SNR
    sd = math.sqrt(1.0 / (2.0 * snr))
    mean1 = ROOTS[(x1 + g) % 4] / math.sqrt(N_SAMPLES)
    mean2 = ROOTS[(x2 + g) % 4] / math.sqrt(N_SAMPLES)
    y1 = mean1[:, None] + rng.normal(scale=sd, size=(n, N_SAMPLES)) + 1j * rng.normal(scale=sd, size=(n, N_SAMPLES))
    y2 = mean2[:, None] + rng.normal(scale=sd, size=(n, N_SAMPLES)) + 1j * rng.normal(scale=sd, size=(n, N_SAMPLES))

    # Full I/Q Q3 coherent (h=ell=0) repeat_x1
    t1 = y1.sum(axis=1) / math.sqrt(N_SAMPLES)
    t3 = ROOTS[(x1 + g) % 4] + (rng.normal(scale=sd, size=n) + 1j * rng.normal(scale=sd, size=n))
    k3 = sector_of_complex(t3)

    # Full I/Q posterior a1
    logw = 2.0 * snr * (t1[:, None] * ROOTS.conj()).real
    logw -= logw.max(axis=1, keepdims=True)
    a1_full = np.exp(logw)
    a1_full /= a1_full.sum(axis=1, keepdims=True)
    dk = awgn_sector_kernel(snr)
    q3_full = a1_full @ np.array([np.roll(dk, w) for w in range(4)])
    nll_full = float(-np.log(q3_full[np.arange(n), k3]).mean())
    acc_full = float((np.argmax(q3_full, axis=1) == k3).mean())

    # 2-bit relative-only cache Z = H = argmax diff(a1_full, a2_full)
    # By C_4 symmetry H is independent of U_1 and yields strictly 1/4 for Q3 -> NLL = log(4)
    t2 = y2.sum(axis=1) / math.sqrt(N_SAMPLES)
    logw2 = 2.0 * snr * (t2[:, None] * ROOTS.conj()).real
    logw2 -= logw2.max(axis=1, keepdims=True)
    a2_full = np.exp(logw2)
    a2_full /= a2_full.sum(axis=1, keepdims=True)
    p_rel = np.stack([(a1_full * np.roll(a2_full, -r, axis=1)).sum(axis=1) for r in range(4)], axis=1)
    h_rel = np.argmax(p_rel, axis=1)
    empirical_k3_given_h = np.zeros((4, 4))
    for hv in range(4):
        mask = h_rel == hv
        empirical_k3_given_h[hv] = np.bincount(k3[mask], minlength=4) / mask.sum()
    assert float(np.max(np.abs(empirical_k3_given_h - 0.25))) < 0.015  # Monte Carlo 40k/4~10k SE~0.005

    # Hand 4-bit and 8-bit coherent Q3
    res_caches = {}
    for bits in (4, 8):
        hand = Hand(bits)
        z1 = hand.encode(y1)
        a1_c = hand.table[z1]
        q3_c = a1_c @ np.array([np.roll(dk, w) for w in range(4)])
        mu3_c = a1_c @ ROOTS
        nll_c = float(-np.log(q3_c[np.arange(n), k3]).mean())
        acc_c = float((np.argmax(q3_c, axis=1) == k3).mean())
        nmse_c = float(np.mean(np.abs(t3 - mu3_c) ** 2) / np.mean(np.abs(t3) ** 2))
        assert nll_c < math.log(4) - 0.1
        res_caches[f"hand_b{bits}"] = {"acc": acc_c, "nll": nll_c, "nmse": nmse_c}

    # Coherent oracle bound at 2 dB
    q_err = 0.5 * float(erfc(math.sqrt(snr / 2.0)))
    oracle_acc = (1.0 - q_err) ** 2
    oracle_nll = -2.0 * (q_err * math.log(q_err) + (1.0 - q_err) * math.log(1.0 - q_err))

    return {
        "snr_db": 2.0,
        "oracle_coherent_acc": oracle_acc,
        "oracle_coherent_nll": oracle_nll,
        "full_iq_coherent_acc": acc_full,
        "full_iq_coherent_nll": nll_full,
        "rel_2bit_max_cond_k3_dev_from_uniform": float(np.max(np.abs(empirical_k3_given_h - 0.25))),
        "rel_2bit_q3_nll": math.log(4),
        "caches": res_caches,
    }


def verify_continuous_quadrature_and_pilot_density() -> dict:
    """Verify 128-vs-256 continuous quadrature, (0,1) invariance of full-IQ Q3, and non-uniform C-plane pilot density."""
    snr = TWO_DB_SNR
    h, ell, tp = 4, 2, 2  # mid-pilot case C3 (sigma_phi = 0.25 rad/step)
    sigma = 0.25
    omega = 0.0
    var2 = h * sigma ** 2
    var3_given_2 = ell * sigma ** 2

    def eval_full_iq_q3_and_pilot(t1, t2, grid):
        # Posterior over phi2 given t2: L_2(phi2) = (1/4) sum_v exp(2*snr*Re(t2 * conj(s_v) * exp(-i*phi2)))
        phi2, p2_prior = wrapped_gaussian_grid(omega * h, var2, grid=grid)
        # t2: shape (B,)
        phases2 = ROOTS[None, :] * np.exp(1j * phi2)[:, None]  # (G, 4)
        loglik2 = 2.0 * snr * np.real(t2[:, None, None] * np.conj(phases2)[None, :, :])  # (B, G, 4)
        m2 = loglik2.max(axis=2, keepdims=True)
        # sum over x2 -> L2(t2, phi2) shape (B, G)
        lik2 = np.exp(loglik2 - m2).sum(axis=2) * np.exp(m2[:, :, 0] - np.max(loglik2, axis=(1, 2), keepdims=False)[:, None])
        post_phi2 = lik2 * p2_prior[None, :]
        post_phi2 /= post_phi2.sum(axis=1, keepdims=True)

        # Propagate phi2 -> phi3 via kernel N(phi2 + omega*ell, var3_given_2)
        phi3 = phi2
        disp = ((phi3[:, None] - phi2[None, :] - omega * ell + np.pi) % (2.0 * np.pi)) - np.pi
        sd3 = math.sqrt(var3_given_2)
        n_img = max(1, math.ceil((10.0 * sd3 + np.pi) / (2.0 * np.pi)))
        trans = np.zeros((grid, grid), dtype=np.float64)
        for m in range(-n_img, n_img + 1):
            trans += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd3) ** 2)
        trans /= trans.sum(axis=0, keepdims=True)
        post_phi3 = post_phi2 @ trans.T  # (B, G)

        # Posterior over u1 given t1
        lw1 = 2.0 * snr * np.real(t1[:, None] * ROOTS[None, :].conj())
        lw1 -= lw1.max(axis=1, keepdims=True)
        a1 = np.exp(lw1)
        a1 /= a1.sum(axis=1, keepdims=True)

        # Integrate M_cont(pi/2*u1 + phi3) over phi3 and u1
        # theta_grid: (4, G)
        theta_grid = (0.5 * math.pi * np.arange(4)[:, None]) + phi3[None, :]
        mc_grid = cont_sector_probs(theta_grid, snr)  # (4, G, 4)
        # q3[b, k] = sum_{u, g} a1[b, u] * post_phi3[b, g] * mc_grid[u, g, k]
        q3 = np.einsum("bu,bg,ugk->bk", a1, post_phi3, mc_grid)
        # mean mu3 = E[exp(i * phi3) * s_u1] = (a1 @ ROOTS) * (post_phi3 @ exp(i*phi3))
        mu3 = (a1 @ ROOTS) * (post_phi3 @ np.exp(1j * phi3))

        # Pre-reveal pilot density p(t_p | y1, y2) on test grid of C-plane points
        # Compute posterior of phi_p at tp=2 (bridge between 0 and h=4: mean = 0.5*phi2, var = (tp*(h-tp)/h)*sigma^2)
        var_p_cond = (tp * (h - tp) / h) * sigma ** 2
        sd_p = math.sqrt(var_p_cond)
        n_img_p = max(1, math.ceil((10.0 * sd_p + np.pi) / (2.0 * np.pi)))
        disp_p = ((phi2[:, None] - (tp / h) * phi2[None, :] + np.pi) % (2.0 * np.pi)) - np.pi
        trans_p = np.zeros((grid, grid), dtype=np.float64)
        for m in range(-n_img_p, n_img_p + 1):
            trans_p += np.exp(-0.5 * ((disp_p + 2.0 * np.pi * m) / sd_p) ** 2)
        trans_p /= trans_p.sum(axis=0, keepdims=True)
        post_phip = post_phi2 @ trans_p.T  # (B, G)
        return q3, mu3, post_phip, phi2

    rng = np.random.default_rng(98765)
    b = 64
    x1 = rng.integers(4, size=b)
    x2 = rng.integers(4, size=b)
    g = rng.integers(4, size=b)
    d_p = rng.normal(scale=math.sqrt(tp * sigma ** 2), size=b)
    d_2p = rng.normal(scale=math.sqrt((h - tp) * sigma ** 2), size=b)
    d_32 = rng.normal(scale=math.sqrt(ell * sigma ** 2), size=b)
    phi_p_true = d_p
    phi_2_true = d_p + d_2p
    phi_3_true = phi_2_true + d_32
    sdn = math.sqrt(1.0 / (2.0 * snr))
    t1 = ROOTS[(x1 + g) % 4] + (rng.normal(scale=sdn, size=b) + 1j * rng.normal(scale=sdn, size=b))
    t2 = ROOTS[(x2 + g) % 4] * np.exp(1j * phi_2_true) + (rng.normal(scale=sdn, size=b) + 1j * rng.normal(scale=sdn, size=b))

    q3_128, mu3_128, phip_128, grid_128 = eval_full_iq_q3_and_pilot(t1, t2, 128)
    q3_256, mu3_256, _, _ = eval_full_iq_q3_and_pilot(t1, t2, 256)
    mesh_diff_q3 = float(np.max(np.abs(q3_128 - q3_256)))
    mesh_diff_mu3 = float(np.max(np.abs(mu3_128 - mu3_256)))
    assert mesh_diff_q3 < 1e-12, mesh_diff_q3
    assert mesh_diff_mu3 < 1e-12, mesh_diff_mu3

    # Check (0, 1) and (1, 1) rotation laws under continuous drift!
    q3_01, mu3_01, _, _ = eval_full_iq_q3_and_pilot(t1, 1j * t2, 128)
    q3_11, mu3_11, _, _ = eval_full_iq_q3_and_pilot(1j * t1, 1j * t2, 128)
    hold_01_err = float(np.max(np.abs(q3_01 - q3_128)))
    rot_11_err = float(np.max(np.abs(q3_11 - np.roll(q3_128, 1, axis=1))))
    mu_01_err = float(np.max(np.abs(mu3_01 - mu3_128)))
    mu_11_err = float(np.max(np.abs(mu3_11 - 1j * mu3_128)))
    assert hold_01_err < 1e-13
    assert rot_11_err < 1e-13
    assert mu_01_err < 1e-13
    assert mu_11_err < 1e-13

    # Check pre-reveal continuous pilot C-plane density vs 4-sector uniformity:
    # Evaluate p(t_p | y1[0], y2[0]) at t=1.0 vs t=0.0 vs t=1.0*exp(i*0.25) vs t=i*1.0
    phip0 = phip_128[0]  # (128,)
    def eval_pilot_pdf(z_arr):
        z_arr = np.asarray(z_arr, dtype=np.complex128)
        centers = ROOTS[:, None] * np.exp(1j * grid_128)[None, :]  # (4, 128)
        dens = (snr / math.pi) * np.exp(-snr * np.abs(z_arr[:, None, None] - centers[None, :, :]) ** 2)
        return np.einsum("g,zkg->z", phip0, dens) / 4.0

    pts_test = np.array([1.0 + 0j, 1j, 0.0 + 0j, np.exp(1j * math.pi / 4.0), 1.0 * np.exp(1j * 0.25), 3.0 + 0j])
    pdfs = eval_pilot_pdf(pts_test)
    # C4 symmetry holds strictly: pdf(1) == pdf(i), yet pdf varies both radially and angularly inside each sector!
    assert abs(pdfs[0] - pdfs[1]) < 1e-13
    assert abs(pdfs[0] - pdfs[2]) > 0.015  # 1+0j peak vs origin trough
    assert abs(pdfs[0] - pdfs[3]) > 1e-3 or abs(pdfs[0] - pdfs[5]) > 0.10  # angular & radial non-uniformity
    # Also verify exact sector integral over sector 0..3 using 128-node quadrature of cont_sector_probs == 0.25
    sector_probs_pre = (0.25 * sum(cont_sector_probs(0.5 * math.pi * g0 + grid_128, snr) for g0 in range(4))).T @ phip0
    max_sector_dev_pre = float(np.max(np.abs(sector_probs_pre - 0.25)))
    assert max_sector_dev_pre < 1e-13

    return {
        "mesh_doubling_128_vs_256_max_q3_diff": mesh_diff_q3,
        "mesh_doubling_128_vs_256_max_mu3_diff": mesh_diff_mu3,
        "full_iq_rot01_hold_err": hold_01_err,
        "full_iq_rot11_roll_err": rot_11_err,
        "pilot_c4_pdf_at_1_and_i_diff": float(abs(pdfs[0] - pdfs[1])),
        "pilot_nonuniform_pdf_1_vs_0_vs_diag_vs_3": [float(pdfs[0]), float(pdfs[2]), float(pdfs[3]), float(pdfs[5])],
        "pre_reveal_pilot_sector_max_dev_from_0.25": max_sector_dev_pre,
    }


def main():
    report = {
        "sector_equivariance": verify_sector_equivariance(),
        "discrete_exact_enumeration": verify_discrete_exact_enumeration(),
        "relative_cache_counterexample_and_bounds": verify_relative_cache_counterexample_and_bounds(),
        "continuous_quadrature_and_pilot_density": verify_continuous_quadrature_and_pilot_density(),
    }
    out_path = Path(__file__).resolve().parent / "verify_theory_output.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("ALL SECTION A THEORY VERIFICATIONS PASSED.")


if __name__ == "__main__":
    main()
