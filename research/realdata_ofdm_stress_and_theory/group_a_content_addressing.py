"""Group A: Empirical Verification of Content-Addressing Theory on Real OTA OFDM.

Tests Propositions A & B, the margin upper bound U, and the shared-observation correlation
bias (Eq. 9, §5.1) from `research/content_addressing_scalar/REPORT.md` on real 435 MHz
SDR OTA OFDM waveforms (`sdr_ofdm_dataset_v1`).

Two complementary, strictly leak-free empirical experiments:
  - Experiment A1 (Real OTA Sparse-Pilot + Train-Only Historical Reference Bank, M=16):
      Evaluates 14-pilot (full), 7-pilot (sparse), and 4-pilot (ultra-sparse) regimes on
      Test (240 TTIs, 12 contiguous blocks x 20 TTIs). Compares:
        1. Single-TTI classical LS + Linear interpolation
        2. Single-TTI classical LS + Cubic Spline interpolation
        3. Fixed Local-k (k=4) uniform historical reference fusion
        4. Global Uniform (M=16) historical reference fusion
        5. Classical inverse-distance reliability weighting (M=16)
        6. Global Content-Addressing Softmax Attention (Dev-calibrated temperature tau* & lambda*)
        7. Boundary failure controls: over-cold hard selection (tau -> 0) and over-hot (20*tau*)
      Reports held-out real pilot MSE, full-101 oracle channel NMSE, 1400-RE 16-QAM BER/SER,
      and 12-block exact sign-flip tests.

  - Experiment A2 (Exact Decomposition Check of Propositions A, B, Margin Bound U, & Eq. 9):
      Splits each real Test TTI across disjoint OFDM symbols into:
        - Target real channel H_target (odd data symbols 1,3,5,7,9,11,13)
        - Independent valid real reference H_valid (even data symbols 0,4,6,8,10,12)
        - Common pilot baseline m(P) (sparse 7 or 4 pilots on symbol 2)
      Constructs equal-marginal real innovation pool {V_1..V_16} with hidden random index J
      and 15 Train-drawn real distractors. Verifies:
        - Proposition A local lower bound L_k = v_0 - D*(k/M) vs empirical local-k MSE
        - Proposition B exact risk R_alpha = v_1 + D*E||e_J - alpha||_2^2 across temperature sweep
        - Margin upper bound U = v_1 + 2D[(1-eta)r^2 + eta] and effective alpha_J, Delta, eta
        - §5.1 Eq. (9) shared-observation correlation discrepancy Delta_corr when the same noisy
          observation is reused for both scoring and value estimation!
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from hypothesis_frontend import build_train_reference_bank
from ofdm_data import (
    DATA_COORDS_101,
    HELD_OUT_PILOTS_FROM_4,
    HELD_OUT_PILOTS_FROM_7,
    N_FFT,
    N_SYM,
    PILOT_SCS_4,
    PILOT_SCS_7,
    PILOT_SCS_14,
    PILOT_SYM,
    USED_SCS,
    RealOFDMDataset,
    build_oracle_channel,
    build_temporal_splits,
    compute_bit_metrics,
    estimate_noise_var_from_grid,
    exact_16qam_llrs,
    exact_sign_flip_test,
    load_real_ofdm_dataset,
    ls_linear_channel,
    pilot_tx_for_scs,
    save_json,
)


def _align_bank_to_pilots(
    hp_obs: np.ndarray,      # [P] complex128
    used_p_idx: np.ndarray,  # [P] int
    ref_bank: np.ndarray,    # [M, 101] complex128
) -> Tuple[np.ndarray, np.ndarray]:
    """Gauge-equivariant optimal complex scalar alignment of each reference j to visible pilots."""
    ref_p = ref_bank[:, used_p_idx]  # [M, P]
    inner = np.sum(np.conj(ref_p) * hp_obs[None, :], axis=1)
    ref_norm2 = np.maximum(np.sum(np.abs(ref_p) ** 2, axis=1), 1e-12)
    c_opt = inner / ref_norm2
    aligned = c_opt[:, None] * ref_bank  # [M, 101]
    resid_p = hp_obs[None, :] - aligned[:, used_p_idx]
    mse_p = np.mean(np.abs(resid_p) ** 2, axis=1)  # [M]
    return aligned, mse_p


def _run_a1_estimator_on_split(
    ds: RealOFDMDataset,
    indices: np.ndarray,
    h_oracle_all: np.ndarray,
    ref_bank: np.ndarray,
    pilot_scs: np.ndarray,
    held_out_pilots: np.ndarray | None,
    method: str,
    *,
    tau_ca: float = 1.0,
    lam_ca: float = 0.5,
    k_local: int = 4,
) -> Tuple[List[dict], dict]:
    """Run one A1 estimator over given TTI indices and return per-TTI metrics + summary."""
    p_tx = pilot_tx_for_scs(pilot_scs)
    used_p_idx = np.searchsorted(USED_SCS, pilot_scs)
    ho_used_idx = np.searchsorted(USED_SCS, held_out_pilots) if held_out_pilots is not None else None
    ho_ptx = pilot_tx_for_scs(held_out_pilots) if held_out_pilots is not None else None

    rows: List[dict] = []
    for t in indices:
        y_grid = ds.iq[t]
        h_ls = ls_linear_channel(y_grid, pilot_scs)
        hp_obs = y_grid[PILOT_SYM, pilot_scs] / p_tx
        n0 = estimate_noise_var_from_grid(y_grid, h_ls, pilot_scs)

        if method == "single_tti_ls_linear":
            h_hat = h_ls
            alpha_max = 0.0
            ent_w = 0.0
        elif method == "single_tti_ls_spline":
            h_hat = CubicSpline(pilot_scs, hp_obs.real)(USED_SCS) + 1j * CubicSpline(pilot_scs, hp_obs.imag)(USED_SCS)
            alpha_max = 0.0
            ent_w = 0.0
        else:
            aligned, mse_p = _align_bank_to_pilots(hp_obs, used_p_idx, ref_bank)
            m_cnt = len(ref_bank)
            if method == "local_k4_uniform":
                w = np.zeros(m_cnt, dtype=np.float64)
                w[:k_local] = 1.0 / float(k_local)
            elif method == "global_m16_uniform":
                w = np.full(m_cnt, 1.0 / float(m_cnt), dtype=np.float64)
            elif method == "classical_invdist_weight":
                eps_d = max(float(np.median(mse_p)) * 0.1, 1e-8)
                inv_d = 1.0 / (mse_p + eps_d)
                w = inv_d / float(np.sum(inv_d))
            elif method in ("content_addr_calibrated", "content_addr_cold_hard", "content_addr_hot"):
                scale_p = max(float(np.mean(np.abs(hp_obs) ** 2)) * 0.05, 1e-6)
                scores = -mse_p / scale_p
                if method == "content_addr_cold_hard":
                    w = np.zeros(m_cnt, dtype=np.float64)
                    w[int(np.argmax(scores))] = 1.0
                else:
                    t_eff = float(tau_ca) if method == "content_addr_calibrated" else float(tau_ca) * 20.0
                    sc = scores / max(t_eff, 1e-8)
                    sc -= float(np.max(sc))
                    ex = np.exp(sc)
                    w = ex / float(np.sum(ex))
            else:
                raise ValueError(f"Unknown A1 method: {method}")

            h_ref = np.sum(w[:, None] * aligned, axis=0)
            h_hat = (1.0 - lam_ca) * h_ls + lam_ca * h_ref
            alpha_max = float(np.max(w))
            ent_w = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))

        # Offline evaluation metrics
        h_true = h_oracle_all[t]
        h_true_pwr = max(float(np.mean(np.abs(h_true) ** 2)), 1e-12)
        ch_mse = float(np.mean(np.abs(h_hat - h_true) ** 2))
        ch_nmse = ch_mse / h_true_pwr

        if held_out_pilots is not None and ho_used_idx is not None and ho_ptx is not None:
            hp_ho_obs = y_grid[PILOT_SYM, held_out_pilots] / ho_ptx
            ho_mse = float(np.mean(np.abs(h_hat[ho_used_idx] - hp_ho_obs) ** 2))
            ho_nmse = ho_mse / max(float(np.mean(np.abs(hp_ho_obs) ** 2)), 1e-12)
        else:
            ho_mse = float("nan")
            ho_nmse = float("nan")

        # End-to-end 1400-RE 16-QAM demodulation
        y_used = y_grid[:, USED_SCS]
        y_data = y_used[DATA_COORDS_101[:, 0], DATA_COORDS_101[:, 1]]
        h_data = h_hat[DATA_COORDS_101[:, 1]]
        llrs = exact_16qam_llrs(y_data, h_data, n0)
        bm = compute_bit_metrics(llrs, ds.labels[t])

        rows.append(
            {
                "tti": int(t),
                "sinr_db": float(ds.sinr_db[t]),
                "ch_mse": ch_mse,
                "ch_nmse": ch_nmse,
                "ho_pilot_mse": ho_mse,
                "ho_pilot_nmse": ho_nmse,
                "ber": bm["ber"],
                "ser": bm["ser"],
                "nll": bm["nll"],
                "ece": bm["ece"],
                "alpha_max": alpha_max,
                "weight_entropy": ent_w,
            }
        )

    summary = {
        "ch_mse": float(np.mean([r["ch_mse"] for r in rows])),
        "ch_nmse": float(np.mean([r["ch_nmse"] for r in rows])),
        "ch_evm_pct": float(100.0 * math.sqrt(np.mean([r["ch_nmse"] for r in rows]))),
        "ho_pilot_mse": float(np.nanmean([r["ho_pilot_mse"] for r in rows])) if held_out_pilots is not None else None,
        "ho_pilot_nmse": float(np.nanmean([r["ho_pilot_nmse"] for r in rows])) if held_out_pilots is not None else None,
        "ber": float(np.mean([r["ber"] for r in rows])),
        "ser": float(np.mean([r["ser"] for r in rows])),
        "nll": float(np.mean([r["nll"] for r in rows])),
        "ece": float(np.mean([r["ece"] for r in rows])),
        "alpha_max_mean": float(np.mean([r["alpha_max"] for r in rows])),
        "weight_entropy_mean": float(np.mean([r["weight_entropy"] for r in rows])),
    }
    return rows, summary


def calibrate_a1_on_dev(
    ds: RealOFDMDataset,
    dev_idx: np.ndarray,
    h_oracle_all: np.ndarray,
    ref_bank: np.ndarray,
    pilot_scs: np.ndarray,
    held_out_pilots: np.ndarray | None,
) -> Tuple[float, float]:
    """Select (tau_ca, lam_ca) strictly on Dev minimizing Dev held-out pilot MSE (or BER when 14 pilots)."""
    best_score = float("inf")
    best_pair = (1.0, 0.5)
    for tau in (0.15, 0.35, 0.7, 1.2, 2.5, 5.0):
        for lam in (0.20, 0.35, 0.50, 0.65, 0.80):
            _, summ = _run_a1_estimator_on_split(
                ds,
                dev_idx,
                h_oracle_all,
                ref_bank,
                pilot_scs,
                held_out_pilots,
                "content_addr_calibrated",
                tau_ca=tau,
                lam_ca=lam,
            )
            # Strictly blind objective on Dev: held-out pilot MSE if sparse, else Dev bit NLL
            obj = summ["ho_pilot_mse"] if held_out_pilots is not None else summ["nll"]
            if obj < best_score:
                best_score = obj
                best_pair = (tau, lam)
    return best_pair


def run_experiment_a2_theory_verification(
    ds: RealOFDMDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_blocks: List[dict],
    pilot_scs: np.ndarray = PILOT_SCS_7,
    m_pool: int = 16,
    k_local: int = 4,
    seed: int = 20261121,
) -> dict:
    """Direct empirical verification of `content_addressing_scalar` Propositions A & B, Margin U, & Eq. (9).

    On each real Test TTI t:
      - Common baseline m(P_t): LS-linear channel from sparse pilot_scs on sym 2
      - Target real channel H_target_t: empirical channel averaged over odd data symbols {1,3,5,7,9,11,13}
      - Valid real reference H_valid_t: empirical channel averaged over even data symbols {0,4,6,8,10,12}
        (same physical real OTA channel, disjoint thermal noise samples!)
      - Target innovation theta_t = H_target_t - m(P_t)  (evaluated on held-out pilot subcarriers for exact scalar/vector risk!)
      - Valid reference innovation V_J = H_valid_t - m(P_t)
      - Distractors V_j (j != J): real Train-split innovations randomly drawn from disjoint Train TTIs,
        phase-randomized and RMS-matched to E||V_J||^2 so the equal-marginal condition u = v_0 + sigma^2 holds!
      - Evaluates BOTH:
          (i)  Independent fingerprint scoring T (independent subcarrier/pilot subspace match -> satisfies Prop B Eq. 2 & Margin Bound Eq. 3)
          (ii) Shared noisy observation scoring (scoring with the exact same noisy V_j used in estimation -> exposes §5.1 Eq. 9 correlation bias!)
    """
    rng = np.random.default_rng(seed)
    n_all = len(ds.sinr_db)
    # Reconstruct full TX grid offline for split-symbol channel estimation (EVALUATION ONLY)
    x_grid = np.zeros((n_all, N_SYM, N_FFT), dtype=np.complex128)
    from ofdm_data import DATA_COORDS_128, PILOT_TX_14

    x_grid[:, DATA_COORDS_128[:, 0], DATA_COORDS_128[:, 1]] = ds.tx_syms
    x_grid[:, PILOT_SYM, PILOT_SCS_14] = PILOT_TX_14[None, :]
    y_used = ds.iq[:, :, USED_SCS]
    x_used = x_grid[:, :, USED_SCS]

    odd_syms = [1, 3, 5, 7, 9, 11, 13]
    even_syms = [0, 4, 6, 8, 10, 12]

    def _sym_channel(idxs, syms):
        ys = y_used[idxs][:, syms, :]
        xs = x_used[idxs][:, syms, :]
        return np.sum(np.conj(xs) * ys, axis=1) / np.maximum(np.sum(np.abs(xs) ** 2, axis=1), 1e-12)

    h_odd_test = _sym_channel(test_idx, odd_syms)    # [240, 101]
    h_even_test = _sym_channel(test_idx, even_syms)  # [240, 101]
    h_even_train = _sym_channel(train_idx, even_syms)  # [340, 101]

    # Focus on the 7 held-out pilot subcarriers for exact innovation estimation,
    # and use the OTHER 87 data subcarriers (disjoint frequency set!) as independent fingerprint features T!
    eval_u_idx = np.searchsorted(USED_SCS, HELD_OUT_PILOTS_FROM_7)  # 7 subcarriers
    fp_u_idx = np.array([i for i in range(len(USED_SCS)) if i not in set(eval_u_idx.tolist())], dtype=np.int64)

    # Compute common pilot baseline m(P_t) from sparse 7 pilots on sym 2
    m_test = np.stack([ls_linear_channel(ds.iq[t], pilot_scs) for t in test_idx], axis=0)
    m_train = np.stack([ls_linear_channel(ds.iq[t], pilot_scs) for t in train_idx], axis=0)

    # Normalize each TTI by its pilot channel RMS so all TTIs live on a dimensionless unit-channel scale
    scale_test = np.sqrt(np.maximum(np.mean(np.abs(m_test) ** 2, axis=1, keepdims=True), 1e-12))
    scale_train = np.sqrt(np.maximum(np.mean(np.abs(m_train) ** 2, axis=1, keepdims=True), 1e-12))

    # Innovations on eval subcarriers [N, 7] and fingerprint subcarriers [N, 94]
    theta_test = (h_odd_test[:, eval_u_idx] - m_test[:, eval_u_idx]) / scale_test
    v_valid_test = (h_even_test[:, eval_u_idx] - m_test[:, eval_u_idx]) / scale_test
    v_train_pool = (h_even_train[:, eval_u_idx] - m_train[:, eval_u_idx]) / scale_train

    # Independent fingerprint vectors on disjoint subcarriers (zero overlap with eval_u_idx!)
    # Plus controlled independent binary/subspace noisy query observation
    q_fp_test = (h_odd_test[:, fp_u_idx] - m_test[:, fp_u_idx]) / scale_test
    k_valid_fp = (h_even_test[:, fp_u_idx] - m_test[:, fp_u_idx]) / scale_test
    k_train_fp = (h_even_train[:, fp_u_idx] - m_train[:, fp_u_idx]) / scale_train

    # Empirical variance decomposition parameters (v0, sigma2, u, lambda, v1, D)
    # Note: theta and V_valid share real innovation + independent thermal noise on disjoint symbols
    v0 = float(np.mean(np.abs(theta_test) ** 2))
    u_valid = float(np.mean(np.abs(v_valid_test) ** 2))
    cov_tv = float(np.real(np.mean(np.conj(theta_test) * v_valid_test)))
    # Optimal linear shrinkage lambda = Re(E[theta* V_J]) / E[|V_J|^2]
    lam_opt = max(min(cov_tv / max(u_valid, 1e-12), 1.0), 0.0)
    xi_oracle = theta_test - lam_opt * v_valid_test
    v1 = float(np.mean(np.abs(xi_oracle) ** 2))
    D_gain = v0 - v1
    L_k = v0 - D_gain * (float(k_local) / float(m_pool))

    # Equal-marginal distractor scaling factor so E|V_distractor|^2 == u_valid
    u_tr_pool = float(np.mean(np.abs(v_train_pool) ** 2))
    dist_scale = math.sqrt(u_valid / max(u_tr_pool, 1e-12))

    n_test = len(test_idx)
    j_true = rng.integers(0, m_pool, size=n_test)

    # Build M=16 pools for every test TTI:
    #   V_pool: [N_test, M, 7]
    #   s_indep: [N_test, M] scores computed strictly from disjoint fingerprint subcarriers (zero overlap with eval!)
    #   s_shared: [N_test, M] scores computed by directly correlating V_pool with noisy target-side probe (shared obs!)
    v_pool = np.empty((n_test, m_pool, len(eval_u_idx)), dtype=np.complex128)
    s_indep = np.empty((n_test, m_pool), dtype=np.float64)
    s_shared = np.empty((n_test, m_pool), dtype=np.float64)

    # Also build exact independent binary-fingerprint channel (d=32, rho=0.12) on the REAL OTA innovation values
    # to verify the EXACT theoretical Proposition B identity with zero residual cross-subcarrier correlation!
    d_fp = 32
    rho_fp = 0.12
    beta_fp = 0.5 * d_fp * math.log((1.0 - rho_fp) / rho_fp)
    s_bin_indep = np.empty((n_test, m_pool), dtype=np.float64)

    for i in range(n_test):
        jt = int(j_true[i])
        d_choices = rng.choice(len(train_idx), size=m_pool, replace=False)
        rand_phases = np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, size=m_pool))
        # Distractors: Train real innovations with random U(1) phase & equal marginal variance u_valid
        v_cand = v_train_pool[d_choices] * dist_scale * rand_phases[:, None]
        k_cand = k_train_fp[d_choices] * rand_phases[:, None]
        # Insert real valid reference at hidden index J
        v_cand[jt] = v_valid_test[i]
        k_cand[jt] = k_valid_fp[i]
        v_pool[i] = v_cand

        # Score 1: Real disjoint-subcarrier content cosine similarity (real RF content matching!)
        qn = q_fp_test[i] / max(float(np.linalg.norm(q_fp_test[i])), 1e-12)
        kn = k_cand / np.maximum(np.linalg.norm(k_cand, axis=1, keepdims=True), 1e-12)
        s_indep[i] = np.real(kn.conj() @ qn)

        # Score 2: Shared-observation scoring (using SAME V_cand directly against theta_test + noise / amplitude bias!)
        # Deliberately couples scoring & value error as in content_addressing_scalar §5.1!
        # Specifically: score by real inner product with a shared noisy observation (V_J + probe) AND magnitude selector
        v_norms = np.mean(np.abs(v_cand) ** 2, axis=1) / max(u_valid, 1e-12)
        # Re-use noisy V_cand against noisy query (0.5*theta + 0.5*V_J) -> direct shared-noise correlation!
        q_shared = 0.5 * theta_test[i] + 0.5 * v_valid_test[i]
        s_shared[i] = np.real(np.sum(np.conj(v_cand) * q_shared[None, :], axis=1)) / max(
            float(np.linalg.norm(q_shared)) * math.sqrt(u_valid * len(eval_u_idx)), 1e-12
        ) - 0.15 * np.abs(v_norms - 1.0)

        # Score 3: Strictly independent binary fingerprint (exact Prop B assumption on real OTA theta & V_J)
        codes = rng.choice([-1.0, 1.0], size=(m_pool, d_fp))
        flips = np.where(rng.random(d_fp) < rho_fp, -1.0, 1.0)
        q_bin = codes[jt] * flips
        s_bin_indep[i] = (codes @ q_bin) / float(d_fp)

    def _eval_scoring_regime(name: str, scores_mat: np.ndarray, tau_base: float, tau_mults: Tuple[float, ...]) -> dict:
        # Local fixed k=4 baseline (oracle-calibrated posterior restricted to S={0..k-1} without re-normalization, Eq. 5!)
        local_res = {}
        sweep = []
        # Compute margin Delta & violation eta at tau_base
        s_j = scores_mat[np.arange(n_test), j_true]
        s_mask = scores_mat.copy()
        s_mask[np.arange(n_test), j_true] = -1e9
        s_other_max = np.max(s_mask, axis=1)
        margins = s_j - s_other_max
        # Choose a clean positive threshold Delta = max(0.05, 25th percentile of positive margins)
        pos_m = margins[margins > 0]
        delta_thr = float(np.percentile(pos_m, 25)) if len(pos_m) > 0 else 0.1
        eta_emp = float(np.mean(margins < delta_thr))

        for mult in tau_mults:
            tau = float(tau_base) * float(mult)
            sc = scores_mat / max(tau, 1e-9)
            sc -= np.max(sc, axis=1, keepdims=True)
            ex = np.exp(sc)
            alpha = ex / np.sum(ex, axis=1, keepdims=True)  # [N, M]

            # Global content-addressed estimate: hat_theta = lam_opt * sum_j alpha_j V_j
            theta_hat = lam_opt * np.sum(alpha[:, :, None] * v_pool, axis=1)  # [N, 7]
            err_per_tti = np.mean(np.abs(theta_test - theta_hat) ** 2, axis=1)  # [N]
            r_emp = float(np.mean(err_per_tti))

            # Proposition B theoretical exact risk: R_pred = v_1 + D * E||e_J - alpha||_2^2
            ej = np.zeros_like(alpha)
            ej[np.arange(n_test), j_true] = 1.0
            l2_sq_per_tti = np.sum((ej - alpha) ** 2, axis=1)
            r_pred = float(v1 + D_gain * np.mean(l2_sq_per_tti))
            alpha_j_mean = float(np.mean(alpha[np.arange(n_test), j_true]))

            # Margin upper bound U = v_1 + 2*D*[(1-eta)*r^2 + eta]
            b_val = (m_pool - 1) * math.exp(-delta_thr / max(tau, 1e-9)) if delta_thr / max(tau, 1e-9) < 500 else 0.0
            r_val = b_val / (1.0 + b_val)
            u_bound = float(v1 + 2.0 * D_gain * ((1.0 - eta_emp) * (r_val ** 2) + eta_emp))

            # Local fixed k=4 estimator using same alpha weights on S={0..k-1} (Eq. 5: un-renormalized!)
            theta_loc = lam_opt * np.sum(alpha[:, :k_local, None] * v_pool[:, :k_local, :], axis=1)
            err_loc_tti = np.mean(np.abs(theta_test - theta_loc) ** 2, axis=1)
            r_loc_emp = float(np.mean(err_loc_tti))

            # Block-level sign-flip test: Global beats Local-k?
            b_diffs = []
            for blk in test_blocks:
                bi = [x - 502 for x in blk["tti_indices"]]
                b_diffs.append(float(np.mean(err_loc_tti[bi] - err_per_tti[bi])))
            sf = exact_sign_flip_test(np.array(b_diffs))

            sweep.append(
                {
                    "tau_mult": float(mult),
                    "tau": float(tau),
                    "R_emp": r_emp,
                    "R_propB_pred": r_pred,
                    "Delta_corr_emp_minus_pred": float(r_emp - r_pred),
                    "rel_discrepancy_pct": float(100.0 * (r_emp - r_pred) / max(r_pred, 1e-12)),
                    "U_margin_bound": u_bound,
                    "L_k_lower_bound": float(L_k),
                    "R_local_k4_emp": r_loc_emp,
                    "alpha_J_mean": alpha_j_mean,
                    "E_l2_sq": float(np.mean(l2_sq_per_tti)),
                    "global_vs_local_gain": float(r_loc_emp - r_emp),
                    "global_vs_local_signflip": sf,
                }
            )

        # Also evaluate global uniform baseline (tau -> inf) and hard argmax baseline (tau -> 0)
        theta_unif = lam_opt * np.mean(v_pool, axis=1)
        r_unif = float(np.mean(np.abs(theta_test - theta_unif) ** 2))
        hard_idx = np.argmax(scores_mat, axis=1)
        theta_hard = lam_opt * v_pool[np.arange(n_test), hard_idx]
        r_hard = float(np.mean(np.abs(theta_test - theta_hard) ** 2))
        hit_rate = float(np.mean(hard_idx == j_true))

        return {
            "regime": name,
            "margin_delta_threshold": delta_thr,
            "margin_median": float(np.median(margins)),
            "margin_mean": float(np.mean(margins)),
            "eta_violation_prob": eta_emp,
            "top1_hit_rate": hit_rate,
            "R_global_uniform_emp": r_unif,
            "R_global_uniform_theory": float(v0 - D_gain / float(m_pool)),
            "R_hard_select_emp": r_hard,
            "temperature_sweep": sweep,
        }

    # Calibrate natural base temperature for disjoint-subcarrier score via variance scaling
    tau_indep_base = 0.08
    tau_shared_base = 0.12
    tau_bin_base = 1.0 / beta_fp

    return {
        "pool_size_M": m_pool,
        "local_window_k": k_local,
        "n_test_ttis": n_test,
        "empirical_model_params": {
            "v0_prior_innovation_var": v0,
            "u_valid_ref_var": u_valid,
            "cov_theta_V": cov_tv,
            "lambda_opt": lam_opt,
            "v1_oracle_residual_var": v1,
            "D_eliminable_var": D_gain,
            "L_k_propA_lower_bound": L_k,
        },
        "regimes": {
            "A2_bin_indep_exact": _eval_scoring_regime(
                "A2_bin_indep_exact (strict T perp (theta,eps)|J on real OTA innovations)",
                s_bin_indep,
                tau_bin_base,
                (0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 25.0),
            ),
            "A2_rf_disjoint_subcarriers": _eval_scoring_regime(
                "A2_rf_disjoint_subcarriers (real RF content match on disjoint 94 subcarriers)",
                s_indep,
                tau_indep_base,
                (0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 25.0),
            ),
            "A2_shared_obs_eq9_bias": _eval_scoring_regime(
                "A2_shared_obs_eq9_bias (shared noisy obs scoring -> §5.1 Eq. 9 bias test)",
                s_shared,
                tau_shared_base,
                (0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 25.0),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    t0 = time.perf_counter()
    ds = load_real_ofdm_dataset(args.pth)
    splits = build_temporal_splits(len(ds.sinr_db), ds.sinr_db)
    train_idx = np.array(splits["train"]["tti_indices"], dtype=np.int64)
    dev_idx = np.array(splits["dev"]["tti_indices"], dtype=np.int64)
    test_idx = np.array(splits["test"]["tti_indices"], dtype=np.int64)
    test_blocks = splits["test"]["blocks"]

    h_oracle_all = build_oracle_channel(ds)
    ref_bank = build_train_reference_bank(ds.iq[train_idx], ds.sinr_db[train_idx], m_refs=16)

    # Experiment A1 across 14, 7, 4 pilots
    pilot_configs = [
        ("14_pilots_full", PILOT_SCS_14, None),
        ("7_pilots_sparse", PILOT_SCS_7, HELD_OUT_PILOTS_FROM_7),
        ("4_pilots_ultrasparse", PILOT_SCS_4, HELD_OUT_PILOTS_FROM_4),
    ]
    a1_methods = [
        "single_tti_ls_linear",
        "single_tti_ls_spline",
        "local_k4_uniform",
        "global_m16_uniform",
        "classical_invdist_weight",
        "content_addr_calibrated",
        "content_addr_cold_hard",
        "content_addr_hot",
    ]

    a1_results = {}
    per_tti_csv_rows = []
    for cname, p_scs, ho_scs in pilot_configs:
        tau_star, lam_star = calibrate_a1_on_dev(ds, dev_idx, h_oracle_all, ref_bank, p_scs, ho_scs)
        cfg_res = {
            "n_pilots": int(len(p_scs)),
            "pilot_scs": p_scs.tolist(),
            "held_out_pilots": ho_scs.tolist() if ho_scs is not None else None,
            "dev_calibrated_tau": tau_star,
            "dev_calibrated_lambda": lam_star,
            "methods": {},
            "block_tests_vs_baselines": {},
        }
        method_rows = {}
        for m in a1_methods:
            rows, summ = _run_a1_estimator_on_split(
                ds,
                test_idx,
                h_oracle_all,
                ref_bank,
                p_scs,
                ho_scs,
                m,
                tau_ca=tau_star,
                lam_ca=lam_star,
                k_local=4,
            )
            cfg_res["methods"][m] = summ
            method_rows[m] = rows
            for r in rows:
                per_tti_csv_rows.append({"config": cname, "method": m, **r})

        # Exact 12-block sign-flip tests comparing content_addr_calibrated vs key baselines
        ca_rows = method_rows["content_addr_calibrated"]
        for base_m in ("single_tti_ls_linear", "local_k4_uniform", "global_m16_uniform", "content_addr_cold_hard"):
            b_rows = method_rows[base_m]
            for metric in ("ch_nmse", "ber", "nll") + (("ho_pilot_nmse",) if ho_scs is not None else ()):
                b_diffs = []
                for blk in test_blocks:
                    bi = [x - 502 for x in blk["tti_indices"]]
                    diff_blk = float(np.mean([b_rows[j][metric] - ca_rows[j][metric] for j in bi]))
                    b_diffs.append(diff_blk)
                cfg_res["block_tests_vs_baselines"][f"ca_vs_{base_m}_{metric}"] = exact_sign_flip_test(
                    np.array(b_diffs)
                )
        a1_results[cname] = cfg_res

    # Experiment A2: Direct Proposition A / B / Margin U / Eq. 9 verification
    a2_results = run_experiment_a2_theory_verification(
        ds, train_idx, test_idx, test_blocks, pilot_scs=PILOT_SCS_7, m_pool=16, k_local=4
    )

    out = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "dataset_sha256": ds.sha256,
        "experiment_A1_sparse_pilot_historical_bank": a1_results,
        "experiment_A2_propAB_and_eq9_verification": a2_results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, args.out_dir / "group_a_results.json")

    # Save per-TTI CSV
    csv_path = args.out_dir / "group_a_per_tti.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tti_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_tti_csv_rows)

    print(f"Group A complete in {out['elapsed_s']}s. Summary:")
    for cname, cfg in a1_results.items():
        print(f"\n[{cname}] dev tau*={cfg['dev_calibrated_tau']}, lam*={cfg['dev_calibrated_lambda']}")
        for m, s in cfg["methods"].items():
            ho_s = f"ho_nmse={s['ho_pilot_nmse']:.4f}" if s["ho_pilot_nmse"] is not None else "ho_nmse=N/A   "
            print(f"  {m:26s}: ch_nmse={s['ch_nmse']:.4f} ({s['ch_evm_pct']:5.2f}%) | {ho_s} | BER={s['ber']:.5f} | NLL={s['nll']:.4f}")

    p = a2_results["empirical_model_params"]
    print(f"\n[A2 Empirical Params] v0={p['v0_prior_innovation_var']:.5f}, v1={p['v1_oracle_residual_var']:.5f}, D={p['D_eliminable_var']:.5f}, L_k(k=4)={p['L_k_propA_lower_bound']:.5f}")
    for rname, rdata in a2_results["regimes"].items():
        print(f"  Regime {rname}: top1_hit={rdata['top1_hit_rate']:.3f}, eta={rdata['eta_violation_prob']:.3f}, unif_emp={rdata['R_global_uniform_emp']:.5f}, hard_emp={rdata['R_hard_select_emp']:.5f}")
        for pt in rdata["temperature_sweep"]:
            if pt["tau_mult"] in (0.05, 1.0, 25.0):
                print(
                    f"    mult={pt['tau_mult']:5.2f}: R_emp={pt['R_emp']:.5f}, R_pred={pt['R_propB_pred']:.5f}, "
                    f"diff={pt['Delta_corr_emp_minus_pred']:+.5f} ({pt['rel_discrepancy_pct']:+.1f}%), "
                    f"U={pt['U_margin_bound']:.5f}, L_loc_emp={pt['R_local_k4_emp']:.5f}, alpha_J={pt['alpha_J_mean']:.3f}"
                )


if __name__ == "__main__":
    main()
