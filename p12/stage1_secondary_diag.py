"""Phase 12 Stage 1B: Secondary-Axis Off-Grid Causal Diagnostics Runner (32 fresh links/regime, 9 dB).

Executes the full first-principles off-grid diagnostic suite strictly on SEED_P12_DIAG:
  1. Explicit separation of P11's `offgrid_full_reset` (preserves delays via delays_override)
     vs `offgrid_full_reset_new_delays` (true full reset redrawing delays independently),
     alongside all other ID + OOD regimes (7 regimes x 32 links = 224 paired links).
  2. Complete Off-Grid Oracle Ladder on every link:
     - `p11_classical_per_ray` & `p11_transport_attn_ens` (3-seed frozen P11 checkpoints)
     - `step1_oracle_best_box_mix`: exact per-row box-constrained QP `min_{0<=w_k<=1} ||H* - H_mix||_F^2`
     - `step2a_trueP_est_delays_est_traj`: oracle ray count P only (isolates K_hat=5 model order)
     - `step2b_true_delays_est_traj`: oracle delays tau* (blind A+B Doppler/chirp)
     - `step2c_est_K5_delays_true_traj`: blind K_hat=5 delays + matched true B phase trajectories (anchor gain fit)
     - `step2d_true_delays_true_traj_anchor_gain`: oracle delays + true B trajectories + 1-anchor gain fit
     - `oracle_true_channel` (+ `oracle_true_channel_true_noise` on burst regime)
  3. Locked-comb `(0,9,10,19)` full 32x32 continuous-delay AUX covariance x roving bilateral 2-fold
     wavefront B-refit matrix (`cov_only`, `wf_blind_no_cov`, `cov_plus_wf_blind`,
     `cov_plus_wf_true_on_acc`, `cov_plus_wf_true_all`).
  4. Paired 2x2 Frame-A Pilot Geometry Intervention on 4 ID regimes:
     locked comb `(0,9,10,19)` vs endpoint-preserving Chebyshev Pareto optimum `(0,4,6,19)`
     crossed with `B-refit off` vs `cov32 + wf_blind`.
  5. Identifiability & Parallel-Transport Connection Audits:
     - Projected-Jacobian Schur-complement Fisher `J_eff` (anchor vs +5 AUX cov vs +2 bilateral known)
     - Global wrong-lobe profiled KL divergence & Pinsker lower bound (`(0,9,10,19)` vs `(0,4,6,19)`)
     - Reverse-edge unitarity audit: P11 query-only phase vs P12 symmetric connection `P_ji = P_ij*`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import numpy as np

from p8.coding import qam_llrs
from p8c.link import PILOTS as SC_PILOTS
from p9.decoder import warmup_numba
from p11.contract import COMPLEX_DTYPE, N, PRIMARY_SNR_DB, PROBE_ROW, REAL_DTYPE, T, enforce_determinism
from p11.freeze import CKPT_DIR, OFFGRID_SEEDS
from p11.offgrid_channel import (
    BURST_NOISE_MULT,
    K_HAT,
    _EYE_N,
    _TIME_B_FULL,
    build_offgrid_branches,
    classical_per_ray_gate,
    compose_offgrid_gate_output,
    discover_frame_a_atoms,
)
from p12.contract import (
    OFDM_PILOT_COMB,
    OFDM_PILOT_PARETO,
    OFFGRID_REGIME_TRUE_RESET,
    OFFGRID_STAGE1_REGIMES,
    SEED_P12_DIAG,
    assert_p12_seed_disjoint,
)
from p12.offgrid_physics import (
    audit_parallel_transport_connection,
    build_offgrid_branches_geom,
    compute_offgrid_fisher_schur_diag,
    compute_wrong_lobe_profile_kl,
    discover_frame_a_atoms_geom,
    fit_known_trajectory_from_blocks,
    run_offgrid_cov32_and_wavefront_refit,
    score_offgrid_channel_est,
    simulate_offgrid_link_p12,
    solve_oracle_best_box_mixture,
)

SECONDARY_DIAG_SEED_OFFSET = 500_000
PILOT_2X2_REGIMES = (
    "offgrid_continuous",
    "offgrid_partial_shift",
    "offgrid_midframe_shift",
    "offgrid_full_reset",
)
FISHER_AUDIT_REGIMES = ("offgrid_continuous", OFFGRID_REGIME_TRUE_RESET)

_TRANSPORT_MODELS: Optional[list[Any]] = None


def _get_transport_models() -> list[Any]:
    global _TRANSPORT_MODELS
    if _TRANSPORT_MODELS is None:
        import torch
        from p11.manifold_gate import ManifoldTrustGate

        models: list[Any] = []
        for s in OFFGRID_SEEDS:
            ckpt = CKPT_DIR / f"em10_transport_attn_seed{s}.pt"
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
            m = ManifoldTrustGate(variant="em10_transport_attn", **payload["arch"])
            m.load_state_dict(payload["state_dict"], strict=True)
            m.eval()
            models.append(m)
        _TRANSPORT_MODELS = models
    return _TRANSPORT_MODELS


def _worker_init(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_numba()
    _get_transport_models()


def _run_transport_ensemble(obs, bundle, truth) -> dict[str, Any]:
    import torch

    models = _get_transport_models()
    feats = models[0].extract_features_for_variant(obs, bundle)
    s_t = torch.from_numpy(feats.scalar_feats).to(torch.float64).unsqueeze(0)
    c_t = torch.from_numpy(feats.complex_feats).to(torch.complex128).unsqueeze(0)
    tau_t = torch.from_numpy(feats.tau_hat).to(torch.float64).unsqueeze(0)
    nu_t = torch.from_numpy(feats.nu_hat).to(torch.float64).unsqueeze(0)
    tp_t = torch.from_numpy(feats.t_phys).to(torch.float64).unsqueeze(0)
    tn_t = torch.from_numpy(feats.t_norm).to(torch.float64).unsqueeze(0)
    ev_t = torch.from_numpy(feats.ev_base).to(torch.float64).unsqueeze(0)
    bl_t = torch.from_numpy(feats.base_logit).to(torch.float64).unsqueeze(0)
    pr_t = torch.from_numpy(feats.is_probe).to(torch.float64).unsqueeze(0)

    logits_list = []
    nm_list = []
    with torch.no_grad():
        for m in models:
            _, logit_t, nm_t = m.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
            logits_list.append(logit_t.squeeze(0).cpu().numpy())
            nm_list.append(nm_t.squeeze(0).cpu().numpy())

    mean_logit = np.mean(logits_list, axis=0)
    w_ens = 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40.0, 40.0)))
    nm_ens = np.exp(np.mean(np.log(np.maximum(nm_list, 1e-12)), axis=0))
    ens_res = compose_offgrid_gate_output(w_ens, nm_ens, obs, bundle)
    return score_offgrid_channel_est(
        ens_res.h_mix,
        ens_res.noise_mix,
        ens_res.gate_out.mix_llrs,
        obs,
        truth,
        probe_pred=ens_res.gate_out.mix_probe_pred,
    )


def _eval_oracle_true_channel(obs, truth, *, use_true_burst_noise: bool = False) -> dict[str, Any]:
    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j
    drows = list(truth.data_rows)
    n0 = float(obs.n0)
    nm = np.ones(T, dtype=REAL_DTYPE)
    if use_true_burst_noise and truth.corrupted_rows:
        for r_c in truth.corrupted_rows:
            nm[int(r_c)] = float(BURST_NOISE_MULT)
    noise_mat = (n0 * nm)[:, None] * np.ones((1, N), dtype=REAL_DTYPE)

    s_mag = math.sqrt(n0)
    h_d = truth.h_b_true_mat[drows] / s_mag
    nv_d = noise_mat[drows] / n0
    y_d = y_b_safe[drows] / s_mag
    adj = h_d.conj().swapaxes(-1, -2) / nv_d[:, None, :]
    cov = np.linalg.inv(_EYE_N[None, :, :] + adj @ h_d)
    est = (cov @ (adj @ y_d[:, :, None]))[..., 0]
    u_var = np.diagonal(cov, axis1=-2, axis2=-1).real
    o_llrs = np.zeros((T, N, 4), dtype=REAL_DTYPE)
    o_llrs[drows] = qam_llrs(est.ravel(), u_var.ravel()).reshape(len(drows), N, 4)
    p_pred = truth.h_b_true_mat[obs.probe_row] @ SC_PILOTS[obs.probe_row]
    return score_offgrid_channel_est(
        truth.h_b_true_mat, noise_mat, o_llrs, obs, truth, probe_pred=p_pred
    )


def _unit_phase_trajectories(g_b_true: np.ndarray, r_anchor: int) -> np.ndarray:
    """Normalize `g_b_true [P, T, N]` by its value at anchor midpoint so |u_p(t)| == 1 exactly."""
    g_anc_mid = g_b_true[:, r_anchor, N // 2]
    mags = np.maximum(np.abs(g_anc_mid), 1e-30)
    u = g_b_true / g_anc_mid[:, None, None]
    # Normalize each sample to exact unit modulus (numerical hygiene)
    u_mag = np.maximum(np.abs(u), 1e-30)
    return (u / u_mag).astype(COMPLEX_DTYPE)


def _eval_one_secondary_link(args: tuple[str, int, int, float, bool]) -> dict[str, Any]:
    regime, link_idx, seed, snr_db, run_ident_audit = args
    t0 = time.perf_counter()

    obs, truth, extra = simulate_offgrid_link_p12(seed, regime, snr_db=snr_db, pilot_rows_a=OFDM_PILOT_COMB)
    r_anchor = int(extra["anchor_row"])
    n0 = float(obs.n0)
    y_b_safe = np.array(obs.y_b, dtype=COMPLEX_DTYPE, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j

    # --- 1) P11 Baselines (classical_per_ray + frozen 3-seed em10_transport_attn) ---
    atoms = discover_frame_a_atoms(obs, k_hat=K_HAT)
    bundle = build_offgrid_branches(obs, atoms=atoms, k_hat=K_HAT)

    cls_res = classical_per_ray_gate(obs, bundle)
    sc_cls = score_offgrid_channel_est(
        cls_res.h_mix, cls_res.noise_mix, cls_res.gate_out.mix_llrs, obs, truth, probe_pred=cls_res.gate_out.mix_probe_pred
    )
    sc_attn = _run_transport_ensemble(obs, bundle, truth)

    # --- 2) Oracle Ladder ---
    # Step 1: exact box-constrained QP mixture inside existing representation
    _, sc_box = solve_oracle_best_box_mixture(obs, bundle, truth)

    # Step 2a: true ray count P only, blind delays & blind trajectory
    atoms_p, t_a_p = discover_frame_a_atoms_geom(obs, pilot_rows_a=OFDM_PILOT_COMB, k_hat=truth.n_rays)
    bundle_p = build_offgrid_branches_geom(obs, atoms_p, t_a_mean=t_a_p)
    _, sc_2a_box = solve_oracle_best_box_mixture(obs, bundle_p, truth)
    cls_2a = classical_per_ray_gate(obs, bundle_p)
    sc_2a_cls = score_offgrid_channel_est(
        cls_2a.h_mix, cls_2a.noise_mix, cls_2a.gate_out.mix_llrs, obs, truth, probe_pred=cls_2a.gate_out.mix_probe_pred
    )

    # Step 2b: true delays tau* (truth.delays = delays_b), blind A+B Doppler/chirp trajectory
    atoms_td, t_a_td = discover_frame_a_atoms_geom(
        obs, pilot_rows_a=OFDM_PILOT_COMB, k_hat=truth.n_rays, fixed_delays=truth.delays
    )
    bundle_td = build_offgrid_branches_geom(obs, atoms_td, t_a_mean=t_a_td)
    _, sc_2b_box = solve_oracle_best_box_mixture(obs, bundle_td, truth)
    cls_2b = classical_per_ray_gate(obs, bundle_td)
    sc_2b_cls = score_offgrid_channel_est(
        cls_2b.h_mix, cls_2b.noise_mix, cls_2b.gate_out.mix_llrs, obs, truth, probe_pred=cls_2b.gate_out.mix_probe_pred
    )

    # True unit phase trajectories on Frame B
    u_true = _unit_phase_trajectories(extra["g_b_true"], r_anchor)  # [P, T, N]
    prior_ray_var = max(float(atoms.p_tot_a) / float(max(truth.n_rays, 1)), 0.1 * n0)

    # Step 2c: estimated K_hat=5 delays + matched true B phase trajectories, anchor-only LMMSE gain fit
    u_est_matched = np.empty((bundle.k_hat, T, N), dtype=COMPLEX_DTYPE)
    for k_i in range(bundle.k_hat):
        nearest_p = int(np.argmin(np.abs(truth.delays - bundle.tau_hat[k_i])))
        u_est_matched[k_i] = u_true[nearest_p]
    sc_2c = fit_known_trajectory_from_blocks(
        bundle.tau_hat,
        u_est_matched,
        (r_anchor,),
        obs.pilot_tx_b[:1],
        y_b_safe[r_anchor : r_anchor + 1],
        np.array([n0], dtype=REAL_DTYPE),
        prior_ray_var,
        obs,
        truth,
    )

    # Step 2d: true delays + true B phase trajectories, 1-anchor LMMSE gain fit
    sc_2d = fit_known_trajectory_from_blocks(
        truth.delays,
        u_true,
        (r_anchor,),
        obs.pilot_tx_b[:1],
        y_b_safe[r_anchor : r_anchor + 1],
        np.array([n0], dtype=REAL_DTYPE),
        prior_ray_var,
        obs,
        truth,
    )

    # Oracle true channel reference (+ true noise variance on burst)
    sc_oracle = _eval_oracle_true_channel(obs, truth, use_true_burst_noise=False)
    sc_oracle_tn = (
        _eval_oracle_true_channel(obs, truth, use_true_burst_noise=True)
        if regime == "offgrid_ood_burst_and_shift"
        else None
    )

    # --- 3) Locked-Comb Full 32x32 AUX Covariance x Roving Bilateral Wavefront B-Refit ---
    brefit_cov_only = run_offgrid_cov32_and_wavefront_refit(
        obs, truth, extra, pilot_rows_a=OFDM_PILOT_COMB, use_aux_cov=True, use_wavefront_refit=False
    )
    brefit_wf_only = run_offgrid_cov32_and_wavefront_refit(
        obs, truth, extra, pilot_rows_a=OFDM_PILOT_COMB, use_aux_cov=False, use_wavefront_refit=True, helper_mode="blind_decoded"
    )
    brefit_cov_wf_blind = run_offgrid_cov32_and_wavefront_refit(
        obs, truth, extra, pilot_rows_a=OFDM_PILOT_COMB, use_aux_cov=True, use_wavefront_refit=True, helper_mode="blind_decoded"
    )
    brefit_cov_wf_true_acc = run_offgrid_cov32_and_wavefront_refit(
        obs, truth, extra, pilot_rows_a=OFDM_PILOT_COMB, use_aux_cov=True, use_wavefront_refit=True, helper_mode="true_on_accepted"
    )
    brefit_cov_wf_true_all = run_offgrid_cov32_and_wavefront_refit(
        obs, truth, extra, pilot_rows_a=OFDM_PILOT_COMB, use_aux_cov=True, use_wavefront_refit=True, helper_mode="true_words_all"
    )

    # --- 4) Paired 2x2 Frame-A Pilot Geometry Intervention: (0,9,10,19) vs (0,4,6,19) ---
    pilot_pareto_diag: Any = None
    if regime in PILOT_2X2_REGIMES:
        obs_p, truth_p, extra_p = simulate_offgrid_link_p12(
            seed, regime, snr_db=snr_db, pilot_rows_a=OFDM_PILOT_PARETO
        )
        p_off = run_offgrid_cov32_and_wavefront_refit(
            obs_p, truth_p, extra_p, pilot_rows_a=OFDM_PILOT_PARETO, use_aux_cov=False, use_wavefront_refit=False
        )
        p_on = run_offgrid_cov32_and_wavefront_refit(
            obs_p, truth_p, extra_p, pilot_rows_a=OFDM_PILOT_PARETO, use_aux_cov=True, use_wavefront_refit=True, helper_mode="blind_decoded"
        )
        pilot_pareto_diag = {
            "pareto_refit_off": p_off,
            "pareto_cov_wf_blind": p_on,
        }

    # --- 5) Identifiability (Fisher J_eff + Wrong-Lobe KL) & Parallel-Transport Connection Audit ---
    ident_audit: Any = None
    if run_ident_audit:
        t_anc_mid = float(_TIME_B_FULL[r_anchor].mean())
        g_b_anc = extra["g_b_true"][:, r_anchor, N // 2]
        # True instantaneous freq at anchor midpoint
        f_b_anc = truth.freqs_b + truth.chirps * (t_anc_mid - (extra["t_shift"] or 0.0))
        b_sym_map = {int(r): extra["b_symbols"][i] for i, r in enumerate(truth.data_rows)}
        fisher = compute_offgrid_fisher_schur_diag(
            truth.delays, g_b_anc, f_b_anc, truth.chirps, r_anchor, n0, b_sym_map
        )
        kl_comb = compute_wrong_lobe_profile_kl(truth.freqs_a, truth.chirps, truth.powers, n0, OFDM_PILOT_COMB)
        kl_pareto = compute_wrong_lobe_profile_kl(truth.freqs_a, truth.chirps, truth.powers, n0, OFDM_PILOT_PARETO)
        conn_audit = audit_parallel_transport_connection(bundle.tau_hat, bundle.nu_hat)
        ident_audit = {
            "fisher_schur": fisher,
            "wrong_lobe_kl_comb": kl_comb,
            "wrong_lobe_kl_pareto": kl_pareto,
            "connection_unitarity": conn_audit,
        }

    # Delay estimation accuracy diagnostic (mean nearest-atom error to true delays)
    delay_match_err = float(np.mean([np.min(np.abs(bundle.tau_hat - td)) for td in truth.delays]))

    return {
        "regime": regime,
        "link_idx": int(link_idx),
        "seed": int(seed),
        "n_rays": int(truth.n_rays),
        "anchor_row": int(r_anchor),
        "delay_match_err": delay_match_err,
        "ladder": {
            "p11_classical_per_ray": sc_cls,
            "p11_transport_attn_ens": sc_attn,
            "step1_oracle_best_box_mix": sc_box,
            "step2a_trueP_cls": sc_2a_cls,
            "step2a_trueP_box": sc_2a_box,
            "step2b_true_delays_cls": sc_2b_cls,
            "step2b_true_delays_box": sc_2b_box,
            "step2c_est_K5_true_traj": sc_2c,
            "step2d_true_delays_true_traj_1anc": sc_2d,
            "oracle_true_channel": sc_oracle,
            "oracle_true_channel_true_noise": sc_oracle_tn,
        },
        "brefit_comb": {
            "cov_only": brefit_cov_only,
            "wf_blind_no_cov": brefit_wf_only,
            "cov_plus_wf_blind": brefit_cov_wf_blind,
            "cov_plus_wf_true_on_acc": brefit_cov_wf_true_acc,
            "cov_plus_wf_true_all": brefit_cov_wf_true_all,
        },
        "pilot_pareto": pilot_pareto_diag,
        "ident_audit": ident_audit,
        "wall_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def aggregate_secondary_diag(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_regime.setdefault(r["regime"], []).append(r)

    def _agg_arm(recs: Sequence[dict[str, Any]], path: tuple[str, ...]) -> Optional[dict[str, Any]]:
        vals = []
        for rec in recs:
            cur: Any = rec
            for k in path:
                if cur is None or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if cur is not None:
                vals.append(cur)
        if not vals:
            return None
        tot_e = sum(v["block_errors"] for v in vals)
        tot_b = sum(v["blocks"] for v in vals)
        out: dict[str, Any] = {
            "bler_pct": 100.0 * tot_e / max(tot_b, 1),
            "errors": int(tot_e),
            "blocks": int(tot_b),
            "mean_rel_frob": float(np.mean([v["mean_rel_frob_err"] for v in vals])),
        }
        p_vals = [v["probe_clean_nmse"] for v in vals if v.get("probe_clean_nmse", -1.0) >= 0.0]
        if p_vals:
            out["mean_probe_clean_nmse"] = float(np.mean(p_vals))
        if "wrong_lobe_pre" in vals[0]:
            out["wrong_lobe_pre_pct"] = 100.0 * float(np.mean([float(v["wrong_lobe_pre"]) for v in vals]))
            out["wrong_lobe_post_pct"] = 100.0 * float(np.mean([float(v["wrong_lobe_post"]) for v in vals]))
            out["mean_dom_nu_err_pre"] = float(np.mean([v["dom_nu_err_pre"] for v in vals]))
            out["mean_dom_nu_err_post"] = float(np.mean([v["dom_nu_err_post"] for v in vals]))
            tot_h = sum(v["helpers_accepted"] for v in vals)
            tot_f = sum(v["false_assimilations"] for v in vals)
            out["helpers_accepted_total"] = int(tot_h)
            out["false_assimilations"] = int(tot_f)
            out["false_assimilation_pct"] = 100.0 * tot_f / max(tot_h, 1)
        return out

    ladder_keys = (
        "p11_classical_per_ray",
        "p11_transport_attn_ens",
        "step1_oracle_best_box_mix",
        "step2a_trueP_cls",
        "step2a_trueP_box",
        "step2b_true_delays_cls",
        "step2b_true_delays_box",
        "step2c_est_K5_true_traj",
        "step2d_true_delays_true_traj_1anc",
        "oracle_true_channel",
        "oracle_true_channel_true_noise",
    )
    brefit_keys = (
        "cov_only",
        "wf_blind_no_cov",
        "cov_plus_wf_blind",
        "cov_plus_wf_true_on_acc",
        "cov_plus_wf_true_all",
    )

    per_regime_summary: dict[str, Any] = {}
    for reg in OFFGRID_STAGE1_REGIMES:
        recs = by_regime.get(reg, [])
        if not recs:
            continue
        reg_out: dict[str, Any] = {
            "n_links": len(recs),
            "mean_delay_match_err": float(np.mean([r["delay_match_err"] for r in recs])),
            "ladder": {},
            "brefit_comb": {},
        }
        for lk in ladder_keys:
            res = _agg_arm(recs, ("ladder", lk))
            if res is not None:
                reg_out["ladder"][lk] = res
        for bk in brefit_keys:
            res = _agg_arm(recs, ("brefit_comb", bk))
            if res is not None:
                reg_out["brefit_comb"][bk] = res
        if reg in PILOT_2X2_REGIMES:
            reg_out["pilot_2x2"] = {
                "comb_refit_off": reg_out["ladder"]["p11_classical_per_ray"],
                "comb_cov_wf_blind": reg_out["brefit_comb"]["cov_plus_wf_blind"],
                "pareto_refit_off": _agg_arm(recs, ("pilot_pareto", "pareto_refit_off")),
                "pareto_cov_wf_blind": _agg_arm(recs, ("pilot_pareto", "pareto_cov_wf_blind")),
            }
        per_regime_summary[reg] = reg_out

    # Pooled 4-ID (matching P11's 4 ID regimes) summary
    p11_id_recs = [r for r in records if r["regime"] in PILOT_2X2_REGIMES]
    pooled_4id: dict[str, Any] = {
        "n_links": len(p11_id_recs),
        "ladder": {lk: _agg_arm(p11_id_recs, ("ladder", lk)) for lk in ladder_keys if lk != "oracle_true_channel_true_noise"},
        "brefit_comb": {bk: _agg_arm(p11_id_recs, ("brefit_comb", bk)) for bk in brefit_keys},
        "pilot_2x2": {
            "comb_refit_off": _agg_arm(p11_id_recs, ("ladder", "p11_classical_per_ray")),
            "comb_cov_wf_blind": _agg_arm(p11_id_recs, ("brefit_comb", "cov_plus_wf_blind")),
            "pareto_refit_off": _agg_arm(p11_id_recs, ("pilot_pareto", "pareto_refit_off")),
            "pareto_cov_wf_blind": _agg_arm(p11_id_recs, ("pilot_pareto", "pareto_cov_wf_blind")),
        },
    }

    # Aggregate Identifiability & Connection Audits
    audit_recs = [r for r in records if r["ident_audit"] is not None]
    ident_summary: dict[str, Any] = {"n_links": len(audit_recs)}
    if audit_recs:
        def _med(vals):
            return float(np.median(vals))
        for cfg in ("anchor_only", "anchor_plus_5aux_cov", "anchor_plus_2bilat_known", "anchor_plus_5aux_plus_2bilat"):
            ident_summary[f"fisher_{cfg}"] = {
                "median_min_eig_all": _med([r["ident_audit"]["fisher_schur"][cfg]["min_eig_all"] for r in audit_recs]),
                "median_min_eig_tau": _med([r["ident_audit"]["fisher_schur"][cfg]["min_eig_tau"] for r in audit_recs]),
                "median_min_eig_nu": _med([r["ident_audit"]["fisher_schur"][cfg]["min_eig_nu"] for r in audit_recs]),
                "median_cond_eff": _med([r["ident_audit"]["fisher_schur"][cfg]["cond_eff"] for r in audit_recs]),
            }
        ident_summary["wrong_lobe_kl"] = {
            "comb_median_kl_nats": _med([r["ident_audit"]["wrong_lobe_kl_comb"]["profile_kl_nats"] for r in audit_recs]),
            "comb_mean_pinsker_lb": float(np.mean([r["ident_audit"]["wrong_lobe_kl_comb"]["pinsker_error_lb"] for r in audit_recs])),
            "pareto_median_kl_nats": _med([r["ident_audit"]["wrong_lobe_kl_pareto"]["profile_kl_nats"] for r in audit_recs]),
            "pareto_mean_pinsker_lb": float(np.mean([r["ident_audit"]["wrong_lobe_kl_pareto"]["pinsker_error_lb"] for r in audit_recs])),
        }
        ident_summary["connection_unitarity"] = {
            "p11_max_rev_err": float(np.max([r["ident_audit"]["connection_unitarity"]["p11_query_only_max_rev_unitarity_err"] for r in audit_recs])),
            "p11_mean_rev_err": float(np.mean([r["ident_audit"]["connection_unitarity"]["p11_query_only_max_rev_unitarity_err"] for r in audit_recs])),
            "p12_max_rev_err": float(np.max([r["ident_audit"]["connection_unitarity"]["p12_symmetric_max_rev_unitarity_err"] for r in audit_recs])),
            "p12_max_conj_sym_err": float(np.max([r["ident_audit"]["connection_unitarity"]["p12_symmetric_max_conj_sym_err"] for r in audit_recs])),
        }

    return {
        "n_total_links": len(records),
        "pooled_4id": pooled_4id,
        "by_regime": per_regime_summary,
        "identifiability_and_connection": ident_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 Stage 1B Secondary-Axis Off-Grid Causal Diagnostics")
    parser.add_argument("--n-links-per-regime", type=int, default=32)
    parser.add_argument("--n-ident-links", type=int, default=16)
    parser.add_argument("--snr-db", type=float, default=PRIMARY_SNR_DB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="p12/results")
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    enforce_determinism(require_env=args.require_env)

    # Disjoint seeds across all 7 regimes
    tasks: list[tuple[str, int, int, float, bool]] = []
    all_seeds: list[int] = []
    for r_idx, reg in enumerate(OFFGRID_STAGE1_REGIMES):
        for l_idx in range(int(args.n_links_per_regime)):
            sd = SEED_P12_DIAG + SECONDARY_DIAG_SEED_OFFSET + r_idx * 10_000 + l_idx
            all_seeds.append(sd)
            do_audit = (reg in FISHER_AUDIT_REGIMES) and (l_idx < int(args.n_ident_links))
            tasks.append((reg, l_idx, sd, float(args.snr_db), do_audit))
    assert_p12_seed_disjoint(all_seeds, expected_namespace="p12_diag")

    if args.workers <= 1:
        _worker_init(bool(args.require_env))
        records = [_eval_one_secondary_link(t) for t in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_worker_init,
            initargs=(bool(args.require_env),),
        ) as pool:
            records = list(pool.map(_eval_one_secondary_link, tasks))

    summary = aggregate_secondary_diag(records)
    summary["total_wall_s"] = float(time.perf_counter() - t0)
    summary["workers"] = int(args.workers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage1_secondary_links.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, allow_nan=False) + "\n")
    with open(out_dir / "stage1_secondary_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
