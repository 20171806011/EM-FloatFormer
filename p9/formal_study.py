"""Closed-book formal Phase 9 evaluation suite (P9-A multi-SNR attribution + P9-B/C ID & OOD gating + replay)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from p8.coding import decode as p8_decode
from p8c.link import extra_rows
from p9.channel import (
    build_sc_channel_matrix,
    effective_row_noise,
    oracle_readout_b,
    simulate_diagnostic,
    simulate_link,
    simulate_snr_sweep,
)
from p9.contract import (
    CORE_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SNR_GRID_DB,
)
from p9.decoder import decode_strong_sc, ml_lower_bound_certificate, warmup_numba
from p9.freeze import (
    FORMAL_CLUSTERS,
    LINKS_PER_CLUSTER_ID,
    LINKS_PER_CLUSTER_OOD,
    NEURAL_SEEDS,
    NEURAL_VARIANTS,
    REPLAY_CLUSTERS,
    RESULTS,
    verify_freeze,
)
from p9.gating import (
    build_candidates_with_priors,
    gate_always_clear,
    gate_always_inherit,
    gate_jeffreys_student_t,
    gate_pre_bayes_factor,
)
from p9.neural_gate import TemporalTrustGate, compose_gate_output


_LOADED_GATES = None


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    warmup_numba()


def _get_neural_gates():
    global _LOADED_GATES
    if _LOADED_GATES is None:
        ckpt_dir = RESULTS / "p9c_dev_ckpts"
        models = {}
        for var in NEURAL_VARIANTS:
            models[var] = []
            for seed in NEURAL_SEEDS:
                m = TemporalTrustGate(variant=var, d_model=12, n_heads=4, n_layers=1)
                ck = torch.load(ckpt_dir / f"{var}_seed{seed}.pt", map_location="cpu", weights_only=True)
                m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
                m.eval()
                models[var].append((seed, m))
        _LOADED_GATES = models
    return _LOADED_GATES


def _ensemble_gate(variant, obs, cand_inh, cand_clr):
    """Frozen equal-weight 3-seed logit ensemble + individual per-seed GateOutputs."""
    models = _get_neural_gates()[variant]
    feats = models[0][1].extract_features_for_variant(obs, cand_inh, cand_clr)
    s_t = torch.from_numpy(feats.scalar_feats).to(torch.float64).unsqueeze(0)
    c_t = torch.from_numpy(feats.complex_feats).to(torch.complex128).unsqueeze(0)
    t_t = torch.from_numpy(feats.t_norm).to(torch.float64).unsqueeze(0)
    ev_t = torch.from_numpy(feats.ev_base).to(torch.float64).unsqueeze(0)
    bl_t = torch.from_numpy(feats.base_logit).to(torch.float64).unsqueeze(0)

    logits_list = []
    per_seed = {}
    with torch.no_grad():
        for seed, m in models:
            w_s, logit_s = m.forward_tensors(s_t, c_t, t_t, ev_t, bl_t)
            logits_list.append(logit_s.squeeze(0).cpu().numpy())
            per_seed[seed] = compose_gate_output(w_s.squeeze(0).cpu().numpy(), cand_inh, cand_clr, probe_row=obs.probe_row)
    mean_logit = np.mean(logits_list, axis=0)
    w_ens = 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40.0, 40.0)))
    ens_out = compose_gate_output(w_ens, cand_inh, cand_clr, probe_row=obs.probe_row)
    return ens_out, per_seed


def _bit_bce_sum(llrs: np.ndarray, code_bits: np.ndarray) -> float:
    signed = 2.0 * code_bits.astype(np.float64) - 1.0
    x = -signed * np.clip(llrs, -40.0, 40.0)
    return float(np.sum(np.logaddexp(0.0, x)))


def _score_gate_arm(obs, truth, gate_out, *, run_extra=True):
    """Score one GateOutput on BOTH native_ldpc and strong_large, returning full auditable metrics."""
    core_rows = list(CORE_ROWS)
    free_rows = list(extra_rows(obs.training_rows)) if run_extra else []

    # 1. Native LDPC decoding & soft-bit diagnostics
    t0_nat = time.perf_counter_ns()
    core_llrs = gate_out.mix_llrs[core_rows].reshape(16, 128)
    nat_wrong = p8_decode(core_llrs) != truth.b_info[:16]
    nat_core_err = int(nat_wrong.any(axis=1).sum())
    nat_info_err = int(nat_wrong.sum())
    nat_coded_err = int(((core_llrs > 0) != truth.b_code[:16]).sum())
    nat_bce_sum = _bit_bce_sum(core_llrs, truth.b_code[:16])
    nat_extra_err = 0
    if free_rows:
        ext_llrs = gate_out.mix_llrs[free_rows].reshape(len(free_rows), 128)
        nat_extra_err = int((p8_decode(ext_llrs) != truth.b_info[16:16 + len(free_rows)]).any(axis=1).sum())
    nat_ms = (time.perf_counter_ns() - t0_nat) / 1e6

    # 2. Strong-Large unified mixture decoding (Turbo MMSE-PIC + BP + OSD-2/3 exact 32x32 waveform re-ranking)
    t0_str = time.perf_counter_ns()
    str_core_err = 0
    str_info_err = 0
    for idx_c, row in enumerate(core_rows):
        H_r = build_sc_channel_matrix(gate_out.mix_mean_ray[:, row])
        noise_r = effective_row_noise(gate_out.mix_var_ray, obs.n0, rows=(row,))[0]
        res = decode_strong_sc(
            obs.y_b[row], H_r, noise_r,
            list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75,
            initial_llrs=gate_out.mix_llrs[row].reshape(128),
        )
        diff = res.info_bits != truth.b_info[idx_c]
        if bool(diff.any()):
            str_core_err += 1
            str_info_err += int(diff.sum())
    str_extra_err = 0
    for idx_e, row in enumerate(free_rows):
        H_r = build_sc_channel_matrix(gate_out.mix_mean_ray[:, row])
        noise_r = effective_row_noise(gate_out.mix_var_ray, obs.n0, rows=(row,))[0]
        res = decode_strong_sc(
            obs.y_b[row], H_r, noise_r,
            list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75,
            initial_llrs=gate_out.mix_llrs[row].reshape(128),
        )
        if not np.array_equal(res.info_bits, truth.b_info[16 + idx_e]):
            str_extra_err += 1
    str_ms = (time.perf_counter_ns() - t0_str) / 1e6

    # 3. Withheld row-19 probe prediction & variance calibration
    pred = gate_out.mix_probe_pred
    pvar = gate_out.mix_probe_var
    if not (np.isfinite(pred).all() and np.isfinite(pvar).all() and (pvar > 0).all()):
        raise ValueError("Nonfinite probe prediction or nonpositive variance")
    probe_sse = float(np.sum(np.abs(truth.y_b_probe_noisy - pred) ** 2))
    clean_sse = float(np.sum(np.abs(truth.y_b_probe_clean - pred) ** 2))
    probe_energy = float(np.sum(np.abs(truth.y_b_probe_noisy) ** 2))
    clean_energy = float(np.sum(np.abs(truth.y_b_probe_clean) ** 2))
    probe_var_sum = float(np.sum(pvar))
    probe_nll_sum = float(np.sum(np.abs(truth.y_b_probe_noisy - pred) ** 2 / pvar + np.log(np.pi * pvar)))

    return dict(
        blocks=16,
        native_block_errors=nat_core_err,
        native_info_errors=nat_info_err,
        native_extra_blocks=len(free_rows),
        native_extra_errors=nat_extra_err,
        strong_block_errors=str_core_err,
        strong_info_errors=str_info_err,
        strong_extra_blocks=len(free_rows),
        strong_extra_errors=str_extra_err,
        coded_bits=2048,
        coded_errors=nat_coded_err,
        bce_sum=nat_bce_sum,
        probes=32,
        probe_sse=probe_sse,
        clean_sse=clean_sse,
        probe_energy=probe_energy,
        clean_energy=clean_energy,
        probe_var_sum=probe_var_sum,
        probe_nll_sum=probe_nll_sum,
        mean_history_weight=float(np.mean(gate_out.history_weight)),
        probe_history_weight=float(gate_out.probe_weight),
        native_ms=nat_ms,
        strong_ms=str_ms,
    )


def _eval_p9a_cluster(cluster_idx: int) -> list[dict]:
    """Evaluate one cluster (32 links) across all 4 SNRs (6,9,12,15 dB) for P9-A attribution & ML certificates."""
    records = []
    for link_idx in range(LINKS_PER_CLUSTER_ID):
        seed = SEED_P9_FORMAL_ID + 1000 * cluster_idx + link_idx
        sweep = simulate_snr_sweep(seed, snr_grid_db=SNR_GRID_DB, regime="continuous", budget="sparse")
        snr_rows = {}
        for snr_db in SNR_GRID_DB:
            obs, truth = sweep[snr_db]
            # True-CSI readout + Strong-Large + ML certificate
            o_llrs, o_pred, o_var = oracle_readout_b(truth, obs.y_b, obs.training_rows, obs.n0)
            nat_true_wrong = (p8_decode(o_llrs[list(CORE_ROWS)].reshape(16, 128)) != truth.b_info[:16]).any(axis=1)

            t0 = time.perf_counter_ns()
            str_true_wrong = np.zeros(16, dtype=bool)
            ml_must_err = 0
            ml_undecided = 0
            ml_single_flip = 0
            ml_extra_witness = 0
            for idx_c, row in enumerate(CORE_ROWS):
                H_true = build_sc_channel_matrix(truth.h_b_true[:, row])
                noise_true = np.full(32, obs.n0, dtype=np.float64)
                res = decode_strong_sc(
                    obs.y_b[row], H_true, noise_true,
                    list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75,
                )
                str_true_wrong[idx_c] = not np.array_equal(res.info_bits, truth.b_info[idx_c])
                cert = ml_lower_bound_certificate(
                    obs.y_b[row], H_true, truth.b_code[idx_c], extra_candidates=res.candidates_u64
                )
                if cert.ml_must_err:
                    ml_must_err += 1
                    if cert.witness_source == "single_flip":
                        ml_single_flip += 1
                    else:
                        ml_extra_witness += 1
                else:
                    ml_undecided += 1
            true_str_ms = (time.perf_counter_ns() - t0) / 1e6

            # Estimated-CSI (inherit sparse) readout + Strong-Large
            _, _, cand_inh, cand_clr = build_candidates_with_priors(obs)
            g_inh = gate_always_inherit(cand_inh, cand_clr)
            nat_est_wrong = (p8_decode(g_inh.mix_llrs[list(CORE_ROWS)].reshape(16, 128)) != truth.b_info[:16]).any(axis=1)

            t1 = time.perf_counter_ns()
            str_est_wrong = np.zeros(16, dtype=bool)
            for idx_c, row in enumerate(CORE_ROWS):
                H_est = build_sc_channel_matrix(g_inh.mix_mean_ray[:, row])
                noise_est = effective_row_noise(g_inh.mix_var_ray, obs.n0, rows=(row,))[0]
                res = decode_strong_sc(
                    obs.y_b[row], H_est, noise_est,
                    list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75,
                    initial_llrs=g_inh.mix_llrs[row].reshape(128),
                )
                str_est_wrong[idx_c] = not np.array_equal(res.info_bits, truth.b_info[idx_c])
            est_str_ms = (time.perf_counter_ns() - t1) / 1e6

            snr_rows[str(snr_db)] = dict(
                blocks=16,
                true_native_errors=int(nat_true_wrong.sum()),
                true_strong_errors=int(str_true_wrong.sum()),
                true_rescued=int((nat_true_wrong & (~str_true_wrong)).sum()),
                true_lost=int(((~nat_true_wrong) & str_true_wrong).sum()),
                est_native_errors=int(nat_est_wrong.sum()),
                est_strong_errors=int(str_est_wrong.sum()),
                est_rescued=int((nat_est_wrong & (~str_est_wrong)).sum()),
                est_lost=int(((~nat_est_wrong) & str_est_wrong).sum()),
                ml_must_err=ml_must_err,
                ml_undecided=ml_undecided,
                ml_single_flip=ml_single_flip,
                ml_extra_witness=ml_extra_witness,
                true_strong_ms=true_str_ms,
                est_strong_ms=est_str_ms,
            )
        records.append(dict(cluster=cluster_idx, link=link_idx, seed=seed, snrs=snr_rows))
    return records


def _eval_p9bc_cluster(args) -> list[dict]:
    """Evaluate one cluster for P9-B/C ID or OOD cohorts across all arms and both budgets."""
    cluster_idx, split = args
    regimes = REGIMES_ID if split == "id" else (REGIMES_OOD + REGIMES_SANITY)
    n_links = LINKS_PER_CLUSTER_ID if split == "id" else LINKS_PER_CLUSTER_OOD
    seed_base = SEED_P9_FORMAL_ID + 200000 if split == "id" else SEED_P9_FORMAL_OOD
    records = []

    for r_idx, reg in enumerate(regimes):
        for link_idx in range(n_links):
            seed = seed_base + r_idx * 50000 + 1000 * cluster_idx + link_idx
            # Paired sparse and dense observations on exact same channel, payload, and noise!
            obs_sp, truth_sp = simulate_link(seed, snr_db=PRIMARY_SNR_DB, regime=reg, budget="sparse")
            obs_dn, truth_dn = simulate_link(seed, snr_db=PRIMARY_SNR_DB, regime=reg, budget="dense")

            # Build sparse candidates
            t_cand0 = time.perf_counter_ns()
            _, _, c_inh_sp, c_clr_sp = build_candidates_with_priors(obs_sp)
            cand_sp_ms = (time.perf_counter_ns() - t_cand0) / 1e6

            sparse_arms = {}
            for name, fn in (
                ("always_inherit", lambda: gate_always_inherit(c_inh_sp, c_clr_sp)),
                ("always_clear", lambda: gate_always_clear(c_inh_sp, c_clr_sp)),
                ("pre_bayes_factor", lambda: gate_pre_bayes_factor(c_inh_sp, c_clr_sp)),
                ("jeffreys_student_t", lambda: gate_jeffreys_student_t(obs_sp, c_inh_sp, c_clr_sp)),
            ):
                t_g0 = time.perf_counter_ns()
                gout = fn()
                gate_ms = (time.perf_counter_ns() - t_g0) / 1e6
                row_m = _score_gate_arm(obs_sp, truth_sp, gout, run_extra=True)
                row_m["gate_ms"] = gate_ms
                row_m["cand_ms"] = cand_sp_ms
                sparse_arms[name] = row_m

            # Frozen neural gates (3-seed ensemble + individual seed disclosure on transport_attn & local)
            for var in NEURAL_VARIANTS:
                t_g0 = time.perf_counter_ns()
                ens_out, per_seed = _ensemble_gate(var, obs_sp, c_inh_sp, c_clr_sp)
                gate_ms = (time.perf_counter_ns() - t_g0) / 1e6
                row_m = _score_gate_arm(obs_sp, truth_sp, ens_out, run_extra=True)
                row_m["gate_ms"] = gate_ms
                row_m["cand_ms"] = cand_sp_ms
                sparse_arms[var] = row_m
                if split == "id" and var in ("transport_attn", "local"):
                    for s_id, s_out in per_seed.items():
                        s_row = _score_gate_arm(obs_sp, truth_sp, s_out, run_extra=False)
                        s_row["gate_ms"] = gate_ms / len(NEURAL_SEEDS)
                        s_row["cand_ms"] = cand_sp_ms
                        sparse_arms[f"{var}_s{s_id}"] = s_row

            # Oracle true-channel reference on sparse
            o_llrs, o_pred, o_var = oracle_readout_b(truth_sp, obs_sp.y_b, obs_sp.training_rows, obs_sp.n0)
            from p9.contract import GateOutput
            o_gate = GateOutput(
                history_weight=np.ones(20, dtype=np.float64),
                probe_weight=1.0,
                mix_llrs=o_llrs,
                mix_probe_pred=o_pred,
                mix_probe_var=o_var,
                mix_mean_ray=truth_sp.h_b_true,
                mix_var_ray=np.zeros_like(truth_sp.h_b_true, dtype=np.float64),
            )
            sparse_arms["oracle_ref"] = _score_gate_arm(obs_sp, truth_sp, o_gate, run_extra=True)

            # Dense reference arms (for engineering non-inferiority & throughput comparison)
            _, _, c_inh_dn, c_clr_dn = build_candidates_with_priors(obs_dn)
            dense_arms = {}
            for name, fn in (
                ("always_clear", lambda: gate_always_clear(c_inh_dn, c_clr_dn)),
                ("always_inherit", lambda: gate_always_inherit(c_inh_dn, c_clr_dn)),
                ("pre_bayes_factor", lambda: gate_pre_bayes_factor(c_inh_dn, c_clr_dn)),
                ("jeffreys_student_t", lambda: gate_jeffreys_student_t(obs_dn, c_inh_dn, c_clr_dn)),
            ):
                dense_arms[name] = _score_gate_arm(obs_dn, truth_dn, fn(), run_extra=False)

            records.append(dict(
                split=split,
                cluster=cluster_idx,
                regime=reg,
                link=link_idx,
                seed=seed,
                changed_ray=truth_sp.changed_ray,
                delta_freq=float(truth_sp.delta_freq),
                shift_row=truth_sp.shift_row,
                budgets=dict(sparse=sparse_arms, dense=dense_arms),
            ))
    return records


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Closed-book formal Phase 9 evaluation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replay-only", action="store_true", help="Single-worker replay of clusters 1 & 16")
    args = parser.parse_args()

    _init_worker()
    freeze_sha = verify_freeze()
    print(f"Verified Phase 9 prospective freeze sha256={freeze_sha}", flush=True)
    t0 = time.perf_counter()

    if args.replay_only:
        p9a_rep = []
        p9bc_rep = []
        for c in REPLAY_CLUSTERS:
            p9a_rep.extend(_eval_p9a_cluster(c))
            p9bc_rep.extend(_eval_p9bc_cluster((c, "id")))
        _write_jsonl(RESULTS / "p9a_formal_replay.jsonl", p9a_rep)
        _write_jsonl(RESULTS / "p9bc_formal_replay.jsonl", p9bc_rep)
        print(f"Single-worker replay of clusters {REPLAY_CLUSTERS} done in {time.perf_counter() - t0:.1f}s", flush=True)
        return

    # 1. P9-A multi-SNR attribution (G=16 clusters = 512 links x 4 SNRs = 8,192 blocks/SNR)
    t_a = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        p9a_chunks = list(pool.map(_eval_p9a_cluster, FORMAL_CLUSTERS))
    p9a_records = [r for ch in p9a_chunks for r in ch]
    _write_jsonl(RESULTS / "p9a_formal_links.jsonl", p9a_records)
    print(f"P9-A formal complete: {len(p9a_records)} links x 4 SNRs in {time.perf_counter() - t_a:.1f}s", flush=True)

    # 1b. P9-A diagnostics (AWGN 9 dB & zero-noise 3-ray, 64 links = 1,024 blocks each)
    diags = {}
    for kind in ("awgn", "zero_noise"):
        nat_e, str_e, tot_b = 0, 0, 0
        for i in range(64):
            obs, truth = simulate_diagnostic(SEED_P9_FORMAL_ID + 900000 + i, kind=kind, snr_db=PRIMARY_SNR_DB, budget="sparse")
            n0_eff = obs.n0 if obs.n0 > 0 else 1e-12
            o_llrs, _, _ = oracle_readout_b(truth, obs.y_b, obs.training_rows, n0_eff)
            nat_e += int((p8_decode(o_llrs[list(CORE_ROWS)].reshape(16, 128)) != truth.b_info[:16]).any(axis=1).sum())
            for idx_c, row in enumerate(CORE_ROWS):
                H_r = build_sc_channel_matrix(truth.h_b_true[:, row])
                res = decode_strong_sc(obs.y_b[row], H_r, np.full(32, n0_eff), list_size="large", turbo_pic=True)
                if not np.array_equal(res.info_bits, truth.b_info[idx_c]):
                    str_e += 1
            tot_b += 16
        diags[kind] = dict(blocks=tot_b, native_errors=nat_e, strong_errors=str_e)
    (RESULTS / "p9a_formal_diagnostics.json").write_text(json.dumps(diags, indent=2) + "\n", encoding="utf-8")

    # 2. P9-B & P9-C Formal ID Cohort (G=16 x 32 links x 3 regimes = 1,536 links = 24,576 blocks)
    t_id = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        id_chunks = list(pool.map(_eval_p9bc_cluster, [(c, "id") for c in FORMAL_CLUSTERS]))
    id_records = [r for ch in id_chunks for r in ch]
    _write_jsonl(RESULTS / "p9bc_formal_links.jsonl", id_records)
    print(f"P9-B/C formal ID complete: {len(id_records)} links in {time.perf_counter() - t_id:.1f}s", flush=True)

    # 3. P9-B & P9-C Formal OOD + Sanity Cohort (G=16 x 16 links x 3 regimes = 768 links)
    t_ood = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        ood_chunks = list(pool.map(_eval_p9bc_cluster, [(c, "ood") for c in FORMAL_CLUSTERS]))
    ood_records = [r for ch in ood_chunks for r in ch]
    _write_jsonl(RESULTS / "p9bc_formal_ood.jsonl", ood_records)
    print(f"P9-B/C formal OOD complete: {len(ood_records)} links in {time.perf_counter() - t_ood:.1f}s", flush=True)

    print(f"ALL FORMAL P9 COHORTS COMPLETE in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
