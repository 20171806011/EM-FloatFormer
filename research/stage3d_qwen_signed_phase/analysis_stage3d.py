"""Stage 3D CPU physical calibration, analytic baselines, dev-pilot sample-size lock, and master bundling."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from research.stage3_ministral_cache.analytic import (
    readout_from_table,
    solve_full_iq_known_param,
)
from research.stage3_ministral_cache.data import generate_mothers
from research.stage3_ministral_cache.schema import (
    ROOTS,
    ROTATIONS,
    matched_sum,
    sector_of_complex,
)
from research.stage3c.A.q2_reader import read as q2_reader_read
from research.stage3d_qwen_signed_phase.data_stage3d import (
    ARM_NAMES,
    BLIND_CELLS,
    CONNECTOR_SEEDS,
    NS_BLIND_Q2_C4,
    NS_DEV_PILOT,
    NS_REF_CALIB,
    Q2_BLIND_ARMS,
    REPLAY_TRAIN_CONDITIONS,
    build_q2_arm_references,
    generate_paired_cfo_blocks,
    load_s17_encoder,
)
from research.stage3d_qwen_signed_phase.recompute_all import (
    BLOCK_SIZE,
    CELL_ORDER,
    Q2_LOGICAL_REF_IDX,
    recompute_all,
    score_vectors,
    softmax_temperature,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3D_DIR = Path(__file__).resolve().parent


def calibrate_signed_reference_table(
    encoder,
    cell_idx: int,
    abs_omega: float,
    h: int,
    ell: int,
    t_p: float,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    blocks_per_batch: int = 128,
    n_batches: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrate quarter-turn pooled 256-code reference table on independent namespace 960501 (65,536 mothers/sign)."""
    counts = np.zeros((2, 4, 4, 4), dtype=np.int64)
    mass = np.zeros((2, 256), dtype=np.float64)
    omegas = (-float(abs_omega), float(abs_omega))
    for chunk_i in range(n_batches):
        b_map, _ = generate_paired_cfo_blocks(
            namespace=NS_REF_CALIB,
            cell_idx=cell_idx * 100 + chunk_i,
            blocks=blocks_per_batch,
            h=h,
            ell=ell,
            t_p=t_p,
            omegas=omegas,
            snr_db=snr_db,
            sigma_phi=sigma_phi,
            split_prefix="ref_cal",
        )
        for s_i, w in enumerate(omegas):
            batch = b_map[w]
            z1 = np.asarray(encoder.encode(batch.y1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(batch.y2), dtype=np.int64)
            codes = 16 * z1 + z2
            targets = batch.k3_repeat
            np.add.at(counts[s_i], (z1 % 4, z2 % 4, (targets - z1 // 4) % 4), 1)
            mass[s_i] += np.bincount(codes, minlength=256)
    full = np.zeros((2, 256, 4), dtype=np.float64)
    for k in range(256):
        z1, z2 = divmod(k, 16)
        full[:, k, :] = np.roll(counts[:, z1 % 4, z2 % 4] + 0.5, z1 // 4, axis=-1)
    ref_probs = full / full.sum(axis=-1, keepdims=True)
    mass_probs = mass / mass.sum(axis=-1, keepdims=True)
    return ref_probs, mass_probs


def compute_blind_cell_index_and_analytic(
    encoder,
    cell_id: str,
    namespace: int,
    blocks: int,
    abs_omega: float,
    h: int,
    ell: int,
    t_p: float,
    cal_cell_idx: int,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
) -> dict:
    """Generate paired blind codes/targets, analytic s17 & full_iq probs, and independent cal_ref."""
    omegas = (-float(abs_omega), float(abs_omega))
    b_map, block_seeds = generate_paired_cfo_blocks(
        namespace=namespace,
        cell_idx=0,
        blocks=blocks,
        h=h,
        ell=ell,
        t_p=t_p,
        omegas=omegas,
        snr_db=snr_db,
        sigma_phi=sigma_phi,
        split_prefix=cell_id,
    )
    n_m = blocks * BLOCK_SIZE
    codes = np.zeros((2, 3, n_m), dtype=np.uint8)
    targets = np.zeros((2, 3, n_m), dtype=np.uint8)
    indep_targets = np.zeros((2, 3, n_m), dtype=np.uint8)
    direct_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)
    full_iq_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)
    rot10_codes = np.zeros((2, n_m), dtype=np.uint8)
    rot10_targets = np.zeros((2, n_m), dtype=np.uint8)
    shuffled_perm = np.zeros((2, 3, n_m), dtype=np.int32)

    for s_i, w in enumerate(omegas):
        batch = b_map[w]
        cond = batch.condition
        for r_i, (s1, s2) in enumerate(ROTATIONS):
            v1, v2 = batch.rotated_view(s1, s2)
            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
            codes[s_i, r_i] = (16 * z1 + z2).astype(np.uint8)
            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)
            indep_targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_indep * ROOTS[s1 % 4])).astype(np.uint8)
            a1, a2 = encoder.table[z1], encoder.table[z2]
            direct_probs[s_i, r_i] = readout_from_table(a1, a2, (), cond, protocol="repeat_x1")["q3"]
            full_iq_probs[s_i, r_i] = solve_full_iq_known_param(v1, v2, (), cond, protocol="repeat_x1")["q3"]
            rng_sh = np.random.default_rng(99123 + 1000 * cal_cell_idx + 10 * s_i + r_i)
            shuffled_perm[s_i, r_i] = rng_sh.permutation(n_m).astype(np.int32)

        v1_10, v2_10 = batch.rotated_view(1, 0)
        z1_10 = np.asarray(encoder.encode(v1_10), dtype=np.int64)
        z2_10 = np.asarray(encoder.encode(v2_10), dtype=np.int64)
        rot10_codes[s_i] = (16 * z1_10 + z2_10).astype(np.uint8)
        rot10_targets[s_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[1])).astype(np.uint8)

    cal_ref, cal_mass = calibrate_signed_reference_table(
        encoder=encoder,
        cell_idx=cal_cell_idx,
        abs_omega=abs_omega,
        h=h,
        ell=ell,
        t_p=t_p,
        snr_db=snr_db,
        sigma_phi=sigma_phi,
    )
    return {
        "codes": codes,
        "targets": targets,
        "indep_targets": indep_targets,
        "direct_table_probs": direct_probs,
        "full_iq_probs": full_iq_probs,
        "rot10_codes": rot10_codes,
        "rot10_targets": rot10_targets,
        "shuffled_perm": shuffled_perm,
        "cal_ref": cal_ref,
        "cal_mass": cal_mass,
        "block_seeds": np.asarray(block_seeds, dtype=np.uint64),
    }


def compute_dev_pilot_sample_size(
    pilot_raw_ll: np.ndarray,
    temps_q3: np.ndarray,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Measure conservative upper block-mean SD on independent dev pilot (16 blocks at |omega|=0.18) and lock B."""
    encoder, _ = load_s17_encoder(repo_root)
    b_map, seeds = generate_paired_cfo_blocks(
        namespace=NS_DEV_PILOT,
        cell_idx=0,
        blocks=16,
        h=2,
        ell=2,
        t_p=4.0,
        omegas=(-0.18, 0.18),
        split_prefix="dev_pilot",
    )
    n_blocks = 16
    n_mothers = n_blocks * BLOCK_SIZE
    codes = np.zeros((2, 3, n_mothers), dtype=np.uint8)
    targets = np.zeros((2, 3, n_mothers), dtype=np.uint8)
    for s_i, w in enumerate((-0.18, 0.18)):
        batch = b_map[w]
        for r_i, (s1, s2) in enumerate(ROTATIONS):
            v1, v2 = batch.rotated_view(s1, s2)
            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
            codes[s_i, r_i] = (16 * z1 + z2).astype(np.uint8)
            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)

    n_arms, n_seeds, n_signs, _, _ = pilot_raw_ll.shape
    hits = np.zeros((n_arms, n_seeds, n_signs, 3, n_mothers), dtype=np.float64)
    nlls = np.zeros((n_arms, n_seeds, n_signs, 3, n_mothers), dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            pt = softmax_temperature(pilot_raw_ll[a, s_idx], float(temps_q3[a, s_idx]))
            for sign_i in range(n_signs):
                pr = pt[sign_i][codes[sign_i]]
                sc = score_vectors(pr.reshape(-1, 4), targets[sign_i].reshape(-1))
                hits[a, s_idx, sign_i] = sc["per_item_hit"].reshape(3, n_mothers)
                nlls[a, s_idx, sign_i] = sc["per_item_nll"].reshape(3, n_mothers)

    blk_hit = hits.mean(axis=(1, 2, 3)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
    blk_nll = nlls.mean(axis=(1, 2, 3)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
    acc_diff = blk_hit[0] - blk_hit[1]
    nll_diff = blk_nll[1] - blk_nll[0]

    sd_acc_obs = float(acc_diff.std(ddof=1))
    sd_nll_obs = float(nll_diff.std(ddof=1))
    df = n_blocks - 1
    # One-sided 95% upper confidence bound on SD under normal block means
    upper_mult = math.sqrt(df / float(chi2.ppf(0.05, df)))
    sd_acc_upper = max(1e-6, sd_acc_obs * upper_mult)
    sd_nll_upper = max(1e-6, sd_nll_obs * upper_mult)

    verify_e_py = repo_root / "research" / "stage3c" / "work_results" / "E" / "verify.py"
    raw_e = subprocess.check_output(
        [
            sys.executable,
            "-B",
            str(verify_e_py),
            "--pilot-block-sd",
            f"{sd_acc_upper:.8f}",
            f"{sd_nll_upper:.8f}",
        ],
        cwd=repo_root,
    )
    e_json = json.loads(raw_e.decode("utf-8"))
    pilot_scenario = e_json["scenarios"][-1]
    b_locked = max(131, int(pilot_scenario["t_blocks"]))
    return {
        "namespace": NS_DEV_PILOT,
        "blocks": n_blocks,
        "mothers": n_mothers,
        "pilot_block_seeds": seeds,
        "observed_acc_diff_mean": float(acc_diff.mean()),
        "observed_acc_diff_block_sd": sd_acc_obs,
        "observed_nll_diff_mean": float(nll_diff.mean()),
        "observed_nll_diff_block_sd": sd_nll_obs,
        "chi2_upper_multiplier_df15_alpha05": upper_mult,
        "conservative_upper_sd_hit": sd_acc_upper,
        "conservative_upper_sd_nll": sd_nll_upper,
        "work_e_scenario": pilot_scenario,
        "b_locked": b_locked,
    }


def assemble_and_analyze_results(
    raw_gpu_dir: Path,
    out_dir: Path,
    g1_blocks: int,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Merge raw GPU scores with CPU physical baselines & references, write compact NPZ, and run recompute_all."""
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, enc_sha = load_s17_encoder(repo_root)

    raw_q3 = np.load(raw_gpu_dir / "raw_q3_gpu.npz", allow_pickle=False)
    raw_q2 = np.load(raw_gpu_dir / "raw_q2_gpu.npz", allow_pickle=False)

    q3_bundle: Dict[str, np.ndarray] = {
        "temps_q3": np.asarray(raw_q3["temps_q3"], dtype=np.float64),
        "temps_q2": np.asarray(raw_q2["temps_q2"], dtype=np.float64),
        "seeds": np.asarray(CONNECTOR_SEEDS, dtype=np.int64),
    }

    cell_configs = [
        ("G1_main_extrap_both", BLIND_CELLS["G1_main_extrap_both"].namespace, g1_blocks, 0.40, 2, 2, 4.0, 0),
        ("G2_interp_both", BLIND_CELLS["G2_interp_both"].namespace, 8, 0.20, 2, 2, 4.0, 1),
        ("G3_extrap_angle", BLIND_CELLS["G3_extrap_angle"].namespace, 8, 0.20, 4, 4, 8.0, 2),
        ("G4_extrap_freq", BLIND_CELLS["G4_extrap_freq"].namespace, 8, 0.40, 1, 1, 2.0, 3),
        # Historical +-0.35 C4 anchor using separate calibration namespace idx 4 and blind test blocks from 950406
        ("H_c4_035", 950406, 16, 0.35, 2, 2, 4.0, 4),
    ]

    for cid, ns, n_blk, w_abs, h, ell, tp, cal_idx in cell_configs:
        phys = compute_blind_cell_index_and_analytic(
            encoder=encoder,
            cell_id=cid,
            namespace=ns,
            blocks=n_blk,
            abs_omega=w_abs,
            h=h,
            ell=ell,
            t_p=tp,
            cal_cell_idx=cal_idx,
        )
        q3_bundle[f"{cid}__raw_ll"] = np.asarray(raw_q3[f"{cid}__raw_ll"], dtype=np.float64)
        for k, v in phys.items():
            q3_bundle[f"{cid}__{k}"] = v
        if cid == "G1_main_extrap_both":
            q3_bundle["g1_block_seeds"] = phys["block_seeds"]
            for extra_k in (
                "G1_main_extrap_both__indep_raw_ll",
                "G1_main_extrap_both__contra_text_flipped_raw_ll",
                "G1_main_extrap_both__contra_phase_flipped_raw_ll",
            ):
                if extra_k in raw_q3:
                    q3_bundle[extra_k] = np.asarray(raw_q3[extra_k], dtype=np.float64)

    np.savez_compressed(out_dir / "blind_q3_scores.npz", **q3_bundle)

    # Historical C4 +0.35 lookup verification on H_c4_035 positive world
    c4_lookup_path = repo_root / "research" / "stage3b_math" / "A" / "c4_s17_lookup.json"
    c4_lookup = json.loads(c4_lookup_path.read_text(encoding="utf-8"))
    cells_16 = [np.asarray(c["q3"], dtype=np.float64) for c in c4_lookup["cells"]]
    q3_pos_tbl = np.zeros((256, 4), dtype=np.float64)
    for k in range(256):
        z1, z2 = divmod(k, 16)
        q3_pos_tbl[k] = np.roll(cells_16[4 * (z1 % 4) + (z2 % 4)], z1 // 4)
    h_pos_codes = q3_bundle["H_c4_035__codes"][1].reshape(-1)
    h_pos_targets = q3_bundle["H_c4_035__targets"][1].reshape(-1)
    h_lookup_probs = q3_pos_tbl[h_pos_codes]
    h_lookup_sc = score_vectors(h_lookup_probs, h_pos_targets)

    # Export Work F schema-compliant +-0.35 tables for phase & zero arms (seed mean) and run check_f.py
    work_f_results = {}
    temps_q3 = q3_bundle["temps_q3"]
    for a_i, aname in enumerate(ARM_NAMES):
        probs_3 = np.stack(
            [softmax_temperature(q3_bundle["H_c4_035__raw_ll"][a_i, s_i], float(temps_q3[a_i, s_i])) for s_i in range(len(CONNECTOR_SEEDS))],
            axis=0,
        )
        mean_tbl = probs_3.mean(axis=0)
        mean_tbl = mean_tbl / mean_tbl.sum(axis=-1, keepdims=True)
        host_json_path = out_dir / f"work_f_host_035_{aname}.json"
        host_json_path.write_text(
            json.dumps(
                {
                    "task": "Q3_repeat",
                    "encoder_sha": enc_sha,
                    "rate_magnitudes": [0.35],
                    "h": 2,
                    "ell": 2,
                    "sigma_phi": 0.1,
                    "snr_db": 2.0,
                    "sign_order": [-1, 1],
                    "pilot": "none",
                    "cache_order": "16*z1+z2",
                    "probabilities": mean_tbl.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        check_f_py = repo_root / "research" / "stage3c" / "F" / "check_f.py"
        cf_out = subprocess.check_output(
            [sys.executable, "-B", str(check_f_py), "--repo", str(repo_root), "--host", str(host_json_path)],
            cwd=repo_root,
        )
        work_f_results[aname] = json.loads(cf_out.decode("utf-8"))["systems"]["host"]

    # Assemble Q2 C4 blind bundle + Work A physical reference (q2_reader.read)
    q2_map, _ = generate_paired_cfo_blocks(
        namespace=NS_BLIND_Q2_C4,
        cell_idx=0,
        blocks=8,
        h=2,
        ell=2,
        t_p=4.0,
        omegas=(0.35,),
        snr_db=2.0,
        sigma_phi=0.10,
        split_prefix="q2_c4",
    )
    q2_batch = q2_map[0.35]
    z1_q2 = np.asarray(encoder.encode(q2_batch.y1), dtype=np.int64)
    z2_q2 = np.asarray(encoder.encode(q2_batch.y2), dtype=np.int64)
    n_q2 = len(q2_batch.x1)

    kernel = json.loads((repo_root / "research" / "stage3c" / "A" / "q2_s17_kernel.json").read_text(encoding="utf-8"))
    work_a_ref = np.zeros((len(Q2_LOGICAL_REF_IDX), n_q2, 4), dtype=np.float64)
    for arm_name, ref_idx in Q2_LOGICAL_REF_IDX.items():
        refs_tuple, _ = build_q2_arm_references(q2_batch, arm_name)
        for row_i in range(n_q2):
            rec = [(r.acquisition_id, bool(r.shares_channel), complex(matched_sum(r.samples[row_i]))) for r in refs_tuple]
            work_a_ref[ref_idx, row_i] = q2_reader_read(kernel, int(z1_q2[row_i]), int(z2_q2[row_i]), rec)

    np.savez_compressed(
        out_dir / "blind_q2_scores.npz",
        q2__raw_ll=np.asarray(raw_q2["q2__raw_ll"], dtype=np.float64),
        temps_q2=np.asarray(raw_q2["temps_q2"], dtype=np.float64),
        q2__labels=q2_batch.x1.astype(np.uint8),
        q2__z1=z1_q2.astype(np.uint8),
        q2__z2=z2_q2.astype(np.uint8),
        q2__work_a_ref_probs=work_a_ref,
        q2__matched_valid1=matched_sum(q2_batch.p_valid1),
        q2__matched_valid2=matched_sum(q2_batch.p_valid2),
        q2__matched_indep=matched_sum(q2_batch.p_indep),
    )

    # Replay C1/C2/C3 forgetting bundle (if provided)
    raw_rep_path = raw_gpu_dir / "raw_replay_gpu.npz"
    if raw_rep_path.exists():
        raw_rep = np.load(raw_rep_path, allow_pickle=False)
        rep_bundle = {}
        for cname in REPLAY_TRAIN_CONDITIONS:
            batch = generate_mothers("s3_test", cname, max_blocks=2)
            cond = batch.condition
            lbl_q1, lbl_q3, lbl_q2 = [], [], []
            dir_q1, dir_q3, dir_q2 = [], [], []
            for s1, s2 in ROTATIONS:
                v1, v2 = batch.rotated_view(s1, s2)
                z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
                z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
                a1, a2 = encoder.table[z1], encoder.table[z2]
                refs_v = batch.references_for_arm("valid")
                lbl_q1.append((batch.x2 - batch.x1 + s2 - s1) % 4)
                lbl_q3.append(sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])))
                lbl_q2.append((batch.x1 + s1) % 4)
                dir_q1.append(readout_from_table(a1, a2, (), cond, protocol="repeat_x1")["q1"])
                dir_q3.append(readout_from_table(a1, a2, (), cond, protocol="repeat_x1")["q3"])
                dir_q2.append(readout_from_table(a1, a2, refs_v, cond, protocol="repeat_x1")["q2"])
            rep_bundle[f"{cname}__q1_raw_ll"] = raw_rep[f"{cname}__q1_raw_ll"]
            rep_bundle[f"{cname}__q3_raw_ll"] = raw_rep[f"{cname}__q3_raw_ll"]
            rep_bundle[f"{cname}__q2v_raw_ll"] = raw_rep[f"{cname}__q2v_raw_ll"]
            rep_bundle[f"{cname}__q1_labels"] = np.concatenate(lbl_q1).astype(np.uint8)
            rep_bundle[f"{cname}__q3_labels"] = np.concatenate(lbl_q3).astype(np.uint8)
            rep_bundle[f"{cname}__q2v_labels"] = np.concatenate(lbl_q2).astype(np.uint8)
            rep_bundle[f"{cname}__q1_direct_probs"] = np.concatenate(dir_q1, axis=0)
            rep_bundle[f"{cname}__q3_direct_probs"] = np.concatenate(dir_q3, axis=0)
            rep_bundle[f"{cname}__q2v_direct_probs"] = np.concatenate(dir_q2, axis=0)
        np.savez_compressed(out_dir / "replay_c123_scores.npz", **rep_bundle)

    summary = recompute_all(out_dir)
    summary["historical_c4_035_lookup_anchor"] = {
        "source": "research/stage3b_math/A/c4_s17_lookup.json",
        "h_c4_pos_acc": h_lookup_sc["acc"],
        "h_c4_pos_nll": h_lookup_sc["nll"],
        "work_f_check_f_py_035": work_f_results,
    }
    (out_dir / "summary_master.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble and analyze Stage 3D GPU raw results on CPU.")
    parser.add_argument("--raw-gpu-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--g1-blocks", type=int, default=131)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    assemble_and_analyze_results(
        raw_gpu_dir=args.raw_gpu_dir,
        out_dir=args.out_dir,
        g1_blocks=args.g1_blocks,
        repo_root=args.repo_root,
    )
