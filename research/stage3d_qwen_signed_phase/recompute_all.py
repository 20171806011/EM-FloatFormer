"""Standalone CPU recomputation of all Stage 3D primary, exploratory, Work F/G/A, and replay metrics.

Depends ONLY on Python standard library, numpy, and scipy — zero PyTorch, GPU, or repository-tree imports required.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t


BLOCK_SIZE = 64
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026311
CELL_ORDER = (
    "G1_main_extrap_both",
    "G2_interp_both",
    "G3_extrap_angle",
    "G4_extrap_freq",
)
CELL_META = {
    "G1_main_extrap_both": {"abs_omega": 0.40, "h": 2, "ell": 2, "t_p": 4.0, "cum_angle_abs": 1.60, "sigma2_t": 0.04},
    "G2_interp_both": {"abs_omega": 0.20, "h": 2, "ell": 2, "t_p": 4.0, "cum_angle_abs": 0.80, "sigma2_t": 0.04},
    "G3_extrap_angle": {"abs_omega": 0.20, "h": 4, "ell": 4, "t_p": 8.0, "cum_angle_abs": 1.60, "sigma2_t": 0.08},
    "G4_extrap_freq": {"abs_omega": 0.40, "h": 1, "ell": 1, "t_p": 2.0, "cum_angle_abs": 0.80, "sigma2_t": 0.02},
}
Q2_ARMS = (
    "none",
    "valid",
    "duplicate_copy_dedup",
    "duplicate_copy_raw",
    "independent_only",
    "valid_plus_independent",
    "fresh_resample",
)
Q2_LOGICAL_REF_IDX = {
    "none": 0,
    "valid": 1,
    "duplicate_copy_dedup": 2,
    "independent_only": 3,
    "valid_plus_independent": 4,
    "fresh_resample": 5,
}


def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:
    scaled = np.asarray(raw_ll, dtype=np.float64) / float(temp)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    ex = np.exp(scaled)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute fractional-tie hits, NLL (clipped at 1e-12), Brier, and 10-bin ECE."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if p.shape != (n, 4):
        raise ValueError(f"Expected {(n, 4)}, got {p.shape}")
    max_p = np.max(p, axis=1, keepdims=True)
    ties = p == max_p
    tie_counts = ties.sum(axis=1)
    hits = ties[np.arange(n), y].astype(np.float64) / tie_counts.astype(np.float64)
    n_ties = int((tie_counts > 1).sum())

    p_true_raw = p[np.arange(n), y]
    n_clipped = int((p_true_raw < 1e-12).sum())
    p_true = np.clip(p_true_raw, 1e-12, 1.0)
    nll = -np.log(p_true)

    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    brier = np.sum((p - one_hot) ** 2, axis=1)

    confs = np.max(p, axis=1)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (confs >= edges[i]) & (confs <= edges[i + 1]) if i == 0 else (confs > edges[i]) & (confs <= edges[i + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / float(n)) * abs(float(hits[mask].mean()) - float(confs[mask].mean()))

    return {
        "n": int(n),
        "acc": float(hits.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": float(ece),
        "tie_count": n_ties,
        "clipped_count": n_clipped,
        "per_item_hit": hits,
        "per_item_nll": nll,
    }


def paired_t_ci(block_diffs: np.ndarray) -> dict:
    arr = np.asarray(block_diffs, dtype=np.float64)
    b = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if b > 1 else 0.0
    se = sd / math.sqrt(b) if b > 0 else 0.0
    t_crit = float(student_t.ppf(0.975, df=b - 1)) if b > 1 else 0.0
    return {
        "n_blocks": int(b),
        "mean": mean,
        "block_sd": sd,
        "se": se,
        "t_crit_95": t_crit,
        "lo95": mean - t_crit * se,
        "hi95": mean + t_crit * se,
    }


def block_bootstrap_ci(block_diffs: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    arr = np.asarray(block_diffs, dtype=np.float64)
    b = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b, size=(resamples, b))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_blocks": int(b),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean": float(arr.mean()),
        "lo95": float(lo),
        "hi95": float(hi),
    }


def compute_16way_group_avg(table_256: np.ndarray) -> np.ndarray:
    """Offline Work G 16-way C4 x C4 alignment average on a (..., 256, 4) Q3_repeat table."""
    orig_shape = table_256.shape
    assert orig_shape[-2:] == (256, 4)
    flat = table_256.reshape(-1, 256, 4)
    out = np.zeros_like(flat)
    for z1 in range(16):
        for z2 in range(16):
            acc = np.zeros((flat.shape[0], 4), dtype=np.float64)
            for a in range(4):
                rz1 = (z1 + 4 * a) % 16
                for b in range(4):
                    rz2 = (z2 + 4 * b) % 16
                    acc += np.roll(flat[:, 16 * rz1 + rz2, :], -a, axis=-1)
            out[:, 16 * z1 + z2, :] = acc / 16.0
    return out.reshape(orig_shape)


def compute_representative_readout(table_256: np.ndarray) -> np.ndarray:
    """Offline Work G single-representative F0 equivariant readout using canonical (z1%4, z2%4)."""
    orig_shape = table_256.shape
    assert orig_shape[-2:] == (256, 4)
    flat = table_256.reshape(-1, 256, 4)
    out = np.zeros_like(flat)
    for z1 in range(16):
        turn1, inner1 = divmod(z1, 4)
        for z2 in range(16):
            _, inner2 = divmod(z2, 4)
            out[:, 16 * z1 + z2, :] = np.roll(flat[:, 16 * inner1 + inner2, :], turn1, axis=-1)
    return out.reshape(orig_shape)


def evaluate_cell_paired(
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes: np.ndarray,
    targets: np.ndarray,
    direct_probs: np.ndarray,
    full_iq_probs: np.ndarray,
    seeds: Sequence[int],
    is_primary: bool = False,
) -> dict:
    """Evaluate paired phase vs zero arms over (signs x rotations x mothers) and 3 seeds."""
    # raw_ll_table: (2, 3, 2, 256, 4); codes/targets: (2, 3, M)
    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
    assert n_arms == 2 and n_signs == 2
    n_rot = codes.shape[1]
    n_mothers = codes.shape[2]
    assert n_mothers % BLOCK_SIZE == 0
    n_blocks = n_mothers // BLOCK_SIZE

    probs_table = np.zeros_like(raw_ll_table, dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            probs_table[a, s_idx] = softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx]))

    # Shape (2, 3, 2, n_rot, M)
    hits = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    nlls = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    briers = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    tie_counts = np.zeros((n_arms, n_seeds), dtype=int)
    clip_counts = np.zeros((n_arms, n_seeds), dtype=int)
    all_probs_flat = [[None] * n_seeds for _ in range(n_arms)]
    all_lbl_flat = targets.reshape(-1)

    for a in range(n_arms):
        for s_idx in range(n_seeds):
            p_list = []
            for sign_i in range(n_signs):
                p_sr = probs_table[a, s_idx, sign_i][codes[sign_i]]  # (n_rot, M, 4)
                p_list.append(p_sr)
                sc = score_vectors(p_sr.reshape(-1, 4), targets[sign_i].reshape(-1))
                hits[a, s_idx, sign_i] = sc["per_item_hit"].reshape(n_rot, n_mothers)
                nlls[a, s_idx, sign_i] = sc["per_item_nll"].reshape(n_rot, n_mothers)
                oh = np.zeros_like(p_sr)
                np.put_along_axis(oh, targets[sign_i][..., None], 1.0, axis=-1)
                briers[a, s_idx, sign_i] = np.sum((p_sr - oh) ** 2, axis=-1)
                tie_counts[a, s_idx] += sc["tie_count"]
                clip_counts[a, s_idx] += sc["clipped_count"]
            all_probs_flat[a][s_idx] = np.stack(p_list, axis=0).reshape(-1, 4)

    # Pooled over signs, rotations, and 3 seeds equally
    # Block aggregation: first mother-level average across (seeds, signs, rotations), then 64-mother block means
    mother_hit = hits.mean(axis=(1, 2, 3))  # (2, M)
    mother_nll = nlls.mean(axis=(1, 2, 3))  # (2, M)
    block_hit = mother_hit.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)
    block_nll = mother_nll.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)

    acc_gain_blocks = block_hit[0] - block_hit[1]        # phase - zero (>0 is better)
    nll_improv_blocks = block_nll[1] - block_nll[0]      # zero - phase (>0 is better)
    acc_t_ci = paired_t_ci(acc_gain_blocks)
    nll_t_ci = paired_t_ci(nll_improv_blocks)
    acc_boot_ci = block_bootstrap_ci(acc_gain_blocks)
    nll_boot_ci = block_bootstrap_ci(nll_improv_blocks)

    # Pooled ECE across all (seed x sign x rot x mother) predictions per arm
    pooled_arm_stats = {}
    for a, aname in enumerate(("phase", "zero")):
        flat_p = np.concatenate(all_probs_flat[a], axis=0)
        flat_y = np.tile(all_lbl_flat, n_seeds)
        sc_pool = score_vectors(flat_p, flat_y)
        pooled_arm_stats[aname] = {
            "acc": float(hits[a].mean()),
            "nll": float(nlls[a].mean()),
            "brier": float(briers[a].mean()),
            "ece": float(sc_pool["ece"]),
            "tie_count": int(tie_counts[a].sum()),
            "clipped_count": int(clip_counts[a].sum()),
            "by_sign": {
                "-omega": {"acc": float(hits[a, :, 0].mean()), "nll": float(nlls[a, :, 0].mean())},
                "+omega": {"acc": float(hits[a, :, 1].mean()), "nll": float(nlls[a, :, 1].mean())},
            },
        }

    # Per-seed breakdown
    per_seed = []
    for s_idx, seed_val in enumerate(seeds):
        b_hit_s = hits[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
        b_nll_s = nlls[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
        acc_g_s = b_hit_s[0] - b_hit_s[1]
        nll_i_s = b_nll_s[1] - b_nll_s[0]
        sc_p = score_vectors(all_probs_flat[0][s_idx], all_lbl_flat)
        sc_z = score_vectors(all_probs_flat[1][s_idx], all_lbl_flat)
        per_seed.append(
            {
                "seed": int(seed_val),
                "phase": {"acc": sc_p["acc"], "nll": sc_p["nll"], "brier": sc_p["brier"], "ece": sc_p["ece"]},
                "zero": {"acc": sc_z["acc"], "nll": sc_z["nll"], "brier": sc_z["brier"], "ece": sc_z["ece"]},
                "acc_gain_phase_minus_zero": paired_t_ci(acc_g_s),
                "nll_improv_zero_minus_phase": paired_t_ci(nll_i_s),
            }
        )

    seed_acc_gains = [x["acc_gain_phase_minus_zero"]["mean"] for x in per_seed]
    seed_nll_improv = [x["nll_improv_zero_minus_phase"]["mean"] for x in per_seed]

    # Analytic baselines on exact same mothers
    sc_dir = score_vectors(direct_probs.reshape(-1, 4), all_lbl_flat)
    sc_iq = score_vectors(full_iq_probs.reshape(-1, 4), all_lbl_flat)

    # Host minus Direct Readout paired block stats
    dir_hit = sc_dir["per_item_hit"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
    dir_nll = sc_dir["per_item_nll"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
    dir_blk_hit = dir_hit.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
    dir_blk_nll = dir_nll.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)

    res = {
        "blocks": int(n_blocks),
        "mothers": int(n_mothers),
        "logical_rows_per_arm": int(n_seeds * n_signs * n_rot * n_mothers),
        "arms": pooled_arm_stats,
        "paired_phase_minus_zero": {
            "acc_gain_t95": acc_t_ci,
            "acc_gain_boot95": acc_boot_ci,
            "nll_improv_t95": nll_t_ci,
            "nll_improv_boot95": nll_boot_ci,
        },
        "per_seed": per_seed,
        "seed_dispersion": {
            "acc_gain_sd": float(np.std(seed_acc_gains, ddof=1)),
            "acc_gain_range": [float(min(seed_acc_gains)), float(max(seed_acc_gains))],
            "nll_improv_sd": float(np.std(seed_nll_improv, ddof=1)),
            "nll_improv_range": [float(min(seed_nll_improv)), float(max(seed_nll_improv))],
        },
        "analytic_same_mothers": {
            "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
            "full_iq_known_param": {k: sc_iq[k] for k in ("acc", "nll", "brier", "ece")},
        },
        "phase_minus_direct_s17": {
            "acc_gain_t95": paired_t_ci(block_hit[0] - dir_blk_hit),
            "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[0]),
        },
        "zero_minus_direct_s17": {
            "acc_gain_t95": paired_t_ci(block_hit[1] - dir_blk_hit),
            "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[1]),
        },
    }
    if is_primary:
        pass_acc = bool(acc_t_ci["lo95"] > 0.02)
        pass_nll = bool(nll_t_ci["lo95"] > 0.02)
        res["dual_practical_threshold"] = {
            "acc_threshold": 0.02,
            "nll_threshold": 0.02,
            "acc_t95_lo": float(acc_t_ci["lo95"]),
            "nll_t95_lo": float(nll_t_ci["lo95"]),
            "pass_acc": pass_acc,
            "pass_nll": pass_nll,
            "pass_both": bool(pass_acc and pass_nll),
        }
    return res


def work_f_sign_probe(
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes_rot0: np.ndarray,
    targets_rot0: np.ndarray,
    cal_ref: np.ndarray,
    cal_mass: np.ndarray,
    seeds: Sequence[int],
    contra_text_flipped: Optional[np.ndarray] = None,
    contra_phase_flipped: Optional[np.ndarray] = None,
) -> dict:
    """Work F signed CFO probe: 4-cell factorial, swapped-minus-matched NLL, orientation gain, and single-cache spotlight."""
    n_arms, n_seeds, _, _, _ = raw_ll_table.shape
    n_mothers = codes_rot0.shape[1]
    assert n_mothers % BLOCK_SIZE == 0
    n_blocks = n_mothers // BLOCK_SIZE

    overlap = np.minimum(cal_mass[0], cal_mass[1])
    weights = overlap / overlap.sum()
    target_diff = cal_ref[1] - cal_ref[0]  # D(z)
    norm_d2 = float(np.sum(weights[:, None] * (target_diff ** 2)))
    norm_d = math.sqrt(norm_d2)

    out: Dict[str, object] = {
        "mothers": int(n_mothers),
        "blocks": int(n_blocks),
        "cal_ref_overlap_mass": float(overlap.sum()),
        "cal_ref_odd_norm_sq": norm_d2,
        "arms": {},
    }

    def _analyze_table(tbl_2_256_4: np.ndarray) -> dict:
        contrast = tbl_2_256_4[1] - tbl_2_256_4[0]
        gain = float(np.sum(weights[:, None] * contrast * target_diff) / norm_d2)
        resid_ratio = float(math.sqrt(np.sum(weights[:, None] * ((contrast - gain * target_diff) ** 2)) / norm_d2))
        mean_sign_tv = float(0.5 * np.sum(weights[:, None] * np.abs(contrast)))
        rot_tvs = []
        for z in range(256):
            z1, z2 = divmod(z, 16)
            rz = 16 * ((z1 + 4) % 16) + z2
            disc = tbl_2_256_4[:, rz, :] - np.roll(tbl_2_256_4[:, z, :], 1, axis=-1)
            rot_tvs.append(0.5 * np.abs(disc).sum(axis=-1).mean())
        mean_rot_tv = float(np.dot(weights, rot_tvs))

        # 4-cell physical factorial: p[world_s, told_t, m, sector]
        p = np.stack([[tbl_2_256_4[t][codes_rot0[s]] for t in range(2)] for s in range(2)], axis=0)
        m_eff = (p[0, 1] - p[0, 0] + p[1, 1] - p[1, 0]) / 4.0
        w_eff = (p[1, 0] - p[0, 0] + p[1, 1] - p[0, 1]) / 4.0
        i_eff = (p[1, 1] - p[1, 0] - p[0, 1] + p[0, 0]) / 4.0
        factorial_norm = {
            "metadata_M_sq": float(np.mean(np.sum(m_eff * m_eff, axis=-1))),
            "wave_W_sq": float(np.mean(np.sum(w_eff * w_eff, axis=-1))),
            "interaction_I_sq": float(np.mean(np.sum(i_eff * i_eff, axis=-1))),
        }

        cell_nll = np.zeros((2, 2), dtype=np.float64)
        cell_acc = np.zeros((2, 2), dtype=np.float64)
        idx_m = np.arange(n_mothers)
        matched_nll_m = np.zeros(n_mothers, dtype=np.float64)
        swapped_nll_m = np.zeros(n_mothers, dtype=np.float64)
        matched_hit_m = np.zeros(n_mothers, dtype=np.float64)
        swapped_hit_m = np.zeros(n_mothers, dtype=np.float64)

        for s in range(2):
            y = targets_rot0[s]
            for t in range(2):
                pr = p[s, t]
                sc = score_vectors(pr, y)
                cell_nll[s, t] = sc["nll"]
                cell_acc[s, t] = sc["acc"]
                if s == t:
                    matched_nll_m += 0.5 * sc["per_item_nll"]
                    matched_hit_m += 0.5 * sc["per_item_hit"]
                else:
                    swapped_nll_m += 0.5 * sc["per_item_nll"]
                    swapped_hit_m += 0.5 * sc["per_item_hit"]

        diff_blocks = (swapped_nll_m - matched_nll_m).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
        return {
            "orientation_gain": gain,
            "residual_ratio": resid_ratio,
            "mean_sign_tv": mean_sign_tv,
            "mean_rotation_10_tv": mean_rot_tv,
            "matched_acc": float(matched_hit_m.mean()),
            "swapped_acc": float(swapped_hit_m.mean()),
            "matched_nll": float(matched_nll_m.mean()),
            "swapped_nll": float(swapped_nll_m.mean()),
            "cell_nll_s_world_t_told": {
                "--": float(cell_nll[0, 0]),
                "-+": float(cell_nll[0, 1]),
                "+-": float(cell_nll[1, 0]),
                "++": float(cell_nll[1, 1]),
            },
            "cell_acc_s_world_t_told": {
                "--": float(cell_acc[0, 0]),
                "-+": float(cell_acc[0, 1]),
                "+-": float(cell_acc[1, 0]),
                "++": float(cell_acc[1, 1]),
            },
            "swapped_minus_matched_nll_t95": paired_t_ci(diff_blocks),
            "swapped_minus_matched_nll_boot95": block_bootstrap_ci(diff_blocks),
            "factorial_mean_sq_norm": factorial_norm,
        }

    # Physical reference anchor on same mothers
    out["physical_cal_reference"] = _analyze_table(cal_ref)

    for a, aname in enumerate(("phase", "zero")):
        seed_tables = np.stack(
            [softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx])) for s_idx in range(n_seeds)],
            axis=0,
        )
        per_seed_res = []
        for s_idx, seed_val in enumerate(seeds):
            r = _analyze_table(seed_tables[s_idx])
            r["seed"] = int(seed_val)
            per_seed_res.append(r)

        # Average probabilities across 3 seeds for pooled behavioral table summary + mean of seed stats
        mean_tbl = seed_tables.mean(axis=0)
        pooled_tbl_res = _analyze_table(mean_tbl)
        out["arms"][aname] = {
            "seed_mean_orientation_gain": float(np.mean([x["orientation_gain"] for x in per_seed_res])),
            "seed_mean_residual_ratio": float(np.mean([x["residual_ratio"] for x in per_seed_res])),
            "seed_mean_sign_tv": float(np.mean([x["mean_sign_tv"] for x in per_seed_res])),
            "seed_mean_swapped_minus_matched_nll": float(
                np.mean([x["swapped_minus_matched_nll_t95"]["mean"] for x in per_seed_res])
            ),
            "seed_mean_matched_acc": float(np.mean([x["matched_acc"] for x in per_seed_res])),
            "seed_mean_swapped_acc": float(np.mean([x["swapped_acc"] for x in per_seed_res])),
            "seed_averaged_table_metrics": pooled_tbl_res,
            "per_seed": per_seed_res,
        }

    # Single concrete cache spotlight on seed 0 (202611) where phase arm flips its argmax when told +omega vs -omega
    p_s0_tbl = softmax_temperature(raw_ll_table[0, 0], float(temps[0, 0]))
    z_s0_tbl = softmax_temperature(raw_ll_table[1, 0], float(temps[1, 0]))
    # Search real caches appearing in world +1 with opposite argmax under told - vs told +
    best_code = None
    best_tv = -1.0
    for m_idx in range(n_mothers):
        c_pos = int(codes_rot0[1, m_idx])
        c_neg = int(codes_rot0[0, m_idx])
        if c_pos == c_neg:
            arg_neg = int(np.argmax(p_s0_tbl[0, c_pos]))
            arg_pos = int(np.argmax(p_s0_tbl[1, c_pos]))
            if arg_neg != arg_pos and arg_pos == int(targets_rot0[1, m_idx]):
                tv = float(0.5 * np.abs(p_s0_tbl[1, c_pos] - p_s0_tbl[0, c_pos]).sum())
                if tv > best_tv:
                    best_tv = tv
                    best_code = (m_idx, c_pos)
    if best_code is None:
        # Fallback: pick highest sign TV cache appearing in G1
        for m_idx in range(n_mothers):
            c = int(codes_rot0[1, m_idx])
            tv = float(0.5 * np.abs(p_s0_tbl[1, c] - p_s0_tbl[0, c]).sum())
            if tv > best_tv:
                best_tv = tv
                best_code = (m_idx, c)

    m_spot, c_spot = best_code
    z1_sp, z2_sp = divmod(c_spot, 16)
    out["concrete_cache_spotlight_seed0"] = {
        "mother_idx": int(m_spot),
        "cache_code": int(c_spot),
        "z1": int(z1_sp),
        "z2": int(z2_sp),
        "physical_label_neg_world": int(targets_rot0[0, m_spot]),
        "physical_label_pos_world": int(targets_rot0[1, m_spot]),
        "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
        "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
        "phase_arm_told_neg": {
            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 0, c_spot]],
            "probs": [float(v) for v in p_s0_tbl[0, c_spot]],
            "pred": int(np.argmax(p_s0_tbl[0, c_spot])),
        },
        "phase_arm_told_pos": {
            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 1, c_spot]],
            "probs": [float(v) for v in p_s0_tbl[1, c_spot]],
            "pred": int(np.argmax(p_s0_tbl[1, c_spot])),
        },
        "zero_arm_told_neg": {
            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 0, c_spot]],
            "probs": [float(v) for v in z_s0_tbl[0, c_spot]],
            "pred": int(np.argmax(z_s0_tbl[0, c_spot])),
        },
        "zero_arm_told_pos": {
            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 1, c_spot]],
            "probs": [float(v) for v in z_s0_tbl[1, c_spot]],
            "pred": int(np.argmax(z_s0_tbl[1, c_spot])),
        },
    }

    # Single-modality contradiction probe on phase arm (if supplied)
    if contra_text_flipped is not None and contra_phase_flipped is not None:
        # contra_* shapes: (3, 2, 256, 4)
        # Index s=0/1 corresponds to the PHYSICAL world / reference sign;
        # For contra_text_flipped: phase_slots = s, text = 1-s (so compare against matched s vs phase=s)
        # For contra_phase_flipped: text = s, phase_slots = 1-s
        tf_res = []
        pf_res = []
        for s_idx, seed_val in enumerate(seeds):
            t_val = float(temps[0, s_idx])
            p_match = softmax_temperature(raw_ll_table[0, s_idx], t_val)
            p_tflip = softmax_temperature(contra_text_flipped[s_idx], t_val)
            p_pflip = softmax_temperature(contra_phase_flipped[s_idx], t_val)

            def _contra_stats(p_alt: np.ndarray) -> dict:
                # TV vs matched (same physical sign) and TV vs full swapped (opposite sign)
                tv_vs_matched = []
                tv_vs_full_swap = []
                nll_list = []
                hit_list = []
                for s in range(2):
                    pr = p_alt[s][codes_rot0[s]]
                    pr_m = p_match[s][codes_rot0[s]]
                    pr_sw = p_match[1 - s][codes_rot0[s]]
                    tv_vs_matched.append(0.5 * np.abs(pr - pr_m).sum(axis=-1))
                    tv_vs_full_swap.append(0.5 * np.abs(pr - pr_sw).sum(axis=-1))
                    sc = score_vectors(pr, targets_rot0[s])
                    nll_list.append(sc["per_item_nll"])
                    hit_list.append(sc["per_item_hit"])
                return {
                    "seed": int(seed_val),
                    "acc": float(np.mean(hit_list)),
                    "nll": float(np.mean(nll_list)),
                    "mean_tv_vs_matched_both": float(np.mean(tv_vs_matched)),
                    "mean_tv_vs_swapped_both": float(np.mean(tv_vs_full_swap)),
                }

            tf_res.append(_contra_stats(p_tflip))
            pf_res.append(_contra_stats(p_pflip))
        out["single_modality_contradiction_phase_arm"] = {
            "text_flipped_phase_kept": {
                "mean_acc": float(np.mean([x["acc"] for x in tf_res])),
                "mean_nll": float(np.mean([x["nll"] for x in tf_res])),
                "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in tf_res])),
                "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in tf_res])),
                "per_seed": tf_res,
            },
            "phase_flipped_text_kept": {
                "mean_acc": float(np.mean([x["acc"] for x in pf_res])),
                "mean_nll": float(np.mean([x["nll"] for x in pf_res])),
                "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in pf_res])),
                "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in pf_res])),
                "per_seed": pf_res,
            },
        }
    return out


def work_g_rotation_and_controls(
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes: np.ndarray,
    targets: np.ndarray,
    rot10_codes: np.ndarray,
    rot10_targets: np.ndarray,
    indep_raw_ll: np.ndarray,
    indep_targets: np.ndarray,
    shuffled_perm: np.ndarray,
    seeds: Sequence[int],
) -> dict:
    """Work G raw rotation equivariance, independent_x3, shuffled_Z, and offline 16-way Pf / F0."""
    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
    n_mothers = codes.shape[2]
    n_blocks = n_mothers // BLOCK_SIZE
    out: Dict[str, object] = {"arms": {}}

    for a, aname in enumerate(("phase", "zero")):
        raw_hits, raw_nlls = [], []
        pf_hits, pf_nlls = [], []
        f0_hits, f0_nlls = [], []
        shuf_hits, shuf_nlls = [], []
        indep_hits, indep_nlls = [], []
        indep_tvs = []

        rot01_max_abs, rot01_mean_abs, rot01_fail_020 = [], [], []
        rot11_max_abs, rot11_mean_abs, rot11_fail_020 = [], [], []
        rot10_max_abs, rot10_mean_abs, rot10_fail_020 = [], [], []
        worst_rot_dev = None

        for s_idx in range(n_seeds):
            t_val = float(temps[a, s_idx])
            p_raw = softmax_temperature(raw_ll_table[a, s_idx], t_val)  # (2, 256, 4)
            p_pf = compute_16way_group_avg(p_raw)
            p_f0 = compute_representative_readout(p_raw)
            p_ind = softmax_temperature(indep_raw_ll[a, s_idx], t_val)

            for sign_i in range(n_signs):
                pr_rot = p_raw[sign_i][codes[sign_i]]     # (3, M, 4)
                pf_rot = p_pf[sign_i][codes[sign_i]]
                f0_rot = p_f0[sign_i][codes[sign_i]]
                pr_10 = p_raw[sign_i][rot10_codes[sign_i]]

                # Rotation checks vs rot=(0,0)
                d01 = np.abs(pr_rot[1] - pr_rot[0]).max(axis=1)
                d11 = np.abs(pr_rot[2] - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)
                d10 = np.abs(pr_10 - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)

                rot01_max_abs.append(float(d01.max()))
                rot01_mean_abs.append(float(d01.mean()))
                rot01_fail_020.append(float((d01 > 0.20).mean()))

                rot11_max_abs.append(float(d11.max()))
                rot11_mean_abs.append(float(d11.mean()))
                rot11_fail_020.append(float((d11 > 0.20).mean()))

                rot10_max_abs.append(float(d10.max()))
                rot10_mean_abs.append(float(d10.mean()))
                rot10_fail_020.append(float((d10 > 0.20).mean()))

                idx_worst = int(np.argmax(d11))
                if worst_rot_dev is None or float(d11[idx_worst]) > worst_rot_dev["max_abs_diff"]:
                    worst_rot_dev = {
                        "seed": int(seeds[s_idx]),
                        "sign_idx": int(sign_i),
                        "mother_idx": idx_worst,
                        "code_00": int(codes[sign_i, 0, idx_worst]),
                        "code_11": int(codes[sign_i, 2, idx_worst]),
                        "probs_00": [float(v) for v in pr_rot[0, idx_worst]],
                        "probs_11": [float(v) for v in pr_rot[2, idx_worst]],
                        "max_abs_diff": float(d11[idx_worst]),
                    }

                sc_raw = score_vectors(pr_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_pf = score_vectors(pf_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_f0 = score_vectors(f0_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                raw_hits.append(sc_raw["per_item_hit"].reshape(3, n_mothers))
                raw_nlls.append(sc_raw["per_item_nll"].reshape(3, n_mothers))
                pf_hits.append(sc_pf["per_item_hit"].reshape(3, n_mothers))
                pf_nlls.append(sc_pf["per_item_nll"].reshape(3, n_mothers))
                f0_hits.append(sc_f0["per_item_hit"].reshape(3, n_mothers))
                f0_nlls.append(sc_f0["per_item_nll"].reshape(3, n_mothers))

                # Shuffled_Z control
                shuf_codes = np.take_along_axis(codes[sign_i], shuffled_perm[sign_i], axis=1)
                pr_shuf = p_raw[sign_i][shuf_codes]
                sc_shuf = score_vectors(pr_shuf.reshape(-1, 4), targets[sign_i].reshape(-1))
                shuf_hits.append(sc_shuf["per_item_hit"])
                shuf_nlls.append(sc_shuf["per_item_nll"])

                # independent_x3 control
                pr_ind = p_ind[sign_i][codes[sign_i]]
                sc_ind = score_vectors(pr_ind.reshape(-1, 4), indep_targets[sign_i].reshape(-1))
                indep_hits.append(sc_ind["per_item_hit"])
                indep_nlls.append(sc_ind["per_item_nll"])
                indep_tvs.append(0.5 * np.abs(pr_ind - 0.25).sum(axis=-1))

        # Block paired gain of offline 16-way group average vs raw host
        raw_m_hit = np.stack(raw_hits, axis=0).mean(axis=(0, 1))
        raw_m_nll = np.stack(raw_nlls, axis=0).mean(axis=(0, 1))
        pf_m_hit = np.stack(pf_hits, axis=0).mean(axis=(0, 1))
        pf_m_nll = np.stack(pf_nlls, axis=0).mean(axis=(0, 1))
        pf_blk_acc_gain = (pf_m_hit - raw_m_hit).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
        pf_blk_nll_improv = (raw_m_nll - pf_m_nll).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)

        out["arms"][aname] = {
            "raw_rotation_checks": {
                "rot01_hold": {
                    "mean_max_abs": float(np.mean(rot01_mean_abs)),
                    "worst_max_abs": float(np.max(rot01_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot01_fail_020)),
                },
                "rot11_roll": {
                    "mean_max_abs": float(np.mean(rot11_mean_abs)),
                    "worst_max_abs": float(np.max(rot11_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot11_fail_020)),
                },
                "rot10_roll": {
                    "mean_max_abs": float(np.mean(rot10_mean_abs)),
                    "worst_max_abs": float(np.max(rot10_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot10_fail_020)),
                },
                "worst_rot11_example": worst_rot_dev,
            },
            "controls": {
                "independent_x3": {
                    "acc": float(np.mean(indep_hits)),
                    "nll": float(np.mean(indep_nlls)),
                    "mean_tv_from_uniform": float(np.mean(indep_tvs)),
                    "max_tv_from_uniform": float(np.max(indep_tvs)),
                },
                "shuffled_Z": {
                    "acc": float(np.mean(shuf_hits)),
                    "nll": float(np.mean(shuf_nlls)),
                },
            },
            "raw_host": {"acc": float(np.mean(raw_hits)), "nll": float(np.mean(raw_nlls))},
            "offline_group_avg_16way": {
                "acc": float(np.mean(pf_hits)),
                "nll": float(np.mean(pf_nlls)),
                "pf_minus_raw_acc_t95": paired_t_ci(pf_blk_acc_gain),
                "raw_minus_pf_nll_t95": paired_t_ci(pf_blk_nll_improv),
            },
            "offline_representative_F0": {
                "acc": float(np.mean(f0_hits)),
                "nll": float(np.mean(f0_nlls)),
            },
        }
    return out


def work_a_q2_analysis(q2_npz, temps_q2: np.ndarray, seeds: Sequence[int]) -> dict:
    """Evaluate all 7 Q2 arms on 512 fresh C4 (omega=+0.35, 2dB, tp=4) mothers vs Work A kernel."""
    raw_ll = q2_npz["q2__raw_ll"]             # (2, 3, 7, M, 4)
    labels = q2_npz["q2__labels"]             # (M,)
    z1 = q2_npz["q2__z1"]
    z2 = q2_npz["q2__z2"]
    work_a_probs = q2_npz["q2__work_a_ref_probs"]  # (6, M, 4)
    n_arms, n_seeds, n_q2_arms, n_mothers, _ = raw_ll.shape
    assert n_q2_arms == len(Q2_ARMS)

    probs_all = np.zeros_like(raw_ll, dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            probs_all[a, s_idx] = softmax_temperature(raw_ll[a, s_idx], float(temps_q2[a, s_idx]))

    # Physical Work A kernel baselines
    work_a_metrics = {}
    work_a_per_arm = {}
    for arm_name, ref_idx in Q2_LOGICAL_REF_IDX.items():
        sc = score_vectors(work_a_probs[ref_idx], labels)
        work_a_per_arm[arm_name] = sc
        work_a_metrics[arm_name] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece")}
    work_a_fresh_diff_acc = work_a_per_arm["fresh_resample"]["per_item_hit"] - work_a_per_arm["valid"]["per_item_hit"]
    work_a_fresh_diff_nll = work_a_per_arm["fresh_resample"]["per_item_nll"] - work_a_per_arm["valid"]["per_item_nll"]
    work_a_metrics["fresh_minus_valid"] = {
        "delta_acc": float(work_a_fresh_diff_acc.mean()),
        "delta_acc_se": float(work_a_fresh_diff_acc.std(ddof=1) / math.sqrt(n_mothers)),
        "delta_nll": float(work_a_fresh_diff_nll.mean()),
        "delta_nll_se": float(work_a_fresh_diff_nll.std(ddof=1) / math.sqrt(n_mothers)),
    }

    arms_out = {}
    for a, aname in enumerate(("phase", "zero")):
        by_q2_arm = {}
        per_seed_q2 = []
        # Pool 3 seeds cleanly
        for qi, qarm in enumerate(Q2_ARMS):
            flat_p = probs_all[a, :, qi].reshape(-1, 4)
            flat_y = np.tile(labels, n_seeds)
            sc = score_vectors(flat_p, flat_y)
            by_q2_arm[qarm] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece", "tie_count")}

        # Paired comparisons across (seed x mother)
        def _pair_delta(arm_b: str, arm_a: str) -> dict:
            ia = Q2_ARMS.index(arm_a)
            ib = Q2_ARMS.index(arm_b)
            pa = probs_all[a, :, ia].reshape(-1, 4)
            pb = probs_all[a, :, ib].reshape(-1, 4)
            ya = np.tile(labels, n_seeds)
            sca = score_vectors(pa, ya)
            scb = score_vectors(pb, ya)
            tv = 0.5 * np.abs(pb - pa).sum(axis=-1)
            max_abs = np.max(np.abs(pb - pa), axis=-1)
            conf_inc = pb.max(axis=-1) - pa.max(axis=-1)
            hit_delta = scb["per_item_hit"] - sca["per_item_hit"]
            nll_delta = scb["per_item_nll"] - sca["per_item_nll"]
            # Mother-averaged SE across mothers (clustering seeds per mother!)
            hit_m = hit_delta.reshape(n_seeds, n_mothers).mean(axis=0)
            nll_m = nll_delta.reshape(n_seeds, n_mothers).mean(axis=0)
            return {
                "delta_acc": float(hit_m.mean()),
                "delta_acc_mother_se": float(hit_m.std(ddof=1) / math.sqrt(n_mothers)),
                "delta_nll": float(nll_m.mean()),
                "delta_nll_mother_se": float(nll_m.std(ddof=1) / math.sqrt(n_mothers)),
                "mean_tv": float(tv.mean()),
                "max_tv": float(tv.max()),
                "max_abs_prob_diff": float(max_abs.max()),
                "max_conf_increase": float(conf_inc.max()),
                "corrected": int(np.sum(hit_delta > 0)),
                "new_errors": int(np.sum(hit_delta < 0)),
            }

        for s_idx, seed_val in enumerate(seeds):
            s_dict = {"seed": int(seed_val), "arms": {}}
            for qi, qarm in enumerate(Q2_ARMS):
                sc_s = score_vectors(probs_all[a, s_idx, qi], labels)
                s_dict["arms"][qarm] = {k: sc_s[k] for k in ("acc", "nll", "brier", "ece")}
            per_seed_q2.append(s_dict)

        arms_out[aname] = {
            "by_arm": by_q2_arm,
            "paired_contrasts": {
                "valid_minus_none": _pair_delta("valid", "none"),
                "independent_only_minus_none": _pair_delta("independent_only", "none"),
                "duplicate_copy_dedup_minus_valid": _pair_delta("duplicate_copy_dedup", "valid"),
                "duplicate_copy_raw_minus_valid": _pair_delta("duplicate_copy_raw", "valid"),
                "valid_plus_independent_minus_valid": _pair_delta("valid_plus_independent", "valid"),
                "fresh_resample_minus_valid": _pair_delta("fresh_resample", "valid"),
            },
            "per_seed": per_seed_q2,
        }

    # Concrete multi-arm trajectory example from seed 0 where Work A valid->fresh helps and host phase arm corrects
    p_s0 = probs_all[0, 0]  # (7, M, 4)
    idx_ex = 0
    for m_i in range(n_mothers):
        pred_none = int(np.argmax(p_s0[Q2_ARMS.index("none"), m_i]))
        pred_val = int(np.argmax(p_s0[Q2_ARMS.index("valid"), m_i]))
        pred_fresh = int(np.argmax(p_s0[Q2_ARMS.index("fresh_resample"), m_i]))
        truth = int(labels[m_i])
        if pred_none != truth and pred_val == truth and pred_fresh == truth:
            idx_ex = m_i
            break

    ex_traj = {
        "mother_idx": int(idx_ex),
        "z1": int(z1[idx_ex]),
        "z2": int(z2[idx_ex]),
        "true_x1": int(labels[idx_ex]),
        "phase_seed0_probs_by_arm": {
            qarm: [float(v) for v in probs_all[0, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
        },
        "zero_seed0_probs_by_arm": {
            qarm: [float(v) for v in probs_all[1, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
        },
        "work_a_ref_probs": {
            qarm: [float(v) for v in work_a_probs[ridx, idx_ex]] for qarm, ridx in Q2_LOGICAL_REF_IDX.items()
        },
    }

    return {
        "condition": {"omega": 0.35, "h": 2, "ell": 2, "t_p": 4.0, "snr_db": 2.0, "sigma_phi": 0.10},
        "mothers": int(n_mothers),
        "work_a_physical_reference": work_a_metrics,
        "arms": arms_out,
        "concrete_trajectory_example": ex_traj,
    }


def recompute_all(data_dir: Path, verify_against: Optional[Path] = None) -> dict:
    """Recompute complete summary_master.json solely from exported .npz raw scores in data_dir."""
    q3_npz = np.load(data_dir / "blind_q3_scores.npz", allow_pickle=False)
    q2_npz = np.load(data_dir / "blind_q2_scores.npz", allow_pickle=False)

    temps_q3 = np.asarray(q3_npz["temps_q3"], dtype=np.float64)
    temps_q2 = np.asarray(q2_npz["temps_q2"], dtype=np.float64)
    seeds = [int(x) for x in q3_npz["seeds"].tolist()]

    cells_res = {}
    for cid in CELL_ORDER:
        cells_res[cid] = {
            **CELL_META[cid],
            **evaluate_cell_paired(
                raw_ll_table=q3_npz[f"{cid}__raw_ll"],
                temps=temps_q3,
                codes=q3_npz[f"{cid}__codes"],
                targets=q3_npz[f"{cid}__targets"],
                direct_probs=q3_npz[f"{cid}__direct_table_probs"],
                full_iq_probs=q3_npz[f"{cid}__full_iq_probs"],
                seeds=seeds,
                is_primary=(cid == "G1_main_extrap_both"),
            ),
        }

    # Work F on primary +-0.40 and historical +-0.35
    work_f_primary = work_f_sign_probe(
        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
        temps=temps_q3,
        codes_rot0=q3_npz["G1_main_extrap_both__codes"][:, 0, :],
        targets_rot0=q3_npz["G1_main_extrap_both__targets"][:, 0, :],
        cal_ref=q3_npz["G1_main_extrap_both__cal_ref"],
        cal_mass=q3_npz["G1_main_extrap_both__cal_mass"],
        seeds=seeds,
        contra_text_flipped=q3_npz["G1_main_extrap_both__contra_text_flipped_raw_ll"] if "G1_main_extrap_both__contra_text_flipped_raw_ll" in q3_npz else None,
        contra_phase_flipped=q3_npz["G1_main_extrap_both__contra_phase_flipped_raw_ll"] if "G1_main_extrap_both__contra_phase_flipped_raw_ll" in q3_npz else None,
    )
    work_f_hist = work_f_sign_probe(
        raw_ll_table=q3_npz["H_c4_035__raw_ll"],
        temps=temps_q3,
        codes_rot0=q3_npz["H_c4_035__codes"][:, 0, :],
        targets_rot0=q3_npz["H_c4_035__targets"][:, 0, :],
        cal_ref=q3_npz["H_c4_035__cal_ref"],
        cal_mass=q3_npz["H_c4_035__cal_mass"],
        seeds=seeds,
    )

    # Work G on primary G1
    work_g = work_g_rotation_and_controls(
        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
        temps=temps_q3,
        codes=q3_npz["G1_main_extrap_both__codes"],
        targets=q3_npz["G1_main_extrap_both__targets"],
        rot10_codes=q3_npz["G1_main_extrap_both__rot10_codes"],
        rot10_targets=q3_npz["G1_main_extrap_both__rot10_targets"],
        indep_raw_ll=q3_npz["G1_main_extrap_both__indep_raw_ll"],
        indep_targets=q3_npz["G1_main_extrap_both__indep_targets"],
        shuffled_perm=q3_npz["G1_main_extrap_both__shuffled_perm"],
        seeds=seeds,
    )

    # Work A Q2
    work_a_q2 = work_a_q2_analysis(q2_npz, temps_q2, seeds)

    # Replay C1/C2/C3 forgetting check (if present)
    replay_c123 = {}
    replay_path = data_dir / "replay_c123_scores.npz"
    if replay_path.exists():
        rp = np.load(replay_path, allow_pickle=False)
        for cname in ("C1_coherent_2db", "C2_discrete_walk", "C3_mid_pilot_cont"):
            if f"{cname}__q3_raw_ll" not in rp:
                continue
            replay_c123[cname] = {}
            for qk in ("q1", "q3", "q2v"):
                lbl = rp[f"{cname}__{qk}_labels"]
                raw = rp[f"{cname}__{qk}_raw_ll"]  # (2, 3, N, 4)
                t_mat = temps_q2 if qk == "q2v" else temps_q3
                arm_res = {}
                for a, aname in enumerate(("phase", "zero")):
                    ps = np.concatenate(
                        [softmax_temperature(raw[a, s_idx], float(t_mat[a, s_idx])) for s_idx in range(len(seeds))],
                        axis=0,
                    )
                    sc = score_vectors(ps, np.tile(lbl, len(seeds)))
                    arm_res[aname] = {k: sc[k] for k in ("acc", "nll", "brier", "ece")}
                sc_d = score_vectors(rp[f"{cname}__{qk}_direct_probs"], lbl)
                arm_res["direct_s17"] = {k: sc_d[k] for k in ("acc", "nll", "brier", "ece")}
                replay_c123[cname][qk] = arm_res

    summary = {
        "seeds": seeds,
        "locked_temps_q3": temps_q3.tolist(),
        "locked_temps_q2": temps_q2.tolist(),
        "cells": cells_res,
        "work_f_primary_040": work_f_primary,
        "work_f_historical_035": work_f_hist,
        "work_g_primary_040": work_g,
        "work_a_q2_c4": work_a_q2,
        "replay_c123_forgetting": replay_c123,
    }

    if verify_against is not None:
        expected = json.loads(verify_against.read_text(encoding="utf-8"))
        max_diff = _compare_nested(expected, summary)
        summary["_verification"] = {"verified_against": str(verify_against), "max_abs_diff": max_diff}
    return summary


def _compare_nested(expected, actual, path: str = "root") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"Type mismatch at {path}")
        max_d = 0.0
        for k, v in expected.items():
            if path == "root" and k in ("_verification", "historical_c4_035_lookup_anchor"):
                continue
            if k not in actual:
                raise AssertionError(f"Missing key {path}.{k}")
            max_d = max(max_d, _compare_nested(v, actual[k], f"{path}.{k}"))
        return max_d
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"List mismatch at {path}")
        return max((_compare_nested(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(expected, actual))), default=0.0)
    if isinstance(expected, float):
        d = abs(float(expected) - float(actual))
        if not math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-10):
            raise AssertionError(f"Float mismatch at {path}: {expected} vs {actual} (diff={d})")
        return d
    if expected != actual:
        raise AssertionError(f"Value mismatch at {path}: {expected!r} vs {actual!r}")
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify-against", type=Path, default=None)
    args = ap.parse_args()
    summary = recompute_all(args.data_dir.resolve(), verify_against=args.verify_against)
    if args.out is not None:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.verify_against is not None:
        print(f"VERIFIED OK: max_abs_diff={summary['_verification']['max_abs_diff']:.3e}")
    else:
        g1 = summary["cells"]["G1_main_extrap_both"]
        print(
            json.dumps(
                {
                    "G1_phase": g1["arms"]["phase"],
                    "G1_zero": g1["arms"]["zero"],
                    "G1_paired_acc_gain_t95": g1["paired_phase_minus_zero"]["acc_gain_t95"],
                    "G1_paired_nll_improv_t95": g1["paired_phase_minus_zero"]["nll_improv_t95"],
                    "G1_dual_threshold": g1["dual_practical_threshold"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
