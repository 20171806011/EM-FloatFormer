"""Deterministic Stage 3D data generators, disjoint seed namespaces, and zero-leakage Chinese prompts."""
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.stage3_ministral_cache.data import MotherBatch, generate_mothers, orbit_key
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ConditionSpec,
    N_SAMPLES,
    Q1_CHOICES,
    Q1_PROMPT,
    Q2_CHOICES,
    Q2_PROMPT,
    Q3_CHOICES,
    Q3_PROMPT_INDEP,
    Q3_PROMPT_REPEAT,
    Q3_PROMPT_REPEAT_AFTER_PILOT,
    ROOTS,
    ROTATIONS,
    Reference,
    dedup_references,
    matched_sum,
    sector_of_complex,
)


EQUIVARIANT_B8_S17_SHA256 = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
ARM_NAMES: Tuple[str, ...] = ("phase", "zero")
CONNECTOR_SEEDS: Tuple[int, ...] = (202611, 202622, 202633)
REPLAY_TRAIN_CONDITIONS: Tuple[str, ...] = ("C1_coherent_2db", "C2_discrete_walk", "C3_mid_pilot_cont")
TRAIN_ARMS: Tuple[str, ...] = ("none", "valid", "independent", "fresh_resample")

# Disjoint seed namespaces (strictly isolated from Stage 3 s3_train/s3_dev/s3_test and Stage 3C work_e 316027)
NS_CFO_TRAIN = 930101
NS_CFO_DEV = 930202
NS_DEV_PILOT = 940202
NS_BLIND_G1 = 950401
NS_BLIND_G2 = 950402
NS_BLIND_G3 = 950403
NS_BLIND_G4 = 950404
NS_BLIND_Q2_C4 = 950405
NS_REF_CALIB = 960501

TRAIN_TIMINGS: Tuple[Tuple[int, int, float], ...] = ((2, 2, 4.0), (2, 3, 1.0))
TRAIN_OMEGAS: Tuple[float, ...] = (-0.25, -0.10, 0.0, 0.10, 0.25)
DEV_OMEGAS: Tuple[float, ...] = (-0.18, 0.18)

Q2_BLIND_ARMS: Tuple[str, ...] = (
    "none",
    "valid",
    "duplicate_copy_dedup",
    "duplicate_copy_raw",
    "independent_only",
    "valid_plus_independent",
    "fresh_resample",
)


@dataclass(frozen=True)
class BlindCellSpec:
    cell_id: str
    role: str
    abs_omega: float
    h: int
    ell: int
    t_p: float
    snr_db: float
    sigma_phi: float
    namespace: int
    default_blocks: int

    @property
    def total_delay(self) -> int:
        return int(self.h + self.ell)

    @property
    def cum_angle_abs(self) -> float:
        return float(self.abs_omega * (self.h + self.ell))

    @property
    def cum_diffusion_var(self) -> float:
        return float((self.sigma_phi ** 2) * (self.h + self.ell))


BLIND_CELLS: Dict[str, BlindCellSpec] = {
    "G1_main_extrap_both": BlindCellSpec(
        cell_id="G1_main_extrap_both",
        role="primary_extrap_freq_and_angle",
        abs_omega=0.40,
        h=2,
        ell=2,
        t_p=4.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G1,
        default_blocks=131,
    ),
    "G2_interp_both": BlindCellSpec(
        cell_id="G2_interp_both",
        role="explore_interp_both",
        abs_omega=0.20,
        h=2,
        ell=2,
        t_p=4.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G2,
        default_blocks=8,
    ),
    "G3_extrap_angle": BlindCellSpec(
        cell_id="G3_extrap_angle",
        role="explore_extrap_angle",
        abs_omega=0.20,
        h=4,
        ell=4,
        t_p=8.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G3,
        default_blocks=8,
    ),
    "G4_extrap_freq": BlindCellSpec(
        cell_id="G4_extrap_freq",
        role="explore_extrap_freq",
        abs_omega=0.40,
        h=1,
        ell=1,
        t_p=2.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G4,
        default_blocks=8,
    ),
}


def load_s17_encoder(repo_root: Optional[Path] = None):
    """Load frozen equivariant_b8_s17 encoder and enforce exact SHA-256 fingerprint."""
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    matched_dir = root / "research" / "matched_cache"
    weight_path = matched_dir / "weights" / "equivariant_b8_s17.npz"
    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if digest != EQUIVARIANT_B8_S17_SHA256:
        raise RuntimeError(f"s17 SHA-256 mismatch: {digest} != {EQUIVARIANT_B8_S17_SHA256}")
    if str(matched_dir) not in sys.path:
        sys.path.insert(0, str(matched_dir))
    import learned  # type: ignore

    return learned.load(weight_path), digest


def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
    """Six public phase slots at t = (h, h + ell, t_p) relative to t1 = 0."""
    if zero_arm:
        return np.zeros(6, dtype=np.float32)
    times = (float(h), float(h + ell), float(t_p))
    return np.array([v for t in times for v in (math.cos(omega * t), math.sin(omega * t))], dtype=np.float32)


def condition_spec_from_params(
    omega: float,
    h: int,
    ell: int,
    t_p: float,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    tag: str = "cfo",
) -> ConditionSpec:
    return ConditionSpec(
        name=tag,
        snr_db=float(snr_db),
        h=int(h),
        ell=int(ell),
        t_p=float(t_p),
        drift_type="cfo_continuous",
        d=0.0,
        sigma_phi=float(sigma_phi),
        omega=float(omega),
        in_support_encoder=False,
        in_support_connector=False,
    )


def format_public_metadata(cond: ConditionSpec) -> str:
    """Zero-leakage public metadata string (excludes condition names, split names, labels, and noise)."""
    if cond.drift_type == "coherent":
        drift_desc = "相干静止(omega=+0.00rad/步,sigma=0.00rad/√步,d=0.00/步)"
    elif cond.drift_type == "discrete":
        drift_desc = f"离散C4游走(omega=+0.00rad/步,sigma=0.00rad/√步,d={cond.d:.2f}/步)"
    else:
        drift_desc = f"连续频偏与布朗漂移(omega={cond.omega:+.2f}rad/步,sigma={cond.sigma_phi:.2f}rad/√步,d=0.00/步)"
    return (
        f"信噪比={cond.snr_db:.1f}dB; "
        f"两窗间隔h={cond.h}, 留出间隔ell={cond.ell}, 导频时刻tp={cond.t_p:.1f}; "
        f"信道={drift_desc}。\n"
        f"已编码波形与公开相位向量：<W1><W2><PHASE>。"
    )


def build_stage3d_prompt(
    question_type: str,
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    references: Sequence[Reference] = (),
    deduplicate: bool = True,
) -> Tuple[str, Tuple[str, ...]]:
    """Construct exact Chinese prompt listing all four candidate choices without any label/ID leakage."""
    if question_type not in ("Q1", "Q2", "Q3", "Q3_after_pilot"):
        raise ValueError(f"Unknown question_type={question_type}")

    meta = format_public_metadata(cond)
    ref_block = ""
    if question_type in ("Q2", "Q3_after_pilot"):
        if len(references) == 0:
            ref_block = "\n补充参考：无参考观测。"
        elif deduplicate:
            seen_ids = set()
            items = []
            for r in references:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                tag = "同信道" if r.shares_channel else "独立无关"
                items.append(f"[编号={r.acquisition_id},状态={tag},时刻={r.t_p:.1f}]<REF>")
            ref_block = "\n补充参考：" + "；".join(items) + f"（已按唯一采集编号去重，共{len(items)}条独立参考）。"
        else:
            items = []
            for r in references:
                tag = "同信道" if r.shares_channel else "独立无关"
                items.append(f"[编号={r.acquisition_id},状态={tag},时刻={r.t_p:.1f}]<REF>")
            ref_block = f"\n补充参考（未去重，共{len(items)}条）：" + "；".join(items) + "。"

    if question_type == "Q1":
        qtext, choices = Q1_PROMPT, Q1_CHOICES
    elif question_type == "Q2":
        qtext, choices = Q2_PROMPT, Q2_CHOICES
    elif question_type == "Q3":
        qtext = Q3_PROMPT_REPEAT if protocol == "repeat_x1" else Q3_PROMPT_INDEP
        choices = Q3_CHOICES
    else:
        qtext, choices = Q3_PROMPT_REPEAT_AFTER_PILOT, Q3_CHOICES

    listed = "\n".join(choices)
    prompt = (
        f"{meta}{ref_block}\n"
        f"问题：{qtext}\n"
        f"四个候选选项（按物理顺序）：\n{listed}\n"
        f"请直接从以上四项中输出最可能的一项完整选项文字。"
    )
    return prompt, choices


def derive_block_seeds(namespace: int, cell_idx: int, blocks: int) -> List[int]:
    """Deterministically derive unique 32-bit block seeds from (namespace, cell_idx, block_id)."""
    return [
        int(np.random.SeedSequence([int(namespace), int(cell_idx), int(b)]).generate_state(1)[0])
        for b in range(int(blocks))
    ]


def generate_paired_cfo_blocks(
    namespace: int,
    cell_idx: int,
    blocks: int,
    h: int,
    ell: int,
    t_p: float,
    omegas: Sequence[float],
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    split_prefix: str = "cfo",
) -> Tuple[Dict[float, MotherBatch], List[int]]:
    """Generate paired MotherBatches sharing balanced triples, Brownian innovations, and additive noises across omegas."""
    block_seeds = derive_block_seeds(namespace, cell_idx, blocks)
    triples_base = list(itertools.product(range(4), repeat=3))
    snr = 10.0 ** (float(snr_db) / 10.0)
    noise_sd = math.sqrt(1.0 / (2.0 * snr))
    norm = math.sqrt(float(N_SAMPLES))
    times = sorted({0.0, float(h), float(h + ell), float(t_p)})
    m_block = 64
    m_total = int(blocks) * m_block

    x1_parts, x2_parts, g_parts, x3_parts, go_parts = [], [], [], [], []
    z_inc_parts: Dict[float, List[np.ndarray]] = {t: [] for t in times[1:]}
    phi_ind_parts = []
    noise_parts = {k: [] for k in ("y1", "y2", "y3_rep", "y3_ind", "p1", "p2", "pind")}

    for b_idx, seed in enumerate(block_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(m_block)
        arr_tr = np.asarray([triples_base[int(p)] for p in perm], dtype=np.int64)
        x1_parts.append(arr_tr[:, 0])
        x2_parts.append(arr_tr[:, 1])
        g_parts.append(arr_tr[:, 2])
        x3_parts.append(rng.integers(4, size=m_block, dtype=np.int64))
        go_parts.append(rng.integers(4, size=m_block, dtype=np.int64))
        for t_cur in times[1:]:
            z_inc_parts[t_cur].append(rng.normal(scale=1.0, size=m_block))
        phi_ind_parts.append(rng.uniform(-math.pi, math.pi, size=m_block))
        for key in noise_parts:
            noise_parts[key].append(
                rng.normal(scale=noise_sd, size=(m_block, N_SAMPLES))
                + 1j * rng.normal(scale=noise_sd, size=(m_block, N_SAMPLES))
            )

    x1 = np.concatenate(x1_parts, axis=0)
    x2 = np.concatenate(x2_parts, axis=0)
    g = np.concatenate(g_parts, axis=0)
    x3_indep = np.concatenate(x3_parts, axis=0)
    g_other = np.concatenate(go_parts, axis=0)
    phi_indep = np.concatenate(phi_ind_parts, axis=0)
    z_incs = {t: np.concatenate(v, axis=0) for t, v in z_inc_parts.items()}
    noises = {k: np.concatenate(v, axis=0) for k, v in noise_parts.items()}

    out: Dict[float, MotherBatch] = {}
    for w_raw in omegas:
        omega = float(w_raw)
        cond_tag = f"{split_prefix}_w{omega:+.2f}_h{h}_l{ell}_tp{t_p:.1f}"
        cond = condition_spec_from_params(omega, h, ell, t_p, snr_db=snr_db, sigma_phi=sigma_phi, tag=cond_tag)
        cont_map = {times[0]: np.zeros(m_total, dtype=np.float64)}
        for t_prev, t_cur in zip(times[:-1], times[1:]):
            dt = float(t_cur - t_prev)
            mean = omega * dt
            sd_p = sigma_phi * math.sqrt(max(0.0, dt))
            cont_map[t_cur] = cont_map[t_prev] + mean + (sd_p * z_incs[t_cur] if sd_p > 0.0 else 0.0)
        phi_map = {t: cont_map[t] - cont_map[0.0] for t in times}
        phi_2 = phi_map[float(h)]
        phi_3 = phi_map[float(h + ell)]
        phi_p = phi_map[float(t_p)]

        def obs(u_labels: np.ndarray, phi: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
            sig = (ROOTS[u_labels % 4] * np.exp(1j * phi)) / norm
            return sig[:, None] + n_arr

        y1 = obs(x1 + g, phi_map[0.0], noises["y1"])
        y2 = obs(x2 + g, phi_2, noises["y2"])
        y3_repeat = obs(x1 + g, phi_3, noises["y3_rep"])
        y3_indep = obs(x3_indep + g, phi_3, noises["y3_ind"])
        p_valid1 = obs(g, phi_p, noises["p1"])
        p_valid2 = obs(g, phi_p, noises["p2"])
        p_indep = obs(g_other, phi_indep, noises["pind"])
        mother_ids = [f"{cond_tag}:{i}" for i in range(m_total)]

        out[omega] = MotherBatch(
            split=split_prefix,
            condition=cond,
            mother_ids=mother_ids,
            x1=x1,
            x2=x2,
            g=g,
            x3_indep=x3_indep,
            y1=y1,
            y2=y2,
            y3_repeat=y3_repeat,
            y3_indep=y3_indep,
            p_valid1=p_valid1,
            p_valid2=p_valid2,
            p_indep=p_indep,
            phi_2=phi_2,
            phi_3=phi_3,
            phi_p=phi_p,
        )
    return out, block_seeds


def build_q2_arm_references(batch: MotherBatch, arm: str, coord_rot: int = 0) -> Tuple[Tuple[Reference, ...], bool]:
    """Return `(references_tuple, deduplicate_flag)` for the 7 Stage 3D Q2 evaluation arms."""
    r = ROOTS[coord_rot % 4]
    tp = batch.condition.t_p
    snr_db = batch.condition.snr_db
    if arm == "none":
        return (), True
    if arm == "valid":
        return (Reference(0, True, batch.p_valid1 * r, tp, snr_db),), True
    if arm == "duplicate_copy_dedup":
        ref0 = Reference(0, True, batch.p_valid1 * r, tp, snr_db)
        return (ref0, ref0), True
    if arm == "duplicate_copy_raw":
        ref0 = Reference(0, True, batch.p_valid1 * r, tp, snr_db)
        return (ref0, ref0), False
    if arm == "independent_only":
        return (Reference(0, False, batch.p_indep * r, tp, snr_db),), True
    if arm == "valid_plus_independent":
        return (
            Reference(0, True, batch.p_valid1 * r, tp, snr_db),
            Reference(1, False, batch.p_indep * r, tp, snr_db),
        ), True
    if arm == "fresh_resample":
        return (
            Reference(0, True, batch.p_valid1 * r, tp, snr_db),
            Reference(1, True, batch.p_valid2 * r, tp, snr_db),
        ), True
    raise ValueError(f"Unknown Q2 arm: {arm}")


def build_blind_seed_manifest(g1_blocks: int = 131) -> dict:
    """Create canonical manifest of all blind namespaces and block seeds before unblinding."""
    cells_manifest = {}
    for cid, spec in BLIND_CELLS.items():
        n_blk = int(g1_blocks) if cid == "G1_main_extrap_both" else int(spec.default_blocks)
        seeds = derive_block_seeds(spec.namespace, 0, n_blk)
        cells_manifest[cid] = {
            "namespace": spec.namespace,
            "blocks": n_blk,
            "mothers": n_blk * 64,
            "abs_omega": spec.abs_omega,
            "h": spec.h,
            "ell": spec.ell,
            "t_p": spec.t_p,
            "snr_db": spec.snr_db,
            "sigma_phi": spec.sigma_phi,
            "block_seeds": seeds,
            "block_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode("utf-8")).hexdigest(),
        }
    q2_seeds = derive_block_seeds(NS_BLIND_Q2_C4, 0, 8)
    cells_manifest["Q2_C4_blind"] = {
        "namespace": NS_BLIND_Q2_C4,
        "blocks": 8,
        "mothers": 512,
        "omega": 0.35,
        "h": 2,
        "ell": 2,
        "t_p": 4.0,
        "snr_db": 2.0,
        "sigma_phi": 0.10,
        "block_seeds": q2_seeds,
        "block_seeds_sha256": hashlib.sha256(json.dumps(q2_seeds).encode("utf-8")).hexdigest(),
    }
    full_sha = hashlib.sha256(json.dumps(cells_manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "manifest_sha256": full_sha,
        "cells": cells_manifest,
    }


def verify_stage3d_orbit_isolation(max_blocks: int = 2) -> Dict[str, object]:
    """Check zero C4-orbit overlap among CFO train, CFO dev, dev pilot, and blind namespaces."""
    sets: Dict[str, set] = {}

    def _add_batches(name: str, batches: Iterable[MotherBatch]):
        s = sets.setdefault(name, set())
        for b in batches:
            for i in range(len(b.x1)):
                s.add(orbit_key(b.y1[i]))
                s.add(orbit_key(b.y2[i]))
                s.add(orbit_key(b.y3_repeat[i]))
                s.add(orbit_key(b.p_valid1[i]))

    for t_idx, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        tr_map, _ = generate_paired_cfo_blocks(NS_CFO_TRAIN, t_idx, max_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train")
        _add_batches("cfo_train", tr_map.values())
        dv_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, t_idx, max_blocks, h, ell, tp, DEV_OMEGAS, split_prefix="cfo_dev")
        _add_batches("cfo_dev", dv_map.values())

    pilot_map, _ = generate_paired_cfo_blocks(NS_DEV_PILOT, 0, max_blocks, 2, 2, 4.0, DEV_OMEGAS, split_prefix="dev_pilot")
    _add_batches("dev_pilot", pilot_map.values())

    for cid, spec in BLIND_CELLS.items():
        bmap, _ = generate_paired_cfo_blocks(
            spec.namespace, 0, max_blocks, spec.h, spec.ell, spec.t_p, (-spec.abs_omega, spec.abs_omega), split_prefix=cid
        )
        _add_batches(cid, bmap.values())

    q2_map, _ = generate_paired_cfo_blocks(NS_BLIND_Q2_C4, 0, max_blocks, 2, 2, 4.0, (0.35,), split_prefix="q2_c4")
    _add_batches("Q2_C4_blind", q2_map.values())

    for cname in REPLAY_TRAIN_CONDITIONS:
        _add_batches("s3_train_replay", [generate_mothers("s3_train", cname, max_blocks=max_blocks)])
        _add_batches("s3_dev_replay", [generate_mothers("s3_dev", cname, max_blocks=max_blocks)])

    collisions = 0
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]].intersection(sets[names[j]])
            collisions += len(inter)
            if inter:
                raise AssertionError(f"Orbit collision between {names[i]} and {names[j]}: {len(inter)}")
    return {"counts": {k: len(v) for k, v in sets.items()}, "cross_split_orbit_collisions": collisions}
