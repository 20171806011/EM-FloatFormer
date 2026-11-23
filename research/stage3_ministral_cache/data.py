"""Deterministic mother-waveform dataset generator and cross-split orbit collision audit."""
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ConditionSpec,
    N_SAMPLES,
    ROOTS,
    Reference,
    matched_sum,
    sector_of_complex,
)


SPLIT_SEEDS: Dict[str, Tuple[int, int]] = {
    "s3_train": (840101, 16),  # 1,024 mothers
    "s3_dev": (840202, 4),     # 256 mothers
    "s3_test": (840303, 16),   # 1,024 mothers
}

PARENT_PARTITIONS: Dict[str, Tuple[int, int]] = {
    "train": (730241, 64),
    "dev": (410061, 4),
    "test": (518339, 16),
}


@dataclass
class MotherBatch:
    split: str
    condition: ConditionSpec
    mother_ids: List[str]
    x1: np.ndarray            # (M,) int64 in 0..3
    x2: np.ndarray            # (M,) int64 in 0..3
    g: np.ndarray             # (M,) int64 in 0..3
    x3_indep: np.ndarray      # (M,) int64 in 0..3
    y1: np.ndarray            # (M, 3) complex128
    y2: np.ndarray            # (M, 3) complex128
    y3_repeat: np.ndarray     # (M, 3) complex128
    y3_indep: np.ndarray      # (M, 3) complex128
    p_valid1: np.ndarray      # (M, 3) complex128
    p_valid2: np.ndarray      # (M, 3) complex128 (fresh resample of valid pilot)
    p_indep: np.ndarray       # (M, 3) complex128 (unrelated channel pilot)
    phi_2: np.ndarray         # (M,) float64 continuous shift or discrete*pi/2
    phi_3: np.ndarray         # (M,) float64
    phi_p: np.ndarray         # (M,) float64

    @property
    def k3_repeat(self) -> np.ndarray:
        return sector_of_complex(matched_sum(self.y3_repeat))

    @property
    def k3_indep(self) -> np.ndarray:
        return sector_of_complex(matched_sum(self.y3_indep))

    @property
    def t3_repeat(self) -> np.ndarray:
        return matched_sum(self.y3_repeat)

    @property
    def t3_indep(self) -> np.ndarray:
        return matched_sum(self.y3_indep)

    def rotated_view(self, s1: int, s2: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.y1 * ROOTS[s1 % 4], self.y2 * ROOTS[s2 % 4]

    def references_for_arm(self, arm: str, coord_rot: int = 0) -> Tuple[Reference, ...]:
        r = ROOTS[coord_rot % 4]
        tp = self.condition.t_p
        snr_db = self.condition.snr_db
        if arm == "none":
            return ()
        if arm == "valid":
            return (Reference(0, True, self.p_valid1 * r, tp, snr_db),)
        if arm == "independent":
            return (Reference(0, False, self.p_indep * r, tp, snr_db),)
        if arm == "duplicate_copy":
            ref0 = Reference(0, True, self.p_valid1 * r, tp, snr_db)
            return (ref0, ref0)
        if arm == "fresh_resample":
            return (
                Reference(0, True, self.p_valid1 * r, tp, snr_db),
                Reference(1, True, self.p_valid2 * r, tp, snr_db),
            )
        raise ValueError(f"Unknown arm={arm}")


def _sample_discrete_jumps(rng: np.random.Generator, d: float, steps: int, size: int) -> np.ndarray:
    if steps <= 0 or d == 0.0:
        return np.zeros(size, dtype=np.int64)
    # Use cumulative mod-4 Markov chain step-by-step
    acc = np.zeros(size, dtype=np.int64)
    probs = [1.0 - d, d / 3.0, d / 3.0, d / 3.0]
    for _ in range(steps):
        acc = (acc + rng.choice(4, size=size, p=probs)) % 4
    return acc


def generate_mothers(split: str, cond_name: str, max_blocks: Optional[int] = None) -> MotherBatch:
    """Generate deterministic balanced 64-mother blocks for `(split, condition)`."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split={split}")
    if cond_name not in CONDITION_SPECS:
        raise ValueError(f"Unknown condition={cond_name}")
    cond = CONDITION_SPECS[cond_name]
    base_seed, total_blocks = SPLIT_SEEDS[split]
    blocks = total_blocks if max_blocks is None else min(total_blocks, int(max_blocks))
    cond_idx = sorted(CONDITION_SPECS.keys()).index(cond_name)
    rng = np.random.default_rng(base_seed * 100 + cond_idx)

    triples_base = list(itertools.product(range(4), repeat=3))
    x1_list, x2_list, g_list = [], [], []
    for _ in range(blocks):
        perm = rng.permutation(64)
        for p_idx in perm:
            a, b, c = triples_base[int(p_idx)]
            x1_list.append(a)
            x2_list.append(b)
            g_list.append(c)

    m = blocks * 64
    x1 = np.asarray(x1_list, dtype=np.int64)
    x2 = np.asarray(x2_list, dtype=np.int64)
    g = np.asarray(g_list, dtype=np.int64)
    x3_indep = rng.integers(4, size=m, dtype=np.int64)
    g_other = rng.integers(4, size=m, dtype=np.int64)
    mother_ids = [f"{split}:{cond_name}:{i}" for i in range(m)]

    # Channel phase trajectory over sorted unique timestamps {0.0, h, h+ell, t_p}
    times = sorted({0.0, float(cond.h), float(cond.h + cond.ell), float(cond.t_p)})
    if cond.drift_type == "coherent":
        phi_map = {t: np.zeros(m, dtype=np.float64) for t in times}
    elif cond.drift_type == "discrete":
        # Check integer steps
        disc_times = sorted({0, int(cond.h), int(cond.h + cond.ell), int(round(cond.t_p))})
        disc_map = {disc_times[0]: np.zeros(m, dtype=np.int64)}
        for t_prev, t_cur in zip(disc_times[:-1], disc_times[1:]):
            jump = _sample_discrete_jumps(rng, cond.d, t_cur - t_prev, m)
            disc_map[t_cur] = (disc_map[t_prev] + jump) % 4
        # Re-anchor so t=0 is phase 0 (discrete Markov is reversible & stationary-symmetric)
        phi_map = {float(t): (disc_map[t] - disc_map[0]) % 4 * (0.5 * math.pi) for t in disc_times}
    else:
        # continuous Brownian (+ linear CFO omega)
        cont_map = {times[0]: np.zeros(m, dtype=np.float64)}
        for t_prev, t_cur in zip(times[:-1], times[1:]):
            dt = float(t_cur - t_prev)
            mean = cond.omega * dt
            sd_p = cond.sigma_phi * math.sqrt(max(0.0, dt))
            inc = mean + (rng.normal(scale=sd_p, size=m) if sd_p > 0.0 else 0.0)
            cont_map[t_cur] = cont_map[t_prev] + inc
        phi_map = {t: cont_map[t] - cont_map[0.0] for t in times}

    phi_2 = phi_map[float(cond.h)]
    phi_3 = phi_map[float(cond.h + cond.ell)]
    phi_p = phi_map[float(cond.t_p)]

    noise_sd = math.sqrt(1.0 / (2.0 * cond.snr))
    norm = math.sqrt(float(N_SAMPLES))

    def observe(u_labels: np.ndarray, phi: np.ndarray) -> np.ndarray:
        sig = (ROOTS[u_labels % 4] * np.exp(1j * phi)) / norm
        n = rng.normal(scale=noise_sd, size=(m, N_SAMPLES)) + 1j * rng.normal(scale=noise_sd, size=(m, N_SAMPLES))
        return sig[:, None] + n

    y1 = observe(x1 + g, phi_map[0.0])
    y2 = observe(x2 + g, phi_2)
    y3_repeat = observe(x1 + g, phi_3)
    y3_indep = observe(x3_indep + g, phi_3)
    p_valid1 = observe(g, phi_p)
    p_valid2 = observe(g, phi_p)  # Fresh independent noise of SAME valid pilot!
    phi_indep = rng.uniform(-math.pi, math.pi, size=m) if cond.drift_type != "coherent" else np.zeros(m)
    p_indep = observe(g_other, phi_indep)

    return MotherBatch(
        split=split,
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


def _window_bytes(window: np.ndarray) -> bytes:
    return b"".join(struct.pack("!dd", float(complex(z).real + 0.0), float(complex(z).imag + 0.0)) for z in window)


def orbit_key(window: np.ndarray) -> bytes:
    """Canonical C_4 quarter-turn equivalence key."""
    return min(_window_bytes(np.asarray(window) * r) for r in ROOTS)


def verify_cross_split_isolation(max_blocks_per_cond: Optional[int] = None) -> Dict[str, object]:
    """Assert zero C_4-orbit collision across s3 splits AND 504f3a44 train/dev/test splits."""
    parent_dir = Path(__file__).resolve().parents[1] / "fourphase_paired_questions"
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import paired

    all_orbits_by_split: Dict[str, set] = {}
    old_splits = dict(paired.SPLITS)
    try:
        paired.SPLITS.update(PARENT_PARTITIONS)
        for psplit in ("train", "dev", "test"):
            s = set()
            for _, _, (w1, w2), pilot, _ in paired.parents(psplit):
                s.add(orbit_key(w1))
                s.add(orbit_key(w2))
                s.add(orbit_key(pilot))
            all_orbits_by_split[f"parent_{psplit}"] = s
    finally:
        paired.SPLITS.clear()
        paired.SPLITS.update(old_splits)

    for s3split in ("s3_train", "s3_dev", "s3_test"):
        s = set()
        for cname in CONDITION_SPECS:
            batch = generate_mothers(s3split, cname, max_blocks=max_blocks_per_cond)
            for i in range(len(batch.x1)):
                s.add(orbit_key(batch.y1[i]))
                s.add(orbit_key(batch.y2[i]))
                s.add(orbit_key(batch.y3_repeat[i]))
                s.add(orbit_key(batch.p_valid1[i]))
        all_orbits_by_split[s3split] = s

    collisions = 0
    names = list(all_orbits_by_split.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Only check across distinct stages/splits (allow parent_* self? checking all pairs)
            inter = all_orbits_by_split[names[i]].intersection(all_orbits_by_split[names[j]])
            collisions += len(inter)
            if inter:
                raise AssertionError(f"Orbit collision between {names[i]} and {names[j]}: {len(inter)}")
    return {
        "counts": {k: len(v) for k, v in all_orbits_by_split.items()},
        "cross_split_orbit_collisions": collisions,
    }
