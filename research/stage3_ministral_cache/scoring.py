"""Shared scoring, symmetry/duplicate invariance checks, calibration, bootstrap CIs, and failure extraction."""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


ECE_BINS = 10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 17429


def compute_classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Acc, NLL, Brier score, and 10-bin ECE on (N, 4) probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if p.shape != (n, 4):
        raise ValueError(f"Expected probs shape {(n, 4)}, got {p.shape}")
    # Lowest-index tie break matches np.argmax and parent evaluate.py
    preds = np.argmax(p, axis=1)
    hits = (preds == y).astype(np.float64)
    p_clip = np.clip(p, 1e-12, 1.0)
    p_true = p_clip[np.arange(n), y]
    nll = -np.log(p_true)

    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    brier = np.sum((p - one_hot) ** 2, axis=1)

    confs = np.max(p, axis=1)
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    ece = 0.0
    for i in range(ECE_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs > lo) & (confs <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / float(n)) * abs(float(hits[mask].mean()) - float(confs[mask].mean()))

    return {
        "n": int(n),
        "acc": float(hits.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": float(ece),
        "per_item_hit": hits,
        "per_item_nll": nll,
    }


def compute_complex_nmse(mu_hat: np.ndarray, target_t3: np.ndarray) -> float:
    num = float(np.mean(np.abs(np.asarray(target_t3, dtype=np.complex128) - np.asarray(mu_hat, dtype=np.complex128)) ** 2))
    den = float(np.mean(np.abs(np.asarray(target_t3, dtype=np.complex128)) ** 2))
    return num / max(den, 1e-30)


def check_vector_relation(
    actual: np.ndarray,
    expected: np.ndarray,
    tol: float = 3e-13,
) -> dict:
    err = np.abs(actual - expected)
    residual = err.max(axis=1)
    failed = residual > tol
    tv = 0.5 * err.sum(axis=1)
    return {
        "rows": int(len(actual)),
        "failed_rows": int(failed.sum()),
        "pass": not bool(failed.any()),
        "max_abs": float(residual.max()),
        "mean_abs": float(residual.mean()),
        "p95_abs": float(np.quantile(residual, 0.95)),
        "mean_tv": float(tv.mean()),
        "max_tv": float(tv.max()),
        "failed_mask": failed,
    }


def paired_bootstrap_ci(
    diff_per_mother: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Clustered paired bootstrap over mothers (2,000 resamples, seed 17429)."""
    arr = np.asarray(diff_per_mother, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    boot = []
    for chunk in np.array_split(draws, 20):
        boot.extend(arr[chunk].mean(axis=1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_mothers": int(n),
        "mean": float(arr.mean()),
        "lo95": float(lo),
        "hi95": float(hi),
    }


def extract_worst_examples(
    mother_ids: List[str],
    probs: np.ndarray,
    labels: np.ndarray,
    extra_meta: Optional[List[dict]] = None,
    top_k: int = 5,
) -> List[dict]:
    """Return top-K highest-NLL replayable failure examples."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    order = np.argsort(-nll)
    out = []
    for idx in order[:top_k]:
        i = int(idx)
        entry = {
            "idx": i,
            "mother_id": mother_ids[i],
            "true_label": int(y[i]),
            "pred_label": int(np.argmax(p[i])),
            "probs": [float(v) for v in p[i]],
            "nll": float(nll[i]),
        }
        if extra_meta is not None:
            entry["meta"] = extra_meta[i]
        out.append(entry)
    return out
