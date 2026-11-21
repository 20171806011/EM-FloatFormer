"""Unified evaluation, calibration, and cluster-paired statistical inference module.

Implements strict segmented scoring across sync header, in-packet pilots, withheld holdout
pilots, and payload symbols/bits, plus engineering cost and coverage accounting.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# Gray-mapped QPSK constellation verified empirically against CTTC f2450_snr20_QPSK:
# (b0, b1) -> ((1 - 2*b0) + 1j * (1 - 2*b1)) / sqrt(2)
# Index order: 0:(0,0)->(+1+1j)/s2, 1:(0,1)->(+1-1j)/s2, 2:(1,0)->(-1+1j)/s2, 3:(1,1)->(-1-1j)/s2
INV_SQRT2 = 1.0 / math.sqrt(2.0)
QPSK_CONSTELLATION = np.array(
    [
        (1.0 + 1.0j) * INV_SQRT2,
        (1.0 - 1.0j) * INV_SQRT2,
        (-1.0 + 1.0j) * INV_SQRT2,
        (-1.0 - 1.0j) * INV_SQRT2,
    ],
    dtype=np.complex128,
)
QPSK_BITS_TABLE = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.uint8,
)


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Map unpacked bits (length 2*N, values in {0, 1}) to unit-energy Gray QPSK symbols."""
    b = np.asarray(bits, dtype=np.int64)
    if b.ndim != 1 or (len(b) % 2) != 0:
        raise ValueError(f"Expected 1-D even-length bit array, got shape {b.shape}")
    b0 = b[0::2]
    b1 = b[1::2]
    return ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) * INV_SQRT2


def qpsk_to_bits(syms: np.ndarray) -> np.ndarray:
    """Hard-demap complex QPSK symbols to unpacked Gray bits [b0_0, b1_0, b0_1, b1_1, ...]."""
    s = np.asarray(syms, dtype=np.complex128)
    out = np.empty(2 * len(s), dtype=np.uint8)
    out[0::2] = (s.real < 0.0).astype(np.uint8)
    out[1::2] = (s.imag < 0.0).astype(np.uint8)
    return out


def qpsk_hard_symbols(syms: np.ndarray) -> np.ndarray:
    """Hard-decide complex equalized symbols onto the unit-energy QPSK constellation."""
    s = np.asarray(syms, dtype=np.complex128)
    re = np.where(s.real >= 0.0, 1.0, -1.0)
    im = np.where(s.imag >= 0.0, 1.0, -1.0)
    return (re + 1j * im) * INV_SQRT2


def qpsk_bit_llrs_gaussian(
    y: np.ndarray,
    h: np.ndarray,
    noise_var: np.ndarray | float,
) -> np.ndarray:
    """Compute exact per-bit LLR = log P(b=1|y) / P(b=0|y) under y = h*x + n, n ~ CN(0, noise_var).

    Returns array of shape (2 * len(y),) interleaved as [b0_0, b1_0, b0_1, b1_1, ...].
    """
    y_c = np.asarray(y, dtype=np.complex128)
    h_c = np.asarray(h, dtype=np.complex128)
    nv = np.maximum(np.asarray(noise_var, dtype=np.float64), 1e-18)
    # Log-likelihood of each of the 4 constellation symbols: -|y - h*x|^2 / nv
    # shape: (N, 4)
    diff = y_c[:, None] - h_c[:, None] * QPSK_CONSTELLATION[None, :]
    ll = -(diff.real ** 2 + diff.imag ** 2) / nv[:, None] if nv.ndim == 1 else -(diff.real ** 2 + diff.imag ** 2) / nv
    return _bit_llrs_from_sym_loglik(ll)


def _logsumexp_2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _bit_llrs_from_sym_loglik(sym_ll: np.ndarray) -> np.ndarray:
    """Given per-symbol log-likelihoods shape (N, 4) for indices (00, 01, 10, 11), return bit LLRs (2N,)."""
    # Convention: LLR = log P(b=1 | y) - log P(b=0 | y)
    # b0=0: indices 0, 1 ; b0=1: indices 2, 3
    ll_b0_0 = _logsumexp_2(sym_ll[:, 0], sym_ll[:, 1])
    ll_b0_1 = _logsumexp_2(sym_ll[:, 2], sym_ll[:, 3])
    # b1=0: indices 0, 2 ; b1=1: indices 1, 3
    ll_b1_0 = _logsumexp_2(sym_ll[:, 0], sym_ll[:, 2])
    ll_b1_1 = _logsumexp_2(sym_ll[:, 1], sym_ll[:, 3])
    out = np.empty(2 * sym_ll.shape[0], dtype=np.float64)
    out[0::2] = ll_b0_1 - ll_b0_0
    out[1::2] = ll_b1_1 - ll_b1_0
    return out


def mixture_qpsk_posterior(
    y: np.ndarray,
    h_cands: np.ndarray,
    nv_cands: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact posterior mixture symbol & bit decoding across L channel hypotheses.

    Implements research/posterior_fusion §5.2:
        g(x) = sum_r alpha_r * CN(y | h_r * x, nv_r)
        LLR_b = log sum_{r, x: u_b(x)=1} alpha_r g_r(x) - log sum_{r, x: u_b(x)=0} alpha_r g_r(x)

    Args:
        y: (N,) complex128 received symbols
        h_cands: (L, N) complex128 per-hypothesis channel gains
        nv_cands: (L, N) or (L,) float64 per-hypothesis noise variance
        weights: (L,) float64 nonnegative posterior weights summing to 1

    Returns:
        sym_hard: (N,) complex128 MAP symbol decision from exact mixture
        bit_llrs: (2N,) float64 exact mixture per-bit LLR = log P(b=1)/P(b=0)
        sym_loglik: (N, 4) float64 unnormalized log mixture likelihood per symbol
    """
    y_c = np.asarray(y, dtype=np.complex128)
    h_c = np.asarray(h_cands, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        w = np.full(len(w), 1.0 / max(len(w), 1), dtype=np.float64)
    else:
        w = w / w_sum

    L, N = h_c.shape
    nv = np.asarray(nv_cands, dtype=np.float64)
    if nv.ndim == 1:
        nv = np.broadcast_to(nv[:, None], (L, N))
    nv = np.maximum(nv, 1e-18)

    # log CN(y | h*x, nv) = -log(pi * nv) - |y - h*x|^2 / nv
    # shape (L, N, 4)
    diff = y_c[None, :, None] - h_c[:, :, None] * QPSK_CONSTELLATION[None, None, :]
    log_g = -np.log(math.pi * nv[:, :, None]) - (diff.real ** 2 + diff.imag ** 2) / nv[:, :, None]
    log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), -1e300)
    joint = log_w[:, None, None] + log_g  # (L, N, 4)

    # Stable log-sum-exp over r = 0..L-1
    max_j = np.max(joint, axis=0)  # (N, 4)
    sym_ll = max_j + np.log(np.sum(np.exp(joint - max_j[None, :, :]), axis=0))

    best_idx = np.argmax(sym_ll, axis=1)
    sym_hard = QPSK_CONSTELLATION[best_idx]
    bit_llrs = _bit_llrs_from_sym_loglik(sym_ll)
    return sym_hard, bit_llrs, sym_ll


def joint_mixture_qpsk_posterior(
    y: np.ndarray,
    h_cands: np.ndarray,
    nv_cands: np.ndarray,
    weights: np.ndarray,
    *,
    payload_mask: np.ndarray | None = None,
    block_temp: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact leave-one-out block-joint posterior mixture decoding across L channel hypotheses.

    Implements research/posterior_fusion §5.2 for an uncoded QPSK block sharing a latent branch R:
        log p(y_m | R=r, I) = log(1/4 sum_{x in Q} CN(y_m | h_r(m) x, nv_r(m)))
        log beta_r^{(-n)} = log alpha_r + block_temp * sum_{m in payload, m != n} log p(y_m | R=r, I)
        g(x_n) = sum_r beta_r^{(-n)} * CN(y_n | h_r(n) x_n, nv_r(n))

    Strictly legal (uses only unlabeled target observations y and known QPSK constellation Q)
    and leave-one-out (symbol n never self-reinforces its own branch weight).
    `block_temp in (0, 1]` accounts for finite-sample RRC/ISI correlation and is calibrated on dev.

    Returns:
        sym_hard: (N,) complex128 MAP symbol decision
        bit_llrs: (2N,) float64 exact mixture per-bit LLR = log P(b=1)/P(b=0)
        sym_loglik: (N, 4) float64 unnormalized log mixture likelihood per symbol
        beta_full: (L,) float64 full-block posterior branch probabilities P(R=r | I, y_payload)
    """
    y_c = np.asarray(y, dtype=np.complex128)
    h_c = np.asarray(h_cands, dtype=np.complex128)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    w_sum = float(np.sum(w))
    w = (w / w_sum) if w_sum > 0.0 else np.full(len(w), 1.0 / max(len(w), 1), dtype=np.float64)

    L, N = h_c.shape
    nv = np.asarray(nv_cands, dtype=np.float64)
    if nv.ndim == 1:
        nv = np.broadcast_to(nv[:, None], (L, N))
    nv = np.maximum(nv, 1e-18)

    # Per-branch per-symbol constellation log-likelihood: (L, N, 4)
    diff = y_c[None, :, None] - h_c[:, :, None] * QPSK_CONSTELLATION[None, None, :]
    log_g = -np.log(math.pi * nv[:, :, None]) - (diff.real ** 2 + diff.imag ** 2) / nv[:, :, None]

    # Exact marginal unlabeled symbol log-likelihood log p(y_n | r) = logsumexp_x(log_g) - log(4)
    max_x = np.max(log_g, axis=2)  # (L, N)
    log_py_sym = max_x + np.log(np.sum(np.exp(log_g - max_x[:, :, None]), axis=2)) - math.log(4.0)

    if payload_mask is None:
        mask_f = np.ones(N, dtype=np.float64)
    else:
        mask_f = np.asarray(payload_mask, dtype=np.float64)

    # Sum over payload symbols only (leaves holdout pilots completely untouched!)
    total_pay_ll = np.sum(log_py_sym * mask_f[None, :], axis=1)  # (L,)
    log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), -1e300)

    # Full-block posterior beta_full (for channel readout / diagnostics)
    log_beta_full = log_w + block_temp * total_pay_ll
    log_beta_full -= float(np.max(log_beta_full))
    beta_full = np.exp(np.clip(log_beta_full, -500.0, 0.0))
    beta_full /= float(np.sum(beta_full))

    # Leave-one-out log-weights per symbol n: (L, N)
    loo_ll = total_pay_ll[:, None] - log_py_sym * mask_f[None, :]
    log_beta_loo = log_w[:, None] + block_temp * loo_ll
    log_beta_loo -= np.max(log_beta_loo, axis=0, keepdims=True)
    log_beta_loo -= np.log(np.sum(np.exp(np.clip(log_beta_loo, -500.0, 0.0)), axis=0, keepdims=True))

    joint = log_beta_loo[:, :, None] + log_g  # (L, N, 4)
    max_j = np.max(joint, axis=0)  # (N, 4)
    sym_ll = max_j + np.log(np.sum(np.exp(joint - max_j[None, :, :]), axis=0))

    best_idx = np.argmax(sym_ll, axis=1)
    sym_hard = QPSK_CONSTELLATION[best_idx]
    bit_llrs = _bit_llrs_from_sym_loglik(sym_ll)
    return sym_hard, bit_llrs, sym_ll, beta_full


def compute_segment_metrics(
    x_eq: np.ndarray,
    x_true: np.ndarray,
    bit_llrs: np.ndarray | None = None,
    hard_syms_override: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute SER, BER, raw symbol MSE, raw EVM, unbiased EVM, and bit calibration metrics."""
    x_e = np.asarray(x_eq, dtype=np.complex128)
    x_t = np.asarray(x_true, dtype=np.complex128)
    n = len(x_t)
    if n == 0:
        return {}

    true_bits = qpsk_to_bits(x_t)
    if hard_syms_override is not None:
        sym_dec = np.asarray(hard_syms_override, dtype=np.complex128)
        dec_bits = qpsk_to_bits(sym_dec)
    elif bit_llrs is not None:
        llrs = np.asarray(bit_llrs, dtype=np.float64)
        dec_bits = (llrs > 0.0).astype(np.uint8)
        sym_dec = bits_to_qpsk(dec_bits)
    else:
        sym_dec = qpsk_hard_symbols(x_e)
        dec_bits = qpsk_to_bits(sym_dec)

    ser = float(np.mean((dec_bits[0::2] != true_bits[0::2]) | (dec_bits[1::2] != true_bits[1::2])))
    ber = float(np.mean(dec_bits != true_bits))
    sym_err_count = int(np.sum((dec_bits[0::2] != true_bits[0::2]) | (dec_bits[1::2] != true_bits[1::2])))
    bit_err_count = int(np.sum(dec_bits != true_bits))

    # Raw symbol reconstruction MSE
    raw_mse = float(np.mean(np.abs(x_e - x_t) ** 2))
    ref_pow = float(np.mean(np.abs(x_t) ** 2))
    raw_evm_ratio = math.sqrt(max(raw_mse, 0.0) / max(ref_pow, 1e-18))
    raw_evm_db = 10.0 * math.log10(max(raw_mse / max(ref_pow, 1e-18), 1e-18))

    # Unbiased EVM (removes scalar shrinkage/gain bias alpha = <x_e, x_t> / ||x_t||^2)
    alpha = np.vdot(x_t, x_e) / max(float(np.vdot(x_t, x_t).real), 1e-18)
    if abs(alpha) > 1e-12:
        x_unbiased = x_e / alpha
        unb_mse = float(np.mean(np.abs(x_unbiased - x_t) ** 2))
    else:
        unb_mse = raw_mse
    unb_evm_ratio = math.sqrt(max(unb_mse, 0.0) / max(ref_pow, 1e-18))
    unb_evm_db = 10.0 * math.log10(max(unb_mse / max(ref_pow, 1e-18), 1e-18))

    out: dict[str, float] = {
        "n_syms": float(n),
        "n_bits": float(2 * n),
        "sym_errors": float(sym_err_count),
        "bit_errors": float(bit_err_count),
        "ser": ser,
        "ber": ber,
        "sym_mse": raw_mse,
        "evm_raw_pct": 100.0 * raw_evm_ratio,
        "evm_raw_db": raw_evm_db,
        "evm_unbiased_pct": 100.0 * unb_evm_ratio,
        "evm_unbiased_db": unb_evm_db,
    }

    if bit_llrs is not None:
        llrs = np.clip(np.asarray(bit_llrs, dtype=np.float64), -35.0, 35.0)
        # Binary cross-entropy (nats/bit) for P(b=1) = sigmoid(llr)
        # -log P(b=true) = log(1 + exp(llr)) - true_bits * llr
        softplus = np.logaddexp(0.0, llrs)
        nll = float(np.mean(softplus - true_bits.astype(np.float64) * llrs))
        probs = 1.0 / (1.0 + np.exp(-llrs))
        brier = float(np.mean((probs - true_bits.astype(np.float64)) ** 2))
        out["bit_nll_nats"] = nll
        out["bit_brier"] = brier
    return out


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) on predicted bit probabilities vs binary labels."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        return float("nan")
    # Confidence = max(p, 1-p), correctness = ((p >= 0.5) == y)
    conf = np.maximum(p, 1.0 - p)
    corr = ((p >= 0.5).astype(np.float64) == y).astype(np.float64)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(p))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(corr[mask])) - float(np.mean(conf[mask])))
    return float(ece)


def compute_holdout_channel_metrics(
    y_hold: np.ndarray,
    x_hold: np.ndarray,
    h_pred: np.ndarray,
    pred_var: np.ndarray | float,
    h_offline_ref: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate withheld holdout-pilot channel prediction accuracy & Gaussian predictive NLL.

    Because real OTA hardware has no noiseless ground-truth channel h_true, we report:
      1. holdout_obs_nmse: sum|y_hold - h_pred*x_hold|^2 / sum|y_hold|^2 (strictly observable)
      2. holdout_ch_nll_nats: mean -log CN(y_hold | h_pred*x_hold, pred_var)
      3. holdout_var_ratio: mean(|y_hold - h_pred*x_hold|^2 / pred_var) (1.0 = calibrated variance)
      4. holdout_ref_nmse (optional): NMSE against offline full-packet smoothed reference channel
         (used strictly offline in evaluator; clearly marked as reference-channel NMSE)
    """
    y_h = np.asarray(y_hold, dtype=np.complex128)
    x_h = np.asarray(x_hold, dtype=np.complex128)
    h_p = np.asarray(h_pred, dtype=np.complex128)
    v_p = np.maximum(np.asarray(pred_var, dtype=np.float64), 1e-18)
    if v_p.ndim == 0:
        v_p = np.full(len(y_h), float(v_p), dtype=np.float64)

    y_hat = h_p * x_h
    resid2 = np.abs(y_h - y_hat) ** 2
    obs_pow = float(np.sum(np.abs(y_h) ** 2))
    obs_nmse = float(np.sum(resid2) / max(obs_pow, 1e-30))
    nll = float(np.mean(np.log(math.pi * v_p) + resid2 / v_p))
    var_ratio = float(np.mean(resid2 / v_p))

    out = {
        "holdout_obs_nmse": obs_nmse,
        "holdout_ch_nll_nats": nll,
        "holdout_var_ratio": var_ratio,
        "holdout_resid_mse": float(np.mean(resid2)),
    }
    if h_offline_ref is not None:
        h_r = np.asarray(h_offline_ref, dtype=np.complex128)
        ref_pow = float(np.sum(np.abs(h_r) ** 2))
        out["holdout_ref_nmse"] = float(np.sum(np.abs(h_p - h_r) ** 2) / max(ref_pow, 1e-30))
        out["holdout_ref_mse"] = float(np.mean(np.abs(h_p - h_r) ** 2))
    return out


@dataclass
class PacketEvalRecord:
    """Standardized per-packet evaluation record for one method on one fixed packet."""
    unit_id: str
    capture_id: str
    freq_mhz: int
    nom_snr_db: int
    cluster_id: str
    split: str
    method: str
    success: bool
    error_msg: str = ""
    elapsed_ms: float = 0.0
    # Payload metrics
    payload_ser: float = float("nan")
    payload_ber: float = float("nan")
    payload_per: float = float("nan")  # 1.0 if any payload symbol/bit error, else 0.0
    payload_sym_mse: float = float("nan")
    payload_evm_raw_db: float = float("nan")
    payload_evm_unbiased_db: float = float("nan")
    payload_bit_nll: float = float("nan")
    payload_bit_brier: float = float("nan")
    payload_n_syms: int = 0
    payload_sym_errors: int = 0
    payload_bit_errors: int = 0
    # Sync & pilot metrics
    sync_ser: float = float("nan")
    sync_ber: float = float("nan")
    sync_evm_unbiased_db: float = float("nan")
    pilot_ser: float = float("nan")
    pilot_ber: float = float("nan")
    pilot_evm_unbiased_db: float = float("nan")
    # Holdout pilot channel & calibration metrics
    holdout_obs_nmse: float = float("nan")
    holdout_ref_nmse: float = float("nan")
    holdout_ch_nll: float = float("nan")
    holdout_var_ratio: float = float("nan")
    # Optional extra diagnostics
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        d = {
            "unit_id": self.unit_id,
            "capture_id": self.capture_id,
            "freq_mhz": self.freq_mhz,
            "nom_snr_db": self.nom_snr_db,
            "cluster_id": self.cluster_id,
            "split": self.split,
            "method": self.method,
            "success": self.success,
            "error_msg": self.error_msg,
            "elapsed_ms": self.elapsed_ms,
            "payload_ser": self.payload_ser,
            "payload_ber": self.payload_ber,
            "payload_per": self.payload_per,
            "payload_sym_mse": self.payload_sym_mse,
            "payload_evm_raw_db": self.payload_evm_raw_db,
            "payload_evm_unbiased_db": self.payload_evm_unbiased_db,
            "payload_bit_nll": self.payload_bit_nll,
            "payload_bit_brier": self.payload_bit_brier,
            "payload_n_syms": self.payload_n_syms,
            "payload_sym_errors": self.payload_sym_errors,
            "payload_bit_errors": self.payload_bit_errors,
            "sync_ser": self.sync_ser,
            "sync_ber": self.sync_ber,
            "sync_evm_unbiased_db": self.sync_evm_unbiased_db,
            "pilot_ser": self.pilot_ser,
            "pilot_ber": self.pilot_ber,
            "pilot_evm_unbiased_db": self.pilot_evm_unbiased_db,
            "holdout_obs_nmse": self.holdout_obs_nmse,
            "holdout_ref_nmse": self.holdout_ref_nmse,
            "holdout_ch_nll": self.holdout_ch_nll,
            "holdout_var_ratio": self.holdout_var_ratio,
        }
        for k, v in self.extra.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                d[f"extra_{k}"] = v
        return d


def aggregate_records(
    records: Sequence[PacketEvalRecord],
    bit_probs_all: np.ndarray | None = None,
    bit_labels_all: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate per-packet records into a comprehensive summary dictionary."""
    n_tot = len(records)
    ok = [r for r in records if r.success]
    n_ok = len(ok)
    n_fail = n_tot - n_ok
    if n_ok == 0:
        return {
            "n_packets_total": n_tot,
            "n_packets_ok": 0,
            "n_packets_fail": n_fail,
            "coverage": 0.0,
        }

    tot_pay_syms = sum(r.payload_n_syms for r in ok)
    tot_pay_bits = 2 * tot_pay_syms
    tot_sym_err = sum(r.payload_sym_errors for r in ok)
    tot_bit_err = sum(r.payload_bit_errors for r in ok)
    tot_pkt_err = sum(1 for r in ok if r.payload_per >= 0.5)

    def _mean_finite(vals: list[float]) -> float:
        arr = np.array([v for v in vals if math.isfinite(v)], dtype=np.float64)
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    def _pct_finite(vals: list[float], q: float) -> float:
        arr = np.array([v for v in vals if math.isfinite(v)], dtype=np.float64)
        return float(np.percentile(arr, q)) if len(arr) > 0 else float("nan")

    # Aggregate raw/unbiased EVM via linear MSE average converted to dB
    raw_mse_mean = _mean_finite([r.payload_sym_mse for r in ok])
    unb_mse_vals = [10.0 ** (r.payload_evm_unbiased_db / 10.0) for r in ok if math.isfinite(r.payload_evm_unbiased_db)]
    unb_mse_mean = float(np.mean(unb_mse_vals)) if unb_mse_vals else float("nan")

    times = [r.elapsed_ms for r in ok]
    summary: dict[str, Any] = {
        "n_packets_total": n_tot,
        "n_packets_ok": n_ok,
        "n_packets_fail": n_fail,
        "coverage": n_ok / float(n_tot),
        "payload_n_syms_total": tot_pay_syms,
        "payload_n_bits_total": tot_pay_bits,
        "payload_sym_errors_total": tot_sym_err,
        "payload_bit_errors_total": tot_bit_err,
        "payload_pkt_errors_total": tot_pkt_err,
        "payload_ser": tot_sym_err / float(max(tot_pay_syms, 1)),
        "payload_ber": tot_bit_err / float(max(tot_pay_bits, 1)),
        "payload_per": tot_pkt_err / float(n_ok),
        "payload_sym_mse": raw_mse_mean,
        "payload_evm_raw_db": 10.0 * math.log10(max(raw_mse_mean, 1e-18)) if math.isfinite(raw_mse_mean) else float("nan"),
        "payload_evm_unbiased_db": 10.0 * math.log10(max(unb_mse_mean, 1e-18)) if math.isfinite(unb_mse_mean) else float("nan"),
        "payload_bit_nll_nats": _mean_finite([r.payload_bit_nll for r in ok]),
        "payload_bit_brier": _mean_finite([r.payload_bit_brier for r in ok]),
        "sync_ser": _mean_finite([r.sync_ser for r in ok]),
        "sync_ber": _mean_finite([r.sync_ber for r in ok]),
        "sync_evm_unbiased_db": _mean_finite([r.sync_evm_unbiased_db for r in ok]),
        "pilot_ser": _mean_finite([r.pilot_ser for r in ok]),
        "pilot_ber": _mean_finite([r.pilot_ber for r in ok]),
        "pilot_evm_unbiased_db": _mean_finite([r.pilot_evm_unbiased_db for r in ok]),
        "holdout_obs_nmse": _mean_finite([r.holdout_obs_nmse for r in ok]),
        "holdout_ref_nmse": _mean_finite([r.holdout_ref_nmse for r in ok]),
        "holdout_ch_nll_nats": _mean_finite([r.holdout_ch_nll for r in ok]),
        "holdout_var_ratio": _mean_finite([r.holdout_var_ratio for r in ok]),
        "elapsed_ms_mean": _mean_finite(times),
        "elapsed_ms_median": _pct_finite(times, 50.0),
        "elapsed_ms_p95": _pct_finite(times, 95.0),
    }
    if bit_probs_all is not None and bit_labels_all is not None and len(bit_probs_all) > 0:
        summary["payload_bit_ece"] = compute_ece(bit_probs_all, bit_labels_all, n_bins=15)
    return summary


def paired_cluster_inference(
    recs_a: Sequence[PacketEvalRecord],
    recs_b: Sequence[PacketEvalRecord],
    metric_attr: str,
    *,
    group_by: str = "cluster_id",
    lower_is_better: bool = True,
    n_bootstrap: int = 10000,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Paired cluster/session-level comparison of method B vs baseline A.

    Computes diff_g = (A_g - B_g) if lower_is_better else (B_g - A_g), so positive diff
    ALWAYS means B improves over baseline A!
    Uses exact sign-flip test when G <= 16 (2^G states, matching P12 methodology) and
    Monte-Carlo sign-flip (50,000 draws) when G > 16, plus cluster bootstrap 95% CI.
    """
    map_a = {r.unit_id: r for r in recs_a if r.success}
    map_b = {r.unit_id: r for r in recs_b if r.success}
    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_ids:
        return {"n_common_units": 0, "n_clusters": 0}

    groups: dict[str, list[tuple[float, float]]] = {}
    for uid in common_ids:
        ra, rb = map_a[uid], map_b[uid]
        va = float(getattr(ra, metric_attr))
        vb = float(getattr(rb, metric_attr))
        if not (math.isfinite(va) and math.isfinite(vb)):
            continue
        gid = str(getattr(ra, group_by))
        groups.setdefault(gid, []).append((va, vb))

    gids = sorted(groups.keys())
    G = len(gids)
    if G == 0:
        return {"n_common_units": len(common_ids), "n_clusters": 0}

    diffs = np.empty(G, dtype=np.float64)
    vals_a = np.empty(G, dtype=np.float64)
    vals_b = np.empty(G, dtype=np.float64)
    for i, gid in enumerate(gids):
        pairs = groups[gid]
        ma = float(np.mean([p[0] for p in pairs]))
        mb = float(np.mean([p[1] for p in pairs]))
        vals_a[i] = ma
        vals_b[i] = mb
        diffs[i] = (ma - mb) if lower_is_better else (mb - ma)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if G > 1 else 0.0
    se_diff = std_diff / math.sqrt(G) if G > 1 else 0.0
    wins = int(np.sum(diffs > 1e-15))
    ties = int(np.sum(np.abs(diffs) <= 1e-15))
    losses = int(np.sum(diffs < -1e-15))

    # Bootstrap 95% CI over clusters
    rng = np.random.default_rng(seed)
    if G > 1:
        idx = rng.integers(0, G, size=(n_bootstrap, G))
        boot_means = np.mean(diffs[idx], axis=1)
        ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    else:
        ci_lo, ci_hi = mean_diff, mean_diff

    # Paired sign-flip two-sided p-value
    abs_obs = abs(mean_diff)
    if G <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=G)), dtype=np.float64)
        sf_means = np.abs(signs @ diffs) / float(G)
        p_two_sided = float(np.mean(sf_means >= abs_obs - 1e-15))
        p_one_sided = float(np.mean((signs @ diffs) / float(G) >= mean_diff - 1e-15))
        sf_mode = f"exact_{1 << G}"
    else:
        n_sf = 50000
        signs = rng.choice([-1.0, 1.0], size=(n_sf, G))
        sf_vals = (signs @ diffs) / float(G)
        p_two_sided = float((1 + np.sum(np.abs(sf_vals) >= abs_obs - 1e-15)) / (n_sf + 1))
        p_one_sided = float((1 + np.sum(sf_vals >= mean_diff - 1e-15)) / (n_sf + 1))
        sf_mode = f"mc_{n_sf}"

    return {
        "metric": metric_attr,
        "group_by": group_by,
        "lower_is_better": lower_is_better,
        "n_common_units": len(common_ids),
        "n_clusters": G,
        "mean_baseline_a": float(np.mean(vals_a)),
        "mean_candidate_b": float(np.mean(vals_b)),
        "mean_improvement": mean_diff,
        "se_improvement": se_diff,
        "boot_ci95_lo": ci_lo,
        "boot_ci95_hi": ci_hi,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "signflip_p_two_sided": p_two_sided,
        "signflip_p_one_sided": p_one_sided,
        "signflip_mode": sf_mode,
    }
