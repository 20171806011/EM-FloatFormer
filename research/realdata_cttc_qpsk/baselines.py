"""Mature classical QPSK receiver baselines for CTTC real-data measurements.

Implements strictly causal, pilot-only packet receivers with zero ground-truth leakage:
  1. Fractional timing synchronization (cubic interpolation over sub-sample offsets)
  2. Joint pilot-aided CFO & phase acquisition (coarse grid + fine parabolic/Brent refinement)
  3. LS scalar channel estimation, regularized MMSE channel estimation (with optional linear
     intra-packet drift / slope tracking), and regularized 3-tap symbol-spaced MMSE equalizer
  4. Hard QPSK demodulation & exact calibrated soft-decision bit LLRs
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .adapter import (
    FS_SYM,
    HOLDOUT_IDX,
    KNOWN_IDX,
    PAD_SYMS,
    PAYLOAD_IDX,
    PILOT_IDX,
    PKT_LEN,
    SPS_NOM,
    SYNC_IDX,
    PacketSample,
)
from .evaluator import (
    PacketEvalRecord,
    compute_holdout_channel_metrics,
    compute_segment_metrics,
    qpsk_bit_llrs_gaussian,
    qpsk_hard_symbols,
    qpsk_to_bits,
)


@dataclass
class ClassicalConfig:
    """Hyperparameters for classical receiver baseline (tuned strictly on dev split)."""
    cfo_search_min_hz: float = -2000.0
    cfo_search_max_hz: float = 2000.0
    cfo_coarse_steps: int = 81  # 50 Hz spacing over [-2000, +2000] Hz
    timing_grid_steps: int = 9  # offsets in [-1.0, +1.0] RX samples (0.25 sample step)
    # Noise blending: nv_est = noise_blend * nv_pilot_resid + (1 - noise_blend) * bg_noise_var
    noise_blend: float = 0.35
    noise_floor_mult: float = 0.85
    # Regularized MMSE shrinkage & intra-packet linear drift regularization
    mmse_prior_scale: float = 1.0
    slope_reg: float = 2.0  # ridge penalty on linear envelope slope (0=disabled when use_slope=False)
    use_slope: bool = False
    use_3tap_mmse: bool = False
    tap3_reg: float = 1.2
    # Variance inflation factor for calibrated predictive NLL / bit LLRs (tuned on dev)
    var_calib_scale: float = 1.0


@dataclass
class ReceiverOutput:
    """Standardized receiver output for one packet across all baselines and experiments."""
    y_sym: np.ndarray  # (256,) complex128 timing-recovered symbol-rate observations
    h_hat: np.ndarray  # (256,) complex128 predicted complex channel gain at each symbol
    nv_hat: np.ndarray  # (256,) float64 predicted effective noise+estimation variance per symbol
    x_eq: np.ndarray  # (256,) complex128 equalized soft symbols
    bit_llrs: np.ndarray  # (512,) float64 per-bit LLR = log P(b=1)/P(b=0)
    hard_syms: np.ndarray  # (256,) complex128 hard QPSK decisions
    est_timing_rx: float
    est_cfo_hz: float
    est_noise_var: float
    elapsed_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


def extract_symbol_stream(y_iq_4x: np.ndarray, timing_rx: float) -> np.ndarray:
    """Extract PKT_LEN symbol-rate complex samples using cubic spline interpolation at `timing_rx`."""
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    t_rx = np.arange(len(y_c), dtype=np.float64)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)
    idx = (PAD_SYMS * SPS_NOM) + timing_rx + SPS_NOM * np.arange(PKT_LEN, dtype=np.float64)
    return cs_re(idx) + 1j * cs_im(idx)


def extract_symbol_stream_padded(y_iq_4x: np.ndarray, timing_rx: float, pad: int = 1) -> np.ndarray:
    """Extract (PKT_LEN + 2*pad) symbol-rate samples for multi-tap equalization."""
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    t_rx = np.arange(len(y_c), dtype=np.float64)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)
    idx = (PAD_SYMS * SPS_NOM) + timing_rx + SPS_NOM * np.arange(-pad, PKT_LEN + pad, dtype=np.float64)
    return cs_re(idx) + 1j * cs_im(idx)


def acquire_timing_and_cfo(
    y_iq_4x: np.ndarray,
    x_known: np.ndarray,
    cfg: ClassicalConfig,
    *,
    cfo_prior_center: float | None = None,
    cfo_prior_width: float | None = None,
) -> tuple[float, float, np.ndarray, float]:
    """Jointly acquire fractional timing offset and CFO strictly from known pilot symbols.

    Returns (best_timing_rx, best_cfo_hz, y_sym_at_best_timing, pilot_corr_peak).
    """
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    t_rx = np.arange(len(y_c), dtype=np.float64)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)

    if cfo_prior_center is not None and cfo_prior_width is not None:
        f_lo = cfo_prior_center - cfo_prior_width
        f_hi = cfo_prior_center + cfo_prior_width
        n_steps = max(21, int(round((f_hi - f_lo) / 40.0)) + 1)
    else:
        f_lo = cfg.cfo_search_min_hz
        f_hi = cfg.cfo_search_max_hz
        n_steps = cfg.cfo_coarse_steps
    f_grid = np.linspace(f_lo, f_hi, n_steps)
    # Precompute derotation matrix for KNOWN_IDX: (F, 16)
    known_f = KNOWN_IDX.astype(np.float64)
    rot_mat = np.exp(-2j * math.pi * f_grid[:, None] * known_f[None, :] / FS_SYM)

    # 1) Coarse joint search over fractional timing offsets & CFO grid
    t_grid = np.linspace(-1.0, 1.0, cfg.timing_grid_steps)
    best_score = -1.0
    best_t = 0.0
    best_fi = 0
    for dt in t_grid:
        idx_k = (PAD_SYMS * SPS_NOM) + dt + SPS_NOM * known_f
        yk = cs_re(idx_k) + 1j * cs_im(idx_k)
        zk = yk * np.conj(x_known)
        scores = np.abs(rot_mat @ zk)
        fi = int(np.argmax(scores))
        sc = float(scores[fi])
        if sc > best_score:
            best_score = sc
            best_t = float(dt)
            best_fi = fi

    # 2) Fine parabolic timing refinement around best_t (+-0.25 rx samples)
    step_t = 2.0 / max(cfg.timing_grid_steps - 1, 1)
    f0 = float(f_grid[best_fi])
    rot0 = np.exp(-2j * math.pi * f0 * known_f / FS_SYM)
    t_tri = [best_t - step_t, best_t, best_t + step_t]
    s_tri = []
    for dt in t_tri:
        idx_k = (PAD_SYMS * SPS_NOM) + dt + SPS_NOM * known_f
        yk = cs_re(idx_k) + 1j * cs_im(idx_k)
        s_tri.append(float(abs(np.dot(yk * np.conj(x_known), rot0))))
    denom = s_tri[0] - 2.0 * s_tri[1] + s_tri[2]
    if denom < -1e-30:
        best_t = float(best_t + np.clip(0.5 * (s_tri[0] - s_tri[2]) / denom, -0.5, 0.5) * step_t)

    # 3) Fine CFO refinement around f0 (+- 1.5 coarse steps, 21 points + parabolic)
    df_coarse = (f_hi - f_lo) / max(n_steps - 1, 1)
    idx_k = (PAD_SYMS * SPS_NOM) + best_t + SPS_NOM * known_f
    yk = cs_re(idx_k) + 1j * cs_im(idx_k)
    zk = yk * np.conj(x_known)
    f_fine = np.linspace(f0 - 1.5 * df_coarse, f0 + 1.5 * df_coarse, 31)
    rot_f = np.exp(-2j * math.pi * f_fine[:, None] * known_f[None, :] / FS_SYM)
    sc_f = np.abs(rot_f @ zk)
    pfi = int(np.argmax(sc_f))
    best_f = float(f_fine[pfi])
    if 0 < pfi < len(f_fine) - 1:
        den_f = sc_f[pfi - 1] - 2.0 * sc_f[pfi] + sc_f[pfi + 1]
        if den_f < -1e-30:
            step_f = f_fine[1] - f_fine[0]
            best_f += float(np.clip(0.5 * (sc_f[pfi - 1] - sc_f[pfi + 1]) / den_f, -0.5, 0.5)) * step_f

    # Extract full 256 symbol stream at refined timing
    idx_all = (PAD_SYMS * SPS_NOM) + best_t + SPS_NOM * np.arange(PKT_LEN, dtype=np.float64)
    y_sym = cs_re(idx_all) + 1j * cs_im(idx_all)
    return best_t, best_f, y_sym, float(sc_f[pfi])


def run_classical_receiver(
    pkt: PacketSample,
    cfg: ClassicalConfig,
    *,
    mode: str = "mmse",
) -> ReceiverOutput:
    """Run classical single-packet QPSK receiver (`ls`, `mmse`, `mmse_slope`, or `mmse_3tap`)."""
    t0 = time.perf_counter()
    timing_rx, cfo_hz, y_sym, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)

    yk = y_derot[KNOWN_IDX]
    xk = pkt.x_known
    n_pil = len(KNOWN_IDX)

    # 1) LS scalar complex channel estimate on pilots
    h_ls = complex(np.vdot(xk, yk) / max(float(np.vdot(xk, xk).real), 1e-18))
    resid_ls = yk - h_ls * xk
    # Unbiased residual variance (1 complex DoF removed, or 2 if CFO estimated -> n_pil - 2)
    nv_resid = float(np.sum(np.abs(resid_ls) ** 2) / max(n_pil - 2, 1))
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    nv_est = max(cfg.noise_floor_mult * nv_bg, cfg.noise_blend * nv_resid + (1.0 - cfg.noise_blend) * nv_bg)

    # Prior signal power estimate from pilot energy minus noise (strictly positive floor)
    p_sig_est = max(float(np.mean(np.abs(yk) ** 2)) - nv_est, 0.05 * nv_est)
    h_var_post = nv_est / float(n_pil)

    if mode == "ls":
        h_env = np.full(PKT_LEN, h_ls, dtype=np.complex128)
        nv_eff = np.full(PKT_LEN, (nv_est + h_var_post) * cfg.var_calib_scale, dtype=np.float64)
        h_hat = h_env * rot_all
        x_eq = y_sym / (h_hat + 1e-18)
        bit_llrs = qpsk_bit_llrs_gaussian(y_sym, h_hat, nv_eff)
        hard_syms = qpsk_hard_symbols(x_eq)
    elif mode == "mmse":
        # Empirical Bayes / LMMSE scalar shrinkage
        p_h = p_sig_est * cfg.mmse_prior_scale
        lam = p_h / (p_h + nv_est / float(n_pil))
        h_mmse = lam * h_ls
        h_var_post = (p_h * (nv_est / float(n_pil))) / (p_h + nv_est / float(n_pil))
        h_env = np.full(PKT_LEN, h_mmse, dtype=np.complex128)
        nv_eff = np.full(PKT_LEN, (nv_est + h_var_post) * cfg.var_calib_scale, dtype=np.float64)
        h_hat = h_env * rot_all
        # LMMSE symbol equalizer: h^* y / (|h|^2 + nv_eff)
        x_eq = np.conj(h_hat) * y_sym / (np.abs(h_hat) ** 2 + nv_eff)
        bit_llrs = qpsk_bit_llrs_gaussian(y_sym, h_hat, nv_eff)
        hard_syms = qpsk_hard_symbols(x_eq)
    elif mode == "mmse_slope":
        # Regularized complex affine channel model h_env(n) = c0 + c1 * t_norm (tracks fast intra-packet drift/beat)
        t_all = (n_idx - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
        t_k = t_all[KNOWN_IDX]
        # Design matrix for yk = (c0 + c1*t_k) * xk + noise
        A = np.column_stack([xk, xk * t_k])  # (16, 2)
        p_h = p_sig_est * cfg.mmse_prior_scale
        # Prior covariance: diag([p_h, p_h / slope_reg])
        reg_diag = np.array([nv_est / max(p_h, 1e-18), nv_est * cfg.slope_reg / max(p_h, 1e-18)], dtype=np.float64)
        gram = A.conj().T @ A + np.diag(reg_diag)
        cov_c = np.linalg.inv(gram) * nv_est
        coef = np.linalg.solve(gram, A.conj().T @ yk)
        B_all = np.column_stack([np.ones(PKT_LEN), t_all])
        h_env = B_all @ coef
        # Exact predictive channel variance per symbol: b_n^T cov_c b_n
        ch_var_n = np.real(np.einsum("ni,ij,nj->n", B_all, cov_c, B_all))
        nv_eff = (nv_est + np.maximum(ch_var_n, 0.0)) * cfg.var_calib_scale
        h_hat = h_env * rot_all
        x_eq = np.conj(h_hat) * y_sym / (np.abs(h_hat) ** 2 + nv_eff)
        bit_llrs = qpsk_bit_llrs_gaussian(y_sym, h_hat, nv_eff)
        hard_syms = qpsk_hard_symbols(x_eq)
    elif mode == "mmse_3tap":
        # Regularized 3-tap symbol-spaced MMSE equalizer (handles fractional ISI & colored RRC noise)
        y_pad = extract_symbol_stream_padded(pkt.y_iq_4x, timing_rx, pad=1)
        n_pad = np.arange(-1, PKT_LEN + 1, dtype=np.float64)
        y_pad_derot = y_pad * np.exp(-2j * math.pi * cfo_hz * n_pad / FS_SYM)
        # Rows for each symbol n in 0..255: [y[n-1], y[n], y[n+1]]
        Y_all = np.column_stack([y_pad_derot[0:PKT_LEN], y_pad_derot[1:PKT_LEN + 1], y_pad_derot[2:PKT_LEN + 2]])
        Y_k = Y_all[KNOWN_IDX]
        # Solve regularized Wiener filter: min ||Y_k w - x_k||^2 + reg * ||w - w0||^2
        # Anchor center tap prior around 1-tap LMMSE equalizer w0 = [0, h_ls^* / (|h_ls|^2 + nv_est), 0]
        w0 = np.array([0.0j, np.conj(h_ls) / (abs(h_ls) ** 2 + nv_est + 1e-18), 0.0j], dtype=np.complex128)
        lam_reg = cfg.tap3_reg * (p_sig_est + nv_est)
        gram = Y_k.conj().T @ Y_k + lam_reg * np.eye(3)
        rhs = Y_k.conj().T @ xk + lam_reg * w0
        w_eq = np.linalg.solve(gram, rhs)
        x_eq = Y_all @ w_eq
        # Effective post-equalization gain & noise for soft LLRs and holdout channel prediction
        p_h = p_sig_est * cfg.mmse_prior_scale
        lam_s = p_h / (p_h + nv_est / float(n_pil))
        h_mmse = lam_s * h_ls
        h_hat = np.full(PKT_LEN, h_mmse, dtype=np.complex128) * rot_all
        # Unbiased gain of 3-tap equalizer on pilots
        g_eff = float(np.real(np.vdot(xk, x_eq[KNOWN_IDX])) / max(float(np.vdot(xk, xk).real), 1e-18))
        g_eff = max(g_eff, 0.1)
        res_eq = x_eq[KNOWN_IDX] - g_eff * xk
        eq_nv_pil = float(np.sum(np.abs(res_eq) ** 2) / max(n_pil - 3, 1))
        eq_nv_th = nv_est / max(abs(h_mmse) ** 2, 1e-18)
        eq_nv = max(0.25 * eq_nv_th, cfg.noise_blend * eq_nv_pil + (1.0 - cfg.noise_blend) * eq_nv_th) * cfg.var_calib_scale
        nv_eff = np.full(PKT_LEN, (nv_est + h_var_post) * cfg.var_calib_scale, dtype=np.float64)
        bit_llrs = qpsk_bit_llrs_gaussian(x_eq, np.full(PKT_LEN, g_eff, dtype=np.complex128), eq_nv)
        hard_syms = qpsk_hard_symbols(x_eq)
    else:
        raise ValueError(f"Unknown classical mode: {mode}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ReceiverOutput(
        y_sym=y_sym,
        h_hat=h_hat,
        nv_hat=nv_eff,
        x_eq=x_eq,
        bit_llrs=bit_llrs,
        hard_syms=hard_syms,
        est_timing_rx=timing_rx,
        est_cfo_hz=cfo_hz,
        est_noise_var=nv_est,
        elapsed_ms=elapsed_ms,
        extra={"h_ls_abs": abs(h_ls), "nv_resid": nv_resid},
    )


def evaluate_receiver_output(
    pkt: PacketSample,
    out: ReceiverOutput,
    method_name: str,
) -> tuple[PacketEvalRecord, np.ndarray, np.ndarray]:
    """Convert ReceiverOutput + ground-truth PacketSample into PacketEvalRecord + bit probs/labels."""
    if not (np.all(np.isfinite(out.x_eq)) and np.all(np.isfinite(out.h_hat)) and np.all(np.isfinite(out.bit_llrs))):
        rec = PacketEvalRecord(
            unit_id=pkt.unit_id,
            capture_id=pkt.capture_id,
            freq_mhz=pkt.freq_mhz,
            nom_snr_db=pkt.nom_snr_db,
            cluster_id=pkt.cluster_id,
            split=pkt.split,
            method=method_name,
            success=False,
            error_msg="non-finite receiver output",
            elapsed_ms=out.elapsed_ms,
        )
        return rec, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint8)

    # Interleaved bit indices for a symbol subset
    def _bit_slice(sym_idx: np.ndarray) -> np.ndarray:
        b = np.empty(2 * len(sym_idx), dtype=np.int64)
        b[0::2] = 2 * sym_idx
        b[1::2] = 2 * sym_idx + 1
        return b

    pay_bidx = _bit_slice(PAYLOAD_IDX)
    sync_bidx = _bit_slice(SYNC_IDX)
    pil_bidx = _bit_slice(PILOT_IDX)

    m_pay = compute_segment_metrics(
        out.x_eq[PAYLOAD_IDX],
        pkt.x_all_true[PAYLOAD_IDX],
        bit_llrs=out.bit_llrs[pay_bidx],
        hard_syms_override=out.hard_syms[PAYLOAD_IDX],
    )
    m_sync = compute_segment_metrics(
        out.x_eq[SYNC_IDX],
        pkt.x_all_true[SYNC_IDX],
        bit_llrs=out.bit_llrs[sync_bidx],
        hard_syms_override=out.hard_syms[SYNC_IDX],
    )
    m_pil = compute_segment_metrics(
        out.x_eq[PILOT_IDX],
        pkt.x_all_true[PILOT_IDX],
        bit_llrs=out.bit_llrs[pil_bidx],
        hard_syms_override=out.hard_syms[PILOT_IDX],
    )
    m_hold = compute_holdout_channel_metrics(
        out.y_sym[HOLDOUT_IDX],
        pkt.x_all_true[HOLDOUT_IDX],
        out.h_hat[HOLDOUT_IDX],
        out.nv_hat[HOLDOUT_IDX],
        h_offline_ref=pkt.h_offline_ref[HOLDOUT_IDX],
    )

    llrs_pay = np.clip(out.bit_llrs[pay_bidx], -35.0, 35.0)
    probs_pay = 1.0 / (1.0 + np.exp(-llrs_pay))
    labels_pay = qpsk_to_bits(pkt.x_all_true[PAYLOAD_IDX])

    rec = PacketEvalRecord(
        unit_id=pkt.unit_id,
        capture_id=pkt.capture_id,
        freq_mhz=pkt.freq_mhz,
        nom_snr_db=pkt.nom_snr_db,
        cluster_id=pkt.cluster_id,
        split=pkt.split,
        method=method_name,
        success=True,
        elapsed_ms=out.elapsed_ms,
        payload_ser=m_pay["ser"],
        payload_ber=m_pay["ber"],
        payload_per=1.0 if m_pay["sym_errors"] > 0 else 0.0,
        payload_sym_mse=m_pay["sym_mse"],
        payload_evm_raw_db=m_pay["evm_raw_db"],
        payload_evm_unbiased_db=m_pay["evm_unbiased_db"],
        payload_bit_nll=m_pay.get("bit_nll_nats", float("nan")),
        payload_bit_brier=m_pay.get("bit_brier", float("nan")),
        payload_n_syms=int(m_pay["n_syms"]),
        payload_sym_errors=int(m_pay["sym_errors"]),
        payload_bit_errors=int(m_pay["bit_errors"]),
        sync_ser=m_sync["ser"],
        sync_ber=m_sync["ber"],
        sync_evm_unbiased_db=m_sync["evm_unbiased_db"],
        pilot_ser=m_pil["ser"],
        pilot_ber=m_pil["ber"],
        pilot_evm_unbiased_db=m_pil["evm_unbiased_db"],
        holdout_obs_nmse=m_hold["holdout_obs_nmse"],
        holdout_ref_nmse=m_hold.get("holdout_ref_nmse", float("nan")),
        holdout_ch_nll=m_hold["holdout_ch_nll_nats"],
        holdout_var_ratio=m_hold["holdout_var_ratio"],
        extra={
            "est_cfo_hz": out.est_cfo_hz,
            "cfo_err_hz": out.est_cfo_hz - pkt.offline_cfo_hz,
            "est_timing_rx": out.est_timing_rx,
            **out.extra,
        },
    )
    return rec, probs_pay, labels_pay
