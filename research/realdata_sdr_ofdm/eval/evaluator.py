"""Unified leak-free evaluator, paired statistical audit, profiler, and stress suite (F7)."""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
import tracemalloc
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import t as student_t

from p12.audit import exact_sign_flip_p, holm_adjust, verdict_superiority
from p9.audit import exact_sign_flip_test
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    RealOFDMFrame,
    apply_stress_perturbation,
)

try:
    import resource as _resource
except ImportError:  # Windows
    _resource = None


def _get_peak_rss_mb() -> float:
    """Return process RSS / peak RSS in MB without hard third-party dependencies."""
    if _resource is not None:
        try:
            usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            if usage > 0:
                if sys.platform == "darwin":
                    return float(usage) / (1024.0 * 1024.0)
                return float(usage) / 1024.0
        except Exception:
            pass

    if os.name == "nt":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = ctypes.c_void_p
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            get_process_memory_info.restype = ctypes.c_int

            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = get_current_process()
            if handle and get_process_memory_info(handle, ctypes.byref(pmc), pmc.cb):
                rss_bytes = max(int(pmc.PeakWorkingSetSize), int(pmc.WorkingSetSize))
                if rss_bytes > 0:
                    return float(rss_bytes) / (1024.0 * 1024.0)
        except Exception:
            pass

    try:
        import psutil  # type: ignore[import-not-found]

        rss_bytes = int(psutil.Process().memory_info().rss)
        if rss_bytes > 0:
            return float(rss_bytes) / (1024.0 * 1024.0)
    except Exception:
        pass

    if tracemalloc.is_tracing():
        _, peak = tracemalloc.get_traced_memory()
        if peak > 0:
            return float(peak) / (1024.0 * 1024.0)
    return 1.0


def _resolve_heldout_indices(frame: RealOFDMFrame, out: Any, pred_len: int) -> Optional[list[int]]:
    aux = getattr(out, "aux_info", None) or {}
    cand = aux.get("heldout_pilot_indices")
    if cand is not None and len(cand) == pred_len:
        return [int(i) for i in cand]

    meta = getattr(frame, "meta", None) or {}
    cand_meta = meta.get("heldout_pilot_indices")
    if cand_meta is not None and len(cand_meta) == pred_len:
        return [int(i) for i in cand_meta]

    if pred_len == 12:
        return list(range(1, 13))

    folds = meta.get("heldout_pilot_folds") or []
    if folds:
        flat: list[int] = []
        seen: set[int] = set()
        for f in folds:
            for i in f.get("heldout_pilot_indices", ()):
                ii = int(i)
                if ii not in seen:
                    seen.add(ii)
                    flat.append(ii)
        if len(flat) == pred_len:
            return sorted(flat)
        fold0 = [int(i) for i in folds[0].get("heldout_pilot_indices", ())]
        if len(fold0) == pred_len:
            return fold0
    return None


def _is_valid_output(frame: RealOFDMFrame, out: Any) -> tuple[bool, Optional[str]]:
    aux = getattr(out, "aux_info", None) or {}
    if bool(aux.get("failed", False)):
        return False, str(aux.get("error") or "receiver marked failed=True")

    try:
        h_est = np.asarray(out.h_est)
        h_var = np.asarray(out.h_var)
        eq_symbols = np.asarray(out.eq_symbols)
        bit_llrs = np.asarray(out.bit_llrs)
        hard_bits = np.asarray(out.hard_bits)
        hard_symbols = np.asarray(out.hard_symbols)
    except Exception as exc:  # noqa: BLE001
        return False, f"AttributeError: {exc}"

    for arr_name, arr in (
        ("h_est", h_est),
        ("h_var", h_var),
        ("eq_symbols", eq_symbols),
        ("bit_llrs", bit_llrs),
        ("hard_bits", hard_bits),
        ("hard_symbols", hard_symbols),
    ):
        if not np.all(np.isfinite(arr)):
            return False, f"Non-finite values in {arr_name}"

    if np.iscomplexobj(h_var) or not np.all(h_var >= 0.0):
        return False, "Negative or complex values in h_var"
    if bit_llrs.shape != frame.tx_bits.shape:
        return False, f"bit_llrs shape {bit_llrs.shape} != {frame.tx_bits.shape}"
    if hard_bits.shape != frame.tx_bits.shape:
        return False, f"hard_bits shape {hard_bits.shape} != {frame.tx_bits.shape}"
    if eq_symbols.shape != frame.tx_symbols.shape:
        return False, f"eq_symbols shape {eq_symbols.shape} != {frame.tx_symbols.shape}"

    p_pred_raw = getattr(out, "heldout_pilot_pred", None)
    p_var_raw = getattr(out, "heldout_pilot_var", None)
    if p_pred_raw is not None:
        try:
            p_pred = np.asarray(p_pred_raw)
        except Exception as exc:  # noqa: BLE001
            return False, f"Invalid heldout_pilot_pred: {exc}"
        if p_pred.size == 0 or not np.all(np.isfinite(p_pred)):
            return False, "Empty or non-finite values in heldout_pilot_pred"
    if p_var_raw is not None:
        try:
            p_var = np.asarray(p_var_raw)
        except Exception as exc:  # noqa: BLE001
            return False, f"Invalid heldout_pilot_var: {exc}"
        if (
            p_var.size == 0
            or np.iscomplexobj(p_var)
            or not np.all(np.isfinite(p_var))
            or not np.all(p_var >= 0.0)
        ):
            return False, "Empty, non-finite, complex, or negative values in heldout_pilot_var"
    if p_pred_raw is not None and p_var_raw is not None and p_pred.shape != p_var.shape:
        return False, f"heldout_pilot_pred shape {p_pred.shape} != heldout_pilot_var shape {p_var.shape}"
    return True, None


def evaluate_receiver(
    pairs: Sequence[tuple[RealOFDMFrame, Any]],
    *,
    warm_frames: Optional[int] = None,
    receiver_fn: Optional[Callable[[RealOFDMFrame], Any]] = None,
    param_count: Optional[int] = None,
    train_time_s: float = 0.0,
    metric_mode: str = "raw",
    peak_rss_mb: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate a sequence of (RealOFDMFrame, ReceiverOutput) pairs under unified F7 metrics."""
    del warm_frames, receiver_fn
    if pairs is None or len(pairs) == 0:
        raise ValueError("pairs must be a non-empty sequence of (frame, output)")

    total_frames = int(len(pairs))
    success_count = 0
    fail_count = 0
    timeout_count = 0
    errors: list[dict[str, Any]] = []

    bit_errors = 0
    total_bits = 0
    frame_errors = 0
    sym_errors = 0
    total_syms = 0
    sq_sym_err = 0.0
    sq_sym_sig = 0.0

    pilot_sq_err = 0.0
    pilot_z_sq = 0.0
    pilot_sig2_sum = 0.0
    pilot_cal_list: list[np.ndarray] = []
    pilot_nll_list: list[np.ndarray] = []

    latencies_ms: list[float] = []
    n_candidates_vals: list[int] = []
    inferred_params: list[int] = []

    for frame, out in pairs:
        aux = getattr(out, "aux_info", None) or {}
        if bool(aux.get("timeout", False)):
            timeout_count += 1

        ok, err_msg = _is_valid_output(frame, out)
        if not ok:
            fail_count += 1
            errors.append({"frame_id": int(frame.frame_id), "error": str(err_msg)})
            continue

        success_count += 1
        hb = np.asarray(out.hard_bits, dtype=np.uint8)
        tb = np.asarray(frame.tx_bits, dtype=np.uint8)
        bit_diff = hb != tb
        be = int(np.sum(bit_diff))
        bit_errors += be
        total_bits += int(tb.size)
        if be > 0:
            frame_errors += 1
        sym_errors += int(np.sum(np.any(bit_diff, axis=-1)))
        total_syms += int(frame.tx_symbols.size)

        eq = np.asarray(out.eq_symbols, dtype=np.complex128)
        txs = np.asarray(frame.tx_symbols, dtype=np.complex128)
        sq_sym_err += float(np.sum(np.abs(eq - txs) ** 2))
        sq_sym_sig += float(np.sum(np.abs(txs) ** 2))

        if "latency_ms" in aux and np.isfinite(float(aux["latency_ms"])):
            latencies_ms.append(float(aux["latency_ms"]))
        n_candidates_vals.append(int(aux.get("n_candidates", 1)))
        if "param_count" in aux:
            inferred_params.append(int(aux["param_count"]))

        p_pred_raw = getattr(out, "heldout_pilot_pred", None)
        p_var_raw = getattr(out, "heldout_pilot_var", None)
        if p_pred_raw is not None and p_var_raw is not None:
            p_pred = np.asarray(p_pred_raw, dtype=np.complex128).ravel()
            p_var = np.asarray(p_var_raw, dtype=np.float64).ravel()
            if (
                p_pred.size > 0
                and p_pred.size == p_var.size
                and np.all(np.isfinite(p_pred))
                and np.all(np.isfinite(p_var))
            ):
                idx = _resolve_heldout_indices(frame, out, int(p_pred.size))
                if idx is not None and len(idx) == p_pred.size:
                    fft_idx = [PILOT_FFT_INDICES[i] for i in idx]
                    y_p = np.asarray(frame.rx_grid[PILOT_SYMBOL_IDX, fft_idx], dtype=np.complex128)
                    p_val = np.asarray(frame.pilot_values[idx], dtype=np.complex128)
                    z_obs = y_p / p_val
                    sigma2_obs = float(frame.est_n0) / (np.abs(p_val) ** 2)
                    v_eff = np.maximum(np.maximum(p_var, 0.0) + sigma2_obs, 1e-30)
                    err = p_pred - z_obs
                    err_sq = np.abs(err) ** 2
                    pilot_sq_err += float(np.sum(err_sq))
                    pilot_z_sq += float(np.sum(np.abs(z_obs) ** 2))
                    pilot_sig2_sum += float(np.sum(sigma2_obs))
                    pilot_cal_list.append(err_sq / v_eff)
                    pilot_nll_list.append(np.log(math.pi * v_eff) + err_sq / v_eff)

    coverage_pct = 100.0 * float(success_count) / float(total_frames)
    ber = float(bit_errors) / float(max(total_bits, 1))
    fer = float(frame_errors) / float(max(success_count, 1))
    ser = float(sym_errors) / float(max(total_syms, 1))
    sym_mse = float(sq_sym_err) / float(max(total_syms, 1))
    evm_pct = 100.0 * math.sqrt(float(sq_sym_err) / float(max(sq_sym_sig, 1e-30)))

    if pilot_cal_list:
        cal_cat = np.concatenate(pilot_cal_list)
        nll_cat = np.concatenate(pilot_nll_list)
        pilot_nmse_raw = float(pilot_sq_err / max(pilot_z_sq, 1e-30))
        pilot_nmse_denoised = float(
            max(0.0, pilot_sq_err - pilot_sig2_sum) / max(pilot_z_sq - pilot_sig2_sum, 1e-12)
        )
        pilot_nmse = pilot_nmse_denoised if metric_mode == "denoised" else pilot_nmse_raw
        pilot_cal_ratio = float(np.mean(cal_cat))
        pilot_nll = float(np.mean(nll_cat))
    else:
        pilot_nmse_raw = 0.0
        pilot_nmse_denoised = 0.0
        pilot_nmse = 0.0
        pilot_cal_ratio = 0.0
        pilot_nll = 0.0

    if latencies_ms:
        lat_arr = np.asarray(latencies_ms, dtype=np.float64)
        mean_ms = float(np.mean(lat_arr))
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
    else:
        mean_ms = 0.0
        median_ms = 0.0
        p95_ms = 0.0

    rss_mb = float(peak_rss_mb) if (peak_rss_mb is not None and float(peak_rss_mb) > 0.0) else _get_peak_rss_mb()
    eff_n_cand = int(max(n_candidates_vals)) if n_candidates_vals else 1
    eff_param_count = (
        int(param_count)
        if param_count is not None
        else (int(max(inferred_params)) if inferred_params else 0)
    )

    return {
        "total_frames": int(total_frames),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "timeout_count": int(timeout_count),
        "coverage_pct": float(coverage_pct),
        "bit_errors": int(bit_errors),
        "total_bits": int(total_bits),
        "ber": float(ber),
        "frame_errors": int(frame_errors),
        "fer": float(fer),
        "sym_errors": int(sym_errors),
        "total_syms": int(total_syms),
        "ser": float(ser),
        "sym_mse": float(sym_mse),
        "evm_pct": float(evm_pct),
        "pilot_nmse": float(pilot_nmse),
        "pilot_nmse_raw": float(pilot_nmse_raw),
        "pilot_nmse_denoised": float(pilot_nmse_denoised),
        "pilot_cal_ratio": float(pilot_cal_ratio),
        "pilot_nll": float(pilot_nll),
        "mean_ms_per_frame": float(mean_ms),
        "median_ms_per_frame": float(median_ms),
        "p95_ms_per_frame": float(p95_ms),
        "peak_rss_mb": float(rss_mb),
        "n_candidates": int(eff_n_cand),
        "param_count": int(eff_param_count),
        "train_time_s": float(train_time_s),
        "errors": errors,
    }


def _group_metric_num_den(
    pairs: Sequence[tuple[RealOFDMFrame, Any]],
    metric: str,
) -> tuple[float, float, float]:
    """Compute (value_in_pp_or_pct, numerator, denominator) for one cluster."""
    m = str(metric).strip().lower()
    if m not in ("ber", "fer", "ser", "sym_mse", "evm_pct", "pilot_nmse"):
        raise ValueError(f"Unsupported paired comparison metric: {metric!r}")

    valid_pairs = [(f, o) for f, o in pairs if _is_valid_output(f, o)[0]]
    if not valid_pairs:
        raise ValueError("Cluster has 0 valid receiver outputs; cannot compute paired comparison metric")

    if m == "ber":
        err = sum(int(np.sum(np.asarray(o.hard_bits, dtype=np.uint8) != np.asarray(f.tx_bits, dtype=np.uint8))) for f, o in valid_pairs)
        tot = sum(int(f.tx_bits.size) for f, _ in valid_pairs)
        return 100.0 * float(err) / float(max(tot, 1)), float(err), float(tot)
    if m == "fer":
        err = sum(1 for f, o in valid_pairs if int(np.sum(np.asarray(o.hard_bits, dtype=np.uint8) != np.asarray(f.tx_bits, dtype=np.uint8))) > 0)
        tot = len(valid_pairs)
        return 100.0 * float(err) / float(max(tot, 1)), float(err), float(tot)
    if m == "ser":
        err = sum(
            int(np.sum(np.any(np.asarray(o.hard_bits, dtype=np.uint8) != np.asarray(f.tx_bits, dtype=np.uint8), axis=-1)))
            for f, o in valid_pairs
        )
        tot = sum(int(f.tx_symbols.size) for f, _ in valid_pairs)
        return 100.0 * float(err) / float(max(tot, 1)), float(err), float(tot)

    res = evaluate_receiver(valid_pairs, peak_rss_mb=1.0)
    if m == "sym_mse":
        val = 100.0 * float(res["sym_mse"])
        return val, float(res["sym_mse"]) * float(res["total_syms"]), float(res["total_syms"])
    if m == "evm_pct":
        return float(res["evm_pct"]), float(res["evm_pct"]), 1.0
    return 100.0 * float(res["pilot_nmse"]), float(res["pilot_nmse"]), 1.0


def _sign_flip_stats(
    diffs_pp: Sequence[float],
    *,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Dispatch exact G=16 -> p12.audit, 2<=G<16 -> p9.audit, and handle G<2 / G>16."""
    d = np.asarray(diffs_pp, dtype=np.float64).ravel()
    g = int(len(d))
    if g == 16:
        return exact_sign_flip_p(d)

    tol = 1e-12
    wins = int(np.sum(d > tol))
    ties = int(np.sum(np.abs(d) <= tol))
    losses = int(np.sum(d < -tol))
    obs = float(np.mean(d)) if g > 0 else 0.0

    if g < 2:
        return {
            "mean_pp": obs,
            "sd_pp": 0.0,
            "se_pp": 0.0,
            "ci95_lo_pp": obs,
            "ci95_hi_pp": obs,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "p_two_sided": 1.0,
            "p_one_sided_pos": 1.0 if obs >= -tol else 0.5,
            "cluster_diffs_pp": [float(x) for x in d],
        }

    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(float(g))
    t_975 = float(student_t.ppf(0.975, df=g - 1))

    if 2 <= g < 16:
        p9_stat = exact_sign_flip_test(d, expected_sign=1)
        n_states = 1 << g
        bits = ((np.arange(n_states, dtype=np.int64)[:, None] >> np.arange(g, dtype=np.int64)[None, :]) & 1)
        signs = np.where(bits == 1, 1.0, -1.0)
        means = (signs @ d) / float(g)
        p_pos = float(np.mean(means >= obs - tol))
        return {
            "mean_pp": float(p9_stat["mean_diff"]),
            "sd_pp": float(p9_stat["std_diff"]),
            "se_pp": se,
            "ci95_lo_pp": float(p9_stat["ci95"][0]),
            "ci95_hi_pp": float(p9_stat["ci95"][1]),
            "wins": int(p9_stat["wins"]),
            "ties": int(p9_stat["ties"]),
            "losses": int(p9_stat["losses"]),
            "p_two_sided": float(p9_stat["raw_p"]),
            "p_one_sided_pos": p_pos,
            "cluster_diffs_pp": [float(x) for x in d],
        }

    # G > 16: deterministic Monte-Carlo sign-flip permutation (65,536 draws)
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(65536, g))
    means = (signs @ d) / float(g)
    p_two = float(np.mean(np.abs(means) >= abs(obs) - tol))
    p_pos = float(np.mean(means >= obs - tol))
    return {
        "mean_pp": obs,
        "sd_pp": sd,
        "se_pp": se,
        "ci95_lo_pp": obs - t_975 * se,
        "ci95_hi_pp": obs + t_975 * se,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_two_sided": p_two,
        "p_one_sided_pos": p_pos,
        "cluster_diffs_pp": [float(x) for x in d],
    }


def compare_arms(
    base_pairs: Sequence[tuple[RealOFDMFrame, Any]],
    cand_pairs: Sequence[tuple[RealOFDMFrame, Any]],
    *,
    metric: str = "ber",
    group_fn: Optional[Callable[[RealOFDMFrame], str]] = None,
    alpha: float = 0.05,
    n_boot: int = 4000,
    seed: int = 20261121,
    p_holm: Optional[float] = None,
) -> dict[str, Any]:
    """Paired cluster-level comparison between anchor `base_pairs` and `cand_pairs`.

    Positive `mean_pp` means `cand` has lower error than `base`.
    """
    if len(base_pairs) == 0 or len(base_pairs) != len(cand_pairs):
        raise ValueError("base_pairs and cand_pairs must be non-empty and equal length")

    group_order: list[str] = []
    grouped_base: dict[str, list[tuple[RealOFDMFrame, Any]]] = {}
    grouped_cand: dict[str, list[tuple[RealOFDMFrame, Any]]] = {}

    for (fb, ob), (fc, oc) in zip(base_pairs, cand_pairs):
        if int(fb.frame_id) != int(fc.frame_id):
            raise ValueError(f"Mismatched paired frame_id: {fb.frame_id} != {fc.frame_id}")
        gid = str(group_fn(fb) if group_fn is not None else fb.group_id)
        if gid not in grouped_base:
            group_order.append(gid)
            grouped_base[gid] = []
            grouped_cand[gid] = []
        grouped_base[gid].append((fb, ob))
        grouped_cand[gid].append((fc, oc))

    diffs_pp: list[float] = []
    base_nums, base_dens = [], []
    cand_nums, cand_dens = [], []
    for gid in group_order:
        try:
            vb, nb, db = _group_metric_num_den(grouped_base[gid], metric)
            vc, nc, dc = _group_metric_num_den(grouped_cand[gid], metric)
        except ValueError as exc:
            if "0 valid receiver outputs" in str(exc):
                raise ValueError(f"Cluster {gid!r} has 0 valid receiver outputs") from exc
            raise
        diffs_pp.append(vb - vc)
        base_nums.append(nb)
        base_dens.append(db)
        cand_nums.append(nc)
        cand_dens.append(dc)

    stat = _sign_flip_stats(diffs_pp, seed=seed)
    g = len(group_order)
    d_arr = np.asarray(diffs_pp, dtype=np.float64)
    bn_arr = np.asarray(base_nums, dtype=np.float64)
    bd_arr = np.asarray(base_dens, dtype=np.float64)
    cn_arr = np.asarray(cand_nums, dtype=np.float64)
    cd_arr = np.asarray(cand_dens, dtype=np.float64)

    if g >= 2 and int(n_boot) > 0:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, g, size=(int(n_boot), g))
        if str(metric).lower() in ("ber", "fer", "ser"):
            b_rates = 100.0 * np.sum(bn_arr[idx], axis=1) / np.maximum(np.sum(bd_arr[idx], axis=1), 1.0)
            c_rates = 100.0 * np.sum(cn_arr[idx], axis=1) / np.maximum(np.sum(cd_arr[idx], axis=1), 1.0)
            boot_diffs = b_rates - c_rates
        else:
            boot_diffs = np.mean(d_arr[idx], axis=1)
        boot_lo = float(np.quantile(boot_diffs, 0.025))
        boot_hi = float(np.quantile(boot_diffs, 0.975))
    else:
        boot_lo = float(stat["mean_pp"])
        boot_hi = float(stat["mean_pp"])

    p_holm_eff = float(stat["p_two_sided"]) if p_holm is None else float(p_holm)
    verdict = verdict_superiority(stat, p_holm_eff, alpha=float(alpha))

    base_tot_err = float(np.sum(bn_arr))
    cand_tot_err = float(np.sum(cn_arr))
    tot_units = float(np.sum(bd_arr))

    out = dict(stat)
    out.update(
        {
            "metric": str(metric),
            "n_groups": int(g),
            "group_ids": list(group_order),
            "p_holm": float(p_holm_eff),
            "verdict": verdict,
            "boot_ci95_lo_pp": boot_lo,
            "boot_ci95_hi_pp": boot_hi,
            "base_total_errors": int(round(base_tot_err)) if str(metric).lower() in ("ber", "fer", "ser") else base_tot_err,
            "cand_total_errors": int(round(cand_tot_err)) if str(metric).lower() in ("ber", "fer", "ser") else cand_tot_err,
            "error_delta": int(round(cand_tot_err - base_tot_err)) if str(metric).lower() in ("ber", "fer", "ser") else float(cand_tot_err - base_tot_err),
            "total_units": int(round(tot_units)) if str(metric).lower() in ("ber", "fer", "ser") else tot_units,
        }
    )
    return out


def compare_multiple_arms(
    arm_pairs: Mapping[str, Sequence[tuple[RealOFDMFrame, Any]]],
    *,
    anchor_name: str,
    metric: str = "ber",
    group_fn: Optional[Callable[[RealOFDMFrame], str]] = None,
    alpha: float = 0.05,
    n_boot: int = 4000,
    seed: int = 20261121,
) -> dict[str, dict[str, Any]]:
    """Run paired comparisons for each arm vs `anchor_name` with Holm step-down correction."""
    if anchor_name not in arm_pairs:
        raise ValueError(f"Anchor arm {anchor_name!r} not found in arm_pairs")
    base_pairs = arm_pairs[anchor_name]
    names = [k for k in arm_pairs if k != anchor_name]
    if not names:
        return {}

    raw_stats = [
        compare_arms(
            base_pairs,
            arm_pairs[name],
            metric=metric,
            group_fn=group_fn,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        for name in names
    ]
    p_holms = holm_adjust([st["p_two_sided"] for st in raw_stats])
    results: dict[str, dict[str, Any]] = {}
    for name, st, ph in zip(names, raw_stats, p_holms):
        st["p_holm"] = float(ph)
        st["verdict"] = verdict_superiority(st, float(ph), alpha=float(alpha))
        results[name] = st
    return results


def subblock_group_ids(
    frames: Sequence[RealOFDMFrame],
    *,
    target_g: int = 16,
) -> dict[int, str]:
    """Partition each contiguous macro-block into sub-blocks totaling `target_g` clusters.

    Caveat: on `sdr_ofdm_dataset_v1`, `h_pilot` lag-32 correlation is ~0.736, so
    sub-blocks within the same macro-block share residual RF channel correlation.
    """
    by_macro: dict[str, list[RealOFDMFrame]] = {}
    macro_order: list[str] = []
    for fr in frames:
        gid = str(fr.group_id)
        if gid not in by_macro:
            macro_order.append(gid)
            by_macro[gid] = []
        by_macro[gid].append(fr)

    m = len(macro_order)
    if m == 0 or int(target_g) < m:
        return {int(fr.frame_id): str(fr.group_id) for fr in frames}

    base_sub = int(target_g) // m
    rem = int(target_g) % m
    mapping: dict[int, str] = {}
    for idx, mgid in enumerate(macro_order):
        k_sub = base_sub + (1 if idx >= (m - rem) else 0)
        sub_frames = sorted(by_macro[mgid], key=lambda f: int(f.frame_id))
        chunks = np.array_split(np.arange(len(sub_frames)), k_sub)
        for s_idx, chunk in enumerate(chunks):
            label = f"{mgid}_sub{s_idx:02d}"
            for pos in chunk:
                mapping[int(sub_frames[int(pos)].frame_id)] = label
    return mapping


def profile_receiver_pinned(
    frames: Sequence[RealOFDMFrame],
    receiver_fn: Callable[[RealOFDMFrame], Any],
    *,
    warmup_frames: int = 8,
    param_count: Optional[int] = None,
    train_time_s: float = 0.0,
) -> dict[str, Any]:
    """Two-pass pinned single-thread profiler (Pass 1 latency, Pass 2 tracemalloc peak)."""
    if not frames:
        raise ValueError("frames must be non-empty")

    n_warm = min(max(int(warmup_frames), 0), len(frames))
    for i in range(n_warm):
        _ = receiver_fn(frames[i])

    # Pass 1: clean wall-clock timing without tracemalloc overhead
    latencies_ms: list[float] = []
    pairs: list[tuple[RealOFDMFrame, Any]] = []
    for fr in frames:
        t0 = time.perf_counter()
        out = receiver_fn(fr)
        dt_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
        latencies_ms.append(dt_ms)
        pairs.append((fr, out))

    # Pass 2: tracemalloc peak allocation on representative subset
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    for fr in frames[: min(16, len(frames))]:
        _ = receiver_fn(fr)
    _, peak_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    res = evaluate_receiver(
        pairs,
        param_count=param_count,
        train_time_s=train_time_s,
    )
    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    res["mean_ms_per_frame"] = float(np.mean(lat_arr))
    res["median_ms_per_frame"] = float(np.median(lat_arr))
    res["p95_ms_per_frame"] = float(np.percentile(lat_arr, 95))
    res["tracemalloc_peak_mb"] = float(peak_bytes) / (1024.0 * 1024.0)
    return res


def run_stress_suite(
    frames: Sequence[RealOFDMFrame],
    receiver_fn: Callable[[RealOFDMFrame], Any],
    *,
    extra_awgn_db: Sequence[Optional[float]] = (None, 10.0, 5.0, 0.0),
    cfo_norm: Sequence[float] = (0.0, 0.02, 0.05),
    phase_drift_rad: Sequence[float] = (0.0, 0.3, 0.6),
    seed: int = 20261122,
    param_count: Optional[int] = None,
) -> dict[str, Any]:
    """Run reproducible single-axis stress sweeps without mutating input frames."""
    if not frames:
        raise ValueError("frames must be non-empty")

    conditions: list[dict[str, Any]] = []
    clean_pairs = [(fr, receiver_fn(fr)) for fr in frames]
    clean_metrics = evaluate_receiver(clean_pairs, param_count=param_count)

    seen_specs: set[tuple[Optional[float], float, float]] = set()
    specs: list[tuple[str, Optional[float], float, float]] = []
    for awgn in extra_awgn_db:
        key = (None if awgn is None else float(awgn), 0.0, 0.0)
        if key not in seen_specs:
            seen_specs.add(key)
            tag = "clean" if awgn is None else f"awgn_{float(awgn):.1f}db"
            specs.append((tag, key[0], 0.0, 0.0))
    for cfo in cfo_norm:
        key = (None, float(cfo), 0.0)
        if key not in seen_specs:
            seen_specs.add(key)
            specs.append((f"cfo_{float(cfo):.3f}", None, float(cfo), 0.0))
    for drift in phase_drift_rad:
        key = (None, 0.0, float(drift))
        if key not in seen_specs:
            seen_specs.add(key)
            specs.append((f"drift_{float(drift):.2f}rad", None, 0.0, float(drift)))

    for idx_cond, (tag, awgn, cfo, drift) in enumerate(specs):
        if awgn is None and cfo == 0.0 and drift == 0.0:
            metrics = dict(clean_metrics)
        else:
            pert_frames = [
                apply_stress_perturbation(
                    fr,
                    extra_awgn_db=awgn,
                    cfo_norm=cfo,
                    phase_drift_rad=drift,
                    seed=int(seed) + idx_cond * 10000 + int(fr.frame_id),
                )
                for fr in frames
            ]
            metrics = evaluate_receiver(
                [(pf, receiver_fn(pf)) for pf in pert_frames],
                param_count=param_count,
            )
        conditions.append(
            {
                "condition": tag,
                "extra_awgn_db": awgn,
                "cfo_norm": float(cfo),
                "phase_drift_rad": float(drift),
                "metrics": metrics,
            }
        )

    return {
        "clean": clean_metrics,
        "conditions": conditions,
    }


paired_comparison = compare_arms
evaluate_paired = compare_arms
evaluate_frames = evaluate_receiver
evaluate_outputs = evaluate_receiver

__all__ = [
    "compare_arms",
    "compare_multiple_arms",
    "evaluate_frames",
    "evaluate_outputs",
    "evaluate_paired",
    "evaluate_receiver",
    "paired_comparison",
    "profile_receiver_pinned",
    "run_stress_suite",
    "subblock_group_ids",
]
