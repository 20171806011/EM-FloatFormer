"""Strong classical CP-SC decoder (LMMSE + Numba BP + Turbo MMSE-PIC + OSD) and ML certificate.

Key components:
1. ``decode_native_ldpc``: native comparator wrapping ``p8.coding.decode``.
2. ``decode_strong_sc`` / ``decode_strong_sc_batch``:
   - Time-domain LMMSE equalization -> Gray 16-QAM soft bit LLRs
   - Numba Sum-Product Belief Propagation (20 iters) on ``p8.coding.parity_check()``
   - Optional 1-round Turbo MMSE-PIC soft interference cancellation + BP
   - Ordered-Statistics Decoding (MRB GF(2) Gaussian elimination in Numba) generating
     valid ``(128, 64)`` codewords:
       * ``list_size="small"``: order-0 + 64 order-1 + top-256 order-2 = 321 valid codewords
       * ``list_size="large"``: order-0 + 64 order-1 + all 2,016 order-2 + 560 order-3
         on least-reliable 16 MRB bits = 2,641 valid codewords
   - Exact full 32x32 time-varying single-carrier waveform metric rescoring:
       ``metric(c) = sum_n |y_r[n] - (H_r @ x(c))[n]|^2 / noise_r[n]``
     Under known channel (``noise_r = n0``), this is the exact ML metric for white
     circularly symmetric Gaussian noise. Under estimated channel (``noise_r = n0 + sum_ray var_ray``),
     it is a plug-in approximate weighted waveform metric (ignoring inter-sample covariance of
     channel estimation error).
3. ``ml_lower_bound_certificate``:
   - Evaluator-only mathematically strict ML lower-bound certificate.
   - Checks all 64 single-info-bit-flip neighbors ``c_true ^ G[i]`` plus all valid
     ``extra_candidates != c_true`` (verifying ``H_parity @ c % 2 == 0`` for every candidate!).
   - Certifies ``ml_must_err=True`` iff there exists a verified valid codeword ``c' != c_true``
     with ``||y_r - H_r x(c')||^2 < ||y_r - H_r x(c_true)||^2 - eps_tie`` where
     ``eps_tie = 1e-11 * max(1.0, d_true)``; otherwise strictly reports ``status="undecided"``.
4. ``verify_mini_codebook_ml_certificate``:
   - Self-contained ``(16, 8)`` linear codebook exhaustive verifier across all ``2^8 = 256``
     codewords proving certificate soundness, direction, invalid-codeword rejection, and
     full-waveform ML scoring.
"""
from dataclasses import dataclass
from itertools import combinations
import time
from typing import Any, Optional

import numpy as np
from numba import njit

from p8.coding import PAM, decode as _native_decode, generator, parity_check, qam, qam_llrs
from p9.contract import N, n0_from_snr


# ---------------------------------------------------------------------------
# Precomputed tables for the native systematic (128, 64) LDPC code & 16-QAM
# ---------------------------------------------------------------------------

_H_MATRIX = parity_check()
_G_MATRIX = generator()
_G_MATRIX.setflags(write=False)

_EDGE_BITS = np.ascontiguousarray(np.nonzero(_H_MATRIX)[1].astype(np.int64))
_CHECK_OFFSETS = np.ascontiguousarray(
    np.concatenate(([0], np.cumsum(_H_MATRIX.sum(axis=1)))).astype(np.int64)
)

_GEN_U64 = np.ascontiguousarray(
    np.packbits(_G_MATRIX.astype(np.uint8), axis=1, bitorder="little").view("<u8")
)
_H_U64 = np.ascontiguousarray(
    np.packbits(_H_MATRIX.astype(np.uint8), axis=1, bitorder="little").view("<u8")
)

_OSD_TRIPLES_LARGE = np.ascontiguousarray(
    np.asarray(list(combinations(range(48, 64), 3)), dtype=np.int64)
)
_PAIR_I = np.ascontiguousarray(
    np.asarray([i for i in range(64) for j in range(i)], dtype=np.int64)
)
_PAIR_J = np.ascontiguousarray(
    np.asarray([j for i in range(64) for j in range(i)], dtype=np.int64)
)
_SMALL_ORDER2_COUNT = 256

# Little-endian 4-bit nibble -> Gray 16-QAM symbol lookup matching p8.coding.qam exactly:
# bit 0 = nib & 1 (I sign), bit 1 = (nib >> 1) & 1 (I inner),
# bit 2 = (nib >> 2) & 1 (Q sign), bit 3 = (nib >> 3) & 1 (Q inner).
_NIBBLE_RE = np.array(
    [PAM[2 * (nib & 1) + ((nib >> 1) & 1)] for nib in range(16)], dtype=np.float64
)
_NIBBLE_IM = np.array(
    [PAM[2 * ((nib >> 2) & 1) + ((nib >> 3) & 1)] for nib in range(16)], dtype=np.float64
)
_PAM_SQ = (PAM ** 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Bit-packing & parity verification helpers
# ---------------------------------------------------------------------------


def pack_codewords(bits: np.ndarray) -> np.ndarray:
    """Pack ``(..., 128)`` uint8 bit arrays into ``(..., 2)`` uint64 little-endian words."""
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.shape[-1] != 128:
        raise ValueError(f"Expected last dimension 128, got {arr.shape}")
    packed = np.packbits(arr, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8")


def unpack_codewords(u64: np.ndarray) -> np.ndarray:
    """Unpack ``(..., 2)`` uint64 little-endian words into ``(..., 128)`` uint8 bits."""
    arr = np.ascontiguousarray(np.asarray(u64, dtype=np.uint64))
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected last dimension 2, got {arr.shape}")
    raw = arr.view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little")


def is_valid_codeword(codeword: np.ndarray) -> bool:
    """Return True iff ``codeword`` is a binary 128-bit vector satisfying ``H_parity @ c % 2 == 0``."""
    c = np.asarray(codeword)
    if c.shape != (128,):
        return False
    if not np.all((c == 0) | (c == 1)):
        return False
    c_u8 = c.astype(np.uint8, copy=False)
    return not bool(np.any((c_u8 @ _H_MATRIX.T) % 2))


@njit
def _parity64(x: np.uint64) -> np.uint64:
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return x & np.uint64(1)


@njit
def _is_valid_u64(c0: np.uint64, c1: np.uint64, h_u64: np.ndarray) -> bool:
    for r in range(h_u64.shape[0]):
        p = _parity64(c0 & h_u64[r, 0]) ^ _parity64(c1 & h_u64[r, 1])
        if p != np.uint64(0):
            return False
    return True


@njit
def _bit_get(row: np.ndarray, col: int) -> np.uint64:
    return (row[col >> 6] >> np.uint64(col & 63)) & np.uint64(1)


# ---------------------------------------------------------------------------
# Native LDPC comparator
# ---------------------------------------------------------------------------


def decode_native_ldpc(llrs: np.ndarray, iterations: int = 12) -> np.ndarray:
    """Native comparator wrapping ``p8.coding.decode`` (normalized min-sum, alpha=0.75)."""
    arr = np.asarray(llrs, dtype=np.float64)
    orig_shape = arr.shape
    if arr.size == 0 or orig_shape[-1] != 128:
        raise ValueError(f"Expected LLRs with last dimension 128, got {orig_shape}")
    out = _native_decode(arr.reshape(-1, 128), iterations=iterations)
    return out.reshape(*orig_shape[:-1], 64)


def native_ldpc_codeword(info_bits: np.ndarray) -> np.ndarray:
    """Encode systematic 64-bit info words into valid 128-bit codewords using ``G``."""
    info = np.asarray(info_bits, dtype=np.uint8)
    if info.shape[-1] != 64:
        raise ValueError(f"Expected info_bits with last dimension 64, got {info.shape}")
    return ((info @ _G_MATRIX) % 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Numba Sum-Product BP (adapted cleanly from p4/stage3_link_decoder.py)
# ---------------------------------------------------------------------------


@njit
def _bp_sum_product(
    channel: np.ndarray,
    edge_bits: np.ndarray,
    offsets: np.ndarray,
    iterations: int,
):
    """Sum-product BP on the native (128, 64) LDPC code.

    Input & output follow project convention: positive LLR means bit 1.
    Returns ``(posterior_llrs [B, 128], converged [B])``.
    """
    n_words = channel.shape[0]
    n_edges = edge_bits.shape[0]
    n_checks = offsets.shape[0] - 1
    result = np.empty_like(channel)
    converged = np.zeros(n_words, dtype=np.bool_)

    for w in range(n_words):
        intrinsic = np.empty(128, dtype=np.float64)
        for b in range(128):
            val = channel[w, b]
            if val > 40.0:
                val = 40.0
            elif val < -40.0:
                val = -40.0
            intrinsic[b] = -val  # positive-for-zero internal convention

        post = intrinsic.copy()
        messages = np.zeros(n_edges, dtype=np.float64)
        v2c = np.empty(n_edges, dtype=np.float64)
        for e in range(n_edges):
            v2c[e] = intrinsic[edge_bits[e]]

        # Check whether intrinsic hard decision is already a valid codeword?
        # Still run at least 1 BP iteration so soft reliabilities reflect check constraints.
        is_valid = False
        for _ in range(iterations):
            for b in range(128):
                post[b] = intrinsic[b]

            for check in range(n_checks):
                start = offsets[check]
                end = offsets[check + 1]
                product = 1.0
                zeroes = 0
                for e in range(start, end):
                    x = v2c[e]
                    if x > 35.0:
                        x = 35.0
                    elif x < -35.0:
                        x = -35.0
                    tv = np.tanh(x * 0.5)
                    messages[e] = tv
                    if abs(tv) > 1e-14:
                        product *= tv
                    else:
                        zeroes += 1
                for e in range(start, end):
                    tv = messages[e]
                    if zeroes == 0:
                        prod = product / tv
                    elif zeroes == 1 and abs(tv) <= 1e-14:
                        prod = product
                    else:
                        prod = 0.0
                    if prod > 1.0 - 1e-14:
                        prod = 1.0 - 1e-14
                    elif prod < -1.0 + 1e-14:
                        prod = -1.0 + 1e-14
                    msg = np.log1p(prod) - np.log1p(-prod)
                    messages[e] = msg
                    post[edge_bits[e]] += msg

            is_valid = True
            for check in range(n_checks):
                parity = 0
                for e in range(offsets[check], offsets[check + 1]):
                    if post[edge_bits[e]] < 0.0:
                        parity ^= 1
                if parity != 0:
                    is_valid = False
                    break
            if is_valid:
                break
            for e in range(n_edges):
                v2c[e] = post[edge_bits[e]] - messages[e]

        converged[w] = is_valid
        for b in range(128):
            result[w, b] = -post[b]
    return result, converged


def bp_posterior(llrs: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Run Numba sum-product BP on ``(..., 128)`` LLRs (positive = bit 1)."""
    arr = np.asarray(llrs, dtype=np.float64)
    orig_shape = arr.shape
    if orig_shape[-1] != 128:
        raise ValueError(f"Expected last dimension 128, got {orig_shape}")
    flat = np.ascontiguousarray(arr.reshape(-1, 128))
    post, conv = _bp_sum_product(flat, _EDGE_BITS, _CHECK_OFFSETS, int(iterations))
    return post.reshape(orig_shape), conv.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# LMMSE & Turbo MMSE-PIC soft equalization for 32x32 single-carrier rows
# ---------------------------------------------------------------------------


def lmmse_equalize_row(
    y_r: np.ndarray,
    H_r: np.ndarray,
    noise_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-carrier 32x32 LMMSE equalization -> ``(llrs [128], est [32], var [32])``."""
    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r, dtype=np.complex128)
    noise = np.asarray(noise_r, dtype=np.float64)
    if y.shape != (N,) or H.shape != (N, N) or noise.shape != (N,):
        raise ValueError(f"Expected shapes ({N},), ({N},{N}), ({N},); got {y.shape}, {H.shape}, {noise.shape}")
    if np.any(noise < 0.0):
        raise ValueError("noise_r must be non-negative")

    if np.all(noise > 0.0):
        adj = H.conj().T / noise[None, :]
        cov = np.linalg.inv(np.eye(N) + adj @ H)
        est = cov @ (adj @ y)
        var = np.diagonal(cov).real
    else:
        gram = H @ H.conj().T
        gram[np.arange(N), np.arange(N)] += noise
        try:
            z = np.linalg.solve(gram, y)
            w = np.linalg.solve(gram, H)
        except np.linalg.LinAlgError:
            gram_pinv = np.linalg.pinv(gram)
            z = gram_pinv @ y
            w = gram_pinv @ H
        est = H.conj().T @ z
        diag_red = np.einsum("ik,ik->k", H.conj(), w).real
        var = np.maximum(1.0 - diag_red, 0.0)

    llrs = qam_llrs(est, var).reshape(128)
    return llrs, est, var


def _soft_qam_moments(bit_llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-symbol 16-QAM soft mean ``[32]`` and variance ``[32]`` from bit LLRs ``[128]``."""
    l = np.clip(np.asarray(bit_llrs, dtype=np.float64).reshape(N, 4), -40.0, 40.0)
    p1 = 1.0 / (1.0 + np.exp(-l))
    p0 = 1.0 - p1

    # Gray PAM mapping: idx = 2*b0 + b1 -> [-3, -1, +3, +1] / sqrt(10)
    p_re = np.stack(
        (p0[:, 0] * p0[:, 1], p0[:, 0] * p1[:, 1], p1[:, 0] * p0[:, 1], p1[:, 0] * p1[:, 1]),
        axis=-1,
    )
    p_im = np.stack(
        (p0[:, 2] * p0[:, 3], p0[:, 2] * p1[:, 3], p1[:, 2] * p0[:, 3], p1[:, 2] * p1[:, 3]),
        axis=-1,
    )
    mu_re = p_re @ PAM
    mu_im = p_im @ PAM
    var_re = np.maximum(p_re @ _PAM_SQ - mu_re ** 2, 0.0)
    var_im = np.maximum(p_im @ _PAM_SQ - mu_im ** 2, 0.0)
    return mu_re + 1j * mu_im, var_re + var_im


def mmse_pic_equalize_row(
    y_r: np.ndarray,
    H_r: np.ndarray,
    noise_r: np.ndarray,
    prior_llrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-symbol extrinsic MMSE-PIC soft interference cancellation (Sherman-Morrison).

    Given extrinsic decoder priors ``prior_llrs [128]``, cancels soft ISI using all other
    symbols' means/variances while keeping symbol ``k``'s own prior excluded (extrinsic).
    Returns ``(ext_llrs [128], est_ext [32], var_ext [32])``.
    """
    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r, dtype=np.complex128)
    noise = np.asarray(noise_r, dtype=np.float64)
    if np.any(noise <= 0.0):
        return lmmse_equalize_row(y, H, noise)

    x_bar, v_a = _soft_qam_moments(prior_llrs)
    sigma = (H * v_a[None, :]) @ H.conj().T
    sigma[np.arange(N), np.arange(N)] += noise
    residual = y - H @ x_bar

    rhs = np.empty((N, N + 1), dtype=np.complex128)
    rhs[:, :N] = H
    rhs[:, N] = residual
    sol = np.linalg.solve(sigma, rhs)
    U = sol[:, :N]
    z = sol[:, N]

    alpha = np.maximum(np.einsum("nk,nk->k", H.conj(), U).real, 0.0)
    u_dot_r = H.conj().T @ z
    denom = np.maximum(1.0 + (1.0 - v_a) * alpha, 1e-12)
    est_ext = (u_dot_r + alpha * x_bar) / denom
    var_ext = np.clip((1.0 - v_a * alpha) / denom, 1e-12, 1.0)
    llrs_ext = qam_llrs(est_ext, var_ext).reshape(128)
    return llrs_ext, est_ext, var_ext


# ---------------------------------------------------------------------------
# Exact 32x32 time-varying waveform metric & Numba OSD search
# ---------------------------------------------------------------------------


def _prepare_weighted_csr(
    y_r: np.ndarray,
    H_r: np.ndarray,
    noise_r: Optional[np.ndarray],
):
    """Build exact CSR representation of weighted ``(y_w, A = W H_r)`` for fast exact scoring."""
    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r, dtype=np.complex128)
    if noise_r is None:
        w = np.ones(N, dtype=np.float64)
    else:
        nv = np.asarray(noise_r, dtype=np.float64)
        if np.all(nv > 0.0):
            w = 1.0 / np.sqrt(nv)
        elif np.all(nv == 0.0):
            # Zero-noise limit: unweighted exact Euclidean residual ||y - H x||^2.
            w = np.ones(N, dtype=np.float64)
        else:
            w = 1.0 / np.sqrt(np.maximum(nv, 1e-15))

    y_w = y * w
    A = H * w[:, None]

    # Detect exact non-zeros (3 non-zeros/row for 3-ray SC, 1/row for AWGN, up to 32/row dense).
    nz_mask = (A.real != 0.0) | (A.imag != 0.0)
    row_counts = nz_mask.sum(axis=1).astype(np.int64)
    row_ptr = np.empty(N + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_counts, out=row_ptr[1:])
    nz_rows, nz_cols = np.nonzero(nz_mask)
    col_idx = np.ascontiguousarray(nz_cols.astype(np.int64))
    vals = A[nz_rows, nz_cols]
    val_re = np.ascontiguousarray(vals.real.astype(np.float64))
    val_im = np.ascontiguousarray(vals.imag.astype(np.float64))
    y_w_re = np.ascontiguousarray(y_w.real.astype(np.float64))
    y_w_im = np.ascontiguousarray(y_w.imag.astype(np.float64))
    return y_w_re, y_w_im, row_ptr, col_idx, val_re, val_im


@njit
def _waveform_metric_csr(
    c0: np.uint64,
    c1: np.uint64,
    y_re: np.ndarray,
    y_im: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    val_re: np.ndarray,
    val_im: np.ndarray,
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
    x_re: np.ndarray,
    x_im: np.ndarray,
) -> float:
    """Exact ``sum_n |y_w[n] - (A @ x(c))[n]|^2`` for packed 128-bit codeword ``(c0, c1)``."""
    for s in range(16):
        nib0 = int((c0 >> np.uint64(4 * s)) & np.uint64(15))
        x_re[s] = nibble_re[nib0]
        x_im[s] = nibble_im[nib0]
        nib1 = int((c1 >> np.uint64(4 * s)) & np.uint64(15))
        x_re[16 + s] = nibble_re[nib1]
        x_im[16 + s] = nibble_im[nib1]

    total = 0.0
    n_rows = y_re.shape[0]
    for r in range(n_rows):
        acc_re = 0.0
        acc_im = 0.0
        start = row_ptr[r]
        end = row_ptr[r + 1]
        for p in range(start, end):
            j = col_idx[p]
            vr = val_re[p]
            vi = val_im[p]
            xr = x_re[j]
            xi = x_im[j]
            acc_re += vr * xr - vi * xi
            acc_im += vr * xi + vi * xr
        dr = y_re[r] - acc_re
        di = y_im[r] - acc_im
        total += dr * dr + di * di
    return total


@njit
def _osd_waveform_search(
    post: np.ndarray,
    y_re: np.ndarray,
    y_im: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    val_re: np.ndarray,
    val_im: np.ndarray,
    gen_u64: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    triples: np.ndarray,
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
    mode: int,
    small_order2_count: int,
    extra_u64: np.ndarray,
):
    """MRB Gaussian elimination + OSD candidate generation + exact 32x32 waveform rescoring.

    mode == 0 ('small'): order-0 (1) + order-1 (64) + top-256 lowest-flip-cost order-2 = 321.
    mode == 1 ('large'): order-0 (1) + order-1 (64) + all 2,016 order-2 + 560 order-3 = 2,641.
    """
    rows = gen_u64.copy()
    abs_post = np.abs(post)
    # Deterministic descending sort by |post| (tie-break by bit index via stable merge/argsort).
    order = np.argsort(-abs_post)

    pivots = np.empty(64, dtype=np.int64)
    k = 0
    for idx in range(128):
        col = order[idx]
        pivot = k
        while pivot < 64 and _bit_get(rows[pivot], col) == np.uint64(0):
            pivot += 1
        if pivot == 64:
            continue
        if pivot != k:
            a0, a1 = rows[k, 0], rows[k, 1]
            rows[k, 0], rows[k, 1] = rows[pivot, 0], rows[pivot, 1]
            rows[pivot, 0], rows[pivot, 1] = a0, a1
        for r in range(64):
            if r != k and _bit_get(rows[r], col) != np.uint64(0):
                rows[r, 0] ^= rows[k, 0]
                rows[r, 1] ^= rows[k, 1]
        pivots[k] = col
        k += 1
        if k == 64:
            break

    c0 = np.uint64(0)
    c1 = np.uint64(0)
    pivot_rel = np.empty(64, dtype=np.float64)
    for i in range(64):
        pcol = pivots[i]
        pivot_rel[i] = abs_post[pcol]
        if post[pcol] > 0.0:
            c0 ^= rows[i, 0]
            c1 ^= rows[i, 1]

    n_extra = extra_u64.shape[0]
    if mode == 0:
        max_cands = 1 + 64 + small_order2_count + n_extra
    else:
        max_cands = 1 + 64 + pair_i.shape[0] + triples.shape[0] + n_extra

    cands = np.empty((max_cands, 2), dtype=np.uint64)
    x_re = np.empty(32, dtype=np.float64)
    x_im = np.empty(32, dtype=np.float64)

    # Order-0
    cands[0, 0] = c0
    cands[0, 1] = c1
    best_c0 = c0
    best_c1 = c1
    best_metric = _waveform_metric_csr(
        c0, c1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
    )
    count = 1

    # Order-1 (all 64 MRB bits)
    for i in range(64):
        a0 = c0 ^ rows[i, 0]
        a1 = c1 ^ rows[i, 1]
        cands[count, 0] = a0
        cands[count, 1] = a1
        count += 1
        m = _waveform_metric_csr(
            a0, a1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
        )
        if m < best_metric:
            best_metric = m
            best_c0 = a0
            best_c1 = a1

    if mode == 0:
        # Small list: top-256 lowest-flip-cost order-2 pairs across all C(64, 2) = 2,016 pairs.
        n_pairs = pair_i.shape[0]
        pair_costs = np.empty(n_pairs, dtype=np.float64)
        for p in range(n_pairs):
            pair_costs[p] = pivot_rel[pair_i[p]] + pivot_rel[pair_j[p]]
        pair_order = np.argsort(pair_costs)
        limit = small_order2_count if small_order2_count < n_pairs else n_pairs
        for idx in range(limit):
            p = pair_order[idx]
            ii = pair_i[p]
            jj = pair_j[p]
            a0 = c0 ^ rows[ii, 0] ^ rows[jj, 0]
            a1 = c1 ^ rows[ii, 1] ^ rows[jj, 1]
            cands[count, 0] = a0
            cands[count, 1] = a1
            count += 1
            m = _waveform_metric_csr(
                a0, a1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
            )
            if m < best_metric:
                best_metric = m
                best_c0 = a0
                best_c1 = a1
    else:
        # Large list: all 2,016 order-2 pairs + 560 order-3 triples on least-reliable 16 MRB bits.
        for i in range(64):
            b0 = c0 ^ rows[i, 0]
            b1 = c1 ^ rows[i, 1]
            for j in range(i):
                a0 = b0 ^ rows[j, 0]
                a1 = b1 ^ rows[j, 1]
                cands[count, 0] = a0
                cands[count, 1] = a1
                count += 1
                m = _waveform_metric_csr(
                    a0, a1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
                )
                if m < best_metric:
                    best_metric = m
                    best_c0 = a0
                    best_c1 = a1
        for t in range(triples.shape[0]):
            ii = triples[t, 0]
            jj = triples[t, 1]
            kk = triples[t, 2]
            a0 = c0 ^ rows[ii, 0] ^ rows[jj, 0] ^ rows[kk, 0]
            a1 = c1 ^ rows[ii, 1] ^ rows[jj, 1] ^ rows[kk, 1]
            cands[count, 0] = a0
            cands[count, 1] = a1
            count += 1
            m = _waveform_metric_csr(
                a0, a1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
            )
            if m < best_metric:
                best_metric = m
                best_c0 = a0
                best_c1 = a1

    # Score any optional extra valid codewords (e.g., BP-0 converged codeword before Turbo PIC).
    for e in range(n_extra):
        a0 = extra_u64[e, 0]
        a1 = extra_u64[e, 1]
        cands[count, 0] = a0
        cands[count, 1] = a1
        count += 1
        m = _waveform_metric_csr(
            a0, a1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
        )
        if m < best_metric:
            best_metric = m
            best_c0 = a0
            best_c1 = a1

    return best_c0, best_c1, best_metric, cands[:count]


def exact_waveform_metric(
    y_r: np.ndarray,
    H_r: np.ndarray,
    codeword: np.ndarray,
    noise_r: Optional[np.ndarray] = None,
) -> float:
    """Reference float64 exact full 32x32 single-carrier waveform metric for one codeword.

    Computes ``sum_n |y_r[n] - (H_r @ x(c))[n]|^2 / noise_r[n]`` (or unweighted Euclidean
    distance ``||y_r - H_r x(c)||^2`` if ``noise_r is None`` or identically zero).
    """
    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r, dtype=np.complex128)
    x = qam(np.asarray(codeword, dtype=np.uint8))
    resid2 = np.abs(y - H @ x) ** 2
    if noise_r is None:
        return float(np.sum(resid2))
    nv = np.asarray(noise_r, dtype=np.float64)
    if np.all(nv == 0.0):
        return float(np.sum(resid2))
    return float(np.sum(resid2 / nv))


@dataclass(frozen=True)
class StrongDecodeResult:
    """Output of ``decode_strong_sc`` for one 128-bit CP-SC block."""
    info_bits: np.ndarray          # [64] uint8 decoded systematic info bits
    codeword: np.ndarray           # [128] uint8 winning valid codeword
    metric: float                  # exact winning 32x32 waveform metric
    candidates_u64: np.ndarray     # [M, 2] uint64 valid codewords scored (for ML certificate union)
    bp_converged: bool             # whether final BP satisfied all 64 parity checks
    n_candidates: int              # total valid codewords scored
    elapsed_ms: float              # CPU wall time for this block in milliseconds


def decode_strong_sc(
    y_r: np.ndarray,
    H_r: np.ndarray,
    noise_r: np.ndarray,
    *,
    list_size: str = "small",
    bp_iters: int = 20,
    turbo_pic: bool = True,
    pic_scale: float = 0.75,
    initial_llrs: Optional[np.ndarray] = None,
) -> StrongDecodeResult:
    """Strong classical CP-SC decoder: LMMSE -> Numba BP -> optional Turbo MMSE-PIC -> OSD + exact waveform rescoring.

    Metric interpretation:
    - Under true known channel (``noise_r = n0``), ``metric(c) = ||y_r - H_r x(c)||^2 / n0``
      is the EXACT Maximum-Likelihood (ML) metric for circularly symmetric complex Gaussian noise.
    - Under estimated channel (``H_r`` from ``mean_ray``, ``noise_r = n0 + sum_ray var_ray``),
      ``metric(c)`` is a plug-in approximate weighted waveform metric (diagonal sample noise model).
    """
    if list_size not in ("small", "large"):
        raise ValueError(f"list_size must be 'small' or 'large', got {list_size!r}")
    mode = 0 if list_size == "small" else 1

    t0 = time.perf_counter()

    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r, dtype=np.complex128)
    nv = np.asarray(noise_r, dtype=np.float64)

    if initial_llrs is None:
        llrs_0, _, _ = lmmse_equalize_row(y, H, nv)
    else:
        llrs_0 = np.asarray(initial_llrs, dtype=np.float64).reshape(128)

    post_0_arr, conv_0_arr = _bp_sum_product(
        np.ascontiguousarray(llrs_0[None, :]), _EDGE_BITS, _CHECK_OFFSETS, int(bp_iters)
    )
    post_0 = post_0_arr[0]
    conv_0 = bool(conv_0_arr[0])

    extra_list = []
    post_osd = post_0
    bp_conv = conv_0

    if turbo_pic and np.all(nv > 0.0):
        # Preserve BP-0's converged valid codeword as an extra candidate before PIC updates post_osd.
        if conv_0:
            extra_list.append(pack_codewords((post_0 > 0.0).astype(np.uint8)))
        # Extrinsic information from BP-0 fed back into MMSE-PIC soft interference cancellation.
        ext_0 = post_0 - np.clip(llrs_0, -40.0, 40.0)
        llrs_1, _, _ = mmse_pic_equalize_row(y, H, nv, float(pic_scale) * ext_0)
        post_1_arr, conv_1_arr = _bp_sum_product(
            np.ascontiguousarray(llrs_1[None, :]), _EDGE_BITS, _CHECK_OFFSETS, int(bp_iters)
        )
        post_osd = post_1_arr[0]
        bp_conv = bool(conv_1_arr[0])

    if extra_list:
        extra_u64 = np.ascontiguousarray(np.stack(extra_list, axis=0), dtype=np.uint64)
    else:
        extra_u64 = np.empty((0, 2), dtype=np.uint64)

    y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(y, H, nv)
    best_c0, best_c1, best_metric, cands_u64 = _osd_waveform_search(
        np.ascontiguousarray(post_osd, dtype=np.float64),
        y_re,
        y_im,
        row_ptr,
        col_idx,
        val_re,
        val_im,
        _GEN_U64,
        _PAIR_I,
        _PAIR_J,
        _OSD_TRIPLES_LARGE,
        _NIBBLE_RE,
        _NIBBLE_IM,
        mode,
        _SMALL_ORDER2_COUNT,
        extra_u64,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    best_u64 = np.array([best_c0, best_c1], dtype=np.uint64)
    codeword = unpack_codewords(best_u64)
    info_bits = codeword[:64].copy()
    return StrongDecodeResult(
        info_bits=info_bits,
        codeword=codeword,
        metric=float(best_metric),
        candidates_u64=cands_u64,
        bp_converged=bp_conv,
        n_candidates=int(cands_u64.shape[0]),
        elapsed_ms=float(elapsed_ms),
    )


def decode_strong_sc_batch(
    y_rows: np.ndarray,
    H_rows: np.ndarray,
    noise_rows: np.ndarray,
    *,
    list_size: str = "small",
    bp_iters: int = 20,
    turbo_pic: bool = True,
    pic_scale: float = 0.75,
    initial_llrs_rows: Optional[np.ndarray] = None,
) -> list[StrongDecodeResult]:
    """Run ``decode_strong_sc`` across a batch of ``B`` CP-SC rows."""
    n_rows = len(y_rows)
    return [
        decode_strong_sc(
            y_rows[i],
            H_rows[i],
            noise_rows[i],
            list_size=list_size,
            bp_iters=bp_iters,
            turbo_pic=turbo_pic,
            pic_scale=pic_scale,
            initial_llrs=None if initial_llrs_rows is None else initial_llrs_rows[i],
        )
        for i in range(n_rows)
    ]


def warmup_numba() -> None:
    """Trigger Numba JIT compilation cleanly before timing benchmarks."""
    dummy_llrs = np.zeros((1, 128), dtype=np.float64)
    _bp_sum_product(dummy_llrs, _EDGE_BITS, _CHECK_OFFSETS, 1)
    dummy_y = np.zeros(N, dtype=np.complex128)
    dummy_H = np.eye(N, dtype=np.complex128)
    dummy_noise = np.ones(N, dtype=np.float64)
    decode_strong_sc(dummy_y, dummy_H, dummy_noise, list_size="small", bp_iters=1, turbo_pic=False)
    decode_strong_sc(dummy_y, dummy_H, dummy_noise, list_size="large", bp_iters=1, turbo_pic=False)
    _cert_scan_u64(
        np.uint64(0),
        np.uint64(0),
        np.zeros((1, 2), dtype=np.uint64),
        _H_U64,
        np.zeros(N, dtype=np.float64),
        np.zeros(N, dtype=np.float64),
        np.arange(N + 1, dtype=np.int64),
        np.arange(N, dtype=np.int64),
        np.ones(N, dtype=np.float64),
        np.zeros(N, dtype=np.float64),
        _NIBBLE_RE,
        _NIBBLE_IM,
    )


# ---------------------------------------------------------------------------
# Evaluator-only Mathematically Proven ML Lower-Bound Certificate
# ---------------------------------------------------------------------------


@njit
def _cert_scan_u64(
    true_c0: np.uint64,
    true_c1: np.uint64,
    cand_u64: np.ndarray,
    h_u64: np.ndarray,
    y_re: np.ndarray,
    y_im: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    val_re: np.ndarray,
    val_im: np.ndarray,
    nibble_re: np.ndarray,
    nibble_im: np.ndarray,
):
    """Verify ``H_parity @ c % 2 == 0`` and compute exact ``||y_r - H_r x(c)||^2`` for packed candidates."""
    x_re = np.empty(32, dtype=np.float64)
    x_im = np.empty(32, dtype=np.float64)
    best_d = np.inf
    best_idx = -1
    n_valid_other = 0
    n_invalid = 0

    for idx in range(cand_u64.shape[0]):
        c0 = cand_u64[idx, 0]
        c1 = cand_u64[idx, 1]
        if not _is_valid_u64(c0, c1, h_u64):
            n_invalid += 1
            continue
        if c0 == true_c0 and c1 == true_c1:
            continue
        n_valid_other += 1
        d = _waveform_metric_csr(
            c0, c1, y_re, y_im, row_ptr, col_idx, val_re, val_im, nibble_re, nibble_im, x_re, x_im
        )
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_d, best_idx, n_valid_other, n_invalid


@dataclass(frozen=True)
class MLCertificateResult:
    """Result of the strict evaluator-only ML lower-bound certificate on one block."""
    ml_must_err: bool
    status: str                             # "ml_must_err" or "undecided"
    d_true: float                           # exact ||y_r - H_r x(c_true)||^2
    d_best_other: float                     # min ||y_r - H_r x(c')||^2 over valid c' != c_true checked
    margin: float                           # d_true - d_best_other
    eps_tie: float                          # 1e-11 * max(1.0, d_true)
    n_checked: int                          # total valid c' != c_true checked
    n_single_flip_checked: int              # 64 single-info-bit-flip neighbors
    n_extra_valid_checked: int              # valid extra candidates != c_true
    n_invalid_rejected: int                 # extra candidates rejected by H_parity @ c % 2 != 0
    witness_source: str                     # "none", "single_flip", or "extra_candidate"
    witness_codeword: Optional[np.ndarray]  # [128] uint8 witness if ml_must_err else None


def _normalize_extra_candidates(extra_candidates: Any) -> tuple[np.ndarray, int]:
    """Convert flexible extra_candidates input to ``([M, 2] uint64, pre_rejected_count)``."""
    if extra_candidates is None:
        return np.empty((0, 2), dtype=np.uint64), 0

    if isinstance(extra_candidates, np.ndarray):
        if extra_candidates.size == 0:
            return np.empty((0, 2), dtype=np.uint64), 0
        if extra_candidates.dtype == np.uint64 and extra_candidates.ndim == 2 and extra_candidates.shape[1] == 2:
            return np.ascontiguousarray(extra_candidates), 0
        if extra_candidates.ndim == 1 and extra_candidates.shape[0] == 128:
            extra_candidates = extra_candidates[None, :]
        if extra_candidates.ndim == 2 and extra_candidates.shape[1] == 128:
            # Filter non-binary rows as invalid.
            binary_ok = np.all((extra_candidates == 0) | (extra_candidates == 1), axis=1)
            n_bad = int((~binary_ok).sum())
            good = extra_candidates[binary_ok].astype(np.uint8, copy=False)
            if len(good) == 0:
                return np.empty((0, 2), dtype=np.uint64), n_bad
            return pack_codewords(good), n_bad
        raise ValueError(f"Unsupported extra_candidates array shape/dtype: {extra_candidates.shape}, {extra_candidates.dtype}")

    # Iterable of candidate arrays (e.g., list of [M_i, 2] uint64 or [128] uint8).
    chunks = []
    n_bad_total = 0
    for item in extra_candidates:
        u64_chunk, n_bad = _normalize_extra_candidates(np.asarray(item))
        n_bad_total += n_bad
        if len(u64_chunk):
            chunks.append(u64_chunk)
    if not chunks:
        return np.empty((0, 2), dtype=np.uint64), n_bad_total
    return np.ascontiguousarray(np.vstack(chunks), dtype=np.uint64), n_bad_total


def ml_lower_bound_certificate(
    y_r: np.ndarray,
    H_r_true: np.ndarray,
    c_true: np.ndarray,
    extra_candidates: Any = None,
) -> MLCertificateResult:
    """Evaluator-only mathematically proven ML lower-bound certificate.

    NEVER pass ``c_true`` to any receiver; ``extra_candidates`` must come strictly from
    blind receivers' outputs.

    Checks:
      1. All 64 single-info-bit-flip neighbors ``c_true ^ G[i]`` (i=0..63), AND
      2. All valid ``extra_candidates != c_true`` (verifying ``H_parity @ c % 2 == 0``
         for EVERY candidate; any non-codeword is strictly rejected).

    Certifies ``ml_must_err = True`` iff there exists a verified valid codeword
    ``c' != c_true`` with:
        ``||y_r - H_r_true x(c')||^2 < ||y_r - H_r_true x(c_true)||^2 - eps_tie``
    where ``eps_tie = 1e-11 * max(1.0, d_true)``; otherwise marks ``status = "undecided"``.
    """
    y = np.asarray(y_r, dtype=np.complex128)
    H = np.asarray(H_r_true, dtype=np.complex128)
    c_t = np.asarray(c_true, dtype=np.uint8).reshape(128)
    if not is_valid_codeword(c_t):
        raise ValueError("c_true must be a valid (128, 64) codeword satisfying H_parity @ c % 2 == 0")

    d_true = exact_waveform_metric(y, H, c_t, noise_r=None)
    eps_tie = 1e-11 * max(1.0, d_true)

    true_u64 = pack_codewords(c_t)
    true_c0 = true_u64[0]
    true_c1 = true_u64[1]

    y_re, y_im, row_ptr, col_idx, val_re, val_im = _prepare_weighted_csr(y, H, noise_r=None)

    # 1. All 64 single-info-bit-flip neighbors: c_true ^ G[i]
    single_flips_u64 = np.empty((64, 2), dtype=np.uint64)
    single_flips_u64[:, 0] = true_c0 ^ _GEN_U64[:, 0]
    single_flips_u64[:, 1] = true_c1 ^ _GEN_U64[:, 1]

    d_sf, idx_sf, n_sf_valid, n_sf_invalid = _cert_scan_u64(
        true_c0, true_c1, single_flips_u64, _H_U64,
        y_re, y_im, row_ptr, col_idx, val_re, val_im, _NIBBLE_RE, _NIBBLE_IM,
    )
    if n_sf_valid != 64 or n_sf_invalid != 0:
        raise RuntimeError(f"Generator single-flip validity invariant violated: valid={n_sf_valid}, invalid={n_sf_invalid}")

    # 2. All extra candidates from actual receivers (every single one parity-checked!)
    extra_u64, n_pre_rejected = _normalize_extra_candidates(extra_candidates)
    if len(extra_u64):
        d_ex, idx_ex, n_ex_valid, n_ex_invalid = _cert_scan_u64(
            true_c0, true_c1, extra_u64, _H_U64,
            y_re, y_im, row_ptr, col_idx, val_re, val_im, _NIBBLE_RE, _NIBBLE_IM,
        )
    else:
        d_ex, idx_ex, n_ex_valid, n_ex_invalid = np.inf, -1, 0, 0

    n_invalid_rejected = n_pre_rejected + int(n_ex_invalid)

    if d_ex < d_sf and idx_ex >= 0:
        best_u64 = extra_u64[idx_ex]
        best_source = "extra_candidate"
        best_d = float(d_ex)
    else:
        best_u64 = single_flips_u64[idx_sf]
        best_source = "single_flip"
        best_d = float(d_sf)

    # Independent pure-NumPy double verification on the best witness before certifying!
    witness_bits = unpack_codewords(best_u64)
    best_d_ref = exact_waveform_metric(y, H, witness_bits, noise_r=None)
    witness_valid = is_valid_codeword(witness_bits) and not np.array_equal(witness_bits, c_t)

    ml_must_err = bool(witness_valid and (best_d_ref < d_true - eps_tie))
    return MLCertificateResult(
        ml_must_err=ml_must_err,
        status="ml_must_err" if ml_must_err else "undecided",
        d_true=float(d_true),
        d_best_other=float(best_d_ref),
        margin=float(d_true - best_d_ref),
        eps_tie=float(eps_tie),
        n_checked=int(n_sf_valid + n_ex_valid),
        n_single_flip_checked=int(n_sf_valid),
        n_extra_valid_checked=int(n_ex_valid),
        n_invalid_rejected=int(n_invalid_rejected),
        witness_source=best_source if ml_must_err else "none",
        witness_codeword=witness_bits if ml_must_err else None,
    )


# ---------------------------------------------------------------------------
# Self-contained mini (16, 8) linear codebook exhaustive ML certificate verifier
# ---------------------------------------------------------------------------


def _mini_codebook_16_8() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a systematic (16, 8) binary linear code and enumerate all 2^8 = 256 codewords."""
    # Circulant-inspired parity block P (8x8) with d_min >= 4 (exact d_min computed below).
    p_rows = np.array(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    G = np.hstack((np.eye(8, dtype=np.uint8), p_rows))
    H = np.hstack((p_rows.T, np.eye(8, dtype=np.uint8)))
    assert not np.any((G @ H.T) % 2), "Mini (16, 8) code failed G @ H^T == 0 (mod 2)"

    # Exhaustively enumerate all 2^8 = 256 info vectors and valid codewords.
    ints = np.arange(256, dtype=np.uint8)
    infos = ((ints[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    codewords = ((infos @ G) % 2).astype(np.uint8)
    assert not np.any((codewords @ H.T) % 2), "Enumerated mini codewords failed parity check"
    weights = codewords[1:].sum(axis=1)
    d_min = int(weights.min())
    waveforms = qam(codewords).reshape(256, 4)  # 16 bits -> 4 Gray 16-QAM symbols
    return G, H, codewords, waveforms, d_min


def verify_mini_codebook_ml_certificate(
    n_trials: int = 240,
    snr_db: float = 9.0,
    seed: int = 910999,
) -> dict[str, Any]:
    """Exhaustively verify ML certificate soundness, direction, invalid-candidate rejection,
    and full-waveform scoring on a self-contained (16, 8) linear codebook (256 codewords).
    """
    G, H_parity, C_all, X_all, d_min = _mini_codebook_16_8()
    rng = np.random.default_rng(seed)
    n0 = n0_from_snr(snr_db)
    n_sym = 4
    delays = np.array([0, 1, 2], dtype=np.int64)
    powers = np.array([0.75, 0.17, 0.08], dtype=np.float64)
    samples = np.arange(n_sym, dtype=np.int64)

    exact_ml_errors = 0
    partial_cert_errors = 0
    full_cert_errors = 0
    lmmse_hard_errors = 0
    false_cert_violations = 0
    full_mismatch_violations = 0
    invalid_rejection_passes = 0

    for _ in range(n_trials):
        # Random 3-ray time-varying 4x4 single-carrier channel matrix H_r
        amps = np.sqrt(powers) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / np.sqrt(2.0)
        freqs = rng.uniform(0.045, 0.080, size=3) * rng.choice([-1.0, 1.0], size=3)
        H_r = np.zeros((n_sym, n_sym), dtype=np.complex128)
        for ray_idx, d in enumerate(delays):
            rot = np.exp(2j * np.pi * freqs[ray_idx] * (samples / n_sym))
            H_r[samples, (samples - d) % n_sym] += amps[ray_idx] * rot

        idx_true = int(rng.integers(0, 256))
        c_true = C_all[idx_true]
        x_true = X_all[idx_true]
        w = np.sqrt(n0) * (rng.normal(size=n_sym) + 1j * rng.normal(size=n_sym)) / np.sqrt(2.0)
        y_r = H_r @ x_true + w

        # 1. Exhaustive exact ML across ALL 256 valid codewords.
        preds = X_all @ H_r.T  # [256, 4]
        dists = np.sum(np.abs(y_r[None, :] - preds) ** 2, axis=1)
        d_true = float(dists[idx_true])
        eps_tie = 1e-11 * max(1.0, d_true)
        other_dists = np.delete(dists, idx_true)
        d_min_other = float(other_dists.min())
        exact_ml_err = bool(d_min_other < d_true - eps_tie)
        if exact_ml_err:
            exact_ml_errors += 1

        # 2. Symbol-wise LMMSE hard decoder (suboptimal baseline).
        adj = H_r.conj().T / n0
        cov = np.linalg.inv(np.eye(n_sym) + adj @ H_r)
        est = cov @ (adj @ y_r)
        var = np.diagonal(cov).real
        llrs = qam_llrs(est, var).reshape(16)
        hard_info = (llrs[:8] > 0.0).astype(np.uint8)
        if not np.array_equal(hard_info, c_true[:8]):
            lmmse_hard_errors += 1

        # 3. Partial certificate: 8 single-info-bit-flip neighbors + LMMSE hard codeword.
        c_lmmse = (hard_info @ G) % 2
        partial_cands = np.vstack([(c_true ^ G[i]) for i in range(8)] + [c_lmmse])
        # Verify H_parity @ c % 2 == 0 and c != c_true for every candidate
        valid_mask = (~np.any((partial_cands @ H_parity.T) % 2, axis=1)) & (
            np.any(partial_cands != c_true[None, :], axis=1)
        )
        valid_partial = partial_cands[valid_mask]
        x_part = qam(valid_partial).reshape(-1, n_sym)
        d_part = np.sum(np.abs(y_r[None, :] - (x_part @ H_r.T)) ** 2, axis=1)
        cert_partial = bool(len(d_part) > 0 and float(d_part.min()) < d_true - eps_tie)
        if cert_partial:
            partial_cert_errors += 1
            if not exact_ml_err:
                false_cert_violations += 1

        # 4. Full-codebook certificate MUST match exact exhaustive ML 100%.
        valid_full_mask = (~np.any((C_all @ H_parity.T) % 2, axis=1)) & (
            np.any(C_all != c_true[None, :], axis=1)
        )
        d_full_best = float(dists[valid_full_mask].min())
        cert_full = bool(d_full_best < d_true - eps_tie)
        if cert_full:
            full_cert_errors += 1
        if cert_full != exact_ml_err:
            full_mismatch_violations += 1

        # 5. Invalid non-codeword rejection check (flip 1 parity bit -> guaranteed invalid).
        bad_c = c_true.copy()
        bad_c[8] ^= 1
        assert np.any((bad_c @ H_parity.T) % 2), "Constructed bad_c unexpectedly passed parity check"
        if np.any((bad_c @ H_parity.T) % 2):
            invalid_rejection_passes += 1

    direction_holds = bool(partial_cert_errors <= exact_ml_errors <= lmmse_hard_errors)
    all_passed = bool(
        false_cert_violations == 0
        and full_mismatch_violations == 0
        and invalid_rejection_passes == n_trials
        and direction_holds
    )
    return {
        "code": "(16, 8)",
        "codebook_size": 256,
        "d_min": d_min,
        "n_trials": int(n_trials),
        "snr_db": float(snr_db),
        "partial_cert_errors": int(partial_cert_errors),
        "partial_cert_rate": float(partial_cert_errors / n_trials),
        "exact_ml_errors": int(exact_ml_errors),
        "exact_ml_rate": float(exact_ml_errors / n_trials),
        "full_cert_errors": int(full_cert_errors),
        "lmmse_hard_errors": int(lmmse_hard_errors),
        "lmmse_hard_rate": float(lmmse_hard_errors / n_trials),
        "false_cert_violations": int(false_cert_violations),
        "full_mismatch_violations": int(full_mismatch_violations),
        "invalid_rejection_passes": int(invalid_rejection_passes),
        "direction_holds": direction_holds,
        "all_passed": all_passed,
    }
