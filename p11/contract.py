"""Locked shared interface contract for Phase 11 (EM-FloatFormer P11).

Chains 9e9d93b's second-order payload covariance reconstruction with parity-certified
wavefront assimilation as Pillar 1, and evaluates off-grid continuous-delay cross-waveform
manifold generalization as Pillar 2 under strictly disjoint seed namespaces, float64/complex128
precision, deterministic runtime guards, and zero hardcoded N0.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional

import numpy as np

from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    CandidateSummary,
    DELAYS,
    FRAME_DURATION,
    GateOutput,
    HiddenTruth,
    LINK_DURATION,
    N,
    OFDM_PILOT_ROWS,
    P8C_BASELINE_COMMIT,
    P8_MANIFOLD_COMMIT,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    PublicObservation,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SC_LAYOUTS,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    SNR_GRID_DB,
    T,
    jeffreys_complex_gaussian,
    mix_bit_llrs,
    n0_from_snr,
)

# --- Anchor & Provenance ---
ANCHOR: int = int(SC_LAYOUTS["sparse"][0])
P9_BASELINE_COMMIT = "faeea639727aab72740cdd0a719106f83824226e"
P10_BASELINE_COMMIT = "9e9d93bcbf6345b8645314b223cd954b2f31e63a"

# --- Canonical Precision ---
REAL_DTYPE = np.float64
COMPLEX_DTYPE = np.complex128

# --- Strictly Disjoint Phase 11 Seed Namespaces ---
# Disjoint from P1..P4, P5..P6, P7, p8_manifold, P8-C, P9, 9e9d93b:p10 (960..982e9),
# EM-FloatFormer-p10 (1010..1050e9), and p4:9102379953802.
SEED_P11_DEV = 1_110_000_000_000
SEED_P11_TRAIN = 1_120_000_000_000
SEED_P11_FORMAL_ID = 1_130_000_000_000
SEED_P11_FORMAL_OOD = 1_140_000_000_000
SEED_P11_E2E = 1_150_000_000_000
P11_NAMESPACE_SPAN = 10_000_000_000

P11_SEED_RANGES: dict[str, tuple[int, int]] = {
    "p11_dev": (SEED_P11_DEV, SEED_P11_DEV + P11_NAMESPACE_SPAN),
    "p11_train": (SEED_P11_TRAIN, SEED_P11_TRAIN + P11_NAMESPACE_SPAN),
    "p11_formal_id": (SEED_P11_FORMAL_ID, SEED_P11_FORMAL_ID + P11_NAMESPACE_SPAN),
    "p11_formal_ood": (SEED_P11_FORMAL_OOD, SEED_P11_FORMAL_OOD + P11_NAMESPACE_SPAN),
    "p11_e2e": (SEED_P11_E2E, SEED_P11_E2E + P11_NAMESPACE_SPAN),
}

# Half-open [lo, hi) intervals covering all historical seeds across P1..P10
HISTORICAL_SEED_RANGES: dict[str, tuple[int, int]] = {
    "p1_to_p4_legacy": (0, 10_000_000),
    "p1_to_p4_extended": (10_000_000, 88_000_000),
    "p8_manifold_dev_ladder": (88_000_000, 100_000_000),
    "p5_p6": (100_000_000, 400_000_000),
    "p3_and_exploratory": (400_000_000, 2_147_483_648),
    "p3b_data_seed": (5_220_000_000, 5_221_000_000),
    "p6_followup": (602_611_000_000, 602_621_000_000),
    "p7": (702_611_000_000, 702_621_000_000),
    "p6_softinfo": (713_611_000_000, 713_621_000_000),
    "p8_manifold": (802_611_000_000, 802_621_000_000),
    "p8_prefix": (812_611_000_000, 812_621_000_000),
    "p6_consistency": (813_611_000_000, 814_621_000_000),
    "p8c": (822_611_000_000, 822_621_000_000),
    "p9_dev": (SEED_P9_DEV, SEED_P9_DEV + 10_000_000_000),
    "p9_train": (SEED_P9_TRAIN, SEED_P9_TRAIN + 10_000_000_000),
    "p9_formal_id": (SEED_P9_FORMAL_ID, SEED_P9_FORMAL_ID + 10_000_000_000),
    "p9_formal_ood": (SEED_P9_FORMAL_OOD, SEED_P9_FORMAL_OOD + 10_000_000_000),
    "p10_9e9d93b": (960_000_000_000, 983_000_000_000),
    "p10_external_dev": (1_010_000_000_000, 1_020_000_000_000),
    "p10_external_train": (1_020_000_000_000, 1_030_000_000_000),
    "p10_external_formal_id": (1_030_000_000_000, 1_040_000_000_000),
    "p10_external_formal_ood": (1_040_000_000_000, 1_050_000_000_000),
    "p10_external_reserved_1050": (1_050_000_000_000, 1_060_000_000_000),
    "p4_stage3_link_genie_literal": (9_102_379_953_802, 9_102_379_953_803),
}


def is_historical_seed(seed: int) -> Optional[str]:
    """Return matching historical seed namespace key if `seed` falls inside a historical range, else None.

    Rejects booleans, timedelta64, and datetime64 with TypeError to avoid Python/NumPy integer-subclass pitfalls.
    """
    if isinstance(seed, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"seed must be an integer (not bool, timedelta64, or datetime64), got {type(seed).__name__}"
        )
    s = int(seed)
    for name, (lo, hi) in HISTORICAL_SEED_RANGES.items():
        if lo <= s < hi:
            return name
    return None


def enforce_determinism(require_env: bool = True) -> dict[str, Any]:
    """Enforce single-threaded BLAS/OMP/MKL environment and deterministic PyTorch execution.

    PyTorch is imported lazily inside this function so pure-NumPy/SciPy audit modules
    importing `p11.contract` do not import `torch` at module load time.
    """
    if not isinstance(require_env, (bool, np.bool_)):
        raise TypeError(f"require_env must be a bool, got {type(require_env).__name__}")
    require_env = bool(require_env)

    if require_env:
        bad: list[str] = []
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            val = os.environ.get(var)
            if val != "1":
                bad.append(f"{var}={val!r} (expected '1')")
        mkl_val = os.environ.get("MKL_NUM_THREADS")
        if mkl_val is not None and mkl_val != "1":
            bad.append(f"MKL_NUM_THREADS={mkl_val!r} (expected '1')")
        if bad:
            raise RuntimeError(
                "Deterministic execution requires single-threaded BLAS/OMP env vars: "
                + ", ".join(bad)
            )
        if mkl_val is None:
            os.environ["MKL_NUM_THREADS"] = "1"
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    n_threads = int(torch.get_num_threads())
    det_enabled = bool(torch.are_deterministic_algorithms_enabled())
    if n_threads != 1 or not det_enabled:
        raise RuntimeError(
            f"Failed to enforce PyTorch determinism: num_threads={n_threads}, "
            f"deterministic_algorithms={det_enabled}"
        )

    return {
        "require_env": bool(require_env),
        "torch_num_threads": n_threads,
        "deterministic_algorithms": det_enabled,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_version": str(torch.__version__),
    }


# --- Pillar 2 (Off-Grid Continuous-Delay Cross-Waveform Manifold) Regimes & Geometry ---
OFFGRID_DELAY_MIN = 0.0
OFFGRID_DELAY_MAX = 6.5
OFFGRID_RAY_COUNTS = (2, 3, 4, 5)
OFFGRID_ANCHOR_ROWS = (3, 6, 10, 13, 16)
OFFGRID_REGIMES_ID = (
    "offgrid_continuous",
    "offgrid_partial_shift",
    "offgrid_midframe_shift",
    "offgrid_full_reset",
)
OFFGRID_REGIMES_OOD = (
    "offgrid_ood_burst_and_shift",
    "offgrid_ood_five_ray_fast",
)
OFFGRID_ALL_REGIMES = OFFGRID_REGIMES_ID + OFFGRID_REGIMES_OOD


# --- Exact Unitary Centered-DFT Fractional Cyclic-Delay Operator ---
_SIGNED_K_CACHE: dict[int, np.ndarray] = {}
_DFT_UNITARY_CACHE: dict[int, np.ndarray] = {}


def _safe_repr(val: Any, max_len: int = 200) -> str:
    """Return a bounded repr(val) without propagating exceptions from pathological __repr__ implementations."""
    try:
        text = repr(val)
    except Exception as exc:
        return f"<{type(val).__name__} (repr failed: {type(exc).__name__})>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _validate_n(n: int) -> int:
    if isinstance(n, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be a positive integer, got {_safe_repr(n)}")
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n_int}")
    return n_int


def signed_subcarrier_indices(n: int = N) -> np.ndarray:
    """Centered DFT subcarrier indices [0, 1, ..., n//2 - 1, -n//2, ..., -1] as float64."""
    n_int = _validate_n(n)
    if n_int not in _SIGNED_K_CACHE:
        k = np.fft.fftfreq(n_int, d=1.0 / n_int).astype(REAL_DTYPE)
        k.setflags(write=False)
        _SIGNED_K_CACHE[n_int] = k
    return _SIGNED_K_CACHE[n_int]


def unitary_dft_matrix(n: int = N) -> np.ndarray:
    """Unitary N x N DFT matrix F with F[k, m] = exp(-2j*pi*k*m/N) / sqrt(N)."""
    n_int = _validate_n(n)
    if n_int not in _DFT_UNITARY_CACHE:
        idx = np.arange(n_int, dtype=REAL_DTYPE)
        f_mat = np.exp(-2j * np.pi * np.outer(idx, idx) / float(n_int)) / np.sqrt(float(n_int))
        f_mat = np.asarray(f_mat, dtype=COMPLEX_DTYPE)
        f_mat.setflags(write=False)
        _DFT_UNITARY_CACHE[n_int] = f_mat
    return _DFT_UNITARY_CACHE[n_int]


_REAL_DTYPES_KINDS = frozenset({"i", "u", "f"})
_COMPLEX_DTYPES_KINDS = frozenset({"i", "u", "f", "c"})


def _validate_numeric_obj(
    obj: Any,
    *,
    allow_complex: bool,
    name: str,
    _seen: Optional[set[int]] = None,
) -> None:
    allowed_kinds = _COMPLEX_DTYPES_KINDS if allow_complex else _REAL_DTYPES_KINDS
    kind_desc = "finite real or complex" if allow_complex else "finite real"
    if isinstance(obj, np.ma.MaskedArray):
        raise ValueError(f"{name} must contain only {kind_desc} values, not MaskedArray")
    if isinstance(obj, (bool, np.bool_, str, bytes, bytearray, memoryview, np.timedelta64, np.datetime64)):
        raise ValueError(f"{name} must contain only {kind_desc} values, got {type(obj).__name__}")
    if not allow_complex and isinstance(obj, (complex, np.complexfloating)):
        raise ValueError(f"{name} must contain only {kind_desc} values, not complex")
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return
    if allow_complex and isinstance(obj, (complex, np.complexfloating)):
        return
    if isinstance(obj, np.ndarray):
        kind = obj.dtype.kind
        if kind in allowed_kinds:
            return
        if kind == "O":
            if obj.size == 0:
                raise ValueError(f"{name} must contain only {kind_desc} values, got empty object array")
            if _seen is None:
                _seen = set()
            obj_id = id(obj)
            if obj_id in _seen:
                raise ValueError(f"{name} must not contain cyclic object array references")
            _seen.add(obj_id)
            try:
                for elem in obj.flat:
                    _validate_numeric_obj(elem, allow_complex=allow_complex, name=name, _seen=_seen)
            finally:
                _seen.remove(obj_id)
            return
        raise ValueError(f"{name} must contain only {kind_desc} values, got ndarray with dtype {obj.dtype}")
    if isinstance(obj, (list, tuple)):
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            raise ValueError(f"{name} must not contain cyclic sequence references")
        _seen.add(obj_id)
        try:
            for item in obj:
                _validate_numeric_obj(item, allow_complex=allow_complex, name=name, _seen=_seen)
        finally:
            _seen.remove(obj_id)
        return
    raise ValueError(f"{name} must contain only {kind_desc} values, got {type(obj).__name__}")


def _coerce_numeric_array(val: Any, *, allow_complex: bool, name: str) -> np.ndarray:
    """Validate and coerce scalar or array-like numeric values to finite float64 or complex128."""
    allowed_kinds = _COMPLEX_DTYPES_KINDS if allow_complex else _REAL_DTYPES_KINDS
    kind_desc = "finite real or complex" if allow_complex else "finite real"
    target_dtype = COMPLEX_DTYPE if allow_complex else REAL_DTYPE
    try:
        _validate_numeric_obj(val, allow_complex=allow_complex, name=name)
        raw = np.asanyarray(val)
        if isinstance(raw, np.ma.MaskedArray):
            raise ValueError(f"{name} must contain only {kind_desc} values, not MaskedArray")
        if raw.dtype.kind not in allowed_kinds and raw.dtype.kind != "O":
            raise ValueError(f"{name} must contain only {kind_desc} values, got dtype {raw.dtype}")
        if raw.dtype.kind == "O":
            if raw.size == 0:
                raise ValueError(f"{name} must contain only {kind_desc} values, got empty object array")
            for elem in raw.flat:
                _validate_numeric_obj(elem, allow_complex=allow_complex, name=name)
        out = np.asarray(raw, dtype=target_dtype)
    except (TypeError, ValueError, OverflowError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain only {kind_desc} values, got {_safe_repr(val)}") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only {kind_desc} values")
    return out


def _coerce_real_delay_array(delay: Any) -> np.ndarray:
    """Validate and coerce scalar or array-like real delays to finite float64."""
    return _coerce_numeric_array(delay, allow_complex=False, name="delay")


def fractional_delay_freq_response(delay: float | np.ndarray, n: int = N) -> np.ndarray:
    """Exact unitary OFDM subcarrier response exp(-2j*pi*k_signed*delay/N).

    Supports scalar delay -> [N] complex128, or 1D/ND array [..., P] -> [..., P, N] complex128.
    """
    n_int = _validate_n(n)
    k_s = signed_subcarrier_indices(n_int)
    d = _coerce_real_delay_array(delay)
    if d.ndim == 0:
        return np.exp(-2j * np.pi * k_s * float(d) / float(n_int)).astype(COMPLEX_DTYPE)
    return np.exp(-2j * np.pi * d[..., None] * k_s[None, :] / float(n_int)).astype(COMPLEX_DTYPE)


def fractional_delay_matrix(delay: float, n: int = N) -> np.ndarray:
    """Exact unitary N x N cyclic fractional-delay operator D_N(delay) = F^H diag(e^{-2j*pi*k_s*delay/N}) F."""
    n_int = _validate_n(n)
    d = _coerce_real_delay_array(delay)
    if d.ndim != 0:
        raise ValueError(f"delay must be a scalar, got shape {d.shape}")
    d_val = float(d)
    f_mat = unitary_dft_matrix(n_int)
    phase = fractional_delay_freq_response(d_val, n_int)
    return np.asarray((f_mat.conj().T * phase[None, :]) @ f_mat, dtype=COMPLEX_DTYPE)


def fractional_delay_kernel(delay: float, n: int = N) -> np.ndarray:
    """Canonical alias for `fractional_delay_matrix(delay, n=n)`."""
    return fractional_delay_matrix(delay, n=n)


def build_offgrid_sc_channel_matrix(
    ray_gains_row: np.ndarray,
    delays: np.ndarray,
    n: int = N,
) -> np.ndarray:
    """Exact N x N time-varying CP-SC channel matrix for continuous off-grid delays."""
    n_int = _validate_n(n)
    g = _coerce_numeric_array(ray_gains_row, allow_complex=True, name="ray_gains_row")
    d = _coerce_real_delay_array(delays)
    if g.ndim != 2 or g.shape[1] != n_int or d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(f"Expected ray_gains_row [P, {n_int}] and delays [P], got {g.shape} and {d.shape}")
    h_mat = np.zeros((n_int, n_int), dtype=COMPLEX_DTYPE)
    for l_idx in range(d.shape[0]):
        d_mat = fractional_delay_matrix(float(d[l_idx]), n=n_int)
        h_mat += g[l_idx, :, None] * d_mat
    return h_mat


def build_fractional_sc_matrix(
    ray_gains_row: np.ndarray,
    delays: np.ndarray,
    n: int = N,
) -> np.ndarray:
    """Canonical alias for `build_offgrid_sc_channel_matrix(ray_gains_row, delays, n=n)`."""
    return build_offgrid_sc_channel_matrix(ray_gains_row, delays, n=n)


def apply_fractional_sc_operator(
    x: np.ndarray,
    ray_gains_row: np.ndarray,
    delays: np.ndarray,
    n: int = N,
) -> np.ndarray:
    """Apply the exact time-varying fractional-delay SC channel operator ray-by-ray.

    Accumulates `sum_l g_l * (x @ D_N(tau_l).T)` directly across rays without forming
    the combined `[N, N]` channel matrix `H_r`, matching `H_r @ x` for 1D `[N]` and
    `x @ H_r.T` along the trailing axis for `[..., N]` complex128 inputs.
    """
    n_int = _validate_n(n)
    x_arr = _coerce_numeric_array(x, allow_complex=True, name="x")
    if x_arr.ndim < 1 or x_arr.shape[-1] != n_int:
        raise ValueError(f"Expected x with trailing dimension {n_int}, got shape {x_arr.shape}")

    g = _coerce_numeric_array(ray_gains_row, allow_complex=True, name="ray_gains_row")
    d = _coerce_real_delay_array(delays)
    if g.ndim != 2 or g.shape[1] != n_int or d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(f"Expected ray_gains_row [P, {n_int}] and delays [P], got {g.shape} and {d.shape}")

    out = np.zeros_like(x_arr, dtype=COMPLEX_DTYPE)
    for l_idx in range(d.shape[0]):
        d_mat = fractional_delay_matrix(float(d[l_idx]), n=n_int)
        out += (x_arr @ d_mat.T) * g[l_idx]
    return out


# --- Phase 11 Dataclasses ---

@dataclass(frozen=True)
class PerRayGateOutput:
    """Output of a per-ray factorized / subset-Bayes history-trust gate (compatible with GateOutput)."""
    gate_out: GateOutput
    ray_weights: np.ndarray
    subset_posteriors: Optional[np.ndarray]
    subset_logliks: Optional[np.ndarray]
    ray_diag: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AssimilationResult:
    """Output of strictly blind parity-certified wavefront cross-block assimilation."""
    gate_out: GateOutput
    decoded_info: dict[int, np.ndarray]
    assimilated_rows: tuple[int, ...]
    row_weights: dict[int, float]
    row_margins: dict[int, float]
    n_rounds: int
    elapsed_ms: float


@dataclass(frozen=True)
class OffGridObservation:
    """Strictly receiver-visible observables for Pillar 2 off-grid continuous-delay cross-waveform links."""
    y_a: np.ndarray
    y_b: np.ndarray
    training_rows: tuple[int, ...]
    probe_row: int
    n0: float
    pilot_tx_b: np.ndarray
    pilot_tx_a: np.ndarray


@dataclass(frozen=True)
class OffGridTruth:
    """Evaluator-only hidden ground truth for Pillar 2 (NEVER passed to any receiver at inference)."""
    regime: str
    n_rays: int
    delays: np.ndarray
    powers: np.ndarray
    freqs_a: np.ndarray
    freqs_b: np.ndarray
    chirps: np.ndarray
    changed_rays: tuple[int, ...]
    shift_row: Optional[int]
    corrupted_rows: tuple[int, ...]
    data_rows: tuple[int, ...]
    b_info: np.ndarray
    b_code: np.ndarray
    y_b_probe_noisy: np.ndarray
    y_b_probe_clean: np.ndarray
    h_b_true_mat: np.ndarray


__all__ = [
    "ALL_REGIMES",
    "ANCHOR",
    "AssimilationResult",
    "COMPLEX_DTYPE",
    "CORE_ROWS",
    "CP",
    "CandidateSummary",
    "DELAYS",
    "FRAME_DURATION",
    "GateOutput",
    "HISTORICAL_SEED_RANGES",
    "HiddenTruth",
    "LINK_DURATION",
    "N",
    "OFDM_PILOT_ROWS",
    "OFFGRID_ALL_REGIMES",
    "OFFGRID_ANCHOR_ROWS",
    "OFFGRID_DELAY_MAX",
    "OFFGRID_DELAY_MIN",
    "OFFGRID_RAY_COUNTS",
    "OFFGRID_REGIMES_ID",
    "OFFGRID_REGIMES_OOD",
    "OffGridObservation",
    "OffGridTruth",
    "P10_BASELINE_COMMIT",
    "P11_NAMESPACE_SPAN",
    "P11_SEED_RANGES",
    "P8C_BASELINE_COMMIT",
    "P8_MANIFOLD_COMMIT",
    "P9_BASELINE_COMMIT",
    "POWERS",
    "PRIMARY_SNR_DB",
    "PROBE_ROW",
    "PerRayGateOutput",
    "PublicObservation",
    "REAL_DTYPE",
    "REGIMES_ID",
    "REGIMES_OOD",
    "REGIMES_SANITY",
    "SC_LAYOUTS",
    "SEED_P11_DEV",
    "SEED_P11_E2E",
    "SEED_P11_FORMAL_ID",
    "SEED_P11_FORMAL_OOD",
    "SEED_P11_TRAIN",
    "SEED_P9_DEV",
    "SEED_P9_FORMAL_ID",
    "SEED_P9_FORMAL_OOD",
    "SEED_P9_TRAIN",
    "SNR_GRID_DB",
    "T",
    "apply_fractional_sc_operator",
    "build_fractional_sc_matrix",
    "build_offgrid_sc_channel_matrix",
    "enforce_determinism",
    "fractional_delay_freq_response",
    "fractional_delay_kernel",
    "fractional_delay_matrix",
    "is_historical_seed",
    "jeffreys_complex_gaussian",
    "mix_bit_llrs",
    "n0_from_snr",
    "signed_subcarrier_indices",
    "unitary_dft_matrix",
]
