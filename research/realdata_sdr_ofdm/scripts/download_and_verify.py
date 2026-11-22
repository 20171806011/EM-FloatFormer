#!/usr/bin/env python3
"""Download, SHA-256 verify, and probe rikluost/sdr_ofdm_dataset_v1 (F3)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import yaml
from torch.utils.data import Dataset

EXPECTED_SIZE_BYTES = 27684430
EXPECTED_BLOB_SHA1 = "9b1aee8673ec48f42f52dc365fb01cc8b3e2b741"
EXPECTED_SHA256 = "e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058"
EXIT_FALLBACK_TRIGGERED = 42

PILOT_ACTIVE_IDX = np.array(
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 101], dtype=np.int64
)
FFT_OFFSET = 13
PILOT_FFT_IDX = FFT_OFFSET + PILOT_ACTIVE_IDX
PILOT_SYMBOL_IDX = 2
DC_FFT_BIN = 64
PILOT_VALUES = np.tile(
    np.array([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j], dtype=np.complex128), 4
)[:14]


class CustomDataset(Dataset):
    """Exact replica of rikluost/sdr_ofdm_dataset_v1/config.py CustomDataset."""

    def __init__(self) -> None:
        self.pdsch_iq: list[Any] = []
        self.labels: list[Any] = []
        self.sinr: list[Any] = []

    def __len__(self) -> int:
        return len(self.pdsch_iq)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        return self.pdsch_iq[index], self.labels[index], self.sinr[index]

    def add_item(self, new_pdsch_iq: Any, new_label: Any, new_sinr: Any) -> None:
        self.pdsch_iq.append(new_pdsch_iq)
        self.labels.append(new_label)
        self.sinr.append(new_sinr)


_ALLOWED_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("_codecs", "encode"),
        ("builtins", "bytearray"),
        ("builtins", "complex"),
        ("builtins", "set"),
        ("builtins", "slice"),
        ("torch", "Tensor"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch", "FloatStorage"),
        ("torch", "DoubleStorage"),
        ("torch", "HalfStorage"),
        ("torch", "BFloat16Storage"),
        ("torch", "ComplexFloatStorage"),
        ("torch", "ComplexDoubleStorage"),
        ("torch", "LongStorage"),
        ("torch", "IntStorage"),
        ("torch", "ShortStorage"),
        ("torch", "CharStorage"),
        ("torch", "ByteStorage"),
        ("torch", "BoolStorage"),
        ("torch.storage", "TypedStorage"),
        ("torch.storage", "UntypedStorage"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_tensor_v3"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }
)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "CustomDataset" and (
            module in {"config", "__main__", __name__}
            or module.endswith(("download_and_verify", "sdr_ofdm_adapter"))
        ):
            return CustomDataset
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def _compat_load(file: Any, **kwargs: Any) -> Any:
    return _CompatUnpickler(file, **kwargs).load()


def _compat_loads(data: bytes | bytearray, **kwargs: Any) -> Any:
    return _CompatUnpickler(io.BytesIO(data), **kwargs).load()


class _CompatPickleModule:
    Unpickler = _CompatUnpickler
    load = staticmethod(_compat_load)
    loads = staticmethod(_compat_loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL


def _compute_hashes(path: Path) -> tuple[int, str, str]:
    data = path.read_bytes()
    size = len(data)
    sha256 = hashlib.sha256(data).hexdigest()
    git_header = f"blob {size}\0".encode("ascii")
    git_blob_sha1 = hashlib.sha1(git_header + data).hexdigest()
    return size, sha256, git_blob_sha1


def _download_file(url: str, dest: Path, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with tmp_dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_dest.replace(dest)


def _nongray_16qam_constellation() -> tuple[np.ndarray, np.ndarray]:
    a = np.arange(4, dtype=np.float64)
    b = a - np.mean(a)
    c = (b[:, None] + 1j * b[None, :]).reshape(-1).astype(np.complex128)
    c /= np.sqrt(np.mean(np.abs(c) ** 2))
    bits = np.array(
        [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)],
        dtype=np.uint8,
    )
    return c, bits


def _gray_16qam_constellation() -> tuple[np.ndarray, np.ndarray]:
    # Standard 3GPP Gray-coded 16-QAM: 00->-3, 01->-1, 11->+1, 10->+3
    levels = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64) / np.sqrt(10.0)
    c = (levels[:, None] + 1j * levels[None, :]).reshape(-1).astype(np.complex128)
    bits = np.array(
        [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)],
        dtype=np.uint8,
    )
    return c, bits


def _demap_hard_bits(z: np.ndarray, pts: np.ndarray, bit_tbl: np.ndarray) -> np.ndarray:
    idx = np.argmin(np.abs(z[..., None] - pts) ** 2, axis=-1)
    return bit_tbl[idx]


def _build_active_data_mask() -> np.ndarray:
    mask = np.ones((14, 128), dtype=bool)
    mask[:, :13] = False
    mask[:, 115:] = False
    mask[:, DC_FFT_BIN] = False
    mask[PILOT_SYMBOL_IDX, PILOT_FFT_IDX] = False
    return mask


def _ls_piecewise_interp_102(h_pilots: np.ndarray) -> np.ndarray:
    # Replicates OFDM_SDR_Functions_torch_v2.channelEstimate_LS piecewise linear interp on active 0..101
    x_active = np.arange(102, dtype=np.float64)
    h_re = np.interp(x_active, PILOT_ACTIVE_IDX.astype(np.float64), h_pilots.real)
    h_im = np.interp(x_active, PILOT_ACTIVE_IDX.astype(np.float64), h_pilots.imag)
    return (h_re + 1j * h_im).astype(np.complex128)


def _companion_phase_unwrap_zf(rx_grid: np.ndarray, h_102: np.ndarray) -> np.ndarray:
    # Replicates OFDM_SDR_Functions_torch_v2.equalize_ZF (including its > 1 rad unwrap quirk)
    h_full = np.broadcast_to(h_102[None, :], (14, 102))
    rx_act = rx_grid[:, 13:115]
    mag = np.abs(rx_act) / np.maximum(np.abs(h_full), 1e-12)
    phase_diff = np.angle(rx_act) - np.angle(h_full)
    # Companion unwrap: threshold > 1 rad along axis=1
    d = np.diff(phase_diff, axis=1)
    jumps = np.abs(d) > 1.0
    unwrapped = phase_diff.copy()
    if np.any(jumps):
        cum_shift = np.cumsum(np.where(jumps, d, 0.0), axis=1)
        unwrapped[:, 1:] -= np.where(jumps, cum_shift, 0.0)
    return mag * np.exp(1j * unwrapped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download, verify, and probe SDR OFDM dataset.")
    parser.add_argument("--config", default="research/realdata_sdr_ofdm/configs/default.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--manifest-out", default="research/realdata_sdr_ofdm/manifests/raw_checksums.json")
    parser.add_argument("--probe-out", default="/tmp/tw_realdata/outputs/probe_summary.json")
    parser.add_argument("--probe-n", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not (16 <= args.probe_n <= 64):
        raise ValueError(f"--probe-n must be in [16, 64], got {args.probe_n}")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds_cfg = cfg["dataset"]
    raw_dir = Path(args.raw_dir or ds_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_pth = raw_dir / ds_cfg["raw_filename"]
    fallback_reasons: list[str] = []
    companion_records: list[dict[str, Any]] = []

    if not args.skip_download:
        try:
            _download_file(ds_cfg["url"], raw_pth)
        except Exception as exc:  # noqa: BLE001
            fallback_reasons.append(f"T1: Primary download failed: {exc}")
        for comp in ds_cfg.get("companion_files", []):
            comp_dest = raw_dir / comp["name"]
            try:
                _download_file(comp["url"], comp_dest)
                c_size, c_sha256, c_blob = _compute_hashes(comp_dest)
                companion_records.append(
                    {
                        "name": comp["name"],
                        "url": comp["url"],
                        "local_path": str(comp_dest),
                        "size_bytes": c_size,
                        "sha256": c_sha256,
                        "git_blob_sha1": c_blob,
                        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                companion_records.append({"name": comp["name"], "url": comp["url"], "error": str(exc)})
    else:
        for comp in ds_cfg.get("companion_files", []):
            comp_dest = raw_dir / comp["name"]
            if comp_dest.exists():
                c_size, c_sha256, c_blob = _compute_hashes(comp_dest)
                companion_records.append(
                    {
                        "name": comp["name"],
                        "url": comp["url"],
                        "local_path": str(comp_dest),
                        "size_bytes": c_size,
                        "sha256": c_sha256,
                        "git_blob_sha1": c_blob,
                    }
                )

    raw_size, raw_sha256, raw_blob_sha1 = (0, "", "")
    if raw_pth.exists():
        raw_size, raw_sha256, raw_blob_sha1 = _compute_hashes(raw_pth)
        expected_size = int(ds_cfg.get("expected_size_bytes", EXPECTED_SIZE_BYTES))
        expected_blob = str(ds_cfg.get("expected_blob_sha1", EXPECTED_BLOB_SHA1))
        raw_expected_sha = ds_cfg.get("expected_sha256", EXPECTED_SHA256)
        expected_sha256 = "" if raw_expected_sha is None else str(raw_expected_sha).strip()
        if raw_size != expected_size:
            fallback_reasons.append(f"T1: size_bytes={raw_size} != expected {expected_size}")
        if raw_blob_sha1 != expected_blob:
            fallback_reasons.append(f"T1: git_blob_sha1={raw_blob_sha1} != expected {expected_blob}")
        # Verify expected_sha256 whenever primary defaults match OR a custom expected_sha256 is configured
        # (E2E fixtures T1-F3.3/T1-F3.4/T2-F3.4 override size/blob on default.yaml without resetting expected_sha256).
        if expected_sha256 and (
            (expected_sha256 != EXPECTED_SHA256)
            or (expected_blob == EXPECTED_BLOB_SHA1 and expected_size == EXPECTED_SIZE_BYTES)
        ):
            if raw_sha256 != expected_sha256:
                fallback_reasons.append(f"T1: sha256={raw_sha256} != expected {expected_sha256}")
    else:
        fallback_reasons.append(f"T1: Raw file missing at {raw_pth}")

    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_cfg = ds_cfg.get("backup", {})
    raw_manifest = {
        "generated_utc": now_utc,
        "primary": {
            "dataset_name": ds_cfg["name"],
            "url": ds_cfg["url"],
            "local_path": str(raw_pth),
            "size_bytes": raw_size,
            "sha256": raw_sha256,
            "git_blob_sha1": raw_blob_sha1,
            "downloaded_utc": now_utc,
        },
        "companion_files": companion_records,
        "backup": {
            "dataset_name": backup_cfg.get("name", "ADALM-PLUTO QPSK"),
            "doi": backup_cfg.get("doi", "10.17632/k72b4pnc5g.1"),
            "url": backup_cfg.get("url", ""),
            "raw_filename": backup_cfg.get("raw_filename", "QPSKlink.mat"),
            "expected_size_bytes": backup_cfg.get("expected_size_bytes", 822762822),
            "expected_sha256": backup_cfg.get("expected_sha256", ""),
            "downloaded": False,
        },
    }
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    probe_summary: dict[str, Any] = {
        "generated_utc": now_utc,
        "raw_path": str(raw_pth),
        "raw_sha256": raw_sha256,
        "probe_n": int(args.probe_n),
        "fallback_triggered": False,
        "fallback_reasons": fallback_reasons,
    }

    dataset = None
    if not fallback_reasons:
        try:
            dataset = torch.load(
                raw_pth,
                map_location="cpu",
                weights_only=False,
                pickle_module=_CompatPickleModule,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_reasons.append(f"T2: Unpickling failed: {exc}")

    if dataset is not None:
        for attr in ("pdsch_iq", "labels", "sinr"):
            if not hasattr(dataset, attr):
                fallback_reasons.append(f"T2: Missing attribute '{attr}' on unpickled dataset")

    if dataset is not None and not fallback_reasons:
        n_iq = len(dataset.pdsch_iq)
        n_lab = len(dataset.labels)
        n_sinr = len(dataset.sinr)
        if not (n_iq == n_lab == n_sinr):
            fallback_reasons.append(f"T2: Unequal lengths iq={n_iq}, labels={n_lab}, sinr={n_sinr}")
        n_tti = n_iq
        probe_summary["n_tti"] = int(n_tti)
        if n_tti < int(cfg["probe"].get("min_tti_count", 200)):
            fallback_reasons.append(f"T3: n_tti={n_tti} < 200")

        try:
            sinr_vals = np.array([float(x) for x in dataset.sinr], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            fallback_reasons.append(f"T2: Non-numeric values in sinr: {exc}")
            sinr_vals = np.array([], dtype=np.float64)

        sinr_all_finite = len(sinr_vals) > 0 and bool(np.all(np.isfinite(sinr_vals)))
        if len(sinr_vals) > 0 and not sinr_all_finite:
            fallback_reasons.append("T2: Non-finite values in sinr")

        if n_tti == 0 or len(sinr_vals) == 0 or not sinr_all_finite:
            probe_summary["sinr_stats"] = None
            if not fallback_reasons:
                fallback_reasons.append("T3: Empty dataset (n_tti=0)")
        else:
            quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            q_vals = np.quantile(sinr_vals, quantiles)
            probe_summary["sinr_stats"] = {
                "min": float(np.min(sinr_vals)),
                "max": float(np.max(sinr_vals)),
                "mean": float(np.mean(sinr_vals)),
                "std": float(np.std(sinr_vals)),
                "quantiles": {str(q): float(v) for q, v in zip(quantiles, q_vals)},
            }

            probe_indices = np.linspace(0, n_tti - 1, args.probe_n, dtype=int).tolist()
            probe_summary["probe_indices"] = probe_indices

            data_mask = _build_active_data_mask()
            data_mask_102 = data_mask[:, 13:115]
            pts_nongray, bits_nongray = _nongray_16qam_constellation()
            pts_gray, bits_gray = _gray_16qam_constellation()

            guard_bins = list(range(0, 13)) + list(range(115, 128))
            per_bin_power_accum = np.zeros(128, dtype=np.float64)
            guard_powers = []
            inner_guard_powers = []
            outer_guard_powers = []
            active_powers = []
            dc_powers = []
            pilot_powers = []
            coherences = []
            null_conj_coherences = []
            null_shift_coherences = []

            probe_nongray_errs = 0
            probe_nongray_bits = 0
            probe_gray_errs = 0
            probe_companion_unwrap_errs = 0

            for idx in probe_indices:
                iq_raw = np.asarray(
                    dataset.pdsch_iq[idx].detach().cpu().numpy()
                    if torch.is_tensor(dataset.pdsch_iq[idx])
                    else dataset.pdsch_iq[idx]
                )
                lab_raw = np.asarray(
                    dataset.labels[idx].detach().cpu().numpy()
                    if torch.is_tensor(dataset.labels[idx])
                    else dataset.labels[idx]
                )
                if iq_raw.shape != (14, 128) or not np.all(np.isfinite(iq_raw)):
                    fallback_reasons.append(f"T2: Invalid iq shape/finite at idx={idx}: {iq_raw.shape}")
                    break
                if lab_raw.shape != (1400, 4) or not np.all(np.isfinite(lab_raw)):
                    fallback_reasons.append(f"T2: Invalid labels shape/finite at idx={idx}: {lab_raw.shape}")
                    break
                if not np.all((lab_raw == 0) | (lab_raw == 1)):
                    fallback_reasons.append(f"T2: Non-binary labels at idx={idx}")
                    break

                iq = iq_raw.astype(np.complex128)
                bits_true = lab_raw.astype(np.uint8)

                bin_p = np.mean(np.abs(iq) ** 2, axis=0)
                per_bin_power_accum += bin_p
                g_p = float(np.mean(bin_p[guard_bins]))
                ig_p = float(np.mean(bin_p[[9, 10, 11, 12, 115, 116, 117, 118, 119]]))
                og_p = float(np.mean(bin_p[[0, 1, 2, 3, 4, 5, 6, 7, 8, 120, 121, 122, 123, 124, 125, 126, 127]]))
                act_p = float(np.mean(bin_p[13:115]))
                dc_p = float(bin_p[DC_FFT_BIN])
                y_p = iq[PILOT_SYMBOL_IDX, PILOT_FFT_IDX]
                pil_p = float(np.mean(np.abs(y_p) ** 2))

                guard_powers.append(g_p)
                inner_guard_powers.append(ig_p)
                outer_guard_powers.append(og_p)
                active_powers.append(act_p)
                dc_powers.append(dc_p)
                pilot_powers.append(pil_p)

                h_p = y_p / PILOT_VALUES
                denom = float(np.sum(np.abs(h_p) ** 2))
                coh = float(np.abs(np.sum(h_p[:-1] * np.conj(h_p[1:]))) / max(denom, 1e-18))
                h_conj_null = y_p / np.conj(PILOT_VALUES)
                coh_conj = float(
                    np.abs(np.sum(h_conj_null[:-1] * np.conj(h_conj_null[1:])))
                    / max(float(np.sum(np.abs(h_conj_null) ** 2)), 1e-18)
                )
                h_shift_null = y_p / np.roll(PILOT_VALUES, 1)
                coh_shift = float(
                    np.abs(np.sum(h_shift_null[:-1] * np.conj(h_shift_null[1:])))
                    / max(float(np.sum(np.abs(h_shift_null) ** 2)), 1e-18)
                )
                coherences.append(coh)
                null_conj_coherences.append(coh_conj)
                null_shift_coherences.append(coh_shift)

                h_102 = _ls_piecewise_interp_102(h_p)
                h_full_102 = np.broadcast_to(h_102[None, :], (14, 102))
                z_clean = iq[:, 13:115][data_mask_102] / np.where(
                    np.abs(h_full_102[data_mask_102]) > 1e-12, h_full_102[data_mask_102], 1e-12
                )
                b_nongray = _demap_hard_bits(z_clean, pts_nongray, bits_nongray)
                b_gray = _demap_hard_bits(z_clean, pts_gray, bits_gray)
                z_comp = _companion_phase_unwrap_zf(iq, h_102)[data_mask_102]
                b_comp = _demap_hard_bits(z_comp, pts_nongray, bits_nongray)

                probe_nongray_errs += int(np.sum(b_nongray != bits_true))
                probe_gray_errs += int(np.sum(b_gray != bits_true))
                probe_companion_unwrap_errs += int(np.sum(b_comp != bits_true))
                probe_nongray_bits += int(bits_true.size)

            # Full-dataset clean LS/ZF non-Gray check for T5
            full_nongray_errs = 0
            full_nongray_bits = 0
            full_frame_errs = 0
            if not fallback_reasons:
                for idx in range(n_tti):
                    iq_raw = np.asarray(
                        dataset.pdsch_iq[idx].detach().cpu().numpy()
                        if torch.is_tensor(dataset.pdsch_iq[idx])
                        else dataset.pdsch_iq[idx]
                    )
                    lab_raw = np.asarray(
                        dataset.labels[idx].detach().cpu().numpy()
                        if torch.is_tensor(dataset.labels[idx])
                        else dataset.labels[idx]
                    )
                    if iq_raw.shape != (14, 128) or lab_raw.shape != (1400, 4):
                        fallback_reasons.append(f"T2: Full-scan invalid shape at idx={idx}")
                        break
                    iq = iq_raw.astype(np.complex128)
                    bits_true = lab_raw.astype(np.uint8)
                    h_p = iq[PILOT_SYMBOL_IDX, PILOT_FFT_IDX] / PILOT_VALUES
                    h_102 = _ls_piecewise_interp_102(h_p)
                    h_full_102 = np.broadcast_to(h_102[None, :], (14, 102))
                    z_clean = iq[:, 13:115][data_mask_102] / np.where(
                        np.abs(h_full_102[data_mask_102]) > 1e-12, h_full_102[data_mask_102], 1e-12
                    )
                    b_nongray = _demap_hard_bits(z_clean, pts_nongray, bits_nongray)
                    errs = int(np.sum(b_nongray != bits_true))
                    full_nongray_errs += errs
                    if errs > 0:
                        full_frame_errs += 1
                    full_nongray_bits += int(bits_true.size)

            n_probed = len(guard_powers)
            per_bin_power = per_bin_power_accum / max(n_probed, 1)
            guard_profile = {str(k): float(per_bin_power[k]) for k in guard_bins}
            active_edge_profile = {str(k): float(per_bin_power[k]) for k in [13, 14, 63, 64, 65, 113, 114]}

            def _opt_mean(vals: list[float]) -> float | None:
                return float(np.mean(vals)) if vals else None

            mean_guard_p = _opt_mean(guard_powers)
            mean_inner_guard_p = _opt_mean(inner_guard_powers)
            mean_outer_guard_p = _opt_mean(outer_guard_powers)
            mean_active_p = _opt_mean(active_powers)
            mean_dc_p = _opt_mean(dc_powers)
            mean_pilot_p = _opt_mean(pilot_powers)
            pilot_guard_ratio = (
                (mean_pilot_p / max(mean_guard_p, 1e-18))
                if (mean_pilot_p is not None and mean_guard_p is not None)
                else None
            )
            pilot_inner_guard_ratio = (
                (mean_pilot_p / max(mean_inner_guard_p, 1e-18))
                if (mean_pilot_p is not None and mean_inner_guard_p is not None)
                else None
            )
            mean_coh = _opt_mean(coherences)
            null_conj_coh = _opt_mean(null_conj_coherences)
            null_shift_coh = _opt_mean(null_shift_coherences)

            probe_ber = (probe_nongray_errs / probe_nongray_bits) if probe_nongray_bits > 0 else None
            probe_gray_ber = (probe_gray_errs / probe_nongray_bits) if probe_nongray_bits > 0 else None
            probe_comp_ber = (probe_companion_unwrap_errs / probe_nongray_bits) if probe_nongray_bits > 0 else None
            full_ber = (full_nongray_errs / full_nongray_bits) if full_nongray_bits > 0 else None
            full_fer = (full_frame_errs / n_tti) if (full_nongray_bits > 0 and n_tti > 0) else None

            probe_summary.update(
                {
                    "power_profile": {
                        "mean_guard_power": mean_guard_p,
                        "mean_inner_guard_power_9_12_115_119": mean_inner_guard_p,
                        "mean_outer_guard_power_0_8_120_127": mean_outer_guard_p,
                        "mean_active_power": mean_active_p,
                        "mean_dc_power": mean_dc_p,
                        "mean_pilot_power": mean_pilot_p,
                        "pilot_guard_power_ratio": pilot_guard_ratio,
                        "pilot_inner_guard_power_ratio": pilot_inner_guard_ratio,
                        "per_bin_guard_power": guard_profile,
                        "per_bin_edge_dc_power": active_edge_profile,
                    },
                    "pilot_coherence": {
                        "mean_pilot_coherence": mean_coh,
                        "null_conj_coherence": null_conj_coh,
                        "null_shift_coherence": null_shift_coh,
                    },
                    "ls_zf_probe": {
                        "nongray_bit_errors": probe_nongray_errs,
                        "total_bits": probe_nongray_bits,
                        "nongray_ber": probe_ber,
                        "gray_bit_errors": probe_gray_errs,
                        "gray_ber": probe_gray_ber,
                        "companion_unwrap_bit_errors": probe_companion_unwrap_errs,
                        "companion_unwrap_ber": probe_comp_ber,
                    },
                    "ls_zf_full": {
                        "nongray_bit_errors": full_nongray_errs,
                        "total_bits": full_nongray_bits,
                        "nongray_ber": full_ber,
                        "frame_errors": full_frame_errs,
                        "total_frames": int(n_tti),
                        "nongray_fer": full_fer,
                    },
                }
            )

            if mean_coh is not None and mean_coh < float(cfg["probe"].get("min_pilot_coherence", 0.30)):
                fallback_reasons.append(f"T4: mean_pilot_coherence={mean_coh:.4f} < 0.30")
            if pilot_guard_ratio is not None and pilot_guard_ratio <= float(
                cfg["probe"].get("min_pilot_guard_power_ratio", 2.0)
            ):
                fallback_reasons.append(f"T4: pilot_guard_power_ratio={pilot_guard_ratio:.4f} <= 2.0")
            if probe_ber is not None and probe_ber >= float(cfg["probe"].get("max_ls_zf_ber", 0.40)):
                fallback_reasons.append(f"T4: ls_zf_nongray_ber={probe_ber:.4f} >= 0.40")
            if probe_ber is not None and probe_gray_ber is not None and probe_ber >= probe_gray_ber:
                fallback_reasons.append(
                    f"T4: ls_zf_nongray_ber={probe_ber:.4f} >= ls_zf_gray_ber={probe_gray_ber:.4f}"
                )
            if full_nongray_bits > 0 and full_nongray_errs == 0 and probe_nongray_errs == 0:
                fallback_reasons.append("T5: Degenerate zero-error regime across all TTIs")
            if (
                probe_summary["sinr_stats"] is not None
                and (probe_summary["sinr_stats"]["max"] - probe_summary["sinr_stats"]["min"] < 1.0)
                and full_nongray_bits > 0
                and full_nongray_errs == 0
            ):
                fallback_reasons.append("T5: Flat SINR (< 1 dB) with zero bit errors")

    if fallback_reasons:
        probe_summary["fallback_triggered"] = True
        probe_summary["fallback_reasons"] = fallback_reasons

    probe_out = Path(args.probe_out)
    probe_out.parent.mkdir(parents=True, exist_ok=True)
    probe_out.write_text(json.dumps(probe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(probe_summary, indent=2, allow_nan=False))
    if fallback_reasons:
        return EXIT_FALLBACK_TRIGGERED
    return 0


if __name__ == "__main__":
    sys.exit(main())
