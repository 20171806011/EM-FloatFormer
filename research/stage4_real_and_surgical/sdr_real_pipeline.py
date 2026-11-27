"""Strictly causal M4/M5 real-SDR pipeline for ADALM-PLUTO and CTTC Indoor QPSK.

Implements:
  1. Verified downloaders with exact byte-size, MD5, and SHA-256 verification for
     Pluto (`QPSKlink.mat`, DOI 10.17632/k72b4pnc5g.1) and CTTC (Zenodo 4895133, 9 captures).
  2. Strictly causal M4 waveform frontends using ONLY `SYNC_SLICE [0:13]` (Pluto Copy 1)
     and `SYNC_IDX [0:6]` (CTTC packet start) plus causal variance floors, with
     adversarial sentinel-poisoning zero-leakage certification.
  3. Bit-exact dual-cache serialization (`4-bit s17` prefix inside `16-bit m5-c16-v1-le`),
     M3 canonical per-row `RefSufficientPool` summaries (`duplicate == two`, `indep == two`),
     and matched-information CPU physical comparisons (`4-bit s17` vs `16-bit c16` vs `continuous`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.interpolate import CubicSpline

from research.stage3_ministral_cache.schema import ROOTS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import load_s17_encoder, public_phase_slots
from research.stage4_real_and_surgical.equi_connector import lift_scores
from research.stage4_real_and_surgical.packet_c16 import (
    WIRE_ID,
    WIRE_QUALITY_EDGES,
    WIRE_RADIAL_EDGES,
    decode_c16,
    encode_c16,
)
from research.stage4_real_and_surgical.real_sdr_causal import CTTC_MANIFEST, PLUTO_MANIFEST


# =============================================================================
# 1. Pinned Download Manifests & CTTC Capture Anchors (Zero Offline Leakage)
# =============================================================================

PLUTO_DATASET_ID = "k72b4pnc5g"
PLUTO_DATASET_VERSION = "1"
PLUTO_METADATA_URL = (
    f"https://data.mendeley.com/public-api/datasets/{PLUTO_DATASET_ID}?version={PLUTO_DATASET_VERSION}"
)
PLUTO_FILES_URL = (
    f"https://data.mendeley.com/public-api/datasets/{PLUTO_DATASET_ID}/files?folder_id=root&version={PLUTO_DATASET_VERSION}"
)
PLUTO_FILENAME = "QPSKlink.mat"
PLUTO_EXPECTED_SIZE = 822_762_822
PLUTO_EXPECTED_SHA256 = "c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7"

ZENODO_RECORD_ID = "4895133"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

# Copied verbatim from origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/download_cttc.py
CTTC_MANIFEST_FILES: Tuple[Dict[str, Any], ...] = (
    {
        "filename": "sources.tar.gz",
        "role": "tx_source_bits",
        "size_bytes": 157_051_651,
        "md5": "48766e3f5ddaa968ccf128eb5891a450",
        "sha256": "f9022be37d3eeadcff481f46b25ca9a067c0e19988bc4bdcb77cca9957ee7b4f",
        "extracted_files": {
            "source_QPSK.dat": {
                "size_bytes": 1_000_000,
                "sha256": "6dc9635b72b34f2d792f7ec6522a0450181397fb616262b629ee593e35f3f1cb",
            }
        },
    },
    {
        "filename": "f2450_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr20_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 20,
        "size_bytes": 211_697_276,
        "md5": "1df69db08519d130faff0857347930de",
        "sha256": "161d618ef2da2412a3c4d7236deb7c52f0b81f31af704b9d852d90f849c6cbb3",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "7e9e6ff2af0f10f2cf0a47679394a6443a1fb8051c6e0c61acacd29c55a5389e"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "c6596f98fd9705c475b8b07340facb6ad7dbc4906e6f196ac2f207a275544ab1"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "20c521344958acda0cd30dcd0449a41ab16d472e6942bee0d5e539c788c7a4db"},
        },
    },
    {
        "filename": "f2450_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr10_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 10,
        "size_bytes": 197_835_355,
        "md5": "545e2f96b666200da371bb51726bb336",
        "sha256": "e6b7462ec66d485b1b10936e3f86601c91e900dddad9b8b9e19b789c3e183d25",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "b9dde9083a6b228929913b577f2bfc2e2e9e50a61638a0dbe68bc7fccf6cd221"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "3d56b0f14b97755ecdfae2011f011165a6837c2caa363a9cbfaedc0dfcd53d0c"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "b91aba4447771e7dfd149515067ee64e1303ae44c6669d65504f17899070f331"},
        },
    },
    {
        "filename": "f2450_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr0_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 0,
        "size_bytes": 191_552_426,
        "md5": "8ef953dfadfb7363d70924f79f4755bc",
        "sha256": "e62681b28bb02c51671ffe3f3c36030d7a0e0839440adbf641a8e63f53b95ea9",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "111e8a12b35d5529e8e49c781de80d2ad05172e73ef2e65f747229ad0f63bf07"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "14cdb93e54351c5d9b2ac254a4f670863dd4a83bfa7ed48695186d64bf2460d2"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "7c390cdf2a55f800c2a1ad46e0cdf732630cc9cc3a48ab763d6f39184c3ff615"},
        },
    },
    {
        "filename": "f708_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr20_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 20,
        "size_bytes": 212_954_092,
        "md5": "afbd783156079f3ab2b1a33690540bad",
        "sha256": "1732da579f2e98ade65e2570f4141e9b881d6c891f8dec83a3bd723e05f8d1af",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a4569e6bce65a47dc227470f36b7d1607db32a1049011bda3440d296cc1b21e5"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "1f2b2b22c2345df8b2e67f0a7a44e525a65615723cf878eb63f231dc302e2436"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "8d05494ec48bae4456f4f96cbfb3a648fc81c9bf161388432e82d05f89b8a361"},
        },
    },
    {
        "filename": "f708_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr10_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 10,
        "size_bytes": 198_519_362,
        "md5": "fe9ceeb85392f1476289214736640b1c",
        "sha256": "b688f19db545edeef24af628283ff5080b7bd32f0277fd593b44a24da5a791f9",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "e9133b9cf31bcee6c36eb5f10e4da78cf60129f2fce22de1f0d9e4cd3917b0c3"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "962c8e5a41d8d4b6827dc8c6ef09bb7a931d1ae4c19c89203794bf1b482cb21e"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "6a7cd6c531d9125a2baa66c391de06d83d1079deebb4cf240a6bf2755051c1d9"},
        },
    },
    {
        "filename": "f708_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr0_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 0,
        "size_bytes": 190_371_348,
        "md5": "d971092391ddccc893791ece9ad71bf3",
        "sha256": "adc850b02149e0cf9a6f27bdba2c93a3290fdf16d2d414725dbcaccca7a22a00",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "4312bb95378d0d9c4fe67f57267a256de9834459a59e6d636e58edcd8ab52a6d"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "6f67ca836a315aafa5e00328a54fae5a0e350dac96d4e9070012205fb24062dd"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "1b7815e226e3ececc50f9a9027a5e15f3fbd09514bb86889b912ba51596c4b85"},
        },
    },
    {
        "filename": "f433_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr20_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 20,
        "size_bytes": 211_968_737,
        "md5": "8dbb0ce9f09b683de8dbc9dc76997b74",
        "sha256": "c6e28fae2d06e1038ae990fdb231b39b7cd37322fe5eb498a1f755766abcd512",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8c9a105246bcb27949fd1717029edaa1079264868316248fbab8825fa1087ab2"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "058e3fa662941648faac9512e2b0695a81fd5d9cdcab62bfb6c245d394c4a01c"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "379753c9b2f1d151d3228b7d8661a9d79e2ac182116ceb22f5b26275483b3e17"},
        },
    },
    {
        "filename": "f433_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr10_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 10,
        "size_bytes": 199_205_337,
        "md5": "8e8fa8e8822eab62320a0baebb87e7c5",
        "sha256": "2067a4fd82671d0b86fd47e8f3fab5f87573e8653c7dbc6321dcba23a701b5d4",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a68edb1cd13814063a1c2036722cc57209f60868a4517dfe884eefd31cf51c42"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "d635beed43c87efd7cbf6988157296d8259c556c34c6c3594d345eec539bf085"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "103d64391bd4dd85ac249bc6f75a5503c92a365e99e7d43bbe4a7d8774786b00"},
        },
    },
    {
        "filename": "f433_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr0_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 0,
        "size_bytes": 193_721_020,
        "md5": "9cc2a791ef1b07977b5c43760a06bda8",
        "sha256": "c7b7bcba97151ad4dadebe3c21fb3503d44126c6912e56e024a19d303b17ef2f",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8f9b54ba0153ae3f9e53a6ba6e231708934c7a1b42957120ee122ebe5ec9aeca"},
            "file_1.dat": {"size_bytes": 176_015_264, "sha256": "84f47c34c7daac92412742af7133ac984b2e713408b8b6e57665708c07a64e43"},
            "file_2.dat": {"size_bytes": 176_015_264, "sha256": "98d19a36fbc990d9fd7214eb7855482703f76650b5fca7ef1d513730e9541d54"},
        },
    },
)

# Public/non-target frame geometry & background thermal floor per capture
# (Note: NO offline_cfo_med_hz, NO h_offline_ref, NO offline_sig_pow exposed!)
CTTC_CAPTURE_ANCHORS: Tuple[Dict[str, Any], ...] = (
    {"capture_id": "f2450_snr20_QPSK", "freq_mhz": 2450, "nom_snr_db": 20, "bg_noise_var": 3.0642262061064685e-09, "anchor_rx_pos": 4000000.717744379, "anchor_src_sym": 492892, "sps_eff": 4.000002029292535, "bit_alignment": 0},
    {"capture_id": "f2450_snr10_QPSK", "freq_mhz": 2450, "nom_snr_db": 10, "bg_noise_var": 3.0842834952693465e-09, "anchor_rx_pos": 4000003.273988869, "anchor_src_sym": 492871, "sps_eff": 4.000001923272496, "bit_alignment": 0},
    {"capture_id": "f2450_snr0_QPSK",  "freq_mhz": 2450, "nom_snr_db": 0,  "bg_noise_var": 2.9808777668449693e-09, "anchor_rx_pos": 4000001.4816942867, "anchor_src_sym": 493763, "sps_eff": 4.000002092946086, "bit_alignment": 0},
    {"capture_id": "f708_snr20_QPSK",  "freq_mhz": 708,  "nom_snr_db": 20, "bg_noise_var": 2.7648849876271697e-09, "anchor_rx_pos": 4000001.7397749024, "anchor_src_sym": 493601, "sps_eff": 4.000002218034584, "bit_alignment": 0},
    {"capture_id": "f708_snr10_QPSK",  "freq_mhz": 708,  "nom_snr_db": 10, "bg_noise_var": 2.777065688519542e-09,  "anchor_rx_pos": 4000000.456007515,  "anchor_src_sym": 493414, "sps_eff": 4.0000021036173665, "bit_alignment": 0},
    {"capture_id": "f708_snr0_QPSK",   "freq_mhz": 708,  "nom_snr_db": 0,  "bg_noise_var": 2.8434006260624756e-09, "anchor_rx_pos": 4000000.960871691,  "anchor_src_sym": 492771, "sps_eff": 4.00000209393422,   "bit_alignment": 0},
    {"capture_id": "f433_snr20_QPSK",  "freq_mhz": 433,  "nom_snr_db": 20, "bg_noise_var": 3.1134732569881862e-09, "anchor_rx_pos": 4000002.3574077785, "anchor_src_sym": 235640, "sps_eff": 4.000001561883662,  "bit_alignment": 1},
    {"capture_id": "f433_snr10_QPSK",  "freq_mhz": 433,  "nom_snr_db": 10, "bg_noise_var": 3.0300049136400276e-09, "anchor_rx_pos": 4000003.2122689085, "anchor_src_sym": 492790, "sps_eff": 4.000002020701151,  "bit_alignment": 0},
    {"capture_id": "f433_snr0_QPSK",   "freq_mhz": 433,  "nom_snr_db": 0,  "bg_noise_var": 3.101694234786123e-09,  "anchor_rx_pos": 4000003.3067990313, "anchor_src_sym": 235845, "sps_eff": 4.000001454072232,  "bit_alignment": 1},
)

# Physical waveform geometry
PLUTO_SPS = 20
PLUTO_N_SYM = 126
PLUTO_FRAME_SAMP = PLUTO_N_SYM * PLUTO_SPS  # 2520
PLUTO_RX_SAMP = 7560
BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)
PLUTO_SYNC_HDR_13 = (0.707 + 0.707j) * BARKER13
V0_NOM = 10.0 ** (-0.2)  # 0.63095734448

CTTC_FS_SYM = 500_000.0
CTTC_FS_RX = 2_000_000.0
CTTC_SPS = 4
CTTC_PKT_LEN = 256
CTTC_PAD_SYMS = 8
CTTC_SRC_N_SYMS = 500_000
CTTC_SYNC_IDX = np.arange(0, 6, dtype=np.int64)
CTTC_HOLDOUT_IDX = np.concatenate([np.arange(40, 48, dtype=np.int64), np.arange(200, 208, dtype=np.int64)])

# Gray QPSK table in standard e^{i pi/4} * ROOTS ordering:
# idx 0:(+1+1j)/s2=(0,0), 1:(-1+1j)/s2=(1,0), 2:(-1-1j)/s2=(1,1), 3:(+1-1j)/s2=(0,1)
_GRAY_BITS_BY_C4 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


# Fixed orthonormal basis (3, 2) orthogonal to e = [1, 1, 1] / sqrt(3) for R3 lift
def _orthonormal_perp_basis() -> np.ndarray:
    b1 = np.array([1.0, -1.0, 0.0], dtype=np.float64) / math.sqrt(2.0)
    b2 = np.array([1.0, 1.0, -2.0], dtype=np.float64) / math.sqrt(6.0)
    return np.column_stack([b1, b2])


_B_PERP = _orthonormal_perp_basis()
_E_DIR = np.full(3, 1.0 / math.sqrt(3.0), dtype=np.float64)


def _sha256_md5_file(path: Path) -> Tuple[int, str, str]:
    h_md5 = hashlib.md5()
    h_sha = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            total += len(chunk)
            h_md5.update(chunk)
            h_sha.update(chunk)
    return total, h_md5.hexdigest(), h_sha.hexdigest()


def _download_stream(url: str, dest: Path, expected_size: Optional[int] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    req = urllib.request.Request(url, headers={"User-Agent": "EM-FloatFormer-Stage4/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        while True:
            buf = resp.read(4 * 1024 * 1024)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
    if expected_size is not None and total != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download size mismatch for {dest.name}: {total} != {expected_size}")
    os.replace(tmp, dest)


def download_and_verify_pluto(raw_dir: Path) -> Path:
    """Download `QPSKlink.mat` from Mendeley DOI 10.17632/k72b4pnc5g.1 and verify SHA-256."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    mat_path = raw_dir / PLUTO_FILENAME
    if mat_path.exists() and mat_path.stat().st_size == PLUTO_EXPECTED_SIZE:
        _, _, sha = _sha256_md5_file(mat_path)
        if sha == PLUTO_EXPECTED_SHA256:
            return mat_path

    req = urllib.request.Request(PLUTO_FILES_URL, headers={"User-Agent": "EM-FloatFormer-Stage4/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        files_meta = json.loads(resp.read().decode("utf-8"))
    target = next((e for e in files_meta if e.get("filename") == PLUTO_FILENAME), None)
    if target is None:
        raise RuntimeError(f"Missing {PLUTO_FILENAME} in Mendeley files metadata")
    details = target["content_details"]
    dl_url = str(details["download_url"])
    api_size = int(details["size"])
    api_sha = str(details["sha256_hash"]).lower()
    assert api_size == PLUTO_EXPECTED_SIZE and api_sha == PLUTO_EXPECTED_SHA256
    _download_stream(dl_url, mat_path, expected_size=PLUTO_EXPECTED_SIZE)
    sz, _, sha = _sha256_md5_file(mat_path)
    if sz != PLUTO_EXPECTED_SIZE or sha != PLUTO_EXPECTED_SHA256:
        raise RuntimeError(f"Pluto SHA-256/size mismatch: size={sz}, sha={sha}")
    return mat_path


def download_and_verify_cttc(raw_dir: Path, work_dir: Path) -> Path:
    """Download and SHA-256/MD5 verify all CTTC Zenodo 4895133 archives and extracted streams."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    src_dir = work_dir / "sources"
    data_dir = work_dir / "data"
    src_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for item in CTTC_MANIFEST_FILES:
        fn = item["filename"]
        dst = raw_dir / fn
        already_extracted = True
        target = src_dir if item["role"] == "tx_source_bits" else data_dir
        for rel, meta in item["extracted_files"].items():
            ep = (src_dir / rel) if item["role"] == "tx_source_bits" else (data_dir / item["capture_id"] / rel)
            if not ep.exists() or ep.stat().st_size != meta["size_bytes"]:
                already_extracted = False
                break
        if already_extracted:
            # Verify extracted SHA-256s cleanly without re-downloading
            ok_sha = True
            for rel, meta in item["extracted_files"].items():
                ep = (src_dir / rel) if item["role"] == "tx_source_bits" else (data_dir / item["capture_id"] / rel)
                _, _, esha = _sha256_md5_file(ep)
                if esha != meta["sha256"]:
                    ok_sha = False
                    break
            if ok_sha:
                continue

        if not dst.exists() or dst.stat().st_size != item["size_bytes"]:
            _download_stream(f"{ZENODO_BASE_URL}/{fn}/content", dst, expected_size=item["size_bytes"])
        sz, md5, sha = _sha256_md5_file(dst)
        if sz != item["size_bytes"] or md5 != item["md5"] or sha != item["sha256"]:
            raise RuntimeError(f"CTTC archive mismatch for {fn}: sz={sz}, md5={md5}, sha={sha}")
        with tarfile.open(dst, "r:gz") as tf:
            tf.extractall(target)
        for rel, meta in item["extracted_files"].items():
            ep = (src_dir / rel) if item["role"] == "tx_source_bits" else (data_dir / item["capture_id"] / rel)
            esz, _, esha = _sha256_md5_file(ep)
            if esz != meta["size_bytes"] or esha != meta["sha256"]:
                raise RuntimeError(f"CTTC extracted mismatch for {ep}: sz={esz}, sha={esha}")
        print(f"[CTTC] verified {fn} ({sz} bytes)", flush=True)
    return work_dir


def load_pluto_arrays(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load `(tx_c64 (N, 126), rx_c64 (N, 7560))` from HDF5 v7.3 or MAT v5/v7."""

    def _to_c64(arr: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(arr):
            return arr.astype(np.complex64, copy=False)
        if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
            return (arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)).astype(np.complex64)
        raise ValueError(f"Unsupported MATLAB complex dtype: {arr.dtype}")

    with mat_path.open("rb") as mf:
        magic = mf.read(8)

    if magic == b"\x89HDF\r\n\x1a\n":
        import h5py

        with h5py.File(mat_path, "r") as f:
            tx_ds = f["Tx_data"]
            rx_ds = f["Rx_data"]
            if tx_ds.shape[0] == rx_ds.shape[0]:
                tx = np.ascontiguousarray(_to_c64(tx_ds[:]))
                rx = np.ascontiguousarray(_to_c64(rx_ds[:]))
            elif tx_ds.shape[1] == rx_ds.shape[1]:
                tx = np.ascontiguousarray(_to_c64(tx_ds[:]).T)
                rx = np.ascontiguousarray(_to_c64(rx_ds[:]).T)
            else:
                raise RuntimeError(f"Pluto HDF5 shape mismatch: Tx={tx_ds.shape}, Rx={rx_ds.shape}")
    else:
        from scipy.io import loadmat

        mat = loadmat(str(mat_path), variable_names=["Tx_data", "Rx_data"])
        tx_raw = mat.pop("Tx_data")
        rx_raw = mat.pop("Rx_data")
        del mat
        if tx_raw.shape[0] == rx_raw.shape[0]:
            tx = np.ascontiguousarray(_to_c64(tx_raw))
            rx = np.ascontiguousarray(_to_c64(rx_raw))
        elif tx_raw.shape[1] == rx_raw.shape[1]:
            tx = np.ascontiguousarray(_to_c64(tx_raw).T)
            rx = np.ascontiguousarray(_to_c64(rx_raw).T)
        else:
            raise RuntimeError(f"Pluto MAT v5/v7 shape mismatch: Tx={tx_raw.shape}, Rx={rx_raw.shape}")
        del tx_raw, rx_raw

    assert tx.shape[0] == rx.shape[0] and tx.shape[1] == PLUTO_N_SYM and rx.shape[1] == PLUTO_RX_SAMP, (
        f"Unexpected Pluto shapes: tx={tx.shape}, rx={rx.shape}"
    )
    return tx, rx


# =============================================================================
# 2. Waveform Primitives, R3 Dual-Cache Encoding & M3 Per-Row Pool
# =============================================================================

def _tx_to_c4_idx(syms: np.ndarray) -> np.ndarray:
    """Map hardware QPSK symbols (`e^{i pi/4} * ROOTS`) into canonical C4 indices `0..3`."""
    return sector_of_complex(np.asarray(syms, dtype=np.complex128) * np.exp(-1j * math.pi / 4.0)).astype(np.int64)


def _catmull_rom_sample(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = len(x)
    i0 = np.floor(pos).astype(np.int64)
    t = pos - i0
    xm1 = x[np.clip(i0 - 1, 0, n - 1)]
    x0 = x[np.clip(i0, 0, n - 1)]
    x1 = x[np.clip(i0 + 1, 0, n - 1)]
    x2 = x[np.clip(i0 + 2, 0, n - 1)]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * x0)
        + (-xm1 + x1) * t
        + (2.0 * xm1 - 5.0 * x0 + 4.0 * x1 - x2) * t2
        + (-xm1 + 3.0 * x0 - 3.0 * x1 + x2) * t3
    )


def _r3_whitened_lift(t_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct SNR-preserving R3 triplet `Y_R3 = e * T + B * eta`, `eta ~ CN_2(0, V0 I)`."""
    n = len(t_norm)
    eta = rng.normal(0.0, math.sqrt(0.5 * V0_NOM), size=(n, 2)) + 1j * rng.normal(
        0.0, math.sqrt(0.5 * V0_NOM), size=(n, 2)
    )
    return t_norm[:, None] * _E_DIR[None, :] + eta @ _B_PERP.T


def _encode_dual_pair(
    t1: np.ndarray,
    res1: np.ndarray,
    v1: np.ndarray,
    t2: np.ndarray,
    res2: np.ndarray,
    v2: np.ndarray,
    encoder,
    anc_rng: np.random.Generator,
    avail: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode `(T1, T2)` pairs into `(s17_codes (N, 2), words_c16 (N, 2))` with bit-exact prefix."""
    y3_1 = _r3_whitened_lift(t1, anc_rng)
    y3_2 = _r3_whitened_lift(t2, anc_rng)
    s1 = np.asarray(encoder.encode(y3_1), dtype=np.int64)
    s2 = np.asarray(encoder.encode(y3_2), dtype=np.int64)
    s1 = np.where(avail[:, 0], s1, 0)
    s2 = np.where(avail[:, 1], s2, 0)
    w1 = encode_c16(s1, t1, res1, np.maximum(v1, 1e-12), available=avail[:, 0])
    w2 = encode_c16(s2, t2, res2, np.maximum(v2, 1e-12), available=avail[:, 1])
    s17_codes = np.stack([s1, s2], axis=-1).astype(np.int64)
    words_c16 = np.stack([w1, w2], axis=-1).astype(np.int64)
    if np.any(avail):
        assert np.array_equal((words_c16[avail] >> 12), s17_codes[avail]), "Bit-exact M5 prefix violated!"
    return s17_codes, words_c16


def _build_per_row_m3_ref_summary(
    records: Sequence[Dict[str, Any]],
    n_rows: int,
    omega_vec: np.ndarray,
    sigma_phi_vec: np.ndarray,
    rate_var_vec: np.ndarray,
) -> np.ndarray:
    """Vectorized M3 `RefSufficientPool` summary immutable to duplicates, distractors, and reorder."""
    seen_ids = set()
    accepted: List[Dict[str, Any]] = []
    for rec in records:
        aid = int(rec["acquisition_id"])
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        if not bool(rec["shares_channel"]):
            continue
        accepted.append(rec)

    accepted.sort(key=lambda r: (float(r["dt_mean"]), int(r["acquisition_id"])))
    summary = np.zeros((n_rows, 29), dtype=np.float32)
    if not accepted:
        summary[:, 0:8] = 0.25
        return summary

    roots_np = np.asarray(ROOTS, dtype=np.complex128)
    # Accumulate exact per-row log likelihood over G in C4 across deduplicated valid pilots
    log_g = np.zeros((n_rows, 4), dtype=np.float64)
    feat_stack: List[np.ndarray] = []
    om = np.asarray(omega_vec, dtype=np.float64)
    sp2 = np.asarray(sigma_phi_vec, dtype=np.float64) ** 2
    rv = np.asarray(rate_var_vec, dtype=np.float64)

    for rec in accepted:
        t_r = np.asarray(rec["t_norm"], dtype=np.complex128)  # (N,)
        v_r = np.maximum(np.asarray(rec["v_norm"], dtype=np.float64), 1e-12)  # (N,)
        dt_r = np.asarray(rec["dt"], dtype=np.float64)  # t_ref - t_w1 (N,)
        snr_r = np.clip(1.0 / v_r, 0.05, 50.0)

        mean_phi = om * dt_r
        var_phi = np.abs(dt_r) * sp2 + (dt_r ** 2) * rv
        damping = np.exp(-0.5 * var_phi)
        dir_r = roots_np[None, :] * (damping * np.exp(1j * mean_phi))[:, None]  # (N, 4)
        log_g += (2.0 * snr_r)[:, None] * np.real(t_r[:, None] * np.conj(dir_r))

        t_derot = t_r * np.exp(-1j * om * dt_r)
        f_r = np.stack(
            [
                np.real(t_derot),
                np.imag(t_derot),
                np.abs(t_derot),
                dt_r,
                snr_r / 10.0,
            ],
            axis=-1,
        ).astype(np.float32)
        feat_stack.append(f_r)

    log_g -= np.max(log_g, axis=1, keepdims=True)
    p_g = np.exp(log_g)
    p_g /= np.sort(p_g, axis=1).sum(axis=1, keepdims=True)

    feat_arr = np.stack(feat_stack, axis=0)  # (K, N, 5)
    summary[:, 0:4] = p_g.astype(np.float32)
    summary[:, 4:8] = p_g.astype(np.float32)
    summary[:, 8] = 1.0
    summary[:, 9] = min(1.0, len(accepted) / 4.0)
    summary[:, 10:15] = feat_arr.mean(axis=0)
    summary[:, 15] = feat_arr[:, :, 2].max(axis=0)
    summary[:, 16] = float(len(accepted))
    return summary


# =============================================================================
# 3. Strictly Causal Pluto Frontend (`SYNC_SLICE [0:13]` ONLY) + Stress
# =============================================================================

def _build_rc_noise_filter(sps: int = PLUTO_SPS, beta: float = 0.35, span_syms: int = 6) -> np.ndarray:
    n_taps = span_syms * sps + 1
    t = (np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0) / float(sps)
    h = np.zeros_like(t)
    for idx, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[idx] = 1.0 - beta + 4.0 * beta / math.pi
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-10:
            h[idx] = (beta / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
            )
        else:
            num = math.sin(math.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * math.cos(math.pi * ti * (1.0 + beta))
            den = math.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[idx] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


_PLUTO_RRC_KERNEL = _build_rc_noise_filter()
PLUTO_STRESS_RECIPES: Tuple[Dict[str, Any], ...] = (
    {"name": "awgn_snr10db", "target_snr_db": 10.0, "cfo_range": None, "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261131},
    {"name": "awgn_snr6db",  "target_snr_db": 6.0,  "cfo_range": None, "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261132},
    {"name": "awgn_snr2db",  "target_snr_db": 2.0,  "cfo_range": None, "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261133},
    {"name": "awgn_snr_m2db","target_snr_db": -2.0, "cfo_range": None, "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261134},
    {"name": "cfo_0p015_raw","target_snr_db": None, "cfo_range": (0.012, 0.018), "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261141},
    {"name": "cfo_0p020_snr6db", "target_snr_db": 6.0, "cfo_range": (0.015, 0.025), "jump_prob": 0.0, "rw_std": 0.0, "seed": 20261142},
    {"name": "phase_jump_snr6db", "target_snr_db": 6.0, "cfo_range": None, "jump_prob": 0.25, "rw_std": 0.03, "seed": 20261151},
)


def _batched_profile_ols(
    x_known: np.ndarray,
    y_sync: np.ndarray,
    t_sync: np.ndarray,
    w_grid: np.ndarray,
) -> Tuple[float, complex, complex, float]:
    """Exact closed-form Frisch-Waugh-Lovell profile likelihood over `(h, d, omega)`."""
    s_rot = x_known[None, :] * np.exp(1j * w_grid[:, None] * t_sync[None, :])  # (W, K)
    s_mean = s_rot.mean(axis=1, keepdims=True)
    s_c = s_rot - s_mean
    s_en = np.maximum(np.sum(s_c.real ** 2 + s_c.imag ** 2, axis=1), 1e-30)
    y_mean = complex(np.mean(y_sync))
    y_c = y_sync - y_mean
    y_en = float(np.sum(y_c.real ** 2 + y_c.imag ** 2))
    ip = np.conj(s_c) @ y_c  # (W,)
    sse = np.maximum(0.0, y_en - (ip.real ** 2 + ip.imag ** 2) / s_en)
    best_i = int(np.argmin(sse))
    h_hat = complex(ip[best_i] / s_en[best_i])
    d_hat = complex(y_mean - h_hat * s_mean[best_i, 0])
    return float(w_grid[best_i]), h_hat, d_hat, float(sse[best_i])


def _pluto_causal_sync_and_fit(
    r_wav: np.ndarray,
    a_floor: float = 1e-4,
    allow_coarse_cfo: bool = True,
) -> Dict[str, Any]:
    """Synchronize Copy 1 and fit initial nuisance state strictly from `SYNC_SLICE [0:13]`."""
    r = np.asarray(r_wav, dtype=np.complex128)
    hdr13 = PLUTO_SYNC_HDR_13
    hdr13_conj = np.conj(hdr13)
    hdr13_en = float(np.sum(np.abs(hdr13) ** 2))
    idx_13 = np.arange(13, dtype=np.int64) * PLUTO_SPS

    # Search integer offset `s` across Copy 0/1 border `[0, 2520)` using ONLY 13-tap Barker correlation.
    # Disambiguate `[0:13]` from `[13:26]` cleanly using the preceding 13-sym slot `[s - 260]`
    # (modulated mod 2520), so Copy 2 and Copy 1's `[14:126]` are NEVER required!
    s_grid = np.arange(PLUTO_FRAME_SAMP, dtype=np.int64)
    pos_cur = s_grid[:, None] + idx_13[None, :]  # max 2519 + 240 = 2759 < 7560
    seg_cur = r[pos_cur]  # (2520, 13)
    pos_pre = (s_grid[:, None] - 13 * PLUTO_SPS + idx_13[None, :]) % PLUTO_FRAME_SAMP
    seg_pre = r[pos_pre]  # (2520, 13)

    if allow_coarse_cfo:
        # Handle up to +-0.03 rad/sym intra-sync CFO via small omega grid
        w_coarse = np.linspace(-0.025, 0.025, 11)
        rot_13 = np.exp(-1j * w_coarse[:, None] * np.arange(13, dtype=np.float64)[None, :])
        en_cur = np.sum(seg_cur.real ** 2 + seg_cur.imag ** 2, axis=1) * hdr13_en + 1e-30
        en_pre = np.sum(seg_pre.real ** 2 + seg_pre.imag ** 2, axis=1) * hdr13_en + 1e-30
        c_cur = np.max(np.abs((seg_cur * hdr13_conj[None, :]) @ rot_13.T), axis=1) / np.sqrt(en_cur)
        c_pre = np.max(np.abs((seg_pre * hdr13_conj[None, :]) @ rot_13.T), axis=1) / np.sqrt(en_pre)
    else:
        c_cur = np.abs(seg_cur @ hdr13_conj) / np.sqrt(
            np.sum(seg_cur.real ** 2 + seg_cur.imag ** 2, axis=1) * hdr13_en + 1e-30
        )
        c_pre = np.abs(seg_pre @ hdr13_conj) / np.sqrt(
            np.sum(seg_pre.real ** 2 + seg_pre.imag ** 2, axis=1) * hdr13_en + 1e-30
        )

    score = c_cur - 0.75 * c_pre
    s_int = int(np.argmax(score))
    pk = float(score[s_int])
    yl = float(score[(s_int - 1) % PLUTO_FRAME_SAMP])
    yr = float(score[(s_int + 1) % PLUTO_FRAME_SAMP])
    denom = yl - 2.0 * pk + yr
    frac = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5)) if denom < -1e-12 else 0.0
    s_exact = float(s_int) + frac

    # Extract STRICTLY causal 13-sym sync samples via Catmull-Rom cubic interpolation
    t_sync = np.arange(13, dtype=np.float64)
    y_sync = _catmull_rom_sample(r, s_exact + t_sync * float(PLUTO_SPS))

    w_grid = np.linspace(-0.04, 0.04, 81) if allow_coarse_cfo else np.array([0.0], dtype=np.float64)
    best_w, best_h, best_d, best_sse = _batched_profile_ols(PLUTO_SYNC_HDR_13, y_sync, t_sync, w_grid)
    if allow_coarse_cfo:
        fine_grid = np.linspace(best_w - 0.002, best_w + 0.002, 21)
        best_w, best_h, best_d, best_sse = _batched_profile_ols(PLUTO_SYNC_HDR_13, y_sync, t_sync, fine_grid)

    dof = max(1, 13 - 3)
    v_sync = max(best_sse / float(dof), 1e-8)
    a_sync = max(abs(best_h), a_floor)
    pos_3 = s_exact + t_sync[:, None] * float(PLUTO_SPS) + np.array([-2.0, 0.0, 2.0])[None, :]
    sync_3 = (_catmull_rom_sample(r, pos_3.ravel()).reshape(13, 3) - best_d) * np.exp(-1j * math.pi / 4.0) / a_sync
    raw_proj = sync_3 / (PLUTO_SYNC_HDR_13[:, None] * np.exp(-1j * math.pi / 4.0) * np.exp(1j * best_w * t_sync)[:, None])
    q_steer = raw_proj.mean(axis=0)
    q_norm = np.linalg.norm(q_steer)
    q_steer = (q_steer / q_norm) if q_norm > 1e-12 else _E_DIR.copy()

    return {
        "r": r,
        "s_int": s_int,
        "s_exact": s_exact,
        "c13_peak": float(c_cur[s_int]),
        "omega_hat": best_w,
        "h_hat": best_h,
        "d_hat": best_d,
        "a_eff": a_sync,
        "v_norm": v_sync / (a_sync ** 2),
        "sync_sig_pow": float(np.mean(y_sync.real ** 2 + y_sync.imag ** 2)),
        "q_steer": q_steer,
    }


def _pluto_sample_normalized_window(
    r: np.ndarray,
    s_exact: float,
    sym_idx: np.ndarray,
    d_hat: complex,
    a_eff: float,
    q_steer: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample fractional `(-2, 0, +2)` sub-symbol stencil and project onto `(T_norm, E_phys)`."""
    k = np.asarray(sym_idx, dtype=np.float64)
    offsets = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    pos = s_exact + k[:, None] * float(PLUTO_SPS) + offsets[None, :]
    samp3 = _catmull_rom_sample(r, pos.ravel()).reshape(len(k), 3)
    samp3_norm = (samp3 - d_hat) * np.exp(-1j * math.pi / 4.0) / a_eff
    t_center = samp3_norm[:, 1]
    t_steer = samp3_norm @ np.conj(q_steer)
    resid = np.sum(np.abs(samp3_norm - t_steer[:, None] * q_steer[None, :]) ** 2, axis=-1)
    return t_center, resid


def _inject_pluto_stress_causal(
    r_raw: np.ndarray,
    spec: Dict[str, Any],
    sync_pow: float,
    rng: np.random.Generator,
    cum_jump: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Causal waveform perturbation calibrated strictly from `sync_pow` (`SYNC_SLICE` only)."""
    r = np.asarray(r_raw, dtype=np.complex128).copy()
    n = len(r)
    n_idx = np.arange(n, dtype=np.float64)
    new_cum = cum_jump
    if spec["jump_prob"] > 0.0:
        if rng.random() < spec["jump_prob"]:
            angs = (math.pi / 2, -math.pi / 2, math.pi, math.pi / 3, -math.pi / 3)
            new_cum = float((cum_jump + angs[int(rng.integers(0, len(angs)))] + math.pi) % (2.0 * math.pi) - math.pi)
        if abs(new_cum) > 0.0:
            r *= np.exp(1j * new_cum)
    if spec["rw_std"] > 0.0:
        n_syms_win = n // PLUTO_SPS + 2
        dphi = rng.normal(0.0, spec["rw_std"], size=n_syms_win)
        dphi[0] = 0.0
        phi_samp = np.interp(n_idx / float(PLUTO_SPS), np.arange(n_syms_win, dtype=np.float64), np.cumsum(dphi))
        r *= np.exp(1j * phi_samp)
    if spec["cfo_range"] is not None:
        lo, hi = spec["cfo_range"]
        sgn = 1.0 if rng.random() >= 0.5 else -1.0
        w_sym = float(sgn * rng.uniform(lo, hi))
        r *= np.exp(1j * (w_sym / float(PLUTO_SPS)) * n_idx)
    if spec["target_snr_db"] is not None:
        snr_lin = 10.0 ** (float(spec["target_snr_db"]) / 10.0)
        nvar = max(sync_pow, 1e-8) / snr_lin
        w_re = rng.normal(0.0, math.sqrt(0.5 * nvar), size=n + len(_PLUTO_RRC_KERNEL) - 1)
        w_im = rng.normal(0.0, math.sqrt(0.5 * nvar), size=n + len(_PLUTO_RRC_KERNEL) - 1)
        w = np.convolve(w_re + 1j * w_im, _PLUTO_RRC_KERNEL, mode="valid")
        r = r + w[:n]
    return r, new_cum


def extract_pluto_causal_dual_cache(
    tx_c64: np.ndarray,
    rx_c64: np.ndarray,
    encoder,
    splits: Sequence[str] = ("train", "dev", "test"),
    include_stress: bool = True,
    seed: int = 202611,
    max_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    """Run strictly causal Pluto M4 extractor producing 4-bit s17 & 16-bit `m5-c16-v1-le` shards."""
    rng_anc = np.random.default_rng(seed + 7101)
    splits_meta = PLUTO_MANIFEST["splits"]

    # Step 1: Calibrate causal gain floor from warmup frames [0:200) ONLY
    w0, w1 = splits_meta["train"]["warmup_start"], splits_meta["train"]["eval_start"]
    warm_amps = []
    for f in range(w0, w1, max(1, (w1 - w0) // 40)):
        if f >= len(rx_c64):
            break
        st_f = _pluto_causal_sync_and_fit(rx_c64[f], a_floor=1e-6, allow_coarse_cfo=False)
        warm_amps.append(abs(st_f["h_hat"]))
    a_floor = float(max(1e-5, 0.25 * float(np.median(warm_amps)))) if warm_amps else 1e-4

    def _run_frames(frame_ids: np.ndarray, rx_list: Sequence[np.ndarray], split_tag: str) -> Dict[str, Any]:
        n = len(frame_ids)
        k1_arr = np.array([30 + (int(f) % 67) for f in frame_ids], dtype=np.int64)
        h_arr = np.array([1 if ((int(f) + (int(f) // 16)) % 2 == 0) else 8 for f in frame_ids], dtype=np.int64)
        k2_arr = k1_arr + h_arr
        ell_arr = PLUTO_N_SYM - h_arr

        t1 = np.empty(n, dtype=np.complex128)
        res1 = np.empty(n, dtype=np.float64)
        t2 = np.empty(n, dtype=np.complex128)
        res2 = np.empty(n, dtype=np.float64)
        tp15 = np.empty(n, dtype=np.complex128)
        tp18 = np.empty(n, dtype=np.complex128)
        t3_copy2 = np.empty(n, dtype=np.complex128)
        omega_hat = np.empty(n, dtype=np.float64)
        v_norm = np.empty(n, dtype=np.float64)
        s_start = np.empty(n, dtype=np.int64)
        sync_ok = np.empty(n, dtype=bool)
        sync_pow = np.empty(n, dtype=np.float64)

        lbl_q1 = np.empty(n, dtype=np.int64)
        lbl_q2 = np.empty(n, dtype=np.int64)
        lbl_q3 = np.empty(n, dtype=np.int64)

        # Normalize TX scale strictly from known header amplitude
        tx_scale = float(np.median(np.abs(tx_c64[frame_ids[0], :13])))
        r15_c4 = _tx_to_c4_idx(tx_c64[frame_ids, 15])
        r18_c4 = _tx_to_c4_idx(tx_c64[frame_ids, 18])

        for i, f in enumerate(frame_ids):
            fit = _pluto_causal_sync_and_fit(rx_list[i], a_floor=a_floor, allow_coarse_cfo=True)
            s_exact = fit["s_exact"]
            a_eff = fit["a_eff"] * max(tx_scale, 1e-12)
            d_hat = fit["d_hat"]
            omega_hat[i] = fit["omega_hat"]
            v_norm[i] = fit["v_norm"] * (max(tx_scale, 1e-12) ** 2)
            s_start[i] = fit["s_int"]
            sync_ok[i] = bool(fit["c13_peak"] >= 0.45)
            sync_pow[i] = fit["sync_sig_pow"]

            # Window 1 & Window 2 on Copy 1
            tc, rc = _pluto_sample_normalized_window(
                fit["r"], s_exact, np.array([k1_arr[i], k2_arr[i]], dtype=np.float64), d_hat, a_eff, fit["q_steer"]
            )
            t1[i], res1[i] = tc[0], rc[0]
            t2[i], res2[i] = tc[1], rc[1]

            # Revealed references at indices 15, 18 (dividing out known symbol C4 root!)
            tr, _ = _pluto_sample_normalized_window(
                fit["r"], s_exact, np.array([15.0, 18.0], dtype=np.float64), d_hat, a_eff, fit["q_steer"]
            )
            tp15[i] = tr[0] * np.conj(ROOTS[r15_c4[i]])
            tp18[i] = tr[1] * np.conj(ROOTS[r18_c4[i]])

            # Scorer-vault Copy-2 Q3 retransmission target at k1 + 126
            t3c, _ = _pluto_sample_normalized_window(
                fit["r"], s_exact, np.array([float(k1_arr[i] + PLUTO_N_SYM)], dtype=np.float64), d_hat, a_eff, fit["q_steer"]
            )
            t3_copy2[i] = t3c[0]

            x1_idx = int(_tx_to_c4_idx(tx_c64[f, k1_arr[i]]))
            x2_idx = int(_tx_to_c4_idx(tx_c64[f, k2_arr[i]]))
            lbl_q2[i] = x1_idx
            lbl_q1[i] = (x2_idx - x1_idx) % 4
            # Q3 physical sector of raw receiver observation on Copy 2
            lbl_q3[i] = int(sector_of_complex(np.array([t3_copy2[i]]))[0])

        avail = np.stack([sync_ok, sync_ok], axis=-1)
        s17_codes, words_c16 = _encode_dual_pair(t1, res1, v_norm, t2, res2, v_norm, encoder, rng_anc, avail)

        # Conservative Brownian diffusion + omega uncertainty calibrated from causal sync residual
        sigma_phi = np.sqrt(np.clip(v_norm, 1e-4, 0.25) / 13.0)
        rate_var = np.clip(v_norm, 1e-5, 0.05) / 140.0
        dt15 = 15.0 - k1_arr.astype(np.float64)
        dt18 = 18.0 - k1_arr.astype(np.float64)
        mean_tp = 16.5 - k1_arr.astype(np.float64)

        phase_slots = np.stack(
            [
                np.cos(omega_hat * h_arr),
                np.sin(omega_hat * h_arr),
                np.cos(omega_hat * (h_arr + ell_arr)),
                np.sin(omega_hat * (h_arr + ell_arr)),
                np.cos(omega_hat * mean_tp),
                np.sin(omega_hat * mean_tp),
            ],
            axis=-1,
        ).astype(np.float32)
        snr_db = np.clip(-10.0 * np.log10(np.maximum(v_norm, 1e-6)), -10.0, 26.0)
        meta_scalars = np.stack(
            [
                h_arr.astype(np.float32),
                ell_arr.astype(np.float32),
                mean_tp.astype(np.float32),
                omega_hat.astype(np.float32),
                sigma_phi.astype(np.float32),
                snr_db.astype(np.float32),
                rate_var.astype(np.float32),
            ],
            axis=-1,
        )

        # Build M3 RefSufficientPool variants
        rec_15 = {"acquisition_id": 15, "shares_channel": True, "t_norm": tp15, "v_norm": v_norm, "dt": dt15, "dt_mean": float(np.mean(dt15))}
        rec_18 = {"acquisition_id": 18, "shares_channel": True, "t_norm": tp18, "v_norm": v_norm, "dt": dt18, "dt_mean": float(np.mean(dt18))}
        indep_rng = np.random.default_rng(seed + 888 + len(frame_ids))
        rec_indep = {
            "acquisition_id": 999,
            "shares_channel": False,
            "t_norm": np.exp(1j * indep_rng.uniform(-math.pi, math.pi, size=n)),
            "v_norm": v_norm,
            "dt": dt15,
            "dt_mean": float(np.mean(dt15)),
        }
        ref_none = _build_per_row_m3_ref_summary((), n, omega_hat, sigma_phi, rate_var)
        ref_one = _build_per_row_m3_ref_summary([rec_15], n, omega_hat, sigma_phi, rate_var)
        ref_two = _build_per_row_m3_ref_summary([rec_15, rec_18], n, omega_hat, sigma_phi, rate_var)
        ref_dup = _build_per_row_m3_ref_summary([rec_15, rec_18, rec_15, rec_18], n, omega_hat, sigma_phi, rate_var)
        ref_ind = _build_per_row_m3_ref_summary([rec_18, rec_indep, rec_15], n, omega_hat, sigma_phi, rate_var)
        assert float(np.max(np.abs(ref_dup - ref_two))) == 0.0, "Pluto M3 duplicate invariance failed!"
        assert float(np.max(np.abs(ref_ind - ref_two))) == 0.0, "Pluto M3 distractor/reorder invariance failed!"

        clusters = (
            ((frame_ids - int(PLUTO_MANIFEST["splits"]["test"]["eval_start"])) // 140)
            if (split_tag == "test" or split_tag.startswith("stress_"))
            else np.zeros(n, dtype=np.int64)
        )
        return {
            "split": split_tag,
            "frame_ids": frame_ids,
            "cluster_ids": clusters.astype(np.int64),
            "s17_codes": s17_codes,
            "words_c16": words_c16,
            "avail": avail,
            "phase_slots": phase_slots,
            "meta_scalars": meta_scalars,
            "discrete_d": np.zeros(n, dtype=np.float32),
            "ref_none": ref_none,
            "ref_one": ref_one,
            "ref_two": ref_two,
            "ref_dup": ref_dup,
            "ref_indep": ref_ind,
            "labels_q1": lbl_q1,
            "labels_q2": lbl_q2,
            "labels_q3": lbl_q3,
            "t1_cont": t1,
            "t2_cont": t2,
            "v_norm": v_norm,
            "h_arr": h_arr,
            "ell_arr": ell_arr,
            "sample_start": s_start,
            "sync_ok": sync_ok,
            "sync_pow": sync_pow,
        }

    out: Dict[str, Any] = {"dataset": "pluto", "wire_id": WIRE_ID, "a_floor": a_floor, "splits": {}}
    for sp_name in splits:
        sp = splits_meta[sp_name]
        f_end = sp["eval_end"] if max_per_split is None else min(sp["eval_end"], sp["eval_start"] + int(max_per_split))
        f_ids = np.arange(sp["eval_start"], f_end, dtype=np.int64)
        out["splits"][sp_name] = _run_frames(f_ids, [rx_c64[f] for f in f_ids], sp_name)

    if include_stress:
        test_sp = splits_meta["test"]
        test_f_end = test_sp["eval_end"] if max_per_split is None else min(test_sp["eval_end"], len(rx_c64), test_sp["eval_start"] + int(max_per_split))
        step_f = 16 if max_per_split is None else max(1, (test_f_end - test_sp["eval_start"]) // max(2, min(14, max_per_split // 16)))
        stress_f = np.arange(test_sp["eval_start"], min(test_f_end, test_sp["eval_start"] + 2240), step_f, dtype=np.int64)
        if max_per_split is not None and len(stress_f) > max(4, max_per_split // 16):
            stress_f = stress_f[: max(4, max_per_split // 16)]
        stress_shards: List[Dict[str, Any]] = []
        raw_pows = out["splits"]["test"]["sync_pow"][np.clip(stress_f - test_sp["eval_start"], 0, len(out["splits"]["test"]["sync_pow"]) - 1)]
        for spec in PLUTO_STRESS_RECIPES:
            rng_s = np.random.default_rng(spec["seed"])
            cum_j = 0.0
            rx_pert_list = []
            for j, f in enumerate(stress_f):
                r_p, cum_j = _inject_pluto_stress_causal(rx_c64[f], spec, float(raw_pows[j]), rng_s, cum_jump=cum_j)
                rx_pert_list.append(r_p)
            sh = _run_frames(stress_f, rx_pert_list, f"stress_{spec['name']}")
            stress_shards.append(sh)
        out["stress"] = _concat_shards(stress_shards, split_name="pluto_stress")
    return out


# =============================================================================
# 4. Strictly Causal CTTC Frontend (`SYNC_IDX [0:6]` ONLY + Capture Commissioning)
# =============================================================================

def _cttc_causal_pkt_fit(
    y_iq_4x: np.ndarray,
    x_sync6: np.ndarray,
    bg_noise_var: float,
    cfo_prior_hz: float,
    a_floor: float = 1e-6,
) -> Dict[str, Any]:
    """Synchronize fractional timing and estimate causal state strictly from `SYNC_IDX = [0:6]`."""
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    t_rx = np.arange(len(y_c), dtype=np.float64)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)

    sync_f = CTTC_SYNC_IDX.astype(np.float64)  # [0..5]
    # Narrow grid around causal capture warmup/commissioning CFO (+-600 Hz)
    f_grid = np.linspace(cfo_prior_hz - 600.0, cfo_prior_hz + 600.0, 25)
    rot_mat = np.exp(-2j * math.pi * f_grid[:, None] * sync_f[None, :] / CTTC_FS_SYM)
    t_grid = np.linspace(-1.0, 1.0, 9)

    best_score = -1.0
    best_t = 0.0
    best_f = float(cfo_prior_hz)
    for dt in t_grid:
        idx_k = (CTTC_PAD_SYMS * CTTC_SPS) + dt + CTTC_SPS * sync_f
        yk = cs_re(idx_k) + 1j * cs_im(idx_k)
        zk = yk * np.conj(x_sync6)
        sc = np.abs(rot_mat @ zk)
        fi = int(np.argmax(sc))
        if float(sc[fi]) > best_score:
            best_score = float(sc[fi])
            best_t = float(dt)
            best_f = float(f_grid[fi])

    # Fine timing + profile likelihood on SYNC_IDX [0:6] ONLY
    step_t = 0.25
    rot0 = np.exp(-2j * math.pi * best_f * sync_f / CTTC_FS_SYM)
    s_tri = []
    for dt in (best_t - step_t, best_t, best_t + step_t):
        idx_k = (CTTC_PAD_SYMS * CTTC_SPS) + dt + CTTC_SPS * sync_f
        yk = cs_re(idx_k) + 1j * cs_im(idx_k)
        s_tri.append(float(abs(np.dot(yk * np.conj(x_sync6), rot0))))
    denom = s_tri[0] - 2.0 * s_tri[1] + s_tri[2]
    if denom < -1e-30:
        best_t = float(best_t + np.clip(0.5 * (s_tri[0] - s_tri[2]) / denom, -0.5, 0.5) * step_t)

    idx_sync = (CTTC_PAD_SYMS * CTTC_SPS) + best_t + CTTC_SPS * sync_f
    y_sync = cs_re(idx_sync) + 1j * cs_im(idx_sync)

    omega_cand = 2.0 * math.pi * np.linspace(best_f - 150.0, best_f + 150.0, 13) / CTTC_FS_SYM
    best_w, best_h, best_d, best_sse = _batched_profile_ols(x_sync6, y_sync, sync_f, omega_cand)

    v_resid = best_sse / 4.0
    v_phys = max(bg_noise_var, 0.35 * v_resid + 0.65 * bg_noise_var, 1e-12)
    a_eff = max(abs(best_h), a_floor)

    def _sample_stencil(sym_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        k_arr = np.asarray(sym_indices, dtype=np.float64)
        offs = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        rx_pos = (CTTC_PAD_SYMS * CTTC_SPS) + best_t + k_arr[:, None] * float(CTTC_SPS) + offs[None, :]
        vals = (cs_re(rx_pos.ravel()) + 1j * cs_im(rx_pos.ravel())).reshape(len(k_arr), 3)
        norm3 = (vals - best_d) * np.exp(-1j * math.pi / 4.0) / a_eff
        t_c = norm3[:, 1]
        # 3-tap RRC steering approximation along (-1, 0, +1) sample offsets -> E orthogonality residual
        res_e = np.sum(np.abs(norm3 - t_c[:, None] * np.array([[0.35, 1.0, 0.35]]) / math.sqrt(1.245)) ** 2, axis=-1)
        return t_c, res_e

    return {
        "best_t": best_t,
        "omega_hat": best_w,
        "h_hat": best_h,
        "d_hat": best_d,
        "a_eff": a_eff,
        "v_norm": v_phys / (a_eff ** 2),
        "sync_sig_pow": float(np.mean(y_sync.real ** 2 + y_sync.imag ** 2)),
        "sync_corr": float(abs(best_h) / math.sqrt(max(float(np.mean(np.abs(y_sync) ** 2)), 1e-30))),
        "sample_stencil": _sample_stencil,
    }


def _estimate_cttc_commissioning_cfo(
    x_mm: np.ndarray,
    sym_ref: np.ndarray,
    anchor: Dict[str, Any],
) -> Tuple[float, float]:
    """Estimate coarse capture CFO strictly from disjoint CTTC training-ref/commissioning region `[496256, 498304)`."""
    rx_loop0 = float(anchor["anchor_rx_pos"]) - float(anchor["anchor_src_sym"]) * float(anchor["sps_eff"])
    while rx_loop0 < 2_900_000.0:
        rx_loop0 += float(CTTC_SRC_N_SYMS) * float(anchor["sps_eff"])
    src_start = 496_256
    seg_syms = 64
    rx_exact = rx_loop0 + float(src_start) * float(anchor["sps_eff"])
    w0 = int(math.floor(rx_exact)) - CTTC_PAD_SYMS * CTTC_SPS
    w1 = w0 + (seg_syms + 2 * CTTC_PAD_SYMS) * CTTC_SPS
    if w0 < 0 or w1 >= len(x_mm):
        return 0.0, 1e-5
    seg = np.asarray(x_mm[w0:w1], dtype=np.complex128)
    frac = float(rx_exact - math.floor(rx_exact))
    t_rx = np.arange(len(seg), dtype=np.float64)
    idx_s = (CTTC_PAD_SYMS * CTTC_SPS) + frac + CTTC_SPS * np.arange(seg_syms, dtype=np.float64)
    y_sym = CubicSpline(t_rx, seg.real)(idx_s) + 1j * CubicSpline(t_rx, seg.imag)(idx_s)
    x_known = sym_ref[src_start : src_start + seg_syms]
    z = y_sym * np.conj(x_known)
    f_grid = np.linspace(-2000.0, 2000.0, 401)
    k_idx = np.arange(seg_syms, dtype=np.float64)
    sc = np.abs(np.exp(-2j * math.pi * f_grid[:, None] * k_idx[None, :] / CTTC_FS_SYM) @ z)
    best_f = float(f_grid[int(np.argmax(sc))])
    amp = float(np.max(sc) / float(seg_syms))
    return best_f, max(1e-6, 0.25 * amp)


def extract_cttc_causal_dual_cache(
    work_dir: Path,
    encoder,
    splits: Sequence[str] = ("train", "dev", "test"),
    n_loops_use: int = 8,
    include_stress: bool = True,
    seed: int = 202611,
    captures: Sequence[Dict[str, Any]] = CTTC_CAPTURE_ANCHORS,
    per_loop_override: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Run strictly causal CTTC extractor across captures (holding out `f708_snr10_QPSK`)."""
    raw_bits = np.fromfile(work_dir / "sources" / "source_QPSK.dat", dtype=np.uint8)
    b0_e, b1_e = raw_bits[0::2], raw_bits[1::2]
    sym_even = ((1.0 - 2.0 * b0_e) + 1j * (1.0 - 2.0 * b1_e)) / math.sqrt(2.0)
    b0_o, b1_o = raw_bits[1:-1:2], raw_bits[2::2]
    sym_odd = ((1.0 - 2.0 * b0_o) + 1j * (1.0 - 2.0 * b1_o)) / math.sqrt(2.0)

    rng_anc = np.random.default_rng(seed + 7201)
    per_loop_tgt = dict(per_loop_override) if per_loop_override is not None else {"train": 20, "dev": 12, "test": 20}
    pkt_rx_len = (CTTC_PKT_LEN + 2 * CTTC_PAD_SYMS) * CTTC_SPS
    splits_ranges = CTTC_MANIFEST["split_source_ranges"]

    split_buckets: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}
    q3_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []
    rng_stress = np.random.default_rng(seed + 7999)

    for cap_idx, anchor in enumerate(captures):
        cid = str(anchor["capture_id"])
        is_holdout = bool(cid == CTTC_MANIFEST["holdout_capture"])
        cap_file = work_dir / "data" / cid / "file_2.dat"
        x_mm = np.memmap(cap_file, dtype=np.complex64, mode="r")
        sym_ref = sym_odd if int(anchor["bit_alignment"]) == 1 else sym_even
        cfo_prior_hz, a_floor = _estimate_cttc_commissioning_cfo(x_mm, sym_ref, anchor)

        rx_loop0 = float(anchor["anchor_rx_pos"]) - float(anchor["anchor_src_sym"]) * float(anchor["sps_eff"])
        while rx_loop0 < 2_900_000.0:
            rx_loop0 += float(CTTC_SRC_N_SYMS) * float(anchor["sps_eff"])

        # Collect Q3 target scalars across loops for disjoint pairs (0,1), (2,3), (4,5), (6,7)
        loop_test_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for loop_id in range(n_loops_use):
            rx_base = rx_loop0 + float(loop_id * CTTC_SRC_N_SYMS) * float(anchor["sps_eff"])
            cluster_num = cap_idx * 8 + loop_id

            for sp_name in splits:
                n_cnt = per_loop_tgt[sp_name]
                lo, hi = splits_ranges[sp_name]["target"]
                stride = (hi - lo - CTTC_PKT_LEN) // n_cnt

                n = n_cnt
                t1 = np.empty(n, dtype=np.complex128)
                res1 = np.empty(n, dtype=np.float64)
                t2 = np.empty(n, dtype=np.complex128)
                res2 = np.empty(n, dtype=np.float64)
                tp126 = np.empty(n, dtype=np.complex128)
                tp127 = np.empty(n, dtype=np.complex128)
                omega_hat = np.empty(n, dtype=np.float64)
                v_norm = np.empty(n, dtype=np.float64)
                h_arr = np.empty(n, dtype=np.int64)
                k1_arr = np.empty(n, dtype=np.int64)
                lbl_q1 = np.empty(n, dtype=np.int64)
                lbl_q2 = np.empty(n, dtype=np.int64)
                sync_ok = np.empty(n, dtype=bool)

                for j in range(n):
                    src_start = lo + j * stride + 128
                    rx_exact = rx_base + float(src_start) * float(anchor["sps_eff"])
                    w0 = int(math.floor(rx_exact)) - CTTC_PAD_SYMS * CTTC_SPS
                    assert 0 <= w0 and w0 + pkt_rx_len <= len(x_mm), f"CTTC OOB capture={cid} loop={loop_id} w0={w0} len={len(x_mm)}"
                    y_iq = np.asarray(x_mm[w0 : w0 + pkt_rx_len], dtype=np.complex64)
                    x_true = sym_ref[src_start : src_start + CTTC_PKT_LEN]
                    k1 = 136 + ((17 * j + 5 * loop_id) % 48)
                    h_step = 1 if ((j + loop_id) % 2 == 0) else 8
                    k2 = k1 + h_step
                    k1_arr[j] = k1
                    h_arr[j] = h_step

                    fit = _cttc_causal_pkt_fit(y_iq, x_true[CTTC_SYNC_IDX], float(anchor["bg_noise_var"]), cfo_prior_hz, a_floor=a_floor)
                    omega_hat[j] = fit["omega_hat"]
                    v_norm[j] = fit["v_norm"]
                    sync_ok[j] = bool(fit["sync_corr"] >= 0.25)

                    tc, rc = fit["sample_stencil"]([k1, k2])
                    t1[j], res1[j] = tc[0], rc[0]
                    t2[j], res2[j] = tc[1], rc[1]

                    tr, _ = fit["sample_stencil"]([126, 127])
                    r126_c4 = int(_tx_to_c4_idx(x_true[126:127])[0])
                    r127_c4 = int(_tx_to_c4_idx(x_true[127:128])[0])
                    tp126[j] = tr[0] * np.conj(ROOTS[r126_c4])
                    tp127[j] = tr[1] * np.conj(ROOTS[r127_c4])

                    x1_i = int(_tx_to_c4_idx(x_true[k1 : k1 + 1])[0])
                    x2_i = int(_tx_to_c4_idx(x_true[k2 : k2 + 1])[0])
                    lbl_q2[j] = x1_i
                    lbl_q1[j] = (x2_i - x1_i) % 4

                    if sp_name == "test":
                        loop_test_cache[(loop_id, j)] = {
                            "t1": t1[j], "res1": res1[j], "t2": t2[j], "res2": res2[j],
                            "v_norm": v_norm[j], "omega": omega_hat[j], "h": h_step, "k1": k1,
                            "y_iq": y_iq, "x_sync": x_true[CTTC_SYNC_IDX], "bg": float(anchor["bg_noise_var"]),
                            "cfo_prior": cfo_prior_hz, "a_floor": a_floor, "tp126": tp126[j], "tp127": tp127[j],
                            "src_idx": int(loop_id // 2), "q1_lbl": lbl_q1[j], "q2_lbl": lbl_q2[j],
                            "capture_num": cap_idx, "is_holdout": is_holdout,
                            "sync_pow": fit["sync_sig_pow"], "sig1": tc[0],
                        }

                ell_arr = CTTC_SRC_N_SYMS - h_arr
                sigma_phi = np.full(n, 0.02, dtype=np.float64)
                rate_var = np.full(n, 1e-6, dtype=np.float64)
                dt126 = 126.0 - k1_arr.astype(np.float64)
                dt127 = 127.0 - k1_arr.astype(np.float64)
                mean_tp = 126.5 - k1_arr.astype(np.float64)

                avail = np.stack([sync_ok, sync_ok], axis=-1)
                s17_codes, words_c16 = _encode_dual_pair(t1, res1, v_norm, t2, res2, v_norm, encoder, rng_anc, avail)
                phase_slots = np.stack(
                    [
                        np.cos(omega_hat * h_arr),
                        np.sin(omega_hat * h_arr),
                        np.cos(omega_hat * (h_arr + ell_arr)),
                        np.sin(omega_hat * (h_arr + ell_arr)),
                        np.cos(omega_hat * mean_tp),
                        np.sin(omega_hat * mean_tp),
                    ],
                    axis=-1,
                ).astype(np.float32)
                snr_db = np.clip(-10.0 * np.log10(np.maximum(v_norm, 1e-6)), -10.0, 26.0)
                meta_scalars = np.stack(
                    [
                        h_arr.astype(np.float32),
                        ell_arr.astype(np.float32),
                        mean_tp.astype(np.float32),
                        omega_hat.astype(np.float32),
                        sigma_phi.astype(np.float32),
                        snr_db.astype(np.float32),
                        rate_var.astype(np.float32),
                    ],
                    axis=-1,
                )
                r126_rec = {"acquisition_id": 126, "shares_channel": True, "t_norm": tp126, "v_norm": v_norm, "dt": dt126, "dt_mean": float(np.mean(dt126))}
                r127_rec = {"acquisition_id": 127, "shares_channel": True, "t_norm": tp127, "v_norm": v_norm, "dt": dt127, "dt_mean": float(np.mean(dt127))}
                r_ind_rec = {
                    "acquisition_id": 999,
                    "shares_channel": False,
                    "t_norm": np.exp(1j * rng_anc.uniform(-math.pi, math.pi, size=n)),
                    "v_norm": v_norm,
                    "dt": dt126,
                    "dt_mean": float(np.mean(dt126)),
                }
                ref_none = _build_per_row_m3_ref_summary((), n, omega_hat, sigma_phi, rate_var)
                ref_one = _build_per_row_m3_ref_summary([r126_rec], n, omega_hat, sigma_phi, rate_var)
                ref_two = _build_per_row_m3_ref_summary([r126_rec, r127_rec], n, omega_hat, sigma_phi, rate_var)
                ref_dup = _build_per_row_m3_ref_summary([r126_rec, r127_rec, r126_rec, r127_rec], n, omega_hat, sigma_phi, rate_var)
                ref_ind = _build_per_row_m3_ref_summary([r127_rec, r_ind_rec, r126_rec], n, omega_hat, sigma_phi, rate_var)
                assert float(np.max(np.abs(ref_dup - ref_two))) == 0.0, "CTTC M3 duplicate invariance failed!"
                assert float(np.max(np.abs(ref_ind - ref_two))) == 0.0, "CTTC M3 distractor/reorder invariance failed!"

                split_buckets[sp_name].append(
                    {
                        "cluster_ids": np.full(n, cluster_num, dtype=np.int64),
                        "capture_ids": np.full(n, cap_idx, dtype=np.int64),
                        "is_holdout": np.full(n, is_holdout, dtype=bool),
                        "s17_codes": s17_codes,
                        "words_c16": words_c16,
                        "avail": avail,
                        "phase_slots": phase_slots,
                        "meta_scalars": meta_scalars,
                        "discrete_d": np.zeros(n, dtype=np.float32),
                        "ref_none": ref_none,
                        "ref_one": ref_one,
                        "ref_two": ref_two,
                        "ref_dup": ref_dup,
                        "ref_indep": ref_ind,
                        "labels_q1": lbl_q1,
                        "labels_q2": lbl_q2,
                        "labels_q3": np.zeros(n, dtype=np.int64),  # populated via Q3 pair shard
                        "t1_cont": t1,
                        "t2_cont": t2,
                        "v_norm": v_norm,
                        "sync_ok": sync_ok,
                    }
                )

        # Build 36 loop-pair Q3 rows & 1,080 CTTC stress rows across this capture
        if "test" in splits and n_loops_use >= 2:
            for pair_idx, (l0, l1) in enumerate(((0, 1), (2, 3), (4, 5), (6, 7))):
                if l1 >= n_loops_use:
                    continue
                q3_cluster = cap_idx * 4 + pair_idx
                for j in range(per_loop_tgt["test"]):
                    e0 = loop_test_cache[(l0, j)]
                    e1 = loop_test_cache[(l1, j)]
                    q3_lbl = int(sector_of_complex(np.array([e1["sig1"]]))[0])
                    q3_rows.append(
                        {
                            "cluster_id": q3_cluster,
                            "capture_id": cap_idx,
                            "is_holdout": is_holdout,
                            "t1": e0["t1"], "res1": e0["res1"], "t2": e0["t2"], "res2": e0["res2"],
                            "v_norm": e0["v_norm"], "omega": e0["omega"], "h": e0["h"], "k1": e0["k1"],
                            "q3_lbl": q3_lbl,
                        }
                    )

            if include_stress:
                flat_idx = 0
                for loop_id in range(n_loops_use):
                    for j in range(per_loop_tgt["test"]):
                        global_i = cap_idx * (n_loops_use * per_loop_tgt["test"]) + flat_idx
                        flat_idx += 1
                        if global_i % 4 == 3:
                            continue  # exactly 3/4 * 1440 = 1,080 rows
                        mode = global_i % 3
                        ent = loop_test_cache[(loop_id, j)]
                        y_p = np.asarray(ent["y_iq"], dtype=np.complex128).copy()
                        n_samp = np.arange(len(y_p), dtype=np.float64)
                        spow = max(float(ent["sync_pow"]), float(ent["bg"]) * 2.0)
                        if mode == 0:  # AWGN 2 dB calibrated from sync power
                            nvar = spow / (10.0 ** 0.2)
                            y_p += rng_stress.normal(0.0, math.sqrt(0.5 * nvar), len(y_p)) + 1j * rng_stress.normal(0.0, math.sqrt(0.5 * nvar), len(y_p))
                        elif mode == 1:  # Residual CFO 0.020 rad/sym
                            sgn = 1.0 if (global_i % 2 == 0) else -1.0
                            y_p *= np.exp(1j * sgn * (0.020 / float(CTTC_SPS)) * n_samp)
                        else:  # Sudden quadrant burst jump + AWGN 6 dB
                            angs = (math.pi / 2, -math.pi / 2, math.pi)
                            y_p *= np.exp(1j * angs[global_i % 3])
                            nvar = spow / (10.0 ** 0.6)
                            y_p += rng_stress.normal(0.0, math.sqrt(0.5 * nvar), len(y_p)) + 1j * rng_stress.normal(0.0, math.sqrt(0.5 * nvar), len(y_p))

                        fit_s = _cttc_causal_pkt_fit(y_p, ent["x_sync"], ent["bg"], ent["cfo_prior"], a_floor=ent["a_floor"])
                        lo_s, hi_s = splits_ranges["test"]["target"]
                        stride_s = (hi_s - lo_s - CTTC_PKT_LEN) // per_loop_tgt["test"]
                        src_s = lo_s + j * stride_s + 128
                        x_s = sym_ref[src_s : src_s + CTTC_PKT_LEN]
                        k1 = ent["k1"]
                        k2 = k1 + ent["h"]
                        tc, rc = fit_s["sample_stencil"]([k1, k2])
                        tr, _ = fit_s["sample_stencil"]([126, 127])
                        r126_c4 = int(_tx_to_c4_idx(x_s[126:127])[0])
                        r127_c4 = int(_tx_to_c4_idx(x_s[127:128])[0])
                        stress_rows.append(
                            {
                                "cluster_id": cap_idx * 8 + loop_id,
                                "capture_id": cap_idx,
                                "is_holdout": is_holdout,
                                "t1": tc[0], "res1": rc[0], "t2": tc[1], "res2": rc[1],
                                "v_norm": fit_s["v_norm"], "omega": fit_s["omega_hat"], "h": ent["h"], "k1": k1,
                                "tp126": tr[0] * np.conj(ROOTS[r126_c4]), "tp127": tr[1] * np.conj(ROOTS[r127_c4]),
                                "q1_lbl": ent["q1_lbl"], "q2_lbl": ent["q2_lbl"],
                                "sync_ok": bool(fit_s["sync_corr"] >= 0.25),
                            }
                        )

        del x_mm

    out: Dict[str, Any] = {"dataset": "cttc", "wire_id": WIRE_ID, "splits": {}}
    for sp_name, blist in split_buckets.items():
        out["splits"][sp_name] = _concat_shards(blist, split_name=sp_name)

    if q3_rows:
        nq = len(q3_rows)
        t1_q = np.array([r["t1"] for r in q3_rows], dtype=np.complex128)
        r1_q = np.array([r["res1"] for r in q3_rows], dtype=np.float64)
        t2_q = np.array([r["t2"] for r in q3_rows], dtype=np.complex128)
        r2_q = np.array([r["res2"] for r in q3_rows], dtype=np.float64)
        v_q = np.array([r["v_norm"] for r in q3_rows], dtype=np.float64)
        om_q = np.array([r["omega"] for r in q3_rows], dtype=np.float64)
        h_q = np.array([r["h"] for r in q3_rows], dtype=np.int64)
        ell_q = CTTC_SRC_N_SYMS - h_q
        avail_q = np.ones((nq, 2), dtype=bool)
        s17_q, w16_q = _encode_dual_pair(t1_q, r1_q, v_q, t2_q, r2_q, v_q, encoder, rng_anc, avail_q)
        ps_q = np.stack(
            [
                np.cos(om_q * h_q), np.sin(om_q * h_q),
                np.cos(om_q * (h_q + ell_q)), np.sin(om_q * (h_q + ell_q)),
                np.ones(nq), np.zeros(nq),
            ],
            axis=-1,
        ).astype(np.float32)
        ms_q = np.stack(
            [
                h_q.astype(np.float32), ell_q.astype(np.float32), np.zeros(nq, dtype=np.float32),
                om_q.astype(np.float32), np.full(nq, 0.02, dtype=np.float32),
                np.clip(-10.0 * np.log10(np.maximum(v_q, 1e-6)), -10.0, 26.0).astype(np.float32),
                np.full(nq, 1e-6, dtype=np.float32),
            ],
            axis=-1,
        )
        out["q3_pairs"] = {
            "split": "cttc_q3_pairs",
            "cluster_ids": np.array([r["cluster_id"] for r in q3_rows], dtype=np.int64),
            "capture_ids": np.array([r["capture_id"] for r in q3_rows], dtype=np.int64),
            "is_holdout": np.array([r["is_holdout"] for r in q3_rows], dtype=bool),
            "s17_codes": s17_q,
            "words_c16": w16_q,
            "avail": avail_q,
            "phase_slots": ps_q,
            "meta_scalars": ms_q,
            "discrete_d": np.zeros(nq, dtype=np.float32),
            "ref_none": _build_per_row_m3_ref_summary((), nq, om_q, np.full(nq, 0.02), np.full(nq, 1e-6)),
            "labels_q3": np.array([r["q3_lbl"] for r in q3_rows], dtype=np.int64),
            "t1_cont": t1_q,
            "v_norm": v_q,
        }

    if stress_rows:
        ns = len(stress_rows)
        t1_s = np.array([r["t1"] for r in stress_rows], dtype=np.complex128)
        r1_s = np.array([r["res1"] for r in stress_rows], dtype=np.float64)
        t2_s = np.array([r["t2"] for r in stress_rows], dtype=np.complex128)
        r2_s = np.array([r["res2"] for r in stress_rows], dtype=np.float64)
        v_s = np.array([r["v_norm"] for r in stress_rows], dtype=np.float64)
        om_s = np.array([r["omega"] for r in stress_rows], dtype=np.float64)
        h_s = np.array([r["h"] for r in stress_rows], dtype=np.int64)
        k1_s = np.array([r["k1"] for r in stress_rows], dtype=np.int64)
        ell_s = CTTC_SRC_N_SYMS - h_s
        so_s = np.array([r["sync_ok"] for r in stress_rows], dtype=bool)
        avail_s = np.stack([so_s, so_s], axis=-1)
        s17_s, w16_s = _encode_dual_pair(t1_s, r1_s, v_s, t2_s, r2_s, v_s, encoder, rng_anc, avail_s)
        dt126_s = 126.0 - k1_s.astype(np.float64)
        dt127_s = 127.0 - k1_s.astype(np.float64)
        ps_s = np.stack(
            [
                np.cos(om_s * h_s), np.sin(om_s * h_s),
                np.cos(om_s * (h_s + ell_s)), np.sin(om_s * (h_s + ell_s)),
                np.cos(om_s * (126.5 - k1_s)), np.sin(om_s * (126.5 - k1_s)),
            ],
            axis=-1,
        ).astype(np.float32)
        ms_s = np.stack(
            [
                h_s.astype(np.float32), ell_s.astype(np.float32), (126.5 - k1_s).astype(np.float32),
                om_s.astype(np.float32), np.full(ns, 0.02, dtype=np.float32),
                np.clip(-10.0 * np.log10(np.maximum(v_s, 1e-6)), -10.0, 26.0).astype(np.float32),
                np.full(ns, 1e-6, dtype=np.float32),
            ],
            axis=-1,
        )
        r126_s = {"acquisition_id": 126, "shares_channel": True, "t_norm": np.array([r["tp126"] for r in stress_rows]), "v_norm": v_s, "dt": dt126_s, "dt_mean": float(np.mean(dt126_s))}
        r127_s = {"acquisition_id": 127, "shares_channel": True, "t_norm": np.array([r["tp127"] for r in stress_rows]), "v_norm": v_s, "dt": dt127_s, "dt_mean": float(np.mean(dt127_s))}
        out["stress"] = {
            "split": "cttc_stress",
            "cluster_ids": np.array([r["cluster_id"] for r in stress_rows], dtype=np.int64),
            "capture_ids": np.array([r["capture_id"] for r in stress_rows], dtype=np.int64),
            "is_holdout": np.array([r["is_holdout"] for r in stress_rows], dtype=bool),
            "s17_codes": s17_s,
            "words_c16": w16_s,
            "avail": avail_s,
            "phase_slots": ps_s,
            "meta_scalars": ms_s,
            "discrete_d": np.zeros(ns, dtype=np.float32),
            "ref_none": _build_per_row_m3_ref_summary((), ns, om_s, np.full(ns, 0.02), np.full(ns, 1e-6)),
            "ref_one": _build_per_row_m3_ref_summary([r126_s], ns, om_s, np.full(ns, 0.02), np.full(ns, 1e-6)),
            "ref_two": _build_per_row_m3_ref_summary([r126_s, r127_s], ns, om_s, np.full(ns, 0.02), np.full(ns, 1e-6)),
            "labels_q1": np.array([r["q1_lbl"] for r in stress_rows], dtype=np.int64),
            "labels_q2": np.array([r["q2_lbl"] for r in stress_rows], dtype=np.int64),
            "t1_cont": t1_s,
            "t2_cont": t2_s,
            "v_norm": v_s,
            "sync_ok": so_s,
        }
    return out


def _concat_shards(shards: Sequence[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split_name}
    keys = [k for k in shards[0].keys() if isinstance(shards[0][k], np.ndarray)]
    for k in keys:
        out[k] = np.concatenate([s[k] for s in shards], axis=0)
    return out


# =============================================================================
# 5. Sentinel-Poisoning Zero-Leakage Proof + Synthetic Offline Generator
# =============================================================================

def verify_zero_leakage_sentinel(
    tx_c64: np.ndarray,
    rx_c64: np.ndarray,
    cttc_work_dir: Path,
    encoder,
    n_test_pluto: int = 8,
    n_test_cttc: int = 4,
) -> dict:
    """Poison withheld TX/RX slots with `1e6 + 1e6j` and prove causal outputs change by `0.0`."""
    # 1. Pluto causal sentinel proof
    pl_clean = extract_pluto_causal_dual_cache(
        tx_c64, rx_c64, encoder, splits=("test",), include_stress=False, seed=99, max_per_split=max(n_test_pluto, 8)
    )
    tx_bad = np.asarray(tx_c64, dtype=np.complex64).copy()
    rx_bad = np.asarray(rx_c64, dtype=np.complex64).copy()
    st_arr = pl_clean["splits"]["test"]["sample_start"]
    f0 = PLUTO_MANIFEST["splits"]["test"]["eval_start"]
    poison_val = np.complex64(1e6 + 1e6j)
    for idx in range(min(n_test_pluto, len(st_arr))):
        f = f0 + idx
        s0 = int(st_arr[idx])
        tx_bad[f, 21:] = poison_val
        p_lo = s0 + 14 * PLUTO_SPS
        p_hi = min(PLUTO_RX_SAMP, s0 + 26 * PLUTO_SPS)
        rx_bad[f, p_lo:p_hi] = poison_val
        rx_bad[f, s0 + PLUTO_N_SYM * PLUTO_SPS :] = poison_val

    pl_bad = extract_pluto_causal_dual_cache(
        tx_bad, rx_bad, encoder, splits=("test",), include_stress=False, seed=99, max_per_split=max(n_test_pluto, 8)
    )
    max_s17_p = int(np.max(np.abs(pl_bad["splits"]["test"]["s17_codes"][:n_test_pluto] - pl_clean["splits"]["test"]["s17_codes"][:n_test_pluto])))
    max_w16_p = int(np.max(np.abs(pl_bad["splits"]["test"]["words_c16"][:n_test_pluto] - pl_clean["splits"]["test"]["words_c16"][:n_test_pluto])))
    max_meta_p = float(np.max(np.abs(pl_bad["splits"]["test"]["meta_scalars"][:n_test_pluto] - pl_clean["splits"]["test"]["meta_scalars"][:n_test_pluto])))
    ref_none_p_diff = float(np.max(np.abs(pl_bad["splits"]["test"]["ref_none"][:n_test_pluto] - pl_clean["splits"]["test"]["ref_none"][:n_test_pluto])))

    # 2. CTTC causal sentinel proof on first capture packet buffer
    raw_bits = np.fromfile(cttc_work_dir / "sources" / "source_QPSK.dat", dtype=np.uint8)
    b0_e, b1_e = raw_bits[0::2], raw_bits[1::2]
    sym_even = ((1.0 - 2.0 * b0_e) + 1j * (1.0 - 2.0 * b1_e)) / math.sqrt(2.0)
    b0_o, b1_o = raw_bits[1:-1:2], raw_bits[2::2]
    sym_odd = ((1.0 - 2.0 * b0_o) + 1j * (1.0 - 2.0 * b1_o)) / math.sqrt(2.0)
    anchor = CTTC_CAPTURE_ANCHORS[0]
    x_mm = np.memmap(cttc_work_dir / "data" / anchor["capture_id"] / "file_2.dat", dtype=np.complex64, mode="r")
    sym_ref = sym_odd if int(anchor["bit_alignment"]) == 1 else sym_even
    cfo_prior, a_floor = _estimate_cttc_commissioning_cfo(x_mm, sym_ref, anchor)
    lo, hi = CTTC_MANIFEST["split_source_ranges"]["test"]["target"]
    rx_loop0 = float(anchor["anchor_rx_pos"]) - float(anchor["anchor_src_sym"]) * float(anchor["sps_eff"])
    while rx_loop0 < 2_900_000.0:
        rx_loop0 += float(CTTC_SRC_N_SYMS) * float(anchor["sps_eff"])
    stride = (hi - lo - CTTC_PKT_LEN) // 20
    pkt_rx_len = (CTTC_PKT_LEN + 2 * CTTC_PAD_SYMS) * CTTC_SPS
    cttc_diffs: List[float] = []
    for j in range(n_test_cttc):
        src_start = lo + j * stride + 128
        w0 = int(math.floor(rx_loop0 + float(src_start) * float(anchor["sps_eff"]))) - CTTC_PAD_SYMS * CTTC_SPS
        y_clean = np.asarray(x_mm[w0 : w0 + pkt_rx_len], dtype=np.complex64)
        y_pois = y_clean.copy()
        h_s0 = (CTTC_PAD_SYMS + 40) * CTTC_SPS
        h_s1 = (CTTC_PAD_SYMS + 48) * CTTC_SPS
        y_pois[h_s0:h_s1] = poison_val
        y_pois[(CTTC_PAD_SYMS + 200) * CTTC_SPS : (CTTC_PAD_SYMS + 256) * CTTC_SPS] = poison_val
        x_sync = sym_ref[src_start : src_start + CTTC_PKT_LEN][CTTC_SYNC_IDX]
        f_c = _cttc_causal_pkt_fit(y_clean, x_sync, float(anchor["bg_noise_var"]), cfo_prior, a_floor=a_floor)
        f_p = _cttc_causal_pkt_fit(y_pois, x_sync, float(anchor["bg_noise_var"]), cfo_prior, a_floor=a_floor)
        cttc_diffs.append(abs(f_c["best_t"] - f_p["best_t"]) + abs(f_c["omega_hat"] - f_p["omega_hat"]) + abs(f_c["h_hat"] - f_p["h_hat"]))

    max_cttc_diff = float(max(cttc_diffs)) if cttc_diffs else 0.0
    ok = bool(max_s17_p == 0 and max_w16_p == 0 and max_meta_p == 0.0 and ref_none_p_diff == 0.0 and max_cttc_diff == 0.0)
    assert ok, f"Zero-leakage sentinel failure! pluto=({max_s17_p},{max_w16_p},{max_meta_p},{ref_none_p_diff}) cttc={max_cttc_diff}"
    return {
        "pluto_s17_max_diff": max_s17_p,
        "pluto_w16_max_diff": max_w16_p,
        "pluto_meta_max_diff": max_meta_p,
        "pluto_ref_none_max_diff": ref_none_p_diff,
        "cttc_causal_fit_max_diff": max_cttc_diff,
        "ok": ok,
    }


MOCK_CTTC_ANCHORS: Tuple[Dict[str, Any], ...] = (CTTC_CAPTURE_ANCHORS[0], CTTC_CAPTURE_ANCHORS[4])


def generate_mock_pluto_cttc_waveforms(work_root: Path, seed: int = 202611) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Generate compact deterministic synthetic Pluto & CTTC raw buffers matching exact geometry for CPU pytest."""
    rng = np.random.default_rng(seed)
    n_frames = 5000  # covers train warmup [0:200], train [200:360], dev [3500:3660], test [4800:4960]
    tx_c64 = np.zeros((n_frames, PLUTO_N_SYM), dtype=np.complex64)
    qpsk_pts = np.array([0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j], dtype=np.complex64)
    rand_idx = rng.integers(0, 4, size=(n_frames, PLUTO_N_SYM))
    tx_c64[:] = qpsk_pts[rand_idx]
    tx_c64[:, :13] = PLUTO_SYNC_HDR_13.astype(np.complex64)[None, :]
    tx_c64[:, 13:26] = PLUTO_SYNC_HDR_13.astype(np.complex64)[None, :]

    rx_c64 = np.zeros((n_frames, PLUTO_RX_SAMP), dtype=np.complex64)
    active_slices = (slice(0, 200), slice(200, 360), slice(3500, 3660), slice(4800, 4960))
    for sl in active_slices:
        m = sl.stop - sl.start
        base_phase = rng.uniform(-0.3, 0.3, size=m)
        up = np.repeat(tx_c64[sl].astype(np.complex128), PLUTO_SPS, axis=1)
        up_3 = np.concatenate([up, up, up], axis=1)
        chan = (0.022 * np.exp(1j * base_phase))[:, None]
        noise = (rng.normal(0.0, 0.0025, size=up_3.shape) + 1j * rng.normal(0.0, 0.0025, size=up_3.shape)).astype(np.complex64)
        rx_c64[sl] = (up_3 * chan).astype(np.complex64) + noise

    cttc_dir = work_root / "mock_cttc"
    (cttc_dir / "sources").mkdir(parents=True, exist_ok=True)
    bits = rng.integers(0, 2, size=1_000_000, dtype=np.uint8)
    bits.tofile(cttc_dir / "sources" / "source_QPSK.dat")
    b0, b1 = bits[0::2], bits[1::2]
    syms = ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) / math.sqrt(2.0)

    n_samp = 8_200_000
    loop_len = CTTC_SRC_N_SYMS * CTTC_SPS  # 2,000,000 samples
    up_loop = np.repeat(syms * 0.0005, CTTC_SPS).astype(np.complex64)
    for anchor in MOCK_CTTC_ANCHORS:
        cdir = cttc_dir / "data" / anchor["capture_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        mm = np.memmap(cdir / "file_2.dat", dtype=np.complex64, mode="w+", shape=(n_samp,))
        rx_l0 = float(anchor["anchor_rx_pos"]) - float(anchor["anchor_src_sym"]) * float(anchor["sps_eff"])
        while rx_l0 < 2_900_000.0:
            rx_l0 += float(loop_len)
        for loop_i in range(2):
            b_i = int(round(rx_l0)) + loop_i * loop_len
            end_i = min(b_i + loop_len, n_samp)
            if b_i < n_samp and end_i > b_i:
                seg_n = end_i - b_i
                noise_c = (
                    rng.normal(0.0, 5e-5, size=seg_n) + 1j * rng.normal(0.0, 5e-5, size=seg_n)
                ).astype(np.complex64)
                mm[b_i:end_i] = up_loop[:seg_n] + noise_c
        mm.flush()
        del mm
    return tx_c64, rx_c64, cttc_dir


# =============================================================================
# 6. CPU Physical Evaluators (4-bit s17 vs 16-bit m5-c16-v1-le vs Continuous Bayes)
# =============================================================================

def _decode_c16_to_complex_and_var(words: np.ndarray, v_nom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode `m5-c16-v1-le` words into calibrated `(T_hat, v_eff)` matching M5 §3.1."""
    dec = decode_c16(words)
    theta = dec["turn"].astype(np.float64) * (0.5 * math.pi) - 0.25 * math.pi + (
        dec["angle_bin"].astype(np.float64) + 0.5
    ) * (0.5 * math.pi / 64.0)
    rc = np.clip(dec["radial_code"], 0, 15)
    low = WIRE_RADIAL_EDGES[np.maximum(0, rc - 1)]
    high = np.where(rc < 15, WIRE_RADIAL_EDGES[rc], WIRE_RADIAL_EDGES[14] * 1.15)
    r_mid = 0.5 * (low + high)
    v_safe = np.maximum(np.asarray(v_nom, dtype=np.float64), 1e-12)
    t_mag = np.where(dec["erased"], 0.0, r_mid * np.sqrt(v_safe))
    t_hat = t_mag * np.exp(1j * theta)
    # Residual quality inflates effective variance smoothly on noisy/saturated bins
    q_mult = np.array([1.0, 1.15, 1.5, 2.5], dtype=np.float64)[dec["quality_bin"]]
    v_eff = v_safe * q_mult
    return t_hat, v_eff


def _c4_posterior_from_scalar(t: np.ndarray, v: np.ndarray, avail: np.ndarray) -> np.ndarray:
    roots_np = np.asarray(ROOTS, dtype=np.complex128)
    snr = np.clip(1.0 / np.maximum(v, 1e-12), 0.05, 50.0)
    lg = (2.0 * snr)[:, None] * np.real(t[:, None] * np.conj(roots_np)[None, :])
    lg -= np.max(lg, axis=1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=1, keepdims=True)
    return np.where(avail[:, None], p, 0.25)


def _eval_physical_probs_from_pair(
    a1: np.ndarray,
    a2: np.ndarray,
    shard: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute matched-information Q1, Q2 stages, and Q3 sector posteriors from `(a1, a2)`."""
    n = len(a1)
    h = shard["meta_scalars"][:, 0].astype(np.float64)
    ell = shard["meta_scalars"][:, 1].astype(np.float64)
    omega = shard["meta_scalars"][:, 3].astype(np.float64)
    sp2 = shard["meta_scalars"][:, 4].astype(np.float64) ** 2
    rv = shard["meta_scalars"][:, 6].astype(np.float64)
    var12 = np.clip(sp2 * h + rv * (h ** 2), 1e-6, 100.0)
    var13 = np.clip(sp2 * (h + ell) + rv * ((h + ell) ** 2), 1e-6, 100.0)

    def _wquad(theta_v: np.ndarray, var_v: np.ndarray) -> np.ndarray:
        out = np.zeros((n, 4), dtype=np.float64)
        sd = np.sqrt(var_v)
        m_img = np.arange(-3, 4, dtype=np.float64)
        from scipy.special import ndtr

        for k in range(4):
            lo = -0.25 * math.pi + 0.5 * math.pi * k + 2.0 * math.pi * m_img[None, :]
            hi = lo + 0.5 * math.pi
            out[:, k] = np.sum(
                ndtr((hi - theta_v[:, None]) / sd[:, None]) - ndtr((lo - theta_v[:, None]) / sd[:, None]),
                axis=1,
            )
        out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out

    # Q1: phase rotation from k1 -> k2
    w12 = _wquad(omega * h, var12)
    a2_unshifted = sum(w12[:, k : k + 1] * np.roll(a2, -k, axis=-1) for k in range(4))
    p_q1 = np.stack([np.sum(a1 * np.roll(a2_unshifted, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)
    p_q1 /= np.maximum(p_q1.sum(axis=-1, keepdims=True), 1e-12)

    # Q2 stages
    q2_out: Dict[str, np.ndarray] = {}
    for arm_k in ("ref_none", "ref_one", "ref_two", "ref_dup", "ref_indep"):
        if arm_k not in shard:
            continue
        bp = shard[arm_k][:, 0:4].astype(np.float64)
        p_q2 = np.stack([np.sum(bp * np.roll(a1, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)
        p_q2 /= np.maximum(p_q2.sum(axis=-1, keepdims=True), 1e-12)
        q2_out[arm_k] = p_q2

    # Q3 retransmission
    w13 = _wquad(omega * (h + ell), var13)
    p_q3 = np.stack([np.sum(a1 * np.roll(w13, c, axis=-1), axis=-1) for c in range(4)], axis=-1)
    p_q3 /= np.maximum(p_q3.sum(axis=-1, keepdims=True), 1e-12)
    return {"q1": p_q1, "q2": q2_out, "q3": p_q3}


def _score_probs(probs: np.ndarray, labels: np.ndarray, gray_eval: bool = False) -> Dict[str, Any]:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    y = np.asarray(labels, dtype=np.int64)
    nll = -np.log(p[np.arange(len(y)), y])
    hits = (p.argmax(axis=-1) == y).astype(np.float64)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "acc": float(np.mean(hits)),
        "nll": float(np.mean(nll)),
        "_nll": nll,
        "_hits": hits,
    }
    if gray_eval:
        # Per-bit NLL & posterior-mean RMS EVM (M4 §2.1)
        true_bits = _GRAY_BITS_BY_C4[y]  # (N, 2)
        p_b0_1 = p[:, 1] + p[:, 2]
        p_b1_1 = p[:, 2] + p[:, 3]
        p_bit1 = np.stack([p_b0_1, p_b1_1], axis=-1)
        p_true_bit = np.where(true_bits > 0.5, p_bit1, 1.0 - p_bit1)
        bit_nll = -np.log(np.clip(p_true_bit, 1e-12, 1.0))
        roots_np = np.asarray(ROOTS, dtype=np.complex128)
        s_hat = p @ roots_np
        s_true = roots_np[y]
        evm_rms = float(np.sqrt(np.sum(np.abs(s_hat - s_true) ** 2) / max(float(np.sum(np.abs(s_true) ** 2)), 1e-12)))
        out["per_bit_nll"] = float(np.mean(bit_nll))
        out["posterior_rms_evm"] = evm_rms
    return out


def _paired_cluster_ci(vals_a: np.ndarray, vals_b: np.ndarray, cluster_ids: np.ndarray) -> dict:
    uniq = np.unique(cluster_ids)
    diffs = np.array([float(np.mean(vals_a[cluster_ids == c] - vals_b[cluster_ids == c])) for c in uniq], dtype=np.float64)
    mean_d = float(np.mean(diffs))
    if len(diffs) >= 2 and float(np.std(diffs, ddof=1)) > 0.0:
        tres = scipy_stats.ttest_rel(
            np.array([float(np.mean(vals_a[cluster_ids == c])) for c in uniq]),
            np.array([float(np.mean(vals_b[cluster_ids == c])) for c in uniq]),
        )
        p_two = float(tres.pvalue)
    else:
        p_two = 1.0 if abs(mean_d) < 1e-12 else 0.0
    rng = np.random.default_rng(202611)
    idx = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    b_m = diffs[idx].mean(axis=1)
    return {
        "n_clusters": int(len(uniq)),
        "mean_diff": mean_d,
        "ci95_low": float(np.percentile(b_m, 2.5)),
        "ci95_high": float(np.percentile(b_m, 97.5)),
        "p_two_sided": p_two,
    }


def evaluate_sdr_cpu_physical(
    pluto_data: Dict[str, Any],
    cttc_data: Dict[str, Any],
    encoder,
) -> Dict[str, Any]:
    """Compare matched-information 4-bit s17, 16-bit m5-c16-v1-le, and continuous causal Bayes on real SDR."""
    prior_tbl = np.asarray(encoder.table, dtype=np.float64)

    def _run_shard_decoders(shard: Dict[str, Any], has_q1: bool = True, has_q2: bool = True, has_q3: bool = True) -> dict:
        s17_1 = shard["s17_codes"][:, 0]
        s17_2 = shard["s17_codes"][:, 1]
        av1, av2 = shard["avail"][:, 0], shard["avail"][:, 1]
        a1_4 = np.where(av1[:, None], prior_tbl[s17_1], 0.25)
        a2_4 = np.where(av2[:, None], prior_tbl[s17_2], 0.25)

        t1_16, v1_16 = _decode_c16_to_complex_and_var(shard["words_c16"][:, 0], shard["v_norm"])
        t2_16, v2_16 = _decode_c16_to_complex_and_var(shard["words_c16"][:, 1], shard["v_norm"])
        a1_16 = _c4_posterior_from_scalar(t1_16, v1_16, av1)
        a2_16 = _c4_posterior_from_scalar(t2_16, v2_16, av2)

        a1_cont = _c4_posterior_from_scalar(shard["t1_cont"], shard["v_norm"], av1)
        t2_arr = shard.get("t2_cont", np.zeros_like(shard["t1_cont"]))
        a2_cont = _c4_posterior_from_scalar(t2_arr, shard["v_norm"], av2)

        res: Dict[str, Any] = {}
        raw_caches: Dict[str, dict] = {}
        for mname, (pa1, pa2) in (
            ("direct_s17_4bit", (a1_4, a2_4)),
            ("dual_cache_c16_16bit", (a1_16, a2_16)),
            ("continuous_causal_bayes", (a1_cont, a2_cont)),
        ):
            probs = _eval_physical_probs_from_pair(pa1, pa2, shard)
            raw_caches[mname] = probs
            md: Dict[str, Any] = {}
            if has_q1:
                sc_q1 = _score_probs(probs["q1"], shard["labels_q1"])
                md["q1"] = {k: v for k, v in sc_q1.items() if not k.startswith("_")}
            if has_q2:
                q2_d: Dict[str, Any] = {}
                for arm_k, p_q2 in probs["q2"].items():
                    sc_2 = _score_probs(p_q2, shard["labels_q2"], gray_eval=True)
                    q2_d[arm_k] = {k: v for k, v in sc_2.items() if not k.startswith("_")}
                if "ref_dup" in probs["q2"] and "ref_two" in probs["q2"]:
                    q2_d["max_prob_diff_dup_vs_two"] = float(np.max(np.abs(probs["q2"]["ref_dup"] - probs["q2"]["ref_two"])))
                if "ref_indep" in probs["q2"] and "ref_two" in probs["q2"]:
                    q2_d["max_prob_diff_indep_vs_two"] = float(np.max(np.abs(probs["q2"]["ref_indep"] - probs["q2"]["ref_two"])))
                if "ref_one" in probs["q2"] and "ref_two" in probs["q2"]:
                    nll_one = _score_probs(probs["q2"]["ref_one"], shard["labels_q2"])["_nll"]
                    nll_two = _score_probs(probs["q2"]["ref_two"], shard["labels_q2"])["_nll"]
                    q2_d["q2_two_minus_one_nll"] = _paired_cluster_ci(nll_two, nll_one, shard["cluster_ids"])
                md["q2"] = q2_d
            if has_q3:
                sc_q3 = _score_probs(probs["q3"], shard["labels_q3"])
                md["q3"] = {k: v for k, v in sc_q3.items() if not k.startswith("_")}
            res[mname] = md

        # Paired 16-bit vs 4-bit comparison (NLL diff = c16 - s17, negative is better)
        pair_16_4: Dict[str, Any] = {}
        cl = shard["cluster_ids"]
        if has_q1:
            n4 = _score_probs(raw_caches["direct_s17_4bit"]["q1"], shard["labels_q1"])["_nll"]
            n16 = _score_probs(raw_caches["dual_cache_c16_16bit"]["q1"], shard["labels_q1"])["_nll"]
            pair_16_4["q1_nll_diff_c16_minus_s17"] = _paired_cluster_ci(n16, n4, cl)
        if has_q2 and "ref_two" in shard:
            n4 = _score_probs(raw_caches["direct_s17_4bit"]["q2"]["ref_two"], shard["labels_q2"])["_nll"]
            n16 = _score_probs(raw_caches["dual_cache_c16_16bit"]["q2"]["ref_two"], shard["labels_q2"])["_nll"]
            pair_16_4["q2_two_nll_diff_c16_minus_s17"] = _paired_cluster_ci(n16, n4, cl)
        if has_q3:
            n4 = _score_probs(raw_caches["direct_s17_4bit"]["q3"], shard["labels_q3"])["_nll"]
            n16 = _score_probs(raw_caches["dual_cache_c16_16bit"]["q3"], shard["labels_q3"])["_nll"]
            pair_16_4["q3_nll_diff_c16_minus_s17"] = _paired_cluster_ci(n16, n4, cl)
        res["paired_16bit_vs_4bit"] = pair_16_4
        return res

    # Split CTTC test cleanly into In-Domain (non-holdout) and Strictly Unseen Holdout (`f708_snr10_QPSK`)
    cttc_test = cttc_data["splits"]["test"]
    in_mask = ~cttc_test["is_holdout"]
    ho_mask = cttc_test["is_holdout"]
    cttc_indom = {k: (v[in_mask] if isinstance(v, np.ndarray) else v) for k, v in cttc_test.items()}
    cttc_hold = {k: (v[ho_mask] if isinstance(v, np.ndarray) else v) for k, v in cttc_test.items()}

    rep: Dict[str, Any] = {
        "pluto_test": _run_shard_decoders(pluto_data["splits"]["test"], has_q1=True, has_q2=True, has_q3=True),
        "cttc_in_domain_test": _run_shard_decoders(cttc_indom, has_q1=True, has_q2=True, has_q3=False),
        "cttc_unseen_holdout_f708_snr10": _run_shard_decoders(cttc_hold, has_q1=True, has_q2=True, has_q3=False),
    }
    if "q3_pairs" in cttc_data:
        rep["cttc_q3_loop_pairs"] = _run_shard_decoders(cttc_data["q3_pairs"], has_q1=False, has_q2=False, has_q3=True)
    if "stress" in pluto_data:
        rep["pluto_stress"] = _run_shard_decoders(pluto_data["stress"], has_q1=True, has_q2=True, has_q3=True)
    if "stress" in cttc_data:
        rep["cttc_stress"] = _run_shard_decoders(cttc_data["stress"], has_q1=True, has_q2=True, has_q3=False)
    return rep


# =============================================================================
# 7. End-to-End CPU Pipeline Driver & Compact Cache Saver
# =============================================================================

def _save_bundle_npz(path: Path, bundle: Dict[str, Any]) -> None:
    flat: Dict[str, np.ndarray] = {}
    for top_k, val in bundle.items():
        if top_k == "splits":
            for sp_k, sp_d in val.items():
                for arr_k, arr_v in sp_d.items():
                    if isinstance(arr_v, np.ndarray):
                        flat[f"splits__{sp_k}__{arr_k}"] = arr_v
        elif isinstance(val, dict):
            for arr_k, arr_v in val.items():
                if isinstance(arr_v, np.ndarray):
                    flat[f"{top_k}__{arr_k}"] = arr_v
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_bundle_npz(path: Path) -> Dict[str, Any]:
    """Load nested dual-cache `.npz` written by `_save_bundle_npz`."""
    z = np.load(path, allow_pickle=False)
    out: Dict[str, Any] = {"splits": {}}
    for k in z.files:
        parts = k.split("__")
        if parts[0] == "splits" and len(parts) == 3:
            out["splits"].setdefault(parts[1], {})[parts[2]] = z[k]
        elif len(parts) == 2:
            out.setdefault(parts[0], {})[parts[1]] = z[k]
    return out


def run_real_sdr_cpu_pipeline(
    repo_root: Path,
    raw_root: Path,
    out_dir: Path,
    mock_synthetic_for_test: bool = False,
    seed: int = 202611,
) -> Dict[str, Any]:
    """Run full verified real-SDR causal extraction, zero-leakage sentinel proof, and CPU benchmarks."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, _ = load_s17_encoder(repo_root)

    if mock_synthetic_for_test:
        tx_c64, rx_c64, cttc_work = generate_mock_pluto_cttc_waveforms(raw_root, seed=seed)
        pluto_mat_info = {"mode": "mock_synthetic"}
        sentinel_rep = verify_zero_leakage_sentinel(tx_c64, rx_c64, cttc_work, encoder, n_test_pluto=4, n_test_cttc=2)
        pluto_data = extract_pluto_causal_dual_cache(
            tx_c64, rx_c64, encoder, splits=("train", "dev", "test"), include_stress=True, seed=seed, max_per_split=160
        )
        cttc_data = extract_cttc_causal_dual_cache(
            cttc_work,
            encoder,
            splits=("train", "dev", "test"),
            n_loops_use=2,
            include_stress=True,
            seed=seed,
            captures=MOCK_CTTC_ANCHORS,
            per_loop_override={"train": 4, "dev": 2, "test": 4},
        )
    else:
        mat_p = download_and_verify_pluto(raw_root / "pluto")
        tx_c64, rx_c64 = load_pluto_arrays(mat_p)
        cttc_work = download_and_verify_cttc(raw_root / "cttc_raw", raw_root / "cttc_work")
        pluto_mat_info = {"mode": "real_hardware", "pluto_sha256": PLUTO_EXPECTED_SHA256, "zenodo_id": ZENODO_RECORD_ID}
        sentinel_rep = verify_zero_leakage_sentinel(tx_c64, rx_c64, cttc_work, encoder, n_test_pluto=6, n_test_cttc=4)
        pluto_data = extract_pluto_causal_dual_cache(
            tx_c64, rx_c64, encoder, splits=("train", "dev", "test"), include_stress=True, seed=seed
        )
        cttc_data = extract_cttc_causal_dual_cache(
            cttc_work, encoder, splits=("train", "dev", "test"), n_loops_use=8, include_stress=True, seed=seed
        )

    _save_bundle_npz(out_dir / "pluto_dual_cache_c16.npz", pluto_data)
    _save_bundle_npz(out_dir / "cttc_dual_cache_c16.npz", cttc_data)

    phys_rep = evaluate_sdr_cpu_physical(pluto_data, cttc_data, encoder)
    counts = {
        "pluto": {k: int(len(v["words_c16"])) for k, v in pluto_data["splits"].items()},
        "pluto_stress": int(len(pluto_data["stress"]["words_c16"])),
        "cttc": {k: int(len(v["words_c16"])) for k, v in cttc_data["splits"].items()},
        "cttc_q3_pairs": int(len(cttc_data["q3_pairs"]["words_c16"])),
        "cttc_stress": int(len(cttc_data["stress"]["words_c16"])),
    }
    report = {
        "wire_id": WIRE_ID,
        "source": pluto_mat_info,
        "wall_s": round(time.perf_counter() - t0, 2),
        "sentinel_zero_leakage": sentinel_rep,
        "counts": counts,
        "cpu_physical_results": phys_rep,
    }
    (out_dir / "real_sdr_cpu_physical_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
