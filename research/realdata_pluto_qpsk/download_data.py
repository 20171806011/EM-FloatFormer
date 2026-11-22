#!/usr/bin/env python3
"""Download and verify Mendeley Data 'Frames of SDR ADALM PLUTO based QPSK Communication Link'.

Dataset DOI: 10.17632/k72b4pnc5g.1
Official URL: https://data.mendeley.com/datasets/k72b4pnc5g/1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

DATASET_ID = "k72b4pnc5g"
DATASET_VERSION = "1"
METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}?version={DATASET_VERSION}"
FILES_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={DATASET_VERSION}"
EXPECTED_FILENAME = "QPSKlink.mat"
EXPECTED_SIZE = 822762822
EXPECTED_SHA256 = "c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7"


def fetch_json(url: str, out_path: Path) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "EM-FloatFormer-Research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    out_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_stream(url: str, dest: Path, expected_size: int | None = None) -> tuple[int, str, float]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    total = 0
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "EM-FloatFormer-Research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            total += len(chunk)
    elapsed = time.perf_counter() - t0
    if expected_size is not None and total != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch: got {total}, expected {expected_size}")
    os.replace(tmp, dest)
    return total, h.hexdigest(), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to store raw dataset and metadata")
    ap.add_argument("--manifest-out", type=Path, default=None, help="Optional path to write verification manifest JSON")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "dataset_metadata.json"
    files_path = out_dir / "files_metadata.json"
    dataset_meta = fetch_json(METADATA_URL, meta_path)
    files_meta = fetch_json(FILES_URL, files_path)

    target_entry = None
    for entry in files_meta:
        if entry.get("filename") == EXPECTED_FILENAME:
            target_entry = entry
            break
    if target_entry is None:
        raise RuntimeError(f"Could not find {EXPECTED_FILENAME} in Mendeley files metadata")

    details = target_entry["content_details"]
    dl_url = details["download_url"]
    api_size = int(details["size"])
    api_sha256 = str(details["sha256_hash"]).lower()

    mat_path = out_dir / EXPECTED_FILENAME
    if mat_path.exists() and mat_path.stat().st_size == api_size:
        local_sha = sha256_file(mat_path)
        if local_sha == api_sha256:
            download_method = "cached_verified"
            elapsed_s = 0.0
            actual_size = api_size
        else:
            actual_size, local_sha, elapsed_s = download_stream(dl_url, mat_path, expected_size=api_size)
            download_method = "urllib_stream_redownload"
    else:
        actual_size, local_sha, elapsed_s = download_stream(dl_url, mat_path, expected_size=api_size)
        download_method = "urllib_stream"

    if local_sha != api_sha256 or local_sha != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: local={local_sha}, api={api_sha256}, expected={EXPECTED_SHA256}")

    with mat_path.open("rb") as f:
        header_bytes = f.read(128)
    is_hdf5 = header_bytes[:8] == b"\x89HDF\r\n\x1a\n"
    mat_header_ascii = header_bytes[:116].decode("latin1", errors="replace").strip("\x00 ")

    manifest = {
        "dataset_title": dataset_meta.get("name"),
        "dataset_doi": f"10.17632/{DATASET_ID}.{DATASET_VERSION}",
        "dataset_url": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
        "public_api_metadata_url": METADATA_URL,
        "public_api_files_url": FILES_URL,
        "download_url": dl_url,
        "download_method": download_method,
        "download_elapsed_sec": round(elapsed_s, 3),
        "filename": EXPECTED_FILENAME,
        "size_bytes": actual_size,
        "sha256": local_sha,
        "sha256_matches_mendeley_api": local_sha == api_sha256,
        "is_hdf5_v73": is_hdf5,
        "header_preview": mat_header_ascii[:100],
        "metadata_sha256": sha256_file(meta_path),
        "files_metadata_sha256": sha256_file(files_path),
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = args.manifest_out or (out_dir / "data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
