"""Download and verify CTTC Indoor Wireless Channel QPSK datasets (Zenodo 4895133).

Official source:
  Adriano Pastore & Armin Ghani (CTTC), "Datasets of Indoor Wireless Channel Measurements
  for Machine Learning Applications", Zenodo, June 2, 2021.
  DOI: 10.5281/zenodo.4895133 | URL: https://zenodo.org/records/4895133

Run on remote compute host (e.g. GitHub Codespace into /tmp/agy_cli1_cttc/raw).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

ZENODO_RECORD_ID = "4895133"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

MANIFEST_FILES: list[dict[str, Any]] = [
    {
        "filename": "README.pdf",
        "role": "documentation",
        "size_bytes": 1549089,
        "md5": "620c116c2a38a6dfbb6316752833850d",
        "sha256": "c8e7ae57e0896dfe2f1cc910d084e9b77e3cc5143b721cf9b24753ab8b496fb4",
    },
    {
        "filename": "sources.tar.gz",
        "role": "tx_source_bits",
        "size_bytes": 157051651,
        "md5": "48766e3f5ddaa968ccf128eb5891a450",
        "sha256": "f9022be37d3eeadcff481f46b25ca9a067c0e19988bc4bdcb77cca9957ee7b4f",
        "extracted_files": {
            "source_QPSK.dat": {
                "size_bytes": 1000000,
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
        "size_bytes": 211697276,
        "md5": "1df69db08519d130faff0857347930de",
        "sha256": "161d618ef2da2412a3c4d7236deb7c52f0b81f31af704b9d852d90f849c6cbb3",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "7e9e6ff2af0f10f2cf0a47679394a6443a1fb8051c6e0c61acacd29c55a5389e"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "c6596f98fd9705c475b8b07340facb6ad7dbc4906e6f196ac2f207a275544ab1"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "20c521344958acda0cd30dcd0449a41ab16d472e6942bee0d5e539c788c7a4db"},
        },
    },
    {
        "filename": "f2450_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr10_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 10,
        "size_bytes": 197835355,
        "md5": "545e2f96b666200da371bb51726bb336",
        "sha256": "e6b7462ec66d485b1b10936e3f86601c91e900dddad9b8b9e19b789c3e183d25",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "b9dde9083a6b228929913b577f2bfc2e2e9e50a61638a0dbe68bc7fccf6cd221"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "3d56b0f14b97755ecdfae2011f011165a6837c2caa363a9cbfaedc0dfcd53d0c"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "b91aba4447771e7dfd149515067ee64e1303ae44c6669d65504f17899070f331"},
        },
    },
    {
        "filename": "f2450_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr0_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 0,
        "size_bytes": 191552426,
        "md5": "8ef953dfadfb7363d70924f79f4755bc",
        "sha256": "e62681b28bb02c51671ffe3f3c36030d7a0e0839440adbf641a8e63f53b95ea9",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "111e8a12b35d5529e8e49c781de80d2ad05172e73ef2e65f747229ad0f63bf07"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "14cdb93e54351c5d9b2ac254a4f670863dd4a83bfa7ed48695186d64bf2460d2"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "7c390cdf2a55f800c2a1ad46e0cdf732630cc9cc3a48ab763d6f39184c3ff615"},
        },
    },
    {
        "filename": "f708_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr20_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 20,
        "size_bytes": 212954092,
        "md5": "afbd783156079f3ab2b1a33690540bad",
        "sha256": "1732da579f2e98ade65e2570f4141e9b881d6c891f8dec83a3bd723e05f8d1af",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a4569e6bce65a47dc227470f36b7d1607db32a1049011bda3440d296cc1b21e5"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "1f2b2b22c2345df8b2e67f0a7a44e525a65615723cf878eb63f231dc302e2436"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "8d05494ec48bae4456f4f96cbfb3a648fc81c9bf161388432e82d05f89b8a361"},
        },
    },
    {
        "filename": "f708_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr10_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 10,
        "size_bytes": 198519362,
        "md5": "fe9ceeb85392f1476289214736640b1c",
        "sha256": "b688f19db545edeef24af628283ff5080b7bd32f0277fd593b44a24da5a791f9",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "e9133b9cf31bcee6c36eb5f10e4da78cf60129f2fce22de1f0d9e4cd3917b0c3"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "962c8e5a41d8d4b6827dc8c6ef09bb7a931d1ae4c19c89203794bf1b482cb21e"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "6a7cd6c531d9125a2baa66c391de06d83d1079deebb4cf240a6bf2755051c1d9"},
        },
    },
    {
        "filename": "f708_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr0_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 0,
        "size_bytes": 190371348,
        "md5": "d971092391ddccc893791ece9ad71bf3",
        "sha256": "adc850b02149e0cf9a6f27bdba2c93a3290fdf16d2d414725dbcaccca7a22a00",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "4312bb95378d0d9c4fe67f57267a256de9834459a59e6d636e58edcd8ab52a6d"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "6f67ca836a315aafa5e00328a54fae5a0e350dac96d4e9070012205fb24062dd"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "1b7815e226e3ececc50f9a9027a5e15f3fbd09514bb86889b912ba51596c4b85"},
        },
    },
    {
        "filename": "f433_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr20_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 20,
        "size_bytes": 211968737,
        "md5": "8dbb0ce9f09b683de8dbc9dc76997b74",
        "sha256": "c6e28fae2d06e1038ae990fdb231b39b7cd37322fe5eb498a1f755766abcd512",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8c9a105246bcb27949fd1717029edaa1079264868316248fbab8825fa1087ab2"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "058e3fa662941648faac9512e2b0695a81fd5d9cdcab62bfb6c245d394c4a01c"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "379753c9b2f1d151d3228b7d8661a9d79e2ac182116ceb22f5b26275483b3e17"},
        },
    },
    {
        "filename": "f433_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr10_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 10,
        "size_bytes": 199205337,
        "md5": "8e8fa8e8822eab62320a0baebb87e7c5",
        "sha256": "2067a4fd82671d0b86fd47e8f3fab5f87573e8653c7dbc6321dcba23a701b5d4",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a68edb1cd13814063a1c2036722cc57209f60868a4517dfe884eefd31cf51c42"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "d635beed43c87efd7cbf6988157296d8259c556c34c6c3594d345eec539bf085"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "103d64391bd4dd85ac249bc6f75a5503c92a365e99e7d43bbe4a7d8774786b00"},
        },
    },
    {
        "filename": "f433_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr0_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 0,
        "size_bytes": 193721020,
        "md5": "9cc2a791ef1b07977b5c43760a06bda8",
        "sha256": "c7b7bcba97151ad4dadebe3c21fb3503d44126c6912e56e024a19d303b17ef2f",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8f9b54ba0153ae3f9e53a6ba6e231708934c7a1b42957120ee122ebe5ec9aeca"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "84f47c34c7daac92412742af7133ac984b2e713408b8b6e57665708c07a64e43"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "98d19a36fbc990d9fd7214eb7855482703f76650b5fca7ef1d513730e9541d54"},
        },
    },
]


def file_hashes(path: Path) -> tuple[int, str, str]:
    h_md5 = hashlib.md5()
    h_sha256 = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            total += len(chunk)
            h_md5.update(chunk)
            h_sha256.update(chunk)
    return total, h_md5.hexdigest(), h_sha256.hexdigest()


def download_and_verify(raw_dir: Path, work_dir: Path, extract: bool = True) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    src_dir = work_dir / "sources"
    data_dir = work_dir / "data"
    if extract:
        src_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    report = {"zenodo_record": ZENODO_RECORD_ID, "doi": f"10.5281/zenodo.{ZENODO_RECORD_ID}", "files": []}
    for item in MANIFEST_FILES:
        fn = item["filename"]
        dst = raw_dir / fn
        url = f"{ZENODO_BASE_URL}/{fn}/content"
        if not dst.exists():
            print(f"[download] {fn} <- {url}")
            urllib.request.urlretrieve(url, dst)
        sz, md5, sha256 = file_hashes(dst)
        assert sz == item["size_bytes"], f"Size mismatch for {fn}: {sz} != {item['size_bytes']}"
        assert md5 == item["md5"], f"MD5 mismatch for {fn}: {md5} != {item['md5']}"
        assert sha256 == item["sha256"], f"SHA256 mismatch for {fn}: {sha256} != {item['sha256']}"
        print(f"[verified] {fn} size={sz} sha256={sha256[:16]}...")

        if extract and fn.endswith(".tar.gz"):
            target = src_dir if item["role"] == "tx_source_bits" else data_dir
            with tarfile.open(dst, "r:gz") as tf:
                tf.extractall(target)
            for rel, meta in item.get("extracted_files", {}).items():
                ep = (src_dir / rel) if item["role"] == "tx_source_bits" else (data_dir / item["capture_id"] / rel)
                esz, _, esha = file_hashes(ep)
                assert esz == meta["size_bytes"], f"Extracted size mismatch: {ep}"
                assert esha == meta["sha256"], f"Extracted sha256 mismatch: {ep}"
        report["files"].append(item)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify CTTC QPSK dataset archives.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/raw"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/work"))
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--write-manifest", type=Path, default=None)
    args = parser.parse_args()
    rep = download_and_verify(args.raw_dir, args.work_dir, extract=not args.no_extract)
    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
