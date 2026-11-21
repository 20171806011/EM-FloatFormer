# Data Sources Catalog (`research/realdata_sdr_ofdm`)

## 1. Selection Summary

- **Primary Dataset**: `rikluost/sdr_ofdm_dataset_v1` (`ofdm_valset_4_8_sdr.pth`) — over-the-air / hardware ADALM-Pluto (AD9364) 16-QAM CP-OFDM dataset (`742` TTIs verified, `5.1–30.4 dB` reported SINR), selected as Primary because it shares the exact 2D OFDM time-frequency pilot/payload grid paradigm of EM-FloatFormer, provides complete per-TTI ground-truth bitstreams and reported SINR, and downloads cleanly (`27.68 MB`) with an MIT-licensed companion PyTorch SDR implementation (`rikluost/ofdm-plutosdr-pytorch`).
- **Backup Dataset (Cataloged Only, Not Downloaded)**: `Frames of SDR ADALM PLUTO based QPSK Communication Link` (`QPSKlink.mat`, DOI `10.17632/k72b4pnc5g.1`), retained as a fully documented fallback per R2 requirements in case primary probe checks had failed (`T1–T5`). All primary probe triggers (`T1–T5`) passed with exit code `0`.

---

## 2. Primary & Backup Data Source Matrix (R2)

| R2 Field | Primary Dataset (`rikluost/sdr_ofdm_dataset_v1`) | Backup Dataset (`ADALM-PLUTO QPSK`) |
|---|---|---|
| **Dataset Name** | Measured Radio Channel Dataset for OFDM Receiver Validation (`sdr_ofdm_dataset_v1`) | Frames of SDR ADALM PLUTO based QPSK Communication Link (`QPSKlink`) |
| **Publisher / Authors** | Riku Luostari (`rikluost`, Aalto Univ. / Nokia) via GitHub | Latha V, Remya Krishna, Pallaviram Sure (M S Ramaiah Univ. of Applied Sciences) via Mendeley Data |
| **Version / Commit / Date** | `main` HEAD `6f9821d6390d212884fea3d6621267ebe70c7246` (`2025-06-10T10:13:46Z`); companion code `rikluost/ofdm-plutosdr-pytorch` `da69f52ae4a33b499a20844b72ff27bc16231bf7` (`2025-07-04T05:14:11Z`) | Version 1 (`published 2026-06-03T19:17:09.569Z`, `created 2026-06-02T10:21:15.689Z`) |
| **Official URL** | `https://github.com/rikluost/sdr_ofdm_dataset_v1` (companion: `https://github.com/rikluost/ofdm-plutosdr-pytorch`) | `https://data.mendeley.com/datasets/k72b4pnc5g` (API: `https://data.mendeley.com/public-api/datasets/k72b4pnc5g`) |
| **DOI** | None allocated (cite GitHub repo + commit SHA `6f9821d6`) | `10.17632/k72b4pnc5g.1` |
| **License** | Dataset repo: Unspecified (no `LICENSE` file); Companion code repo: `MIT License` | `CC BY 4.0` (`http://creativecommons.org/licenses/by/4.0`) |
| **Filename** | `ofdm_valset_4_8_sdr.pth` | `QPSKlink.mat` |
| **Format** | PyTorch pickle (`.pth`) of `config.CustomDataset` (`pdsch_iq` `complex64`, `labels` `float32`, `sinr` `float`) | MATLAB 5.0 MAT-file (`PCWIN64`, `Tx_data`, `Rx_data` complex128) |
| **Size** | `27,684,430` bytes (`27.68 MB`), `742` TTIs (`5600` bits/TTI = `4,155,200` total bits) | `822,762,822` bytes (`822.76 MB`), `7,096` frames |
| **Download Method** | Unauthenticated HTTP GET: `https://raw.githubusercontent.com/rikluost/sdr_ofdm_dataset_v1/main/ofdm_valset_4_8_sdr.pth` | Unauthenticated HTTP GET (302 -> S3): `https://data.mendeley.com/public-files/datasets/k72b4pnc5g/files/b494a472-ab2e-4d39-99ca-86866b76f159/file_downloaded` |
| **Waveform & Modulation** | CP-OFDM; **16-QAM** (`Qm=4`, non-Gray separable natural-binary `mapping_table(4)`, normalized by `sqrt(2.5)`) payload + **QPSK** pilots; `fc = 434–435 MHz` (README: `435 MHz`; `config.py` sim default: `3500 MHz`) | Single-carrier pulse-shaped **QPSK** (raised-cosine TX pulse shaping & RX matched filter); `fc = 850 MHz` |
| **Sampling / Frame / Pilot Params** | `N_fft = 128`, `SCS = 15 kHz`, `fs = 1.92 MHz`, `S = 14` symbols/TTI, `CP = 7` (`config.py`, README notes `6`), `F = 102` active bins (`13..114`), `13` left + `13` right guards (`0..12`, `115..127`), `DC` unmodulated at bin `64` (active `51`), `825 kHz` 5th-order Butterworth LPF. **14 QPSK pilots** in symbol `Sp = 2` at active indices `[0,8,16,24,32,40,48,56,64,72,80,88,96,101]` (`FFT bins [13,21,...,109,114]`), repeating `[-0.7-0.7j, -0.7+0.7j, 0.7-0.7j, 0.7+0.7j]` (`\|p\|^2 = 0.98`); `1400` payload REs/TTI (`5600` bits/TTI) | `126` symbols/frame = `26`-symbol Barker header + `100` QPSK payload symbols; `20` samples/symbol (`2520` TX samples, `7560` RX samples = `3x` capture window); `7,096` frames |
| **TX Records & Label Mapping** | `labels`: `[1400, 4]` binary bitstream (`0/1`) mapped in `(symbol, subcarrier)` row-major order where `OFDM_mask == 1` via `mapping_table(4)`; `pilot_set` known; `sinr` scalar (`5.1–30.4 dB`, mean `14.04 dB`, std `6.05 dB`) | `Tx_data`: `(126, 7096)` complex symbols (`Tx_data[0:26, :]` = constant Barker header; `Tx_data[26:126, :]` = QPSK payload symbols) |
| **Processing Stage** | **Post-sync, post-CP-removal, post-DFT frequency-domain grid** (`pdsch_iq` `[14, 128]`, centered DC at bin `64`, max-abs normalized in time and again after DFT) | **Matched-filtered complex time-domain frames** (`Rx_data` `(7560, 7096)`, unaligned in frame start, timing phase, CFO, and channel) |
| **Sessions / Devices / Scenarios / Payload Repetitions** | `742` TTIs; single ADALM-Pluto (AD9364) SDR setup with per-TTI randomized TX gain (`[-25, -3]` dB, `rx_gain = 8` dB); **0 duplicate payloads** (`742` unique `tx_bits_sha256` verified); high inter-TTI channel coherence (`h_pilot` lag-1 `0.847`, lag-32 `0.736`) with low SINR autocorr (`0.171` lag-1) | `7,096` frames; single ADALM-Pluto (`TX 0 dB`, `RX 20 dB`, `3 m` indoor lab); **duplicate payloads exist** (`Tx_data` payload hash grouping required if used) |
| **Raw SHA-256** | `e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058` (git blob SHA-1: `9b1aee8673ec48f42f52dc365fb01cc8b3e2b741`) | `c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7` (verified via Mendeley API metadata) |
| **Processed Checksums** | Saved in `/tmp/tw_realdata/data_proc/` (`all_frames.npz`: `46,519,970` B, SHA-256 `39f24f59892e3284037103d14895b6e9080008539924eddcfc6e62a5e0cfc206`; see Section 4) | Not downloaded (backup only) |

---

## 3. Companion Files & Verified Checksums (`manifests/raw_checksums.json`)

All raw downloads live strictly in Codespace `/tmp/tw_realdata/data_raw/` (excluded from git):

| File | Size (B) | SHA-256 | Git Blob SHA-1 |
|---|---:|---|---|
| `ofdm_valset_4_8_sdr.pth` | 27,684,430 | `e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058` | `9b1aee8673ec48f42f52dc365fb01cc8b3e2b741` |
| `sdr_ofdm_config.py` | 638 | `2f40ffee12f3f131ff40d99a71f0b812576aa2544570e8c34ed10309edf9f795` | `7e78d492d97ab33d1f8cd49e47a596098a659416` |
| `sdr_ofdm_README.md` | 9,298 | `046fd90ba0fdfca15955d831663416c1833ff8937cace785d3e1a4f47f4fd674` | `ddd306babb0f5d2ee0a02630aa359d2f0f5400ce` |
| `plutosdr_config.py` | 1,817 | `2af454acc2738d91b1edd349aa8371d63f2bcc55753e6062f1ae0c81f19887eb` | `951c53aa61e6b18d3138f21db8fcfe830ab7c2de` |
| `OFDM_SDR_Functions_torch_v2.py` | 31,831 | `224ea1407c64a22fa9e83718876e922e5d6d58c7064b084bf0d91c29a71b76c1` | `5da298973df992209ae8ef7cfe676b5b3717832f` |

---

## 4. Processed Dataset Artifacts (`manifests/proc_checksums.json`)

Raw data (`/tmp/tw_realdata/data_raw/`) and processed tensors (`/tmp/tw_realdata/data_proc/`) are kept strictly separated in Codespace:

| Artifact | Split | TTIs | Size (B) | SHA-256 |
|---|---|---:|---:|---|
| `all_frames.npz` | All | 742 | 46,519,970 | `39f24f59892e3284037103d14895b6e9080008539924eddcfc6e62a5e0cfc206` |
| `train.npz` | Train | 336 | 21,065,674 | `181ab829d5cdba94cb189c66506eaea2c07a9bbe36cb8e7dd3d0b6b7c0ca4414` |
| `dev.npz` | Dev | 92 | 5,769,434 | `ab7372295dcd221d9b08ee8acf57aaecc40021e130c8e9c6ff58c4e553dac495` |
| `test.npz` | Test | 138 | 8,653,450 | `c7f86cd8d6e5047761212031363fc6c0f6d2ae7debdad1f1458ada8f154c73d0` |
| `guard.npz` | Guard | 176 | 11,033,482 | `147dd9e257d461953b7e2645b5c40a9c95933bb415b9ed62d84526b8f1e399ee` |

---

## 5. Empirically Verified Signal Properties & Upstream Doc-vs-Code Errata

1. **Empirical Guard-Band Power Profile (64-Frame Probe)**:
   - Because `SDR_Pluto_v2.py` applies a 5th-order zero-phase Butterworth LPF at `fc_lpf = 825 kHz` (`fs = 1.92 MHz`, `|k - 64| <= 55`, i.e., `-3 dB` cutoff near bins `9` and `119`), outer guard bins `0..8` and `120..127` are attenuated to `mean = 0.000675` (down to `0.000304` at bin `0`).
   - Bins `9` (`0.00521`) and `119` (`0.00516`) sit right on the `-3 dB` LPF edge, while `12` (`0.01482`) and `115` (`0.01552`) sit immediately adjacent to active edge bins `13` (`0.3083`) and `114` (`0.3229`).
   - Therefore, our adapter (`SAFE_GUARD_FFT_INDICES = (10, 11, 117, 118)`) uses the symmetric flat-passband guard bins (`mean power ~ 0.0112`, yielding empirical active/guard SNR `0.3455 / 0.0112 ≈ 14.9 dB` consistent with `mean(sinr) = 14.04 dB`).
   - DC bin `64` exhibits significant LO leakage (`mean_dc_power = 0.0948` vs `mean_active_power = 0.3455`) and is excluded from both normalization `s0` and `est_n0`.
2. **Non-Gray Natural-Binary 16-QAM (`mapping_table(4)`)**:
   - Verified empirically on 64 frames: clean LS/ZF with non-Gray `mapping_table(4)` achieves `BER = 0.07073` (`25,349 / 358,400`), whereas 3GPP Gray-coded 16-QAM gives `BER = 0.26907` (`96,435 / 358,400`).
   - Full-dataset (`742` TTIs) clean LS/ZF gives `BER = 0.09403` (`390,727 / 4,155,200`) and `FER = 0.8221` (`610 / 742`).
3. **Companion `equalize_ZF` Unwrap Defect**:
   - Companion `OFDM_SDR_Functions_torch_v2.equalize_ZF` applies an erratic `|diff| > 1 rad` phase-unwrap (`BER = 0.37981` on the 64-frame probe); our adapter and M2 baselines replace it with exact complex division / LMMSE.
4. **CP & Carrier Frequency Documentation Drift**:
   - `sdr_ofdm_dataset_v1/README.md` states `CP = 6`, `fc = 435 MHz`, `"101 subcarriers including DC"`, and `"coaxial"` vs `"over-the-air"` wording; `ofdm-plutosdr-pytorch/config.py` uses `CP = 7`, `F = 102` active bins (`101` modulated + `1` DC), and `fc = 434–435 MHz` for PlutoSDR capture (`3500 MHz` only for simulation Doppler). Because `pdsch_iq` is post-CP-removal `[14, 128]`, tensor indexing is unaffected; `CP_LEN = 7` is used in continuous-time CFO perturbation hooks and noted as a parameter.
