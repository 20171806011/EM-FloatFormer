"""Unified CTTC Indoor QPSK real-data adapter with strict zero-leakage split & packet indexing.

Reads raw pre-RRC (`file_1.dat`) and official post-RRC (`file_2.dat`) complex64 IQ streams plus
`source_QPSK.dat` and `config.txt`, performs reproducible waveform anchoring, fractional-SFO
packet slicing, and builds strictly disjoint train/dev/test target & reference pools.

Leakage firewall:
  1. Source-symbol index intervals (mod 500,000) are strictly disjoint across train/dev/test
     and between target packets and reference-pool packets, separated by 4,096-symbol guard bands.
  2. Selected packets within each loop are spaced by stride >= 2,048 symbols (gap >= 1,792 symbols
     > 28x RRC filter length), eliminating filter-tail and short-window overlap.
  3. Holdout verification pilots (16 symbols/pkt) and payload symbols (224 symbols/pkt) are never
     exposed to receivers; only `y_iq_4x`, `x_known` (sync+pilots), and `bg_noise_var` are visible.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import signal

from .evaluator import bits_to_qpsk

# --- Fixed physical waveform constants (from CTTC README.pdf & config.txt) ---
FS_RX = 2_000_000.0
FS_SYM = 500_000.0
SPS_NOM = 4
RRC_ALPHA = 0.2
RRC_NTAPS = 256
SOURCE_N_BITS = 1_000_000
SOURCE_N_SYMS = 500_000

# --- Standardized packet structure (256 symbols = 512 bits = 0.512 ms) ---
# Compact realistic pilot budget: 6 preamble sync + 4 sparse mid/tail pilots = 10 known pilots (3.9%)
PKT_LEN = 256
PAD_SYMS = 8  # 8 symbols (=32 RX samples) padding on each side for timing & multi-tap EQ
SYNC_IDX = np.arange(0, 6, dtype=np.int64)
PILOT_IDX = np.array([126, 127, 254, 255], dtype=np.int64)
KNOWN_IDX = np.concatenate([SYNC_IDX, PILOT_IDX])  # 10 known pilot symbols spanning start/mid/end
HOLDOUT_IDX = np.concatenate([np.arange(40, 48, dtype=np.int64), np.arange(200, 208, dtype=np.int64)])
_NON_PAYLOAD = set(KNOWN_IDX.tolist()) | set(HOLDOUT_IDX.tolist())
PAYLOAD_IDX = np.array([i for i in range(PKT_LEN) if i not in _NON_PAYLOAD], dtype=np.int64)

assert len(SYNC_IDX) == 6
assert len(PILOT_IDX) == 4
assert len(KNOWN_IDX) == 10
assert len(HOLDOUT_IDX) == 16
assert len(PAYLOAD_IDX) == 230
assert len(set(KNOWN_IDX) | set(HOLDOUT_IDX) | set(PAYLOAD_IDX)) == PKT_LEN

# --- Strict disjoint source-symbol intervals (mod 500,000) with 4,096-symbol guard bands ---
SPLIT_SRC_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "train": {"target": (0, 160_000), "ref": (164_096, 200_000)},
    "dev": {"target": (204_096, 290_000), "ref": (294_096, 330_000)},
    "test": {"target": (334_096, 455_000), "ref": (459_096, 495_904)},
}

ALL_CAPTURES: tuple[tuple[str, int, int, float], ...] = (
    ("f2450_snr20_QPSK", 2450, 20, -1240.0),
    ("f2450_snr10_QPSK", 2450, 10, -1175.0),
    ("f2450_snr0_QPSK", 2450, 0, -1280.0),
    ("f708_snr20_QPSK", 708, 20, -370.0),
    ("f708_snr10_QPSK", 708, 10, -375.0),
    ("f708_snr0_QPSK", 708, 0, -370.0),
    ("f433_snr20_QPSK", 433, 20, -166.0),
    ("f433_snr10_QPSK", 433, 10, -218.0),
    ("f433_snr0_QPSK", 433, 0, -157.0),
)


def make_rrc_taps(num_taps: int = RRC_NTAPS, alpha: float = RRC_ALPHA, sps: int = SPS_NOM) -> np.ndarray:
    """Design unit-energy Root Raised Cosine (RRC) FIR filter matching GNU Radio specification."""
    t = (np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0) / float(sps)
    h = np.zeros(num_taps, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - alpha + 4.0 * alpha / math.pi
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-12:
            h[i] = (alpha / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * alpha))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * alpha))
            )
        else:
            num = math.sin(math.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * math.cos(math.pi * ti * (1.0 + alpha))
            den = math.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    h /= math.sqrt(float(np.sum(h ** 2)))
    return h


def load_source_qpsk(sources_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `source_QPSK.dat` and return (bits_uint8 [1M], syms_even [500k], syms_odd [500k]).

    Empirically verified on Zenodo 4895133: 7 of 9 QPSK captures use even bit pairing
    (bits[0::2], bits[1::2]), whereas `f433_snr20_QPSK` and `f433_snr0_QPSK` were captured
    with a 1-bit odd offset (bits[1::2], bits[2::2] wrapped mod 1M).
    """
    p = sources_dir / "source_QPSK.dat"
    bits = np.fromfile(p, dtype=np.uint8)
    if len(bits) != SOURCE_N_BITS:
        raise ValueError(f"Expected {SOURCE_N_BITS} bits in {p}, got {len(bits)}")
    syms_even = bits_to_qpsk(bits)
    syms_odd = bits_to_qpsk(np.roll(bits, -1))
    return bits, syms_even, syms_odd


def parse_config_txt(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        k, v = line.split(",", 1)
        cfg[k.strip()] = v.strip()
    return cfg


@dataclass
class CaptureAnchor:
    """Offline capture-level physical characterization & continuous stream anchor."""
    capture_id: str
    freq_mhz: int
    nom_snr_db: int
    timestamp: str
    n_samples: int
    bg_noise_var: float
    sig_total_pow: float
    measured_snr_db: float
    anchor_rx_pos: float  # exact fractional RX sample index of symbol `anchor_src_sym` near t=2.0s
    anchor_src_sym: int
    sps_eff: float
    sfo_ppm: float
    offline_cfo_med_hz: float
    offline_cfo_min_hz: float
    offline_cfo_max_hz: float
    secondary_spur_ratio: float
    secondary_spur_offset_hz: float
    bit_alignment: int = 0  # 0=even (bits[0::2],bits[1::2]), 1=odd (1-bit shifted)
    lock_norm_corr_mean: float = 0.0
    stream_source: str = "file_2.dat"


@dataclass
class PacketSample:
    """Single standardized real-data QPSK evaluation packet with strict receiver/evaluator separation."""
    # Public identifiers & metadata
    unit_id: str
    capture_id: str
    freq_mhz: int
    nom_snr_db: int
    cluster_id: str
    loop_id: int
    split: str
    role: str  # "target" or "ref"
    rx_start_sample: int
    frac_rx_offset: float  # sub-sample residual inside [0, 1) from stream anchor (offline meta only)
    src_sym_start: int
    time_sec: float
    bg_noise_var: float
    # Receiver-visible waveform & known pilot inputs ONLY
    y_iq_4x: np.ndarray  # complex64, shape ((PKT_LEN + 2*PAD_SYMS) * 4,)
    x_known: np.ndarray  # complex128, shape (16,) corresponding to KNOWN_IDX
    # Evaluator-only ground truth & offline reference (NEVER passed to receivers)
    x_all_true: np.ndarray  # complex128, shape (256,)
    h_offline_ref: np.ndarray  # complex128, shape (256,)
    offline_cfo_hz: float
    offline_sig_pow: float
    perturbation_tag: str = "original"
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _parabolic_peak(y_m1: float, y_0: float, y_p1: float) -> float:
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))


def anchor_capture(
    data_dir: Path,
    capture_id: str,
    freq_mhz: int,
    nom_snr_db: int,
    cfo_hint_hz: float,
    sym_ref_even: np.ndarray,
    sym_ref_odd: np.ndarray | None = None,
    *,
    stream_file: str = "file_2.dat",
) -> CaptureAnchor:
    """Measure background noise, detect bit alignment (even vs odd), lock loop, and extract exact SFO."""
    cap_dir = data_dir / capture_id
    cfg = parse_config_txt(cap_dir / "config.txt")
    x = np.memmap(cap_dir / stream_file, dtype=np.complex64, mode="r")
    n_samples = len(x)

    # 1) Robust silence noise floor estimation in t in [0.10s, 0.85s] using 10ms blocks
    blk = 20_000
    sil_pwrs = np.array(
        [float(np.mean(np.abs(x[i * blk : (i + 1) * blk]) ** 2)) for i in range(10, 85)],
        dtype=np.float64,
    )
    # Use lower 60% median to stay immune to transient glitches (e.g. f2450_snr10 block 7..9)
    bg_noise_var = float(np.median(np.sort(sil_pwrs)[:45]))
    sig_pwrs = np.array(
        [float(np.mean(np.abs(x[i * blk : (i + 1) * blk]) ** 2)) for i in range(200, 950, 5)],
        dtype=np.float64,
    )
    sig_total_pow = float(np.median(sig_pwrs))
    measured_snr_db = 10.0 * math.log10(max(sig_total_pow - bg_noise_var, 1e-18) / max(bg_noise_var, 1e-18))

    # 2) Detect even (0) vs odd (1) bit alignment at t=4.0s via FFT cross-correlation
    nfft_ref = 1048576
    f_ref_even = np.fft.fft(sym_ref_even, n=nfft_ref)
    cand_refs: list[tuple[int, np.ndarray, np.ndarray]] = [(0, sym_ref_even, f_ref_even)]
    if sym_ref_odd is not None:
        cand_refs.append((1, sym_ref_odd, np.fft.fft(sym_ref_odd, n=nfft_ref)))

    L_lock = 4096
    n_idx = np.arange(L_lock, dtype=np.float64)
    s_probe = int(4.0 * FS_RX)
    best_align = 0
    best_align_v = -1.0
    best_sym_ref = sym_ref_even
    best_f_ref = f_ref_even
    for al_id, s_r, f_r in cand_refs:
        for ph in range(4):
            q = np.asarray(x[s_probe + ph : s_probe + ph + 4 * L_lock : 4], dtype=np.complex128)
            q = (q - q.mean()) * np.exp(-2j * math.pi * cfo_hint_hz * n_idx / FS_SYM)
            qn = float(np.linalg.norm(q))
            corr = np.fft.ifft(f_r * np.conj(np.fft.fft(q, n=nfft_ref)))[:SOURCE_N_SYMS]
            v = float(np.max(np.abs(corr))) / (qn * math.sqrt(L_lock) + 1e-30)
            if v > best_align_v:
                best_align_v = v
                best_align = al_id
                best_sym_ref = s_r
                best_f_ref = f_r

    # 3) Multi-checkpoint FFT lock at t = 2.0, 4.0, 6.0, 8.0, 10.0 s using locked bit alignment
    locks: list[tuple[float, float, int, float, float]] = []
    spur_ratio = 0.0
    spur_offset = 0.0

    for t_s in (2.0, 4.0, 6.0, 8.0, 10.0):
        s0 = int(t_s * FS_RX)
        if s0 + 4 * 16384 + 64 >= n_samples:
            continue
        best_v, best_ph, best_lag = -1.0, 0, 0
        for ph in range(4):
            q = np.asarray(x[s0 + ph : s0 + ph + 4 * L_lock : 4], dtype=np.complex128)
            q = (q - q.mean()) * np.exp(-2j * math.pi * cfo_hint_hz * n_idx / FS_SYM)
            qn = float(np.linalg.norm(q))
            corr = np.fft.ifft(best_f_ref * np.conj(np.fft.fft(q, n=nfft_ref)))[:SOURCE_N_SYMS]
            mag = np.abs(corr) / (qn * math.sqrt(L_lock) + 1e-30)
            lag = int(np.argmax(mag))
            if float(mag[lag]) > best_v:
                best_v, best_ph, best_lag = float(mag[lag]), ph, lag
        if best_v < 0.35:
            raise RuntimeError(f"Failed to lock {capture_id} at t={t_s}s: max norm_corr={best_v:.4f}")

        # High-resolution ML CFO over 16384 symbols
        Ls = 16384
        ql = np.asarray(x[s0 + best_ph : s0 + best_ph + 4 * Ls : 4], dtype=np.complex128)
        rl = best_sym_ref[(best_lag + np.arange(Ls)) % SOURCE_N_SYMS]
        z = ql * np.conj(rl)
        nfft_z = 262144
        sz = np.abs(np.fft.fft(z * np.hanning(Ls), n=nfft_z))
        fz = np.fft.fftfreq(nfft_z, d=1.0 / FS_SYM)
        pk = int(np.argmax(sz))
        cfo_ml = float(fz[pk])

        if abs(t_s - 4.0) < 0.1:
            sz2 = sz.copy()
            sz2[np.abs(fz - cfo_ml) < 60.0] = 0.0
            pk2 = int(np.argmax(sz2))
            spur_ratio = float(sz2[pk2] / (sz[pk] + 1e-30))
            spur_offset = float(fz[pk2] - cfo_ml)

        # Sub-sample timing refinement over +-4 RX samples
        taps = np.empty(9, dtype=np.float64)
        for k, dt in enumerate(range(-4, 5)):
            qt = np.asarray(x[s0 + best_ph + dt : s0 + best_ph + dt + 4 * L_lock : 4], dtype=np.complex128)
            qt = (qt - qt.mean()) * np.exp(-2j * math.pi * cfo_ml * n_idx / FS_SYM)
            taps[k] = abs(np.mean(qt * np.conj(rl[:L_lock])))
        pt = int(np.argmax(taps))
        fdt = _parabolic_peak(taps[pt - 1], taps[pt], taps[pt + 1]) if 0 < pt < 8 else 0.0
        abs_rx = float(s0 + best_ph + (pt - 4)) + fdt
        locks.append((t_s, abs_rx, best_lag, cfo_ml, best_v))

    # Linear fit of (actual_sym_advance -> abs_rx) across all 5 checkpoints for sub-0.01 ppm SFO precision
    sym_adv = [0.0]
    rx_pos = [locks[0][1]]
    for i in range(1, len(locks)):
        d_rx = locks[i][1] - locks[0][1]
        n_loops = round((d_rx / 4.0 - (locks[i][2] - locks[0][2])) / SOURCE_N_SYMS)
        act_syms = n_loops * SOURCE_N_SYMS + (locks[i][2] - locks[0][2])
        sym_adv.append(float(act_syms))
        rx_pos.append(locks[i][1])
    slope, intercept = np.polyfit(np.array(sym_adv), np.array(rx_pos), 1)
    sps_eff = float(slope)
    sfo_ppm = (sps_eff / 4.0 - 1.0) * 1e6
    cfos = [lk[3] for lk in locks]
    corrs = [lk[4] for lk in locks]

    return CaptureAnchor(
        capture_id=capture_id,
        freq_mhz=freq_mhz,
        nom_snr_db=nom_snr_db,
        timestamp=cfg.get("time", ""),
        n_samples=n_samples,
        bg_noise_var=bg_noise_var,
        sig_total_pow=sig_total_pow,
        measured_snr_db=measured_snr_db,
        anchor_rx_pos=float(intercept),
        anchor_src_sym=locks[0][2],
        sps_eff=sps_eff,
        sfo_ppm=sfo_ppm,
        offline_cfo_med_hz=float(np.median(cfos)),
        offline_cfo_min_hz=float(np.min(cfos)),
        offline_cfo_max_hz=float(np.max(cfos)),
        secondary_spur_ratio=spur_ratio,
        secondary_spur_offset_hz=spur_offset,
        bit_alignment=best_align,
        lock_norm_corr_mean=float(np.mean(corrs)),
        stream_source=stream_file,
    )


def _offline_reference_channel(
    y_iq_4x: np.ndarray,
    frac_offset: float,
    x_all_true: np.ndarray,
    cfo_hint_hz: float,
) -> tuple[np.ndarray, float, float]:
    """Compute high-precision offline reference channel h(n) over all 256 symbols (EVALUATOR ONLY).

    Uses cubic fractional resampling at true stream offset, ML CFO over all 256 known symbols,
    and gentle 2nd-order complex polynomial fit in time to track real intra-packet drift/beat.
    """
    from scipy.interpolate import CubicSpline

    t_rx = np.arange(len(y_iq_4x), dtype=np.float64)
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)
    idx = (PAD_SYMS * 4) + frac_offset + 4.0 * np.arange(PKT_LEN, dtype=np.float64)
    y_sym = cs_re(idx) + 1j * cs_im(idx)

    z = y_sym * np.conj(x_all_true)
    # Fine CFO around cfo_hint_hz +- 350 Hz (covers 1510 Hz beat instantaneous CFO on f708)
    n_grid = np.arange(PKT_LEN, dtype=np.float64)
    f_cands = np.linspace(cfo_hint_hz - 400.0, cfo_hint_hz + 400.0, 161)
    rot = np.exp(-2j * math.pi * f_cands[:, None] * n_grid[None, :] / FS_SYM)
    scores = np.abs(rot @ z)
    f_best = float(f_cands[int(np.argmax(scores))])

    # Complex quadratic envelope after derotating by f_best (tracks amplitude/phase beat smoothly)
    z_d = z * np.exp(-2j * math.pi * f_best * n_grid / FS_SYM)
    t_norm = (n_grid - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    B = np.column_stack([np.ones(PKT_LEN), t_norm, t_norm ** 2])
    coef, _, _, _ = np.linalg.lstsq(B, z_d, rcond=None)
    env = B @ coef
    h_ref = env * np.exp(2j * math.pi * f_best * n_grid / FS_SYM)
    sig_pow = float(np.mean(np.abs(h_ref) ** 2))
    return h_ref, f_best, sig_pow


def extract_packets_for_capture(
    data_dir: Path,
    anchor: CaptureAnchor,
    sym_ref_even: np.ndarray,
    sym_ref_odd: np.ndarray | None = None,
    *,
    per_loop_counts: dict[str, tuple[int, int]] | None = None,
    use_raw_file1_rrc: bool = False,
    n_loops_use: int = 8,
) -> dict[str, dict[str, list[PacketSample]]]:
    """Slice strictly disjoint train/dev/test target & reference packets for one capture.

    Args:
        per_loop_counts: mapping split -> (n_target_per_loop, n_ref_per_loop).
            Defaults to train:(20, 8), dev:(12, 8), test:(20, 8) across 8 loops.
        use_raw_file1_rrc: if True, reads `file_1.dat` and applies our transparent 256-tap RRC
            matched filter locally around each packet (for end-to-end validation against file_2.dat).
    """
    sym_ref = sym_ref_odd if (anchor.bit_alignment == 1 and sym_ref_odd is not None) else sym_ref_even
    if per_loop_counts is None:
        per_loop_counts = {"train": (20, 8), "dev": (12, 8), "test": (20, 8)}

    cap_dir = data_dir / anchor.capture_id
    src_fname = "file_1.dat" if use_raw_file1_rrc else "file_2.dat"
    x_mm = np.memmap(cap_dir / src_fname, dtype=np.complex64, mode="r")
    n_tot = len(x_mm)
    h_rrc = make_rrc_taps() if use_raw_file1_rrc else None

    # Find the earliest loop where src_sym == 0 occurs after t = 1.45s (rx >= 2,900,000)
    # anchor_rx_pos corresponds to anchor_src_sym at ~t=2.0s.
    # Advance to next src_sym = 0 or back by anchor_src_sym:
    rx_loop0_start = anchor.anchor_rx_pos - anchor.anchor_src_sym * anchor.sps_eff
    while rx_loop0_start < 2_900_000.0:
        rx_loop0_start += SOURCE_N_SYMS * anchor.sps_eff

    out: dict[str, dict[str, list[PacketSample]]] = {
        sp: {"target": [], "ref": []} for sp in ("train", "dev", "test")
    }
    pkt_rx_len = (PKT_LEN + 2 * PAD_SYMS) * SPS_NOM
    rrc_margin = 512 if use_raw_file1_rrc else 0

    for loop_id in range(n_loops_use):
        rx_base = rx_loop0_start + loop_id * SOURCE_N_SYMS * anchor.sps_eff
        cluster_id = f"{anchor.capture_id}__L{loop_id:02d}"
        for split in ("train", "dev", "test"):
            n_tgt, n_ref = per_loop_counts[split]
            for role, count in (("target", n_tgt), ("ref", n_ref)):
                if count <= 0:
                    continue
                lo, hi = SPLIT_SRC_RANGES[split][role]
                usable = hi - lo - PKT_LEN
                stride = usable // count
                assert stride >= 2048, f"Stride too small ({stride}) for zero-overlap guarantee!"
                for k in range(count):
                    src_start = lo + k * stride + 128
                    rx_exact = rx_base + src_start * anchor.sps_eff
                    rx_int = int(math.floor(rx_exact))
                    frac = float(rx_exact - rx_int)
                    win_start = rx_int - PAD_SYMS * SPS_NOM
                    win_end = win_start + pkt_rx_len
                    if win_start - rrc_margin < 0 or win_end + rrc_margin >= n_tot:
                        continue

                    if use_raw_file1_rrc:
                        raw_seg = np.asarray(
                            x_mm[win_start - rrc_margin : win_end + rrc_margin], dtype=np.complex128
                        )
                        # GNU Radio's causal RRC filter in file_2.dat ALREADY carries the 127.5-sample
                        # filter group delay relative to file_1.dat; empirical cross-correlation verified
                        # lfilter(h_rrc, [1.0], file_1)[n] aligns with file_2[n+1] (|corr|=0.9775).
                        filt = signal.lfilter(h_rrc, [1.0], raw_seg)
                        y_seg = filt[rrc_margin - 1 : rrc_margin - 1 + pkt_rx_len].astype(np.complex64)
                    else:
                        y_seg = np.asarray(x_mm[win_start:win_end], dtype=np.complex64).copy()

                    x_true = sym_ref[src_start : src_start + PKT_LEN].copy()
                    x_known = x_true[KNOWN_IDX].copy()
                    h_ref, cfo_off, sig_pow = _offline_reference_channel(
                        y_seg, frac, x_true, anchor.offline_cfo_med_hz
                    )
                    uid = f"{anchor.capture_id}__L{loop_id:02d}__{split[0].upper()}{role[0].upper()}{k:03d}__S{src_start:06d}"
                    out[split][role].append(
                        PacketSample(
                            unit_id=uid,
                            capture_id=anchor.capture_id,
                            freq_mhz=anchor.freq_mhz,
                            nom_snr_db=anchor.nom_snr_db,
                            cluster_id=cluster_id,
                            loop_id=loop_id,
                            split=split,
                            role=role,
                            rx_start_sample=rx_int,
                            frac_rx_offset=frac,
                            src_sym_start=src_start,
                            time_sec=rx_exact / FS_RX,
                            bg_noise_var=anchor.bg_noise_var,
                            y_iq_4x=y_seg,
                            x_known=x_known,
                            x_all_true=x_true,
                            h_offline_ref=h_ref,
                            offline_cfo_hz=cfo_off,
                            offline_sig_pow=sig_pow,
                        )
                    )
    return out


def verify_zero_leakage(dataset: dict[str, dict[str, list[PacketSample]]]) -> dict[str, Any]:
    """Assert strictly disjoint source-symbol intervals and unit IDs across splits and target/ref."""
    ranges_seen: dict[tuple[str, str], set[int]] = {}
    all_uids: set[str] = set()
    counts: dict[str, int] = {}
    for sp in ("train", "dev", "test"):
        for role in ("target", "ref"):
            pkts = dataset[sp][role]
            counts[f"{sp}_{role}"] = len(pkts)
            sym_set: set[int] = set()
            for p in pkts:
                assert p.unit_id not in all_uids, f"Duplicate unit_id {p.unit_id}"
                all_uids.add(p.unit_id)
                lo, hi = SPLIT_SRC_RANGES[sp][role]
                assert lo <= p.src_sym_start and (p.src_sym_start + PKT_LEN) <= hi, (
                    f"Packet {p.unit_id} src_start={p.src_sym_start} outside [{lo}, {hi})"
                )
                sym_set.update(range(p.src_sym_start, p.src_sym_start + PKT_LEN))
            ranges_seen[(sp, role)] = sym_set

    keys = list(ranges_seen.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = len(ranges_seen[keys[i]] & ranges_seen[keys[j]])
            assert ov == 0, f"Source-symbol leakage detected between {keys[i]} and {keys[j]}: {ov} symbols!"
    return {"leakage_verified": True, "unique_unit_ids": len(all_uids), "counts": counts}


def apply_controlled_stress(
    pkt: PacketSample,
    rng: np.random.Generator,
    *,
    mode: str,
) -> PacketSample:
    """Create a controlled physical stress-test variant on top of real measured IQ waveform.

    Preserves real hardware channel & ISI while injecting controlled impairments:
      - `stress_low_snr`: inject calibrated complex AWGN so effective SNR drops by ~8-14 dB
        (brings snr20/snr10 into critical ~0-2 dB regime and snr0 into ~-3 dB regime)
      - `stress_cfo_phase_jump`: inject random CFO perturbation in [-600, +600] Hz + mid-packet
        phase jump in {-pi/2, +pi/2, pi} at symbol 128 (with probability 0.5)
      - `stress_burst_impairment`: inject combined extra noise + CFO offset + 12% burst impulse noise
    """
    y = np.asarray(pkt.y_iq_4x, dtype=np.complex128).copy()
    h_ref = pkt.h_offline_ref.copy()
    sig_pow = max(pkt.offline_sig_pow, 1e-18)
    n_rx = len(y)
    t_rx = (np.arange(n_rx, dtype=np.float64) - PAD_SYMS * SPS_NOM) / FS_RX
    t_sym = np.arange(PKT_LEN, dtype=np.float64) / FS_SYM
    new_bg = pkt.bg_noise_var
    meta = dict(pkt.extra_meta)

    if mode == "stress_low_snr":
        # Target post-RRC SNR ~ -1.5 dB for snr0, ~ +1.0 dB for snr10, ~ +2.0 dB for snr20
        target_snr_db = {20: 2.0, 10: 1.0, 0: -1.5}.get(pkt.nom_snr_db, 1.0)
        target_nv = sig_pow / (10.0 ** (target_snr_db / 10.0))
        add_nv = max(0.0, target_nv - pkt.bg_noise_var)
        # Apply RRC-shaped noise correlation or white noise? Inject white complex noise in 4x domain
        # scaled so symbol-rate variance increases by add_nv!
        # Wait: since y is already post-RRC, adding white noise of variance add_nv directly adds add_nv
        # to every decimated symbol sample!
        noise = math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
        y = y + noise
        new_bg = pkt.bg_noise_var + add_nv
        meta["stress_target_snr_db"] = target_snr_db
    elif mode == "stress_cfo_phase_jump":
        df_hz = float(rng.uniform(-650.0, 650.0))
        # Apply CFO rotation
        y = y * np.exp(2j * math.pi * df_hz * t_rx)
        h_ref = h_ref * np.exp(2j * math.pi * df_hz * t_sym)
        # With 50% probability, apply a phase jump after symbol 96 (between sync and mid-pilot!)
        jump_rad = 0.0
        if rng.random() < 0.5:
            jump_rad = float(rng.choice([-math.pi / 2.0, math.pi / 2.0, math.pi]))
            jump_rx = (PAD_SYMS + 96) * SPS_NOM
            y[jump_rx:] *= np.exp(1j * jump_rad)
            h_ref[96:] *= np.exp(1j * jump_rad)
        # Mild extra noise if snr20 so it's non-trivial
        if pkt.nom_snr_db == 20:
            add_nv = max(0.0, sig_pow / (10.0 ** (6.0 / 10.0)) - pkt.bg_noise_var)
            y = y + math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
            new_bg = pkt.bg_noise_var + add_nv
        meta["stress_df_hz"] = df_hz
        meta["stress_jump_rad"] = jump_rad
    elif mode == "stress_burst_impairment":
        df_hz = float(rng.uniform(-450.0, 450.0))
        y = y * np.exp(2j * math.pi * df_hz * t_rx)
        h_ref = h_ref * np.exp(2j * math.pi * df_hz * t_sym)
        target_snr_db = {20: 3.5, 10: 2.0, 0: -0.5}.get(pkt.nom_snr_db, 2.0)
        add_nv = max(0.0, sig_pow / (10.0 ** (target_snr_db / 10.0)) - pkt.bg_noise_var)
        y = y + math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
        new_bg = pkt.bg_noise_var + add_nv
        # Impulsive burst interference on 16 contiguous payload symbols
        b_sym = int(rng.integers(56, 176))
        b_rx0 = (PAD_SYMS + b_sym) * SPS_NOM
        b_rx1 = b_rx0 + 16 * SPS_NOM
        burst_nv = 4.0 * sig_pow
        y[b_rx0:b_rx1] += math.sqrt(burst_nv / 2.0) * (
            rng.standard_normal(b_rx1 - b_rx0) + 1j * rng.standard_normal(b_rx1 - b_rx0)
        )
        meta["stress_df_hz"] = df_hz
        meta["stress_burst_sym"] = b_sym
    else:
        raise ValueError(f"Unknown stress mode: {mode}")

    return PacketSample(
        unit_id=f"{pkt.unit_id}__{mode}",
        capture_id=pkt.capture_id,
        freq_mhz=pkt.freq_mhz,
        nom_snr_db=pkt.nom_snr_db,
        cluster_id=pkt.cluster_id,
        loop_id=pkt.loop_id,
        split=pkt.split,
        role=pkt.role,
        rx_start_sample=pkt.rx_start_sample,
        frac_rx_offset=pkt.frac_rx_offset,
        src_sym_start=pkt.src_sym_start,
        time_sec=pkt.time_sec,
        bg_noise_var=new_bg,
        y_iq_4x=y.astype(np.complex64),
        x_known=pkt.x_known.copy(),
        x_all_true=pkt.x_all_true.copy(),
        h_offline_ref=h_ref,
        offline_cfo_hz=pkt.offline_cfo_hz + float(meta.get("stress_df_hz", 0.0)),
        offline_sig_pow=sig_pow,
        perturbation_tag=mode,
        extra_meta=meta,
    )


def save_anchor_index(anchors: Sequence[CaptureAnchor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [a.__dict__ for a in anchors]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
