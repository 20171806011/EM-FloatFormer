"""Explicit-N0 multi-SNR, multi-regime OFDM(A) -> CP-Single-Carrier(B) channel & tracker.

Design invariants:
- ZERO hardcoded N0 anywhere: every noise generation, OFDM tracking update, SC
  conditioning sweep, and SC readout takes an explicit noise variance ``n0``.
- Paired unit noise vectors across SNR values: for any fixed ``seed``, all
  ``snr_db`` values share identical physical channels, codewords, and unit-variance
  noise realizations scaled solely by ``sqrt(n0)``.
- Exact 9 dB ``continuous`` parity with ``p8c`` (commit ce2ff50).
- Exact complex gain & phase continuity across boundary/mid-frame Doppler shifts
  in ``partial_shift``, ``ood_strong_shift``, and ``ood_midframe_shift``.
- Strict public/hidden boundary: ``PublicObservation.y_b[PROBE_ROW]`` is always
  masked to ``complex(np.nan, np.nan)``.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from scipy.special import logsumexp

from p8.coding import qam_llrs
from p8.link import (
    EVEN,
    Channel,
    Traffic,
    clock,
    complex_normal,
    draw_channel,
    propagate as propagate_ofdm,
    stream,
    traffic,
    tx_grid as ofdm_tx_grid,
)
from p8.tracker import State, advance, evolution, observed_taps, prior
from p8c.link import PILOTS as SC_PILOTS, extra_rows, tx_grid as sc_tx_grid
from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    LINK_DURATION,
    N,
    OFDM_PILOT_ROWS,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    SC_LAYOUTS,
    SNR_GRID_DB,
    T,
    HiddenTruth,
    PublicObservation,
    n0_from_snr,
)


SWEEPS = 4
SAMPLES = np.arange(N, dtype=np.int64)
DELAYED_INDEX = (SAMPLES[None, :] - DELAYS[:, None]) % N

PARTIAL_SHIFT_RANGE = (0.035, 0.065)
OOD_STRONG_SHIFT_RANGE = (0.085, 0.120)
OOD_MIDFRAME_SHIFT_ROW = 6


@dataclass(frozen=True)
class PiecewiseChannel:
    """Two-segment physical channel with exact complex phase continuity at ``shift_time``.

    For ``t < shift_time``, ray gains follow ``pre``; for ``t >= shift_time``, ray
    ``changed_ray`` undergoes an instantaneous Doppler shift of ``delta_freq`` cycles/N
    while preserving exact complex gain (amplitude and phase) at ``t == shift_time``.
    """
    pre: Channel
    post: Channel
    shift_time: float
    shift_row: int
    changed_ray: int
    delta_freq: float


ChannelLike = Union[Channel, PiecewiseChannel]


def row_start_time_b(row: int) -> float:
    """Exact absolute time (in units of N samples) at the start of Frame B row's CP."""
    return float(FRAME_DURATION + int(row) * (N + CP) / N)


def make_shifted_channel(
    base: Channel,
    changed_ray: int,
    delta_freq: float,
    shift_row: int = 0,
) -> PiecewiseChannel:
    """Construct a phase-continuous Doppler-shifted channel starting at Frame B ``shift_row``."""
    ray = int(changed_ray)
    if ray < 0 or ray >= len(DELAYS):
        raise ValueError(f"changed_ray must be in 0..{len(DELAYS) - 1}, got {ray}")
    if shift_row < 0 or shift_row >= T:
        raise ValueError(f"shift_row must be in 0..{T - 1}, got {shift_row}")
    t_shift = row_start_time_b(shift_row)
    df = float(delta_freq)

    gain_post = np.asarray(base.gain, dtype=np.complex128).copy()
    freq_post = np.asarray(base.frequency, dtype=np.float64).copy()
    chirp_post = np.asarray(base.chirp, dtype=np.float64).copy()
    gain_post[ray] = gain_post[ray] * np.exp(-2j * np.pi * df * t_shift)
    freq_post[ray] = freq_post[ray] + df
    post = Channel(gain_post, freq_post, chirp_post)
    return PiecewiseChannel(
        pre=base,
        post=post,
        shift_time=t_shift,
        shift_row=int(shift_row),
        changed_ray=ray,
        delta_freq=df,
    )


def evaluate_ray_gains(channel: ChannelLike, times: np.ndarray) -> np.ndarray:
    """Evaluate exact complex ray gains ``g_l(t)`` at arbitrary absolute times ``t``.

    Args:
        channel: ``Channel`` or ``PiecewiseChannel``.
        times: Array of arbitrary shape ``(...)`` containing times in units of ``N`` samples.

    Returns:
        Complex128 array of shape ``(len(DELAYS), *times.shape)``.
    """
    t = np.asarray(times, dtype=np.float64)
    if isinstance(channel, PiecewiseChannel):
        base = channel.pre
        gain = np.asarray(base.gain, dtype=np.complex128)
        freq = np.asarray(base.frequency, dtype=np.float64)
        chirp = np.asarray(base.chirp, dtype=np.float64)
        expand = (slice(None),) + (None,) * t.ndim
        phase = freq[expand] * t[None, ...] + 0.5 * chirp[expand] * (t[None, ...] ** 2)
        mask = t >= channel.shift_time
        # Cancellation-free form: phase_post(t) = phase_pre(t) + delta_freq * (t - t_shift)
        # Guarantees exact 0.0 phase discontinuity at t == shift_time.
        ray = channel.changed_ray
        phase[ray] = np.where(
            mask,
            phase[ray] + channel.delta_freq * (t - channel.shift_time),
            phase[ray],
        )
        return gain[expand] * np.exp(2j * np.pi * phase)

    gain = np.asarray(channel.gain, dtype=np.complex128)
    freq = np.asarray(channel.frequency, dtype=np.float64)
    chirp = np.asarray(channel.chirp, dtype=np.float64)
    expand = (slice(None),) + (None,) * t.ndim
    phase = freq[expand] * t[None, ...] + 0.5 * chirp[expand] * (t[None, ...] ** 2)
    return gain[expand] * np.exp(2j * np.pi * phase)


def propagate_sc_from_gains(grid: np.ndarray, ray_gains: np.ndarray) -> np.ndarray:
    """Apply time-domain post-CP single-carrier channel defined by ``ray_gains [3, T, N]``."""
    x = np.asarray(grid, dtype=np.complex128)
    g = np.asarray(ray_gains, dtype=np.complex128)
    if x.ndim != 2 or x.shape[1] != N:
        raise ValueError(f"Expected single-carrier grid shape [*, {N}], got {x.shape}")
    if g.shape != (len(DELAYS), x.shape[0], N):
        raise ValueError(f"Expected ray_gains shape {(len(DELAYS), x.shape[0], N)}, got {g.shape}")
    received = np.zeros(x.shape, dtype=np.complex128)
    for ray_idx, delay in enumerate(DELAYS):
        received += g[ray_idx] * np.roll(x, int(delay), axis=-1)
    return received


def propagate_sc(grid: np.ndarray, channel: ChannelLike, start_row: int = T) -> np.ndarray:
    """Propagate a CP-SC grid through ``channel`` starting at ``start_row``."""
    x = np.asarray(grid, dtype=np.complex128)
    times = clock(len(x), start_row=start_row)
    ray_gains = evaluate_ray_gains(channel, times)
    return propagate_sc_from_gains(x, ray_gains)


def make_awgn_channel() -> Channel:
    """Single-ray static unit-gain AWGN diagnostic channel (DELAYS=[0], POWERS=[1], freq=0, chirp=0)."""
    gain = np.zeros(len(DELAYS), dtype=np.complex128)
    gain[0] = 1.0 + 0.0j
    return Channel(
        gain=gain,
        frequency=np.zeros(len(DELAYS), dtype=np.float64),
        chirp=np.zeros(len(DELAYS), dtype=np.float64),
    )


def _ofdm_noise(unit_time_noise: np.ndarray, n0: float) -> np.ndarray:
    """Frequency-domain OFDM noise scaled explicitly by ``sqrt(n0)``."""
    if n0 == 0.0:
        return np.zeros_like(unit_time_noise, dtype=np.complex128)
    return np.fft.fft(np.sqrt(float(n0)) * unit_time_noise, norm="ortho")


def _draw_shift_params(seed: int, regime: str) -> tuple[int, float, int]:
    """Deterministically draw ``(changed_ray, delta_freq, shift_row)`` from ``seed``."""
    rng = stream(seed, 9)
    changed_ray = int(rng.integers(0, len(DELAYS)))
    sign = int(rng.choice([-1, 1]))
    u = float(rng.uniform(0.0, 1.0))
    if regime == "partial_shift":
        lo, hi = PARTIAL_SHIFT_RANGE
        shift_row = 0
    elif regime == "ood_strong_shift":
        lo, hi = OOD_STRONG_SHIFT_RANGE
        shift_row = 0
    elif regime == "ood_midframe_shift":
        lo, hi = PARTIAL_SHIFT_RANGE
        shift_row = OOD_MIDFRAME_SHIFT_ROW
    else:
        raise ValueError(f"Not a Doppler-shift regime: {regime}")
    delta_freq = float(sign * (lo + (hi - lo) * u))
    return changed_ray, delta_freq, shift_row


def simulate_link(
    seed: int,
    snr_db: float = PRIMARY_SNR_DB,
    regime: str = "continuous",
    budget: str = "sparse",
    *,
    n0_override: Optional[float] = None,
) -> tuple[PublicObservation, HiddenTruth]:
    """Generate one OFDM(A) -> CP-SC(B) link with explicit ``n0`` and paired unit noise.

    For a fixed ``seed``, varying ``snr_db`` keeps true channels, payloads, and unit
    noise vectors strictly identical, scaling noise strictly by ``sqrt(n0)``.
    At ``snr_db=9.0`` and ``regime="continuous"``, waveforms match ``p8c.link.simulate``.
    """
    if regime not in ALL_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {ALL_REGIMES}")
    if budget not in SC_LAYOUTS:
        raise ValueError(f"Unknown budget {budget!r}; expected one of {tuple(SC_LAYOUTS)}")

    n0 = float(n0_from_snr(snr_db) if n0_override is None else n0_override)
    if n0 < 0.0:
        raise ValueError(f"n0 must be non-negative, got {n0}")

    training_rows = tuple(SC_LAYOUTS[budget])

    # Shared deterministic streams matching p8c.link.simulate(seed) indices 0..7.
    channel_0 = draw_channel(stream(seed, 0))
    a_words = traffic(stream(seed, 1))
    unit_noise_a = complex_normal(stream(seed, 2), (T, N))
    words_raw = traffic(stream(seed, 3))
    b_words = Traffic(words_raw.info[:18], words_raw.code[:18], words_raw.symbols[:18])
    unit_noise_b = complex_normal(stream(seed, 4), (T, N))

    changed_ray: Optional[int] = None
    delta_freq: float = 0.0
    shift_row: Optional[int] = None

    if regime == "continuous":
        channel_a: ChannelLike = channel_0
        channel_b: ChannelLike = channel_0
        y_a = propagate_ofdm(ofdm_tx_grid(a_words, OFDM_PILOT_ROWS), channel_a) + _ofdm_noise(unit_noise_a, n0)
    elif regime in ("partial_shift", "ood_strong_shift", "ood_midframe_shift"):
        changed_ray, delta_freq, shift_row = _draw_shift_params(seed, regime)
        channel_a = channel_0
        channel_b = make_shifted_channel(channel_0, changed_ray, delta_freq, shift_row=shift_row)
        y_a = propagate_ofdm(ofdm_tx_grid(a_words, OFDM_PILOT_ROWS), channel_a) + _ofdm_noise(unit_noise_a, n0)
    elif regime == "full_reset":
        channel_a = channel_0
        channel_b = draw_channel(stream(seed, 8))
        y_a = propagate_ofdm(ofdm_tx_grid(a_words, OFDM_PILOT_ROWS), channel_a) + _ofdm_noise(unit_noise_a, n0)
    elif regime == "donor_mismatch":
        donor_channel = draw_channel(stream(seed, 5))
        donor_words = traffic(stream(seed, 6))
        unit_noise_donor = complex_normal(stream(seed, 7), (T, N))
        channel_a = donor_channel
        channel_b = channel_0
        y_a = (
            propagate_ofdm(ofdm_tx_grid(donor_words, OFDM_PILOT_ROWS), donor_channel)
            + _ofdm_noise(unit_noise_donor, n0)
        )
    else:
        raise AssertionError(f"Unhandled regime: {regime}")

    grid_b = sc_tx_grid(b_words, training_rows)
    times_b = clock(T, start_row=T)
    h_b_true = evaluate_ray_gains(channel_b, times_b)
    y_b_full_clean = propagate_sc_from_gains(grid_b, h_b_true)
    noise_b = np.zeros((T, N), dtype=np.complex128) if n0 == 0.0 else np.sqrt(n0) * unit_noise_b
    y_b_noisy = y_b_full_clean + noise_b

    y_b_public = y_b_noisy.copy()
    y_b_public[PROBE_ROW] = complex(np.nan, np.nan)

    obs = PublicObservation(
        y_a=np.asarray(y_a, dtype=np.complex128),
        y_b=y_b_public,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        budget=budget,
    )
    truth = HiddenTruth(
        regime=regime,
        channel_a=channel_a,
        channel_b=channel_b,
        changed_ray=changed_ray,
        delta_freq=delta_freq,
        shift_row=shift_row,
        b_info=np.asarray(b_words.info, dtype=np.uint8),
        b_code=np.asarray(b_words.code, dtype=np.uint8),
        b_symbols=np.asarray(b_words.symbols, dtype=np.complex128),
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_full_clean[PROBE_ROW].copy(),
        y_b_full_clean=y_b_full_clean,
        h_b_true=h_b_true,
    )
    return obs, truth


def simulate_snr_sweep(
    seed: int,
    snr_grid_db: tuple[float, ...] = SNR_GRID_DB,
    regime: str = "continuous",
    budget: str = "sparse",
) -> dict[float, tuple[PublicObservation, HiddenTruth]]:
    """Generate paired observations across multiple SNRs sharing exact channels, traffic, and unit noise."""
    return {
        float(snr_db): simulate_link(seed, snr_db=float(snr_db), regime=regime, budget=budget)
        for snr_db in snr_grid_db
    }


def simulate_diagnostic(
    seed: int,
    kind: str = "awgn",
    snr_db: float = PRIMARY_SNR_DB,
    budget: str = "sparse",
) -> tuple[PublicObservation, HiddenTruth]:
    """Generate diagnostic link realizations: ``'awgn'``, ``'zero_noise'``, or ``'true_3ray'``."""
    if budget not in SC_LAYOUTS:
        raise ValueError(f"Unknown budget {budget!r}; expected one of {tuple(SC_LAYOUTS)}")
    if kind == "true_3ray":
        return simulate_link(seed, snr_db=snr_db, regime="continuous", budget=budget)
    if kind == "zero_noise":
        return simulate_link(seed, snr_db=snr_db, regime="continuous", budget=budget, n0_override=0.0)
    if kind != "awgn":
        raise ValueError(f"Unknown diagnostic kind {kind!r}; expected 'awgn', 'zero_noise', or 'true_3ray'")

    n0 = float(n0_from_snr(snr_db))
    training_rows = tuple(SC_LAYOUTS[budget])
    awgn_ch = make_awgn_channel()

    a_words = traffic(stream(seed, 1))
    unit_noise_a = complex_normal(stream(seed, 2), (T, N))
    words_raw = traffic(stream(seed, 3))
    b_words = Traffic(words_raw.info[:18], words_raw.code[:18], words_raw.symbols[:18])
    unit_noise_b = complex_normal(stream(seed, 4), (T, N))

    y_a = propagate_ofdm(ofdm_tx_grid(a_words, OFDM_PILOT_ROWS), awgn_ch) + _ofdm_noise(unit_noise_a, n0)
    grid_b = sc_tx_grid(b_words, training_rows)
    times_b = clock(T, start_row=T)
    h_b_true = evaluate_ray_gains(awgn_ch, times_b)
    y_b_full_clean = propagate_sc_from_gains(grid_b, h_b_true)
    y_b_noisy = y_b_full_clean + (np.sqrt(n0) * unit_noise_b if n0 > 0.0 else 0.0)

    y_b_public = y_b_noisy.copy()
    y_b_public[PROBE_ROW] = complex(np.nan, np.nan)

    obs = PublicObservation(
        y_a=np.asarray(y_a, dtype=np.complex128),
        y_b=y_b_public,
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        budget=budget,
    )
    truth = HiddenTruth(
        regime="continuous",
        channel_a=awgn_ch,
        channel_b=awgn_ch,
        changed_ray=None,
        delta_freq=0.0,
        shift_row=None,
        b_info=np.asarray(b_words.info, dtype=np.uint8),
        b_code=np.asarray(b_words.code, dtype=np.uint8),
        b_symbols=np.asarray(b_words.symbols, dtype=np.complex128),
        y_b_probe_noisy=y_b_noisy[PROBE_ROW].copy(),
        y_b_probe_clean=y_b_full_clean[PROBE_ROW].copy(),
        y_b_full_clean=y_b_full_clean,
        h_b_true=h_b_true,
    )
    return obs, truth


# ---------------------------------------------------------------------------
# Explicit-N0 Frame-A OFDM tracker & Frame-B CP-SC conditioner / readout
# ---------------------------------------------------------------------------


def tap_noise(n0: float) -> float:
    """Explicit-N0 comb-pilot tap observation noise variance (matches p8.tracker.TAP_NOISE at 9 dB)."""
    n0_val = float(n0)
    if n0_val < 0.0:
        raise ValueError(f"n0 must be non-negative, got {n0_val}")
    return float((n0_val + 1.0 - float(np.sinc(0.0625)) ** 2) / len(EVEN))


def update_ofdm_state(state: State, rows: tuple[int, ...], tap_obs: np.ndarray, n0: float) -> State:
    """Explicit-N0 Bayesian bank update on OFDM pilot rows (mirror of p8.tracker.update)."""
    if np.shape(tap_obs) != (len(rows), len(DELAYS)):
        raise ValueError("Expected one complex observation per pilot row and tap")
    row_noise = tap_noise(n0)
    signatures = evolution(state.epoch)[:, list(rows)].mean(axis=-1).T
    mean, variance, log_weight = state.mean, state.variance, state.log_weight
    for u, z in zip(signatures, tap_obs):
        innovation = z[:, None] - u * mean
        total = row_noise + (np.abs(u) ** 2) * variance
        log_weight = log_weight - np.log(total) - (np.abs(innovation) ** 2) / total
        log_weight = log_weight - logsumexp(log_weight, axis=1, keepdims=True)
        mean = mean + variance * u.conj() * innovation / total
        variance = variance * row_noise / total
    return State(state.epoch, log_weight, mean, variance)


def track_frame_a(y_a: np.ndarray, n0: float) -> State:
    """Track OFDM Frame A with explicit ``n0`` and transport posterior to Frame B origin."""
    taps = observed_taps(np.asarray(y_a, dtype=np.complex128), OFDM_PILOT_ROWS)
    post_a = update_ofdm_state(prior(), OFDM_PILOT_ROWS, taps, n0)
    return advance(post_a, FRAME_DURATION)


def clear_prior_b() -> State:
    """Uninformed ('clear') Frame B prior positioned at Frame B origin."""
    return advance(prior(), FRAME_DURATION)


def _validate_training_rows(rows) -> tuple[int, ...]:
    out = tuple(int(t) for t in rows)
    if len(set(out)) != len(out) or any(t < 0 or t >= T or t == PROBE_ROW for t in out):
        raise ValueError("Training rows must be distinct, in range, and exclude the withheld probe row")
    return out


@lru_cache(maxsize=8)
def _training_dictionary(epoch: float, rows: tuple[int, ...]):
    """Per-sample unit-gain responses of each delayed SC pilot to each Doppler/chirp node."""
    rotations = evolution(epoch)[:, list(rows)]
    pilots = SC_PILOTS[list(rows)]
    delayed = np.stack([np.roll(pilots, int(d), axis=-1) for d in DELAYS])
    design = (delayed[:, None] * rotations[None]).reshape(len(DELAYS), len(rotations), -1)
    return design, design.conj(), np.sum(np.abs(design) ** 2, axis=-1)


def condition_frame_b(
    prior_state: State,
    y_b: np.ndarray,
    training_rows,
    n0: float,
) -> State:
    """Explicit-N0 coordinate-ascent conditioning on Frame B SC training blocks.

    Exact explicit-``n0`` mirror of ``p8c.receiver.condition``. Never touches ``PROBE_ROW``.
    """
    rows = _validate_training_rows(training_rows)
    if not rows:
        return prior_state
    n0_val = float(n0)
    if n0_val < 0.0:
        raise ValueError(f"n0 must be non-negative, got {n0_val}")
    n0_eff = max(n0_val, 1e-12)
    y = np.asarray(y_b, dtype=np.complex128)[list(rows)].reshape(-1)
    if not np.isfinite(y).all():
        raise ValueError("Training observations contain non-finite values")

    design, conjugate, energy = _training_dictionary(prior_state.epoch, rows)
    post_var = prior_state.variance / (1.0 + prior_state.variance * energy / n0_eff)
    log_weight = prior_state.log_weight.copy()
    mean = prior_state.mean.copy()
    per_ray = np.einsum("lh,lhq->lq", np.exp(log_weight) * mean, design)
    initial_precision_mean = prior_state.mean / prior_state.variance
    evidence_base = (
        prior_state.log_weight
        + np.log(post_var / prior_state.variance)
        - (np.abs(prior_state.mean) ** 2) / prior_state.variance
    )
    for _ in range(SWEEPS):
        for ray in range(len(DELAYS)):
            residual = y - per_ray.sum(axis=0) + per_ray[ray]
            matched = conjugate[ray] @ residual
            natural_mean = initial_precision_mean[ray] + matched / n0_eff
            mean[ray] = post_var[ray] * natural_mean
            evidence = evidence_base[ray] + post_var[ray] * (np.abs(natural_mean) ** 2)
            log_weight[ray] = evidence - logsumexp(evidence)
            per_ray[ray] = (np.exp(log_weight[ray]) * mean[ray]) @ design[ray]
    return State(prior_state.epoch, log_weight, mean, post_var)


def sample_moments_b(state: State) -> tuple[np.ndarray, np.ndarray]:
    """Time-domain first moments and marginal variances on Frame B, shaped ``[3, T, N]``."""
    weights = np.exp(state.log_weight)
    rotations = evolution(state.epoch).reshape(weights.shape[1], -1)
    mean = ((weights * state.mean) @ rotations).reshape(len(DELAYS), T, N)
    second = (weights * (np.abs(state.mean) ** 2 + state.variance)).sum(axis=1)
    return mean, np.maximum(second[:, None, None] - np.abs(mean) ** 2, 0.0)


def build_sc_channel_matrix(mean_ray_row: np.ndarray) -> np.ndarray:
    """Build the ``[N, N]`` time-varying linear SC channel matrix ``H_r`` for one row."""
    row_gains = np.asarray(mean_ray_row, dtype=np.complex128)
    if row_gains.shape != (len(DELAYS), N):
        raise ValueError(f"Expected mean_ray_row shape {(len(DELAYS), N)}, got {row_gains.shape}")
    H = np.zeros((N, N), dtype=np.complex128)
    for ray, delayed in enumerate(DELAYED_INDEX):
        H[SAMPLES, delayed] += row_gains[ray]
    return H


def build_sc_channel_matrices(mean_ray: np.ndarray, rows=None) -> np.ndarray:
    """Build batched ``[len(rows), N, N]`` time-varying linear SC channel matrices."""
    gains = np.asarray(mean_ray, dtype=np.complex128)
    row_idx = np.arange(gains.shape[1], dtype=np.int64) if rows is None else np.asarray(list(rows), dtype=np.int64)
    design = np.zeros((len(row_idx), N, N), dtype=np.complex128)
    for ray, delayed in enumerate(DELAYED_INDEX):
        design[:, SAMPLES, delayed] += gains[ray, row_idx]
    return design


def effective_row_noise(var_ray: np.ndarray, n0: float, rows=None) -> np.ndarray:
    """Per-sample effective noise variance ``n0 + sum_ray var_ray`` on selected rows."""
    n0_val = float(n0)
    if n0_val < 0.0:
        raise ValueError(f"n0 must be non-negative, got {n0_val}")
    v = np.asarray(var_ray, dtype=np.float64)
    if rows is None:
        return n0_val + v.sum(axis=0)
    return n0_val + v[:, list(rows)].sum(axis=0)


def readout_frame_b(
    mean_ray: np.ndarray,
    var_ray: np.ndarray,
    y_b: np.ndarray,
    training_rows,
    n0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit-N0 single-carrier waveform prediction and data-block LMMSE soft readout.

    Exact explicit-``n0`` mirror of ``p8c.receiver.readout``, plus exact zero-noise
    limit handling when ``n0 == 0`` and ``var_ray == 0``. Never reads ``y_b[PROBE_ROW]``.
    """
    rows = _validate_training_rows(training_rows)
    n0_val = float(n0)
    if n0_val < 0.0:
        raise ValueError(f"n0 must be non-negative, got {n0_val}")

    mean_r = np.asarray(mean_ray, dtype=np.complex128)
    var_r = np.asarray(var_ray, dtype=np.float64)
    y = np.asarray(y_b, dtype=np.complex128)

    known = SC_PILOTS[PROBE_ROW, DELAYED_INDEX]
    prediction = (mean_r[:, PROBE_ROW] * known).sum(axis=0)
    pred_var = n0_val + (var_r[:, PROBE_ROW] * (np.abs(known) ** 2)).sum(axis=0)

    data = np.array([r for r in range(T) if r not in (*rows, PROBE_ROW)], dtype=np.int64)
    llrs = np.zeros((T, N, 4), dtype=np.float64)
    if not len(data):
        return llrs, prediction, pred_var

    noise = n0_val + var_r[:, data].sum(axis=0)
    design = build_sc_channel_matrices(mean_r, data)

    if np.all(noise > 0.0):
        adjoint = design.conj().swapaxes(-1, -2) / noise[:, None, :]
        cov = np.linalg.inv(np.eye(N) + adjoint @ design)
        estimate = (cov @ (adjoint @ y[data, :, None]))[..., 0]
        uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
    else:
        # Exact zero-noise limit: H^H (H H^H + diag(noise))^{-1} y.
        gram = design @ design.conj().swapaxes(-1, -2)
        gram[:, SAMPLES, SAMPLES] += noise
        try:
            z = np.linalg.solve(gram, y[data, :, None])
            w = np.linalg.solve(gram, design)
        except np.linalg.LinAlgError:
            gram_pinv = np.linalg.pinv(gram)
            z = gram_pinv @ y[data, :, None]
            w = gram_pinv @ design
        estimate = (design.conj().swapaxes(-1, -2) @ z)[..., 0]
        diag_reduction = np.einsum("bik,bik->bk", design.conj(), w).real
        uncertainty = np.maximum(1.0 - diag_reduction, 0.0)

    llrs[data] = qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(len(data), N, 4)
    return llrs, prediction, pred_var


def receive_frame_b(
    prior_state: State,
    y_b: np.ndarray,
    training_rows,
    n0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition ``prior_state`` on Frame B training rows and compute LMMSE LLRs & probe moments."""
    rows = _validate_training_rows(training_rows)
    posterior = condition_frame_b(prior_state, y_b, rows, n0)
    mean_ray, var_ray = sample_moments_b(posterior)
    return readout_frame_b(mean_ray, var_ray, y_b, rows, n0)


def oracle_readout_b(
    truth: HiddenTruth,
    y_b: np.ndarray,
    training_rows,
    n0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluator-only true-channel readout using ``truth.h_b_true`` and explicit ``n0``."""
    return readout_frame_b(
        truth.h_b_true,
        np.zeros(truth.h_b_true.shape, dtype=np.float64),
        y_b,
        training_rows,
        n0,
    )
