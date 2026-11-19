"""Two contiguous CP-OFDM frames on one sample-clock, single-antenna R2 link."""
from dataclasses import dataclass

import numpy as np

from .coding import generator, qam


N, CP, T = 32, 16, 20
DELAYS = np.array([0, 2, 4])
POWERS = np.array([.75, .17, .08])
FRAME_DURATION = T * (N + CP) / N
LINK_DURATION = (2 * T * (N + CP) - 1) / N
N0 = 10. ** (-.9)
PILOT_ROWS = (0, 9, 10, 19)
B_ROWS = {"dense": (0, 9, 10), "sparse": (0,)}
EVEN = np.arange(0, N, 2)
PROBE_ROW = 19


def clock(n_rows=T, start_row=0):
    """Useful sample times in units of N samples, from the start of the first CP."""
    rows = np.arange(start_row, start_row + n_rows)
    return (rows[:, None] * (N + CP) + CP + np.arange(N)) / N


def row_mask(rows):
    mask = np.zeros((T, N), dtype=bool)
    mask[np.ix_(list(rows), EVEN)] = True
    return mask


CORE = ~row_mask(PILOT_ROWS)
PROBES = row_mask((PROBE_ROW,))
PILOTS = np.zeros((T, N), dtype=complex)
_bits = np.random.default_rng(800016).choice([-1., 1.], size=(int((~CORE).sum()), 2))
PILOTS[~CORE] = (_bits[:, 0] + 1j * _bits[:, 1]) / np.sqrt(2)


@dataclass(frozen=True)
class Channel:
    gain: np.ndarray       # complex gain at start of A's CP
    frequency: np.ndarray  # cycles per N samples
    chirp: np.ndarray      # cycles per (N samples)^2


def stream(seed, index):
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


def complex_normal(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.)


def draw_channel(rng):
    gain = np.sqrt(POWERS) * complex_normal(rng, len(DELAYS))
    gain[0] = np.sqrt(POWERS[0]) * np.exp(1j * rng.uniform(-np.pi, np.pi))
    signs = rng.choice([-1, 1], len(DELAYS))
    start, end = rng.uniform(.045, .080, size=(2, len(DELAYS))) * signs
    return Channel(gain, start, (end - start) / LINK_DURATION)


def propagate(grid, channel, start_row=0):
    """Delay each CP-guarded symbol and rotate every received sample before the FFT."""
    time = clock(len(grid), start_row)
    tx = np.fft.ifft(grid, norm="ortho")
    received = np.zeros_like(tx)
    for delay, amp, freq, chirp in zip(DELAYS, channel.gain, channel.frequency, channel.chirp):
        angle = 2 * np.pi * (freq * time + .5 * chirp * time**2)
        received += amp * np.exp(1j * angle) * np.roll(tx, delay, axis=-1)
    return np.fft.fft(received, norm="ortho")


@dataclass(frozen=True)
class Traffic:
    info: np.ndarray
    code: np.ndarray
    symbols: np.ndarray


def traffic(rng):
    # 18 systematic words occupy identical grid cells at both pilot budgets.
    # The sparse transmission uses one additional word in the released cells.
    info = rng.integers(0, 2, size=(19, 64), dtype=np.uint8)
    code = (info @ generator()) % 2
    return Traffic(info, code, qam(code).reshape(19, 32))


def tx_grid(words, rows):
    return map_traffic(words, row_mask(rows))


def map_traffic(words, pilot_mask):
    if np.any(pilot_mask & CORE):
        raise ValueError("Acquisition pilots must be on reserved cells")
    grid = np.zeros((T, N), dtype=complex)
    grid[pilot_mask] = PILOTS[pilot_mask]
    grid[CORE] = words.symbols[:18].ravel()
    extra = (~pilot_mask) & (~CORE)
    if extra.any():
        if extra.sum() != 32:
            raise ValueError("Unassigned cells must fit one extra (128,64) word")
        grid[extra] = words.symbols[18]
    return grid


def noise(rng):
    return np.fft.fft(np.sqrt(N0) * complex_normal(rng, (T, N)), norm="ortho")


@dataclass(frozen=True)
class Pair:
    a: np.ndarray
    b: dict
    clean_b: dict
    b_words: Traffic
    donor_a: np.ndarray


def simulate(seed, acquisition_masks=None):
    channel = draw_channel(stream(seed, 0))
    a = propagate(tx_grid(traffic(stream(seed, 1)), PILOT_ROWS), channel) + noise(stream(seed, 2))
    words = traffic(stream(seed, 3))
    b_noise = noise(stream(seed, 4))
    if acquisition_masks is None:
        acquisition_masks = {key: row_mask(rows) for key, rows in B_ROWS.items()}
    clean = {key: propagate(map_traffic(words, mask | PROBES), channel, start_row=T)
             for key, mask in acquisition_masks.items()}
    donor_channel = draw_channel(stream(seed, 5))
    donor_grid = tx_grid(traffic(stream(seed, 6)), PILOT_ROWS)
    donor = propagate(donor_grid, donor_channel) + noise(stream(seed, 7))
    return Pair(a, {key: y + b_noise for key, y in clean.items()}, clean, words, donor)
