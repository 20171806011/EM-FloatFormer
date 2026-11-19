"""A continuous R2 trajectory carrying an OFDM frame A and a single-carrier frame B."""
from dataclasses import dataclass

import numpy as np

from p8 import link as ofdm
from p8.link import CP, DELAYS, N, N0, T


PROBE_ROW = 19
LAYOUTS = {"sparse": (10,), "dense": (0, 9, 10)}
CORE_ROWS = tuple(row for row in range(T) if row not in (*LAYOUTS["dense"], PROBE_ROW))


def _training_sequences():
    # Even-length root-3 Zadoff-Chu sequence. Its circular shifts by the three
    # physical delays are orthogonal on each complete training block.
    n = np.arange(N)
    phase_index = (-3 * n**2) % (2*N)
    root = np.exp(1j * np.pi * phase_index / N)
    return np.stack([1j**(row % 4) * np.roll(root, 3*row) for row in range(T)])


PILOTS = _training_sequences()
PILOTS.setflags(write=False)


def extra_rows(rows):
    """Rows carrying extra codewords under the specified acquisition schedule."""
    rows = tuple(sorted(rows))
    if rows not in LAYOUTS.values():
        raise ValueError("Expected a registered single-carrier acquisition schedule")
    return tuple(row for row in LAYOUTS["dense"] if row not in rows)


def tx_grid(words, rows):
    """Map one native 128-bit codeword to each single-carrier data block."""
    rows = tuple(rows)
    extra = extra_rows(rows)
    training = [*rows, PROBE_ROW]
    grid = np.empty((T, N), dtype=np.complex128)
    grid[training] = PILOTS[training]
    grid[list(CORE_ROWS)] = words.symbols[:len(CORE_ROWS)]
    if extra:
        grid[list(extra)] = words.symbols[len(CORE_ROWS):len(CORE_ROWS) + len(extra)]
    return grid


def propagate(grid, channel, start_row=T):
    """Return post-CP time samples, rotating delayed symbols at absolute receive time."""
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.shape[1] != N:
        raise ValueError("Expected 32 useful time-domain symbols per single-carrier block")
    time = ofdm.clock(len(grid), start_row)
    received = np.zeros(grid.shape, dtype=np.complex128)
    for delay, gain, frequency, chirp in zip(DELAYS, channel.gain, channel.frequency, channel.chirp):
        phase = 2*np.pi * (frequency*time + .5*chirp*time**2)
        received += gain * np.exp(1j*phase) * np.roll(grid, delay, axis=-1)
    return received


@dataclass(frozen=True)
class Pair(ofdm.Pair):
    """The true channel and clean B are evaluator-only fields."""

    channel: ofdm.Channel


def _ofdm_frame(seed, channel, payload_stream, noise_stream):
    words = ofdm.traffic(ofdm.stream(seed, payload_stream))
    grid = ofdm.tx_grid(words, ofdm.PILOT_ROWS)
    return ofdm.propagate(grid, channel) + ofdm.noise(ofdm.stream(seed, noise_stream))


def simulate(seed):
    channel = ofdm.draw_channel(ofdm.stream(seed, 0))
    a = _ofdm_frame(seed, channel, 1, 2)
    words = ofdm.traffic(ofdm.stream(seed, 3))
    b_words = ofdm.Traffic(words.info[:18], words.code[:18], words.symbols[:18])
    noise = np.sqrt(N0) * ofdm.complex_normal(ofdm.stream(seed, 4), (T, N))
    clean = {key: propagate(tx_grid(b_words, rows), channel) for key, rows in LAYOUTS.items()}
    donor = _ofdm_frame(seed, ofdm.draw_channel(ofdm.stream(seed, 5)), 6, 7)
    return Pair(a=a, donor_a=donor, b={key: y + noise for key, y in clean.items()},
                clean_b=clean, b_words=b_words, channel=channel)
