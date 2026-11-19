"""Classical bank of Doppler/chirp hypotheses with a complex Gaussian gain per node."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from .coding import qam_llrs
from .link import DELAYS, EVEN, LINK_DURATION, N, N0, PILOTS, POWERS, PROBE_ROW, T, clock, row_mask


_magnitudes = np.linspace(.045, .080, 8)
_initial, _final = np.meshgrid(_magnitudes, _magnitudes, indexing="ij")
FREQ = np.concatenate((_initial.ravel(), -_initial.ravel()))
_end = np.concatenate((_final.ravel(), -_final.ravel()))
CHIRP = (_end - FREQ) / LINK_DURATION
BASIS = np.exp(-2j * np.pi * DELAYS[:, None] * np.fft.fftfreq(N))
DIFF = (np.arange(N)[:, None] - np.arange(N)) % N
# The comb LS projection averages 16 bins. Unresolved intra-symbol ICI is
# approximated by its power at the midpoint of the public Doppler range.
TAP_NOISE = (N0 + 1. - np.sinc(.0625)**2) / len(EVEN)


@dataclass(frozen=True)
class State:
    epoch: float            # public CP-start time in useful-symbol units
    log_weight: np.ndarray  # [ray, hypothesis]
    mean: np.ndarray        # posterior complex ray gain at epoch
    variance: np.ndarray


def prior():
    size = (len(DELAYS), len(FREQ))
    return State(0., np.full(size, -np.log(len(FREQ))), np.zeros(size, complex),
                 np.broadcast_to(POWERS[:, None], size).copy())


def advance(state, duration):
    """Transport complex gains and the frequency of every node to a new frame origin."""
    frequency = FREQ + CHIRP * state.epoch
    rotation = np.exp(2j * np.pi * (frequency * duration + .5 * CHIRP * duration**2))
    return State(state.epoch + duration, state.log_weight, state.mean * rotation, state.variance)


@lru_cache(maxsize=4)
def evolution(epoch):
    time = clock()[None, :, :]
    freq = (FREQ + CHIRP * epoch)[:, None, None]
    return np.exp(2j * np.pi * (freq * time + .5 * CHIRP[:, None, None] * time**2))


def observed_taps(received, rows):
    """Project the specified known comb pilots; no access to other receive equations."""
    selection = np.ix_(rows, EVEN)
    pilot_ls = received[selection] / PILOTS[selection]
    return pilot_ls @ BASIS[:, EVEN].conj().T / len(EVEN)


def update(state, rows, tap_obs, row_noise=TAP_NOISE):
    if np.shape(tap_obs) != (len(rows), len(DELAYS)):
        raise ValueError("Expected one complex observation per pilot row and tap")
    signatures = evolution(state.epoch)[:, list(rows)].mean(axis=-1).T
    mean, variance, log_weight = state.mean, state.variance, state.log_weight
    for u, z, noise in zip(signatures, tap_obs, np.broadcast_to(row_noise, (len(rows),))):
        innovation = z[:, None] - u * mean
        total = noise + abs(u)**2 * variance
        log_weight = log_weight - np.log(total) - abs(innovation)**2 / total
        log_weight -= logsumexp(log_weight, axis=1, keepdims=True)
        mean = mean + variance * u.conj() * innovation / total
        variance = variance * noise / total
    return State(state.epoch, log_weight, mean, variance)


@lru_cache(maxsize=4)
def probe_dictionary(epoch):
    # For each ray and hypothesis, response to the known probe-row symbols.
    # The unknown-data power of a single unit-gain ray is independent of delay.
    rotation = evolution(epoch)[:, PROBE_ROW, :]
    ici = (np.fft.fft(rotation, axis=-1) / N)[:, DIFF[EVEN]]
    known = np.einsum("hok,lk,k->lho", ici, BASIS, PILOTS[PROBE_ROW])
    unknown_energy = (abs(ici[:, :, 1::2])**2).sum(axis=-1)
    return known, unknown_energy


def probe_moments(state, weights, by_ray, second):
    """Mean and total variance for unseen complex observations (noise included)."""
    known, unknown_energy = probe_dictionary(state.epoch)
    rays = by_ray[:, PROBE_ROW, EVEN, :]
    ray_known_mean = rays @ PILOTS[PROBE_ROW]
    prediction = ray_known_mean.sum(axis=0)
    known_variance = np.einsum("lh,lho->o", weights * second, abs(known)**2)
    known_variance -= (abs(ray_known_mean)**2).sum(axis=0)
    other = rays[:, :, 1::2]
    data_variance = np.einsum("lh,ho->o", weights * second, unknown_energy)
    data_variance += (abs(other.sum(axis=0))**2).sum(axis=-1) - (abs(other)**2).sum(axis=(0, 2))
    return prediction, N0 + np.maximum(known_variance + data_variance, 0.)


def readout(state, received, rows):
    """Predict withheld pilots and return data LLRs using data-bin observations only."""
    return readout_mask(state, received, row_mask((*rows, PROBE_ROW)))


def readout_mask(state, received, mask):
    weights = np.exp(state.log_weight)
    second = abs(state.mean)**2 + state.variance
    mean_rotation = ((weights * state.mean) @ evolution(state.epoch).reshape(len(FREQ), -1))
    mean_rotation = mean_rotation.reshape(len(DELAYS), T, N)
    # Parseval: expected per-output model-error power for white unit-energy input.
    channel_var = np.sum(weights * second) - np.mean(abs(mean_rotation)**2, axis=-1).sum(axis=0)
    effective_noise = N0 + np.maximum(channel_var, 0.)
    ici = np.fft.fft(mean_rotation, axis=-1) / N
    by_ray = ici[:, :, DIFF] * BASIS[:, None, None, :]
    prediction, pred_var = probe_moments(state, weights, by_ray, second)
    g = by_ray.sum(axis=0)
    pilots = np.where(mask, PILOTS, 0.)
    pilot_rx = (g @ pilots[..., None])[..., 0]
    llrs = np.zeros((T, N, 4))
    # Rows with the same comb share one observed submatrix shape.
    for pattern in np.unique(mask, axis=0):
        times = np.flatnonzero(np.all(mask == pattern, axis=1))
        data = np.flatnonzero(~pattern)
        design = g[times][:, data][:, :, data]
        observations = (received[np.ix_(times, data)] - pilot_rx[np.ix_(times, data)])[..., None]
        hermitian = design.conj().swapaxes(-1, -2) / effective_noise[times, None, None]
        cov = np.linalg.inv(np.eye(len(data)) + hermitian @ design)
        estimate = (cov @ (hermitian @ observations))[..., 0]
        uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
        llrs[np.ix_(times, data)] = qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(
            len(times), len(data), 4)
    return llrs, prediction, pred_var
