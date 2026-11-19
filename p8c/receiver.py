"""Physical-state transfer into a cyclic-prefix single-carrier receiver."""
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from p8.coding import qam_llrs
from p8.link import DELAYS, N, N0, T, clock
from p8.tracker import State, evolution
from .link import PILOTS, PROBE_ROW


SWEEPS = 4
SAMPLES = np.arange(N)
DELAYED_INDEX = (SAMPLES - DELAYS[:, None]) % N


def _rows(rows):
    rows = tuple(int(t) for t in rows)
    if len(set(rows)) != len(rows) or any(t < 0 or t >= T or t == PROBE_ROW for t in rows):
        raise ValueError("Training rows must be distinct, in range, and exclude the withheld block")
    return rows


@lru_cache(maxsize=8)
def _training_dictionary(epoch, rows):
    """Per-sample unit-gain responses of each delayed pilot to each Doppler node."""
    rotations = evolution(epoch)[:, list(rows)]
    pilots = PILOTS[list(rows)]
    delayed = np.stack([np.roll(pilots, d, axis=-1) for d in DELAYS])
    design = (delayed[:, None] * rotations[None]).reshape(len(DELAYS), len(rotations), -1)
    return design, design.conj(), np.sum(abs(design)**2, axis=-1)


def condition(prior, received, rows):
    """Coordinate ascent on independent ray factors under the full training likelihood.

    Each coordinate conditions the original prior on the residual after subtracting
    the other rays' posterior mean responses. Conditional on a bank node the complex
    gain update is Gaussian. Integrating out this gain gives the node's evidence.
    """
    rows = _rows(rows)
    if not rows:
        return prior
    y = received[list(rows)].reshape(-1)
    design, conjugate, energy = _training_dictionary(prior.epoch, rows)
    post_var = prior.variance / (1 + prior.variance * energy / N0)
    log_weight = prior.log_weight.copy()
    mean = prior.mean.copy()
    per_ray = np.einsum("lh,lhq->lq", np.exp(log_weight) * mean, design)
    initial_precision_mean = prior.mean / prior.variance
    evidence_base = prior.log_weight + np.log(post_var / prior.variance) - abs(prior.mean)**2 / prior.variance
    for _ in range(SWEEPS):
        for ray in range(len(DELAYS)):
            residual = y - per_ray.sum(axis=0) + per_ray[ray]
            matched = conjugate[ray] @ residual
            natural_mean = initial_precision_mean[ray] + matched / N0
            mean[ray] = post_var[ray] * natural_mean
            evidence = evidence_base[ray] + post_var[ray] * abs(natural_mean)**2
            log_weight[ray] = evidence - logsumexp(evidence)
            per_ray[ray] = (np.exp(log_weight[ray]) * mean[ray]) @ design[ray]
    return State(prior.epoch, log_weight, mean, post_var)


def sample_moments(state):
    """Time-domain first moments and marginal variances, shaped [ray, row, sample]."""
    weights = np.exp(state.log_weight)
    rotations = evolution(state.epoch).reshape(weights.shape[1], -1)
    mean = ((weights * state.mean) @ rotations).reshape(len(DELAYS), T, N)
    second = (weights * (abs(state.mean)**2 + state.variance)).sum(axis=1)
    return mean, np.maximum(second[:, None, None] - abs(mean)**2, 0.)


def readout(mean_ray, var_ray, received, rows):
    """Single-carrier waveform prediction and data-block LMMSE soft decoding."""
    rows = _rows(rows)
    known = PILOTS[PROBE_ROW, DELAYED_INDEX]
    prediction = (mean_ray[:, PROBE_ROW] * known).sum(axis=0)
    pred_var = N0 + (var_ray[:, PROBE_ROW] * abs(known)**2).sum(axis=0)

    data = np.array([r for r in range(T) if r not in (*rows, PROBE_ROW)], dtype=int)
    llrs = np.zeros((T, N, 4))
    if not len(data):
        return llrs, prediction, pred_var
    # For white, unit-power data and independent ray factors the additional
    # received covariance from path uncertainty is diagonal in sample time.
    noise = N0 + var_ray[:, data].sum(axis=0)
    design = np.zeros((len(data), N, N), dtype=complex)
    for ray, delayed in enumerate(DELAYED_INDEX):
        design[:, SAMPLES, delayed] = mean_ray[ray, data]
    adjoint = design.conj().swapaxes(-1, -2) / noise[:, None, :]
    cov = np.linalg.inv(np.eye(N) + adjoint @ design)
    estimate = (cov @ (adjoint @ received[data, :, None]))[..., 0]
    uncertainty = np.diagonal(cov, axis1=-2, axis2=-1).real
    llrs[data] = qam_llrs(estimate.ravel(), uncertainty.ravel()).reshape(len(data), N, 4)
    return llrs, prediction, pred_var


def receive(state, received, rows):
    """B receiver entry point. State is referenced to the start of B's prefix."""
    rows = _rows(rows)
    posterior = condition(state, received, rows)
    return readout(*sample_moments(posterior), received, rows)


def oracle(channel, received, rows):
    """Evaluation-only true-channel reference using the same single-carrier decoder."""
    time = clock(T, start_row=T)
    phase = (channel.frequency[:, None, None] * time
             + .5 * channel.chirp[:, None, None] * time**2)
    gains = channel.gain[:, None, None] * np.exp(2j*np.pi*phase)
    return readout(gains, np.zeros(gains.shape, dtype=float), received, rows)
