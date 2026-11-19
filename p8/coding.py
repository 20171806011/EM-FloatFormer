"""NumPy frontend for the project's native systematic (128,64) LDPC and Gray 16-QAM."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np


P4_H_SHA256 = "5c8a40b0e9046bae409f7d3446a11ed1ed948de37347be90e4dc325177506e72"
PAM = np.array([-3., -1., 3., 1.]) / np.sqrt(10)


@lru_cache(maxsize=1)
def parity_check():
    h = np.zeros((64, 128), np.uint8)
    for row, line in enumerate(Path(__file__).with_name("ldpc_checks.txt").read_text().splitlines()):
        h[row, np.fromstring(line, sep=" ", dtype=int)] = 1
    if sha256(h.tobytes()).hexdigest() != P4_H_SHA256:
        raise ValueError("Parity graph differs from p4/results/p4a_coded_calibration/transfer.json")
    h.setflags(write=False)
    return h


@lru_cache(maxsize=1)
def generator():
    h = parity_check()
    augmented = np.column_stack((h[:, 64:], h[:, :64]))
    for col in range(64):
        pivot = col + np.flatnonzero(augmented[col:, col])[0]
        augmented[[col, pivot]] = augmented[[pivot, col]]
        others = augmented[:, col].astype(bool)
        others[col] = False
        augmented[others] ^= augmented[col]
    return np.column_stack((np.eye(64, dtype=np.uint8), augmented[:, 64:].T))


def qam(bits):
    b = np.asarray(bits).reshape(-1, 4)
    return PAM[2*b[:, 0] + b[:, 1]] + 1j * PAM[2*b[:, 2] + b[:, 3]]


def qam_llrs(estimate, posterior_var):
    """Gray LLRs from a unit-variance-prior LMMSE output, positive for bit one."""
    var = np.clip(posterior_var, 1e-12, 1.)
    gain = 1. - var
    axes = np.stack((estimate.real, estimate.imag), axis=-1)
    logp = (2 * axes[..., None] * PAM - gain[:, None, None] * PAM**2) / var[:, None, None]
    sign = np.logaddexp(logp[..., 2], logp[..., 3]) - np.logaddexp(logp[..., 0], logp[..., 1])
    inner = np.logaddexp(logp[..., 1], logp[..., 3]) - np.logaddexp(logp[..., 0], logp[..., 2])
    return np.stack((sign, inner), axis=-1).reshape(-1, 4)


@lru_cache(maxsize=1)
def _adjacency():
    h = parity_check()
    row, col = np.nonzero(h)
    n_edges = len(row)
    # Both incidence tables use the final edge for neutral padding.
    by_check = np.full((64, int(h.sum(axis=1).max())), n_edges)
    by_bit = np.full((128, int(h.sum(axis=0).max())), n_edges)
    for check, target in enumerate(by_check):
        edges = np.flatnonzero(row == check)
        target[:len(edges)] = edges
    for bit, target in enumerate(by_bit):
        edges = np.flatnonzero(col == bit)
        target[:len(edges)] = edges
    return col, by_check, by_bit


def decode(llrs, iterations=12):
    """P4 normalized min-sum schedule, alpha=.75, per-word syndrome stopping."""
    channel = -np.asarray(llrs, dtype=np.float32).reshape(-1, 128)
    col, by_check, by_bit = _adjacency()
    padding = np.zeros((len(channel), 1), dtype=np.float32)
    active = np.ones(len(channel), dtype=bool)
    decided = np.empty(channel.shape, dtype=np.uint8)
    variable = channel[:, col]
    valid = by_check < len(col)
    for _ in range(iterations):
        grid = np.column_stack((variable, padding))[:, by_check]
        sign = np.where(grid < 0, np.float32(-1.), np.float32(1.))
        magnitude = np.where(valid, np.abs(grid), np.float32(np.inf))
        two = np.partition(magnitude, 1, axis=-1)[..., :2]
        outgoing = np.where(magnitude == two[..., :1], two[..., 1:2], two[..., :1])
        outgoing *= (.75 * sign.prod(axis=-1, keepdims=True)) * sign
        check_messages = outgoing[:, valid]
        post = channel + np.column_stack((check_messages, padding))[:, by_bit].sum(axis=-1)
        hard = (post < 0).astype(np.uint8)
        decided[active] = hard[active]
        syndrome = (hard @ parity_check().T) % 2
        active &= np.any(syndrome, axis=1)
        if not np.any(active):
            break
        variable = post[:, col] - check_messages
    return decided[:, :64]
