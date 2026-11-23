"""Small pilot-free I/Q encoders with an integer index per received window."""
from dataclasses import dataclass

import numpy as np


ROOTS = np.array([1, 1j, -1, -1j])
LABELS = np.arange(4)


def input_features(windows, equivariant):
    iq = np.asarray(windows, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] != 3:
        raise ValueError("Expected an array of three complex samples per window")
    turns = np.zeros(len(iq), dtype=np.int64)
    if equivariant:
        # Half-open quadrant [-45°,45°), and its three quarter-turn shifts.
        anchor = iq.sum(axis=1)
        missing = anchor == 0
        if np.any(missing):
            candidates = iq[missing]
            first = np.argmax(candidates != 0, axis=1)
            anchor[missing] = candidates[np.arange(len(candidates)), first]
        u = anchor.real + anchor.imag
        v = anchor.imag - anchor.real
        turns[(u > 0) & (v >= 0)] = 1
        turns[(u <= 0) & (v > 0)] = 2
        turns[(u < 0) & (v <= 0)] = 3
        iq = iq * ROOTS[turns].conj()[:, None]
    return np.column_stack((iq.real, iq.imag)), turns


def forward(x, weights):
    w1, b1, w2, b2 = weights
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    return hidden, ex / ex.sum(axis=1, keepdims=True)


def fit_network(x, target, rng):
    width = 16
    weights = [
        rng.normal(0, np.sqrt(2 / (6 + width)), (6, width)),
        np.zeros(width),
        rng.normal(0, np.sqrt(2 / (width + 4)), (width, 4)),
        np.zeros(4),
    ]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    step = 0
    for _ in range(48):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = order[start:start + 256]
            xb = x[index]
            hidden, prediction = forward(xb, weights)
            residual = (prediction - target[index]) / len(index)
            delta = (residual @ weights[2].T) * (1 - hidden**2)
            grads = (xb.T @ delta, delta.sum(axis=0), hidden.T @ residual,
                     residual.sum(axis=0))
            step += 1
            for w, m, v, g in zip(weights, first, second, grads):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g*g
                w -= 0.005 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return tuple(weights)


def assign(posterior, centers):
    # The posterior entropy is constant across candidate centers.
    return (posterior @ np.log(centers).T).argmax(axis=1)


def fit_centers(prediction, count, rng):
    if count == 1:
        return prediction.mean(axis=0, keepdims=True)
    centers = [prediction[rng.integers(len(prediction))].copy()]
    # KL-proportional seeding, using the distance to the nearest center so far.
    self_score = np.sum(prediction * np.log(prediction), axis=1)
    distance = np.full(len(prediction), np.inf)
    for _ in range(1, count):
        d = np.maximum(0.0, self_score - prediction @ np.log(centers[-1]))
        distance = np.minimum(distance, d)
        centers.append(prediction[rng.choice(len(prediction), p=distance / distance.sum())].copy())
    centers = np.stack(centers)
    prev = None
    for _ in range(30):
        code = assign(prediction, centers)
        if prev is not None and np.array_equal(prev, code):
            break
        new = centers.copy()
        occupied = np.bincount(code, minlength=count)
        for j in np.flatnonzero(occupied):
            new[j] = prediction[code == j].mean(axis=0)
        if np.any(occupied == 0):
            worst_first = np.argsort(self_score - np.sum(prediction * np.log(centers[code]), axis=1))[::-1]
            for j, i in zip(np.flatnonzero(occupied == 0), worst_first):
                new[j] = prediction[i]
        prev, centers = code, new
    return centers


@dataclass
class Learned:
    equivariant: bool
    weights: tuple
    centers: np.ndarray
    table: np.ndarray

    def encode(self, windows):
        x, turns = input_features(windows, self.equivariant)
        _, prediction = forward(x, self.weights)
        local = assign(prediction, self.centers)
        return turns * len(self.centers) + local

    def save(self, path):
        w1, b1, w2, b2 = self.weights
        np.savez(path, equivariant=self.equivariant, w1=w1, b1=b1, w2=w2, b2=b2,
                 centers=self.centers, table=self.table)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return Learned(bool(data["equivariant"]), tuple(data[k] for k in ("w1", "b1", "w2", "b2")),
                       data["centers"], data["table"])


def fit(windows, soft_targets, bits, seed, equivariant):
    if bits not in (4, 8):
        raise ValueError("The frozen experiment uses 4 or 8 bits for the two-window cache")
    x, turns = input_features(windows, equivariant)
    target = np.asarray(soft_targets, dtype=np.float64)
    if target.shape != (len(x), 4) or not np.allclose(target.sum(axis=1), 1):
        raise ValueError("Expected one normalized four-label posterior per window")
    target = np.take_along_axis(target, (LABELS[None, :] + turns[:, None]) % 4, axis=1)
    rng = np.random.default_rng(seed)
    weights = fit_network(x, target, rng)
    _, prediction = forward(x, weights)
    size = 2 ** (bits // 2)
    centers = fit_centers(prediction, size // 4 if equivariant else size, rng)
    local = assign(prediction, centers)
    if np.any(np.bincount(local, minlength=len(centers)) == 0):
        raise ValueError("Unused training code in the fitted vector quantizer")
    table = np.stack([target[local == i].mean(axis=0) for i in range(len(centers))])
    if equivariant:
        table = np.concatenate([np.roll(table, r, axis=1) for r in range(4)])
    return Learned(bool(equivariant), weights, centers, table)
