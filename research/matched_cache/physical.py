"""Fixed-budget hand encoder for the frozen three-sample 2 dB QPSK window."""

import math

import numpy as np
from scipy.special import ndtr


WINDOW_LENGTH = 3
SNR = 10 ** 0.2
MEAN = 1 / math.sqrt(2)
NOISE_SD = 1 / math.sqrt(2 * SNR)
# Rotating positive-real, positive-imaginary, negative-real, negative-imaginary
# QPSK by pi/4 gives coordinate signs (++), (-+), (--), (+-).
POSITIVE = ((True, True), (False, True), (False, False), (True, False))


def sign_bins(count):
    """Interior cuts on observed coordinate and exact cell posterior means."""
    # Posterior-positive = logistic(2 * MEAN * t / NOISE_SD**2).
    p = np.arange(1, count, dtype=float) / count
    cuts = NOISE_SD**2 / (2 * MEAN) * np.log(p / (1 - p))
    # Make the independently representable positive and negative cuts exact mirrors.
    cuts = (cuts - cuts[::-1]) / 2
    edges = np.r_[-np.inf, cuts, np.inf]
    given_plus = np.diff(ndtr((edges - MEAN) / NOISE_SD))
    # Reflection of a negative-sign observation has the positive-sign law.
    given_minus = given_plus[::-1]
    plus = given_plus / (given_plus + given_minus)
    plus[count // 2:] = 1 - plus[:count // 2][::-1]
    return cuts, plus


class Hand:
    """encode emits one integer per window. Two integers together occupy bits."""

    def __init__(self, bits):
        if bits not in (4, 8):
            raise ValueError("the frozen experiment uses 4 or 8 bits per window pair")
        self.bits = bits
        self.bins = 2 ** (bits // 4)
        self.cuts, positive = sign_bins(self.bins)
        negative = positive[::-1]
        self.table = np.array([
            [(positive[r] if rp else negative[r]) * (positive[i] if ip else negative[i])
             for rp, ip in POSITIVE]
            for r in range(self.bins) for i in range(self.bins)
        ])

    def encode(self, windows):
        data = np.asarray(windows, dtype=np.complex128)
        if data.ndim != 2 or data.shape[1] != WINDOW_LENGTH:
            raise ValueError("expected rows of three complex samples")
        t = data.sum(axis=1) / math.sqrt(WINDOW_LENGTH)
        # Explicit coordinates preserve opposite-quadrant symmetry under i*y.
        real = (t.real - t.imag) / math.sqrt(2)
        imag = (t.real + t.imag) / math.sqrt(2)
        real_bin = np.searchsorted(self.cuts, real, side="right")
        imag_bin = np.searchsorted(self.cuts, imag, side="right")
        return self.bins * real_bin + imag_bin


def self_check():
    """Check table centroids by quadrature and rotations on new synthetic windows."""
    from scipy.integrate import quad
    from scipy.special import expit

    rng = np.random.default_rng(73053)
    windows = rng.normal(size=(4096, WINDOW_LENGTH)) + 1j*rng.normal(size=(4096, WINDOW_LENGTH))
    for bits in (4, 8):
        model = Hand(bits)
        codes = model.encode(windows)
        assert np.unique(codes).size == 2 ** (bits // 2)
        assert model.table.shape == (2 ** (bits // 2), 4)
        np.testing.assert_allclose(model.table.sum(axis=1), 1, rtol=0, atol=3e-16)
        np.testing.assert_array_equal(model.table[model.encode(1j * windows)],
                                      np.roll(model.table[codes], 1, axis=1))
        edges = np.r_[-np.inf, model.cuts, np.inf]
        def density(t):
            norm = 1 / (2 * NOISE_SD * math.sqrt(2 * math.pi))
            return norm * sum(math.exp(-0.5 * ((t - s * MEAN) / NOISE_SD)**2)
                              for s in (-1, 1))
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            support, _ = quad(density, a, b)
            numerator, _ = quad(lambda t: density(t) * expit(2 * MEAN * t / NOISE_SD**2), a, b)
            reconstructed = model.table[j * model.bins, [0, 3]].sum()
            np.testing.assert_allclose(reconstructed, numerator / support, rtol=0, atol=1e-12)
    print("hand cache: 4,096 new windows, both budgets, all code states, rotation and quadrature checks passed")


if __name__ == "__main__":
    self_check()
