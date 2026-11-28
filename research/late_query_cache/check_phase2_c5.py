"""Finite end-to-end check of the C5 dual-scale construction (Python standard library).

The random trials check the implemented codec and fine-error guarantee on the
good event. They do not estimate the asymptotic converse or certify Gaussian tails.
"""

import itertools
import math
import random


def encode_edges(coarse, width):
    half = width // 2
    mask = (1 << half) - 1
    return [(coarse[i] & mask) ^ (coarse[(i + 1) % 5] >> half) for i in range(5)]


def decode_word(i, packets, previous, following, width):
    half = width // 2
    mask = (1 << half) - 1
    left = (previous & mask) ^ packets[(i - 1) % 5]
    right = packets[i] ^ (following >> half)
    return (left << half) | right


def recover_bin(estimated, residue):
    if estimated % 3 == residue:
        return estimated
    if (estimated - 1) % 3 == residue:
        return estimated - 1
    return estimated + 1


def exhaustive():
    cases = 0
    for coarse in itertools.product(range(4), repeat=5):
        packets = encode_edges(coarse, 2)
        for i in range(5):
            left, right = coarse[(i - 1) % 5], coarse[(i + 1) % 5]
            for e_left, e_right in itertools.product((-1, 0, 1), repeat=2):
                decoded = decode_word(i, packets, recover_bin(left + e_left, left % 3),
                                      recover_bin(right + e_right, right % 3), 2)
                assert decoded == coarse[i]
                cases += 1
    # Deliberately force a two-bin error; the false residue-compatible candidate
    # is three bins from truth and changes a demanded half-word.
    assert recover_bin(2, 0) == 3
    assert decode_word(0, [0] * 5, recover_bin(2, 0), 0, 2) != 0
    return cases


def probe(eta, trials, seed):
    rng = random.Random(seed)
    eps = 2.0 ** -20
    sigma = eps ** (eta / 2)
    threshold = math.sqrt(2 * math.log(8 / eps))
    fine = math.ceil(math.log2(1 / math.sqrt(2 * eps)))
    available = max(0, min(fine, math.floor(math.log2(1 / (threshold * sigma)))))
    coarse_width = 2 * (available // 2)
    levels = 1 << coarse_width
    fine_levels = 1 << fine
    low = fine - coarse_width
    remainder_mask = (1 << low) - 1
    squared_errors = [0.0] * 5
    off_good = 0
    corrected_crossings = 0
    for _ in range(trials):
        x = [rng.random() for _ in range(5)]
        words = [math.floor(value * fine_levels) for value in x]
        coarse = [word >> low for word in words]
        residual = [word & remainder_mask for word in words]
        packets = encode_edges(coarse, coarse_width)
        for i in range(5):
            corrected = []
            good = True
            for j in ((i - 1) % 5, (i + 1) % 5):
                z = rng.gauss(0, 1)
                preliminary = math.floor((x[j] + sigma * z) * levels)
                if coarse_width == 0:
                    corrected.append(0)
                else:
                    k = recover_bin(preliminary, coarse[j] % 3)
                    if preliminary != coarse[j] and k == coarse[j]:
                        corrected_crossings += 1
                    corrected.append(max(0, min(levels - 1, k)))
                    good = good and abs(z) < threshold
            high = decode_word(i, packets, *corrected, coarse_width)
            full = (high << low) | residual[i]
            estimate = (full + 0.5) / fine_levels
            error = (estimate - x[i]) ** 2
            if good:
                assert full == words[i] and error <= 1 / (4 * fine_levels ** 2)
            else:
                off_good += 1
            squared_errors[i] += error
    payload = 5 * low + 5 * (coarse_width // 2) + (10 if coarse_width else 0)
    return (eta, fine, coarse_width, payload, corrected_crossings, off_good,
            max(squared_errors) / trials)


if __name__ == "__main__":
    print("exhaustive neighbor decodes:", exhaustive())
    print("eta fine_bits coarse_bits payload_bits corrected_crossings out_of_guard worst_query_empirical_mse")
    for eta in (0, 0.5, 1):
        print(*probe(eta, trials=8192, seed=614029))
