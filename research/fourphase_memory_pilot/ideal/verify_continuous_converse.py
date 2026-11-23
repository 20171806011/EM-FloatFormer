"""Independent zero-drift witness using the exact (unexpanded) posterior."""

import cmath
import json
import math

SYMBOLS = (1, 1j, -1, -1j)


def posterior(observation, snr):
    weights = [math.exp(2 * snr * (observation * s.conjugate()).real) for s in SYMBOLS]
    return [w / sum(weights) for w in weights]


def winner_and_margin(y, p):
    a = posterior(y, snr=1)
    b = posterior(p, snr=1)
    q = [sum(a[(x + g) % 4] * b[g] for g in range(4)) for x in range(4)]
    best, second, *_ = sorted(range(4), key=q.__getitem__, reverse=True)
    return best, q[best] - q[second]


def check():
    phases = [math.pi * j / 48 for j in range(1, 10)]
    # Im(m)/Re(m) = sinh(2 Im y)/sinh(2 Re y); choose Re y = 1.
    centers = [1 + 0.5j * math.asinh(math.sinh(2) * math.tan(phi)) for phi in phases]
    margins = []
    for j, first in enumerate(centers):
        for k in range(j + 1, len(centers)):
            theta = (phases[j] + phases[k]) / 2 - math.pi / 4
            pilot_center = 0.001 * cmath.exp(1j * theta)
            for data_dx in (0, 1e-5, -1e-5, 1e-5j, -1e-5j):
                for pilot_dx in (0, 1e-6, -1e-6, 1e-6j, -1e-6j):
                    p = pilot_center + pilot_dx
                    win1, gap1 = winner_and_margin(first + data_dx, p)
                    win2, gap2 = winner_and_margin(centers[k] + data_dx, p)
                    assert (win1, win2) == (0, 1)
                    margins.append(min(gap1, gap2))
    return {"cache_states_in_pigeonhole_example": 8, "data_centers": len(centers),
            "data_center_pairs": math.comb(len(centers), 2),
            "perturbed_paired_witnesses": len(margins),
            "smallest_computed_posterior_margin": min(margins)}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
