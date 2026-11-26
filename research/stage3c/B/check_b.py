"""Analytic Stage 3C feature collision and uncertain-CFO checks (Python stdlib)."""
import json
import math


def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def entropy(p):
    return -sum(x * math.log(x) for x in p if x)


def noisy_sector_probs(theta, snr):
    # The two axis projections defining one QPSK sector are independent.
    c = math.sqrt(2 * snr)
    return [
        normal_cdf(c * math.cos(theta - k*math.pi/2 + math.pi/4))
        * normal_cdf(c * math.cos(theta - k*math.pi/2 - math.pi/4))
        for k in range(4)
    ]


def wrapped_sector_probs(mu, sd):
    # Count all periodic copies of each sector in [-10sd,+10sd] about the mean.
    half_width = math.pi / 4
    copies = math.ceil((abs(mu) + 10*sd + 2*math.pi) / (2*math.pi)) + 1
    p = []
    for k in range(4):
        centers = (k*math.pi/2 + 2*math.pi*n for n in range(-copies, copies+1))
        p.append(sum(normal_cdf((x+half_width-mu)/sd)
                     - normal_cdf((x-half_width-mu)/sd) for x in centers))
    assert abs(sum(p)-1) < 1e-12
    return p


def grid_labels(turns_per_unit, h, tp):
    # Enumerate equiprobable gauge and source labels. Select exact observations
    # (first received orientation, second received orientation, pilot) == (0,0,0).
    retained = []
    for g in range(4):
        for x1 in range(4):
            for x2 in range(4):
                received = ((x1+g) % 4, (x2+g+turns_per_unit*h) % 4,
                            (g+turns_per_unit*tp) % 4)
                if received == (0, 0, 0):
                    retained.append(((x2-x1) % 4, x1))
    assert len(retained) == 1
    return retained[0]


def phase_features(omega, times):
    return [(math.cos(omega*t), math.sin(omega*t)) for t in times]


def main():
    # Scenario A and B share even the unwrapped total phase.
    ab = [grid_labels(1, 1, 1), grid_labels(1, 2, 2)]
    assert ab == [(3, 1), (2, 2)]
    # Same known time tuple and same wrapped final angle, different offsets.
    alias = [grid_labels(0, 1, 2), grid_labels(1, 1, 2)]
    assert alias == [(0, 0), (3, 2)]
    assert abs(complex(math.cos(2*math.pi), math.sin(2*math.pi)) - 1) < 1e-14
    # Identical endpoint means at all three roles, with different diffusion.
    sig = 0.2
    f1, f2 = phase_features(0.05, (0, 2, 2)), phase_features(0.0005, (0, 200, 200))
    assert f1 == f2
    p = [wrapped_sector_probs(0.1, sig*math.sqrt(t)) for t in (2, 200)]
    assert p[0][0] > 0.98 and p[1][0] < 0.28
    # Noiseless initial cardinal orientation and h=0. Future SNR=2 dB.
    rho = 10**0.2
    plus, minus = [noisy_sector_probs(s * math.pi/2, rho) for s in (1, -1)]
    mix = [(a+b)/2 for a, b in zip(plus, minus)]
    for q in (plus, minus, mix):
        assert abs(sum(q)-1) < 1e-12
    known_acc = (max(plus) + max(minus))/2
    unknown_acc = max(mix)
    gain = entropy(mix) - (entropy(plus)+entropy(minus))/2
    assert known_acc > unknown_acc and gain > 0
    # Sign is shared at both times; independent draws create paths of zero mass.
    shared = [(s, s) for s in (-1, 1)]
    independent = [(s1, s2) for s1 in (-1, 1) for s2 in (-1, 1)]
    diff = lambda x: sum(a != b for a, b in x) / len(x)
    print(json.dumps({
        "unwrapped_total_collision_q1_q2": ab,
        "wrapped_total_alias_with_same_times_q1_q2": alias,
        "diffusion_collision": {"time_h_T_tp": [[0, 2, 2], [0, 200, 200]],
                                "omega": [0.05, 0.0005], "sigma": sig,
                                "shared_phasors": f1, "q3_sector_probabilities": p},
        "unknown_sign_oracle_u1": {"rho": rho, "forward": plus, "reverse": minus,
                                   "mixture": mix, "known_sign_accuracy": known_acc,
                                   "unobserved_sign_accuracy": unknown_acc,
                                   "known_sign_logloss_nats": (entropy(plus)+entropy(minus))/2,
                                   "unobserved_sign_logloss_nats": entropy(mix),
                                   "value_of_sign_nats": gain},
        "shared_vs_independent_sign_flip_probability": [diff(shared), diff(independent)]
    }, indent=2))


if __name__ == "__main__":
    main()
