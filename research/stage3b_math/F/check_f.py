"""Stage 3B planned public CFO factors and a zero-support identifiability witness."""

import argparse
import itertools
import json
import math


RATES = {
    "train": (-0.25, -0.10, 0.0, 0.10, 0.25),
    "dev": (-0.18, 0.18),
    "new_test": (-0.55, -0.35, 0.35, 0.55),
}
TIMES = ((2, 2, 4), (2, 3, 1))


def public_factors(omega, h, ell, pilot_time):
    return [
        [math.cos(omega * t), math.sin(omega * t)]
        for t in (h, h + ell, pilot_time)
    ]


def design():
    """Both hosts and both matched arms receive the same generated observations."""
    for split, rates in RATES.items():
        for omega, (h, ell, tp) in itertools.product(rates, TIMES):
            yield dict(split=split, omega=omega, h=h, ell=ell, pilot_time=tp,
                       snr_db=2.0, sigma_phi=0.1,
                       public_factors=public_factors(omega, h, ell, tp))


def example_posterior(angle, orientation):
    masses = [math.exp(2 * math.cos(orientation * angle - k * math.pi / 2))
              for k in range(4)]
    return [mass / sum(masses) for mass in masses]


def verify():
    rows = list(design())
    assert len(rows) == len({(r["split"], r["omega"], r["h"], r["ell"],
                              r["pilot_time"]) for r in rows}) == 22
    for left, right in itertools.combinations(RATES.values(), 2):
        assert set(left).isdisjoint(right)
    for r in rows:
        pos = r["public_factors"]
        neg = public_factors(-r["omega"], r["h"], r["ell"], r["pilot_time"])
        for (c, s), (cn, sn) in zip(pos, neg):
            assert math.isclose(c * c + s * s, 1)
            assert math.isclose(c, cn) and math.isclose(s, -sn)
    assert example_posterior(0, -1) == example_posterior(0, 1)
    p = example_posterior(math.pi / 2, 1)
    q = example_posterior(math.pi / 2, -1)
    assert p.index(max(p)) == 1 and q.index(max(q)) == 3
    return {"factor_design_rows": len(rows), "heldout_angle": math.pi / 2,
            "opposite_orientation_argmax": [p.index(max(p)), q.index(max(q))],
            "heldout_kl_nats": sum(a * math.log(a / b) for a, b in zip(p, q))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true", help="Emit planned conditions as JSON")
    args = parser.parse_args()
    result = verify()
    print(json.dumps(list(design()) if args.grid else result, indent=2))
