"""Check the frozen strong-matching certificate using only integer/rational arithmetic."""
import json
from fractions import Fraction as F
from math import comb
from pathlib import Path


def binomial(d, probability):
    return [comb(d, h) * probability**h * (1 - probability)**(d - h)
            for h in range(d + 1)]


plan = json.loads((Path(__file__).parent / "verification" / "PLAN.json").read_text())
cfg = plan["defaults"] | next(x for x in plan["scenarios"] if x["name"] == "strong")
m, k, d, gap = (cfg[key] for key in ("M", "k", "d", "gap_bits"))
rho, v0, noise = (F(str(cfg[key])) for key in ("rho", "v0", "sigma2"))
true_dist, null_dist = binomial(d, rho), binomial(d, F(1, 2))
good = sum(mass * sum(null_dist[h + gap:])**(m - 1)
           for h, mass in enumerate(true_dist))
b = (m - 1) * (rho / (1 - rho))**gap  # e^(-beta * 2 * gap / d) at matched temperature
r = b / (1 + b)
v1, gain = v0 * noise / (v0 + noise), v0**2 / (v0 + noise)
upper = v1 + 2 * gain * (good * r**2 + 1 - good)
lower = v0 - gain * F(k, m)
assert upper < lower
print(f"P(good)={float(good):.12f}  upper={float(upper):.12f}  lower={float(lower):.12f}")
print("Strict comparison verified using exact rational arithmetic.")
