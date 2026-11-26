"""Prospective Stage 3C host design. No fitted hosts or test observations are used."""
import argparse
import json
import math
from fractions import Fraction
from statistics import NormalDist


NORMAL = NormalDist()
BLOCK = 64
ALPHA = 0.025
MARGINAL_POWER = 0.95
Z = NORMAL.inv_cdf(1 - ALPHA)


def blocks_for(variance, delta, margin, half_width):
    """variance is the assumed or pilot-estimated variance of a block MEAN."""
    target_se = (delta - margin) / (Z + NORMAL.inv_cdf(MARGINAL_POWER))
    return math.ceil(variance / min(target_se, half_width / Z) ** 2)


def sizing(discordance, rho, nll_sd, pilot_sd=None):
    if not (0 < discordance <= 1 and 0 <= rho <= 1 and nll_sd > 0):
        raise ValueError("Require 0<discordance<=1, 0<=rho<=1, positive NLL SD.")
    # Jensen: E[(mean_j hit_gain_j)^2] <= mean_j P(discordant_j).
    deff = 1 + (BLOCK - 1) * rho
    var = {"accuracy": discordance * deff / BLOCK, "nll": nll_sd**2 * deff / BLOCK}
    if pilot_sd is not None:
        if any(x <= 0 for x in pilot_sd):
            raise ValueError("Pilot block-mean SDs must be positive.")
        var = {k: max(var[k], s*s) for k, s in zip(("accuracy", "nll"), pilot_sd)}
    specs = {"accuracy": (0.05, 0.02, 0.02), "nll": (0.06, 0.02, 0.03)}
    required = {k: blocks_for(var[k], *specs[k]) for k in specs}
    b = max(32, *required.values())
    intervals = {}
    for k, (gain, margin, _) in specs.items():
        se = math.sqrt(var[k] / b)
        intervals[k] = {"expected_95_half_width": Z * se,
                        "approx_power_against_margin": NORMAL.cdf((gain - margin)/se - Z),
                        "block_mean_sd": math.sqrt(var[k])}
    return {"discordance": discordance, "within_block_rho": rho, "nll_mother_sd": nll_sd,
            "design_effect": deff, "required_blocks_by_endpoint": required,
            "blocks": b, "independent_blocks": b, "source_mothers": b*BLOCK,
            "prospective_precision": intervals}


def factor_grid():
    # Fractions keep exact time/product equality checks independent of float tolerances.
    h_train = [(2, 2), (2, 3)]
    train_rates = [Fraction(0), Fraction(1, 10), Fraction(1, 4)]
    train_angles = {w*(h+ell) for w in train_rates for h, ell in h_train}
    rows = []
    for w in (Fraction(1, 5), Fraction(2, 5)):
        for beta in (Fraction(4, 5), Fraction(8, 5)):
            t = beta / w
            assert t.denominator == 1 and beta not in train_angles and w not in train_rates
            assert 0 < float(beta) < math.pi
            rows.append({"abs_rate_rad_per_step": float(w), "abs_Q3_angle_rad": float(beta),
                         "rate_region": "inside" if w < max(train_rates) else "outside",
                         "angle_region": "inside" if beta < max(train_angles) else "outside",
                         "h": int(t)//2, "ell": int(t)//2, "pilot_time": int(t),
                         "evaluate_signs": [-1, 1]})
    assert len({(r["rate_region"], r["angle_region"]) for r in rows}) == 4
    return rows


def invariance_plan(blocks, alpha=0.05, failure_ceiling=0.05):
    """One uniform preselected sentinel per block; exact binomial one-sided test."""
    # Reject H0: failure rate >= failure_ceiling iff P_H0[X<=observed] <= alpha.
    log_mass = blocks * math.log1p(-failure_ceiling)
    cdf = math.exp(log_mass)
    allowed = -1 if cdf > alpha else 0
    for k in range(1, blocks + 1):
        log_mass += math.log((blocks + 1 - k) / k) + math.log(failure_ceiling / (1 - failure_ceiling))
        cdf += math.exp(log_mass)
        if cdf > alpha:
            break
        allowed = k
    return {"independent_sentinels": blocks, "failure_ceiling": failure_ceiling,
            "one_sided_alpha": alpha, "max_failures_to_certify_ceiling": allowed,
            "zero_failures_exact_upper": -math.expm1(math.log(alpha)/blocks),
            "minimum_blocks_if_zero_failures": math.ceil(math.log(alpha)/math.log1p(-failure_ceiling))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--discordance", type=float, default=0.35)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--nll-sd", type=float, default=0.50)
    p.add_argument("--pilot-block-sd", nargs=2, type=float, metavar=("HIT", "NLL"),
                   help="Optional conservative external pilot block-mean SDs")
    args = p.parse_args()
    proposed = sizing(args.discordance, args.rho, args.nll_sd, args.pilot_block_sd)
    result = {"is_host_measurement": False, "design": proposed,
              "conditional_invariance": invariance_plan(proposed["blocks"]),
              "factor_grid_Q3_only": factor_grid(),
              "sensitivity": [sizing(q, r, args.nll_sd) for q, r in
                              ((0.15, 0.0), (0.35, 0.0), (0.35, 0.03), (0.5, 0.03))]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
