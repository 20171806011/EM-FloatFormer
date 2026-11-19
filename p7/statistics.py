"""Exact two-sided 1024-state paired sign-flip permutation inference & Holm-Bonferroni correction."""
from fractions import Fraction
import math


def exact_sign_test(differences):
    """Exact two-sided paired sign-flip test across N differences (positives = improvement)."""
    vals = [float(d) for d in differences]
    n = len(vals)
    if n == 0 or n > 16:
        raise ValueError(f"Exact enumeration expects 1 <= N <= 16, got {n}")
    # Exhaustive exact sum distribution for small N (here N=10 -> 1024 states)
    # Count exact sign-flip configurations where |sum(s_i * d_i)| >= |observed_sum| (with tol=1e-12)
    observed = sum(vals)
    abs_obs = abs(observed)
    extreme = 0
    total = 1 << n
    tol = 1e-12 * max(abs_obs, sum(abs(v) for v in vals), 1.0)
    for mask in range(total):
        s = 0.0
        for i in range(n):
            s += vals[i] if (mask >> i) & 1 else -vals[i]
        if abs(s) >= abs_obs - tol:
            extreme += 1
    frac = Fraction(extreme, total)
    wins = sum(1 for v in vals if v > 1e-12)
    ties = sum(1 for v in vals if abs(v) <= 1e-12)
    losses = sum(1 for v in vals if v < -1e-12)
    mean_diff = observed / n
    sd = math.sqrt(sum((v - mean_diff) ** 2 for v in vals) / max(n - 1, 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    # Two-sided t_{n-1, 0.975} critical factor for descriptive 95% interval
    t_crit = {9: 2.2621571627409915, 2: 4.302652729911275}.get(n - 1, 1.959963984540054)
    ci95 = [mean_diff - t_crit * se, mean_diff + t_crit * se]
    return dict(
        n=n,
        wins=wins,
        ties=ties,
        losses=losses,
        mean_diff=mean_diff,
        ci95=ci95,
        raw_exact=f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator),
        raw_p=float(frac),
        differences=vals,
    )


def holm_adjust(rows):
    """Apply exact step-down Holm-Bonferroni correction within one registered family."""
    m = len(rows)
    indexed = sorted(enumerate(rows), key=lambda item: Fraction(item[1]["raw_exact"]))
    running_max = Fraction(0, 1)
    adjusted = [None] * m
    for rank, (orig_idx, row) in enumerate(indexed):
        factor = m - rank
        val = min(Fraction(1, 1), Fraction(row["raw_exact"]) * factor)
        if val < running_max:
            val = running_max
        running_max = val
        updated = dict(row)
        updated["holm_exact"] = f"{val.numerator}/{val.denominator}" if val.denominator != 1 else str(val.numerator)
        updated["holm_p"] = float(val)
        adjusted[orig_idx] = updated
    return adjusted
