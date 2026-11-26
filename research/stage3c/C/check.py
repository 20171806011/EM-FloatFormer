"""Exact multinomial enumeration for a four-label finite-cell experiment."""
import itertools
import json
import math


P = (0.48, 0.42, 0.07, 0.03)
TEACHERS = {
    "exact": P,
    "close": (0.46, 0.44, 0.07, 0.03),
    "reversed": (0.24, 0.64, 0.08, 0.04),
}


def winner(row):
    # Resolve equal values deterministically at the lowest physical label.
    return max(range(len(row)), key=lambda k: (round(row[k], 14), -k))


def sqdist(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def kl(p, q):
    return sum(x*math.log(x/y) for x, y in zip(p, q) if x)


def enumerate_counts(n):
    # Stars and bars, with exact integer multiplicities and float probabilities.
    for cuts in itertools.combinations(range(n+3), 3):
        edges = (-1,) + cuts + (n+3,)
        counts = tuple(b-a-1 for a, b in zip(edges, edges[1:]))
        ways = math.factorial(n) // math.prod(math.factorial(k) for k in counts)
        yield tuple(k/n for k in counts), ways * math.prod(x**k for x,k in zip(P, counts))


def calc(rows, alpha, teacher):
    err2 = acc_regret = 0.0
    for observed, prob in rows:
        a = tuple((1-alpha)*x+alpha*y for x,y in zip(observed, teacher))
        err2 += prob * sqdist(a, P)
        acc_regret += prob * (max(P)-P[winner(a)])
    return {"expected_excess_brier": err2, "expected_accuracy_regret": acc_regret}


def check():
    v = 1-sum(x*x for x in P)
    report = {"specified_true_p": P, "conditional_hard_gradient_trace": v,
              "rows": {}, "truth_perturbed_example": {}}
    for n in (8, 32):
        rows = list(enumerate_counts(n))
        assert math.isclose(sum(w for _,w in rows), 1., abs_tol=1e-12)
        conds = {}
        for name, teacher in TEACHERS.items():
            bias2 = sqdist(P, teacher)
            best = v/(v+n*bias2)
            arms = {"hard": 0.0, "half": 0.5, "soft": 1.0, "oracle_weight": best}
            entry = {k: calc(rows, a, teacher) for k, a in arms.items()}
            for k, a in arms.items():
                pred = (1-a)**2 * v/n + a*a*bias2
                assert math.isclose(entry[k]["expected_excess_brier"], pred, abs_tol=2e-13)
                assert entry["oracle_weight"]["expected_excess_brier"] <= pred+1e-12
            if name == "exact":
                assert math.isclose(entry["soft"]["expected_excess_brier"], 0., abs_tol=1e-14)
            conds[name] = {"teacher": teacher, "bias_squared": bias2, "oracle_alpha": best,
                           "ce_excess_of_unrestricted_pure_soft_fit": kl(P, teacher),
                           "arms": entry}
        report["rows"][str(n)] = {"multinomial_outcomes": len(rows), "teachers": conds}
    q = TEACHERS["reversed"]
    cutoff = (P[0]-P[1])/((P[0]-P[1])+(q[1]-q[0]))
    assert winner(tuple((1-0.12)*p+0.12*t for p,t in zip(P,q))) == 0
    assert winner(tuple((1-0.14)*p+0.14*t for p,t in zip(P,q))) == 1
    report["truth_perturbed_example"] = {
        "switch_alpha": cutoff,
        "population_regret_above_switch": P[0]-P[1],
        "pure_soft_excess_nll_nats": kl(P,q),
    }
    return report


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True, allow_nan=False))
