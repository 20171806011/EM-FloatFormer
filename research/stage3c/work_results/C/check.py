"""Independently enumerate C's finite cells and audit the frozen C4 cache on CPU."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


STAGE = Path(__file__).resolve().parents[2]
REPO = STAGE.parents[1]
P = tuple(F(x, 100) for x in (48, 42, 7, 3))
TEACHERS = {
    "close": tuple(F(x, 100) for x in (46, 44, 7, 3)),
    "exact": P,
    "reversed": tuple(F(x, 100) for x in (24, 64, 8, 4)),
}


def outcomes(n, p):
    """Traverse a multinomial as successive conditional binomials."""
    if len(p) == 1:
        yield (n,), F(1)
        return
    head = p[0] / sum(p)
    for k in range(n + 1):
        weight = math.comb(n, k) * head**k * (1 - head)**(n - k)
        for tail, tail_weight in outcomes(n - k, p[1:]):
            yield (k, *tail), weight * tail_weight


def risk(rows, p, q, alpha):
    mse = regret = F(0)
    for n, counts, prob in rows:
        mixed = [(1 - alpha) * F(k, n) + alpha * t for k, t in zip(counts, q)]
        # max returns the first attaining index, including exact rational ties.
        choice = max(range(len(mixed)), key=mixed.__getitem__)
        mse += prob * sum((x - y)**2 for x, y in zip(mixed, p))
        regret += prob * (max(p) - p[choice])
    return mse, regret


def compare(reference, replay):
    if isinstance(reference, dict):
        assert reference.keys() == replay.keys()
        return max((compare(v, replay[k]) for k, v in reference.items()), default=0)
    if isinstance(reference, list):
        assert len(reference) == len(replay)
        return max(compare(a, b) for a, b in zip(reference, replay))
    if isinstance(reference, float):
        assert math.isclose(reference, replay, rel_tol=0, abs_tol=2e-13)
        return abs(reference - replay)
    assert reference == replay
    return 0


def enumerate_baseline():
    baseline = STAGE / "C"
    ref = json.loads((baseline / "results.json").read_text())
    replay = json.loads(subprocess.check_output([sys.executable, "-B", baseline / "check.py"]))
    replay_delta = compare(ref, replay)
    v = 1 - sum(x*x for x in P)
    independent = {"specified_true_p": list(map(float, P)), "conditional_hard_gradient_trace": float(v),
                   "rows": {}}
    for n in (8, 32):
        rows = [(n, k, w) for k, w in outcomes(n, P)]
        assert sum(w for _, _, w in rows) == 1
        teachers = {}
        for name, q in TEACHERS.items():
            b = sum((x-y)**2 for x, y in zip(P, q))
            alpha = v / (v + n*b)
            arms = {}
            for arm, a in (("hard", F(0)), ("half", F(1, 2)), ("soft", F(1)), ("oracle_weight", alpha)):
                mse, regret = risk(rows, P, q, a)
                assert mse == (1-a)**2 * v/n + a*a*b
                arms[arm] = {"expected_excess_brier": float(mse), "expected_accuracy_regret": float(regret)}
            teachers[name] = {"teacher": list(map(float, q)), "bias_squared": float(b),
                              "oracle_alpha": float(alpha), "arms": arms,
                              "ce_excess_of_unrestricted_pure_soft_fit": sum(float(x)*math.log(float(x/y))
                                                                            for x, y in zip(P, q))}
        independent["rows"][str(n)] = {"multinomial_outcomes": len(rows), "teachers": teachers}
    q = TEACHERS["reversed"]
    gap = P[0] - P[1]
    independent["truth_perturbed_example"] = {
        "switch_alpha": float(gap / (gap + q[1] - q[0])),
        "population_regret_above_switch": float(gap),
        "pure_soft_excess_nll_nats": independent["rows"]["8"]["teachers"]["reversed"]["ce_excess_of_unrestricted_pure_soft_fit"],
    }
    delta = compare(ref, independent)
    return {"replay_max_delta": replay_delta, "independent_max_delta": delta, **independent}


def new_counterexample():
    n = 3
    p = (F(11, 20), F(9, 20))
    q = (F(1, 5), F(4, 5))
    v = 1 - sum(x*x for x in p)
    b = sum((x-y)**2 for x, y in zip(p, q))
    alpha = v / (v + n*b)
    rows = [(n, k, w) for k, w in outcomes(n, p)]
    hard_mse, hard_regret = risk(rows, p, q, F(0))
    mse, regret = risk(rows, p, q, alpha)
    assert mse < hard_mse and regret > hard_regret
    return {"n": n, "enumerated_outcomes": len(rows), "p": list(map(float, p)), "q": list(map(float, q)),
            "alpha": float(alpha), "hard_mse": float(hard_mse), "blend_mse": float(mse),
            "hard_regret": float(hard_regret), "blend_regret": float(regret)}


def cache_audit():
    matched = REPO / "research/matched_cache"
    sys.path[:0] = [str(matched), str(STAGE / "D")]
    from check_d import BATCH, CONFIRM_SEED, SEED, TABLE_SHA, WEIGHT_SHA, simulate
    from learned import load

    weight_path = matched / "weights/equivariant_b8_s17.npz"
    table_path = REPO / "research/stage3b_math/A/c4_s17_lookup.json"
    table_text = table_path.read_text()
    assert hashlib.sha256(table_text.encode()).hexdigest() == TABLE_SHA
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == WEIGHT_SHA
    table = json.loads(table_text)
    model = load(weight_path)
    assert model.equivariant and len(model.centers) == 4
    total = 2**18
    first = simulate(model, total, SEED)
    second = simulate(model, total, CONFIRM_SEED)
    # A fixed, non-identity linear candidate scorer in relative-label coordinates.
    jac = np.array([[1., 0.], [0., 1.], [-1., 0.5], [0.25, -0.5]])
    scores = jac @ np.array([0.2, -0.3])
    r = np.exp(scores) / np.exp(scores).sum()
    hard = (r - np.eye(4)) @ jac
    mix = 0.5
    reuse = 0.5
    tasks = {}
    for task, a in first.items():
        b = second[task]
        teacher = np.array([cell[task] for cell in table["cells"]])
        p = (a + b) / (a + b).sum(1, keepdims=True)
        g = (r-p) @ jac
        errors = hard[None, :, :] - g[:, None, :]
        actual_mean = p @ hard
        cov = np.einsum("ni,nia,nib->nab", p, errors, errors)
        predicted_cov = np.array([jac.T @ (np.diag(x) - np.outer(x, x)) @ jac for x in p])
        np.testing.assert_allclose(actual_mean, g, atol=1e-14, rtol=0)
        np.testing.assert_allclose(cov, predicted_cov, atol=1e-14, rtol=0)
        # Reuse the very label whose hard gradient is scored (n=1).
        fixed_err = (r-teacher) @ jac - g
        reused_err = (1-reuse)*fixed_err[:, None, :] + reuse*errors
        cross = np.einsum("ni,nia,nia->n", p, errors, reused_err)
        var = np.trace(cov, axis1=1, axis2=2)
        actual = np.einsum("ni,nia,nia->n", p, (1-mix)*errors+mix*reused_err,
                           (1-mix)*errors+mix*reused_err)
        marginal_teacher = np.einsum("ni,nia,nia->n", p, reused_err, reused_err)
        wrong = (1-mix)**2*var + mix**2*marginal_teacher
        np.testing.assert_allclose(cross, reuse*var)
        np.testing.assert_allclose(actual, wrong+2*mix*(1-mix)*cross)
        split_a = (teacher - a/a.sum(1, keepdims=True)) @ jac
        split_b = (teacher - b/b.sum(1, keepdims=True)) @ jac
        tasks[task] = {"counts_seed_a": a.tolist(), "counts_seed_b": b.tolist(),
                       "min_pooled_cell_count": int((a+b).sum(1).min()),
                       "max_gradient_mean_delta": float(np.max(abs(actual_mean-g))),
                       "max_covariance_delta": float(np.max(abs(cov-predicted_cov))),
                       "mean_conditional_hard_variance": float(var.mean()),
                       "mean_mse_omitting_cross_term": float(wrong.mean()),
                       "mean_mse_with_reuse": float(actual.mean()),
                       "mean_missing_cross_term": float((actual-wrong).mean()),
                       "mean_split_projected_teacher_error2_estimate": float(np.mean(np.sum(split_a*split_b, axis=1))),
                       "teacher_accuracy_on_pooled_mothers": float((a+b)[np.arange(16), teacher.argmax(1)].sum()/(2*total))}
    return {"samples_per_seed": total, "seeds": [SEED, CONFIRM_SEED], "batch": BATCH,
            "weight_sha256": WEIGHT_SHA, "teacher_table_sha256": TABLE_SHA,
            "teacher_integration_seed": table["integration_seed"], "jacobian": jac.tolist(),
            "teacher_reuse_fraction": reuse, "mix": mix, "tasks": tasks}


if __name__ == "__main__":
    print(json.dumps({"baseline": enumerate_baseline(), "counterexample": new_counterexample(),
                      "c4_cache": cache_audit()}, indent=2, sort_keys=True, allow_nan=False))
