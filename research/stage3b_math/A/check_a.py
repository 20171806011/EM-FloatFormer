"""Population Q1 and Q3 risks on the Stage 3 C4 generator, per mother.

Run python check_a.py. Optionally pass the path to research/matched_cache
with --matched-dir to include a Monte Carlo integration for equivariant_b8_s17.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr


RHO = 10.0 ** 0.2
NOISE_SD = (2 * RHO) ** -0.5
HALF_PI = math.pi / 2
LABELS = np.arange(4)


def normal_rule(mean, sd, order):
    points, weights = np.polynomial.hermite.hermgauss(order)
    return mean + math.sqrt(2) * sd * points, weights / math.sqrt(math.pi)


def hand_emission(theta):
    """Mass of the 16 frozen hand_b8 cells at an arbitrary input phase."""
    cut_p = np.array([0.25, 0.5, 0.75])
    cuts = NOISE_SD**2 / math.sqrt(2) * np.log(cut_p / (1 - cut_p))
    edges = np.r_[-np.inf, cuts, np.inf]
    ang = np.asarray(theta)[..., None] + math.pi / 4
    r = np.diff(ndtr((edges - np.cos(ang)) / NOISE_SD), axis=-1)
    i = np.diff(ndtr((edges - np.sin(ang)) / NOISE_SD), axis=-1)
    return (r[..., :, None] * i[..., None, :]).reshape((*ang.shape[:-1], 16))


def sector_emission(theta):
    p = ndtr(np.cos(theta + math.pi / 4) / NOISE_SD)
    q = ndtr(np.sin(theta + math.pi / 4) / NOISE_SD)
    return np.stack([p*q, (1-p)*q, (1-p)*(1-q), p*(1-q)], axis=-1)


def population(emission, static_table, order, rules=None):
    # h=2, ell=2, omega=.35, sigma=.10: independent Gaussian increments.
    offsets, weights = normal_rule(.7, math.sqrt(2)*.1, order)
    first = emission(HALF_PI * LABELS)  # [u1,z1]
    second = emission(offsets[:, None] + HALF_PI * LABELS)  # [j,u2,z2]
    marginal2 = np.einsum("j,juk->uk", weights, second)
    q1_joint = np.stack([first.T @ marginal2[(LABELS+k) % 4] / 16
                         for k in LABELS], axis=-1)
    theta3 = (offsets[:, None, None] + offsets[None, :, None]
              + HALF_PI * LABELS[None, None, :])
    targets = np.einsum("l,jluk->juk", weights, sector_emission(theta3))
    q3_joint = np.einsum("ui,j,jv,juk->ivk", first, weights, second.mean(axis=1),
                         targets, optimize=True) / 4
    support = q1_joint.sum(axis=-1)
    np.testing.assert_allclose(q3_joint.sum(axis=-1), support, atol=3e-15)
    np.testing.assert_allclose(support.sum(), 1, atol=3e-14)
    # Stage 3 comparator's C4 continuous table procedure.
    k = np.arange(4)
    masses = np.zeros(4)
    for image in range(-3, 4):
        edge = HALF_PI * k + 2*math.pi*image - .7
        masses += (ndtr((edge + math.pi/4)/(.1*math.sqrt(2)))
                   - ndtr((edge - math.pi/4)/(.1*math.sqrt(2))))
    rotated = sum(masses[k] * np.roll(static_table, -k, axis=1) for k in LABELS)
    legacy1 = np.stack([static_table @ rotated[:, (LABELS+k) % 4].T
                        for k in LABELS], axis=-1)
    direct3 = static_table @ np.einsum("j,juk->uk", weights, targets)
    legacy3 = np.broadcast_to(direct3[:, None, :], q3_joint.shape)
    scores = {}
    for key, joint, published in (("q1", q1_joint, legacy1), ("q3", q3_joint, legacy3)):
        opt = joint / support[..., None]
        def score(prob):
            hits = np.take_along_axis(joint, prob.argmax(-1)[..., None], axis=-1).sum()
            return dict(accuracy=float(hits), nll=float(-np.sum(joint * np.log(prob))))
        scores[key] = dict(opt=score(opt), stage3=score(published),
                           max_abs_posterior_difference=float(np.max(np.abs(opt-published))))
        if rules is not None:
            rules[key] = {"opt": opt, "stage3": published}
    return scores


def independent_check(model, rules, count=131072, seed=79561):
    """Test fixed decision rules on independent forward simulations; paired s.e."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(2)
    squares = np.zeros(2)
    totals = np.zeros((2, 2))
    for start in range(0, count, 4096):
        n = min(count-start, 4096)
        u, v = rng.integers(4, size=(2, n))
        phi2, extra = rng.normal(.7, math.sqrt(2)*.1, (2, n))
        def window(phase):
            noise = rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3))
            return np.exp(1j*phase[:, None])/math.sqrt(3) + NOISE_SD * noise
        z1 = model.encode(window(u*HALF_PI))
        z2 = model.encode(window(v*HALF_PI+phi2))
        third = window(u*HALF_PI+phi2+extra).sum(axis=1)
        k = np.floor((np.angle(third)+math.pi/4)/HALF_PI).astype(int) % 4
        for idx, (key, target) in enumerate([("q1", (v-u) % 4), ("q3", k)]):
            preds = [rules[key][name][z1,z2].argmax(-1) == target for name in ("opt","stage3")]
            totals[idx] += np.sum(preds, axis=1)
            delta = preds[0].astype(float) - preds[1]
            hits[idx] += delta.sum()
            squares[idx] += delta @ delta
    return {key: dict(opt_acc=totals[idx,0]/count, stage3_acc=totals[idx,1]/count,
                      paired_se=math.sqrt((squares[idx]-hits[idx]**2/count)/(count*(count-1))))
            for idx,key in enumerate(("q1","q3"))}


def learned_estimate(matched_dir, per_basis, seed, order):
    """Gaussian importance integration, weighted from four cardinal proposals.

    Pool all exact quarter-turns of proposal records, exploiting this encoder's C4
    code law. Reweight by known full three-complex-sample Gaussian likelihood
    ratio (the matched sum suffices for that ratio). Self-normalize per angle.
    """
    sys.path.insert(0, str(matched_dir))
    import learned
    model = learned.load(matched_dir / "weights" / "equivariant_b8_s17.npz")
    rng = np.random.default_rng(seed)
    observations = (1 / math.sqrt(3) + NOISE_SD *
                    (rng.normal(size=(per_basis, 3)) + 1j*rng.normal(size=(per_basis, 3))))
    means = observations.sum(axis=1) / math.sqrt(3)
    base_codes = model.encode(observations)
    ts = np.concatenate([means * (1j**u) for u in LABELS])
    codes = np.concatenate([((base_codes + 4*u) % 16) for u in LABELS])
    priors = np.stack([2*RHO*np.real(ts * ((-1j)**u)) for u in LABELS])
    log_denom = np.logaddexp.reduce(priors, axis=0) - math.log(4)
    def emission(angles):
        rows = []
        for angle in np.asarray(angles).ravel():
            log_w = 2*RHO*np.real(ts*np.exp(-1j*angle)) - log_denom
            freq = np.bincount(codes, weights=np.exp(log_w), minlength=16)
            rows.append(freq / freq.sum())
        return np.array(rows).reshape((*np.shape(angles), 16))
    rules = {}
    est = population(emission, model.table, order, rules)
    est["independent_forward"] = independent_check(model, rules)
    return est


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matched-dir", type=Path)
    p.add_argument("--per-basis", type=int, default=40_000)
    args = p.parse_args()
    card = hand_emission(HALF_PI * LABELS).T
    table = card / card.sum(axis=1, keepdims=True)
    hand = population(hand_emission, table, 24)
    high = population(hand_emission, table, 36)
    for key in hand:
        for method in ("opt", "stage3"):
            np.testing.assert_allclose(list(hand[key][method].values()),
                                       list(high[key][method].values()), atol=3e-12, rtol=0)
    assert hand["q1"]["stage3"]["nll"] - hand["q1"]["opt"]["nll"] > .01
    assert hand["q3"]["max_abs_posterior_difference"] > 1e-5
    result = {"hand_b8_C4": hand}
    if args.matched_dir:
        result["learned_b8_s17_C4_estimated"] = {
            str(seed): learned_estimate(args.matched_dir, args.per_basis, seed, 20)
            for seed in (41117, 41129)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
