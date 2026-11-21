"""Finite, explicitly normalized posterior-fusion benchmark; Python standard library."""
import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def probability(match, reliability):
    return reliability if match else 1 - reliability


def number(x):
    """Include exact strings and decimal approximations for every reported rational."""
    return {"fraction": str(x), "decimal": float(x)}


def identity_score(weights, values):
    return sum(a * v for a, v in zip(weights, values))


class Model:
    def __init__(self, protocol):
        law = protocol["true"]
        self.p = F(law["p_branch_0"])
        self.a = F(law["p_context_sign_correct"])
        self.b = F(law["p_target_amplitude_label_correct"])
        self.s = F(law["p_target_bpsk_sign_correct"])
        self.v = F(law["p_training_probe_sign_correct"])
        self.h = tuple(law["H_order"])
        self.code = [tuple(map(int, c)) for c in law["codewords_for_00_01_10_11"]]
        self.y = list(itertools.product(range(2), repeat=1 + len(self.code[0])))
        self.qz = [sum(self.p_rz(r, z, self.a) for r in range(2)) for z in range(2)]

    def p_rz(self, r, z, a):
        return probability(r == 0, self.p) * probability(z == r, a)

    def weights0(self, a):
        return [self.p_rz(0, z, a) / sum(self.p_rz(r, z, a) for r in range(2))
                for z in range(2)]

    def target_joint(self, r, y, word):
        amp, *signs = y
        p_y = probability(amp == r, self.b)
        for t, bit in zip(signs, word):
            p_y *= probability(t == (bit ^ r), self.s)
        return p_y / len(self.code)


def train_pairs(model, n, seed):
    """Only return observed Z,V. The evaluator and latent branch are inaccessible to fit."""
    rng = random.Random(seed)
    observed = []
    for _ in range(n):
        r = int(rng.random() >= float(model.p))
        z = r if rng.random() < float(model.a) else 1 - r
        v = r if rng.random() < float(model.v) else 1 - r
        observed.append((z, v))
    return observed


def softmax_weights(z, key0):
    query = [int(z == 0), int(z == 1)]
    scores = [identity_score(query, key0), 0.0]
    shift = max(scores)
    exps = [math.exp(x - shift) for x in scores]
    return [v / sum(exps) for v in exps]


def fit_attention(observed, probe_reliability, clip):
    # For each z the Bernoulli probe parameter is (1-v)+(2v-1)*w_z.
    # Its empirical MLE is the observed fraction projected to the feasible interval.
    counts = Counter(observed)
    fitted = []
    for z in range(2):
        total = counts[z, 0] + counts[z, 1]
        if total:
            prob_v0 = F(counts[z, 0], total)
            w = (prob_v0 - (1 - probe_reliability)) / (2 * probe_reliability - 1)
        else:
            w = F(1, 2)
        fitted.append(max(clip, min(1 - clip, w)))
    keys = [math.log(float(w / (1 - w))) for w in fitted]
    for z in range(2):
        assert abs(softmax_weights(z, keys)[0] - float(fitted[z])) < 1e-14
    record = {"observed_counts_z_then_v": [[counts[z, v] for v in range(2)] for z in range(2)],
              "fitted_key0": keys, "reference_key1": [0, 0]}
    return fitted, keys, record


def decisions(scores, code, candidates):
    block = max(candidates, key=lambda c: scores[c])
    bits = []
    for k in range(2):
        mass0 = sum(scores[c] for c in candidates if code[c][k] == 0)
        mass1 = sum(scores[c] for c in candidates if code[c][k] == 1)
        bits.append(int(mass1 > mass0))
    return tuple(bits), block


def population(model, weights, exact):
    """Fraction arithmetic preserves true posterior ties and integration over all atoms."""
    mse = gap = bound = ber = bler = block_regret = bit_regret = F(0)
    block_strict_mass = bit_strict_mass = restricted_bler = omit_mass = F(0)
    observed_mass = F(0)
    all_c = list(range(len(model.code)))
    restricted = [0, 1]
    for z in range(2):
        w = weights[z]
        h_pred = identity_score([w, 1 - w], model.h)
        oracle_mean = identity_score([exact[z], 1 - exact[z]], model.h)
        mse += sum(model.p_rz(r, z, model.a) * (model.h[r] - h_pred)**2 for r in range(2))
        gap += model.qz[z] * (h_pred - oracle_mean)**2
        bound += model.qz[z] * (abs(model.h[0] - model.h[1]) * abs(w - exact[z]))**2
        for y in model.y:
            branch_scores = [[model.target_joint(r, y, word) for word in model.code]
                             for r in range(2)]
            scores = [w * left + (1 - w) * right for left, right in zip(*branch_scores)]
            joint = [sum(model.p_rz(r, z, model.a) * branch_scores[r][c] for r in range(2))
                     for c in all_c]
            pzy = sum(joint)
            observed_mass += pzy
            bits, block = decisions(scores, model.code, all_c)
            bayes_bits, bayes_block = decisions(joint, model.code, all_c)
            restricted_block = decisions(scores, model.code, restricted)[1]
            restricted_bler += pzy - joint[restricted_block]
            omit_mass += sum(joint[c] for c in all_c if c not in restricted)
            bler += pzy - joint[block]
            block_regret += joint[bayes_block] - joint[block]
            block_strict_mass += pzy * (joint[bayes_block] > joint[block])
            for k in range(2):
                losses = [sum(joint[c] for c in all_c if model.code[c][k] != b)
                          for b in [bits[k], bayes_bits[k]]]
                ber += losses[0] / 2
                bit_regret += (losses[0] - losses[1]) / 2
                bit_strict_mass += pzy * (losses[0] > losses[1]) / 2
    assert observed_mass == 1 and gap == bound
    metrics = dict(mse=mse, ber=ber, bler=bler, projection_gap=gap,
                   weight_bound=bound, bit_map_regret=bit_regret, block_map_regret=block_regret,
                   mean_bitwise_strict_suboptimal_observation_mass=bit_strict_mass,
                   block_strict_suboptimal_observation_mass=block_strict_mass,
                   first_two_only_bler=restricted_bler, first_two_only_omission_probability=omit_mass)
    return {k: number(v) for k, v in metrics.items()} | {"weight0_by_z": [number(v) for v in weights]}


def analytic_checks():
    # Noiseless counterexample. Integrate over the full prior independently for each action.
    p_branch = [F(3, 5), F(2, 5)]
    gains = [1, -2]
    gain_mean = identity_score(p_branch, gains)
    methods = {"exact_mean": gain_mean, "map_branch": 1,
               "reversed_weights": identity_score(p_branch[::-1], gains)}
    noiseless = {}
    for label, estimate in methods.items():
        mse = sum(p * (h - estimate)**2 for p, h in zip(p_branch, gains))
        ber = sum(p / 2 for p, h in zip(p_branch, gains) for x in [1, -1]
                  if (1 if estimate * h * x > 0 else -1) != x)
        noiseless[label] = {"estimate": number(estimate), "mse": number(mse), "plug_in_ber": number(ber)}
    full_ber = sum(p / 2 for p, h in zip(p_branch, gains) for x in [1, -1]
                   if h * x / (1 if abs(h * x) == 1 else -2) != x)
    assert full_ber == 0
    noiseless["joint_map_ber"] = number(full_ber)
    assert F(noiseless["exact_mean"]["mse"]["fraction"]) == F(54, 25)
    assert F(noiseless["map_branch"]["mse"]["fraction"]) == F(18, 5)
    assert F(noiseless["exact_mean"]["plug_in_ber"]["fraction"]) == F(3, 5)
    assert F(noiseless["map_branch"]["plug_in_ber"]["fraction"]) == F(2, 5)
    equal = {}
    # For the three-way example the central branch is uniquely MAP.
    for label, weights, values, selected in [
        ("same_means", [F(3, 5), F(2, 5)], [4, 4], 0),
        ("three_way_selected_barycenter", [F(1, 4), F(1, 2), F(1, 4)], [-2, 0, 2], 1)
    ]:
        risk = lambda action: sum(w * (v - action)**2 for w, v in zip(weights, values))
        soft = risk(identity_score(weights, values))
        hard = risk(values[selected])
        assert soft == hard
        equal[label] = {"weights": list(map(str, weights)), "values": values, "selected": selected,
                        "soft_mse": number(soft), "hard_mse": number(hard)}
    # More extreme wrong weights can be worse than hard in the same noiseless example.
    wrong = F(1, 10)
    wrong_mean = identity_score([wrong, 1 - wrong], gains)
    wrong_mse = sum(a * (h - wrong_mean)**2 for a, h in zip(p_branch, gains))
    assert wrong_mse > F(18, 5)
    value_errors = [F(1, 5), F(1, 5)]
    perturbed = [h - e for h, e in zip(gains, value_errors)]
    approx = identity_score([wrong, 1 - wrong], perturbed)
    bound = 3 * abs(wrong - p_branch[0]) + identity_score([wrong, 1 - wrong], value_errors)
    assert abs(approx - gain_mean) == bound
    # A real Gaussian N(mean,var) preserving both true moments instead yields
    # log p(y|X=+1)/p(y|X=-1) = 2*mean*y/var, which discards the amplitude cue.
    variance = sum(p * (h - gain_mean)**2 for p, h in zip(p_branch, gains))
    gaussian_ber = sum(p / 2 for p, h in zip(p_branch, gains) for x in [1, -1]
                       if (1 if 2 * gain_mean * h * x / variance > 0 else -1) != x)
    assert gaussian_ber == F(3, 5)
    noiseless["moment_matched_gaussian_map_ber"] = number(gaussian_ber)
    return {"given_noiseless": noiseless, "equality": equal,
            "wrong_weights_worse_than_hard": {"weights": ["1/10", "9/10"],
                        "estimated_gain": number(wrong_mean), "mse": number(wrong_mse)},
            "diameter_bound_with_value_errors": {"perturbed_values": list(map(str, perturbed)),
                             "estimate_error": number(abs(approx - gain_mean)), "bound": number(bound)}}


def complex_cov_check():
    # Two-dimensional complex column h. Split each branch into two equiprobable atoms.
    alpha = [0.625, 0.375]
    mu_r = [[1 + 1j, 0.5j], [-2 + 0.5j, 1 - 1j]]
    shift = [[0.25 + 0.5j, -0.5j], [1 - 0.25j, -0.5 + 0.25j]]
    mu = [identity_score(alpha, [m[j] for m in mu_r]) for j in range(2)]
    direct = [[0j] * 2 for _ in range(2)]
    within = [[0j] * 2 for _ in range(2)]
    between = [[0j] * 2 for _ in range(2)]
    atoms = []
    for r in range(2):
        for sign in [-1, 1]:
            h = [m + sign * d for m, d in zip(mu_r[r], shift[r])]
            atoms.append((alpha[r] / 2, h))
            for j, k in itertools.product(range(2), repeat=2):
                direct[j][k] += alpha[r] / 2 * (h[j] - mu[j]) * (h[k] - mu[k]).conjugate()
        for j, k in itertools.product(range(2), repeat=2):
            within[j][k] += alpha[r] * shift[r][j] * shift[r][k].conjugate()
            between[j][k] += alpha[r] * (mu_r[r][j] - mu[j]) * (mu_r[r][k] - mu[k]).conjugate()
    cov_err = max(abs(direct[j][k] - within[j][k] - between[j][k])
                  for j, k in itertools.product(range(2), repeat=2))
    hard = mu_r[0]
    risk = lambda est: sum(p * sum(abs(x - a)**2 for x, a in zip(h, est)) for p, h in atoms)
    gap = sum(abs(a - b)**2 for a, b in zip(hard, mu))
    proj_err = abs(risk(hard) - risk(mu) - gap)
    assert cov_err < 1e-14 and proj_err < 1e-14
    matrix = lambda m: [[[z.real, z.imag] for z in row] for row in m]
    return {"direct_complex_covariance_re_im": matrix(direct), "within": matrix(within),
            "between": matrix(between), "max_abs_decomposition_error": cov_err,
            "hard_minus_soft_mse": gap, "max_abs_projection_error": proj_err}


def infer(model, y, z, weight0, hard):
    # Dense and common word + target likelihood, specialized to skip discarded hard branch.
    w = weight0(z)
    active = [(int(w != 1), 1.0)] if hard else [(0, w), (1, 1 - w)]
    scores = [0.0] * len(model.code)
    for r, mass in active:
        for c, word in enumerate(model.code):
            likelihood = float(model.b if y[0] == r else 1 - model.b)
            for t, bit in zip(y[1:], word):
                likelihood *= model.s_float if t == (bit ^ r) else 1.0 - model.s_float
            scores[c] += mass * likelihood / len(model.code)
    return decisions(scores, model.code, range(len(scores)))


def run_timing(model, entries):
    # Source fractions converted to float before timing. This separate run is descriptive.
    model.s_float = float(model.s)
    observations = list(itertools.product(range(2), model.y))
    iters, reps = 1200, 5
    timings = {}
    for label, evaluator, hard in entries:
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            for i in range(iters):
                z, y = observations[i % len(observations)]
                infer(model, y, z, evaluator, hard)
            samples.append((time.perf_counter() - start) * 1e6 / iters)
        timings[label] = {"all_us_per_decision": samples, "median_us_per_decision": statistics.median(samples)}
    return {"scope": "single-threaded CPython illustrative, no claim about P12; exact/physical prior weights and hard MAP use a two-entry cached Z table; attention computes dot-product softmax live; joint symbol likelihoods computed online",
            "python": sys.version, "machine": platform.machine(), "platform": platform.platform(),
            "cpu": next((line.split(":", 1)[1].strip() for line in
                         Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "")
                    if Path("/proc/cpuinfo").exists() else "",
            "iterations_per_pass": iters, "reps": reps, "measurements": timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timing", type=Path, help="optional environment-specific benchmark output")
    args = parser.parse_args()
    raw = (HERE / "protocol.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == (HERE / "protocol.sha256").read_text().split()[0]
    config = json.loads(raw)
    model = Model(config)
    settings = config["methods"]
    exact = model.weights0(model.a)
    hard = [F(w >= F(1, 2)) for w in exact]
    physical = model.weights0(F(settings["physical_assumed_p_context_sign_correct"]))
    methods = {"exact_soft": exact, "context_map_hard": hard, "physical_soft": physical}
    time_entries = [(name, [float(v) for v in weights].__getitem__, name == "context_map_hard")
                    for name, weights in methods.items()]
    training = {}
    for seed in settings["training_seeds"]:
        observed = train_pairs(model, settings["n_training_frames_per_seed"], seed)
        fitted, keys, record = fit_attention(observed, model.v, F(settings["fit_weight_clip"]))
        name = f"attention_seed_{seed}"
        training[name] = record
        methods[name] = fitted
        time_entries.append((name, lambda z, k=keys: softmax_weights(z, k)[0], False))
    results = {name: population(model, w, exact) for name, w in methods.items()}
    ref = results["exact_soft"]
    for metrics in results.values():
        for key, regret in [("mse", "projection_gap"), ("ber", "bit_map_regret"), ("bler", "block_map_regret")]:
            get = lambda d, k: F(d[k]["fraction"])
            assert get(metrics, key) - get(ref, key) == get(metrics, regret)
    assert F(results["context_map_hard"]["projection_gap"]["fraction"]) > 0
    nwords = len(model.code)
    computations = {"latent_joint_atoms": 2 * 2 * nwords * len(model.y),
            "distinct_context_target_pairs": 2 * len(model.y),
            "legal_words_per_decision": nwords,
            "soft_conditional_word_likelihoods_per_decision": 2 * nwords,
            "hard_conditional_word_likelihoods_per_decision": nwords,
            "each_conditional_likelihood": "one amplitude probability and three sign probabilities",
            "attention_trainable_scalars": 2, "fit_work_per_seed": "one pass over 1500 calibration pairs, four counters, two clipped ratios and two logits, no optimization iterations",
            "hard_context_map_cost": "both context branch hypotheses for initial MAP; cacheable",
            "attention_inference": "two dot-product scores (one fixed zero key), two-way softmax, two weighted branch values for MSE; Z tables cacheable for all methods",
            "model_storage": "two channel values; four common codewords; two cached context weights per method if desired"}
    result = {"protocol_sha256": digest, "metrics": results, "fitted_models": training,
              "finite_costs": computations, "analytical_tests": analytic_checks(),
              "complex_projection_covariance": complex_cov_check()}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.timing:
        args.timing.write_text(json.dumps(run_timing(model, time_entries), indent=2) + "\n")


if __name__ == "__main__":
    main()
