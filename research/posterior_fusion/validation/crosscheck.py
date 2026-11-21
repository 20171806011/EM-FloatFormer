"""Independent population check by enumerating physical noise flips, using only the standard library."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

root = Path(__file__).resolve().parent
protocol = json.loads((root / "protocol.json").read_text())
reference = json.loads((root / "results.json").read_text())
law = protocol["true"]
words = [tuple(map(int, s)) for s in law["codewords_for_00_01_10_11"]]
prior = (F(law["p_branch_0"]), 1 - F(law["p_branch_0"]))
reliabilities = [F(law["p_context_sign_correct"]), F(law["p_target_amplitude_label_correct"])]
reliabilities += [F(law["p_target_bpsk_sign_correct"])] * 3
gains = law["H_order"]
joint, context = defaultdict(F), defaultdict(F)
for (r, pr), word_id, flips in product(enumerate(prior), range(4), product((0, 1), repeat=5)):
    mass = pr / 4
    for flip, reliable in zip(flips, reliabilities):
        mass *= (1 - reliable) if flip else reliable
    z = r ^ flips[0]
    y = (r ^ flips[1], *(b ^ r ^ e for b, e in zip(words[word_id], flips[2:])))
    joint[z, y, r, word_id] += mass
    context[z, r] += mass
assert sum(joint.values()) == 1 and len(joint) == 256
ys = sorted({key[1] for key in joint})
exact = [context[z, 0] / (context[z, 0] + context[z, 1]) for z in (0, 1)]
reliable_q = F(protocol["methods"]["physical_assumed_p_context_sign_correct"])
physical = []
for z in (0, 1):
    a, b = (reliable_q, 1-reliable_q) if z == 0 else (1-reliable_q, reliable_q)
    physical.append(a * prior[0] / (a * prior[0] + b * prior[1]))
policies = {"exact_soft": exact, "context_map_hard": [F(x >= F(1, 2)) for x in exact],
            "physical_soft": physical}
probe = F(law["p_training_probe_sign_correct"])
clip = F(protocol["methods"]["fit_weight_clip"])
for name, record in reference["fitted_models"].items():
    weights = []
    for n0, n1 in record["observed_counts_z_then_v"]:
        w = (F(n0, n0 + n1) - (1 - probe)) / (2 * probe - 1) if n0 + n1 else F(1, 2)
        weights.append(min(1 - clip, max(clip, w)))
    policies[name] = weights

for name, policy in policies.items():
    mse, ber, bler = F(0), F(0), F(0)
    for z, w in enumerate(policy):
        mean = w * gains[0] + (1 - w) * gains[1]
        mse += sum(context[z, r] * (mean - h) ** 2 for r, h in enumerate(gains))
        for y in ys:
            scores = [sum(a * joint[z, y, r, i] / context[z, r] for r, a in enumerate((w, 1-w)))
                      for i in range(len(words))]
            best_word = max(range(len(words)), key=scores.__getitem__)
            bits = [int(sum(s for s, c in zip(scores, words) if c[j]) > sum(scores)/2)
                    for j in range(2)]
            for i, c in enumerate(words):
                mass = joint[z, y, 0, i] + joint[z, y, 1, i]
                ber += mass * sum(a != b for a, b in zip(bits, c[:2])) / 2
                bler += mass * (best_word != i)
    for metric, got in (("mse", mse), ("ber", ber), ("bler", bler)):
        assert got == F(reference["metrics"][name][metric]["fraction"]), (name, metric, got)
    print(f"{name:26s} MSE {float(mse):.9f}  BER {float(ber):.6f}  BLER {float(bler):.6f}")
