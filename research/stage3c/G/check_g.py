"""Exhaustive wrapper/averaging checks on an explicitly named reference law."""
import argparse
import json
import math
from pathlib import Path

from quotient import canonical_input, one_call, rotate


def shift_code(code, turns):
    return (code + 4 * turns) % 16


def brier(p, q):
    return 1 + sum(b*b - 2*a*b for a, b in zip(p, q))


def cross_entropy(p, q):
    return -sum(a*math.log(b) for a, b in zip(p, q))


def check(table):
    def canonical_teacher(i, j, task, _):
        key = "q1" if task == "q1" else "q3"
        return table["cells"][4*i+j][key]

    def biased(z1, z2, task):
        probs = one_call(canonical_teacher, z1, z2, task, None)
        return [0.8*v + (0.15 if k == 0 else 0.05/3) for k, v in enumerate(probs)]

    def averaged(z1, z2, task):
        aligned = []
        for a in range(4):
            for b in range(4):
                shift = b-a if task == "q1" else a
                transformed = biased(shift_code(z1, a), shift_code(z2, b), task)
                aligned.append(rotate(transformed, -shift))
        return [sum(row[k] for row in aligned)/16 for k in range(4)]

    totals = [0.] * 5
    cases = 0
    for task in ("q1", "q3_repeat"):
        for z1 in range(16):
            for z2 in range(16):
                p = one_call(canonical_teacher, z1, z2, task, None)
                q, qa = biased(z1, z2, task), averaged(z1, z2, task)
                quantities = [brier(p,q), brier(p,qa), sum((x-y)**2 for x,y in zip(q,qa)),
                              cross_entropy(p,q), cross_entropy(p,qa)]
                totals = [x+y for x,y in zip(totals, quantities)]
                cases += 1
                for a in range(4):
                    for b in range(4):
                        vals = one_call(canonical_teacher, shift_code(z1,a), shift_code(z2,b), task, None)
                        assert vals == rotate(p, b-a if task == "q1" else a)

    # Check Q2 label shifts with a fixed reference and a deliberately arbitrary reader.
    pilot = object()
    def q2_reader(i, j, task, context):
        assert context is pilot and task == "q2"
        return rotate([.4,.3,.2,.1], i+j)
    for z1 in range(16):
        for z2 in range(16):
            p = one_call(q2_reader,z1,z2,"q2",pilot)
            for a in range(4):
                for b in range(4):
                    assert one_call(q2_reader,shift_code(z1,a),shift_code(z2,b),"q2",pilot) == rotate(p,a)
                    _, _, shift = canonical_input(z1,z2,"q2")
                    assert (z1//4 - shift) % 4 == 0

    old_b, new_b, variance, old_l, new_l = [x/cases for x in totals]
    assert abs(old_b-new_b-variance) < 1e-12 and old_l > new_l
    # All points share one canonical truth=0. Fifteen weak correct decisions
    # and one confidently wrong decision. Their probability average favors 1.
    mean_class0 = (15*.501+.001)/16
    assert mean_class0 < 0.5
    return {"input_reference": "Uniform over 256 cache states; labels drawn from fixed Q1/Q3 reference table",
            "quarter_turn_identities_checked": (cases+256)*16,
            "toy_added_label_bias": {"cross_entropy_before":old_l, "cross_entropy_averaged":new_l,
                                     "brier_improvement":old_b-new_b, "squared_projection_distance":variance},
            "top1_counterexample": {"unprojected_accuracy":15/16,"projected_accuracy":0.0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.table.read_text())), indent=2))
