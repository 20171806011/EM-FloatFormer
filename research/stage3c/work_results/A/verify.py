"""Independent Q2 contraction and CPU audit of the frozen C4 kernel."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp


ROOTS = np.array([1, 1j, -1, -1j])
LABELS = np.arange(4)
CACHE_MAP = (LABELS[:, None] + LABELS[None, :]) % 4
COUNT = 262144


def posterior(table, z1, weights, phase, rho, pilots, separate=False):
    """Use a shared phase across acquisitions, unless running the negative control."""
    means = (np.exp(1j * phase[:, None]) * ROOTS).conj()
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)[:, :, None]
    logs = [2 * rho * (pilot[:, None, None] * means).real for pilot in pilots]
    if separate:
        log_carrier = sum(logsumexp(log_weights + term, axis=1) for term in logs)
    else:
        log_carrier = logsumexp(log_weights + sum(logs), axis=1)
    carrier = np.exp(log_carrier - logsumexp(log_carrier, axis=1, keepdims=True))
    return np.einsum("ng,nkg->nk", carrier, table[z1][:, CACHE_MAP])


def score(prob, truth):
    return np.column_stack((prob.argmax(axis=1) == truth,
                            -np.log(prob[np.arange(len(truth)), truth])))


def paired(after, before):
    delta = after - before
    se = delta.std(axis=0, ddof=1) / math.sqrt(len(delta))
    return dict(acc=float(after[:, 0].mean()), nll=float(after[:, 1].mean()),
                delta_acc=float(delta[:, 0].mean()), paired_acc_se=float(se[0]),
                delta_nll=float(delta[:, 1].mean()), paired_nll_se=float(se[1]),
                corrected=int((delta[:, 0] > 0).sum()),
                new_errors=int((delta[:, 0] < 0).sum()))


def replay(encoder, rho):
    """Use the same draws and 2048-row batching as the frozen Q2 experiment."""
    rng = np.random.default_rng(920617)
    scale = (2 * rho) ** -.5
    for _ in range(COUNT // 2048):
        u, v, g = rng.integers(4, size=(3, 2048))
        phase2, increment = rng.normal(.7, .1 * math.sqrt(2), size=(2, 2048))

        def window(labels, phase):
            noise = rng.normal(size=(2048, 3)) + 1j * rng.normal(size=(2048, 3))
            mean = ROOTS[labels] * np.exp(1j * phase) / math.sqrt(3)
            return mean[:, None] + scale * noise

        yield ((u - g) % 4, encoder.encode(window(u, 0)),
               encoder.encode(window(v, phase2)), window(g, phase2 + increment),
               window(g, phase2 + increment))


def read_reference(kernel, z1, z2, records):
    """Read (acquisition_id, same_channel, time, matched_sum) records at frozen SNR."""
    from research.stage3c.A.q2_reader import read

    if any(valid and time != kernel["pilot_time"] for _, valid, time, _ in records):
        raise ValueError("Valid acquisitions must use the frozen pilot timestamp")
    return read(kernel, z1, z2, [(key, valid, value) for key, valid, _, value in records])


def audit(repo):
    import learned
    from research.stage3c.A.q2_reader import read
    from research.stage3_ministral_cache.data import SPLIT_SEEDS, generate_mothers
    from research.stage3_ministral_cache.schema import CONDITION_SPECS, matched_sum

    frozen = repo / "research/stage3c/A"
    kernel = json.loads((frozen / "q2_s17_kernel.json").read_text())
    baseline = json.loads((frozen / "q2_results.json").read_text())
    checkpoint = repo / "research/matched_cache/weights/equivariant_b8_s17.npz"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == kernel["checkpoint_sha256"]
    encoder = learned.load(checkpoint)
    table = np.array(kernel["u_given_z1"])
    phase = np.array(kernel["pilot_phase_nodes"])
    weights = np.array(kernel["weights_given_z2"])
    rho = kernel["rho"]
    prior = np.polynomial.hermite.hermgauss(len(phase))[1] / math.sqrt(math.pi)

    arms = {}
    for truth, z1, z2, first, second in replay(encoder, rho):
        for arm, pilots in (("valid", [matched_sum(first)]),
                            ("fresh", [matched_sum(first), matched_sum(second)])):
            for name, a, w in (("joint", table, weights[z2]),
                               ("legacy", encoder.table, np.broadcast_to(prior, weights[z2].shape)),
                               ("calibrate_z1", table, np.broadcast_to(prior, weights[z2].shape)),
                               ("condition_z2", encoder.table, weights[z2])):
                prob = posterior(a, z1, w, phase, rho, pilots)
                arms.setdefault((arm, name), []).append(score(prob, truth))
    arms = {key: np.concatenate(parts) for key, parts in arms.items()}
    measured = {}
    max_delta = 0.
    for arm in ("valid", "fresh"):
        measured[arm] = {}
        for name, published in baseline["results"][arm].items():
            base_name = "calibrate_z1" if name == "joint_vs_calibrated" else "legacy"
            name_for_score = "joint" if name == "joint_vs_calibrated" else name
            summary = paired(arms[arm, name_for_score], arms[arm, base_name])
            measured[arm][name] = summary
            max_delta = max(max_delta, *(abs(summary[key] - published[key]) for key in published))
    assert max_delta < 1e-12
    gain = paired(arms["fresh", "joint"], arms["valid", "joint"])
    for key in ("delta_acc", "corrected", "new_errors"):
        assert gain[key] == baseline["fresh_minus_valid"][key]
    assert abs(gain["paired_acc_se"] - baseline["fresh_minus_valid"]["paired_se"]) < 1e-12

    # Check the protocol using the separate, balanced frozen host simulator.
    host = generate_mothers("s3_test", kernel["condition"])
    z1 = encoder.encode(host.y1)
    z2 = encoder.encode(host.y2)
    host_prob = {}
    for arm in ("valid", "fresh_resample", "duplicate_copy", "independent", "none"):
        records = [(r.acquisition_id, r.shares_channel, r.t_p, matched_sum(r.samples))
                   for r in host.references_for_arm(arm)]
        host_prob[arm] = np.array([
            read_reference(kernel, int(i), int(j),
                           [(key, valid, time, value[row])
                            for key, valid, time, value in records])
            for row, (i, j) in enumerate(zip(z1, z2))
        ])
    np.testing.assert_array_equal(host_prob["duplicate_copy"], host_prob["valid"])
    np.testing.assert_array_equal(host_prob["independent"], host_prob["none"])
    np.testing.assert_array_equal(host_prob["none"], np.full((len(z1), 4), .25))

    # Direct enumeration over U, G and the quadrature nodes, including all cache pairs.
    i, j = np.indices((16, 16)).reshape(2, -1)
    t1, t2 = .38 - .76j, 1.21 + .44j
    pilots = [np.full(len(i), t) for t in (t1, t2)]
    max_enum = 0.
    max_reader = 0.
    for n in (1, 2):
        joint = posterior(table, i, weights[j], phase, rho, pilots[:n])
        for row, (a, b) in enumerate(zip(i, j)):
            brute = np.zeros(4)
            for u in LABELS:
                for g in LABELS:
                    for s, w in zip(phase, weights[b]):
                        mean = ROOTS[g] * np.exp(1j * s)
                        likelihood = math.prod(math.exp(-rho * abs(t[row] - mean)**2)
                                               for t in pilots[:n])
                        brute[(u - g) % 4] += table[a, u] * w * likelihood
            max_enum = max(max_enum, float(np.max(np.abs(joint[row] - brute / brute.sum()))))
            refs = [(k, True, t[row]) for k, t in enumerate(pilots[:n])]
            max_reader = max(max_reader, float(np.max(np.abs(joint[row] - read(kernel, int(a), int(b), refs)))))
    assert max(max_enum, max_reader) < 2e-14
    wrong = posterior(table, i, weights[j], phase, rho, pilots, separate=True)
    example = 3 * 16 + 12
    return dict(mothers=COUNT, replay_seed=920617, calibration_seed=kernel["calibration_seed"],
                calibration_windows=kernel["calibration_count"], numpy=np.__version__,
                frozen_results_max_abs_difference=max_delta, results=measured,
                fresh_minus_valid=gain, host_mothers=len(z1),
                host_seed=SPLIT_SEEDS["s3_test"][0] * 100 + sorted(CONDITION_SPECS).index(kernel["condition"]),
                host_duplicate_max_difference=0., host_unrelated_max_difference=0.,
                enumerated_posteriors=512, max_enumeration_difference=max_enum,
                max_public_reader_difference=max_reader,
                example=dict(z1=3, z2=12, pilot_sum_real_imag=[[t.real, t.imag] for t in (t1, t2)],
                             shared=joint[example].tolist(), separate=wrong[example].tolist(),
                             total_variation=float(abs(joint[example] - wrong[example]).sum() / 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "research/matched_cache")]
    print(json.dumps(audit(repo), indent=2))
