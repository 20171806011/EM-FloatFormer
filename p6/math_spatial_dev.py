"""One R3 development IRC candidate against Iteration 5's original receiver.

Each parity arm uses the same three initializations, 48-update scheduled Adam,
and B32 validation frames. Evaluate original weights with IRC toggled and
refit both arms from scratch with IRC. Keep IRC only with fewer Axial errors
than the 49-word baseline, positive within-checkpoint spatial ablation gain,
joint Green >= .34, strict anchoring, and the common final invariant gate.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import time
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex, iter5_dev, iter4_spatial, training_iter3
from p6 import math_spatial, run_dev_iter2 as prior
from p6.data import PAIRS, dev_frame, dev_seed
from p6.diagnostics import anchor_check


OUT = ex.RESULTS / "dev_math" / "spatial"
STEPS = 48
SOURCE = {f: prior.digest(ex.ROOT / f) for f in (
    "math_spatial.py", "math_spatial_dev.py", "axial_em_transformer.py", "axial_operator.py",
    "data.py", "training_iter3.py", "experiment.py", "iter5_dev.py", "parity.py")}


def objective_sensitivities(model, pair):
    batch = dev_frame(3, 5, pair, ex.FIT_BATCH)
    read_handles = solver._physical_handles
    spatial_apply = solver._apply_spatial
    observed = {"delays": [], "spatial_strength": [], "transport": []}

    def read(*args, **kwargs):
        out = read_handles(*args, **kwargs)
        observed["delays"].append(out.delays)
        observed["spatial_strength"].append(out.spatial)
        return out

    def apply(amp, handle):
        operator = handle[0]
        if not observed["transport"] or observed["transport"][-1] is not operator:
            observed["transport"].append(operator)
        return spatial_apply(amp, handle)

    with patch.object(solver, "_physical_handles", read), patch.object(solver, "_apply_spatial", apply):
        loss = training_iter3.objective(batch, model, aux_nmse=.005)[0]
    tensors = [v for stages in observed.values() for v in stages]
    gradients = iter(torch.autograd.grad(loss, tensors))
    stats = {}
    for name, values in observed.items():
        rows = []
        for value in values:
            grad = next(gradients)
            rows.append(dict(rms_value=float(value.detach().abs().square().mean().sqrt()),
                             rms_gradient=float(grad.abs().square().mean().sqrt()),
                             directional_derivative_for_scaling=float((grad.conj() * value.detach()).real.sum())))
        stats[name] = rows
    return stats


@torch.inference_mode()
def posterior_stats(model, frame):
    equalize = solver.innovation_equalize
    rows = []
    def observed(y, h, nv, pilot_mask, pilots, transport, strength):
        correction, gate = math_spatial.precision_correction(y, h, nv, pilot_mask, pilots, transport, strength)
        eigen = 1 + torch.linalg.eigvalsh(correction)
        rows.append(dict(min_relative_cov_eigen=float(eigen.min()), max_relative_cov_eigen=float(eigen.max()),
                         median_gate=float(gate.median()), mean_gate=float(gate.mean())))
        return equalize(y, h, nv, pilot_mask, pilots, transport, strength)
    with patch.object(solver, "innovation_equalize", observed):
        model(frame.obs)
    return rows


def run(key):
    pair, arm = key
    model = iter5_dev.model_for(3, pair, arm, .05, False)
    model.spatial_irc = True
    validation = dev_frame(3, 6, pair, 32)
    original_hash = iter5_dev.study_dev.initial_weights(model)
    initial_probe = prior.probe(model, dev_frame(3, 8, pair, 8), dev_seed(3, 9, pair)) if arm == "parity" else None
    with torch.no_grad():
        anchor = anchor_check(model, validation.obs, arm, 3)
    if not anchor["p6_bit_exact"] or max(anchor["archived_max_llr_cfr"]) != 0:
        raise ArithmeticError(f"R3 IRC breaks anchor: {anchor}")
    historical = iter5_dev.location(3, pair, arm, STEPS, .05, False)
    ref = ex.read(historical)
    if prior.digest(historical.with_suffix(".pt")) != ref["checkpoint_sha256"]:
        raise ValueError("Stale reference weights")
    archived = iter5_dev.model_for(3, pair, arm, .05, False)
    archived.load_state_dict(torch.load(historical.with_suffix(".pt"), weights_only=True)["state"])
    baseline = ex.score([validation], lambda f: archived(f.obs))
    if baseline["block_errors"] != ref["score"]["block_errors"]:
        raise ArithmeticError("Historical baseline block errors changed")
    archived.spatial_irc = True
    retrospective = ex.score([validation], lambda f: archived(f.obs))
    start = time.perf_counter()
    iter5_dev.optimize(model, STEPS, lambda i: dev_frame(3, 5, pair, ex.FIT_BATCH, i))
    seconds = time.perf_counter() - start
    full, details = iter5_dev.diagnose(model.eval(), 3, pair, validation, initial_probe)
    with patch.object(model, "spatial_irc", False):
        irc_off = ex.score([validation], lambda f: model(f.obs))
    cov = posterior_stats(model, validation)
    grad = objective_sensitivities(model, pair)
    dest = OUT / str(pair) / f"{arm}.json"
    source_id = hashlib.sha256(repr(sorted(SOURCE.items())).encode()).hexdigest()
    ex.save_weights(dest.with_suffix(".pt"), model, source_id)
    rec = dict(pair=pair, arm=arm, regime=3, steps=STEPS, seed=dev_seed(3, 6, pair),
               init=original_hash, anchor=anchor, parameters=ex.real_parameters(model),
               source_sha256=SOURCE, checkpoint_sha256=prior.digest(dest.with_suffix(".pt")),
               archived_baseline=baseline, archived_with_irc=retrospective, trained_full=full,
               trained_irc_off=irc_off, fit_seconds=seconds, cov_stats=cov,
               train_batch0_gradient=grad, **details)
    ex.save(dest, rec)
    print(pair, arm, [r["block_errors"] for r in (baseline,retrospective,full,irc_off,details["spatial_off_score"])],
          flush=True)


def summarize():
    groups = {}
    for arm in ex.PARITY_ARMS:
        rows = [ex.read(OUT / str(pair) / f"{arm}.json") for pair in PAIRS]
        group = {k: [r[k]["block_errors"] for r in rows] for k in
                 ("archived_baseline", "archived_with_irc", "trained_full", "trained_irc_off", "spatial_off_score")}
        group["pooled"] = {k: sum(v) for k,v in group.items()}
        group["relative_covariance_eigenvalues"] = [
            min(s["min_relative_cov_eigen"] for r in rows for s in r["cov_stats"]),
            max(s["max_relative_cov_eigen"] for r in rows for s in r["cov_stats"])]
        group["mean_gate"] = sum(s["mean_gate"] for r in rows for s in r["cov_stats"]) / 9
        group["steering"] = iter4_spatial.pool([r["steering"] for r in rows])["waveform_solve_3"]["steering_cosine_after"]
        group["steering_off"] = iter4_spatial.pool([r["steering_off"] for r in rows])["waveform_solve_3"]["steering_cosine_after"]
        if rows[0].get("probe"):
            group["green"] = prior.probe_summary(rows)
        groups[arm] = group
    ex.save(OUT / "summary.json", groups)
    print({a: g["pooled"] for a,g in groups.items()}, flush=True)


if __name__ == "__main__":
    keys = [(p,a) for p in PAIRS for a in ex.PARITY_ARMS]
    with ProcessPoolExecutor(max_workers=2, initializer=ex.setup) as pool:
        list(pool.map(run, keys))
    summarize()
