"""Check three-step development checkpoints at extreme complex gains."""

from dataclasses import replace
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from p5b.diagnostics import SCALES, erased, gain_check
from p6 import axial_em_transformer as solver
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.diagnostics import degenerate
from p6.experiment import save, setup
from p6.run_dev_iter2 import implementation, restored


ROOT = Path(__file__).resolve().parent / "results"
GAUGED_ARMS = ("pure", "parity", "local", "local_parity")
EXTREMES = (1e-35, 1., 1e20)


@torch.inference_mode()
def path_diagnostic(model, regime, pair):
    """Compare inferred ray updates to validation truth only after inference."""
    f = dev_frame(regime, 6, pair, 32)
    first_rot = []
    handles = []
    init = solver._initialize
    read = solver._physical_handles

    def trace_init(*args):
        result = init(*args)
        first_rot.append(result[1])
        return result

    def trace_handles(*args):
        result = read(*args)
        handles.append(result)
        return result

    with patch.object(solver, "_initialize", trace_init), patch.object(solver, "_physical_handles", trace_handles):
        model(f.obs)
    if regime == 3:
        nominal = f.truth.delays.new_tensor(solver.DELAYS[3])
        offsets = (f.truth.delays - nominal)[:, 1:]
        errors = [(h.delays[:, 1:] - offsets).abs().mean().item() for h in handles]
        return dict(units="Nyquist samples; mean over movable rays", nominal_mae=offsets.abs().mean().item(),
                    fitted_mae_by_step=errors)
    target = f.truth.ray_evolution
    def row_phase_error(rot):
        product = rot.mean(2) * target.conj()
        unit = product / product.abs().clamp_min(torch.finfo(torch.float64).tiny)
        aligned = unit * unit.mean(1, keepdim=True).conj()
        return aligned.angle().abs().mean().item()
    rotation = first_rot[0]
    total = torch.zeros_like(handles[0].phase)
    errors = [row_phase_error(rotation)]
    stats = []
    for h in handles:
        total = total + h.phase
        rotation = rotation * torch.exp(1j * h.phase)
        errors.append(row_phase_error(rotation))
        absolute = total.abs().flatten()
        stats.append(dict(max_increment_rad=h.phase.abs().max().item(),
                          max_accumulated_rad=absolute.max().item(),
                          p95_accumulated_rad=absolute.quantile(.95).item(),
                          fraction_over_half_rad=(absolute > .5).double().mean().item()))
    return dict(units="radians", trajectory="per-path sample rotation; excludes common phase",
                mean_absolute_wrapped_row_error_initial_then_steps=errors,
                phase_reference="one freely fitted constant phase per burst and ray",
                steps=stats)


def check_pair(regime, pair):
    obs = dev_frame(regime, 6, pair, 2).obs
    zero_nv = replace(obs, noise_var=torch.zeros_like(obs.noise_var))
    cases = {name: erased(obs, name) for name in
             ("intact", "first_pilot_row", "last_pilot_row", "all_pilots", "whole_field")}
    cases.update(degenerate_pilots=degenerate(obs), zero_reported_noise=zero_nv,
                 noiseless=dev_frame(regime, 6, pair, 2, snr=math.inf).obs,
                 all_pilots_zero_noise=erased(zero_nv, "all_pilots"),
                 whole_field_zero_noise=erased(zero_nv, "whole_field"))
    paths, rows = {}, []
    for arm in GAUGED_ARMS:
        model = restored(regime, pair, 3, arm)
        if regime > 1 and arm in ("parity", "local_parity"):
            paths[arm] = path_diagnostic(model, regime, pair)
        for index, (name, changed) in enumerate(cases.items()):
            result = gain_check(model, changed, SCALES if name == "intact" else EXTREMES,
                                dev_seed(regime, 6, pair, 945 + index))
            if name.startswith("whole_field"):
                with torch.inference_mode():
                    outputs = model(changed)
                result["exact_zero"] = all(bool(torch.isfinite(a).all()) and a.count_nonzero() == 0
                                           for a in outputs)
                result["exact_zero"] = bool(result["exact_zero"])
                result["passed"] &= result["exact_zero"]
            rows.append(dict(arm=arm, condition=name, **result))
    passed = all(row["passed"] for row in rows)
    save(ROOT / "iter2_checks" / f"trained_r{regime}_{pair}.json",
         dict(regime=regime, pair=pair, depth=3, stream=6, batch_size=2,
              implementation=implementation(), rows=rows, path_diagnostic=paths, passed=passed))
    print(f"trained R{regime}/{pair} {sum(x['passed'] for x in rows)}/{len(rows)} "
          f"max abs delta LLR={max(x['max_grid_llr_diff'] for x in rows):.3g}", flush=True)
    return passed


def summarize_checks():
    root = ROOT / "iter2_checks"
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    pre = read("invariants.json")
    suite = read("p6_suite.json")
    legacy = read("legacy.json")
    p4_text = (root / "p4_verifier.txt").read_text(encoding="utf-8")
    assert pre["passed"] and suite["passed"] and legacy["passed"] and "Exit code 0." in p4_text
    all_rows, sources, paths = [], [], []
    for r in SNR:
        for p in PAIRS:
            name = f"trained_r{r}_{p}.json"
            job = read(name)
            assert job["implementation"] == implementation()
            assert len(job["rows"]) == len(GAUGED_ARMS)*10 and job["regime"] == r and job["pair"] == p
            all_rows.extend(dict(regime=r, pair=p, **x) for x in job["rows"])
            sources.append(name)
            if job["path_diagnostic"]:
                paths.append(dict(regime=r, pair=p, arms=job["path_diagnostic"]))
    worst = max(all_rows, key=lambda x: x["max_grid_llr_diff"])
    train = dict(checks=len(all_rows), passed=sum(x["passed"] for x in all_rows),
                 coded_flips=sum(x["coded_flips"] for x in all_rows),
                 coded_decisions=sum(x["decisions"] for x in all_rows),
                 max_abs_llr=worst["max_grid_llr_diff"], worst={k: worst[k] for k in ("regime","pair","arm","condition")},
                 max_rel_cfr=max(x["max_relative_cfr_l2"] for x in all_rows),
                 failures=[x for x in all_rows if not x["passed"]], files=sources,
                 pairwise_physical_estimation=paths)
    gates = pre["rows"]
    grads = [t for m in gates for t in m["gradient"]["tensors"]]
    anchors = [a for m in gates for a in m["anchors"]]
    numerical = dict(regime_arm_cells=len(gates), gradient_directions=len(grads),
                     active_real_per_cell=[m["active_real_parameters"] for m in gates],
                     min_live_real=min(m["active_real_parameters"] for m in gates),
                     max_finite_difference_absolute_error=max(t["absolute_error"] for t in grads),
                     worst_archived_anchor=max(max(a["archived_max_llr_cfr"]) for a in anchors),
                     exact_anchors=sum(a["p6_bit_exact"] for a in anchors), anchors_checked=len(anchors),
                     matched_capacity=pre["matched_capacity"])
    summary = dict(p6_tests=suite["run"], p5_tests=legacy["suites"]["p5"]["run"],
                   p5b_tests=legacy["suites"]["p5b"]["run"], p4_verifier=p4_text,
                   development_fixture_draws=legacy["unique_frame_draws"],
                   preflight=numerical, trained_gauge=train, passed=train["passed"] == train["checks"])
    save(root / "summary.json", summary)
    print(f"trained gain audit {train['passed']}/{train['checks']}, "
          f"{train['coded_flips']} flips, max delta LLR {train['max_abs_llr']:.3g}")
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=int, choices=SNR, nargs="+", default=list(SNR))
    parser.add_argument("--pair", type=int, choices=PAIRS, nargs="+", default=list(PAIRS))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        raise SystemExit(not summarize_checks())
    setup()
    results = [check_pair(regime, pair) for regime in args.regime for pair in args.pair]
    raise SystemExit(not all(results))
