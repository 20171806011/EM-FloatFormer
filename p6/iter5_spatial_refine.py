"""One development-only follow-up to Iteration 5's first spatial candidate.

Before running any candidate fit, fix one formula: replace the spatial readout's
ray-power denominator by root mean-square cross-power, regularized by the
existing 0.04 floor in quadrature. This estimates directional coherence while
remaining continuous for a zero field. Preserve the .18 tanh cap, learned
antenna matrix, optimizer and all other settings. Fit both parity arms from
scratch on pairs 901--903 with both schedules. No additional coefficient grid.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import iter5_dev as study
from p6.data import PAIRS, dev_frame


OUT = study.OUT / "spatial_coherence"
original_handles = solver._physical_handles
original_apply = solver._apply_spatial


def coherence_handles(fields, h, amp, mean_rot, cpe, nominal_basis, regime, x, bank_residual=None):
    handles = original_handles(fields, h, amp, mean_rot, cpe, nominal_basis, regime, x, bank_residual)
    if regime != 3:
        return handles
    taps = (fields[..., 3] @ nominal_basis.mH) / h.shape[-1]
    ray = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
    overlap = (taps * ray.conj()).mean((1, 2)).abs()
    cross_power = taps.abs().square().mean((1, 2)) * ray.abs().square().mean((1, 2))
    return handles._replace(spatial=.18 * (overlap / (cross_power + .04**2).sqrt()).tanh())


def signature():
    return hashlib.sha256((study_signature() + study.prior.digest(Path(__file__))).encode()).hexdigest()


study_signature = study.signature


def fit(key):
    with (patch.object(study, "OUT", OUT), patch.object(study, "signature", signature),
          patch.object(solver, "_physical_handles", coherence_handles)):
        study.run(key)


@torch.inference_mode()
def baseline_stats():
    records = []
    for budget in (36, 48):
        for arm in study.ex.PARITY_ARMS:
            strengths, eigenspectra, changes, score_deltas = [], [], [], []
            for pair in PAIRS:
                model = study.model_for(3, pair, arm, .05, False)
                path = study.location(3, pair, arm, budget, .05, False)
                rec = study.ex.read(path)
                ck = path.with_suffix(".pt")
                if study.prior.digest(ck) != rec["checkpoint_sha256"]:
                    raise ValueError(f"Digest mismatch {ck}")
                model.load_state_dict(torch.load(ck, weights_only=True)["state"])
                frame = dev_frame(3, 6, pair, 32)

                def observe(amp, handle):
                    operator, strength = handle
                    strengths.append(strength.flatten())
                    eigenspectra.append(torch.linalg.eigvalsh(operator.expand(len(amp), -1, -1)))
                    updated = original_apply(amp, handle)
                    changes.append(((updated - amp).norm(dim=1)/amp.norm(dim=1).clamp_min(1.e-15)).flatten())
                    return updated

                with patch.object(solver, "_apply_spatial", observe):
                    score = study.ex.score([frame], lambda f: model(f.obs))
                study.prior.assert_replay(score, rec["score"])
                score_deltas.append({k: score[k] - v for k, v in rec["score"].items() if score[k] != v})
            records.append(dict(budget=budget, arm=arm, eigenvalue_mean=torch.cat(eigenspectra).mean(0).tolist(),
                                strength_quantiles=_quantiles(strengths), update_ratio_quantiles=_quantiles(changes),
                                instrumented_score_deltas=score_deltas))
    study.ex.save(OUT / "initial_diagnostics.json", records)
    for rec in records:
        print(rec, flush=True)


def _quantiles(values):
    quantiles = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
    return torch.quantile(torch.cat(values), quantiles).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("diagnose", "fit"))
    args = parser.parse_args()
    study.ex.setup()
    if args.action == "diagnose":
        baseline_stats()
    else:
        jobs = [(3, pair, arm, budget, .05, False) for budget in (36, 48)
                for pair in PAIRS for arm in study.ex.PARITY_ARMS]
        with ProcessPoolExecutor(max_workers=2, initializer=study.ex.setup) as pool:
            list(pool.map(fit, jobs, chunksize=1))


if __name__ == "__main__":
    main()
