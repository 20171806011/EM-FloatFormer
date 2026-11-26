"""Independent CPU check of the frozen Stage 3C E planning calculation."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import chi2, nct, norm, t


BLOCK = 64
ALPHA = 0.025
POWER = 0.95
SPECS = {"accuracy": (0.05, 0.02, 0.02), "nll": (0.06, 0.02, 0.03)}
COUNTS = np.arange(2, 4097)
SEED = 310327
REPS = 200_000
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.stage3c.verify import compare


def curve(sd, gain, margin, width, use_t):
    excess = (gain - margin) * np.sqrt(COUNTS) / sd
    df = COUNTS - 1
    cutoff = t.isf(ALPHA, df) if use_t else norm.isf(ALPHA)
    power = nct.sf(cutoff, df, excess) if use_t else norm.sf(cutoff - excess)
    plugin = cutoff * sd / np.sqrt(COUNTS)
    c4 = np.exp(gammaln((df + 1) / 2) - gammaln(df / 2) + 0.5 * np.log(2 / df))
    expected = plugin * c4 if use_t else plugin
    # Retain the original SD plug-in planning criterion for the width.
    eligible = COUNTS[(power >= POWER) & (plugin <= width)]
    if not eligible.size:
        raise ValueError("Increase the block search range.")
    return int(eligible[0]), power, plugin, expected


def extract(blocks, curves):
    return {
        name: dict(power=float(power[blocks - 2]), plugin_half_width=float(plugin[blocks - 2]),
                   expected_half_width=float(expected[blocks - 2]))
        for name, (_, power, plugin, expected) in curves.items()
    }


def plan(q, rho, nll_sd, pilot=None):
    deff = 1 + (BLOCK - 1) * rho
    sd = np.sqrt(np.array([q, nll_sd**2]) * deff / BLOCK)
    if pilot is not None:
        sd = np.maximum(sd, pilot)
    curves = [
        {name: curve(s, *spec, use_t) for (name, spec), s in zip(SPECS.items(), sd)}
        for use_t in (False, True)
    ]
    requirements = [{name: entry[0] for name, entry in group.items()} for group in curves]
    normal, exact = [max(32, *r.values()) for r in requirements]
    return dict(q=q, rho=rho, nll_mother_sd=nll_sd, pilot_block_sd=pilot,
                sd=dict(zip(SPECS, sd)),
                normal_by_endpoint=requirements[0], t_by_endpoint=requirements[1],
                normal_blocks=normal, normal_mothers=normal * BLOCK,
                t_blocks=exact, t_mothers=exact * BLOCK,
                normal_at_normal=extract(normal, curves[0]), t_at_normal=extract(normal, curves[1]),
                t_at_selected=extract(exact, curves[1]), t_at_previous=extract(exact - 1, curves[1]))


def crosscheck(design):
    rng = np.random.default_rng(SEED)
    rows = []
    for blocks in sorted({design["normal_blocks"], design["t_blocks"] - 1, design["t_blocks"]}):
        df = blocks - 1
        cutoff = t.isf(ALPHA, df)
        for name, (gain, margin, _) in SPECS.items():
            shift = math.sqrt(blocks) * (gain - margin) / design["sd"][name]
            # Integrate over the standardized sample variance, independently of nct.
            integral, err = quad(
                lambda v: norm.sf(cutoff * math.sqrt(v / df) - shift) * chi2.pdf(v, df),
                chi2.ppf(1e-13, df), chi2.isf(1e-13, df), epsabs=1e-11,
            )
            exact = nct.sf(cutoff, df, shift)
            assert abs(exact - integral) < 1e-9
            z = rng.standard_normal(REPS)
            denominator = np.sqrt(rng.chisquare(df, REPS) / df)
            hits = int(np.count_nonzero((z + shift) / denominator > cutoff))
            empirical = hits / REPS
            mc_se = math.sqrt(exact * (1 - exact) / REPS)
            assert abs(empirical - exact) < 5 * mc_se + 2 / REPS
            rows.append(dict(blocks=blocks, endpoint=name, nct=exact, quadrature=integral,
                             quadrature_error=err, rejections=hits, empirical=empirical, mc_se=mc_se))
    return rows


def waveform_check(blocks):
    sys.path[:0] = [str(ROOT), str(ROOT / "research/matched_cache")]
    from learned import load
    from research.stage3_ministral_cache import data
    from research.stage3_ministral_cache.schema import ROTATIONS, matched_sum, sector_of_complex

    weight = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest == "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
    encoder = load(weight)
    condition = "C4_cfo_ramp"
    tag = "work_e_cpu_only"
    namespace = 316027
    codes_seen = {sign: set() for sign in (-1, 1)}
    seeds = []
    for block_id in range(blocks):
        seed = int(np.random.SeedSequence([namespace, block_id]).generate_state(1)[0])
        seeds.append(seed)
        worlds = []
        for sign in (-1, 1):
            spec = replace(data.CONDITION_SPECS[condition], omega=sign * 0.4)
            with patch.dict(data.SPLIT_SEEDS, {tag: (seed, 1)}), patch.dict(
                data.CONDITION_SPECS, {condition: spec}
            ):
                batch = data.generate_mothers(tag, condition)
            worlds.append(batch)
            np.testing.assert_array_equal(np.sort(16 * batch.x1 + 4 * batch.x2 + batch.g),
                                          np.arange(BLOCK))
            base = [encoder.encode(y) for y in (batch.y1, batch.y2)]
            for a, b in ROTATIONS:
                z1, z2 = [encoder.encode(y) for y in batch.rotated_view(a, b)]
                np.testing.assert_array_equal(z1, (base[0] + 4 * a) % 16)
                np.testing.assert_array_equal(z2, (base[1] + 4 * b) % 16)
                target = sector_of_complex(matched_sum(batch.y3_repeat * data.ROOTS[a]))
                np.testing.assert_array_equal(target, (batch.k3_repeat + a) % 4)
                codes = 16 * z1 + z2
                assert np.all((0 <= codes) & (codes < 256))
                codes_seen[sign].update(codes.tolist())
        neg, pos = worlds
        for field in ("x1", "x2", "g", "y1"):
            np.testing.assert_array_equal(getattr(neg, field), getattr(pos, field))
        for step, field, label, window in (
            (2, "phi_2", "x2", "y2"), (4, "phi_3", "x1", "y3_repeat")
        ):
            np.testing.assert_allclose(getattr(pos, field) - getattr(neg, field),
                                       0.8 * step, rtol=0, atol=1e-12)
            def residual(batch):
                symbols = data.ROOTS[(getattr(batch, label) + batch.g) % 4, None]
                clean = symbols * np.exp(1j * getattr(batch, field)[:, None])
                return getattr(batch, window) - clean / np.sqrt(data.N_SAMPLES)
            np.testing.assert_allclose(residual(neg), residual(pos), rtol=0, atol=2e-14)
    assert len(set(seeds)) == blocks
    return dict(namespace=namespace, blocks=blocks, mothers=blocks * BLOCK,
                weight_sha256=digest, signs=[-1, 1], rotations=ROTATIONS,
                sign_rotation_rows=blocks * BLOCK * 6,
                unique_codes={sign: len(codes) for sign, codes in codes_seen.items()},
                seed_manifest_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-block-sd", nargs=2, type=float, metavar=("HIT", "NLL"))
    args = parser.parse_args()
    if args.pilot_block_sd and any(not math.isfinite(x) or x <= 0 for x in args.pilot_block_sd):
        parser.error("pilot SDs must be positive and finite")
    baseline = ROOT / "research/stage3c/E"
    frozen = json.loads((baseline / "results.json").read_text())
    rerun = json.loads(subprocess.check_output([sys.executable, "-B", str(baseline / "design.py")]))
    compare(frozen, rerun, "original E", rel_tol=2e-13, abs_tol=2e-13)
    scenarios = [plan(q, rho, 0.50) for q, rho in
                 [(0.35, .01), (.15, 0), (.35, 0), (.35, .03), (.50, .03)]]
    for result, original in zip(scenarios, [frozen["design"], *frozen["sensitivity"]]):
        assert result["normal_by_endpoint"] == original["required_blocks_by_endpoint"]
        assert result["normal_mothers"] == original["source_mothers"]
        for name, values in result["normal_at_normal"].items():
            ref = original["prospective_precision"][name]
            assert math.isclose(values["power"], ref["approx_power_against_margin"], abs_tol=1e-12)
            assert math.isclose(values["plugin_half_width"], ref["expected_95_half_width"])
    scenarios.append(plan(.35, .01, .75))
    if args.pilot_block_sd:
        scenarios.append(plan(.35, .01, .50, args.pilot_block_sd))
    print(json.dumps(dict(
        is_host_measurement=False, frozen_design_rerun_within_tolerance=True,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        enumerated_blocks=[int(COUNTS[0]), int(COUNTS[-1])], mc_seed=SEED, mc_reps_per_row=REPS,
        scenarios=scenarios, cpu_checks=crosscheck(scenarios[0]),
        frozen_waveform_check=waveform_check(scenarios[0]["t_blocks"]),
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
