"""Frozen Fourier-refit readout interventions on paired, fresh 18 dB OFDM slots."""

import json
from contextlib import ExitStack
from itertools import product
from pathlib import Path

import torch

from p4.classical_baselines import calibrated_16qam_llrs
from p4.em_floatformer import compute_gauge_anchor
from p4.pilot_prior_probe import FourierPilotProbe
from p4.refit_pilot_prior import CONDITIONS, per_slot_errors
from p4.slot_2d import SlotSimulator2D, _get_default_ldpc, _synchronize_slot, _train_emfloatformer_2d


SEEDS = (2026, 72431, 72432)
SCENARIOS = tuple(c for c in CONDITIONS if c[0] in
                  {"equal4_strong", "doppler005", "long12", "sparse12", "exp16"})
ARMS = tuple("".join(flags) for flags in product("01", repeat=3))
FACTORS = ("channel_residual", "distortion_estimate", "effective_gain")


def readout_terms(model, slot):
    """Capture the native double-precision quantities used by the public forward."""
    terms = {}

    def prior_hook(module, args):
        terms["prior"] = args[0][..., 1]

    def corrected_hook(module, args):
        terms["corrected"] = args[0].squeeze(-1)

    def distortion_hook(module, args, output):
        terms["distortion"] = output.squeeze(-1)

    with ExitStack() as stack:
        stack.enter_context(model.proj_g.register_forward_pre_hook(prior_hook))
        stack.enter_context(model.gamma_eff.register_forward_pre_hook(corrected_hook))
        stack.enter_context(model.head_dist.register_forward_hook(distortion_hook))
        public = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)["llrs"]
    return terms, public


def arm_llrs(model, terms, arm, y, nv, dtype):
    channel = terms["corrected"] if arm[0] == "1" else terms["prior"]
    dist = terms["distortion"] if arm[1] == "1" else torch.zeros_like(channel)
    h = model.gamma_eff(channel.unsqueeze(-1)).squeeze(-1) if arm[2] == "1" else channel
    power = h.real.square() + h.imag.square() + 1e-4
    z = y * h.conj() / power - dist * h.conj() / power
    variance = nv / power + (dist.real.square() + dist.imag.square()) / power + 0.035
    llrs = calibrated_16qam_llrs(z, variance, method="exact")
    return (20.0 * torch.tanh(llrs / 20.0)).to(dtype)


def marginal_effects(counts, blocks, bits):
    """On-minus-off error rate averaged over the four assignments of other factors."""
    effects = {}
    for i, name in enumerate(FACTORS):
        diffs = [0, 0]
        for off in ARMS:
            if off[i] == "0":
                on = off[:i] + "1" + off[i + 1:]
                for j in (0, 1):
                    diffs[j] += counts[on][j] - counts[off][j]
        effects[name] = {"bler": diffs[0] / (4 * blocks), "ber": diffs[1] / (4 * bits)}
    return effects


@torch.inference_mode()
def evaluate(models):
    code = _get_default_ldpc(128)
    pilot_only = FourierPilotProbe().eval()
    for head in (pilot_only.head_h, pilot_only.head_dist):
        for p in head.parameters():
            p.zero_()
    records = []
    comparisons = 0
    for ci, (name, profile, fd, cfo, pa) in enumerate(SCENARIOS):
        sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile=profile, fd=fd, cfo=cfo)
        counts = {seed: {arm: [0, 0] for arm in ARMS} for seed in models}
        blocks = bits = 0
        for bi in range(4):
            gen = torch.Generator().manual_seed(822000 + ci * 10 + bi)
            slot = _synchronize_slot(sim.generate_slot(batch_size=8, snr_db=18, pa_condition=pa,
                                                       coded=True, ldpc=code, generator=gen))
            blocks += slot.info_bits.numel() // code.k
            bits += slot.info_bits.numel()
            scale = compute_gauge_anchor(slot.yp_sparse.to(torch.complex128)).unsqueeze(-1)
            y = slot.y_grid.to(torch.complex128) / scale
            nv = slot.noise_var.double().unsqueeze(-1) / scale.square()
            zero_reference = pilot_only(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)["llrs"]
            for seed, model in models.items():
                terms, public = readout_terms(model, slot)
                for arm in ARMS:
                    llrs = arm_llrs(model, terms, arm, y, nv, public.dtype)
                    if arm in ("000", "111"):
                        reference = public if arm == "111" else zero_reference
                        if not torch.equal(llrs, reference):
                            raise AssertionError(f"LLR replay mismatch: {seed}, {name}, {bi}, {arm}")
                        comparisons += 1
                    failed, errors = per_slot_errors(llrs[:, slot.data_mask, :], slot, code)
                    counts[seed][arm][0] += sum(failed)
                    counts[seed][arm][1] += sum(errors)
        for seed, scores in counts.items():
            records.append({"seed": seed, "condition": name, "profile": profile, "fd": fd,
                            "cfo": cfo, "pa": pa, "slots": 32, "blocks": blocks, "bits": bits,
                            "counts": scores, "effects": marginal_effects(scores, blocks, bits)})
        print(f"Scored {name}", flush=True)
    return records, comparisons


def run():
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent / "results"
    initial = FourierPilotProbe().eval()
    initial.load_state_dict(torch.load(folder / "p4b_fourier_refit_state.pt", weights_only=True))
    models = {SEEDS[0]: initial}
    for seed in SEEDS[1:]:
        models[seed] = _train_emfloatformer_2d(steps=120, batch_size=8, seed=seed, model_factory=FourierPilotProbe)
        print(f"Trained Fourier refit {seed}", flush=True)
    rows, checks = evaluate(models)
    totals = {}
    for seed in models:
        selected = [r for r in rows if r["seed"] == seed]
        pooled = {arm: [sum(r["counts"][arm][i] for r in selected) for i in (0, 1)] for arm in ARMS}
        blocks = sum(r["blocks"] for r in selected)
        bits = sum(r["bits"] for r in selected)
        totals[seed] = {"blocks": blocks, "bits": bits, "counts": pooled,
                        "effects": marginal_effects(pooled, blocks, bits)}
    result = {
        "study": "Frozen inference interventions; all eight arms use the same trained hidden states.",
        "training": "Canonical Fourier refit: 120 steps, batch 8, equal4. Seed 2026 checkpoint reused, 72431 and 72432 freshly trained.",
        "evaluation": "18 dB; four batches of eight common slots per condition across every arm and seed. Evaluation seed 822000 + 10*condition index + batch index in recorded order.",
        "factors_in_bit_order": FACTORS,
        "off_values": ["zero residual (Fourier prior only)", "zero distortion in equalizer and variance", "complex unit gain"],
        "count_order": ["decoded block errors", "decoded information bit errors"],
        "effects": "On minus off error rate averaged over four assignments of the other two factors. Positive means more errors with factor enabled.",
        "bit_exact_control_batch_checks": checks,
        "controls": "All-on vs public trained forward and all-off vs zero-head identity-gain Fourier model. All public LLRs, including pilots, compared exactly.",
        "records": rows, "pooled_by_training_seed": totals,
    }
    (folder / "p4b_readout_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    run()
