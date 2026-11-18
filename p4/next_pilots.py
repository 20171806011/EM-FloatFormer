"""Development and fixed-gate replay for the sparse-pilot and frequency-offset workloads.

Run from the repository root with python -m p4.next_pilots. Receiver functions
take observations, known pilots and declared system parameters. Truth is used by
the scoring and explicitly labelled oracle controls only.
"""
import json
import math
from pathlib import Path

import torch

from p4.slot_2d import (
    EMFloatFormer2D, SLOT_PROFILES, SlotSimulator2D, compensate_cfo_time,
    estimate_cfo_cp_tensor, lmmse_2d_interpolate,
)


DEVELOPMENT_SEEDS = tuple(range(835101, 835113))
LEGACY_SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"


def fft_after_cfo(rx, cfo, n_symb, n_sc, cp):
    corrected = compensate_cfo_time(rx, cfo, n_sc=n_sc)
    useful = corrected.reshape(-1, n_symb, n_sc + cp)[..., cp:]
    return torch.fft.fft(useful, dim=-1, norm="ortho")


def pilot_refined_cfo(rx, pilots, mask, cp, tap_count):
    """Use uncontaminated CP samples and the phase drift between two pilot rows."""
    n_symb, n_sc = mask.shape
    view = rx.reshape(-1, n_symb, n_sc + cp)
    start = tap_count - 1
    if not 0 <= start < cp:
        raise ValueError("Need at least one cyclic-prefix pair beyond channel memory")
    pairs = view[..., start:cp].conj() * view[..., n_sc + start:n_sc + cp]
    coarse = pairs.sum(dim=(-2, -1)).angle() / (2 * math.pi)
    times = torch.where(mask.any(dim=1))[0]
    if len(times) != 2 or not torch.equal(mask[times[0]], mask[times[1]]):
        raise ValueError("Pair correction requires two identical frequency combs")
    y = fft_after_cfo(rx, coarse, n_symb, n_sc, cp)
    observations = (y[:, mask] * pilots.conj()).reshape(len(rx), 2, -1)
    residual_phase = (observations[:, 0].conj() * observations[:, 1]).sum(-1).angle()
    spacing = float(times[1] - times[0]) * (n_sc + cp) / n_sc
    return coarse, coarse + residual_phase / (2 * math.pi * spacing)


def estimate_from_pilots(y, pilots, mask, noise_var, profile, fd, cp):
    return lmmse_2d_interpolate(
        batch_size=len(y), n_symb=y.size(1), n_sc=y.size(2), cp=cp, fd=fd,
        profile=profile, noise_var=noise_var, yp_sparse=y[:, mask] * pilots.conj(),
        pilot_mask=mask,
    )


def joint_soft_channel(y, pilots, mask, noise_var, profile, fd, cp, iterations=6):
    """Pilot-initialized soft 16-QAM expectation/maximization for a diagonal OFDM channel."""
    batch, n_symb, n_sc = y.shape
    y64 = y.to(torch.complex128)
    nv = noise_var.reshape(batch, 1)
    pdp = torch.as_tensor(SLOT_PROFILES[profile], dtype=torch.float64, device=y.device)
    pdp = pdp / pdp.sum()
    time = torch.arange(n_symb, dtype=torch.float64, device=y.device)
    freq = torch.arange(n_sc, dtype=torch.float64, device=y.device)
    delay = torch.arange(len(pdp), dtype=torch.float64, device=y.device)
    dt = (time[:, None] - time[None, :]).abs() * (n_sc + cp) / n_sc
    temporal = torch.special.bessel_j0(2 * math.pi * fd * dt)
    eigenvalues, vectors = torch.linalg.eigh(temporal)
    time_basis = vectors * eigenvalues.clamp_min(0).sqrt()
    phase = -2 * math.pi * freq[:, None] * delay / n_sc
    frequency_basis = torch.polar(torch.ones_like(phase), phase) * pdp.sqrt()
    design = torch.kron(time_basis.contiguous().to(y64.dtype), frequency_basis.contiguous())
    adjoint = design.mH
    identity = torch.eye(design.size(1), dtype=y64.dtype, device=y.device)
    axis = torch.tensor([-3, -1, 1, 3], dtype=torch.float64, device=y.device) / math.sqrt(10)
    symbols = torch.complex(axis[:, None], axis[None, :]).flatten()
    h = estimate_from_pilots(y, pilots, mask, noise_var, profile, fd, cp).to(y64.dtype)
    for _ in range(iterations):
        metric = -(y64[..., None] - h[..., None] * symbols).abs().square()
        belief = torch.softmax(metric / nv.reshape(batch, 1, 1, 1), dim=-1)
        mean = (belief * symbols).sum(-1)
        power = (belief * symbols.abs().square()).sum(-1)
        mean[:, mask] = pilots.to(y64.dtype)
        power[:, mask] = pilots.abs().double().square()
        weight = power.flatten(1) / nv
        normal = identity + adjoint @ (weight[..., None] * design)
        data = (mean.conj() * y64).flatten(1) / nv
        coeff = torch.linalg.solve(normal, adjoint @ data.unsqueeze(-1))
        h = (design @ coeff).reshape_as(y64)
    return h.to(y.dtype)


def bit_signs(z):
    return torch.stack((z.real > 0, z.imag > 0), dim=-1)


@torch.inference_mode()
def legacy_static_cfo(seed=42):
    """Replay the original 0 Hz CP/2D-LMMSE receiver, independently of the public workload."""
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4", fd=0., cfo=.06)
    slot = sim.generate_slot(batch_size=8, snr_db=20., pa_condition="clean",
                             generator=torch.Generator().manual_seed(seed))
    epsilon = estimate_cfo_cp_tensor(slot.rx_time, 14, 32, 16)
    y = fft_after_cfo(slot.rx_time, epsilon, 14, 32, 16)
    h = estimate_from_pilots(y, slot.pilot_tx, slot.pilot_mask, slot.noise_var, "equal4", 0., 16)
    z = y[:, slot.data_mask] / h[:, slot.data_mask]
    phase = (z * slot.data_tx.conj()).sum(-1).angle().abs() * (180. / math.pi)
    return {"residual_phase_deg": float(phase.mean()),
            "quadrant_flips": int((bit_signs(z) != bit_signs(slot.data_tx)).sum()),
            "cfo_est_mean": float(epsilon.mean())}


def s03(seed, fd, model):
    n_symb, n_sc, cp = 14, 32, 16
    sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, cp=cp, profile="equal4", fd=fd, cfo=.06)
    slot = sim.generate_slot(batch_size=8, snr_db=20., pa_condition="clean",
                             generator=torch.Generator().manual_seed(seed))
    cp_all = estimate_cfo_cp_tensor(slot.rx_time, n_symb, n_sc, cp)
    trimmed, refined = pilot_refined_cfo(slot.rx_time, slot.pilot_tx, slot.pilot_mask, cp, 4)
    oracle = torch.full_like(refined, .06)
    mask = slot.data_mask
    truth = bit_signs(slot.data_tx)
    rows = {}
    for name, epsilon in (("legacy_cp", cp_all), ("trimmed_cp", trimmed),
                          ("pilot_refined_cp", refined), ("known_cfo", oracle)):
        y = fft_after_cfo(slot.rx_time, epsilon, n_symb, n_sc, cp)
        h = estimate_from_pilots(y, slot.pilot_tx, slot.pilot_mask, slot.noise_var, "equal4", fd, cp)
        z = y[:, mask] / h[:, mask]
        cpe_deg = (z * slot.data_tx.conj()).sum(-1).angle().abs().rad2deg()
        signed = bit_signs(z) != truth
        rows[name] = {
            "sign_errors": int(signed.sum()), "signs": signed.numel(),
            "quadrant_symbol_errors": int(signed.any(-1).sum()), "data_symbols": z.numel(),
            "slot_mean_abs_phase_deg": cpe_deg.tolist(),
            "abs_cfo_errors": (epsilon - .06).abs().tolist(),
        }
        if name == "pilot_refined_cp":
            out = model(y[:, slot.pilot_mask] * slot.pilot_tx.conj(), y, slot.pilot_mask, slot.noise_var)
            neural_signs = out.llrs[:, mask][:, :, [0, 2]] > 0
            rows["refined_frozen_em_sign_errors"] = int((neural_signs != truth).sum())
    if fd == 0.:
        y_true = fft_after_cfo(slot.rx_time, oracle, n_symb, n_sc, cp)
        known_h = slot.h_true[:, mask]
        errors = bit_signs(y_true[:, mask] / known_h) != truth
        # Payload FFT noise is circular iid after exact correction in the static
        # clean channel. Conditional on channel and data, the two real residuals
        # per carrier have independent Gaussian noise of variance nv/2.
        amp = torch.view_as_real(slot.data_tx).double().abs()
        snr_arg = amp * known_h.abs().double().unsqueeze(-1) / slot.noise_var.double().reshape(-1, 1, 1).sqrt()
        p_sign = 0.5 * torch.erfc(snr_arg)
        rows["known_cfo_and_h"] = {
            "sign_errors": int(errors.sum()), "signs": errors.numel(),
            "conditional_expected_sign_errors": float(p_sign.sum()),
            "log10_conditional_probability_zero_errors": float(torch.log1p(-p_sign).sum() / math.log(10.)),
        }
    return rows


def sparse_slot(seed, comb, pilot_symbol, snr_db):
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="train3",
                          fd=.01, pilot_symbols=(pilot_symbol,), comb=comb)
    return sim.generate_slot(batch_size=8, snr_db=snr_db, pa_condition="clean",
                             generator=torch.Generator().manual_seed(seed))


def sparse_errors(slot, pilot_symbol):
    """Per-slot NMSE. Keep observations separate from the reference used for scoring."""
    mask = slot.pilot_mask
    y = slot.y_grid
    tones = torch.where(mask[pilot_symbol])[0]
    nearest = (tones[None, :] - torch.arange(y.size(-1), device=y.device)[:, None]).abs().argmin(-1)
    h_nearest = slot.yp_sparse[:, nearest].unsqueeze(1).expand_as(y)
    h_lmmse = estimate_from_pilots(y, slot.pilot_tx, mask, slot.noise_var, "train3", .01, 16)
    h_joint = joint_soft_channel(y, slot.pilot_tx, mask, slot.noise_var, "train3", .01, 16)
    power = slot.h_true.abs().square().sum((1, 2))
    return {name: (h - slot.h_true).abs().square().sum((1, 2)) / power for name, h in
            (("nearest", h_nearest), ("pilot_lmmse", h_lmmse), ("joint_soft_map", h_joint))}


@torch.inference_mode()
def run_ultra_sparse(comb=4, pilot_symbol=2, snr_db=20., seed=42):
    """Joint pilot/data workload with the same slot generator, baseline, reference and denominator."""
    slot = sparse_slot(seed, comb, pilot_symbol, snr_db)
    errors = sparse_errors(slot, pilot_symbol)
    nearest = float(errors["nearest"].mean())
    overhead = float(slot.pilot_mask.sum()) / slot.pilot_mask.numel()

    def receiver_row(name):
        nmse = float(errors[name].mean())
        return {"stable": bool(torch.isfinite(errors[name]).all()), "overhead": overhead,
                "gain_over_ls_db": 10 * math.log10(max(nearest, 1e-12) / max(nmse, 1e-12)),
                "nmse_ls": nearest, "nmse_2d": nmse}

    return {**receiver_row("joint_soft_map"), "receiver": "pilot-initialized soft 16-QAM MAP",
            "legacy_pilot_only": receiver_row("pilot_lmmse")}


def s05(seed, model):
    slot = sparse_slot(seed, 4, 2, 20.)
    errors = sparse_errors(slot, 2)
    h_em = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var).h_hat
    power = slot.h_true.abs().square().sum((1, 2))
    curve = torch.special.bessel_j0(2 * math.pi * .01 * (torch.arange(14).double() - 2).abs() * 48 / 32)
    ideal_snapshot = curve[None, :, None] * slot.h_true[:, 2:3]
    out = {name: values.tolist() for name, values in errors.items()}
    for label, h in (("original_frozen_em", h_em), ("exact_midpoint_snapshot_predictor", ideal_snapshot)):
        nmse = (h - slot.h_true).abs().square().sum((1, 2)) / power
        out[label] = nmse.tolist()
    out["ideal_snapshot_population_nmse"] = float((1 - curve.square()).mean())
    return out


def summarize(records):
    gains = {}
    nearest = torch.tensor([n for r in records for n in r["s05"]["nearest"]], dtype=torch.float64)
    for name in ("pilot_lmmse", "joint_soft_map", "original_frozen_em", "exact_midpoint_snapshot_predictor"):
        values = torch.tensor([n for r in records for n in r["s05"][name]], dtype=torch.float64)
        gain = 10 * (nearest.mean() / values.mean()).log10()
        per_seed = 10 * (nearest.reshape(-1, 8).mean(-1) / values.reshape(-1, 8).mean(-1)).log10()
        gains[name] = {"mean_nmse": float(values.mean()), "gain_db": float(gain),
                       "seeds_above_4db": int((per_seed >= 4).sum())}
    cfo = {}
    for scenario in ("s03_static", "s03_150hz"):
        count = sum(r[scenario]["legacy_cp"]["signs"] for r in records)
        summary = {"signs": count}
        for name in ("legacy_cp", "trimmed_cp", "pilot_refined_cp", "known_cfo"):
            rows = [r[scenario][name] for r in records]
            avg = lambda key: sum(sum(z[key]) for z in rows) / sum(len(z[key]) for z in rows)
            summary[name] = {"sign_errors": sum(z["sign_errors"] for z in rows),
                             "mean_abs_phase_deg": avg("slot_mean_abs_phase_deg"),
                             "mean_abs_cfo_error": avg("abs_cfo_errors")}
        summary["refined_frozen_em_sign_errors"] = sum(r[scenario]["refined_frozen_em_sign_errors"] for r in records)
        if scenario == "s03_static":
            oracle = [r[scenario]["known_cfo_and_h"] for r in records]
            summary["known_cfo_and_h"] = {k:sum(z[k] for z in oracle) for k in oracle[0]}
        cfo[scenario] = summary
    return {"groups": len(records), "slots_per_scenario": 8 * len(records), "cfo": cfo,
            "s05": gains, "s05_nearest_nmse": float(nearest.mean())}


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    model = EMFloatFormer2D().eval()
    model.load_state_dict(torch.load(RESULTS / "p4b_seed2026_state.pt", weights_only=True))
    def measure(seed):
        return {"seed": seed, "s03_static": s03(seed, 0., model),
                "s03_150hz": s03(seed, 150. / 30_000, model), "s05": s05(seed, model)}
    development = [measure(seed) for seed in DEVELOPMENT_SEEDS]
    audit = measure(LEGACY_SEED)
    original_cfo = legacy_static_cfo(LEGACY_SEED)
    original_sparse = run_ultra_sparse(seed=LEGACY_SEED)["legacy_pilot_only"]
    audit_summary = summarize([audit])
    assert audit_summary["cfo"]["s03_static"]["legacy_cp"]["sign_errors"] == original_cfo["quadrant_flips"]
    assert abs(audit_summary["s05"]["pilot_lmmse"]["gain_db"] - original_sparse["gain_over_ls_db"]) < 1e-5
    report = {
        "development_seeds": list(DEVELOPMENT_SEEDS), "fixed_audit_seed": LEGACY_SEED,
        "method": "Trim four-tap ISI-contaminated CP samples (skip three), then estimate residual CFO from two pilot rows assuming stable channel. S05 uses six soft 16-QAM posterior-symbol iterations with known train3 covariance, AWGN variance and diagonal observation model; pilots remain fixed.",
        "scope": "S03 legacy uses fd=0; 150 Hz (fd=.005) tests original scenario blueprint separately. True-H/CFO reference applies only to the payload-FFT receiver in the static clean independent uncoded QAM case. The ideal S05 midpoint snapshot predictor cannot access within-symbol ICI or payload observations.",
        "development_summary": summarize(development), "audit_summary": audit_summary,
        "original_helpers": {"s03": original_cfo, "s05": original_sparse},
        "development": development, "audit": audit,
    }
    (RESULTS / "next_pilots.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"development": report["development_summary"], "audit": audit_summary}, indent=2))


if __name__ == "__main__":
    run()
