"""Shared Phase 7 evaluation, classical/P6 comparator adapters, calibration & C* audit helpers."""
from dataclasses import replace
import math

import torch

from p5 import classical
from p5b.diagnostics import SCALES
from p6_consistency.calibration import calibration
from p6_consistency.receiver import ObservationAwareReceiver
from p7.conditions import apply_observation_gain, mask_for_condition
from p7.metrics import (
    calibration_metrics, decoding_metrics, holdout_metrics, holdout_nll_loss,
    sequential_update_metrics,
)


def extract_bin0_and_total(cal_dict):
    """Extract step-2 (final M-step) <0.01 bin stats and overall calibration stats cleanly."""
    s2 = cal_dict["steps"][2]
    b0 = s2["bins"][0]
    tot = s2["total"]
    return dict(
        bin0_count=b0["count"],
        bin0_sum_var=b0["sum_predicted_variance"],
        bin0_sum_mse=b0["sum_squared_error"],
        tot_count=tot["count"],
        tot_sum_var=tot["sum_predicted_variance"],
        tot_sum_mse=tot["sum_squared_error"],
    )


def summarize_bin0_ratio(sum_mse, sum_var):
    return (sum_mse / sum_var) if sum_var > 0 else None


@torch.no_grad()
def evaluate_em7_batch(arm_name, model, frame, cond, batch_idx, cond_seed):
    b = frame.obs.y.shape[0]
    gain1, gain2, probe_mask = mask_for_condition(frame.obs.pilot_mask, cond, b, seed=cond_seed)
    obs1 = apply_observation_gain(frame, gain1).obs
    out1 = model(obs1, gain1)
    score1 = decoding_metrics(frame, out1.bit_llr, out1.h_hat)
    cal1 = calibration_metrics(frame, out1)
    row = dict(
        batch_index=batch_idx, condition=cond, arm=arm_name,
        score=score1, cal=cal1, cal_stats=extract_bin0_and_total(cal1),
        pi_entropy=float(-(out1.pi * out1.pi.clamp_min(1e-300).log()).sum(-1).mean()),
    )
    if cond == "C5_holdout_update":
        hold1 = holdout_metrics(frame, out1, probe_mask)
        obs2 = apply_observation_gain(frame, gain2).obs
        out2 = model(obs2, gain2)
        seq = sequential_update_metrics(frame, out1, out2)
        row["holdout"] = hold1
        row["sequential"] = seq
        row["pass2_cal_stats"] = extract_bin0_and_total(calibration_metrics(frame, out2))
    return row


@torch.no_grad()
def _run_p6_consistent_any_gain(receiver, obs, gain):
    """Run ObservationAwareReceiver supporting either 2D [T,N] or per-burst 3D [B,T,N] gain."""
    if gain.ndim == 2:
        llr, h_hat = receiver(obs, gain)
        return llr, h_hat, receiver.last_trace
    # C4a_roving_window has per-burst [B,T,N] masks; evaluate per burst without touching p6_consistency.
    llrs, hs = [], []
    step_means = [[], [], []]
    step_vars = [[], [], []]
    for i in range(obs.y.shape[0]):
        sub_obs = replace(obs, y=obs.y[i:i + 1], noise_var=obs.noise_var[i:i + 1])
        li, hi = receiver(sub_obs, gain[i])
        llrs.append(li)
        hs.append(hi)
        for s_idx in range(3):
            step_means[s_idx].append(receiver.last_trace[s_idx]["mean"])
            step_vars[s_idx].append(receiver.last_trace[s_idx]["variance"])
    traces = [
        dict(step=s_idx, mean=torch.cat(step_means[s_idx], 0), variance=torch.cat(step_vars[s_idx], 0))
        for s_idx in range(3)
    ]
    return torch.cat(llrs, 0), torch.cat(hs, 0), traces


@torch.no_grad()
def evaluate_p6_consistent_batch(p6_model, frame, cond, batch_idx, cond_seed):
    rx = ObservationAwareReceiver(p6_model, use_variance=True, effective=True, cavity=True)
    b = frame.obs.y.shape[0]
    gain1, gain2, _ = mask_for_condition(frame.obs.pilot_mask, cond, b, seed=cond_seed)
    obs1 = apply_observation_gain(frame, gain1).obs
    llr1, h1, tr1 = _run_p6_consistent_any_gain(rx, obs1, gain1 if cond != "C0_intact" else gain1[0])
    score1 = decoding_metrics(frame, llr1, h1)
    cal1 = calibration(frame, llr1, tr1)
    row = dict(
        batch_index=batch_idx, condition=cond, arm="P6-Aware-Consistent",
        score=score1, cal=cal1, cal_stats=extract_bin0_and_total(cal1),
    )
    if cond == "C5_holdout_update":
        obs2 = apply_observation_gain(frame, gain2).obs
        llr2, h2, tr2 = _run_p6_consistent_any_gain(rx, obs2, gain2)
        score2 = decoding_metrics(frame, llr2, h2)
        row["sequential"] = dict(
            pass1_block_errors=score1["block_errors"],
            pass2_block_errors=score2["block_errors"],
            delta_block_errors=score1["block_errors"] - score2["block_errors"],
            pass1_bce=score1["bce_sum"] / max(score1["bit_count"], 1),
            pass2_bce=score2["bce_sum"] / max(score2["bit_count"], 1),
            delta_bce=(score1["bce_sum"] - score2["bce_sum"]) / max(score1["bit_count"], 1),
        )
    return row


@torch.no_grad()
def evaluate_waveform_dd_batch(frame, cond, batch_idx, cond_seed):
    b = frame.obs.y.shape[0]
    gain1, gain2, _ = mask_for_condition(frame.obs.pilot_mask, cond, b, seed=cond_seed)
    obs1 = apply_observation_gain(frame, gain1).obs
    llr1, h1 = classical.receive(obs1, "waveform_dd")
    score1 = decoding_metrics(frame, llr1, h1)
    row = dict(batch_index=batch_idx, condition=cond, arm="Waveform-DD", score=score1)
    if cond == "C5_holdout_update":
        obs2 = apply_observation_gain(frame, gain2).obs
        llr2, h2 = classical.receive(obs2, "waveform_dd")
        score2 = decoding_metrics(frame, llr2, h2)
        row["sequential"] = dict(
            pass1_block_errors=score1["block_errors"],
            pass2_block_errors=score2["block_errors"],
            delta_block_errors=score1["block_errors"] - score2["block_errors"],
            pass1_bce=score1["bce_sum"] / max(score1["bit_count"], 1),
            pass2_bce=score2["bce_sum"] / max(score2["bit_count"], 1),
        )
    return row


@torch.no_grad()
def audit_model_symmetry(model, frame, cond, seed, threshold=1e-12):
    """Test strict C* gauge equivariance across all 27 SCALES x random phases."""
    b = frame.obs.y.shape[0]
    gain, _, _ = mask_for_condition(frame.obs.pilot_mask, cond, b, seed=seed)
    _, _, probe_mask = mask_for_condition(frame.obs.pilot_mask, "C5_holdout_update", b, seed=seed)
    base_frame = apply_observation_gain(frame, gain)
    ref = model(base_frame.obs, gain)
    ref_nll = holdout_nll_loss(
        ref.log_pi, ref.branch_pred_y, ref.branch_pred_var,
        frame.obs.y[:, 0], frame.obs.noise_var, probe_mask,
    )

    gen = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
    phases = (2.0 * math.pi * torch.rand(len(SCALES), dtype=torch.float64, generator=gen)) - math.pi

    max_llr = 0.0
    max_pred_y = 0.0
    max_pred_var = 0.0
    max_pi = 0.0
    max_nll = 0.0
    flips = 0
    all_finite = True

    for scale_val, angle in zip(SCALES, phases.tolist()):
        alpha = complex(scale_val * math.cos(angle), scale_val * math.sin(angle))
        scaled_obs = replace(
            base_frame.obs,
            y=base_frame.obs.y * alpha,
            noise_var=base_frame.obs.noise_var * (abs(alpha) ** 2),
        )
        got = model(scaled_obs, gain)
        if not (bool(torch.isfinite(got.bit_llr).all()) and bool(torch.isfinite(got.pred_y).all())):
            all_finite = False
        got_nll = holdout_nll_loss(
            got.log_pi, got.branch_pred_y, got.branch_pred_var,
            frame.obs.y[:, 0] * alpha, frame.obs.noise_var * (abs(alpha) ** 2), probe_mask,
        )
        llr_diff = float((got.bit_llr - ref.bit_llr).abs().max())
        rel_py = float(((got.pred_y / alpha) - ref.pred_y).abs().max()
                       / ref.pred_y.abs().max().clamp_min(1e-300))
        rel_pv = float(((got.pred_var / (abs(alpha) ** 2)) - ref.pred_var).abs().max()
                       / ref.pred_var.abs().max().clamp_min(1e-300))
        pi_diff = float((got.pi - ref.pi).abs().max())
        nll_diff = float((got_nll - ref_nll).abs())
        flips += int(((got.bit_llr > 0) != (ref.bit_llr > 0)).sum())

        max_llr = max(max_llr, llr_diff)
        max_pred_y = max(max_pred_y, rel_py)
        max_pred_var = max(max_pred_var, rel_pv)
        max_pi = max(max_pi, pi_diff)
        max_nll = max(max_nll, nll_diff)

    passed = (
        all_finite
        and flips == 0
        and max_llr < threshold
        and max_pred_y < threshold
        and max_pred_var < threshold
        and max_pi < threshold
        and max_nll < threshold
    )
    return dict(
        condition=cond,
        transforms=len(SCALES),
        max_llr_diff=max_llr,
        max_rel_pred_y=max_pred_y,
        max_rel_pred_var=max_pred_var,
        max_pi_diff=max_pi,
        max_nll_diff=max_nll,
        coded_flips=flips,
        all_finite=all_finite,
        passed=passed,
    )
