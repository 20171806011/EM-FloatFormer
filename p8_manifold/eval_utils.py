"""Shared Phase 8 evaluation, holdout NLL, sequential assimilation selectivity, blind precision & C* audit helpers."""
from dataclasses import replace
import math

import torch

from p5b.diagnostics import SCALES
from p8_manifold.manifold_charts import offgrid_2d_lmmse
from p8_manifold.offgrid_sim import (
    calibration_offgrid,
    extract_payload_llrs,
    score_offgrid,
)


REAL = torch.float64
COMPLEX = torch.complex128


def offgrid_holdout_nll(log_pi, branch_pred_y, branch_pred_var, y_true, noise_var, probe_mask):
    """Exact C*-invariant mixture complex Gaussian predictive NLL on held-out probes (normalized by noise_var)."""
    if not bool(probe_mask.any()):
        return log_pi.new_zeros(())
    nv = noise_var[:, None, None, None].clamp_min(1e-300)
    var = branch_pred_var.clamp_min(1e-300)
    err2 = (y_true[:, None] - branch_pred_y).abs().square()
    # log CN(y; mu, var) - log(1 / (pi * nv)) = -err2/var - log(var / nv) - log(pi)
    log_comp = -err2 / var - torch.log(var / nv) - math.log(math.pi)
    log_mix = torch.logsumexp(log_pi[:, :, None, None] + log_comp, dim=1)  # B, T, N
    return -log_mix[probe_mask].mean()


def extract_cal_summary(cal_dict):
    b0 = cal_dict["bins"][0]
    tot = cal_dict["total"]
    return dict(
        bin0_count=b0["count"],
        bin0_sum_var=b0["sum_predicted_variance"],
        bin0_sum_mse=b0["sum_squared_error"],
        tot_count=tot["count"],
        tot_sum_var=tot["sum_predicted_variance"],
        tot_sum_mse=tot["sum_squared_error"],
    )


def summarize_ratio(sum_mse, sum_var):
    return (sum_mse / sum_var) if sum_var > 0 else None


@torch.no_grad()
def holdout_metrics_offgrid(frame, out, probe_mask):
    """Proper held-out predictive NLL, NMSE, calibration ratio, and 3-term variance decomposition on probes."""
    y_true = frame.y_clean_full[:, 0]
    nll = float(offgrid_holdout_nll(
        out.log_pi, out.branch_pred_y, out.branch_pred_var, y_true, frame.obs.noise_var, probe_mask
    ))
    err2 = (y_true - out.pred_y).abs().square()[probe_mask]
    eng = y_true.abs().square()[probe_mask]
    pv = out.pred_var[probe_mask]
    va = out.pred_var_aleatoric[probe_mask]
    ve = out.pred_var_epistemic[probe_mask]
    vb = out.pred_var_between[probe_mask]
    mse_sum = float(err2.sum())
    var_sum = float(pv.sum())
    return dict(
        count=int(probe_mask.sum()),
        nll=nll,
        mse_sum=mse_sum,
        energy_sum=float(eng.sum()),
        nmse=mse_sum / max(float(eng.sum()), 1e-300),
        var_sum=var_sum,
        cal_ratio=mse_sum / max(var_sum, 1e-300),
        var_aleatoric_sum=float(va.sum()),
        var_epistemic_sum=float(ve.sum()),
        var_between_sum=float(vb.sum()),
    )


def _selectivity_quartiles(rank_score, prob_shift, flipped):
    """Compute top-25% vs bottom-25% probability shift and decision flip rate ranked by rank_score."""
    flat_r = rank_score.reshape(-1)
    flat_p = prob_shift.reshape(-1)
    flat_f = flipped.reshape(-1).to(REAL)
    n_tot = int(flat_r.numel())
    k = max(1, n_tot // 4)
    order = torch.argsort(flat_r, stable=True)
    bot_idx = order[:k]
    top_idx = order[-k:]
    top_p = float(flat_p[top_idx].mean())
    bot_p = float(flat_p[bot_idx].mean())
    top_f = float(flat_f[top_idx].mean())
    bot_f = float(flat_f[bot_idx].mean())
    return dict(
        top25_prob_shift=top_p,
        bot25_prob_shift=bot_p,
        prob_shift_ratio=top_p / max(bot_p, 1e-12),
        top25_flip_rate=top_f,
        bot25_flip_rate=bot_f,
        flip_rate_ratio=top_f / max(bot_f, 1e-12),
    )


@torch.no_grad()
def sequential_update_metrics_offgrid(frame, out1, out2):
    """Compare Pass-1 vs assimilated Pass-2 decoding & selective uncertainty revision."""
    s1 = score_offgrid(frame, out1.bit_llr, out1.h_hat)
    s2 = score_offgrid(frame, out2.bit_llr, out2.h_hat)
    pm = frame.obs.pilot_mask
    llr1 = extract_payload_llrs(pm, out1.bit_llr).double()
    llr2 = extract_payload_llrs(pm, out2.bit_llr).double()
    p1 = torch.sigmoid(llr1)
    p2 = torch.sigmoid(llr2)
    prob_shift = (p2 - p1).abs()
    flipped = (llr2 > 0) != (llr1 > 0)

    # Disagreement-ranked quartiles (exact Phase 7 selectivity metric)
    dis1 = extract_payload_llrs(pm, out1.bit_disagreement).double()
    q_dis = _selectivity_quartiles(dis1, prob_shift, flipped)

    # Total symbol-variance-ranked quartiles (also meaningful for M=1 SingleHyp)
    b = frame.bits.shape[0]
    sv1 = torch.stack([out1.symbol_var_total[i, ~pm[i]] for i in range(b)], dim=0)[..., None].expand_as(prob_shift)
    q_var = _selectivity_quartiles(sv1, prob_shift, flipped)

    return dict(
        pass1_block_errors=s1["block_errors"],
        pass2_block_errors=s2["block_errors"],
        blocks=s1["blocks"],
        delta_block_errors=s1["block_errors"] - s2["block_errors"],  # >0 means Pass 2 improved!
        pass1_bce=s1["bce_sum"] / max(s1["bit_count"], 1),
        pass2_bce=s2["bce_sum"] / max(s2["bit_count"], 1),
        delta_bce=(s1["bce_sum"] - s2["bce_sum"]) / max(s1["bit_count"], 1),
        pass1_nmse=s1["channel_error"] / max(s1["channel_energy"], 1e-300),
        pass2_nmse=s2["channel_error"] / max(s2["channel_energy"], 1e-300),
        mean_prob_shift=float(prob_shift.mean()),
        mean_flip_rate=float(flipped.to(REAL).mean()),
        top25_prob_shift=q_dis["top25_prob_shift"],
        bot25_prob_shift=q_dis["bot25_prob_shift"],
        prob_shift_ratio=q_dis["prob_shift_ratio"],
        top25_flip_rate=q_dis["top25_flip_rate"],
        bot25_flip_rate=q_dis["bot25_flip_rate"],
        flip_rate_ratio=q_dis["flip_rate_ratio"],
        u_top25_prob_shift=q_var["top25_prob_shift"],
        u_bot25_prob_shift=q_var["bot25_prob_shift"],
        u_prob_shift_ratio=q_var["prob_shift_ratio"],
        u_top25_flip_rate=q_var["top25_flip_rate"],
        u_bot25_flip_rate=q_var["bot25_flip_rate"],
        u_flip_rate_ratio=q_var["flip_rate_ratio"],
    )


@torch.no_grad()
def blind_weight_metrics(frame, out):
    """Diagnostics of blind Student-t observation weights on O2_blind_burst_erasure."""
    clean_p = frame.obs.pilot_mask & (~frame.corrupt_mask)
    corrupt_p = frame.corrupt_mask
    wc = float(out.pilot_weights[clean_p].mean()) if bool(clean_p.any()) else 0.0
    wb = float(out.pilot_weights[corrupt_p].mean()) if bool(corrupt_p.any()) else 0.0
    return dict(
        clean_count=int(clean_p.sum()),
        corrupt_count=int(corrupt_p.sum()),
        clean_w_sum=float(out.pilot_weights[clean_p].sum()),
        corrupt_w_sum=float(out.pilot_weights[corrupt_p].sum()),
        clean_w_mean=wc,
        corrupt_w_mean=wb,
        rejection_ratio=wc / max(wb, 1e-12),
    )


@torch.no_grad()
def evaluate_em8_batch(arm_name, model, frame, cond, batch_idx):
    """Run full Phase 8 evaluation for one EM8 arm on one Off-Grid batch."""
    out1 = model(frame.obs)
    score1 = score_offgrid(frame, out1.bit_llr, out1.h_hat)
    cal1 = calibration_offgrid(frame, out1.bit_llr, out1.traces)
    row = dict(
        batch_index=batch_idx,
        condition=cond,
        arm=arm_name,
        score=score1,
        cal=cal1,
        cal_stats=extract_cal_summary(cal1),
        pi_entropy=float(-(out1.pi * out1.log_pi).sum(-1).mean()),
    )
    if cond == "O2_blind_burst_erasure":
        row["blind_weights"] = blind_weight_metrics(frame, out1)
    if cond == "O3_holdout_sequential":
        row["holdout"] = holdout_metrics_offgrid(frame, out1, frame.probe_mask)
        out2 = model.assimilate(out1.state, frame.y_clean_full, frame.probe_mask)
        row["sequential"] = sequential_update_metrics_offgrid(frame, out1, out2)
        row["pass2_cal_stats"] = extract_cal_summary(calibration_offgrid(frame, out2.bit_llr, out2.traces))
    return row


@torch.no_grad()
def evaluate_lmmse_batch(frame, cond, batch_idx):
    """Evaluate fair dictionary-free classical 2D GP/Wiener LMMSE comparator on one Off-Grid batch."""
    llr1, h1, tr1 = offgrid_2d_lmmse(frame.obs)
    score1 = score_offgrid(frame, llr1, h1)
    cal1 = calibration_offgrid(frame, llr1, tr1)
    row = dict(
        batch_index=batch_idx,
        condition=cond,
        arm="OffGrid-2D-LMMSE",
        score=score1,
        cal=cal1,
        cal_stats=extract_cal_summary(cal1),
    )
    if cond == "O3_holdout_sequential":
        obs2 = replace(
            frame.obs,
            y=frame.y_clean_full,
            valid_mask=torch.ones_like(frame.obs.valid_mask),
        )
        llr2, h2, tr2 = offgrid_2d_lmmse(obs2)
        score2 = score_offgrid(frame, llr2, h2)
        row["sequential"] = dict(
            pass1_block_errors=score1["block_errors"],
            pass2_block_errors=score2["block_errors"],
            blocks=score1["blocks"],
            delta_block_errors=score1["block_errors"] - score2["block_errors"],
            pass1_bce=score1["bce_sum"] / max(score1["bit_count"], 1),
            pass2_bce=score2["bce_sum"] / max(score2["bit_count"], 1),
            delta_bce=(score1["bce_sum"] - score2["bce_sum"]) / max(score1["bit_count"], 1),
        )
    return row


@torch.no_grad()
def audit_model_symmetry_p8(model, frame, seed, threshold=1e-12):
    """Strict 27-scale SCALES x random phases C* gauge equivariance audit for EM8ManifoldReceiver."""
    ref = model(frame.obs)
    has_probe = bool(frame.probe_mask.any())
    if has_probe:
        ref_nll = float(offgrid_holdout_nll(
            ref.log_pi, ref.branch_pred_y, ref.branch_pred_var,
            frame.y_clean_full[:, 0], frame.obs.noise_var, frame.probe_mask
        ))
        ref_p2 = model.assimilate(ref.state, frame.y_clean_full, frame.probe_mask)
    else:
        ref_nll = 0.0
        ref_p2 = None

    gen = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
    phases = (2.0 * math.pi * torch.rand(len(SCALES), dtype=REAL, generator=gen)) - math.pi

    max_abs_llr = 0.0
    max_rel_llr = 0.0
    max_rel_h = 0.0
    max_rel_py = 0.0
    max_rel_pv = 0.0
    max_pi = 0.0
    max_w = 0.0
    max_nll = 0.0
    max_p2_rel_llr = 0.0
    flips = 0
    all_finite = True

    ref_llr_norm = max(float(ref.bit_llr.abs().max()), 1.0)
    ref_p2_llr_norm = max(float(ref_p2.bit_llr.abs().max()), 1.0) if ref_p2 is not None else 1.0

    for scale_val, angle in zip(SCALES, phases.tolist()):
        alpha = complex(scale_val * math.cos(angle), scale_val * math.sin(angle))
        a2 = abs(alpha) ** 2
        scaled_obs = replace(frame.obs, y=frame.obs.y * alpha, noise_var=frame.obs.noise_var * a2)
        got = model(scaled_obs)
        if not (bool(torch.isfinite(got.bit_llr).all()) and bool(torch.isfinite(got.pred_y).all())):
            all_finite = False

        abs_llr = float((got.bit_llr - ref.bit_llr).abs().max())
        rel_llr = abs_llr / ref_llr_norm
        h_ref = ref.h_hat * alpha
        py_ref = ref.pred_y * alpha
        pv_ref = ref.pred_var * a2
        rel_h = float((got.h_hat - h_ref).abs().max() / h_ref.abs().max().clamp_min(1e-300))
        rel_py = float((got.pred_y - py_ref).abs().max() / py_ref.abs().max().clamp_min(1e-300))
        rel_pv = float((got.pred_var - pv_ref).abs().max() / pv_ref.abs().max().clamp_min(1e-300))
        pi_diff = float((got.pi - ref.pi).abs().max())
        w_diff = float((got.pilot_weights - ref.pilot_weights).abs().max())
        flips += int(((got.bit_llr > 0) != (ref.bit_llr > 0)).sum())

        if has_probe:
            got_nll = float(offgrid_holdout_nll(
                got.log_pi, got.branch_pred_y, got.branch_pred_var,
                frame.y_clean_full[:, 0] * alpha, frame.obs.noise_var * a2, frame.probe_mask
            ))
            max_nll = max(max_nll, abs(got_nll - ref_nll))
            got_p2 = model.assimilate(got.state, frame.y_clean_full * alpha, frame.probe_mask)
            p2_rel = float((got_p2.bit_llr - ref_p2.bit_llr).abs().max()) / ref_p2_llr_norm
            max_p2_rel_llr = max(max_p2_rel_llr, p2_rel)
            flips += int(((got_p2.bit_llr > 0) != (ref_p2.bit_llr > 0)).sum())

        max_abs_llr = max(max_abs_llr, abs_llr)
        max_rel_llr = max(max_rel_llr, rel_llr)
        max_rel_h = max(max_rel_h, rel_h)
        max_rel_py = max(max_rel_py, rel_py)
        max_rel_pv = max(max_rel_pv, rel_pv)
        max_pi = max(max_pi, pi_diff)
        max_w = max(max_w, w_diff)

    passed_strict_abs = (
        all_finite
        and flips == 0
        and max_abs_llr < threshold
        and max_rel_h < threshold
        and max_rel_py < threshold
        and max_rel_pv < threshold
        and max_pi < threshold
        and max_w < threshold
        and max_nll < threshold
        and max_p2_rel_llr < threshold
    )
    passed_rel = (
        all_finite
        and flips == 0
        and max_rel_llr < threshold
        and max_rel_h < threshold
        and max_rel_py < threshold
        and max_rel_pv < threshold
        and max_pi < threshold
        and max_w < threshold
        and max_nll < threshold
        and max_p2_rel_llr < threshold
    )
    return dict(
        condition=frame.condition,
        transforms=len(SCALES),
        max_abs_llr_diff=max_abs_llr,
        max_rel_llr_diff=max_rel_llr,
        max_rel_h=max_rel_h,
        max_rel_pred_y=max_rel_py,
        max_rel_pred_var=max_rel_pv,
        max_pi_diff=max_pi,
        max_w_diff=max_w,
        max_nll_diff=max_nll,
        max_p2_rel_llr=max_p2_rel_llr,
        coded_flips=flips,
        all_finite=all_finite,
        passed_strict_abs=passed_strict_abs,
        passed=passed_rel,
    )
