"""Scoring, held-out pilot predictive consistency, calibration, and sequential update metrics."""
import math

import torch
from torch.nn import functional as F

from p4.simulator import q16
from p5.study import counts
from p5.wavefield_sim import CP, frequency_basis, ray_phase
from p6_consistency.calibration import calibration
from p7.hypotheses import frequency_operators_fast


REAL = torch.float64
COMPLEX = torch.complex128


def true_propagation_operator(frame):
    """Build exact R2 noiseless propagation operator G_true[B,T,N,N] (evaluation only)."""
    truth = frame.truth
    b, _, t, n = frame.obs.y.shape
    clock = (torch.arange(t, dtype=REAL)[:, None] * (n + CP) + CP + torch.arange(n, dtype=REAL)) / n
    duration = (t * (n + CP) - 1) / n
    phase = ray_phase(truth.nu_start, truth.nu_end, clock.flatten(), duration)
    rot = torch.polar(torch.ones_like(phase), phase).reshape(b, -1, t, n).permute(0, 2, 3, 1)
    basis = frequency_basis(truth.delays, n)
    cpe = torch.ones((b, t), dtype=COMPLEX)
    ops = frequency_operators_fast(basis, rot, cpe)
    return torch.einsum("btpfk,bp->btfk", ops, truth.gains)


@torch.no_grad()
def decoding_metrics(frame, bit_llr, h_hat):
    score = counts(frame, bit_llr, h_hat)
    mask = frame.obs.pilot_mask
    payload = bit_llr.detach()[:, ~mask].double()
    signed_bits = 2. * frame.bits.to(device=payload.device, dtype=REAL) - 1.
    bce_sum = float(F.softplus(-signed_bits * payload).sum())
    score["bce_sum"] = bce_sum
    score["bit_count"] = int(payload.numel())
    return score


def holdout_nll_loss(log_pi, branch_pred_y, branch_pred_var, target_y, noise_var, probe_mask):
    """Differentiable scale-normalized exact mixture Gaussian NLL on held-out probe bins.

    Formula per complex probe bin: -logsumexp_m(log(pi_m) - |y - mu_m|^2 / v_m - log(v_m / N0)).
    Strictly invariant to global C* gain alpha (when noise_var scales by |alpha|^2).
    """
    n_probe = probe_mask.sum().clamp_min(1)
    nv = noise_var.double().reshape(-1).clamp_min(1e-300)[:, None, None, None]  # B, 1, 1, 1
    diff2 = (target_y[:, None] - branch_pred_y).abs().square()  # B, M, T, N
    v_m = branch_pred_var.clamp_min(1e-300)
    lp_m = -diff2 / v_m - torch.log(v_m / nv)  # normalized by N0 for strict C* invariance
    nll_grid = -torch.logsumexp(log_pi[:, :, None, None] + lp_m, dim=1)  # B, T, N
    return (nll_grid * probe_mask).sum() / n_probe


@torch.no_grad()
def holdout_metrics(frame, out, probe_mask):
    """Evaluate held-out pilot waveform prediction accuracy and predictive calibration."""
    y0 = frame.obs.y[:, 0]
    nv = frame.obs.noise_var.double()
    y_probe = y0[probe_mask]
    pred_probe = out.pred_y[probe_mask]
    var_probe = out.pred_var[probe_mask]

    sq_err = float((y_probe - pred_probe).abs().square().sum())
    ref_pow = float(y_probe.abs().square().sum())
    sum_var = float(var_probe.sum())
    nll = float(holdout_nll_loss(out.log_pi, out.branch_pred_y, out.branch_pred_var, y0, nv, probe_mask))

    # Clean noiseless probe signal via exact G_true @ tx (evaluation only)
    G_true = true_propagation_operator(frame)
    tx = frame.obs.pilots.expand(y0.shape[0], -1, -1).clone()
    tx[:, ~frame.obs.pilot_mask] = q16(frame.bits.long()).to(COMPLEX)
    clean_y = torch.einsum("btfk,btk->btf", G_true, tx)
    clean_probe = clean_y[probe_mask]
    clean_sq_err = float((clean_probe - pred_probe).abs().square().sum())
    clean_ref_pow = float(clean_probe.abs().square().sum())

    return dict(
        probe_count=int(probe_mask.sum()),
        holdout_sq_err=sq_err,
        holdout_ref_pow=ref_pow,
        holdout_pred_var_sum=sum_var,
        holdout_nmse=sq_err / max(ref_pow, 1e-300),
        holdout_cal_ratio=sq_err / max(sum_var, 1e-300),
        holdout_clean_sq_err=clean_sq_err,
        holdout_clean_ref_pow=clean_ref_pow,
        holdout_clean_nmse=clean_sq_err / max(clean_ref_pow, 1e-300),
        holdout_nll_sum=nll * int(probe_mask.sum()),
        holdout_nll=nll,
    )


def _spearman_corr(x, y):
    """Tie-aware Spearman rank correlation for 1D real tensors."""
    if x.numel() < 3 or float(x.std()) == 0. or float(y.std()) == 0.:
        return 0.0
    def _rank(t):
        order = torch.argsort(t)
        ranks = torch.empty_like(t, dtype=REAL)
        ranks[order] = torch.arange(1, t.numel() + 1, dtype=REAL, device=t.device)
        # Average tied ranks cleanly
        uniq, inv, cnts = torch.unique(t, return_inverse=True, return_counts=True)
        if bool((cnts > 1).any()):
            sum_r = torch.zeros_like(uniq, dtype=REAL).scatter_add_(0, inv, ranks)
            ranks = (sum_r / cnts.to(REAL))[inv]
        return ranks - ranks.mean()
    rx, ry = _rank(x.double()), _rank(y.double())
    denom = rx.norm() * ry.norm()
    return float((rx * ry).sum() / denom) if float(denom) > 0 else 0.0


@torch.no_grad()
def sequential_update_metrics(frame, out1, out2):
    """Quantify selective belief revision from Pass 1 to Pass 2 after revealing 4 pilots."""
    mask = frame.obs.pilot_mask
    d1 = out1.bit_disagreement[:, ~mask].reshape(-1).double()
    llr1 = out1.bit_llr[:, ~mask].reshape(-1).double()
    llr2 = out2.bit_llr[:, ~mask].reshape(-1).double()
    p1 = torch.sigmoid(llr1)
    p2 = torch.sigmoid(llr2)
    prob_shift = (p2 - p1).abs()
    llr_shift = (llr2 - llr1).abs()
    flips = ((llr1 > 0) != (llr2 > 0)).to(REAL)

    q90 = torch.quantile(d1, 0.9)
    q50 = torch.quantile(d1, 0.5)
    high = d1 >= q90
    low = d1 <= q50
    if not bool(high.any()):
        high = d1 == d1.max()
    if not bool(low.any()):
        low = d1 == d1.min()

    n_high = int(high.sum())
    n_low = int(low.sum())
    ps_high_sum = float(prob_shift[high].sum())
    ps_low_sum = float(prob_shift[low].sum())
    ps_high = ps_high_sum / max(n_high, 1)
    ps_low = ps_low_sum / max(n_low, 1)

    fl_high_sum = float(flips[high].sum())
    fl_low_sum = float(flips[low].sum())
    fl_high = fl_high_sum / max(n_high, 1)
    fl_low = fl_low_sum / max(n_low, 1)

    ls_high_sum = float(llr_shift[high].sum())
    ls_low_sum = float(llr_shift[low].sum())
    ls_high = ls_high_sum / max(n_high, 1)
    ls_low = ls_low_sum / max(n_low, 1)

    split_mask = d1 > 0.01
    cert_mask = d1 < 1e-4
    n_split = int(split_mask.sum())
    n_cert = int(cert_mask.sum())
    sp_ps_sum = float(prob_shift[split_mask].sum()) if n_split > 0 else 0.0
    sp_fl_sum = float(flips[split_mask].sum()) if n_split > 0 else 0.0
    ce_ps_sum = float(prob_shift[cert_mask].sum()) if n_cert > 0 else 0.0
    ce_fl_sum = float(flips[cert_mask].sum()) if n_cert > 0 else 0.0
    if n_split > 0 and n_cert > 0:
        sp_ce_ps_ratio = (sp_ps_sum / n_split) / max(ce_ps_sum / n_cert, 1e-12)
        sp_ce_fl_ratio = (sp_fl_sum / n_split) / max(ce_fl_sum / n_cert, 1e-12)
    else:
        sp_ce_ps_ratio = None
        sp_ce_fl_ratio = None

    prob_corr = _spearman_corr(d1, prob_shift)
    llr_corr = _spearman_corr(d1, llr_shift)

    d_1 = decoding_metrics(frame, out1.bit_llr, out1.h_hat)
    d_2 = decoding_metrics(frame, out2.bit_llr, out2.h_hat)
    bce1 = d_1["bce_sum"] / max(d_1["bit_count"], 1)
    bce2 = d_2["bce_sum"] / max(d_2["bit_count"], 1)

    return dict(
        prob_shift_corr=prob_corr,
        high_disagreement_prob_shift=ps_high,
        low_disagreement_prob_shift=ps_low,
        high_disagreement_prob_shift_sum=ps_high_sum,
        low_disagreement_prob_shift_sum=ps_low_sum,
        high_count=n_high,
        low_count=n_low,
        top_vs_bottom_prob_shift_ratio=ps_high / max(ps_low, 1e-12),
        high_disagreement_flip_sum=fl_high_sum,
        low_disagreement_flip_sum=fl_low_sum,
        high_disagreement_flip_rate=fl_high,
        low_disagreement_flip_rate=fl_low,
        top_vs_bottom_flip_ratio=fl_high / max(fl_low, 1e-12),
        split_count=n_split,
        split_prob_shift_sum=sp_ps_sum,
        split_flip_sum=sp_fl_sum,
        certain_count=n_cert,
        certain_prob_shift_sum=ce_ps_sum,
        certain_flip_sum=ce_fl_sum,
        split_vs_certain_prob_shift_ratio=sp_ce_ps_ratio,
        split_vs_certain_flip_ratio=sp_ce_fl_ratio,
        disagreement_llr_shift_corr=llr_corr,
        high_disagreement_llr_shift_sum=ls_high_sum,
        low_disagreement_llr_shift_sum=ls_low_sum,
        top_vs_bottom_llr_shift_ratio=ls_high / max(ls_low, 1e-12),
        pass1_block_errors=d_1["block_errors"],
        pass2_block_errors=d_2["block_errors"],
        delta_block_errors=d_1["block_errors"] - d_2["block_errors"],
        pass1_bce=bce1,
        pass2_bce=bce2,
        delta_bce=bce1 - bce2,
        pass1_score=d_1,
        pass2_score=d_2,
    )


@torch.no_grad()
def calibration_metrics(frame, out):
    return calibration(frame, out.bit_llr, out.traces)
