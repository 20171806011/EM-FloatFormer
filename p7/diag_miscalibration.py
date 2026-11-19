"""Stage 0 diagnostic: disentangle single-point G uncertainty vs multi-hypothesis M=4 on dev pairs 901..903."""
import math
import time

import torch
from torch.nn import functional as F

from p4.simulator import q16
from p5.study import counts, setup
from p5.wavefield_sim import CP, POWERS, frequency_basis, ray_phase
from p5b.operator_em_transformer import _ray_bank
from p5b.parity import min_sum_extrinsic
from p6.data import dev_frame
from p6.axial_em_transformer import _demap_input, _dictionary
from p6_followup.common import corrupt, load_model
from p6_consistency.calibration import calibration
from p6_consistency.observation import canonicalize, gauge, observation_gain
from p6_consistency.receiver import ObservationAwareReceiver
from p6_consistency.reliability import SymbolPrior, _qam_moments, cavity_demap, make_prior
from p6_consistency.weighted import frequency_operators, weighted_amplitude_update
from p6 import experiment as ex


REAL = torch.float64
COMPLEX = torch.complex128
PAIRS = (901, 902, 903)
CONDITIONS = ("intact", "interior_erased", "first_erased")
PAM = torch.tensor([-3., -1., 1., 3.], dtype=REAL) / math.sqrt(10.)


def true_propagation_operator(frame):
    """Build exact linear R2 frequency operator G_true[B, T, N, N] from simulation truth (dev diag only)."""
    truth = frame.truth
    b, _, t, n = frame.obs.y.shape
    clock = (torch.arange(t, dtype=REAL)[:, None] * (n + CP) + CP + torch.arange(n, dtype=REAL)) / n
    duration = (t * (n + CP) - 1) / n
    phase = ray_phase(truth.nu_start, truth.nu_end, clock.flatten(), duration)
    rot = torch.polar(torch.ones_like(phase), phase).reshape(b, -1, t, n).permute(0, 2, 3, 1)
    basis = frequency_basis(truth.delays, n)
    cpe = torch.ones((b, t), dtype=COMPLEX)
    ops = frequency_operators(basis, rot, cpe)  # B, T, P, F, K
    return torch.einsum("btpfk,bp->btfk", ops, truth.gains)


def verify_true_operator():
    frame = dev_frame(2, 6, 901, 2, snr=math.inf)
    G_true = true_propagation_operator(frame)
    tx = frame.obs.pilots.expand(2, -1, -1).clone()
    tx[:, ~frame.obs.pilot_mask] = q16(frame.bits.long()).to(COMPLEX)
    y_pred = torch.einsum("btfk,btk->btf", G_true, tx)
    err = float((y_pred - frame.obs.y[:, 0]).abs().max())
    if not (math.isfinite(err) and err < 1e-12):
        raise AssertionError(f"G_true reconstruction error too large: {err}")
    return err


def _mixture_llr(pi, llrs_m):
    """Exact log-space posterior bit LLR for mixture sum_m pi_m p_m(b | y)."""
    log_pi = pi.clamp_min(1e-300).log()[:, :, None, None, None]  # B, M, 1, 1, 1
    log_p1 = -F.softplus(-llrs_m)
    log_p0 = -F.softplus(llrs_m)
    num = torch.logsumexp(log_pi + log_p1, dim=1)
    den = torch.logsumexp(log_pi + log_p0, dim=1)
    return num - den


def _constellation_log_evidence(z, tau, prior_logits, pilot_mask, valid, eps_var=1e-12):
    """Dimensionless 16-QAM marginal constellation log-evidence over valid payload REs."""
    # z, tau: [B, M, T, N]; prior_logits: [B, M, T, N, 4]
    pam = PAM.to(device=z.device, dtype=z.real.dtype)
    prior = prior_logits.reshape(*prior_logits.shape[:-1], 2, 2)
    sign, inner = prior.unbind(-1)
    lp_prior = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), -1)
    lp_prior = lp_prior - torch.logsumexp(lp_prior, dim=-1, keepdim=True)
    tau_safe = tau.clamp_min(eps_var)[..., None, None]
    dist2 = (torch.view_as_real(z)[..., None] - pam).square()
    # Subtract log(pi * tau) / 2 per real I/Q axis (summed over I and Q gives -log(pi * tau)).
    lp_lik = -dist2 / tau_safe - 0.5 * torch.log(math.pi * tau_safe)
    axis_ev = torch.logsumexp(lp_prior + lp_lik, dim=-1)  # ..., 2 (I and Q)
    mask = (~pilot_mask) & valid  # B, T, N or T, N
    if mask.ndim == 2:
        mask = mask[None, None]
    else:
        mask = mask[:, None]
    return (axis_ev.sum(-1) * mask).sum((-2, -1))


def _physical_log_evidence(y, G, x, v, nv, valid):
    """Observed-waveform Gaussian marginal log-likelihood ratio over valid bins [B, M]."""
    # y: B, T, N; G: B, M, T, F, K; x, v: B, M, T, N; nv: B; valid: B, T, N or T, N
    pred = torch.einsum("bmtfk,bmtk->bmtf", G, x)
    resid2 = (y[:, None] - pred).abs().square()
    noise = nv[:, None, None, None]
    row_ref = G.abs().square().sum(-1).amax(-1, keepdim=True).amax(-2, keepdim=True)
    eps = torch.finfo(y.real.dtype).eps
    noise_safe = torch.maximum(noise, 64. * y.shape[-1] * eps * torch.maximum(row_ref, noise))
    eff_var = noise_safe + torch.einsum("bmtfk,bmtk->bmtf", G.abs().square(), v)
    w = valid[None, None] if valid.ndim == 2 else valid[:, None]
    ll = -((resid2 / eff_var) + torch.log(eff_var / noise_safe))
    return (ll * w).sum((-2, -1))


def _initialize_m4(obs, y, geometry, valid):
    """C*-invariant M=4 hypothesis initializer over dominant ray Doppler octants."""
    b, _, t, n = y.shape
    nominal = geometry[0]
    rows = obs.pilot_mask.any(-1).nonzero().flatten()
    taps, presence, fractions = [], [], []
    for row in rows:
        m_row = obs.pilot_mask[row] & valid[row]
        cnt = int(m_row.sum())
        fractions.append(cnt / max(int(obs.pilot_mask[row].sum()), 1))
        if cnt:
            design = nominal[:, m_row].mT * obs.pilots[row, m_row, None]
            samples = y[:, :, row, m_row]
            taps.append(samples @ torch.linalg.pinv(design).mT)
            presence.append(samples.abs().sum((1, 2)) > 0)
        else:
            taps.append(y.new_zeros(b, 1, nominal.shape[0]))
            presence.append(torch.zeros(b, dtype=torch.bool, device=y.device))
    taps = torch.stack(taps, 2)  # B, 1, S_p, P
    present = torch.stack(presence, -1)  # B, S_p
    row_weight = present * y.real.new_tensor(fractions)
    bank, means, default = _ray_bank(2, t, n)
    n_pos = len(bank) // 2  # 81 positive, 81 negative
    sampled = means[:, rows]  # bank, S_p
    corr = (taps * row_weight[:, None, :, None]).transpose(-1, -2) @ sampled.mH  # B, 1, P, bank
    energy = row_weight @ sampled.abs().square().T  # B, bank
    score = corr.abs().square().sum(1) / torch.where(energy > 0, energy, 1.)[:, None]  # B, P, bank

    def pick_best(sub_score, offset):
        top = sub_score.amax(-1, keepdim=True)
        err = 64. * torch.finfo(y.real.dtype).eps * top.amax(1, keepdim=True).sqrt()
        tol = 2. * top.sqrt() * err + err.square()
        idx = ((top - sub_score) <= tol).int().argmax(-1) + offset
        return idx

    pos_idx = pick_best(score[..., :n_pos], 0)
    neg_idx = pick_best(score[..., n_pos:], n_pos)
    best_all = pick_best(score, 0)
    enough = present.sum(-1, keepdim=True) >= 2
    pos_idx = torch.where(enough, pos_idx, default)
    neg_idx = torch.where(enough, neg_idx, n_pos - 1 if n_pos > 0 else default)
    best_all = torch.where(enough, best_all, default)

    branches = torch.stack([
        torch.stack([pos_idx[:, 0], pos_idx[:, 1], best_all[:, 2]], -1),
        torch.stack([pos_idx[:, 0], neg_idx[:, 1], best_all[:, 2]], -1),
        torch.stack([neg_idx[:, 0], pos_idx[:, 1], best_all[:, 2]], -1),
        torch.stack([neg_idx[:, 0], neg_idx[:, 1], best_all[:, 2]], -1),
    ], dim=1)  # B, 4, P

    rot = bank[branches].permute(0, 1, 3, 4, 2)  # B, M, T, N, P
    mean_rot = means[branches].transpose(-1, -2)  # B, M, T, P
    at_rows = mean_rot[:, :, rows]  # B, M, S_p, P
    norm = (at_rows.abs().square() * row_weight[:, None, :, None]).sum(2)  # B, M, P
    amp = (taps[:, None] * at_rows[:, :, None].conj() * row_weight[:, None, None, :, None]).sum(3)
    amp = amp / torch.where(norm > 0, norm, 1.)[:, :, None]  # B, M, 1, P
    basis = nominal.expand(b, 4, -1, -1)
    return amp, rot, mean_rot, basis, y.new_ones(b, 4, t), present.any(-1)


@torch.no_grad()
def run_oracle_g_true(observed, gain, G_true):
    """Run identical 3-step cavity loop with privileged G_true for calibration attribution."""
    obs, valid = canonicalize(observed, gain)
    y0 = obs.y[:, 0]
    nv = obs.noise_var.double().reshape(-1)
    b, t, n = y0.shape
    weights = valid.to(REAL).expand(b, -1, -1)
    traces = []
    prev_llr = None
    for step in range(3):
        prior = make_prior(y0, obs.pilot_mask, obs.pilots, prev_llr, parity=(step > 0))
        res = cavity_demap(y0, G_true, nv, weights, prior, obs.pilot_mask, obs.pilots)
        traces.append(dict(step=step, mean=res.mean.detach(), variance=res.posterior_variance.detach()))
        prev_llr = res.channel_llr
    final = cavity_demap(y0, G_true, nv, weights, prior, obs.pilot_mask, obs.pilots)
    llr = torch.where(final.informed[..., None], final.channel_llr, 0.)
    h_diag = G_true.diagonal(dim1=-2, dim2=-1)[:, None]
    return llr, h_diag, traces


@torch.no_grad()
def run_fixed_multihyp_m4(observed, gain, fast_diag_inner=False):
    """Zero-parameter M=4 multi-hypothesis EM + cavity posterior mixture."""
    obs, valid = canonicalize(observed, gain)
    y, nv, scale = gauge(obs, valid)
    b, r, t, n = y.shape
    m = 4
    weights = valid.to(REAL).expand(b * m, -1, -1)
    pilot_observed = obs.pilot_mask & valid
    geometry = _dictionary(2, n)
    amp, rot, mean_rot, basis, cpe, has_pilot = _initialize_m4(obs, y, geometry, valid)
    pwr = (y[:, :, pilot_observed].abs().square().mean((1, 2)) if bool(pilot_observed.any())
           else nv.new_zeros(b))
    ref = torch.where(pwr > 0, pwr, torch.where(nv > 0, nv, 1.))
    ridge = torch.diag_embed((nv / ref)[:, None] / nv.new_tensor(POWERS[2]))  # B, P, P
    ridge_m = ridge[:, None].expand(b, m, -1, -1).reshape(b * m, 3, 3)

    ops = frequency_operators(basis.reshape(b * m, 3, n), rot.reshape(b * m, t, n, 3), cpe.reshape(b * m, t))
    G = torch.einsum("btpfk,brp->brtfk", ops, amp.reshape(b * m, 1, 3)).reshape(b, m, t, n, n)
    pi = torch.full((b, m), 1.0 / m, dtype=REAL)

    y_m = y[:, 0:1].expand(b, m, t, n).reshape(b * m, t, n)
    nv_m = nv[:, None].expand(b, m).reshape(b * m)
    traces = []
    prev_mix_llr = None

    def demap_all(G_cur, prior_cur, force_cavity=False):
        if fast_diag_inner and not force_cavity:
            diag_h = G_cur.diagonal(dim1=-2, dim2=-1).reshape(b * m, 1, t, n)
            pm_in = prior_cur.mean.reshape(b, m, t, n)
            pv_in = prior_cur.variance.reshape(b, m, t, n)
            pred = torch.einsum("bmtfk,bmtk->bmtf", G_cur, pm_in).reshape(b * m, 1, t, n)
            # Clean residual after removing off-diagonal ICI mean
            diag_term = diag_h * prior_cur.mean.reshape(b * m, 1, t, n)
            clean_y = y_m[:, None] - (pred - diag_term)
            ici_var = torch.einsum("bmtfk,bmtk->bmtf", G_cur.abs().square(), pv_in)
            diag_pow = diag_h[:, 0].abs().square()
            ici_var = (ici_var.reshape(b * m, t, n) - diag_pow * prior_cur.variance.reshape(b * m, t, n)).clamp_min(0.)
            eff_nv = nv_m[:, None, None] + ici_var
            safe_diag = torch.where(diag_pow > 0, diag_pow, 1.)
            valid_bm = valid[None] if valid.ndim == 2 else valid[:, None].expand(b, m, t, n).reshape(b * m, t, n)
            informed = (diag_pow > 0) & valid_bm
            z = torch.where(informed, clean_y[:, 0] / torch.where(diag_pow > 0, diag_h[:, 0], 1.), prior_cur.mean.reshape(b * m, t, n))
            tau = torch.where(informed, eff_nv / safe_diag, 1.).clamp_min(1e-12)
            pm, pv = _qam_moments(prior_cur.bit_logits, z, tau, informed)
            pm = torch.where(obs.pilot_mask, obs.pilots, pm)
            pv = torch.where(obs.pilot_mask, 0., pv)
            from p5.unrolled_em_transformer import _llrs
            cllr = _llrs(z, tau)
            cllr = torch.where(informed[..., None] & (~obs.pilot_mask)[..., None], cllr, 0.)
            return z.reshape(b, m, t, n), tau.reshape(b, m, t, n), informed.reshape(b, m, t, n), pm.reshape(b, m, t, n), pv.reshape(b, m, t, n), cllr.reshape(b, m, t, n, 4)
        res = cavity_demap(y_m, G_cur.reshape(b * m, t, n, n), nv_m, weights, prior_cur,
                           obs.pilot_mask, obs.pilots)
        return (res.z.reshape(b, m, t, n), res.variance.reshape(b, m, t, n),
                res.informed.reshape(b, m, t, n), res.mean.reshape(b, m, t, n),
                res.posterior_variance.reshape(b, m, t, n), res.channel_llr.reshape(b, m, t, n, 4))

    for step in range(3):
        prior_b = make_prior(y[:, 0], obs.pilot_mask, obs.pilots, prev_mix_llr, parity=(step > 0))
        prior_m = SymbolPrior(
            prior_b.mean[:, None].expand(b, m, t, n).reshape(b * m, t, n),
            prior_b.variance[:, None].expand(b, m, t, n).reshape(b * m, t, n),
            prior_b.bit_logits[:, None].expand(b, m, t, n, 4).reshape(b * m, t, n, 4),
        )
        z_m, tau_m, inf_m, mean_m, var_m, cllr_m = demap_all(G, prior_m, force_cavity=False)
        ell_p = _physical_log_evidence(y[:, 0], G, mean_m, var_m, nv, valid)
        ell_c = _constellation_log_evidence(z_m, tau_m, prior_b.bit_logits[:, None].expand(b, m, t, n, 4),
                                            obs.pilot_mask, valid)
        # Mild damped Bayes log-evidence score for calibrated posterior branch weights
        pi = torch.softmax(0.25 * (ell_p + ell_c), dim=-1)
        mix_mean = (pi[:, :, None, None] * mean_m).sum(1)
        within_v = (pi[:, :, None, None] * var_m).sum(1)
        between_v = (pi[:, :, None, None] * (mean_m - mix_mean[:, None]).abs().square()).sum(1)
        mix_var = within_v + between_v
        traces.append(dict(step=step, mean=mix_mean.detach(), variance=mix_var.detach()))

        fitted = weighted_amplitude_update(
            mean_m.reshape(b * m, t, n), var_m.reshape(b * m, t, n),
            y_m[:, None], basis.reshape(b * m, 3, n), rot.reshape(b * m, t, n, 3),
            cpe.reshape(b * m, t), weights, ridge_m, None,
        )
        amp = fitted.amplitudes.mT.reshape(b, m, 1, 3)
        ops = frequency_operators(basis.reshape(b * m, 3, n), rot.reshape(b * m, t, n, 3), cpe.reshape(b * m, t))
        G = torch.einsum("btpfk,brp->brtfk", ops, amp.reshape(b * m, 1, 3)).reshape(b, m, t, n, n)
        prev_mix_llr = _mixture_llr(pi, cllr_m)

    # Final readout uses exact cavity_demap with same last-step prior_m (consistent with ObservationAwareReceiver)
    z_m, tau_m, inf_m, mean_m, var_m, cllr_m = demap_all(G, prior_m, force_cavity=True)
    ell_p = _physical_log_evidence(y[:, 0], G, mean_m, var_m, nv, valid)
    ell_c = _constellation_log_evidence(z_m, tau_m, prior_b.bit_logits[:, None].expand(b, m, t, n, 4),
                                        obs.pilot_mask, valid)
    pi = torch.softmax(0.25 * (ell_p + ell_c), dim=-1)
    llr = _mixture_llr(pi, cllr_m)
    informed_any = inf_m.any(1) & has_pilot[:, None, None]
    llr = torch.where(informed_any[..., None], llr, 0.)
    h_mix = (pi[:, :, None, None] * G.diagonal(dim1=-2, dim2=-1)).sum(1)[:, None] * scale[:, None, None, None]
    h_mix = torch.where(has_pilot[:, None, None, None], h_mix, 0.)
    return llr, h_mix, traces


def _aggregate_stats(rows):
    blocks = sum(r["score"]["block_errors"] for r in rows)
    total_blocks = sum(r["score"]["blocks"] for r in rows)
    bce = sum(r["cal"]["final"]["bce_sum"] for r in rows)
    bits = sum(r["cal"]["final"]["bit_count"] for r in rows)
    b0_count = sum(r["cal"]["steps"][2]["bins"][0]["count"] for r in rows)
    b0_pvar = sum(r["cal"]["steps"][2]["bins"][0]["sum_predicted_variance"] for r in rows)
    b0_mse = sum(r["cal"]["steps"][2]["bins"][0]["sum_squared_error"] for r in rows)
    tot_pvar = sum(r["cal"]["steps"][2]["total"]["sum_predicted_variance"] for r in rows)
    tot_mse = sum(r["cal"]["steps"][2]["total"]["sum_squared_error"] for r in rows)
    tot_cnt = sum(r["cal"]["steps"][2]["total"]["count"] for r in rows)
    return dict(
        block_errors=blocks, blocks=total_blocks, bler=blocks / total_blocks,
        bit_bce=bce / bits,
        bin0_count=b0_count,
        bin0_pred_var=(b0_pvar / b0_count) if b0_count else None,
        bin0_mse=(b0_mse / b0_count) if b0_count else None,
        bin0_ratio=(b0_mse / b0_pvar) if b0_pvar > 0 else None,
        total_ratio=(tot_mse / tot_pvar) if tot_pvar > 0 else None,
        total_mse=tot_mse / tot_cnt,
        total_pred_var=tot_pvar / tot_cnt,
    )


def main():
    setup()
    err = verify_true_operator()
    print(f"G_true reconstruction max error: {err:.3e}", flush=True)

    methods = ("single_G_est", "oracle_G_true", "fixed_multihyp_M4", "fixed_multihyp_M4_fastdiag")
    by_cond = {c: {m: [] for m in methods} for c in CONDITIONS}

    for pair in PAIRS:
        p6_model, _ = load_model(pair, "parity", development=True)
        single_rx = ObservationAwareReceiver(p6_model, use_variance=True, effective=True, cavity=True)
        for batch in range(4):
            frame = dev_frame(2, 6, pair, 8, batch)
            G_true = true_propagation_operator(frame)
            for cond in CONDITIONS:
                observed = corrupt(frame, cond)
                gain = observation_gain(observed.obs.pilot_mask, cond)

                with torch.no_grad():
                    llr_a, h_a = single_rx(observed.obs, gain)
                by_cond[cond]["single_G_est"].append(dict(
                    score=counts(observed, llr_a, h_a),
                    cal=calibration(observed, llr_a, single_rx.last_trace),
                ))

                llr_b, h_b, tr_b = run_oracle_g_true(observed.obs, gain, G_true)
                by_cond[cond]["oracle_G_true"].append(dict(
                    score=counts(observed, llr_b, h_b),
                    cal=calibration(observed, llr_b, tr_b),
                ))

                llr_c, h_c, tr_c = run_fixed_multihyp_m4(observed.obs, gain, fast_diag_inner=False)
                by_cond[cond]["fixed_multihyp_M4"].append(dict(
                    score=counts(observed, llr_c, h_c),
                    cal=calibration(observed, llr_c, tr_c),
                ))

                llr_d, h_d, tr_d = run_fixed_multihyp_m4(observed.obs, gain, fast_diag_inner=True)
                by_cond[cond]["fixed_multihyp_M4_fastdiag"].append(dict(
                    score=counts(observed, llr_d, h_d),
                    cal=calibration(observed, llr_d, tr_d),
                ))

    summary = {c: {m: _aggregate_stats(by_cond[c][m]) for m in methods} for c in CONDITIONS}

    # B8 wall-time benchmark (1 thread)
    sample = corrupt(dev_frame(2, 6, 901, 8, 0), "interior_erased")
    gain = observation_gain(sample.obs.pilot_mask, "interior_erased")
    timings = {}
    for name, fn in (
        ("fixed_M4_full_cavity", lambda: run_fixed_multihyp_m4(sample.obs, gain, fast_diag_inner=False)),
        ("fixed_M4_fastdiag_inner", lambda: run_fixed_multihyp_m4(sample.obs, gain, fast_diag_inner=True)),
    ):
        fn()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        timings[name + "_ms_B8"] = (time.perf_counter() - t0) * 1000.0 / 5.0

    report = dict(g_true_max_err=err, summary=summary, timings=timings)
    ex.save(ex.ROOT.parent / "p7/results/diag_miscalibration.json", report)

    print("\n=== Stage 0 Dev Calibration Diagnostic (pairs 901..903, 96 bursts, 1728 blocks) ===", flush=True)
    for c in CONDITIONS:
        print(f"\nCondition: {c}", flush=True)
        for m in methods:
            s = summary[c][m]
            r0 = f"{s['bin0_ratio']:.2f}x" if s["bin0_ratio"] is not None else "n/a"
            print(f"  {m:28s} BLER={s['block_errors']:4d}/{s['blocks']} ({100*s['bler']:.2f}%) "
                  f"BCE={s['bit_bce']:.4f} | <0.01 bin n={s['bin0_count']:5d} ratio={r0} "
                  f"| total ratio={s['total_ratio']:.3f}x", flush=True)
    print("\nTimings (1 thread, B=8):", timings, flush=True)


if __name__ == "__main__":
    main()
