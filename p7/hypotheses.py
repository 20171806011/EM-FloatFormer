"""C*-equivariant M=4 multi-hypothesis initialization, fast exact frequency operators,
weighted expected amplitude updates, physical/constellation log-evidence, and mixture posteriors."""
import math

import torch
from torch.nn import functional as F

from p5.unrolled_em_transformer import _llrs
from p5b.operator_em_transformer import _ray_bank
from p5b.parity import min_sum_extrinsic
from p6.math_chirp_fisher import reliability as full_aperture_reliability
from p6_consistency.reliability import SymbolPrior, _qam_moments


REAL = torch.float64
COMPLEX = torch.complex128
PAM = torch.tensor([-3., -1., 1., 3.], dtype=REAL) / math.sqrt(10.)


def canonicalize(obs, gain=None):
    """Zero out invalid/unobserved bins cleanly and return boolean validity mask [B,T,N]."""
    b, _, t, n = obs.y.shape
    if gain is None:
        valid = torch.ones((b, t, n), dtype=torch.bool, device=obs.y.device)
        return obs, valid
    g = gain.to(device=obs.y.device, dtype=REAL)
    if g.ndim == 2:
        g = g.expand(b, -1, -1)
    elif g.shape != (b, t, n):
        raise ValueError(f"Expected gain [T,N] or [{b},{t},{n}], got {tuple(g.shape)}")
    if g.is_complex() or bool((~torch.isfinite(g) | (g < 0)).any()):
        raise ValueError("Observation gain must be real, finite, and nonnegative")
    valid = g > 0
    denom = torch.where(valid, g, 1.)[:, None]
    y_clean = torch.where(valid[:, None], obs.y / denom, 0.)
    from dataclasses import replace
    return replace(obs, y=y_clean), valid


def pilot_gauge(obs, valid):
    """Per-burst full C* = R+ x U(1) canonical gauge computed strictly over valid observed pilots."""
    y = obs.y.to(COMPLEX)
    nv = obs.noise_var.double().reshape(-1)
    b = y.shape[0]
    known = (obs.pilot_mask[None] & valid)[:, None]  # B, 1, T, N
    has_obs = known.any(dim=(1, 2, 3))
    samples = torch.where(known, y, 0.)
    count = known.sum(dim=(1, 2, 3)).clamp_min(1)  # B
    peak = samples.abs().amax(dim=(1, 2, 3))  # B
    active = has_obs & (peak > 0)
    safe_peak = torch.where(peak > 0, peak, 1.)[:, None, None, None]
    s_norm = samples / safe_peak  # O(1) strictly on valid observed pilots
    mean_sq = s_norm.abs().square().sum(dim=(1, 2, 3)) / count
    pilot_rms = peak * torch.where(active, mean_sq, 1.).sqrt()

    # Strictly C*-equivariant global phase reference u in U(1) from valid observed pilots only
    pilots_conj = obs.pilots.to(COMPLEX).conj()[None, None]  # 1, 1, T, N
    proj = (s_norm * pilots_conj).sum(dim=(1, 2, 3))  # B complex
    flat_mag = s_norm.abs().flatten(1)
    dom_idx = flat_mag.argmax(-1)
    batch_idx = torch.arange(b, device=y.device)
    dom_s = s_norm.flatten(1)[batch_idx, dom_idx]
    dom_p = pilots_conj.expand(b, 1, -1, -1).flatten(1)[batch_idx, dom_idx]
    dom_vec = dom_s * dom_p
    ref_vec = torch.where(proj.abs() > 1e-8, proj, dom_vec)
    ref_mag = ref_vec.abs()
    safe_ref_mag = torch.where(ref_mag > 0, ref_mag, 1.)
    unit_phase = torch.where(active & (ref_mag > 0), ref_vec / safe_ref_mag, y.new_ones(b))

    y_valid = torch.where(valid[:, None], y, 0.)
    field_peak = y_valid.abs().flatten(1).amax(-1)
    safe_nv = torch.where(nv > 0, nv, 1.).sqrt()
    fallback = torch.where(nv > 0, safe_nv, torch.where(field_peak > 0, field_peak, 1.))
    scale_r = torch.where(active, pilot_rms, fallback)
    scale_safe = torch.where(scale_r > 0, scale_r, 1.)
    y_norm = (y_valid / scale_safe[:, None, None, None]) * unit_phase.conj()[:, None, None, None]
    nv_norm = nv / scale_safe.square()
    scale_c = scale_safe.to(COMPLEX) * unit_phase
    return y_norm, nv_norm, scale_c


def pilot_reliability_masked(y, noise, pilot_mask, valid):
    """Batched normalized Fisher pilot-aperture reliability supporting [B,T,N] valid masks."""
    if valid.ndim == 2 and bool(valid.all()):
        return full_aperture_reliability(y, noise, pilot_mask)
    if valid.ndim == 3 and bool(valid.all()):
        return full_aperture_reliability(y, noise, pilot_mask)
    b, _, t, n = y.shape
    if valid.ndim == 2:
        valid = valid.expand(b, -1, -1)
    nominal_count = pilot_mask.sum(-1).to(REAL)
    actual_mask = (pilot_mask[None] & valid).to(REAL)  # B, T, N
    actual_count = actual_mask.sum(-1)  # B, T
    time = torch.linspace(-1., 1., t, dtype=REAL, device=y.device)
    design = torch.stack((time, time.square()), -1)
    center = (nominal_count @ design) / nominal_count.sum().clamp_min(1.)
    centered = design - center
    full = centered.mT @ (nominal_count[:, None] * centered)
    if int(torch.linalg.matrix_rank(full)) < 2:
        return y.real.new_zeros(b)
    factor = torch.linalg.cholesky(full)
    rows = torch.linalg.solve_triangular(factor, centered.mT, upper=False).mT  # T, 2
    power = (y.abs().square().mean(1) * actual_mask).sum(-1) / actual_count.clamp_min(1.)
    nv = noise[:, None]
    usable = torch.where(nv > 0, (power / torch.where(nv > 0, nv, 1.) - 1.).clamp(0., 1.),
                         (power > 0).to(REAL))
    weight = actual_count * usable  # B, T
    total = weight.sum(-1, keepdim=True)
    mean = (weight @ rows) / torch.where(total > 0, total, 1.)
    local = rows[None] - mean[:, None]  # B, T, 2
    actual = local.mT @ (weight[..., None] * local)  # B, 2, 2
    return torch.linalg.eigvalsh(actual)[:, 0].clamp(0., 1.)


def frequency_operators_fast(basis, rot, cpe):
    """Exact unitary R2 frequency operator B_p = cpe * F * diag(rot_p) * F^H * diag(basis_p).

    Uses the exact DFT shift property: (F diag(rot_p) F^H)[f, k] = (1/N) FFT(rot_p)[(f - k) % N].
    Returns tensor of shape [B, T, P, F, K].
    """
    b, t, n, p = rot.shape
    if basis.ndim == 2:
        basis = basis.expand(b, -1, -1)
    # rot_spec[b, t, p, d] = (1/N) sum_m rot[b, t, m, p] exp(-2j pi d m / N)
    rot_spec = torch.fft.fft(rot.permute(0, 1, 3, 2), dim=-1, norm="backward") / n  # B, T, P, N
    idx = (torch.arange(n, device=rot.device)[:, None] - torch.arange(n, device=rot.device)[None, :]) % n
    circulant = rot_spec[..., idx]  # B, T, P, F, K
    return circulant * basis[:, None, :, None, :] * cpe[:, :, None, None, None]


def weighted_amplitude_solve(mean, variance, y, operators, weights, ridge, delta_h=None):
    """Exact BLAS-friendly normal-equation amplitude solve matching p6_consistency.weighted."""
    b, t, n = mean.shape
    p = operators.shape[2]
    r = y.shape[1]
    wc = weights.to(operators.dtype)  # B, T, N
    vc = variance.to(operators.dtype)  # B, T, N
    design = torch.einsum("btpfk,btk->btfp", operators, mean)  # B, T, F, P

    # BLAS bmm normal matrices over flattened M = T*F (zero sqrt)
    d_flat = design.reshape(b, t * n, p)
    d_w = (design * wc[..., None]).reshape(b, t * n, p)
    mean_normal = d_w.mH @ d_flat  # B, P, P

    # Expected Gram correction sum_{t,f,k} w[t,f] v[t,k] conj(op[p,f,k]) op[q,f,k]
    wv = (weights[:, :, :, None] * variance[:, :, None, :]).to(operators.dtype)  # B, T, F, K
    op_flat = operators.reshape(b, t, p, n * n).transpose(-1, -2).reshape(b, t * n * n, p)
    op_w = (operators * wv[:, :, None]).reshape(b, t, p, n * n).transpose(-1, -2).reshape(b, t * n * n, p)
    covariance = op_w.mH @ op_flat  # B, P, P

    if delta_h is None:
        residual_mean = y
        cross = y.new_zeros(b, p, r)
    else:
        residual_mean = y - delta_h * mean[:, None]
        diag_op = operators.diagonal(dim1=-2, dim2=-1)  # B, T, P, K
        cross = torch.einsum("btpk,btk,btk,brtk->bpr", diag_op.conj(), vc, wc, delta_h)

    rhs = torch.einsum("btfp,btf,brtf->bpr", design.conj(), wc, residual_mean) - cross
    normal = mean_normal + covariance + ridge.to(operators.dtype)
    normal = (normal + normal.mH) / 2.

    with torch.no_grad():
        _, info = torch.linalg.cholesky_ex(normal.detach(), check_errors=False)
        regular = info == 0
    identity = torch.eye(p, dtype=normal.dtype, device=normal.device)
    safe = torch.where(regular[:, None, None], normal, identity)
    amplitudes = torch.linalg.solve(safe, rhs)
    if not bool(regular.all()):
        amplitudes = amplitudes.clone()
        amplitudes[~regular] = torch.linalg.pinv(normal[~regular], hermitian=True) @ rhs[~regular]
    return amplitudes  # B, P, R


def initialize_hypotheses(obs, y, geometry, valid, multi_hyp=True):
    """Initialize M=4 (sign octant) or M=1 C*-invariant physical branches from valid pilots."""
    b, _, t, n = y.shape
    nominal = geometry[0]
    p_count = nominal.shape[0]
    if valid.ndim == 2:
        valid = valid.expand(b, -1, -1)
    rows = obs.pilot_mask.any(-1).nonzero().flatten()

    taps_list, pres_list, frac_list = [], [], []
    for row in rows:
        row_pilots = obs.pilot_mask[row]  # N bool
        base_cnt = max(int(row_pilots.sum()), 1)
        m_row = valid[:, row] & row_pilots[None]  # B, N
        frac_list.append(m_row.sum(-1).to(REAL) / base_cnt)  # B
        # Design on all pilot positions in this row masked per burst
        col_idx = row_pilots.nonzero().flatten()
        active_cols = m_row[:, col_idx]  # B, n_p
        design_base = nominal[:, col_idx].mT * obs.pilots[row, col_idx, None]  # n_p, P
        design_b = design_base[None] * active_cols[..., None]  # B, n_p, P
        samples_b = y[:, :, row, col_idx] * active_cols[:, None]  # B, 1, n_p
        has_row = active_cols.any(-1)
        pinv_taps = samples_b @ torch.linalg.pinv(design_b).transpose(-1, -2)
        taps_row = torch.where(has_row[:, None, None], pinv_taps, 0.)
        taps_list.append(taps_row)
        pres_list.append(has_row & (samples_b.abs().sum((1, 2)) > 0))

    taps = torch.stack(taps_list, 2)  # B, 1, S_p, P
    present = torch.stack(pres_list, -1)  # B, S_p
    row_weight = present.to(REAL) * torch.stack(frac_list, -1)  # B, S_p

    bank, means, default = _ray_bank(2, t, n)
    n_pos = len(bank) // 2
    sampled = means[:, rows]  # bank, S_p
    corr = (taps * row_weight[:, None, :, None]).transpose(-1, -2) @ sampled.mH  # B, 1, P, bank
    energy = row_weight @ sampled.abs().square().T  # B, bank
    score = corr.abs().square().sum(1) / torch.where(energy > 0, energy, 1.)[:, None]  # B, P, bank

    def pick_best(sub_score, offset):
        top = sub_score.amax(-1, keepdim=True)
        err = 64. * torch.finfo(y.real.dtype).eps * top.amax(1, keepdim=True).sqrt()
        tol = 2. * top.sqrt() * err + err.square()
        return ((top - sub_score) <= tol).int().argmax(-1) + offset

    best_all = pick_best(score, 0)
    enough = present.sum(-1, keepdim=True) >= 2
    best_all = torch.where(enough, best_all, default)

    if multi_hyp:
        pos_idx = torch.where(enough, pick_best(score[..., :n_pos], 0), default)
        neg_idx = torch.where(enough, pick_best(score[..., n_pos:], n_pos), n_pos + default)
        branches = torch.stack([
            torch.stack([pos_idx[:, 0], pos_idx[:, 1], best_all[:, 2]], -1),
            torch.stack([pos_idx[:, 0], neg_idx[:, 1], best_all[:, 2]], -1),
            torch.stack([neg_idx[:, 0], pos_idx[:, 1], best_all[:, 2]], -1),
            torch.stack([neg_idx[:, 0], neg_idx[:, 1], best_all[:, 2]], -1),
        ], dim=1)  # B, 4, P
    else:
        branches = best_all[:, None]  # B, 1, P

    m_cnt = branches.shape[1]
    rot = bank[branches].permute(0, 1, 3, 4, 2)  # B, M, T, N, P
    mean_rot = means[branches].transpose(-1, -2)  # B, M, T, P
    at_rows = mean_rot[:, :, rows]  # B, M, S_p, P
    norm = (at_rows.abs().square() * row_weight[:, None, :, None]).sum(2)  # B, M, P
    amp = (taps[:, None] * at_rows[:, :, None].conj() * row_weight[:, None, None, :, None]).sum(3)
    amp = amp / torch.where(norm > 0, norm, 1.)[:, :, None]  # B, M, 1, P
    basis = nominal.expand(b, m_cnt, -1, -1)
    cpe = y.new_ones(b, m_cnt, t)
    return amp, rot, mean_rot, basis, cpe, present.any(-1)


def diagonal_ici_demap(y, G, nv, valid, prior, pilot_mask, pilots):
    """Differentiable vectorized full-operator ICI demapper across B*M hypotheses."""
    # y: [BM, T, N]; G: [BM, T, F, K]; nv: [BM]; valid: [BM, T, N]
    diag_h = G.diagonal(dim1=-2, dim2=-1)  # BM, T, N
    pred_full = torch.einsum("btfk,btk->btf", G, prior.mean)
    clean_y = y - (pred_full - diag_h * prior.mean)
    row_var = torch.einsum("btfk,btk->btf", G.abs().square(), prior.variance)
    diag_pow = diag_h.abs().square()
    ici_var = (row_var - diag_pow * prior.variance).clamp_min(0.)
    eff_nv = nv[:, None, None] + ici_var
    informed = (diag_pow > 0) & valid
    safe_h = torch.where(diag_pow > 0, diag_h, 1.)
    safe_pow = torch.where(diag_pow > 0, diag_pow, 1.)
    z = torch.where(informed, clean_y / safe_h, prior.mean)
    eps_v = torch.finfo(REAL).eps * torch.maximum(prior.variance, torch.ones_like(prior.variance))
    tau = torch.where(informed, (eff_nv / safe_pow).clamp_min(eps_v.clamp_min(1e-12)), 1.)
    post_mean, post_var = _qam_moments(prior.bit_logits, z, tau, informed)
    post_mean = torch.where(pilot_mask, pilots, post_mean)
    post_var = torch.where(pilot_mask, 0., post_var)
    cllr = _llrs(z, tau)
    zero_noise = (nv == 0)[:, None, None, None]
    cllr = torch.where(zero_noise, -math.log(torch.finfo(REAL).eps) * cllr.sign(), cllr)
    cllr = torch.where(informed[..., None] & (~pilot_mask)[..., None], cllr, 0.)
    return z, tau, informed, post_mean, post_var, cllr


def branch_evidence(y, G, x, v, z, tau, prior_logits, nv, pilot_mask, valid):
    """Return scale-invariant physical waveform & discrete constellation log-evidence [B, M]."""
    b, m, t, n, _ = G.shape
    if valid.ndim == 2:
        valid = valid.expand(b, -1, -1)
    # Observed waveform Gaussian marginal LLR
    pred = torch.einsum("bmtfk,bmtk->bmtf", G, x)
    resid2 = (y[:, None] - pred).abs().square()
    noise = nv[:, None, None, None]
    row_ref = G.abs().square().sum(-1).amax(-1, keepdim=True).amax(-2, keepdim=True)
    eps = torch.finfo(REAL).eps
    noise_safe = torch.maximum(noise, 64. * n * eps * torch.maximum(row_ref, noise))
    eff_var = noise_safe + torch.einsum("bmtfk,bmtk->bmtf", G.abs().square(), v)
    w = valid[:, None].to(REAL)
    n_obs = w.sum((-2, -1)).clamp_min(1.)
    ell_phys = -((resid2 / eff_var + torch.log(eff_var / noise_safe)) * w).sum((-2, -1)) / n_obs

    # Discrete 16-QAM constellation marginal log-evidence on valid observed payload
    pam = PAM.to(device=z.device, dtype=REAL)
    prior = prior_logits.reshape(*prior_logits.shape[:-1], 2, 2)
    sign, inner = prior.unbind(-1)
    lp_prior = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), -1)
    lp_prior = lp_prior - torch.logsumexp(lp_prior, dim=-1, keepdim=True)
    tau_safe = tau.clamp_min(1e-12)[..., None, None]
    dist2 = (torch.view_as_real(z)[..., None] - pam).square()
    lp_lik = -dist2 / tau_safe - 0.5 * torch.log(math.pi * tau_safe)
    axis_ev = torch.logsumexp(lp_prior + lp_lik, dim=-1).sum(-1)  # B, M, T, N
    p_mask = ((~pilot_mask)[None] & valid)[:, None].to(REAL)
    n_pay = p_mask.sum((-2, -1)).clamp_min(1.)
    ell_const = (axis_ev * p_mask).sum((-2, -1)) / n_pay
    return ell_phys, ell_const


def mixture_posterior(pi, log_pi, llrs_m, means_m, vars_m):
    """Exact log-space bit mixture LLR, bit disagreement, and Law-of-Total-Variance decomposition."""
    # pi, log_pi: [B, M]; llrs_m: [B, M, T, N, 4]; means_m & vars_m: [B, M, T, N]
    log_pi_b = log_pi[:, :, None, None, None]
    log_p1 = -F.softplus(-llrs_m)
    log_p0 = -F.softplus(llrs_m)
    log_mix_p1 = torch.logsumexp(log_pi_b + log_p1, dim=1)
    log_mix_p0 = torch.logsumexp(log_pi_b + log_p0, dim=1)
    llr_mix = log_mix_p1 - log_mix_p0

    p1_m = torch.sigmoid(llrs_m)
    p1_mix = (pi[:, :, None, None, None] * p1_m).sum(1)
    bit_disagreement = (pi[:, :, None, None, None] * (p1_m - p1_mix[:, None]).square()).sum(1)

    sym_mean = (pi[:, :, None, None] * means_m).sum(1)
    within_var = (pi[:, :, None, None] * vars_m).sum(1)
    between_var = (pi[:, :, None, None] * (means_m - sym_mean[:, None]).abs().square()).sum(1)
    total_var = within_var + between_var
    return llr_mix, bit_disagreement, sym_mean, total_var, within_var, between_var


def make_code_prior(template, pilot_mask, pilots, previous_mix_llr=None, parity=True):
    """Construct extrinsic LDPC code prior per batch without own-symbol double counting."""
    bit_logits = template.real.new_zeros(*template.shape, 4)
    if previous_mix_llr is not None and parity:
        data = previous_mix_llr[:, ~pilot_mask]
        external = min_sum_extrinsic(data.reshape(template.shape[0], -1, 128))
        bit_logits[:, ~pilot_mask] = external.reshape_as(data)
    mean, variance = _qam_moments(bit_logits)
    mean = torch.where(pilot_mask, pilots, mean)
    variance = torch.where(pilot_mask, 0., variance)
    return SymbolPrior(mean, variance, bit_logits)
