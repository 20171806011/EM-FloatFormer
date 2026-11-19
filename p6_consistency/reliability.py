"""R2 Gaussian cavity messages with full frequency-domain propagation.

Unknown QAM interference is approximated as proper complex Gaussian using
decoder-extrinsic first/second moments.  This is not an exact QAM/codeword
posterior: cross-symbol and improper-complex covariances are omitted, and
repeated outer messages use the same observation.  Final channel LLRs use
the archived positive-one max-log readout and exclude own-symbol priors.
"""
from typing import NamedTuple

import math
import torch

from p5.unrolled_em_transformer import _llrs
from p5b.parity import min_sum_extrinsic


class SymbolPrior(NamedTuple):
    mean: torch.Tensor          # B,T,N complex
    variance: torch.Tensor      # B,T,N real; complex variance E|x-m|^2
    bit_logits: torch.Tensor    # B,T,N,4, positive-one extrinsic messages


class CavityResult(NamedTuple):
    z: torch.Tensor
    variance: torch.Tensor      # cavity effective complex-noise variance
    informed: torch.Tensor
    mean: torch.Tensor          # QAM posterior mean, pilots reinserted
    posterior_variance: torch.Tensor
    channel_llr: torch.Tensor   # no own-symbol prior is re-added
    diagnostics: dict


def _qam_moments(bit_logits, z=None, variance=None, informed=None):
    """Exact finite four-PAM moments for the stated separable belief."""
    prior = bit_logits.reshape(*bit_logits.shape[:-1], 2, 2)
    sign, inner = prior.unbind(-1)
    log_probability = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), -1)
    pam = bit_logits.new_tensor([-3., -1., 1., 3.]) / math.sqrt(10.)
    if z is not None:
        squared_distance = (torch.view_as_real(z)[..., None] - pam).square()
        likelihood = -squared_distance / variance[..., None, None]
        likelihood = torch.where(informed[..., None, None], likelihood, 0.)
        log_probability = log_probability + likelihood
    probabilities = log_probability.softmax(-1)
    axis_mean = (3 * (probabilities[..., 3] - probabilities[..., 0])
                 + probabilities[..., 2] - probabilities[..., 1]) / math.sqrt(10.)
    second = (probabilities * pam.square()).sum(-1)
    mean = torch.complex(axis_mean[..., 0], axis_mean[..., 1])
    variance = (second - axis_mean.square()).sum(-1).clamp_min(0.)
    return mean, variance


def make_prior(template, pilot_mask, pilots, previous_channel_llr=None, parity=True):
    """Build a prior solely from previous channel-LLR check-node messages.

    template has shape B,T,N and supplies shape/device/dtype, not data.
    The first pass (or parity=False) uses a uniform payload constellation.
    In later passes only min_sum_extrinsic(previous_channel_llr) is used;
    neither the old posterior nor its channel LLR is directly re-added.
    The same returned prior should be reused for re-demaps within one pass.
    """
    bit_logits = template.real.new_zeros(*template.shape, 4)
    if previous_channel_llr is not None and parity:
        if previous_channel_llr.shape != bit_logits.shape:
            raise ValueError("Previous channel LLR must have shape B,T,N,4")
        data = previous_channel_llr[:, ~pilot_mask]
        if data.shape[1] * 4 % 128:
            raise ValueError("Parity prior needs complete public 128-bit codewords")
        external = min_sum_extrinsic(data.reshape(template.shape[0], -1, 128))
        bit_logits[:, ~pilot_mask] = external.reshape_as(data)
    mean, variance = _qam_moments(bit_logits)
    mean = torch.where(pilot_mask, pilots, mean)
    variance = torch.where(pilot_mask, 0., variance)
    return SymbolPrior(mean, variance, bit_logits)


def cavity_demap(y, propagation, noise_var, weights, prior, pilot_mask, pilots):
    """One full-frequency Gaussian cavity pass for single-antenna R2.

    y: B,T,N; propagation: B,T,N,N (including diag(delta_h)); noise_var: B;
    weights: B,T,N nonnegative relative observation precisions.  For binary
    weights this exactly excludes missing output rows of the likelihood.
    Fractional weights define a weighted Gaussian likelihood, not an assertion
    that multiplying measured noise by sqrt(weight) leaves its law unchanged.

    With Gw=sqrt(W)G and yw=sqrt(W)y, C=N0 I+Gw V Gw^H, r=yw-Gw m,
    t=Gw^H C^-1 r and s=diag(Gw^H C^-1 Gw):
        z=m+t/s, tau=(1-v*s)/s.
    They equal explicit leave-one-symbol-out Gaussian likelihood parameters
    when the matrices are nonsingular and the numerical floors are inactive.

    A scale-covariant computational noise floor of 64*N*eps times the greater
    of N0 and maximum weighted row energy is used only for extremely low or
    zero reported noise.  Cancellation-induced nonpositive tau is floored in
    dimensionless symbol units, and every such event is reported.  Zero-noise
    channel LLRs use the archived finite sign saturation -log(eps); there is
    no claim of exact zero-noise Bayesian inference.  Zero-information columns
    contribute channel LLR zero and preserve the supplied prior.
    """
    if y.ndim != 3 or propagation.shape != (*y.shape, y.shape[-1]):
        raise ValueError("Expected y B,T,N and propagation B,T,N,N")
    if weights.shape != y.shape or prior.mean.shape != y.shape:
        raise ValueError("Weights and prior moments must match y")
    if bool((weights < 0).any()) or bool((noise_var < 0).any()):
        raise ValueError("Observation precision and noise variance must be nonnegative")
    b, t, n = y.shape
    eps = torch.finfo(y.real.dtype).eps
    root_weight = weights.sqrt()
    weighted_g = propagation * root_weight[..., None]
    weighted_y = y * root_weight
    row_power = weighted_g.abs().square().sum(-1)
    reference = torch.maximum(row_power.amax(-1), noise_var[:, None])
    safe_reference = torch.where(reference > 0, reference, 1.)
    g = weighted_g / safe_reference.sqrt()[..., None, None]
    observed = weighted_y / safe_reference.sqrt()[..., None]
    normalized_noise = noise_var[:, None] / safe_reference
    computational_floor = 64. * n * eps
    effective_noise = normalized_noise.clamp_min(computational_floor)
    floor_used = (normalized_noise < computational_floor) & (reference > 0)
    covariance = (g * prior.variance[..., None, :]) @ g.mH
    identity = torch.eye(n, dtype=y.dtype, device=y.device)
    covariance = (covariance + covariance.mH) / 2 + effective_noise[..., None, None] * identity
    residual = observed - (g @ prior.mean[..., None]).squeeze(-1)
    rhs = torch.cat((residual[..., None], g), -1)
    solved = torch.linalg.solve(covariance, rhs)
    filtered_residual, filtered_columns = solved[..., 0], solved[..., 1:]
    matched = (g.conj() * filtered_residual[..., None]).sum(-2)
    precision = (g.conj() * filtered_columns).sum(-2).real.clamp_min(0.)
    informed = (g.abs().square().sum(-2) > 0) & (precision > 0)
    safe_precision = torch.where(informed, precision, 1.)
    z = prior.mean + matched / safe_precision
    raw_variance = (1. - prior.variance * precision) / safe_precision
    symbol_floor = eps * torch.maximum(prior.variance, torch.ones_like(prior.variance))
    variance_clamped = informed & (raw_variance < symbol_floor)
    variance = torch.maximum(raw_variance, symbol_floor)
    variance = torch.where(informed, variance, 1.)
    z = torch.where(informed, z, prior.mean)
    posterior_mean, posterior_variance = _qam_moments(prior.bit_logits, z, variance, informed)
    posterior_mean = torch.where(pilot_mask, pilots, posterior_mean)
    posterior_variance = torch.where(pilot_mask, 0., posterior_variance)
    channel_llr = _llrs(z, variance)
    zero_noise = (noise_var == 0)[:, None, None, None]
    channel_llr = torch.where(zero_noise, -math.log(eps) * channel_llr.sign(), channel_llr)
    channel_llr = torch.where(informed[..., None], channel_llr, 0.)
    # Pilots never enter the codeword prior or final codeword scoring.
    channel_llr = torch.where(pilot_mask[..., None], 0., channel_llr)
    predictive_quadratic = (residual.conj() * filtered_residual).real.sum(-1)
    valid_count = (weights > 0).sum(-1)
    payload = ~pilot_mask
    payload_llr = channel_llr[:, payload].detach()
    has_payload = payload_llr.numel() > 0
    diagnostics = dict(
        predictive_quadratic_by_burst=predictive_quadratic.detach().sum(-1).tolist(),
        valid_observations_by_burst=valid_count.detach().sum(-1).tolist(),
        informed_payload=int(informed[:, payload].sum()),
        payload_symbols=int(informed[:, payload].numel()),
        computational_noise_floor_frames=int(floor_used.sum()),
        cavity_variance_clamps=int(variance_clamped[:, payload].sum()),
        mean_abs_channel_llr=float(payload_llr.abs().mean()) if has_payload else 0.,
        max_abs_channel_llr=float(payload_llr.abs().max()) if has_payload else 0.,
        mean_prior_variance=float(prior.variance[:, payload].detach().mean()) if has_payload else 0.,
        mean_posterior_variance=float(posterior_variance[:, payload].detach().mean()) if has_payload else 0.,
    )
    return CavityResult(z, variance, informed, posterior_mean, posterior_variance,
                        channel_llr, diagnostics)
