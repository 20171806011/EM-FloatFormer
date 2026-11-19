"""Conditional QAM moments and the R2 waveform expected-Gram correction.

The symbol posterior is the same separable Gray 16-QAM approximation used by
P6.  Its diagonal covariance is a mean-field approximation: parity checks
couple symbols, but this module does not claim to retain those correlations.
For that diagonal covariance, ``waveform_gram_correction`` is exact in the
linear R2 waveform model (including within-symbol ray rotations / ICI).
"""

import math

import torch

from p5.unrolled_em_transformer import _llrs
from p5b.parity import min_sum_extrinsic


def symbol_moments(z, variance, informed, *, pilot_mask=None, pilots=None, parity=False):
    """Return ``E[x]`` and ``E[|x-E[x]|**2]`` on the frequency grid.

    Inputs have the P6 demapper layout ``[batch, symbol, subcarrier]``.
    ``variance`` is the positive complex-noise variance used by P6.  The
    uninformed channel likelihood is uniform, so its no-parity mean is
    exactly zero and its complex variance is one.  With parity enabled,
    only incoming check messages form the prior; the channel likelihood
    enters the symbol posterior once, as in ``soft_parity_symbols``.

    When ``pilot_mask`` is supplied, pilot means equal ``pilots`` (zero
    placeholders if ``pilots`` is omitted) and pilot variances are zero.
    Parity requires that mask and complete public 128-bit codewords.
    """
    pam = z.real.new_tensor([-3., -1., 1., 3.]) / math.sqrt(10)
    if parity:
        if pilot_mask is None:
            raise ValueError("Parity moments require the public pilot mask")
        data = ~pilot_mask
        z_data = z[:, data]
        sigma = variance[:, data]
        observed = informed[:, data]
        channel = torch.where(observed[..., None], _llrs(z_data, sigma), 0.)
        external = min_sum_extrinsic(channel.reshape(z.shape[0], -1, 128)).reshape_as(channel)
        external = external.reshape(*z_data.shape, 2, 2)
        squared_dist = (torch.view_as_real(z_data)[..., None] - pam).square()
        likelihood = torch.where(observed[..., None, None],
                                 -squared_dist / sigma[..., None, None], 0.)
        sign, inner = external.unbind(-1)
        prior = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), dim=-1)
        posterior = (likelihood + prior).softmax(-1)
        # Use exactly the archived parity mean's antisymmetric reduction.
        axes_mean = (3 * (posterior[..., 3] - posterior[..., 0])
                     + posterior[..., 2] - posterior[..., 1]) / math.sqrt(10)
        second = (posterior * pam.square()).sum(-1)
        conditional_variance = (second - axes_mean.square()).sum(-1).clamp_min(0.)
        mean = torch.zeros_like(z)
        symbol_variance = torch.zeros_like(z.real)
        mean[:, data] = torch.complex(axes_mean[..., 0], axes_mean[..., 1])
        symbol_variance[:, data] = conditional_variance
    else:
        squared_dist = (torch.view_as_real(z)[..., None] - pam).square()
        likelihood = -squared_dist / variance[..., None, None]
        posterior = torch.where(informed[..., None, None], likelihood, 0.).softmax(-1)
        axes_mean = (posterior * pam).sum(-1)
        mean = torch.complex(axes_mean[..., 0], axes_mean[..., 1])
        # P6 explicitly sets the uninformed mean to zero.
        mean = torch.where(informed, mean, 0.)
        second = (posterior * pam.square()).sum(-1)
        symbol_variance = (second.sum(-1) - mean.abs().square()).clamp_min(0.)
        symbol_variance = torch.where(informed, symbol_variance, 1.)

    if pilot_mask is not None:
        known = torch.zeros_like(z) if pilots is None else pilots
        mean = torch.where(pilot_mask, known, mean)
        symbol_variance = torch.where(pilot_mask, 0., symbol_variance)
    return mean, symbol_variance


def waveform_gram_correction(symbol_variance, basis, rot, cpe):
    r"""Return ``E[Delta D^H Delta D]`` for the full R2 sample-time design.

    ``symbol_variance``: [B,T,N], ``basis``: [B,P,N] (or [P,N]),
    ``rot``: [B,T,N,P], ``cpe``: [B,T].  If a deterministic rotation is
    absent, pass ``rot=None``.  The unitary-IFFT design is

        D[t,m,p] = cpe[t] rot[t,m,p] IFFT(x[t,:] basis[p,:])[m].

    For independent frequency-symbol errors, the equal-time covariance is
    (1/N) sum_k v[t,k] conj(basis[p,k]) basis[q,k].  Multiplying by the
    sample rotation cross-products, then summing m and t, retains the
    full cross-ray correction; keeping only its diagonal is not exact.
    The result is Hermitian positive semidefinite up to floating roundoff.
    Noise weighting and the amplitude prior remain the caller's responsibility.
    This identity does not extend through nonlinear PA emission in R1/R3.
    """
    b, t, n = symbol_variance.shape
    if basis.ndim == 2:
        basis = basis.expand(b, -1, -1)
    frequency_covariance = torch.einsum("btn,bpn,bqn->btpq",
                                       symbol_variance.to(basis.dtype),
                                       basis.conj(), basis) / n
    if rot is None:
        rotation_cross = frequency_covariance.new_full(frequency_covariance.shape, n)
    else:
        rotation_cross = torch.einsum("btnp,btnq->btpq", rot.conj(), rot)
    return (frequency_covariance * rotation_cross * cpe.abs().square()[:, :, None, None]).sum(1)
