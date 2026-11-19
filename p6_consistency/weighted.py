"""Exact weighted R2 amplitude moments for a fixed diagonal symbol covariance.

The observation weights act on *received frequency bins*.  They are not
transmit-symbol uncertainties and do not change which transmitted pilots are
known.  The caller supplies those pilot means with zero symbol variance.

The propagation and optional learned frequency-diagonal residual enter the
same expected waveform objective.  The covariance identity is exact for the
supplied independent-symbol q; this module does not claim q is the true joint
posterior, or that minimizing its objective improves BER.
"""

from typing import NamedTuple

import torch


class WeightedAmplitudeSolution(NamedTuple):
    amplitudes: torch.Tensor          # B,P,R
    normal: torch.Tensor              # B,P,P
    rhs: torch.Tensor                 # B,P,R, after covariance cross subtraction
    objective: torch.Tensor           # B, summed over receive antennas
    mean_residual_energy: torch.Tensor  # B
    uncertainty_energy: torch.Tensor  # B, includes learned diagonal and cross terms
    prior_energy: torch.Tensor        # B
    gram_correction: torch.Tensor     # B,P,P
    covariance_rhs: torch.Tensor      # B,P,R, positive cross term subtracted from rhs
    used_pinv: torch.Tensor           # B


def frequency_operators(basis, rot, cpe):
    r"""Return B[B,T,P,F,K] with B_p = cpe F diag(rot_p) F^H diag(basis_p).

    ``basis`` is B,P,N or P,N; ``rot`` is B,T,N,P; ``cpe`` is B,T.
    Fourier transforms are unitary.  Axis F labels an observed frequency bin
    and K a transmitted frequency bin.  No channel truth is needed.
    """
    if rot.ndim != 4:
        raise ValueError("Expected sample rotations [B,T,N,P]")
    b, t, n, p = rot.shape
    if basis.ndim == 2:
        basis = basis.expand(b, -1, -1)
    if basis.shape != (b, p, n) or cpe.shape != (b, t):
        raise ValueError("Incompatible basis or common-phase dimensions")
    identity = torch.eye(n, dtype=basis.dtype, device=basis.device)
    inverse_fourier = torch.fft.ifft(identity, dim=-2, norm="ortho")
    delayed = inverse_fourier[None, None, None] * basis[:, None, :, None, :]
    rotated = delayed * rot.permute(0, 1, 3, 2)[..., None]
    return torch.fft.fft(rotated, dim=-2, norm="ortho") * cpe[:, :, None, None, None]


def _inputs(mean, variance, y, weights, ridge, operators, delta_h):
    if mean.ndim != 3 or variance.shape != mean.shape or y.ndim != 4:
        raise ValueError("Expected mean/variance [B,T,N] and y [B,R,T,N]")
    b, t, n = mean.shape
    r = y.shape[1]
    p = operators.shape[2]
    if y.shape != (b, r, t, n) or operators.shape != (b, t, p, n, n):
        raise ValueError("Incompatible observation or propagation dimensions")
    if weights.ndim == 2:
        weights = weights.expand(b, -1, -1)
    if weights.shape != mean.shape:
        raise ValueError("Expected observation weights [B,T,N] or [T,N]")
    if ridge.ndim == 2:
        ridge = ridge.expand(b, -1, -1)
    if ridge.shape != (b, p, p):
        raise ValueError("Expected amplitude prior matrix [B,P,P] or [P,P]")
    ridge = ridge.to(dtype=operators.dtype, device=operators.device)
    if delta_h is None:
        delta_h = torch.zeros_like(y)
    if delta_h.shape != y.shape:
        raise ValueError("Expected fixed diagonal channel residual [B,R,T,N]")
    if variance.is_complex() or weights.is_complex():
        raise ValueError("Symbol variances and observation weights must be real")
    if bool((~torch.isfinite(variance) | (variance < 0)).any()):
        raise ValueError("Symbol variances must be finite and nonnegative")
    if bool((~torch.isfinite(weights) | (weights < 0)).any()):
        raise ValueError("Observation weights must be finite and nonnegative")
    return weights, ridge, delta_h


def weighted_terms(mean, variance, y, operators, weights, ridge, delta_h=None):
    r"""Build D(m), normal, rhs, and the full diagonal-q moment correction.

    Returns (design[B,T,F,P], normal, rhs, C, cross, weights, ridge, delta_h).
    The fixed learned residual is E_r=diag(delta_h[r]); the cross term is

        cross[p,r] = E[D(e)_p^H W E_r e]
                   = sum_{t,k} v[t,k] w[t,k] conj(B_p[k,k]) delta_h[r,t,k].

    The normal equation rhs is D(m)^H W (y-E_r m) - cross.  Omitting that
    subtraction would optimize a different expected objective whenever the
    learned residual and symbol uncertainty are both nonzero.
    """
    weights, ridge, delta_h = _inputs(mean, variance, y, weights, ridge, operators, delta_h)
    design = torch.einsum("btpfk,btk->btfp", operators, mean)
    wc = weights.to(operators.dtype)
    vc = variance.to(operators.dtype)
    mean_normal = torch.einsum("btfp,btf,btfq->bpq", design.conj(), wc, design)
    covariance = torch.einsum("btpfk,btf,btk,btqfk->bpq",
                              operators.conj(), wc, vc, operators)
    diagonal = operators.diagonal(dim1=-2, dim2=-1)  # B,T,P,K
    cross = torch.einsum("btpk,btk,btk,brtk->bpr", diagonal.conj(), vc, wc, delta_h)
    residual_mean = y - delta_h * mean[:, None]
    rhs = torch.einsum("btfp,btf,brtf->bpr", design.conj(), wc, residual_mean) - cross
    normal = mean_normal + covariance + ridge
    # Only remove floating-point anti-Hermitian reduction error.
    normal = (normal + normal.mH) / 2
    return design, normal, rhs, covariance, cross, weights, ridge, delta_h


def weighted_objective(mean, variance, y, operators, amplitudes, weights, ridge, delta_h=None):
    """Evaluate the same objective directly, retaining nonnegative terms.

    Returns (total, mean_residual, uncertainty, prior), each [B].  Directly
    forming the effective operator makes the uncertainty term a nonnegative
    weighted sum, avoiding cancellation between separate covariance terms.
    """
    weights, ridge, delta_h = _inputs(mean, variance, y, weights, ridge, operators, delta_h)
    if amplitudes.shape != (mean.shape[0], operators.shape[2], y.shape[1]):
        raise ValueError("Expected ray amplitudes [B,P,R]")
    effective = torch.einsum("btpfk,bpr->brtfk", operators, amplitudes)
    effective = effective + torch.diag_embed(delta_h)
    predicted_mean = torch.einsum("brtfk,btk->brtf", effective, mean)
    residual = ((y - predicted_mean).abs().square() * weights[:, None]).sum((1, 2, 3))
    uncertainty = (effective.abs().square() * weights[:, None, :, :, None]
                   * variance[:, None, :, None, :]).sum((1, 2, 3, 4))
    prior = (amplitudes.conj() * (ridge @ amplitudes)).real.sum((1, 2))
    return residual + uncertainty + prior, residual, uncertainty, prior


def weighted_amplitude_update(mean, variance, y, basis, rot, cpe, weights, ridge, delta_h=None):
    r"""Solve the frequency-weighted expected full-waveform amplitude step.

    Weights are nonnegative observation precision factors; passing zero
    removes a receive-frequency observation.  Weight/noise normalization and
    any known observation gains must be handled consistently by the caller.
    The supplied Hermitian PSD ridge is in the same objective units.

    Supports multiple receive antennas, each with its own amplitude vector
    and fixed frequency-diagonal learned residual.  Singular normal matrices
    use a Moore-Penrose solution and explicitly report that fact.
    """
    operators = frequency_operators(basis, rot, cpe)
    design, normal, rhs, covariance, cross, weights, ridge, delta_h = weighted_terms(
        mean, variance, y, operators, weights, ridge, delta_h)
    with torch.no_grad():
        _, info = torch.linalg.cholesky_ex(normal.detach(), check_errors=False)
        regular = info == 0
    identity = torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device)
    safe = torch.where(regular[:, None, None], normal, identity)
    amplitudes = torch.linalg.solve(safe, rhs)
    if not bool(regular.all()):
        amplitudes = amplitudes.clone()
        amplitudes[~regular] = torch.linalg.pinv(normal[~regular], hermitian=True) @ rhs[~regular]
    total, residual, uncertainty, prior = weighted_objective(
        mean, variance, y, operators, amplitudes, weights, ridge, delta_h)
    return WeightedAmplitudeSolution(amplitudes, normal, rhs, total, residual, uncertainty,
                                     prior, covariance, cross, ~regular)
