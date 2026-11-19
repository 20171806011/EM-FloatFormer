"""Conditional full-waveform profiling for the R2 inference study.

The fixed-q objective is an expected squared-error surrogate, not a marginal
likelihood.  ``gaussian_log_evidence`` has a separate, narrower contract:
deterministic transmitted symbols and a proper Gaussian amplitude prior.
The evidence and LLR-mixture helpers are mathematical utilities; their presence
does not imply that an end-to-end multi-hypothesis receiver was evaluated.
"""
from typing import NamedTuple

import math
import torch

from p6.axial_em_transformer import _sample_phase
from p6.math_chirp import _chirp_basis
from p6_softinfo.moments import waveform_gram_correction


class ProfileSolution(NamedTuple):
    amplitudes: torch.Tensor  # B,P,R
    objective: torch.Tensor   # B, without an arbitrary factor of 1/2
    normal: torch.Tensor      # B,P,P
    used_pinv: torch.Tensor   # B; no smooth rank-change claim is made


def profile_amplitudes(design, rx, covariance, ridge):
    """Minimize ||rx-Da||² + tr[aᴴ (covariance+ridge) a].

    ``design`` is B,L,P, ``rx`` is B,L,R and the last two arguments are
    Hermitian PSD B,P,P matrices in the same unweighted squared-error units.
    No covariance shrinkage or tunable multiplier is inserted.  Singular
    normal matrices use a Moore-Penrose solution and report that fact.
    The receiver refinement skips these singular bursts.
    """
    regularizer = covariance + ridge
    normal = design.mH @ design + regularizer
    # Remove only floating-point anti-Hermitian reduction error.
    normal = (normal + normal.mH) / 2
    rhs = design.mH @ rx
    with torch.no_grad():
        _, info = torch.linalg.cholesky_ex(normal.detach(), check_errors=False)
        regular = info == 0
    identity = torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device)
    safe = torch.where(regular[:, None, None], normal, identity)
    amplitudes = torch.linalg.solve(safe, rhs)
    if not regular.all():
        amplitudes = amplitudes.clone()
        amplitudes[~regular] = torch.linalg.pinv(normal[~regular], hermitian=True) @ rhs[~regular]
    residual = rx - design @ amplitudes
    penalty = (amplitudes.conj() * (regularizer @ amplitudes)).real.sum((-2, -1))
    objective = residual.abs().square().sum((-2, -1)) + penalty
    return ProfileSolution(amplitudes, objective, normal, ~regular)


def gaussian_log_evidence(design, rx, noise_variance, prior_covariance):
    """Exact log p(rx | D) for CN amplitudes and white proper Gaussian noise.

    Each receive antenna has independent a ~ CN(0, prior_covariance).
    Symbols/design are deterministic.  Positive noise variance and a positive
    definite amplitude prior are required.  This is NOT the exact P6 amplitude
    prior and NOT soft-symbol Bayesian evidence.
    """
    if bool((noise_variance <= 0).any()):
        raise ValueError("Gaussian evidence requires strictly positive noise variance")
    prior_root = torch.linalg.cholesky(prior_covariance)
    whitened_design = design @ prior_root
    identity = torch.eye(design.shape[-1], dtype=design.dtype, device=design.device)
    normal = identity + (whitened_design.mH @ whitened_design) / noise_variance[:, None, None]
    rhs = (whitened_design.mH @ rx) / noise_variance[:, None, None]
    whitened_amplitude = torch.linalg.solve(normal, rhs)
    residual = rx - whitened_design @ whitened_amplitude
    # This positive sum avoids a cancellation-prone Woodbury subtraction.
    quadratic = (residual.abs().square().sum((-2, -1)) / noise_variance
                 + whitened_amplitude.abs().square().sum((-2, -1)))
    root = torch.linalg.cholesky(normal)
    logdet_small = 2 * root.diagonal(dim1=-2, dim2=-1).real.log().sum(-1)
    length, receivers = rx.shape[-2:]
    logdet = length * noise_variance.log() + logdet_small
    return -(quadratic + receivers * logdet + length * receivers * math.log(math.pi))


def mix_bit_llrs(llrs, log_weights):
    """Mix normalized bit probabilities; positive LLR denotes bit one.

    llrs has shape B,S,... and log_weights B,S.  Weights need not already be
    normalized.  At least one finite log weight per burst is required.  These
    weights must come from a separately defined, scientifically justified
    model; this helper does not turn coherence scores into evidence.
    """
    if llrs.ndim < 2 or log_weights.shape != llrs.shape[:2]:
        raise ValueError("Expected llrs B,S,... and matching log_weights B,S")
    if bool((torch.isnan(log_weights) | torch.isposinf(log_weights)).any()):
        raise ValueError("Mode log weights may be finite or negative infinity")
    if bool((~torch.isfinite(log_weights).any(-1)).any()):
        raise ValueError("Every burst needs at least one finite mode log weight")
    log_prior = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
    log_prior = log_prior.reshape(*log_prior.shape, *([1] * (llrs.ndim - 2)))
    log_one = torch.nn.functional.logsigmoid(llrs)
    log_zero = torch.nn.functional.logsigmoid(-llrs)
    return torch.logsumexp(log_prior + log_one, 1) - torch.logsumexp(log_prior + log_zero, 1)


def _lift(theta, t, n):
    # The archived lru_cache can have been populated by a baseline forward
    # inside inference_mode.  A dtype/device no-op would retain that inference
    # tensor, which autograd cannot save for the profile Hessian.  Make an
    # ordinary tensor explicitly even when the cache was primed in that mode.
    with torch.inference_mode(False):
        polynomial = _chirp_basis(t).to(dtype=theta.dtype, device=theta.device).clone()
    row_phase = torch.einsum("bpk,tk->btp", theta, polynomial)
    return _sample_phase(row_phase, n)


def _phase_solution(theta, mean, variance, basis, rot, cpe, rx, ridge):
    """One fixed-q objective evaluation; all rays are solved jointly."""
    b, t, n = mean.shape
    updated_rot = rot * torch.exp(1j * _lift(theta, t, n))
    wave = torch.fft.ifft(mean[..., None] * basis.mT[:, None], dim=2, norm="ortho")
    design = (wave * updated_rot * cpe[:, :, None, None]).reshape(b, t*n, -1)
    covariance = waveform_gram_correction(variance, basis, updated_rot, cpe)
    return profile_amplitudes(design, rx, covariance, ridge), updated_rot


def phase_derivatives(theta, mean, variance, basis, rot, cpe, rx, ridge):
    """Return exact profile objective, gradient and real joint phase Hessian.

    theta is B,P,2.  Batches are independent; differentiating the sum of one
    gradient coordinate across bursts extracts the within-burst Hessian blocks
    without constructing an unnecessary (B*2P) by (B*2P) tensor.  This is an
    exact autograd Hessian of the profiled surrogate, not a Schur GN claim.
    Positive-definite normal matrices and locally fixed rank are required.
    """
    with torch.enable_grad():
        theta = theta.detach().requires_grad_(True)
        solution, _ = _phase_solution(theta, mean, variance, basis, rot, cpe, rx, ridge)
        if bool(solution.used_pinv.any()):
            raise ValueError("Profile derivatives require nonsingular normal matrices")
        gradient = torch.autograd.grad(solution.objective.sum(), theta, create_graph=True)[0]
        flat_gradient = gradient.flatten(1)
        rows = []
        for coordinate in range(flat_gradient.shape[-1]):
            row = torch.autograd.grad(flat_gradient[:, coordinate].sum(), theta,
                                      retain_graph=True)[0]
            rows.append(row.flatten(1))
        hessian = torch.stack(rows, 1)
        hessian = (hessian + hessian.mT) / 2
    return solution.objective.detach(), gradient.detach(), hessian.detach()


@torch.inference_mode(False)
def profile_refine(mean, variance, basis, rot, cpe, rx, ridge, max_steps=2):
    """Refine all R2 path phases under one fixed soft-symbol surrogate.

    Inference-only, with detached q and initial state.  Each exact real Hessian
    is shifted to SPD using a relative eigenvalue floor of 1e-6.  The sample
    phase change of a joint step is capped at 0.25 rad.  Eight deterministic
    backtracking trials (1 through 1/128), Armijo constant 1e-4, require both
    actual nonincrease and the sufficient-decrease inequality.  Neither BER
    improvement nor convergence of the outer receiver is implied.

    Returns (new_rot[B,T,N,P], amplitude[B,P,R], telemetry).  Singular baseline
    normal matrices are left unrefined; their valid minimum-norm amplitudes
    and explicit flags remain available.  No trainable parameters are added.
    """
    if max_steps not in (0, 1, 2):
        raise ValueError("This study fixes max_steps to zero, one or two")
    mean, variance, basis, rot, cpe, rx, ridge = (
        item.detach().clone() for item in (mean, variance, basis, rot, cpe, rx, ridge))
    b, t, n = mean.shape
    paths = basis.shape[-2]
    theta = mean.real.new_zeros(b, paths, 2)
    with torch.no_grad():
        initial, _ = _phase_solution(theta, mean, variance, basis, rot, cpe, rx, ridge)
    before = initial.objective.clone()
    eligible = ~initial.used_pinv
    accepted_count = torch.zeros(b, dtype=torch.int64, device=mean.device)
    evaluations = torch.zeros_like(accepted_count)
    invalid_curvature = torch.zeros_like(eligible)
    if bool(eligible.any()) and max_steps:
        selected = [item[eligible] for item in (mean, variance, basis, rot, cpe, rx, ridge)]
        local_theta = theta[eligible]
        local_counts = accepted_count[eligible]
        local_evals = evaluations[eligible]
        local_invalid = invalid_curvature[eligible]
        for _ in range(max_steps):
            objective, gradient, hessian = phase_derivatives(local_theta, *selected)
            flat = gradient.flatten(1)
            finite = torch.isfinite(hessian).all((-2, -1)) & torch.isfinite(flat).all(-1)
            local_invalid |= ~finite
            identity = torch.eye(2*paths, dtype=hessian.dtype, device=hessian.device)
            safe_hessian = torch.where(finite[:, None, None], hessian, identity)
            safe_gradient = torch.where(finite[:, None], flat, 0.)
            eigenvalues, eigenvectors = torch.linalg.eigh(safe_hessian)
            spectral_scale = torch.maximum(eigenvalues.abs().amax(-1), safe_gradient.norm(dim=-1))
            spectral_scale = torch.where(spectral_scale > 0, spectral_scale, 1.)
            shift = (1.e-6 * spectral_scale - eigenvalues[:, 0]).clamp_min(0.)
            coordinates = (eigenvectors.mT @ safe_gradient[..., None]).squeeze(-1)
            direction = -(eigenvectors @ (coordinates / (eigenvalues + shift[:, None]))[..., None]).squeeze(-1)
            direction = direction.reshape_as(local_theta)
            phase_peak = _lift(direction, t, n).abs().amax((1, 2, 3))
            divisor = torch.where(phase_peak > 0, phase_peak, 1.)
            cap = (.25 / divisor).clamp_max(1.)
            direction = direction * cap[:, None, None]
            slope = (gradient * direction).sum((1, 2))
            active = finite & torch.isfinite(slope) & (slope < 0)
            accepted = ~active
            updated = local_theta.clone()
            with torch.no_grad():
                for trial in range(8):
                    if bool(accepted.all()):
                        break
                    alpha = 2. ** -trial
                    candidate = local_theta + alpha * direction
                    value, _ = _phase_solution(candidate, *selected)
                    local_evals += (~accepted).to(local_evals.dtype)
                    improves = (torch.isfinite(value.objective) & ~value.used_pinv
                                & (value.objective <= objective)
                                & (value.objective <= objective + 1.e-4 * alpha * slope))
                    take = ~accepted & improves
                    updated = torch.where(take[:, None, None], candidate, updated)
                    accepted |= take
                    local_counts += take.to(local_counts.dtype)
            local_theta = updated.detach()
        theta[eligible] = local_theta
        accepted_count[eligible] = local_counts
        evaluations[eligible] = local_evals
        invalid_curvature[eligible] = local_invalid
    with torch.no_grad():
        final, new_rot = _phase_solution(theta, mean, variance, basis, rot, cpe, rx, ridge)
    telemetry = {
        "objective_before": before, "objective_after": final.objective,
        "accepted_steps": accepted_count, "candidate_evaluations": evaluations,
        "used_pinv": initial.used_pinv | final.used_pinv,
        "invalid_curvature": invalid_curvature,
        "max_total_phase_change": _lift(theta, t, n).abs().amax((1, 2, 3)),
    }
    telemetry = {name: value.detach().cpu().tolist() for name, value in telemetry.items()}
    return new_rot.detach(), final.amplitudes.detach(), telemetry
