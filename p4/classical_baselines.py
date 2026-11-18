"""Nonlinear Classical Receiver Baselines for Stage P4-0.

Includes:
- Bussgang Decomposition (analytical 1D quadrature calculation of attenuation gamma and distortion variance sigma_D^2).
- Truncated DFT Delay-Domain Channel Estimation (DFT-8).
- Bussgang-LMMSE Channel Equalization under Nonlinear PA Distortion.
- Calibrated 16-QAM Soft LLR Computation (Exact LogSumExp and Piecewise Max-Log).
- Iterative Decision-Directed PA Cancellation (DD-PA, 2-3 iterations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from p4.simulator import (
    DEFAULT_N,
    DEFAULT_OVERSAMPLE,
    SIGMA_16QAM,
    THRESH_16QAM,
    apply_rapp_pa_samples,
    decimate_time_to_frequency,
    hard_bits_16qam,
    oversample_frequency_to_time,
    q16,
)

# Constellation points along each 4-PAM axis: [-3, -1, +1, +3] / sqrt(10)
PTS_PAM4 = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32) / SIGMA_16QAM


# ---------------------------------------------------------------------------
# Bussgang Decomposition: Analytical Quadrature & Caching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def bussgang_params(
    A: Optional[float] = 2.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
    num_steps: int = 10000,
    r_max: float = 8.0,
) -> Tuple[complex, float, float]:
    """Compute Bussgang decomposition parameters gamma, sigma_D^2, and P_out.

    For input x ~ CN(0, 1) passing through nonlinear PA f(x):
      f(x) = gamma * x + d, where E[x^* d] = 0.
      gamma = E[f(x) x^*] / E[|x|^2]
      P_out = E[|f(x)|^2]
      sigma_D^2 = P_out - |gamma|^2

    Evaluated via high-precision 1D numerical integration over the Rayleigh amplitude distribution.
    """
    if A is None or math.isinf(A):
        return (1.0 + 0.0j, 0.0, 1.0)

    # Simpson's / trapezoidal grid over Rayleigh distribution
    r = torch.linspace(1e-6, r_max, num_steps, dtype=torch.float64)
    two_p = 2.0 * knee
    g_r = r / ((1.0 + (r / A) ** two_p) ** (1.0 / two_p))

    if alpha != 0.0:
        phi_r = alpha * (r ** 2) / (1.0 + (r / beta) ** 2)
    else:
        phi_r = torch.zeros_like(r)

    # Rayleigh PDF for unit complex variance E[|x|^2] = 1: p(r) = 2 * r * exp(-r^2)
    pdf = 2.0 * r * torch.exp(-(r ** 2))

    # E[f(x) x*] = integral_0^inf g(r) * exp(j * phi_r) * r * pdf * dr
    # Phase exp(j * phi_r)
    cos_phi = torch.cos(phi_r)
    sin_phi = torch.sin(phi_r)

    integrand_gamma_real = g_r * cos_phi * r * pdf
    integrand_gamma_imag = g_r * sin_phi * r * pdf
    gamma_real = torch.trapz(integrand_gamma_real, r).item()
    gamma_imag = torch.trapz(integrand_gamma_imag, r).item()
    gamma = complex(gamma_real, gamma_imag)

    # E[|f(x)|^2] = integral_0^inf g(r)^2 * pdf * dr
    integrand_pout = (g_r ** 2) * pdf
    p_out = torch.trapz(integrand_pout, r).item()

    sigma_d2 = max(0.0, p_out - abs(gamma) ** 2)
    return (gamma, sigma_d2, p_out)


# ---------------------------------------------------------------------------
# Channel Estimation Baselines
# ---------------------------------------------------------------------------

def ls_est(yp: torch.Tensor) -> torch.Tensor:
    """Least Squares (LS) channel frequency response estimate from pilot observation."""
    return yp


def delay_est_dft(
    yp: torch.Tensor,
    taps: int = 8,
    n: Optional[int] = None,
) -> torch.Tensor:
    """Truncated DFT delay-domain channel estimate retaining the first `taps` taps."""
    n = yp.shape[-1] if n is None else n
    # Transform pilot observation to delay domain
    h_delay = torch.fft.ifft(yp, dim=-1)
    # Truncate taps beyond delay spread
    h_trunc = F.pad(h_delay[..., :taps], (0, n - taps))
    # Transform back to frequency domain
    return torch.fft.fft(h_trunc, dim=-1)


# ---------------------------------------------------------------------------
# Calibrated 16-QAM Soft LLR Computation
# ---------------------------------------------------------------------------

def calibrated_16qam_llrs(
    z: torch.Tensor,
    sigma2: torch.Tensor,
    method: str = "exact",
) -> torch.Tensor:
    """Compute calibrated posterior log-likelihood ratios for Gray-coded 16-QAM.

    Args:
      z: Complex tensor (..., N) representing conditionally unbiased symbol estimates:
         z[k] = X[k] + v[k], where v[k] ~ CN(0, sigma2[k]).
      sigma2: Real tensor (..., N) or broadcastable representing effective noise-plus-distortion
         variance per complex subcarrier.
      method: 'exact' (numerically stable LogSumExp) or 'maxlog' (piecewise max-log linear).

    Returns:
      LLR tensor of shape (..., N, 4) where:
        llr[..., 0] -> bit b0 (real sign)
        llr[..., 1] -> bit b1 (real magnitude)
        llr[..., 2] -> bit b2 (imag sign)
        llr[..., 3] -> bit b3 (imag magnitude)
        P(b_i = 1 | z) = sigmoid(llr[..., i]).
    """
    pts = PTS_PAM4.to(device=z.device, dtype=z.real.dtype)
    sigma2 = torch.clamp(sigma2, min=1e-8)
    var_real = sigma2 / 2.0  # Variance per real/imag dimension

    if method == "exact":
        # Expand for distance computation against the 4 PAM constellation points
        # dists_real: (..., N, 4)
        z_real_exp = z.real.unsqueeze(-1)
        z_imag_exp = z.imag.unsqueeze(-1)
        var_exp = var_real.unsqueeze(-1)

        dists_real = -((z_real_exp - pts) ** 2) / (2.0 * var_exp)
        dists_imag = -((z_imag_exp - pts) ** 2) / (2.0 * var_exp)

        # Gray mapping for 4-PAM:
        # pts[0] = -3/sqrt(10) -> bits 00
        # pts[1] = -1/sqrt(10) -> bits 01
        # pts[2] = +1/sqrt(10) -> bits 11
        # pts[3] = +3/sqrt(10) -> bits 10

        # Bit 0 (sign of real part): 1 for pts[2], pts[3]; 0 for pts[0], pts[1]
        llr_b0 = torch.logsumexp(dists_real[..., [2, 3]], dim=-1) - torch.logsumexp(
            dists_real[..., [0, 1]], dim=-1
        )
        # Bit 1 (inner magnitude of real part): 1 for pts[1], pts[2]; 0 for pts[0], pts[3]
        llr_b1 = torch.logsumexp(dists_real[..., [1, 2]], dim=-1) - torch.logsumexp(
            dists_real[..., [0, 3]], dim=-1
        )

        # Bit 2 (sign of imag part): 1 for pts[2], pts[3]; 0 for pts[0], pts[1]
        llr_b2 = torch.logsumexp(dists_imag[..., [2, 3]], dim=-1) - torch.logsumexp(
            dists_imag[..., [0, 1]], dim=-1
        )
        # Bit 3 (inner magnitude of imag part): 1 for pts[1], pts[2]; 0 for pts[0], pts[3]
        llr_b3 = torch.logsumexp(dists_imag[..., [1, 2]], dim=-1) - torch.logsumexp(
            dists_imag[..., [0, 3]], dim=-1
        )

    elif method == "maxlog":
        # Calibrated piece-wise max-log approximation
        scale = 4.0 / (SIGMA_16QAM * sigma2)
        boundary = 2.0 / SIGMA_16QAM

        # Real bits
        llr_b1 = scale * (boundary - z.real.abs())
        llr_b0_inner = scale * z.real
        llr_b0_outer = z.real.sign() * (scale * (2.0 * z.real.abs() - boundary))
        llr_b0 = torch.where(z.real.abs() <= boundary, llr_b0_inner, llr_b0_outer)

        # Imag bits
        llr_b3 = scale * (boundary - z.imag.abs())
        llr_b2_inner = scale * z.imag
        llr_b2_outer = z.imag.sign() * (scale * (2.0 * z.imag.abs() - boundary))
        llr_b2 = torch.where(z.imag.abs() <= boundary, llr_b2_inner, llr_b2_outer)

    else:
        raise ValueError(f"Unknown LLR method {method}. Choose from 'exact' or 'maxlog'.")

    return torch.stack((llr_b0, llr_b1, llr_b2, llr_b3), dim=-1)


def hard_slice_16qam(z: torch.Tensor) -> torch.Tensor:
    """Return hard decision bits (..., N, 4) for 16-QAM symbol z."""
    return hard_bits_16qam(z, thresh=THRESH_16QAM)


def soft_slice_16qam(llrs: torch.Tensor) -> torch.Tensor:
    """Reconstruct posterior expected complex symbol E[S | LLR] in C^{..., N}."""
    p0 = torch.sigmoid(llrs[..., 0])
    p1 = torch.sigmoid(llrs[..., 1])
    p2 = torch.sigmoid(llrs[..., 2])
    p3 = torch.sigmoid(llrs[..., 3])

    mean_i = (2.0 * p0 - 1.0) * (3.0 - 2.0 * p1) / SIGMA_16QAM
    mean_q = (2.0 * p2 - 1.0) * (3.0 - 2.0 * p3) / SIGMA_16QAM
    return torch.complex(mean_i, mean_q)


# ---------------------------------------------------------------------------
# Bussgang-LMMSE Equalizer
# ---------------------------------------------------------------------------

@dataclass
class EqualizerResult:
    """Result of LMMSE equalization and soft demapping."""
    h_hat: torch.Tensor  # Estimated physical channel frequency response (B, N)
    h_eff: torch.Tensor  # Effective channel frequency response gamma * h_hat (B, N)
    llrs: torch.Tensor  # Calibrated soft bit LLRs (B, N, 4)
    hard_bits: torch.Tensor  # Hard decision bits (B, N, 4)
    z_hat: torch.Tensor  # Conditionally unbiased equalized symbols (B, N)
    mu: torch.Tensor  # Conditional mean scaling factor (B, N)
    sigma2_eff: torch.Tensor  # Effective residual disturbance variance (B, N)
    w_filter: torch.Tensor  # Frequency-domain MMSE equalization filter weights (B, N)


def bussgang_lmmse(
    yp: torch.Tensor,
    yd: torch.Tensor,
    nv: torch.Tensor,
    A: Optional[float] = 2.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
    est_method: str = "dft8",
    taps: int = 8,
    h_true: Optional[torch.Tensor] = None,
    llr_method: str = "exact",
) -> EqualizerResult:
    """Execute Bussgang-LMMSE equalization accounting for PA attenuation gamma and distortion sigma_D^2."""
    n = yd.shape[-1]

    # Bussgang parameters
    gamma, sigma_d2, _ = bussgang_params(A=A, alpha=alpha, beta=beta, knee=knee)
    gamma_c = torch.tensor(gamma, dtype=torch.complex64, device=yd.device)

    # Channel estimation:
    # If h_true is provided, it is the physical channel H.
    # If estimated from yp (which passed through the transmitter PA and contains gamma * H):
    if h_true is not None:
        h_phys = h_true
        h_eff = gamma_c * h_phys
    elif est_method == "dft8":
        h_raw = delay_est_dft(yp, taps=taps, n=n)
        h_eff = h_raw
        h_phys = h_raw / gamma_c
    elif est_method == "ls":
        h_raw = ls_est(yp)
        h_eff = h_raw
        h_phys = h_raw / gamma_c
    else:
        raise ValueError(f"Unknown estimation method {est_method}. Choose 'dft8' or 'ls'.")

    # Ensure nv has compatible broadcast shape
    if nv.ndim == 1:
        nv_bc = nv.view(-1, 1)
    else:
        nv_bc = nv

    h_pow_eff = h_eff.abs() ** 2
    h_pow_phys = h_phys.abs() ** 2

    # Total disturbance variance: |gamma*H|^2 + |H|^2 * sigma_D^2 + sigma_W^2
    denom = h_pow_eff + h_pow_phys * sigma_d2 + nv_bc + 1e-12

    # MMSE Filter: W[k] = H_eff^*[k] / denom
    w_filter = h_eff.conj() / denom

    # Equalized symbol: X_hat[k] = W[k] * Y_d[k]
    x_hat = w_filter * yd

    # Conditional mean factor: mu[k] = W[k] * H_eff[k] = |H_eff[k]|^2 / denom
    mu = h_pow_eff / denom
    mu = torch.clamp(mu, min=1e-8, max=1.0)

    # Conditionally unbiased observation: Z_hat = X_hat / mu = Y_d / H_eff
    z_hat = x_hat / mu

    # Residual effective variance on Z_hat: sigma2_eff = (1 - mu) / mu
    sigma2_eff = (1.0 - mu) / mu

    # Calibrated LLR computation
    llrs = calibrated_16qam_llrs(z_hat, sigma2_eff, method=llr_method)
    hard_bits = hard_slice_16qam(z_hat)

    return EqualizerResult(
        h_hat=h_phys,
        h_eff=h_eff,
        llrs=llrs,
        hard_bits=hard_bits,
        z_hat=z_hat,
        mu=mu,
        sigma2_eff=sigma2_eff,
        w_filter=w_filter,
    )


# ---------------------------------------------------------------------------
# Iterative Decision-Directed PA Cancellation (DD-PA)
# ---------------------------------------------------------------------------

@dataclass
class DDPAIterationRecord:
    """Record of a single DD-PA iteration."""
    iteration: int
    z_hat: torch.Tensor
    llrs: torch.Tensor
    hard_bits: torch.Tensor
    sigma_d2_rem: float
    dist_est_freq: torch.Tensor


def iterative_dd_pa_cancellation(
    yp: torch.Tensor,
    yd: torch.Tensor,
    nv: torch.Tensor,
    A: Optional[float] = 1.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
    iterations: int = 2,
    est_method: str = "dft8",
    taps: int = 8,
    h_true: Optional[torch.Tensor] = None,
    oversample: int = DEFAULT_OVERSAMPLE,
    llr_method: str = "exact",
    use_soft_decisions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[DDPAIterationRecord]]:
    """Iterative Decision-Directed PA Cancellation receiver (2-3 iterations).

    Steps:
      Iter 0: Initial Bussgang-LMMSE equalization and tentative symbol slicing.
      Iter 1..K:
        1. Transmit symbol reconstruction from decisions (hard or soft).
        2. Oversampled time-domain PA nonlinear distortion estimation.
        3. Low-pass decimation and physical channel filtering: Y_dist = H_phys * D.
        4. Subtraction: Y_cleaned = Y_d - Y_dist.
        5. Re-equalization with updated reduced residual distortion variance.
        6. Re-computation of calibrated soft LLRs and decisions.

    Returns:
      (h_hat, llrs_final, hard_bits_final, history)
    """
    n = yd.shape[-1]
    history: List[DDPAIterationRecord] = []

    # Iteration 0: Initial Bussgang-LMMSE
    res0 = bussgang_lmmse(
        yp=yp,
        yd=yd,
        nv=nv,
        A=A,
        alpha=alpha,
        beta=beta,
        knee=knee,
        est_method=est_method,
        taps=taps,
        h_true=h_true,
        llr_method=llr_method,
    )

    h_phys = res0.h_hat
    h_eff = res0.h_eff
    gamma, sigma_d2_initial, _ = bussgang_params(A=A, alpha=alpha, beta=beta, knee=knee)
    gamma_c = torch.tensor(gamma, dtype=torch.complex64, device=yd.device)

    history.append(
        DDPAIterationRecord(
            iteration=0,
            z_hat=res0.z_hat,
            llrs=res0.llrs,
            hard_bits=res0.hard_bits,
            sigma_d2_rem=sigma_d2_initial,
            dist_est_freq=torch.zeros_like(yd),
        )
    )

    current_llrs = res0.llrs
    current_hard_bits = res0.hard_bits

    if iterations <= 0 or A is None or math.isinf(A):
        return (h_phys, current_llrs, current_hard_bits, history)

    # nv broadcast
    nv_bc = nv.view(-1, 1) if nv.ndim == 1 else nv
    h_pow_eff = h_eff.abs() ** 2
    h_pow_phys = h_phys.abs() ** 2

    # Residual distortion variance decay per iteration (~6 dB reduction per iteration)
    sigma_d2_current = sigma_d2_initial

    for it in range(1, iterations + 1):
        # 1. Reconstruct frequency symbols from prior iteration
        if use_soft_decisions:
            s_hat_freq = soft_slice_16qam(current_llrs)
        else:
            s_hat_freq = q16(current_hard_bits)

        # 2. Oversample to time domain
        s_hat_ov = oversample_frequency_to_time(s_hat_freq, oversample=oversample)

        # 3. Apply PA model to compute total PA output
        s_pa_ov = apply_rapp_pa_samples(s_hat_ov, A=A, alpha=alpha, beta=beta, knee=knee)

        # 4. Nonlinear distortion in oversampled time domain: d = f(x) - gamma * x
        d_ov = s_pa_ov - gamma_c * s_hat_ov

        # 5. Low-pass decimate distortion back to Nyquist frequency domain
        d_hat_freq = decimate_time_to_frequency(d_ov, n=n, oversample=oversample)

        # 6. Filter distortion through physical channel and subtract from received signal
        y_dist = h_phys * d_hat_freq
        y_cleaned = yd - y_dist

        # 7. Update residual distortion variance
        sigma_d2_current = max(1e-7, sigma_d2_current * 0.25)

        # 8. Re-equalization with reduced residual distortion
        denom_it = h_pow_eff + h_pow_phys * sigma_d2_current + nv_bc + 1e-12
        w_it = h_eff.conj() / denom_it
        x_it = w_it * y_cleaned

        mu_it = h_pow_eff / denom_it
        mu_it = torch.clamp(mu_it, min=1e-8, max=1.0)

        z_it = x_it / mu_it
        sigma2_eff_it = (1.0 - mu_it) / mu_it

        current_llrs = calibrated_16qam_llrs(z_it, sigma2_eff_it, method=llr_method)
        current_hard_bits = hard_slice_16qam(z_it)

        history.append(
            DDPAIterationRecord(
                iteration=it,
                z_hat=z_it,
                llrs=current_llrs,
                hard_bits=current_hard_bits,
                sigma_d2_rem=sigma_d2_current,
                dist_est_freq=d_hat_freq,
            )
        )

    return (h_phys, current_llrs, current_hard_bits, history)
