"""Dynamic Channel Simulation & Oversampled Rapp PA Model.

Supports:
- 16-QAM and QPSK constellation mappings (Gray coded, unit energy).
- Oversampled Rapp PA model with AM-AM ($p=2, A \\in \\{1.25, 2.25\\}$) and AM-PM ($\\Phi(r) = \\frac{\\alpha r^2}{1 + (r/\\beta)^2}$) distortion.
- Time-domain dynamic multipath channel simulator with cyclic prefix (CP), continuous Doppler ICI phase evolution, and FIR convolution.
- Static-limit parity check verifying exact equivalence with $Y = H X + W$ when $f_d=0$ and PA is linear ($A \\to \\infty$).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Standard numerology
DEFAULT_N = 32
DEFAULT_CP = 16
DEFAULT_OVERSAMPLE = 4

# Constellation constants
SIGMA_16QAM = math.sqrt(10.0)
THRESH_16QAM = 2.0 / SIGMA_16QAM
GRAY_TABLE_16QAM = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=torch.float32)

# Standard delay spread profiles
PROFILES: Dict[str, Tuple[float, ...]] = {
    "train3": (0.72, 0.20, 0.08),
    "equal4": (0.25, 0.25, 0.25, 0.25),
    "long6": (0.40, 0.22, 0.15, 0.10, 0.08, 0.05),
    "long12": (0.35, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.025, 0.02),
    "sparse12": (0.52, 0.0, 0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18),
}


def complex_randn(
    shape: Tuple[int, ...],
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Generate circularly symmetric complex Gaussian noise CN(0, I)."""
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    r = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    i = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    return torch.complex(r, i) / math.sqrt(2.0)


def q16(bits: torch.Tensor) -> torch.Tensor:
    """Gray-coded 16-QAM symbol mapping normalized to unit average power E_s = 1.

    Bits format: [..., 4] where:
      b0: sign of real part (1 -> >0, 0 -> <0)
      b1: magnitude of real part (1 -> inner, 0 -> outer)
      b2: sign of imag part (1 -> >0, 0 -> <0)
      b3: magnitude of imag part (1 -> inner, 0 -> outer)
    """
    table = GRAY_TABLE_16QAM.to(device=bits.device)
    idx_i = bits[..., 0] * 2 + bits[..., 1]
    idx_q = bits[..., 2] * 2 + bits[..., 3]
    real = table[idx_i]
    imag = table[idx_q]
    return torch.complex(real, imag) / SIGMA_16QAM


def qpsk(bits: torch.Tensor) -> torch.Tensor:
    """Gray-coded QPSK symbol mapping normalized to unit average power E_s = 1."""
    real = (2.0 * bits[..., 0].float() - 1.0) / math.sqrt(2.0)
    imag = (2.0 * bits[..., 1].float() - 1.0) / math.sqrt(2.0)
    return torch.complex(real, imag)


def hard_bits_16qam(z: torch.Tensor, thresh: float = THRESH_16QAM) -> torch.Tensor:
    """Hard slicing for Gray-coded 16-QAM symbols. Returns (..., 4) int64 tensor."""
    b0 = (z.real > 0.0).long()
    b1 = (z.real.abs() < thresh).long()
    b2 = (z.imag > 0.0).long()
    b3 = (z.imag.abs() < thresh).long()
    return torch.stack((b0, b1, b2, b3), dim=-1)


# ---------------------------------------------------------------------------
# Rapp Power Amplifier Model with AM-AM and AM-PM
# ---------------------------------------------------------------------------

def rapp_am_am(
    r: torch.Tensor,
    A: Optional[float] = 2.25,
    knee: float = 2.0,
) -> torch.Tensor:
    """Rapp AM-AM amplitude transfer function g(r) = r / (1 + (r/A)^(2p))^(1/(2p))."""
    if A is None or math.isinf(A):
        return r
    two_p = 2.0 * knee
    inv_two_p = 1.0 / two_p
    denominator = (1.0 + (r / A) ** two_p) ** inv_two_p
    return r / denominator


def rapp_am_pm(
    r: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> torch.Tensor:
    """Rapp / Saleh-extended AM-PM phase distortion function Phi(r) = alpha * r^2 / (1 + (r/beta)^2)."""
    if alpha == 0.0:
        return torch.zeros_like(r)
    return alpha * (r ** 2) / (1.0 + (r / beta) ** 2)


def apply_rapp_pa_samples(
    x_time: torch.Tensor,
    A: Optional[float] = 2.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
) -> torch.Tensor:
    """Apply memoryless AM-AM and AM-PM distortion directly to complex time samples."""
    if (A is None or math.isinf(A)) and alpha == 0.0:
        return x_time
    r = x_time.abs()
    g_r = rapp_am_am(r, A=A, knee=knee)
    phi_r = rapp_am_pm(r, alpha=alpha, beta=beta)
    # Use torch.polar for high numerical accuracy across full dynamic range
    theta = torch.angle(x_time) + phi_r
    return torch.polar(g_r, theta)


def oversample_frequency_to_time(
    x_freq: torch.Tensor,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> torch.Tensor:
    """Zero-pad in frequency domain to oversampling factor L >= 1 and transform via IFFT.

    Preserves exact continuous-time sample values at Nyquist sampling instants:
    x_ov[l * L] == x_nyq[l] for all l.
    """
    if oversample <= 1:
        return torch.fft.ifft(x_freq, norm="ortho")
    n = x_freq.shape[-1]
    n_ov = oversample * n
    scale = math.sqrt(oversample)

    # Pad in standard DFT bin ordering: positive frequencies at start, negative at end
    x_ov_freq = torch.zeros(
        (*x_freq.shape[:-1], n_ov),
        dtype=x_freq.dtype,
        device=x_freq.device,
    )
    n_half = n // 2
    x_ov_freq[..., :n_half] = x_freq[..., :n_half] * scale
    x_ov_freq[..., n_ov - (n - n_half) :] = x_freq[..., n_half:] * scale

    return torch.fft.ifft(x_ov_freq, norm="ortho")


def decimate_time_to_frequency(
    x_ov_time: torch.Tensor,
    n: int = DEFAULT_N,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> torch.Tensor:
    """Transform oversampled time signal via FFT and extract central in-band subcarriers."""
    if oversample <= 1:
        return torch.fft.fft(x_ov_time, norm="ortho")
    n_ov = x_ov_time.shape[-1]
    scale = math.sqrt(oversample)
    x_ov_freq = torch.fft.fft(x_ov_time, norm="ortho")

    n_half = n // 2
    x_freq = torch.zeros(
        (*x_ov_time.shape[:-1], n),
        dtype=x_ov_time.dtype,
        device=x_ov_time.device,
    )
    x_freq[..., :n_half] = x_ov_freq[..., :n_half] / scale
    x_freq[..., n_half:] = x_ov_freq[..., n_ov - (n - n_half) :] / scale
    return x_freq


def rapp_pa(
    x_time: torch.Tensor,
    knee: float = 2.0,
    A: Optional[float] = 2.25,
    alpha: float = 0.5,
    beta: float = 1.0,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> torch.Tensor:
    """Oversampled Rapp PA model operating on Nyquist-rate time-domain symbols.

    Matches PROJECT.md interface contract:
    rapp_pa(x_time, knee=2.0, A=2.25, alpha=0.5, beta=1.0, oversample=4) -> y_time
    """
    if (A is None or math.isinf(A)) and alpha == 0.0:
        return x_time

    n = x_time.shape[-1]
    # Transform to frequency domain
    x_freq = torch.fft.fft(x_time, norm="ortho")
    # Oversample
    x_ov = oversample_frequency_to_time(x_freq, oversample=oversample)
    # Apply nonlinear distortion
    x_pa_ov = apply_rapp_pa_samples(x_ov, A=A, alpha=alpha, beta=beta, knee=knee)
    # Decimate in-band
    x_pa_freq = decimate_time_to_frequency(x_pa_ov, n=n, oversample=oversample)
    # Return to Nyquist time domain
    return torch.fft.ifft(x_pa_freq, norm="ortho")


# ---------------------------------------------------------------------------
# Dynamic Multipath Time-Domain Channel Simulator
# ---------------------------------------------------------------------------

@dataclass
class ChannelState:
    """Container for multipath channel state."""
    h_time_initial: torch.Tensor  # (B, L_taps) initial tap complex gains
    delays: torch.Tensor  # (L_taps,) integer sample delays
    fd_shifts: torch.Tensor  # (B, L_taps) normalized Doppler shifts per tap
    h_freq_static: torch.Tensor  # (B, N) static channel frequency response


class DynamicChannelSimulator:
    """Continuous time-domain multipath channel simulator with Doppler phase evolution."""

    def __init__(
        self,
        n: int = DEFAULT_N,
        cp: int = DEFAULT_CP,
        profile: str = "equal4",
        fd: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ):
        self.n = n
        self.cp = cp
        self.total_len = n + cp
        self.profile_name = profile
        self.fd = fd
        self.device = device

        if profile not in PROFILES:
            raise ValueError(f"Unknown channel profile {profile}. Choose from {list(PROFILES.keys())}")
        self.pdp = torch.tensor(PROFILES[profile], dtype=torch.float32, device=device)
        self.pdp = self.pdp / self.pdp.sum()
        self.num_taps = len(self.pdp)
        self.delays = torch.arange(self.num_taps, device=device)

        if self.delays.max().item() > self.cp:
            raise ValueError(f"Max channel delay {self.delays.max().item()} exceeds CP length {self.cp}")

    def draw_channel(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> ChannelState:
        """Draw fresh initial complex Gaussian taps and Doppler shifts."""
        # Initial tap gains: CN(0, p_l)
        h_init = complex_randn(
            (batch_size, self.num_taps),
            generator=generator,
            device=self.device,
        ) * self.pdp.sqrt().unsqueeze(0)

        # Doppler shifts per tap: fd_l = fd * cos(theta_l) (Jakes 2D isotropic scattering)
        if self.fd > 0.0:
            angles = torch.rand(
                (batch_size, self.num_taps),
                generator=generator,
                device=self.device,
            ) * (2.0 * math.pi)
            fd_shifts = self.fd * torch.cos(angles)
        else:
            fd_shifts = torch.zeros(
                (batch_size, self.num_taps),
                dtype=torch.float32,
                device=self.device,
            )

        # Static channel frequency response (at sample 0)
        h_freq_static = torch.fft.fft(h_init, n=self.n, dim=-1)

        return ChannelState(
            h_time_initial=h_init,
            delays=self.delays,
            fd_shifts=fd_shifts,
            h_freq_static=h_freq_static,
        )

    def filter_time_signal(
        self,
        x_time_cp: torch.Tensor,
        state: ChannelState,
        noise_var: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Execute continuous time-domain FIR convolution with Doppler phase drift:

        y_time[n] = sum_l h_l[n] x_time[n - d_l] + w[n]
        where h_l[n] = h_{l, 0} * exp(j * 2 * pi * fd_l * n / N)
        """
        batch_size = x_time_cp.shape[0]
        total_len = x_time_cp.shape[1]
        assert total_len == self.total_len, f"Expected signal length {self.total_len}, got {total_len}"

        y_time = torch.zeros(
            (batch_size, total_len),
            dtype=x_time_cp.dtype,
            device=x_time_cp.device,
        )

        # Time sample indices: n in [0, total_len - 1]
        n_grid = torch.arange(total_len, dtype=torch.float32, device=self.device).view(1, -1)

        for l in range(self.num_taps):
            delay = int(state.delays[l].item())
            h0_l = state.h_time_initial[:, l : l + 1]  # (B, 1)
            fd_l = state.fd_shifts[:, l : l + 1]  # (B, 1)

            # Continuous Doppler phase: exp(j * 2*pi * fd_l * n / N)
            phase = 2.0 * math.pi * fd_l * (n_grid / float(self.n))
            h_l_n = h0_l * torch.complex(torch.cos(phase), torch.sin(phase))  # (B, total_len)

            # Causal delay: x_time_cp[..., : total_len - delay]
            if delay == 0:
                y_time += h_l_n * x_time_cp
            else:
                y_time[:, delay:] += h_l_n[:, delay:] * x_time_cp[:, : total_len - delay]

        # Add additive complex Gaussian noise if specified
        if noise_var is not None:
            # noise_var has shape (B, 1) or scalar
            if noise_var.ndim == 1:
                noise_var = noise_var.unsqueeze(-1)
            noise = complex_randn(
                y_time.shape,
                generator=generator,
                device=self.device,
                dtype=y_time.dtype,
            ) * noise_var.sqrt()
            y_time += noise

        return y_time

    def compute_transfer_matrix(self, state: ChannelState) -> torch.Tensor:
        """Compute the exact frequency-domain transfer matrix H_mat of shape (B, N, N)

        such that Y = H_mat @ X in the absence of noise.
        When fd = 0, H_mat is strictly diagonal with diag(H_mat) == state.h_freq_static.
        """
        batch_size = state.h_time_initial.shape[0]
        n = self.n
        cp = self.cp
        m_grid = torch.arange(n, dtype=torch.float32, device=self.device)  # (N,)

        # H[k1, k2] = (1/N) * sum_{m=0}^{N-1} sum_l h_l[m + CP] * exp(-j*2*pi*k2*d_l/N) * exp(j*2*pi*(k2 - k1)*m/N)
        h_mat = torch.zeros((batch_size, n, n), dtype=torch.complex64, device=self.device)

        for l in range(self.num_taps):
            delay = int(state.delays[l].item())
            h0_l = state.h_time_initial[:, l : l + 1, None]  # (B, 1, 1)
            fd_l = state.fd_shifts[:, l : l + 1, None]  # (B, 1, 1)

            # Sample times during payload window: m + CP
            sample_times = (m_grid + cp).view(1, 1, -1)  # (1, 1, N)
            phase_doppler = 2.0 * math.pi * fd_l * (sample_times / float(n))
            h_l_m = h0_l * torch.complex(torch.cos(phase_doppler), torch.sin(phase_doppler))  # (B, 1, N)

            for k1 in range(n):
                for k2 in range(n):
                    phase_delay = -2.0 * math.pi * k2 * delay / float(n)
                    phase_ici = 2.0 * math.pi * (k2 - k1) * m_grid / float(n)
                    total_phase = phase_delay + phase_ici.view(1, 1, -1)
                    factor = torch.complex(torch.cos(total_phase), torch.sin(total_phase))
                    h_mat[:, k1, k2] += (h_l_m * factor).mean(dim=-1).squeeze(1)

        return h_mat


# ---------------------------------------------------------------------------
# High-Level Frame Generation and Parity Verification
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Full frame container for receiver baselines and learning architectures."""
    yp: torch.Tensor  # (B, N) received pilot observation (LS channel observation)
    yd: torch.Tensor  # (B, N) received data symbol FFT
    bits: torch.Tensor  # (B, N, 4) ground truth 16-QAM data bits
    h_true: torch.Tensor  # (B, N) ground truth channel frequency response
    noise_var: torch.Tensor  # (B, 1) total noise variance per complex subcarrier
    pilot_tx: torch.Tensor  # (B, N) transmitted pilot symbols
    data_tx: torch.Tensor  # (B, N) transmitted data symbols
    h_matrix: Optional[torch.Tensor] = None  # (B, N, N) transfer matrix under Doppler


def generate_frame(
    batch_size: int,
    n: int = DEFAULT_N,
    cp: int = DEFAULT_CP,
    profile: str = "equal4",
    snr_db: Union[float, Tuple[float, float]] = 10.0,
    fd: float = 0.0,
    pa_condition: str = "mild",  # 'clean', 'mild', 'strong'
    A: Optional[float] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    knee: float = 2.0,
    oversample: int = DEFAULT_OVERSAMPLE,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> FrameData:
    """Generate a complete OFDM pilot + 16-QAM data transmission through the dynamic link."""
    # Determine PA saturation parameter A
    if pa_condition == "clean":
        A_eff = None
        alpha_eff = 0.0
    elif pa_condition == "mild":
        A_eff = 2.25 if A is None else A
        alpha_eff = alpha
    elif pa_condition == "strong":
        A_eff = 1.25 if A is None else A
        alpha_eff = alpha
    else:
        A_eff = A
        alpha_eff = alpha

    # Channel simulator
    sim = DynamicChannelSimulator(n=n, cp=cp, profile=profile, fd=fd, device=device)
    ch_state = sim.draw_channel(batch_size, generator=generator)

    # Generate pilot: QPSK on all subcarriers
    pilot_bits = torch.randint(0, 2, (batch_size, n, 2), generator=generator, device=device)
    pilot_tx = qpsk(pilot_bits)

    # Generate data: 16-QAM on all subcarriers
    data_bits = torch.randint(0, 2, (batch_size, n, 4), generator=generator, device=device)
    data_tx = q16(data_bits)

    # Noise variance
    if isinstance(snr_db, (tuple, list)):
        snr_vals = torch.empty((batch_size, 1), device=device).uniform_(snr_db[0], snr_db[1], generator=generator)
    else:
        snr_vals = torch.full((batch_size, 1), float(snr_db), device=device)
    noise_var = 10.0 ** (-snr_vals / 10.0)

    # Transmitter: IFFT to Nyquist time domain
    p_time = torch.fft.ifft(pilot_tx, norm="ortho")
    d_time = torch.fft.ifft(data_tx, norm="ortho")

    # Transmitter PA with oversampling
    p_time_pa = rapp_pa(p_time, knee=knee, A=A_eff, alpha=alpha_eff, beta=beta, oversample=oversample)
    d_time_pa = rapp_pa(d_time, knee=knee, A=A_eff, alpha=alpha_eff, beta=beta, oversample=oversample)

    # Add Cyclic Prefix
    p_time_cp = torch.cat((p_time_pa[:, -cp:], p_time_pa), dim=-1)
    d_time_cp = torch.cat((d_time_pa[:, -cp:], d_time_pa), dim=-1)

    # Channel filtering
    yp_time = sim.filter_time_signal(p_time_cp, ch_state, noise_var=noise_var, generator=generator)
    yd_time = sim.filter_time_signal(d_time_cp, ch_state, noise_var=noise_var, generator=generator)

    # Receiver: discard CP and take FFT
    yp_rx = torch.fft.fft(yp_time[:, cp:], norm="ortho")
    yd_rx = torch.fft.fft(yd_time[:, cp:], norm="ortho")

    # Pilot channel observation Y_p / X_p (for QPSK, 1/X = X*)
    yp_obs = yp_rx * pilot_tx.conj()

    h_matrix = sim.compute_transfer_matrix(ch_state) if fd > 0.0 else None

    return FrameData(
        yp=yp_obs,
        yd=yd_rx,
        bits=data_bits,
        h_true=ch_state.h_freq_static,
        noise_var=noise_var,
        pilot_tx=pilot_tx,
        data_tx=data_tx,
        h_matrix=h_matrix,
    )


def static_limit_parity_check(
    n: int = DEFAULT_N,
    cp: int = DEFAULT_CP,
    batch_size: int = 128,
    profile: str = "equal4",
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """Verify exact parity with static frequency-domain matrix model Y = H * X + W.

    Returns the maximum relative error ||Y_time - Y_static|| / (||Y_static|| + eps).
    Must strictly satisfy relative error < 1e-5.
    """
    sim = DynamicChannelSimulator(n=n, cp=cp, profile=profile, fd=0.0, device=device)
    ch_state = sim.draw_channel(batch_size, generator=generator)

    # Transmit random unit-power frequency symbols
    data_bits = torch.randint(0, 2, (batch_size, n, 4), generator=generator, device=device)
    x_freq = q16(data_bits)

    # Linear PA (A = infinity, alpha = 0.0)
    x_time = torch.fft.ifft(x_freq, norm="ortho")
    x_time_pa = rapp_pa(x_time, knee=2.0, A=None, alpha=0.0, oversample=DEFAULT_OVERSAMPLE)

    # CP insertion
    x_time_cp = torch.cat((x_time_pa[:, -cp:], x_time_pa), dim=-1)

    # Dynamic time filter (noise-free for pure parity audit)
    y_time = sim.filter_time_signal(x_time_cp, ch_state, noise_var=None, generator=generator)
    y_time_rx = y_time[:, cp:]
    y_sim_freq = torch.fft.fft(y_time_rx, norm="ortho")

    # Static frequency-domain reference: Y = H * X
    y_static_freq = ch_state.h_freq_static * x_freq

    # Tensor Frobenius-norm relative error: ||Y_sim - Y_static||_F / ||Y_static||_F
    rel_err = ((y_sim_freq - y_static_freq).norm() / (y_static_freq.norm() + 1e-12)).item()

    return rel_err
