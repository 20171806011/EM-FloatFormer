"""Pilot and decision-directed delay/Doppler estimation with waveform-domain cancellation."""

import math

import torch

from p4.classical_baselines import bussgang_params, calibrated_16qam_llrs, soft_slice_16qam
from p4.simulator import q16, rapp_pa
from p4.slot_2d import SLOT_PROFILES, _get_default_ldpc, decode_min_sum, encode_ldpc


def _pilot_fit(y, pilots, mask, noise, pdp, gamma, dist_power, cp):
    batch, symbols, n_sc = y.shape
    rows = torch.nonzero(mask.any(-1)).flatten()
    if len(rows) != 2:
        raise ValueError("The two-row delay/Doppler initializer requires two pilot symbols")
    observed = torch.zeros_like(y)
    observed[:, mask] = y[:, mask] * pilots.conj() / gamma
    taps = torch.arange(len(pdp), device=y.device)
    estimates = []
    for row in rows:
        freq = torch.nonzero(mask[row]).flatten()
        if len(freq) < len(pdp):
            raise ValueError("Not enough pilots per row for the declared tap support")
        fourier = torch.exp(-2j * math.pi * freq[:, None].double() * taps / n_sc)
        reg = (noise + dist_power) / abs(gamma) ** 2 / pdp
        matrix = fourier.mH @ fourier + torch.diag_embed(reg)
        target = fourier.mH @ observed[:, row, freq].to(torch.complex128).unsqueeze(-1)
        estimates.append(torch.linalg.solve(matrix, target).squeeze(-1))
    # Frequency is in radians over one slot, amplitude is referenced to its midpoint.
    sample = torch.arange(n_sc, device=y.device, dtype=torch.float64)
    starts = torch.arange(symbols, device=y.device) * (n_sc + cp) + cp
    slot_length = symbols * (n_sc + cp)
    time = (starts[:, None] + sample) / slot_length - 0.5
    pilot_time = (starts[rows] + n_sc / 2) / slot_length - 0.5
    omega = torch.angle(estimates[1] * estimates[0].conj()) / (pilot_time[1] - pilot_time[0])
    amplitude = sum(h * torch.exp(-1j * omega * t) for h, t in zip(estimates, pilot_time)) / 2
    return amplitude, omega, time


def _channel(amplitude, omega, time, n_sc):
    taps_t = amplitude[:, None, :] * torch.exp(1j * time[None, :, None] * omega[:, None, :])
    return torch.fft.fft(taps_t, n=n_sc, dim=-1)


def _code_feedback(llrs, code, physical_to_source):
    """Mix re-encoded hypotheses with the direct soft estimate according to channel disagreement."""
    batch, data_symbols, _ = llrs.shape
    words = data_symbols * 4 // code.n
    estimate = soft_slice_16qam(llrs)
    if words == 0:
        return estimate
    if physical_to_source is not None:
        llrs = llrs.index_select(1, torch.argsort(physical_to_source))
        estimate = soft_slice_16qam(llrs)
    channel = llrs.reshape(batch, -1)[:, :words * code.n].reshape(batch, words, code.n)
    decoded = decode_min_sum(channel, code.h_matrix)
    hypothesis, _ = encode_ldpc(decoded, code)
    # Penalize code hypotheses contradicted by confident received bits.
    cost = (channel.abs() * (hypothesis != (channel > 0))).mean(-1)
    confidence = torch.exp(-cost).unsqueeze(-1)
    qam = q16(hypothesis.reshape(batch, words, -1, 4))
    segment = estimate[:, :words * code.n // 4].reshape_as(qam)
    estimate[:, :words * code.n // 4] = (confidence * qam + (1 - confidence) * segment).flatten(1)
    return estimate if physical_to_source is None else estimate.index_select(1, physical_to_source)


def _fit_waveform(basis, rx, time, omega, noise, pdp):
    """Fit all tap amplitudes and update their phase slopes via projected Gauss--Newton."""
    regularization = torch.diag_embed(noise / pdp)
    for iteration in range(3):
        a = basis * torch.exp(1j * time[:, None] * omega[:, None, :])
        gram = a.mH @ a + regularization
        amplitude = torch.linalg.solve(gram, a.mH @ rx)
        fitted = a @ amplitude
        if iteration == 2:
            break
        jacobian = 1j * time[:, None] * a * amplitude.mT
        projected = jacobian - a @ torch.linalg.solve(gram, a.mH @ jacobian)
        # One radian of posterior slope uncertainty supplies a noise-scaled damping
        # floor for unobservable weak taps. Cap steps at one half-turn per slot.
        ridge = torch.diag_embed(noise.expand_as(omega))
        update = torch.linalg.solve((projected.mH @ projected).real + ridge,
                                    (projected.mH @ (rx - fitted)).real).squeeze(-1)
        omega = omega + update.clamp(-math.pi, math.pi)
    return amplitude.squeeze(-1), omega, fitted.squeeze(-1)


@torch.no_grad()
def delay_doppler_receive(y_grid, pilot_tx, pilot_mask, noise_var, *, cp, profile, A, alpha,
                          beta=1.0, knee=2.0, oversample=4, rounds=3, code_aided=True,
                          physical_to_source=None):
    """Receive a known rate-1/2 LDPC slot from its FFT grid, known DMRS and declared metadata.

    Each tap has an unknown complex amplitude and an unknown constant angular rate.
    No realized Doppler, channel gains, transmitted payload, or simulator labels are inputs.
    Tap-support/power priors and the transmitter PA law are known to the receiver.
    An optional symbol permutation satisfies physical_symbols = logical_symbols[:, physical_to_source].
    Returned LLRs are always in physical order.
    """
    batch, symbols, n_sc = y_grid.shape
    y = y_grid.to(torch.complex128)
    if physical_to_source is not None:
        physical_to_source = torch.as_tensor(physical_to_source, device=y.device)
        expected = torch.arange(int((~pilot_mask).sum()), device=y.device)
        if physical_to_source.dtype != torch.long or not torch.equal(physical_to_source.sort().values, expected):
            raise ValueError("physical_to_source must permute the payload symbol indices")
    noise = noise_var.to(torch.float64).reshape(batch, 1)
    pdp = torch.as_tensor(SLOT_PROFILES[profile], device=y.device, dtype=torch.float64)
    active = torch.nonzero(pdp > 0).flatten()
    # Zero prior taps are structural; a tiny ridge variance leaves their fitted amplitude at zero.
    pdp = (pdp / pdp.sum()).clamp_min(1e-9)
    if int(active.max()) > cp:
        raise ValueError("Tap support exceeds cyclic prefix")
    gamma, dist_power, _ = bussgang_params(A=A, alpha=alpha, beta=beta, knee=knee)
    amplitude, omega, time_grid = _pilot_fit(y, pilot_tx, pilot_mask, noise, pdp, gamma, dist_power, cp)
    time_mid = time_grid[:, 0] + 0.5 / symbols * n_sc / (n_sc + cp)
    h = _channel(amplitude, omega, time_mid, n_sc)
    effective = h * gamma
    variance = noise.unsqueeze(2) + h.abs().square() * dist_power
    received = y
    rx = torch.fft.ifft(y, norm="ortho").reshape(batch, -1, 1)
    code = _get_default_ldpc(128) if code_aided else None
    for iteration in range(rounds + 1):
        power = effective.abs().square().clamp_min(1e-12)
        eq = received * effective.conj() / power
        llrs = calibrated_16qam_llrs(eq, variance / power)[:, ~pilot_mask, :].float()
        if iteration == rounds:
            break
        tx = torch.empty_like(y)
        tx[:, pilot_mask] = pilot_tx.to(tx.dtype)
        tx[:, ~pilot_mask] = (_code_feedback(llrs, code, physical_to_source) if code_aided
                              else soft_slice_16qam(llrs)).to(tx.dtype)
        time_signal = torch.fft.ifft(tx, norm="ortho")
        nonlinear = rapp_pa(time_signal, A=A, alpha=alpha, beta=beta, knee=knee, oversample=oversample)
        basis = torch.stack([nonlinear.roll(delay, -1) for delay in range(len(pdp))], dim=-1)
        amplitude, omega, predicted = _fit_waveform(basis.reshape(batch, -1, len(pdp)), rx,
                                                    time_grid.flatten(), omega, noise, pdp)
        pred_grid = torch.fft.fft(predicted.reshape_as(y), norm="ortho")
        h = _channel(amplitude, omega, time_mid, n_sc)
        effective = h * gamma
        received = y - (pred_grid - effective * tx)
        # The residual includes hard/soft decision error as well as noise. It supplies
        # a conservative per-slot demapper temperature, floored at the physical variance.
        residual_power = (y - pred_grid).abs().square().mean(dim=(-2, -1))
        variance = torch.maximum(noise.squeeze(-1), residual_power).reshape(batch, 1, 1)
    return {"llrs": llrs, "h_hat": h.to(y_grid.dtype)}
