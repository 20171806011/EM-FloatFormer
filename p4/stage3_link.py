"""Frozen neural initialization, waveform cancellation, BICM-ID and native-code list decoding."""
from pathlib import Path

import torch

from p4.classical_baselines import bussgang_params, soft_slice_16qam
from p4.next_classical import _channel, _fit_waveform, _pilot_fit
from p4.next_neural import TapPhaseNoChannel
from p4.simulator import q16, rapp_pa
from p4.slot_2d import SLOT_PROFILES, _get_default_ldpc, decode_min_sum
from p4.stage3_link_decoder import SoftListDecoder


CHECKPOINT = Path(__file__).parent / "results" / "next_neural_tap_phase_no_channel_73161.pt"


def load_neural():
    """The first, prospectively fixed training replicate of the preceding EM study."""
    net = TapPhaseNoChannel().eval()
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["weights"])
    return net


def _mapped(tensor, indices):
    if indices is None:
        return tensor
    return tensor.flatten(0, 1)[indices].reshape_as(tensor)


def _code_part(logical, words, width):
    return logical[:, :words*32].reshape(len(logical), words, width)


def _exchange(channel, prev, words, decoder):
    """Extrinsic check-node evidence in logical code order; filler has no checks."""
    coded = _code_part(channel, words, 128)
    post = decoder.posterior(coded)
    extrinsic = torch.zeros_like(channel)
    candidate = (post - coded).clamp(-20, 20).reshape(len(channel), words*32, 4)
    extrinsic[:, :words*32] = .5*prev[:, :words*32] + .5*candidate
    return post, extrinsic


def _mean_symbol(metric, prior, bits, points):
    score = metric + prior @ bits.T
    return (score.softmax(-1) * points).sum(-1)


def _demap(metric, prior, bits):
    score = metric + prior @ bits.T
    llrs = torch.stack([score[..., bits[:, i] == 1].logsumexp(-1)
                       - score[..., bits[:, i] == 0].logsumexp(-1) for i in range(4)], -1)
    return llrs - prior


@torch.inference_mode()
def receive(y_grid, pilot_tx, pilot_mask, noise_var, *, cp, profile, A, alpha, neural,
            beta=1., knee=2., oversample=4, physical_to_source=None):
    """Return decoded logical info bits and physical soft demapper outputs for two-comb slots.

    Optional physical_to_source permutes the flattened (burst, data RE) array. This CPU
    receiver takes the measured FFT observations, public DMRS, noise and system metadata.
    The caller supplies the frozen neural network. Three extrinsic rounds are fixed.
    """
    if y_grid.device.type != "cpu":
        raise ValueError("CPU code decoder required")
    batch, nsym, nsc = y_grid.shape
    data_mask = ~pilot_mask
    count = int(data_mask.sum())
    words = count // 32
    idx = physical_to_source
    if idx is not None:
        idx = torch.as_tensor(idx)
        if idx.dtype != torch.long or not torch.equal(idx.sort().values, torch.arange(batch*count)):
            raise ValueError("physical_to_source must permute all flattened payload locations")
    inverse = idx.argsort() if idx is not None else None
    code = _get_default_ldpc(128)
    decoder = SoftListDecoder(code)
    observations = y_grid[:,pilot_mask] * pilot_tx.conj()
    init = torch.cat([neural(observations[i:i+2], y_grid[i:i+2], pilot_mask, noise_var[i:i+2])
                      ["llrs"][:,data_mask] for i in range(0,batch,2)]).double()
    logical = _mapped(init, inverse)
    initial_ms = decode_min_sum(_code_part(logical, words, 128).float(), code.h_matrix)
    post, extra = _exchange(logical, torch.zeros_like(logical), words, decoder)
    phys_prior = _mapped(extra, idx)
    estimate = soft_slice_16qam(init + phys_prior)
    y = y_grid.to(torch.complex128)
    noise = noise_var.double().reshape(batch, 1)
    pdp = torch.as_tensor(SLOT_PROFILES[profile], dtype=torch.float64)
    if int(torch.nonzero(pdp>0).max())>cp:
        raise ValueError("Channel support exceeds CP")
    pdp = (pdp / pdp.sum()).clamp_min(1e-9)
    gamma, distortion, _ = bussgang_params(A=A, alpha=alpha, beta=beta, knee=knee)
    _, omega, t = _pilot_fit(y, pilot_tx, pilot_mask, noise, pdp, gamma, distortion, cp)
    mid = t[:,0] + .5/nsym*nsc/(nsc+cp)
    rx = torch.fft.ifft(y, norm="ortho").reshape(batch,-1,1)
    bits = ((torch.arange(16)[:,None] >> torch.arange(4)) & 1).double()
    points = q16(bits.long()).to(torch.complex128)
    for iteration in range(3):
        tx = torch.empty_like(y)
        tx[:, pilot_mask] = pilot_tx.to(y.dtype)
        tx[:, data_mask] = estimate.to(y.dtype)
        waveform = rapp_pa(torch.fft.ifft(tx, norm="ortho"), A=A, alpha=alpha, beta=beta,
                           knee=knee, oversample=oversample)
        basis = torch.stack([waveform.roll(d,-1) for d in range(len(pdp))], -1)
        amp, omega, prediction = _fit_waveform(basis.reshape(batch,-1,len(pdp)), rx,
                                                t.flatten(), omega, noise, pdp)
        predicted = torch.fft.fft(prediction.reshape_as(y), norm="ortho")
        h = _channel(amp,omega,mid,nsc)
        gain = (gamma*h)[:,data_mask]
        residual = y - predicted
        received = residual[:,data_mask] + gain*estimate
        variance = torch.maximum(residual.abs().square().mean((-2,-1)), noise.flatten())
        metric = -(received[:,:,None] - gain[:,:,None]*points).abs().square()/variance[:,None,None]
        phys_llr = _demap(metric, phys_prior, bits)
        logical = _mapped(phys_llr, inverse)
        post, extra = _exchange(logical, extra, words, decoder)
        phys_prior = _mapped(extra, idx)
        if iteration < 2:
            estimate = _mean_symbol(metric, phys_prior, bits, points)
    ordered_metric = _mapped(metric, inverse)[:,:words*32].reshape(batch,words,32,16)
    return {"info_bits": decoder.list_decode(post, ordered_metric), "bp_info_bits": (post[:,:,:64]>0).long(),
            "minsum_info_bits": decode_min_sum(_code_part(logical,words,128).float(),code.h_matrix),
            "init_minsum_info_bits": initial_ms, "llrs": phys_llr.float(), "h_hat": h.to(y_grid.dtype)}
