"""Uncoded DFT-8 comparison on integer and off-grid physical delays, P4-B S07."""
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from p4.classical_baselines import bussgang_params, calibrated_16qam_llrs
from p4.next_neural import ARMS, digest, pilot_control
from p4.refit_pilot_prior import per_slot_errors
from p4.simulator import complex_randn, rapp_pa
from p4.slot_2d import (SLOT_PROFILES, SlotSimulator2D, _apply_continuous_slot_channel,
                        _get_default_ldpc, _synchronize_slot)


FOLDER = Path(__file__).resolve().parent / "results"
SEEDS = (73161, 73162, 73163)
S, N, CP = 14, 32, 16
FD, CFO, SNR = .01, 0., 20.
GEOMETRIES = (("equal4_integer", "equal4", False), ("exp16_integer", "exp16", False),
              ("exp16_fractional", "exp16", True))


def delay_geometry(n_taps, fractional):
    delays = torch.arange(n_taps, dtype=torch.float64)
    if fractional:
        delays[1:] += ((torch.arange(n_taps - 1) % 7) + 1) / 8
    return delays


def cis(angle):
    return torch.polar(torch.ones_like(angle), angle)


def transmit(template):
    waves = torch.fft.ifft(template.x_grid, dim=-1, norm="ortho")
    return rapp_pa(waves, A=template.A, alpha=template.alpha, beta=template.beta,
                   knee=template.knee, oversample=template.oversample)


def propagate(waves, gains, doppler, delays, cfo, noise):
    """Delay the CP-extended piecewise periodic, post-PA waveform in continuous time.

    Within each symbol its N samples define the signed-frequency baseband Fourier
    interpolant. Select the actual emitting symbol at reception time n-delay, so
    leading CP samples may receive previous-symbol energy. Pre-slot emission is zero.
    """
    batch, n_symb, n_sc = waves.shape
    length = n_symb * (n_sc + CP)
    times = torch.arange(length, dtype=torch.float64)
    freqs = torch.fft.fftfreq(n_sc, dtype=torch.float64)
    spectrum = torch.fft.fft(waves.to(torch.complex128), dim=-1)
    received = torch.zeros((batch, length), dtype=torch.complex128)
    for tap, delay in enumerate(delays.tolist()):
        integer = math.ceil(delay)
        advance = integer - delay
        interpolated = torch.fft.ifft(spectrum * cis(2 * math.pi * advance * freqs), dim=-1)
        stream = torch.cat((interpolated[..., -CP:], interpolated), dim=-1).reshape(batch, length)
        rotation = cis(2 * math.pi * doppler[:, tap:tap + 1].double() * times / n_sc)
        received[:, integer:] += (gains[:, tap:tap + 1] * rotation[:, integer:]
                                  * stream[:, :length - integer])
    received *= cis(2 * math.pi * cfo * times / n_sc)
    mid = torch.arange(n_symb, dtype=torch.float64) * (n_sc + CP) + CP + n_sc / 2
    h_t = gains[:, None, :] * cis(2 * math.pi * doppler[:, None, :].double() * mid[:, None] / n_sc)
    h_mid = h_t @ cis(-2 * math.pi * delays[:, None] * freqs)
    return (received + noise).to(torch.complex64), h_mid.to(torch.complex64), h_t.to(torch.complex64)


def make_slot(template, waves, gains, doppler, delays, noise, profile):
    y_time, h, h_t = propagate(waves, gains, doppler, delays, CFO, noise)
    useful = y_time.reshape(len(waves), S, N + CP)[:, :, CP:]
    y = torch.fft.fft(useful, dim=-1, norm="ortho")
    yp = y[:, template.pilot_mask]
    return _synchronize_slot(replace(template, rx_time=y_time, y_grid=y, h_true=h,
        h_time_evol=h_t, fd=FD, cfo=CFO, profile=profile, y_p_raw=yp,
        yp_sparse=yp * template.pilot_tx.conj(), y_d_raw=y[:, template.data_mask]))


def parity():
    gen = torch.Generator().manual_seed(846099)
    waves = complex_randn((2, S, N), generator=gen)
    gains = complex_randn((2, 16), generator=gen)
    doppler = .01 * torch.randn((2, 16), generator=gen)
    delays = delay_geometry(16, False)
    stream = torch.cat((waves[..., -CP:], waves), dim=-1).reshape(2, -1)
    nv = torch.full((2, 1), .01)
    noise = complex_randn(stream.shape, generator=torch.Generator().manual_seed(846098),
                          dtype=torch.complex128) * nv.double().sqrt()
    ref = _apply_continuous_slot_channel(stream, gains, delays.long(), doppler, S, N, CP,
        .02, nv, generator=torch.Generator().manual_seed(846098))
    got = propagate(waves, gains, doppler, delays, .02, noise)
    diffs = [(a-b).abs().max().item() for a,b in zip(ref,got)]
    # Independently evaluate H[k] on every tone for one fractional path.
    delay = torch.tensor([2.375], dtype=torch.float64)
    y, h, _ = propagate(waves, torch.ones((2,1), dtype=torch.complex128),
                        torch.zeros((2,1)), delay, 0., torch.zeros_like(noise))
    analytic = cis(-2 * math.pi * torch.fft.fftfreq(N, dtype=torch.float64) * delay.item())
    spectrum = torch.fft.fft(y.reshape(2,S,N+CP)[:,:,CP:].to(torch.complex128), dim=-1)
    expected = torch.fft.fft(waves.to(torch.complex128), dim=-1) * analytic
    static = (spectrum-expected).abs().max().item()
    cfr_error = (h - analytic).abs().max().item()
    # Separate direct continuous-time synthesis includes the complete prefix,
    # the zero-before-slot boundary and the preceding emitting symbol.
    d = torch.tensor([.375, 3.8, 15.125], dtype=torch.float64)
    short = waves[:1,:3]
    g, fd = gains[:1,:3], doppler[:1,:3]
    actual, _, _ = propagate(short, g, fd, d, .02, torch.zeros_like(noise[:1,:3*(N+CP)]))
    spec = torch.fft.fft(short[0].to(torch.complex128), dim=-1, norm="ortho")
    freq = torch.fft.fftfreq(N, dtype=torch.float64)
    direct = torch.zeros_like(actual[0], dtype=torch.complex128)
    for i, tau in enumerate(d):
        for n in range(len(direct)):
            src = n - tau.item()
            if src < 0:
                continue
            s = int(src // (N + CP))
            x = (spec[s] * cis(2*math.pi*freq*(src - s*(N+CP) - CP))).sum() / math.sqrt(N)
            direct[n] += x*g[0,i]*cis(2*math.pi*(fd[0,i].double()+.02)*n/N)
    prefix_error = (direct-actual[0]).abs().max().item()
    slot = SlotSimulator2D(fd=FD).generate_slot(batch_size=2,snr_db=SNR,generator=gen)
    poison = replace(slot,h_true=torch.full_like(slot.h_true,complex(float("nan"),0)),
        h_time_evol=torch.full_like(slot.h_time_evol,complex(float("nan"),0)),cfo=float("nan"),
        bits=None,data_tx=None,info_bits=None,x_grid=None,profile="unavailable")
    clean, masked = _synchronize_slot(slot), _synchronize_slot(poison)
    no_leak = all(torch.equal(getattr(clean,k),getattr(masked,k))
                  for k in ("yp_sparse","y_grid","pilot_mask","noise_var"))
    if not no_leak or max(*diffs, static, cfr_error, prefix_error) > 3e-6:
        raise AssertionError(f"Fractional channel parity failed: {diffs}, {static}, {cfr_error}")
    return {"integer_time_cfr_taps_max":diffs, "fractional_static_spectrum_max":static,
            "fractional_static_cfr_max":cfr_error, "fractional_prefix_direct_max":prefix_error,
            "synchronizer_observations_exact_with_poisoned_truth":no_leak}


def dft8(yp, y, nv, phase_progression):
    """Estimate 8 taps on each identical comb, and interpolate/extrapolate to all rows."""
    a, b = torch.fft.ifft(yp.to(torch.complex128).reshape(len(yp), 2, N//2), dim=-1)[...,:8].unbind(1)
    t = ((torch.arange(S,dtype=torch.float64) - 2) / 9)[None, :, None]
    if phase_progression:
        turn = (b * a.conj()).angle()
        taps = .5 * (a + b * cis(-turn))[:,None,:] * cis(t * turn[:,None,:])
    else:
        w = t.clamp(0,1)
        taps = (1-w)*a[:,None,:] + w*b[:,None,:]
    eff = torch.fft.fft(taps, n=N, dim=-1)
    gamma, sd2, _ = bussgang_params(A=2.25, alpha=.5)
    # Known fixed PA law for every condition. Pilot LS estimates the effective channel.
    hp = eff.abs().square().clamp_min(1e-10)
    z = y * eff.conj() / hp
    var = nv.reshape(len(y),1,1) / hp + sd2 / abs(gamma)**2
    return calibrated_16qam_llrs(z, var, method="exact"), eff / gamma


def load_models():
    out = {"tap_phase_pilot_only": pilot_control(ARMS["tap_phase_no_channel"])}
    hashes = {}
    for name in ("tap_phase_no_channel", "fourier_no_channel"):
        for seed in SEEDS:
            path = FOLDER / f"next_neural_{name}_{seed}.pt"
            checkpoint = torch.load(path, weights_only=True)
            model = ARMS[name]().eval()
            model.load_state_dict(checkpoint["weights"])
            out[f"{name}_{seed}"] = model
            hashes[path.name] = digest(path)
    return out, hashes


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    checks = parity()
    models, hashes = load_models()
    methods = ("dft8_linear", "dft8_tap_phase", *models)
    sim = SlotSimulator2D(n_symb=S, n_sc=N, cp=CP, profile="equal4", fd=FD, cfo=CFO)
    code = _get_default_ldpc(128)
    rows = []
    for coded in (False, True):
        for batch_i in range(8):
            # Geometry panel shares independent streams of payload, latent taps,
            # Doppler angles and AWGN. Coded and uncoded panels have disjoint streams.
            seed = 846100 + 200 * int(coded) + batch_i * 4
            make_gen = lambda off: torch.Generator().manual_seed(seed + off)
            template = sim.generate_slot(batch_size=8,snr_db=SNR,pa_condition="mild",
                coded=coded,ldpc=code if coded else None,generator=make_gen(0))
            waves = transmit(template)
            white = complex_randn((8,16),generator=make_gen(1))
            dop = FD * torch.cos(2*math.pi*torch.rand((8,16),generator=make_gen(2)))
            noise = complex_randn((8,S*(N+CP)),generator=make_gen(3),dtype=torch.complex128)
            noise *= template.noise_var.double().sqrt()
            for label, profile, fractional in GEOMETRIES:
                pdp = torch.tensor(SLOT_PROFILES[profile],dtype=torch.float32)
                gain = white[:,:len(pdp)] * (pdp / pdp.sum()).sqrt()
                slot = make_slot(template,waves,gain,dop[:,:len(pdp)],
                                 delay_geometry(len(pdp),fractional),noise,profile)
                for name in methods:
                    if name.startswith("dft8_"):
                        llrs, h = dft8(slot.yp_sparse,slot.y_grid,slot.noise_var,name=="dft8_tap_phase")
                    else:
                        pred = models[name](slot.yp_sparse,slot.y_grid,slot.pilot_mask,slot.noise_var)
                        llrs,h = pred.llrs,pred.h_hat
                    data_llr = llrs[:,slot.data_mask,:]
                    if not torch.isfinite(data_llr).all() or not torch.isfinite(h).all():
                        raise FloatingPointError(name)
                    errors = ((data_llr>0) != slot.bits).reshape(8,-1).sum(-1)
                    nmse = (h-slot.h_true).abs().square().sum((1,2))/slot.h_true.abs().square().sum((1,2))
                    row = dict(coded=coded, payload_seed=seed, geometry=label, method=name, slots=8,
                               raw_bits=slot.bits.numel(), raw_errors_by_slot=errors.tolist(), nmse_by_slot=nmse.tolist())
                    if coded:
                        blocks, info = per_slot_errors(data_llr,slot,code)
                        row.update(blocks=slot.info_bits.shape[0]*slot.info_bits.shape[1],
                                   info_bits=slot.info_bits.numel(), failed_by_slot=blocks, wrong_info_by_slot=info)
                    rows.append(row)
    result = {"source_sha256":digest(Path(__file__)), "weights_sha256":hashes, "checks":checks,
              "fractional_delays_samples":delay_geometry(16,True).tolist(),
              "design":"8 batches of 8 per panel, independent IID uncoded and coded traffic; same underlying waveforms, gains, Dopplers and noise across delay geometries. Equal4 has 4 taps, exp16 has 16, with common first four standardized tap draws. Fractional and integer exp16 share all taps.",
              "waveform":"Mild Rapp PA A=2.25 alpha=.5 oversample 4; 20 dB, fd=.01, CFO=0, CP=16. Within each OFDM symbol the post-PA Nyquist samples define a signed-frequency trigonometric waveform with rectangular prefix. No analog pulse-shaping or RF hardware model. Receive-time tap Dopplers and continuous CFO precede AWGN. Original blind CP synchronization for all methods.",
              "threshold":"Truly uncoded exp16 fractional BER minus DFT-8 uncoded BER <= .005; primary DFT-8 uses piecewise linear interpolation and held edges, secondary DFT-8 estimates per-tap phase drift on the same two pilot rows. Both truncate to eight taps and use the common fixed mild PA law to demap and report physical channel NMSE. No tuning.",
              "rows":rows}
    (FOLDER/"next_fractional_holdout.json").write_text(json.dumps(result, indent=2, allow_nan=False),encoding="utf8")
    print(f"{len(rows)} model/geometry/batch/panel records. Checks {checks}")


if __name__ == "__main__":
    run()
