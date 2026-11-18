"""Pilot-only, soft decision-directed and PA-informed reference receivers.

All methods see the same FFT grid, pilots, public experiment regime and AWGN
variance. Only the explicitly selected genie accepts a realized PA trajectory.
Diagonal channel estimates describe propagation (including receiver CPE) before
the transmitter PA. The 2-D Wiener filter processes each receive antenna.
"""

from functools import lru_cache
from math import pi

import torch

from p4.classical_baselines import calibrated_16qam_llrs
from p4.simulator import q16
from p4.slot_2d import _get_default_ldpc, decode_min_sum, encode_ldpc
from p5.wavefield_sim import CP, DELAYS, POWERS, frequency_basis, pa_grid


METHODS = {
    1: ("nominal_bussgang_lmmse", "nominal_idd_pa", "pilot_cpe_dd", "genie_pa"),
    2: ("tap_phase_slope", "lmmse_2d", "waveform_dd"),
    3: ("lmmse_2d", "waveform_dd"),
}
REFERENCES = {1: "nominal_idd_pa", 2: "waveform_dd", 3: "waveform_dd"}
REAL = torch.float64
NOMINAL_PA = (1.775, 2., .30)


def _support(regime):
    return torch.tensor(DELAYS[regime], dtype=REAL), torch.tensor(POWERS[regime], dtype=REAL)


def _delays(regime, n):
    delays, _ = _support(regime)
    return frequency_basis(delays,n).mT


def _time(symbols, n):
    # Time in useful OFDM symbol durations, measured from the start of the CP.
    return (torch.arange(symbols, dtype=REAL)[:, None] * (n + CP)
            + CP + torch.arange(n, dtype=REAL)) / n


@lru_cache(maxsize=1)
def _quadrature():
    # Trapezoidal quadrature in amplitude avoids a second host BLAS runtime.
    radius = torch.linspace(0.,8.,513,dtype=REAL)
    u = radius.square()
    weights = 2 * radius * torch.exp(-u) * (radius[1]-radius[0])
    weights[-1] *= .5
    return u, weights


def _pa_parameters(obs, genie):
    batch, _, symbols, _ = obs.y.shape
    if obs.regime == 2:
        return None, torch.ones(batch, symbols, 1, dtype=torch.complex128), torch.zeros(batch, symbols, 1, dtype=REAL)
    if genie is None:
        args = tuple(torch.full((1, 1, 1), value, dtype=REAL) for value in NOMINAL_PA)
    else:
        args = (genie.A, genie.knee, genie.alpha)
    # E|x|²=1; u=|x|² ~ Exp(1). Compute gamma=E[f(x)x*], and
    # E|d|²=E|f(x)|²-|gamma|² independently for each realized PA symbol.
    u, weights = _quadrature()
    A, knee, alpha = args
    g = (1 + (u / A.square()).pow(knee)).pow(-1 / (2 * knee))
    gain = (weights * u * g * torch.exp(1j * alpha * u / (1 + u))).sum(-1, keepdim=True)
    dist = (weights * u * g.square()).sum(-1, keepdim=True) - gain.abs().square()
    return args, gain, dist


def _pilot_observations(obs):
    return obs.y[:, :, obs.pilot_mask] / obs.pilots[obs.pilot_mask]


@lru_cache(maxsize=12)
def _covariance(regime, symbols, n, pilot_indices):
    """Public ensemble covariance at symbol centers, including R2 chirp prior."""
    p = torch.tensor(pilot_indices, dtype=torch.long)
    t = _time(symbols, n).mean(-1).repeat_interleave(n)
    f = torch.fft.fftfreq(n, dtype=REAL).repeat(symbols)
    dt = t[:, None] - t[p]
    df = f[:, None] - f[p]
    if regime == 2:
        # Integral of linear Doppler over [t_p,t_g]; conditional on a common
        # sign, the initial/final magnitudes are independent uniform draws.
        end = (symbols * (n + CP) - 1) / n
        dq = (t[:, None].square() - t[p].square()) / (2 * end)
        rt = (torch.cos(2*pi*.0625*dt) * torch.sinc(.035*(dt-dq)) * torch.sinc(.035*dq))
    elif regime == 3:
        rt = torch.sinc(.070 * dt)
    else:
        rt = torch.ones_like(dt)
    delay, power = _support(regime)
    components = torch.exp(-2j*pi*df[..., None]*delay)
    if regime == 3:
        # Off-grid delay jitter is uniform over [-.15,.15] on nonzero rays.
        components[..., 1:] *= torch.sinc(.30 * df)[..., None]
    rgp = rt * (components * power).sum(-1)
    return rgp, rgp[p]


def _lmmse(obs, gamma, distortion):
    b, r, s, n = obs.y.shape
    flat_p = obs.pilot_mask.flatten().nonzero().flatten()
    rgp, rpp = _covariance(obs.regime, s, n, tuple(flat_p.tolist()))
    gamma_p = gamma.expand(b,s,n)[:, obs.pilot_mask]
    var = ((obs.noise_var[:, None, None]+distortion) / gamma.abs().square()).expand(b,s,n)[:, obs.pilot_mask]
    gram = rpp + torch.diag_embed(var)
    weights = torch.linalg.solve(gram, rgp.mH.expand(b,-1,-1)).mH
    yp = _pilot_observations(obs) / gamma_p[:, None]
    h = (weights @ yp.transpose(1,2)).transpose(1,2).reshape(b,r,s,n)
    post_var = (1-(weights * rgp.conj()).sum(-1).real).reshape(b,1,s,n)
    return h, post_var


def _pilot_taps(obs, gamma):
    n = obs.y.shape[-1]
    basis = _delays(obs.regime, n)
    rows = obs.pilot_mask.any(-1).nonzero().flatten()
    fits = []
    for row in rows:
        mask = obs.pilot_mask[row]
        yp = obs.y[:,:,row,mask] / (obs.pilots[row,mask] * gamma.expand(-1,obs.y.shape[2],-1)[:,None,row])
        fits.append(yp @ torch.linalg.pinv(basis[mask]).mT)
    return rows, torch.stack(fits, dim=2)


def _tap_slope(obs):
    # The P4B comparator takes the principal slope separately in each slot.
    b, r, s, n = obs.y.shape
    rows, fits = _pilot_taps(obs, torch.ones(1,1,1))
    segments = []
    for i in (0,2):
        first, second = fits[:,:,i], fits[:,:,i+1]
        turn = torch.angle(second * first.conj())
        start, stop = int(rows[i]), int(rows[i+1])
        portion = torch.arange(stop-start+1,dtype=REAL) / (stop-start)
        base = (first + second * torch.exp(-1j * turn)) / 2
        segments.append(base[:,:,None] * torch.exp(1j*turn[:,:,None]*portion[:,None]))
    return torch.cat(segments, 2) @ _delays(obs.regime,n).mT


def _demap(y, effective, variance):
    # For SIMO, whiten each branch, apply its matched filter, then divide by
    # the combined gain. This is the unbiased equivalent of scalar LMMSE.
    precision = (effective.abs().square() / variance).sum(1).clamp_min(1e-30)
    z = ((y * effective.conj() / variance).sum(1) / precision)
    sigma = precision.reciprocal()
    return calibrated_16qam_llrs(z, sigma), z, sigma


def _soft(z, sigma):
    pam = torch.tensor([-3.,-1.,1.,3.],dtype=REAL) / 10**.5
    def component(x):
        logp = -(x[...,None] - pam).square() / sigma[...,None]
        return (logp.softmax(-1) * pam).sum(-1)
    return torch.complex(component(z.real), component(z.imag))


def _code_feedback(x, llr, mask):
    """P4B-style confidence-weighted re-encoding of the public LDPC code."""
    code = _get_default_ldpc(128)
    channel = llr[:,~mask].reshape(x.shape[0], -1, code.n)
    info = decode_min_sum(channel, code.h_matrix, max_iter=6)
    bits, _ = encode_ldpc(info,code)
    qam = q16(bits.reshape(*bits.shape[:2], -1,4)).to(x.dtype)
    cost = (channel.abs() * (bits != (channel > 0))).mean(-1,keepdim=True)
    confidence = (-cost).exp()
    segment = x[:,~mask].reshape_as(qam)
    x[:,~mask] = (confidence*qam + (1-confidence)*segment).flatten(1)
    return x


@lru_cache(maxsize=4)
def _trajectory_bank(regime):
    if regime == 2:
        mags = torch.linspace(.045,.080,9,dtype=REAL)
        pairs = torch.cartesian_prod(mags,mags)
        return torch.cat((pairs,-pairs),0)
    velocities = torch.linspace(-.035,.035,31,dtype=REAL)
    return torch.stack((velocities,velocities),-1)


def _phasor(t, start, finish, duration):
    return torch.exp(2j*pi*(t*start + .5*t.square()*(finish-start)/duration))


def _fit_pilot_trajectory(obs, gamma):
    """Fit a shared Doppler per path to every pilot row and antenna."""
    b, r, s, n = obs.y.shape
    rows, taps = _pilot_taps(obs, gamma)
    t = _time(s,n)
    duration = (s*(n+CP)-1)/n
    bank = _trajectory_bank(obs.regime)
    candidates = _phasor(t[rows,:][None], bank[:,0,None,None], bank[:,1,None,None], duration).mean(-1)
    corr = torch.einsum("brpd,cp->brdc", taps, candidates.conj())
    norm = candidates.abs().square().sum(-1)
    best = (corr.abs().square().sum(1) / norm).argmax(-1)
    nu = bank[best]
    rot = _phasor(t[None,:,:,None], nu[:,None,None,:,0], nu[:,None,None,:,1], duration)
    p = rot[:,rows].mean(2)
    amp = (taps*p[:,None].conj()).sum(2) / p.abs().square().sum(1)[:,None]
    return amp, rot


def _waveform(obs, gamma, distortion, pa_params, method):
    b, r, s, n = obs.y.shape
    f_basis = _delays(obs.regime, n)
    pdp = _support(obs.regime)[1]
    noise = obs.noise_var[:,None,None,None]
    cpe = torch.ones(b,s,dtype=obs.y.dtype)
    track_phase = method in ("pilot_cpe_dd","genie_pa")
    if obs.regime == 1:
        rows, taps = _pilot_taps(obs,gamma)
        if track_phase:
            turn = torch.angle((taps[:,:,1]*taps[:,:,0].conj()).sum((1,2)))
            cpe = torch.exp(1j * turn[:,None] * torch.linspace(0,1,s,dtype=REAL))
        amp = (taps * cpe[:,None,rows,None].conj()).mean(2)
        rot = torch.ones(b,s,n,len(pdp),dtype=obs.y.dtype)
    else:
        amp, rot = _fit_pilot_trajectory(obs, gamma)

    def diagonal():
        return ((amp[:,:,None] * rot.mean(2)[:,None]) @ f_basis.mT) * cpe[:,None,:,None]

    h = diagonal()
    effective = h*gamma[:,None]
    variance = noise + h.abs().square() * distortion[:,None]
    llr, z, sigma = _demap(obs.y,effective,variance)
    rx = torch.fft.ifft(obs.y,norm="ortho")
    for round_index in range(2):
        x = torch.where(obs.pilot_mask,obs.pilots,_soft(z,sigma))
        if method == "waveform_dd" and round_index == 1:
            x = _code_feedback(x,llr,obs.pilot_mask)
        emitted = pa_grid(x,*pa_params) if pa_params is not None else x
        delayed = torch.fft.ifft(emitted[...,None] * f_basis, dim=2, norm="ortho")
        basis = delayed * rot
        design = (basis * cpe[:,:,None,None]).reshape(b,s*n,-1)
        gram = design.mH @ design + torch.diag_embed(obs.noise_var[:,None] / pdp)
        amp = torch.linalg.solve(gram,design.mH @ rx.reshape(b,r,-1).mT).mT
        pred_no_cpe = torch.einsum("btnd,brd->brtn",basis,amp)
        if track_phase:
            # Symbol CPE follows phasor of measured samples against the soft waveform.
            cpe = torch.exp(1j*torch.angle((rx*pred_no_cpe.conj()).sum((1,3))))
        predicted = torch.fft.fft(pred_no_cpe*cpe[:,None,:,None],norm="ortho")
        h = diagonal()
        effective = h*gamma[:,None]
        cleaned = obs.y - (predicted - effective * x[:,None])
        llr, z, sigma = _demap(cleaned,effective,noise)
    return llr, h


@torch.no_grad()
def receive(obs, method, *, genie_pa=None):
    """Return LLRs (positive means one) and propagation channel in physical units."""
    if method not in METHODS[obs.regime]:
        raise ValueError(f"Unsupported method {method} for regime {obs.regime}")
    if (method == "genie_pa") != (genie_pa is not None):
        raise ValueError("Supply a realized PAState exclusively for the genie_pa receiver")
    pa, gamma, distortion = _pa_parameters(obs,genie_pa)
    if method in ("lmmse_2d","nominal_bussgang_lmmse"):
        h, channel_var = _lmmse(obs,gamma,distortion)
        variance = (obs.noise_var[:,None,None,None] + distortion[:,None]*h.abs().square()
                    + channel_var*(gamma.abs().square()+distortion)[:,None])
    elif method == "tap_phase_slope":
        h = _tap_slope(obs)
        variance = obs.noise_var[:,None,None,None]
    else:
        return _waveform(obs,gamma,distortion,pa,method)
    llr, _, _ = _demap(obs.y,h*gamma[:,None],variance)
    return llr, h
