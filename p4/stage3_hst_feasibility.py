"""Offline S01 converse and true-channel counterfactual on the unchanged four-slot waveforms."""
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np
from scipy.linalg import eig_banded
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
import torch

from p4 import stage3_link_bound as blind
from p4.simulator import complex_randn, rapp_pa
from p4.slot_2d import SPEED_OF_LIGHT, _get_default_ldpc
from p4.stage3_link import load_neural
from p4.stage3_link_hst import group, render


OUT = Path(__file__).parent / "results"
SEEDS = [42, *range(912501, 912513)]
HZ = (300/3.6*3e9/SPEED_OF_LIGHT, 500.)


def waveform_operator(taps, symbols, tones, cp):
    """Map PA-output Nyquist samples, before CP duplication, into *all* received samples."""
    h0, delays, rates = taps
    n = np.arange(symbols*(tones+cp))
    rows, columns, entries = [], [], []
    for initial,delay,rate in zip(h0[0].numpy(), delays.numpy(), rates[0].numpy()):
        t = n[n >= delay]
        tx = t-delay
        row = tx//(tones+cp)
        pos = ((tx%(tones+cp))-cp)%tones
        rows.append(t)
        columns.append(row*tones+pos)
        entries.append(initial*np.exp(2j*np.pi*float(rate)*t/tones))
    return coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(columns))),
                      shape=(len(n),symbols*tones)).tocsr()


def channel_bound(op, limit, nv, word_count, k):
    gram = (op.conj().T @ op).tocoo()
    idx = gram.row >= gram.col
    offsets = gram.row[idx]-gram.col[idx]
    band = np.zeros((int(offsets.max())+1,gram.shape[1]),dtype=np.complex128)
    band[offsets,gram.col[idx]] = gram.data[idx]
    eig = eig_banded(band,lower=True,eigvals_only=True,check_finite=False)
    if eig.min() < -1e-10 or not np.isclose(eig.sum(),np.sum(abs(op.data)**2),rtol=1e-10):
        raise ValueError("Full-wave Gram spectrum does not reproduce its trace")
    eig = np.maximum(eig,0.)
    modes = eig[eig>0][::-1]
    floors = nv/modes
    levels = (len(eig)*limit+np.cumsum(floors))/np.arange(1,len(modes)+1)
    last = np.flatnonzero(levels >= floors)[-1]
    assigned = np.maximum(levels[last]-floors,0.)
    if not np.isclose(assigned.sum(),len(eig)*limit,rtol=1e-8):
        raise ValueError("Waterfilling did not conserve the PA output power budget")
    upper = float(np.log2(1+assigned*modes/nv).sum())
    deficit = max(k-upper/word_count,0.)
    def fano_rhs(p):
        ent = 0. if p==0. or p==1. else -p*math.log2(p)-(1-p)*math.log2(1-p)
        return ent+p*k
    lower = float(brentq(lambda p:fano_rhs(p)-deficit,0.,1.)) if deficit>0 else 0.
    return {"upper_information_bits":upper,"payload_information_bits":word_count*k,
            "mean_word_error_lower_bound":lower,"active_gaussian_modes":last.item()+1,
            "mean_channel_column_energy":float(eig.mean()),"max_squared_singular_value":float(eig.max())}


def offline_oracle(taps,sim):
    """Clone function globals locally. Both channel substeps see the true tap trajectory only here."""
    state = {}
    def known_initial(y,pilot,mask,tap_count,cp,gamma,dist,noise,fd_max,cfo):
        b,s,n = y.shape
        length = b*s*(n+cp)
        times = torch.arange(b*s)[:,None]*(n+cp)+cp+torch.arange(n)[None,:]
        t = times.double()/length-.5
        mids = t[:,0]+.5*n/length
        h0, delays, dopplers = taps
        assert torch.equal(delays,torch.arange(tap_count)) and sim.cfo==0.
        omega = 2*math.pi*(dopplers[:1].double()-float(cfo))*length/n
        amp = h0[:1].to(torch.complex128)*torch.exp(.5j*omega)
        state["amp"], state["omega"] = amp,omega
        return t,mids,omega,amp,(-math.inf,math.inf)

    def known_update(basis, observations, time, previous_omega, noise, pdp):
        amp,omega = state["amp"],state["omega"]
        pred = (basis*torch.exp(1j*time[None,:,None]*omega[:,None,:])*amp[:,None,:]).sum(-1)
        return amp,omega,pred

    original = blind.bounded_receive.__wrapped__
    namespace = dict(original.__globals__,_bounded_pilots=known_initial,_fit_waveform=known_update)
    duplicate = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)
    duplicate.__kwdefaults__ = original.__kwdefaults__
    return duplicate


def score(result,slot):
    mismatch = result["info_bits"].reshape_as(slot.info_bits)!=slot.info_bits
    coded = (result["llrs"]>0)!=slot.bits
    return {"block_errors":int(mismatch.any(-1).sum()),"blocks":52,
            "information_bit_errors":int(mismatch.sum()),"information_bits":3328,
            "raw_bit_errors":int(coded.sum()),"raw_bits":6656,
            "block_errors_per_slot":mismatch.any(-1).sum(-1).tolist()}


@torch.inference_mode()
def measure(seed,hz,model):
    sim,source,taps,noise_state = group(hz,15.,1.25,seed)
    slot = render(sim,source,taps,noise_state,"continuous",source.bits,None)
    count,n,cp = 56,sim.n_sc,sim.cp
    op = waveform_operator(taps,count,n,cp)
    actual_pa = rapp_pa(torch.fft.ifft(slot.x_grid,norm="ortho"),A=source.A,alpha=source.alpha,
                        beta=source.beta,knee=source.knee,oversample=source.oversample)
    noiseless = op @ actual_pa.flatten().numpy()
    expected_noise = complex_randn((1,len(noiseless)),generator=torch.Generator().set_state(noise_state),
                                   dtype=torch.complex128).flatten().numpy()*math.sqrt(float(slot.noise_var[0]))
    error = float(abs(noiseless+expected_noise-slot.rx_time.flatten().numpy()).max())
    if error > 5e-7:
        raise RuntimeError(f"Independent full-wave reconstruction failed: {error:g}")
    # Jensen for f(p)=p/(1+(p/A^2)^2)^0.5. Uniform coded magnitude bits and
    # unit-modulus known DMRS give E[mean pre-PA power]=1. Dropping OOB bins
    # can only reduce energy. This bounds ensemble power, not a single codeword.
    power = (1+float(source.A)**(-4))**-.5
    bound = channel_bound(op,power,float(slot.noise_var[0]),52,64)
    args = (slot.rx_time,slot.pilot_tx,slot.pilot_mask,slot.noise_var)
    keywords = dict(cp=cp,profile=sim.profile,A=source.A,alpha=source.alpha,neural=model,fd_max=sim.fd)
    base = blind.bounded_receive(*args,**keywords)
    known = offline_oracle(taps,sim)(*args,**keywords)
    mid = torch.arange(count,dtype=torch.float64)*(n+cp)+cp+.5*n
    target = slot.h_true*torch.exp(-2j*math.pi*float(known["cfo_estimate"])*mid/n).reshape(4,14,1)
    def channel_nmse(result):
        return float((result["h_hat"]-target).abs().square().sum()/target.abs().square().sum())
    if channel_nmse(known)>1e-12 or not torch.equal(base["cfo_estimate"],known["cfo_estimate"]):
        raise RuntimeError("Oracle channel/correction mismatch")
    reports = {}
    for name,rx in (("blind",base),("oracle_taps",known)):
        reports[name] = {**score(rx,slot),"midpoint_channel_nmse":channel_nmse(rx)}
    return {"seed":seed,"doppler_hz":hz,"max_waveform_reconstruction_error":error,
            "conditional_mean_tap_power":float(taps[0][0].abs().double().square().sum()),
            "realized_pa_output_power":float(actual_pa.abs().double().square().mean()),
            "ensemble_pa_power_upper":power,"capacity_bound":bound,**reports}


def run():
    torch.set_num_threads(2)
    code = _get_default_ldpc(128)
    if code.g_matrix.shape!=(64,128) or not (code.g_matrix.sum(0)>0).all():
        raise RuntimeError("Each coded magnitude bit must have balanced marginal over uniform input")
    net = load_neural()
    with threadpool_limits(limits=1):
        rows = []
        for seed in SEEDS:
            rows.extend(measure(seed,hz,net) for hz in HZ)
            print(seed,[(r["blind"]["block_errors"],r["oracle_taps"]["block_errors"],
                         round(r["capacity_bound"]["upper_information_bits"])) for r in rows[-2:]],flush=True)
    published = OUT/"stage3_link_bound.json"
    previous = json.loads(published.read_bytes())["rows"]
    verified = 0
    for r in rows:
        match = [s for s in previous if s["seed"]==r["seed"] and s["doppler_hz"]==r["doppler_hz"]
                 and s["arm"]=="short_sequential"]
        if match:
            if (match[0]["block_errors"],match[0]["raw_bit_errors"],match[0]["bit_errors"])!=(
                r["blind"]["block_errors"],r["blind"]["raw_bit_errors"],r["blind"]["information_bit_errors"]):
                raise RuntimeError("Blind baseline differs from archived same-waveform experiment")
            verified+=1
    return {"scope":"Offline diagnostic. Continuous 56x32-tone burst, CP16, no gaps, fixed native (128,64), "
                    "short sequential mapping. Same PA, SNR, waveforms, data, noise, CP sync, neural weights, "
                    "feedback, and list decoder. Twelve pre-existing development seeds from the bounded-"
                    "tracking study and historical seed 42, all included, two Dopplers paired per seed.",
        "bound":"Condition on taps and pilots; full linear map from all post-PA/pre-CP samples to all received "
                "samples (including CP, intersymbol tails and ICI), with the exact complex noise variance. "
                "E||postPA||^2 <= 56*32*f(1), f(p)=p/sqrt(1+(p/A^2)^2), by concavity, oversampled Parseval and "
                "in-band projection. Uniform information makes every native coded magnitude bit balanced. "
                "Optimize Gaussian mutual information across all singular modes by waterfilling, relaxing "
                "pilots, constellation and allowed input covariance. This gives an upper bound for the fixed "
                "modulation/coding ensemble, independent of the message and noise realization. A bound above "
                "3328 bits cannot demonstrate achievability.",
        "fano":"For 52 uniform independent 64-bit words, average word error p of any decoder satisfies "
               "3328-U <= 52*[h2(p)+64*p] by conditional entropy subadditivity, Fano and concavity. It "
               "bounds expected word error given this trajectory. It does not bound the realized error count.",
        "oracle":"Offline copy only. Exact complex taps and Dopplers replace the pilot fit and three feedback "
                 "channel refits. The neural initializer, demapping approximation, three decoder exchanges, "
                 "and list decoder remain identical. No payload truth enters either prediction path. Residual "
                 "oracle errors include noise, finite code/decoder limitations and demapper mismatch; they are "
                 "not a fundamental irreducible error floor. The paired BLER change is specific to this estimator.",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blind_archive_cells_reproduced":verified,"rows":rows}


if __name__=="__main__":
    result=run()
    target=OUT/"stage3_hst_feasibility.json"
    tmp=target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(target)
