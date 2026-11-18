"""Paired HST four-slot scheduling with a single physical channel trajectory and fixed active-slot power."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch

from p4.simulator import complex_randn, q16, rapp_pa
from p4.slot_2d import DEFAULT_SCS_HZ, SPEED_OF_LIGHT, _synchronize_slot
from p4.stage3_link import CHECKPOINT, load_neural, receive
from p4.stage3_link_hst import PERM, group, render
from p4.stage3_link_long import lifted, receive_long


HERE=Path(__file__).resolve().parent
SEAL=HERE/"results"/"stage3_spaced_protocol.json"
OUTPUT=HERE/"results"/"stage3_spaced_results.json"
DEV=tuple(range(839101,839113))
HERTZ=(300/3.6*3e9/SPEED_OF_LIGHT,500.)
SPACING=(1,8)
ARMS=("qc1664","short128")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def recipe():
    dependencies=[Path(__file__),HERE/"stage3_link_hst.py",HERE/"stage3_link_long.py",HERE/"stage3_link.py",
                  HERE/"stage3_link_decoder.py",HERE/"next_classical.py",HERE/"next_neural.py",CHECKPOINT]
    return {"dependencies":{p.name:digest(p) for p in dependencies},"development_seeds":list(DEV),
            "historical_last":42,"doppler_hz":list(HERTZ),"steps_in_nominal_slot_periods":list(SPACING),
            "conditions":"4 x 14 symbols, 32 tones, 16 CP, 30kHz tone spacing, 15dB, equal4, CFO=0, strong A=1.25, alpha=.5; identical payload, pilots, random QAM permutation and exact receiver noise samples across scheduling choices.",
            "clock":"One complex gain and Doppler per physical path for the entire elapsed burst. Evaluated sample-by-sample. The transmitter is silent in the 21 unallocated slots. No channel/power redraw or normalization. The spacing is fixed at eight nominal slot periods before observing data.",
            "codes":"Primary frozen rate 1/2 QC(1664,832), 4 physical codewords each spanning all active slots; control frozen native (128,64), 52 words also spanning the burst. Keep the same existing symbol permutation seed 892311. All arms carry the original 3328 independent information bits and 52 nonoverlapping 64-information-bit packets.",
            "receiver":"Unchanged frozen first neural replicate 73161 and three independent per-active-slot waveform rounds with cross-slot soft FEC feedback. No realized gain, phase or Doppler in receiver.",
            "scoring":"Report native-word and 64-information-bit-packet error rates, group-level counts, link duration, active resources and non-finites. Strict native BLER <.05 for both Hz, independently expose packet losses; do not choose arms or spacings based on historical result."}


def transmit(source,sim,arm):
    b=4
    code=lifted(13) if arm=="qc1664" else None
    logical=source.bits if code is None else code.encode(source.info_bits.reshape(b,code.k)).reshape_as(source.bits)
    if code:
        assert torch.equal(logical.reshape(b,code.n)[:,:code.k].flatten(),source.info_bits.flatten())
    order=torch.randperm(b*sim.num_data,generator=torch.Generator().manual_seed(PERM))
    bits=logical.flatten(0,1)[order].reshape_as(source.bits)
    data=q16(bits)
    grid=source.x_grid.clone()
    grid[:,sim.data_mask]=data
    wave=rapp_pa(torch.fft.ifft(grid,norm="ortho"),A=source.A,alpha=source.alpha,knee=source.knee,
                 beta=source.beta,oversample=source.oversample)
    return code,order,logical,bits,data,grid,torch.cat((wave[...,-sim.cp:],wave),-1).flatten(1)


def scheduled(sim,source,taps,state,tx,spacing):
    _,_,_,bits,data,grid,wave=tx
    b,duration=wave.shape
    start=torch.arange(b)*spacing*duration
    sample=start[:,None]+torch.arange(duration)[None,:]
    stream=torch.zeros((int(start[-1])+duration,),dtype=torch.complex128)
    stream[sample]=wave.to(stream.dtype)
    gain,delays,freq=taps
    gain=gain[0].to(torch.complex128)
    freq=freq[0].double()
    y=torch.zeros_like(sample,dtype=torch.complex128)
    for i,delay in enumerate(delays.tolist()):
        phase=2*math.pi*freq[i]*(sample.double()/sim.n_sc)
        g=gain[i]*torch.complex(phase.cos(),phase.sin())
        v=sample-delay
        y+=g*stream[v.clamp_min(0)]*(v>=0)
    if sim.cfo:
        phi=2*math.pi*sim.cfo*(sample.double()/sim.n_sc)
        y*=torch.complex(phi.cos(),phi.sin())
    noise=complex_randn((1,b*duration),generator=torch.Generator().set_state(state),dtype=torch.complex128)
    rx=(y+noise.reshape_as(y)*source.noise_var.double().sqrt()).to(source.rx_time.dtype)
    mid=start[:,None]+torch.arange(sim.n_symb)[None,:]*(sim.n_sc+sim.cp)+sim.cp+sim.n_sc/2
    phase=2*math.pi*freq*(mid.double()[...,None]/sim.n_sc)
    ht=gain*torch.complex(phase.cos(),phase.sin())
    k=torch.arange(sim.n_sc).double()
    rot=-2*math.pi*delays.double()[:,None]*k/sim.n_sc
    h=ht@torch.complex(rot.cos(),rot.sin())
    h=h.to(source.h_true.dtype); ht=ht.to(source.h_time_evol.dtype)
    ygrid=torch.fft.fft(rx.reshape(b,sim.n_symb,sim.n_sc+sim.cp)[...,sim.cp:],norm="ortho")
    yp=ygrid[:,sim.pilot_mask]
    return replace(source,x_grid=grid,bits=bits,data_tx=data,rx_time=rx,h_true=h,h_time_evol=ht,
                   y_grid=ygrid,yp_sparse=yp*source.pilot_tx.conj(),y_p_raw=yp,y_d_raw=ygrid[:,sim.data_mask])


def trial(model,sim,source,taps,state,arm,spacing,check=False):
    tx=transmit(source,sim,arm)
    code,order,logical=tx[:3]
    raw_slot=scheduled(sim,source,taps,state,tx,spacing)
    if check:
        reference=render(sim,source,taps,state,"continuous",logical,order)
        if not torch.equal(raw_slot.rx_time,reference.rx_time):
            raise AssertionError(f"Adjacent-schedule received waveform mismatch: {(raw_slot.rx_time-reference.rx_time).abs().max()}")
        torch.testing.assert_close(raw_slot.h_true,reference.h_true,rtol=0,atol=2e-7)
        torch.testing.assert_close(raw_slot.h_time_evol,reference.h_time_evol,rtol=0,atol=2e-7)
    slot=_synchronize_slot(raw_slot)
    args=(slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var)
    kw=dict(cp=sim.cp,profile=sim.profile,A=slot.A,alpha=slot.alpha,neural=model)
    if code is None:
        k=64
        out=receive(*args,physical_to_source=order,**kw)
    else:
        k=code.k
        out=receive_long(*args,code=code,order=order,**kw)
    wrong=out["info_bits"].flatten()!=source.info_bits.flatten()
    blocks=wrong.reshape(-1,k).any(-1)
    packets=wrong.reshape(-1,64).any(-1)
    raw=(out["llrs"]>0)!=slot.bits
    dur=sim.n_symb*(sim.cp+sim.n_sc)/(sim.n_sc*DEFAULT_SCS_HZ)
    return {"arm":arm,"spacing_slots":spacing,"blocks":blocks.numel(),"block_errors":int(blocks.sum()),
            "total_blocks":blocks.numel(),"bler":int(blocks.sum())/blocks.numel(),"total_bits":wrong.numel(),
            "bit_errors":int(wrong.sum()),"packets64":packets.numel(),"packet64_errors":int(packets.sum()),
            "raw_bits":raw.numel(),"raw_bit_errors":int(raw.sum()),"codeword_length":2*k,
            "nonfinite_count":sum(int((~torch.isfinite(x)).sum()) for x in out.values()),
            "active_slots":4,"elapsed_slots":(3*spacing+1),"inter_slot_centers_ms":1000*spacing*dur,
            "elapsed_ms":1000*(3*spacing+1)*dur,"added_wait_ms":1000*3*(spacing-1)*dur,
            "transmit_active_power":float(tx[-1].abs().square().mean()),
            "channel_mean_power_by_slot":raw_slot.h_true.abs().square().mean((-1,-2)).tolist()}


@torch.inference_mode()
def run_spaced_hst(speed_kmh=300.,fc=3.e9,pa_A=1.25,snr_db=15.,seed=42,*,fd_hz=None,spacing=8,arm="qc1664"):
    if type(spacing) is not int or spacing<1 or arm not in ARMS:
        raise ValueError("Positive integer slot spacing and a supported code are required")
    hz=fc*speed_kmh/3.6/SPEED_OF_LIGHT if fd_hz is None else fd_hz
    return {**trial(load_neural(),*group(hz,snr_db,pa_A,seed),arm,spacing),"fd_hz":hz,"seed":seed}


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(2)
    if json.loads(SEAL.read_bytes())!=recipe():
        raise ValueError("Spaced HST experiment differs from seal")
    model=load_neural()
    report={"seal_sha256":digest(SEAL),"rows":[]}
    if OUTPUT.exists():
        report=json.loads(OUTPUT.read_bytes())
        if report["seal_sha256"]!=digest(SEAL):
            raise ValueError("Partial results are from a different seal")
    for seed in (*DEV,42):
        for hz in HERTZ:
            setup=group(hz,15.,1.25,seed)
            for arm in ARMS:
                for spacing in SPACING:
                    key=(seed,hz,arm,spacing)
                    if any((r["seed"],r["hz"],r["arm"],r["spacing_slots"])==key for r in report["rows"]):
                        continue
                    res=trial(model,*setup,arm,spacing,check=(spacing==1))
                    report["rows"].append({"seed":seed,"hz":hz,**res})
            path=OUTPUT.with_suffix(".tmp")
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
            path.replace(OUTPUT)
            print(seed,round(hz,2),[(r["arm"],r["spacing_slots"],r["block_errors"],r["packet64_errors"])
                                   for r in report["rows"] if r["seed"]==seed and r["hz"]==hz],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    task=parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--seal",action="store_true"); task.add_argument("--evaluate",action="store_true")
    if parser.parse_args().seal:
        with SEAL.open("x",encoding="utf8") as f:
            json.dump(recipe(),f,indent=2)
    else:
        evaluate()
