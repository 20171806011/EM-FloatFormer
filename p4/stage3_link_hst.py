"""Paired four-slot HST experiment; transmit exactly the same native source bits in every arm."""
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path

import torch

from p4.simulator import complex_randn,q16,rapp_pa
from p4.slot_2d import (DEFAULT_SCS_HZ,SPEED_OF_LIGHT,SlotSimulator2D,_apply_continuous_slot_channel,
                        _draw_slot_taps,_get_default_ldpc,_synchronize_slot,generate_dmrs)
from p4.stage3_link import load_neural,receive
from p4.stage3_link_long import lifted,receive_long
from p4.stage3_link_study import identity


ROOT=Path(__file__).resolve().parent
SEAL=ROOT/"results"/"stage3_link_hst_protocol.json"
RESULT=ROOT/"results"/"stage3_link_hst_results.json"
OLD=ROOT/"results"/"stage3_link_protocol.json"
PERM=892311
SEEDS=[902071,902072,902073,42]
ARMS=["short_sequential","short_spread","long_slot","long_slot_spread","long_burst_spread"]


def recipe():
    return {"files":{p.name:sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/"stage3_link_long.py",OLD]},
            "previous_receiver_sources_match":identity()==json.loads(OLD.read_bytes()),
            "seeds":SEEDS,"fresh_groups":4,"fresh_group_seed_offset":4096,"historical_groups":1,
            "mapping_seed":PERM,"doppler_hz":[(300/3.6)*3e9/SPEED_OF_LIGHT,500.],
            "channel_modes":["native independent slot amplitudes and frequencies",
                             "single continuous trajectory across 56 OFDM symbols"],
            "pairing":"Four original native slots, same bits, pilots, SNR and original noise sample per position for every arm. Continuous arm propagates first native tap draw for all time.",
            "arms":ARMS,"short":"unchanged (128,64) iterative EM/BP/OSD receiver",
            "long":"QC lift factors 13 and 52 of native parity check, RNG 893011 on systematic edge shifts, parity edge shifts zero. Soft sum-product max40/check-syndrome stop, three waveform rounds. No OSD on long words. Rate 1/2.",
            "metrics":"Native word BLER, native-equivalent 64-information-bit packet loss, BER, raw extrinsic-hard bit counts. Four slots carry 3328 information bits in all arms."}


def group(fd_hz,snr,pa,seed):
    sim=SlotSimulator2D(n_symb=14,n_sc=32,cp=16,profile="equal4",fd=fd_hz/DEFAULT_SCS_HZ)
    code=_get_default_ldpc(128)
    rng=torch.Generator().manual_seed(seed)
    slot=sim.generate_slot(batch_size=4,snr_db=snr,A=pa,alpha=.5,coded=True,ldpc=code,generator=rng)
    replay=torch.Generator().manual_seed(seed)
    assert torch.equal(generate_dmrs(4,sim.num_pilots,generator=replay),slot.pilot_tx)
    assert torch.equal(torch.randint(0,2,slot.info_bits.shape,generator=replay),slot.info_bits)
    taps=_draw_slot_taps(4,sim.n_sc,sim.cp,sim.profile,sim.fd,replay,sim.device)
    state=replay.get_state()
    shape=tuple(slot.rx_time.shape)
    z=complex_randn(shape,generator=torch.Generator().set_state(state),dtype=torch.complex128)
    z2=complex_randn((1,z.numel()),generator=torch.Generator().set_state(state),dtype=torch.complex128)
    assert torch.equal(z.flatten(),z2.flatten())
    return sim,slot,taps,state


def render(sim,original,taps,noise_state,mode,logical,order):
    data=logical.reshape_as(original.bits)
    if order is not None:
        data=data.flatten(0,1)[order].reshape_as(data)
    symbols=q16(data)
    grid=original.x_grid.clone()
    grid[:,sim.data_mask]=symbols
    pa=rapp_pa(torch.fft.ifft(grid,norm="ortho"),A=original.A,alpha=original.alpha,
               beta=original.beta,knee=original.knee,oversample=original.oversample)
    stream=torch.cat((pa[...,-sim.cp:],pa),-1).flatten(1)
    if mode=="continuous":
        stream=stream.reshape(1,-1)
        h,d,f=taps
        taps=(h[:1],d,f[:1])
    elif mode!="native":
        raise ValueError("Unknown propagation")
    sym=sim.n_symb*(4 if mode=="continuous" else 1)
    rx,h,ht=_apply_continuous_slot_channel(stream,*taps,sym,sim.n_sc,sim.cp,sim.cfo,
                    original.noise_var[:len(stream)],torch.Generator().set_state(noise_state))
    rx=rx.reshape_as(original.rx_time)
    y=torch.fft.fft(rx.reshape(4,sim.n_symb,sim.n_sc+sim.cp)[...,sim.cp:],norm="ortho")
    if mode=="native" and order is None and torch.equal(data,original.bits):
        assert torch.equal(rx,original.rx_time) and torch.equal(y,original.y_grid)
    yp=y[:,sim.pilot_mask]
    return replace(original,y_grid=y,rx_time=rx,h_true=h.reshape_as(original.h_true),
                   h_time_evol=ht.reshape_as(original.h_time_evol),data_tx=symbols,bits=data,
                   yp_sparse=yp*original.pilot_tx.conj(),y_p_raw=yp,y_d_raw=y[:,sim.data_mask],x_grid=grid)


@torch.inference_mode()
def trial(model,sim,source,taps,state,mode,arm):
    if arm not in ARMS:
        raise ValueError("Unknown HST link configuration")
    count=sim.num_data*len(source.bits)
    order=(torch.randperm(count,generator=torch.Generator().manual_seed(PERM)) if arm.endswith("spread") else None)
    code=None
    logical=source.bits
    if arm.startswith("long"):
        code=lifted(52 if "burst" in arm else 13)
        logical=code.encode(source.info_bits.reshape(-1,code.k))
        assert torch.equal(logical[:,:code.k].flatten(),source.info_bits.flatten())
    slot=_synchronize_slot(render(sim,source,taps,state,mode,logical,order))
    args=dict(cp=sim.cp,profile=sim.profile,A=slot.A,alpha=slot.alpha,neural=model)
    if code is None:
        k=64
        out=receive(slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var,physical_to_source=order,**args)
    else:
        k=code.k
        out=receive_long(slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var,code=code,order=order,**args)
    err=out["info_bits"].flatten()!=source.info_bits.flatten()
    blocks=err.reshape(-1,k).any(-1)
    chunks=err.reshape(4,-1,64).any(-1)
    raw=(out["llrs"]>0)!=slot.bits
    return {"blocks":blocks.numel(),"block_errors":int(blocks.sum()),"bler":float(blocks.double().mean()),
            "bit_errors":int(err.sum()),"total_bits":err.numel(),"packets64":chunks.numel(),
            "packet64_errors":int(chunks.sum()),"packet64_errors_per_slot":chunks.sum(-1).tolist(),
            "raw_bit_errors":int(raw.sum()),"raw_bits":raw.numel(),"codeword_length":2*k,
            "information_length":k,"physical_interleaver_span_slots":4 if order is not None else 1,
            "nonfinite_count":sum(int((~torch.isfinite(v)).sum()) for v in out.values())}


def run_hst_workload(speed_kmh=300.,fc=3.e9,pa_A=1.25,snr_db=15.,seed=42,*,
                     fd_hz=None,propagation="native",arm="short_sequential"):
    """Complete four-slot HST burst, with explicit physical mapping and code."""
    torch.set_num_threads(2)
    fd_hz=fc*speed_kmh/3.6/SPEED_OF_LIGHT if fd_hz is None else fd_hz
    sim,s,taps,state=group(fd_hz,snr_db,pa_A,seed)
    return dict(trial(load_neural(),sim,s,taps,state,propagation,arm),
                fd_hz=fd_hz,propagation=propagation,arm=arm,seed=seed)


def evaluate():
    assert json.loads(SEAL.read_bytes())==recipe()
    torch.set_num_threads(2)
    net=load_neural()
    result={"protocol":sha256(SEAL.read_bytes()).hexdigest(),"rows":[]}
    if RESULT.exists():
        saved=json.loads(RESULT.read_bytes())
        assert saved["protocol"]==result["protocol"]
        result=saved
    for seed in SEEDS:
        for hz in recipe()["doppler_hz"]:
            for g in range(1 if seed==42 else 4):
                actual=seed+4096*g
                sim,slot,taps,state=group(hz,15.,1.25,actual)
                for mode in ("native","continuous"):
                    for arm in ARMS:
                        keys={"seed":seed,"group":g,"actual_seed":actual,"doppler_hz":hz,"propagation":mode,"arm":arm}
                        if any(all(r[k]==v for k,v in keys.items()) for r in result["rows"]):
                            continue
                        result["rows"].append(dict(keys,**trial(net,sim,slot,taps,state,mode,arm)))
                tmp=RESULT.with_suffix(".tmp")
                tmp.write_text(json.dumps(result,indent=2,allow_nan=False))
                tmp.replace(RESULT)
                print(seed,round(hz,2),g,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal",action="store_true")
    g.add_argument("--evaluate",action="store_true")
    if p.parse_args().seal:
        with SEAL.open("x",encoding="utf-8") as f:
            json.dump(recipe(),f,indent=2)
    else:
        evaluate()
