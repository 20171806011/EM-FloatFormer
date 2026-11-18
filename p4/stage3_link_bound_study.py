"""Frozen bounded-Doppler, unknown-QAM likelihood HST study with held-out and paired historic continuous waveforms."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from p4.slot_2d import SPEED_OF_LIGHT
from p4.stage3_link import load_neural
from p4.stage3_link_hst import PERM,group,render
from p4.stage3_link_long import lifted
from p4.stage3_link_bound import bounded_receive


BASE=Path(__file__).resolve().parent
OUT=BASE/"results"
SEAL=OUT/"stage3_link_bound_protocol.json"
RESULT=OUT/"stage3_link_bound.json"
FRESH=list(range(912501,912513))
PAIRED=list(range(907701,907713))+[s+4096*g for s in [902071,902072,902073] for g in range(4)]
ARMS=["short_sequential","short_spread","long_slot_spread","long_burst_spread"]


def recipe():
    files=[Path(__file__),BASE/"stage3_link_bound.py",BASE/"stage3_link_tracking.py",BASE/"stage3_link_hst.py",
           BASE/"stage3_link_long.py",OUT/"stage3_link_hst_protocol.json",OUT/"stage3_link_protocol.json"]
    return {"source_hash":{p.name:sha256(p.read_bytes()).hexdigest() for p in files},
            "schedule":[["fresh",s] for s in FRESH]+[["paired",s] for s in PAIRED]+[["historical",42]],
            "hz":[300/3.6*3e9/SPEED_OF_LIGHT,500.],"arms":ARMS,"mapping_seed":PERM,
            "design":"Only true continuous 56-symbol waveform. Common fractional CFO estimated from all CPs, then 8 known DMRS rows used to fit per-delay sinusoid by 1025-point physically bounded periodogram using maximum Doppler from the declared speed and carrier plus measured common CFO. Per-tap top and separated alternate peak; up to sixteen joint candidates ranked by observed exact uniform 16-QAM mixture under diagonal initial gain plus Bussgang residual noise; constrained subsequent waveform fit. Single amplitude/rate per delay refined by waveform likelihood over entire burst, three extrinsic rounds. Same trained EM weights with observation-estimated effective prior, same code/decoders and four-slot maps as stage3_link_hst.",
            "metrics":"Native codeword errors, 64-bit packet errors, information bit and raw hard LLR errors, evaluation-only channel NMSE after estimated CFO rotation. Threshold 5% coded BLER. Fresh samples then paired then seed42. All arms reported."}


@torch.inference_mode()
def trial(net,sim,source,taps,state,arm):
    order=torch.randperm(source.data_tx.numel(),generator=torch.Generator().manual_seed(PERM)) if arm.endswith("spread") else None
    code=None if arm.startswith("short") else lifted(13 if arm=="long_slot_spread" else 52)
    logical=source.bits if code is None else code.encode(source.info_bits.reshape(-1,code.k))
    s=render(sim,source,taps,state,"continuous",logical,order)
    d=bounded_receive(s.rx_time,s.pilot_tx,s.pilot_mask,s.noise_var,cp=sim.cp,profile=sim.profile,
                    A=s.A,alpha=s.alpha,neural=net,fd_max=sim.fd,code=code,order=order)
    err=d["info_bits"].flatten()!=source.info_bits.flatten()
    k=64 if code is None else code.k
    blocks=int(err.reshape(-1,k).any(-1).sum())
    packets=int(err.reshape(-1,64).any(-1).sum())
    mid=(torch.arange(sim.n_symb*4,dtype=torch.float64)*(sim.n_sc+sim.cp)+sim.cp+.5*sim.n_sc)
    correction=torch.exp(-2j*math.pi*d["cfo_estimate"].double()*mid/sim.n_sc).reshape(4,sim.n_symb,1)
    truth=s.h_true*correction
    nmse=float((d["h_hat"]-truth).abs().square().sum()/truth.abs().square().sum())
    return {"block_errors":blocks,"blocks":err.numel()//k,"bler":blocks/(err.numel()//k),
            "packet64_errors":packets,"packets64":err.numel()//64,"bit_errors":int(err.sum()),
            "total_bits":err.numel(),"raw_bit_errors":int(((d["llrs"]>0)!=s.bits).sum()),
            "raw_bits":s.bits.numel(),"h_nmse":nmse,"cp_cfo_estimate":float(d["cfo_estimate"].item()),
            "nonfinite_count":sum(int((~torch.isfinite(v)).sum()) for v in d.values())}


def run_hst_workload(speed_kmh=300.,fc=3.e9,pa_A=1.25,snr_db=15.,seed=42,*,fd_hz=None,arm="short_sequential"):
    torch.set_num_threads(2)
    if arm not in ARMS:
        raise ValueError("Unknown coding and mapping")
    fd_hz=fc*speed_kmh/3.6/SPEED_OF_LIGHT if fd_hz is None else fd_hz
    sim,s,taps,state=group(fd_hz,snr_db,pa_A,seed)
    return dict(trial(load_neural(),sim,s,taps,state,arm),fd_hz=fd_hz,seed=seed,arm=arm,
                propagation="continuous",codeword_length=128 if arm.startswith("short") else
                (1664 if arm=="long_slot_spread" else 6656),buffering_slots=4)


def run():
    spec=recipe()
    assert json.loads(SEAL.read_bytes())==spec
    torch.set_num_threads(2)
    m=load_neural()
    digest=sha256(SEAL.read_bytes()).hexdigest()
    data=json.loads(RESULT.read_bytes()) if RESULT.exists() else {"protocol":digest,"rows":[]}
    assert data["protocol"]==digest
    for phase,seed in spec["schedule"]:
        for hz in spec["hz"]:
            sim,s,taps,state=group(hz,15.,1.25,seed)
            for arm in ARMS:
                key={"phase":phase,"seed":seed,"doppler_hz":hz,"arm":arm}
                if any(all(r[k]==v for k,v in key.items()) for r in data["rows"]):
                    continue
                data["rows"].append(dict(key,**trial(m,sim,s,taps,state,arm)))
        tmp=RESULT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False))
        tmp.replace(RESULT)
        print(phase,seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal",action="store_true")
    if p.parse_args().seal:
        with SEAL.open("x",encoding="utf-8") as f:
            json.dump(recipe(),f,indent=2)
    else:
        run()
