"""Prospectively fixed S01/S11 link study. Run --seal once, then --evaluate."""
import argparse
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import time

import numba
import torch

from p4 import slot_2d
from p4.next_interleaving import HST_DOPPLER, transmit
from p4.simulator import q16, rapp_pa
from p4.slot_2d import (DEFAULT_SCS_HZ, SlotSimulator2D, _apply_continuous_slot_channel,
                        _draw_slot_taps, _get_default_ldpc, _synchronize_slot, generate_dmrs)
from p4.stage3_link import CHECKPOINT, load_neural, receive
from p4.stage3_link_decoder import SoftListDecoder


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SEAL = OUT / "stage3_link_protocol.json"
RESULT = OUT / "stage3_link_evaluation.json"
SEEDS = [874121,874122,874123,42]
PERM = 892311


def identity():
    files = {p.name: sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__), HERE/"stage3_link.py", HERE/"stage3_link_decoder.py", HERE/"next_neural.py",
        HERE/"next_classical.py", HERE/"next_interleaving.py", HERE/"pilot_prior_probe.py",
        HERE/"simulator.py", HERE/"classical_baselines.py", CHECKPOINT]}
    owning = [slot_2d.SlotSimulator2D, slot_2d.EMFloatFormer2D, slot_2d._apply_continuous_slot_channel,
              slot_2d._draw_slot_taps, slot_2d._synchronize_slot, slot_2d.encode_ldpc,
              slot_2d.decode_min_sum, slot_2d.LDPCMatrix, slot_2d._build_girth6_rate_half_h,
              slot_2d._compute_systematic_generator, slot_2d.get_slot_masks]
    sources = {f.__name__:sha256(inspect.getsource(f).encode()).hexdigest() for f in owning}
    return {"files":files, "core_functions":sources,
            "seeds_in_execution_order":SEEDS, "permutation_seed":PERM,
            "scenarios":{"S01_speed":"32 tones, 14 symbols, physical 300km/h at 3GHz, strong A1.25, 15dB, native mapping, CP synchronizer",
                         "S01_text":"identical except literal 500Hz",
                         "S11_native":"legacy generator, 50 conditionally independent per-slot tap draws in two chunks of 25, 80 tones, 300Hz, strong A1.25, 15dB, native mapping",
                         "S11_continuous":"same logical payload and DMRS as S11_native; 50 consecutive slots transmitted as one 700-symbol waveform with one initial per-tap amplitude and Doppler draw, contiguous sample time, and independent AWGN, native mapping",
                         "S11_spread":"same continuous waveform channel and AWGN as S11_continuous; fixed 50-slot QAM permutation before PA, inverse before LDPC; 50-slot receive buffering"},
            "workload_sizes":"S01 16 fresh slots each / 4 historical. S11 50 slots every run, 102400 information bits.",
            "targets":{"S01":"BLER < 0.05","S11":"BLER < 0.01, accounting, finite outputs"},
            "receiver":"Frozen first EM replicate 73161, three exact 16-symbol marginalization and decoder extrinsic waveform rounds, half-damping and |extrinsic| <=20, BP max20, followed by OSD2 + triples in weakest 16 basis pivots and quads in weakest 8. All results reported. No test-label feedback.",
            "generator_notes":"S01_speed reproduces historical S01 source waveforms; historical S11_mild A2.25 was not the written workload. S11 uses strong at every point. Continuous and spread arms keep 50-slot payload, power law, rate, pilots and SNR. Continuous draw starts from first original tap vector. Noise seed=seed+46871."}


def native(sim, batches, seed):
    code = _get_default_ldpc(128)
    rng = torch.Generator().manual_seed(seed)
    slots = []
    first_taps = None
    for n in batches:
        replay = torch.Generator().set_state(rng.get_state())
        slot = sim.generate_slot(batch_size=n, snr_db=15., pa_condition="strong", coded=True, ldpc=code, generator=rng)
        if first_taps is None:
            assert torch.equal(generate_dmrs(n,sim.num_pilots,generator=replay),slot.pilot_tx)
            assert torch.equal(torch.randint(0,2,slot.info_bits.shape,generator=replay),slot.info_bits)
            padding = sim.num_data*4 % code.n
            if padding:
                assert torch.equal(torch.randint(0,2,(n,padding),generator=replay),slot.bits.flatten(1)[:,-padding:])
            taps = _draw_slot_taps(n,sim.n_sc,sim.cp,sim.profile,sim.fd,replay,sim.device)
            check = transmit(sim,slot,torch.arange(sim.num_data),taps,replay.get_state())
            assert torch.equal(check.rx_time,slot.rx_time)
            first_taps = (taps[0][:1],taps[1],taps[2][:1])
        slots.append(slot)
    return slots,first_taps


def continuous(sim, slots, taps, order, noise_seed):
    data = torch.cat([s.data_tx for s in slots])
    truth = torch.cat([s.bits for s in slots])
    if order is not None:
        data = data.flatten()[order].reshape_as(data)
        truth = truth.flatten(0,1)[order].reshape_as(truth)
        assert torch.equal(data.flatten()[order.argsort()], torch.cat([s.data_tx for s in slots]).flatten())
    assert torch.equal(q16(truth),data)
    grid = torch.cat([s.x_grid for s in slots])
    grid[:,sim.data_mask] = data
    s = slots[0]
    pa = rapp_pa(torch.fft.ifft(grid,norm="ortho"), A=s.A,alpha=s.alpha,beta=s.beta,knee=s.knee,oversample=s.oversample)
    stream = torch.cat([pa[:,:,-sim.cp:],pa],dim=-1).reshape(1,-1)
    n = len(grid)
    rx,h,ht = _apply_continuous_slot_channel(stream,*taps,sim.n_symb*n,sim.n_sc,sim.cp,sim.cfo,
                                               s.noise_var[:1],torch.Generator().manual_seed(noise_seed))
    rx = rx.reshape(n,-1)
    y = torch.fft.fft(rx.reshape(n,sim.n_symb,sim.n_sc+sim.cp)[...,sim.cp:],norm="ortho")
    yp = y[:,sim.pilot_mask]
    pilot = torch.cat([t.pilot_tx for t in slots])
    return replace(s, y_grid=y,yp_sparse=yp*pilot.conj(),y_p_raw=yp,y_d_raw=y[:,sim.data_mask],
                   pilot_tx=pilot,data_tx=data,x_grid=grid,bits=truth,
                   info_bits=torch.cat([t.info_bits for t in slots]),
                   coded_bits=torch.cat([t.coded_bits for t in slots]),
                   h_true=h.reshape_as(grid),h_time_evol=ht.reshape(n,sim.n_symb,-1),
                   noise_var=torch.cat([t.noise_var for t in slots]),rx_time=rx)


def score(net, sim, slots, order=None, sync=False):
    if sync:
        slots = [_synchronize_slot(s) for s in slots]
    first = slots[0]
    result = receive(torch.cat([s.y_grid for s in slots]),torch.cat([s.pilot_tx for s in slots]),
                     first.pilot_mask,torch.cat([s.noise_var for s in slots]),cp=sim.cp,profile=sim.profile,
                     A=first.A,alpha=first.alpha,neural=net,physical_to_source=order)
    truth = torch.cat([s.info_bits for s in slots])
    invalid = {k:int((~torch.isfinite(v)).sum()) for k,v in result.items()}
    methods = {}
    for key in ("info_bits","bp_info_bits","minsum_info_bits","init_minsum_info_bits"):
        assert result[key].shape==truth.shape
        err = result[key]!=truth
        blocks = err.any(-1).sum(-1).tolist()
        bits = err.sum((-1,-2)).tolist()
        assert sum(bits)==int(err.sum()) and sum(blocks)==int(err.any(-1).sum())
        methods[key]={"blocks_wrong":sum(blocks),"bits_wrong":sum(bits),
                      "blocks_wrong_per_slot":blocks,"bits_wrong_per_slot":bits}
    return {"methods":methods,"nonfinite":invalid,"blocks":truth.numel()//64,"bits":truth.numel(),"slots":len(truth)}


def codec_check():
    code = _get_default_ldpc(128)
    info = torch.randint(0,2,(2,3,64),generator=torch.Generator().manual_seed(874101))
    w,_ = slot_2d.encode_ldpc(info,code)
    bits = ((torch.arange(16)[:,None] >> torch.arange(4)) & 1)
    metric = -100*(q16(w.reshape(2,3,32,4))[...,None]-q16(bits)).abs().square()
    decoder = SoftListDecoder(code)
    post = decoder.posterior(20*(2*w-1))
    assert torch.equal((post>0).int(),w) and torch.equal(decoder.list_decode(post,metric),info)


def evaluate():
    protocol = json.loads(SEAL.read_text(encoding="utf-8"))
    if protocol != identity():
        raise RuntimeError("Source, checkpoint or design differs from freeze")
    torch.set_num_threads(2)
    numba.set_num_threads(1)
    codec_check()
    net = load_neural()
    seal_digest=sha256(SEAL.read_bytes()).hexdigest()
    payload={"seal":seal_digest,"torch":str(torch.__version__),"decoder_selftest":True,"rows":[]}
    if RESULT.exists():
        payload=json.loads(RESULT.read_text(encoding="utf-8"))
        assert payload["seal"]==seal_digest
    def record(case, seed, fn):
        if any(r["scenario"]==case and r["seed"]==seed for r in payload["rows"]):
            return
        start=time.perf_counter()
        result=fn()
        row={"scenario":case,"seed":seed,"wall_sec":round(time.perf_counter()-start,2),**result}
        payload["rows"].append(row)
        tmp=RESULT.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
        tmp.replace(RESULT)
        print(case, seed, {k:v["blocks_wrong"] for k,v in row["methods"].items()}, row["blocks"], row["wall_sec"],flush=True)
    for seed in SEEDS:
        for name, fd in (("S01_speed",HST_DOPPLER),("S01_text",500/DEFAULT_SCS_HZ)):
            sim=SlotSimulator2D(n_symb=14,n_sc=32,cp=16,profile="equal4",fd=fd)
            slots,_=native(sim,[4] if seed==42 else [4]*4,seed)
            record(name,seed,lambda:score(net,sim,slots,sync=True))
        sim=SlotSimulator2D(n_symb=14,n_sc=80,cp=16,profile="equal4",fd=300/DEFAULT_SCS_HZ)
        slots,taps=native(sim,[25,25],seed)
        order=torch.randperm(sim.num_data*50,generator=torch.Generator().manual_seed(PERM))
        record("S11_native",seed,lambda:score(net,sim,slots))
        direct=continuous(sim,slots,taps,None,seed+46871)
        spread=continuous(sim,slots,taps,order,seed+46871)
        assert torch.equal(direct.h_true,spread.h_true) and torch.equal(direct.pilot_tx,spread.pilot_tx)
        assert torch.equal(direct.info_bits,spread.info_bits)
        record("S11_continuous",seed,lambda:score(net,sim,[direct]))
        record("S11_spread",seed,lambda:score(net,sim,[spread],order))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args=parser.parse_args()
    if args.seal:
        with SEAL.open("x",encoding="utf-8") as stream:
            json.dump(identity(),stream,indent=2,allow_nan=False)
    else:
        evaluate()
