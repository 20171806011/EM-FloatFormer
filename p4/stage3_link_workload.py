"""Public continuous S11 burst workload with exact integer counters."""
import math

import torch

from p4.simulator import rapp_pa
from p4.slot_2d import (DEFAULT_CP, DEFAULT_SCS_HZ, SlotSimulator2D, _draw_slot_taps,
                        _get_default_ldpc, generate_dmrs)
from p4.stage3_link import load_neural, receive
from p4.stage3_link_study import PERM, continuous


def _draw(sim,n,rng,code,snr,first):
    replay=torch.Generator().set_state(rng.get_state()) if first else None
    slot=sim.generate_slot(batch_size=n,snr_db=snr,pa_condition="strong",coded=True,ldpc=code,generator=rng)
    if not first:
        return slot,None
    generate_dmrs(n,sim.num_pilots,generator=replay)
    torch.randint(0,2,slot.info_bits.shape,generator=replay)
    pad=sim.num_data*4%code.n
    if pad:
        torch.randint(0,2,(n,pad),generator=replay)
    h,d,f=_draw_slot_taps(n,sim.n_sc,sim.cp,sim.profile,sim.fd,replay,sim.device)
    return slot,(h[:1],d,f[:1])


def _finish_cp_memory(slot,previous_tail,offset,taps,sim):
    """Restore the delayed first CP samples from the preceding chunk."""
    n=sim.n_sc
    h,delays,f=taps
    for tap,d in enumerate(delays.tolist()):
        if d:
            t=offset+torch.arange(d,dtype=torch.float64)
            gain=h[0,tap]*torch.exp((2j*math.pi/n)*f[0,tap].double()*t)
            slot.rx_time[0,:d] += (gain*previous_tail[-d:]).to(slot.rx_time.dtype)


@torch.inference_mode()
def run_mission_link(total_bits: int=100_000,num_slots: int=50,snr_db: float=15.,seed: int=42,
                     *,mapping: str="interleaved"):
    """Run one continuous 300-Hz trajectory. Each interleaver holds up to 50 whole slots."""
    if mapping not in ("sequential","interleaved"):
        raise ValueError("mapping must be 'sequential' or 'interleaved'")
    if num_slots<0 or total_bits<0 or not math.isfinite(snr_db):
        raise ValueError("Expected nonnegative minima and finite SNR")
    torch.set_num_threads(2)
    sim=SlotSimulator2D(n_symb=14,n_sc=80,cp=DEFAULT_CP,profile="equal4",fd=300/DEFAULT_SCS_HZ)
    code=_get_default_ldpc(128)
    word_count=sim.num_data*4//code.n
    n=max(int(num_slots),math.ceil(int(total_bits)/(word_count*code.k)))
    model=load_neural() if n else None
    rng=torch.Generator().manual_seed(seed)
    base_taps=None
    tail=None
    blk,bit,raw=[],[],[]
    nonfinite=0
    for start in range(0,n,50):
        length=min(50,n-start)
        sources=[]
        for i in range(0,length,25):
            s,taps=_draw(sim,min(25,length-i),rng,code,snr_db,base_taps is None)
            if taps is not None:
                base_taps=taps
            sources.append(s)
        h,d,f=base_taps
        offset=start*sim.n_symb*(sim.n_sc+sim.cp)
        taps=(h*torch.exp(2j*math.pi*f.double()*offset/sim.n_sc),d,f) if start else base_taps
        order=(torch.randperm(sim.num_data*length,generator=torch.Generator().manual_seed(PERM))
               if mapping=="interleaved" else None)
        burst=continuous(sim,sources,taps,order,seed+46871+start)
        if tail is not None:
            _finish_cp_memory(burst,tail,offset,base_taps,sim)
        pa=rapp_pa(torch.fft.ifft(burst.x_grid[-1:,-1:],norm="ortho"),A=burst.A,alpha=burst.alpha,
                   beta=burst.beta,knee=burst.knee,oversample=burst.oversample)
        tail=pa.flatten()[-int(d.max()):]
        out=receive(burst.y_grid,burst.pilot_tx,burst.pilot_mask,burst.noise_var,
                    cp=sim.cp,profile=sim.profile,A=burst.A,alpha=burst.alpha,neural=model,physical_to_source=order)
        nonfinite+=sum(int((~torch.isfinite(v)).sum()) for v in out.values())
        e=out["info_bits"]!=burst.info_bits
        assert e.shape==burst.info_bits.shape
        blk.extend(e.any(-1).sum(-1).tolist())
        bit.extend(e.sum((-1,-2)).tolist())
        raw.extend(((out["llrs"]>0)!=burst.bits).sum((-1,-2)).tolist())
    blocks=n*word_count
    bits=blocks*code.k
    raw_total=n*sim.num_data*4
    return {"bler":sum(blk)/max(1,blocks),"ber":sum(bit)/max(1,bits),
            "total_bit_errors":sum(bit),"total_bits":bits,"block_errors":sum(blk),"total_blocks":blocks,
            "raw_bit_errors":sum(raw),"raw_bits":raw_total,"raw_ber":sum(raw)/max(1,raw_total),
            "block_errors_per_slot":blk,"bit_errors_per_slot":bit,"raw_bit_errors_per_slot":raw,
            "raw_bit_definition":"Hard sign of the last extrinsic demapper LLR, in physical order, includes filler.",
            "nonfinite":nonfinite,"exit_code":int(nonfinite!=0),"num_slots":n,"n_sc":sim.n_sc,
            "n_symb":sim.n_symb,"cp":sim.cp,"snr_db":snr_db,"seed":seed,"pa_A":1.25,"alpha":.5,
            "doppler_hz":300.,"continuous_ofdm_symbols":n*sim.n_symb,"mapping":mapping,
            "interleaver_span_slots":min(50,n) if mapping=="interleaved" else min(1,n),
            "codeword_length":code.n,"information_length":code.k,"pilot_re_per_slot":sim.num_pilots}
