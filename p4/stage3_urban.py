"""S02 waveform-feedback receiver with the same nominal long12 prior across both scenarios."""
import json
from dataclasses import replace
from pathlib import Path

import torch

from p4.classical_baselines import bussgang_params
from p4.next_classical import delay_doppler_receive
from p4.next_neural import digest
from p4.slot_2d import SLOT_PROFILES, SlotSimulator2D, _get_default_ldpc, _synchronize_slot, decode_min_sum
from p4.stage3_link import CHECKPOINT, load_neural, receive


HERE = Path(__file__).resolve().parent
DEV_SEEDS = (838101, 838102, 838103, 838104)
PROFILES = ("long12", "sparse12")
NOMINAL = "long12"


def generate(n_sc, profile, pa_A, num_slots, snr_db, seed):
    sim = SlotSimulator2D(n_symb=14,n_sc=n_sc,cp=16,profile=profile,fd=.005,cfo=0.)
    code = _get_default_ldpc(128)
    return sim.generate_slot(batch_size=num_slots,snr_db=snr_db,A=pa_A,alpha=.5,
                             coded=True,ldpc=code,generator=torch.Generator().manual_seed(seed))


def run_receiver(method, y, pilots, mask, nv, prior, pa_A, model):
    inputs = (y, pilots, mask, nv)
    metadata = dict(cp=16,profile=prior,A=pa_A,alpha=.5)
    code = _get_default_ldpc(128)
    if method == "link":
        return receive(*inputs,**metadata,neural=model)
    if method == "classical":
        out = delay_doppler_receive(*inputs,**metadata)
        words = int((~mask).sum()) // 32
        raw = out["llrs"].reshape(len(y),-1)[:,:words*128].reshape(-1,128)
        return {**out,"info_bits":decode_min_sum(raw,code.h_matrix).reshape(len(y),words,64)}
    raise ValueError(f"Unknown receiver {method}")


def score(out, slot):
    errors = out["info_bits"] != slot.info_bits
    block = errors.any(-1).sum(-1)
    bits = errors.sum((-1,-2))
    nmse = (out["h_hat"]-slot.h_true).abs().square().sum((-1,-2))/slot.h_true.abs().square().sum((-1,-2))
    nonfinite = int((~torch.isfinite(out["h_hat"])).sum() + (~torch.isfinite(out["llrs"])).sum())
    b, words, k = errors.shape
    result = {"bler":int(block.sum())/(b*words),"nmse":float(nmse.mean()),
              "block_errors":int(block.sum()),"total_blocks":b*words,"bit_errors":int(bits.sum()),
              "total_bits":b*words*k,"nonfinite":nonfinite,"per_slot_block_errors":block.tolist(),
              "per_slot_bit_errors":bits.tolist(),"per_slot_nmse":nmse.tolist()}
    result["passes_both"] = result["bler"] < .01 and result["nmse"] < .05 and not nonfinite
    for k in ("minsum_info_bits","init_minsum_info_bits","bp_info_bits"):
        if k in out:
            diff = out[k].reshape_as(slot.info_bits) != slot.info_bits
            result[k] = {"bit_errors":int(diff.sum()),"block_errors":int(diff.any(-1).sum())}
    return result


@torch.inference_mode()
def run_urban_workload(n_sc=64, profile="long12", pa_A=2.25, num_slots=20, snr_db=18., seed=42):
    """Single-profile entry using declared PA and a fixed prior shared by the two S02 profiles."""
    slot = _synchronize_slot(generate(n_sc,profile,pa_A,num_slots,snr_db,seed))
    prior = NOMINAL if profile in PROFILES else profile
    if n_sc % 2 or len(SLOT_PROFILES[prior]) > n_sc//2:
        raise ValueError("Waveform receiver requires even N and at least one comb pilot per nominal tap")
    out = run_receiver("link",slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var,prior,pa_A,load_neural())
    return {**score(out,slot),"receiver":"tap-phase initialization, waveform feedback and soft-list LDPC",
            "profile":profile,"receiver_prior":prior,"pilot_overhead":float(slot.pilot_mask.float().mean())}


def run_both_urban_profiles(n_sc=64, pa_A=2.25, num_slots=20, snr_db=18., seed=42):
    return {p:run_urban_workload(n_sc,p,pa_A,num_slots,snr_db,seed) for p in PROFILES}


def validate_no_labels(model):
    original = generate(64,"sparse12",2.25,1,18.,838100)
    contaminated = replace(original,info_bits=1-original.info_bits,bits=1-original.bits,
                           data_tx=-original.data_tx,h_true=original.h_true*1j,
                           h_time_evol=original.h_time_evol*1j,cfo=original.cfo+.25)
    a,b = _synchronize_slot(original),_synchronize_slot(contaminated)
    def args(z):
        return (z.y_grid,z.pilot_tx,z.pilot_mask,z.noise_var)
    if not all(torch.equal(x,y) for x,y in zip(args(a),args(b))):
        raise AssertionError("Scoring label reached the receiver inputs")
    for method in ("classical","link"):
        out = run_receiver(method,*args(a),NOMINAL,2.25,model)
        other = run_receiver(method,*args(b),NOMINAL,2.25,model)
        if not all(torch.equal(out[key],other[key]) for key in ("llrs","info_bits","h_hat")):
            raise AssertionError(f"Receiver output depends on scoring fields: {method}")
    return "Identical synchronized inputs, LLRs, decoded bits and physical channel estimates after poisoning payload bits, data, true channel and true CFO."


def evaluate(profile,seed,model):
    slot = _synchronize_slot(generate(64,profile,2.25,20,18.,seed))
    inputs = (slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var)
    rows = {}
    for method in ("link","classical"):
        for name,prior in (("nominal",NOMINAL),("matched",profile)):
            if name=="matched" and prior==NOMINAL:
                rows[f"{method}_{name}"]=rows[f"{method}_nominal"]
                continue
            out=run_receiver(method,*inputs,prior,2.25,model)
            rows[f"{method}_{name}"]=score(out,slot)
    # Diagnostic prior: known statistical PA gain from the declared law, never from data truth.
    eff = model(slot.yp_sparse,slot.y_grid,slot.pilot_mask,slot.noise_var)["h_hat"]
    gamma = bussgang_params(A=2.25,alpha=.5,beta=1.,knee=2.)[0]
    def nmse(h):
        return float(((h-slot.h_true).abs().square().sum((-1,-2))/slot.h_true.abs().square().sum((-1,-2))).mean())
    return {"profile":profile,"seed":seed,"slots":20,"methods":rows,
            "neural_pilot_frontend":{"raw_output_physical_nmse":nmse(eff),
                                     "divide_by_declared_bussgang_gain_physical_nmse":nmse(eff/gamma),
                                     "gamma_real":gamma.real,"gamma_imag":gamma.imag}}


def aggregate(rows):
    result={}
    for profile in PROFILES:
        cells=[c for c in rows if c["profile"]==profile]
        result[profile]={}
        for method in cells[0]["methods"]:
            group=[c["methods"][method] for c in cells]
            res={k:sum(r[k] for r in group) for k in ("block_errors","total_blocks","bit_errors","total_bits","nonfinite")}
            nmse=[v for r in group for v in r["per_slot_nmse"]]
            res.update(bler=res["block_errors"]/res["total_blocks"],nmse=sum(nmse)/len(nmse),
                       passing_twenty_slot_batches=sum(r["passes_both"] for r in group),batches=len(group))
            result[profile][method]=res
    return result


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    model=load_neural()
    leakage=validate_no_labels(model)
    data=[]
    for seed in (*DEV_SEEDS,42):
        for profile in PROFILES:
            row=evaluate(profile,seed,model)
            data.append(row)
            print(seed,profile,[(k,r["block_errors"],round(r["nmse"],6)) for k,r in row["methods"].items()],flush=True)
    report={"scenario":"S02, 20 independent 14-symbol slots per profile and seed, N=64, CP16, fd=.005, CFO0, mild PA A=2.25 and alpha=.5, 18 dB, two comb-2 DMRS rows.",
            "design":"Four fresh development seeds first; seed 42 fixed historical audit. Frozen existing waveform receivers and seed 73161 network. Nominal long12 receiver is fixed across both generating profiles. Matched has the generating tap power profile. Both know the transmitter PA map, code and declared noise variance. No realized taps, CFO, payload or labels enter waveform estimation, soft LDPC feedback or demapping.",
            "channel_metric":"Unscaled physical midpoint CFR NMSE after observation-only CP synchronization. PA nonlinearity is applied before forward waveform synthesis in numeric tap fitting. Neural-only gain correction is a separate diagnostic from the known PA law; every receiver NMSE is scored directly without any truth-fitted scalar.",
            "primary":"link_nominal uses the frozen neural initializer, three waveform/extrinsic rounds and the native soft-list decoder. Classical uses the independently frozen waveform receiver and min-sum decoder. The plan tests strict BLER < .01 and NMSE < .05 per 20-slot workload.",
            "leakage_check":leakage,"seeds":list(DEV_SEEDS),"source_sha256":digest(Path(__file__)),
            "checkpoint_sha256":digest(CHECKPOINT),
            "dependencies_sha256":{f:digest(HERE/f) for f in ("next_classical.py","stage3_link.py","stage3_link_decoder.py")},
            "development":aggregate(data[:-2]),"historical":aggregate(data[-2:]),"records":data}
    dest=HERE/"results"/"stage3_urban.json"
    temp=dest.with_suffix(".tmp")
    temp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    temp.replace(dest)


if __name__=="__main__":
    run()
