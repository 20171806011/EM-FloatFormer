"""Replay frozen physical heads on authorized, hash-verified IQ. Prints aggregate JSON only.
Default: one Dev TTI smoke, same 3 targets under each readout. See README for scopes.
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "B"))
import audit

PTH_SHA = "e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058"
TAU, STAY = 6.0, 0.8
CONDITIONS = ("C0_raw_ota", "C2_cfo_doppler", "C3_phase_jump_jitter")
SMOKE = ((7, 45), (2, 39), (13, 100))


def vector_heads(bundle, logw, d):
    """The frozen-score cavity needs no per-bit HMM run. Correct for any tau>0."""
    s, k = d.DATA_COORDS_101.T
    v = np.maximum(bundle.nv_grid[s,k], 1e-12)
    h = bundle.h_bank[:,s,k]
    y = bundle.y_used[s,k]
    joint = (-np.abs(y[None,:,None] - h[:,:,None]*d.CONST16)**2/v[None,:,None]
             - np.log(16*np.pi*v)[None,:,None])
    ell = logsumexp(joint, axis=2)
    w = logw[:,s]
    def decode(g):
        joint_x = logsumexp(g[:,:,None]+joint, axis=0)
        return np.stack([logsumexp(joint_x[:,one],axis=1)-logsumexp(joint_x[:,zero],axis=1)
                         for one,zero in zip(d.BIT_ONE_IDX,d.BIT_ZERO_IDX)], axis=1)
    return decode(w), decode(w-ell/TAU)


def tally(acc, name, y, ll, baseline):
    y=np.asarray(y, dtype=bool)
    pred=ll>0
    bad=pred!=y
    prior=(baseline>0)!=y
    if ll.shape!=y.shape or ll.shape[1]!=4 or not np.isfinite(ll).all():
        raise ValueError("unaligned or nonfinite 16-QAM bit outputs")
    row=acc.setdefault(name, {key:np.zeros(4,dtype=float if key=="nll_sum" else np.int64)
                              for key in ("bits","errors","nll_sum","rescued","introduced","flips")})
    row["bits"]+=len(y)
    row["errors"]+=bad.sum(axis=0)
    row["rescued"]+=(prior & ~bad).sum(axis=0)
    row["introduced"]+=(~prior & bad).sum(axis=0)
    row["flips"]+=((baseline>0)!=pred).sum(axis=0)
    row["nll_sum"]+=np.logaddexp(0,np.where(y,-ll,ll)).sum(axis=0)


def summarize(row):
    assert np.array_equal(row["flips"],row["rescued"]+row["introduced"])
    return {**{key:value.tolist() for key,value in row.items()},
            "per_position_nll_nats":(row["nll_sum"]/row["bits"]).tolist(),
            "per_position_ber":(row["errors"]/row["bits"]).tolist(),
            "bits_total":int(row["bits"].sum()),"errors_total":int(row["errors"].sum()),
            "ber":float(row["errors"].sum()/row["bits"].sum()),
            "mean_nll_nats":float(row["nll_sum"].sum()/row["bits"].sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofdm",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--scope",choices=("smoke","history-pair","history-strict"),default="smoke")
    args=parser.parse_args()
    audit.verify_tree(args.ofdm,audit.OFDM_BLOBS)
    sys.path.insert(0,str(args.ofdm.resolve()))
    import ofdm_data as d
    import hypothesis_frontend as front
    from stress_channel import condition_seed, apply_stress_condition
    if d.sha256_file(args.dataset)!=PTH_SHA:
        raise ValueError("Expected pinned upstream .pth digest before deserialization.")
    ds=d.load_real_ofdm_dataset(args.dataset)
    if ds.labels.shape!=(742,1400,4) or ds.iq.shape!=(742,14,128):
        raise ValueError("unexpected frame axes")
    refs=front.build_train_reference_bank(ds.iq[:340],ds.sinr_db[:340],m_refs=16)
    inds=[360] if args.scope=="smoke" else list(range(502,742))
    positions=np.array([np.flatnonzero(np.all(d.DATA_COORDS_101==sk,axis=1))[0] for sk in SMOKE])
    if args.scope!="smoke":
        positions=np.arange(len(d.DATA_COORDS_101))
    report={"schema":1,"source_sha256":PTH_SHA,"scope":args.scope,"indices":inds,
            "tau":TAU,"stay":STAY,"raw_uncalibrated_llr":True,
            "target_indices":None if len(positions)==1400 else positions.tolist(),
            "seconds_scope":"aggregate wall time; task timings have different work volumes as specified, not head speedups",
            "conditions":{}}
    for cond in CONDITIONS:
        acc={}
        timer={"shared_full_grid":0.0,"original_and_explicit_full_grid":0.0,
               "strict_refit_and_decode_selected_targets":0.0}
        frames=[]
        for ti in inds:
            pert=apply_stress_condition(ds.iq[ti],cond,seed=condition_seed(ti,cond))
            start=perf_counter()
            b=front.build_shared_hypothesis_bundle(pert.iq,pert.pilot_scs,refs,ll_temp=TAU)
            lw=front.hmm_smooth_log_posterior(b.sym_ll,front.HYP_LOG_PRIOR,TAU,STAY)
            timer["shared_full_grid"]+=perf_counter()-start
            start=perf_counter()
            orig,exp=vector_heads(b,lw,d)
            timer["original_and_explicit_full_grid"]+=perf_counter()-start
            if ti==inds[0]:
                ref=d.mixture_16qam_llrs(b.y_used,b.h_bank,np.broadcast_to(b.nv_grid,b.h_bank.shape),lw)
                np.testing.assert_allclose(orig,ref[tuple(d.DATA_COORDS_101.T)],rtol=1e-10,atol=4e-10)
            orig,exp=orig[positions],exp[positions]
            y=ds.labels[ti,positions]
            heads={"original":orig,"explicit":exp}
            if args.scope!="history-pair":
                start=perf_counter()
                strict=[]
                for j,idx in enumerate(positions):
                    sk=tuple(d.DATA_COORDS_101[idx])
                    b2=audit.excluded_bundle(pert.iq,pert.pilot_scs,refs,sk,TAU,d,front)
                    w2=front.hmm_smooth_log_posterior(b2.sym_ll,front.HYP_LOG_PRIOR,TAU,STAY)
                    s,k=sk
                    args_local=audit.local_table(b.y_used[s,k],b2.h_bank[:,s,k],
                        np.full(front.R_HYP,b2.nv_grid[s,k]),d.CONST16,d.BITS16)
                    strict.append(audit.record(w2[:,s],*args_local,d.BITS16)["llr"])
                    if args.scope=="smoke":
                        check,_=audit.physical(pert.iq,pert.pilot_scs,refs,sk,TAU,STAY,d,front)
                        for m,actual in [("original",orig[j]),("explicit",exp[j]),("rebuild",strict[-1])]:
                            np.testing.assert_allclose(actual,check[m]["llr"],rtol=1e-10,atol=4e-10)
                timer["strict_refit_and_decode_selected_targets"]+=perf_counter()-start
                heads["strict"]=np.array(strict)
            per_frame={}
            for name,ll in heads.items():
                tally(acc,name,y,ll,orig)
                tally(per_frame,name,y,ll,orig)
            frames.append({"tti":ti,"methods":{n:summarize(r) for n,r in per_frame.items()}})
        report["conditions"][cond]={"paired_vs":"original","times":timer,
              "aggregate":{n:summarize(r) for n,r in acc.items()},"frames":frames}
    print(json.dumps(report,allow_nan=False,indent=2))


if __name__=="__main__":
    main()
