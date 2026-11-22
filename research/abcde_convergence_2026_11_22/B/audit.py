"""Per-target audit. Requires two pinned source trees, numpy, scipy and torch.
Run --help for NPZ schema and synthetic defaults. No dataset or training loop.
"""
import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax
from numpy.lib.stride_tricks import sliding_window_view


def verify_tree(tree, expected):
    for name, sha in expected.items():
        data = (tree / (name + ".py")).read_bytes()
        header = f"blob {len(data)}\0".encode()
        if hashlib.sha1(header + data).hexdigest() != sha:
            raise ValueError(f"Wrong pinned blob: {tree}/{name}.py")


def private_function(fn, changes, overrides):
    """Reuse the pinned formula in a private namespace. Do not edit imported modules."""
    source = inspect.getsource(fn)
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f"Unmatched source in {fn.__name__}: {before}")
        source = source.replace(before, after)
    scope = {**fn.__globals__, **overrides}
    # Compile under the source module's deferred-annotation convention.
    exec("from __future__ import annotations\n" + source, scope)
    return scope[fn.__name__]


def excluded_bundle(grid, pilots, refs, target, temp, d, f):
    s, k = target
    valid = np.ones((d.N_SYM, len(d.USED_SCS)), dtype=bool)
    valid[s, k] = False
    data = d.DATA_MASK_101 & valid
    # The input placeholder is never counted. Its original value is used only at readout.
    fitted_grid = grid.copy()
    fitted_grid[s, d.USED_SCS[k]] = 0
    noise = private_function(d.estimate_noise_var_from_grid, [
        ("np.abs(y_grid[:, USED_SCS]) ** 2", "np.abs(y_grid[:, USED_SCS][valid]) ** 2"),
        ("sym2_mask = DATA_MASK_101[PILOT_SYM]", "sym2_mask = data[PILOT_SYM]"),
    ], {"valid": valid, "data": data})

    def guard(y_used, h_base, n0_base):
        energy = np.abs(y_used)**2
        def box(a):
            blocks = sliding_window_view(np.pad(a, 1, mode="edge"), (3, 3))
            return blocks.sum(axis=(-2,-1))
        local = box(np.where(valid, energy, 0.0)) / box(valid.astype(float))
        signal = np.abs(h_base)**2
        ratio = local / np.maximum(signal[None, :] + n0_base, 1e-12)
        cap = 40 * max(n0_base, float(np.median(signal)) * .5)
        nv = np.where(ratio > 1.75, np.minimum(cap, np.maximum(n0_base, local-signal)), n0_base)
        return nv, np.log(np.maximum(ratio, 1e-6))

    build = private_function(f.build_shared_hypothesis_bundle, [
        ("dmask = DATA_MASK_101.astype", "dmask = data.astype"),
        ("np.mean(np.maximum(excess_log, 0.0), axis=1)",
         "(np.sum(np.maximum(excess_log, 0.0)*valid, axis=1) / valid.sum(axis=1))"),
    ], {"estimate_noise_var_from_grid": noise, "compute_nbi_burst_variance_guard": guard,
        "valid": valid, "data": data})
    return build(fitted_grid, pilots, refs, ll_temp=temp, use_nbi_guard=True)


def local_table(y, h, v, alphabet, bits, dc=0):
    err = y - dc - h[:, None] * alphabet
    joint = -np.abs(err)**2/v[:, None] - np.log(np.pi*v[:, None]) - np.log(len(alphabet))
    ell = logsumexp(joint, axis=-1)
    pbit = softmax(joint, axis=-1) @ bits
    return ell, pbit, joint


def record(logw, ell, pbit, joint, bits, normalized=False):
    weights = softmax(logw)
    if normalized:
        effective = weights
        sym_log = logsumexp(logw[:, None] + joint - ell[:, None], axis=0)
    else:
        effective = softmax(logw + ell)
        sym_log = logsumexp(logw[:, None] + joint, axis=0)
    llr = np.array([logsumexp(sym_log[bits[:, j]==1])-logsumexp(sym_log[bits[:, j]==0])
                    for j in range(bits.shape[1])])
    prob = expit(llr)
    np.testing.assert_allclose(prob, effective @ pbit, atol=3e-12)
    return dict(candidate=np.arange(len(ell)), local_log_marginal=ell, weight=weights,
                effective_weight=effective, branch_p_bit1=pbit, p_bit1=prob,
                llr=llr, hard=(llr>0).astype(int))


def physical(grid, pilots, refs, target, temp, stay, d, f, check_reference=False, model=None):
    if np.shape(grid)!=(d.N_SYM,d.N_FFT) or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a finite 14x128 complex grid")
    if refs.ndim!=2 or refs.shape[0]<1 or refs.shape[1]!=len(d.USED_SCS) or not np.all(np.isfinite(refs)):
        raise ValueError("refs must be a nonempty finite Mx101 pilot reference bank")
    if pilots.ndim!=1 or len(pilots)==0 or not np.all(np.isin(pilots,d.PILOT_SCS_14)) or np.any(np.diff(pilots)<=0):
        raise ValueError("pilots must be a sorted distinct subset of the 14 physical FFT pilot indices")
    s,k = target
    if not (0<=s<d.N_SYM and 0<=k<len(d.USED_SCS) and d.DATA_MASK_101[s,k]):
        raise ValueError("Target must be a payload (row, USED_SCS index)")
    raw = f.build_shared_hypothesis_bundle(grid, pilots, refs, ll_temp=temp)
    rebuilt = excluded_bundle(grid, pilots, refs, target, temp, d, f)
    true_y = grid[s,d.USED_SCS[k]]
    result = {"target":{"symbol":s,"used_index":k,"fft_bin":int(d.USED_SCS[k]),
                        "y":[true_y.real,true_y.imag]}, "tau":temp,"stay":stay}
    inputs = {}
    for mode,b in [("original",raw),("explicit",raw),("rebuild",rebuilt)]:
        v = np.full(f.R_HYP, b.nv_grid[s,k])
        ell,pbit,joint = local_table(true_y,b.h_bank[:,s,k],v,d.CONST16,d.BITS16)
        score = b.sym_ll.copy()
        if mode=="explicit":
            score[:,s] -= ell
        logw = f.hmm_smooth_log_posterior(score, f.HYP_LOG_PRIOR, temp, stay)
        rec = record(logw[:,s],ell,pbit,joint,d.BITS16)
        result[mode] = {**rec,"nv_base":b.nv_base,"target_variance":v[0],"row_score":score[:,s],
                        "delta_complex_features_max":np.max(np.abs(b.cplx_feats-raw.cplx_feats)),
                        "delta_scalar_features_max":np.max(np.abs(b.scal_feats-raw.scal_feats))}
        inputs[mode] = (b,score,logw)
        if mode=="original":
            result["normalized_full"] = record(logw[:,s],ell,pbit,joint,d.BITS16,True)
            if check_reference:
                import group_b_posterior_fusion as group_b
                idx = np.flatnonzero(np.all(d.DATA_COORDS_101==[s,k],axis=1))[0]
                # C0 referenced only with full visible pilots; labels/oracle are metric placeholders.
                if np.array_equal(pilots,d.PILOT_SCS_14):
                    out = group_b.evaluate_group_b_on_tti(
                        grid,np.zeros((1400,4),int),np.ones(101,complex),0,"C0_raw_ota",refs,
                        ll_temp=temp,p_stay=stay)
                    np.testing.assert_allclose(out["B3_mixture_likelihood"]["llrs"][idx],rec["llr"],atol=3e-10)
                ref = d.mixture_16qam_llrs(raw.y_used,raw.h_bank,
                    np.broadcast_to(raw.nv_grid,raw.h_bank.shape),logw)[s,k]
                np.testing.assert_allclose(rec["llr"],ref,atol=3e-10)
    lw = inputs["original"][2][:,s]
    ell = result["original"]["local_log_marginal"]
    np.testing.assert_allclose(softmax(lw-ell/temp), result["explicit"]["weight"],atol=1e-11)
    if temp==1:
        np.testing.assert_allclose(result["normalized_full"]["p_bit1"],result["explicit"]["p_bit1"],atol=1e-11)
    if model is not None:
        # Freeze the trained callable in eval mode on CPU. The explicit probe holds neural
        # corrections (including phase and noise) fixed, only subtracts ell/tau at the gate.
        import torch
        as_input=lambda a:torch.tensor(a[None],dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            for mode in ("original","explicit","rebuild"):
                b,score,logw=inputs[mode]
                if mode!="explicit":
                    a,phase,scale = model(as_input(b.cplx_feats.real),as_input(b.cplx_feats.imag),
                                          as_input(b.scal_feats),as_input(logw.T))
                    a=a[0,s].numpy().astype(float)
                    phase=phase[0,s].numpy().astype(float)
                    scale=float(np.clip(scale[0,s,0].item(),-1,2))
                    if mode=="original":
                        frozen=(a.copy(),phase.copy(),scale)
                else:
                    a,phase,scale=frozen
                    a=a-ell/temp
                vv=np.full(f.R_HYP,max(b.nv_grid[s,k]*np.exp(scale),1e-8))
                loc=local_table(true_y,b.h_bank[:,s,k]*np.exp(1j*phase),vv,d.CONST16,d.BITS16)
                result[mode]["learned"]=record(a,*loc,d.BITS16)
    return result,inputs


def synthetic(d):
    rng=np.random.default_rng(731)
    shape=(14,128)
    grid=.08*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    gain=1+.15*np.sin(np.arange(101)/9)
    grid[:,d.USED_SCS]+=gain*d.CONST16[rng.integers(16,size=(14,101))]
    grid[2,d.PILOT_SCS_14]=gain[d.PILOT_USED_IDX_14]*d.PILOT_TX_14
    grid[7,d.USED_SCS[45]]=3.6+2.1j
    refs=np.stack([gain,1+.1*np.cos(np.arange(101)/8)])
    return grid,d.PILOT_SCS_14,refs


def check_invariance(grid,pilots,refs,target,temp,stay,d,f,inputs):
    s,k=target
    mod=grid.copy()
    mod[s,d.USED_SCS[k]]=-1.3+.1j
    b=excluded_bundle(mod,pilots,refs,target,temp,d,f)
    ref=inputs["rebuild"][0]
    for field in ("nv_base","nv_grid","h_bank","sym_ll","cplx_feats","scal_feats","base_logw_sr"):
        np.testing.assert_array_equal(getattr(b,field),getattr(ref,field))
    np.testing.assert_array_equal(f.hmm_smooth_log_posterior(b.sym_ll,f.HYP_LOG_PRIOR,temp,stay),
                                  inputs["rebuild"][2])


def pluto(y,bank,target,p):
    if not 0<=target<100:
        raise ValueError("PLUTO target must be payload index 0..99")
    all_y=y[p.PAYLOAD_SLICE]
    a=np.log(np.maximum(bank["alpha"],1e-300))
    tabs=[local_table(z,bank["h_all"][:,26+i],bank["obs_var_r"],p.QPSK_CONST,p.QPSK_BITS,
                      bank["d_post"][:,None]) for i,z in enumerate(all_y)]
    ell,pbit,joint=tabs[target]
    full=a+np.sum([u[0] for u in tabs],axis=0)
    cavity=a+np.sum([u[0] for i,u in enumerate(tabs) if i!=target],axis=0)
    b3a=record(a,ell,pbit,joint,p.QPSK_BITS)
    b3b=record(full,ell,pbit,joint,p.QPSK_BITS,True)
    other=record(cavity,ell,pbit,joint,p.QPSK_BITS)
    np.testing.assert_allclose(b3b["p_bit1"],other["p_bit1"],atol=1e-11)
    repo=p.decode_group_b_variants(y,bank)
    np.testing.assert_allclose(repo["B3b_joint_mixture"]["beta"],b3b["weight"],atol=1e-11)
    for name,rec in [("B3a_sym_mixture",b3a),("B3b_joint_mixture",b3b)]:
        np.testing.assert_array_equal(repo[name]["bits_dec"][target],rec["hard"])
    return {"target_payload_index":target,"frame_symbol":26+target,
            "family":bank["c_idx"],"quadrant":bank["q_idx"],"cfo":bank["g_idx"],
            "B3a":b3a,"B3b":b3b,"cavity_raw":other}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ofdm",type=Path,required=True,help="pinned realdata_ofdm_stress_and_theory directory")
    ap.add_argument("--pluto",type=Path,help="pinned realdata_pluto_qpsk directory; add an 84-branch test")
    ap.add_argument("--frame",type=Path,help="one stressed TTI NPZ: grid[14,128], pilots[visible FFT bins], refs[M,101]")
    ap.add_argument("--target",type=int,nargs=2,default=[7,45],metavar=("ROW","USED_K"))
    ap.add_argument("--crosscheck-js",action="store_true",help="compare the 17-state JS reconstruction with the pinned source at three synthetic targets")
    ap.add_argument("--tau",type=float,default=1)
    ap.add_argument("--stay",type=float,default=.88)
    opts=ap.parse_args()
    if not(opts.tau>0 and 0<opts.stay<1):
        ap.error("tau must be positive, stay must be strictly between 0 and 1")
    verify_tree(opts.ofdm, OFDM_BLOBS)
    sys.path.insert(0,str(opts.ofdm.resolve()))
    import ofdm_data as d
    import hypothesis_frontend as f
    if opts.frame is None:
        grid,pilots,refs=synthetic(d)
        targets=list(dict.fromkeys([tuple(opts.target),(2,39),(13,100)]))
    else:
        with np.load(opts.frame,allow_pickle=False) as frame:
            grid,pilots,refs=(frame[n] for n in ("grid","pilots","refs"))
        targets=[tuple(opts.target)]
    result={"OFDM_commit":OFDM_COMMIT,"PLUTO_commit":PLUTO_COMMIT,"OFDM":[]}
    for t in targets:
        rec,inputs=physical(grid,pilots,refs,t,opts.tau,opts.stay,d,f,opts.frame is None)
        check_invariance(grid,pilots,refs,t,opts.tau,opts.stay,d,f,inputs)
        result["OFDM"].append(rec)
    if opts.crosscheck_js:
        import subprocess
        js=Path(__file__).with_name("fullgrid.js")
        raw=np.array(json.loads(subprocess.check_output(["node",str(js),"--grid"])))
        y_js=raw[...,0]+1j*raw[...,1]
        js_report=json.loads(subprocess.check_output(["node",str(js)]))
        maxerr={}
        for entry in js_report["OFDM"]:
            t=entry["target"]
            rec,_=physical(y_js,d.PILOT_SCS_14,np.ones((1,101),complex),
                           (t["symbol"],t["used_index"]),1,.88,d,f,True)
            for mode in ("original","explicit","rebuild"):
                for field in ("row_score","weight","effective_weight","branch_p_bit1","p_bit1",
                              "local_log_marginal","target_variance","nv_base","delta_scalar_features_max"):
                    pinned=np.asarray(rec[mode][field])
                    twin=np.asarray(entry[mode][field])
                    np.testing.assert_allclose(pinned,twin,rtol=2e-12,atol=3e-10,err_msg=field)
                    maxerr[field]=max(maxerr.get(field,0),float(np.max(np.abs(pinned-twin))))
                np.testing.assert_array_equal(rec[mode]["hard"],entry[mode]["hard"])
        result["JS_source_agreement_max_abs"]=maxerr
    if opts.pluto:
        verify_tree(opts.pluto,PLUTO_BLOBS)
        sys.path.insert(0,str(opts.pluto.resolve()))
        import receivers as p
        rng=np.random.default_rng(19)
        y=.16*(rng.normal(size=126)+1j*rng.normal(size=126))
        y[p.PILOT_SLICE]+=.9*p.S_PIL
        y[p.PAYLOAD_SLICE]+=.9*p.QPSK_CONST[rng.integers(4,size=100)]
        sig=p.estimate_single_frame_pilot_stats(y[p.PILOT_SLICE],.8+.1j,.1)["sigma2_est"]
        priors=[(.8+.1j,.1,-.04j),(.6-.3j,.13,-.05+.02j),(.7+.05j,.06,-.02)]
        bank=p.build_shared_b_candidates(y[p.PILOT_SLICE],sig,priors,p.ReceiverHyperparams())
        result["PLUTO"]=pluto(y,bank,16,p)
    print(json.dumps(result,default=lambda a: np.asarray(a).tolist(),allow_nan=False))


OFDM_COMMIT = "5f88a5be984493bcb6aa164d84253b6f29876eef"
PLUTO_COMMIT = "ade8333fb68126251a83e484b69b821b5d97bcea"
OFDM_BLOBS = {"hypothesis_frontend":"735233d5ad34f77788f7ea2536048eb489be34ae","group_b_posterior_fusion":"dd32790b893bb432376622f2bf7a85561955df0a","ofdm_data":"e2ed1a2e7a71415ef205498bb3f104827a1468ae","group_c_transport_attention":"6dd69afa4a1747e2072bd3c1766ecf08dafa85f0","stress_channel":"898bccc0c4d4bcd1ced2100272c5ea4a4977ea06"}
PLUTO_BLOBS = {"receivers":"4569d3dfdb5e7e1d3a2552cdb30b1c25f72b35c5","waveform_frontend":"a3752a434132f2730e807b215c4487e6e41fbd2b","evaluator":"03c9ee07aa79e6c8dc8e3454b53e4a576a35d0db"}

if __name__ == "__main__":
    main()
