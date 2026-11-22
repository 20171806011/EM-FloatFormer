"""Optional one-RE bridge to B/audit.py and the pinned original; CPU NumPy/SciPy/Torch."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from parity import finish


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit",type=Path,required=True,help="B/audit.py from this package")
    parser.add_argument("--ofdm",type=Path,required=True,help="pinned source directory")
    parser.add_argument("--frame",type=Path,help="one stressed NPZ with grid,pilots,refs; default synthetic")
    parser.add_argument("--target",type=int,nargs=2,default=[7,45])
    parser.add_argument("--tau",type=float,default=6.)
    parser.add_argument("--stay",type=float,default=.8)
    args=parser.parse_args()
    if args.tau<=0 or not 0<args.stay<1: parser.error("tau>0, 0<stay<1 required")
    spec=importlib.util.spec_from_file_location("b_single_target",args.audit)
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.verify_tree(args.ofdm,audit.OFDM_BLOBS)
    sys.path.insert(0,str(args.ofdm.resolve()))
    import ofdm_data as d
    import hypothesis_frontend as h
    if args.frame:
        with np.load(args.frame,allow_pickle=False) as z:
            grid,pilots,refs=(z[n] for n in ("grid","pilots","refs"))
    else:
        grid,pilots,refs=audit.synthetic(d)
    s,k=args.target
    rec,fit=audit.physical(grid,pilots,refs,(s,k),args.tau,args.stay,d,h,
                           check_reference=args.frame is None)
    y=grid[s,d.USED_SCS[k]]
    output={"target":rec["target"],"tau":args.tau,"stay":args.stay,"modes":{}}
    for mode,(b,_,logw) in fit.items():
        v=np.full(h.R_HYP,b.nv_grid[s,k])
        _,_,j=audit.local_table(y,b.h_bank[:,s,k],v,d.CONST16,d.BITS16)
        q=finish(logw[:,s].tolist(),j.tolist())
        for field,key in (("p1","p_bit1"),("ell","local_log_marginal"),("weight","weight"),
                          ("effective","effective_weight"),("branch_bit","branch_p_bit1")):
            np.testing.assert_allclose(q[field],rec[mode][key],rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(q["hard"],rec[mode]["hard"])
        output["modes"][mode]=q
    print(json.dumps(output,default=lambda a:np.asarray(a).tolist(),allow_nan=False))


if __name__=="__main__":
    main()
