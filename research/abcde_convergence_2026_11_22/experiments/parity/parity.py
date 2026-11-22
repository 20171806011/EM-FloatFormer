"""Finite-state per-target receiver. Pure Python 3, no numerical dependency."""
import math


def lse(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def normalized(logs):
    z = lse(logs)
    return [math.exp(a-z) for a in logs]


def posterior(logjoint, state_prior, trans, target, tau, scheme):
    """logjoint[row][carrier][state][symbol] includes the symbol prior.
    rebuild uses a logjoint already constructed with target-excluded variances.
    """
    if scheme not in ("original", "explicit", "rebuild") or tau <= 0:
        raise ValueError("invalid scheme or tau")
    s0, k0 = target
    S, R = len(logjoint), len(state_prior)
    ell = [[[lse(joints) for joints in states] for states in row] for row in logjoint]
    emit = []
    for s in range(S):
        emit.append([sum(e[r] for k,e in enumerate(ell[s])
                         if scheme == "original" or (s,k) != (s0,k0))/tau for r in range(R)])
    fwd = [math.log(v) for v in state_prior]
    for s in range(s0):
        fwd = [lse([fwd[r]+emit[s][r]+math.log(trans[r][u]) for r in range(R)]) for u in range(R)]
    bwd = [0.0]*R
    for s in range(S-1,s0,-1):
        bwd = [lse([math.log(trans[r][u])+emit[s][u]+bwd[u] for u in range(R)]) for r in range(R)]
    lw = [fwd[r]+emit[s0][r]+bwd[r] for r in range(R)]
    return finish(lw, logjoint[s0][k0])


def finish(lw, local):
    """Exact target readout shared with the optional pinned NumPy adapter."""
    R = len(lw)
    marginal = [lse(u) for u in local]
    n = len(local[0])
    bits = int(round(math.log(n)/math.log(2.0)))
    bit_idxs = [[i for i in range(n) if (i>>(bits-1-b))&1] for b in range(bits)]
    branch_lp = [[lse([local[r][i] for i in ii])-marginal[r] for ii in bit_idxs] for r in range(R)]
    symbol = [lse([lw[r]+local[r][i] for r in range(R)]) for i in range(n)]
    norm = lse(symbol)
    logpb = [[lse([symbol[i] for i in range(n) if i not in ii])-norm,
              lse([symbol[i] for i in ii])-norm] for ii in bit_idxs]
    return {"ell":marginal, "weight":normalized(lw),
            "effective":normalized([lw[r]+marginal[r] for r in range(R)]),
            "branch_bit":[[math.exp(v) for v in row] for row in branch_lp],
            "p1":[math.exp(pair[1]) for pair in logpb],
            "hard":[int(pair[1]>pair[0]) for pair in logpb],
            "logpb":logpb}


QAM=[complex(2*(i//4)-3,2*(i%4)-3)/math.sqrt(10.0) for i in range(16)]


def fit(y, target, omit):
    """B reduced fixture: known gain energy=1, noise floor=.12, replicated edges.
    Omit the target and each of its replicated copies before averaging.
    """
    S,K=len(y),len(y[0])
    nv=[]
    for s in range(S):
        row=[]
        for k in range(K):
            energies=[]
            for ds in (-1,0,1):
                for dk in (-1,0,1):
                    a=min(S-1,max(0,s+ds))
                    b=min(K-1,max(0,k+dk))
                    if not omit or (a,b)!=target:
                        energies.append(abs(y[a][b])**2)
            power=sum(energies)/len(energies)
            row.append(min(20.0,max(.12,power-1.0)) if power/1.12>1.75 else .12)
        nv.append(row)
    return nv


def observe(y, h, variance):
    return [[[[-abs(z-hr*x)**2/v-math.log(16*math.pi*v) for x in QAM] for hr in h]
              for z,v in zip(yr,vr)] for yr,vr in zip(y,variance)]


def qam_decode(frame, target, tau, scheme):
    y = [[complex(*z) for z in row] for row in frame["y"]]
    h = [complex(*z) for z in frame["h"]]
    v = fit(y,target,scheme=="rebuild")
    rec = posterior(observe(y,h,v),frame["prior"],frame["transition"],target,tau,scheme)
    rec["variance"]=v
    return rec


def discrete_a(y, target, scheme):
    """Task A supplied discrete four-symbol model; observations are paired by row."""
    p = [[1,2,6,9],[8,5,2,1]]
    joints = [[[[math.log((n if v else 10-n)/40.0) for n in branch] for branch in p]
               for v in y[i:i+2]] for i in range(0,len(y),2)]
    return posterior(joints,[.6,.4],[[.9,.1],[.3,.7]],target,1,scheme)

def audit_discrete(cases):
    checks = []
    for y in cases:
        records = []
        for j in range(len(y)):
            pos=(j//2,j%2)
            records.append([discrete_a(y,pos,"original"),discrete_a(y,pos,"explicit")])
        checks.append(records)
    return checks

def run_qam(fixture):
    import time
    clock = getattr(time, "perf_counter", time.time)
    modes = ["original","explicit","rebuild"]
    results=[]
    # Warm each code path first. Timed calls include fitting, table construction,
    # smoothing and readout; exclude fixture loading, checking and serialization.
    first=fixture["frames"][0]
    for tau in (1.0,6.0):
        for mode in modes:
            qam_decode(first,(0,0),tau,mode)
    for idx,frame in enumerate(fixture["frames"]+[fixture["unlabelled"]]):
        times = {}
        for t in (1,6):
            times[str(t)]={m:0.0 for m in modes}
        targets=[(0,1)] if "symbol" not in frame else [(s,k) for s in range(2) for k in range(3)]
        entries=[]
        for n,target in enumerate(targets):
            for tau in (1,6):
                trace={}
                off=(idx+n+(tau==6))%3
                for scheme in modes[off:]+modes[:off]:
                    start=clock()
                    trace[scheme]=qam_decode(frame,target,tau,scheme)
                    times[str(tau)][scheme]+=clock()-start
                entries.append({"target":list(target),"tau":tau,"mode":trace})
        results.append({"id":frame["id"],"stratum":frame["stratum"],"entries":entries,"seconds":times})
    return results

def summarize(fixture, records):
    modes=["original","explicit","rebuild"]
    rows=[]
    pairs=[]
    frames=fixture["frames"]
    for group in ("thermal","localized_burst","all"):
        tagged=[(f,r) for f,r in zip(frames,records) if group=="all" or f["stratum"]==group]
        for tau in (1,6):
            stats={mode:[0,0.0,0.0] for mode in modes}
            compare=[("original","explicit"),("original","rebuild"),("explicit","rebuild")]
            changed={(first,second):[0,0] for first,second in compare}
            denom=0
            for f,r in tagged:
                for mode in modes:
                    stats[mode][2]+=r["seconds"][str(tau)][mode]
                for record in r["entries"]:
                    if record["tau"]!=tau: continue
                    s,k=record["target"]
                    truth=f["symbol"][s][k]
                    for b in range(4):
                        y=(truth>>(3-b))&1
                        denom+=1
                        err={}
                        for mode in modes:
                            q=record["mode"][mode]
                            err[mode]=int(q["hard"][b]!=y)
                            stats[mode][0]+=err[mode]
                            stats[mode][1]-=q["logpb"][b][y]
                        for first,second in compare:
                            if err[first]!=err[second]:
                                changed[(first,second)][err[second]]+=1
            for mode in modes:
                errors,total_loss,elapsed=stats[mode]
                rows.append(dict(stratum=group,tau=tau,mode=mode,frames=len(tagged),
                                 bits=denom,errors=errors,ber=errors/float(denom),
                                 bit_nll_nats=total_loss/denom,seconds=elapsed))
            for first,second in compare:
                rescued,regressed=changed[(first,second)]
                assert stats[second][0]-stats[first][0]==regressed-rescued
                pairs.append(dict(stratum=group,tau=tau,first=first,second=second,
                                  bits=denom,flips=rescued+regressed,rescued=rescued,
                                  new_errors=regressed,delta_ber=(regressed-rescued)/float(denom)))
    return dict(rows=rows,pairs=pairs)

if __name__ == "__main__":
    import argparse
    import gzip
    import json
    from pathlib import Path

    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,default=Path(__file__).with_name("fixture.json"))
    parser.add_argument("--a-reference",type=Path,required=True,help="Task A's reference.json")
    parser.add_argument("--out",type=Path,required=True,help="output directory for this run")
    args=parser.parse_args()
    fixture=json.loads(args.fixture.read_text(encoding="utf-8"))
    ref=json.loads(args.a_reference.read_text(encoding="utf-8"))
    cases=[r["y"] for r in ref["otherReferences"]]+[ref["reference"]["y"]]
    raw={"discrete":audit_discrete(cases),"qam":run_qam(fixture)}
    args.out.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.out/"raw.json.gz","wt",encoding="utf-8") as stream:
        json.dump(raw,stream,allow_nan=False,separators=(",",":"))
    (args.out/"metrics.json").write_text(json.dumps(summarize(fixture,raw["qam"]),indent=2),encoding="utf-8")
