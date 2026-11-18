"""Build or independently verify M3 qualification from recorded public API and original test evidence."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = "p4/results/"
E2E = RESULTS+"stage3_e2e.json"
OBS = RESULTS+"stage3_gate_observations.json"
GATE = RESULTS+"stage3_qualification.json"
BENCH = RESULTS+"stage3_benchmark.json"
ARCHIVE = RESULTS+"p4b_coded_bler.json"
SNAPSHOT = "p4/revisions/pre_stage3_p4.zip"
SNAPSHOT_HASH = "4ebfcf6fb2dd36c7575a3648babd7d4958469e8bc5df9c8fafce8c224371f77a"
UNIT = "p4/tests/test_p4_b_2d_fec.py"
PREFIX = "tests.e2e."
FORMAL = {
    PREFIX+"test_tier1_feature_coverage.TestTier1FeatureCoverage.test_t1_f20_protocol_hash_freeze_verification",
    PREFIX+"test_tier3_cross_feature.TestTier3CrossFeature.test_t3_p20_f18_f20_factorial_ablation_confirmatory_seeds",
    PREFIX+"test_tier4_real_workloads.TestTier4RealWorkloads.test_t4_s08_confirmatory_ten_seed_replication_workload",
}
COUNTS = ("block_errors","total_blocks","bit_errors","total_bits","uncoded_bit_errors","total_uncoded_bits","nonfinite_count")
DOCUMENTS = ("PROJECT.md","p4/P4D_PROTOCOL.md","p4/p4cd_factorial_protocol.json")
SUPPORT = ("stage3_benchmark.json","p4b_coded_bler.json","p4b_gate.json","stage3_urban.json",
           "stage3_cfo.json","stage3_sparse_protocol.json","stage3_sparse_evaluation.json",
           "stage3_sparse_analysis.json","next_neural_freeze.json","stage3_link_protocol.json",
           "stage3_link_hst_results.json","stage3_spaced_results.json")
CONTRACT = {
    "schema":"m3-qualification-v1","observations_schema":"m3-public-observations-v1",
    "original_opaque":253,"original_unit":13,"pre_confirmation_errors_allowed":sorted(FORMAL),
    "S01":"Strict BLER < .05; finite outputs and exact block/information-bit accounting.",
    "S02":"long12 and sparse12 separately: 20 slots x 26 (128,64) codewords, NMSE < .05 and BLER < .01.",
    "S03":"150 Hz: residual phase < 3 degrees, zero wrong carrier-quadrant rows and transitions; preserve raw I/Q errors separately.",
    "S05":"Stable outputs, single comb-4 pilot row; gain over nearest LS >= 4 dB recomputed from its NMSE.",
    "S06":"Strict positive coded block-error reduction over both matched baselines and exact slot sign-flip Holm family p < .05.",
    "S07":"Integer coded gap <= .005; fractional uncoded BER <= DFT8+.005, finite NMSE, no shortcut collapse.",
    "S11":"50 continuous slots and at least 100000 information bits, strong PA at 15 dB, 300 Hz, BLER < .01, exact accounting.",
    "benchmark":"23 historical cells, 115 archived method-cell rows with exact counts and all nine method rows independently totalled.",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tested_paths(root):
    return [p for folder in ("p4","tests") for p in (root/folder).rglob("*.py")] + list(
        (root/RESULTS).glob("*.pt")) + [root/p for p in DOCUMENTS]


def evidence_sources(root=ROOT):
    """Record this before and after collecting public observations; abort collection if it changes."""
    root = Path(root)
    paths = set(tested_paths(root)) | {root/SNAPSHOT,*(root/RESULTS/k for k in SUPPORT)}
    paths.update((root/RESULTS).glob("stage3*protocol.json"))
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def unit_evidence():
    class Result(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.passed = []

        def addSuccess(self,test):
            super().addSuccess(test)
            self.passed.append(test.id())
    result = Result()
    unittest.defaultTestLoader.loadTestsFromName("p4.tests.test_p4_b_2d_fec").run(result)
    return {"run":result.testsRun,"passed_cases":result.passed,"failures":[(t.id(),r) for t,r in result.failures],
            "errors":[(t.id(),r) for t,r in result.errors],"skips":[(t.id(),r) for t,r in result.skipped]}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def integer(n, target=None):
    need(type(n) is int and n >= 0 and (target is None or n==target),f"integer count {n!r}, expected {target}")
    return n


def number(n):
    need(type(n) in (int,float) and math.isfinite(n),f"nonfinite or nonnumeric observation: {n!r}")
    return n


def same(a,b):
    need(math.isclose(number(a),number(b),rel_tol=1e-6,abs_tol=1e-10),f"{a} disagrees with independent recount {b}")


def read(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            need(k not in out,f"duplicate JSON field: {k}")
            out[k]=v
        return out
    return json.loads(path.read_bytes(),object_pairs_hook=unique)


def original_ids(root):
    need(sha(root/SNAPSHOT)==SNAPSHOT_HASH,"pre-stage3 source archive changed")
    out={"opaque":set(),"unit":set()}
    with zipfile.ZipFile(root/SNAPSHOT) as z:
        tests=[p for p in z.namelist() if p.startswith("tests/") and p.endswith(".py")]
        for path in tests+[UNIT,ARCHIVE,RESULTS+"p4b_gate.json"]:
            src=z.read(path)
            need(src==(root/path).read_bytes(),f"historical source or baseline differs: {path}")
            if path!=UNIT and not path.startswith("tests/e2e/test_tier"):
                continue
            module=path[:-3].replace("/",".")
            names=[f"{module}.{cls.name}.{f.name}" for cls in ast.parse(src).body if isinstance(cls,ast.ClassDef)
                   for f in cls.body if isinstance(f,ast.FunctionDef) and f.name.startswith("test_")]
            out["unit" if path==UNIT else "opaque"].update(names)
    need(len(out["opaque"])==253 and len(out["unit"])==13,"original test inventory mismatch")
    return out


def check_e2e(data,inventory,root):
    from .confirmatory_protocol import _historical_digest
    need(data["suite"]=="tests.test_p4_e2e.suite" and data["run"]==253,"wrong suite or test count")
    sources=data["sources"]
    expected={p.relative_to(root).as_posix():sha(p) for p in tested_paths(root)}
    need(sources==expected,"E2E tested sources/weights differ: "+", ".join(
        p for p in sorted(sources.keys()|expected.keys()) if sources.get(p)!=expected.get(p)))
    need(data["historical_checkout"]==_historical_digest(root),"pre-P4 checkout differs from E2E run")
    status=data["passed_cases"]
    errors=data["errors"]
    need(not data["failures"] and not data["skips"],"E2E has an assertion failure or skip")
    need(type(errors) is list and all(type(r) in (list,tuple) and len(r)==2 for r in errors),"bad E2E error rows")
    ids=status+[v[0] for v in errors]
    need(len(ids)==len(set(ids)) and set(ids)==inventory,"missing, duplicate or unknown E2E case IDs")
    integer(data["passed"],len(status))
    need(set(k for k,_ in errors)==FORMAL and len(status)==250,"wrong pre-P4-D missing-case set")
    for case,reason in errors:
        last=reason.strip().splitlines()[-1]
        need(last.startswith("FileNotFoundError:") and "P4D_FREEZE.json" in last,
             f"unrelated formal-case exception {case}: {last}")
    return {"passed":len(status),"dependency_cases":sorted(FORMAL)}


def check_units(data,inventory):
    names=data["passed_cases"]
    need(data["run"]==13 and len(names)==len(set(names))==13 and set(names)==inventory
         and all(data[k]==[] for k in ("failures","errors","skips")),"missing, repeated, failed or skipped M3 unit")
    return {"passed":13}


def counts(r, *, blocks=None, width=None):
    n=integer(r.get("total_blocks",r.get("blocks")),blocks)
    b=integer(r["block_errors"])
    k=integer(r["total_bits"])
    e=integer(r.get("bit_errors",r.get("total_bit_errors")))
    need(n>0 and k>=n and k%n==0 and b<=n and b<=e<=b*(k//n),"invalid information/word totals")
    if width is not None:
        integer(k,width*n)
    same(r["bler"],b/n)
    for key,total,maximum in (("block_errors_per_slot",b,n),("bit_errors_per_slot",e,k)):
        if key in r:
            need(sum(integer(x) for x in r[key])==total and total<=maximum,f"inconsistent {key}")
    return b,n,e,k


def check_s01(r,root):
    b,n,e,k=counts(r,blocks=4,width=832)
    integer(r.get("nonfinite_count",r.get("nonfinite")),0)
    need(r["bler"]<.05 and r["propagation"]=="native" and r["arm"]=="long_slot_spread" and r["seed"]==42,
         "S01 independent-slot acceptance or threshold failed")
    same(r["fd_hz"], 3.e9*(300/3.6)/299792458)
    for field,value in (("physical_interleaver_span_slots",4),("codeword_length",1664),("packets64",52),
                        ("raw_bits",6656)):
        integer(r[field],value)
    records=read(root/RESULTS/"stage3_link_hst_results.json")
    need(records["protocol"]==sha(root/RESULTS/"stage3_link_hst_protocol.json"),"S01 experiment changed")
    panel=[v for v in records["rows"] if v["arm"]==r["arm"] and v["propagation"]=="native"
           and math.isclose(v["doppler_hz"],r["fd_hz"])]
    same(sum(v["blocks"] for v in panel if v["seed"]!=42),48)
    legacy=[v for v in panel if v["seed"]==42]
    need(len(legacy)==1 and all(r[field]==legacy[0][field] for field in
         ("blocks","block_errors","bit_errors","total_bits","raw_bit_errors","packet64_errors")),"S01 audited replay mismatch")
    return {"block_errors":b,"blocks":n,"bit_errors":e,"information_bits":k,
            "fresh_errors":sum(v["block_errors"] for v in panel if v["seed"]!=42),"fresh_blocks":48,
            "channel_scope":"independent fading slots, continuous Doppler within each slot"}


def check_s02(r,root):
    need(set(r)=={"long12","sparse12"},"both S02 profiles required")
    audit=read(root/RESULTS/"stage3_urban.json")
    need(audit["source_sha256"]==sha(root/"p4/stage3_urban.py") and
         audit["checkpoint_sha256"]==sha(root/RESULTS/"next_neural_tap_phase_no_channel_73161.pt")
         and set(audit["dependencies_sha256"])=={"stage3_link.py","stage3_link_decoder.py","next_classical.py"}
         and all(sha(root/"p4"/k)==v for k,v in audit["dependencies_sha256"].items()),"S02 evidence identity")
    historical=audit["historical"]
    for profile,v in r.items():
        b,n,e,k=counts(v,blocks=520,width=64)
        integer(v["nonfinite"],0)
        need(v["profile"]==profile and v["bler"]<.01 and 0<=number(v["nmse"])<.05
             and len(v["per_slot_block_errors"])==20 and sum(map(integer,v["per_slot_block_errors"]))==b
             and len(v["per_slot_bit_errors"])==20 and sum(map(integer,v["per_slot_bit_errors"]))==e
             and len(v["per_slot_nmse"])==20,"S02 failed gates or counts")
        same(v["nmse"],sum(map(number,v["per_slot_nmse"]))/20)
        ref=historical[profile]["link_nominal"]
        integer(b,ref["block_errors"]); integer(e,ref["bit_errors"]); same(v["nmse"],ref["nmse"])
    return {k:{"bler":v["bler"],"nmse":v["nmse"]} for k,v in r.items()}


def check_s03(r,root):
    audit=read(root/RESULTS/"stage3_cfo.json")
    need(audit["source_sha256"]==sha(root/"p4/stage3_cfo.py"),"S03 diagnostic source changed")
    ref=audit["blueprint_candidate_entrypoint"]
    for k,value in (("rows",112),("adjacent_row_pairs",104),("sign_count",6656),("doppler_hz",150),
                    ("wrong_carrier_quadrant_rows",0),("carrier_quadrant_transitions",0),("quadrant_flips",0)):
        same(r[k],value)
    e=integer(r["iq_sign_errors"],ref["quadrant_flips"])
    integer(r["unobservable_rows"],0)
    integer(ref["unobservable_rows"],0)
    integer(r["wrong_symbol_quadrants"],ref["wrong_symbol_quadrants"])
    same(r["residual_phase_deg"],ref["residual_phase_deg"])
    need(0<=r["residual_phase_deg"]<3 and e<=r["sign_count"],"S03 residual or I/Q accounting failed")
    need(r["checkpoint_sha256"]==sha(root/RESULTS/"next_neural_tap_phase_no_channel_73161.pt"),"S03 wrong fit")
    return {"residual_phase_deg":r["residual_phase_deg"],"iq_sign_errors":e,"carrier_quadrant_rows":0}


def check_s05(r,root):
    need(r["stable"] is True and r["receiver"]=="SparseUnrolledEM" and r["training_seed"]==73651
         and number(r["nmse_ls"])>0 and number(r["nmse_2d"])>0,"S05 fit or observations invalid")
    gain=10*math.log10(r["nmse_ls"]/r["nmse_2d"])
    same(r["gain_over_ls_db"],gain); same(r["overhead"],8/448)
    need(gain>=4,"S05 gain < 4 dB")
    analysis=read(root/RESULTS/"stage3_sparse_analysis.json")
    need(analysis["raw_file_sha256"]==sha(root/RESULTS/"stage3_sparse_evaluation.json") and
         analysis["protocol_sha256"]==sha(root/RESULTS/"stage3_sparse_protocol.json") and
         "stage3_sparse_full_73651.pt" in analysis["checkpoint_sha256"],"S05 raw evidence or checkpoint identity")
    same(analysis["gate_default_first_seed"]["gain_db"],gain)
    for name,checksum in analysis["checkpoint_sha256"].items():
        need(checksum==sha(root/RESULTS/name),"S05 fit differs from independent recount")
    return {"gain_db":gain}


def check_s06(r):
    need(r["outputs_finite"] is True and r["reduction_vs_both"] is True,"S06 direction or finiteness")
    slots=integer(r["slots"]); need(1<=slots<=1000,"invalid number of paired slots")
    n=integer(r["total_blocks_per_method"],slots*13)
    integer(r["info_bits_per_method"],n*64)
    data=r["block_errors_per_slot"]
    for name,values in data.items():
        bits=r["bit_errors_per_slot"][name]
        need(len(values)==len(bits)==slots and all(0<=integer(v)<=13 and v<=integer(e)<=64*v
             for v,e in zip(values,bits)),"S06 integer accounting")
    reference=data[r["primary_receiver"]]
    raw={}
    for key in ("2D-LMMSE","2D-IDD-PA"):
        delta=[b-a for a,b in zip(reference,data[key])]
        need(sum(delta)>0,"S06 candidate does not improve on "+key)
        counts_by_sum=Counter({0:1})
        for d in delta:
            next_sums=Counter()
            for s,c in counts_by_sum.items():
                next_sums[s+d]+=c; next_sums[s-d]+=c
            counts_by_sum=next_sums
        p=sum(c for s,c in counts_by_sum.items() if abs(s)>=abs(sum(delta)))/2**slots
        same(p,r["contrasts"][key]["two_sided_p"]); same(sum(delta),r["contrasts"][key]["baseline_minus_model"])
        raw[key]=p
    first,second=sorted(raw,key=raw.get)
    adj=min(1.,2*raw[first]); p=max(adj,raw[second])
    same(r["contrasts"][first]["holm"],adj); same(r["contrasts"][second]["holm"],p); same(r["p_value"],p)
    need(p<.05,"S06 Holm p >= .05")
    return {"holm_max":p,"slots":slots}


def check_s07(r):
    f=r["fractional"]
    n=integer(f["uncoded_bits"],integer(f["slots"])*1664)
    need(n>0,"S07 empty fractional draw")
    err={k:integer(f["bit_errors"][k]) for k in ("model","dft8_linear","dft8_tap_phase")}
    need(all(v<=n for v in err.values()) and all(number(f["channel_nmse"][k])>=0 for k in err),"S07 invalid errors or NMSE")
    gap=(err["model"]-err["dft8_linear"])/n; same(f["ber_minus_dft8"],gap)
    for key in ("id","ood"):
        mistakes=integer(r["integer_decoded_errors"][key]); denom=integer(r["integer_decoded_bits"][key])
        need(0<denom and mistakes<=denom,"S07 integer counts"); same(r["ber_"+key],mistakes/denom)
    same(r["ber_gap"],abs(r["ber_id"]-r["ber_ood"]))
    need(r["shortcut_collapse"] is False and number(r["shift_diff"])<=1 and r["ber_ood"]<=.1
         and r["ber_gap"]<=.005 and gap<=.005,"S07 coded or fractional gate")
    return {"fractional_ber_gap":gap,"integer_coded_gap":r["ber_gap"]}


def check_s11(r):
    b,n,e,k=counts(r,width=64)
    need(r["bler"]<.01 and r["exit_code"]==0 and integer(r["nonfinite"])==0 and k>=100000,"S11 targets")
    for name,expected in (("num_slots",50),("n_symb",14),("cp",16),("pa_A",1.25),("alpha",.5),
                          ("snr_db",15),("doppler_hz",300),("continuous_ofdm_symbols",700)):
        same(r[name],expected)
    need(len(r["block_errors_per_slot"])==len(r["bit_errors_per_slot"])==50,"S11 per-slot counts")
    return {"block_errors":b,"blocks":n,"info_bit_errors":e,"bits":k}


def check_benchmark(root):
    old=read(root/ARCHIVE); new=read(root/BENCH)
    need(new["archive_sha256"]==sha(root/ARCHIVE) and new["source_sha256"]==sha(root/"p4/stage3_benchmark.py"),
         "benchmark archive or runner changed")
    methods={"EMFloatFormer2D","2D-LMMSE","2D-IDD-PA","2D-LMMSE-fixed","2D-IDD-PA-fixed"}
    fits={f"tap_phase_no_channel_{s}" for s in (73161,73162,73163)}
    names={"p4b_seed2026_state.pt",*(f"next_neural_{s}.pt" for s in fits)}
    need(set(new["weights_sha256"])==names and all(sha(root/RESULTS/k)==v for k,v in new["weights_sha256"].items()),
         "benchmark fit identities")
    refs={(r["regime"],r["snr_db"],r["method"]):r for r in old["rows"]}
    cells=new["cells"]
    need(len(refs)==len(old["rows"])==115 and len(cells)==23 and len({(r["regime"],r["snr_db"]) for r in cells})==23,
         "missing/duplicate benchmark stratum")
    seen=set()
    all_names=methods|fits|{"tap_phase_pilot_only"}
    need(set(new["all_23"])==all_names,"benchmark summary methods incomplete")
    for c in cells:
        need(set(c["methods"])==all_names,"method missing in historical stratum")
        for name,r in c["methods"].items():
            for k,v in (("total_blocks",312),("total_bits",19968),("total_uncoded_bits",39936),("nonfinite_count",0)):
                integer(r[k],v)
            for k,limit in (("block_errors",13),("bit_errors",832),("uncoded_bit_errors",1664)):
                values=r["per_slot"][k]
                need(len(values)==24 and all(integer(x)<=limit for x in values),"benchmark per-slot accounts")
                integer(r[k],sum(values))
            if name in methods:
                key=(c["regime"],c["snr_db"],name); ref=refs[key]; seen.add(key)
                need(all(ref[k]==r[k] for k in COUNTS) and all(c[k]==ref[k] for k in ("profile","fd","cfo","pa_condition","eval_seed"))
                     and (c["n_symb"],c["n_sc"],c["cp"],c["slots"])==(14,32,16,24),"archive replay differs")
    need(seen==set(refs),"missing historical method/cell")
    for name in all_names:
        for k in COUNTS:
            integer(new["all_23"][name][k],sum(c["methods"][name][k] for c in cells))
    return {"archived_rows":len(seen),"cells":len(cells),"methods":len(all_names)}


def qualify(root=ROOT, *, e2e=E2E, observations=OBS):
    root=Path(root).resolve()
    def rel(value):
        return (root/value).resolve().relative_to(root).as_posix()
    e2e,observations=rel(e2e),rel(observations)
    checks={}; errors={}
    def check(name,operation):
        try:
            result=operation()
            checks[name]=result
            return result
        except (OSError,ValueError,RuntimeError,KeyError,TypeError,IndexError,ZeroDivisionError,zipfile.BadZipFile) as exc:
            errors[name]=f"{type(exc).__name__}: {exc}"
            return None
    inv=check("historical_sources",lambda: {k:sorted(v) for k,v in original_ids(root).items()})
    sources=check("current_hashes",lambda:evidence_sources(root))
    observed=check("public_artifact",lambda:read(root/observations))
    suite=check("e2e_artifact",lambda:read(root/e2e))
    if suite is not None and inv is not None:
        check("opaque_253",lambda:check_e2e(suite,set(inv["opaque"]),root))
    if observed is not None:
        def identity():
            need(observed["schema"]==CONTRACT["observations_schema"] and sources is not None
                 and observed["sources"]==sources,"public observations do not match current source/weight/evidence hashes")
            return True
        check("public_identity",identity)
        if inv is not None:
            check("m3_units_13",lambda:check_units(observed["m3_units"],set(inv["unit"])))
        for k,fn in (("S01",lambda r:check_s01(r,root)),("S02",lambda r:check_s02(r,root)),("S03",lambda r:check_s03(r,root)),
                     ("S05",lambda r:check_s05(r,root)),("S06",check_s06),("S07",check_s07),("S11",check_s11)):
            check(k,lambda k=k,fn=fn:fn(observed["cases"][k]))
    check("benchmark_23",lambda:check_benchmark(root))
    required={"historical_sources","current_hashes","public_artifact","e2e_artifact","opaque_253","m3_units_13",
              "public_identity","benchmark_23","S01","S02","S03","S05","S06","S07","S11"}
    for key in required-checks.keys()-errors.keys():
        errors[key]="prerequisite evidence unavailable"
    for k in ("current_hashes","public_artifact","e2e_artifact","historical_sources"):
        if k in checks:
            checks[k]=True
    provenance={} if sources is None else dict(sources)
    for p in (e2e,observations):
        if (root/p).is_file():
            provenance[p]=sha(root/p)
    return {"schema":CONTRACT["schema"],"stage":"P4-B","passed":not errors,"contract":CONTRACT,
            "e2e":e2e,"observations":observations,"checks":checks,"failed":dict(sorted(errors.items())),
            "evidence_files":provenance}


def validate_gate(gate_path,root=ROOT):
    """Freeze hook: recalculate all checks from current evidence; return verified dependency hashes."""
    root=Path(root).resolve()
    file=(root/gate_path).resolve()
    file.relative_to(root)
    candidate=read(file)
    result=qualify(root,e2e=candidate["e2e"],observations=candidate["observations"])
    if result["passed"] is not True:
        raise ValueError(f"M3 remains unqualified: {result['failed']}")
    need(candidate==result,"gate report does not reproduce from the current evidence")
    return result["evidence_files"]


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path,default=ROOT)
    cli.add_argument("--e2e",default=E2E); cli.add_argument("--observations",default=OBS)
    cli.add_argument("--output",help="Optional: write the status, including failed checks, at this repository-relative path")
    cli.add_argument("--validate",help="Recompute a previously generated passing gate before freezing")
    args=cli.parse_args()
    if args.validate:
        print(f"Verified {len(validate_gate(args.validate,args.root))} dependencies")
    else:
        report=qualify(args.root,e2e=args.e2e,observations=args.observations)
        if args.output:
            dest=(args.root/args.output).resolve()
            dest.relative_to(args.root.resolve())
            need(dest!=(args.root/RESULTS/"p4b_gate.json").resolve(),"historical gate is immutable")
            tmp=dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8"); tmp.replace(dest)
        print(json.dumps({"passed":report["passed"],"checks":report["checks"],"failed":report["failed"]},indent=2))
        raise SystemExit(0 if report["passed"] else 1)
