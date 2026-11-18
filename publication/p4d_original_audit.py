"""Read-only post-run P4-D artifact audit, independent of the repository's Python modules."""
import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import re
import sys
from fractions import Fraction
from pathlib import Path


OFFICIAL = tuple(range(74001, 74011))
DEVELOPMENT = tuple(range(71461, 71471))
CONDITIONS = ("id", "ood_doppler", "ood_delay")
EM = "emfloatformer2d"
BASE = ("lmmse_2d_fixed", "idd_pa_2d_fixed")
METHODS = (EM, *BASE, "lmmse_2d", "idd_pa_2d")
FIELDS = ("bit_errors", "total_bits", "block_errors", "total_blocks", "uncoded_bit_errors", "total_uncoded_bits")
HEADER = ("seed", "condition", "method", *FIELDS)
ACTIVE_FREEZE = "c021f4ad4251a796125143f74405ae577bb0ff77a28e4dc805a49eff39c15036"
FULL = {"steps": 600, "batch": 64, "microbatch": 4, "eval_batch": 4, "eval_batches": 8}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def obj(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, f"duplicate JSON key {name}")
        result[name] = value
    return result


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=obj)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(x):
    require(type(x) is int and x >= 0, f"expected nonnegative integer: {x!r}")
    return x


def as_csv(records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HEADER)
    writer.writeheader()
    for record in records:
        for row in record["rows"]:
            writer.writerow({"seed": record["seed"], **row})
    output.seek(0)
    return output


def parse_csv(stream, seeds, denominator):
    csv_input = csv.DictReader(stream)
    require(tuple(csv_input.fieldnames or ()) == HEADER, "bad CSV header, duplicate, or missing fields")
    rows = {}
    for i, row in enumerate(csv_input, 2):
        require(None not in row and None not in row.values(), f"CSV row {i} has wrong width")
        for name in ("seed", *FIELDS):
            require(re.fullmatch(r"0|[1-9][0-9]*", row[name]) is not None, f"CSV row {i}: noninteger {name}")
            row[name] = int(row[name])
        key = row["seed"], row["condition"], row["method"]
        require(key not in rows, f"duplicate CSV row {key}")
        b, n, e, k = row["block_errors"], row["total_blocks"], row["bit_errors"], row["total_bits"]
        require(n == denominator and k == 64*n and row["total_uncoded_bits"] == 128*n,
                f"{key}: incorrect block, decoded, or raw denominator")
        require(b <= n and b <= e <= 64*b and row["uncoded_bit_errors"] <= 128*n,
                f"{key}: physically inconsistent error counts")
        rows[key] = {k: row[k] for k in FIELDS}
    expected = set(itertools.product(seeds, CONDITIONS, METHODS))
    require(set(rows) == expected, "incomplete or foreign seed / condition / receiver")
    return rows


def tests_and_holm(rows, seeds, n):
    tests = []
    for condition, baseline in itertools.product(CONDITIONS, BASE):
        delta = [rows[s, condition, baseline]["block_errors"]-rows[s, condition, EM]["block_errors"] for s in seeds]
        # Enumerate all signed subsets, retaining multiplicity from zero deltas and inclusive ties.
        threshold = abs(sum(delta))
        extreme = 0
        for mask in range(1 << len(seeds)):
            statistic = sum(d if mask & (1 << i) else -d for i,d in enumerate(delta))
            extreme += abs(statistic) >= threshold
        p = Fraction(extreme, 1 << len(seeds))
        tests.append({"condition":condition, "baseline":baseline, "baseline_minus_em_block_errors":delta,
                      "denominator_per_seed":n, "mean_bler_reduction":sum(delta)/(len(seeds)*n),
                      "exact_p_value":float(p)})
    bound = Fraction(0)
    for rank,i in enumerate(sorted(range(len(tests)), key=lambda j:tests[j]["exact_p_value"])):
        bound = min(Fraction(1), max(bound, (len(tests)-rank)*Fraction(tests[i]["exact_p_value"])))
        tests[i].update(holm_p_value=float(bound), reject_at_0_05=bound < Fraction(1,20))
    return tests


def score(records, rows, summary, seeds, n):
    require([r["seed"] for r in records] == list(seeds),"missing, duplicate, or unordered seed record")
    for record in records:
        for row in record["rows"]:
            for k in FIELDS:
                integer(row[k])
    require(parse_csv(as_csv(records),seeds,n)==rows, "JSON receiver counts differ from standalone CSV")
    tests = tests_and_holm(rows,seeds,n)
    require(summary["primary_tests"] == tests, "seed contrasts, directions, magnitudes, or Holm results disagree")
    require(summary["exact_p_value"] == tests[0]["exact_p_value"], "top-level p-value differs")
    for k,attr in (("total_bits","total_bits"),("total_bit_errors","bit_errors"),("total_blocks","total_blocks"),
                   ("total_block_errors","block_errors")):
        require(integer(summary[k]) == sum(r[attr] for key,r in rows.items() if key[2]==EM),f"summary {k} differs")
    return tests


def checkpoints(folder, summary, freeze):
    import torch
    seen = set()
    for record in summary["records"]:
        seed = record["seed"]
        require(record == load(folder/str(seed)/"counts.json") and record["freeze"] == freeze,
                f"seed {seed}: counts or freeze mismatch")
        state = torch.load(folder/str(seed)/"training.pt", map_location="cpu", weights_only=True)
        require(state["seed"]==seed and state["step"]==600 and state["budget"]==FULL and state["freeze"]==freeze,
                f"seed {seed}: training checkpoint is incomplete or belongs to different run")
        h = hashlib.sha256()
        for name,tensor in sorted(state["model"].items()):
            meta = json.dumps([name,str(tensor.dtype),list(tensor.shape)],ensure_ascii=False,indent=2)+"\n"
            h.update(meta.encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        require(h.hexdigest()==record["weights_sha256"] and h.hexdigest() not in seen,
                f"seed {seed}: missing, repeated or inconsistent final weights")
        seen.add(h.hexdigest())


def factorial(folder,freeze):
    summary = load(folder/"factorial/summary.json")
    mark = load(folder/"factorial_replay.json")
    require(mark=={"freeze":freeze,"summary_sha256":digest(folder/"factorial/summary.json"),
                   "seeds":10,"independently_retrained_cells":80,"matched_gain_observations":240},
            "factorial replay counts or digest mismatch")
    require(summary["freeze"]==freeze and summary["seed_count"]==10 and summary["arm_count"]==80 and
            {r["factor"] for r in summary["tests"]}=={"gauge","hermitian","joint"} and len(summary["tests"])==3,
            "factorial summary incomplete")
    for seed,arm in itertools.product(OFFICIAL,range(1,9)):
        entry=load(folder/"factorial"/str(seed)/f"arm_{arm}.json")
        row=entry["row"]
        require(entry["freeze"]==freeze and (row["seed"],row["arm_id"])==(seed,arm), "wrong factorial cell")
        require([g["scale"] for g in row["gains"]]==[1/64,1.,64.] and row["total_bits"]==36864,"factorial gain grid")
        require(all(integer(g["total_bits"])==12288 and integer(g["bit_errors"])<=12288
                    and g["ber"]==g["bit_errors"]/12288 for g in row["gains"]), "factorial gain denominator")
        require(row["bit_errors"]==sum(g["bit_errors"] for g in row["gains"]) and row["ber"]==row["bit_errors"]/36864,
                "factorial pooled BER mismatch")


def joint(folder,freeze):
    j=load(folder/"joint_replay.json")
    files=("summary.json","raw_by_seed.csv","coded_replay.json","factorial/summary.json")
    require(j=={"freeze":freeze,"verified_artifacts":{f:digest(folder/f) for f in files},"coded_seeds":10,
               "coded_cells":150,"coded_contrasts":6,"independently_retrained_factorial_cells":80,
               "factorial_gain_cells":240,"factorial_contrasts":3}, "joint replay identity, activity count or digest mismatch")
    require(load(folder/"coded_replay.json")=={"freeze":freeze,"csv_rows":150,"contrasts_reproduced":6},
            "coded replay incomplete")
    factorial(folder,freeze)


def e2e(root, path, seal):
    d=load(path)
    require(d["suite"]=="tests.test_p4_e2e.suite" and d["run"]==d["passed"]==253 and not any(d[k] for k in
            ("failures","errors","skips")), "final 253-case run has unfinished, failed, or skipped cases")
    expected=set()
    for f in sorted((root/"tests/e2e").glob("test_tier*.py")):
        rel=f.relative_to(root).as_posix()
        require(digest(f)==seal["files"][rel]==d["sources"][rel],f"test source drift: {rel}")
        for cls in ast.parse(f.read_bytes()).body:
            if isinstance(cls,ast.ClassDef):
                expected.update(".".join([rel[:-3].replace("/","."),cls.name,m.name]) for m in cls.body
                                if isinstance(m,ast.FunctionDef) and m.name.startswith("test_"))
    require(len(expected)==len(d["passed_cases"])==253 and set(d["passed_cases"])==expected,"missing or repeated E2E IDs")
    require(d["historical_checkout"]==seal["historical_checkout"],"final E2E historical checkout differs from freeze")
    required={p for p in seal["files"] if p.endswith(".py") and p.startswith(("p4/","tests/")) or
              p.startswith("p4/results/") and p.endswith(".pt") or p in ("PROJECT.md","p4/P4D_PROTOCOL.md","p4/p4cd_factorial_protocol.json")}
    require(required<=d["sources"].keys(),"final E2E source inventory incomplete")
    for p,h in d["sources"].items():
        f=(root/p).resolve(); f.relative_to(root)
        require(digest(f)==h and (p not in seal["files"] or seal["files"][p]==h),f"final E2E source differs: {p}")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("repo",type=Path)
    cli.add_argument("--rehearsal",action="store_true",help="Check only stored development records through an in-memory CSV")
    cli.add_argument("--final-e2e",type=Path,help="Read this post-formal 253-test evidence file instead of auto-detection")
    args=cli.parse_args()
    root=args.repo.resolve(); dest=root/"p4/results/p4d"
    if args.rehearsal:
        before=load(root/"p4/results/stage3_protocol_rehearsal.json")
        require(before["seeds"]==list(DEVELOPMENT) and before["coded_budget"]=={
                "steps":2,"batch":4,"microbatch":2,"eval_batch":2,"eval_batches":1}, "wrong development design")
        s=before["coded"]; rows=parse_csv(as_csv(s["records"]),DEVELOPMENT,26)
        tests=score(s["records"],rows,s,DEVELOPMENT,26)
        print(json.dumps({"status":"DEVELOPMENT_REHEARSAL_ONLY","rows":len(rows),"contrasts":tests},indent=2))
        return 0
    if args.final_e2e:
        target=args.final_e2e.resolve()
    else:
        choices=(dest/"final_e2e.json",root/"p4/results/stage3_final_e2e.json",root/"p4/results/final_e2e.json")
        found=[p for p in choices if p.is_file()]
        require(len(found)<=1 or len({digest(p) for p in found})==1,"several conflicting final E2E artifacts; specify --final-e2e")
        target=found[0] if found else choices[0]
    plan={"freeze":dest/"P4D_FREEZE.json","csv":dest/"raw_by_seed.csv","summary":dest/"summary.json",
          "joint_replay":dest/"joint_replay.json","final_e2e":target}
    pending=[k for k,p in plan.items() if not p.is_file()]
    verified=[]; tests=None
    if plan["freeze"].is_file():
        freeze=digest(plan["freeze"]); require(freeze==ACTIVE_FREEZE,"formal files are not the active pre-outcome freeze")
        seal=load(plan["freeze"])
        for p,h in seal["files"].items():
            f=(root/p).resolve(); f.relative_to(root)
            require(digest(f)==h,f"frozen dependency changed: {p}")
        verified.append("frozen source and gate digests")
    if plan["csv"].is_file():
        with plan["csv"].open(encoding="utf-8",newline="") as f:
            rows=parse_csv(f,OFFICIAL,416)
        tests=tests_and_holm(rows,OFFICIAL,416)
        verified.append("150 raw CSV records, six independent 1024-sign tests and joint Holm")
        if plan["summary"].is_file() and plan["freeze"].is_file():
            summary=load(plan["summary"]); require(summary["stage"]=="P4-D" and summary["freeze_sha256"]==freeze,
                                                   "coded summary freeze mismatch")
            score(summary["records"],rows,summary,OFFICIAL,416)
            checkpoints(dest,summary,freeze)
            verified.append("ten 600-step final weight hashes and counts, summary and all six directions/magnitudes/p-values")
    if plan["joint_replay"].is_file():
        require(not any(k in pending for k in ("freeze","csv","summary")),"joint marker precedes required raw artifacts")
        joint(dest,freeze); verified.append("physical count/retraining completion markers, 80 factorial cells and 240 gain rows")
    if target.is_file():
        require("freeze" not in pending,"final E2E marker without source freeze")
        e2e(root,target,seal); verified.append("253 unique fully passing post-formal E2E cases and source identities")
    print(json.dumps({"status":"PENDING" if pending else "COMPLETE","verified":verified,"awaiting":pending,
                      "final_e2e_path":str(target),"primary_tests_from_csv":tests},indent=2,ensure_ascii=False))
    return 2 if pending else 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError) as error:
        print(f"INVALID OR UNREADABLE EVIDENCE: {error}",file=sys.stderr)
        sys.exit(1)
