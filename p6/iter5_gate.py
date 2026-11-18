"""Check the recorded development candidates and report the pre-freeze decision."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/dev_iter5"


def read(path):
    return json.loads(path.read_bytes())


def main():
    scores = read(OUT / "dev_scores.json")
    suite = read(OUT / "audit_suite.json")
    legacy = read(OUT / "legacy_checks.json")
    invariants = read(OUT / "audit_suite_invariants.json")
    checks = read(OUT / "integrity_after.json")
    if not suite["passed"] or not invariants["passed"] or not checks["passed"]:
        raise ValueError("Suite, invariant or integrity verification is incomplete")
    candidates = []
    for steps in (36, 48):
        for cue in (.05, 0.):
            # PA delay tied all Attention counts, hence the original timing is retained.
            group = scores["results"]
            r1 = group[f"r1/{steps}_defer0/parity"]
            r2 = group[f"r2/{steps}_cue{cue:g}/parity"]
            r3 = group[f"r3/{steps}_shrinkage/parity"]
            totals = [r["score"]["block_errors"] for r in (r1, r2, r3)]
            if [r["score"]["blocks"] for r in (r1,r2,r3)] != [384,1728,384]:
                raise ValueError("Incomplete development evaluation")
            repairs = r2["signs"]["repaired"]
            joint = r3["green"]["trained"]["parallel_joint"]["overall"]["mean"]
            spatial = r3["green"]["trained"]["spatial"]["overall"]["mean"]
            metrics_pass = all(x <= t for x,t in zip(totals,(25,340,49),strict=True))
            metrics_pass &= repairs >= 8 and joint >= .34
            gauge = None
            if steps == 48:
                audit = OUT if cue else OUT / "cue_off_audits"
                first = read(audit / "audit_first.json")
                trained = read(audit / "audit_gauge.json")
                if not (first["passed"] and len(first["rows"]) == 45 and trained["cue"] == cue
                        and trained["steps"] == steps and trained["summary"]["checks"] == 360):
                    raise ValueError("Missing first-update or trained checks")
                gauge = trained["summary"]
            candidates.append(dict(steps=steps, cue=cue, deferred_r1_pa=False, errors=totals,
                                   repaired_signs=repairs, joint_green=joint, spatial_green=spatial,
                                   passes_performance_gate=metrics_pass, spatial_48_target=totals[2] <= 48,
                                   gauge=gauge, eligible=bool(metrics_pass and gauge and gauge["passed"])))
    evidence = [OUT / name for name in ("dev_scores.json", "audit_suite.json", "legacy_checks.json",
                "audit_first.json", "audit_gauge.json", "cue_off_audits/audit_first.json",
                "cue_off_audits/audit_gauge.json", "integrity_after.json")]
    result = dict(candidates=candidates, passed=any(c["eligible"] for c in candidates),
                  selected_for_dev=dict(steps=48, cue=.05, deferred_r1_pa=False),
                  spatial_followup_errors_36_48=[
                      scores["spatial_followup"][f"r3/{n}_shrinkage/parity"]["score"]["block_errors"]
                      for n in (36,48)], p6_tests=suite["run"], legacy_passed=legacy["passed"],
                  sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in evidence})
    if not legacy["passed"]:
        raise ValueError("Missing legacy verification")
    (OUT / "gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Development gate passed = {result['passed']}")


if __name__ == "__main__":
    main()
