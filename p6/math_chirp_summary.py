"""Readable machine summary of the bounded R2 optimization and development audits."""
from collections import defaultdict
from pathlib import Path

from p6 import experiment as ex
from p6.math_chirp_dev import OUT


def summary():
    scores = {}
    gauges = {}
    for mode in ("tempered","tangent","hybrid","conditional","conditional_gn","fisher","trust"):
        scores[mode] = {}
        for source in ("legacy","fit"):
            arms = defaultdict(list)
            audit = []
            for path in sorted((OUT / mode / source).glob("*/*.json")):
                row = ex.read(path)
                if "score" in row:
                    arms[row["arm"]].append(row)
                if "rows" in row and "summary" in row:
                    audit += row["rows"]
            if arms:
                scores[mode][source] = {arm:dict(
                    pairs=[r["pair"] for r in records],
                    failed_by_pair=[r["score"]["block_errors"] for r in records],
                    failed=sum(r["score"]["block_errors"] for r in records),
                    words=sum(r["score"]["blocks"] for r in records),
                    repaired=sum(r["signs"]["repaired"] for r in records),
                    regressions=sum(r["signs"]["regressed"] for r in records),
                    initially_wrong=sum(r["signs"]["initial_wrong"] for r in records))
                    for arm,records in arms.items()}
            if audit:
                from p6.extended_audit import summarize
                gauges.setdefault(mode,{})[source] = summarize(audit)
    aperture = {}
    for name,path in (("fisher",OUT/"aperture.json"),("trust",OUT/"trust/aperture.json")):
        if not path.exists():
            continue
        groups=defaultdict(list)
        for row in ex.read(path)["rows"]:
            groups[row["condition"]].append(row)
        aperture[name]={}
        for condition,rows in groups.items():
            support=[v for row in rows for v in row["lambda_by_burst"]]
            aperture[name][condition]=dict(min_support=min(support),mean_support=sum(support)/len(support),
                max_support=max(support),baseline=sum(row["scores"]["baseline"]["block_errors"] for row in rows),
                candidate=sum(row["scores"][name]["block_errors"] for row in rows))
    ex.save(OUT/"summary.json",dict(scores=scores,gauges=gauges,aperture=aperture))


if __name__ == "__main__":
    summary()
