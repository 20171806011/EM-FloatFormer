"""Pool original development fits; keep checkpoint provenance and intervention directions."""
from collections import defaultdict
import statistics

from p6 import experiment as ex, iter4_spatial, iter5_dev
from p6.data import PAIRS
from p6.green_support_control import run as uniform_control
from p6.run_dev_iter2 import pooled, probe_summary, digest


def aggregate(folder):
    groups = defaultdict(list)
    for path in sorted(folder.glob("r*/*/*/*.json")):
        rec = ex.read(path)
        if digest(path.with_suffix(".pt")) != rec["checkpoint_sha256"]:
            raise ValueError(f"Stale weight file {path}")
        group = path.relative_to(folder).parts
        groups[f"{group[0]}/{group[1]}/{rec['arm']}"].append(rec)
    results = {}
    for key, rows in groups.items():
        if tuple(row["pair"] for row in rows) != PAIRS or len({r["implementation"] for r in rows}) != 1:
            raise ValueError(f"Seeds or implementation mismatch in {key}")
        row = dict(score=pooled([r["score"] for r in rows]),
                   failed_by_pair=[r["score"]["block_errors"] for r in rows],
                   parameters=rows[0]["parameters"], implementation=rows[0]["implementation"],
                   mean_fit_seconds=statistics.mean(r["fit_seconds"] for r in rows))
        if "signs" in rows[0]:
            row["signs"] = {k: sum(r["signs"][k] for r in rows)
                            for k in ("initial_wrong","final_wrong","repaired","regressed")}
        if "spatial_off_score" in rows[0]:
            row["spatial_off_by_pair"] = [r["spatial_off_score"]["block_errors"] for r in rows]
            row["spatial_off"] = pooled([r["spatial_off_score"] for r in rows])
            for field in ("steering", "steering_off"):
                stages = iter4_spatial.pool([r[field] for r in rows])
                row[field] = {s: {k: v for k, v in stages[s].items() if k != "sums"} for s in stages}
        if "probe" in rows[0]:
            row["green"] = probe_summary(rows)
            if "probe_off" in rows[0]:
                pairs = [dict(pair=r["pair"], probe=dict(initial=r["probe"]["initial"],
                                                         trained=r["probe_off"])) for r in rows]
                row["green_spatial_off"] = probe_summary(pairs)["trained"]
        results[key] = row
    return results


if __name__ == "__main__":
    ex.setup()
    out = iter5_dev.OUT
    fits = aggregate(out / "fits")
    # Use the same B32 development draws for current fixed and classical references.
    from p6.run_dev_iter2 import references
    refs = {}
    for r in (1, 2, 3):
        scores = [references(r, pair, 3)["scores"] for pair in PAIRS]
        refs[r] = {k: dict(pooled=pooled([s[k] for s in scores]),
                           by_pair=[s[k]["block_errors"] for s in scores]) for k in scores[0]}
    followup = aggregate(out / "spatial_coherence" / "fits")
    ex.save(out / "dev_scores.json", dict(pairs=PAIRS, results=fits, spatial_followup=followup, references=refs,
                                          uniform_green=uniform_control()))
