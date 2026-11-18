"""Collect public P4-B workload outputs for independent qualification checks."""
import json
from pathlib import Path

import torch

from . import slot_2d
from .stage3_qualification import ROOT, OBS, evidence_sources, unit_evidence


def measure():
    bindings = evidence_sources(ROOT)
    cases = {"S01": slot_2d.run_hst_workload(),
             "S02": {p: slot_2d.run_urban_macro_workload(profile=p) for p in ("long12", "sparse12")},
             "S03": slot_2d.run_cfo_drift_workload(),
             "S05": slot_2d.run_ultra_sparse_workload(),
             "S06": slot_2d.run_multi_impairment_stress(),
             "S07": slot_2d.run_ood_generalization(),
             "S11": slot_2d.run_mission_link()}
    units = unit_evidence()
    if evidence_sources(ROOT) != bindings:
        raise RuntimeError("Workload dependencies changed during measurement; retry on a stable revision.")
    return {"schema": "m3-public-observations-v1", "sources": bindings, "m3_units": units, "cases": cases}


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = measure()
    path = ROOT/OBS
    if (ROOT/"p4/results/p4d/P4D_FREEZE.json").exists():
        if json.loads(path.read_bytes()) != result:
            raise RuntimeError("Measurement differs from the evidence used by the frozen study.")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    print(path)
