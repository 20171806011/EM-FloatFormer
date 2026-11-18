"""Independent continuous-S11 burst replication, retaining each complete link outcome."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from p4.stage3_link_study import identity
from p4.stage3_link_workload import run_mission_link


BASE=Path(__file__).resolve().parent
OUT=BASE/"results"
SEAL=OUT/"stage3_link_variability_protocol.json"
RESULT=OUT/"stage3_link_variability.json"


def recipe():
    return {"seeds":list(range(904401,904413)),"mapping":"interleaved","total_bits":100000,
            "num_slots":50,"snr_db":15.,"metric":"all 12 independent burst counts, per-burst <1% and pooled BLER",
            "existing_recipe_matches":identity()==json.loads((OUT/"stage3_link_protocol.json").read_bytes()),
            "source_hashes":{p.name:sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__),BASE/"stage3_link_workload.py",OUT/"stage3_link_protocol.json"]}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal",action="store_true")
    if parser.parse_args().seal:
        with SEAL.open("x",encoding="utf-8") as f:
            json.dump(recipe(),f,indent=2)
    else:
        spec=json.loads(SEAL.read_bytes())
        assert spec==recipe()
        digest=sha256(SEAL.read_bytes()).hexdigest()
        data=json.loads(RESULT.read_bytes()) if RESULT.exists() else {"protocol":digest,"rows":[]}
        assert data["protocol"]==digest
        for seed in spec["seeds"]:
            if any(r["seed"]==seed for r in data["rows"]):
                continue
            r=run_mission_link(total_bits=spec["total_bits"],num_slots=spec["num_slots"],
                               snr_db=spec["snr_db"],mapping=spec["mapping"],seed=seed)
            data["rows"].append(r)
            tmp=RESULT.with_suffix(".tmp")
            tmp.write_text(json.dumps(data,indent=2))
            tmp.replace(RESULT)
            print(seed,r["block_errors"],r["total_blocks"],r["total_bit_errors"],flush=True)
