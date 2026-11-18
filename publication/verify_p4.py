"""Verify the public P4 evidence and optionally reproduce its numerical results."""
import argparse
from pathlib import Path

import torch

from p4 import confirmatory_protocol as formal
from p4 import stage3_factorial_confirm as factorial
from p4.stage3_confirm_audit import count_again
from publication import p4d_original_audit as audit


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "p4/results/p4d"
ARCHIVE = "p4/revisions/pre_stage3_p4.zip"
ORIGINAL_ARCHIVE = "4ebfcf6fb2dd36c7575a3648babd7d4958469e8bc5df9c8fafce8c224371f77a"
PUBLIC_ARCHIVE = "8cd0abe4b909857a18ab029de76b69590993ed9cabc2c438df451fa8ae3a33b8"


def verify_public_files():
    freeze_path = DEST / "P4D_FREEZE.json"
    audit.require(audit.digest(freeze_path) == audit.ACTIVE_FREEZE, "original freeze manifest changed")
    freeze = audit.load(freeze_path)
    files = freeze["files"]
    audit.require(len(files) == 117 and files[ARCHIVE] == ORIGINAL_ARCHIVE, "unexpected freeze inventory")
    for name, original in files.items():
        expected = PUBLIC_ARCHIVE if name == ARCHIVE else original
        audit.require(audit.digest(ROOT / name) == expected, f"wrong public file digest: {name}")
    audit.require(formal._historical_digest(ROOT) == freeze["historical_checkout"], "legacy checkout changed")
    print("230 previous checkout files, 116 original frozen files and one redacted archive verified.", flush=True)
    return freeze


def verify_results(freeze, replay):
    summary = audit.load(DEST / "summary.json")
    audit.require(summary["freeze_sha256"] == audit.ACTIVE_FREEZE and summary["stage"] == "P4-D",
                  "wrong coded summary identity")
    with (DEST / "raw_by_seed.csv").open(encoding="utf-8", newline="") as f:
        cells = audit.parse_csv(f, audit.OFFICIAL, 416)
    audit.score(summary["records"], cells, summary, audit.OFFICIAL, 416)
    audit.checkpoints(DEST, summary, audit.ACTIVE_FREEZE)
    audit.joint(DEST, audit.ACTIVE_FREEZE)
    audit.e2e(ROOT, ROOT / "p4/results/stage3_final_e2e.json", freeze)
    unit = audit.load(ROOT / "p4/results/stage3_final_unit_tests.json")
    audit.require(unit["run"] == len(unit["passed"]) == len(set(unit["passed"])) == 100
                  and not any(unit[k] for k in ("failures", "errors", "skips")), "unit receipt mismatch")
    for name, sha in unit["tests_sha256"].items():
        audit.require(audit.digest(ROOT / name) == sha, f"unit-test source changed: {name}")

    rows = []
    for seed in audit.OFFICIAL:
        for arm in range(1, 9):
            row = audit.load(DEST / "factorial" / str(seed) / f"arm_{arm}.json")["row"]
            if replay:
                audit.require(factorial._fit_and_score(seed, arm, independent=True) == row,
                              f"independent factorial fit differs: {seed}/{arm}")
            rows.append(row)
        if replay:
            record = audit.load(DEST / str(seed) / "counts.json")
            state = torch.load(DEST / str(seed) / "training.pt", map_location="cpu", weights_only=True)
            model = formal.EMFloatFormer2D(d_model=16, n_heads=2, depth=2)
            model.load_state_dict(state["model"])
            audit.require(count_again(model.eval(), seed, formal.FULL) == record["rows"],
                          f"independent coded replay differs: {seed}")
            print(f"Reproduced seed {seed}: eight fresh fits and all coded counts.", flush=True)
    reproduced = {"freeze": audit.ACTIVE_FREEZE, **factorial.analyze(rows, audit.OFFICIAL)}
    audit.require(reproduced == audit.load(DEST / "factorial/summary.json"), "factorial summary mismatch")
    print("150 coded rows, ten final weight hashes, both Holm families and both test receipts verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true",
                        help="Regenerate coded observations and independently retrain all 80 factorial arms.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify_results(verify_public_files(), args.replay)
