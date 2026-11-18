"""Use development seeds to exercise the full analysis, independent counters and replay."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from . import confirmatory_protocol as protocol
from .stage3_confirm_audit import count_again, audit_csv
from .stage3_factorial_confirm import _fit_and_score, _check, analyze


def rehearse():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seeds = tuple(range(71461, 71471))
    assert not set(seeds).intersection(protocol.OFFICIAL)
    rows, records, replayed = [], [], []
    for seed in seeds:
        for arm in range(1, 9):
            row = _fit_and_score(seed, arm)
            if seed == seeds[0]:
                if _fit_and_score(seed, arm, independent=True) != row:
                    raise RuntimeError(f"Development paired weight and count replay failed for arm {arm}")
                replayed.append(arm)
            rows.append(row)
        model = protocol._train(seed, protocol.SMOKE, "rehearsal")
        counts = protocol._score(seed, model, protocol.SMOKE)
        if count_again(model, seed, protocol.SMOKE) != counts:
            raise RuntimeError(f"Independent coded receiver counts differ for {seed}.")
        records.append({"seed": seed, "rows": counts, "weights_sha256": protocol._weights_hash(model)})
        print(f"Completed development factorial and coded count audit for {seed}", flush=True)
    factorial = analyze(rows, seeds)
    coded = protocol._analyze(records, protocol.SMOKE)
    with TemporaryDirectory() as dest:
        path = Path(dest) / "raw.csv"
        path.write_bytes(protocol._result_csv(records))
        audit = audit_csv(path, records, coded["primary_tests"])
    # Verify the rejection paths without consuming formal draws.
    corrupted = [dict(row) for row in rows]
    corrupted[-1]["bit_errors"] += 1
    try:
        _check(corrupted, seeds)
    except ValueError:
        pass
    else:
        raise RuntimeError("Corrupt pooled count was accepted.")
    return {"role": "Development rehearsal. All 80 one-dimensional cells use the planned budget; "
                    "the coded component uses only the smoke budget.", "seeds": seeds,
            "coded_budget": vars(protocol.SMOKE), "independently_retrained_factorial_arms": replayed,
            "coded_audit": audit, "coded": coded, "factorial": factorial, "factorial_rows": rows,
            "history": protocol._historical_digest()}


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "results/stage3_protocol_rehearsal.json"
    path.write_text(json.dumps(rehearse(), indent=2, allow_nan=False), encoding="utf-8")
