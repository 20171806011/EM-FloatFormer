"""Small paired diagnostic for the P12 consistency correction, not a formal study.

Run this same script in separate processes against an immutable baseline worktree
and the corrected worktree. The protocol fixes seeds and public anchor positions;
the output fingerprints observations/truth and the source files actually imported.
No fitting, threshold selection, or significance claims are made from these links.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--label", required=True, choices=("baseline", "corrected"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Set these before NumPy/Numba/PyTorch imports, including in a fresh process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(repo))

    import numpy as np
    from p11.contract import enforce_determinism
    from p12.contract import assert_p12_seed_disjoint
    from p12.offgrid_physics import score_offgrid_channel_est, simulate_offgrid_link_p12
    from p12.offgrid_receiver import run_p12_offgrid_physics, warmup_p12_offgrid_numba

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if args.label == "baseline" and head != protocol["baseline_commit"]:
        raise ValueError("Baseline worktree is not at the protocol's immutable commit")
    if args.label == "baseline":
        tracked_diff = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=repo, text=True)
        if tracked_diff.strip():
            raise ValueError("Baseline worktree has modified tracked files")
    seeds = [int(case["seed"]) for case in protocol["cases"]]
    assert_p12_seed_disjoint(seeds, expected_namespace="p12_e2e")
    runtime = enforce_determinism(require_env=True)

    def imported_sources() -> dict[str, str]:
        paths = {repo / "p8" / "ldpc_checks.txt"}
        for module in tuple(sys.modules.values()):
            name = getattr(module, "__file__", None)
            if name:
                path = Path(name).resolve()
                # PyTorch also exposes virtual modules with relative __file__ names.
                # Fingerprint actual repository files, not those virtual placeholders.
                if path.is_relative_to(repo) and path.suffix == ".py" and path.is_file():
                    paths.add(path)
        return {str(path.relative_to(repo)): sha256_file(path) for path in sorted(paths)}

    source_hashes = imported_sources()
    warmup_p12_offgrid_numba()
    records = []
    started = time.perf_counter()
    for case in protocol["cases"]:
        obs, truth, extra = simulate_offgrid_link_p12(
            int(case["seed"]), case["regime"], snr_db=float(protocol["snr_db"]),
            pilot_rows_a=tuple(protocol["pilot_rows_a"]), anchor_row=int(case["anchor_row"]),
        )
        digest = hashlib.sha256()
        for name, array in (
            ("y_a", obs.y_a), ("y_b", obs.y_b), ("pilot_tx_a", obs.pilot_tx_a),
            ("pilot_tx_b", obs.pilot_tx_b), ("training_rows", np.asarray(obs.training_rows)),
            ("n0", np.asarray(obs.n0)), ("truth_bits", truth.b_info),
            ("truth_channel", truth.h_b_true_mat),
        ):
            arr = np.ascontiguousarray(array)
            digest.update(json.dumps([name, list(arr.shape), arr.dtype.str]).encode("utf-8"))
            digest.update(arr.tobytes())
        result = run_p12_offgrid_physics(obs)
        for array in (result.h_mix, result.noise_mix, result.gate_out.mix_llrs):
            if not np.isfinite(array).all():
                raise AssertionError(f"Non-finite receiver output for seed {case['seed']}")
        scored = score_offgrid_channel_est(
            result.h_mix, result.noise_mix, result.gate_out.mix_llrs, obs, truth,
            probe_pred=result.gate_out.mix_probe_pred,
        )
        row_to_idx = {int(row): i for i, row in enumerate(truth.data_rows)}
        false_assimilations = sum(
            not np.array_equal(item["injected_info"], truth.b_info[row_to_idx[int(item["row"])]])
            for item in result.ledger
        )
        record = {
            **case, **scored, "observation_and_truth_sha256": digest.hexdigest(),
            "n_rays": int(truth.n_rays), "observed_anchor_row": int(extra["anchor_row"]),
            "helpers_accepted": int(result.helpers_accepted),
            "false_assimilations": int(false_assimilations),
            "delay_reset_triggered": bool(result.delay_reset_triggered),
            "receiver_elapsed_ms": float(result.elapsed_ms),
        }
        records.append(record)
        print(f"{args.label}: {case['regime']} anchor={case['anchor_row']} "
              f"errors={scored['block_errors']}/{scored['blocks']}", flush=True)

    if imported_sources() != source_hashes:
        raise RuntimeError("Imported source set/content changed during the diagnostic")
    blocks = sum(record["blocks"] for record in records)
    errors = sum(record["block_errors"] for record in records)
    payload = {
        "kind": "exploratory_consistency_diagnostic_not_confirmatory",
        "label": args.label, "git_head": head,
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "imported_source_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(), "platform": platform.system(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "torch")},
            "determinism": runtime,
        },
        "summary": {"links": len(records), "blocks": blocks, "block_errors": errors,
                    "bler_pct": 100.0 * errors / blocks},
        "records": records, "wall_seconds_after_warmup": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
