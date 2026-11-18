"""Frozen fit, confirmatory scoring and independent checkpoint replay."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import time

import torch

from p5 import study as p5
from p5.wavefield_sim import green_kernel, simulate
from p5.statistics import COUNT_FIELDS
from p5b import study as p5b
from p6 import experiment as ex
from p6.axial_operator import _probe_coordinates
from p6.data import SNR
from p6.math_training import Recipe, receiver, train
from p6.math_verify import FREEZE, sha, verify_freeze
from p6.run_dev_iter2 import probe


ROOT = ex.RESULTS / "formal"
TEST_BATCHES, EVAL_BATCH = 10, 8
COHORTS = {
    "A": {1: range(11, 21), 2: range(6, 16), 3: range(6, 16)},
    "B": {1: range(1, 11), 2: range(1, 6), 3: range(1, 6)},
}


@lru_cache(maxsize=1)
def contract():
    signature = verify_freeze()
    if (TEST_BATCHES, EVAL_BATCH) != (p5.TEST_BATCHES, p5.EVAL_BATCH):
        raise ValueError("Archived simulator batch sizes disagree")
    return signature, Recipe(**ex.read(FREEZE)["recipe"])


def cohort(r, p):
    options = [c for c, regimes in COHORTS.items() if p in regimes[r]]
    if len(options) != 1:
        raise ValueError("Pair is outside the two disjoint registered cohorts")
    return options[0]


def folder(r, p):
    return ROOT / cohort(r, p) / f"r{r}" / f"pair{p:02d}"


def draw(r, p, stream, b, i=0, snr=None):
    contract()
    cohort(r, p)
    if stream not in (5, 7, 8) or (stream == 8 and r != 3) or not 0 <= i < 1000:
        raise ValueError("Unregistered draw")
    return simulate(r, b, SNR[r] if snr is None else snr, p5.rng(r, stream, p, i))


def model(r, p, name):
    _, recipe = contract()
    return receiver(name, p5.rng(r, 5, p, 900), recipe)


def restored(r, p):
    signature, _ = contract()
    result = {}
    for a in ex.ARMS:
        m = model(r, p, a)
        ck = folder(r, p) / (a + ".pt")
        record = ex.read(folder(r, p) / "fit.json")
        if sha(ck) != record["checkpoints"][a] or record["freeze_sha256"] != signature:
            raise ValueError(f"Checkpoint provenance failure {ck}")
        state = torch.load(ck, weights_only=True, map_location="cpu")
        if state["implementation"] != signature:
            raise ValueError(f"Checkpoint freeze mismatch {ck}")
        m.load_state_dict(state["state"])
        result[a] = m.eval()
    return result


def fit(r, p):
    signature, recipe = contract()
    dest = folder(r, p)
    if (dest / "fit.json").is_file():
        restored(r, p)
        return
    start = time.perf_counter()
    checkpoints = {}
    for a in ex.ARMS:
        m = model(r, p, a)
        train(m, lambda i: draw(r, p, 5, 4, i), recipe)
        ck = dest / (a + ".pt")
        ex.save_weights(ck, m, signature)
        checkpoints[a] = sha(ck)
    ex.save(
        dest / "fit.json",
        dict(
            freeze_sha256=signature,
            cohort=cohort(r, p),
            regime=r,
            pair=p,
            checkpoints=checkpoints,
            seconds=time.perf_counter() - start,
        ),
    )


def uniform(frame):
    r, t, n = frame.obs.y.shape[1:]
    coords = _probe_coordinates(r, t, n)
    off = ~torch.eye(len(coords), dtype=torch.bool)
    equal = coords[:, None] == coords[None, :]
    kernel = (
        sum(
            equal[..., [k for k in range(3) if k != axis]].all(-1).double() / size
            for axis, size in enumerate((r, t, n))
        )
        / 3
    )
    return ex.correlation(kernel[off], green_kernel(frame, coords)[:, off]).tolist()


def legacy_check(r, p, rows, frames):
    b_rows = {n: rows["p5b_" + n] for n in (*p5b.ARMS, *p5b.ANCHORS)}
    b_rows.update({n: rows[n] for n in ("p5_full", *ex.METHODS[r])})
    original = ex.read(p5b.fold(p5b.ARCHIVE, r, p) / "record.json")["rows"]
    p5_matches = {}
    for key in ("p5_full", *ex.METHODS[r]):
        previous = original["full" if key == "p5_full" else key]
        integer_diffs = {f: b_rows[key][f] - previous[f] for f in COUNT_FIELDS}
        p5_matches[key] = dict(
            integer_counts_match=not any(integer_diffs.values()),
            differences=integer_diffs,
            all_fields_exact=b_rows[key] == previous,
        )
    archived = ex.read(p5b.fold(p5b.RESULTS, r, p) / "record.json")
    matches = {
        k: all(b_rows[k][f] == archived["rows"][k][f] for f in COUNT_FIELDS) for k in archived["rows"]
    }
    return dict(
        p5_comparisons=p5_matches,
        p5b_integer_matches=matches,
        frame_digests_exact=frames == archived["frames"],
        passed=all(matches.values())
        and all(v["integer_counts_match"] for v in p5_matches.values())
        and frames == archived["frames"],
    )


@torch.no_grad()
def evaluate(r, p):
    models = restored(r, p)
    probe_models = {k: models[k] for k in ("pure", "parity")}
    models.update({a: model(r, p, a).eval() for a in ex.FIXED})
    if cohort(r, p) == "B":
        legacy = p5b.with_controls(p5b.restored(r, p), r, p)
        models.update({(a if a == "p5_full" else "p5b_" + a): m for a, m in legacy.items()})
    methods = [*models, *ex.METHODS[r]]
    rows = {a: {} for a in methods}
    hashes = []
    for i in range(TEST_BATCHES):
        f = draw(r, p, 7, EVAL_BATCH, i)
        hashes.append(
            dict(
                seed=p5.rng(r, 7, p, i),
                batch=EVAL_BATCH,
                field_sha256=p5b.tensor_hash(f.obs.y),
                codewords_sha256=p5b.tensor_hash(f.bits),
            )
        )
        for a in methods:
            out = models[a](f.obs) if a in models else p5.classical(f, a)
            p5.add(rows[a], p5.counts(f, *out))
    kernels = {}
    if r == 3 and cohort(r, p) == "A":
        f = draw(3, p, 8, 8)
        kernels["uniform"] = uniform(f)
        for a, m in probe_models.items():
            kernels[a] = dict(
                initial=probe(model(r, p, a), f, p5.rng(3, 9, p)), trained=probe(m, f, p5.rng(3, 9, p))
            )
    arch = legacy_check(r, p, rows, hashes) if cohort(r, p) == "B" else None
    return dict(
        freeze_sha256=contract()[0],
        regime=r,
        pair=p,
        cohort=cohort(r, p),
        snr_db=SNR[r],
        scores=rows,
        frames=hashes,
        green=kernels,
        legacy=arch,
    )


def job(args):
    action, r, p = args
    contract()
    dest = folder(r, p)
    if action == "fit":
        fit(r, p)
    elif action == "score":
        if (dest / "record.json").is_file():
            existing = ex.read(dest / "record.json")
            if (existing["regime"], existing["pair"], existing["freeze_sha256"]) != (
                r,
                p,
                contract()[0],
            ) or len(existing["frames"]) != TEST_BATCHES:
                raise ValueError("Existing formal receipt has inconsistent provenance")
            return
        row = evaluate(r, p)
        ex.save(dest / "record.json", row)
    elif action == "replay":
        row = evaluate(r, p)
        saved = ex.read(dest / "record.json")
        integer_exact = all(
            all(s[k] == saved["scores"][a][k] for k in COUNT_FIELDS) for a, s in row["scores"].items()
        )
        differences = {
            a: {
                k: dict(recorded=v, replayed=row["scores"][a][k])
                for k, v in s.items()
                if v != row["scores"][a][k]
            }
            for a, s in saved["scores"].items()
            if s != row["scores"][a]
        }
        checks = dict(
            frames_exact=row["frames"] == saved["frames"],
            integer_exact=integer_exact,
            scores_exact=row["scores"] == saved["scores"],
            green_exact=row["green"] == saved["green"],
            archive_exact=row["legacy"] == saved["legacy"],
        )
        ex.save(
            dest / "replay.json",
            dict(
                regime=r,
                pair=p,
                checkpoints=5,
                batches=10,
                **checks,
                passed=all(checks.values()),
                score_differences=differences,
            ),
        )
    print(f"{action} R{r}/{p}", flush=True)


def pairs():
    return [(r, p) for r in (3, 2, 1) for c in ("A", "B") for p in COHORTS[c][r]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "score", "replay"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    contract()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=ex.setup,
        max_tasks_per_child=1 if args.action in ("score", "replay") else None,
    ) as pool:
        list(pool.map(job, [(args.action, r, p) for r, p in pairs()], chunksize=1))
    if args.action == "replay":
        results = [ex.read(folder(r, p) / "replay.json") for r, p in pairs()]
        ex.save(
            ex.RESULTS / "checkpoint_replay.json",
            dict(
                freeze_sha256=contract()[0],
                pairs=len(results),
                checkpoints=sum(x["checkpoints"] for x in results),
                passed=all(x["passed"] for x in results),
                rows=results,
            ),
        )
        if not all(item["passed"] for item in results):
            raise SystemExit("Checkpoint replay discrepancies are recorded in checkpoint_replay.json")
    if args.action == "score":
        archive_checks = [ex.read(folder(r, p) / "record.json")["legacy"]["passed"]
                          for r, p in pairs() if cohort(r, p) == "B"]
        if not all(archive_checks):
            raise SystemExit("Archived integer or frame mismatches are recorded in Cohort B receipts")
