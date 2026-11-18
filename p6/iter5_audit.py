"""Development-only invariant and unit checks for the fifth iteration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import sys
import unittest
from unittest.mock import patch

import torch

from p5b.diagnostics import SCALES
from p6 import experiment as ex, iter5_dev, training_iter3
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.diagnostics import anchor_check
from p6.extended_audit import ARMS as GAUGE_ARMS, gauge_rows, summarize


OUT = ex.RESULTS / "dev_iter5"


def candidate_dir(cue):
    return OUT if cue == .05 else OUT / 'cue_off_audits'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {name: digest(ex.ROOT / name) for name in (
        "iter5_audit.py", "iter5_dev.py", "axial_operator.py", "axial_em_transformer.py",
        "parity.py", "training_iter3.py", "data.py", "extended_audit.py", "diagnostics.py")}


def verify_donor(receipt, sources_now, steps, defer):
    if not (receipt['cue'] == .05 and receipt['steps'] == steps and receipt['defer'] == defer):
        raise ValueError('Inactive-switch reuse requested with different training settings')
    if any(receipt['sources'][name] != value for name, value in sources_now.items() if name != 'iter5_audit.py'):
        raise ValueError('Source changed between the candidate audits')


def first_batch(regime, pair, steps, cue, defer):
    config = iter5_dev.CONFIG
    rows = []
    frame = dev_frame(regime, 5, pair, ex.FIT_BATCH)
    validation = dev_frame(regime, 6, pair, 2)
    reference_state = iter5_dev.model_for(regime, pair, "pure", cue, defer).state_dict()
    for arm in ex.ARMS:
        model = iter5_dev.model_for(regime, pair, arm, cue, defer)
        state = model.state_dict()
        matched = reference_state.keys() == state.keys() and all(
            torch.equal(tensor, reference_state[name]) for name, tensor in state.items())
        anchor = anchor_check(model, validation.obs, arm, config.depth)
        live = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        opt = torch.optim.Adam(live.values(), lr=ex.learning_rate(config, 0, steps))
        loss = training_iter3.objective(frame, model, aux_nmse=config.aux_nmse)[0]
        loss.backward()
        gradients = {}
        for name, parameter in live.items():
            grad = parameter.grad
            real = None if grad is None else torch.view_as_real(grad) if grad.is_complex() else grad
            gradients[name] = dict(
                total=parameter.numel() * (2 if parameter.is_complex() else 1),
                nonzero=int(real.count_nonzero()) if real is not None else 0,
                finite=bool(real is not None and real.isfinite().all()))
        all_live = all(entry['finite'] and entry['nonzero'] == entry['total'] for entry in gradients.values())
        norm = float(torch.nn.utils.clip_grad_norm_(live.values(), config.clip, error_if_nonfinite=True))
        opt.step()
        updated = all(parameter.detach().isfinite().all().item() for parameter in live.values())
        with torch.no_grad():
            new_logits = model(validation.obs)[0]
            fixed = ex.model_for(regime, pair, "physics_parity" if arm in ex.PARITY_ARMS else "physics", config.depth)
            max_change = float((new_logits - fixed(validation.obs)[0]).abs().max())
        rows.append(dict(regime=regime, pair=pair, arm=arm, seed=dev_seed(regime, 5, pair),
                         objective=float(loss.detach()), global_grad_norm=norm, anchor=anchor,
                         matched_init=matched, gradients=gradients, parameters=ex.real_parameters(model),
                         post_update_max_llr_change=max_change, all_real_gradients_live=all_live,
                         passed=matched and anchor['p6_bit_exact'] and max(anchor['archived_max_llr_cfr']) == 0
                         and all_live and updated and max_change > 0))
    return rows


def first(args):
    directory = candidate_dir(args.cue)
    refs = {}
    reused = []
    regimes = SNR
    if args.reuse_inactive and args.cue == 0.:
        donor = OUT / 'audit_first.json'
        original = ex.read(donor)
        verify_donor(original, sources(), args.steps, args.defer)
        reused = [r for r in original['rows'] if r['regime'] in (1, 3)]
        if len(reused) != 30:
            raise ValueError('Original inactive-regime gradient archive incomplete')
        refs[donor.name] = digest(donor)
        regimes = (2,)
    rows = reused + [row for regime in regimes for pair in PAIRS
            for row in first_batch(regime, pair, args.steps, args.cue, args.defer)]
    rows.sort(key=lambda r: (r['regime'], r['pair'], ex.ARMS.index(r['arm'])))
    passed = all(row['passed'] for row in rows) and len({row['parameters'] for row in rows}) == 1
    ex.save(directory / "audit_first.json", dict(sources=sources(), steps=args.steps, cue=args.cue,
                                          defer=args.defer, reuse_receipts=refs, rows=rows, passed=passed))
    print(f"First update and anchors {sum(row['passed'] for row in rows)}/{len(rows)}", flush=True)
    return passed


def trained_cell(task):
    regime, pair, steps, cue, defer, reuse_inactive = task
    rows, checkpoints = [], {}
    filename = f"audit_gauge_r{regime}_{pair}.json"
    persisted = candidate_dir(cue) / filename
    audit_sources = sources()
    if reuse_inactive and cue == 0. and regime in (1, 3):
        donor = OUT / filename
        prior = ex.read(donor)
        verify_donor(prior, audit_sources, steps, defer)
        for arm in GAUGE_ARMS:
            path = iter5_dev.location(regime, pair, arm, steps, cue, defer)
            if path != iter5_dev.location(regime, pair, arm, steps, .05, defer):
                raise ValueError(f'Inactive switch has different fit destination at {path}')
            record = ex.read(path)
            ck = path.with_suffix('.pt')
            prov = prior['checkpoints'][arm]
            if (record['cue'] is not None or record['implementation'] != iter5_dev.signature()
                or prov['receipt_sha256'] != digest(path) or prov['sha256'] != digest(ck)):
                raise ValueError(f'Reused checkpoint metadata mismatch at {path}')
        return {**prior, 'reuse_receipt': {donor.name: digest(donor)}}
    if persisted.is_file():
        prior = ex.read(persisted)
        same = prior.get('sources') == audit_sources and (
            prior['steps'], prior['cue'], prior['defer']) == (steps, cue, defer)
        if same and set(prior['checkpoints']) == set(GAUGE_ARMS):
            paths = {arm: iter5_dev.location(regime, pair, arm, steps, cue, defer) for arm in GAUGE_ARMS}
            if all(digest(path) == prior['checkpoints'][arm]['receipt_sha256'] and
                   digest(path.with_suffix('.pt')) == prior['checkpoints'][arm]['sha256']
                   for arm, path in paths.items()):
                return prior
    for arm in GAUGE_ARMS:
        receipt = iter5_dev.location(regime, pair, arm, steps, cue, defer)
        ck = receipt.with_suffix(".pt")
        record = ex.read(receipt)
        sha = digest(ck)
        identity = (regime, pair, arm, steps, cue if regime == 2 else None, defer if regime == 1 else None)
        recorded = tuple(record[key] for key in ('regime', 'pair', 'arm', 'steps', 'cue', 'defer'))
        if sha != record['checkpoint_sha256'] or identity != recorded:
            raise ValueError(f"Checkpoint/receipt identity mismatch at {ck}")
        model = iter5_dev.model_for(regime, pair, arm, cue, defer)
        saved = torch.load(ck, map_location="cpu", weights_only=True)
        if not saved['implementation'] == record['implementation'] == iter5_dev.signature():
            raise ValueError(f"Checkpoint/fit source mismatch at {ck}")
        model.load_state_dict(saved['state'], strict=True)
        checkpoints[arm] = dict(path=ck.relative_to(ex.ROOT).as_posix(), sha256=sha,
                                receipt_sha256=digest(receipt))
        rows.extend(gauge_rows(model.eval(), regime, pair, arm, "iter5", steps, full_grid=True))
    if sources() != audit_sources:
        raise RuntimeError('Source changed during gauge observations')
    result = dict(regime=regime, pair=pair, steps=steps, cue=cue, defer=defer, checkpoints=checkpoints,
                  sources=audit_sources,
                  rows=rows, summary=summarize(rows))
    ex.save(persisted, result)
    print(f"Trained R{regime}/{pair}: {result['summary']}", flush=True)
    return result


def gauge(args):
    tasks = [(r, p, args.steps, args.cue, args.defer, args.reuse_inactive) for r in args.regimes for p in PAIRS]
    if args.workers == 1:
        cells = list(map(trained_cell, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
            cells = list(pool.map(trained_cell, tasks))
    rows = [row for cell in cells for row in cell['rows']]
    totals = summarize(rows)
    if set(args.regimes) != set(SNR):
        return totals['passed']
    expected = {(regime, pair, arm) for regime in SNR for pair in PAIRS for arm in GAUGE_ARMS}
    conditions = {"intact", "first_pilot_row", "last_pilot_row", "all_pilots", "whole_field",
                  "degenerate_pilots", "zero_noise", "physical_noiseless", "all_pilots_zero_noise",
                  "whole_zero_noise"}
    cases = {(row['regime'], row['pair'], row['arm'], row['condition']) for row in rows}
    coverage = {(r, p, a, c) for r, p, a in expected for c in conditions}
    totals['passed'] &= (len(rows) == len(cases) and cases == coverage and
                         all(row['scales'] == list(SCALES) for row in rows))
    references = {k: v for cell in cells for k, v in cell.get('reuse_receipt', {}).items()}
    ex.save(candidate_dir(args.cue) / "audit_gauge.json", dict(sources=sources(), steps=args.steps, cue=args.cue,
                                         defer=args.defer, reuse_receipts=references,
                                         scales=list(SCALES), failed_cases=[r for r in rows if not r['passed']],
                                         summary=totals))
    print(totals, flush=True)
    return totals['passed']


def suite(args):
    test_hashes = {path.name: digest(path) for path in sorted((ex.ROOT / 'tests').glob('test_*.py'))}
    initial_sources = sources()
    tests = unittest.defaultTestLoader.discover(str(ex.ROOT / "tests"), top_level_dir=str(ex.ROOT.parent))
    log = io.StringIO()
    original = ex.ROOT / 'results/iter4_checks/invariants.json'

    def save_invariants(path, result):
        if path.resolve() != original.resolve():
            raise ValueError(f"Unexpected invariant test write {path}")
        ex.save(OUT / "audit_suite_invariants.json", result)

    # The historical test emits a diagnostic report. Route that one write to this iteration.
    with patch("p6.tests.test_p6_invariants.save", save_invariants):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(tests)
    same_sources = (initial_sources == sources() and test_hashes == {
        path.name: digest(path) for path in sorted((ex.ROOT / 'tests').glob('test_*.py'))})
    passed = result.wasSuccessful() and not result.skipped and result.testsRun > 0 and same_sources
    ex.save(OUT / "audit_suite.json", dict(sources=initial_sources, tests=test_hashes, stable_sources=same_sources,
                                         run=result.testsRun, skipped=len(result.skipped),
                                         failed=len(result.failures), errors=len(result.errors),
                                         suite_log=log.getvalue(), passed=passed))
    print(log.getvalue(), flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("first", "suite", "gauge"))
    parser.add_argument("--steps", choices=(36, 48), type=int, default=36)
    parser.add_argument("--cue", choices=(0., .05), type=float, default=0.)
    parser.add_argument("--defer", action="store_true")
    parser.add_argument("--reuse-inactive", action="store_true", help="Verify and reuse R1/R3 cue-on audits")
    parser.add_argument("--workers", choices=range(1, 9), type=int, default=1)
    parser.add_argument("--regimes", type=int, nargs="+", choices=SNR, default=list(SNR))
    args = parser.parse_args()
    ex.setup()
    sys.exit(0 if globals()[args.action](args) else 1)
