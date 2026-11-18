"""Observe the preidentified half-gain R2/901 erasure case without changing model source."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import math
import sys
from unittest.mock import patch

import torch

from p5b.diagnostics import SCALES, erased
from p6 import axial_em_transformer as solver, experiment as ex, iter5_dev as dev
from p6.data import dev_frame
from p6.iter5_audit import digest


def capture(model, obs, suppress):
    tensors, counts = {}, Counter()
    watched = {fn.__code__: name for name, fn in (
        ('gauge', solver._gauge), ('init', solver._initialize), ('bank', solver._soft_chirp_residual),
        ('demap', solver._demap_input), ('parity', solver.soft_parity_symbols), ('handle', solver._physical_handles))}

    def profile(frame, event, result):
        if event != 'return' or frame.f_code not in watched:
            return
        name = watched[frame.f_code]
        index = counts[name]
        counts[name] += 1
        loc = frame.f_locals

        def put(key, value, charge='neutral'):
            tensors[f'{name}_{index}.{key}'] = (value, charge)

        if name == 'gauge':
            put('normalized_y', result[0], 'phase')
            put('normalized_noise', result[1])
            put('scale', result[2], 'magnitude')
        elif name == 'init':
            put('amp', result[0], 'phase')
            put('rot', result[1])
            put('selected_chirp', loc['best'])
        elif name == 'demap':
            put('clean', loc['y'], 'phase')
            put('CFR', loc['h'], 'phase')
            put('equalized_z', result[0])
            put('variance', result[1])
        elif name == 'parity':
            put('posterior_symbol', result)
        elif name == 'bank':
            put('taps', loc['taps'], 'phase')
            for key in ('coherence', 'weights', 'current', 'target'):
                put(key, loc[key])
            put('residual_raw', result)
        elif name == 'handle':
            put('field', loc['fields'], 'phase')
            put('ray_amp', loc['amp'], 'phase')
            put('phase_increment', result.phase)
            put('delta_h', result.delta_h, 'phase')
            if loc['bank_residual'] is not None:
                put('bank_gate', loc['gate'])
                put('applied_residual', loc['bank_residual'])

    original = solver._soft_chirp_residual
    def zero(*args):
        return torch.zeros_like(original(*args))

    manager = patch.object(solver, '_soft_chirp_residual', zero) if suppress else nullcontext()
    with manager:
        sys.setprofile(profile)
        try:
            llr, h = model(obs)
        finally:
            sys.setprofile(None)
    tensors['output.llr'] = (llr, 'neutral')
    tensors['output.physical_cfr'] = (h, 'physical')
    return tensors


def compare(lhs, rhs, gain):
    rows = {}
    assert lhs.keys() == rhs.keys()
    for name, (reference, charge) in lhs.items():
        transformed = rhs[name][0]
        factor = {'phase': gain / .5, 'physical': gain, 'magnitude': gain.abs()}.get(charge)
        if factor is not None:
            transformed = transformed / factor.reshape([-1] + [1]*(transformed.ndim - 1))
        error = (transformed - reference).abs()
        rows[name] = dict(charge=charge, max_per_burst=error.reshape(len(error), -1).amax(-1).tolist(),
                          peak_abs_per_burst=reference.abs().reshape(len(error), -1).amax(-1).tolist())
    return rows


@torch.no_grad()
def main():
    ex.setup()
    src = dev.OUT / 'audit_gauge_r2_901.json'
    receipt = ex.read(src)
    failed = [r for r in receipt['rows'] if not r['passed']]
    assert len(failed) == 1 and (failed[0]['arm'], failed[0]['condition']) == ('parity', 'first_pilot_row')
    cond = failed[0]
    base = erased(dev_frame(2, 6, 901, 2).obs, cond['condition'])
    gen = torch.Generator().manual_seed(cond['phase_seed'])
    for scale in SCALES:
        phi = 2*math.pi*torch.rand(len(base.y), dtype=torch.float64, generator=gen)
        if scale == .5:
            break
    gain = torch.polar(torch.full_like(phi, .5), phi)
    other = replace(base, y=gain[:, None, None, None]*base.y, noise_var=.25*base.noise_var)
    checkpoint = dev.location(2, 901, 'parity', 48, .05, False).with_suffix('.pt')
    assert digest(checkpoint) == receipt['checkpoints']['parity']['sha256']
    model = dev.model_for(2, 901, 'parity', .05, False).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state'])
    fixed = dev.model_for(2, 901, 'physics_parity', .05, False).eval()
    results = {}
    for name, receiver, suppress in (('trained', model, False), ('trained_bank_off', model, True),
                                     ('fixed_parity', fixed, False)):
        lhs, rhs = [capture(receiver, obs, suppress) for obs in (base, other)]
        rows = compare(lhs, rhs, gain)
        llr, new = lhs['output.llr'][0], rhs['output.llr'][0]
        err = (llr-new).abs()
        coord = tuple(int(j) for j in torch.unravel_index(err.argmax(), err.shape))
        flips = ((llr[:, ~base.pilot_mask] > 0) != (new[:, ~base.pilot_mask] > 0)).count_nonzero().item()
        results[name] = dict(max_delta=float(err.max()), worst_coordinate=coord, flips=flips, stages=rows)
    out = dict(pair=901, regime=2, arm='parity', steps=48, cue=.05, condition=cond['condition'],
               checkpoint_sha256=digest(checkpoint), condition_receipt_sha256=digest(src), gain=.5,
               phase_seed=cond['phase_seed'], phases=phi.tolist(), results=results,
               demap_order=['initial', 'first_redemap', 'EM2_input', 'EM3_input', 'final'],
               elapsed_em_pass_index_for_banks=[1,2],
               charged_comparison='Divide transformed tensors by the expected phase, full gain, or magnitude; report absolute max by burst.')
    ex.save(dev.OUT/'audit_r2_901_failure_trace.json', out)
    for name, row in results.items():
        print(name, row['max_delta'], row['worst_coordinate'], row['flips'])


if __name__ == '__main__':
    main()
