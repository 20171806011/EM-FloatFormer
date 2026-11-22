"""Fold-safe adapter for 5f88a5b. No global monkeypatch or source-tree mutation."""
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import hypothesis_frontend as front
import ofdm_data as data


def _check_module(module, sha):
    raw = Path(inspect.getsourcefile(module)).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != sha:
        raise RuntimeError(f"unexpected source for {module.__name__}: {blob}; expected {sha}")


def _specialize(fn, changes, bindings):
    # Only the two verified local modules below are accepted. Keep native feature algebra.
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new in changes:
        if src.count(old) != 1:
            raise RuntimeError(f"unrecognized source for {fn.__name__}: {old}")
        src = src.replace(old, new)
    scope = dict(fn.__globals__, **bindings)
    exec(compile(src, f"<fold:{fn.__name__}>", "exec"), scope)
    return scope[fn.__name__]


def make_builder(train_ref_bank, ll_temp=6., p_stay=.8):
    """Return the builder for native_bridge.build_contexts; construct once per run."""
    _check_module(data, "e2ed1a2e7a71415ef205498bb3f104827a1468ae")
    _check_module(front, "735233d5ad34f77788f7ea2536048eb489be34ae")
    refs = np.array(train_ref_bank, dtype=np.complex128, copy=True)
    if refs.ndim != 2 or refs.shape[1] != len(data.USED_SCS) or len(refs) == 0 or not np.isfinite(refs).all():
        raise ValueError("expected nonempty finite [M,101] reference bank")
    if not (ll_temp > 0 and 0 < p_stay < 1):
        raise ValueError("invalid HMM calibration")
    refs.flags.writeable = False
    from native_bridge import support_view
    # Validity depends only on the visible pilot layout + fold, so specialize once per layout/fold.
    kernels = {}

    def build(view, valid, pilots, fold):
        key = (tuple(pilots), fold)
        if key not in kernels:
            _, expected = support_view(np.zeros((14, 128), dtype=complex), pilots, fold)
            visible = expected[:, data.USED_SCS]
            data_mask = visible & data.DATA_MASK_101
            masked_noise = _specialize(data.estimate_noise_var_from_grid, [
                ("np.abs(y_grid[:, USED_SCS]) ** 2", "np.abs(y_grid[:, USED_SCS][visible]) ** 2"),
                ("sym2_mask = DATA_MASK_101[PILOT_SYM]", "sym2_mask = data_mask[PILOT_SYM]"),
            ], {"visible": visible, "data_mask": data_mask})

            def box(a):
                return sliding_window_view(np.pad(a, 1, mode="edge"), (3, 3)).sum(axis=(-2,-1))

            count = box(visible.astype(float))
            def guard(y_used, h_base, n0_base):
                signal = np.abs(h_base)**2
                expected_energy = np.maximum(signal[None,:] + n0_base, 1e-12)
                energy = box(np.where(visible, np.abs(y_used)**2, 0.))
                energy = np.divide(energy, count, out=np.broadcast_to(expected_energy, count.shape).copy(), where=count>0)
                ratio = energy/expected_energy
                cap = 40*max(n0_base, .5*float(np.median(signal)))
                nv = np.where(ratio>1.75, np.minimum(cap, np.maximum(n0_base, energy-signal)), n0_base)
                return nv, np.log(np.maximum(ratio, 1e-6))

            masked_front = _specialize(front.build_shared_hypothesis_bundle, [
                ("dmask = DATA_MASK_101.astype", "dmask = data_mask.astype"),
                ("np.mean(np.maximum(excess_log, 0.0), axis=1)",
                 "(np.sum(np.maximum(excess_log, 0.0)*visible, axis=1)/visible.sum(axis=1))"),
            ], {"estimate_noise_var_from_grid": masked_noise, "compute_nbi_burst_variance_guard": guard,
                "visible": visible, "data_mask": data_mask})
            kernels[key] = expected, masked_front
        expected, masked_front = kernels[key]
        if not np.array_equal(valid, expected) or np.any(view[~expected] != 0):
            raise ValueError("use canonical support_view; expected zero-filled withheld samples")
        b = masked_front(view, np.asarray(pilots), refs, ll_temp=ll_temp, use_nbi_guard=True)
        lw = front.hmm_smooth_log_posterior(b.sym_ll, front.HYP_LOG_PRIOR, ll_temp=ll_temp, p_stay=p_stay)
        return {"hLS": b.h_ls, "hCA": b.h_ca, "nv": b.nv_grid, "logw": lw.T,
                "cplx": b.cplx_feats, "scal": b.scal_feats}

    return build
