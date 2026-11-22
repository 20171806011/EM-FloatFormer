"""Use the fixed commit's actual compact PyTorch demapper with two cross-fit packets.

Place research/realdata_ofdm_stress_and_theory from 5f88a5b on PYTHONPATH.
Inputs are in-memory NumPy arrays. No dataset loader, training, or CLI replay.
Node >= 18 required. Subprocess/JSON boundary is for correctness tests, not timing.
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from group_c_transport_attention import differentiable_mixture_llrs_compact
from ofdm_data import DATA_COORDS_101, USED_SCS, PILOT_SYM, PILOT_SCS_14


def _ri(a):
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def _tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)


def support_view(y, pilots, fold):
    """Grid and boolean validity mask. Original 14 pilot slots are never payload."""
    if fold not in (0, 1) or np.shape(y) != (14, 128):
        raise ValueError("expected fold 0/1 and a 14x128 grid")
    pilots = np.asarray(pilots)
    if pilots.ndim != 1 or pilots.size == 0 or not np.isin(pilots, PILOT_SCS_14).all() or np.any(np.diff(pilots) <= 0):
        raise ValueError("expected a sorted, unique subset of visible pilot bins")
    s, k = DATA_COORDS_101.T
    target = k % 2 == fold
    valid = np.ones((14, 128), dtype=bool)
    valid[s[target], USED_SCS[k[target]]] = False
    hidden_pilots = np.setdiff1d(PILOT_SCS_14, pilots)
    valid[PILOT_SYM, hidden_pilots] = False
    view = np.where(valid, y, 0j)
    view.flags.writeable = False
    valid.flags.writeable = False
    return view, valid


def classical_head(y, pilots, fold, ctx, method):
    """All fields of ctx must have been computed without fold's target evidence."""
    logw = np.asarray(ctx["logw"], dtype=float)
    # JSON cannot carry -Inf. A finite sentinel is zero mass at demapper precision.
    if logw.shape != (14, 17) or np.isnan(logw).any() or np.isposinf(logw).any():
        raise ValueError("invalid log weights")
    view, _ = support_view(y, pilots, fold)
    packet = {"y": _ri(view), "visiblePilots": np.asarray(pilots).tolist(),
              "fold": fold, "nuisance": {"hLS": _ri(ctx["hLS"]), "hCA": _ri(ctx["hCA"]),
                                         "nv": np.asarray(ctx["nv"]).tolist()},
              "logw": np.maximum(logw, -1e30).tolist(), "method": method}
    script = Path(__file__).with_name("estimator.cjs")
    out = subprocess.run(["node", str(script), "--head-json"],
                         input=json.dumps(packet, allow_nan=False), text=True,
                         capture_output=True, check=True)
    head = json.loads(out.stdout)
    return _tensor(head["logw"]), _tensor(head["dphi"]), _tensor(head["lognv"]).unsqueeze(-1)


@torch.inference_mode()
def receive(y, pilots, contexts, method, *, phase_only=False):
    """Return [1400,4] LLR in repository order.

    method is "continuous", "discrete", or a frozen CPU eval-mode learning head.
    contexts[0:2] provide hLS, hCA, nv, logw; learned methods additionally need
    fold-safe cplx [14,17,4] and scal [14,17,10]. logw is [14,17].
    """
    if len(contexts) != 2 or np.shape(y) != (14, 128):
        raise ValueError("expected two fold contexts and one fftshifted 14x128 grid")
    s, k = DATA_COORDS_101.T
    yd = np.asarray(y)[s, USED_SCS[k]]
    result = np.empty((len(s), 4), dtype=np.float32)
    for fold, ctx in enumerate(contexts):
        if isinstance(method, str):
            head = classical_head(y, pilots, fold, ctx, method)
        else:
            if method.training:
                raise ValueError("expected frozen head in eval mode")
            c = np.asarray(ctx["cplx"])
            head = method(_tensor(c.real), _tensor(c.imag), _tensor(ctx["scal"]), _tensor(ctx["logw"]))
        if phase_only:
            # Apply to all four methods to isolate the phase correction.
            head = (_tensor(ctx["logw"]), head[1], torch.zeros_like(head[2]))
        h = np.asarray(ctx["hLS"])[k]
        g = np.asarray(ctx["hCA"])[k]
        nv = np.asarray(ctx["nv"])[s, k]
        llr = differentiable_mixture_llrs_compact(
            _tensor(yd.real), _tensor(yd.imag), _tensor(h.real), _tensor(h.imag),
            _tensor(g.real), _tensor(g.imag), _tensor(nv), *head
        )[0].cpu().numpy()
        target = k % 2 == fold
        result[target] = llr[target]
    return result


def build_contexts(y, pilots, builder):
    """Call once per grid; reuse the returned two contexts across all methods.

    builder(view, valid, pilots, fold) uses only these arguments and pre-frozen
    resources. Exclude invalid entries from every statistic and normalize by
    the valid count. Closure access to the original grid defeats isolation.
    """
    return [builder(*support_view(y, pilots, f), np.asarray(pilots).copy(), f)
            for f in (0, 1)]
