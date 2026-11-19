"""Evaluation-only posterior calibration and final bit log-loss summaries.

Only the evaluator calls this module.  It reconstructs transmitted payload
symbols from ``frame.bits`` after receiver inference; it never supplies truth
to the receiver.  The caller selects the three actual M-step posterior points
and supplies their means/variances.  Pilot locations are excluded throughout.
"""

import math
import numbers

import torch
from torch.nn import functional as F

from p4.simulator import q16


VARIANCE_EDGES = (0., .01, .05, .1, .2, .4, .8)


def _finite_number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Calibration summary is nonfinite; no clipping is performed")
    return result


def _finite_json(value):
    """Copy optional diagnostics to ordinary finite JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return _finite_json(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _finite_number(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Diagnostic dictionary keys must be strings")
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    raise TypeError(f"Unsupported calibration diagnostic type: {type(value).__name__}")


def _bit_summary(llr, bits, mask):
    expected = (bits.shape[0], *mask.shape, 4)
    if llr.shape != expected or llr.is_complex():
        raise ValueError("Expected real grid bit LLRs [B,T,N,4]")
    payload = llr.detach()[:, ~mask].double()
    if not bool(torch.isfinite(payload).all()):
        raise ValueError("Payload bit LLRs must be finite; no clipping is performed")
    signed_bits = 2 * bits.to(device=payload.device, dtype=torch.float64) - 1
    # Positive LLR favors bit one. softplus avoids overflow in log(1+exp()).
    loss = F.softplus(-signed_bits * payload)
    absolute = payload.abs()
    count = payload.numel()
    return dict(bce_sum=_finite_number(loss.sum()), bit_count=count,
                sum_abs_llr=_finite_number(absolute.sum()),
                mean_abs_llr=_finite_number(absolute.mean()) if count else 0.,
                max_abs_llr=_finite_number(absolute.max()) if count else 0.)


def _moment_totals(variance, squared_error):
    return dict(count=variance.numel(),
                sum_predicted_variance=_finite_number(variance.sum()),
                sum_squared_error=_finite_number(squared_error.sum()))


@torch.no_grad()
def calibration(frame, llr, traces):
    """Return a finite JSON dictionary for one batch, without modifying inputs.

    ``frame.bits`` has shape B,payload_RE,4 and uses the frozen P4 Gray-QAM
    mapping.  ``llr`` has shape B,T,N,4 with positive values favoring bit one.
    ``traces`` must contain exactly three dictionaries with unique integer
    ``step`` labels, complex ``mean`` and real ``variance`` tensors B,T,N.
    The caller must select actual M-step posteriors, not other intermediate
    or final-demapper quantities.  Optional ``channel_llr`` receives a separate
    payload bit summary; optional ``diagnostics`` are copied as finite JSON.

    Fixed variance intervals are [0,.01), [.01,.05), [.05,.1), [.1,.2),
    [.2,.4), [.4,.8), [.8,infinity).  The last upper bound is encoded as null,
    not non-standard JSON Infinity.  Counts and sums can be added across
    batches without averaging averages.  These are descriptive calibration
    measurements, not new prespecified efficacy endpoints.

    Nonfinite payload inputs or summaries raise instead of being clipped or
    silently omitted.  Pilots never enter the symbol or bit measurements.
    """
    mask = frame.obs.pilot_mask
    bits = frame.bits.detach()
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("Expected a boolean [T,N] pilot mask")
    if bits.ndim != 3 or bits.shape[-1] != 4 or bits.shape[1] != int((~mask).sum()):
        raise ValueError("Payload bits must align with the frame pilot mask")
    if bool(((bits != 0) & (bits != 1)).any()):
        raise ValueError("Expected binary payload bits")
    traces = list(traces)
    if len(traces) != 3:
        raise ValueError("Exactly three selected M-step posterior traces are required")
    if any(type(trace.get("step")) is not int for trace in traces):
        raise ValueError("Each trace needs an integer step label")
    if len({trace["step"] for trace in traces}) != 3:
        raise ValueError("M-step trace labels must be unique")
    shape = (bits.shape[0], *mask.shape)
    truth_grid = torch.zeros(shape, dtype=torch.complex128, device=bits.device)
    truth_grid[:, ~mask] = q16(bits.long()).to(torch.complex128)
    steps = []
    for trace in traces:
        mean, variance = trace["mean"].detach(), trace["variance"].detach()
        if mean.shape != shape or variance.shape != shape or not mean.is_complex() or variance.is_complex():
            raise ValueError("Expected complex mean and real variance grids [B,T,N]")
        payload_mean = mean[:, ~mask].to(torch.complex128)
        payload_variance = variance[:, ~mask].double()
        if not bool(torch.isfinite(payload_mean).all()):
            raise ValueError("Payload symbol means must be finite")
        if bool((~torch.isfinite(payload_variance) | (payload_variance < 0)).any()):
            raise ValueError("Payload symbol variances must be finite and nonnegative")
        truth = truth_grid[:, ~mask].to(payload_mean.device)
        squared_error = (payload_mean - truth).abs().square()
        boundaries = payload_variance.new_tensor(VARIANCE_EDGES[1:])
        indices = torch.bucketize(payload_variance.contiguous(), boundaries, right=True)
        bins = []
        for index, lower in enumerate(VARIANCE_EDGES):
            upper = VARIANCE_EDGES[index + 1] if index + 1 < len(VARIANCE_EDGES) else None
            selected = indices == index
            bins.append(dict(index=index, lower=lower, upper=upper,
                             **_moment_totals(payload_variance[selected], squared_error[selected])))
        row = dict(step=trace["step"], total=_moment_totals(payload_variance, squared_error), bins=bins)
        if trace.get("channel_llr") is not None:
            row["channel"] = _bit_summary(trace["channel_llr"], bits, mask)
        if trace.get("diagnostics") is not None:
            row["diagnostics"] = _finite_json(trace["diagnostics"])
        steps.append(row)
    return dict(kind="evaluation_only_posterior_calibration", batch_size=bits.shape[0],
                payload_symbols=bits.shape[0] * bits.shape[1],
                variance_bin_edges=[*VARIANCE_EDGES, None],
                bin_convention="left_closed_right_open; null upper bound means infinity",
                steps=steps, final=_bit_summary(llr, bits, mask))
