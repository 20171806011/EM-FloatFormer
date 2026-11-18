"""Stage P4-C receiver mechanism and precision probes on the signed P4-A implementation."""
from __future__ import annotations

import argparse
import math
import platform
import statistics
import time
from contextlib import ExitStack, contextmanager
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from p4.classical_baselines import bussgang_lmmse, iterative_dd_pa_cancellation
from p4.em_floatformer import ComplexLinear, GreenAttention, JointPhysicalLoss, compute_gauge_anchor
from p4.factorial import HERE, PROTOCOL, instantiate_ablation_arm
from p4.factorial import (
    compute_interactions, compute_main_effects, get_yates_matrix, observation,
    run_full_factorial_evaluation, scaled_inputs, training_run, write_json,
)
from p4.stage3_factorial_confirm import run as run_ablation_confirmatory
from p4.stage3_precision import audit as run_mixed_precision_audit, shadowing as run_shadowing_swing
from p4.simulator import PROFILES


@lru_cache(maxsize=3)
@torch.enable_grad()
def probe_model(arm_id=1):
    return training_run(arm_id, PROTOCOL["dev_seeds"][0])


def check_arm_parameters():
    return [sum(p.numel() for p in instantiate_ablation_arm(a).parameters()) for a in range(1, 9)]


@torch.no_grad()
def test_ablation_forward(batch_size=1):
    frame = observation(batch_size, seed=73501)
    for arm_id in range(1, 9):
        outputs = instantiate_ablation_arm(arm_id)(*scaled_inputs(frame, 1.0))
        if not all(torch.isfinite(x).all() for x in outputs):
            return False
    return True


def verify_decoupled_gradients(arm_id=4):
    frame = observation(4, seed=73502)
    yd = frame.yd.detach().requires_grad_(True)
    h_hat = instantiate_ablation_arm(arm_id)(frame.yp, yd, frame.noise_var).h_hat
    grad, = torch.autograd.grad(h_hat.abs().square().sum(), yd, allow_unused=True)
    return 0. if grad is None else float(grad.abs().max())


@torch.no_grad()
def check_arm3_phase_sensitivity():
    frame = observation(32, seed=73503, snr_db=20.)
    model = probe_model(3)
    phase = complex(math.cos(math.pi/4), math.sin(math.pi/4))
    base = model(*scaled_inputs(frame, 1.)).llrs
    return float((model(*scaled_inputs(frame, phase)).llrs - base).abs().max())


@torch.no_grad()
def compare_green_vs_real_ablation():
    frame = observation(16, seed=73504)
    obs = scaled_inputs(frame, 1.)
    return float((probe_model(1)(*obs).llrs - probe_model(3)(*obs).llrs).abs().mean())


def autocast_outputs(model, inputs, dtype):
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported autocast audit dtype {dtype}")
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        return model(*inputs)


@torch.no_grad()
def run_precision_forward(dtype=torch.float32):
    frame = observation(16, seed=73505)
    return autocast_outputs(probe_model(), scaled_inputs(frame, 1.), dtype).llrs


@torch.no_grad()
def finite_on(scales, dtypes):
    frame = observation(8, seed=73506)
    model = probe_model()
    return all(all(torch.isfinite(x).all().item() for x in autocast_outputs(model, scaled_inputs(frame, scale), dtype))
               for scale in scales for dtype in dtypes)


def audit_dynamic_range_finiteness():
    return finite_on([2.**p for p in range(-12, 13)], [torch.float32, torch.float16, torch.bfloat16])


def audit_gauge_mixed_precision():
    return audit_dynamic_range_finiteness()


def audit_fp16_scale(scale):
    return finite_on([scale], [torch.float16])


def audit_bf16_wide_range():
    return finite_on([2.**p for p in range(-20, 21, 2)], [torch.bfloat16])


def audit_half_precision_discrepancy():
    ref = run_precision_forward(torch.float32)
    return float((run_precision_forward(torch.float16) - ref).abs().max())


@torch.no_grad()
def check_zero_autocast():
    frame = observation(4, seed=73507)
    inputs = tuple(torch.zeros_like(x) for x in scaled_inputs(frame, 1.))
    return all(torch.isfinite(x).all().item() for x in autocast_outputs(probe_model(), inputs, torch.float16))


def check_autocast_gradients():
    frame = observation(4, seed=73508)
    model = instantiate_ablation_arm(1)
    with torch.autocast("cpu", dtype=torch.float16):
        loss = JointPhysicalLoss()(model(*scaled_inputs(frame, 1.)), frame, A=1.25)
    loss.backward()
    return any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())


@torch.no_grad()
def shadowing_autocast_1d(scales, dtype=torch.bfloat16):
    frame = observation(24, seed=73509)
    model = probe_model()
    base = autocast_outputs(model, scaled_inputs(frame, 1.), dtype).llrs
    max_error, flips, total, nonfinite = 0., 0, 0, 0
    for scale in scales:
        output = autocast_outputs(model, scaled_inputs(frame, scale), dtype)
        nonfinite += sum(int((~torch.isfinite(x)).sum()) for x in output)
        max_error = max(max_error, float((output.llrs - base).abs().max()))
        flips += int(torch.count_nonzero((output.llrs > 0) != (base > 0)))
        total += base.numel()
    return dict(max_logit_error=max_error, flip_rate=flips/total, nan_count=nonfinite, flips=flips, total_bits=total)


@torch.no_grad()
def mixed_autocast_1d(num_slots=100, snr_grid=(0,5,10,15,20)):
    """One-dimensional frames; the production backbone explicitly promotes to complex128."""
    if len(snr_grid) == 0 or num_slots < len(snr_grid):
        raise ValueError("Provide at least one frame per SNR")
    rows, total, flips, nonfinite = [], 0, 0, 0
    for idx, snr in enumerate(snr_grid):
        count = num_slots//len(snr_grid) + int(idx < num_slots%len(snr_grid))
        inp = scaled_inputs(observation(count, seed=71500+idx, snr_db=snr), 1.)
        ref = probe_model()(*inp).llrs
        for dtype in (torch.float16, torch.bfloat16):
            out = autocast_outputs(probe_model(), inp, dtype)
            nbad = sum(int((~torch.isfinite(x)).sum()) for x in out)
            nflip = int(torch.count_nonzero((ref > 0) != (out.llrs > 0)))
            rows.append(dict(dtype=str(dtype), snr_db=snr, frames=count, flips=nflip,
                             bits=ref.numel(), nonfinite=nbad, max_logit_error=float((ref-out.llrs).abs().max())))
            total += ref.numel()
            flips += nflip
            nonfinite += nbad
    return dict(rows=rows, total_bits=total, flips=flips, flip_rate=flips/total, nan_count=nonfinite)


@contextmanager
def reduced_projection_gemms(model, dtype):
    """Replace four real GEMMs of each complex linear projection by measured CPU low-precision GEMMs.

    Inputs and weights are rounded immediately before GEMM; additions/subtractions and
    the rest of the signed receiver remain wide. The stock forward is otherwise intact.
    """
    def replace(module, args, output):
        z = args[0]
        xr, xi = z.real.to(dtype), z.imag.to(dtype)
        wr, wi = module.weight_r.to(dtype), module.weight_i.to(dtype)
        rr = F.linear(xr, wr).double() - F.linear(xi, wi).double()
        ii = F.linear(xr, wi).double() + F.linear(xi, wr).double()
        if module.bias_r is not None:
            rr = rr + module.bias_r
            ii = ii + module.bias_i
        return torch.complex(rr, ii).to(output.dtype)

    with ExitStack() as hooks:
        for module in model.modules():
            if isinstance(module, ComplexLinear):
                hooks.enter_context(module.register_forward_hook(replace))
        yield


def quantize_storage(x, dtype):
    if x.is_complex():
        return torch.complex(x.real.to(dtype).float(), x.imag.to(dtype).float())
    return x.to(dtype).float()


@torch.no_grad()
def audit_storage_and_gemms(model):
    frame = observation(32, seed=73510, snr_db=15.)
    rows = []
    for scale in [1e-8, 2.**-12, 1., 2.**12, 1e8]:
        inputs = scaled_inputs(frame, scale)
        ref = model(*inputs).llrs
        for dtype in (torch.bfloat16, torch.float16):
            raw = tuple(quantize_storage(x, dtype) for x in inputs)
            rejected = sum(int((~torch.isfinite(x)).sum()) for x in raw)
            lost = sum(int(((x != 0) & (y == 0)).sum()) for x, y in zip(inputs, raw))
            row = dict(scale=scale, dtype=str(dtype), raw_input_nonfinite=rejected, raw_nonzero_to_zero=lost)
            if not rejected:
                raw_logits = model(*raw).llrs
                row["raw_storage_flip_count"] = int(torch.count_nonzero((raw_logits > 0) != (ref > 0)))
                row["raw_nonfinite_logits"] = int((~torch.isfinite(raw_logits)).sum())
            # Digitizing already normalized observations avoids raw hardware overflow.
            anchor = compute_gauge_anchor(inputs[0]).float()
            normalized = (inputs[0]/anchor, inputs[1]/anchor, inputs[2].reshape(-1,1)/anchor.square())
            norm_in = tuple(quantize_storage(x, dtype) for x in normalized)
            with reduced_projection_gemms(model, dtype):
                output = model(*norm_in)
            delta = (output.llrs-ref).abs()
            row.update(bits=ref.numel(), normalized_gemm_max_logit_error=float(delta.max()),
                       normalized_gemm_mean_logit_error=float(delta.mean()),
                       normalized_gemm_flip_count=int(torch.count_nonzero((output.llrs > 0) != (ref > 0))),
                       normalized_gemm_nonfinite=sum(int((~torch.isfinite(x)).sum()) for x in output))
            rows.append(row)
    return rows


@torch.no_grad()
def attention_probe(model, seed):
    """Compare mean frequency-lag attention mass with the known covariance magnitude.

    Use a mean over queries and frames at each circular lag, excluding self pairs;
    the comparison to permuted key positions uses exactly the same learned matrices.
    """
    rows = []
    for profile in ("equal4", "long12", "sparse12"):
        frame = observation(32, seed=seed, snr_db=15., profile=profile)
        size = frame.yp.shape[-1]
        pdp = torch.tensor(PROFILES[profile], dtype=torch.float64)
        target = torch.fft.fft(pdp, n=size).abs()
        target = target / target[0].item()
        perm_gen = torch.Generator().manual_seed(seed+10000)
        perms = [torch.randperm(size, generator=perm_gen) for _ in range(32)]
        ids = torch.arange(size)
        offsets = (ids[:,None]+ids[None,:]) % size

        def inspect(name):
            def capture(module, args, kwargs, output):
                _, weights = module.forward(*args, **kwargs, return_weights=True)
                for head, matrix in enumerate(weights.detach().mean(dim=0)):
                    def by_lag(mat):
                        return mat[ids[:,None], offsets].mean(0)
                    def r(vec):
                        return float(torch.corrcoef(torch.stack((vec[1:], target[1:])))[0,1])
                    lags = by_lag(matrix)
                    shuffled = [r(by_lag(matrix[:,p])) for p in perms]
                    rows.append(dict(profile=profile, module=name, head=head, lag_mean=lags.tolist(),
                        target_abs_covariance=target.tolist(), correlation_excluding_self=r(lags),
                        mean_key_shuffle_correlation=statistics.mean(shuffled),
                        shuffled_range=[min(shuffled),max(shuffled)]))
            return capture
        with ExitStack() as hooks:
            for name, module in model.named_modules():
                if isinstance(module, GreenAttention):
                    hooks.enter_context(module.register_forward_hook(inspect(name), with_kwargs=True))
            model(*scaled_inputs(frame, 1.))
    return rows


@torch.inference_mode()
def timings(model):
    """Median single-call CPU latency including demapping. Interleave execution order."""
    rows = []
    compare = {"EMFloatFormer": lambda y,z,v:model(y,z,v),
               "Bussgang-LMMSE": lambda y,z,v:bussgang_lmmse(y,z,v,A=1.25),
               "DD-PA (2 iterations)": lambda y,z,v:iterative_dd_pa_cancellation(y,z,v,A=1.25,iterations=2)}
    for batch in (1,8):
        inp = scaled_inputs(observation(batch, seed=73511), 1.)
        calls = list(compare)
        samples = {name:[] for name in calls}
        for name in calls:
            for _ in range(3):
                compare[name](*inp)
        for repeat in range(30):
            for index in range(len(calls)):
                name = calls[(repeat+index)%len(calls)]
                start = time.perf_counter_ns()
                compare[name](*inp)
                samples[name].append((time.perf_counter_ns()-start)/1e6)
        for name, vals in samples.items():
            rows.append(dict(method=name, batch=batch, repetitions=len(vals), median_ms=statistics.median(vals),
                             p10_ms=float(np.quantile(vals, .1)), p90_ms=float(np.quantile(vals, .9))))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed = PROTOCOL["dev_seeds"][0]
    model = probe_model()
    out = {"training_seed": seed, "execution_device":"cpu", "threads":torch.get_num_threads(),
           "pytorch":str(torch.__version__), "processor":platform.processor(),
           "native_autocast_note":"ComplexLinear receives complex128 and executes real float64 GEMMs in the signed model; autocast does not reduce these operations.",
           "gemm_probe_note":"Experimental FP16/BF16 storage after FP32 gauge anchoring and low-precision real projection GEMMs; summation of the four GEMM outputs, attention and nonlinear layers remain wide.",
           "gauge_fp32_boundary":shadowing_autocast_1d([1e-8, 2.**-12, 1., 2.**12, 1e8],torch.float32),
           "native_autocast":mixed_autocast_1d(),
           "storage_and_projection_precision":audit_storage_and_gemms(model),
           "initialized_attention":attention_probe(instantiate_ablation_arm(1,seed),73512),
           "trained_attention":attention_probe(model,73512),
           "wallclock_cpu":timings(model)}
    write_json(HERE/"results/p4c_mechanism_probes.json",out)

