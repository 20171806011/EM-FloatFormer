"""Replay the final capped parity checkpoints for R2 sign and R3 filtering diagnostics."""
from unittest.mock import patch

import torch

from p5.classical import _time
from p5.wavefield_sim import CP, ray_phase, steer
from p6 import axial_em_transformer as solver
from p6 import study_dev as study
from p6.data import PAIRS, dev_frame, dev_seed


def _wrapped_mae(estimate, target):
    residual = torch.exp(1j * (estimate - target))
    aligned = residual * residual.mean(1, keepdim=True).conj()
    return aligned.angle().abs().mean(1)


def r2_observations(model, frame):
    initial, increments = [], []
    initialize = solver._initialize
    read = solver._physical_handles

    def capture_init(*args):
        result = initialize(*args)
        initial.append(result[1])
        return result

    def capture_increment(*args):
        handles = read(*args)
        increments.append(handles.phase)
        return handles

    with patch.object(solver, "_initialize", capture_init), patch.object(
            solver, "_physical_handles", capture_increment):
        outputs = model(frame.obs)
    if len(initial) != 1 or len(increments) != study.DEPTH:
        raise ValueError("Unexpected number of R2 initialization or trajectory observations")
    _, t, n, _ = initial[0].shape
    middle = n // 2
    clock = _time(t, n)[:, middle]
    duration = (t * (n + CP) - 1) / n
    truth = ray_phase(frame.truth.nu_start, frame.truth.nu_end, clock, duration).mT
    # The initial public bank changes by less than pi between adjacent symbol
    # centers. Add the stored, real-valued increments before any phase wrapping.
    bank = solver._centered_angles(initial[0][:, :, middle].mT).mT
    final = bank + torch.stack([update[:, :, middle] for update in increments]).sum(0)
    truth_slope, initial_slope, final_slope = [
        (state[:, -1] - state[:, 0]) / (clock[-1] - clock[0]) for state in (truth, bank, final)]
    if not torch.equal(truth_slope.sign(), frame.truth.nu_start.sign()):
        raise ValueError("True R2 net slope has unexpected sign")
    difference = bank - truth
    return outputs, dict(span=difference.amax(1) - difference.amin(1), truth=truth_slope,
                         initial=initial_slope, final=final_slope,
                         phase_mae_initial=_wrapped_mae(bank, truth),
                         phase_mae_final=_wrapped_mae(final, truth))


def sign_summary(rows):
    values = {k: torch.cat([row[k].flatten() for row in rows]) for k in rows[0]}
    truth, initial, final = (values[k] for k in ("truth", "initial", "final"))
    initial_right, initial_wrong = initial * truth > 0, initial * truth < 0
    final_right, final_wrong = final * truth > 0, final * truth < 0
    band = (values["span"] > .5) & (values["span"] <= 3.)
    groups = {"all": torch.ones_like(band), "initial_wrong_sign": initial_wrong,
              "initial_correct_sign": initial_right, "initial_span_over_0.5_through_3": band,
              "initial_span_over_3": values["span"] > 3.,
              "wrong_sign_and_initial_span_over_0.5_through_3": initial_wrong & band}
    result = {}
    for name, selected in groups.items():
        count = int(selected.sum())

        def average(tensor):
            return tensor[selected].mean().item() if count else None

        result[name] = dict(
            burst_rays=count, initial_wrong=int((initial_wrong & selected).sum()),
            final_wrong=int((final_wrong & selected).sum()),
            initial_zero_slope=int(((initial == 0) & selected).sum()),
            final_zero_slope=int(((final == 0) & selected).sum()),
            repaired=int((initial_wrong & final_right & selected).sum()),
            regressed=int((initial_right & final_wrong & selected).sum()),
            initial_net_slope_mae=average((initial - truth).abs()),
            final_net_slope_mae=average((final - truth).abs()),
            initial_aligned_phase_mae_rad=average(values["phase_mae_initial"]),
            final_aligned_phase_mae_rad=average(values["phase_mae_final"]))
    return result


def r3_observations(model, frame):
    calls = []
    apply_filter = solver._filter_path_amplitudes

    def capture(amp, gain):
        filtered = apply_filter(amp, gain)
        calls.append((amp, gain, filtered))
        return filtered

    with patch.object(solver, "_filter_path_amplitudes", capture):
        outputs = model(frame.obs)
    labels = ["pilot_redemap", *(f"waveform_solve_{step}" for step in range(1, study.DEPTH + 1))]
    # Compare with the realized ULA direction only after model inference.
    truth_mode = steer(frame.truth.angles, frame.obs.y.shape[1]).mT

    def true_off_mode_power(amp):
        projection = truth_mode * (truth_mode.conj() * amp).sum(1, keepdim=True)
        return (amp - projection).abs().square().sum(1)

    stages = {}
    for name, (amp, gain, filtered) in zip(labels, calls, strict=True):
        projection = apply_filter(amp, torch.ones_like(gain))
        # The projector is held at its observed pre-filter value on both sides.
        stages[name] = dict(gain=gain, before=(amp - projection).abs().square().sum(1),
                            after=(filtered - projection).abs().square().sum(1),
                            true_before=true_off_mode_power(amp), true_after=true_off_mode_power(filtered))
    return outputs, stages


def filter_summary(rows):
    values = {k: torch.cat([row[k].flatten() for row in rows]) for k in rows[0]}
    g = values["gain"]

    def rms_ratio(before, after):
        denominator = values[before].sum().item()
        return (values[after].sum().item() / denominator)**.5 if denominator > 0 else None

    return dict(burst_ray_filter_applications=g.numel(), positive=int((g > 0).sum()),
                negative=int((g < 0).sum()), zero=int((g == 0).sum()),
                positive_percent=100 * (g > 0).double().mean().item(),
                mean_gain=g.mean().item(), median_gain=g.quantile(.5).item(),
                p05_gain=g.quantile(.05).item(), p95_gain=g.quantile(.95).item(),
                estimated_mode_off_rms_ratio_after_over_before=rms_ratio("before", "after"),
                true_mode_off_rms_ratio_after_over_before=rms_ratio("true_before", "true_after"))


def summarize_stages(rows):
    result = {stage: filter_summary([row[stage] for row in rows]) for stage in rows[0]}
    result["all_waveform_solves"] = filter_summary(
        [stage for row in rows for name, stage in row.items() if name.startswith("waveform_solve_")])
    return result


@torch.inference_mode()
def run():
    result = dict(
        implementation=study.implementation(), updates=study.UPDATES, loss="capped",
        pair_ids=PAIRS, stream=6, batch_index=0, validation_bursts_per_pair=32,
        r2_definition=("Net slope in radians per useful OFDM-symbol duration between the first and "
                       "last mid-sample. True slope uses integrated simulator Doppler. Wrong sign "
                       "means the fitted and true slopes have opposite signs. Phase MAE removes "
                       "one constant angle per burst-ray and uses wrapped mid-sample errors."),
        r3_definition=("Each stage compares the same waveform fit immediately before and after "
                       "the branch. The estimated rank-one projector is held fixed. A second "
                       "projection uses the true normalized ULA steering vector after inference. "
                       "RMS ratios pool squared cross-antenna residual magnitudes."),
        checkpoint_sha256={}, r2={}, r3={})
    for arm in study.ex.PARITY_ARMS:
        for regime, inspect, summarize in ((2, r2_observations, sign_summary),
                                            (3, r3_observations, summarize_stages)):
            groups = {}
            observations = []
            for pair in PAIRS:
                frame = dev_frame(regime, 6, pair, 32, 0)
                model = study.restored(regime, pair, study.DEPTH, arm, loss="capped")
                receipt = study.ex.read(study.location(("capped", regime, pair, arm)))
                if receipt["score_seed"] != dev_seed(regime, 6, pair, 0):
                    raise ValueError("Checkpoint and replay score seeds differ")
                outputs, row = inspect(model, frame)
                replay = study.ex.score([frame], lambda _: outputs)
                study.assert_replay(replay, receipt["score"])
                observations.append(row)
                groups[pair] = summarize([row])
                result["checkpoint_sha256"][f"R{regime}/{pair}/{arm}"] = receipt["checkpoint_sha256"]
            result[f"r{regime}"][arm] = dict(by_pair=groups, pooled=summarize(observations))
    return result


if __name__ == "__main__":
    study.ex.setup()
    target = study.ex.RESULTS / "iter3_checks" / "physical_detail.json"
    study.ex.save(target, run())
    print(f"Wrote {target}", flush=True)
