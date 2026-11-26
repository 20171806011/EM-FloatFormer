"""Replay Stage 3D scores, audit CPU gates, and verify the proposed Stage 4 budget."""

import hashlib
import json
import math
import platform
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binomtest, norm, t as student_t


HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "stage3d_qwen_signed_phase" / "results"
CELLS = (
    "G1_main_extrap_both", "G2_interp_both", "G3_extrap_angle",
    "G4_extrap_freq", "H_c4_035",
)
ARMS = ("phase", "zero")
Q2_ARMS = (
    "none", "valid", "duplicate_copy_dedup", "duplicate_copy_raw",
    "independent_only", "valid_plus_independent", "fresh_resample",
)
Q2_REF_INDEX = (0, 1, 2, 2, 3, 4, 5)
TOL = 1e-12

R2 = {
    "seed": 202611,
    "arm_ids": ["0", "0b", "1", "2"],
    "steps_per_arm": 384,
    "pilot_steps_per_arm": 69,
    "batch": 32,
    "microbatch": 16,
    "eval_batch": 64,
    "max_prefix_tokens": 512,
    "max_candidate_tokens": 16,
    "max_train_seconds_per_step": 2.5,
    "max_eval_seconds_per_logical_row": 0.03,
    "max_peak_vram_mib": 65000,
    "natural_lock_coverage_min": 0.95,
    "cttc_cyclic_source_guard_symbols": 4096,
    "pluto_split_guard_frames": 100,
    "probability_parity_max_abs": 1e-5,
    "geometry_max_defect": 1e-6,
    "geometry_rank_relative_tolerance": 1e-6,
    "geometry_min_singular_ratio": 0.05,
    "geometry_max_relative_projection_residual": 0.10,
    "geometry_max_relative_ood_rmse": 0.25,
    "arm2_gamma": 0.15,
    "probability_floor": 1e-6,
    "domain_family_alpha": 0.05,
    "power_floor_each_joint_domain_claim": 0.90,
    "predictive_go_probability_min": 0.10,
    "absolute_chance_accuracy_increment": 0.02,
    "zero_accuracy_increment": 0.02,
    "zero_nll_increment": 0.02,
    "synthetic_1_minus_0b_nll_increment": 0.02,
    "real_two_minus_one_nll_improvement": 0.02,
    "cache_iq_nll_regret_ceiling": 0.05,
    "host_readout_nll_regret_ceiling": 0.05,
    "arm2_host_nll_increment_vs_physical": 0.005,
    "fixed_text_slot_nll_increment": 0.005,
    "brier_increment": 0.005,
    "ece_upper": 0.05,
    "ece_bins": 10,
    "nig_eta": 0.10,
    "pluto": {"packets": 2240, "q3_pairs": 2240, "clusters": 16,
              "reference_indices": [15, 18], "q3_lag_symbols": 126},
    "cttc": {"packets": 1440, "q3_pairs": 720, "clusters": 72, "q3_loop_pairs": 36,
             "sessions": 9, "holdout": "f708_snr10", "holdout_packets": 160, "holdout_q3_pairs": 80,
             "reference_indices": [126, 127], "q3_lag_symbols": 500000},
    "synthetic_blocks_per_cell": 8,
    "synthetic_mothers_per_block": 64,
    "synthetic_cells": ["G1", "G2", "G3", "G4"],
    "synthetic_primary_cells": ["G1", "G3"],
    "synthetic_primary_endpoint": "Q3 repeat pre-reveal",
    "synthetic_geometry_abs_omega_h_ell": {"G1": [0.40, 2, 2], "G2": [0.20, 2, 2],
                                          "G3": [0.20, 4, 4], "G4": [0.40, 1, 1]},
    "synthetic_ref_times_in_h_units": [0.25, 0.75],
    "pilot_dev_rows_per_arm": {"synthetic": 500, "pluto": 250, "cttc": 250},
    "calibration_dev_rows_per_arm": {"synthetic": 1500, "pluto": 750, "cttc": 750},
    "stage5": {"nominal_interval_coverage": 0.90, "lower_ci_coverage_min": 0.85,
               "max_median_width_fraction_on_identifiable_cases": 0.50,
               "max_timing_refinement_for_exact_alias": 0, "min_nll_improvement": 0.02},
    "stage6": {"adapter_rank_max": 16, "slot_count_max": 4, "trainable_params_max": 100000,
               "max_lcb_allowed_text_image_regression_pp": 2, "eig_error_max_nats": 0.005,
               "one_step_eig_regret_ceiling_nats": 0.01,
               "min_policy_gain_over_baselines_nats": 0.02, "read_count_max_per_turn": 1,
               "memory_headroom_fraction_min": 0.10},
}


def probabilities(scores, temperature=1.0):
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    return np.exp(scaled - logsumexp(scaled, axis=-1, keepdims=True))


def score(probs, labels):
    labels = np.broadcast_to(labels, probs.shape[:-1])
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-10, rtol=0)
    ties = probs == probs.max(axis=-1, keepdims=True)
    hit = np.take_along_axis(ties, labels[..., None], axis=-1)[..., 0] / ties.sum(axis=-1)
    true_prob = np.take_along_axis(probs, labels[..., None], axis=-1)[..., 0]
    loss = -np.log(np.clip(true_prob, TOL, 1.0))
    return {
        "acc": float(hit.mean()), "nll": float(loss.mean()),
        "n_prob_clipped": int(np.count_nonzero(true_prob < TOL)),
    }, loss


def check_metrics(actual, expected, diffs):
    for key in ("acc", "nll"):
        delta = abs(actual[key] - expected[key])
        assert delta <= TOL, (key, actual, expected)
        diffs.append(delta)


def paired_block_ci(loss_difference):
    # Collapse seeds, signs and rotations before inference on 64-mother blocks.
    block_means = loss_difference.mean(axis=tuple(range(loss_difference.ndim - 1)))
    blocks = block_means.reshape(-1, 64).mean(axis=1)
    mean = blocks.mean()
    error = student_t.ppf(0.975, len(blocks) - 1) * blocks.std(ddof=1) / math.sqrt(len(blocks))
    return {"blocks": len(blocks), "mean": float(mean), "lo95": float(mean - error), "hi95": float(mean + error)}


def replay_q3(data, summary, diffs):
    result = {}
    signs = np.arange(2)[:, None, None]
    for cell in CELLS:
        labels = data[f"{cell}__targets"]
        code = data[f"{cell}__codes"]
        raw_table = data[f"{cell}__raw_ll"]
        raw = raw_table[:, :, signs, code, :]
        assert raw.shape == raw_table.shape[:2] + labels.shape + (4,)
        cal = probabilities(raw, data["temps_q3"][..., None, None, None, None])
        uncal = probabilities(raw)
        direct, direct_loss = score(data[f"{cell}__direct_table_probs"], labels)
        iq, iq_loss = score(data[f"{cell}__full_iq_probs"], labels)
        row = {
            "mothers": labels.shape[-1], "logical_rows_per_host_arm": int(labels.size * raw.shape[1]),
            "full_iq": iq, "direct_s17": direct,
            "direct_minus_full_iq_nll": paired_block_ci(direct_loss - iq_loss), "host": {},
        }
        expected = summary["cells"].get(cell)
        if expected is not None:
            check_metrics(iq, expected["analytic_same_mothers"]["full_iq_known_param"], diffs)
            check_metrics(direct, expected["analytic_same_mothers"]["direct_s17_table"], diffs)

        for index, arm in enumerate(ARMS):
            locked, locked_loss = score(cal[index], labels)
            unit, unit_loss = score(uncal[index], labels)
            host_excess = paired_block_ci(locked_loss - direct_loss)
            unit_excess = float((unit_loss - direct_loss).mean())
            temperature_shift = float((locked_loss - unit_loss).mean())
            total = float((locked_loss - iq_loss).mean())
            assert abs(total - row["direct_minus_full_iq_nll"]["mean"] - unit_excess - temperature_shift) <= TOL
            row["host"][arm] = {
                "unit_temperature": unit, "locked_temperature": locked,
                "unit_temperature_minus_direct_nll": unit_excess,
                "locked_minus_unit_nll": temperature_shift,
                "locked_minus_direct_nll": host_excess, "locked_minus_full_iq_nll": total,
            }
            if expected is not None:
                check_metrics(locked, expected["arms"][arm], diffs)
                old = expected[f"{arm}_minus_direct_s17"]["nll_improv_t95"]
                for k1, k2 in (("mean", "mean"), ("lo95", "hi95"), ("hi95", "lo95")):
                    assert abs(host_excess[k1] + old[k2]) <= TOL
            else:
                subset, _ = score(cal[index, :, :, 0], labels[:, 0])
                old = summary["work_f_historical_035"]["arms"][arm]
                check_metrics(subset, {
                    "acc": old["seed_mean_matched_acc"],
                    "nll": np.mean([seed["matched_nll"] for seed in old["per_seed"]]),
                }, diffs)
                row["host"][arm]["work_f_identity_rotation_both_signs"] = subset
            if cell == CELLS[0]:
                defect = np.abs(cal[index, :, :, 2] - np.roll(cal[index, :, :, 0], 1, axis=-1)).max(axis=-1)
                full_s17 = data[f"{cell}__direct_table_probs"]
                old_rot = summary["work_g_primary_040"]["arms"][arm]["raw_rotation_checks"]["rot11_roll"]
                stat = {"mean_max_abs": float(defect.mean()), "worst_max_abs": float(defect.max()), "frac_gt_0p20": float((defect > 0.20).mean())}
                np.testing.assert_allclose(list(stat.values()), list(old_rot.values()), atol=TOL, rtol=0)
                row["host"][arm]["rot11_defect"] = stat
                row["direct_rot11_worst_max_abs"] = float(np.abs(full_s17[:, 2] - np.roll(full_s17[:, 0], 1, axis=-1)).max())
        result[cell] = row
    return result


def replay_q2(data, summary, diffs):
    raw = data["q2__raw_ll"]
    labels = data["q2__labels"]
    cal = probabilities(raw, data["temps_q2"][..., None, None, None])
    uncal = probabilities(raw)
    reference = data["q2__work_a_ref_probs"]
    np.testing.assert_array_equal(cal[:, :, 1], cal[:, :, 2])
    np.testing.assert_allclose(reference[0], reference[3], atol=TOL, rtol=0)
    np.testing.assert_allclose(reference[1], reference[2], atol=TOL, rtol=0)
    np.testing.assert_allclose(reference[1], reference[4], atol=TOL, rtol=0)
    expected = summary["work_a_q2_c4"]
    result = {}
    for qi, condition in enumerate(Q2_ARMS):
        ref, _ = score(reference[Q2_REF_INDEX[qi]], labels)
        ref_name = "duplicate_copy_dedup" if condition == "duplicate_copy_raw" else condition
        check_metrics(ref, expected["work_a_physical_reference"][ref_name], diffs)
        row = {"work_a": ref, "work_a_reference_arm": ref_name, "mothers": len(labels), "host": {}}
        for ai, arm in enumerate(ARMS):
            locked, locked_loss = score(cal[ai, :, qi], labels)
            unit, unit_loss = score(uncal[ai, :, qi], labels)
            tv = 0.5 * np.abs(cal[ai, :, qi] - cal[ai, :, 1]).sum(axis=-1)
            check_metrics(locked, expected["arms"][arm]["by_arm"][condition], diffs)
            row["host"][arm] = {
                "unit_temperature": unit, "locked_temperature": locked,
                "locked_minus_unit_nll": float((locked_loss - unit_loss).mean()),
                "locked_minus_work_a_nll": locked["nll"] - ref["nll"],
                "mean_tv_vs_valid": float(tv.mean()), "max_tv_vs_valid": float(tv.max()),
            }
        result[condition] = row
    for condition in Q2_ARMS:
        if condition == "valid":
            continue
        baseline = "none" if condition in ("none", "independent_only") else "valid"
        name = "valid_minus_none" if condition == "none" else f"{condition}_minus_{baseline}"
        first = "valid" if condition == "none" else condition
        for arm in ARMS:
            a = result[first]["host"][arm]["locked_temperature"]
            b = result[baseline]["host"][arm]["locked_temperature"]
            old = expected["arms"][arm]["paired_contrasts"][name]
            check_metrics({"acc": a["acc"] - b["acc"], "nll": a["nll"] - b["nll"]},
                          {"acc": old["delta_acc"], "nll": old["delta_nll"]}, diffs)
    return result


def cpu_geometry():
    # Exact harmonic target: class score Re(exp(i*(state+omega*t-class))).
    def design(omega_grid):
        state, omega = np.meshgrid(np.arange(4) * np.pi / 2, omega_grid, indexing="ij")
        phase = 4 * omega.ravel()
        x = np.column_stack((np.cos(state.ravel()), np.sin(state.ravel())))
        c = np.column_stack((np.cos(phase), np.sin(phase)))
        additive = np.column_stack((np.ones(len(x)), x, c))
        coupling = np.einsum("ni,nj->nij", x, c).reshape(-1, 4)
        target = np.cos(state.ravel()[:, None] + phase[:, None] - np.arange(4) * np.pi / 2)
        return additive, np.column_stack((additive, coupling)), target

    started = perf_counter()
    add, joint, target = design(np.linspace(-0.25, 0.25, 65))
    add_ood, joint_ood, target_ood = design(np.array([-0.40, 0.40]))
    result = {}
    target_delta = target_ood[[0, 1]] - target_ood[[2, 3]]
    assert np.linalg.norm(target_delta[0] - target_delta[1]) > 1.0
    for name, train, test in (("additive", add, add_ood), ("coupled", joint, joint_ood)):
        weights, _, rank, singular = np.linalg.lstsq(train, target, rcond=None)
        result[name] = {
            "design_rank": int(rank), "design_condition": float(singular[0] / singular[-1]),
            "train_rmse": float(np.sqrt(np.mean((train @ weights - target)**2))),
            "ood_rmse": float(np.sqrt(np.mean((test @ weights - target_ood)**2))),
        }
    assert result["additive"]["ood_rmse"] > 0.5 and result["coupled"]["ood_rmse"] < TOL
    result["ood_rectangular_contrast_l2"] = float(np.linalg.norm(target_delta[0] - target_delta[1]))
    result["elapsed_seconds"] = perf_counter() - started
    assert result["elapsed_seconds"] < 30
    return result


def sequential_design():
    alpha, info, prior_var, futility_pp = 0.05, 0.18, 4.0, 0.10
    fixed_boundary = norm.isf(alpha)
    rho = math.sqrt(info)
    increment_sd = math.sqrt(1 - info)
    alpha1 = norm.sf(fixed_boundary / rho)
    early = norm.isf(alpha1)

    def final_reject(boundary, drift, lo):
        # Integrate over Z1; Z2 | Z1=x and drift is Gaussian.
        integrand = lambda x: norm.pdf(x - rho * drift) * norm.sf(
            (boundary - rho * x - (1 - info) * drift) / increment_sd
        )
        return quad(integrand, lo, early, epsabs=1e-12)[0]

    # Lower integration limit -infinity gives the nonbinding efficacy design.
    final = brentq(lambda b: alpha1 + final_reject(b, 0.0, -np.inf) - alpha, 1.5, 2.0, xtol=1e-14)
    post_var = 1 / (1 / prior_var + info)
    z2_mean_per_z1 = rho * (1 + (1 - info) * post_var)
    predictive_sd = math.sqrt(1 - info + (1 - info)**2 * post_var)
    futility_z = (final - norm.isf(futility_pp) * predictive_sd) / z2_mean_per_z1
    assert futility_z < early
    binding_boundary_size = alpha1 + final_reject(final, 0.0, futility_z)
    nonbinding_boundary_size = alpha1 + final_reject(final, 0.0, -np.inf)
    assert abs(nonbinding_boundary_size - alpha) < TOL and binding_boundary_size <= alpha

    # Equal pairing, 10,000 candidates total. Boundary null verifies size separately from the mixture.
    strata = (
        ("structured_broken", 6000, -0.80, 0.10),
        ("boundary_null", 1500, 0.00, 0.90),
        ("working_coupled", 2500, 3.60, 0.99),
    )
    seed, cost = 20261126, 17.12
    pilot_fraction = 0.75 * info / 3 + 0.04
    rng = np.random.default_rng(seed)
    details = {}
    exact_cost = 0.0
    observed_cost = 0.0
    for name, n, drift, cpu_pass_rate in strata:
        z1 = rng.normal(rho * drift, 1, n)
        z2 = rho * z1 + increment_sd * rng.normal(increment_sd * drift, 1, n)
        passed = rng.random(n) < cpu_pass_rate
        early_go = passed & (z1 >= early)
        continued = passed & (z1 >= futility_z) & (z1 < early)
        efficacy = early_go | (continued & (z2 >= final))
        # Early efficacy still buys rollout to three seeds and blind scoring for release.
        funded_full = continued | early_go
        fixed = z2 >= fixed_boundary
        dollars = cost * (pilot_fraction * passed + (1 - pilot_fraction) * funded_full)
        exact_early = norm.sf(early - rho * drift)
        exact_seq = cpu_pass_rate * (exact_early + final_reject(final, drift, futility_z))
        exact_mean = cost * cpu_pass_rate * (pilot_fraction + (1 - pilot_fraction) * norm.sf(futility_z - rho * drift))
        exact_cost += n * exact_mean
        observed_cost += float(dollars.sum())
        ci = binomtest(int(efficacy.sum()), n).proportion_ci(method="exact")
        details[name] = {
            "trials": n, "drift": drift, "cpu_pass_probability_assumed": cpu_pass_rate,
            "cpu_pass_count": int(passed.sum()), "early_efficacy_count": int(early_go.sum()),
            "futility_count": int((passed & ~funded_full).sum()), "full_funded_count": int(funded_full.sum()),
            "seq_efficacy_count": int(efficacy.sum()), "seq_rate_mc": float(efficacy.mean()),
            "seq_rate_exact_binomial_ci95_mc": [ci.low, ci.high], "seq_rate_quadrature": exact_seq,
            "fixed_efficacy_count": int(fixed.sum()), "fixed_rate_mc": float(fixed.mean()),
            "fixed_rate_exact_normal": float(norm.sf(fixed_boundary - drift)),
            "seq_mean_usd_mc": float(dollars.mean()), "seq_mean_usd_exact": exact_mean,
        }
        assert ci.low <= exact_seq <= ci.high

    working = details["working_coupled"]
    assert working["seq_rate_quadrature"] >= 0.90 and working["seq_rate_exact_binomial_ci95_mc"][0] >= 0.90
    assert details["boundary_null"]["seq_rate_exact_binomial_ci95_mc"][1] <= alpha
    n_total = sum(row["trials"] for row in details.values())
    assert n_total == 10_000
    nulls = [details[x] for x in ("structured_broken", "boundary_null")]
    null_n = sum(row["trials"] for row in nulls)
    drift_min = brentq(lambda d: 0.99 * (norm.sf(early - rho * d) + final_reject(final, d, futility_z)) - 0.90, 0, 8)
    no_screen_cost = sum(
        n * cost * (pilot_fraction + (1 - pilot_fraction) * norm.sf(futility_z - rho * drift))
        for _, n, drift, _ in strata
    ) / n_total
    return {
        "rng_seed": seed, "trials": n_total, "alpha": alpha, "information_fraction": info,
        "prior_variance_standardized": prior_var, "predictive_futility_threshold": futility_pp,
        "alpha_first_look": float(alpha1), "early_z": float(early), "final_z": final, "futility_z": float(futility_z),
        "least_favorable_type_I_no_cpu_nonbinding": float(nonbinding_boundary_size),
        "least_favorable_type_I_no_cpu_binding": float(binding_boundary_size),
        "min_working_drift_for_90_percent_power_at_cpu_sensitivity_099": drift_min,
        "pilot_cost_fraction": pilot_fraction, "pilot_usd_assumed": pilot_fraction * cost,
        "fixed_usd": cost, "sequential_mean_usd_mc": observed_cost / n_total,
        "sequential_mean_usd_exact": exact_cost / n_total,
        "cost_reduction_pct_mc": 100 * (1 - observed_cost / (n_total * cost)),
        "cost_reduction_pct_exact": 100 * (1 - exact_cost / (n_total * cost)),
        "sensitivity": {
            "working_power_at_cpu_sensitivity_095": working["seq_rate_quadrature"] * 0.95 / 0.99,
            "mean_sequential_usd_no_cpu_screen_in_same_mixture": float(no_screen_cost),
            "cost_reduction_pct_no_cpu_screen_in_same_mixture": float(100 * (1 - no_screen_cost / cost)),
            "cost_reduction_pct_if_all_working": float(100 * (1 - working["seq_mean_usd_exact"] / cost)),
        },
        "null_mixture_false_efficacy_seq_mc": sum(row["seq_efficacy_count"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_fixed_mc": sum(row["fixed_efficacy_count"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_seq_exact": sum(row["trials"] * row["seq_rate_quadrature"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_fixed_exact": sum(row["trials"] * row["fixed_rate_exact_normal"] for row in nulls) / null_n,
        "strata": details,
    }


def budget():
    d = Decimal
    cpu_rate, ram_rate, gpu_rate = map(d, ("0.0000131", "0.00000222", "0.000694"))
    arm_count = len(R2["arm_ids"])
    pilot_steps = R2["pilot_steps_per_arm"] * arm_count
    remaining_steps = (R2["steps_per_arm"] - R2["pilot_steps_per_arm"]) * arm_count
    step_time = d(str(R2["max_train_seconds_per_step"]))
    row_time = d(str(R2["max_eval_seconds_per_logical_row"]))
    mothers = len(R2["synthetic_cells"]) * R2["synthetic_blocks_per_cell"] * R2["synthetic_mothers_per_block"]
    synthetic_rows = mothers * 5 * arm_count
    pluto, cttc = R2["pluto"], R2["cttc"]
    pluto_rows = (4 * pluto["packets"] + 2 * pluto["q3_pairs"]) * arm_count
    cttc_rows = (4 * cttc["packets"] + 2 * cttc["q3_pairs"]) * arm_count
    stress_rows = (7 * 140 + 1080) * arm_count
    intervention_rows = 2 * (4 * mothers + pluto["packets"] + cttc["packets"])
    pilot_dev_rows = sum(R2["pilot_dev_rows_per_arm"].values()) * arm_count
    post_dev_rows = sum(R2["calibration_dev_rows_per_arm"].values()) * arm_count
    gpu = "A100-80GB"
    # ROM ceilings per work unit. Separate start/stop reserves cover billed idle and metering.
    rows = (
        ("cpu_ingest_weight_staging", "CPU", 1800, 1, 8, None, None, "cpu"),
        ("cpu_causal_frontend_dual_cache_pool", "CPU", 7200, 4, 16, None, None, "cpu"),
        ("cpu_audit_aggregate", "CPU", 1800, 2, 8, None, None, "cpu"),
        ("a100_initial_smoke_setup", gpu, 350, 2, 16, None, None, "pilot"),
        ("pilot_train_four_matched_arms", gpu, math.ceil(pilot_steps * step_time), 2, 16, pilot_steps, None, "pilot"),
        ("pilot_disjoint_dev", gpu, math.ceil(pilot_dev_rows * row_time), 2, 16, None, pilot_dev_rows, "pilot"),
        ("resume_same_four_arms", gpu, math.ceil(remaining_steps * step_time), 2, 16, remaining_steps, None, "arm_comparison"),
        ("post_fit_disjoint_calibration_dev", gpu, math.ceil(post_dev_rows * row_time), 2, 16, None, post_dev_rows, "arm_comparison"),
        ("blind_synthetic_core", gpu, math.ceil(synthetic_rows * row_time), 2, 16, None, synthetic_rows, "arm_comparison"),
        ("blind_pluto_raw", gpu, math.ceil(pluto_rows * row_time), 2, 16, None, pluto_rows, "pluto"),
        ("blind_cttc_raw", gpu, math.ceil(cttc_rows * row_time), 2, 16, None, cttc_rows, "cttc"),
        ("blind_real_stress", gpu, math.ceil(stress_rows * row_time), 2, 16, None, stress_rows, "controls"),
        ("blind_matched_slot_text_controls", gpu, math.ceil(intervention_rows * row_time), 2, 16, None, intervention_rows, "controls"),
        ("a100_remaining_setup_teardown", gpu, 750, 2, 16, None, None, "arm_comparison"),
        ("explicit_retry_reserve", gpu, 900, 2, 16, None, None, "retry"),
    )
    breakdown = []
    totals = {}
    for task, device, seconds, cores, gib, steps, eval_rows, group in rows:
        rate = (gpu_rate if device == gpu else d(0)) + cpu_rate * cores + ram_rate * gib
        cost = rate * seconds
        totals[group] = totals.get(group, d(0)) + cost
        breakdown.append({"task": task, "group": group, "device": device, "max_billed_seconds": seconds,
                          "max_physical_cores": cores, "max_gib": gib, "train_steps": steps,
                          "logical_eval_rows": eval_rows, "max_usd_per_second": str(rate), "max_usd": str(cost)})
    total, credit = d("9.50"), d("30.00") - d("17.12")
    planned = sum(totals.values()) - totals["retry"]
    reserved = sum(totals.values())
    unassigned = total - reserved
    a100_seconds = sum(row["max_billed_seconds"] for row in breakdown if row["device"] == gpu)
    expected_m4 = (d("0.00074684") * 10274 + totals["cpu"])
    archive = json.loads((DATA / "bench_steady.json").read_text(encoding="utf-8"))
    train = archive["train_bench"]["Q3_after_pilot_kv_mb16"]
    evaluation = archive["dev_eval_bench"]["Q3_after_pilot_eval_b64"]
    assert train["peak_vram_mb"] == 52972.5 and evaluation["peak_vram_mb"] == 45409.9
    assert expected_m4 == d("8.44073416")
    assert a100_seconds == 10987 and reserved == d("9.07079564") and unassigned == d("0.42920436")
    assert (planned, totals["cpu"], totals["pilot"]) == (d("8.39064764"), d("0.7677"), d("0.87663520"))
    assert (pilot_steps, remaining_steps) == (276, 1260)
    assert (synthetic_rows, pluto_rows, cttc_rows, stress_rows, intervention_rows) == (40960, 53760, 28800, 8240, 23744)
    assert sum(row["logical_eval_rows"] or 0 for row in breakdown) == 171504
    assert all(row["device"] in ("CPU", gpu) for row in breakdown)
    assert reserved + unassigned == total < credit and credit - total == d("3.38")
    return {
        "pricing_url": "https://modal.com/pricing", "only_gpu": gpu, "rows": breakdown,
        "stage3d_reference_bench": {"train_peak_allocated_mib_mb16": train["peak_vram_mb"],
                                    "eval_peak_allocated_mib_b64": evaluation["peak_vram_mb"],
                                    "train_seconds_per_step": train["steady_32ex_step_ms"] / 1000,
                                    "eval_seconds_per_row": evaluation["ms_per_ex"] / 1000},
        "group_usd": {key: str(value) for key, value in totals.items()},
        "m4_same_scope_at_12_gib_usd": str(expected_m4),
        "a100_16_gib_rate_delta_vs_m4_per_second": str(4 * ram_rate),
        "a100_seconds_including_retry": a100_seconds, "scheduled_usd_without_retry": str(planned),
        "scheduled_usd_with_retry": str(reserved), "unassigned_inside_cap_usd": str(unassigned),
        "total_reserve_inside_cap_usd": str(totals["retry"] + unassigned),
        "cpu_plus_pilot_usd_before_go": str(totals["cpu"] + totals["pilot"]),
        "cap_usd": str(total), "historical_credit_remainder_usd": str(credit),
        "buffer_usd_at_cap": str(credit - total), "required_workspace_net_spend_guard_usd": "0.00",
        "live_workspace_or_gpu_execution_verified": False, "spend_this_round_usd": "0.00",
    }


def disjoint_pilot_design(ledger):
    # Independent dev proxy and blind statistic conditional on effect, with zero early efficacy alpha.
    alpha = R2["domain_family_alpha"] / 3
    critical = norm.isf(alpha)
    slope, prior_var = 0.60, 4.0
    posterior_var = 1 / (1 / prior_var + slope**2)
    posterior_sd_blind = math.sqrt(1 + posterior_var)
    minimum_pp = R2["predictive_go_probability_min"]
    cutoff = (critical - norm.isf(minimum_pp) * posterior_sd_blind) / (posterior_var * slope)
    rng = np.random.default_rng(20261127)
    fixed_cost = float(ledger["scheduled_usd_without_retry"])
    cpu_cost = float(ledger["group_usd"]["cpu"])
    gpu_pilot_cost = float(ledger["group_usd"]["pilot"])
    full_after_pilot = fixed_cost - cpu_cost - gpu_pilot_cost
    strata = {}
    weighted_cost = weighted_cost_exact = 0.0
    for name, n, drift, pass_prob in (("structural_null", 6000, -0.8, 0.10),
                                      ("boundary_null", 1500, 0.0, 0.90),
                                      ("working", 2500, 3.90, 0.99)):
        screen = rng.random(n) < pass_prob
        pilot = rng.normal(slope * drift, 1, n)
        blind = rng.normal(drift, 1, n)
        go = screen & (pilot >= cutoff)
        fixed = blind >= critical
        reject = go & fixed
        exact_go = pass_prob * norm.sf(cutoff - slope * drift)
        exact_fixed = norm.sf(critical - drift)
        exact_reject = exact_go * exact_fixed
        cost = cpu_cost + gpu_pilot_cost * screen.mean() + full_after_pilot * go.mean()
        exact_cost = cpu_cost + gpu_pilot_cost * pass_prob + full_after_pilot * exact_go
        ci = binomtest(int(reject.sum()), n).proportion_ci(method="exact")
        assert ci.low <= exact_reject <= ci.high
        strata[name] = {
            "trials": n, "drift": drift, "cpu_pass_assumed": pass_prob, "go_count": int(go.sum()),
            "fixed_reject_count": int(fixed.sum()), "sequential_reject_count": int(reject.sum()),
            "go_rate_analytic": float(exact_go), "fixed_rate_analytic": float(exact_fixed),
            "sequential_rate_analytic": float(exact_reject), "sequential_rate_mc": float(reject.mean()),
            "sequential_mc_ci95": [ci.low, ci.high], "cost_usd_mc": float(cost), "cost_usd_analytic": float(exact_cost),
        }
        weighted_cost += n * cost
        weighted_cost_exact += n * exact_cost
    trials = sum(row["trials"] for row in strata.values())
    assert trials == 10000 and strata["working"]["sequential_mc_ci95"][0] >= 0.90
    assert strata["boundary_null"]["sequential_rate_analytic"] <= alpha
    return {
        "scope": "One modeled primary endpoint at first Holm threshold. Full-domain joint power needs independent dev qualification.",
        "trials": trials, "first_holm_alpha": alpha, "early_efficacy_alpha": 0.0, "blind_critical_z": float(critical),
        "dev_slope_assumed": slope, "effect_prior_variance": prior_var, "dev_z_go_cutoff_model_only": float(cutoff),
        "first_holm_null_size_bound": alpha, "familywise_alpha_bound": R2["domain_family_alpha"],
        "fixed_same_scope_usd": fixed_cost, "cpu_charge_even_on_screen_failure_usd": cpu_cost,
        "gpu_pilot_usd": gpu_pilot_cost, "mean_sequential_cost_usd_mc": weighted_cost / trials,
        "mean_sequential_cost_usd_analytic": weighted_cost_exact / trials,
        "cost_saving_pct_mc": 100 * (1 - weighted_cost / trials / fixed_cost),
        "cost_saving_pct_analytic": 100 * (1 - weighted_cost_exact / trials / fixed_cost),
        "strata": strata,
    }


def round2_contract_checks():
    """Verify small algebraic fixtures. Real data and host qualification are separate."""
    # All 2^16 packets, including every legacy prefix and all boundary field IDs.
    words = np.arange(65536, dtype=np.uint32)
    turn, inner = (words >> 14) & 3, (words >> 12) & 3
    angle, radial, quality = (words >> 6) & 63, (words >> 2) & 15, words & 3
    rebuilt = (turn << 14) | (inner << 12) | (angle << 6) | (radial << 2) | quality
    np.testing.assert_array_equal(rebuilt, words)
    np.testing.assert_array_equal((turn << 2) | inner, words >> 12)

    def feature_bytes(raw, reads, cutoff):
        # One baseline read and two potential reveal reads. Assert no target overlap.
        authorized = {1, 2, 4, 5}
        if set(reads) - authorized or any(index > cutoff for index in reads):
            raise ValueError("raw read violates the fixture filtration")
        unique = sorted(set(reads))
        return hashlib.sha256(raw[unique].astype("<i8").tobytes()).hexdigest()

    raw = np.arange(12, dtype=np.int64)
    sentinel_checks = 0
    for reads, cutoff in (([1, 2], 3), ([1, 2, 4], 4), ([1, 2, 4, 5], 5)):
        changed = raw.copy()
        changed[list(set(range(len(raw))) - set(reads))] += 10_000
        assert feature_bytes(raw, reads, cutoff) == feature_bytes(changed, reads, cutoff)
        assert feature_bytes(raw, reads + [reads[0]], cutoff) == feature_bytes(raw, reads, cutoff)
        changed[reads[-1]] -= 999
        assert feature_bytes(raw, reads, cutoff) != feature_bytes(changed, reads, cutoff)
        sentinel_checks += 1
    for reads, cutoff in (([1, 2, 4], 3), ([1, 2, 6], 7)):
        try:
            feature_bytes(raw, reads, cutoff)
        except ValueError:
            sentinel_checks += 1
        else:
            raise AssertionError("fixture admitted an unauthorized read")

    # Correlated fresh reference: p(old,new|u)=p(old|u)*p(new|old,u).
    y = np.array([1.15 + .37j, -.41 + .69j])
    means = (1j ** np.arange(4))[:, None] * np.array([1, np.exp(.45j)])[None, :]
    cov = np.array([[1.2, .35], [.35, .8]])
    err = y - means
    joint_ll = -np.einsum("hi,ij,hj->h", err.conj(), np.linalg.inv(cov), err).real - np.log(np.linalg.det(cov))
    innovation = err[:, 1] - cov[1, 0] / cov[0, 0] * err[:, 0]
    variance = cov[1, 1] - cov[1, 0] ** 2 / cov[0, 0]
    sequential_ll = -(abs(err[:, 0])**2 / cov[0, 0] + abs(innovation)**2 / variance) - np.log(cov[0, 0] * variance)
    naive_ll = -(abs(err)**2 / cov.diagonal()).sum(axis=-1)
    np.testing.assert_allclose(probabilities(sequential_ll), probabilities(joint_ll), atol=TOL, rtol=0)
    double_counted_ll = joint_ll - abs(err[:, 0])**2 / cov[0, 0]
    double_count_defect = float(np.max(abs(probabilities(double_counted_ll) - probabilities(joint_ll))))
    assert double_count_defect > .01
    naive_defect = float(np.max(abs(probabilities(naive_ll) - probabilities(joint_ll))))
    assert naive_defect > .01

    # Check task-specific permutations and stabilizer-compatible exact index transport.
    reference = np.array([.41, .32, .18, .09])
    shifts = {"Q1": lambda r, s, p: s-r, "Q2": lambda r, s, p: r-p,
              "Q3_repeat": lambda r, s, p: r, "Q3_indep": lambda r, s, p: 0,
              "Q3_after_pilot": lambda r, s, p: r}
    group = np.indices((4, 4, 4)).reshape(3, -1).T
    for name, chi in shifts.items():
        for r in range(4):
            assert chi(r, r, r) == (r if name in ("Q3_repeat", "Q3_after_pilot") else 0)
        for g in group:
            for h in group:
                np.testing.assert_array_equal(np.roll(np.roll(reference, chi(*g)), chi(*h)),
                                              np.roll(reference, chi(*((g+h) % 4))))

    rng = np.random.default_rng(6)
    x = rng.normal(size=(8, 4))
    p0 = probabilities(rng.normal(size=x.shape))
    c = x - x.mean(axis=-1, keepdims=True)
    residual = R2["arm2_gamma"] * np.tanh(c)
    residual -= residual.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(probabilities(np.log(p0) + 0 * residual), p0, atol=TOL, rtol=0)
    assert np.max(np.ptp(residual, axis=-1)) <= 2 * R2["arm2_gamma"]

    center = np.eye(4) - np.full((4, 4), .25)
    host_widths = {"Qwen3.5-4B": 2560, "Qwen3.5-9B": 4096, "Qwen3.8-27B": 5120}
    host_results = {}
    # A common feature matrix before any host-specific projection.
    u = rng.normal(size=(6, 16))
    shared_digest = hashlib.sha256(u.astype("<f8").tobytes()).hexdigest()
    for name, width in host_widths.items():
        frame, _ = np.linalg.qr(rng.normal(size=(width, 16)))
        j = center @ frame[:, :4].T
        a1 = j @ frame
        z = np.zeros(4)
        a2 = R2["arm2_gamma"] * center @ np.diag(1 - np.tanh(z)**2) @ a1
        scores = {}
        for arm, jac in (("1", a1), ("2", a2)):
            singular = np.linalg.svd(jac, compute_uv=False)
            rank = int(np.count_nonzero(singular > 1e-6 * singular[0]))
            ratio = float(singular[rank - 1] / singular[0])
            assert rank == 3 and ratio >= R2["geometry_min_singular_ratio"]
            scores[arm] = {"centered_rank": rank, "sigma_min_over_max": ratio}
        # Transport on an admissible rank-three frame. The host-specific outputs
        # differ in width, while this constructed readout sees the same evidence.
        np.testing.assert_allclose((u @ frame.T) @ j.T, u[:, :4] @ center, atol=TOL, rtol=0)
        params = (R2["stage6"]["adapter_rank_max"] + 1) * width + 1
        assert params <= R2["stage6"]["trainable_params_max"]
        host_results[name] = {"width": width, "max_shared_adapter_parameters": params, "toy_candidate_jacobians": scores}
    assert hashlib.sha256(u.astype("<f8").tobytes()).hexdigest() == shared_digest

    # A pre-score common-support join uses only eligibility, not answers or loss.
    case_ids = {"ours": {"a", "b", "c"}, "LEMMO": {"b", "c", "d"}}
    common = sorted(case_ids["ours"] & case_ids["LEMMO"])
    assert common == ["b", "c"]
    return {
        "scope": "Algebraic fixtures on CPU. Production causality, release scoring, and actual host Jacobians are pending.",
        "packed_words_replayed": int(words.size), "sentinel_fixture_checks_passed": sentinel_checks,
        "correlated_innovation_posterior_max_abs_error": float(np.max(abs(probabilities(sequential_ll) - probabilities(joint_ll)))),
        "naive_independent_posterior_max_abs_error": naive_defect,
        "double_counted_old_reference_posterior_max_abs_error": double_count_defect,
        "task_pairwise_group_compositions_passed": len(shifts) * len(group)**2,
        "arm2_max_pairwise_residual_logodds": 2 * R2["arm2_gamma"],
        "host_configuration_shape_checks": host_results, "fixture_common_support_ids": common,
    }


def execution_graph():
    dependencies = {
        "4A": [], "4B": ["4A"], "4C": ["4B"], "4D": ["4C"], "4E": ["4D"], "4F": ["4D"],
        "5A": ["4E", "4F"], "5B": ["5A"], "6A": ["5B"], "6B": ["6A"], "6C": ["6B"],
        "6D": ["6A"], "6E": ["6C", "6D"],
    }
    completed = set()
    for node, parents in dependencies.items():
        assert set(parents) <= completed, node
        completed.add(node)
    return {"topological_order": list(dependencies), "parents": dependencies,
            "live_status": dict.fromkeys(dependencies, "awaiting_production_receipt"), "numeric_lock": R2}


def main():
    assert (np.__version__, scipy.__version__) == ("2.3.5", "1.17.0")
    assert platform.python_version_tuple()[:2] == ("3", "12")
    summary = json.loads((DATA / "summary_master.json").read_text(encoding="utf-8"))
    ledger = budget()
    parity_diffs = []
    with np.load(DATA / "blind_q3_scores.npz", allow_pickle=False) as q3, np.load(DATA / "blind_q2_scores.npz", allow_pickle=False) as q2:
        assert q3["seeds"].tolist() == summary["seeds"]
        np.testing.assert_array_equal(q3["temps_q3"], summary["locked_temps_q3"])
        np.testing.assert_array_equal(q2["temps_q2"], summary["locked_temps_q2"])
        report = {
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "stage3d": {"q3": replay_q3(q3, summary, parity_diffs), "q2": replay_q2(q2, summary, parity_diffs)},
            "summary_parity": {"max_abs_acc_nll_diff": max(parity_diffs), "scalar_acc_nll_checks": len(parity_diffs),
                               "h_c4_035_scope": "Work F identity rotation and both signs. The all-rotation ladder is a fresh replay."},
            "cpu_geometry_toy": cpu_geometry(), "round1_sequential_model": sequential_design(),
            "round2_disjoint_pilot_model": disjoint_pilot_design(ledger), "modal_budget": ledger,
            "round2_cpu_contract_checks": round2_contract_checks(), "execution_graph": execution_graph(),
        }
    path = HERE / "results_m6.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"VERIFIED {path.name}: parity max |diff|={max(parity_diffs):.3g}, two 10,000-trial designs, A100 cap ${ledger['cap_usd']}")


if __name__ == "__main__":
    main()
