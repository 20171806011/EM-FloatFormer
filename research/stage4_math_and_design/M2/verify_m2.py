"""Read-only Stage 3D replay and Stage 4 power analysis. Run with the pinned uv command."""

import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import logsumexp, roots_legendre, xlogy
from scipy.stats import chi2, nct, norm, t

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "stage3d_qwen_signed_phase"
sys.path.insert(0, str(SOURCE))
import recompute_all as stage3


ARMS = ("phase", "zero")
METRICS = ("acc", "nll", "brier", "ece")
LOG4 = np.log(4.0)
GROUP = tuple((a, b) for a in range(4) for b in range(4))
CODES = np.arange(256)
SEED = 2026412


def close(actual, expected, label, atol=2e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=label)
    return float(np.max(np.abs(np.asarray(actual) - expected), initial=0))


def check_tree(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        return max((check_tree(actual[k], v, f"{label}.{k}") for k, v in expected.items()), default=0)
    if isinstance(expected, list):
        assert len(actual) == len(expected), label
        return max((check_tree(a, e, f"{label}.{i}") for i, (a, e) in enumerate(zip(actual, expected))), default=0)
    if isinstance(expected, float):
        return close(actual, expected, label)
    assert actual == expected, label
    return 0.0


def divide_counts(counts):
    mass = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, mass, out=np.full_like(counts, 0.25, dtype=float), where=mass > 0)


def histogram(index, labels, size):
    return np.bincount(4 * index.ravel().astype(int) + labels.ravel().astype(int),
                       minlength=4 * size).reshape(size, 4)


def entropy(prob):
    return -xlogy(prob, prob).sum(axis=-1)


def kl(p, q):
    return (xlogy(p, p) - xlogy(p, q)).sum(axis=-1)


def metrics(probs, labels):
    probs = probs.reshape(-1, 4)
    labels = labels.ravel()
    sc = stage3.score_vectors(probs, np.tile(labels, len(probs) // len(labels)))
    assert sc["clipped_count"] == 0
    return {k: sc[k] for k in METRICS}, sc


def cross_entropy(counts, prob):
    return float(-np.sum(xlogy(counts, prob)) / counts.sum())


def calibration_fit(raw, counts, dev_beta):
    """Convex log-loss optimizer in beta=1/T, including the uniform boundary."""
    z = raw - raw.max(axis=-1, keepdims=True)
    total = counts.sum()
    mass = counts.sum(axis=-1, keepdims=True)
    truth = np.sum(counts * z) / total

    def derivative(beta):
        return float(np.sum(mass * stage3.softmax_temperature(z, 1 / beta) * z) / total - truth) if beta else float(np.sum(mass * z) / (4 * total) - truth)

    if derivative(0) >= 0:
        beta = 0.0
    elif truth == 0:
        raise ValueError("Perfect separation needs the T=0 limiting optimizer.")
    else:
        hi = 1.0
        while derivative(hi) <= 0:
            hi *= 2
        beta = brentq(derivative, 0, hi, xtol=1e-13)
    materialized = stage3.softmax_temperature(raw, 1 / beta) if beta else np.full_like(raw, 0.25)
    dev = stage3.softmax_temperature(raw, 1 / dev_beta)
    bregman = float(np.sum(mass[..., 0] * kl(materialized, dev)) / total)
    boundary = (dev_beta - beta) * derivative(beta)
    close(cross_entropy(counts, dev) - cross_entropy(counts, materialized), bregman + boundary,
          "scalar temperature Bregman identity")
    return materialized, {
        "T_star": 1 / beta if beta else "infinity", "beta_star": beta,
        "derivative_at_zero": derivative(0), "derivative_at_dev": derivative(dev_beta),
        "derivative_at_star": derivative(beta), "temperature_bregman_KL": bregman,
        "temperature_boundary_term": boundary,
    }


def fit_temperatures(raw, counts, temps):
    fitted, records = zip(*(calibration_fit(z, w, 1 / temp) for z, w, temp in zip(raw, counts, temps)))
    shared, common = calibration_fit(raw, counts, float(np.mean(1 / temps)))
    for field in ("derivative_at_dev", "temperature_bregman_KL", "temperature_boundary_term"):
        common.pop(field)
    return np.stack(fitted), list(records), shared, common


def joint_split(counts, pred):
    """KL of the two joint laws with the same covariate marginal."""
    mass = counts.sum(axis=-1)
    w = mass / mass.sum()
    p = divide_counts(counts)
    py = (w[..., None] * p).reshape(-1, 4).sum(axis=0)
    qy = (w[..., None] * pred).reshape(-1, 4).sum(axis=0)
    cond = float(np.sum(w * kl(p, pred)))
    prior = float(kl(py, qy))
    p_joint = (w[..., None] * p).reshape(-1, 4)
    q_joint = (w[..., None] * pred).reshape(-1, 4)
    reverse = np.sum(xlogy(p_joint, p_joint / py) - xlogy(p_joint, q_joint / qy))
    h = float(np.sum(w * entropy(p)))
    loss = cross_entropy(counts, pred)
    close([h + cond, prior + reverse], [loss, cond], "joint KL chain")
    return {"entropy": h, "conditional_kl": cond, "prior_kl": prior,
            "reverse_conditional_kl": float(reverse), "model_marginal": qy.tolist(),
            "label_marginal": py.tolist(), "nll": loss}


def forecast_split(table, counts, kind):
    """Exact empirical score resolution with a fixed argmax or the full forecast."""
    pred = table.reshape(-1, 4)
    c = counts.reshape(-1, 4)
    active = c.sum(axis=1) > 0
    pred, c = pred[active], c[active]
    if kind == "argmax_set":
        key = ((pred == pred.max(axis=1, keepdims=True)) * (1 << np.arange(4))).sum(axis=1)
    else:
        key = pred
    _, inv = np.unique(key, axis=0, return_inverse=True)
    agg = np.zeros((inv.max() + 1, 4), dtype=np.int64)
    np.add.at(agg, inv.ravel(), c)
    h = float(np.average(entropy(divide_counts(agg)), weights=agg.sum(axis=-1)))
    hy = float(entropy(c.sum(axis=0) / c.sum()))
    return {"groups": len(agg), "H_Y_given_forecast_feature": h, "resolution": hy - h,
            "residual_nll_minus_entropy": cross_entropy(c, pred) - h}


def rotate_codes(codes, a, b):
    codes = codes.astype(int, copy=False)
    return 16 * ((codes // 16 + 4 * a) % 16) + (codes % 16 + 4 * b) % 16


def align(table, a, b):
    return np.roll(table[..., rotate_codes(CODES, a, b), :], -a, axis=-1)


def project(table):
    return np.stack([align(table, a, b) for a, b in GROUP]).mean(axis=0)


def group_diagnostics(probs, arith, geo, counts):
    log_group = np.stack([np.log(align(probs, a, b)) for a, b in GROUP])
    w = counts.sum(axis=-1) / counts.sum()
    randomized = float(-(counts * log_group.mean(axis=0)).sum() / counts.sum())
    loss_a = cross_entropy(counts, arith)
    loss_r = cross_entropy(counts, probs)
    j = np.log(arith) - log_group.mean(axis=0)
    y_jensen = float((counts * j).sum() / counts.sum())
    close(randomized - loss_a, y_jensen, "label-weighted Jensen gap")
    close(project(arith), arith, "arithmetic projection parity")
    workg = stage3.compute_16way_group_avg(probs)
    close(arith, workg, "Stage 3D group operator")
    z_log = logsumexp(log_group.mean(axis=0), axis=-1)
    close(randomized - cross_entropy(counts, geo), -np.sum(w * z_log), "geometric normalizer")
    return {
        "raw_nll": loss_r, "arithmetic_nll": loss_a, "geometric_nll": cross_entropy(counts, geo),
        "randomized_orbit_raw_nll": randomized, "label_weighted_arithmetic_jensen_gap": y_jensen,
        "design_imbalance_nll": loss_r - randomized,
        "reverse_barycenter_kl": float(np.sum(w[..., None] * arith * j)),
        "geometric_normalizer_gain": float(-np.sum(w * z_log)),
        "raw_equivariance_max": max(float(np.max(np.abs(align(probs, a, b) - probs))) for a, b in GROUP),
        "arith_equivariance_max": max(float(np.max(np.abs(align(arith, a, b) - arith))) for a, b in GROUP),
        "geo_equivariance_max": max(float(np.max(np.abs(align(geo, a, b) - geo))) for a, b in GROUP),
        "arith_idempotence_max": float(np.max(np.abs(project(arith) - arith))),
        "geo_idempotence_max": float(np.max(np.abs(stage3.softmax_temperature(project(np.log(geo)), 1) - geo))),
    }


def paired_breakdown(raw_hit, other_hit):
    d = other_hit - raw_hit
    return {"improved_fraction": float(np.mean(d > 0)), "worsened_fraction": float(np.mean(d < 0)),
            "weighted_gains": float(np.maximum(d, 0).mean()), "weighted_losses": float(np.maximum(-d, 0).mean()),
            "delta_acc": float(d.mean())}


def q3_index(table, codes):
    return np.stack([table[:, s, codes[s]] for s in range(2)], axis=1)


def posterior_reference(prob, labels, model):
    """Soft-reference CE and the realized-vs-soft loss discrepancy."""
    label_indices = np.broadcast_to(labels, model.shape[:-1])[..., None]
    nll = float(-np.mean(np.log(np.take_along_axis(model, label_indices, axis=-1))))
    h = float(entropy(prob).mean())
    gap = float(kl(prob, model).mean())
    ce = float(-np.mean(np.sum(xlogy(prob, model), axis=-1)))
    close(ce, h + gap, "reference expected loss")
    return {"reference_entropy": h, "expected_kl": gap, "expected_nll": ce,
            "realized_label_residual": nll - ce, "realized_nll": nll}


def q3_analysis(bundle, name, temps):
    get = lambda key: bundle[f"{name}__{key}"]
    raw, codes, y = get("raw_ll"), get("codes"), get("targets")
    count = np.stack([histogram(codes[s], y[s], 256) for s in range(2)])
    signed = float(np.average(entropy(divide_counts(count)), weights=count.sum(axis=-1)))
    unsigned = float(np.average(entropy(divide_counts(count.sum(axis=0))), weights=count.sum(axis=(0, 2))))
    counts = np.broadcast_to(count, raw.shape[1:])
    orbit_count = sum(align(count, a, b) for a, b in GROUP)
    orbit_counts = np.broadcast_to(orbit_count, counts.shape)
    oracles = {k: get(k) for k in ("direct_table_probs", "full_iq_probs")}
    empirical = divide_counts(count)
    h_y = float(entropy(count.sum(axis=(0, 1)) / count.sum()))
    cal_ref, cal_mass = get("cal_ref"), get("cal_mass")
    cal_joint = cal_ref * cal_mass[..., None] / 2
    cal_unsigned = divide_counts(cal_joint.sum(axis=0))
    cal_signed_h = float(np.sum(cal_mass * entropy(cal_ref)) / 2)
    cal_unsigned_h = float(np.sum(cal_mass.sum(axis=0) * entropy(cal_unsigned)) / 2)
    code_mass = count.sum(axis=-1)
    active = code_mass > 0
    unsigned_table = np.broadcast_to(divide_counts(count.sum(axis=0)), empirical.shape)
    close(unsigned - signed, np.average(kl(empirical[active], unsigned_table[active]), weights=code_mass[active]),
          "signed conditional MI")
    result = {"raw_shape": list(raw.shape), "codes_shape": list(codes.shape), "blocks": y.shape[-1] // 64,
              "independent_cal_reference": {"H_Y_given_signed_code": cal_signed_h,
                 "H_Y_given_unsigned_code": cal_unsigned_h, "signed_conditional_MI": cal_unsigned_h - cal_signed_h,
                 "blind_metrics": metrics(q3_index(cal_ref[None, ...], codes), y)[0]},
              "count_coverage": int(np.count_nonzero(count.sum(axis=-1))),
              "H_Y": h_y, "H_Y_given_signed_code": signed, "H_Y_given_unsigned_code": unsigned,
              "signed_conditional_MI_plugin": unsigned - signed, "code_MI_plugin": h_y - signed,
              "empirical_signed_oracle": metrics(q3_index(empirical[None, ...], codes), y)[0],
              "references": {k: metrics(v, y)[0] for k, v in oracles.items()}, "arms": {}}
    # Zero-probability classes in the in-sample empirical oracle are absent among its own labels.
    block_scores = {}
    for a, name_a in enumerate(ARMS):
        dev = np.stack([stage3.softmax_temperature(z, temp) for z, temp in zip(raw[a], temps[a])])
        opt, opt_info, shared, shared_info = fit_temperatures(raw[a], counts, temps[a])
        arith = project(dev)
        geo = stage3.softmax_temperature(project(np.log(dev)), 1)
        f0 = stage3.compute_representative_readout(dev)
        arith_opt = project(opt)
        evaluations = {k: metrics(q3_index(v, codes), y) for k, v in
                       {"dev": dev, "opt": opt, "shared_opt": shared, "arith": arith, "geo": geo,
                        "F0": f0, "arith_opt": arith_opt}.items()}
        dev_sc = evaluations["dev"][1]
        for seed, record in enumerate(opt_info):
            record["T_dev"] = float(temps[a, seed])
            record["dev"] = metrics(q3_index(dev[seed:seed + 1], codes), y)[0]
            record["opt"] = metrics(q3_index(opt[seed:seed + 1], codes), y)[0]
        dev_joint = joint_split(counts, dev)
        opt_joint = joint_split(counts, opt)
        end = joint_split(counts, arith_opt)
        pieces = {"bayes_entropy": signed,
                  "temperature_gap": dev_joint["nll"] - opt_joint["nll"],
                  "projection_gap_at_T_star": opt_joint["nll"] - end["nll"],
                  "projected_marginal_prior_kl": end["prior_kl"],
                  "projected_reverse_conditional_kl": end["reverse_conditional_kl"]}
        close(sum(pieces.values()), dev_sc["nll"], "Q3 additive decomposition")
        p_rows = q3_index(dev, codes)
        drop_sign = (count.sum(axis=-1)[..., None] * dev).sum(axis=1)
        mass = count.sum(axis=(0, 2))[..., None]
        drop_sign = np.divide(drop_sign, mass, out=np.full_like(drop_sign, .25), where=mass > 0)
        signless_nll = cross_entropy(counts, np.broadcast_to(drop_sign[:, None], dev.shape))
        sign_gap = signless_nll - dev_joint["nll"]
        arm = {"metrics": {k: v[0] for k, v in evaluations.items()}, "T_star_by_seed": opt_info,
               "single_shared_raw_temperature": shared_info, "additive_path": pieces,
               "raw_joint_KL": dev_joint, "opt_joint_KL": opt_joint,
               "projected_opt_joint_KL": end, "forecast_argmax": forecast_split(dev, counts, "argmax_set"),
               "forecast_vector": forecast_split(dev, counts, "vector"),
               "sign_blind_marginalized_nll": signless_nll, "sign_blind_minus_matched_nll": sign_gap,
               "reference_loss_split": {k: posterior_reference(v, y, p_rows) for k, v in oracles.items()},
               "group_on_saved_design": group_diagnostics(dev, arith, geo, counts),
               "group_on_complete_orbit": group_diagnostics(dev, arith, geo, orbit_counts),
               "group_opt_on_saved_design": group_diagnostics(opt, arith_opt,
                   stage3.softmax_temperature(project(np.log(opt)), 1), counts)}
        close(arm["group_on_complete_orbit"]["design_imbalance_nll"], 0, "complete orbit invariance")
        arm["NIG_realized"] = {k: (h_y - dev_sc["nll"]) / (h_y - v["nll"])
                               for k, v in result["references"].items()}
        arm["NIG_realized"]["empirical_signed_code"] = (h_y - dev_sc["nll"]) / (h_y - signed)
        arm["group_accuracy_transitions"] = {k: paired_breakdown(dev_sc["per_item_hit"], evaluations[k][1]["per_item_hit"])
                                             for k in ("arith", "geo", "F0")}
        orbit_y = np.stack([(y + a0) % 4 for a0, _ in GROUP])
        orbit_codes = np.stack([rotate_codes(codes, a0, b0) for a0, b0 in GROUP])
        orbit_hit = {k: metrics(q3_index(v, orbit_codes.transpose(1, 0, 2, 3)), orbit_y.transpose(1, 0, 2, 3))[1]["per_item_hit"]
                     for k, v in {"raw": dev, "arith": arith, "geo": geo}.items()}
        arm["complete_orbit_accuracy"] = {k: float(v.mean()) for k, v in orbit_hit.items()}
        arm["complete_orbit_transitions"] = {k: paired_breakdown(orbit_hit["raw"], orbit_hit[k]) for k in ("arith", "geo")}
        block_scores[name_a] = {k: dev_sc[f"per_item_{src}"].reshape(len(temps[a]), 2, y.shape[1], -1, 64)
                                .mean(axis=(1, 2, 4)).T for k, src in (("accuracy", "hit"), ("nll", "nll"))}
        result["arms"][name_a] = arm
    phase, zero = (result["arms"][a] for a in ARMS)
    result["delta_nll_zero_minus_phase"] = {
        k: zero["additive_path"][k] - val for k, val in phase["additive_path"].items()}
    delta = zero["metrics"]["dev"]["nll"] - phase["metrics"]["dev"]["nll"]
    close(sum(result["delta_nll_zero_minus_phase"].values()), delta, "Q3 paired delta")
    result["delta_nll_zero_minus_phase"]["total"] = delta
    result["delta_nll_zero_minus_phase"]["at_seedwise_T_star"] = zero["metrics"]["opt"]["nll"] - phase["metrics"]["opt"]["nll"]
    return result, {"accuracy": block_scores["phase"]["accuracy"] - block_scores["zero"]["accuracy"],
                    "nll": block_scores["zero"]["nll"] - block_scores["phase"]["nll"]}


def q2_analysis(bundle, temps):
    raw = bundle["q2__raw_ll"]
    y = bundle["q2__labels"]
    codes = 16 * bundle["q2__z1"].astype(int) + bundle["q2__z2"]
    refs = bundle["q2__work_a_ref_probs"]
    code_count = histogram(codes, y, 256)
    h_y = float(entropy(np.bincount(y, minlength=4) / len(y)))
    h_z = float(np.average(entropy(divide_counts(code_count)), weights=code_count.sum(axis=-1)))
    row_counts = np.broadcast_to(np.eye(4, dtype=int)[y], raw.shape[1:2] + (len(y), 4))
    result = {"raw_shape": list(raw.shape), "H_Y": h_y, "H_Y_given_base_codes": h_z,
              "base_code_coverage": int(np.count_nonzero(code_count.sum(axis=1))),
              "signed_conditional_MI": None, "group_averaging": None, "controls": {}}
    for qi, control in enumerate(stage3.Q2_ARMS):
        # The analytical reader deduplicates the physical acquisition in both duplicate controls.
        reference_key = "duplicate_copy_dedup" if control == "duplicate_copy_raw" else control
        ref = refs[stage3.Q2_LOGICAL_REF_IDX[reference_key]]
        entry = {"reference_key": reference_key, "reference_metrics": metrics(ref, y)[0], "arms": {}}
        denom = h_y - entry["reference_metrics"]["nll"]
        entry["NIG_reference_denominator"] = denom
        for a, arm in enumerate(ARMS):
            logits = raw[a, :, qi]
            dev = np.stack([stage3.softmax_temperature(z, temp) for z, temp in zip(logits, temps[a])])
            opt, opt_info, shared, shared_info = fit_temperatures(logits, row_counts, temps[a])
            dev_metrics = metrics(dev, y)[0]
            opt_metrics = metrics(opt, y)[0]
            ref_dev = posterior_reference(ref, y, dev)
            ref_opt = posterior_reference(ref, y, opt)
            opt_joint = joint_split(row_counts, opt)
            label_marg = np.bincount(y, minlength=4) / len(y)
            avg_ref = ref.mean(axis=0)
            avg_q = opt.mean(axis=(0, 1))
            prior = float(kl(avg_ref, avg_q))
            realized_prior = float(kl(label_marg, avg_q))
            pieces = {
                "reference_entropy": ref_opt["reference_entropy"],
                "temperature_gap": dev_metrics["nll"] - opt_metrics["nll"],
                "reference_marginal_prior_kl": prior,
                "reference_reverse_conditional_kl": ref_opt["expected_kl"] - prior,
                "realized_label_residual": ref_opt["realized_label_residual"],
            }
            close(sum(pieces.values()), dev_metrics["nll"], "Q2 additive path")
            for seed, record in enumerate(opt_info):
                record["T_dev"] = float(temps[a, seed])
                record["dev"] = metrics(dev[seed], y)[0]
                record["opt"] = metrics(opt[seed], y)[0]
            entry["arms"][arm] = {"dev": dev_metrics, "opt": opt_metrics, "shared_opt": metrics(shared, y)[0],
                                   "T_star_by_seed": opt_info, "single_shared_raw_temperature": shared_info,
                                   "additive_path": pieces, "reference_at_T_dev": ref_dev, "reference_at_T_star": ref_opt,
                                   "realized_marginal_prior_kl_at_T_star": realized_prior,
                                   "realized_row_joint_KL_at_T_star": opt_joint,
                                   "forecast_argmax": forecast_split(dev, row_counts, "argmax_set"),
                                   "forecast_vector": forecast_split(dev, row_counts, "vector"),
                                   "NIG_vs_Work_A_realized": (h_y - dev_metrics["nll"]) / denom if denom > 0 else None}
        phase, zero = (entry["arms"][a] for a in ARMS)
        entry["delta_nll_zero_minus_phase"] = {k: zero["additive_path"][k] - v for k, v in phase["additive_path"].items()}
        total = zero["dev"]["nll"] - phase["dev"]["nll"]
        close(sum(entry["delta_nll_zero_minus_phase"].values()), total, "Q2 paired delta")
        entry["delta_nll_zero_minus_phase"]["total"] = total
        entry["delta_nll_zero_minus_phase"]["at_seedwise_T_star"] = zero["opt"]["nll"] - phase["opt"]["nll"]
        result["controls"][control] = entry
    return result


def work_f(bundle, name, temps, seeds, primary):
    def get(key):
        return bundle[f"{name}__{key}"]

    extras = {k: get(k) for k in ("contra_text_flipped_raw_ll", "contra_phase_flipped_raw_ll")} if primary else {}
    text, phase = (extras.get(k) for k in ("contra_text_flipped_raw_ll", "contra_phase_flipped_raw_ll"))
    scores = stage3.work_f_sign_probe(get("raw_ll"), temps, get("codes")[:, 0],
                                      get("targets")[:, 0], get("cal_ref"), get("cal_mass"),
                                      seeds, text, phase)
    if not primary:
        return scores, None
    # Factorial interventions relative to each physical world, evaluated on identical rows.
    raw = get("raw_ll")[0]
    y, codes = get("targets"), get("codes")
    factorial = {}
    for grouping, ix in (("rot0", slice(0, 1)), ("all_saved_rotations", slice(None))):
        variants = {"matched": raw, "text_only_flip": text, "slot_only_flip": phase, "joint_flip": raw[:, ::-1]}
        f, losses = {}, {}
        for k, z in variants.items():
            pred = np.stack([stage3.softmax_temperature(seed, temp) for seed, temp in zip(z, temps[0])])
            f[k], item = metrics(q3_index(pred, codes[:, ix]), y[:, ix])
            rotations = y[:, ix].shape[1]
            losses[k] = item["per_item_nll"].reshape(len(seeds), 2, rotations, -1, 64).mean(axis=(1, 2, 4)).T
        m, a, b, d = (f[k]["nll"] for k in variants)
        f["nll_effects"] = {"text_flip_when_slots_matched": a - m, "slot_flip_when_text_matched": b - m,
                            "difference_in_differences": d - a - b + m,
                            "text_main_effect": (a - m + d - b) / 2,
                            "slot_main_effect": (b - m + d - a) / 2}
        m, a, b, d = (losses[k] for k in variants)
        effect_arrays = {"text_only_flip": a - m, "slot_only_flip": b - m, "joint_flip": d - m,
                         "difference_in_differences": d - a - b + m}
        f["fixed_seed_paired_block_effects"] = {k: stage3.paired_t_ci(v.mean(axis=1)) for k, v in effect_arrays.items()}
        f["effects_by_seed"] = {k: v.mean(axis=0).tolist() for k, v in effect_arrays.items()}
        factorial[grouping] = f
    return scores, factorial


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def components(block_seed):
    b, s = block_seed.shape
    mean = block_seed.mean()
    row = block_seed.mean(axis=1)
    col = block_seed.mean(axis=0)
    res = block_seed - row[:, None] - col + mean
    ms_block = s * row.var(ddof=1)
    ms_seed = b * col.var(ddof=1)
    ms_error = np.sum(res**2) / ((b - 1) * (s - 1))
    signed = [(ms_block - ms_error) / s, (ms_seed - ms_error) / b, ms_error]
    var_u, var_seed, var_e = np.maximum(signed, 0)
    return {"B": b, "S": s, "mean": float(mean),
            "block_conditional_sd": float(row.std(ddof=1)), "across_seed_means_sd": float(col.std(ddof=1)),
            "seed_means": col.tolist(), "mean_squares": [float(v) for v in (ms_block, ms_seed, ms_error)],
            "untruncated_components": [float(v) for v in signed],
            "var_block": float(var_u), "var_seed": float(var_seed), "var_interaction": float(var_e),
            "seed_sd_95_upper": float(np.sqrt(ms_seed * (s - 1) / (b * chi2.ppf(.05, s - 1))))}


def fixed_power(d, b, s, v, alpha):
    vu, _, ve = v
    df = b - 1
    return nct.sf(t.isf(alpha, df), df, d * np.sqrt(b / (vu + ve / s)))


@lru_cache(maxsize=1024)
def chisq_nodes(df, order):
    x, w = roots_legendre(order)
    return chi2.ppf((x + 1) / 2, df) / df, w / 2


def crossed_parameters(b, s, v):
    vu, vs, ve = v
    a = (vu + ve / s) / b
    c = (vs + ve / b) / s
    return a, c, a + vs / s


def crossed_power(d, b, s, v, alpha, order=64):
    a, c, variance = crossed_parameters(b, s, v)
    crit = t.isf(alpha, min(b - 1, s - 1))
    x, wx = chisq_nodes(b - 1, order)
    y, wy = chisq_nodes(s - 1, order)
    return float(wx @ norm.sf((crit * np.sqrt(a * x[:, None] + c * y) - d) / np.sqrt(variance)) @ wy)


def min_fixed(d, s, v, alpha, target, max_b=4096):
    b = np.arange(2, max_b + 1)
    good = np.flatnonzero(fixed_power(d, b, s, v, alpha) >= target)
    return int(b[good[0]]) if good.size else None


def min_crossed(d, s, v, alpha, target, max_b=4096):
    sd_floor = np.sqrt(v[1] / s)
    known_ceiling = float(norm.sf(norm.isf(alpha) - d / sd_floor)) if sd_floor else 1.0
    limit = float(nct.sf(t.isf(alpha, s - 1), s - 1, d / sd_floor)) if sd_floor else 1.0
    if known_ceiling < target:
        return {"B": None, "known_variance_power_ceiling": known_ceiling, "large_B_power": limit,
                "status": "unattainable_even_with_known_variance"}
    # With d fixed, the conditional power is at most P(Z + d/sd_floor > k sqrt(r Y)).
    # For thresholds smaller than d use the floor variance. Above d use probability 1/2
    # as an upper bound, independent of b. This sharper bound applies when b >= s.
    if sd_floor:
        z = d / sd_floor
        k = t.isf(alpha, s - 1)
        y, weights = chisq_nodes(s - 1, 256)
        test_ceiling = float(weights @ np.maximum(.5, norm.sf(k * np.sqrt(y) - z)))
    else:
        test_ceiling = 1.0
    for b in range(2, max_b + 1):
        if b >= s and test_ceiling + 0.001 < target:
            break
        if crossed_power(d, b, s, v, alpha) >= target - .001:
            if crossed_power(d, b, s, v, alpha, 192) >= target:
                return {"B": b, "status": "attained", "known_variance_power_ceiling": known_ceiling,
                        "large_B_power": limit, "searched_through": b}
    return {"B": None, "status": "no_design_in_search_range", "max_b_requested": max_b,
            "searched_through": b - 1 if b >= s and test_ceiling + .001 < target else max_b,
            "known_variance_power_ceiling": known_ceiling, "large_B_power": limit,
            "crossed_power_ceiling_for_B_ge_S_numerical": test_ceiling}


def monte_carlo(v, b, s, d, alpha, rng, reps):
    a, c, variance = crossed_parameters(b, s, v)
    df = b - 1
    z = rng.normal(size=reps)
    xb = rng.chisquare(df, size=reps) / df
    ys = rng.chisquare(s - 1, size=reps) / (s - 1)
    crossed_crit = t.isf(alpha, min(df, s - 1))
    fixed_crit = t.isf(alpha, df)
    crossed_case = crossed_crit * np.sqrt(a * xb + c * ys)
    fixed_case = fixed_crit * np.sqrt(a * xb)
    decisions = (
        ("fixed", z * np.sqrt(a), fixed_case, lambda x: fixed_power(x, b, s, v, alpha)),
        ("crossed", z * np.sqrt(variance), crossed_case, lambda x: crossed_power(x, b, s, v, alpha, 192)),
        ("naive_cluster_t_with_random_seeds", z * np.sqrt(variance), fixed_case,
         lambda x: nct.sf(fixed_crit * np.sqrt(a / variance), df, x / np.sqrt(variance))),
    )
    result = {}
    for label, mean, threshold, power in decisions:
        tests = {}
        for k, effect in (("type_I_boundary", 0.0), ("power", d)):
            empirical = float(np.mean(mean + effect > threshold))
            theoretical = float(power(effect))
            mcse = float(np.sqrt(theoretical * (1 - theoretical) / reps))
            assert abs(empirical - theoretical) < 6 * mcse + .00015, (label, b, effect)
            tests[k] = {"analytic": theoretical, "MC": empirical, "MC_SE": mcse}
        result[label] = tests
    assert result["crossed"]["type_I_boundary"]["analytic"] <= alpha + .0001
    return result


def multiple_testing_mc(rng, reps, alpha):
    # Valid p-values under the true null, with strong shared dependence across claims.
    settings = {}
    for false in (0, 1, 2):
        z = np.sqrt(.6) * rng.normal(size=(reps, 1, 1)) + np.sqrt(.4) * rng.normal(size=(reps, 3, 3))
        p = norm.sf(z)
        p[:, :, 1:] = 0.0  # least-favorable IUT limit when these endpoints have power one
        p[:, :false, :] = 0.0
        p_iut = p.max(axis=-1)
        ix = np.argsort(p_iut, axis=1)
        ordered = np.take_along_axis(p_iut, ix, axis=1)
        rejected = np.logical_and.accumulate(ordered <= alpha / np.arange(3, 0, -1), axis=1)
        fwer = float(np.mean(np.any(rejected & (ix >= false), axis=1)))
        assert fwer <= alpha + .003
        settings[f"false_claim_nulls_{false}"] = {"FWER_MC": fwer,
            "IUT_marginal_type_I_MC": float(np.mean(p_iut[:, false] <= alpha))}
    independent_all_boundary = float(np.mean(np.max(rng.random((reps, 3)), axis=1) <= alpha))
    assert abs(independent_all_boundary - alpha**3) < .0005
    return {"IUT_at_independent_three_endpoint_boundaries": {"exact": alpha**3, "MC": independent_all_boundary},
            "correlated_Holm_IUT": settings}


def stage3_lock_check(pilot):
    scenario = pilot["work_e_scenario"]
    curves = {}
    b = np.arange(2, 4097)
    df = b - 1
    alpha = .025
    for i, (k, gain, width) in enumerate((("accuracy", .05, .02), ("nll", .06, .03))):
        vm = scenario["q"] if i == 0 else scenario["nll_mother_sd"]**2
        sd = max(np.sqrt(vm * (1 + 63 * scenario["rho"]) / 64), scenario["pilot_block_sd"][i])
        close(sd, scenario["sd"][k], "Stage 3D pilot SD")
        crit = t.isf(alpha, df)
        power = nct.sf(crit, df, (gain - .02) * np.sqrt(b) / sd)
        eligible = b[(power >= .95) & (crit * sd / np.sqrt(b) <= width)]
        assert eligible[0] == scenario["t_by_endpoint"][k]
        selected = pilot["b_locked"] - 2
        close(power[selected], scenario["t_at_selected"][k]["power"], "Work E selected NCT power", atol=2e-9)
        curves[k] = {"sd": sd, "min_B": int(eligible[0]), "power_at_131": float(power[selected])}
    return curves


def power_analysis(matrices, pilot):
    alpha, target, s = .05 / 3, .95, 3
    rng = np.random.default_rng(SEED)
    reps = 200_000
    estimates = {cell: {endpoint: components(d) for endpoint, d in entry.items()} for cell, entry in matrices.items()}
    result = {"stage3_lock_replay": stage3_lock_check(pilot), "planning_alpha": alpha,
              "target_marginal_power": target, "seed_count": s, "MC_seed": SEED, "MC_reps": reps,
              "variance_components": estimates, "scenario_proxies": {},
              "multiple_testing_check": multiple_testing_mc(rng, reps, .05), "synthetic_cell_plans": {}}
    g1 = estimates[stage3.CELL_ORDER[0]]
    variance = lambda est: tuple(est[k] for k in ("var_block", "var_seed", "var_interaction"))
    for cell, entry in estimates.items():
        result["synthetic_cell_plans"][cell] = {}
        for endpoint, est in entry.items():
            d = .03 if endpoint == "accuracy" else .04
            v = variance(est)
            result["synthetic_cell_plans"][cell][endpoint] = {
                "effect_above_margin": d, "fixed_min_B": min_fixed(d, s, v, alpha, target),
                "random_min_B": min_crossed(d, s, v, alpha, target),
                "fixed_power_at_131": float(fixed_power(d, 131, s, v, alpha)),
                "random_power_at_131": crossed_power(d, 131, s, v, alpha, 192)}
    designs = [(f"B{b}_independent", b, 1, 0) for b in (16, 72, 131)]
    designs.extend((f"CTTC_9_sessions_x8_rho_{rho:.2f}", 9, 8, rho) for rho in (0, .1, .25))
    for design, b, per_group, rho in designs:
        design_result = {}
        for ep, est in g1.items():
            vu, vs, ve = variance(est)
            v = (vu * (rho + (1 - rho) / per_group), vs, ve / per_group)
            d = .03 if ep == "accuracy" else .04
            ef = max(est["mean"] - .02, 0)
            design_result[ep] = {"independent_groups": b, "clusters_per_group": per_group,
                "assumed_block_effect_session_ICC": rho, "variance_components_at_group_level": list(v),
                "scenarios": {key: {"excess": x, "fixed_min_groups": min_fixed(x, s, v, alpha, target) if x else None,
                                      "random_min_groups": min_crossed(x, s, v, alpha, target) if x else None,
                                      "fixed_power_at_available": float(fixed_power(x, b, s, v, alpha)),
                                      "random_power_at_available": crossed_power(x, b, s, v, alpha, 192)}
                              for key, x in (("planning", d), ("0p02_excess", .02), ("G1_observed_excess", ef))},
                "MC_at_planning_excess": monte_carlo(v, b, s, d, alpha, rng, reps)}
        result["scenario_proxies"][design] = design_result
    result["G1_more_seeds"] = {str(num): {ep: {
        "B131": crossed_power(.03 if ep == "accuracy" else .04, 131, num, variance(est), alpha, 192),
        "B16": crossed_power(.03 if ep == "accuracy" else .04, 16, num, variance(est), alpha, 192)}
        for ep, est in g1.items()} for num in (3, 5, 6, 8, 10, 12)}
    result["ten_seed_sensitivity"] = {}
    for name, scenario in result["scenario_proxies"].items():
        result["ten_seed_sensitivity"][name] = {}
        for endpoint, entry in scenario.items():
            excess = entry["scenarios"]["planning"]["excess"]
            v = entry["variance_components_at_group_level"]
            b = entry["independent_groups"]
            result["ten_seed_sensitivity"][name][endpoint] = {
                "power_at_available": crossed_power(excess, b, 10, v, alpha, 192),
                "random_min_groups": min_crossed(excess, 10, v, alpha, target),
            }
    return result


def dev_futility_check():
    """Independent blind null with dev-only continuation and a dev-chosen plan."""
    information_proxy = .18
    prior_var = 4.0
    prediction_cut = .10
    legacy_cutoff = -.2704380422162607
    alpha_levels = (.05 / 3, .05)
    posterior_var = 1 / (1 / prior_var + information_proxy)
    posterior_slope = np.sqrt(information_proxy) * posterior_var

    # The final standardized score is independent of the dev statistic given the parameter.
    independent_cutoff = ((norm.isf(alpha_levels[-1]) -
                           norm.isf(prediction_cut) * np.sqrt(1 + posterior_var)) / posterior_slope)
    rng = np.random.default_rng(SEED + 1)
    reps = 400_000
    dev = rng.standard_normal(reps)
    numerator = rng.standard_normal(reps)
    # Dev alone chooses the evaluation plan in this check. Generate an exact pivot
    # for each chosen design without observing any blind training or scores.
    use_three = dev >= .8
    df = np.where(use_three, 130, 24)
    blind = numerator / np.sqrt(rng.chisquare(df) / df)
    policies = {}
    for label, cutoff in (("M6_legacy_cutoff_used_as_dev_only_screen", legacy_cutoff),
                          ("independent_normal_predictive_example", independent_cutoff)):
        go = dev >= cutoff
        prob_go = float(norm.sf(cutoff))
        row = {"dev_Z_threshold": float(cutoff), "prob_continue_at_dev_boundary": prob_go,
               "simulated_continue_count": int(go.sum()), "alpha_checks": {}}
        for alpha in alpha_levels:
            reject = blind > t.isf(alpha, df)
            conditional_rate = float(np.mean(reject[go]))
            marginal_rate = float(np.mean(reject & go))
            mc_se = float(np.sqrt(alpha * (1 - alpha) / go.sum()))
            assert abs(conditional_rate - alpha) <= 6 * mc_se
            last = {}
            for s, plan_rows in ((1, ~use_three), (3, use_three)):
                idx = go & plan_rows
                rate = float(np.mean(reject[idx]))
                branch_se = np.sqrt(alpha * (1 - alpha) / idx.sum())
                assert abs(rate - alpha) <= 6 * branch_se
                last[str(s)] = {"simulated_tested": int(idx.sum()), "size_MC": rate,
                                "size_exact": float(t.sf(t.isf(alpha, df[idx][0]), df[idx][0]))}
                close(last[str(s)]["size_exact"], alpha, "conditional plan null size")
            row["alpha_checks"][str(alpha)] = {
                "conditional_size_exact": alpha, "conditional_size_MC": conditional_rate,
                "conditional_size_MC_SE": mc_se, "joint_false_promotion_exact": alpha * prob_go,
                "joint_false_promotion_MC": marginal_rate, "by_selected_seed_count": last,
                "invalid_same_score_screen_conditional_size": min(alpha, prob_go) / prob_go,
            }
            assert abs(marginal_rate - alpha * prob_go) <= 6 * np.sqrt(alpha * prob_go * (1 - alpha * prob_go) / reps)
        policies[label] = row
    return {"seed": SEED + 1, "reps": reps, "dev_information_proxy_for_planning": information_proxy,
            "prior_variance": prior_var, "predictive_continue_probability_threshold": prediction_cut,
            "predictive_calculation_final_alpha": alpha_levels[-1], "policies": policies,
            "illustrative_plan_selection": {"three_seed_dev_Z_cutoff": .8, "one_seed_df": 24, "three_seed_df": 130},
            "size_boundary_assumptions": "Gaussian t pivots conditional on each locked plan, independent dev and blind innovations."}


def conditional_modal_power(primary, pilot):
    """G1 proxy curves for two historical relative endpoints under fixed connectors."""
    alpha = .05 / 3
    target = .95
    excesses = [0, .005, .01, .015, .02, .025, .03, .04, .05, .06, .08, .10]
    result = {"first_step_Holm_alpha": alpha, "per_endpoint_target": target,
              "planning_margin_against_zero": .02, "common_excess_grid": excesses,
              "target_LCB_excess_units": {"accuracy": "proportion", "nll": "nats"},
              "inputs": {}, "designs": {}}
    designs = [
        ("synthetic_32_homogeneous_proxy", 32, 31, 1, 0),
        ("synthetic_32_eight_balanced_strata", 32, 24, 1, 0),
        ("synthetic_131_homogeneous_proxy", 131, 130, 1, 0),
        ("Pluto_16_block_conditional_proxy", 16, 15, 1, 0),
    ]
    for groups_per_capture, name in ((8, "Q2_loops"), (4, "Q3_loop_pairs")):
        designs.extend((f"CTTC_9_captures_{name}_ICC_{rho:.2f}", 9, 8, groups_per_capture, rho)
                       for rho in (0, .10, .25))
    for name, b, df, groups_per_unit, rho in designs:
        entry = {"independent_test_units": b, "student_t_df": df, "groups_per_unit": groups_per_unit,
                 "group_effect_session_ICC": rho, "endpoints": {}}
        cutoff = t.isf(alpha, df)
        standardized_target = brentq(lambda nc: nct.sf(cutoff, df, nc) - target, 0, 30)
        close(nct.sf(cutoff, df, 0), alpha, "conditional power at boundary")
        for endpoint, block_seed in primary.items():
            vc = components(block_seed)
            v_block, v_seed, v_interaction = (vc[k] for k in ("var_block", "var_seed", "var_interaction"))
            plan_excess = .03 if endpoint == "accuracy" else .04
            pilot_sd = pilot["work_e_scenario"]["sd"][endpoint]
            result["inputs"][endpoint] = {
                "vc_stage3d_G1": [v_block, v_seed, v_interaction],
                "stage3_work_E_conservative_block_sd": pilot_sd,
                "observed_seed_202611_block_sd": float(np.std(block_seed[:, 0], ddof=1)),
                "observed_seed_202611_effect_above_margin": float(block_seed[:, 0].mean() - .02),
                "observed_three_seed_bundle_effect_above_margin": float(block_seed.mean() - .02),
                "assumed_planning_effect_above_margin": plan_excess,
            }
            data = {}
            for s in (1, 3):
                frozen = block_seed[:, :s].mean(axis=1)
                obs_excess = float(frozen.mean() - .02)
                deff = rho + (1 - rho) / groups_per_unit
                # ICC belongs to the common group effect in the crossed proxy.
                proxy_sd = np.sqrt(v_block * deff + v_interaction / (s * groups_per_unit))
                # Work E supplies only an aggregate envelope, with no seed/ICC decomposition.
                work_e_sd = pilot_sd * np.sqrt(deff)
                readings = {}
                panel_sd = frozen.std(ddof=1) * np.sqrt(deff)
                for label, sd in (("G1_variance_components_proxy", proxy_sd),
                                  ("historical_frozen_panel_aggregate_ICC_sensitivity", panel_sd),
                                  ("legacy_Work_E_high_variance_sensitivity", work_e_sd)):
                    scale = np.sqrt(b) / sd
                    curve = nct.sf(cutoff, df, np.asarray(excesses) * scale)
                    assert np.all(np.diff(curve) >= -1e-12)
                    min_excess = standardized_target / scale
                    close(nct.sf(cutoff, df, min_excess * scale), target, "power threshold inversion")
                    if df == b - 1 and label == "G1_variance_components_proxy":
                        collapsed = (v_block * deff, v_seed, v_interaction / groups_per_unit)
                        close(nct.sf(cutoff, df, plan_excess * scale),
                              fixed_power(plan_excess, b, s, collapsed, alpha), "conditional power reuse")
                    readings[label] = {"unit_sd": float(sd), "power_curve": curve.tolist(),
                                       "power_at_planning_excess": float(nct.sf(cutoff, df, plan_excess * scale)),
                                       "power_at_historical_observed_excess": float(nct.sf(cutoff, df, obs_excess * scale)),
                                       "minimum_excess_for_95_percent_power": float(min_excess)}
                data[str(s)] = {"historical_effect_above_margin_for_frozen_set": obs_excess, **readings}
            entry["endpoints"][endpoint] = data
        result["designs"][name] = entry
    return result



def run():
    result_dir = SOURCE / "results"
    summary = read_json(result_dir / "summary_master.json")
    lock = read_json(SOURCE / "PROTOCOL_LOCK.json")
    q3_result, score_matrices, parity = {}, {}, {}
    with np.load(result_dir / "blind_q3_scores.npz", allow_pickle=False) as q3, \
         np.load(result_dir / "blind_q2_scores.npz", allow_pickle=False) as q2:
        seeds = q3["seeds"].tolist()
        close(seeds, summary["seeds"], "seed IDs", atol=0)
        t3, t2 = q3["temps_q3"], q2["temps_q2"]
        for q, temp in (("q3", t3), ("q2", t2)):
            close(temp, summary[f"locked_temps_{q}"], f"{q} summary temperatures", atol=0)
            lock_temp = [[lock["checkpoints"][f"{a}_s{s}"][f"temp_{q}"] for s in seeds] for a in ARMS]
            close(temp, lock_temp, f"{q} protocol temperatures", atol=0)
        for name in (*stage3.CELL_ORDER, "H_c4_035"):
            q3_result[name], score_matrices[name] = q3_analysis(q3, name, t3)
            if name not in summary["cells"]:
                continue
            keys = ("raw_ll", "codes", "targets", "direct_table_probs", "full_iq_probs")
            original = stage3.evaluate_cell_paired(q3[f"{name}__raw_ll"], t3,
                *(q3[f"{name}__{k}"] for k in keys[1:]), seeds, is_primary=name == stage3.CELL_ORDER[0])
            expected = summary["cells"][name]
            parity[name] = check_tree({**stage3.CELL_META[name], **original}, expected, name)
            for arm in ARMS:
                independent = q3_result[name]["arms"][arm]["metrics"]["dev"]
                close(list(independent.values()), [expected["arms"][arm][k] for k in METRICS], name + arm)
            for endpoint, key in (("accuracy", "acc_gain_t95"), ("nll", "nll_improv_t95")):
                block_adj = score_matrices[name][endpoint].mean(axis=1)
                check_tree(stage3.paired_t_ci(block_adj), expected["paired_phase_minus_zero"][key], name + endpoint)
        primary = stage3.CELL_ORDER[0]
        f, f_factorial = work_f(q3, primary, t3, seeds, True)
        hist, _ = work_f(q3, "H_c4_035", t3, seeds, False)
        parity["work_f_primary_040"] = check_tree(f, summary["work_f_primary_040"], "Work F primary")
        parity["work_f_historical_035"] = check_tree(hist, summary["work_f_historical_035"], "Work F historical")
        work_g = stage3.work_g_rotation_and_controls(q3[f"{primary}__raw_ll"], t3,
            *(q3[f"{primary}__{k}"] for k in ("codes", "targets", "rot10_codes", "rot10_targets",
              "indep_raw_ll", "indep_targets", "shuffled_perm")), seeds)
        parity["work_g_primary_040"] = check_tree(work_g, summary["work_g_primary_040"], "Work G")
        for a in ARMS:
            for k, field in (("arith", "offline_group_avg_16way"), ("F0", "offline_representative_F0")):
                independent = q3_result[primary]["arms"][a]["metrics"][k]
                close([independent["acc"], independent["nll"]], [work_g["arms"][a][field][x]
                      for x in ("acc", "nll")], f"independent Work G {a} {k}")
        q2_result = q2_analysis(q2, t2)
        q2_original = stage3.work_a_q2_analysis(q2, t2, seeds)
        parity["work_a_q2_c4"] = check_tree(q2_original, summary["work_a_q2_c4"], "Q2 Work A")
        for control, entry in q2_result["controls"].items():
            for a in ARMS:
                replay = q2_original["arms"][a]["by_arm"][control]
                close(list(entry["arms"][a]["dev"].values()), [replay[k] for k in METRICS], f"independent Q2 {control} {a}")
        inventory = {"q3": {k: list(q3[k].shape) for k in q3.files},
                     "q2": {k: list(q2[k].shape) for k in q2.files}}
    report = {"runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
              "Q3": q3_result, "Q2": q2_result, "npz_shapes": inventory,
              "summary_checks_max_absolute_difference": parity,
              "work_f_primary_040": f, "work_f_historical_035": hist,
              "work_f_intervention_factorial": f_factorial,
              "power": power_analysis(score_matrices, lock["dev_pilot"]),
              "round2": {"dev_disjoint_futility": dev_futility_check(),
                         "conditional_modal_power": conditional_modal_power(
                             score_matrices[stage3.CELL_ORDER[0]], lock["dev_pilot"])}}
    destination = HERE / "results_m2.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{destination.name} written. Maximum in-scope summary difference: {max(parity.values()):.3g}.")


if __name__ == "__main__":
    run()
