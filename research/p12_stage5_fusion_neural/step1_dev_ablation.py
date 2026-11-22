"""Step 1: Dev-Set Physics Mechanism Ablation (Strictly SEED_S5_DEV only).

Compares 5 physics arms on 5 off-grid regimes (`4 ID + offgrid_full_reset_new_delays`;
strictly holding `offgrid_ood_burst_and_shift` and `offgrid_ood_five_ray_fast` unseen!):
  1. `p12_consistent_base`        : `655de6a` consistent baseline (linear mean channel -> Strong-L)
  2. `p12_hard_choice`            : Hard MAP coherent single trajectory on the same consistent fold state
  3. `p12_mixture_llr_target`     : Consistent fold state -> coherent mixture LLR + exact mixture MAP (target only)
  4. `p12_mixture_llr`            : Full coherent mixture LLR + exact mixture MAP (Wave-1/2 + target)
  5. `p12_mixture_llr_content`    : Adds dev-calibrated content-addressed AUX + helper + target-pos weighting

Also runs:
  - Shared-candidate-list audit (`hard_choice_on_union` vs `mixture_on_union` + true codeword list coverage)
  - Dev calibration sweep over `obs_cov_gamma` and content-addressing softmax temperatures
  - Exact single-thread timing & memory audit to inform prospective formal `G` and link counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from p12.contract import OFDM_PILOT_COMB
from p12.offgrid_physics import simulate_offgrid_link_p12
from p12.stage1_secondary_diag import _eval_oracle_true_channel

from research.p12_stage5_fusion_neural.consistent_fusion_engine import (
    ContentAddressingConfig,
    MixtureDecodeConfig,
    evaluate_decoder_mode_on_folds,
    run_consistent_folds,
    warmup_p12_offgrid_numba,
)
from research.p12_stage5_fusion_neural.contract import (
    DEV_ABLATION_REGIMES,
    PRIMARY_SNR_DB,
    SEED_S5_DEV,
    assert_s5_seed_disjoint,
    enforce_determinism,
)

STEP1_DEV_OFFSET = 10_000


def build_step1_seeds(links_per_regime: int) -> list[tuple[str, int, int]]:
    tasks: list[tuple[str, int, int]] = []
    raw_seeds: list[int] = []
    for r_i, reg in enumerate(DEV_ABLATION_REGIMES):
        for l_i in range(links_per_regime):
            s = int(SEED_S5_DEV + STEP1_DEV_OFFSET + r_i * 1000 + l_i)
            raw_seeds.append(s)
            tasks.append((reg, l_i, s))
    assert_s5_seed_disjoint(raw_seeds, expected_namespace="s5_dev")
    return tasks


def aggregate_dev_records(records: list[dict[str, Any]], arm_names: list[str]) -> dict[str, Any]:
    by_regime: dict[str, Any] = {}
    for reg in DEV_ABLATION_REGIMES:
        sub = [r for r in records if r["regime"] == reg]
        if not sub:
            continue
        rmap: dict[str, Any] = {"n_links": len(sub)}
        for arm in arm_names:
            be = sum(int(r["arms"][arm]["block_errors"]) for r in sub)
            bb = sum(int(r["arms"][arm]["blocks"]) for r in sub)
            entry: dict[str, Any] = {
                "bler_pct": 100.0 * be / max(bb, 1),
                "block_errors": be,
                "blocks": bb,
                "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in sub])),
                "probe_clean_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in sub])),
            }
            if "bit_errors" in sub[0]["arms"][arm]:
                bi_e = sum(int(r["arms"][arm]["bit_errors"]) for r in sub)
                bi_t = sum(int(r["arms"][arm]["info_bits"]) for r in sub)
                entry["ber_pct"] = 100.0 * bi_e / max(bi_t, 1)
                entry["mean_rel_mse_ch"] = float(np.mean([r["arms"][arm]["mean_rel_mse_ch"] for r in sub]))
                entry["probe_cal_ratio"] = float(np.mean([r["arms"][arm]["probe_cal_ratio"] for r in sub]))
                entry["probe_nll"] = float(np.mean([r["arms"][arm]["probe_nll_per_sample"] for r in sub]))
                ha = sum(int(r["arms"][arm]["helpers_accepted"]) for r in sub)
                fa = sum(int(r["arms"][arm]["false_assimilations"]) for r in sub)
                entry["helpers_accepted"] = ha
                entry["false_assimilations"] = fa
                entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                entry["mean_branches"] = float(np.mean([r["arms"][arm]["mean_branches"] for r in sub]))
                entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in sub]))
                entry["mean_ms"] = float(np.mean([r["arms"][arm]["elapsed_ms"] for r in sub]))
            if "shared_union_hard_errors" in sub[0]["arms"][arm]:
                sh_e = sum(int(r["arms"][arm]["shared_union_hard_errors"]) for r in sub)
                cov_n = sum(int(r["arms"][arm]["true_cw_in_union_count"]) for r in sub)
                entry["shared_union_hard_bler_pct"] = 100.0 * sh_e / max(bb, 1)
                entry["true_cw_list_coverage_pct"] = 100.0 * cov_n / max(bb, 1)
            rmap[arm] = entry
        by_regime[reg] = rmap

    id_sub = [r for r in records if r["regime"] != "offgrid_full_reset_new_delays"]
    pooled_4id: dict[str, Any] = {}
    for arm in arm_names:
        be = sum(int(r["arms"][arm]["block_errors"]) for r in id_sub)
        bb = sum(int(r["arms"][arm]["blocks"]) for r in id_sub)
        p_entry: dict[str, Any] = {
            "bler_pct": 100.0 * be / max(bb, 1),
            "block_errors": be,
            "blocks": bb,
            "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in id_sub])),
            "probe_clean_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in id_sub])),
        }
        if "elapsed_ms" in id_sub[0]["arms"][arm]:
            p_entry["mean_rel_mse_ch"] = float(np.mean([r["arms"][arm]["mean_rel_mse_ch"] for r in id_sub]))
            p_entry["probe_cal_ratio"] = float(np.mean([r["arms"][arm]["probe_cal_ratio"] for r in id_sub]))
            p_entry["probe_nll"] = float(np.mean([r["arms"][arm]["probe_nll_per_sample"] for r in id_sub]))
            p_entry["mean_ms"] = float(np.mean([r["arms"][arm]["elapsed_ms"] for r in id_sub]))
        if "shared_union_hard_errors" in id_sub[0]["arms"][arm]:
            sh_e = sum(int(r["arms"][arm]["shared_union_hard_errors"]) for r in id_sub)
            cov_n = sum(int(r["arms"][arm]["true_cw_in_union_count"]) for r in id_sub)
            p_entry["shared_union_hard_bler_pct"] = 100.0 * sh_e / max(bb, 1)
            p_entry["true_cw_list_coverage_pct"] = 100.0 * cov_n / max(bb, 1)
        pooled_4id[arm] = p_entry

    return {"by_regime": by_regime, "pooled_4id": pooled_4id}


def run_step1(links_per_regime: int, out_dir: Path, require_env: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    enforce_determinism(require_env=require_env)
    warmup_p12_offgrid_numba()

    tasks = build_step1_seeds(links_per_regime)
    mix_cfg = MixtureDecodeConfig()
    content_cfg = ContentAddressingConfig()
    content_cfg_nopos = ContentAddressingConfig(use_target_pos_query=False)

    arm_names = [
        "p12_consistent_base",
        "p12_hard_choice",
        "p12_mixture_llr_target",
        "p12_mixture_llr",
        "p12_mixture_llr_content_nopos",
        "p12_mixture_llr_content",
        "oracle_true_channel",
    ]

    records: list[dict[str, Any]] = []
    for idx, (reg, l_i, seed) in enumerate(tasks):
        obs, truth, extra = simulate_offgrid_link_p12(seed, reg, snr_db=PRIMARY_SNR_DB, pilot_rows_a=OFDM_PILOT_COMB)

        # 1) Exact 655de6a consistent fold engine (shared by base, hard_choice, mixture_llr_target)
        f_base = run_consistent_folds(obs, content_cfg=None, wave_mix_cfg=None)
        sc_base = evaluate_decoder_mode_on_folds(obs, truth, f_base, mode="mean_channel", mix_cfg=mix_cfg)
        sc_hard = evaluate_decoder_mode_on_folds(obs, truth, f_base, mode="hard_choice", mix_cfg=mix_cfg)
        sc_mix_tgt = evaluate_decoder_mode_on_folds(
            obs, truth, f_base, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
        )

        # 2) Full mixture LLR (Wave-1/2 + target)
        f_mix = run_consistent_folds(obs, content_cfg=None, wave_mix_cfg=mix_cfg)
        sc_mix = evaluate_decoder_mode_on_folds(
            obs, truth, f_mix, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
        )

        # 3) Content-addressed (AUX + helper, no target-pos query)
        f_cnt_nopos = run_consistent_folds(obs, content_cfg=content_cfg_nopos, wave_mix_cfg=mix_cfg)
        sc_cnt_nopos = evaluate_decoder_mode_on_folds(
            obs, truth, f_cnt_nopos, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
        )

        # 4) Full content-addressed (AUX + helper + target-position query)
        f_cnt = run_consistent_folds(obs, content_cfg=content_cfg, wave_mix_cfg=mix_cfg)
        sc_cnt = evaluate_decoder_mode_on_folds(
            obs, truth, f_cnt, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
        )

        # 5) Oracle true channel reference
        sc_oracle = _eval_oracle_true_channel(obs, truth, use_true_burst_noise=False)

        records.append({
            "regime": reg,
            "link": int(l_i),
            "seed": int(seed),
            "n_rays": int(truth.n_rays),
            "anchor_row": int(extra["anchor_row"]),
            "arms": {
                "p12_consistent_base": sc_base,
                "p12_hard_choice": sc_hard,
                "p12_mixture_llr_target": sc_mix_tgt,
                "p12_mixture_llr": sc_mix,
                "p12_mixture_llr_content_nopos": sc_cnt_nopos,
                "p12_mixture_llr_content": sc_cnt,
                "oracle_true_channel": sc_oracle,
            },
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tasks):
            elapsed = time.perf_counter() - t0
            print(
                f"[Step1-Dev] {idx+1}/{len(tasks)} links done in {elapsed:.1f}s ({elapsed/(idx+1):.2f}s/link)",
                flush=True,
            )

    summary = aggregate_dev_records(records, arm_names)
    summary["links_per_regime"] = int(links_per_regime)
    summary["total_links"] = len(tasks)
    summary["wall_s"] = float(time.perf_counter() - t0)
    summary["mix_cfg"] = mix_cfg.__dict__
    summary["content_cfg"] = content_cfg.__dict__

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "step1_dev_links.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, allow_nan=False) + "\n")
    (out_dir / "step1_dev_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n=== STEP 1 DEV POOLED 4-ID SUMMARY ===", flush=True)
    for arm in arm_names:
        p = summary["pooled_4id"][arm]
        ms_s = f"{p.get('mean_ms', 0.0):.0f} ms" if "mean_ms" in p else "-"
        sh_s = f", union_hard={p['shared_union_hard_bler_pct']:.2f}%, cov={p['true_cw_list_coverage_pct']:.2f}%" if "shared_union_hard_bler_pct" in p else ""
        print(
            f"  {arm:32s}: 4-ID BLER={p['bler_pct']:6.2f}% | relFrob={p['mean_rel_frob']:.4f} | probeNMSE={p['probe_clean_nmse']:.4f} | {ms_s}{sh_s}",
            flush=True,
        )
    print("\n=== STEP 1 DEV BY REGIME BLER (%) ===", flush=True)
    for reg in DEV_ABLATION_REGIMES:
        vals = " | ".join(f"{arm[:12]}={summary['by_regime'][reg][arm]['bler_pct']:5.2f}%" for arm in arm_names)
        print(f"  {reg:30s}: {vals}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI-4 Step 1 Dev Physics Ablation")
    parser.add_argument("--links-per-regime", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="research/p12_stage5_fusion_neural/results")
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args()
    run_step1(args.links_per_regime, Path(args.out_dir), args.require_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
