"""Step 3: Closed-Book Multi-Cluster Formal Evaluation Runner + Exact Single-Process Replay.

Strictly enforces:
  - `--confirm-frozen` and `verify_protocol_freeze()` BEFORE touching any `SEED_S5_TEST_*` seed.
  - Single-worker (`workers=1`) deterministic execution (`OMP_NUM_THREADS=1 ...`, float64/complex128).
  - Paired evaluation across all physics fusion arms + Stage 5 Role A vs Role B (3-seed ensembles)
    + oracle true-channel reference on 4 ID regimes + true delay reset + 2 unseen OOD regimes.
  - Mandatory exact single-process replay check on `replay_clusters` verifying `max_float_diff == 0.0`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from p12.contract import OFDM_PILOT_COMB
from p12.offgrid_physics import simulate_offgrid_link_p12
from p12.stage1_secondary_diag import _eval_oracle_true_channel

from research.p12_stage5_fusion_neural.audit import audit_stage5_formal
from research.p12_stage5_fusion_neural.consistent_fusion_engine import (
    ContentAddressingConfig,
    MixtureDecodeConfig,
    evaluate_decoder_mode_on_folds,
    run_consistent_folds,
    warmup_p12_offgrid_numba,
)
from research.p12_stage5_fusion_neural.contract import (
    ID_REGIMES,
    OOD_AND_RESET_REGIMES,
    SEED_S5_TEST_ID,
    SEED_S5_TEST_OOD,
    STAGE5_SEEDS,
    enforce_determinism,
)
from research.p12_stage5_fusion_neural.freeze import (
    DEFAULT_FREEZE_PATH,
    FORMAL_ID_OFFSET,
    FORMAL_OOD_OFFSET,
    PKG_DIR,
    verify_protocol_freeze,
)
from research.p12_stage5_fusion_neural.neural_models import infer_stage5_fold_override
from research.p12_stage5_fusion_neural.step2_train_stage5 import load_stage5_checkpoint


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(json.dumps(r, allow_nan=False) + "\n")
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _eval_one_link(
    task: tuple[str, int, str, int, int, float],
    *,
    mix_cfg: MixtureDecodeConfig,
    content_cfg: ContentAddressingConfig | None,
    physics_variant: str,
    models_a: list,
    models_b: list,
    eval_per_seed_on_id: bool,
) -> dict[str, Any]:
    split, cluster, regime, link_idx, seed, snr_db = task
    obs, truth, extra = simulate_offgrid_link_p12(seed, regime, snr_db=snr_db, pilot_rows_a=OFDM_PILOT_COMB)

    # 1) Exact 655de6a consistent fold engine -> shared by base & hard_choice
    f_base = run_consistent_folds(obs, content_cfg=None, wave_mix_cfg=None)
    sc_base = evaluate_decoder_mode_on_folds(obs, truth, f_base, mode="mean_channel", mix_cfg=mix_cfg)
    sc_hard = evaluate_decoder_mode_on_folds(obs, truth, f_base, mode="hard_choice", mix_cfg=mix_cfg)

    # 2) Full mixture LLR physics engine (without content weighting)
    f_mix = run_consistent_folds(obs, content_cfg=None, wave_mix_cfg=mix_cfg)
    sc_mix = evaluate_decoder_mode_on_folds(
        obs, truth, f_mix, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
    )

    # 3) Content-addressed + mixture LLR physics engine (always evaluated for full comparison!)
    cnt_cfg_eval = content_cfg if content_cfg is not None else ContentAddressingConfig()
    f_cnt = run_consistent_folds(obs, content_cfg=cnt_cfg_eval, wave_mix_cfg=mix_cfg)
    sc_cnt = evaluate_decoder_mode_on_folds(
        obs, truth, f_cnt, mode="mixture_llr", mix_cfg=mix_cfg, also_score_shared_hard=True
    )

    # 4) Stage 5 Neural Refiners (Role A vs Role B) strictly on top of the frozen winning physics folds!
    f_stage5 = f_mix if physics_variant == "mixture_llr" else f_cnt

    t_na = time.perf_counter()
    ov_a = {snap.fold_idx: infer_stage5_fold_override(models_a, snap) for snap in f_stage5.folds}
    na_ms = (time.perf_counter() - t_na) * 1000.0
    sc_ra = evaluate_decoder_mode_on_folds(obs, truth, f_stage5, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ov_a)
    sc_ra["neural_ms"] = float(na_ms)
    sc_ra["elapsed_ms"] = float(sc_ra["elapsed_ms"] + na_ms)

    t_nb = time.perf_counter()
    ov_b = {snap.fold_idx: infer_stage5_fold_override(models_b, snap) for snap in f_stage5.folds}
    nb_ms = (time.perf_counter() - t_nb) * 1000.0
    sc_rb = evaluate_decoder_mode_on_folds(obs, truth, f_stage5, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ov_b)
    sc_rb["neural_ms"] = float(nb_ms)
    sc_rb["elapsed_ms"] = float(sc_rb["elapsed_ms"] + nb_ms)

    arms_dict: dict[str, Any] = {
        "p12_consistent_base": sc_base,
        "p12_hard_choice": sc_hard,
        "p12_mixture_llr": sc_mix,
        "p12_mixture_llr_content": sc_cnt,
        "stage5_role_a_ens": sc_ra,
        "stage5_role_b_ens": sc_rb,
    }

    if eval_per_seed_on_id and split == "id":
        for s_val, m_a, m_b in zip(STAGE5_SEEDS, models_a, models_b):
            ova_s = {snap.fold_idx: infer_stage5_fold_override([m_a], snap) for snap in f_stage5.folds}
            ovb_s = {snap.fold_idx: infer_stage5_fold_override([m_b], snap) for snap in f_stage5.folds}
            arms_dict[f"stage5_role_a_s{s_val}"] = evaluate_decoder_mode_on_folds(
                obs, truth, f_stage5, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ova_s
            )
            arms_dict[f"stage5_role_b_s{s_val}"] = evaluate_decoder_mode_on_folds(
                obs, truth, f_stage5, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ovb_s
            )

    arms_dict["oracle_true_channel"] = _eval_oracle_true_channel(obs, truth, use_true_burst_noise=False)

    return {
        "split": split,
        "cluster": int(cluster),
        "regime": regime,
        "link": int(link_idx),
        "seed": int(seed),
        "snr_db": float(snr_db),
        "n_rays": int(truth.n_rays),
        "anchor_row": int(extra["anchor_row"]),
        "arms": arms_dict,
    }


def _build_formal_tasks(manifest: Mapping[str, Any]) -> tuple[list, list]:
    snr_db = float(manifest["snr_db"])
    clusters = [int(c) for c in manifest["clusters"]]
    l_id = int(manifest["links_per_cluster_id"])
    l_ood = int(manifest["links_per_cluster_ood"])

    t_id = [
        ("id", c, reg, l_i, int(SEED_S5_TEST_ID + FORMAL_ID_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in clusters
        for r_i, reg in enumerate(ID_REGIMES)
        for l_i in range(l_id)
    ]
    t_ood = [
        ("ood", c, reg, l_i, int(SEED_S5_TEST_OOD + FORMAL_OOD_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in clusters
        for r_i, reg in enumerate(OOD_AND_RESET_REGIMES)
        for l_i in range(l_ood)
    ]
    return t_id, t_ood


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-4 Step 3: Closed-Book Formal Multi-Cluster Runner")
    parser.add_argument("--confirm-frozen", action="store_true", required=True)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_FREEZE_PATH))
    parser.add_argument("--ckpt-dir", type=str, default=str(PKG_DIR / "ckpts"))
    parser.add_argument("--out-dir", type=str, default=str(PKG_DIR / "results"))
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    enforce_determinism(require_env=args.require_env)
    warmup_p12_offgrid_numba()

    ckpt_dir = Path(args.ckpt_dir)
    manifest = verify_protocol_freeze(Path(args.manifest), ckpt_dir=ckpt_dir)
    mix_cfg = MixtureDecodeConfig(**manifest["mix_cfg"])
    content_cfg = None if manifest["content_cfg"] is None else ContentAddressingConfig(**manifest["content_cfg"])
    physics_variant = str(manifest["physics_variant"])
    upgraded_arm = str(manifest["upgraded_phys_arm"])
    eval_per_seed_on_id = bool(manifest.get("eval_per_seed_on_id", False))

    models_a = [load_stage5_checkpoint(ckpt_dir / f"stage5_role_a_seed{s}.pt") for s in STAGE5_SEEDS]
    models_b = [load_stage5_checkpoint(ckpt_dir / f"stage5_role_b_seed{s}.pt") for s in STAGE5_SEEDS]

    t_id, t_ood = _build_formal_tasks(manifest)
    replay_clusters = set(int(c) for c in manifest["replay_clusters"])
    t_rep = [t for t in (t_id + t_ood) if int(t[1]) in replay_clusters]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== STEP 3 FORMAL CLOSED-BOOK STUDY: G={manifest['n_clusters']}, "
        f"ID={len(t_id)} links, OOD={len(t_ood)} links, Replay={len(t_rep)} links, variant={physics_variant} ===",
        flush=True,
    )

    def _run_list(tasks: list, label: str, save_path: Path) -> list[dict[str, Any]]:
        ts = time.perf_counter()
        recs: list[dict[str, Any]] = []
        for idx, tk in enumerate(tasks):
            recs.append(
                _eval_one_link(
                    tk,
                    mix_cfg=mix_cfg,
                    content_cfg=content_cfg,
                    physics_variant=physics_variant,
                    models_a=models_a,
                    models_b=models_b,
                    eval_per_seed_on_id=eval_per_seed_on_id,
                )
            )
            if (idx + 1) % 32 == 0 or (idx + 1) == len(tasks):
                dt = time.perf_counter() - ts
                print(f"[{label}] {idx+1}/{len(tasks)} links in {dt:.1f}s ({dt/(idx+1):.2f}s/link)", flush=True)
        _write_jsonl(save_path, recs)
        return recs

    id_recs = _run_list(t_id, "Formal-ID", out_dir / "s5_formal_id_links.jsonl")
    ood_recs = _run_list(t_ood, "Formal-OOD", out_dir / "s5_formal_ood_links.jsonl")
    rep_recs = _run_list(t_rep, "Formal-Replay-1w", out_dir / "s5_formal_replay_links.jsonl")

    audit = audit_stage5_formal(
        id_recs,
        ood_recs,
        rep_recs,
        clusters=manifest["clusters"],
        upgraded_phys_arm=upgraded_arm,
    )
    audit["freeze_bundle_sha256"] = manifest["bundle_sha256"]
    audit["physics_variant"] = physics_variant
    audit["total_formal_wall_s"] = float(time.perf_counter() - t0)
    _write_json(out_dir / "s5_formal_audit.json", audit)

    print(f"\n=== STEP 3 FORMAL COMPLETE in {audit['total_formal_wall_s']:.1f}s ===", flush=True)
    print(
        json.dumps(
            {
                "replay_audit": audit["replay_audit"],
                "pooled_4id": {k: round(v["bler_pct"], 2) for k, v in audit["pooled_4id"].items()},
                "pooled_ood_only": {k: round(v["bler_pct"], 2) for k, v in audit["pooled_ood_only"].items()},
                "holm_family_physics": {
                    k: {"mean_pp": round(v["mean_pp"], 2), "p_holm": v["p_holm"], "verdict": v["verdict"]}
                    for k, v in audit["holm_family_physics"].items()
                },
                "holm_family_neural": {
                    k: {"mean_pp": round(v["mean_pp"], 2), "p_holm": v["p_holm"], "verdict": v["verdict"]}
                    for k, v in audit["holm_family_neural"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
