"""Step 2: Offline Fold Dataset Builder & Stage 5 Fair Training (Role A vs Role B, 3 Seeds).

Strictly uses ONLY `SEED_S5_TRAIN` and `SEED_S5_DEV` (offset 50_000, disjoint from Step 1
and strictly disjoint from `SEED_S5_TEST_ID` / `SEED_S5_TEST_OOD` and all P1..P12 seeds).
Trains Role A (`SymmetricTransportRefiner`) and Role B (`NonPhysicsTransformerRefiner`) with:
  - EXACT integer real-parameter parity (`P_A == P_B`)
  - EXACT same fold feature inputs, training links, dev links, loss, optimizer, LR schedule,
    epochs, batch size, gradient clipping, and 3 random seeds `(5001, 5002, 5003)`.
  - Identical dev-only `(fam_scale, res_scale)` output calibration before freezing checkpoints!
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from p11.contract import N, fractional_delay_freq_response, unitary_dft_matrix
from p12.contract import OFDM_PILOT_COMB
from p12.offgrid_physics import simulate_offgrid_link_p12

from research.p12_stage5_fusion_neural.consistent_fusion_engine import (
    ContentAddressingConfig,
    MixtureDecodeConfig,
    evaluate_decoder_mode_on_folds,
    run_consistent_folds,
    warmup_p12_offgrid_numba,
)
from research.p12_stage5_fusion_neural.contract import (
    PRIMARY_SNR_DB,
    SEED_S5_DEV,
    SEED_S5_TRAIN,
    STAGE5_SEEDS,
    TRAIN_REGIMES,
    assert_s5_seed_disjoint,
    enforce_determinism,
    sha256_file,
)
from research.p12_stage5_fusion_neural.neural_models import (
    COMPLEX,
    REAL,
    NonPhysicsTransformerRefiner,
    SymmetricTransportRefiner,
    build_matched_stage5_models,
    compute_family_summaries,
    extract_stage5_fold_features,
    infer_stage5_fold_override,
    stage5_differentiable_loss,
)

TRAIN_SEED_OFFSET = 100_000
DEV_STAGE5_OFFSET = 50_000  # Strictly disjoint from Step 1's 10_000 offset!


def get_physics_configs(variant: str) -> tuple[Optional[ContentAddressingConfig], MixtureDecodeConfig]:
    mix_cfg = MixtureDecodeConfig()
    if variant == "mixture_llr":
        return None, mix_cfg
    if variant == "mixture_llr_content_nopos":
        return ContentAddressingConfig(use_target_pos_query=False), mix_cfg
    if variant == "mixture_llr_content":
        return ContentAddressingConfig(use_target_pos_query=True), mix_cfg
    raise ValueError(f"Unknown physics variant: {variant!r}")


def build_split_tasks(split: str, links_per_regime: int) -> list[tuple[str, int, int]]:
    if split == "train":
        base = SEED_S5_TRAIN + TRAIN_SEED_OFFSET
        ns = "s5_train"
    elif split == "dev":
        base = SEED_S5_DEV + DEV_STAGE5_OFFSET
        ns = "s5_dev"
    else:
        raise ValueError(f"Invalid split {split!r}")
    tasks: list[tuple[str, int, int]] = []
    seeds: list[int] = []
    for r_i, reg in enumerate(TRAIN_REGIMES):
        for l_i in range(links_per_regime):
            s = int(base + r_i * 10_000 + l_i)
            seeds.append(s)
            tasks.append((reg, l_i, s))
    assert_s5_seed_disjoint(seeds, expected_namespace=ns)
    return tasks


def generate_fold_dataset(
    tasks: list[tuple[str, int, int]],
    *,
    content_cfg: Optional[ContentAddressingConfig],
    mix_cfg: MixtureDecodeConfig,
    label: str,
) -> dict[str, np.ndarray]:
    """Generate compact offline fold feature & target arrays (strictly fold-excluded)."""
    t0 = time.perf_counter()
    dft = unitary_dft_matrix(N)

    items: dict[str, list[np.ndarray]] = {
        "scalar_feats": [],
        "complex_feats": [],
        "tau_hat": [],
        "nu_post": [],
        "t_phys": [],
        "t_norm": [],
        "ev_base": [],
        "t_rows_idx": [],
        "probe_row_idx": [],
        "rot_intra_t": [],
        "rot_intra_p": [],
        "fam_logw": [],
        "fam_active": [],
        "fam_mean_t": [],
        "fam_var_t": [],
        "fam_mean_p": [],
        "fam_var_p": [],
        "d_mats": [],
        "phi_f": [],
        "h_true_u": [],
        "y_target_freq_u": [],
        "b_code_target": [],
        "u_probe": [],
        "probe_clean_u": [],
    }

    for idx, (reg, _l_i, seed) in enumerate(tasks):
        obs, truth, _ = simulate_offgrid_link_p12(seed, reg, snr_db=PRIMARY_SNR_DB, pilot_rows_a=OFDM_PILOT_COMB)
        folds = run_consistent_folds(obs, content_cfg=content_cfg, wave_mix_cfg=mix_cfg)
        s_mag = math.sqrt(float(obs.n0))
        row_to_d = {int(r): i for i, r in enumerate(truth.data_rows)}

        for snap in folds.folds:
            feats = extract_stage5_fold_features(snap)
            fam_logw, fam_mean_u, fam_var_u, fam_active = compute_family_summaries(snap)
            t_idx = list(snap.t_rows)
            d_idxs = [row_to_d[r] for r in t_idx]
            y_t_u = obs.y_b[t_idx] / s_mag
            y_t_freq = y_t_u @ dft.T
            phi_f = fractional_delay_freq_response(snap.used_delays, n=N)

            items["scalar_feats"].append(feats.scalar_feats)
            items["complex_feats"].append(feats.complex_feats)
            items["tau_hat"].append(feats.tau_hat)
            items["nu_post"].append(feats.nu_post)
            items["t_phys"].append(feats.t_phys)
            items["t_norm"].append(feats.t_norm)
            items["ev_base"].append(feats.ev_base)
            items["t_rows_idx"].append(np.asarray(t_idx, dtype=np.int64))
            items["probe_row_idx"].append(np.int64(snap.probe_row))
            items["rot_intra_t"].append(feats.rot_intra[:, t_idx, :])
            items["rot_intra_p"].append(feats.rot_intra[:, snap.probe_row, :])
            items["fam_logw"].append(fam_logw)
            items["fam_active"].append(fam_active)
            items["fam_mean_t"].append(fam_mean_u[:, :, t_idx, :])
            items["fam_var_t"].append(fam_var_u[:, :, t_idx, :])
            items["fam_mean_p"].append(fam_mean_u[:, :, snap.probe_row, :])
            items["fam_var_p"].append(fam_var_u[:, :, snap.probe_row, :])
            items["d_mats"].append(snap.d_mats)
            items["phi_f"].append(phi_f)
            items["h_true_u"].append(truth.h_b_true_mat[t_idx] / s_mag)
            items["y_target_freq_u"].append(y_t_freq)
            items["b_code_target"].append(truth.b_code[d_idxs].astype(np.float64).reshape(len(t_idx), N, 4))
            items["u_probe"].append(snap.u_probe)
            items["probe_clean_u"].append(truth.y_b_probe_clean / s_mag)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            dt = time.perf_counter() - t0
            print(f"[{label}-Cache] {idx+1}/{len(tasks)} links ({len(items['tau_hat'])} folds) in {dt:.1f}s", flush=True)

    return {k: np.stack(v, axis=0) for k, v in items.items()}


def _batch_to_tensors(data: dict[str, np.ndarray], idxs: np.ndarray) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, arr in data.items():
        sub = arr[idxs]
        if np.iscomplexobj(sub):
            out[k] = torch.from_numpy(np.ascontiguousarray(sub)).to(COMPLEX)
        elif sub.dtype == np.bool_:
            out[k] = torch.from_numpy(np.ascontiguousarray(sub))
        elif np.issubdtype(sub.dtype, np.integer):
            out[k] = torch.from_numpy(np.ascontiguousarray(sub)).to(torch.long)
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(sub)).to(REAL)
    return out


def eval_dataset_loss(model: torch.nn.Module, data: dict[str, np.ndarray], batch_size: int = 32) -> dict[str, float]:
    n = len(data["tau_hat"])
    tot_sum = mse_sum = bce_sum = prb_sum = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            bt = _batch_to_tensors(data, np.arange(start, end))
            df, dg, dlv = model.forward_tensors(
                bt["scalar_feats"],
                bt["complex_feats"],
                bt["tau_hat"],
                bt["nu_post"],
                bt["t_phys"],
                bt["t_norm"],
                bt["ev_base"],
            )
            tot, l_mse, l_bce, l_prb = stage5_differentiable_loss(
                df,
                dg,
                dlv,
                t_rows_idx=bt["t_rows_idx"],
                probe_row_idx=bt["probe_row_idx"],
                rot_intra_t=bt["rot_intra_t"],
                rot_intra_p=bt["rot_intra_p"],
                fam_logw=bt["fam_logw"],
                fam_active=bt["fam_active"],
                fam_mean_t=bt["fam_mean_t"],
                fam_var_t=bt["fam_var_t"],
                fam_mean_p=bt["fam_mean_p"],
                fam_var_p=bt["fam_var_p"],
                d_mats=bt["d_mats"],
                phi_f=bt["phi_f"],
                h_true_u=bt["h_true_u"],
                y_target_freq_u=bt["y_target_freq_u"],
                b_code_target=bt["b_code_target"],
                u_probe=bt["u_probe"],
                probe_clean_u=bt["probe_clean_u"],
            )
            w = float(end - start)
            tot_sum += float(tot.item()) * w
            mse_sum += float(l_mse.item()) * w
            bce_sum += float(l_bce.item()) * w
            prb_sum += float(l_prb.item()) * w
    inv = 1.0 / float(max(n, 1))
    return {"total": tot_sum * inv, "mse": mse_sum * inv, "bce": bce_sum * inv, "probe": prb_sum * inv}


def calibrate_output_scales_on_dev(model: torch.nn.Module, dev_data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Deterministic dev-only calibration of `(fam_scale, res_scale)` minimizing dev composite loss."""
    grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    best_loss = float("inf")
    best_pair = (1.0, 1.0)
    best_metrics: dict[str, float] = {}
    for fs in grid:
        for rs in grid:
            with torch.no_grad():
                model.fam_scale.fill_(float(fs))
                model.res_scale.fill_(float(rs))
            m = eval_dataset_loss(model, dev_data)
            if m["total"] < best_loss:
                best_loss = m["total"]
                best_pair = (float(fs), float(rs))
                best_metrics = m
    with torch.no_grad():
        model.fam_scale.fill_(best_pair[0])
        model.res_scale.fill_(best_pair[1])
    return {"fam_scale": best_pair[0], "res_scale": best_pair[1], "dev_metrics": best_metrics}


def train_one_seed(
    role: str,
    seed: int,
    train_data: dict[str, np.ndarray],
    dev_data: dict[str, np.ndarray],
    *,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 2.5e-3,
    weight_decay: float = 2e-4,
    ckpt_dir: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    model_a, model_b = build_matched_stage5_models()
    model: torch.nn.Module = model_a if role == "role_a" else model_b
    n_params = int(model.real_parameters())

    step0_dev = eval_dataset_loss(model, dev_data)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    n_train = len(train_data["tau_hat"])
    best_dev_total = step0_dev["total"]
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "dev": step0_dev}]

    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        tr_tot = tr_mse = tr_bce = tr_prb = 0.0
        for start in range(0, n_train, batch_size):
            idxs = perm[start : start + batch_size]
            bt = _batch_to_tensors(train_data, idxs)
            opt.zero_grad()
            df, dg, dlv = model.forward_tensors(
                bt["scalar_feats"],
                bt["complex_feats"],
                bt["tau_hat"],
                bt["nu_post"],
                bt["t_phys"],
                bt["t_norm"],
                bt["ev_base"],
            )
            tot, l_mse, l_bce, l_prb = stage5_differentiable_loss(
                df,
                dg,
                dlv,
                t_rows_idx=bt["t_rows_idx"],
                probe_row_idx=bt["probe_row_idx"],
                rot_intra_t=bt["rot_intra_t"],
                rot_intra_p=bt["rot_intra_p"],
                fam_logw=bt["fam_logw"],
                fam_active=bt["fam_active"],
                fam_mean_t=bt["fam_mean_t"],
                fam_var_t=bt["fam_var_t"],
                fam_mean_p=bt["fam_mean_p"],
                fam_var_p=bt["fam_var_p"],
                d_mats=bt["d_mats"],
                phi_f=bt["phi_f"],
                h_true_u=bt["h_true_u"],
                y_target_freq_u=bt["y_target_freq_u"],
                b_code_target=bt["b_code_target"],
                u_probe=bt["u_probe"],
                probe_clean_u=bt["probe_clean_u"],
            )
            tot.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            opt.step()
            w = float(len(idxs))
            tr_tot += float(tot.item()) * w
            tr_mse += float(l_mse.item()) * w
            tr_bce += float(l_bce.item()) * w
            tr_prb += float(l_prb.item()) * w
        sched.step()

        dev_m = eval_dataset_loss(model, dev_data)
        tr_m = {
            "total": tr_tot / float(n_train),
            "mse": tr_mse / float(n_train),
            "bce": tr_bce / float(n_train),
            "probe": tr_prb / float(n_train),
        }
        history.append({"epoch": ep, "train": tr_m, "dev": dev_m})
        if dev_m["total"] < best_dev_total:
            best_dev_total = dev_m["total"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep in (1, 5, 10, 20, epochs) or ep == best_epoch:
            print(
                f"  [{role} s={seed} ep={ep:02d}/{epochs}] tr={tr_m['total']:.4f} (mse={tr_m['mse']:.4f}, bce={tr_m['bce']:.4f}) "
                f"| dev={dev_m['total']:.4f} (mse={dev_m['mse']:.4f}, bce={dev_m['bce']:.4f}, prb={dev_m['probe']:.4f}) best_ep={best_epoch}",
                flush=True,
            )

    model.load_state_dict(best_state)
    cal = calibrate_output_scales_on_dev(model, dev_data)
    print(
        f"  [{role} s={seed} CAL] fam_scale={cal['fam_scale']:.2f}, res_scale={cal['res_scale']:.2f} "
        f"-> dev_total={cal['dev_metrics']['total']:.4f} (step0={step0_dev['total']:.4f}, gain={step0_dev['total']-cal['dev_metrics']['total']:+.4f})",
        flush=True,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"stage5_{role}_seed{seed}.pt"
    payload = {
        "role": role,
        "seed": int(seed),
        "cfg": model.cfg,
        "real_parameters": n_params,
        "best_epoch": int(best_epoch),
        "step0_dev": step0_dev,
        "calibrated_dev": cal["dev_metrics"],
        "fam_scale": cal["fam_scale"],
        "res_scale": cal["res_scale"],
        "state_dict": model.state_dict(),
    }
    torch.save(payload, ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    return {
        "role": role,
        "seed": int(seed),
        "real_parameters": n_params,
        "cfg": model.cfg,
        "best_epoch": int(best_epoch),
        "step0_dev": step0_dev,
        "uncalibrated_best_dev_total": float(best_dev_total),
        "calibration": cal,
        "ckpt_file": ckpt_path.name,
        "ckpt_bytes": int(ckpt_path.stat().st_size),
        "ckpt_sha256": ckpt_sha,
        "train_wall_s": float(time.perf_counter() - t0),
        "history": history,
    }


def load_stage5_checkpoint(path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    role = payload["role"]
    cfg = payload["cfg"]
    if role == "role_a":
        model: torch.nn.Module = SymmetricTransportRefiner(**cfg)
    elif role == "role_b":
        model = NonPhysicsTransformerRefiner(**cfg)
    else:
        raise ValueError(f"Unknown checkpoint role {role!r}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def eval_stage5_on_dev_links(
    dev_tasks: list[tuple[str, int, int]],
    models_a: list[torch.nn.Module],
    models_b: list[torch.nn.Module],
    *,
    content_cfg: Optional[ContentAddressingConfig],
    mix_cfg: MixtureDecodeConfig,
) -> dict[str, Any]:
    """Quick end-to-end BLER/NMSE verification on the dev links (strictly dev-only)."""
    t0 = time.perf_counter()
    arms = ("phys_upgraded", "role_a_ens", "role_b_ens")
    agg: dict[str, dict[str, dict[str, float]]] = {
        reg: {a: {"be": 0, "bb": 0, "frob": 0.0, "pnmse": 0.0, "n": 0} for a in arms} for reg in TRAIN_REGIMES
    }
    for idx, (reg, _l_i, seed) in enumerate(dev_tasks):
        obs, truth, _ = simulate_offgrid_link_p12(seed, reg, snr_db=PRIMARY_SNR_DB, pilot_rows_a=OFDM_PILOT_COMB)
        folds = run_consistent_folds(obs, content_cfg=content_cfg, wave_mix_cfg=mix_cfg)

        ov_a = {snap.fold_idx: infer_stage5_fold_override(models_a, snap) for snap in folds.folds}
        ov_b = {snap.fold_idx: infer_stage5_fold_override(models_b, snap) for snap in folds.folds}

        res_map = {
            "phys_upgraded": evaluate_decoder_mode_on_folds(obs, truth, folds, mode="mixture_llr", mix_cfg=mix_cfg),
            "role_a_ens": evaluate_decoder_mode_on_folds(obs, truth, folds, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ov_a),
            "role_b_ens": evaluate_decoder_mode_on_folds(obs, truth, folds, mode="mixture_llr", mix_cfg=mix_cfg, neural_overrides=ov_b),
        }
        for a, sc in res_map.items():
            d = agg[reg][a]
            d["be"] += int(sc["block_errors"])
            d["bb"] += int(sc["blocks"])
            d["frob"] += float(sc["mean_rel_frob_err"])
            d["pnmse"] += float(sc["probe_clean_nmse"])
            d["n"] += 1
        if (idx + 1) % 20 == 0 or (idx + 1) == len(dev_tasks):
            print(f"[Step2-DevEval] {idx+1}/{len(dev_tasks)} links in {time.perf_counter()-t0:.1f}s", flush=True)

    summary: dict[str, Any] = {"by_regime": {}, "pooled_4id": {}}
    for reg in TRAIN_REGIMES:
        summary["by_regime"][reg] = {
            a: {
                "bler_pct": 100.0 * agg[reg][a]["be"] / max(agg[reg][a]["bb"], 1),
                "mean_rel_frob": agg[reg][a]["frob"] / max(agg[reg][a]["n"], 1),
                "probe_clean_nmse": agg[reg][a]["pnmse"] / max(agg[reg][a]["n"], 1),
            }
            for a in arms
        }
    for a in arms:
        be4 = sum(agg[r][a]["be"] for r in TRAIN_REGIMES[:4])
        bb4 = sum(agg[r][a]["bb"] for r in TRAIN_REGIMES[:4])
        n4 = sum(agg[r][a]["n"] for r in TRAIN_REGIMES[:4])
        summary["pooled_4id"][a] = {
            "bler_pct": 100.0 * be4 / max(bb4, 1),
            "mean_rel_frob": sum(agg[r][a]["frob"] for r in TRAIN_REGIMES[:4]) / max(n4, 1),
            "probe_clean_nmse": sum(agg[r][a]["pnmse"] for r in TRAIN_REGIMES[:4]) / max(n4, 1),
        }
    summary["wall_s"] = float(time.perf_counter() - t0)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI-4 Step 2: Stage 5 Fair Training (Role A vs Role B)")
    parser.add_argument("--physics-variant", type=str, default="mixture_llr_content")
    parser.add_argument("--train-links-per-regime", type=int, default=60)
    parser.add_argument("--dev-links-per-regime", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--out-dir", type=str, default="research/p12_stage5_fusion_neural/results")
    parser.add_argument("--ckpt-dir", type=str, default="research/p12_stage5_fusion_neural/ckpts")
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args()

    t_all = time.perf_counter()
    enforce_determinism(require_env=args.require_env)
    warmup_p12_offgrid_numba()

    content_cfg, mix_cfg = get_physics_configs(args.physics_variant)
    train_tasks = build_split_tasks("train", args.train_links_per_regime)
    dev_tasks = build_split_tasks("dev", args.dev_links_per_regime)

    print(
        f"=== STEP 2 STAGE 5 TRAINING: variant={args.physics_variant}, "
        f"train={len(train_tasks)} links ({len(train_tasks)*3} folds), dev={len(dev_tasks)} links ({len(dev_tasks)*3} folds) ===",
        flush=True,
    )
    train_data = generate_fold_dataset(train_tasks, content_cfg=content_cfg, mix_cfg=mix_cfg, label="Train")
    dev_data = generate_fold_dataset(dev_tasks, content_cfg=content_cfg, mix_cfg=mix_cfg, label="Dev")

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict[str, Any]]] = {"role_a": [], "role_b": []}
    for role in ("role_a", "role_b"):
        for s in STAGE5_SEEDS:
            print(f"\n--- Training {role} seed={s} ---", flush=True)
            info = train_one_seed(
                role,
                s,
                train_data,
                dev_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                ckpt_dir=ckpt_dir,
            )
            runs[role].append(info)

    # Load all 6 frozen checkpoints and run dev BLER sanity evaluation
    models_a = [load_stage5_checkpoint(ckpt_dir / f"stage5_role_a_seed{s}.pt") for s in STAGE5_SEEDS]
    models_b = [load_stage5_checkpoint(ckpt_dir / f"stage5_role_b_seed{s}.pt") for s in STAGE5_SEEDS]
    dev_eval = eval_stage5_on_dev_links(dev_tasks, models_a, models_b, content_cfg=content_cfg, mix_cfg=mix_cfg)

    report = {
        "physics_variant": args.physics_variant,
        "train_links_per_regime": int(args.train_links_per_regime),
        "dev_links_per_regime": int(args.dev_links_per_regime),
        "train_folds": int(len(train_data["tau_hat"])),
        "dev_folds": int(len(dev_data["tau_hat"])),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seeds": list(STAGE5_SEEDS),
        "real_parameters_role_a": runs["role_a"][0]["real_parameters"],
        "real_parameters_role_b": runs["role_b"][0]["real_parameters"],
        "runs": runs,
        "dev_e2e_eval": dev_eval,
        "total_wall_s": float(time.perf_counter() - t_all),
    }
    (out_dir / "step2_training_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n=== STEP 2 DEV END-TO-END BLER SUMMARY ===", flush=True)
    print(json.dumps(dev_eval, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
