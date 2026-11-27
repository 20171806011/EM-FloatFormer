"""Modal A100-80GB execution harness for EM-FloatFormer Stage 4 (smoke_probe + pilot_gate1)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal

_FILE_P = Path(__file__).resolve()
REPO_ROOT = _FILE_P.parents[2] if len(_FILE_P.parents) >= 3 else Path("/workspace/EM-FloatFormer")
STAGE4_LOCAL = REPO_ROOT / "research" / "stage4_real_and_surgical"
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FREE_CREDIT_LIMIT_USD = 30.00
ABORT_MONTHLY_TOTAL_USD = 28.50
MIN_REMAINING_USD = 1.50

app = modal.App("em-floatformer-stage4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=5.24.0",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "numpy==2.3.5",
        "scipy==1.17.0",
        "flash-linear-attention",
        "pytest",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace/EM-FloatFormer",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "research/stage3d_qwen_signed_phase/checkpoints",
            "research/stage3d_qwen_signed_phase/results",
            "research/stage4_math_and_design/discussions",
            "research/stage4_real_and_surgical/results",
            "research/stage4_real_and_surgical/neuron_surgery",
        ],
    )
)

hf_vol = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("stage4-results", create_if_missing=True)
REMOTE_REPO = Path("/workspace/EM-FloatFormer")
REMOTE_OUT = Path("/vol/stage4")

sdr_cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.3.5", "scipy==1.17.0", "h5py", "torch")
    .env({"PYTHONUTF8": "1"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace/EM-FloatFormer",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "research/stage3d_qwen_signed_phase/checkpoints",
            "research/stage3d_qwen_signed_phase/results",
            "research/stage4_math_and_design/discussions",
            "research/stage4_real_and_surgical/results",
            "research/stage4_real_and_surgical/neuron_surgery",
        ],
    )
)


def _setup_remote_env() -> None:
    os.chdir(REMOTE_REPO)
    if str(REMOTE_REPO) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPO))


def _query_monthly_billing() -> dict:
    raw = subprocess.check_output(
        ["modal", "billing", "report", "--for", "this month", "--show-resources", "--json"],
        text=True,
        encoding="utf-8",
    )
    rows = json.loads(raw)
    total = sum(float(r.get("cost") if r.get("cost") is not None else r.get("Cost", 0.0)) for r in rows)
    stage4_total = sum(
        float(r.get("cost") if r.get("cost") is not None else r.get("Cost", 0.0))
        for r in rows
        if r.get("description") == "em-floatformer-stage4"
    )
    remaining = FREE_CREDIT_LIMIT_USD - total
    return {
        "row_count": len(rows),
        "monthly_total_usd": round(total, 8),
        "stage4_total_usd": round(stage4_total, 8),
        "remaining_of_30_usd": round(remaining, 8),
        "rows": rows,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_smoke_probe() -> dict:
    _setup_remote_env()
    from research.stage4_real_and_surgical.train_eval_stage4 import run_stage4_smoke_and_probe

    report = run_stage4_smoke_and_probe(REMOTE_REPO, REMOTE_OUT)
    hf_vol.commit()
    res_vol.commit()
    return report


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_pilot_gate1(
    seed: int = 202611,
    steps: int = 69,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 1.5e-3,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    scorer_mode: str = "kv_cached",
) -> dict:
    _setup_remote_env()
    from research.stage4_real_and_surgical.train_eval_stage4 import run_gate1_pilot_69step

    out = run_gate1_pilot_69step(
        REMOTE_REPO,
        REMOTE_OUT,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        train_blocks=train_blocks,
        dev_blocks=dev_blocks,
        scorer_mode=scorer_mode,
    )
    res_vol.commit()
    return out


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=5400,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_phaseA_384(
    seed: int = 202611,
    steps: int = 384,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 1.5e-3,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    scorer_mode: str = "kv_cached",
) -> dict:
    _setup_remote_env()
    from research.stage4_real_and_surgical.train_eval_stage4 import run_gate1_pilot_69step

    out = run_gate1_pilot_69step(
        REMOTE_REPO,
        REMOTE_OUT,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        train_blocks=train_blocks,
        dev_blocks=dev_blocks,
        scorer_mode=scorer_mode,
        out_prefix=f"phaseA_{steps}step",
        on_arm_done=lambda _arm_name: res_vol.commit(),
    )
    res_vol.commit()
    return out


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_phaseB_and_blind(
    seed: int = 202611,
    steps: int = 96,
    g1_blocks: int = 32,
) -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage4_real_and_surgical.phaseB_surgical import (
        run_phaseB_surgical_and_lemmo,
        run_stage4_synthetic_blind,
    )

    kfac_p = REMOTE_OUT / "kfac_4b_top256.pt"
    ckpt_dir = REMOTE_OUT / "checkpoints"
    assert kfac_p.exists(), f"Missing {kfac_p} on stage4-results volume"
    for arm in ("zero", "naive_additive", "equivariant_host_pure", "equivariant_physical_residual"):
        req_ckpt = ckpt_dir / f"stage4_phaseA_{arm}_s{seed}_step384.pt"
        assert req_ckpt.exists(), f"Missing {req_ckpt} on stage4-results volume"

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    phaseB = run_phaseB_surgical_and_lemmo(
        REMOTE_REPO,
        REMOTE_OUT,
        ckpt_dir,
        kfac_p,
        seed=seed,
        steps=steps,
        batch_size=32,
        micro_batch_size=16,
        lr=3e-4,
        train_blocks=8,
        dev_blocks=2,
        scorer_mode="kv_cached",
        host=host,
    )
    res_vol.commit()

    blind = run_stage4_synthetic_blind(
        REMOTE_REPO,
        REMOTE_OUT,
        ckpt_dir,
        kfac_p,
        seed=seed,
        g1_blocks=g1_blocks,
        eval_bs=64,
        scorer_mode="kv_cached",
        host=host,
    )
    res_vol.commit()
    return {"phaseB": phaseB, "blind": blind}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_synthetic_blind_only(
    seed: int = 202611,
    g1_blocks: int = 131,
    eval_bs: int = 64,
) -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage4_real_and_surgical.phaseB_surgical import run_stage4_synthetic_blind

    kfac_p = REMOTE_OUT / "kfac_4b_top256.pt"
    ckpt_dir = REMOTE_OUT / "checkpoints"
    assert kfac_p.exists(), f"Missing {kfac_p} on stage4-results volume"
    for arm in ("zero", "naive_additive", "equivariant_host_pure", "equivariant_physical_residual"):
        req_ckpt = ckpt_dir / f"stage4_phaseA_{arm}_s{seed}_step384.pt"
        assert req_ckpt.exists(), f"Missing {req_ckpt} on stage4-results volume"
    for pb_name in (f"stage4_phaseB_surgical96_s{seed}.pt", f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"):
        req_pb = ckpt_dir / pb_name
        assert req_pb.exists(), f"Missing {req_pb} on stage4-results volume"

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    blind = run_stage4_synthetic_blind(
        REMOTE_REPO,
        REMOTE_OUT,
        ckpt_dir,
        kfac_p,
        seed=seed,
        g1_blocks=g1_blocks,
        eval_bs=eval_bs,
        scorer_mode="kv_cached",
        host=host,
        cache_dir=REMOTE_OUT / "blind_cache",
        on_model_done=lambda _m: res_vol.commit(),
    )
    res_vol.commit()
    return blind


@app.function(
    image=sdr_cpu_image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    retries=0,
    volumes={"/vol/stage4": res_vol},
)
def remote_real_sdr_cpu_extract() -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage4_real_and_surgical.sdr_real_pipeline import run_real_sdr_cpu_pipeline

    rep = run_real_sdr_cpu_pipeline(
        REMOTE_REPO,
        Path("/tmp/real_sdr_raw"),
        REMOTE_OUT / "real_sdr_cache",
        mock_synthetic_for_test=False,
    )
    res_vol.commit()
    return rep


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2400,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_real_sdr_and_stage6_4b(seed: int = 202611) -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage4_real_and_surgical.real_sdr_and_stage6_gpu import (
        run_stage4_real_sdr_gpu,
        run_stage6_active_eig_gpu,
    )

    sdr_cache = REMOTE_OUT / "real_sdr_cache"
    assert (sdr_cache / "pluto_dual_cache_c16.npz").exists(), "Missing pluto_dual_cache_c16.npz; run real_sdr_cpu first"
    assert (sdr_cache / "cttc_dual_cache_c16.npz").exists(), "Missing cttc_dual_cache_c16.npz; run real_sdr_cpu first"
    kfac_p = REMOTE_OUT / "kfac_4b_top256.pt"
    ckpt_dir = REMOTE_OUT / "checkpoints"
    assert kfac_p.exists(), f"Missing {kfac_p}"
    for arm in ("zero", "naive_additive", "equivariant_host_pure", "equivariant_physical_residual"):
        assert (ckpt_dir / f"stage4_phaseA_{arm}_s{seed}_step384.pt").exists()
    for pb_name in (f"stage4_phaseB_surgical96_s{seed}.pt", f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"):
        assert (ckpt_dir / pb_name).exists(), f"Missing {pb_name}"

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    sdr_rep = run_stage4_real_sdr_gpu(
        REMOTE_REPO,
        sdr_cache,
        REMOTE_OUT,
        ckpt_dir,
        kfac_p,
        seed=seed,
        eval_bs=128,
        host=host,
        cache_dir=REMOTE_OUT / "real_sdr_4b_cache",
        on_substage_done=lambda _k: res_vol.commit(),
    )
    res_vol.commit()

    s6_rep = run_stage6_active_eig_gpu(
        REMOTE_REPO,
        REMOTE_OUT,
        ckpt_dir,
        kfac_p,
        seed=seed,
        eval_bs=128,
        host=host,
        cache_dir=REMOTE_OUT / "stage6_4b_cache",
        on_model_done=lambda _k: res_vol.commit(),
    )
    res_vol.commit()
    return {"real_sdr_4b": sdr_rep, "stage6_4b": s6_rep}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2400,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage4": res_vol},
)
def remote_cn_sft_4b_probe(
    seed: int = 202611,
    steps: int = 96,
    g1_blocks: int = 131,
) -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage4_real_and_surgical.cn_sft_probe import run_cn_sft_4b_probe

    ckpt_dir = REMOTE_OUT / "checkpoints"
    assert (ckpt_dir / f"stage4_phaseA_equivariant_host_pure_s{seed}_step384.pt").exists(), (
        f"Missing Phase-A pure checkpoint for seed={seed}"
    )
    rep = run_cn_sft_4b_probe(
        REMOTE_REPO,
        REMOTE_OUT,
        ckpt_dir,
        seed=seed,
        steps=steps,
        g1_blocks=g1_blocks,
        eval_bs=64,
        scorer_mode="kv_cached",
        on_step0_done=res_vol.commit,
    )
    res_vol.commit()
    return rep


@app.local_entrypoint()
def main(
    mode: str = "smoke_probe",
    seed: int = 202611,
    steps: int = 69,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 1.5e-3,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    scorer_mode: str = "kv_cached",
    g1_blocks: int = 131,
) -> None:
    results_dir = STAGE4_LOCAL / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pre = _query_monthly_billing()
    print(
        json.dumps(
            {
                "preflight_billing": {
                    "monthly_total_usd": pre["monthly_total_usd"],
                    "stage4_total_usd": pre["stage4_total_usd"],
                    "remaining_of_30_usd": pre["remaining_of_30_usd"],
                }
            }
        ),
        flush=True,
    )
    if pre["monthly_total_usd"] >= ABORT_MONTHLY_TOTAL_USD or pre["remaining_of_30_usd"] < MIN_REMAINING_USD:
        raise RuntimeError(
            f"Billing pre-flight abort: monthly_total=${pre['monthly_total_usd']:.4f}, "
            f"remaining=${pre['remaining_of_30_usd']:.4f}"
        )

    if mode == "smoke_probe":
        rep = remote_smoke_probe.remote()
        (results_dir / "smoke_probe_4b.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
    elif mode == "pilot_gate1":
        out = remote_pilot_gate1.remote(
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            lr=lr,
            train_blocks=train_blocks,
            dev_blocks=dev_blocks,
            scorer_mode=scorer_mode,
        )
        (results_dir / "pilot_69step_4arms.json").write_text(
            json.dumps(out["pilot"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (results_dir / "two_probe_voi_readout.json").write_text(
            json.dumps(out["voi"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(out["voi"], ensure_ascii=False, indent=2), flush=True)
    elif mode == "phaseA_384":
        out = remote_phaseA_384.remote(
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            lr=lr,
            train_blocks=train_blocks,
            dev_blocks=dev_blocks,
            scorer_mode=scorer_mode,
        )
        (results_dir / f"phaseA_{steps}step_4arms.json").write_text(
            json.dumps(out["pilot"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (results_dir / f"two_probe_voi_readout_{steps}.json").write_text(
            json.dumps(out["voi"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(out["voi"], ensure_ascii=False, indent=2), flush=True)
    elif mode == "phaseB_and_blind":
        out = remote_phaseB_and_blind.remote(seed=seed, steps=steps, g1_blocks=32)
        (results_dir / "phaseB_surgical_vs_lemmo.json").write_text(
            json.dumps(out["phaseB"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (results_dir / "stage4_synthetic_blind_results.json").write_text(
            json.dumps(out["blind"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "surgical_96": out["phaseB"]["surgical_96"],
                    "lemmo_tier_p1": out["phaseB"]["lemmo_tier_p1"],
                    "m2_paired_stats": out["blind"]["m2_paired_stats"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif mode == "blind_only":
        blind = remote_synthetic_blind_only.remote(seed=seed, g1_blocks=g1_blocks, eval_bs=64)
        (results_dir / "stage4_synthetic_blind_results.json").write_text(
            json.dumps(blind, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "eval_wall_s": blind["eval_wall_s"],
                    "g1_blocks": blind["g1_blocks"],
                    "m2_paired_stats": blind["m2_paired_stats"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif mode == "real_sdr_cpu":
        rep = remote_real_sdr_cpu_extract.remote()
        (results_dir / "real_sdr_cpu_physical_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "wall_s": rep["wall_s"],
                    "sentinel_zero_leakage": rep["sentinel_zero_leakage"],
                    "counts": rep["counts"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif mode == "cn_sft_4b_probe":
        eff_steps = 96 if steps == 69 else steps
        rep = remote_cn_sft_4b_probe.remote(seed=seed, steps=eff_steps, g1_blocks=g1_blocks)
        (results_dir / "cn_sft_4b_probe_results.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "regime": rep.get("regime"),
                    "early_aborted": rep.get("early_aborted"),
                    "wall_s": rep.get("wall_s"),
                    "step0": rep.get("step0"),
                    "U0_audit": rep.get("U0", {}).get("audit"),
                    "U0_blind_G1": rep.get("U0", {}).get("blind", {}).get("G1_main_extrap_both"),
                    "U1_audit": rep.get("U1", {}).get("audit"),
                    "U1_blind_G1": rep.get("U1", {}).get("blind", {}).get("G1_main_extrap_both"),
                    "U1_blind_G3": rep.get("U1", {}).get("blind", {}).get("G3_extrap_angle"),
                    "U1_Q2": rep.get("U1", {}).get("blind", {}).get("Q2_C4_blind"),
                    "U3_gate_off_audit": rep.get("U3_gate_off_audit"),
                    "U2_audit": rep.get("U2", {}).get("audit"),
                    "U2_blind_G1": rep.get("U2", {}).get("blind", {}).get("G1_main_extrap_both"),
                    "hard_pass_gates": rep.get("hard_pass_gates"),
                    "m2_paired_stats": rep.get("m2_paired_stats"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif mode == "real_sdr_and_stage6_4b":
        out = remote_real_sdr_and_stage6_4b.remote(seed=seed)
        (results_dir / "stage4_real_sdr_4b_results.json").write_text(
            json.dumps(out["real_sdr_4b"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (results_dir / "stage6_active_eig_4b_results.json").write_text(
            json.dumps(out["stage6_4b"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "real_sdr_wall_s": out["real_sdr_4b"]["wall_s"],
                    "holm_bonferroni_real_sdr": out["real_sdr_4b"]["holm_bonferroni_real_sdr"],
                    "stage6_wall_s": out["stage6_4b"]["wall_s"],
                    "stage6_16bit_vs_4bit": out["stage6_4b"]["paired_16bit_vs_4bit_frozen_host"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    else:
        raise ValueError(
            f"Unknown mode={mode!r}; expected 'smoke_probe', 'pilot_gate1', 'phaseA_384', "
            f"'phaseB_and_blind', 'blind_only', 'real_sdr_cpu', 'real_sdr_and_stage6_4b', or 'cn_sft_4b_probe'"
        )

    post = _query_monthly_billing()
    print(
        json.dumps(
            {
                "post_run_billing": {
                    "monthly_total_usd": post["monthly_total_usd"],
                    "stage4_total_usd": post["stage4_total_usd"],
                    "delta_usd": round(post["monthly_total_usd"] - pre["monthly_total_usd"], 8),
                    "remaining_of_30_usd": post["remaining_of_30_usd"],
                }
            }
        ),
        flush=True,
    )
