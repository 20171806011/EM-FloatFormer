"""Modal GPU execution harness for EM-FloatFormer Stage 3D (Qwen3.5-4B + optional historical Ministral)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal


_FILE_P = Path(__file__).resolve()
REPO_ROOT = _FILE_P.parents[2] if len(_FILE_P.parents) >= 3 else Path("/workspace/EM-FloatFormer")
STAGE3D_LOCAL = REPO_ROOT / "research" / "stage3d_qwen_signed_phase"
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = modal.App("em-floatformer-stage3d")

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
        ],
    )
)

hf_vol = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("stage3d-results", create_if_missing=True)
REMOTE_REPO = Path("/workspace/EM-FloatFormer")
REMOTE_OUT = Path("/vol/stage3d")


def _setup_remote_env():
    os.chdir(REMOTE_REPO)
    if str(REMOTE_REPO) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPO))


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_smoke_qwen() -> dict:
    _setup_remote_env()
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import run_smoke_audit

    out_dir = REMOTE_OUT / "results"
    report = run_smoke_audit(REMOTE_REPO, out_dir)
    hf_vol.commit()
    res_vol.commit()
    return report


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_train_one(
    arm: str,
    seed: int,
    epochs: int = 10,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    lr: float = 1.5e-3,
    max_steps_per_epoch: int = 200,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    train_scorer: str = "kv_cached",
    eval_scorer: str = "kv_cached",
) -> dict:
    _setup_remote_env()
    from research.stage3d_qwen_signed_phase.data_stage3d import load_s17_encoder
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import train_single_arm_seed

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    encoder, _ = load_s17_encoder(REMOTE_REPO)
    summary = train_single_arm_seed(
        host=host,
        encoder=encoder,
        arm=arm,
        seed=seed,
        out_dir=REMOTE_OUT,
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        max_steps_per_epoch=max_steps_per_epoch,
        train_blocks=train_blocks,
        dev_blocks=dev_blocks,
        train_scorer=train_scorer,
        eval_scorer=eval_scorer,
    )
    res_vol.commit()
    ckpt_bytes = (REMOTE_OUT / "checkpoints" / summary["checkpoint"]).read_bytes()
    return {"summary": summary, "ckpt_bytes": ckpt_bytes}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_eval_dev_pilot(batch_size: int = 64, scorer_mode: str = "kv_cached") -> bytes:
    _setup_remote_env()
    res_vol.reload()
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import eval_dev_pilot_tables

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    out_path = eval_dev_pilot_tables(
        host=host,
        ckpt_dir=REMOTE_OUT / "checkpoints",
        out_dir=REMOTE_OUT / "results",
        batch_size=batch_size,
        scorer_mode=scorer_mode,
    )
    res_vol.commit()
    return out_path.read_bytes()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_eval_blind(batch_size: int = 64, scorer_mode: str = "kv_cached") -> dict:
    _setup_remote_env()
    res_vol.reload()
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
    )
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import eval_blind_gpu

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    out_dir = REMOTE_OUT / "results"
    meta = eval_blind_gpu(
        repo_root=REMOTE_REPO,
        host=host,
        ckpt_dir=REMOTE_OUT / "checkpoints",
        out_dir=out_dir,
        batch_size=batch_size,
        scorer_mode=scorer_mode,
    )
    res_vol.commit()
    return {
        "meta": meta,
        "parity_bytes": (out_dir / "direct_vs_table_parity.json").read_bytes(),
        "q3_bytes": (out_dir / "raw_q3_gpu.npz").read_bytes(),
        "q2_bytes": (out_dir / "raw_q2_gpu.npz").read_bytes(),
        "replay_bytes": (out_dir / "raw_replay_gpu.npz").read_bytes(),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_eval_historical_ministral(mothers: int = 128) -> dict:
    """Optional post-Qwen historical Ministral-3-3B diagnostic: Stage 3B E prompt replay + Stage 3C F +-0.35 probe."""
    _setup_remote_env()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from research.stage3_ministral_cache.eval_direct import load_frozen_models
    from research.stage3_ministral_cache.ministral_connector import MinistralCacheHost, WaveformCacheConnector
    from research.stage3_ministral_cache.schema import build_query_prompt
    from research.stage3d_qwen_signed_phase.data_stage3d import EQUIVARIANT_B8_S17_SHA256, condition_spec_from_params

    out_dir = REMOTE_OUT / "results" / "historical_ministral"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage 3B E replay_choices.py (using un-modified Stage 3 checkpoint & SHA)
    # Note: replay_choices checks `git diff --quiet BASE`, so on Modal where `.git` isn't uploaded,
    # run the exact replay logic directly while verifying all SHA-256 locks!
    s3_dir = REMOTE_REPO / "research" / "stage3_ministral_cache"
    lock = json.loads((s3_dir / "results" / "host" / "host_dev_lock.json").read_text(encoding="utf-8"))
    ck = lock["encoders"]["equivariant_b8_s17"]
    weights_path = s3_dir / "connector_weights" / "connector_equivariant_b8_s17.pt"
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == ck["checkpoint_sha256"]

    local_snap = snapshot_download(lock["model_id"], revision=lock["hf_sha"])
    host = MinistralCacheHost(model_id=local_snap, dtype=torch.bfloat16)
    assert host.host_sha256 == lock["host_sha256"]
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    connector = WaveformCacheConnector(**state["connector_config"]).to(host.device).eval()
    connector.load_state_dict(state["state_dict"])
    encoder = load_frozen_models()["equivariant_b8_s17"]
    temp = float(ck["locked_temp"])

    # Run Stage 3B E prompt replay logic cleanly
    from research.stage3b_math.E import replay_choices as rc
    from research.stage3_ministral_cache import chat_demo as s3_cd

    rows_e = []
    with torch.no_grad():
        for cname in rc.CONDITIONS:
            batch = s3_cd.generate_mothers("s3_test", cname, max_blocks=lock["host_test_blocks_locked"])
            z1, z2 = encoder.encode(batch.y1), encoder.encode(batch.y2)
            refs = batch.references_for_arm("valid")
            ref_features = [torch.as_tensor(s3_cd.build_ref_feature(ref)) for ref in refs]
            q1 = build_query_prompt("Q1", batch.condition)
            q3 = build_query_prompt("Q3", batch.condition, protocol="repeat_x1")
            q2 = build_query_prompt("Q2", batch.condition, ref_arm="valid", references=refs)
            for idx in range(mothers):
                codes = [torch.tensor([int(z[idx])], dtype=torch.long) for z in (z1, z2)]
                for task, (question, choices), features, truth in (
                    ("Q1", q1, [], (batch.x2[idx] - batch.x1[idx]) % 4),
                    ("Q3_repeat", q3, [], batch.k3_repeat[idx]),
                    ("Q2_valid", q2, [x[idx : idx + 1] for x in ref_features], batch.x1[idx]),
                ):
                    row = {
                        "condition": cname,
                        "mother_id": batch.mother_ids[idx],
                        "task": task,
                        "choices": list(choices),
                        "true_index": int(truth),
                    }
                    variants = {"original": question, **rc.prompts(question, choices)}
                    for arm, prompt in variants.items():
                        prefix = host.build_prefix_embeds(connector, prompt, *codes, ref_feats_list=features)
                        log_probs, _ = host.score_candidates_kv(prefix, choices, temperature=temp)
                        score = int(log_probs[0].argmax().item())
                        if arm == "original":
                            row["original_score"] = score
                            continue
                        reply = host.generate_text(prefix, max_new_tokens=96)
                        row[arm] = {
                            "score_index": score,
                            "reply": reply,
                            "generated_index": rc.parse_choice(reply, choices),
                        }
                    rows_e.append(row)

    e_payload = {
        "source_commit": rc.BASE,
        "hf_revision": lock["hf_sha"],
        "host_fingerprint": host.host_sha256,
        "connector_sha256": ck["checkpoint_sha256"],
        "temperature": temp,
        "mothers_per_condition": mothers,
        "rotations": [[0, 0]],
        "summary": rc.summarize(rows_e),
        "rows": rows_e,
    }
    e_path = out_dir / "ministral_prompt_replay.json"
    e_path.write_text(json.dumps(e_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2. Work F +-0.35 256x2 table on frozen Ministral
    probs_f = np.zeros((2, 256, 4), dtype=np.float64)
    z1_all = np.arange(256, dtype=np.int64) // 16
    z2_all = np.arange(256, dtype=np.int64) % 16
    with torch.no_grad():
        for sign_i, sign in enumerate((-1, 1)):
            cond = condition_spec_from_params(sign * 0.35, 2, 2, 4.0, 2.0, 0.10, tag="C4_cfo_ramp")
            prompt, choices = build_query_prompt("Q3", cond, protocol="repeat_x1")
            chunks = []
            for st in range(0, 256, 64):
                ed = min(256, st + 64)
                pref = host.build_prefix_embeds(
                    connector,
                    prompt,
                    torch.tensor(z1_all[st:ed], dtype=torch.long),
                    torch.tensor(z2_all[st:ed], dtype=torch.long),
                    [],
                )
                lp, _ = host.score_candidates_kv(pref, choices, temperature=temp)
                chunks.append(lp.exp().detach().cpu().numpy().astype(np.float64))
            probs_f[sign_i] = np.concatenate(chunks, axis=0)

    f_host_path = out_dir / "ministral_work_f_host_035.json"
    f_host_path.write_text(
        json.dumps(
            {
                "task": "Q3_repeat",
                "encoder_sha": EQUIVARIANT_B8_S17_SHA256,
                "rate_magnitudes": [0.35],
                "h": 2,
                "ell": 2,
                "sigma_phi": 0.1,
                "snr_db": 2.0,
                "sign_order": [-1, 1],
                "pilot": "none",
                "cache_order": "16*z1+z2",
                "probabilities": probs_f.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    f_res_bytes = subprocess.check_output(
        [sys.executable, "-B", str(REMOTE_REPO / "research/stage3c/F/check_f.py"), "--repo", str(REMOTE_REPO), "--host", str(f_host_path)],
        cwd=REMOTE_REPO,
    )
    (out_dir / "ministral_work_f_results.json").write_bytes(f_res_bytes)
    hf_vol.commit()
    res_vol.commit()
    return {
        "e_bytes": e_path.read_bytes(),
        "f_host_bytes": f_host_path.read_bytes(),
        "f_res_bytes": f_res_bytes,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1200,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_bench_train() -> dict:
    """Measure post-fix multi-microbatch VRAM stability, worst-case Q3_after_pilot/Q2 speed, dev-eval VRAM, and scorer parity."""
    _setup_remote_env()
    import gc
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from research.stage3d_qwen_signed_phase.data_stage3d import (
        NS_BLIND_G1,
        NS_CFO_DEV,
        build_stage3d_prompt,
        condition_spec_from_params,
        generate_paired_cfo_blocks,
        load_s17_encoder,
        public_phase_slots,
    )
    from research.stage3d_qwen_signed_phase.qwen_connector import (
        MODEL_ID_DEFAULT,
        PINNED_REVISION_DEFAULT,
        QwenCacheHost,
        SignedPhaseCacheConnector,
    )
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import (
        _append_batch_groups,
        _score_group_with_host,
        score_256_cache_table,
    )

    host = QwenCacheHost(model_id=MODEL_ID_DEFAULT, revision=PINNED_REVISION_DEFAULT)
    enc, _ = load_s17_encoder(REMOTE_REPO)
    torch.manual_seed(202611)
    conn = SignedPhaseCacheConnector(
        num_states=16, hidden_size=host.hidden_size, target_embed_std=host.target_embed_std
    ).to(host.device)
    opt = torch.optim.AdamW(conn.parameters(), lr=1.5e-3)

    dv_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, 0, 1, 2, 2, 4.0, (0.18,), split_prefix="bench")
    grps = []
    _append_batch_groups(grps, dv_map[0.18], enc, zero_arm=False, source_tag="cfo")
    q3_rep_grp = [g for g in grps if g["qtype"] == "Q3_repeat_x1"][0]
    q3a_grp = [g for g in grps if g["qtype"] == "Q3_after_pilot"][0]
    q2f_grp = [g for g in grps if g["qtype"] == "Q2" and g["arm"] == "fresh_resample"][0]

    res: dict = {"train_bench": {}, "dev_eval_bench": {}, "parity_bench": {}}
    host.set_grad_checkpointing(False)

    # 1. Multi-microbatch leak & peak VRAM check on Q3_after_pilot and Q2_fresh_resample (batch_size=32)
    for gname, grp in (("Q3_after_pilot", q3a_grp), ("Q2_fresh_resample", q2f_grp)):
        for mb in (8, 16):
            key = f"{gname}_kv_mb{mb}"
            try:
                conn.train()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(host.device)

                # Warmup 32-example optimizer step (includes Triton JIT compilation for this shape)
                opt.zero_grad(set_to_none=True)
                for m_st in range(0, 32, mb):
                    m_ed = m_st + mb
                    lp, _ = _score_group_with_host(host, conn, grp, m_st, m_ed, scorer_mode="kv_cached")
                    lbls = torch.tensor(grp["labels"][m_st:m_ed], device=host.device, dtype=torch.long)
                    loss = F.nll_loss(lp, lbls, reduction="sum") / 32.0
                    loss.backward()
                    del lp, loss, lbls
                opt.step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(host.device)

                # 4 consecutive microbatches to detect any reference-cycle retention across iterations
                alloc_mb_trace = []
                opt.zero_grad(set_to_none=True)
                for k in range(4):
                    m_st = k * mb
                    m_ed = m_st + mb
                    lp, _ = _score_group_with_host(host, conn, grp, m_st, m_ed, scorer_mode="kv_cached")
                    lbls = torch.tensor(grp["labels"][m_st:m_ed], device=host.device, dtype=torch.long)
                    loss = F.nll_loss(lp, lbls, reduction="sum") / float(4 * mb)
                    loss.backward()
                    del lp, loss, lbls
                    torch.cuda.synchronize()
                    alloc_mb_trace.append(round(float(torch.cuda.memory_allocated(host.device)) / (1024.0 ** 2), 1))
                opt.step()
                torch.cuda.synchronize()

                # Time 2 steady 32-example optimizer steps
                t0 = time.perf_counter()
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    for m_st in range(0, 32, mb):
                        m_ed = m_st + mb
                        lp, _ = _score_group_with_host(host, conn, grp, m_st, m_ed, scorer_mode="kv_cached")
                        lbls = torch.tensor(grp["labels"][m_st:m_ed], device=host.device, dtype=torch.long)
                        loss = F.nll_loss(lp, lbls, reduction="sum") / 32.0
                        loss.backward()
                        del lp, loss, lbls
                    opt.step()
                torch.cuda.synchronize()
                step_ms_32 = ((time.perf_counter() - t0) * 1000.0) / 2.0
                pk = float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
                gnorm = sum(float(p.grad.detach().float().norm().item()) for p in conn.parameters() if p.grad is not None)
                host_gnz = sum(1 for p in host.lm.parameters() if p.grad is not None)
                res["train_bench"][key] = {
                    "alloc_after_mb1_4_mb": alloc_mb_trace,
                    "peak_vram_mb": round(pk, 1),
                    "steady_32ex_step_ms": round(step_ms_32, 1),
                    "conn_grad_sum": round(gnorm, 3),
                    "host_grads": host_gnz,
                }
                print(f"[BENCH] {key}: {res['train_bench'][key]}", flush=True)
            except Exception as exc:
                res["train_bench"][key] = {"error": repr(exc)}
                print(f"[BENCH] {key} FAILED: {exc!r}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    # 2. Dev-eval no_grad VRAM check on Q3_after_pilot
    conn.eval()
    opt.zero_grad(set_to_none=True)
    for eval_bs in (32, 64):
        key = f"Q3_after_pilot_eval_b{eval_bs}"
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(host.device)
            with torch.no_grad():
                _score_group_with_host(host, conn, q3a_grp, 0, eval_bs, scorer_mode="kv_cached")
                torch.cuda.synchronize()
                t_e0 = time.perf_counter()
                _score_group_with_host(host, conn, q3a_grp, 0, eval_bs, scorer_mode="kv_cached")
                torch.cuda.synchronize()
            pk = float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
            res["dev_eval_bench"][key] = {
                "peak_vram_mb": round(pk, 1),
                "ms_per_ex": round(((time.perf_counter() - t_e0) * 1000.0) / eval_bs, 2),
            }
            print(f"[BENCH] {key}: {res['dev_eval_bench'][key]}", flush=True)
        except Exception as exc:
            res["dev_eval_bench"][key] = {"error": repr(exc)}
            gc.collect()
            torch.cuda.empty_cache()

    # 3. Scorer & Direct-vs-Table parity (b=1 vs naive, direct(b=64)/direct(b=16)/direct(b=1) vs table(b=64))
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for tag, grp in (("Q3_repeat_x1", q3_rep_grp), ("Q3_after_pilot", q3a_grp)):
            z1 = torch.tensor(grp["z1"][:1], dtype=torch.long)
            z2 = torch.tensor(grp["z2"][:1], dtype=torch.long)
            ps = torch.tensor(grp["phase_slots"][:1], dtype=torch.float32)
            r_list = [torch.tensor(rf[:1], dtype=torch.float32) for rf in grp["ref_feats_list"]]
            pref = host.build_prefix_embeds(conn, grp["prompt"], z1, z2, ps, r_list)
            lp_n, ll_n = host.score_candidates_naive_uncached(pref, grp["choices"])
            lp_b, ll_b = host.score_candidates_batched_uncached(pref, grp["choices"])
            lp_k, ll_k = host.score_candidates_kv(pref, grp["choices"])
            res["parity_bench"][f"b1_{tag}"] = {
                "batched_vs_naive_max_ll_diff": float((ll_b - ll_n).abs().max().item()),
                "batched_vs_naive_max_prob_diff": float((lp_b.exp() - lp_n.exp()).abs().max().item()),
                "kv_vs_naive_max_ll_diff": float((ll_k - ll_n).abs().max().item()),
                "kv_vs_naive_max_prob_diff": float((lp_k.exp() - lp_n.exp()).abs().max().item()),
            }

        tbl_64 = score_256_cache_table(
            host, conn, omega=0.40, h=2, ell=2, t_p=4.0, zero_arm=False, batch_size=64, scorer_mode="kv_cached"
        )
        g1_map, _ = generate_paired_cfo_blocks(NS_BLIND_G1, 0, 1, 2, 2, 4.0, (0.40,), split_prefix="g1_parity_bench")
        v1, v2 = g1_map[0.40].rotated_view(0, 1)
        z1_64 = np.asarray(enc.encode(v1), dtype=np.int64)[:64]
        z2_64 = np.asarray(enc.encode(v2), dtype=np.int64)[:64]
        codes_64 = 16 * z1_64 + z2_64
        cond = condition_spec_from_params(0.40, 2, 2, 4.0, 2.0, 0.10, tag="parity")
        prompt_g1, choices_g1 = build_stage3d_prompt("Q3", cond, protocol="repeat_x1")
        ps_64 = np.repeat(public_phase_slots(0.40, 2, 2, 4.0, zero_arm=False)[None, :], 64, axis=0)
        grp_64 = {"prompt": prompt_g1, "choices": choices_g1, "z1": z1_64, "z2": z2_64, "phase_slots": ps_64, "ref_feats_list": []}

        _, dir_64 = _score_group_with_host(host, conn, grp_64, 0, 64, scorer_mode="kv_cached")
        _, dir_16 = _score_group_with_host(host, conn, grp_64, 0, 16, scorer_mode="kv_cached")
        dir_1_list = []
        for idx in range(8):
            _, d1 = _score_group_with_host(host, conn, grp_64, idx, idx + 1, scorer_mode="kv_cached")
            dir_1_list.append(d1.detach().cpu().numpy().astype(np.float64)[0])
        dir_64_np = dir_64.detach().cpu().numpy().astype(np.float64)
        dir_16_np = dir_16.detach().cpu().numpy().astype(np.float64)
        dir_1_np = np.stack(dir_1_list, axis=0)

        def _cmp_tbl(direct_ll, ref_ll):
            def _sm(x):
                s = x - x.max(axis=-1, keepdims=True)
                e = np.exp(s)
                return e / e.sum(axis=-1, keepdims=True)
            return {
                "max_abs_ll_diff": float(np.max(np.abs(direct_ll - ref_ll))),
                "max_abs_prob_diff": float(np.max(np.abs(_sm(direct_ll) - _sm(ref_ll)))),
                "argmax_agreement": float(np.mean(np.argmax(direct_ll, axis=-1) == np.argmax(ref_ll, axis=-1))),
            }

        res["parity_bench"]["g1_direct_b64_vs_table_b64"] = _cmp_tbl(dir_64_np, tbl_64[codes_64])
        res["parity_bench"]["g1_direct_b16_vs_table_b64"] = _cmp_tbl(dir_16_np, tbl_64[codes_64[:16]])
        res["parity_bench"]["g1_direct_b1_vs_table_b64"] = _cmp_tbl(dir_1_np, tbl_64[codes_64[:8]])

    print(json.dumps(res, indent=2), flush=True)
    return res


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    arm: str = "phase",
    seed: int = 202611,
    epochs: int = 10,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    lr: float = 1.5e-3,
    max_steps: int = 200,
    train_scorer: str = "kv_cached",
    eval_scorer: str = "kv_cached",
    eval_batch_size: int = 64,
    seeds_csv: str = "",
):
    results_dir = STAGE3D_LOCAL / "results"
    ckpt_dir = STAGE3D_LOCAL / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        rep = remote_smoke_qwen.remote()
        (results_dir / "smoke_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in rep.items() if k != "candidate_token_audit"}, ensure_ascii=False, indent=2))
    elif mode == "bench":
        rep = remote_bench_train.remote()
        (results_dir / "bench_steady.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    elif mode == "train_one":
        out = remote_train_one.remote(
            arm=arm,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            lr=lr,
            max_steps_per_epoch=max_steps,
            train_scorer=train_scorer,
            eval_scorer=eval_scorer,
        )
        summ = out["summary"]
        (ckpt_dir / summ["checkpoint"]).write_bytes(out["ckpt_bytes"])
        (results_dir / f"train_summary_{arm}_s{seed}.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in summ.items() if k != "history"}, ensure_ascii=False, indent=2))
    elif mode == "train_all":
        from research.stage3d_qwen_signed_phase.data_stage3d import ARM_NAMES, CONNECTOR_SEEDS

        target_seeds = [int(x.strip()) for x in seeds_csv.split(",") if x.strip()] if seeds_csv else list(CONNECTOR_SEEDS)
        jobs = [
            (a, s, epochs, batch_size, micro_batch_size, lr, max_steps, 8, 2, train_scorer, eval_scorer)
            for s in target_seeds
            for a in ARM_NAMES
        ]
        all_summaries = {}
        summary_path = results_dir / "training_summary.json"
        if summary_path.exists():
            all_summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        for out in remote_train_one.starmap(jobs):
            summ = out["summary"]
            key = f"{summ['arm']}_s{summ['seed']}"
            (ckpt_dir / summ["checkpoint"]).write_bytes(out["ckpt_bytes"])
            (results_dir / f"train_summary_{key}.json").write_text(
                json.dumps(summ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            all_summaries[key] = summ
            print(f"[DONE] {key}: best_ep={summ['best_epoch']} sel={summ['best_sel_score']:.5f} T_q3={summ['temp_q3']:.3f}")
        summary_path.write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif mode == "dev_pilot":
        npz_bytes = remote_eval_dev_pilot.remote(batch_size=eval_batch_size, scorer_mode=eval_scorer)
        out_p = results_dir / "dev_pilot_tables.npz"
        out_p.write_bytes(npz_bytes)
        print(f"Saved {out_p} ({len(npz_bytes)} bytes)")
    elif mode == "eval_blind":
        out = remote_eval_blind.remote(batch_size=eval_batch_size, scorer_mode=eval_scorer)
        (results_dir / "direct_vs_table_parity.json").write_bytes(out["parity_bytes"])
        (results_dir / "raw_q3_gpu.npz").write_bytes(out["q3_bytes"])
        (results_dir / "raw_q2_gpu.npz").write_bytes(out["q2_bytes"])
        (results_dir / "raw_replay_gpu.npz").write_bytes(out["replay_bytes"])
        print("Saved raw blind GPU results:", json.dumps(out["meta"], indent=2))
    elif mode == "eval_ministral":
        out = remote_eval_historical_ministral.remote(mothers=128)
        hist_dir = results_dir / "historical_ministral"
        hist_dir.mkdir(parents=True, exist_ok=True)
        (hist_dir / "ministral_prompt_replay.json").write_bytes(out["e_bytes"])
        (hist_dir / "ministral_work_f_host_035.json").write_bytes(out["f_host_bytes"])
        (hist_dir / "ministral_work_f_results.json").write_bytes(out["f_res_bytes"])
        print("Saved historical Ministral diagnostics in", hist_dir)
    else:
        raise ValueError(f"Unknown mode={mode}")
