"""Google Colab Stage 5 & Stage 6 scaling runner (4B Real-SDR + CN-SFT -> 9B -> 27B Mixed-NF4).

Zero Modal GPU runs: fetches verified 4B checkpoints and Real-SDR `.npz` caches
from the GitHub Release mirror (`stage4-4b-colab-artifacts`), enforces W4
Theorem 2 NF4 quantization red lines (never quantizing Gated-DeltaNet,
Attention, embeddings, norms, or FP32 SpecCayley), and runs Stage 4 Real-SDR +
Stage 6 `m5-stage6-blind-v1` active EIG benchmarks on 4B, 9B, and 27B hosts.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3d_qwen_signed_phase.data_stage3d import load_s17_encoder
from research.stage3d_qwen_signed_phase.qwen_connector import (
    MODEL_ID_DEFAULT,
    PINNED_REVISION_DEFAULT,
    QwenCacheHost,
)
from research.stage4_math_and_design.M5 import verify_m5
from research.stage4_real_and_surgical.cn_sft_adapter import (
    CNSFTHostAdapter,
    GENERAL_KFAC_64_PROMPTS,
    QWEN4B_MLP_TUNE_LAYERS,
    _get_final_norm,
    _realify_uc,
    build_three_way_orthogonal_bus,
    calibrate_b_align_gamma_f,
    collect_dual_space_kfac_projectors,
    score_batch_with_cn_sft,
    verify_rmsnorm_gamma,
)
from research.stage4_real_and_surgical.cn_sft_probe import (
    _strip_summary,
    _train_cn_sft_arm,
    audit_cn_sft_pure_and_mix_kl,
    calibrate_fisher_qgate,
    calibrate_template_suffix_bias,
    run_l10_ood_theta_probe,
)
from research.stage4_real_and_surgical.equi_connector import (
    PhysicalKernelS17,
    Stage4MatchedConnector,
)
from research.stage4_real_and_surgical.phaseB_surgical import (
    GENERAL_TEXT_EVAL_PROMPTS,
    GENERAL_TEXT_TRAIN_PROMPTS,
    _build_blind_suites,
    _cluster_paired_stats,
    _make_lr_scheduler,
    _summarize_model_blind,
    _weighted_phaseB_minibatch_order,
    audit_pure_and_mix_kl,
    compute_frozen_general_refs,
)
from research.stage4_real_and_surgical.real_sdr_and_stage6_gpu import (
    Stage6ActiveHostAdapter,
    _evaluate_stage6_weights_on_catalogue,
    _make_real_prompt_set,
    _shard_to_eval_task,
    run_stage4_real_sdr_gpu,
    run_stage6_active_eig_gpu,
)
from research.stage4_real_and_surgical.sdr_real_pipeline import (
    load_bundle_npz,
    run_real_sdr_cpu_pipeline,
)
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    _get_decoder_layers,
)
from research.stage4_real_and_surgical.train_eval_stage4 import (
    EXPECTED_4B_VOCAB,
    build_stage4_pilot_groups,
)


RELEASE_REPO = "20171806011/EM-FloatFormer"
RELEASE_TAG = "stage4-4b-colab-artifacts"

ARTIFACT_SHA256: Dict[str, str] = {
    "cttc_dual_cache_c16.npz": "2795f6f3afecf515ed1359d8af13af72bf333b8c36d2cde8fac43b7be9735936",
    "pluto_dual_cache_c16.npz": "fa003137fabb69d87651b82d2b0020749f32875a54bf3a0373684b53346d548e",
    "stage4_cn_sft_u1_s202611.pt": "e808279ee3afadfda7f0a8409462d372f630e5a5d01ce2048a5b9762efe9ee88",
    "stage4_phaseA_equivariant_host_pure_s202611_step384.pt": "b60cc889c03bbbb278219fcc0ec805a8a9106ede4b267883932c03cb8ff94e95",
    "stage4_phaseA_equivariant_physical_residual_s202611_step384.pt": "bc437b302f4fc44672f8aca5eb07b4565f9242f335a45c482b7f27d8d6809587",
    "stage4_phaseA_naive_additive_s202611_step384.pt": "4ea0ec28a851439e18830bfab693ff9b209d6a4fb59e7c3ca04bfb8d2de5a63c",
    "stage4_phaseA_zero_s202611_step384.pt": "d221925f9ba630c3880d6f9042381907e42a63b5747810cea25663c5f565726c",
    "stage4_phaseB_lemmo_tier_p1_s202611.pt": "5b8527ce92626c7e3e205481386339cb1c9bee1015d10b7792ee6839ec50ea98",
    "stage4_phaseB_surgical96_s202611.pt": "f95e766f8e774e9848dd3d2df8647f8175ebee02693d45948ca9304c7d5875f3",
}

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "4b": {
        "model_id": MODEL_ID_DEFAULT,
        "revision": PINNED_REVISION_DEFAULT,
        "hidden_size": 2560,
        "num_layers": 32,
        "fa_layers": QWEN4B_FA_LAYERS,
        "dn_layers": QWEN4B_DN_LAYERS,
        "mlp_layers": QWEN4B_MLP_TUNE_LAYERS,
        "stop_layer": QWEN4B_STOP_LAYER,
        "default_quant": "bf16",
        "tie_word_embeddings": True,
    },
    "9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "hidden_size": 4096,
        "num_layers": 32,
        "fa_layers": QWEN4B_FA_LAYERS,
        "dn_layers": QWEN4B_DN_LAYERS,
        "mlp_layers": QWEN4B_MLP_TUNE_LAYERS,
        "stop_layer": QWEN4B_STOP_LAYER,
        "default_quant": "bf16",
        "tie_word_embeddings": False,
    },
    "27b": {
        "model_id": "Qwen/Qwen3.8-27B",
        "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "hidden_size": 5120,
        "num_layers": 64,
        "fa_layers": (19, 23, 27, 31, 35, 39, 47, 55),
        "dn_layers": (18, 22, 26, 30, 34, 38),
        "mlp_layers": (23, 31, 39),
        "stop_layer": 17,
        "default_quant": "mixed_nf4",
        "tie_word_embeddings": False,
    },
}

MIXED_NF4_SKIP_MODULES: Tuple[str, ...] = (
    "lm_head",
    "embed_tokens",
    "self_attn",
    "linear_attn",
    "visual",
    "input_layernorm",
    "post_attention_layernorm",
    "norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)

FORBIDDEN_NF4_PATTERNS: Tuple[str, ...] = (
    "linear_attn",
    "self_attn",
    "lm_head",
    "embed_tokens",
    "norm",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blk = f.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def download_stage4_release_artifacts(
    work_dir: Path,
    release_tag: str = RELEASE_TAG,
    repo_root: Optional[Path] = None,
    timeout_s: int = 120,
) -> dict:
    """Download and SHA-256 verify Stage 4 checkpoints and Real-SDR `.npz` caches."""
    work_dir = Path(work_dir)
    sdr_dir = work_dir / "real_sdr_cache"
    ckpt_dir = work_dir / "checkpoints"
    sdr_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    statuses: Dict[str, Any] = {}
    for fname, exp_sha in ARTIFACT_SHA256.items():
        dst = (sdr_dir if fname.endswith(".npz") else ckpt_dir) / fname
        if dst.exists() and _sha256_file(dst) == exp_sha:
            statuses[fname] = {"path": str(dst), "status": "cached_ok", "sha256": exp_sha}
            continue

        url = f"https://github.com/{RELEASE_REPO}/releases/download/{release_tag}/{fname}"
        tmp = dst.parent / f".{fname}.tmp"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp, tmp.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            got_sha = _sha256_file(tmp)
            if got_sha != exp_sha:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch for {fname}: got {got_sha}, expected {exp_sha}")
            tmp.replace(dst)
            statuses[fname] = {"path": str(dst), "status": "downloaded_ok", "sha256": exp_sha}
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            statuses[fname] = {"path": str(dst), "status": f"error: {type(exc).__name__}: {exc}"}

    npz_missing = [
        fn for fn in ("pluto_dual_cache_c16.npz", "cttc_dual_cache_c16.npz") if not (sdr_dir / fn).exists()
    ]
    if npz_missing and repo_root is not None:
        raw_dir = work_dir / "real_sdr_raw"
        run_real_sdr_cpu_pipeline(repo_root, raw_dir, sdr_dir, mock_synthetic_for_test=False)
        for fn in npz_missing:
            statuses[fn] = {
                "path": str(sdr_dir / fn),
                "status": "regenerated_from_source",
                "sha256": _sha256_file(sdr_dir / fn),
            }

    return {
        "release_repo": RELEASE_REPO,
        "release_tag": release_tag,
        "real_sdr_cache_dir": str(sdr_dir),
        "ckpt_dir": str(ckpt_dir),
        "artifacts": statuses,
    }


def detect_colab_gpu_profile(
    vram_gib: Optional[float] = None,
    gpu_name: Optional[str] = None,
) -> dict:
    """Compute safe precision, batch, and gradient-checkpointing profile from GPU VRAM."""
    if vram_gib is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gib = float(props.total_memory) / (1024.0 ** 3)
            gpu_name = gpu_name or str(props.name)
        else:
            vram_gib = 0.0
            gpu_name = gpu_name or "CPU"
    vram_f = float(vram_gib)
    gname = str(gpu_name or "unknown")

    if vram_f >= 70.0:
        tier = "a100_80g_h100"
        models = {
            "4b": {"supported": True, "quant": "bf16", "eval_bs": 128, "batch_size": 32, "micro_batch_size": 16, "grad_ckpt": False, "run_u1_scaled": True},
            "9b": {"supported": True, "quant": "bf16", "eval_bs": 64, "batch_size": 32, "micro_batch_size": 16, "grad_ckpt": False, "run_u1_scaled": True},
            "27b": {"supported": True, "quant": "mixed_nf4", "eval_bs": 32, "batch_size": 16, "micro_batch_size": 8, "grad_ckpt": True, "run_u1_scaled": False},
        }
    elif vram_f >= 35.0:
        tier = "a100_40g"
        models = {
            "4b": {"supported": True, "quant": "bf16", "eval_bs": 64, "batch_size": 32, "micro_batch_size": 8, "grad_ckpt": False, "run_u1_scaled": True},
            "9b": {"supported": True, "quant": "bf16", "eval_bs": 32, "batch_size": 16, "micro_batch_size": 8, "grad_ckpt": True, "run_u1_scaled": False},
            "27b": {"supported": True, "quant": "mixed_nf4", "eval_bs": 16, "batch_size": 16, "micro_batch_size": 4, "grad_ckpt": True, "run_u1_scaled": False},
        }
    elif vram_f >= 20.0:
        tier = "l4_24g"
        models = {
            "4b": {"supported": True, "quant": "bf16", "eval_bs": 32, "batch_size": 16, "micro_batch_size": 4, "grad_ckpt": True, "run_u1_scaled": True},
            "9b": {"supported": True, "quant": "mixed_nf4", "eval_bs": 16, "batch_size": 16, "micro_batch_size": 4, "grad_ckpt": True, "run_u1_scaled": False},
            "27b": {"supported": False, "quant": "mixed_nf4", "reason": "27B Mixed-NF4 needs >=35 GiB VRAM"},
        }
    else:
        tier = "t4_16g" if vram_f > 0.0 else "cpu_mock"
        models = {
            "4b": {"supported": vram_f >= 12.0, "quant": "mixed_nf4", "eval_bs": 16, "batch_size": 16, "micro_batch_size": 4, "grad_ckpt": True, "run_u1_scaled": True},
            "9b": {"supported": False, "quant": "mixed_nf4", "reason": "9B needs >=20 GiB VRAM"},
            "27b": {"supported": False, "quant": "mixed_nf4", "reason": "27B Mixed-NF4 needs >=35 GiB VRAM"},
        }
    return {
        "gpu_name": gname,
        "vram_gib": round(vram_f, 2),
        "tier": tier,
        "models": models,
    }


def _is_4bit_quantized_module(mod: nn.Module) -> bool:
    if getattr(mod, "_is_4bit_nf4_mock", False):
        return True
    if type(mod).__name__ in ("Linear4bit", "Params4bit"):
        return True
    w = getattr(mod, "weight", None)
    return w is not None and hasattr(w, "quant_state")


def audit_mixed_nf4_layout(lm: nn.Module) -> dict:
    """Enforce W4 Theorem 2: NF4 strictly on SwiGLU MLPs, NEVER on DeltaNet/Attention/Norms/Heads."""
    forbidden_quantized: List[str] = []
    mlp_quantized_count = 0
    mlp_unquantized_count = 0
    non_mlp_bf16_count = 0

    for name, mod in lm.named_modules():
        is_q = _is_4bit_quantized_module(mod)
        is_leaf_lin = isinstance(mod, nn.Linear) or is_q
        if not is_leaf_lin:
            continue

        if is_q and any(pat in name for pat in FORBIDDEN_NF4_PATTERNS):
            forbidden_quantized.append(name)

        is_mlp_proj = (".mlp." in name or name.startswith("mlp.")) and any(
            name.endswith(s) for s in ("gate_proj", "up_proj", "down_proj")
        )
        if is_mlp_proj:
            if is_q:
                mlp_quantized_count += 1
            else:
                mlp_unquantized_count += 1
        elif not is_q:
            non_mlp_bf16_count += 1

    assert not forbidden_quantized, f"W4 Theorem 2 violation: quantized forbidden modules {forbidden_quantized[:10]}"
    assert mlp_quantized_count > 0, "Mixed-NF4 audit failed: zero MLP projections quantized"
    assert mlp_unquantized_count == 0, f"Mixed-NF4 incomplete: {mlp_unquantized_count} unquantized MLP linears"

    return {
        "mlp_quantized_count": int(mlp_quantized_count),
        "mlp_unquantized_count": int(mlp_unquantized_count),
        "forbidden_quantized": forbidden_quantized,
        "non_mlp_bf16_count": int(non_mlp_bf16_count),
        "passes_w4_theorem2_redline": True,
    }


def _compute_scaled_host_fingerprint(lm: nn.Module, quant_mode: str = "bf16") -> str:
    """Compute host parameter SHA-256 supporting both pure BF16/FP32 and Params4bit hosts."""
    if quant_mode != "mixed_nf4":
        return _compute_host_fingerprint(lm)

    h = hashlib.sha256()
    for name, p in sorted(lm.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("utf-8"))
        raw = torch.Tensor.as_subclass(p.detach(), torch.Tensor).to(device="cpu", dtype=torch.float32)
        stats = torch.stack([raw.mean(), raw.std(unbiased=False), raw.abs().max(), raw.flatten()[0], raw.flatten()[-1]])
        h.update(stats.numpy().tobytes())
        qs = getattr(p, "quant_state", None)
        if qs is not None and getattr(qs, "absmax", None) is not None:
            am = qs.absmax.detach().to(device="cpu", dtype=torch.float32)
            am_s = torch.stack([am.mean(), am.std(unbiased=False), am.abs().max()])
            h.update(am_s.numpy().tobytes())
    return h.hexdigest()


class ScaledQwenCacheHost(QwenCacheHost):
    """Scaled BF16 / W4 Mixed-NF4 / mock Qwen3.5 host wrapper preserving qwen_connector.py."""

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        revision: Optional[str] = PINNED_REVISION_DEFAULT,
        quant: str = "bf16",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        mock_tiny_for_test: bool = False,
        mock_hidden_size: int = 64,
        mock_num_layers: int = 2,
    ):
        self.quant_mode = str(quant)
        self.quant_audit: Optional[dict] = None

        if mock_tiny_for_test:
            from transformers import Qwen2Config, Qwen2ForCausalLM

            self.model_id = model_id
            self.requested_revision = revision
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            cfg = Qwen2Config(
                vocab_size=256,
                hidden_size=int(mock_hidden_size),
                intermediate_size=int(mock_hidden_size) * 2,
                num_hidden_layers=int(mock_num_layers),
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
            )
            self.dtype = torch.float32
            self.lm = Qwen2ForCausalLM(cfg).to(self.device)
            self.tokenizer = None
            self.hf_sha = f"mock_tiny_{quant}"
            self.transformers_version = "mock"
            self.chat_template_sha256 = "mock"
            self.thinking_mode = "mock"
            self.candidate_token_audit = {}
            self._is_qwen35_mrope = False
            self.lm.eval()
            for p in self.lm.parameters():
                p.requires_grad_(False)
            self.text_config = self.lm.config
            self.hidden_size = int(self.text_config.hidden_size)
            self._supports_logits_to_keep = False
            self._uses_cache_kwarg = False
            embed_w = self.lm.get_input_embeddings().weight.detach().float()
            self.target_embed_std = float(embed_w.std().item())
            self.host_sha256 = _compute_scaled_host_fingerprint(self.lm, self.quant_mode)
            self.frozen_param_count = int(sum(p.numel() for p in self.lm.parameters()))
            return

        if self.quant_mode == "bf16":
            super().__init__(
                model_id=model_id,
                revision=revision,
                device=device,
                dtype=dtype,
                mock_tiny_for_test=False,
            )
            return

        if self.quant_mode != "mixed_nf4":
            raise ValueError(f"Unsupported quant={self.quant_mode!r}; expected 'bf16' or 'mixed_nf4'")

        import inspect
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

        self.model_id = model_id
        self.requested_revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        assert self.device.type == "cuda", "Mixed-NF4 bitsandbytes loading requires a CUDA GPU"
        self.transformers_version = getattr(transformers, "__version__", "unknown")
        self.dtype = torch.bfloat16
        try:
            self.hf_sha = str(HfApi().model_info(model_id, revision=revision).sha)
        except Exception:
            self.hf_sha = str(revision or "unknown")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        tpl = getattr(self.tokenizer, "chat_template", "") or ""
        self.chat_template_sha256 = hashlib.sha256(str(tpl).encode("utf-8")).hexdigest()
        self.thinking_mode = "disabled (enable_thinking=False -> <think>\\n\\n</think>\\n\\n)"

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(MIXED_NF4_SKIP_MODULES),
        )
        dev_idx = 0 if self.device.index is None else int(self.device.index)
        try:
            lm = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                quantization_config=bnb_cfg,
                dtype=self.dtype,
                device_map={"": dev_idx},
                low_cpu_mem_usage=True,
            )
        except TypeError:
            lm = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                quantization_config=bnb_cfg,
                torch_dtype=self.dtype,
                device_map={"": dev_idx},
                low_cpu_mem_usage=True,
            )

        if hasattr(lm, "model") and hasattr(lm.model, "visual"):
            lm.model.visual = None
        if hasattr(lm, "visual"):
            lm.visual = None
        gc.collect()
        torch.cuda.empty_cache()

        self.lm = lm
        self.quant_audit = audit_mixed_nf4_layout(self.lm)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)

        text_cfg = getattr(self.lm.config, "text_config", self.lm.config)
        self.text_config = text_cfg
        self.hidden_size = int(text_cfg.hidden_size)
        fwd_params = inspect.signature(self.lm.forward).parameters
        self._supports_logits_to_keep = "logits_to_keep" in fwd_params
        self._uses_cache_kwarg = "cache" in fwd_params
        embed_weight = self.lm.get_input_embeddings().weight.detach().float()
        self.target_embed_std = float(embed_weight.std().item())
        self._is_qwen35_mrope = True
        self.candidate_token_audit = self._audit_candidate_tokenizations()
        self.host_sha256 = _compute_scaled_host_fingerprint(self.lm, "mixed_nf4")
        self.frozen_param_count = int(sum(p.numel() for p in self.lm.parameters()))

    def set_grad_checkpointing(self, enabled: bool) -> None:
        super().set_grad_checkpointing(enabled)
        if enabled and hasattr(self.lm, "enable_input_require_grads"):
            self.lm.enable_input_require_grads()


def _attach_readonly_qgate_hook(lm: nn.Module, adapter: CNSFTHostAdapter, orig_stop: int) -> None:
    """Update Fisher Query Gate at orig_stop without detaching intermediate activations."""
    layers = _get_decoder_layers(lm)
    if len(layers) >= 4 and 0 <= int(orig_stop) < len(layers):
        def _qgate_readonly(mod, args, out):
            if adapter.g_em == 0:
                return
            h_raw = out[0] if isinstance(out, tuple) else out
            adapter._update_qgate_from_tensor(h_raw.detach())

        adapter.handles.append(layers[int(orig_stop)].register_forward_hook(_qgate_readonly))


def reconstruct_cn_sft_from_checkpoint(
    host: QwenCacheHost,
    ckpt_path: Path,
    disable_stop_grad: bool = False,
) -> CNSFTHostAdapter:
    """Reconstruct a trained CNSFTHostAdapter from a saved `.pt` checkpoint without needing raw K-FAC."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd: Dict[str, torch.Tensor] = ck["state_dict"]

    fa_layers = [int(i) for i in ck["fa_layers"]]
    dn_layers = [int(i) for i in ck["dn_layers"]]
    mlp_layers = [int(i) for i in ck["mlp_layers"]]
    orig_stop = int(ck.get("stop_layer", 9))

    u_c_d: Dict[int, torch.Tensor] = {}
    u_c_o: Dict[int, torch.Tensor] = {}
    u_c_mlp: Dict[int, torch.Tensor] = {}

    for i in fa_layers:
        if f"fa_cayleys.{i}.u_c" in sd:
            u_c_o[i] = sd[f"fa_cayleys.{i}.u_c"]
        if f"fa_loras.{i}_q_proj.u_c" in sd:
            u_c_d[i] = sd[f"fa_loras.{i}_q_proj.u_c"]
    for i in dn_layers:
        if f"dn_cayleys.{i}.u_c" in sd:
            u_c_o[i] = sd[f"dn_cayleys.{i}.u_c"]
        if f"dn_loras.{i}_in_proj_kv.u_c" in sd:
            u_c_d[i] = sd[f"dn_loras.{i}_in_proj_kv.u_c"]
    for i in mlp_layers:
        if f"mlp_overlays.{i}.u_c" in sd:
            u_c_mlp[i] = sd[f"mlp_overlays.{i}.u_c"]

    u_c_bus = sd.get("schur_readout.u_c_last")
    r_c = int(sd[f"fa_cayleys.{fa_layers[0]}.u_re"].shape[1]) if (fa_layers and f"fa_cayleys.{fa_layers[0]}.u_re" in sd) else 3
    r_s = int(sd[f"fa_cayleys.{fa_layers[0]}.sigma_raw"].shape[0]) if (fa_layers and f"fa_cayleys.{fa_layers[0]}.sigma_raw" in sd) else 2
    r_lora = int(sd[f"fa_loras.{fa_layers[0]}_q_proj.lora_a"].shape[0]) if (fa_layers and f"fa_loras.{fa_layers[0]}_q_proj.lora_a" in sd) else 6
    k_tune = int(sd[f"mlp_overlays.{mlp_layers[0]}.w_in_orth"].shape[0] // 4) if (mlp_layers and f"mlp_overlays.{mlp_layers[0]}.w_in_orth" in sd) else 8
    r_schur = int(sd["schur_readout.proj_in"].shape[0])

    # Load integer/tail keys for tail_rms_medians safely whether keys are str or int
    raw_tail = ck["tail_rms_medians"]
    tail_rms = {(int(k) if str(k).isdigit() else k): float(v) for k, v in raw_tail.items()}

    ad = CNSFTHostAdapter(
        host.lm,
        u_c_d=u_c_d,
        u_c_o=u_c_o,
        u_c_mlp=u_c_mlp,
        u_c_bus=u_c_bus,
        b_align_info=ck["b_align_info"],
        e_bus=ck["e_bus"],
        tail_rms_medians=tail_rms,
        fa_layers=fa_layers,
        dn_layers=dn_layers,
        mlp_tune_layers=mlp_layers,
        stop_layer=(-1 if disable_stop_grad else orig_stop),
        arm_mode=str(ck.get("arm", "u1_full_cnsft")),
        regime=int(ck.get("regime", 2)),
        r_c_cayley=r_c,
        r_s_cayley=r_s,
        r_lora=r_lora,
        k_tune=k_tune,
        r_schur=r_schur,
        l10_ridge_proj=ck.get("l10_ridge_proj"),
        l10_ridge_bias=ck.get("l10_ridge_bias"),
    )
    ad.load_state_dict({k: v.cpu() for k, v in sd.items()}, strict=True)
    ad.to(host.device)
    ad.eval()

    if disable_stop_grad:
        _attach_readonly_qgate_hook(host.lm, ad, orig_stop)

    return ad


def _build_real_domain_specs(cache_in_dir: Path) -> Tuple[dict, dict, dict]:
    pluto_bundle = load_bundle_npz(cache_in_dir / "pluto_dual_cache_c16.npz")
    cttc_bundle = load_bundle_npz(cache_in_dir / "cttc_dual_cache_c16.npz")

    def _filter_shard(sh: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: v[mask] for k, v in sh.items() if isinstance(v, np.ndarray)}

    cttc_te_in = _filter_shard(cttc_bundle["splits"]["test"], ~cttc_bundle["splits"]["test"]["is_holdout"])
    cttc_te_ho = _filter_shard(cttc_bundle["splits"]["test"], cttc_bundle["splits"]["test"]["is_holdout"])
    pluto_te = pluto_bundle["splits"]["test"]
    pluto_stress = pluto_bundle["stress"]
    cttc_stress_in = _filter_shard(cttc_bundle["stress"], ~cttc_bundle["stress"]["is_holdout"])
    prompts = _make_real_prompt_set()

    domains = {
        "pluto_test": {
            "q1": ({"shard": pluto_te, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": pluto_te, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_in_domain_test": {
            "q1": ({"shard": cttc_te_in, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": cttc_te_in, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_unseen_holdout_f708_snr10": {
            "q1": ({"shard": cttc_te_ho, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": cttc_te_ho, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "pluto_stress": {
            "q2_two": ({"shard": pluto_stress, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_stress": {
            "q2_two": ({"shard": cttc_stress_in, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
    }
    bw_missing = {
        dom_k: {
            "q2_none": ({"shard": dom_sh, "qtype_key": "Q2", "prompt_key": "Q2_none"}, "ref_none", "labels_q2"),
            "q2_one": ({"shard": dom_sh, "qtype_key": "Q2", "prompt_key": "Q2_one"}, "ref_one", "labels_q2"),
        }
        for dom_k, dom_sh in (
            ("pluto_test", pluto_te),
            ("cttc_in_domain_test", cttc_te_in),
            ("cttc_unseen_holdout_f708_snr10", cttc_te_ho),
        )
    }
    return domains, bw_missing, prompts


def _eval_real_domains_cnsft(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    domains: Dict[str, Dict[str, tuple]],
    bw_missing: Dict[str, Dict[str, tuple]],
    adapter: CNSFTHostAdapter,
    bits: int,
    prompts: Dict[str, Tuple[str, Tuple[str, ...]]],
    eval_bs: int,
    scorer_mode: str = "kv_cached",
) -> dict:
    connector.eval()
    adapter.eval()

    def _run_doms(dom_map: Dict[str, Dict[str, tuple]]) -> dict:
        out: Dict[str, Any] = {}
        with torch.no_grad():
            for dname, tasks in dom_map.items():
                d_rep: Dict[str, Any] = {}
                for tname, (qkey, rkey, lkey) in tasks.items():
                    tk = _shard_to_eval_task(
                        qkey["shard"], qkey["qtype_key"], rkey, prompts[qkey["prompt_key"]], bits=bits, label_key=lkey
                    )
                    n = len(tk["words"])
                    chunks: List[np.ndarray] = []
                    for st in range(0, n, eval_bs):
                        ed = min(n, st + eval_bs)
                        _, _, probs, _ = score_batch_with_cn_sft(
                            connector,
                            host,
                            tk["prompt"],
                            tk["choices"],
                            tk["qtype"],
                            torch.from_numpy(tk["words"][st:ed]).to(host.device),
                            int(bits),
                            torch.from_numpy(tk["phase_slots"][st:ed]).to(host.device),
                            torch.from_numpy(tk["meta_scalars"][st:ed]).to(host.device),
                            adapter=adapter,
                            ref_summary=torch.from_numpy(tk["ref_summary"][st:ed]).to(host.device),
                            scorer_mode=scorer_mode,
                            discrete_d=torch.from_numpy(tk["discrete_d"][st:ed]).to(host.device),
                        )
                        chunks.append(probs.detach().cpu().numpy().astype(np.float64))
                    probs_all = np.concatenate(chunks, axis=0)
                    ys = tk["labels"]
                    nll = -np.log(np.clip(probs_all[np.arange(n), ys], 1e-12, 1.0))
                    hits = (probs_all.argmax(axis=-1) == ys).astype(np.float64)
                    d_rep[tname] = {"n": int(n), "acc": float(np.mean(hits)), "nll": float(np.mean(nll))}
                out[dname] = d_rep
        return out

    summary = _run_doms(domains)
    bw = _run_doms(bw_missing)
    for dom_k, d_vals in bw.items():
        q2_two = summary[dom_k]["q2_two"]
        d_vals["q2_two"] = q2_two
        d_vals["q2_one_minus_none_acc"] = d_vals["q2_one"]["acc"] - d_vals["q2_none"]["acc"]
        d_vals["q2_two_minus_one_acc"] = q2_two["acc"] - d_vals["q2_one"]["acc"]
        d_vals["q2_two_minus_none_acc"] = q2_two["acc"] - d_vals["q2_none"]["acc"]
        d_vals["q2_two_minus_none_nll"] = q2_two["nll"] - d_vals["q2_none"]["nll"]
    return {"summary": summary, "blackwell_staircase": bw}


def _eval_stage6_with_adapter(
    host: QwenCacheHost,
    catalogue: Sequence[Any],
    encoder: Any,
    adapter: Optional[nn.Module],
    n_train: int,
    n_dev: int,
    n_test: int,
    steps: int,
    batch_size: int,
    eval_bs: int,
    bits: int,
    seed: int,
    grad_ckpt: bool,
    mname: str,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Train & evaluate `Stage6ActiveHostAdapter` cleanly with optional frozen adapter and `grad_ckpt`."""
    t_m0 = time.perf_counter()
    torch.manual_seed(seed + bits + len(mname))
    rng = np.random.default_rng(seed + bits + len(mname))

    train_bundles = [verify_m5.geometry_split(g, encoder, fid, n_train, 99) for fid, g in enumerate(catalogue)]
    dev_bundles = [verify_m5.geometry_split(g, encoder, fid, n_dev, 0) for fid, g in enumerate(catalogue)]
    test_bundles = [verify_m5.geometry_split(g, encoder, fid, n_test, 1) for fid, g in enumerate(catalogue)]

    train_words = np.concatenate([b[0] for b in train_bundles], axis=0)
    train_fams = np.concatenate([np.full(len(b[0]), fid, dtype=np.int64) for fid, b in enumerate(train_bundles)], axis=0)
    train_targets = np.concatenate(
        [verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(train_bundles)], axis=0
    ).astype(np.float32)

    prose_ids_by_fam: List[torch.Tensor] = []
    for g in catalogue:
        text = f"公有信息家族：{g.name}。\n{verify_m5.FROZEN_QUERY}"
        if host.tokenizer is None:
            ids = [1, 2, 3, 4, 5, 6]
        else:
            ids = host.tokenizer.encode(text, add_special_tokens=False)
        prose_ids_by_fam.append(torch.tensor(ids, device=host.device, dtype=torch.long))

    head_mod = Stage6ActiveHostAdapter(hidden_size=host.hidden_size).to(host.device)
    if adapter is not None:
        adapter.requires_grad_(False)
        adapter.eval()

    try:
        opt = torch.optim.AdamW(head_mod.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = _make_lr_scheduler(opt, steps)
        head_mod.train()
        if grad_ckpt:
            host.set_grad_checkpointing(True)
        try:
            for st_i in range(steps):
                fid = st_i % 3
                mask_f = np.where(train_fams == fid)[0]
                sel = rng.choice(mask_f, size=min(batch_size, len(mask_f)), replace=False)
                opt.zero_grad(set_to_none=True)
                w_b = torch.from_numpy(train_words[sel].astype(np.int64)).to(host.device)
                f_b = torch.full((len(sel),), fid, device=host.device, dtype=torch.long)
                tgt = torch.from_numpy(train_targets[sel]).to(host.device).clamp_min(1e-12)
                if adapter is not None and grad_ckpt:
                    outer_m_mask = torch.cat(
                        (
                            torch.ones((len(sel), 8), device=host.device, dtype=torch.bool),
                            torch.zeros(
                                (len(sel), prose_ids_by_fam[fid].shape[0]),
                                device=host.device,
                                dtype=torch.bool,
                            ),
                        ),
                        dim=1,
                    )
                    outer_ctx = adapter.activate(g_em=1, m_acc_mask=outer_m_mask)
                else:
                    outer_ctx = nullcontext()
                with outer_ctx:
                    logits = head_mod.forward_states(
                        host, w_b, f_b, prose_ids_by_fam[fid], bits=bits, surgical_adapter=adapter
                    )
                    lp = F.log_softmax(logits, dim=-1)
                    loss = torch.sum(tgt * (torch.log(tgt) - lp), dim=-1).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(head_mod.parameters(), 1.0)
                opt.step()
                sched.step()
                del logits, lp, loss
        finally:
            if grad_ckpt:
                host.set_grad_checkpointing(False)

        head_mod.eval()

        def _predict_weights(bundles: Sequence[tuple]) -> List[np.ndarray]:
            out_w: List[np.ndarray] = []
            with torch.no_grad():
                for fid_i, bnd in enumerate(bundles):
                    ws = bnd[0]
                    chunks: List[np.ndarray] = []
                    for s0 in range(0, len(ws), eval_bs):
                        s1 = min(len(ws), s0 + eval_bs)
                        wb = torch.from_numpy(ws[s0:s1].astype(np.int64)).to(host.device)
                        fb = torch.full((s1 - s0,), fid_i, device=host.device, dtype=torch.long)
                        lg = head_mod.forward_states(
                            host, wb, fb, prose_ids_by_fam[fid_i], bits=bits, surgical_adapter=adapter
                        )
                        chunks.append(F.softmax(lg, dim=-1).cpu().numpy().astype(np.float64))
                    out_w.append(np.concatenate(chunks, axis=0))
            return out_w

        w_dev_p = _predict_weights(dev_bundles)
        w_tst_p = _predict_weights(test_bundles)
        rep, nlls, hits = _evaluate_stage6_weights_on_catalogue(
            catalogue, encoder, n_dev, n_test, w_dev_p, w_tst_p, dev_seed=0, test_seed=1
        )
        head_params = head_mod.param_count()
        a_params = adapter.param_count() if adapter is not None else 0
        cum_params = head_params + a_params
        rep["bits"] = int(bits)
        rep["stage6_adapter_params"] = head_params
        rep["cnsft_adapter_params"] = a_params
        rep["cumulative_params"] = cum_params
        rep["host_param_fraction"] = float(cum_params) / float(max(1, host.frozen_param_count))
        rep["wall_s"] = round(time.perf_counter() - t_m0, 2)
        del opt, sched
        return rep, nlls, hits
    finally:
        del head_mod
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()


def run_real_sdr_cnsft_4b_eval(
    repo_root: Path,
    sdr_cache_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    host: QwenCacheHost,
    seed: int = 202611,
    eval_bs: int = 128,
    scorer_mode: str = "kv_cached",
    mock_tiny_for_test: bool = False,
) -> dict:
    """Evaluate trained 4B `stage4_cn_sft_u1` on Real-SDR (4-bit & 16-bit) + Blackwell + Pure/MixKL."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, _ = load_s17_encoder(repo_root)

    conn = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)
    conn_p = ckpt_dir / f"stage4_phaseA_equivariant_host_pure_s{seed}_step384.pt"
    if conn_p.exists():
        conn.load_state_dict(torch.load(conn_p, map_location=host.device, weights_only=False)["state_dict"])
    elif not mock_tiny_for_test:
        raise FileNotFoundError(f"Missing {conn_p}")
    conn.requires_grad_(False)
    conn.eval()

    cn_ckpt_p = ckpt_dir / f"stage4_cn_sft_u1_s{seed}.pt"
    if not mock_tiny_for_test:
        assert cn_ckpt_p.exists(), f"Missing {cn_ckpt_p}"
    adapter = reconstruct_cn_sft_from_checkpoint(host, cn_ckpt_p, disable_stop_grad=False)

    domains, bw_missing, prompts = _build_real_domain_specs(sdr_cache_dir)
    q_mode = getattr(host, "quant_mode", "bf16")
    try:
        eval_16 = _eval_real_domains_cnsft(host, conn, domains, bw_missing, adapter, 16, prompts, eval_bs, scorer_mode)
        eval_4 = _eval_real_domains_cnsft(host, conn, domains, bw_missing, adapter, 4, prompts, eval_bs, scorer_mode)
        audit = audit_cn_sft_pure_and_mix_kl(host, conn, adapter, GENERAL_TEXT_EVAL_PROMPTS)
        assert audit["passes_pure_kl_zero"], f"Real-SDR 4B CN-SFT PureKL != 0: {audit}"
        assert _compute_scaled_host_fingerprint(host.lm, q_mode) == host.host_sha256, "Host SHA mutated!"
    finally:
        adapter.remove()
        del adapter, conn
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    rep = {
        "seed": seed,
        "model_id": host.model_id,
        "quant_mode": q_mode,
        "host_sha256": host.host_sha256,
        "checkpoint": cn_ckpt_p.name,
        "wall_s": round(time.perf_counter() - t0, 2),
        "real_sdr_c16": eval_16,
        "real_sdr_4bit": eval_4,
        "audit": audit,
    }
    (out_dir / "stage4_real_sdr_4b_cnsft_results.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rep


def run_stage6_4b_full(
    repo_root: Path,
    work_dir: Path,
    results_dir: Path,
    seed: int = 202611,
    quant: str = "bf16",
    steps_stage6: int = 240,
    steps_adapt: int = 96,
    steps_soup: int = 48,
    batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    eval_bs: Optional[int] = None,
    grad_ckpt: Optional[bool] = None,
    mock_tiny_for_test: bool = False,
) -> dict:
    """Run full 4B Stage 4 Real-SDR + CN-SFT Real-SDR + Stage 6 Active EIG (`m5-stage6-blind-v1`)."""
    work_dir = Path(work_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prof = detect_colab_gpu_profile() if not mock_tiny_for_test else detect_colab_gpu_profile(80.0, "CPU-Mock")
    p4 = prof["models"]["4b"]
    eff_bs = int(eval_bs if eval_bs is not None else p4.get("eval_bs", 128))
    eff_train_bs = int(batch_size if batch_size is not None else p4.get("batch_size", 32))
    eff_mb = int(micro_batch_size if micro_batch_size is not None else p4.get("micro_batch_size", 16))
    eff_ckpt = bool(grad_ckpt if grad_ckpt is not None else p4.get("grad_ckpt", False))

    host = ScaledQwenCacheHost(
        model_id=MODEL_SPECS["4b"]["model_id"],
        revision=MODEL_SPECS["4b"]["revision"],
        quant=quant,
        mock_tiny_for_test=mock_tiny_for_test,
    )
    sdr_cache = work_dir / "real_sdr_cache"
    ckpt_dir = work_dir / "checkpoints"
    kfac_p = work_dir / "kfac_4b_top256.pt"

    import research.stage4_real_and_surgical.real_sdr_and_stage6_gpu as _rs6_mod

    orig_fp = _rs6_mod._compute_host_fingerprint
    if quant == "mixed_nf4":
        _rs6_mod._compute_host_fingerprint = lambda lm_obj: _compute_scaled_host_fingerprint(lm_obj, "mixed_nf4")
    try:
        sdr_rep = run_stage4_real_sdr_gpu(
            repo_root,
            sdr_cache,
            results_dir,
            ckpt_dir,
            kfac_p,
            seed=seed,
            steps_adapt=steps_adapt,
            steps_soup=steps_soup,
            batch_size=eff_train_bs,
            micro_batch_size=eff_mb,
            eval_bs=eff_bs,
            mock_tiny_for_test=mock_tiny_for_test,
            host=host,
            cache_dir=work_dir / "real_sdr_4b_cache",
        )
        cnsft_sdr_rep = run_real_sdr_cnsft_4b_eval(
            repo_root,
            sdr_cache,
            results_dir,
            ckpt_dir,
            host=host,
            seed=seed,
            eval_bs=eff_bs,
            mock_tiny_for_test=mock_tiny_for_test,
        )
        s6_rep = run_stage6_active_eig_gpu(
            repo_root,
            results_dir,
            ckpt_dir,
            kfac_p,
            seed=seed,
            steps=steps_stage6,
            batch_size=eff_train_bs,
            eval_bs=eff_bs,
            mock_tiny_for_test=mock_tiny_for_test,
            host=host,
            cache_dir=work_dir / "stage6_4b_cache",
        )

        # Evaluate host_cnsft_c16 on identical sealed m5-stage6-blind-v1 splits
        encoder, _ = load_s17_encoder(repo_root)
        catalogue = verify_m5.geometry_catalogue()
        n_train = 64 if mock_tiny_for_test else 2048
        n_dev = 32 if mock_tiny_for_test else verify_m5.ADAPTIVE_DEV
        n_test = 64 if mock_tiny_for_test else verify_m5.ADAPTIVE_TEST

        cn_ckpt_p = ckpt_dir / f"stage4_cn_sft_u1_s{seed}.pt"
        cn_ad = reconstruct_cn_sft_from_checkpoint(host, cn_ckpt_p, disable_stop_grad=True)
        try:
            cnsft_s6_rep, _, _ = _eval_stage6_with_adapter(
                host,
                catalogue,
                encoder,
                cn_ad,
                n_train=n_train,
                n_dev=n_dev,
                n_test=n_test,
                steps=steps_stage6,
                batch_size=eff_train_bs,
                eval_bs=eff_bs,
                bits=16,
                seed=seed,
                grad_ckpt=eff_ckpt,
                mname="host_cnsft_c16",
            )
            aud_conn = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)
            aud_cnsft = audit_pure_and_mix_kl(host, aud_conn, cn_ad, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=True)
            assert aud_cnsft["passes_pure_kl_zero"], f"Stage 6 host_cnsft_c16 PureKL != 0: {aud_cnsft}"
            cnsft_s6_rep["pure_kl_audit"] = aud_cnsft
            del aud_conn
        finally:
            cn_ad.remove()
            del cn_ad

        assert _compute_scaled_host_fingerprint(host.lm, quant) == host.host_sha256, "4B host SHA mutated!"
        (results_dir / "stage6_active_eig_4b_cnsft_results.json").write_text(
            json.dumps(cnsft_s6_rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "real_sdr_4b": sdr_rep,
            "real_sdr_4b_cnsft": cnsft_sdr_rep,
            "stage6_4b": s6_rep,
            "stage6_4b_cnsft": cnsft_s6_rep,
        }
    finally:
        _rs6_mod._compute_host_fingerprint = orig_fp
        del host
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_scaled_model_stage6(
    model_key: str,
    repo_root: Path,
    work_dir: Path,
    results_dir: Path,
    seed: int = 202611,
    quant_override: Optional[str] = None,
    steps_stage6: int = 240,
    steps_u1: int = 48,
    g1_blocks: int = 32,
    eval_bs: Optional[int] = None,
    batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    grad_ckpt: Optional[bool] = None,
    run_u1: Optional[bool] = None,
    mock_tiny_for_test: bool = False,
) -> dict:
    """Execute CN-SFT + Real-SDR + Stage 6 Active EIG on `Qwen3.5-9B` or `Qwen3.8-27B Mixed-NF4`."""
    if model_key not in ("9b", "27b"):
        raise ValueError(f"Expected model_key in ('9b', '27b'), got {model_key!r}")
    t_all = time.perf_counter()
    spec = MODEL_SPECS[model_key]
    work_dir = Path(work_dir)
    results_dir = Path(results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prof = detect_colab_gpu_profile() if not mock_tiny_for_test else detect_colab_gpu_profile(80.0, "CPU-Mock")
    mp = prof["models"][model_key]
    if not mp.get("supported", True) and not mock_tiny_for_test and quant_override is None:
        raise RuntimeError(f"Model {model_key} unsupported on {prof['gpu_name']} ({prof['vram_gib']} GiB): {mp.get('reason')}")

    quant = str(quant_override or mp.get("quant", spec["default_quant"]))
    eff_eval_bs = int(eval_bs if eval_bs is not None else mp.get("eval_bs", 32))
    eff_bs = int(batch_size if batch_size is not None else mp.get("batch_size", 16))
    eff_mb = int(micro_batch_size if micro_batch_size is not None else mp.get("micro_batch_size", 8))
    eff_ckpt = bool(grad_ckpt if grad_ckpt is not None else mp.get("grad_ckpt", model_key == "27b"))
    do_u1 = bool(run_u1 if run_u1 is not None else mp.get("run_u1_scaled", False))

    host = ScaledQwenCacheHost(
        model_id=spec["model_id"],
        revision=spec["revision"],
        quant=quant,
        mock_tiny_for_test=mock_tiny_for_test,
    )
    if not mock_tiny_for_test:
        assert int(getattr(host.text_config, "vocab_size", 0)) == EXPECTED_4B_VOCAB

    encoder, _ = load_s17_encoder(repo_root)
    kernel_mod = PhysicalKernelS17(encoder.table).to(host.device)
    connector = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)
    connector.requires_grad_(False)
    connector.eval()

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in spec["fa_layers"] if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in spec["dn_layers"] if i < n_l]
    mlp_l = [i for i in spec["mlp_layers"] if i < n_l] or ([0] if n_l >= 1 else [])
    stop_l = min(int(spec["stop_layer"]), max(0, n_l - 2)) if n_l >= 2 else 0

    try:
        # 1. Step-0 Calibration (Dual K-FAC cached cleanly across Colab reconnects)
        norm_errs = {"final_norm": verify_rmsnorm_gamma(_get_final_norm(host.lm))}
        for tidx in range(max(0, n_l - 3), n_l):
            norm_errs[f"layer_{tidx}_post_attn"] = verify_rmsnorm_gamma(layers[tidx].post_attention_layernorm)

        kfac_p = work_dir / f"kfac_{model_key}_dual_64p.pt"
        kfac = None
        if kfac_p.exists() and not mock_tiny_for_test:
            cand = torch.load(kfac_p, map_location="cpu", weights_only=False)
            if int(cand.get("d_model", -1)) == host.hidden_size:
                kfac = cand
        if kfac is None:
            k_prompts = GENERAL_KFAC_64_PROMPTS[:16] if mock_tiny_for_test else GENERAL_KFAC_64_PROMPTS
            kfac = collect_dual_space_kfac_projectors(host, fa_l, dn_l, mlp_l, prompts=k_prompts, stop_layer=stop_l)
            if not mock_tiny_for_test:
                tmp_kp = work_dir / f".{kfac_p.name}.tmp"
                torch.save(kfac, tmp_kp)
                tmp_kp.replace(kfac_p)

        b_align_info = calibrate_b_align_gamma_f(host)
        extra_parts: List[torch.Tensor] = []
        if host.hidden_size >= 512:
            extra_parts.append(connector.out_proj.weight.detach().float().cpu())
            if kfac.get("u_c_tail") is not None and kfac["u_c_tail"].numel() > 0:
                extra_parts.append(_realify_uc(kfac["u_c_tail"], host.hidden_size).float())
        e_bus, bus_stats = build_three_way_orthogonal_bus(
            kfac["u_c_bus"],
            b_align_info["m_eff"],
            bus_dim=22,
            seed=seed,
            extra_block=(torch.cat(extra_parts, dim=1) if extra_parts else None),
        )

        tr_blocks = 1 if mock_tiny_for_test else 8
        dv_blocks = 1 if mock_tiny_for_test else 2
        other_blocks = 1 if mock_tiny_for_test else 8
        tr_groups, _ = build_stage4_pilot_groups(encoder, kernel_mod, train_blocks=tr_blocks, dev_blocks=dv_blocks)

        eval_frozen_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_EVAL_PROMPTS)
        regime, l10_proj, l10_bias, l10_rep = run_l10_ood_theta_probe(
            host, connector, stop_layer=stop_l, seed=seed, mock_tiny_for_test=mock_tiny_for_test
        )
        qgate_w, qgate_b, qgate_diag = calibrate_fisher_qgate(
            host, connector, kfac, stop_layer=stop_l, tr_groups=tr_groups, seed=seed
        )

        def _build_scaled_adapter(arm_mode: str, disable_stop_grad: bool = False) -> CNSFTHostAdapter:
            ad = CNSFTHostAdapter(
                host.lm,
                u_c_d=kfac["u_c_d"],
                u_c_o=kfac["u_c_o"],
                u_c_mlp=kfac["u_c_mlp"],
                u_c_bus=kfac["u_c_bus"],
                b_align_info=b_align_info,
                e_bus=e_bus,
                tail_rms_medians=kfac["tail_rms_medians"],
                fa_layers=fa_l,
                dn_layers=dn_l,
                mlp_tune_layers=mlp_l,
                stop_layer=(-1 if disable_stop_grad else stop_l),
                arm_mode=arm_mode,
                regime=regime,
                l10_ridge_proj=l10_proj,
                l10_ridge_bias=l10_bias,
            ).to(host.device)
            with torch.no_grad():
                ad.schur_readout.qgate_w.copy_(qgate_w.to(host.device))
                ad.schur_readout.qgate_b.copy_(qgate_b.to(host.device))
            if disable_stop_grad:
                _attach_readonly_qgate_hook(host.lm, ad, stop_l)
            return ad

        # 2. U0 Zero-Shot CN-SFT evaluation (Synthetic Blind + Audit + optional Real-SDR)
        suites = _build_blind_suites(encoder, g1_blocks=g1_blocks, other_blocks=other_blocks)
        sdr_cache = work_dir / "real_sdr_cache"
        has_real_sdr = (sdr_cache / "pluto_dual_cache_c16.npz").exists() and (sdr_cache / "cttc_dual_cache_c16.npz").exists()
        domains_bw_prompts = _build_real_domain_specs(sdr_cache) if has_real_sdr else None

        u0 = _build_scaled_adapter("u0_zeroshot")
        try:
            u0_suf_calib = calibrate_template_suffix_bias(host, connector, u0, tr_groups)
            suffix_bias_tbl = u0.circuit_a.template_suffix_canon_bias.detach().cpu().clone()
            u0_audit = audit_cn_sft_pure_and_mix_kl(host, connector, u0, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs)
            from research.stage4_real_and_surgical.cn_sft_probe import _predict_cn_sft_on_suites

            u0_preds = _predict_cn_sft_on_suites(host, connector, suites, u0, eval_bs=eff_eval_bs, scorer_mode="kv_cached")
            u0_blind = _strip_summary(_summarize_model_blind(suites, u0_preds))
            u0_real_sdr = None
            if domains_bw_prompts is not None:
                doms, bw_m, pr_map = domains_bw_prompts
                u0_real_sdr = {
                    "c16": _eval_real_domains_cnsft(host, connector, doms, bw_m, u0, 16, pr_map, eff_eval_bs),
                    "4bit": _eval_real_domains_cnsft(host, connector, doms, bw_m, u0, 4, pr_map, eff_eval_bs),
                }
        finally:
            u0.remove()
            del u0

        # 3. Optional U1 Fine-Tuning on Scaled Model (or reuse U0 for Stage 6 Active EIG)
        u1_report: Optional[dict] = None
        ckpt_for_s6: Optional[Path] = None
        if do_u1:
            train_mix_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_TRAIN_PROMPTS)["mix"]
            tr_minibatches, order = _weighted_phaseB_minibatch_order(tr_groups, batch_size=eff_bs, steps=steps_u1, seed=seed)
            u1 = _build_scaled_adapter("u1_full_cnsft")
            try:
                with torch.no_grad():
                    if u1.circuit_a is not None:
                        u1.circuit_a.template_suffix_canon_bias.copy_(suffix_bias_tbl.to(host.device))
                u1_stats = _train_cn_sft_arm(
                    host,
                    connector,
                    u1,
                    tr_groups,
                    tr_minibatches,
                    order,
                    train_mix_refs,
                    micro_batch_size=eff_mb,
                    lr=1.2e-4,
                    scorer_mode="kv_cached",
                    seed=seed,
                    tag=f"CN-SFT-{model_key.upper()}-{steps_u1}",
                )
                u1_audit = audit_cn_sft_pure_and_mix_kl(host, connector, u1, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs)
                u1_preds = _predict_cn_sft_on_suites(host, connector, suites, u1, eval_bs=eff_eval_bs, scorer_mode="kv_cached")
                u1_blind = _strip_summary(_summarize_model_blind(suites, u1_preds))
                u1_real_sdr = None
                if domains_bw_prompts is not None:
                    doms, bw_m, pr_map = domains_bw_prompts
                    u1_real_sdr = {
                        "c16": _eval_real_domains_cnsft(host, connector, doms, bw_m, u1, 16, pr_map, eff_eval_bs),
                        "4bit": _eval_real_domains_cnsft(host, connector, doms, bw_m, u1, 4, pr_map, eff_eval_bs),
                    }
                ckpt_for_s6 = work_dir / f"stage4_cn_sft_{model_key}_u1_s{seed}.pt"
                torch.save(
                    {
                        "arm": "u1_full_cnsft",
                        "seed": seed,
                        "fa_layers": fa_l,
                        "dn_layers": dn_l,
                        "mlp_layers": mlp_l,
                        "stop_layer": stop_l,
                        "regime": regime,
                        "e_bus": e_bus.cpu(),
                        "b_align_info": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b_align_info.items()},
                        "tail_rms_medians": kfac["tail_rms_medians"],
                        "l10_ridge_proj": l10_proj.cpu(),
                        "l10_ridge_bias": l10_bias.cpu(),
                        "state_dict": {k: v.cpu() for k, v in u1.state_dict().items()},
                    },
                    ckpt_for_s6,
                )
                u1_report = {
                    "steps": steps_u1,
                    "adapter_trainable_params": u1.param_count(),
                    **u1_stats,
                    "audit": u1_audit,
                    "blind": u1_blind,
                    "real_sdr": u1_real_sdr,
                    "checkpoint": ckpt_for_s6.name,
                }
            finally:
                u1.remove()
                del u1
                gc.collect()
                if host.device.type == "cuda":
                    torch.cuda.empty_cache()

        # 4. Stage 6 `m5-stage6-blind-v1` Active EIG Evaluation (Frozen C16/4bit + CN-SFT C16)
        catalogue = verify_m5.geometry_catalogue()
        n_train = 64 if mock_tiny_for_test else 2048
        n_dev = 32 if mock_tiny_for_test else verify_m5.ADAPTIVE_DEV
        n_test = 64 if mock_tiny_for_test else verify_m5.ADAPTIVE_TEST

        dev_bundles = [verify_m5.geometry_split(g, encoder, fid, n_dev, 0) for fid, g in enumerate(catalogue)]
        test_bundles = [verify_m5.geometry_split(g, encoder, fid, n_test, 1) for fid, g in enumerate(catalogue)]
        dev_phys = [verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(dev_bundles)]
        test_phys = [verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(test_bundles)]

        s6_models: Dict[str, Any] = {}
        s6_nlls: Dict[str, np.ndarray] = {}
        s6_hits: Dict[str, np.ndarray] = {}
        s6_models["physical_c16_reference"], s6_nlls["physical_c16_reference"], s6_hits["physical_c16_reference"] = (
            _evaluate_stage6_weights_on_catalogue(catalogue, encoder, n_dev, n_test, dev_phys, test_phys, 0, 1)
        )

        for mname, bits in (("host_frozen_c16", 16), ("host_frozen_s17_4bit", 4)):
            s6_models[mname], s6_nlls[mname], s6_hits[mname] = _eval_stage6_with_adapter(
                host,
                catalogue,
                encoder,
                None,
                n_train=n_train,
                n_dev=n_dev,
                n_test=n_test,
                steps=steps_stage6,
                batch_size=32,
                eval_bs=eff_eval_bs,
                bits=bits,
                seed=seed,
                grad_ckpt=eff_ckpt,
                mname=mname,
            )

        if ckpt_for_s6 is not None and ckpt_for_s6.exists():
            s6_cn_ad = reconstruct_cn_sft_from_checkpoint(host, ckpt_for_s6, disable_stop_grad=True)
        else:
            s6_cn_ad = _build_scaled_adapter("u0_zeroshot", disable_stop_grad=True)
            with torch.no_grad():
                if s6_cn_ad.circuit_a is not None:
                    s6_cn_ad.circuit_a.template_suffix_canon_bias.copy_(suffix_bias_tbl.to(host.device))
        try:
            s6_models["host_cnsft_c16"], s6_nlls["host_cnsft_c16"], s6_hits["host_cnsft_c16"] = _eval_stage6_with_adapter(
                host,
                catalogue,
                encoder,
                s6_cn_ad,
                n_train=n_train,
                n_dev=n_dev,
                n_test=n_test,
                steps=steps_stage6,
                batch_size=32,
                eval_bs=eff_eval_bs,
                bits=16,
                seed=seed,
                grad_ckpt=eff_ckpt,
                mname="host_cnsft_c16",
            )
            aud_s6 = audit_pure_and_mix_kl(host, connector, s6_cn_ad, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=True, frozen_refs=eval_frozen_refs)
            assert aud_s6["passes_pure_kl_zero"], f"Scaled Stage 6 PureKL != 0: {aud_s6}"
            s6_models["host_cnsft_c16"]["pure_kl_audit"] = aud_s6
        finally:
            s6_cn_ad.remove()
            del s6_cn_ad

        assert _compute_scaled_host_fingerprint(host.lm, quant) == host.host_sha256, f"{model_key} host SHA mutated!"

        m_cl = np.arange(len(s6_nlls["host_frozen_c16"]), dtype=np.int64)
        paired_16v4 = {
            "nll_diff_c16_minus_4bit": _cluster_paired_stats(
                np.asarray(s6_nlls["host_frozen_c16"], dtype=np.float64),
                np.asarray(s6_nlls["host_frozen_s17_4bit"], dtype=np.float64),
                m_cl,
                higher_is_better_a=False,
                seed=seed,
            ),
            "acc_diff_c16_minus_4bit": _cluster_paired_stats(
                np.asarray(s6_hits["host_frozen_c16"], dtype=np.float64),
                np.asarray(s6_hits["host_frozen_s17_4bit"], dtype=np.float64),
                m_cl,
                higher_is_better_a=True,
                seed=seed + 1,
            ),
        }

        report = {
            "model_key": model_key,
            "model_id": host.model_id,
            "revision": spec["revision"],
            "quant_mode": quant,
            "quant_audit": host.quant_audit,
            "host_sha256": host.host_sha256,
            "frozen_param_count": host.frozen_param_count,
            "gpu_profile": prof,
            "seed": seed,
            "wall_s": round(time.perf_counter() - t_all, 2),
            "step0": {
                "norm_verify_max_err": norm_errs,
                "b_align_fixed_err": round(b_align_info["fixed_err"], 6),
                "bus_stats": bus_stats,
                "l10_ood_probe": l10_rep,
                "fisher_qgate_calib": qgate_diag,
            },
            "cn_sft_u0": {
                "suffix_calib": u0_suf_calib,
                "audit": u0_audit,
                "blind": u0_blind,
                "real_sdr": u0_real_sdr,
            },
            "cn_sft_u1": u1_report,
            "stage6_active_eig": {
                "wire_id": "m5-c16-v1-le",
                "protocol_id": "m5-stage6-blind-v1",
                "n_test_per_family": n_test,
                "models": s6_models,
                "paired_16bit_vs_4bit_frozen_host": paired_16v4,
            },
        }
        (results_dir / f"stage6_scaling_{model_key}_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        del connector, kernel_mod, host
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
