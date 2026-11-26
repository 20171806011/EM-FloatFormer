"""Frozen Qwen3.5-4B host wrapper and paired SignedPhaseCacheConnector for Stage 3D."""
import gc
import hashlib
import inspect
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.ministral_connector import (
    _compute_host_fingerprint,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    Q1_CHOICES,
    Q2_CHOICES,
    Q3_CHOICES,
)


MODEL_ID_DEFAULT = "Qwen/Qwen3.5-4B"
PINNED_REVISION_DEFAULT = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SYSTEM_PROMPT = (
    "你是电磁波形分析助手。仅根据已编码波形向量、公开相位向量、参考导频与问题描述，"
    "直接选择最符合条件的一项。"
)
MARKER_REGEX = re.compile(r"(<W1>|<W2>|<PHASE>|<REF>)")


class SignedPhaseCacheConnector(nn.Module):
    """Identical architecture & parameter count for both phase and zero arms.

    Takes discrete integer cache indices (z1, z2), 6 public phase slots, and revealed
    12-dim reference features. Zero access to raw IQ waveforms, labels, or Brownian paths.
    """

    def __init__(
        self,
        num_states: int = 16,
        hidden_size: int = 2560,
        target_embed_std: float = 0.02,
        emb_dim: int = 128,
        mlp_hidden: int = 256,
        num_win_tokens: int = 4,
        num_phase_tokens: int = 4,
        num_ref_tokens: int = 4,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.target_embed_std = float(target_embed_std)
        self.emb_dim = int(emb_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.num_win_tokens = int(num_win_tokens)
        self.num_phase_tokens = int(num_phase_tokens)
        self.num_ref_tokens = int(num_ref_tokens)
        self.num_prefix_tokens = int(num_prefix_tokens)

        self.win_emb = nn.Embedding(self.num_states, emb_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.win1_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.win2_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(6, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_phase_tokens * mlp_hidden),
        )
        self.ref_mlp = nn.Sequential(
            nn.Linear(12, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_ref_tokens * mlp_hidden),
        )
        self.prefix_latents = nn.Parameter(torch.randn(num_prefix_tokens, mlp_hidden) * 0.02)
        self.out_proj = nn.Linear(mlp_hidden, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)

        init_scale = max(0.05, 2.0 * float(target_embed_std))
        self.win_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.phase_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.ref_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.prefix_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def encode_windows(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.win_emb(z1)
        e2 = self.win_emb(z2)
        joint = self.pair_mlp(torch.cat([e1, e2, e1 - e2, e1 * e2], dim=-1))
        u1 = self.win1_head(torch.cat([e1, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        u2 = self.win2_head(torch.cat([e2, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        h1 = self.out_norm(self.out_proj(u1)) * self.win_scale
        h2 = self.out_norm(self.out_proj(u2)) * self.win_scale
        return h1, h2

    def encode_phase(self, phase_slots: torch.Tensor) -> torch.Tensor:
        up = self.phase_mlp(phase_slots).view(-1, self.num_phase_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(up)) * self.phase_scale

    def encode_refs(self, ref_feats: torch.Tensor) -> torch.Tensor:
        ur = self.ref_mlp(ref_feats).view(-1, self.num_ref_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(ur)) * self.ref_scale

    def encode_prefix(self, batch_size: int) -> torch.Tensor:
        p = self.out_norm(self.out_proj(self.prefix_latents)) * self.prefix_scale
        return p.unsqueeze(0).expand(batch_size, -1, -1)

    def connector_config(self) -> dict:
        return {
            "num_states": self.num_states,
            "hidden_size": self.hidden_size,
            "target_embed_std": self.target_embed_std,
            "emb_dim": self.emb_dim,
            "mlp_hidden": self.mlp_hidden,
            "num_win_tokens": self.num_win_tokens,
            "num_phase_tokens": self.num_phase_tokens,
            "num_ref_tokens": self.num_ref_tokens,
            "num_prefix_tokens": self.num_prefix_tokens,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_OOP_LA_CLS = None


def _get_oop_linear_attention_layer_cls():
    """Return cached out-of-place LinearAttentionLayer subclass without per-instance closures."""
    global _OOP_LA_CLS
    if _OOP_LA_CLS is None:
        from transformers.cache_utils import LinearAttentionLayer

        class _OutOfPlaceLinearAttentionLayer(LinearAttentionLayer):
            def update_conv_state(
                self,
                conv_states: torch.Tensor,
                state_idx: int = 0,
                conv_kernel_size: Optional[int] = None,
                **kwargs,
            ) -> torch.Tensor:
                if not self.is_conv_states_initialized[state_idx]:
                    ksize = conv_states.shape[-1] if conv_kernel_size is None else int(conv_kernel_size)
                    self.conv_kernel_size[state_idx] = ksize
                    self.dtype, self.device = conv_states.dtype, conv_states.device
                    self.is_conv_states_initialized[state_idx] = True
                if not self.has_previous_state[state_idx]:
                    full = conv_states
                    self.has_previous_state[state_idx] = True
                    ksize = self.conv_kernel_size[state_idx]
                    if full.shape[-1] < ksize:
                        full = F.pad(full, (ksize - full.shape[-1], 0), value=0)
                else:
                    full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
                    ksize = self.conv_kernel_size[state_idx]
                self.conv_states[state_idx] = full[..., -ksize:]
                return full

            def update_recurrent_state(
                self,
                recurrent_states: torch.Tensor,
                state_idx: int = 0,
                **kwargs,
            ) -> torch.Tensor:
                if not self.is_recurrent_states_initialized[state_idx]:
                    if self.device is None:
                        self.dtype, self.device = recurrent_states.dtype, recurrent_states.device
                    self.is_recurrent_states_initialized[state_idx] = True
                self.recurrent_states[state_idx] = recurrent_states
                return recurrent_states

        _OOP_LA_CLS = _OutOfPlaceLinearAttentionLayer
    return _OOP_LA_CLS


def _make_differentiable_qwen35_cache(lm_config):
    """Build a DynamicCache whose LinearAttention layers update out-of-place without reference cycles."""
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer

    cache = DynamicCache(config=lm_config)
    oop_cls = _get_oop_linear_attention_layer_cls()
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            assert type(layer) is LinearAttentionLayer, f"Unexpected LinearAttentionLayer subclass: {type(layer)}"
            layer.__class__ = oop_cls
    return cache


def _clone_expand_qwen35_cache(pref_cache, lm_config, repeats: int):
    """Clone and expand Qwen3.5 hybrid DynamicCache cleanly (differentiable & non-mutating)."""
    new_cache = _make_differentiable_qwen35_cache(lm_config)
    for old_layer, new_layer in zip(pref_cache.layers, new_cache.layers):
        if hasattr(old_layer, "conv_states") and hasattr(old_layer, "recurrent_states"):
            new_layer.dtype = old_layer.dtype
            new_layer.device = old_layer.device
            for s_idx, cs in old_layer.conv_states.items():
                if cs is not None:
                    new_layer.conv_kernel_size[s_idx] = old_layer.conv_kernel_size[s_idx]
                    new_layer.conv_states[s_idx] = cs.repeat_interleave(repeats, dim=0)
                    new_layer.is_conv_states_initialized[s_idx] = True
                    new_layer.has_previous_state[s_idx] = old_layer.has_previous_state[s_idx]
            for s_idx, rs in old_layer.recurrent_states.items():
                if rs is not None:
                    new_layer.recurrent_states[s_idx] = rs.repeat_interleave(repeats, dim=0)
                    new_layer.is_recurrent_states_initialized[s_idx] = True
        elif hasattr(old_layer, "keys") and hasattr(old_layer, "values"):
            if old_layer.keys is not None and old_layer.values is not None:
                new_layer.update(
                    old_layer.keys.repeat_interleave(repeats, dim=0),
                    old_layer.values.repeat_interleave(repeats, dim=0),
                )
        else:
            raise TypeError(f"Unsupported cache layer type: {type(old_layer)}")
    return new_cache


class QwenCacheHost:
    """Frozen Qwen3.5-4B host wrapper using native chat template and enable_thinking=False."""

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        revision: Optional[str] = PINNED_REVISION_DEFAULT,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        mock_tiny_for_test: bool = False,
    ):
        self.model_id = model_id
        self.requested_revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if mock_tiny_for_test:
            from transformers import Qwen2Config, Qwen2ForCausalLM

            cfg = Qwen2Config(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
            )
            self.dtype = torch.float32
            self.lm = Qwen2ForCausalLM(cfg).to(self.device)
            self.tokenizer = None
            self.hf_sha = "mock_tiny"
            self.transformers_version = "mock"
            self.chat_template_sha256 = "mock"
            self.thinking_mode = "mock"
            self.candidate_token_audit = {}
            self._is_qwen35_mrope = False
        else:
            import transformers
            from huggingface_hub import HfApi
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

            self.transformers_version = getattr(transformers, "__version__", "unknown")
            if dtype is None:
                if self.device.type == "cuda":
                    cap_major = torch.cuda.get_device_capability(self.device)[0]
                    self.dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
                else:
                    self.dtype = torch.float32
            else:
                self.dtype = dtype

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

            try:
                lm = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    revision=revision,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
            except TypeError:
                lm = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    revision=revision,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )

            # Drop unused vision encoder cleanly on CPU before moving language backbone to GPU
            if hasattr(lm, "model") and hasattr(lm.model, "visual"):
                lm.model.visual = None
            if hasattr(lm, "visual"):
                lm.visual = None
            gc.collect()

            self.lm = lm.to(device=self.device, dtype=self.dtype)
            actual_dtype = next(self.lm.parameters()).dtype
            assert actual_dtype == self.dtype, f"Loaded dtype {actual_dtype} != {self.dtype}"
            text_cfg = getattr(self.lm.config, "text_config", self.lm.config)
            if getattr(text_cfg, "tie_word_embeddings", False) and hasattr(self.lm, "lm_head"):
                emb = self.lm.get_input_embeddings()
                if emb is not None and self.lm.lm_head.weight.data_ptr() != emb.weight.data_ptr():
                    if hasattr(self.lm, "tie_weights"):
                        self.lm.tie_weights()
                    if self.lm.lm_head.weight.data_ptr() != emb.weight.data_ptr():
                        self.lm.lm_head.weight = emb.weight
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self._is_qwen35_mrope = True
            self.candidate_token_audit = self._audit_candidate_tokenizations()

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
        self.host_sha256 = _compute_host_fingerprint(self.lm)
        self.frozen_param_count = int(sum(p.numel() for p in self.lm.parameters()))

    def set_grad_checkpointing(self, enabled: bool) -> None:
        """Toggle activation checkpointing on frozen host LM for low-VRAM connector training."""
        if enabled:
            if hasattr(self.lm, "gradient_checkpointing_enable"):
                try:
                    self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    self.lm.gradient_checkpointing_enable()
            self.lm.train()
        else:
            if hasattr(self.lm, "gradient_checkpointing_disable"):
                self.lm.gradient_checkpointing_disable()
            self.lm.eval()

    def _audit_candidate_tokenizations(self) -> Dict[str, dict]:
        sample_prefix = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "测试"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix_ids = self.tokenizer.encode(sample_prefix, add_special_tokens=False)
        audit: Dict[str, dict] = {
            "sample_rendered_tail": sample_prefix[-64:],
            "ends_with_disabled_think": sample_prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        }
        assert audit["ends_with_disabled_think"], f"Unexpected chat template suffix: {sample_prefix[-80:]!r}"
        for qname, choices in (("Q1", Q1_CHOICES), ("Q2", Q2_CHOICES), ("Q3", Q3_CHOICES)):
            items = []
            for idx, ch in enumerate(choices):
                ids = self.tokenizer.encode(ch, add_special_tokens=False)
                toks = self.tokenizer.convert_ids_to_tokens(ids)
                full_ids = self.tokenizer.encode(sample_prefix + ch, add_special_tokens=False)
                concat_exact = full_ids == (prefix_ids + ids)
                items.append(
                    {
                        "physical_index": idx,
                        "text": ch,
                        "token_ids": [int(i) for i in ids],
                        "tokens": toks,
                        "length": len(ids),
                        "boundary_concat_exact": bool(concat_exact),
                    }
                )
            audit[qname] = {
                "choices": items,
                "max_len": max(x["length"] for x in items),
                "all_non_empty": all(x["length"] >= 1 for x in items),
                "all_boundary_exact": all(x["boundary_concat_exact"] for x in items),
            }
            assert audit[qname]["all_non_empty"], f"Empty candidate tokenization in {qname}"
            assert audit[qname]["all_boundary_exact"], f"Boundary merge detected in {qname}: {items}"
        return audit

    def format_chat_parts(self, user_prompt: str, extra_history: Optional[List[dict]] = None) -> List[str]:
        """Render using official Qwen3.5 chat_template with thinking disabled, then split on modality markers."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if extra_history:
            messages.extend(extra_history)
        messages.append({"role": "user", "content": user_prompt})
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        parts = MARKER_REGEX.split(rendered)
        return [p for p in parts if p != ""]

    def _make_position_ids(self, pos_1d: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Build explicit position_ids avoiding stale self.model.rope_deltas."""
        # pos_1d: (L,) or (B, L)
        if pos_1d.ndim == 1:
            p2 = pos_1d.unsqueeze(0).expand(batch_size, -1)
        else:
            p2 = pos_1d
        if self._is_qwen35_mrope:
            # 4 axes: (text, temporal, height, width) -> identical 1D positions for text/connector embeddings
            return p2.unsqueeze(0).expand(4, batch_size, -1)
        return p2

    def build_prefix_embeds(
        self,
        connector: Optional[SignedPhaseCacheConnector],
        user_prompt: str,
        z1_batch: Optional[torch.Tensor],
        z2_batch: Optional[torch.Tensor],
        phase_slots_batch: Optional[torch.Tensor] = None,
        ref_feats_list: Sequence[torch.Tensor] = (),
        text_only_mode: bool = False,
        extra_history: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Assemble `(B, L, D)` prefix embeddings with zero leakage."""
        parts = self.format_chat_parts(user_prompt, extra_history=extra_history)
        embed_layer = self.lm.get_input_embeddings()
        batch_size = int(z1_batch.shape[0]) if z1_batch is not None else 1

        if text_only_mode or connector is None:
            e_pref = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w1 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w2 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_phase = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_refs = [
                torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
                for _ in ref_feats_list
            ]
        else:
            e_pref = connector.encode_prefix(batch_size).to(self.dtype)
            w1, w2 = connector.encode_windows(z1_batch.to(self.device), z2_batch.to(self.device))
            e_w1 = w1.to(self.dtype)
            e_w2 = w2.to(self.dtype)
            if phase_slots_batch is None:
                ps = torch.zeros((batch_size, 6), device=self.device, dtype=torch.float32)
            else:
                ps = phase_slots_batch.to(self.device, dtype=torch.float32)
            e_phase = connector.encode_phase(ps).to(self.dtype)
            e_refs = [connector.encode_refs(rf.to(self.device, dtype=torch.float32)).to(self.dtype) for rf in ref_feats_list]

        chunks: List[torch.Tensor] = []
        ref_ptr = 0
        prefix_inserted = False
        for seg in parts:
            if seg == "<W1>":
                if not text_only_mode:
                    if not prefix_inserted:
                        chunks.append(e_pref)
                        prefix_inserted = True
                    chunks.append(e_w1)
            elif seg == "<W2>":
                if not text_only_mode:
                    chunks.append(e_w2)
            elif seg == "<PHASE>":
                if not text_only_mode:
                    chunks.append(e_phase)
            elif seg == "<REF>":
                if ref_ptr >= len(e_refs):
                    raise ValueError("More <REF> markers in prompt than ref_feats_list entries")
                if not text_only_mode:
                    chunks.append(e_refs[ref_ptr])
                ref_ptr += 1
            else:
                ids = self.tokenizer.encode(seg, add_special_tokens=False)
                if ids:
                    t_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                    chunks.append(embed_layer(t_ids))
        if not text_only_mode and ref_ptr != len(e_refs):
            raise ValueError(f"Unused ref_feats_list entries: used {ref_ptr} of {len(e_refs)}")
        return torch.cat(chunks, dim=1)

    def _lm_forward(self, **kwargs):
        if self._uses_cache_kwarg:
            if "past_key_values" in kwargs:
                kwargs["cache"] = kwargs.pop("past_key_values")
        else:
            if "cache" in kwargs:
                kwargs["past_key_values"] = kwargs.pop("cache")
        return self.lm(**kwargs)

    def score_candidates_naive_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference uncached per-candidate forward loop (zero padding, zero cache)."""
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        raw_ll = []
        for ch in choices:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            c_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
            c_emb = embed_layer(c_ids)
            full_emb = torch.cat([prefix_embeds, c_emb], dim=1)
            seq_len = full_emb.shape[1]
            attn = torch.ones((b, seq_len), device=self.device, dtype=torch.long)
            pos = self._make_position_ids(torch.arange(seq_len, device=self.device, dtype=torch.long), b)
            fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=attn, position_ids=pos, use_cache=False)
            logits = self._lm_forward(**fwd_kwargs).logits.float()
            pred_logits = logits[:, pref_l - 1 : pref_l - 1 + len(ids), :]
            lp = F.log_softmax(pred_logits, dim=-1)
            tok_lp = torch.gather(lp, dim=-1, index=c_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            raw_ll.append(tok_lp)
        seq_loglik = torch.stack(raw_ll, dim=-1)
        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    def score_candidates_batched_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact single-pass batched uncached scoring over (B*4, pref_l + max_len) — clean autograd for hybrid DeltaNet."""
        assert len(choices) == 4
        b, pref_l, d_hid = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        cand_id_lists = [self.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
        max_len = max(len(ids) for ids in cand_id_lists)
        pad_id = int(self.tokenizer.pad_token_id or 0)

        cand_ids = torch.full((4, max_len), pad_id, device=self.device, dtype=torch.long)
        cand_attn = torch.zeros((4, max_len), device=self.device, dtype=torch.long)
        tgt_mask = torch.zeros((4, max_len), device=self.device, dtype=torch.float32)
        for k, ids in enumerate(cand_id_lists):
            l_k = len(ids)
            cand_ids[k, :l_k] = torch.tensor(ids, device=self.device, dtype=torch.long)
            cand_attn[k, :l_k] = 1
            tgt_mask[k, :l_k] = 1.0

        pref_exp = prefix_embeds.unsqueeze(1).expand(b, 4, pref_l, d_hid).reshape(b * 4, pref_l, d_hid)
        cand_exp_ids = cand_ids.unsqueeze(0).expand(b, 4, max_len).reshape(b * 4, max_len)
        cand_exp_emb = embed_layer(cand_exp_ids)
        full_emb = torch.cat([pref_exp, cand_exp_emb], dim=1)

        pref_attn = torch.ones((b * 4, pref_l), device=self.device, dtype=torch.long)
        full_attn = torch.cat([pref_attn, cand_attn.unsqueeze(0).expand(b, 4, max_len).reshape(b * 4, max_len)], dim=1)
        total_len = pref_l + max_len
        pos = self._make_position_ids(torch.arange(total_len, device=self.device, dtype=torch.long), b * 4)

        fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=full_attn, position_ids=pos, use_cache=False)
        if self._supports_logits_to_keep:
            fwd_kwargs["logits_to_keep"] = max_len + 1
            out = self._lm_forward(**fwd_kwargs)
            pred_logits = out.logits[:, :max_len, :]
        else:
            out = self._lm_forward(**fwd_kwargs)
            pred_logits = out.logits[:, pref_l - 1 : pref_l - 1 + max_len, :]
        del out
        lp = F.log_softmax(pred_logits, dim=-1, dtype=torch.float32)
        del pred_logits
        tok_lp = torch.gather(lp, dim=-1, index=cand_exp_ids.unsqueeze(-1)).squeeze(-1).reshape(b, 4, max_len)
        del lp
        seq_loglik = torch.sum(tok_lp * tgt_mask.unsqueeze(0), dim=-1)
        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    def score_candidates_kv(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable cached scoring using out-of-place Qwen3.5 hybrid DynamicCache."""
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        cand_id_lists = [self.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
        max_len = max(len(ids) for ids in cand_id_lists)

        pref_attn = torch.ones((b, pref_l), device=self.device, dtype=torch.long)
        pref_pos = self._make_position_ids(torch.arange(pref_l, device=self.device, dtype=torch.long), b)
        pref_cache = _make_differentiable_qwen35_cache(self.lm.config) if max_len > 1 else None
        pref_kwargs = dict(
            inputs_embeds=prefix_embeds,
            attention_mask=pref_attn,
            position_ids=pref_pos,
            use_cache=(max_len > 1),
            cache=pref_cache,
        )
        if self._supports_logits_to_keep:
            pref_kwargs["logits_to_keep"] = 1
        pref_out = self._lm_forward(**pref_kwargs)
        first_logits = F.log_softmax(pref_out.logits[:, -1, :], dim=-1, dtype=torch.float32)

        seq_loglik = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        first_tok_ids = torch.tensor([ids[0] for ids in cand_id_lists], device=self.device, dtype=torch.long)
        seq_loglik += first_logits[:, first_tok_ids]
        del first_logits

        if max_len > 1:
            pad_id = int(self.tokenizer.pad_token_id or 0)
            step_len = max_len - 1
            inp_ids = torch.full((4, step_len), pad_id, device=self.device, dtype=torch.long)
            cand_attn_rem = torch.zeros((4, step_len), device=self.device, dtype=torch.long)
            tgt_ids = torch.full((4, step_len), 0, device=self.device, dtype=torch.long)
            tgt_mask = torch.zeros((4, step_len), device=self.device, dtype=torch.float32)
            for k, ids in enumerate(cand_id_lists):
                if len(ids) > 1:
                    l_rem = len(ids) - 1
                    inp_ids[k, :l_rem] = torch.tensor(ids[:-1], device=self.device, dtype=torch.long)
                    cand_attn_rem[k, :l_rem] = 1
                    tgt_ids[k, :l_rem] = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
                    tgt_mask[k, :l_rem] = 1.0

            out_cache = getattr(pref_out, "cache", None)
            if out_cache is None:
                out_cache = getattr(pref_out, "past_key_values")
            exp_cache = _clone_expand_qwen35_cache(out_cache, self.lm.config, repeats=4)
            del pref_out, out_cache, pref_cache
            inp_exp = inp_ids.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len)
            cand_embeds = embed_layer(inp_exp)
            cand_attn = torch.cat(
                [
                    torch.ones((b * 4, pref_l), device=self.device, dtype=torch.long),
                    cand_attn_rem.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len),
                ],
                dim=1,
            )
            cand_pos = self._make_position_ids(
                torch.arange(pref_l, pref_l + step_len, device=self.device, dtype=torch.long), b * 4
            )
            cand_out = self._lm_forward(
                inputs_embeds=cand_embeds,
                cache=exp_cache,
                attention_mask=cand_attn,
                position_ids=cand_pos,
                use_cache=False,
            )
            del exp_cache
            cand_lp = F.log_softmax(cand_out.logits, dim=-1, dtype=torch.float32)
            del cand_out
            tgt_exp = tgt_ids.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len, 1)
            tok_lp = torch.gather(cand_lp, dim=-1, index=tgt_exp).squeeze(-1).reshape(b, 4, step_len)
            del cand_lp
            seq_loglik = seq_loglik + torch.sum(tok_lp * tgt_mask.unsqueeze(0), dim=-1)
        else:
            del pref_out, pref_cache

        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    @torch.no_grad()
    def generate_text(self, prefix_embeds: torch.Tensor, max_new_tokens: int = 48) -> str:
        """Deterministic greedy decode from prefix_embeds."""
        b, seq_len, _ = prefix_embeds.shape
        attn = torch.ones((b, seq_len), device=self.device, dtype=torch.long)
        if hasattr(self.lm, "model") and hasattr(self.lm.model, "rope_deltas"):
            self.lm.model.rope_deltas = torch.zeros((b, 1), dtype=torch.long, device=self.device)
        out_ids = self.lm.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=int(self.tokenizer.pad_token_id or 0),
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
