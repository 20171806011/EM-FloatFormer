"""Frozen Ministral-3-3B-Instruct-2512-BF16 host + tiny trainable integer-cache connector."""
import gc
import hashlib
import inspect
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.schema import (
    Q1_CHOICES,
    Q2_CHOICES,
    Q3_CHOICES,
    Reference,
    matched_sum,
)


MODEL_ID_DEFAULT = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
SYSTEM_PROMPT = "你是电磁波形分析助手。仅根据已编码波形向量、参考导频与问题描述，直接选择最符合条件的一项。"
MARKER_REGEX = re.compile(r"(<W1>|<W2>|<REF>)")


def build_ref_feature(ref: Reference) -> np.ndarray:
    """12-dim public feature vector extracted ONLY from an explicitly revealed Reference object.

    Shape for batched mothers: (M, 12) float32; for single mother (12,) float32.
    """
    samples = np.asarray(ref.samples, dtype=np.complex128)
    t = matched_sum(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
        t = np.atleast_1d(t)
        squeeze = True
    else:
        squeeze = False
    m = len(samples)
    feats = np.stack(
        [
            t.real,
            t.imag,
            np.abs(t),
            samples[:, 0].real,
            samples[:, 0].imag,
            samples[:, 1].real,
            samples[:, 1].imag,
            samples[:, 2].real,
            samples[:, 2].imag,
            np.full(m, 1.0 if ref.shares_channel else 0.0, dtype=np.float64),
            np.full(m, float(ref.t_p), dtype=np.float64),
            np.full(m, float(ref.snr_db) / 10.0, dtype=np.float64),
        ],
        axis=1,
    ).astype(np.float32)
    return feats[0] if squeeze else feats


class WaveformCacheConnector(nn.Module):
    """Maps discrete cache integers (z1, z2) and revealed Reference features to LM hidden vectors.

    Zero access to raw Y1/Y2 waveforms, decoder probability tables, true labels, or Q1 ground truth.
    """

    def __init__(
        self,
        num_states: int,
        hidden_size: int = 3072,
        target_embed_std: float = 0.02,
        emb_dim: int = 64,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.emb_dim = int(emb_dim)

        self.win_emb = nn.Embedding(self.num_states, emb_dim)
        self.role_emb = nn.Embedding(2, emb_dim)
        self.win_mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.win_norm = nn.LayerNorm(hidden_size)
        self.win_scale = nn.Parameter(torch.tensor(float(target_embed_std), dtype=torch.float32))

        self.ref_mlp = nn.Sequential(
            nn.Linear(12, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ref_norm = nn.LayerNorm(hidden_size)
        self.ref_scale = nn.Parameter(torch.tensor(float(target_embed_std), dtype=torch.float32))

    def encode_windows(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ONLY discrete cache indices z1, z2 into `(B, D)` continuous host tokens."""
        slot0 = torch.zeros_like(z1)
        slot1 = torch.ones_like(z2)
        h1 = self.win_norm(self.win_mlp(self.win_emb(z1) + self.role_emb(slot0))) * self.win_scale
        h2 = self.win_norm(self.win_mlp(self.win_emb(z2) + self.role_emb(slot1))) * self.win_scale
        return h1, h2

    def encode_refs(self, ref_feats: torch.Tensor) -> torch.Tensor:
        """Encode revealed reference tensors `(N_refs, 12)` into `(N_refs, D)` host tokens."""
        return self.ref_norm(self.ref_mlp(ref_feats)) * self.ref_scale

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _compute_host_fingerprint(lm: nn.Module) -> str:
    """Deterministic SHA-256 over sampled parameter tensors of frozen host LM."""
    h = hashlib.sha256()
    for name, p in sorted(lm.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("utf-8"))
        arr = p.detach().float()
        stats = torch.stack([arr.mean(), arr.std(unbiased=False), arr.abs().max(), arr.flatten()[0], arr.flatten()[-1]])
        h.update(stats.cpu().numpy().tobytes())
    return h.hexdigest()


def _clone_expand_past(past_key_values, repeats: int):
    """Clone & repeat KV cache along batch dim cleanly without in-place mutation of original cache."""
    from transformers.cache_utils import DynamicCache

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        new_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(legacy):
            new_cache.update(
                k.repeat_interleave(repeats, dim=0),
                v.repeat_interleave(repeats, dim=0),
                layer_idx,
            )
        return new_cache
    # transformers >= 5.0 DynamicCache may store `.layers` with `.keys` and `.values`
    if hasattr(past_key_values, "layers"):
        new_cache = DynamicCache()
        for layer_idx, layer in enumerate(past_key_values.layers):
            new_cache.update(
                layer.keys.repeat_interleave(repeats, dim=0),
                layer.values.repeat_interleave(repeats, dim=0),
                layer_idx,
            )
        return new_cache
    # key_cache / value_cache fallback
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
        new_cache.update(
            k.repeat_interleave(repeats, dim=0),
            v.repeat_interleave(repeats, dim=0),
            layer_idx,
        )
    return new_cache


class MinistralCacheHost:
    """Wrapper holding frozen Ministral-3-3B language model + tokenizer for stage-3 scoring & chat."""

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        mock_tiny_for_test: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if mock_tiny_for_test:
            # Lightweight local unit-test mock (zero HF downloads!)
            from transformers import LlamaConfig, LlamaForCausalLM

            cfg = LlamaConfig(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
            )
            self.dtype = torch.float32
            self.lm = LlamaForCausalLM(cfg).to(self.device)
            self.tokenizer = None
            self.hf_sha = "mock_tiny"
            self.transformers_version = "mock"
            self.candidate_token_audit = {}
        else:
            import transformers
            from huggingface_hub import HfApi
            from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

            self.transformers_version = getattr(transformers, "__version__", "unknown")
            if dtype is None:
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                elif self.device.type == "cuda":
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.float32
            else:
                self.dtype = dtype

            try:
                self.hf_sha = str(HfApi().model_info(model_id).sha)
            except Exception:
                self.hf_sha = "unknown"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

            self.lm = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            text_cfg = getattr(self.lm.config, "text_config", self.lm.config)
            # Guard against HF Transformers Issue #43404 (tie_word_embeddings under text_config)
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

            self.candidate_token_audit = self._audit_candidate_tokenizations()

        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)

        text_cfg = getattr(self.lm.config, "text_config", self.lm.config)
        self.hidden_size = int(text_cfg.hidden_size)
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(self.lm.forward).parameters
        embed_weight = self.lm.get_input_embeddings().weight.detach().float()
        self.target_embed_std = float(embed_weight.std().item())
        self.host_sha256 = _compute_host_fingerprint(self.lm)
        self.frozen_param_count = int(sum(p.numel() for p in self.lm.parameters()))

    def _audit_candidate_tokenizations(self) -> Dict[str, dict]:
        audit: Dict[str, dict] = {}
        for qname, choices in (("Q1", Q1_CHOICES), ("Q2", Q2_CHOICES), ("Q3", Q3_CHOICES)):
            items = []
            for ch in choices:
                ids = self.tokenizer.encode(ch, add_special_tokens=False)
                toks = self.tokenizer.convert_ids_to_tokens(ids)
                items.append({"text": ch, "token_ids": [int(i) for i in ids], "tokens": toks, "length": len(ids)})
            audit[qname] = {
                "choices": items,
                "max_len": max(x["length"] for x in items),
                "all_non_empty": all(x["length"] >= 1 for x in items),
            }
            assert audit[qname]["all_non_empty"], f"Empty candidate tokenization in {qname}"
        return audit

    def format_chat_parts(self, user_prompt: str, extra_history: Optional[List[dict]] = None) -> List[str]:
        """Apply Ministral chat template (or explicit Mistral v7 delimiters if missing) and split placeholders."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if extra_history:
            msgs.extend(extra_history)
        msgs.append({"role": "user", "content": user_prompt})
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            chunks = [f"<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT]"]
            if extra_history:
                for m in extra_history:
                    if m["role"] == "user":
                        chunks.append(f"[INST]{m['content']}[/INST]")
                    elif m["role"] == "assistant":
                        chunks.append(f"{m['content']}</s>")
            chunks.append(f"[INST]{user_prompt}[/INST]")
            rendered = "".join(chunks)
        parts = MARKER_REGEX.split(rendered)
        return [p for p in parts if p != ""]

    def build_prefix_embeds(
        self,
        connector: Optional[WaveformCacheConnector],
        user_prompt: str,
        z1_batch: Optional[torch.Tensor],
        z2_batch: Optional[torch.Tensor],
        ref_feats_list: Sequence[torch.Tensor] = (),
        text_only_mode: bool = False,
        extra_history: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Assemble `(B, L, D)` prefix embeddings with identical shape for text-only vs waveform."""
        parts = self.format_chat_parts(user_prompt, extra_history=extra_history)
        embed_layer = self.lm.get_input_embeddings()
        batch_size = int(z1_batch.shape[0]) if z1_batch is not None else 1

        if text_only_mode or connector is None:
            e_w1 = torch.zeros((batch_size, 1, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w2 = torch.zeros((batch_size, 1, self.hidden_size), device=self.device, dtype=self.dtype)
            e_refs = [
                torch.zeros((batch_size, 1, self.hidden_size), device=self.device, dtype=self.dtype)
                for _ in ref_feats_list
            ]
        else:
            w1, w2 = connector.encode_windows(z1_batch.to(self.device), z2_batch.to(self.device))
            e_w1 = w1.to(self.dtype).unsqueeze(1)
            e_w2 = w2.to(self.dtype).unsqueeze(1)
            e_refs = [connector.encode_refs(rf.to(self.device)).to(self.dtype).unsqueeze(1) for rf in ref_feats_list]

        chunks: List[torch.Tensor] = []
        ref_ptr = 0
        for seg in parts:
            if seg == "<W1>":
                if not text_only_mode:
                    chunks.append(e_w1)
            elif seg == "<W2>":
                if not text_only_mode:
                    chunks.append(e_w2)
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

    def score_candidates_kv(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
        use_kv_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact full multi-token candidate sequence log-likelihood scoring using differentiable KV cache.

        Returns `(log_probs_normalized (B, 4), raw_seq_loglik (B, 4))` in float32.
        """
        if not use_kv_cache:
            return self.score_candidates_uncached(prefix_embeds, choices, temperature=temperature)
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()

        cand_id_lists = [self.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
        max_len = max(len(ids) for ids in cand_id_lists)

        # Prefill prefix once cleanly with explicit attention_mask and position_ids
        pref_attn = torch.ones((b, pref_l), device=self.device, dtype=torch.long)
        pref_pos = torch.arange(0, pref_l, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        pref_kwargs = dict(
            inputs_embeds=prefix_embeds,
            attention_mask=pref_attn,
            position_ids=pref_pos,
            use_cache=(max_len > 1),
        )
        if self._supports_logits_to_keep:
            pref_kwargs["logits_to_keep"] = 1
        pref_out = self.lm(**pref_kwargs)
        first_logits = F.log_softmax(pref_out.logits[:, -1, :].float(), dim=-1)  # (B, V)

        seq_loglik = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        first_tok_ids = torch.tensor([ids[0] for ids in cand_id_lists], device=self.device, dtype=torch.long)
        seq_loglik += first_logits[:, first_tok_ids]  # (B, 4)

        if max_len > 1:
            # Score remaining tokens (k=0..3, length max_len - 1) in one parallel batched pass over cloned cache!
            pad_id = int(self.tokenizer.pad_token_id or 0)
            step_len = max_len - 1
            inp_ids = torch.full((4, step_len), pad_id, device=self.device, dtype=torch.long)
            tgt_ids = torch.full((4, step_len), 0, device=self.device, dtype=torch.long)
            tgt_mask = torch.zeros((4, step_len), device=self.device, dtype=torch.float32)
            for k, ids in enumerate(cand_id_lists):
                if len(ids) > 1:
                    l_rem = len(ids) - 1
                    inp_ids[k, :l_rem] = torch.tensor(ids[:-1], device=self.device, dtype=torch.long)
                    tgt_ids[k, :l_rem] = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
                    tgt_mask[k, :l_rem] = 1.0

            # Expand per-batch prefix KV cache cleanly across 4 candidates: order [b0_c0, b0_c1, b0_c2, b0_c3, b1_c0, ...]
            exp_cache = _clone_expand_past(pref_out.past_key_values, repeats=4)
            inp_exp = inp_ids.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len)
            cand_embeds = embed_layer(inp_exp)
            cand_attn = torch.ones((b * 4, pref_l + step_len), device=self.device, dtype=torch.long)
            cand_pos = (
                torch.arange(pref_l, pref_l + step_len, device=self.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(b * 4, -1)
            )
            cand_out = self.lm(
                inputs_embeds=cand_embeds,
                past_key_values=exp_cache,
                attention_mask=cand_attn,
                position_ids=cand_pos,
                use_cache=False,
            )
            cand_lp = F.log_softmax(cand_out.logits.float(), dim=-1)  # (B*4, step_len, V)

            tgt_exp = tgt_ids.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len, 1)
            tok_lp = torch.gather(cand_lp, dim=-1, index=tgt_exp).squeeze(-1).reshape(b, 4, step_len)
            seq_loglik = seq_loglik + torch.sum(tok_lp * tgt_mask.unsqueeze(0), dim=-1)

        probs_log = F.log_softmax(seq_loglik / float(temperature), dim=-1)
        return probs_log, seq_loglik

    def score_candidates_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference uncached full-sequence candidate scorer used to verify KV-cache equivalence."""
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        raw_ll = []
        for ch in choices:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            c_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
            c_emb = embed_layer(c_ids)
            full_emb = torch.cat([prefix_embeds, c_emb], dim=1)
            logits = self.lm(inputs_embeds=full_emb, use_cache=False).logits.float()
            # Token ids[0] predicted at pref_l - 1; ids[t] predicted at pref_l - 1 + t
            pred_logits = logits[:, pref_l - 1 : pref_l - 1 + len(ids), :]
            lp = F.log_softmax(pred_logits, dim=-1)
            tok_lp = torch.gather(lp, dim=-1, index=c_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            raw_ll.append(tok_lp)
        seq_loglik = torch.stack(raw_ll, dim=-1)
        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    @torch.no_grad()
    def generate_text(self, prefix_embeds: torch.Tensor, max_new_tokens: int = 80) -> str:
        """Greedy Chinese decode from prefix_embeds via genuine autoregressive generation."""
        b, seq_len, _ = prefix_embeds.shape
        attn = torch.ones((b, seq_len), device=self.device, dtype=torch.long)
        out_ids = self.lm.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=int(self.tokenizer.pad_token_id or 0),
        )
        # Note: when generating from inputs_embeds, HF returns ONLY newly generated tokens (or check length)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
