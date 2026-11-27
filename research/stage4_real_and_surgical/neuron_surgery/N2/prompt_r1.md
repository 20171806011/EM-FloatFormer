# Track N2 (Round 1): Inter-Token Bridge & Intrinsic Gate Connecting Neurons in Hybrid GatedDeltaNet + Full Attention

You are **Track N2**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **From mathematics first, can directly modifying "connecting neurons" inside `Qwen3.5-4B` / `9B` / `27B` (sparse GatedDeltaNet recurrent gate neurons, Full-Attention interleaved output-gate / QK routing neurons, and readout bridges) unlock OOD routing & replace external gating hooks?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N2/` **ONLY**.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `N2/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Run Python **ONLY** via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
- **Exact Deliverables in `N2/`**:
  1. `research/stage4_real_and_surgical/neuron_surgery/N2/THEOREM_N2.md`
  2. `research/stage4_real_and_surgical/neuron_surgery/N2/verify_n2.py` (must exit `0`)

## 1. Grounded Empirical Context (Read-Only Inspect First)
Inspect read-only as needed:
- `research/stage4_real_and_surgical/surgical_adapter.py` (note how `SurgicalHostAdapter` touches `FA` layers `q_proj` interleaved gate, `k_proj`, `v_proj`, `o_proj`, and `DN` layers `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj`)
- `research/stage3d_qwen_signed_phase/qwen_connector.py` (Qwen3.5 hybrid `GatedDeltaNet` + `Full Attention` structure, `hidden_size=2560`)
- `research/stage4_real_and_surgical/results/stage4_synthetic_blind_results.json`

Verified Phase II-C telemetry on `Qwen3.5-4B` (`32` layers = `24` GatedDeltaNet + `8` Full Attention, `d=2560`, `TIER_P1_MAX_PARAMS = 2,184,964`):
- `equivariant_host_pure` (frozen 4B): Blind `G1 (50,304 rows) 24.48% / 1.3259`, Dev `Q3_cfo 46.88%`, `G1` slot-flip KL `0.1795`.
- `mode1_surgical_96` (W1–W4 Cayley+LoRA on Attention/DN projections, `1.24M` params): Dev `Q3_cfo 57.81%` (**+10.94 pp**), Blind `G1 31.36% / 1.3197`, `G3 33.01%`, slot-flip KL `0.3389`, `PureKL = 0.0` (via external Python hook flag `g_em`), `MixKL = 0.00419`.
- `lemmo_tier_p1` (`1.15M` unconstrained LoRA): Blind `G1 37.04% / 1.2694`, `pure_max_logit_diff = 0.50` (`PureKL != 0`).
- `direct_physical` / `equivariant_physical_residual`: Blind `G1 64.62% / 0.9411`.

**Why Inter-Token Connecting Neurons Matter**:
MLP neurons act strictly **token-wise** ($x_t \mapsto \text{MLP}(x_t)$). To fuse information from soft prefix tokens ($t \in \mathcal{M}_{\text{EM}}$), across multi-turn prompt prose of length $L$, or to candidate choice positions $t_{\text{cand}}$ (including MRoPE / KV-cached suffix tokens!), signals MUST travel through **inter-token connecting neurons**: GatedDeltaNet recurrent state heads/gates and Softmax Attention Q/K/V/O + interleaved gate neurons!

## 2. Deep Mathematical Tasks for Track N2
1. **GatedDeltaNet Exact State-Highway Theorem (`in_proj_a/b`, `A_log`, `dt_bias`, `conv1d`, `out_proj`)**:
   - Write down the exact discrete recurrence of Qwen3.5's `GatedDeltaNet`:
     $$g_t = \exp\!\big(-\exp(A_{\log})\,\operatorname{softplus}(a_t + b_{\text{dt}})\big),\quad \beta_t = \sigma(b_t),\quad S_t = g_t S_{t-1} (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.$$
   - Note that each scalar head $h \in \{0,\dots,H_v-1\}$ has its own decay gate neuron $a_t^{(h)} = w_{a,h}^\top x_t$ and write-strength neuron $b_t^{(h)} = w_{b,h}^\top x_t$, plus depthwise `conv1d`!
   - Derive exact conditions under which modifying **just $r_{\text{hd}} \in \{1, 2\}$ dedicated DeltaNet heads' gate/key/value neurons** implements a **zero-forgetting (`g_t -> 1`, `\beta_t -> 0` on prose tokens, `\beta_t -> 1` on EM tokens) complex-state transport channel** across arbitrary prose length $L$, while remaining orthogonal to text heads (`MixKL -> 0`)! Quantify `conv1d` short-kernel bleed across the EM/prose boundary!
2. **Intrinsic Weight-Compiled Trigger Neuron (Eliminating External Hook `g_em`)**:
   - In W1–W4, `PureKL = 0.0` relied on Python wrapper `g_em in {0, 1}`!
   - Can individual **connecting neurons** inside Qwen3.5 implement a **self-contained, hook-free intrinsic trigger** using Qwen3.5's Full-Attention **interleaved per-head output gate (`q_proj` gate half)** and/or thresholded SwiGLU/key-query detector neurons?
   - Formalize the exact separation condition between natural language prefix embeddings $E(\mathcal{V})$ and synthetic soft EM tokens $E_{\text{soft}}(u)$, derive false-positive / `PureKL` probability bounds for natural text tokens vs soft tokens, and show how RoPE distance bias affects retrieval from prefix slots to the candidate answer token `t_ans` (`KV_cached` suffix scoring)!
3. **End-to-End Neuron Graph (`Prefix -> DeltaNet/Attn Highway -> SwiGLU Bilinear Core -> Candidate Logit Bridge`)**:
   - Map the exact inter-layer minimal neuron budget across Qwen3.5-4B (`d=2560`, `24` DN + `8` FA layers) needed to move `(T_1, T_2, omega, h)` from EM tokens to `t_ans`, hand off cleanly into N1's SwiGLU bilinear multiplier, and write into candidate answer subspace (`D_answer`).
4. **Machine-Verified Numerical Proof (`verify_n2.py`)**:
   - Build a PyTorch hybrid `GatedDeltaNet + causal Conv1d + MultiHeadAttention-with-interleaved-gate` toy transformer mimicking Qwen3.5, compile sparse inter-token bridge + intrinsic trigger gate weights ONLY, and test:
     - exact preservation of $(T_1, T_2)$ across variable prose lengths $L \in \{8, 64, 512\}$,
     - intrinsic `g_EM = 0` suppression on out-of-distribution text vectors vs lossless routing on EM soft tokens,
     - exact delta-rule interference check (`I - beta k k^T`) under random natural prose writes.

Think deeply and verify every theorem numerically. Keep all outputs strictly inside `N2/`.
