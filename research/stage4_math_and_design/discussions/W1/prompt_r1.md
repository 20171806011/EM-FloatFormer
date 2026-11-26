# Track W1 (Round 1): Expressivity Separation, Gated-DeltaNet <-> Online Complex Kalman Isomorphism & Mechanistic Circuit Induction

You are **Track W1**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **How can directly modifying LLM backbone weights (`Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`) better optimize & elevate electromagnetic waveform reasoning — examined from mathematics, mechanistic circuits, recurrent state physics, and routing geometry?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/weight_mod/W1/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `research/stage4_math_and_design/weight_mod/W1/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts. **ALWAYS** run Python via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```

## 1. Grounded Empirical & Theoretical Context (Inspect Read-Only First)
1. `docs/LONG_TERM_RESEARCH_PLAN.md` and `docs/MULTIMODAL_INTEGRATION_PLAN.md` (note LEMMO uses `Qwen3.5-4B` with LoRA!)
2. `research/stage3d_qwen_signed_phase/qwen_connector.py` (hybrid **Gated-DeltaNet linear-attention + softmax full-attention** `Qwen3.5-4B`, `hidden_size = 2560`, MRoPE)
3. `research/stage3d_qwen_signed_phase/REPORT.md` & `results/summary_master.json`:
   - Frozen `Qwen3.5-4B` + `SignedPhaseCacheConnector` scored **23.80% / 1.4452** on `G1` (`|\theta|=1.60 rad`) vs `direct_s17` **64.62% / 0.9411**, and **Q2 dropped** `28.78% -> 26.43%` when adding a second fresh pilot (`Work A` jumped `66.41% -> 74.22%`)!
4. Stage 4 frozen-host findings (`M1/THEOREM_M1.md`, `M3/THEOREM_M3.md`, `M6/THEOREM_M6.md`):
   - M6 proved **97.66%** (`+0.504103 nats`) of G1 excess NLL sits AFTER `s17` and derived candidate-score Jacobian $M_{\text{cand}} = \Pi C_{\text{cand}} E_{\text{vocab}} D_{\text{answer}} B$.
   - M1 Round 2 showed a **Host-Pure** equivariant/phase-coupled connector CAN reach `64.60%` on G1 / `61.91%` on G3 on a tiny frozen MLP surrogate — leaving wide open whether real multi-layer hybrid `DeltaNet + FullAttn + MRoPE` transformers suffer internal routing decay, attention-head dilution, or multi-turn pilot accumulation bottlenecks that ONLY **direct weight modification** fixes!

## 2. Deep Multi-Angle Mathematical & Mechanistic Questions for Track W1
1. **Expressivity & Routing Separation Theorem (`Prefix-Only` vs `Direct Weight Modification`)**:
   - Let $H_W(E_\theta(x), \text{prompt}) \in \mathbb{R}^4$ be the candidate log-likelihoods of an $L$-layer hybrid Gated-DeltaNet + Attention LM.
   - Contrast reachable function classes & first-order tangent spaces under:
     - (a) **Frozen-host prefix tuning** ($\Delta\theta$ acting strictly through prefix token embeddings $E_\theta(x)$), vs
     - (b) **Direct backbone weight modification** $\Delta W$ (DeltaNet projections, Softmax Attention $Q/K/V/O$, FFN/SwiGLU, RMSNorm gains $\gamma$, or tied `lm_head` / answer readout directions).
   - Prove rigorously WHEN a frozen pretrained text LM's internal routing $D_{\text{answer}}$ collapses ($s_{\min}(M_{\text{cand}}) \ll 1$ or destructive interference across layers, or RoPE/MRoPE relative-distance attenuation over long multi-turn conversation histories in Stage 5–6!) and prove quantitatively how low-rank $\Delta W$ changes singular geometry & attention routing!
2. **Mathematical Isomorphism: Gated-DeltaNet Recurrence $\leftrightarrow$ Online Complex Phase / Kalman-RLS Tracking**:
   - Inspect Qwen3.5's Gated-DeltaNet linear recurrent update:
     $$S_t = \alpha_t S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top, \qquad o_t = S_t^\top q_t.$$
   - **Prove constructively**:
     - Under exact weight choices in `(W_k, W_v, W_q, W_\alpha, W_\beta)`, show how `S_t` implements **exact online rank-1 error-correcting RLS / discrete-time Kalman state estimation** for complex carrier phasor `G` / phase state `e^{i\phi(t)}`, where decay gate $\alpha_t \leftrightarrow e^{-\sigma_\phi^2 \Delta t / 2}$ encodes Brownian decoherence and step gate $\beta_t \leftrightarrow \text{SNR}_t / (1 + \text{SNR}_t)$ encodes precision weighting!
     - Explain from DeltaNet physics why pretrained text-only gates $\alpha_t < 1$ & $\beta_t$ forget or overwrite earlier `<W1><W2>` waveform evidence when intervening Chinese prose or `<REF><REF>` tokens arrive (and in Stage 6 multi-turn conversations!), and prove WHICH minimal weight edits ($\Delta W_\alpha, \Delta W_\beta, \Delta W_{q,k,v}$) convert generic text forgetting into exact Bayes evidence retention!
3. **Mechanistic Circuit Induction & Multi-Angle Architectural Optimization**:
   - Beyond linear recurrence, analyze how Full-Attention layers + `FFN SwiGLU` + `lm_head` form **Complex-Phasor Induction Circuits** (QK-circuit matching canonical lag/protocol keys; OV-circuit writing $2\times 2$ complex rotation & LLR evidence directly into residual stream aligned with Chinese answer candidate tokens `扇区0..3` / `符号0..3`).
   - Synthesize concrete **circuit-targeted weight-editing prescriptions**: which specific components (`DeltaNet a_proj/b_proj/k_proj/v_proj`, `FullAttn q_proj/k_proj/v_proj/o_proj`, early vs late layers, FFN down/gate projections, RMSNorm scaling) to modify vs leave frozen to maximize signal-to-noise ratio in backbone routing!
4. **Mandatory CPU PyTorch Verification (`verify_w1.py`)**:
   - Build a clean deterministic PyTorch CPU **Hybrid Gated-DeltaNet + Softmax Full-Attention Host** (pretrained-like random orthogonal frozen weights mimicking text decay across intervening distractor/prose tokens!).
   - Compare strictly on identical synthetic CFO extrapolation (`train: |\omega| <= 0.25 -> test: G1/G3 |\theta|=1.60`) AND multi-turn `Q2` reference accumulation across varying intervening text gap lengths (`L_gap = 16, 64` tokens):
     1. Frozen Hybrid Host + Prefix Connector ONLY (`Prefix-Only`);
     2. Same Host + Softmax-Attention-Only LoRA (`Attn-LoRA`);
     3. Same Host + Gated-DeltaNet Gate+Proj LoRA + Attention LoRA (`DeltaNet+Attn LoRA`).
   - Measure routing singular values $\sigma_{\min}(D_{\text{answer}})$, DeltaNet state retention vs `L_gap`, `G1..G4` Acc/NLL, and `Q2` fresh-pilot Blackwell gain -> write `research/stage4_math_and_design/weight_mod/W1/results_w1.json`.

## 3. Deliverables Required (Strictly Inside `research/stage4_math_and_design/weight_mod/W1/`)
1. `research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md`
2. `research/stage4_math_and_design/weight_mod/W1/verify_w1.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/weight_mod/W1/results_w1.json`
4. Crisp summary + open questions for W2/W3/W4 in your final response.
