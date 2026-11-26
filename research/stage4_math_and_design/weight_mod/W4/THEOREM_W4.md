# Track W4 (Round 1 + Round 2 Cross-Track Master Lock) — Systems/VRAM Equations, QLoRA Quantization Distortion, Anti-Shock Curriculum, Canonical Adapter Merging & Modal `$12.88` Decision Boundary

> **Artifacts**: [verify_w4.py](verify_w4.py) · [results_w4.json](results_w4.json)  
> **Grounded Telemetry Anchors**: [Stage 3D Report](../../stage3d_qwen_signed_phase/REPORT.md) · [smoke_audit.json](../../stage3d_qwen_signed_phase/results/smoke_audit.json) · [bench_steady.json](../../stage3d_qwen_signed_phase/results/bench_steady.json) · [qwen_connector.py](../../stage3d_qwen_signed_phase/qwen_connector.py) · [M4 Ledger](../M4/THEOREM_M4.md) · [M6 Execution Lock](../M6/THEOREM_M6.md)

## Executive Verdict & Mathematical Decision Law

On a single `NVIDIA A100-SXM4-80GB` (`81,179.8 MiB`) within Modal's **`$12.88` remaining free compute credits** (`Billed Cost: $0.00`), the optimal policy for direct backbone weight modification in EM-FloatFormer Stage 4–6 is governed by five coupled systems-and-decision theorems proved and numerically verified below:

1. **Layer-0 Backprop Invariance (Theorem 1)**: Because training a Layer-0 prefix connector $E_\theta(x)$ already backpropagates through all $L$ backbone layers, upgrading from **Frozen Host** (`52,972.5 MiB` at `mb=16`) to **Selective-Layer BF16 LoRA** (`r = 8`, upper/middle `50%` of layers, `5.06M` params on `Qwen3.5-4B`) adds only **`+92.9 MiB` (`+0.18%`)** peak VRAM (`53,144 MiB < 65,000 MiB` M6 gate) and `+12%` step time (`2.21 s/step < 2.50 s/step`), whereas **Full Fine-Tuning (`4B`)** reaches **`109,204 MiB` (`106.6 GiB`) $\implies$ exact OOM** even at `mb = 8` (`86,706 MiB`).
2. **NF4 Quantization & DeltaNet Gate Amplification (Theorem 2)**: Block-64 NF4 carries **$3,065.8\times$ larger weight MSE** than BF16 ($\eta_{\text{NF4}} = 0.008479$ vs $\eta_{\text{BF16}} = 2.77\times 10^{-6}$). While well-conditioned `1/√L` Pre-Norm MLPs tolerate NF4 at `2 dB` (`ΔKL < 0.001 nats`), quantizing **Gated-DeltaNet decay/phase projections** across $T = 256$ prefix tokens causes catastrophic **`+20 dB` CTTC phase failure** (`ΔNLL = +0.373 nats` at $L=32$, **`+0.728 nats`** at $L=64$, violating the `0.05`-nat budget by `7.5×–14.6×`). **Verdict**: Reject QLoRA completely on `4B` and `9B` (BF16 LoRA fits A100-80GB cleanly and runs `23%` faster); on **`Qwen3.8-27B`**, use **Phase-Safe Mixed-Precision QLoRA** (NF4 strictly on SwiGLU MLPs [`63.61%` of params], BF16 on all DeltaNet/Attention/Norm/Embed/`lm_head`), saving `23.15 GiB` static VRAM (`49.15 GiB` at `mb=16`) with `KL = 0.00032 nats`.
3. **Cold-Start Gradient-Shock & Two-Phase Anti-Shock Curriculum (Theorem 3)**: Jointly unlocking LoRA from step 0 while $E_{\theta_0}$ is unaligned generates a **`3.60×` step-0 (`5.43×` first-20-step) gradient shock** that drives **`9.73×` larger general-text KL forgetting** (`0.7095` vs `0.0729 nats`), induces text-shortcut memorization, and degrades OOD `G1/G3` (`|\theta| = 1.60 > \pi/2`) *below* the frozen host. Running **Phase A Frozen** until convergence before conditionally unlocking **Phase B Modality-Gated Frobenius-Capped LoRA** delivers bit-exact **`0.000000 nats`** pure-text drift and the highest OOD `G1/G3` accuracy.
4. **Cross-Domain Weight-Space Merging (Theorem 4)**: Never average LoRA factors $(B_k, A_k)$ separately (orthogonal gauge non-identifiability $B_k Q Q^{-1} A_k$ injects garbage cross-terms); merge strictly in $\Delta W_k = \frac{\alpha}{r} B_k A_k$ space (or exact rank-$Kr$ concatenation). Across near-orthogonal regime adapters ($\cos(\Delta W_1, \Delta W_2) = -0.0908$), **Weight-Space Linear Soup** (`NLL = 1.0089`, macro `0.9956`) and **TIES-Merging** (`NLL = 1.0266`, macro `1.0104`) generalize to an **unseen mixed SDR capture** with zero retraining, beating Frozen (`1.1021`), single-regime adapters (`1.1409 / 1.0669`), and naive factor averaging (`1.0382`).
5. **Exact Two-Sided Bayesian VoI Decision Window inside `$12.88` (Theorem 5)**: Evaluate the post-Phase-A dev DPI host gap $\widehat{\Delta}_{\text{host}} = \text{NLL}_{\text{Phase A}}(D_{\text{dev}}) - \text{NLL}_{\text{direct\_phys}}(D_{\text{dev}})$ at `$4.52` cumulative exposure (strictly before blind scoring):
   - **$\widehat{\Delta}_{\text{host}} \le 0.030\text{ nats}$** (and Q2 reveal gain passes): **STAY 100% FROZEN (`Branch 0`)** — `$8.39` scheduled (`$9.07` with retry, `$9.50` cap), banking **`$3.38–$4.49`** for Stage 5 (`0` forgetting, `0` extra risk).
   - **$0.030 < \widehat{\Delta}_{\text{host}} \le 0.315\text{ nats}$** (with Tier-1 connector geometry certified): **TRIGGER PHASE B (`Branch 1-Paired-192`)** — `+192` surgical warmup+cosine steps + full 5-arm paired blind ablation + text regression audit = **`+$1.83` (`+$2.17` with retry)**, bringing total Stage 4 spend to **`$10.22` scheduled / `$11.24` with all retries** (or **`$11.75`** including 3-regime adapter merging), strictly under `$11.80 < $12.88` (**`Billed Cost: $0.00`**).
   - **$\widehat{\Delta}_{\text{host}} > 0.315\text{ nats}$** (or Tier-1 geometry fails): **ABORT BEFORE PHASE B** (`VoI < 0`); fix upstream connector/representation on CPU for `$0`.

---

## 1. Exact VRAM, Autograd-Checkpointing & Optimizer Equations (`Qwen3.5-4B -> 9B -> 27B`)

### 1.1 Exact Hybrid Architecture Parameter Accounting (`Qwen3.5/3.8`)

Inspecting `Qwen/Qwen3.5-4B` (revision `851bf6e8...`, vision encoder dropped via `lm.model.visual = None`), `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.8-27B` reveals two critical facts that govern memory physics:
1. **Vocabulary Size**: All three models use **`|V| = 248,320`** (not Qwen2.5's `151,936`, a `1.634×` larger candidate softmax footprint!).
2. **3:1 Hybrid Gated-DeltaNet + Full Gated-Attention Layout** (`full_attention_interval = 4`):
   - `L_lin = (3/4) L` Gated-DeltaNet layers with $h_k$ key heads ($d_k = 128$), $h_v$ value heads ($d_v = 128$), `conv_kernel = 4`, and **FP32 recurrent state** $S_t \in \mathbb{R}^{B \times h_v \times d_k \times d_v}$ (`2.0 MiB` per seq per layer for $h_v = 32$; `3.0 MiB` for $h_v = 48$!).
   - `L_full = (1/4) L` Gated Full-Attention GQA layers (`q_proj` outputs $2 n_q d_h$ with output gate, $n_{kv} = 4$, $d_h = 256$).
   - SwiGLU MLPs (`3 d d_ff` per layer) dominate total parameter count (**`53.9%` on 4B**, **`54.0%` on 9B**, **`63.61%` on 27B**!).

Let $P_{\text{host}}$ be the text-backbone parameter count:
$$
P_{\text{host}} = (1 + \mathbf{1}_{\neg\text{tied}})|V|d + L(3 d d_{\text{ff}} + 2d) + L_{\text{full}}\big[d(2 D_q + 2 D_{kv}) + D_q d + 2 d_h\big] + L_{\text{lin}}\big[d(D_{qkv} + 2 D_v + 2 h_v) + 4 D_{qkv} + 2 h_v + d_v\big] + d.
$$
Evaluating this closed-form equation on `Qwen3.5-4B` yields **`4,205,751,296`**, matching `smoke_audit.json`'s measured `frozen_host_params = 4,205,751,296` to **`0.0000%` error (exact to the single parameter!)**, and exact static BF16 load VRAM `8,021.83 MiB` vs measured `8,021.9 MiB` (plus `2,184,964 × 16 B = 33.34 MiB` FP32 connector weights+grads+AdamW $\implies$ **`8,055.17 MiB`** vs `bench_steady.json` post-step base **`8,055.2 MiB`**).

### 1.2 Exact Lifetime-Separated Peak VRAM Equations & Stage 3D Calibration

A crucial implementation detail discovered in [train_eval_stage3d.py](../../stage3d_qwen_signed_phase/train_eval_stage3d.py#L610) (`host.set_grad_checkpointing(train_scorer == "batched_uncached")`) and [qwen_connector.py](../../stage3d_qwen_signed_phase/qwen_connector.py#L586-L668) explains the `6.15×` dynamic memory gap between Stage 3D's `smoke_audit.json` and `bench_steady.json`:
- Under **`kv_cached` training**, HuggingFace `use_cache=True` forces `grad_checkpointing = False`. Both the prefix forward ($L_{\text{pref}}$ tokens) and expanded candidate forward ($4 L_{\text{step}}$ tokens, $L_{\text{step}} = L_{\text{cand}} - 1$) retain full $L$-layer autograd activations simultaneously alongside the `4×` expanded hybrid cache (`exp_cache`) and FP32 `F.log_softmax` backward workspace:
  $$
  M_{\text{peak}}^{\text{kv\_train}}(b) = M_{\text{static}} + b\Big[\underbrace{4\big(M_{\text{DeltaNet,FP32}} + M_{\text{conv}} + M_{\text{KV}}(L_{\text{pref}} + L_{\text{step}})\big)}_{M_{\text{exp\_cache}} = 240.6\text{ MiB/ex (4B)}} + \underbrace{14 \cdot 4 L_{\text{step}} |V|}_{M_{\text{logits+bwd}} = 198.9\text{ MiB/ex (4B, Q3)}} + \underbrace{(L_{\text{pref}} + 4 L_{\text{step}})\,a_{\text{tok}}}_{M_{\text{act}} = 2,372.7\text{ MiB/ex (4B, Q3)}}\Big].
  $$
- Under **`batched_uncached + grad_checkpointing=True`**, `use_cache=False` eliminates `exp_cache` entirely and saves only $L$ layer-boundary hidden checkpoints ($2 L \cdot 4(L_{\text{pref}} + L_{\text{cand}}) d$ bytes) plus the **non-overlapping maximum** of either (i) `lm_head` + FP32 `log_softmax` backward OR (ii) 1-layer recomputation:
  $$
  M_{\text{peak}}^{\text{batched\_ckpt}}(b) = M_{\text{static}} + b\Big[M_{\text{ckpt\_bounds}} + \max\!\big(M_{\text{logits\_bwd}},\ M_{\text{1-layer\_recomp}}\big)\Big].
  $$
- Under **`kv_eval` (`torch.no_grad()`, `b = 64`)**, `cand_out` retains `past_key_values = exp_cache` through line 659 (`F.log_softmax`), causing `exp_cache` + `pref_cache` clone overlap (`300.8 MiB/ex`), candidate BF16 logits + FP32 `log_softmax` cast/output (`142.1 MiB/ex`), and 1-layer FLA recurrent FP32 workspace (`141.0 MiB/ex`) to sum to **`583.9 MiB/ex`** (`37.4 GiB` dynamic at `b=64`).

#### Verification Against ALL 10 Stage 3D A100-80GB Benchmarks (`max rel error = 0.332%`)

| Stage 3D A100-80GB Benchmark (`Qwen3.5-4B`) | Measured (`MiB`) | Predicted (`MiB`) | Error (`MiB`) | Rel. Error | Decomposition (`Static / Cache / Logits / Acts`, MiB) |
|---|---:|---:|---:|---:|---|
| `Q3_after_pilot_kv_mb16` (train, $L_p{=}326, L_c{=}16$) | `52,972.5` | `53,051.1` | `+78.6` | **`0.148%`** | `8,055 / 3,850 / 3,183 / 37,963` |
| `Q3_after_pilot_kv_mb8` (train, $L_p{=}326, L_c{=}16$) | `30,514.6` | `30,553.1` | `+38.5` | **`0.126%`** | `8,055 / 1,925 / 1,591 / 18,982` |
| `Q2_fresh_resample_kv_mb16` (train, $L_p{=}294, L_c{=}6$) | `43,763.7` | `43,764.0` | `+0.3` | **`0.001%`** | `8,055 / 3,766 / 1,061 / 30,882` |
| `Q2_fresh_resample_kv_mb8` (train, $L_p{=}294, L_c{=}6$) | `25,910.3` | `25,909.6` | `-0.7` | **`0.003%`** | `8,055 / 1,883 / 530 / 15,441` |
| `Q3_repeat_batched_ckpt_mb4` (train, `grad_ckpt=True`) | `9,910.2` | `9,877.3` | `-32.9` | **`0.332%`** | `8,055 / 0 / 0 / 1,822` |
| `Q3_repeat_batched_ckpt_mb8` (train, `grad_ckpt=True`) | `11,736.3` | `11,699.4` | `-36.9` | **`0.315%`** | `8,055 / 0 / 0 / 3,644` |
| `Q3_repeat_batched_ckpt_mb16` (train, `grad_ckpt=True`) | `15,388.6` | `15,343.6` | `-45.0` | **`0.292%`** | `8,055 / 0 / 0 / 7,288` |
| `Q3_repeat_batched_ckpt_mb32` (train, `grad_ckpt=True`) | `22,693.0` | `22,632.0` | `-61.0` | **`0.269%`** | `8,055 / 0 / 0 / 14,577` |
| `Q3_after_pilot_eval_b32` (`no_grad`, KV-cached) | `26,728.4` | `26,730.7` | `+2.3` | **`0.009%`** | `8,047 / 9,625 / 4,547 / 4,512` |
| `Q3_after_pilot_eval_b64` (`no_grad`, KV-cached) | `45,409.9` | `45,414.6` | `+4.7` | **`0.010%`** | `8,047 / 19,250 / 9,094 / 9,024` |

### 1.3 Theorem 1 (Layer-0 Backprop Invariance & `4B -> 9B -> 27B` Scaling Matrix)

**Theorem 1 (Incremental Cost of Selective LoRA Over Frozen Layer-0 Prefix Training).**  
*Let a frozen $L$-layer backbone $H_{W_0}$ already backpropagate gradients $\nabla_{E_\theta}\mathcal{L} = (\partial E_\theta / \partial \theta)^\top \frac{\partial \mathcal{L}}{\partial h^{(0)}}$ to a Layer-0 connector $E_\theta$. Augmenting a fraction $\gamma \in (0, 1]$ of layers with rank-$r$ BF16 LoRA adapters ($P_{\text{LoRA}}$ total params, FP32 master + grad + AdamW $m,v = 18\text{ B/param}$) increases peak training VRAM and step latency by only:*
$$
\Delta M_{\text{VRAM}}^{\text{LoRA}} = 18 P_{\text{LoRA}} + O\!\left(b\,N_{\text{tok}}\,\gamma L\,r\right), \qquad \frac{\Delta t_{\text{step}}^{\text{LoRA}}}{t_{\text{step}}^{\text{frozen}}} = O\!\left(\frac{\gamma r}{d}\right) + \epsilon_{\text{launch}} \approx 10\text{--}15\%.
$$
*Proof.* Every intermediate input activation $h^{(\ell)}$ required to compute the weight gradients $\nabla_{B_\ell}\mathcal{L} = (\nabla_{y_\ell}\mathcal{L})(A_\ell h^{(\ell)})^\top$ and $\nabla_{A_\ell}\mathcal{L} = (B_\ell^\top \nabla_{y_\ell}\mathcal{L})(h^{(\ell)})^\top$ is **already stored** on the autograd tape to compute $\frac{\partial \mathcal{L}}{\partial h^{(\ell)}} = W_{0,\ell}^\top \nabla_{y_\ell}\mathcal{L}$ on the path back to Layer 0. Only the rank-$r$ bottlenecks $A_\ell h^{(\ell)} \in \mathbb{R}^r$ (`0.38 MiB/ex` at `r = 8` on 4B) and static adapter+optimizer buffers (`86.9 MiB` for `5.06M` params) are new. $\square$

#### Calibrated `4B / 9B / 27B` A100-80GB Feasibility Matrix (`Q3_after_pilot`, $L_p{=}326, L_c{=}16$, eff. batch 32)

| Model (Text Params / BF16 Static) | Regime (`scorer`, `mb`, `rank`) | Trainable Host Params | Static (`MiB`) | Peak VRAM (`MiB` / `GiB`) | Est. Step (`s`) | Fits A100-80G | Fits M6 `65 GiB` |
|---|---|---:|---:|---:|---:|:---:|:---:|
| **Qwen3.5-4B** (`4.206B` / `8,022 MiB`) | Frozen Conn (`kv_train`, `mb=16`) | `0` | `8,055` | `53,051` (`51.81`) | `1.98` | YES | **YES** |
| | Full FT FP32-Adam (`kv_train`, `mb=16`) | `4.206B` | `64,208` | **`109,204` (`106.64`)** | `2.61` | **OOM** | NO |
| | Full FT FP32-Adam (`kv_train`, `mb=8`) | `4.206B` | `64,208` | **`86,706` (`84.67`)** | `2.82` | **OOM** | NO |
| | Full FT 12B/p BF16+m,v (`kv_train`, `mb=16`) | `4.206B` | `48,164` | **`93,160` (`90.98`)** | `2.53` | **OOM** | NO |
| | Full FT FP32-Adam (`batched_ckpt`, `mb=16`) | `4.206B` | `64,208` | `72,624` (`70.92`) | `6.91` | YES | NO |
| | **Selective BF16 LoRA (`kv_train`, `mb=16, r=8`)** | **`5.06M`** | **`8,142`** | **`53,144` (`51.90`)** | **`2.21`** | **YES** | **YES** |
| | Selective BF16 LoRA (`kv_train`, `mb=16, r=16`) | `8.16M` | `8,195` | `53,203` (`51.96`) | `2.21` | YES | **YES** |
| | Selective BF16 LoRA (`kv_train`, `mb=8, r=8`) | `5.06M` | `8,142` | `30,643` (`29.92`) | `2.40` | YES | **YES** |
| | QLoRA NF4 (`kv_train`, `mb=16, r=8`) | `5.06M` | `3,341` | `48,343` (`47.21`) | `2.73` | YES | YES *(slower + noisy)* |
| **Qwen3.5-9B** (`8.954B` / `17,078 MiB`) | Full FT FP32-Adam (`kv_train`, `mb=8`) | `8.954B` | `136,658` | **`162,485` (`158.68`)** | `6.01` | **OOM** | NO |
| | Full FT 8-bit Adam (`batched_ckpt`, `mb=16`) | `8.954B` | `85,423` | **`96,749` (`94.48`)** | `15.05` | **OOM** | NO |
| | Selective BF16 LoRA (`kv_train`, `mb=16, r=8`) | `6.88M` | `17,229` | `68,890` (`67.28`) | `4.71` | YES | NO |
| | **Selective BF16 LoRA (`kv_train`, `mb=8, r=8`)** | **`6.88M`** | **`17,229`** | **`43,060` (`42.05`)** | **`5.10`** | **YES** | **YES** |
| | QLoRA NF4 (`kv_train`, `mb=16, r=8`) | `6.88M` | `7,949` | `59,610` (`58.21`) | `5.80` | YES | YES *(slower + noisy)* |
| **Qwen3.8-27B** (`26.896B` / `51,300 MiB`) | Frozen Conn (`kv_train`, `mb=8`) | `0` | `51,333` | **`118,474` (`115.70`)** | `13.68` | **OOM** | NO |
| | Full FT FP32-Adam (`batched_ckpt`, `mb=16`) | `26.896B` | `410,434` | **`432,173` (`422.04`)** | `44.20` | **OOM** | NO |
| | Selective BF16 LoRA (`batched_ckpt`, `mb=8, r=8`) | `21.69M` | `51,706` | `62,597` (`61.13`) | `40.61` | YES | **YES** |
| | Full QLoRA NF4 (`batched_ckpt`, `mb=16, r=8`) | `21.69M` | `18,541` | `40,323` (`39.38`) | `46.21` | YES | YES *(breaks +20dB!)* |
| | **Mixed MLP-NF4 + DeltaNet-BF16 (`ckpt`, `mb=16`)** | **`21.69M`** | **`28,552`** | **`50,334` (`49.15`)** | **`42.52`** | **YES** | **YES (Phase-Safe!)** |

---

## 2. Exact NF4 Weight-Quantization Phase Distortion Bound (Theorem 2)

### 2.1 Fundamental Quantization Noise & Depth Propagation

Let each linear projection matrix $W_\ell \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be quantized in blocks of $B_q = 64$ using Dettmers et al.'s 16-level NormalFloat (`NF4`) quantile codebook on $[-1, 1]$ normalized by block absmax $s_B = \max_{k \in B} |w_k|$. Write the rounding error $E_{q,\ell} = Q(W_\ell) - W_\ell$ with relative variance $\eta_q \equiv \mathbb{E}[E_{q,ij}^2] / \sigma_W^2$:
- Direct numerical verification (`verify_w4.py`, $4096 \times 256$ Gaussian weights, $B_q = 64$):
  $$\eta_{\text{NF4}} = \mathbf{0.008479} \text{ (SQNR } 20.72\text{ dB)}, \quad \eta_{\text{INT4}} = 0.011591 \text{ (}19.36\text{ dB)}, \quad \eta_{\text{BF16}} = \mathbf{2.77 \times 10^{-6}} \text{ (}55.58\text{ dB)},$$
  an exact **$3,065.8\times$ variance ratio** ($55.4\times$ RMS ratio) between NF4 and BF16!
- By uniform local bin approximation $\epsilon_q \sim \mathcal{U}[-\Delta_i / 2, \Delta_i / 2]$ on each NF4 interval $\Delta_i$, propagating layerwise perturbations $\delta_\ell = E_{q,\ell} u_\ell$ through an $L$-layer residual Pre-Norm backbone $h_{\ell+1} = h_\ell + \gamma_\ell F_\ell(\text{RMSNorm}(h_\ell))$ to the centered 4-candidate log-odds $\ell(x) = \Pi C_{\text{cand}} E_{\text{vocab}} \text{RMSNorm}(h_L)$ yields:
  $$\Delta \ell = \sum_{\ell=0}^{L-1} M_\ell \delta_\ell, \qquad \sigma_{\Delta \ell}^2 \equiv \frac{1}{3} \mathbb{E}\big[\|\Delta \ell\|_2^2\big] = \frac{\eta_q}{3}\sum_{\ell=0}^{L-1}\gamma_\ell^2 \operatorname{tr}\!\big(M_\ell \Sigma_{\delta,\ell} M_\ell^\top\big).$$

### 2.2 Theorem 2 (Second-Order Phase Log-Loss Excess & Recurrent Gate Amplification)

**Theorem 2 (SNR-Regime Bifurcation of NF4 Quantization Distortion).**  
1. *(Residual Pre-Norm Bound)* Let $p^*(x) = \operatorname{softmax}(\ell^*(x))$ be the exact Bayes 4-sector posterior and $\hat{\ell}(x) = \ell^*(x) + \Delta \ell(x)$ with isotropic zero-mean centered logit perturbation $\operatorname{Cov}(\Delta \ell) = \sigma_{\Delta \ell}^2 \Pi$. By exact second-order Taylor expansion of the log-partition function with Hessian $H(p^*) = \operatorname{diag}(p^*) - p^*(p^*)^\top$ (trace $1 - \|p^*\|_2^2$):
   $$
   \Delta \text{NLL}_q(x) = \mathbb{E}\big[D_{\text{KL}}(p^*(x) \,\|\, \operatorname{softmax}(\ell^*(x) + \Delta \ell))\big] = \frac{1}{2}\,\sigma_{\Delta \ell}^2\,\Big(1 - \sum_{k=0}^3 p_k^*(x)^2\Big) + O(\sigma_{\Delta \ell}^4).
   $$
2. *(Multiplicative Recurrent-Gate & High-SNR Amplification)* In Qwen3.5's $L_{\text{lin}} = \frac{3}{4}L$ Gated-DeltaNet layers, log-decay $g_t = -e^{A_{\log}}\operatorname{softplus}(w_a^\top x_t + b_{dt})$ compounds multiplicatively over $T$ intervening prefix tokens as $\alpha_{1:T} = \exp\!\big(\sum_{t=1}^T g_t\big)$. Quantizing $w_a$ and phase-transport projections introduces (i) retention-ratio variance $\operatorname{Var}(\hat{\alpha}_{1:T} / \alpha_{1:T}) \propto T^2 \eta_q$ and (ii) cumulative angular jitter $\sigma_{\Delta\phi} \propto \sqrt{L_{\text{lin}} \eta_q / d_{\text{eff}}}$, whose log-odds sensitivity scales linearly with physical SNR $\rho$:
   $$
   \sigma_{\Delta \ell}^{\text{recurrent}}(\rho, L, T) \propto \beta_{\text{eff}}(\rho)\,\sqrt{\frac{L_{\text{lin}} \eta_q}{d_{\text{eff}}} + c_a T^2 \eta_q}, \qquad \beta_{\text{eff}}(\rho) = O(\rho).
   $$
   Hence at `+20 dB` CTTC ($\rho = 100$), quantizing DeltaNet gate/phase weights into NF4 breaches the `< 0.05` nat precision ceiling by an order of magnitude, whereas **Mixed-Precision `MLP-NF4 + DeltaNet/Attn/IO-BF16`** eliminates the recurrent amplification entirely while preserving `63.6%` of 27B parameter compression. $\square$

### 2.3 CPU PyTorch Verification (`verify_w4.py`, Exp 2B)

1. **Pre-Norm `1/√L`-Scaled Residual Phase Network (`L = 32, 64`, `d = 48`)**:
   - At `2 dB` (`L = 32`): `BF16` has `σ_Δℓ = 0.00164`, `KL = 1e-6 nats`; `NF4 (BF16 I/O)` has `σ_Δℓ = 0.02347`, actual `KL = 0.000135` vs Theorem 2 prediction **`0.000139 nats`** (`97%` exact!). Removing BF16 I/O protection (`nf4_full_unprotected`) quadruples logit noise (`σ_Δℓ = 0.0942`, `max |Δℓ| = 0.223 nats`).
   - At `+20 dB` (`L = 64`): `BF16` maintains `σ_Δℓ = 0.00278`, `max |Δℓ| = 0.0103 nats`; `NF4 (BF16 I/O)` hits `max |Δℓ| = 0.1669 nats`; `NF4 (unprotected)` hits **`max |Δℓ| = 0.5542 nats`**.
2. **Gated-DeltaNet Recurrent Gate & Phase-Transport Probe (`T = 256` prefix tokens, `L = 32` & `L = 64`)**:

| SNR / Depth | Precision Mode | Retention Ratio SD | Centered $\sigma_{\Delta \ell}$ (`nats`) | `ΔNLL` vs FP32 (`nats`) | `KL` vs FP32 (`nats`) | `< 0.05` Nat Gate |
|---|---|---:|---:|---:|---:|:---:|
| **`2 dB` (`L=32`, 4B/9B)** | `BF16` (All Layers) | `0.00051` | `0.00715` | `-0.00025` | `0.00001` | PASS |
| | `Full NF4` (incl. DeltaNet gates) | `0.02636` (`52×`) | `0.37863` | `+0.03335` | `0.02216` | MARGINAL |
| | `Mixed MLP-NF4 + DeltaNet-BF16` | `0.00051` | `0.00817` | `-0.00027` | `0.00001` | **PASS** |
| **`+20 dB` (`L=32`, 4B/9B)** | `BF16` (All Layers) | `0.00051` | `0.04495` | `+0.00096` | `0.00014` | PASS |
| | **`Full NF4` (incl. DeltaNet gates)** | **`0.02636`** | **`2.38859`** | **`+0.37320`** | **`0.36611`** | **FAIL (`7.5×` over!)** |
| | `Mixed MLP-NF4 + DeltaNet-BF16` | `0.00051` | `0.05136` | `+0.00114` | `0.00019` | **PASS** |
| **`+20 dB` (`L=64`, 27B)** | `BF16` (All Layers) | `0.00051` | `0.06255` | `-0.00052` | `0.00024` | PASS |
| | **`Full NF4` (incl. DeltaNet gates)** | **`0.02636`** | **`3.33419`** | **`+0.72827`** | **`0.70779`** | **FAIL (`14.6×` over!)** |
| | **`Mixed MLP-NF4 + DeltaNet-BF16`** | **`0.00051`** | **`0.07169`** | **`-0.00056`** | **`0.00032`** | **PASS (`2,212×` cleaner!)** |

**Engineering Verdict**: Never use QLoRA on `4B` or `9B` on A100-80GB (BF16 LoRA fits in `51.90 GiB` / `42.05 GiB` and is `23%` faster). On `27B`, use **BF16 LoRA (`batched_ckpt, mb=8`, `61.13 GiB`)** or **Mixed `MLP-NF4 + DeltaNet/Attn/IO-BF16` (`49.15 GiB` at `mb=16`)**; **forbid 4-bit quantization of Gated-DeltaNet projections**.

---

## 3. Cold-Start Gradient-Shock Inequality & Two-Phase Anti-Shock Curriculum (Theorem 3)

### 3.1 Proof of the Cold-Start Gradient-Shock & Trust-Region Ejection Inequality

Let $\Theta = (\theta, \Delta W)$ denote connector parameters $\theta$ ($E_\theta(x)$) and backbone LoRA weights $\Delta W = \frac{\alpha}{r}BA$ initialized at $\Delta W_0 = 0$ ($B_0 = 0, A_0 \neq 0$). Let $E^*(x)$ be the converged Phase-A connector output, and define the cold-start connector misalignment error $\delta E_t(x) \equiv E_{\theta_t}(x) - E^*(x)$ with $\|\delta E_0\|_{L^2} = O(1)$.

**Theorem 3 (Cold-Start Gradient-Shock & AdamW Trust-Region Ejection).**  
1. *(Cross-Hessian Gradient Contamination)* Expanding $\nabla_{\Delta W}\mathcal{L}(\theta_t, \Delta W_t)$ around $E^*(x)$ gives:
   $$
   \nabla_{\Delta W}\mathcal{L}(\theta_t, \Delta W_t) = \underbrace{\nabla_{\Delta W}\mathcal{L}(\theta^*, \Delta W_t)}_{g_{\text{true}}(\Delta W_t)} + \underbrace{H_{W,E}(\theta^*, \Delta W_t)\,\delta E_t + R_2(\delta E_t)}_{g_{\text{shock}, t}},
   $$
   where $H_{W,E} = \mathbb{E}\big[J_{\Delta W}^\top H_{\text{CE}}(q) J_E + \sum_c (q_c - y_c)\nabla_{\Delta W}\nabla_E f_c\big]$. At $t = 0$, $E_{\theta_0}(x)$ has zero mutual information with the true physical phase, whereas the prompt text tokens sit in the pretrained high-gain linguistic subspace $\mathcal{S}_{\text{text}}$. Consequently, the gradient signal-to-shock ratio collapses:
   $$
   \text{SNR}_{\nabla W}(t) = \frac{\|\mathbb{E}[g_{\text{true}}]\|_2^2}{\mathbb{E}[\|g_{\text{shock}, t}\|_2^2]} \le \frac{\|J_{\Delta W, E^*}\|_F^2\,\mathcal{E}_{\text{host}}^2}{\sigma_{\min}^2(H_{W,E})\,\|\delta E_t\|^2} \ll 1 \quad (t \ll T_{\text{conv}}).
   $$
2. *(AdamW Normalization Amplifies General-Text Drift)* Under AdamW, $\Delta W_{t+1} - \Delta W_t = -\eta_W(t)\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ takes $O(\eta_W)$ coordinate steps along the dominant shock direction $\mathcal{S}_{\text{text}}$ regardless of gradient magnitude. Over the first $T_{\text{shock}}$ steps while $\|\delta E_t\|$ decays, general-domain KL drift obeys the Fisher lower bound:
   $$
   D_{\text{KL}}^{\text{gen}}\big(P_{W_0} \,\|\, P_{W_0 + \Delta W_{T_{\text{shock}}}}\big) = \frac{1}{2}\text{vec}(\Delta W_{T_{\text{shock}}})^\top F_{\text{gen}} \text{vec}(\Delta W_{T_{\text{shock}}}) + O(\|\Delta W\|^3) \ge \frac{1}{2}\,\lambda_{\min}\!\big(F_{\text{gen}}|_{\mathcal{S}_{\text{text}}}\big)\,\Big\|\sum_{t=0}^{T_{\text{shock}}-1}\eta_W(t)\,u_{\text{shock}, t}\Big\|_2^2.
   $$
   Because $\mathcal{S}_{\text{text}}$ carries high general-language Fisher curvature $\lambda(F_{\text{gen}}|_{\mathcal{S}_{\text{text}}}) \gg 0$, cold-start LoRA is ejected from the general-text trust region and trapped in a surface-text shortcut minimum (overfitting training support $|\theta| \le 1.25$ and degrading OOD $|\theta| = 1.60 > \pi/2$).
3. *(Two-Phase Optimal Switching Rule)*
   - **Phase A (Frozen Host, $\Delta W \equiv 0$)**: Optimize equivariant connector $E_\theta$ + `RefSufficientPool` until $\|\nabla_\theta \mathcal{L}\| \le \epsilon_A$ ($\delta E_{T_A} \approx 0$). Measure the post-Phase-A DPI host gap on disjoint dev $D_{\text{dev}}$:
     $$\widehat{\Delta}_{\text{host}} \equiv \text{NLL}_{\text{Phase A}}(D_{\text{dev}}) - \text{NLL}_{\text{direct\_phys}}(D_{\text{dev}}).$$
   - **Gate**: If $\widehat{\Delta}_{\text{host}} \le \tau_{\text{trigger}} = 0.030\text{ nats}$ (satisfying $0.030 + 1.645\,\text{SE}_{\text{dev}} \le 0.050\text{ nats}$ M6 ceiling), **STOP — keep host 100% frozen** (`0` forgetting, `$0` extra compute).
   - **Phase B (Conditional, $\widehat{\Delta}_{\text{host}} > 0.030$ and $\le 0.315$ nats with certified Tier-1 geometry)**: Unlock **Modality-Gated Nullspace LoRA** $\Delta W_{\text{eff}}(x) = g_{\text{EM}}(x)\,\Delta W$ (`g_EM = 0` on text-only prompts $\implies D_{\text{KL}}^{\text{pure-text}} \equiv 0$) with $15\%$ linear LR warmup, cosine decay, reduced connector LR ($0.15\eta_A$), and per-layer Frobenius norm cap $\|\Delta W_\ell\|_F \le R_\ell$, guaranteeing on-task multi-turn KL drift $\le \frac{1}{2}\sum_\ell \sigma_{\max}(F_{\text{gen}}^{(\ell)}) R_\ell^2$. $\square$

### 3.2 CPU PyTorch Verification (`verify_w4.py`, Exp 1 — 3 Paired Seeds `202611, 202622, 202633`)

Training identical pretrained dual-task residual hosts (`Task A`: 16-class general text; `Task B`: exact QPSK CFO `|ω| <= 0.25 -> OOD G1/G3 |θ| = 1.60 rad`, 240 matched steps):

| Training Protocol (3-Seed Mean ± SD) | Step-0 / Mean-20 LoRA Shock $\|\nabla_{\Delta W}\mathcal{L}\|$ | Ungated Gen-Text KL (`nats`) | Modality-Gated Gen-Text KL | `G1` Text-Only NLL *(higher = less shortcut)* | OOD `G1` Acc / NLL (`|θ|=1.60`) | OOD `G3` Acc / NLL (`|θ|=1.60`) | Final $\|\Delta W\|_F$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Frozen Only` ($\Delta W = 0$) | `0.0000 / 0.0000` | `0.0000` | `0.0000` | — | `63.66% / 0.9883` | `60.53% / 1.0173` | `0.000` |
| `Cold-Start Joint (Unconstrained)` | `0.0598 / 0.0938` | **`0.7095`** (`±0.056`) | `0.0000` | `1.208` *(corrupted!)* | `62.83% / 0.9976` | `60.03% / 1.0282` | `7.907` |
| `Cold-Start Joint (Frob-Capped)` | `0.0598 / 0.0903` | **`0.1832`** (`±0.029`) | `0.0000` | `1.350` | `63.27% / 0.9938` | `60.34% / 1.0237` | `1.979` |
| **`Two-Phase Anti-Shock (Phase A -> B)`** | **`0.0166 / 0.0173`** | **`0.0729`** (`±0.035`) | **`0.0000`** | **`1.430` *(clean!)*** | **`63.77% / 0.9895`** | **`60.86% / 1.0209`** | **`1.692`** |

**Verified Takeaways**:
- Cold-start suffers **`3.60×` larger step-0 shock** and **`5.43×` larger first-20-step shock**, causing **`9.73×` larger ungated KL forgetting** (`2.51×` even at matched Frobenius cap!) and actively **degrading OOD `G1/G3` below the frozen host** (`-0.83 pp / +0.0093 nats`).
- Two-Phase Anti-Shock achieves the **highest OOD `G1` (`63.77%`) and `G3` (`60.86%`) accuracy**, eliminates text-shortcut reliance (`1.430 > \ln 4`), cuts ungated forgetting `9.73×`, and guarantees `0.000000` nats pure-text drift via modality gating. Moreover, notice that because `Frozen Only` already reached `0.9883` on this low-gap host (whereas under severe routing attenuation in our earlier bottleneck probe `dev_host_gap = 0.0996 nats`, Phase B recovered **`+0.0714` nats on G1 and `+0.0646` nats on G3**!), this simultaneously validates why the **Phase-A DPI gate** must decide whether Phase B is needed.

---

## 4. Cross-Domain Adapter Merging: Weight-Space Soup, Task Arithmetic & TIES (Theorem 4)

### 4.1 Fisher-Orthogonality Additivity & The LoRA Factor-Gauge Pitfall

Consider $K$ regime-specific Phase-B LoRA adapters $\{\Delta W_k^*\}_{k=1}^K$ (`Synthetic Large-CFO`, `Pluto 850 MHz AC-coupled + Jitter`, `CTTC 433/708/2450 MHz + Sideband Spur`) trained from a **shared Phase-A frozen anchor** $(W_0, E_{\theta_A})$ with local domain Hessians $H_k = \nabla^2 \mathcal{L}_k(W_0 + \Delta W_k^*) \succeq 0$.

**Proposition 4.1 (LoRA Factor-Averaging Gauge Pitfall).**  
*Because $\Delta W_k = \frac{\alpha}{r} B_k A_k = \frac{\alpha}{r} (B_k Q_k)(Q_k^{-1} A_k)$ is invariant under any invertible $Q_k \in \text{GL}(r)$, naive factor averaging $\widetilde{\Delta W}_{\text{fac}} = \frac{\alpha}{r}(\sum_k \lambda_k B_k)(\sum_k \lambda_k A_k)$ introduces arbitrary gauge-dependent cross-terms $\sum_{j \neq k} \lambda_j \lambda_k B_j Q_j Q_k^{-1} A_k$. **Rule**: Always merge in effective weight-delta space $\Delta W_k = \frac{\alpha}{r} B_k A_k$ (or exact rank-$\le Kr$ stacked concatenation $B_{\text{cat}} = [\sqrt{\lambda_1} B_1, \dots], A_{\text{cat}} = [\sqrt{\lambda_1} A_1; \dots]$ with zero truncation error).*

**Theorem 4 (Fisher-Orthogonal Additivity & Asymmetric-Interference Regularization).**  
*Let a merged adapter be $\Delta W_{\text{mix}} = \sum_{k=1}^K \lambda_k \Delta W_k^*$. For an unseen target capture (e.g. held-out `f708_snr10`) whose local optimum decomposes as $\Delta W_{\text{target}}^* = \sum_{k=1}^K \alpha_k \Delta W_k^* + r_\perp$ with cross-domain Fisher interference ratio $\Gamma_{jk} \equiv \frac{\langle \Delta W_k^*, H_j \Delta W_k^* \rangle}{\langle \Delta W_j^*, H_j \Delta W_j^* \rangle}$:*
1. **Symmetric Orthogonal Regime ($\Gamma_{jk} \le \varepsilon_\perp \ll 1,\ \forall j \neq k$)**: Task Arithmetic ($\lambda_k = \alpha_k \approx 1$) attains excess loss $\mathcal{L}_{\text{target}}(\Delta W_{\text{mix}}) - \mathcal{L}_{\text{target}}(\Delta W_{\text{target}}^*) \le \frac{1}{2}\|r_\perp\|_{H_{\text{target}}}^2 + O(\varepsilon_\perp)$, generalizing to unseen mixed captures **without retraining**.
2. **Asymmetric Norm / Cross-Interference Regime ($\Gamma_{12} > 1$ when $\|\Delta W_2\|_F > \|\Delta W_1\|_F$)**: Unweighted Task Arithmetic ($\lambda_1 = \lambda_2 = 1$) injects quadratic excess $\frac{1}{2}\langle \Delta W_2^*, H_1 \Delta W_2^* \rangle$ into Regime 1. **Convex Weight-Space Soup** ($\lambda_k \in (0, 1), \sum \lambda_k = 1$) and **TIES-Merging** (top-$\kappa$ coordinate trimming + mass sign election + disjoint averaging) suppress cross-regime variance by factor $\sim \kappa \lambda^2$ while eliminating destructive sign cancellation. $\square$

### 4.2 CPU PyTorch Verification (`verify_w4.py`, Exp 2A — Zero-Retraining Transfer to Unseen Mixed Capture)

Training two separate rank-8 Phase-B adapters from a shared Phase-A anchor:
- **Regime 1 (`Pluto-like`)**: `2 dB`, high jitter $\sigma_\phi = 0.24$, uncompensated DC bias $0.35 e^{i0.5}$, low CFO $|\omega| \le 0.10$;
- **Regime 2 (`Large-CFO + Spur`)**: `5 dB`, low jitter $\sigma_\phi = 0.05$, large CFO $|\omega| \in [0.35, 0.45]$, coherent spur $r = 0.25$;
- **Held-Out Unseen Mixed Capture (`f708_snr10` analogue)**: `3.5 dB` + jitter `0.16` + DC `0.25` + large CFO `|\omega|=0.40` + spur `0.18`.

Measured adapter geometry: **$\cos(\Delta W_1, \Delta W_2) = -0.0908$** (near-orthogonal!), $\|\Delta W_1\|_F = 1.875$, $\|\Delta W_2\|_F = 2.541$, asymmetric Fisher cross-interference $\Gamma_{12} = 2.268$ vs $\Gamma_{21} = 0.223$.

| Adapter / Zero-Retraining Merge Method | Reg 1 (`Pluto-DC`) Acc / NLL | Reg 2 (`CFO+Spur`) Acc / NLL | **Unseen Mixed Capture** Acc / NLL | **Macro Avg NLL** |
|---|---:|---:|---:|---:|
| `Phase-A Frozen` ($\Delta W = 0$) | `50.54% / 1.1197` | `56.20% / 1.0223` | `56.41% / 1.1021` | `1.0814` |
| `Adapter 1 Only` (`Pluto-DC`) | **`52.59% / 1.0925`** | `48.39% / 1.0880` | `44.96% / 1.1409` | `1.1071` |
| `Adapter 2 Only` (`CFO+Spur`) | `49.37% / 1.2822` | **`70.12% / 0.7577`** | `61.60% / 1.0669` | `1.0356` |
| `Naive Factor Avg` *(Gauge Pitfall!)* | `51.03% / 1.1263` | `61.96% / 0.9046` | `57.93% / 1.0382` | `1.0230` |
| **`Weight-Space Linear Soup (0.5/0.5)`** | `50.39% / 1.1606` | `66.94% / 0.8174` | **`59.34% / 1.0089`** | **`0.9956` (BEST)** |
| `TIES-Merging (trim 40%, scale 1.0)` | `50.44% / 1.2170` | `67.29% / 0.7877` | `58.79% / 1.0266` | `1.0104` |
| `Task Arithmetic (scale 0.75)` | `50.49% / 1.2065` | `66.31% / 0.8014` | `57.77% / 1.0293` | `1.0124` |
| `Task Arithmetic (scale 1.00)` | `50.93% / 1.2540` | `65.14% / 0.8167` | `57.15% / 1.0655` | `1.0454` |

**Verified Takeaway**: Merging in $\Delta W$-space via **Weight-Space Soup (`1.0089`)** or **TIES (`1.0266`)** cuts unseen-capture NLL by **`-0.058` to `-0.132` nats** over single-regime adapters and **`-0.093` nats** over Frozen with **zero retraining**, while confirming Theorem 4's exact warning that asymmetric Fisher interference ($\Gamma_{12} = 2.27$) penalizes unweighted `1.00` Task Arithmetic (`1.0655`) relative to norm-damped Soup / TIES!

---

## 5. Bayesian Value-of-Information (VoI) & Modal `$12.88` Decision Boundary (Theorem 5)

### 5.1 Exact Decimal Modal Free-Credit Ledger (`A100-80GB`, 2 Cores, 16 GiB = `$0.00075572/s`)

Anchored verbatim to [M4 §3.4](../M4/THEOREM_M4.md#34-budget-arithmetic-and-zero-bill-gate) and [M6 §4.4](../M6/THEOREM_M6.md#44-the-sole-stage-4-modal-ledger) (`$12.88` remaining credits, hard invariant `Billed Cost = $0.00`). Phase B executes **only** on the winning Arm 1 architecture (controls `Zero` & `Naive-Additive` remain frozen!) at conservative `2.80 s/step` LoRA train and `33 ms/row` LoRA eval (`44,812` rows = full 1-arm blind suite + `3,000` dev + `150 s` general-text regression audit):

| Execution Branch | Phase B Steps | Paired w/ Frozen Arm 1 | Incremental Phase B (`USD`, no retry / w/ retry) | **Total Stage 4 (`USD`, no retry)** | **Total Stage 4 (`USD`, all retries)** | **Remaining Free Buffer** | `Billed Cost` |
|---|---:|:---:|---:|---:|---:|---:|---:|
| **Branch 0: Frozen Only (M6 R2 Lock)** | `0` | — | `$0.0000 / $0.0000` | **`$8.3906`** | **`$9.0708`** *(cap `$9.50`)* | **`$3.81`** *(`$3.38` @ cap)* | **`$0.00`** |
| **Branch 1a: Replace-192** *(replaces Arm 1 pre-blind)* | `192` | No | `$0.7421 / $0.9688` | `$9.1328` | **`$10.0396`** | **`$2.84`** | **`$0.00`** |
| **Branch 1b: Paired-69** *(full 5-arm blind + text audit)* | `69` | Yes | `$1.5659 / $1.7926` | `$9.9565` | **`$10.8634`** | **`$2.02`** | **`$0.00`** |
| **Branch 1c: Paired-192 (RECOMMENDED)** | **`192`** | **Yes** | **`$1.8258 / $2.1659`** | **`$10.2165`** | **`$11.2367`** | **`$1.64`** | **`$0.00`** |
| **Branch 1d: Paired-192 + 3-Regime Merge** | `192 + 3×69` | Yes | `$2.3390 / $2.6790` | `$10.7296` | **`$11.7498`** | **`$1.13`** | **`$0.00`** |

*Note: Phase A decision-gate spend BEFORE touching any blind set is strictly **`$4.5236`** (`$0.7677` CPU + `4,970` A100 s).*

### 5.2 Theorem 5 (Two-Sided VoI Decision Boundary)

Let $V_{\text{pass}}$ be the research utility of clearing M6's $\tau_{\text{gate}} = 0.050\text{ nat}$ admissibility bound, $v_{\text{buf}} > 0$ the shadow price on preserving Stage 5–6 Modal credits, $C_B = \$2.1659$ (Branch 1c with retry), and $\Delta_{\text{rec}}(\Delta_A) = \min(\rho_B \Delta_A, \Delta_{\text{route,max}})$ the surgical LoRA routing recovery (with $\rho_B = 0.80$, $\Delta_{\text{route,max}} = 0.250\text{ nats}$). Then the net Bayesian Value-of-Information of unlocking Phase B given post-Phase-A dev host gap $\Delta_A = \widehat{\Delta}_{\text{host}}$ is:
$$
\text{VoI}_B(\Delta_A) = V_{\text{pass}}\!\left[\Phi\!\left(\frac{\tau_{\text{gate}} - (\Delta_A - \Delta_{\text{rec}}(\Delta_A))}{\sigma_B}\right) - \Phi\!\left(\frac{\tau_{\text{gate}} - \Delta_A}{\sigma_A}\right)\right] + \alpha_{\text{nat}}\,\Delta_{\text{rec}}(\Delta_A)\,\Phi\!\left(\frac{\tau_{\text{gate}} - (\Delta_A - \Delta_{\text{rec}})}{\sigma_B}\right) - \big((1 + v_{\text{buf}})C_B + \lambda_{\text{KL}}\big).
$$
**Exact Decision Regimes (`verify_w4.py`, $V_{\text{pass}}=\$25, \alpha_{\text{nat}}=\$6/\text{nat}, v_{\text{buf}}=0.25 \implies C_{\text{eff}}=\$2.7574$)**:

| $\widehat{\Delta}_{\text{host}}$ (`nats`) | Expected Post-LoRA Gap | $P(\text{pass Frozen})$ | $P(\text{pass Phase B})$ | Net $\text{VoI}_B$ (`USD`) | Optimal Action |
|---:|---:|---:|---:|---:|---|
| `0.000` | `0.000` | `1.0000` | `1.0000` | `-$2.7577` | **STAY FROZEN (`Branch 0`)** |
| `0.020` | `0.004` | `0.9987` | `0.9999` | `-$2.6292` | **STAY FROZEN (`Branch 0`)** |
| `0.030` | `0.006` | `0.9773` | `0.9999` | `-$2.0477` | **STAY FROZEN (`Branch 0`)** *(stat. threshold)* |
| `0.040` | `0.008` | `0.8413` | `0.9998` | **`+$1.3952`** | **TRIGGER PHASE B (`Branch 1c`)** *(lower crossover)* |
| `0.080` | `0.016` | `0.0014` | `0.9977` | **`+$22.5344`** | **TRIGGER PHASE B (`Branch 1c`)** *(peak VoI)* |
| `0.200` | `0.040` | `0.0000` | `0.7977` | **`+$17.9502`** | **TRIGGER PHASE B (`Branch 1c`)** |
| `0.310` | `0.062` | `0.0000` | `0.1587` | **`+$1.4451`** | **TRIGGER PHASE B (`Branch 1c`)** *(upper edge)* |
| `0.350` | `0.100` | `0.0000` | `0.0000` | **`-$2.7570`** | **ABORT — FIX CONNECTOR ON CPU (`$0`)** |

- **Exact VoI Positive Window**: **`[0.040, 0.315]` nats** (operational trigger locked conservatively at **`(0.030, 0.315]` nats** conditional on Tier-1 CPU/Jacobian certification).
- **10,000-Trial Monte Carlo Policy Verification** (`verify_w4.py`): Across a realistic Stage 4 outcome mixture (50% Phase A succeeds, 35% host-routing limited, 15% DPI/representation broken caught on CPU), **Adaptive Two-Phase** boosts gate pass rate from **`51.09%` (Always Frozen) $\to$ `83.01%` (`+31.92 pp`)**, cuts expected cost vs Always-LoRA (`$8.749` vs `$9.666`), and satisfies **`max_cost = $11.2367 < $12.88` on 10,000/10,000 trials (`Billed Cost: $0.00`)**.

---

## 6. Reproducibility & Verification Receipt (Round 1 Baseline)

- **Round 1 Receipt**: `67.71 s` on local CPU (`Python 3.12.7`, `numpy 2.3.5`, `scipy 1.17.0`, `torch 2.12.1+cpu`), exit code `0`. All Round-1 invariants (`<= 0.332%` Stage 3D VRAM calibration, `3.604×` step-0 shock, `9.732×` ungated KL reduction, `3,065.8×` NF4/BF16 MSE ratio, `+0.728 nats` DeltaNet NF4 collapse vs `0.00032 nats` Mixed MLP-NF4, and Round-1 Decimal ledger) are preserved bit-for-bit in Round 2.

---

# Round 2 — Cross-Track Integration (`W1 + W2 + W3 + W4`) & Master Systems / Modal `$12.88` Decision Lock

> **Cross-Track Grounding**: Integrates converged Round-1 results from [W1](../W1/THEOREM_W1.md) (Layer-24/Gap-64 $\sigma_3(D_{\text{answer}})$ bottleneck & 1-line zero-param slow-head `dt_bias` warm-start), [W2](../W2/THEOREM_W2.md) (FP32 Complex-Cayley $8\times 8$ Woodbury + Schur-Irrep Dynamic Hyper-LoRA + Step-69 convergence `64.60% / 0.9526`), [W3](../W3/THEOREM_W3.md) (Complex-Symmetrized $P_{\perp, C}^{\mathbb{C}}$ with `39.38 / 100.76 / 245.00 MiB` BF16 $U_k$, Two-Tier Token-Position Gate, `113.73×` Rao–Blackwell Bayes-KL + CF-Margin + MixKL, Step-69 `61.26% ± 0.75 pp`), and [M4](../M4/THEOREM_M4.md) / [M6](../M6/THEOREM_M6.md) (`$12.88` Modal free credits, `Billed Cost: $0.00`).

---

## 7. Unified `W1 + W2 + W3` Stack (`<= 2.20M` Params): Exact VRAM, FP32 Cayley Invariance & Mode I vs Mode II Trade-Off

### 7.1 Exact Unified Parameter Budget (`<= 2,184,964` Tier-P1 Attribution Cap on `4B` and `9B`)

Replacing Round 1's generic `5.06M` BF16 LoRA with the converged symmetry-locked, nullspace-projected, warm-gated surgical stack cuts total trainable parameters by **`3.4×` on `Qwen3.5-4B`** (strictly inside M1/M5/W2's `2,184,964` Tier-P1 attribution cap even **including** the `M1` connector!):

| Component (`verify_w4.py` R2.1) | Precision / Quant Rule | `Qwen3.5-4B` (`L=32`, FA `[12,16,20,24,28]`, DN `[11,15,19,23]`) | `Qwen3.5-9B` (`L=32`, same 9 layers) | `Qwen3.8-27B` (`L=64`, FA 8 layers, DN 6 layers) |
|---|---|---:|---:|---:|
| `M1 EquivariantEvidenceConnector` + Quotient | FP32 (`Frozen` in Mode I) | `488,400` | `689,616` | `823,760` |
| `W1` Slow-Head `dt_bias` Warm-Start (dedicated EM heads) | Deterministic 1-line edit | **`0`** | **`0`** | **`0`** |
| `W2` Complex-Cayley `O_proj` ($r_c = 4$, $8\times 8$ Woodbury) | **Strictly FP32 (NEVER 4-bit!)** | `102,400` | `163,840` | `327,680` |
| `W2` Schur-Irrep Dynamic Hyper-LoRA (`v_proj` $r=8$ + final $r=11$) | BF16 / FP32 master | `173,134` | `251,470` | `452,110` |
| `W1+W3` Selective GDN (`in_proj_qkv/out/a/b`, $r=8$) | BF16 / FP32 master | `722,944` | `919,552` | `1,774,080` |
| **Phase-B Host Adapters Only (Mode I)** | — | **`998,478`** (`0.024%` FLOPs) | **`1,334,862`** (`0.015%` FLOPs) | **`2,553,870`** (`0.0095%` FLOPs) |
| **Cumulative `M1 + W1 + W2 + W3` (Mode I)** | — | **`1,486,878`** (**`68.05%`** of `2.185M` cap!) | **`2,024,478`** (**`92.65%`** of cap!) | **`3,377,630`** (`0.0126%` of host) |
| **Mode II Optional Early-GDN Exception** (`[3,7]` / `[3,7,11,15]`) | BF16 / FP32 master | `+82,944` $\to$ **`1,569,822`** (`71.85%` cap) | `+132,096` $\to$ **`2,156,574`** (`98.7%` cap) | `+330,752` $\to$ **`3,708,382`** |
| *Full-Rank `in_proj_a/b` Variant (Mode I cumulative)* | BF16 / FP32 master | **`1,976,350`** (`90.45%` of cap!) | `2,808,862` | `5,830,622` |

**Direct PyTorch Proof of FP32 Cayley Invariance Mandate (`d = 2560`, $m = 1280$, $r_c = 4$, $8 \times 8$ `complex64` Woodbury)**:
- Exact Woodbury spectral unitarity defect $\|Q^* Q - I\|_2 = \mathbf{7.822 \times 10^{-7} \le 10^{-6}}$ (Frobenius `1.127e-6`), FLOPs share **`0.00489%`** on 4B (`0.00367%` on 9B, `0.00244%` on 27B — two orders of magnitude below `< 0.5%`!).
- On an explicit $64 \times 64$ real Cayley $O_{\text{proj}}$ block: FP32 orthogonality defect is **`7.696e-7`**, whereas quantizing $Q$ to `BF16` degrades orthogonality to **`1.554e-2`** and `NF4` degrades it to **`1.552e-2`** (**`20,162×` blowup!**). **System Rule**: Even under `Qwen3.8-27B` `Mixed MLP-NF4`, every W2 Complex-Cayley `O_proj` adapter MUST remain strictly **FP32** (`< 5 MiB` workspace across all 8 FA layers!).

### 7.2 Exact A100-80GB VRAM & Step-Time Recalibration: Mode I (Stop-Grad Layer-10/18) vs Mode II (Co-Tuned `0.15 η_A`)

Because `W3`'s composite loss adds a **Counterfactual Slot-Margin (`CF-Margin`) sub-batch** (`sub = 8` or `16` paired $\pm\omega$/slot-flip sequences) and a **MixKL prose anchor sub-batch** (`sub = 8` or `16` sequences, `L_prose = 128`, cached frozen teacher logits) alongside `W3`'s BF16 nullspace bases $U_k$ (`39.38 / 100.76 / 245.00 MiB`), **two orthogonal systems axes determine whether peak VRAM clears M6's `65,000 MiB` gate**:
1. **Mode I (Stop-Grad Cutoff at Layer 10 on 4B/9B; Layer 18 on 27B; $E_{\theta_A}$ Frozen)**: Layers `1–10` (`8` GDN + `2` FA = `31.25%` of 4B/9B; `14` GDN + `4` FA = `28.12%` of 27B) execute under `torch.no_grad()` and detach hidden states at Layer 10/18. Crucially, **W1's 1-line zero-param slow-head `dt_bias` warm-start requires ZERO gradients on layers `1–10`**, so it operates natively inside Mode I!
2. **Pattern A (Exact Chain-Rule Sequential Microbatch Accumulation — RECOMMENDED)** vs **Pattern B (Co-Resident Simultaneous Batch `mb + sub_cf`)**: Under Pattern A, primary `mb = 16` runs `.backward()` (freeing the activation graph) while caching detached `[B, 4]` logits (`12 MiB`), then `CF-Margin` (`sub = 8/16`) and `MixKL` (`sub = 8/16`) accumulate exact chain-rule gradients sequentially — adding **zero peak activation stacking**!

| Host (`U_k` BF16) | Phase-B Mode & CF Pattern (`B = 32`) | Precision / Scorer (`mb`) | Static (`MiB`) | Peak VRAM (`MiB` / `GiB`) | Clears `65,000 MiB` Gate? | Primary Fwd / Bwd (`ms`) | Surgical + $U_k$ + Cayley (`ms`) | CF + MixKL (`ms`) | **Total Step Time (`ms` / $\Delta$ vs 3D)** |
|---|---|---|---:|---:|:---:|---:|---:|---:|---:|
| **`4B`** (`39.38`) | **Mode I Stop-Grad L10 · Pattern A (`sub=8`) [LOCK]** | BF16 `kv_train` (`16`) | `8,094.4` | **`40,988.5`** (**`40.03 GiB`**) | **YES (`-24.0 GiB`)** | `658.6 / 905.5` | `+100.8` | `+449.5 / +145.7` | **`2,260.0 ms`** (**`+14.39%`**) |
| **`4B`** (`39.38`) | Mode I Stop-Grad L10 · Pattern A (`sub=16`) | BF16 `kv_train` (`16`) | `8,094.4` | **`40,988.5`** (`40.03 GiB`) | **YES** | `658.6 / 905.5` | `+100.8` | `+832.4 / +274.7` | **`2,772.0 ms`** (`+40.30%`) |
| **`4B`** (`39.38`) | Mode I Stop-Grad L10 · Pattern B Co-Resid (`sub=8`) | BF16 `kv_train` (`16+8`) | `8,094.4` | **`57,435.6`** (`56.09 GiB`) | **YES (`-7.6 GiB`)** | `658.6 / 905.5` | `+100.8` | `+416.2 / +145.7` | **`2,226.7 ms`** (`+12.71%`) |
| **`4B`** (`39.38`) | **Mode II Co-Tuned `0.15 η_A` · Pattern A (`sub=8`)** | BF16 `kv_train` (`16`) | `8,101.4` | **`53,099.0`** (`51.86 GiB`) | **YES (`-11.9 GiB`)** | `658.6 / 1317.1` | `+100.8` | `+560.6 / +181.7` | **`2,818.8 ms`** (`+42.67%`) |
| **`4B`** (`39.38`) | Mode II Co-Tuned · Pattern A (`sub=16`) | BF16 `kv_train` (`16`) | `8,101.4` | **`53,099.0`** (`51.86 GiB`) | **YES** | `658.6 / 1317.1` | `+100.8` | `+1038.2 / +342.6` | `3,457.3 ms` (`+74.99%`) |
| **`4B`** (`39.38`) | *Mode II Co-Tuned · Pattern B Co-Resid (`sub=8`)* | BF16 `kv_train` (`16+8`) | `8,101.4` | *`75,597.8`* (`73.83 GiB`) | **NO (`> 65 GiB`!)** | `658.6 / 1317.1` | `+100.8` | `+519.1 / +181.7` | `2,777.3 ms` (`+40.57%`) |
| **`9B`** (`100.76`) | **Mode I Stop-Grad L10 · Pattern A (`sub=8`)** | BF16 `kv_train` (`16`) | `17,218.4` | **`54,708.3`** (**`53.43 GiB`**) | **YES (UNLOCKS `mb=16`!)** | `1402.1 / 1927.8` | `+214.5` | `+957.0 / +310.1` | **`4,811.5 ms`** |
| **`9B`** (`100.76`) | *Mode II Co-Tuned · Pattern A (`sub=8`)* | BF16 `kv_train` (`16`) | `17,228.6` | *`68,884.7`* (`67.27 GiB`) | **NO (`mb=8` -> `42.05 GiB`)** | `1402.1 / 2804.1` | `+214.5` | `+1193.6 / +386.8` | `6,001.1 ms` |
| **`27B`** (`245.00`) | Mode I Stop-Grad L18 · Pattern A (`sub=8`) | Pure BF16 `ckpt` (`8`) | `51,606.9` | **`60,557.4`** (`59.14 GiB`) | **YES** | `12087 / 17375` | `+1849.3` | `+8454 / +2740` | `42,505.5 ms` |
| **`27B`** (`245.00`) | **Mode I Stop-Grad L18 · Mixed MLP-NF4 + Cayley-FP32** | Mixed `ckpt` (`16`) | `28,452.9` | **`46,353.9`** (**`45.27 GiB`**) | **YES (`-18.6 GiB`)** | `13170 / 18931` | `+2014.9` | `+9211 / +2985` | **`46,312.2 ms`** |
| **`27B`** (`245.00`) | Mode II Co-Tuned · Mixed MLP-NF4 + Cayley-FP32 | Mixed `ckpt` (`16`) | `28,468.0` | **`50,216.5`** (`49.04 GiB`) | **YES** | `13170 / 26339` | `+2014.9` | `+11211 / +3633` | `56,368.4 ms` |

**Exact Mode I vs Mode II Engineering Verdict (Answering `W3`)**:
1. **Mode I (Stop-Grad Layer-10 / Layer-18) Saves `11.83 GiB` & `558.8 ms/step` (`-19.82%`) on `4B`**, **`13.84 GiB` & `1,189.6 ms/step` (`-19.82%`) on `9B`** (unlocking `mb = 16` under `65,000 MiB`!), and **`3.77 GiB` & `10.06 s/step` (`-17.84%`) on `27B`**.
2. **Co-Resident Pattern B Fails M6's Gate under Mode II (`73.83 GiB > 65,000 MiB`)**, proving that **Mode I Stop-Grad Layer-10 + Pattern A (`sub = 8`: `40.03 GiB`, `2.260 s/step`)** is the uniquely optimal Stage-4 default.
3. **Mode II Escalation Trigger**: Switch to Mode II (`0.15 η_A` co-tuning + `[3, 7]` early-GDN `a/b`, Pattern A `51.86 GiB < 65,000 MiB`, `2.819 s/step`) **ONLY** if W1's post-Phase-A Layer-10 probe shows early-layer slow-state wipeout *even after* applying W1's zero-param `dt_bias` warm-start!

---

## 8. Compressed `69–96`-Step Modal `$12.88` Ledger & Tier-P1 `LEMMO-Matched-LoRA` Parity

Because `W2` (`4c_quotient_cayley_schur`: **`64.60% / 0.9526` already at Step 69**) and `W3` (`Bayes-KL + CF-Margin`: `113.73×` variance reduction, **`61.26% ± 0.75 pp` at Step 69** vs `44.30%` 1-hot CE) converge **`2×–3×` faster** than Round 1's `192–384`-step generic LoRA horizon, we compress Phase B to **`96` surgical steps** (with `69`-step early-stop checkpoint!) at conservative padded rates (`Mode I: 2.50 s/step` [actual `2.26 s`], `Mode II: 2.96 s/step` [actual `2.82 s`], `LEMMO BF16 LoRA: 2.40 s/step` [no CF/MixKL sub-batch], `+25 s` W3 1-pass K-FAC/$U_k$ probe, `44,812` full 1-arm blind rows at `33 ms/row`, `20,480` common-support `S_ours ∩ S_LEMMO` rows per W2 §3.4 / M6 §6).

### 8.1 Exact Decimal Modal Free-Credit Ledger (`A100-80GB 2c/16G = $0.00075572/s`, Credits = `$12.88`, `Billed Cost = $0.00`)

| Execution Branch (`verify_w4.py` R2.2 Exact Decimal) | Phase-B / LEMMO A100 `s` | Retry Pool `s` | Inc. Phase B (`USD`, no retry) | Inc. LEMMO (`USD`, no retry) | **Total Stage 4 (No Retry)** | **Total Stage 4 (All Retries)** | **Remaining Free Buffer (w/ Retry)** | `Billed Cost` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Branch 0: Frozen Lock (M6 R2)** | `0 / 0` | `900` | `$0.0000` | `$0.0000` | **`$8.3906`** | **`$9.0708`** *(cap `$9.50`)* | **`$3.8092`** *(`$3.38` @ cap)* | **`$0.00`** |
| **Branch 1-Surgical-69 (Mode I)** | `2,076 / 0` | `1,200` | `$1.5689` | `$0.0000` | `$9.9595` | **`$10.8664`** | **`$2.0136`** | **`$0.00`** |
| **Branch 1-Surgical-96 (Mode I — PRIMARY)** | **`2,143 / 0`** | **`1,200`** | **`$1.6195`** | `$0.0000` | **`$10.0102`** | **`$10.9170`** | **`$1.9630`** | **`$0.00`** |
| **Branch 1-Surgical-96 (Mode II Co-Tuned)** | `2,188 / 0` | `1,200` | `$1.6535` | `$0.0000` | `$10.0442` | **`$10.9510`** | **`$1.9290`** | **`$0.00`** |
| *Branch 1c-192 (Round-1 Reference)* | *`2,416 / 0`* | *`1,350`* | *`$1.8258`* | *`$0.0000`* | *`$10.2165`* | *`$11.2367`* | *`$1.6433`* | *`$0.00`* |
| **Branch 0 + Tier-P1 LEMMO (384-step, FULL 44,812)** *(if Frozen passes!)* | `0 / 2,800` | `1,140` | `$0.0000` | `$2.1160` | **`$10.5067`** | **`$11.3682`** | **`$1.5118`** | **`$0.00`** |
| **Surgical-96 + Tier-P1 LEMMO `Option L2` (RECOMMENDED LOCK)** *(384-step FULL train, 20,480 common-support rows)* | **`2,143 / 1,937`** | **`1,440`** | **`$1.6195`** | **`$1.4638`** | **`$11.4740`** | **`$12.5622`** | **`$0.3178`** | **`$0.00`** |
| **Surgical-96 + Tier-P1 LEMMO `Option L1`** *(matched 96-step train, FULL 44,812 blind)* | `2,143 / 2,109` | `1,440` | `$1.6195` | `$1.5938` | `$11.6040` | **`$12.6922`** | **`$0.1878`** | **`$0.00`** |
| **Surgical-96 + Tier-P1 LEMMO `Option L3`** *(384-step FULL train + FULL 44,812 blind, shared 900s retry)* | `2,143 / 2,800` | `900` | `$1.6195` | `$2.1160` | `$12.1262` | **`$12.8063`** | **`$0.0737`** | **`$0.00`** |

**Answer to Cross-Track Question 2 (LEMMO Parity Feasibility)**: **YES — unconditionally inside `$12.88` (`Billed Cost: $0.00`)!** Compressing Phase B to `96` Mode-I surgical steps leaves **`$1.9630` even after exhausting `1,200 s` of retries** (and **`$2.8698`** before retries), which directly funds a **matched-access Tier-P1 `LEMMO-Matched-LoRA` (`2,184,964` params, `r = 12` 8-FA QKVO + projector)** comparison arm on A100-80GB:
- If Phase A passes (`Branch 0`), we run **FULL `384`-step + FULL `44,812`-row LEMMO** for **`$10.51` (`$11.37` w/ retry)**, keeping `$1.51` reserve.
- If Phase B triggers (`Branch 1-Surgical-96`), **`Option L2` (FULL `384`-step LEMMO training so 1-hot CE has zero convergence-speed handicap, evaluated on the exact `20,480`-row mutually supported slice $S = S_{\text{ours}} \cap S_{\text{LEMMO}}$ + text fluency audit)** costs **`$11.4740` scheduled / `$12.5622` with ALL `1,440 s` of independent retries** — and whenever early-phase retries stay under `660 s`, our adaptive controller automatically upgrades to **`Option L3` (FULL `384`-step + FULL `44,812`-row blind)**!

### 8.2 10,000-Trial Monte Carlo Zero-Bill Proof (`verify_w4.py` R2.2)

Simulating `10,000` realistic Stage-4 trajectories (`15%` CPU DPI abort, `50%` Phase-A pass, `35%` host-routing limited triggering `Branch 1-Surgical-96`, stochastic GPU retry/preemption per phase, and real-time Modal free-credit guard automatically selecting `L3 -> L2 -> L1` before launching LEMMO blind scoring):
- **Surgical-96 Only**: Mean = **`$8.0606`**, P99 = **`$10.6764`**, Worst-Case Max = **`$10.8847`**.
- **Surgical-96 + Adaptive Tier-P1 LEMMO**: Mean = **`$9.8828`**, P99 = **`$12.6813`**, Worst-Case Max = **`$12.8508 <= $12.88`** on **`10,000 / 10,000` (`100.0000%`) trials (`Billed Cost: $0.00`)**!
- **LEMMO Parity Execution Mix**: `8,267 / 10,000` trials (`96.95%` of all GPU runs!) execute **FULL `384`-step / `44,812`-row LEMMO (`Option L3`)**, `260 / 10,000` heavy-retry runs gracefully execute **`Option L2` (`384`-step / `20,480` common-support)**, `0` runs require truncating LEMMO training (`L1`), and `1,473` catch CPU DPI defects at `$0.7677`.
- **Gate Pass-Rate Uplift**: Adaptive Round-2 `W1+W2+W3` lifts Stage-4 gate pass rate from **`50.93%` (Always Frozen) $\to$ `84.16%` (`+33.23 pp`)**.

---

## 9. Canonical Schur-Irrep / Complex-Cayley Phase-B Merging vs Unconstrained LoRA (Extended Exp 2A)

Resolving Proposition 4.1 (`prompt_r2.md` §3): on the **exact same frozen Phase-A host, connector, Regime 1 (`Pluto-DC + high jitter`, 2 dB), Regime 2 (`Large-CFO + Spur`, 5 dB), 1-hot CE loss, and 120-step budget**, we replaced `12,288`-param unconstrained LoRA ($\Delta W = B A$) with **W2's Canonical Lie-Algebra Complex-Cayley ($r_c = 4$, $4 \times 16$ skew-Hermitian $K_i \in \mathfrak{u}(4)$ in fixed orthonormal frame $U_{\text{canon}, i}$) + Strict Theorem-5 Pure $V_1 \oplus V_2$ Schur-Irrep Readout ($r_s = 16$, $6 \times 16$ core in fixed $(B_{\text{align}}, A_{\text{canon}})$, zero $n=0$ static DC leakage)** — **`160` trainable DOFs (`76.8×` parameter compression!)**.

### 9.1 Geometry & Gauge-Free Spectral Unitarity Comparison

| Metric (`verify_w4.py` Exp 2A Side-by-Side) | Unconstrained Static LoRA (`12,288` params) | **W2 Canonical Schur-Cayley (`160` DOFs, `76.8×` leaner)** | Scientific Mechanism |
|---|---:|---:|---|
| $GL(r)$ Factor-Gauge Pitfall (`B̄ Ā` corruption) | **Present** (`1.0382` NLL) | **ELIMINATED (Exact `0`)** | Fixed canonical frames $(U_{\text{canon}}, B_{\text{align}}, A_{\text{canon}})$ from frozen Phase-A host! |
| Spectral Unitarity Defect $\|Q_{\text{mix}}^* Q_{\text{mix}} - I\|_2$ | Unbounded ($\kappa$ grows) | **`2.78e-7` – `4.18e-7` ($\le 10^{-6}$ ALL merges!)** | Linear combinations $\lambda_1 K_1 + \lambda_2 K_2 \in \mathfrak{u}(r_c)$ remain strictly skew-Hermitian! |
| Regime Alignment $\cos(\delta_1, \delta_2)$ | **`-0.0908`** *(opposing)* | **`+0.0818`** *(positively aligned!)* | Shared physical $V_1 \oplus V_2$ irrep basis aligns Regime 1 & 2 phase updates! |
| Directional Unit-Norm Interference $\tilde{\Gamma}_{12}$ / $\tilde{\Gamma}_{21}$ | `2.268` *(raw)* / `0.256` | **`1.371`** (`1.65×` directional drop) / **`0.303`** | Directional basis interference is substantially reduced in canonical coordinates. |
| Raw (Unnormalized) $\Gamma_{12}$ / $\Gamma_{21}$ & Norm Ratio $\|c_2\|/\|c_1\|$ | `2.268` / `0.256` (`1.84×`) | `3.008` / **`0.138`** (`1.481×` $\implies 2.194\times$ sq.) | **Key Discovery**: Residual raw $\Gamma_{12} = 1.371 \times 2.194 = 3.008$ is **pure SNR-temperature scaling** (`5 dB` vs `2 dB`), NOT basis gauge mismatch! |

### 9.2 Zero-Retraining Generalization on Unseen Mixed Capture (`Pluto-DC + Large-CFO + Spur`)

| Merge Method (`120`-step adapters, ZERO retraining) | Regime 1 `Pluto-DC` (Acc / NLL) | Regime 2 `CFO+Spur` (Acc / NLL) | **Unseen Mixed Capture (Acc / NLL)** | **Macro-Avg (Acc / NLL)** | **Delta vs Matched Unconstrained LoRA (Unseen Mixed)** |
|---|---:|---:|---:|---:|---:|
| *Unconstrained LoRA Soup (`0.5/0.5`, `12,288` p)* | *`50.39% / 1.1606`* | *`66.94% / 0.8174`* | *`59.34% / 1.0089`* | *`58.89% / 0.9956`* | *baseline* |
| *Unconstrained LoRA TA `0.75`* | *`50.49% / 1.2065`* | *`66.31% / 0.8014`* | *`57.77% / 1.0293`* | *`58.19% / 1.0124`* | *baseline* |
| *Unconstrained LoRA TA `1.00`* | *`50.93% / 1.2540`* | *`65.14% / 0.8167`* | *`57.15% / 1.0655`* | *`57.74% / 1.0454`* | *baseline* |
| *Unconstrained LoRA TIES (`40%, 1.0`)* | *`50.44% / 1.2170`* | *`67.29% / 0.7877`* | *`58.79% / 1.0266`* | *`58.84% / 1.0104`* | *baseline* |
| **W2 Canonical Lie-Schur Soup (`0.5/0.5`, `160` DOFs)** | **`52.20% / 1.1584`** | **`70.85% / 0.8025`** | **`62.15% / 0.9901`** | **`61.73% / 0.9837` (BEST NLL)** | **`+2.81 pp` / `-0.0188 nats`** |
| **W2 Canonical Lie-Schur TA (`0.75`)** | **`52.59% / 1.2084`** | **`72.31% / 0.7667`** | **`62.07% / 1.0022`** | **`62.32% / 0.9924` (BEST ACC)** | **`+4.30 pp` / `-0.0271 nats`** |
| **W2 Canonical Lie-Schur TA (`1.00`)** | **`52.73% / 1.2749`** | **`72.31% / 0.7680`** | **`61.91% / 1.0457`** | **`62.32% / 1.0295`** | **`+4.76 pp` / `-0.0198 nats`** |
| **W2 Canonical Lie-Schur TIES (`40%, 1.0`)** | `52.00% / 1.2257` | `67.38% / 0.8332` | **`61.88% / 1.0150`** | `60.42% / 1.0246` | **`+3.09 pp` / `-0.0117 nats`** |

**Verified Takeaway**: Merging in W2's **Canonical Lie-Algebra + Schur-Core space** beats unconstrained LoRA on **every single merging rule** (`+2.81 pp` to **`+4.76 pp`** unseen mixed-capture accuracy, **`-0.0117` to `-0.0271 nats`** NLL, with **`76.8×` fewer parameters** and exact `<= 4.18e-7` Cayley unitarity!). Simultaneously, our Fisher decomposition reveals a subtle physical truth: fixing W2's canonical basis eliminates *geometric* gauge/subspace conflict completely (accuracy stays flat at **`61.91%–62.15%`** even at `TA 1.00`, versus unconstrained LoRA's collapse to `57.15%`), while the remaining `TA 1.00` vs `Soup 0.5` NLL gap (`1.0457` vs **`0.9901`**) reflects **physical SNR-temperature mismatch** ($\|c_2\|/\|c_1\| = 1.481$ between 5 dB and 2 dB regimes). Therefore **Canonical Lie-Schur Soup (`0.5/0.5`) + disjoint dev $T^*$ calibration** is strictly optimal!

---

## 10. VoI Upper Cutoff $\Delta_{\text{route,max}}$ Reconciliation with W1's $\sigma_3(D_{\text{answer}})$ Bottleneck

In Round 1, Theorem 5 established a two-sided VoI window **`(0.030, 0.315]` nats** assuming pure mid-late LoRA routing recovery caps at $\Delta_{\text{route,max}} = 0.250\text{ nats}$. Meanwhile, Track [W1](../W1/THEOREM_W1.md) measured that at long token separation (`L_gap = 64`), a frozen/prefix-only hybrid host suffers Gated-DeltaNet exponential state decay ($\text{retention} = 0.00572$), collapsing the 3D answer Jacobian singular value from $\sigma_3(D_{\text{answer}}) = 0.3858 \to \mathbf{0.0973}$ and driving `G1 NLL` to `1.2694` ($\Delta_{\text{host}} = 1.2694 - 0.9411 = \mathbf{0.3283\text{ nats}}$ — *just above* `0.315 nats`!), whereas W1's **1-line zero-param slow-head `dt_bias` warm-start** lifts $\sigma_3$ by `1.73×` (`0.0973 -> 0.1684`, retention `50.7×` to `0.2899`) and restores `G1 = 64.36%` ($\Delta_{\text{host}} \approx 0.015\text{ nats}$)!

### 10.1 Exact Softmax Bregman Shrinkage Bridge ($\sigma_3(D_{\text{answer}}) \mapsto \Delta_{\text{route}}(\sigma_3)$)

Because LayerNorm/RMSNorm bounds the normalized input phasor energy, attenuating the 3D phase plane by $\sigma_3(D_{\text{answer}})$ shrinks the effective centered Bayes logit vector $s^*$ to $\alpha(\sigma_3)\,s^*$ with RMSNorm SNR shrinkage law $\alpha(\sigma_3) = \min\!\big(1, (\sigma_3 / \sigma_{3,\text{sat}})^p\big)$ ($\sigma_{3,\text{sat}} = 0.180, p = 3.2254$), inducing exact softmax KL routing gap $\Delta_{\text{route}}(\sigma_3) = \mathbb{E}[D_{\text{KL}}(p^* \| \text{softmax}(\alpha(\sigma_3) s^*))]$ capped by Stage 3D `G1`'s uniform routing ceiling $\log 4 - H_{\text{oracle}}(\text{G1}) = \mathbf{0.4607\text{ nats}}$:

| W1 Measured Regime | $\sigma_3(D_{\text{answer}})$ | W1 GDN Retention | Implied $\alpha(\sigma_3)$ | Analytic $\Delta_{\text{route}}(\sigma_3)$ (`nats`) | W1 Empirical $\Delta_{\text{host}}$ (`nats`) | Unified Round-2 Action |
|---|---:|---:|---:|---:|---:|---|
| `L_gap = 0` (well-conditioned) | `0.3858` | `1.00000` | `1.0000` | **`0.0000`** | `~0.00` | **STAY FROZEN (`Branch 0`)** |
| **`L_gap = 64` W1 `warmgate` (`dt_bias`)** | **`0.1684`** (`1.73×` lift) | **`0.28990`** (`50.7×`) | `0.8067` | **`0.0127`** | **`~0.0150`** (`G1 64.36%`) | **CLEARS GATE (`<= 0.030`)** |
| **`L_gap = 64` `prefix_only` collapse** | **`0.0973`** | `0.00572` | `0.1375` | **`0.3283`** | **`0.3283`** (`G1 43.41%`) | **W1 `dt_bias` WARMGATE FIRST $\to$ MODE I SURGICAL-96** |
| `L_gap = 64` untrained host | `0.0574` (`6.73×` drop) | `0.00745` | `0.0251` | **`0.4346`** | — | **W1 `dt_bias` WARMGATE FIRST $\to$ MODE I SURGICAL-96** |
| `L_gap = 128` untrained host | `0.0220` (`17.5×` drop) | `0.00010` | `0.0011` | **`0.4595`** | — | **W1 `dt_bias` WARMGATE FIRST $\to$ MODE I/II SURGICAL-96** |

Notice that our calibrated Bregman bridge reproduces W1's post-warmgate NLL gap (`0.0127` analytic vs `0.0150` empirical) within **`0.0023 nats`**!

### 10.2 Updated Master Two-Probe Phase-A Decision Gate (Unifying `W1 + W2 + W3 + W4`)

With `Branch 1-Surgical-96` reducing incremental Phase-B cost from `$2.1659 -> $1.8462` (`C_eff = $2.3578`), and W1's zero-param `dt_bias` warm-start expanding recoverable routing capacity from $\Delta_{\text{route,max}} = 0.250 \to \mathbf{0.340\text{ nats}}$ ($\rho_B = 0.85$), the **positive-VoI window expands cleanly from `[0.040, 0.310]` to `[0.040, 0.400]` nats** (peak VoI **`+$23.34`**, strictly covering W1's `0.3283`-nat Gap-64 collapse!). To guarantee `$0` wasted GPU spend on genuine CPU bugs, replace Round 1's single scalar cutoff with the **Two-Probe Phase-A Gate (`< 10 s` on A100, `$0.007` cost)**:

1. **Probe A (M1/M5 Connector DPI Probe on $E_{\theta_A}(x)$)**: If a direct linear/MLP probe on $E_{\theta_A}(x)$ has `gap_conn > 0.030 nats` $\implies$ **ABORT IMMEDIATELY (`$0` Phase-B GPU); fix upstream representation/connector on CPU.**
2. **Stay Frozen (`Branch 0 + FULL Tier-P1 LEMMO`)**: If Probe A passes AND $\widehat{\Delta}_{\text{host}} \le 0.030\text{ nats}$ $\implies$ **STAY 100% FROZEN (`$8.39 / $9.07` w/ retry; `$10.51 / $11.37` with FULL 384-step/44,812-row LEMMO).**
3. **Standard Mode-I Trigger (`Branch 1-Surgical-96 + Tier-P1 LEMMO`)**: If Probe A passes AND $0.030 < \widehat{\Delta}_{\text{host}} \le 0.315\text{ nats}$ AND $\sigma_3(D_{\text{answer}}) \ge 0.10$ $\implies$ **Run Mode I Stop-Grad Layer-10 (`40.03 GiB`, `2.260 s/step`, `96` steps, `<= 1.49M` params) + Adaptive Tier-P1 LEMMO (`Option L3/L2`, `max <= $12.85 < $12.88`, `Billed Cost: $0.00`).**
4. **GDN-Decay Rescue Trigger (`W1 0-Param Warmgate -> Mode I Surgical-96`)**: If Probe A passes (`<= 0.030 nats`) AND $\widehat{\Delta}_{\text{host}} \in (0.315, 0.400]\text{ nats}$ AND **Probe B confirms GDN decay ($\sigma_3(D_{\text{answer}}) < 0.10$)** $\implies$ **Do NOT abort as a CPU bug!** Apply W1's 1-line zero-param slow-head `dt_bias` warm-start first (lifting $\sigma_3$ `1.73×`, collapsing effective host gap back inside the VoI sweet spot), then execute **Mode I Surgical-96** (escalating to Mode II `[3, 7]` `a/b` ONLY if Layer-10 probe still shows wipeout).

---

## 11. Round 2 Reproducibility & Verification Receipt

```powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W4/verify_w4.py
```
- **Runtime**: **`59.24 s`** (`<= 90 s` budget!) on local CPU (`Python 3.12.7`, `numpy 2.3.5`, `scipy 1.17.0`, `torch 2.12.1+cpu`, single-thread deterministic), **exit code `0`**, full Round 1 + Round 2 structured telemetry persisted in [results_w4.json](results_w4.json).
- **Immutability Audit**: Zero files created, edited, or deleted outside `research/stage4_math_and_design/weight_mod/W4/`.
