# Track W4 (Round 1): Systems/VRAM Equations, QLoRA Quantization Distortion, Anti-Shock Curriculum, Adapter Merging & Modal `$12.88` Decision Boundary

You are **Track W4**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously answering from **systems mathematics, optimization dynamics, quantization theory, model merging, and Bayesian Value-of-Information**:
> **Given `A100-80GB` hardware and `~$12.88` remaining Modal free credits (`Billed Cost: $0.00`), how do direct weight modifications compare in VRAM, step speed, quantization distortion, training stability, multi-dataset merging, and USD cost — and what is the EXACT mathematical decision boundary for staying frozen vs triggering surgical weight updates?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/weight_mod/W4/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `research/stage4_math_and_design/weight_mod/W4/`.
- **Python Environment Guardrail**: **ALWAYS** run Python via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```

## 1. Grounded Empirical Context (Inspect Read-Only First)
1. `research/stage3d_qwen_signed_phase/REPORT.md` & `PROTOCOL_LOCK.json`:
   - Real A100-80GB telemetry for frozen `Qwen/Qwen3.5-4B` + `SignedPhaseCacheConnector` (`2.185M` params, `BF16` host, `grad_checkpointing=True`):
     - Train (`micro_batch = 16`, `batch = 32`): **peak VRAM `52,972.5 MB`** (`~51.73 GiB`), **`1.9757 s / opt step`**;
     - Eval (`batch = 64`, KV-cached): **peak VRAM `45,409.9 MB`**, **`23.88 ms / logical row`**;
     - `$17.12` used -> **`~$12.88` remaining free Modal compute credits** (`Billed Cost: $0.00`).
2. Converged Stage 4 Frozen-Host Baseline (`M4/THEOREM_M4.md`, `M6/THEOREM_M6.md` Round 1/2):
   - Pure A100-80GB frozen-host plan (`1` seed across `4` arms + Pluto `2,240` + CTTC `1,440` + synthetic `40,960` + `69/384`-step pilot gate): **`$8.44–$9.50` cap** (leaving **`$3.38–$4.44`** free buffer inside `$12.88`!).

## 2. Deep Multi-Angle Mathematical & Engineering Questions for Track W4
1. **Exact VRAM, Autograd-Checkpointing & Optimizer Memory Equations (`Qwen3.5-4B -> 9B -> 27B`)**:
   - Calibrate exact component memory decomposition against Stage 3D's `52,972.5 MB` (train) and `45,409.9 MB` (eval) on `A100-80GB`:
     - Decompose static weights (`~8 GB` BF16), candidate scoring logit workspace (`mb * 4 * L_cand * |V| * 4B` FP32 softmax!! Notice `16 * 4 * ~10 * 151,936` logits/intermediates!), activation checkpoints + Gated-DeltaNet recurrent state graphs, and optimizer state (`m, v, master` in FP32 vs BF16 vs 8-bit Adam).
     - Derive exact memory and throughput delta $\Delta M_{\text{VRAM}}, \Delta t_{\text{step}}$ for:
       1. Full FT (4B $\times$ 12–16B $\implies$ exact OOM on A100-80GB);
       2. Selective-Layer BF16 LoRA (`r = 4..16`, check peak `< 65 GiB` microbatch `mb = 8` or `16`!);
       3. **QLoRA (`NF4` 4-bit host quantization)**: Derive the **exact NF4 Weight-Quantization Phase Distortion Bound** — when weight rounding error $\epsilon_q \sim \mathcal{U}[-\Delta/2, \Delta/2]$ propagates through $L$ layers onto delicate continuous phase ratios (`log-odds` requiring `< 0.05` nats precision at `+20 dB` CTTC vs `2 dB` Pluto!), quantify whether NF4 degrades continuous phase fidelity OR saves enough VRAM on `9B / 27B`!
2. **Cold-Start Gradient-Shock Inequality & Two-Phase Anti-Shock Curriculum**:
   - Prove rigorously why jointly optimizing `(untrained connector E_theta + LoRA Delta_W)` from step 0 creates a **Gradient-Shock Transient**: backpropagated gradients $\nabla_{\Delta W}\mathcal{L} = J_{E,\Delta W}^\top \nabla_E \mathcal{L}$ chase random connector initialization errors, drifting $\Delta W$ out of the general-text trust region and getting trapped in broken Equivariance / text-shortcut local minima!
   - Derive the **Two-Phase Optimal Switching Rule**:
     - **Phase A (Frozen Host, $\Delta W = 0$)**: Train Equivariant Connector + `RefSufficientPool` until convergence;
     - **Condition Check ($\Delta_{\text{host}}$)**: Measure post-Phase-A DPI host gap $\Delta_{\text{host}} = \text{NLL}_{\text{Phase A}}(D_{\text{dev}}) - \text{NLL}_{\text{direct\_physical}}(D_{\text{dev}})$!
     - If $\Delta_{\text{host}} \le \tau_{\text{trigger}}$ (say $\le 0.03$ nats, `Host-Pure` already nearly closes the Bayes ceiling!), **KEEP HOST 100% FROZEN (`0` extra risk, `0` forgetting, `$0` extra compute!)**;
     - ONLY if $\Delta_{\text{host}} > \tau_{\text{trigger}}$ does Phase B unlock **Surgical Modality-Gated Nullspace LoRA** with linear warmup + cosine LR + Frobenius norm cap!
3. **Cross-Domain Weight Merging (`Synthetic + Pluto + CTTC` Adapters via Task Arithmetic / TIES / Model Soup)**:
   - Instead of risking catastrophic interference by mixing heterogeneous SNR regimes into one unstable joint run or retraining from scratch on new SDR hardware:
     - Analyze convex **Weight-Space Soup / Linear Interpolation** ($\Delta W_{\text{mix}} = \sum_k \lambda_k \Delta W_k$), **Task Arithmetic**, and **TIES-Merging** (sign-conflict trimming) across separate specialized adapters (`Synthetic Large-CFO`, `Pluto 850MHz AC-coupled`, `CTTC 433/708/2450MHz`), deriving conditions (orthogonal Fisher / low cosine overlap $\approx 0$ between task vectors!) where merging generalizes to unseen captures like `f708_snr10` WITHOUT retraining!
4. **Value-of-Information (VoI) Modal Free-Credit Decision Envelope (`<= $12.88` Hard Limit)**:
   - Build exact USD cost-benefit curves comparing:
     - Branch 0: Stay 100% Frozen (`$8.44–$9.50`, leaving `$3.38–$4.44` for Stage 5!);
     - Branch 1: Phase A Frozen (`~69%` of budget) $\to$ Conditional Phase B Selective Gated-LoRA (`+69–192` surgical warmup steps, `~$2.10–$3.10` incremental cost, STILL keeping total spend **`<= $11.80 < $12.88`** and `Billed Cost: $0.00`!).
5. **Mandatory CPU PyTorch Verification (`verify_w4.py`)**:
   - Run two clean CPU PyTorch experiments + deterministic systems/cost verifications:
     1. **Gradient-Shock vs Two-Phase Warm-Start Experiment**: Train identical synthetic-host + connector + LoRA setups on CPU comparing (i) Cold-Start Joint `(Conn + LoRA)` vs (ii) Two-Phase `(Phase A Frozen Conn -> Phase B LoRA)`: verify early-step gradient shock norm $\|\nabla_{\Delta W}\mathcal{L}\|$, general-task KL drift, and final `G1/G3` Acc/NLL!
     2. **Task-Arithmetic / Model-Soup Adapter Merging + NF4 Quantization Noise Simulation**: Test linear/TIES merging of 2 regime-specific adapters (e.g. Low-SNR+Jitter vs Large-CFO) on a held-out mixed regime, AND simulate 4-bit weight quantization noise vs BF16 on continuous phase log-loss at SNR `2 dB` and `+20 dB`!
     3. Numerically verify all `4B / 9B / 27B` VRAM equations against Stage 3D's `52,972.5 MB` / `45,409.9 MB` and verify the exact Stage 4A+4B USD Modal budget (`<= $12.00 < $12.88`) -> write `research/stage4_math_and_design/weight_mod/W4/results_w4.json`.

## 3. Deliverables Required (Strictly Inside `research/stage4_math_and_design/weight_mod/W4/`)
1. `research/stage4_math_and_design/weight_mod/W4/THEOREM_W4.md`
2. `research/stage4_math_and_design/weight_mod/W4/verify_w4.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/weight_mod/W4/results_w4.json`
4. Crisp summary + open questions for W1/W2/W3 in your final response.
