# Track W3 (Round 1): Anti-Forgetting Fisher/Nullspace Geometry, Modality-Gated Zero-Interference Deltas, Loss Engineering & 4B->9B->27B Layer Selection

You are **Track W3**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **If we directly modify model weights (`Qwen3.5-4B / 9B / 3.8-27B`) to boost EM reasoning, how do we MATHEMATICALLY GUARANTEE zero degradation on ordinary text/vision tasks, select the optimal layer subset across 4B->9B->27B, and optimize the training objectives (loss functions, Bayes distillation, equivariant regularizers, and data curriculum)?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/weight_mod/W3/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `research/stage4_math_and_design/weight_mod/W3/`.
- **Python Environment Guardrail**: **ALWAYS** run Python via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```

## 1. Grounded Empirical & Theoretical Context (Inspect Read-Only First)
1. `docs/MULTIMODAL_INTEGRATION_PLAN.md` (Explicit requirement: ordinary text & image benchmarks MUST undergo paired regression testing; frozen host keeps `host_sha256` exact, so any weight change risks catastrophic forgetting unless bounded!)
2. `research/stage3d_qwen_signed_phase/qwen_connector.py` (`Qwen3.5-4B` architecture, `host_sha256` fingerprint, `<W1><W2><PHASE><REF>` markers)
3. `research/stage4_math_and_design/M2/THEOREM_M2.md`, `M5/THEOREM_M5.md`, `M6/THEOREM_M6.md`:
   - M2 proved Stage 3D's `phase` gain was mostly temperature/overconfidence (`+0.0365` out of `+0.0357` nats on `G1`!) while losing on recalibration!
   - M5 proved `4-bit s17` + `16-bit canonical` dual cache spans Stage 4→6 (`16-bit` drops Stage 5 joint NLL from `1.2788 -> 0.9669`).
   - M6 derived `M_cand = Pi C_cand E_vocab D_answer B` and the DPI ladder.

## 2. Deep Multi-Angle Mathematical, Continual-Learning & Objective-Optimization Questions for Track W3
1. **Exact Second-Order General-Domain KL Forgetting Bound & Nullspace Geometry**:
   - Let $P_W(y \mid x_{\text{gen}})$ be the pretrained text/multimodal distribution. For any weight perturbation $\Delta W$:
     - Derive the exact Fisher-Rao KL bound:
       $$D_{\text{KL}}\big(P_W(\cdot \mid x_{\text{gen}}) \,\|\, P_{W+\Delta W}(\cdot \mid x_{\text{gen}})\big) = \frac{1}{2} \text{vec}(\Delta W)^\top F_{\text{gen}} \text{vec}(\Delta W) + R_3(\|\Delta W\|)$$
       with explicit spectral trust-region condition keeping general-domain regression $\le 10^{-3}$ nats!
     - Rigorously compare **L2 penalty**, **diagonal EWC**, **K-FAC Kronecker Fisher ($G \otimes A$)**, and **Activation-Covariance Nullspace Projection** ($\Delta W = B A$ with $A$ projected onto $\text{ker}_\epsilon(C_{\text{gen}})$ or $(I - U_k U_k^\top)$ left/right principal subspaces of general text activations): prove when EM updates lie orthogonal to high-variance text pathways!
2. **Modality-Gated Zero-Interference Weight Deltas ($\Delta W_{\text{EM}} \cdot g_{\text{EM}}(x)$)**:
   - Exploit EM-FloatFormer's explicit `<W1><W2><PHASE><REF>` prefix protocol:
     - Prove **Bit-Exact Zero-Regression Invariant**: If weights update via $\Delta W_{\text{eff}}(x) = g_{\text{EM}}(x)\,\Delta W$ where $g_{\text{EM}}(x) = 0$ identically whenever NO EM evidence tokens exist in context, general text & image predictions match $W_0$ bit-for-bit (`KL drift = 0.000e+00`)!
     - Analyze the **Mixed Multi-Turn Hard Case**: During an EM conversation AFTER `<W1><W2>` have been fed, the model MUST STILL generate fluent, logical Chinese explanations and follow instructions cleanly! Prove why **combining Modality-Gating (`0` off-task) WITH Nullspace Projection / KL Trust-Region (`on-task` protection)** protects multi-turn linguistic fluency AND Stage-6 instruction following simultaneously!
3. **Layerwise Generalized-Eigenvalue Selection Criterion Across `Qwen3.5-4B -> 9B -> 3.8-27B`**:
   - Derive the optimal layer-budget allocation criterion ranking layers $\ell \in \{1..L\}$ (`early DeltaNet`, `mid DeltaNet`, `Full-Attn layers`, `late FFN`, `lm_head`) by the generalized Rayleigh quotient:
     $$\max_{\Delta W_\ell} \frac{\Delta \text{NLL}_{\text{EM}}}{\Delta \text{KL}_{\text{gen}}} \implies \lambda_{\max}\!\left((F_{\text{gen}}^{(\ell)})^{-1/2} F_{\text{EM}}^{(\ell)} (F_{\text{gen}}^{(\ell)})^{-1/2}\right).$$
   - Characterize how EM-relevant vs linguistic Fisher spectra separate across depth $L$ (low layers = token/feature ingestion; middle DeltaNet/Attn layers = relational transport & pilot fusion; top readout layers = answer-token routing!).
4. **Objective, Distillation & Curriculum Optimization (Beyond Naive Hard-Label Cross-Entropy!)**:
   - In Stage 3D, training used pure noisy hard-label CE over 4 classes! From statistical decision theory (Bregman proper scoring & Rao-Blackwellization!):
     - Prove exact **variance reduction & calibration optimality** of **Exact Physical Posterior KL Distillation** ($\min_\theta \mathbb{E}[\text{KL}(p_{\text{Bayes}}(Y \mid Y_{\text{IQ}}) \| q_\theta)]$ / Stage-4 teacher posterior $\text{KL}(p_{\text{s17/phys}} \| q_\theta)$) vs noisy 1-hot CE (`O(1)` irreducible Bayes label variance eliminated from gradients!!)!
     - Combine with **Explicit Temperature-Decoupled Logit Regularization**, **Symmetric Orbit Consistency**, and **Counterfactual Fixed-Text Margin Losses** (penalizing zero sensitivity to `<PHASE>`/`<REF>` flips vs high sensitivity to irrelevant prose!).
5. **Mandatory CPU PyTorch Verification (`verify_w3.py`)**:
   - Build a clean pretrained dual-task Transformer/Recurrent CPU host (`Task A: correlated General Text sequence next-token task`, frozen to a baseline NLL; `Task B: EM CFO + Q2 phase task`).
   - Run a 2-axis controlled study:
     - **Forgetting axis**: Unconstrained LoRA vs Layer-Selected Generalized-Fisher LoRA vs Nullspace-Projected LoRA vs Modality-Gated+Nullspace Hybrid (measuring Task A pure NLL drift, Task A mixed-prompt fluency retention, and Task B `G1..G4` Acc/NLL);
     - **Loss optimization axis**: Noisy 1-hot CE vs Exact Bayes-Posterior KL Distillation + Counterfactual Slot Margin (measuring sample efficiency, seed stability, calibration ECE, and step-69 pilot signal!).
   - Write full verified metrics -> `research/stage4_math_and_design/weight_mod/W3/results_w3.json`.

## 3. Deliverables Required (Strictly Inside `research/stage4_math_and_design/weight_mod/W3/`)
1. `research/stage4_math_and_design/weight_mod/W3/THEOREM_W3.md`
2. `research/stage4_math_and_design/weight_mod/W3/verify_w3.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/weight_mod/W3/results_w3.json`
4. Crisp summary + open questions for W1/W2/W4 in your final response.
