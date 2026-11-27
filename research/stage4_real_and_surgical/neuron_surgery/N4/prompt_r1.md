# Track N4 (Round 1): Closed-Form Compiled Neurons + Sparse Fine-Tuning, BF16/NF4 Stability (4B/9B/27B), & Falsifiable E2E Proof

You are **Track N4**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **From mathematics first, if we combine closed-form algebraic "connecting neuron" compilation with sparse gradient fine-tuning inside `Qwen3.5-4B` / `9B` / `27B` (under BF16 and 27B Mixed-NF4 quantization!), can it demonstrably smash the `31.36%` `G1` ceiling while keeping parameter count `< 2.18M` and cost `< $0.50`?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N4/` **ONLY**.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `N4/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Run Python **ONLY** via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
- **Exact Deliverables in `N4/`**:
  1. `research/stage4_real_and_surgical/neuron_surgery/N4/THEOREM_N4.md`
  2. `research/stage4_real_and_surgical/neuron_surgery/N4/verify_n4.py` (must exit `0`)

## 1. Grounded Empirical & Budget Context (Read-Only Inspect First)
Inspect read-only as needed:
- `research/stage4_real_and_surgical/surgical_adapter.py` & `phaseB_surgical.py` & `equi_connector.py`
- `research/stage4_real_and_surgical/results/phaseB_surgical_vs_lemmo.json`
- `research/stage4_real_and_surgical/results/stage4_synthetic_blind_results.json`
- `research/stage4_real_and_surgical/modal_stage4.py`

Verified Phase II-C context:
- Models: `Qwen3.5-4B` (`BF16`, `32` layers, `d=2560`, `d_ff=9216`), `Qwen3.5-9B` (`BF16`), and `Qwen3.8-27B` (**`Mixed-NF4`** quantized backbone!). Free Modal budget constraint: only `~$7.29` remaining total, so any follow-up 4B neuron-surgery probe must run in `< $0.50` (`< ~12` minutes on A100-80GB).
- `mode1_surgical_96` (W1–W4 Cayley+LoRA, `1.24M` params, `0` MLP neurons): Dev `Q3_cfo 57.81%` (**+10.94 pp**), Blind `G1 (50,304 rows) 31.36% / 1.3197`, `G3 33.01%`, `PureKL = 0.0`, `MixKL = 0.00419`.
- `lemmo_tier_p1` (`1.15M` unconstrained LoRA): Blind `G1 37.04% / 1.2694`, `pure_max_logit_diff = 0.50`.
- `direct_physical`: Blind `G1 64.62% / 0.9411` (exact Bayes ceiling).
- Core algebraic identity for SwiGLU MLP connecting neurons:
  $$(a^\top x)(b^\top x) \equiv \operatorname{SiLU}(a^\top x)(b^\top x) + \operatorname{SiLU}(-a^\top x)(-b^\top x).$$

## 2. Deep Mathematical Tasks for Track N4
1. **Quantization & Numerical Cancellation Stability Under BF16 and Mixed-NF4 (`4B/9B/27B`)**:
   - Look closely at antipodal SwiGLU cancellation: $\operatorname{SiLU}(u) v + \operatorname{SiLU}(-u)(-v) = u v$.
   - Notice two failure modes under low precision:
     - When prose residual noise dominates ($|h_{\text{prose}}| \sim O(\sqrt{d}) \gg |u| \sim O(1)$) BEFORE RMSNorm or nullspace projection, naive SwiGLU sums large opposite-signed terms and suffers **catastrophic cancellation** in BF16 (7-bit mantissa, ULP $\approx 2^{-7}$) or NF4 (4-bit nonlinear codebook)!
     - Derive exact worst-case & RMS numerical error bounds for antipodal SwiGLU under **FP32**, **BF16**, and **NF4-frozen host + FP32/BF16 sparse neuron overlay**! Prove optimal feature scaling $\alpha$ and whether NF4 base weights make permanent dormant-neuron overwrite unusable on `27B Mixed-NF4` vs sparse additive FP32 neuron-pair delta overlays ($\Delta W_{\text{gate}}, \Delta W_{\text{up}}, \Delta W_{\text{down}}$ of rank/width $K \le 32$)!
2. **Hybrid "Closed-Form Compiled Connecting Neurons + Relaxed Sparse Fine-Tuning" Architecture**:
   - Design an exact concrete PyTorch recipe uniting:
     - Closed-form compiled SwiGLU bilinear complex-rotation neurons (zero-shot exact `G1` trigonometry!) + sparse DeltaNet/Attention bridge neurons,
     - Plus sparse fine-tuned residual connecting neurons (replacing W1–W4's rigid $U(m)$ norm lock and `k_c=256` squeeze with bounded-spectrum scaled Cayley $e^{\sigma} Q$ and adaptive Lagrange-hinge `MixKL` control).
   - Compute exact trainable parameter count on `Qwen3.5-4B` (`<< TIER_P1_MAX_PARAMS = 2,184,964`), VRAM footprint (`<< 75 GB`), and estimated wall-clock / Modal cost.
3. **Concrete End-to-End CPU PyTorch Proof (`verify_n4.py`)**:
   - Build a genuine multi-layer hybrid mock Transformer (`RMSNorm + Attention + GatedDeltaNet + SwiGLU` across >=3 layers with pretrained-style random text weights + general prose background noise, `torch.bfloat16` & mock `NF4` simulation!).
   - Generate a genuine `G1`-style double-extrapolation dataset (`train` on narrow $(\omega, \theta)$ band, `test_G1` far out-of-distribution $(\omega, \theta)$ band!) + general text prompts (`PureKL` / `MixKL`).
   - Empirically compare on CPU:
     1. `Frozen Host + Soft Prefix`
     2. `Unconstrained LoRA (LEMMO-style)`
     3. `Pure Geometric Cayley+LoRA (W1-W4-style)`
     4. `Closed-Form Compiled SwiGLU + Bridge Neurons (Zero-Shot & Sparse Fine-Tuned)`
   - Measure `Train Acc/NLL`, `G1 OOD Extrapolation Acc/NLL`, `PureKL`, `MixKL`, and parameter counts — demonstrate quantitatively whether and WHY connecting-neuron surgery closes the `G1` gap (`~64%` Bayes limit vs `31%–37%`), OR expose honestly where any failure mode remains!
4. **Falsifiable Decision Matrix for Next Modal Probe**:
   - Define exact Pass / Abort thresholds (e.g. `PureKL == 0.0`, `MixKL <= 1e-3`, blind `G1 >= 45%` beating both `mode1_surgical_96 31.36%` and `lemmo_tier_p1 37.04%`, cost `<= $0.50`) and ablation controls (e.g. zero-shot compiled vs compiled+tuned vs relaxed-spectrum).

Think deeply and verify every claim with runnable code in `verify_n4.py`. Keep all outputs strictly inside `N4/`.
