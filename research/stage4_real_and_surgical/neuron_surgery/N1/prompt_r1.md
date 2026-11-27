# Track N1 (Round 1): Exact Algebraic Compilation into SwiGLU Connecting Neurons

You are **Track N1**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **From mathematics first, can directly modifying "connecting neurons" inside `Qwen3.5-4B` / `9B` / `27B` (especially SwiGLU MLP bilinear neurons & sparse attention/DeltaNet gate neurons) break the `31.36%` OOD `G1` ceiling while preserving `PureKL = 0.0`?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N1/` **ONLY**.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `N1/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Run Python **ONLY** via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
- **Exact Deliverables in `N1/`**:
  1. `research/stage4_real_and_surgical/neuron_surgery/N1/THEOREM_N1.md`
  2. `research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py` (must exit `0`)

## 1. Grounded Empirical Context (Read-Only Inspect First)
Inspect read-only as needed:
- `research/stage4_real_and_surgical/surgical_adapter.py` & `equi_connector.py`
- `research/stage4_real_and_surgical/results/phaseB_surgical_vs_lemmo.json`
- `research/stage4_real_and_surgical/results/stage4_synthetic_blind_results.json`
- `research/stage4_math_and_design/SYNTHESIS.md`

Verified Phase II-C telemetry on `Qwen3.5-4B` (`32` layers = `24` GatedDeltaNet + `8` Full Attention, `d=2560`, SwiGLU `d_ff=9216`, interleaved gated `q_proj`, `TIER_P1_MAX_PARAMS = 2,184,964`):
- `equivariant_host_pure` (frozen 4B): Dev Info `61.61% / 0.9296`, Dev `Q3_cfo 46.88% / 1.0379`, Blind `G1 (50,304 rows) 24.48% / 1.3259`, `G3 26.07%`.
- `mode1_surgical_96` (W1–W4 Cayley+LoRA on Attention/DN projections ONLY, `1.24M` adapter params, `0` MLP edits): Dev Info `63.04% / 0.9129`, Dev `Q3_cfo 57.81% / 0.9846` (**+10.94 pp**), Blind `G1 31.36% / 1.3197` (+6.89 pp over frozen, still **-5.68 pp** behind LoRA), `G3 33.01%` (beats LoRA), `PureKL = 0.0`, `MixKL = 0.00419`.
- `lemmo_tier_p1` (unconstrained LoRA on Attention/DN projections, `1.15M` params): Dev Info `61.61% / 0.9281`, Dev `Q3_cfo 46.88%`, Blind `G1 37.04% / 1.2694`, `G3 30.47%`, `pure_max_logit_diff = 0.50` (`PureKL != 0`).
- `direct_physical` / `equivariant_physical_residual`: Blind `G1 64.62% / 0.9411` (exact Bayes ceiling).

**Central Mathematical Discovery Hypothesis**:
Prior W1–W4 adapters NEVER edited SwiGLU MLP neurons! Yet $\operatorname{SiLU}(u) - \operatorname{SiLU}(-u) \equiv u$ holds **identically** for all $u \in \mathbb{R}$, so **2 antipodal SwiGLU neurons** compute exact bilinear multiplication:
$$(a^\top x)(b^\top x) \equiv \operatorname{SiLU}(a^\top x)(b^\top x) + \operatorname{SiLU}(-a^\top x)(-b^\top x).$$

## 2. Deep Mathematical Tasks for Track N1
1. **Exact SwiGLU Neuron Count & Closed-Form Compilation**:
   - Suppose pre-MLP residual vector $x \in \mathbb{R}^d$ encodes orthonormal public slots carrying $T_1 = (u_1, v_1)$, $T_2 = (u_2, v_2)$ and phase-slot carrier $(\cos\omega h, \sin\omega h)$ plus precision-weighted `C4` log-likelihood coordinates (from W2 `B_align` / orthogonal subspace).
   - Derive the **minimum exact number of SwiGLU neurons** (comparing **6** Gauss-Karatsuba neurons vs **8** direct antipodal neurons vs **2N** general bilinear pairs) and write out exact closed-form $(w_{\text{gate}}, w_{\text{up}}, w_{\text{down}})$ to compute:
     - exact complex rotation $T_2' = T_2 e^{-i\omega h}$,
     - exact physical cross-correlation $\operatorname{Re}(T_1^* T_2 e^{-i\omega h})$ & imaginary part $\operatorname{Im}(T_1^* T_2 e^{-i\omega h})$,
     - exact $C_4$ group convolution / cyclic bilinear scoring for `Q1`, `Q2`, and `Q3_repeat_x1`.
2. **Pre-MLP `RMSNorm` Exact Algebra (Crucial Non-Trivial Obstruction!)**:
   - Real transformer blocks apply $z = \operatorname{RMSNorm}(x) = \frac{x}{s(x)} \odot \gamma$ with $s(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}$ **before** `gate_proj` & `up_proj`!
   - When computing a bilinear product $(a^\top z)(b^\top z) = \frac{(a^\top x)(b^\top x)}{s(x)^2}$, every degree-2 product scales by $s(x)^{-2}$, whereas residual skip additions $x + \text{MLP}(\operatorname{RMSNorm}(x))$ mix degree-1 (`O(1)`) and degree-2 (`O(s^{-1})` or `O(s^{-2})`) terms!
   - Prove rigorously when:
     - (a) Pure homogeneous scaling invariance cancels $s(x)$ identically on logits/ratios (and why raw candidate logit shifts break if $x$ is small/large across prose length $L$),
     - (b) An **exact RMSNorm-invariant / self-normalizing SwiGLU gadget** using residual norm anchors or logarithmic/quotient compensation eliminates $s(x)$ sensitivity, or derives the exact error bound when $\gamma \neq 1$.
3. **Multi-Layer Exact Compilation & Zero-Error `G1` Extrapolation Theorem**:
   - Prove a clean **Zero Extrapolation Error Theorem**: with fixed weights $(W_{\text{gate}}, W_{\text{up}}, W_{\text{down}})$ on $K \le 64$ SwiGLU neurons across $1\text{--}2$ consecutive layers + `lm_head` candidate readout directions, does the exact circuit match the continuous `Q1/Q2/Q3` log-posterior to **$0$ functional error** across ALL unseen angles $\theta \in [-\pi, \pi]$ and frequencies $\omega$ (`G1` double extrapolation)?
   - Derive exact parameter count vs `d_ff = 9216 * 32 = 294,912` neurons.
4. **Machine-Verified Numerical Proof (`verify_n1.py`)**:
   - Build a genuine PyTorch RMSNorm + SwiGLU (+ second layer if needed) module mimicking `Qwen3.5-4B` dimensions (`d=2560`, random non-uniform RMSNorm weights $\gamma$, nontrivial prose background vector $h_{\text{prose}} \perp B_{\text{EM}}$ in `B_align` orthogonal complement!), compile your closed-form neuron weights ONLY, and test across adversarial `G1` shifts ($|\omega h|, |\theta|$ well outside any train disk) + prose norm perturbations in **FP64, FP32, and BF16**.

Think deeply and verify every theorem numerically. Keep all outputs strictly inside `N1/`.
