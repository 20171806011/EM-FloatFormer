# Track N3 (Round 1): Zero-Interference Dormant/Nullspace Neuron Surgery & Pareto Optimality (`PureKL`, `MixKL`, Polysemanticity)

You are **Track N3**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **From mathematics first, can directly modifying dormant or nullspace-projected "connecting neurons" in an LLM give BETTER `G1` OOD accuracy AND BETTER `PureKL`/`MixKL` Pareto trade-offs than `mode1_surgical_96` and `lemmo_tier_p1` — or do superposition & false dormancy defeat naive dormant-neuron editing?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N3/` **ONLY**.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `N3/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Run Python **ONLY** via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
- **Exact Deliverables in `N3/`**:
  1. `research/stage4_real_and_surgical/neuron_surgery/N3/THEOREM_N3.md`
  2. `research/stage4_real_and_surgical/neuron_surgery/N3/verify_n3.py` (must exit `0`)

## 1. Grounded Empirical Context (Read-Only Inspect First)
Inspect read-only as needed:
- `research/stage4_real_and_surgical/surgical_adapter.py` & `phaseB_surgical.py`
- `research/stage4_real_and_surgical/results/phaseB_surgical_vs_lemmo.json`
- `research/stage4_real_and_surgical/results/stage4_synthetic_blind_results.json`
- `research/stage4_math_and_design/SYNTHESIS.md`

Verified Phase II-C telemetry on `Qwen3.5-4B` (`32` layers, `d=2560`, SwiGLU `d_ff=9216` -> `294,912` MLP neurons total):
- `equivariant_host_pure` (frozen 4B): Dev Info `61.61% / 0.9296`, Blind `G1 (50,304 rows) 24.48% / 1.3259`, `G3 26.07%`.
- `mode1_surgical_96` (W1–W4 Cayley+LoRA on Attention/DN projections ONLY, `k_c=256` nullspace, strict `U(m)` Cayley norm-lock, `2.5 * MixKL` every 4 steps): Dev Info `63.04%`, Dev `Q3_cfo 57.81%` (**+10.94 pp**), Blind `G1 31.36% / 1.3197` (**-5.68 pp behind LoRA**), `G3 33.01%` (beats LoRA), `PureKL = 0.0`, **`MixKL = 0.00419`** (`passes_mix_kl_1e3 = false`).
- `lemmo_tier_p1` (unconstrained LoRA): Blind `G1 37.04% / 1.2694`, `G3 30.47%`, **`pure_max_logit_diff = 0.50`** (`PureKL != 0`).

**The Core Diagnostic Puzzle**:
Why did `mode1_surgical_96` lose to unconstrained `lemmo_tier_p1` on `G1` (`31.36% < 37.04%`) AND still miss `MixKL <= 1e-3` (`0.00419`), whereas editing dormant/nullspace SwiGLU connecting neurons might break this Pareto frontier?

## 2. Deep Mathematical Tasks for Track N3
1. **Why `mode1_surgical_96` Underperformed `lemmo` on `G1` & Violated `MixKL <= 1e-3` (Exact Geometric Diagnosis)**:
   - Decompose the tangent-space / expressive constraint imposed jointly by:
     (i) `k_c = 256` real/complex nullspace excision $\Pi_\perp^{\text{gen}}$ (`2*k_c = 512` real dims removed out of `2560`),
     (ii) strict unitary `U(m)` Cayley (`||Q x|| == ||x||` forbids radial gain/rescaling across frequencies $\omega$!),
     (iii) unconditional `2.5 * MixKL` penalty + `ModeI` stop-grad at Layer 10.
   - Compare Jacobian rank / conditioning vs **SwiGLU connecting-neuron modification**: show mathematically why MLP `gate_proj` + `up_proj` + `down_proj` neuron edits supply radial scaling AND bilinear phase products in an exponentially lower-interference geometry.
2. **Exact `PureKL` & `MixKL` Bounds for Dormant / Low-Fisher Neuron Surgery**:
   - Suppose $K$ neurons in layer $\ell$ satisfy small text Fisher / activation expectation on general prose, OR have input weight vectors projected into $\Pi_\perp^{\text{gen}} = I - U_{\text{gen}} U_{\text{gen}}^\top$ (and `down_proj` columns projected orthogonal to text readout directions).
   - **Notice a CRITICAL SwiGLU nonlinearity subtlety!** Even if $a^\top U_{\text{gen}} = 0$ on top-$k_c$ PCA/K-FAC span, a real prose vector $h_{\text{prose}}$ has residual component $h_\perp \neq 0$ in the orthogonal complement! Moreover, because $\operatorname{SiLU}(0) = 0$ is smooth (not ReLU $=0$ for $u < 0$!), for small positive/negative tail perturbations $u = a^\top h_\perp \approx \pm 0.1$, **SwiGLU leaks $O(\|h_\perp\|^2)$!**
   - Contrast this carefully with antipodal cancellation $\operatorname{SiLU}(u)v + \operatorname{SiLU}(-u)(-v) \equiv u v$, AND analyze **dead-zone thresholded / ReLU-like gating or ROME/MEMIT key-value steering**!
3. **Formalizing the "False-Dormancy & Superposition Trap"**:
   - A neuron with near-zero activation across `n=12` or `n=256` calibration prompts may actually encode a rare tail concept (superposition, Elhage et al. scaling).
   - Derive a rigorous probabilistic tail bound (concentration / heavy-tailed activation rate $\alpha_j$, number of unseen test tokens $N_{\text{test}}$, Bonferroni collision probability) showing when **"naive dormant neuron overwrite"** causes catastrophic rare-prose corruption (`pure_max_logit_diff` spikes!) vs when **nullspace projection + threshold barrier / EM-slot-token keying** gives provable tail safety (`PureKL <= \epsilon`, `MixKL <= 1e-3`)!
4. **Pareto-Optimal Neuron Surgery Criterion (`G1` vs `PureKL/MixKL`) & Verification (`verify_n3.py`)**:
   - Formulate a constrained optimization objective (e.g. Lagrangian hinge `(MixKL - tau_mix)_+^2` + rank-$K$ SwiGLU+nullspace LoRA relaxation with bounded singular spectrum $[\sigma_{\min}, \sigma_{\max}]$ instead of rigid $\sigma \equiv 1$!) that strictly dominates `mode1_surgical_96` on BOTH `G1` and `MixKL`.
   - Verify every bound numerically in `verify_n3.py` (simulating realistic correlated heavy-tailed prose activations $h_{\text{prose}}$ with top-$k_c$ principal subspace + long-tail orthogonal residuals, comparing naive dormant replacement, single-neuron SiLU leak, antipodal nullspace SwiGLU, and bounded-spectrum relaxed Cayley+SwiGLU).

Think deeply and verify every theorem numerically. Keep all outputs strictly inside `N3/`.
