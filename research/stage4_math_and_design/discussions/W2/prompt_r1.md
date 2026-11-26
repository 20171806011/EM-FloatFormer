# Track W2 (Round 1): Parameterization & Optimization Geometry — Equivariant/Cayley/Phase-Dynamic LoRA vs DoRA/PiSSA/SVFT & LEMMO Parity

You are **Track W2**, one of 4 parallel **original Antigravity CLI (`agy-bin.exe` on `gemini-4-neon-xhigh`, extra-high reasoning)** conversations rigorously investigating:
> **How should we parameterize and optimize direct weight modifications (`LoRA, rsLoRA, DoRA, PiSSA/OLoRA, SVFT, Cayley-Givens Complex LoRA, Phase-Conditioned Dynamic Hyper-LoRA`) from manifold geometry, symmetry theory, and optimization dynamics to maximize EM reasoning & OOD extrapolation?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/weight_mod/W2/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete ANY file outside `research/stage4_math_and_design/weight_mod/W2/`.
- **Python Environment Guardrail**: **ALWAYS** run Python via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```

## 1. Grounded Empirical & Theoretical Context (Inspect Read-Only First)
1. `docs/LONG_TERM_RESEARCH_PLAN.md` and `docs/MULTIMODAL_INTEGRATION_PLAN.md` (benchmarking against LEMMO `arXiv:2611.09476` using `Qwen3.5-4B` + LoRA, scaling 4B/9B/27B)
2. `research/stage3c/G/quotient.py` & `research/stage3d_qwen_signed_phase/recompute_all.py`
3. Round-1/2 findings in `research/stage4_math_and_design/M1/THEOREM_M1.md` and `M5/THEOREM_M5.md`:
   - M1 Work-F Gram spectrum on real `cal_ref`: physical contrast has **rank 3** (effective rank `2.00149`, top-1 energy `50.19%`), while Stage 3D unconstrained Qwen lost equivariance (`rot01/11` max diff up to `0.4817`).

## 2. Deep Multi-Angle Mathematical & Optimization Questions for Track W2
1. **Why Unconstrained Standard LoRA / DoRA Fails Symmetries & Out-of-Distribution Phase Extrapolation**:
   - Analyze standard low-rank update $\Delta W = \frac{\alpha}{r} B A$ (`LoRA`), rank-stabilized `rsLoRA` ($\frac{\alpha}{\sqrt{r}}$), weight-decomposed `DoRA` ($m \frac{W_0 + BA}{\|W_0 + BA\|_c}$), SVD-initialized `PiSSA` / `OLoRA`, and singular-vector fine-tuning `SVFT`:
     - Prove exact symmetry-breaking excess degrees of freedom when $\Delta W$ operates on non-canonicalized embeddings (`O(r d)` non-equivariant variance on $C_4$ orbits!).
     - Prove why ANY static pretrained-or-adapted matrix perturbation $W_0 + \Delta W_{\text{static}}$ cannot natively preserve continuous group rotation $\theta = \omega(h+\ell) \in (\pi/2, \pi]$ when trained only on $|\omega| \le 0.25, |\theta| \le 1.25$, unless either (i) quotient-canonicalized, OR (ii) parameterized via exact harmonic / unitary phase modulation!
     - Analyze **LoRA+ asymmetric learning rates** ($\eta_B \gg \eta_A$) and **spectral initialization (`PiSSA` / `OLoRA` vs zero-$B$ Kaiming-$A$)** for fast convergence under tight `$9.50` GPU step budgets (`<= 384` steps!).
2. **Exact Symmetry-Preserving & Phase-Conditioned Weight Parameterizations**:
   Construct and rigorously analyze three superior weight-modification structures:
   1. **Complex-Paired Cayley / Givens Orthogonal LoRA (`Complex-Cayley LoRA`)**: Pair channels into complex doublets $(x_{2k}, x_{2k+1}) \in \mathbb{C}$ and parameterize weight perturbations on skew-symmetric generators $S = U V^\top - V U^\top$ via Cayley map $Q = (I - \tfrac{1}{2} S)(I + \tfrac{1}{2} S)^{-1}$ or low-rank commuting $2\times 2$ blocks $\begin{pmatrix} u & -v \\ v & u \end{pmatrix}$: prove exact Stiefel/orthogonal conditioning ($\|Q\|_2 = 1$, zero gradient explosion/vanishing across depth!), exact $SO(2)$-equivariance, and 2× parameter savings!
   2. **Phase-Conditioned Dynamic Hyper-LoRA ($\Delta W(\theta, V) = \sum_{n=0}^2 \alpha_n(V)[\cos(n\theta)\Delta W_{c,n} + \sin(n\theta)\Delta W_{s,n}]$)**: Connect $\alpha_n(V) = e^{-n^2 \sigma_\phi^2 (h+\ell)/2}$ directly to M1 Theorem 3's circular moment semigroup! Prove exact $2\pi$-periodicity, exact analytical extrapolation across unseen $|\theta| \in [0, 2\pi)$, AND prove the exact **Minimal Rank Theorem ($r \ge 3$)** linking M1's Work-F rank-3 physical spectrum to rank sufficiency of Dynamic LoRA!
   3. **Quotient-Wrapped Weight Modification**: Apply LoRA/DoRA strictly inside M1's $C_4$ canonical representative slice (`quotient.py`) + exact post-backbone group lift -> prove **exact `0.000e+00` $C_4$-equivariance defect** for ANY nonlinear internal Transformer/DeltaNet adapter!
3. **Exact Scientific Parity Protocol Against LEMMO (`arXiv:2611.09476`, Qwen3.5-4B + LoRA)**:
   - Formalize matching trainable parameter count, optimizer budget (matched FLOPs/tokens), matched cache bit budget (`4b s17` vs `16b canonical`), and fairness guarantees comparing EM-FloatFormer (`Frozen-Connector` vs `Dynamic/Cayley-LoRA`) against LEMMO's LoRA baseline!
4. **Mandatory CPU PyTorch Verification (`verify_w2.py`)**:
   - Build a controlled CPU multi-layer surrogate experiment (`train: |\omega| <= 0.25 -> test: G1..G4` + 4-way $C_4$ orbits + limited-step budget mimicking short Stage 4 runs!) comparing at matched trainable parameter budgets:
     1. Standard Zero-Init LoRA (`B=0`);
     2. Spectral-Init `PiSSA` / `DoRA` (`static weights`);
     3. Quotient-Wrapped Cayley/Givens LoRA;
     4. Dynamic Phase-Conditioned Hyper-LoRA (`cos/sin harmonics + Brownian damping`).
   - Measure speed of convergence (`step 69` Stage-1 pilot checkpoint & final step!), `G1..G4` Acc/NLL, `C4` equivariance defect, parameter efficiency, and condition number/Work-F rank spectrum -> write `research/stage4_math_and_design/weight_mod/W2/results_w2.json`.

## 3. Deliverables Required (Strictly Inside `research/stage4_math_and_design/weight_mod/W2/`)
1. `research/stage4_math_and_design/weight_mod/W2/THEOREM_W2.md`
2. `research/stage4_math_and_design/weight_mod/W2/verify_w2.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/weight_mod/W2/results_w2.json`
4. Crisp summary + open questions for W1/W3/W4 in your final response.
