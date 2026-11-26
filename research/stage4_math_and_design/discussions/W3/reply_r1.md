Running the CPU PyTorch verification suite (`verify_w3.py`) via `uv run` in the background; I will inspect `results_w3.json` and author `THEOREM_W3.md` once it completes.
Waiting for the short CPU thread-scaling benchmark (`task-46`) to return before optimizing `verify_w3.py`.
Waiting for `task-51` thread benchmark results.
Waiting for the parallel WY Gated-DeltaNet parity and speed benchmark (`task-56`) to complete.
Optimized `verify_w3.py` using the exact parallel WY unit-lower-triangular Gated-DeltaNet formulation (`max_abs_diff = 2.38e-6` vs sequential recurrence, >10x CPU speedup) and relaunched via `uv run` (`task-67`). Waiting for completion to inspect `results_w3.json` and author `THEOREM_W3.md`.
`task-67` is running smoothly and has already verified Blocks 1–2 (`self_concordance_brackets = True`, Kronecker spectral identity error `1.59e-12`, `147.98x` nullspace KL reduction) and the first two Axis-1 methods (`layer_selected_gen_fisher_lora` reduced pure-text KL drift by `6.4x` vs unconstrained LoRA while improving `G1–G4` accuracy and `Q2` Blackwell gain). Waiting for the remaining Axis-1 and 3-seed Axis-2 runs to complete.
Track **W3 (Round 1)** is complete. All outputs were written strictly inside `research/stage4_math_and_design/weight_mod/W3/` and verified end-to-end via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B` (zero edits outside `W3/`).

### Deliverables
- [THEOREM_W3.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W3/THEOREM_W3.md) — Complete formal proofs (Theorems 1–7), `4B → 9B → 27B` layer budgets, and calibrated loss/curriculum specification.
- [verify_w3.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W3/verify_w3.py) — Deterministic 8-layer 3:1 Hybrid Gated-DeltaNet (exact parallel WY formulation, `max_abs_diff = 2.38e-6` vs sequential recurrence) + Full-Attention CPU host (`tied lm_head`) dual-task verification suite.
- [results_w3.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W3/results_w3.json) — Full per-block, per-method, and 3-seed (`202611, 202622, 202633`) verified metrics (`ALL W3 INTEGRITY ASSERTIONS PASSED OK`).

---

### Crisp Summary of Verified Results

1. **Exact Fisher–Rao KL Bound, Self-Concordance & Proxy Geometry (Theorems 1–2)**
   - **Self-Concordance Trust Region**: Two-sided Bach generalized self-concordance KL bracket holds `5/5` across $\|\Delta W\|_F \in [0.01, 0.60]$ (`0.0595%` quadratic error at `KL = 1.409e-4`, `0.1803%` at `KL = 1.267e-3`); coupled condition $\tilde{\gamma} \le 0.25$ and $\Delta\theta^\top F_{\mathrm{gen}}\Delta\theta \le 1.65\times 10^{-3}$ certifies general-domain regression $\le 10^{-3}$ nats ($<0.10\%$ perplexity shift).
   - **Proxy Distortion (`layer_3.o_proj`, matched $\|\Delta W\|_F = 0.35$)**: K-FAC matches exact Task-A KL within `1.3%–2.9%`; Diagonal EWC underestimates aligned correlated-text KL by `5.29×` (analytically up to $d\times$: `64.0×` at $d=64$, `2560×–5120×` on Qwen!) while falsely over-penalizing safe nullspace directions by `16.8×`.
   - **Input-Only Nullspace Projection ($\Delta W = B \tilde{A} P_{\perp, C}$)**: Cuts single-layer Task-A KL **`147.98×`** vs top text eigendirection (`4.014e-5 → 2.712e-7`) while leaving $B$ unrestricted so late layers retain full candidate readout rank $M_{\mathrm{cand}}$ (Asymmetric Readout Lemma).

2. **Modality-Gated Zero-Interference & Mixed Multi-Turn Fluency (Theorems 3–4, Axis 1)**
   - Frozen host baseline fails EM routing (`G1 = 25.88% / 1.4476`, `Q2` fresh−valid `-2.93 pp`), while the exact Bayes Oracle hits `G1 = 64.55% / 0.9281` and `Q2` `+6.64 pp`.
   - **Unconstrained LoRA (`60,800` params)**: `G1 = 63.57% / 0.9552`, but destroys Task A (`PureKL = 0.6611`, `MixKL = 0.8132`).
   - **Layer-Selected Gen-Fisher LoRA (`30,400` params, 50% cut)**: Cuts `PureKL` **`6.4×`** (`0.1028`) and `MixKL` **`4.6×`** (`0.1749`) while *improving* `G1` (`63.77%`) and `Q2` gain (`+6.25 pp`).
   - **Nullspace-Projected LoRA (`30,400` params)**: Cuts `PureKL` **`180×`** (`3.681e-3`), but still leaks `MixKL = 0.1249` on post-EM prose tokens.
   - **Modality-Gated Unconstrained (Theorem 4 Ablation)**: Achieves **bit-exact `PureKL = 0.000e+00`** (`max_logit_diff = 0.00`), **but collapses mixed-turn fluency (`MixKL = 0.7518` nats, `+0.709` NLL drift!)**.
   - **Modality-Gated + Nullspace Hybrid (Recommended W3)**: **Bit-exact `PureKL = 0.000e+00`** (`max_logit_diff = 0.00`, `host_sha256` identical) **AND** slashes `MixKL` **`80.2×`** (`0.01014` nats; mixed NLL drift **`+0.00148` nats**, **`508×` lower**), achieving **`G1 = 63.18% / 0.9623`** and the **highest `Q2` Blackwell fresh-pilot gain (`+7.62 pp / +0.1324 nats`)**.

3. **Generalized-Eigenvalue Layer Selection (`Qwen3.5-4B → 9B → 3.8-27B`, Theorem 5)**
   - Exact Kronecker spectral factorization $\Lambda_{\max}^{(\ell)} = \Lambda_C^{(\ell)} \Lambda_G^{(\ell)}$ verified to **`1.59e-12`** error.
   - Keeping tied `lm_head`, `embed_tokens`, FFNs, and early lexical layers strictly frozen and targeting mid-to-late Full-Attention + paired DeltaNet layers ($r=8$) yields **`1.64M` (`4B`, `3.81×` cut) / `2.38M` (`9B`, `4.03×` cut) / `4.69M` (`27B`, `5.15×` cut)** trainable parameters, requiring only `25–72 MiB` (LoRA+AdamW) and `39–245 MiB` (low-rank BF16 $U_k$ projectors).

4. **Objective Engineering & Step-69 Pilot Signal (Theorems 6–7, Axis 2, 3 Seeds)**
   - **Rao–Blackwell Gradient Variance Reduction (`CE / KL`)**: **`3.58×`** at Step 16 and **`113.73×`** at convergence (`21.02` vs `0.185`, confirming `99.1%` of converged 1-hot CE gradient energy is pure irreducible Bayes label noise!).
   - **M6 Step-69 Pilot**: Hard-label CE stalls at `G1 = 44.30% ± 4.69 pp` (`NLL = 1.2199`, `Q2 = -0.85 pp`), whereas **W3 Composite Loss (`Bayes-KL + Logit-Norm + CF-Margin + Prose-Inv + C4-Orbit`)** hits **`61.26% ± 0.75 pp`** (**`+16.96 pp`**, **`6.3×` lower seed SD**), **`NLL = 0.9942`**, **`ECE = 0.0431 < 0.05`**, **`T* = 0.971`**, and **`Q2 = +4.88 pp / +0.0806 nats`**.
   - **Converged Step 240**: Best `G1 NLL = 0.9633` (`-0.0429` vs CE), best `ECE = 0.0582` (`T* = 0.898`), Work-F orientation gain **`0.8612`** (slot-flip penalty `+1.866` nats), `37%` lower prose-perturbation TV (`0.0193`), `47%` lower native $C_4$ TV (`0.0854`), and `Q2` gain **`+6.25 pp / +0.1274 nats`**.

---

### Open Questions for Cross-Track Convergence (W1 / W2 / W4)

- **For Track W1 (Gated-DeltaNet Circuits & Routing)**:
  1. Our Layer-0 Rayleigh spike (`mean_eta_em = 0.2584` vs `0.0147` at Layer 7) shows EM evidence starts strongly orthogonal to text subspaces but mixes rapidly across nonlinear layers. If we keep early layers `1–10` frozen on `Qwen3.5-4B/9B` (to cut forgetting & backward VRAM), does pretrained DeltaNet decay $\alpha_t < 1$ in layers `1–10` wipe out `<W1><W2>` before layer `11` over long Stage 5–6 contexts (`L_gap >= 64`), or does a **Tier-1 Token-Position-Gated** edit restricted strictly to early DeltaNet $(\Delta W_\alpha, \Delta W_\beta)$ suffice without unfreezing early $Q/K/V/O$?
  2. Does our proposed **Tier-1 Token-Position K/V Gate** ($m_{\mathrm{EM}}(t)=1$ on EM positions only for $K, V, \beta$; sequence gate $g_{\mathrm{EM}}=1$ + nullspace $P_{\perp,C}$ for $Q, O, \alpha$) preserve full Kalman-RLS error correction in DeltaNet across intervening Chinese prose tokens?
- **For Track W2 (Cayley / Phase-Dynamic Parameterizations & LEMMO Parity)**:
  1. When combining W2's **Complex-Cayley Orthogonal LoRA** or **Phase-Conditioned Dynamic Hyper-LoRA** $\Delta W(\theta, V)$ with W3's input nullspace projector $P_{\perp, C} = I - U_k U_k^\top$, does restricting the skew-symmetric generator $S = P_{\perp, C}(U V^\top - V U^\top)P_{\perp, C}$ onto the real nullspace preserve exact $SO(2)$ channel-pairing structure (requiring $U_k$ to be computed over complex doublet pairs so $P_{\perp, C}$ commutes with the $2\times 2$ complex block structure $J$)?
  2. When W2 applies **Quotient-Wrapped Weight Modification** (exact `0.000` outer $C_4$ defect), should $\mathcal{L}_{\mathrm{orbit}}$ in W3's composite objective be set to `0` on single-turn Q1–Q3 to save 25% step compute, and re-enabled only on uncanonicalized Stage 5–6 multi-turn conversations?
- **For Track W4 (VRAM Equations, Anti-Shock Curriculum & `$12.88` Modal Budget)**:
  1. Given our exact selected-layer budgets (`Qwen3.5-4B`: `1.64M` params, `25.06 MiB` LoRA+AdamW + `39.38 MiB` BF16 $U_k$ projectors, early layers `1–10` frozen so autograd stops at layer `11`), how much backward activation-checkpointing VRAM and step time (`s/step`) do we save relative to Stage 3D's `52,972.5 MiB` / `1.976 s/step` — and does adding our batched counterfactual sub-batch (`sub = 16`) + mixed-prose KL anchor stay strictly under M6's `65,000 MiB` peak VRAM ceiling and `$12.88` credit cap?
  2. Since W3's `Bayes-KL + CF-Margin` objective achieves `61.26% ± 0.75 pp` (`NLL = 0.9942`, `ECE = 0.0431`) already at **Step 69** (vs `44.30%` for 1-hot CE), does W4's Value-of-Information switching rule allow shortening conditional Phase B from `192` steps down to **`69–128` surgical steps**, keeping total Stage 4A+4B Modal spend well below `$11.00`?
