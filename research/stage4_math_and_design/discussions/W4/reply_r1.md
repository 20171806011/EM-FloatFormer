Running [verify_w4.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/verify_w4.py) via `uv run` in the background; waiting for completion before inspecting `results_w4.json` and authoring `THEOREM_W4.md`.
Cancelled the slow `L=64` quantization loop in `verify_w4.py` and checking the per-benchmark Stage 3D VRAM calibration residuals before optimizing and re-running.
Executing the refined [verify_w4.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/verify_w4.py) via `uv run`; waiting for completion to inspect [results_w4.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/results_w4.json).
Executing the finalized [verify_w4.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/verify_w4.py) via `uv run` with single-threaded CPU PyTorch; waiting for completion before reading [results_w4.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/results_w4.json) and writing `THEOREM_W4.md`.
Running updated [verify_w4.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/verify_w4.py) via `uv run`; waiting for completion to inspect the final [results_w4.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/results_w4.json) and author `THEOREM_W4.md`.
All instructions in [prompt_r1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/discussions/W4/prompt_r1.md) have been executed verbatim, with all outputs written strictly inside `research/stage4_math_and_design/weight_mod/W4/` and verified via `uv run`.

### Deliverables
- [THEOREM_W4.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/THEOREM_W4.md) — Full mathematical derivations, calibrated `4B / 9B / 27B` VRAM & step-time equations, NF4 quantization phase-distortion bounds, Cold-Start Gradient-Shock proof, cross-domain adapter merging theorems, and Decimal Modal `$12.88` Bayesian VoI decision boundary.
- [verify_w4.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/verify_w4.py) — Executed cleanly via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B` (exit code `0`, `67.71 s` on CPU).
- [results_w4.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W4/results_w4.json) — Complete structured verification outputs.

---

### Crisp Summary of Verified Results

1. **Exact `4B -> 9B -> 27B` VRAM Calibration & Layer-0 Backprop Invariance (Theorem 1)**:
   - Closed-form hybrid architecture count matches `Qwen/Qwen3.5-4B` to **`0.0000%` error** (`4,205,751,296` exact; `|V| = 248,320`, `24` Gated-DeltaNet + `8` Full Gated-Attention layers), and calibrates **all 10 Stage 3D A100-80GB benchmarks** (`52,972.5 MiB` train `mb=16`, `45,409.9 MiB` eval `b=64`, `batched_ckpt` `mb=4..32`, `Q2_kv`) within **`0.332%` max relative error**.
   - **Full FT (`4B`)** hits **`109,204 MiB` (`106.6 GiB`) at `mb=16` (`86,706 MiB` at `mb=8`) $\implies$ exact OOM** on A100-80GB.
   - **Selective BF16 LoRA (`r = 8`, `50%` layers = `5.06M` params)** adds only **`+92.9 MiB` (`+0.18%`)** above Frozen Host (`53,144 MiB < 65,000 MiB` M6 gate) and `+12%` step time (`2.21 s/step < 2.50 s/step`), because Layer-0 connector training already stores full $L$-layer autograd activations.
   - On **`Qwen3.5-9B`** (`8.954B`), Selective BF16 LoRA (`kv_train, mb=8, r=8`) fits comfortably at **`43,060 MiB` (`42.05 GiB`)**. On **`Qwen3.8-27B`** (`26.896B`), `kv_train` OOMs even frozen (`118.5 GiB` at `mb=8`); **`batched_ckpt` BF16 LoRA (`mb=8`)** fits at **`62,597 MiB` (`61.13 GiB`)**, and **Mixed `MLP-NF4 + DeltaNet/Attn/IO-BF16`** fits `mb=16` at **`50,334 MiB` (`49.15 GiB`)**.

2. **NF4 Quantization Phase Distortion Bound (Theorem 2)**:
   - Block-64 NF4 carries **$3,065.8\times$ higher weight variance** than BF16 ($\eta_{\text{NF4}} = 0.008479$ vs $\eta_{\text{BF16}} = 2.77 \times 10^{-6}$).
   - While `1/√L` Pre-Norm residual MLPs with BF16 I/O incur `< 0.001` nat average KL at `2 dB` (matching $\Delta \text{KL}_q \approx \frac{1}{2}\sigma_{\Delta \ell}^2 \mathbb{E}[1-\|p^*\|_2^2]$ to `97%`), quantizing **Gated-DeltaNet decay/phase projections** over `T = 256` prefix tokens amplifies retention jitter `52×` and blows up `+20 dB` CTTC phase NLL by **`+0.373 nats` (`L=32`)** and **`+0.728 nats` (`L=64`)** (`7.5×–14.6×` over the `0.05`-nat budget).
   - **Decision**: Strictly reject QLoRA on `4B/9B` (BF16 LoRA fits and is `23%` faster); on `27B`, quantize **only SwiGLU MLPs (`63.61%` of params)** to NF4 while keeping all DeltaNet/Attention/Norm/Embed/`lm_head` layers in BF16 (`+20 dB L=64` KL = **`0.00032 nats`**).

3. **Cold-Start Gradient-Shock vs Two-Phase Anti-Shock Curriculum (Theorem 3)**:
   - Across 3 paired seeds (`202611, 202622, 202633`), unlocking LoRA at step 0 with an unaligned connector causes **`3.60×` larger step-0 shock (`5.43×` first-20-step shock)**, **`9.73×` larger general-text ungated KL drift** (`0.7095` vs `0.0729 nats`; still `2.51×` larger even under an identical Frobenius cap), text-shortcut memorization (`1.208` vs `1.430` text-only NLL), and worse OOD `G1/G3` (`62.83% / 0.9976`) than keeping the host frozen (`63.66% / 0.9883`).
   - **Two-Phase Anti-Shock** (`Phase A Frozen -> DPI Check -> Phase B Warmup+Cosine+FrobCap + Modality-Gate`) achieves the highest OOD accuracy (`G1: 63.77%`, `G3: 60.86%`) and **bit-exact `0.000000 nats`** pure-text KL drift.

4. **Cross-Domain Weight-Space Merging (Theorem 4)**:
   - Across near-orthogonal regime adapters ($\cos(\Delta W_1, \Delta W_2) = -0.0908$) trained separately on `Pluto-like (2dB + jitter + DC)` and `Large-CFO + Spur`, **Weight-Space Linear Soup** (`1.0089` NLL, macro `0.9956`) and **TIES-Merging** (`1.0266` NLL, macro `1.0104`) generalize to an **unseen mixed capture (`f708_snr10` analogue)** with **zero retraining**, outperforming Frozen (`1.1021`), single-regime adapters (`1.1409 / 1.0669`), and gauge-corrupted naive factor averaging (`1.0382`). Asymmetric Fisher cross-interference ($\Gamma_{12} = 2.27$ vs $\Gamma_{21} = 0.22$) penalizes unweighted `1.0` Task Arithmetic (`1.0655`).

5. **Modal `$12.88` Hard Decision Envelope & Two-Sided VoI Boundary (Theorem 5)**:
   - Evaluated on disjoint dev at **`$4.5236`** Phase-A exposure (before touching blind sets), the exact positive-VoI window for triggering Phase B (`Branch 1c: Paired-192` = **`+$1.83` / `+$2.17` with retry $\implies$ `$10.22` scheduled / `$11.24` all-retry total**, leaving `$1.64` buffer; or **`$11.75`** with 3-regime merging, **`Billed Cost = $0.00`**) is **`(0.030, 0.315]` nats**:
     - $\widehat{\Delta}_{\text{host}} \le 0.030\text{ nats}$ $\implies$ **Stay 100% Frozen (`Branch 0`, `$8.39–$9.50`, banking `$3.38–$4.49`)**;
     - $0.030 < \widehat{\Delta}_{\text{host}} \le 0.315\text{ nats}$ (Tier-1 geometry certified) $\implies$ **Trigger Phase B Surgical Gated-LoRA (`Branch 1c`)**;
     - $\widehat{\Delta}_{\text{host}} > 0.315\text{ nats}$ (or Tier-1 failure) $\implies$ **Abort before Phase B** and fix representation on CPU (`$0`).
   - Across 10,000 Monte Carlo trials, Adaptive Two-Phase raises gate pass rate from **`51.09% -> 83.01%`** at mean cost `$8.75`, with worst-case cost **`$11.24 < $12.88`** on `100%` of trials.

---

### Open Questions for Round 2 Cross-Track Integration (W1 / W2 / W3)

1. **For Track W1 (Gated-DeltaNet Circuits & Precision Sensitivity)**:
   - Our Theorem 2 probe predicts that `Qwen3.5`'s Gated-DeltaNet decay projections (`a_proj, b_proj`) and phase-transport projections are the dominant quantization failure mode at `+20 dB` (`ΔNLL = +0.728 nats` under NF4 at `L = 64`), whereas SwiGLU MLPs (`63.6%` of 27B params) tolerate NF4 cleanly. Does W1's exact DeltaNet $\leftrightarrow$ Kalman-RLS circuit construction confirm that **zero MLP weights** need Phase-B modification (allowing all Phase-B rank-8 LoRA budget to concentrate exclusively on DeltaNet `a/b/qkv/out` + Full-Attn `q/k/v/o` in `5.06M` params)?
   - Can W1 quantify $\Delta_{\text{route,max}}$ (our VoI upper cutoff $\approx 0.25\text{–}0.315\text{ nats}$) as a function of real `Qwen3.5-4B` candidate-Jacobian singular decay $\sigma_{\min}(D_{\text{answer}})$?

2. **For Track W2 (Cayley/Dynamic LoRA Parameterization & Merging Gauge)**:
   - Proposition 4.1 shows that arbitrary orthogonal gauge ambiguity $B_k Q_k Q_k^{-1} A_k$ breaks factor-level merging and forces merging in $\Delta W_k$-space (or rank-$Kr$ concatenation). Do W2's **Complex-Cayley / Givens Orthogonal LoRA** or **Phase-Conditioned Dynamic Hyper-LoRA** ($\Delta W(\theta, V) = \sum_n \alpha_n(V)[\cos(n\theta)\Delta W_{c,n} + \sin(n\theta)\Delta W_{s,n}]$) induce a **canonical aligned basis** across regimes (`Synthetic`, `Pluto`, `CTTC`) that makes cross-domain merging strictly Fisher-orthogonal ($\Gamma_{12} \ll 1$) and removes the asymmetric-norm penalty we observed in Exp 2A?
   - Under `Phase B-192` (`192` surgical steps inside `$11.24`), how much faster do **LoRA+ ($\eta_B \gg \eta_A$)** or **PiSSA/OLoRA spectral init** converge *after* Phase-A connector warm-start without triggering a Phase-B step-0 trust-region shock?

3. **For Track W3 (Nullspace Trust-Region, Modality Gating & Loss Engineering)**:
   - In Exp 1, combining **Two-Phase Warm-Start + Frobenius Cap** already reduced ungated general-text KL drift by `9.73×` (and `0.000000` pure-text KL under modality gating). What exact per-layer Frobenius radius $R_\ell$ (paired with W3's K-FAC / activation-covariance nullspace projection) guarantees **mixed multi-turn Chinese fluency KL $\le 10^{-3}$ nats** during EM conversations *after* `<W1><W2>` have turned $g_{\text{EM}}(x) = 1$?
   - Does replacing noisy 1-hot CE in Phase A + Phase B with **W3's Exact Bayes-Posterior KL Distillation** reduce the required Phase-B step budget from `192` steps down to **`69` steps** (`Branch 1b: $10.86` all-retry total, banking **`$2.02`** free Modal credit buffer instead of `$1.64`)?
