Running upgraded Round 2 `verify_w1.py` (all 14 FP64 proofs and the pre-training `B_align` conditioning audit completed in <10 s); waiting for the multi-arm benchmark to finish and write `results_w1.json`.
Running a quick smoke test on the two new Round 2 arms (`tier1_pos_gated_deltanet_early_frozen` and `unified_w1_w2_w3_warmgate_schur`) at `L_gap = 64` to verify convergence and speed before locking the final `<= 150 s` `verify_w1.py` schedule.
Running the full Round 2 `verify_w1.py` benchmark (with separated `Q`/`KV` Tier-1 gating, zero-DC candidate-nullspace Schur skip, and fast alternating single-pass schedule); waiting for completion to inspect `results_w1.json`.
`task-166` is finishing the remaining 6 runs (`warmgate`, `tier1_pos_gated_deltanet_early_frozen`, and `unified_w1_w2_w3_warmgate_schur` at `L_gap=64`); waiting for completion to inspect all metrics and eliminate the eval overhead so total wall-clock stays strictly `<= 150 s`.
Running the finalized Round 2 `verify_w1.py` benchmark via `uv run` (targeting `<= 150 s` wall-clock); waiting for completion to verify all metrics and update `THEOREM_W1.md`.
`task-184` is at 12/15 runs (finishing the last `tier1` seed and both `unified_w1_w2_w3_warmgate_schur` seeds at `L_gap = 64`); waiting ~35s to inspect their exact metrics before trimming ~60s of runtime so `verify_w1.py` finishes strictly `<= 150 s`.
Running Round 2 verification script [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) in the background (`uv run`, targeting `<= 150 s` CPU wall-clock); awaiting completion notification before inspecting `results_w1.json` and appending the Round 2 cross-track sections to [THEOREM_W1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md).
Awaiting completion of the profiling snippet to inspect dataset build time and CPU thread scaling before finalizing the speed and convergence tuning of [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py).
Checking exact train vs. eval wall-clock breakdown from the running diagnostic before finalizing `verify_w1.py` optimization.
Benchmarking exact chunked WY Gated-DeltaNet (`C = 24, 32`) equivalence and CPU speedup before updating [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py).
Running a quick 4-arm joint-training and chunked-WY (`C = 24`) convergence/speed check to lock optimal parameters for [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) strictly under `150 s`.
Awaiting the chunked-WY train/eval micro-profile to lock the optimal `< 130 s` schedule for [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py).
Checking per-block (`GDN` vs. `FullAttn`) CPU latency and thread overhead before finalizing edits to [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py).
Running the parallelized Round 2 verification suite [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) (`15` matched arms across `L_gap ∈ {16, 64}` on 6 CPU workers, Round-1-matched `d = 32` host with chunked WY GDN and W2 Theorem 7 Complex-Symmetrized Nullspace Cayley); awaiting completion to inspect `results_w1.json` and append Round 2 sections to [THEOREM_W1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md).
Running the final Round 2 verification suite [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) on 8 parallel CPU workers (Longest-Job-First scheduling, targeting `<= 150 s`); awaiting completion to inspect `results_w1.json` and append the Round 2 cross-track convergence sections to [THEOREM_W1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md).
Running the tuned 6-physical-core LJF verification suite [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) (`112` joint Q3+Q2 steps, `B = 20`, targeting `~115 s <= 150 s`); awaiting completion to inspect `results_w1.json` and update [THEOREM_W1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md).
All Round 2 instructions in [prompt_r2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/discussions/W1/prompt_r2.md) have been executed verbatim strictly inside `research/stage4_math_and_design/weight_mod/W1/`.

### Deliverables & Verification Receipt
- **Theory & Cross-Track Proofs**: Appended Round 2 Sections **7–9** cleanly to [THEOREM_W1.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/THEOREM_W1.md) (459 lines; Round 1 §§1–6 preserved intact).
- **Upgraded & Optimized Verification Suite**: [verify_w1.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/verify_w1.py) $\to$ [results_w1.json](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/weight_mod/W1/results_w1.json).
- **Execution Command & Runtime**:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W1/verify_w1.py
  ```
  **Exit code `0`**, total wall-clock **`126.39 s`** (**`<= 150 s` gate passed**, `4.2×` faster than Round 1 via FLA-style chunked WY Gated-DeltaNet [`1.11e-15` FP64 exactness] + Longest-Job-First 6-physical-core parallelism). `git diff --name-only` confirms zero edits outside `research/stage4_math_and_design/weight_mod/W1/`.

---

### Key Resolved Cross-Track Theorems & Verified Round 2 Metrics

1. **Joint `W1 × W2 × W3 × M3` Circuit, Rank & Anti-Forgetting Resolution (Theorems W1.5A–W1.5C)**:
   - **W1 vs. W3 Early-Layer Freeze Conflict (Theorem W1.5A)**: Proved Pre-Norm vertical skip survival carries `<W1><W2><PHASE>` at $t_{\text{mod}}$ vertically into Layer 11 with **zero horizontal $\bar{\alpha}^{L_{\text{gap}}}$ decay**; applying W1 warmgate + W3 Tier-1 Token-Position Gate on Layer 11+ locks lossless `1.000` retention across `L_gap = 64 → OOD@128`. Whenever a live checkpoint probe shows early vertical attenuation ($\sigma_3 < 0.05\,\sigma_1$), the **Minimal Early-GDN Preservation Exception** (`2` early DN layers `3, 7`, `4` Tier-1 pos-gated slow heads only = `+40,976` params on 4B, `PureKL = 0.0`) restores full `2.41×` $\sigma_3$ gain.
   - **W2 $B_{\text{align}} = M_{\text{cand}}^+ F_{\text{real}}$ Pre-Training Conditioning Audit (Theorem W1.5B)**:
     | Pre-Training Config (`d = 32`) | `L=64` $(\sigma_1, \sigma_2, \sigma_3)$ | `L=64` $\kappa_3$ / Curv. Amp ($1/\sigma_3^2$) | `L=128` $\sigma_3$ / $1/\sigma_3^2$ | $\cos(V_3(64), V_3(128))$ |
     |---|---:|---:|---:|---:|
     | `frozen_base` | `(0.0588, 0.0504, 0.0405)` | `1.45` / **`609.0`** | `0.0156` / **`4,090.2`** | `0.579 / 0.649` |
     | `warmgate_all_gdn` | `(0.0762, 0.0645, 0.0592)` | `1.29` / `285.8` (`2.13×↓`) | `0.0250` / `1,594.5` | `0.612 / 0.708` |
     | `tier1_pos_gated_early_frozen` | `(0.0837, 0.0752, 0.0663)` | **`1.26`** / `227.5` (`2.68×↓`) | `0.0365` / `752.5` | **`0.752 / 0.807`** |
     | `tier1_pos_gated_all_gdn` | **`(0.1282, 0.1215, 0.0977)`** | `1.31` / **`104.8`** (**`5.81×↓`**) | **`0.0813`** / **`151.2`** (**`27.1×↓`**) | **`0.728 / 0.854`** |
   - **Exact Algebraic Composition & `M3/ref_pool.py` Parity (All 14 FP64 Identities `< 1e-12` in `0.08 s`)**:
     - W1 **3-Key Schur-Locked GDN** ($V_1 \oplus V_2$, zero DC): $2\pi$-periodicity `4.16e-16`, $\frac{\pi}{2}$-covariance `8.33e-17`; composes directly with W2 Theorem 7 **Complex-Symmetrized Nullspace Cayley** (`orth_err = 1.41e-15`, `SO(2) = 0.0`) and **zero-DC candidate-nullspace skip** $P_{\text{cand,null}}$.
     - **M3 Correlated-Noise Schur-Complement Innovation & Immutable `seen_ids` Ledger inside Tier-1 Gated GDN**: exact joint log-likelihood err **`1.78e-15`**, duplicate/distractor state drift **`0.0` / `0.0`**.

2. **Reconciled `<= 2.20M` Surgical Parameter Budget (`|V| = 248,320`) & W4 VoI Derivation (Theorem W1.6)**:
   - **`Qwen3.5-4B`** (`4,205,751,296` total): Core unified `W1×W2×W3×W4` = **`1,974,782`** (`0.047%`); + 2-layer early-GDN exception = **`2,015,758`**; + optional Step-69-triggered sparse Tier-2 SwiGLU (`2` upper FFN layers, `r = 2`) = **`2,116,094`** — all strictly **`<= 2.20M`** and below `EM-Frozen-Connector`'s `2,184,964` parity ceiling (`Qwen3.5-9B`: `2,586,110` core; `Qwen3.8-27B`: `4,754,982` core).
   - **W4 Upper VoI Bound**: Derived $\Delta_{\text{route,max}}(\gamma) = \mathbb{E}[\mathrm{KL}(p^* \,\|\, \mathrm{softmax}(\gamma \log p^*))]$ from $\gamma = \sigma_3^{\text{frozen}}(L_{\text{gap}})/\sigma_{\text{crit}}$ and `RMSNorm` radial saturation: uniform ceiling `0.3479 nats` ($\gamma = 0$), evaluating directly to **`0.242–0.311 nats`** across measured frozen `L_gap = 64` contraction $\gamma \in [0.05, 0.15]$ (explaining W4's empirical `0.25–0.315 nats` cutoff).

3. **Round 2 Matched `L_gap = 64` Multi-Arm Benchmark (2-Seed Mean, All Arms Bitwise `C4 = 0.0e+00`)**:

   | Arm (`Host Params`) | `G1` Acc / NLL | `G5 (θ=π)` Acc / NLL | `Q2` `v → f` (`Gain`) | `Q2` `vi` / `dup` ($\Delta$) | `OOD@128` `G1 / G5` | `L2 Ret(64/128)` | `PureKL` | `MixKL` |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | `prefix_only` (`0`) | `33.40% / 1.4652` | `17.77% / 2.2524` | `25.0 → 27.3%` (`+2.3`) | `+0.5` / `+2.6` | `33.4% / 17.8%` | `0.017 / 0.0003` | `0.0000` | `0.0000` |
   | `attn_lora` (`2,048`) | `62.99% / 1.1393` | `12.01% / 2.2099` | `51.8 → 58.6%` (`+6.8`) | **`-16.7`** / `+2.1` | `33.4% / 17.8%` | `0.023 / 0.0005` | `0.3062` | `0.2898` |
   | `deltanet_attn_lora` (`7,892`) | `63.04% / 1.1410` | `8.25% / 2.2905` | `52.9 → 62.5%` (`+9.6`) | **`-14.8`** / `+2.6` | `33.4% / 17.8%` | `0.016 / 0.0006` | `0.3974` | `0.3658` |
   | `deltanet_attn_lora_warmgate` (`7,892`) | `63.72% / 1.1211` | `5.71% / 2.2376` | `56.8 → 64.6%` (`+7.8`) | `-12.0` / `+0.8` | `33.4% / 17.8%` | `0.420 / 0.231` | `0.3491` | `0.5327` |
   | **`tier1_pos_gated_deltanet_early_frozen`** (`3,996`) | `64.21% / 1.1196` | `10.11% / 2.2585` | `51.3 → 63.0%` (`+11.7`) | **`-0.3`** / **`-0.8`** | **`60.7%`** / `11.9%` | **`1.000 / 1.000`** | **`0.0000`** | **`0.0545`** |
   | **`unified_w1_w2_w3_warmgate_schur`** (`3,328`) | **`64.36% / 0.9715`** | **`63.28% / 1.0118`** | **`60.2 → 69.0%`** (**`+8.9 / +0.171 nats`**) | **`-0.8`** / **`0.0`** | **`64.4% / 63.3%`** | **`1.000 / 1.000`** | **`0.0000`** | **`0.0449`** (**`11.9×↓`**) |

   - `unified_w1_w2_w3_warmgate_schur` uses **`58%` fewer host params** than ungated `warmgate` (`3,328` vs. `7,892`), matches exact physical oracle top-1 on `G1` (`64.36%`), `G2` (`47.17%`), `G3` (`62.11%`), and unseen `G5(θ=π)` (`63.28%`), sustains zero degradation on `OOD@128` (`64.4% / 63.3%`, `Q2 51.6% → 64.1%`), achieves exact duplicate idempotence (`+0.0 pp`) and distractor immunity (`-0.8 pp`), preserves bit-exact `PureKL = 0.0000`, and cuts `MixKL` **`11.9×`**.
