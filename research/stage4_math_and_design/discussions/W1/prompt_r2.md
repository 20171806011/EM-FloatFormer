# Track W1 — Round 2 Cross-Track Integration & Resolution

**Your Track:** `W1` (Mechanistic Circuits, Gated-DeltaNet $\leftrightarrow$ Complex Kalman-RLS Isomorphism, Expressivity Separation & Spectral Rotation Circuits)
**Persistent Thread:** Keep full continuity with your Round 1 derivations in [THEOREM_W1.md](../../weight_mod/W1/THEOREM_W1.md) and [verify_w1.py](../../weight_mod/W1/verify_w1.py).

---

## Verified Converged Findings from `M1–M6` Round 2 and Peer Tracks `W2`, `W3`, `W4`

You now have complete, independently verified Round-1/Round-2 results across the entire 10-track research swarm:

1. **Converged `M1–M6` Round 2 Constraints (`L_gap = 0` Oracle Baseline & Wire Lock)**:
   - **M1 Round 2**: On a gap-free (`L_gap = 0`) frozen 1-layer host surrogate, `Host-Pure` Equivariant Connector reached **`64.60%` (`G1`) / `61.91%` (`G3`)** with `0.0` canonical equivariance defect (`<= 1.49e-7` ungrouped); your W1 Round 1 proved why this breaks at realistic prompt gaps (`L_gap = 64 -> Prefix-Only` collapses `G1: 60.40% -> 43.41%`, `Q2_valid: 59.28% -> 27.64%`, GDN retention `0.0057`), while your `deltanet_attn_lora_warmgate` restored **`G1 = 64.36%`**, **`Q2 = 65.04% -> 72.07%`**, zero distractor bleed (**`+0.20 pp`**), `1.73×` larger `σ_min(D_answer) = 0.1684`, and `50.7×` / `1,576×` higher `L=64` / `L=128` GDN retention (`0.2899 / 0.1068`).
   - **M3 Round 2 (`M3/ref_pool.py`)**: Locks an idempotent, order-invariant, acquisition-ID-deduplicated physical `RefSufficientPool` (exact Schur anti-double-counting to `8.0e-15`, shared-CFO two-clock NLL `0.147654` vs pooled `0.378354`).
   - **M5 Round 2 (`m5-c16-v1-le`, `m5-stage6-blind-v1`)**: Extends `s17` to a 16-bit self-contained word (`[2b turn | 2b s17_inner | 6b angle | 4b log_radial | 2b quality]`) and requires Stage 5–6 multi-turn active acquisition & causal forecasting (`<= 2.20M` trainable connector/adapter ceiling on Qwen3.5-4B for fair attribution).

2. **Peer Track `W2` (Verified `110.67 s`, `exit 0`)**:
   - Proved **Quotient-Wrapped `Complex-Cayley + Schur-Irrep Dynamic Hyper-LoRA` (`4c`, `3,022` params)** hits **`G1 = 64.60% / 0.9526` already at Step 69**, **`G1 = 64.57% / 0.9491` & `G5(θ=π) = 61.33% / 0.9988`** (vs static PEFT collapse `15.2%–19.5%`) with **exact bitwise `0.000e+00` $C_4 \times C_4$ defect**, Work-F orientation gain **`0.9537` (`r=3`: `0.9743`)**, and depth-32 Cayley gradient norm ratio **`0.99999988`** (vs unconstrained `503.61`).
   - **Direct Question from `W2` to `W1`**: `W2`'s Schur-Irrep Hyper-LoRA constructs its 3D candidate-aligned frame $B_{\text{align}} = M_{\text{cand}}^+ F_{\text{real}}$ from $M_{\text{cand}} = \Pi C_{\text{cand}} E_{\text{vocab}} D_{\text{answer}} B$. On your hybrid `[GDN×3, FullAttn×1]` host at `L_gap = 64`, does frozen $M_{\text{cand}}$ stay full centered rank 3 with sufficient $\sigma_3/\sigma_1$ BEFORE GDN gate adaptation, OR does `warmgate` (slow-head GDN survival) have to stabilize $D_{\text{answer}}$ first so $B_{\text{align}}$ stays well-conditioned across `L_gap = 64` and `OOD@128`? Second, can W1's **2-Key GDN Bilinear Readout** (`S_w1 q_phase = R(θ)u_1`) combine directly with W2's **Schur-Irrep ($V_1 \oplus V_2$) + Complex-Cayley**?

3. **Peer Track `W3` (Completed `exit 0`, Independent Verify Running)**:
   - Proved **Modality-Gated (`g_EM ∈ {0,1}`) + Input Nullspace Projection (`ΔW = B Ã P_{\perp, C}`)** yields **bit-exact `0.000e+00` pure-text KL**, cuts mixed multi-turn KL `80.2×` (`MixKL = 0.01014`, `MixNLLd = +0.00148`), and **Bayes-Posterior KL + CF-Margin Distillation** cuts Rao–Blackwell gradient variance **`113.732×`**, lifting **Step-69 pilot `G1` by `+16.96 pp`** (`44.30% -> 61.26%`, `ECE = 0.0431`).
   - **Direct Tension & Questions from `W3` to `W1`**:
     - *Early-Layer Freeze Conflict*: `W3` recommends freezing early Qwen layers `1–10` (editing only FA `12,16,20,24,28` + DN `11,15,19,23`, `1.64M` params on 4B) to save ~31% backward VRAM & lexical forgetting. Yet your Theorem W1.1 warns that if frozen early layers `1–10` contain fast-decaying pretrained GDN heads ($\bar{\alpha}^{64} = 0.007$), they could erase `<W1><W2>` *before Layer 11 ever sees them*! Can we resolve this cleanly via a **Minimal Early-GDN Preservation Exception** (applying a zero-extra-parameter / tiny Tier-1 Token-Position-Gated `dt_bias` + slow-head mask strictly on `2–4` dedicated EM heads across GDN layers, leaving lexical/text FFN and prose K/V untouched)?
     - *Two-Tier Token-Position Gate & `M3/ref_pool.py`*: Verify formally and empirically whether W3's **Tier-1 Token-Position Gate ($m_{\mathrm{EM}}(t) \in \{0,1\}$)** on GDN write keys/values/β ($\beta_t = 0, g_t = 0$ on non-EM tokens; $\beta_t = K_t, \alpha_t = e^{-\sigma_\phi^2\Delta t/2}$ only on `<REF>` tokens; `g_EM` sequence gate on $Q, O$) simultaneously achieves (i) W1's exact Kalman-RLS tracking across `L_gap = 64 -> OOD@128`, (ii) exact `M3/ref_pool.py` duplicate-ID idempotence & distractor immunity (`0.0` drift), AND (iii) W3's near-zero mixed-turn prose forgetting!

4. **Peer Track `W4` (Verified `62.71 s`, `exit 0`)**:
   - Proved **NF4 quantization of GDN decay/phase gates** destroys `+20 dB L=64` phase accuracy (`3,065.8×` MSE, **`+0.72827` nat NLL explosion** vs BF16), whereas **`Mixed MLP-NF4 + DeltaNet/Attn/IO-BF16`** restores `KL = 0.00032 nats`.
   - Proved a **`3.60×` Cold-Start Gradient Shock** (`9.73×` KL forgetting) if backbone weights unfreeze before Phase A converges, locking **Two-Phase Anti-Shock** inside a **`$12.88` Modal cap** (`<= $11.24–$11.75`, `Billed Cost: $0.00`).
   - Also proved **Qwen3.5-4B/9B/3.8-27B** exact vocabulary size is **`|V| = 248,320`** (`4,205,751,296` params on 4B), and asked W1 to specify an **ultra-compact `<= 2.20M` surgical parameter tier on 4B** (dropping W1 Round 1's optional Tier-2 `4.52M` FFN LoRA unless Step-69 diagnostics prove SwiGLU nonlinearity necessary, or isolating `<= 0.5M` sparse Tier-2 if needed) AND quantify how early-GDN $\sigma_3(M_{\text{cand}})$ bounds W4's upper VoI cutoff ($\Delta_{\text{route,max}} \approx 0.25\text{–}0.315$ nats).

---

## Round 2 Action Items for Track W1 (Execute Strictly Inside `weight_mod/W1/`)

1. **Resolve ALL Cross-Track Mathematical & Architectural Questions in `THEOREM_W1.md`** (append cleanly as Round 2 sections without breaking Round 1 proofs):
   - **Joint `W1 × W2 × W3 × M3` Circuit & Rank Theorem**: Prove exact conditions under which (a) frozen early layers `1–10` survive vs require 2–4 `warmgate` / Tier-1 Token-Position-Gated EM heads, (b) `M_cand` retains rank 3 (`σ_3 / σ_1`) at `L_gap = 64` for `W2`'s `B_align`, (c) `W1`'s **2-Key GDN Bilinear State + Householder/SwiGLU** integrates with `W2`'s `Quotient + Complex-Cayley + Schur-Irrep` and `W3`'s **Tier-1 Token-Position Gate + Bayes-KL distillation**, and (d) GDN orthogonal clock keys $k_{t_j}$ implement `M3/ref_pool.py`'s immutable acquisition-ID deduplication rule (`seen_ids` mask $\to \beta_t = 0$) and Schur orthogonal innovation update under correlated references.
   - **Reconcile Surgical Parameter Budget (`<= 2.20M` Cap on `Qwen3.5-4B`)**: Show an explicit unified parameter table (`4B / 9B / 27B` with `|V| = 248,320`) reconciling W1 Tier-1A/1B/1C (slow-head selective GDN `in_proj_a/b/dt_bias/qkv/out` + FullAttn `QKVO`) with W2 (`Complex-Cayley + Schur-Irrep`) and W3 (`Layer-selected FA+DN + Tier-1 Token-Position Gate`), strictly keeping total trainable params **`<= 2.20M` on 4B** for Phase-B parity with `EM-Frozen-Connector` (and noting when optional Tier-2 upper-FFN `~0.5M` calibration LoRA is triggered). Quantify `W4`'s $\Delta_{\text{route,max}}$ in terms of $\sigma_3(D_{\text{answer}}(L_{\text{gap}}))$ and RMSNorm radial saturation.
2. **Upgrade & Speed-Optimize `verify_w1.py` (`<= 150 s` CPU Wall-Clock!!)**:
   - **MANDATORY SPEED TARGET**: Round 1 `verify_w1.py` took ~530 s because of 16 sequential CPU training runs (`L_gap ∈ {16, 64}` × 4 arms × 2 seeds). Vectorize/optimize batch & eval evaluation cleanly (e.g., cached fixed prose prefix forward, streamlined single-thread/batched eval, or concise deterministic step schedule tuned cleanly) so the ENTIRE upgraded `verify_w1.py` executes end-to-end in **`<= 150 seconds`** via `uv run` with exit code `0`!
   - **New Round 2 CPU Experiments inside `verify_w1.py` (saved cleanly to `results_w1.json`)**:
     1. Measure centered singular values `(σ_1, σ_2, σ_3)` and condition number `σ_1 / σ_3` of $D_{\text{answer}}$ / $M_{\text{cand}}$ at `L_gap = 64` across Frozen vs `warmgate` (evaluating viability of W2's $B_{\text{align}}$).
     2. Add **2 new unified arms** at `L_gap = 64` (+ `OOD@128` + unseen `G5(θ=π)` + `Q2_valid / fresh / valid+indep / dup_raw` + **MixKL prose drift**):
        - `tier1_pos_gated_deltanet_early_frozen`: tests W3's early-GDN-frozen hypothesis vs slow-head exception at `L_gap = 64`!
        - `unified_w1_w2_w3_warmgate_schur`: integrates `W1 warmgate` + `W3 Tier-1 Token-Position Gate + Bayes-KL` + `W2 Complex-Cayley/Schur-Irrep` (`G1/G3`, **`G5(θ=π)`**, **`OOD@128`**, **`Q2` Blackwell**, **`PureKL = 0.0`**, **`MixKL`**, and **exact `M3/ref_pool.py` duplicate/distractor parity**).
3. **Final Verification & Reporting**:
   - Re-run `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W1/verify_w1.py` and confirm **exit code `0`** and **wall time `<= 150 s`**.
   - Return a crisp summary of all new verified Round 2 metrics and unified conclusions. Do NOT touch any file outside `research/stage4_math_and_design/weight_mod/W1/`.
