# Track W3 — Round 2 Cross-Track Integration & Resolution

**Your Track:** `W3` (Fisher/Nullspace Geometry, Modality Gating, Layer Selection `4B -> 9B -> 27B`, Objective & Distillation Engineering)
**Persistent Thread:** Keep full continuity with Round 1 in [THEOREM_W3.md](../../weight_mod/W3/THEOREM_W3.md) and [verify_w3.py](../../weight_mod/W3/verify_w3.py).

---

## Verified Converged Findings & Cross-Track Challenges (`W1, W2, W4, M1–M6`)

Your Round-1 deliverables verified `1.59e-12` Kronecker spectral error, `147.98×` nullspace KL reduction, `PureKL = 0.000e+00` + `80.2×` `MixKL` reduction (`MixKL = 0.01014`, `MixNLLd = +0.00148`), `113.732×` Rao–Blackwell gradient variance reduction, and **`+16.96 pp` Step-69 G1 lift (`44.30% -> 61.26%`, `ECE = 0.0431`)**. Now resolve four sharp empirical/theoretical points surfaced by `W1`, `W2`, and `W4`:

1. **MANDATORY Architecture Telemetry Correction (`|V| = 248,320`, NOT `151,936`!) from `W4`**:
   - `W4` audited the live `Qwen/Qwen3.5-4B` checkpoint against `smoke_audit.json` (`4,205,751,296` exact parameters, `0.0000%` error) and verified that **`Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.8-27B` ALL have vocabulary `|V| = 248,320`** (1.634× larger than `151,936`, driving 1.634× larger `lm_head` logit Fisher, backward activation footprint, and tied `embed_tokens`/`lm_head` tying on 4B — check `tie_word_embeddings` on 9B/27B as well!).
   - Update Theorem 5 / Block 5 parameter & memory tables in `THEOREM_W3.md` and `verify_w3.py` so every number uses exact **`|V| = 248,320`** and matches `W4` identically!

2. **Two Unresolved Architectural Tensions with `W1` and `W4`**:
   - **Tension A (Early Layers `1–10` Frozen vs Slow-Head `GDN` Starvation — `W1`)**:
     - Your Round 1 recommended freezing early layers `1–10` (4B/9B) and editing FA `12,16,20,24,28` + DN `11,15,19,23` (`1.64M` params).
     - `W1` proved that if frozen early GDN layers `1–10` have pretrained text decay $\alpha_{\text{prose}} < 1$, long intervening prose (`L_gap = 64 -> 128`) attenuates EM tokens by `99.3%–99.99%` *before reaching layer 11* (`Prefix-Only Q2_valid = 27.64%`), AND standard LoRA on `in_proj_a/b` hits **Gate-Gradient Starvation** unless 1–2 slow heads per GDN layer receive a zero-parameter constructive `dt_bias` warm-start (`50.7×` higher `L=64` retention `0.2899`, `1,576×` at `L=128`)!
   - **Tension B (`Stop-Grad` Backward VRAM Illusion — `W4 Theorem 1`)**:
     - `W4` proved that freezing early layers `1–10` saves **0% backward activations/compute** in Phase B if the Layer-0 EM connector $E_\theta(x)$ is co-trained (`0.15 η_A`), because gradients back to Layer 0 MUST traverse all `1..L` layers! You only capture the `~31%` backward VRAM/time savings if **$E_{\theta_A}$ is strictly frozen after Phase A (`stop_grad` after layer 10)** — OR if 1–2 slow EM heads in early GDN carry `warmgate` with `stop_grad` from general prose! Quantify BOTH options honestly!

3. **Closing `MixKL` Below `1e-3` Nats via Theorem 4.3 Token-Position Gating (`m_EM(t)`)**:
   - In Round 1, `4b modality_gated_nullspace_hybrid` reached `PureKL = 0.000e+00`, while `MixKL` sat at `8.40e-3` (seed `202611`: `1.014e-2`, `MixNLLd = +0.00148`) — slightly above your `1e-3` KL trust-region target!
   - Implement and benchmark your own **Theorem 4.3 Two-Tier Token-Position K/V Gate** (`m_EM(t) ∈ {0,1}` restricting ΔW_K, ΔW_V, ΔW_β strictly to EM token indices `t ∈ I_EM`, while keeping `g_EM = 1` + `P_{\perp, C}` + `MixKL` anchor on `Q, O, α`) inside `verify_w3.py`! Does `MixKL` drop strictly **`<= 1e-3` nats** while preserving `G1`, `Q2 (+7 pp)`, and `L_gap` retention?

4. **Symmetry Compatibility with `W2` (`Complex-Symmetrized P_{\perp, C}^{\mathbb{C}}` + Quotient `λ_orbit = 0`) & Shortened Pilot**:
   - `W2` proved naive real PCA `P_{\perp, C}` breaks `[P_{\perp, C}, J_d] = 0` needed by `Complex-Cayley` LoRA (`503.61 -> 1.0000` depth-32 grad norm, `2×` param efficiency). Co-design **`P_{\perp, C}^{\mathbb{C}}`** (`C_gen^{\mathbb{C}} = 0.5(C_gen + J_d^T C_gen J_d)`) and verify in `verify_w3.py` that it preserves BOTH `SO(2)` doublet pairing (`[P_C, J_d] = 0` to machine precision) AND `>100×` nullspace KL suppression!
   - Also confirm setting `λ_orbit = 0` when wrapped in `M1/W2` Quotient Wrapper (`C4 = 0.000e+00`) saves compute with zero accuracy loss, and measure `Step-69` vs `Step-96` vs `Step-240` convergence speed under `Bayes-KL + CF-Margin` so `W4` can assess cutting Phase-B steps from `192 -> 69–96` inside Modal `$12.88`!

---

## Round 2 Action Items for Track W3 (Execute Strictly Inside `weight_mod/W3/`)

1. **MANDATORY PERFORMANCE REFACTOR OF `verify_w3.py` (`947 s -> <= 160 s` CPU Wall-Clock!!)**:
   - Round 1 `verify_w3.py` took **`947.39 s`** (~16 minutes!) on CPU, which is way too slow for `verify_all.py`.
   - Optimize `verify_w3.py` mathematically/algorithmically WITHOUT losing rigor: keep all 5 core proofs/ablations + exact 8-layer 3:1 Hybrid WY model, cache frozen Task-A base pass logits/activations (since Task A base weights never change!), tune `torch.set_num_threads(2)`, vectorize K-FAC & eval passes, and streamline iterations/widths if cleanly justified so total runtime is **strictly `<= 160 seconds`** (`wall_seconds <= 160`) while passing all integrity assertions with exit code `0`!
2. **Add & Verify Round 2 Modules in `verify_w3.py` & `results_w3.json`**:
   - **Exact `|V| = 248,320`** in Block 5 (`4B / 9B / 27B` scaling ledger + `stop_grad` Phase-B Connector-Frozen vs Co-Tuned comparison).
   - **Theorem 4.3 Two-Tier Token-Position K/V Gate (`m_EM(t)`)** arm in Axis 1: test `PureKL = 0.000e+00` and measure `MixKL` reduction toward/below **`1.0e-3` nats** across both `L_gap = 8` and long-gap stress (`L_gap = 32` / slow-head GDN exception test!).
   - **Complex-Symmetrized Nullspace `P_{\perp, C}^{\mathbb{C}}` check**: verify `||[P_{\perp, C}^{\mathbb{C}}, J_d]||_F < 1e-12`, measure KL reduction vs real $P_{\perp, C}$, and confirm compatibility with `W2`'s Complex-Cayley adapter.
   - **Step-69 vs Step-96 vs Step-160 Convergence Profile** under `W3 Composite Bayes-KL + CF-Margin` loss (`λ_orbit = 0` quotient comparison included).
3. **Update `THEOREM_W3.md`**:
   - Fix `|V| = 248,320` everywhere, append clean Round 2 sections resolving Tensions A & B (`W1` early-GDN slow-head exception, `W4` stop-grad vs co-tuning backward VRAM, `W2` complex-symmetrized $P_{\perp, C}^{\mathbb{C}}$ & `λ_orbit = 0`), and tabulate all updated `results_w3.json` numbers (`wall_seconds <= 160 s`).
4. **Verify & Report**:
   - Run `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W3/verify_w3.py` and confirm **exit code `0`** and **`<= 160 s`**!
   - Return a crisp summary of verified metrics and converged architecture. Zero edits outside `weight_mod/W3/`.
