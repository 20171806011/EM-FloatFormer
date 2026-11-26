# Track M2 (Round 2): Reconciling Power vs Modal Free-Credit Scope, Causal Arm Gate & Multi-Claim Hierarchy

You are resuming persistent Codex session **Track M2** (`01a21948-8b7b-71d0-a50a-d7c7eaf61018`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M2/` and execute Python ONLY via `uv run --python 3.12 ...`.

## 1. Critical Cross-Track Findings & Tensions to Resolve (From M1, M3, M4, M5, M6)
1. **The Biggest Statistical-Budget Tension (M2 vs M4 & M6)**:
   - Your Round 1 analysis proved that across **retraining seeds**, 3 random seeds cap even infinite-block crossed random-seed power at `71.54%` (Acc) / `85.94%` (NLL) under G1 proxy variances (with empirical Monte Carlo ~22–30%), requiring **`>= 10` paired seeds** for population-of-seeds inference!
   - Meanwhile, M4 (`$8.44` total on A100-80GB) and M6 (`$9.50` cap against `$12.88` remaining Modal free credits, `Billed Cost: $0.00`) CANNOT afford 10 seeds (`10 × $2.50 = $25+`!). M4 specifically proposed `1` seed (`202611`) mixed synthetic+real fitting (`384` steps, `$2.87`) while M6 proposed a `1`-seed pilot (`<= 173/960` steps, `$0.49`) continuing up to `3` seeds (`$5.10`) ONLY if pilot clears futility!
   - **You must resolve this cleanly**: Define the exact mathematical distinction between **Fixed-Connector / Conditional Seed Claim** (`S=1` or `S=3` frozen seeds, valid conditional $t$-/permutation-inference over independent blind blocks/captures) vs **Random-Seed Retraining Claim** (strictly prohibited below sufficient `S`, or scoped cleanly!). Also address: under M6's 1-seed futility stop ($t=0.18$ on disjoint dev), prove exact conditional Type-I control when the blind test uses disjoint blind blocks/sessions!
2. **Causal Isolation Gate for M1's `Host-Pure` vs `Physical-Residual` Arms + M6 DPI**:
   - M1's Round 1 `B` achieved `64.62%` via a physical S17 head + tiny host residual. Both you and M6 flagged: that doesn't by itself prove frozen Qwen reads coupled evidence!
   - Lock the exact paired contrasts for M1's 4-arm matrix (`Zero`, `Naive-Additive`, `Host-Pure Equivariant`, `Physical+HostResidual`, plus `Direct S17` & `Full IQ`), including:
     - When `Host-Pure` claims LLM phase-coupled extrapolation (`Arm 1` vs `Zero` & `Naive-Additive` + dev-calibrated NIG $\eta_{\min} \ge 0.10$),
     - When `Physical+HostResidual` claims residual value over `Direct Physical` (`Arm 2` vs `Direct Physical` on blind NLL/Acc with matched-text slot interventions!),
     - Pre-registered calibration/temp-fitting scope: separate **dev-locked zero-shot calibration** vs **oracle retrospective sensitivity**, strictly preventing blind recalibration from ever passing a gate!
3. **M3 (`RefSufficientPool`) & M4 (Pluto / CTTC) Real-Data Inference Structure**:
   - M4 locked: Pluto (`2,240` test frames in `16` chronological blocks, `140` stress frames, pilots `15, 18`), CTTC (`1,440` packets across `9` captures × `8` loops = `72` session-loops, `720` Q3 loop-pair groups, held-out `f708_snr10` capture, shared cyclic TX source `[334_096, 455_000)`!), and zero-permutation Q2 `valid / duplicate_raw / valid+indep / fresh` checks.
   - Specify the exact **Holm-Bonferroni / IUT Hierarchy** protecting Family-Wise Error Rate (`alpha = 0.05`) across `(Claim 1: Synthetic Extrapolation + Q2, Claim 2: Pluto, Claim 3: CTTC + held-out f708_snr10)`, handling CTTC's capture-level (`n=9`) vs loop-pair (`n=36`) vs cluster (`n=72`) correlation cleanly (and never claiming `n=72` i.i.d. if source sequence & captures induce session dependence!).
4. **M5's Stage 6 Gate Compatibility**:
   - Include a crisp specification for Stage 6's `max_EIG` vs `{random, nearest, fixed_best_on_dev}` gate (plus `4-bit s17` vs `16-bit canonical` comparison)!

## 2. Specific Round-2 Convergence Tasks for Track M2
Update `research/stage4_math_and_design/M2/THEOREM_M2.md`, `verify_m2.py`, and `results_m2.json`:
1. Formalize the **Scoped Three-Tier Claim Ladder** (Conditional Frozen-Connector Claim `S=1..3` vs Multi-Seed Retrain Claim `S>=10`) + prove conditional validity after M6's Stage-1 dev futility screen + compare power curves for Stage 4's exact `1`-seed and `3`-seed Modal plans on synthetic `32` vs `131` blocks, Pluto `16` blocks, and CTTC `9` captures / `72` clusters.
2. Lock the **Exact Stage 4 Formal Preregistered Statistical Gate Table** unifying M1's `Arm 0/1/2` contrasts, M3's `Q2 two-valid > one-valid > none` & exact `0.0` invariance assertions, M4's Pluto/CTTC real-data hierarchy, and M5's `fixed_best_on_dev` Stage-6 gate.
3. Add a fast deterministic numerical check in `verify_m2.py` verifying: (a) conditional Type-I error under M6's dev-disjoint futility selection, and (b) synthetic `32`-block (M4 budget) vs `131`-block power & Pluto/CTTC conditional fixed-seed power curves (`results_m2.json`). Run via `uv run` and summarize cleanly in `reply_r2.md`.
