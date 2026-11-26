# Track M5 (Round 2): Non-Trivial Multi-Geometry Adaptive EIG, Exact 16-Bit/RefPool Integration & Cross-Host Stage 6 Lock

You are resuming persistent Codex session **Track M5** (`01a2194c-3b55-7153-bc9f-570b6f8c3c3f`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M5/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Findings & Round-1 Critique to Resolve (From M1–M6 & Self-Audit)
1. **Your Own Critical Finding in Round 1 (`fixed_time_4` Tie!)**:
   - In `results_m5.json` Round 1, `max_EIG` crushed `random` (`90.63% vs 71.63%`) and `nearest` (`70.21%`) on selective-fade — **yet time `4` won ALL 2,048 recordings**, meaning a static `fixed_best_on_dev = time 4` rule tied active per-recording `max_EIG`!
   - Wait: **what mathematically forces `max_EIG` to strictly dominate EVERY static `fixed_best_on_dev` window `a_const`?** Obviously whenever **recording-specific old cache `Z` carries conditional information about WHERE the ambiguity lies** (e.g., recording-varying selective deep fades seen via `log_radial(Z)`, OR heterogeneous change-point intervals $\tau$ localized differently per recording by `Z`, OR competing CFO slopes aliased on different candidate subsets per recording)!
2. **Locking the Unified 16-Bit Packet & Causal Real-Data Compatibility with M1, M3, M4**:
   - M4 showed that on real hardware without Stage 3's fixed SNR/amplitude, forcing SNR down to `2 dB` destroys `20 dB` (`0.00% -> 19.89%` SER), while M4's R3 whitened lift `(T, E)` preserves scalar precision cleanly, and M4 proposed **extracting BOTH `4-bit s17` and your `16-bit canonical` packet in Stage 4's zero-GPU CPU shard pass**!
   - Also check M3's `RefSufficientPool`: when revealing ONE candidate window $Y_{a^*}$ in Stage 6, $Y_{a^*}$ enters the same immutable `(acquisition_id, shares_channel, timestamp, K_ref, matched_T)` `RefSufficientPool` contract, handling shared CFO uncertainty $(\hat\omega, v_\omega)$ and correlated diffusion without double-counting!
3. **From M6 & M2 (Stage 6 Single Frozen Query Contract Across 4B / 9B / 27B & LEMMO)**:
   - M6 asked: can you freeze ONE concrete **Stage 6 Ultimate Blind Test Query & Evidence Contract** common to `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`, and `LEMMO` under identical pre-question cache bits (`16 bits/window`), candidate read budget (`k = 1` sealed read), pre-reveal forecast commitment scoring, AND M2's paired statistical gate (requiring strict win over `random`, `nearest`, AND **`fixed_best_on_dev`**!)?

## 2. Specific Round-2 Convergence Tasks for Track M5
Update `research/stage4_math_and_design/M5/THEOREM_M5.md`, `verify_m5.py`, and `results_m5.json`:
1. **Prove & Empirically Demonstrate Strict Per-Recording Adaptive EIG Dominance Over `fixed_best_on_dev`**:
   - Prove the exact mutual-information condition $I(a^*(Z); Y_a \text{ utility} \mid \text{marg}) > 0$ under which `max_EIG(Z)` strictly improves expected posterior NLL and accuracy over $\max_{a \in \mathcal{C}} \mathbb{E}_Z[\dots]$ (`fixed_best_on_dev`), i.e., Bayes value of conditional adaptive experiment design!
   - Add a new deterministic benchmark section in `verify_m5.py` (**`multi_geometry_adaptive_suite`** — e.g., varying localized jump intervals + varying candidate-window fading reliability + CFO aliasing across recordings) and verify empirically via paired CIs that **`max_EIG(Z)` strictly beats ALL FOUR controls: `random`, `nearest`, AND `fixed_best_oracle/dev`** on both post-reveal NLL and accuracy!
2. **Harmonize the Exact Bit-Layout of the 16-Bit Packet & `M1/M3/M4` Stage 4→6 Integration Table**:
   - Align byte/bit fields cleanly with M4 (`log_radial`, `resid_quality`, `s17_inner`, `turn`, `canon_angle`, erasure bit when zero/missing), show how Stage 4 CPU extractors populate BOTH `4-bit s17` and `16-bit canonical` views, and lock the Stage 6 `4B / 9B / 27B / LEMMO` frozen benchmark specification!
3. Run updated `verify_m5.py` via `uv run` cleanly and summarize exact numbers (`max_EIG` vs `fixed_best_on_dev`, `random`, `nearest`) in `reply_r2.md`.
