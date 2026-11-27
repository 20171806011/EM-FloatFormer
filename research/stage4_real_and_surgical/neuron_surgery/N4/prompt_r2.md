# Track N4 (Round 2): Cross-Track Reconciliation & Unified 4B/9B/27B Neuron-Surgery Blueprint

This is **Round 2 (Cross-Review & Convergence)** resuming your Track **N4** session (`495a9a02-4de6-4d71-9bbd-dfd8d9a79421`).

## 0. Strict Workspace & Verification Rules
- **Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N4/` **ONLY**. Do NOT edit any file outside `N4/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Always execute via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>`.

## 1. Peer Round-1 Findings to Inspect Read-Only & Audit
Read your three peer tracks' Round-1 summaries (and inspect their `THEOREM_N*.md` / `verify_n*.py` as needed):
- `research/stage4_real_and_surgical/neuron_surgery/N1/reply_r1.md` (`THEOREM_N1.md` — minimal `6–16` SwiGLU neuron bounds, universal Fermi-Dirac $\mathcal{F}_q$ `RMSNorm` quotient across 2–3 layers, **`9.49%` missing $D_{\gamma_f}$ frame tilt** in `surgical_adapter.py#L291`, **`1.352` BF16 Fermi-Dirac drift** if ungated)
- `research/stage4_real_and_surgical/neuron_surgery/N2/reply_r1.md` (`THEOREM_N2.md` — `RMSNormGated` `1,756x` blowup & single ballast channel, `r_hd=2` complete KV-head group, 192-dim NoPE FA intrinsic trigger + Candidate-Orthogonal Bus, `138` active neurons = `445,556` dense params)
- `research/stage4_real_and_surgical/neuron_surgery/N3/reply_r1.md` (`THEOREM_N3.md` — **`9/9` Cayleys had `u_c=None`** because `2560 != 4096`!, **`in_proj_a` m_EM mask bug**, Zipf false-dormancy 100% spike, `SpecCayley` + Clipped Keyed SwiGLU + AL-Hinge hitting `MixKL = 7.21e-4` & beating `lemmo` by `+3.6 pp` on `G1_hard`)

## 2. Specific Cross-Track Challenges for Track N4
1. **Unify `N1 + N2 + N3 + N4` Bugs & Remedies into ONE Exact `Qwen3.5-4B` (`BF16`) / `27B` (`Mixed-NF4`) Blueprint**:
   - N3 found **`9/9` Cayleys in `surgical_adapter.py` actually ran with `u_c = None`** (because `o_proj.in_features = 4096 != d_model = 2560`), AND N1 found **`B_align` omitted $D_{\gamma_f}$** (`9.49%` tilt), AND N4 found **`4`-prompt K-FAC rank `<= 88 < 256`**, AND N2 found **`r_hd=2` GDN ballast + NoPE FA bus**.
   - Audit N4's own `verify_n4.py` / `CN-SFT` design against these findings:
     - Incorporate $B_{\text{align}}^{(\gamma_f)} = (M_{\text{cand}} D_{\gamma_f})^+ F_{\text{real}}$ (`N1`), head/value-space K-FAC projection matching `o_proj.in_features=4096` (`N3`), `N1`'s Circuit A/B `RMSNorm`-invariant Fermi-Dirac / κ-pregain check across $4\times$ prose norm variations, and `N3`'s Stationary-Keyed Dead-Zone + AL-Hinge `MixKL` controller (`MixKL <= 1e-3`).
2. **Quantify Combined Effect on `BF16` & `27B Mixed-NF4` + Compare Parameter & Cost Budgets**:
   - Check N1 (`1.352` BF16 Fermi-Dirac drift when ungated) + N3 (`MixKL = 7.21e-4` with keyed clipped dead-zone) inside N4's `BF16` and `Mixed-NF4 + FP32 Overlay` framework! Prove whether combining `g_EM` hard gate + N2's NoPE/Fisher query gate + N1's Circuit A/B (`kappa_scratch = 8.0`) + N3's AL-Hinge achieves **simultaneous** `PureKL == 0.0`, `MixKL <= 1e-3`, exact oracle `G1/G5`, `< 1.85M` params (`< 85%` of `TIER_P1_MAX_PARAMS`), and `< $0.45` Modal A100 runtime!
3. **Lock Final Pre-Flight & Modal A/B Test Matrix (`4B` -> `9B/27B` Transfer)**:
   - Reconcile the exact ablation arms (say 3–4 lean arms within `$0.45`) and the exact `$0.02` Layer-10 linear recoverability check + `K-FAC 4096-dim` projection sanity check so future Modal execution has near-zero trial-and-error cost.

## 3. Deliverables & Re-Verification
1. Append a rigorous **"Round 2 Cross-Track Reconciliation & Converged Implementation Spec"** section to `research/stage4_real_and_surgical/neuron_surgery/N4/THEOREM_N4.md`.
2. Update `research/stage4_real_and_surgical/neuron_surgery/N4/verify_n4.py` if needed (e.g. testing $D_{\gamma_f}$ `B_align` + $4\times$ norm invariance + combined safeguards under `BF16`/`Mixed-NF4`) and verify via `uv run` confirming **exit `0`**.
3. Report concise final converged numbers & exact A/B gates in your Round 2 reply.
