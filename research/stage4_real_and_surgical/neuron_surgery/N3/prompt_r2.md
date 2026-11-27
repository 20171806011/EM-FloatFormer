# Track N3 (Round 2): Cross-Track Reconciliation of Live Codebase Bugs, Keyed SwiGLU Linearity, & 27B NF4 Overlay

This is **Round 2 (Cross-Review & Convergence)** resuming your Track **N3** session (`ae187adf-d975-4509-aaee-b28c185f6180`).

## 0. Strict Workspace & Verification Rules
- **Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N3/` **ONLY**. Do NOT edit any file outside `N3/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Always execute via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>`.

## 1. Peer Round-1 Findings to Inspect Read-Only & Audit
Read your three peer tracks' Round-1 summaries (and inspect their `THEOREM_N*.md` / `verify_n*.py` as needed):
- `research/stage4_real_and_surgical/neuron_surgery/N1/reply_r1.md` (`THEOREM_N1.md` — Fermi-Dirac `RMSNorm` invariant SwiGLU quotient $\mathcal{F}_q$, `9.49%` `B_align` missing $D_{\gamma_f}$ frame tilt, `1.352` BF16 ungated Fermi-Dirac leak)
- `research/stage4_real_and_surgical/neuron_surgery/N2/reply_r1.md` (`THEOREM_N2.md` — `RMSNormGated` 1,756x amplification trap + constant-norm ballast, `r_hd=2` GDN value group, 192-dim NoPE FA intrinsic trigger + Candidate-Orthogonal Bus)
- `research/stage4_real_and_surgical/neuron_surgery/N4/reply_r1.md` (`THEOREM_N4.md` — `27B Mixed-NF4` `12.15%` asymmetry & `25–48%` block-64 drift ruling out permanent weight overwrite, `15.40x` folded `(AP)x` nullspace speedup, Layer-10 frozen nonlinear cap `13.77%`, `$0.02` pre-flight check)

## 2. Specific Cross-Track Challenges for Track N3
1. **Cross-Reconcile Codebase Diagnostics & Fix Priority Across N1, N3, N4**:
   - N3 discovered **4 critical bugs** in existing W1–W4 code (`9/9 Cayleys` had `u_c=None` since `2560 != 4096`; 4-prompt K-FAC `k_c=256` rank deficiency; `in_proj_a` accidentally masked by `m_EM`; un-gated `SchurIrrepReadout` leaking on `MixKL`); N1 discovered **`B_align` omitted final RMSNorm $D_{\gamma_f}$** (`9.49%` tilt); N2 discovered **`RMSNormGated(o, z)` small-signal explosion** + grouped-key sibling contention (`r_hd=1` vs `r_hd=2`); N4 proved **`15.40x` speed-up via folded `(AP)x`** and flagged **frozen Layer-10 nonlinearity**.
   - Check N3's `in_proj_a` diagnosis against N2: wait — in `surgical_adapter.py#L538`, masking `in_proj_a` by `m_EM` stopped DeltaNet adapters from forgetting **prose tokens**, whereas N2 warns open/leaky prose writes in `in_proj_b`/`z` cause a `1,756x` `RMSNormGated` blowup! Reconcile cleanly: should `in_proj_a` stay unmasked with AL-Hinge/K-FAC nullspace, OR dual-tier gated (`EM` write + answer-pos readout)?
2. **Audit N3's `Dead-Zone Keyed Clipped SwiGLU` Compatibility with N1 (`RMSNorm` + $\mathcal{F}_q$), N2 (NoPE Bus), & N4 (`27B NF4`)**:
   - When N3 attenuates SwiGLU using stationary key $\kappa(h) \in [0, 1]$, does variable $0 < \kappa(h) < 1$ on weak/noisy EM inputs distort N1's bilinear cross-correlation or pre-MLP RMSNorm normalization, unless saturated cleanly onto $\{0, 1\}$ (hard plateau for $\|k\| \ge \tau_{\text{high}}$)?
   - Reconcile adaptive K-FAC rank (`N3`: `k_c_d=12, k_c_o=18, k_c_m=10` vs `N4`: `32-64` prompts): prove what minimum calibration prompt count $N_{\text{cal}}$ eliminates finite-sample K-FAC degeneracy!
3. **Final Converged Architecture Budget (`Qwen3.5-4B` & `Qwen3.8-27B Mixed-NF4`)**:
   - Confirm exact parameter count & `PureKL=0.0, MixKL < 1e-3` guarantee when combining N3's `SpecCayley + AL-Hinge + Tier-2 K-FAC + Clipped Keyed SwiGLU` with N1's `D_{\gamma_f}` / Fermi-Dirac, N2's `r_hd=2` NoPE gate, and N4's FP32 sparse additive overlay for NF4.

## 3. Deliverables & Re-Verification
1. Append a rigorous **"Round 2 Cross-Track Reconciliation & Unified Consensus"** section to `research/stage4_real_and_surgical/neuron_surgery/N3/THEOREM_N3.md`.
2. Re-run (or update & verify) `research/stage4_real_and_surgical/neuron_surgery/N3/verify_n3.py` via `uv run` confirming **exit `0`**.
3. Report concise final verified results in your Round 2 reply.
