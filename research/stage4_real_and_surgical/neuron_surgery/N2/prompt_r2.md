# Track N2 (Round 2): Cross-Track Audit of GDN/NoPE Bridge vs Fermi-Dirac, False-Dormancy, & NF4 Overlay

This is **Round 2 (Cross-Review & Convergence)** resuming your Track **N2** session (`23db3a3c-7df0-4373-8645-95e285560672`).

## 0. Strict Workspace & Verification Rules
- **Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N2/` **ONLY**. Do NOT edit any file outside `N2/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Always execute via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>`.

## 1. Peer Round-1 Findings to Inspect Read-Only & Audit
Read your three peer tracks' Round-1 summaries (and inspect their `THEOREM_N*.md` / `verify_n*.py` as needed):
- `research/stage4_real_and_surgical/neuron_surgery/N1/reply_r1.md` (`THEOREM_N1.md` — Fermi-Dirac $\mathcal{F}_q$ RMSNorm cancellation, `9.49%` final `RMSNorm` $\gamma_f$ readout tilt in `B_align`, `1.352` BF16 Fermi-Dirac orthogonality rounding amplification when ungated)
- `research/stage4_real_and_surgical/neuron_surgery/N3/reply_r1.md` (`THEOREM_N3.md` — 9/9 Cayley `u_c=None` bug, `in_proj_a` mask bug in `surgical_adapter.py#L538`, Zipf polysemantic false-dormancy 100% tail spike, `MixKL = 7.21e-4` via Clipped Keyed SwiGLU + `SpecCayley`)
- `research/stage4_real_and_surgical/neuron_surgery/N4/reply_r1.md` (`THEOREM_N4.md` — Layer-10 frozen nonlinear upstream cap at `13.77%`, `27B Mixed-NF4` `12.15%` skew & block-64 ripple ruling out in-place edits, `$0.02` 30s Layer-10 pre-flight probe)

## 2. Specific Cross-Track Challenges for Track N2
1. **Stress-Test N2's Intrinsic Hook-Free NoPE Trigger Under Heavy-Tailed Natural Prose (`N3` Zipf Trap & `N1` BF16 Fermi-Dirac Leak)**:
   - N1 found ungated BF16 Fermi-Dirac amplifies 7-bit rounding to `1.352` logits; N3 proved bias-free small-sample calibration hits 100% false-dormancy spikes on long-tail Zipf tokens.
   - Audit honestly: under non-Gaussian, anisotropic, heavy-tailed Student-$t$ or Zipf-superposed natural text token embeddings (not just Gaussian isotropic OOD prose!), what is the true false-trigger rate of N2's antipodal NoPE FA detector? Clarify cleanly when structural `g_EM ∈ {0,1}` MUST remain the hard mathematical guarantee (`PureKL = 0.0` unconditionally) vs when N2's intrinsic NoPE gate serves cleanly as an auxiliary hook-free switch and `MixKL` answer-token isolate (`t == t_ans`)!
2. **Audit `surgical_adapter.py` Bugs (`N3` Finding) + Non-Uniform $\gamma$ (`N1` Finding) Across N2's Bridge**:
   - Notice N3 found `in_proj_a` was masked by `m_EM(t)` in `surgical_adapter.py#L538` (freezing $\Delta\alpha_{\text{prose}} \equiv 0$) and N1 found `pinv(M_cand)` missed $D_{\gamma_f}$. Show explicitly in `N2` how N2's dual-barrier (`b_prose <= -20`, `z_prose <= -14`) interacts with `in_proj_a` unmasked vs masked (wait: keeping `in_proj_a` unmasked or dual-gated during `g_EM = 1` prose spans!) and how N2's Candidate-Orthogonal Bus incorporates $D_{\gamma_f}$ in `B_align^{(\gamma_f)}`.
3. **Reconcile N4's `13.77%` Layer-10 Frozen Bottleneck + `27B Mixed-NF4`**:
   - Does N2's `r_hd=2` Shadow GDN Bus + `264`-param orthogonal carrier injection cleanly bypass N4's frozen Layer-10 compression, AND can N2's `GDN` + `NoPE FA` bridge run strictly as an **additive FP32/BF16 hook/delta overlay** without mutating `27B Mixed-NF4` block-64 quantized `conv1d` / `W_down` weights (Cure B + additive GDN heads)?

## 3. Deliverables & Re-Verification
1. Append a rigorous **"Round 2 Cross-Track Reconciliation & Unified Consensus"** section to `research/stage4_real_and_surgical/neuron_surgery/N2/THEOREM_N2.md`.
2. Update `research/stage4_real_and_surgical/neuron_surgery/N2/verify_n2.py` if needed (e.g. adding heavy-tailed prose gate stress or $D_{\gamma_f}$ readout check) and verify via `uv run` that `verify_n2.py` exits **`0`**.
3. Report concise final verified results in your Round 2 reply.
