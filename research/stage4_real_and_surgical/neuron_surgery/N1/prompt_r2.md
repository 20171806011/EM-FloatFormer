# Track N1 (Round 2): Cross-Track Mathematical Audit & Converged SwiGLU+Bridge Reconciliation

This is **Round 2 (Cross-Review & Convergence)** resuming your Track **N1** session (`4d9a1e46-ac0b-4c55-8195-a4ed1d18ce8e`).

## 0. Strict Workspace & Verification Rules
- **Writable Directory**: `research/stage4_real_and_surgical/neuron_surgery/N1/` **ONLY**. Do NOT edit any file outside `N1/`. Do NOT run Modal GPU commands.
- **Python Guardrail**: Always execute via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>`.

## 1. Peer Round-1 Findings to Inspect Read-Only & Audit
Read your three peer tracks' Round-1 summaries (and inspect their `THEOREM_N*.md` / `verify_n*.py` sections as needed):
- `research/stage4_real_and_surgical/neuron_surgery/N2/reply_r1.md` (`THEOREM_N2.md` — GatedDeltaNet `r_hd=2` state highway, `RMSNormGated` ballast, 192-dim NoPE FA intrinsic trigger, Candidate-Orthogonal Bus)
- `research/stage4_real_and_surgical/neuron_surgery/N3/reply_r1.md` (`THEOREM_N3.md` — 9/9 `u_c=None` bug, `in_proj_a` mask bug, false-dormancy 100% collision, `SpecCayley` + Clipped Keyed SwiGLU + AL-Hinge hitting `MixKL = 7.21e-4`)
- `research/stage4_real_and_surgical/neuron_surgery/N4/reply_r1.md` (`THEOREM_N4.md` — NF4 `12.15%` skew & block-64 ripple impossibility, 4-prompt K-FAC squeeze, Layer-10 frozen nonlinear connector bottleneck blocking downstream `CN-SFT` at `13.77%`, `$0.02` 30s pre-flight probe)

## 2. Specific Cross-Track Challenges for Track N1
1. **Reconcile N1 (`RMSNorm` Fermi-Dirac Circuit B + $B_{\text{align}}^{(\gamma_f)}$) with N4's Upstream Bottleneck & N2's `Candidate-Orthogonal Bus`**:
   - N4 proved that if frozen upstream layers (`0..10`) already compressed/bent `(T_1, T_2, omega*h)` nonlinearly, even exact downstream SwiGLU compilation plateaus at `13.77%` (`G1`), whereas N2 uses an early GDN `r_hd=2` + NoPE FA bus ($\mathcal{B}_{\text{EM}} \subset \mathcal{N}_{\text{text}} \cap V_{\text{cand}}^\perp$) with `1.18e-8` error at `L=512`.
   - Reconcile rigorously: if Circuit B sits at `L15..L17` (or `L11..L19`), what exact orthogonal carrier subspace $\mathcal{V}_{\text{in}} \subset \mathcal{V}_{\text{prose}}^\perp \cap (V_{\text{cand}}^{(\gamma_f)})^\perp$ must N2's bus deliver into N1, and does N4's `264`-param orthogonal linear carrier injection satisfy N1's $\gamma \neq \mathbf{1}$ preconditioning + Fermi-Dirac preconditions?
2. **Audit N2's Intrinsic Trigger & N3's Keyed Clipped SwiGLU Under N1's BF16 Fermi-Dirac Instability Finding**:
   - N1 found ungated BF16 Fermi-Dirac amplifies 7-bit orthogonality rounding to `1.352` logit drift, while N2 claims bit-exact `0.0` BF16 `PureKL` from an antipodal NoPE FA detector and N3 uses a stationary $SO(2)$-invariant key $\kappa(h)$ with clipped dead-zone.
   - Audit whether N2's residual gate leak (`<= 1.02e-8` in FP32, `0.0` in BF16 synthetic) OR rare Zipf prose bursts (`N3` Trap) can destabilize Fermi-Dirac $\mathcal{F}_q$ in mixed-turn (`MixKL`) vs pure-prose (`PureKL`) contexts, and derive the combined safeguard (`g_EM` hard invariant + optional NoPE/clipped key + Circuit A/B pre-gain $\kappa_{\text{scratch}}$).
3. **Final Converged Bugfix & Surgery Ledger**:
   - Explicitly validate N3's 3 live bug catches (`u_c=None`, `in_proj_a`, un-gated Schur on `MixKL`) alongside N1's $D_{\gamma_f}$ (`9.49%` tilt) catch against `surgical_adapter.py` and state one exact unified mathematical spec for Qwen3.5-4B (`BF16`) and Qwen3.8-27B (`Mixed-NF4`).

## 3. Deliverables & Re-Verification
1. Append a rigorous **"Round 2 Cross-Track Reconciliation & Unified Consensus"** section to `research/stage4_real_and_surgical/neuron_surgery/N1/THEOREM_N1.md`.
2. If any new cross-track lemma (e.g. N1 Fermi-Dirac + N2 Bus + N3 Keyed Dead-Zone compatibility under BF16) benefits from an assertion, add it to `research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py` and re-run it via `uv run` confirming **exit `0`**.
3. Report concise final verified numbers inside your Round 2 reply.
