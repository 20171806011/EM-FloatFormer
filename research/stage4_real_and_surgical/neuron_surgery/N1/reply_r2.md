Implemented Certificate 5 in [verify_n1.py](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py) covering the finite-$M$ Fermi-Dirac half-line amplification audit, 6-regime cross-track safety matrix, $D_{\gamma_f}$ bus upgrade proof, and end-to-end 4-track `BF16` / `Mixed-NF4` pipeline; waiting for the `uv run` verification task to complete.
Updated [verify_n1.py](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py) to isolate the BF16 activation-quantization orthogonality floor (`1.060` logit explosion even with an ungated FP32 overlay when the host residual stream is BF16) vs. the combined FP32 overlay + N3 stationary dead-zone key safeguard, and re-launched the `uv run` verification suite.
# Track N1 (Round 2): Cross-Track Mathematical Audit & Converged SwiGLU+Bridge Reconciliation

All Round 2 cross-track reconciliations, new proofs (**Lemma N1.5**, **Theorem N1.5B**, **Theorem N1.6**), and machine-verified numerical proofs (**Certificate 5**) are appended to [THEOREM_N1.md](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N1/THEOREM_N1.md#L325-L450) and verified in [verify_n1.py](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py#L1490-L2172) (`uv run --python 3.12 ...` **exit code `0`** in **`9.10 s`**, full JSON receipt in [results_n1.json](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N1/results_n1.json#L703-L876)). Zero files outside `N1/` were modified.

---

## 1. Upstream Bottleneck & Bus-to-SwiGLU Reconciliation (`N1 × N2 × N4`, Theorem N1.6)

- **Mandatory $D_{\gamma_f}$ Upgrade to N2's `Candidate-Orthogonal Bus`**: N2's Round-1 specification $\mathcal{B}_{\text{EM}} \subset \mathcal{N}_{\text{text}} \cap (\operatorname{row} M_{\text{cand}})^\perp$ omits `Qwen3.5`'s non-uniform final `RMSNorm` gain $D_{\gamma_f}$. At `d = 2560` (Cert. 5B.1), an uncompensated 16D bus leaks **`0.1322` max spurious digit logits** (`KL_mean = 9.14e-4`, burning `91.4%` of the `1e-3` `MixKL` budget from transport alone). Upgrading BOTH N2's bus and N4's `264`-param Regime-2 carrier to
  $$\mathcal{V}_{\text{in}} \;\subset\; \mathcal{V}_{\text{prose}}^\perp \;\cap\; \big(\operatorname{row}(M_{\text{cand}} D_{\gamma_f})\big)^\perp, \qquad \dim(\mathcal{V}_{\text{in}}) = d_{\text{carrier}} + 1$$
  slashes bus-to-logit contamination to **`4.22e-15`** (`KL_mean = 1.47e-17`, **`3.13 × 10^{13}×` suppression**).
- **Mandatory $+1$ Anchor Slot ($c_0 e_{\text{anc}}$) & Per-Layer $D_{\gamma^{(\ell)}}^{-1}$ Division of Labor**: N1's Fermi-Dirac `RMSNorm` quotient needs one constant-amplitude anchor slot $c_0 e_{\text{anc}} \in \mathcal{V}_{\text{in}}$ alongside the `6` physical coordinates (`7` active slots total).
  - **N2's `r_hd=2` Shadow GDN Ballast Bus** (`b_0 = 8.0`) supplies $c_0 e_{\text{anc}}$ directly from its constant-norm ballast channel across $L_{\text{gap}} = 64$ prose tokens at **`2.79e-9`** max rel. err (`G1`) / **`2.53e-9`** (`G5`) with **zero extra heads**.
  - **N4's `264`-param Regime-2 Carrier** (`16 × 16 + 8`, activated iff N4's `$0.02` 30s Layer-10 OOD-$\theta$ probe reports $R^2_{\text{OOD}} < 0.85$) fits all `7` slots cleanly inside its `16D` orthogonal basis at **zero extra parameters**.
  - Neither bus nor carrier pre-inverts layer-specific $\gamma^{(\ell)}$; each N1 overlay layer $\ell$ applies exact $D_{\gamma^{(\ell)}}^{-1}$ locally in FP32.
- **End-to-End 4-Track Pipeline Closes N4's Zero-Shot `KL_Bayes` Gap (`0.8009 → 5.44e-4 nats`)**: On real locked Stage 3D `s17` (`d = 2560`, `L_gap = 64` Shadow GDN bus $\to$ N1 `K = 64` Circuit B FP32 Overlay + N3 Stationary Key $\to$ $B_{\text{align}}^{(\gamma_f)}$):

| Stage 3D Regime | Rows | Exact Bayes Oracle (`Acc / NLL`) | `BF16 Host + FP32 Overlay` (`Acc / Agree / NLL / KL_Bayes`) | `Mixed-NF4 Host + FP32 Overlay` (`Acc / Agree / NLL / KL_Bayes`) |
|---|---:|---|---|---|
| **`G1_main_extrap_both`** ($|\theta|=1.60$) | `1,024` | `64.36% / 0.9394` | **`64.36%` / `100.0%` / `0.9409` / `5.44e-4`** | **`64.36%` / `100.0%` / `0.9430` / `7.83e-4`** |
| **`G5_half_turn_pi`** ($|\theta|=\pi$) | `512` | `61.33% / 1.0409` | **`61.33%` / `100.0%` / `1.0414` / `5.55e-4`** | **`61.33%` / `100.0%` / `1.0402` / `8.59e-4`** |

---

## 2. Safety Audit: Asymmetric Half-Line Gain Law & BF16 Activation Floor (`Lemma N1.5` & `Theorem N1.5B`)

- **Root Cause of Fermi-Dirac Instability (`Lemma N1.5`)**: On the negative half-line $\eta = -\delta < 0$ (50% of zero-mean rounding or Zipf bursts), $-t_m \eta = +t_m \delta > 0$ enters the unbounded linear branch of $\operatorname{SiLU}$, obeying $G_q^{(M)} \delta |v| \le |\widehat{\mathcal{F}}_q^{(M)}(-\delta, v)| \le 2 G_q^{(M)} \delta |v|$ with $G_q^{(M)} = \frac{1}{2}\sum_m w_m t_m$. Measured gains: **`G2_circuitA = 122,383.27`**, **`G2_circuitB = 138,820.15`**, **`G3_anc = 293,057.38`** (`4.35×` neg/pos asymmetry at $\delta = 0.02$).
- **Key Round 2 Finding (`Theorem N1.5B` — BF16 Activation Floor)**: Even when N2's trigger outputs **exact `0.0`** ($h_{\text{EM}} \equiv 0$) **AND** all surgical weights run in **N4's FP32 Sparse Additive Overlay**, the host's own BF16 residual rounding injects `0.002577` per-slot RMS (`0.01986` max 7D norm) isotropically into $\mathcal{V}_{\text{in}}$, which still explodes ungated Fermi-Dirac to **`1.0602` max logit drift (`KL_max = 0.2573`)**! Thus N4's FP32 overlay is **necessary** (eliminating BF16 weight rounding and 27B NF4's `12.15%` skew / block-64 ripple) but **not sufficient alone** for ungated Fermi-Dirac on a BF16 residual stream, and N2's `0.0` BF16 trigger claim holds **only** for pure bilinear SwiGLU ($G_{\text{bilin}} = 1$).
- **Verified 8-Regime Safety Matrix (`Cert. 5A`, `d = 2560`, `B = 512`, Circuit A)**:

| Regime | Configuration | `max \|Δlogit\|` | `KL_mean` | `KL_max` | Verdict |
|---|---|---:|---:|---:|---|
| **R1** | FP32 stream + N2 leak (`1.02e-8`) $\to$ FP32 Overlay | `1.78e-10` | `~0` | `3.33e-16` | **SAFE** |
| **R2** | BF16 (`h_EM=0`) $\to$ Ungated Native BF16 | **`1.9659`** | `3.25e-2` | `0.6512` | **EXPLODES** |
| **R3a** | BF16 (`h_EM=0`) $\to$ **Ungated FP32 Overlay** *(Thm N1.5B!)* | **`1.0602`** | `3.69e-3` | `0.2573` | **EXPLODES (BF16 act floor)** |
| **R3b** | BF16 (`h_EM=0`) $\to$ **FP32 Overlay + N3 Dead-Zone Key** | **`0.0`** | **`0.0`** | **`0.0`** | **BITWISE ZERO** |
| **R3c** | FP32 (`h_EM=0`) $\to$ Ungated FP32 Overlay | `1.43e-10` | `1.40e-18` | `2.77e-16` | **SAFE** |
| **R4** | Mixed-turn (`g_EM=1`) + Student-$t_{4.5}$ Zipf $\to$ Ungated FP32 | **`0.9778`** | `4.37e-3` | `0.1779` | **FAILS `MixKL <= 1e-3`** |
| **R5** | Mixed-turn (`g_EM=1`) + Student-$t_{4.5}$ Zipf $\to$ **Combined Safeguard** | **`0.0`** | **`0.0`** | **`0.0`** | **PASS (`100%` bulk zero; `key_min = 1.0` on EM)** |
| **R6** | Pure prose + Hard `g_EM = 0` | **`0.0`** | **`0.0`** | **`0.0`** | **BITWISE EXACT (`PureKL = 0.0`)** |

**Converged 4-Layer Safeguard**: Hard `g_EM` $\in \{0,1\}$ $\times$ **FP32 Sparse Additive Overlay** $\times$ **N3 Stationary $SO(2)$-Invariant Dead-Zone Key $\eta_{0.25, 0.65}(\|U_{\text{in}}^\top h_t\|_2)$** (applied multiplicatively on the Fermi-Dirac output/up branch; optionally composed with N2's NoPE detector / N4's Fisher query gate) $\times$ **N1 scratch pre-gains** ($\kappa_{\text{scratch}} = 8.0, \kappa_2 = 4.0$).

---

## 3. Validated 5-Bug Forensic Audit & Unified `4B / 27B` Spec

All 5 live codebase bugs are confirmed against source lines:
1. **`9/9` Cayley `u_c = None`** ([surgical_adapter.py:378](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/surgical_adapter.py#L378), [L410](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/surgical_adapter.py#L410)) $\to$ collect `4096`-dim $U_c^{(o)}$, fold $A P_\perp$ onto weights (`15.4×` speedup), upgrade to Bounded-Spectrum **`SpecCayley`** ($\sigma \in [0.5, 2.0]$).
2. **`in_proj_a` masked by `m_EM(t)`** ([surgical_adapter.py:530-542](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/surgical_adapter.py#L530-L542)) $\to$ unmask into Tier-2 + N2/N4 `r_hd=2` Shadow GDN Ballast Bus.
3. **Un-gated `SchurIrrepReadout` on `MixKL`** ([surgical_adapter.py:546-559](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/surgical_adapter.py#L546-L559)) $\to$ gate by N3 Stationary Dead-Zone Key + N4 Fisher query gate + Primal-Dual AL-Hinge `MixKL` ($\epsilon_{\text{target}} = 3\times 10^{-4}$).
4. **Missing $D_{\gamma_f}$ in `calibrate_b_align_from_host`** ([surgical_adapter.py:291](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/surgical_adapter.py#L291), `9.49%` tilt) $\to$ upgrade to $B_{\text{align}}^{(\gamma_f)} = (M_{\text{cand}} D_{\gamma_f})^+ F_{\text{real}}$ (`5.55e-16` err).
5. **4-Prompt Rank-Deficient `k_c = 256` K-FAC & Double-Autograd VRAM** ([train_eval_stage4.py:65-70](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/train_eval_stage4.py#L65-L70)) $\to$ `>= 32–128` prompts (`>= 1024` tokens), energy-adaptive $k_c \le 96$, sequential CF backward (`~42.0 GiB` VRAM).

**Unified Parameter Ledger (Cert. 5C — Strictly Additive, Bitwise Immutable Host)**:
- **`Qwen3.5-4B` (`BF16` host + FP32 Overlay)**: **`1,032,592`** adapter trainable (`1,520,992` incl. Phase-A connector = **`69.61%`** of `TIER_P1_MAX_PARAMS = 2,184,964`, `0.0362%` of host); **`1,831,116`** (`83.81%`) incl. Circuit A (`K=39`) + N2 buffers, or **`2,023,116`** (**`92.59%` PASS**) incl. ALL Circuit B (`K=64`) + N2 frozen compiled FP32 buffers (`0.0481%` of host; `~42.0 GiB` VRAM, `$0.29` core / `$0.43` 3-arm Modal A100 probe).
- **`Qwen3.8-27B` (`Mixed-NF4` host + FP32 Overlay)**: **`2,607,784`** adapter trainable (`3,431,544` incl. Phase-A connector = **`0.0128%`** of host); **`4,432,868`** (`0.0165%` of host) incl. ALL Circuit B (`K=64`) + N2 frozen compiled FP32 buffers (`~45.5 GiB` VRAM with grad checkpointing).
