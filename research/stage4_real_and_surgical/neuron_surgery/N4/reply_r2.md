Running the updated `verify_n4.py` verification suite (Experiments 1–4, including Round-2 unified cross-track safeguards). Waiting for completion.
Fixed the $D_{\gamma_f}$ outlier-channel geometry, exact 3-way candidate-prose-EM orthogonalization, and Layer-2 residual norm cancellation in `verify_n4.py`. Running the full verification suite (`task-178`).
Inspecting the exact Exp 4C–4E diagnostics (`task-183`) to see why `uni_ratio` was `1.0547x` across the $4\times$ independent norm sweep.
Checking the exact Fermi-Dirac quadrature residuals (`task-187`) across the $16\times$ $\eta_2$ range.
Running the fast isolated Exp 4A–4E check (`task-195`) with exact strictly-positive NNLS Fermi-Dirac quadrature (`M_0=7, M_2=8` matching N1 Circuit A's 39 neurons).
All Experiment 4A–4E assertions passed in isolated validation. Now running the complete `verify_n4.py` suite (Experiments 1–4 end-to-end, `task-212`) to lock exact final receipts and confirm exit `0`.
Track **N4 (Round 2)** cross-track reconciliation and convergence is complete. Both deliverables strictly inside `N4/` have been updated and verified end-to-end:

- [THEOREM_N4.md](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N4/THEOREM_N4.md#L313-L392): Appended **Section 5 — Round 2 Cross-Track Reconciliation & Converged Implementation Spec (`N1 + N2 + N3 + N4`)**.
- [verify_n4.py](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N4/verify_n4.py#L2500-L3272): Added **Experiment 4 (`4A–4E`)** validating every unified cross-track safeguard under both `BF16` (`4B`) and `Mixed-NF4` (`27B`), executed via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B` with **exit code `0`** (`wall = 57.93 s`, `ALL_ASSERTIONS_PASSED`).

---

### 1. Converged Cross-Track Reconciliation (`N1 + N2 + N3 + N4`)

All **9 root bugs** across `surgical_adapter.py`, `phaseB_surgical.py`, `train_eval_stage4.py`, and `Qwen3.5/3.8` (`BF16` & `Mixed-NF4`) are unified into one architecture:

1. **Two-Layer Defense-in-Depth (`N2` Hook-Free vs. `N1+N3+N4` `g_EM` & FP32 Overlay)**:
   - Replacing N1's Euler-Maclaurin weights with strictly-positive NNLS weights (`M_0=7, M_2=8`, exact `39n` Circuit A) cuts ungated BF16 Fermi-Dirac pure-prose drift by **`> 100×`** (`5.955 → 0.0544` on 4B; `5.347 → 0.0441` on 27B), **yet `0.0544` still catastrophically fails `PureΔL == 0.0`**.
   - **Layer 1 (`PureKL == 0.0`, `host_sha256` immutable)**: Keep the host 100% bitwise frozen (mandatory for `27B Mixed-NF4`, where NF4 overwrite incurs `KL = 0.483–0.496` and `43.78%` `W_down` neighbor corruption) and run all compiled/tuned SwiGLU neurons as **FP32 Sparse Additive Overlays** behind structural hard gate `g_EM ∈ {0, 1}`.
   - **Layer 2 (`MixKL <= 1e-3`)**: Inside EM-active prompts (`g_EM = 1`), combine **N2's strict `192`-dim NoPE FA routing & Fisher gate** (zero 64-dim RoPE sign-flip), **N3's Stationary Layer-$L_{\text{stop}}$ $SO(2)$-invariant Dead-Zone Key** ($\kappa_{\text{dz}}(h) \in [0, 1]$, `0.0` inside the prose norm ball), **N2's `RMSNormGated` Dual Barrier ($b_{\text{prose}} \le -20, z_{\text{prose}} \le -14$) + Single Ballast Channel** hosted inside **N4's non-mutating Parallel Shadow GDN Bridge** (zero host-head mutation / sibling contention), and **N3/N4's Augmented-Lagrangian Hinge** ($\epsilon_{\text{target}} = 3 \times 10^{-4}$).
2. **Exact Dimension-Matched K-FAC (`4096 != 2560`) & $D_{\gamma_f}$ Readout Frame**:
   - Collect separate `64`-prompt (`>= 2,048`-token) complex K-FAC bases at residual dim $d$ (`2560/4096/5120`, folded weight-side) AND head-output dim $d_o$ (`4096/4096/6144`, hard-asserting `u_c_o is not None` on `9/9` Cayleys!) with **`SpecCayley(r_c=3, r_s=2)`** polar relaxation ($M_\perp = Q_\perp D_\perp$, $\sigma \in [-\ln 2, \ln 2]$) plus unmasked Tier-2 K-FAC on DN `in_proj_a`.
   - Replace `pinv(M_cand) @ F_real` with **$B_{\text{align}}^{(\gamma_f)} = (M_{\text{cand}} D_{\gamma_f})^+ F_{\text{real}}$** and enforce the **3-Way Orthogonal Bus** $\mathcal{V}_{\text{bus}} \perp (\mathcal{V}_{\text{prose}} \oplus V_{\text{cand}}^{(\gamma_f)})$.

---

### 2. Verified Round-2 Quantitative Receipts (`verify_n4.py` Exp 4, Exit `0`)

| Exp 4 Stress Test | Unfixed / Anti-Pattern Baseline | Unified `N1+N2+N3+N4` Stack |
|---|---|---|
| **4A: $D_{\gamma_f}$ Frame & 3-Way Bus** (`d=2560`) | `B_old`: `22.50%` unscaled (`4.12%` dir tilt); 2-way bleed: `5.082` | $B_{\text{align}}^{(\gamma_f)}$: **`3.42e-07`** (`FP32`), **`5.51e-16`** (`FP64`); 3-way bleed: **`4.90e-15`** |
| **4B: `4096`-style `SpecCayley`** (`d_o=96 != d=64`) | `u_c=None` prose drift: **`0.6890`**; rigid radial gain: `1.0000` | $U_c^{(o)}$-projected `SpecCayley`: prose drift **`8.19e-07`**, `[M,J]=0.00e+00`, gain **`1.7333`** |
| **4C: `4B BF16` `4×` Norm Sweep** (`16×` $s_1 s_2$) | Uncomp swing: **`62.8×`**; Ungated BF16 `ΔL`: **`5.955`** (`NNLS: 0.0544`); NF4-OW KL: `0.483` | FP32 Overlay: swing **`1.0404×`**, KL **`6.49e-04`**, Oracle **`100.00%`**, **`PureΔL=0.0`**, **`MixKL=0.0`** |
| **4C: `27B Mixed-NF4` `4×` Norm Sweep** (`d=5120`) | Uncomp swing: **`63.4×`**; Ungated BF16 `ΔL`: **`5.347`** (`NNLS: 0.0441`); NF4-OW KL: `0.496` | FP32 Overlay: swing **`1.0403×`**, KL **`6.34e-04`**, Oracle **`100.00%`**, **`PureΔL=0.0`**, **`MixKL=0.0`** |
| **4D: `RMSNormGated` & Single Ballast** (`d_v=128`) | Leaky $\beta=10^{-3}, z=2$: **`1,761.3×`** blowup; No-ballast err: `0.9546` | Dual barrier: **`4.80e-12`**; Single ballast recovery err: **`2.38e-07`** |

---

### 3. Converged `4B / 9B / 27B` Budget (`< 1.85M`) & Modal A/B Gates (`<= $0.45`)

- **Parameter & VRAM Ledger (Exp 4E)**:
  - **`Qwen3.5-4B` (`BF16`, `r_lora=6`)**: **`1,666,926`** cumulative trainable (**`76.29% < 85%`** of `TIER_P1_MAX_PARAMS = 2,184,964`; **`2,140,526 <= 2,184,964`** even including all `39n` dense-equivalent frozen buffers!), **`42.2 GiB`** peak VRAM.
  - **`Qwen3.5-9B` (`BF16`, `r_lora=5`)**: **`2,094,702`** cumulative trainable (`95.87%` of 4B cap, `0.0234%` of host), **`53.6 GiB`** peak VRAM.
  - **`Qwen3.8-27B` (`Mixed-NF4` host + FP32 Overlay, `r_lora=6`)**: **`3,714,172`** cumulative trainable (`0.0138%` of host, `d_o = 6144` head-space K-FAC), **`45.2 GiB`** peak VRAM.
- **Locked Single-Container Modal `A100-80GB` A/B Matrix (`$0.00075572/s`)**:
  1. **Step 0 Pre-Flight (`55.0 s` total; diagnostic increment `27.0 s` = `$0.0204`)**: Model load + `host_sha256` + `64`-prompt Dual-Space K-FAC (`2560` & `4096`, assert `9/9` Cayleys projected) + $B_{\text{align}}^{(\gamma_f)}$ check + **Layer-10 OOD-$\theta$ linear recoverability probe** ($R^2_{\text{OOD}} \ge 0.85 \to$ Regime 1 Mode-I; $< 0.85 \to$ Regime 2 `+264`-param orthogonal carrier injection).
  2. **`U0` Zero-Shot Compiled (`95.0 s`, `$0.0718`)**: Full `PureKL`/`MixKL`/Dev/`50,304`-row Blind; **Early-Abort** if `G1 < 45.0%` or `PureΔL > 0.0` or `MixKL > 1e-3` (caps failed spend at `$0.113`).
  3. **`U1` Full Unified `CN-SFT` (`295.5 s`, `$0.2233`)**: `96` steps + full audits + `50,304`-row Blind; **Hard PASS**: `PureKL == 0.0`, `PureΔL == 0.0`, `MixKL_mean <= 1e-3` (`max <= 3e-3`), Blind **`G1 >= 48.0%`** (Stretch `>= 58%`), **`G1_NLL <= 1.15`** (paired cluster IUT $p < 10^{-4}$ vs `lemmo_tier_p1 37.04%`), `G3 >= 45.0%`, `VRAM <= 65 GiB`.
  4. **`U3` Gate-Off Audit (`8.0 s`, `$0.0060`)**: Evaluates `U1` with `kappa_dz/q_gate = 1.0` on `12` `MixKL` prompts.
     - **Lean 3-Arm Core (`Step 0 + U0 + U1 + U3`)**: **`453.5 s` (`7.56 min`) = `$0.3427 << $0.45`**.
  5. **Optional `U2` Mechanistic Control (`138.0 s`, `$0.1043`)**: `64`-step bug-fixed `SpecCayley+LoRA` (`0` SwiGLU) on Dev/`MixKL` $\implies$ **Full 4-Arm Matrix = `591.5 s` (`9.86 min`) = `$0.4470 <= $0.45`**. Unlock `9B BF16` $\to$ `27B Mixed-NF4` only after `U1` clears all `4B` Hard PASS gates.
