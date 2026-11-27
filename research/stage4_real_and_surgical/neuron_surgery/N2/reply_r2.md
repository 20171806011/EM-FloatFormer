Running the expanded 9-block `verify_n2.py` suite (Blocks 1–6 Round 1 + Blocks 7–9 Round 2 cross-track stress tests, triple-$\gamma$ & `in_proj_a` audit, and `Mixed-NF4` Layer-10 bypass verification); waiting for completion.
Re-running the 9-block `verify_n2.py` suite after fixing the 4-way codeword detector key-gain normalization; waiting for completion.
Running the full 9-block `verify_n2.py` suite; waiting for completion.
Running all 9 blocks of `verify_n2.py` (including both i.i.d. and anisotropic outlier $\gamma_f$ readout tilts in Block 8B and Mixed-NF4 Layer-10 bypass in Block 9); waiting for completion.
Running the complete 9-block `verify_n2.py` verification suite with N4's Step-0 Layer-10 OOD-$R^2$ Pre-Flight Probe and `Mixed-NF4` additive FP32 overlay; waiting for completion.
`verify_n2.py` is currently executing Block 9 (`L=512` Mixed-NF4 host + Shadow GDN runs with `n_ev=256`); waiting for completion.
Track **N2 (Round 2 Cross-Review & Convergence)** is complete and machine-verified strictly inside `research/stage4_real_and_surgical/neuron_surgery/N2/`:

- **[THEOREM_N2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N2/THEOREM_N2.md)**: Appended **Section 7 ("Round 2 Cross-Track Reconciliation & Unified Consensus")** with Theorems `N2.8`–`N2.9`, heavy-tail / BF16 Fermi-Dirac audit tables, triple non-uniform $\gamma$ & `in_proj_a` reconciliation, and the unified `N1 × N2 × N3 × N4` Phase II-D consensus lock.
- **[verify_n2.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N2/verify_n2.py)**: Extended with **Blocks 7–9** and verified via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B ...` — **`ALL 9 VERIFICATION BLOCKS PASSED STRICT ASSERTIONS in 154.52s (exit 0)`**.

---

### Key Round-2 Verified Cross-Track Findings

1. **Heavy-Tail Breakdown of 1D Intrinsic NoPE Triggers & Mandatory Structural $g_{\text{EM}}$ Split (`Theorem N2.8`, Block 7)**:
   - Stress-testing N2's 1D Antipodal NoPE Detector (`d_h = 256`, `1,000` sequences × `L = 64` = `64,000` tokens/regime, $\sigma_{\text{tail}} = 0.05$) exposes the exact heavy-tail limit: while Gaussian bulk gives `0.00%` FP (`g_max = 5.19e-09`), **Student-$t_{3.0}$ breaks at `0.30%` FP (`max_p = 0.8942, g_max = 1.00`)** and **Zipf single-feature rare spikes ($\alpha=10^{-3}, A_r=3.2$) break at `6.90%` FP (`max_p = 1.3965, g_max = 1.00`)**, confirming N3's Bonferroni phase transition.
   - **BF16 Fermi-Dirac $q=2$ Audit (`d = 2560`, N1 interaction)**: Even with exact orthogonal weights ($a_\eta, a_v \perp \mathcal{V}_{\text{prose}}$, `FP64 = 2.85e-24`, `FP32 = 2.27e-07`), ungated **and pre-FD-gated** BF16 Fermi-Dirac explodes to **`20.2500`** (7-bit rounding occurs on $h_{\text{prose}}$ inside the matmul itself). Post-FD 1D intrinsic gating suppresses Gaussian bulk (`1.05e-07`) **but explodes right back to `20.2500` on any Zipf 1D false trigger**.
   - **4-Way Codeword Upgrade & Role Split**: Replacing 1D $u_{\text{trig}}$ by a 4-way orthogonal codeword conjunction slashes false triggers to **`0.00%` across Gaussian, Student-$t_{4.5}$, Student-$t_{3.0}$, and single-feature Zipf (`g_max <= 4.18e-09`, `3.1×10⁸×` cleaner)** while triggering EM sequences under Student-$t_{4.5}$ prose (`min_EM_pres = 0.7123, min_gate = 0.9998`). **Unified Consensus**: structural out-of-band $g_{\text{EM}} \in \{0, 1\}$ (+ FP32 surgical overlays) **MUST remain the hard guarantee for off-task `PureKL == 0.0` / `PureΔL == 0.0`**, while N2's 4-way NoPE / dual-barrier bridge handles **within-turn `g_EM = 1` mixed-prose (`MixKL <= 1e-3`) & KV-suffix isolation** (and serves as a bounded-KL auxiliary switch only in hookless edge deployments).

2. **`surgical_adapter.py#L538` `in_proj_a` Mask Bug (`N3`) & Triple Non-Uniform $\gamma$ Audit (`N1` + New `RMSNormGated` Finding, Block 8)**:
   - **4-Regime `in_proj_a` Audit (`L = 64`)**:
     - Masked `in_proj_a` (`surgical_adapter.py#L538` bug): retention collapses to **`6.549e-03`** (`99.35%` erased).
     - Naive unmasked `in_proj_a` only: triggers W3's **Leaky-Beta Trap** (`2.24×` state blowup, `prose_drift = 0.1654`).
     - In-place host-head hijacking: triggers N4's **`MixKL = 0.40` trap** (`prose_drift = 0.4387`).
     - **N2+N4 Non-Mutating Parallel 2-Head Shadow GDN Bridge** (unmasked $g \to 1$, dual-barrier `b_prose <= -20`, `z_prose <= -14`): **`ret = 0.999999868`, `payload_err = 7.270e-09`, `host_prose_drift = 0.000e+00`** (with existing host `in_proj_a` LoRAs unmasked strictly under Tier-2 $P_{\perp, C}^{\mathbb{C}}$ + K-FAC damping).
   - **Triple Non-Uniform $\gamma$ Audit (`d = 2560, d_v = 128`)**:
     - Site 1 (`Pre-RMSNorm` $D_{\gamma_{\text{pre}}}^{-1}$): prose leak **`0.4523 -> 1.75e-15`**.
     - Site 2 (**New N2 Finding** — `Qwen3_5RMSNormGated` non-uniform $\gamma_h$ breaks antipodal $\mathrm{SiLU}(+p)-\mathrm{SiLU}(-p)=p$!): even-parity distortion **`2.1241 -> 1.33e-15`** via $D_{\gamma_h}^{-1}$ column pre-division (or $\gamma_h \equiv \mathbf{1}$ on the Shadow GDN Bridge).
     - Site 3 (Final `RMSNorm` $\gamma_f$): $B_{\text{align}}^{(\gamma_f)} = (M_{\text{cand}} D_{\gamma_f})^+ F_{\text{real}}$ fixes frame tilt (**`0.99%` i.i.d. / `12.04%` outlier-aligned $\to$ `4.56e-16 / 9.88e-16`**), and updating N2's Candidate-Orthogonal Bus to $\mathcal{B}_{\text{EM}}^{(\gamma_f)} \subset \mathcal{N}_{\text{text}} \cap (\operatorname{row}(M_{\text{cand}} D_{\gamma_f}))^\perp$ eliminates bus-to-digit leakage (**`0.1810 -> 6.18e-16`**).

3. **Layer-10 Frozen Nonlinear Bypass & Zero-Mutation `27B Mixed-NF4` Additive Overlay (`Theorem N2.9`, Block 9)**:
   - Reproduced N4's frozen narrow-band upstream `GELU` collapse via N4's Step-0 DPI probe (`G1 R²_OOD = 0.515 < 0.85` triggering Regime 2; **`G5 acc = 3.1%`, `R²_OOD = -2.832`**).
   - Wiring N4's `264`-param orthogonal linear carrier directly into **N2's Non-Mutating Parallel 2-Head Shadow GDN Bridge + FP32 Antipodal SwiGLU Overlay** strictly **after** `stop_grad` on a **Dettmers block-64 `Mixed-NF4` host** achieves **`100.0%` OOD QPSK accuracy** on both `G1 (1.60)` and `G5 (pi)` across $L \in \{8, 64, 512\}$ (`fp32_overlay_err = 1.49e-6` to `1.78e-6`), with **bitwise `host_sha256` immutability (`True`)**, **`0` host `conv1d` / NF4 `W_down` mutations**, **`stop_grad_upstream_grad = None`** (Option B1 VRAM savings preserved), and **`g_EM = 0` `pure_diff = 0.0`**.
