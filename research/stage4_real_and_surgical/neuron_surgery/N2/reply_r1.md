Running the Track N2 numerical verification suite (`verify_n2.py`) across all 6 blocks; waiting for completion.
Re-running `verify_n2.py` with exact Qwen3.5 `d_h=256` Full-Attention geometry and Pre-RMSNorm invariance check; waiting for completion.
Running `verify_n2.py` across all 6 blocks; waiting for completion.
Running updated `verify_n2.py` through all 6 verification blocks; waiting for completion.
Running `verify_n2.py` with full GQA group-1 isolation; waiting for completion.
Running `verify_n2.py` with candidate-orthogonal residual bus ($\mathcal{B}_{\text{EM}} \subset \mathcal{N}_{\text{text}} \cap V_{\text{cand}}^\perp$); waiting for completion.
Track **N2 (Round 1)** is complete and machine-verified. Both deliverables have been written strictly inside `research/stage4_real_and_surgical/neuron_surgery/N2/`:

- [THEOREM_N2.md](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N2/THEOREM_N2.md)
- [verify_n2.py](file:///C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_real_and_surgical/neuron_surgery/N2/verify_n2.py) (verified via `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B ...` — **exit `0`**, all 6 verification blocks passing strict assertions in `52.21s`).

### Summary of Proved & Verified Results

1. **GatedDeltaNet Exact State-Highway & Differential-Pair Ballast Linearization (`Theorem N2.1`, `Proposition N2.1`)**:
   - Solved both Qwen3.5 nonlinear obstructions on the recurrent path: post-`conv1d` $\mathrm{SiLU}$ lower-bounding ($\min_z \mathrm{SiLU}(z) \approx -0.2785$) via **antipodal value pairs** ($\mathrm{SiLU}(+x)-\mathrm{SiLU}(-x)=x$) and post-recurrence **`RMSNormGated(o, z)`** radial collapse via a **single constant-norm ballast channel**.
   - On **$r_{\text{hd}} = 2$ value heads** (1 complete key-head group on `4B/9B`, where $G_v = H_v / H_k = 2$), $(T_1, T_2, \omega, h, \cos\theta, \sin\theta)$ is transported across $L \in \{8, 64, 512\}$ prose tokens with **`1.18e-8` max abs error**, **`0.999998945`** retention at $L=512$ (vs **`6.529e-11`** frozen baseline, **`1.53×10¹⁰×`** gain), **`1.110e-16`** KV-cached suffix parity, and **`0.000e+00`** drift on all untouched text heads. By contrast, $r_{\text{hd}} = 1$ exposes **grouped-key sibling contention** (`sibling_head_drift = 0.5220`).
2. **Exact Causal `conv1d` ($K_c = 4$) Boundary Bleed & Dual Cures (`Theorem N2.2`)**:
   - Proved and verified the exact leading-edge Lipschitz bound ($L_{\mathrm{SiLU}} = 1.099839$; naive bleed `1.965 -> 1.408 -> 0.804 -> 0.0`) and 3-token frozen-text trailing bleed (`0.105 -> 0.056 -> 0.034 -> 0.0`).
   - Exploiting the Qwen3.5 asymmetry that **`in_proj_a/b/z` bypass `conv1d`**: **Cure A** (Dirac `conv1d` $c_\tau = \delta_{\tau, 0}$, `112` scalars) yields exact **`0.000e+00`** bleed; **Cure B** (100% frozen `conv1d` + nullspace `in_proj_qkv` + $4\times 4$ causal Toeplitz deconvolution) yields **`0.000e+00`** upstream prose bleed, **`8.882e-16`** deconvolution error, and trailing state bleed **`<= 3.973e-9`**.
3. **Delta-Rule Interference ($I - \beta k k^\top$) & `RMSNormGated` Small-Signal Amplification Trap (`Theorem N2.3`)**:
   - Verified at exact Qwen3.5 $d_k = d_v = 128, L = 512$: orthogonal prose keys yield **`0.000e+00`** error even at $\beta = 0.80$; mild leaky prose writes ($\beta = 10^{-3}$) trigger a **`1,756×` `RMSNormGated` amplification** (`1.81e-4 -> 0.3179`) when `z` is open; enforcing our dual-barrier ($b_{\text{prose}} \le -20$ + $z_{\text{prose}} \le -14$) holds state error to **`9.327e-9`** and pure-text output to **`4.176e-12`**.
4. **Hook-Free Intrinsic Trigger (`g_em` Eliminated) & Partial-RoPE Distance Law (`Theorems N2.4–N2.6`)**:
   - Proved closed-form sub-Gaussian `PureKL` bound $\mathbb{E}[\sigma(\kappa(z-\tau))^2] \le 2\exp(-\eta^2 d / (8\sigma_{\text{nat}}^2))$ for $\kappa \ge \kappa^* = \tau/(2\sigma_{\text{nat}}^2)$ (verified over `200,000` MC trials).
   - Constructed a **bias-free, BOS-free antipodal NoPE Full-Attention detector pair** (2 Q heads sharing 1 GQA KV head inside Qwen3.5's 192-dim NoPE subspace, $\sqrt{d_h} = 16.0$), driving downstream **interleaved `q_proj` output gates** to **`<= 1.02e-8`** on `2,000` antipodal OOD pure-text sequences (`tail_sigma <= 0.05`) and **`1.00000000`** on EM sequences — achieving **bit-exact `0.0` diff in BF16** without Python hooks.
   - Quantified Qwen3.5 Partial-RoPE (`factor=0.25, theta=1e7, d_h=256`): high-freq pair $m=0$ flips sign within 2 KV-cached suffix tokens (`+16.0 -> -6.66 -> -15.84`), whereas pure-NoPE routing ($\rho_{\text{rot}} = 0$) achieves **`0.0000e+00`** distance and KV-suffix variance across $L \in [8, 512]$.
5. **End-to-End Minimal Neuron Graph & Budget (`Theorem N2.7`)**:
   - Identified the **Candidate-Orthogonal Bus Condition** ($\mathcal{B}_{\text{EM}} \subset \mathcal{N}_{\text{text}} \cap V_{\text{cand}}^\perp$) eliminating intermediate residual-to-logit contamination (`1.639 -> 4.11e-7`).
   - Full hook-free 4-layer hybrid verification (`GDN Highway -> NoPE FA Interleaved Gate -> N1 8-Neuron SwiGLU Bilinear Core -> B_align Readout`) hits **`100.0%` OOD QPSK accuracy** (`|\theta| \in [0, \pi]`, including `G1 = ±1.60` and `G5 = ±pi`, `100.0%` Pre-RMSNorm invariant) across $L \in \{8, 64, 512\}$ with simultaneous **`PureKL = 0.000e+00`** (`FP64` and `BF16`).
   - Exact budget on `Qwen3.5-4B`: **`138` active connecting neurons** (`68` GDN + `49` FA + `18` SwiGLU + `3` readout) = **`445,556` dense neuron-row params** (`0.0106%` of host, `2.78×` smaller than `mode1_surgical_96`, `4.90×` below `TIER_P1_MAX_PARAMS`) or **`10,604` subspace-sparse synapses** (`712,820` on `9B`; `890,996` / `901,236` on `27B`).
