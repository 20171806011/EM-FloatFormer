M3 Round 2 is complete. The CPU verifier exited **0 in 32.28 seconds**. All repository edits are inside M3.

- **No double-counting.** A consumed acquisition adds zero likelihood. If its noise is correlated with a fresh pilot, its retained value still conditions the new likelihood. The Schur update matched the full Gaussian calculation within \(8.0\times10^{-15}\).
- **Byte-exact invariance.** Copies, distractors, padding, and input order produced identical fixed-slot tensors and deterministic prefix embeddings. Maximum KL and TV were both `0.0`. The v2 adapter matched `q2_reader` within \(8.9\times10^{-16}\) on all 512 C4 mothers.
- **Shared CFO and Stage 6.** On 4,096 paired synthetic draws, the full two-clock model scored **0.147654 NLL**, versus **0.378354** for the optimal posterior given the pooled sum and **0.593499** for coherent pooling. The Stage 6 fixture selected the **0.433721-nat EIG** probe, and its pre-reveal forecast matched the one-read update to floating-point precision.

The dev-locked Qwen replay provided no evidence that the offline residual improved `q2_reader`. The prefix and hardware-role checks used CPU fixtures. Actual Round 2 Qwen execution and measured Pluto/CTTC noise covariance remain integration work.

[Round 2 summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/reply_r2.md) · [Theorems and schema](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/THEOREM_M3.md) · [Implementation](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/ref_pool.py) · [Verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/verify_m3.py) · [Results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/results_m3.json)

