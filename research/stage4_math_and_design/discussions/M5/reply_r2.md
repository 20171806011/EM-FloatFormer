**Max EIG beat all four controls on both post-reveal NLL and accuracy** across 12,288 held-out synthetic recordings. All four candidate IDs won within each of the jump, fading, and CFO-alias families, resolving the Round 1 fixed-window tie.

| Policy | NLL (nat) | Accuracy |
|---|---:|---:|
| **Max EIG** | **0.04823** | **98.20%** |
| Random | 0.51722 | 64.96% |
| Nearest | 0.51098 | 66.06% |
| Fixed best on dev | 0.51010 | 66.23% |
| Retrospective fixed oracle | 0.50672 | 66.38% |

Against the dev-locked policy, max EIG gained **0.46187 nat** and **31.97 percentage points**, with simultaneous paired 95% intervals of **[0.45034, 0.47340] nat** and **[30.62, 33.33] points**. All comparisons cleared the paired gate.

The proof gives the exact NLL condition. Adaptation helps when no constant window is optimal almost surely within a positive-measure set of public strata. With unique maximizers, this is equivalent to positive conditional mutual information between the winning action and the utility vector. Accuracy has a separate criterion and was tested directly.

The **16-bit wire format** preserves the s17 prefix and merges M4’s two lowest radial bins to make room for erasure. The pinned CPU replay verified the conversion against M4’s packer and matched M3’s shared pool on 24 selected reveals within \(4.0\times10^{-15}\). The common 4B/9B/27B and LEMMO contract is specified. Host runs, the matched LEMMO adapter, and held-out SDR qualification remain pending.

[Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/reply_r2.md) · [Proof and contract](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/THEOREM_M5.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/verify_m5.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/results_m5.json)

