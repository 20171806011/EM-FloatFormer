Delivered [THEOREM_M1.md](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/THEOREM_M1.md>), [verify_m1.py](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/verify_m1.py>), and [results_m1.json](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/results_m1.json>). The prescribed `uv run` verification exited **0**.

| Blind cell | A · Separate tokens | B · Phase-coupled quotient | C · Direct S17 |
|---|---:|---:|---:|
| G1 | 62.40% | **64.62%** | 64.62% |
| G2 | 44.59% | **46.78%** | 46.78% |
| G3 | 59.06% | **62.11%** | 62.11% |
| G4 | 46.10% | **47.75%** | 47.75% |

B had **0.0** equivariance defect with canonical reuse and at most \(5.96\times10^{-8}\) in ungrouped batches. The five task lifts passed exactly. End-to-end training covered Q3 repeat. **The requested A collapse did not reproduce** on the frozen CPU host. The unrotated S17 MAP control reached **15.25%** on G1 and **16.60%** on G3. B’s accuracy parity with the physical oracle leaves the incremental value of the host residual unmeasured.

The proofs establish the task-specific \(C_4\times C_4\) quotient, including Q2’s pilot-frame dependence, and the continuous Brownian phase transport that produces the sector-boundary dip and quarter-turn recovery. The additive rank obstruction applies to **centered additive logits**. Frozen attention can compute cross-token interactions. [Scaled dot-product attention](https://arxiv.org/abs/1706.03762)

The archived G1 physical contrast has **rank 3**, effective rank **2.0015**, and a **0.7058** relative residual after its best rank-one approximation. The Work G replay confirmed that probability averaging lowered NLL from **1.4452 to 1.4113** while accuracy fell from **23.80% to 17.05%**. The theorem gives the NLL guarantee under orbit-invariant evaluation, and a complete-orbit example proves that accuracy can fall even under that condition.

**Round 2 questions**

- **M2** — Which matched Qwen ablation will isolate the physical head, quotient sharing, and coupled tokens?
- **M3** — Can `RefSufficientPool` return the joint timestamped pilot likelihood and common pilot frame, including pilots observed after the target?
- **M4** — What are the authoritative CFO sign, clock, and residual conventions, and can the front end expose shared CFO uncertainty across time edges?

