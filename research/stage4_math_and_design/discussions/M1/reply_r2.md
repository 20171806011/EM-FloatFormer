**Host-Pure cleared 55% on both extrapolation cells in every seed**, reaching **64.60% on G1** and **61.91% on G3** using frozen-host scores alone. Host-Pure and Physical+Residual both had **zero equivariance defect** with canonical reuse. Their worst ungrouped defects were \(1.49\times10^{-7}\) and \(5.96\times10^{-8}\).

The matched controls resolve three design decisions:

- **Keep quotient sharing.** It improved NLL in all four cells at both coupling settings. Adding explicit multiplicative terms at fixed quotient increased G1 NLL by **0.00860 nats**, so carry the quotient-separate comparator into the Qwen test.
- **Keep physical-only as Arm 3.** Host-Pure used the signed slots: swapping them at fixed text increased G1 NLL by **2.0172 nats**. The bounded residual left top-1 unchanged across all cells and increased G1 NLL by **0.00109 nats** against physical-only.
- **Use the locked cache and gauge contract.** The 16-bit packet preserves S17 bit-exactly. A common receive turn fixes Q1 and Q2 and rolls Q3 receiver sectors. M3’s joint factor keeps Q2 in the TX frame, and CFO uncertainty remains shared across clocks. These contracts passed synthetic checks for both packet formats. Extended-field predictive gains, Qwen margins, and Pluto/CTTC performance remain unmeasured.

The updated verifier exited **0** under the prescribed `uv run` command. [Round 2 handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/reply_r2.md>) · [Theory and contracts](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/THEOREM_M1.md:207>) · [Results and candidate-Jacobian diagnostics](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/results_m1.json>)

