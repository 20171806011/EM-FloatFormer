**M6 Round 1 is complete.** The Stage 3D replay puts the largest measured loss after the cache. On G1, Full IQ scored **64.641% / 0.929038 nats**, s17 scored **64.617% / 0.941126**, and Qwen phase scored **23.799% / 1.445229**. The increments were **0.012088 nats** from Full IQ to s17, **0.478026** from s17 to unit-temperature Qwen, and **0.026077** from applying the locked temperature. The archive leaves the separate connector and host information losses unidentified.

On Q2 C4, a fresh reference improved the physical readout from **66.406% to 74.219%** while Qwen phase fell from **28.776% to 26.432%**. The verifier covered all five Q3 cells and seven Q2 arms, with a maximum summary discrepancy of **3.56×10⁻¹⁶**.

- **Tier 0:** Establish the causal cache-risk bounds and enforce rotation, deduplication, masking, and padding invariants.
- **Tier 1:** Check held-out cache–phase interactions and the actual frozen host’s candidate-score Jacobian.
- **Tier 2:** Cap the pilot at **173 of 960 steps** on one seed. Continue to three seeds and independent blind scoring only after the pilot passes.

The **10,000-trial simulation** reduced modeled mean cost from **$17.12 to $6.10**, a **64.36%** reduction, with **93.28%** working-architecture power. Under the specified Gaussian model, the worst-case Type-I error was **4.4695%** with binding futility and **5.0000%** with nonbinding futility. The proposed Stage 4 cap is **$9.50**, leaving **$3.38** against the historical credit balance. Actual Qwen geometry, real-data risk bounds, and scoring throughput remain launch gates. Round 1 spent **$0.00** on cloud compute. [Modal pricing](https://modal.com/pricing)

[Executive memo and M1–M5 questions](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r1.md>) · [Theorems, budget, and Stage 4–6 DAG](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Executed verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Full results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)

