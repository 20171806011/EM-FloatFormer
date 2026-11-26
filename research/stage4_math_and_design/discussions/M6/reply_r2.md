**M6 Round 2 is complete.** The Stage 4 plan uses A100-80GB for every GPU job and one fixed seed across four matched arms. Each arm starts with **69 of 384 steps**. The development pilot spends zero efficacy alpha. A pass releases the remaining training, and blind inference is conditional on the frozen checkpoints.

| Stage 4 allocation | USD |
|---|---:|
| CPU causal frontend and dual-cache export | $0.7677 |
| A100 pilot and pilot-dev scoring | $0.8766 |
| Four-arm completion, calibration, and synthetic scoring | $4.1481 |
| Pluto, 2,240 natural frames | $1.2190 |
| CTTC, 1,440 natural packets including held-out `f708_snr10` | $0.6529 |
| Stress tests and slot/text controls | $0.7262 |
| Reserve inside the cap | $1.1094 |
| **Gross cap** | **$9.50** |

The cap leaves **$3.38** against the historical $12.88 credit balance. Launch still requires live credit and zero-net-spend receipts, causal data checks, actual-host Jacobians for both Host-Pure and Hybrid, and an A100 memory and throughput smoke test. [Modal pricing](https://modal.com/pricing) · [Modal spend limits](https://modal.com/docs/guide/budgets)

The pinned `uv` run passed **102 Stage 3D parity checks**, with a maximum discrepancy of **3.56×10⁻¹⁶**. The Round 2 10,000-trial model estimated **59.98% cost savings**, **94.04% working single-endpoint power**, and **0.733% boundary-null promotion** at the 1.667% first-Holm threshold. Production joint power remains unverified. **No cloud jobs ran.**

[Execution handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r2.md>) · [Numerical gates, Stage 4–6 DAG, and LEMMO comparison protocol](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)

