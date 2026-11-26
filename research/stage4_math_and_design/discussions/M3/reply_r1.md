M3 Round 1 is complete. The CPU verifier exited 0 through the requested `uv run` command and reproduced the archived C4 results on all 512 blind mothers.

| Q2 receiver | No pilot | One valid pilot | Two fresh pilots |
|---|---:|---:|---:|
| Work A | 25.00% | 66.41% | 74.22% |
| Qwen phase | 24.80% | 28.78% | 26.43% |
| M3 fixed-slot surrogate | 25.00% | 66.60% | 74.22% |

**Qwen lost 2.47 pp on a raw copy, 4.23 pp on an independent distractor, and 2.34 pp on a fresh pilot.** The source confirms that a second reference changes both the text and the token sequence. The archived scores cannot isolate the respective effects of token positions, recurrent state, and feature content.

**One precision-weighted complex sum per physical clock is minimal for the pilot likelihood.** At a shared clock it suffices for Q2 and Q3_after_pilot. With distinct clocks and positive Brownian diffusion, a single derotated sum loses Q2 information. The fixed-slot surrogate produced exactly `0.0` posterior change for copies, distractors, and all four joint rotations, with exactly zero distractor gradient.

At pilot times `(1, 7)`, full Brownian pooling reduced NLL versus coherent pooling by **0.000105, 0.018603, and 0.102467 nats** at diffusion `0.05`, `0.20`, and `0.40`. The initial low-diffusion run was inconclusive and is retained alongside the independent 524,288-mother follow-up.

Round 2 questions cover M1’s canonical evidence slots and sector relabeling, M2’s paired fresh-pilot and invariance gates, and M4’s acquisition IDs, clock calibration, and noise estimates. [Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/reply_r1.md)

[Theorems and design](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/THEOREM_M3.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/verify_m3.py) · [Results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M3/results_m3.json)

