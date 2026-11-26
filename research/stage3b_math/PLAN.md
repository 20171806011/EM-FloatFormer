# Stage 3B — mathematical optimum before host scaling

Starting point: `research/stage3-third-window-ministral` at `69fbd29acdb4eba7961d56176693786b6181477a`. The published C4 Q3 paired accuracies are 20.57% (Ministral) and 60.16% (same-cache direct decoder). The latter is an achievable comparator; its optimality requires a derivation.

## Primary estimand
For the fixed frozen 8-bit cache, a declared public channel model and timing, a question, and any permitted revealed reference, find the conditional law of its target. State the optimal decisions separately for log loss and top-1 accuracy. The first estimand concerns the simulator and fixed encoder; learning a universally optimal compressor, inferring unknown oscillator parameters, and cross-host generalization require additional evidence.

## Parallel investigations
| Track | Experiment or proof | Completion evidence |
|---|---|---|
| A | Integrate over the shared channel trajectory and quantizer cells for the matched-information posterior. Audit the baseline's exactness. | Derivation, assumptions, counterexample, runnable check |
| B | Determine when known carrier rotation can be represented by a permutation after compression. | Proof or counterexample, sign/time checks |
| C | Derive decision risk for a finite-choice host and separate calibrated probabilities from verbal choices. | Decision/regret identities and finite examples |
| D | Compare phase-aware and phase-blind cache decoders on paired synthetic waves. | Code, seeds, per-method results, uncertainty or scoped surrogate |
| E | Test the omission of answer choices in the Ministral generation prompt. | Explicit-options replay procedure, tests, status of actual host inference |
| F | Specify a matched Qwen replication and isolate training exposure from numerical phase representation. | Run protocol, environment preflight, exact status of any model run |

## Integration
Use the same question, sample access, cache and public parameters in each comparison. Keep pre-quantization compensation, full I/Q and teacher probabilities labeled separately whenever they have different access. Publish positive and negative findings, the scripts and outputs actually produced, the commands and a final status table on this branch. Verify the remote commit and tracked content before declaring the mathematical round complete. State separately which host results remain pending.
