# Phase 6 waveform receiver study

Five learned receivers share their physical solver, trainable capacity, initialization, minibatches, objective and schedule. The confirmatory cohort contains ten independent pairs per regime. The legacy cohort reproduces seeds and test batches from P5 and P5-B.

Axial-Parity improved decoded block errors against the registered classical receiver in all 30 prospective pairs. Against Fixed+Parity the learned handles improved R2 and R3. All three Attention versus Local parity comparisons had Holm p = 1. The formal symmetry audit passed 599/600 conditions. One R2 Axial-Pure degenerate-pilot case exceeded the strict LLR tolerance with zero hard-decision flips. All 250 checkpoints replayed exactly.

[Formal evaluation and exact paired inference](P6_REPORT.md) · [Protocol](PROTOCOL.md) · [Development mathematics](DEV_ITER5_MATH.md)

[Numerical summary](results/summary.json) · [Pair records and weights](results/formal/) · [Freeze manifest](results/freeze.json) · [Checkpoint replay](results/checkpoint_replay.json) · [Trained symmetry](results/trained_gauge.json) · [Handle interventions](results/interventions.json) · [Single-thread latency](results/latency.json) · [Checkout verification](results/checkout_verification.json)

To regenerate the tables from completed receipts, run `python -B -m p6.formal_report` from the repository root. The report checks the frozen source identity before reading formal records.
