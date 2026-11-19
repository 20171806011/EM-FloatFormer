# P6 soft-information / full-waveform profile study

This directory is an isolated research extension to the published P6 and
previous mathematical follow-up. It does not modify historical results or
promote a new default receiver. Read [REPORT.md](REPORT.md) for measured
outcomes and limitations, [MATH.md](MATH.md) for the objective and
[PROTOCOL.md](PROTOCOL.md) for the prospective design.

## What is implemented

- Conditional 16-QAM first/second moments, preserving the existing parity
  extrinsic calculation and bit-exact symbol means.
- The full cross-ray expected-Gram correction under diagonal conditional
  symbol covariance for the linear R2 sample-time waveform.
- An optional two-step joint six-coordinate phase refinement of the
  fixed-posterior waveform profile objective with exact Hessian, SPD
  eigenshift, phase cap and sufficient-decrease backtracking.
- Separately tested deterministic-symbol Gaussian evidence and normalized
  bit-probability mixture utilities. No end-to-end eight-mode receiver is
  claimed from those utility tests.

## Reproduce

Run from the repository root with the P6 dependencies. Recorded environment:
Python 3.12.14, Torch 2.12.1+cpu, NumPy 1.26.4, SciPy 1.13.1, Numba 0.60.0,
CPU float64/complex128, one intraop and one interop thread per worker.
The external evaluation decoder uses the existing public LDPC implementation.

```bash
python -m p6_softinfo.independent_audit --output p6_softinfo/results/replay_local.json
```

This command validates the published freeze, all fresh result cells and
six primary comparisons, then replays fresh indices 1 and 10. It writes
its new receipt under this directory's results tree. Use a new output
name on each repeat; published evidence is protected against overwrite.

For a clean reproduction checkout, first move the published results
directory aside (retain it for comparison), then run:

```bash
python -m p6_softinfo.checks --inherited
python -m p6_softinfo.study development
python -m p6_softinfo.study score --workers 3
python -m p6_softinfo.statistics
python -m p6_softinfo.numerical_audit --workers 3
python -m p6_softinfo.independent_audit
python -m p6_softinfo.latency
```

Keep the published `FREEZE.json`: `score` verifies its sources and
checkpoints. `study freeze` is the original prospective registration action;
it refuses to overwrite an existing freeze and is not needed for a replay.
Wait for experiment/audit workers to stop before timing. Scoped old chart
hooks are process-global, so use separate processes, never concurrent
inference threads sharing those hooks. Profile derivatives are an
inference-only local computation, not a claim of end-to-end differentiable
training.

## Evidence boundaries

Fresh evaluation reuses ten archived trained checkpoints with disjoint
simulator seeds. The independent audit is by a separate agent/process in
the same software/hardware environment; external independent retraining is
absent. The study does not establish unseen-channel-family generalization,
R1/R3 nonlinear-PA applicability, low-precision deployment, or calibrated
joint Doppler-mode inference. Stage labels do not establish model superiority.
