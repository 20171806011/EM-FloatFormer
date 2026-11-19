# P6 mathematical follow-up — optional inference research

This directory adds a **continuous confidence-gated Doppler sign chart** and an
optional **exact-arithmetic R2 emission simplification** to the published P6
receiver. No historical source, checkpoint or result is replaced.

**Status:** fresh paired inference tests show a modest BLER improvement over the
frozen trust update under two digital pilot-erasure stresses. The new chart does
**not** beat the old hard-chart control in aggregate BLER, and fails one condition
of the inherited strict `1e-12` numerical gauge gate. Keep this opt-in; it is not
a new default or a claim of a completed new phase.

- [Full report / 中文报告](REPORT.md)
- [Mathematical derivation](MATH.md)
- [Prospective-in-session protocol](PROTOCOL.md) and [frozen inputs](FREEZE.json)
- [All fresh raw records](results/fresh), [statistics](results/summary.json)
- [Numerical audit, including failures](results/numerical_audit.json)
- [Independent-agent inference/statistical check](results/independent_audit.json)
- [Same-machine latency](results/latency.json) and [unit/regression tests](results/tests.json)

## Reproduce

From the repository root, use Python 3.12 and the pinned dependencies in
[`p4/requirements.txt`](../p4/requirements.txt). This run used Torch 2.12.1+cpu,
NumPy 1.26.4, SciPy 1.13.1 and Numba 0.60.0 on Linux, float64/complex128.
Use a disposable checkout for regeneration. The audit runners refuse to overwrite
published receipts. Preserve these outputs before running the commands:

```bash
python -m pip install -r p4/requirements.txt
python - <<'PY'
from pathlib import Path
root = Path("p6_followup/results")
backup = Path("p6_followup/replay_receipts")
backup.mkdir(exist_ok=False)
for name in ("independent_audit.json", "numerical_audit.json", "tests.json",
             "p6_regression_numeric.json", "summary.json", "latency.json"):
    (root / name).rename(backup / name)
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m p6_followup.checks
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m p6_followup.statistics
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m p6_followup.independent_audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m p6_followup.numerical_audit
```

The independent audit replays preselected indices 1 and 10 without replacing the
fresh score records. To replay every cluster into memory while keeping evidence:

```python
from p6 import experiment
from p6_followup.study import score_pair

experiment.setup()
for index in range(1, 11):
    replay = score_pair(index, replay=True)
    # Compare replay['frames'] and replay['rows'] with results/fresh/pairXX.json.
```

`python -m p6_followup.study score --workers 3` is the original fresh-scoring
command. It intentionally refuses to overwrite existing raw results. The frozen
manifest is not regenerated for a replay. Run `python -m p6_followup.latency`
separately with other compute jobs stopped; timing is environment dependent.

## Opt-in use

```python
from p6_followup.common import AdaptedReceiver, load_model

model, provenance = load_model(6)  # archived P6 Axial-Parity checkpoint
candidate = AdaptedReceiver(model, mode="smooth", exact_emit=False)
llr, channel = candidate(observation)
```

The adapter scopes temporary bindings and restores them after each forward call.
It uses **process-global hooks** to leave the frozen implementation intact; use
one receiver execution thread per process. Use separate processes for parallel
experiments. `exact_emit=True` separately enables the R2 FFT identity; primary
BLER comparisons did not enable it. R1/R3 emission delegates to the original code.

The evidence is reused-checkpoint inference on new waveforms plus an independent
agent's replay in the same environment. It is neither new training nor external
laboratory reproduction, unseen-channel validation, or low-precision validation.
