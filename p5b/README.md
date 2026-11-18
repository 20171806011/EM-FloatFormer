# Phase 5-B · Complex-operator unrolled EM receiver

Read [the research report](P5B_REPORT.md) for decoded link performance, operator ablations, one-thread timings, and the R3 covariance probe. The [prospective protocol](PROTOCOL.md) specifies the paired comparisons and the numerical amendment made during development.

## Receiving one burst

```python
import torch
from p5.wavefield_sim import simulate
from p5b.operator_em_transformer import OperatorEMTransformer

torch.set_num_threads(1)
model = OperatorEMTransformer("parity").eval()
state = torch.load("p5b/results/r2/pair01/parity.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
frame = simulate(2, 1, 9, 207001000)
with torch.inference_mode():
    llr, channel = model(frame.obs)
```

Only the observation enters the model. The two returns are positive-for-one bit LLRs on the grid and the estimated propagation diagonal in observation units. Use the payload mask before decoding. The example shows inference on one burst. To recreate the formal frames exactly, use the study runner's batch size of eight.

## Architecture and code map

| File | Responsibility |
|---|---|
| [operator_em_transformer.py](operator_em_transformer.py) | Pilot gauge, physical initialization, tied waveform refits, Hermitian or local propagation, interpolation and application of six complex operator outputs |
| [parity.py](parity.py) | One differentiable 0.75-normalized min-sum check round on complete payload codewords and posterior Gray 16-QAM means |
| [study.py](study.py) | Disjoint development fitting, formal training and scoring, archive comparison, checkpoint restore, kernel probes and one-thread latency |
| [diagnostics.py](diagnostics.py) and [tests](tests/test_p5b_invariants.py) | Gradient directions, anchor checks, gain transformations, erasures, degeneracies, public-code reference, locality, and smooth interpolation |
| [sweep.py](sweep.py) and [replay.py](replay.py) | Prospective source identity, three independent regime workers, and exact replay from saved checkpoints |
| [interventions.py](interventions.py) | Fixed-weight removal of individual learned output groups on the primary evaluation streams |
| [statistics.py](statistics.py) and [plots.py](plots.py) | Count validation, pooling, exhaustive paired sign flips, Holm families and figures |
| [verify_checkout.py](verify_checkout.py) | Archive byte identity and both test suites, P4 publication verification and tracked/staged Git checks |

The live and frozen copies of the same head cancel exactly at initialization. Each learned arm has 1,158 trainable real scalars and another 1,158 fixed reference scalars. Use variants `pure`, `parity`, `local`, `local_parity`, and `no_gauge` for the five trained arms. The two zero-parameter variants are `physics` and `physics_parity`. The model uses CPU complex128 and float64. The terminal LDPC decoder and simulator are inherited from the supplied P4/P5 tree.

## Reproduction from the repository root

Use the archived P5 directory with its checkpoints and calibrated operating points beside this package. The numerical implementation uses Python, PyTorch CPU and NumPy. Figures use Matplotlib in a separate process. The following commands assume the validated inputs and the published read-only P5 digest inventory. On a new worktree without that inventory, `python -B -m p5b.verify_checkout --before` creates it once before the study.

```powershell
python -B -m unittest discover -s p5b/tests -v
python -B -m p5b.diagnostics
python -B -m p5b.study pilot --regime 1
python -B -m p5b.study pilot --regime 2
python -B -m p5b.sweep
python -B -m p5b.replay
python -B -m p5b.diagnostics --checkpoints
python -B -m p5b.verify_frozen
python -B -m p5b.interventions
python -B -m p5b.study latency --regime 1
python -B -m p5b.study latency --regime 2
python -B -m p5b.study latency --regime 3
python -B -m p5b.statistics
python -B -m p5b.plots
python -B -m p5b.report_tables
python -B -m p5b.verify_checkout
```

A focused replay is available with `python -B -m p5b.study replay --regime 2 --pair 1`. The complete replay receipt is written by `p5b.replay`. Run the three latency commands serially with the other experiments finished.

## Reading the evidence

* Each `results/rN/pairNN/record.json` holds the actual training trajectory, seeds, all integer scores, field and payload digests for each held-out batch, and the comparison with archived P5 counts. R3 records also contain the per-layer and per-burst kernel scores and permutations. The five adjacent state dictionaries are final optimizer iterates.
* [raw_by_seed.csv](results/raw_by_seed.csv) and [summary.json](results/summary.json) provide the complete seed-level and pooled outcomes. [r3_kernel.json](results/r3_kernel.json) records the kernel means and signed comparisons. The three `latency_rN.json` files include all eleven timed samples per method and batch.
* [freeze.json](results/freeze.json), [preflight.json](results/preflight.json), [trained_gauge.json](results/trained_gauge.json), [checkpoint_replay.json](results/checkpoint_replay.json) and [checkout_verification.json](results/checkout_verification.json) record the tested identities and outcomes. The [numerical repair](results/antipodal_repair.json) and original development diagnostics are retained under `results/`.
