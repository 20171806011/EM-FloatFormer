# Phase 7 (`p7/`): Multi-Hypothesis C*-Equivariant Evidence Routing & Held-Out Waveform Consistency

- **Branch**: `research/p7-multihypothesis-routing` (base `a4e6684830f68e6dfe715470dafb4ff95972e508`)
- **Status**: Complete prospective held-out study (`N=10` fresh pairs, `5,760` LDPC blocks per condition), `30/30` checkpoint replay verified, full cross-phase regression verified.

## Reproduction Entrypoints (Run From Repository Root)

```powershell
python -B -m unittest discover -s p7/tests -v
python -B -m p7.diag_miscalibration
python -B -m p7.dev_study
python -B -m p7.freeze
python -B -m p7.formal_study
python -B -m p7.replay
python -B -m p7.audit_gauge
python -B -m p7.latency
python -B -m p7.report
```

## Key Artifacts
- [PROTOCOL.md](PROTOCOL.md) — Prospective freeze design and 4 registered Holm-Bonferroni test families
- [P7_REPORT.md](P7_REPORT.md) — Complete Chinese technical report & full quantitative/statistical tables
- [results/freeze.json](results/freeze.json) — Immutable SHA-256 freeze manifest
- [results/summary.json](results/summary.json) — Complete machine-readable results
- [results/p7_figures.png](results/p7_figures.png) — 4-panel quantitative & calibration figure
