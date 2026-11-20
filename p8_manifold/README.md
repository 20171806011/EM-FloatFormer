# `p8_manifold` — Off-Grid Coherent Parallel-Transport Wave Attention & Multi-Chart Manifold Receiver

Self-contained Phase 8 Manifold research track (`research/p8-coherent-transport-manifold`), cleanly isolated from prior phases (`p4..p7`) and separate classical multi-frame branches (`p8/`, `p8c/`).

## Quick Reproduction
```bash
python -B -m unittest discover -s p8_manifold/tests -v
python -B -m p8_manifold.dev_study --steps 40 --workers 3
python -B -m p8_manifold.scaling_ladder --steps 40 --workers 4
python -B -m p8_manifold.freeze
python -B -m p8_manifold.formal_study --workers 4
python -B -m p8_manifold.replay
python -B -m p8_manifold.audit_gauge --workers 4
python -B -m p8_manifold.latency
python -B -m p8_manifold.report
```

See [PROTOCOL.md](PROTOCOL.md) and [P8_MANIFOLD_REPORT.md](P8_MANIFOLD_REPORT.md) for complete protocol, statistics, and audit receipts.
