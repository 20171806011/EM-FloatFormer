# EM-FloatFormer Phase 9 (`p9`): Cross-Waveform Decoding Attribution, History-Trust Gating & Fair Attention

- **Branch**: `research/p9-cross-waveform-trust-gating` (from P8-C `ce2ff50`; sibling `p8_manifold` `e76c61e`)
- **Freeze SHA-256**: `3da6678b45391813c76671f14b464d7ff3982ed43aa776a2643a94ae2b0c5feb`

## Reproduction
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m unittest discover -s p9/tests -v
python -B -m unittest discover -s p8c/tests -v
python -B -m unittest discover -s p8/tests -v
python -B -m p9.freeze
python -B -m p9.formal_study --workers 4
python -B -m p9.formal_study --replay-only
python -B -m p9.report
```

See [PROTOCOL.md](PROTOCOL.md) and [P9_REPORT.md](P9_REPORT.md) for complete statistical tables, audit receipts, and figures.
