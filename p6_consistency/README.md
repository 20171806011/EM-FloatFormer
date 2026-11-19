# P6 observation and reliability consistency

Start with [the research report](REPORT.md), [frozen protocol](PROTOCOL.md),
[mathematical scope](MATH.md), [source/checkpoint freeze](FREEZE.json), and
[complete new-file checksums](MANIFEST_SHA256.json).

This directory is an isolated inference-only follow-up to `p6_softinfo`.
It preserves all historical source, checkpoints and result directories.
The known receive-gain map is privileged diagnostic information, not a blind
erasure detector. The `aware_consistent` name describes its effective-operator
bookkeeping; it is not an assertion of exact global EM or proven superiority.

## Run

Run from the repository root with the archived P6 CPU dependencies available.
The recorded environment is Python 3.12, Torch 2.12.1+cpu, NumPy 1.26.4,
SciPy 1.13.1 and Numba 0.60.0; see the full environment receipts for details.
Matplotlib is needed only for the descriptive figure.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m p6_consistency.checks --attempt independent
python -m p6_consistency.statistics
```

The included `FREEZE.json` and original raw results are immutable evidence.
`development`, `freeze`, `score`, `numerical_audit`, `latency`, and the audit
driver refuse to overwrite their original evidence. To reproduce the original
pipeline in a disposable checkout, archive/remove only that checkout's copied
`p6_consistency/results` and `p6_consistency/FREEZE.json`, then run:

```bash
python -m p6_consistency.study development_core
python -m p6_consistency.study development
python -m p6_consistency.checks --inherited
python -m p6_consistency.study freeze
python -m p6_consistency.study score --workers 3
python -m p6_consistency.statistics
python -m p6_consistency.numerical_audit --workers 3
python -m p6_consistency.independent_audit
# Wait for all other experiment jobs to finish before timing.
python -m p6_consistency.latency
python -m p6_consistency.evidence
```

A newly generated freeze has a new timestamp/hash; match its source/design/
checkpoint contents to the published freeze rather than claiming the same
manifest bytes. The audit uses replay mode for clusters 1 and 10 and never
overwrites original fresh rows. Source-level and numerical checks are not a
replacement for independently rerunning trained-checkpoint experiments.

## Implementation map

- `observation.py`: known receive-gain canonicalization, valid-only gauge and
  pilot fitting, original TX-aperture Fisher reference.
- `weighted.py`: expected weighted full-frequency amplitude normal equations,
  including the learned diagonal residual's covariance cross term.
- `reliability.py`: Gaussian cavity messages, discrete QAM moments, separate
  code-extrinsic priors and channel-only max-log final LLRs.
- `receiver.py`: seven frozen-weight ablations and actual M-step traces.
- `calibration.py`: evaluation-only means/variances and stable uncapped BCE.
- `study.py`, `statistics.py`: paired exposure, raw integer evidence, fixed
  two-test Holm family, descriptive calibration bins.
- `test_*.py`, `checks.py`: formula, mask, message, integration and regression checks.
- `numerical_audit.py`, `independent_audit.py`, `latency.py`: full gain grid,
  independent same-environment arithmetic/replay, and isolated CPU timing.

Inference reads observable fields and, for aware arms, the explicitly supplied
gain map. Labels enter only evaluation. All posterior calculations and physical
solves use CPU float64/complex128; the inherited final LDPC decoder uses float32.
The current dense implementation prioritizes auditable equations, not deployment
efficiency. Learned channel/phase uncertainty is still not marginalized.
