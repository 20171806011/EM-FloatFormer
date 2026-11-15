# EM-FloatFormer P3-A2 — Controlled positional shortcut and residual-floor ablation

Companion to the main [EM-FloatFormer](https://github.com/20171806011/EM-FloatFormer) research repository. Simulated, **receive-only civilian** QPSK/OFDM; no emitters, warfare, or active RF control.

## What is new

- Frozen, public pre-training [P3-A2 protocol](P3_A_MECHANISM_PROTOCOL.md) and a 2×2 factorial intervention on the prior P3 985-parameter delay-domain model.
- Five new seeds (47101–47105), 20 checkpoint trainings, 900 per-seed method × condition rows, 3,200 per-tap diagnostics.
- Independent validation: 11 structural tests, 20 SHA256-verified checkpoints, 240 exact error/bit-count replays, 60 U(1) trained checks, and 20 circular shift checks.
- Negative result retained: while the combined intervention reduces unseen long-delay BER, a simple classical noise-threshold DFT still wins on the two primary held-out channel profiles.

Read [`P3_A_MECHANISM_REPORT.md`](P3_A_MECHANISM_REPORT.md), [`P3_A_MECHANISM_SCOPE_ADDENDUM.md`](P3_A_MECHANISM_SCOPE_ADDENDUM.md), and the raw `p3a2_results` CSV; the source/result public ZIP includes figures and full per-tap data. Model weights are omitted from GitHub but included in the separate full experiment archive.

## Reproduce

Install Python 3.11+ / PyTorch and (for the optional figures) matplotlib. Run from the repository root:

```bash
python -m unittest test_p3_a_mechanism -v
python p3_a_mechanism.py --mode train --seeds 47101 47102 47103 47104 47105 --steps 600 --batch 64 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism.py --mode eval --seeds 47101 47102 47103 47104 47105 --frames 768 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism.py --mode diagnose --seeds 47101 47102 47103 47104 47105 --frames 768 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism_audit.py --out p3a2_results --ckpt p3a2_checkpoints
python p3_a_mechanism_make_report.py
```

This stage is strictly exploratory and does not amend P2-E confirmation or original P3-A results.
