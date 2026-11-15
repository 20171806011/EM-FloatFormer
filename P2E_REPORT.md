# EM-FloatFormer P2-E — Final fresh-seed replication and audit

**Completed 2026-11-15. Simulated, receive-only civilian QPSK/OFDM ONLY.**

The model and both primary endpoints were frozen before final training: [`P2E_FREEZE.json`](P2E_FREEZE.json), SHA256 935ac73e1cf56a2e9c95f51613a883d4ebf05d15eb672a3ec14dd5b189241342.

## Design and sample size

- 10 **fresh** seeds 2201–2210; both P2 equivariant FIR and P1 circular CNN/log retrained from scratch, identical per-seed training examples, **600 steps** each, batch64, AdamW 0.002, train3 only.
- Per seed × condition: 2048 held-out frames × 32 QPSK carriers × 2 bits = 131,072 bits; per receiver / condition total **1,310,720 bits** across 10 seeds.
- Testing profiles train3, equal4, long6 × 0/10/20 dB. Sample unit for confirmatory test: **training seed**, not individual correlated bits.
- Primary (predeclared) effect: BER(P1 CNN/log) - BER(P2 equivariant FIR), in **percentage points**; positive favors P2. Two-sided exact paired sign-flip test and Holm adjustment across equal4@10dB and long6@10dB.

## 10 dB held-out pooled BER (%)

| Method | Train3 | Unseen equal4 | Unseen long6 |
|---|---:|---:|---:|
| P2 strict common-phase-equivariant complex FIR (2,060 params) | 5.3633 | 6.3468 | 7.3739 |
| P1 CNN/log (38,018 params) | 5.2469 | 6.5742 | 8.3229 |
| Pilot-only LS | 7.9145 | 8.0913 | 8.0070 |
| Fixed 8-tap DFT smoothing (known max-delay support) | 5.2641 | 5.3918 | 5.3361 |
| LMMSE train3 prior | 4.6476 | 14.5509 | 13.5779 |
| LMMSE matched test prior (privileged) | 4.6476 | 4.9016 | 5.0815 |

## Predeclared statistical endpoints

| Profile | Mean P1 minus P2 (BER pp) | 95% descriptive t interval (pp) | Raw exact p | Holm p |
|---|---:|---|---:|---:|
| equal4 | +0.2274 | [+0.1021, +0.3526] | 0.003906 | 0.003906 |
| long6 | +0.9490 | [+0.7396, +1.1584] | 0.001953 | 0.003906 |

Per-seed primary effects (BER percentage points):

- equal4: +0.3548, +0.4150, +0.1305, +0.1236, +0.0641, +0.0412, +0.2998, +0.3693, +0.4799, -0.0046
- long6: +1.0818, +0.8011, +0.9323, +0.7477, +0.7240, +0.5051, +1.2039, +1.5526, +1.0506, +0.8911

**Interpretation:** P2 is consistently better than retrained P1 CNN on long6; equal4 includes one seed with a tiny negative difference. Both predeclared two-sided tests meet the 0.05 cutoff after Holm correction in this synthetic benchmark. However, **DFT-8 is still better on both unseen profiles**, and matched-prior LMMSE (privileged) is better still. No universal superiority, no physical radio validation, and no novelty claim for the architectural components. Common phase-equivariance is an exact architecture property to finite precision, not by itself causal proof of BER improvement.

## Audits

- Recomputed and matched **18** stored error-count and bit-count entries from a saved seed checkpoint, with no training.
- Complex-common-phase test max channel equivariance error 2.4e-07, max equalized-logit difference 1.91e-06, hard bit flips 0.
- Independent Fourier-domain Wiener shrinkage LMMSE vs matrix posterior solve: max difference 3.41e-06.
- All original per-seed integer errors, bit counts, profiles, held-out PRNGs and checkpoint SHA256 are saved; all 10 seed weights are in the separate local full archive, omitted from public GitHub tree.

## CPU-only inference overhead (single measured final-seed run, batch sizes 1/64; excludes RF hardware, FFT chain, transmission and protocol timing)

| Method | Params | Batch 1 median ms | Batch 64 median ms |
|---|---:|---:|---:|
| res_equiv_local | 2060 | 0.1559 | 0.7792 |
| p1_cnn_log | 38018 | 0.4058 | 2.3973 |

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest test_p2_integrity -v
python p2_run.py --mode smoke
python p2_run.py --mode train --names res_equiv_local p1_cnn_log --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --steps 600 --batch 64 --ckpt p2e_checkpoints --out p2e_training
python p2_run.py --mode eval --names res_equiv_local p1_cnn_log --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --frames 2048 --ckpt p2e_checkpoints --out p2e_recomputed
python p2_finalize.py
```

## Limits and next steps

Pilot fully occupies every subcarrier; only one pilot and one payload symbol; static linear channel within pair; idealized perfect synchronization and Gaussian noise. No channel coding, realistic scheduling, dynamic physical impairments, moving vehicles, sample-rate changes, actual antennas, power consumption, or real data. Neural models use the known simulator training distribution and are judged on a handful of synthetic test profiles. Confirmation here means fresh training seeds **within the same implementation and simulator**; not independent laboratory replication. Protocol frozen before testing, but previous exploratory P2 results informed candidate selection. Future P3, if ever approved, requires a stronger multi-symbol, sparse-pilot, receiver-realism benchmark and comparison to classical estimators with matched information. **P2 ends here.**
