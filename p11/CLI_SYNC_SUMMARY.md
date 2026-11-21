# P11 Cross-CLI Research Synchronization Summary

- **Baseline Commit**: `9e9d93bcbf6345b8645314b223cd954b2f31e63a`
- **Freeze SHA-256**: `d5f5da3c09c2dfbb53da617f0120acb2e67e7ecd5c7aac3e2240f09949496d15`
- **Mode**: `all`

## Verbatim Verdicts (`支持` / `反证` / `证据不足`)

| Axis | Claim / Contrast | Verdict |
| :--- | :--- | :---: |
| Primary | `chained_vs_p9_jt` | **支持** |
| Primary | `chained_vs_payload_only` | **支持** |
| Primary | `engineering_noninferiority` | **支持** |
| Secondary | `transport_attn_vs_abs_no_transport` | **支持** |
| Secondary | `transport_attn_vs_local` | **证据不足** |
| Secondary | `transport_attn_vs_no_history` | **支持** |

## Primary Axis Highlights
- `p11_chained_wavefront` 3-ID BLER: `12.88%`
- `p11_payload_only` 3-ID BLER: `46.76%`
- `p9_jeffreys_student_t` 3-ID BLER: `49.96%`

## Secondary Axis Highlights
- `em10_transport_attn` 4-ID BLER: `66.02%` (OOD: `81.47%`)
- `em10_abs_no_transport` 4-ID BLER: `66.46%`
- `em10_local` 4-ID BLER: `66.02%`
- `em10_no_history` 4-ID BLER: `77.92%`

---

## Pre-Experiment Freeze & Codespace Provenance
- **P10 Baseline (`9e9d93b`)**: `9e9d93bcbf6345b8645314b223cd954b2f31e63a`
- **Freeze Commit C1 (Code/Checkpoints)**: `cec54bfd7835f0b586cde2863174830c2d619838`
- **Freeze Commit C2 (`freeze_manifest.json`)**: `cb6b2bbb33608fe26a50ddb6e18555cb35df6dd4`
- **Manifest SHA-256**: `d5f5da3c09c2dfbb53da617f0120acb2e67e7ecd5c7aac3e2240f09949496d15`
- **Formal Codespace**: `em-floatformer-cloud-jjg7qww5jvr5fpv96` (`AMD EPYC 7763`, `Python 3.12.3`, `numpy 2.5.5`, `scipy 1.18.1`, `torch 2.15.0+cpu`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.use_deterministic_algorithms(True)`, `float64/complex128`)
- **Exact Replay (`REPLAY_CLUSTERS = (1, 16)`)**: `max_float_diff = 0.0`, `mismatch_count = 0` on both Primary (`192/192` ID links) and Secondary (`192/192` links: `128 ID + 64 OOD`), `seed_disjointness.disjoint = True` across all 24 historical intervals (`3,840` unique formal seeds).
- **Primary OOD / Sanity Unvarnished Results** (`256` links / `4,096` blocks per regime):
  - `ood_strong_shift`: `p11_chained_wavefront = 10.21%` (`fold0=11.86%`, `fold1=6.56%`, `clean_nmse=0.0530`, `cal=1.09x`, `fa=13/2132=0.61%`) vs `p11_payload_only = 69.46%`, `p9_jeffreys_student_t = 69.78%`, `dense/always_clear = 52.05%`.
  - `ood_midframe_shift`: `p11_chained_wavefront = 11.11%` (`fold0=12.64%`, `fold1=7.73%`, `clean_nmse=0.0615`, `cal=1.14x`, `fa=25/2060=1.21%`) vs `p11_payload_only = 80.42%`, `p9_jeffreys_student_t = 67.50%`, `dense/always_clear = 57.06%`.
  - `donor_mismatch` (sanity): `p11_chained_wavefront = 20.07%` (`fold0=24.43%`, `fold1=10.47%`, `clean_nmse=0.1023`, `cal=0.86x`, `fa=14/1800=0.78%`) vs `p9_always_inherit = 93.53%`, `p9_jeffreys_student_t = 66.14%`.
  - **Frozen Bank Disclosure**: `p11/payload_rebuild.py` registers `ONSET_ROWS = (0, 6)` and `DELTAS_OOD = (0.085..0.135)` inside the 144-hypothesis shift bank, so Primary OOD tests multi-hypothesis selection among registered candidate onsets/deltas under continuous shift draws rather than unseen-onset extrapolation.
- **Secondary OOD Contrasts** (`9,216` blocks/arm): `ood_transport_attn_vs_abs_no_transport` (`+0.25 pp`, `p_holm=0.148071`, **证据不足**); `ood_transport_attn_vs_local` (`-0.31 pp`, `p_holm=0.148071`, **证据不足**); `ood_transport_attn_vs_no_history` (`+7.40 pp`, `p_holm=0.000092`, **支持**). `oracle_true_channel`: `5.35%` ID / `11.91%` OOD.

## Codex CLI (`gpt-6.2-astra`, `ultra`) Mathematical Verification & Critique Summary
- **Exact Reproduction**: Recomputed all 8 exact `65,536`-sign-pattern Holm tests, `continuous` NI (`mean_diff = -0.0854 pp`, `upper_95 = +0.6218 pp < +1.00 pp`, `df=15`), and delivered-bit decomposition (`+121.500` from reclaimed pilot rows `(0, 9)` + `+0.875` from 7 core blocks = `+122.375 bit/frame`, `+13.00%`).
- **`full_reset = 20.00%` Mechanism**: `84.31%` (`1,381/1,638`) of `full_reset` errors occur in Wave-1-correct folds (`Fold 0: 1,141/5,005 = 22.80%` vs `Fold 1: 240/2,515 = 9.54%`; `54/58` absent units are also in Fold 0). Under whole-fold isolation, Fold 0 (`11` targets) has only two pre-anchor helpers `(9,)` and `(0,)` (row-center time-leverage SS `1.125` at Wave 1, `136.5` max vs Fold 1's `4.5` and `665.0`), leaving rows `11..18` outside coherent coverage; meanwhile `AUX` rows cannot be converted post-hoc into virtual pilots because they already shaped the `160/5832` proposal support.
- **Secondary Transport & Ceiling**: `em10_transport_attn` saves `82` ID blocks vs `em10_abs_no_transport` (`+0.44 pp`, **支持**) and `1` ID block vs `em10_local` (`12,168` vs `12,169`, **证据不足**), because `em10_local` itself performs 3-neighbor signed U(1) phase transport. The ~`61 pp` ID / `70 pp` OOD gap to `oracle_true_channel` reflects Frame-A 4-pilot comb near-aliasing (`Delta_nu = 1/14.55`, overlap `0.969815`), single roving Frame-B anchor, and the `w*mu_inherit + (1-w)*mu_clear` mixture constraint without Frame-B payload refit (`clean_nmse = 0.5720` ID / `0.6836` OOD).

## Antigravity CLI (`gemini-4-neon-xhigh`) Cross-CLI Synchronization Response
- **Independent Sanity Check (PASS)**: Verified C1/C2/manifest hashes (`0` mismatches across 27 files & 12 checkpoints), `3,840` unique formal seeds (`disjoint = True` across all 24 historical intervals), `float_tol = 0.0` replay parity, and independently recomputed all 8 exact-Fraction Holm `p`-values, NI bounds, Wave-1 stratifications, Primary OOD aggregations, and Secondary per-regime tables (`em10_local` outperforms global transport on 3/4 shift/burst regimes and `classical_per_ray_gate` `87.80%` leads `offgrid_ood_five_ray_fast`). Flagged minor methodological caveats: Primary OOD/sanity excluded from `p11_audit.json`/replay/`p11.report` summary, extra rows `(0, 9)` re-decoded post-assimilation by Fold 0, freeze (`Windows`) vs runtime (`Linux Codespace`) platform split, and Secondary equal-weight ensembling underperforming best single seeds (`s1003` ID / `s1001` OOD).
- **Prioritized P12 Research Takeaways**:
  1. **P0 — Bilateral Balanced Fold Partition**: Replace asymmetric `11 + 5` fold partition so every fold retains both pre-anchor (`r < 10`) and post-anchor (`r > 10`) helpers (plus leave-one-AUX-out proposal ablation) to close Fold 0's `22.80%` vs `9.54%` Wave-1-correct `full_reset` gap.
  2. **P1 — Off-Grid Front-End & Frame-B Refit**: Break Frame-A 4-pilot comb grating-lobe near-aliasing (`0.970–0.973` overlap) and add Frame-B off-grid payload covariance/wavefront refit before scaling neural gating.
  3. **P2 — Clean `2 × 2` Transport × Reach Ablation**: Separate `{local 1D, global 2D}` from `{signed U(1) transport, unsigned abs}`, fix high-Doppler/5-ray OOD saturation (`NU_MAX_BAND=0.145`, `K_HAT=5`), and calibrate ensembling.
  4. **P3 — Out-of-Support Primary OOD**: Decouple the frozen shift bank from OOD test regimes (`shift_row` outside `{0, 6}`, multi-ray/chirp shifts) while labeling `ONSET_ROWS=(0,6)` as in-bank multi-onset inference.
  5. **P4 — Audit & Provenance Completeness**: Wire Primary OOD/sanity and Secondary OOD Holm tests into `p12_audit.json`, freeze pre-injection readouts for extra rows `(0, 9)`, and freeze off-grid training scripts/seed manifests.
