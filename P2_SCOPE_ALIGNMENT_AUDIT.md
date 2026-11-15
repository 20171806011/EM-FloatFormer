# P2 scope alignment audit (pre-supplement)

This is a traceability audit of **three specific missed pieces** of the originally discussed P2 plan, not a finding of misconduct, and does not alter frozen P2-E.

| Requested original-plan entry | Verified actual evidence | Pre-supplement status |
|---|---|---|
| RMS-only and **local** RMS-normalized IQ separate from log-scale sidepath | `P2B_REPORT.md`: raw IQ, log/phase, dual (frame-RMS norm + frame log RMS); `p2_models.py`: `encode_complex` has only raw, log, dual. | **Missing**: no standalone RMS-only, no local-window RMS control. |
| Sparse pilot density and appropriate pilot-aware classical baselines | `P2D_REPORT.md`: frozen weights, five multipath profiles and perturbations; `ofdm_baseline.py`: `yp=h+noise` on all 32 carriers and known pilot for every carrier; `p2_run.py`: no pilot mask input. | **Missing**: not a sparse-pilot experiment; full-pilot only. |
| Independent mixed train-profile comparison with matched prior access | `P2_PROTOCOL.md` specifies train3 600 steps for A/B/C, `p2_run.py` trains with `TRAIN_PROFILE`, `P2E_FREEZE.json` and `P2E_REPORT.md` use only train3. Matched-test-profile LMMSE receives extra information. | **Missing**: no separate mixed training arm or mixture-covariance conventional comparison. |

The user-requested supplements S1, S2, S3 are separately frozen in `P2_SUPPLEMENT_PROTOCOL.md` **before execution**. Original pre-registered P2-E conclusions and files are not edited. Final results and deviations, if any, are recorded only in separate supplement artifacts.
