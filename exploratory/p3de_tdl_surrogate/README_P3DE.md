# EM-FloatFormer — Static TDL-profile surrogate (NOT NVIDIA Sionna runtime)

This subproject is a separate exploratory, civilian receive-only experiment prepared as a **contingency** when Sionna 2.2 was unavailable. It is **not** the completed formal P3-D or P3-E. Read `P3DE_SURROGATE_PROTOCOL.md`, `P3DE_SURROGATE_REPORT.md`, and `P3DE_SCOPE_ADDENDUM.md` before interpreting results.

## Run

```bash
python -m pip install torch numpy scipy matplotlib
python p3de_profile_bridge.py
python p3de_experiment.py --mode smoke
python p3de_experiment.py --mode train --out p3de_results
python p3de_experiment.py --mode eval --out p3de_results
python p3de_experiment.py --mode finalize --out p3de_results
python p3de_independent_audit.py
python p3de_make_report.py
```

The test uses PyTorch 2.10 CPU, TDL-A/C/D 3GPP TR38.901 v19.2 tap-delay/power parameters from NVIDIA Sionna v2.2.0 (Apache 2.0), a *static independent* harmonic frequency-response model, and no real RF hardware.

## Result interpretation

Training profile family: TDL-A/C at 500ns; out-of-distribution profiles: TDL-C at 1000ns and TDL-D at 750ns. Two 985-parameter previously defined neural architectures trained with **10 new random seeds**. We compare to LS, DFT8/12, hard threshold DFT, a nonprivileged train-population covariance LMMSE, and extra-information test covariance/true-channel references. QPSK BER and channel MSE are separate metrics; end-to-end latency, modern 5G/6G processing and real-world performance remain untested.

The project does **not** generate or optimize intentional RF interference, transmit signals to the air, or control devices.
