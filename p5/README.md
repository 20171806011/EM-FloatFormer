# Phase 5 EM-native receiver experiments

The [study report](P5_REPORT.md) gives the measured link performance, mechanism tests and limitations. The [prospective protocol](PROTOCOL.md) specifies the seeds, operating-point selection, budget, endpoints and multiple-testing families. The entire study runs from the repository root on CPU. The baseline checkout is `8380084c6dece64119ab72f9c3c1c212e154fb17`.

## Files and receiver inputs

| File | Role |
|---|---|
| [wavefield_sim.py](wavefield_sim.py) | OFDM, oversampled nonlinear PA, continuous Doppler, oscillator and four-element array, independent RNG streams and coded frames |
| [unrolled_em_transformer.py](unrolled_em_transformer.py) | Pilot Fourier dictionary, residual-based solver, centered Hermitian attention, explicitly matched learning controls |
| [classical.py](classical.py) | Ensemble-covariance LMMSE, nominal PA reconstruction, CPE tracking, trajectory detection, iterative waveform/code feedback, privileged PA comparison |
| [study.py](study.py) | Calibration, fit, scoring, kernel probe, checkpoint replay, latency |
| [tests](tests) | Receiver invariance and independent waveform/FEC checks |
| [results](results) | Calibration grids, training traces, final weights, integer counts, attention measurements, CPU timings and verification |

Nonprivileged receivers accept `Observation(y, pilot_mask, pilots, noise_var, regime)`. The waveform generator exposes `Frame.truth` separately to the scorer. The PA genie receives only `Frame.truth.pa`. Every SNR is transmit-symbol energy divided by complex AWGN variance per receive element. The reported channel target is the ideal post-CP propagation diagonal, before PA and receiver oscillator. The whole R2 burst contains two contiguous ten-symbol laboratory slots. R3 carries one spatial stream on four receive elements.

## Reproduce from the root in PowerShell

Use the checkout's Python 3.12 and CPU PyTorch environment. The recorded run used Python 3.12.7 and PyTorch 2.12.1+cpu. The study sets one PyTorch thread per process and uses deterministic algorithms. The plotting stage also uses Matplotlib. The following commands regenerate the results at their original paths.

```powershell
python -B -m unittest discover -s p5/tests -v
python -B -m p5.exp_blind_pa_phn calibrate
python -B -m p5.exp_super_aliased_doppler calibrate
python -B -m p5.exp_mimo_3d_green calibrate
python -B -m p5.exp_blind_pa_phn fit
python -B -m p5.exp_super_aliased_doppler fit
python -B -m p5.exp_mimo_3d_green fit
1..3 | ForEach-Object { python -B -m p5.study latency --regime $_ }
python -B -m p5.statistics
python -B -m p5.audit_model
python -B -m p5.verify_results
python -B -m p5.verify_amendment
python -B -m p5.verify_checkout
```

Each fit invocation can take repeated `--pair` arguments. The original run assigned R1 odd and even pairs to two processes, R2 to a third, and R3 to a fourth. Separate pairs use disjoint RNG streams and write separate directories. Run latency after all fitting processes have exited. With the supplied final checkpoints, the following commands regenerate each test frame and every receiver count without refitting.

```powershell
python -B -m p5.verify_replay
```

The [original frozen run](revisions/first_run.zip) preserves the pre-amendment model and its measurements. The amendment repairs the exact-zero reference in a pilot erasure. The full protocol was repeated with identical seeds. The amendment verifier reads the archive in place and compares the two revisions. The checkout verifier runs all P5 tests, `python -B -m publication.verify_p4`, checks HEAD, and checks both staged and unstaged tracked paths. It expects the original clean baseline plus the new untracked `p5/` directory.

## Reading a record

`results/r2/pair01/record.json`, for example, contains four validation/training traces, elapsed wall time, and counts for all receivers on the same 80 bursts. `bit_errors/bits` measures pre-decoder decisions on coded bits. `info_errors/info_bits` and `block_errors/blocks` measure the common terminal LDPC decoder. One wrong information bit makes an erroneous block. Add the channel numerators and denominators across seeds before converting their ratio to dB. R3 records also contain each probed layer's token coordinates, per-burst correlations, and eight key permutations before and after training. The calibration JSON files retain both full grids, the independent validation result, and SHA-256 identities of the protocol and core experiment sources. The checkpoint files include the frozen initialization map used in the forward pass.
