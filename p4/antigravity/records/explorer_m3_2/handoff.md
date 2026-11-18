# Handoff Report — `explorer_m3_2` (Milestone 3 / Stage P4-B Alignment)

**Handoff Type:** Hard (Task Complete)
**Sender:** `explorer_m3_2`
**Detailed Analysis:** `explorer_m3_2\analysis.md`

---

## 1. Observation

Direct inspection of M1, M2, and E2E test contracts on disk confirms the following exact file states, primitives, and invariants:

1. **Package & Isolation (`p4/__init__.py:1-134`, `PROJECT.md:114-140`, `sub_orch_m3/SCOPE.md:20-23`)**:
   - `p4/__init__.py` exports M1 primitives from `p4.simulator` (`rapp_pa`, `apply_rapp_pa_samples`, `oversample_frequency_to_time`, `decimate_time_to_frequency`, `DynamicChannelSimulator`, `generate_frame`, `static_limit_parity_check`, `q16`, `qpsk`, `hard_bits_16qam`, `PROFILES`), M1 baselines from `p4.classical_baselines` (`bussgang_params`, `bussgang_lmmse`, `calibrated_16qam_llrs`, `iterative_dd_pa_cancellation`, `delay_est_dft`, `ls_est`, `hard_slice_16qam`, `soft_slice_16qam`), and M2 components from `p4.em_floatformer` (`compute_gauge_anchor`, `apply_gauge_normalization`, `rescale_channel_estimate`, `ComplexRMSNorm`, `ComplexLinear`, `ComplexRoPE`, `hermitian_attention_scores`, `RadialActivation`, `GreenAttention`, `JointEmbedding`, `BidirectionalBlock`, `FloatFormerOutput`, `EMFloatFormer`, `create_floatformer`, `soft_reconstruct_16qam`, `JointPhysicalLoss`, `compute_ece`, etc.).
   - All M1/M2 source files (`p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, `p4/em_floatformer.py`) and tests (`p4/tests/test_p4_0_simulator.py`, `p4/tests/test_p4_a_equivariance.py`) must remain unmodified.

2. **`p4/simulator.py` (`1–523`)**:
   - Numerology & profiles (`:20-36`): `DEFAULT_N = 32`, `DEFAULT_CP = 16`, `DEFAULT_OVERSAMPLE = 4`, `SIGMA_16QAM = math.sqrt(10.0)`, `THRESH_16QAM = 2.0 / SIGMA_16QAM`, `PROFILES = {"train3", "equal4", "long6", "long12", "sparse12"}`.
   - Shape-agnostic constellation & noise helpers: `complex_randn` (`:39-49`), `q16` (`:52-66`), `qpsk` (`:69-73`), `hard_bits_16qam` (`:76-82`).
   - Oversampled Rapp PA (`:89-209`): `rapp_am_am`, `rapp_am_pm`, `apply_rapp_pa_samples`, `oversample_frequency_to_time`, `decimate_time_to_frequency`, and `rapp_pa` all operate along `dim=-1` with `norm="ortho"` FFT/IFFT and broadcast cleanly over `(B, N_symb, N_sc)`.
   - `DynamicChannelSimulator` (`:225-378`): `draw_channel` (`:253-289`) draws `h_init` and Jakes `fd_shifts`, while `filter_time_signal` (`:291-344`) asserts `total_len == self.n + self.cp` (1D single-symbol length) and starts `n_grid` at `0`. Static-limit parity (`:485-523`) requires relative Frobenius error `< 1e-5` under `fd=0`, `A=None`, `alpha=0.0`, `norm="ortho"`.

3. **`p4/classical_baselines.py` (`1–466`)**:
   - `bussgang_params` (`:41-92`) returns `(gamma, sigma_d2, p_out)` via cached 1D Rayleigh quadrature; returns `(1+0j, 0.0, 1.0)` when `A is None or math.isinf(A)`.
   - `calibrated_16qam_llrs(z, sigma2, method="exact")` (`:123-203`) accepts arbitrary leading dims `(..., N)`, clamps `sigma2 >= 1e-8`, preserves `z.real.dtype` (`float32`/`float64`), and uses convention **`P(b_i = 1 | z) = sigmoid(llr)`** (`llr > 0 => bit = 1`).
   - `hard_slice_16qam` (`:206-208`) and `soft_slice_16qam` (`:211-220`) operate on arbitrary leading dims.
   - `bussgang_lmmse` (`:240-318`) and `iterative_dd_pa_cancellation` (`:336-466`) assume 1D full-pilot `(B, N)` frames.

4. **`p4/em_floatformer.py` (`1–1817`)**:
   - Gauge & equivariant layers: `compute_gauge_anchor(yp, eps=1e-38)` (`:42-54`), `apply_gauge_normalization` (`:57-67`), `rescale_channel_estimate` (`:70-72`), `ComplexRMSNorm` (`:75-102`, `amax`-rescaled in `complex128` with `active = m > 1e-38`), `ComplexLinear(..., bias=False)` (`:157-201`), `hermitian_attention_scores` (`:233-247`), `RadialActivation` (`:250-281`, `complex128` with `m > 1e150` rescaling + `(eps/m_safe)**2` guard), `FloatFormerOutput` (`:507-517`).
   - `ComplexRoPE` (`:204-230`) & `GreenAttention` (`:284-359`) are 1D sequential (`pos_q = 0..N_q-1` at `:350`).
   - `complex128` & LLR calibration contract (`:592-655`, `sub_orch_m2_resumed/handoff.md:32-42`): internal backbone/heads/equalization execute in `complex128`/`float64` (casting outputs back to caller dtype at boundary), with:
     ```python
     sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 3.5e-2
     llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
     llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)
     ```
   - `JointPhysicalLoss` (`:800-907`) and `p4a_comparative_benchmark.json` (`:1-1149`): deterministic seeding (`seed`, `seed+1`, ...), `torch.set_num_threads(4)`, exact integer `bit_errors` / `total_bits`, `compute_ece`, and mixed-PA `A` resolution.

5. **E2E Test Contracts (`tests/e2e/test_tier{1,2,3,4}*.py`)**:
   - Expect `p4/slot_2d.py` to expose specific symbols for F13–F17, including `SlotSimulator2D`, `get_dmrs_mask`, `get_slot_masks`, `generate_dmrs`, `get_custom_dmrs_mask`, `simulate_slot_channel`, `measure_ici_leakage`, `measure_cfo_phase_drift`, `estimate_cfo_cp`, `compute_doppler_spectrum`, `check_phase_continuity`, `LDPCMatrix(codeword_len=128)` (`h_matrix.shape == (64, 128)`), `encode_ldpc(u) -> (c, h_mat)`, `decode_min_sum(llrs, h_mat, max_iter=..., return_iters=...)` (with `(2*c - 1)*20.0` sign convention `> 0 => 1` and `iters == 1` early stopping), `measure_coding_gain`, `correct_single_bit`, `compute_2d_correlation`, `lmmse_2d_interpolate`, `compare_1d_vs_2d_lmmse`, `idd_pa_2d`, `idd_pa_2d_cleanup`, `check_block_error`, `evaluate_bler_curve`, `run_block_counter`, `check_doppler_bler_sensitivity`, `evaluate_ood_profile_bler`, `compute_ci`, plus Tier 2/3/4 helper/workload functions (cataloged in `analysis.md` Section 4.6).

---

## 2. Logic Chain

1. **Isolation via Direct Import**: Because `rapp_pa`, `oversample_frequency_to_time`, `decimate_time_to_frequency`, `apply_rapp_pa_samples`, `q16`, `qpsk`, `bussgang_params`, `calibrated_16qam_llrs`, `soft_slice_16qam`, `compute_gauge_anchor`, `ComplexRMSNorm`, `ComplexLinear`, `hermitian_attention_scores`, `RadialActivation`, `FloatFormerOutput`, `JointPhysicalLoss`, and `compute_ece` are already shape-agnostic or operate on `-1`, `p4/slot_2d.py` can import and compose them directly with zero edits to M1/M2 files.
2. **2D Continuous Channel vs 1D Simulator**: Since `DynamicChannelSimulator.filter_time_signal` enforces single-symbol length `N + CP` and resets time `n = 0`, `slot_2d.py` should reuse `DynamicChannelSimulator.draw_channel` (or tap statistics) while executing continuous FIR filtering + sample-level CFO `exp(j * 2*pi * cfo * n / N_sc)` over the concatenated `N_symb * (N_sc + N_cp)` sample stream so phase has zero discontinuity across CP boundaries and preserves static-limit parity (`fd=0, cfo=0, A=None, alpha=0`).
3. **2D Slot Bit Packing & Fast PyTorch LDPC**: With default `(14, 32)` and comb-2 DMRS on symbols `(2, 11)`, pilot overhead is `32 / 448 = 7.14% <= 10%`, leaving `416` payload REs = `1664` 16-QAM bits = **`13 * 128` exact codewords** (and `(14, 64)` gives `3328` bits = **`26 * 128` exact codewords**). Constructing a girth $\ge 6$ Rate-1/2 `(64, 128)` QC/PEG-LDPC with precomputed GF(2) systematic generator `G = [I_64 | P]`, flipping incoming `LLR > 0 => 1` signs into vectorized padded-CSR Normalized Min-Sum (`alpha = 0.75`, syndrome early-stop), achieves `> 3 dB` coding gain, `iters == 1` on clean LLRs, and `< 100 ms` execution per 50-slot evaluation on 4 CPU threads.
4. **2D $C^*$-Equivariant Transformer Extension**: 1D `ComplexRoPE`/`GreenAttention` assume sequential `0..N-1` indices. Extending them in `slot_2d.py` to coordinate-aware `ComplexRoPE2D` + `GreenAttention2D` (axial time/freq self-attention + sparse-DMRS bidirectional cross-attention at exact `(t_p, f_p)` vs `(t, f)` coordinates), anchored by `compute_gauge_anchor(yp_sparse)` and a real-valued RBF initial channel interpolation prior `h0_bar`, executed end-to-end in `complex128`/`float64` with the `3.5e-2` floor + `20.0 * tanh` output head, mathematically guarantees `< 1e-5` LLR invariance and `0` bit flips across 24 octaves and full phase rotations while supporting ultra-sparse (1.78% or 1-RE) DMRS.

---

## 3. Caveats

- All findings are from static inspection of `p4/`, `tests/e2e/`, `PROJECT.md`, and M1/M2 handoffs; no code was executed or modified during this read-only exploration.
- `compute_gauge_anchor(0.0)` edge case noted in `sub_orch_m2_resumed/handoff.md:79` (NaN gradient only when `yp` is identically `0.0` AND `yp.requires_grad=True`): irrelevant for inference/standard training where `yp` is an observation tensor, and does not affect model parameters.
- For `test_t4_s07_ood_channel_generalization_workload` (unseen 16-tap exponential PDP), `delays = 0..15` fits `DEFAULT_CP = 16` (`15 <= 16`) without modifying `PROFILES` in `p4/simulator.py`.

---

## 4. Conclusion

1. **Zero M1/M2 Edits Required**: `p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, and `p4/em_floatformer.py` expose all required low-level PA, constellation, LLR, gauge-norm, complex-layer, and loss primitives cleanly. Only `p4/__init__.py` needs M3 symbol re-exports appended once `p4/slot_2d.py` is implemented.
2. **Concrete Blueprint Ready**: `analysis.md` Section 4 provides the full architecture for `p4/slot_2d.py` covering F13–F17, including:
   - Continuous sample-level multipath + Doppler + CFO slot simulator with CP-correlation CFO recovery;
   - Girth $\ge 6$ systematic Rate-1/2 `(64, 128)` LDPC with vectorized Normalized Min-Sum BP matching `LLR > 0 => bit = 1`;
   - 2D separable LMMSE (`R_t \otimes R_f`) + 2D IDD-PA baselines;
   - `EMFloatFormer2D` with `ComplexRoPE2D`, axial + sparse-DMRS bidirectional `GreenAttention2D`, `complex128` backbone, `3.5e-2` / `20.0 * tanh` calibration contract, and exact $C^*$ equivariance (`< 1e-5`, 0 flips);
   - Complete symbol checklist matching all Tier 1–4 E2E test expectations.

---

## 5. Verification Method

After implementing `p4/slot_2d.py`, `p4/tests/test_p4_b_2d_fec.py`, and updating `p4/__init__.py`, verify from ``:
1. Confirm M1/M2 files untouched: `git diff --name-only p4/simulator.py p4/classical_baselines.py p4/benchmark_latency.py p4/em_floatformer.py p4/tests/test_p4_0_simulator.py p4/tests/test_p4_a_equivariance.py` (must be empty).
2. Run existing M1 & M2 regressions:
   - `python -m unittest p4.tests.test_p4_0_simulator p4.tests.test_p4_a_equivariance -v` (must remain 14/14 + 53/53 OK).
3. Run new M3 unit tests:
   - `python -m unittest p4.tests.test_p4_b_2d_fec -v` (verifying F13–F17, 24-octave $C^*$ equivariance `< 1e-5` & 0 flips, static parity `< 1e-5`, LDPC coding gain `> 3 dB`, and `p4/results/p4b_coded_bler.{json,csv}` parity).
4. Run E2E suite for M3 active tests:
   - `python -m unittest tests.test_p4_e2e -v` (confirming all M1 + M2 + M3 E2E tests in Tiers 1–4 pass with 0 failures and 0 errors).
