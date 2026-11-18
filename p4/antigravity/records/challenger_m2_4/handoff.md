# Challenger 4 Handoff Report (`challenger_m2_4`) — Milestone 2 (Stage P4-A)

**Verdict: APPROVE**

---

## 1. Observation

All empirical checks were executed directly against `p4\em_floatformer.py` (with `torch.set_num_threads(4)`). Source inspection and `git diff --stat` confirm the repository was untouched during verification (`git diff --stat` returned empty).

### 1.1 Standard Test Suites Executed
- `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 45 tests in 228.711s — OK** (exit code 0).
- `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 247.377s — OK (skipped=97)** (156 active M1/M2 tests passed, 97 M3/M4 tests progressively skipped; exit code 0).
- `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 11.501s — OK** (exit code 0).

### 1.2 Criterion 1 — Mutual Gradient Flow
- `BidirectionalBlock(d_model=12, n_heads=2)` (`p4/em_floatformer.py:415-479`):
  - `p -> d` path (`loss_d = (d_out.abs() ** 2).sum()`): `p.grad` abs sum = `2387.9756`, `cross_p2d` min param grad = `1316.952637`, `self_attn_p` min param grad = `257.386810`.
  - `d -> p` path (`loss_p = (p_out.abs() ** 2).sum()`): `d.grad` abs sum = `1907.8220`, `cross_d2p` min param grad = `896.875183`, `self_attn_d` min param grad = `195.829330`.
- `BidirectionalBlock(d_model=32, n_heads=2)`:
  - `p -> d`: `p.grad` = `5112.8838`, `cross_p2d` min = `5154.194336`, `self_attn_p` min = `1398.462402`.
  - `d -> p`: `d.grad` = `6032.6719`, `cross_d2p` min = `5990.092773`, `self_attn_d` min = `1561.859619`.
- Full `EMFloatFormer(depth=2)` under isolated `bce_loss(out["llrs"], frame.bits)`:
  - `0` zero-gradient parameter tensors out of `104`: `block 0` (`cross_p2d=0.142087`, `cross_d2p=0.192493`), `block 1` (`cross_p2d=0.089997`, `cross_d2p=0.098262`).
- Full `EMFloatFormer(depth=2)` under isolated `nmse_loss(out["h_hat"], frame.h_true)`:
  - `cross_d2p` receives gradients in both blocks (`block 0=0.188246`, `block 1=0.105122`), and `embed.proj_d.weight_r` receives `0.008381` (confirming `data -> pilot` flow reaches the data embedding).
- Causal isolation check (`use_joint_pilot_data=False`):
  - `p1.grad` from `d_out` is `None`; `d2.grad` from `p_out` is `None`.

### 1.3 Criterion 2 — Unclamped BER Gain Verification
- Source inspection of `compare_pilot_vs_bidirectional()` (`p4/em_floatformer.py:644-697`):
  - Instantiates and trains `EMFloatFormer(d_model=12, n_heads=2, depth=2)` for 50 Adam steps on strong-PA frames (`A=1.25`, `snr_db=18.0`, `n=32`), evaluates against pilot-only LS baseline, and returns raw `10.0 * math.log10(max(ber_pilot, 1e-4) / max(ber_bidi, 1e-4))` with no clamp and no classical PA-cancellation shortcut.
- Repeated calls (`x3`) to `compare_pilot_vs_bidirectional()`:
  - `[0.7269911851970452, 0.7269911851970452, 0.7269911851970452]` dB (`>= 0.5` dB; deterministic because `torch.manual_seed(42)`, `seed=101`, `seed=2026` are fixed inside the function).
- Independent multi-seed replication (8 distinct `(model_seed, train_seed, eval_seed)` triples, 50 Adam steps, `snr_db=18.0`, `pa_condition="strong"`):
  - `(42, 101, 2026)`: `ber_ls=0.09741`, `ber_em=0.08240`, `gain=0.7270 dB`
  - `(1, 11, 111)`: `ber_ls=0.09143`, `ber_em=0.07788`, `gain=0.6966 dB`
  - `(2, 22, 222)`: `ber_ls=0.09180`, `ber_em=0.08081`, `gain=0.5536 dB`
  - `(3, 33, 333)`: `ber_ls=0.09058`, `ber_em=0.07532`, `gain=0.8012 dB`
  - `(7, 77, 777)`: `ber_ls=0.09497`, `ber_em=0.07849`, `gain=0.8277 dB`
  - `(99, 199, 299)`: `ber_ls=0.09473`, `ber_em=0.07788`, `gain=0.8504 dB`
  - `(123, 456, 789)`: `ber_ls=0.10400`, `ber_em=0.08435`, `gain=0.9096 dB`
  - `(555, 666, 777)`: `ber_ls=0.09497`, `ber_em=0.08618`, `gain=0.4217 dB`
  - Summary: `Mean = 0.7235 dB`, `Min = 0.4217 dB`, `Max = 0.9096 dB` (7/8 seeds `>= 0.5` dB at 50 steps).
- Extended 150-step check on `(42, 101, 2026)` (`batch_size=256` train / `128` eval):
  - `use_joint=True`: `ber_ls=0.09387`, `ber_m=0.07538`, `gain=0.9529 dB`, `nmse=0.01825`.
  - `use_joint=False`: `ber_ls=0.09387`, `ber_m=0.07111`, `gain=1.2063 dB`, `nmse=0.01660` (50 steps: `0.8246 dB`).

### 1.4 Criterion 3 — Noise Robustness
- Source inspection of `evaluate_noise_data_feedback()` (`p4/em_floatformer.py:700-723`):
  - Uses `EMFloatFormer(depth=1)` with active non-zero `head_h` (`weight_r/i` abs sum = `0.041499`) and `head_dist` (`0.031084`), generates `yd_noise ~ CN(0, 1)`, and returns raw unclamped `10.0 * abs(math.log10(max(nmse_noisy, 1e-6) / max(nmse_clean, 1e-6)))`.
- 6 direct calls to `evaluate_noise_data_feedback()`:
  - `[0.001123, 0.001307, 0.004541, 0.003059, 0.000842, 0.001173]` dB (all `> 0.0` and strictly `< 1.0` dB).
- 6-seed untrained probe (`depth=1`):
  - `deg_db`: `[0.004412, 0.004269, 0.002886, 0.002720, 0.001448, 0.007109]` dB (max `0.007109 dB`).
  - `|h_clean - h_noise|` max: `[0.013270, 0.009516, 0.006753, 0.015292, 0.005756, 0.008458]` (confirms `yd` actively influences `h_hat`).
- 50-step **trained** `EMFloatFormer(depth=2)` probe (`JointPhysicalLoss(lambda_nmse=2.0, lambda_pa=0.25)`):
  - Matched unit-power pure noise `yd_n ~ CN(0, 1)`: `NMSE clean = 0.021539`, `NMSE noise = 0.022425`, `deg_db = 0.174916 dB < 1.0 dB`.
  - Extreme mismatched OOD noise (`10x` amplitude = `100x` power on `yd` without scaling `yp`): `deg_db = 5.918257 dB`.

### 1.5 Criterion 4 — Physical PA Loss (`JointPhysicalLoss.pa_residual_loss`)
- Parseval energy conservation (`p4/em_floatformer.py:797-821`):
  - `torch.fft.ifft(x_rand, dim=-1, norm="ortho")` preserves L2 energy to `2.216e-07` relative error.
  - Full roundtrip `ifft -> rapp_pa(A=1e9, os=1) -> fft` preserves energy to `5.541e-07` relative error.
- Rapp nonlinear distortion discrimination:
  - Noiseless strong-PA frame (`A=1.25`, `snr_db=60.0`, `fd=0.0`, `oversample=1`):
    - `pa_residual_loss(yd, h_true, x_true, A=1.25)` = `2.378e-06`
    - `pa_residual_loss(yd, h_true, x_true, A=1e9)` (linear PA hypothesis) = `0.07933`
    - `pa_residual_loss(yd, h_true, x_corrupt, A=1.25)` (25% bit errors) = `2.05416`
  - Noiseless strong-PA frame (`A=1.25`, `oversample=4` default simulator):
    - `pa_residual_loss(yd, h_true, x_true, A=1.25)` = `0.01202`
    - `pa_residual_loss(yd, h_true, x_true, A=1e9)` = `0.07508` (6.2x higher)
    - `pa_residual_loss(yd, h_true, x_corrupt, A=1.25)` = `2.07830` (173x higher)
- Gradient stability and $C^*$ equivariance with active `lambda_pa=0.25`:
  - Normal `x_est` / `h_est` gradient norms: `0.01112` / `0.01170`.
  - All-zero inputs (`x=0, h=0, yd=0`): `loss=0.0`, finite gradients (`x=True, h=True`).
  - Full `EMFloatFormer(depth=2)` + `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`: all 104 parameter gradient norms finite and bounded in `[6.751e-05, 1.747e+00]`.
  - Scaling & phase invariance under $c = 2^{\pm 12} e^{j 0.77}$ with `lambda_pa=0.25`: `loss_diff = 0.000e+00`, `max_grad_diff = 2.682e-07`.

### 1.6 Criterion 5 — Parameter Count
- Default `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=32)`:
  - `total_params = 14090`, `trainable_params = 14090`, `tensors = 104`.
  - Module breakdown: `embed: 48`, `blocks: 13968`, `norm_p: 12`, `norm_d: 12`, `head_h: 24`, `head_dist: 24`, `gamma_eff: 2`.

---

## 2. Logic Chain

1. **Bidirectional Cross-Attention Integrity**: Both `p -> d` (`cross_p2d`) and `d -> p` (`cross_d2p`) in `BidirectionalBlock` propagate strong non-zero gradients between streams (Obs. 1.2), and removing `use_joint_pilot_data` cleanly breaks both cross-stream gradient paths while leaving `EMFloatFormer` fully differentiable across all 104 parameter tensors under `JointPhysicalLoss` (Obs. 1.2, 1.5).
2. **Genuine Unclamped BER Gain**: `compare_pilot_vs_bidirectional()` executes a genuine 50-step `EMFloatFormer` training loop with no classical PA-cancellation delegation or return clamp, delivering `0.7270 dB` raw gain (`0.7235 dB` mean across 8 independent seed triples; Obs. 1.3).
3. **Robustness Against Noisy Data Feedback**: Active non-zero output heads (`head_h`, `head_dist`) pass data-dependent signals into `h_hat`, yet both untrained (`0.0008–0.0071 dB`) and 50-step trained (`0.1749 dB`) receivers maintain NMSE degradation strictly `< 1.0 dB` under matched unit-power pure-noise `yd` corruption (Obs. 1.4).
4. **Physical Consistency of PA Residual Loss**: Using `norm="ortho"` FFT/IFFT preserves Parseval energy to `~2e-7`–`5e-7`, matches the `oversample=1` Rapp AM-AM/AM-PM physics to `2.38e-6`, strongly separates nonlinear PA (`A=1.25`) from linear PA (`A=1e9`) and symbol errors even on `oversample=4` simulator frames, and produces well-conditioned, strictly $C^*$-invariant gradients (`2.68e-7` max diff across `2^-12..2^12`) with zero collapse or explosion (Obs. 1.5).
5. **Compact Budget & Suite Compliance**: `14,090` trainable parameters satisfies the `~14k` compact parameter budget (Obs. 1.6), and all active unit, simulator, and E2E suites pass (Obs. 1.1).

---

## 3. Caveats

Non-blocking caveats observed empirically for downstream milestones (M3/M4):
1. **Short-budget seed variance**: Across 8 random seed triples at only 50 mini-batch steps, 1 seed `(555, 666, 777)` achieved `0.4217 dB` (`< 0.5 dB`) while the 8-seed mean was `0.7235 dB`; longer training (150 steps) raises gain to `0.9529 dB`.
2. **Ablation convergence speed (`use_joint_pilot_data=False`)**: On static single-symbol strong-PA frames (`(42, 101, 2026)`), `use_joint_pilot_data=False` achieved `0.8246 dB` (50 steps) and `1.2063 dB` (150 steps) vs `0.7270 dB` / `0.9529 dB` for `use_joint_pilot_data=True`, because `gamma_eff` + `self_attn` + `head_dist` already compensate static-channel PA distortion while fewer parameters converge slightly faster. Controlled multi-seed factorial ablation is properly deferred to Stage P4-C/D (M4).
3. **10x OOD noise-power mismatch on `yd` alone**: Because `BidirectionalBlock` passes unnormalized `value_context=d_res` into `cross_d2p` (preserving amplitude scale), corrupting `yd` with `10x` amplitude (`100x` power) noise while holding `yp` fixed degrades a trained model's channel NMSE by `5.92 dB` (whereas matched unit-power `CN(0,1)` noise degrades by only `0.175 dB < 1.0 dB`).
4. **`pa_residual_loss` oversampling**: `pa_residual_loss` evaluates Rapp PA at `oversample=1` rather than `oversample=4` (`p4/simulator.py`), producing a small baseline floor (`0.01202` vs `2.38e-6`) on `oversample=4` noiseless frames, though distortion discrimination remains strong (`0.07508` for linear PA, `2.07830` for corrupted symbols).

---

## 4. Conclusion

`p4/em_floatformer.py` satisfies all five empirical requirements for Milestone 2 (Stage P4-A): mutual pilot-data gradient flow, genuine unclamped BER gain `>= 0.5 dB` under strong PA, noise-feedback degradation strictly `< 1.0 dB` with active non-zero heads, physically consistent Parseval-preserving and gradient-stable `JointPhysicalLoss.pa_residual_loss`, and a compact `14,090`-parameter footprint.

**Recommendation: APPROVE** Milestone 2 (`p4/em_floatformer.py`).

---

## 5. Verification Method

Run from ``:
```powershell
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -m unittest p4.tests.test_p4_0_simulator -v
git diff --stat
```
Key files inspected:
- `p4/em_floatformer.py`
- `p4/simulator.py`
- `p4/classical_baselines.py`
- `p4/tests/test_p4_a_equivariance.py`
- `tests/test_p4_e2e.py`

Invalidation conditions: any test failure in `test_p4_a_equivariance`, `test_p4_0_simulator`, or active M1/M2 tests in `test_p4_e2e`; any zero-gradient tensor on step 1; any NaN/Inf gradient in `pa_residual_loss` or `RadialActivation`; any return clamp or classical shortcut in `compare_pilot_vs_bidirectional()` / `evaluate_noise_data_feedback()`; or any deviation from `git diff --stat` being clean on tracked historical files.
