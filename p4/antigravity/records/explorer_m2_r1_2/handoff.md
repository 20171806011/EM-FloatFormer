# Explorer Report (`explorer_m2_r1_2`): Audit & Calibration of LLR Variance Floor (`sigma_eff2`) and Saturation Bounds in `p4/em_floatformer.py`

**Core Summary:** The Iteration 3 tightening at `p4/em_floatformer.py:640-642` (`+ 5.0e-2` floor and `8.0 * torch.tanh(llrs64 / 8.0)`) is unnecessarily restrictive given that `EMFloatFormer.forward` now executes its internal backbone in `complex128`/`float64`: it compresses high-SNR LLR dynamic range to `[-8, +8]` (vs `~20–11,000` in `bussgang_lmmse`), imposes an effective ~13 dB SNR ceiling (`sigma_eff2 >= 0.05`), and degrades Rate-1/2 Min-Sum LDPC coded BLER by `+1.0%` to `+2.0%` absolute BLER under `strong` PA. However, lowering `sigma_eff2` below `2.0e-2`–`2.5e-2` or raising `|LLR|` above `~64` **mathematically violates** `max_llr_err < 1e-5` on `complex64` caller-scaled inputs `(yp * c, yd * c)` due to `complex64` input quantization (`~1e-7`) amplified by `calibrated_16qam_llrs` slope near decision boundaries (`1.265..2.530 / sigma_eff2`) and `float32` output ULP rounding; **we recommend calibrating line 640 to `+ 2.5e-2` and line 642 to `20.0 * torch.tanh(llrs64 / 20.0)` (or `25.0 * torch.tanh(llrs64 / 25.0)`).**

---

## 1. Observation

### 1.1 Current Implementation in `p4/em_floatformer.py`
- **`EMFloatFormer.forward` (`p4/em_floatformer.py:575-644`)**:
  - Lines `584-597`: Captures `out_complex_dtype = yp.dtype`, `out_real_dtype = yp.real.dtype` and casts `yp, yd, nv` to `complex128`/`float64`.
  - Lines `600-602`: `yp_bar, yd_bar, s0 = apply_gauge_normalization(yp64, yd64)` (`compute_gauge_anchor` at lines `33-45`, `eps=1e-38`) and `nv_bar = nv64 / (s0 ** 2)`.
  - Lines `621-635`: Computes `h_bar = yp_bar + self.head_h(p_out).squeeze(-1)`, `dist_bar = self.head_dist(d_out).squeeze(-1)`, `h_eff64 = self.gamma_eff(h_bar.unsqueeze(-1)).squeeze(-1)`, `h_power = h_eff64.real ** 2 + h_eff64.imag ** 2`, `dist_eq = (dist_bar * h_eff64.conj()) / (h_power + 1e-4)`, and `z_eq = (yd_bar * h_eff64.conj()) / (h_power + 1e-4) - dist_eq`.
  - Lines `638-642` (target region):
    ```python
    h_power_det = h_power.detach()
    dist_power_det = (dist_bar.real ** 2 + dist_bar.imag ** 2).detach()
    sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 5.0e-2
    llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
    llrs = (8.0 * torch.tanh(llrs64 / 8.0)).to(out_real_dtype)
    ```
- **`JointPhysicalLoss` (`p4/em_floatformer.py:787-855`)**:
  - `bce_loss` (`798-800`): `F.binary_cross_entropy_with_logits(llrs, bits.float())`
  - `nmse_loss` (`802-813`): scale-invariant channel NMSE against `frame.h_true` (physical channel $H$, not $\gamma H$).
  - `pa_residual_loss` (`815-839`): forward Rapp PA residual with `norm="ortho"`.

### 1.2 Comparison with `p4/classical_baselines.py` & `tests/e2e/` Expectations
- **`calibrated_16qam_llrs` (`p4/classical_baselines.py:123-204`)**:
  - Clamps `sigma2 = torch.clamp(sigma2, min=1e-8)` (`line 146`).
  - Exact LogSumExp over 4-PAM points `[-3, -1, 1, 3] / sqrt(10)` with `var_real = sigma2 / 2.0`; NO `tanh` cap.
  - Piecewise max-log slope (`line 185`): `scale = 4.0 / (SIGMA_16QAM * sigma2)` ($\approx 1.265 / \sigma^2$ inner, $2.530 / \sigma^2$ outer).
- **`bussgang_params` & `bussgang_lmmse` (`p4/classical_baselines.py:42-92, 240-318`)**:
  - Measured analytical Bussgang parameters (Task 35):
    - `clean` (`A=None`): `gamma = 1.0000+0.0000j`, `|gamma|^2 = 1.0000`, `sigma_d2 = 0.00000`
    - `mild` (`A=2.25`): `gamma = 0.9121+0.2770j`, `|gamma|^2 = 0.9086`, `sigma_d2 = 0.00912` (`sigma_d2 / |gamma|^2 = 0.01004`)
    - `strong` (`A=1.25`): `gamma = 0.7711+0.2257j`, `|gamma|^2 = 0.6456`, `sigma_d2 = 0.02332` (`sigma_d2 / |gamma|^2 = 0.03612`)
  - `bussgang_lmmse` sets `sigma2_eff = (1 - mu) / mu = (|H|^2 * sigma_d2 + nv) / |H_eff|^2` with no additive constant floor and no `tanh` cap. Measured `bussgang_lmmse` LLR/ECE/BCE (Task 35, 64×32 frames):
    - `clean`: SNR 15 dB `max|LLR|=368.46, ECE=0.0079`; SNR 20 dB `max|LLR|=1137.11, ECE=0.0024`; SNR 30 dB `max|LLR|=11131.80, ECE=0.0002`
    - `mild`: SNR 20 dB `max|LLR|=171.95, ECE=0.0022`; SNR 30 dB `max|LLR|=200.82, ECE=0.0000`
    - `strong`: SNR 20 dB `max|LLR|=60.83, ECE=0.0018`; SNR 30 dB `max|LLR|=72.38, ECE=0.0006`
- **E2E & Unit Test Contract (`TEST_INFRA.md:152-158`, `p4/tests/test_p4_a_equivariance.py`, `challenger_m2_3/handoff.md`)**:
  - F8/F12 $C^*$ invariance: `max_llr_err < 1e-5` and `flips == 0` across $r \in [2^{-12}, 2^{12}]$ and random phases on `complex64` caller inputs `(yp * c, yd * c, nv * r^2)` including trained `depth=2` models at 20–30 dB SNR.
  - F5 calibration: `ECE < 0.05`.
  - Stage P4-B (`test_tier1_feature_coverage.py:835-861`, `test_tier3_cross_feature.py:136-143`, `test_tier4_real_workloads.py:47-68, 171-180`): Rate-1/2 Min-Sum soft LDPC decoding depends on relative LLR confidence across subcarriers.

### 1.3 Empirical Measurements Across `sigma_eff2` Floors & `tanh` Bounds

#### A. Joint-Loss Training Convergence, Dynamic Range, ECE, BCE & C* Equivariance (Task 57)
Trained `EMFloatFormer(depth=2)` for 150 Adam steps (`lr=1e-2` backbone, `lr=3e-2` `gamma_eff`) on fresh `batch_size=32, snr_db=(10, 25)` frames with `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`, then evaluated on 128 held-out frames across 5 non-power-of-2 `(r, phi)` pairs `(2^-12, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)` in `complex64`:

| PA | Floor | `tanh` `B` | SNR 15 (BER / NMSE / ECE / BCE / `p95` / `max_llr_err`) | SNR 20 (BER / NMSE / ECE / BCE / `p95` / `max_llr_err`) | SNR 30 (BER / NMSE / ECE / BCE / `p95` / `max_llr_err`) |
|---|---|---|---|---|---|
| `clean` | `5.0e-2` | `8.0` | `0.0630 / 0.0176 / 0.0062 / 0.1429 / 8.0 / 1.91e-6` | `0.0236 / 0.0072 / 0.0061 / 0.0592 / 8.0 / 2.38e-6` | `0.0047 / 0.0031 / 0.0011 / 0.0154 / 8.0 / 2.15e-6` |
| `clean` | `2.5e-2` | `20.0` | `0.0599 / 0.0198 / 0.0047 / 0.1330 / 18.9 / 3.81e-6` | `0.0233 / 0.0102 / 0.0028 / 0.0549 / 19.8 / 3.81e-6` | `0.0057 / 0.0064 / 0.0028 / 0.0207 / 19.9 / 4.77e-6` |
| `clean` | `2.5e-2` | `30.0` | `0.0625 / 0.0180 / 0.0071 / 0.1378 / 25.3 / 3.81e-6` | `0.0236 / 0.0084 / 0.0031 / 0.0572 / 28.4 / 4.77e-6` | `0.0065 / 0.0049 / 0.0039 / 0.0237 / 29.3 / 5.72e-6` |
| `clean` | `1.0e-2` | `20.0` | `0.0619 / 0.0195 / 0.0151 / 0.1367 / 19.8 / 4.77e-6` | `0.0239 / 0.0092 / 0.0077 / 0.0575 / 20.0 / 5.72e-6` | `0.0061 / 0.0052 / 0.0041 / 0.0241 / 20.0 / 1.00e-5` (**borderline**) |
| `clean` | `5.0e-3` | `30.0` | `0.0634 / 0.0202 / 0.0180 / 0.1451 / 29.2 / 7.63e-6` | `0.0251 / 0.0080 / 0.0095 / 0.0653 / 30.0 / 8.58e-6` | `0.0061 / 0.0034 / 0.0046 / 0.0359 / 30.0 / 1.24e-5` (**FAIL**) |
| `clean` | `1.0e-3` | `None` | `0.0638 / 0.0196 / 0.0196 / 0.1446 / 76.2 / 6.10e-5` (**FAIL**) | `0.0246 / 0.0087 / 0.0095 / 0.0628 / 211.8 / 1.22e-4` (**FAIL**) | `0.0061 / 0.0046 / 0.0044 / 0.0337 / 865.5 / 2.44e-4` (**FAIL**) |
| `clean` | `adaptive` | `25.0` | `0.0636 / 0.0178 / 0.0073 / 0.1381 / 24.2 / 5.72e-6` | `0.0251 / 0.0075 / 0.0001 / 0.0579 / 25.0 / 5.96e-6` | `0.0070 / 0.0036 / 0.0003 / 0.0184 / 25.0 / 5.72e-6` |
| `strong` | `5.0e-2` | `8.0` | `0.0955 / 0.0202 / 0.0004 / 0.2102 / 7.9 / 1.79e-6` | `0.0468 / 0.0120 / 0.0024 / 0.1131 / 8.0 / 2.15e-6` | `0.0164 / 0.0088 / 0.0014 / 0.0497 / 8.0 / 2.62e-6` |
| `strong` | `2.5e-2` | `20.0` | `0.1006 / 0.0289 / 0.0278 / 0.2343 / 17.9 / 2.86e-6` | `0.0530 / 0.0162 / 0.0203 / 0.1374 / 19.6 / 3.81e-6` | `0.0241 / 0.0113 / 0.0147 / 0.0861 / 19.9 / 4.77e-6` |
| `strong` | `2.5e-2` | `30.0` | `0.1006 / 0.0282 / 0.0247 / 0.2286 / 22.6 / 3.81e-6` | `0.0529 / 0.0173 / 0.0187 / 0.1316 / 27.5 / 4.77e-6` | `0.0217 / 0.0133 / 0.0124 / 0.0780 / 29.3 / 5.72e-6` |
| `strong` | `1.0e-2` | `20.0` | `0.0985 / 0.0272 / 0.0309 / 0.2368 / 19.2 / 4.29e-6` | `0.0504 / 0.0149 / 0.0228 / 0.1399 / 20.0 / 6.68e-6` | `0.0189 / 0.0104 / 0.0134 / 0.0937 / 20.0 / 7.99e-6` |
| `strong` | `5.0e-3` | `30.0` | `0.1000 / 0.0303 / 0.0331 / 0.2408 / 27.2 / 5.72e-6` | `0.0496 / 0.0173 / 0.0238 / 0.1524 / 29.9 / 1.19e-5` (**FAIL**) | `0.0201 / 0.0126 / 0.0162 / 0.1302 / 30.0 / 1.76e-5` (**FAIL**) |
| `strong` | `1.0e-3` | `None` | `0.1007 / 0.0334 / 0.0390 / 0.2612 / 51.8 / 3.05e-5` (**FAIL**) | `0.0509 / 0.0175 / 0.0277 / 0.1735 / 140.4 / 9.16e-5` (**FAIL**) | `0.0199 / 0.0115 / 0.0168 / 0.2154 / 541.1 / 1.22e-4` (**FAIL**) |
| `strong` | `adaptive` | `25.0` | `0.0992 / 0.0269 / 0.0279 / 0.2337 / 23.2 / 5.72e-6` | `0.0527 / 0.0173 / 0.0185 / 0.1392 / 24.9 / 8.58e-6` | `0.0222 / 0.0134 / 0.0100 / 0.0723 / 25.0 / 1.29e-5` (**FAIL**) |

(`flips == 0` for every row above; `adaptive` formula tested was `nv_exp/(hp+1e-4) + 0.25*dp/(hp+1e-4) + 0.015/(hp+0.1) + 1e-3`).

#### B. Fixed-Weights Mixed-PA Sweep (Task 49, 200 mixed clean/mild/strong training steps, 128 eval frames)
- At `fl = 2.5e-2` with `B = 20.0` or `30.0`, `max_llr_err <= 5.72e-06 < 1e-5` across all 15 `(PA, SNR)` combinations (`clean`/`mild`/`strong` × `10..30 dB`).
- At `fl = 1.0e-2` with `B = 20.0` or `30.0`, `strong SNR=30.0 dB` yielded `max_llr_err = 1.07e-05 > 1e-5` (**FAIL**).
- At `fl = 5.0e-3` even with `B = 8.0`, `strong SNR=25/30 dB` and `mild SNR=25 dB` yielded `max_llr_err = 1.13e-05..1.72e-05 > 1e-5` (**FAIL**), proving that `8.0 * tanh` does NOT protect against low-floor boundary error amplification.
- Notice also `dist_pow_med = 0.0001` after proper mini-batch training (Task 49), whereas single-batch overfitting (Task 41) spiked `dist_pow` to `0.0941`.

#### C. Soft-FEC Rate-1/2 Min-Sum LDPC Coded BLER (Tasks 61 & 65, `equal4`)
Task 61 (200 blocks × 256 bits, `strong equal4`) and Task 65 (300 blocks × 256 bits, `equal4 clean/mild/strong`):

| Configuration (`floor`, `B`) | `clean` 13 dB | `clean` 16 dB | `mild` 14 dB | `mild` 17 dB | `strong` 14 dB | `strong` 17 dB | `strong` 19 dB |
|---|---|---|---|---|---|---|---|
| `5.0e-2, B=8.0` (current) | `0.4533` | `0.2033` | `0.4033` | `0.2033` | `0.6750` | `0.4250` | `0.2667` |
| **`2.5e-2, B=20.0`** | **`0.4433`** | **`0.1900`** | **`0.3867`** | **`0.1900`** | **`0.6550`** | `0.4150` | **`0.2567`** |
| **`2.5e-2, B=30.0`** | **`0.4433`** | **`0.1900`** | **`0.3867`** | **`0.1900`** | **`0.6550`** | `0.4150` | **`0.2567`** |
| `2.0e-2, B=20.0` | `0.4400` | `0.1867` | `0.3867` | `0.1900` | — | — | `0.2667` |
| `1.0e-2, B=20.0` | `0.4333` | `0.1933` | `0.3867` | `0.1933` | `0.6650` | **`0.4050`** | `0.2700` |
| `5.0e-3, B=30.0` | `0.4267` | `0.1933` | `0.3900` | `0.1933` | `0.6650` | **`0.4050`** | `0.2800` |
| `Bussgang-LMMSE` | `0.4433` | `0.1800` | `0.4100` | `0.1833` | `0.6800` | `0.4200` | `0.2533` |

---

## 2. Logic Chain

1. **Why Iteration 3 over-tightened `2.5e-2 / 20.0` to `5.0e-2 / 8.0`**: In Iteration 2 (`challenger_m2_3/handoff.md:23, 63`), `EMFloatFormer` ran its Transformer backbone in `complex64`, introducing `~5e-7` drift in `z_eq` that caused `max_llr_err = 8.011e-05` under `2.5e-2` floor. Moving the backbone to `complex128`/`float64` (`p4/em_floatformer.py:595-642`) already eliminated backbone drift (`challenger_m2_3/handoff.md:29` and Task 57 confirm `<= 4.77e-6` at `2.5e-2 / 20.0`). The extra tightening to `5.0e-2 / 8.0` was redundant for equivariance and overly aggressive for LLR dynamic range.
2. **Mathematical lower bound on `sigma_eff2` for `< 1e-5` C* equivariance under `complex64` inputs**:
   - Even with an exact `complex128` backbone, a caller passing `(yp * c, yd * c, nv * r^2)` in `complex64` injects $\Delta_{\text{in}} \approx 10^{-7}$ relative rounding error before `forward()`.
   - Near 16-QAM decision boundaries (`LLR ≈ 0`), `d/dx [B * tanh(x / B)]|_{x=0} = 1`, so `tanh` provides zero attenuation there. Since `calibrated_16qam_llrs` has slope $\approx [1.265 / \sigma_{\text{eff}}^2, 2.530 / \sigma_{\text{eff}}^2]$ (`p4/classical_baselines.py:185`), maintaining `max_llr_err < 1e-5` across large `complex64` batches requires $\sigma_{\text{eff}}^2 \gtrsim 2.53 \times 10^{-7} / 10^{-5} \approx 2.5 \times 10^{-2}$.
   - Tasks 49 & 57 confirm this threshold empirically: `floor = 2.5e-2` stays at `<= 5.72e-6 < 1e-5` everywhere, whereas `1.0e-2`, `5.0e-3`, `1.0e-3`, and `adaptive` all breach `1e-5` at 20–30 dB (`1.00e-5` to `2.44e-4`).
3. **Mathematical upper bound on `tanh` saturation `B` from `float32` ULP + PA error floors**:
   - Output `llrs` is cast to `float32` (`line 642`). In `float32`, $\text{ULP}(32) = 3.815 \times 10^{-6} < 10^{-5}$, $\text{ULP}(64) = 7.629 \times 10^{-6}$, and $\text{ULP}(128) = 1.526 \times 10^{-5} > 10^{-5}$. Thus unbounded `llrs` (`|LLR| > 100`) at 20–30 dB automatically violate `< 1e-5` purely from `float32` representation quantization (seen as exact power-of-2 errors `3.05e-5, 6.10e-5, 1.22e-4, 2.44e-4` in Tasks 45 & 57).
   - Furthermore, under `strong` PA (`sigma_d2 / |gamma|^2 = 0.0361`), `~1.6%–2.4%` of bits remain wrong even at 30 dB; without saturation (`B=None`) or with too small a floor (`1e-3`), wrong bits receive `|LLR| ~ 500`, inflating BCE (`0.2154` vs `0.078–0.086` at `2.5e-2 / 20..30`) and hurting LDPC BLER (`0.6800` vs `0.6550` at 14 dB, Task 61).
4. **Why `2.5e-2` + `20.0 * tanh(llrs64 / 20.0)` (or `25.0`) resolves the Stage P4-B tension**:
   - Raising `B` from `8.0` to `20.0` (or `25.0`) restores 2.5×–3.1× LLR dynamic range (`p95|LLR|` increases from `8.0` to `18.9–24.9`), eliminates the severe compression of moderate LLRs (`8 * tanh(8/8) = 6.09` vs `20 * tanh(8/20) = 7.60`), preserves subcarrier reliability contrast in Min-Sum LDPC (`0.6550` vs `0.6750` BLER at 14 dB `strong equal4`, outperforming Bussgang `0.6800`), lowers clean-PA 15–20 dB ECE/BCE (`0.0047/0.1330` and `0.0028/0.0549` vs `0.0062/0.1429` and `0.0061/0.0592`), and maintains `max_llr_err <= 4.77e-6 < 1e-5` (`<= 5.72e-6` for `B=25..30`) with `0` bit flips.

---

## 3. Caveats

- All measurements used `EMFloatFormer(depth=2)` trained for 150–200 Adam steps on CPU (`torch.set_num_threads(4)`); deeper/longer-trained checkpoints or Stage P4-B 2D slots (`p4/slot_2d.py`, not yet implemented) could have slightly different residual `h_nmse` and `dist_pow`.
- In `p4/em_floatformer.py:640`, `dist_power_det / (h_power_det + 1e-4)` uses the magnitude of `dist_bar` (which was subtracted in `z_eq` at line `635`) as a proxy for residual PA disturbance; because trained `dist_pow` stays `~1e-4` on mini-batch training (Task 49), it does not inflate `sigma_eff2`, though a single-batch overfit (Task 41) can temporarily inflate `dist_pow`.
- Stage P4-B 2D slot module (`p4/slot_2d.py`) is not yet implemented; soft-FEC LDPC BLER was evaluated using a self-contained Rate-1/2 systematic regular LDPC (`K=128, N=256`, 20-iteration normalized Min-Sum) over `equal4` and `sparse12` frames.

---

## 4. Conclusion

**Recommendation for `p4/em_floatformer.py:640-642`:**
Replace the over-tightened Iteration 3 constants (`5.0e-2` and `8.0 * tanh(llrs64 / 8.0)`) with:
```python
sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 2.5e-2
llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)
```
*(Note: `B = 25.0` is also safe at `max_llr_err <= 5.72e-6 < 1e-5`, while `B = 20.0` gives a wider `2.1x` safety margin below `1e-5` and matches `float32` ULP `<= 1.907e-6`, with `ECE <= 0.0278 < 0.05` across all PA/SNR regimes and improved soft-FEC LDPC coded BLER.)*

Do **not** drop the `sigma_eff2` floor below `2.0e-2`–`2.5e-2` or use `B > 32` / unbounded `float32` LLRs while `max_llr_err < 1e-5` is enforced on `complex64` scaled inputs, as `complex64` caller quantization and `float32` ULP rounding will breach `1e-5` at 20–30 dB.

---

## 5. Verification Method

After applying the edit to `p4/em_floatformer.py:640-642`, run from ``:
1. Existing unit & E2E suites:
   ```powershell
   python -m unittest p4.tests.test_p4_0_simulator -v
   python -m unittest p4.tests.test_p4_a_equivariance -v
   python -m unittest tests.test_p4_e2e -v
   ```
2. Trained `depth=2` multi-SNR/multi-PA C* equivariance + ECE check:
   ```powershell
   python -c "
   import torch
   torch.set_num_threads(4)
   from p4.em_floatformer import EMFloatFormer, JointPhysicalLoss
   from p4.simulator import generate_frame
   torch.manual_seed(11)
   m = EMFloatFormer(depth=2)
   opt = torch.optim.Adam([
       {'params': [p for n, p in m.named_parameters() if 'gamma_eff' not in n], 'lr': 1e-2},
       {'params': m.gamma_eff.parameters(), 'lr': 3e-2},
   ])
   lf = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
   g_tr = torch.Generator().manual_seed(101)
   for s in range(150):
       tf = generate_frame(32, 32, snr_db=(10.0, 25.0), pa_condition=['clean','mild','strong'][s%3], generator=g_tr)
       opt.zero_grad(); lf(m(tf.yp, tf.yd, tf.noise_var), tf).backward(); opt.step()
   m.eval()
   with torch.no_grad():
       for pa in ['clean', 'strong']:
           for snr in [15.0, 20.0, 30.0]:
               fr = generate_frame(256, 32, snr_db=snr, pa_condition=pa, generator=torch.Generator().manual_seed(31415))
               o1 = m(fr.yp, fr.yd, fr.noise_var)
               probs = torch.sigmoid(o1['llrs']); preds = (probs > 0.5).long()
               ece = abs((preds == fr.bits).float().mean().item() - probs.where(preds==1, 1.0-probs).mean().item())
               assert ece < 0.05, f'{pa} {snr} ECE={ece}'
               for r, phi in [(2.0**-12, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)]:
                   c = r * torch.exp(1j * torch.tensor(phi))
                   o2 = m(fr.yp * c, fr.yd * c, fr.noise_var * (r ** 2))
                   err = (o1['llrs'] - o2['llrs']).abs().max().item()
                   flips = ((o1['llrs'] > 0) != (o2['llrs'] > 0)).sum().item()
                   assert err < 1e-5 and flips == 0, f'{pa} {snr} ({r},{phi}): err={err}, flips={flips}'
   print('PASS: all ECE < 0.05, max_llr_err < 1e-5, flips == 0')
   "
   ```
- **Invalidation condition**: Any `(r, phi)` in `r in [2^-12, 2^12]` producing `max_llr_err >= 1e-5` or `flips > 0`, `ECE >= 0.05`, or regression in `compare_pilot_vs_bidirectional()` (`>= 0.5 dB`).
