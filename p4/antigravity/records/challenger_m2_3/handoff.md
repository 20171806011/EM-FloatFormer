# Milestone 2 Challenger 3 Handoff Report (`challenger_m2_3`)

**Verdict: `REQUEST_CHANGES`**

## 1. Observation

### 1.1 Existing Test Suites Executed Independently
1. `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 45 tests in 249.739s — OK** (exit code 0).
2. `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 269.020s — OK (skipped=97)** (156 active passed, 97 M3/M4 skipped, exit code 0).

### 1.2 Empirical Adversarial Stress Test Findings

#### Bug 1 [HIGH — Blocking]: `EMFloatFormer(depth=2)` Violates `< 1e-5` Max Absolute LLR Error Under Active / Trained Weights (and Fresh `seed=11`)
- **Fresh untrained `EMFloatFormer(depth=2)`** (`torch.manual_seed(11)`, `yp/yd` `seed=42`, `nv=0.01`):
  - `max_h_rel = 1.470e-07`, `max_llr_err = 1.049e-05` (`> 1e-5`) at `(r=0.0625, phi=2.041200876235962)`, `total_flips = 0`.
- **Trained `EMFloatFormer(depth=2)`** (50 Adam steps on `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`, resulting in active non-zero heads `head_h=0.0782`, `head_dist=0.1198`, `gamma_eff=(0.9052, 0.0486)`):
  - `randn` (`4x32`, `nv=0.01`): `max_llr_err = 1.526e-05` (`r=2^-12, phi=0.95`), `total_flips = 0`.
  - `frame20` (`8x32`, `snr=20dB`, strong PA): `max_llr_err = 2.766e-05` (`r=2^-10, phi=-0.6076`), `total_flips = 0`.
  - `frame30` (`8x32`, `snr=30dB`, clean PA): `max_llr_err = 4.768e-05` (`r=256.0, phi=2.5356`), `total_flips = 0`.
  - `frame30` (`32x32`):
    - Pure scale `r=1023.5` (zero phase rotation): `max_llr_err = 4.005e-05`.
    - Pure phase `phi=2.5356` (zero gain scaling): `max_llr_err = 2.575e-05`.
  - `frame30` (`256x32`, strong PA): `min |LLR| = 1.691e-05`, `max_llr_err = 8.011e-05` at `(r=0.13, phi=-1.7000)`, `6.962e-05` at `(r=1023.5, phi=2.5356)`, `6.199e-05` at `(r=3.7, phi=0.95)`, `4.768e-05` at `(r=4096.0, phi=-0.6076)`, `3.910e-05` at `(r=2^-12, phi=2.7183)`, `flips = 0`.
  - `frame25` (`1024x32`, strong PA): `max_llr_err = 5.579e-05`, `total_flips = 0`.
- **Boosted O(1)-head `EMFloatFormer(depth=2)`** (`head_h=0.9551, head_dist=0.4218, gamma_eff=0.85+0.15j`):
  - `randn`: `max_llr_err = 5.770e-05`, `total_flips = 0`.
  - `frame20`: `max_llr_err = 6.771e-05`, `total_flips = 0`.
- **Precision Root-Cause Verification**: Converting the exact same trained `m_trained` to `.double()` (`complex128`/`float64`) on `frame30[:16]` with `(r=1023.5, phi=2.5356)`:
  - With `complex64` caller rounding on `(yp*c, yd*c)`: `max_llr_err = 4.768e-06 < 1e-5`.
  - With `complex128` exact scaling: `max_llr_err = 0.000e+00`.
- **Channel estimation relative error**: `max_h_rel` remained `< 1.12e-6 < 1e-5` across all tests above.

#### Bug 2 [HIGH — Blocking]: `compute_gauge_anchor` / `apply_gauge_normalization` Collapse `1e-35` Inputs Instead of Producing Unit-RMS Outputs
- In `p4/em_floatformer.py:33-58`, `compute_gauge_anchor(yp, eps=1e-12)` executes `s0 = torch.clamp(m * rms_scaled, min=eps)`.
- Empirical result on `yp = torch.full((2, 32), 1e-35, dtype=torch.complex64)` (normal `float32` value `> 1.175e-38`):
  - `s0 = [9.999999960041972e-13, 9.999999960041972e-13]` (clamped to `1e-12` instead of `1e-35`).
  - `yp_bar, yd_bar, _ = apply_gauge_normalization(yp, yp)` returns `rms_bar = [0.0, 0.0]` (`~1e-23`), failing unit-RMS normalization by 23 orders of magnitude.

#### Bug 3 [HIGH — Blocking]: `ComplexRMSNorm` Produces `NaN+NaNj` Backward Gradients at `1e-35`
- In `p4/em_floatformer.py:79-90`:
  - Forward on `z_sub = torch.full((2, 32), 1e-35, dtype=torch.complex64, requires_grad=True)` (and `1e-35 - 1e-35j`) produces finite unit-RMS `[1.0, 1.0]`.
  - Backward via `(out.real.sum() + out.imag.sum()).backward()` produces:
    - `z_sub.grad finite: False sample: tensor([nan+nanj, nan+nanj, nan+nanj, nan+nanj])`.

#### Bug 4 [MEDIUM — Blocking]: `RadialActivation` Overflows to `NaN` Forward & Backward at `1e20` Complex Inputs
- In `p4/em_floatformer.py:244-258`, `mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + self.eps ** 2)`.
- Empirical result on `z = torch.complex(torch.full((2, 4), 1e20), torch.full((2, 4), 1e20))` (`complex64`):
  - `RadialActivation(silu)[huge_1e20]: out_finite=False, grad_finite=False`
  - `RadialActivation(gelu)[huge_1e20]: out_finite=False, grad_finite=False`
  - `RadialActivation(relu)[huge_1e20]: out_finite=False, grad_finite=False`
  - (`tanh` remained finite only because `tanh(inf) = 1.0`, though `scale = 1 / inf = 0`.)

#### Observation 5 [PASS on Full Model / LOW on Ablation]: Step-1 Gradients with `JointPhysicalLoss`
- `EMFloatFormer(depth=1)` (`58` tensors), `depth=2` (`104` tensors), `depth=4` (`196` tensors): **0** missing/zero/non-finite gradients on step 1 with `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`.
- Gradient equivariance on `trained_d2` at `(0.13, -1.7)`, `(3.7, 0.95)`, `(1023.5, 2.5)`: `max_diff <= 2.757e-07 < 1e-5`.
- Minor hygiene gap (`LOW`): `create_floatformer(use_joint_pilot_data=False)` leaves `blocks.*.norm_p2.gamma` and `blocks.*.norm_d2.gamma` unused with `grad = None`.

---

## 2. Logic Chain

1. **Why existing tests missed Bug 1**: Almost all equivariance checks in `p4/em_floatformer.py` (`audit_gain_equivariance`, `audit_phase_equivariance`, `audit_c_star_equivariance`, `sweep_phases_dense`, `audit_full_equivariance`) instantiate an untrained `EMFloatFormer(depth=1)` whose `head_h`/`head_dist` weights are initialized to `std ≈ 0.002`. The single `depth=2` test (`test_non_power_of_two_scale_phase_active_backbone`) checked only one seed and one `(r=3.7, phi=0.95)` on that near-zero head init.
2. **Mechanism of Bug 1**: `apply_gauge_normalization` (`p4/em_floatformer.py:56-57`) casts `yp_bar, yd_bar` back to `complex64`, and the entire Transformer (`embed`, `blocks`, `norm_p`, `norm_d`, `head_h`, `head_dist`, `gamma_eff`) runs in `complex64`/`float32`. Under `(r, phi)` scaling/rotation, `complex64` rounding + 2 layers of `ComplexRMSNorm` / `GreenAttention` / `RadialActivation` introduce `~5e-7` drift in `h_bar` and `dist_bar`. Once `head_h`, `head_dist`, and `gamma_eff` have active trained magnitudes, `calibrated_16qam_llrs` amplifies that `~5e-7` `z_eq` drift by `~40x–160x` (since `sigma_eff2 ~ 0.025`), producing `1.5e-5` to `8.011e-05` LLR errors (exceeding `< 1e-5` by 8x, and exceeding `min |LLR| = 1.691e-05` on the 256-frame batch). Running the exact same trained weights in `complex128`/`float64` eliminates the backbone drift and reduces `max_llr_err` to `4.768e-06 < 1e-5` even when inputs are scaled in `complex64`.
3. **Mechanism of Bug 2**: `eps=1e-12` in `compute_gauge_anchor` (`p4/em_floatformer.py:44`) is 23 orders of magnitude above `1e-35`, clamping `s0` to `1e-12` and collapsing `yp_bar` to `1e-23` instead of unit RMS. Using `eps=1e-38` (or `torch.where(m > 0, m * rms_scaled, eps)`) preserves unit-RMS down to `float32` normal floor `1.175e-38`.
4. **Mechanism of Bug 3**: In `ComplexRMSNorm.forward` (`p4/em_floatformer.py:88-90`), multiplying `m` back into `rms = m * rms_scaled` yields `scale = gamma / rms ≈ 1e35` for `z = 1e-35`; PyTorch autograd computes `1 / rms^2 ≈ 1e70` in `float32`, overflowing to `inf` and producing `inf - inf = NaN` in `z.grad`. Keeping the normalization factored as `z_scaled * (self.gamma / rms_scaled)` (where `rms_scaled ~ 1.0`) or executing `ComplexRMSNorm` in `float64`/`complex128` eliminates the `1 / m^2` overflow.
5. **Mechanism of Bug 4**: `RadialActivation.forward` (`p4/em_floatformer.py:245`) computes `z.real ** 2 + z.imag ** 2` directly in `float32`, which overflows to `inf` for `|z| > 1.84e19` (`1e20`), yielding `inf / inf = NaN` for `silu`, `gelu`, and `relu`. Using `amax` rescaling or `torch.hypot` in `float64` prevents overflow across the full `float32` range.

---

## 3. Caveats

- When calling `EMFloatFormer(yp * 1e20, yd * 1e20, nv * 1e40)` in `float32`, `nv * (1e20 ** 2)` itself overflows `float32` (`1e40 > 3.4e38`) at the caller before entering the model; therefore full-model singular stress at `1e-35` / `1e20` is properly evaluated on the normalization/activation modules (`ComplexRMSNorm`, `compute_gauge_anchor`, `apply_gauge_normalization`, `RadialActivation`) and within representable `nv` range (`r in [2^-12, 2^12]` and up to `1e10`).
- Only CPU execution (`torch.set_num_threads(4)`) was tested; mixed-precision (`FP16`/`BF16`) is scheduled for Milestone 4.

---

## 4. Conclusion

**Verdict: `REQUEST_CHANGES` (Blocking).**

While the existing 45 unit tests and 156 active E2E tests pass and step-1 gradients are healthy on the full model, empirical adversarial testing disproves `C*` LLR equivariance and singular-input stability under active/trained and extreme conditions:
1. Trained and boosted-head `EMFloatFormer(depth=2)` models (and even fresh `depth=2` on `seed=11`) fail `< 1e-5` max LLR error (reaching **`8.011e-05`**) under $r \in [2^{-12}, 2^{12}]$ and non-power-of-2 scales (`0.13, 3.7, 1023.5`). **Recommended fix**: run `EMFloatFormer.forward` backbone (or `apply_gauge_normalization` through `head_h`/`head_dist`/`gamma_eff`) in `complex128`/`float64` (cast output tensors `h_hat`, `dist_est`, `llrs` back to caller dtype at the boundary), and add unit tests covering trained / O(1)-head `depth=2` models across 24 octaves + non-power-of-2 scales.
2. Fix `compute_gauge_anchor` (`eps` floor `1e-12` -> `1e-38` / zero-only guard) so `1e-35` inputs normalize to unit RMS (`1.0`) rather than collapsing to `0.0`.
3. Fix `ComplexRMSNorm` backward `NaN` at `1e-35` (avoid `1 / rms^2` overflow via factored `z_scaled * (gamma / rms_scaled)` or `float64` internal computation).
4. Fix `RadialActivation` `1e20` overflow/NaN (`amax` rescaling or `float64` norm computation) and test `ComplexRMSNorm` backward at `1e-35` and `RadialActivation` forward/backward at `1e20` in `test_p4_a_equivariance.py`.
5. (Low/Optional) Guard `norm_p2`/`norm_d2` instantiation in `BidirectionalBlock` with `if use_joint_pilot_data else None` so all parameters receive gradients in ablation mode too.

---

## 5. Verification Method

From ``:
1. Reproduce Bug 1 (trained `EMFloatFormer(depth=2)` LLR error `> 1e-5`):
```powershell
python -c "
import math, torch
torch.set_num_threads(4)
from p4.em_floatformer import EMFloatFormer, JointPhysicalLoss
from p4.simulator import generate_frame
torch.manual_seed(11)
m = EMFloatFormer(depth=2)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
lf = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
tf = generate_frame(batch_size=16, n=32, snr_db=18.0, pa_condition='strong', generator=torch.Generator().manual_seed(99))
for _ in range(50):
    opt.zero_grad(); lf(m(tf.yp, tf.yd, tf.noise_var), tf).backward(); opt.step()
m.eval()
fr = generate_frame(batch_size=256, n=32, snr_db=30.0, pa_condition='strong', generator=torch.Generator().manual_seed(31415))
with torch.no_grad():
    o1 = m(fr.yp, fr.yd, fr.noise_var)
    for r, phi in [(0.000244140625, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)]:
        c = r * torch.exp(1j * torch.tensor(phi))
        o2 = m(fr.yp * c, fr.yd * c, fr.noise_var * (r ** 2))
        err = (o1['llrs'] - o2['llrs']).abs().max().item()
        flips = ((o1['llrs'] > 0) != (o2['llrs'] > 0)).sum().item()
        assert err < 1e-5 and flips == 0, f'FAILED at r={r}, phi={phi}: err={err}, flips={flips}'
"
```
2. Reproduce Bugs 2, 3, 4 (`1e-35` anchor collapse, `ComplexRMSNorm` `NaN` backward at `1e-35`, `RadialActivation` `NaN` at `1e20`):
```powershell
python -c "
import torch
from p4.em_floatformer import ComplexRMSNorm, RadialActivation, apply_gauge_normalization
yp_sub = torch.full((2, 32), 1e-35, dtype=torch.complex64)
yp_bar, _, _ = apply_gauge_normalization(yp_sub, yp_sub)
rms = (yp_bar.abs() ** 2).mean(dim=-1).sqrt()
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f'Gauge 1e-35 collapsed: {rms}'
z_sub = torch.full((2, 32), 1e-35, dtype=torch.complex64, requires_grad=True)
out_n = ComplexRMSNorm(32)(z_sub)
(out_n.real.sum() + out_n.imag.sum()).backward()
assert torch.isfinite(z_sub.grad).all(), f'ComplexRMSNorm NaN grad at 1e-35: {z_sub.grad[0, :2]}'
z_huge = torch.complex(torch.full((2, 4), 1e20), torch.full((2, 4), 1e20))
out_r = RadialActivation('silu')(z_huge)
assert torch.isfinite(out_r).all(), f'RadialActivation NaN at 1e20: {out_r}'
"
```
3. Full regression suites:
```powershell
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
```
Invalidation conditions: remediation passes when all 3 verification commands above exit 0 with `max_llr_err < 1e-5`, `flips == 0`, finite gradients/outputs on `1e-35` and `1e20`, and all unit/E2E tests passing.
