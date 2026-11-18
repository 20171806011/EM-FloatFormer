# Handoff Report — `explorer_m2_r2_2` (Milestone 2 / Stage P4-A Iteration 2)

**Summary:** Verified both `reviewer_m2_r1_2` reproducibility defects (`run_p4a_comparative_benchmarks` constructing models before `torch.manual_seed(seed)` in `_train_comparative_model`, and unseeded `EMFloatFormer(depth=1)` in `evaluate_noise_data_feedback` / `evaluate_perfect_pilot_demapping`), cataloged all 15 additional unseeded `nn.Module` audit helpers in `p4/em_floatformer.py`, designed exact bit-for-bit reproducible fixes, and empirically isolated why `LocalEquivConv16QAM` (`BER=0.0453`) leads `EMFloatFormer` (`BER=0.0503`–`0.0508`) on 1D dense-pilot `equal4` (not an LR/step-count/cosine-schedule artifact; without `dist_est`, `EMFloatFormer` leads `0.0508` vs `0.0541`, whereas circular `ComplexConv1d` learns an active `dist_est` `dist_rms=0.0848` vs `0.0168` and lower channel `h_MSE=0.0254` vs `0.0401` on smooth `equal4` with 100% pilot density).

---

## 1. Observation

### 1.1 Finding 1 (Major): Pre-Seeding Model Instantiation in `run_p4a_comparative_benchmarks`
Inspected `p4/em_floatformer.py:1588-1795`:
- `_train_comparative_model` (`lines 1588-1644`) accepts an already-instantiated `model: nn.Module` (`line 1589`) and only calls `torch.manual_seed(seed)` at `line 1600`, followed by `gen = torch.Generator().manual_seed(seed + 1000)` at `line 1601`.
- During the training loop (`lines 1620-1642`), `generate_frame(..., generator=gen)` uses the isolated `gen`, and `torch.optim.Adam` / `CosineAnnealingLR` / `clip_grad_norm_` consume zero global PyTorch RNG.
- In `run_p4a_comparative_benchmarks` (`lines 1647-1795`):
  1. `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=n_sc)` (`line 1662`) is constructed **before** any `torch.manual_seed(...)` call, consuming whatever ambient global PyTorch RNG state exists on entry (`EMFloatFormer.__init__` calls `ComplexLinear` / `GreenAttention` `torch.randn_like` at `lines 306-307` and `reset_parameters` `nn.init.normal_` at `lines 570-578`).
  2. `LocalEquivConv16QAM(width=14, kernel_size=5, depth=2, ffn_dim=12)` (`line 1673`) is constructed after `_train_comparative_model(emff, ..., seed=seed)` finishes; because `emff` training consumed zero global RNG after `torch.manual_seed(seed)` (`line 1600`), `LocalEquivConv16QAM` accidentally inherits `seed` (`2026`) rather than `seed + 1` (`2027`).
  3. `TinyJoint16QAM(width=16, taps=8)` (`line 1684`) similarly inherits `seed + 1` (`2027`) rather than `seed + 2` (`2028`).
- Parameter counts verified (`python -c`): `EMFloatFormer = 14,090`, `LocalEquivConv16QAM = 14,198`, `TinyJoint16QAM = 1,880`.

### 1.2 Finding 2 (Minor) & Complete Inventory of Unseeded `nn.Module` Sites in `p4/em_floatformer.py`
Inspected all `1,795` lines of `p4/em_floatformer.py` and verified caller usages in `p4/tests/` and `tests/e2e/`:
- **Finding 2 sites**:
  - `evaluate_noise_data_feedback` (`p4/em_floatformer.py:727-750`): instantiates `model = EMFloatFormer(depth=1)` at `line 733` *before* `g = torch.Generator().manual_seed(42)` (`line 735`), with no `torch.manual_seed(...)`.
  - `evaluate_perfect_pilot_demapping` (`p4/em_floatformer.py:753-770`): instantiates `model = EMFloatFormer(depth=1)` at `line 759` *before* `g = torch.Generator().manual_seed(99)` (`line 761`), with no `torch.manual_seed(...)`.
  - Measured variation across ambient seeds `[0, 1, 7, 123, 999]` (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`):
    - `evaluate_noise_data_feedback`: `[0.000465, 0.001589, 0.005297, 0.001703, 0.002025]` (11.4x spread).
    - `evaluate_perfect_pilot_demapping`: `[0.0, 0.0, 0.000244, 0.000244, 0.0]`.
  - Measured with explicit deterministic seeding (`torch.manual_seed(42)` for `evaluate_noise_data_feedback`, `torch.manual_seed(99)` for `evaluate_perfect_pilot_demapping`):
    - `deg_db(seed=42) = 0.001448` (`0.0 < deg_db < 1.0`: `True`).
    - `ber(seed=99) = 0.000244` (`ber < 1e-3`: `True`).
- **Additional unseeded `nn.Module` instantiations in `p4/em_floatformer.py`** (15 audit/verification helpers; `compare_pilot_vs_bidirectional` at `line 677` already calls `torch.manual_seed(42)`):
  1. `audit_single_scale` (`line 113`): `EMFloatFormer()`
  2. `audit_single_phase` (`line 135`): `EMFloatFormer()`
  3. `verify_loss_gradient_invariance` (`line 908`): `EMFloatFormer(depth=1)`
  4. `audit_loss_scale_invariance` (`line 955`): `EMFloatFormer(depth=1)`
  5. `train_step_joint_loss` (`line 992`): `EMFloatFormer(depth=1)`
  6. `audit_gain_equivariance` (`line 1027`): `EMFloatFormer(depth=1)`
  7. `audit_phase_equivariance` (`line 1053`): `EMFloatFormer(depth=1)`
  8. `audit_c_star_equivariance` (`line 1080`): `EMFloatFormer(depth=1)`
  9. `audit_channel_equivariance` (`line 1111`): `EMFloatFormer(depth=1)`
  10. `sweep_phases_dense` (`line 1153`): `EMFloatFormer(depth=1)`
  11. `check_irrational_phases` (`line 1177`): `EMFloatFormer(depth=1)`
  12. `check_noise_quad_scaling` (`line 1200`): `EMFloatFormer(depth=1)`
  13. `verify_gauge_attention_joint` (`line 1217`): `EMFloatFormer(depth=1)`
  14. `audit_full_equivariance` (`line 1242`): `EMFloatFormer(depth=1)`
  15. `verify_bidirectional_green_prop` (`line 1272`): `BidirectionalBlock(d_model=32, n_heads=2)`
- Verified E2E references (`tests/e2e/test_tier{1,2,3}*.py`) and unit tests (`p4/tests/test_p4_a_equivariance.py`): all call these helpers with zero arguments (except `audit_single_scale(scale)`, `audit_single_phase(phase)`, `sweep_phases_dense(num_points)`, etc.) and none assert RNG side effects across calls.

### 1.3 Empirical Analysis of `EMFloatFormer` vs `LocalEquivConv16QAM` (`p4a_comparative_benchmark.csv`)
Executed read-only ablation/diagnostic runs on `CPU` (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`), evaluating on `eval_seed=202602` (`256` frames, `n_sc=32`, `profile='equal4'`, `pa_condition='strong'`, `snr_db=20.0`, plus `clean 20 dB` `eval_seed=202622` and `clean 30 dB` `eval_seed=202624`):

1. **LR / Step-Count / Cosine Schedule Sweep (`strong 20 dB`, `EMFloatFormer` seeded `2026`, `LocalEquivConv16QAM` seeded `2027`)**:
   - `EMFloatFormer` (`backbone_lr=8e-3, gamma_lr=2e-2, steps=240`): `BER=0.0503, NMSE=0.0408, BCE=0.1239, ECE=0.0129, dist_rms=0.0168, sr=0.9809, gamma=0.923+0.169j`
   - `EMFloatFormer` (`backbone_lr=5e-3, gamma_lr=1.5e-2, steps=240`): `BER=0.0527, NMSE=0.0417, BCE=0.1274, ECE=0.0153, dist_rms=0.0212, sr=0.9793`
   - `EMFloatFormer` (`backbone_lr=3e-3, gamma_lr=1e-2, steps=240`): `BER=0.0540, NMSE=0.0403, BCE=0.1334, ECE=0.0171, dist_rms=0.0300, sr=0.9806`
   - `EMFloatFormer` (`backbone_lr=8e-3, gamma_lr=2e-2, steps=480`): `BER=0.0501, NMSE=0.0357, BCE=0.1215, ECE=0.0131, dist_rms=0.0096, sr=0.9789`
   - `LocalEquivConv16QAM` (`backbone_lr=3e-3, gamma_lr=1e-2, steps=240`): `BER=0.0453, NMSE=0.0260, BCE=0.1046, ECE=0.0055, dist_rms=0.0848, sr=0.9986, gamma=0.970+0.157j`
   - `LocalEquivConv16QAM` (`backbone_lr=5e-3, gamma_lr=1.5e-2, steps=240`): `BER=0.0452, NMSE=0.0285, BCE=0.1046, ECE=0.0064, dist_rms=0.0781, sr=0.9911`
   - `LocalEquivConv16QAM` (`backbone_lr=8e-3, gamma_lr=2e-2, steps=240`): `BER=0.0456, NMSE=0.0287, BCE=0.1073, ECE=0.0033, dist_rms=0.1394, sr=0.9920`
   - `LocalEquivConv16QAM` (`backbone_lr=3e-3, gamma_lr=1e-2, steps=480`): `BER=0.0447, NMSE=0.0260, BCE=0.1037, ECE=0.0070, dist_rms=0.0438, sr=0.9946`

2. **Clean vs Strong Error Decomposition & Distortion (`dist_est`) Ablation**:
   - At `strong 20 dB` (`eval_seed=202602`):
     - `EMFloatFormer`: `ber_out=0.0503`, **`ber_noDist=0.0508`**, `ber_wDist=0.0504` (`fade20%=0.1638`, `top80%=0.0219`), `h_MSE=0.0401`, `dist_rms=0.0168`.
     - `LocalEquivConv16QAM`: `ber_out=0.0453`, **`ber_noDist=0.0541`**, `ber_wDist=0.0454` (`fade20%=0.1507`, `top80%=0.0189`), `h_MSE=0.0254`, `dist_rms=0.0848`.
   - At `clean 20 dB` (`eval_seed=202622`) and `clean 30 dB` (`eval_seed=202624`):
     - `clean 20 dB`: `EMFloatFormer` `h_MSE=0.0396 < Local 0.0603`, `symMSE_wDist=0.0841 < Local 0.1418`, yet `BER=0.0282` vs `Local 0.0250` (`dist_rms`: `0.0203` vs `0.1004`).
     - `clean 30 dB`: `EMFloatFormer` `h_MSE=0.0377 < Local 0.0621`, `symMSE_wDist=0.0224` vs `Local 0.0201`, `BER=0.0059` vs `Local 0.0042` (`dist_rms`: `0.0213` vs `0.1030`).

---

## 2. Logic Chain

1. **Why Finding 1 broke reproducibility for `EMFloatFormer` only**: Because `run_p4a_comparative_benchmarks` (`lines 1661-1692`) instantiates all three models as eager arguments before `_train_comparative_model` calls `torch.manual_seed(seed)` (`line 1600`), `EMFloatFormer` consumes unseeded ambient RNG. Subsequent training consumes zero global RNG (`generate_frame` uses `gen`), so `LocalEquivConv16QAM` and `TinyJoint16QAM` deterministically (but unintentionally) inherit `seed` (`2026`) and `seed + 1` (`2027`) in a fresh process.
2. **Exact Fix Design for Finding 1 & Finding 2**:
   - **Finding 1 (`_train_comparative_model` + `run_p4a_comparative_benchmarks`)**:
     Refactor `_train_comparative_model` (`p4/em_floatformer.py:1588-1602`) to accept `model_or_factory: Union[nn.Module, Callable[[], nn.Module]]`, invoke `torch.manual_seed(seed)` **before** instantiating the factory, and pass lambdas (`lambda: EMFloatFormer(...)`, `lambda: LocalEquivConv16QAM(...)`, `lambda: TinyJoint16QAM(...)`) from `run_p4a_comparative_benchmarks`:
     ```python
     # In _train_comparative_model (lines 1588-1603):
     def _train_comparative_model(
         model_or_factory: Union[nn.Module, Callable[[], nn.Module]],
         ...
     ) -> Tuple[nn.Module, List[float]]:
         torch.manual_seed(seed)
         model = (
             model_or_factory
             if isinstance(model_or_factory, nn.Module)
             else model_or_factory()
         )
         gen = torch.Generator().manual_seed(seed + 1000)
         model.train()
     ```
     And in `run_p4a_comparative_benchmarks` (`lines 1661-1692`):
     ```python
     emff, emff_scales = _train_comparative_model(
         lambda: EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=n_sc),
         ...,
         seed=seed,
         ...,
     )
     local_conv, local_scales = _train_comparative_model(
         lambda: LocalEquivConv16QAM(width=14, kernel_size=5, depth=2, ffn_dim=12),
         ...,
         seed=seed + 1,
         ...,
     )
     tiny_joint, tiny_scales = _train_comparative_model(
         lambda: TinyJoint16QAM(width=16, taps=8),
         ...,
         seed=seed + 2,
         ...,
     )
     ```
     (Remember to add `Callable` to `from typing import ...` at `line 17`.)
     Optionally wrap `run_p4a_comparative_benchmarks` (or restore `torch.get_rng_state()` in a `try/finally`) if caller RNG isolation is desired, though calling `torch.manual_seed(seed)` before each model factory already guarantees `run_p4a_comparative_benchmarks(seed=2026)` produces bit-for-bit identical `train_final_scale_ratio` and `rows` regardless of prior global RNG state!
     Then regenerate `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv`, and add a fast RNG-perturbation determinism test in `p4/tests/test_p4_a_equivariance.py`.
   - **Finding 2 (`evaluate_noise_data_feedback` & `evaluate_perfect_pilot_demapping`)**:
     Add `torch.manual_seed(42)` at `p4/em_floatformer.py:733` (before `EMFloatFormer(depth=1)`) and `torch.manual_seed(99)` at `p4/em_floatformer.py:759` (before `EMFloatFormer(depth=1)`). Both were verified to pass their thresholds deterministically (`deg_db=0.001448`, `ber=0.000244`).
   - **Remaining 15 audit helpers**:
     Seeding `torch.manual_seed(...)` (e.g., `42` or matching their local `g` seed, optionally saving/restoring `torch.get_rng_state()`) inside the 15 helpers listed in Observation 1.2 eliminates all remaining ambient RNG dependence across `p4/em_floatformer.py`.

3. **Why `LocalEquivConv16QAM` (`0.0453`) beats `EMFloatFormer` (`0.0503`–`0.0508`) in `p4a_comparative_benchmark.csv`**:
   - **Not an LR / step-count / cosine-schedule issue**: Across `{3e-3, 5e-3, 8e-3}` and both `240` and `480` steps, `EMFloatFormer` stays at `BER=0.0501–0.0540` (with `8e-3` already optimal among the three LRs), while `LocalEquivConv16QAM` stays at `BER=0.0447–0.0456`.
   - **Real physical/architectural mechanism**:
     1. In Stage P4-A (`equal4`, $N_{\text{sc}}=32$, $f_d=0$, 100% dense pilots on all 32 subcarriers), the 4-tap contiguous delay spread induces smooth local frequency correlation across adjacent subcarriers. `LocalEquivConv16QAM`'s circular `ComplexConv1d(kernel_size=3, 5)` directly matches this shift-equivariant local smoothness, reducing strong-PA channel estimation `h_MSE` to `0.0254` vs `EMFloatFormer`'s `0.0401` (where `JointEmbedding` is point-wise and global softmax `GreenAttention` must learn local smoothing from scratch).
     2. Crucially, the distortion ablation (`task-49`) proves that **without `dist_est`**, `EMFloatFormer` actually outperforms `LocalEquivConv16QAM` (`ber_noDist = 0.0508` vs `0.0541`). With `dist_est` enabled, `LocalEquivConv16QAM` learns an active residual/distortion output (`dist_rms = 0.0848`) that cuts BER from `0.0541` to `0.0453` (`-16.3%` relative), whereas `EMFloatFormer`'s `head_dist` remains near zero (`dist_rms = 0.0168` at 240 steps, `0.0096` at 480 steps), contributing only `0.0508 -> 0.0503`.
     3. No hyperparameter or benchmark hack should be applied in Stage P4-A; this is an objective 1D dense-pilot inductive-bias finding, while `EMFloatFormer`'s global Hermitian propagator attention targets Stage P4-B's sparse DMRS pilots, long/sparse delay spreads, and 2D Doppler ICI.

---

## 3. Caveats

- No source files, test files, or `p4/results/*` artifacts were modified (`git status --short` confirms clean read-only state).
- Diagnostic measurements were executed on CPU with `torch.set_num_threads(4)` / `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`.
- Under the proposed factory fix, `LocalEquivConv16QAM` will use `seed + 1` (`2027`) and `TinyJoint16QAM` will use `seed + 2` (`2028`) instead of accidentally inheriting `2026` and `2027`, so their exact regenerated bit-error counts in `p4a_comparative_benchmark.{json,csv}` will reflect `seed+1` and `seed+2` (already measured at `strong 20 dB` for `LocalEquivConv16QAM(seed=2027)`: `BER=0.0453`, identical to 4 decimal places).

---

## 4. Conclusion

1. **Implement factory-callable seeding in `_train_comparative_model` (`p4/em_floatformer.py:1588-1603`) and `run_p4a_comparative_benchmarks` (`lines 1661-1692`)** so `EMFloatFormer`, `LocalEquivConv16QAM`, and `TinyJoint16QAM` are each instantiated *after* `torch.manual_seed(seed + idx)` (`idx = 0, 1, 2`).
2. **Add deterministic seeding in `evaluate_noise_data_feedback` (`line 733`, `seed=42`) and `evaluate_perfect_pilot_demapping` (`line 759`, `seed=99`)**, and optionally in the 15 other unseeded audit helpers (`lines 113, 135, 908, 955, 992, 1027, 1053, 1080, 1111, 1153, 1177, 1200, 1217, 1242, 1272`).
3. **Regenerate `p4/results/p4a_comparative_benchmark.json` and `.csv`** via `run_p4a_comparative_benchmarks()` and add an RNG-perturbation reproducibility unit test in `p4/tests/test_p4_a_equivariance.py`.
4. **Leave comparative benchmark hyperparameters untouched**: `LocalEquivConv16QAM` vs `EMFloatFormer` (`0.0453` vs `0.0503`) is a genuine empirical outcome of 1D circular Conv1d's dense-pilot `equal4` inductive bias + active `dist_est` (`0.0848` vs `0.0168`), not an LR/schedule defect.

---

## 5. Verification Method

After the Worker applies the fix and regenerates `p4/results/p4a_comparative_benchmark.{json,csv}`:
```powershell
Set-Location ""
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -c "
import json, torch
from pathlib import Path
from p4.em_floatformer import (
    run_p4a_comparative_benchmarks,
    evaluate_noise_data_feedback,
    evaluate_perfect_pilot_demapping,
)
torch.randn(17)
d1 = evaluate_noise_data_feedback()
p1 = evaluate_perfect_pilot_demapping()
torch.randn(53)
d2 = evaluate_noise_data_feedback()
p2 = evaluate_perfect_pilot_demapping()
assert d1 == d2 and p1 == p2, (d1, d2, p1, p2)

saved = json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8'))
torch.randn(19)
repro = run_p4a_comparative_benchmarks(output_dir=None)
assert saved['train_final_scale_ratio'] == repro['train_final_scale_ratio']
assert saved['rows'] == repro['rows']
print('ALL REPRODUCIBILITY CHECKS PASSED')
"
git diff --name-only
```
**Invalidation conditions**: Any mismatch between `saved` and `repro`, any non-determinism across ambient RNG perturbations in Finding 1 or Finding 2 functions, any failure in the three unittest suites, or any modification to Stage P4-0 or historical P1–P3 / `exploratory/*` / `MANIFEST_SHA256.json`.
