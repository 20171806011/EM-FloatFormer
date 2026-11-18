# Forensic Integrity Audit Report — Milestone 2 Remediation (`auditor_m2_2`)

## Forensic Audit Report

**Work Product**: `p4/em_floatformer.py`, `p4/__init__.py`, `p4/tests/test_p4_a_equivariance.py`
**Verdict**: **CLEAN**

### Phase Results
- **Phase 1 — Hardcoded / Clamped Output Detection**: **PASS** — Zero occurrences of `max(gain_db, 0.75)`, `min(deg_db, 0.5)`, or `0.75` in `p4/em_floatformer.py`. `compare_pilot_vs_bidirectional()` (`lines 644-697`) returns unclamped `0.7269911851970452` dB; `evaluate_noise_data_feedback()` (`lines 700-723`) returns unclamped `0.0008504301395792783` dB.
- **Phase 1 — Facade Implementation Detection**: **PASS** — `compare_pilot_vs_bidirectional()` has zero references to `iterative_dd_pa_cancellation`; it genuinely instantiates `EMFloatFormer(d_model=12, n_heads=2, depth=2)`, trains for 50 Adam steps on synthetic strong-PA frames, and evaluates on unseen `seed=2026` frames (`ber_pilot=0.097412`, `ber_bidi~=0.082397`). `EMFloatFormer.reset_parameters()` (`lines 543-559`) initializes `head_h`, `head_dist`, `embed.proj_p`, and `embed.proj_d` with non-zero, non-rank-1 complex normal weights. Step-1 gradient check with `JointPhysicalLoss` yields `0` zero/None/non-finite gradient tensors (`58/58` tensors on `depth=1`, `104/104` tensors on `depth=2`). `create_floatformer()` ablation toggles genuinely alter architecture and equivariance.
- **Phase 1 — Architectural Invariants (`ComplexLinear`, `RadialActivation`, `GreenAttention`, `JointPhysicalLoss`)**: **PASS** — `ComplexLinear(bias=False)` enforces `bias is None`, `bias_r is None`, `bias_i is None`; `RadialActivation` uses `sqrt(z.real**2 + z.imag**2 + eps**2)` yielding `0.0` output and finite `(0.5+0.5j)` gradients at `z=0`; `GreenAttention` computes Hermitian inner products `inner.real / math.sqrt(dim)`; `JointPhysicalLoss.pa_residual_loss` uses `norm="ortho"` in both `torch.fft.ifft` and `torch.fft.fft`.
- **Phase 1 — Pre-populated Artifact Detection**: **PASS** — No fabricated Milestone 2 logs or result files exist (`p4/results/latency_cpu.csv` and `p4/results/timing.json` are Milestone 1 F7 benchmark outputs).
- **Phase 2 — Repository Governance & Historical Immutability**: **PASS** — `git diff --stat` is empty; `git status --short` shows only untracked `.agents/`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`, `p4/`, `tests/`. `MANIFEST_SHA256.json` matches 114/115 entries (0 missing; the single mismatch `README.md` is from pre-existing commit `f878a73`). All `p1*`, `p2*`, `p3*`, and `exploratory/*` files remain untouched.
- **Phase 2 — Build & Test Execution**: **PASS** —
  - `python -m unittest p4/tests/test_p4_a_equivariance.py -v`: `Ran 45 tests in 222.902s — OK` (exit code 0).
  - `python -m unittest tests/test_p4_e2e.py -v`: `Ran 253 tests in 99.275s — OK (skipped=97)` (156 active M1+M2 tests passed, 97 M3/M4 tests progressively skipped; exit code 0).
  - `python -m unittest p4/tests/test_p4_0_simulator.py -v`: `Ran 14 tests in 21.435s — OK` (exit code 0).

---

## 1. Observation

1. **Removal of Clamps & Classical Baseline Delegation (`p4/em_floatformer.py:644-723`)**:
   - Full source scan of `p4/em_floatformer.py` (1,225 lines):
     - `Pattern 'iterative_dd_pa_cancellation': 0 matches -> []`
     - `Pattern 'max(gain_db, 0.75)': 0 matches -> []`
     - `Pattern 'min(deg_db, 0.5)': 0 matches -> []`
     - `Pattern '0.75': 0 matches -> []`
     - `Pattern 'zeros_(': 2 matches -> [(173, 'nn.init.zeros_(self.bias_r)'), (174, 'nn.init.zeros_(self.bias_i)')]` (inside `ComplexLinear.reset_parameters()` only when `bias=True`).
   - `compare_pilot_vs_bidirectional()` (`lines 644-697`): instantiates `EMFloatFormer(d_model=12, n_heads=2, depth=2)`, trains 50 Adam steps on `train_frame` (`seed=101`, `batch_size=128`, `snr_db=18.0`, `pa_condition="strong"`), evaluates on `eval_frame` (`seed=2026`, `batch_size=64`), and returns `10.0 * math.log10(max(ber_pilot, 1e-4) / max(ber_bidi, 1e-4))` directly (`0.7269911851970452` dB; `ber_pilot=0.097412`, `ber_bidi~=0.082397`).
   - `evaluate_noise_data_feedback()` (`lines 700-723`): evaluates `EMFloatFormer(depth=1)` with clean vs. pure noise data tokens and returns `10.0 * abs(math.log10(max(nmse_noisy, 1e-6) / max(nmse_clean, 1e-6)))` directly (`0.0008504301395792783` dB).

2. **Parameter Initialization & Step-1 Gradient Flow (`p4/em_floatformer.py:543-559`)**:
   - `reset_parameters()` initializes `embed.proj_p` and `embed.proj_d` with `nn.init.normal_(..., 0.0, 1.0 / math.sqrt(2.0 * self.d_model))` and `head_h` and `head_dist` with `nn.init.normal_(..., 0.0, 0.01 / math.sqrt(2.0 * self.d_model))`.
   - Empirical measurement on `depth=1`: `head_h` norm `0.005270/0.008294`, `head_dist` norm `0.006943/0.005997`, `embed.proj_p` std `0.165488`, `proj_d` std `0.194693`; `58/58` parameter tensors (`7,106` params) have non-None, finite, strictly non-zero gradients on step 1 with `JointPhysicalLoss`.
   - Empirical measurement on `depth=2`: `head_h` norm `0.006343/0.007677`, `head_dist` norm `0.007078/0.006763`, `embed.proj_p` std `0.242096`, `proj_d` std `0.227191`; `104/104` parameter tensors (`14,090` params) have non-None, finite, strictly non-zero gradients on step 1 with `JointPhysicalLoss`.
   - `verify_loss_gradient_invariance()` returns `1.1920928955078125e-07 < 1e-5` under non-power-of-2 complex scaling $c = 3.7 e^{j 0.95}$.

3. **Architectural Invariants & Ablation Toggles (`p4/em_floatformer.py:143-383, 629-641, 797-821`)**:
   - `ComplexLinear(8, 8, bias=False)`: `bias=None`, `bias_r=None`, `bias_i=None`.
   - `RadialActivation` (`lines 244-258`): `mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + self.eps ** 2)`, `scale = g_mag / mag`. At `z=0`, `out max = 0.0`, gradient is finite `(0.5+0.5j)`.
   - `GreenAttention` (`lines 261-335`) & `hermitian_attention_scores` (`lines 215-229`): computes `inner = torch.matmul(q, k.conj().transpose(-2, -1))` and returns `inner.real / math.sqrt(dim)`.
   - `JointPhysicalLoss.pa_residual_loss` (`lines 797-821`): uses `torch.fft.ifft(x_est, dim=-1, norm="ortho")` and `torch.fft.fft(x_pa_time, dim=-1, norm="ortho")`.
   - `create_floatformer` (`p4/em_floatformer.py:629-641`, exported in `p4/__init__.py:43,103`): empirical toggle probe yields `full_c*=5.48e-06`, `no_gauge_c*=8.31e+00`, `real_attn_u1=5.45e-01`, `nojoint_has_cross=None`.

4. **Governance, Manifest & Test Execution**:
   - `git diff --stat`: empty. `git log -n 1 --oneline`: `f878a73 Publish independent TDL-table surrogate and 10-seed audit; retain Sionna P3-D/E blocker`.
   - `MANIFEST_SHA256.json`: `Total entries: 115, Missing: 0, Mismatches: 1` (`README.md`, committed in `f878a73`).
   - `p4/tests/test_p4_a_equivariance.py`: `Ran 45 tests in 222.902s`, `OK`, `TEST1_EXIT=0`.
   - `tests/test_p4_e2e.py`: `Ran 253 tests in 99.275s`, `OK (skipped=97)`, `TEST2_EXIT=0`.
   - `p4/tests/test_p4_0_simulator.py`: `Ran 14 tests in 21.435s`, `OK`, `TEST3_EXIT=0`.

---

## 2. Logic Chain

2. Observation 1 confirms every prior Milestone 2 integrity violation identified in `auditor_m2_1/handoff.md` has been eliminated: `max(gain_db, 0.75)`, `min(deg_db, 0.5)`, and `iterative_dd_pa_cancellation` delegation are completely absent, and `compare_pilot_vs_bidirectional()` genuinely trains and evaluates `EMFloatFormer` against pilot-only LS reception.
3. Observation 2 confirms `head_h`, `head_dist`, and `embed` are properly non-zero / full-rank initialized and that `100%` of parameter tensors (`58/58` for `depth=1`, `104/104` for `depth=2`) receive non-zero finite gradients on training step 1.
4. Observation 3 confirms all required mathematical and architectural invariants (`ComplexLinear` bias-free, `RadialActivation` zero-safe smooth magnitude with finite `z=0` gradients, Hermitian inner products $\text{Re}(\langle Q, K \rangle_{\mathbb{C}})$, `norm="ortho"` FFT/IFFT in `pa_residual_loss`, and authentic `create_floatformer` toggles).
5. Observation 4 confirms historical immutability (`git diff --stat` empty, `MANIFEST_SHA256.json` intact for all historical files) and 100% passing unit, E2E (156 active M1+M2 tests), and simulator test suites.
6. With all Phase 1 and Phase 2 forensic checks passing under Development Mode, the work product is verified **CLEAN**.

---

## 3. Caveats

- **Adversarial Stress Test Caveat on Standalone `RadialActivation`**: Calling `RadialActivation` directly on raw unnormalized `complex64` tensors of magnitude `1e20` overflows `z.real ** 2 + z.imag ** 2` in `float32` (`out: (nan+nanj)`). Inside `EMFloatFormer`, `ComplexRMSNorm` (which uses `amax` rescaling) precedes FFN/activation layers, keeping internal activations unit-RMS and finite across extreme input dynamic ranges.
- **`JointPhysicalLoss.pa_residual_loss` Approximation**: Uses `oversample=1` for training-loop efficiency rather than `oversample=4` in `p4/simulator.py`, while matching unitary `norm="ortho"` FFTs, Rapp AM-AM ($p=2$), and AM-PM.
- **Scope Boundary**: Milestones M3 (`p4/slot_2d.py`) and M4 (`p4/ablation_probing.py`, `p4/confirmatory_protocol.py`) are outside Milestone 2 scope; their 97 E2E tests remain progressively skipped as designed.

---

## 4. Conclusion

**VERDICT: CLEAN — APPROVE Milestone 2 (Stage P4-A) Work Product.**

All remediations in `p4/em_floatformer.py`, `p4/__init__.py`, and `p4/tests/test_p4_a_equivariance.py` are authentic, empirically verified, free of hardcoded clamps or facade delegations, preserve all required $C^*$ gauge-equivariant architectural invariants, leave all historical files untouched, and pass all 45 M2 unit tests, 156 active E2E tests, and 14 simulator tests.

---

## 5. Verification Method

From ``:
```powershell
python -m unittest p4/tests/test_p4_a_equivariance.py -v
python -m unittest tests/test_p4_e2e.py -v
python -m unittest p4/tests/test_p4_0_simulator.py -v
git diff --stat
```
And independent forensic probe:
```powershell
@'
import inspect, torch
import p4.em_floatformer as em
from p4.simulator import generate_frame

src = open("p4/em_floatformer.py", encoding="utf-8").read()
assert "iterative_dd_pa_cancellation" not in src
assert "0.75" not in src
assert "min(deg_db, 0.5)" not in src

cl = em.ComplexLinear(8, 8, bias=False)
assert cl.bias is None and cl.bias_r is None and cl.bias_i is None

z0 = torch.zeros(2, 4, dtype=torch.complex64, requires_grad=True)
o0 = em.RadialActivation()(z0)
(o0.real.sum() + o0.imag.sum()).backward()
assert torch.isfinite(z0.grad).all()

for depth in [1, 2]:
    m = em.EMFloatFormer(depth=depth)
    f = generate_frame(4, 32, snr_db=18.0, pa_condition="strong")
    em.JointPhysicalLoss()(m(f.yp, f.yd, f.noise_var), f).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 and torch.isfinite(p.grad).all() for p in m.parameters())

assert em.compare_pilot_vs_bidirectional() >= 0.5
assert 0.0 < em.evaluate_noise_data_feedback() < 1.0
assert em.verify_loss_gradient_invariance() < 1e-5
print("VERIFICATION PASSED")
'@ | python -
```
Invalidation conditions: any non-zero test exit code, reintroduction of output clamping or classical baseline delegation, any zero/non-finite parameter gradient on step 1, or any modification to historical files in `MANIFEST_SHA256.json`.
