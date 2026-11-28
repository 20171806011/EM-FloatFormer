import Mathlib.Basic.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.GCongr

/-!
# Polylogarithmic Evanescent-Mode Scaling & Reverse Water-Filling (`WavePolylog.lean`)

Formalizes the load-bearing mathematical core of Conjecture 1 (Physical Wave/EM Field
Evanescent-Mode Late-Query Cache):

1. **Multipole Bit-Allocation Sums (Section 1)**:
   - Exact 3D spherical harmonic sums:
     `∑ l ∈ Finset.range (L + 1), (2 * l + 1) = (L + 1) ^ 2`
     `6 * (∑ l ∈ Finset.range (L + 1), (2 * l + 1) * (L - l)) = L * (L + 1) * (2 * L + 1)`
     with Real form `= L * (L + 1) * (2 * L + 1) / 6` and sharp bounds
     `L ^ 3 / 3 ≤ ∑ ... ≤ (L + 1) ^ 3 / 3`.
   - Exact 2D multipole sums (1-mode over `0..L`, 2-mode over `1..L`, full circular harmonics
     with monopole `l = 0`, and general `m`-mode over `ℝ`) with sharp quadratic bounds.
2. **Reverse Water-Filling Lagrangian Optimality (Section 2)**:
   - Pointwise Lagrangian inequality via `Real.log_le_sub_one_of_pos`.
   - Constructive feasibility witness `D* i = min (lam i) θ` (proving non-vacuity).
   - Finite-index and weighted multipole reverse water-filling lower bounds.
3. **Geometric-Decay Ellipsoid Polylog Bounds & Achievability (Section 3)**:
   - Constructive cutoff existence `exists_evanescent_cutoff` (non-vacuity of cutoff hypotheses).
   - Non-asymptotic and `(1 - δ)`-asymptotic polylogarithmic lower bounds in 3D and 2D.
   - Truncated water-filling achievability (upper bound) with exact distortion and rate.
4. **Strict Geometric Denominator Monotonicity for `R₂ < R₁` (Section 4)**:
   - Strict monotonicity of `Real.logb 2 (R / a)`, strict upward jump of 2D and 3D leading
     coefficients, exact jump ratios `> 1`, and exact additive gap identities.
-/

namespace MathLab.LateQueryFrontier.WaveField

/-! ## Section 1: Multipole Bit-Allocation Sums (Nat & Real + Sharp Bounds) -/

/-- Sum of 3D spherical harmonic mode counts `2 * l + 1` for `l = 0, ..., L` equals `(L + 1) ^ 2`. -/
theorem sum_odd_eq_sq (L : ℕ) :
    ∑ l ∈ Finset.range (L + 1), (2 * l + 1) = (L + 1) ^ 2 := by
  induction L with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    ring

/-- Exact 3D multipole bit-allocation sum over `ℕ`:
`6 * ∑_{l=0}^L (2l + 1)(L - l) = L(L + 1)(2L + 1)`. -/
theorem sum_3d_bit_nat (L : ℕ) :
    6 * (∑ l ∈ Finset.range (L + 1), (2 * l + 1) * (L - l)) = L * (L + 1) * (2 * L + 1) := by
  induction L with
  | zero => simp
  | succ n ih =>
    have htop :
        ∑ l ∈ Finset.range (n + 1 + 1), (2 * l + 1) * (n + 1 - l) =
          ∑ l ∈ Finset.range (n + 1), (2 * l + 1) * (n + 1 - l) := by
      rw [Finset.sum_range_succ]
      simp
    have hsplit :
        ∑ l ∈ Finset.range (n + 1), (2 * l + 1) * (n + 1 - l) =
          ∑ l ∈ Finset.range (n + 1), ((2 * l + 1) * (n - l) + (2 * l + 1)) := by
      refine Finset.sum_congr rfl ?_
      intro l hl
      have hle : l ≤ n := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
      have hsub : n + 1 - l = (n - l) + 1 := by omega
      rw [hsub]
      ring
    rw [htop, hsplit, Finset.sum_add_distrib, sum_odd_eq_sq]
    have hmul :
        6 * (∑ l ∈ Finset.range (n + 1), (2 * l + 1) * (n - l) + (n + 1) ^ 2) =
          6 * (∑ l ∈ Finset.range (n + 1), (2 * l + 1) * (n - l)) + 6 * (n + 1) ^ 2 := by
      ring
    rw [hmul, ih]
    ring

private lemma nat_sub_cast_eq {L l : ℕ} (hl : l ≤ L) :
    ((L - l : ℕ) : ℝ) = (L : ℝ) - (l : ℝ) := by
  have h : (L - l : ℕ) + l = L := Nat.sub_add_cancel hl
  have hr : ((L - l : ℕ) : ℝ) + (l : ℝ) = (L : ℝ) := by exact_mod_cast h
  linarith

/-- Exact 3D multipole bit-allocation sum over `ℝ`. -/
theorem sum_3d_bit_real (L : ℕ) :
    ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) =
      (L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6 := by
  have hcast :
      ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) =
        ((∑ l ∈ Finset.range (L + 1), (2 * l + 1) * (L - l) : ℕ) : ℝ) := by
    rw [Nat.cast_sum]
    refine Finset.sum_congr rfl ?_
    intro l hl
    have hle : l ≤ L := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
    rw [Nat.cast_mul, nat_sub_cast_eq hle]
  rw [hcast]
  have hnat : ((6 * (∑ l ∈ Finset.range (L + 1), (2 * l + 1) * (L - l)) : ℕ) : ℝ) =
      ((L * (L + 1) * (2 * L + 1) : ℕ) : ℝ) := by
    exact_mod_cast sum_3d_bit_nat L
  rw [Nat.cast_mul 6, Nat.cast_ofNat] at hnat
  have hnatR : ((L * (L + 1) * (2 * L + 1) : ℕ) : ℝ) =
      (L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) := by
    push_cast
    ring
  rw [hnatR] at hnat
  linarith

/-- Sharp cubic lower and upper bounds on the 3D multipole bit-allocation sum. -/
theorem sum_3d_bit_bounds (L : ℕ) :
    (L : ℝ) ^ 3 / 3 ≤ ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) ∧
      ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) ≤
        ((L : ℝ) + 1) ^ 3 / 3 := by
  rw [sum_3d_bit_real]
  have hL : (0 : ℝ) ≤ (L : ℝ) := Nat.cast_nonneg L
  constructor <;> nlinarith

/-- Exact 2D 1-mode-per-order sum over `ℕ`: `2 * ∑_{l=0}^L (L - l) = L(L + 1)`. -/
theorem sum_2d_one_mode_nat (L : ℕ) :
    2 * (∑ l ∈ Finset.range (L + 1), (L - l)) = L * (L + 1) := by
  induction L with
  | zero => simp
  | succ n ih =>
    have htop :
        ∑ l ∈ Finset.range (n + 1 + 1), (n + 1 - l) =
          ∑ l ∈ Finset.range (n + 1), (n + 1 - l) := by
      rw [Finset.sum_range_succ]
      simp
    have hsplit :
        ∑ l ∈ Finset.range (n + 1), (n + 1 - l) =
          ∑ l ∈ Finset.range (n + 1), ((n - l) + 1) := by
      refine Finset.sum_congr rfl ?_
      intro l hl
      have hle : l ≤ n := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
      omega
    rw [htop, hsplit, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, smul_eq_mul,
      mul_one]
    have hmul :
        2 * (∑ l ∈ Finset.range (n + 1), (n - l) + (n + 1)) =
          2 * (∑ l ∈ Finset.range (n + 1), (n - l)) + 2 * (n + 1) := by
      ring
    rw [hmul, ih]
    ring

/-- Exact 2D 1-mode-per-order sum over `ℝ`. -/
theorem sum_2d_one_mode_real (L : ℕ) :
    ∑ l ∈ Finset.range (L + 1), ((L : ℝ) - (l : ℝ)) = (L : ℝ) * ((L : ℝ) + 1) / 2 := by
  have hcast :
      ∑ l ∈ Finset.range (L + 1), ((L : ℝ) - (l : ℝ)) =
        ((∑ l ∈ Finset.range (L + 1), (L - l) : ℕ) : ℝ) := by
    rw [Nat.cast_sum]
    refine Finset.sum_congr rfl ?_
    intro l hl
    exact (nat_sub_cast_eq (Nat.lt_succ_iff.mp (Finset.mem_range.mp hl))).symm
  rw [hcast]
  have hnat : ((2 * (∑ l ∈ Finset.range (L + 1), (L - l)) : ℕ) : ℝ) =
      ((L * (L + 1) : ℕ) : ℝ) := by
    exact_mod_cast sum_2d_one_mode_nat L
  rw [Nat.cast_mul 2, Nat.cast_ofNat] at hnat
  have hnatR : ((L * (L + 1) : ℕ) : ℝ) = (L : ℝ) * ((L : ℝ) + 1) := by
    push_cast
    ring
  rw [hnatR] at hnat
  linarith

/-- Sharp quadratic bounds on the 2D 1-mode-per-order sum. -/
theorem sum_2d_one_mode_bounds (L : ℕ) :
    (L : ℝ) ^ 2 / 2 ≤ ∑ l ∈ Finset.range (L + 1), ((L : ℝ) - (l : ℝ)) ∧
      ∑ l ∈ Finset.range (L + 1), ((L : ℝ) - (l : ℝ)) ≤ ((L : ℝ) + 1) ^ 2 / 2 := by
  rw [sum_2d_one_mode_real]
  have hL : (0 : ℝ) ≤ (L : ℝ) := Nat.cast_nonneg L
  constructor <;> nlinarith

/-- Exact 2D 2-mode-per-order sum over `l = 1, ..., L`: `∑_{l=1}^L 2(L - l) = L(L - 1)`. -/
theorem sum_2d_two_mode_nat (L : ℕ) :
    ∑ l ∈ Finset.range L, 2 * (L - (l + 1)) = L * (L - 1) := by
  cases L with
  | zero => simp
  | succ n =>
    have hsimp :
        ∑ l ∈ Finset.range (n + 1), 2 * (n + 1 - (l + 1)) =
          2 * (∑ l ∈ Finset.range (n + 1), (n - l)) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro l _
      have hsub : n + 1 - (l + 1) = n - l := by omega
      rw [hsub]
    rw [hsimp, sum_2d_one_mode_nat]
    simp [Nat.mul_comm]

/-- Exact 2D 2-mode-per-order sum over `ℝ`. -/
theorem sum_2d_two_mode_real (L : ℕ) :
    ∑ l ∈ Finset.range L, (2 : ℝ) * ((L : ℝ) - ((l + 1 : ℕ) : ℝ)) = (L : ℝ) * ((L : ℝ) - 1) := by
  cases L with
  | zero => simp
  | succ n =>
    have hcast :
        ∑ l ∈ Finset.range (n + 1), (2 : ℝ) * (((n + 1 : ℕ) : ℝ) - ((l + 1 : ℕ) : ℝ)) =
          ((∑ l ∈ Finset.range (n + 1), 2 * (n + 1 - (l + 1)) : ℕ) : ℝ) := by
      rw [Nat.cast_sum]
      refine Finset.sum_congr rfl ?_
      intro l hl
      have hle : l + 1 ≤ n + 1 := Nat.succ_le_of_lt (Finset.mem_range.mp hl)
      rw [Nat.cast_mul, nat_sub_cast_eq hle]
      norm_num
    rw [hcast, sum_2d_two_mode_nat]
    simp

/-- Full 2D circular harmonics sum (1 mode at `l = 0`, 2 modes for `1 ≤ l ≤ L`) over `ℕ`. -/
theorem sum_2d_full_nat (L : ℕ) :
    L + ∑ l ∈ Finset.range L, 2 * (L - (l + 1)) = L ^ 2 := by
  rw [sum_2d_two_mode_nat]
  cases L with
  | zero => simp
  | succ n =>
    simp only [Nat.add_one_sub_one]
    ring

/-- Full 2D circular harmonics sum over `ℝ`. -/
theorem sum_2d_full_real (L : ℕ) :
    (L : ℝ) + ∑ l ∈ Finset.range L, (2 : ℝ) * ((L : ℝ) - ((l + 1 : ℕ) : ℝ)) = (L : ℝ) ^ 2 := by
  rw [sum_2d_two_mode_real]
  ring

/-- Multiplicity function for full 2D circular harmonics (`1` at `l = 0`, `2` at `l ≥ 1`). -/
def w2dFull (l : ℕ) : ℕ :=
  if l = 0 then 1 else 2

/-- Weighted form of the full 2D circular harmonics mode count `∑_{l=0}^L w2dFull l = 2L + 1`. -/
theorem sum_w2dFull_eq (L : ℕ) :
    ∑ l ∈ Finset.range (L + 1), w2dFull l = 2 * L + 1 := by
  induction L with
  | zero => simp [w2dFull]
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    simp [w2dFull]
    ring

/-- Weighted form of the full 2D circular harmonics bit-allocation sum `= L ^ 2`. -/
theorem sum_2d_full_weighted_nat (L : ℕ) :
    ∑ l ∈ Finset.range (L + 1), w2dFull l * (L - l) = L ^ 2 := by
  cases L with
  | zero => simp [w2dFull]
  | succ n =>
    rw [Finset.sum_range_succ']
    simp only [w2dFull, ↓reduceIte, Nat.sub_zero, one_mul]
    have htail :
        ∑ k ∈ Finset.range (n + 1), (if k + 1 = 0 then 1 else 2) * (n + 1 - (k + 1)) =
          ∑ k ∈ Finset.range (n + 1), 2 * (n + 1 - (k + 1)) := by
      refine Finset.sum_congr rfl ?_
      intro k _
      simp
    rw [htail, Nat.add_comm, sum_2d_full_nat]

/-- General `m`-mode-per-order 2D sum over `l = 1, ..., L` in `ℝ`. -/
theorem sum_2d_gen_mode_real (m : ℝ) (L : ℕ) :
    ∑ l ∈ Finset.range L, m * ((L : ℝ) - ((l + 1 : ℕ) : ℝ)) = m * (L : ℝ) * ((L : ℝ) - 1) / 2 := by
  have hfactor :
      ∑ l ∈ Finset.range L, m * ((L : ℝ) - ((l + 1 : ℕ) : ℝ)) =
        (m / 2) * ∑ l ∈ Finset.range L, (2 : ℝ) * ((L : ℝ) - ((l + 1 : ℕ) : ℝ)) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro l _
    ring
  rw [hfactor, sum_2d_two_mode_real]
  ring

/-- General `m`-mode-per-order 2D sum over `l = 0, ..., L` in `ℝ` and sharp quadratic bounds. -/
theorem sum_2d_gen_mode_zero_to_L_bounds {m : ℝ} (hm : 0 ≤ m) (L : ℕ) :
    m * (L : ℝ) ^ 2 / 2 ≤ ∑ l ∈ Finset.range (L + 1), m * ((L : ℝ) - (l : ℝ)) ∧
      ∑ l ∈ Finset.range (L + 1), m * ((L : ℝ) - (l : ℝ)) ≤ m * ((L : ℝ) + 1) ^ 2 / 2 := by
  have heq :
      ∑ l ∈ Finset.range (L + 1), m * ((L : ℝ) - (l : ℝ)) =
        m * ∑ l ∈ Finset.range (L + 1), ((L : ℝ) - (l : ℝ)) := by
    rw [Finset.mul_sum]
  rw [heq, sum_2d_one_mode_real]
  have hL : (0 : ℝ) ≤ (L : ℝ) := Nat.cast_nonneg L
  constructor <;> nlinarith

/-! ## Section 2: Finite Reverse Water-Filling Lagrangian Optimality -/

/-- Pointwise Lagrangian inequality for reverse water-filling via `Real.log_le_sub_one_of_pos`. -/
theorem waterfilling_pointwise_lb {lam θ D : ℝ}
    (hlam : 0 < lam) (hθ : 0 < θ) (hD : 0 < D) (hDle : D ≤ lam) :
    Real.log (lam / min lam θ) + (min lam θ - D) / θ ≤ Real.log (lam / D) := by
  by_cases hle : lam ≤ θ
  · have hmin : min lam θ = lam := min_eq_left hle
    rw [hmin, div_self hlam.ne', Real.log_one, zero_add]
    have hpos : 0 < D / lam := div_pos hD hlam
    have hlog := Real.log_le_sub_one_of_pos hpos
    have hlog_eq : Real.log (lam / D) = -Real.log (D / lam) := by
      rw [Real.log_div hlam.ne' hD.ne', Real.log_div hD.ne' hlam.ne']
      ring
    have h1 : (lam - D) / lam ≤ Real.log (lam / D) := by
      rw [hlog_eq]
      have h2 : (lam - D) / lam = 1 - D / lam := by
        field_simp [hlam.ne']
      linarith
    have h3 : (lam - D) / θ ≤ (lam - D) / lam := by
      have hnum : 0 ≤ lam - D := sub_nonneg.mpr hDle
      exact div_le_div_of_nonneg_left hnum hlam hle
    linarith
  · have hlt : θ < lam := lt_of_not_ge hle
    have hmin : min lam θ = θ := min_eq_right hlt.le
    rw [hmin]
    have hpos : 0 < D / θ := div_pos hD hθ
    have hlog := Real.log_le_sub_one_of_pos hpos
    have hlog_eq : Real.log (lam / D) - Real.log (lam / θ) = -Real.log (D / θ) := by
      rw [Real.log_div hlam.ne' hD.ne', Real.log_div hlam.ne' hθ.ne',
        Real.log_div hD.ne' hθ.ne']
      ring
    have hdiff : (θ - D) / θ = 1 - D / θ := by
      field_simp [hθ.ne']
    linarith

/-- Base-2 logarithmic pointwise Lagrangian inequality (scaled by `1 / 2` for real modes). -/
theorem waterfilling_pointwise_logb2_lb {lam θ D : ℝ}
    (hlam : 0 < lam) (hθ : 0 < θ) (hD : 0 < D) (hDle : D ≤ lam) :
    (1 / 2 : ℝ) * Real.logb 2 (lam / min lam θ) + (min lam θ - D) / (2 * θ * Real.log 2) ≤
      (1 / 2 : ℝ) * Real.logb 2 (lam / D) := by
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hpt := waterfilling_pointwise_lb hlam hθ hD hDle
  have hdiv :
      (Real.log (lam / min lam θ) + (min lam θ - D) / θ) / (2 * Real.log 2) ≤
        Real.log (lam / D) / (2 * Real.log 2) :=
    div_le_div_of_nonneg_right hpt (by positivity)
  have hLHS :
      (Real.log (lam / min lam θ) + (min lam θ - D) / θ) / (2 * Real.log 2) =
        (1 / 2 : ℝ) * Real.logb 2 (lam / min lam θ) + (min lam θ - D) / (2 * θ * Real.log 2) := by
    rw [Real.logb]
    field_simp [hθ.ne', hlog2.ne']
  have hRHS : Real.log (lam / D) / (2 * Real.log 2) = (1 / 2 : ℝ) * Real.logb 2 (lam / D) := by
    rw [Real.logb]
    ring
  linarith

/-- Constructive feasibility witness `D* i = min (lam i) θ`, proving the reverse water-filling
hypothesis set is non-vacuous. -/
theorem waterfilling_feasible {ι : Type*} (U : Finset ι) (lam : ι → ℝ) (θ : ℝ)
    (hlam : ∀ i ∈ U, 0 < lam i) (hθ : 0 < θ) :
    let Dstar : ι → ℝ := fun i => min (lam i) θ
    (∀ i ∈ U, 0 < Dstar i ∧ Dstar i ≤ lam i) ∧
      ∑ i ∈ U, Dstar i ≤ ∑ i ∈ U, min (lam i) θ := by
  intro Dstar
  refine ⟨?_, le_rfl⟩
  intro i hi
  exact ⟨lt_min (hlam i hi) hθ, min_le_left _ _⟩

/-- Exact finite-index Lagrangian inequality for reverse water-filling in base-2 bits. -/
theorem reverse_waterfilling_logb2_lagrangian {ι : Type*} (U : Finset ι) (lam D : ι → ℝ) (θ : ℝ)
    (hθ : 0 < θ) (hlam : ∀ i ∈ U, 0 < lam i)
    (hD_pos : ∀ i ∈ U, 0 < D i) (hD_le : ∀ i ∈ U, D i ≤ lam i) :
    (1 / 2 : ℝ) * ∑ i ∈ U, Real.logb 2 (lam i / min (lam i) θ) +
        (∑ i ∈ U, min (lam i) θ - ∑ i ∈ U, D i) / (2 * θ * Real.log 2) ≤
      (1 / 2 : ℝ) * ∑ i ∈ U, Real.logb 2 (lam i / D i) := by
  have hsum :
      ∑ i ∈ U, ((1 / 2 : ℝ) * Real.logb 2 (lam i / min (lam i) θ) +
          (min (lam i) θ - D i) / (2 * θ * Real.log 2)) ≤
        ∑ i ∈ U, (1 / 2 : ℝ) * Real.logb 2 (lam i / D i) := by
    refine Finset.sum_le_sum ?_
    intro i hi
    exact waterfilling_pointwise_logb2_lb (hlam i hi) hθ (hD_pos i hi) (hD_le i hi)
  have hLHS :
      ∑ i ∈ U, ((1 / 2 : ℝ) * Real.logb 2 (lam i / min (lam i) θ) +
          (min (lam i) θ - D i) / (2 * θ * Real.log 2)) =
        (1 / 2 : ℝ) * ∑ i ∈ U, Real.logb 2 (lam i / min (lam i) θ) +
          (∑ i ∈ U, min (lam i) θ - ∑ i ∈ U, D i) / (2 * θ * Real.log 2) := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_div, Finset.sum_sub_distrib]
  rw [hLHS, ← Finset.mul_sum] at hsum
  exact hsum

/-- Finite-index reverse water-filling optimality lower bound: any feasible distortion allocation `D`
with total distortion `≤ ∑_{i ∈ U} min (lam i) θ` has rate at least the water-filling rate. -/
theorem reverse_waterfilling_logb2_lower_bound {ι : Type*} (U : Finset ι) (lam D : ι → ℝ) (θ : ℝ)
    (hθ : 0 < θ) (hlam : ∀ i ∈ U, 0 < lam i)
    (hD_pos : ∀ i ∈ U, 0 < D i) (hD_le : ∀ i ∈ U, D i ≤ lam i)
    (hsum : ∑ i ∈ U, D i ≤ ∑ i ∈ U, min (lam i) θ) :
    (1 / 2 : ℝ) * ∑ i ∈ U, Real.logb 2 (lam i / min (lam i) θ) ≤
      (1 / 2 : ℝ) * ∑ i ∈ U, Real.logb 2 (lam i / D i) := by
  have hlag := reverse_waterfilling_logb2_lagrangian U lam D θ hθ hlam hD_pos hD_le
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hslack : 0 ≤ (∑ i ∈ U, min (lam i) θ - ∑ i ∈ U, D i) / (2 * θ * Real.log 2) := by
    exact div_nonneg (sub_nonneg.mpr hsum) (by positivity)
  linarith

/-- Weighted multipole reverse water-filling optimality over orders `l ∈ 0..L` with multiplicity
`w : ℕ → ℕ`. -/
theorem weighted_reverse_waterfilling_logb2_lower_bound
    (w : ℕ → ℕ) (L : ℕ) (lam D : ℕ → ℝ) (θ : ℝ)
    (hθ : 0 < θ) (hlam : ∀ l ∈ Finset.range (L + 1), 0 < lam l)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ lam l)
    (hsum :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l ≤
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * min (lam l) θ) :
    (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * Real.logb 2 (lam l / min (lam l) θ) ≤
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * Real.logb 2 (lam l / D l) := by
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hpt :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) *
          ((1 / 2 : ℝ) * Real.logb 2 (lam l / min (lam l) θ) +
            (min (lam l) θ - D l) / (2 * θ * Real.log 2)) ≤
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * ((1 / 2 : ℝ) * Real.logb 2 (lam l / D l)) := by
    refine Finset.sum_le_sum ?_
    intro l hl
    exact mul_le_mul_of_nonneg_left
      (waterfilling_pointwise_logb2_lb (hlam l hl) hθ (hD_pos l hl) (hD_le l hl))
      (Nat.cast_nonneg (w l))
  have hLHS :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) *
          ((1 / 2 : ℝ) * Real.logb 2 (lam l / min (lam l) θ) +
            (min (lam l) θ - D l) / (2 * θ * Real.log 2)) =
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * Real.logb 2 (lam l / min (lam l) θ) +
          (∑ l ∈ Finset.range (L + 1), (w l : ℝ) * min (lam l) θ -
            ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l) / (2 * θ * Real.log 2) := by
    have hterm :
        ∀ l : ℕ, (w l : ℝ) *
            ((1 / 2 : ℝ) * Real.logb 2 (lam l / min (lam l) θ) +
              (min (lam l) θ - D l) / (2 * θ * Real.log 2)) =
          (1 / 2 : ℝ) * ((w l : ℝ) * Real.logb 2 (lam l / min (lam l) θ)) +
            ((w l : ℝ) * min (lam l) θ - (w l : ℝ) * D l) / (2 * θ * Real.log 2) := by
      intro l
      ring
    simp_rw [hterm]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_div, Finset.sum_sub_distrib]
  have hRHS :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * ((1 / 2 : ℝ) * Real.logb 2 (lam l / D l)) =
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * Real.logb 2 (lam l / D l) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro l _
    ring
  rw [hLHS, hRHS] at hpt
  have hslack :
      0 ≤ (∑ l ∈ Finset.range (L + 1), (w l : ℝ) * min (lam l) θ -
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l) / (2 * θ * Real.log 2) :=
    div_nonneg (sub_nonneg.mpr hsum) (by positivity)
  linarith

/-! ## Section 3: Geometric-Decay Ellipsoid Polylog Bounds & Achievability -/

/-- Geometric-decay modal variance `c * (a / R) ^ (2 * l)` at multipole order `l`. -/
noncomputable def geomVar (c a R : ℝ) (l : ℕ) : ℝ :=
  c * (a / R) ^ (2 * l)

theorem geomVar_pos {c a R : ℝ} (hc : 0 < c) (ha : 0 < a) (hR : a < R) (l : ℕ) :
    0 < geomVar c a R l := by
  have hRa : 0 < a / R := div_pos ha (ha.trans hR)
  unfold geomVar
  positivity

theorem geomVar_antitone {c a R : ℝ} (hc : 0 < c) (ha : 0 < a) (hR : a < R)
    {l L : ℕ} (hl : l ≤ L) :
    geomVar c a R L ≤ geomVar c a R l := by
  unfold geomVar
  have hRpos : 0 < R := ha.trans hR
  have hratio_nonneg : 0 ≤ a / R := (div_pos ha hRpos).le
  have hratio_le : a / R ≤ 1 := (div_le_one hRpos).mpr hR.le
  have hpow : (a / R) ^ (2 * L) ≤ (a / R) ^ (2 * l) :=
    pow_le_pow_of_le_one hratio_nonneg hratio_le (Nat.mul_le_mul_left 2 hl)
  exact mul_le_mul_of_nonneg_left hpow hc.le

theorem geomVar_ratio_logb2 {c a R : ℝ} (hc : 0 < c) (ha : 0 < a) (hR : a < R)
    {l L : ℕ} (hl : l ≤ L) :
    (1 / 2 : ℝ) * Real.logb 2 (geomVar c a R l / geomVar c a R L) =
      ((L : ℝ) - (l : ℝ)) * Real.logb 2 (R / a) := by
  have hRpos : 0 < R := ha.trans hR
  have haR_pos : 0 < a / R := div_pos ha hRpos
  have hRa_pos : 0 < R / a := div_pos hRpos ha
  have hsub : 2 * L = 2 * l + 2 * (L - l) := by omega
  have hdiv_eq : geomVar c a R l / geomVar c a R L = (R / a) ^ (2 * (L - l)) := by
    unfold geomVar
    rw [hsub, pow_add]
    have hpow_ne : (a / R) ^ (2 * l) ≠ 0 := (pow_pos haR_pos _).ne'
    have hmul_inv : (a / R) * (R / a) = 1 := by field_simp [ha.ne', hRpos.ne']
    have hpow_inv : (a / R) ^ (2 * (L - l)) * (R / a) ^ (2 * (L - l)) = 1 := by
      rw [← mul_pow, hmul_inv, one_pow]
    field_simp [hc.ne', hpow_ne]
    exact hpow_inv.symm
  rw [hdiv_eq, Real.logb_pow]
  push_cast
  rw [nat_sub_cast_eq hl]
  ring

/-- Exact closed form for the weighted reverse water-filling rate on a geometric-decay ellipsoid
at threshold `θ = geomVar c a R L`. -/
theorem geom_weighted_waterfilling_rate_eq (w : ℕ → ℕ) (L : ℕ) {c a R : ℝ}
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) :
    (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) *
        Real.logb 2 (geomVar c a R l / min (geomVar c a R l) (geomVar c a R L)) =
      (∑ l ∈ Finset.range (L + 1), (w l : ℝ) * ((L : ℝ) - (l : ℝ))) * Real.logb 2 (R / a) := by
  have hcongr :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) *
          Real.logb 2 (geomVar c a R l / min (geomVar c a R l) (geomVar c a R L)) =
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * (2 * ((L : ℝ) - (l : ℝ)) * Real.logb 2 (R / a)) := by
    refine Finset.sum_congr rfl ?_
    intro l hl
    have hle : l ≤ L := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
    have hmin : min (geomVar c a R l) (geomVar c a R L) = geomVar c a R L :=
      min_eq_right (geomVar_antitone hc ha hR hle)
    rw [hmin]
    have hlog := geomVar_ratio_logb2 hc ha hR hle
    have hlog2 : Real.logb 2 (geomVar c a R l / geomVar c a R L) =
        2 * ((L : ℝ) - (l : ℝ)) * Real.logb 2 (R / a) := by
      linarith
    rw [hlog2]
  rw [hcongr]
  have hfactor :
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * (2 * ((L : ℝ) - (l : ℝ)) * Real.logb 2 (R / a)) =
        (∑ l ∈ Finset.range (L + 1), (w l : ℝ) * ((L : ℝ) - (l : ℝ))) * Real.logb 2 (R / a) := by
    rw [Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl ?_
    intro l _
    ring
  exact hfactor

/-- Constructive existence of a non-vacuous evanescent cutoff order `L ≥ 1` satisfying both
`ε ≤ geomVar c a R L` and `Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1`,
whenever `ε ≤ c * (a / R) ^ 2`. -/
theorem exists_evanescent_cutoff {c a R ε : ℝ}
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) (hε : 0 < ε) (hregime : ε ≤ c * (a / R) ^ 2) :
    ∃ L : ℕ, 1 ≤ L ∧ ε ≤ geomVar c a R L ∧
      Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1 := by
  classical
  have hRpos : 0 < R := ha.trans hR
  have hRa_gt_one : 1 < R / a := (one_lt_div ha).mpr hR
  have hlogRa_pos : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) hRa_gt_one
  set X : ℝ := Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))
  have hX_ge_one : 1 ≤ X := by
    have hc_eps : (R / a) ^ 2 ≤ c / ε := by
      have haR_pos : 0 < a / R := div_pos ha hRpos
      rw [le_div_iff₀ hε]
      calc
        (R / a) ^ 2 * ε ≤ (R / a) ^ 2 * (c * (a / R) ^ 2) :=
          mul_le_mul_of_nonneg_left hregime (by positivity)
        _ = c := by field_simp [ha.ne', hRpos.ne']
    have hlog_le : Real.logb 2 ((R / a) ^ 2) ≤ Real.logb 2 (c / ε) :=
      Real.logb_le_logb_of_le (by norm_num) (by positivity) hc_eps
    rw [Real.logb_pow] at hlog_le
    push_cast at hlog_le
    change 1 ≤ Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))
    rw [le_div_iff₀ (by positivity)]
    linarith
  have hex : ∃ n : ℕ, X < (n : ℝ) := exists_nat_gt X
  let m : ℕ := Nat.find hex
  have hm_gt : X < (m : ℝ) := Nat.find_spec hex
  have hm_ge_two : 2 ≤ m := by
    by_contra hlt
    have hm_le_one : m ≤ 1 := Nat.lt_succ_iff.mp (lt_of_not_ge hlt)
    have hm_real : (m : ℝ) ≤ 1 := by exact_mod_cast hm_le_one
    linarith
  let L : ℕ := m - 1
  have hL_ge_one : 1 ≤ L := by omega
  have hLm : L + 1 = m := by omega
  have hL_lt_m : L < m := by omega
  have hL_le_X : (L : ℝ) ≤ X := by
    have hnot : ¬ (X < (L : ℝ)) := Nat.find_min hex hL_lt_m
    exact not_lt.mp hnot
  have hX_lt_L1 : X ≤ (L : ℝ) + 1 := by
    have hm_cast : (m : ℝ) = (L : ℝ) + 1 := by exact_mod_cast hLm.symm
    linarith
  have hε_le : ε ≤ geomVar c a R L := by
    have h2L : (2 * (L : ℝ)) * Real.logb 2 (R / a) ≤ Real.logb 2 (c / ε) := by
      have h1 : (L : ℝ) * (2 * Real.logb 2 (R / a)) ≤ Real.logb 2 (c / ε) :=
        (le_div_iff₀ (by positivity)).mp hL_le_X
      linarith
    have hlog_pow : Real.logb 2 ((R / a) ^ (2 * L)) ≤ Real.logb 2 (c / ε) := by
      rw [Real.logb_pow]
      push_cast
      exact h2L
    have hpow_le : (R / a) ^ (2 * L) ≤ c / ε := by
      by_contra hlt
      have hlt' : c / ε < (R / a) ^ (2 * L) := lt_of_not_ge hlt
      have hlog_lt : Real.logb 2 (c / ε) < Real.logb 2 ((R / a) ^ (2 * L)) :=
        Real.logb_lt_logb (by norm_num) (div_pos hc hε) hlt'
      linarith
    unfold geomVar
    have hmul : (R / a) ^ (2 * L) * ε ≤ c := (le_div_iff₀ hε).mp hpow_le
    calc
      ε = ε * ((R / a) ^ (2 * L) * (a / R) ^ (2 * L)) := by
        have hinv : (R / a) ^ (2 * L) * (a / R) ^ (2 * L) = 1 := by
          rw [← mul_pow]
          have h1 : (R / a) * (a / R) = 1 := by field_simp [ha.ne', hRpos.ne']
          rw [h1, one_pow]
        rw [hinv, mul_one]
      _ = ((R / a) ^ (2 * L) * ε) * (a / R) ^ (2 * L) := by ring
      _ ≤ c * (a / R) ^ (2 * L) := mul_le_mul_of_nonneg_right hmul (by positivity)
  exact ⟨L, hL_ge_one, hε_le, hX_lt_L1⟩

/-- Generic weighted geometric-decay water-filling lower bound whenever `1 ≤ ∑ w` and
`ε ≤ geomVar c a R L`. -/
theorem geom_weighted_lower_bound (w : ℕ → ℕ) (L : ℕ) {c a R ε : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R)
    (hw_pos : 1 ≤ ∑ l ∈ Finset.range (L + 1), w l)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L) :
    (∑ l ∈ Finset.range (L + 1), (w l : ℝ) * ((L : ℝ) - (l : ℝ))) * Real.logb 2 (R / a) ≤
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * Real.logb 2 (geomVar c a R l / D l) := by
  have hθ_pos : 0 < geomVar c a R L := geomVar_pos hc ha hR L
  have hlam_pos : ∀ l ∈ Finset.range (L + 1), 0 < geomVar c a R l :=
    fun l _ => geomVar_pos hc ha hR l
  have hw_real_ge : (1 : ℝ) ≤ ∑ l ∈ Finset.range (L + 1), (w l : ℝ) := by
    rw [← Nat.cast_sum]
    exact_mod_cast hw_pos
  have hsum_wf :
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l ≤
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * min (geomVar c a R l) (geomVar c a R L) := by
    have hmin_sum :
        ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * min (geomVar c a R l) (geomVar c a R L) =
          (∑ l ∈ Finset.range (L + 1), (w l : ℝ)) * geomVar c a R L := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl ?_
      intro l hl
      have hle : l ≤ L := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
      rw [min_eq_right (geomVar_antitone hc ha hR hle)]
    rw [hmin_sum]
    calc
      ∑ l ∈ Finset.range (L + 1), (w l : ℝ) * D l ≤ ε := hD_tot
      _ ≤ geomVar c a R L := hε_le
      _ = 1 * geomVar c a R L := (one_mul _).symm
      _ ≤ (∑ l ∈ Finset.range (L + 1), (w l : ℝ)) * geomVar c a R L :=
        mul_le_mul_of_nonneg_right hw_real_ge hθ_pos.le
  have hwf := weighted_reverse_waterfilling_logb2_lower_bound w L (geomVar c a R) D
    (geomVar c a R L) hθ_pos hlam_pos hD_pos hD_le hsum_wf
  rw [geom_weighted_waterfilling_rate_eq w L hc ha hR] at hwf
  exact hwf

/-- 3D spherical-harmonic reverse water-filling lower bound in terms of `L` and `logb 2 (R / a)`. -/
theorem evanescent_3d_lower_bound_L (L : ℕ) {c a R ε : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L) :
    ((L : ℝ) ^ 3 / 3) * Real.logb 2 (R / a) ≤
        ((L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6) * Real.logb 2 (R / a) ∧
      ((L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6) * Real.logb 2 (R / a) ≤
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / D l) := by
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  have hw_pos : 1 ≤ ∑ l ∈ Finset.range (L + 1), (2 * l + 1) := by
    rw [sum_odd_eq_sq]
    exact Nat.one_le_pow 2 (L + 1) (Nat.succ_pos L)
  have hlb := geom_weighted_lower_bound (fun l => 2 * l + 1) L D hc ha hR hw_pos hD_pos hD_le
    hD_tot hε_le
  have hsum_eq :
      ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) =
        (L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6 :=
    sum_3d_bit_real L
  rw [hsum_eq] at hlb
  have h3_lb : (L : ℝ) ^ 3 / 3 ≤ (L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6 := by
    have hb := (sum_3d_bit_bounds L).1
    linarith [sum_3d_bit_real L]
  exact ⟨mul_le_mul_of_nonneg_right h3_lb hlogRa.le, hlb⟩

/-- Non-asymptotic 3D polylogarithmic lower bound in `(logb 2 (c / ε)) ^ 3 / (logb 2 (R / a)) ^ 2`. -/
theorem evanescent_3d_polylog_lower_bound (L : ℕ) {c a R ε : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) (hε : 0 < ε) (hL : 1 ≤ L)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L)
    (hcut : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1) :
    (Real.logb 2 (c / ε)) ^ 3 / (192 * (Real.logb 2 (R / a)) ^ 2) ≤
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
        ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / D l) := by
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  have hL_real : (1 : ℝ) ≤ (L : ℝ) := by exact_mod_cast hL
  have hhalf : Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) ≤ (L : ℝ) := by
    have h1 : (L : ℝ) + 1 ≤ 2 * (L : ℝ) := by linarith
    have h2 : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ 2 * (L : ℝ) := hcut.trans h1
    have hdiv2 :
        Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) =
          (Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))) / 2 := by
      field_simp [hlogRa.ne']
      ring
    rw [hdiv2]
    linarith
  have hX_nonneg : 0 ≤ Real.logb 2 (c / ε) := by
    have hc_ge : ε ≤ c := by
      have hgeom_le_c : geomVar c a R L ≤ c := by
        have h := geomVar_antitone hc ha hR (Nat.zero_le L)
        simpa [geomVar] using h
      exact hε_le.trans hgeom_le_c
    have h1 : 1 ≤ c / ε := (one_le_div hε).mpr hc_ge
    exact Real.logb_nonneg (by norm_num) h1
  have hcube :
      (Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 3 ≤ (L : ℝ) ^ 3 :=
    pow_le_pow_left₀ (by positivity) hhalf 3
  have hmain := (evanescent_3d_lower_bound_L L D hc ha hR hD_pos hD_le hD_tot hε_le).1.trans
    (evanescent_3d_lower_bound_L L D hc ha hR hD_pos hD_le hD_tot hε_le).2
  have hLHS :
      (Real.logb 2 (c / ε)) ^ 3 / (192 * (Real.logb 2 (R / a)) ^ 2) =
        ((Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 3 / 3) * Real.logb 2 (R / a) := by
    field_simp [hlogRa.ne']
    ring
  rw [hLHS]
  exact le_trans (mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right hcube (by norm_num)) hlogRa.le)
    hmain

/-- Asymptotic `(1 - δ)` 3D polylogarithmic lower bound recovering the sharp constant `1 / 24`. -/
theorem evanescent_3d_polylog_asymptotic_lower_bound (L : ℕ) {c a R ε δ : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L)
    (hcut : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1)
    (hlarge : 1 / δ ≤ Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))) :
    ((1 - δ) ^ 3 / 24) * ((Real.logb 2 (c / ε)) ^ 3 / (Real.logb 2 (R / a)) ^ 2) ≤
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
        ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / D l) := by
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  set X : ℝ := Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))
  have h1_le_δX : 1 ≤ δ * X := by
    have h := mul_le_mul_of_nonneg_left hlarge hδ0.le
    have hmul : δ * (1 / δ) = 1 := by field_simp [hδ0.ne']
    linarith
  have hL_ge : (1 - δ) * X ≤ (L : ℝ) := by linarith
  have hX_pos : 0 < X := lt_of_lt_of_le (by positivity) hlarge
  have h1δ_nonneg : 0 ≤ 1 - δ := by linarith
  have hcube : ((1 - δ) * X) ^ 3 ≤ (L : ℝ) ^ 3 :=
    pow_le_pow_left₀ (mul_nonneg h1δ_nonneg hX_pos.le) hL_ge 3
  have hmain := (evanescent_3d_lower_bound_L L D hc ha hR hD_pos hD_le hD_tot hε_le).1.trans
    (evanescent_3d_lower_bound_L L D hc ha hR hD_pos hD_le hD_tot hε_le).2
  have hLHS :
      ((1 - δ) ^ 3 / 24) * ((Real.logb 2 (c / ε)) ^ 3 / (Real.logb 2 (R / a)) ^ 2) =
        (((1 - δ) * X) ^ 3 / 3) * Real.logb 2 (R / a) := by
    dsimp [X]
    field_simp [hlogRa.ne']
    ring
  rw [hLHS]
  exact le_trans (mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right hcube (by norm_num)) hlogRa.le)
    hmain

/-- 2D (1-mode-per-order) reverse water-filling lower bound in terms of `L` and non-asymptotic
polylog form `(logb 2 (c / ε)) ^ 2 / (32 * logb 2 (R / a))`. -/
theorem evanescent_2d_one_mode_lower_bound (L : ℕ) {c a R ε : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) (hε : 0 < ε) (hL : 1 ≤ L)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L)
    (hcut : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1) :
    ((L : ℝ) * ((L : ℝ) + 1) / 2) * Real.logb 2 (R / a) ≤
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), Real.logb 2 (geomVar c a R l / D l) ∧
      (Real.logb 2 (c / ε)) ^ 2 / (32 * Real.logb 2 (R / a)) ≤
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), Real.logb 2 (geomVar c a R l / D l) := by
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  have hw_pos : 1 ≤ ∑ l ∈ Finset.range (L + 1), (1 : ℕ) := by simp
  have hD_tot' : ∑ l ∈ Finset.range (L + 1), ((1 : ℕ) : ℝ) * D l ≤ ε := by
    simpa using hD_tot
  have hlb := geom_weighted_lower_bound (fun _ => 1) L D hc ha hR hw_pos hD_pos hD_le hD_tot' hε_le
  have hsum_eq :
      ∑ l ∈ Finset.range (L + 1), ((1 : ℕ) : ℝ) * ((L : ℝ) - (l : ℝ)) =
        (L : ℝ) * ((L : ℝ) + 1) / 2 := by
    simpa using sum_2d_one_mode_real L
  rw [hsum_eq] at hlb
  have hlb' : ((L : ℝ) * ((L : ℝ) + 1) / 2) * Real.logb 2 (R / a) ≤
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1), Real.logb 2 (geomVar c a R l / D l) := by
    simpa using hlb
  have hL_real : (1 : ℝ) ≤ (L : ℝ) := by exact_mod_cast hL
  have hhalf : Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) ≤ (L : ℝ) := by
    have h1 : (L : ℝ) + 1 ≤ 2 * (L : ℝ) := by linarith
    have h2 : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ 2 * (L : ℝ) := hcut.trans h1
    have hdiv2 :
        Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) =
          (Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))) / 2 := by
      field_simp [hlogRa.ne']
      ring
    rw [hdiv2]
    linarith
  have hX_nonneg : 0 ≤ Real.logb 2 (c / ε) := by
    have hc_ge : ε ≤ c := by
      have h := geomVar_antitone hc ha hR (Nat.zero_le L)
      exact hε_le.trans (by simpa [geomVar] using h)
    exact Real.logb_nonneg (by norm_num) ((one_le_div hε).mpr hc_ge)
  have hsq : (Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 2 ≤ (L : ℝ) ^ 2 :=
    pow_le_pow_left₀ (by positivity) hhalf 2
  have h2_lb : (L : ℝ) ^ 2 / 2 ≤ (L : ℝ) * ((L : ℝ) + 1) / 2 := by
    have hb := (sum_2d_one_mode_bounds L).1
    linarith [sum_2d_one_mode_real L]
  have hLHS :
      (Real.logb 2 (c / ε)) ^ 2 / (32 * Real.logb 2 (R / a)) =
        ((Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 2 / 2) * Real.logb 2 (R / a) := by
    field_simp [hlogRa.ne']
    ring
  refine ⟨hlb', ?_⟩
  rw [hLHS]
  exact le_trans
    (mul_le_mul_of_nonneg_right ((div_le_div_of_nonneg_right hsq (by norm_num)).trans h2_lb)
      hlogRa.le)
    hlb'

/-- 2D full circular harmonics (`1` mode at `l = 0`, `2` modes at `1 ≤ l ≤ L`) reverse water-filling
lower bound: exact `L ^ 2 * logb 2 (R / a)` and polylog `(logb 2 (c / ε)) ^ 2 / (16 * logb 2 (R / a))`. -/
theorem evanescent_2d_full_lower_bound (L : ℕ) {c a R ε : ℝ} (D : ℕ → ℝ)
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) (hε : 0 < ε) (hL : 1 ≤ L)
    (hD_pos : ∀ l ∈ Finset.range (L + 1), 0 < D l)
    (hD_le : ∀ l ∈ Finset.range (L + 1), D l ≤ geomVar c a R l)
    (hD_tot : ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * D l ≤ ε)
    (hε_le : ε ≤ geomVar c a R L)
    (hcut : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ (L : ℝ) + 1) :
    ((L : ℝ) ^ 2) * Real.logb 2 (R / a) ≤
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          (w2dFull l : ℝ) * Real.logb 2 (geomVar c a R l / D l) ∧
      (Real.logb 2 (c / ε)) ^ 2 / (16 * Real.logb 2 (R / a)) ≤
        (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          (w2dFull l : ℝ) * Real.logb 2 (geomVar c a R l / D l) := by
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  have hw_pos : 1 ≤ ∑ l ∈ Finset.range (L + 1), w2dFull l := by
    rw [sum_w2dFull_eq]
    omega
  have hlb := geom_weighted_lower_bound w2dFull L D hc ha hR hw_pos hD_pos hD_le hD_tot hε_le
  have hsum_eq :
      ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * ((L : ℝ) - (l : ℝ)) = (L : ℝ) ^ 2 := by
    have hcast :
        ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * ((L : ℝ) - (l : ℝ)) =
          ((∑ l ∈ Finset.range (L + 1), w2dFull l * (L - l) : ℕ) : ℝ) := by
      rw [Nat.cast_sum]
      refine Finset.sum_congr rfl ?_
      intro l hl
      rw [Nat.cast_mul, nat_sub_cast_eq (Nat.lt_succ_iff.mp (Finset.mem_range.mp hl))]
    rw [hcast, sum_2d_full_weighted_nat]
    push_cast
    rfl
  rw [hsum_eq] at hlb
  have hL_real : (1 : ℝ) ≤ (L : ℝ) := by exact_mod_cast hL
  have hhalf : Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) ≤ (L : ℝ) := by
    have h1 : (L : ℝ) + 1 ≤ 2 * (L : ℝ) := by linarith
    have h2 : Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a)) ≤ 2 * (L : ℝ) := hcut.trans h1
    have hdiv2 :
        Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a)) =
          (Real.logb 2 (c / ε) / (2 * Real.logb 2 (R / a))) / 2 := by
      field_simp [hlogRa.ne']
      ring
    rw [hdiv2]
    linarith
  have hX_nonneg : 0 ≤ Real.logb 2 (c / ε) := by
    have hc_ge : ε ≤ c := by
      have h := geomVar_antitone hc ha hR (Nat.zero_le L)
      exact hε_le.trans (by simpa [geomVar] using h)
    exact Real.logb_nonneg (by norm_num) ((one_le_div hε).mpr hc_ge)
  have hsq : (Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 2 ≤ (L : ℝ) ^ 2 :=
    pow_le_pow_left₀ (by positivity) hhalf 2
  have hLHS :
      (Real.logb 2 (c / ε)) ^ 2 / (16 * Real.logb 2 (R / a)) =
        (Real.logb 2 (c / ε) / (4 * Real.logb 2 (R / a))) ^ 2 * Real.logb 2 (R / a) := by
    field_simp [hlogRa.ne']
    ring
  refine ⟨hlb, ?_⟩
  rw [hLHS]
  exact le_trans (mul_le_mul_of_nonneg_right hsq hlogRa.le) hlb

/-- Achievability / upper bound at the truncated 3D water-filling witness
`D_trunc l = c * (a / R) ^ (2 * L)`: proves feasibility, exact total distortion
`(L + 1) ^ 2 * geomVar c a R L`, and exact rate `≤ ((L + 1) ^ 3 / 3) * logb 2 (R / a)`. -/
theorem evanescent_3d_upper_bound_truncated (L : ℕ) {c a R : ℝ}
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) :
    let Dtrunc : ℕ → ℝ := fun _ => geomVar c a R L
    (∀ l ∈ Finset.range (L + 1), 0 < Dtrunc l ∧ Dtrunc l ≤ geomVar c a R l) ∧
      ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * Dtrunc l =
        ((L : ℝ) + 1) ^ 2 * geomVar c a R L ∧
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
        ((L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6) * Real.logb 2 (R / a) ∧
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) ≤
        (((L : ℝ) + 1) ^ 3 / 3) * Real.logb 2 (R / a) := by
  intro Dtrunc
  have hlogRa : 0 < Real.logb 2 (R / a) := Real.logb_pos (by norm_num) ((one_lt_div ha).mpr hR)
  have hfeas : ∀ l ∈ Finset.range (L + 1), 0 < Dtrunc l ∧ Dtrunc l ≤ geomVar c a R l := by
    intro l hl
    have hle : l ≤ L := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
    exact ⟨geomVar_pos hc ha hR L, geomVar_antitone hc ha hR hle⟩
  have hdist :
      ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * Dtrunc l =
        ((L : ℝ) + 1) ^ 2 * geomVar c a R L := by
    rw [← Finset.sum_mul]
    have hsum : ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) = ((L : ℝ) + 1) ^ 2 := by
      have hnat : ((∑ l ∈ Finset.range (L + 1), (2 * l + 1) : ℕ) : ℝ) =
          (((L + 1) ^ 2 : ℕ) : ℝ) := by
        exact_mod_cast sum_odd_eq_sq L
      rw [Nat.cast_sum] at hnat
      rw [hnat]
      push_cast
      rfl
    rw [hsum]
  have hrate :
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
        ((L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6) * Real.logb 2 (R / a) := by
    have hmin :
        ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
          ∑ l ∈ Finset.range (L + 1), ((2 * l + 1 : ℕ) : ℝ) *
            Real.logb 2 (geomVar c a R l / min (geomVar c a R l) (geomVar c a R L)) := by
      refine Finset.sum_congr rfl ?_
      intro l hl
      have hle : l ≤ L := Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)
      rw [min_eq_right (geomVar_antitone hc ha hR hle)]
    rw [hmin, geom_weighted_waterfilling_rate_eq (fun l => 2 * l + 1) L hc ha hR, sum_3d_bit_real]
  have hub :
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          ((2 * l + 1 : ℕ) : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) ≤
        (((L : ℝ) + 1) ^ 3 / 3) * Real.logb 2 (R / a) := by
    rw [hrate]
    have h3_ub : (L : ℝ) * ((L : ℝ) + 1) * (2 * (L : ℝ) + 1) / 6 ≤ ((L : ℝ) + 1) ^ 3 / 3 := by
      have hb := (sum_3d_bit_bounds L).2
      linarith [sum_3d_bit_real L]
    exact mul_le_mul_of_nonneg_right h3_ub hlogRa.le
  exact ⟨hfeas, hdist, hrate, hub⟩

/-- Achievability / upper bound at the truncated 2D full circular harmonics witness
`D_trunc l = c * (a / R) ^ (2 * L)`. -/
theorem evanescent_2d_full_upper_bound_truncated (L : ℕ) {c a R : ℝ}
    (hc : 0 < c) (ha : 0 < a) (hR : a < R) :
    let Dtrunc : ℕ → ℝ := fun _ => geomVar c a R L
    (∀ l ∈ Finset.range (L + 1), 0 < Dtrunc l ∧ Dtrunc l ≤ geomVar c a R l) ∧
      ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * Dtrunc l =
        (2 * (L : ℝ) + 1) * geomVar c a R L ∧
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          (w2dFull l : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
        (L : ℝ) ^ 2 * Real.logb 2 (R / a) := by
  intro Dtrunc
  have hfeas : ∀ l ∈ Finset.range (L + 1), 0 < Dtrunc l ∧ Dtrunc l ≤ geomVar c a R l := by
    intro l hl
    exact ⟨geomVar_pos hc ha hR L,
      geomVar_antitone hc ha hR (Nat.lt_succ_iff.mp (Finset.mem_range.mp hl))⟩
  have hdist :
      ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * Dtrunc l =
        (2 * (L : ℝ) + 1) * geomVar c a R L := by
    rw [← Finset.sum_mul]
    have hsum : ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) = 2 * (L : ℝ) + 1 := by
      have hnat : ((∑ l ∈ Finset.range (L + 1), w2dFull l : ℕ) : ℝ) =
          ((2 * L + 1 : ℕ) : ℝ) := by
        exact_mod_cast sum_w2dFull_eq L
      rw [Nat.cast_sum] at hnat
      push_cast at hnat
      exact hnat
    rw [hsum]
  have hrate :
      (1 / 2 : ℝ) * ∑ l ∈ Finset.range (L + 1),
          (w2dFull l : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
        (L : ℝ) ^ 2 * Real.logb 2 (R / a) := by
    have hmin :
        ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * Real.logb 2 (geomVar c a R l / Dtrunc l) =
          ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) *
            Real.logb 2 (geomVar c a R l / min (geomVar c a R l) (geomVar c a R L)) := by
      refine Finset.sum_congr rfl ?_
      intro l hl
      rw [min_eq_right (geomVar_antitone hc ha hR (Nat.lt_succ_iff.mp (Finset.mem_range.mp hl)))]
    rw [hmin, geom_weighted_waterfilling_rate_eq w2dFull L hc ha hR]
    have hcast :
        ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * ((L : ℝ) - (l : ℝ)) = (L : ℝ) ^ 2 := by
      have hnat : ((∑ l ∈ Finset.range (L + 1), w2dFull l * (L - l) : ℕ) : ℝ) =
          ((L ^ 2 : ℕ) : ℝ) := by
        exact_mod_cast sum_2d_full_weighted_nat L
      rw [Nat.cast_sum] at hnat
      have hsub :
          ∑ l ∈ Finset.range (L + 1), (w2dFull l : ℝ) * ((L : ℝ) - (l : ℝ)) =
            ∑ l ∈ Finset.range (L + 1), ((w2dFull l * (L - l) : ℕ) : ℝ) := by
        refine Finset.sum_congr rfl ?_
        intro l hl
        rw [Nat.cast_mul, nat_sub_cast_eq (Nat.lt_succ_iff.mp (Finset.mem_range.mp hl))]
      rw [hsub, hnat]
      push_cast
      rfl
    rw [hcast]
  exact ⟨hfeas, hdist, hrate⟩

/-! ## Section 4: Strict Geometric Denominator Monotonicity (`R₂ < R₁`) -/

/-- When an active near-field probe shrinks the effective observation radius from `R₁` to
`R₂ < R₁` (with `0 < a < R₂`), `logb 2 (R / a)` strictly decreases while remaining strictly positive. -/
theorem geom_logb2_pos_and_strictMono {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    0 < Real.logb 2 (R₂ / a) ∧ Real.logb 2 (R₂ / a) < Real.logb 2 (R₁ / a) := by
  have h1 : 1 < R₂ / a := (one_lt_div ha).mpr hR₂
  have h2 : R₂ / a < R₁ / a := div_lt_div_of_pos_right hR₁₂ ha
  have hpos₂ : 0 < R₂ / a := by linarith
  exact ⟨Real.logb_pos (by norm_num) h1, Real.logb_lt_logb (by norm_num) hpos₂ h2⟩

/-- Strict upward jump of the 2D leading polylog coefficient `1 / logb 2 (R / a)` when
`R₁` drops to `R₂ < R₁`. -/
theorem near_field_jump_2d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    1 / Real.logb 2 (R₁ / a) < 1 / Real.logb 2 (R₂ / a) := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  exact one_div_lt_one_div_of_lt hpos₂ hlt

/-- Strict upward jump of the 3D leading polylog coefficient `1 / (logb 2 (R / a)) ^ 2` when
`R₁` drops to `R₂ < R₁`. -/
theorem near_field_jump_3d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    1 / (Real.logb 2 (R₁ / a)) ^ 2 < 1 / (Real.logb 2 (R₂ / a)) ^ 2 := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  have hsq_pos : 0 < (Real.logb 2 (R₂ / a)) ^ 2 := sq_pos_of_pos hpos₂
  have hsq_lt : (Real.logb 2 (R₂ / a)) ^ 2 < (Real.logb 2 (R₁ / a)) ^ 2 := by
    nlinarith
  exact one_div_lt_one_div_of_lt hsq_pos hsq_lt

/-- Exact 2D near-field jump ratio `(> 1)` when `R₁` drops to `R₂ < R₁`. -/
theorem near_field_jump_ratio_2d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    (1 / Real.logb 2 (R₂ / a)) / (1 / Real.logb 2 (R₁ / a)) =
        Real.logb 2 (R₁ / a) / Real.logb 2 (R₂ / a) ∧
      1 < Real.logb 2 (R₁ / a) / Real.logb 2 (R₂ / a) := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  have hpos₁ : 0 < Real.logb 2 (R₁ / a) := hpos₂.trans hlt
  refine ⟨?_, (one_lt_div hpos₂).mpr hlt⟩
  field_simp [hpos₂.ne', hpos₁.ne']

/-- Exact 3D near-field jump ratio `(> 1)` when `R₁` drops to `R₂ < R₁`. -/
theorem near_field_jump_ratio_3d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    (1 / (Real.logb 2 (R₂ / a)) ^ 2) / (1 / (Real.logb 2 (R₁ / a)) ^ 2) =
        (Real.logb 2 (R₁ / a) / Real.logb 2 (R₂ / a)) ^ 2 ∧
      1 < (Real.logb 2 (R₁ / a) / Real.logb 2 (R₂ / a)) ^ 2 := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  have hpos₁ : 0 < Real.logb 2 (R₁ / a) := hpos₂.trans hlt
  have hratio_gt : 1 < Real.logb 2 (R₁ / a) / Real.logb 2 (R₂ / a) := (one_lt_div hpos₂).mpr hlt
  refine ⟨?_, ?_⟩
  · field_simp [hpos₂.ne', hpos₁.ne']
  · nlinarith

/-- Exact additive gap identity (`> 0`) for the 2D leading denominator reciprocal. -/
theorem near_field_additive_gap_2d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    1 / Real.logb 2 (R₂ / a) - 1 / Real.logb 2 (R₁ / a) =
        Real.logb 2 (R₁ / R₂) / (Real.logb 2 (R₂ / a) * Real.logb 2 (R₁ / a)) ∧
      0 < Real.logb 2 (R₁ / R₂) / (Real.logb 2 (R₂ / a) * Real.logb 2 (R₁ / a)) := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  have hpos₁ : 0 < Real.logb 2 (R₁ / a) := hpos₂.trans hlt
  have hR₂_pos : 0 < R₂ := ha.trans hR₂
  have hR₁_pos : 0 < R₁ := hR₂_pos.trans hR₁₂
  have hsub : Real.logb 2 (R₁ / a) - Real.logb 2 (R₂ / a) = Real.logb 2 (R₁ / R₂) := by
    rw [Real.logb_div hR₁_pos.ne' ha.ne', Real.logb_div hR₂_pos.ne' ha.ne',
      Real.logb_div hR₁_pos.ne' hR₂_pos.ne']
    ring
  have hlog₁₂_pos : 0 < Real.logb 2 (R₁ / R₂) :=
    Real.logb_pos (by norm_num) ((one_lt_div hR₂_pos).mpr hR₁₂)
  refine ⟨?_, div_pos hlog₁₂_pos (mul_pos hpos₂ hpos₁)⟩
  rw [← hsub]
  field_simp [hpos₂.ne', hpos₁.ne']

/-- Exact additive gap identity (`> 0`) for the 3D leading denominator reciprocal. -/
theorem near_field_additive_gap_3d {a R₂ R₁ : ℝ}
    (ha : 0 < a) (hR₂ : a < R₂) (hR₁₂ : R₂ < R₁) :
    1 / (Real.logb 2 (R₂ / a)) ^ 2 - 1 / (Real.logb 2 (R₁ / a)) ^ 2 =
        (Real.logb 2 (R₁ / R₂) * (Real.logb 2 (R₁ / a) + Real.logb 2 (R₂ / a))) /
          ((Real.logb 2 (R₂ / a)) ^ 2 * (Real.logb 2 (R₁ / a)) ^ 2) ∧
      0 < (Real.logb 2 (R₁ / R₂) * (Real.logb 2 (R₁ / a) + Real.logb 2 (R₂ / a))) /
        ((Real.logb 2 (R₂ / a)) ^ 2 * (Real.logb 2 (R₁ / a)) ^ 2) := by
  obtain ⟨hpos₂, hlt⟩ := geom_logb2_pos_and_strictMono ha hR₂ hR₁₂
  have hpos₁ : 0 < Real.logb 2 (R₁ / a) := hpos₂.trans hlt
  have hR₂_pos : 0 < R₂ := ha.trans hR₂
  have hR₁_pos : 0 < R₁ := hR₂_pos.trans hR₁₂
  have hsub : Real.logb 2 (R₁ / a) - Real.logb 2 (R₂ / a) = Real.logb 2 (R₁ / R₂) := by
    rw [Real.logb_div hR₁_pos.ne' ha.ne', Real.logb_div hR₂_pos.ne' ha.ne',
      Real.logb_div hR₁_pos.ne' hR₂_pos.ne']
    ring
  have hlog₁₂_pos : 0 < Real.logb 2 (R₁ / R₂) :=
    Real.logb_pos (by norm_num) ((one_lt_div hR₂_pos).mpr hR₁₂)
  refine ⟨?_, div_pos (mul_pos hlog₁₂_pos (add_pos hpos₁ hpos₂)) (mul_pos (sq_pos_of_pos hpos₂) (sq_pos_of_pos hpos₁))⟩
  rw [← hsub]
  field_simp [hpos₂.ne', hpos₁.ne']
  ring

#print axioms sum_odd_eq_sq
#print axioms sum_3d_bit_nat
#print axioms sum_3d_bit_real
#print axioms sum_3d_bit_bounds
#print axioms sum_2d_one_mode_nat
#print axioms sum_2d_one_mode_real
#print axioms sum_2d_one_mode_bounds
#print axioms sum_2d_two_mode_nat
#print axioms sum_2d_two_mode_real
#print axioms sum_2d_full_nat
#print axioms sum_2d_full_real
#print axioms sum_w2dFull_eq
#print axioms sum_2d_full_weighted_nat
#print axioms sum_2d_gen_mode_real
#print axioms sum_2d_gen_mode_zero_to_L_bounds
#print axioms waterfilling_pointwise_lb
#print axioms waterfilling_pointwise_logb2_lb
#print axioms waterfilling_feasible
#print axioms reverse_waterfilling_logb2_lagrangian
#print axioms reverse_waterfilling_logb2_lower_bound
#print axioms weighted_reverse_waterfilling_logb2_lower_bound
#print axioms geomVar_pos
#print axioms geomVar_antitone
#print axioms geomVar_ratio_logb2
#print axioms geom_weighted_waterfilling_rate_eq
#print axioms exists_evanescent_cutoff
#print axioms geom_weighted_lower_bound
#print axioms evanescent_3d_lower_bound_L
#print axioms evanescent_3d_polylog_lower_bound
#print axioms evanescent_3d_polylog_asymptotic_lower_bound
#print axioms evanescent_2d_one_mode_lower_bound
#print axioms evanescent_2d_full_lower_bound
#print axioms evanescent_3d_upper_bound_truncated
#print axioms evanescent_2d_full_upper_bound_truncated
#print axioms geom_logb2_pos_and_strictMono
#print axioms near_field_jump_2d
#print axioms near_field_jump_3d
#print axioms near_field_jump_ratio_2d
#print axioms near_field_jump_ratio_3d
#print axioms near_field_additive_gap_2d
#print axioms near_field_additive_gap_3d

end MathLab.LateQueryFrontier.WaveField
