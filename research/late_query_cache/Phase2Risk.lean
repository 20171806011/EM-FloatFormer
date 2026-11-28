import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Quantitative noisy-neighbor risk bound on an arbitrary probability space. -/

open MeasureTheory ProbabilityTheory Set

namespace LateQueryPhase2.Risk

/-- One coarse step of error before taking floors means at most one bin of uncertainty,
    including negative preliminary bins and boundary points. -/
theorem floorBinWindow (x obs step : ℝ) (hs : 0 < step) (h : |x - obs| ≤ step) :
    ⌊x / step⌋ - 1 ≤ ⌊obs / step⌋ ∧ ⌊obs / step⌋ ≤ ⌊x / step⌋ + 1 := by
  have h' : |x / step - obs / step| ≤ 1 := by
    rw [← sub_div, abs_div, abs_of_pos hs, div_le_iff₀ hs, one_mul]
    exact h
  obtain ⟨hlo, hhi⟩ := abs_le.mp h'
  constructor
  · rw [← Int.floor_sub_one]
    apply Int.floor_le_floor
    linarith
  · rw [← Int.floor_add_one]
    apply Int.floor_le_floor
    linarith

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

def tail (z : Ω → ℝ) (t : ℝ) : Set Ω := {ω | t ≤ z ω} ∪ {ω | t ≤ -z ω}

theorem tailMeasurable (z : Ω → ℝ) (hz : Measurable z) (t : ℝ) :
    MeasurableSet (tail z t) :=
  (measurableSet_le measurable_const hz).union (measurableSet_le measurable_const hz.neg)

theorem normalSubgaussian (z : Ω → ℝ) (hz : Measurable z)
    (law : μ.map z = gaussianReal 0 1) : HasSubgaussianMGF z 1 μ := by
  have h : HasSubgaussianMGF (id : ℝ → ℝ) 1 (gaussianReal 0 1) := by
    constructor
    · intro t
      exact integrable_exp_mul_gaussianReal t
    · intro t
      simp [mgf_id_gaussianReal]
  rw [← law] at h
  exact HasSubgaussianMGF.of_map (X := id) hz.aemeasurable h

theorem twoSidedTail (z : Ω → ℝ) (hz : HasSubgaussianMGF z 1 μ) {t : ℝ} (ht : 0 ≤ t) :
    μ.real (tail z t) ≤ 2 * Real.exp (-t^2 / 2) := by
  have hp : μ.real {ω | t ≤ z ω} ≤ Real.exp (-t^2 / 2) := by
    simpa using hz.measure_ge_le ht
  have hn : μ.real {ω | t ≤ -z ω} ≤ Real.exp (-t^2 / 2) := by
    simpa using hz.neg.measure_ge_le ht
  exact (measureReal_union_le _ _).trans (by linarith [hp, hn])

/-- Continuous expectation, using the real measure of the bad event. -/
theorem splitRisk (loss : Ω → ℝ) (hl : Integrable loss μ) (bad : Set Ω)
    (hb : MeasurableSet bad) (δ : ℝ)
    (good : ∀ ω ∉ bad, loss ω ≤ δ^2 / 4)
    (clipped : ∀ ω, loss ω ≤ 1) :
    (∫ ω, loss ω ∂μ) ≤ δ^2 / 4 + μ.real bad := by
  have hconst : Integrable (fun _ : Ω => δ^2 / 4) μ := integrable_const _
  have hind : Integrable (bad.indicator (1 : Ω → ℝ)) μ :=
    (integrable_const (μ := μ) (1 : ℝ)).indicator hb
  calc
    (∫ ω, loss ω ∂μ) ≤ ∫ ω, δ^2 / 4 + bad.indicator (1 : Ω → ℝ) ω ∂μ := by
      apply integral_mono hl (hconst.add hind)
      intro ω
      change loss ω ≤ δ^2 / 4 + bad.indicator (1 : Ω → ℝ) ω
      by_cases h : ω ∈ bad
      · simp only [indicator_of_mem h, Pi.one_apply]
        linarith [clipped ω, sq_nonneg δ]
      · simpa only [indicator_of_notMem h, add_zero] using good ω h
    _ = δ^2 / 4 + μ.real bad := by
      rw [integral_add hconst hind, integral_const, integral_indicator_one hb]
      simp

/-- No independence between the two Gaussian noises is required for this upper bound.
    The good-event premise is the deterministic fine-quantizer/side-word specification. -/
theorem twoNeighborDoubleScale (z₁ z₂ loss : Ω → ℝ) (h₁ : Measurable z₁)
    (h₂ : Measurable z₂) (law₁ : μ.map z₁ = gaussianReal 0 1)
    (law₂ : μ.map z₂ = gaussianReal 0 1) (hl : Integrable loss μ)
    (δ t : ℝ) (ht : 0 ≤ t)
    (good : ∀ ω ∉ tail z₁ t ∪ tail z₂ t, loss ω ≤ δ^2 / 4)
    (clipped : ∀ ω, loss ω ≤ 1) :
    (∫ ω, loss ω ∂μ) ≤ δ^2 / 4 + 4 * Real.exp (-t^2 / 2) := by
  have hb := (tailMeasurable z₁ h₁ t).union (tailMeasurable z₂ h₂ t)
  have hr := splitRisk loss hl _ hb δ good clipped
  have hu := measureReal_union_le (μ := μ) (tail z₁ t) (tail z₂ t)
  have hP := twoSidedTail z₁ (normalSubgaussian z₁ h₁ law₁) ht
  have hQ := twoSidedTail z₂ (normalSubgaussian z₂ h₂ law₂) ht
  linarith

/-- The same guard applies to an arbitrary finite (possibly empty) neighbor list. -/
def neighborTails (t : ℝ) : List (Ω → ℝ) → Set Ω
  | [] => ∅
  | z :: zs => tail z t ∪ neighborTails t zs

theorem neighborTailMeasurable (t : ℝ) (zs : List (Ω → ℝ))
    (hm : ∀ z ∈ zs, Measurable z) : MeasurableSet (neighborTails t zs) := by
  induction zs with
  | nil => exact MeasurableSet.empty
  | cons z zs ih =>
    exact (tailMeasurable z (hm z (by simp)) t).union
      (ih (fun u hu => hm u (List.mem_cons_of_mem z hu)))

theorem finiteGaussianTails (zs : List (Ω → ℝ)) {t : ℝ} (ht : 0 ≤ t)
    (hm : ∀ z ∈ zs, Measurable z) (hlaw : ∀ z ∈ zs, μ.map z = gaussianReal 0 1) :
    μ.real (neighborTails t zs) ≤ (2 * (zs.length : ℝ)) * Real.exp (-t^2 / 2) := by
  induction zs with
  | nil => simp [neighborTails]
  | cons z zs ih =>
    have hrest := ih (fun u hu => hm u (List.mem_cons_of_mem z hu))
      (fun u hu => hlaw u (List.mem_cons_of_mem z hu))
    have hfirst := twoSidedTail z (normalSubgaussian z (hm z (by simp))
      (hlaw z (by simp))) ht
    calc
      μ.real (neighborTails t (z :: zs)) ≤ μ.real (tail z t) + μ.real (neighborTails t zs) :=
        measureReal_union_le _ _
      _ ≤ 2 * Real.exp (-t^2 / 2) + (2 * (zs.length : ℝ)) * Real.exp (-t^2 / 2) :=
        add_le_add hfirst hrest
      _ = (2 * ((z :: zs).length : ℝ)) * Real.exp (-t^2 / 2) := by
        simp only [List.length_cons, Nat.cast_add, Nat.cast_one]
        ring

/-- Finite-neighbor operational risk: independent Gaussian components are unnecessary.
Repeating an entry is allowed, and simply counts it again in the union bound. -/
theorem finiteNeighborDoubleScale (zs : List (Ω → ℝ)) (loss : Ω → ℝ)
    (hm : ∀ z ∈ zs, Measurable z) (hlaw : ∀ z ∈ zs, μ.map z = gaussianReal 0 1)
    (hl : Integrable loss μ) (δ t : ℝ) (ht : 0 ≤ t)
    (good : ∀ ω ∉ neighborTails t zs, loss ω ≤ δ^2 / 4)
    (clipped : ∀ ω, loss ω ≤ 1) :
    (∫ ω, loss ω ∂μ) ≤ δ^2 / 4 + (2 * (zs.length : ℝ)) * Real.exp (-t^2 / 2) :=
  (splitRisk loss hl _ (neighborTailMeasurable t zs hm) δ good clipped).trans
    (add_le_add_left (finiteGaussianTails zs ht hm hlaw) _)


/-- Exact log-tail calibration used in the achievability proof. -/
theorem calibratedRisk {risk δ ε t : ℝ} (he : 0 < ε)
    (hg : δ^2 / 4 ≤ ε / 2) (ht : t^2 = 2 * Real.log (8 / ε))
    (h : risk ≤ δ^2 / 4 + 4 * Real.exp (-t^2 / 2)) : risk ≤ ε := by
  have h8 : 0 < 8 / ε := div_pos (by norm_num) he
  have hexp : Real.exp (-t^2 / 2) = ε / 8 := by
    rw [ht, show -(2 * Real.log (8 / ε)) / 2 = -Real.log (8 / ε) by ring,
      Real.exp_neg, Real.exp_log h8, inv_div]
  rw [hexp] at h
  linarith

end LateQueryPhase2.Risk

#print axioms LateQueryPhase2.Risk.floorBinWindow
#print axioms LateQueryPhase2.Risk.normalSubgaussian
#print axioms LateQueryPhase2.Risk.twoSidedTail
#print axioms LateQueryPhase2.Risk.splitRisk
#print axioms LateQueryPhase2.Risk.twoNeighborDoubleScale
#print axioms LateQueryPhase2.Risk.calibratedRisk
#print axioms LateQueryPhase2.Risk.finiteGaussianTails
#print axioms LateQueryPhase2.Risk.finiteNeighborDoubleScale
