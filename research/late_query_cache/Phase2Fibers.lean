import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.MeasureTheory.Integral.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! The following incidence estimate is independent of any regularity of the two
decoder curves beyond their measurability in the Tonelli statement. -/

open Set MeasureTheory
open scoped ENNReal

namespace Phase2Fibers

/-- The two queries predict z by f(y) and g(x) + x*y respectively. -/
def sectionAt (f g : ℝ → ℝ) (r x : ℝ) : Set ℝ :=
  {y | |f y - g x - x*y| ≤ 2*r}

/-- Projecting two successful reconstructions onto the (x,y) plane gives a strip. -/
theorem projectHits {f g : ℝ → ℝ} {r x y z : ℝ}
    (h₁ : |z - f y| ≤ r) (h₂ : |z - x*y - g x| ≤ r) :
    y ∈ sectionAt f g r x := by
  obtain ⟨h₁l, h₁u⟩ := abs_le.mp h₁
  obtain ⟨h₂l, h₂u⟩ := abs_le.mp h₂
  apply abs_le.mpr
  constructor <;> linarith

/-- Eliminating the same f(y) twice traps y in an interval of width
8*r/|x-x'|. This holds for arbitrary functions f,g. -/
theorem intersectionStrip {f g : ℝ → ℝ} {r x x' y : ℝ} (h : x ≠ x')
    (hy : y ∈ sectionAt f g r x ∩ sectionAt f g r x') :
    |y + (g x - g x') / (x-x')| ≤ 4*r / |x-x'| := by
  have hd : x - x' ≠ 0 := sub_ne_zero.mpr h
  have h₁ : |f y - g x - x*y| ≤ 2*r := hy.1
  have h₂ : |f y - g x' - x'*y| ≤ 2*r := hy.2
  obtain ⟨h₁l, h₁u⟩ := abs_le.mp h₁
  obtain ⟨h₂l, h₂u⟩ := abs_le.mp h₂
  have hbase : |(x-x')*y+(g x-g x')| ≤ 4*r := by
    apply abs_le.mpr
    constructor <;> nlinarith
  apply (le_div_iff₀ (abs_pos.mpr hd)).mpr
  calc
    |y + (g x - g x') / (x-x')| * |x-x'|
        = |(y + (g x - g x') / (x-x')) * (x-x')| := (abs_mul _ _).symm
    _ = |(x-x')*y+(g x-g x')| := by
      congr 1
      field_simp [hd] <;> ring
    _ ≤ _ := hbase

/-- Lebesgue measure of the common section; no finite counting surrogate. -/
theorem overlapVolume {f g : ℝ → ℝ} {r x x' : ℝ} (h : x ≠ x') :
    volume (sectionAt f g r x ∩ sectionAt f g r x') ≤
      ENNReal.ofReal (8*r / |x-x'|) := by
  let c := -((g x - g x') / (x-x'))
  let w := 4*r / |x-x'|
  have hsub : sectionAt f g r x ∩ sectionAt f g r x' ⊆ Icc (c-w) (c+w) := by
    intro y hy
    obtain ⟨hl, hu⟩ := abs_le.mp (intersectionStrip h hy)
    dsimp [c, w]
    constructor <;> linarith
  calc
    _ ≤ volume (Icc (c-w) (c+w)) := measure_mono hsub
    _ = _ := by
      rw [Real.volume_Icc]
      congr 1
      dsimp [c,w]
      ring

/-- A three-dimensional pair-incidence event: the same vertical coordinate
belongs to both decoder sections and to the unit interval. -/
def pairEvent (f g : ℝ → ℝ) (r : ℝ) : Set ((ℝ × ℝ) × ℝ) :=
  {p | p.2 ∈ Icc (0:ℝ) 1 ∧ p.2 ∈ sectionAt f g r p.1.1 ∧ p.2 ∈ sectionAt f g r p.1.2}

/-- Product-measure converse interface. The law of the two abscissas may be any
measure, including the product of two independent uniforms, and may charge
the diagonal. The outer integral is in the extended nonnegative reals. -/
theorem pairIncidenceIntegral (μ : Measure (ℝ × ℝ)) {f g : ℝ → ℝ}
    (hf : Measurable f) (hg : Measurable g) (r : ℝ) :
    (μ.prod volume) (pairEvent f g r) ≤
      ∫⁻ p, if p.1 = p.2 then (1 : ℝ≥0∞)
        else min 1 (ENNReal.ofReal (8*r / |p.1-p.2|)) ∂μ := by
  have h₀ : MeasurableSet {p : (ℝ × ℝ) × ℝ | p.2 ∈ Icc (0:ℝ) 1} :=
    measurable_snd measurableSet_Icc
  have h₁ : Measurable (fun p : (ℝ × ℝ) × ℝ => |f p.2 - g p.1.1 - p.1.1*p.2|) := by
    fun_prop
  have h₂ : Measurable (fun p : (ℝ × ℝ) × ℝ => |f p.2 - g p.1.2 - p.1.2*p.2|) := by
    fun_prop
  have he : MeasurableSet (pairEvent f g r) :=
    h₀.inter ((measurableSet_le h₁ measurable_const).inter (measurableSet_le h₂ measurable_const))
  rw [Measure.prod_apply he]
  apply lintegral_mono
  intro p
  have hs₀ : Prod.mk p ⁻¹' pairEvent f g r ⊆ Icc (0:ℝ) 1 := fun _ hy => hy.1
  have hs₁ : volume (Prod.mk p ⁻¹' pairEvent f g r) ≤ 1 := by
    calc
      _ ≤ volume (Icc (0:ℝ) 1) := measure_mono hs₀
      _ = 1 := by norm_num [Real.volume_Icc]
  by_cases hh : p.1 = p.2
  · simpa only [if_pos hh] using hs₁
  · simp only [if_neg hh]
    apply le_min hs₁
    have hs : Prod.mk p ⁻¹' pairEvent f g r ⊆ sectionAt f g r p.1 ∩ sectionAt f g r p.2 :=
      fun _ hy => ⟨hy.2.1, hy.2.2⟩
    exact (measure_mono hs).trans (overlapVolume hh)


/-- Square of an ordinary nonnegative integral on a probability space is
bounded by its second moment, also for extended (infinite) values. -/
theorem squareMass {β : Type*} [MeasurableSpace β] (ν : Measure β)
    [IsProbabilityMeasure ν] {h : β → ℝ≥0∞} (hm : Measurable h) :
    (∫⁻ y, h y ∂ν)^2 ≤ ∫⁻ y, (h y)^2 ∂ν := by
  have hc : (∫⁻ y, h y ∂ν) ≤ (∫⁻ y, (h y)^2 ∂ν) ^ ((1:ℝ)/2) := by
    simpa [Pi.mul_apply, ENNReal.rpow_two] using
      (ENNReal.lintegral_mul_le_Lp_mul_Lq ν Real.HolderConjugate.two_two
        hm.aemeasurable (g := fun _ => (1:ℝ≥0∞)) measurable_const.aemeasurable)
  calc
    _ ≤ ((∫⁻ y, (h y)^2 ∂ν) ^ ((1:ℝ)/2))^(2:ℕ) := pow_le_pow_left' hc 2
    _ = _ := by
      rw [← ENNReal.rpow_two, ← ENNReal.rpow_mul]
      norm_num

/-- Probability that two independent points are incident with a shared y
bounds the squared probability of a single incidence, expressed directly
as measures of product-space sets. -/
theorem collisionSecondMoment {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SFinite μ] [IsProbabilityMeasure ν]
    {s : Set (α × β)} (hs : MeasurableSet s) :
    (μ.prod ν s)^2 ≤ ((μ.prod μ).prod ν)
      {p : (α × α) × β | (p.1.1,p.2) ∈ s ∧ (p.1.2,p.2) ∈ s} := by
  have hu : Measurable (fun p : (α × α) × β => (p.1.1,p.2)) := by fun_prop
  have hv : Measurable (fun p : (α × α) × β => (p.1.2,p.2)) := by fun_prop
  have hd : MeasurableSet {p : (α × α) × β | (p.1.1,p.2) ∈ s ∧ (p.1.2,p.2) ∈ s} :=
    (hu hs).inter (hv hs)
  rw [Measure.prod_apply_symm hs, Measure.prod_apply_symm hd]
  refine (squareMass ν (measurable_measure_prodMk_right (μ := μ) hs)).trans ?_
  apply le_of_eq
  apply lintegral_congr
  intro y
  change (μ ((fun x => (x,y)) ⁻¹' s))^2 =
    (μ.prod μ) (((fun x => (x,y)) ⁻¹' s) ×ˢ ((fun x => (x,y)) ⁻¹' s))
  rw [Measure.prod_prod, pow_two]


/-- The four-point exponential mixed increment controls both independent
scaled displacements from below, uniformly for base exponent in [0,+∞). -/
theorem expMixedLower {base u v : ℝ} (hb : 0 ≤ base) (hu : 0 ≤ u) (hv : 0 ≤ v) :
    u*v ≤ Real.exp (base+u+v) - Real.exp (base+u) - Real.exp (base+v) + Real.exp base := by
  have h₀ : 1 ≤ Real.exp base := by linarith [Real.add_one_le_exp base]
  have h₁ : u ≤ Real.exp u-1 := by linarith [Real.add_one_le_exp u]
  have h₂ : v ≤ Real.exp v-1 := by linarith [Real.add_one_le_exp v]
  have hn : 0 ≤ (Real.exp u-1)*(Real.exp v-1) :=
    mul_nonneg (hu.trans h₁) (hv.trans h₂)
  calc
    u*v ≤ (Real.exp u-1)*(Real.exp v-1) :=
      mul_le_mul h₁ h₂ hv (hu.trans h₁)
    _ ≤ Real.exp base*((Real.exp u-1)*(Real.exp v-1)) := by
      simpa only [one_mul] using mul_le_mul_of_nonneg_right h₀ hn
    _ = _ := by simp only [Real.exp_add]; ring

/-- For one pair of actions at distance D>0, choose axes so that the four
exponents have increasing displacements dk/D and dl/D, with the smallest
exponent base≥0. The two arbitrary decoder functions have values F₀,F₁
and G₀,G₁ at these coordinates. Four simultaneous hits force a k-diameter
bound uniform in the action gap. -/
theorem expEndpointDiameter {base dk dl D r F₀ F₁ G₀ G₁ : ℝ}
    (hb : 0 ≤ base) (hk : 0 ≤ dk) (hl : 0 < dl) (hD : 0 < D)
    (h₀₀ : |F₀ - G₀ - Real.exp base| ≤ 2*r)
    (h₁₀ : |F₁ - G₀ - Real.exp (base+dk/D)| ≤ 2*r)
    (h₀₁ : |F₀ - G₁ - Real.exp (base+dl/D)| ≤ 2*r)
    (h₁₁ : |F₁ - G₁ - Real.exp (base+dk/D+dl/D)| ≤ 2*r) :
    dk ≤ 8*r*D^2/dl := by
  obtain ⟨h₀₀l,h₀₀u⟩ := abs_le.mp h₀₀
  obtain ⟨h₁₀l,h₁₀u⟩ := abs_le.mp h₁₀
  obtain ⟨h₀₁l,h₀₁u⟩ := abs_le.mp h₀₁
  obtain ⟨h₁₁l,h₁₁u⟩ := abs_le.mp h₁₁
  have h := expMixedLower hb (div_nonneg hk hD.le) (div_nonneg hl.le hD.le)
  have hp : (dk/D)*(dl/D) ≤ 8*r := by linarith
  apply (le_div_iff₀ hl).mpr
  calc
    dk*dl = (dk/D)*(dl/D)*D^2 := by field_simp [ne_of_gt hD] <;> ring
    _ ≤ (8*r)*D^2 := mul_le_mul_of_nonneg_right hp (sq_nonneg D)

/-- Converts an arbitrary (even nonmeasurable) subset's diameter certificate
into a Lebesgue outer-measure estimate. This is the actual slice-volume step. -/
theorem measureFromPairwise {S : Set ℝ} {w : ℝ}
    (pairwise_bound : ∀ x ∈ S, ∀ y ∈ S, |x-y| ≤ w) :
    volume S ≤ ENNReal.ofReal w := by
  apply (Real.volume_le_diam S).trans
  apply EMetric.diam_le
  intro x hx y hy
  rw [edist_dist, Real.dist_eq]
  exact ENNReal.ofReal_le_ofReal (pairwise_bound x hx y hy)

end Phase2Fibers

#print axioms Phase2Fibers.projectHits
#print axioms Phase2Fibers.intersectionStrip
#print axioms Phase2Fibers.overlapVolume
#print axioms Phase2Fibers.pairIncidenceIntegral

#print axioms Phase2Fibers.squareMass
#print axioms Phase2Fibers.collisionSecondMoment

#print axioms Phase2Fibers.expMixedLower
#print axioms Phase2Fibers.expEndpointDiameter
#print axioms Phase2Fibers.measureFromPairwise
