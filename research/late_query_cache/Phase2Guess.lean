import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! One-shot entropy versus the independent-set number of a confusion graph.
    Every deterministic decoder profile can succeed on at most α words.
    Weighting profiles by independent random decoder laws gives the expected
    cardinality bound; the finite Gibbs inequality converts it to an entropy
    converse. Natural logarithms are used; division by log 2 changes units. -/

namespace Phase2Guess
open scoped BigOperators

/-- Generate a whole local lookup table by independently drawing every entry
    from that context's posterior. Its probability of returning the requested
    answer at a designated context is exactly the corresponding posterior. -/
theorem independentTableMarginal {C A : Type*} [Fintype C] [DecidableEq C]
    [Fintype A] [DecidableEq A] (ρ : C → A → ℝ) (hn : ∀ c, ∑ a, ρ c a = 1)
    (c : C) (a : A) :
    (∑ f : C → A, (∏ j, ρ j (f j)) * (if f c = a then 1 else 0)) = ρ c a := by
  classical
  let t : C → A → ℝ := fun j y => if j = c ∧ y ≠ a then 0 else ρ j y
  have hr (j : C) : (∑ y, t j y) = if j = c then ρ c a else 1 := by
    by_cases h : j = c
    · subst j
      rw [if_pos rfl]
      calc
        (∑ y, t c y) = ∑ y, if y = a then ρ c y else 0 := by
          apply Finset.sum_congr rfl
          intro y _
          by_cases hy : y = a <;> simp [t, hy]
        _ = ρ c a := by simp
    · simp [t, h, hn]
  calc
    _ = ∑ f : C → A, ∏ j, t j (f j) := by
      apply Finset.sum_congr rfl
      intro f _
      by_cases h : f c = a
      · rw [if_pos h, mul_one]
        apply Finset.prod_congr rfl
        intro j _
        by_cases hj : j = c <;> simp [t, hj, h]
      · rw [if_neg h, mul_zero]
        symm
        exact Finset.prod_eq_zero (Finset.mem_univ c) (by simp [t, h])
    _ = ∏ j, ∑ y, t j y := (Fintype.prod_sum t).symm
    _ = ρ c a := by simp only [hr]; simp

/-- A weighted finite log-sum inequality, allowing zero posterior masses. -/
theorem logMassBound {W : Type*} [Fintype W] (p w : W → ℝ) (A : ℝ)
    (hA : 0 < A) (hp : ∀ x, 0 ≤ p x) (hn : ∑ x, p x = 1)
    (hw : ∀ x, 0 ≤ w x) (hpos : ∀ x, 0 < p x → 0 < w x)
    (hmass : ∑ x, w x ≤ A) :
    (∑ x, p x * Real.log (w x / p x)) ≤ Real.log A := by
  classical
  have one (x : W) :
      p x * Real.log (w x / p x) ≤ p x * Real.log A + w x / A - p x := by
    by_cases h : p x = 0
    · simpa [h] using div_nonneg (hw x) hA.le
    · have hx : 0 < p x := lt_of_le_of_ne (hp x) (Ne.symm h)
      have hy := hpos x hx
      have ht := Real.log_le_sub_one_of_pos (div_pos (div_pos hy hx) hA)
      rw [Real.log_div (div_ne_zero hy.ne' hx.ne') hA.ne'] at ht
      have ht' := mul_le_mul_of_nonneg_left ht hx.le
      have eqv : p x * (w x / p x / A - 1) = w x / A - p x := by
        field_simp [h, hA.ne'] <;> ring
      rw [eqv, mul_sub] at ht'
      linarith only [ht']
  calc
    _ ≤ ∑ x, (p x * Real.log A + w x / A - p x) := Finset.sum_le_sum fun x _ => one x
    _ = (∑ x, p x) * Real.log A + (∑ x, w x) / A - (∑ x, p x) := by
      simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_mul,
        div_eq_mul_inv]
    _ ≤ Real.log A := by
      rw [hn, one_mul]
      have hb : (∑ x, w x) / A ≤ 1 := (div_le_iff₀ hA).mpr (by simpa using hmass)
      linarith only [hb]

section IndependentStrategies

variable {I W : Type*} [Fintype I] [DecidableEq I] [Fintype W]
variable {D : I → Type*} [∀ i, Fintype (D i)]

/-- For a word x, localHit is the probability that local decoder i answers
    correctly. The local decoder object d can be an entire lookup table. -/
def localHit (law : ∀ i, D i → ℝ) (hit : ∀ i, D i → W → ℝ) (i : I) (x : W) : ℝ :=
  ∑ d, law i d * hit i d x

theorem independentStrategyMass (law : ∀ i, D i → ℝ) (hit : ∀ i, D i → W → ℝ)
    (hl : ∀ i d, 0 ≤ law i d) (hn : ∀ i, ∑ d, law i d = 1)
    (A : ℝ) (detBound : ∀ f : ∀ i, D i, (∑ x, ∏ i, hit i (f i) x) ≤ A) :
    (∑ x, ∏ i, localHit law hit i x) ≤ A := by
  classical
  calc
    _ = ∑ x, ∑ f : ∀ i, D i, ∏ i, law i (f i) * hit i (f i) x := by
      apply Finset.sum_congr rfl
      intro x _
      exact Fintype.prod_sum (fun i d => law i d * hit i d x)
    _ = ∑ f : ∀ i, D i, ∑ x, ∏ i, law i (f i) * hit i (f i) x := Finset.sum_comm
    _ = ∑ f : ∀ i, D i, (∏ i, law i (f i)) * ∑ x, ∏ i, hit i (f i) x := by
      simp only [Finset.mul_sum, Finset.prod_mul_distrib]
    _ ≤ ∑ f : ∀ i, D i, (∏ i, law i (f i)) * A :=
      Finset.sum_le_sum fun f _ => mul_le_mul_of_nonneg_left (detBound f)
        (Finset.prod_nonneg fun i _ => hl i (f i))
    _ = A := by
      rw [← Finset.sum_mul, ← Fintype.prod_sum]
      simp [hn]

/-- The left side is H(W) minus the total local conditional cross-entropies.
    Sampling the table entries from their respective posterior conditionals
    makes these cross-entropies exactly the conditional entropies. -/
theorem confusionEntropyConverse (law : ∀ i, D i → ℝ) (hit : ∀ i, D i → W → ℝ)
    (hl : ∀ i d, 0 ≤ law i d) (hn : ∀ i, ∑ d, law i d = 1)
    (hh : ∀ i d x, 0 ≤ hit i d x) (A : ℝ) (hA : 0 < A)
    (detBound : ∀ f : ∀ i, D i, (∑ x, ∏ i, hit i (f i) x) ≤ A)
    (p : W → ℝ) (hp : ∀ x, 0 ≤ p x) (pn : ∑ x, p x = 1)
    (positiveHit : ∀ x, 0 < p x → ∀ i, 0 < localHit law hit i x) :
    (∑ x, p x * ((∑ i, Real.log (localHit law hit i x)) - Real.log (p x))) ≤ Real.log A := by
  classical
  have hmass := independentStrategyMass law hit hl hn A detBound
  have h0 (i : I) (x : W) : 0 ≤ localHit law hit i x :=
    Finset.sum_nonneg fun d _ => mul_nonneg (hl i d) (hh i d x)
  have harg := logMassBound p (fun x => ∏ i, localHit law hit i x) A hA hp pn
    (fun x => Finset.prod_nonneg fun i _ => h0 i x)
    (fun x hx => Finset.prod_pos fun i _ => positiveHit x hx i) hmass
  refine le_trans (le_of_eq ?_) harg
  apply Finset.sum_congr rfl
  intro x _
  by_cases e : p x = 0
  · simp [e]
  · have xpos : 0 < p x := lt_of_le_of_ne (hp x) (Ne.symm e)
    have lp := positiveHit x xpos
    rw [Real.log_div (Finset.prod_ne_zero_iff.mpr (fun i _ => (lp i).ne')) e,
      Real.log_prod _ _ (fun i _ => (lp i).ne')]

end IndependentStrategies
end Phase2Guess

#print axioms Phase2Guess.independentTableMarginal
#print axioms Phase2Guess.logMassBound
#print axioms Phase2Guess.independentStrategyMass
#print axioms Phase2Guess.confusionEntropyConverse
