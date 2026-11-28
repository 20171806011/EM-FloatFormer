import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-! The conditional posterior on finite coarse words at fixed broadcast and
public residual is arbitrary. All entropies in this file use natural logarithms.
The measure-theoretic disintegration and the Gaussian approximation are external
arguments, not Lean axioms. -/

namespace Phase2Confusion

open scoped BigOperators

/-- Confusion is witnessed by one receiver whose visible labels coincide. -/
def Confusable {I Word : Type*} (sees : I → I → Prop) (u v : I → Word) : Prop :=
  ∃ i, u i ≠ v i ∧ ∀ j, sees i j → u j = v j

def Decoder {I Word : Type*} (sees : I → I → Prop) :=
  (i : I) → ({j : I // sees i j} → Word) → Word

def Correct {I Word : Type*} {sees : I → I → Prop} (dec : @Decoder I Word sees)
    (u : I → Word) : Prop :=
  ∀ i, dec i (fun j => u j.val) = u i

/-- A fixed broadcast and fixed residual specify a decoder for each receiver.
    The entire simultaneous success set is an independent set. -/
theorem successIndependent {I Word : Type*} {sees : I → I → Prop}
    (dec : @Decoder I Word sees) {u v : I → Word}
    (hu : Correct dec u) (hv : Correct dec v) : ¬Confusable sees u v := by
  rintro ⟨i, different, sameSide⟩
  have hviz : (fun j : {j : I // sees i j} => u j.val) =
      (fun j : {j : I // sees i j} => v j.val) := by
    funext j
    exact sameSide j.val j.property
  exact different ((hu i).symm.trans ((congrArg (dec i) hviz).trans (hv i)))

/-- One term of the relative-entropy bound, including a zero-mass atom. -/
private theorem logWeight (p w : ℝ) (hp : 0 ≤ p) (hw : 0 < w) :
    -p * Real.log p ≤ -p * Real.log w + (w-p) := by
  by_cases hz : p = 0
  · subst p
    simp [hw.le]
  · have hpos : 0 < p := lt_of_le_of_ne hp (Ne.symm hz)
    have logRatio := Real.log_le_sub_one_of_pos (div_pos hw hpos)
    rw [Real.log_div hw.ne' hz] at logRatio
    have scaled := mul_le_mul_of_nonneg_left logRatio hp
    have cancel : p * (w / p - 1) = w-p := by
      field_simp [hz]
    rw [cancel] at scaled
    linarith

/-- Use subprobability weight 1/(2A) on each atom in this part of a partition. -/
private theorem entropyPart {W : Type*} (s : Finset W) (p : W → ℝ)
    (A : ℝ) (pos : 0 < A) (cap : (s.card : ℝ) ≤ A) (hp : ∀ w, 0 ≤ p w) :
    (∑ w ∈ s, -p w * Real.log (p w)) ≤
      (∑ w ∈ s, p w) * Real.log (2*A) + (1/2 - ∑ w ∈ s, p w) := by
  have atoms (w : W) :
      -p w * Real.log (p w) ≤ p w * Real.log (2*A) + ((2*A)⁻¹ - p w) := by
    have h := logWeight (p w) ((2*A)⁻¹) (hp w) (by positivity)
    rw [Real.log_inv] at h
    linarith
  have upper : (s.card : ℝ) * (2*A)⁻¹ ≤ 1/2 := by
    calc
      (s.card : ℝ) * (2*A)⁻¹ ≤ A * (2*A)⁻¹ :=
        mul_le_mul_of_nonneg_right cap (by positivity)
      _ = 1/2 := by field_simp [pos.ne']
  calc
    _ ≤ ∑ w ∈ s, (p w * Real.log (2*A) + ((2*A)⁻¹ - p w)) :=
      Finset.sum_le_sum fun w _ => atoms w
    _ = (∑ w ∈ s, p w) * Real.log (2*A) +
        ((s.card : ℝ) * (2*A)⁻¹ - ∑ w ∈ s, p w) := by
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_mul,
        Finset.sum_const, nsmul_eq_mul]
    _ ≤ _ := by linarith

/-- Entropy converse for an arbitrary conditional posterior, split over its
    successful and failed words. The uses are G = success, B = its complement,
    A = independence number, N = size of the whole word space. No uniformity of
    the conditional masses p is assumed. The right side is
    log 2 + (1-e) log A + e log N where e is the posterior error mass. -/
theorem graphPosteriorEntropy {I Word : Type*} {sees : I → I → Prop}
    (dec : @Decoder I Word sees) (G B : Finset (I → Word)) (p : (I → Word) → ℝ)
    (A N : ℝ) (posA : 0 < A) (posN : 0 < N)
    (hp : ∀ w, 0 ≤ p w)
    (mass : (∑ w ∈ G, p w) + (∑ w ∈ B, p w) = 1)
    (succeeds : ∀ w ∈ G, Correct dec w)
    (alpha : ∀ S : Finset (I → Word),
      (∀ u ∈ S, ∀ v ∈ S, ¬Confusable sees u v) → (S.card : ℝ) ≤ A)
    (space : (B.card : ℝ) ≤ N) :
    (∑ w ∈ G, -p w * Real.log (p w)) + (∑ w ∈ B, -p w * Real.log (p w)) ≤
      Real.log 2 + (∑ w ∈ G, p w) * Real.log A + (∑ w ∈ B, p w) * Real.log N := by
  have ga := alpha G (fun u hu v hv => successIndependent dec (succeeds u hu) (succeeds v hv))
  have g := entropyPart G p A posA ga hp
  have b := entropyPart B p N posN space hp
  rw [Real.log_mul (by norm_num) posA.ne'] at g
  rw [Real.log_mul (by norm_num) posN.ne'] at b
  have split : ((∑ w ∈ G, p w) + (∑ w ∈ B, p w)) * Real.log 2 = Real.log 2 := by
    rw [mass, one_mul]
  nlinarith only [g, b, mass, split]

end Phase2Confusion

#print axioms Phase2Confusion.successIndependent
#print axioms Phase2Confusion.graphPosteriorEntropy
