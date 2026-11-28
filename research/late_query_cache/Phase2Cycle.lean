import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Powerset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith

/-! A zero-error or approximate decoding increment enters only as a lower bound.
The vertices may have different increments. The certificate does not require
rotational symmetry of a source, message, proof witness or set function. -/

namespace LateQueryPhase2

open scoped BigOperators

def mask (n : Nat) : Finset (Fin 5) :=
  Finset.univ.filter (fun i => n.testBit i.val = true)

def SupermodularSet {V : Type*} [DecidableEq V] (u : Finset V → ℝ) : Prop :=
  ∀ A B, u A + u B ≤ u (A ∩ B) + u (A ∪ B)

private theorem maskSquare (u : Finset (Fin 5) → ℝ) (hs : SupermodularSet u)
    (a b i j : Nat) (hi : mask a ∩ mask b = mask i) (hj : mask a ∪ mask b = mask j) :
    u (mask a) + u (mask b) ≤ u (mask i) + u (mask j) := by
  simpa only [hi, hj] using hs (mask a) (mask b)

/-- Twice the five-cycle broadcast cost dominates the sum of its five
    decoding increments. This is an unnormalized inequality. -/
theorem cycle5SupermodularConverse (u : Finset (Fin 5) → ℝ)
    (hm : Monotone u) (hs : SupermodularSet u) :
    (u (mask 19) - u (mask 18)) + (u (mask 7) - u (mask 5)) +
    (u (mask 14) - u (mask 10)) + (u (mask 28) - u (mask 20)) +
    (u (mask 25) - u (mask 9)) ≤ 2 * (u (mask 31) - u (mask 0)) := by
  have m0 := hm (show mask 0 ⊆ mask 1 by decide)
  have m1 := hm (show mask 0 ⊆ mask 2 by decide)
  have m2 := hm (show mask 0 ⊆ mask 4 by decide)
  have m3 := hm (show mask 0 ⊆ mask 8 by decide)
  have m4 := hm (show mask 0 ⊆ mask 16 by decide)
  have s0 := maskSquare u hs 1 4 0 5 (by decide) (by decide)
  have s1 := maskSquare u hs 19 22 18 23 (by decide) (by decide)
  have s2 := maskSquare u hs 5 12 4 13 (by decide) (by decide)
  have s3 := maskSquare u hs 23 30 22 31 (by decide) (by decide)
  have s4 := maskSquare u hs 2 8 0 10 (by decide) (by decide)
  have s5 := maskSquare u hs 7 13 5 15 (by decide) (by decide)
  have s6 := maskSquare u hs 11 25 9 27 (by decide) (by decide)
  have s7 := maskSquare u hs 14 28 12 30 (by decide) (by decide)
  have s8 := maskSquare u hs 4 16 0 20 (by decide) (by decide)
  have s9 := maskSquare u hs 15 27 11 31 (by decide) (by decide)
  linarith only [m0, m1, m2, m3, m4, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

/-- For increments κ−δᵢ this gives u(V)−u(∅) ≥ (5κ−∑δᵢ)/2. -/
theorem cycle5WithUnequalDeficits (u : Finset (Fin 5) → ℝ)
    (hm : Monotone u) (hs : SupermodularSet u) (k e0 e1 e2 e3 e4 : ℝ)
    (h0 : k-e0 ≤ u (mask 19)-u (mask 18)) (h1 : k-e1 ≤ u (mask 7)-u (mask 5))
    (h2 : k-e2 ≤ u (mask 14)-u (mask 10)) (h3 : k-e3 ≤ u (mask 28)-u (mask 20))
    (h4 : k-e4 ≤ u (mask 25)-u (mask 9)) :
    (5*k-(e0+e1+e2+e3+e4))/2 ≤ u (mask 31)-u (mask 0) := by
  linarith only [cycle5SupermodularConverse u hm hs, h0, h1, h2, h3, h4]

section ArbitraryGraph

variable {V : Type*} [Fintype V] [DecidableEq V]

inductive GraphRow (V : Type*) where
  | mono (A B : Finset V)
  | supermod (A B : Finset V)
  | decoder (v : V)

def atom (A T : Finset V) : ℝ := if T = A then 1 else 0

private theorem sumAtom (A : Finset V) (u : Finset V → ℝ) :
    ∑ T : Finset V, atom A T * u T = u A := by
  classical
  simp [atom, ite_mul]

/-- Row coefficients in the coordinate basis indexed by all vertex subsets. -/
def rowCoeff (nb : V → Finset V) : GraphRow V → Finset V → ℝ
  | .mono A B, T => atom (A ∪ B) T - atom A T
  | .supermod A B, T => atom (A ∩ B) T + atom (A ∪ B) T - atom A T - atom B T
  | .decoder v, T => atom (insert v (nb v)) T - atom (nb v) T

def rowReward (k : V → ℝ) : GraphRow V → ℝ
  | .decoder v => k v
  | _ => 0

theorem graphRowSound (nb : V → Finset V) (k : V → ℝ) (u : Finset V → ℝ)
    (hm : Monotone u) (hs : SupermodularSet u)
    (hd : ∀ v, k v ≤ u (insert v (nb v)) - u (nb v)) (row : GraphRow V) :
    rowReward k row ≤ ∑ T : Finset V, rowCoeff nb row T * u T := by
  cases row with
  | mono A B =>
    simp only [rowReward, rowCoeff, sub_mul, Finset.sum_sub_distrib, sumAtom]
    exact sub_nonneg.mpr (hm (Finset.subset_union_left))
  | supermod A B =>
    simp only [rowReward, rowCoeff, add_mul, sub_mul, Finset.sum_sub_distrib,
      Finset.sum_add_distrib, sumAtom]
    linarith only [hs A B]
  | decoder v =>
    simpa only [rowReward, rowCoeff, sub_mul, Finset.sum_sub_distrib, sumAtom] using hd v

/-- A nonnegative linear combination of graph rows is a converse whenever the
    coefficient at every subset agrees with the desired endpoint. Directed
    neighborhoods and arbitrary nonnegative real (hence fractional) multipliers
    are allowed. This is soundness, not completeness of the Shannon LP. -/
theorem fractionalGraphDual {I : Type*} [Fintype I]
    (nb : V → Finset V) (k : V → ℝ) (u : Finset V → ℝ)
    (hm : Monotone u) (hs : SupermodularSet u)
    (hd : ∀ v, k v ≤ u (insert v (nb v)) - u (nb v))
    (rows : I → GraphRow V) (w : I → ℝ) (hw : ∀ i, 0 ≤ w i) (c : ℝ)
    (balance : ∀ T, (∑ i, w i * rowCoeff nb (rows i) T) =
      c * (atom Finset.univ T - atom ∅ T)) :
    (∑ i, w i * rowReward k (rows i)) ≤ c * (u Finset.univ - u ∅) := by
  classical
  calc
    (∑ i, w i * rowReward k (rows i)) ≤
        ∑ i, w i * (∑ T, rowCoeff nb (rows i) T * u T) :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left
        (graphRowSound nb k u hm hs hd (rows i)) (hw i)
    _ = ∑ i, ∑ T, w i * rowCoeff nb (rows i) T * u T := by
      simp only [Finset.mul_sum, mul_assoc]
    _ = ∑ T, ∑ i, w i * rowCoeff nb (rows i) T * u T := Finset.sum_comm
    _ = ∑ T, (c * (atom Finset.univ T - atom ∅ T)) * u T := by
      apply Finset.sum_congr rfl
      intro T _
      rw [← Finset.sum_mul, balance T]
    _ = c * (u Finset.univ - u ∅) := by
      simp only [mul_assoc, sub_mul, Finset.sum_sub_distrib, ← Finset.mul_sum, sumAtom]

end ArbitraryGraph

end LateQueryPhase2

#print axioms LateQueryPhase2.cycle5SupermodularConverse
#print axioms LateQueryPhase2.cycle5WithUnequalDeficits
#print axioms LateQueryPhase2.graphRowSound
#print axioms LateQueryPhase2.fractionalGraphDual
