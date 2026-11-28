import Std
import Init.GrindInstances.Ring.Rat

/-! Finite rational laws and the central action-gap inequality, using Lean 4.24's
    standard distribution. The geometric quantization theorem is proved separately. -/

namespace ActiveCore

def total {ι : Type} (f : ι → Rat) : List ι → Rat
  | [] => 0
  | i :: xs => f i + total f xs

theorem squaredRiskExpansion {ι : Type} (xs : List ι) (p y : ι → Rat) (m t : Rat) :
    total (fun i => p i * (y i - t)^2) xs =
      total (fun i => p i * (y i - m)^2) xs +
      2 * (m - t) * total (fun i => p i * (y i - m)) xs + (m - t)^2 * total p xs := by
  induction xs with
  | nil => simp only [total]; grind
  | cons i xs ih => simp only [total]; rw [ih]; grind

theorem centeredSquaredRisk {ι : Type} (xs : List ι) (p y : ι → Rat) (m t : Rat)
    (mass : total p xs = 1) (center : total (fun i => p i * (y i - m)) xs = 0) :
    total (fun i => p i * (y i - t)^2) xs =
      total (fun i => p i * (y i - m)^2) xs + (m - t)^2 := by
  have h := squaredRiskExpansion xs p y m t
  rw [center, mass] at h
  simpa only [Rat.mul_zero, Rat.mul_one, Rat.add_zero] using h

private theorem addR (x y : Rat) (h : 0 ≤ y) : x ≤ x + y := by
  simpa only [Rat.add_zero] using (Rat.add_le_add_left (c := x)).mpr h

private theorem addL (x y : Rat) (h : 0 ≤ y) : x ≤ y + x := by
  simpa only [Rat.zero_add] using (Rat.add_le_add_right (c := x)).mpr h

/-- If a matched action controls continuation distortion by fitting error and
    each mismatched action incurs a fixed gap, both cases obey one bound. -/
theorem actionGapConverse (matched : Prop) (dist fit regret cap gap p0 : Rat)
    (hFit : 0 ≤ fit) (hRegret : 0 ≤ regret) (hCap : 0 ≤ cap)
    (hGap : 0 ≤ gap) (hP : 0 ≤ p0)
    (same : matched → p0 * dist ≤ fit)
    (wrongDist : ¬matched → dist ≤ cap) (wrongGap : ¬matched → gap ≤ regret) :
    gap * p0 * dist ≤ gap * fit + p0 * cap * regret := by
  by_cases h : matched
  · rw [Rat.mul_assoc gap p0 dist]
    exact Rat.le_trans (Rat.mul_le_mul_of_nonneg_left (same h) hGap)
      (addR _ _ (Rat.mul_nonneg (Rat.mul_nonneg hP hCap) hRegret))
  · have h1 := Rat.mul_le_mul_of_nonneg_left (wrongDist h) (Rat.mul_nonneg hGap hP)
    have h2 := Rat.mul_le_mul_of_nonneg_left (wrongGap h) (Rat.mul_nonneg hP hCap)
    have reorder : (gap * p0) * cap = (p0 * cap) * gap := by
      rw [Rat.mul_assoc gap p0 cap, Rat.mul_comm gap (p0 * cap)]
    rw [reorder] at h1
    exact Rat.le_trans (Rat.le_trans h1 h2) (addL _ _ (Rat.mul_nonneg hGap hFit))

end ActiveCore

#print axioms ActiveCore.centeredSquaredRisk
#print axioms ActiveCore.actionGapConverse
