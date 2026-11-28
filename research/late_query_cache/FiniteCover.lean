import Std

/-! Counting-measure version of the small-ball union bound used in the converse.
    The finite sample and the list of reconstruction centers can contain repeats. -/

namespace LateQueryCover

def countBy {α : Type} (p : α → Bool) : List α → Nat
  | [] => 0
  | x :: xs => (if p x then 1 else 0) + countBy p xs

def covered {α β : Type} (near : α → β → Bool) : List β → α → Bool
  | [], _ => false
  | c :: cs, x => near x c || covered near cs x

theorem countFalse {α : Type} (xs : List α) : countBy (fun _ => false) xs = 0 := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [countBy, ih]

theorem countOrLe {α : Type} (p q : α → Bool) (xs : List α) :
    countBy (fun x => p x || q x) xs ≤ countBy p xs + countBy q xs := by
  induction xs with
  | nil => simp [countBy]
  | cons x xs ih =>
    cases hp : p x <;> cases hq : q x <;> simp [countBy, hp, hq] <;> omega

/-- If a single center can cover at most K sample entries, then N cache
    reconstructions can cover at most N*K entries. -/
theorem smallBallBound {α β : Type} (near : α → β → Bool) (xs : List α)
    (K : Nat) (cs : List β)
    (single : ∀ c ∈ cs, countBy (fun x => near x c) xs ≤ K) :
    countBy (covered near cs) xs ≤ cs.length * K := by
  induction cs with
  | nil =>
    change countBy (fun _ => false) xs ≤ 0 * K
    simp [countFalse]
  | cons c cs ih =>
    have head := single c (by simp)
    have tail := ih (fun d hd => single d (List.mem_cons_of_mem c hd))
    calc
      countBy (covered near (c :: cs)) xs ≤
          countBy (fun x => near x c) xs + countBy (covered near cs) xs :=
        countOrLe (fun x => near x c) (covered near cs) xs
      _ ≤ K + cs.length * K := Nat.add_le_add head tail
      _ = (c :: cs).length * K := by simp [Nat.succ_mul, Nat.add_comm]

theorem missesWhenTooFew {α β : Type} (near : α → β → Bool) (xs : List α)
    (K : Nat) (cs : List β)
    (single : ∀ c ∈ cs, countBy (fun x => near x c) xs ≤ K)
    (few : cs.length * K < xs.length) :
    countBy (covered near cs) xs < xs.length :=
  Nat.lt_of_le_of_lt (smallBallBound near xs K cs single) few

end LateQueryCover

#print axioms LateQueryCover.smallBallBound
#print axioms LateQueryCover.missesWhenTooFew

namespace LateQueryCover

def totalLoss {α : Type} (loss : α → Nat) : List α → Nat
  | [] => 0
  | x :: xs => loss x + totalLoss loss xs

/-- Each point outside a Boolean cover incurs at least its charge. -/
theorem chargeMisses {α : Type} (hit : α → Bool) (loss : α → Nat)
    (charge : Nat) (xs : List α) :
    (∀ x ∈ xs, hit x = false → charge ≤ loss x) →
    xs.length * charge ≤ countBy hit xs * charge + totalLoss loss xs := by
  induction xs with
  | nil => simp [countBy, totalLoss]
  | cons x xs ih =>
    intro hp
    have ht := ih (fun y hy => hp y (List.mem_cons_of_mem x hy))
    cases hx : hit x with
    | false =>
      have hl := hp x (by simp) hx
      simp [countBy, totalLoss, hx, Nat.succ_mul, Nat.add_mul] <;> omega
    | true =>
      simp [countBy, totalLoss, hx, Nat.succ_mul, Nat.add_mul] <;> omega

/-- If a ball contains at most K data points, the uncovered part of the sample
    pays the lesser of the wrong-action gap and the matched-action fitting cost.
    The integer-valued loss can count units of a common rational cost scale. -/
theorem riskBudgetConverse {α β : Type} (near : α → β → Bool)
    (matched : α → Bool) (loss : α → Nat) (xs : List α) (cs : List β)
    (K gap fitFloor : Nat)
    (single : ∀ c ∈ cs, countBy (fun x => near x c) xs ≤ K)
    (wrong : ∀ x ∈ xs, matched x = false → gap ≤ loss x)
    (poorFit : ∀ x ∈ xs, matched x = true → covered near cs x = false →
      fitFloor ≤ loss x) :
    (xs.length - cs.length * K) * min gap fitFloor ≤ totalLoss loss xs := by
  have missed : ∀ x ∈ xs, covered near cs x = false → min gap fitFloor ≤ loss x := by
    intro x hx uncovered
    cases hm : matched x with
    | false => exact Nat.le_trans (Nat.min_le_left _ _) (wrong x hx hm)
    | true => exact Nat.le_trans (Nat.min_le_right _ _) (poorFit x hx hm uncovered)
  have costs := chargeMisses (covered near cs) loss (min gap fitFloor) xs missed
  have bound := Nat.mul_le_mul_right (min gap fitFloor) (smallBallBound near xs K cs single)
  rw [Nat.sub_mul]
  omega

end LateQueryCover

#print axioms LateQueryCover.riskBudgetConverse
