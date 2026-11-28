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
