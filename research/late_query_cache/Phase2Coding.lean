import Std

/-! Bit-level C5 index code and an explicit modulo-three guard for noisy side information.
The runtime argument uses blocks of arbitrary even length, not a truth table at a fixed width. -/

namespace LateQueryPhase2.Coding

abbrev Word (w : Nat) := Fin w → Bool
abbrev Vertex := Fin 5

def prev (i : Vertex) : Vertex := ⟨(i.val + 4) % 5, Nat.mod_lt _ (by decide)⟩
def next (i : Vertex) : Vertex := ⟨(i.val + 1) % 5, Nat.mod_lt _ (by decide)⟩

theorem nextPrev (i : Vertex) : next (prev i) = i := by
  apply Fin.ext
  change ((i.val + 4) % 5 + 1) % 5 = i.val
  have hi := i.isLt
  omega

def leftHalf {w : Nat} (x : Word (2*w)) : Word w :=
  fun j => x ⟨j.val, by have h := j.isLt; omega⟩

def rightHalf {w : Nat} (x : Word (2*w)) : Word w :=
  fun j => x ⟨w + j.val, by have h := j.isLt; omega⟩

def join {w : Nat} (l r : Word w) : Word (2*w) :=
  fun j => if h : j.val < w then l ⟨j.val, h⟩
    else r ⟨j.val - w, by have h' := j.isLt; omega⟩

theorem joinSplit {w : Nat} (x : Word (2*w)) : join (leftHalf x) (rightHalf x) = x := by
  funext j
  by_cases h : j.val < w
  · simp only [join, dif_pos h, leftHalf]
  · simp only [join, dif_neg h, rightHalf]
    apply congrArg x
    apply Fin.ext
    change w + (j.val - w) = j.val
    omega

def xorW {w : Nat} (x y : Word w) : Word w := fun j => Bool.xor (x j) (y j)

theorem xorCancelLeft {w : Nat} (x y : Word w) : xorW x (xorW x y) = y := by
  funext j
  change Bool.xor (x j) (Bool.xor (x j) (y j)) = y j
  cases x j <;> cases y j <;> decide

theorem xorCancelRight {w : Nat} (x y : Word w) : xorW (xorW x y) y = x := by
  funext j
  change Bool.xor (Bool.xor (x j) (y j)) (y j) = x j
  cases x j <;> cases y j <;> decide

/-- Exactly five broadcast words of width w for five source words of width 2w. -/
def encodeC5 {w : Nat} (x : Vertex → Word (2*w)) : Vertex → Word w :=
  fun i => xorW (rightHalf (x i)) (leftHalf (x (next i)))

/-- The receiver has access to only its two neighbor words and the public packets. -/
def decodeC5 {w : Nat} (i : Vertex) (p : Vertex → Word w)
    (sidePrev sideNext : Word (2*w)) : Word (2*w) :=
  join (xorW (rightHalf sidePrev) (p (prev i)))
       (xorW (p i) (leftHalf sideNext))

theorem cycleDecodeExact {w : Nat} (x : Vertex → Word (2*w)) (i : Vertex) :
    decodeC5 i (encodeC5 x) (x (prev i)) (x (next i)) = x i := by
  simp only [decodeC5, encodeC5, nextPrev, xorCancelLeft, xorCancelRight, joinSplit]

/-- Correct a preliminary integer coarse-bin index using a residue sent by the encoder. -/
def recoverBin (z residue : Int) : Int :=
  if z % 3 = residue then z else if (z-1) % 3 = residue then z-1 else z+1

theorem recoverBinNear (k z : Int) (lo : k-1 ≤ z) (hi : z ≤ k+1) :
    recoverBin z (k % 3) = k := by
  unfold recoverBin
  split
  · omega
  · split <;> omega

/-- The same decoder works on noisy preliminary bins after two independent guards.
The label map is arbitrary, so this covers any agreed binary labeling of the quantizer. -/
theorem guardedCycleDecodeExact {w : Nat} (k : Vertex → Int) (label : Int → Word (2*w))
    (i : Vertex) (zPrev zNext : Int)
    (pl : k (prev i)-1 ≤ zPrev) (pu : zPrev ≤ k (prev i)+1)
    (nl : k (next i)-1 ≤ zNext) (nu : zNext ≤ k (next i)+1) :
    decodeC5 i (encodeC5 (fun j => label (k j)))
      (label (recoverBin zPrev (k (prev i) % 3)))
      (label (recoverBin zNext (k (next i) % 3))) = label (k i) := by
  rw [recoverBinNear _ _ pl pu, recoverBinNear _ _ nl nu]
  exact cycleDecodeExact (fun j => label (k j)) i

end LateQueryPhase2.Coding

#print axioms LateQueryPhase2.Coding.cycleDecodeExact
#print axioms LateQueryPhase2.Coding.recoverBinNear
#print axioms LateQueryPhase2.Coding.guardedCycleDecodeExact
