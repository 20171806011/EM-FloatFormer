import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Algebra.Order.Ring.Pow
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pigeonhole
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr

/-!
# MemoryPhase: Conjecture 4 Load-Bearing Formalization

Machine-checked formalization of the four core theorem families for
**Conjecture 4 (Sequential Re-Compression Forgetting Phase Transition & Causal-State Critical Dimension)**
under `MathLab.Projects.LateQueryFrontier.MemoryPhase`:

1. **Family 1 (`|λ| > 1` Sub-critical Recursion, Kernel Gap & Data-Rate Pigeonhole)**:
   exact error recursion, unforced exponential divergence, rank-nullity kernel gap for `d_mem < r`,
   projection gap lower bound, and finite-alphabet `K`-level data-rate pigeonhole blow-up.
2. **Family 2 (`|λ| = 1` Jordan-Block Polynomial Divergence & Non-Normal Transient)**:
   exact `k × k` unit Jordan block iteration formula, `Ω(t^{k-1})` polynomial lower bound,
   `k = 2` forced sum-of-squares identity `Θ(t^3)`, and `2 × 2` non-normal transient amplification.
3. **Family 3 (`|λ| < 1` Geometric Stability & Super-Critical Composite Bound)**:
   exact geometric contraction bound and uniform `(t, τ)` composite bound with `r` retained
   `E^{u/c}` modes and `d_s` contractive `E^s` modes.
4. **Family 4 (Multi-Step SDPI / Dobrushin TV Contraction)**:
   Dobrushin coefficient bounds (`0 ≤ η_TV ≤ 1`), Doeblin minorization bound,
   single-step Dobrushin TV contraction, and heterogeneous/homogeneous multi-step SDPI contraction.

## Honest Scope Boundary
- **Lean 4 Kernel-Verified Here**: All deterministic algebraic recursions, rank-nullity kernel gaps,
  finite-alphabet pigeonhole bounds, exact Jordan-block binomial expansions and polynomial bounds,
  geometric stability bounds, non-normal `2 × 2` transient formula, and discrete Dobrushin/Doeblin
  TV contraction inequalities.
- **Analytic Only in `THEORY.md`**: Real Jordan canonical form decomposition on `ℝ^d`, continuous
  topological Invariance of Domain, Gaussian differential-entropy / EPI / AM-GM data-rate bounds,
  hockey-stick integral proof of `η_KL(K) ≤ η_TV(K)`, Kreiss matrix theorem, Luenberger observer
  for coupled `C_s ≠ 0`, and Crutchfield ε-machine causal-state collapse.
-/

namespace MathLab.Projects.LateQueryFrontier.MemoryPhase

open Finset

/-! ## Family 1: Sub-critical `|λ| > 1` Recursion, Kernel Gap & Data-Rate Pigeonhole -/

/-- Exact closed-form solution for the scalar linear error recursion
`e (t + 1) = lam * e t + w t`. -/
theorem linear_error_closed_form (lam : ℝ) (e w : ℕ → ℝ)
    (hrec : ∀ t : ℕ, e (t + 1) = lam * e t + w t) (t : ℕ) :
    e t = lam ^ t * e 0 + ∑ k ∈ Finset.range t, lam ^ (t - 1 - k) * w k := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [hrec t, ih, Finset.sum_range_succ]
    have hcongr :
        ∑ k ∈ Finset.range t, lam ^ (t + 1 - 1 - k) * w k =
          lam * ∑ k ∈ Finset.range t, lam ^ (t - 1 - k) * w k := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro k hk
      rw [Finset.mem_range] at hk
      have hsub : t + 1 - 1 - k = (t - 1 - k) + 1 := by omega
      rw [hsub, pow_succ']
      ring
    rw [hcongr]
    have hlast : t + 1 - 1 - t = 0 := by omega
    rw [hlast, pow_zero, one_mul, pow_succ']
    ring

/-- Unboundedness / exponential divergence of an unforced scalar recursion `e (t + 1) = lam * e t`
with `1 < |lam|` and `e 0 ≠ 0`. -/
theorem unforced_exp_divergence (lam : ℝ) (hlam : 1 < |lam|) (e : ℕ → ℝ)
    (hrec : ∀ t : ℕ, e (t + 1) = lam * e t) (he0 : e 0 ≠ 0) :
    ∀ B : ℝ, ∃ T : ℕ, ∀ t : ℕ, T ≤ t → B ≤ |e t| := by
  have he_form : ∀ t : ℕ, e t = lam ^ t * e 0 := by
    intro t
    induction t with
    | zero => simp
    | succ t ih => rw [hrec t, ih, pow_succ']; ring
  have hbern : ∀ t : ℕ, 1 + (t : ℝ) * (|lam| - 1) ≤ |lam| ^ t := by
    intro t
    have h0 : 0 ≤ |lam| - 1 := by linarith
    induction t with
    | zero => simp
    | succ t iht =>
      have h1 : 1 ≤ |lam| := by linarith
      calc
        1 + ((t + 1 : ℕ) : ℝ) * (|lam| - 1)
            = (1 + (t : ℝ) * (|lam| - 1)) + (|lam| - 1) := by push_cast; ring
        _ ≤ (1 + (t : ℝ) * (|lam| - 1)) * |lam| := by
            have hnn : 0 ≤ 1 + (t : ℝ) * (|lam| - 1) := by positivity
            have ht0 : (0 : ℝ) ≤ (t : ℝ) := Nat.cast_nonneg t
            nlinarith [mul_nonneg ht0 (mul_nonneg h0 h0)]
        _ ≤ |lam| ^ t * |lam| := mul_le_mul_of_nonneg_right iht (abs_nonneg lam)
        _ = |lam| ^ (t + 1) := (pow_succ |lam| t).symm
  have hpos_e0 : 0 < |e 0| := abs_pos.mpr he0
  have hpos_delta : 0 < |lam| - 1 := by linarith
  have hpos_den : 0 < (|lam| - 1) * |e 0| := mul_pos hpos_delta hpos_e0
  intro B
  obtain ⟨T, hT⟩ := exists_nat_gt (B / ((|lam| - 1) * |e 0|))
  refine ⟨T, ?_⟩
  intro t ht
  rw [he_form t, abs_mul, abs_pow]
  have ht_real : (T : ℝ) ≤ (t : ℝ) := Nat.cast_le.mpr ht
  have hB_lt : B < (t : ℝ) * ((|lam| - 1) * |e 0|) := by
    have h1 : B / ((|lam| - 1) * |e 0|) < (t : ℝ) := lt_of_lt_of_le hT ht_real
    exact (div_lt_iff₀ hpos_den).mp h1
  have hbern_t : 1 + (t : ℝ) * (|lam| - 1) ≤ |lam| ^ t := hbern t
  have hmul : (1 + (t : ℝ) * (|lam| - 1)) * |e 0| ≤ |lam| ^ t * |e 0| :=
    mul_le_mul_of_nonneg_right hbern_t (abs_nonneg (e 0))
  nlinarith

/-- Elementary rank-nullity kernel existence: any `d × r` real linear map with `d < r`
has a nontrivial kernel vector `v : Fin r → ℝ`. -/
theorem exists_nonzero_ker_of_lt :
    ∀ {d r : ℕ}, d < r → ∀ (A : Fin d → Fin r → ℝ),
      ∃ v : Fin r → ℝ, (∃ j : Fin r, v j ≠ 0) ∧ ∀ i : Fin d, ∑ j : Fin r, A i j * v j = 0 := by
  intro d
  induction d with
  | zero =>
    intro r hdr A
    refine ⟨fun _ => 1, ⟨⟨0, hdr⟩, one_ne_zero⟩, ?_⟩
    intro i
    exact Fin.elim0 i
  | succ d ih =>
    intro r hdr A
    by_cases hrow0 : ∀ j : Fin r, A 0 j = 0
    · have hdr' : d < r := by omega
      obtain ⟨v, hv_nz, hv_ker⟩ := ih hdr' (fun i j => A i.succ j)
      refine ⟨v, hv_nz, ?_⟩
      intro i
      refine Fin.cases ?_ ?_ i
      · simp [hrow0]
      · exact hv_ker
    · push Not at hrow0
      obtain ⟨j₀, hj₀⟩ := hrow0
      obtain ⟨r', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : r ≠ 0)
      have hdr' : d < r' := by omega
      let A' : Fin d → Fin r' → ℝ := fun i k =>
        A i.succ (j₀.succAbove k) - (A i.succ j₀ / A 0 j₀) * A 0 (j₀.succAbove k)
      obtain ⟨u, ⟨k₀, hk₀⟩, hu_ker⟩ := ih hdr' A'
      let c : ℝ := -(∑ k : Fin r', A 0 (j₀.succAbove k) * u k) / A 0 j₀
      let v : Fin (r' + 1) → ℝ := j₀.insertNth c u
      refine ⟨v, ⟨j₀.succAbove k₀, ?_⟩, ?_⟩
      · simpa [v] using hk₀
      · intro i
        refine Fin.cases ?_ ?_ i
        · rw [Fin.sum_univ_succAbove _ j₀]
          have hv0 : v j₀ = c := by simp [v]
          have hvs : ∀ k : Fin r', v (j₀.succAbove k) = u k := by simp [v]
          simp only [hv0, hvs, c]
          field_simp [hj₀]
          ring
        · intro i'
          rw [Fin.sum_univ_succAbove _ j₀]
          have hv0 : v j₀ = c := by simp [v]
          have hvs : ∀ k : Fin r', v (j₀.succAbove k) = u k := by simp [v]
          simp only [hv0, hvs, c]
          have hzero := hu_ker i'
          have hrewrite :
              A i'.succ j₀ * (-(∑ k : Fin r', A 0 (j₀.succAbove k) * u k) / A 0 j₀) +
                ∑ k : Fin r', A i'.succ (j₀.succAbove k) * u k =
              ∑ k : Fin r', A' i' k * u k := by
            simp only [A', sub_mul, Finset.sum_sub_distrib, mul_assoc]
            have hpull :
                ∑ k : Fin r', (A i'.succ j₀ / A 0 j₀) * (A 0 (j₀.succAbove k) * u k) =
                  (A i'.succ j₀ / A 0 j₀) * ∑ k : Fin r', A 0 (j₀.succAbove k) * u k := by
              rw [Finset.mul_sum]
            rw [hpull]
            ring
          rw [hrewrite, hzero]

/-- Sub-critical linear projection gap: whenever `d_mem < r`, any linear encoder
`P : Fin d_mem → Fin r → ℝ` and arbitrary decoder `dec` suffer worst-case error at horizon `τ`
at least `(1 / 2) * |lam j₀| ^ τ * |v j₀|` over `s₀ ∈ {0, v}` for a fixed nonzero kernel vector `v`. -/
theorem subcritical_linear_projection_gap (d_mem r : ℕ) (hdr : d_mem < r)
    (lam : Fin r → ℝ) (P : Fin d_mem → Fin r → ℝ)
    (dec : (Fin d_mem → ℝ) → (Fin r → ℝ)) (τ : ℕ) :
    ∃ v : Fin r → ℝ, ∃ j₀ : Fin r, v j₀ ≠ 0 ∧ (∀ i : Fin d_mem, ∑ j : Fin r, P i j * v j = 0) ∧
      ∃ s₀ : Fin r → ℝ, (s₀ = 0 ∨ s₀ = v) ∧
        (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| ≤
          |dec (fun i => ∑ j : Fin r, P i j * s₀ j) j₀ - lam j₀ ^ τ * s₀ j₀| := by
  obtain ⟨v, ⟨j₀, hj₀⟩, hv_ker⟩ := exists_nonzero_ker_of_lt hdr P
  refine ⟨v, j₀, hj₀, hv_ker, ?_⟩
  let enc : (Fin r → ℝ) → (Fin d_mem → ℝ) := fun s i => ∑ j : Fin r, P i j * s j
  have henc0 : enc 0 = 0 := by
    ext i
    simp [enc]
  have hencv : enc v = 0 := by
    ext i
    exact hv_ker i
  let y : ℝ := dec 0 j₀
  have htri : |lam j₀ ^ τ * v j₀| ≤ |y - 0| + |y - lam j₀ ^ τ * v j₀| := by
    calc
      |lam j₀ ^ τ * v j₀| = |(y - 0) - (y - lam j₀ ^ τ * v j₀)| := by ring_nf
      _ ≤ |y - 0| + |y - lam j₀ ^ τ * v j₀| := abs_sub _ _
  have hmul_abs : |lam j₀ ^ τ * v j₀| = |lam j₀| ^ τ * |v j₀| := by
    rw [abs_mul, abs_pow]
  rw [hmul_abs] at htri
  by_cases hcase : (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| ≤ |y - 0|
  · refine ⟨0, Or.inl rfl, ?_⟩
    have hsub0 : (fun i : Fin d_mem => ∑ j : Fin r, P i j * (0 : Fin r → ℝ) j) = 0 := henc0
    rw [hsub0]
    simpa [y] using hcase
  · refine ⟨v, Or.inr rfl, ?_⟩
    have hother : (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| ≤ |y - lam j₀ ^ τ * v j₀| := by linarith
    have hsubv : (fun i : Fin d_mem => ∑ j : Fin r, P i j * v j) = 0 := hencv
    rw [hsubv]
    simpa [y] using hother

/-- Unbounded asymptotic blow-up of the sub-critical linear projection gap when all retained
target modes are unstable (`1 < |lam j|`). -/
theorem subcritical_linear_projection_blowup (d_mem r : ℕ) (hdr : d_mem < r)
    (lam : Fin r → ℝ) (hlam : ∀ j : Fin r, 1 < |lam j|)
    (P : Fin d_mem → Fin r → ℝ) (dec : (Fin d_mem → ℝ) → (Fin r → ℝ)) :
    ∃ v : Fin r → ℝ, ∃ j₀ : Fin r, v j₀ ≠ 0 ∧ (∀ i : Fin d_mem, ∑ j : Fin r, P i j * v j = 0) ∧
      ∀ B : ℝ, ∃ τ₀ : ℕ, ∀ τ : ℕ, τ₀ ≤ τ →
        ∃ s₀ : Fin r → ℝ, (s₀ = 0 ∨ s₀ = v) ∧
          B ≤ |dec (fun i => ∑ j : Fin r, P i j * s₀ j) j₀ - lam j₀ ^ τ * s₀ j₀| := by
  obtain ⟨v, ⟨j₀, hj₀⟩, hv_ker⟩ := exists_nonzero_ker_of_lt hdr P
  refine ⟨v, j₀, hj₀, hv_ker, ?_⟩
  let e : ℕ → ℝ := fun τ => (1 / 2 : ℝ) * lam j₀ ^ τ * v j₀
  have hrec : ∀ τ : ℕ, e (τ + 1) = lam j₀ * e τ := by
    intro τ
    simp only [e, pow_succ']
    ring
  have he0 : e 0 ≠ 0 := by
    simp only [e, pow_zero, mul_one, ne_eq, mul_eq_zero, one_div, inv_eq_zero, OfNat.ofNat_ne_zero,
      false_or]
    exact hj₀
  have hdiv := unforced_exp_divergence (lam j₀) (hlam j₀) e hrec he0
  intro B
  obtain ⟨τ₀, hτ₀⟩ := hdiv B
  refine ⟨τ₀, ?_⟩
  intro τ hτ
  have hB : B ≤ (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| := by
    have h1 := hτ₀ τ hτ
    have h2 : |e τ| = (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| := by
      simp only [e, abs_mul, abs_pow]
      norm_num
    linarith
  let enc : (Fin r → ℝ) → (Fin d_mem → ℝ) := fun s i => ∑ j : Fin r, P i j * s j
  have henc0 : enc 0 = 0 := by ext i; simp [enc]
  have hencv : enc v = 0 := by ext i; exact hv_ker i
  let y : ℝ := dec 0 j₀
  have htri : |lam j₀| ^ τ * |v j₀| ≤ |y - 0| + |y - lam j₀ ^ τ * v j₀| := by
    have hmul_abs : |lam j₀ ^ τ * v j₀| = |lam j₀| ^ τ * |v j₀| := by rw [abs_mul, abs_pow]
    rw [← hmul_abs]
    calc
      |lam j₀ ^ τ * v j₀| = |(y - 0) - (y - lam j₀ ^ τ * v j₀)| := by ring_nf
      _ ≤ |y - 0| + |y - lam j₀ ^ τ * v j₀| := abs_sub _ _
  by_cases hcase : (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| ≤ |y - 0|
  · refine ⟨0, Or.inl rfl, ?_⟩
    have h0 : B ≤ |y - 0| := le_trans hB hcase
    have hsub0 : (fun i : Fin d_mem => ∑ j : Fin r, P i j * (0 : Fin r → ℝ) j) = 0 := henc0
    rw [hsub0]
    simpa [y] using h0
  · refine ⟨v, Or.inr rfl, ?_⟩
    have hother : (1 / 2 : ℝ) * |lam j₀| ^ τ * |v j₀| ≤ |y - lam j₀ ^ τ * v j₀| := by linarith
    have hv' : B ≤ |y - lam j₀ ^ τ * v j₀| := le_trans hB hother
    have hsubv : (fun i : Fin d_mem => ∑ j : Fin r, P i j * v j) = 0 := hencv
    rw [hsubv]
    simpa [y] using hv'

/-- Finite-alphabet `K`-level data-rate pigeonhole lower bound:
for `K ^ t + 1` grid points on `[0, Δ]` mapped through a `K ^ t`-state summary,
at least one initial point incurs error `≥ (Δ / 2) * (|lam| / K) ^ t`. -/
theorem data_rate_pigeonhole_lower_bound (K t : ℕ) (hK : 1 ≤ K) (lam Δ c_t : ℝ) (hΔ : 0 < Δ)
    (enc : Fin (K ^ t + 1) → Fin (K ^ t)) (dec : Fin (K ^ t) → ℝ) :
    ∃ idx : Fin (K ^ t + 1),
      (Δ / 2) * (|lam| / (K : ℝ)) ^ t ≤
        |dec (enc idx) - (lam ^ t * ((idx.1 : ℝ) * (Δ / (K ^ t : ℝ))) + c_t)| := by
  have hcard : Fintype.card (Fin (K ^ t)) < Fintype.card (Fin (K ^ t + 1)) := by
    simp
  obtain ⟨i, j, hij, henc⟩ := Fintype.exists_ne_map_eq_of_card_lt enc hcard
  have hpos_Kt : (0 : ℝ) < (K : ℝ) ^ t := by
    have hKpos : (0 : ℝ) < (K : ℝ) := by exact_mod_cast (by omega : 0 < K)
    exact pow_pos hKpos t
  have hij_nat : i.1 ≠ j.1 := fun h => hij (Fin.ext h)
  have hdiff_ge_one : (1 : ℝ) ≤ |(i.1 : ℝ) - (j.1 : ℝ)| := by
    rcases lt_or_gt_of_ne hij_nat with hlt | hgt
    · have h1 : i.1 + 1 ≤ j.1 := hlt
      have h2 : (i.1 : ℝ) + 1 ≤ (j.1 : ℝ) := by exact_mod_cast h1
      rw [abs_of_neg (by linarith)]
      linarith
    · have h1 : j.1 + 1 ≤ i.1 := hgt
      have h2 : (j.1 : ℝ) + 1 ≤ (i.1 : ℝ) := by exact_mod_cast h1
      rw [abs_of_nonneg (by linarith)]
      linarith
  let x : Fin (K ^ t + 1) → ℝ := fun u => (u.1 : ℝ) * (Δ / (K ^ t : ℝ))
  let s : Fin (K ^ t + 1) → ℝ := fun u => lam ^ t * x u + c_t
  have hs_diff : Δ * (|lam| / (K : ℝ)) ^ t ≤ |s i - s j| := by
    have hx_diff : Δ / (K : ℝ) ^ t ≤ |x i - x j| := by
      have hxf : x i - x j = ((i.1 : ℝ) - (j.1 : ℝ)) * (Δ / (K : ℝ) ^ t) := by
        simp only [x]
        ring
      rw [hxf, abs_mul, abs_of_pos (div_pos hΔ hpos_Kt)]
      calc
        Δ / (K : ℝ) ^ t = 1 * (Δ / (K : ℝ) ^ t) := by ring
        _ ≤ |(i.1 : ℝ) - (j.1 : ℝ)| * (Δ / (K : ℝ) ^ t) :=
          mul_le_mul_of_nonneg_right hdiff_ge_one (le_of_lt (div_pos hΔ hpos_Kt))
    have hsf : |s i - s j| = |lam| ^ t * |x i - x j| := by
      have h1 : s i - s j = lam ^ t * (x i - x j) := by simp only [s]; ring
      rw [h1, abs_mul, abs_pow]
    rw [hsf]
    calc
      Δ * (|lam| / (K : ℝ)) ^ t = |lam| ^ t * (Δ / (K : ℝ) ^ t) := by
        rw [div_pow]; ring
      _ ≤ |lam| ^ t * |x i - x j| :=
        mul_le_mul_of_nonneg_left hx_diff (pow_nonneg (abs_nonneg lam) t)
  have htri : |s i - s j| ≤ |dec (enc i) - s i| + |dec (enc j) - s j| := by
    rw [← henc]
    calc
      |s i - s j| = |(dec (enc i) - s j) - (dec (enc i) - s i)| := by ring_nf
      _ ≤ |dec (enc i) - s j| + |dec (enc i) - s i| := abs_sub _ _
      _ = |dec (enc i) - s i| + |dec (enc i) - s j| := add_comm _ _
  have hsum_ge : Δ * (|lam| / (K : ℝ)) ^ t ≤ |dec (enc i) - s i| + |dec (enc j) - s j| :=
    le_trans hs_diff htri
  by_cases hi : (Δ / 2) * (|lam| / (K : ℝ)) ^ t ≤ |dec (enc i) - s i|
  · exact ⟨i, hi⟩
  · refine ⟨j, ?_⟩
    linarith

/-- Unbounded sub-critical blow-up when the bit-rate alphabet size satisfies `K < |lam|`. -/
theorem data_rate_subcritical_blowup (K : ℕ) (hK : 1 ≤ K) (lam Δ : ℝ) (hΔ : 0 < Δ)
    (hsub : (K : ℝ) < |lam|) :
    ∀ B : ℝ, ∃ T : ℕ, ∀ t : ℕ, T ≤ t →
      ∀ (c_t : ℝ) (enc : Fin (K ^ t + 1) → Fin (K ^ t)) (dec : Fin (K ^ t) → ℝ),
        ∃ idx : Fin (K ^ t + 1),
          B ≤ |dec (enc idx) - (lam ^ t * ((idx.1 : ℝ) * (Δ / (K ^ t : ℝ))) + c_t)| := by
  have hKpos : (0 : ℝ) < (K : ℝ) := by exact_mod_cast (by omega : 0 < K)
  have hratio : 1 < |lam| / (K : ℝ) := (one_lt_div hKpos).mpr hsub
  let e : ℕ → ℝ := fun t => (Δ / 2) * (|lam| / (K : ℝ)) ^ t
  have hrec : ∀ t : ℕ, e (t + 1) = (|lam| / (K : ℝ)) * e t := by
    intro t
    simp only [e, pow_succ']
    ring
  have he0 : e 0 ≠ 0 := by
    simp only [e, pow_zero, mul_one]
    linarith
  have hlt1 : 1 < |(|lam| / (K : ℝ))| := by
    rw [abs_of_pos (by linarith)]
    exact hratio
  have hdiv := unforced_exp_divergence (|lam| / (K : ℝ)) hlt1 e hrec he0
  intro B
  obtain ⟨T, hT⟩ := hdiv B
  refine ⟨T, ?_⟩
  intro t ht c_t enc dec
  obtain ⟨idx, hidx⟩ := data_rate_pigeonhole_lower_bound K t hK lam Δ c_t hΔ enc dec
  refine ⟨idx, ?_⟩
  have hB : B ≤ (Δ / 2) * (|lam| / (K : ℝ)) ^ t := by
    have h1 := hT t ht
    have he_pos : 0 ≤ e t := by
      simp only [e]
      positivity
    rwa [abs_of_nonneg he_pos] at h1
  exact le_trans hB hidx

/-! ## Family 2: `|λ| = 1` Jordan-Block Polynomial Divergence & Non-Normal Transient -/

/-- Action of the `k × k` unit upper-triangular Jordan block `J_k(1)` on `v : Fin k → ℝ`. -/
def jordanStep (k : ℕ) (v : Fin k → ℝ) (i : Fin k) : ℝ :=
  if h : i.1 + 1 < k then v i + v ⟨i.1 + 1, h⟩ else v i

/-- Exact binomial expansion for the `t`-th iterate of `jordanStep k`. -/
theorem jordanStep_iterate_formula (k t : ℕ) (v : Fin k → ℝ) (i : Fin k) :
    (jordanStep k)^[t] v i =
      ∑ j ∈ Finset.range (k - i.1), ((Nat.choose t j : ℕ) : ℝ) *
        (if h : i.1 + j < k then v ⟨i.1 + j, h⟩ else 0) := by
  induction t generalizing i with
  | zero =>
    have hk_pos : 1 ≤ k - i.1 := by omega
    obtain ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k - i.1 ≠ 0)
    rw [Function.iterate_zero_apply]
    have hrange : Finset.range (k - i.1) = Finset.range (m + 1) := by rw [hm]
    rw [Finset.sum_eq_single 0]
    · simp [i.isLt]
    · intro b _hb hb0
      have hch : Nat.choose 0 b = 0 := Nat.choose_eq_zero_of_lt (Nat.pos_of_ne_zero hb0)
      simp [hch]
    · intro h0
      exfalso
      apply h0
      rw [Finset.mem_range]
      omega
  | succ t ih =>
    rw [Function.iterate_succ_apply']
    by_cases h1 : i.1 + 1 < k
    · have hstep : jordanStep k ((jordanStep k)^[t] v) i =
          (jordanStep k)^[t] v i + (jordanStep k)^[t] v ⟨i.1 + 1, h1⟩ := by
        simp [jordanStep, h1]
      rw [hstep, ih i, ih ⟨i.1 + 1, h1⟩]
      have hlen : k - i.1 = (k - (i.1 + 1)) + 1 := by omega
      rw [hlen, Finset.sum_range_succ', Finset.sum_range_succ']
      have hleft :
          ∑ x ∈ Finset.range (k - (i.1 + 1)),
              ((Nat.choose t (x + 1) : ℕ) : ℝ) *
                (if h : i.1 + (x + 1) < k then v ⟨i.1 + (x + 1), h⟩ else 0) +
            ((Nat.choose t 0 : ℕ) : ℝ) * (if h : i.1 + 0 < k then v ⟨i.1 + 0, h⟩ else 0) +
            ∑ x ∈ Finset.range (k - (i.1 + 1)),
              ((Nat.choose t x : ℕ) : ℝ) *
                (if h : i.1 + 1 + x < k then v ⟨i.1 + 1 + x, h⟩ else 0) =
          ∑ x ∈ Finset.range (k - (i.1 + 1)),
              ((Nat.choose (t + 1) (x + 1) : ℕ) : ℝ) *
                (if h : i.1 + (x + 1) < k then v ⟨i.1 + (x + 1), h⟩ else 0) +
            ((Nat.choose (t + 1) 0 : ℕ) : ℝ) * (if h : i.1 + 0 < k then v ⟨i.1 + 0, h⟩ else 0) := by
        have hsum_combine :
            ∑ x ∈ Finset.range (k - (i.1 + 1)),
                ((Nat.choose t (x + 1) : ℕ) : ℝ) *
                  (if h : i.1 + (x + 1) < k then v ⟨i.1 + (x + 1), h⟩ else 0) +
              ∑ x ∈ Finset.range (k - (i.1 + 1)),
                ((Nat.choose t x : ℕ) : ℝ) *
                  (if h : i.1 + 1 + x < k then v ⟨i.1 + 1 + x, h⟩ else 0) =
            ∑ x ∈ Finset.range (k - (i.1 + 1)),
                ((Nat.choose (t + 1) (x + 1) : ℕ) : ℝ) *
                  (if h : i.1 + (x + 1) < k then v ⟨i.1 + (x + 1), h⟩ else 0) := by
          rw [← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl ?_
          intro x _
          have heq : i.1 + 1 + x = i.1 + (x + 1) := by omega
          have hchoose : Nat.choose (t + 1) (x + 1) = Nat.choose t x + Nat.choose t (x + 1) :=
            Nat.choose_succ_succ t x
          simp only [heq, hchoose, Nat.cast_add]
          ring
        rw [add_right_comm, hsum_combine, Nat.choose_zero_right, Nat.choose_zero_right]
      exact hleft
    · have hstep : jordanStep k ((jordanStep k)^[t] v) i = (jordanStep k)^[t] v i := by
        simp [jordanStep, h1]
      rw [hstep, ih i]
      have hlen : k - i.1 = 1 := by omega
      rw [hlen]
      simp [Nat.choose_zero_right]

/-- Polynomial lower bound `((t + 1 - k)^(k - 1) / (k - 1)!) * |v_{k-1}| ≤ |(J_k(1)^t v)_0|`
for a perturbation supported on the top coordinate of a size-`k` Jordan block (`k ≥ 2`). -/
theorem jordan_top_mode_polynomial_lower_bound (k t : ℕ) (hk : 2 ≤ k) (ht : k - 1 ≤ t)
    (v : Fin k → ℝ) (hv_low : ∀ j : Fin k, j.1 < k - 1 → v j = 0) :
    ((t + 1 - k : ℝ) ^ (k - 1) / (Nat.factorial (k - 1) : ℝ)) * |v ⟨k - 1, by omega⟩| ≤
      |(jordanStep k)^[t] v ⟨0, by omega⟩| := by
  have hform := jordanStep_iterate_formula k t v ⟨0, by omega⟩
  simp only [Nat.sub_zero, zero_add] at hform
  have hsingle :
      (jordanStep k)^[t] v ⟨0, by omega⟩ =
        ((Nat.choose t (k - 1) : ℕ) : ℝ) * v ⟨k - 1, by omega⟩ := by
    rw [hform]
    rw [Finset.sum_eq_single (k - 1)]
    · have hlt : k - 1 < k := by omega
      simp [hlt]
    · intro j hj hne
      rw [Finset.mem_range] at hj
      have hjlt : j < k - 1 := by omega
      have hjk : j < k := by omega
      simp [hjk, hv_low ⟨j, hjk⟩ hjlt]
    · intro hmem
      exfalso
      apply hmem
      rw [Finset.mem_range]
      omega
  rw [hsingle, abs_mul]
  have hchoose_nonneg : (0 : ℝ) ≤ (Nat.choose t (k - 1) : ℝ) := by positivity
  rw [abs_of_nonneg hchoose_nonneg]
  have hdesc_nat : ∀ m : ℕ, m ≤ t → (t - m) ^ m ≤ Nat.descFactorial t m := by
    intro m
    induction m with
    | zero => intro _; simp
    | succ m ihm =>
      intro hm
      rw [Nat.descFactorial_succ, pow_succ']
      have hm_le : m ≤ t := by omega
      have hfactor : t - (m + 1) ≤ t - m := by omega
      calc
        (t - (m + 1)) * (t - (m + 1)) ^ m
            ≤ (t - m) * (t - m) ^ m :=
          Nat.mul_le_mul hfactor (Nat.pow_le_pow_left hfactor m)
        _ ≤ (t - m) * Nat.descFactorial t m :=
          Nat.mul_le_mul_left (t - m) (ihm hm_le)
  have hnat_bound : (t - (k - 1)) ^ (k - 1) ≤ Nat.choose t (k - 1) * Nat.factorial (k - 1) := by
    rw [mul_comm, ← Nat.descFactorial_eq_factorial_mul_choose]
    exact hdesc_nat (k - 1) ht
  have hreal_bound :
      ((t - (k - 1) : ℕ) : ℝ) ^ (k - 1) ≤
        (Nat.choose t (k - 1) : ℝ) * (Nat.factorial (k - 1) : ℝ) := by
    exact_mod_cast hnat_bound
  have hcast_eq : ((t - (k - 1) : ℕ) : ℝ) = (t : ℝ) + 1 - (k : ℝ) := by
    rw [Nat.cast_sub ht, Nat.cast_sub (by omega : 1 ≤ k)]
    ring
  rw [hcast_eq] at hreal_bound
  have hfac_pos : (0 : ℝ) < (Nat.factorial (k - 1) : ℝ) := by
    exact_mod_cast Nat.factorial_pos (k - 1)
  have hdiv_bound :
      (t + 1 - k : ℝ) ^ (k - 1) / (Nat.factorial (k - 1) : ℝ) ≤ (Nat.choose t (k - 1) : ℝ) :=
    (div_le_iff₀ hfac_pos).mpr (by simpa using hreal_bound)
  exact mul_le_mul_of_nonneg_right hdiv_bound (abs_nonneg _)

/-- Exact sum-of-squares identity `6 * ∑ m < t, m^2 = t * (t - 1) * (2 * t - 1)`
underlying `Θ(t^3)` stochastic variance growth in a `2 × 2` Jordan block. -/
theorem jordan2_forced_sum_sq (t : ℕ) :
    6 * (∑ m ∈ Finset.range t, (m : ℝ) ^ 2) = (t : ℝ) * ((t : ℝ) - 1) * (2 * (t : ℝ) - 1) := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Finset.sum_range_succ, mul_add, ih]
    push_cast
    ring

/-- Action of a `2 × 2` upper-triangular non-normal block `[ρ, γ; 0, ρ]` on `Fin 2 → ℝ`. -/
def nonnormal2Step (ρ γ : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ
  | 0 => ρ * v 0 + γ * v 1
  | 1 => ρ * v 1

/-- Exact `t`-step transient amplification formula for the `2 × 2` upper-triangular
non-normal block `[ρ, γ; 0, ρ]`, producing the `t * ρ^(t - 1) * γ * v₁` spike. -/
theorem nonnormal_2x2_transient (ρ γ : ℝ) (v : Fin 2 → ℝ) (t : ℕ) (ht : 1 ≤ t) :
    (nonnormal2Step ρ γ)^[t] v 0 = ρ ^ t * v 0 + (t : ℝ) * ρ ^ (t - 1) * γ * v 1 ∧
      (nonnormal2Step ρ γ)^[t] v 1 = ρ ^ t * v 1 := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : t ≠ 0)
  induction m with
  | zero =>
    simp [nonnormal2Step]
  | succ m ih =>
    rw [Function.iterate_succ_apply']
    have h0 := (ih (by omega)).1
    have h1 := (ih (by omega)).2
    constructor
    · simp only [nonnormal2Step, h0, h1]
      have hpow : ρ ^ (m + 1 + 1) = ρ * ρ ^ (m + 1) := pow_succ' ρ (m + 1)
      have hpow2 : ρ ^ (m + 1) = ρ * ρ ^ m := pow_succ' ρ m
      rw [hpow]
      push_cast
      rw [hpow2]
      ring
    · simp only [nonnormal2Step, h1]
      rw [pow_succ', pow_succ']
      ring

/-! ## Family 3: `|λ| < 1` Geometric Stability & Super-Critical Composite Bound -/

/-- Geometric stability of a forced scalar contractive recursion `e (t + 1) = rho * e t + w t`
with `|rho| < 1` and `|w t| ≤ W`: both exact geometric bound and uniform bound hold. -/
theorem contractive_error_bound (rho W : ℝ) (hrho : |rho| < 1) (hW : 0 ≤ W)
    (e w : ℕ → ℝ) (hrec : ∀ t : ℕ, e (t + 1) = rho * e t + w t)
    (hw : ∀ t : ℕ, |w t| ≤ W) (t : ℕ) :
    |e t| ≤ |rho| ^ t * |e 0| + W * ((1 - |rho| ^ t) / (1 - |rho|)) ∧
      |e t| ≤ |e 0| + W / (1 - |rho|) := by
  have h1m : 0 < 1 - |rho| := by linarith
  have hgeom : |e t| ≤ |rho| ^ t * |e 0| + W * ((1 - |rho| ^ t) / (1 - |rho|)) := by
    induction t with
    | zero => simp
    | succ t ih =>
      rw [hrec t]
      have htri : |rho * e t + w t| ≤ |rho| * |e t| + |w t| := by
        calc
          |rho * e t + w t| ≤ |rho * e t| + |w t| := abs_add_le _ _
          _ = |rho| * |e t| + |w t| := by rw [abs_mul]
      have hstep :
          |rho| * |e t| + |w t| ≤
            |rho| * (|rho| ^ t * |e 0| + W * ((1 - |rho| ^ t) / (1 - |rho|))) + W :=
        add_le_add (mul_le_mul_of_nonneg_left ih (abs_nonneg rho)) (hw t)
      have halg :
          |rho| * (|rho| ^ t * |e 0| + W * ((1 - |rho| ^ t) / (1 - |rho|))) + W =
            |rho| ^ (t + 1) * |e 0| + W * ((1 - |rho| ^ (t + 1)) / (1 - |rho|)) := by
        rw [pow_succ']
        field_simp [ne_of_gt h1m]
        ring
      exact le_trans htri (le_trans hstep (le_of_eq halg))
  refine ⟨hgeom, ?_⟩
  have hpow_le : |rho| ^ t ≤ 1 := pow_le_one₀ (abs_nonneg rho) (le_of_lt hrho)
  have hpow_nonneg : 0 ≤ |rho| ^ t := pow_nonneg (abs_nonneg rho) t
  have hfirst : |rho| ^ t * |e 0| ≤ |e 0| := by
    calc
      |rho| ^ t * |e 0| ≤ 1 * |e 0| := mul_le_mul_of_nonneg_right hpow_le (abs_nonneg (e 0))
      _ = |e 0| := one_mul _
  have hsecond : W * ((1 - |rho| ^ t) / (1 - |rho|)) ≤ W / (1 - |rho|) := by
    calc
      W * ((1 - |rho| ^ t) / (1 - |rho|))
          ≤ W * (1 / (1 - |rho|)) := by
            refine mul_le_mul_of_nonneg_left ?_ hW
            exact div_le_div_of_nonneg_right (by linarith) (le_of_lt h1m)
      _ = W / (1 - |rho|) := by ring
  linarith

/-- Super-critical composite uniform `(t, τ)` bound: `r` exactly retained `E^{u/c}` modes
(`e_uc = 0`) plus `d_s` contractive `E^s` modes (`|rho j| ≤ rho_bar < 1`) satisfy a uniform
squared-error bound for all `(t, τ)` with exponential decay `rho_bar ^ (2 * τ)` in `τ`. -/
theorem supercritical_composite_uniform_bound (r d_s : ℕ)
    (lam_uc : Fin r → ℝ) (e_uc : ℕ → Fin r → ℝ) (he_uc : ∀ t : ℕ, ∀ i : Fin r, e_uc t i = 0)
    (rho : Fin d_s → ℝ) (rho_bar E0 W : ℝ)
    (hrho_bar_lt : rho_bar < 1) (hrho : ∀ j : Fin d_s, |rho j| ≤ rho_bar)
    (hE0 : 0 ≤ E0) (hW : 0 ≤ W)
    (e_s w_s : ℕ → Fin d_s → ℝ)
    (he0 : ∀ j : Fin d_s, |e_s 0 j| ≤ E0)
    (hw : ∀ t : ℕ, ∀ j : Fin d_s, |w_s t j| ≤ W)
    (hrec : ∀ t : ℕ, ∀ j : Fin d_s, e_s (t + 1) j = rho j * e_s t j + w_s t j)
    (t τ : ℕ) :
    (∑ i : Fin r, (lam_uc i ^ τ * e_uc t i) ^ 2 + ∑ j : Fin d_s, (rho j ^ τ * e_s t j) ^ 2) ≤
      (d_s : ℝ) * rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 ∧
      (d_s : ℝ) * rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 ≤
        (d_s : ℝ) * (E0 + W / (1 - rho_bar)) ^ 2 := by
  have huc_zero : ∑ i : Fin r, (lam_uc i ^ τ * e_uc t i) ^ 2 = 0 := by
    simp [he_uc]
  rw [huc_zero, zero_add]
  have h1m_bar : 0 < 1 - rho_bar := by linarith
  have hB_nonneg : 0 ≤ E0 + W / (1 - rho_bar) := by positivity
  have hterm :
      ∀ j : Fin d_s, (rho j ^ τ * e_s t j) ^ 2 ≤ rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 := by
    intro j
    have hrho_j : |rho j| < 1 := lt_of_le_of_lt (hrho j) hrho_bar_lt
    have h1m_j : 0 < 1 - |rho j| := by linarith
    have hle_j : 1 - rho_bar ≤ 1 - |rho j| := by linarith [hrho j]
    have hbound_j := (contractive_error_bound (rho j) W hrho_j hW
      (fun k => e_s k j) (fun k => w_s k j) (fun k => hrec k j) (fun k => hw k j) t).2
    have he_j : |e_s t j| ≤ E0 + W / (1 - rho_bar) := by
      have hdiv_le : W / (1 - |rho j|) ≤ W / (1 - rho_bar) :=
        div_le_div_of_nonneg_left hW h1m_bar hle_j
      linarith [he0 j]
    have hrho_pow : |rho j| ^ τ ≤ rho_bar ^ τ :=
      pow_le_pow_left₀ (abs_nonneg (rho j)) (hrho j) τ
    have hr_nonneg : 0 ≤ rho_bar ^ τ := by
      have hbar_nonneg : 0 ≤ rho_bar := le_trans (abs_nonneg (rho j)) (hrho j)
      exact pow_nonneg hbar_nonneg τ
    have habs_prod : |rho j ^ τ * e_s t j| ≤ rho_bar ^ τ * (E0 + W / (1 - rho_bar)) := by
      rw [abs_mul, abs_pow]
      exact mul_le_mul hrho_pow he_j (abs_nonneg _) hr_nonneg
    have hsq : (rho j ^ τ * e_s t j) ^ 2 ≤ (rho_bar ^ τ * (E0 + W / (1 - rho_bar))) ^ 2 := by
      rw [← sq_abs (rho j ^ τ * e_s t j)]
      exact sq_le_sq' (by linarith [abs_nonneg (rho j ^ τ * e_s t j)]) habs_prod
    have hpow2 : (rho_bar ^ τ * (E0 + W / (1 - rho_bar))) ^ 2 =
        rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 := by
      rw [mul_pow, ← pow_mul, mul_comm τ 2]
    linarith
  have hsum_le :
      ∑ j : Fin d_s, (rho j ^ τ * e_s t j) ^ 2 ≤
        (d_s : ℝ) * rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 := by
    calc
      ∑ j : Fin d_s, (rho j ^ τ * e_s t j) ^ 2
          ≤ ∑ _j : Fin d_s, rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 :=
        Finset.sum_le_sum (fun j _ => hterm j)
      _ = (d_s : ℝ) * (rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2) := by
        simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ = (d_s : ℝ) * rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2 := by ring
  refine ⟨hsum_le, ?_⟩
  by_cases hds : d_s = 0
  · subst hds; simp
  · have hj0 : Fin d_s := ⟨0, Nat.pos_of_ne_zero hds⟩
    have hbar_nonneg : 0 ≤ rho_bar := le_trans (abs_nonneg (rho hj0)) (hrho hj0)
    have hpow_le_one : rho_bar ^ (2 * τ) ≤ 1 := pow_le_one₀ hbar_nonneg (le_of_lt hrho_bar_lt)
    have hds_nonneg : (0 : ℝ) ≤ (d_s : ℝ) := by positivity
    calc
      (d_s : ℝ) * rho_bar ^ (2 * τ) * (E0 + W / (1 - rho_bar)) ^ 2
          ≤ (d_s : ℝ) * 1 * (E0 + W / (1 - rho_bar)) ^ 2 := by
            gcongr
      _ = (d_s : ℝ) * (E0 + W / (1 - rho_bar)) ^ 2 := by ring

/-! ## Family 4: Multi-Step SDPI Contraction Inequality (Dobrushin TV Contraction) -/

/-- A discrete finite-state Markov kernel `K : Fin m → Fin n → ℝ` with nonnegative
entries and unit row sums. -/
structure MarkovKernel (m n : ℕ) where
  K : Fin m → Fin n → ℝ
  nonneg : ∀ i : Fin m, ∀ j : Fin n, 0 ≤ K i j
  row_sum : ∀ i : Fin m, ∑ j : Fin n, K i j = 1

/-- Total variation distance `(1 / 2) * ∑ i, |p i - q i|` between finite vectors. -/
noncomputable def tvDist {m : ℕ} (p q : Fin m → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ i : Fin m, |p i - q i|

/-- Pushforward of a distribution/signed vector `p` through a Markov kernel `M`. -/
def push {m n : ℕ} (p : Fin m → ℝ) (M : MarkovKernel m n) (j : Fin n) : ℝ :=
  ∑ i : Fin m, p i * M.K i j

/-- Dobrushin contraction coefficient `η_TV(M) = max_{i, i'} (1 / 2) * ∑_j |K(i, j) - K(i', j)|`. -/
noncomputable def dobrushinCoeff {m n : ℕ} (hm : 0 < m) (M : MarkovKernel m n) : ℝ :=
  Finset.univ.sup'
    (by
      let i0 : Fin m := ⟨0, hm⟩
      exact ⟨(i0, i0), Finset.mem_univ _⟩)
    (fun pair : Fin m × Fin m => (1 / 2 : ℝ) * ∑ j : Fin n, |M.K pair.1 j - M.K pair.2 j|)

/-- Nonnegativity of `dobrushinCoeff`. -/
theorem dobrushinCoeff_nonneg {m n : ℕ} (hm : 0 < m) (M : MarkovKernel m n) :
    0 ≤ dobrushinCoeff hm M := by
  let i0 : Fin m := ⟨0, hm⟩
  have hle :
      (1 / 2 : ℝ) * ∑ j : Fin n, |M.K i0 j - M.K i0 j| ≤ dobrushinCoeff hm M :=
    Finset.le_sup'
      (fun pair : Fin m × Fin m => (1 / 2 : ℝ) * ∑ j : Fin n, |M.K pair.1 j - M.K pair.2 j|)
      (Finset.mem_univ (i0, i0))
  simpa using hle

/-- Upper bound `dobrushinCoeff hm M ≤ 1`. -/
theorem dobrushinCoeff_le_one {m n : ℕ} (hm : 0 < m) (M : MarkovKernel m n) :
    dobrushinCoeff hm M ≤ 1 := by
  refine Finset.sup'_le _ _ ?_
  intro pair _
  have hsum :
      ∑ j : Fin n, |M.K pair.1 j - M.K pair.2 j| ≤
        ∑ j : Fin n, (M.K pair.1 j + M.K pair.2 j) := by
    refine Finset.sum_le_sum ?_
    intro j _
    calc
      |M.K pair.1 j - M.K pair.2 j| ≤ |M.K pair.1 j| + |M.K pair.2 j| := abs_sub _ _
      _ = M.K pair.1 j + M.K pair.2 j := by
        rw [abs_of_nonneg (M.nonneg pair.1 j), abs_of_nonneg (M.nonneg pair.2 j)]
  rw [Finset.sum_add_distrib, M.row_sum pair.1, M.row_sum pair.2] at hsum
  linarith

/-- Doeblin minorization bound: if `∀ i j, α j ≤ M.K i j`, then
`dobrushinCoeff hm M ≤ 1 - ∑ j, α j`. -/
theorem doeblin_contraction_bound {m n : ℕ} (hm : 0 < m) (M : MarkovKernel m n)
    (α : Fin n → ℝ) (hα : ∀ i : Fin m, ∀ j : Fin n, α j ≤ M.K i j) :
    dobrushinCoeff hm M ≤ 1 - ∑ j : Fin n, α j := by
  refine Finset.sup'_le _ _ ?_
  intro pair _
  have hsum :
      ∑ j : Fin n, |M.K pair.1 j - M.K pair.2 j| ≤
        ∑ j : Fin n, ((M.K pair.1 j - α j) + (M.K pair.2 j - α j)) := by
    refine Finset.sum_le_sum ?_
    intro j _
    have h1 : 0 ≤ M.K pair.1 j - α j := sub_nonneg.mpr (hα pair.1 j)
    have h2 : 0 ≤ M.K pair.2 j - α j := sub_nonneg.mpr (hα pair.2 j)
    have hrewrite : M.K pair.1 j - M.K pair.2 j = (M.K pair.1 j - α j) - (M.K pair.2 j - α j) := by
      ring
    rw [hrewrite]
    calc
      |(M.K pair.1 j - α j) - (M.K pair.2 j - α j)|
          ≤ |M.K pair.1 j - α j| + |M.K pair.2 j - α j| := abs_sub _ _
      _ = (M.K pair.1 j - α j) + (M.K pair.2 j - α j) := by
        rw [abs_of_nonneg h1, abs_of_nonneg h2]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    M.row_sum pair.1, M.row_sum pair.2] at hsum
  linarith

/-- Single-step Dobrushin TV contraction inequality: whenever `∑ i, p i = ∑ i, q i`,
`tvDist (push p M) (push q M) ≤ dobrushinCoeff hm M * tvDist p q`. -/
theorem dobrushin_tv_contraction {m n : ℕ} (hm : 0 < m) (M : MarkovKernel m n)
    (p q : Fin m → ℝ) (hsum : ∑ i : Fin m, p i = ∑ i : Fin m, q i) :
    tvDist (push p M) (push q M) ≤ dobrushinCoeff hm M * tvDist p q := by
  let δ : Fin m → ℝ := fun i => p i - q i
  let dp : Fin m → ℝ := fun i => max (δ i) 0
  let dn : Fin m → ℝ := fun i => max (-δ i) 0
  have hdp_nonneg : ∀ i, 0 ≤ dp i := fun i => le_max_right _ _
  have hdn_nonneg : ∀ i, 0 ≤ dn i := fun i => le_max_right _ _
  have hdecomp : ∀ i, δ i = dp i - dn i := by
    intro i
    simp only [dp, dn]
    rcases le_total (δ i) 0 with h | h
    · rw [max_eq_right h, max_eq_left (by linarith)]; ring
    · rw [max_eq_left h, max_eq_right (by linarith)]; ring
  have habs_decomp : ∀ i, |δ i| = dp i + dn i := by
    intro i
    simp only [dp, dn]
    rcases le_total (δ i) 0 with h | h
    · rw [abs_of_nonpos h, max_eq_right h, max_eq_left (by linarith)]; ring
    · rw [abs_of_nonneg h, max_eq_left h, max_eq_right (by linarith)]; ring
  have hsum_δ : ∑ i : Fin m, δ i = 0 := by
    simp only [δ, Finset.sum_sub_distrib, hsum, sub_self]
  have hsum_dp_dn : ∑ i : Fin m, dp i = ∑ i : Fin m, dn i := by
    have h1 : ∑ i : Fin m, (dp i - dn i) = 0 := by
      simpa [← hdecomp] using hsum_δ
    rw [Finset.sum_sub_distrib] at h1
    linarith
  have htv_eq : tvDist p q = ∑ i : Fin m, dp i := by
    simp only [tvDist]
    have h2 : ∑ i : Fin m, |p i - q i| = ∑ i : Fin m, (dp i + dn i) :=
      Finset.sum_congr rfl (fun i _ => habs_decomp i)
    rw [h2, Finset.sum_add_distrib, ← hsum_dp_dn]
    ring
  let S : ℝ := tvDist p q
  have hS_nonneg : 0 ≤ S := by
    change 0 ≤ tvDist p q
    rw [htv_eq]
    exact Finset.sum_nonneg (fun i _ => hdp_nonneg i)
  rcases eq_or_lt_of_le hS_nonneg with hS0 | hSpos
  · have hdp0 : ∀ i : Fin m, dp i = 0 := by
      have hsum0 : ∑ i : Fin m, dp i = 0 := by linarith [htv_eq]
      exact fun i => Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hdp_nonneg j) |>.mp hsum0 i
        (Finset.mem_univ i)
    have hdn0 : ∀ i : Fin m, dn i = 0 := by
      have hsum0 : ∑ i : Fin m, dn i = 0 := by linarith [htv_eq, hsum_dp_dn]
      exact fun i => Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hdn_nonneg j) |>.mp hsum0 i
        (Finset.mem_univ i)
    have hδ0 : ∀ i : Fin m, δ i = 0 := fun i => by rw [hdecomp i, hdp0 i, hdn0 i, sub_zero]
    have hpush0 : ∀ j : Fin n, push p M j - push q M j = 0 := by
      intro j
      simp only [push, ← Finset.sum_sub_distrib]
      refine Finset.sum_eq_zero ?_
      intro i _
      have hsub : p i * M.K i j - q i * M.K i j = δ i * M.K i j := by simp only [δ]; ring
      rw [hsub, hδ0 i, zero_mul]
    have htv_push0 : tvDist (push p M) (push q M) = 0 := by
      simp [tvDist, hpush0]
    have hS_tv : tvDist p q = 0 := hS0.symm
    rw [htv_push0, hS_tv, mul_zero]
  · have hS_dp : S = ∑ i : Fin m, dp i := htv_eq
    have hS_dn : S = ∑ i : Fin m, dn i := by linarith [htv_eq, hsum_dp_dn]
    have hmul_S :
        ∀ j : Fin n, S * (push p M j - push q M j) =
          ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * (M.K i j - M.K i' j) := by
      intro j
      have hdiff : push p M j - push q M j =
          ∑ i : Fin m, dp i * M.K i j - ∑ i' : Fin m, dn i' * M.K i' j := by
        simp only [push, ← Finset.sum_sub_distrib]
        have h1 :
            ∑ i : Fin m, (p i * M.K i j - q i * M.K i j) =
              ∑ i : Fin m, (dp i * M.K i j - dn i * M.K i j) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          have h2 : p i * M.K i j - q i * M.K i j = δ i * M.K i j := by simp only [δ]; ring
          rw [h2, hdecomp i]
          ring
        rw [h1, Finset.sum_sub_distrib]
      rw [hdiff, mul_sub]
      have hL :
          S * ∑ i : Fin m, dp i * M.K i j =
            ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * M.K i j := by
        conv_lhs => rw [hS_dn]
        rw [Finset.sum_mul]
        calc
          ∑ x : Fin m, dn x * ∑ i : Fin m, dp i * M.K i j
              = ∑ x : Fin m, ∑ i : Fin m, dn x * (dp i * M.K i j) := by
                refine Finset.sum_congr rfl (fun x _ => Finset.mul_sum _ _ _)
          _ = ∑ i : Fin m, ∑ i' : Fin m, dn i' * (dp i * M.K i j) := Finset.sum_comm
          _ = ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * M.K i j := by
            refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun i' _ => by ring))
      have hR :
          S * ∑ i' : Fin m, dn i' * M.K i' j =
            ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * M.K i' j := by
        conv_lhs => rw [hS_dp]
        rw [Finset.sum_mul]
        calc
          ∑ i : Fin m, dp i * ∑ i' : Fin m, dn i' * M.K i' j
              = ∑ i : Fin m, ∑ i' : Fin m, dp i * (dn i' * M.K i' j) := by
                refine Finset.sum_congr rfl (fun i _ => Finset.mul_sum _ _ _)
          _ = ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * M.K i' j := by
            refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun i' _ => by ring))
      rw [hL, hR, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl ?_
      intro i' _
      ring
    have habs_j :
        ∀ j : Fin n, S * |push p M j - push q M j| ≤
          ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * |M.K i j - M.K i' j| := by
      intro j
      have h1 : S * |push p M j - push q M j| = |S * (push p M j - push q M j)| := by
        rw [abs_mul, abs_of_pos hSpos]
      rw [h1, hmul_S j]
      calc
        |∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * (M.K i j - M.K i' j)|
            ≤ ∑ i : Fin m, |∑ i' : Fin m, dp i * dn i' * (M.K i j - M.K i' j)| :=
          Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ i : Fin m, ∑ i' : Fin m, |dp i * dn i' * (M.K i j - M.K i' j)| := by
          refine Finset.sum_le_sum (fun i _ => Finset.abs_sum_le_sum_abs _ _)
        _ = ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * |M.K i j - M.K i' j| := by
          refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun i' _ => ?_))
          rw [abs_mul, abs_mul, abs_of_nonneg (hdp_nonneg i), abs_of_nonneg (hdn_nonneg i')]
    have htotal :
        S * tvDist (push p M) (push q M) ≤ S * (dobrushinCoeff hm M * S) := by
      calc
        S * tvDist (push p M) (push q M)
            = (1 / 2 : ℝ) * ∑ j : Fin n, S * |push p M j - push q M j| := by
              simp only [tvDist, Finset.mul_sum]
              refine Finset.sum_congr rfl (fun j _ => by ring)
        _ ≤ (1 / 2 : ℝ) * ∑ j : Fin n, ∑ i : Fin m, ∑ i' : Fin m,
              dp i * dn i' * |M.K i j - M.K i' j| := by
            refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum (fun j _ => habs_j j)) (by norm_num)
        _ = ∑ i : Fin m, ∑ i' : Fin m,
              dp i * dn i' * ((1 / 2 : ℝ) * ∑ j : Fin n, |M.K i j - M.K i' j|) := by
            rw [Finset.sum_comm]
            have hswap :
                (∑ i : Fin m, ∑ j : Fin n, ∑ i' : Fin m, dp i * dn i' * |M.K i j - M.K i' j|) =
                  ∑ i : Fin m, ∑ i' : Fin m, ∑ j : Fin n, dp i * dn i' * |M.K i j - M.K i' j| :=
              Finset.sum_congr rfl (fun i _ => Finset.sum_comm)
            rw [hswap, Finset.mul_sum]
            refine Finset.sum_congr rfl (fun i _ => ?_)
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl (fun i' _ => ?_)
            rw [Finset.mul_sum]
            have h1 : ∀ j : Fin n, (1 / 2 : ℝ) * (dp i * dn i' * |M.K i j - M.K i' j|) =
                (dp i * dn i' * (1 / 2 : ℝ)) * |M.K i j - M.K i' j| := fun j => by ring
            simp_rw [h1, ← Finset.mul_sum]
            ring
        _ ≤ ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * dobrushinCoeff hm M := by
            refine Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun i' _ => ?_))
            have hpair :
                (1 / 2 : ℝ) * ∑ j : Fin n, |M.K i j - M.K i' j| ≤ dobrushinCoeff hm M :=
              Finset.le_sup'
                (fun pair : Fin m × Fin m =>
                  (1 / 2 : ℝ) * ∑ j : Fin n, |M.K pair.1 j - M.K pair.2 j|)
                (Finset.mem_univ (i, i'))
            exact mul_le_mul_of_nonneg_left hpair (mul_nonneg (hdp_nonneg i) (hdn_nonneg i'))
        _ = (∑ i : Fin m, dp i) * (∑ i' : Fin m, dn i') * dobrushinCoeff hm M := by
            have h1 :
                ∑ i : Fin m, ∑ i' : Fin m, dp i * dn i' * dobrushinCoeff hm M =
                  ∑ i : Fin m, dp i * ((∑ i' : Fin m, dn i') * dobrushinCoeff hm M) := by
              refine Finset.sum_congr rfl (fun i _ => ?_)
              rw [Finset.sum_mul, Finset.mul_sum]
              refine Finset.sum_congr rfl (fun i' _ => by ring)
            rw [h1, ← Finset.sum_mul]
            ring
        _ = S * (dobrushinCoeff hm M * S) := by
            rw [← hS_dp, ← hS_dn]; ring
    exact le_of_mul_le_mul_left htotal hSpos

/-- `t`-step pushforward along a time-varying Markov chain `M : ℕ → MarkovKernel n n`. -/
def chainPush {n : ℕ} (p : Fin n → ℝ) (M : ℕ → MarkovKernel n n) : ℕ → Fin n → ℝ
  | 0 => p
  | t + 1 => push (chainPush p M t) (M t)

/-- `push` preserves total mass. -/
theorem push_sum {m n : ℕ} (p : Fin m → ℝ) (M : MarkovKernel m n) :
    ∑ j : Fin n, push p M j = ∑ i : Fin m, p i := by
  unfold push
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [← Finset.mul_sum, M.row_sum i, mul_one]

/-- `chainPush` preserves total mass for every `t`. -/
theorem chainPush_sum {n : ℕ} (p : Fin n → ℝ) (M : ℕ → MarkovKernel n n) (t : ℕ) :
    ∑ j : Fin n, chainPush p M t j = ∑ i : Fin n, p i := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [chainPush]
    rw [push_sum, ih]

/-- Multi-step heterogeneous SDPI TV contraction inequality. -/
theorem multistep_sdpi_tv_contraction {n : ℕ} (hn : 0 < n) (M : ℕ → MarkovKernel n n)
    (p q : Fin n → ℝ) (hsum : ∑ i : Fin n, p i = ∑ i : Fin n, q i) (t : ℕ) :
    tvDist (chainPush p M t) (chainPush q M t) ≤
      (∏ s ∈ Finset.range t, dobrushinCoeff hn (M s)) * tvDist p q := by
  induction t with
  | zero => simp [chainPush]
  | succ t ih =>
    simp only [chainPush]
    have hsum_t : ∑ i : Fin n, chainPush p M t i = ∑ i : Fin n, chainPush q M t i := by
      rw [chainPush_sum p M t, chainPush_sum q M t, hsum]
    have hstep := dobrushin_tv_contraction hn (M t) (chainPush p M t) (chainPush q M t) hsum_t
    have hcomb := le_trans hstep
      (mul_le_mul_of_nonneg_left ih (dobrushinCoeff_nonneg hn (M t)))
    rw [Finset.prod_range_succ]
    calc
      tvDist (push (chainPush p M t) (M t)) (push (chainPush q M t) (M t))
          ≤ dobrushinCoeff hn (M t) *
              ((∏ s ∈ Finset.range t, dobrushinCoeff hn (M s)) * tvDist p q) := hcomb
      _ = (∏ s ∈ Finset.range t, dobrushinCoeff hn (M s)) * dobrushinCoeff hn (M t) *
            tvDist p q := by ring

/-- Multi-step homogeneous exponential SDPI contraction `≤ eta_bar ^ t * tvDist p q`. -/
theorem multistep_sdpi_homogeneous {n : ℕ} (hn : 0 < n) (M : ℕ → MarkovKernel n n)
    (eta_bar : ℝ) (heta : ∀ s : ℕ, dobrushinCoeff hn (M s) ≤ eta_bar)
    (p q : Fin n → ℝ) (hsum : ∑ i : Fin n, p i = ∑ i : Fin n, q i) (t : ℕ) :
    tvDist (chainPush p M t) (chainPush q M t) ≤ eta_bar ^ t * tvDist p q := by
  have hprod :
      ∏ s ∈ Finset.range t, dobrushinCoeff hn (M s) ≤ eta_bar ^ t := by
    induction t with
    | zero => simp
    | succ t iht =>
      rw [Finset.prod_range_succ, pow_succ]
      have heta_nonneg : 0 ≤ eta_bar := le_trans (dobrushinCoeff_nonneg hn (M t)) (heta t)
      exact mul_le_mul iht (heta t) (dobrushinCoeff_nonneg hn (M t)) (pow_nonneg heta_nonneg t)
  have hhet := multistep_sdpi_tv_contraction hn M p q hsum t
  have htv_nonneg : 0 ≤ tvDist p q := by
    simp only [tvDist]
    exact mul_nonneg (by norm_num) (Finset.sum_nonneg (fun _ _ => abs_nonneg _))
  exact le_trans hhet (mul_le_mul_of_nonneg_right hprod htv_nonneg)

/-! ## Non-Vacuity Witness Examples for All Four Families -/

/-- Family 1 witness: `lam = ![2, 3]` and `d_mem = 1 < r = 2` linear projection gap. -/
example :
    ∃ v : Fin 2 → ℝ, ∃ j₀ : Fin 2, v j₀ ≠ 0 ∧
      (∀ i : Fin 1, ∑ j : Fin 2, (fun _ : Fin 1 => fun j' : Fin 2 => if j' = 0 then (1 : ℝ) else 0) i j * v j = 0) ∧
      ∃ s₀ : Fin 2 → ℝ, (s₀ = 0 ∨ s₀ = v) ∧
        (1 / 2 : ℝ) * |((fun j' : Fin 2 => if j' = 0 then (2 : ℝ) else 3) j₀)| ^ 3 * |v j₀| ≤
          |(fun _ : Fin 1 → ℝ => (fun _ : Fin 2 => (0 : ℝ)))
              (fun i => ∑ j : Fin 2, (fun _ : Fin 1 => fun j' : Fin 2 => if j' = 0 then (1 : ℝ) else 0) i j * s₀ j) j₀ -
            ((fun j' : Fin 2 => if j' = 0 then (2 : ℝ) else 3) j₀) ^ 3 * s₀ j₀| :=
  subcritical_linear_projection_gap 1 2 (by norm_num)
    (fun j' : Fin 2 => if j' = 0 then (2 : ℝ) else 3)
    (fun _ : Fin 1 => fun j' : Fin 2 => if j' = 0 then (1 : ℝ) else 0)
    (fun _ => fun _ => 0) 3

/-- Family 1 data-rate witness: `K = 2, lam = 3, Δ = 1` gives unbounded blow-up. -/
example :
    ∀ B : ℝ, ∃ T : ℕ, ∀ t : ℕ, T ≤ t →
      ∀ (c_t : ℝ) (enc : Fin (2 ^ t + 1) → Fin (2 ^ t)) (dec : Fin (2 ^ t) → ℝ),
        ∃ idx : Fin (2 ^ t + 1),
          B ≤ |dec (enc idx) - ((3 : ℝ) ^ t * ((idx.1 : ℝ) * (1 / (2 ^ t : ℝ))) + c_t)| :=
  data_rate_subcritical_blowup 2 (by norm_num) 3 1 (by norm_num) (by norm_num)

/-- Family 2 witness: `2 × 2` Jordan block growth and `2 × 2` transient amplification. -/
example :
    ((5 + 1 - 2 : ℝ) ^ 1 / (Nat.factorial 1 : ℝ)) *
        |((fun j : Fin 2 => if j = 0 then (0 : ℝ) else 1) ⟨1, by norm_num⟩)| ≤
      |(jordanStep 2)^[5] (fun j : Fin 2 => if j = 0 then (0 : ℝ) else 1) ⟨0, by norm_num⟩| := by
  exact jordan_top_mode_polynomial_lower_bound 2 5 (by norm_num) (by norm_num)
    (fun j : Fin 2 => if j = 0 then (0 : ℝ) else 1)
    (fun j hj => by
      have hj0 : j = 0 := Fin.ext (by omega)
      simp [hj0])

example :
    6 * (∑ m ∈ Finset.range 4, (m : ℝ) ^ 2) = 84 := by
  have h := jordan2_forced_sum_sq 4
  norm_num at h
  exact h

example :
    (nonnormal2Step (1 / 2) 10)^[2] (fun j : Fin 2 => if j = 0 then (0 : ℝ) else 1) 0 = 10 := by
  have h := (nonnormal_2x2_transient (1 / 2) 10
    (fun j : Fin 2 => if j = 0 then (0 : ℝ) else 1) 2 (by norm_num)).1
  norm_num at h
  exact h

/-- Family 3 witness: `rho = 1/2, W = 1, e 0 = 2` uniformly bounded by `4`. -/
example (e w : ℕ → ℝ) (he0 : e 0 = 2)
    (hrec : ∀ t : ℕ, e (t + 1) = (1 / 2 : ℝ) * e t + w t)
    (hw : ∀ t : ℕ, |w t| ≤ 1) (t : ℕ) :
    |e t| ≤ 4 := by
  have h := (contractive_error_bound (1 / 2) 1 (by norm_num) (by norm_num) e w hrec hw t).2
  rw [he0] at h
  norm_num at h
  exact h

/-- Family 3 composite witness: 1 exact retained `E^{u/c}` mode + 1 contractive `E^s` mode. -/
example (e_s w_s : ℕ → Fin 1 → ℝ) (he0 : ∀ j : Fin 1, |e_s 0 j| ≤ 2)
    (hw : ∀ t : ℕ, ∀ j : Fin 1, |w_s t j| ≤ 1)
    (hrec : ∀ t : ℕ, ∀ j : Fin 1, e_s (t + 1) j = (1 / 2 : ℝ) * e_s t j + w_s t j)
    (t τ : ℕ) :
    (∑ _i : Fin 1, ((2 : ℝ) ^ τ * (0 : ℝ)) ^ 2 + ∑ j : Fin 1, ((1 / 2 : ℝ) ^ τ * e_s t j) ^ 2) ≤
      (1 : ℝ) * (1 / 2 : ℝ) ^ (2 * τ) * 16 := by
  have h := (supercritical_composite_uniform_bound 1 1 (fun _ => 2) (fun _ _ => 0)
    (fun _ _ => rfl) (fun _ => 1 / 2) (1 / 2) 2 1 (by norm_num)
    (fun _ => by norm_num) (by norm_num) (by norm_num) e_s w_s he0 hw hrec t τ).1
  norm_num at h ⊢
  exact h

/-- Family 4 witness: Binary Symmetric Channel `BSC(1/4)` contracts TV by `(1 / 2) ^ t`. -/
noncomputable def bscQuarter : MarkovKernel 2 2 where
  K i j := if i = j then (3 / 4 : ℝ) else (1 / 4 : ℝ)
  nonneg i j := by split_ifs <;> norm_num
  row_sum i := by
    rw [Fin.sum_univ_two]
    refine Fin.cases ?_ ?_ i
    · norm_num
    · intro j; refine Fin.cases ?_ ?_ j <;> [norm_num; exact fun k => Fin.elim0 k]

example (p q : Fin 2 → ℝ) (hsum : ∑ i : Fin 2, p i = ∑ i : Fin 2, q i) (t : ℕ) :
    tvDist (chainPush p (fun _ => bscQuarter) t) (chainPush q (fun _ => bscQuarter) t) ≤
      (1 / 2 : ℝ) ^ t * tvDist p q := by
  refine multistep_sdpi_homogeneous (by norm_num) (fun _ => bscQuarter) (1 / 2) ?_ p q hsum t
  intro _
  have hdoeb := doeblin_contraction_bound (by norm_num : 0 < 2) bscQuarter (fun _ => (1 / 4 : ℝ))
    (fun i j => by
      simp only [bscQuarter]
      split_ifs <;> norm_num)
  norm_num at hdoeb
  exact hdoeb

/-! ## Axiom Audit for Every Main Theorem -/

#print axioms linear_error_closed_form
#print axioms unforced_exp_divergence
#print axioms exists_nonzero_ker_of_lt
#print axioms subcritical_linear_projection_gap
#print axioms subcritical_linear_projection_blowup
#print axioms data_rate_pigeonhole_lower_bound
#print axioms data_rate_subcritical_blowup
#print axioms jordanStep_iterate_formula
#print axioms jordan_top_mode_polynomial_lower_bound
#print axioms jordan2_forced_sum_sq
#print axioms nonnormal_2x2_transient
#print axioms contractive_error_bound
#print axioms supercritical_composite_uniform_bound
#print axioms dobrushinCoeff_nonneg
#print axioms dobrushinCoeff_le_one
#print axioms doeblin_contraction_bound
#print axioms dobrushin_tv_contraction
#print axioms push_sum
#print axioms chainPush_sum
#print axioms multistep_sdpi_tv_contraction
#print axioms multistep_sdpi_homogeneous

end MathLab.Projects.LateQueryFrontier.MemoryPhase
