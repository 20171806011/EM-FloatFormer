import Mathlib.Basic.Real.Basic
import Mathlib.Analysis.InnerProductSpace.GramMatrix
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.LinearAlgebra.Vandermonde
import Mathlib.LinearAlgebra.Dimension.StrongRankCondition
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace MathLab.Projects.LateQueryFrontier.CompInflation

open scoped BigOperators InnerProductSpace Matrix

/-!
# Conjecture 2: Computation-Limited Readout & Dimension Inflation (`CompInflation`)

Formal verification of the core load-bearing structural obstructions for Conjecture 2:
* **R2 (Linear Readout Inflation)**:
  1. `linear_readout_irreducible_error`: Subspace-rank obstruction (`r < K`) in any real
     `InnerProductSpace` yields strictly positive irreducible error bounded below by the
     minimum Gram eigenvalue.
  2. `monomial_linearIndependent_Icc`: Linear independence of `t ↦ t ^ (k + 1)` on `[0, 1]`
     via `Matrix.det_vandermonde`.
  3. `monomial_empirical_L2_irreducible_error`: Empirical `L²` instantiation on `K` distinct
     nodes in `(0, 1]` in `EuclideanSpace ℝ (Fin K)`.
* **R3 (Lipschitz Readout Tax)**:
  4. `lipschitz_radius_inflation`: Metric distortion & representation radius lower bound.
  5. `lipschitz_covering_lower_bound`: Multi-query `L`-Lipschitz `L²` covering lower bound
     `(N : ℝ) / 2 ≤ K_cov` via exact Rademacher double-counting.
* **R4 (Shallow ReLU Collapse Boundary)**:
  6. `affine_quadratic_interval_floor`: 3-point second-difference `L^∞` floor `(u₁ - u₀)^2 / 8`
     for affine approximation of `u^2` on `[u₀, u₁]`.
  7. `piecewise_affine_quadratic_floor`: `1 / (8 * (W + 1)^2)` floor on a `(W + 1)`-piece
     partition of `[0, 1]`.
  8. `relu_depth2_quadratic_floor`: Direct floor `1 / (8 * (W + 1)^2)` for any depth-2
     width-`W` ReLU network on `[0, 1]`.
-/

/-! ## R2: Linear Readout Inflation -/

/-- **R2 (Subspace-Rank Irreducible Error)**: In any real `InnerProductSpace`, `K` linearly
independent targets `T : Fin K → H` and cache dimension `r < K` yield `ε₀ > 0` (the minimum
Gram eigenvalue) such that every `r`-dimensional linear readout incurs squared error at least `ε₀`. -/
theorem linear_readout_irreducible_error
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    {K r : ℕ} (hK : 0 < K) (hrK : r < K)
    (T : Fin K → H) (hT : LinearIndependent ℝ T) :
    ∃ ε₀ : ℝ, 0 < ε₀ ∧
      ∀ (φ : Fin r → H) (W : Fin K → Fin r → ℝ),
        ε₀ ≤ ∑ i : Fin K, ‖T i - ∑ j : Fin r, W i j • φ j‖ ^ 2 := by
  let G : Matrix (Fin K) (Fin K) ℝ := Matrix.gram ℝ T
  have hG_herm : G.IsHermitian := Matrix.isHermitian_gram ℝ T
  have hG_pos : G.PosDef := Matrix.posDef_gram_of_linearIndependent hT
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  let ε₀ : ℝ := Finset.univ.inf' Finset.univ_nonempty hG_herm.eigenvalues
  have hε₀_pos : 0 < ε₀ :=
    (Finset.lt_inf'_iff Finset.univ_nonempty).mpr fun i _ => hG_pos.eigenvalues_pos i
  have hε₀_le : ∀ i : Fin K, ε₀ ≤ hG_herm.eigenvalues i :=
    fun i => Finset.inf'_le _ (Finset.mem_univ i)
  have h_coercive : ∀ c : Fin K → ℝ,
      ε₀ * ∑ i : Fin K, (c i) ^ 2 ≤ ‖∑ i : Fin K, c i • T i‖ ^ 2 := by
    intro c
    let U : Matrix (Fin K) (Fin K) ℝ := (hG_herm.eigenvectorUnitary : Matrix (Fin K) (Fin K) ℝ)
    let y : Fin K → ℝ := Uᵀ *ᵥ c
    have hU_star : star U = Uᵀ := by
      ext i j
      simp
    have hU_mul : U * Uᵀ = 1 := by
      rw [← hU_star]
      exact Unitary.coe_mul_star_self hG_herm.eigenvectorUnitary
    have hy_norm : y ⬝ᵥ y = ∑ i : Fin K, (c i) ^ 2 := by
      have h1 : y ⬝ᵥ y = (c ᵥ* U) ⬝ᵥ (Uᵀ *ᵥ c) := by
        simp [y, Matrix.mulVec_transpose]
      rw [h1, Matrix.dotProduct_mulVec, Matrix.vecMul_vecMul, hU_mul, Matrix.vecMul_one]
      simp [dotProduct, sq]
    have hG_spec : G = U * Matrix.diagonal hG_herm.eigenvalues * Uᵀ := by
      have hs := hG_herm.spectral_theorem
      rw [Unitary.conjStarAlgAut_apply] at hs
      simpa [U, hU_star, Function.comp_def] using hs
    have h_norm_eq : ‖∑ i : Fin K, c i • T i‖ ^ 2 = c ⬝ᵥ (G *ᵥ c) := by
      rw [norm_sq_eq_re_inner (𝕜 := ℝ), RCLike.re_to_real]
      have hgram := Matrix.star_dotProduct_gram_mulVec (𝕜 := ℝ) T c c
      simpa [G] using hgram.symm
    have h_quad : c ⬝ᵥ (G *ᵥ c) = ∑ i : Fin K, hG_herm.eigenvalues i * (y i) ^ 2 := by
      conv_lhs => rw [hG_spec]
      rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
      have h2 : c ⬝ᵥ (U *ᵥ (Matrix.diagonal hG_herm.eigenvalues *ᵥ y)) =
          y ⬝ᵥ (Matrix.diagonal hG_herm.eigenvalues *ᵥ y) := by
        rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose]
      rw [h2]
      simp only [dotProduct, Matrix.mulVec_diagonal]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [h_norm_eq, h_quad, ← hy_norm]
    have hsum_y : ε₀ * (y ⬝ᵥ y) = ∑ i : Fin K, ε₀ * (y i) ^ 2 := by
      simp [dotProduct, Finset.mul_sum, sq]
    rw [hsum_y]
    exact Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_right (hε₀_le i) (sq_nonneg (y i))
  refine ⟨ε₀, hε₀_pos, fun φ W => ?_⟩
  let L : (Fin K → ℝ) →ₗ[ℝ] (Fin r → ℝ) :=
    { toFun := fun c j => ⟪∑ i : Fin K, c i • T i, φ j⟫_ℝ
      map_add' := fun c₁ c₂ => by
        ext j
        simp [add_smul, Finset.sum_add_distrib, inner_add_left]
      map_smul' := fun a c => by
        ext j
        simp only [RingHom.id_apply, Pi.smul_apply, smul_eq_mul, mul_smul,
          ← Finset.smul_sum, real_inner_smul_left] }
  have hL_not_inj : ¬ Function.Injective L := by
    intro h_inj
    have hfin := LinearMap.finrank_le_finrank_of_injective h_inj
    simp only [Module.finrank_fin_fun] at hfin
    omega
  rw [Function.Injective] at hL_not_inj
  push Not at hL_not_inj
  rcases hL_not_inj with ⟨c₁, c₂, hLc, hne⟩
  let c : Fin K → ℝ := c₁ - c₂
  have hc_ne : c ≠ 0 := sub_ne_zero.mpr hne
  have hLc_zero : ∀ j : Fin r, ⟪∑ i : Fin K, c i • T i, φ j⟫_ℝ = 0 := by
    intro j
    have hj := congr_fun hLc j
    dsimp [L] at hj
    have hsub : ∑ i : Fin K, c i • T i = (∑ i : Fin K, c₁ i • T i) - (∑ i : Fin K, c₂ i • T i) := by
      simp only [c, Pi.sub_apply, sub_smul, Finset.sum_sub_distrib]
    rw [hsub, inner_sub_left, hj, sub_self]
  let g : Fin K → H := fun i => ∑ j : Fin r, W i j • φ j
  let v : H := ∑ i : Fin K, c i • T i
  let w : H := ∑ i : Fin K, c i • g i
  have hvw_orth : ⟪v, w⟫_ℝ = 0 := by
    dsimp [v, w, g]
    simp_rw [Finset.smul_sum, inner_sum, smul_smul, real_inner_smul_right, hLc_zero, mul_zero,
      Finset.sum_const_zero]
  have h_pyth : ‖v‖ ^ 2 ≤ ‖v - w‖ ^ 2 := by
    rw [norm_sub_sq_real, hvw_orth]
    nlinarith [sq_nonneg ‖w‖]
  have h_vw_eq : v - w = ∑ i : Fin K, c i • (T i - g i) := by
    simp [v, w, smul_sub, Finset.sum_sub_distrib]
  have h_cs : ‖v - w‖ ^ 2 ≤ (∑ i : Fin K, (c i) ^ 2) * (∑ i : Fin K, ‖T i - g i‖ ^ 2) := by
    have h_tri : ‖v - w‖ ≤ ∑ i : Fin K, |c i| * ‖T i - g i‖ := by
      rw [h_vw_eq]
      refine (norm_sum_le _ _).trans_eq ?_
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [norm_smul, Real.norm_eq_abs]
    have h_sq : ‖v - w‖ ^ 2 ≤ (∑ i : Fin K, |c i| * ‖T i - g i‖) ^ 2 :=
      pow_le_pow_left₀ (norm_nonneg _) h_tri 2
    have h_sum_cs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin K))
      (fun i => |c i|) (fun i => ‖T i - g i‖)
    have h_abs_sq : (∑ i : Fin K, |c i| ^ 2) = ∑ i : Fin K, (c i) ^ 2 := by
      refine Finset.sum_congr rfl fun i _ => sq_abs (c i)
    rw [h_abs_sq] at h_sum_cs
    exact h_sq.trans h_sum_cs
  have hc_sq_pos : 0 < ∑ i : Fin K, (c i) ^ 2 := by
    obtain ⟨i₀, hi₀⟩ := Function.ne_iff.mp hc_ne
    have hpos : 0 < (c i₀) ^ 2 := sq_pos_of_ne_zero hi₀
    have hle : (c i₀) ^ 2 ≤ ∑ i : Fin K, (c i) ^ 2 :=
      Finset.single_le_sum (fun i _ => sq_nonneg (c i)) (Finset.mem_univ i₀)
    linarith
  have h_mul_le : ε₀ * (∑ i : Fin K, (c i) ^ 2) ≤
      (∑ i : Fin K, (c i) ^ 2) * (∑ i : Fin K, ‖T i - g i‖ ^ 2) :=
    (h_coercive c).trans (h_pyth.trans h_cs)
  nlinarith

/-- **R2 (Monomial Linear Independence on `[0, 1]`)**: The shifted monomials
`t ↦ t ^ (k + 1)` (`k : Fin K`) are linearly independent over `ℝ` on `Set.Icc (0 : ℝ) 1`,
proved via `Matrix.det_vandermonde` on `K` distinct interior/boundary nodes in `(0, 1]`. -/
theorem monomial_linearIndependent_Icc (K : ℕ) :
    LinearIndependent ℝ (fun (k : Fin K) (t : Set.Icc (0 : ℝ) 1) => (t.1) ^ (k.1 + 1)) := by
  rw [Fintype.linearIndependent_iff]
  intro c hc k
  let x : Fin K → ℝ := fun i => ((i.1 : ℝ) + 1) / ((K : ℝ) + 1)
  have hx_pos : ∀ i : Fin K, 0 < x i := fun i => by positivity
  have hx_le : ∀ i : Fin K, x i ≤ 1 := fun i => by
    dsimp [x]
    rw [div_le_one (by positivity)]
    have hi : (i.1 : ℝ) + 1 ≤ K := by exact_mod_cast Nat.succ_le_of_lt i.isLt
    linarith
  have hx_inj : Function.Injective x := by
    intro i j hij
    dsimp [x] at hij
    have hden : (K : ℝ) + 1 ≠ 0 := by positivity
    have hnum : (i.1 : ℝ) + 1 = (j.1 : ℝ) + 1 := (div_left_inj₀ hden).mp hij
    have hnat : i.1 = j.1 := by exact_mod_cast add_right_cancel hnum
    exact Fin.ext hnat
  have h_sum_zero : ∀ i : Fin K, (∑ m : Fin K, c m * (x i) ^ (m : ℕ)) = 0 := by
    intro i
    have h_eval := congr_fun hc ⟨x i, (hx_pos i).le, hx_le i⟩
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h_eval
    have h_fact : (∑ m : Fin K, c m * (x i) ^ (m.1 + 1)) =
        x i * (∑ m : Fin K, c m * (x i) ^ (m : ℕ)) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun m _ => ?_
      rw [pow_succ']
      ring
    rw [h_fact] at h_eval
    exact (mul_eq_zero.mp h_eval).resolve_left (hx_pos i).ne'
  have hc_zero := Matrix.eq_zero_of_forall_index_sum_mul_pow_eq_zero hx_inj h_sum_zero
  exact congr_fun hc_zero k

/-- **R2 (Empirical `L²` Irreducible Error for Monomials)**: Instantiation of
`linear_readout_irreducible_error` on `EuclideanSpace ℝ (Fin K)` over any `K` distinct nodes
`t : Fin K → ℝ` in `(0, 1]`, proving strictly positive irreducible error whenever `r < K`. -/
theorem monomial_empirical_L2_irreducible_error
    {K r : ℕ} (hK : 0 < K) (hrK : r < K)
    (t : Fin K → ℝ) (ht_pos : ∀ i : Fin K, 0 < t i) (ht_le : ∀ i : Fin K, t i ≤ 1)
    (ht_inj : Function.Injective t) :
    let T : Fin K → EuclideanSpace ℝ (Fin K) :=
      fun k => WithLp.toLp 2 (fun i => (t i) ^ (k.1 + 1))
    ∃ ε₀ : ℝ, 0 < ε₀ ∧
      ∀ (φ : Fin r → EuclideanSpace ℝ (Fin K)) (W : Fin K → Fin r → ℝ),
        ε₀ ≤ ∑ k : Fin K, ‖T k - ∑ j : Fin r, W k j • φ j‖ ^ 2 := by
  intro T
  have _ := ht_le
  have hT_li : LinearIndependent ℝ T := by
    rw [Fintype.linearIndependent_iff]
    intro c hc k
    have h_sum_zero : ∀ i : Fin K, (∑ m : Fin K, c m * (t i) ^ (m : ℕ)) = 0 := by
      intro i
      have h_eval := congr_fun (congr_arg WithLp.ofLp hc) i
      simp only [T, WithLp.ofLp_sum, WithLp.ofLp_smul,
        Finset.sum_apply, Pi.smul_apply, smul_eq_mul, WithLp.ofLp_zero, Pi.zero_apply] at h_eval
      have h_fact : (∑ m : Fin K, c m * (t i) ^ (m.1 + 1)) =
          t i * (∑ m : Fin K, c m * (t i) ^ (m : ℕ)) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun m _ => ?_
        rw [pow_succ']
        ring
      rw [h_fact] at h_eval
      exact (mul_eq_zero.mp h_eval).resolve_left (ht_pos i).ne'
    exact congr_fun (Matrix.eq_zero_of_forall_index_sum_mul_pow_eq_zero ht_inj h_sum_zero) k
  exact linear_readout_irreducible_error hK hrK T hT_li

/-! ## R3: Lipschitz / Bounded-Gradient Nonlinear Readout Tax -/

/-- **R3 (Lipschitz Metric Distortion & Radius Inflation)**: Any `L`-Lipschitz readout `Φ`
approximating `T` within `δ` on `z : α → E` forces representation separation
`‖z x - z x'‖ ≥ (|T x - T x'| - 2δ) / L` and norm radius `2 L R_z ≥ |T x - T x'| - 2δ`. -/
theorem lipschitz_radius_inflation
    {α E : Type*} [NormedAddCommGroup E]
    (T : α → ℝ) (z : α → E) (Φ : E → ℝ) {L Rz δ : ℝ}
    (hL : 0 ≤ L)
    (hLip : ∀ u v : E, |Φ u - Φ v| ≤ L * ‖u - v‖)
    (hBound : ∀ x : α, ‖z x‖ ≤ Rz)
    (hApprox : ∀ x : α, |T x - Φ (z x)| ≤ δ) :
    ∀ x x' : α, |T x - T x'| - 2 * δ ≤ L * ‖z x - z x'‖ ∧ |T x - T x'| - 2 * δ ≤ 2 * L * Rz := by
  intro x x'
  have h1 : |T x - T x'| ≤ |T x - Φ (z x)| + |Φ (z x) - Φ (z x')| + |Φ (z x') - T x'| := by
    have hsplit : T x - T x' = (T x - Φ (z x)) + ((Φ (z x) - Φ (z x')) + (Φ (z x') - T x')) := by ring
    have h_tri1 := abs_add_le (T x - Φ (z x)) ((Φ (z x) - Φ (z x')) + (Φ (z x') - T x'))
    have h_tri2 := abs_add_le (Φ (z x) - Φ (z x')) (Φ (z x') - T x')
    rw [hsplit]
    linarith
  have hax : |T x - Φ (z x)| ≤ δ := hApprox x
  have hax' : |Φ (z x') - T x'| ≤ δ := by
    rw [abs_sub_comm]
    exact hApprox x'
  have hphi : |Φ (z x) - Φ (z x')| ≤ L * ‖z x - z x'‖ := hLip (z x) (z x')
  have hdist : L * ‖z x - z x'‖ ≤ 2 * L * Rz := by
    have htri : ‖z x - z x'‖ ≤ ‖z x‖ + ‖z x'‖ := norm_sub_le (z x) (z x')
    have h2 : ‖z x - z x'‖ ≤ 2 * Rz := by linarith [hBound x, hBound x']
    nlinarith
  constructor
  · linarith
  · linarith

/-- **R3 (Multi-Query `L`-Lipschitz Covering Lower Bound)**: If `N > 0` cache points `{z i}`
are covered by `K_cov` balls of radius `A / (4 L)` and the average squared error of `L`-Lipschitz
readouts `Φ σ` across all sign patterns `σ : Fin N → Bool` is at most `A^2 / 64` per point,
then `K_cov ≥ N / 2`. -/
theorem lipschitz_covering_lower_bound
    {E : Type*} [NormedAddCommGroup E]
    {N K_cov : ℕ} (hN : 0 < N) {A L : ℝ} (hA : 0 < A) (hL : 0 < L)
    (z : Fin N → E) (center : Fin K_cov → E)
    (hCov : ∀ i : Fin N, ∃ m : Fin K_cov, ‖z i - center m‖ ≤ A / (4 * L))
    (Φ : (Fin N → Bool) → E → ℝ)
    (hLip : ∀ (σ : Fin N → Bool) (u v : E), |Φ σ u - Φ σ v| ≤ L * ‖u - v‖)
    (hErr : (∑ σ : Fin N → Bool, ∑ i : Fin N,
      (Φ σ (z i) - (if σ i then A else -A)) ^ 2) ≤ (2 ^ N : ℝ) * N * (A ^ 2 / 64)) :
    (N : ℝ) / 2 ≤ K_cov := by
  classical
  have _ := hN
  let assign : Fin N → Fin K_cov := fun i => Classical.choose (hCov i)
  have h_assign : ∀ i : Fin N, ‖z i - center (assign i)‖ ≤ A / (4 * L) :=
    fun i => Classical.choose_spec (hCov i)
  let B : Fin K_cov → Finset (Fin N) := fun m => Finset.univ.filter (fun i => assign i = m)
  have h_part : ∀ (f : Fin N → ℝ),
      (∑ i : Fin N, f i) = ∑ m : Fin K_cov, ∑ i ∈ B m, f i := by
    intro f
    rw [← Finset.sum_fiberwise_of_maps_to (g := assign) (t := Finset.univ)
      (fun i _ => Finset.mem_univ (assign i))]
  have h_card_sum : (∑ m : Fin K_cov, ((B m).card : ℝ)) = N := by
    have h1 := h_part (fun _ => (1 : ℝ))
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one, Finset.card_univ] at h1
    simpa [nsmul_eq_mul, mul_one] using h1.symm
  -- Pairwise sign-disagreement sum: for i ≠ j in Fin N, |{σ | σ i = true ∧ σ j = false}| = 2^N / 4.
  have h_pair_count : ∀ {i j : Fin N}, i ≠ j →
      4 * (∑ σ : Fin N → Bool, if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) = (2 ^ N : ℝ) := by
    intro i j hij
    let flip : Fin N → (Fin N → Bool) ≃ (Fin N → Bool) := fun k =>
      { toFun := fun σ => Function.update σ k (!(σ k))
        invFun := fun σ => Function.update σ k (!(σ k))
        left_inv := fun σ => by
          ext t
          by_cases ht : t = k
          · subst ht; simp
          · simp [ht]
        right_inv := fun σ => by
          ext t
          by_cases ht : t = k
          · subst ht; simp
          · simp [ht] }
    have h_tf_ff : (∑ σ : Fin N → Bool, if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) =
        ∑ σ : Fin N → Bool, if (σ i = false ∧ σ j = false) then (1 : ℝ) else 0 := by
      refine Fintype.sum_equiv (flip i) _ _ fun σ => ?_
      have h_eq_i : (flip i σ) i = !(σ i) := by simp [flip]
      have h_eq_j : (flip i σ) j = σ j := by simp [flip, hij.symm]
      rw [h_eq_i, h_eq_j]
      cases σ i <;> cases σ j <;> simp
    have h_tf_tt : (∑ σ : Fin N → Bool, if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) =
        ∑ σ : Fin N → Bool, if (σ i = true ∧ σ j = true) then (1 : ℝ) else 0 := by
      refine Fintype.sum_equiv (flip j) _ _ fun σ => ?_
      have h_eq_i : (flip j σ) i = σ i := by simp [flip, hij]
      have h_eq_j : (flip j σ) j = !(σ j) := by simp [flip]
      rw [h_eq_i, h_eq_j]
      cases σ i <;> cases σ j <;> simp
    have h_tt_ft : (∑ σ : Fin N → Bool, if (σ i = true ∧ σ j = true) then (1 : ℝ) else 0) =
        ∑ σ : Fin N → Bool, if (σ i = false ∧ σ j = true) then (1 : ℝ) else 0 := by
      refine Fintype.sum_equiv (flip i) _ _ fun σ => ?_
      have h_eq_i : (flip i σ) i = !(σ i) := by simp [flip]
      have h_eq_j : (flip i σ) j = σ j := by simp [flip, hij.symm]
      rw [h_eq_i, h_eq_j]
      cases σ i <;> cases σ j <;> simp
    have h_tf_ft := h_tf_tt.trans h_tt_ft
    have h_sum_all : (∑ σ : Fin N → Bool,
        ((if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) +
         (if (σ i = false ∧ σ j = false) then (1 : ℝ) else 0) +
         (if (σ i = true ∧ σ j = true) then (1 : ℝ) else 0) +
         (if (σ i = false ∧ σ j = true) then (1 : ℝ) else 0))) = (2 ^ N : ℝ) := by
      have h_pt : ∀ σ : Fin N → Bool,
          ((if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) +
           (if (σ i = false ∧ σ j = false) then (1 : ℝ) else 0) +
           (if (σ i = true ∧ σ j = true) then (1 : ℝ) else 0) +
           (if (σ i = false ∧ σ j = true) then (1 : ℝ) else 0)) = 1 := by
        intro σ
        cases σ i <;> cases σ j <;> norm_num
      simp_rw [h_pt]
      simp [Fintype.card_bool, Fintype.card_fin]
    simp_rw [Finset.sum_add_distrib] at h_sum_all
    linarith
  -- Per-ball lower bound over all σ
  have h_ball_err : ∀ m : Fin K_cov,
      (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4) ≤
        ∑ σ : Fin N → Bool, ∑ i ∈ B m, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
    intro m
    let n_plus (σ : Fin N → Bool) : ℕ := ((B m).filter (fun i => σ i = true)).card
    let n_minus (σ : Fin N → Bool) : ℕ := ((B m).filter (fun i => σ i = false)).card
    have h_split_card : ∀ σ : Fin N → Bool, n_plus σ + n_minus σ = (B m).card := by
      intro σ
      dsimp [n_plus, n_minus]
      have hdisj : Disjoint ((B m).filter (fun i => σ i = true))
          ((B m).filter (fun i => σ i = false)) := by
        rw [Finset.disjoint_filter]
        intro i _ h1 h2
        rw [h1] at h2
        cases h2
      have hunion : (B m).filter (fun i => σ i = true) ∪
          (B m).filter (fun i => σ i = false) = B m := by
        ext i
        simp only [Finset.mem_union, Finset.mem_filter]
        cases σ i <;> simp
      rw [← Finset.card_union_of_disjoint hdisj, hunion]
    have h_per_sigma : ∀ σ : Fin N → Bool,
        (min (n_plus σ) (n_minus σ) : ℝ) * (A ^ 2 / 4) ≤
          ∑ i ∈ B m, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
      intro σ
      have h_near : ∀ i ∈ B m, |Φ σ (z i) - Φ σ (center m)| ≤ A / 4 := by
        intro i hi
        have him : assign i = m := (Finset.mem_filter.mp hi).2
        have h1 := hLip σ (z i) (center m)
        have h2 : ‖z i - center m‖ ≤ A / (4 * L) := him ▸ h_assign i
        have h3 : L * ‖z i - center m‖ ≤ L * (A / (4 * L)) :=
          mul_le_mul_of_nonneg_left h2 hL.le
        have h4 : L * (A / (4 * L)) = A / 4 := by
          field_simp [hL.ne']
        linarith
      by_cases hv : Φ σ (center m) ≤ 0
      · have h_each : ∀ i ∈ (B m).filter (fun i => σ i = true),
            A ^ 2 / 4 ≤ (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
          intro i hi
          obtain ⟨hiB, hsi⟩ := Finset.mem_filter.mp hi
          have hni := h_near i hiB
          have h_upper : Φ σ (z i) ≤ A / 4 := by linarith [(abs_le.mp hni).2]
          simp only [hsi, ↓reduceIte]
          have h_diff : A / 2 ≤ A - Φ σ (z i) := by linarith
          nlinarith
        have h_sub : ((B m).filter (fun i => σ i = true)) ⊆ B m := Finset.filter_subset _ _
        have h_sum1 : (n_plus σ : ℝ) * (A ^ 2 / 4) ≤
            ∑ i ∈ (B m).filter (fun i => σ i = true),
              (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
          have := Finset.sum_le_sum (fun i hi => h_each i hi)
          simpa [n_plus, Finset.sum_const, nsmul_eq_mul] using this
        have h_sum2 : ∑ i ∈ (B m).filter (fun i => σ i = true),
              (Φ σ (z i) - (if σ i then A else -A)) ^ 2 ≤
            ∑ i ∈ B m, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg h_sub (fun i _ _ => sq_nonneg _)
        have h_min : (min (n_plus σ) (n_minus σ) : ℝ) ≤ (n_plus σ : ℝ) := by
          exact_mod_cast min_le_left (n_plus σ) (n_minus σ)
        exact (mul_le_mul_of_nonneg_right h_min (by positivity)).trans (h_sum1.trans h_sum2)
      · have hv_pos : 0 ≤ Φ σ (center m) := le_of_not_ge hv
        have h_each : ∀ i ∈ (B m).filter (fun i => σ i = false),
            A ^ 2 / 4 ≤ (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
          intro i hi
          obtain ⟨hiB, hsi⟩ := Finset.mem_filter.mp hi
          have hni := h_near i hiB
          have h_lower : -(A / 4) ≤ Φ σ (z i) := by linarith [(abs_le.mp hni).1]
          simp only [hsi, Bool.false_eq_true, ↓reduceIte]
          have h_diff : A / 2 ≤ Φ σ (z i) - (-A) := by linarith
          nlinarith
        have h_sub : ((B m).filter (fun i => σ i = false)) ⊆ B m := Finset.filter_subset _ _
        have h_sum1 : (n_minus σ : ℝ) * (A ^ 2 / 4) ≤
            ∑ i ∈ (B m).filter (fun i => σ i = false),
              (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
          have := Finset.sum_le_sum (fun i hi => h_each i hi)
          simpa [n_minus, Finset.sum_const, nsmul_eq_mul] using this
        have h_sum2 : ∑ i ∈ (B m).filter (fun i => σ i = false),
              (Φ σ (z i) - (if σ i then A else -A)) ^ 2 ≤
            ∑ i ∈ B m, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg h_sub (fun i _ _ => sq_nonneg _)
        have h_min : (min (n_plus σ) (n_minus σ) : ℝ) ≤ (n_minus σ : ℝ) := by
          exact_mod_cast min_le_right (n_plus σ) (n_minus σ)
        exact (mul_le_mul_of_nonneg_right h_min (by positivity)).trans (h_sum1.trans h_sum2)
    have h_sum_min : (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) ≤
        ∑ σ : Fin N → Bool, (min (n_plus σ) (n_minus σ) : ℝ) := by
      by_cases hBm_le : (B m).card ≤ 1
      · have h_lhs_le : (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) ≤ 0 := by
          have h1 : ((B m).card : ℝ) ≤ 1 := by exact_mod_cast hBm_le
          have h2 : ((B m).card : ℝ) - 1 ≤ 0 := by linarith
          exact mul_nonpos_of_nonpos_of_nonneg h2 (by positivity)
        have h_rhs_nonneg : 0 ≤ ∑ σ : Fin N → Bool, (min (n_plus σ) (n_minus σ) : ℝ) :=
          Finset.sum_nonneg fun σ _ => by positivity
        exact h_lhs_le.trans h_rhs_nonneg
      · have hBm_pos : 0 < ((B m).card : ℝ) := by
          have : 0 < (B m).card := by omega
          exact_mod_cast this
        have h_prod_le : ∀ σ : Fin N → Bool,
            (n_plus σ : ℝ) * (n_minus σ : ℝ) ≤
              ((B m).card : ℝ) * (min (n_plus σ) (n_minus σ) : ℝ) := by
          intro σ
          have hnat : n_plus σ * n_minus σ ≤ (B m).card * min (n_plus σ) (n_minus σ) := by
            have hmax : max (n_plus σ) (n_minus σ) ≤ (B m).card := by
              rw [← h_split_card σ]
              exact max_le (Nat.le_add_right _ _) (Nat.le_add_left _ _)
            have hpm : n_plus σ * n_minus σ =
                max (n_plus σ) (n_minus σ) * min (n_plus σ) (n_minus σ) := by
              rcases le_total (n_plus σ) (n_minus σ) with h | h
              · rw [max_eq_right h, min_eq_left h, mul_comm]
              · rw [max_eq_left h, min_eq_right h]
            rw [hpm]
            exact Nat.mul_le_mul_right _ hmax
          exact_mod_cast hnat
        have h_double_count :
            (∑ σ : Fin N → Bool, (n_plus σ : ℝ) * (n_minus σ : ℝ)) =
              ((B m).card : ℝ) * (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) := by
          have h_expand : ∀ σ : Fin N → Bool,
              (n_plus σ : ℝ) * (n_minus σ : ℝ) =
                ∑ i ∈ B m, ∑ j ∈ B m, if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0 := by
            intro σ
            have hp : (n_plus σ : ℝ) = ∑ i ∈ B m, if σ i = true then (1 : ℝ) else 0 := by
              simp only [n_plus, Finset.card_filter, Nat.cast_sum]
              refine Finset.sum_congr rfl fun i _ => ?_
              split_ifs <;> simp
            have hm : (n_minus σ : ℝ) = ∑ j ∈ B m, if σ j = false then (1 : ℝ) else 0 := by
              simp only [n_minus, Finset.card_filter, Nat.cast_sum]
              refine Finset.sum_congr rfl fun j _ => ?_
              split_ifs <;> simp
            rw [hp, hm, Finset.sum_mul]
            refine Finset.sum_congr rfl fun i _ => ?_
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun j _ => ?_
            by_cases h1 : σ i = true <;> by_cases h2 : σ j = false <;> simp [h1, h2]
          simp_rw [h_expand]
          rw [Finset.sum_comm]
          have h_swap : (∑ i ∈ B m, ∑ σ : Fin N → Bool, ∑ j ∈ B m,
              if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) =
              ∑ i ∈ B m, ∑ j ∈ B m, ∑ σ : Fin N → Bool,
                if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0 :=
            Finset.sum_congr rfl fun i _ => Finset.sum_comm
          rw [h_swap]
          have h_inner : ∀ i ∈ B m,
              (∑ j ∈ B m, ∑ σ : Fin N → Bool,
                if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) =
                (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) := by
            intro i hi
            have h_split_j : (∑ j ∈ B m, ∑ σ : Fin N → Bool,
                if (σ i = true ∧ σ j = false) then (1 : ℝ) else 0) =
                ∑ j ∈ B m, if j = i then (0 : ℝ) else ((2 ^ N : ℝ) / 4) := by
              refine Finset.sum_congr rfl fun j _ => ?_
              by_cases hji : j = i
              · subst hji
                simp
              · simp only [hji, ↓reduceIte]
                have := h_pair_count (fun h => hji h.symm)
                linarith
            rw [h_split_j, Finset.sum_ite, Finset.sum_const_zero, zero_add]
            have h_erase : ((B m).filter (fun j => ¬ j = i)).card = (B m).card - 1 := by
              rw [Finset.filter_ne', Finset.card_erase_of_mem hi]
            simp only [Finset.sum_const, nsmul_eq_mul, h_erase]
            have h1_le : 1 ≤ (B m).card := Finset.card_pos.mpr ⟨i, hi⟩
            rw [Nat.cast_sub h1_le, Nat.cast_one]
          have h_outer : (∑ i ∈ B m, (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4)) =
              ((B m).card : ℝ) * (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) := by
            simp [Finset.sum_const, nsmul_eq_mul, mul_assoc]
          exact (Finset.sum_congr rfl h_inner).trans h_outer
        have h_scaled : ((B m).card : ℝ) * ((((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4)) ≤
            ((B m).card : ℝ) * (∑ σ : Fin N → Bool, (min (n_plus σ) (n_minus σ) : ℝ)) := by
          rw [← mul_assoc, ← h_double_count, Finset.mul_sum]
          exact Finset.sum_le_sum fun σ _ => h_prod_le σ
        exact le_of_mul_le_mul_left h_scaled hBm_pos
    have h_sum_per : (∑ σ : Fin N → Bool, (min (n_plus σ) (n_minus σ) : ℝ) * (A ^ 2 / 4)) ≤
        ∑ σ : Fin N → Bool, ∑ i ∈ B m, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 :=
      Finset.sum_le_sum fun σ _ => h_per_sigma σ
    rw [← Finset.sum_mul] at h_sum_per
    exact (mul_le_mul_of_nonneg_right h_sum_min (by positivity)).trans h_sum_per
  have h_total_lower : ((N : ℝ) - K_cov) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4) ≤
      ∑ σ : Fin N → Bool, ∑ i : Fin N, (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
    have h_rewrite : (∑ σ : Fin N → Bool, ∑ i : Fin N,
        (Φ σ (z i) - (if σ i then A else -A)) ^ 2) =
        ∑ m : Fin K_cov, ∑ σ : Fin N → Bool, ∑ i ∈ B m,
          (Φ σ (z i) - (if σ i then A else -A)) ^ 2 := by
      simp_rw [h_part]
      exact Finset.sum_comm
    rw [h_rewrite]
    have h_sum_balls :
        (∑ m : Fin K_cov, (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4)) ≤
        ∑ m : Fin K_cov, ∑ σ : Fin N → Bool, ∑ i ∈ B m,
          (Φ σ (z i) - (if σ i then A else -A)) ^ 2 :=
      Finset.sum_le_sum fun m _ => h_ball_err m
    have h_lhs : (∑ m : Fin K_cov, (((B m).card : ℝ) - 1) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4)) =
        ((N : ℝ) - K_cov) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4) := by
      simp_rw [← Finset.sum_mul, Finset.sum_sub_distrib, h_card_sum]
      simp [Finset.sum_const, nsmul_eq_mul]
    linarith
  have h_chain : ((N : ℝ) - K_cov) * ((2 ^ N : ℝ) / 4) * (A ^ 2 / 4) ≤
      (2 ^ N : ℝ) * N * (A ^ 2 / 64) :=
    h_total_lower.trans hErr
  have h_pos : 0 < (2 ^ N : ℝ) * A ^ 2 := by positivity
  have h_factor : ((N : ℝ) - K_cov) / 16 * ((2 ^ N : ℝ) * A ^ 2) ≤
      ((N : ℝ) / 64) * ((2 ^ N : ℝ) * A ^ 2) := by
    linarith
  have h_div : ((N : ℝ) - K_cov) / 16 ≤ (N : ℝ) / 64 :=
    le_of_mul_le_mul_right h_factor h_pos
  linarith

/-! ## R4: Shallow ReLU Collapse Boundary -/

/-- **R4 (3-Point Affine Floor on `[u₀, u₁]`)**: Any affine function `u ↦ α * u + β`
incurs `L^∞` error at least `(u₁ - u₀)^2 / 8` against `u^2` on `[u₀, u₁]`. -/
theorem affine_quadratic_interval_floor
    {u₀ u₁ α β : ℝ} (hu : u₀ ≤ u₁) :
    ∃ u ∈ Set.Icc u₀ u₁, (u₁ - u₀) ^ 2 / 8 ≤ |u ^ 2 - (α * u + β)| := by
  let um : ℝ := (u₀ + u₁) / 2
  have hu₀ : u₀ ∈ Set.Icc u₀ u₁ := ⟨le_rfl, hu⟩
  have hum : um ∈ Set.Icc u₀ u₁ := by
    dsimp [um]
    exact ⟨by linarith, by linarith⟩
  have hu₁ : u₁ ∈ Set.Icc u₀ u₁ := ⟨hu, le_rfl⟩
  by_contra h_all
  push Not at h_all
  have h0 := abs_lt.mp (h_all u₀ hu₀)
  have hm := abs_lt.mp (h_all um hum)
  have h1 := abs_lt.mp (h_all u₁ hu₁)
  have h_id : (u₀ ^ 2 - (α * u₀ + β)) - 2 * (um ^ 2 - (α * um + β)) +
      (u₁ ^ 2 - (α * u₁ + β)) = (u₁ - u₀) ^ 2 / 2 := by
    dsimp [um]
    ring
  linarith

/-- **R4 (Piecewise-Affine Partition Floor on `[0, 1]`)**: Any piecewise-affine function on
`[0, 1]` with `W + 1` subintervals incurs `L^∞` error at least `1 / (8 * (W + 1)^2)` against `u^2`. -/
theorem piecewise_affine_quadratic_floor
    {W : ℕ} (pts : Fin (W + 2) → ℝ)
    (h0 : pts 0 = 0) (h1 : pts ⟨W + 1, by omega⟩ = 1)
    (hMono : Monotone pts)
    (g : ℝ → ℝ)
    (hAff : ∀ i : Fin (W + 1), ∃ α β : ℝ,
      ∀ u ∈ Set.Icc (pts i.castSucc) (pts i.succ), g u = α * u + β) :
    ∃ u ∈ Set.Icc (0 : ℝ) 1, 1 / (8 * ((W : ℝ) + 1) ^ 2) ≤ |u ^ 2 - g u| := by
  have h_tele_gen : ∀ (n : ℕ) (f : Fin (n + 1) → ℝ),
      (∑ i : Fin n, (f i.succ - f i.castSucc)) = f ⟨n, by omega⟩ - f 0 := by
    intro n
    induction n with
    | zero =>
      intro f
      simp
    | succ n ih =>
      intro f
      rw [Fin.sum_univ_castSucc]
      have ih_val := ih (fun i => f i.castSucc)
      simp only [Fin.succ_castSucc] at ih_val ⊢
      have h_sub : f (Fin.castSucc ⟨n, by omega⟩) = f ⟨n, by omega⟩ := rfl
      have h_zero : f (Fin.castSucc 0) = f 0 := rfl
      have h_succ_last : f (Fin.last n).succ = f ⟨n + 1, by omega⟩ := rfl
      have h_cast_last : f (Fin.last n).castSucc = f ⟨n, by omega⟩ := rfl
      rw [ih_val, h_sub, h_zero, h_succ_last, h_cast_last]
      ring
  have h_tele : (∑ i : Fin (W + 1), (pts i.succ - pts i.castSucc)) = 1 := by
    have ht := h_tele_gen (W + 1) pts
    rw [h0, h1, sub_zero] at ht
    exact ht
  have h_exists_long : ∃ i : Fin (W + 1), 1 / ((W : ℝ) + 1) ≤ pts i.succ - pts i.castSucc := by
    by_contra h_all
    push Not at h_all
    have h_sum_lt : (∑ i : Fin (W + 1), (pts i.succ - pts i.castSucc)) <
        ∑ _i : Fin (W + 1), (1 / ((W : ℝ) + 1)) :=
      Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty (fun i _ => h_all i)
    have h_rhs : (∑ _i : Fin (W + 1), (1 / ((W : ℝ) + 1))) = 1 := by
      simp only [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
      have hW : (W : ℝ) + 1 ≠ 0 := by positivity
      push_cast
      field_simp [hW]
    linarith
  rcases h_exists_long with ⟨i, hi_len⟩
  rcases hAff i with ⟨α, β, hgi⟩
  have h_le : pts i.castSucc ≤ pts i.succ := hMono (Fin.castSucc_le_succ i)
  rcases affine_quadratic_interval_floor (α := α) (β := β) h_le with ⟨u, hu_mem, hu_err⟩
  have hu_01 : u ∈ Set.Icc (0 : ℝ) 1 := by
    have h0_le : 0 ≤ pts i.castSucc := by
      rw [← h0]
      exact hMono (Fin.zero_le _)
    have h1_ge : pts i.succ ≤ 1 := by
      rw [← h1]
      exact hMono (Fin.le_last _)
    exact ⟨h0_le.trans hu_mem.1, hu_mem.2.trans h1_ge⟩
  have h_sq_le : 1 / (8 * ((W : ℝ) + 1) ^ 2) ≤ (pts i.succ - pts i.castSucc) ^ 2 / 8 := by
    have hW_pos : 0 < (W : ℝ) + 1 := by positivity
    have h_pow : (1 / ((W : ℝ) + 1)) ^ 2 ≤ (pts i.succ - pts i.castSucc) ^ 2 :=
      pow_le_pow_left₀ (by positivity) hi_len 2
    have h_eq : (1 / ((W : ℝ) + 1)) ^ 2 / 8 = 1 / (8 * ((W : ℝ) + 1) ^ 2) := by
      field_simp
    linarith
  refine ⟨u, hu_01, ?_⟩
  rw [hgi u hu_mem]
  exact h_sq_le.trans hu_err

private lemma exists_sorted_partition :
    ∀ (n : ℕ) (k : Fin n → ℝ), (∀ j : Fin n, k j ∈ Set.Icc (0 : ℝ) 1) →
      ∃ pts : Fin (n + 2) → ℝ,
        pts 0 = 0 ∧ pts ⟨n + 1, by omega⟩ = 1 ∧ Monotone pts ∧
        ∀ j : Fin n, ∃ idx : Fin (n + 2), pts idx = k j
  | 0, k, _ => by
    refine ⟨fun i => if i.1 = 0 then 0 else 1, rfl, rfl, ?_, fun j => Fin.elim0 j⟩
    intro i j hij
    have hij' : i.1 ≤ j.1 := hij
    by_cases hi : i.1 = 0 <;> by_cases hj : j.1 = 0
    · simp [hi, hj]
    · simp [hi, hj]
    · omega
    · simp [hi, hj]
  | n + 1, k, hk => by
    obtain ⟨pts, h0, h1, hMono, hcov⟩ :=
      exists_sorted_partition n (fun j => k j.castSucc) (fun j => hk j.castSucc)
    let x : ℝ := k ⟨n, by omega⟩
    have hx_mem : x ∈ Set.Icc (0 : ℝ) 1 := hk ⟨n, by omega⟩
    let S : Finset (Fin (n + 2)) := Finset.univ.filter (fun i => pts i ≤ x)
    have h0S : (0 : Fin (n + 2)) ∈ S :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ 0, by rw [h0]; exact hx_mem.1⟩
    let p : Fin (n + 2) := S.max' ⟨0, h0S⟩
    have hp_mem : p ∈ S := Finset.max'_mem S ⟨0, h0S⟩
    have hp_le : pts p ≤ x := (Finset.mem_filter.mp hp_mem).2
    have h_gt_p : ∀ i : Fin (n + 2), p < i → x < pts i := by
      intro i hi
      by_contra hle
      push Not at hle
      have hiS : i ∈ S := Finset.mem_filter.mpr ⟨Finset.mem_univ i, hle⟩
      have hip : i ≤ p := Finset.le_max' S i hiS
      exact (lt_irrefl p) (hi.trans_le hip)
    let pts' : Fin (n + 3) → ℝ := fun q =>
      if hq : q.1 ≤ p.1 then pts ⟨q.1, by omega⟩
      else if q.1 = p.1 + 1 then x
      else pts ⟨q.1 - 1, by omega⟩
    refine ⟨pts', ?_, ?_, ?_, ?_⟩
    · have hp0 : (0 : ℕ) ≤ p.1 := Nat.zero_le _
      simpa [pts', hp0] using h0
    · by_cases hp_last : p.1 = n + 1
      · have hp_eq : p = ⟨n + 1, by omega⟩ := Fin.ext hp_last
        have hx_eq : x = 1 := by
          have : 1 ≤ x := by
            rw [← h1, ← hp_eq]
            exact hp_le
          linarith [hx_mem.2]
        dsimp [pts']
        split_ifs
        · omega
        · exact hx_eq
        · omega
      · dsimp [pts']
        split_ifs
        · omega
        · omega
        · exact h1
    · intro q₁ q₂ hq
      have hq' : q₁.1 ≤ q₂.1 := hq
      by_cases h1p : q₁.1 ≤ p.1 <;> by_cases h2p : q₂.1 ≤ p.1
      · simp only [pts', h1p, h2p, ↓reduceDIte]
        exact hMono (show (⟨q₁.1, by omega⟩ : Fin (n + 2)) ≤ ⟨q₂.1, by omega⟩ from hq')
      · simp only [pts', h1p, h2p, ↓reduceDIte]
        by_cases h2eq : q₂.1 = p.1 + 1
        · simp only [h2eq, ↓reduceIte]
          have hq1_le : (⟨q₁.1, by omega⟩ : Fin (n + 2)) ≤ p := h1p
          exact (hMono hq1_le).trans hp_le
        · simp only [h2eq, ↓reduceIte]
          have hq1_le : (⟨q₁.1, by omega⟩ : Fin (n + 2)) ≤ ⟨q₂.1 - 1, by omega⟩ := by
            change q₁.1 ≤ q₂.1 - 1
            omega
          exact hMono hq1_le
      · omega
      · simp only [pts', h1p, h2p, ↓reduceDIte]
        by_cases h1eq : q₁.1 = p.1 + 1 <;> by_cases h2eq : q₂.1 = p.1 + 1
        · simp [h1eq, h2eq]
        · simp only [h1eq, h2eq, ↓reduceIte]
          have hp_lt : p < (⟨q₂.1 - 1, by omega⟩ : Fin (n + 2)) := by
            change p.1 < q₂.1 - 1
            omega
          exact (h_gt_p ⟨q₂.1 - 1, by omega⟩ hp_lt).le
        · omega
        · simp only [h1eq, h2eq, ↓reduceIte]
          have h_sub_le : (⟨q₁.1 - 1, by omega⟩ : Fin (n + 2)) ≤ ⟨q₂.1 - 1, by omega⟩ := by
            change q₁.1 - 1 ≤ q₂.1 - 1
            omega
          exact hMono h_sub_le
    · intro j
      by_cases hjn : j.1 = n
      · refine ⟨⟨p.1 + 1, by omega⟩, ?_⟩
        have h_not_le : ¬ (p.1 + 1 ≤ p.1) := by omega
        simp only [pts', h_not_le, ↓reduceDIte, ↓reduceIte]
        have hj_eq : j = ⟨n, by omega⟩ := Fin.ext hjn
        rw [hj_eq]
      · have hj_lt : j.1 < n := by omega
        obtain ⟨idx, hidx⟩ := hcov ⟨j.1, hj_lt⟩
        have hj_eq : (⟨j.1, hj_lt⟩ : Fin n).castSucc = j := Fin.ext rfl
        rw [hj_eq] at hidx
        by_cases hidx_p : idx.1 ≤ p.1
        · refine ⟨⟨idx.1, by omega⟩, ?_⟩
          simp only [pts', hidx_p, ↓reduceDIte]
          exact hidx
        · refine ⟨⟨idx.1 + 1, by omega⟩, ?_⟩
          have h_not_le : ¬ (idx.1 + 1 ≤ p.1) := by omega
          have h_ne : idx.1 + 1 ≠ p.1 + 1 := by omega
          simp only [pts', h_not_le, h_ne, ↓reduceDIte, ↓reduceIte, Nat.add_sub_cancel]
          exact hidx

/-- **R4 (Depth-2 Width-`W` ReLU Approximation Floor on `[0, 1]`)**: Any depth-2 width-`W`
ReLU network `g u = c₀ + ∑ j : Fin W, a j * max 0 (w j * u + b j)` induces a sorted
partition of `[0, 1]` into at most `W + 1` affine subintervals and hence incurs `L^∞` error
at least `1 / (8 * (W + 1)^2)` against `u^2` on `[0, 1]`. -/
theorem relu_depth2_quadratic_floor
    {W : ℕ} (c₀ : ℝ) (a w b : Fin W → ℝ) :
    let g : ℝ → ℝ := fun u => c₀ + ∑ j : Fin W, a j * max 0 (w j * u + b j)
    ∃ u ∈ Set.Icc (0 : ℝ) 1, 1 / (8 * ((W : ℝ) + 1) ^ 2) ≤ |u ^ 2 - g u| := by
  classical
  intro g
  let bp : Fin W → ℝ := fun j =>
    if w j ≠ 0 ∧ (-b j / w j) ∈ Set.Icc (0 : ℝ) 1 then -b j / w j else 0
  have hbp : ∀ j : Fin W, bp j ∈ Set.Icc (0 : ℝ) 1 := by
    intro j
    dsimp [bp]
    split_ifs with h
    · exact h.2
    · exact ⟨le_rfl, zero_le_one⟩
  obtain ⟨pts, h0, h1, hMono, hcov⟩ := exists_sorted_partition W bp hbp
  have hAff : ∀ i : Fin (W + 1), ∃ α β : ℝ,
      ∀ u ∈ Set.Icc (pts i.castSucc) (pts i.succ), g u = α * u + β := by
    intro i
    have h_term : ∀ j : Fin W, ∃ αj βj : ℝ,
        ∀ u ∈ Set.Icc (pts i.castSucc) (pts i.succ),
          a j * max 0 (w j * u + b j) = αj * u + βj := by
      intro j
      have h_const_sign :
          (∀ u ∈ Set.Icc (pts i.castSucc) (pts i.succ), 0 ≤ w j * u + b j) ∨
          (∀ u ∈ Set.Icc (pts i.castSucc) (pts i.succ), w j * u + b j ≤ 0) := by
        by_cases hw : w j = 0
        · rcases le_total 0 (b j) with hb | hb
          · left; intro u _; simp [hw, hb]
          · right; intro u _; simp [hw, hb]
        · let r : ℝ := -b j / w j
          have h_no_split : r ≤ pts i.castSucc ∨ pts i.succ ≤ r := by
            by_contra h_strict
            push Not at h_strict
            have h0_le : 0 ≤ pts i.castSucc := by
              rw [← h0]; exact hMono (Fin.zero_le _)
            have h1_ge : pts i.succ ≤ 1 := by
              rw [← h1]; exact hMono (Fin.le_last _)
            have hr_mem : r ∈ Set.Icc (0 : ℝ) 1 :=
              ⟨h0_le.trans h_strict.1.le, h_strict.2.le.trans h1_ge⟩
            have hbp_eq : bp j = r := by
              dsimp [bp]
              split_ifs with h
              · rfl
              · exfalso; exact h ⟨hw, hr_mem⟩
            obtain ⟨idx, hidx⟩ := hcov j
            rw [hbp_eq] at hidx
            by_cases h_idx : idx.1 ≤ i.1
            · have : pts idx ≤ pts i.castSucc := hMono h_idx
              linarith
            · have h_idx' : i.succ.1 ≤ idx.1 := by
                change i.1 + 1 ≤ idx.1
                omega
              have : pts i.succ ≤ pts idx := hMono h_idx'
              linarith
          have h_lin_eq : ∀ u : ℝ, w j * u + b j = w j * (u - r) := by
            intro u
            dsimp [r]
            field_simp [hw]
            ring
          rcases h_no_split with hr_left | hr_right
          · rcases le_total 0 (w j) with hw_pos | hw_neg
            · left; intro u hu; rw [h_lin_eq u]
              exact mul_nonneg hw_pos (sub_nonneg.mpr (hr_left.trans hu.1))
            · right; intro u hu; rw [h_lin_eq u]
              exact mul_nonpos_of_nonpos_of_nonneg hw_neg (sub_nonneg.mpr (hr_left.trans hu.1))
          · rcases le_total 0 (w j) with hw_pos | hw_neg
            · right; intro u hu; rw [h_lin_eq u]
              exact mul_nonpos_of_nonneg_of_nonpos hw_pos (sub_nonpos.mpr (hu.2.trans hr_right))
            · left; intro u hu; rw [h_lin_eq u]
              exact mul_nonneg_of_nonpos_of_nonpos hw_neg (sub_nonpos.mpr (hu.2.trans hr_right))
      rcases h_const_sign with h_nonneg | h_nonpos
      · refine ⟨a j * w j, a j * b j, fun u hu => ?_⟩
        rw [max_eq_right (h_nonneg u hu)]
        ring
      · refine ⟨0, 0, fun u hu => ?_⟩
        rw [max_eq_left (h_nonpos u hu)]
        ring
    let αj : Fin W → ℝ := fun j => Classical.choose (h_term j)
    let βj : Fin W → ℝ := fun j => Classical.choose (Classical.choose_spec (h_term j))
    refine ⟨∑ j : Fin W, αj j, c₀ + ∑ j : Fin W, βj j, fun u hu => ?_⟩
    dsimp [g]
    have h_sum_eq : (∑ j : Fin W, a j * max 0 (w j * u + b j)) =
        ∑ j : Fin W, (αj j * u + βj j) :=
      Finset.sum_congr rfl fun j _ => Classical.choose_spec (Classical.choose_spec (h_term j)) u hu
    rw [h_sum_eq, Finset.sum_add_distrib, ← Finset.sum_mul]
    ring
  exact piecewise_affine_quadratic_floor pts h0 h1 hMono g hAff

#print axioms linear_readout_irreducible_error
#print axioms monomial_linearIndependent_Icc
#print axioms monomial_empirical_L2_irreducible_error
#print axioms lipschitz_radius_inflation
#print axioms lipschitz_covering_lower_bound
#print axioms affine_quadratic_interval_floor
#print axioms piecewise_affine_quadratic_floor
#print axioms relu_depth2_quadratic_floor

end MathLab.Projects.LateQueryFrontier.CompInflation
