import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.LinearAlgebra.Prod
import Mathlib.LinearAlgebra.Isomorphisms
import Mathlib.LinearAlgebra.Projection
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.FDeriv.Congr
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

/-!
# Observational vs. Interventional Causal Cache Premium (Conjecture 3)

This module formalizes the core load-bearing algebraic, rank/kernel, bi-Lipschitz,
backdoor subspace-decomposition, local Jacobian obstruction, and active-probe
theorems for the Causal Cache Premium `Δ_d_causal := d_{obs ∪ do} - d_obs` in
late-query caching over Structural Causal Models (SCMs).
-/

namespace MathLab.LateQueryFrontier.CausalPremium

open scoped Topology

/-! ## Section 1: General Linear Kernel / Rank Characterization -/

section LinearTheory

variable {K X W_obs W_do : Type*}
variable [Field K]
variable [AddCommGroup X] [Module K X]
variable [AddCommGroup W_obs] [Module K W_obs]
variable [AddCommGroup W_do] [Module K W_do]

/-- The joint linear query map `(T_obs, T_do) : X →ₗ[K] W_obs × W_do`. -/
def jointMap (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    X →ₗ[K] W_obs × W_do :=
  T_obs.prod T_do

/-- Minimal sufficient linear cache dimension for observational queries. -/
noncomputable def obsDim (T_obs : X →ₗ[K] W_obs) : ℕ :=
  Module.finrank K (LinearMap.range T_obs)

/-- Minimal sufficient linear cache dimension for interventional queries. -/
noncomputable def doDim (T_do : X →ₗ[K] W_do) : ℕ :=
  Module.finrank K (LinearMap.range T_do)

/-- Minimal sufficient linear cache dimension for joint `(obs, do)` queries. -/
noncomputable def jointDim (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) : ℕ :=
  Module.finrank K (LinearMap.range (jointMap T_obs T_do))

/-- The Causal Cache Premium `Δ_d_causal := d_{obs ∪ do} - d_obs`. -/
noncomputable def causalPremium (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) : ℕ :=
  jointDim T_obs T_do - obsDim T_obs

@[simp]
theorem ker_jointMap (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    LinearMap.ker (jointMap T_obs T_do) = LinearMap.ker T_obs ⊓ LinearMap.ker T_do :=
  LinearMap.ker_prod T_obs T_do

theorem ker_le_iff_exists_factor (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    LinearMap.ker T_obs ≤ LinearMap.ker T_do ↔ ∃ L : W_obs →ₗ[K] W_do, L.comp T_obs = T_do := by
  constructor
  · intro h_ker
    obtain ⟨q, hq⟩ := Submodule.exists_isCompl (LinearMap.range T_obs)
    let proj : W_obs →ₗ[K] LinearMap.range T_obs :=
      (LinearMap.range T_obs).projectionOnto q hq
    let q_lift : X ⧸ LinearMap.ker T_obs →ₗ[K] W_do :=
      (LinearMap.ker T_obs).liftQ T_do h_ker
    let L : W_obs →ₗ[K] W_do :=
      q_lift.comp (T_obs.quotKerEquivRange.symm.toLinearMap.comp proj)
    refine ⟨L, ?_⟩
    ext x
    have h_mem : T_obs x ∈ LinearMap.range T_obs := LinearMap.mem_range_self T_obs x
    have h_proj : proj (T_obs x) = ⟨T_obs x, h_mem⟩ :=
      Submodule.projectionOnto_apply_of_mem_left hq h_mem
    simp only [L, LinearMap.comp_apply, LinearEquiv.coe_toLinearMap, h_proj,
      LinearMap.quotKerEquivRange_symm_apply_image, q_lift, Submodule.mkQ_apply,
      Submodule.liftQ_apply]
  · rintro ⟨L, rfl⟩
    exact LinearMap.ker_le_ker_comp T_obs L

theorem ker_le_iff_exists_set_factor (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    LinearMap.ker T_obs ≤ LinearMap.ker T_do ↔ ∃ g : W_obs → W_do, ∀ x, T_do x = g (T_obs x) := by
  constructor
  · intro h_ker
    obtain ⟨L, hL⟩ := (ker_le_iff_exists_factor T_obs T_do).mp h_ker
    exact ⟨L, fun x => by simpa [LinearMap.comp_apply] using (LinearMap.congr_fun hL x).symm⟩
  · rintro ⟨g, hg⟩ x hx
    rw [LinearMap.mem_ker] at hx ⊢
    calc
      T_do x = g (T_obs x) := hg x
      _ = g 0 := by rw [hx]
      _ = g (T_obs 0) := by rw [map_zero]
      _ = T_do 0 := (hg 0).symm
      _ = 0 := map_zero T_do

section FiniteDim

variable [FiniteDimensional K X]

theorem obsDim_le_jointDim (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    obsDim T_obs ≤ jointDim T_obs T_do := by
  have h_obs := LinearMap.finrank_range_add_finrank_ker T_obs
  have h_joint := LinearMap.finrank_range_add_finrank_ker (jointMap T_obs T_do)
  have h_ker :
      Module.finrank K (LinearMap.ker (jointMap T_obs T_do)) ≤
        Module.finrank K (LinearMap.ker T_obs) := by
    rw [ker_jointMap]
    exact Submodule.finrank_mono inf_le_left
  dsimp only [obsDim, jointDim]
  omega

theorem doDim_le_jointDim (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    doDim T_do ≤ jointDim T_obs T_do := by
  have h_do := LinearMap.finrank_range_add_finrank_ker T_do
  have h_joint := LinearMap.finrank_range_add_finrank_ker (jointMap T_obs T_do)
  have h_ker :
      Module.finrank K (LinearMap.ker (jointMap T_obs T_do)) ≤
        Module.finrank K (LinearMap.ker T_do) := by
    rw [ker_jointMap]
    exact Submodule.finrank_mono inf_le_right
  dsimp only [doDim, jointDim]
  omega

theorem obsDim_add_causalPremium (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    obsDim T_obs + causalPremium T_obs T_do = jointDim T_obs T_do := by
  dsimp only [causalPremium]
  have h := obsDim_le_jointDim T_obs T_do
  omega

theorem causalPremium_eq_finrank_ker_sub (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    causalPremium T_obs T_do =
      Module.finrank K (LinearMap.ker T_obs) -
        Module.finrank K (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) := by
  have h_obs := LinearMap.finrank_range_add_finrank_ker T_obs
  have h_joint := LinearMap.finrank_range_add_finrank_ker (jointMap T_obs T_do)
  rw [ker_jointMap] at h_joint
  dsimp only [causalPremium, jointDim, obsDim]
  omega

theorem causalPremium_eq_finrank_map_ker (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    causalPremium T_obs T_do = Module.finrank K ((LinearMap.ker T_obs).map T_do) := by
  let f := T_do.domRestrict (LinearMap.ker T_obs)
  have h_rn := LinearMap.finrank_range_add_finrank_ker f
  have h_range : LinearMap.range f = (LinearMap.ker T_obs).map T_do :=
    LinearMap.range_domRestrict (LinearMap.ker T_obs) T_do
  let e : LinearMap.ker f ≃ₗ[K] (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) :=
    { toFun := fun v => ⟨v.1.1, v.1.2, v.2⟩
      invFun := fun v => ⟨⟨v.1, v.2.1⟩, v.2.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
  have h_ker_eq :
      Module.finrank K (LinearMap.ker f) =
        Module.finrank K (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) :=
    e.finrank_eq
  have h_eq :
      Module.finrank K ((LinearMap.ker T_obs).map T_do) +
          Module.finrank K (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) =
        Module.finrank K (LinearMap.ker T_obs) := by
    rw [← h_range, ← h_ker_eq]
    exact h_rn
  have h_sub := causalPremium_eq_finrank_ker_sub T_obs T_do
  omega

theorem causalPremium_le_doDim (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    causalPremium T_obs T_do ≤ doDim T_do := by
  rw [causalPremium_eq_finrank_map_ker]
  dsimp only [doDim]
  have h_le : (LinearMap.ker T_obs).map T_do ≤ LinearMap.range T_do := by
    intro y hy
    obtain ⟨x, _, rfl⟩ := hy
    exact LinearMap.mem_range_self T_do x
  exact Submodule.finrank_mono h_le

theorem causalPremium_eq_zero_iff_ker_le (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    causalPremium T_obs T_do = 0 ↔ LinearMap.ker T_obs ≤ LinearMap.ker T_do := by
  rw [causalPremium_eq_finrank_ker_sub]
  have h_le :
      Module.finrank K (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) ≤
        Module.finrank K (LinearMap.ker T_obs) :=
    Submodule.finrank_mono inf_le_left
  constructor
  · intro h_zero
    have h_eq_dim :
        Module.finrank K (LinearMap.ker T_obs) ≤
          Module.finrank K (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) := by
      omega
    have h_sub_eq :
        (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) = LinearMap.ker T_obs :=
      Submodule.eq_of_le_of_finrank_le inf_le_left h_eq_dim
    intro x hx
    have hx' : x ∈ (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) := by
      rw [h_sub_eq]
      exact hx
    exact hx'.2
  · intro h_ker
    have h_sub_eq :
        (LinearMap.ker T_obs ⊓ LinearMap.ker T_do : Submodule K X) = LinearMap.ker T_obs :=
      inf_eq_left.mpr h_ker
    rw [h_sub_eq]
    omega

theorem causalPremium_eq_zero_iff_exists_factor (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    causalPremium T_obs T_do = 0 ↔ ∃ L : W_obs →ₗ[K] W_do, L.comp T_obs = T_do :=
  (causalPremium_eq_zero_iff_ker_le T_obs T_do).trans (ker_le_iff_exists_factor T_obs T_do)

theorem causalPremium_pos_iff_exists_witness (T_obs : X →ₗ[K] W_obs) (T_do : X →ₗ[K] W_do) :
    0 < causalPremium T_obs T_do ↔ ∃ x : X, T_obs x = 0 ∧ T_do x ≠ 0 := by
  rw [Nat.pos_iff_ne_zero, ne_eq, causalPremium_eq_zero_iff_ker_le,
    IsConcreteLE.not_le_iff_exists]
  simp only [LinearMap.mem_ker]

end FiniteDim

end LinearTheory

/-! ## Section 2: Sharp Canonical 2-Node SCM Example -/

section CanonicalExample

/-- Observational linear query `(a, z) ↦ α * a + β * z`. -/
def canonObsMap (α β : ℝ) : ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := α * p.1 + β * p.2
  map_add' p q := by simp only [Prod.fst_add, Prod.snd_add]; ring
  map_smul' c p := by simp; ring

/-- Interventional linear query `(a, z) ↦ β * z`. -/
def canonDoMap (β : ℝ) : ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := β * p.2
  map_add' p q := by simp only [Prod.snd_add]; ring
  map_smul' c p := by simp; ring

@[simp]
theorem canonObsMap_apply (α β a z : ℝ) : canonObsMap α β (a, z) = α * a + β * z := rfl

@[simp]
theorem canonDoMap_apply (β a z : ℝ) : canonDoMap β (a, z) = β * z := rfl

/-- Explicit linear equivalence for the canonical 2D joint cache when `α ≠ 0` and `β ≠ 0`. -/
noncomputable def canonJointEquiv {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) :
    (ℝ × ℝ) ≃ₗ[ℝ] (ℝ × ℝ) where
  toFun p := (α * p.1 + β * p.2, β * p.2)
  invFun q := ((q.1 - q.2) / α, q.2 / β)
  left_inv p := by
    ext <;> dsimp
    · field_simp [hα]
      ring
    · field_simp [hβ]
  right_inv q := by
    ext <;> dsimp
    · field_simp [hα, hβ]
      ring
    · field_simp [hβ]
  map_add' p₁ p₂ := by
    ext <;> dsimp <;> ring
  map_smul' c p := by
    ext <;> dsimp <;> ring

theorem canonJointMap_bijective_iff (α β : ℝ) :
    Function.Bijective (jointMap (canonObsMap α β) (canonDoMap β)) ↔ α ≠ 0 ∧ β ≠ 0 := by
  constructor
  · intro h_bij
    have h_inj := h_bij.injective
    constructor
    · intro hα
      have h_eq :
          jointMap (canonObsMap α β) (canonDoMap β) (1, 0) =
            jointMap (canonObsMap α β) (canonDoMap β) (0, 0) := by
        simp [jointMap, hα]
      have hp := h_inj h_eq
      have h10 : (1 : ℝ) = 0 := congrArg Prod.fst hp
      norm_num at h10
    · intro hβ
      have h_eq :
          jointMap (canonObsMap α β) (canonDoMap β) (0, 1) =
            jointMap (canonObsMap α β) (canonDoMap β) (0, 0) := by
        simp [jointMap, hβ]
      have hp := h_inj h_eq
      have h10 : (1 : ℝ) = 0 := congrArg Prod.snd hp
      norm_num at h10
  · rintro ⟨hα, hβ⟩
    exact (canonJointEquiv hα hβ).bijective

/-- Quantitative `ℓ₁` bi-Lipschitz equivalence for the canonical 2D joint cache map. -/
theorem canonJointMap_bilipschitz {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) :
    ∃ c C : ℝ, 0 < c ∧ 0 < C ∧
      ∀ p₁ p₂ : ℝ × ℝ,
        c * (|p₁.1 - p₂.1| + |p₁.2 - p₂.2|) ≤
          |(α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)| + |β * p₁.2 - β * p₂.2| ∧
        |(α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)| + |β * p₁.2 - β * p₂.2| ≤
          C * (|p₁.1 - p₂.1| + |p₁.2 - p₂.2|) := by
  let c : ℝ := min |α| |β| / 2
  let C : ℝ := max |α| (2 * |β|)
  have ha_pos : 0 < |α| := abs_pos.mpr hα
  have hb_pos : 0 < |β| := abs_pos.mpr hβ
  have hc_pos : 0 < c := by
    dsimp only [c]
    have hmin : 0 < min |α| |β| := lt_min ha_pos hb_pos
    linarith
  have hC_pos : 0 < C := by
    dsimp only [C]
    exact lt_of_lt_of_le ha_pos (le_max_left _ _)
  refine ⟨c, C, hc_pos, hC_pos, ?_⟩
  intro p₁ p₂
  set da : ℝ := p₁.1 - p₂.1
  set dz : ℝ := p₁.2 - p₂.2
  have h_eq1 : (α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2) = α * da + β * dz := by
    dsimp only [da, dz]; ring
  have h_eq2 : β * p₁.2 - β * p₂.2 = β * dz := by
    dsimp only [dz]; ring
  rw [h_eq1, h_eq2]
  constructor
  · have h_tri : |α| * |da| ≤ |α * da + β * dz| + |β * dz| := by
      have h1 : |α * da| = |(α * da + β * dz) + (-(β * dz))| := by
        congr 1
        ring
      calc
        |α| * |da| = |α * da| := (abs_mul α da).symm
        _ = |(α * da + β * dz) + (-(β * dz))| := h1
        _ ≤ |α * da + β * dz| + |-(β * dz)| := abs_add_le _ _
        _ = |α * da + β * dz| + |β * dz| := by rw [abs_neg]
    have h_z : |β| * |dz| ≤ |α * da + β * dz| + |β * dz| := by
      rw [← abs_mul]
      linarith [abs_nonneg (α * da + β * dz)]
    have hc_le_a : c ≤ |α| / 2 := by
      dsimp only [c]
      have : min |α| |β| ≤ |α| := min_le_left _ _
      linarith
    have hc_le_b : c ≤ |β| / 2 := by
      dsimp only [c]
      have : min |α| |β| ≤ |β| := min_le_right _ _
      linarith
    have h_sum : c * (|da| + |dz|) ≤ (|α| / 2) * |da| + (|β| / 2) * |dz| := by
      nlinarith [abs_nonneg da, abs_nonneg dz]
    linarith
  · have h_up1 : |α * da + β * dz| ≤ |α| * |da| + |β| * |dz| := by
      calc
        |α * da + β * dz| ≤ |α * da| + |β * dz| := abs_add_le _ _
        _ = |α| * |da| + |β| * |dz| := by rw [abs_mul, abs_mul]
    have h_up2 : |β * dz| = |β| * |dz| := abs_mul β dz
    have ha_le_C : |α| ≤ C := le_max_left _ _
    have hb_le_C : 2 * |β| ≤ C := le_max_right _ _
    nlinarith [abs_nonneg da, abs_nonneg dz]

/-- Quantitative Euclidean quadratic (`ℓ₂²`) bi-Lipschitz equivalence for `canonJointMap`. -/
theorem canonJointMap_sq_norm_bilipschitz {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) :
    ∃ c_sq C_sq : ℝ, 0 < c_sq ∧ 0 < C_sq ∧
      ∀ p₁ p₂ : ℝ × ℝ,
        c_sq * ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2) ≤
          ((α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)) ^ 2 + (β * p₁.2 - β * p₂.2) ^ 2 ∧
        ((α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)) ^ 2 + (β * p₁.2 - β * p₂.2) ^ 2 ≤
          C_sq * ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2) := by
  let denom : ℝ := 2 / α ^ 2 + 1 / β ^ 2
  let c_sq : ℝ := denom⁻¹
  let C_sq : ℝ := max (2 * α ^ 2) (3 * β ^ 2)
  have hα2 : 0 < α ^ 2 := sq_pos_of_ne_zero hα
  have hβ2 : 0 < β ^ 2 := sq_pos_of_ne_zero hβ
  have hdenom_pos : 0 < denom := by
    dsimp only [denom]
    positivity
  have hc_pos : 0 < c_sq := inv_pos.mpr hdenom_pos
  have hC_pos : 0 < C_sq := by
    dsimp only [C_sq]
    exact lt_of_lt_of_le (by positivity) (le_max_left _ _)
  refine ⟨c_sq, C_sq, hc_pos, hC_pos, ?_⟩
  intro p₁ p₂
  set da : ℝ := p₁.1 - p₂.1
  set dz : ℝ := p₁.2 - p₂.2
  set u₁ : ℝ := (α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)
  set u₂ : ℝ := β * p₁.2 - β * p₂.2
  have hu₁ : u₁ = α * da + β * dz := by dsimp only [u₁, da, dz]; ring
  have hu₂ : u₂ = β * dz := by dsimp only [u₂, dz]; ring
  constructor
  · have hda : da = (u₁ - u₂) / α := by
      rw [hu₁, hu₂]
      field_simp [hα]
      ring
    have hdz : dz = u₂ / β := by
      rw [hu₂]
      field_simp [hβ]
    have h_quad : (u₁ - u₂) ^ 2 ≤ 2 * (u₁ ^ 2 + u₂ ^ 2) := by
      have h_id : 2 * (u₁ ^ 2 + u₂ ^ 2) - (u₁ - u₂) ^ 2 = (u₁ + u₂) ^ 2 := by ring
      nlinarith [sq_nonneg (u₁ + u₂)]
    have h_inv_ineq : da ^ 2 + dz ^ 2 ≤ denom * (u₁ ^ 2 + u₂ ^ 2) := by
      rw [hda, hdz, div_pow, div_pow]
      have h1 : (u₁ - u₂) ^ 2 / α ^ 2 ≤ 2 * (u₁ ^ 2 + u₂ ^ 2) / α ^ 2 :=
        div_le_div_of_nonneg_right h_quad (le_of_lt hα2)
      have h2 : u₂ ^ 2 / β ^ 2 ≤ (u₁ ^ 2 + u₂ ^ 2) / β ^ 2 :=
        div_le_div_of_nonneg_right (by linarith [sq_nonneg u₁]) (le_of_lt hβ2)
      have h_sum_eq :
          2 * (u₁ ^ 2 + u₂ ^ 2) / α ^ 2 + (u₁ ^ 2 + u₂ ^ 2) / β ^ 2 = denom * (u₁ ^ 2 + u₂ ^ 2) := by
        dsimp only [denom]
        ring
      linarith
    have h_mul : c_sq * (da ^ 2 + dz ^ 2) ≤ c_sq * (denom * (u₁ ^ 2 + u₂ ^ 2)) :=
      mul_le_mul_of_nonneg_left h_inv_ineq (le_of_lt hc_pos)
    have hc_cancel : c_sq * (denom * (u₁ ^ 2 + u₂ ^ 2)) = u₁ ^ 2 + u₂ ^ 2 := by
      dsimp only [c_sq]
      rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hdenom_pos), one_mul]
    linarith
  · have h_quad_up : (α * da + β * dz) ^ 2 ≤ 2 * (α * da) ^ 2 + 2 * (β * dz) ^ 2 := by
      have h_id : 2 * (α * da) ^ 2 + 2 * (β * dz) ^ 2 - (α * da + β * dz) ^ 2 =
          (α * da - β * dz) ^ 2 := by ring
      nlinarith [sq_nonneg (α * da - β * dz)]
    have h_sum_up : u₁ ^ 2 + u₂ ^ 2 ≤ 2 * α ^ 2 * da ^ 2 + 3 * β ^ 2 * dz ^ 2 := by
      rw [hu₁, hu₂]
      nlinarith [h_quad_up]
    have ha_le : 2 * α ^ 2 ≤ C_sq := le_max_left _ _
    have hb_le : 3 * β ^ 2 ≤ C_sq := le_max_right _ _
    nlinarith [sq_nonneg da, sq_nonneg dz]

/-- Converse failure: if `α = 0` or `β = 0`, no positive anti-Lipschitz lower bound exists. -/
theorem canonJointMap_not_antilipschitz_of_eq_zero {α β : ℝ} (h : α = 0 ∨ β = 0) :
    ∀ c : ℝ, 0 < c → ∃ p₁ p₂ : ℝ × ℝ,
      |(α * p₁.1 + β * p₁.2) - (α * p₂.1 + β * p₂.2)| + |β * p₁.2 - β * p₂.2| <
        c * (|p₁.1 - p₂.1| + |p₁.2 - p₂.2|) := by
  intro c hc
  rcases h with rfl | rfl
  · refine ⟨(1, 0), (0, 0), ?_⟩
    simpa using hc
  · refine ⟨(0, 1), (0, 0), ?_⟩
    simpa using hc

theorem canonCausalPremium_eq_one_iff (α β : ℝ) :
    causalPremium (canonObsMap α β) (canonDoMap β) = 1 ↔ α ≠ 0 ∧ β ≠ 0 := by
  have h_obs_le : obsDim (canonObsMap α β) ≤ 1 := by
    dsimp only [obsDim]
    have h := Submodule.finrank_le (LinearMap.range (canonObsMap α β))
    simpa [CommSemiring.finrank_self] using h
  constructor
  · intro h_prem
    by_contra h_neg
    have h_or : α = 0 ∨ β = 0 := by
      by_cases hα : α = 0
      · exact Or.inl hα
      · exact Or.inr (by tauto)
    have h_ker : LinearMap.ker (canonObsMap α β) ≤ LinearMap.ker (canonDoMap β) := by
      rcases h_or with rfl | rfl
      · rintro ⟨a, z⟩ hx
        simp only [LinearMap.mem_ker, canonObsMap_apply, zero_mul, zero_add] at hx
        simp only [LinearMap.mem_ker, canonDoMap_apply, hx]
      · rintro ⟨a, z⟩ _
        simp only [LinearMap.mem_ker, canonDoMap_apply, zero_mul]
    have h_zero : causalPremium (canonObsMap α β) (canonDoMap β) = 0 :=
      (causalPremium_eq_zero_iff_ker_le _ _).mpr h_ker
    omega
  · rintro ⟨hα, hβ⟩
    have h_joint : jointDim (canonObsMap α β) (canonDoMap β) = 2 := by
      dsimp only [jointDim]
      have h_surj : Function.Surjective (jointMap (canonObsMap α β) (canonDoMap β)) :=
        (canonJointEquiv hα hβ).surjective
      have h_range : LinearMap.range (jointMap (canonObsMap α β) (canonDoMap β)) = ⊤ :=
        LinearMap.range_eq_top.mpr h_surj
      rw [h_range, finrank_top, Module.finrank_prod, CommSemiring.finrank_self]
    have h_obs_ge : 1 ≤ obsDim (canonObsMap α β) := by
      dsimp only [obsDim]
      have h_ne_bot : LinearMap.range (canonObsMap α β) ≠ ⊥ := by
        intro h_bot
        have h_mem : canonObsMap α β (1, 0) ∈ LinearMap.range (canonObsMap α β) :=
          LinearMap.mem_range_self _ _
        rw [h_bot, Submodule.mem_bot] at h_mem
        simp only [canonObsMap_apply, mul_one, mul_zero, add_zero] at h_mem
        exact hα h_mem
      exact Nat.pos_of_ne_zero (mt Submodule.finrank_eq_zero.mp h_ne_bot)
    have h_obs : obsDim (canonObsMap α β) = 1 := le_antisymm h_obs_le h_obs_ge
    dsimp only [causalPremium]
    rw [h_joint, h_obs]

theorem canonCausalPremium_eq_zero_iff (α β : ℝ) :
    causalPremium (canonObsMap α β) (canonDoMap β) = 0 ↔ α = 0 ∨ β = 0 := by
  constructor
  · intro h_zero
    by_contra h_neg
    have hα : α ≠ 0 := fun h => h_neg (Or.inl h)
    have hβ : β ≠ 0 := fun h => h_neg (Or.inr h)
    have h_one : causalPremium (canonObsMap α β) (canonDoMap β) = 1 :=
      (canonCausalPremium_eq_one_iff α β).mpr ⟨hα, hβ⟩
    omega
  · intro h_or
    have h_ker : LinearMap.ker (canonObsMap α β) ≤ LinearMap.ker (canonDoMap β) := by
      rcases h_or with rfl | rfl
      · rintro ⟨a, z⟩ hx
        simp only [LinearMap.mem_ker, canonObsMap_apply, zero_mul, zero_add] at hx
        simp only [LinearMap.mem_ker, canonDoMap_apply, hx]
      · rintro ⟨a, z⟩ _
        simp only [LinearMap.mem_ker, canonDoMap_apply, zero_mul]
    exact (causalPremium_eq_zero_iff_ker_le _ _).mpr h_ker

/-- Diagonal inclusion `z ↦ (z, z)` representing positivity violation `Var(η) = 0` (`A = Z`). -/
def canonDiagIncl : ℝ →ₗ[ℝ] ℝ × ℝ :=
  LinearMap.id.prod LinearMap.id

@[simp]
theorem canonDiagIncl_apply (z : ℝ) : canonDiagIncl z = (z, z) := rfl

/-- When `A = Z` (`Var(η) = 0`) and `α + β ≠ 0`, restricting to the 1D support collapses `Δ_d_causal` to `0`. -/
theorem canonDiag_causalPremium_eq_zero {α β : ℝ} (h_sum : α + β ≠ 0) :
    causalPremium ((canonObsMap α β).comp canonDiagIncl) ((canonDoMap β).comp canonDiagIncl) = 0 := by
  rw [causalPremium_eq_zero_iff_ker_le]
  intro z hz
  simp only [LinearMap.mem_ker, LinearMap.comp_apply, canonDiagIncl_apply,
    canonObsMap_apply, canonDoMap_apply] at hz ⊢
  have hz_factor : (α + β) * z = 0 := by linarith
  have hz0 : z = 0 := (mul_eq_zero.mp hz_factor).resolve_left h_sum
  simp [hz0]

/-- Unfaithful cancellation edge case on the diagonal (`α + β = 0`, `β ≠ 0`): `Δ_d_causal = 1` with `d_obs = 0`. -/
theorem canonDiag_unfaithful_premium_eq_one {α β : ℝ} (h_sum : α + β = 0) (hβ : β ≠ 0) :
    obsDim ((canonObsMap α β).comp canonDiagIncl) = 0 ∧
      causalPremium ((canonObsMap α β).comp canonDiagIncl) ((canonDoMap β).comp canonDiagIncl) = 1 := by
  have h_obs_zero : (canonObsMap α β).comp canonDiagIncl = 0 := by
    apply LinearMap.ext
    intro z
    simp only [LinearMap.comp_apply, canonDiagIncl_apply, canonObsMap_apply, LinearMap.zero_apply]
    linarith [show (α + β) * z = 0 by rw [h_sum, zero_mul]]
  have h_obs_dim : obsDim ((canonObsMap α β).comp canonDiagIncl) = 0 := by
    dsimp only [obsDim]
    rw [h_obs_zero, LinearMap.range_zero, finrank_bot]
  refine ⟨h_obs_dim, ?_⟩
  rw [causalPremium_eq_finrank_map_ker, h_obs_zero, LinearMap.ker_zero]
  have h_map_top :
      (⊤ : Submodule ℝ ℝ).map ((canonDoMap β).comp canonDiagIncl) =
        LinearMap.range ((canonDoMap β).comp canonDiagIncl) :=
    Submodule.map_top _
  have h_range_top : LinearMap.range ((canonDoMap β).comp canonDiagIncl) = ⊤ := by
    rw [LinearMap.range_eq_top]
    intro y
    refine ⟨y / β, ?_⟩
    simp only [LinearMap.comp_apply, canonDiagIncl_apply, canonDoMap_apply]
    field_simp [hβ]
  rw [h_map_top, h_range_top, finrank_top, CommSemiring.finrank_self]

end CanonicalExample

/-! ## Section 3: Multi-Dimensional Backdoor Subspace Decomposition -/

section BackdoorSubspace

section InnerProductDecomposition

variable {E W_obs W_do : Type*}
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable [AddCommGroup W_obs] [Module ℝ W_obs]
variable [AddCommGroup W_do] [Module ℝ W_do]

/-- The observational sufficient subspace `(ker T_obs)ᗮ` in the initial feature space `E`. -/
noncomputable def obsSubspace (T_obs : E →ₗ[ℝ] W_obs) : Submodule ℝ E :=
  (LinearMap.ker T_obs)ᗮ

/-- The interventional sufficient subspace `(ker T_do)ᗮ` in the initial feature space `E`. -/
noncomputable def doSubspace (T_do : E →ₗ[ℝ] W_do) : Submodule ℝ E :=
  (LinearMap.ker T_do)ᗮ

theorem jointSubspace_eq_sup (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do) :
    (LinearMap.ker (jointMap T_obs T_do))ᗮ = obsSubspace T_obs ⊔ doSubspace T_do := by
  rw [ker_jointMap]
  dsimp only [obsSubspace, doSubspace]
  have h_inf :
      (LinearMap.ker T_obs)ᗮᗮ ⊓ (LinearMap.ker T_do)ᗮᗮ =
        ((LinearMap.ker T_obs)ᗮ ⊔ (LinearMap.ker T_do)ᗮ)ᗮ :=
    Submodule.inf_orthogonal (LinearMap.ker T_obs)ᗮ (LinearMap.ker T_do)ᗮ
  rw [Submodule.orthogonal_orthogonal, Submodule.orthogonal_orthogonal] at h_inf
  rw [h_inf, Submodule.orthogonal_orthogonal]

theorem obsDim_eq_finrank_obsSubspace (T_obs : E →ₗ[ℝ] W_obs) :
    obsDim T_obs = Module.finrank ℝ (obsSubspace T_obs) := by
  have h_rn := LinearMap.finrank_range_add_finrank_ker T_obs
  have h_orth := Submodule.finrank_add_finrank_orthogonal (LinearMap.ker T_obs)
  rw [add_comm] at h_orth
  have h_eq :
      Module.finrank ℝ (LinearMap.range T_obs) + Module.finrank ℝ (LinearMap.ker T_obs) =
        Module.finrank ℝ (LinearMap.ker T_obs)ᗮ + Module.finrank ℝ (LinearMap.ker T_obs) :=
    h_rn.trans h_orth.symm
  exact Nat.add_right_cancel h_eq

theorem doDim_eq_finrank_doSubspace (T_do : E →ₗ[ℝ] W_do) :
    doDim T_do = Module.finrank ℝ (doSubspace T_do) := by
  have h_rn := LinearMap.finrank_range_add_finrank_ker T_do
  have h_orth := Submodule.finrank_add_finrank_orthogonal (LinearMap.ker T_do)
  rw [add_comm] at h_orth
  have h_eq :
      Module.finrank ℝ (LinearMap.range T_do) + Module.finrank ℝ (LinearMap.ker T_do) =
        Module.finrank ℝ (LinearMap.ker T_do)ᗮ + Module.finrank ℝ (LinearMap.ker T_do) :=
    h_rn.trans h_orth.symm
  exact Nat.add_right_cancel h_eq

theorem jointDim_eq_finrank_sup (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do) :
    jointDim T_obs T_do = Module.finrank ℝ (obsSubspace T_obs ⊔ doSubspace T_do : Submodule ℝ E) := by
  have h_rn := LinearMap.finrank_range_add_finrank_ker (jointMap T_obs T_do)
  have h_orth := Submodule.finrank_add_finrank_orthogonal (LinearMap.ker (jointMap T_obs T_do))
  rw [jointSubspace_eq_sup] at h_orth
  dsimp only [jointDim]
  omega

theorem causalPremium_eq_finrank_orthogonal_residual (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do) :
    causalPremium T_obs T_do =
      Module.finrank ℝ (((obsSubspace T_obs)ᗮ ⊓ (obsSubspace T_obs ⊔ doSubspace T_do) : Submodule ℝ E)) := by
  have h_add := Submodule.finrank_add_inf_finrank_orthogonal
    (le_sup_left : obsSubspace T_obs ≤ obsSubspace T_obs ⊔ doSubspace T_do)
  rw [← obsDim_eq_finrank_obsSubspace T_obs, ← jointDim_eq_finrank_sup T_obs T_do] at h_add
  have h_prem := obsDim_add_causalPremium T_obs T_do
  omega

theorem causalPremium_eq_zero_iff_subspace_le (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do) :
    causalPremium T_obs T_do = 0 ↔ doSubspace T_do ≤ obsSubspace T_obs := by
  rw [causalPremium_eq_zero_iff_ker_le]
  dsimp only [obsSubspace, doSubspace]
  exact Submodule.orthogonal_le_orthogonal_iff.symm

theorem causalPremium_eq_doDim_iff_disjoint (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do) :
    causalPremium T_obs T_do = doDim T_do ↔ Disjoint (obsSubspace T_obs) (doSubspace T_do) := by
  have h_sup_inf :=
    Submodule.finrank_sup_add_finrank_inf_eq (obsSubspace T_obs) (doSubspace T_do)
  rw [← obsDim_eq_finrank_obsSubspace T_obs, ← doDim_eq_finrank_doSubspace T_do,
    ← jointDim_eq_finrank_sup T_obs T_do] at h_sup_inf
  have h_prem := obsDim_add_causalPremium T_obs T_do
  constructor
  · intro h_eq
    have h_inf_zero :
        Module.finrank ℝ (obsSubspace T_obs ⊓ doSubspace T_do : Submodule ℝ E) = 0 := by
      omega
    have h_inf_bot : (obsSubspace T_obs ⊓ doSubspace T_do : Submodule ℝ E) = ⊥ :=
      Submodule.finrank_eq_zero.mp h_inf_zero
    exact disjoint_iff.mpr h_inf_bot
  · intro h_disj
    have h_inf_bot : (obsSubspace T_obs ⊓ doSubspace T_do : Submodule ℝ E) = ⊥ :=
      disjoint_iff.mp h_disj
    have h_inf_zero :
        Module.finrank ℝ (obsSubspace T_obs ⊓ doSubspace T_do : Submodule ℝ E) = 0 := by
      rw [h_inf_bot, finrank_bot]
    omega

theorem causalPremium_eq_doDim_of_isOrtho (T_obs : E →ₗ[ℝ] W_obs) (T_do : E →ₗ[ℝ] W_do)
    (h_ortho : obsSubspace T_obs ⟂ doSubspace T_do) :
    causalPremium T_obs T_do = doDim T_do :=
  (causalPremium_eq_doDim_iff_disjoint T_obs T_do).mpr h_ortho.disjoint

end InnerProductDecomposition

section BackdoorBlockDecomposition

variable {K E_A E_Z F_Y : Type*}
variable [Field K]
variable [AddCommGroup E_A] [Module K E_A]
variable [AddCommGroup E_Z] [Module K E_Z]
variable [AddCommGroup F_Y] [Module K F_Y]

/-- Range of the backdoor-adjusted observational map `L_A.coprod L_Z` equals `range L_A ⊔ range L_Z`. -/
theorem range_coprod_eq_sup (L_A : E_A →ₗ[K] F_Y) (L_Z : E_Z →ₗ[K] F_Y) :
    LinearMap.range (L_A.coprod L_Z) = LinearMap.range L_A ⊔ LinearMap.range L_Z := by
  have h_top : (⊤ : Submodule K (E_A × E_Z)) = Submodule.prod ⊤ ⊤ := by
    ext ⟨a, z⟩
    simp
  rw [← Submodule.map_top (L_A.coprod L_Z), h_top, LinearMap.coprod_map_prod,
    Submodule.map_top, Submodule.map_top]

section BackdoorFiniteDim

variable [FiniteDimensional K E_A] [FiniteDimensional K E_Z]

/-- Multi-node backdoor joint cache dimension equals `rank(L_A) + rank(L_Z)`. -/
theorem backdoor_joint_rank (L_A : E_A →ₗ[K] F_Y) (L_Z : E_Z →ₗ[K] F_Y) :
    jointDim (L_A.coprod L_Z) (L_Z.comp (LinearMap.snd K E_A E_Z)) =
      Module.finrank K (LinearMap.range L_A) + Module.finrank K (LinearMap.range L_Z) := by
  let obsMap := L_A.coprod L_Z
  let doMap := L_Z.comp (LinearMap.snd K E_A E_Z)
  let shear : (F_Y × F_Y) ≃ₗ[K] (F_Y × F_Y) :=
    { toFun := fun q => (q.1 - q.2, q.2)
      invFun := fun q => (q.1 + q.2, q.2)
      left_inv := fun q => by ext <;> simp
      right_inv := fun q => by ext <;> simp
      map_add' := fun q₁ q₂ => by ext <;> simp [sub_eq_add_neg]; abel
      map_smul' := fun c q => by ext <;> simp [smul_sub] }
  have h_comp : shear.toLinearMap.comp (jointMap obsMap doMap) = L_A.prodMap L_Z := by
    apply LinearMap.ext
    rintro ⟨a, z⟩
    simp [shear, jointMap, obsMap, doMap]
  have h_range_eq :
      (LinearMap.range (jointMap obsMap doMap)).map shear.toLinearMap =
        (LinearMap.range L_A).prod (LinearMap.range L_Z) := by
    rw [← LinearMap.range_comp, h_comp, LinearMap.range_prodMap]
  let e_range :
      LinearMap.range (jointMap obsMap doMap) ≃ₗ[K]
        (LinearMap.range L_A).prod (LinearMap.range L_Z) :=
    (shear.submoduleMap (LinearMap.range (jointMap obsMap doMap))).trans
      (LinearEquiv.ofEq _ _ h_range_eq)
  let e_prod :
      ↥((LinearMap.range L_A).prod (LinearMap.range L_Z)) ≃ₗ[K]
        ↥(LinearMap.range L_A) × ↥(LinearMap.range L_Z) :=
    { toFun := fun v => (⟨v.1.1, v.2.1⟩, ⟨v.1.2, v.2.2⟩)
      invFun := fun v => ⟨(v.1.1, v.2.1), v.1.2, v.2.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
  have : FiniteDimensional K (LinearMap.range L_A) :=
    LinearMap.finiteDimensional_range L_A
  have : FiniteDimensional K (LinearMap.range L_Z) :=
    LinearMap.finiteDimensional_range L_Z
  dsimp only [jointDim]
  rw [(e_range.trans e_prod).finrank_eq]
  exact Module.finrank_prod

/-- Exact backdoor subspace intersection theorem:
`Δ_d_causal = dim(im(Θ_YA) ∩ im(Θ_YZ))`. -/
theorem backdoor_causalPremium_eq_inf_range (L_A : E_A →ₗ[K] F_Y) (L_Z : E_Z →ₗ[K] F_Y) :
    causalPremium (L_A.coprod L_Z) (L_Z.comp (LinearMap.snd K E_A E_Z)) =
      Module.finrank K (LinearMap.range L_A ⊓ LinearMap.range L_Z : Submodule K F_Y) := by
  have : FiniteDimensional K (LinearMap.range L_A) :=
    LinearMap.finiteDimensional_range L_A
  have : FiniteDimensional K (LinearMap.range L_Z) :=
    LinearMap.finiteDimensional_range L_Z
  have h_sup_inf :=
    Submodule.finrank_sup_add_finrank_inf_eq (LinearMap.range L_A) (LinearMap.range L_Z)
  have h_obs :
      obsDim (L_A.coprod L_Z) =
        Module.finrank K (LinearMap.range L_A ⊔ LinearMap.range L_Z : Submodule K F_Y) := by
    dsimp only [obsDim]
    rw [range_coprod_eq_sup]
  have h_joint := backdoor_joint_rank L_A L_Z
  have h_prem := obsDim_add_causalPremium (L_A.coprod L_Z) (L_Z.comp (LinearMap.snd K E_A E_Z))
  omega

/-- Multi-node backdoor causal invariance holds iff `im(Θ_YA)` and `im(Θ_YZ)` are disjoint. -/
theorem backdoor_causalInvariance_iff (L_A : E_A →ₗ[K] F_Y) (L_Z : E_Z →ₗ[K] F_Y) :
    causalPremium (L_A.coprod L_Z) (L_Z.comp (LinearMap.snd K E_A E_Z)) = 0 ↔
      Disjoint (LinearMap.range L_A) (LinearMap.range L_Z) := by
  have : FiniteDimensional K (LinearMap.range L_A) :=
    LinearMap.finiteDimensional_range L_A
  have : FiniteDimensional K (LinearMap.range L_A ⊓ LinearMap.range L_Z : Submodule K F_Y) :=
    Submodule.finiteDimensional_inf_left (LinearMap.range L_A) (LinearMap.range L_Z)
  rw [backdoor_causalPremium_eq_inf_range, Submodule.finrank_eq_zero, disjoint_iff]

end BackdoorFiniteDim

end BackdoorBlockDecomposition

end BackdoorSubspace

/-! ## Section 4: Smooth Nonlinear Local Jacobian Obstruction -/

section NonlinearJacobian

variable {E W_obs W_do : Type*}
variable [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [NormedAddCommGroup W_obs] [NormedSpace ℝ W_obs]
variable [NormedAddCommGroup W_do] [NormedSpace ℝ W_do]

theorem hasFDerivAt_joint_query {q_obs : E → W_obs} {q_do : E → W_do}
    {J_obs : E →L[ℝ] W_obs} {J_do : E →L[ℝ] W_do} {x₀ : E}
    (h_obs : HasFDerivAt q_obs J_obs x₀) (h_do : HasFDerivAt q_do J_do x₀) :
    HasFDerivAt (fun x => (q_obs x, q_do x)) (J_obs.prod J_do) x₀ :=
  h_obs.prodMk h_do

section NonlinearFiniteDim

variable [FiniteDimensional ℝ E]

theorem jacobian_causalPremium_eq_zero_of_local_factor
    {q_obs : E → W_obs} {q_do : E → W_do} {h : W_obs → W_do}
    {J_obs : E →L[ℝ] W_obs} {J_do : E →L[ℝ] W_do} {Dh : W_obs →L[ℝ] W_do} {x₀ : E}
    (h_obs : HasFDerivAt q_obs J_obs x₀)
    (h_do : HasFDerivAt q_do J_do x₀)
    (h_factor_deriv : HasFDerivAt h Dh (q_obs x₀))
    (h_factor_eq : q_do =ᶠ[𝓝 x₀] fun x => h (q_obs x)) :
    causalPremium (J_obs : E →ₗ[ℝ] W_obs) (J_do : E →ₗ[ℝ] W_do) = 0 := by
  have h_comp : HasFDerivAt (fun x => h (q_obs x)) (Dh.comp J_obs) x₀ :=
    h_factor_deriv.comp x₀ h_obs
  have h_do_comp : HasFDerivAt q_do (Dh.comp J_obs) x₀ :=
    h_comp.congr_of_eventuallyEq h_factor_eq
  have hJ_eq : J_do = Dh.comp J_obs := h_do.unique h_do_comp
  rw [causalPremium_eq_zero_iff_exists_factor]
  refine ⟨(Dh : W_obs →ₗ[ℝ] W_do), ?_⟩
  have h_lin : (J_do : E →ₗ[ℝ] W_do) = (Dh : W_obs →ₗ[ℝ] W_do).comp (J_obs : E →ₗ[ℝ] W_obs) := by
    rw [hJ_eq]
    rfl
  exact h_lin.symm

theorem no_local_diff_factor_of_jacobian_premium_pos
    {q_obs : E → W_obs} {q_do : E → W_do}
    {J_obs : E →L[ℝ] W_obs} {J_do : E →L[ℝ] W_do} {x₀ : E}
    (h_obs : HasFDerivAt q_obs J_obs x₀)
    (h_do : HasFDerivAt q_do J_do x₀)
    (h_pos : 0 < causalPremium (J_obs : E →ₗ[ℝ] W_obs) (J_do : E →ₗ[ℝ] W_do)) :
    ¬ ∃ (h : W_obs → W_do) (Dh : W_obs →L[ℝ] W_do),
        HasFDerivAt h Dh (q_obs x₀) ∧ (q_do =ᶠ[𝓝 x₀] fun x => h (q_obs x)) := by
  rintro ⟨h, Dh, h_deriv, h_eq⟩
  have h_zero := jacobian_causalPremium_eq_zero_of_local_factor h_obs h_do h_deriv h_eq
  omega

end NonlinearFiniteDim

end NonlinearJacobian

/-! ## Section 5: Active-Probe Side-Information & Failure Witnesses -/

section ActiveProbe

section GeneralMonotonicity

variable {K X W_obs W_probe W_do : Type*}
variable [Field K]
variable [AddCommGroup X] [Module K X] [FiniteDimensional K X]
variable [AddCommGroup W_obs] [Module K W_obs]
variable [AddCommGroup W_probe] [Module K W_probe]
variable [AddCommGroup W_do] [Module K W_do]

theorem causalPremium_sideInfo_mono
    (T_obs : X →ₗ[K] W_obs) (T_probe : X →ₗ[K] W_probe) (T_do : X →ₗ[K] W_do) :
    causalPremium (T_obs.prod T_probe) T_do ≤ causalPremium T_obs T_do := by
  rw [causalPremium_eq_finrank_map_ker, causalPremium_eq_finrank_map_ker, LinearMap.ker_prod]
  exact Submodule.finrank_mono (Submodule.map_mono inf_le_left)

theorem causalPremium_sideInfo_eq_zero_iff
    (T_obs : X →ₗ[K] W_obs) (T_probe : X →ₗ[K] W_probe) (T_do : X →ₗ[K] W_do) :
    causalPremium (T_obs.prod T_probe) T_do = 0 ↔
      LinearMap.ker T_obs ⊓ LinearMap.ker T_probe ≤ LinearMap.ker T_do := by
  rw [causalPremium_eq_zero_iff_ker_le, LinearMap.ker_prod]

end GeneralMonotonicity

section ConcreteProbeWitnesses

/-- Pre-probe linear observational query `(a, z, u) ↦ a + u`. -/
def probeObsMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.1 + p.2.2
  map_add' p q := by simp only [Prod.fst_add, Prod.snd_add]; ring
  map_smul' c p := by simp; ring

/-- Interventional query `(a, z, u) ↦ u`. -/
def probeDoMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.2.2
  map_add' p q := by simp only [Prod.snd_add]
  map_smul' c p := by simp

/-- Active probe reading `(a, z, u) ↦ u` (noiseless deconfounding). -/
def probeReadMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.2.2
  map_add' p q := by simp only [Prod.snd_add]
  map_smul' c p := by simp

theorem probe_collapse_before : causalPremium probeObsMap probeDoMap = 1 := by
  have h_pos : 0 < causalPremium probeObsMap probeDoMap := by
    rw [causalPremium_pos_iff_exists_witness]
    refine ⟨(-1, 0, 1), ?_, ?_⟩
    · simp [probeObsMap]
    · simp [probeDoMap]
  have h_le : causalPremium probeObsMap probeDoMap ≤ doDim probeDoMap :=
    causalPremium_le_doDim probeObsMap probeDoMap
  have h_do_le : doDim probeDoMap ≤ 1 := by
    dsimp only [doDim]
    have h := Submodule.finrank_le (LinearMap.range probeDoMap)
    simpa [CommSemiring.finrank_self] using h
  omega

theorem probe_collapse_after :
    causalPremium (probeObsMap.prod probeReadMap) probeDoMap = 0 := by
  rw [causalPremium_sideInfo_eq_zero_iff]
  intro p hp
  simp only [Submodule.mem_inf, LinearMap.mem_ker, probeReadMap, LinearMap.coe_mk,
    AddHom.coe_mk] at hp ⊢
  exact hp.2

/-- 2-confounder observational query `(a, u₁, u₂) ↦ a + u₁ + u₂`. -/
def twoConfObsMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.1 + p.2.1 + p.2.2
  map_add' p q := by simp only [Prod.fst_add, Prod.snd_add]; ring
  map_smul' c p := by simp; ring

/-- 2-confounder interventional query `(a, u₁, u₂) ↦ u₁ + u₂`. -/
def twoConfDoMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.2.1 + p.2.2
  map_add' p q := by simp only [Prod.fst_add, Prod.snd_add]; ring
  map_smul' c p := by simp; ring

/-- Rank-1 probe resolving only `u₁`: `(a, u₁, u₂) ↦ u₁`. -/
def twoConfProbeMap : ℝ × ℝ × ℝ →ₗ[ℝ] ℝ where
  toFun p := p.2.1
  map_add' p q := by simp only [Prod.fst_add, Prod.snd_add]
  map_smul' c p := by simp

theorem probe_two_confounder_fail_premium :
    causalPremium (twoConfObsMap.prod twoConfProbeMap) twoConfDoMap = 1 := by
  have h_pos : 0 < causalPremium (twoConfObsMap.prod twoConfProbeMap) twoConfDoMap := by
    rw [causalPremium_pos_iff_exists_witness]
    refine ⟨(-1, 0, 1), ?_, ?_⟩
    · simp [twoConfObsMap, twoConfProbeMap]
    · simp [twoConfDoMap]
  have h_le : causalPremium (twoConfObsMap.prod twoConfProbeMap) twoConfDoMap ≤ doDim twoConfDoMap :=
    causalPremium_le_doDim _ _
  have h_do_le : doDim twoConfDoMap ≤ 1 := by
    dsimp only [doDim]
    have h := Submodule.finrank_le (LinearMap.range twoConfDoMap)
    simpa [CommSemiring.finrank_self] using h
  omega

/-- Deterministic injective nonlinear probe: for `a_probe > 0`, `u ↦ a_probe * u + u ^ 2`
is strictly monotone and hence injective on `{u : ℝ | 0 < u}`. -/
theorem probe_nonlinear_inj {a_probe : ℝ} (ha : 0 < a_probe) :
    ∀ u₁ u₂ : ℝ, 0 < u₁ → 0 < u₂ →
      a_probe * u₁ + u₁ ^ 2 = a_probe * u₂ + u₂ ^ 2 → u₁ = u₂ := by
  intro u₁ u₂ hu₁ hu₂ h_eq
  have h_factor : (u₁ - u₂) * (a_probe + u₁ + u₂) = 0 := by
    linarith [show (u₁ - u₂) * (a_probe + u₁ + u₂) =
      (a_probe * u₁ + u₁ ^ 2) - (a_probe * u₂ + u₂ ^ 2) by ring]
  have h_pos : 0 < a_probe + u₁ + u₂ := by linarith
  have h_diff : u₁ - u₂ = 0 := (mul_eq_zero.mp h_factor).resolve_right (ne_of_gt h_pos)
  linarith

/-- Zero-sensitivity active probe failure: `Y(a, u) = α * a + (a - a_probe) * u` is constant
in `u` at `a = a_probe`, yet depends non-trivially on `u` for all `a ≠ a_probe`. -/
theorem probe_zero_sensitivity_fail (α a_probe : ℝ) :
    (∀ u₁ u₂ : ℝ,
      α * a_probe + (a_probe - a_probe) * u₁ = α * a_probe + (a_probe - a_probe) * u₂) ∧
    (∀ a u₁ u₂ : ℝ, a ≠ a_probe → u₁ ≠ u₂ →
      α * a + (a - a_probe) * u₁ ≠ α * a + (a - a_probe) * u₂) := by
  constructor
  · intro u₁ u₂; ring
  · intro a u₁ u₂ ha hu h_eq
    have h_factor : (a - a_probe) * (u₁ - u₂) = 0 := by linarith
    rcases mul_eq_zero.mp h_factor with h1 | h2
    · exact ha (sub_eq_zero.mp h1)
    · exact hu (sub_eq_zero.mp h2)

/-- 2D confounder dimension-deficit failure: for `Y(a, u₁, u₂) = u₁ + a * u₂`, at any `a_probe`
there exist distinct latent pairs `(u₁, u₂) ≠ (u₁', u₂')` colliding under `a_probe` that diverge
at every other intervention `a ≠ a_probe`. -/
theorem probe_2d_deficit_fail (a_probe : ℝ) :
    ∃ (u₁ u₂ u₁' u₂' : ℝ),
      (u₁ ≠ u₁' ∨ u₂ ≠ u₂') ∧
      u₁ + a_probe * u₂ = u₁' + a_probe * u₂' ∧
      ∀ a : ℝ, a ≠ a_probe → u₁ + a * u₂ ≠ u₁' + a * u₂' := by
  refine ⟨a_probe, 0, 0, 1, Or.inr (by norm_num), by ring, ?_⟩
  intro a ha h_eq
  have h_sub : a_probe - a = 0 := by linarith
  exact ha (sub_eq_zero.mp h_sub).symm

end ConcreteProbeWitnesses

end ActiveProbe

/-! ## Axiom Verification for All Main Theorems -/

#print axioms obsDim_le_jointDim
#print axioms doDim_le_jointDim
#print axioms obsDim_add_causalPremium
#print axioms causalPremium_eq_finrank_ker_sub
#print axioms causalPremium_eq_finrank_map_ker
#print axioms causalPremium_le_doDim
#print axioms causalPremium_eq_zero_iff_ker_le
#print axioms ker_le_iff_exists_factor
#print axioms ker_le_iff_exists_set_factor
#print axioms causalPremium_eq_zero_iff_exists_factor
#print axioms causalPremium_pos_iff_exists_witness

#print axioms canonJointMap_bijective_iff
#print axioms canonJointMap_bilipschitz
#print axioms canonJointMap_sq_norm_bilipschitz
#print axioms canonJointMap_not_antilipschitz_of_eq_zero
#print axioms canonCausalPremium_eq_one_iff
#print axioms canonCausalPremium_eq_zero_iff
#print axioms canonDiag_causalPremium_eq_zero
#print axioms canonDiag_unfaithful_premium_eq_one

#print axioms jointSubspace_eq_sup
#print axioms obsDim_eq_finrank_obsSubspace
#print axioms doDim_eq_finrank_doSubspace
#print axioms jointDim_eq_finrank_sup
#print axioms causalPremium_eq_finrank_orthogonal_residual
#print axioms causalPremium_eq_zero_iff_subspace_le
#print axioms causalPremium_eq_doDim_iff_disjoint
#print axioms causalPremium_eq_doDim_of_isOrtho
#print axioms range_coprod_eq_sup
#print axioms backdoor_joint_rank
#print axioms backdoor_causalPremium_eq_inf_range
#print axioms backdoor_causalInvariance_iff

#print axioms hasFDerivAt_joint_query
#print axioms jacobian_causalPremium_eq_zero_of_local_factor
#print axioms no_local_diff_factor_of_jacobian_premium_pos

#print axioms causalPremium_sideInfo_mono
#print axioms causalPremium_sideInfo_eq_zero_iff
#print axioms probe_collapse_before
#print axioms probe_collapse_after
#print axioms probe_two_confounder_fail_premium
#print axioms probe_nonlinear_inj
#print axioms probe_zero_sensitivity_fail
#print axioms probe_2d_deficit_fail

end MathLab.LateQueryFrontier.CausalPremium
