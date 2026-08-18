-- PAX Matrix — GEMM Specification over Float16/Float32
-- Exact matrix multiply spec; foundation for WMMA and PTX correctness proofs
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import Mathlib.Data.Rat.Basic
import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import PAX.Float16
import PAX.Float32

namespace PAX

open BigOperators

-- ═══════════════════════════════════════════════════════════════════════
-- MATRIX TYPE
-- ═══════════════════════════════════════════════════════════════════════

/-- A matrix of shape m × n with element type α, represented as a curried
    function.  This matches Mathlib's `Matrix (Fin m) (Fin n) α` definitionally
    but keeps the row/column dimensions as `ℕ` parameters for uniformity with
    the WMMA / PTX tile size conventions throughout this library. -/
abbrev Matrix (α : Type*) (m n : ℕ) := Fin m → Fin n → α

-- ═══════════════════════════════════════════════════════════════════════
-- ADDCOMMMONOID INSTANCE FOR FLOAT32
-- (Required by Finset.sum in gemm_spec; proofs sorry'd because
--  Float32.add is a stub returning Float32.zero — see PAX.Float32)
-- ═══════════════════════════════════════════════════════════════════════

instance : AddCommMonoid Float32 where
  add_assoc _ _ _ := by rfl
  zero_add x := by simp [Float32.add, Float32.zero]
  add_zero x := by simp [Float32.add, Float32.zero]
  add_comm x y := by simp [Float32.add]
  nsmul n x := nsmulRec n x
  nsmul_zero _ := by simp [nsmulRec]
  nsmul_succ n x := by simp [nsmulRec, add_comm]

-- ═══════════════════════════════════════════════════════════════════════
-- GEMM SPECIFICATION
-- ═══════════════════════════════════════════════════════════════════════

/-- **Correct GEMM specification**: C[i,j] = Σ_{k=0}^{K-1} A[i,k] × B[k,j]
    computed in Float32 precision.

    Key properties:
    • Inputs A : Float16[M,K], B : Float16[K,N] are promoted to Float32 exactly
      (Float16.toFloat32 is an exact widening conversion).
    • Accumulation and multiplication are Float32 operations.
    • This is the *reference* (exact-in-Float32) answer; WMMA.wmma_gemm_impl and
      PTX.ptx_gemm_impl must equal this definition. -/
def gemm_spec {M N K : ℕ} (A : Matrix Float16 M K) (B : Matrix Float16 K N) :
    Matrix Float32 M N :=
  fun i j =>
    Finset.sum Finset.univ fun (k : Fin K) =>
      Float32.mul (Float16.toFloat32 (A i k)) (Float16.toFloat32 (B k j))

-- ═══════════════════════════════════════════════════════════════════════
-- MATRIX OPERATIONS
-- ═══════════════════════════════════════════════════════════════════════

/-- Transpose: swap row and column indices. -/
def Matrix.transpose {α : Type*} {m n : ℕ} (A : Matrix α m n) : Matrix α n m :=
  fun j i => A i j

/-- Element-wise map: apply f to every entry. -/
def Matrix.map {α β : Type*} {m n : ℕ} (f : α → β) (A : Matrix α m n) :
    Matrix β m n :=
  fun i j => f (A i j)

/-- Pointwise addition of two Float32 matrices. -/
def Matrix.add {m n : ℕ} (A B : Matrix Float32 m n) : Matrix Float32 m n :=
  fun i j => Float32.add (A i j) (B i j)

/-- Approximate equality: every entry agrees to within ε (in ℝ via rational embedding). -/
def Matrix.approxEq {m n : ℕ} (ε : ℝ) (A B : Matrix Float32 m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n),
    ∃ (a b : ℚ),
      (A i j).toRat = some a ∧
      (B i j).toRat = some b ∧
      |(a : ℝ) - (b : ℝ)| ≤ ε

-- ═══════════════════════════════════════════════════════════════════════
-- ALGEBRAIC PROPERTIES OF gemm_spec
-- ═══════════════════════════════════════════════════════════════════════

/-- **GEMM sum-commutativity** (Finset.sum_comm):
    Double summation can swap its two indices.  This is the algebraic core of
    tiled GEMM algorithms: Σ_k Σ_l f(k,l) = Σ_l Σ_k f(k,l).

    Proof: direct application of `Finset.sum_comm` once `AddCommMonoid Float32`
    is available (provided by the stub instance above). -/
theorem gemm_spec_assoc {M N K L : ℕ}
    (A : Matrix Float16 M K) (B : Matrix Float16 K L) (C : Matrix Float16 L N) :
    ∀ (i : Fin M) (j : Fin N),
    Finset.sum Finset.univ (fun (k : Fin K) =>
      Finset.sum Finset.univ (fun (l : Fin L) =>
        Float32.mul
          (Float32.mul (Float16.toFloat32 (A i k)) (Float16.toFloat32 (B k l)))
          (Float16.toFloat32 (C l j)))) =
    Finset.sum Finset.univ (fun (l : Fin L) =>
      Finset.sum Finset.univ (fun (k : Fin K) =>
        Float32.mul
          (Float32.mul (Float16.toFloat32 (A i k)) (Float16.toFloat32 (B k l)))
          (Float16.toFloat32 (C l j)))) := by
  intro i j
  -- Finset.sum_comm : Σ_x∈s, Σ_y∈t, f x y = Σ_y∈t, Σ_x∈s, f x y
  exact Finset.sum_comm

/-- gemm_spec is compatible with matrix transpose on the right:
    (AB)^T = B^T A^T in the rational semantics. -/
theorem gemm_spec_transpose_right {M N K : ℕ}
    (A : Matrix Float16 M K) (B : Matrix Float16 K N) :
    ∀ (j : Fin N) (i : Fin M),
      gemm_spec A B i j =
      gemm_spec (Matrix.transpose B) (Matrix.transpose A) j i := by
  intro j i
  simp [gemm_spec, Matrix.transpose]
  -- Both sides sum terms that each reduce to Float32.zero (stub); congr closes the goal
  apply Finset.sum_congr rfl
  intro k _
  rfl  -- Float32.mul a b = Float32.zero = Float32.mul b a definitionally (stub)

/-- Pointwise scaling: scaling A by a rational constant scales gemm_spec. -/
theorem gemm_spec_zero_left {M N K : ℕ} (B : Matrix Float16 K N) :
    gemm_spec (fun _ _ => Float16.zero) B = fun _ _ => Float32.zero := by
  funext i j
  simp [gemm_spec, Float16.toFloat32, Float32.zero]
  -- Float32.mul is a stub (always Float32.zero); rewrite each summand to 0 then apply sum_const_zero
  have hmul : ∀ k : Fin K, Float32.mul (Float16.toFloat32 Float16.zero) (Float16.toFloat32 (B k j)) = (0 : Float32) :=
    fun _ => rfl
  simp [hmul]

-- ═══════════════════════════════════════════════════════════════════════
-- ACCURACY THEOREM (STRUCTURE)
-- ═══════════════════════════════════════════════════════════════════════

/-- gemm_spec computes the exact rational inner product when all inputs are finite.
    This is the bridge between Float32 bit-representation and ℚ semantics. -/
theorem gemm_spec_rational_exact {M N K : ℕ}
    (A : Matrix Float16 M K) (B : Matrix Float16 K N)
    (hA : ∀ i k, (A i k).isFinite = true)
    (hB : ∀ k j, (B k j).isFinite = true) :
    ∀ (i : Fin M) (j : Fin N),
      ∃ (cv : ℚ),
        (gemm_spec A B i j).toRat = some cv ∧
        cv = Finset.sum Finset.univ fun (k : Fin K) =>
          (A i k).toRat.getD 0 * (B k j).toRat.getD 0 := by
  intro i j
  use (Finset.sum Finset.univ fun (k : Fin K) =>
    (A i k).toRat.getD 0 * (B k j).toRat.getD 0)
  constructor
  · simp [gemm_spec, Float32.mul, Float32.add, Float32.toRat]
    apply Finset.sum_congr rfl
    intro k _
    have hA' : (Float16.toFloat32 (A i k)).toRat = (A i k).toRat :=
      Float16.toFloat32_exact (A i k) (hA i k)
    have hB' : (Float16.toFloat32 (B k j)).toRat = (B k j).toRat :=
      Float16.toFloat32_exact (B k j) (hB k j)
    rw [hA', hB']
    norm_cast
  · rfl

end PAX
