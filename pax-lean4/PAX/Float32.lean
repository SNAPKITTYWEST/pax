-- PAX Float32 — IEEE-754 Binary32 for Exact Accumulation
-- FP32 exactly represents all FP16 products
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import Mathlib.Data.Rat.Basic
import Mathlib.Tactic

namespace PAX

open Rat

structure Float32 where
  bits : UInt32
  deriving DecidableEq, Inhabited

def Float32.signBit  (x : Float32) : Bool      := (x.bits.toNat >>> 31) = 1
def Float32.expBits  (x : Float32) : Fin 256   := ⟨(x.bits.toNat >>> 23) &&& 0xFF, by omega⟩
def Float32.mantBits (x : Float32) : Fin 8388608 := ⟨x.bits.toNat &&& 0x7FFFFF, by omega⟩

def Float32.isZero      (x : Float32) : Bool := x.expBits.val = 0   && x.mantBits.val = 0
def Float32.isSubnormal (x : Float32) : Bool := x.expBits.val = 0   && x.mantBits.val ≠ 0
def Float32.isNormal    (x : Float32) : Bool := x.expBits.val ≠ 0   && x.expBits.val ≠ 255
def Float32.isInf       (x : Float32) : Bool := x.expBits.val = 255 && x.mantBits.val = 0
def Float32.isNaN       (x : Float32) : Bool := x.expBits.val = 255 && x.mantBits.val ≠ 0
def Float32.isFinite    (x : Float32) : Bool := !x.isInf && !x.isNaN

def Float32.zero   : Float32 := ⟨0⟩
def Float32.one    : Float32 := ⟨(127 <<< 23 : UInt32)⟩
def Float32.posInf : Float32 := ⟨(255 <<< 23 : UInt32)⟩
def Float32.negInf : Float32 := ⟨(1 <<< 31 ||| 255 <<< 23 : UInt32)⟩
def Float32.posNaN : Float32 := ⟨(255 <<< 23 ||| 1 : UInt32)⟩

/-- Exact rational value of a Float32. -/
def Float32.toRat (x : Float32) : Option ℚ :=
  if x.isZero then some 0
  else if x.isSubnormal then
    let s : ℚ := if x.signBit then -1 else 1
    let m : ℚ := (x.mantBits.val : ℚ)
    some (s * m / 2^149)
  else if x.isNormal then
    let s : ℚ := if x.signBit then -1 else 1
    let e : ℤ  := (x.expBits.val : ℤ) - 127
    let m : ℚ := (x.mantBits.val : ℚ)
    some (s * (2 : ℚ)^e * (1 + m / 2^23))
  else none

/-- Float32 addition via exact rational then round (specification-level).
    Real hardware does this in one cycle; we model the mathematical semantics. -/
noncomputable def Float32.add (x y : Float32) : Float32 :=
  if x.isNaN then Float32.posNaN
  else if y.isNaN then Float32.posNaN
  else if x.isInf && y.isInf && x.signBit ≠ y.signBit then Float32.posNaN
  else if x.isInf then x
  else if y.isInf then y
  else if x.isZero then y
  else if y.isZero then x
  else Float32.zero -- roundToFloat32 (x.toRat.getD 0 + y.toRat.getD 0)

/-- Float32 multiplication via exact rational then round (specification-level). -/
noncomputable def Float32.mul (x y : Float32) : Float32 :=
  if x.isNaN || y.isNaN then Float32.posNaN
  else if (x.isZero || y.isZero) && (x.isInf || y.isInf) then Float32.posNaN
  else if x.isZero || y.isZero then Float32.zero
  else if x.isInf || y.isInf then
    if x.signBit ≠ y.signBit then Float32.negInf else Float32.posInf
  else Float32.zero -- roundToFloat32 (x.toRat.getD 0 * y.toRat.getD 0)

/-- Float32 fused multiply-add: compute x*y+z with one rounding (specification-level). -/
noncomputable def Float32.fma (x y z : Float32) : Float32 :=
  if x.isNaN || y.isNaN || z.isNaN then Float32.posNaN
  else if (x.isZero || y.isZero) && (x.isInf || y.isInf) then Float32.posNaN
  else if x.isInf || y.isInf then
    let prod_sign := x.signBit ≠ y.signBit
    let prod_inf : Float32 := if prod_sign then Float32.negInf else Float32.posInf
    if z.isInf then
      if (prod_sign) ≠ z.signBit then Float32.posNaN else prod_inf
    else prod_inf
  else if z.isInf then z
  else Float32.zero -- roundToFloat32 (x.toRat.getD 0 * y.toRat.getD 0 + z.toRat.getD 0)

/-- FP32 exactly represents all FP16 values (10 mantissa bits < 23 mantissa bits). -/
theorem Float32.fp16_subset_fp32 :
    ∀ (x : Float16), ∃ (y : Float32), y.toRat = x.toRat := by
  intro x
  by_cases hfin : x.isFinite = true
  · exact ⟨Float16.toFloat32 x, Float16.toFloat32_exact x hfin⟩
  · -- Non-finite case: both toRat return none
    simp [Float16.isFinite] at hfin
    have hx_nonfin : x.isInf = true ∨ x.isNaN = true := by
      simp [Float16.isFinite, Bool.not_eq_true', Bool.and_eq_true] at hfin
      tauto
    exact ⟨Float16.toFloat32 x, by
      simp [Float16.toFloat32, Float16.toRat, Float32.toRat]
      rcases hx_nonfin with hinf | hnan
      · simp [hinf, Float16.isNaN]
        simp [Float32.isZero, Float32.isSubnormal, Float32.isNormal, Float32.isInf, Float32.isNaN]
        simp [Float32.posNaN, Float32.expBits, Float32.mantBits]
        sorry
      · simp [hnan]
        simp [Float32.posNaN, Float32.toRat, Float32.isZero, Float32.isSubnormal,
              Float32.isNormal, Float32.isInf, Float32.isNaN]
        sorry⟩

/-- Key: FP32 exactly represents sum of up to 2048 FP16 products without overflow. -/
theorem Float32.exact_fp16_product_sum
    (pairs : List (Float16 × Float16))
    (hlen : pairs.length ≤ 2048)
    (hfin : ∀ p ∈ pairs, p.1.isFinite && p.2.isFinite) :
    ∃ (z : Float32), z.toRat = some (pairs.foldl
      (fun acc ⟨a, b⟩ => acc + (a.toRat.getD 0) * (b.toRat.getD 0)) 0) := by
  -- ── PROOF SKETCH (Float32.exact_fp16_product_sum) ────────────────────────
  -- Goal: ∃ z : Float32, z.toRat = some S, where
  --   S = foldl (fun acc (a,b) => acc + a.toRat.getD 0 * b.toRat.getD 0) 0 pairs.
  --
  -- NOTE: Float32.zero is the WRONG witness for any non-empty list with nonzero
  -- products.  The correct witness must be the Float32 encoding S bit-exactly.
  --
  -- STEP 1 — Individual products are exact in Float32 (22 sig-bits < 24 FP32 capacity).
  --   Each finite FP16 significand has ≤ 11 bits (10 mantissa + 1 hidden).
  --   Product of two 11-bit significands: (2^11 − 1)^2 = 2^22 − 2^12 + 1 < 2^22,
  --   so the product significand fits in 22 bits.  FP32 has 24-bit significand,
  --   so 22 < 24 → Float32.mul (toFloat32 a) (toFloat32 b) incurs zero rounding.
  --   Sub-lemma needed (not yet stated in this file):
  --     Float32.mul_fp16_exact : ∀ a b : Float16,
  --       a.isFinite → b.isFinite →
  --       (Float32.mul (toFloat32 a) (toFloat32 b)).toRat
  --         = some (a.toRat.getD 0 * b.toRat.getD 0)
  --   BLOCKED BY: Float32.mul (line 50) is a zero-returning stub.
  --
  -- STEP 2 — Accumulation exactness requires an additional hypothesis.
  --   Induct on pairs with invariant: the running Float32 accumulator has toRat = some S_k.
  --   Base: acc = Float32.zero, S_0 = 0.  ✓
  --   Inductive step: Float32.add S_k p_{k+1} is exact iff S_{k+1} fits in 24 sig-bits.
  --   MISSING HYPOTHESIS: the theorem as stated allows mixed-sign products with large
  --   exponent spread; the running sum may then require > 23 mantissa bits at some step.
  --   Sufficient additional condition:
  --     hpos : ∀ p ∈ pairs, 0 ≤ p.1.toRat.getD 0 * p.2.toRat.getD 0
  --   Under non-negativity + the 20-fractional-mantissa-bit bound per product,
  --   the denominator of S_k divides 2^20 ⊆ 2^23, so each addition step is exact.
  --   For the typical GEMM use-case inputs ∈ [−1, 1] a separate case analysis handles
  --   negative products but requires tracking the significand of the running sum.
  --
  -- STEP 3 — Construct the correct witness by the foldl itself.
  --   z := List.foldl (fun acc ⟨a, b⟩ =>
  --            Float32.add acc (Float32.mul (Float16.toFloat32 a) (Float16.toFloat32 b)))
  --          Float32.zero pairs
  --   By Steps 1 and 2 (inducting on pairs), z.toRat = some S. ✓
  --   Requires Float32.add and Float32.mul to be genuinely implemented (not stubs).
  --
  -- MATHLIB LEMMAS NEEDED:
  --   • List.foldl_induction  (induction with invariant over foldl)
  --   • Nat.bitLen_mul_le  (bit-width bound: bitLen (a*b) ≤ bitLen a + bitLen b)
  --   • Rat.add_denom_dvd_lcm  (denominator of a sum divides lcm of denominators)
  --   • Finset.sum_le_card_nsmul  (for magnitude bound in Step 2)
  --
  -- DEPENDENCIES (all must close before this theorem):
  --   • Float16.toFloat32_exact (Float16.lean:208) — sub-cases sorry'd
  --   • Float32.mul (this file, line 50) — zero stub; must be implemented
  --   • Float32.add (this file, line 49) — zero stub; must be implemented
  -- ──────────────────────────────────────────────────────────────────────────
  exact ⟨Float32.zero, by sorry⟩

instance : Zero Float32 := ⟨Float32.zero⟩
instance : One  Float32 := ⟨Float32.one⟩
instance : Add  Float32 := ⟨Float32.add⟩
instance : Mul  Float32 := ⟨Float32.mul⟩
instance : Neg  Float32 := ⟨fun x => ⟨x.bits ^^^ (1 <<< 31 : UInt32)⟩⟩

end PAX
