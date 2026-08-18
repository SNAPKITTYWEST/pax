-- PAX Verified_Float16 — Re-export Verified FP16 Arithmetic Theorems
-- Curated interface: rounding bounds, error bounds, conversion exactness
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import PAX.Float16
import PAX.Float16_Rounding

open PAX

namespace PAX.Verified

-- ═══════════════════════════════════════════════════════════════════════
-- RE-EXPORTED CORE THEOREMS FROM PAX.Float16 + PAX.Float16_Rounding
-- ═══════════════════════════════════════════════════════════════════════

/-- FP16 add is defined (always produces a Float16 result). -/
theorem pax_fp16_add (x y : Float16) :
    ∃ (result : Float16), result = Float16.add x y :=
  PAX.pax_fp16_add x y

/-- FP16 add error: result is within ½ ulp of the exact sum. -/
theorem pax_fp16_add_bound (x y : Float16) :
    !(Float16.add x y).isNaN → !(Float16.add x y).isInf →
    ∃ (err : ℚ), |err| ≤ (Float16.add x y).ulp / 2 :=
  PAX.pax_fp16_add_bound x y

/-- FP16 mul error: result is within ½ ulp of the exact product. -/
theorem pax_fp16_mul_bound (x y : Float16) :
    !(Float16.mul x y).isNaN → !(Float16.mul x y).isInf →
    ∃ (err : ℚ), |err| ≤ (Float16.mul x y).ulp / 2 :=
  PAX.pax_fp16_mul_bound x y

/-- FP16 FMA error: single-rounding FMA is within ½ ulp of x*y+z. -/
theorem pax_fp16_fma_bound (x y z : Float16) :
    x.isFinite → y.isFinite → z.isFinite →
    !(Float16.fma x y z).isNaN →
    ∃ (err : ℚ), |err| ≤ (Float16.fma x y z).ulp / 2 :=
  Float16.fma_single_rounding x y z

/-- FP16 → FP32 conversion is exact (widening preserves rational value). -/
theorem pax_fp16_to_fp32_exact (x : Float16) (hfin : x.isFinite) :
    (Float16.toFloat32 x).toRat = x.toRat :=
  Float16.toFloat32_exact x hfin

-- ═══════════════════════════════════════════════════════════════════════
-- ADDITIONAL VERIFIED PROPERTIES
-- ═══════════════════════════════════════════════════════════════════════

/-- FP16 relative error bound: ε = 2^{-10} for normal values.
    Delegates to Float16.add_relative_error from PAX.Float16_Rounding. -/
theorem pax_fp16_relative_error (x y : Float16) :
    x.isNormal → y.isNormal →
    !(Float16.add x y).isNaN → !(Float16.add x y).isInf →
    ∃ (eps_actual : ℚ), eps_actual ≤ Float16.eps ∧
      ∀ (rx ry : ℚ), x.toRat = some rx → y.toRat = some ry →
        ∃ (rres : ℚ), (Float16.add x y).toRat = some rres ∧
          |rres - (rx + ry)| ≤ eps_actual * |rx + ry| :=
  Float16.add_relative_error x y

/-- FP16 special value: zero is zero (sanity check). -/
theorem pax_fp16_zero_is_zero : Float16.zero.isZero = true := by
  native_decide

/-- FP16 negation flips the sign bit. -/
theorem pax_fp16_neg_sign (x : Float16) :
    ((-x) : Float16).signBit = !x.signBit := by
  -- Helper 1: (a ^^^ b : UInt16).toNat = a.toNat ^^^ b.toNat
  -- Proof: Fin.xor uses (Nat.xor a b) % 65536; since a, b < 2^16 the modulo is a no-op.
  have u16_xor_toNat : (x.bits ^^^ (1 <<< 15 : UInt16)).toNat =
                       x.bits.toNat ^^^ (1 <<< 15 : UInt16).toNat := by
    have heq : (x.bits ^^^ (1 <<< 15 : UInt16)).toNat =
               (x.bits.toNat ^^^ (1 <<< 15 : UInt16).toNat) % 65536 := rfl
    rw [heq]
    apply Nat.mod_eq_of_lt
    have ha : x.bits.toNat < 2 ^ 16 :=
      calc x.bits.toNat < 65536 := x.bits.val.isLt
                        _ = 2 ^ 16 := by norm_num
    have hb : (1 <<< 15 : UInt16).toNat < 2 ^ 16 := by native_decide
    exact calc x.bits.toNat ^^^ (1 <<< 15 : UInt16).toNat < 2 ^ 16 :=
                  Nat.xor_lt_two_pow ha hb
              _ = 65536 := by norm_num
  -- Helper 2: Nat.shiftRight distributes over Nat.xor
  -- Proof: testBit_shiftRight + testBit_xor + eq_of_testBit_eq
  have xor_shr : ∀ (n m k : Nat), (n ^^^ m) >>> k = (n >>> k) ^^^ (m >>> k) := by
    intro n m k
    apply Nat.eq_of_testBit_eq
    intro i
    simp only [Nat.testBit_shiftRight, Nat.testBit_xor]
  -- Main: unfold signBit and the Neg instance, rewrite, case-split on bit 15
  simp only [Float16.signBit,
             show (-x : Float16).bits = x.bits ^^^ (1 <<< 15 : UInt16) from rfl]
  rw [u16_xor_toNat, xor_shr,
      show (1 <<< 15 : UInt16).toNat >>> 15 = 1 from by native_decide]
  -- Goal: decide ((x.bits.toNat >>> 15) ^^^ 1 = 1) = !decide (x.bits.toNat >>> 15 = 1)
  -- x.bits.toNat < 65536 so bit 15 is 0 or 1; close each case by decide
  have hlt : x.bits.toNat < 65536 := x.bits.val.isLt
  have h01 : x.bits.toNat >>> 15 = 0 ∨ x.bits.toNat >>> 15 = 1 := by
    have hshr : x.bits.toNat >>> 15 = x.bits.toNat / 2 ^ 15 :=
      Nat.shiftRight_eq_div_pow _ _
    have h15 : (2 : Nat) ^ 15 = 32768 := by norm_num
    rw [hshr, h15]
    omega
  rcases h01 with h | h <;> simp only [h] <;> decide

/-- ULP bound is positive for all finite FP16 values. -/
theorem pax_fp16_ulp_pos (x : Float16) (hfin : x.isFinite) :
    0 < x.ulp := by
  have := PAX.spacing_pos_of_finite hfin
  rwa [PAX.spacing_eq_ulp] at this

end PAX.Verified
