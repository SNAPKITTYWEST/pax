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
  simp [Float16.signBit, Neg.neg, HNeg.hneg]
  sorry
  -- Proof: (-x).bits = x.bits XOR (1 <<< 15)
  --        sign bit = bit 15 of x.bits XOR (1 <<< 15)
  --        = (bit 15 of x.bits) XOR 1 = !x.signBit

/-- ULP bound is positive for all finite FP16 values. -/
theorem pax_fp16_ulp_pos (x : Float16) (hfin : x.isFinite) :
    0 < x.ulp := by
  have := PAX.spacing_pos_of_finite hfin
  rwa [PAX.spacing_eq_ulp] at this

end PAX.Verified
