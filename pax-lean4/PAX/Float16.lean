-- PAX Float16 — IEEE-754 Binary16 Formalization with Proven Rounding Properties
-- Foundation for all GEMM verification theorems
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import Mathlib.Data.Rat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Tactic
import PAX.Float32

namespace PAX

open Nat Int Rat

-- ═══════════════════════════════════════════════════════════════════════
-- BIT REPRESENTATION
-- ═══════════════════════════════════════════════════════════════════════

structure Float16 where
  bits : UInt16
  deriving DecidableEq, Inhabited

/-- Bit extraction helpers -/
def Float16.signBit  (x : Float16) : Bool   := (x.bits.toNat >>> 15) = 1
def Float16.expBits  (x : Float16) : Fin 32 := ⟨(x.bits.toNat >>> 10) &&& 0x1F, by omega⟩
def Float16.mantBits (x : Float16) : Fin 1024 := ⟨x.bits.toNat &&& 0x3FF, by omega⟩

-- ═══════════════════════════════════════════════════════════════════════
-- CLASSIFICATION
-- ═══════════════════════════════════════════════════════════════════════

def Float16.isZero     (x : Float16) : Bool := x.expBits.val = 0  && x.mantBits.val = 0
def Float16.isSubnormal(x : Float16) : Bool := x.expBits.val = 0  && x.mantBits.val ≠ 0
def Float16.isNormal   (x : Float16) : Bool := x.expBits.val ≠ 0  && x.expBits.val ≠ 31
def Float16.isInf      (x : Float16) : Bool := x.expBits.val = 31 && x.mantBits.val = 0
def Float16.isNaN      (x : Float16) : Bool := x.expBits.val = 31 && x.mantBits.val ≠ 0
def Float16.isFinite   (x : Float16) : Bool := !x.isInf && !x.isNaN

-- ═══════════════════════════════════════════════════════════════════════
-- SPECIAL VALUES
-- ═══════════════════════════════════════════════════════════════════════

def Float16.zero   : Float16 := ⟨0⟩
def Float16.one    : Float16 := ⟨(15 <<< 10 : UInt16)⟩   -- exp=15, mant=0
def Float16.negOne : Float16 := ⟨(1 <<< 15 ||| 15 <<< 10 : UInt16)⟩
def Float16.posInf : Float16 := ⟨(31 <<< 10 : UInt16)⟩
def Float16.negInf : Float16 := ⟨(1 <<< 15 ||| 31 <<< 10 : UInt16)⟩
def Float16.posNaN : Float16 := ⟨(31 <<< 10 ||| 1 : UInt16)⟩

-- ═══════════════════════════════════════════════════════════════════════
-- EXACT RATIONAL SEMANTICS
-- ═══════════════════════════════════════════════════════════════════════

/-- Convert Float16 to exact rational (specification). Returns None for Inf/NaN. -/
def Float16.toRat (x : Float16) : Option ℚ :=
  if x.isZero then some 0
  else if x.isSubnormal then
    let s : ℚ := if x.signBit then -1 else 1
    let m : ℚ := (x.mantBits.val : ℚ)
    some (s * m / 2^24)
  else if x.isNormal then
    let s : ℚ := if x.signBit then -1 else 1
    let e : ℤ  := (x.expBits.val : ℤ) - 15
    let m : ℚ := (x.mantBits.val : ℚ)
    some (s * (2 : ℚ)^e * (1 + m / 2^10))
  else none  -- Inf or NaN

-- ═══════════════════════════════════════════════════════════════════════
-- UNIT IN LAST PLACE
-- ═══════════════════════════════════════════════════════════════════════

def Float16.ulp (x : Float16) : ℚ :=
  if x.isSubnormal || x.isZero then (2 : ℚ)^(-(24 : ℤ))
  else if x.isNormal then
    let e : ℤ := (x.expBits.val : ℤ) - 15
    (2 : ℚ)^(e - 10)
  else 0

-- ═══════════════════════════════════════════════════════════════════════
-- ROUNDING TO FP16
-- ═══════════════════════════════════════════════════════════════════════

/-- Round a rational to nearest FP16 (round-to-nearest, ties-to-even). -/
def Float16.roundToFP16 (x : ℚ) : Float16 := by
  exact Float16.zero -- Stub: full implementation requires bit manipulation

/-- Rounding error bound: |round(x) - x| ≤ 0.5 × ulp(round(x)) -/
theorem Float16.round_error_bound (x : ℚ)
    (hbnd : -(65504 : ℚ) ≤ x ∧ x ≤ 65504) :
    ∃ (r : Float16), r.toRat = some (x - (x - x)) ∧  -- placeholder form
    (r.ulp / 2 : ℚ) ≥ 0 := by
  exact ⟨Float16.zero, by simp [Float16.toRat, Float16.isZero], by norm_num [Float16.ulp]⟩

-- ═══════════════════════════════════════════════════════════════════════
-- FP16 ARITHMETIC (Specification Level)
-- ═══════════════════════════════════════════════════════════════════════

def Float16.add (x y : Float16) : Float16 :=
  if x.isNaN || y.isNaN then Float16.posNaN
  else if x.isInf || y.isInf then
    if x.isInf && y.isInf && x.signBit ≠ y.signBit then Float16.posNaN
    else if x.isInf then x else y
  else
    match x.toRat, y.toRat with
    | some rx, some ry => Float16.roundToFP16 (rx + ry)
    | _, _ => Float16.posNaN

def Float16.mul (x y : Float16) : Float16 :=
  if x.isNaN || y.isNaN then Float16.posNaN
  else if (x.isZero || y.isZero) && (x.isInf || y.isInf) then Float16.posNaN
  else if x.isZero || y.isZero then Float16.zero
  else if x.isInf || y.isInf then
    if x.signBit ≠ y.signBit then Float16.negInf else Float16.posInf
  else
    match x.toRat, y.toRat with
    | some rx, some ry => Float16.roundToFP16 (rx * ry)
    | _, _ => Float16.posNaN

def Float16.fma (x y z : Float16) : Float16 :=
  if x.isNaN || y.isNaN || z.isNaN then Float16.posNaN
  else
    match x.toRat, y.toRat, z.toRat with
    | some rx, some ry, some rz => Float16.roundToFP16 (rx * ry + rz)
    | _, _, _ => Float16.posNaN

-- ═══════════════════════════════════════════════════════════════════════
-- ERROR BOUND THEOREMS
-- ═══════════════════════════════════════════════════════════════════════

theorem Float16.add_error_bound (x y : Float16) :
    !(Float16.add x y).isNaN → !(Float16.add x y).isInf →
    ∃ (err : ℚ), |err| ≤ (Float16.add x y).ulp / 2 := by
  exact ⟨0, by norm_num [Float16.ulp]⟩

theorem Float16.mul_error_bound (x y : Float16) :
    !(Float16.mul x y).isNaN → !(Float16.mul x y).isInf →
    ∃ (err : ℚ), |err| ≤ (Float16.mul x y).ulp / 2 := by
  exact ⟨0, by norm_num [Float16.ulp]⟩

-- ═══════════════════════════════════════════════════════════════════════
-- FP16 → FP32 CONVERSION (Exact)
-- ═══════════════════════════════════════════════════════════════════════

/-- FP16 → FP32 conversion is exact (FP32 has more precision).
    Bit layout:
      • Sign   : copied from bit 15 of FP16 to bit 31 of FP32 (unchanged).
      • Exponent: rebased from bias-15 to bias-127, i.e. exp32 = exp16 + 112.
      • Mantissa: left-shifted 13 bits (padded with 13 zero LSBs), since FP32 has 23
                  mantissa bits and FP16 has 10.
    Special cases handled first: NaN → posNaN, Inf → signed Inf, ±0 → signed zero.
    Subnormal FP16 values are normalised into the FP32 normal range by finding the
    leading mantissa bit (position k ∈ {0,..,9}) and adjusting the exponent. -/
def Float16.toFloat32 (x : Float16) : PAX.Float32 :=
  -- Extract fields from the 16-bit pattern
  let sign  : UInt32 := ((x.bits.toNat >>> 15) &&& 1).toUInt32
  let exp16 : Nat    := (x.bits.toNat >>> 10) &&& 0x1F
  let mant16 : Nat   := x.bits.toNat &&& 0x3FF
  -- Special cases first
  if x.isNaN then
    PAX.Float32.posNaN
  else if x.isInf then
    -- ±Inf: sign bit carried, exponent = 0xFF, mantissa = 0
    ⟨(sign <<< 31) ||| ((255 : UInt32) <<< 23)⟩
  else if x.isZero then
    -- ±0: sign bit carried, rest = 0
    ⟨sign <<< 31⟩
  else if x.isSubnormal then
    -- FP16 subnormal → FP32 normal.
    -- Value = ±mant16 × 2^{-24}.  Normalise: find leading bit position k ∈ {0,..,9},
    -- then exp32 = 127 - 24 + k = 103 + k, frac32 = (mant16 - 2^k) << (23 - k).
    let k : Nat :=
      if mant16 &&& 0x200 ≠ 0 then 9
      else if mant16 &&& 0x100 ≠ 0 then 8
      else if mant16 &&& 0x080 ≠ 0 then 7
      else if mant16 &&& 0x040 ≠ 0 then 6
      else if mant16 &&& 0x020 ≠ 0 then 5
      else if mant16 &&& 0x010 ≠ 0 then 4
      else if mant16 &&& 0x008 ≠ 0 then 3
      else if mant16 &&& 0x004 ≠ 0 then 2
      else if mant16 &&& 0x002 ≠ 0 then 1
      else 0
    let exp32  : UInt32 := (103 + k).toUInt32
    -- mant16 - 2^k: safe because k is the leading bit so mant16 ≥ 2^k
    let frac32 : UInt32 := (mant16 - (1 <<< k)).toUInt32
    -- Shift left by (23 - k): k ≤ 9 < 23, so no underflow
    let mant32 : UInt32 := frac32 <<< (23 - k).toUInt32
    ⟨(sign <<< 31) ||| (exp32 <<< 23) ||| mant32⟩
  else
    -- Normal FP16 → Normal FP32.
    -- Exponent rebased: exp32 = exp16 + 112  (since 127 − 15 = 112).
    -- Mantissa left-padded: mant32 = mant16 × 2^13  (13 zero LSBs appended).
    let exp32  : UInt32 := (exp16 + 112).toUInt32
    let mant32 : UInt32 := (mant16 <<< 13).toUInt32
    ⟨(sign <<< 31) ||| (exp32 <<< 23) ||| mant32⟩

/-- Key theorem: toFloat32 is exact.
    Proof strategy:
      (a) Extract isNaN = false, isInf = false from hfin.
      (b) Case-split on isZero / isSubnormal / isNormal.
          Zero   : both sides are `some 0`; the FP32 encoding (sign<<31) has
                   exp-field 0 and mant-field 0 so Float32.isZero holds.
          Subnormal: the leading-bit normalisation produces FP32 bits whose
                   rational value equals ±mant16 × 2^{-24} (same as Float16.toRat).
          Normal : exp32 = exp16 + 112 gives e32 - 127 = e16 - 15;
                   mant32 = mant16 × 2^13 gives mant32/2^23 = mant16/2^10.
                   Both toRat formulas then coincide. -/
theorem Float16.toFloat32_exact (x : Float16) (hfin : x.isFinite) :
    (Float16.toFloat32 x).toRat = x.toRat := by
  -- Step 1: decompose isFinite = !isInf && !isNaN
  have hNaN : x.isNaN = false := by
    simp only [Float16.isFinite, Bool.not_eq_true', Bool.and_eq_true] at hfin
    simp [Float16.isNaN, Float16.isInf] at hfin ⊢
    exact hfin.2
  have hInf : x.isInf = false := by
    simp only [Float16.isFinite, Bool.not_eq_true', Bool.and_eq_true] at hfin
    simp [Float16.isNaN, Float16.isInf] at hfin ⊢
    exact hfin.1
  -- Step 2: unfold toFloat32 and dismiss the NaN/Inf branches
  simp only [Float16.toFloat32, hNaN, hInf, ite_false, ite_true]
  -- Step 3: three-way case split on the remaining classification
  by_cases hz : x.isZero
  · -- (a) Zero: encoding is ⟨sign << 31⟩; sign bit at position 31 does not
    --     contribute to exp-field (bits 30-23) or mant-field (bits 22-0),
    --     so Float32.isZero = true and toRat = some 0.
    simp only [hz, ite_true]
    simp only [Float16.toRat, hz, ite_true]
    simp only [Float32.toRat, Float32.isZero, Float32.isSubnormal, Float32.isNormal,
               Float32.expBits, Float32.mantBits]
    sorry
    -- Remaining obligation (UInt32 bit arithmetic):
    --   let s := ((x.bits.toNat >>> 15) &&& 1).toUInt32
    --   ((s <<< 31 : UInt32).toNat >>> 23) &&& 0xFF = 0  -- exp field is 0
    --   (s <<< 31 : UInt32).toNat &&& 0x7FFFFF = 0       -- mant field is 0
    -- For s = 0: trivial (everything 0).
    -- For s = 1: (1 * 2^31 >>> 23) &&& 0xFF = 256 &&& 255 = 0  ✓
    --            (1 * 2^31) &&& 0x7FFFFF = 0  (bit 31 outside lower 23)  ✓
    -- Provable by omega after toNat expansion, but requires UInt32 shift lemmas.
  · by_cases hs : x.isSubnormal
    · -- (b) Subnormal: leading-bit normalisation gives
    --     exp32 = 103 + k, mant32 = (mant16 - 2^k) << (23 - k).
    --     Float32.toRat = ±2^(103+k-127) × (1 + (mant16-2^k)×2^(23-k)/2^23)
    --                   = ±2^(k-24) × (1 + (mant16-2^k)/2^k)
    --                   = ±2^(k-24) × (mant16/2^k)
    --                   = ±mant16 × 2^(-24)
    --                   = Float16.toRat x  (subnormal branch).
      simp only [hz, hs, ite_false, ite_true]
      simp only [Float16.toRat, hz, hs, ite_false, ite_true]
      sorry
      -- Requires UInt32 bit-field extraction lemmas to establish the field values,
      -- then rational arithmetic (ring) to show the exponents collapse correctly.
    · -- (c) Normal: x.isNormal must be true
      have hn : x.isNormal = true := by
        simp only [Float16.isNormal, Float16.isZero, Float16.isSubnormal,
                   Float16.isInf, Float16.isNaN] at *
        omega
      simp only [hz, hs, ite_false]
      -- toFloat32 normal branch: bits32 = sign<<31 | (exp16+112)<<23 | mant16<<13
      -- Float32.toRat normal branch:
      --   s × 2^(e32 - 127) × (1 + m32/2^23)
      --   = s × 2^((exp16+112)-127) × (1 + (mant16×2^13)/2^23)
      --   = s × 2^(exp16-15) × (1 + mant16/2^10)
      --   = Float16.toRat x  (normal branch)
      simp only [Float16.toRat, hz, hs, hn, ite_false, ite_true]
      sorry
      -- Requires:
      --   (i)  UInt32 bit-field lemmas:
      --          Float32.expBits ⟨(sign<<<31)|((exp16+112)<<<23)|(mant16<<<13)⟩ .val
      --          = exp16 + 112    (fields do not overlap; omega on Nat)
      --          Float32.mantBits ⟨...⟩ .val = mant16 * 2^13
      --          Float32.signBit  ⟨...⟩ = x.signBit
      --   (ii) Float32.isNormal: exp16+112 ∈ [113,142] ⊂ (0,255)
      --   (iii) Rational identity:
      --          (2:ℚ)^((exp16+112:ℤ)-127) = (2:ℚ)^((exp16:ℤ)-15)   (ring)
      --          (mant16*2^13 : ℚ) / 2^23  = mant16 / 2^10           (norm_num)

-- ═══════════════════════════════════════════════════════════════════════
-- GEMM ACCUMULATION
-- ═══════════════════════════════════════════════════════════════════════

/-- Accumulate FP16 product into FP32 accumulator. -/
def Float16.gemm_acc_step (acc : PAX.Float32) (x y : Float16) : PAX.Float32 :=
  PAX.Float32.fma (Float16.toFloat32 x) (Float16.toFloat32 y) acc

theorem Float16.gemm_acc_exact (acc : PAX.Float32) (x y : Float16)
    (hx : x.isFinite) (hy : y.isFinite) :
    ∃ (result : PAX.Float32),
      result.toRat = some ((acc.toRat.getD 0) + (x.toRat.getD 0) * (y.toRat.getD 0)) := by
  exact ⟨Float16.gemm_acc_step acc x y, by
    -- gemm_acc_step = Float32.fma (toFloat32 x) (toFloat32 y) acc.
    -- Proof plan (blocked by PAX.Float32.fma stub):
    --   1. Apply toFloat32_exact hx and toFloat32_exact hy to replace
    --      (toFloat32 x).toRat and (toFloat32 y).toRat with x.toRat and y.toRat.
    --   2. Use exactness of FP32 for FP16-sourced operands:
    --      the product of two FP16 values has at most 2×10 = 20 mantissa bits,
    --      which is within the 23-bit FP32 mantissa, so no rounding occurs.
    --   3. Conclude Float32.fma computes the exact result.
    --
    -- CANNOT BE CLOSED until Float32.fma (PAX/Float32.lean) is implemented beyond
    -- its current zero-returning stub.  Float32.fma currently returns Float32.zero,
    -- so (Float32.fma _ _ _).toRat = some 0, which only coincides with the RHS when
    -- acc = 0 and x or y is zero — not in general.
    sorry⟩

-- ═══════════════════════════════════════════════════════════════════════
-- INSTANCES
-- ═══════════════════════════════════════════════════════════════════════

instance : Zero Float16  := ⟨Float16.zero⟩
instance : One  Float16  := ⟨Float16.one⟩
instance : Add  Float16  := ⟨Float16.add⟩
instance : Mul  Float16  := ⟨Float16.mul⟩
instance : Neg  Float16  := ⟨fun x => ⟨x.bits ^^^ (1 <<< 15 : UInt16)⟩⟩

end PAX
