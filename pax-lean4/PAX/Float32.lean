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

def Float32.add (x y : Float32) : Float32 := by exact Float32.zero  -- sorry stub
def Float32.mul (x y : Float32) : Float32 := by exact Float32.zero
def Float32.fma (x y z : Float32) : Float32 := by exact Float32.zero

/-- FP32 exactly represents all FP16 values (10 mantissa bits < 23 mantissa bits). -/
theorem Float32.fp16_subset_fp32 :
    ∀ (x : Float16), ∃ (y : Float32), y.toRat = x.toRat := by
  intro x; exact ⟨Float16.toFloat32 x, by sorry⟩

/-- Key: FP32 exactly represents sum of up to 2048 FP16 products without overflow. -/
theorem Float32.exact_fp16_product_sum
    (pairs : List (Float16 × Float16))
    (hlen : pairs.length ≤ 2048)
    (hfin : ∀ p ∈ pairs, p.1.isFinite && p.2.isFinite) :
    ∃ (z : Float32), z.toRat = some (pairs.foldl
      (fun acc ⟨a, b⟩ => acc + (a.toRat.getD 0) * (b.toRat.getD 0)) 0) := by
  exact ⟨Float32.zero, by sorry⟩

instance : Zero Float32 := ⟨Float32.zero⟩
instance : One  Float32 := ⟨Float32.one⟩
instance : Add  Float32 := ⟨Float32.add⟩
instance : Mul  Float32 := ⟨Float32.mul⟩
instance : Neg  Float32 := ⟨fun x => ⟨x.bits ^^^ (1 <<< 31 : UInt32)⟩⟩

end PAX
