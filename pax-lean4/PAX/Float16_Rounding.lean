-- PAX Float16 Rounding — IEEE-754 RNE Rounding Specification
-- Mathematically precise Round-to-Nearest-Even algorithm for Binary16
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import Mathlib.Data.Rat.Basic
import Mathlib.Tactic
import PAX.Float16

namespace PAX

open Rat Int

-- ═══════════════════════════════════════════════════════════════════════
-- FP16 REPRESENTABLE VALUE SET
-- ═══════════════════════════════════════════════════════════════════════

/-- The set of all rationals exactly representable as finite FP16 values.
    Excludes ±Inf and NaN (toRat returns none for those). -/
def FP16_Values : Set ℚ :=
  { q | ∃ (f : Float16), f.isFinite = true ∧ f.toRat = some q }

theorem zero_in_FP16_Values : (0 : ℚ) ∈ FP16_Values :=
  ⟨Float16.zero, by native_decide, by simp [Float16.toRat, Float16.isZero,
    Float16.zero, Float16.expBits, Float16.mantBits]⟩

-- ═══════════════════════════════════════════════════════════════════════
-- SPACING / ULP (ALIAS FOR ROUNDING PROOFS)
-- ═══════════════════════════════════════════════════════════════════════

/-- Spacing between adjacent FP16 values near x, i.e. the unit in last place.
    For subnormal/zero: 2^(-24)  (all subnormals have the same spacing).
    For normal exponent e ∈ [-14,15]: 2^(e-10).
    Named `spacing` (rather than `ulp`) to clarify role in rounding-error bounds:
      |round(x) - x| ≤ spacing(round(x)) / 2. -/
def spacing (x : Float16) : ℚ :=
  if x.isSubnormal || x.isZero then (2 : ℚ) ^ (-(24 : ℤ))
  else if x.isNormal then
    let e : ℤ := (x.expBits.val : ℤ) - 15
    (2 : ℚ) ^ (e - 10)
  else 0  -- Inf / NaN: spacing undefined, return 0

@[simp]
theorem spacing_eq_ulp (x : Float16) : spacing x = x.ulp := by
  simp [spacing, Float16.ulp]

theorem spacing_pos_of_finite {x : Float16} (hfin : x.isFinite = true) :
    0 < spacing x := by
  simp [spacing]
  by_cases hz : x.isZero
  · simp [hz, Float16.isFinite, Float16.isInf, Float16.isNaN] at *
    positivity
  · by_cases hs : x.isSubnormal
    · simp [hz, hs]; positivity
    · have hn : x.isNormal := by
        simp [Float16.isNormal, Float16.isZero] at hz
        simp [Float16.isSubnormal] at hs
        simp [Float16.isFinite, Float16.isInf, Float16.isNaN] at hfin
        omega
      simp [hs, hz, hn]
      positivity

-- ═══════════════════════════════════════════════════════════════════════
-- RNE HALF-INTEGER TIE-BREAKING HELPER
-- ═══════════════════════════════════════════════════════════════════════

/-- Apply the Round-to-Nearest-Even rule to an integer part and a fractional
    remainder.  `int_part` is the candidate mantissa; `frac ∈ [0, 1)` is the
    scaled remainder after subtracting `int_part`.
    • frac < 1/2 → round down  (return int_part)
    • frac > 1/2 → round up    (return int_part + 1)
    • frac = 1/2 → round to the even integer (ties-to-even) -/
def rne_round (int_part : ℤ) (frac : ℚ) : ℤ :=
  if frac < 1 / 2 then int_part
  else if frac > 1 / 2 then int_part + 1
  else  -- exact half: choose the even one
    if int_part % 2 = 0 then int_part else int_part + 1

theorem rne_round_down {ip : ℤ} {f : ℚ} (hf : f < 1 / 2) :
    rne_round ip f = ip := by
  simp [rne_round, hf]

theorem rne_round_up {ip : ℤ} {f : ℚ} (hf : f > 1 / 2) :
    rne_round ip f = ip + 1 := by
  simp [rne_round, hf, not_lt.mpr (le_of_lt hf)]

-- ═══════════════════════════════════════════════════════════════════════
-- MATHEMATICALLY-DEFINED RNE ROUNDING ALGORITHM
-- ═══════════════════════════════════════════════════════════════════════

/-- Round a rational number to the nearest representable FP16 value using
    Round-to-Nearest-Even (IEEE-754 default mode).

    Algorithm (for x ∈ [-65504, 65504] \ {0}):
      1. Determine sign bit s.
      2. Work with ax = |x|.
      3. Compute k = ⌊log₂(ax)⌋, clamped to the FP16 normal range [-14, 15].
         If ax < 2^(-14) the number is subnormal (exp field = 0).
      4. For normal: scale m = ax / 2^k, giving m ∈ [1, 2).
         Write m = 1 + frac where frac ∈ [0, 1).
         Mantissa bits = rne_round(⌊frac × 2^10⌋, (frac × 2^10 - ⌊frac × 2^10⌋)).
         Biased exponent field = k + 15 (∈ [1, 30]).
      5. For subnormal: scale m = ax × 2^24 ∈ [0, 1024).
         Mantissa bits = rne_round(⌊m⌋, m - ⌊m⌋); exp field = 0.
      6. Reconstruct UInt16 bits: sign | exp | mantissa.

    The `sorry` below covers the bit-level reconstruction which requires
    integer-arithmetic lemmas about floor, clamping, and UInt16 packing.
    The error-bound theorems below hold independently of this construction. -/
noncomputable def roundToFP16 (x : ℚ) : Float16 :=
  if x < -65504 then Float16.negInf
  else if x > 65504  then Float16.posInf
  else if x = 0      then Float16.zero
  else
    -- RNE algorithm: determine sign, work with |x|, then apply subnormal or normal path.
    let s : UInt16 := if x < 0 then (1 : UInt16) else (0 : UInt16)
    let ax : ℚ := |x|
    if ax < (2 : ℚ) ^ (-(14 : ℤ)) then
      -- Subnormal path: biased exponent = 0, mantissa = RNE(ax × 2^24)
      let scaled : ℚ := ax * (2 : ℚ) ^ (24 : ℤ)
      let ip : ℤ := ⌊scaled⌋
      let m : ℤ := rne_round ip (scaled - (ip : ℚ))
      ⟨(s <<< 15) ||| (m.toNat : UInt16)⟩
    else
      -- Normal path: compute k = ⌊log₂ ax⌋ via Nat.log on num/den with 1-step correction.
      -- For ax = p/q (reduced), Nat.log 2 p - Nat.log 2 q equals ⌊log₂ ax⌋ or is 1 too high;
      -- the conditional corrects the off-by-one when 2^k₀ > ax.
      let p : ℕ := ax.num.natAbs
      let q : ℕ := ax.den
      let k₀ : ℤ := (Nat.log 2 p : ℤ) - (Nat.log 2 q : ℤ)
      let k₀c : ℤ := if (2 : ℚ) ^ k₀ ≤ ax then k₀ else k₀ - 1
      let k : ℤ := max (-14 : ℤ) (min 15 k₀c)
      -- Scale to [1, 2) and extract the 10 fractional mantissa bits via RNE.
      let mscaled : ℚ := (ax / (2 : ℚ) ^ k - 1) * (2 : ℚ) ^ (10 : ℤ)
      let ip : ℤ := ⌊mscaled⌋
      let m : ℤ := rne_round ip (mscaled - (ip : ℚ))
      let e : UInt16 := ((k + 15).toNat : UInt16)
      ⟨(s <<< 15) ||| (e <<< 10) ||| (m.toNat : UInt16)⟩

/-- roundToFP16 produces negInf for x < -65504. -/
@[simp]
theorem roundToFP16_negInf {x : ℚ} (h : x < -65504) :
    roundToFP16 x = Float16.negInf := by
  simp [roundToFP16, h]

/-- roundToFP16 produces posInf for x > 65504. -/
@[simp]
theorem roundToFP16_posInf {x : ℚ} (h : x > 65504) :
    roundToFP16 x = Float16.posInf := by
  have hlo : ¬ (x < -65504) := by linarith
  simp [roundToFP16, hlo, h]

/-- roundToFP16 produces zero for x = 0. -/
@[simp]
theorem roundToFP16_zero : roundToFP16 0 = Float16.zero := by
  simp [roundToFP16]

-- ═══════════════════════════════════════════════════════════════════════
-- ERROR BOUND HELPER LEMMAS
-- ═══════════════════════════════════════════════════════════════════════

/-- Rounding error for a positive normal input is bounded by half the spacing.

    Proof sketch (sorry'd — requires interval arithmetic on ℤ/ℚ):
    For ax ∈ [2^k, 2^(k+1)) with k ∈ [-14, 15], the FP16 normal values in this
    interval form a grid of 1024 points spaced 2^(k-10) apart.
    RNE selects the closest grid point; at exact midpoints it chooses the one
    with even mantissa LSB.  In all cases:
      |rv - x| ≤ 2^(k-10) / 2 = spacing(round(x)) / 2. -/
lemma round_positive_normal_error (x : ℚ)
    (hpos : 0 < x)
    (hnorm : (2 : ℚ) ^ (-(14 : ℤ)) ≤ x)
    (hbnd : x ≤ 65504) :
    ∃ (rv : ℚ),
      (roundToFP16 x).toRat = some rv ∧
      |rv - x| ≤ spacing (roundToFP16 x) / 2 := by
  sorry
  -- Step 1: x > 0, x ≤ 65504, x ≥ 2^(-14) → roundToFP16 x is finite normal.
  -- Step 2: Let k = ⌊log₂ x⌋, k ∈ [-14, 15].
  --         Grid spacing = 2^(k-10); roundToFP16 picks the nearest grid point.
  -- Step 3: |rv - x| ≤ (grid spacing)/2 = 2^(k-10)/2 = spacing(round(x))/2.

/-- Rounding error for a positive subnormal input is bounded by half the spacing.

    Proof sketch (sorry'd):
    For ax ∈ (0, 2^(-14)), the subnormal FP16 values form a uniform grid from 0
    to (1023/1024) × 2^(-14) with step 2^(-24).
    RNE selects the closest; |error| ≤ 2^(-25) = spacing(round(x))/2. -/
lemma round_positive_subnormal_error (x : ℚ)
    (hpos : 0 < x)
    (hsubnorm : x < (2 : ℚ) ^ (-(14 : ℤ))) :
    ∃ (rv : ℚ),
      (roundToFP16 x).toRat = some rv ∧
      |rv - x| ≤ spacing (roundToFP16 x) / 2 := by
  sorry
  -- Step 1: x ∈ (0, 2^(-14)) → roundToFP16 x is finite subnormal.
  -- Step 2: Subnormal grid: {k × 2^(-24) | k ∈ [0, 1023]}, spacing = 2^(-24).
  -- Step 3: RNE picks nearest; max error = 2^(-24)/2 = 2^(-25).

-- ═══════════════════════════════════════════════════════════════════════
-- MAIN RNE ERROR BOUND THEOREM
-- ═══════════════════════════════════════════════════════════════════════

/-- **RNE rounding error bound (IEEE-754 §4.3.1)**:
    For every finite x ∈ [-65504, 65504],
      ∃ rv, roundToFP16(x).toRat = some rv  ∧
            |rv - x| ≤ spacing(roundToFP16(x)) / 2.

    Proof structure (cases on sign and magnitude):
      (a) x = 0        : error = 0, trivial.
      (b) x < 0        : symmetry — roundToFP16(-x) = neg(roundToFP16(x));
                         reduce to x > 0 cases.
      (c) 0 < x < 2^(-14) : subnormal case → round_positive_subnormal_error.
      (d) 2^(-14) ≤ x ≤ 65504 : normal case → round_positive_normal_error. -/
theorem round_error_bound :
    ∀ x : ℚ, x ∈ Set.Icc (-(65504 : ℚ)) 65504 →
    ∃ (rv : ℚ),
      (roundToFP16 x).toRat = some rv ∧
      |rv - x| ≤ spacing (roundToFP16 x) / 2 := by
  intro x ⟨hlo, hhi⟩
  -- Case split on x
  by_cases hz : x = 0
  · -- (a) x = 0: roundToFP16 0 = Float16.zero, error = 0
    subst hz
    simp [roundToFP16_zero, Float16.toRat, Float16.isZero, Float16.zero,
          Float16.expBits, Float16.mantBits, spacing, Float16.isSubnormal]
    norm_num [Float16.toRat, Float16.isZero, spacing, Float16.isSubnormal]
  · by_cases hneg : x < 0
    · -- (b) x < 0: reduce to positive case by RNE sign symmetry
      sorry
      -- roundToFP16 x = neg (roundToFP16 (-x)) since RNE commutes with negation
      -- |rv - x| = |-rv' - (-(-x))| = |rv' - (-x)| ≤ spacing/2 by positive case
    · -- x > 0
      push_neg at hneg
      have hpos : 0 < x := lt_of_le_of_ne (le_of_not_lt hneg) (Ne.symm hz)
      by_cases hsub : x < (2 : ℚ) ^ (-(14 : ℤ))
      · -- (c) subnormal
        exact round_positive_subnormal_error x hpos hsub
      · -- (d) normal
        push_neg at hsub
        exact round_positive_normal_error x hpos hsub hhi

-- ═══════════════════════════════════════════════════════════════════════
-- COMPLETE ROUNDING SPECIFICATION
-- ═══════════════════════════════════════════════════════════════════════

/-- **Full roundToFP16 specification**: collects all cases into one theorem.
    This is the master correctness statement for the RNE algorithm. -/
theorem roundToFP16_spec (x : ℚ) :
    (x < -65504 → roundToFP16 x = Float16.negInf) ∧
    (65504 < x  → roundToFP16 x = Float16.posInf) ∧
    (x = 0      → roundToFP16 x = Float16.zero)   ∧
    (x ∈ Set.Icc (-(65504 : ℚ)) 65504 →
      ∃ rv : ℚ,
        (roundToFP16 x).toRat = some rv ∧
        |rv - x| ≤ spacing (roundToFP16 x) / 2) := by
  refine ⟨roundToFP16_negInf, roundToFP16_posInf, roundToFP16_zero, ?_⟩
  exact round_error_bound x

/-- roundToFP16 always yields a value whose toRat lives in FP16_Values
    (for inputs in the finite range). -/
theorem roundToFP16_mem_FP16_Values (x : ℚ)
    (hbnd : x ∈ Set.Icc (-(65504 : ℚ)) 65504) :
    ∃ q, (roundToFP16 x).toRat = some q ∧ q ∈ FP16_Values := by
  sorry
  -- roundToFP16 x is finite by construction (no overflow in the interval).
  -- Then toRat (roundToFP16 x) = some q for some q.
  -- By definition of FP16_Values, q ∈ FP16_Values.

end PAX
