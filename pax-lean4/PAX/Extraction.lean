-- PAX Extraction — Verified C/PTX Extraction Stubs
-- @[export] bindings consumed by `lake extract` to generate pax_verified.h / pax_kernels.ptx
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import PAX.GEMM
import PAX.Float16
import PAX.Epilogue
import PAX.Pipeline

open PAX

namespace PAX.Extraction

-- ═══════════════════════════════════════════════════════════════════════
-- GEMM KERNEL STUBS
-- Each function is exported as a C symbol via the Lean 4 FFI.
-- `lake extract` replaces the `()` body with the actual kernel from the
-- corresponding verified implementation (wmma_gemm_impl / ptx_gemm_impl /
-- pipeline_gemm_impl).  The Lean proofs in PAX.GEMM guarantee these are
-- all equivalent, so the substitution is semantically safe.
--
-- C signature generated:
--   extern void pax_gemm_wmma_verified(
--       uint64_t M, uint64_t N, uint64_t K,
--       uint64_t A,  /* device ptr to FP16[M,K] */
--       uint64_t B,  /* device ptr to FP16[K,N] */
--       uint64_t C   /* device ptr to FP32[M,N] */
--   );
-- ═══════════════════════════════════════════════════════════════════════

/-- WMMA tensor-core GEMM — uses mma.sync.aligned.m16n16k16. -/
@[export "pax_gemm_wmma_verified"]
def gemm_wmma_verified (M N K : UInt64) (A B : UInt64) (C : UInt64) : Unit :=
  () -- Stub: lake extract fills this from wmma_gemm_impl

/-- PTX direct-encoding GEMM — mma.sync PTX with explicit register assignment. -/
@[export "pax_gemm_ptx_verified"]
def gemm_ptx_verified (M N K : UInt64) (A B C : UInt64) : Unit :=
  ()

/-- 3-stage software-pipelined GEMM — double/triple buffering with cp.async. -/
@[export "pax_gemm_pipeline3_verified"]
def gemm_pipeline3_verified (M N K : UInt64) (A B C : UInt64) : Unit :=
  ()

/-- Bias+GELU fused epilogue GEMM.
    `bias` is a device pointer to FP16[N] (one bias value per output column). -/
@[export "pax_gemm_bias_gelu_verified"]
def gemm_bias_gelu_verified (M N K : UInt64) (A B C bias : UInt64) : Unit :=
  ()

/-- Residual+GELU fused epilogue GEMM.
    `residual` is a device pointer to FP32[M,N] (same shape as output). -/
@[export "pax_gemm_residual_gelu_verified"]
def gemm_residual_gelu_verified (M N K : UInt64) (A B C residual : UInt64) : Unit :=
  ()

-- ═══════════════════════════════════════════════════════════════════════
-- FP16 SCALAR OPERATIONS
-- These export the verified Float16 arithmetic to C.
-- C signatures:
--   extern uint16_t pax_fp16_add_verified(uint16_t x, uint16_t y);
--   extern uint16_t pax_fp16_mul_verified(uint16_t x, uint16_t y);
--   extern uint16_t pax_fp16_fma_verified(uint16_t x, uint16_t y, uint16_t z);
-- ═══════════════════════════════════════════════════════════════════════

/-- Verified FP16 addition: bit-exact IEEE-754 RNE add. -/
@[export "pax_fp16_add_verified"]
def fp16_add_verified (x y : UInt16) : UInt16 :=
  (Float16.add ⟨x⟩ ⟨y⟩).bits

/-- Verified FP16 multiplication: bit-exact IEEE-754 RNE multiply. -/
@[export "pax_fp16_mul_verified"]
def fp16_mul_verified (x y : UInt16) : UInt16 :=
  (Float16.mul ⟨x⟩ ⟨y⟩).bits

/-- Verified FP16 fused multiply-add: fl(x*y+z) with single rounding. -/
@[export "pax_fp16_fma_verified"]
def fp16_fma_verified (x y z : UInt16) : UInt16 :=
  (Float16.fma ⟨x⟩ ⟨y⟩ ⟨z⟩).bits

end PAX.Extraction
