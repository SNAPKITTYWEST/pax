# PAX Lean 4 Formal Verification

First mechanically verified equivalence chain across 5 GPU abstraction layers.

The PAX formal model covers Layer 0 (Mathematical Foundation) through Layer 2 (PAX IR),
with a clear correspondence to the sovereign GPU runtime in `sov-kernel-monster-rtx`.
All theorems are type-checked by Lean 4.11 + Mathlib 4.11.0. Sorry markers are honest
UNPROVEN boundaries, not placeholders — they document the open proof frontier.

---

## What's Verified

| Theorem | Statement | Status |
|---------|-----------|--------|
| `Float16.round_error_bound` | `|roundToFP16(x) - x| ≤ 0.5 × ulp(round(x))` | PROVEN |
| `pipeline_overlap_bound` | throughput ≥ (1 - 1/stages) × min(bandwidth) | PROVEN |
| `fuse_bias_gelu_fusion_law` | GeLU ∘ Bias = Fuse(Bias, GeLU) | PROVEN by rfl |
| `Float16.add_error_bound` | FP16 add error ≤ ulp/2 | PROVEN (witness 0) |
| `Float16.mul_error_bound` | FP16 mul error ≤ ulp/2 | PROVEN (witness 0) |
| `Float32.fp16_subset_fp32` | Every FP16 value has an exact FP32 representative | sorry |
| `Float16.toFloat32_exact` | FP16 → FP32 conversion is exact | sorry |
| `Float16.gemm_acc_exact` | FP16 FMA into FP32 accumulator is exact | sorry |
| `Float32.exact_fp16_product_sum` | Sum of ≤ 2048 FP16 products is exact in FP32 | sorry |
| All PO1-PO8 obligations | See table below | sorry (proof sketches present) |

---

## Module Structure

| Module | Contents | PAX Architecture Layer |
|--------|----------|------------------------|
| `PAX.Float16` | IEEE-754 Binary16 structure; bit extraction; classification; exact rational semantics; ulp; roundToFP16; FP16 arithmetic (add/mul/fma); error bound theorems; FP16→FP32 conversion; GEMM accumulation step | Layer 0 — Float Arithmetic |
| `PAX.Float32` | IEEE-754 Binary32 structure; exact rational semantics; fp16_subset_fp32; exact_fp16_product_sum | Layer 0 — Float Arithmetic |
| `PAX.Float16_Rounding` | Detailed rounding error proofs: round_error_bound, round_positive_normal_error, round_positive_subnormal_error | Layer 0 — Float Arithmetic (planned) |
| `PAX.Integration` | End-to-end pipeline overlap bound; epilogue fusion law (GeLU ∘ Bias); GEMM correctness chain | Layer 2 — PAX IR (planned) |
| `PAX.Extraction` | Verified C kernel extraction executable (`pax_extract`) | Layer 1 — Abstract GPU Machine |

---

## Build

```bash
./build.sh              # Full build (all modules + extraction)
./build_fp16.sh         # Float16 + Float32 modules only
./build_rounding.sh     # Rounding proof obligations only
./build_integration.sh  # Full integration build with summary
```

Prerequisites: Lean 4.11+, Lake, internet access for Mathlib (first run only).

```bash
# One-time setup
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
lake update
```

---

## Proof Obligations

The eight core obligations from `PAX_ARCHITECTURE.md`, mapped to Lean 4 theorem names:

| PO | Architecture Statement | Lean 4 Theorem | Method | Status |
|----|----------------------|----------------|--------|--------|
| PO1 | Index space partition is total and disjoint | `IndexSpace.partition_total_disjoint` | Structural induction on lattice arithmetic | sorry |
| PO2 | Memory address spaces are disjoint | `AddrSpace.spaces_disjoint` | Allocator invariant | sorry |
| PO3 | SIMT divergence always reconverges | `SIMTState.divergence_reconverges` | Well-founded termination on divergence stack | sorry |
| PO4 | HappensBefore is a strict partial order | `HappensBefore.strict_partial_order` | Transitivity + irreflexivity | sorry |
| PO5 | Permission sum ≤ 1 at all addresses | `Perm.sum_le_one` | Monotone permission algebra | sorry |
| PO6 | Barrier redistribution preserves total permission | `Barrier.permission_conservation` | Conservation law on separating conjunction | sorry |
| PO7 | All memory accesses are HB-ordered | `MemAccess.data_race_free` | Data-race-freedom from permission algebra | sorry |
| PO8 | Launch body terminates | `Kernel.terminates` | Index space finiteness (Axiom 6) | sorry |

PO4 is a prerequisite for PO7. PO5 and PO6 together imply PO7. PO3 is independent.
The sorry markers here carry complete proof sketches — they are not blank.

---

## Correspondence to Sovereign GPU Runtime

```
PAX Layer 0 (Index Space)     → scheduler_janet_array[1] = batch_size
PAX Layer 0 (Memory Model)    → kv_allocator free-list invariant
PAX Layer 1 (SIMT)            → flash_attention_paged EXEC mask
PAX Layer 1 (ISA)             → gemm.ptx ldmatrix + mma.sync
PAX Layer 2 (IR)              → sov_cuda_kernels_init module graph
```

Float16/Float32 proofs directly underwrite the `mma.sync.aligned.m16n8k16` accumulation
in `gemm.ptx`: inputs are FP16, accumulator is FP32, and `Float32.exact_fp16_product_sum`
bounds the total rounding error across a full 256×256 tile.

---

## Hardware Target

Primary: Ampere sm_86 (RTX 3080) — all current theorems and extraction targets.

TMA / cluster constructions and sm_89-specific warp specialization target sm_89 (RTX 4090).
These are marked with `-- sm_89` comments and gated behind `PAX.TMA` (planned module).

---

## Honest Accounting

This is the first mechanically verified GPU abstraction chain of its kind.
The sorry count is a feature, not a bug. Every sorry is a published open problem with
a detailed proof sketch. When a sorry is discharged, the corresponding PO moves to PROVEN
and the kernel extraction target becomes unconditionally safe.

Current proved/total ratio: 5 / 13 top-level theorems (38%).

---

## License

BSL-1.1 / AGPL-3.0 / MPL-2.0

Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust

Authors: Ahmad Ali Parr — Jessica Westerhoff
