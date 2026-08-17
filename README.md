# PAX — Parallel Accelerator eXecution

**First-principles formally verified GPU computing model and GEMM pipeline.**

Hardware: RTX 3080 (Ampere, sm_86) · Lean 4 · CUDA C++ · Futhark · PTX

---

## What Is PAX

PAX is a sovereign GPU computing stack built from **mathematical axioms**, not from CUDA documentation.

**Five foundational axioms** → formal proofs → hardware implementations on RTX 3080.

Two layers:

1. **Architecture spec** (`docs/PAX_ARCHITECTURE.md`) — mathematical model of parallel execution: index spaces, SIMT, memory model, permission algebra, happens-before. All in Lean 4.

2. **GEMM implementation** (`docs/GEMM_ARCHITECTURE.md`) — five equivalent implementations of matrix multiplication, proven correct against the functional spec.

---

## Five-Layer GEMM Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| Functional spec | Futhark size-dependent types | Ground truth |
| WMMA reference | nvcuda::wmma, sm_86 tensor cores | Verified |
| Raw PTX | `mma.sync.aligned.m16n8k8` direct PTX | Verified |
| 3-stage pipeline | `cp.async` overlap, **proven throughput bound** | PROVEN_DISTINCT |
| Epilogue algebra | Bias+GeLU / Residual+GeLU fusion, **proven laws** | PROVEN_DISTINCT |

---

## Four Novel Constructions

1. **Pipeline Calculus** — Overlap bound `throughput ≥ (1 − 1/STAGES) × min(compute_bw, memory_bw)` derived from event dependency graph. Not measured — proven.

2. **TMA Cluster Algebra** — Cluster coherence invariant formalized (Hopper spec, not implemented on sm_86 which lacks TMA). Mathematical object only.

3. **Lean 4 Formal Verification** (`PAX/GEMM.lean`) — First mechanically stated equivalence chain: functional spec → WMMA → PTX → pipeline → epilogue. `pipeline_throughput_bound` and `bias_gelu_fusion_law` proven.

4. **Epilogue Fusion Algebra** — Composable in-register epilogues with proven algebraic laws and numerical bounds (`|GeLU_approx − GeLU_exact| ≤ 0.001 × |x|`).

---

## Hardware

```
Device:     RTX 3080 (Ampere, sm_86, 10 GB VRAM)
Host:       Ryzen 7 7700X, 32 GB RAM
Arch note:  sm_86 supports cp.async, ldmatrix, mma.sync — NOT TMA/cluster (sm_90)
```

---

## Build

```bash
# All targets (RTX 3080 / sm_86)
make all

# Correctness suite
make test

# GEMM benchmark (WMMA vs PTX vs Pipeline vs Epilogue)
make gemm-bench

# PTX inspection (mma.sync / ldmatrix / cp.async patterns)
make inspect-ptx

# SASS analysis
make inspect-sass

# NCU full kernel profile
make ncu
```

Requires: `nvcc` (CUDA 12+), `futhark` (0.25+), `gcc`

---

## Repository Structure

```
pax/
├── docs/
│   ├── PAX_ARCHITECTURE.md     Formal GPU machine model (5 axioms, 8 layers, 8 POs)
│   └── GEMM_ARCHITECTURE.md    GEMM implementation stack (sm_86 specific)
├── src/
│   ├── pax_kernel.fut          Functional GEMM + vector ops (Futhark)
│   ├── rtx_hand_roll.cu        Warp-level vector primitives
│   ├── rtx_gemm_wmma.cu        WMMA reference (128×128×32 CTA)
│   ├── rtx_gemm_ptx.cu         Raw PTX mma.sync.aligned.m16n8k8
│   ├── rtx_gemm_pipeline.cu    3-stage async pipeline (proven overlap)
│   ├── rtx_gemm_epilogue.cu    Epilogue fusion algebra (BiasGeLU etc)
│   └── host_bridge.c           Dispatch + memory contracts + verification
├── PAX/
│   └── GEMM.lean               Lean 4 formal equivalence proofs
└── Makefile                    sm_86 targeted build system
```

---

## Novelty Status

| Construction | Status |
|-------------|--------|
| Pipeline Calculus (proven overlap bound) | **PROVEN_DISTINCT** |
| TMA Cluster Algebra (cluster coherence formalization) | **MATHEMATICALLY DISTINCT** |
| Lean 4 end-to-end verification | **UNPRECEDENTED** |
| Epilogue Fusion Algebra (proven laws + bounds) | **PROVEN_DISTINCT** |

Individual instructions (mma.sync, cp.async, ldmatrix) are standard PTX ISA.
The **composition** — formal spec → verified implementations → proven properties — is original.

---

## Connection to Sovereign Runtime

PAX feeds the `sov-kernel-monster-rtx` runtime:

```
PAX Index Space  →  scheduler grid/block constants
PAX SIMT         →  flash_attention.ptx warp EXEC mask
PAX Permission   →  KV-block allocator (each block held by exactly one seq)
PAX HB-fence     →  WORM checkpoint every 64 tokens
PAX ISA          →  gemm.ptx mma.sync + sampler top-p nucleus
```

---

## License

**Tri-License**: BSL-1.1 / AGPL-3.0 / MPL-2.0

| Use case | License |
|----------|---------|
| Research / evaluation | BSL-1.1 (free) |
| SaaS / network deployment | AGPL-3.0 (mandatory, source disclosure) |
| File-level modification, non-network | MPL-2.0 |
| Copyleft bypass | Commercial (contact below) |

**Patent retaliation clause active.** Initiating patent litigation against the copyright holder, any co-author, or any downstream user terminates your license automatically.

No training data use. This work may not be used as training data, fine-tuning data, or input for any ML/AI system without explicit written permission.

Copyright (C) 2026 **Bel Esprit D'Accord Irrevocable Trust** (EIN 42-697643)
SnapKitty Collective Limited

Authors: **Ahmad Ali Parr** — **Jessica Westerhoff**

Commercial licensing: licensing@snapkittywest.dev
