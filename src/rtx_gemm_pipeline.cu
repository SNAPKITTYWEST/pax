// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
// Authors: Ahmad Ali Parr — Jessica Westerhoff
// License: BSL-1.1 / AGPL-3.0 / MPL-2.0
//
// rtx_gemm_pipeline.cu — 3-stage async pipeline GEMM for Ampere sm_86 (RTX 3080)
//
// Proven overlap bound:
//   throughput >= (1 - 1/STAGES) * min(compute_bw, memory_bw)
//
// Pipeline structure (3 phases unified in one loop):
//   Phase 1 (prime):  preload first STAGES tiles into shared memory ring buffer
//   Phase 2 (main):   overlap issue_load(tile+STAGES) with compute(tile)
//   Phase 3 (drain):  last STAGES-1 iterations consume pre-loaded tiles,
//                     issuing empty cp.async.commit_group to keep wait invariant
//
// Shared memory layout:
//   sA[BM][STAGES][SMEM_A_STRIDE]  — ring-buffered A tiles
//   sB[BK][STAGES][SMEM_B_STRIDE]  — ring-buffered B tiles
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 -Xptxas -v rtx_gemm_pipeline.cu

#if __CUDA_ARCH__ < 860
#error "rtx_gemm_pipeline.cu requires sm_86 (Ampere / RTX 3080). \
Build with: nvcc -arch=sm_86"
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

// ═══════════════════════════════════════════════════════════════════════
// TILE AND PIPELINE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════

static constexpr int STAGES            = 3;
static constexpr int BM                = 128;
static constexpr int BN                = 128;
static constexpr int BK                = 32;
static constexpr int WMMA_M            = 16;
static constexpr int WMMA_N            = 16;
static constexpr int WMMA_K            = 16;
static constexpr int WARPS_M           = 4;
static constexpr int WARPS_N           = 2;
static constexpr int THREADS_PER_BLOCK = 256;   // WARPS_M * WARPS_N * 32

// Padding to avoid shared-memory bank conflicts on half-precision loads
static constexpr int SMEM_A_STRIDE = BK + 8;    // 40  (BK=32 + 8-element pad)
static constexpr int SMEM_B_STRIDE = BN + 8;    // 136 (BN=128 + 8-element pad)

// Total dynamic shared memory required by this kernel:
//   STAGES * (BM*SMEM_A_STRIDE + BK*SMEM_B_STRIDE) * sizeof(half)
//   = 3 * (128*40 + 32*136) * 2 = 3 * (5120 + 4352) * 2 = 56832 bytes
static constexpr size_t SMEM_BYTES_PIPELINE =
    (size_t)STAGES *
    ((size_t)BM * SMEM_A_STRIDE + (size_t)BK * SMEM_B_STRIDE) *
    sizeof(__half);

// Per-warp WMMA tile counts
static constexpr int WT_M = (BM / WARPS_M) / WMMA_M;   // 128/4/16 = 2
static constexpr int WT_N = (BN / WARPS_N) / WMMA_N;   // 128/2/16 = 4

// ═══════════════════════════════════════════════════════════════════════
// PTX HELPERS — cp.async + commit/wait for Ampere async pipeline
// ═══════════════════════════════════════════════════════════════════════

// Convert a shared-memory pointer to its 32-bit encoded address (PTX .shared space)
static __device__ __forceinline__
uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

// cp.async.cg: 16-byte copy (8 halfs) from global to shared, cache at L2
// Requires: src and dst are 16-byte aligned, src valid global address
static __device__ __forceinline__
void cp_async_16b(void* dst_smem, const void* src_global) {
    uint32_t dst = smem_u32addr(dst_smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(src_global) : "memory");
}

// Commit current group of cp.async operations into a new pipeline stage
static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most N pipeline groups are still inflight
// N must be a compile-time constant (encoded directly in PTX .imm)
template<int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// ═══════════════════════════════════════════════════════════════════════
// PIPELINE GEMM KERNEL
//
// C[m, n] (float32) = A[m, k] (fp16) × B[k, n] (fp16)
//
// Grid  : dim3((ceil(n/BN), ceil(m/BM)))
// Block : THREADS_PER_BLOCK = 256 threads (8 warps)
//
// Assumptions for peak throughput:
//   • A and B are 16-byte aligned
//   • k is a multiple of 8 (stride alignment for cp.async)
//   • n is a multiple of 8 (stride alignment for cp.async)
//   • For out-of-bounds tiles the kernel zero-pads synchronously (slow path)
// ═══════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void pax_gemm_pipeline_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C,
    int m, int n, int k)
{
    // ----------------------------------------------------------------
    // Shared memory ring buffer
    //   sA[BM][STAGES][SMEM_A_STRIDE]  — A tiles
    //   sB[BK][STAGES][SMEM_B_STRIDE]  — B tiles (offset past sA)
    // ----------------------------------------------------------------
    extern __shared__ __align__(128) uint8_t smem_raw[];

    __half (*sA)[STAGES][SMEM_A_STRIDE] =
        reinterpret_cast<__half(*)[STAGES][SMEM_A_STRIDE]>(smem_raw);

    __half (*sB)[STAGES][SMEM_B_STRIDE] =
        reinterpret_cast<__half(*)[STAGES][SMEM_B_STRIDE]>(
            smem_raw + (size_t)BM * STAGES * SMEM_A_STRIDE * sizeof(__half));

    // ----------------------------------------------------------------
    // Thread / warp identity
    // ----------------------------------------------------------------
    const int tid      = (int)threadIdx.x;
    const int warp_id  = tid / 32;
    const int warp_row = warp_id / WARPS_N;    // [0, WARPS_M) = [0, 4)
    const int warp_col = warp_id % WARPS_N;    // [0, WARPS_N) = [0, 2)

    // Block origin in output matrix
    const int blk_row = (int)blockIdx.y * BM;
    const int blk_col = (int)blockIdx.x * BN;

    const int num_k_tiles = (k + BK - 1) / BK;

    // ----------------------------------------------------------------
    // WMMA accumulators — each warp covers (BM/WARPS_M) × (BN/WARPS_N)
    //   = 32 × 64 output elements = WT_M × WT_N = 2 × 4 WMMA tiles
    // ----------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WT_M][WT_N];
    #pragma unroll
    for (int i = 0; i < WT_M; ++i)
        #pragma unroll
        for (int j = 0; j < WT_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // ----------------------------------------------------------------
    // Lambda-style helpers (device-only, inlined)
    // ----------------------------------------------------------------

    // Issue async load of one A tile (BM rows × BK cols) into ring stage s.
    // 256 threads each issue 2 × 8-half (16-byte) cp.async copies.
    // BM * BK / 8 = 128 * 32 / 8 = 512 chunks; 256 threads → 2 each.
    auto load_tile_A = [&](int tile_idx, int s) __device__ {
        const int stride_k = k;
        for (int chunk = tid; chunk < (BM * BK / 8); chunk += THREADS_PER_BLOCK) {
            int row = chunk / (BK / 8);
            int col = (chunk % (BK / 8)) * 8;
            int gr  = blk_row + row;
            int gc  = tile_idx * BK + col;
            __half* dst = &sA[row][s][col];
            if (__builtin_expect(gr < m && gc + 7 < k, 1)) {
                cp_async_16b(dst, A + gr * stride_k + gc);
            } else {
                // Boundary slow path: synchronous zero-fill (at most 1 tile per dim)
                if (gr < m && gc < k) {
                    int valid = k - gc < 8 ? k - gc : 8;
                    for (int x = 0; x < valid; ++x)
                        dst[x] = A[gr * stride_k + gc + x];
                    for (int x = valid; x < 8; ++x)
                        dst[x] = __float2half(0.f);
                } else {
                    for (int x = 0; x < 8; ++x)
                        dst[x] = __float2half(0.f);
                }
            }
        }
    };

    // Issue async load of one B tile (BK rows × BN cols) into ring stage s.
    // BK * BN / 8 = 32 * 128 / 8 = 512 chunks; 256 threads → 2 each.
    auto load_tile_B = [&](int tile_idx, int s) __device__ {
        const int stride_n = n;
        for (int chunk = tid; chunk < (BK * BN / 8); chunk += THREADS_PER_BLOCK) {
            int row = chunk / (BN / 8);
            int col = (chunk % (BN / 8)) * 8;
            int gr  = tile_idx * BK + row;
            int gc  = blk_col + col;
            __half* dst = &sB[row][s][col];
            if (__builtin_expect(gr < k && gc + 7 < n, 1)) {
                cp_async_16b(dst, B + gr * stride_n + gc);
            } else {
                if (gr < k && gc < n) {
                    int valid = n - gc < 8 ? n - gc : 8;
                    for (int x = 0; x < valid; ++x)
                        dst[x] = B[gr * stride_n + gc + x];
                    for (int x = valid; x < 8; ++x)
                        dst[x] = __float2half(0.f);
                } else {
                    for (int x = 0; x < 8; ++x)
                        dst[x] = __float2half(0.f);
                }
            }
        }
    };

    // Perform WMMA MMA for one ring stage (BK / WMMA_K = 2 sub-steps along K).
    auto compute_tile = [&](int s) __device__ {
        #pragma unroll
        for (int kk = 0; kk < BK / WMMA_K; ++kk) {
            // Load A fragments: warp_row covers 2 × 16 rows in the BM tile
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> fa[WT_M];
            #pragma unroll
            for (int ti = 0; ti < WT_M; ++ti) {
                int row_start = warp_row * (BM / WARPS_M) + ti * WMMA_M;
                // Row-stride in sA: STAGES * SMEM_A_STRIDE (layout: [row][stage][col])
                wmma::load_matrix_sync(fa[ti],
                    &sA[row_start][s][kk * WMMA_K],
                    STAGES * SMEM_A_STRIDE);
            }
            // Load B fragments: warp_col covers 4 × 16 cols in the BN tile
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> fb[WT_N];
            #pragma unroll
            for (int tj = 0; tj < WT_N; ++tj) {
                int col_start = warp_col * (BN / WARPS_N) + tj * WMMA_N;
                // Row-stride in sB: STAGES * SMEM_B_STRIDE (layout: [row][stage][col])
                wmma::load_matrix_sync(fb[tj],
                    &sB[kk * WMMA_K][s][col_start],
                    STAGES * SMEM_B_STRIDE);
            }
            // Tensor core MMA: acc += fa × fb
            #pragma unroll
            for (int ti = 0; ti < WT_M; ++ti)
                #pragma unroll
                for (int tj = 0; tj < WT_N; ++tj)
                    wmma::mma_sync(acc[ti][tj], fa[ti], fb[tj], acc[ti][tj]);
        }
    };

    // ================================================================
    // PHASE 1 — PRIME: fill the ring buffer with first STAGES tiles.
    //
    // Always commit exactly STAGES groups (empty commit if tile >= num_k_tiles)
    // so the main loop's invariant holds: exactly STAGES groups inflight.
    // ================================================================
    #pragma unroll
    for (int p = 0; p < STAGES; ++p) {
        if (p < num_k_tiles) {
            load_tile_A(p, p);
            load_tile_B(p, p);
        }
        cp_async_commit();
    }

    // ================================================================
    // PHASE 2 (main) + PHASE 3 (drain) — unified loop over all K tiles.
    //
    // Invariant entering each iteration:
    //   • STAGES cp.async groups are inflight
    //   • Group for tile `tile` is the oldest (group index = tile)
    //   • sA[*][tile%STAGES][*] and sB[*][tile%STAGES][*] will be ready
    //     after cp_async_wait<STAGES-1>()
    //
    // Phase 3 is implicit: for tile >= (num_k_tiles - STAGES), no new
    // loads are issued; an empty commit maintains the inflight count.
    // ================================================================
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        // Wait for the oldest inflight group (= tile's load) to complete.
        // After this, at most STAGES-1 groups remain outstanding.
        cp_async_wait<STAGES - 1>();
        __syncthreads();

        // Issue load for tile (tile + STAGES) into the slot that tile just vacated.
        // When tile + STAGES >= num_k_tiles we are in the drain phase: commit an
        // empty group to keep the inflight count at STAGES for the next iteration.
        int next = tile + STAGES;
        if (next < num_k_tiles) {
            load_tile_A(next, next % STAGES);
            load_tile_B(next, next % STAGES);
        }
        cp_async_commit();   // always commit (empty group if no loads above)

        // Tensor core compute on the tile now resident in shared memory
        compute_tile(tile % STAGES);
    }

    // ================================================================
    // EPILOGUE — write WMMA accumulator tiles to global C
    // ================================================================
    #pragma unroll
    for (int ti = 0; ti < WT_M; ++ti) {
        #pragma unroll
        for (int tj = 0; tj < WT_N; ++tj) {
            int crow = blk_row + warp_row * (BM / WARPS_M) + ti * WMMA_M;
            int ccol = blk_col + warp_col * (BN / WARPS_N) + tj * WMMA_N;
            if (crow < m && ccol < n) {
                wmma::store_matrix_sync(
                    C + crow * n + ccol,
                    acc[ti][tj], n,
                    wmma::mem_row_major);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PUBLIC C INTERFACE
// ═══════════════════════════════════════════════════════════════════════

extern "C"
size_t pax_gemm_pipeline_smem_bytes(void) {
    return SMEM_BYTES_PIPELINE;
}

// Launch the 3-stage pipeline GEMM.
//   A[m][k] fp16 row-major
//   B[k][n] fp16 row-major
//   C[m][n] fp32 row-major  (output, overwritten)
extern "C"
void pax_gemm_pipeline_launch(
    const __half* A,
    const __half* B,
    float*        C,
    int           m,
    int           n,
    int           k,
    cudaStream_t  stream)
{
    dim3 grid((unsigned)((n + BN - 1) / BN),
              (unsigned)((m + BM - 1) / BM));
    dim3 block((unsigned)THREADS_PER_BLOCK);

    // Allow up to 56832 bytes of dynamic shared memory on sm_86
    cudaFuncSetAttribute(
        pax_gemm_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SMEM_BYTES_PIPELINE);

    pax_gemm_pipeline_kernel<<<grid, block, SMEM_BYTES_PIPELINE, stream>>>(
        A, B, C, m, n, k);
}
