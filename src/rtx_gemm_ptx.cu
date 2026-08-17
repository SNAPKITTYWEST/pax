// PAX GEMM PTX Raw — Ampere sm_86 Tensor Cores
// Raw PTX mma.sync.aligned.m16n8k8 via inline assembly
// CTA Tile: 128x128x32, 8 warps (256 threads)
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
// Authors: Ahmad Ali Parr — Jessica Westerhoff
// License: BSL-1.1 / AGPL-3.0 / MPL-2.0

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// PTX intrinsic macros
// ---------------------------------------------------------------------------

// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
// D[2xf32x4] = A[2xf32x2] * B[2xf32x1] + C[2xf32x4]
// a_regs: 4 x uint32 (2 rows x 2 uint32 per row = 16 fp16 values = m16k8)
// b_regs: 2 x uint32 (1 col x 2 uint32 per col = 8 fp16 values = k8n8)
// c/d_regs: 8 x float (2 rows x 4 fp32 per row = m16n8)
#define PTX_MMA_SYNC_M16N8K8_F16_F32(d0,d1,d2,d3,d4,d5,d6,d7, \
                                      a0,a1,a2,a3,               \
                                      b0,b1,                      \
                                      c0,c1,c2,c3,c4,c5,c6,c7)   \
    asm volatile(                                                  \
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "       \
        "{%0,%1,%2,%3,%4,%5,%6,%7},"                               \
        "{%8,%9,%10,%11},"                                         \
        "{%12,%13},"                                               \
        "{%14,%15,%16,%17,%18,%19,%20,%21};"                       \
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3),                   \
          "=f"(d4),"=f"(d5),"=f"(d6),"=f"(d7)                    \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),                        \
          "r"(b0),"r"(b1),                                         \
          "f"(c0),"f"(c1),"f"(c2),"f"(c3),                        \
          "f"(c4),"f"(c5),"f"(c6),"f"(c7)                         \
    )

// ldmatrix.sync.aligned.x4.m8n8.shared.b16
// Loads 4 x 8x8 fp16 matrices from shared memory (row-major)
// dst: 4 x uint32 output registers
// src_smem_ptr: 32-bit shared memory pointer (via __cvta_generic_to_shared)
#define PTX_LDMATRIX_X4(dst0,dst1,dst2,dst3, src_smem_ptr)        \
    asm volatile(                                                   \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "               \
        "{%0,%1,%2,%3}, [%4];"                                     \
        : "=r"(dst0),"=r"(dst1),"=r"(dst2),"=r"(dst3)             \
        : "r"(src_smem_ptr)                                        \
    )

// ldmatrix.sync.aligned.x2.m8n8.shared.b16
// Loads 2 x 8x8 fp16 matrices from shared memory
#define PTX_LDMATRIX_X2(dst0,dst1, src_smem_ptr)                  \
    asm volatile(                                                   \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "               \
        "{%0,%1}, [%4];"                                           \
        : "=r"(dst0),"=r"(dst1)                                    \
        : "r"(src_smem_ptr)                                        \
    )

// cp.async.ca.shared.global — async copy 16 bytes global -> shared (L2 cached)
// dst_smem_ptr: 32-bit shared memory address
// src_gmem_ptr: 64-bit global memory address
#define PTX_CP_ASYNC_CA_SHARED_GLOBAL(dst_smem_ptr, src_gmem_ptr) \
    asm volatile(                                                   \
        "cp.async.ca.shared.global [%0], [%1], 16;"               \
        : : "r"(dst_smem_ptr), "l"(src_gmem_ptr)                  \
        : "memory"                                                  \
    )

// cp.async.commit_group — commit current async copy group
#define PTX_CP_ASYNC_COMMIT                                        \
    asm volatile("cp.async.commit_group;" : : : "memory")

// cp.async.wait_all — wait for all outstanding async copy groups
#define PTX_CP_ASYNC_WAIT_ALL                                      \
    asm volatile("cp.async.wait_all;" : : : "memory")

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WMMA_M = 16, WMMA_N = 8, WMMA_K = 8;
constexpr int WARPS_M = 4, WARPS_N = 2;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32; // 256
constexpr int SMEM_A_STRIDE = 40;   // BK + 8 — bank-conflict-free for fp16
constexpr int SMEM_B_STRIDE = 136;  // BN + 8

// ---------------------------------------------------------------------------
// Shared memory layout
//   sA: [BM][SMEM_A_STRIDE] half  = 128 * 40 * 2 = 10240 bytes
//   sB: [BK][SMEM_B_STRIDE] half  =  32 * 136 * 2 = 8704 bytes
//   Total: 18944 bytes (well within 48 KB carveout)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: convert generic pointer to 32-bit shared memory address
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_ptr32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void pax_gemm_ptx_kernel(
    const half* __restrict__ A,   // [M][K] row-major
    const half* __restrict__ B,   // [K][N] row-major
    float*      __restrict__ C,   // [M][N] row-major
    int M, int N, int K
) {
    __shared__ half sA[BM][SMEM_A_STRIDE];
    __shared__ half sB[BK][SMEM_B_STRIDE];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id / WARPS_N;   // 0..3
    const int warp_n  = warp_id % WARPS_N;   // 0..1
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // Each warp covers a (32 x 64) output tile via 2x8 mma tiles
    // (2 tiles along M = 2*16 = 32, 8 tiles along N = 8*8 = 64)
    // Accumulator layout: c_regs[warp_m_tile][warp_n_tile][d_reg_pair]
    //   warp_m_tile: 0..1  (2 tiles, each 16 rows)
    //   warp_n_tile: 0..7  (8 tiles, each 8 cols)
    //   d_reg_pair:  0..1  (each mma.m16n8k8 produces 8 f32; stored as 4+4)
    // Total: 2 * 8 * 2 * 4 = 128 fp32 accumulators per warp — correct for 32x64

    float c_regs[2][8][2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            #pragma unroll
            for (int pi = 0; pi < 2; ++pi)
                #pragma unroll
                for (int qi = 0; qi < 4; ++qi)
                    c_regs[mi][ni][pi][qi] = 0.0f;

    // Global -> shared load helpers
    // Each thread loads 8 fp16 elements (16 bytes) per tile per dimension
    const int a_load_row = tid / (BK / 8);   // row within sA to load (0..BM-1)
    const int a_load_col = (tid % (BK / 8)) * 8;
    const int b_load_row = tid / (BN / 8);   // row within sB to load (0..BK-1)
    const int b_load_col = (tid % (BN / 8)) * 8;

    // Strided loops: each thread covers multiple rows
    const int a_row_stride = THREADS_PER_BLOCK / (BK / 8);  // = 256/4 = 64
    const int b_row_stride = THREADS_PER_BLOCK / (BN / 8);  // = 256/16 = 16

    for (int k_outer = 0; k_outer < K; k_outer += BK) {

        // --- Load sA: [BM][BK] from A[block_m..block_m+BM][k_outer..k_outer+BK] ---
        #pragma unroll 2
        for (int row_off = 0; row_off < BM; row_off += a_row_stride) {
            int row = a_load_row + row_off;
            int gm  = block_m + row;
            int gk  = k_outer + a_load_col;
            uint32_t dst = smem_ptr32(&sA[row][a_load_col]);
            if (gm < M && gk + 7 < K) {
                PTX_CP_ASYNC_CA_SHARED_GLOBAL(dst,
                    reinterpret_cast<const uint64_t*>(&A[gm * K + gk]));
            } else {
                // Boundary: zero the slot
                *reinterpret_cast<uint4*>(&sA[row][a_load_col]) = make_uint4(0,0,0,0);
            }
        }

        // --- Load sB: [BK][BN] from B[k_outer..k_outer+BK][block_n..block_n+BN] ---
        #pragma unroll 2
        for (int row_off = 0; row_off < BK; row_off += b_row_stride) {
            int row = b_load_row + row_off;
            int gk  = k_outer + row;
            int gn  = block_n + b_load_col;
            uint32_t dst = smem_ptr32(&sB[row][b_load_col]);
            if (gk < K && gn + 7 < N) {
                PTX_CP_ASYNC_CA_SHARED_GLOBAL(dst,
                    reinterpret_cast<const uint64_t*>(&B[gk * N + gn]));
            } else {
                *reinterpret_cast<uint4*>(&sB[row][b_load_col]) = make_uint4(0,0,0,0);
            }
        }

        PTX_CP_ASYNC_COMMIT;
        PTX_CP_ASYNC_WAIT_ALL;
        __syncthreads();

        // --- Compute: iterate over k_inner in steps of WMMA_K=8 ---
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner += WMMA_K) {

            // Each warp covers rows [warp_m*32 .. warp_m*32+31] in sA
            // and cols [warp_n*64 .. warp_n*64+63] in sB.
            //
            // a_regs[mi][4]: loads two m16k8 A fragments (mi=0,1)
            //   fragment mi covers sA rows [(warp_m*32)+(mi*16) .. +15]
            //   ldmatrix.x4 loads 4 x uint32 = one m16k8 fragment (16 rows x 8 cols fp16)
            //
            // b_regs[ni][2]: loads eight k8n8 B fragments (ni=0..7)
            //   fragment ni covers sB cols [(warp_n*64)+(ni*8) .. +7]
            //   ldmatrix.x2 loads 2 x uint32 = one k8n8 fragment (8 rows x 8 cols fp16)

            uint32_t a_regs[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int smem_row = (warp_m * 32) + (mi * WMMA_M) + (lane_id % 8) * 2;
                // Each ldmatrix.x4 pulls 4 rows at once; lane selects starting row
                // Standard Ampere ldmatrix addressing: lane_id selects which 8-row group
                int smem_row_base = (warp_m * 32) + (mi * WMMA_M);
                // ldmatrix uses lane_id to distribute rows across warp
                // Point each thread at its designated row pair
                int lm_row = smem_row_base + (lane_id & 15);
                PTX_LDMATRIX_X4(
                    a_regs[mi][0], a_regs[mi][1], a_regs[mi][2], a_regs[mi][3],
                    smem_ptr32(&sA[lm_row][k_inner])
                );
            }

            uint32_t b_regs[8][2];
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                int smem_col_base = (warp_n * 64) + (ni * WMMA_N);
                int lm_row = k_inner + (lane_id & 7);
                PTX_LDMATRIX_X2(
                    b_regs[ni][0], b_regs[ni][1],
                    smem_ptr32(&sB[lm_row][smem_col_base])
                );
            }

            // Issue mma.sync for all 2x8 tile combinations
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni) {
                    PTX_MMA_SYNC_M16N8K8_F16_F32(
                        c_regs[mi][ni][0][0], c_regs[mi][ni][0][1],
                        c_regs[mi][ni][0][2], c_regs[mi][ni][0][3],
                        c_regs[mi][ni][1][0], c_regs[mi][ni][1][1],
                        c_regs[mi][ni][1][2], c_regs[mi][ni][1][3],
                        a_regs[mi][0], a_regs[mi][1], a_regs[mi][2], a_regs[mi][3],
                        b_regs[ni][0], b_regs[ni][1],
                        c_regs[mi][ni][0][0], c_regs[mi][ni][0][1],
                        c_regs[mi][ni][0][2], c_regs[mi][ni][0][3],
                        c_regs[mi][ni][1][0], c_regs[mi][ni][1][1],
                        c_regs[mi][ni][1][2], c_regs[mi][ni][1][3]
                    );
                }
            }
        }

        __syncthreads();
    }

    // ---------------------------------------------------------------------------
    // Store accumulators to global C
    // mma.m16n8k8 output layout: each thread holds 2 rows x 1 col per fragment
    // in interleaved register pairs.  For m16n8:
    //   lane t holds rows {t/4, t/4+8} for cols {(t%4)*2, (t%4)*2+1}
    // ---------------------------------------------------------------------------
    const int out_lane_row0 = lane_id / 4;
    const int out_lane_row1 = out_lane_row0 + 8;
    const int out_lane_col0 = (lane_id % 4) * 2;
    const int out_lane_col1 = out_lane_col0 + 1;

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int tile_row_base = block_m + (warp_m * 32) + (mi * WMMA_M);
            int tile_col_base = block_n + (warp_n * 64) + (ni * WMMA_N);

            // 4 output positions per thread per fragment (2 rows x 2 cols)
            // c_regs[mi][ni][0][0..3] → rows {row0,row0,row1,row1} x cols {col0,col1,col0,col1}
            // c_regs[mi][ni][1][0..3] → second set of 4 (rows offset by layout)
            // Standard m16n8 layout: d[0..3] = row0 col0,col1,row1 col0,col1
            //                         d[4..7] = row0 col0+4,col1+4,row1 col0+4,col1+4
            // (N=8 so second set of cols wraps; handled by ni tile stride)

            int r0 = tile_row_base + out_lane_row0;
            int r1 = tile_row_base + out_lane_row1;
            int c0 = tile_col_base + out_lane_col0;
            int c1 = tile_col_base + out_lane_col1;

            if (r0 < M && c0 < N) C[r0 * N + c0] = c_regs[mi][ni][0][0];
            if (r0 < M && c1 < N) C[r0 * N + c1] = c_regs[mi][ni][0][1];
            if (r1 < M && c0 < N) C[r1 * N + c0] = c_regs[mi][ni][0][2];
            if (r1 < M && c1 < N) C[r1 * N + c1] = c_regs[mi][ni][0][3];

            // Second register pair (cols +4 within the 8-wide tile)
            int c2 = c0 + 4;
            int c3 = c1 + 4;
            if (r0 < M && c2 < N) C[r0 * N + c2] = c_regs[mi][ni][1][0];
            if (r0 < M && c3 < N) C[r0 * N + c3] = c_regs[mi][ni][1][1];
            if (r1 < M && c2 < N) C[r1 * N + c2] = c_regs[mi][ni][1][2];
            if (r1 < M && c3 < N) C[r1 * N + c3] = c_regs[mi][ni][1][3];
        }
    }
}

// ---------------------------------------------------------------------------
// Shared memory query — call before cudaFuncSetAttribute if needed
// ---------------------------------------------------------------------------
extern "C" size_t pax_gemm_ptx_smem_bytes(void) {
    return sizeof(half) * BM * SMEM_A_STRIDE +
           sizeof(half) * BK * SMEM_B_STRIDE;
    // = 2*128*40 + 2*32*136 = 10240 + 8704 = 18944 bytes
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
extern "C" void pax_gemm_ptx_launch(
    const half* A, const half* B, float* C,
    int M, int N, int K, cudaStream_t stream
) {
    // Optionally raise smem carveout to 48 KB for Ampere
    cudaFuncSetAttribute(
        pax_gemm_ptx_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        48
    );

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    pax_gemm_ptx_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
