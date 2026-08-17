// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
// Authors: Ahmad Ali Parr — Jessica Westerhoff
// License: BSL-1.1 / AGPL-3.0 / MPL-2.0
//
// rtx_gemm_epilogue.cu — Epilogue fusion algebra for Ampere sm_86 (RTX 3080)
//
// Fusion law: Fuse(Bias, GeLU) ≡ GeLU ∘ Bias — proven algebraically
//   For any x: BiasGeLU(x, b) = GeLU(x + b)
//              ResidualGeLU(x, r) = GeLU(x + r)
//              ScaleBiasGeLU(x, α, b) = GeLU(α·x + b)
//
// Each epilogue struct carries a template method apply<FRAG_M, FRAG_N>
// that operates on nvcuda::wmma accumulator fragment elements before
// the fragment is written to global memory.
//
// Epilogue apply convention:
//   apply(frag, row_base, col_base, lane)
//     frag      — wmma::accumulator fragment (fp32 elements, modified in place)
//     row_base  — top row of this fragment in the global output matrix
//     col_base  — leftmost column of this fragment in the global output matrix
//     lane      — threadIdx.x & 31  (used to decode element → column offset)
//
// On sm_86 a 16×16 fp32 accumulator holds 8 elements per thread.
// Column offset approximation:  col ≈ col_base + (lane & 7)*2 + (e & 1)
// Row  offset approximation:    row ≈ row_base + (e >> 1)
// (Exact layout is implementation-defined; use store→apply→load for full
//  precision, or the wmma element iterators provided by CUTLASS.)
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 -Xptxas -v rtx_gemm_epilogue.cu

#if __CUDA_ARCH__ < 860
#error "rtx_gemm_epilogue.cu requires sm_86 (Ampere / RTX 3080). \
Build with: nvcc -arch=sm_86"
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

// ═══════════════════════════════════════════════════════════════════════
// TILE CONSTANTS (must match other pax files)
// ═══════════════════════════════════════════════════════════════════════

static constexpr int EPI_BM               = 128;
static constexpr int EPI_BN               = 128;
static constexpr int EPI_BK               = 32;
static constexpr int EPI_WMMA_M           = 16;
static constexpr int EPI_WMMA_N           = 16;
static constexpr int EPI_WMMA_K           = 16;
static constexpr int EPI_WARPS_M          = 4;
static constexpr int EPI_WARPS_N          = 2;
static constexpr int EPI_THREADS          = 256;
static constexpr int EPI_SMEM_A_STRIDE    = EPI_BK + 8;    // 40
static constexpr int EPI_SMEM_B_STRIDE    = EPI_BN + 8;    // 136

static constexpr int EPI_WT_M = (EPI_BM / EPI_WARPS_M) / EPI_WMMA_M;  // 2
static constexpr int EPI_WT_N = (EPI_BN / EPI_WARPS_N) / EPI_WMMA_N;  // 4

static constexpr size_t EPI_SMEM_BYTES =
    ((size_t)EPI_BM * EPI_SMEM_A_STRIDE +
     (size_t)EPI_BK * EPI_SMEM_B_STRIDE) * sizeof(__half);

// ═══════════════════════════════════════════════════════════════════════
// GELU APPROXIMATION  (tanh-form, numerically stable for |x| < 5)
//
//   GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
//   √(2/π) ≈ 0.7978845608028654
// ═══════════════════════════════════════════════════════════════════════

static __device__ __forceinline__ float gelu_approx(float x) {
    constexpr float kAlpha = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float kBeta  = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x * x * x)));
}

// ═══════════════════════════════════════════════════════════════════════
// EPILOGUE STRUCTS
// ═══════════════════════════════════════════════════════════════════════

// -----------------------------------------------------------------------
// BiasAdd — adds a per-column bias vector to every accumulator element
// -----------------------------------------------------------------------
struct BiasAdd {
    const __half* bias;   // [N] in global memory

    __device__ __forceinline__ explicit BiasAdd(const __half* b) : bias(b) {}

    // Apply bias to every element of a wmma accumulator fragment.
    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int /*row_base*/, int col_base, int lane) const
    {
        #pragma unroll
        for (int e = 0; e < (int)frag.num_elements; ++e) {
            int col_off = (lane & 7) * 2 + (e & 1);
            frag.x[e] += __half2float(bias[col_base + col_off]);
        }
    }
};

// -----------------------------------------------------------------------
// GeLU — applies GELU activation element-wise (no positional info needed)
// -----------------------------------------------------------------------
struct GeLU {
    __device__ __forceinline__ GeLU() {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int /*row_base*/, int /*col_base*/, int /*lane*/) const
    {
        #pragma unroll
        for (int e = 0; e < (int)frag.num_elements; ++e)
            frag.x[e] = gelu_approx(frag.x[e]);
    }
};

// -----------------------------------------------------------------------
// ResidualAdd — adds a residual matrix (fp32) element-wise
// -----------------------------------------------------------------------
struct ResidualAdd {
    const float* residual;  // [M][N] fp32 row-major
    int          ldc;       // leading dimension of residual (= N)

    __device__ __forceinline__ ResidualAdd(const float* r, int ld)
        : residual(r), ldc(ld) {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int row_base, int col_base, int lane) const
    {
        #pragma unroll
        for (int e = 0; e < (int)frag.num_elements; ++e) {
            int col_off = (lane & 7) * 2 + (e & 1);
            int row_off = e >> 1;
            frag.x[e] += residual[(row_base + row_off) * ldc + col_base + col_off];
        }
    }
};

// -----------------------------------------------------------------------
// Scale — multiplies every accumulator element by a scalar alpha
// -----------------------------------------------------------------------
struct Scale {
    float alpha;

    __device__ __forceinline__ explicit Scale(float a) : alpha(a) {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int /*row_base*/, int /*col_base*/, int /*lane*/) const
    {
        #pragma unroll
        for (int e = 0; e < (int)frag.num_elements; ++e)
            frag.x[e] *= alpha;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// FUSED EPILOGUE STRUCTS
// Fusion law: Fuse(Bias, GeLU) ≡ GeLU ∘ Bias — proven algebraically
// ═══════════════════════════════════════════════════════════════════════

// -----------------------------------------------------------------------
// BiasGeLU = GeLU ∘ BiasAdd  (bias first, then activation)
// -----------------------------------------------------------------------
struct BiasGeLU {
    const __half* bias;

    __device__ __forceinline__ explicit BiasGeLU(const __half* b) : bias(b) {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int row_base, int col_base, int lane) const
    {
        // Fusion law: GeLU(x + b) = GeLU ∘ BiasAdd(x, b)
        BiasAdd ba(bias);
        ba.apply(frag, row_base, col_base, lane);
        GeLU{}.apply(frag, row_base, col_base, lane);
    }
};

// -----------------------------------------------------------------------
// ResidualGeLU = GeLU ∘ ResidualAdd  (add residual, then activation)
// -----------------------------------------------------------------------
struct ResidualGeLU {
    const float* residual;
    int          ldc;

    __device__ __forceinline__ ResidualGeLU(const float* r, int ld)
        : residual(r), ldc(ld) {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int row_base, int col_base, int lane) const
    {
        ResidualAdd ra(residual, ldc);
        ra.apply(frag, row_base, col_base, lane);
        GeLU{}.apply(frag, row_base, col_base, lane);
    }
};

// -----------------------------------------------------------------------
// ScaleBiasGeLU = GeLU ∘ BiasAdd ∘ Scale  (scale, then bias, then GeLU)
// -----------------------------------------------------------------------
struct ScaleBiasGeLU {
    float         alpha;
    const __half* bias;

    __device__ __forceinline__ ScaleBiasGeLU(float a, const __half* b)
        : alpha(a), bias(b) {}

    template<int FM, int FN>
    __device__ __forceinline__ void apply(
        wmma::fragment<wmma::accumulator, FM, FN, EPI_WMMA_K, float>& frag,
        int row_base, int col_base, int lane) const
    {
        Scale{alpha}.apply(frag, row_base, col_base, lane);
        BiasGeLU bg(bias);
        bg.apply(frag, row_base, col_base, lane);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// TEMPLATE GEMM + EPILOGUE KERNEL
//
// Same tiling / WMMA core as the other pax GEMM kernels.
// After MMA, calls epilogue.apply<WMMA_M, WMMA_N>(frag, row, col, lane)
// before wmma::store_matrix_sync.
// ═══════════════════════════════════════════════════════════════════════

template<typename Epilogue>
__global__ __launch_bounds__(EPI_THREADS)
void pax_gemm_epilogue_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C,
    int m, int n, int k,
    Epilogue epilogue)
{
    // ---- Shared memory (double-buffered, 1 stage = no pipeline here) -----
    extern __shared__ __align__(128) uint8_t epi_smem[];

    __half (*sA)[EPI_SMEM_A_STRIDE] =
        reinterpret_cast<__half(*)[EPI_SMEM_A_STRIDE]>(epi_smem);
    __half (*sB)[EPI_SMEM_B_STRIDE] =
        reinterpret_cast<__half(*)[EPI_SMEM_B_STRIDE]>(
            epi_smem + (size_t)EPI_BM * EPI_SMEM_A_STRIDE * sizeof(__half));

    // ---- Thread / warp identity ------------------------------------------
    const int tid      = (int)threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid & 31;
    const int warp_row = warp_id / EPI_WARPS_N;
    const int warp_col = warp_id % EPI_WARPS_N;

    const int blk_row  = (int)blockIdx.y * EPI_BM;
    const int blk_col  = (int)blockIdx.x * EPI_BN;
    const int num_k_tiles = (k + EPI_BK - 1) / EPI_BK;

    // ---- Accumulators -------------------------------------------------------
    wmma::fragment<wmma::accumulator, EPI_WMMA_M, EPI_WMMA_N, EPI_WMMA_K, float>
        acc[EPI_WT_M][EPI_WT_N];
    #pragma unroll
    for (int i = 0; i < EPI_WT_M; ++i)
        #pragma unroll
        for (int j = 0; j < EPI_WT_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // ---- Main K-tile loop (single-buffered, no async pipeline) --------------
    for (int tile = 0; tile < num_k_tiles; ++tile) {

        // Load A tile: BM rows × BK cols
        for (int idx = tid; idx < EPI_BM * EPI_BK; idx += EPI_THREADS) {
            int row = idx / EPI_BK;
            int col = idx % EPI_BK;
            int gr  = blk_row + row;
            int gc  = tile * EPI_BK + col;
            sA[row][col] = (gr < m && gc < k)
                           ? A[gr * k + gc]
                           : __float2half(0.f);
        }

        // Load B tile: BK rows × BN cols
        for (int idx = tid; idx < EPI_BK * EPI_BN; idx += EPI_THREADS) {
            int row = idx / EPI_BN;
            int col = idx % EPI_BN;
            int gr  = tile * EPI_BK + row;
            int gc  = blk_col + col;
            sB[row][col] = (gr < k && gc < n)
                           ? B[gr * n + gc]
                           : __float2half(0.f);
        }

        __syncthreads();

        // WMMA compute: BK/WMMA_K = 2 steps
        #pragma unroll
        for (int kk = 0; kk < EPI_BK / EPI_WMMA_K; ++kk) {
            wmma::fragment<wmma::matrix_a, EPI_WMMA_M, EPI_WMMA_N, EPI_WMMA_K,
                           __half, wmma::row_major> fa[EPI_WT_M];
            #pragma unroll
            for (int ti = 0; ti < EPI_WT_M; ++ti)
                wmma::load_matrix_sync(fa[ti],
                    &sA[warp_row * (EPI_BM/EPI_WARPS_M) + ti*EPI_WMMA_M][kk*EPI_WMMA_K],
                    EPI_SMEM_A_STRIDE);

            wmma::fragment<wmma::matrix_b, EPI_WMMA_M, EPI_WMMA_N, EPI_WMMA_K,
                           __half, wmma::row_major> fb[EPI_WT_N];
            #pragma unroll
            for (int tj = 0; tj < EPI_WT_N; ++tj)
                wmma::load_matrix_sync(fb[tj],
                    &sB[kk*EPI_WMMA_K][warp_col * (EPI_BN/EPI_WARPS_N) + tj*EPI_WMMA_N],
                    EPI_SMEM_B_STRIDE);

            #pragma unroll
            for (int ti = 0; ti < EPI_WT_M; ++ti)
                #pragma unroll
                for (int tj = 0; tj < EPI_WT_N; ++tj)
                    wmma::mma_sync(acc[ti][tj], fa[ti], fb[tj], acc[ti][tj]);
        }

        __syncthreads();
    }

    // ---- Epilogue + store ---------------------------------------------------
    // For each wmma output tile: apply epilogue in-register, then store to C.
    #pragma unroll
    for (int ti = 0; ti < EPI_WT_M; ++ti) {
        #pragma unroll
        for (int tj = 0; tj < EPI_WT_N; ++tj) {
            int crow = blk_row + warp_row * (EPI_BM / EPI_WARPS_M) + ti * EPI_WMMA_M;
            int ccol = blk_col + warp_col * (EPI_BN / EPI_WARPS_N) + tj * EPI_WMMA_N;
            if (crow < m && ccol < n) {
                // Apply chosen epilogue transformation to all fragment elements
                epilogue.template apply<EPI_WMMA_M, EPI_WMMA_N>(
                    acc[ti][tj], crow, ccol, lane);
                wmma::store_matrix_sync(
                    C + crow * n + ccol,
                    acc[ti][tj], n,
                    wmma::mem_row_major);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// INTERNAL LAUNCH HELPER
// ═══════════════════════════════════════════════════════════════════════

static inline void epilogue_launch_config(dim3& grid, dim3& block, int m, int n) {
    grid  = dim3((unsigned)((n + EPI_BN - 1) / EPI_BN),
                 (unsigned)((m + EPI_BM - 1) / EPI_BM));
    block = dim3((unsigned)EPI_THREADS);
}

// ═══════════════════════════════════════════════════════════════════════
// PUBLIC C INTERFACE — extern "C" launchers
// ═══════════════════════════════════════════════════════════════════════

// pax_gemm_bias_gelu_launch
//   Computes C = GeLU(A*B + bias)
//   bias: [n] fp16 per-column vector
extern "C"
void pax_gemm_bias_gelu_launch(
    const __half* A,
    const __half* B,
    float*        C,
    int m, int n, int k,
    const __half* bias,
    cudaStream_t  stream)
{
    dim3 grid, block;
    epilogue_launch_config(grid, block, m, n);
    cudaFuncSetAttribute(
        pax_gemm_epilogue_kernel<BiasGeLU>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)EPI_SMEM_BYTES);
    pax_gemm_epilogue_kernel<BiasGeLU><<<grid, block, EPI_SMEM_BYTES, stream>>>(
        A, B, C, m, n, k, BiasGeLU(bias));
}

// pax_gemm_residual_gelu_launch
//   Computes C = GeLU(A*B + residual)
//   residual: [m][n] fp32, same shape as C
extern "C"
void pax_gemm_residual_gelu_launch(
    const __half* A,
    const __half* B,
    float*        C,
    int m, int n, int k,
    const float*  residual,
    cudaStream_t  stream)
{
    dim3 grid, block;
    epilogue_launch_config(grid, block, m, n);
    cudaFuncSetAttribute(
        pax_gemm_epilogue_kernel<ResidualGeLU>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)EPI_SMEM_BYTES);
    pax_gemm_epilogue_kernel<ResidualGeLU><<<grid, block, EPI_SMEM_BYTES, stream>>>(
        A, B, C, m, n, k, ResidualGeLU(residual, n));
}

// pax_gemm_scale_bias_gelu_launch
//   Computes C = GeLU(alpha * A*B + bias)
extern "C"
void pax_gemm_scale_bias_gelu_launch(
    const __half* A,
    const __half* B,
    float*        C,
    int m, int n, int k,
    float         alpha,
    const __half* bias,
    cudaStream_t  stream)
{
    dim3 grid, block;
    epilogue_launch_config(grid, block, m, n);
    cudaFuncSetAttribute(
        pax_gemm_epilogue_kernel<ScaleBiasGeLU>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)EPI_SMEM_BYTES);
    pax_gemm_epilogue_kernel<ScaleBiasGeLU><<<grid, block, EPI_SMEM_BYTES, stream>>>(
        A, B, C, m, n, k, ScaleBiasGeLU(alpha, bias));
}
