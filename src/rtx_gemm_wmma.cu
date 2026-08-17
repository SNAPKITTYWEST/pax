// PAX GEMM WMMA Reference — Ampere (sm_86) Tensor Cores
// nvcuda::wmma portable verified baseline
// CTA Tile: 128x128x32, 8 warps (256 threads)
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
// Authors: Ahmad Ali Parr — Jessica Westerhoff
// License: BSL-1.1 / AGPL-3.0 / MPL-2.0

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 4, WARPS_N = 2;
constexpr int WARPS_TOTAL = WARPS_M * WARPS_N;     // 8
constexpr int THREADS_PER_BLOCK = WARPS_TOTAL * 32; // 256
constexpr int SMEM_A_STRIDE = BK + 8;  // 40 — eliminates bank conflicts
constexpr int SMEM_B_STRIDE = BN + 8;  // 136

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void pax_gemm_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half sA[BM][SMEM_A_STRIDE];
    __shared__ half sB[BK][SMEM_B_STRIDE];

    const int tid    = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int load_a_row = tid / (BK / 8);
    const int load_a_col = (tid % (BK / 8)) * 8;
    const int load_b_row = tid / (BN / 8);
    const int load_b_col = (tid % (BN / 8)) * 8;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (int k_outer = 0; k_outer < K; k_outer += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += (THREADS_PER_BLOCK / (BK / 8))) {
            int row = load_a_row + i;
            if (row >= BM) continue;
            int gmem_m = block_m + row, gmem_k = k_outer + load_a_col;
            if (gmem_m < M && gmem_k < K)
                *reinterpret_cast<uint4*>(&sA[row][load_a_col]) =
                    *reinterpret_cast<const uint4*>(&A[gmem_m * K + gmem_k]);
            else
                *reinterpret_cast<uint4*>(&sA[row][load_a_col]) = make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (int i = 0; i < BK; i += (THREADS_PER_BLOCK / (BN / 8))) {
            int row = load_b_row + i;
            if (row >= BK) continue;
            int gmem_k = k_outer + row, gmem_n = block_n + load_b_col;
            if (gmem_k < K && gmem_n < N)
                *reinterpret_cast<uint4*>(&sB[row][load_b_col]) =
                    *reinterpret_cast<const uint4*>(&B[gmem_k * N + gmem_n]);
            else
                *reinterpret_cast<uint4*>(&sB[row][load_b_col]) = make_uint4(0,0,0,0);
        }
        __syncthreads();

        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a_frag[i], &sA[(warp_m*32)+(i*WMMA_M)][k_inner], SMEM_A_STRIDE);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                wmma::load_matrix_sync(b_frag[j], &sB[k_inner][(warp_n*64)+(j*WMMA_N)], SMEM_B_STRIDE);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int out_m = block_m + (warp_m*32) + (i*WMMA_M);
            int out_n = block_n + (warp_n*64) + (j*WMMA_N);
            if (out_m < M && out_n < N)
                wmma::store_matrix_sync(&C[out_m*N+out_n], c_frag[i][j], N, wmma::mem_row_major);
        }
}

extern "C" void pax_gemm_wmma_launch(
    const half* A, const half* B, float* C,
    int M, int N, int K, cudaStream_t stream
) {
    pax_gemm_wmma_kernel<<<dim3((N+BN-1)/BN,(M+BM-1)/BM), dim3(THREADS_PER_BLOCK), 0, stream>>>(A, B, C, M, N, K);
}
