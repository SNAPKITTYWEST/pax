// PAX Vector Ops — RTX 3080 (Ampere, sm_86)
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
// Authors: Ahmad Ali Parr — Jessica Westerhoff
// License: BSL-1.1 / AGPL-3.0 / MPL-2.0

#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float warp_reduce_sum_xor(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp;
        asm volatile(
            "shfl.sync.xor.b32 %0, %1, %2, 0xffffffff;"
            : "=f"(tmp) : "f"(val), "r"(offset)
        );
        val += tmp;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_bfly(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp;
        asm volatile(
            "shfl.sync.bfly.b32 %0, %1, %2, 0xffffffff;"
            : "=f"(tmp) : "f"(val), "r"(offset)
        );
        val += tmp;
    }
    return val;
}

extern "C" __global__ void rtx_vector_scale_kernel(
    float* __restrict__ d_out,
    const float* __restrict__ d_in,
    const float alpha,
    int64_t N
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) d_out[idx] = d_in[idx] * alpha;
}

extern "C" __global__ void rtx_custom_fused_kernel(
    float* __restrict__ d_out,
    const float* __restrict__ d_in,
    const float alpha,
    int64_t N
) {
    const int64_t idx  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id  = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x / 32;

    float val = (idx < N) ? d_in[idx] * alpha : 0.0f;
    val = warp_reduce_sum_xor(val);

    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_xor(v);
        if (lane_id == 0) atomicAdd(d_out, v);
    }
}
