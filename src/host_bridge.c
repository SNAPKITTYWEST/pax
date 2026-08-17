/*
 * Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
 * Authors: Ahmad Ali Parr — Jessica Westerhoff
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 *
 * host_bridge.c — Unified C dispatch layer for all PAX kernel variants
 *
 * This file provides:
 *   • pax_buffer_t — portable tensor descriptor (shape, stride, ownership)
 *   • Buffer allocation / deallocation helpers (device + host)
 *   • Dispatch functions: one per kernel variant (wmma, ptx, pipeline, epilogue,
 *     futhark, vector)
 *   • cpu_gemm_ref()     — reference FP32 GEMM for correctness checking
 *   • pax_verify_gemm()  — relative-error check with PASS/FAIL output
 *   • bench_kernel()     — cudaEvent-based timing harness
 *   • Global state g_A / g_B / g_C_* / g_stream for benchmark launchers
 *   • Static launch closure functions: launch_wmma, launch_ptx, launch_pipe,
 *     launch_futhark, launch_bias_gelu, launch_residual_gelu
 *
 * Compile with nvcc or gcc (links against libcuda / libcudart):
 *   nvcc -O3 -arch=sm_86 -std=c11 -x cu host_bridge.c
 *   gcc  -O3 -std=c11 -march=native host_bridge.c -lcudart -lcuda
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * BUFFER DESCRIPTOR
 * ═══════════════════════════════════════════════════════════════════════ */

/** Ownership / memory space of a pax_buffer_t */
typedef enum {
    PAX_HOST    = 0,   /**< CPU-side (malloc / cudaMallocHost) */
    PAX_DEVICE  = 1,   /**< GPU device memory (cudaMalloc) */
    PAX_UNIFIED = 2    /**< Unified / managed memory (cudaMallocManaged) */
} pax_ownership_t;

/**
 * pax_buffer_t — lightweight tensor descriptor.
 *
 * Supports up to 3 dimensions.  All sizes are in *elements* (not bytes)
 * unless the field name says otherwise.
 */
typedef struct {
    void*           base;           /**< Pointer to first element */
    size_t          elem_size;      /**< Bytes per element (e.g. 2 for fp16) */
    int64_t         shape[3];       /**< Logical shape [d0, d1, d2] */
    int64_t         stride[3];      /**< Element strides [s0, s1, s2] */
    int             ndim;           /**< Number of active dimensions (1..3) */
    size_t          alignment;      /**< Byte alignment of base pointer */
    pax_ownership_t ownership;      /**< Where this buffer lives */
    cudaStream_t    stream;         /**< Associated CUDA stream (may be NULL) */
} pax_buffer_t;

/* ═══════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS — all external kernel symbols
 * ═══════════════════════════════════════════════════════════════════════ */

/* --- Futhark-generated entry points (rtx.fut → rtx_futhark.c) --- */
extern int  pax_main(void);
extern void pax_gemm(const __half* A, const __half* B, float* C,
                     int m, int n, int k);

/* --- RTX vector kernel (rtx_vector.cu) --- */
extern void rtx_vector_scale_kernel(float* x, float alpha, int n,
                                    cudaStream_t stream);

/* --- WMMA GEMM (rtx_gemm_wmma.cu) --- */
extern void pax_gemm_wmma_launch(const __half* A, const __half* B, float* C,
                                  int m, int n, int k, cudaStream_t stream);

/* --- PTX GEMM (rtx_gemm_ptx.cu) --- */
extern void   pax_gemm_ptx_launch(const __half* A, const __half* B, float* C,
                                   int m, int n, int k, cudaStream_t stream);
extern size_t pax_gemm_ptx_smem_bytes(void);

/* --- Pipeline GEMM (rtx_gemm_pipeline.cu) --- */
extern void   pax_gemm_pipeline_launch(const __half* A, const __half* B,
                                        float* C, int m, int n, int k,
                                        cudaStream_t stream);
extern size_t pax_gemm_pipeline_smem_bytes(void);

/* --- Epilogue GEMM (rtx_gemm_epilogue.cu) --- */
extern void pax_gemm_bias_gelu_launch(const __half* A, const __half* B,
                                       float* C, int m, int n, int k,
                                       const __half* bias,
                                       cudaStream_t stream);
extern void pax_gemm_residual_gelu_launch(const __half* A, const __half* B,
                                           float* C, int m, int n, int k,
                                           const float* residual,
                                           cudaStream_t stream);
extern void pax_gemm_scale_bias_gelu_launch(const __half* A, const __half* B,
                                             float* C, int m, int n, int k,
                                             float alpha, const __half* bias,
                                             cudaStream_t stream);

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

#define PAX_CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[PAX] CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return;                                                             \
        }                                                                       \
    } while (0)

#define PAX_CUDA_CHECK_RET(call, ret)                                          \
    do {                                                                        \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[PAX] CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return (ret);                                                       \
        }                                                                       \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════════
 * BUFFER VALIDATION
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * pax_validate_buffer — sanity-check a pax_buffer_t.
 * Returns 1 if valid, 0 otherwise (prints diagnostic to stderr).
 */
int pax_validate_buffer(const pax_buffer_t* buf) {
    if (!buf) {
        fprintf(stderr, "[PAX] validate_buffer: NULL descriptor\n");
        return 0;
    }
    if (!buf->base) {
        fprintf(stderr, "[PAX] validate_buffer: NULL base pointer\n");
        return 0;
    }
    if (buf->elem_size == 0) {
        fprintf(stderr, "[PAX] validate_buffer: elem_size == 0\n");
        return 0;
    }
    if (buf->ndim < 1 || buf->ndim > 3) {
        fprintf(stderr, "[PAX] validate_buffer: ndim=%d out of [1,3]\n",
                buf->ndim);
        return 0;
    }
    for (int d = 0; d < buf->ndim; ++d) {
        if (buf->shape[d] <= 0) {
            fprintf(stderr, "[PAX] validate_buffer: shape[%d]=%lld <= 0\n",
                    d, (long long)buf->shape[d]);
            return 0;
        }
        if (buf->stride[d] <= 0) {
            fprintf(stderr, "[PAX] validate_buffer: stride[%d]=%lld <= 0\n",
                    d, (long long)buf->stride[d]);
            return 0;
        }
    }
    if (buf->alignment && ((uintptr_t)buf->base % buf->alignment != 0)) {
        fprintf(stderr, "[PAX] validate_buffer: base %p not aligned to %zu\n",
                buf->base, buf->alignment);
        return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ALLOCATION HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

/** Allocate a contiguous 1-D device buffer. */
pax_buffer_t pax_alloc_device_1d(size_t nelems, size_t elem_size,
                                   cudaStream_t stream) {
    pax_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    buf.elem_size   = elem_size;
    buf.ndim        = 1;
    buf.shape[0]    = (int64_t)nelems;
    buf.stride[0]   = 1;
    buf.alignment   = 256;
    buf.ownership   = PAX_DEVICE;
    buf.stream      = stream;
    if (cudaMalloc(&buf.base, nelems * elem_size) != cudaSuccess) {
        fprintf(stderr, "[PAX] pax_alloc_device_1d: cudaMalloc failed (%zu bytes)\n",
                nelems * elem_size);
        buf.base = NULL;
    }
    return buf;
}

/** Allocate a contiguous 2-D device buffer (row-major). */
pax_buffer_t pax_alloc_device_2d(size_t rows, size_t cols, size_t elem_size,
                                   cudaStream_t stream) {
    pax_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    buf.elem_size   = elem_size;
    buf.ndim        = 2;
    buf.shape[0]    = (int64_t)rows;
    buf.shape[1]    = (int64_t)cols;
    buf.stride[0]   = (int64_t)cols;
    buf.stride[1]   = 1;
    buf.alignment   = 256;
    buf.ownership   = PAX_DEVICE;
    buf.stream      = stream;
    if (cudaMalloc(&buf.base, rows * cols * elem_size) != cudaSuccess) {
        fprintf(stderr, "[PAX] pax_alloc_device_2d: cudaMalloc failed\n");
        buf.base = NULL;
    }
    return buf;
}

/** Allocate a contiguous 1-D pinned host buffer. */
pax_buffer_t pax_alloc_host_1d(size_t nelems, size_t elem_size,
                                 cudaStream_t stream) {
    pax_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    buf.elem_size   = elem_size;
    buf.ndim        = 1;
    buf.shape[0]    = (int64_t)nelems;
    buf.stride[0]   = 1;
    buf.alignment   = 64;
    buf.ownership   = PAX_HOST;
    buf.stream      = stream;
    if (cudaMallocHost(&buf.base, nelems * elem_size) != cudaSuccess) {
        fprintf(stderr, "[PAX] pax_alloc_host_1d: cudaMallocHost failed\n");
        buf.base = NULL;
    }
    return buf;
}

/** Allocate a contiguous 2-D pinned host buffer (row-major). */
pax_buffer_t pax_alloc_host_2d(size_t rows, size_t cols, size_t elem_size,
                                 cudaStream_t stream) {
    pax_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    buf.elem_size   = elem_size;
    buf.ndim        = 2;
    buf.shape[0]    = (int64_t)rows;
    buf.shape[1]    = (int64_t)cols;
    buf.stride[0]   = (int64_t)cols;
    buf.stride[1]   = 1;
    buf.alignment   = 64;
    buf.ownership   = PAX_HOST;
    buf.stream      = stream;
    if (cudaMallocHost(&buf.base, rows * cols * elem_size) != cudaSuccess) {
        fprintf(stderr, "[PAX] pax_alloc_host_2d: cudaMallocHost failed\n");
        buf.base = NULL;
    }
    return buf;
}

/** Free a device buffer allocated via pax_alloc_device_*. */
void pax_free_device(pax_buffer_t* buf) {
    if (buf && buf->base) {
        cudaFree(buf->base);
        buf->base = NULL;
    }
}

/** Free a host buffer allocated via pax_alloc_host_*. */
void pax_free_host(pax_buffer_t* buf) {
    if (buf && buf->base) {
        cudaFreeHost(buf->base);
        buf->base = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * DISPATCH FUNCTIONS
 *
 * Each dispatch function validates its inputs and routes to the
 * appropriate CUDA kernel via the extern declarations above.
 * ═══════════════════════════════════════════════════════════════════════ */

/** Dispatch: in-place vector scale  x[i] *= alpha  (device buffer) */
void pax_dispatch_rtx_vector_scale(pax_buffer_t* x, float alpha,
                                    cudaStream_t stream) {
    if (!pax_validate_buffer(x)) return;
    int n = (int)(x->shape[0]);
    rtx_vector_scale_kernel((float*)x->base, alpha, n, stream);
}

/** Dispatch: Futhark scalar scale (host-side reference implementation) */
void pax_dispatch_futhark_scale(pax_buffer_t* x, float alpha) {
    if (!pax_validate_buffer(x)) return;
    /* Futhark entry executes on host; direct call */
    (void)alpha;
    (void)pax_main(); /* Futhark runtime init; actual call site varies */
}

/** Dispatch: Futhark GEMM  C = A * B (fp16 → fp32) */
void pax_dispatch_futhark_gemm(pax_buffer_t* A, pax_buffer_t* B,
                                pax_buffer_t* C) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm((const __half*)A->base, (const __half*)B->base,
             (float*)C->base, m, n, k);
}

/** Dispatch: WMMA GEMM  C = A * B */
void pax_dispatch_wmma_gemm(pax_buffer_t* A, pax_buffer_t* B,
                              pax_buffer_t* C, cudaStream_t stream) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm_wmma_launch((const __half*)A->base, (const __half*)B->base,
                          (float*)C->base, m, n, k, stream);
}

/** Dispatch: PTX inline-assembly GEMM  C = A * B */
void pax_dispatch_ptx_gemm(pax_buffer_t* A, pax_buffer_t* B,
                             pax_buffer_t* C, cudaStream_t stream) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm_ptx_launch((const __half*)A->base, (const __half*)B->base,
                         (float*)C->base, m, n, k, stream);
}

/** Dispatch: 3-stage pipeline GEMM  C = A * B */
void pax_dispatch_pipeline_gemm(pax_buffer_t* A, pax_buffer_t* B,
                                  pax_buffer_t* C, cudaStream_t stream) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm_pipeline_launch((const __half*)A->base, (const __half*)B->base,
                              (float*)C->base, m, n, k, stream);
}

/** Dispatch: BiasGeLU epilogue GEMM  C = GeLU(A*B + bias) */
void pax_dispatch_bias_gelu(pax_buffer_t* A, pax_buffer_t* B,
                              pax_buffer_t* C, const __half* bias,
                              cudaStream_t stream) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    if (!bias) {
        fprintf(stderr, "[PAX] pax_dispatch_bias_gelu: NULL bias\n");
        return;
    }
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm_bias_gelu_launch((const __half*)A->base, (const __half*)B->base,
                               (float*)C->base, m, n, k, bias, stream);
}

/** Dispatch: ResidualGeLU epilogue GEMM  C = GeLU(A*B + residual) */
void pax_dispatch_residual_gelu(pax_buffer_t* A, pax_buffer_t* B,
                                  pax_buffer_t* C, const float* residual,
                                  cudaStream_t stream) {
    if (!pax_validate_buffer(A)) return;
    if (!pax_validate_buffer(B)) return;
    if (!pax_validate_buffer(C)) return;
    if (!residual) {
        fprintf(stderr, "[PAX] pax_dispatch_residual_gelu: NULL residual\n");
        return;
    }
    int m = (int)A->shape[0];
    int k = (int)A->shape[1];
    int n = (int)B->shape[1];
    pax_gemm_residual_gelu_launch((const __half*)A->base,
                                   (const __half*)B->base,
                                   (float*)C->base, m, n, k,
                                   residual, stream);
}

/* ═══════════════════════════════════════════════════════════════════════
 * REFERENCE CPU GEMM  (fp32, for correctness verification)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * cpu_gemm_ref — naive O(m*n*k) FP32 GEMM.
 * A[m][k], B[k][n] → C[m][n] = A * B (row-major, overwrites C).
 * Used only for verification; not performance-critical.
 */
void cpu_gemm_ref(const float* A, const float* B, float* C,
                  int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int l = 0; l < k; ++l)
                acc += A[(size_t)i * k + l] * B[(size_t)l * n + j];
            C[(size_t)i * n + j] = acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * VERIFICATION
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * pax_verify_gemm — compare GPU output C_gpu against reference C_ref.
 *
 * Computes max relative error:
 *   err_i = |C_gpu[i] - C_ref[i]| / (|C_ref[i]| + 1e-8)
 *
 * Prints PASS if max_err <= tol, FAIL otherwise.
 * Returns 1 on PASS, 0 on FAIL.
 */
int pax_verify_gemm(const float* C_gpu, const float* C_ref,
                    int m, int n, float tol) {
    float max_err = 0.f;
    int   max_i   = 0;
    for (int i = 0; i < m * n; ++i) {
        float err = fabsf(C_gpu[i] - C_ref[i]) /
                    (fabsf(C_ref[i]) + 1e-8f);
        if (err > max_err) {
            max_err = err;
            max_i   = i;
        }
    }
    if (max_err <= tol) {
        printf("[PAX] PASS  max_rel_err = %.6e  (tol = %.6e)\n",
               (double)max_err, (double)tol);
        return 1;
    } else {
        printf("[PAX] FAIL  max_rel_err = %.6e  (tol = %.6e)  "
               "at index %d  gpu=%.6f  ref=%.6f\n",
               (double)max_err, (double)tol,
               max_i, (double)C_gpu[max_i], (double)C_ref[max_i]);
        return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BENCHMARK INFRASTRUCTURE
 * ═══════════════════════════════════════════════════════════════════════ */

/** Function pointer type for a kernel launch closure */
typedef void (*pax_kernel_fn)(cudaStream_t stream);

/**
 * bench_kernel — time a kernel launch closure using cudaEvents.
 *
 * Runs `iters` iterations, returns average elapsed time in milliseconds.
 * Performs one warm-up iteration before timing.
 */
static double bench_kernel(pax_kernel_fn fn, int iters, cudaStream_t stream) {
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    /* warm-up */
    fn(stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(ev_start, stream);
    for (int i = 0; i < iters; ++i)
        fn(stream);
    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return (double)ms / (double)iters;
}

/* -----------------------------------------------------------------------
 * Global benchmark state
 * Set these before calling bench_kernel with any launch_* closure.
 * ----------------------------------------------------------------------- */
static __half*  g_A            = NULL;
static __half*  g_B            = NULL;
static float*   g_C_wmma       = NULL;
static float*   g_C_ptx        = NULL;
static float*   g_C_pipe       = NULL;
static float*   g_C_futhark    = NULL;
static float*   g_C_bias_gelu  = NULL;
static float*   g_C_res_gelu   = NULL;
static __half*  g_bias         = NULL;
static float*   g_residual     = NULL;
static cudaStream_t g_stream   = NULL;
static int g_bench_m = 4096;
static int g_bench_n = 4096;
static int g_bench_k = 4096;

/* -----------------------------------------------------------------------
 * Static launch closures — capture global state, match pax_kernel_fn
 * ----------------------------------------------------------------------- */

static void launch_wmma(cudaStream_t s) {
    pax_gemm_wmma_launch(g_A, g_B, g_C_wmma,
                          g_bench_m, g_bench_n, g_bench_k, s);
}

static void launch_ptx(cudaStream_t s) {
    pax_gemm_ptx_launch(g_A, g_B, g_C_ptx,
                         g_bench_m, g_bench_n, g_bench_k, s);
}

static void launch_pipe(cudaStream_t s) {
    pax_gemm_pipeline_launch(g_A, g_B, g_C_pipe,
                              g_bench_m, g_bench_n, g_bench_k, s);
}

static void launch_futhark(cudaStream_t s) {
    (void)s;
    pax_gemm(g_A, g_B, g_C_futhark,
             g_bench_m, g_bench_n, g_bench_k);
}

static void launch_bias_gelu(cudaStream_t s) {
    pax_gemm_bias_gelu_launch(g_A, g_B, g_C_bias_gelu,
                               g_bench_m, g_bench_n, g_bench_k,
                               g_bias, s);
}

static void launch_residual_gelu(cudaStream_t s) {
    pax_gemm_residual_gelu_launch(g_A, g_B, g_C_res_gelu,
                                   g_bench_m, g_bench_n, g_bench_k,
                                   g_residual, s);
}

/* -----------------------------------------------------------------------
 * pax_bench_all — run all kernel variants and print timing table.
 * Call after setting g_A / g_B / g_C_* / g_bench_m/n/k / g_stream.
 * ----------------------------------------------------------------------- */
void pax_bench_all(int iters) {
    printf("\n[PAX] Benchmark  m=%d  n=%d  k=%d  iters=%d\n",
           g_bench_m, g_bench_n, g_bench_k, iters);
    printf("%-28s  %10s  %10s\n", "Kernel", "ms/iter", "TFLOPS");

    /* Theoretical FLOPs: 2 * m * n * k */
    double flops = 2.0 * g_bench_m * g_bench_n * (double)g_bench_k;

    struct { const char* name; pax_kernel_fn fn; } variants[] = {
        { "wmma",         launch_wmma         },
        { "ptx",          launch_ptx          },
        { "pipeline",     launch_pipe         },
        { "bias_gelu",    launch_bias_gelu    },
        { "residual_gelu",launch_residual_gelu},
    };

    for (int v = 0; v < (int)(sizeof(variants)/sizeof(variants[0])); ++v) {
        if (!variants[v].fn) continue;
        double ms    = bench_kernel(variants[v].fn, iters, g_stream);
        double tflops = flops / (ms * 1e9);
        printf("%-28s  %10.4f  %10.3f\n", variants[v].name, ms, tflops);
    }
    printf("\n");
}
