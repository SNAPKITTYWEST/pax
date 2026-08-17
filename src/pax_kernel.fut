-- PAX Functional Kernels: Vector Ops + GEMM
-- Compiles to CUDA via: futhark cuda --library
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

def pax_vector_scale [n] (alpha: f32) (xs: [n]f32) : [n]f32 =
  map (\x -> x * alpha) xs

def pax_vector_add [n] (xs: [n]f32) (ys: [n]f32) : [n]f32 =
  map2 (\x y -> x + y) xs ys

def pax_vector_dot [n] (xs: [n]f32) (ys: [n]f32) : f32 =
  reduce (\a b -> a + b) 0.0 (map2 (\x y -> x * y) xs ys)

-- GEMM: FP16 inputs, FP32 accumulation
-- Size-dependent types enforce [M][K] x [K][N] -> [M][N] at compile time
def pax_gemm [M][N][K] (A: [M][K]f16) (B: [K][N]f16) : [M][N]f32 =
  map (\i ->
    map (\j ->
      reduce (\acc k ->
        acc + f32.f16 A[i,k] * f32.f16 B[k,j]
      ) 0.0 (iota K)
    ) (iota N)
  ) (iota M)

def pax_batch_gemm [B][M][N][K] (As: [B][M][K]f16) (Bs: [B][K][N]f16) : [B][M][N]f32 =
  map (\(A,Bm) -> pax_gemm A Bm) (zip As Bs)

-- Entry points (exposed to C host via Futhark --library backend)
entry main [n] (alpha: f32) (xs: [n]f32) : [n]f32 = pax_vector_scale alpha xs
entry add [n] (xs: [n]f32) (ys: [n]f32) : [n]f32 = pax_vector_add xs ys
entry dot [n] (xs: [n]f32) (ys: [n]f32) : f32 = pax_vector_dot xs ys
entry gemm [M][N][K] (A: [M][K]f16) (B: [K][N]f16) : [M][N]f32 = pax_gemm A B
entry batch_gemm [B][M][N][K] (As: [B][M][K]f16) (Bs: [B][K][N]f16) : [B][M][N]f32 = pax_batch_gemm As Bs
