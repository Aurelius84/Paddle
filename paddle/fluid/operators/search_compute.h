/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <immintrin.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
void call_gemm(const math::BlasT<DeviceContext, T>& blas,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename T>
void call_gemm(const framework::ExecutionContext& ctx,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename DeviceContext, typename T>
void call_gemm_with_lda(const math::BlasT<DeviceContext, T>& blas,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const T alpha, const T* A, const T* B,
                        const T beta, T* C, int lda) {
  int ldb = (TransB == CblasNoTrans) ? N : K;

  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename T>
void call_gemm_batched(const framework::ExecutionContext& ctx,
                       const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const T alpha, const T** A, const T** B,
                       const T beta, T** C, const int batch) {
  for (int i = 0; i < batch; ++i) {
    call_gemm(ctx, TransA, TransB, M, N, K, alpha, A[i], B[i], beta, C[i]);
  }
}

#ifndef TYPE_USE_FLOAT
#define TYPE_USE_FLOAT
#endif
#ifndef USE_SSE
#define USE_SSE
#endif

#if defined(TYPE_USE_FLOAT)

#define __m256x __m256
#define __m128x __m128

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int AVX_CUT_LEN_MASK = 7U;
static const unsigned int SSE_CUT_LEN_MASK = 3U;

#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss

#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_max_px _mm_max_ps

#endif

template <typename T>
void lego_cpu_axpby(const int N, const T alpha, const T* x, const T beta,
                    T* y) {
#ifdef TYPE_USE_FLOAT
  cblas_saxpby(N, alpha, x, 1, beta, y, 1);
#else
  cblas_daxpby(N, alpha, x, 1, beta, y, 1);
#endif
}

template <typename T>
inline void sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(USE_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }

#elif defined(USE_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }

#endif
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

template <typename T>
inline void sse_max(const T* input_f, T* max_data, int* max_index,
                    size_t M_f, size_t N) {
  if (M_f == 0) {
    return;
  }
  unsigned int j = 0;
  unsigned int lll = 0;

  // init by first row
  memcpy(max_data, input_f, sizeof(T) * N);
  const T* input = input_f + N;
  size_t m = M_f - 1;
#if defined(USE_AVX)
  lll = N & ~AVX_CUT_LEN_MASK;
  __m256x mm0;
  for (j = 0; j < lll; j += AVX_STEP_SIZE) {
    mm0 = _mm256_load_px(max_data + j);
    const DTYPE* input_t = input + j;
    for (size_t i = 0; i < m; i++) {
      const DTYPE* input_data = input_t + i * N;
      __m256x mm_input = _mm256_load_px(input_data);
      mm0 = _mm256_max_px(mm_input, mm0);
      mm_input = _mm256_load_px(input_data + AVX_STEP_SIZE);
    }
    _mm256_store_px(max_data + j, mm0);
  }
#elif defined(USE_SSE)
  lll = N & ~SSE_CUT_LEN_MASK;
  __m128x mm0;
  for (j = 0; j < lll; j += SSE_STEP_SIZE) {
    mm0 = _mm_load_px(max_data + j);
    const T* input_t = input + j;
    for (size_t i = 0; i < m; i++) {
      const T* input_data = input_t + i * N;
      __m128x mm_input = _mm_load_px(input_data);
      mm0 = _mm_max_px(mm_input, mm0);
    }
    _mm_store_px(max_data + j, mm0);
  }
#endif
  for (; j < N; ++j) {
    for (size_t i = 0; i < m; i++) {
      max_data[j] = std::max(max_data[j], input[i * N + j]);
    }
  }
  if (max_index != NULL) {
    // find max_index
    for (int i = 0; i < N; i++) {
      T max = max_data[i];
      for (int j = 0; j < M_f; j++) {
        if (max == input_f[i + j * N]) {
          max_index[i] = j;
          break;
        }
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle
