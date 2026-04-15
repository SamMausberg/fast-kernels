#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "ops/rdkng/explicit_sketch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Recycled Deflated Krylov Natural Gradient (RDK-NG)
// Explicit-sketch CUDA core for one parameter block.
//
// This file implements the practical operator
//     A = Y Y^T + lambda I,
// where Y is a column-major BF16 sketch with shape [n, s].
// If the caller constructs Y from sketched per-sample gradients, then Y Y^T is a
// sketched empirical Fisher block.
//
// The solver computes an approximate natural-gradient direction v ~= A^{-1} g
// using:
//   1) a recycled low-rank basis U (BF16, column-major),
//   2) an exact reduced solve on span(U),
//   3) a short deflated PCG correction,
//   4) an append-and-orthonormalize basis refresh.
//
// Design choices for RTX 5070 Ti / Blackwell-class GPUs:
//   - BF16 inputs for Y and U to reduce memory traffic.
//   - FP32 accumulation for stability.
//   - cuBLAS GEMMEx for the large Y^T X and Y T multiplies.
//   - Small custom kernels for the tiny reduced system and basis algebra.
//   - All tensors remain on device. No host/device copies of large arrays.
//
// Compile (CUDA 12.8+ recommended for Blackwell native support):
//   nvcc -O3 -std=c++17 --use_fast_math \
//        -gencode arch=compute_120,code=sm_120 \
//        -lcublas rdkng_solver_blackwell.cu -o rdkng_solver_blackwell
// -----------------------------------------------------------------------------

#ifndef RDKNG_MAX_R
#define RDKNG_MAX_R 64
#endif

#define CHECK_CUDA(expr)                                                          \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                   cudaGetErrorString(_err), __FILE__, __LINE__);                \
      throw std::runtime_error("CUDA failure");                                  \
    }                                                                             \
  } while (0)

#define CHECK_CUBLAS(expr)                                                        \
  do {                                                                            \
    cublasStatus_t _st = (expr);                                                  \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                        \
                   static_cast<int>(_st), __FILE__, __LINE__);                   \
      throw std::runtime_error("cuBLAS failure");                                \
    }                                                                             \
  } while (0)

namespace fast_kernels::rdkng {

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }

  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < (BLOCK_SIZE + 31) / 32) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
  }
  return value;
}

__global__ void fp32_to_bf16_kernel(const float* __restrict__ in,
                                    __nv_bfloat16* __restrict__ out,
                                    int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < count; idx += stride) {
    out[idx] = __float2bfloat16(in[idx]);
  }
}

__global__ void copy_fp32_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < count; idx += stride) {
    out[idx] = in[idx];
  }
}

__global__ void zero_fp32_kernel(float* __restrict__ out, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < count; idx += stride) {
    out[idx] = 0.0f;
  }
}

__global__ void add_lambda_from_bf16_kernel(float* __restrict__ out,
                                            const __nv_bfloat16* __restrict__ x,
                                            float lambda,
                                            int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < count; idx += stride) {
    out[idx] += lambda * __bfloat162float(x[idx]);
  }
}

__global__ void add_lambda_from_fp32_kernel(float* __restrict__ out,
                                            const float* __restrict__ x,
                                            float lambda,
                                            int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < count; idx += stride) {
    out[idx] += lambda * x[idx];
  }
}

__global__ void residual_kernel(const float* __restrict__ g,
                                const float* __restrict__ Av,
                                float* __restrict__ r,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    r[idx] = g[idx] - Av[idx];
  }
}

__global__ void update_diag_ema_kernel(const __nv_bfloat16* __restrict__ Y,
                                       float* __restrict__ diag,
                                       int n,
                                       int s,
                                       float beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    float acc = 0.0f;
    for (int j = 0; j < s; ++j) {
      float y = __bfloat162float(Y[idx + j * n]);
      acc += y * y;
    }
    diag[idx] = beta * diag[idx] + (1.0f - beta) * acc;
  }
}

__global__ void apply_diag_preconditioner_kernel(const float* __restrict__ r,
                                                 const float* __restrict__ diag,
                                                 float lambda,
                                                 float* __restrict__ z,
                                                 int n,
                                                 int use_diag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (use_diag) {
    for (; idx < n; idx += stride) {
      z[idx] = r[idx] / (diag[idx] + lambda);
    }
  } else {
    for (; idx < n; idx += stride) {
      z[idx] = r[idx];
    }
  }
}

template <int BLOCK_SIZE>
__global__ void bf16_matT_vec_kernel(const __nv_bfloat16* __restrict__ M,
                                     const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int n,
                                     int cols) {
  int col = blockIdx.x;
  if (col >= cols) return;

  float sum = 0.0f;
  const __nv_bfloat16* col_ptr = M + static_cast<size_t>(col) * n;
  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    sum += __bfloat162float(col_ptr[i]) * x[i];
  }
  sum = block_reduce_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    out[col] = sum;
  }
}

template <int BLOCK_SIZE>
__global__ void fp32_matT_vec_kernel(const float* __restrict__ M,
                                     const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int n,
                                     int cols) {
  int col = blockIdx.x;
  if (col >= cols) return;

  float sum = 0.0f;
  const float* col_ptr = M + static_cast<size_t>(col) * n;
  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    sum += col_ptr[i] * x[i];
  }
  sum = block_reduce_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    out[col] = sum;
  }
}

template <int BLOCK_SIZE>
__global__ void gram_bf16_fp32_kernel(const __nv_bfloat16* __restrict__ U,
                                      const float* __restrict__ AU,
                                      float* __restrict__ H,
                                      int n,
                                      int r,
                                      float jitter) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  if (i >= r || j >= r) return;

  float sum = 0.0f;
  const __nv_bfloat16* ui = U + static_cast<size_t>(i) * n;
  const float* auj = AU + static_cast<size_t>(j) * n;
  for (int t = threadIdx.x; t < n; t += BLOCK_SIZE) {
    sum += __bfloat162float(ui[t]) * auj[t];
  }
  sum = block_reduce_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    H[i * r + j] = sum + ((i == j) ? jitter : 0.0f);
  }
}

__global__ void basis_lincomb_from_bf16_kernel(const __nv_bfloat16* __restrict__ U,
                                               const float* __restrict__ coeff,
                                               float* __restrict__ out,
                                               int n,
                                               int r,
                                               float alpha,
                                               int add) {
  __shared__ float c[RDKNG_MAX_R];
  if (threadIdx.x < r) {
    c[threadIdx.x] = coeff[threadIdx.x];
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    float acc = 0.0f;
    for (int j = 0; j < r; ++j) {
      acc += __bfloat162float(U[idx + static_cast<size_t>(j) * n]) * c[j];
    }
    if (add) {
      out[idx] += alpha * acc;
    } else {
      out[idx] = alpha * acc;
    }
  }
}

__global__ void cholesky_inplace_kernel(float* H, int r) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int k = 0; k < r; ++k) {
    float sum = H[k * r + k];
    for (int j = 0; j < k; ++j) {
      float Ljk = H[k * r + j];
      sum -= Ljk * Ljk;
    }
    sum = fmaxf(sum, 1.0e-12f);
    float Lkk = sqrtf(sum);
    H[k * r + k] = Lkk;

    for (int i = k + 1; i < r; ++i) {
      float s = H[i * r + k];
      for (int j = 0; j < k; ++j) {
        s -= H[i * r + j] * H[k * r + j];
      }
      H[i * r + k] = s / Lkk;
    }
    for (int j = k + 1; j < r; ++j) {
      H[k * r + j] = 0.0f;
    }
  }
}

__global__ void cholesky_solve_kernel(const float* L,
                                      const float* rhs,
                                      float* x,
                                      int r) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float y[RDKNG_MAX_R];
  for (int i = 0; i < r; ++i) {
    float sum = rhs[i];
    for (int j = 0; j < i; ++j) {
      sum -= L[i * r + j] * y[j];
    }
    y[i] = sum / L[i * r + i];
  }
  for (int i = r - 1; i >= 0; --i) {
    float sum = y[i];
    for (int j = i + 1; j < r; ++j) {
      sum -= L[j * r + i] * x[j];
    }
    x[i] = sum / L[i * r + i];
  }
}

__global__ void copy_small_kernel(const float* src, float* dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) dst[idx] = src[idx];
}

__global__ void write_bf16_column_from_fp32_kernel(const float* __restrict__ x,
                                                   __nv_bfloat16* __restrict__ U,
                                                   int n,
                                                   int col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  __nv_bfloat16* col_ptr = U + static_cast<size_t>(col) * n;
  for (; idx < n; idx += stride) {
    col_ptr[idx] = __float2bfloat16(x[idx]);
  }
}

struct ExplicitSketchSolver::Impl {
  Impl(int n, int s, int r, int max_cg) : n_(n), s_(s), r_(r), max_cg_(max_cg) {
    if (n_ <= 0 || s_ <= 0 || max_cg_ < 0) {
      throw std::invalid_argument("Invalid RDKNG dimensions");
    }
    if (r_ < 0 || r_ > RDKNG_MAX_R) {
      throw std::invalid_argument("Rank r must satisfy 0 <= r <= RDKNG_MAX_R");
    }

    CHECK_CUBLAS(cublasCreate(&handle_));
    CHECK_CUBLAS(cublasSetStream(handle_, stream_));
    CHECK_CUBLAS(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));

    allocate();
  }

  ~Impl() {
    release();
  }

  [[nodiscard]] int n() const {
    return n_;
  }

  [[nodiscard]] int s() const {
    return s_;
  }

  [[nodiscard]] int r() const {
    return r_;
  }

  [[nodiscard]] int max_cg() const {
    return max_cg_;
  }

  [[nodiscard]] std::size_t workspace_bytes() const;

  void step(const __nv_bfloat16* Y,          // [n, s] column-major, BF16
            const float* grad,               // [n] FP32
            const __nv_bfloat16* U_in,       // [n, r] column-major, BF16
            __nv_bfloat16* U_out,            // [n, r] column-major, BF16 or nullptr
            float lambda,
            int cg_iters,
            float tol,
            float* out_v,                    // [n] FP32
            float* diag_ema = nullptr,       // [n] FP32 or nullptr
            float diag_ema_beta = 0.95f,
            int basis_replace_col = -1,
            float basis_append_threshold = 1.0e-4f,
            cudaStream_t stream = 0,
            StepStats* host_stats = nullptr) {
    if (Y == nullptr || grad == nullptr || out_v == nullptr) {
      throw std::invalid_argument("Y, grad, and out_v must point to valid device buffers");
    }
    if (cg_iters < 0 || cg_iters > max_cg_) {
      throw std::invalid_argument("cg_iters out of range");
    }
    if (lambda <= 0.0f) {
      throw std::invalid_argument("lambda must be positive");
    }
    if (tol <= 0.0f) {
      throw std::invalid_argument("tol must be positive");
    }
    stream_ = stream;
    CHECK_CUBLAS(cublasSetStream(handle_, stream_));

    const int threads = 256;
    const int blocks_n = std::max(1, (n_ + threads - 1) / threads);

    if (diag_ema != nullptr) {
      update_diag_ema_kernel<<<blocks_n, threads, 0, stream_>>>(Y, diag_ema, n_, s_, diag_ema_beta);
      CHECK_CUDA(cudaGetLastError());
    }

    bool use_basis = (r_ > 0 && U_in != nullptr);

    if (use_basis) {
      apply_A_to_basis(Y, U_in, lambda, d_AU_, d_tmp_fp32_, d_tmp_bf16_);

      dim3 gridH(static_cast<unsigned>(r_), static_cast<unsigned>(r_), 1);
      gram_bf16_fp32_kernel<256><<<gridH, 256, 0, stream_>>>(U_in, d_AU_, d_H_, n_, r_, 1.0e-6f);
      CHECK_CUDA(cudaGetLastError());

      bf16_matT_vec_kernel<256><<<r_, 256, 0, stream_>>>(U_in, grad, d_rhs_, n_, r_);
      CHECK_CUDA(cudaGetLastError());

      cholesky_inplace_kernel<<<1, 1, 0, stream_>>>(d_H_, r_);
      CHECK_CUDA(cudaGetLastError());

      cholesky_solve_kernel<<<1, 1, 0, stream_>>>(d_H_, d_rhs_, d_coeff_, r_);
      CHECK_CUDA(cudaGetLastError());

      zero_fp32_kernel<<<blocks_n, threads, 0, stream_>>>(out_v, n_);
      CHECK_CUDA(cudaGetLastError());
      basis_lincomb_from_bf16_kernel<<<blocks_n, threads, 0, stream_>>>(U_in, d_coeff_, out_v, n_, r_, 1.0f, 1);
      CHECK_CUDA(cudaGetLastError());
    } else {
      zero_fp32_kernel<<<blocks_n, threads, 0, stream_>>>(out_v, n_);
      CHECK_CUDA(cudaGetLastError());
    }

    apply_A_to_fp32_vectors(Y, out_v, 1, lambda, d_Av_, d_x_bf16_, d_tmp_fp32_, d_tmp_bf16_);
    residual_kernel<<<blocks_n, threads, 0, stream_>>>(grad, d_Av_, d_r_, n_);
    CHECK_CUDA(cudaGetLastError());

    float g_norm = 0.0f;
    float r_norm = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle_, n_, grad, 1, &g_norm));
    CHECK_CUBLAS(cublasSnrm2(handle_, n_, d_r_, 1, &r_norm));

    if (host_stats) {
      host_stats->initial_residual_norm = r_norm;
      host_stats->final_residual_norm = r_norm;
      host_stats->cg_steps_taken = 0;
      host_stats->basis_refreshed = 0;
    }

    if (cg_iters == 0 || r_norm <= tol * std::max(1.0f, g_norm)) {
      bool basis_refreshed =
          maybe_refresh_basis(U_in, U_out, out_v, basis_replace_col, basis_append_threshold);
      if (host_stats) {
        host_stats->basis_refreshed = basis_refreshed ? 1 : 0;
      }
      return;
    }

    // z = M^{-1} r
    apply_diag_preconditioner_kernel<<<blocks_n, threads, 0, stream_>>>(
        d_r_, diag_ema, lambda, d_z_, n_, diag_ema != nullptr ? 1 : 0);
    CHECK_CUDA(cudaGetLastError());
    if (use_basis) {
      project_to_basis_complement(U_in, d_AU_, d_H_, d_z_);
    }

    CHECK_CUBLAS(cublasScopy(handle_, n_, d_z_, 1, d_p_, 1));

    float rz_old = 0.0f;
    CHECK_CUBLAS(cublasSdot(handle_, n_, d_r_, 1, d_z_, 1, &rz_old));

    int steps_taken = 0;
    for (int k = 0; k < cg_iters; ++k) {
      apply_A_to_fp32_vectors(Y, d_p_, 1, lambda, d_Ap_, d_x_bf16_, d_tmp_fp32_, d_tmp_bf16_);

      float pAp = 0.0f;
      CHECK_CUBLAS(cublasSdot(handle_, n_, d_p_, 1, d_Ap_, 1, &pAp));
      if (std::fabs(pAp) < 1.0e-20f) {
        break;
      }
      float alpha = rz_old / pAp;

      CHECK_CUBLAS(cublasSaxpy(handle_, n_, &alpha, d_p_, 1, out_v, 1));
      float neg_alpha = -alpha;
      CHECK_CUBLAS(cublasSaxpy(handle_, n_, &neg_alpha, d_Ap_, 1, d_r_, 1));

      CHECK_CUBLAS(cublasSnrm2(handle_, n_, d_r_, 1, &r_norm));
      steps_taken = k + 1;
      if (r_norm <= tol * std::max(1.0f, g_norm)) {
        break;
      }

      apply_diag_preconditioner_kernel<<<blocks_n, threads, 0, stream_>>>(
          d_r_, diag_ema, lambda, d_z_, n_, diag_ema != nullptr ? 1 : 0);
      CHECK_CUDA(cudaGetLastError());
      if (use_basis) {
        project_to_basis_complement(U_in, d_AU_, d_H_, d_z_);
      }

      float rz_new = 0.0f;
      CHECK_CUBLAS(cublasSdot(handle_, n_, d_r_, 1, d_z_, 1, &rz_new));
      if (std::fabs(rz_old) < 1.0e-30f) {
        break;
      }
      float beta = rz_new / rz_old;
      CHECK_CUBLAS(cublasSscal(handle_, n_, &beta, d_p_, 1));
      float one = 1.0f;
      CHECK_CUBLAS(cublasSaxpy(handle_, n_, &one, d_z_, 1, d_p_, 1));
      rz_old = rz_new;
    }

    if (host_stats) {
      host_stats->final_residual_norm = r_norm;
      host_stats->cg_steps_taken = steps_taken;
    }

    bool basis_refreshed =
        maybe_refresh_basis(U_in, U_out, out_v, basis_replace_col, basis_append_threshold);
    if (host_stats) {
      host_stats->basis_refreshed = basis_refreshed ? 1 : 0;
    }
  }

 private:
  int n_ = 0;
  int s_ = 0;
  int r_ = 0;
  int max_cg_ = 0;
  cudaStream_t stream_ = 0;
  cublasHandle_t handle_ = nullptr;

  // Large working buffers.
  __nv_bfloat16* d_x_bf16_ = nullptr;     // [n * 1] or [n * r]
  float* d_tmp_fp32_ = nullptr;           // [s * max(1, r)]
  __nv_bfloat16* d_tmp_bf16_ = nullptr;   // [s * max(1, r)]
  float* d_AU_ = nullptr;                 // [n * max(1, r)]

  float* d_Av_ = nullptr;                 // [n]
  float* d_r_ = nullptr;                  // [n]
  float* d_z_ = nullptr;                  // [n]
  float* d_p_ = nullptr;                  // [n]
  float* d_Ap_ = nullptr;                 // [n]
  float* d_candidate_ = nullptr;          // [n]

  // Small buffers.
  float* d_H_ = nullptr;                  // [r * r]
  float* d_rhs_ = nullptr;                // [r]
  float* d_coeff_ = nullptr;              // [r]
  float* d_proj_ = nullptr;               // [r]

  void allocate() {
    const size_t nr = static_cast<size_t>(n_) * std::max(1, r_);
    const size_t sr = static_cast<size_t>(s_) * std::max(1, r_);
    const size_t n1 = static_cast<size_t>(n_);
    const size_t rr = static_cast<size_t>(std::max(1, r_)) * std::max(1, r_);

    CHECK_CUDA(cudaMalloc(&d_x_bf16_, sizeof(__nv_bfloat16) * nr));
    CHECK_CUDA(cudaMalloc(&d_tmp_fp32_, sizeof(float) * sr));
    CHECK_CUDA(cudaMalloc(&d_tmp_bf16_, sizeof(__nv_bfloat16) * sr));
    CHECK_CUDA(cudaMalloc(&d_AU_, sizeof(float) * nr));

    CHECK_CUDA(cudaMalloc(&d_Av_, sizeof(float) * n1));
    CHECK_CUDA(cudaMalloc(&d_r_, sizeof(float) * n1));
    CHECK_CUDA(cudaMalloc(&d_z_, sizeof(float) * n1));
    CHECK_CUDA(cudaMalloc(&d_p_, sizeof(float) * n1));
    CHECK_CUDA(cudaMalloc(&d_Ap_, sizeof(float) * n1));
    CHECK_CUDA(cudaMalloc(&d_candidate_, sizeof(float) * n1));

    CHECK_CUDA(cudaMalloc(&d_H_, sizeof(float) * rr));
    CHECK_CUDA(cudaMalloc(&d_rhs_, sizeof(float) * std::max(1, r_)));
    CHECK_CUDA(cudaMalloc(&d_coeff_, sizeof(float) * std::max(1, r_)));
    CHECK_CUDA(cudaMalloc(&d_proj_, sizeof(float) * std::max(1, r_)));
  }

  void release() {
    auto free_ptr = [](void* p) {
      if (p) cudaFree(p);
    };
    free_ptr(d_x_bf16_);
    free_ptr(d_tmp_fp32_);
    free_ptr(d_tmp_bf16_);
    free_ptr(d_AU_);
    free_ptr(d_Av_);
    free_ptr(d_r_);
    free_ptr(d_z_);
    free_ptr(d_p_);
    free_ptr(d_Ap_);
    free_ptr(d_candidate_);
    free_ptr(d_H_);
    free_ptr(d_rhs_);
    free_ptr(d_coeff_);
    free_ptr(d_proj_);
    d_x_bf16_ = nullptr;
    d_tmp_fp32_ = nullptr;
    d_tmp_bf16_ = nullptr;
    d_AU_ = nullptr;
    d_Av_ = nullptr;
    d_r_ = nullptr;
    d_z_ = nullptr;
    d_p_ = nullptr;
    d_Ap_ = nullptr;
    d_candidate_ = nullptr;
    d_H_ = nullptr;
    d_rhs_ = nullptr;
    d_coeff_ = nullptr;
    d_proj_ = nullptr;
    if (handle_) {
      cublasDestroy(handle_);
      handle_ = nullptr;
    }
  }

  void apply_A_to_basis(const __nv_bfloat16* Y,
                        const __nv_bfloat16* X_basis,
                        float lambda,
                        float* out_fp32,
                        float* tmp_fp32,
                        __nv_bfloat16* tmp_bf16) {
    if (r_ <= 0) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // tmp = Y^T * X   where Y:[n,s], X:[n,r] -> tmp:[s,r]
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              s_,
                              r_,
                              n_,
                              &alpha,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              X_basis,
                              CUDA_R_16BF,
                              n_,
                              &beta,
                              tmp_fp32,
                              CUDA_R_32F,
                              s_,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int count_sr = s_ * r_;
    int threads = 256;
    int blocks = std::max(1, (count_sr + threads - 1) / threads);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream_>>>(tmp_fp32, tmp_bf16, count_sr);
    CHECK_CUDA(cudaGetLastError());

    // out = Y * tmp   where Y:[n,s], tmp:[s,r] -> out:[n,r]
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n_,
                              r_,
                              s_,
                              &alpha,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              tmp_bf16,
                              CUDA_R_16BF,
                              s_,
                              &beta,
                              out_fp32,
                              CUDA_R_32F,
                              n_,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int count_nr = n_ * r_;
    blocks = std::max(1, (count_nr + threads - 1) / threads);
    add_lambda_from_bf16_kernel<<<blocks, threads, 0, stream_>>>(out_fp32, X_basis, lambda, count_nr);
    CHECK_CUDA(cudaGetLastError());
  }

  void apply_A_to_fp32_vectors(const __nv_bfloat16* Y,
                               const float* X_fp32,
                               int k,
                               float lambda,
                               float* out_fp32,
                               __nv_bfloat16* x_bf16,
                               float* tmp_fp32,
                               __nv_bfloat16* tmp_bf16) {
    const int threads = 256;
    int count_nk = n_ * k;
    int blocks = std::max(1, (count_nk + threads - 1) / threads);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream_>>>(X_fp32, x_bf16, count_nk);
    CHECK_CUDA(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // tmp = Y^T * X   where Y:[n,s], X:[n,k] -> tmp:[s,k]
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              s_,
                              k,
                              n_,
                              &alpha,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              x_bf16,
                              CUDA_R_16BF,
                              n_,
                              &beta,
                              tmp_fp32,
                              CUDA_R_32F,
                              s_,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int count_sk = s_ * k;
    blocks = std::max(1, (count_sk + threads - 1) / threads);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream_>>>(tmp_fp32, tmp_bf16, count_sk);
    CHECK_CUDA(cudaGetLastError());

    // out = Y * tmp  where Y:[n,s], tmp:[s,k] -> out:[n,k]
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n_,
                              k,
                              s_,
                              &alpha,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              tmp_bf16,
                              CUDA_R_16BF,
                              s_,
                              &beta,
                              out_fp32,
                              CUDA_R_32F,
                              n_,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    blocks = std::max(1, (count_nk + threads - 1) / threads);
    add_lambda_from_fp32_kernel<<<blocks, threads, 0, stream_>>>(out_fp32, X_fp32, lambda, count_nk);
    CHECK_CUDA(cudaGetLastError());
  }

  void project_to_basis_complement(const __nv_bfloat16* U,
                                   const float* AU,
                                   const float* L_factor,
                                   float* z) {
    if (r_ <= 0) return;
    fp32_matT_vec_kernel<256><<<r_, 256, 0, stream_>>>(AU, z, d_rhs_, n_, r_);
    CHECK_CUDA(cudaGetLastError());

    cholesky_solve_kernel<<<1, 1, 0, stream_>>>(L_factor, d_rhs_, d_proj_, r_);
    CHECK_CUDA(cudaGetLastError());

    const int threads = 256;
    const int blocks = std::max(1, (n_ + threads - 1) / threads);
    basis_lincomb_from_bf16_kernel<<<blocks, threads, 0, stream_>>>(U, d_proj_, z, n_, r_, -1.0f, 1);
    CHECK_CUDA(cudaGetLastError());
  }

  bool maybe_refresh_basis(const __nv_bfloat16* U_in,
                           __nv_bfloat16* U_out,
                           const float* v,
                           int basis_replace_col,
                           float threshold) {
    if (U_out == nullptr || r_ <= 0) {
      return false;
    }

    if (U_in != nullptr && U_in != U_out) {
      CHECK_CUDA(cudaMemcpyAsync(U_out,
                                 U_in,
                                 sizeof(__nv_bfloat16) * static_cast<size_t>(n_) * r_,
                                 cudaMemcpyDeviceToDevice,
                                 stream_));
    }

    CHECK_CUBLAS(cublasScopy(handle_, n_, v, 1, d_candidate_, 1));

    if (U_in != nullptr) {
      bf16_matT_vec_kernel<256><<<r_, 256, 0, stream_>>>(U_in, d_candidate_, d_rhs_, n_, r_);
      CHECK_CUDA(cudaGetLastError());

      const int threads = 256;
      const int blocks = std::max(1, (n_ + threads - 1) / threads);
      basis_lincomb_from_bf16_kernel<<<blocks, threads, 0, stream_>>>(U_in, d_rhs_, d_candidate_, n_, r_, -1.0f, 1);
      CHECK_CUDA(cudaGetLastError());
    }

    float cand_norm = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle_, n_, d_candidate_, 1, &cand_norm));
    if (cand_norm <= threshold) {
      return false;
    }

    float inv_norm = 1.0f / cand_norm;
    CHECK_CUBLAS(cublasSscal(handle_, n_, &inv_norm, d_candidate_, 1));

    int replace_col = basis_replace_col;
    if (replace_col < 0 || replace_col >= r_) {
      replace_col = (U_in == nullptr) ? 0 : (r_ - 1);
    }

    const int threads = 256;
    const int blocks = std::max(1, (n_ + threads - 1) / threads);
    write_bf16_column_from_fp32_kernel<<<blocks, threads, 0, stream_>>>(d_candidate_, U_out, n_, replace_col);
    CHECK_CUDA(cudaGetLastError());
    return true;
  }
};

std::size_t ExplicitSketchSolver::Impl::workspace_bytes() const {
  const size_t nr = static_cast<size_t>(n_) * std::max(1, r_);
  const size_t sr = static_cast<size_t>(s_) * std::max(1, r_);
  const size_t n1 = static_cast<size_t>(n_);
  const size_t rr = static_cast<size_t>(std::max(1, r_)) * std::max(1, r_);
  return (sizeof(__nv_bfloat16) * nr) + (sizeof(float) * sr) + (sizeof(__nv_bfloat16) * sr) +
         (sizeof(float) * nr) + (sizeof(float) * n1 * 6) + (sizeof(float) * rr) +
         (sizeof(float) * std::max(1, r_) * 3);
}

ExplicitSketchSolver::ExplicitSketchSolver(int n, int s, int r, int max_cg)
    : impl_(std::make_unique<Impl>(n, s, r, max_cg)) {}

ExplicitSketchSolver::~ExplicitSketchSolver() = default;

int ExplicitSketchSolver::n() const {
  return impl_->n();
}

int ExplicitSketchSolver::s() const {
  return impl_->s();
}

int ExplicitSketchSolver::r() const {
  return impl_->r();
}

int ExplicitSketchSolver::max_cg() const {
  return impl_->max_cg();
}

std::size_t ExplicitSketchSolver::workspace_bytes() const {
  return impl_->workspace_bytes();
}

void ExplicitSketchSolver::step(std::uintptr_t y_ptr,
                                std::uintptr_t grad_ptr,
                                std::uintptr_t basis_in_ptr,
                                std::uintptr_t basis_out_ptr,
                                float lambda,
                                int cg_iters,
                                float tol,
                                std::uintptr_t out_v_ptr,
                                std::uintptr_t diag_ema_ptr,
                                float diag_ema_beta,
                                int basis_replace_col,
                                float basis_append_threshold,
                                std::uintptr_t stream_ptr,
                                StepStats* host_stats) {
  impl_->step(reinterpret_cast<const __nv_bfloat16*>(y_ptr),
              reinterpret_cast<const float*>(grad_ptr),
              reinterpret_cast<const __nv_bfloat16*>(basis_in_ptr),
              reinterpret_cast<__nv_bfloat16*>(basis_out_ptr),
              lambda,
              cg_iters,
              tol,
              reinterpret_cast<float*>(out_v_ptr),
              reinterpret_cast<float*>(diag_ema_ptr),
              diag_ema_beta,
              basis_replace_col,
              basis_append_threshold,
              reinterpret_cast<cudaStream_t>(stream_ptr),
              host_stats);
}

}  // namespace fast_kernels::rdkng
