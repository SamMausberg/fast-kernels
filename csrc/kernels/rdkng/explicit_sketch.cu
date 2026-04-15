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

// CUDA solver for one explicit-sketch block with recycled basis, CG, and a
// direct small-sketch fast path for A = Y Y^T + lambda I.

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

// Device-side stats returned with each step.
struct DeviceStepStats {
  float initial_residual_norm;
  float final_residual_norm;
  int cg_steps_taken;
};

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

template <int BLOCK_SIZE>
__global__ void bf16_matT_bf16_mat_kernel(const __nv_bfloat16* __restrict__ A,
                                          const __nv_bfloat16* __restrict__ B,
                                          float* __restrict__ out,
                                          int n,
                                          int a_cols,
                                          int b_cols) {
  const int a_col = blockIdx.x;
  const int b_col = blockIdx.y;
  if (a_col >= a_cols || b_col >= b_cols) return;

  float sum = 0.0f;
  const __nv_bfloat16* a_ptr = A + static_cast<size_t>(a_col) * n;
  const __nv_bfloat16* b_ptr = B + static_cast<size_t>(b_col) * n;
  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    sum += __bfloat162float(a_ptr[i]) * __bfloat162float(b_ptr[i]);
  }
  sum = block_reduce_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    out[a_col + static_cast<size_t>(b_col) * a_cols] = sum;
  }
}

__global__ void apply_small_sketch_vector_kernel(const __nv_bfloat16* __restrict__ Y,
                                                 const float* __restrict__ ytx,
                                                 const float* __restrict__ x,
                                                 float lambda,
                                                 float* __restrict__ out,
                                                 int n,
                                                 int s) {
  __shared__ float coeff[16];
  if (threadIdx.x < s) {
    coeff[threadIdx.x] = ytx[threadIdx.x];
  }
  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    float acc = lambda * x[i];
    for (int j = 0; j < s; ++j) {
      acc += __bfloat162float(Y[i + static_cast<size_t>(j) * n]) * coeff[j];
    }
    out[i] = acc;
  }
}

__global__ void apply_small_sketch_basis_kernel(const __nv_bfloat16* __restrict__ Y,
                                                const float* __restrict__ ytu,
                                                const __nv_bfloat16* __restrict__ U,
                                                float lambda,
                                                float* __restrict__ out,
                                                int n,
                                                int s,
                                                int r) {
  __shared__ float coeff[16 * 8];
  if (threadIdx.x < (s * r)) {
    coeff[threadIdx.x] = ytu[threadIdx.x];
  }
  __syncthreads();

  const int basis_col = blockIdx.y;
  if (basis_col >= r) {
    return;
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float* coeff_col = coeff + static_cast<size_t>(basis_col) * s;
  for (int i = idx; i < n; i += stride) {
    float acc = lambda * __bfloat162float(U[i + static_cast<size_t>(basis_col) * n]);
    for (int j = 0; j < s; ++j) {
      acc += __bfloat162float(Y[i + static_cast<size_t>(j) * n]) * coeff_col[j];
    }
    out[i + static_cast<size_t>(basis_col) * n] = acc;
  }
}

__global__ void add_diagonal_kernel(float* matrix, int dim, float value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    matrix[idx * dim + idx] += value;
  }
}

__global__ void woodbury_reconstruct_kernel(const __nv_bfloat16* __restrict__ Y,
                                            const float* __restrict__ coeff,
                                            const float* __restrict__ grad,
                                            float inv_lambda,
                                            float* __restrict__ out,
                                            int n,
                                            int s) {
  __shared__ float shared_coeff[16];
  if (threadIdx.x < s) {
    shared_coeff[threadIdx.x] = coeff[threadIdx.x];
  }
  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    float correction = 0.0f;
    for (int j = 0; j < s; ++j) {
      correction += __bfloat162float(Y[i + static_cast<size_t>(j) * n]) * shared_coeff[j];
    }
    out[i] = (grad[i] - correction) * inv_lambda;
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

__global__ void init_cg_state_kernel(const float* __restrict__ g_norm,
                                     const float* __restrict__ r_norm,
                                     float tol,
                                     int cg_iters,
                                     int* __restrict__ cg_continue,
                                     DeviceStepStats* __restrict__ stats) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const float g_value = *g_norm;
  const float r_value = *r_norm;
  stats->initial_residual_norm = r_value;
  stats->final_residual_norm = r_value;
  stats->cg_steps_taken = 0;
  *cg_continue = (cg_iters > 0 && r_value > (tol * fmaxf(1.0f, g_value))) ? 1 : 0;
}

__global__ void cg_update_solution_residual_kernel(const int* __restrict__ cg_continue,
                                                   const float* __restrict__ rz_old,
                                                   const float* __restrict__ pAp,
                                                   const float* __restrict__ p,
                                                   const float* __restrict__ Ap,
                                                   float* __restrict__ x,
                                                   float* __restrict__ r,
                                                   int n) {
  __shared__ float alpha;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = *cg_continue;
    alpha = 0.0f;
    if (active != 0) {
      const float denom = *pAp;
      if (fabsf(denom) < 1.0e-20f) {
        active = 0;
      } else {
        alpha = (*rz_old) / denom;
      }
    }
  }
  __syncthreads();

  if (active == 0) {
    return;
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
  }
}

__global__ void update_cg_progress_kernel(int* __restrict__ cg_continue,
                                          DeviceStepStats* __restrict__ stats,
                                          const float* __restrict__ r_norm,
                                          const float* __restrict__ g_norm,
                                          float tol,
                                          int completed_steps) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const float r_value = *r_norm;
  stats->final_residual_norm = r_value;
  if (*cg_continue == 0) {
    return;
  }
  stats->cg_steps_taken = completed_steps;
  if (r_value <= (tol * fmaxf(1.0f, *g_norm))) {
    *cg_continue = 0;
  }
}

__global__ void cg_update_direction_kernel(const int* __restrict__ cg_continue,
                                           float* __restrict__ rz_old,
                                           const float* __restrict__ rz_new,
                                           const float* __restrict__ z,
                                           float* __restrict__ p,
                                           int n) {
  __shared__ float beta;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = *cg_continue;
    beta = 0.0f;
    if (active != 0) {
      const float denom = *rz_old;
      if (fabsf(denom) < 1.0e-30f) {
        active = 0;
      } else {
        beta = (*rz_new) / denom;
        *rz_old = *rz_new;
      }
    }
  }
  __syncthreads();

  if (active == 0) {
    return;
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    p[i] = z[i] + (beta * p[i]);
  }
}

struct ExplicitSketchSolver::Impl {
  static constexpr int kFastPathMaxSketch = 16;
  static constexpr int kFastPathMaxRank = 8;
  static constexpr float kHostOne_ = 1.0f;
  static constexpr float kHostZero_ = 0.0f;

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
    pointer_mode_ = CUBLAS_POINTER_MODE_HOST;
    fast_path_enabled_ = (s_ <= kFastPathMaxSketch && r_ <= kFastPathMaxRank);

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

  // Y and U are column-major BF16 device matrices; grad/out/diag are FP32 vectors.
  void step(const __nv_bfloat16* Y,
            const float* grad,
            const __nv_bfloat16* U_in,
            __nv_bfloat16* U_out,
            float lambda,
            int cg_iters,
            float tol,
            float* out_v,
            float* diag_ema = nullptr,
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

    bool use_basis = (r_ > 0 && U_in != nullptr);

    if (try_run_small_exact_step(Y,
                                 grad,
                                 U_in,
                                 U_out,
                                 lambda,
                                 cg_iters,
                                 out_v,
                                 diag_ema,
                                 diag_ema_beta,
                                 basis_replace_col,
                                 basis_append_threshold,
                                 host_stats)) {
      return;
    }

    if (use_basis &&
        try_run_fast_graph_step(Y,
                                grad,
                                U_in,
                                U_out,
                                lambda,
                                cg_iters,
                                tol,
                                out_v,
                                diag_ema,
                                diag_ema_beta,
                                basis_replace_col,
                                basis_append_threshold,
                                host_stats)) {
      return;
    }

    set_pointer_mode(CUBLAS_POINTER_MODE_HOST);

    if (diag_ema != nullptr) {
      update_diag_ema_kernel<<<blocks_n, threads, 0, stream_>>>(Y, diag_ema, n_, s_, diag_ema_beta);
      CHECK_CUDA(cudaGetLastError());
    }

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

    // Precondition the residual before CG.
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
  bool fast_path_enabled_ = false;
  bool graph_disabled_ = false;
  cudaStream_t stream_ = 0;
  cublasHandle_t handle_ = nullptr;
  cublasPointerMode_t pointer_mode_ = CUBLAS_POINTER_MODE_HOST;

  // Dense vector scratch and BF16 staging.
  __nv_bfloat16* d_x_bf16_ = nullptr;
  float* d_tmp_fp32_ = nullptr;
  __nv_bfloat16* d_tmp_bf16_ = nullptr;
  float* d_AU_ = nullptr;
  float* d_Av_ = nullptr;
  float* d_r_ = nullptr;
  float* d_z_ = nullptr;
  float* d_p_ = nullptr;
  float* d_Ap_ = nullptr;
  float* d_candidate_ = nullptr;

  // Reduced-system, scalar, and exact small-sketch scratch.
  float* d_H_ = nullptr;
  float* d_rhs_ = nullptr;
  float* d_coeff_ = nullptr;
  float* d_proj_ = nullptr;
  float* d_one_ = nullptr;
  float* d_zero_ = nullptr;
  float* d_g_norm_ = nullptr;
  float* d_r_norm_ = nullptr;
  float* d_rz_old_ = nullptr;
  float* d_rz_new_ = nullptr;
  float* d_pAp_ = nullptr;
  int* d_cg_continue_ = nullptr;
  DeviceStepStats* d_step_stats_ = nullptr;
  float* d_exact_gram_ = nullptr;
  float* d_exact_rhs_ = nullptr;
  float* d_exact_coeff_ = nullptr;

  // Graph replay inputs and host-visible stats.
  __nv_bfloat16* d_graph_y_ = nullptr;
  float* d_graph_grad_ = nullptr;
  float* d_graph_out_ = nullptr;
  DeviceStepStats* h_step_stats_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaStream_t graph_stream_ = nullptr;
  const __nv_bfloat16* graph_basis_ptr_ = nullptr;
  float* graph_diag_ptr_ = nullptr;
  float graph_lambda_ = 0.0f;
  float graph_tol_ = 0.0f;
  int graph_cg_iters_ = -1;
  float graph_diag_ema_beta_ = 0.0f;

  void set_pointer_mode(cublasPointerMode_t mode) {
    if (pointer_mode_ == mode) {
      return;
    }
    CHECK_CUBLAS(cublasSetPointerMode(handle_, mode));
    pointer_mode_ = mode;
  }

  void invalidate_graph() {
    if (graph_exec_ != nullptr) {
      cudaGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    graph_stream_ = nullptr;
    graph_basis_ptr_ = nullptr;
    graph_diag_ptr_ = nullptr;
    graph_lambda_ = 0.0f;
    graph_tol_ = 0.0f;
    graph_cg_iters_ = -1;
    graph_diag_ema_beta_ = 0.0f;
  }

  [[nodiscard]] bool can_use_fast_graph_path(const __nv_bfloat16* U_in,
                                             __nv_bfloat16* U_out,
                                             float lambda,
                                             int cg_iters,
                                             float tol,
                                             float* diag_ema,
                                             float diag_ema_beta) const {
    return fast_path_enabled_ && !graph_disabled_ && U_in != nullptr && U_out != nullptr &&
           U_in == U_out && cg_iters > 0 && lambda > 0.0f && tol > 0.0f &&
           (diag_ema != nullptr) && diag_ema_beta > 0.0f;
  }

  [[nodiscard]] bool graph_matches(const __nv_bfloat16* basis_ptr,
                                   float* diag_ptr,
                                   float lambda,
                                   int cg_iters,
                                   float tol,
                                   float diag_ema_beta,
                                   cudaStream_t stream) const {
    return graph_exec_ != nullptr && graph_stream_ == stream && graph_basis_ptr_ == basis_ptr &&
           graph_diag_ptr_ == diag_ptr && graph_lambda_ == lambda && graph_tol_ == tol &&
           graph_cg_iters_ == cg_iters && graph_diag_ema_beta_ == diag_ema_beta;
  }

  // Solve the sketched system directly when s is small enough.
  bool try_run_small_exact_step(const __nv_bfloat16* Y,
                                const float* grad,
                                const __nv_bfloat16* U_in,
                                __nv_bfloat16* U_out,
                                float lambda,
                                int cg_iters,
                                float* out_v,
                                float* diag_ema,
                                float diag_ema_beta,
                                int basis_replace_col,
                                float basis_append_threshold,
                                StepStats* host_stats) {
    if (!fast_path_enabled_ || cg_iters <= 0) {
      return false;
    }

    const int threads = 256;
    const int blocks_n = std::max(1, (n_ + threads - 1) / threads);

    if (diag_ema != nullptr) {
      update_diag_ema_kernel<<<blocks_n, threads, 0, stream_>>>(Y, diag_ema, n_, s_, diag_ema_beta);
      CHECK_CUDA(cudaGetLastError());
    }

    dim3 grid_gram(static_cast<unsigned>(s_), static_cast<unsigned>(s_), 1);
    bf16_matT_bf16_mat_kernel<256><<<grid_gram, 256, 0, stream_>>>(Y, Y, d_exact_gram_, n_, s_, s_);
    CHECK_CUDA(cudaGetLastError());

    const int diag_threads = 32;
    const int diag_blocks = std::max(1, (s_ + diag_threads - 1) / diag_threads);
    add_diagonal_kernel<<<diag_blocks, diag_threads, 0, stream_>>>(d_exact_gram_, s_, lambda);
    CHECK_CUDA(cudaGetLastError());

    bf16_matT_vec_kernel<256><<<s_, 256, 0, stream_>>>(Y, grad, d_exact_rhs_, n_, s_);
    CHECK_CUDA(cudaGetLastError());

    cholesky_inplace_kernel<<<1, 1, 0, stream_>>>(d_exact_gram_, s_);
    CHECK_CUDA(cudaGetLastError());
    cholesky_solve_kernel<<<1, 1, 0, stream_>>>(d_exact_gram_, d_exact_rhs_, d_exact_coeff_, s_);
    CHECK_CUDA(cudaGetLastError());

    woodbury_reconstruct_kernel<<<blocks_n, threads, 0, stream_>>>(
        Y, d_exact_coeff_, grad, 1.0f / lambda, out_v, n_, s_);
    CHECK_CUDA(cudaGetLastError());

    bool basis_refreshed = false;
    if (U_out != nullptr) {
      if (U_in != nullptr && U_in != U_out) {
        CHECK_CUDA(cudaMemcpyAsync(U_out,
                                   U_in,
                                   sizeof(__nv_bfloat16) * static_cast<size_t>(n_) * r_,
                                   cudaMemcpyDeviceToDevice,
                                   stream_));
      }
      if (U_in == nullptr) {
        basis_refreshed =
            maybe_refresh_basis(U_in, U_out, out_v, basis_replace_col, basis_append_threshold);
      }
    }

    if (host_stats != nullptr) {
      host_stats->initial_residual_norm = 0.0f;
      host_stats->final_residual_norm = 0.0f;
      host_stats->cg_steps_taken = 0;
      host_stats->basis_refreshed = basis_refreshed ? 1 : 0;
    }
    return true;
  }

  // Replay the recycled-basis + CG path without host round-trips.
  void launch_fast_graph_body(const __nv_bfloat16* Y,
                              const float* grad,
                              const __nv_bfloat16* U,
                              float lambda,
                              int cg_iters,
                              float tol,
                              float* out_v,
                              float* diag_ema,
                              float diag_ema_beta) {
    const int threads = 256;
    const int blocks_n = std::max(1, (n_ + threads - 1) / threads);

    update_diag_ema_kernel<<<blocks_n, threads, 0, stream_>>>(Y, diag_ema, n_, s_, diag_ema_beta);
    CHECK_CUDA(cudaGetLastError());

    apply_A_to_basis(Y, U, lambda, d_AU_, d_tmp_fp32_, d_tmp_bf16_, true);

    dim3 gridH(static_cast<unsigned>(r_), static_cast<unsigned>(r_), 1);
    gram_bf16_fp32_kernel<256><<<gridH, 256, 0, stream_>>>(U, d_AU_, d_H_, n_, r_, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());

    bf16_matT_vec_kernel<256><<<r_, 256, 0, stream_>>>(U, grad, d_rhs_, n_, r_);
    CHECK_CUDA(cudaGetLastError());

    cholesky_inplace_kernel<<<1, 1, 0, stream_>>>(d_H_, r_);
    CHECK_CUDA(cudaGetLastError());

    cholesky_solve_kernel<<<1, 1, 0, stream_>>>(d_H_, d_rhs_, d_coeff_, r_);
    CHECK_CUDA(cudaGetLastError());

    zero_fp32_kernel<<<blocks_n, threads, 0, stream_>>>(out_v, n_);
    CHECK_CUDA(cudaGetLastError());
    basis_lincomb_from_bf16_kernel<<<blocks_n, threads, 0, stream_>>>(
        U, d_coeff_, out_v, n_, r_, 1.0f, 1);
    CHECK_CUDA(cudaGetLastError());

    apply_A_to_fp32_vectors(Y, out_v, 1, lambda, d_Av_, d_x_bf16_, d_tmp_fp32_, d_tmp_bf16_, true);
    residual_kernel<<<blocks_n, threads, 0, stream_>>>(grad, d_Av_, d_r_, n_);
    CHECK_CUDA(cudaGetLastError());

    set_pointer_mode(CUBLAS_POINTER_MODE_DEVICE);
    CHECK_CUBLAS(cublasSnrm2(handle_, n_, grad, 1, d_g_norm_));
    CHECK_CUBLAS(cublasSnrm2(handle_, n_, d_r_, 1, d_r_norm_));
    init_cg_state_kernel<<<1, 1, 0, stream_>>>(d_g_norm_,
                                               d_r_norm_,
                                               tol,
                                               cg_iters,
                                               d_cg_continue_,
                                               d_step_stats_);
    CHECK_CUDA(cudaGetLastError());

    apply_diag_preconditioner_kernel<<<blocks_n, threads, 0, stream_>>>(
        d_r_, diag_ema, lambda, d_z_, n_, 1);
    CHECK_CUDA(cudaGetLastError());
    project_to_basis_complement(U, d_AU_, d_H_, d_z_);
    CHECK_CUDA(cudaMemcpyAsync(d_p_,
                               d_z_,
                               sizeof(float) * static_cast<size_t>(n_),
                               cudaMemcpyDeviceToDevice,
                               stream_));
    CHECK_CUBLAS(cublasSdot(handle_, n_, d_r_, 1, d_z_, 1, d_rz_old_));

    for (int iter = 0; iter < cg_iters; ++iter) {
      apply_A_to_fp32_vectors(
          Y, d_p_, 1, lambda, d_Ap_, d_x_bf16_, d_tmp_fp32_, d_tmp_bf16_, true);
      CHECK_CUBLAS(cublasSdot(handle_, n_, d_p_, 1, d_Ap_, 1, d_pAp_));
      cg_update_solution_residual_kernel<<<blocks_n, threads, 0, stream_>>>(
          d_cg_continue_, d_rz_old_, d_pAp_, d_p_, d_Ap_, out_v, d_r_, n_);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUBLAS(cublasSnrm2(handle_, n_, d_r_, 1, d_r_norm_));
      update_cg_progress_kernel<<<1, 1, 0, stream_>>>(
          d_cg_continue_, d_step_stats_, d_r_norm_, d_g_norm_, tol, iter + 1);
      CHECK_CUDA(cudaGetLastError());

      apply_diag_preconditioner_kernel<<<blocks_n, threads, 0, stream_>>>(
          d_r_, diag_ema, lambda, d_z_, n_, 1);
      CHECK_CUDA(cudaGetLastError());
      project_to_basis_complement(U, d_AU_, d_H_, d_z_);
      CHECK_CUBLAS(cublasSdot(handle_, n_, d_r_, 1, d_z_, 1, d_rz_new_));
      cg_update_direction_kernel<<<blocks_n, threads, 0, stream_>>>(
          d_cg_continue_, d_rz_old_, d_rz_new_, d_z_, d_p_, n_);
      CHECK_CUDA(cudaGetLastError());
    }
  }

  bool build_fast_graph(const __nv_bfloat16* basis_ptr,
                        float* diag_ptr,
                        float lambda,
                        int cg_iters,
                        float tol,
                        float diag_ema_beta) {
    invalidate_graph();

    cudaError_t capture_error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (capture_error != cudaSuccess) {
      return false;
    }

    bool capture_ok = true;
    try {
      launch_fast_graph_body(
          d_graph_y_, d_graph_grad_, basis_ptr, lambda, cg_iters, tol, d_graph_out_, diag_ptr,
          diag_ema_beta);
    } catch (...) {
      capture_ok = false;
    }

    cudaError_t end_error = cudaSuccess;
    if (capture_ok) {
      end_error = cudaStreamEndCapture(stream_, &graph_);
    } else {
      end_error = cudaStreamEndCapture(stream_, &graph_);
    }

    if (!capture_ok || end_error != cudaSuccess || graph_ == nullptr) {
      invalidate_graph();
      return false;
    }

    cudaError_t instantiate_error = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (instantiate_error != cudaSuccess) {
      invalidate_graph();
      return false;
    }

    graph_stream_ = stream_;
    graph_basis_ptr_ = basis_ptr;
    graph_diag_ptr_ = diag_ptr;
    graph_lambda_ = lambda;
    graph_tol_ = tol;
    graph_cg_iters_ = cg_iters;
    graph_diag_ema_beta_ = diag_ema_beta;
    return true;
  }

  bool try_run_fast_graph_step(const __nv_bfloat16* Y,
                               const float* grad,
                               const __nv_bfloat16* U_in,
                               __nv_bfloat16* U_out,
                               float lambda,
                               int cg_iters,
                               float tol,
                               float* out_v,
                               float* diag_ema,
                               float diag_ema_beta,
                               int basis_replace_col,
                               float basis_append_threshold,
                               StepStats* host_stats) {
    if (!can_use_fast_graph_path(
            U_in, U_out, lambda, cg_iters, tol, diag_ema, diag_ema_beta)) {
      return false;
    }

    try {
      CHECK_CUDA(cudaMemcpyAsync(d_graph_y_,
                                 Y,
                                 sizeof(__nv_bfloat16) * static_cast<size_t>(n_) * s_,
                                 cudaMemcpyDeviceToDevice,
                                 stream_));
      CHECK_CUDA(cudaMemcpyAsync(d_graph_grad_,
                                 grad,
                                 sizeof(float) * static_cast<size_t>(n_),
                                 cudaMemcpyDeviceToDevice,
                                 stream_));

      if (!graph_matches(U_in, diag_ema, lambda, cg_iters, tol, diag_ema_beta, stream_)) {
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        if (!build_fast_graph(U_in, diag_ema, lambda, cg_iters, tol, diag_ema_beta)) {
          graph_disabled_ = true;
          set_pointer_mode(CUBLAS_POINTER_MODE_HOST);
          return false;
        }
      }

      set_pointer_mode(CUBLAS_POINTER_MODE_DEVICE);
      CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_));
      const bool basis_refreshed =
          maybe_refresh_basis(U_in, U_out, d_graph_out_, basis_replace_col, basis_append_threshold);
      CHECK_CUDA(cudaMemcpyAsync(out_v,
                                 d_graph_out_,
                                 sizeof(float) * static_cast<size_t>(n_),
                                 cudaMemcpyDeviceToDevice,
                                 stream_));

      if (host_stats != nullptr) {
        CHECK_CUDA(cudaMemcpyAsync(h_step_stats_,
                                   d_step_stats_,
                                   sizeof(DeviceStepStats),
                                   cudaMemcpyDeviceToHost,
                                   stream_));
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        host_stats->initial_residual_norm = h_step_stats_->initial_residual_norm;
        host_stats->final_residual_norm = h_step_stats_->final_residual_norm;
        host_stats->cg_steps_taken = h_step_stats_->cg_steps_taken;
        host_stats->basis_refreshed = basis_refreshed ? 1 : 0;
      }
      return true;
    } catch (...) {
      invalidate_graph();
      graph_disabled_ = true;
      set_pointer_mode(CUBLAS_POINTER_MODE_HOST);
      return false;
    }
  }

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
    CHECK_CUDA(cudaMalloc(&d_one_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_zero_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g_norm_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r_norm_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rz_old_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rz_new_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pAp_, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cg_continue_, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_step_stats_, sizeof(DeviceStepStats)));
    if (fast_path_enabled_) {
      CHECK_CUDA(cudaMalloc(&d_exact_gram_, sizeof(float) * static_cast<size_t>(s_) * s_));
      CHECK_CUDA(cudaMalloc(&d_exact_rhs_, sizeof(float) * static_cast<size_t>(s_)));
      CHECK_CUDA(cudaMalloc(&d_exact_coeff_, sizeof(float) * static_cast<size_t>(s_)));
    }
    CHECK_CUDA(cudaMemcpy(d_one_, &kHostOne_, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zero_, &kHostZero_, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMallocHost(&h_step_stats_, sizeof(DeviceStepStats)));
    if (fast_path_enabled_) {
      CHECK_CUDA(cudaMalloc(&d_graph_y_, sizeof(__nv_bfloat16) * static_cast<size_t>(n_) * s_));
      CHECK_CUDA(cudaMalloc(&d_graph_grad_, sizeof(float) * n1));
      CHECK_CUDA(cudaMalloc(&d_graph_out_, sizeof(float) * n1));
    }
  }

  void release() {
    auto free_ptr = [](void* p) {
      if (p) cudaFree(p);
    };
    invalidate_graph();
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
    free_ptr(d_one_);
    free_ptr(d_zero_);
    free_ptr(d_g_norm_);
    free_ptr(d_r_norm_);
    free_ptr(d_rz_old_);
    free_ptr(d_rz_new_);
    free_ptr(d_pAp_);
    free_ptr(d_cg_continue_);
    free_ptr(d_step_stats_);
    free_ptr(d_exact_gram_);
    free_ptr(d_exact_rhs_);
    free_ptr(d_exact_coeff_);
    free_ptr(d_graph_y_);
    free_ptr(d_graph_grad_);
    free_ptr(d_graph_out_);
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
    d_one_ = nullptr;
    d_zero_ = nullptr;
    d_g_norm_ = nullptr;
    d_r_norm_ = nullptr;
    d_rz_old_ = nullptr;
    d_rz_new_ = nullptr;
    d_pAp_ = nullptr;
    d_cg_continue_ = nullptr;
    d_step_stats_ = nullptr;
    d_exact_gram_ = nullptr;
    d_exact_rhs_ = nullptr;
    d_exact_coeff_ = nullptr;
    d_graph_y_ = nullptr;
    d_graph_grad_ = nullptr;
    d_graph_out_ = nullptr;
    if (h_step_stats_ != nullptr) {
      cudaFreeHost(h_step_stats_);
      h_step_stats_ = nullptr;
    }
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
                        __nv_bfloat16* tmp_bf16,
                        bool use_device_scalars = false) {
    if (r_ <= 0) return;
    if (fast_path_enabled_) {
      // Small sketches are faster with custom dot-product kernels than cuBLAS.
      dim3 grid_ytu(static_cast<unsigned>(s_), static_cast<unsigned>(r_), 1);
      bf16_matT_bf16_mat_kernel<256><<<grid_ytu, 256, 0, stream_>>>(
          Y, X_basis, tmp_fp32, n_, s_, r_);
      CHECK_CUDA(cudaGetLastError());

      const int threads = 256;
      const int blocks = std::max(1, (n_ + threads - 1) / threads);
      dim3 grid_apply(static_cast<unsigned>(blocks), static_cast<unsigned>(r_), 1);
      apply_small_sketch_basis_kernel<<<grid_apply, threads, 0, stream_>>>(
          Y, tmp_fp32, X_basis, lambda, out_fp32, n_, s_, r_);
      CHECK_CUDA(cudaGetLastError());
      return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* alpha_ptr = use_device_scalars ? static_cast<const void*>(d_one_)
                                               : static_cast<const void*>(&alpha);
    const void* beta_ptr = use_device_scalars ? static_cast<const void*>(d_zero_)
                                              : static_cast<const void*>(&beta);
    set_pointer_mode(use_device_scalars ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              s_,
                              r_,
                              n_,
                              alpha_ptr,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              X_basis,
                              CUDA_R_16BF,
                              n_,
                              beta_ptr,
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
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n_,
                              r_,
                              s_,
                              alpha_ptr,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              tmp_bf16,
                              CUDA_R_16BF,
                              s_,
                              beta_ptr,
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
                               __nv_bfloat16* tmp_bf16,
                               bool use_device_scalars = false) {
    if (fast_path_enabled_ && k == 1) {
      // The rank-1 path is small enough that custom kernels beat cuBLAS setup cost.
      bf16_matT_vec_kernel<256><<<s_, 256, 0, stream_>>>(Y, X_fp32, tmp_fp32, n_, s_);
      CHECK_CUDA(cudaGetLastError());
      const int threads = 256;
      const int blocks = std::max(1, (n_ + threads - 1) / threads);
      apply_small_sketch_vector_kernel<<<blocks, threads, 0, stream_>>>(
          Y, tmp_fp32, X_fp32, lambda, out_fp32, n_, s_);
      CHECK_CUDA(cudaGetLastError());
      return;
    }

    const int threads = 256;
    int count_nk = n_ * k;
    int blocks = std::max(1, (count_nk + threads - 1) / threads);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream_>>>(X_fp32, x_bf16, count_nk);
    CHECK_CUDA(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* alpha_ptr = use_device_scalars ? static_cast<const void*>(d_one_)
                                               : static_cast<const void*>(&alpha);
    const void* beta_ptr = use_device_scalars ? static_cast<const void*>(d_zero_)
                                              : static_cast<const void*>(&beta);
    set_pointer_mode(use_device_scalars ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              s_,
                              k,
                              n_,
                              alpha_ptr,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              x_bf16,
                              CUDA_R_16BF,
                              n_,
                              beta_ptr,
                              tmp_fp32,
                              CUDA_R_32F,
                              s_,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int count_sk = s_ * k;
    blocks = std::max(1, (count_sk + threads - 1) / threads);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream_>>>(tmp_fp32, tmp_bf16, count_sk);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUBLAS(cublasGemmEx(handle_,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n_,
                              k,
                              s_,
                              alpha_ptr,
                              Y,
                              CUDA_R_16BF,
                              n_,
                              tmp_bf16,
                              CUDA_R_16BF,
                              s_,
                              beta_ptr,
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
    // Replace one basis column with the normalized unexplained part of v.
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
    set_pointer_mode(CUBLAS_POINTER_MODE_HOST);
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
  size_t bytes = (sizeof(__nv_bfloat16) * nr) + (sizeof(float) * sr) +
                 (sizeof(__nv_bfloat16) * sr) + (sizeof(float) * nr) +
                 (sizeof(float) * n1 * 6) + (sizeof(float) * rr) +
                 (sizeof(float) * std::max(1, r_) * 3) + (sizeof(float) * 7) +
                 sizeof(int) + sizeof(DeviceStepStats);
  if (fast_path_enabled_) {
    bytes += (sizeof(float) * static_cast<size_t>(s_) * (s_ + 2)) +
             (sizeof(__nv_bfloat16) * static_cast<size_t>(n_) * s_) + (sizeof(float) * n1 * 2);
  }
  return bytes;
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
