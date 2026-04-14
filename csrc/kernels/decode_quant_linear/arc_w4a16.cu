#include "ops/decode_quant_linear/arc_w4a16.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace fast_kernels::decode_quant_linear {
namespace {

constexpr int kColumnsPerTile = 128;
constexpr int kThreadCount = 128;
static_assert(kColumnsPerTile == kThreadCount, "ARC uses one thread per output column");

int expected_packet_stride_bytes(int group_size) {
  return (kColumnsPerTile * (group_size / 2)) + (2 * kColumnsPerTile * static_cast<int>(sizeof(__half)));
}

void validate_dimensions(int batch, int n, int k, int group_size) {
  if (batch <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("batch, n, and k must be positive");
  }
  if (group_size != 64 && group_size != 128) {
    throw std::invalid_argument("group_size must be 64 or 128");
  }
  if (n % kColumnsPerTile != 0) {
    throw std::invalid_argument("n must be a multiple of 128");
  }
  if (k % group_size != 0) {
    throw std::invalid_argument("k must be divisible by group_size");
  }
}

void validate_packet_stride(int group_size, int packet_stride_bytes) {
  const int expected = expected_packet_stride_bytes(group_size);
  if (packet_stride_bytes != expected) {
    std::ostringstream oss;
    oss << "packet_stride_bytes=" << packet_stride_bytes << " does not match expected " << expected;
    throw std::invalid_argument(oss.str());
  }
}

void throw_cuda_error(cudaError_t error, const char* expression) {
  std::ostringstream oss;
  oss << expression << " failed: " << cudaGetErrorString(error);
  throw std::runtime_error(oss.str());
}

void throw_cublas_error(cublasStatus_t status, const char* expression) {
  std::ostringstream oss;
  oss << expression << " failed: " << cublasGetStatusString(status);
  throw std::runtime_error(oss.str());
}

#define FK_CUDA_CHECK(expr)                        \
  do {                                             \
    const cudaError_t status__ = (expr);           \
    if (status__ != cudaSuccess) {                 \
      throw_cuda_error(status__, #expr);           \
    }                                              \
  } while (false)

#define FK_CUBLAS_CHECK(expr)                      \
  do {                                             \
    const cublasStatus_t status__ = (expr);        \
    if (status__ != CUBLAS_STATUS_SUCCESS) {       \
      throw_cublas_error(status__, #expr);         \
    }                                              \
  } while (false)

template <int GROUP_SIZE>
__global__ void pack_arc_packets_kernel(
    const std::uint8_t* __restrict__ q,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    std::uint8_t* __restrict__ packets,
    int k,
    int num_groups,
    int packet_stride_bytes
) {
  constexpr int kGroupBytes = GROUP_SIZE / 2;
  constexpr int kQTileBytes = kColumnsPerTile * kGroupBytes;

  const int n_tile = static_cast<int>(blockIdx.x);
  const int group_idx = static_cast<int>(blockIdx.y);
  const int row_start = n_tile * kColumnsPerTile;
  const int q_row_stride = k / 2;
  std::uint8_t* packet = packets + ((n_tile * num_groups + group_idx) * packet_stride_bytes);

  for (int idx = static_cast<int>(threadIdx.x); idx < kQTileBytes; idx += static_cast<int>(blockDim.x)) {
    const int row = idx / kGroupBytes;
    const int byte_in_group = idx % kGroupBytes;
    packet[idx] = q[((row_start + row) * q_row_stride) + (group_idx * kGroupBytes) + byte_in_group];
  }

  // Each packet is laid out as:
  //   [ 128 rows of packed q bytes ][ 128 alpha values ][ 128 beta values ]
  // in the exact order the compute CTA consumes them.
  __half* alpha_dst = reinterpret_cast<__half*>(packet + kQTileBytes);
  __half* beta_dst = reinterpret_cast<__half*>(packet + kQTileBytes + (kColumnsPerTile * static_cast<int>(sizeof(__half))));
  for (int idx = static_cast<int>(threadIdx.x); idx < kColumnsPerTile; idx += static_cast<int>(blockDim.x)) {
    alpha_dst[idx] = alpha[((row_start + idx) * num_groups) + group_idx];
    beta_dst[idx] = beta[((row_start + idx) * num_groups) + group_idx];
  }
}

template <int GROUP_SIZE>
__global__ void dequant_w4a16_kernel(
    const std::uint8_t* __restrict__ q,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    __half* __restrict__ output,
    int n,
    int k,
    int num_groups
) {
  const int idx = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)) + static_cast<int>(threadIdx.x);
  const int total = n * k;
  if (idx >= total) {
    return;
  }

  const int row = idx / k;
  const int col = idx % k;
  const int group_idx = col / GROUP_SIZE;
  const int q_pair = col / 2;
  const std::uint8_t packed = q[(row * (k / 2)) + q_pair];
  const int q_value = (col & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
  const float alpha_value = __half2float(alpha[(row * num_groups) + group_idx]);
  const float beta_value = __half2float(beta[(row * num_groups) + group_idx]);
  output[idx] = __float2half_rn((alpha_value * static_cast<float>(q_value)) + beta_value);
}

template <int GROUP_SIZE, int M_TILE>
__device__ void accumulate_arc_group_range(
    const __half* __restrict__ activations,
    const std::uint8_t* __restrict__ packets,
    int row_start,
    int active_rows,
    int k,
    int group_begin,
    int group_end,
    int num_groups,
    int packet_stride_bytes,
    __half2 shared_activation_pairs[M_TILE][GROUP_SIZE / 2],
    float shared_sums[M_TILE],
    float2 q_pair_lut[256],
    float accumulators[M_TILE]
) {
  constexpr int kGroupBytes = GROUP_SIZE / 2;
  constexpr int kQTileBytes = kColumnsPerTile * kGroupBytes;
  constexpr int kPackedWords = kGroupBytes / 4;

  for (int group_idx = group_begin; group_idx < group_end; ++group_idx) {
    // Stage one activation group as half2 pairs so the inner loop consumes
    // packed q bytes against vectorized activation loads.
    for (int linear = static_cast<int>(threadIdx.x); linear < active_rows * kGroupBytes;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kGroupBytes;
      const int pair_idx = linear % kGroupBytes;
      const int batch_row = row_start + row;
      const __half2* activation_pairs =
          reinterpret_cast<const __half2*>(activations + (batch_row * k) + (group_idx * GROUP_SIZE));
      shared_activation_pairs[row][pair_idx] = activation_pairs[pair_idx];
    }
    __syncthreads();

    if (static_cast<int>(threadIdx.x) < active_rows) {
      float sum = 0.0f;
      for (int pair_idx = 0; pair_idx < kGroupBytes; ++pair_idx) {
        const float2 a_pair = __half22float2(shared_activation_pairs[threadIdx.x][pair_idx]);
        sum += a_pair.x + a_pair.y;
      }
      shared_sums[threadIdx.x] = sum;
    }
    __syncthreads();

    const std::uint8_t* packet = packets + (((static_cast<int>(blockIdx.x) * num_groups) + group_idx) * packet_stride_bytes);
    const std::uint8_t* q_tile = packet;
    const __half* alpha = reinterpret_cast<const __half*>(packet + kQTileBytes);
    const __half* beta =
        reinterpret_cast<const __half*>(packet + kQTileBytes + (kColumnsPerTile * static_cast<int>(sizeof(__half))));
    const float alpha_value = __half2float(alpha[threadIdx.x]);
    const float beta_value = __half2float(beta[threadIdx.x]);
    const std::uint8_t* q_row = q_tile + (threadIdx.x * kGroupBytes);
    const std::uint32_t* q_row_words = reinterpret_cast<const std::uint32_t*>(q_row);

#pragma unroll
    for (int row = 0; row < M_TILE; ++row) {
      if (row >= active_rows) {
        continue;
      }

      // ARC's hot path is a q-dot-product. The affine offset is applied separately
      // as beta * sum(a_group), which is the exact groupwise rank-1 correction.
      float partial = 0.0f;
#pragma unroll
      for (int word_idx = 0; word_idx < kPackedWords; ++word_idx) {
        const std::uint32_t packed_word = q_row_words[word_idx];

#pragma unroll
        for (int byte_offset = 0; byte_offset < 4; ++byte_offset) {
          const int packed_idx = (word_idx * 4) + byte_offset;
          const std::uint8_t packed = static_cast<std::uint8_t>((packed_word >> (byte_offset * 8)) & 0xffU);
          const float2 q_pair = q_pair_lut[packed];
          const float2 a_pair = __half22float2(shared_activation_pairs[row][packed_idx]);
          partial = fmaf(q_pair.x, a_pair.x, partial);
          partial = fmaf(q_pair.y, a_pair.y, partial);
        }
      }

      accumulators[row] += (alpha_value * partial) + (beta_value * shared_sums[row]);
    }
    __syncthreads();
  }
}

template <int GROUP_SIZE, int M_TILE>
__global__ void arc_w4a16_kernel(
    const __half* __restrict__ activations,
    const std::uint8_t* __restrict__ packets,
    __half* __restrict__ output,
    int batch,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes
) {
  __shared__ __half2 shared_activation_pairs[M_TILE][GROUP_SIZE / 2];
  __shared__ float shared_sums[M_TILE];
  __shared__ float2 q_pair_lut[256];

  // Each packed byte holds two 4-bit q values. The LUT turns that byte into the
  // corresponding (low_nibble, high_nibble) pair once per CTA instead of once per dot.
  for (int idx = static_cast<int>(threadIdx.x); idx < 256; idx += static_cast<int>(blockDim.x)) {
    q_pair_lut[idx] = make_float2(
        static_cast<float>(idx & 0x0f),
        static_cast<float>((idx >> 4) & 0x0f));
  }
  __syncthreads();

  const int row_start = static_cast<int>(blockIdx.y) * M_TILE;
  if (row_start >= batch) {
    return;
  }
  const int rows_remaining = batch - row_start;
  const int active_rows = rows_remaining < M_TILE ? rows_remaining : M_TILE;
  const int col = (static_cast<int>(blockIdx.x) * kColumnsPerTile) + static_cast<int>(threadIdx.x);
  if (col >= n) {
    return;
  }

  float accumulators[M_TILE];
#pragma unroll
  for (int row = 0; row < M_TILE; ++row) {
    accumulators[row] = 0.0f;
  }

  accumulate_arc_group_range<GROUP_SIZE, M_TILE>(
      activations,
      packets,
      row_start,
      active_rows,
      k,
      0,
      num_groups,
      num_groups,
      packet_stride_bytes,
      shared_activation_pairs,
      shared_sums,
      q_pair_lut,
      accumulators);

#pragma unroll
  for (int row = 0; row < M_TILE; ++row) {
    if (row >= active_rows) {
      continue;
    }
    output[((row_start + row) * n) + col] = __float2half_rn(accumulators[row]);
  }
}

template <int GROUP_SIZE, int M_TILE>
__global__ void arc_w4a16_split_k_kernel(
    const __half* __restrict__ activations,
    const std::uint8_t* __restrict__ packets,
    float* __restrict__ partials,
    int batch,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes,
    int groups_per_slice
) {
  __shared__ __half2 shared_activation_pairs[M_TILE][GROUP_SIZE / 2];
  __shared__ float shared_sums[M_TILE];
  __shared__ float2 q_pair_lut[256];

  // Same byte-to-q-pair decode used by the direct kernel. Split-K changes the
  // group range each CTA owns, but it does not change the packet math inside a group.
  for (int idx = static_cast<int>(threadIdx.x); idx < 256; idx += static_cast<int>(blockDim.x)) {
    q_pair_lut[idx] = make_float2(
        static_cast<float>(idx & 0x0f),
        static_cast<float>((idx >> 4) & 0x0f));
  }
  __syncthreads();

  const int slice_idx = static_cast<int>(blockIdx.z);
  const int group_begin = slice_idx * groups_per_slice;
  const int group_end = (group_begin + groups_per_slice) < num_groups ? (group_begin + groups_per_slice) : num_groups;
  if (group_begin >= group_end) {
    return;
  }

  const int row_start = static_cast<int>(blockIdx.y) * M_TILE;
  if (row_start >= batch) {
    return;
  }
  const int rows_remaining = batch - row_start;
  const int active_rows = rows_remaining < M_TILE ? rows_remaining : M_TILE;
  const int col = (static_cast<int>(blockIdx.x) * kColumnsPerTile) + static_cast<int>(threadIdx.x);
  if (col >= n) {
    return;
  }

  float accumulators[M_TILE];
#pragma unroll
  for (int row = 0; row < M_TILE; ++row) {
    accumulators[row] = 0.0f;
  }

  // Split-K is only used for skinny-N decode cases where the direct column grid
  // under-fills the GPU. Extra K slices create enough independent CTAs to hide latency.
  accumulate_arc_group_range<GROUP_SIZE, M_TILE>(
      activations,
      packets,
      row_start,
      active_rows,
      k,
      group_begin,
      group_end,
      num_groups,
      packet_stride_bytes,
      shared_activation_pairs,
      shared_sums,
      q_pair_lut,
      accumulators);

#pragma unroll
  for (int row = 0; row < M_TILE; ++row) {
    if (row >= active_rows) {
      continue;
    }
    partials[((slice_idx * batch + (row_start + row)) * n) + col] = accumulators[row];
  }
}

__global__ void reduce_arc_w4a16_split_k_partials_kernel(
    const float* __restrict__ partials,
    __half* __restrict__ output,
    int total,
    int batch,
    int n,
    int split_k_slices
) {
  const int idx = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)) + static_cast<int>(threadIdx.x);
  if (idx >= total) {
    return;
  }

  float sum = 0.0f;
  for (int slice_idx = 0; slice_idx < split_k_slices; ++slice_idx) {
    sum += partials[(slice_idx * batch * n) + idx];
  }
  output[idx] = __float2half_rn(sum);
}

template <int GROUP_SIZE>
void launch_packet_builder(
    const std::uint8_t* q,
    const __half* alpha,
    const __half* beta,
    std::uint8_t* packets,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes,
    cudaStream_t stream
) {
  const dim3 grid(static_cast<unsigned int>(n / kColumnsPerTile), static_cast<unsigned int>(num_groups), 1U);
  pack_arc_packets_kernel<GROUP_SIZE><<<grid, kThreadCount, 0, stream>>>(
      q, alpha, beta, packets, k, num_groups, packet_stride_bytes);
  FK_CUDA_CHECK(cudaGetLastError());
}

template <int GROUP_SIZE>
void launch_dequant_kernel(
    const std::uint8_t* q,
    const __half* alpha,
    const __half* beta,
    __half* output,
    int n,
    int k,
    int num_groups,
    cudaStream_t stream
) {
  constexpr int kDequantThreads = 256;
  const int total = n * k;
  const int blocks = (total + kDequantThreads - 1) / kDequantThreads;
  dequant_w4a16_kernel<GROUP_SIZE><<<blocks, kDequantThreads, 0, stream>>>(q, alpha, beta, output, n, k, num_groups);
  FK_CUDA_CHECK(cudaGetLastError());
}

template <int GROUP_SIZE, int M_TILE>
void launch_arc_kernel(
    const __half* activations,
    const std::uint8_t* packets,
    __half* output,
    int batch,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes,
    cudaStream_t stream
) {
  const dim3 grid(
      static_cast<unsigned int>(n / kColumnsPerTile),
      static_cast<unsigned int>((batch + M_TILE - 1) / M_TILE),
      1U);
  arc_w4a16_kernel<GROUP_SIZE, M_TILE><<<grid, kThreadCount, 0, stream>>>(
      activations, packets, output, batch, n, k, num_groups, packet_stride_bytes);
  FK_CUDA_CHECK(cudaGetLastError());
}

template <int GROUP_SIZE>
void launch_arc_split_k_kernel(
    const __half* activations,
    const std::uint8_t* packets,
    float* partials,
    int batch,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes,
    int split_k_slices,
    cudaStream_t stream
) {
  const int groups_per_slice = (num_groups + split_k_slices - 1) / split_k_slices;
  if (batch <= 1) {
    const dim3 grid(
        static_cast<unsigned int>(n / kColumnsPerTile),
        static_cast<unsigned int>(batch),
        static_cast<unsigned int>(split_k_slices));
    arc_w4a16_split_k_kernel<GROUP_SIZE, 1><<<grid, kThreadCount, 0, stream>>>(
        activations, packets, partials, batch, n, k, num_groups, packet_stride_bytes, groups_per_slice);
    FK_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (batch <= 2) {
    const dim3 grid(
        static_cast<unsigned int>(n / kColumnsPerTile),
        static_cast<unsigned int>((batch + 1) / 2),
        static_cast<unsigned int>(split_k_slices));
    arc_w4a16_split_k_kernel<GROUP_SIZE, 2><<<grid, kThreadCount, 0, stream>>>(
        activations, packets, partials, batch, n, k, num_groups, packet_stride_bytes, groups_per_slice);
    FK_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (batch <= 4) {
    const dim3 grid(
        static_cast<unsigned int>(n / kColumnsPerTile),
        static_cast<unsigned int>((batch + 3) / 4),
        static_cast<unsigned int>(split_k_slices));
    arc_w4a16_split_k_kernel<GROUP_SIZE, 4><<<grid, kThreadCount, 0, stream>>>(
        activations, packets, partials, batch, n, k, num_groups, packet_stride_bytes, groups_per_slice);
    FK_CUDA_CHECK(cudaGetLastError());
    return;
  }

  const dim3 grid(
      static_cast<unsigned int>(n / kColumnsPerTile),
      static_cast<unsigned int>((batch + 7) / 8),
      static_cast<unsigned int>(split_k_slices));
  arc_w4a16_split_k_kernel<GROUP_SIZE, 8><<<grid, kThreadCount, 0, stream>>>(
      activations, packets, partials, batch, n, k, num_groups, packet_stride_bytes, groups_per_slice);
  FK_CUDA_CHECK(cudaGetLastError());
}

void launch_arc_split_k_reduction(
    const float* partials,
    __half* output,
    int batch,
    int n,
    int split_k_slices,
    cudaStream_t stream
) {
  constexpr int kReductionThreads = 256;
  const int total = batch * n;
  const int blocks = (total + kReductionThreads - 1) / kReductionThreads;
  reduce_arc_w4a16_split_k_partials_kernel<<<blocks, kReductionThreads, 0, stream>>>(
      partials, output, total, batch, n, split_k_slices);
  FK_CUDA_CHECK(cudaGetLastError());
}

template <int GROUP_SIZE>
void dispatch_arc_kernel(
    const __half* activations,
    const std::uint8_t* packets,
    __half* output,
    int batch,
    int n,
    int k,
    int num_groups,
    int packet_stride_bytes,
    cudaStream_t stream
) {
  if (batch <= 1) {
    launch_arc_kernel<GROUP_SIZE, 1>(
        activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
    return;
  }
  if (batch <= 2) {
    launch_arc_kernel<GROUP_SIZE, 2>(
        activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
    return;
  }
  if (batch <= 4) {
    launch_arc_kernel<GROUP_SIZE, 4>(
        activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
    return;
  }
  launch_arc_kernel<GROUP_SIZE, 8>(
      activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
}

cublasLtHandle_t get_cublaslt_handle() {
  static std::once_flag once;
  static cublasLtHandle_t handle = nullptr;
  std::call_once(once, [] { FK_CUBLAS_CHECK(cublasLtCreate(&handle)); });
  return handle;
}

void run_cublaslt_matmul(
    const __half* activations,
    const __half* weights,
    __half* output,
    void* workspace,
    std::size_t workspace_bytes,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;

  FK_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FK_CUBLAS_CHECK(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  FK_CUBLAS_CHECK(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  FK_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, batch, k, k));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, n, k, k));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, batch, n, n));

  FK_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  FK_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  FK_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  FK_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      get_cublaslt_handle(),
      operation_desc,
      a_desc,
      b_desc,
      c_desc,
      c_desc,
      preference,
      1,
      &heuristic,
      &returned_results));

  if (returned_results == 0 && workspace_bytes != 0) {
    constexpr std::size_t kZeroWorkspace = 0;
    FK_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kZeroWorkspace,
        sizeof(kZeroWorkspace)));
    FK_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        get_cublaslt_handle(),
        operation_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic,
        &returned_results));
    workspace = nullptr;
    workspace_bytes = 0;
  }

  if (returned_results == 0) {
    throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no usable algorithm");
  }

  FK_CUBLAS_CHECK(cublasLtMatmul(
      get_cublaslt_handle(),
      operation_desc,
      &alpha,
      activations,
      a_desc,
      weights,
      b_desc,
      &beta,
      output,
      c_desc,
      output,
      c_desc,
      &heuristic.algo,
      workspace,
      workspace_bytes,
      stream));

  FK_CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  FK_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  FK_CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));
}

}  // namespace

void pack_arc_w4a16_packets(
    std::uintptr_t q_ptr,
    std::uintptr_t alpha_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t packets_ptr,
    int n,
    int k,
    int group_size,
    int packet_stride_bytes,
    std::uintptr_t stream_ptr
) {
  validate_dimensions(1, n, k, group_size);
  validate_packet_stride(group_size, packet_stride_bytes);

  const int num_groups = k / group_size;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto* q = reinterpret_cast<const std::uint8_t*>(q_ptr);
  auto* alpha = reinterpret_cast<const __half*>(alpha_ptr);
  auto* beta = reinterpret_cast<const __half*>(beta_ptr);
  auto* packets = reinterpret_cast<std::uint8_t*>(packets_ptr);

  if (group_size == 64) {
    launch_packet_builder<64>(q, alpha, beta, packets, n, k, num_groups, packet_stride_bytes, stream);
    return;
  }
  launch_packet_builder<128>(q, alpha, beta, packets, n, k, num_groups, packet_stride_bytes, stream);
}

void arc_w4a16_forward(
    std::uintptr_t activations_ptr,
    std::uintptr_t packets_ptr,
    std::uintptr_t output_ptr,
    int batch,
    int n,
    int k,
    int group_size,
    int packet_stride_bytes,
    std::uintptr_t stream_ptr
) {
  validate_dimensions(batch, n, k, group_size);
  validate_packet_stride(group_size, packet_stride_bytes);

  const int num_groups = k / group_size;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto* activations = reinterpret_cast<const __half*>(activations_ptr);
  auto* packets = reinterpret_cast<const std::uint8_t*>(packets_ptr);
  auto* output = reinterpret_cast<__half*>(output_ptr);

  if (group_size == 64) {
    dispatch_arc_kernel<64>(
        activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
    return;
  }
  dispatch_arc_kernel<128>(
      activations, packets, output, batch, n, k, num_groups, packet_stride_bytes, stream);
}

void arc_w4a16_forward_split_k(
    std::uintptr_t activations_ptr,
    std::uintptr_t packets_ptr,
    std::uintptr_t partials_ptr,
    int batch,
    int n,
    int k,
    int group_size,
    int packet_stride_bytes,
    int split_k_slices,
    std::uintptr_t stream_ptr
) {
  validate_dimensions(batch, n, k, group_size);
  validate_packet_stride(group_size, packet_stride_bytes);
  if (split_k_slices <= 1) {
    throw std::invalid_argument("split_k_slices must be greater than 1");
  }

  const int num_groups = k / group_size;
  if (split_k_slices > num_groups) {
    throw std::invalid_argument("split_k_slices must not exceed the number of groups");
  }

  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto* activations = reinterpret_cast<const __half*>(activations_ptr);
  auto* packets = reinterpret_cast<const std::uint8_t*>(packets_ptr);
  auto* partials = reinterpret_cast<float*>(partials_ptr);

  if (group_size == 64) {
    launch_arc_split_k_kernel<64>(
        activations,
        packets,
        partials,
        batch,
        n,
        k,
        num_groups,
        packet_stride_bytes,
        split_k_slices,
        stream);
    return;
  }
  launch_arc_split_k_kernel<128>(
      activations,
      packets,
      partials,
      batch,
      n,
      k,
      num_groups,
      packet_stride_bytes,
      split_k_slices,
      stream);
}

void reduce_arc_w4a16_split_k_partials(
    std::uintptr_t partials_ptr,
    std::uintptr_t output_ptr,
    int batch,
    int n,
    int split_k_slices,
    std::uintptr_t stream_ptr
) {
  if (batch <= 0 || n <= 0 || split_k_slices <= 1) {
    throw std::invalid_argument("batch, n, and split_k_slices must be positive with split_k_slices > 1");
  }

  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_arc_split_k_reduction(
      reinterpret_cast<const float*>(partials_ptr),
      reinterpret_cast<__half*>(output_ptr),
      batch,
      n,
      split_k_slices,
      stream);
}

void dequant_w4a16_to_fp16(
    std::uintptr_t q_ptr,
    std::uintptr_t alpha_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t output_ptr,
    int n,
    int k,
    int group_size,
    std::uintptr_t stream_ptr
) {
  validate_dimensions(1, n, k, group_size);

  const int num_groups = k / group_size;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto* q = reinterpret_cast<const std::uint8_t*>(q_ptr);
  auto* alpha = reinterpret_cast<const __half*>(alpha_ptr);
  auto* beta = reinterpret_cast<const __half*>(beta_ptr);
  auto* output = reinterpret_cast<__half*>(output_ptr);

  if (group_size == 64) {
    launch_dequant_kernel<64>(q, alpha, beta, output, n, k, num_groups, stream);
    return;
  }
  launch_dequant_kernel<128>(q, alpha, beta, output, n, k, num_groups, stream);
}

void cublaslt_fp16_after_dequant(
    std::uintptr_t activations_ptr,
    std::uintptr_t q_ptr,
    std::uintptr_t alpha_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t output_ptr,
    std::uintptr_t weight_ptr,
    std::uintptr_t workspace_ptr,
    std::size_t workspace_bytes,
    int batch,
    int n,
    int k,
    int group_size,
    std::uintptr_t stream_ptr
) {
  validate_dimensions(batch, n, k, group_size);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dequant_w4a16_to_fp16(q_ptr, alpha_ptr, beta_ptr, weight_ptr, n, k, group_size, stream_ptr);
  run_cublaslt_matmul(
      reinterpret_cast<const __half*>(activations_ptr),
      reinterpret_cast<const __half*>(weight_ptr),
      reinterpret_cast<__half*>(output_ptr),
      reinterpret_cast<void*>(workspace_ptr),
      workspace_bytes,
      batch,
      n,
      k,
      stream);
}

}  // namespace fast_kernels::decode_quant_linear
