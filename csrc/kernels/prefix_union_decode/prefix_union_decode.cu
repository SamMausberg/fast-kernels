#include "ops/prefix_union_decode/prefix_union_decode.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace fast_kernels::prefix_union_decode {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxClusterSize = 8;

#define FK_CUDA_CHECK(expr)                                                                        \
  do {                                                                                             \
    const cudaError_t error__ = (expr);                                                            \
    if (error__ != cudaSuccess) {                                                                  \
      throw std::runtime_error(cudaGetErrorString(error__));                                       \
    }                                                                                              \
  } while (false)

enum class KvFormat : int32_t {
  kBFloat16 = 0,
  kFp8E4M3 = 1,
  kInt8 = 2,
};

struct alignas(8) SharedPageRef {
  int32_t page_id;
  int16_t valid_tokens;
  int16_t reserved;
};

struct alignas(8) TailPageRef {
  int32_t page_id;
  int16_t valid_tokens;
  int16_t logical_page_index;
};

struct alignas(32) ConsumerRef {
  int32_t query_index;
  int32_t output_index;
  int32_t tail_page_offset;
  int32_t query_position;
  int16_t num_tail_pages;
  int16_t reserved0;
  int32_t reserved1;
};

struct alignas(32) PrefixUnionTask {
  int32_t shared_page_offset;
  int32_t num_shared_pages;
  int32_t consumer_offset;
  int32_t num_consumers;
  int32_t kv_head_index;
  int32_t reserved0;
  int64_t reserved1;
};

__device__ inline float bf16_to_float(const __nv_bfloat16 value) { return __bfloat162float(value); }

__device__ inline __nv_bfloat16 float_to_bf16(const float value) {
  return __float2bfloat16_rn(value);
}

__device__ inline float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return __shfl_sync(0xFFFFFFFFu, value, 0);
}

__device__ inline float rope_inv_freq(const int pair_index, const int head_dim,
                                      const float rope_theta) {
  return powf(rope_theta, -2.0f * static_cast<float>(pair_index) / static_cast<float>(head_dim));
}

template <typename T>
__device__ inline float storage_to_float(const T value);

template <>
__device__ inline float storage_to_float<__nv_bfloat16>(const __nv_bfloat16 value) {
  return bf16_to_float(value);
}

template <>
__device__ inline float storage_to_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float storage_to_float<int8_t>(const int8_t value) {
  return static_cast<float>(value);
}

template <KvFormat KV>
struct KvStorage;

template <>
struct KvStorage<KvFormat::kBFloat16> {
  using type = __nv_bfloat16;
};

template <>
struct KvStorage<KvFormat::kFp8E4M3> {
  using type = __nv_fp8_e4m3;
};

template <>
struct KvStorage<KvFormat::kInt8> {
  using type = int8_t;
};

template <KvFormat KV, int HeadDim, int PageSize>
__device__ inline float load_key_global(const void* key_cache, const float* key_scales,
                                        const int page_id, const int kv_head_index,
                                        const int num_kv_heads, const int token_index,
                                        const int dim_index) {
  const std::size_t flat_index =
      (((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
        static_cast<std::size_t>(kv_head_index)) *
           static_cast<std::size_t>(PageSize) +
       static_cast<std::size_t>(token_index)) *
          static_cast<std::size_t>(HeadDim) +
      static_cast<std::size_t>(dim_index);

  if constexpr (KV == KvFormat::kBFloat16) {
    const auto* key_data = reinterpret_cast<const __nv_bfloat16*>(key_cache);
    return bf16_to_float(key_data[flat_index]);
  } else {
    constexpr int kDimBlock = 64;
    const int scale_block = dim_index / kDimBlock;
    const std::size_t scale_index =
        ((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
         static_cast<std::size_t>(kv_head_index)) *
            static_cast<std::size_t>(HeadDim / kDimBlock) +
        static_cast<std::size_t>(scale_block);
    const float scale = key_scales[scale_index];
    const auto* key_data = reinterpret_cast<const typename KvStorage<KV>::type*>(key_cache);
    return storage_to_float<typename KvStorage<KV>::type>(key_data[flat_index]) * scale;
  }
}

template <KvFormat KV, int HeadDim, int PageSize>
__device__ inline float load_value_global(const void* value_cache, const float* value_scales,
                                          const int page_id, const int kv_head_index,
                                          const int num_kv_heads, const int token_index,
                                          const int dim_index) {
  const std::size_t flat_index =
      (((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
        static_cast<std::size_t>(kv_head_index)) *
           static_cast<std::size_t>(PageSize) +
       static_cast<std::size_t>(token_index)) *
          static_cast<std::size_t>(HeadDim) +
      static_cast<std::size_t>(dim_index);

  if constexpr (KV == KvFormat::kBFloat16) {
    const auto* value_data = reinterpret_cast<const __nv_bfloat16*>(value_cache);
    return bf16_to_float(value_data[flat_index]);
  } else {
    constexpr int kDimBlock = 64;
    const int scale_block = dim_index / kDimBlock;
    const std::size_t scale_index =
        ((((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
           static_cast<std::size_t>(kv_head_index)) *
              static_cast<std::size_t>(PageSize)) +
         static_cast<std::size_t>(token_index)) *
            static_cast<std::size_t>(HeadDim / kDimBlock) +
        static_cast<std::size_t>(scale_block);
    const float scale = value_scales[scale_index];
    const auto* value_data = reinterpret_cast<const typename KvStorage<KV>::type*>(value_cache);
    return storage_to_float<typename KvStorage<KV>::type>(value_data[flat_index]) * scale;
  }
}

template <KvFormat KV, int HeadDim, int PageSize>
struct SharedPageLoader;

template <int HeadDim, int PageSize>
struct SharedPageLoader<KvFormat::kBFloat16, HeadDim, PageSize> {
  static __device__ inline void load(const void* key_cache, const void* value_cache,
                                     const float* /*key_scales*/, const float* /*value_scales*/,
                                     const int num_kv_heads, const int kv_head_index,
                                     const int page_id, const int thread_linear,
                                     const int thread_stride, __nv_bfloat16* smem_k,
                                     __nv_bfloat16* smem_v) {
    const auto* key_pages = reinterpret_cast<const __nv_bfloat16*>(key_cache);
    const auto* value_pages = reinterpret_cast<const __nv_bfloat16*>(value_cache);
    const std::size_t page_stride = static_cast<std::size_t>(PageSize) * HeadDim;
    const std::size_t base =
        ((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
         static_cast<std::size_t>(kv_head_index)) *
        page_stride;
    for (int idx = thread_linear; idx < static_cast<int>(page_stride); idx += thread_stride) {
      smem_k[idx] = key_pages[base + static_cast<std::size_t>(idx)];
      smem_v[idx] = value_pages[base + static_cast<std::size_t>(idx)];
    }
  }
};

template <int HeadDim, int PageSize>
struct SharedPageLoader<KvFormat::kFp8E4M3, HeadDim, PageSize> {
  static __device__ inline void load(const void* key_cache, const void* value_cache,
                                     const float* key_scales, const float* value_scales,
                                     const int num_kv_heads, const int kv_head_index,
                                     const int page_id, const int thread_linear,
                                     const int thread_stride, __nv_bfloat16* smem_k,
                                     __nv_bfloat16* smem_v) {
    constexpr int kDimBlock = 64;
    const auto* key_pages = reinterpret_cast<const __nv_fp8_e4m3*>(key_cache);
    const auto* value_pages = reinterpret_cast<const __nv_fp8_e4m3*>(value_cache);
    const std::size_t page_stride = static_cast<std::size_t>(PageSize) * HeadDim;
    const std::size_t page_head_base =
        ((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
         static_cast<std::size_t>(kv_head_index));
    const std::size_t base = page_head_base * page_stride;
    const std::size_t key_scale_base = page_head_base * static_cast<std::size_t>(HeadDim / kDimBlock);
    const std::size_t value_scale_base =
        page_head_base * static_cast<std::size_t>(PageSize) * static_cast<std::size_t>(HeadDim / kDimBlock);
    for (int idx = thread_linear; idx < static_cast<int>(page_stride); idx += thread_stride) {
      const int token_index = idx / HeadDim;
      const int dim_index = idx % HeadDim;
      const int scale_block = dim_index / kDimBlock;
      const float key_value =
          static_cast<float>(key_pages[base + static_cast<std::size_t>(idx)]) *
          key_scales[key_scale_base + static_cast<std::size_t>(scale_block)];
      const float value_value =
          static_cast<float>(value_pages[base + static_cast<std::size_t>(idx)]) *
          value_scales[value_scale_base +
                       static_cast<std::size_t>(token_index * (HeadDim / kDimBlock) + scale_block)];
      smem_k[idx] = float_to_bf16(key_value);
      smem_v[idx] = float_to_bf16(value_value);
    }
  }
};

template <int HeadDim, int PageSize>
struct SharedPageLoader<KvFormat::kInt8, HeadDim, PageSize> {
  static __device__ inline void load(const void* key_cache, const void* value_cache,
                                     const float* key_scales, const float* value_scales,
                                     const int num_kv_heads, const int kv_head_index,
                                     const int page_id, const int thread_linear,
                                     const int thread_stride, __nv_bfloat16* smem_k,
                                     __nv_bfloat16* smem_v) {
    constexpr int kDimBlock = 64;
    const auto* key_pages = reinterpret_cast<const int8_t*>(key_cache);
    const auto* value_pages = reinterpret_cast<const int8_t*>(value_cache);
    const std::size_t page_stride = static_cast<std::size_t>(PageSize) * HeadDim;
    const std::size_t page_head_base =
        ((static_cast<std::size_t>(page_id) * static_cast<std::size_t>(num_kv_heads)) +
         static_cast<std::size_t>(kv_head_index));
    const std::size_t base = page_head_base * page_stride;
    const std::size_t key_scale_base = page_head_base * static_cast<std::size_t>(HeadDim / kDimBlock);
    const std::size_t value_scale_base =
        page_head_base * static_cast<std::size_t>(PageSize) * static_cast<std::size_t>(HeadDim / kDimBlock);
    for (int idx = thread_linear; idx < static_cast<int>(page_stride); idx += thread_stride) {
      const int token_index = idx / HeadDim;
      const int dim_index = idx % HeadDim;
      const int scale_block = dim_index / kDimBlock;
      const float key_value =
          static_cast<float>(key_pages[base + static_cast<std::size_t>(idx)]) *
          key_scales[key_scale_base + static_cast<std::size_t>(scale_block)];
      const float value_value =
          static_cast<float>(value_pages[base + static_cast<std::size_t>(idx)]) *
          value_scales[value_scale_base +
                       static_cast<std::size_t>(token_index * (HeadDim / kDimBlock) + scale_block)];
      smem_k[idx] = float_to_bf16(key_value);
      smem_v[idx] = float_to_bf16(value_value);
    }
  }
};

template <int HeadDim>
struct QueryAccumulator {
  float q_even0 = 0.0f;
  float q_odd0 = 0.0f;
  float acc_even0 = 0.0f;
  float acc_odd0 = 0.0f;
  float q_even1 = 0.0f;
  float q_odd1 = 0.0f;
  float acc_even1 = 0.0f;
  float acc_odd1 = 0.0f;

  __device__ inline void zero_acc() {
    acc_even0 = 0.0f;
    acc_odd0 = 0.0f;
    if constexpr (HeadDim == 128) {
      acc_even1 = 0.0f;
      acc_odd1 = 0.0f;
    }
  }
};

template <int HeadDim>
__device__ inline void load_rotated_query(const __nv_bfloat16* query_row, const int query_position,
                                          const float rope_theta, const int lane_id,
                                          QueryAccumulator<HeadDim>& q) {
  const int pair_index0 = lane_id;
  const int dim_even0 = lane_id * 2;
  const float inv_freq0 = rope_inv_freq(pair_index0, HeadDim, rope_theta);
  float cos0 = 0.0f;
  float sin0 = 0.0f;
  sincosf(static_cast<float>(query_position) * inv_freq0, &sin0, &cos0);
  const float q_in_even0 = bf16_to_float(query_row[dim_even0]);
  const float q_in_odd0 = bf16_to_float(query_row[dim_even0 + 1]);
  q.q_even0 = (q_in_even0 * cos0) - (q_in_odd0 * sin0);
  q.q_odd0 = (q_in_even0 * sin0) + (q_in_odd0 * cos0);
  if constexpr (HeadDim == 128) {
    const int pair_index1 = lane_id + kWarpSize;
    const int dim_even1 = dim_even0 + 64;
    const float inv_freq1 = rope_inv_freq(pair_index1, HeadDim, rope_theta);
    float cos1 = 0.0f;
    float sin1 = 0.0f;
    sincosf(static_cast<float>(query_position) * inv_freq1, &sin1, &cos1);
    const float q_in_even1 = bf16_to_float(query_row[dim_even1]);
    const float q_in_odd1 = bf16_to_float(query_row[dim_even1 + 1]);
    q.q_even1 = (q_in_even1 * cos1) - (q_in_odd1 * sin1);
    q.q_odd1 = (q_in_even1 * sin1) + (q_in_odd1 * cos1);
  }
  q.zero_acc();
}

template <int HeadDim>
__device__ inline float score_token_shared(const QueryAccumulator<HeadDim>& q,
                                           const __nv_bfloat16* key_row, const int lane_id) {
  const int dim_even0 = lane_id * 2;
  float partial = (q.q_even0 * bf16_to_float(key_row[dim_even0])) +
                  (q.q_odd0 * bf16_to_float(key_row[dim_even0 + 1]));
  if constexpr (HeadDim == 128) {
    const int dim_even1 = dim_even0 + 64;
    partial += (q.q_even1 * bf16_to_float(key_row[dim_even1])) +
               (q.q_odd1 * bf16_to_float(key_row[dim_even1 + 1]));
  }
  return warp_sum(partial);
}

template <KvFormat KV, int HeadDim, int PageSize>
__device__ inline float score_token_global(const QueryAccumulator<HeadDim>& q, const void* key_cache,
                                           const float* key_scales, const int page_id,
                                           const int kv_head_index, const int num_kv_heads,
                                           const int token_index, const int lane_id) {
  const int dim_even0 = lane_id * 2;
  float partial =
      (q.q_even0 *
       load_key_global<KV, HeadDim, PageSize>(key_cache, key_scales, page_id, kv_head_index,
                                              num_kv_heads, token_index, dim_even0)) +
      (q.q_odd0 *
       load_key_global<KV, HeadDim, PageSize>(key_cache, key_scales, page_id, kv_head_index,
                                              num_kv_heads, token_index, dim_even0 + 1));
  if constexpr (HeadDim == 128) {
    const int dim_even1 = dim_even0 + 64;
    partial +=
        (q.q_even1 *
         load_key_global<KV, HeadDim, PageSize>(key_cache, key_scales, page_id, kv_head_index,
                                                num_kv_heads, token_index, dim_even1)) +
        (q.q_odd1 *
         load_key_global<KV, HeadDim, PageSize>(key_cache, key_scales, page_id, kv_head_index,
                                                num_kv_heads, token_index, dim_even1 + 1));
  }
  return warp_sum(partial);
}

template <int HeadDim>
__device__ inline void online_softmax_step(const float score, const float value_even0,
                                           const float value_odd0, float& running_m,
                                           float& running_l,
                                           QueryAccumulator<HeadDim>& running_acc,
                                           const float value_even1 = 0.0f,
                                           const float value_odd1 = 0.0f) {
  const float merged_m = fmaxf(running_m, score);
  const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - merged_m);
  const float beta = expf(score - merged_m);

  running_acc.acc_even0 = alpha * running_acc.acc_even0 + beta * value_even0;
  running_acc.acc_odd0 = alpha * running_acc.acc_odd0 + beta * value_odd0;
  if constexpr (HeadDim == 128) {
    running_acc.acc_even1 = alpha * running_acc.acc_even1 + beta * value_even1;
    running_acc.acc_odd1 = alpha * running_acc.acc_odd1 + beta * value_odd1;
  }
  running_l = alpha * running_l + beta;
  running_m = merged_m;
}

template <int HeadDim, int PageSize>
__device__ inline void accumulate_shared_page_exact(const QueryAccumulator<HeadDim>& q,
                                                    const __nv_bfloat16* page_k,
                                                    const __nv_bfloat16* page_v,
                                                    const int valid_tokens,
                                                    const float softmax_scale,
                                                    const int lane_id, float& running_m,
                                                    float& running_l,
                                                    QueryAccumulator<HeadDim>& running_acc) {
  for (int token_index = 0; token_index < valid_tokens; ++token_index) {
    const __nv_bfloat16* key_row = page_k + static_cast<std::size_t>(token_index) * HeadDim;
    const __nv_bfloat16* value_row = page_v + static_cast<std::size_t>(token_index) * HeadDim;
    const float score = softmax_scale * score_token_shared<HeadDim>(q, key_row, lane_id);
    const int dim_even0 = lane_id * 2;
    const float value_even0 = bf16_to_float(value_row[dim_even0]);
    const float value_odd0 = bf16_to_float(value_row[dim_even0 + 1]);
    if constexpr (HeadDim == 128) {
      const int dim_even1 = dim_even0 + 64;
      online_softmax_step<HeadDim>(
          score, value_even0, value_odd0, running_m, running_l, running_acc,
          bf16_to_float(value_row[dim_even1]), bf16_to_float(value_row[dim_even1 + 1]));
    } else {
      online_softmax_step<HeadDim>(score, value_even0, value_odd0, running_m, running_l,
                                   running_acc);
    }
  }
}

template <KvFormat KV, int HeadDim, int PageSize>
__device__ inline void accumulate_global_page_exact(const QueryAccumulator<HeadDim>& q,
                                                    const void* key_cache, const void* value_cache,
                                                    const float* key_scales,
                                                    const float* value_scales,
                                                    const int page_id, const int kv_head_index,
                                                    const int num_kv_heads,
                                                    const int valid_tokens,
                                                    const float softmax_scale,
                                                    const int lane_id, float& running_m,
                                                    float& running_l,
                                                    QueryAccumulator<HeadDim>& running_acc) {
  for (int token_index = 0; token_index < valid_tokens; ++token_index) {
    const float score = softmax_scale * score_token_global<KV, HeadDim, PageSize>(
                                          q, key_cache, key_scales, page_id, kv_head_index,
                                          num_kv_heads, token_index, lane_id);
    const int dim_even0 = lane_id * 2;
    const float value_even0 =
        load_value_global<KV, HeadDim, PageSize>(value_cache, value_scales, page_id,
                                                 kv_head_index, num_kv_heads, token_index,
                                                 dim_even0);
    const float value_odd0 =
        load_value_global<KV, HeadDim, PageSize>(value_cache, value_scales, page_id,
                                                 kv_head_index, num_kv_heads, token_index,
                                                 dim_even0 + 1);
    if constexpr (HeadDim == 128) {
      const int dim_even1 = dim_even0 + 64;
      online_softmax_step<HeadDim>(
          score, value_even0, value_odd0, running_m, running_l, running_acc,
          load_value_global<KV, HeadDim, PageSize>(value_cache, value_scales, page_id,
                                                   kv_head_index, num_kv_heads, token_index,
                                                   dim_even1),
          load_value_global<KV, HeadDim, PageSize>(value_cache, value_scales, page_id,
                                                   kv_head_index, num_kv_heads, token_index,
                                                   dim_even1 + 1));
    } else {
      online_softmax_step<HeadDim>(score, value_even0, value_odd0, running_m, running_l,
                                   running_acc);
    }
  }
}

__device__ inline void tcgen05_sync_hint() {
#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL))
  cuda::ptx::tcgen05_fence_before_thread_sync();
  __syncwarp();
  cuda::ptx::tcgen05_fence_after_thread_sync();
#endif
}

template <int HeadDim, int PageSize, int ClusterSize, KvFormat KV>
__global__ __launch_bounds__(kThreadsPerBlock) void prefix_union_decode_cluster_kernel(
    const __nv_bfloat16* __restrict__ query, const void* __restrict__ key_cache,
    const void* __restrict__ value_cache, const float* __restrict__ key_scales,
    const float* __restrict__ value_scales, int* __restrict__ task_counter,
    const PrefixUnionTask* __restrict__ tasks, const SharedPageRef* __restrict__ shared_pages,
    const TailPageRef* __restrict__ tail_pages, const ConsumerRef* __restrict__ consumers,
    const int num_tasks, const int num_kv_heads, const float softmax_scale,
    const float rope_theta, __nv_bfloat16* __restrict__ output) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(PageSize == 16 || PageSize == 32);
  static_assert(ClusterSize == 1 || ClusterSize == 2 || ClusterSize == 4 || ClusterSize == 8);

  const cg::cluster_group cluster = cg::this_cluster();
  const int block_rank = static_cast<int>(cluster.block_rank());
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;

  __shared__ __nv_bfloat16 smem_k[2][PageSize][HeadDim];
  __shared__ __nv_bfloat16 smem_v[2][PageSize][HeadDim];
  __shared__ int smem_valid_tokens[2];
  __shared__ int smem_task_index;

  const __nv_bfloat16* remote_k[2] = {
      cluster.map_shared_rank(&smem_k[0][0][0], 0),
      cluster.map_shared_rank(&smem_k[1][0][0], 0),
  };
  const __nv_bfloat16* remote_v[2] = {
      cluster.map_shared_rank(&smem_v[0][0][0], 0),
      cluster.map_shared_rank(&smem_v[1][0][0], 0),
  };
  const int* remote_valid_tokens[2] = {
      cluster.map_shared_rank(&smem_valid_tokens[0], 0),
      cluster.map_shared_rank(&smem_valid_tokens[1], 0),
  };
  const int* remote_task_index = cluster.map_shared_rank(&smem_task_index, 0);
  constexpr bool kHasDedicatedLoaderBlock = ClusterSize > 1;
  constexpr int kConsumerBlockBase = kHasDedicatedLoaderBlock ? 1 : 0;
  constexpr int kConsumerBlockCount = kHasDedicatedLoaderBlock ? (ClusterSize - 1) : ClusterSize;
  constexpr int kConsumerWarpsPerCluster = kConsumerBlockCount * kWarpsPerBlock;
  const int consumer_cluster_warp_id =
      (block_rank < kConsumerBlockBase) ? -1 : ((block_rank - kConsumerBlockBase) * kWarpsPerBlock) + warp_id;

  while (true) {
    if (block_rank == 0 && threadIdx.x == 0) {
      smem_task_index = atomicAdd(task_counter, 1);
    }
    cluster.sync();
    const int task_index = *remote_task_index;
    if (task_index >= num_tasks) {
      return;
    }

    const PrefixUnionTask task = tasks[task_index];

    for (int consumer_wave_base = 0; consumer_wave_base < task.num_consumers;
         consumer_wave_base += kConsumerWarpsPerCluster) {
      const int consumer_slot = consumer_wave_base + consumer_cluster_warp_id;
      const bool active_consumer =
          (consumer_cluster_warp_id >= 0) && (consumer_slot < task.num_consumers);

      ConsumerRef consumer{};
      QueryAccumulator<HeadDim> state{};
      float running_m = -CUDART_INF_F;
      float running_l = 0.0f;

      if (active_consumer) {
        consumer = consumers[task.consumer_offset + consumer_slot];
        const __nv_bfloat16* query_row = query +
                                         static_cast<std::size_t>(consumer.query_index) *
                                             static_cast<std::size_t>(HeadDim);
        load_rotated_query<HeadDim>(query_row, consumer.query_position, rope_theta, lane_id,
                                    state);
      }

      if (task.num_shared_pages > 0) {
        if constexpr (kHasDedicatedLoaderBlock) {
          if (block_rank == 0) {
            const SharedPageRef first_page = shared_pages[task.shared_page_offset];
            if (threadIdx.x == 0) {
              smem_valid_tokens[0] = first_page.valid_tokens;
            }
            SharedPageLoader<KV, HeadDim, PageSize>::load(
                key_cache, value_cache, key_scales, value_scales, num_kv_heads,
                task.kv_head_index, first_page.page_id, static_cast<int>(threadIdx.x),
                static_cast<int>(blockDim.x), &smem_k[0][0][0], &smem_v[0][0][0]);
          }
          cluster.sync();
          for (int page_idx = 0; page_idx < task.num_shared_pages; ++page_idx) {
            const int buffer_index = page_idx & 1;
            const int next_buffer_index = buffer_index ^ 1;
            if (block_rank == 0 && (page_idx + 1) < task.num_shared_pages) {
              const SharedPageRef next_page = shared_pages[task.shared_page_offset + page_idx + 1];
              if (threadIdx.x == 0) {
                smem_valid_tokens[next_buffer_index] = next_page.valid_tokens;
              }
              SharedPageLoader<KV, HeadDim, PageSize>::load(
                  key_cache, value_cache, key_scales, value_scales, num_kv_heads,
                  task.kv_head_index, next_page.page_id, static_cast<int>(threadIdx.x),
                  static_cast<int>(blockDim.x), &smem_k[next_buffer_index][0][0],
                  &smem_v[next_buffer_index][0][0]);
            }

            tcgen05_sync_hint();
            if (active_consumer) {
              accumulate_shared_page_exact<HeadDim, PageSize>(
                  state, remote_k[buffer_index], remote_v[buffer_index],
                  *remote_valid_tokens[buffer_index], softmax_scale, lane_id, running_m,
                  running_l, state);
            }

            cluster.sync();
          }
        } else {
          for (int page_idx = 0; page_idx < task.num_shared_pages; ++page_idx) {
            if (block_rank == 0) {
              const SharedPageRef page = shared_pages[task.shared_page_offset + page_idx];
              if (threadIdx.x == 0) {
                smem_valid_tokens[0] = page.valid_tokens;
              }
              SharedPageLoader<KV, HeadDim, PageSize>::load(
                  key_cache, value_cache, key_scales, value_scales, num_kv_heads,
                  task.kv_head_index, page.page_id, static_cast<int>(threadIdx.x),
                  static_cast<int>(blockDim.x), &smem_k[0][0][0], &smem_v[0][0][0]);
            }
            cluster.sync();
            tcgen05_sync_hint();
            if (active_consumer) {
              accumulate_shared_page_exact<HeadDim, PageSize>(
                  state, remote_k[0], remote_v[0], *remote_valid_tokens[0], softmax_scale,
                  lane_id, running_m, running_l, state);
            }
            cluster.sync();
          }
        }
      }

      if (active_consumer) {
        for (int tail_index = 0; tail_index < consumer.num_tail_pages; ++tail_index) {
          const TailPageRef page = tail_pages[consumer.tail_page_offset + tail_index];
          accumulate_global_page_exact<KV, HeadDim, PageSize>(
              state, key_cache, value_cache, key_scales, value_scales, page.page_id,
              task.kv_head_index, num_kv_heads, page.valid_tokens, softmax_scale, lane_id,
              running_m, running_l, state);
        }

        const float inv_l = (running_l == 0.0f) ? 0.0f : (1.0f / running_l);
        __nv_bfloat16* out_row = output +
                                 static_cast<std::size_t>(consumer.output_index) *
                                     static_cast<std::size_t>(HeadDim);
        const int dim_even0 = lane_id * 2;
        out_row[dim_even0] = float_to_bf16(state.acc_even0 * inv_l);
        out_row[dim_even0 + 1] = float_to_bf16(state.acc_odd0 * inv_l);
        if constexpr (HeadDim == 128) {
          const int dim_even1 = dim_even0 + 64;
          out_row[dim_even1] = float_to_bf16(state.acc_even1 * inv_l);
          out_row[dim_even1 + 1] = float_to_bf16(state.acc_odd1 * inv_l);
        }
      }

      cluster.sync();
    }
  }
}

template <int HeadDim, int PageSize, int ClusterSize, KvFormat KV>
void launch_prefix_union_variant(const __nv_bfloat16* query, const void* key_cache,
                                 const void* value_cache, const float* key_scales,
                                 const float* value_scales, int* task_counter,
                                 const PrefixUnionTask* tasks,
                                 const SharedPageRef* shared_pages,
                                 const TailPageRef* tail_pages,
                                 const ConsumerRef* consumers, const int num_tasks,
                                 const int num_kv_heads, const int grid_clusters,
                                 const float softmax_scale, const float rope_theta,
                                 __nv_bfloat16* output, cudaStream_t stream) {
  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = static_cast<unsigned>(ClusterSize);
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_clusters * ClusterSize), 1u, 1u);
  config.blockDim = dim3(kThreadsPerBlock, 1u, 1u);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 1;

  void* args[] = {
      const_cast<__nv_bfloat16**>(&query), const_cast<void**>(&key_cache),
      const_cast<void**>(&value_cache),   const_cast<float**>(&key_scales),
      const_cast<float**>(&value_scales), &task_counter,
      const_cast<PrefixUnionTask**>(&tasks),
      const_cast<SharedPageRef**>(&shared_pages), const_cast<TailPageRef**>(&tail_pages),
      const_cast<ConsumerRef**>(&consumers), const_cast<int*>(&num_tasks),
      const_cast<int*>(&num_kv_heads), const_cast<float*>(&softmax_scale),
      const_cast<float*>(&rope_theta), &output,
  };

  FK_CUDA_CHECK(cudaLaunchKernelExC(
      &config,
      reinterpret_cast<const void*>(
          prefix_union_decode_cluster_kernel<HeadDim, PageSize, ClusterSize, KV>),
      args));
}

template <int HeadDim, int PageSize, int ClusterSize>
void dispatch_prefix_union_format(const __nv_bfloat16* query, const void* key_cache,
                                  const void* value_cache, const float* key_scales,
                                  const float* value_scales, int* task_counter,
                                  const PrefixUnionTask* tasks, const SharedPageRef* shared_pages,
                                  const TailPageRef* tail_pages,
                                  const ConsumerRef* consumers, const int num_tasks,
                                  const int num_kv_heads, const int grid_clusters,
                                  const int kv_format,
                                  const float softmax_scale, const float rope_theta,
                                  __nv_bfloat16* output, cudaStream_t stream) {
  switch (static_cast<KvFormat>(kv_format)) {
  case KvFormat::kBFloat16:
    launch_prefix_union_variant<HeadDim, PageSize, ClusterSize, KvFormat::kBFloat16>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, softmax_scale, rope_theta, output, stream);
    return;
  case KvFormat::kFp8E4M3:
    launch_prefix_union_variant<HeadDim, PageSize, ClusterSize, KvFormat::kFp8E4M3>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, softmax_scale, rope_theta, output, stream);
    return;
  case KvFormat::kInt8:
    launch_prefix_union_variant<HeadDim, PageSize, ClusterSize, KvFormat::kInt8>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, softmax_scale, rope_theta, output, stream);
    return;
  }
  throw std::invalid_argument("unsupported kv_format");
}

template <int HeadDim, int PageSize>
void dispatch_prefix_union_cluster_size(const __nv_bfloat16* query, const void* key_cache,
                                        const void* value_cache, const float* key_scales,
                                        const float* value_scales, int* task_counter,
                                        const PrefixUnionTask* tasks,
                                        const SharedPageRef* shared_pages,
                                        const TailPageRef* tail_pages,
                                        const ConsumerRef* consumers, const int num_tasks,
                                        const int num_kv_heads, const int cluster_size,
                                        const int grid_clusters, const int kv_format,
                                        const float softmax_scale,
                                        const float rope_theta, __nv_bfloat16* output,
                                        cudaStream_t stream) {
  switch (cluster_size) {
  case 1:
    dispatch_prefix_union_format<HeadDim, PageSize, 1>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, kv_format, softmax_scale, rope_theta, output, stream);
    return;
  case 2:
    dispatch_prefix_union_format<HeadDim, PageSize, 2>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, kv_format, softmax_scale, rope_theta, output, stream);
    return;
  case 4:
    dispatch_prefix_union_format<HeadDim, PageSize, 4>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, kv_format, softmax_scale, rope_theta, output, stream);
    return;
  case 8:
    dispatch_prefix_union_format<HeadDim, PageSize, 8>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, grid_clusters, kv_format, softmax_scale, rope_theta, output, stream);
    return;
  }
  throw std::invalid_argument("cluster_size must be one of {1, 2, 4, 8}");
}

void launch_prefix_union_decode(const __nv_bfloat16* query, const void* key_cache,
                                const void* value_cache, const float* key_scales,
                                const float* value_scales, int* task_counter,
                                const PrefixUnionTask* tasks,
                                const SharedPageRef* shared_pages,
                                const TailPageRef* tail_pages, const ConsumerRef* consumers,
                                const int num_tasks, const int num_kv_heads, const int head_dim,
                                const int page_size, const int cluster_size,
                                const int grid_clusters, const int kv_format,
                                const float softmax_scale, const float rope_theta,
                                __nv_bfloat16* output, cudaStream_t stream) {
  if (head_dim == 64 && page_size == 16) {
    dispatch_prefix_union_cluster_size<64, 16>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, cluster_size, grid_clusters, kv_format, softmax_scale, rope_theta, output,
        stream);
    return;
  }
  if (head_dim == 64 && page_size == 32) {
    dispatch_prefix_union_cluster_size<64, 32>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, cluster_size, grid_clusters, kv_format, softmax_scale, rope_theta, output,
        stream);
    return;
  }
  if (head_dim == 128 && page_size == 16) {
    dispatch_prefix_union_cluster_size<128, 16>(
        query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
        shared_pages, tail_pages, consumers, num_tasks,
        num_kv_heads, cluster_size, grid_clusters, kv_format, softmax_scale, rope_theta, output,
        stream);
    return;
  }
  dispatch_prefix_union_cluster_size<128, 32>(
      query, key_cache, value_cache, key_scales, value_scales, task_counter, tasks,
      shared_pages, tail_pages, consumers, num_tasks,
      num_kv_heads, cluster_size, grid_clusters, kv_format, softmax_scale, rope_theta, output,
      stream);
}

} // namespace

void prefix_union_decode_forward(
    std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
    std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr, std::uintptr_t tasks_ptr,
    std::uintptr_t shared_pages_ptr, std::uintptr_t tail_pages_ptr, std::uintptr_t consumers_ptr,
    std::uintptr_t scheduler_counter_ptr, std::uintptr_t output_ptr, const int num_tasks,
    const int num_pages, const int num_q_heads, const int num_kv_heads, const int head_dim,
    const int page_size, const int cluster_size, const int kv_format, const int keys_are_rotated,
    const float softmax_scale, const float rope_theta, std::uintptr_t stream_ptr) {
  if (num_tasks < 0 || num_pages < 0 || num_q_heads <= 0 || num_kv_heads <= 0) {
    throw std::invalid_argument("prefix_union_decode_forward received invalid sizes");
  }
  if (head_dim != 64 && head_dim != 128) {
    throw std::invalid_argument("prefix_union_decode_forward only supports head_dim 64 or 128");
  }
  if (page_size != 16 && page_size != 32) {
    throw std::invalid_argument("prefix_union_decode_forward only supports page_size 16 or 32");
  }
  if (cluster_size != 1 && cluster_size != 2 && cluster_size != 4 && cluster_size != 8) {
    throw std::invalid_argument("cluster_size must be one of {1, 2, 4, 8}");
  }
  if (num_q_heads % num_kv_heads != 0) {
    throw std::invalid_argument("num_q_heads must be divisible by num_kv_heads");
  }
  if (kv_format != static_cast<int>(KvFormat::kBFloat16) &&
      kv_format != static_cast<int>(KvFormat::kFp8E4M3) &&
      kv_format != static_cast<int>(KvFormat::kInt8)) {
    throw std::invalid_argument("unsupported kv_format");
  }
  if (keys_are_rotated == 0) {
    throw std::invalid_argument(
        "prefix_union_decode_forward currently requires pre-rotated key pages");
  }
  if (num_tasks == 0) {
    return;
  }

  int device = -1;
  FK_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  FK_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major < 12) {
    throw std::invalid_argument(
        "prefix_union_decode_forward requires a Blackwell-class GPU (compute capability 12.x)");
  }

  const auto* query = reinterpret_cast<const __nv_bfloat16*>(query_ptr);
  const auto* key_cache = reinterpret_cast<const void*>(key_ptr);
  const auto* value_cache = reinterpret_cast<const void*>(value_ptr);
  const auto* key_scales = reinterpret_cast<const float*>(key_scales_ptr);
  const auto* value_scales = reinterpret_cast<const float*>(value_scales_ptr);
  const auto* tasks = reinterpret_cast<const PrefixUnionTask*>(tasks_ptr);
  const auto* shared_pages = reinterpret_cast<const SharedPageRef*>(shared_pages_ptr);
  const auto* tail_pages = reinterpret_cast<const TailPageRef*>(tail_pages_ptr);
  const auto* consumers = reinterpret_cast<const ConsumerRef*>(consumers_ptr);
  auto* scheduler_counter = reinterpret_cast<int*>(scheduler_counter_ptr);
  auto* output = reinterpret_cast<__nv_bfloat16*>(output_ptr);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  int grid_clusters = props.multiProcessorCount * 2;
  if (grid_clusters <= 0) {
    grid_clusters = 1;
  }
  grid_clusters = (num_tasks < grid_clusters) ? num_tasks : grid_clusters;
  if (grid_clusters <= 0) {
    grid_clusters = 1;
  }

  launch_prefix_union_decode(query, key_cache, value_cache, key_scales, value_scales,
                             scheduler_counter, tasks, shared_pages, tail_pages, consumers,
                             num_tasks, num_kv_heads, head_dim, page_size, cluster_size,
                             grid_clusters, kv_format, softmax_scale, rope_theta, output,
                             stream);
  FK_CUDA_CHECK(cudaGetLastError());
}

} // namespace fast_kernels::prefix_union_decode
