#include "ops/clustered_page_decode/clustered_page_decode.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace fast_kernels::clustered_page_decode {

namespace {

constexpr int kWarpSize = 32;
constexpr int kClusterThreadsPerBlock = 256;
constexpr int kMaxDirectGroupSize = 8;
constexpr int kMaxGroupTile = 8;
constexpr int kMaxHeadDim = 128;
constexpr int kDimBlock = 64;
constexpr int kMaxClusterSize = 8;

#define FK_CUDA_CHECK(expr)                                                                        \
  do {                                                                                             \
    const cudaError_t error__ = (expr);                                                            \
    if (error__ != cudaSuccess) {                                                                  \
      throw std::runtime_error(cudaGetErrorString(error__));                                       \
    }                                                                                              \
  } while (false)

__device__ inline float bf16_to_float(const __nv_bfloat16 value) { return __bfloat162float(value); }

__device__ inline __nv_bfloat16 float_to_bf16(const float value) {
  return __float2bfloat16_rn(value);
}

__device__ inline float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ inline float rope_inv_freq(const int pair_index, const int head_dim,
                                      const float rope_theta) {
  return powf(rope_theta, -2.0f * static_cast<float>(pair_index) / static_cast<float>(head_dim));
}

__device__ inline void sincos_pair(const float angle, float& c, float& s) {
  sincosf(angle, &s, &c);
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

template <KvFormat KV>
__device__ inline float load_storage_element(const typename KvStorage<KV>::type* data,
                                             const std::size_t data_index, const float scale) {
  if constexpr (KV == KvFormat::kBFloat16) {
    return bf16_to_float(data[data_index]);
  } else if constexpr (KV == KvFormat::kFp8E4M3) {
    return static_cast<float>(data[data_index]) * scale;
  } else {
    return static_cast<float>(data[data_index]) * scale;
  }
}

template <KvFormat KV>
__device__ inline float load_element(const void* data_ptr, const float* scales,
                                     const std::size_t data_index, const std::size_t scale_index) {
  const auto* data = reinterpret_cast<const typename KvStorage<KV>::type*>(data_ptr);
  if constexpr (KV == KvFormat::kBFloat16) {
    return load_storage_element<KV>(data, data_index, 1.0f);
  } else {
    return load_storage_element<KV>(data, data_index, scales[scale_index]);
  }
}

__device__ inline void apply_llama_rope_pair(const float in_even, const float in_odd,
                                             const int pair_index, const int head_dim,
                                             const int position, const float rope_theta,
                                             float& out_even, float& out_odd) {
  const float inv_freq = rope_inv_freq(pair_index, head_dim, rope_theta);
  float c = 0.0f;
  float s = 0.0f;
  sincos_pair(static_cast<float>(position) * inv_freq, c, s);
  out_even = (in_even * c) - (in_odd * s);
  out_odd = (in_even * s) + (in_odd * c);
}

template <int ThreadsPerBlock, int HeadDim, int PageSize, KvFormat KV, bool KeysAreRotated>
__global__ __launch_bounds__(ThreadsPerBlock) void direct_page_decode_kernel(
    const __nv_bfloat16* query, const void* key_cache, const void* value_cache,
    const float* key_scales, const float* value_scales, const int32_t* run_base_pages,
    const int32_t* run_page_counts, const int32_t* run_logical_starts,
    const int32_t* run_last_page_lens, const int32_t* request_run_offsets, const int32_t* seq_lens,
    __nv_bfloat16* output, const int num_q_heads, const int num_kv_heads, const float softmax_scale,
    const float rope_theta) {
  static_assert(ThreadsPerBlock % kWarpSize == 0);
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(PageSize == 16 || PageSize == 32);

  constexpr int kDimBlocks = HeadDim / kDimBlock;

  const int task_index = static_cast<int>(blockIdx.x);
  const int request_index = task_index / num_kv_heads;
  const int kv_head_index = task_index % num_kv_heads;
  const int group_size = num_q_heads / num_kv_heads;
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;

  if (warp_id >= group_size) {
    return;
  }

  const int q_head_index = kv_head_index * group_size + warp_id;
  const std::size_t output_base =
      ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
        static_cast<std::size_t>(q_head_index)) *
       static_cast<std::size_t>(HeadDim));
  const int query_position = seq_lens[request_index] - 1;
  const int run_begin = request_run_offsets[request_index];
  const int run_end = request_run_offsets[request_index + 1];
  if (query_position < 0 || run_begin == run_end) {
    output[output_base + static_cast<std::size_t>(lane_id * 2)] = float_to_bf16(0.0f);
    output[output_base + static_cast<std::size_t>(lane_id * 2 + 1)] = float_to_bf16(0.0f);
    if constexpr (HeadDim == 128) {
      const int dim_even = (lane_id * 2) + kDimBlock;
      output[output_base + static_cast<std::size_t>(dim_even)] = float_to_bf16(0.0f);
      output[output_base + static_cast<std::size_t>(dim_even + 1)] = float_to_bf16(0.0f);
    }
    return;
  }

  const std::size_t query_base =
      ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
        static_cast<std::size_t>(q_head_index)) *
       static_cast<std::size_t>(HeadDim));

  const int dim_even0 = lane_id * 2;
  const float inv_freq0 = rope_inv_freq(lane_id, HeadDim, rope_theta);
  float q_c0 = 0.0f;
  float q_s0 = 0.0f;
  sincos_pair(static_cast<float>(query_position) * inv_freq0, q_c0, q_s0);
  const float q_in_even0 = bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even0)]);
  const float q_in_odd0 =
      bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even0 + 1)]);
  const float q_even0 = (q_in_even0 * q_c0) - (q_in_odd0 * q_s0);
  const float q_odd0 = (q_in_even0 * q_s0) + (q_in_odd0 * q_c0);
  float acc_even0 = 0.0f;
  float acc_odd0 = 0.0f;

  float q_even1 = 0.0f;
  float q_odd1 = 0.0f;
  float acc_even1 = 0.0f;
  float acc_odd1 = 0.0f;
  float inv_freq1 = 0.0f;
  if constexpr (HeadDim == 128) {
    const int dim_even1 = dim_even0 + kDimBlock;
    inv_freq1 = rope_inv_freq(lane_id + kWarpSize, HeadDim, rope_theta);
    float q_c1 = 0.0f;
    float q_s1 = 0.0f;
    sincos_pair(static_cast<float>(query_position) * inv_freq1, q_c1, q_s1);
    const float q_in_even1 = bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even1)]);
    const float q_in_odd1 =
        bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even1 + 1)]);
    q_even1 = (q_in_even1 * q_c1) - (q_in_odd1 * q_s1);
    q_odd1 = (q_in_even1 * q_s1) + (q_in_odd1 * q_c1);
  }

  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (int run_index = run_begin; run_index < run_end; ++run_index) {
    const int base_page = run_base_pages[run_index];
    const int page_count = run_page_counts[run_index];
    const int logical_start = run_logical_starts[run_index];
    const int last_page_len = run_last_page_lens[run_index];

    for (int page_offset = 0; page_offset < page_count; ++page_offset) {
      const int page_index = base_page + page_offset;
      const int logical_page = logical_start + page_offset;
      const int tokens_in_page = (page_offset == page_count - 1) ? last_page_len : PageSize;
      const std::size_t page_head_base =
          (static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads)) +
          static_cast<std::size_t>(kv_head_index);
      const std::size_t key_scale_base = page_head_base * static_cast<std::size_t>(kDimBlocks);

      float rope_cos0 = 0.0f;
      float rope_sin0 = 0.0f;
      float rope_delta_cos0 = 0.0f;
      float rope_delta_sin0 = 0.0f;
      float rope_cos1 = 0.0f;
      float rope_sin1 = 0.0f;
      float rope_delta_cos1 = 0.0f;
      float rope_delta_sin1 = 0.0f;
      if constexpr (!KeysAreRotated) {
        const float page_start = static_cast<float>(logical_page * PageSize);
        sincos_pair(page_start * inv_freq0, rope_cos0, rope_sin0);
        sincos_pair(inv_freq0, rope_delta_cos0, rope_delta_sin0);
        if constexpr (HeadDim == 128) {
          sincos_pair(page_start * inv_freq1, rope_cos1, rope_sin1);
          sincos_pair(inv_freq1, rope_delta_cos1, rope_delta_sin1);
        }
      }

      for (int token_index = 0; token_index < tokens_in_page; ++token_index) {
        const std::size_t token_base = ((page_head_base * static_cast<std::size_t>(PageSize)) +
                                        static_cast<std::size_t>(token_index)) *
                                       static_cast<std::size_t>(HeadDim);
        float partial_dot = 0.0f;

        const float key_even0 =
            load_element<KV>(key_cache, key_scales,
                             token_base + static_cast<std::size_t>(dim_even0), key_scale_base);
        const float key_odd0 =
            load_element<KV>(key_cache, key_scales,
                             token_base + static_cast<std::size_t>(dim_even0 + 1), key_scale_base);
        float rotated_even0 = key_even0;
        float rotated_odd0 = key_odd0;
        if constexpr (!KeysAreRotated) {
          rotated_even0 = (key_even0 * rope_cos0) - (key_odd0 * rope_sin0);
          rotated_odd0 = (key_even0 * rope_sin0) + (key_odd0 * rope_cos0);
        }
        partial_dot += (q_even0 * rotated_even0) + (q_odd0 * rotated_odd0);

        if constexpr (HeadDim == 128) {
          const int dim_even1 = dim_even0 + kDimBlock;
          const float key_even1 = load_element<KV>(key_cache, key_scales,
                                                   token_base + static_cast<std::size_t>(dim_even1),
                                                   key_scale_base + 1u);
          const float key_odd1 = load_element<KV>(
              key_cache, key_scales, token_base + static_cast<std::size_t>(dim_even1 + 1),
              key_scale_base + 1u);
          float rotated_even1 = key_even1;
          float rotated_odd1 = key_odd1;
          if constexpr (!KeysAreRotated) {
            rotated_even1 = (key_even1 * rope_cos1) - (key_odd1 * rope_sin1);
            rotated_odd1 = (key_even1 * rope_sin1) + (key_odd1 * rope_cos1);
          }
          partial_dot += (q_even1 * rotated_even1) + (q_odd1 * rotated_odd1);
        }

        const float logit = warp_sum(partial_dot) * softmax_scale;
        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane_id == 0) {
          const float new_m = fmaxf(local_m, logit);
          alpha = (local_l == 0.0f) ? 0.0f : expf(local_m - new_m);
          beta = expf(logit - new_m);
          local_l = (alpha * local_l) + beta;
          local_m = new_m;
        }
        alpha = __shfl_sync(0xFFFFFFFFu, alpha, 0);
        beta = __shfl_sync(0xFFFFFFFFu, beta, 0);
        local_m = __shfl_sync(0xFFFFFFFFu, local_m, 0);
        local_l = __shfl_sync(0xFFFFFFFFu, local_l, 0);

        const std::size_t value_scale_base =
            (((page_head_base * static_cast<std::size_t>(PageSize)) +
              static_cast<std::size_t>(token_index)) *
             static_cast<std::size_t>(kDimBlocks));
        const float value_even0 =
            load_element<KV>(value_cache, value_scales,
                             token_base + static_cast<std::size_t>(dim_even0), value_scale_base);
        const float value_odd0 = load_element<KV>(
            value_cache, value_scales, token_base + static_cast<std::size_t>(dim_even0 + 1),
            value_scale_base);
        acc_even0 = (alpha * acc_even0) + (beta * value_even0);
        acc_odd0 = (alpha * acc_odd0) + (beta * value_odd0);

        if constexpr (HeadDim == 128) {
          const int dim_even1 = dim_even0 + kDimBlock;
          const float value_even1 = load_element<KV>(
              value_cache, value_scales, token_base + static_cast<std::size_t>(dim_even1),
              value_scale_base + 1u);
          const float value_odd1 = load_element<KV>(
              value_cache, value_scales, token_base + static_cast<std::size_t>(dim_even1 + 1),
              value_scale_base + 1u);
          acc_even1 = (alpha * acc_even1) + (beta * value_even1);
          acc_odd1 = (alpha * acc_odd1) + (beta * value_odd1);
        }

        if constexpr (!KeysAreRotated) {
          const float next_cos0 = (rope_cos0 * rope_delta_cos0) - (rope_sin0 * rope_delta_sin0);
          const float next_sin0 = (rope_sin0 * rope_delta_cos0) + (rope_cos0 * rope_delta_sin0);
          rope_cos0 = next_cos0;
          rope_sin0 = next_sin0;
          if constexpr (HeadDim == 128) {
            const float next_cos1 = (rope_cos1 * rope_delta_cos1) - (rope_sin1 * rope_delta_sin1);
            const float next_sin1 = (rope_sin1 * rope_delta_cos1) + (rope_cos1 * rope_delta_sin1);
            rope_cos1 = next_cos1;
            rope_sin1 = next_sin1;
          }
        }
      }
    }
  }

  const float inv_l = (local_l == 0.0f) ? 0.0f : (1.0f / local_l);
  output[output_base + static_cast<std::size_t>(dim_even0)] = float_to_bf16(acc_even0 * inv_l);
  output[output_base + static_cast<std::size_t>(dim_even0 + 1)] = float_to_bf16(acc_odd0 * inv_l);
  if constexpr (HeadDim == 128) {
    const int dim_even1 = dim_even0 + kDimBlock;
    output[output_base + static_cast<std::size_t>(dim_even1)] = float_to_bf16(acc_even1 * inv_l);
    output[output_base + static_cast<std::size_t>(dim_even1 + 1)] = float_to_bf16(acc_odd1 * inv_l);
  }
}

template <int HeadDim, int PageSize, int GroupTile, KvFormat KV, bool KeysAreRotated,
          int ClusterSize>
__global__ __launch_bounds__(kClusterThreadsPerBlock) void clustered_page_decode_cluster_kernel(
    const __nv_bfloat16* query, const void* key_cache, const void* value_cache,
    const float* key_scales, const float* value_scales, const int32_t* run_base_pages,
    const int32_t* run_page_counts, const int32_t* run_logical_starts,
    const int32_t* run_last_page_lens, const int32_t* request_run_offsets, const int32_t* seq_lens,
    __nv_bfloat16* output, const int num_q_heads, const int num_kv_heads, const float softmax_scale,
    const float rope_theta) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(PageSize == 16 || PageSize == 32);
  static_assert(GroupTile == 1 || GroupTile == 2 || GroupTile == 4 || GroupTile == 8);
  static_assert(ClusterSize >= 1 && ClusterSize <= kMaxClusterSize);

  constexpr int kDimBlocks = HeadDim / kDimBlock;
  constexpr int kPairCount = HeadDim / 2;
  using StorageT = typename KvStorage<KV>::type;

  const cg::cluster_group cluster = cg::this_cluster();
  const int block_rank = static_cast<int>(cluster.block_rank());
  const int group_size = num_q_heads / num_kv_heads;
  const int q_tile_count = (group_size + GroupTile - 1) / GroupTile;
  const int task_index = static_cast<int>(blockIdx.x) / ClusterSize;
  const int task_stride = num_kv_heads * q_tile_count;
  const int request_index = task_index / task_stride;
  const int task_offset = task_index % task_stride;
  const int kv_head_index = task_offset / q_tile_count;
  const int q_tile_index = task_offset % q_tile_count;
  const int q_head_tile_base = q_tile_index * GroupTile;
  const int remaining_heads = group_size - q_head_tile_base;
  const int tile_heads = remaining_heads < GroupTile ? remaining_heads : GroupTile;
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
  const int query_position = seq_lens[request_index] - 1;
  const int run_begin = request_run_offsets[request_index];
  const int run_end = request_run_offsets[request_index + 1];

  __shared__ __nv_bfloat16 shared_key[PageSize][HeadDim];
  __shared__ __nv_bfloat16 shared_value[PageSize][HeadDim];
  __shared__ float shared_key_scales[kDimBlocks];
  __shared__ float shared_value_scales[PageSize][kDimBlocks];
  __shared__ float shared_m[GroupTile];
  __shared__ float shared_l[GroupTile];
  __shared__ float shared_acc[GroupTile][HeadDim];
  __shared__ float merge_weights[GroupTile][ClusterSize];
  __shared__ float merge_denominator[GroupTile];

  if (query_position < 0 || run_begin == run_end) {
    if (block_rank == 0) {
      for (int linear_index = static_cast<int>(threadIdx.x); linear_index < tile_heads * HeadDim;
           linear_index += static_cast<int>(blockDim.x)) {
        const int group_head_offset = linear_index / HeadDim;
        const int dim_index = linear_index % HeadDim;
        const int q_head_index =
            (kv_head_index * group_size) + q_head_tile_base + group_head_offset;
        const std::size_t output_index =
            ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
              static_cast<std::size_t>(q_head_index)) *
             static_cast<std::size_t>(HeadDim)) +
            static_cast<std::size_t>(dim_index);
        output[output_index] = float_to_bf16(0.0f);
      }
    }
    return;
  }

  const int dim_even0 = lane_id * 2;
  const float inv_freq0 = rope_inv_freq(lane_id, HeadDim, rope_theta);
  float inv_freq1 = 0.0f;
  if constexpr (HeadDim == 128) {
    inv_freq1 = rope_inv_freq(lane_id + kWarpSize, HeadDim, rope_theta);
  }

  const bool active_warp = warp_id < tile_heads;

  float q_even0 = 0.0f;
  float q_odd0 = 0.0f;
  float q_even1 = 0.0f;
  float q_odd1 = 0.0f;
  float acc_even0 = 0.0f;
  float acc_odd0 = 0.0f;
  float acc_even1 = 0.0f;
  float acc_odd1 = 0.0f;
  float local_m = -INFINITY;
  float local_l = 0.0f;
  int q_head_index = 0;

  if (active_warp) {
    q_head_index = (kv_head_index * group_size) + q_head_tile_base + warp_id;
    const std::size_t query_base =
        ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
          static_cast<std::size_t>(q_head_index)) *
         static_cast<std::size_t>(HeadDim));

    float q_c0 = 0.0f;
    float q_s0 = 0.0f;
    sincos_pair(static_cast<float>(query_position) * inv_freq0, q_c0, q_s0);
    const float q_in_even0 =
        bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even0)]);
    const float q_in_odd0 =
        bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even0 + 1)]);
    q_even0 = (q_in_even0 * q_c0) - (q_in_odd0 * q_s0);
    q_odd0 = (q_in_even0 * q_s0) + (q_in_odd0 * q_c0);

    if constexpr (HeadDim == 128) {
      const int dim_even1 = dim_even0 + kDimBlock;
      float q_c1 = 0.0f;
      float q_s1 = 0.0f;
      sincos_pair(static_cast<float>(query_position) * inv_freq1, q_c1, q_s1);
      const float q_in_even1 =
          bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even1)]);
      const float q_in_odd1 =
          bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even1 + 1)]);
      q_even1 = (q_in_even1 * q_c1) - (q_in_odd1 * q_s1);
      q_odd1 = (q_in_even1 * q_s1) + (q_in_odd1 * q_c1);
    }
  }

  for (int run_index = run_begin; run_index < run_end; ++run_index) {
      const int base_page = run_base_pages[run_index];
      const int page_count = run_page_counts[run_index];
      const int logical_start = run_logical_starts[run_index];
      const int last_page_len = run_last_page_lens[run_index];
      const int page_chunk_begin = (page_count * block_rank) / ClusterSize;
      const int page_chunk_end = (page_count * (block_rank + 1)) / ClusterSize;

      for (int page_offset = page_chunk_begin; page_offset < page_chunk_end; ++page_offset) {
        const int page_index = base_page + page_offset;
        const int logical_page = logical_start + page_offset;
        const int tokens_in_page = (page_offset == page_count - 1) ? last_page_len : PageSize;
        const std::size_t page_head_index =
            (static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads)) +
            static_cast<std::size_t>(kv_head_index);
        const auto* key_page =
            reinterpret_cast<const StorageT*>(key_cache) +
            (page_head_index * static_cast<std::size_t>(PageSize) * static_cast<std::size_t>(HeadDim));
        const auto* value_page =
            reinterpret_cast<const StorageT*>(value_cache) +
            (page_head_index * static_cast<std::size_t>(PageSize) * static_cast<std::size_t>(HeadDim));
        const float* key_scale_page = key_scales == nullptr
                                          ? nullptr
                                          : key_scales + (page_head_index * kDimBlocks);
        const float* value_scale_page =
            value_scales == nullptr
                ? nullptr
                : value_scales + (page_head_index * static_cast<std::size_t>(PageSize) * kDimBlocks);

        if constexpr (KV != KvFormat::kBFloat16) {
          if (threadIdx.x < kDimBlocks) {
            shared_key_scales[threadIdx.x] = key_scale_page[threadIdx.x];
          }
          for (int scale_index = static_cast<int>(threadIdx.x);
               scale_index < tokens_in_page * kDimBlocks;
               scale_index += static_cast<int>(blockDim.x)) {
            shared_value_scales[scale_index / kDimBlocks][scale_index % kDimBlocks] =
                value_scale_page[scale_index];
          }
        }
        __syncthreads();

        for (int pair_linear = static_cast<int>(threadIdx.x);
             pair_linear < tokens_in_page * kPairCount;
             pair_linear += static_cast<int>(blockDim.x)) {
          const int token_index = pair_linear / kPairCount;
          const int pair_index = pair_linear % kPairCount;
          const int dim_even = pair_index * 2;
          const int dim_block = dim_even / kDimBlock;
          const std::size_t element_base =
              (static_cast<std::size_t>(token_index) * static_cast<std::size_t>(HeadDim)) +
              static_cast<std::size_t>(dim_even);

          float key_even = 0.0f;
          float key_odd = 0.0f;
          if constexpr (KV == KvFormat::kBFloat16) {
            key_even = load_storage_element<KV>(key_page, element_base, 1.0f);
            key_odd = load_storage_element<KV>(key_page, element_base + 1u, 1.0f);
          } else {
            const float key_scale = shared_key_scales[dim_block];
            key_even = load_storage_element<KV>(key_page, element_base, key_scale);
            key_odd = load_storage_element<KV>(key_page, element_base + 1u, key_scale);
          }

          if constexpr (!KeysAreRotated) {
            apply_llama_rope_pair(key_even, key_odd, pair_index, HeadDim,
                                  (logical_page * PageSize) + token_index, rope_theta, key_even,
                                  key_odd);
          }

          float value_even = 0.0f;
          float value_odd = 0.0f;
          if constexpr (KV == KvFormat::kBFloat16) {
            value_even = load_storage_element<KV>(value_page, element_base, 1.0f);
            value_odd = load_storage_element<KV>(value_page, element_base + 1u, 1.0f);
          } else {
            const float value_scale = shared_value_scales[token_index][dim_block];
            value_even = load_storage_element<KV>(value_page, element_base, value_scale);
            value_odd = load_storage_element<KV>(value_page, element_base + 1u, value_scale);
          }

          shared_key[token_index][dim_even] = float_to_bf16(key_even);
          shared_key[token_index][dim_even + 1] = float_to_bf16(key_odd);
          shared_value[token_index][dim_even] = float_to_bf16(value_even);
          shared_value[token_index][dim_even + 1] = float_to_bf16(value_odd);
        }
        __syncthreads();

        if (active_warp) {
          for (int token_index = 0; token_index < tokens_in_page; ++token_index) {
            float partial_dot = 0.0f;

            const float key_even0 =
                bf16_to_float(shared_key[token_index][dim_even0]);
            const float key_odd0 =
                bf16_to_float(shared_key[token_index][dim_even0 + 1]);
            partial_dot += (q_even0 * key_even0) + (q_odd0 * key_odd0);

            if constexpr (HeadDim == 128) {
              const int dim_even1 = dim_even0 + kDimBlock;
              const float key_even1 =
                  bf16_to_float(shared_key[token_index][dim_even1]);
              const float key_odd1 =
                  bf16_to_float(shared_key[token_index][dim_even1 + 1]);
              partial_dot += (q_even1 * key_even1) + (q_odd1 * key_odd1);
            }

            const float logit = warp_sum(partial_dot) * softmax_scale;
            float alpha = 0.0f;
            float beta = 0.0f;
            if (lane_id == 0) {
              const float new_m = fmaxf(local_m, logit);
              alpha = (local_l == 0.0f) ? 0.0f : expf(local_m - new_m);
              beta = expf(logit - new_m);
              local_l = (alpha * local_l) + beta;
              local_m = new_m;
            }
            alpha = __shfl_sync(0xFFFFFFFFu, alpha, 0);
            beta = __shfl_sync(0xFFFFFFFFu, beta, 0);
            local_m = __shfl_sync(0xFFFFFFFFu, local_m, 0);
            local_l = __shfl_sync(0xFFFFFFFFu, local_l, 0);

            const float value_even0 =
                bf16_to_float(shared_value[token_index][dim_even0]);
            const float value_odd0 =
                bf16_to_float(shared_value[token_index][dim_even0 + 1]);
            acc_even0 = (alpha * acc_even0) + (beta * value_even0);
            acc_odd0 = (alpha * acc_odd0) + (beta * value_odd0);

            if constexpr (HeadDim == 128) {
              const int dim_even1 = dim_even0 + kDimBlock;
              const float value_even1 =
                  bf16_to_float(shared_value[token_index][dim_even1]);
              const float value_odd1 =
                  bf16_to_float(shared_value[token_index][dim_even1 + 1]);
              acc_even1 = (alpha * acc_even1) + (beta * value_even1);
              acc_odd1 = (alpha * acc_odd1) + (beta * value_odd1);
            }
          }
        }
        __syncthreads();
      }
    }

  if (active_warp && lane_id == 0) {
    shared_m[warp_id] = local_m;
    shared_l[warp_id] = local_l;
  }
  if (active_warp) {
    shared_acc[warp_id][dim_even0] = acc_even0;
    shared_acc[warp_id][dim_even0 + 1] = acc_odd0;
    if constexpr (HeadDim == 128) {
      const int dim_even1 = dim_even0 + kDimBlock;
      shared_acc[warp_id][dim_even1] = acc_even1;
      shared_acc[warp_id][dim_even1 + 1] = acc_odd1;
    }
  }
  __syncthreads();
  cluster.sync();

  if (active_warp && block_rank == 0) {
    if constexpr (ClusterSize == 1) {
      const float inv_l = (shared_l[warp_id] == 0.0f) ? 0.0f : (1.0f / shared_l[warp_id]);
      const std::size_t output_base =
          ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
            static_cast<std::size_t>(q_head_index)) *
           static_cast<std::size_t>(HeadDim));
      output[output_base + static_cast<std::size_t>(dim_even0)] =
          float_to_bf16(shared_acc[warp_id][dim_even0] * inv_l);
      output[output_base + static_cast<std::size_t>(dim_even0 + 1)] =
          float_to_bf16(shared_acc[warp_id][dim_even0 + 1] * inv_l);
      if constexpr (HeadDim == 128) {
        const int dim_even1 = dim_even0 + kDimBlock;
        output[output_base + static_cast<std::size_t>(dim_even1)] =
            float_to_bf16(shared_acc[warp_id][dim_even1] * inv_l);
        output[output_base + static_cast<std::size_t>(dim_even1 + 1)] =
            float_to_bf16(shared_acc[warp_id][dim_even1 + 1] * inv_l);
      }
    } else {
      if (lane_id == 0) {
        float final_m = -INFINITY;
        for (int rank = 0; rank < ClusterSize; ++rank) {
          const float* remote_m = cluster.map_shared_rank(&shared_m[warp_id], rank);
          final_m = fmaxf(final_m, *remote_m);
        }

        float denominator = 0.0f;
        for (int rank = 0; rank < ClusterSize; ++rank) {
          const float* remote_m = cluster.map_shared_rank(&shared_m[warp_id], rank);
          const float* remote_l = cluster.map_shared_rank(&shared_l[warp_id], rank);
          const float weight = (*remote_l == 0.0f) ? 0.0f : expf(*remote_m - final_m);
          merge_weights[warp_id][rank] = weight;
          denominator += weight * (*remote_l);
        }
        merge_denominator[warp_id] = denominator;
      }
      __syncwarp();

      float merged_even = 0.0f;
      float merged_odd = 0.0f;
      for (int rank = 0; rank < ClusterSize; ++rank) {
        const float* remote_acc = cluster.map_shared_rank(&shared_acc[warp_id][0], rank);
        const float weight = merge_weights[warp_id][rank];
        merged_even += weight * remote_acc[dim_even0];
        merged_odd += weight * remote_acc[dim_even0 + 1];
      }

      const float denominator = merge_denominator[warp_id];
      const float inv_denominator = (denominator == 0.0f) ? 0.0f : (1.0f / denominator);
      const std::size_t output_base =
          ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
            static_cast<std::size_t>(q_head_index)) *
           static_cast<std::size_t>(HeadDim));
      output[output_base + static_cast<std::size_t>(dim_even0)] =
          float_to_bf16(merged_even * inv_denominator);
      output[output_base + static_cast<std::size_t>(dim_even0 + 1)] =
          float_to_bf16(merged_odd * inv_denominator);

      if constexpr (HeadDim == 128) {
        const int dim_even1 = dim_even0 + kDimBlock;
        float merged_even1 = 0.0f;
        float merged_odd1 = 0.0f;
        for (int rank = 0; rank < ClusterSize; ++rank) {
          const float* remote_acc = cluster.map_shared_rank(&shared_acc[warp_id][0], rank);
          const float weight = merge_weights[warp_id][rank];
          merged_even1 += weight * remote_acc[dim_even1];
          merged_odd1 += weight * remote_acc[dim_even1 + 1];
        }
        output[output_base + static_cast<std::size_t>(dim_even1)] =
            float_to_bf16(merged_even1 * inv_denominator);
        output[output_base + static_cast<std::size_t>(dim_even1 + 1)] =
            float_to_bf16(merged_odd1 * inv_denominator);
      }
    }
  }
  cluster.sync();
}

template <int HeadDim, int PageSize, int ThreadsPerBlock, KvFormat KV>
void launch_direct_variant(const __nv_bfloat16* query, const void* key_cache,
                           const void* value_cache, const float* key_scales,
                           const float* value_scales, const int32_t* run_base_pages,
                           const int32_t* run_page_counts, const int32_t* run_logical_starts,
                           const int32_t* run_last_page_lens, const int32_t* request_run_offsets,
                           const int32_t* seq_lens, __nv_bfloat16* output, const int batch,
                           const int num_q_heads, const int num_kv_heads, const bool keys_are_rotated,
                           const float softmax_scale, const float rope_theta, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(batch * num_kv_heads), 1u, 1u);
  const dim3 block(static_cast<unsigned>(ThreadsPerBlock), 1u, 1u);
  if (keys_are_rotated) {
    direct_page_decode_kernel<ThreadsPerBlock, HeadDim, PageSize, KV, true>
        <<<grid, block, 0, stream>>>(query, key_cache, value_cache, key_scales, value_scales,
                                     run_base_pages, run_page_counts, run_logical_starts,
                                     run_last_page_lens, request_run_offsets, seq_lens, output,
                                     num_q_heads, num_kv_heads, softmax_scale, rope_theta);
  } else {
    direct_page_decode_kernel<ThreadsPerBlock, HeadDim, PageSize, KV, false>
        <<<grid, block, 0, stream>>>(query, key_cache, value_cache, key_scales, value_scales,
                                     run_base_pages, run_page_counts, run_logical_starts,
                                     run_last_page_lens, request_run_offsets, seq_lens, output,
                                     num_q_heads, num_kv_heads, softmax_scale, rope_theta);
  }
}

inline int select_direct_threads(const int group_size) {
  if (group_size <= 1) {
    return 32;
  }
  if (group_size <= 2) {
    return 64;
  }
  if (group_size <= 4) {
    return 128;
  }
  return 256;
}

template <int HeadDim, int PageSize, int ThreadsPerBlock>
void dispatch_direct_format(const __nv_bfloat16* query, const void* key_cache,
                            const void* value_cache, const float* key_scales,
                            const float* value_scales, const int32_t* run_base_pages,
                            const int32_t* run_page_counts, const int32_t* run_logical_starts,
                            const int32_t* run_last_page_lens, const int32_t* request_run_offsets,
                            const int32_t* seq_lens, __nv_bfloat16* output, const int batch,
                            const int num_q_heads, const int num_kv_heads, const int kv_format,
                            const bool keys_are_rotated, const float softmax_scale,
                            const float rope_theta, cudaStream_t stream) {
  switch (static_cast<KvFormat>(kv_format)) {
  case KvFormat::kBFloat16:
    launch_direct_variant<HeadDim, PageSize, ThreadsPerBlock, KvFormat::kBFloat16>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case KvFormat::kFp8E4M3:
    launch_direct_variant<HeadDim, PageSize, ThreadsPerBlock, KvFormat::kFp8E4M3>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case KvFormat::kInt8:
    launch_direct_variant<HeadDim, PageSize, ThreadsPerBlock, KvFormat::kInt8>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  throw std::invalid_argument("unsupported kv_format");
}

template <int HeadDim, int PageSize>
void dispatch_direct_threads(const __nv_bfloat16* query, const void* key_cache,
                             const void* value_cache, const float* key_scales,
                             const float* value_scales, const int32_t* run_base_pages,
                             const int32_t* run_page_counts, const int32_t* run_logical_starts,
                             const int32_t* run_last_page_lens, const int32_t* request_run_offsets,
                             const int32_t* seq_lens, __nv_bfloat16* output, const int batch,
                             const int num_q_heads, const int num_kv_heads, const int kv_format,
                             const bool keys_are_rotated, const float softmax_scale,
                             const float rope_theta, cudaStream_t stream) {
  const int group_size = num_q_heads / num_kv_heads;
  switch (select_direct_threads(group_size)) {
  case 32:
    dispatch_direct_format<HeadDim, PageSize, 32>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 64:
    dispatch_direct_format<HeadDim, PageSize, 64>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 128:
    dispatch_direct_format<HeadDim, PageSize, 128>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 256:
    dispatch_direct_format<HeadDim, PageSize, 256>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  throw std::invalid_argument("unsupported direct thread selection");
}

void launch_direct_decode(const __nv_bfloat16* query, const void* key_cache,
                          const void* value_cache, const float* key_scales,
                          const float* value_scales, const int32_t* run_base_pages,
                          const int32_t* run_page_counts, const int32_t* run_logical_starts,
                          const int32_t* run_last_page_lens, const int32_t* request_run_offsets,
                          const int32_t* seq_lens, __nv_bfloat16* output, const int batch,
                          const int num_q_heads, const int num_kv_heads, const int head_dim,
                          const int page_size, const int kv_format, const bool keys_are_rotated,
                          const float softmax_scale, const float rope_theta, cudaStream_t stream) {
  if (head_dim == 64 && page_size == 16) {
    dispatch_direct_threads<64, 16>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  if (head_dim == 64 && page_size == 32) {
    dispatch_direct_threads<64, 32>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  if (head_dim == 128 && page_size == 16) {
    dispatch_direct_threads<128, 16>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  dispatch_direct_threads<128, 32>(
      query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
      run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
      num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
}

template <int HeadDim, int PageSize, int GroupTile, KvFormat KV, int ClusterSize>
void launch_clustered_variant(const __nv_bfloat16* query, const void* key_cache,
                              const void* value_cache, const float* key_scales,
                              const float* value_scales, const int32_t* run_base_pages,
                              const int32_t* run_page_counts, const int32_t* run_logical_starts,
                              const int32_t* run_last_page_lens,
                              const int32_t* request_run_offsets, const int32_t* seq_lens,
                              __nv_bfloat16* output, const int batch, const int num_q_heads,
                              const int num_kv_heads, const bool keys_are_rotated,
                              const float softmax_scale, const float rope_theta,
                              cudaStream_t stream) {
  const int group_size = num_q_heads / num_kv_heads;
  const int q_tile_count = (group_size + GroupTile - 1) / GroupTile;

  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = static_cast<unsigned>(ClusterSize);
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim =
      dim3(static_cast<unsigned>(batch * num_kv_heads * q_tile_count * ClusterSize), 1u, 1u);
  config.blockDim = dim3(kClusterThreadsPerBlock, 1u, 1u);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 1;

  void* args[] = {
      const_cast<__nv_bfloat16**>(&query),
      const_cast<void**>(&key_cache),
      const_cast<void**>(&value_cache),
      const_cast<float**>(&key_scales),
      const_cast<float**>(&value_scales),
      const_cast<int32_t**>(&run_base_pages),
      const_cast<int32_t**>(&run_page_counts),
      const_cast<int32_t**>(&run_logical_starts),
      const_cast<int32_t**>(&run_last_page_lens),
      const_cast<int32_t**>(&request_run_offsets),
      const_cast<int32_t**>(&seq_lens),
      &output,
      const_cast<int*>(&num_q_heads),
      const_cast<int*>(&num_kv_heads),
      const_cast<float*>(&softmax_scale),
      const_cast<float*>(&rope_theta),
  };

  if (keys_are_rotated) {
    FK_CUDA_CHECK(cudaLaunchKernelExC(
        &config,
        reinterpret_cast<const void*>(
            clustered_page_decode_cluster_kernel<HeadDim, PageSize, GroupTile, KV, true,
                                                 ClusterSize>),
        args));
  } else {
    FK_CUDA_CHECK(cudaLaunchKernelExC(
        &config,
        reinterpret_cast<const void*>(
            clustered_page_decode_cluster_kernel<HeadDim, PageSize, GroupTile, KV, false,
                                                 ClusterSize>),
        args));
  }
}

template <int HeadDim, int PageSize, int GroupTile, int ClusterSize>
void dispatch_cluster_format(const __nv_bfloat16* query, const void* key_cache,
                             const void* value_cache, const float* key_scales,
                             const float* value_scales, const int32_t* run_base_pages,
                             const int32_t* run_page_counts, const int32_t* run_logical_starts,
                             const int32_t* run_last_page_lens,
                             const int32_t* request_run_offsets, const int32_t* seq_lens,
                             __nv_bfloat16* output, const int batch, const int num_q_heads,
                             const int num_kv_heads, const int kv_format,
                             const bool keys_are_rotated, const float softmax_scale,
                             const float rope_theta, cudaStream_t stream) {
  switch (static_cast<KvFormat>(kv_format)) {
  case KvFormat::kBFloat16:
    launch_clustered_variant<HeadDim, PageSize, GroupTile, KvFormat::kBFloat16, ClusterSize>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case KvFormat::kFp8E4M3:
    launch_clustered_variant<HeadDim, PageSize, GroupTile, KvFormat::kFp8E4M3, ClusterSize>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case KvFormat::kInt8:
    launch_clustered_variant<HeadDim, PageSize, GroupTile, KvFormat::kInt8, ClusterSize>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  throw std::invalid_argument("unsupported kv_format");
}

template <int HeadDim, int PageSize, int GroupTile>
void dispatch_cluster_size(const __nv_bfloat16* query, const void* key_cache,
                           const void* value_cache, const float* key_scales,
                           const float* value_scales, const int32_t* run_base_pages,
                           const int32_t* run_page_counts, const int32_t* run_logical_starts,
                           const int32_t* run_last_page_lens,
                           const int32_t* request_run_offsets, const int32_t* seq_lens,
                           __nv_bfloat16* output, const int batch, const int num_q_heads,
                           const int num_kv_heads, const int cluster_size, const int kv_format,
                           const bool keys_are_rotated, const float softmax_scale,
                           const float rope_theta, cudaStream_t stream) {
  switch (cluster_size) {
  case 1:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 1>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 2:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 2>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 3:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 3>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 4:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 4>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 5:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 5>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 6:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 6>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 7:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 7>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  case 8:
    dispatch_cluster_format<HeadDim, PageSize, GroupTile, 8>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, kv_format, keys_are_rotated, softmax_scale, rope_theta, stream);
    return;
  }
  throw std::invalid_argument("cluster_size must be in [1, 8]");
}

template <int HeadDim, int PageSize>
void dispatch_cluster_group_tile(const __nv_bfloat16* query, const void* key_cache,
                                 const void* value_cache, const float* key_scales,
                                 const float* value_scales, const int32_t* run_base_pages,
                                 const int32_t* run_page_counts,
                                 const int32_t* run_logical_starts,
                                 const int32_t* run_last_page_lens,
                                 const int32_t* request_run_offsets, const int32_t* seq_lens,
                                 __nv_bfloat16* output, const int batch, const int num_q_heads,
                                 const int num_kv_heads, const int group_tile,
                                 const int cluster_size, const int kv_format,
                                 const bool keys_are_rotated, const float softmax_scale,
                                 const float rope_theta, cudaStream_t stream) {
  switch (group_tile) {
  case 1:
    dispatch_cluster_size<HeadDim, PageSize, 1>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, cluster_size, kv_format, keys_are_rotated, softmax_scale,
        rope_theta, stream);
    return;
  case 2:
    dispatch_cluster_size<HeadDim, PageSize, 2>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, cluster_size, kv_format, keys_are_rotated, softmax_scale,
        rope_theta, stream);
    return;
  case 4:
    dispatch_cluster_size<HeadDim, PageSize, 4>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, cluster_size, kv_format, keys_are_rotated, softmax_scale,
        rope_theta, stream);
    return;
  case 8:
    dispatch_cluster_size<HeadDim, PageSize, 8>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, cluster_size, kv_format, keys_are_rotated, softmax_scale,
        rope_theta, stream);
    return;
  }
  throw std::invalid_argument("group_tile must be one of {1, 2, 4, 8}");
}

void launch_clustered_decode(const __nv_bfloat16* query, const void* key_cache,
                             const void* value_cache, const float* key_scales,
                             const float* value_scales, const int32_t* run_base_pages,
                             const int32_t* run_page_counts, const int32_t* run_logical_starts,
                             const int32_t* run_last_page_lens,
                             const int32_t* request_run_offsets, const int32_t* seq_lens,
                             __nv_bfloat16* output, const int batch, const int num_q_heads,
                             const int num_kv_heads, const int head_dim, const int page_size,
                             const int group_tile, const int cluster_size, const int kv_format,
                             const bool keys_are_rotated, const float softmax_scale,
                             const float rope_theta, cudaStream_t stream) {
  if (head_dim == 64 && page_size == 16) {
    dispatch_cluster_group_tile<64, 16>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, group_tile, cluster_size, kv_format, keys_are_rotated,
        softmax_scale, rope_theta, stream);
    return;
  }
  if (head_dim == 64 && page_size == 32) {
    dispatch_cluster_group_tile<64, 32>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, group_tile, cluster_size, kv_format, keys_are_rotated,
        softmax_scale, rope_theta, stream);
    return;
  }
  if (head_dim == 128 && page_size == 16) {
    dispatch_cluster_group_tile<128, 16>(
        query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
        run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
        num_q_heads, num_kv_heads, group_tile, cluster_size, kv_format, keys_are_rotated,
        softmax_scale, rope_theta, stream);
    return;
  }
  dispatch_cluster_group_tile<128, 32>(
      query, key_cache, value_cache, key_scales, value_scales, run_base_pages, run_page_counts,
      run_logical_starts, run_last_page_lens, request_run_offsets, seq_lens, output, batch,
      num_q_heads, num_kv_heads, group_tile, cluster_size, kv_format, keys_are_rotated,
      softmax_scale, rope_theta, stream);
}

} // namespace

void clustered_page_decode_forward(
    std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
    std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr,
    std::uintptr_t run_base_pages_ptr, std::uintptr_t run_page_counts_ptr,
    std::uintptr_t run_logical_starts_ptr, std::uintptr_t run_last_page_lens_ptr,
    std::uintptr_t request_run_offsets_ptr, std::uintptr_t seq_lens_ptr, std::uintptr_t output_ptr,
    int batch, int num_runs, int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    int group_tile, int use_clustered_kernel, int cluster_size, int kv_format, int keys_are_rotated,
    float softmax_scale, float rope_theta, std::uintptr_t stream_ptr) {
  if (batch <= 0 || num_q_heads <= 0 || num_kv_heads <= 0) {
    throw std::invalid_argument("clustered_page_decode_forward received empty inputs");
  }
  if (num_runs < 0) {
    throw std::invalid_argument("num_runs must be non-negative");
  }
  if (head_dim != 64 && head_dim != 128) {
    throw std::invalid_argument("clustered_page_decode_forward only supports head_dim 64 or 128");
  }
  if (page_size != 16 && page_size != 32) {
    throw std::invalid_argument("clustered_page_decode_forward only supports page_size 16 or 32");
  }
  if (num_q_heads % num_kv_heads != 0) {
    throw std::invalid_argument("num_q_heads must be divisible by num_kv_heads");
  }
  if (cluster_size < 1 || cluster_size > kMaxClusterSize) {
    throw std::invalid_argument("cluster_size must be in [1, 8]");
  }
  if (group_tile != 1 && group_tile != 2 && group_tile != 4 && group_tile != 8) {
    throw std::invalid_argument("group_tile must be one of {1, 2, 4, 8}");
  }
  const int group_size = num_q_heads / num_kv_heads;
  if (group_size < 1) {
    throw std::invalid_argument("group_size must be positive");
  }
  if (use_clustered_kernel == 0 && group_size > kMaxDirectGroupSize) {
    throw std::invalid_argument(
        "legacy direct decode only supports group_size in [1, 8]; use clustered decode");
  }
  if (kv_format != static_cast<int>(KvFormat::kBFloat16) &&
      kv_format != static_cast<int>(KvFormat::kFp8E4M3) &&
      kv_format != static_cast<int>(KvFormat::kInt8)) {
    throw std::invalid_argument("unsupported kv_format");
  }

  auto* query = reinterpret_cast<const __nv_bfloat16*>(query_ptr);
  auto* output = reinterpret_cast<__nv_bfloat16*>(output_ptr);
  auto* run_base_pages = reinterpret_cast<const int32_t*>(run_base_pages_ptr);
  auto* run_page_counts = reinterpret_cast<const int32_t*>(run_page_counts_ptr);
  auto* run_logical_starts = reinterpret_cast<const int32_t*>(run_logical_starts_ptr);
  auto* run_last_page_lens = reinterpret_cast<const int32_t*>(run_last_page_lens_ptr);
  auto* request_run_offsets = reinterpret_cast<const int32_t*>(request_run_offsets_ptr);
  auto* seq_lens = reinterpret_cast<const int32_t*>(seq_lens_ptr);
  const float* key_scales = reinterpret_cast<const float*>(key_scales_ptr);
  const float* value_scales = reinterpret_cast<const float*>(value_scales_ptr);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const void* key_cache = reinterpret_cast<const void*>(key_ptr);
  const void* value_cache = reinterpret_cast<const void*>(value_ptr);
  const bool key_cache_is_rotated = keys_are_rotated != 0;

  if (use_clustered_kernel == 0) {
    launch_direct_decode(query, key_cache, value_cache, key_scales, value_scales, run_base_pages,
                         run_page_counts, run_logical_starts, run_last_page_lens,
                         request_run_offsets, seq_lens, output, batch, num_q_heads, num_kv_heads,
                         head_dim, page_size, kv_format, key_cache_is_rotated, softmax_scale,
                         rope_theta, stream);
    FK_CUDA_CHECK(cudaGetLastError());
    return;
  }

  launch_clustered_decode(query, key_cache, value_cache, key_scales, value_scales, run_base_pages,
                          run_page_counts, run_logical_starts, run_last_page_lens,
                          request_run_offsets, seq_lens, output, batch, num_q_heads,
                          num_kv_heads, head_dim, page_size, group_tile, cluster_size, kv_format,
                          key_cache_is_rotated, softmax_scale, rope_theta, stream);
  FK_CUDA_CHECK(cudaGetLastError());
}

} // namespace fast_kernels::clustered_page_decode
