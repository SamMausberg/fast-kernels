#include "ops/clustered_page_decode/clustered_page_decode.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace fast_kernels::clustered_page_decode {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxGroupSize = 8;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxPairsPerLane = 2;
constexpr int kDimBlock = 64;
constexpr int kMaxClusterSize = 8;

#define FK_CUDA_CHECK(expr)                                                                  \
  do {                                                                                       \
    const cudaError_t error__ = (expr);                                                      \
    if (error__ != cudaSuccess) {                                                            \
      throw std::runtime_error(cudaGetErrorString(error__));                                 \
    }                                                                                        \
  } while (false)

__device__ inline float bf16_to_float(const __nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ inline __nv_bfloat16 float_to_bf16(const float value) {
  return __float2bfloat16_rn(value);
}

__device__ inline float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ inline float load_k_element(
    const void* key_ptr,
    const float* key_scales,
    const int kv_format,
    const int page_index,
    const int kv_head_index,
    const int token_index,
    const int dim_index,
    const int num_kv_heads,
    const int page_size,
    const int head_dim) {
  const std::size_t flat_index =
      (((static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads) +
         static_cast<std::size_t>(kv_head_index)) *
            static_cast<std::size_t>(page_size) +
        static_cast<std::size_t>(token_index)) *
           static_cast<std::size_t>(head_dim)) +
      static_cast<std::size_t>(dim_index);
  if (kv_format == static_cast<int>(KvFormat::kBFloat16)) {
    const auto* data = reinterpret_cast<const __nv_bfloat16*>(key_ptr);
    return bf16_to_float(data[flat_index]);
  }

  const int dim_block_index = dim_index / kDimBlock;
  const std::size_t scale_index =
      ((static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads) +
        static_cast<std::size_t>(kv_head_index)) *
           static_cast<std::size_t>(head_dim / kDimBlock)) +
      static_cast<std::size_t>(dim_block_index);
  const float scale = key_scales[scale_index];
  if (kv_format == static_cast<int>(KvFormat::kFp8E4M3)) {
    const auto* data = reinterpret_cast<const __nv_fp8_e4m3*>(key_ptr);
    return static_cast<float>(data[flat_index]) * scale;
  }

  const auto* data = reinterpret_cast<const int8_t*>(key_ptr);
  return static_cast<float>(data[flat_index]) * scale;
}

__device__ inline float load_v_element(
    const void* value_ptr,
    const float* value_scales,
    const int kv_format,
    const int page_index,
    const int kv_head_index,
    const int token_index,
    const int dim_index,
    const int num_kv_heads,
    const int page_size,
    const int head_dim) {
  const std::size_t flat_index =
      (((static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads) +
         static_cast<std::size_t>(kv_head_index)) *
            static_cast<std::size_t>(page_size) +
        static_cast<std::size_t>(token_index)) *
           static_cast<std::size_t>(head_dim)) +
      static_cast<std::size_t>(dim_index);
  if (kv_format == static_cast<int>(KvFormat::kBFloat16)) {
    const auto* data = reinterpret_cast<const __nv_bfloat16*>(value_ptr);
    return bf16_to_float(data[flat_index]);
  }

  const int dim_block_index = dim_index / kDimBlock;
  const std::size_t scale_index =
      ((((static_cast<std::size_t>(page_index) * static_cast<std::size_t>(num_kv_heads) +
          static_cast<std::size_t>(kv_head_index)) *
             static_cast<std::size_t>(page_size)) +
        static_cast<std::size_t>(token_index)) *
           static_cast<std::size_t>(head_dim / kDimBlock)) +
      static_cast<std::size_t>(dim_block_index);
  const float scale = value_scales[scale_index];
  if (kv_format == static_cast<int>(KvFormat::kFp8E4M3)) {
    const auto* data = reinterpret_cast<const __nv_fp8_e4m3*>(value_ptr);
    return static_cast<float>(data[flat_index]) * scale;
  }

  const auto* data = reinterpret_cast<const int8_t*>(value_ptr);
  return static_cast<float>(data[flat_index]) * scale;
}

__device__ inline void apply_llama_rope_pair(
    const float in_even,
    const float in_odd,
    const int pair_index,
    const int head_dim,
    const int position,
    const float rope_theta,
    float& out_even,
    float& out_odd) {
  const float inv_freq = powf(rope_theta, -2.0f * static_cast<float>(pair_index) /
                                              static_cast<float>(head_dim));
  const float angle = static_cast<float>(position) * inv_freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  out_even = (in_even * c) - (in_odd * s);
  out_odd = (in_even * s) + (in_odd * c);
}

__global__ __launch_bounds__(kThreadsPerBlock) void clustered_page_decode_kernel(
    const __nv_bfloat16* query,
    const void* key_cache,
    const void* value_cache,
    const float* key_scales,
    const float* value_scales,
    const int32_t* run_base_pages,
    const int32_t* run_page_counts,
    const int32_t* run_logical_starts,
    const int32_t* run_last_page_lens,
    const int32_t* request_run_offsets,
    const int32_t* seq_lens,
    __nv_bfloat16* output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int page_size,
    const int kv_format,
    const float softmax_scale,
    const float rope_theta) {
  const cg::cluster_group cluster = cg::this_cluster();
  const int cluster_size = static_cast<int>(cluster.dim_blocks().x);
  const int task_index = static_cast<int>(blockIdx.x) / cluster_size;
  const int request_index = task_index / num_kv_heads;
  const int kv_head_index = task_index % num_kv_heads;
  const int group_size = num_q_heads / num_kv_heads;
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
  const int block_rank = static_cast<int>(cluster.block_rank());
  const bool active_warp = warp_id < group_size;

  __shared__ float shared_m[kMaxGroupSize];
  __shared__ float shared_l[kMaxGroupSize];
  __shared__ float shared_acc[kMaxGroupSize][kMaxHeadDim];
  __shared__ float merge_weights[kMaxGroupSize][kMaxClusterSize];
  __shared__ float merge_denominator[kMaxGroupSize];

  int q_head_index = 0;
  int pair_indices[kMaxPairsPerLane] = {0, 0};
  float q_even[kMaxPairsPerLane] = {0.0f, 0.0f};
  float q_odd[kMaxPairsPerLane] = {0.0f, 0.0f};
  float acc_even[kMaxPairsPerLane] = {0.0f, 0.0f};
  float acc_odd[kMaxPairsPerLane] = {0.0f, 0.0f};
  int pair_count = 0;

  float local_m = -INFINITY;
  float local_l = 0.0f;
  if (active_warp) {
    q_head_index = kv_head_index * group_size + warp_id;
    const int query_position = seq_lens[request_index] - 1;
    const std::size_t query_base =
        ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
          static_cast<std::size_t>(q_head_index)) *
         static_cast<std::size_t>(head_dim));

    for (int pair_index = lane_id; pair_index < head_dim / 2; pair_index += kWarpSize) {
      const int dim_even = pair_index * 2;
      const int dim_odd = dim_even + 1;
      const float in_even =
          bf16_to_float(query[query_base + static_cast<std::size_t>(dim_even)]);
      const float in_odd =
          bf16_to_float(query[query_base + static_cast<std::size_t>(dim_odd)]);
      float out_even = 0.0f;
      float out_odd = 0.0f;
      apply_llama_rope_pair(
          in_even, in_odd, pair_index, head_dim, query_position, rope_theta, out_even, out_odd);
      pair_indices[pair_count] = pair_index;
      q_even[pair_count] = out_even;
      q_odd[pair_count] = out_odd;
      pair_count += 1;
    }

    const int run_begin = request_run_offsets[request_index];
    const int run_end = request_run_offsets[request_index + 1];

    for (int run_index = run_begin; run_index < run_end; ++run_index) {
      const int base_page = run_base_pages[run_index];
      const int page_count = run_page_counts[run_index];
      const int logical_start = run_logical_starts[run_index];
      const int last_page_len = run_last_page_lens[run_index];

      for (int page_offset = block_rank; page_offset < page_count; page_offset += cluster_size) {
        const int page_index = base_page + page_offset;
        const int logical_page = logical_start + page_offset;
        const int tokens_in_page =
            (page_offset == page_count - 1) ? last_page_len : page_size;
        float rope_cos[kMaxPairsPerLane] = {0.0f, 0.0f};
        float rope_sin[kMaxPairsPerLane] = {0.0f, 0.0f};
        float rope_delta_cos[kMaxPairsPerLane] = {0.0f, 0.0f};
        float rope_delta_sin[kMaxPairsPerLane] = {0.0f, 0.0f};
        for (int slot = 0; slot < pair_count; ++slot) {
          const float inv_freq =
              powf(rope_theta, -2.0f * static_cast<float>(pair_indices[slot]) /
                                    static_cast<float>(head_dim));
          const float start_angle =
              static_cast<float>(logical_page * page_size) * inv_freq;
          rope_cos[slot] = cosf(start_angle);
          rope_sin[slot] = sinf(start_angle);
          rope_delta_cos[slot] = cosf(inv_freq);
          rope_delta_sin[slot] = sinf(inv_freq);
        }

        for (int token_index = 0; token_index < tokens_in_page; ++token_index) {
          float partial_dot = 0.0f;
          for (int slot = 0; slot < pair_count; ++slot) {
            const int pair_index = pair_indices[slot];
            const int dim_even = pair_index * 2;
            const int dim_odd = dim_even + 1;
            const float key_even = load_k_element(
                key_cache,
                key_scales,
                kv_format,
                page_index,
                kv_head_index,
                token_index,
                dim_even,
                num_kv_heads,
                page_size,
                head_dim);
            const float key_odd = load_k_element(
                key_cache,
                key_scales,
                kv_format,
                page_index,
                kv_head_index,
                token_index,
                dim_odd,
                num_kv_heads,
                page_size,
                head_dim);
            const float rotated_even =
                (key_even * rope_cos[slot]) - (key_odd * rope_sin[slot]);
            const float rotated_odd =
                (key_even * rope_sin[slot]) + (key_odd * rope_cos[slot]);
            partial_dot += (q_even[slot] * rotated_even) + (q_odd[slot] * rotated_odd);
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

          for (int slot = 0; slot < pair_count; ++slot) {
            const int pair_index = pair_indices[slot];
            const int dim_even = pair_index * 2;
            const int dim_odd = dim_even + 1;
            const float value_even = load_v_element(
                value_cache,
                value_scales,
                kv_format,
                page_index,
                kv_head_index,
                token_index,
                dim_even,
                num_kv_heads,
                page_size,
                head_dim);
            const float value_odd = load_v_element(
                value_cache,
                value_scales,
                kv_format,
                page_index,
                kv_head_index,
                token_index,
                dim_odd,
                num_kv_heads,
                page_size,
                head_dim);
            acc_even[slot] = (alpha * acc_even[slot]) + (beta * value_even);
            acc_odd[slot] = (alpha * acc_odd[slot]) + (beta * value_odd);

            const float next_cos =
                (rope_cos[slot] * rope_delta_cos[slot]) -
                (rope_sin[slot] * rope_delta_sin[slot]);
            const float next_sin =
                (rope_sin[slot] * rope_delta_cos[slot]) +
                (rope_cos[slot] * rope_delta_sin[slot]);
            rope_cos[slot] = next_cos;
            rope_sin[slot] = next_sin;
          }
        }
      }
    }
  }

  if (active_warp && lane_id == 0) {
    shared_m[warp_id] = local_m;
    shared_l[warp_id] = local_l;
  }
  if (active_warp) {
    for (int slot = 0; slot < pair_count; ++slot) {
      const int pair_index = pair_indices[slot];
      shared_acc[warp_id][pair_index * 2] = acc_even[slot];
      shared_acc[warp_id][pair_index * 2 + 1] = acc_odd[slot];
    }
  }
  __syncthreads();
  cluster.sync();

  if (active_warp && block_rank == 0) {
    if (lane_id == 0) {
      float final_m = -INFINITY;
      for (int rank = 0; rank < cluster_size; ++rank) {
        const float* remote_m = cluster.map_shared_rank(&shared_m[warp_id], rank);
        final_m = fmaxf(final_m, *remote_m);
      }
      float denominator = 0.0f;
      for (int rank = 0; rank < cluster_size; ++rank) {
        const float* remote_m = cluster.map_shared_rank(&shared_m[warp_id], rank);
        const float* remote_l = cluster.map_shared_rank(&shared_l[warp_id], rank);
        const float weight = (*remote_l == 0.0f) ? 0.0f : expf(*remote_m - final_m);
        merge_weights[warp_id][rank] = weight;
        denominator += weight * (*remote_l);
      }
      merge_denominator[warp_id] = denominator;
    }
    __syncwarp();

    const float denominator = merge_denominator[warp_id];
    for (int pair_index = lane_id; pair_index < head_dim / 2; pair_index += kWarpSize) {
      float merged_even = 0.0f;
      float merged_odd = 0.0f;
      for (int rank = 0; rank < cluster_size; ++rank) {
        const float* remote_acc = cluster.map_shared_rank(&shared_acc[warp_id][0], rank);
        const float weight = merge_weights[warp_id][rank];
        merged_even += weight * remote_acc[pair_index * 2];
        merged_odd += weight * remote_acc[pair_index * 2 + 1];
      }

      const std::size_t output_base =
          ((static_cast<std::size_t>(request_index) * static_cast<std::size_t>(num_q_heads) +
            static_cast<std::size_t>(q_head_index)) *
           static_cast<std::size_t>(head_dim));
      output[output_base + static_cast<std::size_t>(pair_index * 2)] =
          float_to_bf16(merged_even / denominator);
      output[output_base + static_cast<std::size_t>(pair_index * 2 + 1)] =
          float_to_bf16(merged_odd / denominator);
    }
  }
  cluster.sync();
}

}  // namespace

void clustered_page_decode_forward(
    std::uintptr_t query_ptr,
    std::uintptr_t key_ptr,
    std::uintptr_t value_ptr,
    std::uintptr_t key_scales_ptr,
    std::uintptr_t value_scales_ptr,
    std::uintptr_t run_base_pages_ptr,
    std::uintptr_t run_page_counts_ptr,
    std::uintptr_t run_logical_starts_ptr,
    std::uintptr_t run_last_page_lens_ptr,
    std::uintptr_t request_run_offsets_ptr,
    std::uintptr_t seq_lens_ptr,
    std::uintptr_t output_ptr,
    int batch,
    int num_runs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int cluster_size,
    int kv_format,
    float softmax_scale,
    float rope_theta,
    std::uintptr_t stream_ptr) {
  if (batch <= 0 || num_runs <= 0 || num_q_heads <= 0 || num_kv_heads <= 0) {
    throw std::invalid_argument("clustered_page_decode_forward received empty inputs");
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

  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = static_cast<unsigned>(cluster_size);
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(
      static_cast<unsigned>(batch * num_kv_heads * cluster_size), 1u, 1u);
  config.blockDim = dim3(kThreadsPerBlock, 1u, 1u);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 1;

  void* args[] = {
      &query,
      reinterpret_cast<void*>(&key_ptr),
      reinterpret_cast<void*>(&value_ptr),
      const_cast<float**>(&key_scales),
      const_cast<float**>(&value_scales),
      const_cast<int32_t**>(&run_base_pages),
      const_cast<int32_t**>(&run_page_counts),
      const_cast<int32_t**>(&run_logical_starts),
      const_cast<int32_t**>(&run_last_page_lens),
      const_cast<int32_t**>(&request_run_offsets),
      const_cast<int32_t**>(&seq_lens),
      &output,
      &num_q_heads,
      &num_kv_heads,
      &head_dim,
      &page_size,
      &kv_format,
      &softmax_scale,
      &rope_theta,
  };

  const void* key_cache = reinterpret_cast<const void*>(key_ptr);
  const void* value_cache = reinterpret_cast<const void*>(value_ptr);
  args[1] = const_cast<void**>(&key_cache);
  args[2] = const_cast<void**>(&value_cache);

  FK_CUDA_CHECK(cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(clustered_page_decode_kernel), args));
  FK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fast_kernels::clustered_page_decode
