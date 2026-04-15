#pragma once

#include <cstddef>
#include <cstdint>

namespace fast_kernels::clustered_page_decode {

enum class KvFormat : int {
  kBFloat16 = 0,
  kFp8E4M3 = 1,
  kInt8 = 2,
};

void clustered_page_decode_forward(
    std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
    std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr,
    std::uintptr_t run_base_pages_ptr, std::uintptr_t run_page_counts_ptr,
    std::uintptr_t run_logical_starts_ptr, std::uintptr_t run_last_page_lens_ptr,
    std::uintptr_t request_run_offsets_ptr, std::uintptr_t seq_lens_ptr, std::uintptr_t output_ptr,
    int batch, int num_runs, int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    int group_tile, int use_clustered_kernel, int cluster_size, int kv_format,
    int keys_are_rotated, float softmax_scale, float rope_theta, std::uintptr_t stream_ptr);

} // namespace fast_kernels::clustered_page_decode
