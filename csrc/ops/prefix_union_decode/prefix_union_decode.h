#pragma once

#include <cstddef>
#include <cstdint>

namespace fast_kernels::prefix_union_decode {

void prefix_union_decode_forward(
    std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
    std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr, std::uintptr_t tasks_ptr,
    std::uintptr_t shared_pages_ptr, std::uintptr_t tail_pages_ptr, std::uintptr_t consumers_ptr,
    std::uintptr_t scheduler_counter_ptr, std::uintptr_t output_ptr, int num_tasks, int num_pages,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size, int cluster_size,
    int kv_format, int keys_are_rotated, float softmax_scale, float rope_theta,
    std::uintptr_t stream_ptr);

} // namespace fast_kernels::prefix_union_decode
