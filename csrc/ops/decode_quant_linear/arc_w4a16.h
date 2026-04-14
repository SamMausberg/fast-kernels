#pragma once

#include <cstddef>
#include <cstdint>

namespace fast_kernels::decode_quant_linear {

void pack_arc_w4a16_packets(std::uintptr_t q_ptr, std::uintptr_t alpha_ptr, std::uintptr_t beta_ptr,
                            std::uintptr_t packets_ptr, int n, int k, int group_size,
                            int packet_stride_bytes, std::uintptr_t stream_ptr);

void compute_arc_w4a16_group_sums(std::uintptr_t activations_ptr, std::uintptr_t sums_ptr,
                                  int batch, int k, int group_size, std::uintptr_t stream_ptr);

void arc_w4a16_forward(std::uintptr_t activations_ptr, std::uintptr_t packets_ptr,
                       std::uintptr_t group_sums_ptr, std::uintptr_t output_ptr, int batch, int n,
                       int k, int group_size, int packet_stride_bytes, std::uintptr_t stream_ptr);

void arc_w4a16_forward_split_k(std::uintptr_t activations_ptr, std::uintptr_t packets_ptr,
                               std::uintptr_t group_sums_ptr, std::uintptr_t partials_ptr,
                               int batch, int n, int k, int group_size, int packet_stride_bytes,
                               int split_k_slices, std::uintptr_t stream_ptr);

void reduce_arc_w4a16_split_k_partials(std::uintptr_t partials_ptr, std::uintptr_t output_ptr,
                                       int batch, int n, int split_k_slices,
                                       std::uintptr_t stream_ptr);

void dequant_w4a16_to_fp16(std::uintptr_t q_ptr, std::uintptr_t alpha_ptr, std::uintptr_t beta_ptr,
                           std::uintptr_t output_ptr, int n, int k, int group_size,
                           std::uintptr_t stream_ptr);

void dequant_arc_w4a16_packets_to_fp16(std::uintptr_t packets_ptr, std::uintptr_t output_ptr, int n,
                                       int k, int group_size, int packet_stride_bytes,
                                       std::uintptr_t stream_ptr);

void cublaslt_fp16_with_weight(std::uintptr_t activations_ptr, std::uintptr_t weight_ptr,
                               std::uintptr_t output_ptr, std::uintptr_t workspace_ptr,
                               std::size_t workspace_bytes, int batch, int n, int k,
                               std::uintptr_t stream_ptr);

void cublaslt_fp16_after_packet_dequant(std::uintptr_t activations_ptr, std::uintptr_t packets_ptr,
                                        std::uintptr_t output_ptr, std::uintptr_t weight_ptr,
                                        std::uintptr_t workspace_ptr, std::size_t workspace_bytes,
                                        int batch, int n, int k, int group_size,
                                        int packet_stride_bytes, std::uintptr_t stream_ptr);

void cublaslt_fp16_after_dequant(std::uintptr_t activations_ptr, std::uintptr_t q_ptr,
                                 std::uintptr_t alpha_ptr, std::uintptr_t beta_ptr,
                                 std::uintptr_t output_ptr, std::uintptr_t weight_ptr,
                                 std::uintptr_t workspace_ptr, std::size_t workspace_bytes,
                                 int batch, int n, int k, int group_size,
                                 std::uintptr_t stream_ptr);

} // namespace fast_kernels::decode_quant_linear
