#include <pybind11/pybind11.h>

#include <string>
#include <stdexcept>

#include "ops/decode_quant_linear/arc_w4a16.h"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native build metadata for fast-kernels.";
  m.def("build_info", []() {
    py::dict info;
    info["project_version"] = std::string(FK_PROJECT_VERSION);
    info["compiled_with_cuda"] = static_cast<bool>(FK_COMPILED_WITH_CUDA);
    info["cxx_compiler_id"] = std::string(FK_CXX_COMPILER_ID);
    info["cxx_compiler_version"] = std::string(FK_CXX_COMPILER_VERSION);
    info["cuda_compiler_version"] = std::string(FK_CUDA_COMPILER_VERSION);
    info["cuda_architectures"] = std::string(FK_CUDA_ARCHITECTURES);
    return info;
  });

#if FK_COMPILED_WITH_CUDA
  m.def(
      "compute_arc_w4a16_group_sums",
      [](std::uintptr_t activations_ptr,
         std::uintptr_t sums_ptr,
         int batch,
         int k,
         int group_size,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::compute_arc_w4a16_group_sums(
            activations_ptr, sums_ptr, batch, k, group_size, stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("sums_ptr"),
      py::arg("batch"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("stream_ptr"));
  m.def(
      "pack_arc_w4a16_packets",
      [](std::uintptr_t q_ptr,
         std::uintptr_t alpha_ptr,
         std::uintptr_t beta_ptr,
         std::uintptr_t packets_ptr,
         int n,
         int k,
         int group_size,
         int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::pack_arc_w4a16_packets(
            q_ptr, alpha_ptr, beta_ptr, packets_ptr, n, k, group_size, packet_stride_bytes, stream_ptr);
      },
      py::arg("q_ptr"),
      py::arg("alpha_ptr"),
      py::arg("beta_ptr"),
      py::arg("packets_ptr"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "arc_w4a16_forward",
      [](std::uintptr_t activations_ptr,
         std::uintptr_t packets_ptr,
         std::uintptr_t group_sums_ptr,
         std::uintptr_t output_ptr,
         int batch,
         int n,
         int k,
         int group_size,
         int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::arc_w4a16_forward(
            activations_ptr,
            packets_ptr,
            group_sums_ptr,
            output_ptr,
            batch,
            n,
            k,
            group_size,
            packet_stride_bytes,
            stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("packets_ptr"),
      py::arg("group_sums_ptr"),
      py::arg("output_ptr"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "arc_w4a16_forward_split_k",
      [](std::uintptr_t activations_ptr,
         std::uintptr_t packets_ptr,
         std::uintptr_t group_sums_ptr,
         std::uintptr_t partials_ptr,
         int batch,
         int n,
         int k,
         int group_size,
         int packet_stride_bytes,
         int split_k_slices,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::arc_w4a16_forward_split_k(
            activations_ptr,
            packets_ptr,
            group_sums_ptr,
            partials_ptr,
            batch,
            n,
            k,
            group_size,
            packet_stride_bytes,
            split_k_slices,
            stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("packets_ptr"),
      py::arg("group_sums_ptr"),
      py::arg("partials_ptr"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("packet_stride_bytes"),
      py::arg("split_k_slices"),
      py::arg("stream_ptr"));
  m.def(
      "reduce_arc_w4a16_split_k_partials",
      [](std::uintptr_t partials_ptr,
         std::uintptr_t output_ptr,
         int batch,
         int n,
         int split_k_slices,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::reduce_arc_w4a16_split_k_partials(
            partials_ptr, output_ptr, batch, n, split_k_slices, stream_ptr);
      },
      py::arg("partials_ptr"),
      py::arg("output_ptr"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("split_k_slices"),
      py::arg("stream_ptr"));
  m.def(
      "dequant_w4a16_to_fp16",
      [](std::uintptr_t q_ptr,
         std::uintptr_t alpha_ptr,
         std::uintptr_t beta_ptr,
         std::uintptr_t output_ptr,
         int n,
         int k,
         int group_size,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::dequant_w4a16_to_fp16(
            q_ptr, alpha_ptr, beta_ptr, output_ptr, n, k, group_size, stream_ptr);
      },
      py::arg("q_ptr"),
      py::arg("alpha_ptr"),
      py::arg("beta_ptr"),
      py::arg("output_ptr"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("stream_ptr"));
  m.def(
      "dequant_arc_w4a16_packets_to_fp16",
      [](std::uintptr_t packets_ptr,
         std::uintptr_t output_ptr,
         int n,
         int k,
         int group_size,
         int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::dequant_arc_w4a16_packets_to_fp16(
            packets_ptr, output_ptr, n, k, group_size, packet_stride_bytes, stream_ptr);
      },
      py::arg("packets_ptr"),
      py::arg("output_ptr"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_with_weight",
      [](std::uintptr_t activations_ptr,
         std::uintptr_t weight_ptr,
         std::uintptr_t output_ptr,
         std::uintptr_t workspace_ptr,
         std::size_t workspace_bytes,
         int batch,
         int n,
         int k,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_with_weight(
            activations_ptr,
            weight_ptr,
            output_ptr,
            workspace_ptr,
            workspace_bytes,
            batch,
            n,
            k,
            stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("weight_ptr"),
      py::arg("output_ptr"),
      py::arg("workspace_ptr"),
      py::arg("workspace_bytes"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("k"),
      py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_after_packet_dequant",
      [](std::uintptr_t activations_ptr,
         std::uintptr_t packets_ptr,
         std::uintptr_t output_ptr,
         std::uintptr_t weight_ptr,
         std::uintptr_t workspace_ptr,
         std::size_t workspace_bytes,
         int batch,
         int n,
         int k,
         int group_size,
         int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_after_packet_dequant(
            activations_ptr,
            packets_ptr,
            output_ptr,
            weight_ptr,
            workspace_ptr,
            workspace_bytes,
            batch,
            n,
            k,
            group_size,
            packet_stride_bytes,
            stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("packets_ptr"),
      py::arg("output_ptr"),
      py::arg("weight_ptr"),
      py::arg("workspace_ptr"),
      py::arg("workspace_bytes"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_after_dequant",
      [](std::uintptr_t activations_ptr,
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
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_after_dequant(
            activations_ptr,
            q_ptr,
            alpha_ptr,
            beta_ptr,
            output_ptr,
            weight_ptr,
            workspace_ptr,
            workspace_bytes,
            batch,
            n,
            k,
            group_size,
            stream_ptr);
      },
      py::arg("activations_ptr"),
      py::arg("q_ptr"),
      py::arg("alpha_ptr"),
      py::arg("beta_ptr"),
      py::arg("output_ptr"),
      py::arg("weight_ptr"),
      py::arg("workspace_ptr"),
      py::arg("workspace_bytes"),
      py::arg("batch"),
      py::arg("n"),
      py::arg("k"),
      py::arg("group_size"),
      py::arg("stream_ptr"));
#else
  const auto unavailable = []() {
    throw std::runtime_error("fast-kernels was built without CUDA support");
  };
  m.def(
      "compute_arc_w4a16_group_sums",
      [unavailable](std::uintptr_t, std::uintptr_t, int, int, int, std::uintptr_t) { unavailable(); });
  m.def(
      "pack_arc_w4a16_packets",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) { unavailable(); });
  m.def(
      "arc_w4a16_forward",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    int,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) {
        unavailable();
      });
  m.def(
      "arc_w4a16_forward_split_k",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) { unavailable(); });
  m.def(
      "reduce_arc_w4a16_split_k_partials",
      [unavailable](std::uintptr_t, std::uintptr_t, int, int, int, std::uintptr_t) { unavailable(); });
  m.def(
      "dequant_w4a16_to_fp16",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    int,
                    int,
                    int,
                    std::uintptr_t) {
        unavailable();
      });
  m.def(
      "dequant_arc_w4a16_packets_to_fp16",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) {
        unavailable();
      });
  m.def(
      "cublaslt_fp16_with_weight",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::size_t,
                    int,
                    int,
                    int,
                    std::uintptr_t) {
        unavailable();
      });
  m.def(
      "cublaslt_fp16_after_packet_dequant",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::size_t,
                    int,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) {
        unavailable();
      });
  m.def(
      "cublaslt_fp16_after_dequant",
      [unavailable](std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::uintptr_t,
                    std::size_t,
                    int,
                    int,
                    int,
                    int,
                    std::uintptr_t) { unavailable(); });
#endif
}
