#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "ops/clustered_page_decode/clustered_page_decode.h"
#include "ops/decode_quant_linear/arc_w4a16.h"
#include "ops/prefix_union_decode/prefix_union_decode.h"
#if FK_COMPILED_WITH_CUDA
#include "ops/rdkng/explicit_sketch.h"
#endif

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
  py::class_<fast_kernels::rdkng::StepStats>(m, "RDKNGStepStats")
      .def(py::init<>())
      .def_readonly("initial_residual_norm",
                    &fast_kernels::rdkng::StepStats::initial_residual_norm)
      .def_readonly("final_residual_norm",
                    &fast_kernels::rdkng::StepStats::final_residual_norm)
      .def_readonly("cg_steps_taken", &fast_kernels::rdkng::StepStats::cg_steps_taken)
      .def_readonly("basis_refreshed", &fast_kernels::rdkng::StepStats::basis_refreshed);

  py::class_<fast_kernels::rdkng::ExplicitSketchSolver>(m, "RDKNGExplicitSketchSolverHandle")
      .def(py::init<int, int, int, int>(), py::arg("n"), py::arg("s"), py::arg("r"),
           py::arg("max_cg"))
      .def_property_readonly("n", &fast_kernels::rdkng::ExplicitSketchSolver::n)
      .def_property_readonly("s", &fast_kernels::rdkng::ExplicitSketchSolver::s)
      .def_property_readonly("r", &fast_kernels::rdkng::ExplicitSketchSolver::r)
      .def_property_readonly("max_cg", &fast_kernels::rdkng::ExplicitSketchSolver::max_cg)
      .def("workspace_bytes", &fast_kernels::rdkng::ExplicitSketchSolver::workspace_bytes)
      .def(
          "step",
          [](fast_kernels::rdkng::ExplicitSketchSolver& solver, std::uintptr_t y_ptr,
             std::uintptr_t grad_ptr, std::uintptr_t basis_in_ptr, std::uintptr_t basis_out_ptr,
             float lambda, int cg_iters, float tol, std::uintptr_t out_v_ptr,
             std::uintptr_t diag_ema_ptr, float diag_ema_beta, int basis_replace_col,
             float basis_append_threshold, std::uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            fast_kernels::rdkng::StepStats stats;
            solver.step(y_ptr, grad_ptr, basis_in_ptr, basis_out_ptr, lambda, cg_iters, tol,
                        out_v_ptr, diag_ema_ptr, diag_ema_beta, basis_replace_col,
                        basis_append_threshold, stream_ptr, &stats);
            return stats;
          },
          py::arg("y_ptr"), py::arg("grad_ptr"), py::arg("basis_in_ptr") = 0,
          py::arg("basis_out_ptr") = 0, py::arg("lambda_"), py::arg("cg_iters"),
          py::arg("tol"), py::arg("out_v_ptr"), py::arg("diag_ema_ptr") = 0,
          py::arg("diag_ema_beta") = 0.95f, py::arg("basis_replace_col") = -1,
          py::arg("basis_append_threshold") = 1.0e-4f, py::arg("stream_ptr") = 0);

  m.def(
      "clustered_page_decode_forward",
      [](std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
         std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr,
         std::uintptr_t run_base_pages_ptr, std::uintptr_t run_page_counts_ptr,
         std::uintptr_t run_logical_starts_ptr, std::uintptr_t run_last_page_lens_ptr,
         std::uintptr_t request_run_offsets_ptr, std::uintptr_t seq_lens_ptr,
         std::uintptr_t output_ptr, int batch, int num_runs, int num_q_heads, int num_kv_heads,
         int head_dim, int page_size, int group_tile, int use_clustered_kernel, int cluster_size,
         int kv_format, int keys_are_rotated, float softmax_scale, float rope_theta,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::clustered_page_decode::clustered_page_decode_forward(
            query_ptr, key_ptr, value_ptr, key_scales_ptr, value_scales_ptr, run_base_pages_ptr,
            run_page_counts_ptr, run_logical_starts_ptr, run_last_page_lens_ptr,
            request_run_offsets_ptr, seq_lens_ptr, output_ptr, batch, num_runs, num_q_heads,
            num_kv_heads, head_dim, page_size, group_tile, use_clustered_kernel, cluster_size,
            kv_format, keys_are_rotated, softmax_scale, rope_theta, stream_ptr);
      },
      py::arg("query_ptr"), py::arg("key_ptr"), py::arg("value_ptr"), py::arg("key_scales_ptr"),
      py::arg("value_scales_ptr"), py::arg("run_base_pages_ptr"), py::arg("run_page_counts_ptr"),
      py::arg("run_logical_starts_ptr"), py::arg("run_last_page_lens_ptr"),
      py::arg("request_run_offsets_ptr"), py::arg("seq_lens_ptr"), py::arg("output_ptr"),
      py::arg("batch"), py::arg("num_runs"), py::arg("num_q_heads"), py::arg("num_kv_heads"),
      py::arg("head_dim"), py::arg("page_size"), py::arg("group_tile"),
      py::arg("use_clustered_kernel"), py::arg("cluster_size"), py::arg("kv_format"),
      py::arg("keys_are_rotated"), py::arg("softmax_scale"), py::arg("rope_theta"),
      py::arg("stream_ptr"));
  m.def(
      "prefix_union_decode_forward",
      [](std::uintptr_t query_ptr, std::uintptr_t key_ptr, std::uintptr_t value_ptr,
         std::uintptr_t key_scales_ptr, std::uintptr_t value_scales_ptr,
         std::uintptr_t tasks_ptr, std::uintptr_t shared_pages_ptr,
         std::uintptr_t tail_pages_ptr, std::uintptr_t consumers_ptr,
         std::uintptr_t scheduler_counter_ptr, std::uintptr_t output_ptr, int num_tasks,
         int num_pages, int num_q_heads, int num_kv_heads, int head_dim, int page_size,
         int cluster_size, int kv_format, int keys_are_rotated, float softmax_scale,
         float rope_theta, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::prefix_union_decode::prefix_union_decode_forward(
            query_ptr, key_ptr, value_ptr, key_scales_ptr, value_scales_ptr, tasks_ptr,
            shared_pages_ptr, tail_pages_ptr, consumers_ptr, scheduler_counter_ptr, output_ptr,
            num_tasks, num_pages, num_q_heads, num_kv_heads, head_dim, page_size, cluster_size,
            kv_format, keys_are_rotated, softmax_scale, rope_theta, stream_ptr);
      },
      py::arg("query_ptr"), py::arg("key_ptr"), py::arg("value_ptr"), py::arg("key_scales_ptr"),
      py::arg("value_scales_ptr"), py::arg("tasks_ptr"), py::arg("shared_pages_ptr"),
      py::arg("tail_pages_ptr"), py::arg("consumers_ptr"), py::arg("scheduler_counter_ptr"),
      py::arg("output_ptr"), py::arg("num_tasks"), py::arg("num_pages"),
      py::arg("num_q_heads"), py::arg("num_kv_heads"),
      py::arg("head_dim"), py::arg("page_size"), py::arg("cluster_size"), py::arg("kv_format"),
      py::arg("keys_are_rotated"), py::arg("softmax_scale"), py::arg("rope_theta"),
      py::arg("stream_ptr"));
  m.def(
      "compute_arc_w4a16_group_sums",
      [](std::uintptr_t activations_ptr, std::uintptr_t sums_ptr, int batch, int k, int group_size,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::compute_arc_w4a16_group_sums(
            activations_ptr, sums_ptr, batch, k, group_size, stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("sums_ptr"), py::arg("batch"), py::arg("k"),
      py::arg("group_size"), py::arg("stream_ptr"));
  m.def(
      "pack_arc_w4a16_packets",
      [](std::uintptr_t q_ptr, std::uintptr_t alpha_ptr, std::uintptr_t beta_ptr,
         std::uintptr_t packets_ptr, int n, int k, int group_size, int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::pack_arc_w4a16_packets(q_ptr, alpha_ptr, beta_ptr,
                                                                  packets_ptr, n, k, group_size,
                                                                  packet_stride_bytes, stream_ptr);
      },
      py::arg("q_ptr"), py::arg("alpha_ptr"), py::arg("beta_ptr"), py::arg("packets_ptr"),
      py::arg("n"), py::arg("k"), py::arg("group_size"), py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "arc_w4a16_forward",
      [](std::uintptr_t activations_ptr, std::uintptr_t packets_ptr, std::uintptr_t group_sums_ptr,
         std::uintptr_t output_ptr, int batch, int n, int k, int group_size,
         int packet_stride_bytes, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::arc_w4a16_forward(
            activations_ptr, packets_ptr, group_sums_ptr, output_ptr, batch, n, k, group_size,
            packet_stride_bytes, stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("packets_ptr"), py::arg("group_sums_ptr"),
      py::arg("output_ptr"), py::arg("batch"), py::arg("n"), py::arg("k"), py::arg("group_size"),
      py::arg("packet_stride_bytes"), py::arg("stream_ptr"));
  m.def(
      "arc_w4a16_forward_split_k",
      [](std::uintptr_t activations_ptr, std::uintptr_t packets_ptr, std::uintptr_t group_sums_ptr,
         std::uintptr_t partials_ptr, int batch, int n, int k, int group_size,
         int packet_stride_bytes, int split_k_slices, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::arc_w4a16_forward_split_k(
            activations_ptr, packets_ptr, group_sums_ptr, partials_ptr, batch, n, k, group_size,
            packet_stride_bytes, split_k_slices, stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("packets_ptr"), py::arg("group_sums_ptr"),
      py::arg("partials_ptr"), py::arg("batch"), py::arg("n"), py::arg("k"), py::arg("group_size"),
      py::arg("packet_stride_bytes"), py::arg("split_k_slices"), py::arg("stream_ptr"));
  m.def(
      "reduce_arc_w4a16_split_k_partials",
      [](std::uintptr_t partials_ptr, std::uintptr_t output_ptr, int batch, int n,
         int split_k_slices, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::reduce_arc_w4a16_split_k_partials(
            partials_ptr, output_ptr, batch, n, split_k_slices, stream_ptr);
      },
      py::arg("partials_ptr"), py::arg("output_ptr"), py::arg("batch"), py::arg("n"),
      py::arg("split_k_slices"), py::arg("stream_ptr"));
  m.def(
      "dequant_w4a16_to_fp16",
      [](std::uintptr_t q_ptr, std::uintptr_t alpha_ptr, std::uintptr_t beta_ptr,
         std::uintptr_t output_ptr, int n, int k, int group_size, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::dequant_w4a16_to_fp16(
            q_ptr, alpha_ptr, beta_ptr, output_ptr, n, k, group_size, stream_ptr);
      },
      py::arg("q_ptr"), py::arg("alpha_ptr"), py::arg("beta_ptr"), py::arg("output_ptr"),
      py::arg("n"), py::arg("k"), py::arg("group_size"), py::arg("stream_ptr"));
  m.def(
      "dequant_arc_w4a16_packets_to_fp16",
      [](std::uintptr_t packets_ptr, std::uintptr_t output_ptr, int n, int k, int group_size,
         int packet_stride_bytes, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::dequant_arc_w4a16_packets_to_fp16(
            packets_ptr, output_ptr, n, k, group_size, packet_stride_bytes, stream_ptr);
      },
      py::arg("packets_ptr"), py::arg("output_ptr"), py::arg("n"), py::arg("k"),
      py::arg("group_size"), py::arg("packet_stride_bytes"), py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_with_weight",
      [](std::uintptr_t activations_ptr, std::uintptr_t weight_ptr, std::uintptr_t output_ptr,
         std::uintptr_t workspace_ptr, std::size_t workspace_bytes, int batch, int n, int k,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_with_weight(
            activations_ptr, weight_ptr, output_ptr, workspace_ptr, workspace_bytes, batch, n, k,
            stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("weight_ptr"), py::arg("output_ptr"),
      py::arg("workspace_ptr"), py::arg("workspace_bytes"), py::arg("batch"), py::arg("n"),
      py::arg("k"), py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_after_packet_dequant",
      [](std::uintptr_t activations_ptr, std::uintptr_t packets_ptr, std::uintptr_t output_ptr,
         std::uintptr_t weight_ptr, std::uintptr_t workspace_ptr, std::size_t workspace_bytes,
         int batch, int n, int k, int group_size, int packet_stride_bytes,
         std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_after_packet_dequant(
            activations_ptr, packets_ptr, output_ptr, weight_ptr, workspace_ptr, workspace_bytes,
            batch, n, k, group_size, packet_stride_bytes, stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("packets_ptr"), py::arg("output_ptr"),
      py::arg("weight_ptr"), py::arg("workspace_ptr"), py::arg("workspace_bytes"), py::arg("batch"),
      py::arg("n"), py::arg("k"), py::arg("group_size"), py::arg("packet_stride_bytes"),
      py::arg("stream_ptr"));
  m.def(
      "cublaslt_fp16_after_dequant",
      [](std::uintptr_t activations_ptr, std::uintptr_t q_ptr, std::uintptr_t alpha_ptr,
         std::uintptr_t beta_ptr, std::uintptr_t output_ptr, std::uintptr_t weight_ptr,
         std::uintptr_t workspace_ptr, std::size_t workspace_bytes, int batch, int n, int k,
         int group_size, std::uintptr_t stream_ptr) {
        py::gil_scoped_release release;
        fast_kernels::decode_quant_linear::cublaslt_fp16_after_dequant(
            activations_ptr, q_ptr, alpha_ptr, beta_ptr, output_ptr, weight_ptr, workspace_ptr,
            workspace_bytes, batch, n, k, group_size, stream_ptr);
      },
      py::arg("activations_ptr"), py::arg("q_ptr"), py::arg("alpha_ptr"), py::arg("beta_ptr"),
      py::arg("output_ptr"), py::arg("weight_ptr"), py::arg("workspace_ptr"),
      py::arg("workspace_bytes"), py::arg("batch"), py::arg("n"), py::arg("k"),
      py::arg("group_size"), py::arg("stream_ptr"));
#else
  const auto unavailable = []() {
    throw std::runtime_error("fast-kernels was built without CUDA support");
  };
  m.def("clustered_page_decode_forward",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int,
                      int, int, int, int, int, int, int, int, int, float, float,
                      std::uintptr_t) { unavailable(); });
  m.def("prefix_union_decode_forward",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int, int, int, int,
                      int, int, int, int, float, float, std::uintptr_t) { unavailable(); });
  m.def("compute_arc_w4a16_group_sums", [unavailable](std::uintptr_t, std::uintptr_t, int, int, int,
                                                      std::uintptr_t) { unavailable(); });
  m.def("pack_arc_w4a16_packets",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int, int,
                      int, std::uintptr_t) { unavailable(); });
  m.def("arc_w4a16_forward",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int, int,
                      int, int, std::uintptr_t) { unavailable(); });
  m.def("arc_w4a16_forward_split_k",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int, int,
                      int, int, int, std::uintptr_t) { unavailable(); });
  m.def("reduce_arc_w4a16_split_k_partials", [unavailable](std::uintptr_t, std::uintptr_t, int, int,
                                                           int, std::uintptr_t) { unavailable(); });
  m.def("dequant_w4a16_to_fp16",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, int, int, int,
                      std::uintptr_t) { unavailable(); });
  m.def("dequant_arc_w4a16_packets_to_fp16",
        [unavailable](std::uintptr_t, std::uintptr_t, int, int, int, int, std::uintptr_t) {
          unavailable();
        });
  m.def("cublaslt_fp16_with_weight",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::size_t,
                      int, int, int, std::uintptr_t) { unavailable(); });
  m.def("cublaslt_fp16_after_packet_dequant",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::size_t, int, int, int, int, int,
                      std::uintptr_t) { unavailable(); });
  m.def("cublaslt_fp16_after_dequant",
        [unavailable](std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                      std::uintptr_t, std::uintptr_t, std::uintptr_t, std::size_t, int, int, int,
                      int, std::uintptr_t) { unavailable(); });
#endif
}
