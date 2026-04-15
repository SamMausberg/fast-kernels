#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace fast_kernels::rdkng {

struct StepStats {
  float initial_residual_norm = 0.0f;
  float final_residual_norm = 0.0f;
  int cg_steps_taken = 0;
  int basis_refreshed = 0;
};

class ExplicitSketchSolver {
 public:
  ExplicitSketchSolver(int n, int s, int r, int max_cg);
  ~ExplicitSketchSolver();

  ExplicitSketchSolver(const ExplicitSketchSolver&) = delete;
  ExplicitSketchSolver& operator=(const ExplicitSketchSolver&) = delete;

  [[nodiscard]] int n() const;
  [[nodiscard]] int s() const;
  [[nodiscard]] int r() const;
  [[nodiscard]] int max_cg() const;

  [[nodiscard]] std::size_t workspace_bytes() const;

  void step(std::uintptr_t y_ptr,
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
            StepStats* host_stats = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fast_kernels::rdkng
