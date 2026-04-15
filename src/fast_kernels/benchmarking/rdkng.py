from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any, Literal

from fast_kernels.ops import RDKNGExplicitSketchSolver, cuda_rdkng_available
from fast_kernels.schemas import BenchmarkCase, BenchmarkSuite, ShapeCase

SubjectKind = Literal["kernel", "baseline"]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_galore_torch() -> Any:
    try:
        from galore_torch import GaLoreAdamW
        from galore_torch.galore_projector import GaLoreProjector
    except ImportError:
        return None
    return {
        "GaLoreAdamW": GaLoreAdamW,
        "GaLoreProjector": GaLoreProjector,
    }


def _make_case_id(
    subject_kind: SubjectKind,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
) -> str:
    normalized_subject = subject_id.replace("/", "__")
    return f"{subject_kind}-{normalized_subject}-{dtype}-{layout}-{shape_name}"


def _skipped_case(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
    dimensions: dict[str, int],
    reason: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=_make_case_id(subject_kind, subject_id, dtype, layout, shape_name),
        subject_kind=subject_kind,
        subject_id=subject_id,
        dtype=dtype,
        layout=layout,
        shape_name=shape_name,
        dimensions=dimensions,
        status="skipped",
        reason=reason,
    )


def _failed_case(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
    dimensions: dict[str, int],
    reason: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=_make_case_id(subject_kind, subject_id, dtype, layout, shape_name),
        subject_kind=subject_kind,
        subject_id=subject_id,
        dtype=dtype,
        layout=layout,
        shape_name=shape_name,
        dimensions=dimensions,
        status="failed",
        reason=reason,
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _make_ok_case(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
    dimensions: dict[str, int],
    latencies_us: list[float],
    metrics: dict[str, float],
) -> BenchmarkCase:
    latency_us_median = float(median(latencies_us))
    latency_us_p95 = _p95(latencies_us)
    throughput = 1_000_000.0 / latency_us_median
    return BenchmarkCase(
        case_id=_make_case_id(subject_kind, subject_id, dtype, layout, shape_name),
        subject_kind=subject_kind,
        subject_id=subject_id,
        dtype=dtype,
        layout=layout,
        shape_name=shape_name,
        dimensions=dimensions,
        status="ok",
        latency_us_median=latency_us_median,
        latency_us_p95=latency_us_p95,
        wall_latency_us_median=latency_us_median,
        wall_latency_us_p95=latency_us_p95,
        throughput=throughput,
        metrics=metrics,
    )


def _column_major_empty(torch: Any, rows: int, cols: int, *, device: Any, dtype: Any) -> Any:
    return torch.empty_strided((rows, cols), (1, rows), device=device, dtype=dtype)


def _orthonormal_columns(matrix: Any, *, rank: int) -> Any | None:
    torch = _require_torch()
    if matrix.numel() == 0 or rank <= 0:
        return None
    q, _ = torch.linalg.qr(matrix.float().contiguous(), mode="reduced")
    if q.ndim != 2 or q.shape[1] == 0:
        return None
    return q[:, : min(rank, int(q.shape[1]))].contiguous()


def _make_column_major_bf16(torch: Any, matrix: Any) -> tuple[Any, Any]:
    rows, cols = matrix.shape
    out = _column_major_empty(torch, rows, cols, device=matrix.device, dtype=torch.bfloat16)
    out.copy_(matrix.to(torch.bfloat16))
    return out, out.to(torch.float32)


def _apply_explicit_sketch(y: Any, x: Any, lambda_: float) -> Any:
    yt_x = y.transpose(0, 1).matmul(x)
    return y.matmul(yt_x) + (lambda_ * x)


def _woodbury_solve(y: Any, grad: Any, lambda_: float) -> Any:
    torch = _require_torch()
    gram = y.transpose(0, 1).matmul(y)
    rhs = y.transpose(0, 1).matmul(grad)
    middle = torch.linalg.solve(
        torch.eye(int(y.shape[1]), device=y.device, dtype=torch.float32) + (gram / lambda_),
        rhs,
    )
    return (grad / lambda_) - (y.matmul(middle) / (lambda_ * lambda_))


def _a_metric_norm(y: Any, x: Any, lambda_: float) -> Any:
    torch = _require_torch()
    yt_x = y.transpose(0, 1).matmul(x)
    return torch.sqrt((lambda_ * (x * x).sum()) + (yt_x * yt_x).sum())


def _model_gain(y: Any, grad: Any, v: Any, lambda_: float) -> float:
    torch = _require_torch()
    av = _apply_explicit_sketch(y, v, lambda_)
    return float((torch.dot(grad, v) - (0.5 * torch.dot(v, av))).item())


def _reference_metrics(
    y: Any,
    grad: Any,
    reference: Any,
    approx: Any,
    lambda_: float,
) -> dict[str, float]:
    torch = _require_torch()
    delta = approx - reference
    denom = float(_a_metric_norm(y, reference, lambda_).item())
    error = float(_a_metric_norm(y, delta, lambda_).item()) / max(denom, 1.0e-8)
    residual = grad - _apply_explicit_sketch(y, approx, lambda_)
    residual_ratio = float(residual.norm().item()) / max(float(grad.norm().item()), 1.0e-8)
    ref_gain = _model_gain(y, grad, reference, lambda_)
    approx_gain = _model_gain(y, grad, approx, lambda_)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            approx.reshape(1, -1), reference.reshape(1, -1), dim=1
        ).item()
    )
    metrics = {
        "a_metric_rel_error": error,
        "model_gain_fraction": approx_gain / max(ref_gain, 1.0e-8),
        "residual_ratio": residual_ratio,
        "cosine_to_reference": cosine,
    }
    return metrics


def _diag_preconditioner(y: Any, lambda_: float) -> Any:
    return (y * y).sum(dim=1) + lambda_


def _plain_cg_solve(
    y: Any,
    grad: Any,
    *,
    lambda_: float,
    cg_iters: int,
    tol: float,
    use_diag_preconditioner: bool,
) -> tuple[Any, dict[str, float]]:
    torch = _require_torch()
    x = torch.zeros_like(grad)
    r = grad.clone()
    diag = _diag_preconditioner(y, lambda_) if use_diag_preconditioner else None
    z = r / diag if diag is not None else r.clone()
    p = z.clone()
    rz_old = torch.dot(r, z)
    initial_residual_norm = float(r.norm().item())
    final_residual_norm = initial_residual_norm
    steps_taken = 0
    for idx in range(cg_iters):
        ap = _apply_explicit_sketch(y, p, lambda_)
        p_ap = torch.dot(p, ap)
        if float(abs(p_ap).item()) < 1.0e-20:
            break
        alpha = rz_old / p_ap
        x = x + (alpha * p)
        r = r - (alpha * ap)
        final_residual_norm = float(r.norm().item())
        steps_taken = idx + 1
        if final_residual_norm <= tol * max(float(grad.norm().item()), 1.0):
            break
        z = r / diag if diag is not None else r.clone()
        rz_new = torch.dot(r, z)
        if float(abs(rz_old).item()) < 1.0e-30:
            break
        beta = rz_new / rz_old
        p = z + (beta * p)
        rz_old = rz_new
    return x, {
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": final_residual_norm,
        "cg_steps_taken": float(steps_taken),
        "basis_refreshed": 0.0,
        "workspace_bytes": 0.0,
    }


def _reduced_solve(y: Any, grad: Any, basis: Any | None, *, lambda_: float) -> Any:
    torch = _require_torch()
    if basis is None or basis.numel() == 0:
        return torch.zeros_like(grad)
    ab = _apply_explicit_sketch(y, basis, lambda_)
    h = basis.transpose(0, 1).matmul(ab)
    rhs = basis.transpose(0, 1).matmul(grad)
    jitter = 1.0e-6 * torch.eye(int(h.shape[0]), device=h.device, dtype=h.dtype)
    coeff = torch.linalg.solve(h + jitter, rhs)
    return basis.matmul(coeff)


@dataclass(slots=True)
class _TrajectoryStep:
    y_bf16: Any
    y_f32: Any
    grad: Any
    reference: Any
    matrix_shape: tuple[int, int]


@dataclass(slots=True)
class _SubjectStepResult:
    direction: Any
    initial_residual_norm: float
    final_residual_norm: float
    cg_steps_taken: int
    basis_refreshed: bool
    workspace_bytes: int
    basis_reset: bool = False


class _RDKNGSubject:
    applies_update = False

    def __init__(
        self,
        *,
        n: int,
        s: int,
        basis_rank: int,
        max_cg: int,
        cg_iters: int,
        lambda_: float,
        tol: float,
        reset_threshold: float,
        use_diag_preconditioner: bool,
    ) -> None:
        self._solver = RDKNGExplicitSketchSolver(n, s, basis_rank, max_cg)
        self._cg_iters = cg_iters
        self._lambda = lambda_
        self._tol = tol
        self._reset_threshold = reset_threshold
        self._use_diag_preconditioner = use_diag_preconditioner

    def step(self, trajectory_step: _TrajectoryStep) -> _SubjectStepResult:
        grad_norm = float(trajectory_step.grad.norm().item())
        result = self._solver.step(
            trajectory_step.y_bf16,
            trajectory_step.grad,
            lambda_=self._lambda,
            cg_iters=self._cg_iters,
            tol=self._tol,
            use_diag_preconditioner=self._use_diag_preconditioner,
        )
        final_ratio = result.final_residual_norm / max(grad_norm, 1.0)
        basis_reset = False
        if final_ratio > self._reset_threshold:
            self._solver.reset_state(reset_basis=True, reset_diag=False)
            basis_reset = True
        return _SubjectStepResult(
            direction=result.direction,
            initial_residual_norm=result.initial_residual_norm,
            final_residual_norm=result.final_residual_norm,
            cg_steps_taken=result.cg_steps_taken,
            basis_refreshed=result.basis_refreshed,
            workspace_bytes=self._solver.workspace_bytes,
            basis_reset=basis_reset,
        )


class _PlainCGSubject:
    applies_update = False

    def __init__(self, *, lambda_: float, cg_iters: int, tol: float, use_diag_preconditioner: bool):
        self._lambda = lambda_
        self._cg_iters = cg_iters
        self._tol = tol
        self._use_diag_preconditioner = use_diag_preconditioner

    def step(self, trajectory_step: _TrajectoryStep) -> _SubjectStepResult:
        direction, stats = _plain_cg_solve(
            trajectory_step.y_f32,
            trajectory_step.grad,
            lambda_=self._lambda,
            cg_iters=self._cg_iters,
            tol=self._tol,
            use_diag_preconditioner=self._use_diag_preconditioner,
        )
        return _SubjectStepResult(
            direction=direction,
            initial_residual_norm=float(stats["initial_residual_norm"]),
            final_residual_norm=float(stats["final_residual_norm"]),
            cg_steps_taken=int(stats["cg_steps_taken"]),
            basis_refreshed=False,
            workspace_bytes=0,
        )


class _OfficialGaLoreProjectorSubject:
    applies_update = False

    def __init__(
        self,
        *,
        rank: int,
        update_proj_gap: int,
        scale: float,
        proj_type: str,
    ) -> None:
        galore = _require_galore_torch()
        if galore is None:
            raise RuntimeError(
                "galore-torch is required for the official GaLore projector baseline"
            )
        self._projector = galore["GaLoreProjector"](
            rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
        )
        self._iter = 0
        self._update_proj_gap = update_proj_gap

    def step(self, trajectory_step: _TrajectoryStep) -> _SubjectStepResult:
        grad_matrix = trajectory_step.grad.reshape(trajectory_step.matrix_shape).contiguous()
        refreshed = self._iter == 0 or (self._iter % self._update_proj_gap == 0)
        low_rank_grad = self._projector.project(grad_matrix, self._iter)
        direction = self._projector.project_back(low_rank_grad).reshape(-1).float().contiguous()
        self._iter += 1
        residual = trajectory_step.grad - direction
        return _SubjectStepResult(
            direction=direction,
            initial_residual_norm=float(trajectory_step.grad.norm().item()),
            final_residual_norm=float(residual.norm().item()),
            cg_steps_taken=0,
            basis_refreshed=refreshed,
            workspace_bytes=0,
        )


class _OfficialGaLoreAdamWSubject:
    applies_update = True

    def __init__(
        self,
        *,
        parameter: Any,
        lr: float,
        rank: int,
        update_proj_gap: int,
        scale: float,
        proj_type: str,
    ) -> None:
        galore = _require_galore_torch()
        if galore is None:
            raise RuntimeError(
                "galore-torch is required for the official GaLore optimizer baseline"
            )
        self._parameter = parameter
        self._optimizer = galore["GaLoreAdamW"](
            [
                {
                    "params": [parameter],
                    "rank": rank,
                    "update_proj_gap": update_proj_gap,
                    "scale": scale,
                    "proj_type": proj_type,
                }
            ],
            lr=lr,
            weight_decay=0.0,
            no_deprecation_warning=True,
        )
        self._lr = lr
        self._iter = 0
        self._update_proj_gap = update_proj_gap

    def step(self, trajectory_step: _TrajectoryStep) -> _SubjectStepResult:
        before = self._parameter.detach().float().reshape(-1).clone()
        refreshed = self._iter == 0 or (self._iter % self._update_proj_gap == 0)
        self._optimizer.step()
        self._optimizer.zero_grad()
        after = self._parameter.detach().float().reshape(-1).contiguous()
        direction = ((before - after) / self._lr).contiguous()
        self._iter += 1
        residual = trajectory_step.grad - direction
        return _SubjectStepResult(
            direction=direction,
            initial_residual_norm=float(trajectory_step.grad.norm().item()),
            final_residual_norm=float(residual.norm().item()),
            cg_steps_taken=0,
            basis_refreshed=refreshed,
            workspace_bytes=0,
        )


@dataclass(slots=True)
class _BlockConfig:
    trajectory_steps: int
    warmup_steps: int
    basis_rank: int
    hybrid_cg_iters: int
    cg_reference_iters: int
    lambda_: float
    tol: float
    solution_rank: int
    reset_threshold: float
    use_diag_preconditioner: bool
    galore_update_proj_gap: int
    galore_scale: float
    galore_proj_type: str


@dataclass(slots=True)
class _TrainingConfig:
    steps: int
    basis_rank: int
    hybrid_cg_iters: int
    cg_reference_iters: int
    lambda_: float
    tol: float
    lr: float
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int
    reset_threshold: float
    target_fraction: float
    use_diag_preconditioner: bool
    galore_update_proj_gap: int
    galore_scale: float
    galore_proj_type: str


def _parse_block_config(suite: BenchmarkSuite) -> _BlockConfig:
    raw = dict(suite.metadata.get("rdkng", {}))
    return _BlockConfig(
        trajectory_steps=int(raw.get("trajectory_steps", 10)),
        warmup_steps=int(raw.get("warmup_steps", 2)),
        basis_rank=int(raw.get("basis_rank", 8)),
        hybrid_cg_iters=int(raw.get("hybrid_cg_iters", 4)),
        cg_reference_iters=int(raw.get("cg_reference_iters", 8)),
        lambda_=float(raw.get("lambda", 1.0e-3)),
        tol=float(raw.get("tol", 1.0e-4)),
        solution_rank=int(raw.get("solution_rank", 8)),
        reset_threshold=float(raw.get("reset_threshold", 0.5)),
        use_diag_preconditioner=bool(raw.get("use_diag_preconditioner", True)),
        galore_update_proj_gap=int(raw.get("galore_update_proj_gap", 200)),
        galore_scale=float(raw.get("galore_scale", 1.0)),
        galore_proj_type=str(raw.get("galore_proj_type", "std")),
    )


def _parse_training_config(suite: BenchmarkSuite) -> _TrainingConfig:
    raw = dict(suite.metadata.get("rdkng", {}))
    return _TrainingConfig(
        steps=int(raw.get("steps", 6)),
        basis_rank=int(raw.get("basis_rank", 8)),
        hybrid_cg_iters=int(raw.get("hybrid_cg_iters", 4)),
        cg_reference_iters=int(raw.get("cg_reference_iters", 8)),
        lambda_=float(raw.get("lambda", 1.0e-3)),
        tol=float(raw.get("tol", 1.0e-4)),
        lr=float(raw.get("lr", 0.1)),
        d_model=int(raw.get("d_model", 256)),
        num_heads=int(raw.get("num_heads", 8)),
        d_ff=int(raw.get("d_ff", 1024)),
        vocab_size=int(raw.get("vocab_size", 4096)),
        reset_threshold=float(raw.get("reset_threshold", 0.5)),
        target_fraction=float(raw.get("target_fraction", 0.75)),
        use_diag_preconditioner=bool(raw.get("use_diag_preconditioner", True)),
        galore_update_proj_gap=int(raw.get("galore_update_proj_gap", 200)),
        galore_scale=float(raw.get("galore_scale", 1.0)),
        galore_proj_type=str(raw.get("galore_proj_type", "std")),
    )


def _make_block_trajectory(
    torch: Any,
    *,
    rows: int,
    cols: int,
    s: int,
    steps: int,
    regime: str,
    lambda_: float,
    solution_rank: int,
    device: Any,
    seed: int,
) -> list[_TrajectoryStep]:
    n = rows * cols
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    basis = _orthonormal_columns(
        torch.randn((n, solution_rank), generator=generator, device=device, dtype=torch.float32),
        rank=solution_rank,
    )
    if basis is None:
        raise ValueError("solution_rank must be positive")
    coeff = torch.randn((solution_rank,), generator=generator, device=device, dtype=torch.float32)
    sketch_base = torch.randn((n, s), generator=generator, device=device, dtype=torch.float32)
    trajectory: list[_TrajectoryStep] = []
    for _ in range(steps):
        if regime == "compressible_drift":
            coeff = (0.95 * coeff) + (
                0.15 * torch.randn((solution_rank,), generator=generator, device=device)
            )
            reference_seed = basis.matmul(coeff) + (
                0.01 * torch.randn((n,), generator=generator, device=device)
            )
            sketch = sketch_base + (
                0.01 * torch.randn((n, s), generator=generator, device=device)
            )
        elif regime == "noncompressible_control":
            reference_seed = torch.randn((n,), generator=generator, device=device)
            sketch = torch.randn((n, s), generator=generator, device=device)
        else:
            raise ValueError(f"unsupported RDK-NG layout/regime: {regime}")
        reference_seed = reference_seed / max(float(reference_seed.norm().item()), 1.0e-6)
        y_bf16, y_f32 = _make_column_major_bf16(torch, sketch / math.sqrt(float(s)))
        grad = _apply_explicit_sketch(y_f32, reference_seed, lambda_).to(torch.float32).contiguous()
        reference = _woodbury_solve(y_f32, grad, lambda_).to(torch.float32).contiguous()
        trajectory.append(
            _TrajectoryStep(
                y_bf16=y_bf16,
                y_f32=y_f32,
                grad=grad,
                reference=reference,
                matrix_shape=(rows, cols),
            )
        )
    return trajectory


def _make_block_subject_runner(
    subject_id: str,
    *,
    n: int,
    s: int,
    basis_rank: int,
    hybrid_cg_iters: int,
    cg_reference_iters: int,
    lambda_: float,
    tol: float,
    reset_threshold: float,
    use_diag_preconditioner: bool,
    galore_update_proj_gap: int,
    galore_scale: float,
    galore_proj_type: str,
) -> Any:
    if subject_id == "rdkng/explicit_sketch_hybrid":
        return _RDKNGSubject(
            n=n,
            s=s,
            basis_rank=basis_rank,
            max_cg=max(hybrid_cg_iters, cg_reference_iters),
            cg_iters=hybrid_cg_iters,
            lambda_=lambda_,
            tol=tol,
            reset_threshold=reset_threshold,
            use_diag_preconditioner=use_diag_preconditioner,
        )
    if subject_id == "rdkng/explicit_sketch_lowrank":
        return _RDKNGSubject(
            n=n,
            s=s,
            basis_rank=basis_rank,
            max_cg=max(hybrid_cg_iters, cg_reference_iters),
            cg_iters=0,
            lambda_=lambda_,
            tol=tol,
            reset_threshold=reset_threshold,
            use_diag_preconditioner=use_diag_preconditioner,
        )
    if subject_id == "torch/plain_cg_reference":
        return _PlainCGSubject(
            lambda_=lambda_,
            cg_iters=cg_reference_iters,
            tol=tol,
            use_diag_preconditioner=use_diag_preconditioner,
        )
    if subject_id == "official/galore_projector":
        return _OfficialGaLoreProjectorSubject(
            rank=basis_rank,
            update_proj_gap=galore_update_proj_gap,
            scale=galore_scale,
            proj_type=galore_proj_type,
        )
    raise ValueError(f"unsupported RDK-NG subject id: {subject_id}")


def _run_block_subject(
    subject_kind: SubjectKind,
    subject_id: str,
    *,
    dtype: str,
    layout: str,
    shape: ShapeCase,
    config: _BlockConfig,
    trajectory: list[_TrajectoryStep],
) -> BenchmarkCase:
    torch = _require_torch()
    rows = shape.require_dimension("m")
    cols = shape.require_dimension("n")
    n = rows * cols
    s = shape.require_dimension("k")
    runner = _make_block_subject_runner(
        subject_id,
        n=n,
        s=s,
        basis_rank=config.basis_rank,
        hybrid_cg_iters=config.hybrid_cg_iters,
        cg_reference_iters=config.cg_reference_iters,
        lambda_=config.lambda_,
        tol=config.tol,
        reset_threshold=config.reset_threshold,
        use_diag_preconditioner=config.use_diag_preconditioner,
        galore_update_proj_gap=config.galore_update_proj_gap,
        galore_scale=config.galore_scale,
        galore_proj_type=config.galore_proj_type,
    )
    latencies_us: list[float] = []
    metric_lists: dict[str, list[float]] = defaultdict(list)
    for step_index, step in enumerate(trajectory):
        torch.cuda.synchronize()
        start = perf_counter()
        step_result = runner.step(step)
        torch.cuda.synchronize()
        elapsed_us = (perf_counter() - start) * 1_000_000.0
        if step_index < config.warmup_steps:
            continue
        latencies_us.append(elapsed_us)
        for name, value in _reference_metrics(
            step.y_f32,
            step.grad,
            step.reference,
            step_result.direction,
            config.lambda_,
        ).items():
            metric_lists[name].append(value)
        metric_lists["cg_steps_taken"].append(float(step_result.cg_steps_taken))
        metric_lists["curvature_products"].append(float(1 + step_result.cg_steps_taken))
        metric_lists["workspace_bytes"].append(float(step_result.workspace_bytes))
        metric_lists["basis_refresh_rate"].append(1.0 if step_result.basis_refreshed else 0.0)
        metric_lists["basis_reset_rate"].append(1.0 if step_result.basis_reset else 0.0)
        metric_lists["final_residual_norm"].append(float(step_result.final_residual_norm))
        metric_lists["initial_residual_norm"].append(float(step_result.initial_residual_norm))
    if not latencies_us:
        raise RuntimeError("RDK-NG benchmark produced no timed samples")
    averaged_metrics = {
        name: float(sum(values) / len(values)) for name, values in metric_lists.items() if values
    }
    return _make_ok_case(
        subject_kind=subject_kind,
        subject_id=subject_id,
        dtype=dtype,
        layout=layout,
        shape_name=shape.name,
        dimensions=shape.dimensions(),
        latencies_us=latencies_us,
        metrics=averaged_metrics,
    )


class _TinyMLP:
    def __init__(self, module: Any) -> None:
        self._module = module
        self.last_input = None
        self.last_output = None

    @property
    def up_proj(self) -> Any:
        return self._module.up_proj

    def __call__(self, x: Any, *, capture: bool) -> Any:
        self.last_input = x
        up = self._module.up_proj(x)
        if capture:
            up.retain_grad()
            self.last_output = up
        else:
            self.last_output = None
        return self._module.down_proj(self._module.activation(up))


def _build_tiny_transformer(torch: Any, config: _TrainingConfig, *, device: Any, seed: int) -> Any:
    import torch.nn as nn
    import torch.nn.functional as functional

    class _TinyMLPModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
            self.activation = functional.gelu

    class _TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(config.d_model)
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.ln2 = nn.LayerNorm(config.d_model)
            self.mlp_core = _TinyMLPModule()
            self.mlp = _TinyMLP(self.mlp_core)

        def forward(self, x: Any, *, capture: bool) -> Any:
            batch, seq_len, _ = x.shape
            head_dim = config.d_model // config.num_heads
            h = self.ln1(x)
            qkv = self.qkv(h).view(batch, seq_len, 3, config.num_heads, head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).reshape(batch, seq_len, config.d_model)
            x = x + self.out_proj(attn)
            return x + self.mlp(self.ln2(x), capture=capture)

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.position_embedding = nn.Embedding(512, config.d_model)
            self.block = _TinyBlock()
            self.ln_f = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        def forward(self, tokens: Any, *, capture: bool) -> Any:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
            x = self.block(x, capture=capture)
            x = self.ln_f(x)
            return self.lm_head(x)

    torch.manual_seed(seed)
    model = _TinyModel().to(device=device, dtype=torch.bfloat16)
    return model


def _configure_teacher_student(
    torch: Any,
    config: _TrainingConfig,
    *,
    device: Any,
    seed: int,
) -> tuple[Any, Any, Any]:
    teacher = _build_tiny_transformer(torch, config, device=device, seed=seed)
    student = _build_tiny_transformer(torch, config, device=device, seed=seed)
    student.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        trainable = student.block.mlp.up_proj.weight
        trainable.add_(0.05 * torch.randn_like(trainable, dtype=torch.float32).to(trainable.dtype))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name == "block.mlp_core.up_proj.weight")
    return teacher.eval(), student.train(), student.block.mlp.up_proj.weight


def _make_training_batch(
    torch: Any,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: Any,
    seed: int,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )


def _training_step_trajectory(
    torch: Any,
    *,
    student: Any,
    teacher: Any,
    trainable: Any,
    tokens: Any,
    lambda_: float,
    batch_size: int,
) -> tuple[_TrajectoryStep, float]:
    import torch.nn.functional as functional

    student.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_logits = teacher(tokens, capture=False)
    logits = student(tokens, capture=True)
    loss_per_sample = functional.mse_loss(
        logits.float(),
        teacher_logits.float(),
        reduction="none",
    ).mean(
        dim=(1, 2)
    )
    loss = loss_per_sample.mean()
    loss.backward()
    up_input = student.block.mlp.last_input.detach().float()
    up_grad = (student.block.mlp.last_output.grad.detach().float() * batch_size)
    per_sample_grads = torch.einsum("bto,bti->boi", up_grad, up_input).reshape(batch_size, -1)
    sketch = per_sample_grads.transpose(0, 1).contiguous() / math.sqrt(float(batch_size))
    y_bf16, y_f32 = _make_column_major_bf16(torch, sketch)
    grad = trainable.grad.detach().float().reshape(-1).contiguous()
    reference = _woodbury_solve(y_f32, grad, lambda_).to(torch.float32).contiguous()
    return (
        _TrajectoryStep(
            y_bf16=y_bf16,
            y_f32=y_f32,
            grad=grad,
            reference=reference,
            matrix_shape=tuple(int(dim) for dim in trainable.shape),
        ),
        float(loss.item()),
    )


def _make_training_subject_runner(
    subject_id: str,
    *,
    n: int,
    s: int,
    basis_rank: int,
    hybrid_cg_iters: int,
    cg_reference_iters: int,
    lambda_: float,
    tol: float,
    reset_threshold: float,
    use_diag_preconditioner: bool,
    parameter: Any,
    lr: float,
    galore_update_proj_gap: int,
    galore_scale: float,
    galore_proj_type: str,
) -> Any:
    if subject_id == "official/galore_adamw":
        return _OfficialGaLoreAdamWSubject(
            parameter=parameter,
            lr=lr,
            rank=basis_rank,
            update_proj_gap=galore_update_proj_gap,
            scale=galore_scale,
            proj_type=galore_proj_type,
        )
    return _make_block_subject_runner(
        subject_id,
        n=n,
        s=s,
        basis_rank=basis_rank,
        hybrid_cg_iters=hybrid_cg_iters,
        cg_reference_iters=cg_reference_iters,
        lambda_=lambda_,
        tol=tol,
        reset_threshold=reset_threshold,
        use_diag_preconditioner=use_diag_preconditioner,
        galore_update_proj_gap=galore_update_proj_gap,
        galore_scale=galore_scale,
        galore_proj_type=galore_proj_type,
    )


def _run_training_subject(
    subject_kind: SubjectKind,
    subject_id: str,
    *,
    dtype: str,
    layout: str,
    shape: ShapeCase,
    config: _TrainingConfig,
    device: Any,
    seed: int,
) -> BenchmarkCase:
    torch = _require_torch()
    batch_size = shape.batch
    seq_len = shape.require_dimension("max_seq_len")
    n = config.d_model * config.d_ff
    s = batch_size
    teacher, student, trainable = _configure_teacher_student(
        torch,
        config,
        device=device,
        seed=seed,
    )
    runner = _make_training_subject_runner(
        subject_id,
        n=n,
        s=s,
        basis_rank=config.basis_rank,
        hybrid_cg_iters=config.hybrid_cg_iters,
        cg_reference_iters=config.cg_reference_iters,
        lambda_=config.lambda_,
        tol=config.tol,
        reset_threshold=config.reset_threshold,
        use_diag_preconditioner=config.use_diag_preconditioner,
        parameter=trainable,
        lr=config.lr,
        galore_update_proj_gap=config.galore_update_proj_gap,
        galore_scale=config.galore_scale,
        galore_proj_type=config.galore_proj_type,
    )
    latencies_us: list[float] = []
    losses: list[float] = []
    metric_lists: dict[str, list[float]] = defaultdict(list)
    target_loss: float | None = None
    elapsed_to_target_s: float | None = None
    cumulative_time_s = 0.0
    for step_idx in range(config.steps):
        tokens = _make_training_batch(
            torch,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            device=device,
            seed=seed + (step_idx * 17),
        )
        torch.cuda.synchronize()
        start = perf_counter()
        trajectory_step, loss_value = _training_step_trajectory(
            torch,
            student=student,
            teacher=teacher,
            trainable=trainable,
            tokens=tokens,
            lambda_=config.lambda_,
            batch_size=batch_size,
        )
        step_result = runner.step(trajectory_step)
        if not bool(getattr(runner, "applies_update", False)):
            with torch.no_grad():
                trainable.add_(
                    step_result.direction.reshape_as(trainable).to(trainable.dtype),
                    alpha=-config.lr,
                )
        torch.cuda.synchronize()
        elapsed_s = perf_counter() - start
        cumulative_time_s += elapsed_s
        latencies_us.append(elapsed_s * 1_000_000.0)
        losses.append(loss_value)
        if target_loss is None:
            target_loss = loss_value * config.target_fraction
        if elapsed_to_target_s is None and target_loss is not None and loss_value <= target_loss:
            elapsed_to_target_s = cumulative_time_s
        for name, value in _reference_metrics(
            trajectory_step.y_f32,
            trajectory_step.grad,
            trajectory_step.reference,
            step_result.direction,
            config.lambda_,
        ).items():
            metric_lists[name].append(value)
        metric_lists["cg_steps_taken"].append(float(step_result.cg_steps_taken))
        metric_lists["curvature_products"].append(float(1 + step_result.cg_steps_taken))
        metric_lists["workspace_bytes"].append(float(step_result.workspace_bytes))
        metric_lists["basis_refresh_rate"].append(1.0 if step_result.basis_refreshed else 0.0)
        metric_lists["basis_reset_rate"].append(1.0 if step_result.basis_reset else 0.0)
        metric_lists["loss"].append(loss_value)
    averaged_metrics = {
        name: float(sum(values) / len(values)) for name, values in metric_lists.items() if values
    }
    averaged_metrics["loss_after_budget"] = float(losses[-1])
    if elapsed_to_target_s is not None:
        averaged_metrics["time_to_target_s"] = float(elapsed_to_target_s)
    return _make_ok_case(
        subject_kind=subject_kind,
        subject_id=subject_id,
        dtype=dtype,
        layout=layout,
        shape_name=shape.name,
        dimensions=shape.dimensions() | {"n": n, "k": s},
        latencies_us=latencies_us,
        metrics=averaged_metrics,
    )


def _apply_speedups(cases: list[BenchmarkCase]) -> None:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for case in cases:
        if case.status != "ok" or case.wall_latency_us_median is None:
            continue
        key = (case.dtype, case.layout, case.shape_name)
        grouped[key][case.subject_id] = case.wall_latency_us_median
    for case in cases:
        if case.status != "ok" or case.wall_latency_us_median is None:
            continue
        latencies = grouped[(case.dtype, case.layout, case.shape_name)]
        case.speedup_vs = {
            baseline_id: baseline_latency / case.wall_latency_us_median
            for baseline_id, baseline_latency in latencies.items()
            if baseline_id
            in {
                "torch/plain_cg_reference",
                "official/galore_projector",
                "official/galore_adamw",
            }
            and baseline_id != case.subject_id
        }


def _registry_groups(suite: BenchmarkSuite) -> tuple[tuple[SubjectKind, list[str]], ...]:
    return (("kernel", suite.kernels.ids), ("baseline", suite.baselines.ids))


def run_rdkng_suite(suite: BenchmarkSuite) -> tuple[list[BenchmarkCase], list[str]]:
    torch = _require_torch()
    galore = _require_galore_torch()
    notes = [
        "RDK-NG suites benchmark explicit-sketch natural-gradient solves on CUDA tensors.",
        (
            "The GaLore comparison uses the official galore-torch projector/optimizer path "
            "from the GaLore authors."
        ),
    ]
    if torch is None:
        reason = "PyTorch is required for RDK-NG benchmarks."
        skipped_cases: list[BenchmarkCase] = []
        for subject_kind, subject_ids in _registry_groups(suite):
            for subject_id in subject_ids:
                for dtype in suite.dtypes:
                    for layout in suite.layouts:
                        for shape in suite.shapes:
                            skipped_cases.append(
                                _skipped_case(
                                    subject_kind=subject_kind,
                                    subject_id=subject_id,
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=shape.dimensions(),
                                    reason=reason,
                                )
                            )
        notes.append(reason)
        return skipped_cases, notes

    if galore is None:
        reason = "galore-torch is required for the official GaLore baselines."
        skipped_cases = []
        for subject_kind, subject_ids in _registry_groups(suite):
            for subject_id in subject_ids:
                for dtype in suite.dtypes:
                    for layout in suite.layouts:
                        for shape in suite.shapes:
                            skipped_cases.append(
                                _skipped_case(
                                    subject_kind=subject_kind,
                                    subject_id=subject_id,
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=shape.dimensions(),
                                    reason=reason,
                                )
                            )
        notes.append(reason)
        return skipped_cases, notes

    if not torch.cuda.is_available() or not cuda_rdkng_available():
        reason = "CUDA, PyTorch, and a CUDA-enabled native build are required for RDK-NG."
        skipped_cases = []
        for subject_kind, subject_ids in _registry_groups(suite):
            for subject_id in subject_ids:
                for dtype in suite.dtypes:
                    for layout in suite.layouts:
                        for shape in suite.shapes:
                            skipped_cases.append(
                                _skipped_case(
                                    subject_kind=subject_kind,
                                    subject_id=subject_id,
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=shape.dimensions(),
                                    reason=reason,
                                )
                            )
        notes.append(reason)
        return skipped_cases, notes

    device = torch.device("cuda")
    cases: list[BenchmarkCase] = []
    base_seed = int(suite.metadata.get("seed", 13))
    if suite.id == "rdkng_block":
        config = _parse_block_config(suite)
        notes.append(
            "Block mode uses synthetic explicit-sketch trajectories with compressible and "
            "noncompressible regimes."
        )
        for dtype in suite.dtypes:
            if dtype != "bf16":
                raise ValueError("rdkng_block currently requires dtype=bf16")
            for layout in suite.layouts:
                for shape_index, shape in enumerate(suite.shapes):
                    trajectory = _make_block_trajectory(
                        torch,
                        rows=shape.require_dimension("m"),
                        cols=shape.require_dimension("n"),
                        s=shape.require_dimension("k"),
                        steps=config.trajectory_steps,
                        regime=layout,
                        lambda_=config.lambda_,
                        solution_rank=config.solution_rank,
                        device=device,
                        seed=base_seed
                        + (shape_index * 101)
                        + (17 if layout == "noncompressible_control" else 0),
                    )
                    for subject_kind, subject_ids in _registry_groups(suite):
                        for subject_id in subject_ids:
                            try:
                                cases.append(
                                    _run_block_subject(
                                        subject_kind,
                                        subject_id,
                                        dtype=dtype,
                                        layout=layout,
                                        shape=shape,
                                        config=config,
                                        trajectory=trajectory,
                                    )
                                )
                            except Exception as exc:
                                cases.append(
                                    _failed_case(
                                        subject_kind=subject_kind,
                                        subject_id=subject_id,
                                        dtype=dtype,
                                        layout=layout,
                                        shape_name=shape.name,
                                        dimensions=shape.dimensions(),
                                        reason=str(exc),
                                    )
                                )
    elif suite.id == "rdkng_training":
        config = _parse_training_config(suite)
        notes.append(
            "Training mode runs a tiny teacher-student transformer and only updates the MLP "
            "up-projection block with each subject's direction."
        )
        for dtype in suite.dtypes:
            if dtype != "bf16":
                raise ValueError("rdkng_training currently requires dtype=bf16")
            for layout in suite.layouts:
                if layout != "teacher_student":
                    raise ValueError("rdkng_training currently requires layout=teacher_student")
                for shape_index, shape in enumerate(suite.shapes):
                    for subject_kind, subject_ids in _registry_groups(suite):
                        for subject_id in subject_ids:
                            try:
                                cases.append(
                                    _run_training_subject(
                                        subject_kind,
                                        subject_id,
                                        dtype=dtype,
                                        layout=layout,
                                        shape=shape,
                                        config=config,
                                        device=device,
                                        seed=base_seed + (shape_index * 101),
                                    )
                                )
                            except Exception as exc:
                                cases.append(
                                    _failed_case(
                                        subject_kind=subject_kind,
                                        subject_id=subject_id,
                                        dtype=dtype,
                                        layout=layout,
                                        shape_name=shape.name,
                                        dimensions=shape.dimensions(),
                                        reason=str(exc),
                                    )
                                )
    else:
        raise ValueError(f"unsupported RDK-NG suite id: {suite.id}")

    _apply_speedups(cases)
    return cases, notes
