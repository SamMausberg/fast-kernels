from __future__ import annotations

from typing import Any

import pytest

from fast_kernels.ops import RDKNGExplicitSketchSolver, cuda_rdkng_available


def _load_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


torch = _load_torch()


def _cuda_ready() -> bool:
    return torch is not None and bool(torch.cuda.is_available()) and cuda_rdkng_available()


pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="RDK-NG GPU tests require PyTorch, torch.cuda, and a CUDA-enabled build.",
)


def _column_major_bf16(matrix: Any) -> tuple[Any, Any]:
    out = torch.empty_strided(
        matrix.shape,
        (1, int(matrix.shape[0])),
        device=matrix.device,
        dtype=torch.bfloat16,
    )
    out.copy_(matrix.to(torch.bfloat16))
    return out, out.to(torch.float32)


def _apply_explicit_sketch(y: Any, x: Any, lambda_: float) -> Any:
    return y @ (y.transpose(0, 1) @ x) + (lambda_ * x)


def _woodbury_solve(y: Any, grad: Any, lambda_: float) -> Any:
    eye = torch.eye(int(y.shape[1]), device=y.device, dtype=torch.float32)
    gram = y.transpose(0, 1) @ y
    rhs = y.transpose(0, 1) @ grad
    middle = torch.linalg.solve(eye + (gram / lambda_), rhs)
    return (grad / lambda_) - (y @ middle) / (lambda_ * lambda_)


def _make_case(*, n: int = 512, s: int = 8, rank: int = 4, seed: int = 7) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    basis = torch.linalg.qr(
        torch.randn((n, rank), generator=generator, device="cuda", dtype=torch.float32),
        mode="reduced",
    ).Q
    coeff = torch.randn((rank,), generator=generator, device="cuda", dtype=torch.float32)
    reference_seed = basis @ coeff
    reference_seed = reference_seed / max(float(reference_seed.norm().item()), 1.0e-6)
    sketch = torch.randn((n, s), generator=generator, device="cuda", dtype=torch.float32)
    y_bf16, y_f32 = _column_major_bf16(sketch / (s**0.5))
    grad = _apply_explicit_sketch(y_f32, reference_seed, 1.0e-3).contiguous()
    return y_bf16, y_f32, grad


def test_rdkng_bootstraps_basis_without_input() -> None:
    y_bf16, _y_f32, grad = _make_case()
    solver = RDKNGExplicitSketchSolver(int(y_bf16.shape[0]), int(y_bf16.shape[1]), 4, 4)
    result = solver.step(y_bf16, grad, lambda_=1.0e-3, cg_iters=2)
    assert result.basis is not None
    assert result.basis_refreshed


def test_rdkng_hybrid_reduces_residual_vs_lowrank() -> None:
    y_bf16, y_f32, grad = _make_case()
    lowrank = RDKNGExplicitSketchSolver(int(y_bf16.shape[0]), int(y_bf16.shape[1]), 4, 4)
    hybrid = RDKNGExplicitSketchSolver(int(y_bf16.shape[0]), int(y_bf16.shape[1]), 4, 4)
    lowrank_result = lowrank.step(y_bf16, grad, lambda_=1.0e-3, cg_iters=0)
    hybrid_result = hybrid.step(y_bf16, grad, lambda_=1.0e-3, cg_iters=3)
    lowrank_residual = (
        grad - _apply_explicit_sketch(y_f32, lowrank_result.direction, 1.0e-3)
    ).norm()
    hybrid_residual = (
        grad - _apply_explicit_sketch(y_f32, hybrid_result.direction, 1.0e-3)
    ).norm()
    assert float(hybrid_residual.item()) <= float(lowrank_residual.item()) + 1.0e-5


def test_rdkng_rank_zero_tracks_plain_reference() -> None:
    y_bf16, y_f32, grad = _make_case(rank=2)
    solver = RDKNGExplicitSketchSolver(int(y_bf16.shape[0]), int(y_bf16.shape[1]), 0, 6)
    result = solver.step(y_bf16, grad, lambda_=1.0e-3, cg_iters=6)
    exact = _woodbury_solve(y_f32, grad, 1.0e-3)
    approx_error = (result.direction - exact).norm().item()
    zero_error = exact.norm().item()
    assert approx_error < zero_error
