from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_kernels.native import native_build_info, native_module


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without benchmark deps
        raise RuntimeError("PyTorch is required for RDK-NG ops") from exc
    return torch


def cuda_rdkng_available() -> bool:
    info = native_build_info()
    if not bool(info.get("available", False)) or not bool(info.get("compiled_with_cuda", False)):
        return False
    native = native_module()
    return hasattr(native, "RDKNGExplicitSketchSolverHandle")


def _require_cuda_backend() -> Any:
    if not cuda_rdkng_available():
        raise RuntimeError("fast-kernels native module was built without RDK-NG CUDA support")
    return native_module()


def _current_stream_ptr(device: Any) -> int:
    torch = _require_torch()
    return int(torch.cuda.current_stream(device=device).cuda_stream)


def _as_column_major_matrix(tensor: Any, *, name: str, rows: int, cols: int, dtype: Any) -> Any:
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != 2 or tuple(tensor.shape) != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols})")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.stride() == (1, rows):
        return tensor
    return tensor.transpose(0, 1).contiguous().transpose(0, 1)


def _require_vector(tensor: Any, *, name: str, length: int, dtype: Any, device: Any) -> Any:
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != 1 or tensor.numel() != length:
        raise ValueError(f"{name} must have shape ({length},)")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _column_major_empty(torch: Any, rows: int, cols: int, *, device: Any, dtype: Any) -> Any:
    return torch.empty_strided((rows, cols), (1, rows), device=device, dtype=dtype)


@dataclass(slots=True)
class RDKNGStepResult:
    direction: Any
    basis: Any | None
    diag_ema: Any | None
    initial_residual_norm: float
    final_residual_norm: float
    cg_steps_taken: int
    basis_refreshed: bool


class RDKNGExplicitSketchSolver:
    def __init__(self, n: int, s: int, r: int, max_cg: int, *, device: Any | None = None) -> None:
        torch = _require_torch()
        native = _require_cuda_backend()
        if device is None:
            device = torch.device("cuda")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._device = resolved_device
        if self._device.type != "cuda":
            raise ValueError("RDKNGExplicitSketchSolver requires a CUDA device")
        self._n = int(n)
        self._s = int(s)
        self._r = int(r)
        self._handle = native.RDKNGExplicitSketchSolverHandle(self._n, self._s, self._r, max_cg)
        self._basis = (
            _column_major_empty(torch, self._n, self._r, device=self._device, dtype=torch.bfloat16)
            if self._r > 0
            else None
        )
        if self._basis is not None:
            self._basis.zero_()
        self._basis_initialized = False
        self._basis_replace_cursor = 0
        self._diag_ema: Any | None = None

    @property
    def n(self) -> int:
        return self._n

    @property
    def s(self) -> int:
        return self._s

    @property
    def r(self) -> int:
        return self._r

    @property
    def max_cg(self) -> int:
        return int(self._handle.max_cg)

    @property
    def workspace_bytes(self) -> int:
        return int(self._handle.workspace_bytes())

    @property
    def basis(self) -> Any | None:
        return self._basis

    @property
    def diag_ema(self) -> Any | None:
        return self._diag_ema

    def reset_state(self, *, reset_basis: bool = True, reset_diag: bool = True) -> None:
        if reset_basis and self._basis is not None:
            self._basis.zero_()
            self._basis_initialized = False
            self._basis_replace_cursor = 0
        if reset_diag and self._diag_ema is not None:
            self._diag_ema.zero_()

    def step(
        self,
        y: Any,
        grad: Any,
        *,
        lambda_: float,
        cg_iters: int,
        tol: float = 1.0e-4,
        use_diag_preconditioner: bool = True,
        diag_ema_beta: float = 0.95,
        basis_replace_col: int | None = None,
        basis_append_threshold: float = 1.0e-4,
        reset_basis: bool = False,
    ) -> RDKNGStepResult:
        torch = _require_torch()
        if reset_basis:
            self.reset_state(reset_basis=True, reset_diag=False)

        y_col_major = _as_column_major_matrix(
            y,
            name="Y",
            rows=self._n,
            cols=self._s,
            dtype=torch.bfloat16,
        )
        if y_col_major.device != self._device:
            raise ValueError(f"Y must be on device {self._device}")
        grad_vec = _require_vector(
            grad,
            name="grad",
            length=self._n,
            dtype=torch.float32,
            device=y_col_major.device,
        )
        out_v = torch.empty((self._n,), device=self._device, dtype=torch.float32)

        diag_ptr = 0
        if use_diag_preconditioner:
            if self._diag_ema is None:
                self._diag_ema = torch.zeros((self._n,), device=self._device, dtype=torch.float32)
            diag_ptr = int(
                _require_vector(
                    self._diag_ema,
                    name="diag_ema",
                    length=self._n,
                    dtype=torch.float32,
                    device=self._device,
                ).data_ptr()
            )

        basis_in_ptr = 0
        basis_out_ptr = 0
        replace_col = basis_replace_col
        if self._basis is not None:
            basis_out_ptr = int(self._basis.data_ptr())
            if self._basis_initialized:
                basis_in_ptr = basis_out_ptr
            if replace_col is None:
                replace_col = self._basis_replace_cursor
                self._basis_replace_cursor = (self._basis_replace_cursor + 1) % self._r

        stats = self._handle.step(
            y_ptr=int(y_col_major.data_ptr()),
            grad_ptr=int(grad_vec.data_ptr()),
            basis_in_ptr=basis_in_ptr,
            basis_out_ptr=basis_out_ptr,
            lambda_=float(lambda_),
            cg_iters=int(cg_iters),
            tol=float(tol),
            out_v_ptr=int(out_v.data_ptr()),
            diag_ema_ptr=diag_ptr,
            diag_ema_beta=float(diag_ema_beta),
            basis_replace_col=-1 if replace_col is None else int(replace_col),
            basis_append_threshold=float(basis_append_threshold),
            stream_ptr=_current_stream_ptr(self._device),
        )

        if bool(stats.basis_refreshed):
            self._basis_initialized = True

        return RDKNGStepResult(
            direction=out_v,
            basis=self._basis,
            diag_ema=self._diag_ema,
            initial_residual_norm=float(stats.initial_residual_norm),
            final_residual_norm=float(stats.final_residual_norm),
            cg_steps_taken=int(stats.cg_steps_taken),
            basis_refreshed=bool(stats.basis_refreshed),
        )
