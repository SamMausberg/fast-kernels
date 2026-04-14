from __future__ import annotations

import math
from functools import partial
from statistics import median
from time import perf_counter
from typing import Any, Literal

from fast_kernels.ops import (
    arc_w4a16_forward,
    arc_w4a16_supported_impls,
    cublaslt_fp16_after_dequant,
    dequant_w4a16_to_fp16,
    group_size_for_layout,
    pack_arc_w4a16_packets,
)
from fast_kernels.schemas import BenchmarkCase, BenchmarkSuite

WARMUP_ITERS = 4
TIMING_ITERS = 12
SubjectKind = Literal["kernel", "baseline"]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


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


def _timed_case(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
    dimensions: dict[str, int],
    latency_us_median: float,
    latency_us_p95: float,
    throughput: float,
    speedup_vs: dict[str, float] | None = None,
) -> BenchmarkCase:
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
        throughput=throughput,
        speedup_vs=speedup_vs or {},
    )


def _throughput_tokens_per_second(batch: int, latency_us: float) -> float:
    return (batch * 1_000_000.0) / latency_us


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _time_callable(fn: Any, torch: Any) -> tuple[float, float]:
    torch.cuda.synchronize()
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(TIMING_ITERS):
        torch.cuda.synchronize()
        start = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_us.append((perf_counter() - start) * 1_000_000.0)

    return float(median(samples_us)), _p95(samples_us)


def _seed_for_case(batch: int, n: int, k: int, group_size: int) -> int:
    return (batch * 1_000_003 + n * 1009 + k * 101 + group_size) % (2**31 - 1)


def _make_case_inputs(
    torch: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    seed = _seed_for_case(batch, n, k, group_size)
    torch.manual_seed(seed)

    num_groups = k // group_size
    activations = (0.25 * torch.randn((batch, k), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
    alpha = (0.02 + (0.18 * torch.rand((n, num_groups), device=device, dtype=torch.float32))).to(
        torch.float16
    )
    zero_points = torch.randint(0, 16, (n, num_groups), device=device, dtype=torch.int16)
    beta = -(alpha * zero_points.to(dtype=torch.float16))
    return activations.contiguous(), q_u8.contiguous(), alpha.contiguous(), beta.contiguous()


def _explicit_reference_output(
    torch: Any,
    activations: Any,
    q_u8: Any,
    alpha: Any,
    beta: Any,
    *,
    group_size: int,
) -> Any:
    low = torch.bitwise_and(q_u8, 0x0F).to(torch.float32)
    high = torch.bitwise_right_shift(q_u8, 4).to(torch.float32)
    q_values = torch.stack((low, high), dim=-1).reshape(q_u8.shape[0], q_u8.shape[1] * 2)
    alpha_expanded = alpha.to(torch.float32).repeat_interleave(group_size, dim=1)
    beta_expanded = beta.to(torch.float32).repeat_interleave(group_size, dim=1)
    weights = (alpha_expanded * q_values) + beta_expanded
    return torch.matmul(activations.to(torch.float32), weights.transpose(0, 1)).to(torch.float16)


def _assert_close(actual: Any, expected: Any, *, atol: float, rtol: float, label: str) -> None:
    if bool(actual.shape != expected.shape):
        raise AssertionError(
            f"{label} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    if not bool(actual.dtype == expected.dtype):
        raise AssertionError(f"{label} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if bool(actual.allclose(expected, atol=atol, rtol=rtol)):
        return
    diff = (actual - expected).abs()
    max_diff = float(diff.max().item())
    raise AssertionError(f"{label} mismatch: max_abs_diff={max_diff:.6f}")


def _run_reference(
    torch: Any,
    q_u8: Any,
    alpha: Any,
    beta: Any,
    *,
    group_size: int,
    reference_weight: Any,
    activations: Any,
    reference_output: Any,
) -> tuple[Any, Any]:
    return (
        dequant_w4a16_to_fp16(
            q_u8,
            alpha,
            beta,
            group_size=group_size,
            output=reference_weight,
        ),
        torch.mm(
            activations,
            reference_weight.transpose(0, 1),
            out=reference_output,
        ),
    )


def _run_vendor(
    activations: Any,
    q_u8: Any,
    alpha: Any,
    beta: Any,
    *,
    group_size: int,
    output: Any,
    weight_buffer: Any,
    workspace: Any,
) -> Any:
    return cublaslt_fp16_after_dequant(
        activations,
        q_u8,
        alpha,
        beta,
        group_size=group_size,
        output=output,
        weight_buffer=weight_buffer,
        workspace=workspace,
    )


def _run_kernel(
    activations: Any,
    packets: Any,
    *,
    n: int,
    k: int,
    group_size: int,
    output: Any,
    impl: str | None = None,
) -> Any:
    return arc_w4a16_forward(
        activations,
        packets,
        n=n,
        k=k,
        group_size=group_size,
        output=output,
        impl=impl,
    )


def _kernel_impl_for_id(kernel_id: str) -> str | None:
    if kernel_id == "decode/w4a16_linear":
        return None
    if kernel_id == "decode/w4a16_linear_scalar":
        return "scalar"
    if kernel_id == "decode/w4a16_linear_tc":
        return "tc"
    if kernel_id == "decode/w4a16_linear_wgmma":
        return "wgmma"
    raise ValueError(f"unsupported kernel id: {kernel_id}")


def run_decode_linear_w4a16_suite(suite: BenchmarkSuite) -> tuple[list[BenchmarkCase], list[str]]:
    cases: list[BenchmarkCase] = []
    notes = [
        "Decode suite runs real CUDA work when torch and a CUDA-enabled native build are "
        "available.",
        f"Timing uses {WARMUP_ITERS} warmup iterations and "
        f"{TIMING_ITERS} measured iterations per subject.",
        "Kernel subjects benchmark auto, scalar, Tensor Core, and WGMMA-capable dispatch paths.",
    ]

    torch = _require_torch()
    if torch is None:
        reason = (
            "PyTorch is not installed. Run `uv sync --extra benchmark` to execute the decode suite."
        )
        for dtype in suite.dtypes:
            for layout in suite.layouts:
                for shape in suite.shapes:
                    dimensions = shape.dimensions()
                    for kernel_id in suite.kernels.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
                    for baseline_id in suite.baselines.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="baseline",
                                subject_id=baseline_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
        return cases, notes

    from fast_kernels.ops import cuda_decode_available

    if not cuda_decode_available():
        reason = (
            "Native module is not compiled with CUDA. Reinstall with "
            "`uv sync --extra benchmark` or opt back into CPU-only mode explicitly."
        )
        for dtype in suite.dtypes:
            for layout in suite.layouts:
                for shape in suite.shapes:
                    dimensions = shape.dimensions()
                    for kernel_id in suite.kernels.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
                    for baseline_id in suite.baselines.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="baseline",
                                subject_id=baseline_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
        return cases, notes

    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() is false on this machine."
        for dtype in suite.dtypes:
            for layout in suite.layouts:
                for shape in suite.shapes:
                    dimensions = shape.dimensions()
                    for kernel_id in suite.kernels.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
                    for baseline_id in suite.baselines.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="baseline",
                                subject_id=baseline_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=reason,
                            )
                        )
        return cases, notes

    device = torch.device("cuda")
    supported_impls = set(arc_w4a16_supported_impls(device))

    for dtype in suite.dtypes:
        dtype_reason: str | None = None
        if dtype != "fp16":
            dtype_reason = "Only fp16 is implemented for the ARC decode suite."

        for layout in suite.layouts:
            layout_reason: str | None = None
            try:
                group_size = group_size_for_layout(layout)
            except ValueError as exc:
                group_size = -1
                layout_reason = str(exc)

            for shape in suite.shapes:
                dimensions = shape.dimensions()
                batch = shape.batch
                n = shape.require_dimension("n")
                k = shape.require_dimension("k")

                skip_reason: str | None = dtype_reason or layout_reason
                if skip_reason is None and shape.require_dimension("m") != 1:
                    skip_reason = "decode suite currently requires shape.m == 1"
                if skip_reason is None and n % 128 != 0:
                    skip_reason = "decode suite requires n to be divisible by 128"
                if skip_reason is None and k % group_size != 0:
                    skip_reason = "decode suite requires k to be divisible by group_size"

                if skip_reason is not None:
                    for baseline_id in suite.baselines.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="baseline",
                                subject_id=baseline_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=skip_reason,
                            )
                        )
                    for kernel_id in suite.kernels.ids:
                        cases.append(
                            _skipped_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=skip_reason,
                            )
                        )
                    continue

                try:
                    activations, q_u8, alpha, beta = _make_case_inputs(
                        torch,
                        batch=batch,
                        n=n,
                        k=k,
                        group_size=group_size,
                        device=device,
                    )
                    packets = pack_arc_w4a16_packets(q_u8, alpha, beta, group_size=group_size)
                    reference_weight = torch.empty((n, k), device=device, dtype=torch.float16)
                    reference_output = torch.empty((batch, n), device=device, dtype=torch.float16)
                    vendor_output = torch.empty((batch, n), device=device, dtype=torch.float16)
                    vendor_weight = torch.empty((n, k), device=device, dtype=torch.float16)
                    kernel_output = torch.empty((batch, n), device=device, dtype=torch.float16)
                    workspace = torch.empty(8 * 1024 * 1024, device=device, dtype=torch.uint8)

                    dequant_w4a16_to_fp16(
                        q_u8,
                        alpha,
                        beta,
                        group_size=group_size,
                        output=reference_weight,
                    )
                    torch.mm(activations, reference_weight.transpose(0, 1), out=reference_output)

                    explicit_output = _explicit_reference_output(
                        torch,
                        activations,
                        q_u8,
                        alpha,
                        beta,
                        group_size=group_size,
                    )
                    _assert_close(
                        reference_output,
                        explicit_output,
                        atol=suite.tolerances.atol,
                        rtol=suite.tolerances.rtol,
                        label="reference output",
                    )

                    cublaslt_fp16_after_dequant(
                        activations,
                        q_u8,
                        alpha,
                        beta,
                        group_size=group_size,
                        output=vendor_output,
                        weight_buffer=vendor_weight,
                        workspace=workspace,
                    )
                    _assert_close(
                        vendor_output,
                        reference_output,
                        atol=suite.tolerances.atol,
                        rtol=suite.tolerances.rtol,
                        label="vendor baseline output",
                    )

                    for kernel_id in suite.kernels.ids:
                        impl = _kernel_impl_for_id(kernel_id)
                        if impl is not None and impl not in supported_impls:
                            continue
                        arc_w4a16_forward(
                            activations,
                            packets,
                            n=n,
                            k=k,
                            group_size=group_size,
                            output=kernel_output,
                            impl=impl,
                        )
                        _assert_close(
                            kernel_output,
                            reference_output,
                            atol=suite.tolerances.atol,
                            rtol=suite.tolerances.rtol,
                            label=f"{kernel_id} output",
                        )
                except Exception as exc:  # pragma: no cover - exercised on GPU-only failures
                    for baseline_id in suite.baselines.ids:
                        cases.append(
                            _failed_case(
                                subject_kind="baseline",
                                subject_id=baseline_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=str(exc),
                            )
                        )
                    for kernel_id in suite.kernels.ids:
                        cases.append(
                            _failed_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=str(exc),
                            )
                        )
                    continue

                subject_latencies: dict[str, float] = {}

                reference_subject = suite.baselines.ids[0]
                vendor_subject = suite.baselines.ids[1]

                reference_median, reference_p95 = _time_callable(
                    partial(
                        _run_reference,
                        torch,
                        q_u8,
                        alpha,
                        beta,
                        group_size=group_size,
                        reference_weight=reference_weight,
                        activations=activations,
                        reference_output=reference_output,
                    ),
                    torch,
                )
                subject_latencies[reference_subject] = reference_median

                vendor_median, vendor_p95 = _time_callable(
                    partial(
                        _run_vendor,
                        activations,
                        q_u8,
                        alpha,
                        beta,
                        group_size=group_size,
                        output=vendor_output,
                        weight_buffer=vendor_weight,
                        workspace=workspace,
                    ),
                    torch,
                )
                subject_latencies[vendor_subject] = vendor_median

                vendor_speedup_denominator = subject_latencies[vendor_subject]

                cases.append(
                    _timed_case(
                        subject_kind="baseline",
                        subject_id=reference_subject,
                        dtype=dtype,
                        layout=layout,
                        shape_name=shape.name,
                        dimensions=dimensions,
                        latency_us_median=reference_median,
                        latency_us_p95=reference_p95,
                        throughput=_throughput_tokens_per_second(batch, reference_median),
                        speedup_vs={vendor_subject: vendor_speedup_denominator / reference_median},
                    )
                )
                cases.append(
                    _timed_case(
                        subject_kind="baseline",
                        subject_id=vendor_subject,
                        dtype=dtype,
                        layout=layout,
                        shape_name=shape.name,
                        dimensions=dimensions,
                        latency_us_median=vendor_median,
                        latency_us_p95=vendor_p95,
                        throughput=_throughput_tokens_per_second(batch, vendor_median),
                    )
                )
                for kernel_id in suite.kernels.ids:
                    impl = _kernel_impl_for_id(kernel_id)
                    if impl is not None and impl not in supported_impls:
                        cases.append(
                            _skipped_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=(
                                    f"{kernel_id} is unsupported on device capability "
                                    f"{torch.cuda.get_device_capability(device)}"
                                ),
                            )
                        )
                        continue

                    kernel_median, kernel_p95 = _time_callable(
                        partial(
                            _run_kernel,
                            activations,
                            packets,
                            n=n,
                            k=k,
                            group_size=group_size,
                            output=kernel_output,
                            impl=impl,
                        ),
                        torch,
                    )
                    subject_latencies[kernel_id] = kernel_median
                    cases.append(
                        _timed_case(
                            subject_kind="kernel",
                            subject_id=kernel_id,
                            dtype=dtype,
                            layout=layout,
                            shape_name=shape.name,
                            dimensions=dimensions,
                            latency_us_median=kernel_median,
                            latency_us_p95=kernel_p95,
                            throughput=_throughput_tokens_per_second(batch, kernel_median),
                            speedup_vs={vendor_subject: vendor_speedup_denominator / kernel_median},
                        )
                    )

    return cases, notes
