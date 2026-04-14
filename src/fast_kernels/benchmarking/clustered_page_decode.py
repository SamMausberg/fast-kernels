from __future__ import annotations

import importlib.util
import math
from functools import partial
from statistics import median
from time import perf_counter
from typing import Any, Literal

from fast_kernels.ops import (
    clustered_page_decode,
    cuda_clustered_decode_available,
    estimate_page_decode_metrics,
    pack_paged_kv_bf16,
    plan_clustered_page_decode,
    quantize_paged_kv_fp8,
    quantize_paged_kv_int8,
    reference_clustered_page_decode,
)
from fast_kernels.schemas import BenchmarkCase, BenchmarkSuite, ShapeCase

WARMUP_ITERS = 2
TIMING_ITERS = 8
SubjectKind = Literal["kernel", "baseline"]

KERNEL_TO_IMPL = {
    "decode/clustered_page_decode_auto": "auto",
    "decode/clustered_page_decode_direct": "direct",
    "decode/clustered_page_decode_clustered": "clustered",
}


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
    speedup_vs: dict[str, float],
    metrics: dict[str, float],
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
        speedup_vs=speedup_vs,
        metrics=metrics,
    )


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


def _case_seed(shape: ShapeCase, layout: str) -> int:
    return (
        hash(
            (
                shape.name,
                shape.batch,
                shape.require_dimension("max_seq_len"),
                shape.require_dimension("num_q_heads"),
                shape.require_dimension("num_kv_heads"),
                shape.require_dimension("head_dim"),
                shape.require_dimension("page_size"),
                layout,
            )
        )
        & 0x7FFFFFFF
    )


def _make_inputs(torch: Any, shape: ShapeCase, layout: str) -> tuple[Any, Any, Any, Any]:
    batch = shape.batch
    max_seq_len = shape.require_dimension("max_seq_len")
    num_q_heads = shape.require_dimension("num_q_heads")
    num_kv_heads = shape.require_dimension("num_kv_heads")
    head_dim = shape.require_dimension("head_dim")
    page_size = shape.require_dimension("page_size")
    seed = _case_seed(shape, layout)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    query = (0.25 * torch.randn((batch, num_q_heads, head_dim), device=device)).to(torch.bfloat16)
    keys = (0.35 * torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device)).to(
        torch.bfloat16
    )
    values = (0.35 * torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device)).to(
        torch.bfloat16
    )
    seq_low = max(page_size, max_seq_len - (2 * page_size) + 1)
    seq_lens = torch.randint(
        low=seq_low,
        high=max_seq_len + 1,
        size=(batch,),
        dtype=torch.int32,
    ).contiguous()
    return query.contiguous(), keys.contiguous(), values.contiguous(), seq_lens


def _build_cache(keys: Any, values: Any, seq_lens: Any, *, page_size: int, layout: str) -> Any:
    if layout == "bf16_kv":
        return pack_paged_kv_bf16(
            keys, values, seq_lens, page_size=page_size, fragment_pages=True, seed=7
        )
    if layout == "fp8_kv":
        return quantize_paged_kv_fp8(
            keys, values, seq_lens, page_size=page_size, fragment_pages=True, seed=7
        )
    if layout == "int8_kv":
        return quantize_paged_kv_int8(
            keys, values, seq_lens, page_size=page_size, fragment_pages=True, seed=7
        )
    raise ValueError(f"unsupported layout: {layout}")


def _max_abs_diff(actual: Any, expected: Any) -> float:
    torch = _require_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to compute max_abs_error")
    return float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item())


def _throughput_tokens_per_second(batch: int, latency_us: float) -> float:
    return (batch * 1_000_000.0) / latency_us


def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def run_clustered_page_decode_suite(
    suite: BenchmarkSuite,
) -> tuple[list[BenchmarkCase], list[str]]:
    torch = _require_torch()
    notes = [
        "Clustered page decode benchmarks use repo-native paged caches and "
        "Blackwell-first heuristics.",
        "FlashInfer remains optional; the baseline is skipped when the runtime is unavailable.",
    ]
    if torch is None:
        notes.append("PyTorch is unavailable, so all clustered page decode cases were skipped.")
        skipped_cases: list[BenchmarkCase] = []
        registry_groups: tuple[tuple[SubjectKind, list[str]], ...] = (
            ("kernel", suite.kernels.ids),
            ("baseline", suite.baselines.ids),
        )
        for subject_kind, subject_ids in registry_groups:
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
                                    reason=(
                                        "PyTorch is required for clustered_page_decode benchmarks."
                                    ),
                                )
                            )
        return skipped_cases, notes

    if not torch.cuda.is_available() or not cuda_clustered_decode_available():
        notes.append("CUDA or the native CUDA extension is unavailable, so all cases were skipped.")
        skipped_cases = []
        registry_groups = (
            ("kernel", suite.kernels.ids),
            ("baseline", suite.baselines.ids),
        )
        for subject_kind, subject_ids in registry_groups:
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
                                    reason=(
                                        "CUDA, PyTorch, and a CUDA-enabled native build "
                                        "are required."
                                    ),
                                )
                            )
        return skipped_cases, notes

    cases: list[BenchmarkCase] = []
    for dtype in suite.dtypes:
        if dtype != "bf16":
            raise ValueError("clustered_page_decode suites currently only support dtype=bf16")
        for layout in suite.layouts:
            for shape in suite.shapes:
                query, keys, values, seq_lens = _make_inputs(torch, shape, layout)
                dimensions = shape.dimensions()
                try:
                    cache = _build_cache(
                        keys,
                        values,
                        seq_lens,
                        page_size=shape.require_dimension("page_size"),
                        layout=layout,
                    )
                except Exception as exc:
                    reason = f"cache build failed: {exc}"
                    for subject_kind, subject_ids in (
                        ("kernel", suite.kernels.ids),
                        ("baseline", suite.baselines.ids),
                    ):
                        for subject_id in subject_ids:
                            cases.append(
                                _failed_case(
                                    subject_kind=subject_kind,
                                    subject_id=subject_id,
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=dimensions,
                                    reason=reason,
                                )
                            )
                    continue

                plan = plan_clustered_page_decode(
                    page_table=cache.page_table,
                    seq_lens=cache.seq_lens,
                    num_q_heads=shape.require_dimension("num_q_heads"),
                    num_kv_heads=shape.require_dimension("num_kv_heads"),
                    head_dim=shape.require_dimension("head_dim"),
                    page_size=shape.require_dimension("page_size"),
                    kv_layout=layout,
                )
                reference_output = reference_clustered_page_decode(query, cache)
                reference_fn = partial(
                    reference_clustered_page_decode,
                    query,
                    cache,
                )
                reference_latency_us, reference_p95_us = _time_callable(reference_fn, torch)
                reference_throughput = _throughput_tokens_per_second(
                    shape.batch, reference_latency_us
                )
                reference_metrics = estimate_page_decode_metrics(cache, plan)
                reference_metrics["max_abs_error"] = 0.0

                for kernel_id in suite.kernels.ids:
                    force_impl = KERNEL_TO_IMPL[kernel_id]
                    try:
                        kernel_output = clustered_page_decode(
                            query,
                            cache,
                            plan=plan,
                            force_impl=force_impl,
                        )
                        torch.testing.assert_close(
                            kernel_output,
                            reference_output,
                            atol=suite.tolerances.atol,
                            rtol=suite.tolerances.rtol,
                        )
                        fn = partial(
                            clustered_page_decode,
                            query,
                            cache,
                            plan=plan,
                            force_impl=force_impl,
                        )
                        latency_us, p95_us = _time_callable(fn, torch)
                        metrics = estimate_page_decode_metrics(cache, plan)
                        metrics["max_abs_error"] = _max_abs_diff(kernel_output, reference_output)
                        cases.append(
                            _timed_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                latency_us_median=latency_us,
                                latency_us_p95=p95_us,
                                throughput=_throughput_tokens_per_second(shape.batch, latency_us),
                                speedup_vs={
                                    "torch/reference_clustered_page_decode": (
                                        reference_latency_us / latency_us
                                    )
                                },
                                metrics=metrics,
                            )
                        )
                    except Exception as exc:
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

                cases.append(
                    _timed_case(
                        subject_kind="baseline",
                        subject_id="torch/reference_clustered_page_decode",
                        dtype=dtype,
                        layout=layout,
                        shape_name=shape.name,
                        dimensions=dimensions,
                        latency_us_median=reference_latency_us,
                        latency_us_p95=reference_p95_us,
                        throughput=reference_throughput,
                        speedup_vs={},
                        metrics=reference_metrics,
                    )
                )

                if _flashinfer_available():
                    cases.append(
                        _skipped_case(
                            subject_kind="baseline",
                            subject_id="vendor/flashinfer_clustered_page_decode",
                            dtype=dtype,
                            layout=layout,
                            shape_name=shape.name,
                            dimensions=dimensions,
                            reason=(
                                "FlashInfer is installed, but this repo still needs a "
                                "version-locked adapter for the local runtime build."
                            ),
                        )
                    )
                else:
                    cases.append(
                        _skipped_case(
                            subject_kind="baseline",
                            subject_id="vendor/flashinfer_clustered_page_decode",
                            dtype=dtype,
                            layout=layout,
                            shape_name=shape.name,
                            dimensions=dimensions,
                            reason="FlashInfer is not installed.",
                        )
                    )

    return cases, notes
