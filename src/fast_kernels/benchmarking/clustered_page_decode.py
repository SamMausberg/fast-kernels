from __future__ import annotations

import hashlib
import importlib.util
import math
from dataclasses import dataclass
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
E2E_WARMUP_ITERS = 1
E2E_TIMING_ITERS = 3
SubjectKind = Literal["kernel", "baseline"]
FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024

KERNEL_TO_IMPL = {
    "decode/clustered_page_decode_auto": "auto",
    "decode/clustered_page_decode_direct": "direct",
    "decode/clustered_page_decode_clustered": "clustered",
}


@dataclass(frozen=True, slots=True)
class TimingStats:
    wall_latency_us_median: float
    wall_latency_us_p95: float
    device_latency_us_median: float
    device_latency_us_p95: float


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
    timing: TimingStats,
    decode_tokens_per_second: float,
    context_tokens_per_second: float,
    effective_kv_gib_per_second: float,
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
        latency_us_median=timing.wall_latency_us_median,
        latency_us_p95=timing.wall_latency_us_p95,
        wall_latency_us_median=timing.wall_latency_us_median,
        wall_latency_us_p95=timing.wall_latency_us_p95,
        device_latency_us_median=timing.device_latency_us_median,
        device_latency_us_p95=timing.device_latency_us_p95,
        throughput=decode_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        context_tokens_per_second=context_tokens_per_second,
        effective_kv_gib_per_second=effective_kv_gib_per_second,
        speedup_vs=speedup_vs,
        metrics=metrics,
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _time_callable(
    fn: Any,
    torch: Any,
    *,
    warmup_iters: int = WARMUP_ITERS,
    timing_iters: int = TIMING_ITERS,
) -> TimingStats:
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    wall_samples_us: list[float] = []
    device_samples_us: list[float] = []
    for _ in range(timing_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = perf_counter()
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        wall_samples_us.append((perf_counter() - wall_start) * 1_000_000.0)
        device_samples_us.append(float(start_event.elapsed_time(end_event) * 1_000.0))

    return TimingStats(
        wall_latency_us_median=float(median(wall_samples_us)),
        wall_latency_us_p95=_p95(wall_samples_us),
        device_latency_us_median=float(median(device_samples_us)),
        device_latency_us_p95=_p95(device_samples_us),
    )


def _case_seed(shape: ShapeCase, layout: str) -> int:
    digest = hashlib.sha256(
        "|".join(
            [
                shape.name,
                str(shape.batch),
                str(shape.require_dimension("max_seq_len")),
                str(shape.require_dimension("num_q_heads")),
                str(shape.require_dimension("num_kv_heads")),
                str(shape.require_dimension("head_dim")),
                str(shape.require_dimension("page_size")),
                layout,
            ]
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little") & 0x7FFFFFFF


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


def _decode_tokens_per_second(batch: int, latency_us: float) -> float:
    return (batch * 1_000_000.0) / latency_us


def _context_tokens_per_second(seq_lens: Any, latency_us: float) -> float:
    total_context_tokens = float(seq_lens.to(dtype=_require_torch().float32).sum().item())
    return (total_context_tokens * 1_000_000.0) / latency_us


def _effective_kv_gib_per_second(
    hbm_bytes_per_decode_token: float,
    decode_tokens_per_second: float,
) -> float:
    return (hbm_bytes_per_decode_token * decode_tokens_per_second) / float(1024**3)


def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def _flashinfer_skip_reason(cache: Any) -> str | None:
    if cache.kv_layout != "bf16_kv":
        return (
            "FlashInfer baseline currently supports only bf16_kv; this repo's fp8/int8 paged "
            "cache formats use block-scaled layouts that are not wired into the adapter yet."
        )
    if cache.keys_are_rotated:
        return (
            "FlashInfer baseline currently expects unrotated key pages so it can apply "
            "ROPE_LLAMA inside the decode kernel."
        )
    if int(cache.seq_lens.min().item()) <= 0:
        return "FlashInfer baseline currently skips empty-sequence decode cases."
    return None


def _flashinfer_page_metadata(cache: Any, device: Any) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    seq_lens_cpu = cache.seq_lens.to(device="cpu", dtype=torch.int32).contiguous()
    pages_per_request = ((seq_lens_cpu + (cache.page_size - 1)) // cache.page_size).to(torch.int32)
    indptr_cpu = torch.empty((int(seq_lens_cpu.shape[0]) + 1,), dtype=torch.int32)
    indptr_cpu[0] = 0
    indptr_cpu[1:] = torch.cumsum(pages_per_request, dim=0)

    page_slices = []
    for request_index, num_pages in enumerate(pages_per_request.tolist()):
        if num_pages <= 0:
            continue
        page_slices.append(
            cache.page_table[request_index, :num_pages].to(device="cpu", dtype=torch.int32)
        )
    flat_indices_cpu = (
        torch.cat(page_slices, dim=0).contiguous()
        if page_slices
        else torch.empty((0,), dtype=torch.int32)
    )
    last_page_len_cpu = torch.where(
        seq_lens_cpu > 0,
        ((seq_lens_cpu - 1) % cache.page_size) + 1,
        torch.zeros_like(seq_lens_cpu),
    )
    return (
        indptr_cpu.to(device=device, dtype=torch.int32),
        flat_indices_cpu.to(device=device, dtype=torch.int32),
        last_page_len_cpu.to(device=device, dtype=torch.int32),
    )


def _make_flashinfer_decode_fn(
    query: Any,
    cache: Any,
    *,
    softmax_scale: float,
    rope_theta: float,
) -> Any:
    torch = _require_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for the FlashInfer baseline.")
    try:
        import flashinfer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only when flashinfer is absent
        raise RuntimeError("FlashInfer is not installed.") from exc

    skip_reason = _flashinfer_skip_reason(cache)
    if skip_reason is not None:
        raise NotImplementedError(skip_reason)

    indptr, indices, last_page_len = _flashinfer_page_metadata(cache, query.device)
    workspace = torch.zeros(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=query.device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="HND",
        use_tensor_cores=(int(query.shape[1]) // cache.num_kv_heads) > 1,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        int(query.shape[1]),
        cache.num_kv_heads,
        cache.head_dim,
        cache.page_size,
        pos_encoding_mode="ROPE_LLAMA",
        q_data_type=query.dtype,
        kv_data_type=cache.key_pages.dtype,
        o_data_type=query.dtype,
        sm_scale=float(softmax_scale),
        rope_theta=float(rope_theta),
    )
    paged_kv_cache = (cache.key_pages, cache.value_pages)
    return partial(wrapper.run, query, paged_kv_cache)


def _make_repo_e2e_decode_fn(
    query: Any,
    cache: Any,
    *,
    force_impl: str,
    softmax_scale: float,
    rope_theta: float,
) -> Any:
    def _run() -> Any:
        plan = plan_clustered_page_decode(
            page_table=cache.page_table,
            seq_lens=cache.seq_lens,
            num_q_heads=int(query.shape[1]),
            num_kv_heads=cache.num_kv_heads,
            head_dim=cache.head_dim,
            page_size=cache.page_size,
            kv_layout=cache.kv_layout,
            use_cache=False,
        )
        return clustered_page_decode(
            query,
            cache,
            plan=plan,
            softmax_scale=softmax_scale,
            rope_theta=rope_theta,
            force_impl=force_impl,
        )

    return _run


def _make_flashinfer_e2e_decode_fn(
    query: Any,
    cache: Any,
    *,
    softmax_scale: float,
    rope_theta: float,
) -> Any:
    def _run() -> Any:
        return _make_flashinfer_decode_fn(
            query,
            cache,
            softmax_scale=softmax_scale,
            rope_theta=rope_theta,
        )()

    return _run


def run_clustered_page_decode_suite(
    suite: BenchmarkSuite,
) -> tuple[list[BenchmarkCase], list[str]]:
    torch = _require_torch()
    notes = [
        "Clustered page decode benchmarks use repo-native paged caches and "
        "Blackwell-first heuristics.",
        "Primary timings are steady-state decode-step measurements with warmed plans/wrappers.",
        "decode tok/s is batch / step latency; context tok/s is sum(seq_lens) / step latency.",
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
                softmax_scale = 1.0 / math.sqrt(cache.head_dim)
                hbm_bytes_per_decode_token = estimate_page_decode_metrics(cache, plan)[
                    "hbm_bytes_per_token"
                ]
                reference_output = reference_clustered_page_decode(query, cache)
                reference_fn = partial(
                    reference_clustered_page_decode,
                    query,
                    cache,
                )
                reference_timing = _time_callable(reference_fn, torch)
                reference_decode_toks = _decode_tokens_per_second(
                    shape.batch, reference_timing.wall_latency_us_median
                )
                reference_context_toks = _context_tokens_per_second(
                    cache.seq_lens, reference_timing.wall_latency_us_median
                )
                reference_metrics = estimate_page_decode_metrics(cache, plan)
                reference_metrics["max_abs_error"] = 0.0
                reference_metrics["e2e_wall_latency_us_median"] = (
                    reference_timing.wall_latency_us_median
                )
                reference_metrics["e2e_wall_latency_us_p95"] = reference_timing.wall_latency_us_p95
                reference_metrics["e2e_device_latency_us_median"] = (
                    reference_timing.device_latency_us_median
                )
                reference_metrics["e2e_device_latency_us_p95"] = (
                    reference_timing.device_latency_us_p95
                )
                reference_metrics["e2e_decode_tokens_per_second"] = reference_decode_toks
                reference_metrics["e2e_context_tokens_per_second"] = reference_context_toks

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
                        timing = _time_callable(fn, torch)
                        e2e_timing = _time_callable(
                            _make_repo_e2e_decode_fn(
                                query,
                                cache,
                                force_impl=force_impl,
                                softmax_scale=softmax_scale,
                                rope_theta=10000.0,
                            ),
                            torch,
                            warmup_iters=E2E_WARMUP_ITERS,
                            timing_iters=E2E_TIMING_ITERS,
                        )
                        decode_toks = _decode_tokens_per_second(
                            shape.batch, timing.wall_latency_us_median
                        )
                        context_toks = _context_tokens_per_second(
                            cache.seq_lens, timing.wall_latency_us_median
                        )
                        metrics = estimate_page_decode_metrics(cache, plan)
                        metrics["max_abs_error"] = _max_abs_diff(kernel_output, reference_output)
                        metrics["e2e_wall_latency_us_median"] = e2e_timing.wall_latency_us_median
                        metrics["e2e_wall_latency_us_p95"] = e2e_timing.wall_latency_us_p95
                        metrics["e2e_device_latency_us_median"] = (
                            e2e_timing.device_latency_us_median
                        )
                        metrics["e2e_device_latency_us_p95"] = e2e_timing.device_latency_us_p95
                        metrics["e2e_decode_tokens_per_second"] = _decode_tokens_per_second(
                            shape.batch, e2e_timing.wall_latency_us_median
                        )
                        metrics["e2e_context_tokens_per_second"] = _context_tokens_per_second(
                            cache.seq_lens, e2e_timing.wall_latency_us_median
                        )
                        cases.append(
                            _timed_case(
                                subject_kind="kernel",
                                subject_id=kernel_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                timing=timing,
                                decode_tokens_per_second=decode_toks,
                                context_tokens_per_second=context_toks,
                                effective_kv_gib_per_second=_effective_kv_gib_per_second(
                                    hbm_bytes_per_decode_token, decode_toks
                                ),
                                speedup_vs={
                                    "torch/reference_clustered_page_decode": (
                                        reference_timing.wall_latency_us_median
                                        / timing.wall_latency_us_median
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
                        timing=reference_timing,
                        decode_tokens_per_second=reference_decode_toks,
                        context_tokens_per_second=reference_context_toks,
                        effective_kv_gib_per_second=_effective_kv_gib_per_second(
                            hbm_bytes_per_decode_token, reference_decode_toks
                        ),
                        speedup_vs={},
                        metrics=reference_metrics,
                    )
                )

                if _flashinfer_available():
                    skip_reason = _flashinfer_skip_reason(cache)
                    if skip_reason is not None:
                        cases.append(
                            _skipped_case(
                                subject_kind="baseline",
                                subject_id="vendor/flashinfer_clustered_page_decode",
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=dimensions,
                                reason=skip_reason,
                            )
                        )
                    else:
                        try:
                            vendor_fn = _make_flashinfer_decode_fn(
                                query,
                                cache,
                                softmax_scale=softmax_scale,
                                rope_theta=10000.0,
                            )
                            vendor_output = vendor_fn()
                            torch.testing.assert_close(
                                vendor_output,
                                reference_output,
                                atol=suite.tolerances.atol,
                                rtol=suite.tolerances.rtol,
                            )
                            vendor_timing = _time_callable(vendor_fn, torch)
                            vendor_e2e_timing = _time_callable(
                                _make_flashinfer_e2e_decode_fn(
                                    query,
                                    cache,
                                    softmax_scale=softmax_scale,
                                    rope_theta=10000.0,
                                ),
                                torch,
                                warmup_iters=E2E_WARMUP_ITERS,
                                timing_iters=E2E_TIMING_ITERS,
                            )
                            vendor_decode_toks = _decode_tokens_per_second(
                                shape.batch, vendor_timing.wall_latency_us_median
                            )
                            vendor_context_toks = _context_tokens_per_second(
                                cache.seq_lens, vendor_timing.wall_latency_us_median
                            )
                            vendor_metrics = dict(reference_metrics)
                            vendor_metrics["max_abs_error"] = _max_abs_diff(
                                vendor_output, reference_output
                            )
                            vendor_metrics["e2e_wall_latency_us_median"] = (
                                vendor_e2e_timing.wall_latency_us_median
                            )
                            vendor_metrics["e2e_wall_latency_us_p95"] = (
                                vendor_e2e_timing.wall_latency_us_p95
                            )
                            vendor_metrics["e2e_device_latency_us_median"] = (
                                vendor_e2e_timing.device_latency_us_median
                            )
                            vendor_metrics["e2e_device_latency_us_p95"] = (
                                vendor_e2e_timing.device_latency_us_p95
                            )
                            vendor_metrics["e2e_decode_tokens_per_second"] = (
                                _decode_tokens_per_second(
                                    shape.batch, vendor_e2e_timing.wall_latency_us_median
                                )
                            )
                            vendor_metrics["e2e_context_tokens_per_second"] = (
                                _context_tokens_per_second(
                                    cache.seq_lens, vendor_e2e_timing.wall_latency_us_median
                                )
                            )
                            cases.append(
                                _timed_case(
                                    subject_kind="baseline",
                                    subject_id="vendor/flashinfer_clustered_page_decode",
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=dimensions,
                                    timing=vendor_timing,
                                    decode_tokens_per_second=vendor_decode_toks,
                                    context_tokens_per_second=vendor_context_toks,
                                    effective_kv_gib_per_second=_effective_kv_gib_per_second(
                                        hbm_bytes_per_decode_token, vendor_decode_toks
                                    ),
                                    speedup_vs={
                                        "torch/reference_clustered_page_decode": (
                                            reference_timing.wall_latency_us_median
                                            / vendor_timing.wall_latency_us_median
                                        )
                                    },
                                    metrics=vendor_metrics,
                                )
                            )
                        except Exception as exc:
                            cases.append(
                                _failed_case(
                                    subject_kind="baseline",
                                    subject_id="vendor/flashinfer_clustered_page_decode",
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                    dimensions=dimensions,
                                    reason=str(exc),
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
