from __future__ import annotations

import hashlib
import math
from functools import partial
from typing import Any

from fast_kernels.benchmarking.clustered_page_decode import (
    E2E_TIMING_ITERS,
    E2E_WARMUP_ITERS,
    SubjectKind,
    _context_tokens_per_second,
    _decode_tokens_per_second,
    _effective_kv_gib_per_second,
    _failed_case,
    _max_abs_diff,
    _require_torch,
    _skipped_case,
    _time_callable,
    _timed_case,
)
from fast_kernels.ops import (
    cuda_prefix_union_decode_available,
    estimate_prefix_union_decode_metrics,
    pack_paged_kv_bf16,
    plan_prefix_union_decode,
    prefix_union_decode,
    quantize_paged_kv_fp8,
    quantize_paged_kv_int8,
    reference_prefix_union_decode,
)
from fast_kernels.schemas import BenchmarkCase, BenchmarkSuite, ShapeCase

KERNEL_TO_IMPL = {
    "decode/prefix_union_decode_auto": "auto",
    "decode/prefix_union_decode_union": "union",
    "decode/prefix_union_decode_fallback": "fallback",
}
ROPE_THETA = 10000.0


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
                str(shape.require_dimension("shared_prefix_pages")),
                str(shape.require_dimension("prefix_group_size")),
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
    shared_prefix_pages = shape.require_dimension("shared_prefix_pages")
    prefix_group_size = shape.require_dimension("prefix_group_size")
    seed = _case_seed(shape, layout)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    shared_prefix_tokens = shared_prefix_pages * page_size
    min_seq_len = min(max_seq_len, max(page_size, shared_prefix_tokens + 1))
    query = (0.25 * torch.randn((batch, num_q_heads, head_dim), device=device)).to(torch.bfloat16)
    keys = (0.35 * torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device)).to(
        torch.bfloat16
    )
    values = (0.35 * torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device)).to(
        torch.bfloat16
    )
    seq_lens = torch.randint(
        low=min_seq_len,
        high=max_seq_len + 1,
        size=(batch,),
        dtype=torch.int32,
    ).contiguous()

    if shared_prefix_tokens > 0:
        for group_start in range(0, batch, prefix_group_size):
            group_end = min(batch, group_start + prefix_group_size)
            if group_end - group_start < 2:
                continue
            leader = group_start
            for follower in range(group_start + 1, group_end):
                keys[follower, :, :shared_prefix_tokens, :] = keys[
                    leader, :, :shared_prefix_tokens, :
                ]
                values[follower, :, :shared_prefix_tokens, :] = values[
                    leader, :, :shared_prefix_tokens, :
                ]
    return query.contiguous(), keys.contiguous(), values.contiguous(), seq_lens


def _build_cache(keys: Any, values: Any, seq_lens: Any, *, page_size: int, layout: str) -> Any:
    common_kwargs = dict(
        page_size=page_size,
        fragment_pages=True,
        seed=11,
        key_rope_theta=ROPE_THETA,
        deduplicate_identical_pages=True,
    )
    if layout == "bf16_kv":
        return pack_paged_kv_bf16(keys, values, seq_lens, **common_kwargs)
    if layout == "fp8_kv":
        return quantize_paged_kv_fp8(keys, values, seq_lens, **common_kwargs)
    if layout == "int8_kv":
        return quantize_paged_kv_int8(keys, values, seq_lens, **common_kwargs)
    raise ValueError(f"unsupported layout: {layout}")


def _make_repo_e2e_decode_fn(
    query: Any,
    cache: Any,
    *,
    force_impl: str,
    softmax_scale: float,
) -> Any:
    def _run() -> Any:
        plan = plan_prefix_union_decode(
            page_table=cache.page_table,
            seq_lens=cache.seq_lens,
            num_q_heads=int(query.shape[1]),
            num_kv_heads=cache.num_kv_heads,
            head_dim=cache.head_dim,
            page_size=cache.page_size,
            kv_layout=cache.kv_layout,
            keys_are_rotated=cache.keys_are_rotated,
            use_cache=False,
        )
        return prefix_union_decode(
            query,
            cache,
            plan=plan,
            softmax_scale=softmax_scale,
            rope_theta=ROPE_THETA,
            force_impl=force_impl,
        )

    return _run


def run_prefix_union_decode_suite(
    suite: BenchmarkSuite,
) -> tuple[list[BenchmarkCase], list[str]]:
    torch = _require_torch()
    notes = [
        "Prefix-union decode benchmarks build deduplicated paged KV caches with shared physical prefixes.",
        "Union mode requires pre-rotated key pages and a Blackwell-class GPU.",
        "Fallback timings route through the existing clustered_page_decode implementation.",
    ]
    if torch is None:
        notes.append("PyTorch is unavailable, so all prefix_union_decode cases were skipped.")
        skipped_cases: list[BenchmarkCase] = []
        for subject_kind, subject_ids in (
            ("kernel", suite.kernels.ids),
            ("baseline", suite.baselines.ids),
        ):
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
                                    reason="PyTorch is required for prefix_union_decode benchmarks.",
                                )
                            )
        return skipped_cases, notes

    if not torch.cuda.is_available() or not cuda_prefix_union_decode_available():
        notes.append("CUDA or the native CUDA extension is unavailable, so all cases were skipped.")
        skipped_cases = []
        for subject_kind, subject_ids in (
            ("kernel", suite.kernels.ids),
            ("baseline", suite.baselines.ids),
        ):
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
                                    reason="CUDA, PyTorch, and a CUDA-enabled native build are required.",
                                )
                            )
        return skipped_cases, notes

    cases: list[BenchmarkCase] = []
    for dtype in suite.dtypes:
        if dtype != "bf16":
            raise ValueError("prefix_union_decode suites currently only support dtype=bf16")
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
                    plan = plan_prefix_union_decode(
                        page_table=cache.page_table,
                        seq_lens=cache.seq_lens,
                        num_q_heads=shape.require_dimension("num_q_heads"),
                        num_kv_heads=shape.require_dimension("num_kv_heads"),
                        head_dim=shape.require_dimension("head_dim"),
                        page_size=shape.require_dimension("page_size"),
                        kv_layout=layout,
                        keys_are_rotated=cache.keys_are_rotated,
                    )
                except Exception as exc:
                    reason = f"cache build or planning failed: {exc}"
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

                softmax_scale = 1.0 / math.sqrt(cache.head_dim)
                reference_output = reference_prefix_union_decode(
                    query,
                    cache,
                    rope_theta=ROPE_THETA,
                )
                reference_fn = partial(
                    reference_prefix_union_decode,
                    query,
                    cache,
                    rope_theta=ROPE_THETA,
                )
                reference_timing = _time_callable(reference_fn, torch)
                reference_decode_toks = _decode_tokens_per_second(
                    shape.batch,
                    reference_timing.wall_latency_us_median,
                )
                reference_context_toks = _context_tokens_per_second(
                    cache.seq_lens,
                    reference_timing.wall_latency_us_median,
                )
                reference_metrics = estimate_prefix_union_decode_metrics(cache, plan)
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
                hbm_bytes_per_decode_token = reference_metrics["hbm_bytes_per_token"]

                for kernel_id in suite.kernels.ids:
                    force_impl = KERNEL_TO_IMPL[kernel_id]
                    try:
                        kernel_output = prefix_union_decode(
                            query,
                            cache,
                            plan=plan,
                            rope_theta=ROPE_THETA,
                            force_impl=force_impl,
                        )
                        torch.testing.assert_close(
                            kernel_output,
                            reference_output,
                            atol=suite.tolerances.atol,
                            rtol=suite.tolerances.rtol,
                        )
                        fn = partial(
                            prefix_union_decode,
                            query,
                            cache,
                            plan=plan,
                            rope_theta=ROPE_THETA,
                            force_impl=force_impl,
                        )
                        timing = _time_callable(fn, torch)
                        e2e_timing = _time_callable(
                            _make_repo_e2e_decode_fn(
                                query,
                                cache,
                                force_impl=force_impl,
                                softmax_scale=softmax_scale,
                            ),
                            torch,
                            warmup_iters=E2E_WARMUP_ITERS,
                            timing_iters=E2E_TIMING_ITERS,
                        )
                        decode_toks = _decode_tokens_per_second(
                            shape.batch,
                            timing.wall_latency_us_median,
                        )
                        context_toks = _context_tokens_per_second(
                            cache.seq_lens,
                            timing.wall_latency_us_median,
                        )
                        metrics = estimate_prefix_union_decode_metrics(cache, plan)
                        metrics["max_abs_error"] = _max_abs_diff(kernel_output, reference_output)
                        metrics["e2e_wall_latency_us_median"] = e2e_timing.wall_latency_us_median
                        metrics["e2e_wall_latency_us_p95"] = e2e_timing.wall_latency_us_p95
                        metrics["e2e_device_latency_us_median"] = (
                            e2e_timing.device_latency_us_median
                        )
                        metrics["e2e_device_latency_us_p95"] = e2e_timing.device_latency_us_p95
                        metrics["e2e_decode_tokens_per_second"] = _decode_tokens_per_second(
                            shape.batch,
                            e2e_timing.wall_latency_us_median,
                        )
                        metrics["e2e_context_tokens_per_second"] = _context_tokens_per_second(
                            cache.seq_lens,
                            e2e_timing.wall_latency_us_median,
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
                                    hbm_bytes_per_decode_token,
                                    decode_toks,
                                ),
                                speedup_vs={
                                    "torch/reference_prefix_union_decode": (
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
                        subject_id="torch/reference_prefix_union_decode",
                        dtype=dtype,
                        layout=layout,
                        shape_name=shape.name,
                        dimensions=dimensions,
                        timing=reference_timing,
                        decode_tokens_per_second=reference_decode_toks,
                        context_tokens_per_second=reference_context_toks,
                        effective_kv_gib_per_second=_effective_kv_gib_per_second(
                            hbm_bytes_per_decode_token,
                            reference_decode_toks,
                        ),
                        speedup_vs={},
                        metrics=reference_metrics,
                    )
                )

    return cases, notes
