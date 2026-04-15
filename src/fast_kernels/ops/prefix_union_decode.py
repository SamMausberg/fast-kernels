from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from fast_kernels.ops.clustered_page_decode import (
    ClusteredPageDecodePlan,
    PagedKVCache,
    _check_cuda_tensor,
    _check_same_cuda_device,
    _current_stream_ptr,
    _device_or_zero,
    _layout_id,
    _require_cuda_backend,
    _require_layout,
    _require_torch,
    clustered_page_decode,
    cuda_clustered_decode_available,
    estimate_page_decode_metrics,
    plan_clustered_page_decode,
    reference_clustered_page_decode,
)

FORCE_IMPLS = {"auto", "union", "fallback"}
DEFAULT_MIN_SHARED_PAGES = 2
DEFAULT_MIN_CONSUMERS = 2
_PLAN_CACHE: dict[tuple[Any, ...], "PrefixUnionDecodePlan"] = {}


@dataclass(slots=True)
class PrefixUnionDecodePlan:
    batch: int
    page_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    kv_layout: Literal["bf16_kv", "fp8_kv", "int8_kv"]
    cluster_size: int
    min_shared_pages: int
    min_consumers: int
    launch_mode: Literal["union", "fallback"]
    estimated_saved_hbm_bytes: int
    tasks_cpu: Any
    shared_pages_cpu: Any
    tail_pages_cpu: Any
    consumers_cpu: Any
    fallback_plan: ClusteredPageDecodePlan
    _device_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def num_tasks(self) -> int:
        return int(self.tasks_cpu.shape[0])

    def device_tensors(self, device: Any) -> dict[str, Any]:
        torch = _require_torch()
        device_key = str(device)
        cached = self._device_cache.get(device_key)
        if cached is not None:
            return cached
        tensors = {
            "tasks": self.tasks_cpu.to(device=device, dtype=torch.int32),
            "shared_pages": self.shared_pages_cpu.to(device=device, dtype=torch.int32),
            "tail_pages": self.tail_pages_cpu.to(device=device, dtype=torch.int32),
            "consumers": self.consumers_cpu.to(device=device, dtype=torch.int32),
            "scheduler_counter": torch.zeros((1,), device=device, dtype=torch.int32),
        }
        self._device_cache[device_key] = tensors
        return tensors


def cuda_prefix_union_decode_available() -> bool:
    if not cuda_clustered_decode_available():
        return False
    native = _require_cuda_backend()
    return hasattr(native, "prefix_union_decode_forward")


def _cpu_struct_tensor(rows: list[list[int]], width: int) -> Any:
    torch = _require_torch()
    if not rows:
        return torch.empty((0, width), dtype=torch.int32)
    return torch.tensor(rows, dtype=torch.int32).reshape(len(rows), width).contiguous()


def _request_page_lists(
    page_table_cpu: Any,
    seq_lens_cpu: Any,
    page_size: int,
) -> tuple[list[tuple[int, ...]], list[list[int]]]:
    request_pages: list[tuple[int, ...]] = []
    request_valid_tokens: list[list[int]] = []
    batch = int(page_table_cpu.shape[0])
    for request_index in range(batch):
        seq_len = int(seq_lens_cpu[request_index].item())
        num_pages = int(((seq_len + (page_size - 1)) // page_size))
        pages = tuple(int(v) for v in page_table_cpu[request_index, :num_pages].tolist())
        valid_tokens = [
            page_size if page_slot < (num_pages - 1) else max(0, seq_len - (page_slot * page_size))
            for page_slot in range(num_pages)
        ]
        request_pages.append(pages)
        request_valid_tokens.append(valid_tokens)
    return request_pages, request_valid_tokens


def _best_shared_prefixes(
    request_pages: list[tuple[int, ...]],
    *,
    min_shared_pages: int,
    min_consumers: int,
) -> list[tuple[int, ...]]:
    prefix_counts: dict[tuple[int, ...], int] = {}
    for pages in request_pages:
        for prefix_len in range(min_shared_pages, len(pages) + 1):
            prefix = pages[:prefix_len]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    best_prefixes: list[tuple[int, ...]] = []
    for pages in request_pages:
        chosen: tuple[int, ...] = ()
        for prefix_len in range(len(pages), min_shared_pages - 1, -1):
            prefix = pages[:prefix_len]
            if prefix_counts.get(prefix, 0) >= min_consumers:
                chosen = prefix
                break
        best_prefixes.append(chosen)
    return best_prefixes


def _default_cluster_size(max_consumers: int, max_shared_pages: int) -> int:
    _ = max_consumers
    _ = max_shared_pages
    return 1


def _shared_page_bytes(page_size: int, head_dim: int, kv_layout: str) -> int:
    element_bytes = {"bf16_kv": 2, "fp8_kv": 1, "int8_kv": 1}[kv_layout]
    kv_bytes = 2 * page_size * head_dim * element_bytes
    if kv_layout == "bf16_kv":
        return kv_bytes
    dim_blocks = head_dim // 64
    scale_bytes = (dim_blocks * 4) + (page_size * dim_blocks * 4)
    return kv_bytes + scale_bytes


def _is_blackwell_device() -> bool:
    torch = _require_torch()
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(torch.device("cuda"))
    return major >= 12


def plan_prefix_union_decode(
    *,
    page_table: Any,
    seq_lens: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_layout: str,
    keys_are_rotated: bool,
    cluster_size: int | None = None,
    min_shared_pages: int = DEFAULT_MIN_SHARED_PAGES,
    min_consumers: int = DEFAULT_MIN_CONSUMERS,
    use_cache: bool = True,
) -> PrefixUnionDecodePlan:
    torch = _require_torch()
    layout = _require_layout(kv_layout)
    page_table_cpu = page_table.to(device="cpu", dtype=torch.int32).contiguous()
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.int32).contiguous()
    if page_table_cpu.ndim != 2:
        raise ValueError("page_table must be a 2D int tensor")
    if seq_lens_cpu.ndim != 1 or int(seq_lens_cpu.shape[0]) != int(page_table_cpu.shape[0]):
        raise ValueError("seq_lens must be a 1D tensor with one entry per request")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if head_dim not in {64, 128}:
        raise ValueError("head_dim must be 64 or 128")
    if page_size not in {16, 32}:
        raise ValueError("page_size must be 16 or 32")
    if min_shared_pages < 1:
        raise ValueError("min_shared_pages must be >= 1")
    if min_consumers < 2:
        raise ValueError("min_consumers must be >= 2")

    batch = int(page_table_cpu.shape[0])
    request_pages, request_valid_tokens = _request_page_lists(
        page_table_cpu,
        seq_lens_cpu,
        page_size,
    )
    best_prefixes = _best_shared_prefixes(
        request_pages,
        min_shared_pages=min_shared_pages,
        min_consumers=min_consumers,
    )
    group_size = num_q_heads // num_kv_heads

    cache_key = (
        tuple(int(x) for x in seq_lens_cpu.tolist()),
        tuple(tuple(int(v) for v in row.tolist()) for row in page_table_cpu),
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        layout,
        int(keys_are_rotated),
        min_shared_pages,
        min_consumers,
        cluster_size,
    )
    if use_cache and cache_key in _PLAN_CACHE:
        return _PLAN_CACHE[cache_key]

    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for request_index, prefix in enumerate(best_prefixes):
        for kv_head_index in range(num_kv_heads):
            groups.setdefault((kv_head_index, prefix), []).append(request_index)

    tasks_rows: list[list[int]] = []
    shared_rows: list[list[int]] = []
    tail_rows: list[list[int]] = []
    consumer_rows: list[list[int]] = []
    estimated_saved_hbm_bytes = 0
    max_consumers = 0
    max_shared_pages = 0
    page_bytes = _shared_page_bytes(page_size, head_dim, layout)

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (len(item[0][1]) * len(item[1]), len(item[1]), item[0][0]),
        reverse=True,
    )
    for (kv_head_index, prefix), request_indices in sorted_groups:
        shared_page_offset = len(shared_rows)
        prefix_len = len(prefix)
        if prefix_len > 0:
            exemplar = request_indices[0]
            for page_slot, page_id in enumerate(prefix):
                shared_rows.append(
                    [
                        int(page_id),
                        int(request_valid_tokens[exemplar][page_slot]),
                    ]
                )

        consumer_offset = len(consumer_rows)
        task_consumers = 0
        for request_index in request_indices:
            seq_len = int(seq_lens_cpu[request_index].item())
            pages = request_pages[request_index]
            q_head_base = kv_head_index * group_size
            for q_head_offset in range(group_size):
                q_head_index = q_head_base + q_head_offset
                tail_page_offset = len(tail_rows)
                for logical_page_index in range(prefix_len, len(pages)):
                    packed_meta = int(request_valid_tokens[request_index][logical_page_index]) | (
                        int(logical_page_index) << 16
                    )
                    tail_rows.append(
                        [
                            int(pages[logical_page_index]),
                            packed_meta,
                        ]
                    )
                consumer_rows.append(
                    [
                        (request_index * num_q_heads) + q_head_index,
                        (request_index * num_q_heads) + q_head_index,
                        tail_page_offset,
                        max(0, seq_len - 1),
                        int(len(pages) - prefix_len),
                        0,
                        0,
                        0,
                    ]
                )
                task_consumers += 1

        tasks_rows.append(
            [
                shared_page_offset,
                prefix_len,
                consumer_offset,
                task_consumers,
                kv_head_index,
                0,
                0,
                0,
            ]
        )
        max_consumers = max(max_consumers, task_consumers)
        max_shared_pages = max(max_shared_pages, prefix_len)
        if prefix_len > 0 and task_consumers > 1:
            estimated_saved_hbm_bytes += (task_consumers - 1) * prefix_len * page_bytes

    effective_cluster_size = (
        _default_cluster_size(max_consumers, max_shared_pages)
        if cluster_size is None
        else int(cluster_size)
    )
    if effective_cluster_size not in {1, 2, 4, 8}:
        raise ValueError("cluster_size must be one of {1, 2, 4, 8}")

    fallback_plan = plan_clustered_page_decode(
        page_table=page_table_cpu,
        seq_lens=seq_lens_cpu,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        kv_layout=layout,
        cluster_size=cluster_size,
        use_cache=use_cache,
    )
    launch_mode: Literal["union", "fallback"] = "union"
    if not keys_are_rotated or not _is_blackwell_device() or estimated_saved_hbm_bytes <= 0:
        launch_mode = "fallback"

    plan = PrefixUnionDecodePlan(
        batch=batch,
        page_size=page_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=layout,
        cluster_size=effective_cluster_size,
        min_shared_pages=min_shared_pages,
        min_consumers=min_consumers,
        launch_mode=launch_mode,
        estimated_saved_hbm_bytes=estimated_saved_hbm_bytes,
        tasks_cpu=_cpu_struct_tensor(tasks_rows, 8),
        shared_pages_cpu=_cpu_struct_tensor(shared_rows, 2),
        tail_pages_cpu=_cpu_struct_tensor(tail_rows, 2),
        consumers_cpu=_cpu_struct_tensor(consumer_rows, 8),
        fallback_plan=fallback_plan,
    )
    if use_cache:
        _PLAN_CACHE[cache_key] = plan
    return plan


def prefix_union_decode(
    query: Any,
    cache: PagedKVCache,
    *,
    plan: PrefixUnionDecodePlan | None = None,
    output: Any | None = None,
    softmax_scale: float | None = None,
    rope_theta: float = 10000.0,
    force_impl: str = "auto",
) -> Any:
    torch = _require_torch()
    if force_impl not in FORCE_IMPLS:
        raise ValueError(f"force_impl must be one of {sorted(FORCE_IMPLS)}")
    _check_cuda_tensor(query, name="query", dtype=torch.bfloat16, ndim=3)
    _check_same_cuda_device(
        ("query", query),
        ("key_pages", cache.key_pages),
        ("value_pages", cache.value_pages),
        ("key_scales", cache.key_scales),
        ("value_scales", cache.value_scales),
    )

    batch, num_q_heads, head_dim = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]))
    if num_q_heads % cache.num_kv_heads != 0:
        raise ValueError("query head count must be divisible by cache.num_kv_heads")
    if head_dim != cache.head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    if plan is None:
        plan = plan_prefix_union_decode(
            page_table=cache.page_table,
            seq_lens=cache.seq_lens,
            num_q_heads=num_q_heads,
            num_kv_heads=cache.num_kv_heads,
            head_dim=head_dim,
            page_size=cache.page_size,
            kv_layout=cache.kv_layout,
            keys_are_rotated=cache.keys_are_rotated,
        )
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if output is None:
        output = torch.empty_like(query)
    _check_cuda_tensor(output, name="output", dtype=torch.bfloat16, ndim=3)

    use_union_impl = force_impl == "union" or (
        force_impl == "auto" and plan.launch_mode == "union"
    )
    if use_union_impl:
        if not cache.keys_are_rotated:
            raise ValueError("prefix_union_decode requires pre-rotated key pages for the union path")
        if cache.key_rope_theta is None:
            raise ValueError("rotated key caches must record key_rope_theta")
        if not math.isclose(
            float(cache.key_rope_theta),
            float(rope_theta),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("rope_theta must match the key_rope_theta used to build the cache")
        if not _is_blackwell_device():
            raise RuntimeError("prefix_union_decode requires a Blackwell-class GPU for union mode")
        native = _require_cuda_backend()
        if not hasattr(native, "prefix_union_decode_forward"):
            raise RuntimeError("native module does not expose prefix_union_decode_forward")
        descriptor_tensors = plan.device_tensors(query.device)
        descriptor_tensors["scheduler_counter"].zero_()
        native.prefix_union_decode_forward(
            query.data_ptr(),
            cache.key_pages.data_ptr(),
            cache.value_pages.data_ptr(),
            _device_or_zero(cache.key_scales),
            _device_or_zero(cache.value_scales),
            descriptor_tensors["tasks"].data_ptr(),
            descriptor_tensors["shared_pages"].data_ptr(),
            descriptor_tensors["tail_pages"].data_ptr(),
            descriptor_tensors["consumers"].data_ptr(),
            descriptor_tensors["scheduler_counter"].data_ptr(),
            output.data_ptr(),
            plan.num_tasks,
            int(cache.key_pages.shape[0]),
            num_q_heads,
            cache.num_kv_heads,
            head_dim,
            cache.page_size,
            plan.cluster_size,
            _layout_id(cache.kv_layout),
            int(cache.keys_are_rotated),
            float(scale),
            float(rope_theta),
            _current_stream_ptr(query.device),
        )
        return output

    return clustered_page_decode(
        query,
        cache,
        plan=plan.fallback_plan,
        output=output,
        softmax_scale=softmax_scale,
        rope_theta=rope_theta,
        force_impl="auto",
    )


def estimate_prefix_union_decode_metrics(
    cache: PagedKVCache,
    plan: PrefixUnionDecodePlan,
) -> dict[str, float]:
    metrics = dict(estimate_page_decode_metrics(cache, plan.fallback_plan))
    saved_per_token = float(plan.estimated_saved_hbm_bytes) / max(1.0, float(plan.batch))
    metrics["hbm_bytes_per_token"] = max(0.0, metrics["hbm_bytes_per_token"] - saved_per_token)
    metrics["workspace_bytes_per_token"] = 0.0
    metrics["dsm_bytes_per_token"] = float(
        plan.cluster_size * plan.page_size * plan.head_dim * 4 * 2
    )
    metrics["estimated_saved_hbm_bytes"] = float(plan.estimated_saved_hbm_bytes)
    return metrics


def reference_prefix_union_decode(
    query: Any,
    cache: PagedKVCache,
    *,
    softmax_scale: float | None = None,
    rope_theta: float = 10000.0,
    output: Any | None = None,
) -> Any:
    return reference_clustered_page_decode(
        query,
        cache,
        softmax_scale=softmax_scale,
        rope_theta=rope_theta,
        output=output,
    )
