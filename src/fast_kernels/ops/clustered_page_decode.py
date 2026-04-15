from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fast_kernels.native import native_build_info, native_module

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    pass


KV_LAYOUTS = {"bf16_kv", "fp8_kv", "int8_kv"}
KV_FORMAT_IDS = {"bf16_kv": 0, "fp8_kv": 1, "int8_kv": 2}
FORCE_IMPLS = {"auto", "direct", "clustered"}
DEFAULT_DIRECT_PAGE_THRESHOLD = 4
DEFAULT_PDL_PAGE_THRESHOLD = 256
_PLAN_CACHE: dict[tuple[Any, ...], ClusteredPageDecodePlan] = {}


@dataclass(slots=True, frozen=True)
class PagedKVCache:
    key_pages: Any
    value_pages: Any
    page_table: Any
    seq_lens: Any
    key_scales: Any | None
    value_scales: Any | None
    reference_key_pages: Any
    reference_value_pages: Any
    kv_layout: Literal["bf16_kv", "fp8_kv", "int8_kv"]
    page_size: int
    num_kv_heads: int
    head_dim: int
    keys_are_rotated: bool = False
    key_rope_theta: float | None = None

    @property
    def total_pages(self) -> int:
        return int(self.key_pages.shape[0])


@dataclass(slots=True)
class ClusteredPageDecodePlan:
    batch: int
    page_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    kv_layout: Literal["bf16_kv", "fp8_kv", "int8_kv"]
    max_pages: int
    group_tile: int
    q_head_tiles: int
    cluster_size: int
    direct_page_threshold: int
    pdl_page_threshold: int
    launch_mode: Literal["direct", "clustered"]
    run_base_pages_cpu: Any
    run_page_counts_cpu: Any
    run_logical_starts_cpu: Any
    run_last_page_lens_cpu: Any
    request_run_offsets_cpu: Any
    seq_lens_cpu: Any
    _device_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def num_runs(self) -> int:
        return int(self.run_base_pages_cpu.shape[0])

    def device_tensors(self, device: Any) -> dict[str, Any]:
        torch = _require_torch()
        device_key = str(device)
        cached = self._device_cache.get(device_key)
        if cached is not None:
            return cached
        tensors = {
            "run_base_pages": self.run_base_pages_cpu.to(device=device, dtype=torch.int32),
            "run_page_counts": self.run_page_counts_cpu.to(device=device, dtype=torch.int32),
            "run_logical_starts": self.run_logical_starts_cpu.to(device=device, dtype=torch.int32),
            "run_last_page_lens": self.run_last_page_lens_cpu.to(device=device, dtype=torch.int32),
            "request_run_offsets": self.request_run_offsets_cpu.to(
                device=device, dtype=torch.int32
            ),
            "seq_lens": self.seq_lens_cpu.to(device=device, dtype=torch.int32),
        }
        self._device_cache[device_key] = tensors
        return tensors


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without benchmark deps
        raise RuntimeError("PyTorch is required for clustered_page_decode ops") from exc
    return torch


def cuda_clustered_decode_available() -> bool:
    info = native_build_info()
    return bool(info.get("available", False)) and bool(info.get("compiled_with_cuda", False))


def _require_cuda_backend() -> Any:
    if not cuda_clustered_decode_available():
        raise RuntimeError("fast-kernels native module was built without CUDA support")
    return native_module()


def _current_stream_ptr(device: Any) -> int:
    torch = _require_torch()
    return int(torch.cuda.current_stream(device=device).cuda_stream)


def _layout_id(layout: str) -> int:
    try:
        return KV_FORMAT_IDS[layout]
    except KeyError as exc:
        raise ValueError(f"unsupported kv layout: {layout}") from exc


def _require_layout(layout: str) -> Literal["bf16_kv", "fp8_kv", "int8_kv"]:
    if layout not in KV_LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(KV_LAYOUTS)}")
    return layout  # type: ignore[return-value]


def _check_cuda_tensor(
    tensor: Any,
    *,
    name: str,
    dtype: Any,
    ndim: int,
) -> None:
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_cuda_device(*named_tensors: tuple[str, Any]) -> None:
    devices = {tensor.device for _, tensor in named_tensors if tensor is not None}
    if len(devices) <= 1:
        return
    summary = ", ".join(
        f"{name}={tensor.device}" for name, tensor in named_tensors if tensor is not None
    )
    raise ValueError(f"all CUDA tensors must be on the same device: {summary}")


def _cpu_int_tensor(values: list[int]) -> Any:
    torch = _require_torch()
    return torch.tensor(values, dtype=torch.int32)


def _pages_per_request(seq_lens: Any, page_size: int) -> Any:
    return (seq_lens + (page_size - 1)) // page_size


def _build_logical_pages(
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    page_size: int,
) -> tuple[Any, Any]:
    torch = _require_torch()
    batch, num_kv_heads, _, head_dim = keys.shape
    pages_per_request = _pages_per_request(seq_lens, page_size)
    total_pages = int(pages_per_request.sum().item())
    key_pages = torch.zeros(
        (total_pages, num_kv_heads, page_size, head_dim),
        device=keys.device,
        dtype=torch.bfloat16,
    )
    value_pages = torch.zeros_like(key_pages)
    page_cursor = 0
    for request_index in range(batch):
        seq_len = int(seq_lens[request_index].item())
        num_pages = int(pages_per_request[request_index].item())
        for page_slot in range(num_pages):
            token_start = page_slot * page_size
            token_end = min(token_start + page_size, seq_len)
            token_count = token_end - token_start
            key_pages[page_cursor, :, :token_count, :] = keys[
                request_index, :, token_start:token_end, :
            ]
            value_pages[page_cursor, :, :token_count, :] = values[
                request_index, :, token_start:token_end, :
            ]
            page_cursor += 1
    return key_pages.contiguous(), value_pages.contiguous()


def _rotate_dense_keys(
    keys: Any,
    seq_lens: Any,
    rope_theta: float,
) -> Any:
    torch = _require_torch()
    rotated = keys.clone()
    for request_index in range(int(seq_lens.shape[0])):
        seq_len = int(seq_lens[request_index].item())
        if seq_len <= 0:
            continue
        positions = torch.arange(seq_len, device=keys.device, dtype=torch.float32)
        rotated[request_index, :, :seq_len, :] = _apply_llama_rope_torch(
            rotated[request_index, :, :seq_len, :],
            positions,
            rope_theta,
        )
    return rotated


def _build_page_table(
    seq_lens: Any,
    *,
    page_size: int,
    total_pages: int,
    fragment_pages: bool,
    seed: int | None,
    device: Any,
) -> tuple[Any, Any]:
    torch = _require_torch()
    batch = int(seq_lens.shape[0])
    pages_per_request = _pages_per_request(seq_lens, page_size)
    max_pages = int(pages_per_request.max().item())
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32)
    if fragment_pages:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        physical_to_logical = torch.randperm(total_pages, generator=generator, device=device)
        logical_to_physical = torch.empty_like(physical_to_logical)
        logical_to_physical[physical_to_logical] = torch.arange(total_pages, device=device)
    else:
        logical_to_physical = torch.arange(total_pages, device=device)
        physical_to_logical = logical_to_physical

    page_cursor = 0
    for request_index in range(batch):
        num_pages = int(pages_per_request[request_index].item())
        request_pages = logical_to_physical[page_cursor : page_cursor + num_pages]
        page_table[request_index, :num_pages] = request_pages.to(dtype=torch.int32, device="cpu")
        page_cursor += num_pages
    return page_table, physical_to_logical.to(dtype=torch.int64)


def _reorder_physical_pages(logical_pages: Any, physical_to_logical: Any) -> Any:
    if int(physical_to_logical.numel()) == 0:
        return logical_pages
    return logical_pages.index_select(0, physical_to_logical).contiguous()


def pack_paged_kv_bf16(
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    page_size: int,
    fragment_pages: bool = True,
    seed: int | None = None,
    key_rope_theta: float | None = None,
) -> PagedKVCache:
    torch = _require_torch()
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            "keys and values must be 4D tensors shaped [batch, kv_heads, seq, head_dim]"
        )
    if keys.shape != values.shape:
        raise ValueError("keys and values must share the same shape")
    if page_size not in {16, 32}:
        raise ValueError("page_size must be 16 or 32")
    if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != int(keys.shape[0]):
        raise ValueError("seq_lens must be 1D with one length per request")

    keys_bf16 = keys.to(dtype=torch.bfloat16).contiguous()
    values_bf16 = values.to(dtype=torch.bfloat16).contiguous()
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.int32).contiguous()
    if key_rope_theta is not None:
        keys_bf16 = _rotate_dense_keys(keys_bf16, seq_lens_cpu, float(key_rope_theta))
    logical_key_pages, logical_value_pages = _build_logical_pages(
        keys_bf16, values_bf16, seq_lens_cpu, page_size=page_size
    )
    page_table, physical_to_logical = _build_page_table(
        seq_lens_cpu,
        page_size=page_size,
        total_pages=int(logical_key_pages.shape[0]),
        fragment_pages=fragment_pages,
        seed=seed,
        device=keys_bf16.device,
    )
    key_pages = _reorder_physical_pages(logical_key_pages, physical_to_logical)
    value_pages = _reorder_physical_pages(logical_value_pages, physical_to_logical)
    return PagedKVCache(
        key_pages=key_pages,
        value_pages=value_pages,
        page_table=page_table.contiguous(),
        seq_lens=seq_lens_cpu,
        key_scales=None,
        value_scales=None,
        reference_key_pages=key_pages,
        reference_value_pages=value_pages,
        kv_layout="bf16_kv",
        page_size=page_size,
        num_kv_heads=int(keys.shape[1]),
        head_dim=int(keys.shape[3]),
        keys_are_rotated=key_rope_theta is not None,
        key_rope_theta=None if key_rope_theta is None else float(key_rope_theta),
    )


def _quantize_kv_int8(
    key_pages: Any,
    value_pages: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = _require_torch()
    dim_blocks = int(key_pages.shape[-1] // 64)
    key_blocks = key_pages.float().reshape(*key_pages.shape[:-1], dim_blocks, 64)
    key_scales = key_blocks.abs().amax(dim=(-3, -1)) / 127.0
    key_scales = torch.clamp(key_scales, min=1e-6)
    quantized_keys = torch.clamp(
        torch.round(key_blocks / key_scales.unsqueeze(2).unsqueeze(-1)),
        min=-127,
        max=127,
    ).to(torch.int8)
    key_bytes = quantized_keys.reshape_as(key_pages).contiguous()

    value_blocks = value_pages.float().reshape(*value_pages.shape[:-1], dim_blocks, 64)
    value_scales = value_blocks.abs().amax(dim=-1) / 127.0
    value_scales = torch.clamp(value_scales, min=1e-6)
    quantized_values = torch.clamp(
        torch.round(value_blocks / value_scales[..., None]), min=-127, max=127
    ).to(torch.int8)
    value_bytes = quantized_values.reshape_as(value_pages).contiguous()
    return key_bytes, value_bytes, key_scales.contiguous(), value_scales.contiguous()


def _require_fp8_dtype() -> Any:
    torch = _require_torch()
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn")
    return torch.float8_e4m3fn


def _quantize_kv_fp8(
    key_pages: Any,
    value_pages: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = _require_torch()
    float8_dtype = _require_fp8_dtype()
    dim_blocks = int(key_pages.shape[-1] // 64)
    key_blocks = key_pages.float().reshape(*key_pages.shape[:-1], dim_blocks, 64)
    key_scales = key_blocks.abs().amax(dim=(-3, -1)) / 448.0
    key_scales = torch.clamp(key_scales, min=1e-6)
    quantized_keys = (key_blocks / key_scales.unsqueeze(2).unsqueeze(-1)).to(float8_dtype)
    key_bytes = quantized_keys.view(torch.uint8).reshape_as(key_pages).contiguous()

    value_blocks = value_pages.float().reshape(*value_pages.shape[:-1], dim_blocks, 64)
    value_scales = value_blocks.abs().amax(dim=-1) / 448.0
    value_scales = torch.clamp(value_scales, min=1e-6)
    quantized_values = (value_blocks / value_scales[..., None]).to(float8_dtype)
    value_bytes = quantized_values.view(torch.uint8).reshape_as(value_pages).contiguous()
    return key_bytes, value_bytes, key_scales.contiguous(), value_scales.contiguous()


def quantize_paged_kv_int8(
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    page_size: int,
    fragment_pages: bool = True,
    seed: int | None = None,
    key_rope_theta: float | None = None,
) -> PagedKVCache:
    packed = pack_paged_kv_bf16(
        keys,
        values,
        seq_lens,
        page_size=page_size,
        fragment_pages=fragment_pages,
        seed=seed,
        key_rope_theta=key_rope_theta,
    )
    key_bytes, value_bytes, key_scales, value_scales = _quantize_kv_int8(
        packed.reference_key_pages, packed.reference_value_pages
    )
    return PagedKVCache(
        key_pages=key_bytes,
        value_pages=value_bytes,
        page_table=packed.page_table,
        seq_lens=packed.seq_lens,
        key_scales=key_scales,
        value_scales=value_scales,
        reference_key_pages=packed.reference_key_pages,
        reference_value_pages=packed.reference_value_pages,
        kv_layout="int8_kv",
        page_size=packed.page_size,
        num_kv_heads=packed.num_kv_heads,
        head_dim=packed.head_dim,
        keys_are_rotated=packed.keys_are_rotated,
        key_rope_theta=packed.key_rope_theta,
    )


def quantize_paged_kv_fp8(
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    page_size: int,
    fragment_pages: bool = True,
    seed: int | None = None,
    key_rope_theta: float | None = None,
) -> PagedKVCache:
    packed = pack_paged_kv_bf16(
        keys,
        values,
        seq_lens,
        page_size=page_size,
        fragment_pages=fragment_pages,
        seed=seed,
        key_rope_theta=key_rope_theta,
    )
    key_bytes, value_bytes, key_scales, value_scales = _quantize_kv_fp8(
        packed.reference_key_pages, packed.reference_value_pages
    )
    return PagedKVCache(
        key_pages=key_bytes,
        value_pages=value_bytes,
        page_table=packed.page_table,
        seq_lens=packed.seq_lens,
        key_scales=key_scales,
        value_scales=value_scales,
        reference_key_pages=packed.reference_key_pages,
        reference_value_pages=packed.reference_value_pages,
        kv_layout="fp8_kv",
        page_size=packed.page_size,
        num_kv_heads=packed.num_kv_heads,
        head_dim=packed.head_dim,
        keys_are_rotated=packed.keys_are_rotated,
        key_rope_theta=packed.key_rope_theta,
    )


def _default_group_tile(group_size: int) -> int:
    if group_size <= 1:
        return 1
    if group_size <= 2:
        return 2
    if group_size <= 4:
        return 4
    return 8


def _default_cluster_size(max_pages: int, direct_page_threshold: int, group_size: int) -> int:
    torch = _require_torch()
    if max_pages <= direct_page_threshold:
        return 1
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(torch.device("cuda"))
        if major >= 12:
            if max_pages >= 192:
                return 4
            if max_pages >= 128:
                return 2 if group_size > 8 else 4
            if max_pages >= 64:
                return 2
            return 1
        if major >= 9:
            if max_pages >= 128:
                return 4 if group_size >= 8 else 2
            return 2 if group_size >= 8 and max_pages >= 64 else 1
    return 1


def plan_clustered_page_decode(
    *,
    page_table: Any,
    seq_lens: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_layout: str,
    cluster_size: int | None = None,
    direct_page_threshold: int = DEFAULT_DIRECT_PAGE_THRESHOLD,
    pdl_page_threshold: int = DEFAULT_PDL_PAGE_THRESHOLD,
    use_cache: bool = True,
) -> ClusteredPageDecodePlan:
    torch = _require_torch()
    layout = _require_layout(kv_layout)
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.int32).contiguous()
    page_table_cpu = page_table.to(device="cpu", dtype=torch.int32).contiguous()
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

    batch = int(page_table_cpu.shape[0])
    max_pages = int(_pages_per_request(seq_lens_cpu, page_size).max().item())
    group_size = num_q_heads // num_kv_heads
    group_tile = _default_group_tile(group_size)
    q_head_tiles = int(math.ceil(group_size / group_tile))
    effective_cluster_size = (
        _default_cluster_size(max_pages, direct_page_threshold, group_size)
        if cluster_size is None
        else int(cluster_size)
    )
    launch_mode: Literal["direct", "clustered"]
    launch_mode = "clustered" if (group_size > 8 or max_pages > direct_page_threshold) else "direct"

    cache_key = (
        tuple(int(x) for x in seq_lens_cpu.tolist()),
        tuple(tuple(int(v) for v in row.tolist()) for row in page_table_cpu),
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        layout,
        group_tile,
        effective_cluster_size,
        direct_page_threshold,
        pdl_page_threshold,
    )
    if use_cache and cache_key in _PLAN_CACHE:
        return _PLAN_CACHE[cache_key]

    run_base_pages: list[int] = []
    run_page_counts: list[int] = []
    run_logical_starts: list[int] = []
    run_last_page_lens: list[int] = []
    request_run_offsets: list[int] = [0]

    for request_index in range(batch):
        seq_len = int(seq_lens_cpu[request_index].item())
        num_pages = int(math.ceil(seq_len / page_size))
        last_page_len = ((seq_len - 1) % page_size) + 1
        if num_pages == 0:
            request_run_offsets.append(len(run_base_pages))
            continue
        current_base = int(page_table_cpu[request_index, 0].item())
        current_logical = 0
        current_count = 1
        for logical_page in range(1, num_pages):
            physical_page = int(page_table_cpu[request_index, logical_page].item())
            if physical_page == current_base + current_count:
                current_count += 1
                continue
            run_base_pages.append(current_base)
            run_page_counts.append(current_count)
            run_logical_starts.append(current_logical)
            run_last_page_lens.append(page_size)
            current_base = physical_page
            current_logical = logical_page
            current_count = 1
        run_base_pages.append(current_base)
        run_page_counts.append(current_count)
        run_logical_starts.append(current_logical)
        run_last_page_lens.append(last_page_len)
        request_run_offsets.append(len(run_base_pages))

    plan = ClusteredPageDecodePlan(
        batch=batch,
        page_size=page_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=layout,
        max_pages=max_pages,
        group_tile=group_tile,
        q_head_tiles=q_head_tiles,
        cluster_size=effective_cluster_size,
        direct_page_threshold=direct_page_threshold,
        pdl_page_threshold=pdl_page_threshold,
        launch_mode=launch_mode,
        run_base_pages_cpu=_cpu_int_tensor(run_base_pages),
        run_page_counts_cpu=_cpu_int_tensor(run_page_counts),
        run_logical_starts_cpu=_cpu_int_tensor(run_logical_starts),
        run_last_page_lens_cpu=_cpu_int_tensor(run_last_page_lens),
        request_run_offsets_cpu=_cpu_int_tensor(request_run_offsets),
        seq_lens_cpu=seq_lens_cpu,
    )
    if use_cache:
        _PLAN_CACHE[cache_key] = plan
    return plan


def _device_or_zero(tensor: Any | None) -> int:
    return 0 if tensor is None else int(tensor.data_ptr())


def clustered_page_decode(
    query: Any,
    cache: PagedKVCache,
    *,
    plan: ClusteredPageDecodePlan | None = None,
    output: Any | None = None,
    softmax_scale: float | None = None,
    rope_theta: float = 10000.0,
    force_impl: str = "auto",
) -> Any:
    torch = _require_torch()
    native = _require_cuda_backend()
    if force_impl not in FORCE_IMPLS:
        raise ValueError(f"force_impl must be one of {sorted(FORCE_IMPLS)}")
    _check_cuda_tensor(query, name="query", dtype=torch.bfloat16, ndim=3)
    _check_same_cuda_device(
        ("query", query),
        ("key_pages", cache.key_pages),
        ("value_pages", cache.value_pages),
    )
    batch, num_q_heads, head_dim = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]))
    if num_q_heads % cache.num_kv_heads != 0:
        raise ValueError("query head count must be divisible by cache.num_kv_heads")
    if head_dim != cache.head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    if plan is None:
        plan = plan_clustered_page_decode(
            page_table=cache.page_table,
            seq_lens=cache.seq_lens,
            num_q_heads=num_q_heads,
            num_kv_heads=cache.num_kv_heads,
            head_dim=head_dim,
            page_size=cache.page_size,
            kv_layout=cache.kv_layout,
        )

    effective_cluster_size = plan.cluster_size
    if effective_cluster_size < 1:
        effective_cluster_size = 1
    group_size = num_q_heads // cache.num_kv_heads
    use_direct_impl = force_impl == "direct" or (
        force_impl == "auto" and plan.launch_mode == "direct"
    )
    if use_direct_impl and group_size > 8:
        raise ValueError(
            'force_impl="direct" only supports group_size <= 8; use auto or force_impl="clustered"'
        )
    if output is None:
        output = torch.empty_like(query)
    _check_cuda_tensor(output, name="output", dtype=torch.bfloat16, ndim=3)
    descriptor_tensors = plan.device_tensors(query.device)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    if cache.keys_are_rotated:
        if cache.key_rope_theta is None:
            raise ValueError("rotated key caches must record key_rope_theta")
        if not math.isclose(
            float(cache.key_rope_theta),
            float(rope_theta),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("rope_theta must match the key_rope_theta used to build the cache")
    if cache.key_scales is not None or cache.value_scales is not None:
        _check_same_cuda_device(
            ("query", query),
            ("key_scales", cache.key_scales),
            ("value_scales", cache.value_scales),
        )
    native.clustered_page_decode_forward(
        query.data_ptr(),
        cache.key_pages.data_ptr(),
        cache.value_pages.data_ptr(),
        _device_or_zero(cache.key_scales),
        _device_or_zero(cache.value_scales),
        descriptor_tensors["run_base_pages"].data_ptr(),
        descriptor_tensors["run_page_counts"].data_ptr(),
        descriptor_tensors["run_logical_starts"].data_ptr(),
        descriptor_tensors["run_last_page_lens"].data_ptr(),
        descriptor_tensors["request_run_offsets"].data_ptr(),
        descriptor_tensors["seq_lens"].data_ptr(),
        output.data_ptr(),
        batch,
        plan.num_runs,
        num_q_heads,
        cache.num_kv_heads,
        head_dim,
        cache.page_size,
        plan.group_tile,
        int(not use_direct_impl),
        effective_cluster_size,
        _layout_id(cache.kv_layout),
        int(cache.keys_are_rotated),
        float(scale),
        float(rope_theta),
        _current_stream_ptr(query.device),
    )
    return output


def _apply_llama_rope_torch(x: Any, positions: Any, rope_theta: float) -> Any:
    torch = _require_torch()
    head_dim = int(x.shape[-1])
    pair_count = head_dim // 2
    inv_freq = rope_theta ** (
        -2.0 * torch.arange(pair_count, device=x.device, dtype=torch.float32) / head_dim
    )
    angles = positions.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    pair_view = x.to(torch.float32).reshape(*x.shape[:-1], pair_count, 2)
    out_even = (pair_view[..., 0] * cos) - (pair_view[..., 1] * sin)
    out_odd = (pair_view[..., 0] * sin) + (pair_view[..., 1] * cos)
    return torch.stack((out_even, out_odd), dim=-1).reshape_as(x)


def materialize_reference_dense_kv(cache: PagedKVCache) -> tuple[Any, Any]:
    torch = _require_torch()
    batch = int(cache.page_table.shape[0])
    max_seq_len = int(cache.seq_lens.max().item())
    dense_keys = torch.zeros(
        (batch, cache.num_kv_heads, max_seq_len, cache.head_dim),
        device=cache.reference_key_pages.device,
        dtype=torch.bfloat16,
    )
    dense_values = torch.zeros_like(dense_keys)
    for request_index in range(batch):
        seq_len = int(cache.seq_lens[request_index].item())
        num_pages = int(math.ceil(seq_len / cache.page_size))
        for page_slot in range(num_pages):
            physical_page = int(cache.page_table[request_index, page_slot].item())
            token_start = page_slot * cache.page_size
            token_end = min(token_start + cache.page_size, seq_len)
            token_count = token_end - token_start
            dense_keys[request_index, :, token_start:token_end, :] = cache.reference_key_pages[
                physical_page, :, :token_count, :
            ]
            dense_values[request_index, :, token_start:token_end, :] = cache.reference_value_pages[
                physical_page, :, :token_count, :
            ]
    return dense_keys, dense_values


def reference_clustered_page_decode(
    query: Any,
    cache: PagedKVCache,
    *,
    softmax_scale: float | None = None,
    rope_theta: float = 10000.0,
    output: Any | None = None,
) -> Any:
    torch = _require_torch()
    if query.ndim != 3:
        raise ValueError("query must be 3D [batch, q_heads, head_dim]")
    batch, num_q_heads, head_dim = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]))
    if head_dim != cache.head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    group_size = num_q_heads // cache.num_kv_heads
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    dense_keys, dense_values = materialize_reference_dense_kv(cache)
    reference = torch.empty_like(query) if output is None else output
    for request_index in range(batch):
        seq_len = int(cache.seq_lens[request_index].item())
        positions = torch.arange(seq_len, device=query.device, dtype=torch.float32)
        rotated_query = _apply_llama_rope_torch(
            query[request_index],
            torch.full((num_q_heads,), seq_len - 1, device=query.device),
            rope_theta,
        )
        for q_head_index in range(num_q_heads):
            kv_head_index = q_head_index // group_size
            key_slice = dense_keys[request_index, kv_head_index, :seq_len, :]
            rotated_keys = (
                key_slice
                if cache.keys_are_rotated
                else _apply_llama_rope_torch(key_slice, positions, rope_theta)
            )
            logits = (
                torch.matmul(
                    rotated_keys.to(torch.float32),
                    rotated_query[q_head_index].to(torch.float32),
                )
                * scale
            )
            probs = torch.softmax(logits, dim=0)
            reference[request_index, q_head_index] = torch.matmul(
                probs.to(torch.float32),
                dense_values[request_index, kv_head_index, :seq_len, :].to(torch.float32),
            ).to(torch.bfloat16)
    return reference


def estimate_page_decode_metrics(
    cache: PagedKVCache,
    plan: ClusteredPageDecodePlan,
) -> dict[str, float]:
    element_bytes = {"bf16_kv": 2.0, "fp8_kv": 1.0, "int8_kv": 1.0}[cache.kv_layout]
    avg_tokens = float(cache.seq_lens.to(dtype=_require_torch().float32).mean().item())
    group_size = plan.num_q_heads // plan.num_kv_heads
    reuse_factor = float(group_size) if plan.launch_mode == "direct" else float(plan.q_head_tiles)
    kv_bytes = avg_tokens * cache.num_kv_heads * cache.head_dim * 2.0 * element_bytes * reuse_factor
    scale_bytes = 0.0
    if cache.kv_layout != "bf16_kv":
        scale_bytes = (
            avg_tokens * cache.num_kv_heads * (cache.head_dim / 64.0) * 4.0 * reuse_factor
        )
    dsm_bytes = 0.0
    if plan.launch_mode == "clustered" and plan.cluster_size > 1:
        dsm_bytes = float(plan.cluster_size * (cache.head_dim * 4) * group_size)
    return {
        "hbm_bytes_per_token": kv_bytes + scale_bytes,
        "workspace_bytes_per_token": 0.0,
        "dsm_bytes_per_token": dsm_bytes,
    }
