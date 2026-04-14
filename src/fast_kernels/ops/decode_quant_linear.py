from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fast_kernels.native import native_build_info, native_module

if TYPE_CHECKING:  # pragma: no cover - imported only for static checking
    import torch


_ARC_GROUP_SUMS_CACHE: dict[tuple[int, int, int], Any] = {}
_ARC_SPLIT_K_PARTIALS_CACHE: dict[tuple[int, int, int, int], Any] = {}
_ARC_WEIGHT_BUFFER_CACHE: dict[tuple[int, int, int], Any] = {}
_ARC_WORKSPACE_CACHE: dict[tuple[int, int], Any] = {}
_ARC_SPLIT_K_TUNING_CACHE: dict[tuple[int, int, int, int, int], int] = {}
_ARC_IMPL_TUNING_CACHE: dict[tuple[int, int, int, int, int, int | None], str] = {}

_ARC_IMPL_ENV = "FK_ARC_W4A16_IMPL"
_ARC_IMPL_AUTO = "auto"
_ARC_IMPL_SCALAR = "scalar"
_ARC_IMPL_TC = "tc"
_ARC_IMPL_WGMMA = "wgmma"
_ARC_IMPLS = {_ARC_IMPL_AUTO, _ARC_IMPL_SCALAR, _ARC_IMPL_TC, _ARC_IMPL_WGMMA}
_TC_MIN_CAPABILITY = (8, 0)
_WGMMA_MIN_CAPABILITY = (9, 0)
_TC_WORKSPACE_BYTES = 0
_WGMMA_WORKSPACE_BYTES = 32 * 1024 * 1024


LAYOUT_TO_GROUP_SIZE: dict[str, int] = {
    "groupwise_64": 64,
    "groupwise_128": 128,
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without benchmark deps
        raise RuntimeError("PyTorch is required for decode_quant_linear ops") from exc
    return torch


def cuda_decode_available() -> bool:
    info = native_build_info()
    return bool(info.get("available", False)) and bool(info.get("compiled_with_cuda", False))


def group_size_for_layout(layout: str) -> int:
    try:
        return LAYOUT_TO_GROUP_SIZE[layout]
    except KeyError as exc:
        raise ValueError(f"unsupported layout: {layout}") from exc


def arc_packet_stride_bytes(group_size: int) -> int:
    if group_size not in {64, 128}:
        raise ValueError("group_size must be 64 or 128")
    return (128 * (group_size // 2)) + (2 * 128 * 2)


def _require_cuda_backend() -> Any:
    if not cuda_decode_available():
        raise RuntimeError("fast-kernels native module was built without CUDA support")
    return native_module()


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
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_cuda_device(*named_tensors: tuple[str, Any]) -> None:
    devices = {tensor.device for _, tensor in named_tensors}
    if len(devices) <= 1:
        return
    device_summary = ", ".join(f"{name}={tensor.device}" for name, tensor in named_tensors)
    raise ValueError(f"all CUDA tensors must be on the same device: {device_summary}")


def _current_stream_ptr(device: Any) -> int:
    torch = _require_torch()
    return int(torch.cuda.current_stream(device=device).cuda_stream)


def _device_index(device: Any) -> int:
    return int(device.index or 0)


def _cuda_capability(device: Any) -> tuple[int, int]:
    torch = _require_torch()
    capability = torch.cuda.get_device_capability(device)
    return cast(tuple[int, int], tuple(int(x) for x in capability))


def _capability_gte(actual: tuple[int, int], required: tuple[int, int]) -> bool:
    return actual >= required


def arc_w4a16_supported_impls(device: Any | None = None) -> tuple[str, ...]:
    impls = [_ARC_IMPL_AUTO, _ARC_IMPL_SCALAR]
    if not cuda_decode_available():
        return tuple(impls)

    torch = _require_torch()
    if device is None:
        device = torch.device("cuda")
    capability = _cuda_capability(device)
    if _capability_gte(capability, _TC_MIN_CAPABILITY):
        impls.append(_ARC_IMPL_TC)
    if _capability_gte(capability, _WGMMA_MIN_CAPABILITY):
        impls.append(_ARC_IMPL_WGMMA)
    return tuple(impls)


def _resolve_arc_impl(device: Any, impl: str | None) -> str:
    if impl is None:
        impl = os.environ.get(_ARC_IMPL_ENV, _ARC_IMPL_AUTO)
    normalized = impl.strip().lower()
    if normalized not in _ARC_IMPLS:
        raise ValueError(f"impl must be one of {sorted(_ARC_IMPLS)} or omitted; received {impl!r}")

    supported = set(arc_w4a16_supported_impls(device))
    if normalized != _ARC_IMPL_AUTO and normalized not in supported:
        raise ValueError(
            f"impl={normalized!r} is unsupported on device {device}; "
            f"supported implementations are {sorted(supported)}"
        )
    return normalized


def _group_sums_buffer(
    *,
    device: Any,
    batch: int,
    num_groups: int,
) -> Any:
    torch = _require_torch()
    cache_key = (_device_index(device), batch, num_groups)
    group_sums = _ARC_GROUP_SUMS_CACHE.get(cache_key)
    if (
        group_sums is None
        or tuple(group_sums.shape) != (batch, num_groups)
        or group_sums.device != device
    ):
        group_sums = torch.empty((batch, num_groups), device=device, dtype=torch.float32)
        _ARC_GROUP_SUMS_CACHE[cache_key] = group_sums
    return group_sums


def _split_k_partials(
    *,
    device: Any,
    batch: int,
    n: int,
    split_k_slices: int,
) -> Any:
    torch = _require_torch()
    cache_key = (_device_index(device), split_k_slices, batch, n)
    partials = _ARC_SPLIT_K_PARTIALS_CACHE.get(cache_key)
    if (
        partials is None
        or tuple(partials.shape) != (split_k_slices, batch, n)
        or partials.device != device
    ):
        partials = torch.empty((split_k_slices, batch, n), device=device, dtype=torch.float32)
        _ARC_SPLIT_K_PARTIALS_CACHE[cache_key] = partials
    return partials


def _weight_buffer(
    *,
    device: Any,
    n: int,
    k: int,
) -> Any:
    torch = _require_torch()
    cache_key = (_device_index(device), n, k)
    weight = _ARC_WEIGHT_BUFFER_CACHE.get(cache_key)
    if weight is None or tuple(weight.shape) != (n, k) or weight.device != device:
        weight = torch.empty((n, k), device=device, dtype=torch.float16)
        _ARC_WEIGHT_BUFFER_CACHE[cache_key] = weight
    return weight


def _vendor_workspace(
    *,
    device: Any,
    workspace_bytes: int,
) -> Any | None:
    torch = _require_torch()
    if workspace_bytes <= 0:
        return None
    cache_key = (_device_index(device), workspace_bytes)
    workspace = _ARC_WORKSPACE_CACHE.get(cache_key)
    if workspace is None or workspace.numel() != workspace_bytes or workspace.device != device:
        workspace = torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
        _ARC_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def _compute_arc_group_sums(
    native: Any,
    activations: Any,
    group_sums: Any,
    *,
    batch: int,
    k: int,
    group_size: int,
    stream_ptr: int,
) -> None:
    native.compute_arc_w4a16_group_sums(
        activations.data_ptr(),
        group_sums.data_ptr(),
        batch,
        k,
        group_size,
        stream_ptr,
    )


def _arc_forward_scalar_direct(
    native: Any,
    activations: Any,
    packets: Any,
    group_sums: Any,
    output: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
    stream_ptr: int,
) -> None:
    native.arc_w4a16_forward(
        activations.data_ptr(),
        packets.data_ptr(),
        group_sums.data_ptr(),
        output.data_ptr(),
        batch,
        n,
        k,
        group_size,
        packet_stride,
        stream_ptr,
    )


def _arc_forward_scalar_split_k(
    native: Any,
    activations: Any,
    packets: Any,
    group_sums: Any,
    output: Any,
    partials: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
    split_k_slices: int,
    stream_ptr: int,
) -> None:
    native.arc_w4a16_forward_split_k(
        activations.data_ptr(),
        packets.data_ptr(),
        group_sums.data_ptr(),
        partials.data_ptr(),
        batch,
        n,
        k,
        group_size,
        packet_stride,
        split_k_slices,
        stream_ptr,
    )
    native.reduce_arc_w4a16_split_k_partials(
        partials.data_ptr(),
        output.data_ptr(),
        batch,
        n,
        split_k_slices,
        stream_ptr,
    )


def _vendor_workspace_bytes_for_impl(impl: str) -> int:
    if impl == _ARC_IMPL_TC:
        return _TC_WORKSPACE_BYTES
    if impl == _ARC_IMPL_WGMMA:
        return _WGMMA_WORKSPACE_BYTES
    raise ValueError(f"workspace bytes are undefined for impl={impl!r}")


def _arc_forward_vendor_from_packets(
    native: Any,
    activations: Any,
    packets: Any,
    output: Any,
    *,
    weight_buffer: Any,
    workspace: Any | None,
    workspace_bytes: int,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
    stream_ptr: int,
) -> None:
    workspace_ptr = 0 if workspace is None else int(workspace.data_ptr())
    native.cublaslt_fp16_after_packet_dequant(
        activations.data_ptr(),
        packets.data_ptr(),
        output.data_ptr(),
        weight_buffer.data_ptr(),
        workspace_ptr,
        workspace_bytes,
        batch,
        n,
        k,
        group_size,
        packet_stride,
        stream_ptr,
    )


def _arc_split_k_candidate_slices(
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    device: Any,
) -> list[int]:
    torch = _require_torch()
    if batch <= 1:
        return [1]

    if batch <= 2:
        m_tile = 2
    elif batch <= 4:
        m_tile = 4
    else:
        m_tile = 8

    num_tiles = n // 128
    batch_tiles = (batch + m_tile - 1) // m_tile
    resident_ctas = num_tiles * batch_tiles
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    if resident_ctas >= sm_count:
        return [1]

    num_groups = k // group_size
    desired_slices = (sm_count + resident_ctas - 1) // resident_ctas
    preferred = [1, 2, 4, 6, 8, 12, 16]
    candidates = {candidate for candidate in preferred if 1 <= candidate <= num_groups}
    candidates.add(min(num_groups, max(2, desired_slices)))
    candidates.add(min(num_groups, max(2, desired_slices * 2)))
    return sorted(candidates)


def _run_scalar_once(
    native: Any,
    activations: Any,
    packets: Any,
    group_sums: Any,
    output: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
    split_k_slices: int,
    partials: Any | None,
    stream_ptr: int,
) -> None:
    _compute_arc_group_sums(
        native,
        activations,
        group_sums,
        batch=batch,
        k=k,
        group_size=group_size,
        stream_ptr=stream_ptr,
    )
    if split_k_slices <= 1:
        _arc_forward_scalar_direct(
            native,
            activations,
            packets,
            group_sums,
            output,
            batch=batch,
            n=n,
            k=k,
            group_size=group_size,
            packet_stride=packet_stride,
            stream_ptr=stream_ptr,
        )
        return
    if partials is None:
        raise ValueError("partials are required when split_k_slices > 1")
    _arc_forward_scalar_split_k(
        native,
        activations,
        packets,
        group_sums,
        output,
        partials,
        batch=batch,
        n=n,
        k=k,
        group_size=group_size,
        packet_stride=packet_stride,
        split_k_slices=split_k_slices,
        stream_ptr=stream_ptr,
    )


def _autotune_arc_split_k_slices(
    native: Any,
    activations: Any,
    packets: Any,
    group_sums: Any,
    output: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
) -> int:
    torch = _require_torch()
    cache_key = (_device_index(activations.device), batch, n, k, group_size)
    cached = _ARC_SPLIT_K_TUNING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates = _arc_split_k_candidate_slices(
        batch=batch,
        n=n,
        k=k,
        group_size=group_size,
        device=activations.device,
    )
    if candidates == [1]:
        _ARC_SPLIT_K_TUNING_CACHE[cache_key] = 1
        return 1

    stream_ptr = _current_stream_ptr(activations.device)
    best_slices = 1
    best_us: float | None = None

    for split_k_slices in candidates:
        partials = None
        if split_k_slices > 1:
            partials = _split_k_partials(
                device=activations.device,
                batch=batch,
                n=n,
                split_k_slices=split_k_slices,
            )

        for _ in range(2):
            _run_scalar_once(
                native,
                activations,
                packets,
                group_sums,
                output,
                batch=batch,
                n=n,
                k=k,
                group_size=group_size,
                packet_stride=packet_stride,
                split_k_slices=split_k_slices,
                partials=partials,
                stream_ptr=stream_ptr,
            )

        samples: list[float] = []
        for _ in range(4):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_scalar_once(
                native,
                activations,
                packets,
                group_sums,
                output,
                batch=batch,
                n=n,
                k=k,
                group_size=group_size,
                packet_stride=packet_stride,
                split_k_slices=split_k_slices,
                partials=partials,
                stream_ptr=stream_ptr,
            )
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0)

        candidate_us = min(samples)
        if best_us is None or candidate_us < best_us:
            best_us = candidate_us
            best_slices = split_k_slices

    _ARC_SPLIT_K_TUNING_CACHE[cache_key] = best_slices
    return best_slices


def _candidate_impls(
    *,
    device: Any,
    split_k_slices: int | None,
) -> list[str]:
    if split_k_slices is not None and split_k_slices > 1:
        return [_ARC_IMPL_SCALAR]
    return [impl for impl in arc_w4a16_supported_impls(device) if impl != _ARC_IMPL_AUTO]


def _autotune_arc_impl(
    native: Any,
    activations: Any,
    packets: Any,
    group_sums: Any,
    output: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
    split_k_slices: int | None,
) -> str:
    torch = _require_torch()
    capability = _cuda_capability(activations.device)
    cache_key = (
        _device_index(activations.device),
        batch,
        n,
        k,
        group_size,
        split_k_slices,
    )
    cached = _ARC_IMPL_TUNING_CACHE.get(cache_key)
    if cached is not None:
        if cached == _ARC_IMPL_WGMMA and not _capability_gte(capability, _WGMMA_MIN_CAPABILITY):
            pass
        elif cached == _ARC_IMPL_TC and not _capability_gte(capability, _TC_MIN_CAPABILITY):
            pass
        else:
            return cached

    candidates = _candidate_impls(device=activations.device, split_k_slices=split_k_slices)
    if len(candidates) == 1:
        _ARC_IMPL_TUNING_CACHE[cache_key] = candidates[0]
        return candidates[0]

    stream_ptr = _current_stream_ptr(activations.device)
    best_impl = candidates[0]
    best_us: float | None = None

    for candidate in candidates:
        run_candidate: Callable[[], None]
        if candidate == _ARC_IMPL_SCALAR:
            chosen_split_k = (
                _autotune_arc_split_k_slices(
                    native,
                    activations,
                    packets,
                    group_sums,
                    output,
                    batch=batch,
                    n=n,
                    k=k,
                    group_size=group_size,
                    packet_stride=packet_stride,
                )
                if split_k_slices is None
                else split_k_slices
            )
            partials = None
            if chosen_split_k > 1:
                partials = _split_k_partials(
                    device=activations.device,
                    batch=batch,
                    n=n,
                    split_k_slices=chosen_split_k,
                )

            def run_candidate(
                *,
                _chosen_split_k: int = chosen_split_k,
                _partials: Any | None = partials,
            ) -> None:
                _run_scalar_once(
                    native,
                    activations,
                    packets,
                    group_sums,
                    output,
                    batch=batch,
                    n=n,
                    k=k,
                    group_size=group_size,
                    packet_stride=packet_stride,
                    split_k_slices=_chosen_split_k,
                    partials=_partials,
                    stream_ptr=stream_ptr,
                )

        else:
            workspace_bytes = _vendor_workspace_bytes_for_impl(candidate)
            weight_buffer = _weight_buffer(device=activations.device, n=n, k=k)
            workspace = _vendor_workspace(
                device=activations.device,
                workspace_bytes=workspace_bytes,
            )

            def run_candidate(
                *,
                _weight_buffer: Any = weight_buffer,
                _workspace: Any | None = workspace,
                _workspace_bytes: int = workspace_bytes,
            ) -> None:
                _arc_forward_vendor_from_packets(
                    native,
                    activations,
                    packets,
                    output,
                    weight_buffer=_weight_buffer,
                    workspace=_workspace,
                    workspace_bytes=_workspace_bytes,
                    batch=batch,
                    n=n,
                    k=k,
                    group_size=group_size,
                    packet_stride=packet_stride,
                    stream_ptr=stream_ptr,
                )

        for _ in range(2):
            run_candidate()

        samples: list[float] = []
        for _ in range(4):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_candidate()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0)

        candidate_us = min(samples)
        if best_us is None or candidate_us < best_us:
            best_us = candidate_us
            best_impl = candidate

    _ARC_IMPL_TUNING_CACHE[cache_key] = best_impl
    return best_impl


def pack_arc_w4a16_packets(
    q_u8: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    group_size: int,
    packets: torch.Tensor | None = None,
) -> torch.Tensor:
    torch = _require_torch()
    native = _require_cuda_backend()
    _check_cuda_tensor(q_u8, name="q_u8", dtype=torch.uint8, ndim=2)
    _check_cuda_tensor(alpha, name="alpha", dtype=torch.float16, ndim=2)
    _check_cuda_tensor(beta, name="beta", dtype=torch.float16, ndim=2)

    n, packed_k = q_u8.shape
    k = packed_k * 2
    num_groups = k // group_size
    if alpha.shape != (n, num_groups):
        raise ValueError("alpha must have shape [n, k // group_size]")
    if beta.shape != (n, num_groups):
        raise ValueError("beta must have shape [n, k // group_size]")

    packet_stride = arc_packet_stride_bytes(group_size)
    num_tiles = n // 128
    if n % 128 != 0:
        raise ValueError("n must be divisible by 128")
    expected_shape = (num_tiles * num_groups, packet_stride)

    if packets is None:
        packets = torch.empty(expected_shape, device=q_u8.device, dtype=torch.uint8)
    else:
        _check_cuda_tensor(packets, name="packets", dtype=torch.uint8, ndim=2)
        if packets.shape != expected_shape:
            raise ValueError(f"packets must have shape {expected_shape}")

    _check_same_cuda_device(("q_u8", q_u8), ("alpha", alpha), ("beta", beta), ("packets", packets))
    native.pack_arc_w4a16_packets(
        q_u8.data_ptr(),
        alpha.data_ptr(),
        beta.data_ptr(),
        packets.data_ptr(),
        n,
        k,
        group_size,
        packet_stride,
        _current_stream_ptr(q_u8.device),
    )
    return packets


def arc_w4a16_forward(
    activations: torch.Tensor,
    packets: torch.Tensor,
    *,
    n: int,
    k: int,
    group_size: int,
    output: torch.Tensor | None = None,
    split_k_slices: int | None = None,
    partials: torch.Tensor | None = None,
    impl: str | None = None,
) -> torch.Tensor:
    torch = _require_torch()
    native = _require_cuda_backend()
    _check_cuda_tensor(activations, name="activations", dtype=torch.float16, ndim=2)
    _check_cuda_tensor(packets, name="packets", dtype=torch.uint8, ndim=2)

    batch = activations.shape[0]
    if activations.shape[1] != k:
        raise ValueError("activations.shape[1] must match k")

    packet_stride = arc_packet_stride_bytes(group_size)
    num_groups = k // group_size
    expected_packets = ((n // 128) * num_groups, packet_stride)
    if packets.shape != expected_packets:
        raise ValueError(f"packets must have shape {expected_packets}")

    if output is None:
        output = torch.empty((batch, n), device=activations.device, dtype=torch.float16)
    else:
        _check_cuda_tensor(output, name="output", dtype=torch.float16, ndim=2)
        if output.shape != (batch, n):
            raise ValueError(f"output must have shape {(batch, n)}")

    _check_same_cuda_device(("activations", activations), ("packets", packets), ("output", output))

    resolved_impl = _resolve_arc_impl(activations.device, impl)
    if split_k_slices is not None and split_k_slices < 1:
        raise ValueError("split_k_slices must be positive")
    if resolved_impl in {_ARC_IMPL_TC, _ARC_IMPL_WGMMA} and split_k_slices not in {None, 1}:
        raise ValueError(f"split_k_slices is only supported by impl={_ARC_IMPL_SCALAR!r}")

    group_sums = _group_sums_buffer(
        device=activations.device,
        batch=batch,
        num_groups=num_groups,
    )

    if resolved_impl == _ARC_IMPL_AUTO:
        resolved_impl = _autotune_arc_impl(
            native,
            activations,
            packets,
            group_sums,
            output,
            batch=batch,
            n=n,
            k=k,
            group_size=group_size,
            packet_stride=packet_stride,
            split_k_slices=split_k_slices,
        )

    stream_ptr = _current_stream_ptr(activations.device)
    if resolved_impl == _ARC_IMPL_SCALAR:
        chosen_split_k = split_k_slices
        if chosen_split_k is None:
            chosen_split_k = _autotune_arc_split_k_slices(
                native,
                activations,
                packets,
                group_sums,
                output,
                batch=batch,
                n=n,
                k=k,
                group_size=group_size,
                packet_stride=packet_stride,
            )

        if chosen_split_k <= 1:
            _run_scalar_once(
                native,
                activations,
                packets,
                group_sums,
                output,
                batch=batch,
                n=n,
                k=k,
                group_size=group_size,
                packet_stride=packet_stride,
                split_k_slices=1,
                partials=None,
                stream_ptr=stream_ptr,
            )
            return output

        if partials is None:
            partials = _split_k_partials(
                device=activations.device,
                batch=batch,
                n=n,
                split_k_slices=chosen_split_k,
            )
        else:
            _check_cuda_tensor(partials, name="partials", dtype=torch.float32, ndim=3)
            if tuple(partials.shape) != (chosen_split_k, batch, n):
                raise ValueError(f"partials must have shape {(chosen_split_k, batch, n)}")
            _check_same_cuda_device(
                ("activations", activations),
                ("packets", packets),
                ("output", output),
                ("partials", partials),
            )

        _run_scalar_once(
            native,
            activations,
            packets,
            group_sums,
            output,
            batch=batch,
            n=n,
            k=k,
            group_size=group_size,
            packet_stride=packet_stride,
            split_k_slices=chosen_split_k,
            partials=partials,
            stream_ptr=stream_ptr,
        )
        return output

    workspace_bytes = _vendor_workspace_bytes_for_impl(resolved_impl)
    weight_buffer = _weight_buffer(device=activations.device, n=n, k=k)
    workspace = _vendor_workspace(
        device=activations.device,
        workspace_bytes=workspace_bytes,
    )
    named_tensors: list[tuple[str, Any]] = [
        ("activations", activations),
        ("packets", packets),
        ("output", output),
        ("weight_buffer", weight_buffer),
    ]
    if workspace is not None:
        named_tensors.append(("workspace", workspace))
    _check_same_cuda_device(*named_tensors)
    _arc_forward_vendor_from_packets(
        native,
        activations,
        packets,
        output,
        weight_buffer=weight_buffer,
        workspace=workspace,
        workspace_bytes=workspace_bytes,
        batch=batch,
        n=n,
        k=k,
        group_size=group_size,
        packet_stride=packet_stride,
        stream_ptr=stream_ptr,
    )
    return output


def dequant_w4a16_to_fp16(
    q_u8: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    group_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    torch = _require_torch()
    native = _require_cuda_backend()
    _check_cuda_tensor(q_u8, name="q_u8", dtype=torch.uint8, ndim=2)
    _check_cuda_tensor(alpha, name="alpha", dtype=torch.float16, ndim=2)
    _check_cuda_tensor(beta, name="beta", dtype=torch.float16, ndim=2)

    n, packed_k = q_u8.shape
    k = packed_k * 2
    num_groups = k // group_size
    if alpha.shape != (n, num_groups):
        raise ValueError("alpha must have shape [n, k // group_size]")
    if beta.shape != (n, num_groups):
        raise ValueError("beta must have shape [n, k // group_size]")

    if output is None:
        output = torch.empty((n, k), device=q_u8.device, dtype=torch.float16)
    else:
        _check_cuda_tensor(output, name="output", dtype=torch.float16, ndim=2)
        if output.shape != (n, k):
            raise ValueError(f"output must have shape {(n, k)}")

    _check_same_cuda_device(("q_u8", q_u8), ("alpha", alpha), ("beta", beta), ("output", output))
    native.dequant_w4a16_to_fp16(
        q_u8.data_ptr(),
        alpha.data_ptr(),
        beta.data_ptr(),
        output.data_ptr(),
        n,
        k,
        group_size,
        _current_stream_ptr(q_u8.device),
    )
    return output


def cublaslt_fp16_after_dequant(
    activations: torch.Tensor,
    q_u8: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    group_size: int,
    output: torch.Tensor | None = None,
    weight_buffer: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    torch = _require_torch()
    native = _require_cuda_backend()
    _check_cuda_tensor(activations, name="activations", dtype=torch.float16, ndim=2)
    _check_cuda_tensor(q_u8, name="q_u8", dtype=torch.uint8, ndim=2)
    _check_cuda_tensor(alpha, name="alpha", dtype=torch.float16, ndim=2)
    _check_cuda_tensor(beta, name="beta", dtype=torch.float16, ndim=2)

    batch, k = activations.shape
    n, packed_k = q_u8.shape
    if packed_k * 2 != k:
        raise ValueError("q_u8.shape[1] must match activations.shape[1] / 2")
    num_groups = k // group_size
    if alpha.shape != (n, num_groups):
        raise ValueError("alpha must have shape [n, k // group_size]")
    if beta.shape != (n, num_groups):
        raise ValueError("beta must have shape [n, k // group_size]")

    if output is None:
        output = torch.empty((batch, n), device=activations.device, dtype=torch.float16)
    else:
        _check_cuda_tensor(output, name="output", dtype=torch.float16, ndim=2)
        if output.shape != (batch, n):
            raise ValueError(f"output must have shape {(batch, n)}")

    if weight_buffer is None:
        weight_buffer = torch.empty((n, k), device=activations.device, dtype=torch.float16)
    else:
        _check_cuda_tensor(weight_buffer, name="weight_buffer", dtype=torch.float16, ndim=2)
        if weight_buffer.shape != (n, k):
            raise ValueError(f"weight_buffer must have shape {(n, k)}")

    if workspace is None:
        workspace = torch.empty(8 * 1024 * 1024, device=activations.device, dtype=torch.uint8)
    else:
        _check_cuda_tensor(workspace, name="workspace", dtype=torch.uint8, ndim=1)

    _check_same_cuda_device(
        ("activations", activations),
        ("q_u8", q_u8),
        ("alpha", alpha),
        ("beta", beta),
        ("output", output),
        ("weight_buffer", weight_buffer),
        ("workspace", workspace),
    )
    native.cublaslt_fp16_after_dequant(
        activations.data_ptr(),
        q_u8.data_ptr(),
        alpha.data_ptr(),
        beta.data_ptr(),
        output.data_ptr(),
        weight_buffer.data_ptr(),
        workspace.data_ptr(),
        workspace.numel(),
        batch,
        n,
        k,
        group_size,
        _current_stream_ptr(activations.device),
    )
    return output
