from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_kernels.native import native_build_info, native_module

if TYPE_CHECKING:  # pragma: no cover - imported only for static checking
    import torch


_ARC_SPLIT_K_PARTIALS_CACHE: dict[tuple[int, int, int, int], Any] = {}
_ARC_SPLIT_K_TUNING_CACHE: dict[tuple[int, int, int, int, int], int] = {}


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


def _split_k_partials(
    *,
    device: Any,
    batch: int,
    n: int,
    split_k_slices: int,
) -> Any:
    torch = _require_torch()
    cache_key = (device.index or 0, split_k_slices, batch, n)
    partials = _ARC_SPLIT_K_PARTIALS_CACHE.get(cache_key)
    if (
        partials is None
        or tuple(partials.shape) != (split_k_slices, batch, n)
        or partials.device != device
    ):
        partials = torch.empty((split_k_slices, batch, n), device=device, dtype=torch.float32)
        _ARC_SPLIT_K_PARTIALS_CACHE[cache_key] = partials
    return partials


def _arc_forward_direct(
    native: Any,
    activations: Any,
    packets: Any,
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
        output.data_ptr(),
        batch,
        n,
        k,
        group_size,
        packet_stride,
        stream_ptr,
    )


def _arc_forward_split_k(
    native: Any,
    activations: Any,
    packets: Any,
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
    candidates = {
        candidate
        for candidate in preferred
        if 1 <= candidate <= num_groups
    }
    candidates.add(min(num_groups, max(2, desired_slices)))
    candidates.add(min(num_groups, max(2, desired_slices * 2)))
    return sorted(candidates)


def _autotune_arc_split_k_slices(
    native: Any,
    activations: Any,
    packets: Any,
    output: Any,
    *,
    batch: int,
    n: int,
    k: int,
    group_size: int,
    packet_stride: int,
) -> int:
    torch = _require_torch()
    cache_key = (activations.device.index or 0, batch, n, k, group_size)
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
            if split_k_slices <= 1:
                _arc_forward_direct(
                    native,
                    activations,
                    packets,
                    output,
                    batch=batch,
                    n=n,
                    k=k,
                    group_size=group_size,
                    packet_stride=packet_stride,
                    stream_ptr=stream_ptr,
                )
            else:
                _arc_forward_split_k(
                    native,
                    activations,
                    packets,
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

        samples: list[float] = []
        for _ in range(4):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if split_k_slices <= 1:
                _arc_forward_direct(
                    native,
                    activations,
                    packets,
                    output,
                    batch=batch,
                    n=n,
                    k=k,
                    group_size=group_size,
                    packet_stride=packet_stride,
                    stream_ptr=stream_ptr,
                )
            else:
                _arc_forward_split_k(
                    native,
                    activations,
                    packets,
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
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0)

        candidate_us = min(samples)
        if best_us is None or candidate_us < best_us:
            best_us = candidate_us
            best_slices = split_k_slices

    _ARC_SPLIT_K_TUNING_CACHE[cache_key] = best_slices
    return best_slices


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
    if split_k_slices is None:
        split_k_slices = _autotune_arc_split_k_slices(
            native,
            activations,
            packets,
            output,
            batch=batch,
            n=n,
            k=k,
            group_size=group_size,
            packet_stride=packet_stride,
        )
    elif split_k_slices < 1:
        raise ValueError("split_k_slices must be positive")

    stream_ptr = _current_stream_ptr(activations.device)
    if split_k_slices <= 1:
        _arc_forward_direct(
            native,
            activations,
            packets,
            output,
            batch=batch,
            n=n,
            k=k,
            group_size=group_size,
            packet_stride=packet_stride,
            stream_ptr=stream_ptr,
        )
        return output

    if partials is None:
        partials = _split_k_partials(
            device=activations.device,
            batch=batch,
            n=n,
            split_k_slices=split_k_slices,
        )
    else:
        _check_cuda_tensor(partials, name="partials", dtype=torch.float32, ndim=3)
        if tuple(partials.shape) != (split_k_slices, batch, n):
            raise ValueError(f"partials must have shape {(split_k_slices, batch, n)}")
        _check_same_cuda_device(
            ("activations", activations),
            ("packets", packets),
            ("output", output),
            ("partials", partials),
        )

    _arc_forward_split_k(
        native,
        activations,
        packets,
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
