from __future__ import annotations

from typing import Any

import pytest

from fast_kernels.native import native_build_info

torch = pytest.importorskip("torch")

from fast_kernels.ops import arc_w4a16_forward, dequant_w4a16_to_fp16, pack_arc_w4a16_packets


def _cuda_ready() -> bool:
    info = native_build_info()
    return bool(info.get("compiled_with_cuda", False)) and bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="GPU decode tests require torch.cuda and a CUDA-enabled native build.",
)


def _explicit_reference_output(
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


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("batch", [1, 4])
def test_arc_kernel_matches_explicit_reference(group_size: int, batch: int) -> None:
    torch.manual_seed(7 + batch + group_size)
    device = torch.device("cuda")
    n = 128
    k = group_size * 2
    num_groups = k // group_size

    activations = torch.randn((batch, k), device=device, dtype=torch.float16).contiguous()
    q_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8).contiguous()
    alpha = (0.05 + (0.1 * torch.rand((n, num_groups), device=device, dtype=torch.float32))).to(torch.float16)
    zero_points = torch.randint(0, 16, (n, num_groups), device=device, dtype=torch.int16)
    beta = -(alpha * zero_points.to(dtype=torch.float16))

    packets = pack_arc_w4a16_packets(q_u8, alpha, beta, group_size=group_size)
    kernel_output = arc_w4a16_forward(activations, packets, n=n, k=k, group_size=group_size)
    reference_output = _explicit_reference_output(
        activations,
        q_u8,
        alpha,
        beta,
        group_size=group_size,
    )

    torch.testing.assert_close(kernel_output, reference_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("group_size", [64, 128])
def test_native_dequant_matches_explicit_reference(group_size: int) -> None:
    torch.manual_seed(123 + group_size)
    device = torch.device("cuda")
    n = 128
    k = group_size * 2
    num_groups = k // group_size

    q_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8).contiguous()
    alpha = (0.03 + (0.2 * torch.rand((n, num_groups), device=device, dtype=torch.float32))).to(torch.float16)
    zero_points = torch.randint(0, 16, (n, num_groups), device=device, dtype=torch.int16)
    beta = -(alpha * zero_points.to(dtype=torch.float16))

    native_weight = dequant_w4a16_to_fp16(q_u8, alpha, beta, group_size=group_size)

    low = torch.bitwise_and(q_u8, 0x0F).to(torch.float32)
    high = torch.bitwise_right_shift(q_u8, 4).to(torch.float32)
    q_values = torch.stack((low, high), dim=-1).reshape(n, k)
    expected_weight = (
        alpha.to(torch.float32).repeat_interleave(group_size, dim=1) * q_values
        + beta.to(torch.float32).repeat_interleave(group_size, dim=1)
    ).to(torch.float16)

    torch.testing.assert_close(native_weight, expected_weight, atol=1e-3, rtol=1e-3)


def test_arc_split_k_override_matches_explicit_reference() -> None:
    torch.manual_seed(4242)
    device = torch.device("cuda")
    group_size = 128
    batch = 8
    n = 128
    k = group_size * 4
    num_groups = k // group_size

    activations = torch.randn((batch, k), device=device, dtype=torch.float16).contiguous()
    q_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8).contiguous()
    alpha = (0.04 + (0.12 * torch.rand((n, num_groups), device=device, dtype=torch.float32))).to(torch.float16)
    zero_points = torch.randint(0, 16, (n, num_groups), device=device, dtype=torch.int16)
    beta = -(alpha * zero_points.to(dtype=torch.float16))
    packets = pack_arc_w4a16_packets(q_u8, alpha, beta, group_size=group_size)
    partials = torch.empty((4, batch, n), device=device, dtype=torch.float32)

    kernel_output = arc_w4a16_forward(
        activations,
        packets,
        n=n,
        k=k,
        group_size=group_size,
        split_k_slices=4,
        partials=partials,
    )
    reference_output = _explicit_reference_output(
        activations,
        q_u8,
        alpha,
        beta,
        group_size=group_size,
    )

    torch.testing.assert_close(kernel_output, reference_output, atol=1e-2, rtol=1e-2)


def test_arc_kernel_respects_current_stream() -> None:
    torch.manual_seed(2112)
    device = torch.device("cuda")
    group_size = 128
    n = 128
    k = group_size * 2
    num_groups = k // group_size

    expected_activations = torch.randn((1, k), device=device, dtype=torch.float16).contiguous()
    activations = torch.zeros((1, k), device=device, dtype=torch.float16)
    q_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8).contiguous()
    alpha = (0.05 + (0.1 * torch.rand((n, num_groups), device=device, dtype=torch.float32))).to(torch.float16)
    zero_points = torch.randint(0, 16, (n, num_groups), device=device, dtype=torch.int16)
    beta = -(alpha * zero_points.to(dtype=torch.float16))
    packets = pack_arc_w4a16_packets(q_u8, alpha, beta, group_size=group_size)
    expected_output = _explicit_reference_output(
        expected_activations,
        q_u8,
        alpha,
        beta,
        group_size=group_size,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        activations.copy_(expected_activations)
        stream_output = arc_w4a16_forward(activations, packets, n=n, k=k, group_size=group_size)

    stream.synchronize()
    torch.cuda.synchronize()
    torch.testing.assert_close(stream_output, expected_output, atol=1e-2, rtol=1e-2)
