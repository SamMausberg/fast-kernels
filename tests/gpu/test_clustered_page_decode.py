from __future__ import annotations

from typing import Any

import pytest

from fast_kernels.native import native_build_info
from fast_kernels.ops import (
    clustered_page_decode,
    pack_paged_kv_bf16,
    plan_clustered_page_decode,
    quantize_paged_kv_fp8,
    quantize_paged_kv_int8,
    reference_clustered_page_decode,
)


def _load_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


torch = _load_torch()


def _cuda_ready() -> bool:
    if torch is None:
        return False
    info = native_build_info()
    return bool(info.get("compiled_with_cuda", False)) and bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="Clustered page decode GPU tests require PyTorch, torch.cuda, and a CUDA-enabled build.",
)


def _make_case_inputs(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    page_size: int,
) -> tuple[Any, Any, Any, Any]:
    torch.manual_seed(batch + num_q_heads + num_kv_heads + head_dim + max_seq_len + page_size)
    device = torch.device("cuda")
    query = torch.randn((batch, num_q_heads, head_dim), device=device).to(torch.bfloat16)
    keys = torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device).to(
        torch.bfloat16
    )
    values = torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device).to(
        torch.bfloat16
    )
    seq_lens = torch.randint(
        low=max(page_size, max_seq_len - page_size + 1),
        high=max_seq_len + 1,
        size=(batch,),
        dtype=torch.int32,
    )
    return query.contiguous(), keys.contiguous(), values.contiguous(), seq_lens.contiguous()


def _assert_case(layout: str, force_impl: str) -> None:
    query, keys, values, seq_lens = _make_case_inputs(
        batch=2,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=128,
        page_size=16,
    )
    if layout == "bf16_kv":
        cache = pack_paged_kv_bf16(
            keys,
            values,
            seq_lens,
            page_size=16,
            fragment_pages=True,
            seed=1,
        )
    elif layout == "fp8_kv":
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn")
        cache = quantize_paged_kv_fp8(
            keys, values, seq_lens, page_size=16, fragment_pages=True, seed=1
        )
    else:
        cache = quantize_paged_kv_int8(
            keys, values, seq_lens, page_size=16, fragment_pages=True, seed=1
        )

    plan = plan_clustered_page_decode(
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        num_q_heads=int(query.shape[1]),
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
        page_size=cache.page_size,
        kv_layout=cache.kv_layout,
        cluster_size=4 if force_impl == "clustered" else 1,
    )
    actual = clustered_page_decode(query, cache, plan=plan, force_impl=force_impl)
    expected = reference_clustered_page_decode(query, cache)
    torch.testing.assert_close(actual, expected, atol=1.25e-1, rtol=3e-2)


@pytest.mark.parametrize("layout", ["bf16_kv", "int8_kv"])
@pytest.mark.parametrize("force_impl", ["direct", "clustered"])
def test_clustered_page_decode_matches_reference(layout: str, force_impl: str) -> None:
    _assert_case(layout, force_impl)


def test_clustered_page_decode_fp8_matches_reference() -> None:
    _assert_case("fp8_kv", "clustered")
