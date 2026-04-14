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


def _build_cache(
    layout: str,
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    key_rope_theta: float | None = None,
) -> Any:
    if layout == "bf16_kv":
        return pack_paged_kv_bf16(
            keys,
            values,
            seq_lens,
            page_size=16,
            fragment_pages=True,
            seed=1,
            key_rope_theta=key_rope_theta,
        )
    if layout == "fp8_kv":
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn")
        return quantize_paged_kv_fp8(
            keys,
            values,
            seq_lens,
            page_size=16,
            fragment_pages=True,
            seed=1,
            key_rope_theta=key_rope_theta,
        )
    return quantize_paged_kv_int8(
        keys,
        values,
        seq_lens,
        page_size=16,
        fragment_pages=True,
        seed=1,
        key_rope_theta=key_rope_theta,
    )


def _assert_case(layout: str, force_impl: str, *, key_rope_theta: float | None = None) -> None:
    query, keys, values, seq_lens = _make_case_inputs(
        batch=2,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=128,
        page_size=16,
    )
    cache = _build_cache(layout, keys, values, seq_lens, key_rope_theta=key_rope_theta)

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


@pytest.mark.parametrize("layout", ["bf16_kv", "fp8_kv", "int8_kv"])
@pytest.mark.parametrize("force_impl", ["direct", "clustered"])
def test_clustered_page_decode_pre_rotated_keys_match_reference(
    layout: str,
    force_impl: str,
) -> None:
    _assert_case(layout, force_impl, key_rope_theta=10000.0)


def test_clustered_page_decode_rejects_mismatched_pre_rotated_rope_theta() -> None:
    query, keys, values, seq_lens = _make_case_inputs(
        batch=2,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=128,
        page_size=16,
    )
    cache = _build_cache("bf16_kv", keys, values, seq_lens, key_rope_theta=10000.0)
    with pytest.raises(ValueError, match="rope_theta must match"):
        clustered_page_decode(query, cache, rope_theta=5000.0, force_impl="direct")


def test_plan_clustered_page_decode_sm120_heuristic_prefers_direct_for_small_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (12, 0))

    page_table = torch.arange(128, dtype=torch.int32).view(1, 128)
    seq_lens = torch.tensor([2048], dtype=torch.int32)
    plan = plan_clustered_page_decode(
        page_table=page_table,
        seq_lens=seq_lens,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        kv_layout="bf16_kv",
    )
    assert plan.cluster_size == 1
    assert plan.launch_mode == "direct"


def test_plan_clustered_page_decode_sm120_heuristic_keeps_group8_on_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (12, 0))

    page_table = torch.arange(128, dtype=torch.int32).view(1, 128)
    seq_lens = torch.tensor([4096], dtype=torch.int32)
    plan = plan_clustered_page_decode(
        page_table=page_table,
        seq_lens=seq_lens,
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=128,
        page_size=32,
        kv_layout="bf16_kv",
    )
    assert plan.cluster_size == 1
    assert plan.launch_mode == "direct"
