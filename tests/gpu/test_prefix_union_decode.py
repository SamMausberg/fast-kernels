from __future__ import annotations

from typing import Any

import pytest

from fast_kernels.native import native_build_info
from fast_kernels.ops import (
    pack_paged_kv_bf16,
    plan_prefix_union_decode,
    prefix_union_decode,
    quantize_paged_kv_fp8,
    quantize_paged_kv_int8,
    reference_prefix_union_decode,
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


def _blackwell_ready() -> bool:
    if not _cuda_ready():
        return False
    major, _minor = torch.cuda.get_device_capability(torch.device("cuda"))
    return major >= 12


pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="Prefix union decode GPU tests require PyTorch, torch.cuda, and a CUDA-enabled build.",
)


def _make_shared_prefix_case(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    page_size: int,
    shared_prefix_pages: int,
    prefix_group_size: int,
) -> tuple[Any, Any, Any, Any]:
    torch.manual_seed(
        batch
        + num_q_heads
        + num_kv_heads
        + head_dim
        + max_seq_len
        + page_size
        + shared_prefix_pages
        + prefix_group_size
    )
    device = torch.device("cuda")
    shared_prefix_tokens = shared_prefix_pages * page_size
    min_seq_len = max(page_size, shared_prefix_tokens + 1)
    query = torch.randn((batch, num_q_heads, head_dim), device=device).to(torch.bfloat16)
    keys = torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device).to(
        torch.bfloat16
    )
    values = torch.randn((batch, num_kv_heads, max_seq_len, head_dim), device=device).to(
        torch.bfloat16
    )
    seq_lens = torch.randint(
        low=min_seq_len,
        high=max_seq_len + 1,
        size=(batch,),
        dtype=torch.int32,
    ).contiguous()
    for group_start in range(0, batch, prefix_group_size):
        group_end = min(batch, group_start + prefix_group_size)
        if group_end - group_start < 2:
            continue
        leader = group_start
        for follower in range(group_start + 1, group_end):
            keys[follower, :, :shared_prefix_tokens, :] = keys[leader, :, :shared_prefix_tokens, :]
            values[follower, :, :shared_prefix_tokens, :] = values[
                leader, :, :shared_prefix_tokens, :
            ]
    return query.contiguous(), keys.contiguous(), values.contiguous(), seq_lens


def _build_shared_cache(
    layout: str,
    keys: Any,
    values: Any,
    seq_lens: Any,
    *,
    page_size: int,
    key_rope_theta: float | None = 10000.0,
) -> Any:
    common_kwargs = dict(
        page_size=page_size,
        fragment_pages=True,
        seed=3,
        key_rope_theta=key_rope_theta,
        deduplicate_identical_pages=True,
    )
    if layout == "bf16_kv":
        return pack_paged_kv_bf16(keys, values, seq_lens, **common_kwargs)
    if layout == "fp8_kv":
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn")
        return quantize_paged_kv_fp8(keys, values, seq_lens, **common_kwargs)
    return quantize_paged_kv_int8(keys, values, seq_lens, **common_kwargs)


@pytest.mark.skipif(not _blackwell_ready(), reason="Union mode requires a Blackwell-class GPU.")
@pytest.mark.parametrize("layout", ["bf16_kv", "fp8_kv", "int8_kv"])
def test_prefix_union_decode_matches_reference(layout: str) -> None:
    query, keys, values, seq_lens = _make_shared_prefix_case(
        batch=8,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=1024,
        page_size=16,
        shared_prefix_pages=4,
        prefix_group_size=4,
    )
    cache = _build_shared_cache(layout, keys, values, seq_lens, page_size=16)
    plan = plan_prefix_union_decode(
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        num_q_heads=int(query.shape[1]),
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
        page_size=cache.page_size,
        kv_layout=cache.kv_layout,
        keys_are_rotated=cache.keys_are_rotated,
    )
    assert plan.launch_mode == "union"
    assert plan.num_tasks > 0
    assert int(plan.shared_pages_cpu.shape[0]) > 0

    actual = prefix_union_decode(
        query,
        cache,
        plan=plan,
        force_impl="union",
        rope_theta=10000.0,
    )
    expected = reference_prefix_union_decode(query, cache, rope_theta=10000.0)
    torch.testing.assert_close(actual, expected, atol=1.25e-1, rtol=3e-2)


def test_prefix_union_decode_auto_falls_back_for_unrotated_keys() -> None:
    query, keys, values, seq_lens = _make_shared_prefix_case(
        batch=4,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=256,
        page_size=16,
        shared_prefix_pages=2,
        prefix_group_size=2,
    )
    cache = _build_shared_cache(
        "bf16_kv",
        keys,
        values,
        seq_lens,
        page_size=16,
        key_rope_theta=None,
    )
    plan = plan_prefix_union_decode(
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        num_q_heads=int(query.shape[1]),
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
        page_size=cache.page_size,
        kv_layout=cache.kv_layout,
        keys_are_rotated=cache.keys_are_rotated,
    )
    assert plan.launch_mode == "fallback"
    actual = prefix_union_decode(query, cache, plan=plan, force_impl="auto")
    expected = reference_prefix_union_decode(query, cache)
    torch.testing.assert_close(actual, expected, atol=1.25e-1, rtol=3e-2)


@pytest.mark.skipif(not _blackwell_ready(), reason="Union mode requires a Blackwell-class GPU.")
def test_prefix_union_decode_force_union_rejects_unrotated_keys() -> None:
    query, keys, values, seq_lens = _make_shared_prefix_case(
        batch=4,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=256,
        page_size=16,
        shared_prefix_pages=2,
        prefix_group_size=2,
    )
    cache = _build_shared_cache(
        "bf16_kv",
        keys,
        values,
        seq_lens,
        page_size=16,
        key_rope_theta=None,
    )
    plan = plan_prefix_union_decode(
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        num_q_heads=int(query.shape[1]),
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
        page_size=cache.page_size,
        kv_layout=cache.kv_layout,
        keys_are_rotated=cache.keys_are_rotated,
    )
    with pytest.raises(ValueError, match="pre-rotated key pages"):
        prefix_union_decode(query, cache, plan=plan, force_impl="union")
