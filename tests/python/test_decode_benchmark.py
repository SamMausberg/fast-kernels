from __future__ import annotations

from pathlib import Path

import pytest

from fast_kernels.benchmarking import benchmark_suite
from fast_kernels.benchmarking import clustered_page_decode as clustered_bench
from fast_kernels.benchmarking.suites import load_suite
from fast_kernels.reporting import load_result_bundle


def test_decode_suite_writes_artifacts_when_skipped(tmp_path: Path) -> None:
    run_dir = benchmark_suite(
        Path("benchmarks/suites/decode_linear_w4a16.toml"),
        output_root=tmp_path,
    )
    bundle = load_result_bundle(run_dir)
    assert bundle.metadata.suite_id == "decode_linear_w4a16"
    assert bundle.cases
    statuses = {case.status for case in bundle.cases}
    assert statuses <= {"ok", "skipped", "failed"}
    assert (run_dir / "summary.md").exists()


def test_clustered_page_decode_suite_writes_artifacts_when_skipped(tmp_path: Path) -> None:
    run_dir = benchmark_suite(
        Path("benchmarks/suites/clustered_page_decode.toml"),
        output_root=tmp_path,
    )
    bundle = load_result_bundle(run_dir)
    assert bundle.metadata.suite_id == "clustered_page_decode"
    assert bundle.cases
    statuses = {case.status for case in bundle.cases}
    assert statuses <= {"ok", "skipped", "failed"}
    assert (run_dir / "summary.md").exists()


def _load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def test_flashinfer_skip_reason_only_accepts_unrotated_bf16() -> None:
    torch = _load_torch()
    if torch is None:
        pytest.skip("PyTorch is required")

    base_cache = type(
        "FakeCache",
        (),
        {
            "kv_layout": "bf16_kv",
            "keys_are_rotated": False,
            "seq_lens": torch.tensor([16], dtype=torch.int32),
        },
    )()
    assert clustered_bench._flashinfer_skip_reason(base_cache) is None

    fp8_cache = type(
        "FakeCache",
        (),
        {
            "kv_layout": "fp8_kv",
            "keys_are_rotated": False,
            "seq_lens": torch.tensor([16], dtype=torch.int32),
        },
    )()
    assert "supports only bf16_kv" in clustered_bench._flashinfer_skip_reason(fp8_cache)

    rotated_cache = type(
        "FakeCache",
        (),
        {
            "kv_layout": "bf16_kv",
            "keys_are_rotated": True,
            "seq_lens": torch.tensor([16], dtype=torch.int32),
        },
    )()
    assert "unrotated key pages" in clustered_bench._flashinfer_skip_reason(rotated_cache)


def test_flashinfer_page_metadata_flattens_valid_pages() -> None:
    torch = _load_torch()
    if torch is None:
        pytest.skip("PyTorch is required")

    cache = type(
        "FakeCache",
        (),
        {
            "seq_lens": torch.tensor([48, 32], dtype=torch.int32),
            "page_size": 16,
            "page_table": torch.tensor([[7, 8, 9], [4, 5, -1]], dtype=torch.int32),
        },
    )()
    indptr, indices, last_page_len = clustered_bench._flashinfer_page_metadata(cache, "cpu")
    assert indptr.tolist() == [0, 3, 5]
    assert indices.tolist() == [7, 8, 9, 4, 5]
    assert last_page_len.tolist() == [16, 16]


def test_case_seed_is_stable_across_calls() -> None:
    suite = load_suite(Path("benchmarks/suites/clustered_page_decode.toml"))
    shape = next(s for s in suite.shapes if s.name == "mqa32_long_page32")
    seed0 = clustered_bench._case_seed(shape, "bf16_kv")
    seed1 = clustered_bench._case_seed(shape, "bf16_kv")
    assert seed0 == seed1
