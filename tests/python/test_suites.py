from pathlib import Path

from fast_kernels.benchmarking import load_suite, verify_suite


def test_template_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/template_gemm.toml"))
    assert suite.id == "template_gemm"
    assert len(suite.shapes) == 2
    assert suite.kernels.ids == ["template/noop_gemm"]


def test_template_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/template_gemm.toml"))
    assert verify_suite(suite) == []


def test_decode_linear_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/decode_linear_w4a16.toml"))
    assert suite.id == "decode_linear_w4a16"
    assert suite.kernels.ids == ["decode/w4a16_linear"]
    assert "torch/torchao_w4a16_linear" in suite.baselines.ids
    assert len(suite.shapes) == 7


def test_decode_linear_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/decode_linear_w4a16.toml"))
    assert verify_suite(suite) == []
