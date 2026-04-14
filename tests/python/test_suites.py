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
    assert suite.kernels.ids == [
        "decode/w4a16_linear",
        "decode/w4a16_linear_scalar",
        "decode/w4a16_linear_tc",
        "decode/w4a16_linear_wgmma",
    ]
    assert "torch/reference_w4a16_linear" in suite.baselines.ids
    assert "groupwise_64" in suite.layouts
    assert len(suite.shapes) == 12


def test_decode_linear_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/decode_linear_w4a16.toml"))
    assert verify_suite(suite) == []
