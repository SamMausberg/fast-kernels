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
