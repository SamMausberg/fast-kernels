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


def test_clustered_page_decode_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/clustered_page_decode.toml"))
    assert suite.id == "clustered_page_decode"
    assert suite.family == "clustered_page_decode"
    assert suite.kernels.ids == [
        "decode/clustered_page_decode_auto",
        "decode/clustered_page_decode_direct",
        "decode/clustered_page_decode_clustered",
    ]
    assert suite.baselines.ids == [
        "torch/reference_clustered_page_decode",
        "vendor/flashinfer_clustered_page_decode",
    ]
    assert suite.layouts == ["bf16_kv", "fp8_kv", "int8_kv"]
    assert len(suite.shapes) == 5
    assert any(shape.name == "mqa32_long_page32" for shape in suite.shapes)


def test_clustered_page_decode_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/clustered_page_decode.toml"))
    assert verify_suite(suite) == []


def test_prefix_union_decode_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/prefix_union_decode.toml"))
    assert suite.id == "prefix_union_decode"
    assert suite.family == "prefix_union_decode"
    assert suite.kernels.ids == [
        "decode/prefix_union_decode_auto",
        "decode/prefix_union_decode_union",
        "decode/prefix_union_decode_fallback",
    ]
    assert suite.baselines.ids == ["torch/reference_prefix_union_decode"]
    assert suite.layouts == ["bf16_kv", "fp8_kv", "int8_kv"]
    assert len(suite.shapes) == 4
    assert any(shape.name == "mqa8_prefix8_page32" for shape in suite.shapes)


def test_prefix_union_decode_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/prefix_union_decode.toml"))
    assert verify_suite(suite) == []


def test_rdkng_block_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/rdkng_block.toml"))
    assert suite.id == "rdkng_block"
    assert suite.family == "rdkng"
    assert suite.kernels.ids == [
        "rdkng/explicit_sketch_hybrid",
        "rdkng/explicit_sketch_lowrank",
    ]
    assert suite.baselines.ids == [
        "torch/plain_cg_reference",
        "official/galore_projector",
    ]
    assert suite.layouts == ["compressible_drift", "noncompressible_control"]
    assert len(suite.shapes) == 2
    assert suite.shapes[0].m == 64
    assert suite.shapes[0].n == 64


def test_rdkng_block_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/rdkng_block.toml"))
    assert verify_suite(suite) == []


def test_rdkng_training_suite_loads() -> None:
    suite = load_suite(Path("benchmarks/suites/rdkng_training.toml"))
    assert suite.id == "rdkng_training"
    assert suite.family == "rdkng"
    assert suite.layouts == ["teacher_student"]
    assert suite.baselines.ids == [
        "torch/plain_cg_reference",
        "official/galore_adamw",
    ]
    assert len(suite.shapes) == 1
    assert suite.shapes[0].batch == 8
    assert suite.shapes[0].max_seq_len == 64


def test_rdkng_training_suite_verifies() -> None:
    suite = load_suite(Path("benchmarks/suites/rdkng_training.toml"))
    assert verify_suite(suite) == []
