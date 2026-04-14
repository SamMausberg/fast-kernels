from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    baseline_id: str
    source: str
    description: str
    maturity: str


def baseline_registry() -> dict[str, BaselineSpec]:
    return {
        "torch/torchao_w4a16_linear": BaselineSpec(
            baseline_id="torch/torchao_w4a16_linear",
            source="framework",
            description="TorchAO weight-only linear baseline placeholder.",
            maturity="scaffold",
        ),
        "vendor/cublaslt_fp16_after_dequant": BaselineSpec(
            baseline_id="vendor/cublaslt_fp16_after_dequant",
            source="vendor",
            description="Separate dequant plus cuBLASLt linear baseline placeholder.",
            maturity="scaffold",
        ),
        "torch/reference_gemm": BaselineSpec(
            baseline_id="torch/reference_gemm",
            source="framework",
            description="PyTorch reference GEMM baseline placeholder.",
            maturity="scaffold",
        ),
        "vendor/cublaslt_gemm": BaselineSpec(
            baseline_id="vendor/cublaslt_gemm",
            source="vendor",
            description="cuBLASLt GEMM baseline placeholder.",
            maturity="scaffold",
        ),
    }
