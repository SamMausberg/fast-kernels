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
        "torch/reference_w4a16_linear": BaselineSpec(
            baseline_id="torch/reference_w4a16_linear",
            source="framework",
            description="Reference dequantize-plus-torch.mm W4A16 baseline.",
            maturity="experimental",
        ),
        "vendor/cublaslt_fp16_after_dequant": BaselineSpec(
            baseline_id="vendor/cublaslt_fp16_after_dequant",
            source="vendor",
            description="Vendor cuBLASLt FP16 matmul after fused native W4A16 dequantization.",
            maturity="experimental",
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
        "torch/reference_clustered_page_decode": BaselineSpec(
            baseline_id="torch/reference_clustered_page_decode",
            source="framework",
            description=(
                "Torch exact paged decode reference with in-kernel-equivalent "
                "RoPE semantics."
            ),
            maturity="experimental",
        ),
        "vendor/flashinfer_clustered_page_decode": BaselineSpec(
            baseline_id="vendor/flashinfer_clustered_page_decode",
            source="vendor",
            description=(
                "FlashInfer paged batch decode baseline adapter when the runtime "
                "is available."
            ),
            maturity="experimental",
        ),
    }
