from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelSpec:
    kernel_id: str
    family: str
    description: str
    maturity: str
    ptx_hotspots: bool = False


def kernel_registry() -> dict[str, KernelSpec]:
    return {
        "decode/w4a16_linear": KernelSpec(
            kernel_id="decode/w4a16_linear",
            family="decode_quant_linear",
            description=(
                "Auto-dispatched ARC decode kernel for affine W4A16 weight-only linear layers."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/w4a16_linear_scalar": KernelSpec(
            kernel_id="decode/w4a16_linear_scalar",
            family="decode_quant_linear",
            description=(
                "Scalar ARC decode kernel with packet reuse, activation-sum prepass, "
                "and split-K support."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/w4a16_linear_tc": KernelSpec(
            kernel_id="decode/w4a16_linear_tc",
            family="decode_quant_linear",
            description=("Tensor Core-backed ARC decode path using packet dequant plus cublasLt."),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/w4a16_linear_wgmma": KernelSpec(
            kernel_id="decode/w4a16_linear_wgmma",
            family="decode_quant_linear",
            description=("Warpgroup-capable ARC decode path using packet dequant plus cublasLt."),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "template/noop_gemm": KernelSpec(
            kernel_id="template/noop_gemm",
            family="gemm",
            description=(
                "Scaffold-only GEMM entry point for wiring up launchers and suite definitions."
            ),
            maturity="scaffold",
            ptx_hotspots=False,
        ),
        "decode/clustered_page_decode_auto": KernelSpec(
            kernel_id="decode/clustered_page_decode_auto",
            family="clustered_page_decode",
            description=(
                "Auto-selected clustered page-stream decode path with direct-vs-clustered dispatch."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/clustered_page_decode_direct": KernelSpec(
            kernel_id="decode/clustered_page_decode_direct",
            family="clustered_page_decode",
            description=(
                "Single-cluster direct page-stream decode path for short-context paged attention."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/clustered_page_decode_clustered": KernelSpec(
            kernel_id="decode/clustered_page_decode_clustered",
            family="clustered_page_decode",
            description=(
                "Multi-block clustered page-stream decode path with exact local state merging."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/prefix_union_decode_auto": KernelSpec(
            kernel_id="decode/prefix_union_decode_auto",
            family="prefix_union_decode",
            description=(
                "Auto-selected Blackwell prefix-union decode path with shared-page reuse and "
                "cluster fallback."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/prefix_union_decode_union": KernelSpec(
            kernel_id="decode/prefix_union_decode_union",
            family="prefix_union_decode",
            description=(
                "Blackwell prefix-union decode path that stages shared prefix pages once per "
                "cluster wave."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
        "decode/prefix_union_decode_fallback": KernelSpec(
            kernel_id="decode/prefix_union_decode_fallback",
            family="prefix_union_decode",
            description=(
                "Prefix-union decode API routed through the existing clustered_page_decode "
                "fallback path."
            ),
            maturity="experimental",
            ptx_hotspots=True,
        ),
    }
