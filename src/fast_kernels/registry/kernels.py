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
        "template/noop_gemm": KernelSpec(
            kernel_id="template/noop_gemm",
            family="gemm",
            description=(
                "Scaffold-only GEMM entry point for wiring up launchers and suite definitions."
            ),
            maturity="scaffold",
            ptx_hotspots=False,
        ),
    }
