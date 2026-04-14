from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ShapeCase(BaseModel):
    name: str
    m: int | None = None
    n: int | None = None
    k: int | None = None
    batch: int = 1
    page_size: int | None = None
    max_seq_len: int | None = None
    num_q_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    gqa_group_size: int | None = None
    max_pages: int | None = None

    def dimensions(self) -> dict[str, int]:
        dims: dict[str, int] = {"batch": self.batch}
        for field_name in (
            "m",
            "n",
            "k",
            "page_size",
            "max_seq_len",
            "num_q_heads",
            "num_kv_heads",
            "head_dim",
            "gqa_group_size",
            "max_pages",
        ):
            value = getattr(self, field_name)
            if value is not None:
                dims[field_name] = int(value)
        return dims

    def require_dimension(self, name: str) -> int:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"shape {self.name!r} is missing required dimension {name!r}")
        return int(value)


class ToleranceConfig(BaseModel):
    atol: float = 1e-3
    rtol: float = 1e-3


class RegistryConfig(BaseModel):
    ids: list[str] = Field(default_factory=list)


class BenchmarkSuite(BaseModel):
    schema_version: int = 1
    id: str
    family: str
    description: str
    metric: str = "latency_us"
    tags: list[str] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    layouts: list[str] = Field(default_factory=list)
    shapes: list[ShapeCase] = Field(default_factory=list)
    kernels: RegistryConfig = Field(default_factory=RegistryConfig)
    baselines: RegistryConfig = Field(default_factory=RegistryConfig)
    tolerances: ToleranceConfig = Field(default_factory=ToleranceConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
