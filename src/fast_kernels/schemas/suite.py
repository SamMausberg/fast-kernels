from __future__ import annotations

from pydantic import BaseModel, Field


class ShapeCase(BaseModel):
    name: str
    m: int
    n: int
    k: int
    batch: int = 1

    def dimensions(self) -> dict[str, int]:
        return {"m": self.m, "n": self.n, "k": self.k, "batch": self.batch}


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
