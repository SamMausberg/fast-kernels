from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    schema_version: int = 1
    run_id: str
    suite_id: str
    created_at: str
    git_sha: str | None = None
    python_version: str
    platform: str
    machine: str
    native: dict[str, Any] = Field(default_factory=dict)
    nvidia_smi: dict[str, str] | None = None
    nvcc_version: str | None = None
    torch_version: str | None = None
    notes: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    case_id: str
    subject_kind: Literal["kernel", "baseline"]
    subject_id: str
    dtype: str
    layout: str
    shape_name: str
    dimensions: dict[str, int]
    status: Literal["ok", "skipped", "not_implemented", "failed"]
    latency_us_median: float | None = None
    latency_us_p95: float | None = None
    throughput: float | None = None
    speedup_vs: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None


class ResultBundle(BaseModel):
    schema_version: int = 1
    metadata: RunMetadata
    cases: list[BenchmarkCase] = Field(default_factory=list)
