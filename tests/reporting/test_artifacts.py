from __future__ import annotations

from pathlib import Path

from fast_kernels.reporting import load_result_bundle
from fast_kernels.reporting.artifacts import write_result_bundle
from fast_kernels.schemas import BenchmarkCase, ResultBundle, RunMetadata


def test_result_bundle_round_trip(tmp_path: Path) -> None:
    bundle = ResultBundle(
        metadata=RunMetadata(
            run_id="test-run",
            suite_id="demo",
            created_at="2026-04-14T00:00:00+00:00",
            python_version="3.13.0",
            platform="linux",
            machine="x86_64",
            native={"available": True},
            notes=["demo"],
        ),
        cases=[
            BenchmarkCase(
                case_id="kernel-demo-fp16-row_major-shape0",
                subject_kind="kernel",
                subject_id="template/noop_gemm",
                dtype="fp16",
                layout="row_major",
                shape_name="shape0",
                dimensions={"m": 1, "n": 2, "k": 3, "batch": 1},
                status="not_implemented",
                reason="scaffold",
            )
        ],
    )

    write_result_bundle(tmp_path, bundle)
    loaded = load_result_bundle(tmp_path)
    assert loaded.schema_version == 2
    assert loaded.metadata.schema_version == 2
    assert loaded.metadata.run_id == "test-run"
    assert loaded.cases[0].case_id == "kernel-demo-fp16-row_major-shape0"
