from __future__ import annotations

from pathlib import Path

from fast_kernels.benchmarking import benchmark_suite
from fast_kernels.reporting import load_result_bundle


def test_decode_suite_writes_artifacts_when_skipped(tmp_path: Path) -> None:
    run_dir = benchmark_suite(
        Path("benchmarks/suites/decode_linear_w4a16.toml"),
        output_root=tmp_path,
    )
    bundle = load_result_bundle(run_dir)
    assert bundle.metadata.suite_id == "decode_linear_w4a16"
    assert bundle.cases
    statuses = {case.status for case in bundle.cases}
    assert statuses <= {"ok", "skipped", "failed"}
    assert (run_dir / "summary.md").exists()
