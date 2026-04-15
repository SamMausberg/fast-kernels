from __future__ import annotations

from pathlib import Path

from fast_kernels.benchmarking import benchmark_suite
from fast_kernels.reporting import load_result_bundle


def test_rdkng_block_suite_writes_artifacts(tmp_path: Path) -> None:
    run_dir = benchmark_suite(
        Path("benchmarks/suites/rdkng_block.toml"),
        output_root=tmp_path,
    )
    bundle = load_result_bundle(run_dir)
    assert bundle.metadata.suite_id == "rdkng_block"
    assert bundle.cases
    statuses = {case.status for case in bundle.cases}
    assert statuses <= {"ok", "skipped", "failed"}
    assert (run_dir / "summary.md").exists()


def test_rdkng_training_suite_writes_artifacts(tmp_path: Path) -> None:
    run_dir = benchmark_suite(
        Path("benchmarks/suites/rdkng_training.toml"),
        output_root=tmp_path,
    )
    bundle = load_result_bundle(run_dir)
    assert bundle.metadata.suite_id == "rdkng_training"
    assert bundle.cases
    statuses = {case.status for case in bundle.cases}
    assert statuses <= {"ok", "skipped", "failed"}
    assert (run_dir / "summary.md").exists()
