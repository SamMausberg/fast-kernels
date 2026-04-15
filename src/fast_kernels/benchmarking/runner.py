from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fast_kernels.benchmarking.clustered_page_decode import run_clustered_page_decode_suite
from fast_kernels.benchmarking.decode_linear_w4a16 import run_decode_linear_w4a16_suite
from fast_kernels.benchmarking.prefix_union_decode import run_prefix_union_decode_suite
from fast_kernels.benchmarking.rdkng import run_rdkng_suite
from fast_kernels.benchmarking.suites import load_suite
from fast_kernels.env import collect_environment
from fast_kernels.paths import default_results_root, repo_root
from fast_kernels.registry import baseline_registry, kernel_registry
from fast_kernels.reporting.artifacts import write_result_bundle
from fast_kernels.schemas import BenchmarkCase, BenchmarkSuite, ResultBundle, RunMetadata


def _make_case_id(
    subject_kind: str,
    subject_id: str,
    dtype: str,
    layout: str,
    shape_name: str,
) -> str:
    normalized_subject = subject_id.replace("/", "__")
    return f"{subject_kind}-{normalized_subject}-{dtype}-{layout}-{shape_name}"


def verify_suite(suite: BenchmarkSuite) -> list[str]:
    errors: list[str] = []
    kernels = kernel_registry()
    baselines = baseline_registry()
    if not suite.kernels.ids:
        errors.append("suite defines no kernels")
    if not suite.baselines.ids:
        errors.append("suite defines no baselines")
    for kernel_id in suite.kernels.ids:
        if kernel_id not in kernels:
            errors.append(f"unknown kernel id: {kernel_id}")
    for baseline_id in suite.baselines.ids:
        if baseline_id not in baselines:
            errors.append(f"unknown baseline id: {baseline_id}")
    if not suite.shapes:
        errors.append("suite defines no shapes")
    if not suite.dtypes:
        errors.append("suite defines no dtypes")
    if not suite.layouts:
        errors.append("suite defines no layouts")
    return errors


def _scaffold_reason(subject_kind: str, subject_id: str) -> str:
    if subject_kind == "baseline":
        return (
            f"{subject_id} is registered but still scaffold-only. "
            "Install runtime deps and add an adapter."
        )
    return f"{subject_id} is registered but still scaffold-only. Add the launcher and timing path."


def _materialize_cases(suite: BenchmarkSuite) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    registry_groups: tuple[
        tuple[Literal["kernel", "baseline"], list[str]],
        tuple[Literal["kernel", "baseline"], list[str]],
    ] = (
        ("kernel", suite.kernels.ids),
        ("baseline", suite.baselines.ids),
    )
    for subject_kind, subject_ids in registry_groups:
        for subject_id in subject_ids:
            for dtype in suite.dtypes:
                for layout in suite.layouts:
                    for shape in suite.shapes:
                        cases.append(
                            BenchmarkCase(
                                case_id=_make_case_id(
                                    subject_kind=subject_kind,
                                    subject_id=subject_id,
                                    dtype=dtype,
                                    layout=layout,
                                    shape_name=shape.name,
                                ),
                                subject_kind=subject_kind,
                                subject_id=subject_id,
                                dtype=dtype,
                                layout=layout,
                                shape_name=shape.name,
                                dimensions=shape.dimensions(),
                                status="not_implemented",
                                reason=_scaffold_reason(subject_kind, subject_id),
                            )
                        )
    return cases


def _execute_suite(suite: BenchmarkSuite) -> tuple[list[BenchmarkCase], list[str]]:
    if suite.id == "decode_linear_w4a16":
        return run_decode_linear_w4a16_suite(suite)
    if suite.family == "clustered_page_decode":
        return run_clustered_page_decode_suite(suite)
    if suite.family == "prefix_union_decode":
        return run_prefix_union_decode_suite(suite)
    if suite.family == "rdkng":
        return run_rdkng_suite(suite)

    return (
        _materialize_cases(suite),
        [
            "Scaffold run: no real kernels or baselines are wired yet.",
            "Result artifacts are meant to validate the repo flow before performance code lands.",
        ],
    )


def benchmark_suite(
    suite_path: str | Path,
    output_root: str | Path | None = None,
) -> Path:
    suite = load_suite(suite_path)
    errors = verify_suite(suite)
    if errors:
        formatted_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Suite validation failed:\n{formatted_errors}")

    repo = repo_root()
    env = collect_environment(repo)
    created_at = datetime.now(tz=UTC)
    git_sha = env.git_sha or "worktree"
    run_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{git_sha}"
    cases, notes = _execute_suite(suite)
    metadata = RunMetadata(
        run_id=run_id,
        suite_id=suite.id,
        created_at=created_at.isoformat(),
        git_sha=env.git_sha,
        python_version=env.python_version,
        platform=env.platform,
        machine=env.machine,
        native=env.native,
        nvidia_smi=env.nvidia_smi,
        nvcc_version=env.nvcc_version,
        torch_version=env.torch_version,
        notes=notes,
    )
    bundle = ResultBundle(metadata=metadata, cases=cases)
    root = Path(output_root) if output_root is not None else default_results_root()
    run_dir = root / suite.id / run_id
    write_result_bundle(run_dir, bundle)
    return run_dir


def summarize_statuses(bundle: ResultBundle) -> dict[str, int]:
    return dict(Counter(case.status for case in bundle.cases))
