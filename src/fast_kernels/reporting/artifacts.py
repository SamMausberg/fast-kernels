from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fast_kernels.reporting.markdown import render_summary_markdown
from fast_kernels.reporting.plots import render_plot
from fast_kernels.schemas import BenchmarkCase, ResultBundle


def _case_to_row(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "subject_kind": case.subject_kind,
        "subject_id": case.subject_id,
        "dtype": case.dtype,
        "layout": case.layout,
        "shape_name": case.shape_name,
        "m": case.dimensions["m"],
        "n": case.dimensions["n"],
        "k": case.dimensions["k"],
        "batch": case.dimensions["batch"],
        "status": case.status,
        "latency_us_median": case.latency_us_median,
        "latency_us_p95": case.latency_us_p95,
        "throughput": case.throughput,
        "reason": case.reason,
        "speedup_vs": json.dumps(case.speedup_vs, sort_keys=True),
    }


def load_result_bundle(run_dir: str | Path) -> ResultBundle:
    result_path = Path(run_dir) / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return ResultBundle.model_validate(payload)


def write_result_bundle(run_dir: str | Path, bundle: ResultBundle) -> None:
    destination = Path(run_dir)
    plots_dir = destination / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = destination / "metadata.json"
    results_json_path = destination / "results.json"
    results_csv_path = destination / "results.csv"
    summary_path = destination / "summary.md"
    plot_path = plots_dir / "status_counts.svg"

    metadata_path.write_text(
        json.dumps(bundle.metadata.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    results_json_path.write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = list(_case_to_row(bundle.cases[0]).keys()) if bundle.cases else []
    with results_csv_path.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_case_to_row(case) for case in bundle.cases)

    summary_path.write_text(render_summary_markdown(bundle), encoding="utf-8")
    render_plot(bundle, plot_path)
