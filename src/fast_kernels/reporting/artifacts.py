from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fast_kernels.reporting.markdown import render_summary_markdown
from fast_kernels.reporting.plots import render_plot
from fast_kernels.schemas import BenchmarkCase, ResultBundle


def _case_to_row(case: BenchmarkCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "subject_kind": case.subject_kind,
        "subject_id": case.subject_id,
        "dtype": case.dtype,
        "layout": case.layout,
        "shape_name": case.shape_name,
        "status": case.status,
        "latency_us_median": case.latency_us_median,
        "latency_us_p95": case.latency_us_p95,
        "wall_latency_us_median": case.wall_latency_us_median,
        "wall_latency_us_p95": case.wall_latency_us_p95,
        "device_latency_us_median": case.device_latency_us_median,
        "device_latency_us_p95": case.device_latency_us_p95,
        "throughput": case.throughput,
        "decode_tokens_per_second": case.decode_tokens_per_second,
        "context_tokens_per_second": case.context_tokens_per_second,
        "effective_kv_gib_per_second": case.effective_kv_gib_per_second,
        "reason": case.reason,
        "speedup_vs": json.dumps(case.speedup_vs, sort_keys=True),
        "metrics": json.dumps(case.metrics, sort_keys=True),
        "dimensions": json.dumps(case.dimensions, sort_keys=True),
    }
    for key in ("m", "n", "k", "batch"):
        row[key] = case.dimensions.get(key)
    return row


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
