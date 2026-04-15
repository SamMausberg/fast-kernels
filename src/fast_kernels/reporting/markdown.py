from __future__ import annotations

from collections import Counter

from fast_kernels.schemas import ResultBundle


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def render_summary_markdown(bundle: ResultBundle) -> str:
    status_counts = Counter(case.status for case in bundle.cases)
    lines = [
        f"# {bundle.metadata.suite_id}",
        "",
        f"- Run ID: `{bundle.metadata.run_id}`",
        f"- Created: `{bundle.metadata.created_at}`",
        f"- Git SHA: `{bundle.metadata.git_sha or 'uncommitted'}`",
        f"- Python: `{bundle.metadata.python_version}`",
        f"- Platform: `{bundle.metadata.platform}`",
        f"- Native backend available: `{bundle.metadata.native.get('available', False)}`",
        "",
        "## Status Summary",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in bundle.metadata.notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            (
                "| case | subject | dtype | layout | shape | status | "
                "wall us | device us | p95 us | decode tok/s | "
                "ctx tok/s | KV GiB/s | speedup | metrics |"
            ),
            (
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | "
                "---: | ---: | ---: | ---: | --- | --- |"
            ),
        ]
    )
    for case in bundle.cases:
        speedup = ", ".join(
            f"{baseline}={value:.2f}x" for baseline, value in sorted(case.speedup_vs.items())
        )
        metrics = ", ".join(f"{name}={value:.2f}" for name, value in sorted(case.metrics.items()))
        lines.append(
            "| "
            f"`{case.case_id}` | `{case.subject_id}` | `{case.dtype}` | "
            f"`{case.layout}` | `{case.shape_name}` | `{case.status}` | "
            f"{_format_metric(case.wall_latency_us_median or case.latency_us_median)} | "
            f"{_format_metric(case.device_latency_us_median)} | "
            f"{_format_metric(case.wall_latency_us_p95 or case.latency_us_p95)} | "
            f"{_format_metric(case.decode_tokens_per_second or case.throughput)} | "
            f"{_format_metric(case.context_tokens_per_second)} | "
            f"{_format_metric(case.effective_kv_gib_per_second)} | "
            f"{speedup} | "
            f"{metrics} |"
        )
    lines.append("")
    return "\n".join(lines)
