from __future__ import annotations

from collections import Counter

from fast_kernels.schemas import ResultBundle


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
            "| case | subject | dtype | layout | shape | status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in bundle.cases:
        lines.append(
            "| "
            f"`{case.case_id}` | `{case.subject_id}` | `{case.dtype}` | "
            f"`{case.layout}` | `{case.shape_name}` | `{case.status}` |"
        )
    lines.append("")
    return "\n".join(lines)
