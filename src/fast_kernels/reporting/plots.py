from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

from fast_kernels.schemas import ResultBundle

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def render_plot(bundle: ResultBundle, destination: str | Path) -> Path:
    output_path = Path(destination)
    status_counts = Counter(case.status for case in bundle.cases)
    labels = list(status_counts.keys())
    values = [status_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(labels)])
    ax.set_title(f"{bundle.metadata.suite_id} status counts")
    ax.set_ylabel("cases")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
