from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fast_kernels.benchmarking import benchmark_suite, load_suite, verify_suite
from fast_kernels.env import collect_environment
from fast_kernels.reporting import load_result_bundle, render_plot, render_summary_markdown

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fk", description="fast-kernels development CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Print environment and build information.")
    env_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    verify_parser = subparsers.add_parser("verify", help="Validate a benchmark suite.")
    verify_parser.add_argument("suite", type=Path, help="Path to the suite TOML file.")

    bench_parser = subparsers.add_parser("bench", help="Run a benchmark suite and write result artifacts.")
    bench_parser.add_argument("suite", type=Path, help="Path to the suite TOML file.")
    bench_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the root directory used for result artifacts.",
    )

    report_parser = subparsers.add_parser("report", help="Regenerate summary markdown for a run.")
    report_parser.add_argument("run_dir", type=Path, help="Path to a run directory.")

    plot_parser = subparsers.add_parser("plot", help="Regenerate SVG plots for a run.")
    plot_parser.add_argument("run_dir", type=Path, help="Path to a run directory.")

    return parser


def _print_env_table() -> None:
    snapshot = collect_environment()
    table = Table(title="fast-kernels environment")
    table.add_column("field")
    table.add_column("value")
    table.add_row("python", snapshot.python_version)
    table.add_row("platform", snapshot.platform)
    table.add_row("machine", snapshot.machine)
    table.add_row("git_sha", snapshot.git_sha or "uncommitted")
    table.add_row("git_dirty", str(snapshot.git_dirty))
    table.add_row("torch_version", snapshot.torch_version or "not installed")
    table.add_row("nvcc", snapshot.nvcc_version or "not found")
    if snapshot.nvidia_smi is not None:
        table.add_row("gpu", snapshot.nvidia_smi["name"])
        table.add_row("driver", snapshot.nvidia_smi["driver_version"])
    table.add_row("native_available", str(snapshot.native.get("available", False)))
    console.print(table)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "env":
        snapshot = collect_environment()
        if args.json:
            print(snapshot.to_json())
        else:
            _print_env_table()
        return 0

    if args.command == "verify":
        suite = load_suite(args.suite)
        errors = verify_suite(suite)
        if errors:
            console.print("[red]Suite validation failed:[/red]")
            for error in errors:
                console.print(f"- {error}")
            return 1
        console.print(f"[green]Suite valid:[/green] {suite.id}")
        console.print(
            f"{len(suite.kernels.ids)} kernels, {len(suite.baselines.ids)} baselines, "
            f"{len(suite.shapes)} shapes."
        )
        return 0

    if args.command == "bench":
        run_dir = benchmark_suite(args.suite, args.output_root)
        console.print(f"[green]Wrote benchmark run to[/green] {run_dir}")
        return 0

    if args.command == "report":
        bundle = load_result_bundle(args.run_dir)
        summary_path = Path(args.run_dir) / "summary.md"
        summary_path.write_text(render_summary_markdown(bundle), encoding="utf-8")
        console.print(f"[green]Updated[/green] {summary_path}")
        return 0

    if args.command == "plot":
        bundle = load_result_bundle(args.run_dir)
        plot_path = Path(args.run_dir) / "plots" / "status_counts.svg"
        render_plot(bundle, plot_path)
        console.print(f"[green]Updated[/green] {plot_path}")
        return 0

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
