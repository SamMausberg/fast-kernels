# Contributing

This repo is optimized for performance experiments, not drive-by churn. Keep changes narrow, measurable, and easy to validate.

## Local setup

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
uv run mkdocs build --strict
```

For CUDA-backed development:

```bash
CMAKE_ARGS=-DFK_ENABLE_CUDA=ON uv sync --extra benchmark
```

## Adding work

1. Register the kernel or baseline ID in the Python registry.
2. Add the native launcher or adapter implementation.
3. Add or extend a benchmark suite under `benchmarks/suites/`.
4. Run `fk verify` before `fk bench`.
5. Commit only text artifacts under `results/` unless there is a strong reason to do otherwise.

## Artifact policy

- Keep plots as SVG.
- Keep raw results as JSON and CSV.
- Do not commit large binary profiler dumps.
- Note methodology changes in the same PR that changes the benchmark output.

