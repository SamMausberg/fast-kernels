# Development

## Standard workflow

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
uv run mkdocs build --strict
```

## CUDA-enabled workflow

The editable install defaults to `FK_ENABLE_CUDA=OFF` so the repo works on CPU-only CI and developer machines.

Enable CUDA explicitly when you want the native extension to compile CUDA sources:

```bash
CMAKE_ARGS=-DFK_ENABLE_CUDA=ON uv sync --extra benchmark
```

For direct CMake iteration:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```

## Results workflow

Run suites through the CLI so output lands in the expected layout:

```bash
uv run fk verify benchmarks/suites/template_gemm.toml
uv run fk bench benchmarks/suites/template_gemm.toml
```

