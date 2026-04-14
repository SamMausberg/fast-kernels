# fast-kernels

CUDA kernel experiments, with a benchmark-first layout from the start.

The default path here is hand-written CUDA C++. Python is used for suite configs, environment capture, reporting, and CLI tooling.

## Layout

- `csrc/`: native code, headers, launchers, and kernel-family directories
- `src/fast_kernels/`: CLI, schemas, registry, environment capture, and reporting
- `benchmarks/`: suite definitions, shapes, baseline adapters, and runners
- `results/`: versioned benchmark outputs

## Quickstart

```bash
uv sync
uv run fk env
uv run fk verify benchmarks/suites/template_gemm.toml
uv run fk bench benchmarks/suites/template_gemm.toml
```

That benchmark command writes a run under `results/template_gemm/<run-id>/`.

## CUDA build

Editable installs default to `FK_ENABLE_CUDA=ON` and build for the active GPU architecture.

Default CUDA setup:

```bash
uv sync --extra benchmark
uv run fk env
```

CPU-only opt-out:

```bash
CMAKE_ARGS=-DFK_ENABLE_CUDA=OFF uv sync
```

For direct CMake builds:

```bash
cmake --preset dev
cmake --build --preset dev
```
