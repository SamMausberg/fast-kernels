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

Editable installs default to `FK_ENABLE_CUDA=OFF` so the repo works on CPU-only CI and non-GPU machines.

To build with CUDA enabled:

```bash
CMAKE_ARGS=-DFK_ENABLE_CUDA=ON uv sync --extra benchmark
uv run fk env
```

For direct CMake builds:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```
