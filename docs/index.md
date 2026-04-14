# fast-kernels

This repo is organized like a performance lab instead of a notebook dump. The goal is to make new kernels easy to add while keeping benchmark evidence, environment metadata, and result interpretation close to the code.

## Principles

- benchmark-first, not kernel-first;
- framework and vendor baselines side by side;
- native CUDA C++ as the default authoring surface;
- PTX used only where it earns its keep;
- text artifacts committed with the same discipline as code.

## First commands

```bash
uv sync
uv run fk env
uv run fk verify benchmarks/suites/template_gemm.toml
uv run fk bench benchmarks/suites/template_gemm.toml
```

The first suite is intentionally scaffold-only. Its job is to exercise the repo flow and keep the benchmark/reporting path honest before any performance claims land.

