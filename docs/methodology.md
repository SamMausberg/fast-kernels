# Methodology

Every meaningful benchmark in this repo should carry enough context for another engineer to judge the claim without guessing.

## Required metadata

- git SHA
- Python version
- native build metadata
- GPU model and driver version when available
- CUDA toolkit version when available
- framework version when applicable

## Baseline policy

Each real suite should include:

- one framework baseline that is easy to reproduce;
- one vendor baseline when the workload has a credible vendor implementation;
- correctness checks before timing.

## Artifact policy

Commit small, reviewable artifacts:

- `metadata.json`
- `results.json`
- `results.csv`
- `summary.md`
- SVG plots

Avoid committing large profiler dumps or binary traces unless the PR explicitly depends on them.

