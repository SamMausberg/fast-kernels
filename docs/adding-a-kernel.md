# Adding a Kernel

## Minimum path

1. Add native code under the appropriate `csrc/kernels/<family>/` directory.
2. Wire the host launcher under `csrc/ops/`.
3. Register the kernel ID in `src/fast_kernels/registry/kernels.py`.
4. Add or extend a suite under `benchmarks/suites/`.
5. Add correctness and artifact tests.

## Expectations

- Keep kernels grouped by workload family.
- Add PTX only where the performance or control benefit is real.
- Update methodology notes when benchmark semantics change.
- Treat result artifacts as part of the review surface.

