from __future__ import annotations

import tomllib
from pathlib import Path

from fast_kernels.schemas import BenchmarkSuite


def load_suite(path: str | Path) -> BenchmarkSuite:
    suite_path = Path(path)
    with suite_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return BenchmarkSuite.model_validate(payload)
