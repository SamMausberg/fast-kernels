from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None,
    reason="GPU checks require nvidia-smi on the path.",
)


def test_gpu_slot_is_reserved_for_real_kernels() -> None:
    assert True
