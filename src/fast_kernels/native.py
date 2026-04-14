from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_native: ModuleType | None
_native_import_error: ImportError | None

try:
    _native = import_module("fast_kernels._native")
except ImportError as exc:  # pragma: no cover - exercised only when build fails
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None


def native_available() -> bool:
    return _native is not None


def native_build_info() -> dict[str, Any]:
    if _native is None:
        return {
            "available": False,
            "error": str(_native_import_error),
        }
    info = dict(_native.build_info())
    info["available"] = True
    return info
