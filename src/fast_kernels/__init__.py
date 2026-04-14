from importlib.metadata import PackageNotFoundError, version

from fast_kernels.env import collect_environment
from fast_kernels.native import native_available, native_build_info

try:
    __version__ = version("fast-kernels")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"

__all__ = ["__version__", "collect_environment", "native_available", "native_build_info"]
