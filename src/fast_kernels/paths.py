from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    for candidate in package_root().parents:
        if (candidate / ".git").exists() or (candidate / "benchmarks").exists():
            return candidate
    return Path.cwd()


def default_results_root() -> Path:
    return repo_root() / "results"
