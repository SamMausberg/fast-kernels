from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fast_kernels.native import native_build_info


@dataclass(slots=True)
class EnvironmentSnapshot:
    python_version: str
    platform: str
    machine: str
    git_sha: str | None
    git_dirty: bool
    native: dict[str, Any]
    nvidia_smi: dict[str, str] | None
    nvcc_version: str | None
    torch_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _run_capture(command: list[str], cwd: Path | None = None) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=cwd,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_sha(repo: Path | None = None) -> str | None:
    return _run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo)


def _git_dirty(repo: Path | None = None) -> bool:
    output = _run_capture(["git", "status", "--porcelain"], cwd=repo)
    return bool(output)


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return str(torch.__version__)


def _nvidia_smi() -> dict[str, str] | None:
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if output is None:
        return None
    first_line = output.splitlines()[0]
    name, _, driver = first_line.partition(",")
    return {"name": name.strip(), "driver_version": driver.strip()}


def _nvcc_version() -> str | None:
    output = _run_capture(["nvcc", "--version"])
    if output is None:
        return None
    for line in reversed(output.splitlines()):
        marker = "release "
        if marker in line:
            return line.split(marker, maxsplit=1)[1].split(",", maxsplit=1)[0].strip()
    return output.splitlines()[-1]


def collect_environment(repo: Path | None = None) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        git_sha=_git_sha(repo),
        git_dirty=_git_dirty(repo),
        native=native_build_info(),
        nvidia_smi=_nvidia_smi(),
        nvcc_version=_nvcc_version(),
        torch_version=_torch_version(),
    )
