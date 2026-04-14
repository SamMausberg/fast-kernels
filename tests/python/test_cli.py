from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from fast_kernels.cli.main import main


def test_env_command_json(capsys: CaptureFixture[str]) -> None:
    assert main(["env", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "python_version" in payload
    assert "native" in payload


def test_bench_command_writes_artifacts(tmp_path: Path) -> None:
    suite_path = Path("benchmarks/suites/template_gemm.toml")
    assert main(["bench", str(suite_path), "--output-root", str(tmp_path)]) == 0
    suite_root = tmp_path / "template_gemm"
    run_dirs = list(suite_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "plots" / "status_counts.svg").exists()
