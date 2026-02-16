"""Import regression tests for lowrank modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_lowrank_engine_in_fresh_interpreter() -> None:
    """Lowrank engine should import without circular GLM dependencies."""
    repo_root = Path(__file__).resolve().parents[2]
    code = "import fmrimod.lowrank.engine"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Fresh import of fmrimod.lowrank.engine failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
