"""Regression tests for import-order circular dependency bugs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_in_fresh_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_utils_then_regressor_import_order_is_cycle_free() -> None:
    """Importing utils.misc before regressor should not deadlock on circular imports."""
    code = "\n".join(
        [
            "import importlib",
            "misc = importlib.import_module('fmrimod.utils.misc')",
            "importlib.import_module('fmrimod.regressor')",
            "misc.single_trial_regressor(0.0)",
        ]
    )
    result = _run_in_fresh_interpreter(code)
    assert result.returncode == 0, (
        "Fresh import order (utils.misc -> regressor) failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_regressor_then_utils_import_order_is_cycle_free() -> None:
    """Importing regressor before utils.misc should also remain cycle-free."""
    code = "\n".join(
        [
            "import importlib",
            "import typing",
            "importlib.import_module('fmrimod.regressor')",
            "misc = importlib.import_module('fmrimod.utils.misc')",
            "typing.get_type_hints(misc.single_trial_regressor)",
        ]
    )
    result = _run_in_fresh_interpreter(code)
    assert result.returncode == 0, (
        "Fresh import order (regressor -> utils.misc) failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
