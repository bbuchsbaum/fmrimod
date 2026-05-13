from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = ROOT / "examples" / "first_trust_contrast.py"


def test_first_trust_contrast_example_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else f"{ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "### trial_type_omnibus" in result.stdout
    assert "- intent: omnibus" in result.stdout
    assert "- statistic: F" in result.stdout
    assert "FitProvenance(" in result.stdout
