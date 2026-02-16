"""Regression checks for high-signal static analysis findings."""

import shutil
import subprocess

import pytest


def test_no_high_signal_ruff_violations_in_fmrimod_source():
    """Ensure selected high-signal ruff warnings remain fixed in source."""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not installed in the test environment")

    result = subprocess.run(
        [ruff, "check", "fmrimod", "--select", "F541,B904,F811"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"ruff reported high-signal issues:\n{result.stdout or result.stderr}")
