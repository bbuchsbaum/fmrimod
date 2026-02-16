"""Regression tests for HRF derivative module imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_derivatives_has_no_scipy_misc_deprecation() -> None:
    """Importing derivatives module should not rely on ``scipy.misc``."""
    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import warnings; "
        "warnings.filterwarnings("
        "'error', "
        "message='scipy.misc is deprecated and will be removed in 2.0.0', "
        "category=DeprecationWarning"
        "); "
        "import fmrimod.hrf.derivatives"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Importing fmrimod.hrf.derivatives raised a scipy.misc deprecation "
        f"warning or failed unexpectedly.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_derivative_matches_finite_difference() -> None:
    """Local numeric derivative should match analytic expectations."""
    from fmrimod.hrf.derivatives import derivative

    value = derivative(lambda x: x**3, 2.0, dx=1e-5, n=1)
    assert abs(value - 12.0) < 1e-6

    second = derivative(lambda x: x**4, 2.0, dx=1e-5, n=2)
    assert abs(second - 48.0) < 1e-4


def test_derivative_signature_keeps_scipy_compat() -> None:
    """Keep SciPy-compatibility kwargs by accepting and ignoring ``order``."""
    from fmrimod.hrf.derivatives import derivative

    # scipy.misc.derivative accepts `order`, which we preserve for compatibility
    value = derivative(lambda x: x**2, 1.5, dx=1e-6, n=1, order=5)
    assert abs(value - 3.0) < 1e-5
