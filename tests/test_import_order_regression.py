"""Regression tests for import-order robustness."""

import itertools
import subprocess
import sys


def test_regressor_misc_import_order_is_acyclic():
    modules = [
        "fmrimod.utils.misc",
        "fmrimod.regressor",
        "fmrimod.regressor.core",
        "fmrimod.regressor.convolution",
    ]

    for order in itertools.permutations(modules, 3):
        code = "import " + "; import ".join(order)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Import order failed: {order}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
