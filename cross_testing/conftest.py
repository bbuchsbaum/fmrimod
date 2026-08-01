"""Pytest configuration for cross-testing."""

import os

import pytest
import numpy as np
import warnings


# --- R-availability reporting (fmrimod#16) -------------------------------
#
# These tests are the mechanism that is supposed to catch fmrimod diverging
# from the R reference. Before this, a machine without rpy2 skipped every one
# of them silently, so a green run meant "not checked" rather than "in
# agreement" -- which is how the fmrihrf#45 defects reached fmrimod unflagged.
#
# Two changes:
#   * the terminal summary always states how many R-backed tests actually ran,
#     so "0 failures" can never be confused with "0 tests run";
#   * setting FMRIMOD_REQUIRE_R=1 turns the skip into a hard failure, for CI
#     jobs whose whole purpose is the R comparison.
#
# `--require-r` is offered too, but only binds when cross_testing/ is named
# directly on the command line (pytest only honours pytest_addoption from an
# initial conftest); the environment variable works in every invocation.

_R_STATUS = {"ran": 0, "skipped": 0, "reason": None}

REQUIRE_R_ENV = "FMRIMOD_REQUIRE_R"


def pytest_addoption(parser):
    parser.addoption(
        "--require-r",
        action="store_true",
        default=False,
        help=(
            "Fail (instead of skip) R cross-tests when rpy2 or the R fmrihrf "
            "package is unavailable. Also settable via FMRIMOD_REQUIRE_R=1."
        ),
    )


def _require_r(config) -> bool:
    if os.environ.get(REQUIRE_R_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        return bool(config.getoption("--require-r"))
    except ValueError:
        # Option not registered (cross_testing/conftest.py was not an initial
        # conftest for this invocation). The env var above still applies.
        return False


def _unavailable(config, reason: str) -> None:
    """Record and then skip-or-fail, depending on strictness."""
    _R_STATUS["skipped"] += 1
    _R_STATUS["reason"] = reason
    if _require_r(config):
        pytest.fail(
            f"{reason}\nR is required because {REQUIRE_R_ENV}=1 (or --require-r) "
            f"is set. Install the R stack with:\n"
            f"  uv pip install --python .venv/bin/python -e '.[cross-test]'\n"
            f"  R -e 'remotes::install_github(\"bbuchsbaum/fmrihrf\")'",
            pytrace=False,
        )
    pytest.skip(reason)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Always say whether the R comparison actually happened."""
    ran, skipped = _R_STATUS["ran"], _R_STATUS["skipped"]
    if ran == 0 and skipped == 0:
        return  # no R-backed tests were selected at all
    write = terminalreporter.write_line
    write("")
    if skipped and ran == 0:
        write(
            f"R PARITY: NOT CHECKED - all {skipped} R-backed test(s) skipped "
            f"({_R_STATUS['reason']}). A green run here does NOT mean fmrimod "
            f"agrees with R. Set {REQUIRE_R_ENV}=1 to make this a failure.",
            red=True,
            bold=True,
        )
    elif skipped:
        write(
            f"R PARITY: PARTIAL - {ran} R-backed test(s) ran, {skipped} skipped "
            f"({_R_STATUS['reason']}).",
            yellow=True,
            bold=True,
        )
    else:
        write(
            f"R PARITY: {ran} R-backed test(s) ran against the R reference.", green=True
        )


@pytest.fixture
def numerical_tolerance():
    """Define acceptable numerical tolerances."""
    return {
        "rtol": 1e-10,  # Relative tolerance
        "atol": 1e-12,  # Absolute tolerance
        "matrix_rtol": 1e-8,  # For larger matrices
        "sparse_rtol": 1e-6,  # For sparse operations
    }


@pytest.fixture(scope="session")
def r_session(request):
    """Create persistent R session."""
    try:
        import rpy2.robjects as ro

        ro.r("library(fmrihrf)")
        return ro.r
    except ImportError:
        _unavailable(request.config, "rpy2 not available - skipping R cross-tests")
    except Exception as e:
        _unavailable(request.config, f"R fmrihrf package not available: {e}")


@pytest.fixture(autouse=True)
def check_rpy2(request):
    """Skip only tests that explicitly require rpy2/R.

    This keeps fitlins parity tests runnable without an R stack while preserving
    existing behavior for R-Python equivalence tests. Either way the outcome is
    counted so the terminal summary can distinguish "agreed with R" from
    "never asked R" (fmrimod#16).
    """
    needs_rpy2 = (
        request.node.get_closest_marker("rpy2") is not None
        or request.node.get_closest_marker("cross_test") is not None
        or "r_tester" in request.fixturenames
        or "r_session" in request.fixturenames
    )
    if not needs_rpy2:
        return
    try:
        import rpy2  # noqa: F401
    except ImportError:
        _unavailable(request.config, "rpy2 not available - skipping cross-test")
    _R_STATUS["ran"] += 1


@pytest.fixture
def r_tester(request):
    """Create REquivalenceTester instance."""
    from .utils import REquivalenceTester

    try:
        return REquivalenceTester()
    except RuntimeError as e:
        _unavailable(request.config, str(e))
