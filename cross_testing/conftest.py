"""Pytest configuration for cross-testing."""

import pytest
import numpy as np
import warnings


@pytest.fixture
def numerical_tolerance():
    """Define acceptable numerical tolerances."""
    return {
        'rtol': 1e-10,      # Relative tolerance
        'atol': 1e-12,      # Absolute tolerance
        'matrix_rtol': 1e-8,  # For larger matrices
        'sparse_rtol': 1e-6,  # For sparse operations
    }


@pytest.fixture(scope='session')
def r_session():
    """Create persistent R session."""
    try:
        import rpy2.robjects as ro
        ro.r('library(fmrihrf)')
        return ro.r
    except ImportError:
        pytest.skip("rpy2 not available - skipping R cross-tests")
    except Exception as e:
        pytest.skip(f"R fmrihrf package not available: {e}")


@pytest.fixture(autouse=True)
def check_rpy2():
    """Check if rpy2 is available and skip test if not."""
    try:
        import rpy2
    except ImportError:
        pytest.skip("rpy2 not available - skipping cross-test")


@pytest.fixture
def r_tester():
    """Create REquivalenceTester instance."""
    from .utils import REquivalenceTester
    try:
        return REquivalenceTester()
    except RuntimeError as e:
        pytest.skip(str(e))