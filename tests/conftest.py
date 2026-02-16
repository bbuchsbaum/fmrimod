"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_loky_max_cpu_count(monkeypatch):
    """Avoid environment-dependent loky physical-core warnings in tests."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(1)
    yield
    # Reset after test
    np.random.seed(None)


@pytest.fixture
def time_grid():
    """Standard time grid for testing."""
    return np.arange(0, 30, 0.5)


@pytest.fixture
def sampling_frame_simple():
    """Simple sampling frame for testing."""
    from fmrimod.sampling import SamplingFrame
    return SamplingFrame(blocklens=100, tr=2.0)


@pytest.fixture
def sampling_frame_multi_block():
    """Multi-block sampling frame for testing."""
    from fmrimod.sampling import SamplingFrame
    return SamplingFrame(
        blocklens=[50, 75, 100],
        tr=[2.0, 2.0, 1.5],
        start_time=[0.0, 110.0, 280.0]
    )
