"""Tests for nuisance projection utility."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single._project import project_nuisance


@pytest.fixture
def rng():
    return np.random.default_rng(111)


class TestProjectNuisance:
    def test_single_target(self, rng):
        n = 50
        X_nuis = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, 10))
        Y_proj = project_nuisance(X_nuis, Y)
        # Projected Y should be orthogonal to X_nuis
        assert_allclose(X_nuis.T @ Y_proj, 0.0, atol=1e-10)

    def test_multiple_targets(self, rng):
        n = 50
        X_nuis = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, 10))
        Z = rng.standard_normal((n, 5))
        Y_proj, Z_proj = project_nuisance(X_nuis, Y, Z)
        assert_allclose(X_nuis.T @ Y_proj, 0.0, atol=1e-10)
        assert_allclose(X_nuis.T @ Z_proj, 0.0, atol=1e-10)

    def test_preserves_shape(self, rng):
        n = 50
        X_nuis = rng.standard_normal((n, 2))
        Y = rng.standard_normal((n, 10))
        Y_proj = project_nuisance(X_nuis, Y)
        assert Y_proj.shape == Y.shape

    def test_idempotent(self, rng):
        """Projecting twice should give same result."""
        n = 50
        X_nuis = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, 10))
        Y_proj1 = project_nuisance(X_nuis, Y)
        Y_proj2 = project_nuisance(X_nuis, Y_proj1)
        assert_allclose(Y_proj1, Y_proj2, atol=1e-10)
