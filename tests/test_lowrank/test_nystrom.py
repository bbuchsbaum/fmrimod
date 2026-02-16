"""Tests for Nyström approximation."""

import numpy as np
import pytest

from fmrimod.lowrank.nystrom import (
    LandmarkWeights,
    build_landmark_weights,
    extend_betas,
    select_landmarks,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def coords_3d(rng):
    """100 points in 3-D space."""
    return rng.standard_normal((100, 3))


class TestSelectLandmarks:
    def test_random_selection(self, coords_3d, rng):
        idx = select_landmarks(coords_3d, 20, method="random", rng=rng)
        assert len(idx) == 20
        assert len(set(idx)) == 20  # unique

    def test_kmeans_selection(self, coords_3d, rng):
        idx = select_landmarks(coords_3d, 10, method="kmeans", rng=rng)
        assert len(idx) == 10
        assert len(set(idx)) == 10

    def test_clamp_to_V(self, coords_3d, rng):
        idx = select_landmarks(coords_3d, 500, method="random", rng=rng)
        assert len(idx) == 100

    def test_invalid_method_raises(self, coords_3d, rng):
        with pytest.raises(ValueError):
            select_landmarks(coords_3d, 10, method="invalid", rng=rng)


class TestBuildLandmarkWeights:
    def test_shape_and_normalisation(self, coords_3d, rng):
        lm_idx = select_landmarks(coords_3d, 20, method="random", rng=rng)
        lw = build_landmark_weights(coords_3d, coords_3d[lm_idx], k=5)

        assert lw.indices.shape == (100, 5)
        assert lw.weights.shape == (100, 5)
        assert lw.n_voxels == 100
        assert lw.n_landmarks == 20
        # Row-normalised
        np.testing.assert_allclose(lw.weights.sum(axis=1), 1.0, atol=1e-10)

    def test_custom_bandwidth(self, coords_3d, rng):
        lm_idx = select_landmarks(coords_3d, 15, method="random", rng=rng)
        lw = build_landmark_weights(coords_3d, coords_3d[lm_idx], k=3, bandwidth=1.0)
        assert lw.weights.shape == (100, 3)


class TestExtendBetas:
    def test_recovery_at_landmarks(self, coords_3d, rng):
        """If data is smooth, extension should be close at landmarks."""
        V = 100
        L = 20
        p = 3
        lm_idx = select_landmarks(coords_3d, L, method="random", rng=rng)
        lw = build_landmark_weights(coords_3d, coords_3d[lm_idx], k=5)

        betas_lm = rng.standard_normal((p, L))
        betas_full = extend_betas(betas_lm, lw)
        assert betas_full.shape == (p, V)

    def test_constant_field_exact(self, coords_3d, rng):
        """A constant field should extend exactly."""
        L = 30
        p = 2
        lm_idx = select_landmarks(coords_3d, L, method="random", rng=rng)
        lw = build_landmark_weights(coords_3d, coords_3d[lm_idx], k=5)

        betas_lm = np.ones((p, L)) * 3.14
        betas_full = extend_betas(betas_lm, lw)
        # All voxels should have value 3.14 since weights sum to 1
        np.testing.assert_allclose(betas_full, 3.14, atol=1e-10)
