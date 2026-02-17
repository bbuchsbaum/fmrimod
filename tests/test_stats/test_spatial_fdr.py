"""Tests for spatially-aware FDR correction."""

import numpy as np
import pytest

from fmrimod.stats.spatial_fdr import (
    SpatialFdrResult,
    create_3d_blocks,
    create_group_neighbors,
    estimate_pi0,
    estimate_pi0_grouped,
    smooth_pi0,
    spatial_fdr,
    weighted_bh,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestEstimatePi0:
    def test_all_null(self, rng):
        """All null p-values should give pi0 near 1."""
        p_vals = rng.uniform(0, 1, 1000)
        pi0 = estimate_pi0(p_vals)
        assert 0.85 < pi0 <= 1.0

    def test_all_signal(self):
        """All very small p-values should give pi0 near min_pi0."""
        p_vals = np.full(1000, 1e-10)
        pi0 = estimate_pi0(p_vals, min_pi0=0.05)
        assert pi0 == pytest.approx(0.05)

    def test_empty(self):
        pi0 = estimate_pi0(np.array([]))
        assert pi0 == 1.0


class TestCreate3dBlocks:
    def test_basic(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        group_ids, n_groups = create_3d_blocks(mask, block_size=3)
        V = mask.sum()
        assert len(group_ids) == V
        assert n_groups > 1
        assert group_ids.min() == 0
        assert group_ids.max() == n_groups - 1

    def test_single_voxel(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        group_ids, n_groups = create_3d_blocks(mask, block_size=3)
        assert len(group_ids) == 1
        assert n_groups == 1


class TestGroupNeighbors:
    def test_adjacency(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[1:9, 1:9, 1:9] = True
        group_ids, n_groups = create_3d_blocks(mask, block_size=4)
        neighbors = create_group_neighbors(mask, group_ids, n_groups, block_size=4)
        assert len(neighbors) == n_groups
        # Groups should have at least one neighbour (except corner blocks)
        has_nbrs = sum(1 for n in neighbors if n)
        assert has_nbrs > 0


class TestSmoothPi0:
    def test_no_smoothing(self):
        pi0 = np.array([0.5, 0.8, 0.6])
        result = smooth_pi0(pi0, [[], [], []], lam=0.0)
        np.testing.assert_array_equal(result, pi0)

    def test_smoothing_averages(self):
        pi0 = np.array([0.2, 0.8])
        neighbors = [[1], [0]]
        result = smooth_pi0(pi0, neighbors, lam=1.0)
        # Full smoothing: each gets neighbour's value
        np.testing.assert_allclose(result, [0.8, 0.2])


class TestWeightedBH:
    def test_all_null(self, rng):
        """Under null, should reject few."""
        p = rng.uniform(0, 1, 500)
        w = np.ones(500)
        reject, q, _ = weighted_bh(p, w, alpha=0.05)
        assert np.mean(reject) < 0.1

    def test_strong_signal(self):
        """Very small p-values with high weight should be rejected."""
        p = np.array([1e-10, 1e-8, 0.5, 0.9])
        w = np.array([2.0, 2.0, 0.5, 0.5])
        reject, q, _ = weighted_bh(p, w, alpha=0.05)
        assert reject[0] and reject[1]
        assert not reject[3]

    def test_empty(self):
        reject, q, threshold = weighted_bh(
            np.array([]), np.array([]), alpha=0.05,
        )
        assert len(reject) == 0


class TestSpatialFdr:
    def test_with_mask(self, rng):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        V = mask.sum()
        p_vals = rng.uniform(0, 1, V)
        # Inject some signal
        p_vals[:20] = 1e-6

        result = spatial_fdr(p_vals, mask=mask, alpha=0.05, block_size=3)
        assert isinstance(result, SpatialFdrResult)
        assert len(result.reject) == V
        assert len(result.qvalues) == V
        # Should reject the strong signals
        assert np.sum(result.reject[:20]) > 10

    def test_with_group_ids(self, rng):
        V = 100
        p_vals = rng.uniform(0, 1, V)
        group_ids = np.repeat(np.arange(10), 10).astype(np.intp)

        result = spatial_fdr(p_vals, group_ids=group_ids, alpha=0.05)
        assert len(result.pi0_raw) == 10

    def test_single_group_fallback(self, rng):
        """No mask, no groups → standard BH."""
        V = 100
        p_vals = rng.uniform(0, 1, V)
        result = spatial_fdr(p_vals, alpha=0.05)
        assert result.pi0_raw.shape == (1,)

    def test_null_fdr_control(self, rng):
        """Under pure null, FDR should be controlled."""
        V = 500
        p_vals = rng.uniform(0, 1, V)
        group_ids = np.repeat(np.arange(50), 10).astype(np.intp)
        result = spatial_fdr(p_vals, group_ids=group_ids, alpha=0.05)
        # Should reject < 10% under null
        assert np.mean(result.reject) < 0.15

    def test_group_ids_length_mismatch_raises_clear_error(self):
        """Group assignments must align exactly with p-value length."""
        p_vals = np.array([0.01, 0.2, 0.4, 0.8], dtype=np.float64)
        group_ids = np.array([0, 1], dtype=np.intp)

        with pytest.raises(ValueError, match="group_ids length .* p_values length"):
            spatial_fdr(p_vals, group_ids=group_ids, alpha=0.05)

    def test_arbitrary_group_labels_match_dense_remap(self):
        """Arbitrary integer labels should be remapped densely (fmrireg parity)."""
        p_vals = np.array([0.001, 0.2, 0.3, 0.8, 0.05], dtype=np.float64)
        labels = np.array([-10, 5, 5, -10, 42], dtype=np.intp)
        dense = np.array([0, 1, 1, 0, 2], dtype=np.intp)

        result_labels = spatial_fdr(p_vals, group_ids=labels, alpha=0.05)
        result_dense = spatial_fdr(p_vals, group_ids=dense, alpha=0.05)

        np.testing.assert_array_equal(result_labels.reject, result_dense.reject)
        np.testing.assert_allclose(result_labels.qvalues, result_dense.qvalues, atol=1e-12)
        np.testing.assert_allclose(result_labels.weights, result_dense.weights, atol=1e-12)
