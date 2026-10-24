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

    @pytest.mark.parametrize("lam", [1.0, 2.0])
    def test_smoothing_is_one_step_of_r_sabha(self, lam):
        # fmrigds sabha_pi0_smooth_cpp: (pi0 + lam * mean_neighbours) / (1 + lam).
        # lam = 1 and lam = 2 tell this apart from (1 - lam) * pi0 + lam * mean.
        pi0 = np.array([0.2, 0.8, 0.5])
        neighbors = [[1, 2], [0], [0, 1]]
        result = smooth_pi0(pi0, neighbors, lam=lam)
        expected = [
            (0.2 + lam * 0.65) / (1 + lam),
            (0.8 + lam * 0.2) / (1 + lam),
            (0.5 + lam * 0.5) / (1 + lam),
        ]
        np.testing.assert_allclose(result, expected, rtol=1e-12)


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

    def test_invalid_alpha_raises_clear_error(self):
        p_vals = np.array([0.01, 0.2, 0.4, 0.8], dtype=np.float64)
        for alpha in [0.0, 1.0, -0.1, 1.1]:
            with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
                spatial_fdr(p_vals, alpha=alpha)

    def test_invalid_tau_raises_clear_error(self):
        p_vals = np.array([0.01, 0.2, 0.4, 0.8], dtype=np.float64)
        for tau in [0.0, 1.0, -0.1, 1.1]:
            with pytest.raises(ValueError, match="tau must be in \\(0, 1\\)"):
                spatial_fdr(p_vals, tau=tau)

    def test_invalid_min_pi0_raises_clear_error(self):
        p_vals = np.array([0.01, 0.2, 0.4, 0.8], dtype=np.float64)
        for min_pi0 in [-0.1, 1.1]:
            with pytest.raises(ValueError, match="min_pi0 must be in \\[0, 1\\]"):
                spatial_fdr(p_vals, min_pi0=min_pi0)

    def test_negative_smooth_lambda_warns_and_matches_zero(self):
        p_vals = np.array([0.01, 0.2, 0.4, 0.8], dtype=np.float64)
        groups = np.array([0, 0, 1, 1], dtype=np.intp)
        with pytest.warns(UserWarning, match="smooth_lambda < 0"):
            neg = spatial_fdr(p_vals, group_ids=groups, smooth_lambda=-1.0)
        zero = spatial_fdr(p_vals, group_ids=groups, smooth_lambda=0.0)
        np.testing.assert_array_equal(neg.reject, zero.reject)
        np.testing.assert_allclose(neg.qvalues, zero.qvalues, atol=1e-12)


def _r_sabha_p(p, group, alpha=0.05, tau=0.5, min_pi0=0.05):
    """fmrigds spatial_fdr_sabha (p input, no neighbours), R/posthoc-fdr-sabha.R."""
    p = np.asarray(p, float)
    _, gid = np.unique(group, return_inverse=True)
    G = int(gid.max()) + 1
    fin = np.isfinite(p)
    m_g = np.bincount(gid[fin], minlength=G)
    tail = np.bincount(gid[fin & (np.nan_to_num(p, nan=0.0) > tau)], minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi0_raw = np.where(m_g > 0, np.clip(tail / ((1 - tau) * m_g), 0, 1), 1.0)
    w = np.where(fin, (1 / np.clip(pi0_raw, min_pi0, 1.0))[gid], 0.0)
    n = int(fin.sum())
    w = w * n / w[fin].sum()
    pw = p[fin] / w[fin]
    order = np.argsort(pw)
    adj = np.minimum.accumulate((pw[order] * n / np.arange(1, n + 1))[::-1])[::-1].clip(0, 1)
    q = np.full(p.shape, np.nan)
    q_fin = np.empty(n)
    q_fin[order] = adj
    q[fin] = q_fin
    below = np.nonzero(np.sort(pw) <= alpha * np.arange(1, n + 1) / n)[0]
    reject = np.zeros(p.shape, bool)
    if below.size:
        reject[fin] = pw <= alpha * (below.max() + 1) / n
    return pi0_raw, w, q, reject


class TestSpatialFdrRParity:
    """inventory v2 REG-B-03, REG-B-04, GDS-06 (fmrireg c8c6021/af12983, fmrigds sabha)."""

    def test_missing_p_values_are_left_out_of_pi0_and_bh(self):
        rng = np.random.default_rng(3)
        grp = np.repeat([0, 1, 2, 3], 250)
        p = rng.uniform(size=1000)
        p[grp == 0] = rng.uniform(0, 0.02, size=250)
        p[250:400] = np.nan  # 60% missing in a pure-null group

        res = spatial_fdr(p, group_ids=grp, smooth_lambda=0.0)
        pi0_raw, w, q, reject = _r_sabha_p(p, grp)

        np.testing.assert_allclose(res.pi0_raw, pi0_raw, rtol=1e-12)
        assert res.pi0_raw[1] > 0.8
        np.testing.assert_array_equal(np.isfinite(res.qvalues), np.isfinite(p))
        np.testing.assert_allclose(res.qvalues, q, rtol=1e-12)
        np.testing.assert_array_equal(res.reject, reject)
        assert not res.reject[np.isnan(p)].any()

    def test_weights_are_normalised_over_features(self):
        # Equal-sized groups cannot tell group-mean from feature-sum normalisation.
        rng = np.random.default_rng(3)
        grp = np.concatenate([np.zeros(900, int), np.ones(100, int)])
        p = rng.uniform(size=1000)
        p[900:] = rng.uniform(0, 0.01, size=100)
        p[:60] = rng.uniform(0, 0.002, size=60)

        res = spatial_fdr(p, group_ids=grp, smooth_lambda=0.0)
        _, w, q, reject = _r_sabha_p(p, grp)

        assert res.weights.mean() == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(res.weights, w, rtol=1e-12)
        np.testing.assert_allclose(res.qvalues, q, rtol=1e-12)
        np.testing.assert_array_equal(res.reject, reject)

    def test_min_pi0_is_applied_after_smoothing(self):
        # Two adjacent one-voxel blocks: raw pi0 0 (signal) and 1 (null).
        mask = np.zeros((2, 1, 1), dtype=bool)
        mask[:] = True
        res = spatial_fdr(
            np.array([1e-6, 0.9]), mask=mask, block_size=1, smooth_lambda=1.0, min_pi0=0.05
        )
        np.testing.assert_allclose(res.pi0_raw, [0.0, 1.0])
        np.testing.assert_allclose(res.pi0_smooth, [0.5, 0.5])
