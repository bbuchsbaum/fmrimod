"""Tests for per-voxel HRF estimation and HRF-aware LSS."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.voxel_hrf import estimate_voxel_hrf, lss_with_voxel_hrf
from fmrimod.single._types import VoxelHrfResult, SingleTrialResult


@pytest.fixture
def rng():
    return np.random.default_rng(55)


@pytest.fixture
def multi_basis_data(rng):
    """Synthetic multi-basis data for voxel-HRF tests."""
    T, N, K, V = 100, 10, 3, 20
    basis = rng.standard_normal((30, K))
    # Interleaved format: [t1_b1, t1_b2, t1_b3, t2_b1, ...]
    X_trials = rng.standard_normal((T, N * K))
    Y = rng.standard_normal((T, V))
    return Y, X_trials, basis, T, N, K, V


class TestEstimateVoxelHrf:
    def test_basic(self, multi_basis_data):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        assert isinstance(result, VoxelHrfResult)
        assert result.coefficients.shape == (K, V)

    def test_unit_norm(self, multi_basis_data):
        """Coefficients should be normalised to unit L2 norm per voxel."""
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        norms = np.linalg.norm(result.coefficients, axis=0)
        assert_allclose(norms, 1.0, atol=1e-10)

    def test_infer_K_from_basis(self, multi_basis_data):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis)
        assert result.coefficients.shape[0] == K

    def test_with_confounds(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        confounds = rng.standard_normal((T, 3))
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K, confounds=confounds)
        assert result.coefficients.shape == (K, V)

    def test_bad_ncol_raises(self, rng):
        T, V, K = 50, 10, 3
        Y = rng.standard_normal((T, V))
        X_trials = rng.standard_normal((T, 11))  # 11 not divisible by 3
        basis = rng.standard_normal((20, K))
        with pytest.raises(ValueError, match="not divisible"):
            estimate_voxel_hrf(Y, X_trials, basis, K=K)


class TestLssWithVoxelHrf:
    def test_basic(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        result = lss_with_voxel_hrf(Y, X_trials, hrf_result)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "lss_voxel_hrf"
        assert result.betas.shape == (N, V)

    def test_with_confounds(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        confounds = rng.standard_normal((T, 2))
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K, confounds=confounds)
        result = lss_with_voxel_hrf(Y, X_trials, hrf_result, confounds=confounds)
        assert result.betas.shape == (N, V)

    def test_voxel_mismatch_raises(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        # Wrong number of voxels
        Y_wrong = rng.standard_normal((T, V + 5))
        with pytest.raises(ValueError, match="voxels"):
            lss_with_voxel_hrf(Y_wrong, X_trials, hrf_result)

    def test_small_chunk(self, multi_basis_data, rng):
        """Test chunked processing with small chunk_size."""
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        result_small = lss_with_voxel_hrf(Y, X_trials, hrf_result, chunk_size=5)
        result_big = lss_with_voxel_hrf(Y, X_trials, hrf_result, chunk_size=1000)
        assert_allclose(result_small.betas, result_big.betas, atol=1e-10)
